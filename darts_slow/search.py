""" Search cell """
import os
import torch
import torch.nn as nn
import torch.distributions.multivariate_normal as gaussian
import numpy as np
from tensorboardX import SummaryWriter
from config import SearchConfig
import utils
from models.search_cnn import SearchCNNController
from architect import Architect
from visualize import plot

from threading import Lock
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
from torch.distributed.optim import DistributedOptimizer

config = SearchConfig()
noise_add = config.noise

device = torch.device("cuda")

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
config.print_params(logger.info)

shape_gaussian = {}


def call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)


def remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), call_method, args=args, kwargs=kwargs)


input_size, input_channels, n_classes, train_data = utils.get_data(
        config.dataset, config.data_path, cutout_length=0, validation=False)
# net_crit = nn.CrossEntropyLoss().to(device)


# --------- Parameter Server --------------------
class ParameterServer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SearchCNNController(input_channels, config.init_channels, n_classes, config.layers,
                                         net_crit, device_ids=config.gpus).to(device)
        self.input_device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, inp):
        print("forwarding ps")
        inp = inp.to(self.input_device)
        out = self.model(inp)
        # This output is forwarded over RPC, which as of 1.5.0 only accepts CPU tensors.
        # Tensors must be moved in and out of GPU memory due to this.
        out = out.to("cpu")
        return out

    # Use dist autograd to retrieve gradients accumulated for this model.
    # Primarily used for verification.
    def get_dist_gradients(self, cid):
        grads = dist_autograd.get_gradients(cid)
        # This output is forwarded over RPC, which as of 1.5.0 only accepts CPU tensors.
        # Tensors must be moved in and out of GPU memory due to this.
        cpu_grads = {}
        for k, v in grads.items():
            k_cpu, v_cpu = k.to("cpu"), v.to("cpu")
            cpu_grads[k_cpu] = v_cpu
        return cpu_grads

    def genotype(self):
        return rpc.RRef(self.model.genotype())

    def weights(self):
        param_rrefs = [rpc.RRef(param) for param in self.model.parameters()]
        return param_rrefs

    def named_weights(self):
        param_rrefs = [rpc.RRef(param) for param in self.model.named_parameters()]
        return param_rrefs

    def alphas(self):
        param_rrefs = [rpc.RRef(p) for n, p in self.model._alphas]
        return param_rrefs

    def named_alphas(self):
        param_rrefs = [(rpc.RRef(n), rpc.RRef(p)) for n, p in self.model._alphas]
        return param_rrefs


param_server = None
global_lock = Lock()


def create_lr_scheduler(opt_rref):
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_rref, config.epochs, eta_min=config.w_lr_min)
    return lr_scheduler


def lrs_step(lrs_rref):
    lrs_rref.local_value().step()


def get_lrs_value(lrs_rref):
    return lrs_rref.local_value().get_lr()[0]


def get_parameter_server():
    global param_server
    # Ensure that we get only one handle to the ParameterServer.
    with global_lock:
        if not param_server:
            # construct it once
            param_server = ParameterServer()
        return param_server


def run_parameter_server(rank, world_size):
    # The parameter server just acts as a host for the model and responds to
    # requests from trainers, hence it does not need to run a loop.
    # rpc.shutdown() will wait for all workers to complete by default, which
    # in this case means that the parameter server will wait for all trainers
    # to complete, and then exit.
    print("PS master initializing RPC")
    rpc.init_rpc(name="parameter_server", rank=rank, world_size=world_size)
    print("RPC initialized! Running parameter server...")
    rpc.shutdown()
    print("RPC shutdown on parameter server.")


class TrainerNet(nn.Module):
    def __init__(self, criterion):
        super().__init__()
        self.param_server_rref = rpc.remote("parameter_server", get_parameter_server)
        self.criterion = criterion

    def forward(self, x):
        print("forwarding trainer net")
        model_output = remote_method(
            ParameterServer.forward, self.param_server_rref, x)
        return model_output

    def loss(self, X, y):
        logits = self.forward(X)
        return net_crit(logits, y)

    def genotype(self):
        remote_params = remote_method(
            ParameterServer.genotype,
            self.param_server_rref)
        return remote_params

    def weights(self):
        remote_params = remote_method(
            ParameterServer.weights,
            self.param_server_rref)
        return remote_params

    def named_weights(self):
        remote_params = remote_method(
            ParameterServer.named_weights,
            self.param_server_rref)
        return remote_params

    def alphas(self):
        remote_params = remote_method(
            ParameterServer.alphas,
            self.param_server_rref)
        return remote_params

    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p


def train(train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    cur_step = epoch * len(train_loader)
    writer.add_scalar('train/lr', lr, cur_step)

    model.train()

    for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(train_loader, valid_loader)):
        with dist_autograd.context() as cid:
            trn_X, trn_y = trn_X.to(device, non_blocking=True), trn_y.to(device, non_blocking=True)
            val_X, val_y = val_X.to(device, non_blocking=True), val_y.to(device, non_blocking=True)
            N = trn_X.size(0)

            # phase 2. architect step (alpha)
            # alpha_optim.zero_grad()
            architect.unrolled_backward(trn_X, trn_y, val_X, val_y, lr, w_optim, logger)
            alpha_optim.step()

            # phase 1. child network step (w)
            # w_optim.zero_grad()
            logits = model(trn_X)
            loss = model.criterion(logits, trn_y)
            dist_autograd.backward(cid, [loss])
            # loss.backward()
            # gradient clipping
            nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)

            if noise_add:
                for i, param in enumerate(model.parameters()):
                    noise = shape_gaussian[param.grad.shape].sample() / config.batch_size
                    noise = noise.to(param.grad.device)
                    param.grad += noise

            w_optim.step()

        prec1, prec5 = utils.accuracy(logits, trn_y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % config.print_freq == 0 or step == len(train_loader) - 1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch + 1, config.epochs, step, len(train_loader) - 1, losses=losses,
                    top1=top1, top5=top5))

        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/top1', prec1.item(), cur_step)
        writer.add_scalar('train/top5', prec5.item(), cur_step)
        cur_step += 1

    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, config.epochs, top1.avg))


def validate(valid_loader, model, epoch, cur_step):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)

            logits = model(X)
            loss = model.criterion(logits, y)

            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if step % config.print_freq == 0 or step == len(valid_loader) - 1:
                logger.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch + 1, config.epochs, step, len(valid_loader) - 1, losses=losses,
                        top1=top1, top5=top5))

    writer.add_scalar('val/loss', losses.avg, cur_step)
    writer.add_scalar('val/top1', top1.avg, cur_step)
    writer.add_scalar('val/top5', top5.avg, cur_step)

    logger.info("Valid: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, config.epochs, top1.avg))

    return top1.avg


if __name__ == "__main__":
    logger.info("Logger is set - training start")

    # set default gpu device id
    torch.cuda.set_device(config.gpus[0])

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.benchmark = True

    # get data with meta info
    # input_size, input_channels, n_classes, train_data = utils.get_data(
    #     config.dataset, config.data_path, cutout_length=0, validation=False)
    #
    net_crit = nn.CrossEntropyLoss().to(device)
    # model = SearchCNNController(input_channels, config.init_channels, n_classes, config.layers,
    #                             net_crit, device_ids=config.gpus)
    # model = model.to(device)
    model = TrainerNet(net_crit)

    # weights optimizer
    # w_optim = torch.optim.SGD(model.weights(), config.w_lr, momentum=config.w_momentum,
    #                           weight_decay=config.w_weight_decay)
    w_optim = DistributedOptimizer(torch.optim.SGD, model.weights(), lr=config.w_lr,
                                   momentum=config.w_momentum, weight_decay=config.w_weight_decay)
    # alphas optimizer
    # alpha_optim = torch.optim.Adam(model.alphas(), config.alpha_lr, betas=(0.5, 0.999),
    #                                weight_decay=config.alpha_weight_decay)
    alpha_optim = DistributedOptimizer(torch.optim.Adam, model.alphas(), lr=config.alpha_lr,
                                       betas=(0.5, 0.999), weight_decay=config.alpha_weight_decay)

    # split data to train/validation
    n_train = len(train_data)
    split = n_train // 2
    world = config.world_size
    rank = config.rank
    indices = list(range(n_train))
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(
        indices[int(rank*split/world):int((rank+1)*split/world)])
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(
        indices[split+int(rank*(n_train-split)/world):split+int(int((rank+1)*(n_train-split)/world))])
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               sampler=train_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               sampler=valid_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)

    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     w_optim, config.epochs, eta_min=config.w_lr_min)
    lrs_rrefs = []
    for opt_rref in w_optim.remote_optimizers:
        lrs_rrefs = rpc.remote(opt_rref.owner(), create_lr_scheduler, args=(opt_rref,))

    architect = Architect(model, config.w_momentum, config.w_weight_decay, noise_add)

    if noise_add:
        logger.info("Adding noise")
        for param in model.parameters():
            shape_gaussian[param.data.shape] = gaussian.MultivariateNormal(
                torch.zeros(param.data.shape), torch.eye(param.data.shape[-1]))
    else:
        logger.info("Not adding noise")

    # training loop
    best_top1 = 0.
    for epoch in range(config.epochs):

        with dist_autograd.context() as cid:
            futs = []
            for lrs_rref in lrs_rrefs:
                futs.append(rpc.rpc_async(lrs_rref.owner(), lrs_step, args=(lrs_rref,)))
            [fut.wait() for fut in futs]
            lr = remote_method(get_lrs_value, lrs_rrefs.owner(), args=(lrs_rrefs[0],))
        # lr_scheduler.step()
        # lr = lr_scheduler.get_lr()[0]

        # model.print_alphas(logger)

        # training
        train(train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch)

        # validation
        cur_step = (epoch + 1) * len(train_loader)
        top1 = validate(valid_loader, model, epoch, cur_step)

        # log
        # genotype
        genotype = model.genotype()
        logger.info("genotype = {}".format(genotype))

        # genotype as a image
        plot_path = os.path.join(config.plot_path, "EP{:02d}".format(epoch + 1))
        caption = "Epoch {}".format(epoch + 1)
        plot(genotype.normal, plot_path + "-normal", caption)
        plot(genotype.reduce, plot_path + "-reduce", caption)

        # save
        if best_top1 < top1:
            best_top1 = top1
            best_genotype = genotype
            is_best = True
        else:
            is_best = False
        utils.save_checkpoint(model, config.path, is_best)
        print("")

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    logger.info("Best Genotype = {}".format(best_genotype))
