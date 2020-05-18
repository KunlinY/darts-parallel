""" Search cell """
import os
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from config import SearchConfig
import utils
from models.search_cnn import SearchCNNController
from architect import Architect
from visualize import plot
import asyncio

import syft as sy
from syft.workers.websocket_client import WebsocketClientWorker
from syft.frameworks.torch.fl.utils import federated_avg

hook = sy.TorchHook(torch)


config = SearchConfig()

default_device = torch.device("cuda:0")

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
config.print_params(logger.info)

kwargs_websocket = {"hook": hook, "host": "0.0.0.0"}
alice = WebsocketClientWorker(id="0", port=8777, **kwargs_websocket)
bob = WebsocketClientWorker(id="1", port=8778, **kwargs_websocket)
workers = [alice, bob]

for wcw in workers:
    wcw.clear_objects_remote()
    # hook.local_worker.add_worker(wcw)

remote_train_data = ([], [])
remote_valid_data = ([], [])


async def main():
    logger.info("Logger is set - training start")

    # set default gpu device id
    torch.cuda.set_device(config.gpus[0])

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.benchmark = True

    # get data with meta info
    input_size, input_channels, n_classes, train_data = utils.get_data(
        config.dataset, config.data_path, cutout_length=0, validation=False)

    net_crit = nn.CrossEntropyLoss().to(default_device)
    model = SearchCNNController(input_channels, config.init_channels, n_classes, config.layers,
                                net_crit, device_ids=config.gpus)
    model = model.to(default_device)

    # weights optimizer
    w_optim = torch.optim.SGD(model.weights(), config.w_lr, momentum=config.w_momentum,
                              weight_decay=config.w_weight_decay)
    # alphas optimizer
    alpha_optim = torch.optim.Adam(model.alphas(), config.alpha_lr, betas=(0.5, 0.999),
                                   weight_decay=config.alpha_weight_decay)

    # split data to train/validation
    n_train = len(train_data)
    split = n_train // 2
    indices = list(range(n_train))
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
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

    for idx, (data, target) in enumerate(train_loader):
        wid = idx % len(workers)
        data = data.send(workers[wid])
        target = target.send(workers[wid])
        remote_train_data[wid].append((data, target))

    for idx, (data, target) in enumerate(valid_loader):
        wid = idx % len(workers)
        data = data.send(workers[wid])
        target = target.send(workers[wid])
        remote_valid_data[wid].append((data, target))

    print("finish sampler")

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        w_optim, config.epochs, eta_min=config.w_lr_min)
    architect = Architect(model, config.w_momentum, config.w_weight_decay)

    # training loop
    best_top1 = 0.
    for epoch in range(config.epochs):
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()[0]

        model.print_alphas(logger)

        # training
        await train(train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch)

        # validation
        cur_step = (epoch+1) * len(train_loader)
        top1 = validate(valid_loader, model, epoch, cur_step)

        # log
        # genotype
        genotype = model.genotype()
        logger.info("genotype = {}".format(genotype))

        # genotype as a image
        plot_path = os.path.join(config.plot_path, "EP{:02d}".format(epoch+1))
        caption = "Epoch {}".format(epoch+1)
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


def update(step, wid, model, alpha_optim, w_optim, architect, lr):
    device = torch.device('cuda:' + str(wid))

    for w in workers:
        print(w.id, 'has objects ', len(w._objects))

    trn_X, trn_y = remote_train_data[wid][step]
    trn_X, trn_y = trn_X.to(device, non_blocking=True), trn_y.to(device, non_blocking=True)
    val_X, val_y = remote_valid_data[wid][step]
    val_X, val_y = val_X.to(device, non_blocking=True), val_y.to(device, non_blocking=True)
    N = trn_X.size(0)

    for w in workers:
        print(w.id, 'has objects ', len(w._objects))

    print(model)
    model = model.copy().send(trn_X.location)
    print(model)
    model = model.to(device)

    for w in workers:
        print(w.id, 'has objects ', len(w._objects))

    # phase 2. architect step (alpha)
    alpha_optim.zero_grad()
    architect.unrolled_backward(model, trn_X, trn_y, val_X, val_y, lr, w_optim)
    alpha_optim.step()

    # phase 1. child network step (w)
    w_optim.zero_grad()
    logits = model(trn_X)
    loss = model.criterion(logits, trn_y)
    loss.backward()
    # gradient clipping
    nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
    w_optim.step()

    prec1, prec5 = utils.accuracy(logits, trn_y, topk=(1, 5))

    return wid, model, loss, prec1, prec5, N


async def train(train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    cur_step = epoch*len(train_loader)
    writer.add_scalar('train/lr', lr, cur_step)

    model.train()

    for step in range(len(remote_train_data[0]) - 1):
        results = await asyncio.gather(
            *[
                update(step, i, model, alpha_optim, w_optim, architect, lr)
                for i in range(len(workers))
            ]
        )

        models = {}

        for i, r in enumerate(results):
            models[r[0]] = r[1]
            losses.update(r[2].item(), r[5])
            top1.update(r[3].item(), r[5])
            top5.update(r[4].item(), r[5])

            writer.add_scalar('train/loss ' + str(i), r[2].item(), cur_step)
            writer.add_scalar('train/top1 ' + str(i), r[3].item(), cur_step)
            writer.add_scalar('train/top5 ' + str(i), r[4].item(), cur_step)

        model = federated_avg(models)

        if step % config.print_freq == 0 or step == len(train_loader)-1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch+1, config.epochs, step, len(train_loader)-1, losses=losses,
                    top1=top1, top5=top5))

        cur_step += 1

    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))


def validate(valid_loader, model, epoch, cur_step):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(default_device, non_blocking=True), y.to(default_device, non_blocking=True)
            N = X.size(0)

            logits = model(X)
            loss = model.criterion(logits, y)

            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if step % config.print_freq == 0 or step == len(valid_loader)-1:
                logger.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch+1, config.epochs, step, len(valid_loader)-1, losses=losses,
                        top1=top1, top5=top5))

    writer.add_scalar('val/loss', losses.avg, cur_step)
    writer.add_scalar('val/top1', top1.avg, cur_step)
    writer.add_scalar('val/top5', top5.avg, cur_step)

    logger.info("Valid: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))

    return top1.avg


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
