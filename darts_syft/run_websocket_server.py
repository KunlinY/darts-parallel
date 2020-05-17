from multiprocessing import Process
import argparse
import os
import logging
import syft as sy
from syft.workers.websocket_server import WebsocketServerWorker
import torch
import numpy as np
from torchvision import datasets
from torchvision import transforms
from syft.frameworks.torch.fl import utils

KEEP_LABELS_DICT = {
    "alice": [0, 1, 2, 3],
    "bob": [4, 5, 6],
    "charlie": [7, 8, 9],
    "testing": list(range(10)),
    None: list(range(10)),
}


def start_websocket_server_worker(
    id, host, port, hook, verbose
):
    """Helper function for spinning up a websocket server and setting up the local datasets."""

    server = WebsocketServerWorker(id=id, host=host, port=port, hook=hook, verbose=verbose)

    # # Setup toy data (mnist example)
    # mnist_dataset = datasets.MNIST(
    #     root="./data",
    #     train=True,
    #     download=True,
    #     transform=transforms.Compose(
    #         [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    #     ),
    # )
    #
    # indices = np.isin(mnist_dataset.targets, keep_labels).astype("uint8")
    # logger.info("number of true indices: %s", indices.sum())
    # selected_data = (
    #     torch.native_masked_select(mnist_dataset.data.transpose(0, 2), torch.tensor(indices))
    #         .view(28, 28, -1)
    #         .transpose(2, 0)
    # )
    # logger.info("after selection: %s", selected_data.shape)
    # selected_targets = torch.native_masked_select(mnist_dataset.targets, torch.tensor(indices))
    #
    # dataset = sy.BaseDataset(
    #     data=selected_data, targets=selected_targets, transform=mnist_dataset.transform
    # )
    # key = "mnist"
    #
    # # Adding Dataset
    # server.add_dataset(dataset, key=key)
    #
    # logger.info("datasets: %s", server.datasets)

    server.start()
    return server


if __name__ == "__main__":

    # Logging setup
    FORMAT = "%(asctime)s %(levelname)s %(filename)s(l:%(lineno)d, p:%(process)d) - %(message)s"
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger("run_websocket_server")
    logger.setLevel(level=logging.DEBUG)

    # Parse args
    parser = argparse.ArgumentParser(description="Run websocket server worker.")
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        help="port number of the websocket server worker, e.g. --port 8777",
    )
    parser.add_argument("--host", type=str, default="localhost", help="host for the connection")
    parser.add_argument(
        "--id", type=str, help="name (id) of the websocket server worker, e.g. --id alice"
    )
    parser.add_argument(
        "--testing",
        action="store_true",
        help="if set, websocket server worker will load the test dataset instead of the training dataset",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="""if set, websocket server worker will be started in verbose mode""",
    )
    parser.add_argument(
        "--notebook",
        type=str,
        default="normal",
        help="""can run websocket server for websockets examples of mnist/mnist-parallel or
    pen_testing/steal_data_over_sockets. Type 'mnist' for starting server
    for websockets-example-MNIST, `mnist-parallel` for websockets-example-MNIST-parallel
    and 'steal_data' for pen_tesing stealing data over sockets""",
    )
    args = parser.parse_args()

    # Hook and start server
    hook = sy.TorchHook(torch)
    server = WebsocketServerWorker(id=args.id, host=args.host, port=args.port, hook=args.hook, verbose=args.verbose)
    server.start()

    # server = start_websocket_server_worker(
    #         id=args.id,
    #         host=args.host,
    #         port=args.port,
    #         hook=hook,
    #         verbose=args.verbose
    #     )
