from sys import argv

from resnet import ResNet
from train import train, IMAGE_SIZE
from utils import time_running_gpu


def run() -> None:
    """
    Defines the ResNet model and trains it.
    """

    resnet = ResNet(
        input_shape=IMAGE_SIZE
    )

    train(resnet, argv)


if __name__ == '__main__':
    time_running_gpu(run)
