from sys import argv

from train import train, IMAGE_SIZE
from unet import UNet
from utils import time_running_gpu


def run() -> None:
    """
    Defines the UNet model and trains it.
    """

    unet = UNet(
        input_shape=IMAGE_SIZE,
        filters=(16, 32, 64, 64, 64, 128),
        kernels=(7, 7, 5, 5, 3, 3),
        skip_filters=(4,) * 6,
        skip_kernels=(1,) * 6,
        is_gated=False
    )

    train(unet, argv)


if __name__ == '__main__':
    time_running_gpu(run)
