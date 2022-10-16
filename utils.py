from enum import Enum, unique

import tensorflow as tf
import numpy as np
import random


def set_global_seed(seed: int) -> None:
    """
    Set the global seed for numpy, random, and tensorflow.

    Args:
        seed (int): The seed to use
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@unique
class InpaintMethod(Enum):
    PATCH_MATCH: str = 'patch_match'
    NAVIER_STOKES: str = 'navier_stokes'
    TELEA: str = 'telea'


@unique
class MaskColor(Enum):
    """
    The color of the mask.
    """
    WHITE: int = 255
    BLACK: int = 0
