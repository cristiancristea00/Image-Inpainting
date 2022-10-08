import tensorflow as tf
from enum import Enum, unique

import numpy as np
import random

from mask_generator import MaskGenerator


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
class MaskColor(Enum):
    """
    The color of the mask.
    """
    WHITE = 255
    BLACK = 0


def apply_mask(image: np.ndarray, color: MaskColor = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a mask to an image.

    Args:
        image (np.ndarray): The image to apply the mask to
        color (MaskColor): The color of the mask

    Returns:
        tuple[np.ndarray, np.ndarray]: The masked image and the mask
    """
    height, width, _ = image.shape
    mask = MaskGenerator.generate_mask((height, width))

    masked = image.copy()

    if color == MaskColor.WHITE:
        masked[mask == MaskGenerator.MAX_MASK_VALUE] = MaskColor.WHITE.value
    elif color == MaskColor.BLACK:
        masked[mask == MaskGenerator.MAX_MASK_VALUE] = MaskColor.BLACK.value

    return masked, mask
