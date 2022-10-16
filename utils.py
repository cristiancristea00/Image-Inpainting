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
