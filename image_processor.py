import cv2 as cv
import numpy as np
import tensorflow as tf

from mask_generator import MaskGenerator
from utils import MaskColor


class ImageProcessor:
    """
    Class for processing images.
    """

    @staticmethod
    def apply_mask_with_return(image: np.ndarray | tf.Tensor, ratio: tuple[float, float] = MaskGenerator.DEFAULT_MASK_RATIO,
                               color: MaskColor = MaskColor.WHITE) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply a mask to an image.

        Args:
            image (np.ndarray | tf.Tensor): The image to apply the mask to
            ratio (tuple[float, float], optional): Percentage interval of the image that is covered by the mask. Defaults to DEFAULT_MASK_RATIO
            color (MaskColor, optional): The color of the mask

        Returns:
            tuple[np.ndarray, np.ndarray]: The masked image and the mask
        """
        height, width, _ = image.shape

        mask, count = tf.py_function(MaskGenerator.generate_mask, inp=[(height, width), ratio], Tout=(tf.uint8, tf.int64))

        initial_mask = mask

        if isinstance(image, tf.Tensor):

            mask = tf.stack([mask, mask, mask], axis=-1)
            mask = tf.where(mask)

            if color == MaskColor.WHITE:

                update_values = tf.fill((count * 3,), MaskColor.WHITE.value)
                update_values = tf.cast(update_values, tf.float32)

            elif color == MaskColor.BLACK:

                update_values = tf.fill((count * 3,), MaskColor.BLACK.value)
                update_values = tf.cast(update_values, tf.float32)

            else:

                raise ValueError('Invalid color')

            masked = tf.tensor_scatter_nd_update(image, mask, update_values)

        else:

            masked = image.copy()

            if color == MaskColor.WHITE:

                masked[mask == MaskGenerator.MASK_VALUE] = MaskColor.WHITE.value

            elif color == MaskColor.BLACK:

                masked[mask == MaskGenerator.MASK_VALUE] = MaskColor.BLACK.value

            else:

                raise ValueError('Invalid color')

        return masked, initial_mask

    @staticmethod
    def apply_mask(image: np.ndarray | tf.Tensor, ratio: tuple[float, float] = MaskGenerator.DEFAULT_MASK_RATIO,
                   color: MaskColor = MaskColor.WHITE) -> np.ndarray:
        """
        Apply a mask to an image.

        Args:
            image (np.ndarray | tf.Tensor): The image to apply the mask to
            ratio (tuple[float, float], optional): Percentage interval of the image that is covered by the mask. Defaults to DEFAULT_MASK_RATIO
            color (MaskColor, optional): The color of the mask. Defaults to MaskColor.WHITE

        Returns:
            np.ndarray: The masked image
        """
        return ImageProcessor.apply_mask_with_return(image, ratio, color)[0]

    @staticmethod
    def inpaint_navier_stokes(image: np.ndarray | tf.Tensor, mask: np.ndarray | tf.Tensor, radius: float = 3.0) -> np.ndarray:
        """
        Inpaint an image using the Navier-Stokes algorithm.

        Args:
            image (np.ndarray | tf.Tensor): The image to inpaint
            mask (np.ndarray | tf.Tensor): The mask to use
            radius (float, optional): The inpaint radius to use. Defaults to 3.0.

        Returns:
            np.ndarray: The inpainted image
        """
        if isinstance(image, tf.Tensor):

            image = image.numpy().astype(np.uint8)

        if isinstance(mask, tf.Tensor):

            mask = mask.numpy().astype(np.uint8)

        inpainted = cv.inpaint(image, mask, radius, cv.INPAINT_NS)

        return inpainted.astype(np.float32)

    @staticmethod
    def inpaint_telea(image: np.ndarray | tf.Tensor, mask: np.ndarray | tf.Tensor, radius: float = 3.0) -> np.ndarray:
        """
        Inpaint an image using the Telea algorithm.

        Args:
            image (np.ndarray | tf.Tensor): The image to inpaint
            mask (np.ndarray | tf.Tensor): The mask to use
            radius (float, optional): The inpaint radius to use. Defaults to 3.0.

        Returns:
            np.ndarray: The inpainted image
        """
        if isinstance(image, tf.Tensor):

            image = image.numpy().astype(np.uint8)

        if isinstance(mask, tf.Tensor):

            mask = mask.numpy().astype(np.uint8)

        inpainted = cv.inpaint(image, mask, radius, cv.INPAINT_TELEA)

        return inpainted.astype(np.float32)
