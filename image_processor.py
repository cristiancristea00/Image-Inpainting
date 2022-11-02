from mask_generator import MaskGenerator
from utils import MaskColor
import tensorflow as tf
import numpy as np
import cv2 as cv


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
            image (np.ndarray): The image to apply the mask to
            ratio (tuple[float, float]): Percentage interval of the image that is covered by the mask. Defaults to DEFAULT_MASK_RATIO
            color (MaskColor): The color of the mask

        Returns:
            tuple[np.ndarray, np.ndarray]: The masked image and the mask
        """
        height, width, _ = image.shape
        mask = MaskGenerator.generate_mask(mask_size=(height, width), ratio=ratio)

        if type(image) == tf.Tensor or True:

            mask = tf.stack([mask, mask, mask], axis=-1)
            mask = tf.where(mask)

            if color == MaskColor.WHITE:

                update_values = tf.constant(MaskColor.WHITE.value, dtype=tf.float32, shape=(mask.shape[0]))

            elif color == MaskColor.BLACK:

                update_values = tf.constant(MaskColor.BLACK.value, dtype=tf.float32, shape=(mask.shape[0]))

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

        return masked, mask

    @staticmethod
    def apply_mask(image: np.ndarray | tf.Tensor, ratio: tuple[float, float] = MaskGenerator.DEFAULT_MASK_RATIO,
                   color: MaskColor = MaskColor.WHITE) -> np.ndarray:
        return ImageProcessor.apply_mask_with_return(image, ratio, color)[0]

    @staticmethod
    def inpaint_navier_stokes(image: np.ndarray, mask: np.ndarray, radius: float = 3.0) -> np.ndarray:
        """
        Inpaint an image using the Navier-Stokes algorithm.

        Args:
            image (np.ndarray): The image to inpaint
            mask (np.ndarray): The mask to use
            radius (float, optional): The inpaint radius to use. Defaults to 3.0.

        Returns:
            np.ndarray: The inpainted image
        """
        return cv.inpaint(image, mask, radius, cv.INPAINT_NS)

    @staticmethod
    def inpaint_telea(image: np.ndarray, mask: np.ndarray, radius: float = 3.0) -> np.ndarray:
        """
        Inpaint an image using the Telea algorithm.

        Args:
            image (np.ndarray): The image to inpaint
            mask (np.ndarray): The mask to use
            radius (float, optional): The inpaint radius to use. Defaults to 3.0.

        Returns:
            np.ndarray: The inpainted image
        """
        return cv.inpaint(image, mask, radius, cv.INPAINT_TELEA)
