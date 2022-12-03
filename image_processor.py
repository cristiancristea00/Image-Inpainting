import cv2 as cv
import numpy as np
import tensorflow as tf

from mask_generator import MaskGenerator
from utils import MaskColor


class ImageProcessor:
    """
    Class for processing images.
    """

    def __init__(self, mask_generator: MaskGenerator, batch_size: int) -> None:
        """
        Initialize the ImageProcessor.

        Args:
            mask_generator (MaskGenerator): The mask generator to use
            batch_size (int): The size of the batch
        """
        self.batch_size = batch_size

        self.mask_generator = mask_generator
        mask_generator_dataset = tf.data.Dataset.from_generator(
            self.mask_generator,
            output_signature=(
                tf.TensorSpec(shape=self.image_size, dtype=tf.int64),
                tf.TensorSpec(shape=(), dtype=tf.int64)
            )
        )
        mask_generator_dataset = mask_generator_dataset.prefetch(tf.data.AUTOTUNE)
        self.mask_generator_dataset = mask_generator_dataset

    @property
    def batch_size(self) -> int:
        """
        Get the size of the batch.

        Returns:
            int: The size of the batch
        """

        return self.__batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int) -> None:
        """
        Set the size of the batch.

        Args:
            batch_size (int): The size of the batch
        """

        if batch_size < 1:
            raise ValueError('The batch size must be at least 1')
        self.__batch_size = batch_size

    @property
    def image_size(self) -> tuple[int, int]:
        """
        Get the image size.

        Returns:
            tuple[int, int]: The image size
        """

        return self.mask_generator.mask_size

    @property
    def mask_generator_dataset(self) -> tf.data.Dataset:
        """
        Get the mask generator dataset.

        Returns:
            tf.data.Dataset: The mask generator dataset
        """

        return self.__mask_generator_dataset

    @mask_generator_dataset.setter
    def mask_generator_dataset(self, mask_generator_dataset: tf.data.Dataset) -> None:
        """
        Set the mask generator dataset.

        Args:
            mask_generator_dataset (tf.data.Dataset): The mask generator dataset
        """

        if mask_generator_dataset is None:
            raise ValueError('Mask generator dataset cannot be None.')
        elif not isinstance(mask_generator_dataset, tf.data.Dataset):
            raise TypeError('Mask generator dataset must be a tf.data.Dataset.')
        self.__mask_generator_dataset = mask_generator_dataset

    def apply_mask_with_return(self, image: np.ndarray | tf.Tensor, color: MaskColor = MaskColor.WHITE) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply a mask to an image.

        Args:
            image (np.ndarray | tf.Tensor): The image to apply the mask to
            color (MaskColor, optional): The color of the mask

        Returns:
            tuple[np.ndarray, np.ndarray]: The masked image and the mask
        """

        mask_and_count = self.mask_generator_dataset.take(1)
        mask, count = next(iter(mask_and_count))
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

    def apply_mask(self, image: np.ndarray | tf.Tensor, color: MaskColor = MaskColor.WHITE) -> np.ndarray:
        """
        Apply a mask to an image.

        Args:
            image (np.ndarray | tf.Tensor): The image to apply the mask to
            color (MaskColor, optional): The color of the mask. Defaults to MaskColor.WHITE

        Returns:
            np.ndarray: The masked image
        """

        return self.apply_mask_with_return(image, color)[0]

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
