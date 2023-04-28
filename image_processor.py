import cv2 as cv
import numpy as np
import tensorflow as tf
from skimage.util import view_as_windows

from mask_generator import MaskGenerator
from utils import MaskColor


class ImageProcessor:
    """
    Class for processing images.
    """

    def __init__(self, mask_generator: MaskGenerator) -> None:
        """
        Initialize the ImageProcessor.

        Args:
            mask_generator (MaskGenerator): The mask generator to use
        """

        self.mask_generator = mask_generator

    @property
    def mask_generator(self) -> MaskGenerator:
        """
        Get the mask generator.

        Returns:
            MaskGenerator: The mask generator
        """

        return self._mask_generator

    @mask_generator.setter
    def mask_generator(self, mask_generator: MaskGenerator) -> None:
        """
        Set the mask generator.

        Args:
            mask_generator (MaskGenerator): The mask generator
        """

        if not isinstance(mask_generator, MaskGenerator):
            raise TypeError(F'Expected mask_generator to be of type MaskGenerator, but got {type(mask_generator)}.')
        elif mask_generator is None:
            raise ValueError('Expected mask_generator to not be None.')
        else:
            self._mask_generator = mask_generator

    @property
    def image_size(self) -> tuple[int, int]:
        """
        Get the image size.

        Returns:
            tuple[int, int]: The image size
        """

        return self.mask_generator.mask_size

    def apply_mask(self, image: tf.Tensor, color: MaskColor = MaskColor.WHITE) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Apply a mask to an image.

        Args:
            image (tf.Tensor): The image to apply the mask to
            color (MaskColor, optional): The color of the mask

        Returns:
            tuple[tf.Tensor, tf.Tensor]: The masked image and the mask
        """

        # We use this to avoid deterministic masks
        mask, count = tf.py_function(self.mask_generator.generate, [], (tf.uint8, tf.uint32))
        initial_mask = mask

        mask = tf.stack([mask, mask, mask], axis=-1)
        mask = tf.where(mask)

        if color == MaskColor.WHITE:

            update_values = tf.fill((count * 3,), MaskColor.WHITE.value)
            update_values = tf.cast(update_values, tf.uint8)

        elif color == MaskColor.BLACK:

            update_values = tf.fill((count * 3,), MaskColor.BLACK.value)
            update_values = tf.cast(update_values, tf.uint8)

        else:

            raise ValueError('Invalid colour')

        masked = tf.tensor_scatter_nd_update(image, mask, update_values)

        return masked, initial_mask

    def apply_mask_numpy(self, image: np.ndarray, color: MaskColor = MaskColor.WHITE) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply a mask to an image.

        Args:
            image (np.ndarray): The image to apply the mask to
            color (MaskColor, optional): The color of the mask

        Returns:
            tuple[np.ndarray, np.ndarray]: The masked image and the mask
        """

        mask, _ = self.mask_generator.generate()
        masked = image.copy()

        if color == MaskColor.WHITE:

            masked[mask == MaskGenerator.__MASK_VALUE] = MaskColor.WHITE.value

        elif color == MaskColor.BLACK:

            masked[mask == MaskGenerator.__MASK_VALUE] = MaskColor.BLACK.value

        else:

            raise ValueError('Invalid colour')

        return masked, mask

    @staticmethod
    def inpaint_navier_stokes(image: np.ndarray | tf.Tensor, mask: np.ndarray | tf.Tensor,
                              radius: float = 3.0) -> np.ndarray:
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

    @staticmethod
    def extract_patches(image: np.ndarray, patch_size: int, step: int = 16) -> tuple[np.ndarray, tuple[int, int, int]]:
        """
        Extract patches from an image. The image is padded with the reflect mode to ensure that the patches are extracted from the entire image.

        Args:
            image (ndarray): The image
            patch_size (int): The patch size
            step (int): The step size. Defaults to 16.

        Returns:
            tuple[ndarray, tuple[int, int, int]]: The patches and the shape of the padded image
        """

        height, width, channels = image.shape

        x_start, y_start = 0, 0
        x_end, y_end = height, width

        row_padding = (patch_size - (height - patch_size) % step) % step
        col_padding = (patch_size - (width - patch_size) % step) % step

        height_padding = (0, row_padding)
        width_padding = (0, col_padding)
        channel_padding = (0, 0)
        padded_image = np.pad(image, (height_padding, width_padding, channel_padding), mode='reflect')

        x_end += row_padding
        y_end += col_padding

        patches = view_as_windows(padded_image[x_start:x_end, y_start:y_end], (patch_size, patch_size, channels), step=step)
        patches = patches.squeeze()

        return patches, padded_image.shape

    @staticmethod
    def reconstruct_image(patches: np.ndarray, initial_shape: tuple[int, int, int], padded_shape: tuple[int, int, int], step: int = 16) -> np.ndarray:
        """
        Reconstruct an image from patches.

        Args:
            patches (ndarray): The patches
            initial_shape (tuple[int, int, int]): The initial shape of the image
            padded_shape (tuple[int, int, int]): The shape of the padded image
            step (int): The step size. Defaults to 16.

        Returns:
            ndarray: The reconstructed image
        """

        patch_size = patches.shape[2]

        reconstructed = np.zeros(padded_shape, dtype=np.float64)
        count = np.zeros(padded_shape)

        for x in range(patches.shape[0]):
            for y in range(patches.shape[1]):
                x_pos, y_pos = x * step, y * step
                reconstructed[x_pos:x_pos + patch_size, y_pos:y_pos + patch_size] += patches[x, y]
                count[x_pos:x_pos + patch_size, y_pos:y_pos + patch_size] += 1

        mean = np.uint8(np.round(reconstructed / count))

        height, width, _ = initial_shape
        result = mean[:height, :width]

        return result
