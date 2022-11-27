from __future__ import annotations

from enum import Enum, unique
from pathlib import Path
from typing import Callable, Final

import numpy as np
import tensorflow as tf

from image_processor import ImageProcessor
from utils import InpaintingMethod


@unique
class CATEGORY(Enum):
    """
    Enum for the categories.
    """
    TRAIN: str = 'train'
    TEST: str = 'test'


class ImageBrowser:
    """
    Class for browsing images.
    """

    __DEFAULT_PATH: Final[Path] = Path('..', 'images', 'original')

    __MAX_IMAGE_VALUE: Final[float] = 255.0

    def __init__(self, image_size: int, batch_size: int, mask_ratio: tuple[float, float], path: Path = __DEFAULT_PATH) -> None:
        """
        Args:
            image_size (int): The size of the images
            batch_size (int): The size of the batch
            mask_ratio (tuple[float, float]): The ratio of the mask
        """
        self.image_size = image_size
        self.batch_size = batch_size
        self.mask_ratio = mask_ratio
        self.path = path

    @property
    def image_size(self) -> tuple[int, int]:
        """
        Get the size of the images.

        Returns:
            int: The size of the images
        """
        return self.__image_size

    @image_size.setter
    def image_size(self, image_size: int) -> None:
        """
        Set the size of the images.

        Args:
            image_size (int): The size of the images
        """
        if image_size <= 0:
            raise ValueError('The image size must be positive.')
        if image_size < 64 or image_size > 512:
            raise ValueError('The image size must be between 64 and 512.')
        self.__image_size = (image_size, image_size)

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
    def mask_ratio(self) -> tuple[float, float]:
        """
        Get the ratio of the mask.

        Returns:
            tuple[float, float]: The ratio of the mask
        """
        return self.__mask_ratio

    @mask_ratio.setter
    def mask_ratio(self, mask_ratio: tuple[float, float]) -> None:
        """
        Set the ratio of the mask.

        Args:
            mask_ratio (tuple[float, float]): The ratio of the mask
        """
        if mask_ratio[0] < 0 or mask_ratio[1] > 100 or mask_ratio[0] > mask_ratio[1]:
            raise ValueError('The mask ratio values must be between 0 and 100')
        self.__mask_ratio = mask_ratio

    @property
    def path(self) -> Path:
        """
        Get the path of the images.

        Returns:
            Path: The path of the images
        """
        return self.__path

    @path.setter
    def path(self, path: Path) -> None:
        """
        Set the path of the images.

        Args:
            path (Path): The path of the images
        """
        if not path.exists():
            raise FileNotFoundError('The path does not exist')
        self.__path = path

    @staticmethod
    def __prefetch(dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Prefetches the dataset.

        Args:
            dataset (tf.data.Dataset): The dataset to prefetch

        Returns:
            tf.data.Dataset: The prefetched dataset
        """
        return dataset.prefetch(tf.data.AUTOTUNE)

    def __get_originals(self, category: CATEGORY, shuffle: bool = False) -> tf.data.Dataset:
        """
        Browses the original images and returns them as a dataset.

        Returns:
            tf.data.Dataset: The dataset of the original images
        """
        return tf.keras.utils.image_dataset_from_directory(self.__DEFAULT_PATH / category.value, image_size=self.image_size, labels=None,
                                                           batch_size=self.batch_size, shuffle=shuffle)

    @tf.autograph.experimental.do_not_convert
    def __normalize_pair(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        def normalize(image1: tf.Tensor, image2: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
            return image1 / self.__MAX_IMAGE_VALUE, image2 / self.__MAX_IMAGE_VALUE

        return dataset.map(normalize)

    @tf.autograph.experimental.do_not_convert
    def __normalize(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        def normalize(image: tf.Tensor) -> tf.Tensor:
            return image / self.__MAX_IMAGE_VALUE

        return dataset.map(normalize)

    def __get_masked_dataset_tuple(self, image_batch: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        def mask_transformer(image: tf.Tensor | np.ndarray) -> tuple[tf.Tensor, tf.Tensor] | tuple[np.ndarray, np.ndarray]:
            return ImageProcessor.apply_mask_with_return(image, ratio=self.mask_ratio)

        image_batch = tf.cast(image_batch, tf.float32)
        masked_image, mask = tf.map_fn(mask_transformer, image_batch, fn_output_signature=(tf.float32, tf.uint8))
        original_image = tf.cast(image_batch, tf.float32)
        masked_image = tf.cast(masked_image, tf.float32)
        mask = tf.cast(mask, tf.float32)
        return masked_image, original_image, mask

    def __get_masked_dataset_pair(self, image_batch: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        def mask_transformer(image: tf.Tensor | np.ndarray) -> tf.Tensor | np.ndarray:
            return ImageProcessor.apply_mask(image, ratio=self.mask_ratio)

        image_batch = tf.cast(image_batch, tf.float32)
        masked_image = tf.map_fn(mask_transformer, image_batch)
        original_image = tf.cast(image_batch, tf.float32)
        masked_image = tf.cast(masked_image, tf.float32)
        return masked_image, original_image

    def __get_masked_tuple(self) -> tf.data.Dataset:
        """
        Browses the original images and returns them as a dataset along with the
        masked version.

        Returns:
            tf.data.Dataset: The dataset of the masked images
        """
        originals = self.__get_originals(CATEGORY.TEST)
        masked = originals.map(self.__get_masked_dataset_tuple, num_parallel_calls=tf.data.AUTOTUNE)
        return self.__prefetch(masked)

    def __get_masked_pair(self) -> tf.data.Dataset:
        """
        Browses the original images and returns them as a dataset along with the
        masked version.

        Returns:
            tf.data.Dataset: The dataset of the masked images
        """
        originals = self.__get_originals(CATEGORY.TEST)
        masked = originals.map(self.__get_masked_dataset_pair, num_parallel_calls=tf.data.AUTOTUNE)
        return self.__prefetch(masked)

    @classmethod
    def __get_inpaint_transformer(cls, inpaint_method: InpaintingMethod) -> Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]]:
        """
        Returns a function that applies the inpainting method to the masked image.

        Args:
            inpaint_method (InpaintingMethod): The inpainting method

        Returns:
            Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]]: The function that applies the inpainting method
        """
        if inpaint_method is InpaintingMethod.NAVIER_STOKES:
            inpaint_func = ImageProcessor.inpaint_navier_stokes
        elif inpaint_method is InpaintingMethod.TELEA:
            inpaint_func = ImageProcessor.inpaint_telea
        else:
            raise ValueError('The inpainting method is not valid')

        def inpaint_transformer(masked: tf.Tensor, original: tf.Tensor, mask: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
            def inner_transformer(masked_and_mask: tuple[tf.Tensor, tf.Tensor]) -> tuple[tf.Tensor, tf.Tensor]:
                def transformer_wrapper(masked_image: tf.Tensor, mask_layers: tf.Tensor):
                    return tf.py_function(inpaint_func, inp=[masked_image, mask_layers], Tout=tf.float32)

                return transformer_wrapper(masked_and_mask[0], masked_and_mask[1])

            inpainted_image = tf.map_fn(inner_transformer, (masked, mask), fn_output_signature=tf.float32)
            return inpainted_image, original

        return inpaint_transformer

    def get_navier_stokes(self) -> tf.data.Dataset:
        """
        Browses the inpainted images using Navier-Stokes and returns them as a dataset.

        Returns:
            tf.data.Dataset: The dataset of the inpainted images
        """

        inpaint_transformer = self.__get_inpaint_transformer(InpaintingMethod.NAVIER_STOKES)
        masked_dataset = self.__get_masked_tuple()
        inpainted_dataset = masked_dataset.map(inpaint_transformer, num_parallel_calls=tf.data.AUTOTUNE)
        normalized = self.__normalize_pair(inpainted_dataset)
        return self.__prefetch(normalized)

    def get_telea(self) -> tf.data.Dataset:
        """
        Browses the inpainted images using Telea and returns them as a dataset.

        Returns:
            tf.data.Dataset: The dataset of the inpainted images
        """

        inpaint_transformer = self.__get_inpaint_transformer(InpaintingMethod.TELEA)
        masked_dataset = self.__get_masked_tuple()
        inpainted_dataset = masked_dataset.map(inpaint_transformer, num_parallel_calls=tf.data.AUTOTUNE)
        normalized = self.__normalize_pair(inpainted_dataset)
        return self.__prefetch(normalized)

    def get_train_dataset(self) -> tf.data.Dataset:
        """
        Browses the original images and returns them as a dataset along with the
        masked version.

        Returns:
            tf.data.Dataset: The dataset of the masked images
        """
        originals = self.__get_originals(CATEGORY.TRAIN, shuffle=True)
        masked = originals.map(self.__get_masked_dataset_pair, num_parallel_calls=tf.data.AUTOTUNE)
        normalized = self.__normalize_pair(masked)
        return self.__prefetch(normalized)

    def get_test_dataset(self) -> tf.data.Dataset:
        """
        Browses the original images and returns them as a dataset along with the
        masked version.

        Returns:
            tf.data.Dataset: The dataset of the masked images
        """
        originals = self.__get_originals(CATEGORY.TEST)
        masked = originals.map(self.__get_masked_dataset_pair, num_parallel_calls=tf.data.AUTOTUNE)
        normalized = self.__normalize_pair(masked)
        return self.__prefetch(normalized)

    def get_model_inpainted(self, model: tf.keras.Model) -> tf.data.Dataset:
        """
        Browses the images and inpaints them using the model and returns them as
        a dataset.

        Args:
            model (tf.keras.Model): The model to use for inpainting

        Returns:
            tf.data.Dataset: The dataset of the inpainted images
        """
        masked_dataset = self.__get_masked_pair()
        normalized = self.__normalize_pair(masked_dataset)
        inpainted_dataset = normalized.map(lambda masked, original: (model(masked), original), num_parallel_calls=tf.data.AUTOTUNE)
        return self.__prefetch(inpainted_dataset)
