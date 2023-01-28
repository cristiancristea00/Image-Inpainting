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

    def __init__(self, image_processor: ImageProcessor, path: Path = __DEFAULT_PATH) -> None:
        """
        Args:
            image_processor (ImageProcessor): The image processor
            path (Path, optional): The path to the images. Defaults to __DEFAULT_PATH.
        """

        self.image_processor = image_processor
        self.path = path

    @property
    def image_processor(self) -> ImageProcessor:
        """
        Get the image processor.

        Returns:
            ImageProcessor: The image processor
        """

        return self.__image_processor

    @image_processor.setter
    def image_processor(self, image_processor: ImageProcessor) -> None:
        """
        Set the image processor.

        Args:
            image_processor (ImageProcessor): The image processor
        """

        if image_processor is None:
            raise ValueError('The image processor cannot be None.')
        self.__image_processor = image_processor

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

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

        dataset = tf.keras.utils.image_dataset_from_directory(self.__DEFAULT_PATH / category.value, image_size=self.image_processor.image_size,
                                                              labels=None, batch_size=self.image_processor.batch_size, shuffle=shuffle)
        dataset = dataset.with_options(options)
        return self.__prefetch(dataset)

    @tf.autograph.experimental.do_not_convert
    def __normalize_pair(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Normalizes the dataset.

        Args:
            dataset (tf.data.Dataset): The dataset to normalize

        Returns:
            tf.data.Dataset: The normalized dataset
        """

        def normalize(image1: tf.Tensor, image2: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
            """
            Normalizes the images.

            Args:
                image1 (tf.Tensor): The first image
                image2 (tf.Tensor): The second image

            Returns:
                tuple[tf.Tensor, tf.Tensor]: The normalized images
            """

            return image1 / self.__MAX_IMAGE_VALUE, image2 / self.__MAX_IMAGE_VALUE

        return dataset.map(normalize)

    @tf.autograph.experimental.do_not_convert
    def __normalize(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Normalizes the dataset.

        Args:
            dataset (tf.data.Dataset): The dataset to normalize

        Returns:
            tf.data.Dataset: The normalized dataset
        """

        def normalize(image: tf.Tensor) -> tf.Tensor:
            """
            Normalizes the image.

            Args:
                image (tf.Tensor): The image

            Returns:
                tf.Tensor: The normalized image
            """

            return image / self.__MAX_IMAGE_VALUE

        return dataset.map(normalize)

    def __get_masked_dataset_tuple(self, image_batch: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Gets the masked dataset tuple.

        Args:
            image_batch (tf.Tensor): The image batch

        Returns:
            tuple[tf.Tensor, tf.Tensor, tf.Tensor]: The masked dataset tuple
        """

        def mask_transformer(image: tf.Tensor | np.ndarray) -> tuple[tf.Tensor, tf.Tensor] | tuple[np.ndarray, np.ndarray]:
            """
            Transforms the image into a masked image and a mask.

            Args:
                image (tf.Tensor | np.ndarray): The image

            Returns:
                tuple[tf.Tensor, tf.Tensor] | tuple[np.ndarray, np.ndarray]: The masked image and the mask
            """

            return self.image_processor.apply_mask_with_return(image)

        image_batch = tf.cast(image_batch, tf.float32)
        masked_image, mask = tf.map_fn(mask_transformer, image_batch, fn_output_signature=(tf.float32, tf.uint8))
        original_image = tf.cast(image_batch, tf.float32)
        masked_image = tf.cast(masked_image, tf.float32)
        mask = tf.cast(mask, tf.float32)
        return masked_image, original_image, mask

    def __get_masked_dataset_pair(self, image_batch: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Gets the masked dataset pair.

        Args:
            image_batch (tf.Tensor): The image batch

        Returns:
            tuple[tf.Tensor, tf.Tensor]: The masked dataset pair
        """

        def mask_transformer(image: tf.Tensor | np.ndarray) -> tf.Tensor | np.ndarray:
            """
            Transforms the image into a masked image.

            Args:
                image (tf.Tensor | np.ndarray): The image

            Returns:
                tf.Tensor | np.ndarray: The masked image
            """

            return self.image_processor.apply_mask(image)

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

    def __get_inpaint_transformer(self, inpaint_method: InpaintingMethod) -> Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]]:
        """
        Returns a function that applies the inpainting method to the masked image.

        Args:
            inpaint_method (InpaintingMethod): The inpainting method

        Returns:
            Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]]: The function that applies the inpainting method
        """

        if inpaint_method is InpaintingMethod.NAVIER_STOKES:
            inpaint_func = self.image_processor.inpaint_navier_stokes
        elif inpaint_method is InpaintingMethod.TELEA:
            inpaint_func = self.image_processor.inpaint_telea
        else:
            raise ValueError('The inpainting method is not valid')

        def inpaint_transformer(masked: tf.Tensor, original: tf.Tensor, mask: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
            """
            Applies the inpainting method to the masked image.

            Args:
                masked (tf.Tensor): The masked image
                original (tf.Tensor): The original image
                mask (tf.Tensor): The mask

            Returns:
                tuple[tf.Tensor, tf.Tensor]: The inpainted image and the original image
            """

            def inner_transformer(masked_and_mask: tuple[tf.Tensor, tf.Tensor]) -> tuple[tf.Tensor, tf.Tensor]:
                """
                Applies the inpainting method to the masked image.

                Args:
                    masked_and_mask (tuple[tf.Tensor, tf.Tensor]): The masked image and the mask

                Returns:
                    tuple[tf.Tensor, tf.Tensor]: The inpainted image and the mask
                """

                def transformer_wrapper(masked_image: tf.Tensor, mask_layer: tf.Tensor) -> np.ndarray | tf.Tensor:
                    """
                    Wraps the inpaint function.

                    Args:
                        masked_image (tf.Tensor): The masked image
                        mask_layer (tf.Tensor): The mask layer

                    Returns:
                        np.ndarray | tf.Tensor: The inpainted image
                    """

                    return tf.py_function(inpaint_func, inp=[masked_image, mask_layer], Tout=tf.float32)

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
        cached = normalized.cache()
        return self.__prefetch(cached)

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
        cached = normalized.cache()
        return self.__prefetch(cached)

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
        cached = normalized.cache()
        return self.__prefetch(cached)

    def get_test_dataset(self) -> tf.data.Dataset:
        """
        Browses the original images and returns them as a dataset along with the
        masked version.

        Returns:
            tf.data.Dataset: The dataset of the masked images
        """

        originals = self.__get_originals(CATEGORY.TEST, shuffle=False)
        masked = originals.map(self.__get_masked_dataset_pair, num_parallel_calls=tf.data.AUTOTUNE)
        normalized = self.__normalize_pair(masked)
        cached = normalized.cache()
        return self.__prefetch(cached)

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
        cached = inpainted_dataset.cache()
        return self.__prefetch(cached)
