from __future__ import annotations

import shutil
from collections.abc import Generator
from enum import Enum, unique
from pathlib import Path
from typing import Callable, Final

import cv2 as cv
import numpy as np
import tensorflow as tf

from image_processor import ImageProcessor
from utils import InpaintingMethod


@unique
class Category(Enum):
    """
    Enum for the categories.
    """

    TRAIN: str = 'train'
    VALIDATION: str = 'val'
    TEST: str = 'test'


class ImageBrowser:
    """
    Class for browsing images.
    """

    __DEFAULT_PATH: Final[Path] = Path('..', 'images', 'coco')

    __MAX_IMAGE_VALUE: Final[float] = 255.0

    def __init__(self, image_processor: ImageProcessor, path: Path = __DEFAULT_PATH, batch_size: int = 1, should_crop: bool = False) -> None:
        """
        Args:
            image_processor (ImageProcessor): The image processor
            path (Path, optional): The path to the images. Defaults to __DEFAULT_PATH.
            batch_size (int, optional): The size of the batch. Defaults to 1.
            should_crop (bool, optional): Whether the images should be cropped. Defaults to False.
        """

        self.image_processor = image_processor
        self.path = path.resolve()
        self.cache_path = self.path / 'cache'
        self.batch_size = batch_size
        self.should_crop = should_crop

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

    @property
    def cache_path(self) -> Path:
        """
        Get the path of the cache.

        Returns:
            Path: The path of the cache
        """

        return self.__cache_path

    @cache_path.setter
    def cache_path(self, cache_path: Path) -> None:
        """
        Set the path of the cache.

        Args:
            cache_path (Path): The path of the cache
        """

        if cache_path.exists():
            shutil.rmtree(cache_path)

        cache_path.mkdir(parents=True)
        self.__cache_path = cache_path

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
    def should_crop(self) -> bool:
        """
        Get whether the images should be cropped.

        Returns:
            bool: Whether the images should be cropped
        """

        return self.__should_crop

    @should_crop.setter
    def should_crop(self, should_crop: bool) -> None:
        """
        Set whether the images should be cropped.

        Args:
            should_crop (bool): Whether the images should be cropped
        """

        if not isinstance(should_crop, bool):
            raise TypeError('The should_crop must be a boolean.')
        self.__should_crop = should_crop

    @property
    def mask_ratio(self) -> tuple[int, int]:
        """
        Get the ratio of the mask.

        Returns:
            int: The ratio of the mask
        """

        return self.image_processor.mask_generator.mask_ratio

    def __prefetch_and_batch(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Prefetches and batches the dataset.

        Args:
            dataset (tf.data.Dataset): The dataset

        Returns:
            tf.data.Dataset: The prefetched and batched dataset
        """

        batched = dataset.batch(self.batch_size)
        return batched.prefetch(tf.data.AUTOTUNE)

    def __load_image(self, filepath: Path | str) -> tf.Tensor:
        """
        Loads an image from the filepath.

        Args:
            filepath (Path | str): The filepath of the image

        Returns:
            tf.Tensor: The image
        """

        image = tf.io.read_file(filepath)
        image = tf.image.decode_image(image, channels=3)

        if self.should_crop:

            image_size: Final[int] = self.image_processor.image_size[0]
            image = tf.image.random_crop(image, size=(image_size, image_size, 3))

        return image

    def __get_originals(self, category: Category, shuffle: bool = False) -> tf.data.Dataset:
        """
        Browses the original images and returns them as a dataset.

        Returns:
            tf.data.Dataset: The dataset of the original images
        """

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

        files_path: Final[str] = str(self.path / category.value / '*')
        dataset = tf.data.Dataset.list_files(files_path, shuffle=shuffle).with_options(options)
        dataset = dataset.map(self.__load_image, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset.prefetch(tf.data.AUTOTUNE)

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

            image1 = tf.cast(image1, tf.float32)
            image2 = tf.cast(image2, tf.float32)
            return image1 / self.__MAX_IMAGE_VALUE, image2 / self.__MAX_IMAGE_VALUE

        return dataset.map(normalize)

    def __get_masked_dataset_tuple(self, image: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Gets the masked dataset tuple.

        Args:
            image (tf.Tensor): The image

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

        image = tf.cast(image, tf.float32)
        image = tf.expand_dims(image, axis=0)
        masked_image, mask = tf.map_fn(mask_transformer, image, fn_output_signature=(tf.float32, tf.int64))
        masked_image = tf.squeeze(masked_image)
        image = tf.squeeze(image)
        original_image = tf.cast(image, tf.float32)
        masked_image = tf.cast(masked_image, tf.float32)
        mask = tf.cast(mask, tf.float32)
        return masked_image, original_image, mask

    def __get_masked_dataset_pair(self, image: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Gets the masked dataset pair.

        Args:
            image (tf.Tensor): The image

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

        image = tf.cast(image, tf.float32)
        image = tf.expand_dims(image, axis=0)
        masked_image = tf.map_fn(mask_transformer, image)
        masked_image = tf.squeeze(masked_image)
        image = tf.squeeze(image)
        original_image = tf.cast(image, tf.float32)
        masked_image = tf.cast(masked_image, tf.float32)
        return masked_image, original_image

    def __get_masked_tuple(self) -> tf.data.Dataset:
        """
        Browses the original images and returns them as a dataset along with the
        masked version.

        Returns:
            tf.data.Dataset: The dataset of the masked images
        """

        originals = self.__get_originals(Category.TEST)
        masked = originals.map(self.__get_masked_dataset_tuple, num_parallel_calls=tf.data.AUTOTUNE)
        return masked.prefetch(tf.data.AUTOTUNE)

    def __get_masked_pair(self) -> tf.data.Dataset:
        """
        Browses the original images and returns them as a dataset along with the
        masked version.

        Returns:
            tf.data.Dataset: The dataset of the masked images
        """

        originals = self.__get_originals(Category.TEST)
        masked = originals.map(self.__get_masked_dataset_pair, num_parallel_calls=tf.data.AUTOTUNE)
        return masked.prefetch(tf.data.AUTOTUNE)

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

            def inner_transformer(masked_and_mask: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
                """
                Applies the inpainting method to the masked image.

                Args:
                    masked_and_mask (tuple[tf.Tensor, tf.Tensor]): The masked image and the mask

                Returns:
                    tf.Tensor: The inpainted image
                """

                def transformer_wrapper(masked_image: tf.Tensor, mask_layer: tf.Tensor) -> tf.Tensor:
                    """
                    Wraps the inpaint function.

                    Args:
                        masked_image (tf.Tensor): The masked image
                        mask_layer (tf.Tensor): The mask layer

                    Returns:
                        tf.Tensor: The inpainted image
                    """

                    return tf.py_function(inpaint_func, inp=[masked_image, mask_layer], Tout=tf.float32)

                return transformer_wrapper(masked_and_mask[0], masked_and_mask[1])

            masked = tf.expand_dims(masked, axis=0)
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
        return self.__prefetch_and_batch(normalized)

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
        return self.__prefetch_and_batch(normalized)

    def get_patch_match(self) -> tf.data.Dataset:
        """
        Browses the inpainted images using PatchMatch and returns them as a dataset.

        Returns:
            tf.data.Dataset: The dataset of the inpainted images
        """

        def get_patch_match_images_generator() -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
            """
            Browses the inpainted images using PatchMatch and returns them as a tuple.

            Yields:
                tuple[np.ndarray, np.ndarray]: The inpainted image and the original image
            """

            patch_match_path = Path('..', 'images', 'patch_match').resolve()
            mask_ratio = self.image_processor.mask_generator.mask_ratio
            inpainted_path = patch_match_path / str(mask_ratio)

            images_path = self.__DEFAULT_PATH / Category.TEST.value

            for image_path in images_path.iterdir():
                original = cv.imread(str(image_path)).astype(np.float64)
                original = np.expand_dims(original, axis=0)
                masked = cv.imread(str(inpainted_path / image_path.with_suffix('.png').name)).astype(np.float64)
                masked = np.expand_dims(masked, axis=0)

                yield masked, original

        image_size = self.image_processor.image_size[0]

        inpainted_dataset = tf.data.Dataset.from_generator(
            generator=get_patch_match_images_generator,
            output_signature=(
                tf.TensorSpec(shape=(None, image_size, image_size, 3), dtype=tf.float64),
                tf.TensorSpec(shape=(None, image_size, image_size, 3), dtype=tf.float64)
            )
        )
        normalized = self.__normalize_pair(inpainted_dataset)
        return self.__prefetch_and_batch(normalized)

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
        inpainted_dataset = normalized.map(lambda masked, original: (model(tf.expand_dims(masked, axis=0)), original), num_parallel_calls=tf.data.AUTOTUNE)
        return self.__prefetch_and_batch(inpainted_dataset)

    def get_train_dataset(self, shuffle: bool = True) -> tf.data.Dataset:
        """
        Browses the original images and returns the train ones as a dataset
        along with the masked version.

        Args:
            shuffle (bool): Whether to shuffle the dataset

        Returns:
            tf.data.Dataset: The dataset of the masked images
        """

        originals = self.__get_originals(Category.TRAIN, shuffle=shuffle)
        masked = originals.map(self.__get_masked_dataset_pair, num_parallel_calls=tf.data.AUTOTUNE)
        normalized = self.__normalize_pair(masked)
        cached = normalized.cache()
        return self.__prefetch_and_batch(cached)

    def get_val_dataset(self, shuffle: bool = False) -> tf.data.Dataset:
        """
        Browses the original images and returns the validation ones as a dataset
        along with the masked version.

        Args:
            shuffle (bool): Whether to shuffle the dataset

        Returns:
            tf.data.Dataset: The dataset of the masked images
        """

        originals = self.__get_originals(Category.VALIDATION, shuffle=shuffle)
        masked = originals.map(self.__get_masked_dataset_pair, num_parallel_calls=tf.data.AUTOTUNE)
        normalized = self.__normalize_pair(masked)
        cached = normalized.cache()
        return self.__prefetch_and_batch(cached)

    def get_test_dataset(self, shuffle: bool = False) -> tf.data.Dataset:
        """
        Browses the original images and returns the test ones as a dataset along
        with the masked version.

        Args:
            shuffle (bool): Whether to shuffle the dataset

        Returns:
            tf.data.Dataset: The dataset of the masked images
        """

        originals = self.__get_originals(Category.TEST, shuffle=shuffle)
        masked = originals.map(self.__get_masked_dataset_pair, num_parallel_calls=tf.data.AUTOTUNE)
        normalized = self.__normalize_pair(masked)
        return self.__prefetch_and_batch(normalized)
