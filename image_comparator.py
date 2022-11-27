from __future__ import annotations

from collections.abc import Iterator

import tensorflow as tf
from lpips import LPIPS, im2tensor

from image_browser import ImageBrowser
from utils import InpaintingMethod


class ImageComparator:
    """
    Class for comparing images.
    """

    __perceptual_loss: LPIPS = None

    @classmethod
    def __compute_mean(cls, dataset: tf.data.Dataset) -> float:
        """
        Compute the mean of the dataset.

        Args:
            dataset (tf.data.Dataset): The dataset

        Returns:
            float: The mean of the dataset
        """

        mean = tf.keras.metrics.Mean()
        for batch in dataset:
            mean.update_state(batch)

        return mean.result().numpy().item()

    @classmethod
    def compute_mse(cls, dataset: tf.data.Dataset) -> float:
        """
        Compute the mean squared error of the images.

        Args:
            dataset (tf.data.Dataset): The dataset

        Returns:
            float: The mean squared error
        """

        def compute_mse(image_batch1: tf.Tensor, image_batch2: tf.Tensor) -> tf.Tensor:
            return tf.reduce_mean(tf.square(image_batch1 - image_batch2))

        dataset = dataset.map(compute_mse)
        return cls.__compute_mean(dataset)

    @classmethod
    def compute_mae(cls, dataset: tf.data.Dataset) -> float:
        """
        Compute the mean absolute error of the images.

        Args:
            dataset (tf.data.Dataset): The dataset

        Returns:
            float: The mean absolute error
        """

        def compute_mae(image_batch1: tf.Tensor, image_batch2: tf.Tensor) -> tf.Tensor:
            return tf.reduce_mean(tf.abs(image_batch1 - image_batch2))

        dataset = dataset.map(compute_mae)
        return cls.__compute_mean(dataset)

    @classmethod
    def compute_psnr(cls, dataset: tf.data.Dataset, max_image_value: float = 1.0) -> float:
        """
        Compute the peak signal-to-noise ratio of the images.

        Args:
            dataset (tf.data.Dataset): The dataset
            max_image_value (float): The maximum value of the images

        Returns:
            float: The peak signal-to-noise ratio
        """

        def compute_psnr(image_batch1: tf.Tensor, image_batch2: tf.Tensor) -> tf.Tensor:
            return tf.image.psnr(image_batch1, image_batch2, max_val=max_image_value)

        dataset = dataset.map(compute_psnr)
        return cls.__compute_mean(dataset)

    @classmethod
    def compute_ssim(cls, dataset: tf.data.Dataset, max_image_value: float = 1.0) -> float:
        """
        Compute the structural similarity index of the images.

        Args:
            dataset (tf.data.Dataset): The dataset
            max_image_value (float): The maximum value of the images

        Returns:
            float: The structural similarity index
        """

        def compute_ssim(image_batch1: tf.Tensor, image_batch2: tf.Tensor) -> tf.Tensor:
            return tf.image.ssim(image_batch1, image_batch2, max_val=max_image_value)

        dataset = dataset.map(compute_ssim)
        return cls.__compute_mean(dataset)

    @classmethod
    @tf.autograph.experimental.do_not_convert
    def compute_lpips(cls, dataset: tf.data.Dataset) -> float:
        """
        Compute the LPIPS of the images.

        Args:
            dataset (tf.data.Dataset): The dataset

        Returns:
            float: The LPIPS
        """

        if cls.__perceptual_loss is None:
            cls.__perceptual_loss = LPIPS(net='vgg')

        def perceptual_loss_wrapper(image_batch1: tf.Tensor, image_batch2: tf.Tensor) -> tf.Tensor:
            def inner_perceptual_loss(inner_image_batch1: tf.Tensor, inner_image_batch2: tf.Tensor) -> float:
                inner_image_batch1 = inner_image_batch1.numpy()
                inner_image_batch2 = inner_image_batch2.numpy()
                inner_image_batch1 = im2tensor(inner_image_batch1)
                inner_image_batch2 = im2tensor(inner_image_batch2)
                return cls.__perceptual_loss(inner_image_batch1, inner_image_batch2).item()

            return tf.py_function(inner_perceptual_loss, inp=[image_batch1, image_batch2], Tout=float)

        dataset = dataset.unbatch()
        dataset = dataset.map(perceptual_loss_wrapper)
        return cls.__compute_mean(dataset)
