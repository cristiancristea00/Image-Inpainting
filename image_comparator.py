from __future__ import annotations

from os import environ
from pathlib import Path
from typing import Final

import numpy as np
import tensorflow as tf
from lpips import LPIPS, im2tensor


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

        return float(mean.result().numpy().item())

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
            """
            Compute the mean squared error of the images.

            Args:
                image_batch1 (tf.Tensor): The first image batch
                image_batch2 (tf.Tensor): The second image batch

            Returns:
                tf.Tensor: The mean squared error
            """

            image_batch1 = tf.squeeze(image_batch1)
            image_batch2 = tf.squeeze(image_batch2)
            return tf.reduce_mean(tf.square(image_batch1 - image_batch2))

        dataset = dataset.map(compute_mse, num_parallel_calls=tf.data.AUTOTUNE)
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
            """
            Compute the mean absolute error of the images.

            Args:
                image_batch1 (tf.Tensor): The first image batch
                image_batch2 (tf.Tensor): The second image batch

            Returns:
                tf.Tensor: The mean absolute error
            """

            image_batch1 = tf.squeeze(image_batch1)
            image_batch2 = tf.squeeze(image_batch2)
            return tf.reduce_mean(tf.abs(image_batch1 - image_batch2))

        dataset = dataset.map(compute_mae, num_parallel_calls=tf.data.AUTOTUNE)
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
            """
            Compute the peak signal-to-noise ratio of the images.

            Args:
                image_batch1 (tf.Tensor): The first image batch
                image_batch2 (tf.Tensor): The second image batch

            Returns:
                tf.Tensor: The peak signal-to-noise ratio
            """

            image_batch1 = tf.squeeze(image_batch1)
            image_batch2 = tf.squeeze(image_batch2)
            batch_psnr = tf.image.psnr(image_batch1, image_batch2, max_val=max_image_value)
            return remove_infinity(batch_psnr)

        def remove_infinity(psnr: tf.Tensor) -> tf.Tensor:
            """
            Remove infinity values.

            Args:
                psnr (tf.Tensor): The psnr

            Returns:
                tf.Tensor: The psnr without infinity values
            """

            mask = tf.math.is_finite(psnr)
            mask.set_shape((None,))
            return tf.boolean_mask(psnr, mask)

        dataset = dataset.map(compute_psnr, num_parallel_calls=tf.data.AUTOTUNE)
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
            """
            Compute the structural similarity index of the images.

            Args:
                image_batch1 (tf.Tensor): The first image batch
                image_batch2 (tf.Tensor): The second image batch

            Returns:
                tf.Tensor: The structural similarity index
            """

            image_batch1 = tf.squeeze(image_batch1)
            image_batch2 = tf.squeeze(image_batch2)
            return tf.image.ssim(image_batch1, image_batch2, max_val=max_image_value)

        dataset = dataset.map(compute_ssim, num_parallel_calls=tf.data.AUTOTUNE)
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
            torch_home: Path = Path('..', 'torch').resolve()
            environ['TORCH_HOME'] = str(torch_home)
            cls.__perceptual_loss = LPIPS(verbose=False)

        def perceptual_loss_wrapper(image1: tf.Tensor, image2: tf.Tensor) -> tf.Tensor:
            """
            Compute the perceptual loss of the images.

            Args:
                image1 (tf.Tensor): The first image
                image2 (tf.Tensor): The second image

            Returns:
                tf.Tensor: The perceptual loss
            """

            def inner_perceptual_loss(inner_image1: tf.Tensor, inner_image2: tf.Tensor) -> float:
                """
                Compute the perceptual loss of the images.

                Args:
                    inner_image1 (tf.Tensor): The first image
                    inner_image2 (tf.Tensor): The second image

                Returns:
                    float: The perceptual loss
                """

                max_pixel_value: Final[float] = 255.0

                inner_image1 = inner_image1.numpy() * max_pixel_value
                inner_image2 = inner_image2.numpy() * max_pixel_value
                inner_image1 = im2tensor(np.squeeze(inner_image1))
                inner_image2 = im2tensor(np.squeeze(inner_image2))
                return 1 - float(cls.__perceptual_loss(inner_image1, inner_image2).item())

            return tf.py_function(inner_perceptual_loss, inp=[image1, image2], Tout=float)

        dataset = dataset.unbatch()
        dataset = dataset.map(perceptual_loss_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
        return cls.__compute_mean(dataset)
