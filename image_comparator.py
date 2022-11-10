from __future__ import annotations

from collections.abc import Iterator

import cv2 as cv
from lpips import LPIPS, im2tensor
import numpy as np
import tensorflow as tf

from image_browser import ImageBrowser
from utils import InpaintingMethod


class ImageComparator:
    """
    Class for comparing images.
    """

    @classmethod
    def compute_mse(cls, original_image: np.ndarray | tf.Tensor, inpainted_image: np.ndarray | tf.Tensor) -> tf.Tensor | float:
        """
        Compute the MSE of an inpainted image and the original image.

        Args:
            original_image (np.ndarray | tf.Tensor): The original image
            inpainted_image (np.ndarray | tf.Tensor): The inpainted image

        Returns:
            float: The MSE
        """

        if isinstance(original_image, tf.Tensor) or isinstance(inpainted_image, tf.Tensor):

            return tf.reduce_mean(tf.square(original_image - inpainted_image))

        else:

            mse = cv.quality.QualityMSE_compute(original_image, inpainted_image)
            return np.mean(mse[1]).item()

    @classmethod
    def compute_mae(cls, original_image: np.ndarray | tf.Tensor, inpainted_image: np.ndarray | tf.Tensor) -> tf.Tensor | float:
        """
        Compute the MAE of an inpainted image and the original image.

        Args:
            original_image (np.ndarray | tf.Tensor): The original image
            inpainted_image (np.ndarray | tf.Tensor): The inpainted image

        Returns:
            float: The MAE
        """

        if isinstance(original_image, tf.Tensor) or isinstance(inpainted_image, tf.Tensor):

            return tf.reduce_mean(tf.abs(original_image - inpainted_image))

        else:

            mae = np.abs(original_image - inpainted_image)
            return np.mean(mae).item()

    @classmethod
    def compute_psnr(cls, original_image: np.ndarray | tf.Tensor, inpainted_image: np.ndarray | tf.Tensor) -> tf.Tensor | float:
        """
        Compute the PSNR of an inpainted image and the original image.

        Args:
            original_image (np.ndarray | tf.Tensor): The original image
            inpainted_image (np.ndarray | tf.Tensor): The inpainted image

        Returns:
            float: The PSNR
        """

        if isinstance(original_image, tf.Tensor) or isinstance(inpainted_image, tf.Tensor):

            return tf.image.psnr(original_image, inpainted_image, 1)

        else:

            psnr = cv.quality.QualityPSNR_compute(original_image, inpainted_image)
            return np.mean(psnr[1]).item()

    @classmethod
    def compute_ssim(cls, original_image: np.ndarray | tf.Tensor, inpainted_image: np.ndarray | tf.Tensor) -> tf.Tensor | float:
        """
        Compute the SSIM of an inpainted image and the original image.

        Args:
            original_image (np.ndarray | tf.Tensor): The original image
            inpainted_image (np.ndarray | tf.Tensor): The inpainted image

        Returns:
            float: The SSIM
        """

        if isinstance(original_image, tf.Tensor) or isinstance(inpainted_image, tf.Tensor):

            return tf.image.ssim(original_image, inpainted_image, 1)

        else:

            ssim = cv.quality.QualitySSIM_compute(original_image, inpainted_image)
            return np.mean(ssim[1]).item()

    @classmethod
    def compute_lpips(cls, original_image: np.ndarray, inpainted_image: np.ndarray) -> float:
        """
        Compute the LPIPS of an inpainted image and the original image.

        Args:
            original_image (np.ndarray): The original image
            inpainted_image (np.ndarray): The inpainted image

        Returns:
            float: The LPIPS
        """

        perceptual_loss = LPIPS(net='vgg')

        original = im2tensor(original_image)
        inpainted = im2tensor(inpainted_image)

        return perceptual_loss(original, inpainted).item()

    @classmethod
    def compute_results(cls, inpaint_method: InpaintingMethod) -> list[tuple[str, str, str, str, str]]:
        """
        Compute the MSE, MAE, PSNR and SSIM of all images for a given inpaint method.

        Args:
            inpaint_method (InpaintMethod): The inpaint method

        Returns:
            list[tuple[str, str, str, str]]: The results
        """

        def compute_average(values: Iterator[float]) -> float:
            """
            Compute the average of a list of values.

            Args:
                values (list): The values

            Returns:
                float: The average
            """
            values = list(values)
            return sum(value for value in values) / len(values)

        def compute_std(values: Iterator[float], computed_mse: float) -> float:
            """
            Compute the standard deviation of a list of values.

            Args:
                values (list): The values
                computed_mse (float): The computed MSE

            Returns:
                float: The standard deviation
            """
            values = list(values)
            return (sum((value - computed_mse) ** 2 for value in values) / len(values)) ** 0.5

        results = []

        for elem in ImageBrowser.generate_all(inpaint_method):
            original, _, inpainted, name = elem
            mse = cls.compute_mse(original, inpainted)
            mae = cls.compute_mae(original, inpainted)
            psnr = cls.compute_psnr(original, inpainted)
            ssim = cls.compute_ssim(original, inpainted)
            lpips = cls.compute_lpips(original, inpainted)
            results.append((name, mse, mae, psnr, ssim, lpips))

        results.sort(key=lambda pair: int(pair[0][:-4]))

        average_mse = compute_average((result[1] for result in results))
        average_mae = compute_average((result[2] for result in results))
        average_psnr = compute_average((result[3] for result in results))
        average_ssim = compute_average((result[4] for result in results))
        average_lpips = compute_average((result[5] for result in results))

        std_mse = compute_std((result[1] for result in results), average_mse)
        std_mae = compute_std((result[2] for result in results), average_mae)
        std_psnr = compute_std((result[3] for result in results), average_psnr)
        std_ssim = compute_std((result[4] for result in results), average_ssim)
        std_lpips = compute_std((result[5] for result in results), average_lpips)

        results = [(F'{values[0][:-4]}', F'{values[1]:.3f}', F'{values[2]:.3f}', F'{values[3]:.3f}', F'{values[4]:.3f}', F'{values[5]:.3f}') for values in results]

        averages = ('AVG', F'{average_mse:.3f}', F'{average_mae:.3f}', F'{average_psnr:.3f}', F'{average_ssim:.3f}', F'{average_lpips:.3f}')
        results.append(averages)

        stds = ('STD', F'{std_mse:.3f}', F'{std_mae:.3f}', F'{std_psnr:.3f}', F'{std_ssim:.3f}', F'{std_lpips:.3f}')
        results.append(stds)

        return results
