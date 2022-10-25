from collections.abc import Iterator

import numpy as np
import cv2 as cv

from image_browser import ImageBrowser
from utils import InpaintMethod


class ImageComparator:
    """
    Class for comparing images.
    """

    @classmethod
    def compute_mse(cls, original_image: np.ndarray, inpainted_image: np.ndarray) -> float:
        """
        Compute the MSE of an inpainted image and the original image.

        Args:
            original_image (np.ndarray): The original image
            inpainted_image (np.ndarray): The inpainted image

        Returns:
            float: The MSE
        """

        return cv.quality.QualityMSE_compute(original_image, inpainted_image)[0][0]

    @classmethod
    def compute_psnr(cls, original_image: np.ndarray, inpainted_image: np.ndarray) -> float:
        """
        Compute the PSNR of an inpainted image and the original image.

        Args:
            original_image (np.ndarray): The original image
            inpainted_image (np.ndarray): The inpainted image

        Returns:
            float: The PSNR
        """

        return cv.quality.QualityPSNR_compute(original_image, inpainted_image)[0][0]

    @classmethod
    def compute_ssim(cls, original_image: np.ndarray, inpainted_image: np.ndarray) -> float:
        """
        Compute the SSIM of an inpainted image and the original image.

        Args:
            original_image (np.ndarray): The original image
            inpainted_image (np.ndarray): The inpainted image

        Returns:
            float: The SSIM
        """

        return cv.quality.QualitySSIM_compute(original_image, inpainted_image)[0][0]

    @classmethod
    def compute_results(cls, inpaint_method: InpaintMethod) -> list[tuple[str, str, str, str]]:
        """
        Compute the MSE, PSNR and SSIM of all images for a given inpaint method.

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
            original, masked, inpainted, name = elem
            mse = cls.compute_mse(original, inpainted)
            psnr = cls.compute_psnr(original, inpainted)
            ssim = cls.compute_ssim(original, inpainted)
            results.append((name, mse, psnr, ssim))

        results.sort(key=lambda pair: int(pair[0][:-4]))

        average_mse = compute_average((result[1] for result in results))
        average_psnr = compute_average((result[2] for result in results))
        average_ssim = compute_average((result[3] for result in results))

        std_mse = compute_std((result[1] for result in results), average_mse)
        std_psnr = compute_std((result[2] for result in results), average_psnr)
        std_ssim = compute_std((result[3] for result in results), average_ssim)

        results = [(F'{values[0][:-4]}', F'{values[1]:.3f}', F'{values[2]:.3f}', F'{values[3]:.3f}')
                   for values in results]

        averages = ('AVG', F'{average_mse:.3f}', F'{average_psnr:.3f}', F'{average_ssim:.3f}')
        results.append(averages)

        stds = ('STD', F'{std_mse:.3f}', F'{std_psnr:.3f}', F'{std_ssim:.3f}')
        results.append(stds)

        return results
