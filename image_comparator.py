import numpy as np


class ImageComparator:
    """
    Class for comparing images.
    """

    @classmethod
    def __compute_mse(cls, first_image: np.ndarray, second_image: np.ndarray) -> float:
        """
        Compute the mean squared error between two images.

        Args:
            first_image (np.ndarray): The first image
            second_image (np.ndarray): The second image

        Returns:
            float: The mean squared error
        """
        return np.mean((first_image - second_image) ** 2)

    @classmethod
    def compute_difference(cls, original_image: np.ndarray, masked_image: np.ndarray, inpainted_image: np.ndarray) -> float:
        """
        Compute the difference of an inpainted image.

        Args:
            original_image (np.ndarray): The original image
            masked_image (np.ndarray): The masked image
            inpainted_image (np.ndarray): The inpainted image

        Returns:
            float: The difference
        """

        mse_masked = cls.__compute_mse(original_image, masked_image)
        mse_inpainted = cls.__compute_mse(original_image, inpainted_image)
        return np.abs(mse_masked - mse_inpainted)
