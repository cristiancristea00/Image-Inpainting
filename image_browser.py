from collections.abc import Iterator
from utils import InpaintMethod
from typing import Final
from pathlib import Path
import numpy as np
import cv2 as cv


class ImageBrowser:
    """
    Class for browsing images.
    """

    __ORIGINAL_PATH: Final = Path('images/original')
    __MASKED_PATH: Final = Path('images/masked')
    __MASKS_PATH: Final = Path('images/masks')
    __INPAINTED_PATH: Final = Path('images/inpainted')

    @classmethod
    def generate_originals(cls) -> Iterator[np.ndarray, str]:
        """
        Browses the original images and yields them.

        Yields:
            Iterator[np.ndarray, str]: The original image and its name
        """
        for elem in cls.__ORIGINAL_PATH.resolve(strict=True).iterdir():
            yield cv.imread(str(elem)), elem.name

    @classmethod
    def generate_masked(cls) -> Iterator[np.ndarray, str]:
        """
        Browses the masked images and yields them.

        Yields:
            Iterator[np.ndarray, str]: The masked image and its name
        """
        for elem in cls.__MASKED_PATH.resolve(strict=True).iterdir():
            yield cv.imread(str(elem)), elem.name

    @classmethod
    def generate_masks(cls) -> Iterator[np.ndarray, str]:
        """
        Browses the masks and yields them.

        Yields:
            Iterator[np.ndarray, str]: The mask and its name
        """
        for elem in cls.__MASKS_PATH.resolve(strict=True).iterdir():
            yield cv.imread(str(elem)), elem.name

    @classmethod
    def __generate_inpainted(cls, inpaint_method: InpaintMethod) -> Iterator[np.ndarray, str]:
        """
        Browses the inpainted images and yields them.

        Args:
            inpaint_method (InpaintMethod): The inpaint method used

        Yields:
            Iterator[np.ndarray, str]: The inpainted image and its name
        """
        inpaint_path = cls.__INPAINTED_PATH / inpaint_method.value
        for elem in inpaint_path.resolve(strict=True).iterdir():
            yield cv.imread(str(elem)), elem.name

    @classmethod
    def generate_patch_match(cls) -> Iterator[np.ndarray, str]:
        """
        Browses the inpainted images using Patch Match and yields them.

        Yields:
            Iterator[np.ndarray, str]: The inpainted image and its name
        """
        yield from cls.__generate_inpainted(InpaintMethod.PATCH_MATCH)

    @classmethod
    def generate_navier_stokes(cls) -> Iterator[np.ndarray, str]:
        """
        Browses the inpainted images using Navier-Stokes and yields them.

        Yields:
            Iterator[np.ndarray, str]: The inpainted image and its name
        """
        yield from cls.__generate_inpainted(InpaintMethod.NAVIER_STOKES)

    @classmethod
    def generate_telea(cls) -> Iterator[np.ndarray, str]:
        """
        Browses the inpainted images using Telea and yields them.

        Yields:
            Iterator[np.ndarray, str]: The inpainted image and its name
        """
        yield from cls.__generate_inpainted(InpaintMethod.TELEA)

    @classmethod
    def generate_all(cls, inpaint_method: InpaintMethod) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
        """
        Browses the original, masked and inpainted images and yields them.

        Args:
            inpaint_method (InpaintMethod): The inpaint method used

        Yields:
            Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, str]]: The original, masked and inpainted images and their name
        """
        for elem in cls.__ORIGINAL_PATH.resolve(strict=True).iterdir():
            masked_path = (cls.__MASKED_PATH / elem.name).resolve(strict=True)
            inpainted_path = (cls.__INPAINTED_PATH / inpaint_method.value / elem.name).resolve(strict=True)

            original = cv.imread(str(elem))
            masked = cv.imread(str(masked_path))
            inpainted = cv.imread(str(inpainted_path))

            yield original, masked, inpainted, elem.name
