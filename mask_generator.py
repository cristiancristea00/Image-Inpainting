from __future__ import annotations

import os
import random
from collections.abc import Iterable
from typing import Final

import cv2 as cv
import numpy as np

from utils import MaskColor


class MaskGenerator:
    """
    Generator that yields random masks by drawing lines, circles and ellipses.
    The white part represents the area of interest.
    """

    MASK_VALUE: Final = MaskColor.WHITE.value

    DEFAULT_DRAW_SCALE: Final[float] = 0.015
    DEFAULT_MASK_RATIO: Final[tuple[float, float]] = (0, 0)

    def __init__(self, mask_size: int | tuple[int, int], ratio: tuple[float, float] = DEFAULT_MASK_RATIO,
                 draw_scale: float = DEFAULT_DRAW_SCALE) -> None:
        """
        Initializes the generator.

        Args:
            mask_size (tuple[int, int]): Mask dimensions
            ratio (tuple[float, float]): Percentage interval of the image that is covered by the mask. Defaults to DEFAULT_MASK_RATIO
            draw_scale (float, optional): Drawing objects scaling factor. Defaults to DEFAULT_DRAW_SCALE
        """

        self.mask_size = mask_size  # type: ignore
        self.mask_ratio = ratio
        self.width, self.height = self.mask_size
        self.ratio_min, self.ratio_max = self.mask_ratio
        self.draw_scale = draw_scale

        self.random_generator = random.SystemRandom(int(os.urandom(32).hex(), 16))

    @property
    def mask_size(self) -> tuple[int, int]:
        """
        Mask size property.

        Returns:
            tuple[int, int]: Mask size
        """
        return self.__mask_size

    @mask_size.setter
    def mask_size(self, mask_size: int | tuple[int, int]) -> None:
        """
        Mask size setter.

        Args:
            mask_size (int | tuple[int, int]): Mask size

        Raises:
            ValueError: If the mask size is less than 28
        """

        if isinstance(mask_size, int):
            mask_size = (mask_size, mask_size)

        width, height = mask_size

        if width < 28 or height < 28:
            raise ValueError('Mask size must be at least 28x28')

        self.__mask_size = mask_size

    @property
    def mask_ratio(self) -> tuple[float, float]:
        """
        Mask ratio property.

        Returns:
            tuple[float, float]: Mask ratio
        """

        return self.__ratio

    @mask_ratio.setter
    def mask_ratio(self, ratio: tuple[float, float]) -> None:
        """
        Mask ratio setter.

        Args:
            ratio (tuple[float, float]): Mask ratio

        Raises:
            ValueError: If the ratio is not in the interval [0, 100]
            ValueError: If the ratio min is greater than the ratio max
        """

        ratio_min, ratio_max = ratio

        if ratio_min < 0 or ratio_max < 0:
            raise ValueError('Mask ratio must be positive')
        if ratio_min > ratio_max:
            raise ValueError('Ratio min must be less than ratio max')
        if ratio_min > 100 or ratio_max > 100:
            raise ValueError('Mask ratio must be less than 100')

        self.__ratio = ratio

    @property
    def width(self) -> int:
        """
        Read only property for the mask width.

        Returns:
            int: Mask width
        """

        return self.__width

    @width.setter
    def width(self, value: int) -> None:
        """
        Setter for the mask width.

        Args:
            value (int): Mask width

        Raises:
            ValueError: If the value is negative
            ValueError: If the value is less than 28
        """

        if value <= 0:
            raise ValueError("Width value must be positive")
        if value < 28:
            raise ValueError("Width value must be greater than 28")
        self.__width = value

    @property
    def height(self) -> int:
        """
        Read only property for the mask height.

        Returns:
            int: Mask height
        """

        return self.__height

    @height.setter
    def height(self, value: int) -> None:
        """
        Setter for the mask height.

        Args:
            value (int): Mask height

        Raises:
            ValueError: If the value is negative
            ValueError: If the value is less than 28
        """

        if value <= 0:
            raise ValueError("Height value must be positive")
        if value < 28:
            raise ValueError("Height value must be greater than 28")
        self.__height = value

    @property
    def ratio_min(self) -> float:
        """
        Read only property for the minimum of the ratio.

        Returns:
            float: Minimum ratio
        """

        return self.__ratio_min

    @ratio_min.setter
    def ratio_min(self, value: float) -> None:
        """
        Setter for the minimum of the ratio.

        Args:
            value (float): Minimum ratio

        Raises:
            ValueError: If the value is negative
        """

        if value < 0:
            raise ValueError("Minimum ratio value must be positive")
        self.__ratio_min = value

    @property
    def ratio_max(self) -> float:
        """
        Read only property for the maximum of the ratio.

        Returns:
            float: Maximum ratio
        """

        return self.__ratio_max

    @ratio_max.setter
    def ratio_max(self, value: float) -> None:
        """
        Setter for the maximum of the ratio.

        Args:
            value (float): Maximum ratio

        Raises:
            ValueError: If the value is negative
            ValueError: If the value is higher than 100
        """

        if value < 0:
            raise ValueError("Maximum ratio value must be positive")
        if value > 100:
            raise ValueError("Maximum ratio value must be lower than 100")
        self.__ratio_max = value

    @property
    def draw_scale(self) -> float:
        """
        Read only property for the drawing objects scaling factor.

        Returns:
            float: Drawing objects scaling factor
        """

        return self.__draw_scale

    @draw_scale.setter
    def draw_scale(self, value: float) -> None:
        """
        Setter for the drawing objects scaling factor.

        Args:
            value (float): Drawing objects scaling factor

        Raises:
            ValueError: If the value is negative
        """

        if value <= 0:
            raise ValueError("Draw value scale must be positive")
        self.__draw_scale = value

    @property
    def random_generator(self) -> random.SystemRandom:
        """
        Read only property for the random generator.

        Returns:
            random.SystemRandom: Random generator
        """

        return self.__random_generator

    @random_generator.setter
    def random_generator(self, value: random.SystemRandom) -> None:
        """
        Setter for the random generator.

        Args:
            value (random.SystemRandom): Random generator
        """

        self.__random_generator = value

    def __generate_uniform_number(self, start: float, stop: float) -> float:
        """
        Generate a radnom number between start and stop from a uniform
        distribution.

        Args:
            start (float): Start value
            stop (float): Stop value

        Returns:
            float: Random number
        """

        self.random_generator.seed(int(os.urandom(4).hex(), 16))
        result = self.random_generator.uniform(start, stop)
        return result

    @classmethod
    def __compute_mask_ratio(cls, mask: np.ndarray) -> float:
        """
        Compute the hole-to-image ratio of the mask.

        Args:
            mask (np.ndarray): Mask to compute the ratio

        Returns:
            float: Ratio of the mask
        """

        mask_size = float(mask.size)
        result = np.count_nonzero(mask) / mask_size
        return 100.0 * result

    def __draw_line(self, mask: np.ndarray, size: int, height: int, width: int, color: MaskColor = MaskColor.WHITE) -> None:
        """
        Draws random lines on the mask.

        Args:
            mask (np.ndarray): Mask
            size (int): Line size
            height (int): Mask height
            width (int): Mask width
        """

        if self.__generate_uniform_number(0, 1) > 0.5:
            start_x, stop_x = np.random.randint(0, width, 2, dtype=np.uint16)
            start_y, stop_y = np.random.randint(0, height, 2, dtype=np.uint16)
            thickness = random.randint(1, size)
            cv.line(mask, (start_x, start_y), (stop_x, stop_y), color.value, thickness)

    def __draw_circle(self, mask: np.ndarray, size: int, height: int, width: int, color: MaskColor = MaskColor.WHITE) -> None:
        """
        Draws random circles on the mask.

        Args:
            mask (np.ndarray): Mask
            size (int): Circle size
            height (int): Mask height
            width (int): Mask width
        """

        if self.__generate_uniform_number(0, 1) > 0.5:
            center_x = random.randint(0, width)
            center_y = random.randint(0, height)
            radius = random.randint(1, size)
            cv.circle(mask, (center_x, center_y), radius, color.value, -1)

    def __draw_ellipse(self, mask: np.ndarray, size: int, height: int, width: int, color: MaskColor = MaskColor.WHITE) -> None:
        """
        Draws random ellipses on the mask.

        Args:
            mask (np.ndarray): Mask
            size (int): Ellipse size
            height (int): Mask height
            width (int): Mask width
        """

        if self.__generate_uniform_number(0, 1) > 0.5:
            center_x = random.randint(0, width)
            center_y = random.randint(0, height)
            axis1 = random.randint(1, size)
            axis2 = random.randint(1, size)
            rotation_angle = random.randint(0, 360)
            start_arc_angle = random.randint(0, 180)
            stop_arc_angle = random.randint(0, 180)
            thickness = random.randint(1, size)
            cv.ellipse(mask, (center_x, center_y), (axis1, axis2), rotation_angle,
                       start_arc_angle, stop_arc_angle, color.value, thickness)

    def __generate_mask_helper(self, height: int, width: int, ratio_min: float, ratio_max: float, draw_scale: float) -> tuple[np.ndarray, int]:
        """
        Helper method for the mask generation that can be called from the
        generator or from the class itself.

        Args:
            height (int): Mask height
            width (int): Mask width
            ratio_min (float): Minimum ratio
            ratio_max (float): Maximum ratio
            draw_scale (float): Drawing objects scaling factor

        Returns:
            tuple(np.ndarray, int): Mask and the number of pixels in the mask
        """

        mask = np.zeros((height, width), dtype=np.uint8)

        size = float(width + height) * draw_scale
        size = int(size) if size > 1 else 1

        ratio_min = float(ratio_min)
        ratio_max = float(ratio_max)

        while True:
            if self.__compute_mask_ratio(mask) > ratio_min:
                break
            self.__draw_line(mask, size, height, width, MaskColor.WHITE)
            if self.__compute_mask_ratio(mask) > ratio_min:
                break
            self.__draw_circle(mask, size, height, width, MaskColor.WHITE)
            if self.__compute_mask_ratio(mask) > ratio_min:
                break
            self.__draw_ellipse(mask, size, height, width, MaskColor.WHITE)

        while True:
            if self.__compute_mask_ratio(mask) < ratio_max:
                break
            self.__draw_line(mask, size, height, width, MaskColor.BLACK)
            if self.__compute_mask_ratio(mask) < ratio_max:
                break
            self.__draw_circle(mask, size, height, width, MaskColor.BLACK)
            if self.__compute_mask_ratio(mask) < ratio_max:
                break
            self.__draw_ellipse(mask, size, height, width, MaskColor.BLACK)

        return mask, np.count_nonzero(mask == MaskGenerator.MASK_VALUE)

    def __generate_mask(self) -> tuple[np.ndarray, int]:
        """
        Generates a random mask and the number of pixels in the mask.

        Returns:
            tuple(np.ndarray, int): Mask and the number of pixels in the mask
        """

        return self.__generate_mask_helper(self.height, self.width, self.ratio_min, self.ratio_max, self.draw_scale)

    def generate(self) -> tuple[np.ndarray, int]:
        """
        Generates a random mask and the number of pixels in the mask.

        Returns:
            tuple(np.ndarray, int): Mask and the number of pixels in the mask
        """

        return self.__generate_mask()

    def __call__(self) -> Iterable[tuple[np.ndarray, int]]:
        """
        Call method for the MaskGenerator class.

        Yields:
            tuple(np.ndarray, int): Mask and the number of pixels in the mask
        """

        while True:
            yield self.__generate_mask()
