from __future__ import annotations

import random
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
    DEFAULT_MASK_RATIO: Final[tuple[float, float]] = (5, 10)

    def __init__(self, mask_size: tuple[int, int], count: int = 0, ratio: tuple[float, float] = DEFAULT_MASK_RATIO,
                 draw_scale: float = DEFAULT_DRAW_SCALE) -> None:
        """
        Initializes the generator.

        Args:
            mask_size (tuple[int, int]): Mask dimensions
            count (int, optional): Numbers of masks to generate. Defaults to 0
            ratio (tuple[float, float]): Percentage interval of the image that is covered by the mask. Defaults to DEFAULT_MASK_RATIO
            draw_scale (float, optional): Drawing objects scaling factor. Defaults to DEFAULT_DRAW_SCALE
        """
        self.width, self.height = mask_size
        self.ratio_min, self.ratio_max = ratio
        self.count = count
        self.draw_scale = draw_scale
        self.current = 0

    @property
    def width(self) -> int:
        """
        Read only property for the mask width.

        Returns:
            int: Mask width
        """
        return self.__width

    @property
    def height(self) -> int:
        """
        Read only property for the mask height.

        Returns:
            int: Mask height
        """
        return self.__height

    @property
    def ratio_min(self) -> float:
        """
        Read only property for the minimum of the ratio.

        Returns:
            float: Minimum ratio
        """
        return self.__ratio_min

    @property
    def ratio_max(self) -> float:
        """
        Read only property for the maximum of the ratio.

        Returns:
            float: Maximum ratio
        """
        return self.__ratio_max

    @property
    def count(self) -> int:
        """
        Read only property for the number of masks to generate.

        Returns:
            int: Number of masks to generate
        """
        return self.__count

    @property
    def draw_scale(self) -> float:
        """
        Read only property for the drawing objects scaling factor.

        Returns:
            float: Drawing objects scaling factor
        """
        return self.__draw_scale

    @property
    def current(self) -> int:
        """
        Read only property for the current number of generated masks.

        Returns:
            int: Current number of generated masks
        """
        return self.__current

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
        elif value < 28:
            raise ValueError("Width value must be greater than 28")
        self.__width = value

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
        elif value < 28:
            raise ValueError("Height value must be greater than 28")
        self.__height = value

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

    @count.setter
    def count(self, value: int) -> None:
        """
        Setter for the number of masks to generate.

        Args:
            value (int): Number of masks to generate

        Raises:
            ValueError: If the value is negative
        """
        if value < 0:
            raise ValueError("Count value must be positive")
        self.__count = value

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

    @current.setter
    def current(self, value: int) -> None:
        """
        Setter for the current number of generated masks.

        Args:
            value (int): Current number of generated masks

        Raises:
            ValueError: If the value is negative
        """
        if value < 0:
            raise ValueError("Current value must be positive")
        self.__current = value

    @classmethod
    def __compute_mask_ratio(cls, mask: np.ndarray) -> float:
        """
        Compute the hole-to-image ratio of the mask.

        Args:
            mask (np.ndarray): Mask to compute the ratio

        Returns:
            float: Ratio of the mask
        """
        return 100.0 * np.count_nonzero(mask) / mask.size

    @classmethod
    def __generate_mask_helper(cls, height: int, width: int, ratio_min: float, ratio_max: float, draw_scale: float) -> tuple[np.ndarray, int]:
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
        size = int(size + 1e-6)

        ratio_min = float(ratio_min)
        ratio_max = float(ratio_max)

        while True:
            if cls.__compute_mask_ratio(mask) > ratio_min:
                break
            cls.__draw_line(mask, size, height, width, MaskColor.WHITE)
            if cls.__compute_mask_ratio(mask) > ratio_min:
                break
            cls.__draw_circle(mask, size, height, width, MaskColor.WHITE)
            if cls.__compute_mask_ratio(mask) > ratio_min:
                break
            cls.__draw_ellipse(mask, size, height, width, MaskColor.WHITE)

        while True:
            if cls.__compute_mask_ratio(mask) < ratio_max:
                break
            cls.__draw_line(mask, size, height, width, MaskColor.BLACK)
            if cls.__compute_mask_ratio(mask) < ratio_max:
                break
            cls.__draw_circle(mask, size, height, width, MaskColor.BLACK)
            if cls.__compute_mask_ratio(mask) < ratio_max:
                break
            cls.__draw_ellipse(mask, size, height, width, MaskColor.BLACK)

        return mask, np.count_nonzero(mask == MaskGenerator.MASK_VALUE)

    def __generate_mask(self) -> tuple[np.ndarray, int]:
        """
        Generates a random mask.

        Returns:
            np.ndarray: Mask
        """
        return self.__generate_mask_helper(self.height, self.width, self.ratio_min, self.ratio_max, self.draw_scale)

    @classmethod
    def __draw_line(cls, mask: np.ndarray, size: int, height: int, width: int, color: MaskColor = MaskColor.WHITE) -> None:
        """
        Draws random lines on the mask.

        Args:
            mask (np.ndarray): Mask
            size (int): Line size
            height (int): Mask height
            width (int): Mask width
        """
        if random.uniform(0, 1) > 0.5:
            start_x, stop_x = np.random.randint(0, width, 2, dtype=np.uint16)
            start_y, stop_y = np.random.randint(0, height, 2, dtype=np.uint16)
            thickness = random.randint(1, size)
            cv.line(mask, (start_x, start_y), (stop_x, stop_y), color.value, thickness)

    @classmethod
    def __draw_circle(cls, mask: np.ndarray, size: int, height: int, width: int, color: MaskColor = MaskColor.WHITE) -> None:
        """
        Draws random circles on the mask.

        Args:
            mask (np.ndarray): Mask
            size (int): Circle size
            height (int): Mask height
            width (int): Mask width
        """
        if random.uniform(0, 1) > 0.5:
            center_x = random.randint(0, width)
            center_y = random.randint(0, height)
            radius = random.randint(1, size)
            cv.circle(mask, (center_x, center_y), radius, color.value, -1)

    @classmethod
    def __draw_ellipse(cls, mask: np.ndarray, size: int, height: int, width: int, color: MaskColor = MaskColor.WHITE) -> None:
        """
        Draws random ellipses on the mask.

        Args:
            mask (np.ndarray): Mask
            size (int): Ellipse size
            height (int): Mask height
            width (int): Mask width
        """
        if random.uniform(0, 1) > 0.5:
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

    @classmethod
    def generate_mask(cls, mask_size: tuple[int, int], ratio: tuple[float, float] = DEFAULT_MASK_RATIO,
                      draw_scale: float = DEFAULT_DRAW_SCALE) -> tuple[np.ndarray, int]:
        """
        Generates a single random mask.

        Args:
            mask_size (tuple[int, int]): Mask dimensions
            ratio (tuple[float, float]): Percentage interval of the image that is covered by the mask. Defaults to (15, 20).
            draw_scale (float, optional): Drawing objects scaling factor. Defaults to DEFAULT_DRAW_SCALE

        Returns:
            tuple(np.ndarray, int): Mask and number of pixels in the mask
        """
        return cls.__generate_mask_helper(mask_size[0], mask_size[1], ratio[0], ratio[1], draw_scale)

    def __iter__(self) -> MaskGenerator:
        """
        Iterator for the MaskGenerator class.

        Returns:
            MaskGenerator: MaskGenerator instance
        """
        return self

    def __next__(self) -> np.ndarray:
        """
        Generates the next mask.

        Raises:
            StopIteration: If all the masks have been generated

        Returns:
            np.ndarray: Mask
        """
        if self.__current < self.count:
            self.__current += 1
            return self.__generate_mask()[0]
        else:
            raise StopIteration
