from __future__ import annotations

import os
import random
from collections.abc import Iterable
from enum import Enum
from typing import Final, Literal, TypeAlias, List, Tuple, Any

import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw

from utils import MaskColor


class MaskImpl(Enum):
    """
    Enum that represents the different implementations of the mask generation.
    """

    RANDOM_SHAPES = 'random_shapes'
    FREEFORM = 'freeform'


MaskImplType: TypeAlias = Literal[
    MaskImpl.RANDOM_SHAPES,
    MaskImpl.FREEFORM
]


class MaskGenerator:
    """
    Generator that yields random masks. The white part represents the area of interest.
    """

    __MASK_VALUE: Final = MaskColor.WHITE.value

    __DEFAULT_MASK_RATIO: Final[tuple[float, float]] = (25, 30)
    __DEFAULT_IMPLEMENTATION: Final[MaskImplType] = MaskImpl.FREEFORM

    __DEFAULT_RANDOM_SHAPES_DRAW_SCALE: Final[float] = 0.015

    __DEFAULT_FREEFORM_MIN_VERTICES: Final[int] = 1
    __DEFAULT_FREEFORM_MAX_VERTICES: Final[int] = 10
    __DEFAULT_FREEFORM_MEAN_ANGLE: Final[float] = 2 * np.pi / 5
    __DEFAULT_FREEFORM_ANGLE_RANGE: Final[float] = 2 * np.pi / 15

    def __init__(self, mask_size: int | tuple[int, int], ratio: tuple[float, float] = __DEFAULT_MASK_RATIO,
                 implementation: MaskImplType = __DEFAULT_IMPLEMENTATION) -> None:
        """
        Initializes the generator.

        Args:
            mask_size (tuple[int, int]): Mask dimensions
            ratio (tuple[float, float]): Percentage interval of the image that is covered by the mask.
                                         Defaults to __DEFAULT_MASK_RATIO
            implementation (MaskImplType): Mask implementation. Defaults to __DEFAULT_IMPLEMENTATION
        """

        self.mask_size = mask_size  # type: ignore
        self.mask_ratio = ratio
        self.width, self.height = self.mask_size
        self.ratio_min, self.ratio_max = self.mask_ratio
        self.implementation = implementation

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
    def implementation(self) -> MaskImplType:
        """
        Mask implementation property.

        Returns:
            MaskImplType: Mask implementation
        """

        return self.__implementation

    @implementation.setter
    def implementation(self, implementation: MaskImplType) -> None:
        """
        Mask implementation setter.

        Args:
            implementation (MaskImplType): Mask implementation
        """

        self.__implementation = implementation

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

    def __generate_gaussian_number(self, mean: float, std: float) -> float:
        """
        Generate a random number from a gaussian distribution.

        Args:
            mean (float): Mean value
            std (float): Standard deviation

        Returns:
            float: Random number
        """

        self.random_generator.seed(int(os.urandom(4).hex(), 16))
        result = self.random_generator.gauss(mean, std)
        return result

    def __generate_random_int(self, start: int, stop: int) -> int:
        """
        Generate a random integer between start and stop.

        Args:
            start (int): Start value
            stop (int): Stop value

        Returns:
            int: Random integer
        """

        self.random_generator.seed(int(os.urandom(4).hex(), 16))
        result = self.random_generator.randint(start, stop)
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
        result = np.count_nonzero(mask == MaskGenerator.__MASK_VALUE) / mask_size
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

    def __generate_random_shapes(self, height: int, width: int, ratio_min: float, ratio_max: float) -> tuple[np.ndarray, int]:
        """
        Generate a random mask with random shapes.

        Args:
            height (int): Mask height
            width (int): Mask width
            ratio_min (float): Minimum ratio
            ratio_max (float): Maximum ratio

        Returns:
            tuple(np.ndarray, int): Mask and the number of pixels in the mask
        """

        mask = np.zeros((height, width), dtype=np.uint8)

        size = float(width + height) * self.__DEFAULT_RANDOM_SHAPES_DRAW_SCALE
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

        return mask, np.count_nonzero(mask == MaskGenerator.__MASK_VALUE)

    def __generate_freeform(self, height: int, width: int, ratio_min: float, ratio_max: float) -> tuple[np.ndarray, int]:
        """
        Generate a random mask with freeform shapes.

        Args:
            height (int): Mask height
            width (int): Mask width
            ratio_min (float): Minimum ratio
            ratio_max (float): Maximum ratio

        Returns:
            tuple(np.ndarray, int): Mask and the number of pixels in the mask
        """

        mask = np.zeros((height, width), dtype=np.uint8)

        average_radius: Final[float] = np.hypot(height, width) / 8
        mid_width: Final[float] = np.sqrt(height * width) / 10
        max_width: Final[float] = np.sqrt(height * width) / 5

        target_percent = self.__generate_uniform_number(ratio_min, ratio_max)
        target_area = (height * width) * (target_percent / 100)

        current_area = 0

        while current_area < target_area:
            mask = Image.fromarray(mask)

            num_vertex = random.randint(self.__DEFAULT_FREEFORM_MIN_VERTICES, self.__DEFAULT_FREEFORM_MAX_VERTICES)
            angle_min = self.__DEFAULT_FREEFORM_MEAN_ANGLE - self.__generate_uniform_number(0, self.__DEFAULT_FREEFORM_ANGLE_RANGE)
            angle_max = self.__DEFAULT_FREEFORM_MEAN_ANGLE + self.__generate_uniform_number(0, self.__DEFAULT_FREEFORM_ANGLE_RANGE)

            angles: list[float] = []

            for idx in range(num_vertex):
                if idx % 2 == 0:
                    angles.append(2 * np.pi - self.__generate_uniform_number(angle_min, angle_max))
                else:
                    angles.append(self.__generate_uniform_number(angle_min, angle_max))

            vertices: list[tuple[int, int]] = []

            start_x = self.__generate_random_int(0, width)
            start_y = self.__generate_random_int(0, height)

            vertices.append((start_y, start_x))

            for idx in range(num_vertex):
                radius = np.clip(self.__generate_gaussian_number(average_radius, average_radius // 2), 0, 2 * average_radius)
                new_x = np.clip(vertices[-1][0] + radius * np.cos(angles[idx]), 0, width)
                new_y = np.clip(vertices[-1][1] + radius * np.sin(angles[idx]), 0, height)
                vertices.append((int(new_x), int(new_y)))

            width = int(self.__generate_uniform_number(mid_width, max_width))

            draw = ImageDraw.Draw(mask)

            draw.line(vertices, fill=MaskGenerator.__MASK_VALUE, width=width)

            for elem_x, elem_y in vertices:
                start_x = elem_x - width // 2
                start_y = elem_y - width // 2
                end_x = elem_x + width // 2
                end_y = elem_y + width // 2

                draw.ellipse((start_x, start_y, end_x, end_y), fill=MaskGenerator.__MASK_VALUE)

            mask = np.asarray(mask)
            current_area = np.count_nonzero(mask == MaskGenerator.__MASK_VALUE)

        return mask, current_area

    def generate(self) -> tuple[np.ndarray, int]:
        """
        Generates a random mask and the number of pixels in the mask.

        Returns:
            tuple(np.ndarray, int): Mask and the number of pixels in the mask
        """

        if self.implementation == MaskImpl.FREEFORM:
            return self.__generate_freeform(self.height, self.width, self.ratio_min, self.ratio_max)

        return self.__generate_random_shapes(self.height, self.width, self.ratio_min, self.ratio_max)

    def __call__(self) -> Iterable[tuple[np.ndarray, int]]:
        """
        Call method for the MaskGenerator class.

        Yields:
            tuple(np.ndarray, int): Mask and the number of pixels in the mask
        """

        if self.implementation == MaskImpl.FREEFORM:
            while True:
                yield self.__generate_freeform(self.height, self.width, self.ratio_min, self.ratio_max)
        else:
            while True:
                yield self.__generate_random_shapes(self.height, self.width, self.ratio_min, self.ratio_max)
