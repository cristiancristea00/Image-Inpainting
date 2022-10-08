from __future__ import annotations

from typing import final

import cv2 as cv
import numpy as np
import random


class MaskGenerator:
    """
    Generator that yields random masks by drawing lines, circles and ellipses.
    The white part represents the area of interest.
    """
    DEFAULT_MAX_OBJECTS: final = 15
    DEFAULT_DRAW_SCALE: final = 0.015

    MAX_MASK_VALUE: final = 255

    def __init__(self, mask_size: tuple[int, int], count: int = 0, draw_scale: float = DEFAULT_DRAW_SCALE,
                 max_objects: int = DEFAULT_MAX_OBJECTS) -> None:
        """
        Initializes the generator.

        Args:
            mask_size (tuple[int, int]): Mask dimensions
            count (int, optional): Numbers of masks to generate. Defaults to 0
            draw_scale (float, optional): Drawing objects scaling factor. Defaults to DEFAULT_DRAW_SCALE
            max_objects (int, optional): Upper limit of every type of drawing. Defaults to DEFAULT_MAX_OBJECTS
        """
        self.width, self.height = mask_size
        self.count = count
        self.draw_scale = draw_scale
        self.max_objects = max_objects
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
    def max_objects(self) -> int:
        """
        Read only property for the upper limit of every type of drawing.

        Returns:
            int: Upper limit of every type of drawing
        """
        return self.__max_objects

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

    @max_objects.setter
    def max_objects(self, value: int) -> None:
        """
        Setter for the upper limit of every type of drawing.

        Args:
            value (int): Upper limit of every type of drawing

        Raises:
            ValueError: If the value is negative
            ValueError: If the value is higher than 20
        """
        if value <= 0:
            raise ValueError("Max value objects must be positive")
        elif value > 20:
            raise ValueError("Max value objects must be less than 20")
        self.__max_objects = value

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
    def __generate_mask_helper(cls, height: int, width: int, draw_scale: float, max_objects: int) -> np.ndarray:
        """
        Helper method for the mask generation that can be called from the
        generator or from the class itself.

        Args:
            height (int): Mask height
            width (int): Mask width
            draw_scale (float): Drawing objects scaling factor
            max_objects (int): Upper limit of every type of drawing

        Returns:
            np.ndarray: Mask
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        size = int((width + height) * draw_scale)

        cls.__draw_lines(mask, size, height, width, max_objects)
        cls.__draw_circles(mask, size, height, width, max_objects)
        cls.__draw_ellipses(mask, size, height, width, max_objects)

        return mask

    def __generate_mask(self) -> np.ndarray:
        """
        Generates a random mask.

        Returns:
            np.ndarray: Mask
        """
        return self.__generate_mask_helper(self.height, self.width, self.draw_scale, self.max_objects)

    @classmethod
    def __draw_lines(cls, mask: np.ndarray, size: int, height: int, width: int, max_objects: int) -> None:
        """
        Draws random lines on the mask.

        Args:
            mask (np.ndarray): Mask
            size (int): Line size
            height (int): Mask height
            width (int): Mask width
            max_objects (int): Upper limit of every type of drawing
        """
        for _ in range(random.randint(1, max_objects)):
            start_x, stop_x = np.random.randint(0, width, 2, dtype=np.uint16)
            start_y, stop_y = np.random.randint(0, height, 2, dtype=np.uint16)
            thickness = random.randint(1, size)
            cv.line(mask, (start_x, start_y), (stop_x, stop_y), 255, thickness)

    @classmethod
    def __draw_circles(cls, mask: np.ndarray, size: int, height: int, width: int, max_objects: int) -> None:
        """
        Draws random circles on the mask.

        Args:
            mask (np.ndarray): Mask
            size (int): Circle size
            height (int): Mask height
            width (int): Mask width
            max_objects (int): Upper limit of every type of drawing
        """
        for _ in range(random.randint(1, max_objects)):
            center_x = random.randint(0, width)
            center_y = random.randint(0, height)
            radius = random.randint(1, size)
            cv.circle(mask, (center_x, center_y), radius, 255, -1)

    @classmethod
    def __draw_ellipses(cls, mask: np.ndarray, size: int, height: int, width: int, max_objects: int) -> None:
        """
        Draws random ellipses on the mask.

        Args:
            mask (np.ndarray): Mask
            size (int): Ellipse size
            height (int): Mask height
            width (int): Mask width
            max_objects (int): Upper limit of every type of drawing
        """
        for _ in range(random.randint(1, max_objects)):
            center_x = random.randint(0, width)
            center_y = random.randint(0, height)
            axis1 = random.randint(1, size)
            axis2 = random.randint(1, size)
            rotation_angle = random.randint(0, 360)
            start_arc_angle = random.randint(0, 180)
            stop_arc_angle = random.randint(0, 180)
            thickness = random.randint(1, size)
            cv.ellipse(mask, (center_x, center_y), (axis1, axis2), rotation_angle,
                       start_arc_angle, stop_arc_angle, 255, thickness)

    @classmethod
    def generate_mask(cls, mask_size: tuple[int, int], draw_scale: float = DEFAULT_DRAW_SCALE, max_objects: int = DEFAULT_MAX_OBJECTS) -> np.ndarray:
        """
        Generates a single random mask.

        Args:
            mask_size (tuple[int, int]): Mask dimensions.
            draw_scale (float, optional): Drawing objects scaling factor. Defaults to DEFAULT_DRAW_SCALE
            max_objects (int, optional): Upper limit of every type of drawing. Defaults to DEFAULT_MAX_OBJECTS

        Returns:
            np.ndarray: Mask
        """
        return cls.__generate_mask_helper(mask_size[0], mask_size[1], draw_scale, max_objects)

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
            return self.__generate_mask()
        else:
            raise StopIteration
