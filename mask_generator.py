from random import randint
import numpy as np
import cv2 as cv


class MaskGenerator:

    def __init__(self, mask_size: tuple[int, int], count: int = 0, draw_scale: float = 0.025, max_objects: int = 20, seed: int = 42):
        self.width, self.height = mask_size
        self.count = count
        self.draw_scale = draw_scale
        self.max_objects = max_objects
        self.current = 0

        np.random.seed(seed)

    @property
    def width(self):
        return self.__width

    @property
    def height(self):
        return self.__height

    @property
    def count(self):
        return self.__count

    @property
    def draw_scale(self):
        return self.__draw_scale

    @property
    def max_objects(self):
        return self.__max_objects

    @property
    def current(self):
        return self.__current

    @width.setter
    def width(self, value: int):
        if value <= 0:
            raise ValueError("Width value must be positive")
        elif value < 28:
            raise ValueError("Width value must be greater than 28")
        self.__width = value

    @height.setter
    def height(self, value: int):
        if value <= 0:
            raise ValueError("Height value must be positive")
        elif value < 28:
            raise ValueError("Height value must be greater than 28")
        self.__height = value

    @count.setter
    def count(self, value: int):
        if value < 0:
            raise ValueError("Count value must be positive")
        self.__count = value

    @draw_scale.setter
    def draw_scale(self, value: float):
        if value <= 0:
            raise ValueError("Draw value scale must be positive")
        self.__draw_scale = value

    @max_objects.setter
    def max_objects(self, value: int):
        if value <= 0:
            raise ValueError("Max value objects must be positive")
        elif value > 20:
            raise ValueError("Max value objects must be less than 20")
        self.__max_objects = value

    @current.setter
    def current(self, value: int):
        if value < 0:
            raise ValueError("Current value must be positive")
        self.__current = value

    def __generate_mask(self):
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        size = int((self.width + self.height) * self.draw_scale)

        # Draw random lines
        for _ in range(randint(1, self.max_objects)):
            start_x, stop_x = np.random.randint(0, self.width, 2, dtype=np.uint16)
            start_y, stop_y = np.random.randint(0, self.height, 2, dtype=np.uint16)
            thickness = randint(1, size)
            cv.line(mask, (start_x, start_y), (stop_x, stop_y), 255, thickness)

        # Draw random circles
        for _ in range(randint(1, self.max_objects)):
            center_x = randint(0, self.width)
            center_y = randint(0, self.height)
            radius = randint(1, size)
            cv.circle(mask, (center_x, center_y), radius, 255, -1)

        # Draw random ellipses
        for _ in range(randint(1, self.max_objects)):
            center_x = randint(0, self.width)
            center_y = randint(0, self.height)
            axis1 = randint(1, size)
            axis2 = randint(1, size)
            rotation_angle = randint(0, 360)
            start_arc_angle = randint(0, 180)
            stop_arc_angle = randint(0, 180)
            thickness = randint(1, size)
            cv.ellipse(mask, (center_x, center_y), (axis1, axis2), rotation_angle, start_arc_angle, stop_arc_angle, 255, thickness)

        return mask

    def __iter__(self):
        return self

    def __next__(self):
        if self.__current < self.count:
            self.__current += 1
            return self.__generate_mask()
        else:
            raise StopIteration
