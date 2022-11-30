from __future__ import annotations

import json
from enum import Enum, unique
from typing import Any

import numpy as np


@unique
class InpaintingMethod(Enum):
    """
    The inpainting method to use.
    """
    PATCH_MATCH: str = 'patch_match'
    NAVIER_STOKES: str = 'navier_stokes'
    TELEA: str = 'telea'


@unique
class MaskColor(Enum):
    """
    The color of the mask.
    """
    WHITE: int = 255
    BLACK: int = 0


class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for numpy arrays.
    """

    def default(self, obj: Any) -> Any | list | int | float:
        """
        Encode numpy data types to JSON compatible types.

        Args:
            obj (Any:): The object to encode

        Returns:
            Any | list | int | float: The encoded object
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return json.JSONEncoder.default(self, obj)
