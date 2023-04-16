from __future__ import annotations

import json
import time
import traceback
from enum import Enum, unique
from typing import Any, Callable

import numpy as np
import tensorflow as tf
from colorama import Fore, Style
import multiprocessing as mp


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

    def default(self, o: Any) -> Any | list | int | float:
        """
        Encode numpy data types to JSON compatible types.

        Args:
            o (Any:): The object to encode

        Returns:
            Any | list | int | float: The encoded object
        """

        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        return json.JSONEncoder.default(self, o)


def time_running(run: Callable[[], None]) -> None:
    print(Fore.MAGENTA + 'Starting script...' + Style.RESET_ALL, flush=True)

    start_time = time.perf_counter()
    try:

        run()

    except KeyboardInterrupt:

        print(Fore.RED + '\nScript interrupted by the user!' + Style.RESET_ALL, flush=True)

    except Exception:

        print(Fore.RED, flush=True)
        print('Script failed with the following error:', flush=True)
        traceback.print_exc()
        print(Style.RESET_ALL, flush=True)

    end_time = time.perf_counter()

    elapsed = end_time - start_time
    elapsed_time = time.strftime('%H:%M:%S', time.gmtime(elapsed))

    print(Fore.YELLOW + F'Total time: {elapsed_time}' + Style.RESET_ALL, flush=True)
    print(Fore.MAGENTA + 'Everything done!' + Style.RESET_ALL, flush=True)


def time_running_gpu(run: Callable[[], None]) -> None:
    def internal_run() -> None:
        gpus = tf.config.list_logical_devices('GPU')
        strategy = tf.distribute.MirroredStrategy(gpus)
        with strategy.scope():
            run()

    time_running(internal_run)


def time_running_cpu(run: Callable[[], None]) -> None:
    def internal_run() -> None:
        mp.set_start_method('fork')
        run()

    time_running(internal_run)
