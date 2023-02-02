from __future__ import annotations

import multiprocessing as mp
import subprocess as sp
import time
import traceback
from pathlib import Path
from typing import Final

import numpy as np
from colorama import Fore, Style

from mask_generator import MaskGenerator

IMAGE_SIZE: Final[int] = 64
MASK_RATIO: Final[tuple[float, float]] = (5, 10)


def print_path(path: Path, mask_and_size: tuple[np.ndarray, int]) -> None:
    mask, no_pixels = mask_and_size
    print(Fore.GREEN + F'Processing {path.name} with {no_pixels} pixels mask' + Style.RESET_ALL, flush=True)


def run() -> None:
    path_match_path = Path('..', 'patch_match', 'PatchMatch').resolve()
    images_path = Path('..', 'images', 'original', 'test').resolve()
    mask_generator = MaskGenerator(IMAGE_SIZE, MASK_RATIO)

    try:
        sp.run(path_match_path, check=True)
    except sp.CalledProcessError:
        print(Fore.RED + 'PatchMatch failed!' + Style.RESET_ALL)

    with mp.Pool() as processes_pool:
        processes_pool.starmap(print_path, zip(images_path.iterdir(), mask_generator()), chunksize=100)


def main() -> None:
    print(Fore.MAGENTA + 'Starting script...' + Style.RESET_ALL)

    start_time = time.perf_counter()
    try:
        run()

    except KeyboardInterrupt:

        print(Fore.RED + '\nScript interrupted by the user!' + Style.RESET_ALL)

    except Exception:

        print(Fore.RED)
        print('Script failed with the following error:')
        traceback.print_exc()
        print(Style.RESET_ALL)

    end_time = time.perf_counter()

    elapsed = end_time - start_time
    elapsed_time = time.strftime('%H:%M:%S', time.gmtime(elapsed))

    print(Fore.YELLOW + F'Total time: {elapsed_time}' + Style.RESET_ALL)
    print(Fore.MAGENTA + 'Everything done!' + Style.RESET_ALL)


if __name__ == '__main__':
    main()
