from __future__ import annotations

import multiprocessing as mp
import subprocess as sp
import time
import traceback
from functools import partial
from pathlib import Path
from typing import Final

import cv2 as cv
import tensorflow as tf
from colorama import Fore, Style

from image_processor import ImageProcessor
from mask_generator import MaskGenerator

IMAGE_SIZE: Final[int] = 128
MASK_RATIO: Final = ((5, 10), (15, 20), (25, 30), (35, 40), (45, 50))

MASKED_PATH: Final[Path] = Path('..', 'images', 'masked').resolve()
MASKS_PATH: Final[Path] = Path('..', 'images', 'masks').resolve()
OUTPUT_PATH: Final[Path] = Path('..', 'images', 'patch_match').resolve()

PATCH_MATCH_PATH: Final[Path] = Path('..', 'PatchMatch').resolve()

MASK_GENERATOR: MaskGenerator = MaskGenerator(IMAGE_SIZE, (1, 1))
IMAGE_PROCESSOR: ImageProcessor = ImageProcessor(MASK_GENERATOR)


def mask_image(path: Path, mask_ratio: tuple[int, int]) -> None:
    path_name = path.with_suffix('.png').name

    masked_path = MASKED_PATH / str(mask_ratio) / path_name
    mask_path = MASKS_PATH / str(mask_ratio) / path_name
    output_path = OUTPUT_PATH / str(mask_ratio) / path_name

    image = cv.imread(str(path))
    cropped = tf.image.random_crop(image, size=(IMAGE_SIZE, IMAGE_SIZE, 3)).numpy()
    masked, mask = IMAGE_PROCESSOR.apply_mask_with_return_numpy(cropped)

    cv.imwrite(str(masked_path), masked.astype('uint8'))
    cv.imwrite(str(mask_path), mask.astype('uint8'))

    patch_match_arguments = (PATCH_MATCH_PATH, masked_path, mask_path, output_path)

    try:

        sp.run(patch_match_arguments, check=True, capture_output=True)

    except sp.CalledProcessError:

        print(Fore.RED + F'PatchMatch failed for image {path.name} on {mask_ratio}!' + Style.RESET_ALL, flush=True)

    print(Fore.GREEN + F'Masked and PatchMatched image {path.name} on {mask_ratio}' + Style.RESET_ALL, flush=True)


def run() -> None:
    images_path = Path('..', 'images', 'original', 'test').resolve()

    for mask_ratio in MASK_RATIO:
        global MASK_GENERATOR, IMAGE_PROCESSOR

        MASK_GENERATOR = MaskGenerator(IMAGE_SIZE, mask_ratio)
        IMAGE_PROCESSOR = ImageProcessor(MASK_GENERATOR)

        masked_path = MASKED_PATH / str(mask_ratio)
        masked_path.mkdir(parents=True, exist_ok=True)
        mask_path = MASKS_PATH / str(mask_ratio)
        mask_path.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_PATH / str(mask_ratio)
        output_path.mkdir(parents=True, exist_ok=True)

        with mp.Pool() as processes_pool:
            function = partial(mask_image, mask_ratio=mask_ratio)
            processes_pool.map(function, images_path.iterdir(), chunksize=200)


def main() -> None:
    print(Fore.MAGENTA + 'Starting script...' + Style.RESET_ALL, flush=True)

    start_time = time.perf_counter()
    try:
        mp.set_start_method('fork')
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


if __name__ == '__main__':
    main()
