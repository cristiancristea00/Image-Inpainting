from __future__ import annotations

import time
import traceback
from typing import Final
from pathlib import Path

from colorama import Fore, Style

from image_browser import ImageBrowser
from image_comparator import ImageComparator
from image_processor import ImageProcessor
from mask_generator import MaskGenerator

BATCH: Final[int] = 256
IMAGE_SIZE: Final[int] = 64
MASK_RATIO: Final[tuple[tuple[float, float]]] = ((5, 10), (15, 20), (25, 30), (35, 40), (45, 50))


def run() -> None:

    results_path = Path('results', 'results.txt')

    with open(results_path, 'w') as results_file:

        for mask_ratio in MASK_RATIO:

            print(Fore.YELLOW + F'Processing dataset with mask ratio {mask_ratio}...' + Style.RESET_ALL, end='\n\n')
            print(F'Mask ratio: {mask_ratio}:', file=results_file)

            mask_generator = MaskGenerator(IMAGE_SIZE, mask_ratio)
            image_processor = ImageProcessor(mask_generator, BATCH)
            image_browser = ImageBrowser(image_processor)

            print(Fore.GREEN + 'Loading Telea dataset...' + Style.RESET_ALL)
            telea_images = image_browser.get_telea()

            print(Fore.GREEN + 'Processing Telea images...' + Style.RESET_ALL)

            psnr_metric = ImageComparator.compute_psnr(telea_images)
            ssim_metric = ImageComparator.compute_ssim(telea_images)

            telea_psnr: str = F'Telea - PSNR: {psnr_metric:.4f}'
            telea_ssim: str = F'Telea - SSIM: {ssim_metric:.4f}'
            print(Fore.CYAN + telea_psnr + Style.RESET_ALL)
            print(Fore.CYAN + telea_ssim + Style.RESET_ALL)
            print(telea_psnr, file=results_file)
            print(telea_ssim, file=results_file)

            print(Fore.GREEN + 'Loading Navier-Stokes dataset...' + Style.RESET_ALL)
            navier_stokes_images = image_browser.get_navier_stokes()

            psnr_metric = ImageComparator.compute_psnr(navier_stokes_images)
            ssim_metric = ImageComparator.compute_ssim(navier_stokes_images)

            navier_stokes_psnr: str = F'Navier-Stokes - PSNR: {psnr_metric:.4f}'
            navier_stokes_ssim: str = F'Navier-Stokes - SSIM: {ssim_metric:.4f}'
            print(Fore.CYAN + navier_stokes_psnr + Style.RESET_ALL)
            print(Fore.CYAN + navier_stokes_ssim + Style.RESET_ALL)
            print(navier_stokes_psnr, file=results_file)
            print(navier_stokes_ssim, file=results_file)

            print(end='\n\n')
            print(end='\n\n', file=results_file)


def main() -> None:
    print(Fore.MAGENTA + 'Starting script...' + Style.RESET_ALL)

    start_time = time.time()
    try:

        run()

    except KeyboardInterrupt:

        print(Fore.RED + '\nScript interrupted by the user!' + Style.RESET_ALL)

    except:

        print(Fore.RED)
        print('Script failed with the following error:')
        traceback.print_exc()
        print(Style.RESET_ALL)

    end_time = time.time()

    elapsed_time = end_time - start_time
    elapsed_time = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))

    print(Fore.YELLOW + F'Total time: {elapsed_time}' + Style.RESET_ALL)
    print(Fore.MAGENTA + 'Everything done!' + Style.RESET_ALL)


if __name__ == '__main__':
    main()
