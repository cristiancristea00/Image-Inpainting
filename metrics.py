from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Final

import tensorflow as tf
from colorama import Fore, Style

from image_browser import ImageBrowser
from image_comparator import ImageComparator
from image_processor import ImageProcessor
from mask_generator import MaskGenerator
from metrics_and_losses import PSNR, SSIM

BATCH: Final[int] = 256
IMAGE_SIZE: Final[int] = 64
MASK_RATIO: Final = ((5, 10), (15, 20), (25, 30), (35, 40), (45, 50))


def run() -> None:
    results_path = Path('results', 'results.txt')

    with open(results_path, 'w', encoding='UTF-8') as results_file:
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
            lpips_metric = ImageComparator.compute_lpips(telea_images)

            telea_psnr: str = F'Telea - PSNR: {psnr_metric:.4f}'
            telea_ssim: str = F'Telea - SSIM: {ssim_metric:.4f}'
            telea_lpips: str = F'Telea - LPIPS: {lpips_metric:.4f}'
            print(Fore.CYAN + telea_psnr + Style.RESET_ALL)
            print(Fore.CYAN + telea_ssim + Style.RESET_ALL)
            print(Fore.CYAN + telea_lpips + Style.RESET_ALL)
            print(telea_psnr, file=results_file)
            print(telea_ssim, file=results_file)
            print(telea_lpips, file=results_file)

            print(Fore.GREEN + 'Loading Navier-Stokes dataset...' + Style.RESET_ALL)
            navier_stokes_images = image_browser.get_navier_stokes()

            psnr_metric = ImageComparator.compute_psnr(navier_stokes_images)
            ssim_metric = ImageComparator.compute_ssim(navier_stokes_images)
            lpips_metric = ImageComparator.compute_lpips(navier_stokes_images)

            navier_stokes_psnr: str = F'Navier-Stokes - PSNR: {psnr_metric:.4f}'
            navier_stokes_ssim: str = F'Navier-Stokes - SSIM: {ssim_metric:.4f}'
            navier_stokes_lpips: str = F'Navier-Stokes - LPIPS: {lpips_metric:.4f}'
            print(Fore.CYAN + navier_stokes_psnr + Style.RESET_ALL)
            print(Fore.CYAN + navier_stokes_ssim + Style.RESET_ALL)
            print(Fore.CYAN + navier_stokes_lpips + Style.RESET_ALL)
            print(navier_stokes_psnr, file=results_file)
            print(navier_stokes_ssim, file=results_file)
            print(navier_stokes_lpips, file=results_file)

            print(Fore.GREEN + 'Loading Inpainting with MAE (L1) loss dataset...' + Style.RESET_ALL)
            model_path: Path = Path('models', F'UNet MAE {mask_ratio}')
            model = tf.keras.models.load_model(model_path, custom_objects={'PSNR': PSNR, 'SSIM': SSIM})
            mae_inpainted_images = image_browser.get_model_inpainted(model)

            lpips_metric = ImageComparator.compute_lpips(mae_inpainted_images)

            mae_inpainted_lpips: str = F'UNet MAE - LPIPS: {lpips_metric:.4f}'
            print(Fore.CYAN + mae_inpainted_lpips + Style.RESET_ALL)
            print(mae_inpainted_lpips, file=results_file)

            print(Fore.GREEN + 'Loading Inpainting with MSE (L2) loss dataset...' + Style.RESET_ALL)
            model_path: Path = Path('models', F'UNet MSE {mask_ratio}')
            model = tf.keras.models.load_model(model_path, custom_objects={'PSNR': PSNR, 'SSIM': SSIM})
            mse_inpainted_images = image_browser.get_model_inpainted(model)

            lpips_metric = ImageComparator.compute_lpips(mse_inpainted_images)

            mse_inpainted_lpips: str = F'UNet MSE - LPIPS: {lpips_metric:.4f}'
            print(Fore.CYAN + mse_inpainted_lpips + Style.RESET_ALL)
            print(mse_inpainted_lpips, file=results_file)

            print(end='\n\n')
            print(end='\n\n', file=results_file)


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
