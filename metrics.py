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
from metrics_and_losses import PSNR, SSIM, SSIM_L1

DATASET: Final[str] = 'COCO'
BATCH: Final[int] = 64
IMAGE_SIZE: Final[int] = 128
MASK_RATIO: Final = ((5, 10), (15, 20), (25, 30), (35, 40), (45, 50))


def run() -> None:
    results_path = Path('results', F'results_{DATASET.lower()}.txt')

    with open(results_path, 'w', encoding='UTF-8') as results_file:
        for mask_ratio in MASK_RATIO:
            print(Fore.YELLOW + F'Processing {DATASET} dataset with mask ratio {mask_ratio}...' + Style.RESET_ALL, end='\n\n', flush=True)
            print(F'Mask ratio: {mask_ratio}:', file=results_file, flush=True)

            mask_generator = MaskGenerator(IMAGE_SIZE, mask_ratio)
            image_processor = ImageProcessor(mask_generator)
            image_browser = ImageBrowser(image_processor, batch_size=BATCH, should_crop=True)

            print(Fore.GREEN + 'Loading Telea dataset...' + Style.RESET_ALL, flush=True)
            telea_images = image_browser.get_telea()

            print(Fore.GREEN + 'Processing Telea images...' + Style.RESET_ALL, flush=True)

            psnr_metric = ImageComparator.compute_psnr(telea_images)
            ssim_metric = ImageComparator.compute_ssim(telea_images)
            lpips_metric = ImageComparator.compute_lpips(telea_images)

            telea_psnr: str = F'Telea - PSNR: {psnr_metric:.4f}'
            telea_ssim: str = F'Telea - SSIM: {ssim_metric:.4f}'
            telea_lpips: str = F'Telea - LPIPS: {lpips_metric:.4f}'
            print(Fore.CYAN + telea_psnr + Style.RESET_ALL, flush=True)
            print(Fore.CYAN + telea_ssim + Style.RESET_ALL, flush=True)
            print(Fore.CYAN + telea_lpips + Style.RESET_ALL, flush=True)
            print(telea_psnr, file=results_file, flush=True)
            print(telea_ssim, file=results_file, flush=True)
            print(telea_lpips, file=results_file, flush=True)

            print(Fore.GREEN + 'Loading Navier-Stokes dataset...' + Style.RESET_ALL, flush=True)
            navier_stokes_images = image_browser.get_navier_stokes()

            print(Fore.GREEN + 'Processing Navier-Stokes images...' + Style.RESET_ALL, flush=True)

            psnr_metric = ImageComparator.compute_psnr(navier_stokes_images)
            ssim_metric = ImageComparator.compute_ssim(navier_stokes_images)
            lpips_metric = ImageComparator.compute_lpips(navier_stokes_images)

            navier_stokes_psnr: str = F'Navier-Stokes - PSNR: {psnr_metric:.4f}'
            navier_stokes_ssim: str = F'Navier-Stokes - SSIM: {ssim_metric:.4f}'
            navier_stokes_lpips: str = F'Navier-Stokes - LPIPS: {lpips_metric:.4f}'
            print(Fore.CYAN + navier_stokes_psnr + Style.RESET_ALL, flush=True)
            print(Fore.CYAN + navier_stokes_ssim + Style.RESET_ALL, flush=True)
            print(Fore.CYAN + navier_stokes_lpips + Style.RESET_ALL, flush=True)
            print(navier_stokes_psnr, file=results_file, flush=True)
            print(navier_stokes_ssim, file=results_file, flush=True)
            print(navier_stokes_lpips, file=results_file, flush=True)

            print(Fore.GREEN + 'Loading PatchMatch dataset...' + Style.RESET_ALL, flush=True)
            patch_match_images = image_browser.get_patch_match()

            print(Fore.GREEN + 'Processing PatchMatch images...' + Style.RESET_ALL, flush=True)

            psnr_metric = ImageComparator.compute_psnr(patch_match_images)
            ssim_metric = ImageComparator.compute_ssim(patch_match_images)
            lpips_metric = ImageComparator.compute_lpips(patch_match_images)

            patch_match_psnr: str = F'PatchMatch - PSNR: {psnr_metric:.4f}'
            patch_match_ssim: str = F'PatchMatch - SSIM: {ssim_metric:.4f}'
            patch_match_lpips: str = F'PatchMatch - LPIPS: {lpips_metric:.4f}'
            print(Fore.CYAN + patch_match_psnr + Style.RESET_ALL, flush=True)
            print(Fore.CYAN + patch_match_ssim + Style.RESET_ALL, flush=True)
            print(Fore.CYAN + patch_match_lpips + Style.RESET_ALL, flush=True)
            print(patch_match_psnr, file=results_file, flush=True)
            print(patch_match_ssim, file=results_file, flush=True)
            print(patch_match_lpips, file=results_file, flush=True)

            print(Fore.GREEN + 'Loading Inpainting with MAE loss dataset...' + Style.RESET_ALL, flush=True)
            model_path: Path = Path('models', F'UNet MAE {DATASET} {mask_ratio}')
            model = tf.keras.models.load_model(model_path, custom_objects={'PSNR': PSNR, 'SSIM': SSIM})
            mae_inpainted_images = image_browser.get_model_inpainted(model)

            lpips_metric = ImageComparator.compute_lpips(mae_inpainted_images)

            mae_inpainted_lpips: str = F'UNet MAE - LPIPS: {lpips_metric:.4f}'
            print(Fore.CYAN + mae_inpainted_lpips + Style.RESET_ALL, flush=True)
            print(mae_inpainted_lpips, file=results_file, flush=True)

            print(Fore.GREEN + 'Loading Inpainting with MSE loss dataset...' + Style.RESET_ALL, flush=True)
            model_path: Path = Path('models', F'UNet MSE {DATASET} {mask_ratio}')
            model = tf.keras.models.load_model(model_path, custom_objects={'PSNR': PSNR, 'SSIM': SSIM})
            mse_inpainted_images = image_browser.get_model_inpainted(model)

            lpips_metric = ImageComparator.compute_lpips(mse_inpainted_images)

            mse_inpainted_lpips: str = F'UNet MSE - LPIPS: {lpips_metric:.4f}'
            print(Fore.CYAN + mse_inpainted_lpips + Style.RESET_ALL, flush=True)
            print(mse_inpainted_lpips, file=results_file, flush=True)

            print(Fore.GREEN + 'Loading Inpainting with SSIM+MAE loss dataset...' + Style.RESET_ALL, flush=True)
            model_path: Path = Path('models', F'UNet SSIM+MAE {DATASET} {mask_ratio}')
            model = tf.keras.models.load_model(model_path, custom_objects={'PSNR': PSNR, 'SSIM': SSIM, 'SSIM_L1': SSIM_L1})
            ssim_mae_inpainted_images = image_browser.get_model_inpainted(model)

            lpips_metric = ImageComparator.compute_lpips(ssim_mae_inpainted_images)

            ssim_mae_inpainted_lpips: str = F'UNet SSIM+MAE - LPIPS: {lpips_metric:.4f}'
            print(Fore.CYAN + ssim_mae_inpainted_lpips + Style.RESET_ALL, flush=True)
            print(ssim_mae_inpainted_lpips, file=results_file, flush=True)

            print(end='\n\n', flush=True)
            print(end='\n\n', file=results_file, flush=True)


def main() -> None:
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


if __name__ == '__main__':
    main()
