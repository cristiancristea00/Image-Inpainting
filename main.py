from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from prettytable import MARKDOWN, PrettyTable

from image_browser import InpaintMethod
from image_comparator import ImageComparator


def main() -> None:
    results_path = Path('results/RESULTS.md')

    with ThreadPoolExecutor() as executor:
        patch_match = executor.submit(ImageComparator.compute_results, InpaintMethod.PATCH_MATCH)
        navier_stokes = executor.submit(ImageComparator.compute_results, InpaintMethod.NAVIER_STOKES)
        telea = executor.submit(ImageComparator.compute_results, InpaintMethod.TELEA)

        patch_match = patch_match.result()
        navier_stokes = navier_stokes.result()
        telea = telea.result()

    table = PrettyTable()
    table.set_style(MARKDOWN)
    table.field_names = ['Image', 'MSE', 'PSNR', 'SSIM', 'LPIPS']
    table.add_rows(patch_match)

    with results_path.open('w') as file:
        file.write(R'# Patch Match method' + '\n\n')
        file.write(table.get_string() + '\n')

    table.clear()
    table.set_style(MARKDOWN)
    table.field_names = ['Image', 'MSE', 'PSNR', 'SSIM', 'LPIPS']
    table.add_rows(navier_stokes)

    with results_path.open('a') as file:
        file.write(R'# Navier Stokes method' + '\n\n')
        file.write(table.get_string() + '\n')

    table.clear()
    table.set_style(MARKDOWN)
    table.field_names = ['Image', 'MSE', 'PSNR', 'SSIM', 'LPIPS']
    table.add_rows(telea)

    with results_path.open('a') as file:
        file.write(R'# Telea method' + '\n\n')
        file.write(table.get_string() + '\n')


if __name__ == '__main__':
    main()
