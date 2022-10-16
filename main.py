from image_browser import ImageBrowser, InpaintMethod
from image_comparator import ImageComparator
from prettytable import PrettyTable, MARKDOWN
from pathlib import Path


def main():
    patch_match = []
    navier_stokes = []
    telea = []

    for elem in ImageBrowser.generate_all(InpaintMethod.PATCH_MATCH):
        original, masked, inpainted, name = elem
        difference = ImageComparator.compute_difference(original, masked, inpainted)
        patch_match.append((name, difference))

    for elem in ImageBrowser.generate_all(InpaintMethod.NAVIER_STOKES):
        original, masked, inpainted, name = elem
        difference = ImageComparator.compute_difference(original, masked, inpainted)
        navier_stokes.append((name, difference))

    for elem in ImageBrowser.generate_all(InpaintMethod.TELEA):
        original, masked, inpainted, name = elem
        difference = ImageComparator.compute_difference(original, masked, inpainted)
        telea.append((name, difference))

    patch_match.sort(key=lambda pair: int(pair[0][:-4]))
    navier_stokes.sort(key=lambda pair: int(pair[0][:-4]))
    telea.sort(key=lambda pair: int(pair[0][:-4]))

    average_patch_match = sum([pair[1] for pair in patch_match]) / len(patch_match)
    average_navier_stokes = sum([pair[1] for pair in navier_stokes]) / len(navier_stokes)
    average_telea = sum([pair[1] for pair in telea]) / len(telea)

    images = [pair[0][:-4] for pair in patch_match]
    patch_match = [F'{pair[1]:.2f}' for pair in patch_match]
    navier_stokes = [F'{pair[1]:.2f}' for pair in navier_stokes]
    telea = [F'{pair[1]:.2f}' for pair in telea]

    table = PrettyTable()
    table.set_style(MARKDOWN)
    table.add_column('Image', images)
    table.add_column('PatchMatch', patch_match)
    table.add_column('Navier-Stokes', navier_stokes)
    table.add_column('Telea', telea)
    table.add_row(['Average', F'{average_patch_match:.2f}', F'{average_navier_stokes:.2f}', F'{average_telea:.2f}'])

    results_path = Path('results/RESULTS.md')

    with results_path.open('w') as file:
        file.write(R'# Difference gained between the masked and the inpainted images' + '\n\n')
        file.write(table.get_string() + '\n')


if __name__ == '__main__':
    main()
