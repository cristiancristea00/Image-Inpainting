from __future__ import annotations

import json
import multiprocessing as mp
import time
import traceback
from pathlib import Path

from colorama import Fore, Style

from graphs_generator import GraphsGenerator


def create_graph(path: Path) -> None:
    print(Fore.CYAN + F'Processing {path.name}...' + Style.RESET_ALL)

    with open(path / 'model_training.json', 'r', encoding='UTF-8') as file:
        history = json.load(file)

    graphs_generator = GraphsGenerator(history, path)
    graphs_generator.create_all_graphs()


def run() -> None:
    models_path = Path('models')

    with mp.Pool() as processes_pool:
        processes_pool.map(create_graph, models_path.iterdir())


def main() -> None:
    print(Fore.MAGENTA + 'Starting script...' + Style.RESET_ALL)

    start_time = time.perf_counter()
    try:
        mp.set_start_method('fork')
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
