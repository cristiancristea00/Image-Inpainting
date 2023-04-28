from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

from colorama import Fore, Style

from graphs_generator import GraphsGenerator
from utils import time_running_cpu


def create_graph(path: Path) -> None:
    print(Fore.CYAN + F'Processing {path.name}...' + Style.RESET_ALL, flush=True)

    with open(path / 'model_training.json', 'r', encoding='UTF-8') as file:
        history = json.load(file)

    graphs_generator = GraphsGenerator(history, path)
    graphs_generator.create_all_graphs()


def main() -> None:
    models_path = Path('models')

    for model_folder in models_path.iterdir():
        with mp.Pool() as processes_pool:
            processes_pool.map(create_graph, model_folder.iterdir())


if __name__ == '__main__':
    time_running_cpu(main)
