from pathlib import Path
from typing import Final

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np


class GraphsGenerator:

    __PLOT_SIZE: Final[tuple[int, int]] = (16, 9)

    __PRIMARY_COLOR: Final[str] = 'purple'
    __SECONDARY_COLOR: Final[str] = 'green'

    __LINE_THICKNESS: Final[int] = 3

    __LEGEND_COLOR: Final[str] = 'white'
    __LEGEND_ALPHA: Final[float] = 0.75

    __TITLE_PADDING: Final[int] = 15
    __LABEL_PADDING: Final[int] = 10

    __TITLE_FONT_SIZE: Final[int] = 32
    __LABEL_FONT_SIZE: Final[int] = 28
    __TICKS_FONT_SIZE: Final[int] = 20
    __LEGEND_FONT_SIZE: Final[int] = 26

    __PLOT_SKIP_SIZE: Final[int] = 10

    def __init__(self, history: dict, output_dir: Path) -> None:
        mpl.rcParams['font.sans-serif'] = ['Readex Pro']
        plt.style.use('ggplot')

        self.history = history
        self.output_dir = output_dir

    @property
    def history(self):
        return self.__history

    @history.setter
    def history(self, value):
        self.__history = value

    @property
    def output_dir(self):
        return self.__output_dir

    @output_dir.setter
    def output_dir(self, value):
        self.__output_dir = value

    def __create_graph(self, parameter: str, title: str, legend_position: str) -> plt.Figure:
        figure, axis = plt.subplots(1, 1, figsize=self.__PLOT_SIZE)

        length = len(self.history['loss'])
        dims = np.arange(0, length)
        ticks = np.arange(0, length + 1, self.__PLOT_SKIP_SIZE)

        train_parameter = parameter
        train_parameter_label = F'Training {train_parameter.capitalize()}'
        val_parameter = F'val_{train_parameter}'
        val_parameter_label = F'Validation {val_parameter.capitalize()}'

        axis.plot(dims, self.history[train_parameter], label=train_parameter_label, color=self.__PRIMARY_COLOR, linewidth=self.__LINE_THICKNESS)
        axis.plot(dims, self.history[val_parameter], label=val_parameter_label, color=self.__SECONDARY_COLOR, linewidth=self.__LINE_THICKNESS)
        axis.set(xlabel='Epoch', ylabel=parameter.capitalize())
        axis.set_title(title, pad=self.__TITLE_PADDING, size=self.__TITLE_FONT_SIZE)
        axis.legend(fontsize=self.__LEGEND_FONT_SIZE, facecolor=self.__LEGEND_COLOR, framealpha=self.__LEGEND_ALPHA, loc=legend_position)
        axis.set_xlim(xmin=1, xmax=length)
        axis.tick_params(axis='both', which='major', labelsize=self.__TICKS_FONT_SIZE)
        axis.xaxis.label.set_size(self.__LABEL_FONT_SIZE)
        axis.yaxis.label.set_size(self.__LABEL_FONT_SIZE)
        axis.xaxis.labelpad = self.__LABEL_PADDING
        axis.yaxis.labelpad = self.__LABEL_PADDING
        axis.set_xticks(ticks)

        return figure

    def __save_graph(self, figure: plt.Figure, parameter: str) -> None:
        save_path = self.output_dir / F'model_history_{parameter}.png'
        figure.savefig(save_path, dpi=100, bbox_inches='tight')

    def create_loss_graph(self) -> None:
        parameter = 'loss'
        title = parameter.capitalize()
        figure = self.__create_graph(parameter, title, 'upper right')
        self.__save_graph(figure, parameter)

    def create_psnr_graph(self) -> None:
        parameter = 'psnr'
        title = parameter.upper()
        figure = self.__create_graph(parameter, title, 'lower right')
        self.__save_graph(figure, parameter)

    def create_ssim_graph(self) -> None:
        parameter = 'ssim'
        title = parameter.upper()
        figure = self.__create_graph(parameter, title, 'lower right')
        self.__save_graph(figure, parameter)

    def create_all_graphs(self) -> None:
        for method in dir(self):
            if method.startswith('create_') and method.endswith('_graph'):
                getattr(self, method)()
