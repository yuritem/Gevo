from typing import Union, Tuple
from matplotlib import pyplot as plt

from sim import Simulation
from db import SimulationDirectory
from utils import sigma_mean


# {'er': '-.', 'roc': '-,', 'ring': '--'}
class SigmaPlot:

    def __init__(
            self,
            x_left: Union[int, float] = 0.,
            x_right: Union[int, float, None] = None,
            y_top: Union[int, float] = 1.,
            y_bottom: Union[int, float] = 0.,
            alpha: float = 1.,
            color: str = 'black',
            linestyle: str = '-',
            linewidth: float = 2.,
            figsize: Tuple[Union[int, float]] = (10, 10),
            dpi: int = 300,
            x_label: str = r"$t$",
            y_label: str = r"$\sigma(t)$",
            plot_title: Union[str, None] = None,
            ax_title_size: int = 24,
            ax_label_size: int = 24,
            ax_tick_size: int = 16,
            grid_color: str = '#b0b0b0',
            grid_linestyle: str = '-',
            grid_linewidth: float = 0.8,
            display_legend: bool = False,
            legend_fontsize: int = 18
    ):
        self.x_left = x_left
        self.x_right = x_right
        self.y_top = y_top
        self.y_bottom = y_bottom
        self.alpha = alpha
        self.color = color
        self.linestyle = linestyle
        self.linewidth = linewidth
        self._figsize = figsize
        self._dpi = dpi
        self.x_label = x_label
        self.y_label = y_label
        self.plot_title = plot_title
        self.ax_title_size = ax_title_size
        self.ax_label_size = ax_label_size
        self.ax_tick_size = ax_tick_size
        self.grid_color = grid_color
        self.grid_linestyle = grid_linestyle
        self.grid_linewidth = grid_linewidth
        self.display_legend = display_legend
        self.legend_fontsize = legend_fontsize

        self.fig = None
        self.ax = None
        self._setup()

    def _setup(self):
        self.fig, self.ax = plt.subplots(figsize=self._figsize, dpi=self._dpi)
        self.ax.tick_params(axis='both', which='major', labelsize=self.ax_tick_size)
        self.update_axlims()
        self.set_axlabels()
        self.ax.grid(
            color=self.grid_color,
            linestyle=self.grid_linestyle,
            linewidth=self.grid_linewidth
        )

    def update_axlims(
            self,
            x_left: Union[int, float, None] = None,
            x_right: Union[int, float, None] = None,
            y_top: Union[int, float, None] = None,
            y_bottom: Union[int, float, None] = None,
    ):
        self.x_left = x_left or self.x_left
        self.x_right = x_right or self.x_right
        self.y_top = y_top or self.y_top
        self.y_bottom = y_bottom or self.y_bottom
        self.ax.set_xlim(left=self.x_left, right=x_right)
        self.ax.set_ylim(bottom=self.y_bottom, top=self.y_top)

    def set_axlabels(self):
        self.ax.set_xlabel(self.x_label, fontsize=self.ax_label_size)
        self.ax.set_ylabel(self.y_label, fontsize=self.ax_label_size)

    def set_plot_title(self, plot_title: str):
        self.plot_title = plot_title
        self.ax.set_title(self.plot_title, y=1.03, fontsize=self.ax_title_size)

    @staticmethod
    def get_plot_title(sim: Simulation):
        newline = '\n'
        return fr"{sim}{newline}$\beta_0={sim.beta_0}$, $\beta_1={sim.beta_1}$, $\varepsilon={sim.eps}$"

    def plot(
            self,
            sdir: SimulationDirectory,
            x_max: Union[int, None] = None,
            alpha: Union[float, None] = None,
            color: Union[str, None] = None,
            label: str = '',
            linestyle: Union[str, None] = None,
            linewidth: Union[float, None] = None
    ):
        alpha = alpha or self.alpha
        color = color or self.color
        linestyle = linestyle or self.linestyle
        linewidth = linewidth or self.linewidth

        sdir.cache()

        sigma_avg = sigma_mean(sdir.state_histories, pad_to=x_max)

        lines_2d = self.ax.plot(
            range(len(sigma_avg)),
            sigma_avg,
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            ls=linestyle,
            label=label
        )

        self.update_axlims()

        if self.display_legend:
            self.ax.legend(fontsize=self.legend_fontsize)

        return lines_2d[0]
