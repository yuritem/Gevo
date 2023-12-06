import gc
import os
import imageio
import numpy as np
from tqdm import tqdm
import networkx as nx
from typing import List, Union
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from sim import Simulation, SimulationResults
from graph import GraphType
from rw_utils import deserialize_boolean_array
from config import COLOR_0, COLOR_1

nxdraw = {
    "node_size": 150,
    # "node_color": COLOR_0,
    "linewidths": 1.5,  # node outline
    "width": 1.5,  # edge thickness
    "edge_color": '#595959',
    "edgecolors": '#595959',
    "font_color": '#595959',
    "font_weight": 'semibold',
    "font_size": 16
}


def figure_to_image(figure):
    canvas = FigureCanvas(figure)
    canvas.draw()  # draw the canvas, cache the renderer
    width, height = figure.get_size_inches() * figure.get_dpi()
    image = np.array(canvas.buffer_rgba(), dtype='uint8').reshape(int(height), int(width), 4)
    plt.close(figure)
    return image


def create_gif(
        images: List[np.ndarray],
        filename: str,
        path: Union[str, bytes, os.PathLike],
        fps: int,
        overwrite: bool = False
):
    """
    Creates an .mkv animation from a list of figures lasting `duration` seconds.
    """

    ext = 'mkv'

    if path is None:
        path = 'output'

    if not filename.endswith(f".{ext}"):
        if '.' in filename:
            filename, _ = os.path.split(filename)
        filename = f"{filename}.{ext}"

    filepath = os.path.join(path, filename)

    if os.path.exists(filepath):
        print(f"{filepath} already exists.", end=' ')
        if overwrite:
            print(f"Overwriting...")
        else:
            print(f"Returning...")
            return

    imageio.mimsave(filepath, images, fps=fps)


def figures_to_images(fig_list):
    images = []
    pbar = tqdm(fig_list, position=0, leave=True)
    for figure in pbar:
        pbar.set_description("Converting fig2img", refresh=False)
        images.append(figure_to_image(figure))
        plt.close(figure)
        del figure
    gc.collect()
    return images


def plot_graph_figures(
        sim: Simulation,
        sr: SimulationResults,
        t_start: int = 0,
        t_end: Union[int, None] = None,
        dpi=72,
):
    color_0 = COLOR_0
    color_1 = COLOR_1

    if t_end is None:
        t_end = sr.t
    else:
        t_end = min(sr.t, t_end)

    layout = nx.circular_layout(sim.graph.graph) if sim.graph.graph_type == GraphType.ring else nx.kamada_kawai_layout(sim.graph.graph)
    colors = np.where(
        deserialize_boolean_array(sr.state_history, shape=(sr.t, sim.graph.num_nodes)),
        color_1,
        color_0
    )

    figs = []
    pbar = tqdm(range(t_start, t_end), leave=True)
    for i in pbar:
        pbar.set_description("Plotting figures", refresh=False)

        fig = plt.figure(figsize=(8, 8), dpi=dpi)

        nx.draw(sim.graph.graph, pos=layout, node_color=colors[i], **nxdraw)

        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)  # clips title
        plt.margins(x=0, y=0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        figs.append(fig)
        plt.close(fig)

    return figs


def animate(
        sim: Simulation,
        sr: SimulationResults,
        filename: str,
        path: Union[str, bytes, os.PathLike],
        t_start: int = 0,
        t_end: Union[int, None] = None,
        dpi: int = 72,
        fps: int = 30,
        overwrite: bool = False
):
    figs = plot_graph_figures(sim, sr, t_start, t_end, dpi)
    images = figures_to_images(figs)
    create_gif(images, filename, path, fps, overwrite)


def main():
    from sim import LinearSimulationStatic
    from graph import RocGraph
    params = {
        'graph': RocGraph(num_cliques=3, clique_size=30),
        'eps': 0.1,
        'alpha_0': 0.0,
        'alpha_1': 0.0,
        'beta_0': 0.025,
        'beta_1': 0.05,
        'X': np.array([[.05, .95], [.05, .95]]),
        'max_num_iter': 10_000
    }
    sim = LinearSimulationStatic(**params)
    sim.run()
    sr = SimulationResults(sim)
    animate(sim, sr, filename='test3', path='output', t_end=30, overwrite=False)


if __name__ == '__main__':
    main()
