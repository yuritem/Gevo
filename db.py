import re
import os
import numpy as np
from typing import Union

from rw_utils import read_pickled, deserialize_boolean_array
from sim import StaticSimulation
from config import PARAM_FILENAME, RESULT_FILENAME_PATTERN


class SimulationDirectory:
    def __init__(
            self,
            path: Union[str, bytes, os.PathLike]
    ):
        self.path = path
        self.sim = read_pickled(PARAM_FILENAME, path)
        self.fnames = [
            fname
            for fname in os.listdir(path)
            if re.match(RESULT_FILENAME_PATTERN, fname)
        ]
        self.num_runs = len(self.fnames)
        self.static = isinstance(self.sim, StaticSimulation)

        # initialized once cached
        self.completed = None
        self.outcomes = None
        self.times = None
        self.t_mean_0 = None
        self.t_mean_1 = None
        self.t_max = None
        self.nu = None
        self.state_histories = None
        self.cached_partially = False
        self.cached = False

    def results(self):
        return [
            read_pickled(fname, self.path)
            for fname in self.fnames
        ]

    def partial_cache(self, results=None):
        if not self.cached_partially:
            if not results:
                results = self.results()
            self.completed = np.array([sr.completed for sr in results])
            assert all(self.completed), "Found uncompleted simulations."

            self.outcomes = np.array([sr.outcome for sr in results])
            self.times = np.array([sr.t for sr in results])

            if not self.static:
                self.t_mean_0 = self.times[~self.outcomes].mean()

            self.t_mean_1 = self.times[self.outcomes].mean()
            self.t_max = max(self.times)
            self.nu = self.outcomes.mean()

            self.cached_partially = True

    def cache(self):
        if not self.cached:
            results = self.results()
            self.partial_cache(results)
            self.state_histories = [
                deserialize_boolean_array(sr.state_history, shape=(sr.t, self.sim.graph.num_nodes))
                for sr in results
            ]
            self.cached = True

    def info(self, file=None):
        if not self.cached:
            raise RuntimeError("Cache sdir first.")
        print(f"{self}\nE[t|~A] = {self.t_mean_0}\nE[t|A] = {self.t_mean_1}\ntmax = {self.t_max}", file=file)

    def __repr__(self):
        static_str = '_static' if self.static else ''
        return f"{self.__class__.__name__}_{self.sim.func_type}{static_str}_{self.sim.graph.graph_type}_e{self.sim.graph.graph.number_of_edges()}_run{self.num_runs}"

    def __str__(self):
        return f"{self.__class__.__name__} for {self.sim}, {self.sim.graph.graph.number_of_edges()} edges ({self.num_runs} runs)"


class SimulationDatabase:
    def __init__(
            self,
            path: Union[str, bytes, os.PathLike]
    ):
        self.path = path
        self.sdirs = [
            SimulationDirectory(os.path.join(path, d))
            for d in os.listdir(path)
            if os.path.isfile(os.path.join(path, d, PARAM_FILENAME))
        ]
