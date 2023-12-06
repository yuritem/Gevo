import os
import datetime
import numpy as np
from tqdm import tqdm
from typing import Union
from enum import StrEnum, auto

from graph import Graph
from rw_utils import serialize_boolean_array, pickle_obj
from utils import get_outcome, digits
from config import PARAM_FILENAME, OUTPUT_PATH, DATETIME_TEMPLATE


class FunctionType(StrEnum):
    linear = auto()
    sigmoid = auto()
    longstep = auto()


class Simulation:
    func_type: FunctionType

    def __init__(
            self,
            graph: Graph,
            eps: Union[int, float] = 0.1,
            beta_0: Union[int, float] = 1,
            beta_1: Union[int, float] = 1,
            alpha_0: Union[int, float] = 0,
            alpha_1: Union[int, float] = 0,
            max_num_iter: int = 50000,
            **kwargs
    ):
        self.graph = graph
        self.eps = eps
        self.alpha_0 = alpha_0
        self.alpha_1 = alpha_1
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.max_num_iter = max_num_iter

        # auxiliary/convenience variables
        self.nums_neighbors = self.graph.adj_matrix @ np.ones(shape=self.graph.num_nodes, dtype=np.float32)
        self.delta_0 = self.beta_0 - self.alpha_0
        self.delta_1 = self.beta_1 - self.alpha_1

        # non-constants - initialized in self.setup()
        self.t = None  # discrete time
        self.f = None  # fraction of 1's among node's neighbors
        self.p = None  # transition probabilities
        self.states = None
        self.transitions = None
        self.state_history = None
        self.completed = None  # True if simulation terminates before max_num_iter is achieved

    def setup(self, light=False):
        """if `soft`, don't initialize self.state_history to save memory."""
        self.t = 0
        self.f = np.zeros(shape=self.graph.num_nodes, dtype=np.float32)
        self.p = np.zeros(shape=self.graph.num_nodes, dtype=np.float32)
        self.transitions = np.ones(shape=self.graph.num_nodes, dtype=bool)
        self.initialize_node_states()
        if light:
            self.state_history = None
        else:
            self.state_history = np.zeros(shape=(self.max_num_iter, self.graph.num_nodes), dtype=bool)
        self.completed = False

    def initialize_node_states(self):
        self.states = np.random.random(size=self.graph.num_nodes) < self.eps

    def update_f(self):
        self.f = self.graph.adj_matrix @ self.states.astype(np.float32)
        self.f /= self.nums_neighbors

    def update_p(self):
        raise NotImplementedError

    def perform_state_transitions(self):
        self.update_f()
        self.update_p()
        self.transitions = np.random.random(size=self.graph.num_nodes) < self.p  # True if node should change state
        self.states = np.logical_xor(self.states, self.transitions)  # performs state transitions

    def is_completed(self):
        if np.all(self.states == self.states[0]):
            self.completed = True
            return True

    def reached_max_iter(self):
        return self.t >= self.max_num_iter

    def termination_condition(self):
        return self.is_completed() or self.reached_max_iter()

    def update_state_history(self):
        self.state_history[self.t] = self.states

    def run(self):
        self.setup()
        while not self.termination_condition():
            self.update_state_history()
            self.t += 1
            self.perform_state_transitions()
        if self.completed:
            self.update_state_history()
            self.t += 1
            self.state_history = self.state_history[:self.t]

    def __repr__(self):
        return f"{self.__class__.__name__}_{self.graph.name}"

    def __str__(self):
        return f"{self.__class__.__name__} on {self.graph.name}"


class LinearMixin:
    func_type: FunctionType = FunctionType.linear
    states: np.ndarray
    f: np.ndarray
    p: np.ndarray
    alpha_0: float
    delta_0: float
    beta_1: float
    delta_1: float

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_p(self):
        self.p = np.where(
            self.states,
            self.beta_1 - self.delta_1 * self.f,
            self.alpha_0 + self.delta_0 * self.f
        )


class LongstepMixin:
    func_type: FunctionType = FunctionType.longstep
    states: np.ndarray
    f: np.ndarray
    p: np.ndarray
    alpha_0: float
    alpha_1: float
    delta_0: float
    beta_0: float
    beta_1: float
    delta_1: float

    def __init__(self, X: np.ndarray, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.X = X
        self.k_0 = self.delta_0 / (X[0, 1] - X[0, 0])
        self.k_1 = self.delta_1 / (X[1, 1] - X[1, 0])
        self.b_0 = self.alpha_0 - self.k_0 * X[0, 0]
        self.b_1 = self.beta_1 + self.k_1 * X[1, 0]

    def update_p(self):
        neg_states = ~self.states
        left_0 = neg_states & (self.f < self.X[0, 0])
        mid_0 = neg_states & (self.f >= self.X[0, 0]) & (self.f < self.X[0, 1])
        right_0 = neg_states & (self.f >= self.X[0, 1])

        left_1 = self.states & (self.f < self.X[1, 0])
        mid_1 = self.states & (self.f >= self.X[1, 0]) & (self.f < self.X[1, 1])
        right_1 = self.states & (self.f >= self.X[1, 1])

        self.p[left_0] = self.alpha_0
        self.p[mid_0] = self.b_0 + self.k_0 * self.f[mid_0]
        self.p[right_0] = self.beta_0

        self.p[left_1] = self.beta_1
        self.p[mid_1] = self.b_1 - self.k_1 * self.f[mid_1]
        self.p[right_1] = self.alpha_1


class StaticSimulation(Simulation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.static = None

    def initialize_node_states(self):
        super().initialize_node_states()
        self.static = self.states.copy()

    def perform_state_transitions(self):
        super().perform_state_transitions()
        self.states[self.static] = True


class LinearSimulation(LinearMixin, Simulation):
    pass


class LinearSimulationStatic(LinearMixin, StaticSimulation):
    pass


class LongstepSimulation(LongstepMixin, Simulation):
    pass


class LongstepSimulationStatic(LongstepMixin, StaticSimulation):
    pass


class SimulationResults:
    def __init__(self, sim: Simulation):
        self.t = sim.t
        self.completed = sim.completed
        self.outcome = get_outcome(sim.state_history)
        self.state_history = serialize_boolean_array(sim.state_history)


class SimulationEnsemble:
    def __init__(self, sim: Simulation, num_runs: int = 100):
        sim.setup(light=True)
        self.sim = sim
        self.num_runs = num_runs

    @staticmethod
    def get_dir_name(sim: Simulation, timestamp: bool = True):
        if timestamp:
            id_ = datetime.datetime.now().strftime(DATETIME_TEMPLATE)
        else:
            raise NotImplementedError("Future version will have an option to use ID instead of Timestamp for output folders.")
        static = isinstance(sim, StaticSimulation)
        dir_name = f"{id_}_{sim.func_type}_{sim.graph.name}"
        if static:
            dir_name += "_static"
        return dir_name

    def run(self):
        fmt_str = f'0{digits(self.num_runs - 1)}d'  # format for the number of binary file
        path = os.path.join(OUTPUT_PATH, self.get_dir_name(self.sim))
        pickle_obj(self.sim, filename=PARAM_FILENAME, path=path)
        pbar = tqdm(range(self.num_runs), leave=True, colour='green')
        for i in pbar:
            pbar.set_description(f'{self.sim} ({self.sim.graph.graph.number_of_edges()} edges)', refresh=False)
            self.sim.run()
            sim_results = SimulationResults(self.sim)
            pickle_obj(sim_results, filename=f"{i:{fmt_str}}", path=path)
