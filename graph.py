import warnings
import numpy as np
import networkx as nx
from typing import Iterable
from enum import StrEnum, auto


def ring_graph(num_nodes: int, degree: int) -> nx.Graph:

    if degree >= num_nodes:
        raise ValueError("degree >= num_nodes, choose smaller k or larger n")
    if degree % 2:
        warnings.warn(
            f"Odd k in ring_graph(). Using degree = {degree - 1} instead.",
            category=RuntimeWarning,
            stacklevel=2
        )

    g = nx.Graph()
    nodes = list(range(num_nodes))  # nodes are labeled 0 to n-1

    for j in range(1, degree // 2 + 1):  # connect each node to k/2 neighbors
        targets = nodes[j:] + nodes[0:j]  # first j nodes are now last
        g.add_edges_from(zip(nodes, targets))

    return g


class GraphType(StrEnum):
    complete = auto()
    ring = auto()
    roc = auto()
    er = auto()


class Graph:
    graph = nx.empty_graph()
    num_nodes: int
    adj_matrix: np.ndarray
    name: str
    graph_type: GraphType

    @staticmethod
    def _get_adjacency_matrix(graph):
        return nx.to_numpy_array(graph).astype(np.float32)

    def is_connected_after_node_removal(self, nodes_to_remove: Iterable) -> bool:
        g = self.graph.copy()
        g.remove_nodes_from(nodes_to_remove)
        return nx.is_connected(g)

    def __repr__(self):
        return self.name


class CompleteGraph(Graph):
    graph_type = GraphType.complete

    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.graph = nx.complete_graph(self.num_nodes)
        self.name = f"{self.graph_type}(n={self.num_nodes})"
        self.adj_matrix = self._get_adjacency_matrix(self.graph)


class RingGraph(Graph):
    graph_type = GraphType.ring

    def __init__(self, num_nodes: int, degree: int):
        self.num_nodes = num_nodes
        self.degree = degree
        self.graph = ring_graph(self.num_nodes, self.degree)
        self.name = f"{self.graph_type}(n={self.num_nodes},k={self.degree})"
        self.adj_matrix = self._get_adjacency_matrix(self.graph)


class RocGraph(Graph):
    graph_type = GraphType.roc

    def __init__(self, num_cliques: int, clique_size: int):
        self.num_nodes = num_cliques * clique_size
        self.num_cliques = num_cliques
        self.clique_size = clique_size
        self.graph = nx.ring_of_cliques(self.num_cliques, self.clique_size)
        self.name = f"{self.graph_type}(k={self.num_cliques},l={self.clique_size})"
        self.adj_matrix = self._get_adjacency_matrix(self.graph)


class ErGraph(Graph):
    graph_type = GraphType.er

    def __init__(self, num_nodes: int, probability: float = None, num_edges: int = None):
        self.num_nodes = num_nodes
        if (probability is None) == (num_edges is None):
            raise ValueError("Supply either p or m")
        if probability is not None:
            self.probability = probability
            self.graph = nx.erdos_renyi_graph(self.num_nodes, self.probability)
            while not nx.is_connected(self.graph):
                self.graph = nx.erdos_renyi_graph(self.num_nodes, self.probability)
            self.name = f"{self.graph_type}(n={self.num_nodes},p={self.probability})"
        if num_edges is not None:
            self.num_edges = num_edges
            self.graph = nx.gnm_random_graph(self.num_nodes, self.num_edges)
            while not nx.is_connected(self.graph):
                self.graph = nx.gnm_random_graph(self.num_nodes, self.num_edges)
            self.name = f"{self.graph_type}(n={self.num_nodes},m={self.num_edges})"
        self.adj_matrix = self._get_adjacency_matrix(self.graph)
