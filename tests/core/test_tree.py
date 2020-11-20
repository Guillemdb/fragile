import networkx
import judo
from judo.tests.test_tree import TestNetworkxTree, to_node_id
import pytest

from fragile.core.tree import HistoryTree


def random_powerlaw():
    g = networkx.DiGraph()
    t = networkx.random_powerlaw_tree(500, gamma=3, tries=1000, seed=160290)
    graph = networkx.compose(g, t)
    mapping = {n: to_node_id(n) for n in graph.nodes}
    return networkx.relabel_nodes(graph, mapping)


def small_tree():
    node_data = {"a": judo.arange(10), "b": judo.zeros(10)}
    edge_data = {"c": judo.ones(10)}
    g = networkx.DiGraph()
    for i in range(8):
        g.add_node(to_node_id(i), **node_data)
    pairs = [(0, 1), (1, 2), (2, 3), (2, 4), (2, 5), (3, 6), (3, 7)]
    for a, b in pairs:
        g.add_edge(to_node_id(a), to_node_id(b), **edge_data)
    return g


@pytest.fixture(params=[random_powerlaw, small_tree], scope="function")
def tree(request):
    tree = HistoryTree()
    tree.data = request.param()
    return tree
