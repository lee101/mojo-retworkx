"""Behavioral parity checks against the installable retworkx compatibility package."""

import math
import warnings

import numpy as np
import pytest
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import retworkx as upstream

import mojo_retworx as rx


def make_graph(cls):
    graph = cls()
    graph.add_nodes_from(range(8))
    graph.add_edges_from([
        (0, 1, 1.0), (0, 2, 8.0), (1, 2, 1.5), (1, 3, 3.0),
        (2, 3, 1.0), (2, 4, 4.0), (3, 4, 1.0), (4, 5, 2.0),
        (5, 6, 1.0), (1, 6, 20.0),
    ])
    return graph


def pairs(graph):
    return [(u, v, weight) for u, v, weight in graph.weighted_edge_list()]


def test_graph_container_matches_upstream():
    ours, theirs = make_graph(rx.PyDiGraph), make_graph(upstream.PyDiGraph)
    assert ours.node_indices() == list(theirs.node_indices())
    assert ours.edge_list() == list(theirs.edge_list())
    assert ours.weighted_edge_list() == list(theirs.weighted_edge_list())
    assert ours.successor_indices(1) == list(theirs.successor_indices(1))
    assert ours.predecessor_indices(3) == list(theirs.predecessor_indices(3))


def test_container_payload_indexing_and_unweighted_edge_insertion():
    graph = rx.PyGraph(multigraph=False)
    assert graph.add_node("before") == 0
    graph[0] = "after"
    assert graph[0] == "after"
    assert graph.add_nodes_from(["one", "two"]) == [1, 2]
    assert graph.add_edges_from_no_data([(0, 1), (1, 2)]) == [0, 1]
    assert graph.edge_list() == [(0, 1), (1, 2)]
    assert graph.successor_indices(1) == [2, 0]


def test_extend_from_weighted_edge_list():
    graph = rx.PyGraph()
    graph.extend_from_weighted_edge_list([(2, 4, "a"), (0, 1, "b")])
    assert graph.num_nodes() == 5
    assert graph.weighted_edge_list() == [(2, 4, "a"), (0, 1, "b")]
    assert graph.successor_indices(2) == [4]
    assert graph.successor_indices(4) == [2]


@pytest.mark.parametrize("ours_cls,theirs_cls,ours_fn,theirs_fn", [
    (rx.PyDiGraph, upstream.PyDiGraph, rx.digraph_dijkstra_shortest_path_lengths, upstream.digraph_dijkstra_shortest_path_lengths),
    (rx.PyGraph, upstream.PyGraph, rx.graph_dijkstra_shortest_path_lengths, upstream.graph_dijkstra_shortest_path_lengths),
])
def test_dijkstra_lengths_parity(ours_cls, theirs_cls, ours_fn, theirs_fn):
    ours, theirs = make_graph(ours_cls), make_graph(theirs_cls)
    actual = ours_fn(ours, 0, lambda edge: edge)
    expected = dict(theirs_fn(theirs, 0, lambda edge: edge))
    assert actual == expected
    assert ours_fn(ours, 0, lambda edge: edge, goal=6) == dict(theirs_fn(theirs, 0, lambda edge: edge, goal=6))


@pytest.mark.parametrize("ours_cls,theirs_cls,ours_fn,theirs_fn", [
    (rx.PyDiGraph, upstream.PyDiGraph, rx.digraph_dijkstra_shortest_paths, upstream.digraph_dijkstra_shortest_paths),
    (rx.PyGraph, upstream.PyGraph, rx.graph_dijkstra_shortest_paths, upstream.graph_dijkstra_shortest_paths),
])
def test_dijkstra_paths_parity(ours_cls, theirs_cls, ours_fn, theirs_fn):
    ours, theirs = make_graph(ours_cls), make_graph(theirs_cls)
    actual = ours_fn(ours, 0, weight_fn=lambda edge: edge)
    expected = dict(theirs_fn(theirs, 0, weight_fn=lambda edge: edge))
    assert actual == expected
    assert ours_fn(ours, 0, target=6, weight_fn=lambda edge: edge) == {6: expected[6]}


@pytest.mark.parametrize("ours_cls,theirs_cls,lengths,upstream_lengths,paths,upstream_paths", [
    (rx.PyDiGraph, upstream.PyDiGraph, rx.digraph_all_pairs_dijkstra_path_lengths, upstream.digraph_all_pairs_dijkstra_path_lengths, rx.digraph_all_pairs_dijkstra_shortest_paths, upstream.digraph_all_pairs_dijkstra_shortest_paths),
    (rx.PyGraph, upstream.PyGraph, rx.graph_all_pairs_dijkstra_path_lengths, upstream.graph_all_pairs_dijkstra_path_lengths, rx.graph_all_pairs_dijkstra_shortest_paths, upstream.graph_all_pairs_dijkstra_shortest_paths),
])
def test_all_pairs_dijkstra_lengths_and_paths(ours_cls, theirs_cls, lengths, upstream_lengths, paths, upstream_paths):
    ours, theirs = make_graph(ours_cls), make_graph(theirs_cls)
    actual = lengths(ours, lambda edge: edge)
    expected = {key: dict(value) for key, value in upstream_lengths(theirs, lambda edge: edge).items()}
    assert actual == expected
    actual_paths = paths(ours, lambda edge: edge)
    expected_paths = {key: dict(value) for key, value in upstream_paths(theirs, lambda edge: edge).items()}
    assert actual_paths == expected_paths


@pytest.mark.parametrize("ours_cls,theirs_cls,ours_fn,theirs_fn,edges", [
    (rx.PyDiGraph, upstream.PyDiGraph, rx.digraph_floyd_warshall_numpy, upstream.digraph_floyd_warshall_numpy, [(0, 1, 2.0), (1, 2, -1.0), (2, 3, 4.0), (0, 3, 10.0)]),
    (rx.PyGraph, upstream.PyGraph, rx.graph_floyd_warshall_numpy, upstream.graph_floyd_warshall_numpy, [(0, 1, 2.0), (1, 2, 1.0), (2, 3, 4.0), (0, 3, 10.0)]),
])
def test_floyd_warshall_parity_with_negative_edge(ours_cls, theirs_cls, ours_fn, theirs_fn, edges):
    ours, theirs = ours_cls(), theirs_cls()
    ours.add_nodes_from(range(5)); theirs.add_nodes_from(range(5))
    ours.add_edges_from(edges); theirs.add_edges_from(edges)
    assert np.array_equal(ours_fn(ours, weight_fn=lambda edge: edge), theirs_fn(theirs, weight_fn=lambda edge: edge))


def test_dijkstra_rejects_negative_or_nan_weights():
    graph = rx.PyDiGraph(); graph.add_nodes_from(range(2)); graph.add_edge(0, 1, -1.0)
    with pytest.raises(ValueError):
        rx.digraph_dijkstra_shortest_path_lengths(graph, 0, lambda edge: edge)


@pytest.mark.parametrize("weight", [math.inf, np.finfo(np.float64).max])
def test_dijkstra_and_floyd_preserve_extreme_float64_weights(weight):
    ours, theirs = rx.PyDiGraph(), upstream.PyDiGraph()
    ours.add_nodes_from(range(2)); theirs.add_nodes_from(range(2))
    ours.add_edge(0, 1, weight); theirs.add_edge(0, 1, weight)
    assert rx.digraph_dijkstra_shortest_path_lengths(ours, 0, lambda edge: edge) == dict(
        upstream.digraph_dijkstra_shortest_path_lengths(theirs, 0, lambda edge: edge)
    )
    assert rx.digraph_dijkstra_shortest_paths(ours, 0, weight_fn=lambda edge: edge) == dict(
        upstream.digraph_dijkstra_shortest_paths(theirs, 0, weight_fn=lambda edge: edge)
    )
    assert np.array_equal(
        rx.digraph_floyd_warshall_numpy(ours, weight_fn=lambda edge: edge),
        upstream.digraph_floyd_warshall_numpy(theirs, weight_fn=lambda edge: edge),
    )
    graph = rx.PyDiGraph(); graph.add_nodes_from(range(2)); graph.add_edge(0, 1, math.nan)
    with pytest.raises(ValueError):
        rx.digraph_dijkstra_shortest_path_lengths(graph, 0, lambda edge: edge)


def test_dijkstra_simd_tail_and_topology_cache_invalidation():
    graph = rx.PyDiGraph()
    graph.add_nodes_from(range(5))
    graph.add_edges_from([(0, 1, 1.0), (1, 2, 2.0)])
    calls = []

    def weight(value):
        calls.append(value)
        return value

    assert rx.digraph_dijkstra_shortest_path_lengths(graph, 0, weight) == {1: 1.0, 2: 3.0}
    assert rx.digraph_dijkstra_shortest_path_lengths(graph, 0, weight) == {1: 1.0, 2: 3.0}
    assert calls == [1.0, 2.0, 1.0, 2.0]
    graph.add_edge(2, 4, 3.0)
    assert rx.digraph_dijkstra_shortest_path_lengths(graph, 0, weight) == {1: 1.0, 2: 3.0, 4: 6.0}
    assert calls == [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0]
    graph.add_node(5)
    assert rx.digraph_dijkstra_shortest_path_lengths(graph, 0, weight) == {1: 1.0, 2: 3.0, 4: 6.0}


def test_large_sparse_dijkstra_reachability_prepass():
    ours, theirs = rx.PyDiGraph(), upstream.PyDiGraph()
    ours.add_nodes_from(range(5000)); theirs.add_nodes_from(range(5000))
    reachable = [(node, node + 1, 1.0) for node in range(12)]
    unreachable = [(100 + index % 4900, 100 + (index * 17 + 1) % 4900, 2.0) for index in range(17_000)]
    ours.add_edges_from(reachable + unreachable)
    theirs.add_edges_from(reachable + unreachable)
    actual = rx.digraph_dijkstra_shortest_path_lengths(ours, 0, lambda edge: edge)
    expected = dict(upstream.digraph_dijkstra_shortest_path_lengths(theirs, 0, lambda edge: edge))
    assert actual == expected


def test_topological_sort_is_valid_and_cycle_detection_matches():
    ours, theirs = make_graph(rx.PyDiGraph), make_graph(upstream.PyDiGraph)
    actual = rx.topological_sort(ours)
    assert sorted(actual) == list(range(8))
    positions = {node: i for i, node in enumerate(actual)}
    assert all(positions[u] < positions[v] for u, v, _ in ours.weighted_edge_list())
    assert rx.is_directed_acyclic_graph(ours) == upstream.is_directed_acyclic_graph(theirs)
    ours.add_edge(6, 0, 1.0); theirs.add_edge(6, 0, 1.0)
    assert not rx.is_directed_acyclic_graph(ours)
    with pytest.raises(rx.DAGHasCycle):
        rx.topological_sort(ours)


def test_self_loop_cycle_fast_path():
    graph = rx.PyDiGraph()
    graph.add_nodes_from(range(4))
    graph.add_edges_from([(0, 1, None), (3, 3, None)])
    assert not rx.is_directed_acyclic_graph(graph)
    with pytest.raises(rx.DAGHasCycle):
        rx.topological_sort(graph)


def test_dag_rejects_cycle():
    graph = rx.PyDAG(); graph.add_nodes_from(range(3)); graph.add_edges_from_no_data([(0, 1), (1, 2)])
    with pytest.raises(rx.DAGHasCycle):
        graph.add_edge(2, 0, None)
    with pytest.raises(rx.DAGHasCycle):
        graph.add_edge(0, 0, None)


def test_dfs_and_components_match_upstream():
    ours, theirs = make_graph(rx.PyDiGraph), make_graph(upstream.PyDiGraph)
    assert rx.digraph_dfs_edges(ours, 0) == list(upstream.digraph_dfs_edges(theirs, 0))
    assert {frozenset(x) for x in rx.weakly_connected_components(ours)} == {frozenset(x) for x in upstream.weakly_connected_components(theirs)}
    assert {frozenset(x) for x in rx.strongly_connected_components(ours)} == {frozenset(x) for x in upstream.strongly_connected_components(theirs)}
    assert rx.number_weakly_connected_components(ours) == upstream.number_weakly_connected_components(theirs)
    assert rx.number_strongly_connected_components(ours) == upstream.number_strongly_connected_components(theirs)


def test_undirected_dfs_and_connected_component_apis_match_upstream():
    ours, theirs = make_graph(rx.PyGraph), make_graph(upstream.PyGraph)
    assert rx.graph_dfs_edges(ours, 0) == list(upstream.graph_dfs_edges(theirs, 0))
    assert {frozenset(x) for x in rx.connected_components(ours)} == {frozenset(x) for x in upstream.connected_components(theirs)}
    assert rx.number_connected_components(ours) == upstream.number_connected_components(theirs)
