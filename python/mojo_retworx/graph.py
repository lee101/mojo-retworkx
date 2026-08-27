"""Integer-indexed graph objects and the covered retworkx algorithm surface."""

from __future__ import annotations

from collections import deque
from typing import Callable

import numpy as np

from ._lib import addr, f64, i64, lib


class DAGHasCycle(Exception):
    pass


class _BaseGraph:
    directed = False

    def __init__(self, multigraph: bool = True, /, **_):
        self.multigraph = bool(multigraph)
        self._nodes: list[object] = []
        self._edges: list[tuple[int, int, object]] = []
        self._topology_cache: dict[bool, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._has_self_loop = False

    def __len__(self) -> int:
        return len(self._nodes)

    def __getitem__(self, index: int):
        return self._nodes[index]

    def __setitem__(self, index: int, value) -> None:
        self._nodes[index] = value

    def add_node(self, obj, /) -> int:
        self._nodes.append(obj)
        self._topology_cache.clear()
        return len(self._nodes) - 1

    def add_nodes_from(self, obj_list, /) -> list[int]:
        return [self.add_node(obj) for obj in obj_list]

    def node_indices(self):
        return list(range(len(self._nodes)))

    def nodes(self):
        return list(self._nodes)

    def add_edge(self, parent: int, child: int, obj, /) -> int:
        self._check_node(parent)
        self._check_node(child)
        if not self.multigraph:
            for index, (u, v, _) in enumerate(self._edges):
                if (u, v) == (parent, child) or (not self.directed and (v, u) == (parent, child)):
                    self._edges[index] = (u, v, obj)
                    self._topology_cache.clear()
                    return index
        self._edges.append((parent, child, obj))
        self._has_self_loop |= parent == child
        self._topology_cache.clear()
        return len(self._edges) - 1

    def add_edges_from(self, obj_list, /) -> list[int]:
        return [self.add_edge(u, v, data) for u, v, data in obj_list]

    def add_edge_from_index(self, parent: int, child: int, obj, /) -> int:
        return self.add_edge(parent, child, obj)

    def add_edges_from_no_data(self, obj_list, /) -> list[int]:
        return [self.add_edge(u, v, None) for u, v in obj_list]

    def extend_from_edge_list(self, edge_list, /) -> None:
        for u, v in edge_list:
            self._extend_nodes(max(u, v))
            self.add_edge(u, v, None)

    def extend_from_weighted_edge_list(self, edge_list, /) -> None:
        for u, v, data in edge_list:
            self._extend_nodes(max(u, v))
            self.add_edge(u, v, data)

    def edge_list(self):
        return [(u, v) for u, v, _ in self._edges]

    def weighted_edge_list(self):
        return list(self._edges)

    def num_nodes(self) -> int:
        return len(self._nodes)

    def num_edges(self) -> int:
        return len(self._edges)

    def has_edge(self, u: int, v: int, /) -> bool:
        return any((a == u and b == v) or (not self.directed and a == v and b == u) for a, b, _ in self._edges)

    def out_edge_list(self, node: int, /):
        self._check_node(node)
        if self.directed:
            return [(u, v, data) for u, v, data in reversed(self._edges) if u == node]
        return [(node, v if u == node else u, data) for u, v, data in reversed(self._edges) if u == node or v == node]

    def successor_indices(self, node: int, /):
        return [v for _, v, _ in self.out_edge_list(node)]

    def _extend_nodes(self, last: int) -> None:
        while len(self._nodes) <= last:
            self._nodes.append(None)

    def _check_node(self, node: int) -> None:
        if node < 0 or node >= len(self._nodes):
            raise IndexError(f"node index {node} is not in the graph")


class PyDiGraph(_BaseGraph):
    directed = True

    def __init__(self, check_cycle: bool = False, multigraph: bool = True, /):
        super().__init__(multigraph)
        self.check_cycle = bool(check_cycle)

    def add_edge(self, parent: int, child: int, obj, /) -> int:
        self._check_node(parent)
        self._check_node(child)
        if self.check_cycle and _reachable(self, child, parent):
            raise DAGHasCycle("adding this edge would create a cycle")
        return super().add_edge(parent, child, obj)

    def predecessor_indices(self, node: int, /):
        self._check_node(node)
        return [u for u, v, _ in reversed(self._edges) if v == node]


class PyGraph(_BaseGraph):
    directed = False


class PyDAG(PyDiGraph):
    def __init__(self, multigraph: bool = True, /):
        super().__init__(True, multigraph)


def _topology_csr(graph: _BaseGraph, undirected: bool = False):
    bidirectional = undirected or not graph.directed
    cached = graph._topology_cache.get(bidirectional)
    if cached is not None:
        return cached
    edges = graph._edges
    m = len(edges)
    sources = np.fromiter((u for u, _, _ in edges), dtype=np.int64, count=m)
    targets = np.fromiter((v for _, v, _ in edges), dtype=np.int64, count=m)
    if bidirectional:
        source_pairs = np.empty(m * 2, dtype=np.int64)
        target_pairs = np.empty(m * 2, dtype=np.int64)
        source_pairs[0::2], source_pairs[1::2] = sources, targets
        target_pairs[0::2], target_pairs[1::2] = targets, sources
        sources, targets = source_pairs, target_pairs
    order = np.argsort(sources, kind="stable")
    offsets = np.empty(len(graph) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(np.bincount(sources, minlength=len(graph)), out=offsets[1:])
    cached = (offsets, targets[order], order)
    graph._topology_cache[bidirectional] = cached
    return cached


def _csr(graph: _BaseGraph, weight_fn: Callable | None = None, default_weight: float = 1.0, undirected: bool = False):
    offsets, targets, order = _topology_csr(graph, undirected)
    if weight_fn is None:
        costs = np.full(len(targets), default_weight, dtype=np.float64)
    else:
        values = np.fromiter((float(weight_fn(data)) for _, _, data in graph._edges), dtype=np.float64, count=len(graph._edges))
        if undirected or not graph.directed:
            values = np.repeat(values, 2)
        costs = values[order]
    return offsets, targets, costs


def _weighted_csr(graph, weight_fn, default_weight, undirected=False, allow_negative=False):
    arrays = _csr(graph, weight_fn, default_weight, undirected)
    if not allow_negative and (np.any(arrays[2] < 0) or np.isnan(arrays[2]).any()):
        raise ValueError("edge weights must be non-negative and not NaN")
    return arrays


def _dijkstra(graph, source, weight_fn, default_weight, undirected=False, goal=None):
    graph._check_node(source)
    if goal is not None:
        graph._check_node(goal)
    offsets, targets, order = _topology_csr(graph, undirected)
    n = len(graph)
    dist = np.empty(n, dtype=np.float64)
    pred = np.empty(n, dtype=np.int64)
    heap_capacity = max(n, len(targets)) + 2
    heap_nodes = np.empty(heap_capacity, dtype=np.int64)
    heap_values = np.empty(heap_capacity, dtype=np.float64)
    if weight_fn is not None and len(targets) >= 16_384:
        costs = np.empty(len(targets), dtype=np.float64)
        active_count = lib().mrx_reachable_edges(
            addr(offsets), addr(targets), addr(costs), addr(pred), addr(heap_nodes), n, source,
        )
        active = costs[:active_count].astype(np.int64)
        original = order[active]
        if undirected or not graph.directed:
            original = original // 2
        values = np.fromiter(
            (float(weight_fn(graph._edges[index][2])) for index in original),
            dtype=np.float64,
            count=active_count,
        )
        if np.any(values < 0) or np.isnan(values).any():
            raise ValueError("edge weights must be non-negative and not NaN")
        costs[active] = values
    else:
        _, _, costs = _csr(graph, weight_fn, default_weight, undirected)
        if np.any(costs < 0) or np.isnan(costs).any():
            raise ValueError("edge weights must be non-negative and not NaN")
    ok = lib().mrx_dijkstra(addr(offsets), addr(targets), addr(costs), addr(dist), addr(pred), addr(heap_nodes), addr(heap_values), n, source, -1 if goal is None else goal)
    if not ok:
        raise ValueError("invalid source")
    return dist, pred


def _paths_from(dist, pred, source, goal=None):
    reachable = (pred >= 0)
    reachable[source] = True
    destinations = [goal] if goal is not None and reachable[goal] else np.flatnonzero(reachable).tolist()
    answer = {}
    for target in destinations:
        if target == source:
            continue
        path = [int(target)]
        while path[-1] != source:
            parent = int(pred[path[-1]])
            if parent < 0:
                break
            path.append(parent)
        if path[-1] == source:
            answer[int(target)] = list(reversed(path))
    return answer


def _lengths(graph, source, edge_cost_fn, goal=None, undirected=False):
    dist, pred = _dijkstra(graph, source, edge_cost_fn, 1.0, undirected, goal)
    if goal is not None:
        return {goal: float(dist[goal])} if goal == source or pred[goal] >= 0 else {}
    # A predecessor records reachability even when the shortest length is inf.
    # Reuse the predecessor returned by the kernel rather than a finite-value
    # sentinel: Float64's largest finite value is a valid result.
    return {int(v): float(dist[v]) for v in np.flatnonzero(pred >= 0) if v != source}


def digraph_dijkstra_shortest_path_lengths(graph, node, edge_cost_fn, /, goal=None):
    return _lengths(graph, node, edge_cost_fn, goal)


def graph_dijkstra_shortest_path_lengths(graph, node, edge_cost_fn, /, goal=None):
    return _lengths(graph, node, edge_cost_fn, goal)


def _shortest_paths(graph, source, target=None, weight_fn=None, default_weight=1.0, as_undirected=False):
    dist, pred = _dijkstra(graph, source, weight_fn, default_weight, as_undirected, target)
    return _paths_from(dist, pred, source, target)


def digraph_dijkstra_shortest_paths(graph, source, /, target=None, weight_fn=None, default_weight=1.0, as_undirected=False):
    return _shortest_paths(graph, source, target, weight_fn, default_weight, as_undirected)


def graph_dijkstra_shortest_paths(graph, source, /, target=None, weight_fn=None, default_weight=1.0):
    return _shortest_paths(graph, source, target, weight_fn, default_weight)


def _all_lengths(graph, edge_cost_fn):
    return {source: _lengths(graph, source, edge_cost_fn) for source in graph.node_indices()}


def digraph_all_pairs_dijkstra_path_lengths(graph, edge_cost_fn, /):
    return _all_lengths(graph, edge_cost_fn)


def graph_all_pairs_dijkstra_path_lengths(graph, edge_cost_fn, /):
    return _all_lengths(graph, edge_cost_fn)


def _all_paths(graph, edge_cost_fn):
    return {source: _shortest_paths(graph, source, weight_fn=edge_cost_fn) for source in graph.node_indices()}


def digraph_all_pairs_dijkstra_shortest_paths(graph, edge_cost_fn, /):
    return _all_paths(graph, edge_cost_fn)


def graph_all_pairs_dijkstra_shortest_paths(graph, edge_cost_fn, /):
    return _all_paths(graph, edge_cost_fn)


def _floyd(graph, weight_fn=None, default_weight=1.0, undirected=False):
    offsets, targets, costs = _weighted_csr(graph, weight_fn, default_weight, undirected, allow_negative=True)
    matrix = np.empty((len(graph), len(graph)), dtype=np.float64)
    lib().mrx_floyd_warshall(addr(offsets), addr(targets), addr(costs), addr(matrix), len(graph))
    return matrix


def digraph_floyd_warshall_numpy(graph, /, weight_fn=None, as_undirected=False, default_weight=1.0, parallel_threshold=300):
    return _floyd(graph, weight_fn, default_weight, as_undirected)


def graph_floyd_warshall_numpy(graph, /, weight_fn=None, default_weight=1.0, parallel_threshold=300):
    return _floyd(graph, weight_fn, default_weight)


def topological_sort(graph, /):
    if graph._has_self_loop:
        raise DAGHasCycle("graph contains a cycle")
    offsets, targets, _ = _topology_csr(graph)
    n = len(graph)
    indegree = np.empty(n, dtype=np.int64)
    queue = np.empty(n, dtype=np.int64)
    order = np.empty(n, dtype=np.int64)
    count = lib().mrx_toposort(addr(offsets), addr(targets), addr(indegree), addr(queue), addr(order), n)
    if count != n:
        raise DAGHasCycle("graph contains a cycle")
    return order.tolist()


def is_directed_acyclic_graph(graph, /) -> bool:
    if graph._has_self_loop:
        return False
    try:
        topological_sort(graph)
        return True
    except DAGHasCycle:
        return False


def _adjacency(graph, undirected=False):
    adjacent = [[] for _ in range(len(graph))]
    for u, v, _ in graph._edges:
        adjacent[u].append(v)
        if undirected or not graph.directed:
            adjacent[v].append(u)
    return adjacent


def _reachable(graph, source, target):
    todo, seen, adjacent = [source], {source}, _adjacency(graph)
    while todo:
        node = todo.pop()
        if node == target:
            return True
        for nxt in adjacent[node]:
            if nxt not in seen:
                seen.add(nxt)
                todo.append(nxt)
    return False


def _dfs_edges(graph, source=None):
    if not len(graph):
        return []
    source = 0 if source is None else source
    graph._check_node(source)
    adjacent, seen, result = _adjacency(graph), {source}, []
    stack = [(source, 0)]
    while stack:
        node, index = stack[-1]
        if index == len(adjacent[node]):
            stack.pop()
            continue
        nxt = adjacent[node][index]
        stack[-1] = (node, index + 1)
        if nxt not in seen:
            seen.add(nxt)
            result.append((node, nxt))
            stack.append((nxt, 0))
    return result


def digraph_dfs_edges(graph, /, source=None):
    return _dfs_edges(graph, source)


def graph_dfs_edges(graph, /, source=None):
    return _dfs_edges(graph, source)


def _components(graph, undirected=True):
    adjacent, unseen, result = _adjacency(graph, undirected), set(range(len(graph))), []
    while unseen:
        root, component, todo = min(unseen), set(), [min(unseen)]
        unseen.remove(root)
        while todo:
            node = todo.pop()
            component.add(node)
            for nxt in adjacent[node]:
                if nxt in unseen:
                    unseen.remove(nxt)
                    todo.append(nxt)
        result.append(component)
    return result


def connected_components(graph, /):
    return _components(graph)


def number_connected_components(graph, /):
    return len(connected_components(graph))


def weakly_connected_components(graph, /):
    return _components(graph)


def number_weakly_connected_components(graph, /):
    return len(weakly_connected_components(graph))


def is_weakly_connected(graph, /):
    return len(graph) == 0 or number_weakly_connected_components(graph) == 1


def strongly_connected_components(graph, /):
    adjacent = _adjacency(graph)
    reverse = [[] for _ in range(len(graph))]
    for u, row in enumerate(adjacent):
        for v in row:
            reverse[v].append(u)
    visited, finish = set(), []
    def visit(node):
        visited.add(node)
        for nxt in adjacent[node]:
            if nxt not in visited:
                visit(nxt)
        finish.append(node)
    for node in range(len(graph)):
        if node not in visited:
            visit(node)
    visited, result = set(), []
    def collect(node, component):
        visited.add(node)
        component.append(node)
        for nxt in reverse[node]:
            if nxt not in visited:
                collect(nxt, component)
    for node in reversed(finish):
        if node not in visited:
            component = []
            collect(node, component)
            result.append(component)
    return result


def number_strongly_connected_components(graph, /):
    return len(strongly_connected_components(graph))
