"""Run with `pixi run bench`; the Pixi task serializes machine-wide timing."""

from __future__ import annotations

import math
import time
import warnings

import mojo_retworx as mrx
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import retworkx as rx


def timeit(fn, repeat=3):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def graph(cls, n, span, seed=7):
    result = cls()
    result.add_nodes_from(range(n))
    edges = []
    state = seed
    for u in range(n):
        for step in range(1, span + 1):
            v = (u * 1103515245 + step * 12345 + state) % n
            edges.append((u, v, float((u + step) % 17 + 1)))
    result.add_edges_from(edges)
    return result


def report(name, ours, theirs):
    ours(); theirs()
    a, b = timeit(ours), timeit(theirs)
    verdict = "faster" if a < b else "slower"
    def duration(seconds):
        return f"{seconds * 1e3:.1f} ms" if seconds >= 1e-3 else f"{seconds * 1e6:.1f} us"
    ratio = b / a
    ratio_text = "<0.01x" if ratio < 0.01 else f"{ratio:.2f}x"
    print(f"| {name} | {duration(a)} | {duration(b)} | {ratio_text} {verdict} |")


def main():
    print("| case | mojo-retworkx | retworkx | ratio |")
    print("| --- | ---: | ---: | --- |")
    ours, theirs = graph(mrx.PyDiGraph, 30_000, 6), graph(rx.PyDiGraph, 30_000, 6)
    report("Dijkstra lengths (30k nodes, 180k edges)", lambda: mrx.digraph_dijkstra_shortest_path_lengths(ours, 0, lambda x: x), lambda: rx.digraph_dijkstra_shortest_path_lengths(theirs, 0, lambda x: x))
    ours, theirs = graph(mrx.PyDiGraph, 260, 8), graph(rx.PyDiGraph, 260, 8)
    report("Floyd-Warshall (260 nodes, 2,080 edges)", lambda: mrx.digraph_floyd_warshall_numpy(ours, weight_fn=lambda x: x), lambda: rx.digraph_floyd_warshall_numpy(theirs, weight_fn=lambda x: x))
    ours, theirs = graph(mrx.PyDiGraph, 100_000, 3), graph(rx.PyDiGraph, 100_000, 3)
    report("topological sort on cyclic graph (100k, 300k)", lambda: mrx.is_directed_acyclic_graph(ours), lambda: rx.is_directed_acyclic_graph(theirs))


if __name__ == "__main__":
    main()
