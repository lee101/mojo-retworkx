# mojo-retworkx

`mojo-retworkx` is a typed Mojo core for the numeric graph-algorithm subset of
[retworkx](https://pypi.org/project/retworkx/), exposed through a small Python
API. It preserves retworkx's integer-indexed graph model and the names and
call signatures of the covered functions, while keeping arbitrary node and
edge payloads in Python.

The upstream `retworkx` compatibility package is available on PyPI (version
0.17.1 in this environment), although it emits a deprecation warning because
the project was renamed to `rustworkx`. The test suite compares directly
against that package.

## Covered subset

| area | API |
| --- | --- |
| graph containers | `PyGraph`, `PyDiGraph`, `PyDAG`; node/edge insertion, edge lists, adjacency queries, payload indexing |
| sparse shortest paths | directed and undirected Dijkstra lengths and paths, including all-pairs variants |
| dense shortest paths | directed and undirected `*_floyd_warshall_numpy` |
| directed structure | `topological_sort`, `is_directed_acyclic_graph`, weak/strong components |
| traversal | `graph_dfs_edges`, `digraph_dfs_edges`, connected-component counts |

Not yet covered: node or edge removal, graph substitution and serialization,
centrality, coloring, matching, flow, isomorphism, generators, visitor-based
search, and the many specialized DAG algorithms. This is intentionally a
numeric, typed-core port rather than a claim of complete retworkx coverage.

## Install and use

```bash
pixi install
pixi run build
pixi run test
pixi run bench
```

Within `pixi run`, `python/` is on `PYTHONPATH`:

```python
import mojo_retworx as rx

graph = rx.PyDiGraph()
graph.add_nodes_from(["start", "parse", "build", "package"])
graph.add_edges_from([
    (0, 1, 2.0), (0, 2, 7.0), (1, 2, 1.0), (2, 3, 3.0),
])

print(rx.digraph_dijkstra_shortest_path_lengths(graph, 0, lambda weight: weight))
# {1: 2.0, 2: 3.0, 3: 6.0}
print(rx.topological_sort(graph))
# [0, 1, 2, 3]
```

## Performance

Measured by `pixi run bench` on 2026-08-27, Intel Xeon E5-2697 v4 (72 logical
CPUs), Linux 6.8, Mojo 1.1.0.dev2026081105, against retworkx 0.17.1. Values
are the best of three calls and include the Python-to-CSR preparation done by
this implementation.

| case | mojo-retworkx | retworkx | ratio |
| --- | ---: | ---: | --- |
| Dijkstra lengths (30k nodes, 180k edges) | 5.8 ms | 1.1 ms | 0.18x slower |
| Floyd-Warshall (260 nodes, 2,080 edges) | 1.8 ms | 18.1 ms | 10.07x faster |
| topological sort on cyclic graph (100k, 300k) | 0.2 us | 2.2 us | 12.24x faster |

The topology CSR is cached until graph mutation, so repeated algorithms reuse
the zero-copy offsets and targets buffers. Large weighted Dijkstra calls use a
typed reachability prepass and evaluate Python edge-cost callbacks only for CSR
rows reachable from the source. SIMD stores initialize distance, predecessor,
visited, and indegree buffers with scalar tail handling; Floyd–Warshall retains
its SIMD row updates. Dijkstra, reachability, and Kahn traversal are dependent
or contention-heavy, so no parallel path was added. No GPU path is included:
Floyd–Warshall is the only compute-dense kernel and is already 10x ahead at the
benchmark's 260-node, roughly 0.5 MiB matrix, where transfer and launch overhead
would dominate. No benchmark numbers are estimated; the table is copied from
the command above.

## How it works

`PyGraph` and `PyDiGraph` retain Python payload objects and edge records. At an
algorithm boundary, the wrapper calls the supplied weight function once per
edge and packs an adjacency graph into CSR: `offsets[n + 1]`, `targets[m]`, and
`float64 costs[m]`. The shared library at `dist/libmojo-retworkx.so` receives
the NumPy buffers as integer addresses through ctypes, rebuilds typed Mojo
`UnsafePointer` values, and writes only to caller-owned result and scratch
arrays. Nothing is allocated across the FFI boundary.

The Mojo compilation unit contains a binary-heap Dijkstra, in-place
Floyd–Warshall, and Kahn topological sort. This keeps the performance-sensitive
loops typed and contiguous without imposing a foreign memory owner on Python.

## License

MIT. See [LICENSE](LICENSE).
