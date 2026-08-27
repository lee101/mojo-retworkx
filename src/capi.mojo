"""Dense graph kernels over caller-owned CSR buffers."""

from std.sys.info import simd_width_of as simdwidthof

comptime FloatPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IntPtr = UnsafePointer[Int, AnyOrigin[mut=True]]


def fp(addr: Int) -> FloatPtr:
    return FloatPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IntPtr:
    return IntPtr(unsafe_from_address=addr)


def heap_push(nodes: IntPtr, values: FloatPtr, size: Int, node: Int, value: Float64):
    var child = size
    while child > 0:
        var parent = (child - 1) // 2
        if values[parent] <= value:
            break
        nodes[child] = nodes[parent]
        values[child] = values[parent]
        child = parent
    nodes[child] = node
    values[child] = value


def heap_pop(nodes: IntPtr, values: FloatPtr, size: Int, node_out: IntPtr, value_out: FloatPtr):
    node_out[0] = nodes[0]
    value_out[0] = values[0]
    var last_node = nodes[size - 1]
    var last_value = values[size - 1]
    var parent = 0
    while parent * 2 + 1 < size - 1:
        var child = parent * 2 + 1
        if child + 1 < size - 1 and values[child + 1] < values[child]:
            child += 1
        if values[child] >= last_value:
            break
        nodes[parent] = nodes[child]
        values[parent] = values[child]
        parent = child
    nodes[parent] = last_node
    values[parent] = last_value


@export("mrx_reachable_edges")
def mrx_reachable_edges(
    offsets_addr: Int, targets_addr: Int, edge_indices_addr: Int,
    visited_addr: Int, stack_addr: Int, n: Int, source: Int,
) abi("C") -> Int:
    var offsets = ip(offsets_addr)
    var targets = ip(targets_addr)
    var edge_indices = fp(edge_indices_addr)
    var visited = ip(visited_addr)
    var stack = ip(stack_addr)
    comptime W = simdwidthof[DType.int]()
    var i = 0
    var zeros = SIMD[DType.int, W](0)
    while i + W <= n:
        visited.store(i, zeros)
        i += W
    while i < n:
        visited[i] = 0
        i += 1
    if source < 0 or source >= n:
        return 0
    visited[source] = 1
    var stack_size = 1
    stack[0] = source
    var edge_count = 0
    while stack_size > 0:
        stack_size -= 1
        var u = stack[stack_size]
        for e in range(offsets[u], offsets[u + 1]):
            edge_indices[edge_count] = Float64(e)
            edge_count += 1
            var v = targets[e]
            if visited[v] == 0:
                visited[v] = 1
                stack[stack_size] = v
                stack_size += 1
    return edge_count


@export("mrx_dijkstra")
def mrx_dijkstra(
    offsets_addr: Int, targets_addr: Int, costs_addr: Int,
    dist_addr: Int, pred_addr: Int, heap_nodes_addr: Int, heap_values_addr: Int,
    n: Int, source: Int, goal: Int,
) abi("C") -> Int:
    var offsets = ip(offsets_addr)
    var targets = ip(targets_addr)
    var costs = fp(costs_addr)
    var dist = fp(dist_addr)
    var pred = ip(pred_addr)
    var heap_nodes = ip(heap_nodes_addr)
    var heap_values = fp(heap_values_addr)
    # Use IEEE infinity rather than the largest finite Float64: the latter is
    # a valid edge cost and must not be mistaken for an unreachable distance.
    var zero = 0.0
    var inf = 1.0 / zero
    comptime W = simdwidthof[DType.float64]()
    var i = 0
    var infs = SIMD[DType.float64, W](inf)
    var no_predecessors = SIMD[DType.int, W](-1)
    while i + W <= n:
        dist.store(i, infs)
        pred.store(i, no_predecessors)
        i += W
    while i < n:
        dist[i] = inf
        pred[i] = -1
        i += 1
    if source < 0 or source >= n:
        return 0
    dist[source] = 0.0
    var heap_size = 1
    heap_nodes[0] = source
    heap_values[0] = 0.0
    while heap_size > 0:
        heap_pop(heap_nodes, heap_values, heap_size, heap_nodes + heap_size, heap_values + heap_size)
        heap_size -= 1
        var u = heap_nodes[heap_size + 1]
        var current = heap_values[heap_size + 1]
        if current != dist[u]:
            continue
        if goal >= 0 and u == goal:
            return 1
        for e in range(offsets[u], offsets[u + 1]):
            var v = targets[e]
            var candidate = current + costs[e]
            # An infinite path is still a path.  `pred` distinguishes the
            # initial infinity of an unseen vertex from an infinity reached
            # through an edge, without repeatedly enqueueing infinity cycles.
            if candidate < dist[v] or (candidate == inf and pred[v] < 0 and v != source):
                dist[v] = candidate
                pred[v] = u
                heap_push(heap_nodes, heap_values, heap_size, v, candidate)
                heap_size += 1
    return 1


@export("mrx_floyd_warshall")
def mrx_floyd_warshall(
    offsets_addr: Int, targets_addr: Int, costs_addr: Int, matrix_addr: Int,
    n: Int,
) abi("C"):
    var offsets = ip(offsets_addr)
    var targets = ip(targets_addr)
    var costs = fp(costs_addr)
    var matrix = fp(matrix_addr)
    var zero = 0.0
    var inf = 1.0 / zero
    for i in range(n):
        for j in range(n):
            matrix[i * n + j] = 0.0 if i == j else inf
    for u in range(n):
        for e in range(offsets[u], offsets[u + 1]):
            var v = targets[e]
            if costs[e] < matrix[u * n + v]:
                matrix[u * n + v] = costs[e]
    for k in range(n):
        for i in range(n):
            var through_k = matrix[i * n + k]
            if through_k == inf:
                continue
            comptime W = simdwidthof[DType.float64]()
            var j = 0
            var through = SIMD[DType.float64, W](through_k)
            while j + W <= n:
                var candidate = through + matrix.load[width=W](k * n + j)
                matrix.store(i * n + j, min(matrix.load[width=W](i * n + j), candidate))
                j += W
            while j < n:
                var candidate = through_k + matrix[k * n + j]
                if candidate < matrix[i * n + j]:
                    matrix[i * n + j] = candidate
                j += 1


@export("mrx_toposort")
def mrx_toposort(
    offsets_addr: Int, targets_addr: Int, indegree_addr: Int, queue_addr: Int,
    order_addr: Int, n: Int,
) abi("C") -> Int:
    var offsets = ip(offsets_addr)
    var targets = ip(targets_addr)
    var indegree = ip(indegree_addr)
    var queue = ip(queue_addr)
    var order = ip(order_addr)
    comptime W = simdwidthof[DType.int]()
    var i = 0
    var zeros = SIMD[DType.int, W](0)
    while i + W <= n:
        indegree.store(i, zeros)
        i += W
    while i < n:
        indegree[i] = 0
        i += 1
    for u in range(n):
        for e in range(offsets[u], offsets[u + 1]):
            indegree[targets[e]] += 1
    var tail = 0
    for i in range(n):
        if indegree[i] == 0:
            queue[tail] = i
            tail += 1
    var head = 0
    while head < tail:
        var u = queue[head]
        head += 1
        order[head - 1] = u
        for e in range(offsets[u], offsets[u + 1]):
            var v = targets[e]
            indegree[v] -= 1
            if indegree[v] == 0:
                queue[tail] = v
                tail += 1
    return tail
