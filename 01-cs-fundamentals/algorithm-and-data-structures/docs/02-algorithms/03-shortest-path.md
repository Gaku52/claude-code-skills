# Shortest Path Algorithms

> Systematically understand techniques for solving shortest path problems in weighted graphs using the three major algorithms: Dijkstra, Bellman-Ford, and Floyd-Warshall

## What You Will Learn

1. Efficiently find single-source shortest paths using **Dijkstra's algorithm** with a priority queue implementation
2. Handle negative edge weights and detect negative cycles using **the Bellman-Ford algorithm**
3. Compute all-pairs shortest paths using **the Floyd-Warshall algorithm** with dynamic programming
4. Understand advanced algorithms such as **Johnson's algorithm**, **0-1 BFS**, and **A* search**
5. Accurately determine the applicability and appropriate selection of each algorithm


## Prerequisites

Having the following knowledge will deepen your understanding of this guide:

- Basic programming knowledge
- Understanding of related foundational concepts
- Familiarity with the content in [Graph Traversal Algorithms](./02-graph-traversal.md)

---

## 1. Classification of Shortest Path Problems

```
+-----------------------------------------------------+
|            Shortest Path Problems                     |
+-------------------+---------------------------------+
|  Single-Source     |  All-Pairs                       |
|  (SSSP)           |  (APSP)                         |
+-------------------+---------------------------------+
| Dijkstra          | Floyd-Warshall                   |
| (no negative      | O(V^3)                           |
|  edges)           |                                   |
| O((V+E) log V)   |                                   |
+-------------------+                                   |
| Bellman-Ford      | Johnson                           |
| (negative edges   | (for sparse graphs)               |
|  allowed)         |                                   |
| O(VE)             | O(V^2 log V + VE)                 |
+-------------------+                                   |
| DAG Shortest Path |                                   |
| (DAG only)        |                                   |
| O(V + E)          |                                   |
+-------------------+                                   |
| A* Search         |                                   |
| (heuristic)       |                                   |
| O(E) best case    |                                   |
+-------------------+---------------------------------+
```

### Foundational Concept for Shortest Path Problems

```
Relaxation -- the fundamental operation common to all algorithms:

  if dist[u] + weight(u, v) < dist[v]:
      dist[v] = dist[u] + weight(u, v)
      prev[v] = u

  "If going to v via u is shorter, update v's distance"

  This operation is performed:
  - Dijkstra  -> each time a vertex is finalized
  - Bellman   -> V-1 times over all edges
  - Floyd     -> while incrementally adding relay vertices
  - DAG       -> in topological order
```

---

## 2. Dijkstra's Algorithm

Finds single-source shortest paths in a graph with non-negative edge weights. The core idea is to repeatedly "finalize the unvisited vertex with the shortest known distance."

```
Graph:
      2        3
  A -----> B -----> D
  |        ^        ^
  |1       |1       |1
  v        |        |
  C -----> E -----> F
      4        2

Shortest distances from source A:
  Step 0: A=0, B=inf, C=inf, D=inf, E=inf, F=inf
  Step 1: A=0* -> C=1, B=2
  Step 2: C=1* -> E=5
  Step 3: B=2* -> D=5, E=min(5,3)=3
  Step 4: E=3* -> F=5
  Step 5: D=5* (via B: 2+3=5)
  Step 6: F=5* (via E: 3+2=5)

  Shortest distances: A=0, B=2, C=1, D=5, E=3, F=5
```

### 2.1 Priority Queue Implementation

```python
import heapq
from collections import defaultdict

def dijkstra(graph: dict, start: str) -> tuple:
    """Dijkstra's algorithm - O((V+E) log V)
    graph: {u: [(v, weight), ...], ...}
    Returns: (distance dict, predecessor dict)
    """
    dist = defaultdict(lambda: float('inf'))
    prev = {}
    dist[start] = 0

    # Min-heap of (distance, vertex)
    pq = [(0, start)]

    while pq:
        d, u = heapq.heappop(pq)

        # Skip stale entries
        if d > dist[u]:
            continue

        for v, weight in graph.get(u, []):
            new_dist = dist[u] + weight
            if new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(pq, (new_dist, v))

    return dict(dist), prev

def reconstruct_path(prev: dict, start: str, end: str) -> list:
    """Reconstruct path from predecessor dictionary"""
    path = []
    current = end
    while current != start:
        if current not in prev:
            return []  # Unreachable
        path.append(current)
        current = prev[current]
    path.append(start)
    return path[::-1]

# Usage example
graph = {
    'A': [('B', 2), ('C', 1)],
    'B': [('D', 3)],
    'C': [('E', 4)],
    'E': [('B', 1), ('F', 2)],
    'F': [('D', 1)],
}
dist, prev = dijkstra(graph, 'A')
print(dist)                            # {'A': 0, 'B': 2, 'C': 1, 'D': 5, 'E': 5, 'F': 7}
print(reconstruct_path(prev, 'A', 'D'))  # ['A', 'B', 'D']
```

### 2.2 Proof of Correctness for Dijkstra's Algorithm (Overview)

```
Greedy choice property of Dijkstra's algorithm:

  Theorem: In a non-negative edge graph, when we finalize vertex u
           with the smallest distance among unfinalized vertices,
           dist[u] is the true shortest distance to u.

  Proof outline (by induction and contradiction):
  1. Assume dist[u] is not the shortest
  2. Then a shorter path via another unfinalized vertex w exists
  3. But w is unfinalized, so dist[w] >= dist[u]
  4. Since edge weights are non-negative, the path via w is >= dist[u]
  5. Contradiction -> dist[u] is the shortest distance

  This proof holds under the condition: all edge weights are non-negative
  With negative edges, Step 4 fails
```

### 2.3 Variations of Dijkstra's Algorithm

#### Visited Flag Version (Memory-Efficient)

```python
def dijkstra_with_visited(graph: dict, start: str) -> dict:
    """Dijkstra with visited flag (fewer stale entries accumulate in the heap)"""
    dist = defaultdict(lambda: float('inf'))
    dist[start] = 0
    visited = set()
    pq = [(0, start)]

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)

        for v, weight in graph.get(u, []):
            if v not in visited:
                new_dist = d + weight
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    heapq.heappush(pq, (new_dist, v))

    return dict(dist)
```

#### Finding the k-th Shortest Path with Dijkstra

```python
def dijkstra_kth_shortest(graph: dict, start: str, end: str, k: int) -> list:
    """Return distances of the k shortest paths
    Continue exploring until each vertex has been reached at most k times
    """
    count = defaultdict(int)  # Number of times each vertex has been reached
    pq = [(0, start)]
    distances = []

    while pq:
        d, u = heapq.heappop(pq)
        count[u] += 1

        if u == end:
            distances.append(d)
            if len(distances) == k:
                return distances

        if count[u] > k:
            continue

        for v, weight in graph.get(u, []):
            heapq.heappush(pq, (d + weight, v))

    return distances

# Usage example
graph_multi = {
    'A': [('B', 1), ('C', 3)],
    'B': [('C', 1), ('D', 4)],
    'C': [('D', 1)],
    'D': [],
}
print(dijkstra_kth_shortest(graph_multi, 'A', 'D', 3))
# [3, 4, 5]  (A->B->C->D=3, A->C->D=4, A->B->D=5)
```

### 2.4 Dijkstra on Grids

```python
def dijkstra_grid(grid: list, start: tuple, end: tuple) -> int:
    """Dijkstra on a 2D grid (where each cell has a traversal cost)"""
    rows, cols = len(grid), len(grid[0])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    dist = [[float('inf')] * cols for _ in range(rows)]
    dist[start[0]][start[1]] = grid[start[0]][start[1]]

    pq = [(grid[start[0]][start[1]], start[0], start[1])]

    while pq:
        d, r, c = heapq.heappop(pq)

        if (r, c) == end:
            return d

        if d > dist[r][c]:
            continue

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                new_dist = d + grid[nr][nc]
                if new_dist < dist[nr][nc]:
                    dist[nr][nc] = new_dist
                    heapq.heappush(pq, (new_dist, nr, nc))

    return -1

# Traversal cost per cell
cost_grid = [
    [1, 3, 1, 2],
    [1, 5, 1, 1],
    [4, 2, 1, 3],
    [1, 1, 1, 1],
]
print(dijkstra_grid(cost_grid, (0, 0), (3, 3)))  # 7
```

---

## 3. Bellman-Ford Algorithm

A single-source shortest path algorithm that allows negative edge weights. It relaxes all edges V-1 times, then detects negative cycles on the V-th iteration.

```
The concept of relaxation:
  if dist[u] + weight(u,v) < dist[v]:
      dist[v] = dist[u] + weight(u,v)

  A --(5)--> B
  |           |
 (2)        (-3)
  |           |
  C --(4)--> D

  Step 0: A=0, B=inf, C=inf, D=inf
  Step 1: A->B: B=5, A->C: C=2
  Step 2: B->D: D=2, C->D: D=min(2,6)=2
  Step 3: No change -> converged

Why V-1 iterations are sufficient:
  The shortest path contains at most V-1 edges (when cycle-free).
  Each iteration finalizes at least one edge of the shortest path.
  Therefore, V-1 iterations find all shortest paths.
```

### 3.1 Basic Implementation

```python
def bellman_ford(vertices: list, edges: list, start) -> tuple:
    """Bellman-Ford algorithm - O(VE)
    edges: [(u, v, weight), ...]
    Returns: (distance dict, predecessor dict, negative cycle flag)
    """
    dist = {v: float('inf') for v in vertices}
    prev = {v: None for v in vertices}
    dist[start] = 0

    # V-1 relaxation passes
    for _ in range(len(vertices) - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
                updated = True
        if not updated:  # Early termination
            break

    # V-th pass: detect negative cycles
    has_negative_cycle = False
    for u, v, w in edges:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            has_negative_cycle = True
            break

    return dist, prev, has_negative_cycle

# Usage example (with negative edges)
vertices = ['A', 'B', 'C', 'D', 'E']
edges = [
    ('A', 'B', 4), ('A', 'C', 2),
    ('B', 'D', 3), ('C', 'B', -1),
    ('C', 'D', 5), ('D', 'E', 1),
]
dist, prev, neg_cycle = bellman_ford(vertices, edges, 'A')
print(dist)       # {'A': 0, 'B': 1, 'C': 2, 'D': 4, 'E': 5}
print(neg_cycle)  # False

# Negative cycle example
edges_neg = [('A', 'B', 1), ('B', 'C', -3), ('C', 'A', 1)]
_, _, neg = bellman_ford(['A','B','C'], edges_neg, 'A')
print(neg)  # True
```

### 3.2 Identifying Vertices Affected by Negative Cycles

```python
def bellman_ford_with_negative_cycle_detection(vertices, edges, start):
    """Identify all vertices affected by negative cycles"""
    dist = {v: float('inf') for v in vertices}
    prev = {v: None for v in vertices}
    dist[start] = 0

    # V-1 relaxation passes
    for _ in range(len(vertices) - 1):
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u

    # Vertices updated in the V-th pass are affected by negative cycles
    affected = set()
    for u, v, w in edges:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            affected.add(v)

    # All vertices reachable from the negative cycle are also affected
    # Propagate via BFS
    queue = deque(affected)
    while queue:
        node = queue.popleft()
        for u, v, w in edges:
            if u == node and v not in affected:
                affected.add(v)
                queue.append(v)

    return dist, affected

vertices_nc = ['A', 'B', 'C', 'D', 'E']
edges_nc = [
    ('A', 'B', 1), ('B', 'C', -3), ('C', 'B', 1),  # Negative cycle between B-C
    ('C', 'D', 2), ('D', 'E', 1),
]
dist, affected = bellman_ford_with_negative_cycle_detection(vertices_nc, edges_nc, 'A')
print(f"Vertices affected by negative cycle: {affected}")
# {'B', 'C', 'D', 'E'} -- B,C are on the cycle; D,E are reachable from it
```

### 3.3 SPFA (Shortest Path Faster Algorithm)

An improvement over Bellman-Ford that only processes vertices whose distances have been updated. Faster on average, but worst case remains O(VE).

```python
from collections import deque

def spfa(graph: dict, start) -> tuple:
    """SPFA - improved Bellman-Ford
    graph: {u: [(v, weight), ...]}
    Significantly faster than Bellman-Ford on average
    """
    all_vertices = set(graph.keys())
    for neighbors in graph.values():
        for v, _ in neighbors:
            all_vertices.add(v)

    dist = {v: float('inf') for v in all_vertices}
    dist[start] = 0
    in_queue = {v: False for v in all_vertices}
    count = {v: 0 for v in all_vertices}  # Queue entry count per vertex

    queue = deque([start])
    in_queue[start] = True
    count[start] = 1

    while queue:
        u = queue.popleft()
        in_queue[u] = False

        for v, w in graph.get(u, []):
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                if not in_queue[v]:
                    queue.append(v)
                    in_queue[v] = True
                    count[v] += 1
                    if count[v] >= len(all_vertices):
                        return dist, True  # Negative cycle detected

    return dist, False

graph_spfa = {
    'A': [('B', 4), ('C', 2)],
    'B': [('D', 3)],
    'C': [('B', -1), ('D', 5)],
    'D': [('E', 1)],
    'E': [],
}
dist, has_neg = spfa(graph_spfa, 'A')
print(dist)      # {'A': 0, 'B': 1, 'C': 2, 'D': 4, 'E': 5}
print(has_neg)   # False
```

---

## 4. Floyd-Warshall Algorithm

Computes all-pairs shortest paths using dynamic programming. Updates the distance matrix by incrementally adding relay vertices.

```
Initial distance matrix:           k=1 (via A):
     A    B    C    D         A    B    C    D
A [  0,   3, inf,   7]   A [  0,   3, inf,   7]
B [  8,   0,   2, inf]   B [  8,   0,   2,  15]
C [  5, inf,   0,   1]   C [  5,   8,   0,   1]
D [  2, inf, inf,   0]   D [  2,   5, inf,   0]

k=2 (via B):              Final result:
     A    B    C    D         A    B    C    D
A [  0,   3,   5,   7]   A [  0,   3,   5,   6]
B [  8,   0,   2,   3]   B [  5,   0,   2,   3]
C [  5,   8,   0,   1]   C [  3,   6,   0,   1]
D [  2,   5,   7,   0]   D [  2,   5,   7,   0]

DP recurrence:
  dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

  "Compare the distance from i to j with and without using k as a relay vertex"
```

### 4.1 Basic Implementation

```python
def floyd_warshall(n: int, edges: list) -> list:
    """Floyd-Warshall algorithm - O(V^3)
    edges: [(u, v, weight), ...]
    Returns: distance matrix
    """
    INF = float('inf')
    dist = [[INF] * n for _ in range(n)]

    # Initialization
    for i in range(n):
        dist[i][i] = 0
    for u, v, w in edges:
        dist[u][v] = w

    # DP: incrementally add relay vertex k
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    return dist
```

### 4.2 Implementation with Path Reconstruction

```python
def floyd_warshall_with_path(n: int, edges: list) -> tuple:
    """Floyd-Warshall + path reconstruction"""
    INF = float('inf')
    dist = [[INF] * n for _ in range(n)]
    nxt = [[None] * n for _ in range(n)]

    for i in range(n):
        dist[i][i] = 0
    for u, v, w in edges:
        dist[u][v] = w
        nxt[u][v] = v

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    nxt[i][j] = nxt[i][k]

    return dist, nxt

def get_path(nxt: list, u: int, v: int) -> list:
    """Reconstruct the path"""
    if nxt[u][v] is None:
        return []
    path = [u]
    while u != v:
        u = nxt[u][v]
        path.append(u)
    return path

# Usage example
edges = [(0,1,3), (0,3,7), (1,0,8), (1,2,2), (2,0,5), (2,3,1), (3,0,2)]
dist, nxt = floyd_warshall_with_path(4, edges)
print(dist[1][3])           # 3
print(get_path(nxt, 1, 3))  # [1, 2, 3]
```

### 4.3 Negative Cycle Detection with Floyd-Warshall

```python
def floyd_warshall_with_negative_cycle(n: int, edges: list) -> tuple:
    """Floyd-Warshall + negative cycle detection"""
    INF = float('inf')
    dist = [[INF] * n for _ in range(n)]

    for i in range(n):
        dist[i][i] = 0
    for u, v, w in edges:
        dist[u][v] = min(dist[u][v], w)

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] != INF and dist[k][j] != INF:
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]

    # Negative diagonal element -> negative cycle exists
    has_negative_cycle = any(dist[i][i] < 0 for i in range(n))

    return dist, has_negative_cycle
```

### 4.4 Floyd-Warshall Application: Transitive Closure

```python
def transitive_closure(n: int, edges: list) -> list:
    """Transitive closure: compute a reachability matrix indicating
    whether vertex i can reach vertex j.
    A variant of Floyd-Warshall (managing reachability instead of weights)
    """
    reach = [[False] * n for _ in range(n)]

    for i in range(n):
        reach[i][i] = True
    for u, v in edges:
        reach[u][v] = True

    for k in range(n):
        for i in range(n):
            for j in range(n):
                reach[i][j] = reach[i][j] or (reach[i][k] and reach[k][j])

    return reach

edges_reach = [(0, 1), (1, 2), (2, 3)]
reach = transitive_closure(4, edges_reach)
print(reach[0][3])  # True (0->1->2->3)
print(reach[3][0])  # False (3 cannot reach 0)
```

---

## 5. DAG Shortest Path

In a DAG (Directed Acyclic Graph), shortest paths can be found in O(V+E) by relaxing edges in topological order.

```python
from collections import defaultdict, deque

def dag_shortest_path(graph: dict, start) -> dict:
    """Shortest path on a DAG - O(V + E)"""
    # Topological sort (Kahn's)
    in_degree = defaultdict(int)
    all_verts = set(graph.keys())
    for u in graph:
        for v, _ in graph[u]:
            in_degree[v] += 1
            all_verts.add(v)

    queue = deque([v for v in all_verts if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for neighbor, _ in graph.get(v, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Relax in topological order
    dist = {v: float('inf') for v in all_verts}
    dist[start] = 0
    for u in topo_order:
        if dist[u] != float('inf'):
            for v, w in graph.get(u, []):
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w

    return dist

dag = {
    'A': [('B', 2), ('C', 6)],
    'B': [('C', 1), ('D', 3)],
    'C': [('D', 1)],
    'D': [],
}
print(dag_shortest_path(dag, 'A'))  # {'A': 0, 'B': 2, 'C': 3, 'D': 4}
```

### DAG Longest Path

In a DAG, the longest path can also be found in O(V+E) by negating weights or using max. Used for critical path analysis.

```python
def dag_longest_path(graph: dict, start) -> dict:
    """Longest path on a DAG - O(V + E)
    Used for critical path analysis in project management
    """
    in_degree = defaultdict(int)
    all_verts = set(graph.keys())
    for u in graph:
        for v, _ in graph[u]:
            in_degree[v] += 1
            all_verts.add(v)

    queue = deque([v for v in all_verts if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for neighbor, _ in graph.get(v, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    dist = {v: float('-inf') for v in all_verts}
    dist[start] = 0

    for u in topo_order:
        if dist[u] != float('-inf'):
            for v, w in graph.get(u, []):
                if dist[u] + w > dist[v]:
                    dist[v] = dist[u] + w

    return dist

# Project task DAG (task name: [(next task, duration in days)])
project = {
    'Design':      [('Development', 5), ('Test Design', 3)],
    'Development': [('Testing', 8)],
    'Test Design': [('Testing', 2)],
    'Testing':     [('Release', 3)],
    'Release':     [],
}
longest = dag_longest_path(project, 'Design')
print(longest)
# {'Design': 0, 'Development': 5, 'Test Design': 3, 'Testing': 13, 'Release': 16}
# Critical path: Design -> Development -> Testing -> Release = 16 days
```

---

## 6. Johnson's Algorithm (All-Pairs Shortest Paths for Sparse Graphs)

Uses Bellman-Ford to reweight edges to non-negative values, then runs Dijkstra from each vertex. Faster than Floyd-Warshall for sparse graphs.

```python
def johnson(n: int, edges: list) -> list:
    """Johnson's algorithm - O(V^2 log V + VE)
    edges: [(u, v, weight), ...]
    """
    INF = float('inf')

    # Step 1: Add virtual vertex s with zero-weight edges to all vertices
    new_edges = edges + [(n, v, 0) for v in range(n)]

    # Step 2: Run Bellman-Ford from s to compute shortest distances h
    h = [INF] * (n + 1)
    h[n] = 0
    for _ in range(n):
        for u, v, w in new_edges:
            if h[u] != INF and h[u] + w < h[v]:
                h[v] = h[u] + w

    # Negative cycle check
    for u, v, w in new_edges:
        if h[u] != INF and h[u] + w < h[v]:
            raise ValueError("Negative cycle detected")

    # Step 3: Reweight edges to non-negative: w'(u,v) = w(u,v) + h[u] - h[v]
    reweighted_graph = defaultdict(list)
    for u, v, w in edges:
        new_w = w + h[u] - h[v]
        reweighted_graph[u].append((v, new_w))

    # Step 4: Run Dijkstra from each vertex
    dist = [[INF] * n for _ in range(n)]
    for s in range(n):
        d = dijkstra_array(reweighted_graph, s, n)
        for t in range(n):
            if d[t] != INF:
                # Restore original weights
                dist[s][t] = d[t] - h[s] + h[t]

    return dist

def dijkstra_array(graph, start, n):
    """Array-based Dijkstra (for Johnson's)"""
    INF = float('inf')
    dist = [INF] * n
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph.get(u, []):
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))
    return dist
```

---

## 7. A* Search

An extension of Dijkstra that adds a heuristic function. Finds shortest paths faster than Dijkstra when the goal is well-defined.

```
Dijkstra: f(n) = g(n)        <- actual cost from source
A*:       f(n) = g(n) + h(n) <- actual cost + estimated cost to goal

Conditions for heuristic h(n):
  - Admissibility: h(n) <= actual cost (never overestimates)
  - Consistency:   h(n) <= cost(n, n') + h(n')

  Admissibility guarantees an optimal solution
```

```python
def astar(graph: dict, start, goal, heuristic) -> tuple:
    """A* search
    graph: {u: [(v, weight), ...]}
    heuristic: h(node) -> estimated cost to goal
    Returns: (distance, path)
    """
    open_set = [(heuristic(start), 0, start)]  # (f, g, node)
    came_from = {}
    g_score = defaultdict(lambda: float('inf'))
    g_score[start] = 0
    closed = set()

    while open_set:
        f, g, current = heapq.heappop(open_set)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return g, path[::-1]

        if current in closed:
            continue
        closed.add(current)

        for neighbor, weight in graph.get(current, []):
            if neighbor in closed:
                continue
            tentative_g = g + weight
            if tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                f = tentative_g + heuristic(neighbor)
                heapq.heappush(open_set, (f, tentative_g, neighbor))

    return float('inf'), []

# A* search on a grid (using Manhattan distance as the heuristic)
def manhattan_distance(node, goal=(9, 9)):
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

# Build grid graph
def build_grid_graph(rows, cols, blocked=set()):
    graph = {}
    for r in range(rows):
        for c in range(cols):
            if (r, c) in blocked:
                continue
            neighbors = []
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in blocked:
                    neighbors.append(((nr, nc), 1))
            graph[(r, c)] = neighbors
    return graph

grid_graph = build_grid_graph(10, 10, blocked={(3,3), (3,4), (3,5)})
dist, path = astar(grid_graph, (0, 0), (9, 9), manhattan_distance)
print(f"Shortest distance: {dist}")  # 18
print(f"Path length: {len(path)}")
```

### A* vs Dijkstra Comparison

```
          Dijkstra                    A*
        +---------+             +---------+
        | o o o o |             | . . . . |
        | o o o o |             | . o o . |
     S -> o o o o -> G         S -> . o o -> G
        | o o o o |             | . o o . |
        | o o o o |             | . . . . |
        +---------+             +---------+
     Explored vertices: many    Explored vertices: few

  o = explored vertex
  . = unexplored vertex

  A* uses h(n) to focus the search toward the goal,
  achieving significantly faster performance on large graphs
```

---

## 8. Specialized Shortest Path Algorithms

### 8.1 0-1 BFS

When all edge weights are either 0 or 1, use a deque to find shortest distances in O(V+E).

```python
def bfs_01(graph: dict, start) -> dict:
    """0-1 BFS - O(V + E)
    graph: {u: [(v, weight), ...]}  weight is 0 or 1
    """
    dist = defaultdict(lambda: float('inf'))
    dist[start] = 0
    dq = deque([start])

    while dq:
        u = dq.popleft()
        for v, w in graph.get(u, []):
            new_dist = dist[u] + w
            if new_dist < dist[v]:
                dist[v] = new_dist
                if w == 0:
                    dq.appendleft(v)  # Weight 0 -> push to front
                else:
                    dq.append(v)      # Weight 1 -> push to back

    return dict(dist)
```

### 8.2 Bidirectional Dijkstra

Runs Dijkstra from both source and target, determining the shortest distance when the two search spaces overlap. Effective for point-to-point shortest paths on large graphs.

```python
def bidirectional_dijkstra(graph: dict, reverse_graph: dict, start, end) -> int:
    """Bidirectional Dijkstra
    graph: forward adjacency list
    reverse_graph: reverse adjacency list
    """
    INF = float('inf')

    dist_f = defaultdict(lambda: INF)
    dist_b = defaultdict(lambda: INF)
    dist_f[start] = 0
    dist_b[end] = 0

    pq_f = [(0, start)]
    pq_b = [(0, end)]

    visited_f = set()
    visited_b = set()

    best = INF

    while pq_f or pq_b:
        # Forward search
        if pq_f:
            d, u = heapq.heappop(pq_f)
            if d <= best:
                visited_f.add(u)
                for v, w in graph.get(u, []):
                    new_d = d + w
                    if new_d < dist_f[v]:
                        dist_f[v] = new_d
                        heapq.heappush(pq_f, (new_d, v))
                    if v in visited_b:
                        best = min(best, dist_f[v] + dist_b[v])

        # Backward search
        if pq_b:
            d, u = heapq.heappop(pq_b)
            if d <= best:
                visited_b.add(u)
                for v, w in reverse_graph.get(u, []):
                    new_d = d + w
                    if new_d < dist_b[v]:
                        dist_b[v] = new_d
                        heapq.heappush(pq_b, (new_d, v))
                    if v in visited_f:
                        best = min(best, dist_f[v] + dist_b[v])

        # Terminate when both minimum distances >= best
        min_f = pq_f[0][0] if pq_f else INF
        min_b = pq_b[0][0] if pq_b else INF
        if min_f + min_b >= best:
            break

    return best
```

### 8.3 Dial's Algorithm

When edge weights are small non-negative integers, operates in O(V + E + W_max) using buckets.

```python
def dial_shortest_path(graph: dict, start, n: int, max_weight: int) -> list:
    """Dial's algorithm - O(V + E + W_max * V)
    Effective when edge weights are non-negative integers in [0, max_weight]
    """
    INF = float('inf')
    dist = [INF] * n
    dist[start] = 0

    # Buckets (circular buffer)
    num_buckets = max_weight * n + 1
    buckets = [[] for _ in range(num_buckets)]
    buckets[0].append(start)

    idx = 0
    found = 0

    while found < n:
        while not buckets[idx % num_buckets]:
            idx += 1

        u = buckets[idx % num_buckets].pop()
        if dist[u] != idx:
            continue

        found += 1

        for v, w in graph.get(u, []):
            new_dist = dist[u] + w
            if new_dist < dist[v]:
                dist[v] = new_dist
                buckets[new_dist % num_buckets].append(v)

    return dist
```

---

## 9. Practical Application Patterns

### 9.1 Navigation Route Search

```python
# In real-world navigation systems, the following techniques are used:
#
# 1. Contraction Hierarchies (CH)
#    - Preprocessing "contracts" low-importance vertices and adds shortcut edges
#    - Queries use bidirectional Dijkstra on the preprocessed graph
#    - Preprocessing: O(n log n) ~ O(n^2), Query: milliseconds
#
# 2. ALT (A*, Landmarks, Triangle inequality)
#    - Precompute distances to landmark vertices
#    - Use the triangle inequality to compute heuristics
#    - Used as the heuristic for A*

def travel_time_dijkstra(road_network, start, end, departure_time):
    """Time-dependent route search (accounting for rush hours)"""
    dist = defaultdict(lambda: float('inf'))
    dist[start] = departure_time
    pq = [(departure_time, start)]

    while pq:
        time, u = heapq.heappop(pq)
        if u == end:
            return time - departure_time
        if time > dist[u]:
            continue
        for v, travel_time_func in road_network.get(u, []):
            # travel_time_func(t) returns the travel time at time t
            arrival = time + travel_time_func(time)
            if arrival < dist[v]:
                dist[v] = arrival
                heapq.heappush(pq, (arrival, v))
    return float('inf')
```

### 9.2 Network Routing

```python
def ospf_routing(network: dict, router_id: str) -> dict:
    """OSPF (Open Shortest Path First) routing table computation
    Real OSPF is based on Dijkstra's algorithm
    """
    dist, prev = dijkstra(network, router_id)

    routing_table = {}
    for dest in dist:
        if dest == router_id:
            continue
        # Find the next hop
        next_hop = dest
        while prev.get(next_hop) != router_id:
            next_hop = prev.get(next_hop)
            if next_hop is None:
                break
        routing_table[dest] = {
            'next_hop': next_hop,
            'cost': dist[dest],
            'path': reconstruct_path(prev, router_id, dest),
        }

    return routing_table
```

### 9.3 Currency Arbitrage Detection

```python
import math

def detect_arbitrage(currencies: list, rates: dict) -> list:
    """Detect currency arbitrage using Bellman-Ford (negative cycle detection)
    rates: {(from, to): rate, ...}

    Arbitrage: A -> B -> C -> A yields a profit
    Converting rates to -log(rate) transforms negative cycles into arbitrage opportunities
    """
    n = len(currencies)
    currency_idx = {c: i for i, c in enumerate(currencies)}

    # Build edge list (weight = -log(rate))
    edges = []
    for (src, dst), rate in rates.items():
        if rate > 0:
            edges.append((currency_idx[src], currency_idx[dst], -math.log(rate)))

    # Bellman-Ford
    dist = [float('inf')] * n
    prev = [None] * n
    dist[0] = 0

    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u

    # Negative cycle detection
    for u, v, w in edges:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            # Found a negative cycle -> arbitrage is possible
            # Recover the cycle
            cycle = []
            visited = set()
            x = v
            for _ in range(n):
                x = prev[x]
            start = x
            cycle.append(currencies[start])
            x = prev[start]
            while x != start:
                cycle.append(currencies[x])
                x = prev[x]
            cycle.append(currencies[start])
            return cycle[::-1]

    return []  # No arbitrage

currencies = ['USD', 'EUR', 'GBP', 'JPY']
rates = {
    ('USD', 'EUR'): 0.85, ('EUR', 'USD'): 1.20,  # 1.20 > 1/0.85
    ('EUR', 'GBP'): 0.86, ('GBP', 'EUR'): 1.17,
    ('GBP', 'USD'): 1.30, ('USD', 'GBP'): 0.78,
    ('USD', 'JPY'): 110.0, ('JPY', 'USD'): 0.0091,
}
cycle = detect_arbitrage(currencies, rates)
if cycle:
    print(f"Arbitrage: {' -> '.join(cycle)}")
```

---

## 10. Algorithm Comparison Table

| Algorithm | Complexity | Negative Edges | Negative Cycle Detection | Type | Use Case |
|:---|:---|:---|:---|:---|:---|
| Dijkstra | O((V+E) log V) | No | No | Single-source | Non-negative edge graphs |
| Bellman-Ford | O(VE) | Yes | Yes | Single-source | Negative edges |
| SPFA | O(VE) worst | Yes | Yes | Single-source | Improved Bellman-Ford |
| Floyd-Warshall | O(V^3) | Yes | Yes (negative diagonal) | All-pairs | Small/dense graphs |
| Johnson | O(V^2 log V + VE) | Yes | Yes | All-pairs | Sparse graphs |
| DAG Shortest Path | O(V+E) | Yes | N/A (DAG) | Single-source | DAG only |
| BFS | O(V+E) | No (weight=1) | No | Single-source | Unweighted |
| 0-1 BFS | O(V+E) | No (weight 0/1) | No | Single-source | Weights 0 or 1 |
| A* | O(E) best | No | No | Point-to-point | When heuristic is available |

## Selection Guide

```
Algorithm selection flow for shortest paths:

  Are all edge weights equal (or absent)?
    |-- YES -> BFS  O(V+E)
    +-- NO  -> Are edge weights only 0 or 1?
              |-- YES -> 0-1 BFS  O(V+E)
              +-- NO  -> Are there negative edges?
                        |-- NO  -> Is it a DAG?
                        |         |-- YES -> DAG topological order  O(V+E)
                        |         +-- NO  -> Is the goal well-defined?
                        |                   |-- YES -> A*
                        |                   +-- NO  -> Dijkstra  O((V+E)logV)
                        +-- YES -> Do you need all-pairs?
                                  |-- YES -> Dense? -> Floyd  O(V^3)
                                  |         Sparse? -> Johnson  O(V^2logV+VE)
                                  +-- NO  -> Bellman-Ford  O(VE)
```

| Condition | Recommended | Reason |
|:---|:---|:---|
| Non-negative edges + single-source | Dijkstra | Fastest |
| Negative edges + single-source | Bellman-Ford | Handles negative edges |
| All-pairs shortest distances | Floyd-Warshall | Simple implementation |
| DAG | Topological relaxation | Fastest at O(V+E) |
| All edges weight 1 | BFS | O(V+E) |
| Sparse graph + all-pairs | Johnson | Dijkstra V times |
| Well-defined goal | A* | Speedup via heuristic |
| Edge weights 0/1 | 0-1 BFS | deque for O(V+E) |

---

## 11. Anti-Patterns

### Anti-Pattern 1: Using Dijkstra with Negative Edges

```python
# BAD: Dijkstra on a graph with negative edges
graph_neg = {
    'A': [('B', 1), ('C', 4)],
    'B': [('C', -3)],   # Negative edge!
    'C': [],
}
# Dijkstra finalizes A->C=4, but the actual shortest is
# A->B->C = 1+(-3) = -2

# GOOD: Use Bellman-Ford
vertices = ['A', 'B', 'C']
edges = [('A','B',1), ('A','C',4), ('B','C',-3)]
dist, _, _ = bellman_ford(vertices, edges, 'A')
# dist['C'] = -2 (correct)
```

### Anti-Pattern 2: Using Floyd-Warshall on Large Graphs

```python
# BAD: Floyd-Warshall on a graph with V=10000
# -> O(V^3) = 10^12 operations -> takes hours

# GOOD: Use Dijkstra for single-source queries
# Reconsider whether all-pairs is truly needed
# For sparse graphs, consider Johnson's algorithm
```

### Anti-Pattern 3: Not Skipping Stale Heap Entries on Distance Update

```python
# When distances are updated in Dijkstra, stale entries remain in the heap
# -> Must compare with dist[u] on heappop to skip them

# BAD: Forgetting to skip
def bad_dijkstra(graph, start):
    dist = defaultdict(lambda: float('inf'))
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        # No skip -> wasteful processing with stale data
        for v, w in graph[u]:
            ...

# GOOD: Skip stale entries
def good_dijkstra(graph, start):
    dist = defaultdict(lambda: float('inf'))
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:  # Stale entry -> skip
            continue
        for v, w in graph[u]:
            ...
```

### Anti-Pattern 4: Non-Admissible Heuristic in A*

```python
# BAD: Heuristic overestimates actual cost -> optimal solution not guaranteed
def bad_heuristic(node, goal):
    # Euclidean distance x 2 -> overestimates
    return 2 * math.sqrt((node[0]-goal[0])**2 + (node[1]-goal[1])**2)

# GOOD: Admissible heuristic
def good_heuristic(node, goal):
    # Manhattan distance (admissible on grids)
    return abs(node[0]-goal[0]) + abs(node[1]-goal[1])
```

### Anti-Pattern 5: Forgetting to Handle Unreachable Vertices

```python
# BAD: No check for unreachable case
def bad_path_reconstruction(prev, start, end):
    path = [end]
    current = end
    while current != start:
        current = prev[current]  # Possible KeyError!
        path.append(current)
    return path[::-1]

# GOOD: Check reachability
def good_path_reconstruction(prev, start, end):
    if end not in prev and end != start:
        return []  # Unreachable
    path = [end]
    current = end
    while current != start:
        if current not in prev:
            return []
        current = prev[current]
        path.append(current)
    return path[::-1]
```

---

## 12. Detailed Complexity Analysis

### Deriving Dijkstra's Complexity

```
Using a priority queue (binary heap):

  Operation                Count            Per-op cost   Total
  ---------------------------------------------------------------
  heappush (init)          1                O(1)          O(1)
  heappop                  at most V+E      O(log(V+E))   O((V+E) log V)
  heappush (relaxation)    at most E        O(log(V+E))   O(E log V)
  ---------------------------------------------------------------
  Total                                                   O((V+E) log V)

  Note: Heap size is at most V+E (due to stale entries)
  log(V+E) = O(log V) (since E <= V^2, log(V+E) <= log(V^2 + V) ~ 2 log V)

  With a Fibonacci heap:
  - decrease-key becomes O(1) (amortized)
  - Total: O(V log V + E)
  - Theoretically faster, but complex to implement with large constant factors
```

### Space Optimization for Floyd-Warshall

```python
# Standard version: O(V^2) space (the k dimension is unnecessary -- overwriting is correct)
# Reason: dist[i][k] and dist[k][j] do not change during the k-th iteration

# Note: If path reconstruction is not needed, only the dist matrix suffices
# If path reconstruction is needed, the nxt matrix is also required, totaling O(V^2) space
```


---

## Exercises

### Exercise 1: Basic Implementation

Implement code that satisfies the following requirements.

**Requirements:**
- Perform input data validation
- Implement proper error handling
- Write test code as well

```python
# Exercise 1: Basic implementation template
class Exercise1:
    """Exercise for basic implementation patterns"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """Validate input value"""
        if value is None:
            raise ValueError("Input value is None")
        return True

    def process(self, value):
        """Main processing logic"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """Get processing results"""
        return {
            'count': len(self.data),
            'data': self.data
        }

# Tests
def test_exercise1():
    ex = Exercise1()
    assert ex.process(1) == [1]
    assert ex.process(2) == [1, 2]
    assert ex.get_results()['count'] == 2

    try:
        ex.process(None)
        assert False, "Exception should have been raised"
    except ValueError:
        pass

    print("All tests passed!")

test_exercise1()
```

### Exercise 2: Applied Patterns

Extend the basic implementation by adding the following features.

```python
# Exercise 2: Applied patterns
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """Exercise for applied patterns"""

    def __init__(self, max_size: int = 100):
        self._items: List[Dict] = []
        self._max_size = max_size
        self._created_at = datetime.now()

    def add(self, key: str, value: any) -> bool:
        """Add an item (with size limit)"""
        if len(self._items) >= self._max_size:
            return False
        self._items.append({
            'key': key,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        return True

    def find(self, key: str) -> Optional[Dict]:
        """Search by key"""
        for item in reversed(self._items):
            if item['key'] == key:
                return item
        return None

    def remove(self, key: str) -> bool:
        """Remove by key"""
        for i, item in enumerate(self._items):
            if item['key'] == key:
                self._items.pop(i)
                return True
        return False

    def stats(self) -> Dict:
        """Statistics"""
        return {
            'total_items': len(self._items),
            'max_size': self._max_size,
            'usage_percent': len(self._items) / self._max_size * 100,
            'uptime': str(datetime.now() - self._created_at)
        }

# Tests
def test_advanced():
    ex = AdvancedExercise(max_size=3)
    assert ex.add("a", 1) == True
    assert ex.add("b", 2) == True
    assert ex.add("c", 3) == True
    assert ex.add("d", 4) == False  # Size limit
    assert ex.find("b")['value'] == 2
    assert ex.remove("b") == True
    assert ex.find("b") is None
    stats = ex.stats()
    assert stats['total_items'] == 2
    print("All applied tests passed!")

test_advanced()
```

### Exercise 3: Performance Optimization

Improve the performance of the following code.

```python
# Exercise 3: Performance optimization
import time
from functools import lru_cache

# Before optimization (O(n^2))
def slow_search(data: list, target: int) -> int:
    """Inefficient search"""
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] + data[j] == target:
                return (i, j)
    return (-1, -1)

# After optimization (O(n))
def fast_search(data: list, target: int) -> tuple:
    """Efficient search using a hash map"""
    seen = {}
    for i, num in enumerate(data):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return (-1, -1)

# Benchmark
def benchmark():
    import random
    data = list(range(5000))
    random.shuffle(data)
    target = data[100] + data[4000]

    start = time.time()
    result1 = slow_search(data, target)
    slow_time = time.time() - start

    start = time.time()
    result2 = fast_search(data, target)
    fast_time = time.time() - start

    print(f"Inefficient: {slow_time:.4f}s")
    print(f"Efficient:   {fast_time:.6f}s")
    print(f"Speedup:     {slow_time/fast_time:.0f}x")

benchmark()
```

**Key Points:**
- Be mindful of algorithmic complexity
- Choose appropriate data structures
- Measure the effect with benchmarks
---

## 13. FAQ

### Q1: Why can't Dijkstra's algorithm handle negative edges?

**A:** Dijkstra's algorithm relies on the greedy assumption that "once a shortest distance is finalized, it never changes." With negative edges, a path discovered later may be shorter than an already finalized distance, violating this assumption. Example: after finalizing A->C (weight 4), A->B->C (1+(-3)=-2) might be found but cannot be corrected.

### Q2: Where does the O((V+E) log V) complexity of Dijkstra come from?

**A:** Each vertex is removed from the heap at most once (V heappop operations = V log V), and each edge causes at most one heap insertion during relaxation (E heappush operations = E log V). Total: O((V+E) log V). Using a Fibonacci heap improves this to O(V log V + E), but the implementation complexity makes binary heaps the practical standard.

### Q3: How do you implement path reconstruction?

**A:** Record the "predecessor" for each vertex -- which vertex it was reached from. Then trace back from the goal to the start to reconstruct the path. See the `reconstruct_path` function above.

### Q4: What is 0-1 BFS?

**A:** A shortest path algorithm specialized for graphs where edge weights are only 0 or 1. Instead of a regular queue, it uses a deque (double-ended queue), pushing weight-0 edges to the front and weight-1 edges to the back. Runs in O(V+E). Effective for problems like moving through a grid where breaking walls costs 1.

### Q5: How do you choose the heuristic function for A* search?

**A:** For grid problems, Manhattan distance is typical for 4-directional movement, Chebyshev distance for 8-directional movement, and Euclidean distance for free movement. The key requirement is "admissibility" (never overestimating the true distance). A non-admissible heuristic is faster but may miss the optimal solution.

### Q6: Why can't shortest paths be defined when negative cycles exist?

**A:** Traversing a negative cycle multiple times can reduce the distance indefinitely, making the "shortest" undefined. Specifically, if the source can reach a negative cycle and the negative cycle can reach the destination, the shortest distance is negative infinity. Bellman-Ford detects this situation during the V-th relaxation pass.

---


## FAQ

### Q1: What is the most important point to keep in mind when learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory, but by actually writing code and verifying how it works.

### Q2: What are common mistakes that beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this used in practice?

Knowledge of this topic is frequently applied in day-to-day development work. It becomes especially important during code reviews and architecture design.

---

## 14. Summary

| Topic | Key Points |
|:---|:---|
| Dijkstra's Algorithm | Single-source shortest path for non-negative edges. O((V+E) log V) with priority queue |
| Bellman-Ford Algorithm | Allows negative edges. V-1 edge relaxations in O(VE). Can detect negative cycles |
| Floyd-Warshall Algorithm | All-pairs shortest paths. DP in O(V^3). For small graphs |
| Johnson's Algorithm | All-pairs shortest paths for sparse graphs. BF + Dijkstra V times |
| DAG Shortest Path | Topological-order relaxation in O(V+E). DAG only |
| A* Search | Dijkstra with heuristic. Optimal for point-to-point shortest paths |
| Relaxation | Fundamental operation common to all algorithms |
| Path reconstruction | Trace backward from goal using a predecessor array |

---

## Recommended Next Guides

- [Graph Traversal](./02-graph-traversal.md) -- BFS/DFS fundamentals (prerequisite for shortest paths)
- [Dynamic Programming](./04-dynamic-programming.md) -- The DP concepts behind Floyd-Warshall
- [Greedy Algorithms](./05-greedy.md) -- The greedy strategy behind Dijkstra's algorithm
- [Network Flow](../03-advanced/03-network-flow.md) -- Advanced graph optimization

---

## References

1. Cormen, T. H. et al. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. -- Chapters 22-25
2. Dijkstra, E. W. (1959). "A note on two problems in connexion with graphs." *Numerische Mathematik*.
3. Bellman, R. (1958). "On a routing problem." *Quarterly of Applied Mathematics*.
4. Floyd, R. W. (1962). "Algorithm 97: Shortest Path." *Communications of the ACM*.
5. Johnson, D. B. (1977). "Efficient algorithms for shortest paths in sparse networks." *Journal of the ACM*.
6. Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). "A Formal Basis for the Heuristic Determination of Minimum Cost Paths." *IEEE Transactions on Systems Science and Cybernetics*.
