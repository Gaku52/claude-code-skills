# Graphs — Representations, Adjacency Lists/Matrices, and Weighted Graphs

> Learn the fundamental concepts of graphs that represent diverse relationships such as networks, dependencies, and maps, along with the characteristics of each representation method.

---

## What You Will Learn in This Chapter

1. **Basic graph terminology** — vertices, edges, directed/undirected, weights
2. **Adjacency lists and adjacency matrices** — implementation and when to use each
3. **Weighted graphs** and representations of special graphs
4. **Union-Find** — disjoint set data structure
5. **Practical applications** — social graphs, dependency resolution, pathfinding


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Familiarity with the content of [Heaps -- Binary Heaps, Heapsort, and Priority Queue Implementation](./05-heaps.md)

---

## 1. Fundamental Graph Concepts

### 1.1 Types of Graphs

```
Undirected graph:               Directed graph:
  A --- B                    A -> B
  |   / |                    ^    |
  |  /  |                    D <- C
  | /   |
  C --- D

Weighted graph:                 DAG (Directed Acyclic Graph):
  A -5- B                    A -> B -> D
  |     |                    |    |
  3     2                    C -> E
  |     |
  C -1- D

Complete graph K4:              Bipartite graph:
  A --- B                    L1 --- R1
  |\ /|                     |  \ / |
  | X  |                     |   X  |
  |/ \ |                     |  / \ |
  C --- D                    L2 --- R2

Multigraph:                     Self-loop:
  A ==== B                    A <->
  (2 edges)                   (edge to itself)
```

### 1.2 Terminology

```
Vertex (Node): a point in the graph
Edge: a connection between vertices
Degree: the number of edges connected to a vertex
  - Directed graph: in-degree + out-degree
Path: a sequence of vertices where consecutive edges exist
  - Simple path: no vertex repetition
Cycle: a path where the start and end vertices are the same
  - DAG: a directed graph with no cycles
Connected: a path exists between all vertex pairs (undirected graph)
Strongly connected: a directed path exists between all vertex pairs (directed graph)
Connected component: a maximal connected subgraph
Tree: a connected graph with no cycles (E = V - 1)
Forest: a graph with no cycles (a collection of trees)
```

### 1.3 Fundamental Graph Theorems

```python
# Important graph theorems:
#
# 1. Handshaking Lemma:
#    Sum of all vertex degrees in an undirected graph = 2 * number of edges
#    sum(deg(v)) = 2|E|
#
# 2. Tree theorem:
#    A tree with n vertices always has n-1 edges
#    Adding one edge to a tree creates exactly one cycle
#
# 3. Euler's formula:
#    For planar graphs: V - E + F = 2
#    (V: vertices, E: edges, F: faces)
#
# 4. Upper bound on edges:
#    Undirected simple graph: E <= V(V-1)/2
#    Directed simple graph: E <= V(V-1)

def graph_info(vertices, edges, directed=False):
    """Display basic graph information"""
    v = len(vertices)
    e = len(edges)
    max_edges = v * (v - 1) if directed else v * (v - 1) // 2
    density = e / max_edges if max_edges > 0 else 0

    print(f"Number of vertices: {v}")
    print(f"Number of edges: {e}")
    print(f"Maximum edges: {max_edges}")
    print(f"Density: {density:.4f}")
    print(f"Sparse/Dense: {'Dense' if density > 0.5 else 'Sparse'}")
    print(f"Could be a tree: {'Yes' if e == v - 1 else 'No'}")
```

---

## 2. Adjacency List

### 2.1 Dictionary-Based Implementation (Most Common)

```python
class Graph:
    """Dictionary-based adjacency list

    Optimal for sparse graphs. Easy dynamic vertex/edge addition.
    Space complexity: O(V + E)
    """
    def __init__(self, directed=False):
        self.adj = {}
        self.directed = directed

    def add_vertex(self, v):
        """Add a vertex — O(1)"""
        if v not in self.adj:
            self.adj[v] = []

    def add_edge(self, u, v, weight=1):
        """Add an edge — O(1)"""
        self.add_vertex(u)
        self.add_vertex(v)
        self.adj[u].append((v, weight))
        if not self.directed:
            self.adj[v].append((u, weight))

    def remove_edge(self, u, v):
        """Remove an edge — O(degree)"""
        self.adj[u] = [(w, wt) for w, wt in self.adj.get(u, []) if w != v]
        if not self.directed:
            self.adj[v] = [(w, wt) for w, wt in self.adj.get(v, []) if w != u]

    def remove_vertex(self, v):
        """Remove a vertex and all its edges — O(V + E)"""
        if v not in self.adj:
            return
        # Remove edges referencing v
        for u in self.adj:
            self.adj[u] = [(w, wt) for w, wt in self.adj[u] if w != v]
        del self.adj[v]

    def neighbors(self, v):
        """Return adjacent vertices — O(1)"""
        return [(w, wt) for w, wt in self.adj.get(v, [])]

    def has_edge(self, u, v):
        """Check edge existence — O(degree(u))"""
        return any(w == v for w, _ in self.adj.get(u, []))

    def vertices(self):
        """All vertices — O(V)"""
        return list(self.adj.keys())

    def edges(self):
        """All edges — O(V + E)"""
        result = []
        visited = set()
        for u in self.adj:
            for v, w in self.adj[u]:
                edge = (min(u, v), max(u, v)) if not self.directed else (u, v)
                if edge not in visited:
                    result.append((u, v, w))
                    visited.add(edge)
        return result

    def degree(self, v):
        """Degree — O(1)"""
        return len(self.adj.get(v, []))

    def in_degree(self, v):
        """In-degree (directed graph) — O(V + E)"""
        count = 0
        for u in self.adj:
            count += sum(1 for w, _ in self.adj[u] if w == v)
        return count

    def out_degree(self, v):
        """Out-degree (directed graph) — O(1)"""
        return len(self.adj.get(v, []))

    def __repr__(self):
        lines = []
        for v in sorted(self.adj.keys(), key=str):
            neighbors = [(w, wt) for w, wt in self.adj[v]]
            lines.append(f"  {v}: {neighbors}")
        return "Graph(\n" + "\n".join(lines) + "\n)"

# Usage example
g = Graph()
g.add_edge('A', 'B', 5)
g.add_edge('A', 'C', 3)
g.add_edge('B', 'D', 2)
g.add_edge('C', 'D', 1)
print(g)
print(f"Neighbors of A: {g.neighbors('A')}")
print(f"Edge B-D exists: {g.has_edge('B', 'D')}")
print(f"All edges: {g.edges()}")
```

### 2.2 Concise Implementation Using defaultdict

```python
from collections import defaultdict

class SimpleGraph:
    """Concise implementation using defaultdict

    Suitable for competitive programming and quick prototyping
    """
    def __init__(self, directed=False):
        self.adj = defaultdict(list)
        self.directed = directed

    def add_edge(self, u, v, w=1):
        self.adj[u].append((v, w))
        if not self.directed:
            self.adj[v].append((u, w))

    def __getitem__(self, v):
        return self.adj[v]

# Even simpler for unweighted graphs
class UnweightedGraph:
    def __init__(self, directed=False):
        self.adj = defaultdict(set)
        self.directed = directed

    def add_edge(self, u, v):
        self.adj[u].add(v)
        if not self.directed:
            self.adj[v].add(u)

    def has_edge(self, u, v):
        return v in self.adj[u]  # O(1) average with set
```

### 2.3 Visual Illustration of Adjacency List Representation

```
Adjacency list representation:

  Undirected weighted graph:
  A: [(B,5), (C,3)]
  B: [(A,5), (D,2)]
  C: [(A,3), (D,1)]
  D: [(B,2), (C,1)]

  Directed graph:
  A: [B, C]
  B: [D]
  C: [D]
  D: []

Memory: O(V + E) (undirected stores each edge twice: O(V + 2E))
```

---

## 3. Adjacency Matrix

### 3.1 Basic Implementation

```python
class GraphMatrix:
    """Adjacency matrix: optimal for dense or small-scale graphs

    Edge existence check is O(1)
    Space complexity: O(V^2)
    """
    def __init__(self, n):
        self.n = n
        self.matrix = [[0] * n for _ in range(n)]

    def add_edge(self, u, v, weight=1):
        """Add an edge — O(1)"""
        self.matrix[u][v] = weight
        self.matrix[v][u] = weight  # For undirected graph

    def remove_edge(self, u, v):
        """Remove an edge — O(1)"""
        self.matrix[u][v] = 0
        self.matrix[v][u] = 0

    def has_edge(self, u, v):
        """Check edge existence — O(1)"""
        return self.matrix[u][v] != 0

    def neighbors(self, v):
        """Return adjacent vertices — O(V)"""
        return [u for u in range(self.n) if self.matrix[v][u] != 0]

    def degree(self, v):
        """Degree — O(V)"""
        return sum(1 for u in range(self.n) if self.matrix[v][u] != 0)

    def edge_weight(self, u, v):
        """Return edge weight — O(1)"""
        return self.matrix[u][v]

    def __repr__(self):
        header = "    " + " ".join(f"{i:3d}" for i in range(self.n))
        rows = []
        for i in range(self.n):
            row = f"{i:3d} " + " ".join(f"{self.matrix[i][j]:3d}" for j in range(self.n))
            rows.append(row)
        return header + "\n" + "\n".join(rows)

# Usage example
gm = GraphMatrix(4)  # A=0, B=1, C=2, D=3
gm.add_edge(0, 1, 5)  # A-B: 5
gm.add_edge(0, 2, 3)  # A-C: 3
gm.add_edge(1, 3, 2)  # B-D: 2
gm.add_edge(2, 3, 1)  # C-D: 1
print(gm)
print(f"Neighbors of A: {gm.neighbors(0)}")  # [1, 2]
print(f"Weight of B-D: {gm.edge_weight(1, 3)}")  # 2
```

### 3.2 Fast Adjacency Matrix Using NumPy

```python
import numpy as np

class NumpyGraphMatrix:
    """NumPy-based adjacency matrix

    Optimal for matrix operations on large graphs (Floyd-Warshall, etc.)
    """
    def __init__(self, n):
        self.n = n
        self.matrix = np.zeros((n, n), dtype=np.float64)

    def add_edge(self, u, v, weight=1.0):
        self.matrix[u][v] = weight
        self.matrix[v][u] = weight

    def shortest_paths_floyd(self):
        """Floyd-Warshall algorithm — O(V^3)

        Computes all-pairs shortest distances
        """
        dist = self.matrix.copy()
        dist[dist == 0] = np.inf
        np.fill_diagonal(dist, 0)

        for k in range(self.n):
            for i in range(self.n):
                for j in range(self.n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        return dist

    def transitive_closure(self):
        """Transitive closure: reachability matrix"""
        reach = (self.matrix > 0).astype(int)
        np.fill_diagonal(reach, 1)
        for k in range(self.n):
            for i in range(self.n):
                for j in range(self.n):
                    reach[i][j] = reach[i][j] or (reach[i][k] and reach[k][j])
        return reach

    def degree_matrix(self):
        """Degree matrix"""
        degrees = np.sum(self.matrix > 0, axis=1)
        return np.diag(degrees)

    def laplacian_matrix(self):
        """Laplacian matrix = degree matrix - adjacency matrix

        Number of connected components = multiplicity of eigenvalue 0 in the Laplacian
        """
        adj = (self.matrix > 0).astype(float)
        return self.degree_matrix() - adj
```

### 3.3 Visual Illustration of Adjacency Matrix Representation

```
Adjacency matrix representation (A=0, B=1, C=2, D=3):

      A  B  C  D
  A [ 0  5  3  0 ]
  B [ 5  0  0  2 ]
  C [ 3  0  0  1 ]
  D [ 0  2  1  0 ]

Memory: O(V^2)

Matrix operations are possible:
- A^k[i][j] = number of paths of length k from i to j
- Eigenvalue analysis for spectral analysis of the graph
```

---

## 4. Edge List Representation

```python
class EdgeListGraph:
    """Edge list: represents a graph as a collection of edges

    Optimal for Kruskal's algorithm (edges sorted by weight)
    Space complexity: O(E)
    """
    def __init__(self):
        self.edges = []
        self.vertices = set()

    def add_edge(self, u, v, weight=1):
        """O(1)"""
        self.edges.append((u, v, weight))
        self.vertices.add(u)
        self.vertices.add(v)

    def sorted_edges(self):
        """Sort by weight — O(E log E)"""
        return sorted(self.edges, key=lambda e: e[2])

    def to_adjacency_list(self, directed=False):
        """Convert to adjacency list"""
        adj = {v: [] for v in self.vertices}
        for u, v, w in self.edges:
            adj[u].append((v, w))
            if not directed:
                adj[v].append((u, w))
        return adj

# Usage example
g = EdgeListGraph()
g.add_edge('A', 'B', 5)
g.add_edge('A', 'C', 3)
g.add_edge('B', 'D', 2)
g.add_edge('C', 'D', 1)
print(f"Edges (sorted by weight): {g.sorted_edges()}")
# [('C', 'D', 1), ('B', 'D', 2), ('A', 'C', 3), ('A', 'B', 5)]
```

---

## 5. Union-Find (Disjoint Set Data Structure)

```python
class UnionFind:
    """Union-Find: management of disjoint sets

    Two optimizations:
    1. Path Compression: connect nodes directly to root during find
    2. Union by Rank: attach smaller tree to larger tree

    Nearly O(1) operations (inverse Ackermann function alpha(n) ~ 5)
    """
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n
        self.components = n

    def find(self, x):
        """Return root — O(alpha(n)) ~ O(1)"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path Compression
        return self.parent[x]

    def union(self, x, y):
        """Merge two sets — O(alpha(n)) ~ O(1)

        Returns: True if merged, False if already in same set
        """
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False

        # Union by Rank
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        self.size[rx] += self.size[ry]
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        self.components -= 1
        return True

    def connected(self, x, y):
        """Check if in the same set — O(alpha(n))"""
        return self.find(x) == self.find(y)

    def component_size(self, x):
        """Size of the set containing x"""
        return self.size[self.find(x)]

    def num_components(self):
        """Number of connected components"""
        return self.components

# Usage example
uf = UnionFind(7)
uf.union(0, 1)
uf.union(1, 2)
uf.union(3, 4)
print(uf.connected(0, 2))  # True
print(uf.connected(0, 3))  # False
print(uf.num_components())  # 4 (groups: {0,1,2}, {3,4}, {5}, {6})
print(uf.component_size(0)) # 3
```

### 5.1 Applications of Union-Find

```python
# === Kruskal's Minimum Spanning Tree ===
def kruskal(vertices, edges):
    """Kruskal's algorithm — O(E log E)

    Sort edges by weight and add edges that don't create cycles
    """
    n = len(vertices)
    uf = UnionFind(n)
    vertex_idx = {v: i for i, v in enumerate(vertices)}

    # Sort edges by weight
    sorted_edges = sorted(edges, key=lambda e: e[2])

    mst = []
    total_weight = 0
    for u, v, w in sorted_edges:
        ui, vi = vertex_idx[u], vertex_idx[v]
        if uf.union(ui, vi):
            mst.append((u, v, w))
            total_weight += w
            if len(mst) == n - 1:
                break

    return mst, total_weight

# Usage example
vertices = ['A', 'B', 'C', 'D', 'E']
edges = [
    ('A', 'B', 4), ('A', 'C', 2), ('B', 'C', 1),
    ('B', 'D', 5), ('C', 'D', 8), ('C', 'E', 10),
    ('D', 'E', 2),
]
mst, weight = kruskal(vertices, edges)
print(f"Minimum spanning tree: {mst}")
print(f"Total weight: {weight}")
# Minimum spanning tree: [('B', 'C', 1), ('A', 'C', 2), ('D', 'E', 2), ('A', 'B', 4)]
# Total weight: 9

# === Connected Component Detection ===
def count_connected_components(n, edges):
    """Count connected components in an undirected graph"""
    uf = UnionFind(n)
    for u, v in edges:
        uf.union(u, v)
    return uf.num_components()

print(count_connected_components(5, [(0,1), (1,2), (3,4)]))  # 2

# === Cycle Detection ===
def has_cycle_undirected(n, edges):
    """Cycle detection in an undirected graph"""
    uf = UnionFind(n)
    for u, v in edges:
        if uf.connected(u, v):
            return True  # Already connected -> cycle
        uf.union(u, v)
    return False

print(has_cycle_undirected(4, [(0,1), (1,2), (2,3)]))        # False
print(has_cycle_undirected(4, [(0,1), (1,2), (2,3), (3,0)])) # True
```

---

## 6. Special Graphs

### 6.1 Bipartite Graphs

```python
def is_bipartite(graph, n):
    """Bipartite graph check (BFS coloring) — O(V+E)

    Determines whether the graph can be 2-colored.
    A prerequisite for matching problems.
    """
    from collections import deque
    color = [-1] * n
    for start in range(n):
        if color[start] != -1:
            continue
        queue = deque([start])
        color[start] = 0
        while queue:
            u = queue.popleft()
            for v in graph[u]:
                if color[v] == -1:
                    color[v] = 1 - color[u]
                    queue.append(v)
                elif color[v] == color[u]:
                    return False, []
    return True, color

# Usage example
graph = {0: [1, 3], 1: [0, 2], 2: [1, 3], 3: [0, 2]}
is_bip, coloring = is_bipartite(graph, 4)
print(f"Bipartite: {is_bip}, Coloring: {coloring}")
# Bipartite: True, Coloring: [0, 1, 0, 1]
```

### 6.2 Treating Grids as Graphs

```python
def grid_to_graph(grid):
    """Adjacency relations on a 2D grid

    Many graph problems are posed on grids.
    Maze search, number of islands, shortest path, etc.
    """
    rows, cols = len(grid), len(grid[0])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def neighbors(r, c):
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                yield nr, nc

    return neighbors

# Number of Islands
def num_islands(grid):
    """Count the number of islands using DFS — O(R x C)"""
    if not grid:
        return 0

    rows, cols = len(grid), len(grid[0])
    get_neighbors = grid_to_graph(grid)
    visited = set()
    count = 0

    def dfs(r, c):
        visited.add((r, c))
        for nr, nc in get_neighbors(r, c):
            if (nr, nc) not in visited and grid[nr][nc] == '1':
                dfs(nr, nc)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1' and (r, c) not in visited:
                dfs(r, c)
                count += 1

    return count

grid = [
    ['1', '1', '0', '0', '0'],
    ['1', '1', '0', '0', '0'],
    ['0', '0', '1', '0', '0'],
    ['0', '0', '0', '1', '1'],
]
print(f"Number of islands: {num_islands(grid)}")  # 3

# For 8-directional movement
def grid_8dir_neighbors(grid):
    rows, cols = len(grid), len(grid[0])
    directions = [
        (0, 1), (0, -1), (1, 0), (-1, 0),
        (1, 1), (1, -1), (-1, 1), (-1, -1)
    ]
    def neighbors(r, c):
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                yield nr, nc
    return neighbors
```

### 6.3 Implicit Graphs

```python
# Implicit graphs: graphs that don't explicitly store edges,
# but dynamically generate adjacent vertices via functions

# Example 1: Number puzzle (state-space graph)
def word_ladder(begin_word, end_word, word_list):
    """Word Ladder: reach the target word by changing one letter at a time

    Vertices: each word
    Edges: word pairs differing by exactly one letter
    """
    from collections import deque
    word_set = set(word_list)
    if end_word not in word_set:
        return 0

    queue = deque([(begin_word, 1)])
    visited = {begin_word}

    while queue:
        word, steps = queue.popleft()
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                next_word = word[:i] + c + word[i+1:]
                if next_word == end_word:
                    return steps + 1
                if next_word in word_set and next_word not in visited:
                    visited.add(next_word)
                    queue.append((next_word, steps + 1))
    return 0

print(word_ladder("hit", "cog", ["hot","dot","dog","lot","log","cog"]))  # 5

# Example 2: Minimum knight moves
def min_knight_moves(x, y):
    """Minimum moves for a chess knight from (0,0) to (x,y)"""
    from collections import deque
    moves = [
        (2, 1), (2, -1), (-2, 1), (-2, -1),
        (1, 2), (1, -2), (-1, 2), (-1, -2)
    ]
    x, y = abs(x), abs(y)
    visited = {(0, 0)}
    queue = deque([(0, 0, 0)])

    while queue:
        cx, cy, steps = queue.popleft()
        if cx == x and cy == y:
            return steps
        for dx, dy in moves:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) not in visited and -2 <= nx <= x + 2 and -2 <= ny <= y + 2:
                visited.add((nx, ny))
                queue.append((nx, ny, steps + 1))
    return -1
```

### 6.4 Topological Sort

```python
from collections import deque

def topological_sort_kahn(graph, n):
    """Kahn's algorithm (BFS-based) — O(V+E)

    Process nodes with in-degree 0 first.
    If the graph is not a DAG (has cycles), not all nodes can be processed.
    """
    in_degree = [0] * n
    for u in range(n):
        for v in graph.get(u, []):
            in_degree[v] += 1

    queue = deque([v for v in range(n) if in_degree[v] == 0])
    result = []

    while queue:
        u = queue.popleft()
        result.append(u)
        for v in graph.get(u, []):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    if len(result) != n:
        return None  # Cycle exists
    return result

def topological_sort_dfs(graph, n):
    """DFS-based topological sort — O(V+E)"""
    visited = [False] * n
    stack = []
    has_cycle = [False]

    def dfs(u, in_stack):
        if has_cycle[0]:
            return
        visited[u] = True
        in_stack.add(u)
        for v in graph.get(u, []):
            if v in in_stack:
                has_cycle[0] = True
                return
            if not visited[v]:
                dfs(v, in_stack)
        in_stack.discard(u)
        stack.append(u)

    for v in range(n):
        if not visited[v]:
            dfs(v, set())

    if has_cycle[0]:
        return None
    return stack[::-1]

# Usage example: task dependencies
# 0->1->3
# 0->2->3
graph = {0: [1, 2], 1: [3], 2: [3], 3: []}
print(topological_sort_kahn(graph, 4))  # [0, 1, 2, 3] or [0, 2, 1, 3]
```

### 6.5 Strongly Connected Components (Kosaraju's Algorithm)

```python
def kosaraju_scc(graph, n):
    """Kosaraju's algorithm: SCC decomposition — O(V+E)

    Step 1: Compute post-order via DFS
    Step 2: Transpose the graph
    Step 3: DFS on transposed graph in reverse post-order -> each SCC
    """
    # Step 1: Post-order
    visited = [False] * n
    order = []

    def dfs1(u):
        visited[u] = True
        for v in graph.get(u, []):
            if not visited[v]:
                dfs1(v)
        order.append(u)

    for v in range(n):
        if not visited[v]:
            dfs1(v)

    # Step 2: Transposed graph
    reversed_graph = {i: [] for i in range(n)}
    for u in graph:
        for v in graph[u]:
            reversed_graph[v].append(u)

    # Step 3: DFS in reverse order
    visited = [False] * n
    sccs = []

    def dfs2(u, component):
        visited[u] = True
        component.append(u)
        for v in reversed_graph.get(u, []):
            if not visited[v]:
                dfs2(v, component)

    for u in reversed(order):
        if not visited[u]:
            component = []
            dfs2(u, component)
            sccs.append(component)

    return sccs

# Usage example
graph = {0: [1], 1: [2], 2: [0, 3], 3: [4], 4: [5], 5: [3]}
sccs = kosaraju_scc(graph, 6)
print(f"Strongly connected components: {sccs}")  # [[0, 2, 1], [3, 5, 4]]
```

---

## 7. Graph Representation Conversion

```python
def adj_list_to_matrix(adj, vertices):
    """Convert adjacency list to adjacency matrix"""
    n = len(vertices)
    v_idx = {v: i for i, v in enumerate(vertices)}
    matrix = [[0] * n for _ in range(n)]
    for u in adj:
        for v, w in adj[u]:
            matrix[v_idx[u]][v_idx[v]] = w
    return matrix

def adj_matrix_to_list(matrix, vertices=None):
    """Convert adjacency matrix to adjacency list"""
    n = len(matrix)
    if vertices is None:
        vertices = list(range(n))
    adj = {v: [] for v in vertices}
    for i in range(n):
        for j in range(n):
            if matrix[i][j] != 0:
                adj[vertices[i]].append((vertices[j], matrix[i][j]))
    return adj

def edge_list_to_adj_list(edges, directed=False):
    """Convert edge list to adjacency list"""
    from collections import defaultdict
    adj = defaultdict(list)
    for u, v, w in edges:
        adj[u].append((v, w))
        if not directed:
            adj[v].append((u, w))
    return dict(adj)
```

---

## 8. Practical Application Patterns

### 8.1 Social Graph Analysis

```python
from collections import deque

def bfs_shortest_path(graph, start, end):
    """Shortest distance between two users (six degrees of separation)"""
    if start == end:
        return 0, [start]

    visited = {start}
    queue = deque([(start, [start])])

    while queue:
        node, path = queue.popleft()
        for neighbor in graph.get(node, []):
            if neighbor == end:
                return len(path), path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return -1, []  # Unreachable

def mutual_friends(graph, user_a, user_b):
    """Return mutual friends — O(min(deg(A), deg(B)))"""
    friends_a = set(graph.get(user_a, []))
    friends_b = set(graph.get(user_b, []))
    return friends_a & friends_b

def friend_recommendations(graph, user, max_recs=10):
    """Recommend friends-of-friends (2 hops away)

    Rank people who are not already friends by number of mutual friends
    """
    from collections import Counter
    friends = set(graph.get(user, []))
    friends.add(user)

    candidates = Counter()
    for friend in graph.get(user, []):
        for fof in graph.get(friend, []):
            if fof not in friends:
                candidates[fof] += 1  # Count mutual friends

    return candidates.most_common(max_recs)

# Usage example
social = {
    "Alice": ["Bob", "Charlie", "David"],
    "Bob": ["Alice", "Charlie", "Eve"],
    "Charlie": ["Alice", "Bob", "Frank"],
    "David": ["Alice", "Frank"],
    "Eve": ["Bob"],
    "Frank": ["Charlie", "David"],
}
print(mutual_friends(social, "Alice", "Bob"))  # {'Charlie'}
print(friend_recommendations(social, "Alice"))
# [('Frank', 2), ('Eve', 1)]  Frank via Charlie and David
```

### 8.2 Dependency Resolution

```python
def resolve_dependencies(packages):
    """Resolve package dependencies using topological sort

    packages: {package: [dependencies]}
    """
    from collections import deque

    # Compute in-degrees
    in_degree = {pkg: 0 for pkg in packages}
    for pkg, deps in packages.items():
        for dep in deps:
            if dep in in_degree:
                in_degree[dep] = in_degree.get(dep, 0)

    # Build reverse graph (dependency -> dependent)
    reverse = {pkg: [] for pkg in packages}
    for pkg, deps in packages.items():
        for dep in deps:
            if dep in reverse:
                reverse[dep].append(pkg)
                in_degree[pkg] += 1

    # Topological sort
    queue = deque([pkg for pkg, deg in in_degree.items() if deg == 0])
    install_order = []

    while queue:
        pkg = queue.popleft()
        install_order.append(pkg)
        for dependent in reverse.get(pkg, []):
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(install_order) != len(packages):
        return None  # Circular dependency

    return install_order

# Usage example
packages = {
    "express": ["body-parser", "cookie-parser"],
    "body-parser": ["bytes", "content-type"],
    "cookie-parser": ["cookie"],
    "bytes": [],
    "content-type": [],
    "cookie": [],
}
order = resolve_dependencies(packages)
print(f"Install order: {order}")
# ['bytes', 'content-type', 'cookie', 'body-parser', 'cookie-parser', 'express']
```

### 8.3 Course Schedule

```python
def can_finish_courses(num_courses, prerequisites):
    """Determine if all courses can be completed (cycle detection)

    prerequisites: [[course, prereq], ...]
    """
    from collections import defaultdict, deque

    graph = defaultdict(list)
    in_degree = [0] * num_courses

    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1

    # Start with courses having in-degree 0
    queue = deque([i for i in range(num_courses) if in_degree[i] == 0])
    count = 0

    while queue:
        course = queue.popleft()
        count += 1
        for next_course in graph[course]:
            in_degree[next_course] -= 1
            if in_degree[next_course] == 0:
                queue.append(next_course)

    return count == num_courses

print(can_finish_courses(4, [[1,0], [2,0], [3,1], [3,2]]))  # True
print(can_finish_courses(2, [[1,0], [0,1]]))  # False (circular dependency)
```

### 8.4 Graph Coloring

```python
def graph_coloring(graph, n, max_colors=None):
    """Greedy graph coloring — O(V + E)

    Assign each vertex a color different from its neighbors.
    No optimality guarantee, but can color with at most max_degree + 1 colors.
    """
    if max_colors is None:
        max_colors = n

    colors = [-1] * n

    for v in range(n):
        # Collect colors used by neighbors
        used_colors = set()
        for u in graph.get(v, []):
            if colors[u] != -1:
                used_colors.add(colors[u])

        # Assign the smallest unused color
        for c in range(max_colors):
            if c not in used_colors:
                colors[v] = c
                break

    return colors

# Usage example: scheduling (time slot assignment)
# Meetings that cannot occur simultaneously are represented as adjacent edges
meetings = {0: [1, 2], 1: [0, 3], 2: [0, 3], 3: [1, 2]}
colors = graph_coloring(meetings, 4)
print(f"Coloring result: {colors}")  # [0, 1, 1, 0] — 2 colors suffice
```

### 8.5 Shortest Path (BFS: Unweighted)

```python
from collections import deque

def shortest_path_bfs(graph, start, end):
    """Shortest path in an unweighted graph — O(V + E)"""
    visited = {start}
    queue = deque([(start, [start])])

    while queue:
        node, path = queue.popleft()
        if node == end:
            return path

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None  # Unreachable

def all_shortest_paths_bfs(graph, start):
    """Shortest distances from source to all vertices — O(V + E)"""
    dist = {start: 0}
    queue = deque([start])

    while queue:
        u = queue.popleft()
        for v in graph.get(u, []):
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)

    return dist
```

---

## 9. Comparison Tables

### Table 1: Comparison of Representation Methods

| Operation | Adjacency List | Adjacency Matrix | Edge List |
|-----------|---------------|-----------------|-----------|
| Space | O(V+E) | O(V^2) | O(E) |
| Add edge | O(1) | O(1) | O(1) |
| Check edge existence | O(degree) | O(1) | O(E) |
| List neighbors | O(degree) | O(V) | O(E) |
| List all edges | O(V+E) | O(V^2) | O(E) |
| Add vertex | O(1) | O(V^2) | O(1) |
| Remove edge | O(degree) | O(1) | O(E) |
| Best for | Sparse | Dense | Sort-based |
| Memory efficiency | Good | Poor (sparse) | Best |

### Table 2: Graph Types and Characteristics

| Type | Characteristics | Example | Edge Count |
|------|----------------|---------|------------|
| Undirected graph | Edges have no direction | SNS friendships | - |
| Directed graph | Edges have direction | Web link structure | - |
| Weighted graph | Edges have costs | Road networks | - |
| DAG | Directed + no cycles | Task dependencies | - |
| Bipartite graph | 2-colorable | Matching problems | - |
| Complete graph | Edges between all vertex pairs | - | V(V-1)/2 |
| Tree | Connected + no cycles | Hierarchical structures | V-1 |
| Planar graph | No edge crossings | V-E+F=2 | E <= 3V-6 |

### Table 3: Graph Algorithm Complexities

| Algorithm | Complexity | Use Case |
|-----------|-----------|----------|
| BFS | O(V+E) | Shortest path (unweighted), connectivity |
| DFS | O(V+E) | Cycle detection, topological sort |
| Dijkstra | O((V+E) log V) | Shortest path (non-negative weights) |
| Bellman-Ford | O(VE) | Shortest path (negative weights allowed) |
| Floyd-Warshall | O(V^3) | All-pairs shortest path |
| Kruskal | O(E log E) | Minimum spanning tree |
| Prim | O((V+E) log V) | Minimum spanning tree |
| Tarjan SCC | O(V+E) | Strongly connected components |
| Kosaraju SCC | O(V+E) | Strongly connected components |

---

## 10. Anti-Patterns

### Anti-Pattern 1: Using adjacency matrix for sparse graphs

```python
# BAD: 10,000 vertices with 20,000 edges (sparse graph)
# Adjacency matrix: 10,000 x 10,000 = 100,000,000 elements (~800MB)
matrix = [[0] * 10000 for _ in range(10000)]

# GOOD: Adjacency list — O(V+E) = ~O(30,000)
from collections import defaultdict
adj = defaultdict(list)
```

### Anti-Pattern 2: Not considering dynamic vertex addition/removal

```python
# BAD: Fixed-size adjacency matrix with dynamic vertex addition
class FixedGraph:
    def __init__(self):
        self.matrix = [[0] * 100 for _ in range(100)]
    # Breaks when exceeding 100 vertices

# GOOD: Dictionary-based adjacency list — dynamically resizable
class DynamicGraph:
    def __init__(self):
        self.adj = {}
    def add_vertex(self, v):
        if v not in self.adj:
            self.adj[v] = []
```

### Anti-Pattern 3: Forgetting visited set in BFS/DFS

```python
# BAD: No visited check -> infinite loop
def bad_bfs(graph, start):
    queue = [start]
    while queue:
        node = queue.pop(0)
        for neighbor in graph[node]:
            queue.append(neighbor)  # Visits same node repeatedly!

# GOOD: Use a visited set
def good_bfs(graph, start):
    visited = {start}
    queue = [start]
    while queue:
        node = queue.pop(0)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

### Anti-Pattern 4: Using Dijkstra with negative weights

```python
# BAD: Using Dijkstra with negative weights -> incorrect results
# Dijkstra is greedy, so negative weights can cause it to miss shortest paths

# GOOD: Use Bellman-Ford when negative weights exist
def bellman_ford(vertices, edges, start):
    dist = {v: float('inf') for v in vertices}
    dist[start] = 0

    for _ in range(len(vertices) - 1):
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w

    # Negative cycle detection
    for u, v, w in edges:
        if dist[u] + w < dist[v]:
            return None  # Negative cycle exists
    return dist
```


---

## Troubleshooting

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| Initialization error | Configuration file issues | Verify config file path and format |
| Timeout | Network latency / resource shortage | Adjust timeout values, add retry logic |
| Out of memory | Data volume increase | Introduce batch processing, implement pagination |
| Permission error | Insufficient access rights | Check user permissions, review configuration |
| Data inconsistency | Concurrency conflicts | Introduce locking mechanisms, transaction management |

### Debugging Steps

1. **Check error messages**: read the stack trace to identify the location
2. **Establish reproduction steps**: reproduce the error with minimal code
3. **Form hypotheses**: list possible causes
4. **Verify systematically**: use log output or debuggers to test hypotheses
5. **Fix and regression test**: after fixing, also run tests on related areas

```python
# Debugging utility
import logging
import traceback
from functools import wraps

# Logger configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def debug_decorator(func):
    """Decorator that logs function input/output"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Call: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Return: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"Exception in {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

@debug_decorator
def process_data(items):
    """Data processing (debug target)"""
    if not items:
        raise ValueError("Empty data")
    return [item * 2 for item in items]
```

### Performance Issue Diagnosis

Steps for diagnosing performance issues:

1. **Identify bottlenecks**: measure with profiling tools
2. **Check memory usage**: check for memory leaks
3. **Check I/O waits**: verify disk and network I/O status
4. **Check connection counts**: verify connection pool status

| Issue Type | Diagnostic Tool | Countermeasure |
|-----------|----------------|----------------|
| CPU load | cProfile, py-spy | Algorithm improvement, parallelization |
| Memory leak | tracemalloc, objgraph | Proper reference release |
| I/O bottleneck | strace, iostat | Async I/O, caching |
| DB delay | EXPLAIN, slow query log | Indexing, query optimization |

---

## Team Development Practices

### Code Review Checklist

Points to check during code reviews related to this topic:

- [ ] Are naming conventions consistent?
- [ ] Is error handling appropriate?
- [ ] Is test coverage sufficient?
- [ ] Is there any performance impact?
- [ ] Are there any security issues?
- [ ] Is documentation updated?

### Knowledge Sharing Best Practices

| Method | Frequency | Audience | Benefit |
|--------|-----------|----------|---------|
| Pair programming | As needed | Complex tasks | Immediate feedback |
| Tech talks | Weekly | Entire team | Horizontal knowledge transfer |
| ADR (Architecture Decision Records) | As needed | Future members | Decision transparency |
| Retrospectives | Biweekly | Entire team | Continuous improvement |
| Mob programming | Monthly | Important designs | Consensus building |

### Technical Debt Management

```
Priority matrix:

        Impact High
          |
    +-----+-----+
    | Plan |  Act |
    | and  | Imme-|
    |sched-|diate-|
    | ule  |  ly  |
    +------+------+
    |Record| Next |
    | only | Sprint|
    |      |      |
    +------+------+
          |
        Impact Low
    Frequency Low  Frequency High
```

---

## Security Considerations

### Common Vulnerabilities and Countermeasures

| Vulnerability | Risk Level | Countermeasure | Detection Method |
|--------------|------------|----------------|------------------|
| Injection attacks | High | Input validation, parameterized queries | SAST/DAST |
| Authentication flaws | High | Multi-factor auth, session management hardening | Penetration testing |
| Sensitive data exposure | High | Encryption, access control | Security audit |
| Misconfiguration | Medium | Security headers, principle of least privilege | Configuration scan |
| Insufficient logging | Medium | Structured logging, audit trails | Log analysis |

### Secure Coding Best Practices

```python
# Secure coding example
import hashlib
import secrets
import hmac
from typing import Optional

class SecurityUtils:
    """Security utilities"""

    @staticmethod
    def generate_token(length: int = 32) -> str:
        """Generate cryptographically secure token"""
        return secrets.token_urlsafe(length)

    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> tuple:
        """Hash a password"""
        if salt is None:
            salt = secrets.token_hex(16)
        hashed = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            iterations=100000
        )
        return hashed.hex(), salt

    @staticmethod
    def verify_password(password: str, hashed: str, salt: str) -> bool:
        """Verify a password"""
        new_hash, _ = SecurityUtils.hash_password(password, salt)
        return hmac.compare_digest(new_hash, hashed)

    @staticmethod
    def sanitize_input(value: str) -> str:
        """Sanitize input value"""
        dangerous_chars = ['<', '>', '"', "'", '&', '\\']
        result = value
        for char in dangerous_chars:
            result = result.replace(char, '')
        return result.strip()

# Usage example
token = SecurityUtils.generate_token()
hashed, salt = SecurityUtils.hash_password("my_password")
is_valid = SecurityUtils.verify_password("my_password", hashed, salt)
```

### Security Checklist

- [ ] All input values are validated
- [ ] Sensitive information is not output to logs
- [ ] HTTPS is enforced
- [ ] CORS policy is properly configured
- [ ] Dependency vulnerability scans have been performed
- [ ] Error messages do not contain internal information
---

## 11. FAQ

### Q1: How do you convert between directed and undirected graphs?

**A:** An undirected graph can be converted to a directed graph by replacing each edge with two directed edges in both directions. Conversely, ignoring directions in a directed graph yields an undirected graph, but information is lost.

### Q2: Where is the boundary between dense and sparse graphs?

**A:** If E is approximately V^2, the graph is dense; if E is approximately V, it is sparse. In practice, if E < V^2 / 10, adjacency lists are advantageous. Social graphs are typically sparse (many users but limited friends), while small complete graphs are dense.

### Q3: When should you use NetworkX vs. a custom implementation?

**A:** NetworkX is convenient for prototyping and analysis (rich algorithms and visualization). For competitive programming or systems with strict performance requirements, use custom implementations. NetworkX is pure Python and can be slow for large-scale graphs (millions of nodes). For large-scale graphs, consider igraph or graph-tool (C++ backends).

### Q4: How should you choose a graph representation?

**A:**
- **Adjacency list**: the default for most cases. Optimal for sparse graphs and dynamic graphs
- **Adjacency matrix**: dense graphs, frequent edge existence checks, matrix operations (Floyd-Warshall, etc.)
- **Edge list**: Kruskal's algorithm, problems that assume sorted edges
- **Implicit graph**: state-space search (puzzles, games)

### Q5: When should you use Union-Find?

**A:** Optimal for problems centered on "Do they belong to the same group?" and "Merge groups":
- Connected component management
- Kruskal's algorithm
- Cycle detection (undirected graphs)
- Equivalence class management
- Dynamic connectivity problems

### Q6: How do you determine if a graph is a DAG?

**A:** If topological sort completes successfully, it is a DAG. If it doesn't complete, a cycle exists. In Kahn's algorithm (in-degree based), if the number of nodes in the result equals the total number of nodes, it is a DAG. In DFS-based approaches, a cycle exists if a back edge (an edge to a node currently being traversed) is detected.

---

## 12. Design Decision Flowchart

```
Graph representation selection flow:

  [START]
    |
    v
  Is E close to V^2?
    |
  Yes --> Use adjacency matrix
    |       (Floyd-Warshall, O(1) edge existence check)
  No
    |
    v
  Is edge sorting assumed? (Kruskal, etc.)
    |
  Yes --> Use edge list
    |
  No
    |
    v
  Is there dynamic vertex/edge addition or removal?
    |
  Yes --> Use dictionary-based adjacency list
    |       (defaultdict or dict)
  No
    |
    v
  Fixed vertex count with numeric indices?
    |
  Yes --> Use array-based adjacency list
    |       (list[list[int]])
  No
    |
    v
  Dictionary-based adjacency list (general-purpose default)
```

---

## 13. Exercises

### Beginner: Basic Graph Operations

**Problem:** Represent the following undirected weighted graph using both an adjacency list and an adjacency matrix, and enumerate all paths from vertex A to D.

```
        2
    A ----- B
    |       |
  4 |       | 3
    |       |
    C ----- D
        1
```

```python
# Solution: enumerate all paths (DFS + backtracking)
def find_all_paths(graph, start, end, path=None):
    """Enumerate all paths from start to end using DFS — O(V! worst case)

    Explores while backtracking to unvisit visited vertices.
    """
    if path is None:
        path = []
    path = path + [start]

    if start == end:
        return [path]

    paths = []
    for neighbor, weight in graph.get(start, []):
        if neighbor not in path:
            new_paths = find_all_paths(graph, neighbor, end, path)
            paths.extend(new_paths)
    return paths

# Graph definition
graph = {
    'A': [('B', 2), ('C', 4)],
    'B': [('A', 2), ('D', 3)],
    'C': [('A', 4), ('D', 1)],
    'D': [('B', 3), ('C', 1)],
}
all_paths = find_all_paths(graph, 'A', 'D')
for p in all_paths:
    print(" -> ".join(p))
# A -> B -> D
# A -> C -> D
```

### Intermediate: Building and Verifying a Minimum Spanning Tree

**Problem:** Manually trace Kruskal's algorithm on a 6-vertex graph and find the MST. Record the Union-Find state transitions as well.

```
Vertices: {0, 1, 2, 3, 4, 5}
Edges: (0,1,6) (0,2,1) (0,3,5) (1,2,5) (1,4,3) (2,3,5)
       (2,4,6) (2,5,4) (3,5,2) (4,5,6)

Steps:
  1. Sort edges by weight: (0,2,1) (3,5,2) (1,4,3) (2,5,4) ...
  2. (0,2,1): UF={0,2} others are singletons -> accept
  3. (3,5,2): UF={3,5} -> accept
  4. (1,4,3): UF={1,4} -> accept
  5. (2,5,4): merge {0,2} and {3,5} -> accept
  6. (1,2,5): merge {1,4} and {0,2,3,5} -> accept
  -> MST edges = 5 = V-1, total weight = 1+2+3+4+5 = 15
```

### Advanced: Computing Graph Diameter and Center

**Problem:** Implement an algorithm to compute the diameter (shortest distance between the two farthest vertices) and center (set of vertices with minimum eccentricity) of an arbitrary connected undirected graph.

```python
from collections import deque

def graph_diameter_and_center(graph, vertices):
    """Compute graph diameter and center — O(V * (V + E))

    Eccentricity: the distance from vertex v to the farthest vertex
    Diameter: the maximum eccentricity across all vertices
    Center: the set of vertices with minimum eccentricity
    """
    def bfs_distances(start):
        dist = {start: 0}
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in graph.get(u, []):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        return dist

    eccentricity = {}
    for v in vertices:
        distances = bfs_distances(v)
        eccentricity[v] = max(distances.values()) if distances else 0

    diameter = max(eccentricity.values())
    radius = min(eccentricity.values())
    center = [v for v in vertices if eccentricity[v] == radius]

    return diameter, radius, center, eccentricity

# Usage example: path graph 0-1-2-3-4
graph = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3]}
diam, rad, center, ecc = graph_diameter_and_center(graph, [0,1,2,3,4])
print(f"Diameter: {diam}, Radius: {rad}, Center: {center}")
# Diameter: 4, Radius: 2, Center: [2]
print(f"Eccentricity: {ecc}")
# {0: 4, 1: 3, 2: 2, 3: 3, 4: 4}
```

---

## 14. Advanced Topics

### 14.1 Euler Paths and Hamilton Paths

```
Euler path:
  A path that traverses every "edge" exactly once
  +-->--+
  A      B      Existence conditions:
  |    / |      - Undirected: 0 vertices with odd degree (circuit)
  |  /   |                    or 2 vertices with odd degree (path)
  v/     v      - Directed: in-degree = out-degree for all vertices (circuit)
  C-->--D                or source has out-in=1, sink has in-out=1 (path)

Hamilton path:
  A path that visits every "vertex" exactly once
  A -> B -> D -> C   Existence conditions:
                     No general polynomial-time decision method is known (NP-complete)
                     Dirac's theorem: if all vertices have degree >= V/2,
                     a Hamilton circuit exists
```

### 14.2 Graph Density and Real-World Scale

```
Density = E / E_max

Graph scale and representation selection guidelines:

  Scale       | Vertices    | Density | Recommended Representation
  ------------|-------------|---------|---------------------------
  Small       | ~100        | Any     | Adjacency matrix or list
  Medium      | ~10,000     | Sparse  | Adjacency list
  Large       | ~1,000,000  | Sparse  | Compressed adjacency list (CSR)
  Very large  | ~10^9       | V. sparse| Distributed graph (Pregel, etc.)

CSR (Compressed Sparse Row) format:
  - Vertex array: start position of each vertex's edge list
  - Edge array: all edge destinations stored contiguously
  - High memory locality and cache efficiency
```

### 14.3 Relation to Graph Databases

The demand for persisting graph structures as databases has grown with the proliferation of social networks and knowledge graphs. Below is a summary of representative graph databases and their query languages.

| Graph DB | Query Language | Primary Use |
|----------|---------------|-------------|
| Neo4j | Cypher | Social graphs, recommendations |
| Amazon Neptune | Gremlin / SPARQL | Knowledge graphs, fraud detection |
| JanusGraph | Gremlin | Large-scale distributed graphs |
| ArangoDB | AQL | Multi-model (document + graph) |

---


## FAQ

### Q1: What is the most important point when learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory, but by actually writing code and verifying its behavior.

### Q2: What are common mistakes that beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving to the next step.

### Q3: How is this used in practice?

Knowledge of this topic is frequently applied in daily development work. It becomes particularly important during code reviews and architecture design.

---

## 15. Summary

| Item | Key Point |
|------|-----------|
| Adjacency list | Optimal for sparse graphs. O(V+E) space. The default choice |
| Adjacency matrix | Dense graphs, O(1) edge existence check. Convenient for matrix operations |
| Edge list | For algorithms requiring sorting. Optimal for Kruskal |
| Union-Find | Connected component management. Nearly O(1) find/union |
| Topological sort | Linear ordering of a DAG. Dependency resolution |
| Strongly connected components | Decomposition of directed graphs. Kosaraju / Tarjan |
| Representation choice | Determined by graph density and operation patterns |
| Directed/undirected | Choose based on problem symmetry |
| Weights | Assign costs to edges. Used in shortest path problems |
| Exercises | Beginner: enumerate all paths, Intermediate: MST trace, Advanced: diameter and center |
| Advanced topics | Euler/Hamilton paths, CSR format, graph databases |

---

## Recommended Next Guides

- [Graph Traversal -- BFS/DFS](../02-algorithms/02-graph-traversal.md)
- [Shortest Paths -- Dijkstra, Bellman-Ford](../02-algorithms/03-shortest-path.md)

---

## References

1. Cormen, T.H. et al. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. -- Chapter 20 "Elementary Graph Algorithms", Chapter 21 "Minimum Spanning Trees"
2. Sedgewick, R. & Wayne, K. (2011). *Algorithms* (4th ed.). Addison-Wesley. -- Graph representations and algorithms
3. Skiena, S.S. (2020). *The Algorithm Design Manual* (3rd ed.). Springer. -- Practical graph design
4. Tarjan, R.E. (1972). "Depth-first search and linear graph algorithms." *SIAM Journal on Computing*, 1(2), 146-160.
5. Kosaraju, S.R. (1978). Unpublished manuscript, referenced in Aho, Hopcroft, Ullman (1983).
