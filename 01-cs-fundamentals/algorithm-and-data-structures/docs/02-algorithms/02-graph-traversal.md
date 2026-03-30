# Graph Traversal Algorithms

> Understand and master BFS, DFS, and topological sort for systematically visiting vertices and edges of a graph, including their implementations and application patterns

## What You Will Learn

1. Accurately understand the **operating principles, implementation, and computational complexity** of BFS (Breadth-First Search) and DFS (Depth-First Search)
2. Understand **graph representation methods** (adjacency list, adjacency matrix) and when to use each
3. Implement **topological sort** and its applications (dependency resolution, build systems, etc.)
4. Understand advanced traversal algorithms such as **strongly connected components, bipartite graph detection, and Eulerian paths**


## Prerequisites

Having the following knowledge will deepen your understanding of this guide:

- Basic programming knowledge
- Understanding of related foundational concepts
- Familiarity with the content in [Search Algorithms](./01-searching.md)

---

## 1. Graph Representations

### 1.1 Adjacency List and Adjacency Matrix

```
Graph G:
    0 --- 1
    |   / |
    |  /  |
    2 --- 3
        |
        4

Adjacency List:                Adjacency Matrix:
0: [1, 2]                       0  1  2  3  4
1: [0, 2, 3]                 0 [0, 1, 1, 0, 0]
2: [0, 1, 3]                 1 [1, 0, 1, 1, 0]
3: [1, 2, 4]                 2 [1, 1, 0, 1, 0]
4: [3]                        3 [0, 1, 1, 0, 1]
                              4 [0, 0, 0, 1, 0]
```

### 1.2 Comparison of Representations

| Property | Adjacency List | Adjacency Matrix |
|:---|:---|:---|
| Space complexity | O(V + E) | O(V^2) |
| Edge existence check | O(degree(v)) | O(1) |
| Enumerate all neighbors | O(degree(v)) | O(V) |
| Add edge | O(1) | O(1) |
| Remove edge | O(degree(v)) | O(1) |
| Best suited for | Sparse graphs (E << V^2) | Dense graphs (E ~ V^2) |
| Memory efficiency | High | Low (for sparse graphs) |

### 1.3 Implementation of Each Representation in Python

```python
from collections import defaultdict, deque

class Graph:
    """Graph representation using an adjacency list"""
    def __init__(self, directed=False):
        self.adj = defaultdict(list)
        self.directed = directed

    def add_edge(self, u, v, weight=None):
        if weight is not None:
            self.adj[u].append((v, weight))
            if not self.directed:
                self.adj[v].append((u, weight))
        else:
            self.adj[u].append(v)
            if not self.directed:
                self.adj[v].append(u)

    def remove_edge(self, u, v):
        self.adj[u] = [x for x in self.adj[u] if x != v]
        if not self.directed:
            self.adj[v] = [x for x in self.adj[v] if x != u]

    def vertices(self):
        verts = set()
        for u in self.adj:
            verts.add(u)
            for v in self.adj[u]:
                if isinstance(v, tuple):
                    verts.add(v[0])
                else:
                    verts.add(v)
        return verts

    def degree(self, v):
        """Return the degree of vertex v"""
        return len(self.adj[v])

    def has_edge(self, u, v):
        """Check whether edge (u, v) exists"""
        return v in self.adj[u]

    def __repr__(self):
        return '\n'.join(f'{u}: {neighbors}' for u, neighbors in self.adj.items())


class AdjacencyMatrix:
    """Graph representation using an adjacency matrix"""
    def __init__(self, n, directed=False):
        self.n = n
        self.directed = directed
        self.matrix = [[0] * n for _ in range(n)]

    def add_edge(self, u, v, weight=1):
        self.matrix[u][v] = weight
        if not self.directed:
            self.matrix[v][u] = weight

    def remove_edge(self, u, v):
        self.matrix[u][v] = 0
        if not self.directed:
            self.matrix[v][u] = 0

    def has_edge(self, u, v):
        return self.matrix[u][v] != 0

    def neighbors(self, u):
        return [v for v in range(self.n) if self.matrix[u][v] != 0]

    def degree(self, u):
        return sum(1 for v in range(self.n) if self.matrix[u][v] != 0)
```

### 1.4 Edge List Representation

```python
class EdgeListGraph:
    """Edge list graph representation (useful for Kruskal's, etc.)"""
    def __init__(self, directed=False):
        self.edges = []
        self.directed = directed

    def add_edge(self, u, v, weight=1):
        self.edges.append((u, v, weight))
        if not self.directed:
            self.edges.append((v, u, weight))

    def vertices(self):
        verts = set()
        for u, v, _ in self.edges:
            verts.add(u)
            verts.add(v)
        return verts

    def neighbors(self, u):
        return [(v, w) for src, v, w in self.edges if src == u]
```

### 1.5 Practical Guidelines for Choosing a Representation

```
Decision Flow:

  Is the graph sparse? (E << V^2)
    |-- YES -> Adjacency list (default choice)
    +-- NO  -> Dense graph?
              |-- YES -> Adjacency matrix
              +-- Either -> Adjacency list is the safe choice

  Are edge existence checks frequent?
    |-- YES -> Adjacency matrix or set-based adjacency list
    +-- NO  -> Adjacency list

  Do you need to sort and process edges?
    |-- YES -> Edge list (Kruskal's, etc.)
    +-- NO  -> Adjacency list
```

---

## 2. BFS (Breadth-First Search)

Uses a queue to visit vertices in order of increasing distance from the source. Guarantees shortest paths (unweighted).

```
Source: 0

Level 0:  [0]               Queue: [0]
Level 1:  [1, 2]            Queue: [1, 2]
Level 2:  [3]               Queue: [3]
Level 3:  [4]               Queue: [4]

Visit order: 0 -> 1 -> 2 -> 3 -> 4

    0 --- 1
    |   / |       BFS spreads like "ripples"
    |  /  |       Visits all vertices at the same distance first
    2 --- 3
          |
          4
```

### 2.1 Basic Implementation

```python
def bfs(graph: dict, start) -> list:
    """Breadth-First Search - O(V + E)"""
    visited = set([start])
    queue = deque([start])
    order = []

    while queue:
        vertex = queue.popleft()
        order.append(vertex)

        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return order

# Shortest path (unweighted)
def bfs_shortest_path(graph: dict, start, end) -> list:
    """Reconstruct the shortest path using BFS"""
    visited = {start}
    queue = deque([(start, [start])])

    while queue:
        vertex, path = queue.popleft()
        if vertex == end:
            return path

        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return []  # Unreachable

# Shortest distances (memory-efficient version)
def bfs_shortest_distance(graph: dict, start) -> dict:
    """Compute shortest distances to all vertices (memory-efficient)"""
    dist = {start: 0}
    queue = deque([start])

    while queue:
        vertex = queue.popleft()
        for neighbor in graph[vertex]:
            if neighbor not in dist:
                dist[neighbor] = dist[vertex] + 1
                queue.append(neighbor)

    return dist

# Usage example
g = {0: [1, 2], 1: [0, 2, 3], 2: [0, 1, 3], 3: [1, 2, 4], 4: [3]}
print(bfs(g, 0))                    # [0, 1, 2, 3, 4]
print(bfs_shortest_path(g, 0, 4))   # [0, 1, 3, 4]
print(bfs_shortest_distance(g, 0))  # {0: 0, 1: 1, 2: 1, 3: 2, 4: 3}
```

### 2.2 BFS Application: Level-Order Traversal

```python
def bfs_levels(graph: dict, start) -> list:
    """Group vertices by level (distance)"""
    visited = {start}
    queue = deque([start])
    levels = []

    while queue:
        level_size = len(queue)
        current_level = []

        for _ in range(level_size):
            vertex = queue.popleft()
            current_level.append(vertex)

            for neighbor in graph[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        levels.append(current_level)

    return levels

print(bfs_levels(g, 0))  # [[0], [1, 2], [3], [4]]
```

### 2.3 BFS Application: Bipartite Graph Detection

A bipartite graph is one whose vertex set can be partitioned into two groups such that no edge connects vertices within the same group. BFS level assignment can be used for detection.

```python
def is_bipartite(graph: dict, vertices: set) -> tuple:
    """Bipartite graph detection - O(V + E)
    Returns: (is_bipartite, 2-coloring dictionary)
    """
    color = {}

    for start in vertices:
        if start in color:
            continue

        # 2-color using BFS
        color[start] = 0
        queue = deque([start])

        while queue:
            v = queue.popleft()
            for neighbor in graph.get(v, []):
                if neighbor not in color:
                    color[neighbor] = 1 - color[v]
                    queue.append(neighbor)
                elif color[neighbor] == color[v]:
                    return False, {}

    return True, color

# Bipartite example (trees are always bipartite)
g_bipartite = {
    0: [1, 3],
    1: [0, 2],
    2: [1, 3],
    3: [0, 2],
}
result, coloring = is_bipartite(g_bipartite, {0, 1, 2, 3})
print(f"Bipartite: {result}")   # True
print(f"Coloring: {coloring}")  # {0: 0, 1: 1, 3: 1, 2: 0}

# Non-bipartite example (odd cycle)
g_odd = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
result, _ = is_bipartite(g_odd, {0, 1, 2})
print(f"Bipartite: {result}")  # False
```

### 2.4 BFS Application: Multi-Source BFS

A technique that runs BFS simultaneously from multiple source vertices. Computes the distance from each vertex to the nearest source in one pass.

```python
def multi_source_bfs(graph: dict, sources: list) -> dict:
    """Multi-source BFS - O(V + E)
    Computes the distance from each vertex to the nearest source
    """
    dist = {}
    queue = deque()

    for s in sources:
        dist[s] = 0
        queue.append(s)

    while queue:
        v = queue.popleft()
        for neighbor in graph.get(v, []):
            if neighbor not in dist:
                dist[neighbor] = dist[v] + 1
                queue.append(neighbor)

    return dist

# Practical example: shortest distance from each cell to the nearest facility on a grid
# Used in game development for generating "distance to nearest enemy/ally" maps
grid_graph = {
    (0,0): [(0,1), (1,0)],
    (0,1): [(0,0), (0,2), (1,1)],
    (0,2): [(0,1), (1,2)],
    (1,0): [(0,0), (1,1), (2,0)],
    (1,1): [(1,0), (0,1), (1,2), (2,1)],
    (1,2): [(1,1), (0,2), (2,2)],
    (2,0): [(1,0), (2,1)],
    (2,1): [(2,0), (1,1), (2,2)],
    (2,2): [(2,1), (1,2)],
}
sources = [(0,0), (2,2)]
distances = multi_source_bfs(grid_graph, sources)
print(distances)
# {(0,0): 0, (2,2): 0, (0,1): 1, (1,0): 1, (2,1): 1, (1,2): 1,
#  (1,1): 2, (0,2): 2, (2,0): 2}
```

### 2.5 0-1 BFS

A technique for finding shortest distances in O(V+E) on graphs where all edge weights are either 0 or 1, using a deque instead of Dijkstra's algorithm.

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

# Example: grid where wall destruction costs 1, corridor movement costs 0
graph_01 = {
    'A': [('B', 0), ('C', 1)],
    'B': [('A', 0), ('D', 1)],
    'C': [('A', 1), ('D', 0)],
    'D': [('B', 1), ('C', 0)],
}
print(bfs_01(graph_01, 'A'))
# {'A': 0, 'B': 0, 'C': 1, 'D': 1}
```

---

## 3. DFS (Depth-First Search)

Uses a stack (or recursion) to go as deep as possible before backtracking.

```
Source: 0

Exploration flow:
  0 -> 1 -> 2 (0 already visited) -> 3 -> 4 (dead end)
                                ^ backtrack
                             3 (all neighbors visited)
                             ^ backtrack
                          ...done

Visit order: 0 -> 1 -> 2 -> 3 -> 4

    0 --- 1
    |   / |       DFS goes deep along a single path
    |  /  |       Backtracks at dead ends
    2 --- 3
          |
          4
```

### 3.1 Basic Implementation

```python
def dfs_recursive(graph: dict, start, visited=None) -> list:
    """DFS recursive version - O(V + E)"""
    if visited is None:
        visited = set()

    visited.add(start)
    order = [start]

    for neighbor in graph[start]:
        if neighbor not in visited:
            order.extend(dfs_recursive(graph, neighbor, visited))

    return order

def dfs_iterative(graph: dict, start) -> list:
    """DFS iterative version (using a stack)"""
    visited = set()
    stack = [start]
    order = []

    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            order.append(vertex)
            # Push in reverse order to visit in lexicographic order
            for neighbor in reversed(graph[vertex]):
                if neighbor not in visited:
                    stack.append(neighbor)

    return order

g = {0: [1, 2], 1: [0, 2, 3], 2: [0, 1, 3], 3: [1, 2, 4], 4: [3]}
print(dfs_recursive(g, 0))  # [0, 1, 2, 3, 4]
print(dfs_iterative(g, 0))  # [0, 1, 2, 3, 4]
```

### 3.2 DFS Timestamps (Discovery and Finish Times)

```python
class DFSWithTimestamp:
    """DFS with timestamps
    Collects information needed for edge classification and topological sort
    """
    def __init__(self, graph: dict):
        self.graph = graph
        self.discovery = {}   # Discovery time
        self.finish = {}      # Finish time
        self.parent = {}      # Parent in the DFS tree
        self.time = 0

    def dfs(self):
        """Run DFS from all vertices"""
        all_vertices = set(self.graph.keys())
        for neighbors in self.graph.values():
            all_vertices.update(neighbors)

        for v in all_vertices:
            if v not in self.discovery:
                self.parent[v] = None
                self._visit(v)

    def _visit(self, u):
        self.time += 1
        self.discovery[u] = self.time

        for v in self.graph.get(u, []):
            if v not in self.discovery:
                self.parent[v] = u
                self._visit(v)

        self.time += 1
        self.finish[u] = self.time

    def classify_edge(self, u, v):
        """Classify edge (u, v)"""
        if self.parent.get(v) == u:
            return "tree"      # Tree edge
        elif (self.discovery[u] < self.discovery[v] and
              self.finish[u] > self.finish[v]):
            return "forward"   # Forward edge
        elif (self.discovery[u] > self.discovery[v] and
              self.finish[u] < self.finish[v]):
            return "back"      # Back edge (indicates a cycle)
        else:
            return "cross"     # Cross edge

# Usage example
g_directed = {
    'A': ['B', 'C'],
    'B': ['D'],
    'C': ['D'],
    'D': ['E'],
    'E': [],
}
dfs_ts = DFSWithTimestamp(g_directed)
dfs_ts.dfs()
print("Discovery times:", dfs_ts.discovery)
print("Finish times:", dfs_ts.finish)
```

### 3.3 Edge Classification

```
Edge classification in DFS on directed graphs:

  Tree Edge     : An edge in the DFS tree. Discovers a new vertex
  Back Edge     : An edge back to an ancestor. Indicates cycle existence
  Forward Edge  : An edge to a descendant (other than tree edges)
  Cross Edge    : An edge to a different subtree

  Classification criteria (timestamps d[u], f[u]):
  - Tree/Forward: d[u] < d[v] < f[v] < f[u]
  - Back:         d[v] < d[u] < f[u] < f[v]
  - Cross:        d[v] < f[v] < d[u] < f[u]

  Cycle detection: A back edge exists <=> A cycle exists
```

### 3.4 DFS Application: Finding Connected Components

```python
def find_connected_components(graph: dict, vertices: set) -> list:
    """Enumerate connected components of an undirected graph"""
    visited = set()
    components = []

    for v in vertices:
        if v not in visited:
            component = []
            stack = [v]
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    component.append(node)
                    for neighbor in graph.get(node, []):
                        if neighbor not in visited:
                            stack.append(neighbor)
            components.append(component)

    return components

# Disconnected graph
g2 = {0: [1], 1: [0], 2: [3], 3: [2], 4: []}
print(find_connected_components(g2, {0,1,2,3,4}))
# [[0, 1], [2, 3], [4]]
```

### 3.5 DFS Application: Cycle Detection

```python
def has_cycle_directed(graph: dict) -> bool:
    """Cycle detection in a directed graph (3-color method)"""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = defaultdict(int)  # All vertices start WHITE

    def dfs(u):
        color[u] = GRAY  # Currently being explored
        for v in graph.get(u, []):
            if color[v] == GRAY:  # Returned to a vertex under exploration -> cycle
                return True
            if color[v] == WHITE and dfs(v):
                return True
        color[u] = BLACK  # Exploration complete
        return False

    for vertex in graph:
        if color[vertex] == WHITE:
            if dfs(vertex):
                return True
    return False

def has_cycle_undirected(graph: dict) -> bool:
    """Cycle detection in an undirected graph"""
    visited = set()

    def dfs(v, parent):
        visited.add(v)
        for neighbor in graph.get(v, []):
            if neighbor not in visited:
                if dfs(neighbor, v):
                    return True
            elif neighbor != parent:
                return True  # Visited vertex other than parent -> cycle
        return False

    for vertex in graph:
        if vertex not in visited:
            if dfs(vertex, None):
                return True
    return False

# Cycle present: 0 -> 1 -> 2 -> 0
g_cycle = {0: [1], 1: [2], 2: [0]}
print(has_cycle_directed(g_cycle))  # True

# No cycle: DAG
g_dag = {0: [1], 1: [2], 2: []}
print(has_cycle_directed(g_dag))  # False

# Undirected graph cycle
g_undirected_cycle = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
print(has_cycle_undirected(g_undirected_cycle))  # True
```

### 3.6 DFS Application: Recovering the Actual Cycle Path

```python
def find_cycle_directed(graph: dict) -> list:
    """Find one cycle in a directed graph and return the path"""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = defaultdict(int)
    parent = {}
    cycle_start = None
    cycle_end = None

    def dfs(u):
        nonlocal cycle_start, cycle_end
        color[u] = GRAY
        for v in graph.get(u, []):
            if color[v] == GRAY:
                cycle_start = v
                cycle_end = u
                return True
            if color[v] == WHITE:
                parent[v] = u
                if dfs(v):
                    return True
        color[u] = BLACK
        return False

    for vertex in graph:
        if color[vertex] == WHITE:
            parent[vertex] = None
            if dfs(vertex):
                break

    if cycle_start is None:
        return []

    # Recover the cycle path
    cycle = [cycle_start]
    current = cycle_end
    while current != cycle_start:
        cycle.append(current)
        current = parent[current]
    cycle.reverse()
    return cycle

g_cycle = {0: [1], 1: [2], 2: [3], 3: [1]}
print(find_cycle_directed(g_cycle))  # [1, 2, 3]
```

---

## 4. Topological Sort

Arranges vertices of a DAG (Directed Acyclic Graph) in an order consistent with edge directions.

```
  Course dependencies:
  Math -> Physics -> Quantum Mechanics
  Math -> Linear Algebra -> Quantum Mechanics
  Programming -> Algorithms

  DAG:
  Math -----> Physics ----------> Quantum Mechanics
    |                              ^
    +-----> Linear Algebra --------+
  Programming -----> Algorithms

  One possible topological order:
  [Math, Programming, Physics, Linear Algebra, Algorithms, Quantum Mechanics]
```

### 4.1 DFS-Based Implementation (Tarjan)

```python
def topological_sort_dfs(graph: dict) -> list:
    """DFS-based topological sort - O(V + E)"""
    visited = set()
    stack = []  # Accumulate results in reverse order

    def dfs(v):
        visited.add(v)
        for neighbor in graph.get(v, []):
            if neighbor not in visited:
                dfs(neighbor)
        stack.append(v)  # Add on backtrack

    # DFS from all vertices
    all_vertices = set(graph.keys())
    for v in graph.values():
        all_vertices.update(v)

    for vertex in all_vertices:
        if vertex not in visited:
            dfs(vertex)

    return stack[::-1]  # Reverse gives the answer

dag = {
    "Math": ["Physics", "Linear Algebra"],
    "Physics": ["Quantum Mechanics"],
    "Linear Algebra": ["Quantum Mechanics"],
    "Programming": ["Algorithms"],
    "Quantum Mechanics": [],
    "Algorithms": [],
}
print(topological_sort_dfs(dag))
```

### 4.2 Kahn's Algorithm (BFS-Based)

```python
def topological_sort_kahn(graph: dict) -> list:
    """Kahn's algorithm (in-degree based) - O(V + E)"""
    # Collect all vertices
    all_vertices = set(graph.keys())
    for neighbors in graph.values():
        all_vertices.update(neighbors)

    # Compute in-degrees
    in_degree = {v: 0 for v in all_vertices}
    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1

    # Enqueue vertices with in-degree 0
    queue = deque([v for v in all_vertices if in_degree[v] == 0])
    result = []

    while queue:
        vertex = queue.popleft()
        result.append(vertex)

        for neighbor in graph.get(vertex, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(result) != len(all_vertices):
        raise ValueError("Cycle detected")

    return result

print(topological_sort_kahn(dag))
```

### 4.3 Enumerating All Topological Orders

```python
def all_topological_sorts(graph: dict) -> list:
    """Enumerate all topological orders (backtracking)"""
    all_vertices = set(graph.keys())
    for neighbors in graph.values():
        all_vertices.update(neighbors)

    in_degree = {v: 0 for v in all_vertices}
    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1

    result = []
    current = []
    visited = set()

    def backtrack():
        if len(current) == len(all_vertices):
            result.append(current[:])
            return

        for v in sorted(all_vertices):
            if v not in visited and in_degree[v] == 0:
                # Choose
                visited.add(v)
                current.append(v)
                for neighbor in graph.get(v, []):
                    in_degree[neighbor] -= 1

                backtrack()

                # Undo
                visited.discard(v)
                current.pop()
                for neighbor in graph.get(v, []):
                    in_degree[neighbor] += 1

    backtrack()
    return result

# Small DAG example
small_dag = {'A': ['C'], 'B': ['C'], 'C': []}
print(all_topological_sorts(small_dag))
# [['A', 'B', 'C'], ['B', 'A', 'C']]
```

### 4.4 Practical Applications of Topological Sort

```python
# Practical example 1: Build system dependency resolution
build_deps = {
    "utils.o": ["utils.c", "utils.h"],
    "main.o": ["main.c", "utils.h"],
    "app": ["main.o", "utils.o"],
    "utils.c": [],
    "utils.h": [],
    "main.c": [],
}

def build_order(deps: dict) -> list:
    """Determine build order"""
    # Reverse dependencies (A depends on B -> B -> A)
    graph = defaultdict(list)
    all_files = set(deps.keys())
    for target, sources in deps.items():
        for src in sources:
            graph[src].append(target)
            all_files.add(src)

    # Kahn's algorithm
    in_deg = {f: 0 for f in all_files}
    for u in graph:
        for v in graph[u]:
            in_deg[v] += 1

    queue = deque([f for f in all_files if in_deg[f] == 0])
    order = []
    while queue:
        f = queue.popleft()
        order.append(f)
        for dep in graph[f]:
            in_deg[dep] -= 1
            if in_deg[dep] == 0:
                queue.append(dep)

    return order

print(build_order(build_deps))
# ['utils.c', 'utils.h', 'main.c', 'utils.o', 'main.o', 'app']


# Practical example 2: Task scheduling (identifying parallelizable tasks)
def schedule_tasks_parallel(graph: dict) -> list:
    """Group parallelizable tasks by stage"""
    all_vertices = set(graph.keys())
    for neighbors in graph.values():
        all_vertices.update(neighbors)

    in_degree = {v: 0 for v in all_vertices}
    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1

    queue = deque([v for v in all_vertices if in_degree[v] == 0])
    stages = []

    while queue:
        stage = []
        next_queue = deque()
        while queue:
            v = queue.popleft()
            stage.append(v)
            for neighbor in graph.get(v, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    next_queue.append(neighbor)
        stages.append(stage)
        queue = next_queue

    return stages

task_deps = {
    'compile_a': ['link'],
    'compile_b': ['link'],
    'compile_c': ['link'],
    'link': ['test'],
    'test': ['deploy'],
    'deploy': [],
}
print(schedule_tasks_parallel(task_deps))
# [['compile_a', 'compile_b', 'compile_c'], ['link'], ['test'], ['deploy']]
```

---

## 5. Strongly Connected Components (SCC)

In a directed graph, a strongly connected component is a maximal set of vertices such that every vertex is reachable from every other vertex in the set.

### 5.1 Kosaraju's Algorithm

```python
def kosaraju_scc(graph: dict) -> list:
    """Kosaraju's algorithm - O(V + E)
    1. Run DFS on the original graph -> record vertices in finish order
    2. Create the transpose of the graph
    3. Run DFS on the transpose graph in reverse finish order -> each DFS yields an SCC
    """
    # Collect all vertices
    all_vertices = set(graph.keys())
    for neighbors in graph.values():
        all_vertices.update(neighbors)

    # Step 1: DFS on the original graph, record finish order
    visited = set()
    finish_order = []

    def dfs1(v):
        visited.add(v)
        for neighbor in graph.get(v, []):
            if neighbor not in visited:
                dfs1(neighbor)
        finish_order.append(v)

    for v in all_vertices:
        if v not in visited:
            dfs1(v)

    # Step 2: Create the transpose graph
    transpose = defaultdict(list)
    for u in graph:
        for v in graph[u]:
            transpose[v].append(u)

    # Step 3: DFS on the transpose in reverse order
    visited.clear()
    sccs = []

    def dfs2(v, component):
        visited.add(v)
        component.append(v)
        for neighbor in transpose.get(v, []):
            if neighbor not in visited:
                dfs2(neighbor, component)

    for v in reversed(finish_order):
        if v not in visited:
            component = []
            dfs2(v, component)
            sccs.append(component)

    return sccs

# Usage example
g_scc = {
    0: [1],
    1: [2],
    2: [0, 3],  # 0->1->2->0 forms an SCC
    3: [4],
    4: [5],
    5: [3],     # 3->4->5->3 forms an SCC
}
print(kosaraju_scc(g_scc))
# [[0, 2, 1], [3, 5, 4]] (order may vary)
```

### 5.2 Tarjan's SCC Algorithm

```python
def tarjan_scc(graph: dict) -> list:
    """Tarjan's SCC algorithm - O(V + E)
    Finds all SCCs in a single DFS pass (more practical than Kosaraju's)
    """
    index_counter = [0]
    stack = []
    lowlink = {}
    index = {}
    on_stack = set()
    sccs = []

    all_vertices = set(graph.keys())
    for neighbors in graph.values():
        all_vertices.update(neighbors)

    def strongconnect(v):
        index[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)

        for w in graph.get(v, []):
            if w not in index:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], index[w])

        # If v is a root, pop the SCC
        if lowlink[v] == index[v]:
            component = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                component.append(w)
                if w == v:
                    break
            sccs.append(component)

    for v in all_vertices:
        if v not in index:
            strongconnect(v)

    return sccs

print(tarjan_scc(g_scc))
```

---

## 6. BFS vs DFS Comparison

| Property | BFS | DFS |
|:---|:---|:---|
| Data structure | Queue (FIFO) | Stack (LIFO) / Recursion |
| Visit order | Nearest vertices first | Deepest vertices first |
| Shortest path (unweighted) | Guaranteed | Not guaranteed |
| Memory usage | O(V) (proportional to width) | O(V) (proportional to depth) |
| Tree traversal | Level-order | Pre-order / In-order / Post-order |
| Implementation ease | Slightly more complex | Concise with recursion |
| Cycle detection | Possible | Easy with 3-color method |
| Edge classification | Not possible | Possible (4 types) |
| Completeness (infinite graphs) | Guaranteed | Not guaranteed |

## Use Cases for Traversal Algorithms

| Use Case | Recommended Algorithm | Reason |
|:---|:---|:---|
| Shortest path (unweighted) | BFS | Guarantees shortest distance |
| Connected components | DFS | Simpler implementation |
| Topological sort | DFS / Kahn | Ordering of DAGs |
| Cycle detection | DFS (3-color) | Easy back edge detection |
| Bipartite check | BFS | Level-based 2-coloring |
| Shortest path in a maze | BFS | Shortest search on grids |
| Puzzle solving | DFS + Backtracking | Exhaustive state space search |
| Web crawler | BFS | Shallow pages first |
| Strongly connected components | DFS (Tarjan/Kosaraju) | Completed in 1-2 DFS passes |
| Articulation points and bridges | DFS | Detected using lowlink values |
| Eulerian path | DFS (Hierholzer) | Path traversing each edge once |

---

## 7. BFS on Grids

```python
def bfs_grid(grid: list, start: tuple, end: tuple) -> int:
    """Shortest path on a 2D grid (BFS)"""
    rows, cols = len(grid), len(grid[0])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up

    visited = {start}
    queue = deque([(start, 0)])  # (position, distance)

    while queue:
        (r, c), dist = queue.popleft()
        if (r, c) == end:
            return dist

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols and
                grid[nr][nc] != 1 and (nr, nc) not in visited):
                visited.add((nr, nc))
                queue.append(((nr, nc), dist + 1))

    return -1  # Unreachable

# 0: corridor, 1: wall
maze = [
    [0, 0, 0, 0],
    [1, 1, 0, 1],
    [0, 0, 0, 0],
    [0, 1, 1, 0],
]
print(bfs_grid(maze, (0, 0), (3, 3)))  # 6
```

### 8-Directional Grid BFS

```python
def bfs_grid_8dir(grid: list, start: tuple, end: tuple) -> int:
    """Shortest path with 8-directional movement"""
    rows, cols = len(grid), len(grid[0])
    # 8 directions: up/down/left/right + 4 diagonals
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    visited = {start}
    queue = deque([(start, 0)])

    while queue:
        (r, c), dist = queue.popleft()
        if (r, c) == end:
            return dist

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols and
                grid[nr][nc] != 1 and (nr, nc) not in visited):
                visited.add((nr, nc))
                queue.append(((nr, nc), dist + 1))

    return -1
```

### Grid BFS with Path Reconstruction

```python
def bfs_grid_with_path(grid: list, start: tuple, end: tuple) -> list:
    """Reconstruct the shortest path on a 2D grid"""
    rows, cols = len(grid), len(grid[0])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    visited = {start}
    parent = {start: None}
    queue = deque([start])

    while queue:
        r, c = queue.popleft()
        if (r, c) == end:
            # Path reconstruction
            path = []
            current = end
            while current is not None:
                path.append(current)
                current = parent[current]
            return path[::-1]

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols and
                grid[nr][nc] != 1 and (nr, nc) not in visited):
                visited.add((nr, nc))
                parent[(nr, nc)] = (r, c)
                queue.append((nr, nc))

    return []  # Unreachable

path = bfs_grid_with_path(maze, (0, 0), (3, 3))
print(path)
# [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 3), (3, 3)]
```

---

## 8. Articulation Points and Bridges

Detect vertices (articulation points) and edges (bridges) in an undirected graph whose removal disconnects the graph. Used for network vulnerability analysis.

```python
def find_articulation_points_and_bridges(graph: dict, vertices: set):
    """Detect articulation points and bridges - O(V + E)"""
    discovery = {}
    low = {}
    parent = {}
    ap = set()          # Articulation points
    bridges = []        # Bridges
    time_counter = [0]

    def dfs(u):
        discovery[u] = low[u] = time_counter[0]
        time_counter[0] += 1
        children = 0

        for v in graph.get(u, []):
            if v not in discovery:
                children += 1
                parent[v] = u
                dfs(v)
                low[u] = min(low[u], low[v])

                # u is root and has 2+ children -> articulation point
                if parent[u] is None and children > 1:
                    ap.add(u)

                # u is not root and child's lowlink >= u's discovery -> articulation point
                if parent[u] is not None and low[v] >= discovery[u]:
                    ap.add(u)

                # Bridge condition: low[v] > discovery[u]
                if low[v] > discovery[u]:
                    bridges.append((u, v))
            elif v != parent.get(u):
                low[u] = min(low[u], discovery[v])

    for v in vertices:
        if v not in discovery:
            parent[v] = None
            dfs(v)

    return ap, bridges

# Usage example
g_bridge = {
    0: [1, 2],
    1: [0, 2],
    2: [0, 1, 3],
    3: [2, 4],
    4: [3],
}
ap, bridges = find_articulation_points_and_bridges(g_bridge, {0,1,2,3,4})
print(f"Articulation points: {ap}")  # {2, 3}
print(f"Bridges: {bridges}")         # [(2, 3), (3, 4)]
```

---

## 9. Eulerian Paths and Circuits

A path that traverses every edge exactly once (Eulerian path) and one that returns to the starting vertex (Eulerian circuit).

```
Conditions for Eulerian path existence:
  Undirected graph: 0 vertices of odd degree (circuit) or 2 vertices of odd degree (path)
  Directed graph: in-degree = out-degree for all vertices (circuit)
              or 1 vertex with out-degree = in-degree + 1, 1 vertex with in-degree = out-degree + 1 (path)
```

```python
def find_euler_path(graph: dict) -> list:
    """Find an Eulerian path/circuit using Hierholzer's algorithm
    graph: adjacency list (edges are consumed during traversal)
    """
    # Copy the graph (edges are consumed)
    adj = defaultdict(list)
    for u in graph:
        for v in graph[u]:
            adj[u].append(v)

    # Determine the starting vertex
    odd_vertices = [v for v in adj if len(adj[v]) % 2 == 1]
    if len(odd_vertices) == 0:
        start = next(iter(adj))  # Eulerian circuit: any vertex
    elif len(odd_vertices) == 2:
        start = odd_vertices[0]   # Eulerian path: start from an odd-degree vertex
    else:
        return []  # No Eulerian path/circuit exists

    stack = [start]
    path = []

    while stack:
        v = stack[-1]
        if adj[v]:
            u = adj[v].pop()
            # For undirected graphs, also remove the reverse edge
            adj[u].remove(v)
            stack.append(u)
        else:
            path.append(stack.pop())

    return path[::-1]

# Konigsberg bridges (a variant of the problem Euler proved unsolvable)
g_euler = {
    'A': ['B', 'C'],
    'B': ['A', 'C', 'D', 'D'],
    'C': ['A', 'B', 'D'],
    'D': ['B', 'B', 'C'],
}
path = find_euler_path(g_euler)
print(f"Eulerian path: {path}")
```

---

## 10. Practical Application Patterns

### 10.1 Social Network Analysis

```python
def mutual_friends(graph: dict, u, v) -> set:
    """Find mutual friends"""
    friends_u = set(graph.get(u, []))
    friends_v = set(graph.get(v, []))
    return friends_u & friends_v

def degrees_of_separation(graph: dict, u, v) -> int:
    """Degrees of separation between two people (shortest distance)"""
    if u == v:
        return 0
    visited = {u}
    queue = deque([(u, 0)])
    while queue:
        node, dist = queue.popleft()
        for neighbor in graph.get(node, []):
            if neighbor == v:
                return dist + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return -1  # Disconnected

def influence_score(graph: dict, start, max_depth: int = 3) -> int:
    """Influence score: number of vertices reachable within max_depth hops"""
    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
    return len(visited) - 1  # Exclude self
```

### 10.2 Basic Web Crawler Structure

```python
def web_crawler_bfs(start_url: str, max_pages: int = 100) -> list:
    """BFS-based web crawler (conceptual implementation)"""
    visited = {start_url}
    queue = deque([start_url])
    crawled = []

    while queue and len(crawled) < max_pages:
        url = queue.popleft()
        crawled.append(url)

        # In practice, fetch the page via HTTP request and extract links
        links = extract_links(url)  # Hypothetical function

        for link in links:
            if link not in visited:
                visited.add(link)
                queue.append(link)

    return crawled

def extract_links(url):
    """Hypothetical link extraction (actual implementation requires HTML parsing)"""
    return []  # Placeholder
```

### 10.3 Dependency Deadlock Detection

```python
def detect_deadlock(resource_graph: dict) -> list:
    """Detect deadlock (cycle) in a resource allocation graph"""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = defaultdict(int)
    path = []
    cycle = []

    def dfs(u):
        color[u] = GRAY
        path.append(u)
        for v in resource_graph.get(u, []):
            if color[v] == GRAY:
                # Cycle found -> deadlock
                idx = path.index(v)
                cycle.extend(path[idx:])
                return True
            if color[v] == WHITE and dfs(v):
                return True
        path.pop()
        color[u] = BLACK
        return False

    for vertex in resource_graph:
        if color[vertex] == WHITE:
            if dfs(vertex):
                return cycle

    return []  # No deadlock

# Process P1 -> Resource R1 -> Process P2 -> Resource R2 -> P1
resource_graph = {
    'P1': ['R1'],
    'R1': ['P2'],
    'P2': ['R2'],
    'R2': ['P1'],
}
deadlock = detect_deadlock(resource_graph)
print(f"Deadlock: {deadlock}")
# Deadlock: ['P1', 'R1', 'P2', 'R2']
```

---

## 11. Anti-Patterns

### Anti-Pattern 1: Incorrect Timing of visited Checks

```python
# BAD: Check visited when dequeuing
# -> Same vertex enters the queue multiple times, wasting memory and time
def bad_bfs(graph, start):
    queue = deque([start])
    visited = set()
    while queue:
        v = queue.popleft()
        if v in visited:  # Checking here -> slow
            continue
        visited.add(v)
        for n in graph[v]:
            queue.append(n)  # Duplicate additions

# GOOD: Check visited when enqueuing
def good_bfs(graph, start):
    queue = deque([start])
    visited = {start}  # Mark on addition
    while queue:
        v = queue.popleft()
        for n in graph[v]:
            if n not in visited:
                visited.add(n)  # Mark here
                queue.append(n)
```

### Anti-Pattern 2: Stack Overflow with Recursive DFS

```python
# BAD: Recursive DFS on large graphs -> RecursionError
import sys
sys.setrecursionlimit(10**6)  # Stopgap measure, but unstable

# GOOD: Use iterative DFS
def safe_dfs(graph, start):
    visited = set()
    stack = [start]
    while stack:
        v = stack.pop()
        if v not in visited:
            visited.add(v)
            for n in graph.get(v, []):
                if n not in visited:
                    stack.append(n)
```

### Anti-Pattern 3: Using Adjacency Matrix for Sparse Graphs

```python
# BAD: Adjacency matrix for a sparse graph with 100,000 vertices
# -> Memory: 100,000 x 100,000 = 10^10 elements -> out of memory
n = 100000
matrix = [[0] * n for _ in range(n)]  # MemoryError!

# GOOD: Use an adjacency list
graph = defaultdict(list)
# With E << V^2, memory is O(V + E)
```

### Anti-Pattern 4: Path Recording Method in BFS

```python
# BAD: Copy path each time -> O(V^2) memory
def bad_bfs_path(graph, start, end):
    queue = deque([(start, [start])])  # Copies entire path
    visited = {start}
    while queue:
        v, path = queue.popleft()
        if v == end:
            return path
        for n in graph[v]:
            if n not in visited:
                visited.add(n)
                queue.append((n, path + [n]))  # O(V) copy

# GOOD: Use a predecessor dictionary and reconstruct afterward -> O(V) memory
def good_bfs_path(graph, start, end):
    prev = {start: None}
    queue = deque([start])
    while queue:
        v = queue.popleft()
        if v == end:
            path = []
            while v is not None:
                path.append(v)
                v = prev[v]
            return path[::-1]
        for n in graph[v]:
            if n not in prev:
                prev[n] = v
                queue.append(n)
    return []
```

---

## 12. Complexity Summary

| Algorithm | Time Complexity | Space Complexity | Notes |
|:---|:---|:---|:---|
| BFS | O(V + E) | O(V) | Queue + visited |
| DFS (recursive) | O(V + E) | O(V) | Call stack |
| DFS (iterative) | O(V + E) | O(V) | Explicit stack |
| Topological sort (DFS) | O(V + E) | O(V) | DAG only |
| Topological sort (Kahn) | O(V + E) | O(V) | DAG only |
| SCC (Kosaraju) | O(V + E) | O(V) | 2 DFS passes |
| SCC (Tarjan) | O(V + E) | O(V) | 1 DFS pass |
| Articulation points/bridges | O(V + E) | O(V) | Uses lowlink |
| Eulerian path (Hierholzer) | O(V + E) | O(E) | Edge management |
| Bipartite check | O(V + E) | O(V) | BFS/DFS |


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

### Q1: Are the complexities of BFS and DFS the same?

**A:** Yes, both are O(V + E). They process every vertex and every edge once. The difference lies in visit order and how memory is used. BFS consumes memory proportional to the "width," while DFS consumes memory proportional to the "depth." For trees, BFS uses O(V) memory for maximum width, while DFS uses O(log V) to O(V) memory for maximum depth.

### Q2: Is the result of topological sort unique?

**A:** Generally no. Multiple valid orderings can exist. The result is unique only when there is always exactly one vertex with in-degree 0 at each level (i.e., when a Hamiltonian path exists). To find the lexicographically smallest topological order, use Kahn's algorithm with a `heapq` (min-heap) instead of a deque.

### Q3: How do you find strongly connected components in a directed graph?

**A:** Use Tarjan's algorithm or Kosaraju's algorithm. Both run in O(V + E). Tarjan requires 1 DFS pass, while Kosaraju requires 2 (original graph + transpose graph). Kosaraju is simpler to implement; Tarjan is more efficient.

### Q4: What if the graph is defined implicitly?

**A:** For puzzle state spaces or maze grids, do not pre-build an adjacency list. Instead, generate neighbor vertices on-demand during BFS/DFS traversal. This is memory-efficient. For example, when solving a Rubik's Cube, the next states are generated by applying all possible moves from the current state.

### Q5: What is bidirectional BFS?

**A:** A technique that runs BFS simultaneously from both the source and the target, terminating when the two search frontiers meet. While standard BFS explores O(b^d) vertices (b = branching factor, d = distance), bidirectional BFS explores O(b^(d/2)) x 2 = O(2 * b^(d/2)), providing an exponential reduction in search space.

```python
def bidirectional_bfs(graph: dict, start, end) -> int:
    """Bidirectional BFS - searches from both source and target"""
    if start == end:
        return 0

    front = {start}
    back = {end}
    visited_front = {start: 0}
    visited_back = {end: 0}
    depth = 0

    while front and back:
        depth += 1
        # Expand the smaller frontier (optimization)
        if len(front) > len(back):
            front, back = back, front
            visited_front, visited_back = visited_back, visited_front

        next_front = set()
        for v in front:
            for neighbor in graph.get(v, []):
                if neighbor in visited_back:
                    return visited_front[v] + 1 + visited_back[neighbor]
                if neighbor not in visited_front:
                    visited_front[neighbor] = depth
                    next_front.add(neighbor)

        front = next_front

    return -1  # Unreachable
```

### Q6: What edge cases should be considered during graph traversal?

**A:** Key edge cases: (1) Empty graph (no vertices or edges), (2) Isolated vertices (vertices with no edges), (3) Self-loops (edges u -> u), (4) Multi-edges (multiple edges between the same vertices), (5) Disconnected graphs (starting from all vertices is necessary to visit all vertices), (6) Single-vertex graphs. A robust implementation must account for all of these.

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
| BFS | Uses queue, level-order traversal, guarantees shortest path (unweighted) |
| DFS | Uses stack/recursion, depth-first, suited for backtracking |
| Topological sort | Ordering of DAGs. DFS (Tarjan) or BFS (Kahn) |
| Strongly connected components | Tarjan (1 DFS pass) or Kosaraju (2 DFS passes) |
| Graph representation | Sparse -> adjacency list, Dense -> adjacency matrix. Adjacency list for most cases |
| Complexity | Both BFS and DFS are O(V + E) |
| Applications | Shortest path, connected components, cycle detection, bipartite check, dependency resolution |
| Practical uses | Social network analysis, web crawlers, build systems, deadlock detection |

---

## Recommended Next Guides

- [Shortest Path Algorithms](./03-shortest-path.md) -- Shortest paths in weighted graphs (Dijkstra, etc.)
- [Backtracking](./07-backtracking.md) -- Exhaustive search using DFS
- [Union-Find](../03-advanced/00-union-find.md) -- Efficient data structure for managing connected components

---

## References

1. Cormen, T. H. et al. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. -- Chapters 20-22
2. Sedgewick, R. & Wayne, K. (2011). *Algorithms* (4th ed.). Addison-Wesley. -- Part 5: Graphs
3. Kahn, A. B. (1962). "Topological sorting of large networks." *Communications of the ACM*.
4. Tarjan, R. E. (1972). "Depth-first search and linear graph algorithms." *SIAM Journal on Computing*.
5. Kosaraju, S. R. (1978). Unpublished manuscript. -- Two-pass algorithm for strongly connected components
6. Hierholzer, C. (1873). "Uber die Moglichkeit, einen Linienzug ohne Wiederholung und ohne Unterbrechung zu umfahren." *Mathematische Annalen*.
