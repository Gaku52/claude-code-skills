# Graphs (as a Data Structure)

> A graph is a collection of nodes and edges; trees, linked lists, and even arrays are all special cases of graphs.

## Learning Objectives

- [ ] Implement graph representations (adjacency list, adjacency matrix)
- [ ] Understand the differences between weighted/directed/undirected graphs
- [ ] Know graph representation patterns used in practice
- [ ] Master BFS/DFS implementations and applications
- [ ] Understand Union-Find (disjoint set data structure)
- [ ] Implement shortest path and minimum spanning tree algorithms
- [ ] Grasp the mechanism and applications of topological sort

## Prerequisites


---

## 1. Graph Fundamentals

### 1.1 Types of Graphs

```
Graph classification:

  1. By direction:
     Undirected graph: Edges have no direction
       A --- B    Friend relationships, road networks
       |     |
       C --- D

     Directed graph (digraph): Edges have a direction
       A --> B    Follow relationships, dependencies
       ^     |
       C <-- D

  2. By weight:
     Unweighted: Edges have no weight (all equal)
     Weighted: Edges carry a numerical value (distance, cost, etc.)
       A -(5)- B
       |       |
      (3)     (2)
       |       |
       C -(7)- D

  3. Other classifications:
     Simple graph: No self-loops or multi-edges
     Multigraph: Multiple edges between two vertices
     Complete graph: An edge between every pair of vertices (Kn)
     Bipartite graph: Vertices split into 2 groups with no edges within the same group
     DAG: Directed Acyclic Graph
     Connected graph: A path exists between all pairs of vertices

  Basic graph quantities:
     V: Number of vertices (nodes)
     E: Number of edges (arcs)
     Degree: Number of edges connected to a node
     In-degree: Number of incoming edges in a directed graph
     Out-degree: Number of outgoing edges in a directed graph

  Important theorems:
     Undirected graph: Sum of degree(v) = 2|E| (sum of all degrees = twice the edge count)
     Directed graph: Sum of in-degree(v) = Sum of out-degree(v) = |E|
     Complete graph K_n edge count: n(n-1)/2
     Tree edge count: V - 1
```

### 1.2 Graph Terminology

```
Important graph terms:

  Path: A sequence of vertices v1, v2, ..., vk (an edge exists between each adjacent pair)
  Simple path: A path that does not visit the same vertex twice
  Cycle: A path where the start and end vertices are the same
  DAG: A directed graph with no cycles

  Connected: In an undirected graph, a path exists between any two vertices
  Strongly connected: In a directed graph, a bidirectional path exists between any two vertices
  Connected component: A maximal connected subgraph

  Articulation point (cut vertex): A vertex whose removal disconnects the graph
  Bridge: An edge whose removal disconnects the graph

  Clique: A subgraph that is a complete graph
  Independent set: A set of vertices with no edges between them

  Sparse: A graph where E << V^2
  Dense: A graph where E ~ V^2
```

---

## 2. Graph Representations

### 2.1 Adjacency List

```python
# Undirected graph
from collections import defaultdict

class Graph:
    def __init__(self):
        self.adj = defaultdict(list)

    def add_edge(self, u, v, weight=1):
        self.adj[u].append((v, weight))
        self.adj[v].append((u, weight))  # Undirected graph

    def neighbors(self, u):
        return self.adj[u]

# Space: O(V + E)
# Edge addition: O(1)
# Edge existence check: O(degree)
# Enumerate all edges: O(V + E)
# -> Optimal for sparse graphs (few edges)
```

### 2.2 Directed Graph Adjacency List

```python
class DirectedGraph:
    """Adjacency list representation of a directed graph"""

    def __init__(self):
        self.adj = defaultdict(list)        # Outgoing edges
        self.reverse_adj = defaultdict(list) # Incoming edges (reverse graph)
        self.vertices = set()

    def add_edge(self, u, v, weight=1):
        self.adj[u].append((v, weight))
        self.reverse_adj[v].append((u, weight))
        self.vertices.add(u)
        self.vertices.add(v)

    def out_neighbors(self, u):
        """Adjacent nodes via outgoing edges"""
        return self.adj[u]

    def in_neighbors(self, v):
        """Adjacent nodes via incoming edges"""
        return self.reverse_adj[v]

    def out_degree(self, u):
        return len(self.adj[u])

    def in_degree(self, v):
        return len(self.reverse_adj[v])

    def has_edge(self, u, v):
        return any(node == v for node, _ in self.adj[u])

    def remove_edge(self, u, v):
        self.adj[u] = [(node, w) for node, w in self.adj[u] if node != v]
        self.reverse_adj[v] = [(node, w) for node, w in self.reverse_adj[v] if node != u]

    def get_all_edges(self):
        """Return a list of all edges"""
        edges = []
        for u in self.adj:
            for v, w in self.adj[u]:
                edges.append((u, v, w))
        return edges

# Usage example
g = DirectedGraph()
g.add_edge("A", "B", 5)
g.add_edge("A", "C", 3)
g.add_edge("B", "D", 2)
g.add_edge("C", "D", 7)
g.add_edge("D", "A", 1)

print(g.out_neighbors("A"))  # [('B', 5), ('C', 3)]
print(g.in_neighbors("D"))   # [('B', 2), ('C', 7)]
print(g.out_degree("A"))     # 2
print(g.in_degree("D"))      # 2
```

### 2.3 Adjacency Matrix

```python
class GraphMatrix:
    def __init__(self, n):
        self.n = n
        self.matrix = [[0] * n for _ in range(n)]

    def add_edge(self, u, v, weight=1):
        self.matrix[u][v] = weight
        self.matrix[v][u] = weight  # Undirected graph

    def has_edge(self, u, v):
        return self.matrix[u][v] != 0

# Space: O(V^2)
# Edge addition: O(1)
# Edge existence check: O(1)
# -> Optimal for dense graphs (many edges)
# -> Suitable for Floyd-Warshall
```

### 2.4 Edge List

```python
class EdgeListGraph:
    """Edge list representation: list of (u, v, weight)"""

    def __init__(self):
        self.edges = []
        self.vertices = set()

    def add_edge(self, u, v, weight=1):
        self.edges.append((u, v, weight))
        self.vertices.add(u)
        self.vertices.add(v)

    def sort_by_weight(self):
        """Sort by weight (used in Kruskal's algorithm)"""
        self.edges.sort(key=lambda e: e[2])

# Space: O(E)
# Edge addition: O(1)
# Edge existence check: O(E)
# -> Optimal for Kruskal's algorithm, Bellman-Ford algorithm
```

### 2.5 Choosing Between Representations

```
Selection criteria:

  +------------------+--------------+--------------+
  | Condition        | Adjacency    | Adjacency    |
  |                  | List         | Matrix       |
  +------------------+--------------+--------------+
  | Sparse graph     | OK (saves    | NG (wasteful)|
  |                  | memory)      |              |
  | Dense graph      | Fair         | OK (efficient|
  | Edge existence   | O(degree)    | O(1) OK      |
  | All neighbors    | O(degree) OK | O(V)         |
  | Memory           | O(V+E) OK    | O(V^2)       |
  | BFS/DFS          | O(V+E) OK    | O(V^2)       |
  | Floyd-Warshall   | O(V^3)       | O(V^3) OK    |
  | Edge addition    | O(1)         | O(1)         |
  | Edge removal     | O(E)         | O(1) OK      |
  +------------------+--------------+--------------+

  Practical guidelines:
  - Most cases: Adjacency list (graphs are usually sparse)
  - V < 1000 and dense graph: Consider adjacency matrix
  - Social networks (hundreds of millions of nodes): Adjacency list only
  - All-pairs shortest paths: Adjacency matrix + Floyd-Warshall
```

---

## 3. Graph Traversal

### 3.1 BFS (Breadth-First Search)

```python
from collections import deque

def bfs(graph, start):
    """Breadth-first search: O(V + E)"""
    visited = set()
    queue = deque([start])
    visited.add(start)
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)

        for neighbor, _ in graph.adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return order


# BFS shortest path (unweighted)
def bfs_shortest_path(graph, start, end):
    """Shortest path in an unweighted graph: O(V + E)"""
    if start == end:
        return [start]

    visited = {start}
    queue = deque([(start, [start])])

    while queue:
        node, path = queue.popleft()
        for neighbor, _ in graph.adj[node]:
            if neighbor == end:
                return path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None  # No path exists


# BFS all shortest distances
def bfs_distances(graph, start):
    """Shortest distances from start to all nodes: O(V + E)"""
    distances = {start: 0}
    queue = deque([start])

    while queue:
        node = queue.popleft()
        for neighbor, _ in graph.adj[node]:
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)

    return distances


# BFS by levels
def bfs_levels(graph, start):
    """Group nodes by level"""
    visited = {start}
    queue = deque([start])
    levels = []

    while queue:
        level_size = len(queue)
        level = []
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node)
            for neighbor, _ in graph.adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        levels.append(level)

    return levels
```

### 3.2 DFS (Depth-First Search)

```python
# Recursive DFS
def dfs_recursive(graph, start, visited=None):
    """Depth-first search (recursive): O(V + E)"""
    if visited is None:
        visited = set()

    visited.add(start)
    order = [start]

    for neighbor, _ in graph.adj[start]:
        if neighbor not in visited:
            order.extend(dfs_recursive(graph, neighbor, visited))

    return order


# Iterative DFS (using a stack)
def dfs_iterative(graph, start):
    """Depth-first search (iterative): O(V + E)"""
    visited = set()
    stack = [start]
    order = []

    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)

        # Add neighbors in reverse order to the stack
        for neighbor, _ in reversed(graph.adj[node]):
            if neighbor not in visited:
                stack.append(neighbor)

    return order


# DFS path search
def dfs_all_paths(graph, start, end):
    """Find all paths from start to end"""
    result = []

    def backtrack(node, path, visited):
        if node == end:
            result.append(path[:])
            return

        for neighbor, _ in graph.adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                backtrack(neighbor, path, visited)
                path.pop()
                visited.remove(neighbor)

    backtrack(start, [start], {start})
    return result


# DFS connected component detection
def find_connected_components(graph, all_vertices):
    """Detect all connected components: O(V + E)"""
    visited = set()
    components = []

    for v in all_vertices:
        if v not in visited:
            component = []
            stack = [v]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                for neighbor, _ in graph.adj[node]:
                    if neighbor not in visited:
                        stack.append(neighbor)
            components.append(component)

    return components
```

### 3.3 Cycle Detection

```python
# Cycle detection in directed graphs (DFS + 3-color method)
def has_cycle_directed(graph, all_vertices):
    """Determine if a directed graph has a cycle: O(V + E)"""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {v: WHITE for v in all_vertices}

    def dfs(node):
        color[node] = GRAY  # Currently being explored

        for neighbor, _ in graph.adj[node]:
            if color[neighbor] == GRAY:
                return True  # Back edge -> Cycle
            if color[neighbor] == WHITE and dfs(neighbor):
                return True

        color[node] = BLACK  # Exploration complete
        return False

    for v in all_vertices:
        if color[v] == WHITE:
            if dfs(v):
                return True

    return False


# Cycle detection in undirected graphs
def has_cycle_undirected(graph, all_vertices):
    """Determine if an undirected graph has a cycle: O(V + E)"""
    visited = set()

    def dfs(node, parent):
        visited.add(node)
        for neighbor, _ in graph.adj[node]:
            if neighbor not in visited:
                if dfs(neighbor, node):
                    return True
            elif neighbor != parent:
                return True  # Visited node other than parent -> Cycle
        return False

    for v in all_vertices:
        if v not in visited:
            if dfs(v, None):
                return True

    return False


# Cycle detection and reconstruction
def find_cycle(graph, all_vertices):
    """Find and return a cycle (directed graph)"""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {v: WHITE for v in all_vertices}
    parent = {}
    cycle_start = None
    cycle_end = None

    def dfs(node):
        nonlocal cycle_start, cycle_end
        color[node] = GRAY

        for neighbor, _ in graph.adj[node]:
            if color[neighbor] == GRAY:
                cycle_start = neighbor
                cycle_end = node
                return True
            if color[neighbor] == WHITE:
                parent[neighbor] = node
                if dfs(neighbor):
                    return True

        color[node] = BLACK
        return False

    for v in all_vertices:
        if color[v] == WHITE:
            if dfs(v):
                # Reconstruct the cycle
                cycle = [cycle_start]
                current = cycle_end
                while current != cycle_start:
                    cycle.append(current)
                    current = parent[current]
                cycle.reverse()
                return cycle

    return None  # No cycle
```

---

## 4. Topological Sort

### 4.1 DFS-Based

```python
def topological_sort_dfs(graph, all_vertices):
    """DFS-based topological sort: O(V + E)"""
    visited = set()
    order = []

    def dfs(node):
        visited.add(node)
        for neighbor, _ in graph.adj[node]:
            if neighbor not in visited:
                dfs(neighbor)
        order.append(node)  # Add during post-processing

    for v in all_vertices:
        if v not in visited:
            dfs(v)

    order.reverse()
    return order
```

### 4.2 Kahn's Algorithm (BFS-Based)

```python
from collections import deque

def topological_sort_kahn(graph, all_vertices):
    """Kahn's algorithm: O(V + E)
    Can also detect cycles simultaneously"""

    # Calculate in-degrees
    in_degree = {v: 0 for v in all_vertices}
    for u in all_vertices:
        for v, _ in graph.adj[u]:
            in_degree[v] = in_degree.get(v, 0) + 1

    # Add nodes with in-degree 0 to the queue
    queue = deque([v for v in all_vertices if in_degree[v] == 0])
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)

        for neighbor, _ in graph.adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(order) != len(all_vertices):
        raise ValueError("The graph contains a cycle")

    return order


# Usage example: Task dependencies
g = DirectedGraph()
g.add_edge("compile", "link")
g.add_edge("compile", "test")
g.add_edge("link", "deploy")
g.add_edge("test", "deploy")
g.add_edge("init", "compile")

order = topological_sort_kahn(g, g.vertices)
print(order)  # ['init', 'compile', 'link', 'test', 'deploy'] etc.
```

### 4.3 Applications of Topological Sort

```python
# 1. Course enrollment order
def find_course_order(num_courses, prerequisites):
    """
    prerequisites: [(course, prerequisite), ...]
    Returns: List representing enrollment order (empty list if impossible)
    """
    graph = defaultdict(list)
    in_degree = [0] * num_courses

    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1

    queue = deque([i for i in range(num_courses) if in_degree[i] == 0])
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return order if len(order) == num_courses else []

# Usage example
print(find_course_order(4, [(1,0), (2,0), (3,1), (3,2)]))
# [0, 1, 2, 3] or [0, 2, 1, 3]


# 2. Build system dependency resolution
def resolve_dependencies(packages):
    """
    packages: {name: [dependencies]}
    Returns: Installation order
    """
    graph = defaultdict(list)
    all_pkgs = set()

    for pkg, deps in packages.items():
        all_pkgs.add(pkg)
        for dep in deps:
            graph[dep].append(pkg)
            all_pkgs.add(dep)

    in_degree = {pkg: 0 for pkg in all_pkgs}
    for pkg, deps in packages.items():
        in_degree[pkg] = len(deps)

    queue = deque([pkg for pkg in all_pkgs if in_degree[pkg] == 0])
    order = []

    while queue:
        pkg = queue.popleft()
        order.append(pkg)
        for dependent in graph[pkg]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(order) != len(all_pkgs):
        raise ValueError("Circular dependency detected")

    return order

# Usage example
packages = {
    "app": ["framework", "database"],
    "framework": ["utils", "logging"],
    "database": ["utils"],
    "utils": [],
    "logging": [],
}
print(resolve_dependencies(packages))
# ['utils', 'logging', 'framework', 'database', 'app']
```

---

## 5. Union-Find (Disjoint Set Data Structure)

### 5.1 Basic Implementation

```python
class UnionFind:
    """Union-Find (Disjoint Set Data Structure)
    With path compression + union by rank: O(alpha(n)) ~ O(1)"""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = n  # Number of connected components

    def find(self, x):
        """Find root (with path compression)"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        """Merge two sets (union by rank)"""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False  # Already in the same set

        # Attach the lower-rank tree under the higher-rank tree
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        self.count -= 1
        return True

    def connected(self, x, y):
        """Check if they belong to the same set"""
        return self.find(x) == self.find(y)

    def get_count(self):
        """Return the number of connected components"""
        return self.count


# Usage example
uf = UnionFind(10)
uf.union(0, 1)
uf.union(2, 3)
uf.union(1, 3)
print(uf.connected(0, 3))  # True (0-1-3-2 are in the same set)
print(uf.connected(0, 5))  # False
print(uf.get_count())       # 7 ({0,1,2,3}, {4}, {5}, ..., {9})
```

### 5.2 Union-Find Applications

```python
# 1. Detecting redundant edges
def find_redundant_connection(edges):
    """Find the edge that creates a cycle in an undirected graph"""
    n = len(edges)
    uf = UnionFind(n + 1)

    for u, v in edges:
        if not uf.union(u, v):
            return [u, v]  # This edge creates a cycle

    return []


# 2. Counting islands (2D grid)
def num_islands(grid):
    """Count the number of connected components of '1's"""
    if not grid:
        return 0

    rows, cols = len(grid), len(grid[0])
    uf = UnionFind(rows * cols)
    water_count = 0

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '0':
                water_count += 1
                continue
            # Merge with right and bottom adjacent cells
            for dr, dc in [(0, 1), (1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == '1':
                    uf.union(r * cols + c, nr * cols + nc)

    return uf.get_count() - water_count


# 3. Account merging
def accounts_merge(accounts):
    """Merge accounts that share the same email address"""
    email_to_id = {}
    email_to_name = {}
    uf = UnionFind(len(accounts))

    for i, account in enumerate(accounts):
        name = account[0]
        for email in account[1:]:
            email_to_name[email] = name
            if email in email_to_id:
                uf.union(i, email_to_id[email])
            email_to_id[email] = i

    # Group emails by merged accounts
    groups = defaultdict(set)
    for email, account_id in email_to_id.items():
        root = uf.find(account_id)
        groups[root].add(email)

    result = []
    for root, emails in groups.items():
        name = email_to_name[next(iter(emails))]
        result.append([name] + sorted(emails))

    return result


# 4. Minimum spanning tree (Kruskal's algorithm)
def kruskal_mst(n, edges):
    """Find the minimum spanning tree using Kruskal's algorithm: O(E log E)"""
    # Sort edges by weight
    edges.sort(key=lambda e: e[2])

    uf = UnionFind(n)
    mst = []
    total_weight = 0

    for u, v, weight in edges:
        if not uf.connected(u, v):
            uf.union(u, v)
            mst.append((u, v, weight))
            total_weight += weight

            if len(mst) == n - 1:
                break  # All vertices connected

    return mst, total_weight

# Usage example
edges = [
    (0, 1, 4), (0, 7, 8),
    (1, 2, 8), (1, 7, 11),
    (2, 3, 7), (2, 5, 4),
    (2, 8, 2), (3, 4, 9),
    (3, 5, 14), (4, 5, 10),
    (5, 6, 2), (6, 7, 1),
    (6, 8, 6), (7, 8, 7),
]
mst, weight = kruskal_mst(9, edges)
print(f"MST weight: {weight}")  # 37
print(f"MST edges: {mst}")
```

---

## 6. Shortest Path Algorithms

### 6.1 Dijkstra's Algorithm

```python
import heapq

def dijkstra(graph, start):
    """Dijkstra's algorithm: O((V + E) log V)
    Single-source shortest paths for graphs with non-negative weights"""

    distances = {v: float('inf') for v in graph.adj}
    distances[start] = 0
    predecessors = {v: None for v in graph.adj}
    pq = [(0, start)]  # (distance, node)

    while pq:
        dist, node = heapq.heappop(pq)

        if dist > distances[node]:
            continue  # Skip outdated entries

        for neighbor, weight in graph.adj[node]:
            new_dist = dist + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                predecessors[neighbor] = node
                heapq.heappush(pq, (new_dist, neighbor))

    return distances, predecessors


def reconstruct_path(predecessors, start, end):
    """Reconstruct the shortest path"""
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = predecessors[current]
    path.reverse()
    return path if path[0] == start else []


# Usage example
g = Graph()
g.add_edge("A", "B", 4)
g.add_edge("A", "C", 2)
g.add_edge("B", "D", 3)
g.add_edge("C", "D", 1)
g.add_edge("C", "B", 1)
g.add_edge("D", "E", 5)

distances, preds = dijkstra(g, "A")
print(distances)  # {'A': 0, 'B': 3, 'C': 2, 'D': 3, 'E': 8}
print(reconstruct_path(preds, "A", "E"))  # ['A', 'C', 'D', 'E']
```

### 6.2 Bellman-Ford Algorithm

```python
def bellman_ford(vertices, edges, start):
    """Bellman-Ford algorithm: O(V * E)
    Handles negative edges. Can detect negative cycles"""

    distances = {v: float('inf') for v in vertices}
    distances[start] = 0
    predecessors = {v: None for v in vertices}

    # V-1 relaxation passes
    for i in range(len(vertices) - 1):
        updated = False
        for u, v, w in edges:
            if distances[u] + w < distances[v]:
                distances[v] = distances[u] + w
                predecessors[v] = u
                updated = True
        if not updated:
            break  # Early termination

    # Negative cycle detection
    for u, v, w in edges:
        if distances[u] + w < distances[v]:
            raise ValueError("Negative cycle detected")

    return distances, predecessors

# Usage example
vertices = ["A", "B", "C", "D"]
edges = [
    ("A", "B", 4),
    ("A", "C", 2),
    ("B", "D", 3),
    ("C", "B", -1),  # Negative edge
    ("C", "D", 5),
]
distances, _ = bellman_ford(vertices, edges, "A")
print(distances)  # {'A': 0, 'B': 1, 'C': 2, 'D': 4}
```

### 6.3 Floyd-Warshall Algorithm

```python
def floyd_warshall(n, edges):
    """Floyd-Warshall algorithm: O(V^3)
    All-pairs shortest paths"""

    INF = float('inf')
    dist = [[INF] * n for _ in range(n)]
    next_node = [[None] * n for _ in range(n)]

    # Initialization
    for i in range(n):
        dist[i][i] = 0

    for u, v, w in edges:
        dist[u][v] = w
        next_node[u][v] = v

    # Dynamic programming
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_node[i][j] = next_node[i][k]

    # Negative cycle detection
    for i in range(n):
        if dist[i][i] < 0:
            raise ValueError("Negative cycle detected")

    return dist, next_node


def reconstruct_fw_path(next_node, u, v):
    """Path reconstruction for Floyd-Warshall"""
    if next_node[u][v] is None:
        return []
    path = [u]
    while u != v:
        u = next_node[u][v]
        path.append(u)
    return path
```

### 6.4 A* Algorithm

```python
import heapq

def a_star(graph, start, goal, heuristic):
    """A* algorithm: O((V + E) log V)
    Optimizes search using a heuristic function

    heuristic(node): Estimated cost from node to goal
    Guarantees optimal solution if the heuristic is admissible (never overestimates)
    """

    open_set = [(0 + heuristic(start), 0, start)]  # (f, g, node)
    g_scores = {start: 0}
    came_from = {}

    while open_set:
        f, g, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1], g

        for neighbor, weight in graph.adj[current]:
            tentative_g = g + weight

            if tentative_g < g_scores.get(neighbor, float('inf')):
                g_scores[neighbor] = tentative_g
                came_from[neighbor] = current
                f = tentative_g + heuristic(neighbor)
                heapq.heappush(open_set, (f, tentative_g, neighbor))

    return None, float('inf')  # No path found


# Usage example: Pathfinding on a 2D grid
def grid_a_star(grid, start, goal):
    """A* pathfinding on a 2D grid"""

    def heuristic(pos):
        """Manhattan distance"""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    rows, cols = len(grid), len(grid[0])
    open_set = [(heuristic(start), 0, start)]
    g_scores = {start: 0}
    came_from = {}

    while open_set:
        _, g, current = heapq.heappop(open_set)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = current[0] + dr, current[1] + dc
            neighbor = (nr, nc)

            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != 1:
                tentative_g = g + 1
                if tentative_g < g_scores.get(neighbor, float('inf')):
                    g_scores[neighbor] = tentative_g
                    came_from[neighbor] = current
                    f = tentative_g + heuristic(neighbor)
                    heapq.heappush(open_set, (f, tentative_g, neighbor))

    return None  # No path

# Usage example
grid = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0],
]
path = grid_a_star(grid, (0, 0), (4, 4))
print(path)  # [(0,0), (0,1), (0,2), (0,3), (1,3), (2,3), (2,4), (3,4), (4,4)]
```

---

## 7. Advanced Graph Algorithms

### 7.1 Bipartite Graph Detection

```python
def is_bipartite(graph, all_vertices):
    """Determine if a graph is bipartite using BFS: O(V + E)"""
    color = {}

    for start in all_vertices:
        if start in color:
            continue

        queue = deque([start])
        color[start] = 0

        while queue:
            node = queue.popleft()
            for neighbor, _ in graph.adj[node]:
                if neighbor not in color:
                    color[neighbor] = 1 - color[node]
                    queue.append(neighbor)
                elif color[neighbor] == color[node]:
                    return False  # Adjacent nodes with same color -> Not bipartite

    return True

# Bipartite graph applications:
# - Matching problems (optimal matching of job seekers and positions)
# - 2-coloring problems
# - Detecting graphs with only even-length cycles
```

### 7.2 Strongly Connected Components (Kosaraju's Algorithm)

```python
def kosaraju_scc(graph, all_vertices):
    """Find strongly connected components using Kosaraju's algorithm: O(V + E)"""

    # Pass 1: DFS on original graph, record finish order
    visited = set()
    finish_order = []

    def dfs1(node):
        visited.add(node)
        for neighbor, _ in graph.adj[node]:
            if neighbor not in visited:
                dfs1(neighbor)
        finish_order.append(node)

    for v in all_vertices:
        if v not in visited:
            dfs1(v)

    # Build reverse graph
    reverse_graph = defaultdict(list)
    for u in graph.adj:
        for v, w in graph.adj[u]:
            reverse_graph[v].append((u, w))

    # Pass 2: DFS on reverse graph in reverse finish order
    visited = set()
    sccs = []

    def dfs2(node, component):
        visited.add(node)
        component.append(node)
        for neighbor, _ in reverse_graph[node]:
            if neighbor not in visited:
                dfs2(neighbor, component)

    for v in reversed(finish_order):
        if v not in visited:
            component = []
            dfs2(v, component)
            sccs.append(component)

    return sccs

# Usage example: Analyzing link structures of web pages
# Strongly connected components = groups of pages mutually reachable via links
```

### 7.3 Minimum Spanning Tree (Prim's Algorithm)

```python
def prim_mst(graph, start):
    """Find the minimum spanning tree using Prim's algorithm: O((V + E) log V)"""
    visited = set()
    mst = []
    total_weight = 0

    # (weight, current node, parent node)
    pq = [(0, start, None)]

    while pq:
        weight, node, parent = heapq.heappop(pq)

        if node in visited:
            continue

        visited.add(node)
        if parent is not None:
            mst.append((parent, node, weight))
            total_weight += weight

        for neighbor, w in graph.adj[node]:
            if neighbor not in visited:
                heapq.heappush(pq, (w, neighbor, node))

    return mst, total_weight
```

---

## 8. Graphs in Practice

### 8.1 Real-World Graph Problems

```
Graph representations and applications in practice:

  1. RDB: Relations between tables -> Implicit graph
     users, follows tables -> Social graph

  2. Neo4j: Graph database
     Cypher: MATCH (a)-[:FOLLOWS]->(b) WHERE a.name = 'Alice'

  3. GraphQL: Graph structure of APIs

  4. npm/pip: Package dependency graph (DAG)

  5. Kubernetes: Inter-service communication (service mesh)

  6. Google Maps: Road network -> Dijkstra/A*

  7. Social networks: Follow/friend relationships -> Social graph

  8. Recommendation systems: User x item bipartite graph

  9. Compilers: Control flow graph (CFG)

  10. CI/CD: Pipeline task dependencies (DAG)
      GitHub Actions, Airflow, Terraform
```

### 8.2 Graph Databases and Queries

```python
# Graph analysis with NetworkX (Python's standard graph library)
import networkx as nx

# Create a graph
G = nx.DiGraph()  # Directed graph
G.add_weighted_edges_from([
    ("Alice", "Bob", 1),
    ("Alice", "Charlie", 1),
    ("Bob", "David", 1),
    ("Charlie", "David", 1),
    ("David", "Eve", 1),
])

# Basic analysis
print(f"Node count: {G.number_of_nodes()}")
print(f"Edge count: {G.number_of_edges()}")
print(f"Degrees: {dict(G.degree())}")

# Shortest path
print(nx.shortest_path(G, "Alice", "Eve"))
# ['Alice', 'Bob', 'David', 'Eve']

# PageRank
pr = nx.pagerank(G)
print(f"PageRank: {pr}")

# Centrality analysis
betweenness = nx.betweenness_centrality(G)
print(f"Betweenness centrality: {betweenness}")

# Connected components
components = list(nx.weakly_connected_components(G))
print(f"Weakly connected components: {len(components)}")


# Neo4j Cypher query examples
"""
// Search followers of followers (2 hops)
MATCH (a:User {name: 'Alice'})-[:FOLLOWS]->(b)-[:FOLLOWS]->(c)
WHERE a <> c
RETURN c.name AS recommended_friend

// Shortest path
MATCH path = shortestPath(
  (a:User {name: 'Alice'})-[:FOLLOWS*..10]-(b:User {name: 'Eve'})
)
RETURN path

// Community detection
CALL gds.louvain.stream('social-graph')
YIELD nodeId, communityId
RETURN gds.util.asNode(nodeId).name AS name, communityId
"""
```

### 8.3 System Design with Graphs

```python
# 1. Task scheduler (DAG-based)
class TaskScheduler:
    """Task scheduler with dependency awareness"""

    def __init__(self):
        self.tasks = {}      # task_name -> callable
        self.deps = defaultdict(list)  # task -> [dependencies]

    def add_task(self, name, func, dependencies=None):
        self.tasks[name] = func
        if dependencies:
            for dep in dependencies:
                self.deps[name].append(dep)

    def execute(self):
        """Execute tasks in topological order"""
        # Calculate in-degrees
        in_degree = {task: 0 for task in self.tasks}
        graph = defaultdict(list)
        for task, deps in self.deps.items():
            for dep in deps:
                graph[dep].append(task)
                in_degree[task] += 1

        # BFS topological sort
        queue = deque([t for t in self.tasks if in_degree[t] == 0])
        results = {}

        while queue:
            task = queue.popleft()
            print(f"Executing: {task}")
            results[task] = self.tasks[task]()
            for dependent in graph[task]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        return results


# 2. Social graph analysis
class SocialGraph:
    """Graph analysis for social networks"""

    def __init__(self):
        self.graph = defaultdict(set)  # user -> set of friends

    def add_friendship(self, user1, user2):
        self.graph[user1].add(user2)
        self.graph[user2].add(user1)

    def mutual_friends(self, user1, user2):
        """Get mutual friends"""
        return self.graph[user1] & self.graph[user2]

    def friend_recommendations(self, user, top_n=5):
        """Recommend friends-of-friends (ranked by mutual friend count)"""
        scores = defaultdict(int)
        friends = self.graph[user]

        for friend in friends:
            for fof in self.graph[friend]:
                if fof != user and fof not in friends:
                    scores[fof] += 1  # Mutual friend count as score

        return sorted(scores.items(), key=lambda x: -x[1])[:top_n]

    def degrees_of_separation(self, user1, user2):
        """Separation between two users (shortest distance)"""
        if user1 == user2:
            return 0

        visited = {user1}
        queue = deque([(user1, 0)])

        while queue:
            node, dist = queue.popleft()
            for friend in self.graph[node]:
                if friend == user2:
                    return dist + 1
                if friend not in visited:
                    visited.add(friend)
                    queue.append((friend, dist + 1))

        return -1  # Unreachable

    def clustering_coefficient(self, user):
        """Clustering coefficient: Degree of interconnection among friends"""
        friends = list(self.graph[user])
        if len(friends) < 2:
            return 0.0

        # Count edges among friends
        edges = 0
        for i in range(len(friends)):
            for j in range(i + 1, len(friends)):
                if friends[j] in self.graph[friends[i]]:
                    edges += 1

        # Maximum possible edges
        max_edges = len(friends) * (len(friends) - 1) / 2
        return edges / max_edges if max_edges > 0 else 0.0


# Usage example
sg = SocialGraph()
sg.add_friendship("Alice", "Bob")
sg.add_friendship("Alice", "Charlie")
sg.add_friendship("Bob", "Charlie")
sg.add_friendship("Bob", "David")
sg.add_friendship("Charlie", "Eve")
sg.add_friendship("David", "Eve")

print(sg.mutual_friends("Alice", "David"))    # {'Bob'}
print(sg.friend_recommendations("Alice"))     # [('David', 1), ('Eve', 1)]
print(sg.degrees_of_separation("Alice", "Eve"))  # 2
print(sg.clustering_coefficient("Alice"))      # 1.0 (all friends are connected)


# 3. Route planner
class RoutePlanner:
    """Route planning with weighted graphs"""

    def __init__(self):
        self.graph = defaultdict(list)
        self.coordinates = {}  # node -> (lat, lon)

    def add_road(self, city1, city2, distance):
        self.graph[city1].append((city2, distance))
        self.graph[city2].append((city1, distance))

    def set_coordinates(self, city, lat, lon):
        self.coordinates[city] = (lat, lon)

    def shortest_route(self, start, end):
        """Calculate the shortest route using Dijkstra's algorithm"""
        distances = {start: 0}
        predecessors = {start: None}
        pq = [(0, start)]

        while pq:
            dist, city = heapq.heappop(pq)
            if city == end:
                break
            if dist > distances.get(city, float('inf')):
                continue
            for neighbor, weight in self.graph[city]:
                new_dist = dist + weight
                if new_dist < distances.get(neighbor, float('inf')):
                    distances[neighbor] = new_dist
                    predecessors[neighbor] = city
                    heapq.heappush(pq, (new_dist, neighbor))

        # Path reconstruction
        if end not in predecessors:
            return None, float('inf')

        path = []
        current = end
        while current is not None:
            path.append(current)
            current = predecessors[current]
        path.reverse()
        return path, distances[end]

# Usage example
planner = RoutePlanner()
planner.add_road("Tokyo", "Yokohama", 30)
planner.add_road("Tokyo", "Chiba", 40)
planner.add_road("Yokohama", "Nagoya", 350)
planner.add_road("Chiba", "Nagoya", 380)
planner.add_road("Nagoya", "Osaka", 180)

path, dist = planner.shortest_route("Tokyo", "Osaka")
print(f"Route: {' -> '.join(path)}")   # Tokyo -> Yokohama -> Nagoya -> Osaka
print(f"Total distance: {dist}km")     # 560km
```

---

## 9. Graph Visualization

```python
# Graph drawing with matplotlib + networkx
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
import networkx as nx

def visualize_graph(edges, directed=False, weighted=True):
    """Save graph as a PNG image"""
    G = nx.DiGraph() if directed else nx.Graph()

    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(10, 8))
    nx.draw(G, pos,
            with_labels=True,
            node_color='lightblue',
            node_size=800,
            font_size=12,
            font_weight='bold',
            edge_color='gray',
            arrows=directed)

    if weighted:
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title("Graph Visualization")
    plt.tight_layout()
    plt.savefig("graph.png", dpi=150)
    plt.close()

# ASCII graph display
def print_graph_ascii(graph):
    """Display adjacency list in ASCII"""
    for node in sorted(graph.adj.keys()):
        neighbors = [(n, w) for n, w in graph.adj[node]]
        neighbor_str = ", ".join(f"{n}({w})" for n, w in sorted(neighbors))
        print(f"  {node} -> [{neighbor_str}]")
```

---

## 10. Practice Exercises

### Exercise 1: Graph Construction (Basic)
Implement a graph using both adjacency list and adjacency matrix representations, and execute BFS/DFS. Include the following operations:
- Add and remove vertices and edges
- Check edge existence
- Get all neighboring nodes
- BFS/DFS traversal

### Exercise 2: Bipartite Graph Detection (Intermediate)
Implement a function that determines whether a graph is bipartite using BFS. If it is bipartite, return the two groups.

### Exercise 3: Minimum Spanning Tree (Advanced)
Implement minimum spanning tree computation using both Kruskal's and Prim's algorithms, and verify that the results match.

### Exercise 4: Dijkstra's Algorithm (Intermediate)
Implement Dijkstra's algorithm on a weighted graph, returning the shortest path and distance. Also implement the ability to switch to the Bellman-Ford algorithm when negative edges are present.

### Exercise 5: Topological Sort (Intermediate)
Represent a university curriculum as a directed graph and implement the following:
- Determine the course enrollment order via topological sort
- Detect and report circular dependencies
- Group courses that can be taken concurrently

### Exercise 6: Social Graph Analysis (Advanced)
Build a social network graph and implement the following:
- Search for mutual friends
- Friend recommendations (score friends-of-friends)
- Compute degrees of separation
- Community detection (connected component analysis)

### Exercise 7: A* Algorithm (Advanced)
Find the shortest path on a 2D grid with obstacles using the A* algorithm:
- Compare Manhattan distance and Euclidean distance heuristics
- Performance comparison with BFS/Dijkstra
- Visualize the result on the grid


---

## Troubleshooting

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| Initialization error | Configuration file issues | Check configuration file path and format |
| Timeout | Network latency / insufficient resources | Adjust timeout values, add retry logic |
| Out of memory | Data volume growth | Introduce batch processing, implement pagination |
| Permission error | Insufficient access rights | Verify user permissions, review settings |
| Data inconsistency | Concurrency conflicts | Introduce locking mechanisms, transaction management |

### Debugging Steps

1. **Check error messages**: Read the stack trace to identify where the error occurred
2. **Establish reproduction steps**: Reproduce the error with minimal code
3. **Formulate hypotheses**: List possible causes
4. **Verify incrementally**: Use log output or a debugger to test hypotheses
5. **Fix and regression test**: After fixing, also run tests on related areas

```python
# Debugging utilities
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

### Diagnosing Performance Issues

Steps for diagnosing performance issues:

1. **Identify bottlenecks**: Measure with profiling tools
2. **Check memory usage**: Look for memory leaks
3. **Check I/O wait**: Examine disk and network I/O conditions
4. **Check concurrent connections**: Inspect connection pool status

| Problem Type | Diagnostic Tools | Countermeasures |
|-------------|-----------------|-----------------|
| CPU load | cProfile, py-spy | Algorithm improvement, parallelization |
| Memory leak | tracemalloc, objgraph | Proper reference release |
| I/O bottleneck | strace, iostat | Async I/O, caching |
| DB latency | EXPLAIN, slow query log | Indexing, query optimization |

---

## Design Decision Guide

### Selection Criteria Matrix

The following summarizes decision criteria for technical choices.

| Criterion | When to prioritize | When compromise is acceptable |
|-----------|-------------------|------------------------------|
| Performance | Real-time processing, large-scale data | Admin dashboards, batch processing |
| Maintainability | Long-term operation, team development | Prototypes, short-term projects |
| Scalability | Growing services | Internal tools, fixed users |
| Security | Personal data, financial data | Public data, internal use |
| Development speed | MVP, time-to-market | Quality-focused, mission-critical |

### Architecture Pattern Selection

```
+--------------------------------------------------+
|            Architecture Selection Flow            |
+--------------------------------------------------+
|                                                  |
|  1. Team size?                                   |
|    +-- Small (1-5) -> Monolith                   |
|    +-- Large (10+) -> Go to 2                    |
|                                                  |
|  2. Deployment frequency?                        |
|    +-- Weekly or less -> Monolith + modules      |
|    +-- Daily/multiple -> Go to 3                 |
|                                                  |
|  3. Team independence?                           |
|    +-- High -> Microservices                     |
|    +-- Moderate -> Modular monolith              |
|                                                  |
+--------------------------------------------------+
```

### Trade-off Analysis

Technical decisions always involve trade-offs. Analyze from the following perspectives:

**1. Short-term vs. Long-term Cost**
- A short-term fast approach can become technical debt in the long run
- Conversely, over-engineering carries high short-term costs and can delay the project

**2. Consistency vs. Flexibility**
- A unified technology stack has lower learning costs
- Adopting diverse technologies enables the right tool for each job but increases operational costs

**3. Level of Abstraction**
- High abstraction offers high reusability but can make debugging difficult
- Low abstraction is intuitive but tends to produce code duplication

```python
# Design decision record template
class ArchitectureDecisionRecord:
    """Create an ADR (Architecture Decision Record)"""

    def __init__(self, title: str):
        self.title = title
        self.context = ""
        self.decision = ""
        self.consequences = []
        self.alternatives = []

    def set_context(self, context: str):
        """Describe background and challenges"""
        self.context = context
        return self

    def set_decision(self, decision: str):
        """Describe the decision"""
        self.decision = decision
        return self

    def add_consequence(self, consequence: str, positive: bool = True):
        """Add a consequence"""
        self.consequences.append({
            'description': consequence,
            'type': 'positive' if positive else 'negative'
        })
        return self

    def add_alternative(self, name: str, reason_rejected: str):
        """Add a rejected alternative"""
        self.alternatives.append({
            'name': name,
            'reason_rejected': reason_rejected
        })
        return self

    def to_markdown(self) -> str:
        """Output in Markdown format"""
        md = f"# ADR: {self.title}\n\n"
        md += f"## Context\n{self.context}\n\n"
        md += f"## Decision\n{self.decision}\n\n"
        md += "## Consequences\n"
        for c in self.consequences:
            icon = "+" if c['type'] == 'positive' else "!"
            md += f"- {icon} {c['description']}\n"
        md += "\n## Rejected Alternatives\n"
        for a in self.alternatives:
            md += f"- **{a['name']}**: {a['reason_rejected']}\n"
        return md
```
---


## FAQ

### Q1: What is the most important point when studying this topic?

Gaining practical experience is the most important thing. Understanding deepens not just from theory, but by actually writing code and verifying its behavior.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts covered in this guide before moving on.

### Q3: How is this applied in practice?

Knowledge of this topic is frequently used in everyday development work. It is particularly important during code reviews and architecture design.

---

## Summary

| Representation | Space | Edge Check | Best For |
|---------------|-------|-----------|----------|
| Adjacency list | O(V+E) | O(degree) | Sparse graphs, general use |
| Adjacency matrix | O(V^2) | O(1) | Dense graphs, small scale |
| Edge list | O(E) | O(E) | Kruskal's algorithm |

| Algorithm | Complexity | Use Case |
|-----------|-----------|----------|
| BFS | O(V+E) | Shortest path (unweighted), level traversal |
| DFS | O(V+E) | Cycle detection, topological sort |
| Dijkstra | O((V+E)log V) | Shortest path (non-negative weights) |
| Bellman-Ford | O(VE) | Shortest path (negative edges allowed) |
| Floyd-Warshall | O(V^3) | All-pairs shortest paths |
| A* | O((V+E)log V) | Shortest path with heuristic |
| Kruskal | O(E log E) | Minimum spanning tree |
| Prim | O((V+E)log V) | Minimum spanning tree |
| Kahn | O(V+E) | Topological sort |
| Kosaraju | O(V+E) | Strongly connected components |

---

## Recommended Next Reading

---

## References
1. Cormen, T. H. "Introduction to Algorithms." Chapters 22-26.
2. Sedgewick, R. "Algorithms." Chapter 4.1-4.4.
3. Kleinberg, J., Tardos, E. "Algorithm Design." Chapters 3-7.
4. Skiena, S. S. "The Algorithm Design Manual." Chapter 5-7.
5. Hart, P. E., Nilsson, N. J., Raphael, B. "A Formal Basis for the Heuristic Determination of Minimum Cost Paths." 1968.
6. Dijkstra, E. W. "A Note on Two Problems in Connexion with Graphs." 1959.
7. Kruskal, J. B. "On the Shortest Spanning Subtree of a Graph." 1956.
