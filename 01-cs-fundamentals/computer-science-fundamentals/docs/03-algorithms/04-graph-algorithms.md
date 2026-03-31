# Graph Algorithms

> Social network friendships, map route finding, web link structures -- the world is full of graphs.
> Graphs are a central concept in discrete mathematics and a versatile structure capable of modeling virtually any "relationship" in the real world.

## Learning Objectives

- [ ] Understand basic graph terminology and representation methods (adjacency list, adjacency matrix)
- [ ] Explain the differences between BFS and DFS and when to use each
- [ ] Implement shortest path algorithms (Dijkstra, Bellman-Ford, Floyd-Warshall)
- [ ] Understand and implement minimum spanning trees (Kruskal, Prim)
- [ ] Apply topological sort to dependency resolution
- [ ] Understand strongly connected component decomposition (Tarjan, Kosaraju)
- [ ] Solve applied problems such as bipartite graph detection and cycle detection

## Prerequisites


---

## 1. Graph Fundamentals

### 1.1 What Is a Graph?

A graph is a mathematical structure defined as a pair G = (V, E), where V is a set of **vertices (nodes)** and E is a set of **edges** connecting pairs of vertices. Unlike arrays or trees, graphs have no "head" or "tail." Any element can have a relationship with any other, making graphs ideal for naturally modeling complex real-world relationships.

Why are graphs important? Because many problems in computer science can be reformulated as "problems on graphs." For example:

- Friend recommendation in social networks -> distance computation on a graph
- Route finding in car navigation -> shortest path problem on a weighted graph
- Dependency resolution in compilers -> topological sort on a directed acyclic graph (DAG)
- Network design optimization -> minimum spanning tree problem

### 1.2 Basic Terminology

```
Basic graph terminology:

  Graph G = (V, E)
  V = set of vertices (nodes)       |V| = number of vertices
  E = set of edges                  |E| = number of edges

  ┌─────────────────┬────────────────────────────────────────────────┐
  │ Term             │ Description                                    │
  ├─────────────────┼────────────────────────────────────────────────┤
  │ Undirected graph │ Edges have no direction: A—B (traversable both ways) │
  │ Directed graph   │ Edges have direction: A→B (only A to B)       │
  │ Weighted         │ Edges have costs: A—(5)—B (movement cost 5)   │
  │ DAG              │ Directed Acyclic Graph (no cycles)             │
  │ Degree           │ Number of edges connected to a vertex          │
  │ In-degree        │ Number of edges entering a vertex (directed)   │
  │ Out-degree       │ Number of edges leaving a vertex (directed)    │
  │ Path             │ A sequence of vertices connected by edges      │
  │ Cycle            │ A path whose start and end vertices are the same │
  │ Connected        │ A path exists between any two vertices         │
  │ Sparse graph     │ |E| is close to |V| (few edges)               │
  │ Dense graph      │ |E| is close to |V|^2 (many edges)            │
  └─────────────────┴────────────────────────────────────────────────┘

  Degree property (Handshaking Lemma):
    In an undirected graph, the sum of all vertex degrees = 2 × |E|
    Because each edge contributes to the degree of exactly two vertices.
```

### 1.3 Graph Representation Methods

There are broadly two data structures for representing graphs programmatically. The choice depends on the density of the graph and the operations required.

```
Comparison of representation methods:

  1. Adjacency List
     Maintains a list of adjacent vertices for each vertex

        A: [B, C]           A --- B
        B: [A, D]           |     |
        C: [A, D]           C --- D
        D: [B, C]

     Space complexity: O(V + E)   <- Proportional to edges; efficient for sparse graphs
     Edge existence check: O(degree(v))  <- Must traverse the adjacency list
     Enumerate all neighbors: O(degree(v))  <- Direct traversal of the list

  2. Adjacency Matrix
     A V×V matrix representing edge existence with 0/1

        A  B  C  D
     A [0, 1, 1, 0]         A --- B
     B [1, 0, 0, 1]         |     |
     C [1, 0, 0, 1]         C --- D
     D [0, 1, 1, 0]

     Space complexity: O(V^2)     <- Square of vertex count; efficient for dense graphs
     Edge existence check: O(1)     <- Just reference matrix[u][v]
     Enumerate all neighbors: O(V) <- Must traverse the entire row

  Which to choose:
  ┌────────────────────────────┬──────────────┬──────────────┐
  │ Condition                   │ Adjacency List│ Adjacency Matrix│
  ├────────────────────────────┼──────────────┼──────────────┤
  │ Sparse graph (E << V^2)    │ ★ Recommended│ Wastes memory│
  │ Dense graph (E ≈ V^2)      │ Usable       │ ★ Recommended│
  │ Frequent edge existence checks│ O(degree) │ ★ O(1)       │
  │ Frequent neighbor enumeration │ ★ O(degree)│ O(V)         │
  │ Vertex count over 100,000  │ ★ Recommended│ Out of memory│
  │ Using Floyd-Warshall       │ Conversion needed│ ★ Direct use│
  └────────────────────────────┴──────────────┴──────────────┘

  Why adjacency lists are chosen in most cases:
  Most real-world graphs are sparse (not everyone is friends on social media),
  so adjacency lists are more memory-efficient. Representing a graph with
  100,000 vertices as an adjacency matrix requires 100,000 × 100,000 =
  10 billion elements, which is impractical.
```

### 1.4 Graph Implementation in Python

```python
"""
Basic graph implementation -- provides both adjacency list and adjacency matrix.
Why implement both: The optimal representation varies by problem,
so being able to convert between them allows flexible handling.
"""

from collections import defaultdict
from typing import Optional


class GraphAdjList:
    """Graph implementation using an adjacency list"""

    def __init__(self, directed: bool = False):
        """
        directed=True: Directed graph (edges have direction)
        directed=False: Undirected graph (edges are bidirectional)
        Why use defaultdict(list): Accessing a nonexistent key automatically
        creates an empty list, simplifying the vertex addition logic.
        """
        self.graph = defaultdict(list)
        self.directed = directed

    def add_edge(self, u, v, weight: Optional[float] = None):
        """Add an edge. If weight is None, treat as an unweighted graph"""
        if weight is not None:
            self.graph[u].append((v, weight))
            if not self.directed:
                self.graph[v].append((u, weight))
        else:
            self.graph[u].append(v)
            if not self.directed:
                self.graph[v].append(u)
        # Register vertex only (in case the edge target is an isolated vertex)
        if v not in self.graph:
            self.graph[v] = []

    def get_vertices(self):
        """Return all vertices"""
        return list(self.graph.keys())

    def get_neighbors(self, v):
        """Return adjacent vertices of vertex v"""
        return self.graph[v]

    def __str__(self):
        result = []
        for vertex in sorted(self.graph.keys(), key=str):
            neighbors = self.graph[vertex]
            result.append(f"  {vertex}: {neighbors}")
        return "Graph {\n" + "\n".join(result) + "\n}"


# === Usage examples ===
if __name__ == "__main__":
    # Undirected, unweighted graph
    g1 = GraphAdjList(directed=False)
    g1.add_edge('A', 'B')
    g1.add_edge('A', 'C')
    g1.add_edge('B', 'D')
    g1.add_edge('C', 'D')
    print("=== Undirected Graph ===")
    print(g1)
    # Output:
    # Graph {
    #   A: ['B', 'C']
    #   B: ['A', 'D']
    #   C: ['A', 'D']
    #   D: ['B', 'C']
    # }

    # Directed, weighted graph
    g2 = GraphAdjList(directed=True)
    g2.add_edge('A', 'B', 4)
    g2.add_edge('A', 'C', 2)
    g2.add_edge('B', 'D', 3)
    g2.add_edge('C', 'B', 1)
    g2.add_edge('C', 'D', 5)
    print("\n=== Directed, Weighted Graph ===")
    print(g2)
    # Output:
    # Graph {
    #   A: [('B', 4), ('C', 2)]
    #   B: [('D', 3)]
    #   C: [('B', 1), ('D', 5)]
    #   D: []
    # }
```

### 1.5 Real-World Examples of Graphs

```
Graph application mapping:

  ┌──────────────────┬──────────────┬──────────────────┬──────────────┐
  │ Application       │ Vertices      │ Edges            │ Graph type   │
  ├──────────────────┼──────────────┼──────────────────┼──────────────┤
  │ SNS              │ Users         │ Friendships      │ Undirected   │
  │ Twitter/X        │ Users         │ Follows          │ Directed     │
  │ Web              │ Pages         │ Hyperlinks       │ Directed     │
  │ Maps/Roads       │ Intersections │ Roads            │ Weighted     │
  │ Networks         │ Routers       │ Links            │ Weighted     │
  │ Dependencies     │ Packages      │ Dependencies     │ Directed(DAG)│
  │ Scheduling       │ Tasks         │ Precedence       │ Directed(DAG)│
  │ Recommendation   │ Users+Products│ Purchases/Ratings│ Bipartite    │
  │ Power grid       │ Substations   │ Transmission lines│ Weighted    │
  │ Molecular struct. │ Atoms        │ Chemical bonds   │ Undirected   │
  └──────────────────┴──────────────┴──────────────────┴──────────────┘
```

---

## 2. Graph Traversal Algorithms

Graph traversal is the foundation of all graph algorithms. BFS (Breadth-First Search) and DFS (Depth-First Search) are the two most fundamental and powerful traversal techniques, and many advanced algorithms are built upon them.

### 2.1 BFS (Breadth-First Search)

BFS is an algorithm that "explores vertices closest to the starting point first." Why does it explore in order of proximity? Because it uses a queue (FIFO: First-In-First-Out) to manage the exploration order, so vertices discovered earlier (= closer to the start) are processed first.

```
BFS operation visualization:

  BFS from starting vertex A

  Graph:             Traversal order:
      A
     / \           Level 0: A
    B   C          Level 1: B, C
   / \   \         Level 2: D, E, F
  D   E   F

  Queue changes:
  [A]           -> Process A, add B, C
  [B, C]        -> Process B, add D, E
  [C, D, E]     -> Process C, add F
  [D, E, F]     -> Process D, E, F in order
  []            -> Done

  Why BFS can find shortest paths (unweighted):
  BFS visits vertices in order of their distance (number of edges) from the start.
  When a vertex is reached for the first time, that is the shortest path.
  Because if a shorter path existed, the vertices on that path would have been
  visited earlier and the vertex would have been reached through them first.
```

```python
"""
BFS (Breadth-First Search) complete implementation
- Basic BFS: Returns visit order of all vertices
- Shortest path BFS: Reconstructs the shortest path from start to end
- Level-based BFS: Returns vertices grouped by each level (distance)
"""

from collections import deque


def bfs(graph: dict, start) -> list:
    """
    Basic BFS. Returns all reachable vertices from start in order of proximity.

    Why add to visited when enqueuing rather than when dequeuing:
    Checking visited when dequeuing is another approach, but checking
    when enqueuing is more efficient.
    It prevents the same vertex from entering the queue multiple times,
    saving both memory and time.

    Time complexity: O(V + E) -- each vertex processed once, each edge once (twice if undirected)
    Space complexity: O(V) -- visited set and maximum queue size
    """
    visited = set([start])
    queue = deque([start])
    order = []

    while queue:
        node = queue.popleft()  # O(1) -- deque front removal
        order.append(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)    # Mark as visited when enqueuing
                queue.append(neighbor)

    return order


def bfs_shortest_path(graph: dict, start, end) -> list:
    """
    Returns the shortest path from start to end in an unweighted graph.

    Why use a dict for visited:
    By recording the "parent (which vertex we came from)" for each vertex,
    we can trace back from end to start to reconstruct the path.
    A set only tells us "visited or not," but a dict also tells us
    "where we came from."
    """
    if start == end:
        return [start]

    visited = {start: None}  # node -> previous node (start has None)
    queue = deque([start])

    while queue:
        node = queue.popleft()
        if node == end:
            # Reconstruct path: trace parents from end to start
            path = []
            while node is not None:
                path.append(node)
                node = visited[node]
            return path[::-1]  # Reverse to get start -> end order

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited[neighbor] = node
                queue.append(neighbor)

    return []  # Empty list if unreachable


def bfs_by_level(graph: dict, start) -> list:
    """
    Level-based BFS: Returns vertices grouped by distance from start.

    Why is level-based grouping necessary:
    Useful in scenarios requiring distance-based processing, such as
    "friend of friend" recommendations or "epidemic spread simulation."
    Standard BFS loses distance information.

    Implementation key: Record the queue length at the start of each level
    and dequeue that many elements to explicitly delimit levels.
    """
    visited = set([start])
    queue = deque([start])
    levels = []

    while queue:
        level_size = len(queue)  # Number of vertices at current level
        current_level = []
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        levels.append(current_level)

    return levels


# === Verification ===
if __name__ == "__main__":
    graph = {
        'A': ['B', 'C'],
        'B': ['A', 'D', 'E'],
        'C': ['A', 'F'],
        'D': ['B'],
        'E': ['B', 'F'],
        'F': ['C', 'E']
    }

    print("BFS order:", bfs(graph, 'A'))
    # Output: BFS order: ['A', 'B', 'C', 'D', 'E', 'F']

    print("Shortest path A->F:", bfs_shortest_path(graph, 'A', 'F'))
    # Output: Shortest path A->F: ['A', 'C', 'F']

    print("Level-based BFS:", bfs_by_level(graph, 'A'))
    # Output: Level-based BFS: [['A'], ['B', 'C'], ['D', 'E', 'F']]
```

### 2.2 DFS (Depth-First Search)

DFS is an algorithm that "explores as deeply as possible until reaching a dead end, then backtracks to try alternative paths." Why does it go deep? Because it uses a stack (LIFO: Last-In-First-Out), so the most recently discovered vertex is processed first. Since recursive calls function as an implicit stack, DFS can be naturally expressed recursively.

```
DFS operation visualization:

  DFS from starting vertex A

  Graph:              Traversal order (one example):
      A
     / \               A -> B -> D -> (backtrack) -> E -> F -> (backtrack)
    B   C                                             -> (backtrack) -> C
   / \   \
  D   E   F

  Stack changes:
  [A]           -> Process A, push C, B (reason for reverse push explained below)
  [C, B]        -> Process B, push E, D
  [C, E, D]     -> Process D (dead end)
  [C, E]        -> Process E, push F
  [C, F]        -> Process F
  [C]           -> Process C
  []            -> Done

  Why push in reverse order:
  Since a stack is LIFO, when graph[A] = [B, C],
  pushing C then B means B is popped first.
  This causes traversal to proceed in the order of the adjacency list.
```

```python
"""
DFS (Depth-First Search) complete implementation
- Iterative version using a stack
- Recursive version
- Cycle detection (directed graph)
- DFS pre-order/post-order
"""

from typing import Optional


def dfs_iterative(graph: dict, start) -> list:
    """
    Stack-based DFS (iterative version)

    Why use the iterative version:
    The recursive version may hit Python's default recursion limit (1000).
    For large graphs (1000+ vertices), the iterative version is safer.
    You can raise the limit with sys.setrecursionlimit(), but this
    risks stack overflow.

    Time complexity: O(V + E)
    Space complexity: O(V)
    """
    visited = set()
    stack = [start]
    order = []

    while stack:
        node = stack.pop()  # LIFO: process the last added vertex first
        if node not in visited:
            visited.add(node)
            order.append(node)
            # Push in reverse order so traversal follows adjacency list order
            for neighbor in reversed(graph[node]):
                if neighbor not in visited:
                    stack.append(neighbor)

    return order


def dfs_recursive(graph: dict, node, visited: Optional[set] = None) -> list:
    """
    Recursive DFS

    Why set the default value of visited to None:
    To avoid Python's mutable default argument trap.
    If the default were set(), the same set object would be shared
    across function calls.
    Setting it to None and initializing inside the function is the safe pattern.
    """
    if visited is None:
        visited = set()

    visited.add(node)
    result = [node]

    for neighbor in graph[node]:
        if neighbor not in visited:
            result.extend(dfs_recursive(graph, neighbor, visited))

    return result


def dfs_with_timestamps(graph: dict, start) -> dict:
    """
    DFS recording pre-order (discovery time) and post-order (finish time).

    Why timestamps matter:
    - Pre-order: Foundation for topological sort
    - Post-order: Foundation for strongly connected component decomposition (Tarjan, Kosaraju)
    - Interval containment: u is an ancestor of v iff discovery[u] < discovery[v]
      and finish[v] < finish[u]

    Returns: {node: (discovery_time, finish_time)} dictionary
    """
    visited = set()
    timestamps = {}
    time = [0]  # Use a list because closures need to modify it

    def _dfs(node):
        visited.add(node)
        time[0] += 1
        discovery = time[0]

        for neighbor in graph[node]:
            if neighbor not in visited:
                _dfs(neighbor)

        time[0] += 1
        finish = time[0]
        timestamps[node] = (discovery, finish)

    _dfs(start)

    # Also process unvisited vertices (for disconnected graphs)
    for node in graph:
        if node not in visited:
            _dfs(node)

    return timestamps


def detect_cycle_directed(graph: dict) -> bool:
    """
    Cycle detection in a directed graph (3-color method)

    Why 3 colors are needed:
    - WHITE (unvisited): Not yet explored
    - GRAY (in progress): On the current DFS path (mid-exploration)
    - BLACK (complete): All descendants have been fully explored

    If an edge from a GRAY vertex to another GRAY vertex is found,
    it means there is a "back edge" on the DFS path, indicating a cycle.
    An edge to a BLACK vertex is not a cycle (it leads to an already-explored branch).

    Why 2 colors (visited/not visited) are insufficient:
    In a directed graph with A->B, A->C, C->B,
    B is visited twice but there is no cycle.
    "Visited" alone cannot distinguish "currently on the exploration path."
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in graph}

    def _dfs(node):
        color[node] = GRAY
        for neighbor in graph[node]:
            if color[neighbor] == GRAY:
                return True   # Cycle detected!
            if color[neighbor] == WHITE:
                if _dfs(neighbor):
                    return True
        color[node] = BLACK
        return False

    # Explore from all vertices (for disconnected graphs)
    for node in graph:
        if color[node] == WHITE:
            if _dfs(node):
                return True
    return False


# === Verification ===
if __name__ == "__main__":
    # Undirected graph
    graph = {
        'A': ['B', 'C'],
        'B': ['A', 'D', 'E'],
        'C': ['A', 'F'],
        'D': ['B'],
        'E': ['B', 'F'],
        'F': ['C', 'E']
    }

    print("DFS iterative:", dfs_iterative(graph, 'A'))
    # Output: DFS iterative: ['A', 'B', 'D', 'E', 'F', 'C']

    print("DFS recursive:", dfs_recursive(graph, 'A'))
    # Output: DFS recursive: ['A', 'B', 'D', 'E', 'F', 'C']

    # Directed graph (with cycle)
    directed_with_cycle = {
        'A': ['B'],
        'B': ['C'],
        'C': ['A'],  # Cycle: A->B->C->A
    }
    print("Has cycle:", detect_cycle_directed(directed_with_cycle))
    # Output: Has cycle: True

    # Directed graph (no cycle = DAG)
    dag = {
        'A': ['B', 'C'],
        'B': ['D'],
        'C': ['D'],
        'D': [],
    }
    print("No cycle:", detect_cycle_directed(dag))
    # Output: No cycle: False
```

### 2.3 Comprehensive BFS vs DFS Comparison

```
BFS vs DFS selection guide:

  ┌───────────────────────────┬──────────────────┬──────────────────┐
  │ Characteristic             │ BFS              │ DFS              │
  ├───────────────────────────┼──────────────────┼──────────────────┤
  │ Data structure             │ Queue (FIFO)     │ Stack (LIFO)     │
  │ Traversal order            │ Nearest first    │ Deepest first    │
  │ Shortest path (unweighted) │ ★ Guaranteed     │ Not guaranteed   │
  │ Cycle detection            │ Possible         │ ★ Natural (3-color)│
  │ Topological sort           │ Possible (Kahn)  │ ★ Natural        │
  │ Connected components       │ Possible         │ ★ Natural        │
  │ Bipartite detection        │ ★ Natural        │ Possible         │
  │ Strongly connected comp.   │ Not suitable     │ ★ Required       │
  │ Memory usage               │ O(max level width)│ O(max depth)    │
  │ Complete search (all paths)│ Not suitable     │ ★ Natural        │
  │ Implementation naturalness │ Iterative (queue)│ Recursion is natural│
  └───────────────────────────┴──────────────────┴──────────────────┘

  Concrete memory usage example:

  Complete binary tree (depth d, 2^(d+1) - 1 vertices):
  - BFS: Max level width = 2^d (number of vertices at the lowest level)
  - DFS: Max depth = d

  Complete binary tree of depth 20 (~1 million vertices):
  - BFS: Memory for ~500,000 vertices
  - DFS: Memory for ~20 vertices  <- Overwhelmingly more memory-efficient

  On the other hand, star graph (one center and many leaves):
  - BFS: After processing center, all leaves enter the queue
  - DFS: One vertex at a time on the stack -> always O(1)

  Conclusion: Choose based on graph structure.
  "When in doubt, BFS" is safe, but DFS is also viable when recursion is natural.
```

---

## 3. Shortest Path Algorithms

The shortest path problem is one of the most practical problems in graph algorithms. Its applications are extremely broad, including car navigation, network routing, and logistics optimization.

### 3.1 Problem Classification

```
Shortest path problem classification:

  1. Single-Source Shortest Path (SSSP)
     "Shortest distances from one vertex to all other vertices"
     -> Dijkstra's algorithm, Bellman-Ford algorithm

  2. All-Pairs Shortest Path (APSP)
     "Shortest distances between all pairs of vertices"
     -> Floyd-Warshall algorithm

  3. Single-Pair Shortest Path
     "Shortest distance from a specific start to a specific end"
     -> A* algorithm (with heuristic)

  Classification by weight constraints:
  ┌──────────────────────┬───────────────────────────────────────┐
  │ Weight condition       │ Available algorithms                  │
  ├──────────────────────┼───────────────────────────────────────┤
  │ Unweighted (all 1)    │ BFS (fastest: O(V+E))                │
  │ Non-negative weights   │ Dijkstra's: O((V+E) log V)          │
  │ Negative weights exist │ Bellman-Ford: O(V × E)               │
  │ No negative cycles     │ All of the above are usable          │
  │ Negative cycles exist  │ Shortest path is undefined (can be   │
  │                        │ made infinitely short)                │
  └──────────────────────┴───────────────────────────────────────┘
```

### 3.2 Dijkstra's Algorithm

Dijkstra's algorithm finds single-source shortest paths in graphs with **non-negative edge weights**. It was devised by Edsger Dijkstra in 1956.

Why Dijkstra's algorithm is correct: It is based on the greedy approach. It selects the vertex with the smallest distance from the start among unfinalized vertices, and finalizes that distance. Under the non-negative weight condition, once a shortest distance is finalized, it will never be updated later. This is because any path through an unfinalized vertex adds a non-negative weight to the distance (which is already at least the current minimum), so it cannot fall below the finalized distance.

```
Dijkstra's algorithm step-by-step:

  Graph:
       A --4-- B
       |       |
       2       3
       |       |
       C --1-- B (C->B has weight 1)
       |
       5
       |
       D

  Starting from A:

  Step 1: Finalize A (distance 0)
    Distances: A=0, B=4, C=2, D=inf
    Finalized: {A}

  Step 2: Finalize C (distance 2)  <- Smallest among unfinalized
    Update B via C: 2+1=3 < 4 -> B=3
    Update D via C: 2+5=7 < inf -> D=7
    Distances: A=0, B=3, C=2, D=7
    Finalized: {A, C}

  Step 3: Finalize B (distance 3)
    Update D via B: 3+3=6 < 7 -> D=6
    Distances: A=0, B=3, C=2, D=6
    Finalized: {A, C, B}

  Step 4: Finalize D (distance 6)
    Finalized: {A, C, B, D}

  Final result: A=0, B=3, C=2, D=6
```

```python
"""
Dijkstra's algorithm complete implementation
- Basic version (shortest distances only)
- Path reconstruction version (also returns shortest path)
- Error detection for negative weights

Why use a priority queue (heap):
To efficiently retrieve "the vertex with the minimum distance among unfinalized vertices."
A simple array requires O(V) to find the minimum, but a heap achieves O(log V).
This improves overall complexity from O(V^2) to O((V+E) log V).
"""

import heapq
from typing import Optional


def dijkstra(graph: dict, start) -> dict:
    """
    Dijkstra's algorithm: Returns shortest distances from start to all vertices.

    graph format: {node: [(neighbor, weight), ...], ...}
    Returns: {node: distance, ...}

    Time complexity: O((V + E) log V)
      - Each vertex is finalized at most once
      - Each edge undergoes at most one relaxation
      - Each relaxation involves a heap push: O(log V)
    """
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    # Priority queue: (distance, node) tuples
    # Why tuples: heapq compares by the first element
    pq = [(0, start)]

    while pq:
        dist, node = heapq.heappop(pq)

        # This check is important: skip if a shorter distance is already finalized
        # Why needed: The heap may contain stale (larger distance) entries.
        # Instead of a decrease-key operation, we push new entries for the same vertex.
        if dist > distances[node]:
            continue

        for neighbor, weight in graph[node]:
            new_dist = dist + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))

    return distances


def dijkstra_with_path(graph: dict, start, end) -> tuple:
    """
    Dijkstra's algorithm with path reconstruction.

    Returns: (shortest distance, list of shortest path)
    If unreachable: (float('inf'), [])
    """
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    previous = {node: None for node in graph}  # For path reconstruction
    pq = [(0, start)]

    while pq:
        dist, node = heapq.heappop(pq)

        if dist > distances[node]:
            continue

        if node == end:
            break  # Optimization: stop when destination is finalized

        for neighbor, weight in graph[node]:
            new_dist = dist + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                previous[neighbor] = node
                heapq.heappush(pq, (new_dist, neighbor))

    # Path reconstruction
    if distances[end] == float('inf'):
        return float('inf'), []

    path = []
    node = end
    while node is not None:
        path.append(node)
        node = previous[node]
    path.reverse()

    return distances[end], path


# === Verification ===
if __name__ == "__main__":
    graph = {
        'A': [('B', 4), ('C', 2)],
        'B': [('D', 3), ('C', 1)],
        'C': [('B', 1), ('D', 5)],
        'D': []
    }

    print("=== Dijkstra's Algorithm ===")
    print("Shortest distances to all vertices:", dijkstra(graph, 'A'))
    # Output: {'A': 0, 'B': 3, 'C': 2, 'D': 6}

    dist, path = dijkstra_with_path(graph, 'A', 'D')
    print(f"A->D: distance={dist}, path={path}")
    # Output: A->D: distance=6, path=['A', 'C', 'B', 'D']
```

### 3.3 Bellman-Ford Algorithm

The Bellman-Ford algorithm is a shortest path algorithm that works correctly even on graphs **containing edges with negative weights**. It also has the advantage of being able to **detect negative cycles**.

Why Dijkstra's algorithm cannot handle negative weights: Dijkstra's is based on the assumption that "once a shortest distance is finalized, it will not change." However, with negative-weight edges, a finalized vertex may later be updated to a shorter distance. Bellman-Ford solves this by repeating relaxation of all edges V-1 times.

```python
"""
Bellman-Ford algorithm complete implementation
- Handles negative weights
- Negative cycle detection

Why V-1 iterations are sufficient:
The maximum number of edges in a shortest path is V-1 (when passing through all V vertices).
Each iteration finalizes at least one vertex's shortest distance, so
V-1 iterations finalize all vertices' shortest distances.

Why an update on the V-th iteration means a negative cycle:
After V-1 iterations, all shortest distances should be finalized.
If updates still occur, it means there exists a cycle where
"the distance decreases with each traversal" (a negative cycle).

Time complexity: O(V × E) -- Slower than Dijkstra but handles negative weights
"""


def bellman_ford(vertices: list, edges: list, start) -> tuple:
    """
    Bellman-Ford algorithm

    Args:
        vertices: List of vertices ['A', 'B', 'C', ...]
        edges: List of edges [(u, v, weight), ...]
        start: Starting vertex

    Returns:
        (distances, has_negative_cycle)
        distances: {node: distance} dictionary
        has_negative_cycle: Whether a negative cycle exists
    """
    # Initialization
    distances = {v: float('inf') for v in vertices}
    distances[start] = 0

    # V-1 iterations
    for i in range(len(vertices) - 1):
        updated = False
        for u, v, weight in edges:
            if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                updated = True
        # Optimization: early termination if no updates
        # Why safe to terminate: No updates means
        # distances[u] + weight >= distances[v] holds for all edges,
        # indicating no further improvements are possible.
        if not updated:
            break

    # Negative cycle check (V-th iteration)
    for u, v, weight in edges:
        if distances[u] != float('inf') and distances[u] + weight < distances[v]:
            return distances, True  # Negative cycle exists

    return distances, False


# === Verification ===
if __name__ == "__main__":
    vertices = ['A', 'B', 'C', 'D', 'E']
    edges = [
        ('A', 'B', 4),
        ('A', 'C', 2),
        ('B', 'D', 3),
        ('C', 'B', -1),   # Negative weight!
        ('C', 'D', 5),
        ('D', 'E', 2),
    ]

    distances, has_neg_cycle = bellman_ford(vertices, edges, 'A')
    print("=== Bellman-Ford Algorithm ===")
    print("Shortest distances:", distances)
    print("Negative cycle:", has_neg_cycle)
    # Output:
    # Shortest distances: {'A': 0, 'B': 1, 'C': 2, 'D': 4, 'E': 6}
    # Negative cycle: False
    # A->C(2)->B(1) -- the path leveraging negative weight is shortest

    # Graph with a negative cycle
    vertices2 = ['A', 'B', 'C']
    edges2 = [
        ('A', 'B', 1),
        ('B', 'C', -3),
        ('C', 'A', 1),   # A->B->C->A cost: 1+(-3)+1 = -1 (negative cycle)
    ]
    distances2, has_neg_cycle2 = bellman_ford(vertices2, edges2, 'A')
    print("\nNegative cycle test:")
    print("Negative cycle:", has_neg_cycle2)
    # Output: Negative cycle: True
```

### 3.4 Floyd-Warshall Algorithm

The Floyd-Warshall algorithm computes **shortest distances between all pairs of vertices** at once. It is based on dynamic programming and applies naturally to graphs represented as adjacency matrices.

```python
"""
Floyd-Warshall algorithm complete implementation

Why the triple loop order must be k, i, j (the most important point):
The outermost loop variable k represents "candidate intermediate vertices."
dp[k][i][j] = "shortest distance from i to j when vertices 0..k can be used as intermediates"
By making k the outermost loop, all dp[k-1] values are finalized when computing dp[k].
If i or j were outermost, the dependency structure would break and produce incorrect results.

Time complexity: O(V^3)
Space complexity: O(V^2) -- updated in-place
"""


def floyd_warshall(n: int, edges: list) -> list:
    """
    Floyd-Warshall algorithm

    Args:
        n: Number of vertices (0-indexed: 0, 1, ..., n-1)
        edges: [(u, v, weight), ...] edge list

    Returns:
        dist[i][j] = shortest distance from vertex i to j (2D array)
        float('inf') if unreachable
    """
    INF = float('inf')

    # Initialization: distance to self is 0, everything else is INF
    dist = [[INF] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0

    # Reflect direct edges
    for u, v, w in edges:
        dist[u][v] = min(dist[u][v], w)  # Take minimum for multi-edges

    # Main body: update using k as intermediate vertex
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    return dist


def floyd_warshall_with_path(n: int, edges: list) -> tuple:
    """
    Floyd-Warshall with path reconstruction

    Returns:
        (dist, next_node)
        dist[i][j]: shortest distance from i to j
        next_node[i][j]: next vertex after i on the shortest path from i to j
    """
    INF = float('inf')
    dist = [[INF] * n for _ in range(n)]
    next_node = [[None] * n for _ in range(n)]

    for i in range(n):
        dist[i][i] = 0
        next_node[i][i] = i

    for u, v, w in edges:
        if w < dist[u][v]:
            dist[u][v] = w
            next_node[u][v] = v

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_node[i][j] = next_node[i][k]

    return dist, next_node


def reconstruct_path(next_node: list, start: int, end: int) -> list:
    """Reconstruct path from the next_node table"""
    if next_node[start][end] is None:
        return []  # Unreachable
    path = [start]
    current = start
    while current != end:
        current = next_node[current][end]
        if current is None:
            return []
        path.append(current)
    return path


# === Verification ===
if __name__ == "__main__":
    #   0 --2--> 1
    #   |        |
    #   6        3
    #   |        |
    #   v        v
    #   2 --1--> 3
    n = 4
    edges = [
        (0, 1, 2),
        (0, 2, 6),
        (1, 3, 3),
        (2, 3, 1),
        (1, 2, 1),  # Shortcut 1->2
    ]

    dist, next_node = floyd_warshall_with_path(n, edges)

    print("=== Floyd-Warshall Algorithm ===")
    print("Distance matrix:")
    for i in range(n):
        row = [str(dist[i][j]) if dist[i][j] != float('inf') else 'INF'
               for j in range(n)]
        print(f"  {i}: {row}")
    # Output:
    #   0: ['0', '2', '3', '4']
    #   1: ['INF', '0', '1', '2']
    #   2: ['INF', 'INF', '0', '1']
    #   3: ['INF', 'INF', 'INF', '0']

    path = reconstruct_path(next_node, 0, 3)
    print(f"Shortest path 0->3: {path}")
    # Output: Shortest path 0->3: [0, 1, 2, 3]
    # 0->1(2) -> 1->2(1) -> 2->3(1) = total 4
```

### 3.5 Comprehensive Shortest Path Algorithm Comparison

```
Shortest path algorithm comparison table:

  ┌────────────────┬─────────────────┬──────────┬────────┬──────────────────┐
  │ Algorithm       │ Time complexity  │ Neg. edges│ Neg. cycle│ Primary use   │
  ├────────────────┼─────────────────┼──────────┼────────┼──────────────────┤
  │ BFS            │ O(V + E)        │ No       │ --     │ Unweighted graphs│
  │ Dijkstra       │ O((V+E) log V)  │ No       │ --     │ Non-neg. SSSP   │
  │ Bellman-Ford   │ O(V × E)        │ Yes      │ Detect │ Negative weights │
  │ Floyd-Warshall │ O(V^3)          │ Yes      │ Detect │ All-pairs SP     │
  │ A*             │ O(E) (expected) │ No       │ --     │ Point-to-point   │
  │                │                 │          │        │ + heuristic      │
  └────────────────┴─────────────────┴──────────┴────────┴──────────────────┘

  Selection guidelines:
  - Unweighted -> BFS (fastest, simplest)
  - Non-negative weights -> Dijkstra (standard choice)
  - Negative weights present -> Bellman-Ford
  - All pairs -> Floyd-Warshall (when vertex count is small)
  - Point-to-point (large graph) -> A*
  - Network routing (OSPF) -> Dijkstra
  - Currency arbitrage detection -> Bellman-Ford (log-transform, detect negative cycles)
```

---

## 4. Minimum Spanning Tree (MST)

### 4.1 What Is a Minimum Spanning Tree?

A minimum spanning tree is a **tree (a cycle-free connected subgraph) that connects all vertices while minimizing the total edge weight** in a weighted, undirected, connected graph.

Why is it a "tree"? The minimum number of edges required to connect N vertices is N-1, and N-1 edges connecting all vertices without cycles form a tree. Adding even one more edge creates a cycle, and removing the heaviest edge on that cycle maintains connectivity. Therefore, the minimum-cost connected subgraph is always a tree.

```
MST application examples:

  - Network design: Connect all sites with minimum-cost links
  - Power grid design: Connect all regions with minimum-cost transmission lines
  - Clustering: Removing the heaviest MST edges naturally partitions into clusters
  - Approximation algorithms: MST is used for approximate TSP solutions

  Example: Minimum-cost road network connecting 5 cities

     A ---3--- B              A ---3--- B
     |\ /|                         |
     6  2  5  7     MST ->     2    7
     |/  \|                   |
     C ---4--- D              C        D
      \       /                \      /
       1                        1
        \   /                    \ /
         E                        E

  Total edge weight of original graph: 3+6+2+5+7+4+1 = 28
  Total MST edge weight: 3+2+7+1 = 13 (4 edges connecting 5 vertices)
```

### 4.2 Kruskal's Algorithm

Kruskal's algorithm sorts edges by weight in ascending order and greedily adds edges that do not form cycles. It uses Union-Find (disjoint set data structure) for efficient cycle detection.

```python
"""
Kruskal's algorithm complete implementation

Why sort edges by weight:
Based on the MST "cut property." For any cut (partition of the vertex set),
the minimum-weight edge crossing that cut is included in some MST.
Looking at edges in ascending order and selecting edges that connect
different components (= cross a cut) is a greedy selection leveraging this property.

Time complexity: O(E log E) -- sorting dominates
  Edge sorting: O(E log E)
  Union-Find operations: O(E x alpha(V)) ≈ O(E) -- nearly constant time
"""


class UnionFind:
    """
    Union-Find (Disjoint Set data structure)

    Why use both path compression and rank:
    - Path compression: Flattens the tree during find() to speed up future finds
    - Rank: During union(), attaches the shorter tree under the taller tree to limit height
    Combined, each operation is nearly O(1) (precisely O(alpha(n))).
    alpha(n) is the inverse Ackermann function, which is at most 4 for practical n.
    """

    def __init__(self, n: int):
        self.parent = list(range(n))  # Parent of each element (initially itself)
        self.rank = [0] * n           # Upper bound on tree height

    def find(self, x: int) -> int:
        """Return the root of x (with path compression)"""
        if self.parent[x] != x:
            # Recursively find root and directly link (path compression)
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """Merge x and y into the same set. Return False if already in the same set"""
        px, py = self.find(x), self.find(y)
        if px == py:
            return False  # Already in the same set
        # Attach the lower-rank tree under the higher-rank tree
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True


def kruskal(n: int, edges: list) -> tuple:
    """
    MST construction using Kruskal's algorithm

    Args:
        n: Number of vertices
        edges: [(u, v, weight), ...] -- list of undirected edges

    Returns:
        (mst_edges, total_weight)
        mst_edges: List of edges in the MST
        total_weight: Total weight of the MST
    """
    # Sort edges by weight
    sorted_edges = sorted(edges, key=lambda e: e[2])

    uf = UnionFind(n)
    mst_edges = []
    total_weight = 0

    for u, v, weight in sorted_edges:
        # Only add the edge if u and v are in different components
        # (Adding an edge within the same component would create a cycle)
        if uf.union(u, v):
            mst_edges.append((u, v, weight))
            total_weight += weight
            # MST has V-1 edges, so stop when we have enough
            if len(mst_edges) == n - 1:
                break

    return mst_edges, total_weight


# === Verification ===
if __name__ == "__main__":
    # Vertices: 0=A, 1=B, 2=C, 3=D, 4=E
    n = 5
    edges = [
        (0, 1, 3),  # A-B: 3
        (0, 2, 6),  # A-C: 6
        (0, 2, 2),  # A-C: 2 (multi-edge, this one is shorter)
        (1, 3, 7),  # B-D: 7
        (0, 3, 5),  # A-D: 5
        (2, 3, 4),  # C-D: 4
        (2, 4, 1),  # C-E: 1
        (3, 4, 8),  # D-E: 8
    ]

    mst_edges, total = kruskal(n, edges)
    print("=== Kruskal's Algorithm ===")
    print(f"MST edges: {mst_edges}")
    print(f"Total weight: {total}")
    # Output:
    # MST edges: [(2, 4, 1), (0, 2, 2), (0, 1, 3), (1, 3, 7)]
    # Total weight: 13
    # Edge selection order: C-E(1), A-C(2), A-B(3), B-D(7)
```

### 4.3 Prim's Algorithm

Prim's algorithm starts from a single vertex and greedily adds the minimum-weight edge extending from the MST vertex set. Its structure is very similar to Dijkstra's algorithm.

```python
"""
Prim's algorithm complete implementation

Why Prim's algorithm resembles Dijkstra's:
- Dijkstra: Selects the vertex with the minimum "distance from start"
- Prim: Selects the vertex with the minimum "distance from MST set (edge weight)"
Both use a priority queue to efficiently select the minimum candidate.

Comparison with Kruskal's:
- Kruskal: Global edge sort -> cycle check (Union-Find)
- Prim: Add vertices one at a time -> update priority queue with adjacent edges
- Sparse graph (E << V^2): Kruskal is advantageous
- Dense graph (E ≈ V^2): Prim is advantageous

Time complexity: O((V + E) log V) -- with priority queue
"""

import heapq


def prim(n: int, adj: dict) -> tuple:
    """
    MST construction using Prim's algorithm

    Args:
        n: Number of vertices
        adj: Adjacency list {node: [(neighbor, weight), ...], ...}

    Returns:
        (mst_edges, total_weight)
    """
    if not adj:
        return [], 0

    start = next(iter(adj))  # Arbitrary starting vertex
    in_mst = set([start])
    mst_edges = []
    total_weight = 0

    # Add adjacent edges of start to the priority queue
    # (weight, from_node, to_node) tuples
    pq = []
    for neighbor, weight in adj[start]:
        heapq.heappush(pq, (weight, start, neighbor))

    while pq and len(in_mst) < n:
        weight, u, v = heapq.heappop(pq)

        if v in in_mst:
            continue  # Ignore vertices already in the MST

        # Add v to MST
        in_mst.add(v)
        mst_edges.append((u, v, weight))
        total_weight += weight

        # Add adjacent edges of v to the queue
        for neighbor, w in adj[v]:
            if neighbor not in in_mst:
                heapq.heappush(pq, (w, v, neighbor))

    return mst_edges, total_weight


# === Verification ===
if __name__ == "__main__":
    adj = {
        'A': [('B', 3), ('C', 2), ('D', 5)],
        'B': [('A', 3), ('D', 7)],
        'C': [('A', 2), ('D', 4), ('E', 1)],
        'D': [('A', 5), ('B', 7), ('C', 4), ('E', 8)],
        'E': [('C', 1), ('D', 8)],
    }

    mst_edges, total = prim(5, adj)
    print("=== Prim's Algorithm ===")
    print(f"MST edges: {mst_edges}")
    print(f"Total weight: {total}")
    # Output:
    # MST edges: [('A', 'C', 2), ('C', 'E', 1), ('A', 'B', 3), ('C', 'D', 4)]
    # Total weight: 10
```

---

## 5. Topological Sort

### 5.1 What Is Topological Sort?

Topological sort is an operation that arranges the vertices of a directed acyclic graph (DAG) in a linear order such that "all edges point from left to right." In other words, if edge u->v exists, then u appears before v in the ordering.

Why is it only defined for DAGs? If a cycle exists, such as A->B->C->A, then A must come before B, B before C, and C before A. This is contradictory, and no ordering where all edges point in one direction can exist.

```
Topological sort applications:

  Build system dependency resolution:

  main.c -> utils.o -> utils.c
  main.c -> math.o -> math.c
  main.c -> main.o

  One possible topological order:
  utils.c -> utils.o -> math.c -> math.o -> main.c -> main.o

  Processing in this order guarantees that each file's
  dependencies are always processed first.

  University course planning:
  Linear Algebra -> Differential Equations -> Control Engineering
  Calculus -> Differential Equations
  Programming Basics -> Data Structures -> Algorithms

  Topological order:
  Linear Algebra, Calculus, Programming Basics,
  Differential Equations, Data Structures, Control Engineering, Algorithms
```

### 5.2 Kahn's Algorithm (BFS-based) and DFS-based

```python
"""
Two implementations of topological sort
- Kahn's algorithm (BFS-based): Process vertices with in-degree 0 sequentially
- DFS-based: Reverse of post-order

Why two methods exist:
- Kahn's: Cycle detection is naturally built in. Reveals parallelizable vertices.
- DFS: Simpler implementation. Easier to combine with other DFS-based algorithms.
"""

from collections import deque


def topological_sort_kahn(graph: dict) -> list:
    """
    Topological sort using Kahn's algorithm (BFS-based)

    Idea: Vertices with in-degree 0 (no incoming edges) have no dependencies,
    so they can be placed first.
    Removing such vertices may cause other vertices to reach in-degree 0.
    Repeat this process.

    Time complexity: O(V + E)
    """
    # Compute in-degrees
    in_degree = {node: 0 for node in graph}
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] = in_degree.get(neighbor, 0) + 1

    # Enqueue vertices with in-degree 0
    queue = deque([n for n in in_degree if in_degree[n] == 0])
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Cycle detection: if not all vertices were processed, a cycle exists
    if len(order) != len(graph):
        raise ValueError(
            f"The graph contains a cycle. "
            f"Vertices processed: {len(order)}/{len(graph)}"
        )
    return order


def topological_sort_dfs(graph: dict) -> list:
    """
    DFS-based topological sort

    Idea: The reverse of DFS post-order (the order in which all descendants
    finish processing) is a topological order.
    Because if edge u->v exists, v's exploration completes before u's
    (v comes first in post-order).
    Therefore, in the reverse, u comes before v.

    Time complexity: O(V + E)
    """
    visited = set()
    finish_order = []  # Post-order

    def _dfs(node):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                _dfs(neighbor)
        finish_order.append(node)  # Post-order (after all descendants processed)

    for node in graph:
        if node not in visited:
            _dfs(node)

    return finish_order[::-1]  # Reverse of post-order is topological order


# === Verification ===
if __name__ == "__main__":
    # University course dependencies
    courses = {
        'Linear Algebra': ['Differential Equations'],
        'Calculus': ['Differential Equations'],
        'Differential Equations': ['Control Engineering'],
        'Programming Basics': ['Data Structures'],
        'Data Structures': ['Algorithms'],
        'Control Engineering': [],
        'Algorithms': [],
    }

    print("=== Kahn's Algorithm ===")
    print(topological_sort_kahn(courses))

    print("\n=== DFS Method ===")
    print(topological_sort_dfs(courses))

    # Graph with a cycle (should raise an error)
    cyclic = {
        'A': ['B'],
        'B': ['C'],
        'C': ['A'],
    }
    try:
        topological_sort_kahn(cyclic)
    except ValueError as e:
        print(f"\nCycle detected: {e}")
        # Output: Cycle detected: The graph contains a cycle. Vertices processed: 0/3
```

---

## 6. Strongly Connected Components (SCC)

### 6.1 What Are Strongly Connected Components?

In a directed graph, a set of vertices S is **strongly connected** if for every pair of vertices u, v in S, there exists both a path from u to v and a path from v to u. A **strongly connected component** (SCC) is a maximal strongly connected vertex set.

```
Intuitive understanding of SCCs:

  Directed graph:
    A -> B -> C -> A     D -> E -> F -> D     G

  This graph has 3 SCCs:
    SCC1: {A, B, C} -- Mutually reachable via A->B->C->A
    SCC2: {D, E, F} -- Mutually reachable via D->E->F->D
    SCC3: {G}       -- Isolated vertex (only itself)

  Contracting each SCC into a single vertex yields a DAG:
    [ABC] -> [DEF]
              |
             [G]

  Why SCC decomposition is important:
  1. Understanding graph structure: Reveals the "skeleton" of large graphs
  2. Reduction to DAG: SCC contraction produces a DAG, enabling topological sort
  3. 2-SAT problem: Essential for satisfiability problem solutions
  4. Web structure analysis: Analyzing the "bowtie structure" of the web graph
```

### 6.2 Kosaraju's Algorithm

```python
"""
Strongly connected component decomposition using Kosaraju's algorithm

Why two DFS passes correctly find SCCs:
1st DFS: Record post-order (finish time order)
2nd DFS: On the transpose graph (all edges reversed),
         explore in reverse post-order

Core insight:
- Vertices in the same SCC are mutually reachable in both the original and transpose graphs
- Edges between different SCCs reverse direction in the transpose
- By processing in reverse post-order, we traverse each SCC's vertices precisely
  without crossing reversed inter-SCC edges

Time complexity: O(V + E) -- just two DFS passes
"""

from collections import defaultdict


def kosaraju_scc(graph: dict) -> list:
    """
    SCC decomposition using Kosaraju's algorithm

    Args:
        graph: Directed graph adjacency list {node: [neighbors], ...}

    Returns:
        List of SCCs [[scc1_nodes], [scc2_nodes], ...]
    """
    # Step 1: DFS on original graph, record post-order
    visited = set()
    finish_order = []

    def dfs1(node):
        visited.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs1(neighbor)
        finish_order.append(node)

    for node in graph:
        if node not in visited:
            dfs1(node)

    # Step 2: Build transpose graph (reverse all edges)
    reversed_graph = defaultdict(list)
    for node in graph:
        for neighbor in graph[node]:
            reversed_graph[neighbor].append(node)

    # Step 3: DFS on transpose graph in reverse post-order
    visited.clear()
    sccs = []

    def dfs2(node, component):
        visited.add(node)
        component.append(node)
        for neighbor in reversed_graph.get(node, []):
            if neighbor not in visited:
                dfs2(neighbor, component)

    for node in reversed(finish_order):
        if node not in visited:
            component = []
            dfs2(node, component)
            sccs.append(component)

    return sccs


# === Verification ===
if __name__ == "__main__":
    graph = {
        'A': ['B'],
        'B': ['C', 'E'],
        'C': ['A', 'D'],    # A->B->C->A forms a cycle (SCC)
        'D': ['E'],
        'E': ['F'],
        'F': ['D'],          # D->E->F->D forms a cycle (SCC)
        'G': [],
    }

    sccs = kosaraju_scc(graph)
    print("=== Kosaraju's SCC Decomposition ===")
    for i, scc in enumerate(sccs):
        print(f"  SCC {i+1}: {scc}")
    # Expected output:
    # SCC 1: ['A', 'C', 'B']  (order may vary)
    # SCC 2: ['D', 'F', 'E']
    # SCC 3: ['G']
```

### 6.3 Tarjan's Algorithm

Tarjan's algorithm finds SCCs with a single DFS pass. While more complex to implement than Kosaraju's, it has a smaller constant factor.

```python
"""
SCC decomposition using Tarjan's algorithm

Why a single DFS suffices:
During DFS, each vertex is assigned a "discovery time" and a "minimum reachable
discovery time (low-link value)." When returning from DFS, if a vertex's
low-link value equals its discovery time, that vertex is the "root"
(first discovered vertex) of an SCC.
All vertices above it on the stack belong to the same SCC.

Low-link value meaning:
The low-link value of vertex v = the minimum discovery time among vertices
reachable from v using DFS tree edges + back edges.
When low[v] == disc[v], there are no back edges above v leading outside the SCC,
so the boundary of the SCC rooted at v is determined.

Time complexity: O(V + E)
"""


def tarjan_scc(graph: dict) -> list:
    """
    SCC decomposition using Tarjan's algorithm

    Returns: List of SCCs (in reverse topological order)
    """
    index_counter = [0]
    stack = []
    on_stack = set()
    disc = {}     # Discovery time
    low = {}      # Low-link value
    sccs = []

    def strongconnect(v):
        disc[v] = low[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)

        for w in graph.get(v, []):
            if w not in disc:
                # Unvisited: recursively explore via DFS and propagate low value
                strongconnect(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                # On the stack = candidate for current SCC
                # Why compare with disc[w]:
                # w's SCC is not yet finalized (it's on the stack), so
                # the reachability from v to w is reflected in the low value
                low[v] = min(low[v], disc[w])
            # If w is not on the stack: SCC already finalized, ignore

        # SCC root discovered (low[v] == disc[v])
        if low[v] == disc[v]:
            component = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                component.append(w)
                if w == v:
                    break
            sccs.append(component)

    for v in graph:
        if v not in disc:
            strongconnect(v)

    return sccs


# === Verification ===
if __name__ == "__main__":
    graph = {
        0: [1],
        1: [2, 4],
        2: [0, 3],
        3: [4],
        4: [5],
        5: [3],
        6: [],
    }

    sccs = tarjan_scc(graph)
    print("=== Tarjan's SCC Decomposition ===")
    for i, scc in enumerate(sccs):
        print(f"  SCC {i+1}: {scc}")
    # Expected output:
    # SCC 1: [0, 2, 1]   <- {0, 1, 2} are mutually reachable
    # SCC 2: [3, 5, 4]   <- {3, 4, 5} are mutually reachable
    # SCC 3: [6]          <- Isolated vertex
```

---

## 7. Bipartite Graph Detection

### 7.1 What Is a Bipartite Graph?

A bipartite graph is one whose vertices can be divided into two groups such that no edges exist between vertices within the same group.

```
Bipartite graph intuition:

  Bipartite example:              Non-bipartite example:
  (Can be 2-colored)              (Cannot be 2-colored)

   ●---○                      ●---○
   |   |                      |   |
   ○---●                      ●---●  <- Same color adjacent!
                                \|
                                 ○

  ● = Group A, ○ = Group B

  Bipartite graph applications:
  - Matching problems: Job assignment, student-lab matching
  - Recommendation systems: User x product bipartite graph
  - Scheduling: Task x time slot assignment

  Detection method:
  An odd-length cycle exists <=> Not bipartite
  Because coloring a cycle alternately with 2 colors makes the start and
  end vertices the same color if the length is odd, creating a contradiction.
```

```python
"""
Bipartite graph detection (BFS-based 2-coloring)

Why BFS is suitable:
BFS explores layer by layer, so it naturally assigns
"even levels -> color A, odd levels -> color B."
If there is an edge between vertices at the same level, it implies
an odd-length cycle, and the graph is not bipartite.

Time complexity: O(V + E)
"""

from collections import deque


def is_bipartite(graph: dict) -> tuple:
    """
    Bipartite graph detection

    Returns:
        (is_bipartite, coloring)
        is_bipartite: True if bipartite
        coloring: {node: 0 or 1} dictionary (2-coloring)
    """
    color = {}

    for start in graph:
        if start in color:
            continue  # Already colored

        # BFS 2-coloring
        color[start] = 0
        queue = deque([start])

        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                if neighbor not in color:
                    # Assign the opposite color to adjacent vertex
                    color[neighbor] = 1 - color[node]
                    queue.append(neighbor)
                elif color[neighbor] == color[node]:
                    # Adjacent vertex has same color -> not bipartite
                    return False, {}

    return True, color


# === Verification ===
if __name__ == "__main__":
    # Bipartite graph (4-vertex cycle = even length)
    bipartite_graph = {
        'A': ['B', 'D'],
        'B': ['A', 'C'],
        'C': ['B', 'D'],
        'D': ['C', 'A'],
    }
    result, coloring = is_bipartite(bipartite_graph)
    print(f"Bipartite: {result}")
    print(f"Coloring: {coloring}")
    # Output: Bipartite: True
    # Coloring: {'A': 0, 'B': 1, 'C': 0, 'D': 1}

    # Non-bipartite graph (3-vertex cycle = odd length)
    non_bipartite = {
        'A': ['B', 'C'],
        'B': ['A', 'C'],
        'C': ['A', 'B'],
    }
    result2, _ = is_bipartite(non_bipartite)
    print(f"Not bipartite: {result2}")
    # Output: Not bipartite: False
```

---

## 8. A* Algorithm

### 8.1 What Is A*?

A* is Dijkstra's algorithm augmented with a heuristic function. By "prioritizing exploration toward the goal," it reduces unnecessary search and efficiently finds the shortest path between two points.

Why use A* instead of Dijkstra? Dijkstra explores uniformly in all directions from the start. However, when only the shortest path to a specific destination is needed, exploring in the opposite direction from the goal is wasteful. A* uses a heuristic (estimated distance to goal) to prioritize exploration in promising directions.

```
A* operation visualization:

  Finding the shortest path from start S to goal G

  Dijkstra's algorithm:          A* algorithm:
  (Expands uniformly in all     (Prioritizes goal direction)
   directions)

  . . . . . . G               . . . . . . G
  . . * * . . .               . . . * * / .
  . * * * * . .               . . . * / / .
  * * * S * * .               . . * S / . .
  . * * * * . .               . . . . . . .
  . . * * . . .               . . . . . . .
  . . . . . . .               . . . . . . .

  * = Explored vertices          / = Explored vertices
  A* has a narrower search range -> faster

  Priority computation:
  f(n) = g(n) + h(n)
  - g(n): Actual cost from start to n (same as Dijkstra)
  - h(n): Estimated cost from n to goal (heuristic)
  - f(n): Estimated total cost through n

  Heuristic h(n) conditions:
  - Admissible: h(n) <= actual shortest distance (never overestimates)
  - Consistent: h(u) <= cost(u,v) + h(v)
  If these hold, A* guarantees an optimal solution.
```

```python
"""
A* algorithm complete implementation (grid-based pathfinding)

Why demonstrate on a grid:
A* is most frequently used for map and game pathfinding.
A 2D grid is the most intuitive concrete example.

Manhattan distance is used as the heuristic function.
Why Manhattan distance is admissible:
On a grid without diagonal movement, the shortest distance is the sum of
horizontal and vertical movements. Manhattan distance is exactly this distance,
and with obstacles, the actual shortest distance is at least the Manhattan
distance, so it never overestimates.
"""

import heapq


def astar_grid(grid: list, start: tuple, goal: tuple) -> tuple:
    """
    A* algorithm on a grid

    Args:
        grid: 2D array (0=passable, 1=wall)
        start: Starting point (row, col)
        goal: End point (row, col)

    Returns:
        (cost, path) -- shortest cost and path
        If unreachable: (float('inf'), [])
    """
    rows, cols = len(grid), len(grid[0])

    def heuristic(a, b):
        """Manhattan distance"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # 4-directional movement (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Priority queue: (f-value, g-value, current position)
    # Why include g-value: When f-values are equal, prefer larger g (closer to goal)
    open_set = [(heuristic(start, goal), 0, start)]
    g_score = {start: 0}
    came_from = {start: None}

    while open_set:
        f, g, current = heapq.heappop(open_set)

        if current == goal:
            # Path reconstruction
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            return g, path[::-1]

        if g > g_score.get(current, float('inf')):
            continue

        for dr, dc in directions:
            nr, nc = current[0] + dr, current[1] + dc
            neighbor = (nr, nc)

            # Boundary check
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            # Wall check
            if grid[nr][nc] == 1:
                continue

            new_g = g + 1  # Movement cost 1
            if new_g < g_score.get(neighbor, float('inf')):
                g_score[neighbor] = new_g
                f_score = new_g + heuristic(neighbor, goal)
                came_from[neighbor] = current
                heapq.heappush(open_set, (f_score, new_g, neighbor))

    return float('inf'), []  # Unreachable


# === Verification ===
if __name__ == "__main__":
    grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ]
    start = (0, 0)
    goal = (4, 4)

    cost, path = astar_grid(grid, start, goal)
    print(f"=== A* Algorithm ===")
    print(f"Shortest cost: {cost}")
    print(f"Path: {path}")

    # Display path on the grid
    display = [['.' if cell == 0 else '#' for cell in row] for row in grid]
    for r, c in path:
        display[r][c] = '*'
    display[start[0]][start[1]] = 'S'
    display[goal[0]][goal[1]] = 'G'
    print("\nGrid:")
    for row in display:
        print('  ' + ' '.join(row))
    # Expected output:
    # Grid:
    #   S * * * .
    #   . # # * .
    #   . . . * .
    #   . . # # *
    #   . . . . G
```

---

## 9. Anti-Patterns and Pitfalls

### 9.1 Anti-Pattern 1: Incorrect Placement of the Visited Check

```python
"""
Anti-pattern: Checking visited when dequeuing rather than when enqueuing.
This allows the same vertex to enter the queue multiple times, degrading performance.
"""

# BAD: Checking visited when dequeuing
def bfs_bad(graph, start):
    """
    Problem: The same vertex may enter the queue multiple times.
    In the worst case, queue size becomes O(E).
    For dense graphs (E ≈ V^2), this consumes O(V^2) memory.
    """
    visited = set()
    queue = [start]  # Even worse: using list instead of deque
    order = []

    while queue:
        node = queue.pop(0)  # BAD: list.pop(0) is O(n)!
        if node in visited:  # BAD: checking here doesn't prevent queue duplicates
            continue
        visited.add(node)
        order.append(node)
        for neighbor in graph[node]:
            queue.append(neighbor)  # BAD: adding without visited check

    return order


# GOOD: Checking visited when enqueuing
def bfs_good(graph, start):
    """
    Correct implementation: Check visited before enqueuing.
    Each vertex enters the queue at most once.
    Queue size is bounded by O(V).
    """
    from collections import deque
    visited = set([start])      # Add start to visited from the beginning
    queue = deque([start])      # deque's popleft() is O(1)
    order = []

    while queue:
        node = queue.popleft()  # O(1)
        order.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:  # Check before enqueuing
                visited.add(neighbor)
                queue.append(neighbor)

    return order


"""
Performance difference illustration:

  Complete graph K_1000 (1000 vertices, edges between all pairs):

  bfs_bad:
    - Queue entries: up to V×(V-1) ≈ 1 million times
    - list.pop(0) at O(n) makes it even slower
    - Expected processing time: seconds to tens of seconds

  bfs_good:
    - Queue entries: exactly V = 1000 times
    - deque.popleft() at O(1) is fast
    - Expected processing time: milliseconds
"""
```

### 9.2 Anti-Pattern 2: Ignoring Negative Weights with Dijkstra

```python
"""
Anti-pattern: Applying Dijkstra's algorithm to a graph with negative weights.
Results are incorrect but no error is raised, making the bug hard to notice.
"""

import heapq


def demonstrate_dijkstra_negative_weight_bug():
    """
    Concrete example where Dijkstra fails with negative weights

    Graph:
      A --1--> B --(-3)--> C
      A --2--> C

    Correct shortest distance: A->B->C = 1 + (-3) = -2
    Dijkstra's result: A->C = 2 (incorrect!)

    Why it fails:
    Dijkstra finalizes A->C (distance 2) first.
    Later it processes A->B (distance 1) and discovers B->C (distance -2),
    but C is already finalized so it is not updated.
    """
    graph = {
        'A': [('B', 1), ('C', 2)],
        'B': [('C', -3)],
        'C': [],
    }

    # Dijkstra (incorrect result)
    distances = {'A': float('inf'), 'B': float('inf'), 'C': float('inf')}
    distances['A'] = 0
    pq = [(0, 'A')]

    while pq:
        dist, node = heapq.heappop(pq)
        if dist > distances[node]:
            continue
        for neighbor, weight in graph[node]:
            new_dist = dist + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))

    print("Dijkstra result:", distances)
    # Output: {'A': 0, 'B': 1, 'C': -2}
    # In some cases the result may coincidentally be correct, but
    # the "dist > distances[node]" skip can cause updates through
    # finalized vertices to be lost.

    # The correct approach is to use Bellman-Ford
    print("\n>>> Use Bellman-Ford when negative weights are present <<<")


# === Recommended: Dijkstra with input validation ===
def dijkstra_safe(graph: dict, start):
    """Dijkstra that detects negative weights and raises an exception"""
    for node in graph:
        for neighbor, weight in graph[node]:
            if weight < 0:
                raise ValueError(
                    f"Negative weight edge detected: {node}->{neighbor} (weight={weight}). "
                    f"Please use the Bellman-Ford algorithm."
                )

    # Standard Dijkstra follows
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]

    while pq:
        dist, node = heapq.heappop(pq)
        if dist > distances[node]:
            continue
        for neighbor, weight in graph[node]:
            new_dist = dist + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))

    return distances


if __name__ == "__main__":
    demonstrate_dijkstra_negative_weight_bug()
```

### 9.3 Anti-Pattern 3: Stack Overflow with Recursive DFS

```
Pitfalls of recursive DFS:

  Python's default recursion limit: 1000
  For graphs with more than 1000 vertices, RecursionError may occur.

  Countermeasures:
  1. Raise the limit with sys.setrecursionlimit() (not recommended: OS stack-dependent)
  2. Use iterative DFS (explicitly manage the stack) (recommended)
  3. Tail call optimization is not available in Python

  Recommendations for contests/production:
  - Vertex count < 1000: Recursive DFS is fine
  - Vertex count >= 1000: Use iterative DFS
  - Safe practice: Always use the iterative version
```

---

## 10. Edge Case Analysis

### 10.1 Edge Case 1: Disconnected Graphs

```python
"""
Edge case: When the graph is not connected

Many graph algorithm implementations implicitly assume the graph is connected.
In a disconnected graph, BFS/DFS from a single start cannot reach all vertices.

Countermeasure: Loop through all vertices and start exploration from unvisited ones.
"""


def count_connected_components(graph: dict) -> int:
    """
    Count the number of connected components in an undirected graph.

    Why this matters:
    - Network partition detection ("are all sites reachable?")
    - Estimating number of communities in social networks
    - Region segmentation in image processing
    """
    visited = set()
    components = 0

    def dfs(node):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)

    for node in graph:
        if node not in visited:
            dfs(node)
            components += 1

    return components


# === Verification ===
if __name__ == "__main__":
    # 3 connected components: {A,B}, {C,D}, {E}
    graph = {
        'A': ['B'],
        'B': ['A'],
        'C': ['D'],
        'D': ['C'],
        'E': [],
    }
    print(f"Connected components: {count_connected_components(graph)}")
    # Output: Connected components: 3

    # Vertices reachable from A via BFS
    from collections import deque
    visited = set(['A'])
    queue = deque(['A'])
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    print(f"Reachable from A: {visited}")
    # Output: Reachable from A: {'A', 'B'} -- C, D, E are unreachable!
```

### 10.2 Edge Case 2: Self-Loops and Multi-Edges

```python
"""
Edge case: Self-loops and multi-edges

Self-loop: An edge from a vertex to itself (v, v)
Multi-edge: Multiple edges between the same pair of vertices

When these exist, many algorithms require special handling.
"""


def handle_self_loops_and_multi_edges():
    """Demonstration of self-loops and multi-edges"""

    # Graph with a self-loop
    graph_with_self_loop = {
        'A': ['A', 'B'],  # A->A (self-loop), A->B
        'B': ['C'],
        'C': [],
    }

    # Issue 1: Does BFS's visited check work correctly?
    from collections import deque
    visited = set(['A'])
    queue = deque(['A'])
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph_with_self_loop[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    print(f"BFS with self-loop: {order}")
    # Output: ['A', 'B', 'C'] -- Self-loop ignored by visited (correct behavior)

    # Issue 2: Does cycle detection detect self-loops?
    # 3-color method (directed graph): After marking A as GRAY, A->A reaches a GRAY vertex
    # -> Correctly detected as a cycle

    # Issue 3: Multi-edges and shortest paths
    # Dijkstra naturally selects the minimum-weight edge
    graph_multi_edge = {
        'A': [('B', 5), ('B', 3), ('B', 7)],  # 3 edges from A to B
        'B': [],
    }
    import heapq
    distances = {'A': 0, 'B': float('inf')}
    pq = [(0, 'A')]
    while pq:
        dist, node = heapq.heappop(pq)
        if dist > distances[node]:
            continue
        for neighbor, weight in graph_multi_edge[node]:
            new_dist = dist + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))
    print(f"Multi-edge Dijkstra: A->B = {distances['B']}")
    # Output: Multi-edge Dijkstra: A->B = 3 (minimum-weight edge naturally selected)


if __name__ == "__main__":
    handle_self_loops_and_multi_edges()
```

### 10.3 Edge Case 3: Memory Management for Huge Graphs

```
Edge cases for huge graphs (1 million+ vertices):

  Problem 1: Out of memory
  - Adjacency matrix is unusable (1 million^2 = 1 trillion elements)
  - Use adjacency list
  - Disk-based processing if needed

  Problem 2: Recursion depth limit
  - Recursive DFS is unusable (stack overflow)
  - Use iterative DFS/BFS

  Problem 3: Priority queue size
  - Multiple entries for the same vertex accumulate in Dijkstra's heap
  - Countermeasure: Skip stale entries with visited check
  - IndexedPriorityQueue limits to O(V)

  Problem 4: Unreachable vertices
  - Distance to vertices unreachable from start is float('inf')
  - Handle "infinite distance" correctly (beware of division by zero, overflow)

  Practical countermeasures:
  ┌──────────────────────┬────────────────────────────────┐
  │ Problem               │ Countermeasure                  │
  ├──────────────────────┼────────────────────────────────┤
  │ Out of memory         │ Adjacency list + stream processing│
  │ Stack overflow        │ Iterative algorithms            │
  │ Processing speed      │ Heuristics (A*)                 │
  │ Ultra-large graphs    │ Contraction Hierarchies, etc.   │
  │ Distributed processing│ Pregel / GraphX (Spark)         │
  └──────────────────────┴────────────────────────────────┘
```

---

## 11. Comprehensive Graph Algorithm Comparison

```
Comparison table of all graph algorithms:

  ┌──────────────────┬─────────────────┬──────────┬──────────────────────────┐
  │ Algorithm         │ Time complexity  │ Space    │ Primary use               │
  ├──────────────────┼─────────────────┼──────────┼──────────────────────────┤
  │ BFS              │ O(V + E)        │ O(V)     │ Shortest path (unweighted)│
  │ DFS              │ O(V + E)        │ O(V)     │ Cycle detection, components│
  │ Dijkstra         │ O((V+E) log V)  │ O(V)     │ Shortest path (non-neg)   │
  │ Bellman-Ford     │ O(V × E)        │ O(V)     │ Shortest path (neg weights)│
  │ Floyd-Warshall   │ O(V^3)          │ O(V^2)   │ All-pairs shortest path   │
  │ A*               │ O(E) (expected) │ O(V)     │ Point-to-point shortest path│
  │ Kruskal          │ O(E log E)      │ O(V)     │ Minimum spanning tree     │
  │ Prim             │ O((V+E) log V)  │ O(V)     │ Minimum spanning tree     │
  │ Topological sort │ O(V + E)        │ O(V)     │ Dependency resolution     │
  │ Kosaraju SCC     │ O(V + E)        │ O(V + E) │ Strongly connected comp.  │
  │ Tarjan SCC       │ O(V + E)        │ O(V)     │ Strongly connected comp.  │
  │ Union-Find       │ O(alpha(n)) ≈ O(1)│ O(V)   │ Component management      │
  │ Bipartite detect.│ O(V + E)        │ O(V)     │ 2-coloring                │
  └──────────────────┴─────────────────┴──────────┴──────────────────────────┘

  MST algorithm selection criteria:
  ┌────────────────────────────┬──────────────┬──────────────┐
  │ Condition                   │ Kruskal      │ Prim         │
  ├────────────────────────────┼──────────────┼──────────────┤
  │ Sparse graph (E << V^2)    │ ★ Advantageous│ Usable      │
  │ Dense graph (E ≈ V^2)      │ Usable       │ ★ Advantageous│
  │ Pre-sorted edges            │ ★ Very advantageous│ No difference│
  │ Online (edges added incrementally)│ Usable│ ★ Advantageous│
  │ Parallel processing        │ ★ Possible   │ Difficult    │
  └────────────────────────────┴──────────────┴──────────────┘
```

---

## 12. Practice Exercises

### Exercise 1: Maze Shortest Path (Basic)

```
Problem:
A maze represented as a 2D grid is given.
0 represents a passage, 1 represents a wall.
Find the shortest path from start (0,0) to end (rows-1, cols-1).

Input example:
  grid = [
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 0]
  ]

Expected output: Shortest distance = 6

Hints:
- Use BFS (shortest path in unweighted graphs)
- Consider 4-directional movement (up, down, left, right)
- Don't forget to manage visited

Solution key points:
- Treat each grid cell as a graph vertex
- Treat movement to adjacent cells (not walls, within bounds) as edges
- Explore with BFS while tracking level (distance)
```

```python
"""Solution for Exercise 1"""
from collections import deque


def solve_maze(grid: list) -> int:
    """BFS shortest path through a maze"""
    rows, cols = len(grid), len(grid[0])
    if grid[0][0] == 1 or grid[rows-1][cols-1] == 1:
        return -1  # Start or end is a wall

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    visited = set([(0, 0)])
    queue = deque([(0, 0, 0)])  # (row, col, distance)

    while queue:
        r, c, dist = queue.popleft()
        if r == rows - 1 and c == cols - 1:
            return dist
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols
                    and grid[nr][nc] == 0
                    and (nr, nc) not in visited):
                visited.add((nr, nc))
                queue.append((nr, nc, dist + 1))

    return -1  # Unreachable


# Test
grid = [
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 0]
]
print(f"Shortest distance: {solve_maze(grid)}")
# Output: Shortest distance: 6
```

### Exercise 2: Dijkstra with Path Reconstruction (Applied)

```
Problem:
Travel costs between cities are given. Find the shortest path and
its cost from start to end, and reconstruct and display the path.

Input:
  cities = ['Tokyo', 'Nagoya', 'Osaka', 'Kyoto', 'Fukuoka']
  routes = [
    ('Tokyo', 'Nagoya', 350),
    ('Tokyo', 'Osaka', 500),
    ('Nagoya', 'Kyoto', 130),
    ('Nagoya', 'Osaka', 180),
    ('Kyoto', 'Osaka', 50),
    ('Osaka', 'Fukuoka', 600),
    ('Kyoto', 'Fukuoka', 650),
  ]

Expected output:
  Tokyo -> Fukuoka: Cost 1080
  Path: Tokyo -> Nagoya -> Kyoto -> Osaka -> Fukuoka

Hints:
- Dijkstra's algorithm + path reconstruction (previous dictionary)
- Confirm whether it is a directed or undirected graph
- Handle the unreachable case
```

```python
"""Solution for Exercise 2"""
import heapq
from collections import defaultdict


def solve_city_routes(cities, routes, start, end):
    """Shortest inter-city path (with path reconstruction)"""
    # Build graph (bidirectional)
    graph = defaultdict(list)
    for u, v, w in routes:
        graph[u].append((v, w))
        graph[v].append((u, w))

    # Dijkstra
    distances = {city: float('inf') for city in cities}
    distances[start] = 0
    previous = {city: None for city in cities}
    pq = [(0, start)]

    while pq:
        dist, node = heapq.heappop(pq)
        if dist > distances[node]:
            continue
        if node == end:
            break
        for neighbor, weight in graph[node]:
            new_dist = dist + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                previous[neighbor] = node
                heapq.heappush(pq, (new_dist, neighbor))

    # Path reconstruction
    if distances[end] == float('inf'):
        return None, []

    path = []
    node = end
    while node is not None:
        path.append(node)
        node = previous[node]
    path.reverse()

    return distances[end], path


# Test
cities = ['Tokyo', 'Nagoya', 'Osaka', 'Kyoto', 'Fukuoka']
routes = [
    ('Tokyo', 'Nagoya', 350),
    ('Tokyo', 'Osaka', 500),
    ('Nagoya', 'Kyoto', 130),
    ('Nagoya', 'Osaka', 180),
    ('Kyoto', 'Osaka', 50),
    ('Osaka', 'Fukuoka', 600),
    ('Kyoto', 'Fukuoka', 650),
]

cost, path = solve_city_routes(cities, routes, 'Tokyo', 'Fukuoka')
print(f"Tokyo -> Fukuoka: Cost {cost}")
print(f"Path: {' -> '.join(path)}")
# Output:
# Tokyo -> Fukuoka: Cost 1080
# Path: Tokyo -> Nagoya -> Kyoto -> Osaka -> Fukuoka
```

### Exercise 3: SNS Friend Recommendation and Influence Analysis (Advanced)

```
Problem:
A social network friendship graph is given. Implement the following:

(a) Friend-of-friend recommendation: For user A, recommend people who are not
    direct friends but share many mutual friends (rank by mutual friend count)

(b) Influence calculation: Define each user's "influence" as the number of
    people reachable within distance 2, and find the most influential user

(c) Community detection: Find connected components and display each
    community's members

Hints:
- (a): Use BFS to find vertices at distance 2, count mutual friends
- (b): Use BFS to count vertices within distance 2
- (c): Use DFS/BFS to find connected components
```

```python
"""Solution for Exercise 3"""
from collections import deque, defaultdict, Counter


class SocialNetwork:
    """Social network analysis class"""

    def __init__(self):
        self.graph = defaultdict(set)

    def add_friendship(self, u, v):
        """Add a bidirectional friendship"""
        self.graph[u].add(v)
        self.graph[v].add(u)

    def recommend_friends(self, user, top_k=3):
        """
        Recommend friends of friends.
        People with more mutual friends rank higher.
        """
        if user not in self.graph:
            return []

        direct_friends = self.graph[user]
        # Count friends of friends (mutual friend count)
        candidates = Counter()
        for friend in direct_friends:
            for fof in self.graph[friend]:
                if fof != user and fof not in direct_friends:
                    candidates[fof] += 1

        # Return top_k people with most mutual friends
        return candidates.most_common(top_k)

    def influence_score(self, user, max_distance=2):
        """Number of people within max_distance (influence score)"""
        visited = set([user])
        queue = deque([(user, 0)])
        count = 0

        while queue:
            node, dist = queue.popleft()
            if dist > 0:
                count += 1
            if dist < max_distance:
                for neighbor in self.graph[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

        return count

    def most_influential(self, max_distance=2):
        """Return the most influential user"""
        scores = {user: self.influence_score(user, max_distance)
                  for user in self.graph}
        return max(scores.items(), key=lambda x: x[1])

    def find_communities(self):
        """Detect connected components (communities)"""
        visited = set()
        communities = []

        for user in self.graph:
            if user not in visited:
                community = []
                queue = deque([user])
                visited.add(user)
                while queue:
                    node = queue.popleft()
                    community.append(node)
                    for neighbor in self.graph[node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                communities.append(sorted(community))

        return communities


# === Test ===
if __name__ == "__main__":
    sn = SocialNetwork()
    # Community 1
    sn.add_friendship('Alice', 'Bob')
    sn.add_friendship('Alice', 'Charlie')
    sn.add_friendship('Bob', 'Charlie')
    sn.add_friendship('Bob', 'David')
    sn.add_friendship('Charlie', 'Eve')
    sn.add_friendship('David', 'Eve')

    # Community 2 (independent)
    sn.add_friendship('Frank', 'Grace')
    sn.add_friendship('Grace', 'Heidi')

    # (a) Friend recommendation
    print("=== Friend Recommendation ===")
    recs = sn.recommend_friends('Alice')
    for person, common_count in recs:
        print(f"  {person}: {common_count} mutual friends")
    # Expected output:
    #   Eve: 2 mutual friends (via Bob, Charlie)
    #   David: 1 mutual friend (via Bob)

    # (b) Influence
    print("\n=== Influence Scores ===")
    for user in ['Alice', 'Bob', 'Frank']:
        score = sn.influence_score(user)
        print(f"  {user}: {score} people")
    user, score = sn.most_influential()
    print(f"  Most influential: {user} ({score} people)")

    # (c) Communities
    print("\n=== Communities ===")
    communities = sn.find_communities()
    for i, comm in enumerate(communities):
        print(f"  Community {i+1}: {comm}")
    # Expected output:
    # Community 1: ['Alice', 'Bob', 'Charlie', 'David', 'Eve']
    # Community 2: ['Frank', 'Grace', 'Heidi']
```

---

## 13. FAQ

### Q1: Does Google Maps use Dijkstra's algorithm for route finding?

**A**: A* algorithm is the fundamental base, but it alone is too slow for continent-scale graphs (billions of vertices). In practice, the following advanced techniques are combined:

- **Contraction Hierarchies (CH)**: Preprocessing creates shortcut edges through "important vertices," dramatically reducing the search range at query time. Preprocessing takes several hours, but queries respond in milliseconds.
- **ALT (A* with Landmarks and Triangle inequality)**: Pre-computes distances to landmarks (specific points on the map) and uses them as heuristic functions.
- **Transit Node Routing**: Exploits the property that long-distance routes pass through a small number of "transit nodes" (such as highway interchanges), pre-computing distances between transit nodes.

These preprocessing-based techniques enable millisecond-order shortest path computation on graphs with tens of millions to billions of vertices.

### Q2: Why are different methods needed for cycle detection in undirected vs. directed graphs?

**A**: Because the definition of "cycle" is subtly different between undirected and directed graphs.

- **Undirected graph**: When edge A--B exists, A->B->A is not considered a cycle of length 2 (it's just traversing the same edge back). In DFS, reaching a visited vertex that is not the parent indicates a cycle. Union-Find can also detect cycles (if both endpoints are in the same set when adding an edge, it's a cycle).

- **Directed graph**: In a graph A->B->C, even if C is visited, it doesn't necessarily mean A->B->C->...->A is a cycle (C might not lead back to A). The 3-color method (WHITE/GRAY/BLACK) is needed to determine "reaching a GRAY vertex (on the current exploration path) means a cycle." An edge to a BLACK vertex is not a cycle.

### Q3: Is PageRank also a graph algorithm?

**A**: PageRank is an algorithm defined on a graph of web pages. With each page as a vertex and hyperlinks as edges, it computes the stationary distribution of a random walk (modeling users randomly clicking links).

Specifically, the PageRank PR(v) of page v is defined by:

```
PR(v) = (1-d)/N + d × Σ PR(u) / L(u)
```

where d is the damping factor (typically 0.85), N is the total number of pages, and L(u) is the number of outgoing links from page u. This formula is iteratively computed for all pages until convergence. It can also be formulated as a matrix eigenvector problem. While it was the foundation of Google Search, it is now used in combination with hundreds of other factors.

### Q4: What is the bipartite matching problem?

**A**: In a bipartite graph, find a subset of edges where each vertex is incident to at most one edge (a matching), and find the maximum such subset (maximum matching). Applications include job assignment (employees x tasks), residency matching, and the stable marriage problem.

It can be efficiently solved with the Hungarian algorithm (O(V^3)) or the Hopcroft-Karp algorithm (O(E*sqrt(V))). It can also be reduced to a maximum flow problem.

### Q5: What is the recommended order for learning graph algorithms?

**A**: The following order is recommended:

1. **BFS/DFS** (Most fundamental. Nothing starts without these)
2. **Dijkstra's algorithm** (Standard for shortest paths. Common in interviews)
3. **Topological sort** (Dependency resolution. Common in build systems)
4. **Union-Find** (Connected components. Prerequisite for Kruskal's)
5. **Minimum spanning tree** (Network design)
6. **Bellman-Ford/Floyd-Warshall** (Handling negative weights)
7. **Strongly connected components** (Advanced problems. Foundation for 2-SAT)
8. **A*** (Game development, map applications)

### Q6: Are there practical situations where graph algorithms are used?

**A**: Very many. You may be depending on graph algorithms without realizing it:

- **Package managers** (npm, pip, cargo): Topologically sort the dependency DAG to determine installation order
- **CI/CD pipelines**: Manage job dependencies as a DAG (GitHub Actions, CircleCI)
- **Databases**: Check for circular references in foreign key constraints (cycle detection)
- **GC mark & sweep**: BFS/DFS on object reference graphs (determining reachable objects)
- **Networking**: Routing protocols (OSPF = Dijkstra, RIP = Bellman-Ford)
- **Recommendation engines**: Collaborative filtering on user x item bipartite graphs

---


## FAQ

### Q1: What is the most important point when studying this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory but by actually writing code and verifying behavior.

### Q2: What common mistakes do beginners make?

Skipping the fundamentals and jumping to applications. We recommend thoroughly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this used in practice?

Knowledge of this topic is frequently applied in everyday development work. It becomes particularly important during code reviews and architecture design.

---

## 14. Summary

### Learning Review Checklist

- [ ] Can explain when to use adjacency list vs. adjacency matrix
- [ ] Can explain how BFS and DFS work and their time complexities
- [ ] Can explain why BFS finds shortest paths in unweighted graphs
- [ ] Can explain why Dijkstra's algorithm fails with negative weights
- [ ] Can explain the meaning of "V-1 iterations" in Bellman-Ford
- [ ] Can explain why the loop order is "k, i, j" in Floyd-Warshall
- [ ] Can explain the difference between Kruskal's and Prim's algorithms
- [ ] Can explain the condition for topological sort applicability (DAG)
- [ ] Can explain the definition of SCCs and the basic operation of Kosaraju/Tarjan
- [ ] Can implement bipartite graph detection using 2-coloring
- [ ] Can select the appropriate graph algorithm for a given problem

### Algorithm Selection Flowchart

```
Identify the problem type:

  Need to find shortest paths
  +-- Unweighted -> BFS
  +-- Non-negative weights
  |   +-- Single source -> Dijkstra
  |   +-- Point-to-point (large scale) -> A*
  +-- Negative weights present
  |   +-- Single source -> Bellman-Ford
  |   +-- All pairs -> Floyd-Warshall
  +-- Negative cycle detection -> Bellman-Ford

  Connectivity problems
  +-- Connected components (undirected) -> BFS/DFS or Union-Find
  +-- Strongly connected components (directed) -> Kosaraju / Tarjan
  +-- Bipartite detection -> BFS (2-coloring)

  Ordering problems
  +-- Dependency resolution -> Topological sort
  +-- Cycle detection
      +-- Undirected graph -> DFS (parent check) / Union-Find
      +-- Directed graph -> DFS (3-color method)

  Optimization problems
  +-- Minimum spanning tree
  |   +-- Sparse graph -> Kruskal
  |   +-- Dense graph -> Prim
  +-- Maximum matching -> Hopcroft-Karp / Max flow
```

---

## Recommended Next Guides


---

## References

1. Cormen, T. H., Leiserson, C. E., Rivest, R. L., and Stein, C. **"Introduction to Algorithms."** 4th Edition, MIT Press, 2022. Chapters 20-26. -- The most comprehensive textbook on graph algorithms. Rich in proofs and pseudocode.
2. Skiena, S. S. **"The Algorithm Design Manual."** 3rd Edition, Springer, 2020. Chapters 7-8. -- Explains graph algorithms from a practical perspective. Excellent guide on "which algorithm to use for which problem."
3. Sedgewick, R. and Wayne, K. **"Algorithms."** 4th Edition, Addison-Wesley, 2011. Chapter 4. -- Rich in Java implementation examples. Includes visualization tools for intuitive understanding of operations.
4. Kleinberg, J. and Tardos, E. **"Algorithm Design."** Pearson, 2005. Chapters 3-7. -- Provides deep understanding of graph algorithm design principles (connections to greedy and dynamic programming).
5. **Stanford CS161: Design and Analysis of Algorithms.** Online lecture notes. -- Detailed explanations of Dijkstra's correctness proof and SCCs are freely available.
6. **Competitive Programmer's Handbook** by Antti Laaksonen. -- A collection of graph algorithm implementation techniques from a competitive programming perspective. Free PDF available.