# Network Flow

> Understand maximum flow problems on graphs through Ford-Fulkerson, Dinic's algorithm, bipartite matching, and minimum cost flow, and master practical application patterns

## What You Will Learn in This Chapter

1. **The definition of the maximum flow problem** and the concepts of residual graphs and augmenting paths
2. **Ford-Fulkerson method** and the Edmonds-Karp algorithm with BFS, implemented correctly
3. **Dinic's algorithm** for computing maximum flow efficiently
4. **Bipartite matching** reduced to maximum flow, solving job assignment and matching problems
5. **Max-flow min-cut theorem and minimum cost flow** -- their theory and applications


## Prerequisites

The following knowledge will help deepen your understanding before reading this guide:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding of [String Algorithms](./02-string-algorithms.md)

---

## 1. Fundamental Concepts of Network Flow

Network flow problems are formulated as finding the maximum amount of "stuff" that can flow through pipelines or communication networks.

```
Flow network:
  - Directed graph G = (V, E)
  - Capacity function c(u,v) >= 0 (maximum flow on each edge)
  - Source s, sink t

Constraints:
  1. Capacity constraint: 0 <= f(u,v) <= c(u,v)
  2. Flow conservation: inflow = outflow at each vertex (except s, t)
  3. Skew symmetry: f(u,v) = -f(v,u)

Example:
         10        10
    s -----> A -----> t
    |       ^       ^
    |5      |15     |10
    v       |       |
    B -----> C -----> D
         10        10

Maximum flow = 19 (s->A->t:10, s->B->C->D->t:5, s->B->C->A->t:0, ...)
```

### Real-world Examples of Flow Problems

```
1. Logistics network
   Factory(s) -> Warehouse -> Distribution center -> Store(t)
   Edge capacity = truck transport capacity
   Maximum flow = maximum transport volume

2. Communication network
   Sender(s) -> Routers -> Receiver(t)
   Edge capacity = bandwidth
   Maximum flow = maximum data transfer rate

3. Water pipeline network
   Water source(s) -> Pipes -> Households(t)
   Edge capacity = pipe diameter
   Maximum flow = maximum water distribution

4. Scheduling
   Start(s) -> Workers -> Tasks -> End(t)
   Capacity = processing capacity of each worker
   Maximum flow = maximum throughput
```

---

## 2. Residual Graphs and Augmenting Paths

```
Original graph:              Current flow:
    s --(10)--> A           s --7/10--> A
    |           |           |           |
   (5)       (10)         5/5        7/10
    v           v           v           v
    B --(10)--> t           B --5/10--> t

Residual graph:
  - Forward direction: remaining capacity = c(u,v) - f(u,v)
  - Backward direction: cancellable amount = f(u,v)

    s --(3)--> A           remaining capacity
    | <-(7)-- A             backward (cancel)
    |           |
   (0)->      (3)->
   <-(5)      <-(7)
    v           v
    B --(5)--> t
    B <-(5)-- t

Augmenting path = s->t path in the residual graph
Bottleneck = minimum residual capacity on the path
```

### Importance of the Residual Graph

```
Why are backward edges (cancellation) necessary?

Example:
    s --(1)--> A --(1)--> t
    |                       ^
   (1)                    (1)
    v                       |
    B -------(1)---------> C

Optimal flow:
  s->A->t: 1
  s->B->C->t: 1
  Total: 2

Without backward edges, a greedy approach:
  s->A->C->t would not work (no direct edge from C to t in this case)
  -> Backward edges allow "undoing" flow that was already sent
  -> Makes it possible to reach the optimal solution
```

---

## 3. Ford-Fulkerson Method

Repeatedly finds augmenting paths in the residual graph and sends flow along them.

```python
from collections import defaultdict, deque

class MaxFlow:
    """Ford-Fulkerson method (Edmonds-Karp: augmenting path search via BFS)"""

    def __init__(self, n: int):
        self.n = n
        self.graph = defaultdict(lambda: defaultdict(int))  # capacity

    def add_edge(self, u: int, v: int, cap: int):
        """Add a directed edge"""
        self.graph[u][v] += cap

    def bfs(self, source: int, sink: int, parent: dict) -> int:
        """Find an augmenting path via BFS and return the bottleneck capacity"""
        visited = {source}
        queue = deque([(source, float('inf'))])

        while queue:
            u, flow = queue.popleft()
            for v in self.graph[u]:
                if v not in visited and self.graph[u][v] > 0:
                    visited.add(v)
                    parent[v] = u
                    new_flow = min(flow, self.graph[u][v])
                    if v == sink:
                        return new_flow
                    queue.append((v, new_flow))

        return 0

    def max_flow(self, source: int, sink: int) -> int:
        """Compute maximum flow - O(VE^2)"""
        total_flow = 0

        while True:
            parent = {}
            path_flow = self.bfs(source, sink, parent)

            if path_flow == 0:
                break  # no augmenting path -> maximum flow reached

            total_flow += path_flow

            # Update flow (update residual graph)
            v = sink
            while v != source:
                u = parent[v]
                self.graph[u][v] -= path_flow  # forward: reduce capacity
                self.graph[v][u] += path_flow  # backward: increase cancellable amount
                v = u

        return total_flow

# Usage example
mf = MaxFlow(6)
# s=0, A=1, B=2, C=3, D=4, t=5
mf.add_edge(0, 1, 10)  # s -> A
mf.add_edge(0, 2, 10)  # s -> B
mf.add_edge(1, 3, 4)   # A -> C
mf.add_edge(1, 4, 8)   # A -> D
mf.add_edge(2, 4, 9)   # B -> D
mf.add_edge(3, 5, 10)  # C -> t
mf.add_edge(4, 3, 6)   # D -> C
mf.add_edge(4, 5, 10)  # D -> t

print(mf.max_flow(0, 5))  # 19
```

### Ford-Fulkerson Execution Trace

```
Initial state:
  s -> A: 10, s -> B: 10
  A -> C: 4,  A -> D: 8
  B -> D: 9
  C -> t: 10, D -> C: 6, D -> t: 10

Iteration 1: BFS finds s->A->D->t (bottleneck = min(10,8,10) = 8)
  After update: s->A: 2, A->D: 0, D->t: 2

Iteration 2: BFS finds s->A->C->t (bottleneck = min(2,4,10) = 2)
  After update: s->A: 0, A->C: 2, C->t: 8

Iteration 3: BFS finds s->B->D->C->t (bottleneck = min(10,9,6,8) = 6)
  After update: s->B: 4, B->D: 3, D->C: 0, C->t: 2

Iteration 4: BFS finds s->B->D->t (bottleneck = min(4,3,2) = 2)
  After update: s->B: 2, B->D: 1, D->t: 0

Iteration 5: No augmenting path -> done

Maximum flow = 8 + 2 + 6 + 2 + 1 = 19
```

---

## 4. Dinic's Algorithm (Faster Version)

Finds blocking flows on a level graph (built via BFS). Faster than Edmonds-Karp: O(V^2 E) for integer capacities, O(E sqrt(V)) for unit capacities.

```python
class Dinic:
    """Dinic's algorithm - O(V^2 E)
    O(E sqrt(V)) for bipartite matching
    """

    def __init__(self, n: int):
        self.n = n
        self.graph = [[] for _ in range(n)]

    def add_edge(self, u: int, v: int, cap: int):
        """Add an edge (reverse edge is also added)"""
        self.graph[u].append([v, cap, len(self.graph[v])])
        self.graph[v].append([u, 0, len(self.graph[u]) - 1])

    def bfs(self, s: int, t: int) -> bool:
        """Build the level graph"""
        self.level = [-1] * self.n
        self.level[s] = 0
        queue = deque([s])

        while queue:
            u = queue.popleft()
            for v, cap, _ in self.graph[u]:
                if cap > 0 and self.level[v] < 0:
                    self.level[v] = self.level[u] + 1
                    queue.append(v)

        return self.level[t] >= 0

    def dfs(self, u: int, t: int, f: int) -> int:
        """Find a blocking flow"""
        if u == t:
            return f
        while self.iter[u] < len(self.graph[u]):
            v, cap, rev = self.graph[u][self.iter[u]]
            if cap > 0 and self.level[v] == self.level[u] + 1:
                d = self.dfs(v, t, min(f, cap))
                if d > 0:
                    self.graph[u][self.iter[u]][1] -= d
                    self.graph[v][rev][1] += d
                    return d
            self.iter[u] += 1
        return 0

    def max_flow(self, s: int, t: int) -> int:
        """Compute maximum flow"""
        flow = 0
        while self.bfs(s, t):
            self.iter = [0] * self.n
            while True:
                f = self.dfs(s, t, float('inf'))
                if f == 0:
                    break
                flow += f
        return flow

    def min_cut(self, s: int) -> list:
        """Find the minimum cut (call after max_flow)
        Returns: list of vertices on the s-side
        """
        visited = [False] * self.n
        queue = deque([s])
        visited[s] = True
        while queue:
            u = queue.popleft()
            for v, cap, _ in self.graph[u]:
                if cap > 0 and not visited[v]:
                    visited[v] = True
                    queue.append(v)
        return [i for i in range(self.n) if visited[i]]

# Usage example
dinic = Dinic(6)
dinic.add_edge(0, 1, 10)
dinic.add_edge(0, 2, 10)
dinic.add_edge(1, 3, 4)
dinic.add_edge(1, 4, 8)
dinic.add_edge(2, 4, 9)
dinic.add_edge(3, 5, 10)
dinic.add_edge(4, 3, 6)
dinic.add_edge(4, 5, 10)
print(dinic.max_flow(0, 5))  # 19
```

### Dinic's Algorithm vs Edmonds-Karp

```
Edmonds-Karp:
  - BFS finds one shortest augmenting path
  - Sends flow along it
  - BFS again -> ... repeat
  - Complexity: O(VE^2)

Dinic:
  - BFS builds the level graph
  - DFS processes multiple augmenting paths at once on the level graph (blocking flow)
  - Repeats DFS until the level graph changes
  - BFS again -> ... repeat
  - Complexity: O(V^2 E)

  Key points:
  - BFS runs at most V-1 times (level increases by at least 1 each time)
  - Each BFS phase: blocking flow via DFS is O(VE)
  - The iter array avoids re-exploring the same edges -> DFS efficiency
```

---

## 5. Max-Flow Min-Cut Theorem

```
Maximum flow = Minimum cut

Minimum cut = the minimum total capacity of edges that
              separate the network into s-side and t-side

Example:
    s --(3)--> A --(2)--> t
    |                      ^
   (5)                   (4)
    v                      |
    B ---------(6)-------> C

Maximum flow = 7
Minimum cut: {(A,t): 2, (s,B): 5} = 7

-> After max flow, find the set of vertices reachable from s in the residual graph (set S)
   and unreachable vertices (set T). Edges from S to T form the minimum cut.
```

```python
def find_min_cut_edges(n: int, edges: list, source: int, sink: int) -> list:
    """Find the edges of the minimum cut"""
    dinic = Dinic(n)
    original_edges = []

    for u, v, cap in edges:
        edge_idx = len(dinic.graph[u])
        dinic.add_edge(u, v, cap)
        original_edges.append((u, v, cap, edge_idx))

    max_flow_value = dinic.max_flow(source, sink)

    # Identify s-side vertices
    s_side = set(dinic.min_cut(source))

    # Edges from S to T are cut edges
    cut_edges = []
    for u, v, cap, _ in original_edges:
        if u in s_side and v not in s_side:
            cut_edges.append((u, v, cap))

    return max_flow_value, cut_edges

edges = [(0, 1, 3), (0, 2, 5), (1, 3, 2), (2, 3, 6), (2, 1, 4)]
# s=0, A=1, B=2, t=3
flow, cuts = find_min_cut_edges(4, edges, 0, 3)
print(f"Maximum flow: {flow}")
print(f"Minimum cut edges: {cuts}")
```

### Outline of the Max-Flow Min-Cut Theorem Proof

```
Theorem: In any flow network,
         maximum flow value = minimum cut capacity

Proof sketch:
1. For any flow f and any cut (S, T),
   |f| <= c(S, T) (flow value <= cut capacity)

2. When Ford-Fulkerson terminates,
   there is no s->t path in the residual graph

3. Let S be the set of vertices reachable from s, and T be the rest.
   All edges from S to T are saturated (f(u,v) = c(u,v))
   All edges from T to S have zero flow (f(v,u) = 0)

4. Therefore |f| = c(S, T) = minimum cut capacity
```

---

## 6. Bipartite Matching

Reducing maximum matching in a bipartite graph to maximum flow.

```
Bipartite graph:                Flow network construction:
  Students <-> Projects          s -> Students -> Projects -> t
                                 All capacities = 1

  A -- P1                    s --> A --> P1 --> t
  A -- P2                    s --> A --> P2 --> t
  B -- P1                    s --> B --> P1 --> t
  B -- P3                    s --> B --> P3 --> t
  C -- P2                    s --> C --> P2 --> t
  C -- P3                    s --> C --> P3 --> t

  Maximum matching = Maximum flow = 3
  Example: A-P1, B-P3, C-P2
```

```python
def bipartite_matching(left: int, right: int, edges: list) -> tuple:
    """Bipartite matching (max-flow based)
    left: number of left vertices, right: number of right vertices
    edges: [(left vertex, right vertex), ...]
    """
    n = left + right + 2
    source = 0
    sink = n - 1

    dinic = Dinic(n)

    # source -> left vertices (capacity 1)
    for i in range(left):
        dinic.add_edge(source, i + 1, 1)

    # left -> right (capacity 1)
    for l, r in edges:
        dinic.add_edge(l + 1, left + r + 1, 1)

    # right vertices -> sink (capacity 1)
    for j in range(right):
        dinic.add_edge(left + j + 1, sink, 1)

    max_matching = dinic.max_flow(source, sink)
    return max_matching

# Hungarian algorithm (direct DFS-based implementation)
def hungarian(n: int, m: int, adj: list) -> tuple:
    """Bipartite matching (Hungarian method) - O(VE)
    n: left vertex count, m: right vertex count
    adj[i]: list of right vertices adjacent to left vertex i
    """
    match_l = [-1] * n  # left match partner
    match_r = [-1] * m  # right match partner

    def dfs(u, visited):
        for v in adj[u]:
            if visited[v]:
                continue
            visited[v] = True
            # v is unmatched or v's match partner can be reassigned
            if match_r[v] == -1 or dfs(match_r[v], visited):
                match_l[u] = v
                match_r[v] = u
                return True
        return False

    matching = 0
    for u in range(n):
        visited = [False] * m
        if dfs(u, visited):
            matching += 1

    return matching, match_l, match_r

# Usage example: Students(0,1,2) -> Projects(0,1,2)
adj = [
    [0, 1],  # Student 0 -> P0, P1
    [0, 2],  # Student 1 -> P0, P2
    [1, 2],  # Student 2 -> P1, P2
]
count, ml, mr = hungarian(3, 3, adj)
print(f"Maximum matching: {count}")  # 3
print(f"Left->Right: {ml}")         # [0, 2, 1] or similar
```

### Hopcroft-Karp Algorithm

The fastest algorithm for bipartite matching. O(E sqrt(V)).

```python
def hopcroft_karp(n: int, m: int, adj: list) -> tuple:
    """Hopcroft-Karp algorithm - O(E sqrt(V))
    n: left vertex count, m: right vertex count
    adj[i]: list of right vertices adjacent to left vertex i
    """
    INF = float('inf')
    match_l = [-1] * n
    match_r = [-1] * m
    dist = [0] * n

    def bfs():
        queue = deque()
        for u in range(n):
            if match_l[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = INF

        found = False
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                w = match_r[v]
                if w == -1:
                    found = True
                elif dist[w] == INF:
                    dist[w] = dist[u] + 1
                    queue.append(w)
        return found

    def dfs(u):
        for v in adj[u]:
            w = match_r[v]
            if w == -1 or (dist[w] == dist[u] + 1 and dfs(w)):
                match_l[u] = v
                match_r[v] = u
                return True
        dist[u] = INF
        return False

    matching = 0
    while bfs():
        for u in range(n):
            if match_l[u] == -1:
                if dfs(u):
                    matching += 1

    return matching, match_l, match_r

# Usage example
adj = [
    [0, 1],   # Left 0 -> Right 0, Right 1
    [0, 2],   # Left 1 -> Right 0, Right 2
    [1, 2],   # Left 2 -> Right 1, Right 2
]
count, ml, mr = hopcroft_karp(3, 3, adj)
print(f"Maximum matching: {count}")  # 3
```

---

## 7. Application Patterns

### Minimum Vertex Cover (Konig's Theorem)

```python
def minimum_vertex_cover(n: int, m: int, adj: list) -> list:
    """Find the minimum vertex cover of a bipartite graph (Konig's theorem)
    Minimum vertex cover = Maximum matching (bipartite graphs only)
    Returns: list of vertices in the cover
    """
    _, match_l, match_r = hungarian(n, m, adj)

    # Traverse alternating paths from unmatched left vertices
    visited_l = [False] * n
    visited_r = [False] * m

    # Enqueue unmatched left vertices
    queue = deque()
    for u in range(n):
        if match_l[u] == -1:
            queue.append(u)
            visited_l[u] = True

    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if not visited_r[v]:
                visited_r[v] = True
                w = match_r[v]
                if w != -1 and not visited_l[w]:
                    visited_l[w] = True
                    queue.append(w)

    # Cover = unreachable left vertices + reachable right vertices
    cover = []
    for u in range(n):
        if not visited_l[u]:
            cover.append(('L', u))
    for v in range(m):
        if visited_r[v]:
            cover.append(('R', v))

    return cover
```

### Project Assignment Problem

```python
def project_assignment(students: list, projects: list,
                       preferences: dict) -> dict:
    """Assign the maximum number of students to projects"""
    n_students = len(students)
    n_projects = len(projects)

    student_idx = {s: i for i, s in enumerate(students)}
    project_idx = {p: i for i, p in enumerate(projects)}

    adj = [[] for _ in range(n_students)]
    for student, prefs in preferences.items():
        for proj in prefs:
            adj[student_idx[student]].append(project_idx[proj])

    count, match_l, _ = hungarian(n_students, n_projects, adj)

    assignment = {}
    for i, j in enumerate(match_l):
        if j != -1:
            assignment[students[i]] = projects[j]

    return assignment

students = ["Alice", "Bob", "Charlie"]
projects = ["Web", "AI", "DB"]
prefs = {
    "Alice": ["Web", "AI"],
    "Bob": ["AI", "DB"],
    "Charlie": ["Web", "DB"],
}
result = project_assignment(students, projects, prefs)
print(result)  # {'Alice': 'Web', 'Bob': 'AI', 'Charlie': 'DB'} etc.
```

### Vertex-Disjoint Paths (Vertex Splitting)

```python
def vertex_disjoint_paths(n: int, edges: list, s: int, t: int) -> int:
    """Maximum number of vertex-disjoint paths from s to t
    Split each vertex v into v_in and v_out, with capacity 1 on v_in -> v_out
    """
    # Vertex v -> v_in = 2*v, v_out = 2*v + 1
    dinic = Dinic(2 * n)

    for v in range(n):
        if v == s or v == t:
            dinic.add_edge(2 * v, 2 * v + 1, float('inf'))  # s, t are unlimited
        else:
            dinic.add_edge(2 * v, 2 * v + 1, 1)  # each vertex can be used only once

    for u, v in edges:
        dinic.add_edge(2 * u + 1, 2 * v, 1)  # u_out -> v_in
        dinic.add_edge(2 * v + 1, 2 * u, 1)  # v_out -> u_in (for undirected graphs)

    return dinic.max_flow(2 * s, 2 * t + 1)

# Usage example
edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]
print(vertex_disjoint_paths(5, edges, 0, 4))  # 2
```

### Image Segmentation (Min-Cut Application)

```python
def image_segmentation(rows: int, cols: int,
                        foreground_cost: list,
                        background_cost: list,
                        neighbor_penalty: float) -> list:
    """Min-cut based method to partition pixels into foreground/background
    foreground_cost[i]: cost of labeling pixel i as foreground
    background_cost[i]: cost of labeling pixel i as background
    neighbor_penalty: penalty when adjacent pixels have different labels
    """
    n = rows * cols
    source = n      # foreground source
    sink = n + 1    # background sink
    dinic = Dinic(n + 2)

    for i in range(n):
        # source -> pixel: foreground cost (cut when labeled as background)
        dinic.add_edge(source, i, int(background_cost[i] * 100))
        # pixel -> sink: background cost (cut when labeled as foreground)
        dinic.add_edge(i, sink, int(foreground_cost[i] * 100))

    # Neighbor penalty
    penalty = int(neighbor_penalty * 100)
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            for dr, dc in [(0, 1), (1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    nidx = nr * cols + nc
                    dinic.add_edge(idx, nidx, penalty)
                    dinic.add_edge(nidx, idx, penalty)

    dinic.max_flow(source, sink)

    # s-side = foreground, t-side = background
    s_side = set(dinic.min_cut(source))
    labels = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if r * cols + c in s_side:
                labels[r][c] = 1  # foreground

    return labels
```

---

## 8. Minimum Cost Flow

A problem where each edge has a per-unit flow cost, and the goal is to send a specified amount of flow at minimum total cost.

```python
class MinCostFlow:
    """Minimum cost flow (Primal-Dual / SPFA version)
    Successive Shortest Paths algorithm
    """

    def __init__(self, n: int):
        self.n = n
        self.graph = [[] for _ in range(n)]

    def add_edge(self, u: int, v: int, cap: int, cost: int):
        """Add an edge (capacity cap, unit cost cost)"""
        self.graph[u].append([v, cap, cost, len(self.graph[v])])
        self.graph[v].append([u, 0, -cost, len(self.graph[u]) - 1])

    def min_cost_flow(self, s: int, t: int, max_flow: int) -> tuple:
        """Send max_flow units from s to t at minimum cost
        Returns: (actual_flow, total_cost) or (-1, -1) if infeasible
        """
        total_flow = 0
        total_cost = 0

        while total_flow < max_flow:
            # Find shortest path using SPFA (improved Bellman-Ford)
            dist = [float('inf')] * self.n
            dist[s] = 0
            in_queue = [False] * self.n
            prev_node = [-1] * self.n
            prev_edge = [-1] * self.n

            queue = deque([s])
            in_queue[s] = True

            while queue:
                u = queue.popleft()
                in_queue[u] = False

                for i, (v, cap, cost, _) in enumerate(self.graph[u]):
                    if cap > 0 and dist[u] + cost < dist[v]:
                        dist[v] = dist[u] + cost
                        prev_node[v] = u
                        prev_edge[v] = i
                        if not in_queue[v]:
                            queue.append(v)
                            in_queue[v] = True

            if dist[t] == float('inf'):
                break  # t is unreachable

            # Bottleneck capacity on the path
            path_flow = max_flow - total_flow
            v = t
            while v != s:
                u = prev_node[v]
                e = prev_edge[v]
                path_flow = min(path_flow, self.graph[u][e][1])
                v = u

            # Update flow
            v = t
            while v != s:
                u = prev_node[v]
                e = prev_edge[v]
                self.graph[u][e][1] -= path_flow
                self.graph[v][self.graph[u][e][3]][1] += path_flow
                v = u

            total_flow += path_flow
            total_cost += path_flow * dist[t]

        return total_flow, total_cost

# Usage example: send 2 units of flow at minimum cost
mcf = MinCostFlow(4)
# s=0, A=1, B=2, t=3
mcf.add_edge(0, 1, 2, 1)  # s->A: capacity 2, cost 1
mcf.add_edge(0, 2, 2, 3)  # s->B: capacity 2, cost 3
mcf.add_edge(1, 3, 1, 2)  # A->t: capacity 1, cost 2
mcf.add_edge(2, 3, 2, 1)  # B->t: capacity 2, cost 1
mcf.add_edge(1, 2, 1, 1)  # A->B: capacity 1, cost 1

flow, cost = mcf.min_cost_flow(0, 3, 2)
print(f"Flow: {flow}, Cost: {cost}")
# Flow: 2, Cost: 6 (s->A->t: 1*3=3, s->A->B->t: 1*(1+1+1)=3 or similar)
```

### Min-Cost Flow Application: Minimizing Job Assignment Cost

```python
def min_cost_assignment(workers: list, tasks: list,
                         costs: dict) -> tuple:
    """Assign one task to each worker, minimizing total cost
    costs[(worker, task)] = cost
    """
    n_workers = len(workers)
    n_tasks = len(tasks)
    worker_idx = {w: i for i, w in enumerate(workers)}
    task_idx = {t: i for i, t in enumerate(tasks)}

    # Network: s -> workers -> tasks -> t
    n = n_workers + n_tasks + 2
    source = n - 2
    sink = n - 1
    mcf = MinCostFlow(n)

    for i in range(n_workers):
        mcf.add_edge(source, i, 1, 0)

    for (w, t), cost in costs.items():
        wi = worker_idx[w]
        ti = task_idx[t] + n_workers
        mcf.add_edge(wi, ti, 1, cost)

    for j in range(n_tasks):
        mcf.add_edge(n_workers + j, sink, 1, 0)

    flow, total_cost = mcf.min_cost_flow(source, sink, min(n_workers, n_tasks))

    return flow, total_cost

workers = ["Alice", "Bob", "Charlie"]
tasks = ["Task1", "Task2", "Task3"]
costs = {
    ("Alice", "Task1"): 5, ("Alice", "Task2"): 3, ("Alice", "Task3"): 7,
    ("Bob", "Task1"): 2, ("Bob", "Task2"): 6, ("Bob", "Task3"): 4,
    ("Charlie", "Task1"): 8, ("Charlie", "Task2"): 1, ("Charlie", "Task3"): 3,
}
flow, cost = min_cost_assignment(workers, tasks, costs)
print(f"Assignments: {flow}, Total cost: {cost}")
# Optimal: Alice-Task2(3), Bob-Task1(2), Charlie-Task3(3) -> Total cost: 8
```

---

## 9. Algorithm Comparison Table

| Algorithm | Complexity | Features |
|:---|:---|:---|
| Ford-Fulkerson (DFS) | O(E * max_flow) | May not terminate with irrational capacities |
| Edmonds-Karp (BFS) | O(VE^2) | Shortest augmenting path via BFS |
| Dinic | O(V^2 E) | Level graph + blocking flow |
| Push-Relabel | O(V^2 E) or O(V^3) | Preflow-relabel method |
| Hungarian | O(VE) | Specialized for bipartite matching |
| Hopcroft-Karp | O(E sqrt(V)) | Fastest bipartite matching |
| MCMC/SPFA | O(VE * flow) | Minimum cost flow |

## Reduction Relationships Among Flow Problems

```
Many combinatorial optimization problems reduce to flow problems:

Maximum bipartite matching <--- Maximum flow (capacity 1)
     |
     v  Konig's theorem
Minimum vertex cover <----- n - Maximum independent set
     |
     v  complement
Maximum independent set

Minimum path cover <------- n - Maximum matching (on DAG)

Edge-disjoint path count <-- Maximum flow (edge capacity 1)
Vertex-disjoint path count <-- Maximum flow (vertex splitting)
```

## Applications of Flow Problems

| Problem | Reduction Target | Capacity Setting |
|:---|:---|:---|
| Bipartite matching | Maximum flow | All edge capacities = 1 |
| Minimum vertex cover | Maximum matching | Konig's theorem |
| Maximum independent set | n - Minimum vertex cover | Complement |
| Edge-disjoint path count | Maximum flow | Edge capacity 1 |
| Vertex-disjoint path count | Maximum flow | Vertex splitting (capacity 1) |
| Minimum cost flow | SPFA + augmenting path | Edges with costs |
| Minimum path cover (DAG) | n - Maximum matching | In-degree/out-degree |
| Image segmentation | Minimum cut | Inter-pixel penalty |

---

## 10. Practical Applications

### Scheduling Problem

```python
def schedule_tasks(n_workers: int, tasks: list) -> int:
    """Each task has a start time, end time, and required number of workers.
    Find the minimum number of workers needed to satisfy all tasks simultaneously.
    (Sometimes solved as the inverse of max flow; here we verify via flow)
    """
    # Discretize time points
    times = set()
    for start, end, _ in tasks:
        times.add(start)
        times.add(end)
    times = sorted(times)
    time_idx = {t: i for i, t in enumerate(times)}

    n_times = len(times)
    source = n_times + len(tasks)
    sink = source + 1
    total_nodes = sink + 1
    dinic = Dinic(total_nodes)

    # Connect time nodes (worker flow)
    for i in range(n_times - 1):
        dinic.add_edge(i, i + 1, n_workers)

    # Each task: start time -> task node -> end time
    for task_id, (start, end, required) in enumerate(tasks):
        task_node = n_times + task_id
        si = time_idx[start]
        ei = time_idx[end]
        dinic.add_edge(si, task_node, required)
        dinic.add_edge(task_node, ei, required)

    # source -> first time point, last time point -> sink
    dinic.add_edge(source, 0, n_workers)
    dinic.add_edge(n_times - 1, sink, n_workers)

    return dinic.max_flow(source, sink)
```

### Network Reliability Analysis

```python
def network_reliability(n: int, edges: list, s: int, t: int) -> dict:
    """Compute network reliability metrics
    - Edge connectivity: minimum number of edges to disconnect s-t
    - Vertex connectivity: minimum number of vertices to disconnect s-t
    """
    # Edge connectivity = max flow (all edge capacities = 1)
    dinic_edge = Dinic(n)
    for u, v in edges:
        dinic_edge.add_edge(u, v, 1)
        dinic_edge.add_edge(v, u, 1)
    edge_connectivity = dinic_edge.max_flow(s, t)

    # Vertex connectivity = max flow after vertex splitting
    vertex_connectivity = vertex_disjoint_paths(n, edges, s, t)

    return {
        'edge_connectivity': edge_connectivity,
        'vertex_connectivity': vertex_connectivity,
    }

edges = [(0, 1), (0, 2), (1, 3), (2, 3), (1, 2)]
result = network_reliability(4, edges, 0, 3)
print(f"Edge connectivity: {result['edge_connectivity']}")
print(f"Vertex connectivity: {result['vertex_connectivity']}")
```

---

## 11. Anti-patterns

### Anti-pattern 1: Forgetting to Add Reverse Edges

```python
# BAD: Not adding reverse edges to the residual graph
def bad_add_edge(self, u, v, cap):
    self.graph[u][v] = cap
    # Missing graph[v][u] = 0 (reverse edge)!
    # -> Cannot cancel flow, so max flow cannot be computed

# GOOD: Always add reverse edges
def good_add_edge(self, u, v, cap):
    self.graph[u][v] += cap
    # Ensure reverse edge (initial capacity 0) exists
    if v not in self.graph or u not in self.graph[v]:
        self.graph[v][u] += 0
```

### Anti-pattern 2: Brute-Force for Bipartite Matching

```python
# BAD: Try all permutations for maximum matching -> O(n!)
from itertools import permutations
def bad_matching(adj, n, m):
    max_match = 0
    for perm in permutations(range(m)):
        count = sum(1 for i in range(min(n,m)) if perm[i] in adj[i])
        max_match = max(max_match, count)
    return max_match  # n! is far too large

# GOOD: Hungarian method or max flow -> O(VE) or O(V^2 E)
count, _, _ = hungarian(n, m, adj)
```

### Anti-pattern 3: Forgetting to Reset iter Array in Dinic's Algorithm

```python
# BAD: Not resetting iter for each BFS phase
def bad_max_flow(self, s, t):
    flow = 0
    self.iter = [0] * self.n  # initialized only once
    while self.bfs(s, t):
        # iter carries over from previous phase -> DFS doesn't work correctly
        f = self.dfs(s, t, float('inf'))
        flow += f

# GOOD: Reset iter for each BFS phase
def good_max_flow(self, s, t):
    flow = 0
    while self.bfs(s, t):
        self.iter = [0] * self.n  # reset every time
        while True:
            f = self.dfs(s, t, float('inf'))
            if f == 0:
                break
            flow += f
    return flow
```

### Anti-pattern 4: Incorrect Edge Addition for Undirected Graphs

```python
# BAD: Adding an undirected edge as only one directed edge
dinic.add_edge(u, v, cap)  # only u->v

# GOOD: Add both directions for undirected edges
dinic.add_edge(u, v, cap)
dinic.add_edge(v, u, cap)
# Note: Dinic's add_edge internally adds a reverse edge (capacity 0),
# so for undirected edges, you need to manually add both directions
```


---

## Troubleshooting

### Common Errors and Solutions

| Error | Cause | Solution |
|--------|------|--------|
| Initialization error | Configuration file issues | Check configuration file path and format |
| Timeout | Network latency / insufficient resources | Adjust timeout values, add retry logic |
| Out of memory | Data volume growth | Introduce batch processing, implement pagination |
| Permission error | Insufficient access rights | Check execution user permissions, review settings |
| Data inconsistency | Concurrent processing conflicts | Introduce locking mechanisms, manage transactions |

### Debugging Procedure

1. **Check error messages**: Read the stack trace and identify the location
2. **Establish reproduction steps**: Reproduce the error with minimal code
3. **Formulate hypotheses**: List possible causes
4. **Verify step by step**: Use logging or a debugger to verify hypotheses
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

Steps for diagnosing performance problems:

1. **Identify bottlenecks**: Measure with profiling tools
2. **Check memory usage**: Check for memory leaks
3. **Check I/O wait**: Monitor disk and network I/O status
4. **Check connection count**: Monitor connection pool state

| Problem Type | Diagnostic Tool | Countermeasure |
|-----------|-----------|------|
| CPU load | cProfile, py-spy | Algorithm improvement, parallelization |
| Memory leak | tracemalloc, objgraph | Proper reference release |
| I/O bottleneck | strace, iostat | Async I/O, caching |
| DB latency | EXPLAIN, slow query log | Index, query optimization |

---

## Design Decision Guide

### Selection Criteria Matrix

The following summarizes criteria for making technology choices.

| Criterion | Prioritize When | Can Compromise When |
|---------|------------|-------------|
| Performance | Real-time processing, large-scale data | Admin panels, batch processing |
| Maintainability | Long-term operation, team development | Prototypes, short-term projects |
| Scalability | Services expected to grow | Internal tools, fixed user base |
| Security | Personal information, financial data | Public data, internal use |
| Development speed | MVP, time to market | Quality-focused, mission-critical |

### Architecture Pattern Selection

```
+--------------------------------------------------+
|        Architecture Selection Flow                |
+--------------------------------------------------+
|                                                   |
|  (1) Team size?                                   |
|    +-- Small (1-5) -> Monolith                    |
|    +-- Large (10+) -> Go to (2)                   |
|                                                   |
|  (2) Deploy frequency?                            |
|    +-- Weekly or less -> Monolith + module split   |
|    +-- Daily/multiple -> Go to (3)                 |
|                                                   |
|  (3) Team independence?                            |
|    +-- High -> Microservices                       |
|    +-- Moderate -> Modular monolith                |
|                                                   |
+--------------------------------------------------+
```

### Trade-off Analysis

Technical decisions always involve trade-offs. Analyze from these perspectives:

**1. Short-term vs Long-term Cost**
- The fastest short-term approach may become technical debt in the long run
- Conversely, over-engineering increases short-term costs and can delay the project

**2. Consistency vs Flexibility**
- A unified tech stack has lower learning costs
- Diverse technology adoption enables best-fit choices but increases operational costs

**3. Level of Abstraction**
- Higher abstraction increases reusability but can make debugging harder
- Lower abstraction is intuitive but tends to result in code duplication

```python
# Design decision recording template
class ArchitectureDecisionRecord:
    """ADR (Architecture Decision Record) creation"""

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
            md += f"- [{icon}] {c['description']}\n"
        md += "\n## Rejected Alternatives\n"
        for a in self.alternatives:
            md += f"- **{a['name']}**: {a['reason_rejected']}\n"
        return md
```
---

## 12. FAQ

### Q1: What can the max-flow min-cut theorem be used for?

**A:** It is used to identify "bottlenecks" in networks. Examples include the most vulnerable link in a communication network, sections causing congestion in road networks, and maximum transport volume in pipelines. It is also applied to image segmentation (foreground/background separation).

### Q2: What is minimum cost flow?

**A:** A problem where each edge has a per-unit flow cost, and the goal is to send a specified amount of flow at minimum total cost. It finds shortest paths using SPFA (improved Bellman-Ford) while sending augmenting paths. Applications include job assignment cost minimization and transportation planning.

### Q3: What are real-world use cases for bipartite matching?

**A:** (1) Student-to-lab assignment, (2) task-to-worker assignment, (3) reviewer-to-paper matching, (4) stable marriage problem (Gale-Shapley), (5) register allocation in compilers. Hall's marriage theorem can determine the existence condition for a perfect matching.

### Q4: Should I use Dinic's algorithm or Edmonds-Karp?

**A:** Dinic's algorithm is almost always recommended. Its complexity is O(V^2 E), better than Edmonds-Karp's O(VE^2), and it becomes O(E sqrt(V)) for bipartite matching. Implementation complexity is roughly the same.

### Q5: What if the max flow problem has real-valued (floating-point) capacities?

**A:** Ford-Fulkerson (DFS version) may not converge (Zwick-Paterson counterexample). Edmonds-Karp (BFS version) and Dinic's algorithm work correctly even with rational capacities. In practice, either round to integers or use rational arithmetic.

### Q6: What is Hall's marriage theorem?

**A:** In a bipartite graph G = (L, R, E), a necessary and sufficient condition for matching all vertices in L (a perfect matching exists) is that "for any subset S of L, the neighborhood N(S) has size at least |S|." This can be derived from the max-flow min-cut theorem.

---

## 13. Push-Relabel Algorithm

While Ford-Fulkerson-type algorithms search for global augmenting paths, the Push-Relabel algorithm repeatedly performs local operations (push and relabel) to find the maximum flow. It delivers high performance on large-scale graphs.

### Algorithm Overview

```
Push-Relabel basic concepts:

1. Height label h(v): Assign a label to each vertex
   - h(s) = |V|, h(t) = 0
   - If edge (u,v) has residual capacity -> h(u) <= h(v) + 1

2. Excess flow e(v): inflow - outflow at each vertex
   - Vertices (other than s, t) with e(v) > 0 are "active"

3. Push operation: Send flow from active vertex u to adjacent vertex v
   Condition: h(u) = h(v) + 1 and residual capacity > 0
   Amount sent: min(e(u), residual capacity(u,v))

4. Relabel operation: Raise label when push is not possible
   h(u) = min(h(v) + 1 | (u,v) has residual capacity)

Flow of processing:
   Saturate all adjacent edges from s (initial push)
   -> Repeat Push or Relabel as long as active vertices exist
   -> When all vertex labels are stable, e(t) is the maximum flow

Complexity: O(V^2 E)  *O(V^3) with FIFO selection rule
```

### Push-Relabel Visualization

```
  Height label image (water flows from high to low):

  h=6(s)
   |
   | push
   v
  h=1(A) --push--> h=0(t)
   |
   | push
   v
  h=1(B) --push--> h=0(t)

  Vertices that cannot push are relabeled to raise their height:

  Before relabel:         After relabel:
  h=1(A)                  h=2(A)  <- raised
    v push impossible       v push now possible
  h=1(B)                  h=1(B)
    v                        v
  h=0(t)                  h=0(t)
```

```python
class PushRelabel:
    """Push-Relabel algorithm (FIFO selection rule) - O(V^3)"""

    def __init__(self, n: int):
        self.n = n
        self.cap = [[0] * n for _ in range(n)]
        self.flow = [[0] * n for _ in range(n)]

    def add_edge(self, u: int, v: int, c: int):
        self.cap[u][v] += c

    def max_flow(self, s: int, t: int) -> int:
        n = self.n
        height = [0] * n
        excess = [0] * n
        height[s] = n

        # Initial push: saturate all adjacent edges from s
        for v in range(n):
            if self.cap[s][v] > 0:
                f = self.cap[s][v]
                self.flow[s][v] = f
                self.flow[v][s] = -f
                excess[v] = f
                excess[s] -= f

        # FIFO queue of active vertices
        active = deque()
        in_queue = [False] * n
        for v in range(n):
            if v != s and v != t and excess[v] > 0:
                active.append(v)
                in_queue[v] = True

        while active:
            u = active.popleft()
            in_queue[u] = False
            self._discharge(u, s, t, height, excess, active, in_queue)

        return excess[t]

    def _discharge(self, u, s, t, height, excess, active, in_queue):
        n = self.n
        while excess[u] > 0:
            pushed = False
            for v in range(n):
                residual = self.cap[u][v] - self.flow[u][v]
                if residual > 0 and height[u] == height[v] + 1:
                    # Push
                    d = min(excess[u], residual)
                    self.flow[u][v] += d
                    self.flow[v][u] -= d
                    excess[u] -= d
                    excess[v] += d
                    if v != s and v != t and not in_queue[v] and excess[v] > 0:
                        active.append(v)
                        in_queue[v] = True
                    pushed = True
                    if excess[u] == 0:
                        break
            if not pushed:
                # Relabel
                min_height = float('inf')
                for v in range(n):
                    if self.cap[u][v] - self.flow[u][v] > 0:
                        min_height = min(min_height, height[v])
                height[u] = min_height + 1

        if excess[u] > 0 and not in_queue[u]:
            active.append(u)
            in_queue[u] = True

# Usage example
pr = PushRelabel(6)
pr.add_edge(0, 1, 10)
pr.add_edge(0, 2, 10)
pr.add_edge(1, 3, 4)
pr.add_edge(1, 4, 8)
pr.add_edge(2, 4, 9)
pr.add_edge(3, 5, 10)
pr.add_edge(4, 3, 6)
pr.add_edge(4, 5, 10)
print(pr.max_flow(0, 5))  # 19
```

### Algorithm Selection Guidelines

```
Decision flow for choosing the right algorithm:

Problem type?
+-- Bipartite matching -> Hopcroft-Karp  O(E sqrt(V))
+-- Minimum cost flow  -> SPFA + Successive Shortest Paths
+-- Simple max flow    -> Dinic's algorithm  O(V^2 E)
|    +-- Dense graph (E ~ V^2) -> Push-Relabel  O(V^3)
|    +-- Sparse graph (E ~ V)  -> Dinic's algorithm
+-- Small scale (V < 100) -> Any will work (Edmonds-Karp is simplest to implement)
```

---

## 14. Exercises (3 Levels)

### Beginner: Basic Flow Computation

**Problem 1:** Compute the maximum flow of the following graph by hand.

```
         8         6
    s -----> A -----> t
    |               ^
    |3              |5
    v               |
    B -------------> C
           7
```

**Hint:** Find augmenting paths one by one and update the residual graph.

**Solution:**

```
Path 1: s -> A -> t  Bottleneck = min(8, 6) = 6
  Residual: s->A: 2, A->t: 0, t->A: 6

Path 2: s -> B -> C -> t  Bottleneck = min(3, 7, 5) = 3
  Residual: s->B: 0, B->C: 4, C->t: 2

Path 3: No augmenting path (edges from s: s->A: 2 but A->t: 0, A->... unreachable)
  -> Search for s -> A -> ... t path in residual graph

  Checking residual graph:
  s->A: 2, A->t: 0, t->A: 6
  s->B: 0, B->C: 4, C->t: 2
  (including reverse edges)

  Actually, using s->A (capacity 2)... A cannot reach t directly
  Considering reverse edges etc., no additional path exists.

Maximum flow = 6 + 3 = 9

Verification: Minimum cut is not {s->A: 8, s->B: 3} nor
      {A->t: 6, C->t: 5} = 11.
      Actual minimum cut = {s->B: 3, A->t: 6} = 9  <- matches
```

**Problem 2:** Write code using the Ford-Fulkerson method to compute the maximum flow of the following 4-vertex graph.

```
s=0, A=1, B=2, t=3
Edges: s->A(cap 10), s->B(cap 5), A->B(cap 15), A->t(cap 10), B->t(cap 10)
```

### Intermediate: Bipartite Matching and Minimum Cut

**Problem 3:** There are 5 students and 5 labs. Find the maximum matching given the following preference lists.

```
Student 0: Labs {0, 1, 3}
Student 1: Labs {1, 2}
Student 2: Labs {0, 3}
Student 3: Labs {2, 4}
Student 4: Labs {1, 3, 4}
```

**Solution:**

```python
adj = [
    [0, 1, 3],   # Student 0
    [1, 2],       # Student 1
    [0, 3],       # Student 2
    [2, 4],       # Student 3
    [1, 3, 4],    # Student 4
]
count, match_l, match_r = hopcroft_karp(5, 5, adj)
print(f"Maximum matching: {count}")  # 5 (perfect matching possible)
# Example: Student 0->Lab 0, Student 1->Lab 2, Student 2->Lab 3,
#          Student 3->Lab 4, Student 4->Lab 1
```

**Problem 4:** Find the minimum cut edges of the network and identify the most critical bottleneck link.

```
6-vertex network:
s(0)->A(1): 16, s(0)->B(2): 13
A(1)->B(2): 4,  A(1)->C(3): 12
B(2)->A(1): 10, B(2)->D(4): 14
C(3)->B(2): 9,  C(3)->t(5): 20
D(4)->C(3): 7,  D(4)->t(5): 4
```

### Advanced: Minimum Cost Flow and Compound Problems

**Problem 5:** There are 3 factories and 4 stores. Given the supply of each factory, demand of each store, and transportation costs, find the transportation plan that minimizes total cost using minimum cost flow.

```
Factories: F1(supply 20), F2(supply 30), F3(supply 25)
Stores: S1(demand 15), S2(demand 20), S3(demand 25), S4(demand 15)

Transportation cost (per unit):
      S1  S2  S3  S4
F1:    4   6   8   5
F2:    6   3   5   7
F3:    3   8   4   6
```

**Hint:** Add a super-source s and super-sink t, then construct the network as: s->factories (capacity=supply, cost=0), stores->t (capacity=demand, cost=0), factories->stores (capacity=large enough, cost=transport cost).

```python
# Solution skeleton
mcf = MinCostFlow(2 + 3 + 4)  # s, t, 3 factories, 4 stores
source, sink = 0, 1
factories = [2, 3, 4]      # node numbers
stores = [5, 6, 7, 8]      # node numbers
supply = [20, 30, 25]
demand = [15, 20, 25, 15]
cost_matrix = [
    [4, 6, 8, 5],
    [6, 3, 5, 7],
    [3, 8, 4, 6],
]

for i, f in enumerate(factories):
    mcf.add_edge(source, f, supply[i], 0)

for i, f in enumerate(factories):
    for j, s in enumerate(stores):
        mcf.add_edge(f, s, min(supply[i], demand[j]), cost_matrix[i][j])

for j, s in enumerate(stores):
    mcf.add_edge(s, sink, demand[j], 0)

total_demand = sum(demand)  # 75
flow, total_cost = mcf.min_cost_flow(source, sink, total_demand)
print(f"Total transport volume: {flow}, Total cost: {total_cost}")
```

**Problem 6:** Find the minimum path cover on a directed acyclic graph (DAG). Use the fact that minimum path cover = n - maximum matching.

```
DAG (6 vertices):
0 -> 1, 0 -> 2
1 -> 3
2 -> 3, 2 -> 4
4 -> 5
```

---

## 15. Push-Relabel vs Dinic: Performance Comparison

| Aspect | Dinic's Algorithm | Push-Relabel |
|:---|:---|:---|
| Complexity (general) | O(V^2 E) | O(V^2 E) / O(V^3) |
| Complexity (unit capacity) | O(E sqrt(V)) | O(E sqrt(V)) |
| Ease of implementation | Moderate | Somewhat complex |
| Performance on sparse graphs | Excellent | Standard |
| Performance on dense graphs | Standard | Excellent |
| Memory usage | Adjacency list (lightweight) | Adjacency matrix version is O(V^2) |
| Competitive programming | Most commonly used | Rarely used |
| Production (large-scale) | Moderate | Advantageous due to high parallelism |

### Recommended Algorithms by Problem Scale

| Graph Scale | Edge Count | Recommended Algorithm | Reason |
|:---|:---|:---|:---|
| V < 100 | E < 1000 | Edmonds-Karp | Simplest to implement |
| V < 1000 | E < 10000 | Dinic | Balanced performance |
| V < 10000 | E < 100000 | Dinic | Fast on sparse graphs |
| V > 10000 | E ~ V^2 | Push-Relabel | Strong on dense graphs |
| Bipartite | - | Hopcroft-Karp | Fastest at O(E sqrt(V)) |
| With costs | - | SPFA + SSP | Only option |

---

## 16. Additional Anti-patterns

### Anti-pattern 5: Computing Only the Max-Flow Value Without Path Recovery

```python
# BAD: Computed the max-flow value but cannot determine which edges carry how much flow
flow_value = dinic.max_flow(s, t)
# Trying to output "how many units flow through each edge" requires
# reverse-computing from the residual graph

# GOOD: Design that records flow values alongside paths
class DinicWithFlowRecovery(Dinic):
    def get_flow_on_edges(self):
        """Recover the actual flow amount on each edge"""
        result = []
        for u in range(self.n):
            for i, (v, cap, rev) in enumerate(self.graph[u]):
                # Flow on original edges (not reverse edges)
                if i % 2 == 0:  # forward edge added by add_edge
                    original_cap = cap + self.graph[v][rev][1]
                    flow = self.graph[v][rev][1]
                    if flow > 0:
                        result.append((u, v, flow, original_cap))
        return result
```

### Anti-pattern 6: Incorrect Handling of Multiple Edges

```python
# BAD: Overwriting parallel edges with adjacency matrix
cap[u][v] = 5   # first edge
cap[u][v] = 3   # second edge -> overwrites! becomes 3 instead of total 8

# GOOD: Add capacities
cap[u][v] += 5   # first edge
cap[u][v] += 3   # second edge -> total 8

# With Dinic's adjacency list version, edges are automatically added as separate entries,
# so this problem does not occur (only the adjacency matrix version needs attention)
```

---

## 17. Additional FAQ

### Q7: Can flow problems have negative capacities?

**A:** In standard flow problems, capacities are non-negative (c(u,v) >= 0). However, in lower-bounded flow problems, each edge has a minimum flow requirement. In this case, a variable substitution can reduce it to a standard max-flow problem. Specifically, an edge with lower bound l(u,v) is transformed into "an edge with capacity c(u,v) - l(u,v)," and a super-source and super-sink are added.

### Q8: Can the max-flow problem be formulated as a linear program?

**A:** Yes. The max-flow problem is equivalent to the following linear program (LP):

```
maximize  sum f(s,v)  (total outflow from s)
subject to:
  0 <= f(u,v) <= c(u,v)           (capacity constraints)
  sum f(u,v) = sum f(v,w)  for all v != s,t  (flow conservation)
```

The max-flow min-cut theorem is derived from strong duality between this LP and its dual (the minimum cut). For integer capacities, the LP relaxation automatically yields an integer optimal solution (total unimodularity).

### Q9: How do you compute max flow when the graph changes dynamically?

**A:** In dynamic flow problems where edges are added or removed, recomputing from scratch each time is inefficient. When an edge is added, you can keep the existing flow and search for augmenting paths in the residual graph including the new edge (incremental computation). When an edge is removed, if no flow passes through it, nothing needs to be done. If flow does pass through it, a "cancellation" operation (sending flow in the reverse direction) is needed, which is somewhat more complex.

---


## FAQ

### Q1: What is the most important point when learning this topic?

Building practical experience is the most important aspect. Understanding deepens not only through theory but also by actually writing code and verifying its behavior.

### Q2: What common mistakes do beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving to the next step.

### Q3: How is this used in practice?

Knowledge of this topic is frequently applied in day-to-day development work. It becomes especially important during code reviews and architecture design.

---

## 18. Summary

| Topic | Key Points |
|:---|:---|
| Maximum flow problem | Repeatedly search for augmenting paths in the residual graph |
| Ford-Fulkerson | BFS version (Edmonds-Karp) runs in O(VE^2) |
| Dinic's algorithm | Level graph + blocking flow in O(V^2 E) |
| Push-Relabel | Local operations (push/relabel) in O(V^3), strong on dense graphs |
| Max-flow min-cut | Maximum flow = minimum cut (duality) |
| Bipartite matching | Reduce to max flow (capacity 1) or use Hungarian method |
| Hopcroft-Karp | Solves bipartite matching in O(E sqrt(V)) |
| Minimum cost flow | Cost minimization with cost-bearing edges |
| Applications | Assignment, covering, independent set, path separation, image processing, transportation planning |

### Learning Roadmap

```
Step 1: Fundamentals
  Flow definition -> Residual graph -> Augmenting path -> Ford-Fulkerson
  |
Step 2: Speed Improvements
  Edmonds-Karp -> Dinic's algorithm -> Push-Relabel
  |
Step 3: Applications
  Bipartite matching -> Minimum cut -> Vertex splitting -> Minimum cost flow
  |
Step 4: Advanced Topics
  Lower-bounded flow -> LP duality -> Dynamic flow -> Approximation algorithms
```

---

## Recommended Next Guides

- [Shortest Path](../02-algorithms/03-shortest-path.md) -- Prerequisite knowledge for flow algorithms
- [Graph Traversal](../02-algorithms/02-graph-traversal.md) -- BFS/DFS fundamentals
- [Competitive Programming](../04-practice/01-competitive-programming.md) -- Flow problems in practice

---

## References

1. Cormen, T. H. et al. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. -- Chapters 24-26: Comprehensive coverage of network flow
2. Ford, L. R. & Fulkerson, D. R. (1956). "Maximal flow through a network." *Canadian Journal of Mathematics*. -- Original paper on the maximum flow problem
3. Dinic, E. A. (1970). "Algorithm for solution of a problem of maximum flow in networks with power estimation." *Soviet Mathematics Doklady*. -- Original paper on Dinic's algorithm
4. Kleinberg, J. & Tardos, E. (2005). *Algorithm Design*. Pearson. -- Chapter 7: Rich coverage of network flow applications
5. Hopcroft, J. E. & Karp, R. M. (1973). "An n^{5/2} Algorithm for Maximum Matchings in Bipartite Graphs." *SIAM Journal on Computing*. -- Original paper on the Hopcroft-Karp algorithm
6. Konig, D. (1931). "Grafok es matrixok." *Matematikai es Fizikai Lapok*. -- Konig's theorem (minimum vertex cover = maximum matching)
7. Goldberg, A. V. & Tarjan, R. E. (1988). "A new approach to the maximum-flow problem." *Journal of the ACM*. -- Original paper on Push-Relabel
8. Ahuja, R. K., Magnanti, T. L. & Orlin, J. B. (1993). *Network Flows: Theory, Algorithms, and Applications*. Prentice Hall. -- The definitive text on network flow theory
9. Schrijver, A. (2003). *Combinatorial Optimization: Polyhedra and Efficiency*. Springer. -- Flow problems from the perspective of optimization theory
