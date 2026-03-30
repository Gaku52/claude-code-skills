# Greedy Algorithm

> Understand the design technique of efficiently finding a globally optimal solution by repeatedly making locally optimal choices at each step

## Learning Objectives

1. **Identify the conditions for applying greedy algorithms** (greedy choice property and optimal substructure) and verify their correctness
2. **Correctly solve the activity selection problem, Huffman coding, and minimum spanning tree** using greedy algorithms
3. **Judge the choice between greedy algorithms and DP**, and identify cases where greedy algorithms cannot be used
4. **Understand the basics of matroid theory** and systematically determine the correctness of greedy algorithms


## Prerequisites

Before reading this guide, the following knowledge will help deepen your understanding:

- Basic programming knowledge
- Understanding of related fundamental concepts
- Understanding of the content in [Dynamic Programming](./04-dynamic-programming.md)

---

## 1. Principles of Greedy Algorithms

```
+----------------------------------------------+
|        Two Conditions for Greedy Algorithms   |
+----------------------------------------------+
|                                               |
|  1. Greedy Choice Property                    |
|     -> A locally optimal choice leads to a    |
|        globally optimal solution              |
|                                               |
|  2. Optimal Substructure                      |
|     -> An optimal solution to the overall     |
|        problem can be obtained from optimal   |
|        solutions to subproblems               |
|                                               |
+----------------------------------------------+
|                                               |
|  Difference from DP:                          |
|  DP     -> Try all choices and select the     |
|            optimal one                        |
|  Greedy -> Make an immediate decision at each |
|            step (no backtracking)             |
|                                               |
|  Greedy is faster than DP but has a narrower  |
|  range of applicability                       |
+----------------------------------------------+
```

### Design Procedure for Greedy Algorithms

```
1. Formulate the problem as a sequence of choices
2. Define the greedy criterion for each step
3. Prove the greedy choice property (exchange argument or matroid)
4. Verify optimal substructure
5. Implement

Note: Skipping Step 3 risks making incorrect intuitive decisions
```

### Decision Flow for Greedy Applicability

```
When examining a problem:

  Is it an optimization problem?
    +- NO  -> Not a candidate for greedy algorithms
    +- YES -> Does local optimum = global optimum hold?
              +- YES -> Solvable by greedy (proof required)
              |         +- Can it be proved by exchange argument? -> Implement
              |         +- Does it have matroid structure? -> Implement
              +- NO or Unknown -> Consider DP
                    +- If a counterexample is found -> DP is confirmed
```

### 1.1 Exchange Argument in Detail

The exchange argument is the most common technique for proving the correctness of a greedy algorithm. The basic idea is to "assume that the optimal solution differs from the greedy solution, and show that replacing elements of the optimal solution with elements of the greedy solution does not compromise optimality."

```
+------------------------------------------------------+
|        General Procedure of the Exchange Argument     |
+------------------------------------------------------+
|                                                       |
|  Step 1: Let OPT be any optimal solution, G be the   |
|          greedy solution                              |
|                                                       |
|  Step 2: Identify the "first point of difference"     |
|          between OPT and G                            |
|          OPT = {o1, o2, ..., ok}                      |
|          G   = {g1, g2, ..., gm}                      |
|          Find the smallest i where oi != gi           |
|                                                       |
|  Step 3: Create OPT' by replacing oi with gi in OPT  |
|          OPT' = {o1, ..., o(i-1), gi, o(i+1), ...}   |
|                                                       |
|  Step 4: Show that OPT' satisfies the following:      |
|          (a) OPT' is a valid solution                 |
|          (b) Objective value of OPT' >= that of OPT   |
|                                                       |
|  Step 5: Show that repeated application can transform |
|          OPT into G                                   |
|          -> |G| = |OPT| holds -> G is optimal         |
|                                                       |
+------------------------------------------------------+
```

### 1.2 General Template for Greedy Algorithms

```python
def greedy_template(problem_input):
    """General template for greedy algorithms"""
    # Step 1: Sort input by greedy criterion
    candidates = sort_by_greedy_criterion(problem_input)

    solution = []

    # Step 2: Evaluate each candidate
    for candidate in candidates:
        if is_feasible(solution, candidate):
            # Step 3: Add to solution if it satisfies constraints
            solution.append(candidate)

    return solution
```

---

## 2. Activity Selection Problem

Select the activity with the earliest finish time first, and schedule as many activities as possible.

```
Activities: Start  End
 a1:   1 --- 4
 a2:     3 ----- 5
 a3:  0 ---- 6
 a4:       5 --- 7
 a5:         3 ----- 9
 a6:            5 --------- 9
 a7:              6 --- 8
 a8:                  8 --- 11
 a9:                    8 ----- 12
 a10:                      2 ---------- 14
 a11:                              12 --- 16

Timeline:
0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16
|--a1--|     |--a4--|  |--a7--|   |--a8--|   |--a11--|
               <- Greedy selection by finish time -> Max 4 activities
```

### Correctness Proof (Exchange Argument)

```
Theorem: The greedy algorithm that selects the activity with the
earliest finish time is optimal.

Proof (Exchange Argument):
  Let OPT be the optimal solution, G be the greedy solution.
  Suppose the first activity in OPT differs from the first
  activity a1 in G.

  Replacing the first activity in OPT with a1:
  - a1 has the earliest finish time among all activities
  - Therefore a1 finishes no later than the first activity of OPT
  - This does not conflict with the second and subsequent
    activities of OPT
  - The result is still a valid solution with the same number
    of activities

  Repeating this process can transform OPT into G.
  Therefore |G| = |OPT|, i.e., the greedy solution is optimal. []
```

```python
def activity_selection(activities: list) -> list:
    """Activity Selection Problem - O(n log n)
    activities: [(start, end), ...]
    """
    # Sort by finish time
    sorted_acts = sorted(activities, key=lambda x: x[1])
    selected = [sorted_acts[0]]
    last_end = sorted_acts[0][1]

    for start, end in sorted_acts[1:]:
        if start >= last_end:  # Does not overlap with previous activity
            selected.append((start, end))
            last_end = end

    return selected

activities = [(1,4), (3,5), (0,6), (5,7), (3,9), (5,9),
              (6,8), (8,11), (8,12), (2,14), (12,16)]
result = activity_selection(activities)
print(f"Selected activities: {result}")
# [(1, 4), (5, 7), (8, 11), (12, 16)]
print(f"Number of activities: {len(result)}")  # 4
```

### Weighted Activity Selection Problem

When activities have weights (profits), the greedy algorithm cannot solve the problem. DP is required.

```python
import bisect

def weighted_activity_selection(activities: list) -> int:
    """Weighted Activity Selection Problem - O(n log n)
    activities: [(start, end, weight), ...]
    Cannot be solved greedily; uses DP instead
    """
    # Sort by finish time
    activities.sort(key=lambda x: x[1])
    n = len(activities)

    # For each activity i, find the latest non-conflicting activity via binary search
    ends = [a[1] for a in activities]

    def latest_non_conflict(i):
        target = activities[i][0]
        idx = bisect.bisect_right(ends, target) - 1
        return idx if idx < i else idx

    # DP
    dp = [0] * (n + 1)
    for i in range(1, n + 1):
        # Exclude activity i-1
        dp[i] = dp[i - 1]
        # Include activity i-1
        j = latest_non_conflict(i - 1)
        dp[i] = max(dp[i], dp[j + 1] + activities[i - 1][2])

    return dp[n]

activities_w = [(1, 4, 5), (3, 5, 6), (0, 6, 8), (5, 7, 4), (6, 9, 2)]
print(weighted_activity_selection(activities_w))  # 13 (activities (0,6,8) + (6,9,2)?)
```

---

## 3. Huffman Coding

Assign longer codes to less frequent characters and shorter codes to more frequent characters, minimizing the total number of bits.

```
Characters and frequencies:
  a:45  b:13  c:12  d:16  e:9  f:5

Constructing the Huffman tree:
Step1: f(5) + e(9) = 14
Step2: c(12) + b(13) = 25
Step3: 14 + d(16) = 30
Step4: 25 + 30 = 55
Step5: a(45) + 55 = 100

         (100)
        /      \
     a(45)    (55)
             /     \
          (25)     (30)
         /    \    /    \
       c(12) b(13) (14) d(16)
                  /    \
                f(5)  e(9)

Code assignment:
  a: 0       (1 bit)
  c: 100     (3 bits)
  b: 101     (3 bits)
  f: 1100    (4 bits)
  e: 1101    (4 bits)
  d: 111     (3 bits)
```

### Optimality of Huffman Coding

```
Theorem: Huffman coding is an optimal prefix code.

Prefix code: No codeword is a prefix of another codeword
  -> Can be decoded unambiguously

Intuition for optimality:
  - The two least frequent characters are placed at the deepest
    sibling nodes in the optimal tree
  - Merging these two characters does not change the problem
    structure (optimal substructure)
  - Always merge the two with the lowest frequency (greedy choice)
```

```python
import heapq
from collections import Counter

class HuffmanNode:
    def __init__(self, char=None, freq=0, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(freq: dict) -> HuffmanNode:
    """Build a Huffman tree - O(n log n)"""
    heap = [HuffmanNode(char=c, freq=f) for c, f in freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(freq=left.freq + right.freq,
                             left=left, right=right)
        heapq.heappush(heap, merged)

    return heap[0]

def build_codes(root: HuffmanNode, prefix="", codes=None) -> dict:
    """Generate Huffman codes"""
    if codes is None:
        codes = {}

    if root.char is not None:
        codes[root.char] = prefix or "0"
        return codes

    if root.left:
        build_codes(root.left, prefix + "0", codes)
    if root.right:
        build_codes(root.right, prefix + "1", codes)

    return codes

def huffman_encode(text: str) -> tuple:
    """Huffman encoding"""
    freq = Counter(text)
    tree = build_huffman_tree(freq)
    codes = build_codes(tree)
    encoded = ''.join(codes[c] for c in text)
    return encoded, codes, tree

def huffman_decode(encoded: str, tree: HuffmanNode) -> str:
    """Huffman decoding"""
    result = []
    node = tree
    for bit in encoded:
        if bit == '0':
            node = node.left
        else:
            node = node.right
        if node.char is not None:
            result.append(node.char)
            node = tree
    return ''.join(result)

# Usage example
text = "aaaaabbbccddddeefffff"
encoded, codes, tree = huffman_encode(text)
print("Code table:", codes)
print(f"Original size: {len(text) * 8} bits")
print(f"Compressed: {len(encoded)} bits")
print(f"Compression ratio: {len(encoded) / (len(text) * 8):.1%}")

# Decode and verify
decoded = huffman_decode(encoded, tree)
print(f"Decode matches: {decoded == text}")  # True
```

### Adaptive Huffman Coding

```python
# In practice, "adaptive Huffman" is used more often than "static Huffman"
#
# Static Huffman:
#   - Processes the text in 2 passes (1st pass for frequency counting, 2nd for encoding)
#   - The code table must be stored alongside the data
#
# Adaptive Huffman:
#   - Processes in 1 pass (updates the tree while reading characters)
#   - No need to transmit the code table
#   - Used in gzip, DEFLATE algorithm, etc.
#
# Practical compression libraries:
#   - zlib: DEFLATE (LZ77 + Huffman)
#   - brotli: LZ77 + Huffman + context modeling
#   - zstd: LZ77 + FSE (Finite State Entropy)
```

---

## 4. Minimum Spanning Tree

### 4.1 Kruskal's Algorithm

Examine edges in ascending order of weight and add edges that do not create a cycle.

```
Graph:
    A ---4--- B
    |       / |
    8     2   6
    |   /     |
    C ---3--- D
      \     /
       7   9
        \ /
         E

Edges in weight order: (B,C,2) -> (C,D,3) -> (A,B,4) -> (B,D,6) -> (C,E,7) -> (A,C,8) -> (D,E,9)

Step1: B-C (2) added  <- No cycle
Step2: C-D (3) added  <- No cycle
Step3: A-B (4) added  <- No cycle
Step4: B-D (6) skipped <- Cycle via B-C-D!
Step5: C-E (7) added  <- No cycle -> V-1=4 edges -> Done

MST: B-C(2), C-D(3), A-B(4), C-E(7) = Total 16
```

```python
class UnionFind:
    """Union-Find (for Kruskal's)"""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        self.size[px] += self.size[py]
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

    def connected(self, x, y):
        return self.find(x) == self.find(y)

def kruskal(n: int, edges: list) -> tuple:
    """Kruskal's Algorithm - O(E log E)
    edges: [(weight, u, v), ...]
    Returns: (MST edge list, total weight)
    """
    edges.sort()  # Sort by weight
    uf = UnionFind(n)
    mst = []
    total = 0

    for w, u, v in edges:
        if uf.union(u, v):
            mst.append((u, v, w))
            total += w
            if len(mst) == n - 1:
                break

    return mst, total

# Vertices: 0=A, 1=B, 2=C, 3=D, 4=E
edges = [(4,0,1), (8,0,2), (2,1,2), (6,1,3), (3,2,3), (7,2,4), (9,3,4)]
mst, total = kruskal(5, edges)
print(f"MST edges: {mst}")     # [(1, 2, 2), (2, 3, 3), (0, 1, 4), (2, 4, 7)]
print(f"Total weight: {total}")  # 16
```

### 4.2 Prim's Algorithm

Builds the MST vertex by vertex. More efficient than Kruskal for dense graphs.

```python
import heapq

def prim(graph: dict, start: int = 0) -> tuple:
    """Prim's Algorithm - O((V + E) log V)
    graph: {u: [(v, weight), ...]}
    Returns: (MST edge list, total weight)
    """
    mst = []
    total = 0
    visited = {start}
    # Heap of (weight, from, to)
    edges = [(w, start, v) for v, w in graph[start]]
    heapq.heapify(edges)

    while edges and len(mst) < len(graph) - 1:
        w, u, v = heapq.heappop(edges)
        if v in visited:
            continue
        visited.add(v)
        mst.append((u, v, w))
        total += w

        for next_v, next_w in graph[v]:
            if next_v not in visited:
                heapq.heappush(edges, (next_w, v, next_v))

    return mst, total

graph_prim = {
    0: [(1, 4), (2, 8)],
    1: [(0, 4), (2, 2), (3, 6)],
    2: [(0, 8), (1, 2), (3, 3), (4, 7)],
    3: [(1, 6), (2, 3), (4, 9)],
    4: [(2, 7), (3, 9)],
}
mst, total = prim(graph_prim)
print(f"MST edges: {mst}")     # Same result as Kruskal
print(f"Total weight: {total}")  # 16
```

### 4.3 Kruskal vs Prim

| Property | Kruskal | Prim |
|:---|:---|:---|
| Approach | Edge-based | Vertex-based |
| Data Structure | Union-Find | Priority queue |
| Time Complexity | O(E log E) | O((V+E) log V) |
| Sparse graphs | Efficient | Somewhat inefficient |
| Dense graphs | Inefficient | Efficient |
| Disconnected graphs | Returns a forest | Returns a single tree |
| Implementation simplicity | Somewhat complex (UF required) | Relatively simple |

### 4.4 MST Correctness: Cut Property

```
Cut Property:
  Consider a cut (partition of the vertex set into two groups) of the graph.
  Among the edges crossing the cut, the one with minimum weight is
  guaranteed to be in some MST.

  From this property:
  - Kruskal: Selects the globally minimum edge -> minimum across
    some cut between two connected components
  - Prim: Selects the minimum edge across the cut between the tree
    and non-tree vertices

  Both are correct based on the cut property.
```

### 4.5 Boruvka's Algorithm

In addition to Kruskal and Prim, Boruvka's algorithm is another MST algorithm. It is a greedy approach well-suited for parallel processing.

```
+------------------------------------------------------+
|          How Boruvka's Algorithm Works                |
+------------------------------------------------------+
|                                                       |
|  Phase 1: Each vertex is an independent component     |
|    A(0)  B(0)  C(0)  D(0)  E(0)                      |
|                                                       |
|  Phase 2: Select the minimum edge from each           |
|           component (parallelizable)                  |
|    A->B(4), B->C(2), C->B(2), D->C(3), E->C(7)      |
|    Added: B-C(2), A-B(4), C-D(3), C-E(7)             |
|                                                       |
|  -> All vertices connected in 1 phase -> Done         |
|  Total: 2 + 3 + 4 + 7 = 16                           |
|                                                       |
|  Key property: The number of components halves        |
|  (or more) in each phase                              |
|  -> O(E log V), at most O(log V) phases               |
+------------------------------------------------------+
```

```python
def boruvka(n: int, edges: list) -> tuple:
    """Boruvka's Algorithm - O(E log V)
    edges: [(u, v, weight), ...]
    MST algorithm well-suited for parallel processing
    """
    uf = UnionFind(n)
    mst = []
    total = 0
    num_components = n

    while num_components > 1:
        # Record the minimum edge for each component
        cheapest = [None] * n  # cheapest[comp] = (weight, u, v)

        for u, v, w in edges:
            comp_u = uf.find(u)
            comp_v = uf.find(v)

            if comp_u == comp_v:
                continue  # Same component

            if cheapest[comp_u] is None or w < cheapest[comp_u][0]:
                cheapest[comp_u] = (w, u, v)
            if cheapest[comp_v] is None or w < cheapest[comp_v][0]:
                cheapest[comp_v] = (w, u, v)

        # Add the minimum edge from each component
        for comp in range(n):
            if cheapest[comp] is not None:
                w, u, v = cheapest[comp]
                if uf.find(u) != uf.find(v):
                    uf.union(u, v)
                    mst.append((u, v, w))
                    total += w
                    num_components -= 1

    return mst, total

# Usage example
edges_b = [(0,1,4), (0,2,8), (1,2,2), (1,3,6), (2,3,3), (2,4,7), (3,4,9)]
mst_b, total_b = boruvka(5, edges_b)
print(f"Boruvka MST: {mst_b}")
print(f"Total weight: {total_b}")  # 16
```

---

## 5. Other Examples of Greedy Algorithms

### 5.1 Fractional Knapsack

```python
def fractional_knapsack(items: list, capacity: float) -> float:
    """Fractional Knapsack - O(n log n)
    items: [(weight, value), ...]
    """
    # Sort by value-to-weight ratio (descending)
    items_sorted = sorted(items, key=lambda x: x[1]/x[0], reverse=True)

    total_value = 0.0
    remaining = capacity

    for weight, value in items_sorted:
        if remaining >= weight:
            total_value += value
            remaining -= weight
        else:
            total_value += value * (remaining / weight)
            break

    return total_value

items = [(10, 60), (20, 100), (30, 120)]
print(fractional_knapsack(items, 50))  # 240.0
```

### 5.2 Interval Scheduling Maximization

```python
def interval_scheduling(intervals: list) -> int:
    """Maximum number of non-overlapping intervals - O(n log n)"""
    intervals.sort(key=lambda x: x[1])  # Sort by finish time
    count = 0
    last_end = float('-inf')

    for start, end in intervals:
        if start >= last_end:
            count += 1
            last_end = end

    return count

intervals = [(1,3), (2,5), (4,7), (1,8), (5,9), (8,10)]
print(interval_scheduling(intervals))  # 3: (1,3), (4,7), (8,10)
```

### 5.3 Minimum Interval Cover

```python
def min_interval_cover(intervals: list, target_start: int, target_end: int) -> list:
    """Cover [target_start, target_end] with minimum number of intervals - O(n log n)"""
    intervals.sort()
    result = []
    i = 0
    n = len(intervals)
    current = target_start

    while current < target_end and i < n:
        # Select the interval starting at or before current that extends farthest
        best_end = current
        while i < n and intervals[i][0] <= current:
            best_end = max(best_end, intervals[i][1])
            i += 1

        if best_end == current:
            return []  # Cannot cover

        result.append(best_end)
        current = best_end

    return result if current >= target_end else []

intervals = [(0, 3), (1, 5), (2, 7), (4, 9), (6, 10)]
print(min_interval_cover(intervals, 0, 10))  # [3, 7, 10]
```

### 5.4 Job Scheduling with Deadlines

```python
def job_scheduling_with_deadlines(jobs: list) -> tuple:
    """Job scheduling with deadlines - O(n^2 log n)
    jobs: [(deadline, profit), ...]
    Each job takes 1 unit of time. Profit is earned if completed by the deadline.
    """
    # Sort by profit in descending order
    jobs_sorted = sorted(enumerate(jobs), key=lambda x: x[1][1], reverse=True)
    max_deadline = max(d for d, _ in jobs)

    # Slot management (1-indexed)
    slots = [False] * (max_deadline + 1)
    result = []
    total_profit = 0

    for idx, (deadline, profit) in jobs_sorted:
        # Find the latest available slot before the deadline
        for t in range(min(deadline, max_deadline), 0, -1):
            if not slots[t]:
                slots[t] = True
                result.append((idx, t, profit))
                total_profit += profit
                break

    return total_profit, result

jobs = [(2, 100), (1, 19), (2, 27), (1, 25), (3, 15)]
profit, schedule = job_scheduling_with_deadlines(jobs)
print(f"Maximum profit: {profit}")    # 142
print(f"Schedule: {schedule}")
```

### 5.5 String Compression (Run-Length Encoding)

```python
def run_length_encode(s: str) -> str:
    """Run-length encoding -- a form of greedy algorithm"""
    if not s:
        return ""

    result = []
    count = 1

    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            count += 1
        else:
            result.append(f"{s[i-1]}{count}")
            count = 1

    result.append(f"{s[-1]}{count}")
    return ''.join(result)

def run_length_decode(encoded: str) -> str:
    """Run-length decoding"""
    result = []
    i = 0
    while i < len(encoded):
        char = encoded[i]
        i += 1
        num = []
        while i < len(encoded) and encoded[i].isdigit():
            num.append(encoded[i])
            i += 1
        result.append(char * int(''.join(num)))
    return ''.join(result)

original = "AAABBBCCDDDDEEFFFFF"
encoded = run_length_encode(original)
print(f"Encoded: {encoded}")      # A3B3C2D4E2F5
print(f"Decoded: {run_length_decode(encoded)}")  # AAABBBCCDDDDEEFFFFF
```

### 5.6 Dijkstra's Algorithm (Viewed as a Greedy Algorithm)

```python
# Dijkstra's algorithm can be understood as an example of a greedy algorithm:
#
# Greedy choice: "Finalize the unvisited vertex with the smallest distance"
#
# Proof of greedy choice property:
#   - Unvisited vertex u has the smallest distance d[u]
#   - Suppose another path s -> ... -> w -> ... -> u is shorter than d[u]
#   - But w is unvisited, so d[w] >= d[u]
#   - Since edge weights are non-negative, the path via w is >= d[w] >= d[u]
#   - Contradiction -> d[u] is the shortest
#
# Condition for this proof to hold: All edge weights are non-negative
# With negative edges this breaks -> Bellman-Ford (DP-based) is needed
```

### 5.7 Gas Station Problem

The problem of minimizing the number of gas station stops during a trip is a classic greedy problem.

```python
def min_gas_stops(stations: list, tank_capacity: int, total_distance: int) -> list:
    """Gas Station Problem - O(n)
    stations: List of gas station positions (distances), sorted
    tank_capacity: Maximum distance on a full tank
    total_distance: Total distance to destination

    Greedy strategy: Refuel at the farthest reachable station
    """
    stops = []
    current_fuel = tank_capacity
    current_pos = 0

    # Add destination to the list
    all_points = stations + [total_distance]

    for point in all_points:
        distance = point - current_pos

        if distance > tank_capacity:
            return []  # Unreachable

        if current_fuel < distance:
            # Insufficient fuel -> should have refueled at previous station
            # This algorithm uses the "go as far as possible then refuel" strategy
            stops.append(current_pos)
            current_fuel = tank_capacity

        current_fuel -= distance
        current_pos = point

    return stops


def min_gas_stops_greedy(stations: list, tank_capacity: int, total_distance: int) -> list:
    """Improved version: Greedy algorithm that goes as far as possible before refueling - O(n)"""
    if not stations:
        return [] if tank_capacity >= total_distance else [-1]

    stops = []
    current_fuel = tank_capacity
    prev_pos = 0

    for i, station_pos in enumerate(stations):
        dist = station_pos - prev_pos

        if dist > current_fuel:
            # Cannot reach here -> should have refueled at the previous station
            # If there is no previous station, destination is unreachable
            if not stops and prev_pos == 0:
                return [-1]  # Unreachable
            current_fuel = tank_capacity
            dist = station_pos - prev_pos
            if dist > current_fuel:
                return [-1]

        current_fuel -= dist
        prev_pos = station_pos

        # Check if we can reach the next point
        next_point = stations[i + 1] if i + 1 < len(stations) else total_distance
        if current_fuel < next_point - station_pos:
            stops.append(station_pos)
            current_fuel = tank_capacity

    # Final segment
    if current_fuel < total_distance - prev_pos:
        return [-1]

    return stops

# Usage example
stations = [100, 200, 375, 550, 750]
tank = 400
distance = 900
result = min_gas_stops_greedy(stations, tank, distance)
print(f"Refueling points: {result}")  # Minimum stops to reach destination
```

---

## 6. Matroid Theory and Greedy Algorithms

```
Definition of a Matroid:
  A set S and a family of independent sets I form a matroid (S, I) if:
  1. The empty set is independent (empty set in I)
  2. Hereditary property: If A in I and B is a subset of A, then B in I
  3. Exchange property: If A, B in I and |A| < |B|, then there exists
     some b in B\A such that A union {b} is in I

Theorem (Rado-Edmonds):
  The maximum weight independent set of a weighted matroid can be found
  by a greedy algorithm (adding elements in decreasing order of weight
  while maintaining independence).

Example:
  - Edge set of a graph + forest (no cycles) condition -> Graphic Matroid
  - -> This is why the MST can be found greedily (Kruskal)
```

### 6.1 Concrete Examples of Matroids

```
+----------------------------------------------------------+
|             Common Types of Matroids                      |
+----------------------------------------------------------+
|                                                           |
|  1. Graphic Matroid                                       |
|     S = edge set of a graph                               |
|     I = forests (subsets of edges without cycles)          |
|     Application: Minimum Spanning Tree (Kruskal)          |
|                                                           |
|  2. Uniform Matroid                                       |
|     S = n elements                                        |
|     I = subsets with at most k elements                   |
|     Application: Selecting the top k                      |
|                                                           |
|  3. Partition Matroid                                      |
|     S partitioned into groups, selecting at most ki       |
|     from each group                                       |
|     Application: Selection with per-category limits       |
|                                                           |
|  4. Linear Matroid                                        |
|     S = set of vectors                                    |
|     I = linearly independent subsets of vectors           |
|     Application: Basis selection in linear algebra        |
|                                                           |
|  5. Transversal Matroid                                   |
|     Independent sets based on bipartite graph matchings   |
|     Application: Substructure of assignment problems      |
|                                                           |
+----------------------------------------------------------+
```

### 6.2 Verifying the Relationship Between Matroids and Greedy Algorithms

```python
def verify_matroid_greedy(elements: list, weights: dict,
                          is_independent) -> list:
    """Find the maximum weight independent set on a matroid using greedy
    elements: the ground set
    weights: weight of each element
    is_independent: function to test independence
    """
    # Sort by weight in descending order
    sorted_elements = sorted(elements, key=lambda x: weights[x], reverse=True)

    solution = []
    for elem in sorted_elements:
        candidate = solution + [elem]
        if is_independent(candidate):
            solution.append(elem)

    return solution

# Uniform matroid example: Select top k elements by weight
elements = ['a', 'b', 'c', 'd', 'e']
weights = {'a': 10, 'b': 30, 'c': 20, 'd': 5, 'e': 25}
k = 3

def uniform_independent(subset):
    return len(subset) <= k

result = verify_matroid_greedy(elements, weights, uniform_independent)
print(f"Selection: {result}")  # ['b', 'e', 'c'] (weights: 30, 25, 20)
print(f"Total weight: {sum(weights[x] for x in result)}")  # 75
```

---

## 7. Greedy vs DP Comparison Table

| Property | Greedy | Dynamic Programming |
|:---|:---|:---|
| Selection method | Immediately decide on local optimum | Compare all options |
| Backtracking | None | None (all explored) |
| Time complexity | Typically O(n log n) | Typically O(n^2) or more |
| Correctness proof | Required (check for counterexamples) | Proved by correctness of transitions |
| Applicability | Narrow (strict conditions) | Wide |
| Implementation simplicity | Simple | Somewhat complex |
| Space complexity | O(1) to O(n) | O(n) to O(n^2) |
| Optimality guarantee | Guaranteed if proven | Always guaranteed |

## Problems Solvable and Unsolvable by Greedy

| Problem | Solvable by Greedy? | Reason |
|:---|:---|:---|
| Activity Selection | Yes | Greedy by finish time is optimal |
| Fractional Knapsack | Yes | Selection by unit value is optimal |
| 0/1 Knapsack | No | Items cannot be split -> DP needed |
| Huffman Coding | Yes | Merging least frequent pair is optimal |
| Minimum Spanning Tree | Yes | Correctness by cut property |
| Shortest Path (non-negative edges) | Yes | Dijkstra is greedy |
| Coin Problem (general) | No | Greedy works only for specific denominations |
| Weighted Activity Selection | No | DP is needed |
| Set Cover Problem | Approximation only | NP-hard, greedy gives ln(n) approximation |
| Minimum Graph Coloring | No | NP-hard |

---

## 8. Practical Application Patterns

### 8.1 CDN Server Placement

```python
def greedy_facility_placement(cities: list, k: int) -> list:
    """Greedily place k facilities (minimize maximum distance)
    This is an approximation algorithm (guarantees within 2x of optimal)
    """
    n = len(cities)
    if k >= n:
        return list(range(n))

    # First facility: arbitrary (here, city 0)
    facilities = [0]
    min_dist = [float('inf')] * n

    for _ in range(k - 1):
        # Update the distance from each city to the nearest facility
        last = facilities[-1]
        for j in range(n):
            d = abs(cities[j][0] - cities[last][0]) + abs(cities[j][1] - cities[last][1])
            min_dist[j] = min(min_dist[j], d)

        # Choose the city farthest from its nearest facility
        farthest = max(range(n), key=lambda j: min_dist[j] if j not in facilities else -1)
        facilities.append(farthest)

    return facilities
```

### 8.2 Task Deadline Optimization

```python
def minimize_lateness(tasks: list) -> tuple:
    """Scheduling to minimize maximum lateness
    tasks: [(processing_time, deadline), ...]
    Optimal strategy: Process in order of earliest deadline (EDF: Earliest Deadline First)
    """
    indexed = [(d, p, i) for i, (p, d) in enumerate(tasks)]
    indexed.sort()  # Sort by deadline

    schedule = []
    current_time = 0
    max_lateness = 0

    for deadline, proc_time, idx in indexed:
        start = current_time
        finish = current_time + proc_time
        lateness = max(0, finish - deadline)
        max_lateness = max(max_lateness, lateness)
        schedule.append((idx, start, finish, lateness))
        current_time = finish

    return max_lateness, schedule

tasks = [(3, 6), (2, 8), (1, 9), (4, 9), (3, 14), (2, 15)]
max_late, sched = minimize_lateness(tasks)
print(f"Maximum lateness: {max_late}")
for idx, start, finish, late in sched:
    print(f"  Task {idx}: {start}-{finish} (lateness: {late})")
```

### 8.3 Page Cache Replacement Strategy

```python
def optimal_page_replacement(pages: list, cache_size: int) -> int:
    """Belady's Optimal Page Replacement Algorithm (assumes knowledge of future accesses)
    Returns the number of page faults
    """
    cache = set()
    faults = 0

    for i, page in enumerate(pages):
        if page in cache:
            continue

        faults += 1

        if len(cache) < cache_size:
            cache.add(page)
        else:
            # Evict the page in cache whose next use is farthest in the future
            farthest = -1
            victim = None
            for cached_page in cache:
                try:
                    next_use = pages[i+1:].index(cached_page)
                except ValueError:
                    next_use = float('inf')  # Never used again

                if next_use > farthest:
                    farthest = next_use
                    victim = cached_page

            cache.remove(victim)
            cache.add(page)

    return faults

pages = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1]
print(f"Page faults (OPT): {optimal_page_replacement(pages, 3)}")   # 9
print(f"Page faults (OPT): {optimal_page_replacement(pages, 4)}")   # 6
```

### 8.4 Greedy Approximation: Set Cover Problem

```python
def greedy_set_cover(universe: set, subsets: list) -> list:
    """Greedy approximation for the Set Cover Problem - O(|U| * |S|)
    NP-hard, but greedy achieves an approximation within ln(|U|)+1 factor
    """
    uncovered = set(universe)
    selected = []

    while uncovered:
        # Select the set that covers the most uncovered elements
        best = max(range(len(subsets)),
                   key=lambda i: len(subsets[i] & uncovered) if i not in selected else 0)

        if not (subsets[best] & uncovered):
            break  # No further coverage possible

        selected.append(best)
        uncovered -= subsets[best]

    return selected

universe = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
subsets = [
    {1, 2, 3, 8},      # S0
    {1, 2, 3, 4, 5},   # S1
    {4, 5, 7},          # S2
    {5, 6, 7},          # S3
    {6, 7, 8, 9, 10},  # S4
]
selected = greedy_set_cover(universe, subsets)
print(f"Selected sets: {selected}")  # [1, 4] or similar
```

### 8.5 Greedy Graph Coloring (Approximation)

Graph coloring is NP-hard, but a greedy algorithm can efficiently produce an approximate solution.

```python
def greedy_graph_coloring(graph: dict) -> dict:
    """Greedy graph coloring - O(V + E)
    graph: {node: [neighbors]}
    Does not guarantee optimal, but colors with at most max_degree+1 colors (Brook's theorem)
    """
    colors = {}

    for node in graph:
        # Collect colors used by adjacent vertices
        used_colors = set()
        for neighbor in graph[node]:
            if neighbor in colors:
                used_colors.add(colors[neighbor])

        # Assign the smallest unused color number
        color = 0
        while color in used_colors:
            color += 1
        colors[node] = color

    return colors

# Usage example: Part of the Petersen graph
graph = {
    0: [1, 4, 5],
    1: [0, 2, 6],
    2: [1, 3, 7],
    3: [2, 4, 8],
    4: [3, 0, 9],
    5: [0, 7, 8],
    6: [1, 8, 9],
    7: [2, 5, 9],
    8: [3, 5, 6],
    9: [4, 6, 7],
}
coloring = greedy_graph_coloring(graph)
num_colors = len(set(coloring.values()))
print(f"Coloring result: {coloring}")
print(f"Number of colors used: {num_colors}")

# Verify validity of coloring
valid = all(
    coloring[u] != coloring[v]
    for u in graph for v in graph[u]
)
print(f"Coloring is valid: {valid}")  # True
```

### 8.6 Optimal Merge Pattern (File Merging)

The problem of merging multiple sorted files two at a time into one. It has the same structure as Huffman coding.

```python
import heapq

def optimal_merge_pattern(file_sizes: list) -> tuple:
    """Optimal Merge Pattern - O(n log n)
    At each step, merge the two smallest files (same strategy as Huffman)
    Returns: (total merge cost, merge order)
    """
    heap = list(file_sizes)
    heapq.heapify(heap)
    total_cost = 0
    merge_order = []

    while len(heap) > 1:
        first = heapq.heappop(heap)
        second = heapq.heappop(heap)
        merged = first + second
        total_cost += merged
        merge_order.append((first, second, merged))
        heapq.heappush(heap, merged)

    return total_cost, merge_order

# Usage example
sizes = [20, 30, 10, 5, 30]
cost, order = optimal_merge_pattern(sizes)
print(f"Minimum merge cost: {cost}")
for f1, f2, merged in order:
    print(f"  {f1} + {f2} = {merged}")
# Example output:
# 5 + 10 = 15
# 15 + 20 = 35
# 30 + 30 = 60
# 35 + 60 = 95
# Total cost = 15 + 35 + 60 + 95 = 205
```

---

## 9. Anti-patterns

### Anti-pattern 1: Unproven Greedy Choice Property

```python
# BAD: Applying greedy because "it seems intuitively correct"
# Coin problem: denominations = [1, 3, 4], amount = 6
# Greedy (largest first): 4 + 1 + 1 = 3 coins
# Optimal: 3 + 3 = 2 coins  <- Greedy fails!

def bad_coin_greedy(coins, amount):
    coins.sort(reverse=True)
    count = 0
    for coin in coins:
        count += amount // coin
        amount %= coin
    return count

print(bad_coin_greedy([1, 3, 4], 6))  # 3 (incorrect, answer is 2)

# GOOD: Use DP for this problem
def good_coin_dp(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for c in coins:
            if c <= i:
                dp[i] = min(dp[i], dp[i - c] + 1)
    return dp[amount]

print(good_coin_dp([1, 3, 4], 6))  # 2 (correct)
```

### Anti-pattern 2: Greedy for 0/1 Knapsack

```python
# BAD: Applying unit-value greedy to 0/1 Knapsack
# Items: (weight=10, value=60), (weight=20, value=100), (weight=30, value=120)
# Capacity: 50
# Greedy (by unit value): 60 + 100 = 160
# Optimal: 100 + 120 = 220  <- weight 20+30=50 fits

# GOOD: Use DP for 0/1 Knapsack
```

### Anti-pattern 3: Wrong Sorting Criterion

```python
# BAD: Sorting by "start time" in activity selection -> Not optimal
def bad_activity_selection(activities):
    activities.sort(key=lambda x: x[0])  # Sort by start time -> incorrect
    ...

# GOOD: Sort by "finish time"
def good_activity_selection(activities):
    activities.sort(key=lambda x: x[1])  # Sort by finish time -> correct
    ...
```

### Anti-pattern 4: Confusing "Approximation" with "Optimal" in Greedy

```python
# BAD: Using greedy for the Set Cover Problem and claiming "optimal solution obtained"
# -> Greedy gives an approximate solution, not necessarily optimal

# GOOD: State the approximation ratio explicitly
# "Approximate solution by greedy. Guaranteed within ln(n)+1 factor of optimal."
```

### Anti-pattern 5: The Trap of Local Optima -- Traveling Salesman Problem

```python
# BAD: Applying nearest neighbor (greedy) to TSP and claiming "optimal"
def bad_tsp_nearest_neighbor(dist_matrix: list, start: int = 0) -> tuple:
    """Nearest neighbor is a greedy heuristic that does not guarantee optimality"""
    n = len(dist_matrix)
    visited = [False] * n
    visited[start] = True
    tour = [start]
    total = 0
    current = start

    for _ in range(n - 1):
        nearest = -1
        nearest_dist = float('inf')
        for j in range(n):
            if not visited[j] and dist_matrix[current][j] < nearest_dist:
                nearest = j
                nearest_dist = dist_matrix[current][j]
        visited[nearest] = True
        tour.append(nearest)
        total += nearest_dist
        current = nearest

    total += dist_matrix[current][start]
    tour.append(start)
    return total, tour

# Distance matrix demonstrating a counterexample
dist = [
    [0, 1, 15, 6],
    [1, 0, 7, 3],
    [15, 7, 0, 1],
    [6, 3, 1, 0],
]
greedy_cost, greedy_tour = bad_tsp_nearest_neighbor(dist, 0)
print(f"Nearest neighbor: cost={greedy_cost}, tour={greedy_tour}")
# The optimal tour may be different

# GOOD: For TSP, use exact methods (e.g., branch and bound) or
# improve with local search such as 2-opt
```

### Anti-pattern 6: Applying Dijkstra to Graphs with Negative Weights

```python
# BAD: Applying Dijkstra (greedy) to a graph with negative edges
# -> The greedy choice property breaks down with negative edges
#
# Example:  A --1--> B --(-3)--> C
#           A --2--> C
#
# Dijkstra: A->B(1), A->C(2) -> Distance to C = 2
# Correct:  A->B->C = 1+(-3) = -2
#
# GOOD: Use Bellman-Ford when negative edges are present

def bellman_ford(n: int, edges: list, source: int) -> list:
    """Shortest paths in a graph with negative edges - O(VE)"""
    dist = [float('inf')] * n
    dist[source] = 0

    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w

    # Detect negative cycles
    for u, v, w in edges:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            raise ValueError("Negative cycle detected")

    return dist

edges = [(0, 1, 1), (1, 2, -3), (0, 2, 2)]
dist = bellman_ford(3, edges, 0)
print(f"Shortest distances: {dist}")  # [0, 1, -2]
```

---

## 10. When Greedy Works for the Coin Problem

```
Conditions under which greedy (using the largest denomination first) is optimal
for the coin problem:

1. The denominations form a canonical coin system
   Example: [1, 5, 10, 25, 50, 100] (US dollars) -> Greedy is optimal
   Example: [1, 5, 10, 50, 100, 500] (Japanese yen) -> Greedy is optimal

2. Non-canonical examples:
   [1, 3, 4]     -> Greedy fails for amount 6
   [1, 5, 6, 9]  -> Greedy fails for amount 11 (9+1+1=3 coins), optimal is (6+5=2 coins)

How to determine:
   Verify for all combinations of denominations, or
   use Pearson's (2005) algorithm for polynomial-time verification
```

---

## 11. Complexity Analysis Patterns for Greedy Algorithms

In most cases, sorting is the bottleneck of greedy algorithms.

```
+------------------------------------------------------------+
|       Complexity Patterns for Greedy Algorithms             |
+------------------+-----------------+-----------------------+
| Algorithm        | Time Complexity | Bottleneck            |
+------------------+-----------------+-----------------------+
| Activity Select. | O(n log n)      | Sorting               |
| Frac. Knapsack   | O(n log n)      | Sorting               |
| Huffman Coding   | O(n log n)      | Heap operations       |
| Kruskal          | O(E log E)      | Sorting + Union-Find  |
| Prim (heap)      | O((V+E) log V)  | Heap operations       |
| Prim (array)     | O(V^2)          | Minimum search        |
| Boruvka          | O(E log V)      | Phase iteration       |
| Dijkstra (heap)  | O((V+E) log V)  | Heap operations       |
| Interval Sched.  | O(n log n)      | Sorting               |
| Job Scheduling   | O(n^2)          | Slot search           |
| Set Cover(approx)| O(|U| * |S|)   | Set traversal         |
| Graph Color(appr)| O(V + E)       | Adjacency list trav.  |
+------------------+-----------------+-----------------------+

Key points:
  - The most common pattern is sorting at O(n log n) being dominant
  - Using heaps allows dynamic min/max retrieval in O(log n)
  - The greedy loop itself is often O(n)
  - Preprocessing (sorting or heap construction) determines overall complexity
```

---

## 12. Exercises

### Foundation Level

**Problem B1: Making Change**

Write a program that makes change using Japanese coins [500, 100, 50, 10, 5, 1] with the minimum number of coins. Verify that the greedy algorithm is optimal for these denominations.

```python
def min_coins_japan(amount: int) -> dict:
    """Make change with minimum coins using Japanese denominations"""
    coins = [500, 100, 50, 10, 5, 1]
    result = {}
    remaining = amount

    for coin in coins:
        if remaining >= coin:
            count = remaining // coin
            result[coin] = count
            remaining -= coin * count

    return result

# Test
change = min_coins_japan(1376)
print(f"Change for 1376 yen: {change}")
# {500: 2, 100: 3, 50: 1, 10: 2, 5: 1, 1: 1}
total_coins = sum(change.values())
print(f"Total coins: {total_coins}")  # 10
```

**Problem B2: Meeting Room Allocation**

Given start and end times of N meetings, find the maximum number of meetings that can be held in one room.

```python
def max_meetings(meetings: list) -> list:
    """Meeting room allocation (application of activity selection)"""
    indexed = [(end, start, i) for i, (start, end) in enumerate(meetings)]
    indexed.sort()

    selected = []
    last_end = -1

    for end, start, idx in indexed:
        if start >= last_end:
            selected.append((idx, start, end))
            last_end = end

    return selected

meetings = [(0, 6), (1, 4), (3, 5), (5, 7), (5, 9), (8, 9)]
result = max_meetings(meetings)
print(f"Number of meetings possible: {len(result)}")
for idx, s, e in result:
    print(f"  Meeting {idx}: {s}-{e}")
# Meeting 1: 1-4, Meeting 3: 5-7, Meeting 5: 8-9 -> 3 meetings
```

**Problem B3: Maximum Allocation Problem**

Distribute candies to children. Each child has a satisfaction threshold, and each candy has a size. Give one candy per child and maximize the number of satisfied children.

```python
def assign_cookies(children: list, cookies: list) -> int:
    """Candy distribution problem - O(n log n + m log m)
    children: satisfaction threshold for each child
    cookies: size of each candy
    """
    children_sorted = sorted(children)
    cookies_sorted = sorted(cookies)

    child_i = 0
    cookie_i = 0

    while child_i < len(children_sorted) and cookie_i < len(cookies_sorted):
        if cookies_sorted[cookie_i] >= children_sorted[child_i]:
            child_i += 1  # This child is satisfied
        cookie_i += 1  # Next candy

    return child_i

children = [1, 2, 3]
cookies = [1, 1]
print(f"Satisfied children: {assign_cookies(children, cookies)}")  # 1

children = [1, 2]
cookies = [1, 2, 3]
print(f"Satisfied children: {assign_cookies(children, cookies)}")  # 2
```

### Intermediate Level

**Problem A1: Minimum Number of Platforms**

Given arrival and departure times of trains, find the minimum number of platforms needed to accommodate all trains stopped at the same time.

```python
def min_platforms(arrivals: list, departures: list) -> int:
    """Minimum number of platforms - O(n log n)
    Greedy algorithm using event sorting
    """
    events = []
    for a in arrivals:
        events.append((a, 1))   # Arrival: +1
    for d in departures:
        events.append((d, -1))  # Departure: -1

    events.sort(key=lambda x: (x[0], x[1]))  # At same time, process departures first

    current = 0
    max_platforms = 0

    for _, event_type in events:
        current += event_type
        max_platforms = max(max_platforms, current)

    return max_platforms

arrivals   = [900, 940, 950, 1100, 1500, 1800]
departures = [910, 1200, 1120, 1130, 1900, 2000]
print(f"Minimum platforms: {min_platforms(arrivals, departures)}")  # 3
```

**Problem A2: Lexicographically Smallest String**

Given a string s, build a new string by appending each character of s to either the front or back. Find the lexicographically smallest result.

```python
def smallest_string_by_appending(s: str) -> str:
    """Greedy algorithm to build the lexicographically smallest string - O(n^2)
    Append each character to the front or back
    """
    from collections import deque
    result = deque()

    for char in s:
        if result and char < result[0]:
            result.appendleft(char)
        else:
            result.append(char)

    return ''.join(result)

# More refined solution: compare with the remaining string
def smallest_string_precise(s: str) -> str:
    """Build the lexicographically smallest string (precise version) - O(n^2)"""
    from collections import deque
    result = deque()
    n = len(s)
    left = 0
    right = n - 1

    chars = list(s)

    while left <= right:
        if chars[left] < chars[right]:
            result.append(chars[left])
            left += 1
        elif chars[left] > chars[right]:
            result.append(chars[right])
            right -= 1
        else:
            # If equal, compare inner characters to decide
            l, r = left, right
            while l < r and chars[l] == chars[r]:
                l += 1
                r -= 1
            if l >= r or chars[l] < chars[r]:
                result.append(chars[left])
                left += 1
            else:
                result.append(chars[right])
                right -= 1

    return ''.join(result)

print(smallest_string_by_appending("ACBDFE"))
print(smallest_string_precise("CBABC"))  # Correct lexicographically smallest
```

**Problem A3: Minimum Group Partition of Intervals**

Given N intervals, partition them into the minimum number of groups such that overlapping intervals are not in the same group (minimum coloring of an interval graph = maximum overlap count).

```python
def min_groups(intervals: list) -> int:
    """Minimum group partition of intervals - O(n log n)
    Find the maximum simultaneous overlap (= minimum groups = chromatic number of interval graph)
    """
    events = []
    for start, end in intervals:
        events.append((start, 1))
        events.append((end, -1))

    events.sort(key=lambda x: (x[0], x[1]))

    current = 0
    max_overlap = 0
    for _, delta in events:
        current += delta
        max_overlap = max(max_overlap, current)

    return max_overlap

intervals = [(1, 5), (2, 6), (4, 7), (6, 8), (7, 10)]
print(f"Minimum groups: {min_groups(intervals)}")  # 3
```

### Advanced Level

**Problem E1: Matroid Intersection (Advanced Topic)**

Consider the problem of finding the maximum-size common independent set of two matroids. Below is an example formulating bipartite matching as matroid intersection.

```python
def bipartite_matching_as_matroid_intersection(
    left_nodes: list, right_nodes: list, edges: list
) -> list:
    """Bipartite matching (concrete example of matroid intersection)
    Implemented here using augmenting paths (special case of matroid intersection)
    """
    match_left = {}
    match_right = {}

    def augment(u, visited):
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                if v not in match_right or augment(match_right[v], visited):
                    match_left[u] = v
                    match_right[v] = u
                    return True
        return False

    adj = {u: [] for u in left_nodes}
    for u, v in edges:
        adj[u].append(v)

    for u in left_nodes:
        augment(u, set())

    return list(match_left.items())

left = ['a', 'b', 'c']
right = ['x', 'y', 'z']
edges = [('a','x'), ('a','y'), ('b','x'), ('b','z'), ('c','y')]
matching = bipartite_matching_as_matroid_intersection(left, right, edges)
print(f"Maximum matching: {matching}")
print(f"Matching size: {len(matching)}")
# By matroid intersection theory, this can be solved as a generalization of greedy
```

**Problem E2: Online Greedy -- Secretary Problem**

Interview n candidates in sequence, making an immediate hire/reject decision after each interview. A rejected candidate cannot be recalled. Implement the strategy that maximizes the probability of selecting the best candidate.

```python
import random
import math

def secretary_problem_strategy(candidates: list) -> tuple:
    """Optimal strategy for the Secretary Problem (1/e strategy)
    Observe the first n/e candidates (set a baseline)
    Then hire the first candidate that exceeds the baseline

    The probability of selecting the best candidate converges to 1/e ~ 36.8%
    """
    n = len(candidates)
    # Observe the first n/e candidates (exploration phase)
    observe_count = max(1, int(n / math.e))

    # Record the highest score during the observation phase
    threshold = max(candidates[:observe_count])

    # Decision phase: hire the first candidate exceeding the threshold
    for i in range(observe_count, n):
        if candidates[i] > threshold:
            return candidates[i], i, True  # (score, index, hired)

    # If no one exceeds the threshold, hire the last candidate
    return candidates[-1], n - 1, False

# Simulation
def simulate_secretary(n: int, trials: int = 10000) -> float:
    """Simulation of the Secretary Problem"""
    successes = 0

    for _ in range(trials):
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        best = max(candidates)

        chosen, _, _ = secretary_problem_strategy(candidates)
        if chosen == best:
            successes += 1

    return successes / trials

random.seed(42)
for n in [10, 50, 100, 1000]:
    success_rate = simulate_secretary(n)
    print(f"n={n:4d}: success rate = {success_rate:.3f} (theoretical ~ {1/math.e:.3f})")
```

**Problem E3: Competitive Ratio Analysis of Greedy -- Online Ski Rental Problem**

You can either rent skis (r yen per day) or buy them (p yen). How to decide when you do not know how many days you will ski? Analyze the online strategy.

```python
def ski_rental_deterministic(daily_rent: int, purchase_price: int,
                              actual_days: int) -> dict:
    """Deterministic strategy for the Ski Rental Problem
    Switch from renting to buying on the break-even day (p/r days)
    Competitive ratio: 2 - r/p (within 2x of optimal in worst case)
    """
    break_even = purchase_price // daily_rent

    # Strategy: buy on the break-even day
    if actual_days <= break_even:
        # Trip ends before purchase -> rent only
        online_cost = actual_days * daily_rent
    else:
        # Rent for break_even days + purchase
        online_cost = break_even * daily_rent + purchase_price

    # Optimal solution (determined in hindsight)
    optimal_cost = min(actual_days * daily_rent, purchase_price)

    return {
        'online_cost': online_cost,
        'optimal_cost': optimal_cost,
        'competitive_ratio': online_cost / optimal_cost if optimal_cost > 0 else float('inf'),
        'strategy': f"{'Rent only' if actual_days <= break_even else f'Buy on day {break_even}'}"
    }

# Usage example
rent = 100
price = 1000

for days in [5, 10, 15, 20, 50]:
    result = ski_rental_deterministic(rent, price, days)
    print(f"Days={days:2d}: online={result['online_cost']:5d}, "
          f"optimal={result['optimal_cost']:5d}, "
          f"competitive ratio={result['competitive_ratio']:.2f}, "
          f"strategy={result['strategy']}")
```

---

## 13. Design Pattern Classification for Greedy Algorithms

A summary of typical design patterns encountered in greedy algorithms.

| Pattern Name | Greedy Criterion | Representative Problem | Complexity |
|:---|:---|:---|:---|
| Endpoint Sort | Sort by finish time/deadline | Activity Selection, EDF | O(n log n) |
| Ratio Sort | Sort by value/cost | Fractional Knapsack | O(n log n) |
| Minimum Merge | Merge minimum elements in pairs | Huffman Coding, Merge Pattern | O(n log n) |
| Minimum Edge Selection | Minimum edge in graph | Kruskal, Boruvka | O(E log E) |
| Nearest Neighbor Expansion | Adjacent minimum cost | Prim, Dijkstra | O((V+E)log V) |
| Event Sweep | Scan timeline | Minimum Platforms | O(n log n) |
| Farthest-first | Select farthest element | k-center approximation | O(nk) |
| Maximum Margin | Select with most slack | Secretary Problem | O(n) |

---

## 14. FAQ

### Q1: How do you prove the correctness of a greedy algorithm?

**A:** There are two main proof techniques. (1) **Exchange argument**: Assume the optimal solution differs from the greedy solution, and show that replacing elements with greedy choices maintains (or improves) optimality. (2) **Matroid theory**: If the problem structure satisfies the matroid axioms, the greedy algorithm is optimal. In practice, the common approach is to search for counterexamples, and if none are found, prove correctness via exchange argument.

### Q2: What is the difference between a greedy algorithm and a heuristic?

**A:** A greedy algorithm guarantees an optimal solution when its correctness is proven. A heuristic is a method for quickly obtaining an approximate solution without optimality guarantees. When a greedy algorithm is applied to a problem where the greedy choice property does not hold, it becomes a (potentially inaccurate) heuristic.

### Q3: When should Prim's vs Kruskal's be used?

**A:** Both are greedy algorithms for finding the minimum spanning tree. Kruskal is edge-based (E log E) and works well on sparse graphs. Prim is vertex-based (V log V + E with priority queue) and works well on dense graphs. Use Kruskal when the number of edges is small, and Prim when there are many edges.

### Q4: Is Dijkstra's algorithm a greedy algorithm?

**A:** Yes. Dijkstra's algorithm repeats the greedy choice of "finalizing the unvisited vertex with minimum distance." Under the condition of non-negative edges, this greedy choice is proven to be optimal. With negative edges, the greedy choice property breaks down, requiring Bellman-Ford (DP-based).

### Q5: When should greedy algorithms NOT be used?

**A:** (1) When a counterexample is found. (2) When the problem is NP-hard and an optimal solution is required (greedy gives only an approximation). (3) When a choice affects future options (e.g., 0/1 knapsack). (4) When you need to enumerate "all solutions" rather than find the "optimal" one.

### Q6: What is the relationship between greedy algorithms and beam search?

**A:** Beam search is an extension of greedy algorithms. While greedy maintains only one best candidate at each step, beam search maintains the top k candidates. k=1 is exactly greedy. Increasing k improves solution quality but increases computation. It is widely used in decoding for natural language processing.

### Q7: Can greedy algorithms be used as online algorithms?

**A:** In some cases, yes. An online algorithm receives input sequentially and makes irrevocable decisions at each point. The "no backtracking" property of greedy algorithms has strong affinity with online algorithms. The 1/e strategy for the Secretary Problem and the break-even strategy for the Ski Rental Problem are representative examples. However, in the online setting, algorithms are evaluated by their "competitive ratio" (worst-case multiple compared to optimal) rather than optimality.

### Q8: How do you debug greedy algorithms?

**A:** Bugs in greedy algorithms fall into two categories: "the algorithm is correct but the implementation is wrong" and "greedy is not applicable to the problem in the first place." The former is addressed by checking the sorting criterion and edge cases (empty input, handling of equal values). The latter is identified by comparing results with brute-force on small test cases to find counterexamples. Below is test code for debugging.

```python
import itertools

def verify_greedy(greedy_func, brute_force_func, test_cases):
    """Verify greedy results by comparing with brute-force"""
    for i, test_input in enumerate(test_cases):
        greedy_result = greedy_func(test_input)
        bf_result = brute_force_func(test_input)
        if greedy_result != bf_result:
            print(f"Counterexample found! Test {i}: input={test_input}")
            print(f"  Greedy: {greedy_result}")
            print(f"  Brute-force: {bf_result}")
            return False
    print("All tests passed")
    return True
```

### Q9: Are greedy algorithms suitable for parallel processing?

**A:** It depends on the problem. Algorithms like Boruvka's, where processing of each component is independent, have high parallelism. On the other hand, sequential greedy algorithms like activity selection, which depend on previous choices, are difficult to parallelize. When computing MST of large-scale graphs using the MapReduce framework, Boruvka is preferred.

---


## FAQ

### Q1: What is the most important point in learning this topic?

Gaining practical experience is the most important aspect. Understanding deepens not just from theory but from actually writing code and observing its behavior.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving to the next step.

### Q3: How is this knowledge applied in practice?

The knowledge of this topic is frequently utilized in daily development work. It becomes especially important during code reviews and architecture design.

---

## 15. Summary

| Item | Key Points |
|:---|:---|
| Conditions for Greedy | Greedy choice property + Optimal substructure |
| Activity Selection | Select by finish time. Foundation of interval scheduling |
| Huffman Coding | Merge least frequent pairs. Optimal prefix code |
| Kruskal | Add edges by weight. Cycle detection with Union-Find |
| Prim | Build MST vertex by vertex. Advantageous for dense graphs |
| Boruvka | MST algorithm well-suited for parallel processing |
| Fractional Knapsack | Select by unit value. Unlike 0/1, greedy is optimal |
| Matroid | Theoretical framework guaranteeing greedy correctness |
| Choosing between Greedy and DP | Use greedy if applicable (faster); otherwise DP |
| Correctness proof | Proof via exchange argument or matroid theory is essential |
| Approximation algorithms | Greedy often provides good approximations for NP-hard problems |
| Online setting | Competitive ratio analysis can guarantee quality of greedy strategies |

---

## Recommended Next Guides

- [Dynamic Programming](./04-dynamic-programming.md) -- Handling problems unsolvable by greedy
- [Divide and Conquer](./06-divide-conquer.md) -- Another design paradigm
- [Union-Find](../03-advanced/00-union-find.md) -- Data structure essential for Kruskal

---

## References

1. Cormen, T. H. et al. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. -- Chapter 15: Greedy Algorithms
2. Huffman, D. A. (1952). "A Method for the Construction of Minimum-Redundancy Codes." *Proceedings of the IRE*, 40(9), 1098-1101.
3. Kruskal, J. B. (1956). "On the shortest spanning subtree of a graph and the traveling salesman problem." *Proceedings of the AMS*, 7(1), 48-50.
4. Prim, R. C. (1957). "Shortest connection networks and some generalizations." *Bell System Technical Journal*, 36(6), 1389-1401.
5. Kleinberg, J. & Tardos, E. (2005). *Algorithm Design*. Pearson. -- Chapter 4: Greedy Algorithms
6. Oxley, J. G. (2011). *Matroid Theory* (2nd ed.). Oxford University Press.
7. Borodin, A. & El-Yaniv, R. (1998). *Online Computation and Competitive Analysis*. Cambridge University Press. -- Theory of online greedy algorithms and competitive analysis
8. Vazirani, V. V. (2001). *Approximation Algorithms*. Springer. -- Systematic treatment of greedy approximation algorithms
9. Lawler, E. L. (1976). *Combinatorial Optimization: Networks and Matroids*. Holt, Rinehart and Winston. -- Classic reference on matroids and greedy algorithms
