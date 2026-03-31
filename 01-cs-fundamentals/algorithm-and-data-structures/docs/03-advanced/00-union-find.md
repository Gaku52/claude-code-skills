# Union-Find (Disjoint Set Data Structure)

> Understand the data structure that performs element grouping and membership queries in nearly constant time, through path compression, union by rank, and Kruskal's algorithm applications

## What You Will Learn in This Chapter

1. **The two operations of Union-Find** (Find/Union) and their optimization via path compression and union by rank
2. **The inverse Ackermann function** alpha(n) and the nearly constant-time complexity it yields
3. **Practical application patterns** such as Kruskal's algorithm, connected components, and equivalence class partitioning
4. **Advanced variants** including weighted Union-Find and persistent Union-Find
5. **Real-world applications** in clustering and network management


## Prerequisites

Before reading this guide, the following knowledge will help deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts

---

## 1. Concept of Union-Find

Union-Find (Disjoint Set Union: DSU) is a data structure that manages a collection of elements partitioned into non-overlapping (disjoint) subsets. It primarily provides the following two operations:

- **Find(x)**: Returns the representative (root) of the set to which element x belongs
- **Union(x, y)**: Merges the sets containing elements x and y

```
Union-Find = Disjoint Set Forest

Initial state (each element is independent):
  {0} {1} {2} {3} {4} {5} {6} {7}

Union(0,1), Union(2,3), Union(4,5):
  {0,1} {2,3} {4,5} {6} {7}

Union(0,2), Union(4,6):
  {0,1,2,3} {4,5,6} {7}

Union(0,4):
  {0,1,2,3,4,5,6} {7}

Find(5) → 0 (representative)
Find(7) → 7 (representative)
Find(5) == Find(3) → True  (same set)
Find(5) == Find(7) → False (different sets)
```

### Why Union-Find Matters

Despite its apparent simplicity, Union-Find is an extremely powerful data structure in the following scenarios:

1. **Dynamic connectivity queries**: When edges are added incrementally to a graph, answering "Are vertices u and v connected?" in nearly O(1)
2. **Minimum Spanning Tree (MST)**: Essential for cycle detection in Kruskal's algorithm
3. **Equivalence class management**: Efficiently maintaining the transitive closure of equivalence relations
4. **Image processing**: Labeling (identifying connected components)
5. **Network design**: Link redundancy verification, routing

```
Real-world examples:

1. Social networks
   "Are person A and person B connected (via friends of friends)?"
   → Manage friendships with Union-Find → O(alpha(n)) ≈ O(1) query with Find

2. Computer networks
   "Can machine X communicate with machine Y?"
   → Register cable connections with Union → Check reachability with Find

3. Sudoku region management
   "Do cell A and cell B belong to the same block?"

4. Compiler type inference
   "Do type variables alpha and beta resolve to the same type?"
```

---

## 2. Tree-Based Representation

Internally, Union-Find is represented as a forest (a collection of trees). Each set corresponds to one tree, and the root of the tree serves as the representative of that set.

```
Naive implementation (tree becomes skewed):

Union(0,1): 1→0      Union(0,1,2,3,4):
Union(0,2): 2→0          0
Union(0,3): 3→0         /|\ \
Union(0,4): 4→0        1 2 3 4    ← Well balanced

Worst case (chain-shaped):
  Union(0,1), Union(1,2), Union(2,3), Union(3,4)
    0 ← 1 ← 2 ← 3 ← 4    ← Find(4) is O(n)!

After path compression:
         0
       / | \ \
      1  2  3  4    ← Find(4) is O(1)!
```

### Array Representation

Union-Find is represented using a `parent` array. `parent[i]` stores the parent of element i, and for roots, `parent[i] = i`.

```
Array representation example:

Managing sets {0,1,2,3} and {4,5,6}:

parent: [0, 0, 0, 0, 4, 4, 4]
         ↑     ↑           ↑
        root  child of 0  root

Tree structure:
    0       4
   /|\     / \
  1 2 3   5   6

Find(3) → parent[3] = 0 → parent[0] = 0 → root is 0
Find(6) → parent[6] = 4 → parent[4] = 4 → root is 4
```

---

## 3. Basic Implementation

```python
class UnionFind:
    """Union-Find (Disjoint Set Data Structure)
    Path compression + union by rank for O(alpha(n)) ≈ O(1)
    """

    def __init__(self, n: int):
        self.parent = list(range(n))  # Parent of each element
        self.rank = [0] * n           # Upper bound on tree height
        self.size = [1] * n           # Size of each set
        self.count = n                # Number of sets

    def find(self, x: int) -> int:
        """Returns the representative (root) of x - with path compression"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """Merges the sets of x and y - union by rank
        Returns: Whether a merge was performed (False if already in the same set)
        """
        px, py = self.find(x), self.find(y)
        if px == py:
            return False

        # Attach the tree with lower rank under the one with higher rank
        if self.rank[px] < self.rank[py]:
            px, py = py, px

        self.parent[py] = px
        self.size[px] += self.size[py]

        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

        self.count -= 1
        return True

    def connected(self, x: int, y: int) -> bool:
        """Whether x and y belong to the same set"""
        return self.find(x) == self.find(y)

    def get_size(self, x: int) -> int:
        """Size of the set containing x"""
        return self.size[self.find(x)]

    def get_groups(self) -> dict:
        """Returns all groups as a dictionary {representative: [element list]}"""
        groups = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)
        return groups

# Usage example
uf = UnionFind(8)
uf.union(0, 1)
uf.union(2, 3)
uf.union(0, 2)
print(uf.connected(1, 3))  # True
print(uf.connected(0, 5))  # False
print(uf.get_size(0))      # 4
print(uf.count)             # 5 (number of sets)
print(uf.get_groups())     # {0: [0, 1, 2, 3], 4: [4], 5: [5], 6: [6], 7: [7]}
```

### C++ Implementation

```cpp
#include <vector>
#include <numeric>

class UnionFind {
    std::vector<int> parent, rank_, size_;
    int count_;

public:
    UnionFind(int n) : parent(n), rank_(n, 0), size_(n, 1), count_(n) {
        std::iota(parent.begin(), parent.end(), 0);
    }

    int find(int x) {
        if (parent[x] != x)
            parent[x] = find(parent[x]);  // Path compression
        return parent[x];
    }

    bool unite(int x, int y) {
        int px = find(x), py = find(y);
        if (px == py) return false;

        if (rank_[px] < rank_[py]) std::swap(px, py);
        parent[py] = px;
        size_[px] += size_[py];
        if (rank_[px] == rank_[py]) rank_[px]++;
        count_--;
        return true;
    }

    bool connected(int x, int y) { return find(x) == find(y); }
    int getSize(int x) { return size_[find(x)]; }
    int getCount() const { return count_; }
};

// Usage example
// UnionFind uf(8);
// uf.unite(0, 1);
// uf.unite(2, 3);
// cout << uf.connected(1, 3) << endl;  // 0 (false)
// uf.unite(0, 2);
// cout << uf.connected(1, 3) << endl;  // 1 (true)
```

### TypeScript Implementation

```typescript
class UnionFind {
    private parent: number[];
    private rank: number[];
    private size: number[];
    private _count: number;

    constructor(n: number) {
        this.parent = Array.from({ length: n }, (_, i) => i);
        this.rank = new Array(n).fill(0);
        this.size = new Array(n).fill(1);
        this._count = n;
    }

    find(x: number): number {
        if (this.parent[x] !== x) {
            this.parent[x] = this.find(this.parent[x]);
        }
        return this.parent[x];
    }

    union(x: number, y: number): boolean {
        let px = this.find(x);
        let py = this.find(y);
        if (px === py) return false;

        if (this.rank[px] < this.rank[py]) [px, py] = [py, px];
        this.parent[py] = px;
        this.size[px] += this.size[py];
        if (this.rank[px] === this.rank[py]) this.rank[px]++;
        this._count--;
        return true;
    }

    connected(x: number, y: number): boolean {
        return this.find(x) === this.find(y);
    }

    getSize(x: number): number {
        return this.size[this.find(x)];
    }

    get count(): number {
        return this._count;
    }
}
```

---

## 4. Path Compression in Detail

Path compression is an optimization technique applied during the Find operation that directly connects all nodes along the path to the root. This speeds up subsequent Find operations.

```
Path compression during Find(7):

Before:           After:
    0                 0
    |              / | \ \
    1             1  3  5  7
    |
    3
    |
    5
    |
    7

Find(7): 7→5→3→1→0  (root found)
Compress: 7→0, 5→0, 3→0, 1→0  (all nodes directly under root)
```

```python
# Three methods of path compression

# Method 1: Recursive (implementation above) - full path compression
def find_recursive(self, x):
    if self.parent[x] != x:
        self.parent[x] = self.find(self.parent[x])
    return self.parent[x]

# Method 2: Iterative (avoids stack overflow) - full path compression
def find_iterative(self, x):
    root = x
    while self.parent[root] != root:
        root = self.parent[root]
    # Connect all nodes along the path directly to the root
    while self.parent[x] != root:
        next_x = self.parent[x]
        self.parent[x] = root
        x = next_x
    return root

# Method 3: Path splitting - concise implementation
def find_splitting(self, x):
    while self.parent[x] != x:
        self.parent[x] = self.parent[self.parent[x]]  # Connect to grandparent
        x = self.parent[x]
    return x

# Method 4: Path halving - variant of path splitting
def find_halving(self, x):
    while self.parent[x] != x:
        self.parent[x] = self.parent[self.parent[x]]  # Connect to grandparent
        x = self.parent[x]  # Advance two steps
    return x
```

### Visualizing the Effect of Path Compression

```python
def visualize_compression():
    """Demo to visualize the effect of path compression"""
    uf = UnionFind(10)

    # Create a chain-shaped tree
    # 0 ← 1 ← 2 ← 3 ← 4 ← 5 ← 6 ← 7 ← 8 ← 9
    for i in range(9):
        uf.parent[i + 1] = i  # Forcibly create a chain
    uf.rank[0] = 9
    uf.count = 1
    for i in range(10):
        uf.size[i] = 1
    uf.size[0] = 10

    print("Before compression:")
    print(f"  parent = {uf.parent}")
    # [0, 0, 1, 2, 3, 4, 5, 6, 7, 8]

    # Path compression triggered by Find(9)
    root = uf.find(9)
    print(f"\nAfter find(9):")
    print(f"  root = {root}")
    print(f"  parent = {uf.parent}")
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # All nodes are now directly connected to the root

visualize_compression()
```

---

## 5. Union by Rank and Union by Size

### Union by Rank

Uses tree height (rank) as the criterion, attaching the shorter tree under the taller one. This keeps the tree height bounded at O(log n).

```python
def union_by_rank(self, x, y):
    px, py = self.find(x), self.find(y)
    if px == py:
        return False

    # Attach the tree with lower rank under the one with higher rank
    if self.rank[px] < self.rank[py]:
        px, py = py, px  # px has the higher rank

    self.parent[py] = px
    if self.rank[px] == self.rank[py]:
        self.rank[px] += 1  # Increment rank when equal

    return True
```

### Union by Size

Uses set size as the criterion, attaching the smaller set under the larger one. The implementation is simpler and provides equivalent practical performance to union by rank.

```python
def union_by_size(self, x, y):
    px, py = self.find(x), self.find(y)
    if px == py:
        return False

    # Attach the smaller set under the larger one
    if self.size[px] < self.size[py]:
        px, py = py, px  # px is the larger one

    self.parent[py] = px
    self.size[px] += self.size[py]

    return True
```

### Comparison of Merge Strategies

```
Union by Rank vs Union by Size:

       Union by Rank               Union by Size
  Merges based on tree height   Merges based on set size
  Requires rank array           Requires size array (useful for other purposes too)
  Theoretically optimal         Equivalent practical performance

  Recommendation: Union by Size
  Reason: The size array also supports queries like "How many elements are in this set?"
```

---

## 6. Application to Kruskal's Algorithm

Union-Find is essential for Kruskal's minimum spanning tree algorithm. By performing cycle detection in O(alpha(n)), the overall complexity is O(E log E).

```python
def kruskal_mst(n: int, edges: list) -> tuple:
    """Kruskal's Algorithm - Using Union-Find - O(E log E)
    edges: [(weight, u, v), ...]
    returns: (mst_edges, mst_weight)
    """
    edges.sort()  # Sort by weight
    uf = UnionFind(n)
    mst_edges = []
    mst_weight = 0

    for weight, u, v in edges:
        if not uf.connected(u, v):  # Cycle detection
            uf.union(u, v)
            mst_edges.append((u, v, weight))
            mst_weight += weight

            if len(mst_edges) == n - 1:
                break  # MST complete

    return mst_edges, mst_weight

# Usage example
# Vertices: 0-4, Edges: (weight, source, destination)
edges = [
    (1, 0, 1), (4, 0, 2), (3, 1, 2),
    (2, 1, 3), (5, 2, 3), (7, 2, 4), (6, 3, 4)
]
mst, total = kruskal_mst(5, edges)
print(f"MST edges: {mst}")        # [(0, 1, 1), (1, 3, 2), (1, 2, 3), (3, 4, 6)]
print(f"Total weight: {total}")    # 12
```

### Detailed Trace of Kruskal's Algorithm

```
Graph:
    0 ---1--- 1
    |  \      |  \
    4   3     2    2
    |     \   |      \
    2 ---5--- 3 ---6--- 4
              |
              7
              |
              4 (weight of edge 2-4)

Sorted edges: (1,0,1) (2,1,3) (3,1,2) (4,0,2) (5,2,3) (6,3,4) (7,2,4)

Step 1: (1, 0, 1)  → 0-1 not connected → Union(0,1) ✓  MST: {(0,1,1)}
Step 2: (2, 1, 3)  → 1-3 not connected → Union(1,3) ✓  MST: {(0,1,1), (1,3,2)}
Step 3: (3, 1, 2)  → 1-2 not connected → Union(1,2) ✓  MST: {(0,1,1), (1,3,2), (1,2,3)}
Step 4: (4, 0, 2)  → 0-2 connected     → Skip ✗       (would create a cycle)
Step 5: (5, 2, 3)  → 2-3 connected     → Skip ✗       (would create a cycle)
Step 6: (6, 3, 4)  → 3-4 not connected → Union(3,4) ✓  MST: + (3,4,6)

Result: MST weight = 1 + 2 + 3 + 6 = 12
```

---

## 7. Application Patterns

### Number of Connected Components (Graph)

```python
def count_components(n: int, edges: list) -> int:
    """Number of connected components in a graph"""
    uf = UnionFind(n)
    for u, v in edges:
        uf.union(u, v)
    return uf.count

print(count_components(5, [(0,1), (2,3)]))  # 3 ({0,1}, {2,3}, {4})
```

### Friends of Friends (Social Network Graph)

```python
def friend_groups(n: int, friendships: list) -> list:
    """Returns a list of friend groups"""
    uf = UnionFind(n)
    for a, b in friendships:
        uf.union(a, b)

    groups = {}
    for i in range(n):
        root = uf.find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    return list(groups.values())

friends = [(0,1), (1,2), (3,4)]
print(friend_groups(6, friends))
# [[0, 1, 2], [3, 4], [5]]
```

### Number of Islands (Grid Problem)

```python
def num_islands(grid: list) -> int:
    """Count the number of islands in a 2D grid using Union-Find"""
    if not grid:
        return 0

    rows, cols = len(grid), len(grid[0])
    uf = UnionFind(rows * cols)
    water = 0

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '0':
                water += 1
                continue

            # Union with right and bottom adjacent cells
            for dr, dc in [(0, 1), (1, 0)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < rows and 0 <= nc < cols
                    and grid[nr][nc] == '1'):
                    uf.union(r * cols + c, nr * cols + nc)

    return uf.count - water

grid = [
    ['1','1','0','0','0'],
    ['1','1','0','0','0'],
    ['0','0','1','0','0'],
    ['0','0','0','1','1'],
]
print(num_islands(grid))  # 3
```

### Detecting Redundant Connections

```python
def find_redundant_connection(edges: list) -> tuple:
    """Find the extra edge in a graph that is a tree with one extra edge added
    LeetCode 684: Redundant Connection
    """
    n = len(edges)
    uf = UnionFind(n + 1)

    for u, v in edges:
        if uf.connected(u, v):
            return (u, v)  # This edge creates a cycle = redundant
        uf.union(u, v)

    return None

edges = [(1,2), (1,3), (2,3)]
print(find_redundant_connection(edges))  # (2, 3)

edges = [(1,2), (2,3), (3,4), (1,4), (1,5)]
print(find_redundant_connection(edges))  # (1, 4)
```

### Minimizing the Maximum Edge (Bottleneck Shortest Path)

```python
def min_bottleneck_path(n: int, edges: list, s: int, t: int) -> int:
    """Minimize the maximum edge weight on a path from s to t
    Kruskal-like approach: add edges in order of increasing weight,
    and the weight of the edge when s-t become connected is the answer
    """
    edges.sort(key=lambda e: e[2])  # Sort by weight
    uf = UnionFind(n)

    for u, v, w in edges:
        uf.union(u, v)
        if uf.connected(s, t):
            return w

    return -1  # s-t are unreachable

edges = [(0, 1, 3), (0, 2, 5), (1, 2, 1), (1, 3, 4), (2, 3, 2)]
print(min_bottleneck_path(4, edges, 0, 3))  # 3 (0→1→2→3, max edge = max(3,1,2) = 3)
```

### Dynamic Connectivity and Offline Queries

```python
def process_connectivity_queries(n: int, operations: list) -> list:
    """Process connectivity queries offline
    operations: [('union', u, v), ('query', u, v), ...]
    """
    uf = UnionFind(n)
    results = []

    for op in operations:
        if op[0] == 'union':
            _, u, v = op
            uf.union(u, v)
        elif op[0] == 'query':
            _, u, v = op
            results.append(uf.connected(u, v))
        elif op[0] == 'size':
            _, u = op[0], op[1]
            results.append(uf.get_size(op[1]))

    return results

ops = [
    ('union', 0, 1),
    ('union', 2, 3),
    ('query', 0, 3),   # False
    ('union', 1, 2),
    ('query', 0, 3),   # True
    ('size', 0, None),  # 4
]
# Result: [False, True, 4]
```

### Equivalence Class Partitioning (String Equivalence)

```python
def equivalent_strings(pairs: list, s1: str, s2: str) -> bool:
    """Determine if two strings are equivalent based on character equivalence relations
    LeetCode 839-style approach
    pairs: [(a, b), ...] means character a is equivalent to character b
    """
    uf = UnionFind(26)  # a-z

    for a, b in pairs:
        uf.union(ord(a) - ord('a'), ord(b) - ord('a'))

    if len(s1) != len(s2):
        return False

    for c1, c2 in zip(s1, s2):
        if not uf.connected(ord(c1) - ord('a'), ord(c2) - ord('a')):
            return False

    return True

# 'a' is equivalent to 'b', 'c' is equivalent to 'd'
pairs = [('a', 'b'), ('c', 'd')]
print(equivalent_strings(pairs, "abc", "bac"))  # True
print(equivalent_strings(pairs, "abc", "bae"))  # False
```

---

## 8. Optimization Effect Comparison Table

| Optimization | Find | Union | Notes |
|:---|:---|:---|:---|
| Naive (array) | O(1) | O(n) | Quick-Find |
| Naive (tree) | O(n) | O(n) | Worst case |
| Path compression only | Amortized O(log n) | O(log n) | Significant improvement |
| Union by rank only | O(log n) | O(log n) | Tree height bounded |
| Both (optimal) | Amortized O(alpha(n)) | Amortized O(alpha(n)) | Effectively O(1) |

### Quick-Find and Quick-Union

Quick-Find and Quick-Union are the fundamental implementation approaches for Union-Find.

```python
class QuickFind:
    """Quick-Find: Find is O(1) but Union is O(n)"""

    def __init__(self, n: int):
        self.id = list(range(n))  # Group ID of each element
        self.n = n

    def find(self, x: int) -> int:
        return self.id[x]  # O(1)

    def union(self, x: int, y: int) -> None:
        px, py = self.id[x], self.id[y]
        if px == py:
            return
        # Change all elements belonging to px to py → O(n)
        for i in range(self.n):
            if self.id[i] == px:
                self.id[i] = py

class QuickUnion:
    """Quick-Union: Simple tree structure (no optimizations)"""

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            x = self.parent[x]  # O(n) worst case
        return x

    def union(self, x: int, y: int) -> None:
        self.parent[self.find(x)] = self.find(y)  # O(n) worst case
```

## Union-Find vs Other Approaches

| Approach | Connectivity Query | Merge | List All Components | Use Case |
|:---|:---|:---|:---|:---|
| Union-Find | O(alpha(n)) | O(alpha(n)) | O(n) | Dynamic connectivity |
| BFS/DFS | O(V+E) | - | O(V+E) | Static graphs |
| Adjacency matrix | O(1) | O(n) | O(n^2) | Dense graphs |

### Performance Benchmark

```python
import time

def benchmark_union_find(n: int, ops: int):
    """Measure Union-Find performance"""
    import random

    # Union-Find (optimized)
    uf_good = UnionFind(n)
    start = time.time()
    for _ in range(ops):
        x, y = random.randint(0, n-1), random.randint(0, n-1)
        uf_good.union(x, y)
    for _ in range(ops):
        x, y = random.randint(0, n-1), random.randint(0, n-1)
        uf_good.connected(x, y)
    good_time = time.time() - start

    print(f"n={n}, ops={ops}")
    print(f"  Optimized: {good_time:.3f}s")

# benchmark_union_find(1_000_000, 2_000_000)
# Example result: n=1000000, ops=2000000 → Optimized: 1.2s
```

---

## 9. Weighted Union-Find

Weighted Union-Find is a data structure that manages relative weights (distances/differences) between elements. It can efficiently handle transitive relations such as "The difference between A and B is 5" and "The difference between B and C is 3" implying "The difference between A and C is 8."

```python
class WeightedUnionFind:
    """Weighted Union-Find
    Manages relative weights (distances/differences) between elements
    """
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.weight = [0] * n  # Weight difference from parent

    def find(self, x):
        if self.parent[x] != x:
            root = self.find(self.parent[x])
            self.weight[x] += self.weight[self.parent[x]]
            self.parent[x] = root
        return self.parent[x]

    def get_weight(self, x):
        """Cumulative weight from the root to x"""
        self.find(x)  # Path compression
        return self.weight[x]

    def diff(self, x, y):
        """weight(y) - weight(x)"""
        if self.find(x) != self.find(y):
            return None  # Different sets
        return self.get_weight(y) - self.get_weight(x)

    def union(self, x, y, w):
        """Add the relation: weight(y) - weight(x) = w
        Returns: True if successfully merged, False if contradiction
        """
        px, py = self.find(x), self.find(y)
        if px == py:
            # Already in the same set → check for contradiction
            return self.diff(x, y) == w

        w = w + self.weight[x] - self.weight[y]

        if self.rank[px] < self.rank[py]:
            self.parent[px] = py
            self.weight[px] = -w
        else:
            self.parent[py] = px
            self.weight[py] = w
            if self.rank[px] == self.rank[py]:
                self.rank[px] += 1

        return True

# Usage example: A - B = 3, B - C = 5 → A - C = 8
wuf = WeightedUnionFind(3)
wuf.union(0, 1, 3)  # weight[1] - weight[0] = 3
wuf.union(1, 2, 5)  # weight[2] - weight[1] = 5
print(wuf.diff(0, 2))  # 8 (weight[2] - weight[0] = 3+5)
```

### Weighted Union-Find Application: Height Differences

```python
def solve_height_differences(n: int, relations: list, queries: list) -> list:
    """Height difference problem
    relations: [(i, j, diff), ...] → person j is diff cm taller than person i
    queries: [(i, j), ...] → How many cm taller is person j than person i?
    """
    wuf = WeightedUnionFind(n)

    for i, j, diff in relations:
        wuf.union(i, j, diff)

    results = []
    for i, j in queries:
        d = wuf.diff(i, j)
        if d is None:
            results.append("Unknown")
        else:
            results.append(f"{d}cm")

    return results

# Persons 0, 1, 2, 3, 4
# Person 1 is 10cm taller than person 0
# Person 2 is 5cm taller than person 1
# Person 4 is 8cm taller than person 3
relations = [(0, 1, 10), (1, 2, 5), (3, 4, 8)]
queries = [(0, 2), (0, 4), (3, 4)]
print(solve_height_differences(5, relations, queries))
# ['15cm', 'Unknown', '8cm']
```

### Weighted Union-Find Application: Relative Scoring in Online Judges

```python
def relative_scoring(n: int, comparisons: list) -> list:
    """Contradiction detection in relative evaluation
    comparisons: [(i, j, diff), ...] → score_j - score_i = diff
    Returns the indices of contradictory pairs
    """
    wuf = WeightedUnionFind(n)
    contradictions = []

    for idx, (i, j, diff) in enumerate(comparisons):
        if not wuf.union(i, j, diff):
            # Contradiction detected
            actual_diff = wuf.diff(i, j)
            contradictions.append({
                'index': idx,
                'claimed': diff,
                'actual': actual_diff,
                'pair': (i, j)
            })

    return contradictions

comparisons = [
    (0, 1, 3),   # score_1 - score_0 = 3
    (1, 2, 5),   # score_2 - score_1 = 5
    (0, 2, 7),   # score_2 - score_0 = 7 ← contradicts 3+5=8!
]
result = relative_scoring(3, comparisons)
print(result)
# [{'index': 2, 'claimed': 7, 'actual': 8, 'pair': (0, 2)}]
```

---

## 10. Persistent Union-Find

The persistent data structure version of Union-Find allows access to any past state, supporting time-travel queries.

```python
class PersistentUnionFind:
    """Persistent Union-Find (simplified version: rollback support)
    Supports undo (rollback) of operations
    Note: Does not use path compression (union by rank only)
    """

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.history = []  # Records state before each operation

    def find(self, x: int) -> int:
        """Find without path compression (for persistence)"""
        while self.parent[x] != x:
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        px, py = self.find(x), self.find(y)
        if px == py:
            self.history.append(None)  # Record that nothing was done
            return False

        if self.rank[px] < self.rank[py]:
            px, py = py, px

        # Save state before the operation
        self.history.append((py, self.parent[py], self.rank[px]))

        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

        return True

    def rollback(self):
        """Undo the most recent union operation"""
        if not self.history:
            return

        record = self.history.pop()
        if record is None:
            return  # Operation that did nothing

        py, old_parent, old_rank_px = record
        px = self.parent[py]
        self.parent[py] = old_parent
        self.rank[px] = old_rank_px

    def save(self) -> int:
        """Snapshot ID of the current state (length of history)"""
        return len(self.history)

    def restore(self, snapshot: int):
        """Rollback to the specified snapshot"""
        while len(self.history) > snapshot:
            self.rollback()

# Usage example
puf = PersistentUnionFind(5)
snap0 = puf.save()

puf.union(0, 1)
puf.union(2, 3)
snap1 = puf.save()

puf.union(0, 2)
print(puf.find(0) == puf.find(3))  # True

puf.restore(snap1)
print(puf.find(0) == puf.find(3))  # False (after rollback)

puf.restore(snap0)
print(puf.find(0) == puf.find(1))  # False (fully rolled back)
```

---

## 11. Practical Applications of Union-Find

### Network Failure Detection

```python
def detect_network_partitions(n_servers: int, connections: list,
                               failures: list) -> dict:
    """Detect network partitions during failures
    connections: [(server_a, server_b), ...] full connection list
    failures: [(server_a, server_b), ...] failed connections
    """
    failure_set = set((min(a,b), max(a,b)) for a, b in failures)
    uf = UnionFind(n_servers)

    # Union only non-failed connections
    for a, b in connections:
        key = (min(a,b), max(a,b))
        if key not in failure_set:
            uf.union(a, b)

    partitions = uf.get_groups()
    result = {
        'num_partitions': len(partitions),
        'partitions': list(partitions.values()),
        'isolated_servers': [g[0] for g in partitions.values() if len(g) == 1],
        'largest_partition': max(len(g) for g in partitions.values()),
    }
    return result

connections = [(0,1), (1,2), (2,3), (3,4), (0,4), (2,5), (5,6)]
failures = [(2,3), (2,5)]
result = detect_network_partitions(7, connections, failures)
print(f"Number of partitions: {result['num_partitions']}")
print(f"Partitions: {result['partitions']}")
print(f"Isolated servers: {result['isolated_servers']}")
```

### Connected Component Labeling in Images

```python
def label_connected_components(image: list) -> list:
    """Connected component labeling for binary images
    image: 2D array (1=foreground, 0=background)
    Returns: Label for each pixel (0=background)
    """
    rows, cols = len(image), len(image[0])
    uf = UnionFind(rows * cols)

    # Union adjacent foreground pixels using 4-connectivity
    for r in range(rows):
        for c in range(cols):
            if image[r][c] == 0:
                continue
            for dr, dc in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and image[nr][nc] == 1:
                    uf.union(r * cols + c, nr * cols + nc)

    # Assign labels
    label_map = {}
    label_counter = 1
    labels = [[0] * cols for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            if image[r][c] == 0:
                continue
            root = uf.find(r * cols + c)
            if root not in label_map:
                label_map[root] = label_counter
                label_counter += 1
            labels[r][c] = label_map[root]

    return labels

image = [
    [1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1],
    [0, 0, 0, 1, 0],
    [1, 1, 0, 0, 0],
]
labels = label_connected_components(image)
for row in labels:
    print(row)
# [1, 1, 0, 0, 2]
# [1, 0, 0, 2, 2]
# [0, 0, 0, 2, 0]
# [3, 3, 0, 0, 0]
```

### Application to Clustering

```python
def single_linkage_clustering(points: list, k: int) -> list:
    """Single-linkage clustering
    points: [(x, y), ...]
    k: Target number of clusters
    Returns: Cluster label for each point
    """
    import math
    n = len(points)

    # Compute distances between all pairs
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            dist = math.sqrt(dx * dx + dy * dy)
            edges.append((dist, i, j))

    edges.sort()  # Sort by distance

    uf = UnionFind(n)

    # Merge until the number of clusters reaches k
    for dist, i, j in edges:
        if uf.count <= k:
            break
        uf.union(i, j)

    # Assign cluster labels
    label_map = {}
    label_counter = 0
    labels = []
    for i in range(n):
        root = uf.find(i)
        if root not in label_map:
            label_map[root] = label_counter
            label_counter += 1
        labels.append(label_map[root])

    return labels

points = [(0,0), (1,1), (0,1), (10,10), (11,11), (10,11)]
labels = single_linkage_clustering(points, 2)
print(labels)  # [0, 0, 0, 1, 1, 1]
```

---

## 12. Advanced Variants

### Partially Persistent Union-Find

```python
class PartiallyPersistentUnionFind:
    """Partially Persistent Union-Find
    Answers connected queries at any past time in O(log n)
    """

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.time = [float('inf')] * n  # Time when the node ceased to be a root
        self.size_history = [[(0, 1)] for _ in range(n)]  # (time, size)
        self.now = 0

    def find(self, x: int, t: int = None) -> int:
        """Representative at time t"""
        if t is None:
            t = self.now
        while self.time[x] <= t:
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        self.now += 1
        x = self.find(x)
        y = self.find(y)
        if x == y:
            return False

        if self.rank[x] < self.rank[y]:
            x, y = y, x

        self.parent[y] = x
        self.time[y] = self.now

        new_size = self.size_history[x][-1][1] + self.size_history[y][-1][1]
        self.size_history[x].append((self.now, new_size))

        if self.rank[x] == self.rank[y]:
            self.rank[x] += 1

        return True

    def connected(self, x: int, y: int, t: int = None) -> bool:
        return self.find(x, t) == self.find(y, t)
```

### Union-Find with Undo (Rollback Support)

Often used in combination with divide and conquer. Handles offline edge additions and deletions.

```python
class UnionFindWithUndo:
    """Union-Find with Undo
    Uses only union by rank (no path compression) for O(log n)
    Undo operation rolls back operations stored on a stack
    """

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.stack = []  # Operation recording stack

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        px, py = self.find(x), self.find(y)
        if px == py:
            self.stack.append(None)
            return False

        if self.rank[px] < self.rank[py]:
            px, py = py, px

        self.stack.append((py, self.rank[px]))
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

        return True

    def undo(self):
        record = self.stack.pop()
        if record is None:
            return
        py, old_rank_px = record
        px = self.parent[py]
        self.parent[py] = py
        self.rank[px] = old_rank_px
```

---

## 13. Theory of the Inverse Ackermann Function

The inverse Ackermann function alpha(n) appears in the complexity analysis of Union-Find.

```
Ackermann function A(m, n):
  A(0, n) = n + 1
  A(m, 0) = A(m-1, 1)
  A(m, n) = A(m-1, A(m, n-1))

Grows extremely rapidly:
  A(0, 0) = 1
  A(1, 1) = 3
  A(2, 2) = 7
  A(3, 3) = 61
  A(4, 4) = 2^(2^(2^...)) - 3  (a power tower of height 65536)

Inverse Ackermann function alpha(n):
  alpha(n) = min{k : A(k, k) >= n}

  alpha(1) = 0
  alpha(4) = 2
  alpha(65536) = 3
  alpha(2^65536) = 4
  alpha(A(4,4)) = 5

  For all practical inputs (n <= 10^80), alpha(n) <= 4

→ O(alpha(n)) is effectively O(1)
```

### Overview of Tarjan's Proof

Theorem proved by Tarjan in 1975:

> For Union-Find with path compression and union by rank, the total complexity of m operations (a mix of Find and Union) is O(m * alpha(n)).

Furthermore, in 1984, Tarjan and van Leeuwen proved that this is asymptotically optimal. That is, no algorithm for the Union-Find problem can do better than O(m * alpha(n)) (in a certain computational model).

---

## 14. Anti-Patterns

### Anti-Pattern 1: Omitting Path Compression and Union by Rank

```python
# BAD: Naive Union-Find → Find degrades to O(n)
class BadUnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            x = self.parent[x]  # No path compression!
        return x

    def union(self, x, y):
        self.parent[self.find(x)] = self.find(y)  # No rank consideration!

# GOOD: Apply both optimizations
class GoodUnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    # ... also implement union by rank
```

### Anti-Pattern 2: O(n) Connectivity Check Every Time

```python
# BAD: BFS/DFS for connectivity check after every edge addition → O(E * (V+E))
for u, v in edges:
    graph[u].append(v)
    if bfs_connected(graph, 0, target):  # O(V+E) every time
        ...

# GOOD: Union-Find for O(E * alpha(n)) ≈ O(E)
uf = UnionFind(n)
for u, v in edges:
    uf.union(u, v)
    if uf.connected(0, target):  # O(alpha(n)) ≈ O(1)
        ...
```

### Anti-Pattern 3: Attempting Edge Deletion with Union-Find

```python
# BAD: Union-Find cannot efficiently handle edge deletion
# Union-Find specializes in set merging and does not support splitting

# GOOD: Approaches when edge deletion is needed
# Method 1: Offline processing (add edges in reverse order)
# Method 2: Use a Link-Cut Tree
# Method 3: Rollback with Persistent Union-Find (only in reverse order of operations)
```

### Anti-Pattern 4: Not Considering Recursion Depth

```python
# BAD: Recursive Find causes stack overflow for large n
import sys
# Even sys.setrecursionlimit(10**6) is risky

# GOOD: Use iterative Find
def find_iterative(self, x):
    root = x
    while self.parent[root] != root:
        root = self.parent[root]
    while self.parent[x] != root:
        next_x = self.parent[x]
        self.parent[x] = root
        x = next_x
    return root
```

### Anti-Pattern 5: Mixing 0-indexed and 1-indexed

```python
# BAD: Problem uses 1-indexed vertices but Union-Find is created with 0-indexed
n = int(input())
uf = UnionFind(n)  # 0 ~ n-1
for _ in range(m):
    u, v = map(int, input().split())
    uf.union(u, v)  # u, v are 1 ~ n → IndexError!

# GOOD: Use size n+1 or subtract 1 from input
uf = UnionFind(n + 1)  # 0 ~ n (0 is unused)
for _ in range(m):
    u, v = map(int, input().split())
    uf.union(u, v)  # Can use 1-indexed as-is
```

---

## 15. Union-Find Template for Competitive Programming

```python
import sys
input = sys.stdin.readline

class UnionFind:
    """Union-Find template for competitive programming"""
    __slots__ = ['parent', 'rank', 'size', 'count']

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n
        self.count = n

    def find(self, x: int) -> int:
        # Iterative version (avoids stack overflow)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, x: int, y: int) -> bool:
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        self.size[px] += self.size[py]
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        self.count -= 1
        return True

    def connected(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)

    def get_size(self, x: int) -> int:
        return self.size[self.find(x)]

# Typical AtCoder ABC solution template
def solve():
    N, M = map(int, input().split())
    uf = UnionFind(N)

    for _ in range(M):
        a, b = map(int, input().split())
        a -= 1; b -= 1  # Convert to 0-indexed
        uf.union(a, b)

    # Number of connected components
    print(uf.count)

    # Size of the largest connected component
    max_size = max(uf.get_size(i) for i in range(N))
    print(max_size)
```


---

## Practical Exercises

### Exercise 1: Basic Implementation

Implement code that satisfies the following requirements.

**Requirements:**
- Validate input data
- Implement proper error handling
- Include test code

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
        """Retrieve processing results"""
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

Extend the basic implementation to add the following features.

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
        """Delete by key"""
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
- Select appropriate data structures
- Measure effectiveness with benchmarks

---

## Troubleshooting

### Common Errors and Solutions

| Error | Cause | Solution |
|--------|------|--------|
| Initialization error | Missing or incorrect config file | Check config file path and format |
| Timeout | Network latency / insufficient resources | Adjust timeout values, add retry logic |
| Out of memory | Growing data volume | Introduce batch processing, implement pagination |
| Permission error | Insufficient access rights | Verify executing user permissions, review settings |
| Data inconsistency | Concurrency conflicts | Introduce locking mechanisms, manage transactions |

### Debugging Procedure

1. **Check the error message**: Read the stack trace and identify the location
2. **Establish reproduction steps**: Reproduce the error with minimal code
3. **Formulate hypotheses**: List possible causes
4. **Verify step by step**: Use log output and debuggers to test hypotheses
5. **Fix and regression test**: After fixing, run tests on related areas

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
    """Decorator that logs function input and output"""
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

1. **Identify the bottleneck**: Measure with profiling tools
2. **Check memory usage**: Look for memory leaks
3. **Check I/O waits**: Monitor disk and network I/O status
4. **Check concurrent connections**: Verify connection pool state

| Problem Type | Diagnostic Tool | Countermeasure |
|-----------|-----------|------|
| CPU load | cProfile, py-spy | Algorithm improvement, parallelization |
| Memory leak | tracemalloc, objgraph | Proper reference cleanup |
| I/O bottleneck | strace, iostat | Async I/O, caching |
| DB latency | EXPLAIN, slow query log | Index optimization, query optimization |
---

## 16. FAQ

### Q1: What is the inverse Ackermann function alpha(n)?

**A:** alpha(n) is the inverse of the Ackermann function. For all practical inputs (up to the number of atoms in the universe, ~10^80), it is at most 5. This means O(alpha(n)) is effectively O(1). This complexity is achieved when using both path compression and union by rank. It was proved by Tarjan in 1975.

### Q2: Can elements be deleted from Union-Find?

**A:** Standard Union-Find cannot efficiently delete elements or split groups. When needed, consider: (1) logical deletion using a "deleted" flag and substituting with a new element, (2) using advanced data structures like Link-Cut Trees, or (3) rebuilding the entire structure.

### Q3: Is Union-Find effective for online problems?

**A:** Yes. It is ideal for managing connectivity in situations where edges are dynamically added (online queries). While BFS/DFS requires recomputation whenever an edge is added, Union-Find handles updates incrementally in O(alpha(n)).

### Q4: Should I use union by rank or union by size?

**A:** Theoretically, both achieve the same complexity O(alpha(n)). In practice, union by size is recommended because the size information directly supports additional queries like "How many elements are in this set?" Rank is merely an upper bound on tree height and has no direct semantic meaning.

### Q5: What is the space complexity of Union-Find?

**A:** O(n). Both the parent array and the rank (or size) array are needed, each containing n elements. Space increases when using additional data structures (weighted, persistent, etc.).

### Q6: What if edges in the graph are being deleted?

**A:** Union-Find does not support edge deletion. Workarounds include: (1) offline processing where edge additions are processed in reverse order (from the perspective of reversal, edge deletion becomes edge addition), (2) using Link-Cut Trees (Euler Tour Trees) which handle online edge additions and deletions in O(log n) per operation, (3) divide and conquer + Union-Find with Undo.

### Q7: Is Union-Find suitable for parallel processing?

**A:** Standard Union-Find is not thread-safe. Wait-Free Union-Find (based on CAS operations) has been studied as a parallel variant. In practice, you can either use locks or create partial Union-Find structures in each thread and merge them at the end.

---


## FAQ

### Q1: What is the most important point when studying this topic?

Gaining practical experience is the most important aspect. Understanding deepens not just through theory alone, but by actually writing code and verifying its behavior.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals and jumping into advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before proceeding to the next step.

### Q3: How is this used in practice?

Knowledge of this topic is frequently utilized in day-to-day development work. It becomes particularly important during code reviews and architecture design.

---

## 17. Summary

| Item | Key Points |
|:---|:---|
| Basic operations | Find (get representative) and Union (merge sets) |
| Path compression | Connect all nodes directly to root during Find → flatten the tree |
| Union by rank | Attach shorter tree under taller tree → bound tree height |
| Complexity | Both optimizations yield O(alpha(n)) ≈ effectively O(1) |
| Kruskal application | Used for cycle detection. O(E log E) for MST |
| Weighted extension | Manages relative weights between elements |
| Persistent version | Rollback support (no path compression) |
| Practical applications | Network management, image processing, clustering |

---

## Recommended Next Guides

- [Graph Traversal](../02-algorithms/02-graph-traversal.md) -- Alternative approach (BFS/DFS) for connected components
- [Greedy Algorithms](../02-algorithms/05-greedy.md) -- Details on Kruskal's algorithm
- [Segment Tree](./01-segment-tree.md) -- Another advanced data structure

---

## References

1. Cormen, T. H. et al. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. -- Chapter 19
2. Tarjan, R. E. (1975). "Efficiency of a Good But Not Linear Set Union Algorithm." *JACM*.
3. Tarjan, R. E. & van Leeuwen, J. (1984). "Worst-case Analysis of Set Union Algorithms." *JACM*.
4. Sedgewick, R. & Wayne, K. (2011). *Algorithms* (4th ed.). Addison-Wesley. -- 1.5 Union-Find
5. Galil, Z. & Italiano, G. F. (1991). "Data Structures and Algorithms for Disjoint Set Union Problems." *ACM Computing Surveys*.
6. Alstrup, S. et al. (2014). "Union-Find with Constant Time Deletions." *ACM Transactions on Algorithms*.
