# Greedy Algorithms and Backtracking

> Greedy algorithms are an optimistic strategy that always picks "the best move right now." Backtracking is a cautious strategy that "tries every possibility and retreats upon failure."

## What You Will Learn in This Chapter

- [ ] Understand the conditions under which greedy algorithms yield optimal solutions
- [ ] Implement classic greedy algorithms
- [ ] Understand how backtracking works along with pruning techniques
- [ ] Learn methods for proving the correctness of greedy algorithms
- [ ] Master advanced pruning techniques
- [ ] Understand approximation algorithms and heuristics used in practice

## Prerequisites


---

## 1. Greedy Algorithms

### 1.1 Basic Concepts

```
Greedy Algorithm: Make the locally optimal choice at each step

  Characteristics:
  - Once a choice is made, it is never changed (no backtracking)
  - Fast (typically O(n log n) or less)
  - Proof is required to guarantee optimality

  Two conditions for a greedy algorithm to be optimal:
  1. Greedy choice property: A locally optimal choice is part of the globally optimal solution
  2. Optimal substructure: The remaining subproblem can also be solved optimally
```

### 1.2 Classic Greedy Algorithms

```python
# 1. Activity Selection Problem (Interval Scheduling)
def activity_selection(activities):
    """Select the maximum number of non-overlapping activities"""
    # Sort by finish time
    activities.sort(key=lambda x: x[1])
    selected = [activities[0]]
    last_end = activities[0][1]

    for start, end in activities[1:]:
        if start >= last_end:
            selected.append((start, end))
            last_end = end

    return selected

# Example: [(1,4), (3,5), (0,6), (5,7), (3,9), (5,9), (6,10), (8,11)]
# -> [(1,4), (5,7), (8,11)] -- maximum of 3 activities

# 2. Huffman Coding (Optimal Prefix Code)
import heapq

def huffman(freq):
    """Build an optimal Huffman tree from character frequencies"""
    heapq.heapify(heap)

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    return sorted(heap[0][1:], key=lambda x: (len(x[1]), x[0]))

# 3. Change-Making Problem (optimal only for specific coin sets)
def coin_greedy(coins, amount):
    """Greedily use the largest coins first"""
    coins.sort(reverse=True)
    result = []
    for coin in coins:
        while amount >= coin:
            result.append(coin)
            amount -= coin
    return result if amount == 0 else None

# Note: Optimal for coins=[1,5,10,25]
# For coins=[1,3,4] with amount=6 -> greedy: [4,1,1]=3 coins != optimal: [3,3]=2 coins
```

### 1.3 Greedy vs DP: When to Choose

```
When can a greedy algorithm be used?

  Greedy is optimal:
  - Interval scheduling (sort by finish time)
  - Huffman coding
  - Kruskal's algorithm (minimum spanning tree)
  - Dijkstra's algorithm (shortest path)
  - US coin change-making
  - Fractional knapsack
  - Task scheduling with deadlines

  Greedy is NOT optimal (DP is needed):
  - 0-1 Knapsack
  - Coin problem (general coin sets)
  - Longest Common Subsequence
  - Edit distance
  - Traveling Salesman Problem

  Decision hints:
  - Can you make a "no-regret choice" at each step?
  - Can you find a counterexample?
  - Does the problem have a matroid structure? (mathematically rigorous criterion)
```

### 1.4 Proving Greedy Correctness

```python
# Three methods for proving greedy correctness

# Method 1: Exchange Argument
# "Assume the optimal solution differs from the greedy solution, and show that
#  transforming it into the greedy solution does not make it worse."

# Example: Proof for interval scheduling
# Consider optimal solution O and greedy solution G.
# For the first activity o1 in O and g1 in G:
# g1 has the minimum finish time -> g1.end <= o1.end
# Replacing o1 with g1 does not increase conflicts with remaining activities
# -> |G| >= |O| -> The greedy solution is optimal

# Method 2: Induction
# "Show inductively that after selecting k items, the solution contains
#  k items that are part of the optimal solution."

# Method 3: Matroid Theory
# If the problem has a matroid structure, the greedy algorithm is optimal
# Three matroid conditions:
# 1. The empty set is an independent set
# 2. Every subset of an independent set is independent
# 3. Exchange axiom: If |A| < |B|, there exists x in B\A such that A union {x} is independent
```

### 1.5 Fractional Knapsack Problem

```python
def fractional_knapsack(weights, values, capacity):
    """Knapsack problem where items can be divided
    -> Greedy by value density (value/weight) in descending order is optimal"""

    n = len(weights)
    # Sort by value density
    items = sorted(range(n), key=lambda i: values[i] / weights[i], reverse=True)

    total_value = 0
    remaining = capacity

    fractions = [0.0] * n

    for i in items:
        if remaining <= 0:
            break
        if weights[i] <= remaining:
            # Take the entire item
            fractions[i] = 1.0
            total_value += values[i]
            remaining -= weights[i]
        else:
            # Take a fraction of the item
            fraction = remaining / weights[i]
            fractions[i] = fraction
            total_value += values[i] * fraction
            remaining = 0

    return total_value, fractions

# Example: weights=[10, 20, 30], values=[60, 100, 120], capacity=50
# Density: [6, 5, 4]
# Item 0 (10kg, 60) fully + Item 1 (20kg, 100) fully + Item 2 (20/30, 80)
# = 60 + 100 + 80 = 240

# Difference from 0-1 Knapsack:
# Fractional knapsack: Items can be divided -> Greedy is optimal O(n log n)
# 0-1 Knapsack: Items cannot be divided -> DP is needed O(nW)
```

### 1.6 Task Scheduling

```python
def task_scheduling_with_deadline(tasks):
    """Schedule tasks with profits and deadlines to maximize total profit
    tasks: [(profit, deadline), ...]"""

    # Sort by profit in descending order
    tasks.sort(key=lambda x: x[0], reverse=True)

    max_deadline = max(t[1] for t in tasks)
    slots = [False] * (max_deadline + 1)  # Time slots
    total_profit = 0
    scheduled = []

    for profit, deadline in tasks:
        # Search for an available slot in reverse from the deadline
        for slot in range(deadline, 0, -1):
            if not slots[slot]:
                slots[slot] = True
                total_profit += profit
                scheduled.append((profit, deadline, slot))
                break

    return total_profit, scheduled

# Example:
tasks = [(100, 2), (19, 1), (27, 2), (25, 1), (15, 3)]
profit, schedule = task_scheduling_with_deadline(tasks)
# Profit order: 100, 27, 25, 19, 15
# Slot 1: 27 (d=2 goes to slot 2, 25 with d=1 goes to slot 1...)
# Optimal: slot 1=27, slot 2=100, slot 3=15 -> 142

# Minimum Lateness Scheduling
def min_lateness_scheduling(jobs):
    """Minimize maximum lateness given jobs with processing times and deadlines
    jobs: [(processing_time, deadline), ...]"""

    # Sort by deadline (EDF: Earliest Deadline First)
    indexed_jobs = sorted(enumerate(jobs), key=lambda x: x[1][1])

    current_time = 0
    max_lateness = 0
    schedule = []

    for idx, (proc_time, deadline) in indexed_jobs:
        current_time += proc_time
        lateness = max(0, current_time - deadline)
        max_lateness = max(max_lateness, lateness)
        schedule.append({
            'job': idx,
            'start': current_time - proc_time,
            'finish': current_time,
            'lateness': lateness
        })

    return max_lateness, schedule

# Example: Jobs: [(1, 2), (2, 4), (3, 6), (4, 8)]
# Processing order: as-is (already sorted by deadline)
# Finish times: 1, 3, 6, 10
# Lateness: 0, 0, 0, 2 -> maximum lateness = 2
```

### 1.7 Interval-Related Greedy Algorithms

```python
def min_intervals_to_cover(intervals, start, end):
    """Cover the range [start, end] with the minimum number of sub-intervals"""
    intervals.sort()
    count = 0
    i = 0
    current = start

    while current < end and i < len(intervals):
        best_end = current

        # Among intervals that contain current, choose the one extending farthest
        while i < len(intervals) and intervals[i][0] <= current:
            best_end = max(best_end, intervals[i][1])
            i += 1

        if best_end == current:
            return -1  # Cannot cover

        count += 1
        current = best_end

    return count if current >= end else -1

# Example: intervals=[(0,3),(2,5),(3,7),(6,10)], start=0, end=10
# Selection: (0,3) -> (2,5) -> (3,7) -> (6,10) = 4 intervals...
# Optimal: (0,3) -> (3,7) -> (6,10) = 3 intervals

# Merging Overlapping Intervals
def merge_intervals(intervals):
    """Merge overlapping intervals"""
    if not intervals:
        return []

    intervals.sort()
    merged = [intervals[0]]

    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    return merged

# Example: [(1,3), (2,6), (8,10), (15,18)]
# -> [(1,6), (8,10), (15,18)]

# Minimum Number of Arrows to Burst Balloons
def min_arrows_to_burst_balloons(balloons):
    """Burst overlapping balloons with minimum arrows (variant of interval scheduling)"""
    if not balloons:
        return 0

    balloons.sort(key=lambda x: x[1])
    arrows = 1
    end = balloons[0][1]

    for s, e in balloons[1:]:
        if s > end:  # A new arrow is needed
            arrows += 1
            end = e

    return arrows

# Example: balloons = [(10,16), (2,8), (1,6), (7,12)]
# After sorting: [(1,6), (2,8), (7,12), (10,16)]
# Arrow 1: x=6 -> bursts (1,6) and (2,8)
# Arrow 2: x=12 -> bursts (7,12) and (10,16)
# -> 2 arrows
```

### 1.8 Kruskal's and Prim's Algorithms (Minimum Spanning Tree)

```python
class UnionFind:
    """Union-Find (Disjoint Set data structure)"""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

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
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

def kruskal(n, edges):
    """Kruskal's algorithm: Add edges in ascending order of weight (greedy)"""
    edges.sort(key=lambda x: x[2])  # (u, v, weight)
    uf = UnionFind(n)
    mst = []
    total_weight = 0

    for u, v, w in edges:
        if uf.union(u, v):  # Add if it does not form a cycle
            mst.append((u, v, w))
            total_weight += w
            if len(mst) == n - 1:
                break

    return total_weight, mst

# Time complexity: O(E log E) (dominated by sorting)
# Correctness: Cut property -> The minimum-weight cross edge is safe

def prim(n, adj):
    """Prim's algorithm: Add vertices one at a time (greedy selection via priority queue)"""
    import heapq
    visited = [False] * n
    mst = []
    total_weight = 0
    heap = [(0, 0, -1)]  # (weight, vertex, parent)

    while heap and len(mst) < n:
        w, u, parent = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        total_weight += w
        if parent != -1:
            mst.append((parent, u, w))

        for v, weight in adj[u]:
            if not visited[v]:
                heapq.heappush(heap, (weight, v, u))

    return total_weight, mst

# Time complexity: O(E log V) (heap operations)
# Kruskal: Advantageous for sparse graphs (fewer edges)
# Prim: Advantageous for dense graphs (fewer vertices)

# Usage example
edges = [(0,1,4), (0,7,8), (1,2,8), (1,7,11), (2,3,7),
         (2,5,4), (2,8,2), (3,4,9), (3,5,14), (4,5,10),
         (5,6,2), (6,7,1), (6,8,6), (7,8,7)]
weight, mst = kruskal(9, edges)
print(f"MST weight: {weight}")  # 37
```

### 1.9 Dijkstra's Algorithm

```python
import heapq

def dijkstra(adj, start):
    """Dijkstra's algorithm: Single-source shortest path (non-negative weights only)"""
    n = len(adj)
    dist = [float('inf')] * n
    dist[start] = 0
    prev = [-1] * n
    heap = [(0, start)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue  # A shorter path has already been found

        for v, w in adj[u]:
            new_dist = dist[u] + w
            if new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(heap, (new_dist, v))

    return dist, prev

def reconstruct_path(prev, start, end):
    """Reconstruct the shortest path"""
    path = []
    current = end
    while current != -1:
        path.append(current)
        current = prev[current]
    path.reverse()
    return path if path[0] == start else []

# Usage example
adj = [
    [(1, 4), (7, 8)],     # 0
    [(0, 4), (2, 8)],     # 1
    [(1, 8), (3, 7)],     # 2
    [(2, 7), (4, 9)],     # 3
    [(3, 9), (5, 10)],    # 4
    [(4, 10), (6, 2)],    # 5
    [(5, 2), (7, 1)],     # 6
    [(0, 8), (6, 1)],     # 7
]
dist, prev = dijkstra(adj, 0)
path = reconstruct_path(prev, 0, 4)
print(f"Shortest distance 0->4: {dist[4]}")   # 19
print(f"Path: {path}")                         # [0, 1, 2, 3, 4]

# Why is the greedy approach correct?
# In Dijkstra's algorithm, the distance of a vertex popped from the heap is finalized.
# Reason: With non-negative weights, going through unprocessed vertices cannot yield a shorter path.
# Note: With negative weights, the greedy choice property breaks -> use Bellman-Ford instead.
```

---

## 2. Backtracking

### 2.1 Basic Concepts

```
Backtracking: Build candidate solutions and retreat upon constraint violation

  Difference from exhaustive search:
  - Exhaustive search: Generate all combinations, then check
  - Backtracking: Detect constraint violations during construction -> prune

  Search tree visualization:
          root
        /  |  \
       a   b   c     <- Choice for 1st character
      /|\ /|\ /|\
     a b c a b c ...  <- Choice for 2nd character
     ^     ^
     OK    Constraint violation -> retreat (backtrack)
```

### 2.2 Backtracking Template

```python
def backtrack_template(candidates, constraints):
    """General backtracking template"""
    results = []

    def backtrack(state, choices):
        # Base case: A solution is complete
        if is_solution(state):
            results.append(state.copy())
            return

        for choice in choices:
            # Pruning: Skip if the choice violates constraints
            if not is_valid(state, choice, constraints):
                continue

            # Make the choice
            state.append(choice)  # make choice

            # Recurse
            backtrack(state, next_choices(choices, choice))

            # Undo the choice (backtrack)
            state.pop()  # undo choice

    backtrack([], candidates)
    return results

# Three elements of backtracking:
# 1. Choice: What to pick
# 2. Constraint: What constitutes a valid choice
# 3. Goal: When a solution is complete
```

### 2.3 Classic Backtracking Problems

```python
# 1. N-Queens Problem
def solve_n_queens(n):
    """Place queens on an NxN board so that no two queens attack each other"""
    solutions = []

    def is_safe(board, row, col):
        for i in range(row):
            if board[i] == col:  # Same column
                return False
            if abs(board[i] - col) == abs(i - row):  # Diagonal
                return False
        return True

    def backtrack(board, row):
        if row == n:
            solutions.append(board[:])
            return
        for col in range(n):
            if is_safe(board, row, col):
                board[row] = col
                backtrack(board, row + 1)
                # board[row] will be overwritten in the next iteration,
                # so an explicit "undo" operation is not needed

    backtrack([0] * n, 0)
    return solutions

# Number of N-Queens solutions:
# N=4: 2, N=5: 10, N=6: 4, N=7: 40, N=8: 92, N=12: 14200

# Visualize a solution
def print_queens(board):
    n = len(board)
    for row in range(n):
        line = ""
        for col in range(n):
            if board[row] == col:
                line += "Q "
            else:
                line += ". "
        print(line)
    print()

# 2. Generating Permutations
def permutations(nums):
    result = []
    def backtrack(path, remaining):
        if not remaining:
            result.append(path[:])
            return
        for i in range(len(remaining)):
            path.append(remaining[i])
            backtrack(path, remaining[:i] + remaining[i+1:])
            path.pop()  # Backtrack (undo)
    backtrack([], nums)
    return result

# Permutations with Duplicate Elements
def permutations_with_duplicates(nums):
    """Exclude duplicate permutations when elements have duplicates"""
    nums.sort()
    result = []
    used = [False] * len(nums)

    def backtrack(path):
        if len(path) == len(nums):
            result.append(path[:])
            return

        for i in range(len(nums)):
            if used[i]:
                continue
            # Eliminate duplicates: Use an element with the same value only after using the previous one
            if i > 0 and nums[i] == nums[i-1] and not used[i-1]:
                continue

            used[i] = True
            path.append(nums[i])
            backtrack(path)
            path.pop()
            used[i] = False

    backtrack([])
    return result

# Example: [1, 1, 2] -> [[1,1,2], [1,2,1], [2,1,1]]

# 3. Sudoku Solver
def solve_sudoku(board):
    def is_valid(board, row, col, num):
        for i in range(9):
            if board[row][i] == num: return False
            if board[i][col] == num: return False
        box_r, box_c = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_r, box_r + 3):
            for j in range(box_c, box_c + 3):
                if board[i][j] == num: return False
        return True

    def backtrack():
        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    for num in range(1, 10):
                        if is_valid(board, i, j, num):
                            board[i][j] = num
                            if backtrack():
                                return True
                            board[i][j] = 0  # Backtrack
                    return False  # No valid number for this cell
        return True  # All cells filled
    backtrack()
```

### 2.4 Combination Enumeration

```python
# Combinations
def combinations(nums, k):
    """All combinations of choosing k elements from nums"""
    result = []

    def backtrack(start, path):
        if len(path) == k:
            result.append(path[:])
            return

        # Pruning: Not enough remaining elements
        remaining = len(nums) - start
        needed = k - len(path)
        if remaining < needed:
            return

        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    backtrack(0, [])
    return result

# Example: combinations([1,2,3,4], 2)
# -> [[1,2], [1,3], [1,4], [2,3], [2,4], [3,4]]

# Combinations that sum to target (each element used at most once)
def combination_sum_unique(candidates, target):
    """Combinations without duplicates that sum to target"""
    candidates.sort()
    result = []

    def backtrack(start, remaining, path):
        if remaining == 0:
            result.append(path[:])
            return
        if remaining < 0:
            return

        for i in range(start, len(candidates)):
            # Skip duplicates of the same value
            if i > start and candidates[i] == candidates[i-1]:
                continue
            if candidates[i] > remaining:
                break  # Already sorted, so all subsequent values exceed remaining

            path.append(candidates[i])
            backtrack(i + 1, remaining - candidates[i], path)
            path.pop()

    backtrack(0, target, [])
    return result

# Combinations that sum to target (each element can be used multiple times)
def combination_sum_repeat(candidates, target):
    """Combinations with repeated use that sum to target"""
    candidates.sort()
    result = []

    def backtrack(start, remaining, path):
        if remaining == 0:
            result.append(path[:])
            return

        for i in range(start, len(candidates)):
            if candidates[i] > remaining:
                break

            path.append(candidates[i])
            backtrack(i, remaining - candidates[i], path)  # Restart from i (repetition allowed)
            path.pop()

    backtrack(0, target, [])
    return result

# Example: candidates=[2,3,6,7], target=7
# -> [[2,2,3], [7]]
```

### 2.5 Subset Enumeration

```python
def subsets(nums):
    """Enumerate all subsets (backtracking version)"""
    result = []

    def backtrack(start, path):
        result.append(path[:])  # Every intermediate state is also a solution

        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    backtrack(0, [])
    return result

# Subsets with Duplicate Elements
def subsets_with_duplicates(nums):
    """Subset enumeration when elements have duplicates"""
    nums.sort()
    result = []

    def backtrack(start, path):
        result.append(path[:])

        for i in range(start, len(nums)):
            # Skip the same value at the same level
            if i > start and nums[i] == nums[i-1]:
                continue
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    backtrack(0, [])
    return result

# Example: [1, 2, 2]
# -> [[], [1], [1,2], [1,2,2], [2], [2,2]]
```

### 2.6 Pruning Techniques

```
Pruning: Eliminating unnecessary searches in advance

  1. Feasibility pruning: Terminate search when a constraint violation is certain
     -> N-Queens: Stop immediately if a queen is on the same column/diagonal

  2. Optimality pruning: Terminate if the current path cannot reach the best solution
     -> Branch and bound: Prune if the best-case estimate is no better than the current best

  3. Symmetry pruning: Explore only one of symmetric solutions
     -> N-Queens: Restrict the first queen to the upper half

  4. Ordering pruning: Arrange the search order to promote early termination
     -> Sort and try larger values first -> Hit constraints sooner

  5. Precomputation pruning: Precompute impossible states
     -> Sudoku: Manage candidates for each cell using bitmasks

  Pruning effectiveness:
  - N-Queens (N=8): Exhaustive search 16,777,216 states -> with pruning 15,720 states
  - Sudoku: Exhaustive search 6.67x10^21 -> solved instantly with pruning
```

### 2.7 Advanced Backtracking: Sudoku Solver with Constraint Propagation

```python
def solve_sudoku_advanced(board):
    """High-speed Sudoku solver with constraint propagation + backtracking"""

    # Manage candidates for each cell using bitmasks
    rows = [0] * 9
    cols = [0] * 9
    boxes = [0] * 9

    empty_cells = []

    for i in range(9):
        for j in range(9):
            if board[i][j] != 0:
                bit = 1 << board[i][j]
                rows[i] |= bit
                cols[j] |= bit
                boxes[(i // 3) * 3 + j // 3] |= bit
            else:
                empty_cells.append((i, j))

    def get_candidates(i, j):
        """List of numbers that can be placed in cell (i, j)"""
        used = rows[i] | cols[j] | boxes[(i // 3) * 3 + j // 3]
        return [num for num in range(1, 10) if not (used & (1 << num))]

    def backtrack(idx):
        if idx == len(empty_cells):
            return True

        i, j = empty_cells[idx]
        box_idx = (i // 3) * 3 + j // 3

        for num in get_candidates(i, j):
            bit = 1 << num
            board[i][j] = num
            rows[i] |= bit
            cols[j] |= bit
            boxes[box_idx] |= bit

            if backtrack(idx + 1):
                return True

            board[i][j] = 0
            rows[i] ^= bit
            cols[j] ^= bit
            boxes[box_idx] ^= bit

        return False

    # MRV (Minimum Remaining Values) heuristic
    # Fill cells with fewer candidates first
    empty_cells.sort(key=lambda cell: len(get_candidates(cell[0], cell[1])))

    backtrack(0)
    return board

# The MRV heuristic minimizes the branching factor of the search tree.
# Cells with only one candidate (naked singles) are determined immediately.
```

---

## 3. Exhaustive Search Techniques

### 3.1 Bitmask Enumeration

```python
# Bitmask Enumeration: Enumerate all 2^n subsets

def subsets_bitmask(nums):
    """Enumerate all subsets"""
    n = len(nums)
    result = []
    for mask in range(1 << n):  # 0 to 2^n - 1
        subset = []
        for i in range(n):
            if mask & (1 << i):  # Check if the i-th bit is set
                subset.append(nums[i])
        result.append(subset)
    return result

# nums = [1, 2, 3]
# mask=000 -> []
# mask=001 -> [1]
# mask=010 -> [2]
# mask=011 -> [1, 2]
# mask=100 -> [3]
# mask=101 -> [1, 3]
# mask=110 -> [2, 3]
# mask=111 -> [1, 2, 3]

# Applicable when: n <= 20 or so (2^20 ~ 1 million)
```

### 3.2 Meet in the Middle

```python
def subset_sum_meet_in_middle(nums, target):
    """Meet in the Middle: Split 2^n into 2^(n/2) x 2"""
    n = len(nums)
    half = n // 2

    # Enumerate subset sums of the first half
    sums_first = {}
    for mask in range(1 << half):
        s = sum(nums[i] for i in range(half) if mask & (1 << i))
        sums_first[s] = sums_first.get(s, 0) + 1

    # For each second-half subset, look up target - s
    count = 0
    remaining = n - half
    for mask in range(1 << remaining):
        s = sum(nums[half + i] for i in range(remaining) if mask & (1 << i))
        complement = target - s
        if complement in sums_first:
            count += sums_first[complement]

    return count

# Time complexity: O(2^(n/2) x n)
# For n=40: 2^40 ~ 1 trillion -> 2^20 ~ 1 million (feasible)

# Usage example
nums = list(range(1, 41))  # 1 to 40
target = 410  # Half of 1+2+...+40 = 820
# Exhaustive search over 2^40 is infeasible, but Meet in the Middle makes it possible
```

### 3.3 Search State Space and Complexity

```
Complexity of each search technique:

  +----------------------+-------------+--------------+
  | Technique            | Complexity  | Applicable   |
  +----------------------+-------------+--------------+
  | All permutations     | O(n!)       | n <= 10      |
  | Bitmask enumeration  | O(2^n x n)  | n <= 20      |
  | Meet in the Middle   | O(2^(n/2) x n)| n <= 40    |
  | Backtracking         | O(exp) *    | Depends on   |
  |                      |             | pruning      |
  | DFS/BFS             | O(V + E)    | Graph size   |
  +----------------------+-------------+--------------+

  * Backtracking complexity depends heavily on pruning efficiency
  Without pruning: O(n!) or O(k^n) range
  With good pruning: Can approach O(n x k) depending on the problem
```

---

## 4. Branch and Bound

### 4.1 Basic Concepts

```python
def branch_and_bound_knapsack(weights, values, capacity):
    """0-1 Knapsack problem using branch and bound"""
    n = len(weights)

    # Sort by value density (for upper bound computation)
    items = sorted(range(n), key=lambda i: values[i] / weights[i], reverse=True)
    sorted_weights = [weights[i] for i in items]
    sorted_values = [values[i] for i in items]

    def upper_bound(idx, remaining_cap, current_value):
        """Compute upper bound using the fractional knapsack"""
        bound = current_value
        cap = remaining_cap

        for i in range(idx, n):
            if sorted_weights[i] <= cap:
                bound += sorted_values[i]
                cap -= sorted_weights[i]
            else:
                bound += sorted_values[i] * (cap / sorted_weights[i])
                break

        return bound

    best = [0]

    def backtrack(idx, remaining_cap, current_value):
        if current_value > best[0]:
            best[0] = current_value

        if idx == n:
            return

        # Pruning: Do not explore if the upper bound is no better than the current best
        if upper_bound(idx, remaining_cap, current_value) <= best[0]:
            return

        # Include the item
        if sorted_weights[idx] <= remaining_cap:
            backtrack(idx + 1, remaining_cap - sorted_weights[idx],
                     current_value + sorted_values[idx])

        # Exclude the item
        backtrack(idx + 1, remaining_cap, current_value)

    backtrack(0, capacity, 0)
    return best[0]

# Key points of branch and bound:
# 1. Compute an upper bound (optimistic estimate)
# 2. Prune a branch if its upper bound is no better than the current best solution
# 3. A good initial solution allows more branches to be pruned
# -> It is effective to first obtain an approximate solution via a greedy algorithm
```

### 4.2 Branch and Bound for TSP

```python
def tsp_branch_and_bound(dist):
    """Solve TSP using branch and bound"""
    n = len(dist)
    INF = float('inf')
    best_cost = [INF]
    best_path = [None]

    def lower_bound(visited, current, cost):
        """Compute a lower bound as the sum of minimum outgoing edges of unvisited cities"""
        bound = cost
        for i in range(n):
            if i not in visited:
                min_edge = min(
                    dist[i][j] for j in range(n)
                    if j != i and (j not in visited or j == 0)
                )
                bound += min_edge
        return bound

    def backtrack(current, visited, path, cost):
        if len(visited) == n:
            total = cost + dist[current][0]
            if total < best_cost[0]:
                best_cost[0] = total
                best_path[0] = path[:]
            return

        # Pruning by lower bound
        if lower_bound(visited, current, cost) >= best_cost[0]:
            return

        for next_city in range(n):
            if next_city not in visited:
                new_cost = cost + dist[current][next_city]
                if new_cost < best_cost[0]:
                    visited.add(next_city)
                    path.append(next_city)
                    backtrack(next_city, visited, path, new_cost)
                    path.pop()
                    visited.remove(next_city)

    backtrack(0, {0}, [0], 0)
    best_path[0].append(0)
    return best_cost[0], best_path[0]
```

---

## 5. Approximation Algorithms in Practice

### 5.1 Approaches to NP-Hard Problems

```
Practical approaches to NP-hard problems:

  1. Exact algorithms (small inputs)
     - Brute force: n <= 10
     - Bitmask enumeration: n <= 20
     - Bitmask DP: n <= 20
     - Branch and bound: n <= 30 or so (problem-dependent)

  2. Approximation algorithms (with guarantees)
     - Vertex cover: 2-approximation (within 2x of optimal)
     - TSP (with triangle inequality): 1.5-approximation (Christofides)
     - Set cover: O(log n)-approximation

  3. Heuristics (no guarantees)
     - Simulated Annealing (SA)
     - Genetic Algorithm (GA)
     - Tabu Search
     - Local Search
     - Randomized algorithms

  4. Problem restriction/relaxation
     - Reduce to a special case
     - Limit input size
     - Accept a compromise on solution quality
```

### 5.2 Simulated Annealing Implementation

```python
import random
import math

def simulated_annealing_tsp(dist, initial_temp=10000, cooling_rate=0.9995,
                            min_temp=1e-8, max_iterations=1000000):
    """Approximate TSP solution using simulated annealing"""
    n = len(dist)

    # Initial solution: random permutation
    current = list(range(n))
    random.shuffle(current)

    def tour_cost(tour):
        return sum(dist[tour[i]][tour[(i+1) % n]] for i in range(n))

    current_cost = tour_cost(current)
    best = current[:]
    best_cost = current_cost
    temp = initial_temp

    for iteration in range(max_iterations):
        if temp < min_temp:
            break

        # Neighborhood: 2-opt (swap two random edges)
        i, j = sorted(random.sample(range(n), 2))
        new_tour = current[:i] + current[i:j+1][::-1] + current[j+1:]
        new_cost = tour_cost(new_tour)

        delta = new_cost - current_cost

        # Always accept improvements; accept deteriorations with a probability
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current = new_tour
            current_cost = new_cost

            if current_cost < best_cost:
                best = current[:]
                best_cost = current_cost

        temp *= cooling_rate

    return best_cost, best

# Simulated annealing parameter tuning:
# - Initial temperature: Higher values widen the initial search range
# - Cooling rate: Closer to 1 means slower cooling (higher quality but slower)
# - Neighborhood definition: Choose an operation suited to the problem
```

### 5.3 Local Search and 2-opt

```python
def two_opt_tsp(dist):
    """Improve TSP solution via 2-opt local search"""
    n = len(dist)
    tour = list(range(n))

    def tour_cost(tour):
        return sum(dist[tour[i]][tour[(i+1) % n]] for i in range(n))

    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if j == n - 1 and i == 0:
                    continue  # Same edge

                # Cost change when swapping two edges
                delta = (
                    dist[tour[i]][tour[j]] +
                    dist[tour[i+1]][tour[(j+1) % n]] -
                    dist[tour[i]][tour[i+1]] -
                    dist[tour[j]][tour[(j+1) % n]]
                )

                if delta < -1e-10:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    improved = True

    return tour_cost(tour), tour

# 2-opt time complexity: O(n^2) per iteration
# Typically converges in a small number of iterations
# No optimality guarantee, but produces good solutions in practice
```

---

## 6. Practical Exercises

### Exercise 1: Greedy Algorithm (Basic)
Solve the fractional knapsack problem (where items can be divided) using a greedy algorithm. Explain the difference from the 0-1 knapsack.

### Exercise 2: Backtracking (Intermediate)
Given an array of numbers, find all combinations that sum to a target value (each element can be used at most once).

### Exercise 3: Interval Scheduling (Intermediate)
Solve the weighted interval scheduling problem. Each activity has a profit; maximize the total profit of non-overlapping activities (Hint: DP + binary search).

### Exercise 4: Graph Problem (Intermediate)
Implement a program that finds the minimum spanning tree using Kruskal's algorithm. Also implement the Union-Find data structure.

### Exercise 5: Applied Backtracking (Advanced)
Implement a Sudoku solver with constraint propagation and compare its performance against plain backtracking.

### Exercise 6: Optimization (Advanced)
Implement a program that solves the Traveling Salesman Problem using backtracking with pruning (branch and bound), and measure how execution time changes as the number of cities increases.

### Exercise 7: Approximation Algorithm (Advanced)
Solve TSP using simulated annealing and compare solution quality with the exact solution (bitmask DP). Analyze the impact of parameter choices.


---

## Troubleshooting

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| Initialization error | Misconfigured settings file | Verify the path and format of the configuration file |
| Timeout | Network latency / insufficient resources | Adjust timeout values, add retry logic |
| Out of memory | Increased data volume | Introduce batch processing, implement pagination |
| Permission error | Insufficient access rights | Verify user permissions, review settings |
| Data inconsistency | Race conditions in concurrent processing | Introduce locking mechanisms, manage transactions |

### Debugging Procedure

1. **Check error messages**: Read the stack trace to identify the point of failure
2. **Establish reproduction steps**: Reproduce the error with minimal code
3. **Formulate hypotheses**: List possible causes
4. **Verify step by step**: Use log output or a debugger to test hypotheses
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
3. **Check I/O waits**: Examine disk and network I/O conditions
4. **Check concurrent connections**: Verify connection pool status

| Issue Type | Diagnostic Tool | Countermeasure |
|-----------|----------------|----------------|
| CPU load | cProfile, py-spy | Algorithm improvement, parallelization |
| Memory leak | tracemalloc, objgraph | Proper reference cleanup |
| I/O bottleneck | strace, iostat | Async I/O, caching |
| DB latency | EXPLAIN, slow query log | Indexing, query optimization |

---

## Design Decision Guide

### Selection Criteria Matrix

A summary of decision criteria for technology choices:

| Criterion | Prioritize When | Acceptable to Compromise When |
|-----------|-----------------|-------------------------------|
| Performance | Real-time processing, large-scale data | Admin panels, batch processing |
| Maintainability | Long-term operation, team development | Prototypes, short-term projects |
| Scalability | Services expected to grow | Internal tools, fixed user base |
| Security | Personal data, financial data | Public data, internal use |
| Development speed | MVP, time-to-market | Quality-focused, mission-critical |

### Architecture Pattern Selection

```
+---------------------------------------------------+
|          Architecture Selection Flow               |
+---------------------------------------------------+
|                                                   |
|  (1) Team size?                                   |
|    +-- Small (1-5 people) -> Monolith             |
|    +-- Large (10+ people) -> Go to (2)            |
|                                                   |
|  (2) Deployment frequency?                        |
|    +-- Once a week or less -> Monolith + modules  |
|    +-- Daily / multiple times -> Go to (3)        |
|                                                   |
|  (3) Team independence?                           |
|    +-- High -> Microservices                      |
|    +-- Moderate -> Modular monolith               |
|                                                   |
+---------------------------------------------------+
```

### Trade-off Analysis

Technical decisions always involve trade-offs. Analyze from the following perspectives:

**1. Short-term vs Long-term Cost**
- A method that is fast in the short term can become technical debt in the long run
- Conversely, over-engineering incurs high short-term costs and delays the project

**2. Consistency vs Flexibility**
- A unified technology stack lowers the learning curve
- Diverse technologies enable best-fit choices but increase operational costs

**3. Level of Abstraction**
- High abstraction provides reusability but can make debugging harder
- Low abstraction is intuitive but leads to code duplication

```python
# Template for recording design decisions
class ArchitectureDecisionRecord:
    """Create an ADR (Architecture Decision Record)"""

    def __init__(self, title: str):
        self.title = title
        self.context = ""
        self.decision = ""
        self.consequences = []
        self.alternatives = []

    def set_context(self, context: str):
        """Describe the background and problem"""
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

## Team Development

### Code Review Checklist

Key points to check in code reviews related to this topic:

- [ ] Naming conventions are consistent
- [ ] Error handling is appropriate
- [ ] Test coverage is sufficient
- [ ] There is no negative impact on performance
- [ ] There are no security concerns
- [ ] Documentation has been updated

### Knowledge Sharing Best Practices

| Method | Frequency | Target | Effect |
|--------|-----------|--------|--------|
| Pair programming | As needed | Complex tasks | Immediate feedback |
| Tech talks | Weekly | Entire team | Horizontal knowledge transfer |
| ADR (Decision Records) | As needed | Future members | Decision transparency |
| Retrospectives | Bi-weekly | Entire team | Continuous improvement |
| Mob programming | Monthly | Important designs | Consensus building |

### Technical Debt Management

```
Priority Matrix:

        Impact High
          |
    +-----+-----+
    | Plan |Imme-|
    | ned  |diate|
    |      |     |
    +------+-----+
    |Record|Next |
    | only |Sprint|
    |      |     |
    +------+-----+
          |
        Impact Low
    Frequency Low  Frequency High
```

---

## Security Considerations

### Common Vulnerabilities and Countermeasures

| Vulnerability | Risk Level | Countermeasure | Detection Method |
|--------------|-----------|----------------|-----------------|
| Injection attacks | High | Input validation, parameterized queries | SAST/DAST |
| Authentication flaws | High | Multi-factor authentication, session management hardening | Penetration testing |
| Sensitive data exposure | High | Encryption, access control | Security audit |
| Misconfiguration | Medium | Security headers, principle of least privilege | Configuration scanning |
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
        """Generate a cryptographically secure token"""
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
        """Sanitize input"""
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
- [ ] Vulnerability scanning of dependencies has been performed
- [ ] Error messages do not contain internal information

---

## Migration Guide

### Notes for Version Upgrades

| Version | Major Changes | Migration Work | Impact Scope |
|---------|--------------|----------------|-------------|
| v1.x -> v2.x | API design overhaul | Endpoint changes | All clients |
| v2.x -> v3.x | Authentication method change | Token format update | Auth-related |
| v3.x -> v4.x | Data model change | Run migration scripts | DB-related |

### Gradual Migration Steps

```python
# Migration script template
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Callable

logger = logging.getLogger(__name__)

class MigrationRunner:
    """Gradual migration execution engine"""

    def __init__(self, migration_dir: str):
        self.migration_dir = Path(migration_dir)
        self.migrations: List[Dict] = []
        self.completed: List[str] = []

    def register(self, version: str, description: str,
                 up: Callable, down: Callable):
        """Register a migration"""
        self.migrations.append({
            'version': version,
            'description': description,
            'up': up,
            'down': down,
            'registered_at': datetime.now().isoformat()
        })

    def run_up(self, target_version: str = None):
        """Run migrations (upgrade)"""
        for migration in self.migrations:
            if migration['version'] in self.completed:
                continue
            logger.info(f"Running: {migration['version']} - "
                       f"{migration['description']}")
            try:
                migration['up']()
                self.completed.append(migration['version'])
                logger.info(f"Completed: {migration['version']}")
            except Exception as e:
                logger.error(f"Failed: {migration['version']}: {e}")
                raise
            if target_version and migration['version'] == target_version:
                break

    def run_down(self, target_version: str):
        """Rollback migrations"""
        for migration in reversed(self.migrations):
            if migration['version'] not in self.completed:
                continue
            if migration['version'] == target_version:
                break
            logger.info(f"Rolling back: {migration['version']}")
            migration['down']()
            self.completed.remove(migration['version'])

    def status(self) -> Dict:
        """Check migration status"""
        return {
            'total': len(self.migrations),
            'completed': len(self.completed),
            'pending': len(self.migrations) - len(self.completed),
            'versions': {
                m['version']: 'completed'
                if m['version'] in self.completed else 'pending'
                for m in self.migrations
            }
        }
```

### Rollback Plan

Always prepare a rollback plan for migration work:

1. **Data backup**: Take a full backup before migration
2. **Test environment verification**: Pre-validate in an environment equivalent to production
3. **Gradual rollout**: Deploy incrementally using canary releases
4. **Enhanced monitoring**: Shorten monitoring intervals during migration
5. **Clear decision criteria**: Define rollback criteria in advance
---

## FAQ

### Q1: How do you prove the correctness of a greedy algorithm?
**A**: Three methods: (1) Exchange argument: Show that transforming the optimal solution into the greedy solution does not make it worse. (2) Induction: Show that optimality is maintained at each step. (3) Matroid theory: Show that the problem has a matroid structure. In practice, the most efficient first step is to look for counterexamples. Try comparing results with exhaustive search on small cases.

### Q2: How do you choose between backtracking and dynamic programming?
**A**: Use DP if subproblems overlap. Use backtracking if there is no overlap and you need to enumerate all patterns. Problems requiring "enumerate all solutions" suit backtracking. Problems requiring "find only the optimal value" suit DP. However, note that backtracking + memoization = top-down DP.

### Q3: How do you handle NP-hard problems in practice?
**A**: (1) Approximation algorithms (guarantee within a constant factor of optimal). (2) Heuristics (simulated annealing, genetic algorithms). (3) Restrict problem size (bitmask enumeration for n<=20). (4) Reduce to a special case. It is important to first analyze the problem structure and check for exploitable special properties.

### Q4: How do you evaluate the effectiveness of pruning?
**A**: (1) Count search nodes and compare with no pruning. (2) Measure execution time. (3) Analyze the theoretical complexity improvement. Good pruning reduces the search space exponentially. However, if the cost of computing the pruning criterion itself is too high, it can be counterproductive, so it is important to use bounds that are easy to compute.

### Q5: Is it a problem to use DP when greedy would work?
**A**: You will get the correct answer, but with unnecessarily higher complexity. For example, interval scheduling is O(n log n) with a greedy approach but O(n^2) with DP. However, if you are not confident about the correctness of the greedy approach, using DP as a safe strategy is a reasonable decision.

### Q6: What kinds of problems can Meet in the Middle be used for?
**A**: Problems where the input can be split into two halves, each half can be exhaustively searched independently, and the results can be merged. A typical example is the subset sum problem (n<=40). Each half requires only 2^(n/2) operations, reducing the total from 2^n to 2^(n/2) x 2. Merge by sorting and binary search, or by using a hash map.

---

## Summary

| Technique | Complexity | Optimality | Use Case |
|-----------|-----------|-----------|----------|
| Greedy | O(n log n)~ | Conditionally optimal | Interval scheduling, MST, shortest path |
| Backtracking | O(exp)~ | Complete search (accelerated by pruning) | N-Queens, Sudoku, combination enumeration |
| Bitmask enumeration | O(2^n x n) | Complete | Subset problems with n<=20 |
| Meet in the Middle | O(2^(n/2) x n) | Complete | Subset problems with n<=40 |
| Branch and bound | O(exp) * | Complete (depends on pruning efficiency) | Knapsack, TSP |
| Simulated annealing | User-specified | Approximate (no guarantee) | NP-hard optimization problems |
| Approximation algorithms | Polynomial | Approximate (with guarantee) | Vertex cover, set cover |

---

## Recommended Next Guides

---

## References
1. Cormen, T. H. et al. "Introduction to Algorithms." Chapters 16-17.
2. Skiena, S. S. "The Algorithm Design Manual." Chapters 8-9.
3. Papadimitriou, C., Steiglitz, K. "Combinatorial Optimization." Dover, 1998.
4. Aarts, E., Korst, J. "Simulated Annealing and Boltzmann Machines." 1989.
5. Cook, W. "In Pursuit of the Traveling Salesman." Princeton University Press, 2012.
6. Lawler, E. L. et al. "The Traveling Salesman Problem." Wiley, 1985.
