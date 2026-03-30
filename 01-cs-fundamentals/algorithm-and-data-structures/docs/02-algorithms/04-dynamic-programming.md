# Dynamic Programming

> Systematically understand the design technique for efficiently solving overlapping subproblems, through memoization, bottom-up approaches, and representative problems

## What You Will Learn

1. Distinguish between and apply **memoization (top-down) and bottom-up (tabulation)** approaches
2. Identify the conditions for DP applicability: **optimal substructure and overlapping subproblems**
3. Accurately implement **knapsack, LCS, LIS, edit distance, interval DP, and bitmask DP**
4. Master the framework of **DP state design, recurrence derivation, and space optimization**


## Prerequisites

Having the following knowledge will deepen your understanding of this guide:

- Basic programming knowledge
- Understanding of related foundational concepts
- Familiarity with the content in [Shortest Path Algorithms](./03-shortest-path.md)

---

## 1. Principles of Dynamic Programming

```
Two conditions for DP applicability:

1. Optimal Substructure
   -> The optimal solution to the problem can be composed from optimal solutions to subproblems

2. Overlapping Subproblems
   -> The same subproblems appear multiple times

   Recursion tree for fib(5):
                    fib(5)
                   /      \
              fib(4)       fib(3)      <- fib(3) is repeated!
             /     \       /    \
         fib(3)  fib(2) fib(2) fib(1)  <- fib(2) is repeated!
        /    \
    fib(2)  fib(1)

   Without memoization: O(2^n) -> With memoization: O(n)
```

### How to Identify Problems Suited for DP

```
Suspect DP when the problem statement contains these keywords:

  - "maximum" "minimum" "longest" "shortest" "optimal"
  - "number of ways" "number of combinations"
  - "is it possible" (yes/no decision problems)
  - "subsequence" "substring" "subset"
  - "minimize cost" "maximize profit"

Decision flow:
  1. Can it be solved recursively? -> YES, proceed
  2. Do subproblems overlap? -> YES, use DP
  3. No overlap? -> Divide and conquer (no memoization needed)
  4. Local optimum = global optimum? -> Consider greedy first
```

---

## 2. Memoization (Top-Down)

Recursion + caching of results. Preserves the natural recursive structure as-is.

```python
from functools import lru_cache

# Method 1: Memoization with a dictionary
def fib_memo(n: int, memo: dict = None) -> int:
    if memo is None:
        memo = {}
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib_memo(n - 1, memo) + fib_memo(n - 2, memo)
    return memo[n]

# Method 2: lru_cache decorator (Pythonic)
@lru_cache(maxsize=None)
def fib_cached(n: int) -> int:
    if n <= 1:
        return n
    return fib_cached(n - 1) + fib_cached(n - 2)

# Method 3: Generic memoization decorator
def memoize(func):
    """Generic memoization decorator (for hashable arguments)"""
    cache = {}
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    wrapper.cache = cache
    return wrapper

@memoize
def fib_memoized(n: int) -> int:
    if n <= 1:
        return n
    return fib_memoized(n - 1) + fib_memoized(n - 2)

print(fib_memo(50))      # 12586269025
print(fib_cached(50))    # 12586269025
print(fib_memoized(50))  # 12586269025
```

---

## 3. Bottom-Up (Tabulation)

Solves subproblems from smallest to largest, filling in a table. No recursion overhead.

```python
def fib_bottom_up(n: int) -> int:
    """Bottom-up DP - O(n) time, O(n) space"""
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]

def fib_optimized(n: int) -> int:
    """Space-optimized - O(n) time, O(1) space"""
    if n <= 1:
        return n
    prev2, prev1 = 0, 1
    for _ in range(2, n + 1):
        prev2, prev1 = prev1, prev2 + prev1
    return prev1

print(fib_bottom_up(50))   # 12586269025
print(fib_optimized(50))   # 12586269025
```

```
Memoization vs Bottom-Up:

Top-Down (Memoization):          Bottom-Up:
  fib(5)                           dp[0]=0
    -> fib(4)                      dp[1]=1
      -> fib(3)                    dp[2]=1
        -> fib(2)                  dp[3]=2
          -> fib(1) = 1            dp[4]=3
          -> fib(0) = 0            dp[5]=5
        -> result=1 (cached)      Answer: dp[5]
      -> fib(2) -> cache hit!
    -> fib(3) -> cache hit!
  Answer: 5
```

---

## 4. 0/1 Knapsack Problem

Fill a knapsack with weight limit W to maximize total value, choosing from items with weight w and value v.

```
Items: [(weight=2, value=3), (weight=3, value=4), (weight=4, value=5), (weight=5, value=6)]
Capacity: W = 8

DP table dp[i][w] = max value using items 0..i with capacity w:

       w: 0  1  2  3  4  5  6  7  8
item 0:   0  0  3  3  3  3  3  3  3
item 1:   0  0  3  4  4  7  7  7  7
item 2:   0  0  3  4  5  7  8  9  9
item 3:   0  0  3  4  5  7  8  9  10

Answer: dp[3][8] = 10
```

```python
def knapsack_01(weights: list, values: list, W: int) -> int:
    """0/1 Knapsack - O(nW)"""
    n = len(weights)
    dp = [[0] * (W + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(W + 1):
            # Don't include item i-1
            dp[i][w] = dp[i - 1][w]
            # Include item i-1 (if capacity allows)
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i][w],
                               dp[i - 1][w - weights[i - 1]] + values[i - 1])

    return dp[n][W]

# Space-optimized version (1D DP)
def knapsack_01_optimized(weights: list, values: list, W: int) -> int:
    """0/1 Knapsack (space-optimized) - O(nW) time, O(W) space"""
    dp = [0] * (W + 1)

    for i in range(len(weights)):
        # Update in reverse order (to avoid using the same item twice)
        for w in range(W, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])

    return dp[W]

# Recovering the selected items
def knapsack_01_with_items(weights: list, values: list, W: int) -> tuple:
    """0/1 Knapsack + selected item recovery"""
    n = len(weights)
    dp = [[0] * (W + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(W + 1):
            dp[i][w] = dp[i - 1][w]
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i][w],
                               dp[i - 1][w - weights[i - 1]] + values[i - 1])

    # Recover selected items (backtrack)
    selected = []
    w = W
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected.append(i - 1)  # Item i-1 was selected
            w -= weights[i - 1]

    return dp[n][W], selected[::-1]

weights = [2, 3, 4, 5]
values = [3, 4, 5, 6]
print(knapsack_01(weights, values, 8))            # 10
print(knapsack_01_optimized(weights, values, 8))  # 10
max_val, items = knapsack_01_with_items(weights, values, 8)
print(f"Max value: {max_val}, Selected: {items}")  # Max value: 10, Selected: [0, 1, 2]
```

### Unbounded Knapsack (Unlimited Use of Each Item)

```python
def knapsack_unbounded(weights: list, values: list, W: int) -> int:
    """Unbounded Knapsack - O(nW)
    Each item can be used any number of times
    """
    dp = [0] * (W + 1)

    for i in range(len(weights)):
        # Update in forward order (allows using the same item multiple times)
        for w in range(weights[i], W + 1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])

    return dp[W]

# 0/1 uses reverse order, unbounded uses forward order -- this distinction is critical!
print(knapsack_unbounded([2, 3, 4, 5], [3, 4, 5, 6], 8))  # 12 (weight 2 x 4)
```

---

## 5. Longest Common Subsequence (LCS)

Finds the longest common subsequence of two strings. The foundation of the diff command.

```
X = "ABCBDAB"
Y = "BDCAB"

DP table:
     ""  B  D  C  A  B
  ""  0  0  0  0  0  0
  A   0  0  0  0  1  1
  B   0  1  1  1  1  2
  C   0  1  1  2  2  2
  B   0  1  1  2  2  3
  D   0  1  2  2  2  3
  A   0  1  2  2  3  3
  B   0  1  2  2  3  4

LCS = "BCAB" (length 4)

Recurrence:
  If X[i] == Y[j]: dp[i][j] = dp[i-1][j-1] + 1
  If X[i] != Y[j]: dp[i][j] = max(dp[i-1][j], dp[i][j-1])
```

```python
def lcs(X: str, Y: str) -> tuple:
    """Longest Common Subsequence - O(mn)"""
    m, n = len(X), len(Y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Reconstruction
    result = []
    i, j = m, n
    while i > 0 and j > 0:
        if X[i - 1] == Y[j - 1]:
            result.append(X[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return dp[m][n], ''.join(reversed(result))

# Space-optimized version (length only)
def lcs_length_optimized(X: str, Y: str) -> int:
    """LCS length only - O(mn) time, O(min(m,n)) space"""
    if len(X) < len(Y):
        X, Y = Y, X  # Make Y the shorter one

    m, n = len(X), len(Y)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)

    return prev[n]

length, subseq = lcs("ABCBDAB", "BDCAB")
print(f"Length: {length}, LCS: {subseq}")  # Length: 4, LCS: BCAB
```

### Practical Application of LCS: Computing Diffs

```python
def compute_diff(original: list, modified: list) -> list:
    """Compute the diff between two texts using LCS"""
    m, n = len(original), len(modified)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if original[i-1] == modified[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    # Generate diff
    diff = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and original[i-1] == modified[j-1]:
            diff.append(('  ', original[i-1]))  # Unchanged
            i -= 1
            j -= 1
        elif j > 0 and (i == 0 or dp[i][j-1] >= dp[i-1][j]):
            diff.append(('+ ', modified[j-1]))   # Added
            j -= 1
        else:
            diff.append(('- ', original[i-1]))   # Deleted
            i -= 1

    return diff[::-1]

original = ["def hello():", "    print('hello')", "    return True"]
modified = ["def hello():", "    print('hello, world')", "    return True", "    # comment"]
for prefix, line in compute_diff(original, modified):
    print(f"{prefix}{line}")
```

---

## 6. Coin Change (Minimum Number of Coins)

```python
def coin_change(coins: list, amount: int) -> int:
    """Coin change (minimum coins) - O(n * amount)"""
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i and dp[i - coin] + 1 < dp[i]:
                dp[i] = dp[i - coin] + 1

    return dp[amount] if dp[amount] != float('inf') else -1

def coin_change_ways(coins: list, amount: int) -> int:
    """Coin change (number of ways) - O(n * amount)"""
    dp = [0] * (amount + 1)
    dp[0] = 1

    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]

    return dp[amount]

# Usage examples
print(coin_change([1, 5, 10, 25], 30))       # 2 (25+5)
print(coin_change([3, 7], 5))                  # -1 (impossible)
print(coin_change_ways([1, 5, 10, 25], 30))   # 18 ways
```

---

## 7. Longest Increasing Subsequence (LIS)

```python
import bisect

def lis_dp(arr: list) -> int:
    """LIS (DP version) - O(n^2)"""
    n = len(arr)
    dp = [1] * n
    for i in range(1, n):
        for j in range(i):
            if arr[j] < arr[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)

def lis_binary_search(arr: list) -> int:
    """LIS (binary search version) - O(n log n)"""
    tails = []  # tails[i] = smallest tail of an IS of length i+1
    for num in arr:
        pos = bisect.bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    return len(tails)

def lis_with_reconstruction(arr: list) -> tuple:
    """LIS + actual subsequence reconstruction - O(n log n)"""
    n = len(arr)
    if n == 0:
        return 0, []

    tails = []
    tails_idx = []      # Original array index for each position in tails
    prev_idx = [-1] * n  # Previous element index in the LIS for each element

    for i, num in enumerate(arr):
        pos = bisect.bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
            tails_idx.append(i)
        else:
            tails[pos] = num
            tails_idx[pos] = i

        if pos > 0:
            prev_idx[i] = tails_idx[pos - 1]

    # Reconstruction
    length = len(tails)
    result = []
    idx = tails_idx[-1]
    while idx != -1:
        result.append(arr[idx])
        idx = prev_idx[idx]

    return length, result[::-1]

data = [10, 9, 2, 5, 3, 7, 101, 18]
print(lis_dp(data))                          # 4
print(lis_binary_search(data))               # 4
length, subseq = lis_with_reconstruction(data)
print(f"Length: {length}, LIS: {subseq}")    # Length: 4, LIS: [2, 3, 7, 18]
```

---

## 8. Edit Distance (Levenshtein Distance)

```python
def edit_distance(s1: str, s2: str) -> int:
    """Edit distance - O(mn)"""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # Delete
                    dp[i][j - 1],      # Insert
                    dp[i - 1][j - 1],  # Replace
                )

    return dp[m][n]

def edit_distance_with_operations(s1: str, s2: str) -> tuple:
    """Edit distance + operation sequence reconstruction"""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    # Reconstruct operation sequence
    operations = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s1[i-1] == s2[j-1]:
            operations.append(('keep', s1[i-1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            operations.append(('replace', s1[i-1], s2[j-1]))
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            operations.append(('insert', s2[j-1]))
            j -= 1
        else:
            operations.append(('delete', s1[i-1]))
            i -= 1

    return dp[m][n], operations[::-1]

print(edit_distance("kitten", "sitting"))  # 3
dist, ops = edit_distance_with_operations("kitten", "sitting")
print(f"Distance: {dist}")
for op in ops:
    print(f"  {op}")
# ('replace', 'k', 's'), ('keep', 'i'), ('keep', 't'), ('keep', 't'),
# ('replace', 'e', 'i'), ('keep', 'n'), ('insert', 'g')
```

### Practical Application of Edit Distance: Fuzzy Search

```python
def fuzzy_search(query: str, dictionary: list, max_distance: int = 2) -> list:
    """Fuzzy search: return words within the distance threshold"""
    results = []
    for word in dictionary:
        dist = edit_distance(query.lower(), word.lower())
        if dist <= max_distance:
            results.append((word, dist))
    results.sort(key=lambda x: x[1])
    return results

dictionary = ["python", "pytorch", "pycharm", "piton", "prism", "prison"]
print(fuzzy_search("pyton", dictionary))
# [('piton', 1), ('python', 1), ('prism', 2), ('prison', 2)]
```

---

## 9. Interval DP

A technique that computes the optimal solution for an interval [l, r] from the solutions of smaller intervals.

### Matrix Chain Multiplication

```python
def matrix_chain_order(dims: list) -> tuple:
    """Minimum number of multiplications for matrix chain multiplication - O(n^3)
    dims: list of matrix dimensions (n+1 elements)
    Matrix A_i has dimensions dims[i] x dims[i+1]
    """
    n = len(dims) - 1
    dp = [[0] * n for _ in range(n)]
    split = [[0] * n for _ in range(n)]

    # l: interval length (2 or more)
    for l in range(2, n + 1):
        for i in range(n - l + 1):
            j = i + l - 1
            dp[i][j] = float('inf')
            for k in range(i, j):
                cost = dp[i][k] + dp[k+1][j] + dims[i] * dims[k+1] * dims[j+1]
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    split[i][j] = k

    return dp[0][n-1], split

def print_optimal_parens(split: list, i: int, j: int) -> str:
    """Print the optimal parenthesization"""
    if i == j:
        return f"A{i}"
    k = split[i][j]
    left = print_optimal_parens(split, i, k)
    right = print_optimal_parens(split, k + 1, j)
    return f"({left} x {right})"

# Matrices: A0(30x35), A1(35x15), A2(15x5), A3(5x10), A4(10x20), A5(20x25)
dims = [30, 35, 15, 5, 10, 20, 25]
min_ops, split = matrix_chain_order(dims)
print(f"Minimum multiplications: {min_ops}")  # 15125
print(f"Optimal parenthesization: {print_optimal_parens(split, 0, 5)}")
# ((A0 x (A1 x A2)) x ((A3 x A4) x A5))
```

### Palindrome Partitioning

```python
def min_palindrome_cuts(s: str) -> int:
    """Minimum cuts to partition a string into palindromes - O(n^2)"""
    n = len(s)
    if n <= 1:
        return 0

    # is_pal[i][j] = whether s[i..j] is a palindrome
    is_pal = [[False] * n for _ in range(n)]
    for i in range(n):
        is_pal[i][i] = True
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if length == 2:
                is_pal[i][j] = (s[i] == s[j])
            else:
                is_pal[i][j] = (s[i] == s[j] and is_pal[i+1][j-1])

    # dp[i] = minimum cuts to partition s[0..i] into palindromes
    dp = list(range(n))  # Worst case: split into individual characters
    for i in range(1, n):
        if is_pal[0][i]:
            dp[i] = 0
            continue
        for j in range(1, i + 1):
            if is_pal[j][i]:
                dp[i] = min(dp[i], dp[j-1] + 1)

    return dp[n-1]

print(min_palindrome_cuts("aab"))       # 1 ("aa" + "b")
print(min_palindrome_cuts("abcba"))     # 0 (entire string is a palindrome)
print(min_palindrome_cuts("abcdef"))    # 5 (split at each character)
```

---

## 10. Bitmask DP

A technique that represents states as bits of an integer, efficiently managing subsets.

### Traveling Salesman Problem (TSP)

```python
def tsp(dist_matrix: list) -> tuple:
    """Traveling Salesman Problem - O(2^n * n^2)
    dist_matrix[i][j]: distance from city i to city j
    Returns: (minimum distance, route)
    """
    n = len(dist_matrix)
    INF = float('inf')

    # dp[S][v] = minimum distance when the set S of cities has been visited
    #            and currently at city v
    # S is a bitmask: bit i is set = city i has been visited
    dp = [[INF] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]
    dp[1][0] = 0  # Start from city 0

    for S in range(1 << n):
        for u in range(n):
            if dp[S][u] == INF:
                continue
            if not (S & (1 << u)):
                continue
            for v in range(n):
                if S & (1 << v):
                    continue  # Already visited
                new_S = S | (1 << v)
                new_dist = dp[S][u] + dist_matrix[u][v]
                if new_dist < dp[new_S][v]:
                    dp[new_S][v] = new_dist
                    parent[new_S][v] = u

    # After visiting all cities, return to the starting point
    full = (1 << n) - 1
    min_dist = INF
    last = -1
    for v in range(n):
        total = dp[full][v] + dist_matrix[v][0]
        if total < min_dist:
            min_dist = total
            last = v

    # Route reconstruction
    path = [0]
    S = full
    v = last
    while v != 0:
        path.append(v)
        u = parent[S][v]
        S ^= (1 << v)
        v = u
    path.append(0)
    path.reverse()

    return min_dist, path

# 4-city example
dist_matrix = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0],
]
min_dist, path = tsp(dist_matrix)
print(f"Minimum tour distance: {min_dist}")  # 80
print(f"Route: {path}")                       # [0, 1, 3, 2, 0]
```

### Bitmask DP: Optimal Assignment over Sets

```python
def min_cost_assignment(cost: list) -> int:
    """Minimum cost assignment problem - O(2^n * n)
    cost[i][j]: cost of assigning task j to person i
    Assign one task to each person, covering all tasks
    """
    n = len(cost)
    INF = float('inf')
    dp = [INF] * (1 << n)
    dp[0] = 0

    for mask in range(1 << n):
        person = bin(mask).count('1')  # How many people have been assigned
        if person >= n:
            continue
        for task in range(n):
            if mask & (1 << task):
                continue  # This task is already assigned
            new_mask = mask | (1 << task)
            dp[new_mask] = min(dp[new_mask], dp[mask] + cost[person][task])

    return dp[(1 << n) - 1]

cost_matrix = [
    [9, 2, 7, 8],
    [6, 4, 3, 7],
    [5, 8, 1, 8],
    [7, 6, 9, 4],
]
print(min_cost_assignment(cost_matrix))  # 13 (2+3+1+7? or 2+4+1+4=11)
```

---

## 11. Tree DP

DP on tree-structured graphs. Each vertex's value is computed from its children's values.

```python
def tree_dp_max_independent_set(tree: dict, root: int) -> int:
    """Maximum independent set size on a tree - O(V)
    Independent set: a subset of vertices with no adjacent vertices
    tree: {node: [children]}
    """
    # dp[v][0] = max independent set size of v's subtree when v is NOT included
    # dp[v][1] = max independent set size of v's subtree when v IS included
    dp = {}

    def dfs(v, parent):
        dp[v] = [0, 1]  # [not included, included]

        for child in tree.get(v, []):
            if child == parent:
                continue
            dfs(child, v)
            dp[v][0] += max(dp[child][0], dp[child][1])  # Child can be included or not
            dp[v][1] += dp[child][0]  # If v is included, children must not be

    dfs(root, -1)
    return max(dp[root][0], dp[root][1])

# Tree:      1
#           / \
#          2   3
#         / \
#        4   5
tree = {1: [2, 3], 2: [1, 4, 5], 3: [1], 4: [2], 5: [2]}
print(tree_dp_max_independent_set(tree, 1))  # 3 (vertices 3, 4, 5)


def tree_diameter(tree: dict, root: int) -> int:
    """Tree diameter (length of the longest path) - O(V)"""
    diameter = [0]

    def dfs(v, parent) -> int:
        """Return the longest distance from the root of v's subtree"""
        max1 = max2 = 0  # Largest and second largest

        for child in tree.get(v, []):
            if child == parent:
                continue
            depth = dfs(child, v) + 1
            if depth > max1:
                max2 = max1
                max1 = depth
            elif depth > max2:
                max2 = depth

        diameter[0] = max(diameter[0], max1 + max2)
        return max1

    dfs(root, -1)
    return diameter[0]

print(tree_diameter(tree, 1))  # 3 (4->2->1->3 or 5->2->1->3)
```

---

## 12. Probability DP / Expected Value DP

```python
def expected_coin_flips(target_heads: int) -> float:
    """Expected number of flips to get target_heads heads with a fair coin
    dp[i] = expected number of flips to get i more heads
    """
    dp = [0.0] * (target_heads + 1)
    for i in range(1, target_heads + 1):
        # Heads: transition to dp[i-1] (probability 1/2)
        # Tails: transition to dp[i] (probability 1/2) -> 1 wasted flip
        # dp[i] = 1 + 0.5 * dp[i-1] + 0.5 * dp[i]
        # -> dp[i] = 2 + dp[i-1]
        dp[i] = 2 + dp[i - 1]
    return dp[target_heads]

print(expected_coin_flips(3))  # 6.0 (on average 6 flips to get 3 heads)


def dice_probability(n_dice: int, target: int) -> float:
    """Probability that the sum of n dice equals target"""
    # dp[i][j] = number of ways to get sum j with i dice
    dp = [[0] * (target + 1) for _ in range(n_dice + 1)]
    dp[0][0] = 1

    for i in range(1, n_dice + 1):
        for j in range(i, min(6 * i, target) + 1):
            for face in range(1, 7):
                if j - face >= 0:
                    dp[i][j] += dp[i-1][j-face]

    total_outcomes = 6 ** n_dice
    return dp[n_dice][target] / total_outcomes if target <= 6 * n_dice else 0

print(f"2 dice sum 7: {dice_probability(2, 7):.4f}")   # 0.1667
print(f"3 dice sum 10: {dice_probability(3, 10):.4f}")  # 0.1250
```

---

## 13. DP Design Framework

```
+---------------------------------------------+
|       5 Steps to Solve DP Problems           |
+---------------------------------------------+
| 1. Define the state                          |
|    -> Clearly specify what dp[i] / dp[i][j]  |
|       represents                              |
|                                              |
| 2. Derive the recurrence                     |
|    -> dp[i] = f(dp[i-1], dp[i-2], ...)      |
|                                              |
| 3. Set the base cases                        |
|    -> dp[0] = ?, dp[1] = ?                  |
|                                              |
| 4. Determine the computation order           |
|    -> Bottom-up fill order                   |
|                                              |
| 5. Extract the answer                        |
|    -> dp[n] / max(dp) / reconstruction       |
+---------------------------------------------+
```

### Design Example: Staircase Climbing Problem

```python
# Problem: How many ways to climb n stairs taking 1 or 2 steps at a time?
#
# Step 1. State definition: dp[i] = number of ways to reach step i
# Step 2. Recurrence:       dp[i] = dp[i-1] + dp[i-2]
#                           (1 step from i-1 or 2 steps from i-2)
# Step 3. Base cases:       dp[0] = 1 (on the ground: 1 way)
#                           dp[1] = 1 (step 1: 1 way)
# Step 4. Computation order: i = 2, 3, ..., n (small to large)
# Step 5. Answer:           dp[n]

def climb_stairs(n: int) -> int:
    if n <= 1:
        return 1
    dp = [0] * (n + 1)
    dp[0] = dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]

print(climb_stairs(10))  # 89
```

---

## 14. Memoization vs Bottom-Up Comparison

| Property | Memoization (Top-Down) | Bottom-Up |
|:---|:---|:---|
| Implementation style | Recursion + cache | Loop + table |
| Subproblems computed | Only those needed | All subproblems |
| Stack overflow | Possible | Not possible |
| Space optimization | Difficult | Possible (dimension reduction) |
| Coding ease | Natural recursive thinking | Need to determine fill order |
| Debugging | Somewhat difficult | Easy to inspect the table |
| Constant factor | Function call overhead | Loops are faster |

## Classic DP Patterns

| Pattern | Representative Problem | State | Complexity |
|:---|:---|:---|:---|
| 1D DP | Fibonacci, stairs | dp[i] | O(n) |
| 2D DP | LCS, edit distance | dp[i][j] | O(mn) |
| Knapsack | 0/1 Knapsack | dp[i][w] | O(nW) |
| Interval DP | Matrix chain multiplication | dp[l][r] | O(n^3) |
| Bitmask DP | Traveling salesman | dp[S][v] | O(2^n * n) |
| Tree DP | Max independent set on tree | dp[v][0/1] | O(V) |
| Probability DP | Expected value computation | dp[state] | Problem-dependent |
| Digit DP | Count numbers <= N satisfying conditions | dp[pos][tight][...] | O(D * states) |

---

## 15. Digit DP

Counts non-negative integers up to N that satisfy specific conditions.

```python
def count_numbers_with_digit_sum(N: int, target_sum: int) -> int:
    """Count non-negative integers <= N whose digit sum equals target_sum"""
    digits = [int(d) for d in str(N)]
    n = len(digits)

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def dp(pos, remaining_sum, tight, started):
        """
        pos: current digit position
        remaining_sum: remaining digit sum needed
        tight: whether the upper bound constraint is active
        started: whether we have passed leading zeros
        """
        if remaining_sum < 0:
            return 0
        if pos == n:
            return 1 if remaining_sum == 0 and started else 0

        limit = digits[pos] if tight else 9
        count = 0

        for d in range(0, limit + 1):
            count += dp(
                pos + 1,
                remaining_sum - d,
                tight and (d == limit),
                started or (d > 0),
            )

        return count

    return dp(0, target_sum, True, False)

# Count of numbers <= 1000 with digit sum 10
print(count_numbers_with_digit_sum(1000, 10))  # 63
```

---

## 16. Anti-Patterns

### Anti-Pattern 1: Recursion Without Memoization

```python
# BAD: No memoization -> O(2^n) explosion
def fib_bad(n):
    if n <= 1:
        return n
    return fib_bad(n-1) + fib_bad(n-2)
# fib_bad(40) takes tens of seconds

# GOOD: O(n) with memoization
@lru_cache(maxsize=None)
def fib_good(n):
    if n <= 1:
        return n
    return fib_good(n-1) + fib_good(n-2)
# fib_good(1000) completes instantly
```

### Anti-Pattern 2: Forward Update in 0/1 Knapsack

```python
# BAD: Forward update in 1D DP -> same item used multiple times
def bad_knapsack(weights, values, W):
    dp = [0] * (W + 1)
    for i in range(len(weights)):
        for w in range(weights[i], W + 1):  # Forward -> becomes unbounded knapsack
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    return dp[W]

# GOOD: Reverse update
def good_knapsack(weights, values, W):
    dp = [0] * (W + 1)
    for i in range(len(weights)):
        for w in range(W, weights[i] - 1, -1):  # Reverse -> each item used at most once
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    return dp[W]
```

### Anti-Pattern 3: Ambiguous DP State Definition

```python
# BAD: Implementing without a clear definition of what dp[i] represents
dp = [0] * n
for i in range(n):
    dp[i] = ???  # What are we computing...

# GOOD: Clearly define the state before implementing
# dp[i] = "length of the longest increasing subsequence ending at element i"
dp = [1] * n
for i in range(n):
    for j in range(i):
        if arr[j] < arr[i]:
            dp[i] = max(dp[i], dp[j] + 1)
```

### Anti-Pattern 4: Unnecessary Dimensions in State Design

```python
# BAD: 3D for knapsack (item x capacity x selection count)
# -> Selection count is often unnecessary

# GOOD: Design with the minimum necessary dimensions
# For 0/1 knapsack, dp[w] with 1 dimension suffices (after space optimization)
```

### Anti-Pattern 5: Floating-Point DP

```python
# BAD: Using floating-point as keys -> precision issues
memo = {}
def bad_dp(x):
    if x in memo:  # 0.1 + 0.2 != 0.3 problem
        return memo[x]
    ...

# GOOD: Convert to integers or apply appropriate rounding
def good_dp(x_cents):  # Integer in cents
    ...
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

## Troubleshooting

### Common Errors and Solutions

| Error | Cause | Solution |
|--------|------|--------|
| Initialization error | Configuration file issues | Verify configuration file path and format |
| Timeout | Network latency/resource shortage | Adjust timeout values, add retry logic |
| Out of memory | Data volume growth | Introduce batch processing, implement pagination |
| Permission error | Insufficient access rights | Check execution user permissions, review configuration |
| Data inconsistency | Concurrent processing conflicts | Introduce locking mechanisms, manage transactions |

### Debugging Steps

1. **Check error messages**: Read the stack trace to identify the point of failure
2. **Establish reproduction steps**: Reproduce the error with minimal code
3. **Formulate hypotheses**: List possible causes
4. **Verify step by step**: Use log output or a debugger to test hypotheses
5. **Fix and regression test**: After fixing, run tests on related areas as well

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
    """Decorator that logs function inputs and outputs"""
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
2. **Check memory usage**: Look for memory leaks
3. **Check I/O waits**: Examine disk and network I/O status
4. **Check concurrent connections**: Verify connection pool status

| Problem Type | Diagnostic Tool | Countermeasure |
|-----------|-----------|------|
| CPU load | cProfile, py-spy | Algorithm improvement, parallelization |
| Memory leak | tracemalloc, objgraph | Proper release of references |
| I/O bottleneck | strace, iostat | Async I/O, caching |
| DB latency | EXPLAIN, slow query log | Indexing, query optimization |
---

## 17. FAQ

### Q1: What is the difference between DP and divide and conquer?

**A:** Both decompose problems, but the core difference is "subproblem overlap." Divide and conquer (e.g., merge sort) has independent, non-overlapping subproblems. DP handles cases where the same subproblems appear multiple times, caching results for reuse. If there is no overlap, use divide and conquer; if there is, use DP.

### Q2: How do you determine the number of dimensions (states) in DP?

**A:** The number of dimensions is the minimum number of parameters needed to uniquely represent the problem. Fibonacci has 1 parameter n (1D), LCS has 2 string positions i, j (2D). More dimensions increase expressiveness but also complexity, so find the right balance of necessary and sufficient dimensions.

### Q3: What if stack overflow occurs with memoized recursion?

**A:** Three countermeasures: (1) Increase `sys.setrecursionlimit()` (stopgap). (2) Rewrite as bottom-up DP (recommended). (3) Convert tail recursion to a loop if possible. In Python, (2) is the safest approach.

### Q4: How do you debug a DP table?

**A:** Hand-calculate the table for small inputs and compare with program output. For 2D DP, print the entire table with `for row in dp: print(row)`. Check in order: is the recurrence correct, are the base cases correct, is the computation order correct.

### Q5: How can you improve DP complexity?

**A:** (1) Reduce the number of states (remove unnecessary dimensions). (2) Speed up transitions (exploit monotonicity or Convex Hull Trick). (3) Space optimization (keep only previous row/column). (4) Matrix exponentiation (for linear recurrences).

---


## FAQ

### Q1: What is the most important point to keep in mind when learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory, but by actually writing code and verifying how it works.

### Q2: What are common mistakes that beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this used in practice?

Knowledge of this topic is frequently applied in day-to-day development work. It becomes especially important during code reviews and architecture design.

---

## 18. Summary

| Topic | Key Points |
|:---|:---|
| Two DP conditions | Optimal substructure + overlapping subproblems |
| Memoization | Top-down, recursion + cache, computes only what is needed |
| Bottom-up | Tabulation, loops, space optimization possible |
| Knapsack | 0/1 uses reverse update, unbounded uses forward update |
| LCS | Representative 2D DP problem. Applied in diff/spell-checking |
| Interval DP | Manages intervals with dp[l][r]. Matrix chain multiplication is the classic example |
| Bitmask DP | Represents sets with bitmasks. TSP is the classic example |
| Tree DP | Computes parent values from children's results |
| Design steps | State definition -> recurrence -> base cases -> computation order -> answer extraction |

---

## 19. Practical Application Patterns

### 19.1 Text Editor Autocomplete (Edit Distance-Based)

```python
def autocomplete_with_edit_distance(prefix: str, dictionary: list, max_suggestions: int = 5) -> list:
    """Return autocomplete candidates based on edit distance"""
    candidates = []

    for word in dictionary:
        # Compute edit distance with the prefix (compare only the beginning of the word)
        min_len = min(len(prefix), len(word))
        partial_dist = edit_distance(prefix, word[:min_len])

        # Exact prefix match gets highest priority
        if word.startswith(prefix):
            candidates.append((word, 0))
        else:
            candidates.append((word, partial_dist))

    candidates.sort(key=lambda x: (x[1], len(x[0])))
    return [word for word, _ in candidates[:max_suggestions]]
```

### 19.2 DNA Sequence Alignment

```python
def sequence_alignment(seq1: str, seq2: str,
                       match_score: int = 2,
                       mismatch_penalty: int = -1,
                       gap_penalty: int = -2) -> tuple:
    """Needleman-Wunsch algorithm (global alignment)
    Used for comparing DNA/protein sequences
    """
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i * gap_penalty
    for j in range(n + 1):
        dp[0][j] = j * gap_penalty

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = dp[i-1][j-1] + (match_score if seq1[i-1] == seq2[j-1] else mismatch_penalty)
            delete = dp[i-1][j] + gap_penalty
            insert = dp[i][j-1] + gap_penalty
            dp[i][j] = max(match, delete, insert)

    # Alignment reconstruction
    align1, align2 = [], []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            score = match_score if seq1[i-1] == seq2[j-1] else mismatch_penalty
            if dp[i][j] == dp[i-1][j-1] + score:
                align1.append(seq1[i-1])
                align2.append(seq2[j-1])
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i-1][j] + gap_penalty:
            align1.append(seq1[i-1])
            align2.append('-')
            i -= 1
        else:
            align1.append('-')
            align2.append(seq2[j-1])
            j -= 1

    return dp[m][n], ''.join(reversed(align1)), ''.join(reversed(align2))

score, a1, a2 = sequence_alignment("AGTACG", "ACATAG")
print(f"Score: {score}")
print(f"Seq 1: {a1}")
print(f"Seq 2: {a2}")
```

### 19.3 Regular Expression Matching

```python
def regex_match(text: str, pattern: str) -> bool:
    """Regular expression matching ('.' matches any single character,
    '*' matches zero or more of the preceding character)
    Equivalent to LeetCode #10
    """
    m, n = len(text), len(pattern)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True

    # Initialize for patterns starting with "a*b*c*"
    for j in range(2, n + 1):
        if pattern[j - 1] == '*':
            dp[0][j] = dp[0][j - 2]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pattern[j - 1] == '*':
                # '*' matches 0 times
                dp[i][j] = dp[i][j - 2]
                # '*' matches 1 or more times
                if pattern[j - 2] == '.' or pattern[j - 2] == text[i - 1]:
                    dp[i][j] = dp[i][j] or dp[i - 1][j]
            elif pattern[j - 1] == '.' or pattern[j - 1] == text[i - 1]:
                dp[i][j] = dp[i - 1][j - 1]

    return dp[m][n]

print(regex_match("aab", "c*a*b"))     # True
print(regex_match("mississippi", "mis*is*p*."))  # False
print(regex_match("ab", ".*"))          # True
```

### 19.4 Maximum Profit from Stock Trading

```python
def max_profit_k_transactions(prices: list, k: int) -> int:
    """Maximum profit with at most k transactions - O(nk)
    dp[j][0] = max profit at transaction j without holding stock
    dp[j][1] = max profit at transaction j while holding stock
    """
    n = len(prices)
    if n <= 1 or k <= 0:
        return 0

    # If k is large enough, allow unlimited trades
    if k >= n // 2:
        return sum(max(prices[i+1] - prices[i], 0) for i in range(n - 1))

    dp = [[[0, 0] for _ in range(k + 1)] for _ in range(n)]

    for j in range(k + 1):
        dp[0][j][1] = -prices[0]

    for i in range(1, n):
        for j in range(1, k + 1):
            dp[i][j][0] = max(dp[i-1][j][0], dp[i-1][j][1] + prices[i])
            dp[i][j][1] = max(dp[i-1][j][1], dp[i-1][j-1][0] - prices[i])

    return max(dp[n-1][j][0] for j in range(k + 1))

prices = [3, 2, 6, 5, 0, 3]
print(max_profit_k_transactions(prices, 2))  # 7 (buy at 2, sell at 6 + buy at 0, sell at 3)
```

### 19.5 Longest Palindromic Substring

```python
def longest_palindrome_substring(s: str) -> str:
    """Longest palindromic substring - O(n^2)
    dp[i][j] = whether s[i..j] is a palindrome
    """
    n = len(s)
    if n < 2:
        return s

    dp = [[False] * n for _ in range(n)]
    start = 0
    max_len = 1

    # Length 1: all are palindromes
    for i in range(n):
        dp[i][i] = True

    # Length 2
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            dp[i][i + 1] = True
            start = i
            max_len = 2

    # Length 3 and above
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j] and dp[i + 1][j - 1]:
                dp[i][j] = True
                if length > max_len:
                    start = i
                    max_len = length

    return s[start:start + max_len]

print(longest_palindrome_substring("babad"))    # "bab" or "aba"
print(longest_palindrome_substring("cbbd"))     # "bb"
print(longest_palindrome_substring("racecar"))  # "racecar"
```

---

## 20. DP Acceleration via Matrix Exponentiation

DP with linear recurrences can be accelerated to O(k^3 log n) using matrix exponentiation.

```python
import numpy as np

def matrix_power(M, n, mod=None):
    """Matrix exponentiation via repeated squaring - O(k^3 log n)"""
    result = [[0] * len(M) for _ in range(len(M))]
    for i in range(len(M)):
        result[i][i] = 1  # Identity matrix

    base = [row[:] for row in M]

    while n > 0:
        if n % 2 == 1:
            result = matrix_multiply(result, base, mod)
        base = matrix_multiply(base, base, mod)
        n //= 2

    return result

def matrix_multiply(A, B, mod=None):
    """Matrix multiplication"""
    n = len(A)
    m = len(B[0])
    k = len(B)
    C = [[0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            for l in range(k):
                C[i][j] += A[i][l] * B[l][j]
                if mod:
                    C[i][j] %= mod
    return C

def fib_matrix(n: int, mod: int = 10**9 + 7) -> int:
    """Compute Fibonacci numbers via matrix exponentiation - O(log n)
    [F(n+1)]   [1, 1]^n   [1]
    [F(n)  ] = [1, 0]   * [0]
    """
    if n <= 1:
        return n
    M = [[1, 1], [1, 0]]
    result = matrix_power(M, n, mod)
    return result[0][1] % mod

print(fib_matrix(10))     # 55
print(fib_matrix(100))    # 782204094 (mod 10^9+7)
print(fib_matrix(10**18)) # Computable in O(log n)
```

---

## Recommended Next Guides

- [Greedy Algorithms](./05-greedy.md) -- Efficient approach when DP is not needed
- [Divide and Conquer](./06-divide-conquer.md) -- Design for non-overlapping subproblems
- [Problem Solving](../04-practice/00-problem-solving.md) -- Pattern recognition for identifying DP problems

---

## References

1. Cormen, T. H. et al. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. -- Chapter 14
2. Bellman, R. (1957). *Dynamic Programming*. Princeton University Press.
3. Skiena, S. S. (2020). *The Algorithm Design Manual* (3rd ed.). Springer. -- Chapter 10
4. Halim, S. & Halim, F. (2013). *Competitive Programming 3*. -- Chapter 3: Dynamic Programming
5. Knuth, D. E. (1997). *The Art of Computer Programming, Volume 1*. Addison-Wesley.
