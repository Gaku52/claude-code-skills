# Dynamic Programming (DP)

> The essence of dynamic programming is "never compute the same thing twice." By memoizing results of overlapping subproblems, it reduces exponential time to polynomial time.

## Learning Objectives

- [ ] Understand the difference between memoization (top-down) and bottom-up DP
- [ ] Identify the characteristics of problems where DP is applicable
- [ ] Implement classic DP patterns (knapsack, LCS, etc.)
- [ ] Systematically master state design techniques
- [ ] Use space optimization and dimension reduction techniques proficiently
- [ ] Understand advanced DP: interval DP, tree DP, digit DP, bitmask DP, and more

## Prerequisites


---

## 1. Fundamental DP Concepts

### 1.1 Understanding DP Through Fibonacci Numbers

```python
# BAD: Naive recursion: O(2^n) -- exponential time
def fib_naive(n):
    if n <= 1:
        return n
    return fib_naive(n - 1) + fib_naive(n - 2)

# fib(5) call tree:
#              fib(5)
#            /        \
#       fib(4)        fib(3)
#       /    \        /    \
#   fib(3)  fib(2)  fib(2) fib(1)
#   /   \    / \     / \
# f(2) f(1) f(1) f(0) f(1) f(0)
# -> fib(2) is computed 3 times, fib(3) is computed 2 times (wasteful!)

# GOOD: Memoized recursion (top-down DP): O(n)
def fib_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib_memo(n - 1, memo) + fib_memo(n - 2, memo)
    return memo[n]

# GOOD: Bottom-up DP (tabulation): O(n)
def fib_dp(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]

# GOOD: Space-optimized: O(1)
def fib_optimized(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
```

### 1.2 Two Conditions for DP Applicability

```
Characteristics of problems solvable with DP:

  1. Optimal Substructure
     -> The optimal solution to the problem is composed of optimal solutions to subproblems
     -> Example: A sub-path of a shortest path is also a shortest path

  2. Overlapping Subproblems
     -> The same subproblems appear repeatedly
     -> Example: fib(n) = fib(n-1) + fib(n-2) -> fib(n-2) is needed in multiple places

  Examples where DP does NOT apply:
  - Longest simple path -> No optimal substructure (sub-paths interfere)
  - Divide-and-conquer with no subproblem overlap -> Memoization is meaningless
```

### 1.3 Memoization vs Bottom-Up

```
Top-down (memoized recursion):
  Pros: Natural recursive thinking, only computes needed subproblems
  Cons: Stack overflow risk with deep recursion
  Implementation: @functools.lru_cache

Bottom-up (tabulation):
  Pros: No stack overflow, easier to optimize space
  Cons: Must determine computation order in advance
  Implementation: For loop + array
```

### 1.4 Memoization with Python's lru_cache

```python
from functools import lru_cache

# Using lru_cache eliminates the need for manual memoization
@lru_cache(maxsize=None)
def fib_cached(n):
    if n <= 1:
        return n
    return fib_cached(n - 1) + fib_cached(n - 2)

# Python 3.9+ also offers the cache decorator
from functools import cache

@cache
def fib_py39(n):
    if n <= 1:
        return n
    return fib_py39(n - 1) + fib_py39(n - 2)

# lru_cache caveats:
# 1. Arguments must be hashable (no lists, but tuples are fine)
# 2. maxsize=None means unlimited cache (be mindful of memory)
# 3. Watch out for recursion depth limits (adjustable via sys.setrecursionlimit)
# 4. Does not pollute global dictionaries (independent per function)

# Usage example: Number of paths on a grid
@lru_cache(maxsize=None)
def grid_paths(m, n):
    """Number of paths from top-left to bottom-right in an m x n grid (right and down moves only)"""
    if m == 1 or n == 1:
        return 1
    return grid_paths(m - 1, n) + grid_paths(m, n - 1)

print(grid_paths(10, 10))  # 48620
```

### 1.5 DP Table Visualization Technique

```python
def visualize_dp_table(dp, row_labels=None, col_labels=None, title="DP Table"):
    """Utility to display DP tables in a readable format"""
    print(f"\n--- {title} ---")

    if isinstance(dp[0], list):
        # 2D table
        rows = len(dp)
        cols = len(dp[0])

        # Header
        if col_labels:
            header = "     " + "".join(f"{l:>5}" for l in col_labels)
            print(header)
            print("     " + "-" * (cols * 5))

        for i in range(rows):
            label = f"{row_labels[i]:>3} |" if row_labels else f"{i:>3} |"
            row = "".join(f"{dp[i][j]:>5}" for j in range(cols))
            print(label + row)
    else:
        # 1D table
        if col_labels:
            header = "".join(f"{l:>5}" for l in col_labels)
            print(header)
        row = "".join(f"{v:>5}" for v in dp)
        print(row)

# Usage example: LCS table visualization
s1, s2 = "ABCB", "BDCAB"
m, n = len(s1), len(s2)
dp = [[0] * (n + 1) for _ in range(m + 1)]
for i in range(1, m + 1):
    for j in range(1, n + 1):
        if s1[i-1] == s2[j-1]:
            dp[i][j] = dp[i-1][j-1] + 1
        else:
            dp[i][j] = max(dp[i-1][j], dp[i][j-1])

visualize_dp_table(
    dp,
    row_labels=['""'] + list(s1),
    col_labels=['""'] + list(s2),
    title="LCS Table"
)
```

---

## 2. Classic DP Patterns

### 2.1 0-1 Knapsack Problem

```python
def knapsack(weights, values, capacity):
    """Maximize value within a weight constraint"""
    n = len(weights)
    # dp[i][w] = maximum value using the first i items with weight limit w
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            dp[i][w] = dp[i-1][w]  # Don't include item i
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w],
                    dp[i-1][w - weights[i-1]] + values[i-1])  # Include it

    return dp[n][capacity]

# Example: weights=[2,3,4,5], values=[3,4,5,6], capacity=8
# -> Maximum value = 10 (items 2+4: weight 8, value 10)

# Time complexity: O(n x W)  (W = capacity)
# Space: O(n x W) -> can be optimized to O(W)

# Space-optimized 1D DP:
def knapsack_optimized(weights, values, capacity):
    dp = [0] * (capacity + 1)
    for i in range(len(weights)):
        for w in range(capacity, weights[i] - 1, -1):  # Reverse order is crucial!
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    return dp[capacity]
```

#### Knapsack Item Recovery

```python
def knapsack_with_items(weights, values, capacity):
    """Recover not only max value but also which items were selected"""
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            dp[i][w] = dp[i-1][w]
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w],
                    dp[i-1][w - weights[i-1]] + values[i-1])

    # Recover selected items (trace backwards)
    selected = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            selected.append(i - 1)  # 0-indexed
            w -= weights[i-1]

    selected.reverse()
    return dp[n][capacity], selected

# Usage example
weights = [2, 3, 4, 5]
values  = [3, 4, 5, 6]
capacity = 8
max_val, items = knapsack_with_items(weights, values, capacity)
print(f"Maximum value: {max_val}")        # 10
print(f"Selected items: {items}")         # [1, 3] (0-indexed)
print(f"Total weight: {sum(weights[i] for i in items)}")  # 8
print(f"Total value: {sum(values[i] for i in items)}")    # 10
```

#### Unbounded Knapsack (Complete Knapsack)

```python
def unbounded_knapsack(weights, values, capacity):
    """Knapsack where each item can be used any number of times"""
    dp = [0] * (capacity + 1)

    for w in range(1, capacity + 1):
        for i in range(len(weights)):
            if weights[i] <= w:
                dp[w] = max(dp[w], dp[w - weights[i]] + values[i])

    return dp[capacity]

# Difference from 0-1 knapsack:
# - 0-1: Each item used at most once -> inner loop in reverse order
# - Unbounded: Each item usable any number of times -> inner loop in forward order
# Why the difference:
#   Reverse order: dp[w - weights[i]] references "state before using item i"
#   Forward order: dp[w - weights[i]] may include "state after using item i"

# Bounded knapsack (limited quantities)
def bounded_knapsack(weights, values, counts, capacity):
    """Knapsack with a usage limit per item"""
    dp = [0] * (capacity + 1)

    for i in range(len(weights)):
        # Binary decomposition technique: decompose count into groups of 1, 2, 4, ...
        count = counts[i]
        k = 1
        while count > 0:
            actual = min(k, count)
            w_group = weights[i] * actual
            v_group = values[i] * actual
            # Process as 0-1 knapsack
            for w in range(capacity, w_group - 1, -1):
                dp[w] = max(dp[w], dp[w - w_group] + v_group)
            count -= actual
            k *= 2

    return dp[capacity]

# Example: Item A (w=2, v=3) x5, Item B (w=3, v=5) x3
# -> Binary decomposition: A: groups of 1+2+2, B: groups of 1+2
# -> Time complexity: O(W x sum(log count_i)) <- faster than O(W x sum(count_i))
```

### 2.2 Longest Common Subsequence (LCS)

```python
def lcs(s1, s2):
    """Length of the longest common subsequence of two strings"""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]

# Example: lcs("ABCBDAB", "BDCAB") = 4 ("BCAB")
# Time complexity: O(m x n)
# Applications: diff command, DNA sequence comparison, version control

# LCS table visualization:
#     ""  B  D  C  A  B
# ""   0  0  0  0  0  0
#  A   0  0  0  0  1  1
#  B   0  1  1  1  1  2
#  C   0  1  1  2  2  2
#  B   0  1  1  2  2  3
#  D   0  1  2  2  2  3
#  A   0  1  2  2  3  3
#  B   0  1  2  2  3  4
```

#### LCS String Recovery

```python
def lcs_string(s1, s2):
    """Recover the actual LCS string"""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    # Recovery (trace backwards)
    result = []
    i, j = m, n
    while i > 0 and j > 0:
        if s1[i-1] == s2[j-1]:
            result.append(s1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1

    return "".join(reversed(result))

# Usage example
print(lcs_string("ABCBDAB", "BDCAB"))  # "BCAB"

# diff-command-style output
def print_diff(s1, s2):
    """Display diff of two strings (lists of lines)"""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    # Trace backwards to collect diff operations
    ops = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s1[i-1] == s2[j-1]:
            ops.append((' ', s1[i-1]))
            i -= 1
            j -= 1
        elif j > 0 and (i == 0 or dp[i][j-1] >= dp[i-1][j]):
            ops.append(('+', s2[j-1]))
            j -= 1
        else:
            ops.append(('-', s1[i-1]))
            i -= 1

    ops.reverse()
    for op, line in ops:
        print(f"{op} {line}")

# Usage example
lines1 = ["apple", "banana", "cherry", "date"]
lines2 = ["apple", "blueberry", "cherry", "elderberry"]
print_diff(lines1, lines2)
# Output:
#   apple
# - banana
# + blueberry
#   cherry
# - date
# + elderberry
```

#### Space-Optimized LCS

```python
def lcs_space_optimized(s1, s2):
    """Compute LCS length in O(min(m,n)) space"""
    # Use the shorter string for the inner loop
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    m, n = len(s1), len(s2)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev, curr = curr, [0] * (n + 1)

    return prev[n]

# Note: This optimization does not allow string recovery
# For O(n) space recovery, use Hirschberg's algorithm
```

### 2.3 Coin Change Problem (Complete Knapsack)

```python
def coin_change(coins, amount):
    """Make amount using the minimum number of coins"""
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i and dp[i - coin] + 1 < dp[i]:
                dp[i] = dp[i - coin] + 1

    return dp[amount] if dp[amount] != float('inf') else -1

# Example: coins=[1,5,10,25], amount=36
# -> 25 + 10 + 1 = 3 coins (greedy also optimal here, but for coins=[1,3,4], amount=6 greedy fails)
# Greedy: 4+1+1 = 3 coins BAD
# DP:     3+3   = 2 coins GOOD

# Time complexity: O(amount x len(coins))
```

#### Number of Coin Combinations

```python
def coin_combinations(coins, amount):
    """Total number of ways to make amount (order does not matter)"""
    dp = [0] * (amount + 1)
    dp[0] = 1

    # Outer loop over coins -> order-independent (combinations)
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]

    return dp[amount]

def coin_permutations(coins, amount):
    """Total number of ways to make amount (order matters)"""
    dp = [0] * (amount + 1)
    dp[0] = 1

    # Outer loop over amounts -> order-dependent (permutations)
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] += dp[i - coin]

    return dp[amount]

# Example: coins=[1,2,3], amount=4
# Combinations: {1+1+1+1, 1+1+2, 1+3, 2+2} = 4 ways
# Permutations: above + {2+1+1, 3+1, 2+1+1, 1+2+1, ...} = 7 ways

# This difference arises from the loop order in DP
# A very commonly asked point in interviews
```

#### Coin Usage Recovery

```python
def coin_change_with_trace(coins, amount):
    """Return minimum number of coins and their breakdown"""
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    parent = [-1] * (amount + 1)  # Which coin was used

    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i and dp[i - coin] + 1 < dp[i]:
                dp[i] = dp[i - coin] + 1
                parent[i] = coin

    if dp[amount] == float('inf'):
        return -1, []

    # Recovery
    used_coins = []
    current = amount
    while current > 0:
        used_coins.append(parent[current])
        current -= parent[current]

    return dp[amount], used_coins

# Usage example
min_coins, coins_used = coin_change_with_trace([1, 3, 4], 6)
print(f"Minimum coins: {min_coins}")    # 2
print(f"Coins used: {coins_used}")      # [3, 3]
```

### 2.4 Longest Increasing Subsequence (LIS)

```python
import bisect

def lis(arr):
    """Length of the longest increasing subsequence (O(n log n))"""
    tails = []  # tails[i] = minimum tail value of LIS with length i+1

    for x in arr:
        pos = bisect.bisect_left(tails, x)
        if pos == len(tails):
            tails.append(x)
        else:
            tails[pos] = x

    return len(tails)

# Example: arr = [10, 9, 2, 5, 3, 7, 101, 18]
# LIS = [2, 3, 7, 18] or [2, 5, 7, 18] -> length 4

# O(n^2) version: dp[i] = length of LIS ending at arr[i]
# O(n log n) version: manage tails array with binary search
```

#### O(n^2) LIS with Recovery

```python
def lis_quadratic_with_recovery(arr):
    """O(n^2) LIS + recover the actual subsequence"""
    n = len(arr)
    if n == 0:
        return 0, []

    dp = [1] * n
    parent = [-1] * n

    for i in range(1, n):
        for j in range(i):
            if arr[j] < arr[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j

    # Find the position of the maximum value
    max_len = max(dp)
    max_idx = dp.index(max_len)

    # Recovery
    result = []
    idx = max_idx
    while idx != -1:
        result.append(arr[idx])
        idx = parent[idx]

    result.reverse()
    return max_len, result

# Usage example
length, subsequence = lis_quadratic_with_recovery([10, 9, 2, 5, 3, 7, 101, 18])
print(f"Length: {length}")            # 4
print(f"Subsequence: {subsequence}")  # [2, 3, 7, 18] or [2, 5, 7, 101] etc.
```

#### Longest Non-Decreasing Subsequence (Non-Strict)

```python
def longest_non_decreasing_subsequence(arr):
    """When equal elements are allowed (non-strictly increasing)"""
    tails = []
    for x in arr:
        # Use bisect_right (allows equal elements)
        pos = bisect.bisect_right(tails, x)
        if pos == len(tails):
            tails.append(x)
        else:
            tails[pos] = x
    return len(tails)

# Strict vs non-strict difference:
# Strictly increasing: bisect_left -> [2,3,7,18] (equal elements not allowed)
# Non-decreasing:      bisect_right -> [2,3,3,7,18] (equal elements allowed)
```

### 2.5 Edit Distance (Levenshtein Distance)

```python
def edit_distance(s1, s2):
    """Compute the edit distance between two strings"""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Base cases
    for i in range(m + 1):
        dp[i][0] = i  # Delete all first i characters of s1
    for j in range(n + 1):
        dp[0][j] = j  # Insert all first j characters of s2

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]  # Match -> no cost
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],      # Delete
                    dp[i][j-1],      # Insert
                    dp[i-1][j-1]     # Replace
                )

    return dp[m][n]

# Example:
# edit_distance("kitten", "sitting") = 3
#   kitten -> sitten (replace k->s)
#   sitten -> sittin (replace e->i)
#   sittin -> sitting (insert g)

# Applications:
# - Spell checkers
# - DNA sequence alignment
# - Fuzzy matching (approximate search)
# - Natural language processing (word similarity)

# Time complexity: O(m x n)
# Space: O(m x n) -> can be optimized to O(min(m,n))
```

#### Edit Distance Space Optimization and Operation Recovery

```python
def edit_distance_optimized(s1, s2):
    """Compute edit distance in O(min(m,n)) space"""
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    m, n = len(s1), len(s2)
    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                curr[j] = prev[j-1]
            else:
                curr[j] = 1 + min(prev[j], curr[j-1], prev[j-1])
        prev, curr = curr, [0] * (n + 1)

    return prev[n]

def edit_operations(s1, s2):
    """Return edit distance and the specific sequence of operations"""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    # Operation recovery
    ops = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s1[i-1] == s2[j-1]:
            ops.append(('MATCH', s1[i-1], i-1))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            ops.append(('REPLACE', f"{s1[i-1]}->{s2[j-1]}", i-1))
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            ops.append(('INSERT', s2[j-1], i))
            j -= 1
        else:
            ops.append(('DELETE', s1[i-1], i-1))
            i -= 1

    ops.reverse()
    return dp[m][n], ops

# Usage example
dist, ops = edit_operations("kitten", "sitting")
print(f"Edit distance: {dist}")
for op, char, pos in ops:
    if op != 'MATCH':
        print(f"  {op}: '{char}' at position {pos}")
```

### 2.6 Subset Sum Problem

```python
def subset_sum(nums, target):
    """Can a subset of nums sum to target?"""
    dp = [False] * (target + 1)
    dp[0] = True

    for num in nums:
        # Process in reverse order (each element used at most once)
        for j in range(target, num - 1, -1):
            dp[j] = dp[j] or dp[j - num]

    return dp[target]

# Application: Split array into two groups minimizing the difference
def min_subset_difference(nums):
    """Minimize the difference between two groups' sums"""
    total = sum(nums)
    half = total // 2

    dp = [False] * (half + 1)
    dp[0] = True

    for num in nums:
        for j in range(half, num - 1, -1):
            dp[j] = dp[j] or dp[j - num]

    # Find the largest j where dp[j] is True
    for j in range(half, -1, -1):
        if dp[j]:
            return total - 2 * j

# Example: nums = [1, 6, 11, 5]
# Group 1: {1, 5, 6} = 12, Group 2: {11} = 11
# Difference = 1

# Application: Can the array be partitioned into k groups with equal sums?
def can_partition_k(nums, k):
    """Can the array be split into k groups with equal sums?"""
    total = sum(nums)
    if total % k != 0:
        return False
    target = total // k

    nums.sort(reverse=True)
    if nums[0] > target:
        return False

    buckets = [0] * k

    def backtrack(idx):
        if idx == len(nums):
            return all(b == target for b in buckets)

        seen = set()
        for i in range(k):
            if buckets[i] + nums[idx] <= target and buckets[i] not in seen:
                seen.add(buckets[i])
                buckets[i] += nums[idx]
                if backtrack(idx + 1):
                    return True
                buckets[i] -= nums[idx]

        return False

    return backtrack(0)
```

---

## 3. DP State Design

### 3.1 How to Define States

```
DP design thought process:

  1. Define the state
     -> dp[i] = "optimal value considering up to the i-th element"
     -> dp[i][j] = "optimal value considering up to the i-th element with remaining capacity j"

  2. Formulate the transition
     -> dp[i] = f(dp[i-1], dp[i-2], ...)
     -> Express the current state as a combination of previous states

  3. Determine base cases
     -> dp[0] = ? dp[1] = ?

  4. Determine computation order
     -> Small to large, following dependencies

  5. Identify the answer
     -> dp[n] or max(dp) or dp[n][W]

  Common state patterns:
  ┌──────────────────┬──────────────────────────────────┐
  │ Pattern           │ State example                     │
  ├──────────────────┼──────────────────────────────────┤
  │ Linear DP        │ dp[i] = optimal value up to i     │
  │ Interval DP      │ dp[i][j] = optimal for interval [i,j]│
  │ Knapsack DP      │ dp[i][w] = optimal for i items, capacity w│
  │ Tree DP          │ dp[v] = optimal for subtree v     │
  │ Digit DP         │ dp[pos][tight] = up to position pos│
  │ Bitmask DP       │ dp[mask] = state for set mask     │
  └──────────────────┴──────────────────────────────────┘
```

### 3.2 State Design in Practice

```python
# Example 1: Maximum subarray sum (Kadane's algorithm)
# State: dp[i] = maximum sum of contiguous subarray ending at arr[i]
def max_subarray(arr):
    dp = arr[0]
    best = arr[0]
    for i in range(1, len(arr)):
        dp = max(arr[i], dp + arr[i])  # Start fresh or continue
        best = max(best, dp)
    return best

# arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
# dp:   -2  1  -2  4   3  5  6   1  5
# best: -2  1   1  4   4  5  6   6  6
# Answer: 6 (subarray [4, -1, 2, 1])

# Example 2: House painting problem (minimize color cost)
# State: dp[i][c] = minimum cost when painting house i with color c
def paint_houses(costs):
    """Paint n houses with 3 colors. Adjacent houses must differ in color"""
    if not costs:
        return 0

    n = len(costs)
    # dp[i][c] can be computed using only the previous house -> O(1) space
    prev = costs[0][:]

    for i in range(1, n):
        curr = [0, 0, 0]
        curr[0] = costs[i][0] + min(prev[1], prev[2])
        curr[1] = costs[i][1] + min(prev[0], prev[2])
        curr[2] = costs[i][2] + min(prev[0], prev[1])
        prev = curr

    return min(prev)

# Example: costs = [[17,2,17],[16,16,5],[14,3,19]]
# House 0 green(2) + House 1 blue(5) + House 2 green(3) = 10

# Example 3: Wine problem (a type of interval DP)
# Sell n wines one per year. Can only choose from the left or right end.
# Selling price = wine's price x year number
def max_wine_profit(prices):
    """Sell wines one per year from left or right end, maximizing profit"""
    n = len(prices)

    @lru_cache(maxsize=None)
    def dp(left, right):
        year = n - (right - left)  # Which year (1-indexed)
        if left > right:
            return 0
        if left == right:
            return prices[left] * year

        sell_left = prices[left] * year + dp(left + 1, right)
        sell_right = prices[right] * year + dp(left, right - 1)
        return max(sell_left, sell_right)

    return dp(0, n - 1)
```

---

## 4. Advanced DP Patterns

### 4.1 Interval DP

```python
# Interval DP: dp[i][j] = optimal solution for interval [i, j]
# Classic problems: matrix chain multiplication, optimal BST, palindrome partition

def matrix_chain_multiplication(dims):
    """Minimum number of multiplications for matrix chain product
    dims[i-1] x dims[i] is the size of the i-th matrix
    """
    n = len(dims) - 1  # Number of matrices
    dp = [[0] * n for _ in range(n)]

    # Process interval lengths from 2 upward
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            for k in range(i, j):
                cost = dp[i][k] + dp[k+1][j] + dims[i] * dims[k+1] * dims[j+1]
                dp[i][j] = min(dp[i][j], cost)

    return dp[0][n-1]

# Example: 4 matrices A(10x30), B(30x5), C(5x60), D(60x10)
# dims = [10, 30, 5, 60, 10]
# Optimal parenthesization: (A x B) x (C x D)
# Multiplications: 10*30*5 + 5*60*10 + 10*5*10 = 1500 + 3000 + 500 = 5000
# Worst parenthesization: A x (B x (C x D)) = 30*5*60 + 30*60*10 + 10*30*10 = 9000+18000+3000 = 30000

# Parenthesization recovery
def matrix_chain_with_paren(dims):
    n = len(dims) - 1
    dp = [[0] * n for _ in range(n)]
    split = [[0] * n for _ in range(n)]

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            for k in range(i, j):
                cost = dp[i][k] + dp[k+1][j] + dims[i] * dims[k+1] * dims[j+1]
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    split[i][j] = k

    def build_paren(i, j):
        if i == j:
            return f"A{i+1}"
        k = split[i][j]
        left = build_paren(i, k)
        right = build_paren(k+1, j)
        return f"({left} x {right})"

    return dp[0][n-1], build_paren(0, n-1)
```

#### Palindrome Partition

```python
def min_palindrome_partitions(s):
    """Minimum number of cuts to partition a string into palindromes"""
    n = len(s)

    # is_pal[i][j] = whether s[i:j+1] is a palindrome
    is_pal = [[False] * n for _ in range(n)]
    for i in range(n):
        is_pal[i][i] = True
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if length == 2:
                is_pal[i][j] = (s[i] == s[j])
            else:
                is_pal[i][j] = (s[i] == s[j]) and is_pal[i+1][j-1]

    # dp[i] = minimum cuts to partition s[0:i+1] into palindromes
    dp = list(range(n))  # Worst case: each character is a single palindrome

    for i in range(1, n):
        if is_pal[0][i]:
            dp[i] = 0
            continue
        for j in range(1, i + 1):
            if is_pal[j][i]:
                dp[i] = min(dp[i], dp[j-1] + 1)

    return dp[n-1]

# Example: "aab" -> ["aa", "b"] -> 1 cut
# Example: "abcba" -> 0 cuts (entire string is a palindrome)
# Example: "abcde" -> 4 cuts (each character is a palindrome)
```

### 4.2 Tree DP

```python
# Tree DP: DP on tree structures
# Classic: tree diameter, maximum independent set, centroid decomposition

class TreeNode:
    def __init__(self, val=0):
        self.val = val
        self.children = []

def max_independent_set(root):
    """Maximum weight independent set of a tree (cannot select adjacent nodes simultaneously)"""

    def dp(node):
        if not node:
            return 0, 0

        # include: maximum value when this node is included
        # exclude: maximum value when this node is excluded
        include = node.val
        exclude = 0

        for child in node.children:
            child_inc, child_exc = dp(child)
            include += child_exc      # Must exclude children
            exclude += max(child_inc, child_exc)  # Children can be included or excluded

        return include, exclude

    inc, exc = dp(root)
    return max(inc, exc)

# Tree diameter (longest path)
def tree_diameter(adj, n):
    """Find the diameter of a tree represented as an adjacency list"""
    diameter = [0]

    def dfs(node, parent):
        max1 = max2 = 0  # Longest and second longest from children

        for neighbor in adj[node]:
            if neighbor != parent:
                depth = dfs(neighbor, node)
                if depth > max1:
                    max2 = max1
                    max1 = depth
                elif depth > max2:
                    max2 = depth

        diameter[0] = max(diameter[0], max1 + max2)
        return max1 + 1

    dfs(0, -1)
    return diameter[0]

# Usage example (adjacency list)
# Tree:    0
#         / \
#        1   2
#       / \   \
#      3   4   5
adj = [
    [1, 2],    # 0's neighbors
    [0, 3, 4], # 1's neighbors
    [0, 5],    # 2's neighbors
    [1],       # 3's neighbors
    [1],       # 4's neighbors
    [2],       # 5's neighbors
]
print(tree_diameter(adj, 6))  # 4 (3->1->0->2->5)
```

### 4.3 Digit DP

```python
def count_numbers_with_digit(n, d):
    """Total count of digit d appearing in integers from 1 to n"""
    digits = list(map(int, str(n)))
    num_digits = len(digits)

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def dp(pos, count, tight, started):
        """
        pos:     Current digit position
        count:   Count of digit d appearances so far
        tight:   Whether upper bound constraint is active
        started: Whether the number has started (to exclude leading zeros)
        """
        if pos == num_digits:
            return count if started else 0

        limit = digits[pos] if tight else 9
        result = 0

        for digit in range(0, limit + 1):
            new_started = started or (digit > 0)
            new_count = count + (1 if digit == d and new_started else 0)
            new_tight = tight and (digit == limit)
            result += dp(pos + 1, new_count, new_tight, new_started)

        return result

    return dp(0, 0, True, False)

# Example: How many times does "1" appear in integers from 1 to 100?
# 1, 10, 11(2 times), 12, ..., 19, 21, 31, ..., 91, 100 = 21 times

# Digit DP applications:
# - Count of "lucky numbers" (containing only 4 and 7) up to N
# - Count of numbers where adjacent digit differences are at most k
# - Count of numbers whose digit sum is a multiple of S
```

#### Another Classic Digit DP Example

```python
def count_step_numbers(low, high):
    """Count of step numbers (adjacent digit difference is 1) between low and high"""

    def count_up_to(n):
        if n < 0:
            return 0
        digits = list(map(int, str(n)))
        num_digits = len(digits)

        @lru_cache(maxsize=None)
        def dp(pos, prev_digit, tight, started):
            if pos == num_digits:
                return 1 if started else 0

            limit = digits[pos] if tight else 9
            result = 0

            for digit in range(0, limit + 1):
                if not started and digit == 0:
                    result += dp(pos + 1, -1, False, False)
                elif not started or abs(digit - prev_digit) == 1:
                    new_tight = tight and (digit == limit)
                    result += dp(pos + 1, digit, new_tight, True)

            return result

        return dp(0, -1, True, False)

    return count_up_to(high) - count_up_to(low - 1)

# Example: Step numbers between 10 and 100
# 10, 12, 21, 23, 32, 34, 43, 45, 54, 56, 65, 67, 76, 78, 87, 89, 98 = 17
```

### 4.4 Bitmask DP

```python
def tsp(dist):
    """Solve the Traveling Salesman Problem (TSP) with bitmask DP"""
    n = len(dist)
    # dp[mask][i] = minimum cost visiting the cities in set mask, currently at city i
    INF = float('inf')
    dp = [[INF] * n for _ in range(1 << n)]
    dp[1][0] = 0  # Start from city 0

    for mask in range(1 << n):
        for u in range(n):
            if dp[mask][u] == INF:
                continue
            if not (mask & (1 << u)):
                continue
            for v in range(n):
                if mask & (1 << v):
                    continue  # Already visited
                new_mask = mask | (1 << v)
                new_cost = dp[mask][u] + dist[u][v]
                dp[new_mask][v] = min(dp[new_mask][v], new_cost)

    # Visit all cities and return to city 0
    full_mask = (1 << n) - 1
    result = min(dp[full_mask][i] + dist[i][0] for i in range(n))

    return result

# Time complexity: O(n^2 x 2^n)
# Space: O(n x 2^n)
# Practical for n up to 20 (2^20 x 20^2 ~ 400 million)

# Path recovery
def tsp_with_path(dist):
    n = len(dist)
    INF = float('inf')
    dp = [[INF] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]
    dp[1][0] = 0

    for mask in range(1 << n):
        for u in range(n):
            if dp[mask][u] == INF or not (mask & (1 << u)):
                continue
            for v in range(n):
                if mask & (1 << v):
                    continue
                new_mask = mask | (1 << v)
                new_cost = dp[mask][u] + dist[u][v]
                if new_cost < dp[new_mask][v]:
                    dp[new_mask][v] = new_cost
                    parent[new_mask][v] = u

    full_mask = (1 << n) - 1
    last = min(range(n), key=lambda i: dp[full_mask][i] + dist[i][0])
    min_cost = dp[full_mask][last] + dist[last][0]

    # Path recovery
    path = []
    mask = full_mask
    node = last
    while node != -1:
        path.append(node)
        prev = parent[mask][node]
        mask ^= (1 << node)
        node = prev
    path.reverse()
    path.append(0)  # Return to starting point

    return min_cost, path

# Usage example
dist = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]
cost, path = tsp_with_path(dist)
print(f"Minimum cost: {cost}")    # 80
print(f"Path: {path}")            # [0, 1, 3, 2, 0]
```

#### Bitmask DP Application: Job Assignment Problem

```python
def min_cost_assignment(cost):
    """Assign n jobs to n people one-to-one, minimizing total cost"""
    n = len(cost)
    INF = float('inf')
    dp = [INF] * (1 << n)
    dp[0] = 0

    for mask in range(1 << n):
        person = bin(mask).count('1')  # How many people assigned so far
        if person >= n:
            continue
        for job in range(n):
            if mask & (1 << job):
                continue
            new_mask = mask | (1 << job)
            dp[new_mask] = min(dp[new_mask], dp[mask] + cost[person][job])

    return dp[(1 << n) - 1]

# Example:
cost = [
    [9, 2, 7, 8],
    [6, 4, 3, 7],
    [5, 8, 1, 8],
    [7, 6, 9, 4]
]
print(min_cost_assignment(cost))  # 13
```

### 4.5 Probability DP

```python
def expected_dice_rolls(target):
    """Expected number of dice rolls until the total reaches at least target"""
    dp = [0.0] * (target + 7)

    # dp[i] = expected rolls to reach target or more when current total is i
    for i in range(target - 1, -1, -1):
        dp[i] = 1  # Roll the die once
        for face in range(1, 7):
            dp[i] += dp[i + face] / 6

    return dp[0]

# Example: target=10 -> approximately 3.77 rolls

# Random walk arrival probability
def random_walk_probability(n, target, steps):
    """Probability of being at position target after a 1D random walk of given steps"""
    # dp[step][pos] = probability of being at position pos after step steps
    # Position ranges from -steps to +steps
    offset = steps
    dp = [[0.0] * (2 * steps + 1) for _ in range(steps + 1)]
    dp[0][offset] = 1.0  # Initial position 0

    for step in range(steps):
        for pos in range(2 * steps + 1):
            if dp[step][pos] == 0:
                continue
            # Move left
            if pos > 0:
                dp[step + 1][pos - 1] += dp[step][pos] * 0.5
            # Move right
            if pos < 2 * steps:
                dp[step + 1][pos + 1] += dp[step][pos] * 0.5

    target_idx = target + offset
    if 0 <= target_idx < 2 * steps + 1:
        return dp[steps][target_idx]
    return 0.0
```

### 4.6 String DP

```python
# Regular expression matching ('.' and '*' only)
def regex_match(text, pattern):
    """Regex matching: '.' matches any single char, '*' matches zero or more of preceding char"""
    m, n = len(text), len(pattern)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True

    # Pattern starting with "X*Y*..." can match empty string
    for j in range(2, n + 1):
        if pattern[j-1] == '*':
            dp[0][j] = dp[0][j-2]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pattern[j-1] == '*':
                # Use the character before '*' zero times
                dp[i][j] = dp[i][j-2]
                # Use the character before '*' one or more times
                if pattern[j-2] == '.' or pattern[j-2] == text[i-1]:
                    dp[i][j] = dp[i][j] or dp[i-1][j]
            elif pattern[j-1] == '.' or pattern[j-1] == text[i-1]:
                dp[i][j] = dp[i-1][j-1]

    return dp[m][n]

# Tests
assert regex_match("aa", "a") == False
assert regex_match("aa", "a*") == True
assert regex_match("ab", ".*") == True
assert regex_match("aab", "c*a*b") == True

# Wildcard matching ('?' and '*')
def wildcard_match(text, pattern):
    """'?' matches any single char, '*' matches any string (zero or more chars)"""
    m, n = len(text), len(pattern)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True

    for j in range(1, n + 1):
        if pattern[j-1] == '*':
            dp[0][j] = dp[0][j-1]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pattern[j-1] == '*':
                dp[i][j] = dp[i-1][j] or dp[i][j-1]
            elif pattern[j-1] == '?' or pattern[j-1] == text[i-1]:
                dp[i][j] = dp[i-1][j-1]

    return dp[m][n]
```

### 4.7 Game Theory DP

```python
def stone_game(piles):
    """Stone game: Two players alternately take piles from left or right end.
    Returns the score difference when the first player plays optimally"""
    n = len(piles)
    # dp[i][j] = how much more the first player can take than the second in interval [i,j]
    dp = [[0] * n for _ in range(n)]

    for i in range(n):
        dp[i][i] = piles[i]

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = max(
                piles[i] - dp[i+1][j],  # Take left end
                piles[j] - dp[i][j-1]   # Take right end
            )

    return dp[0][n-1]

# Example: piles = [5, 3, 4, 5]
# First player optimally gets 5+4=9, second player gets 3+5=8 -> difference is 1
# stone_game([5,3,4,5]) = 1

# Nim Game
def nim_game(piles):
    """Nim: Take any number from any pile. Last to take wins.
    Returns whether the first player can win"""
    xor_sum = 0
    for p in piles:
        xor_sum ^= p
    return xor_sum != 0  # First player wins iff XOR sum is non-zero

# Sprague-Grundy theorem:
# Any impartial game can be reduced to Nim
# Grundy number = mex(set of Grundy numbers of successor states)
# mex(S) = smallest non-negative integer not in S

def grundy_number(pos, moves, memo={}):
    """Compute the Grundy number from position pos"""
    if pos in memo:
        return memo[pos]

    reachable = set()
    for m in moves:
        if pos >= m:
            reachable.add(grundy_number(pos - m, moves, memo))

    # Compute mex
    g = 0
    while g in reachable:
        g += 1

    memo[pos] = g
    return g
```

---

## 5. DP in Practice

### 5.1 Where DP Excels

```
Practical DP applications:

  1. Text Processing
     - diff algorithm (LCS)
     - Spell checking (edit distance)
     - Natural language processing (CYK parsing)
     - String alignment

  2. Machine Learning
     - Viterbi algorithm (Hidden Markov Models)
     - CTC (Connectionist Temporal Classification)
     - Beam search (approximate DP)
     - Value iteration in reinforcement learning

  3. Optimization
     - Resource allocation
     - Scheduling
     - Inventory management
     - Financial engineering (option pricing)

  4. Games
     - Game tree evaluation (minimax + memoization)
     - Optimal strategy computation
     - Win/loss determination

  5. Bioinformatics
     - DNA/RNA sequence alignment
     - Protein structure prediction
     - Phylogenetic tree construction

  6. Image Processing
     - Seam carving (image resizing)
     - Stereo matching
     - Character recognition (CTC)
```

### 5.2 Practical DP Optimization Techniques

```python
# 1. Space optimization: Use only 2 rows (rolling array)
def lcs_rolling(s1, s2):
    m, n = len(s1), len(s2)
    if m < n:
        s1, s2 = s2, s1
        m, n = n, m

    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev, curr = curr, [0] * (n + 1)

    return prev[n]

# 2. Transition speedup: prefix sums, segment trees
# dp[i] = max(dp[j] + f(j, i)) for j < i
# -> If f(j, i) is additive, use prefix sums for O(1)
# -> For range maximum, use segment tree for O(log n)

# 3. Knuth's optimization
# dp[i][j] = min(dp[i][k] + dp[k+1][j] + C[i][j]) for k in [i, j)
# If the split point is monotone: O(n^3) -> O(n^2)

# 4. Divide and Conquer DP
# dp[i] = min(dp[j] + C[j+1][i]) when C satisfies Concave Monge condition
# O(n^2) -> O(n log n)

# 5. Convex Hull Trick
def convex_hull_trick_example():
    """
    Speed up transitions of the form dp[i] = min(dp[j] + (a[i] - a[j])^2)
    O(n^2) -> O(n) or O(n log n)

    Idea:
    dp[i] = min(dp[j] + a[i]^2 - 2*a[i]*a[j] + a[j]^2)
          = a[i]^2 + min(-2*a[j]*a[i] + (dp[j] + a[j]^2))

    In the form y = mx + b:
    m = -2*a[j], b = dp[j] + a[j]^2, x = a[i]
    -> Minimum over a set of lines -> manage with convex hull
    """
    pass

# 6. SOS DP (Sum over Subsets)
def sos_dp(values, n):
    """Efficiently compute the sum over all subsets
    dp[mask] = sum(values[sub]) for all sub that are subsets of mask
    """
    dp = values[:]
    for bit in range(n):
        for mask in range(1 << n):
            if mask & (1 << bit):
                dp[mask] += dp[mask ^ (1 << bit)]
    return dp
# Time complexity: O(n x 2^n) -- naive approach would be O(3^n)
```

### 5.3 DP Debugging Techniques

```python
# 1. Manually verify with small cases
def debug_knapsack(weights, values, capacity):
    """Display the DP table for debugging"""
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            dp[i][w] = dp[i-1][w]
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w], dp[i-1][w - weights[i-1]] + values[i-1])

    # Display table
    print("DP Table:")
    print("     ", end="")
    for w in range(capacity + 1):
        print(f"w={w:2d} ", end="")
    print()

    for i in range(n + 1):
        if i == 0:
            print("none ", end="")
        else:
            print(f"i={i:2d} ", end="")
        for w in range(capacity + 1):
            print(f"{dp[i][w]:4d} ", end="")
        print()

    return dp[n][capacity]

# 2. Comparison testing against brute force
import random

def test_dp_correctness():
    """Verify DP correctness with random tests"""
    def brute_force_knapsack(weights, values, capacity):
        n = len(weights)
        best = 0
        for mask in range(1 << n):
            total_w = sum(weights[i] for i in range(n) if mask & (1 << i))
            total_v = sum(values[i] for i in range(n) if mask & (1 << i))
            if total_w <= capacity:
                best = max(best, total_v)
        return best

    for _ in range(1000):
        n = random.randint(1, 15)
        weights = [random.randint(1, 20) for _ in range(n)]
        values = [random.randint(1, 100) for _ in range(n)]
        capacity = random.randint(1, 50)

        dp_result = knapsack_optimized(weights, values, capacity)
        bf_result = brute_force_knapsack(weights, values, capacity)

        assert dp_result == bf_result, \
            f"Mismatch: w={weights}, v={values}, cap={capacity}"

    print("All tests passed!")

# 3. Transition formula verification
# Checklist for verifying DP transitions:
# [ ] Are base cases correct?
# [ ] Is the transition direction correct? (Are dependent values already computed?)
# [ ] Are boundary conditions correct? (No out-of-bounds array access?)
# [ ] Where is the answer? (dp[n]? max(dp)? dp[0]?)
# [ ] Is the reverse loop correct? (0-1 knapsack space optimization)
```

---

## 6. Common DP Problem Patterns

### 6.1 Staircase Climbing Problem

```python
def climb_stairs(n, steps=[1, 2]):
    """Total number of ways to climb n stairs with step sizes in steps"""
    dp = [0] * (n + 1)
    dp[0] = 1

    for i in range(1, n + 1):
        for step in steps:
            if i >= step:
                dp[i] += dp[i - step]

    return dp[n]

# Example: climb_stairs(5) = 8
# climb_stairs(5, [1, 2, 3]) = 13

# Variation: Staircase problem with costs
def min_cost_climbing(cost):
    """Each step has a cost; reach the top with minimum total cost"""
    n = len(cost)
    dp = [0] * (n + 1)

    for i in range(2, n + 1):
        dp[i] = min(dp[i-1] + cost[i-1], dp[i-2] + cost[i-2])

    return dp[n]
```

### 6.2 Path Counting Problems

```python
def unique_paths(m, n):
    """Number of paths from top-left to bottom-right in an m x n grid (right and down only)"""
    dp = [[1] * n for _ in range(m)]

    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]

    return dp[m-1][n-1]

# With obstacles
def unique_paths_with_obstacles(grid):
    """Number of paths with obstacles (1)"""
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]

    # If first cell is an obstacle, return 0
    dp[0][0] = 1 if grid[0][0] == 0 else 0

    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                dp[i][j] = 0
                continue
            if i > 0:
                dp[i][j] += dp[i-1][j]
            if j > 0:
                dp[i][j] += dp[i][j-1]

    return dp[m-1][n-1]

# Minimum cost path
def min_path_sum(grid):
    """Minimum cost path from top-left to bottom-right"""
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = grid[0][0]

    for j in range(1, n):
        dp[0][j] = dp[0][j-1] + grid[0][j]
    for i in range(1, m):
        dp[i][0] = dp[i-1][0] + grid[i][0]

    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])

    return dp[m-1][n-1]
```

### 6.3 String Interleaving

```python
def is_interleave(s1, s2, s3):
    """Is s3 an interleaving of s1 and s2?"""
    m, n = len(s1), len(s2)
    if m + n != len(s3):
        return False

    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True

    for i in range(1, m + 1):
        dp[i][0] = dp[i-1][0] and s1[i-1] == s3[i-1]
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j-1] and s2[j-1] == s3[j-1]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = (
                (dp[i-1][j] and s1[i-1] == s3[i+j-1]) or
                (dp[i][j-1] and s2[j-1] == s3[i+j-1])
            )

    return dp[m][n]

# Example: s1="aab", s2="axy", s3="aaxaby" -> True
# "aaxaby" = a(s1) + a(s2) + x(s2) + a(s1) + b(s1) + y(s2)
```

### 6.4 Stock Trading Problems

```python
def max_profit_k_transactions(prices, k):
    """Maximum profit with at most k transactions"""
    n = len(prices)
    if n <= 1:
        return 0

    # If k >= n//2, it's effectively unlimited
    if k >= n // 2:
        return sum(max(0, prices[i+1] - prices[i]) for i in range(n-1))

    # dp[t][i] = maximum profit with up to t transactions through day i
    dp = [[0] * n for _ in range(k + 1)]

    for t in range(1, k + 1):
        max_diff = -prices[0]
        for i in range(1, n):
            dp[t][i] = max(dp[t][i-1], prices[i] + max_diff)
            max_diff = max(max_diff, dp[t-1][i] - prices[i])

    return dp[k][n-1]

# Time complexity: O(k x n)

# Variation: With cooldown period
def max_profit_with_cooldown(prices):
    """Cannot buy the day after selling (1-day cooldown)"""
    n = len(prices)
    if n <= 1:
        return 0

    # hold: maximum profit while holding stock
    # sold: maximum profit after selling today
    # rest: maximum profit holding nothing & in cooldown/waiting
    hold = -prices[0]
    sold = 0
    rest = 0

    for i in range(1, n):
        prev_hold = hold
        hold = max(hold, rest - prices[i])
        rest = max(rest, sold)
        sold = prev_hold + prices[i]

    return max(sold, rest)

# Variation: With transaction fee
def max_profit_with_fee(prices, fee):
    """Each transaction incurs a fee"""
    n = len(prices)
    hold = -prices[0]  # Holding stock
    cash = 0           # Not holding stock

    for i in range(1, n):
        hold = max(hold, cash - prices[i])
        cash = max(cash, hold + prices[i] - fee)

    return cash
```

---

## 7. Practice Exercises

### Exercise 1: Basic DP (Fundamental)
Staircase climbing problem: Find the total number of ways to climb N stairs taking 1 or 2 steps at a time.

### Exercise 2: 2D DP (Applied)
Implement a function that computes edit distance (Levenshtein distance). Find the edit distance from "kitten" to "sitting."

### Exercise 3: Bitmask DP (Advanced)
Implement a program that solves the Traveling Salesman Problem (TSP) with bitmask DP in O(n^2 x 2^n).

### Exercise 4: Interval DP (Advanced)
Solve the matrix chain multiplication problem and recover the optimal parenthesization.

### Exercise 5: Practical Application (Advanced)
Implement a diff tool using LCS to display differences between two text files.

### Exercise 6: Digit DP (Advanced)
Count how many times each digit (0-9) appears in integers from 1 to N.

### Exercise 7: Game Theory DP (Advanced)
Find the optimal strategy for the stone game (take stones from left or right end to maximize your total).

### Exercise 8: Probability DP (Advanced)
In a board game, calculate the expected number of trials to land exactly on goal (square N) by rolling a 6-sided die.

---

## FAQ

### Q1: What is the difference between DP and divide-and-conquer?
**A**: Divide-and-conquer has independent subproblems (merge sort: left and right are independent). DP has overlapping subproblems (Fibonacci: fib(n-1) and fib(n-2) share fib(n-3)). Divide-and-conquer + memoization = DP, in a sense.

### Q2: What are tips for DP state design?
**A**: First think about "what must be decided so the rest is determined." The state should be "the minimum information needed to solve the remaining problem." If there are too many states, consider dimension reduction. Start with a naive DP and then optimize.

### Q3: When to use greedy vs. DP?
**A**: Greedy works only when local optima lead to global optima (proof required). DP considers all choices and is always correct. When in doubt, DP is safer. When greedy works, it has lower complexity.

### Q4: Should I use memoization or bottom-up?
**A**: Generally, memoization is easier to write and avoids computing unnecessary subproblems. However, Python's recursion depth limit (default 1000) means bottom-up is safer for large inputs. Space optimization is also easier with bottom-up. Bottom-up is mainstream in competitive programming.

### Q5: What if the DP table is too large?
**A**: (1) Space optimization (rolling array) (2) Redefine states (dimension reduction) (3) Memoized recursion computing only needed states (4) Bitmask compression (5) Hash map-based memoization. If none work, switch to an approximation algorithm.

### Q6: Where specifically does DP knowledge help in practice?
**A**: (1) Text editor diff functionality (LCS) is fundamental to version control (2) Spell checker candidate suggestions (edit distance) (3) Network routing (DP-based shortest path approaches) (4) Viterbi algorithm in machine learning (speech recognition, POS tagging) (5) Database query optimization (determining join order) (6) Compiler instruction selection (a form of tree DP)

### Q7: What does pseudo-polynomial time complexity mean?
**A**: The O(nW) of the knapsack problem depends on the input value (W), not the input size. In binary representation, W takes log W bits, so it is exponential in the number of input bits. This is called "pseudo-polynomial time." NP-hard problems can still be solved practically fast when input values are small.

---

## Summary

| Concept | Key Point |
|---------|-----------|
| DP conditions | Optimal substructure + overlapping subproblems |
| Memoization | Top-down. Recursion + cache |
| Bottom-up | Tabulation. For loop + array |
| Classic patterns | Knapsack, LCS, coin change, LIS, edit distance |
| State design | Make the state "the minimum information needed to solve the rest" |
| Space optimization | Rolling array, 1D reduction |
| Advanced DP | Interval DP, tree DP, digit DP, bitmask DP, probability DP |
| Transition speedup | Prefix sums, Convex Hull Trick, Knuth's optimization |
| Debugging | Manual small-case verification, brute-force comparison, table display |

---

## Recommended Next Guides

---

## References
1. Cormen, T. H. et al. "Introduction to Algorithms." Chapter 15: Dynamic Programming.
2. Bellman, R. "Dynamic Programming." Princeton University Press, 1957.
3. Skiena, S. S. "The Algorithm Design Manual." Chapter 10.
4. Halim, S. "Competitive Programming 3." Chapter 3.5: Dynamic Programming.
5. Knuth, D. E. "The Art of Computer Programming, Volume 3." Sorting and Searching.
6. Dasgupta, S., Papadimitriou, C., Vazirani, U. "Algorithms." Chapter 6: Dynamic Programming.