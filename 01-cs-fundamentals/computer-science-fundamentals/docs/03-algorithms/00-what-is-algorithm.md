# What Is an Algorithm?

> An algorithm is "a well-defined set of steps for solving a problem." The essence of programming lies in the design and implementation of algorithms.

## Learning Objectives

- [ ] Explain the definition of an algorithm and its five essential properties
- [ ] Understand the difference between good and bad algorithms
- [ ] Develop the thought process of problem → algorithm → code
- [ ] Be able to choose among major algorithm design strategies
- [ ] Understand basic techniques for proving the correctness of algorithms

## Prerequisites

- Basic programming experience

---

## 1. Definition of an Algorithm

### 1.1 The Five Properties of an Algorithm

```
Algorithm:
  A well-defined computational procedure that solves a problem
  in a finite number of steps.

  Five Properties (Knuth, 1968):

  1. Input:
     → Receives zero or more external data
     → Example: Sorting → an array of numbers

  2. Output:
     → Produces one or more results
     → Example: Sorting → a sorted array

  3. Definiteness:
     → Each step is unambiguously defined
     → Bad: "Sort them in a nice way somehow"
     → Good: "Compare adjacent elements and swap if the left is greater"

  4. Finiteness:
     → Terminates after a finite number of steps
     → An infinite loop is not an algorithm

  5. Effectiveness:
     → Each step is sufficiently basic and executable
     → "List all prime numbers" is invalid (infinite)
     → "List all prime numbers up to N" is valid
```

### 1.2 Etymology and History of Algorithms

```
History of Algorithms:

  Etymology:
  - Named after the 9th-century Persian mathematician al-Khwarizmi
  - His name was Latinized as "Algoritmi"
  - His work "On the Calculation with Hindu Numerals" introduced
    decimal arithmetic procedures

  Historically Important Algorithms:
  +--------------------+------+--------------------------------------+
  | Algorithm          | Era  | Overview                             |
  +--------------------+------+--------------------------------------+
  | Euclidean          | 300  | Finds the GCD of two numbers         |
  | Algorithm          | BC   | → The oldest known algorithm          |
  | Sieve of           | 200  | Enumerates primes by sieving         |
  | Eratosthenes       | BC   |                                      |
  | al-Khwarizmi's     | 830  | Solutions for linear/quadratic       |
  | Algebra            |      | equations → Origin of "algebra"      |
  | Newton's Method    | 1669 | Iteratively finds approximate roots  |
  | Gaussian           | 1809 | Solves systems of linear equations   |
  | Elimination        |      |                                      |
  | Babbage's          | 1837 | The concept of mechanical            |
  | Analytical Engine  |      | computation → Prototype of programs  |
  | Turing Machine     | 1936 | Theoretical foundation of            |
  |                    |      | computability                        |
  | Von Neumann        | 1945 | Stored-program architecture          |
  | Merge Sort         | 1945 | Invented by von Neumann              |
  +--------------------+------+--------------------------------------+

  Turning points when "algorithm" acquired its modern meaning:
  - 1936: Turing rigorously defined "computable"
  - 1945: Von Neumann proposed the stored-program architecture
  - 1960s: Knuth published "The Art of Computer Programming"
  - → Algorithm analysis became an independent field of research
```

### 1.3 Algorithms in Everyday Life

```
A Cooking Recipe = An Algorithm:

  How to Make Curry:
  Input: 2 onions, 300g meat, 1 box curry roux, 800ml water
  Output: 4 servings of curry

  1. Slice the onions thinly
  2. Add oil to a pot and heat on medium
  3. Saute the onions until golden brown (about 10 minutes)
  4. Add the meat and stir-fry until it changes color
  5. Add 800ml of water and bring to a boil
  6. Simmer on low heat for 20 minutes
  7. Turn off the heat, break in the roux and dissolve it
  8. Simmer on low heat for 5 minutes
  → Done

  This satisfies all five properties of an algorithm:
  Input: Ingredients
  Output: Curry
  Definiteness: Each step is specific
  Finiteness: Completes in 8 steps
  Effectiveness: Each step is executable
```

```
Other Everyday Examples of Algorithms:

  1. Looking up a word in a dictionary (Binary Search):
  +----------------------------------------------+
  | Input: A dictionary, the word to look up     |
  | Output: The meaning of the word              |
  |                                              |
  | 1. Open the dictionary roughly to the middle |
  | 2. Compare the word on the page with the     |
  |    target word                               |
  | 3. If the target comes "before," search the  |
  |    first half                                |
  | 4. If the target comes "after," search the   |
  |    second half                               |
  | 5. Repeat steps 2-4 until found              |
  |                                              |
  | → Even a 1000-page dictionary takes only     |
  |   about 10 lookups!                          |
  +----------------------------------------------+

  2. Sorting a hand of playing cards (Insertion Sort):
  +----------------------------------------------+
  | Input: Cards in your hand                    |
  | Output: Cards sorted by number               |
  |                                              |
  | 1. Take the second card                      |
  | 2. Compare it with cards to its left         |
  | 3. Insert it at the correct position         |
  | 4. Repeat steps 2-3 for the next card        |
  |                                              |
  | → The sorting method humans naturally use!   |
  +----------------------------------------------+

  3. Solving a maze (DFS / Wall-Following):
  +----------------------------------------------+
  | Input: A maze (with an entrance and exit)    |
  | Output: A path from entrance to exit         |
  |                                              |
  | Wall-following method:                       |
  | 1. Place your right hand on the wall         |
  | 2. Walk along the wall                       |
  | 3. Repeat until you reach the exit           |
  |                                              |
  | → Simple but guaranteed to reach the exit    |
  |   (for simply connected mazes)               |
  +----------------------------------------------+

  4. Making change (Greedy Algorithm):
  +----------------------------------------------+
  | Input: The amount of change                  |
  | Output: The minimum number of coins          |
  |                                              |
  | 1. Use the largest coin denomination first,  |
  |    as many as possible                       |
  |    500 yen → 100 yen → 50 yen → 10 yen      |
  |    → 5 yen → 1 yen                           |
  | 2. Repeat until the remaining amount is 0    |
  |                                              |
  | Example: 680 yen in change                   |
  | 500 yen x1, 100 yen x1, 50 yen x1,          |
  | 10 yen x3 = 6 coins total                   |
  |                                              |
  | → Produces optimal results with the Japanese |
  |   coin system (but not always in general)    |
  +----------------------------------------------+
```

### 1.4 Examples of Things That Are Not Algorithms

```
Things That Are NOT Algorithms:

  1. "Sort them somehow"
     → Lacks definiteness
     → "By what criterion?" and "How to sort?" are unclear

  2. "List all prime numbers"
     → Violates finiteness
     → Prime numbers are infinite, so it never terminates

  3. "Intuitively pick the best element from the array"
     → Lacks effectiveness
     → "Intuition" is not an executable computational step

  4. while True: print("Hello")
     → Violates finiteness
     → Never terminates

  Note: The difference between an algorithm and a "program"
  +----------------------------------------------+
  | Algorithm:                                   |
  | - Must terminate in finite time              |
  | - An abstract procedure independent of       |
  |   programming language                       |
  |                                              |
  | Program:                                     |
  | - Need not terminate (e.g., OS, servers)     |
  | - Written in a specific programming language |
  |                                              |
  | → Every algorithm can be made into a program,|
  |   but not every program is an algorithm      |
  +----------------------------------------------+
```

---

## 2. Algorithm Design Strategies

### 2.1 Major Design Patterns

```
Seven Major Algorithm Design Strategies:

  +-------------------+----------------------------------------------+
  | Strategy          | Overview                                     |
  +-------------------+----------------------------------------------+
  | Brute Force       | Try all possibilities (exhaustive search)    |
  | Divide & Conquer  | Divide → solve individually → combine        |
  | Dynamic           | Memoize and reuse solutions to subproblems   |
  | Programming       |                                              |
  | Greedy            | Make the locally optimal choice at each step |
  | Backtracking      | Explore, then backtrack when stuck           |
  | Binary Search     | Narrow the range by half at each step        |
  | Graph Search      | Explore the state space with BFS/DFS         |
  +-------------------+----------------------------------------------+
```

### 2.2 Brute Force

```python
# Brute Force: Naively try all possibilities
# Pros: Guaranteed to find the correct answer, simple to implement
# Cons: High computational complexity

# Example 1: PIN brute-force attack (4-digit number)
def brute_force_pin():
    """Try all 4-digit PINs"""
    for pin in range(10000):  # 0000-9999
        if try_pin(f"{pin:04d}"):
            return f"{pin:04d}"
    return None
# Complexity: O(10^4) = O(10000)

# Example 2: Subset Sum Problem
def subset_sum(nums, target):
    """Check if any subset of nums sums to target"""
    n = len(nums)
    # Enumerate all subsets using bitmasks
    for mask in range(1 << n):  # 2^n subsets
        total = 0
        for i in range(n):
            if mask & (1 << i):
                total += nums[i]
        if total == target:
            return True
    return False
# Complexity: O(n * 2^n)

# Example 3: Anagram check (naive approach)
def is_anagram_brute(s1, s2):
    """Check if s1 and s2 are anagrams by examining all permutations"""
    from itertools import permutations
    if len(s1) != len(s2):
        return False
    for perm in permutations(s1):
        if ''.join(perm) == s2:
            return True
    return False
# Complexity: O(n! * n) — extremely inefficient!

# Improved version: Compare using sorting
def is_anagram_sort(s1, s2):
    return sorted(s1) == sorted(s2)
# Complexity: O(n log n) — significantly improved

# Further improved: Compare using a hash map
from collections import Counter
def is_anagram_hash(s1, s2):
    return Counter(s1) == Counter(s2)
# Complexity: O(n) — optimal

# → Brute force is the starting point. From here, consider more efficient methods.
```

### 2.3 Divide and Conquer

```python
# Divide and Conquer: Split the problem → solve each part → combine
# Three steps: Divide → Conquer → Combine

# Classic Example 1: Merge Sort
def merge_sort(arr):
    """Divide → Sort → Combine"""
    if len(arr) <= 1:
        return arr

    # Divide: Split in half
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])    # Conquer
    right = merge_sort(arr[mid:])   # Conquer

    # Combine: Merge
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
# T(n) = 2T(n/2) + O(n) → O(n log n)

# Classic Example 2: Closest Pair of Points
def closest_pair(points):
    """Find the two closest points among n points on a 2D plane

    Brute force: O(n^2) — check all pairs
    Divide and conquer: O(n log n)

    1. Sort by x-coordinate
    2. Split into left and right halves at the center
    3. Recursively find the closest pair in each half
    4. Also check pairs that span the dividing line (strip region)
    5. Return the minimum
    """
    # Detailed implementation omitted (focus is on the concept)
    pass

# Classic Example 3: Karatsuba Algorithm (fast multiplication)
def karatsuba(x, y):
    """Improves multiplication from O(n^2) to O(n^1.585)"""
    if x < 10 or y < 10:
        return x * y

    n = max(len(str(x)), len(str(y)))
    m = n // 2

    # x = a * 10^m + b
    # y = c * 10^m + d
    a, b = divmod(x, 10**m)
    c, d = divmod(y, 10**m)

    # 3 recursive calls (instead of the usual 4)
    ac = karatsuba(a, c)
    bd = karatsuba(b, d)
    ad_bc = karatsuba(a + b, c + d) - ac - bd

    return ac * 10**(2*m) + ad_bc * 10**m + bd
# T(n) = 3T(n/2) + O(n) → O(n^log_2(3)) ≈ O(n^1.585)
# → The key insight is reducing 4 multiplications to 3

# Conditions for divide and conquer to be effective:
# 1. The problem can be split into smaller problems of the same type
# 2. The smaller problems can be solved independently
# 3. Partial solutions can be combined efficiently
# 4. The base case is trivially solvable
```

### 2.4 Dynamic Programming (DP)

```python
# Dynamic Programming: Memoize and reuse solutions to subproblems
# Two conditions:
# 1. Optimal substructure: The optimal solution is composed of optimal subproblem solutions
# 2. Overlapping subproblems: The same subproblems are solved repeatedly

# Classic Example: Fibonacci Sequence

# Bad: Naive recursion O(2^n) — repeats the same computation many times
def fib_naive(n):
    if n <= 1:
        return n
    return fib_naive(n-1) + fib_naive(n-2)

# Recursion tree (for n=5):
#                fib(5)
#              /        \
#           fib(4)      fib(3)
#          /    \       /    \
#       fib(3) fib(2) fib(2) fib(1)
#       /  \
#    fib(2) fib(1)
# → fib(2) is computed 3 times, fib(3) is computed 2 times!

# Good: Memoization (Top-Down DP): O(n)
def fib_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]

# Good: Tabulation (Bottom-Up DP): O(n)
def fib_table(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]

# Good: Space-optimized: O(n) time, O(1) space
def fib_optimal(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# Practical DP Example: Knapsack Problem
def knapsack(weights, values, capacity):
    """Maximize value within a weight constraint"""
    n = len(weights)
    # dp[i][w] = maximum value using the first i items with weight limit w
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # Do not include item i
            dp[i][w] = dp[i-1][w]
            # Include item i
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w],
                    dp[i-1][w - weights[i-1]] + values[i-1])

    return dp[n][capacity]
# Complexity: O(n * capacity)
```

### 2.5 Greedy Algorithms

```python
# Greedy Algorithm: Choose "the best option at each step"
# Pros: Fast, simple to implement
# Cons: Does not always yield the optimal solution

# Examples where greedy yields the optimal solution:

# 1. Activity Selection Problem
def activity_selection(activities):
    """Select the maximum number of non-overlapping activities
    activities = [(start, end), ...]
    """
    # Sort by end time (greedy selection criterion)
    sorted_activities = sorted(activities, key=lambda x: x[1])

    selected = [sorted_activities[0]]
    for activity in sorted_activities[1:]:
        if activity[0] >= selected[-1][1]:  # No overlap
            selected.append(activity)

    return selected
# Complexity: O(n log n) (sort) + O(n) = O(n log n)
# → The greedy approach is proven to yield the optimal solution

# 2. Huffman Coding
import heapq
def huffman_encoding(freq):
    """Generate optimal variable-length codes from character frequencies

    Example: {'a': 5, 'b': 9, 'c': 12, 'd': 13, 'e': 16, 'f': 45}
    """
    # Build a min-heap
    heap = [(f, c) for c, f in freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        # Extract and combine the two smallest (greedy choice)
        freq1, node1 = heapq.heappop(heap)
        freq2, node2 = heapq.heappop(heap)
        heapq.heappush(heap, (freq1 + freq2, (node1, node2)))

    return heap[0]
# → Produces an optimal prefix code (proven by information theory)

# Example where greedy does NOT yield the optimal solution:

# Change-making problem (special coin system)
# Coins: [1, 3, 4], Change: 6
# Greedy: 4 + 1 + 1 = 3 coins
# Optimal: 3 + 3 = 2 coins!
# → DP is needed in this case

# Conditions for greedy to work:
# 1. Greedy choice property: A locally optimal choice leads to a globally optimal solution
# 2. Optimal substructure: The remaining subproblem can be solved in the same way
# → Both properties need to be proven
```

### 2.6 Backtracking

```python
# Backtracking: Explore + backtrack when stuck
# "Systematic trial and error"

# Classic Example 1: N-Queens Problem
def solve_n_queens(n):
    """Place n queens on an N×N chessboard so that no two attack each other"""
    solutions = []
    board = [-1] * n  # board[row] = col

    def is_safe(row, col):
        for prev_row in range(row):
            prev_col = board[prev_row]
            # Check if same column or diagonal
            if prev_col == col or \
               abs(prev_col - col) == abs(prev_row - row):
                return False
        return True

    def backtrack(row):
        if row == n:
            solutions.append(board[:])
            return

        for col in range(n):
            if is_safe(row, col):
                board[row] = col        # Choose
                backtrack(row + 1)      # Explore
                board[row] = -1         # Undo (backtrack)

    backtrack(0)
    return solutions

# 8-Queens: 92 solutions (when distinguishing rotations and reflections)
# Complexity: Worst case O(n!), significantly reduced by pruning

# Classic Example 2: Sudoku Solver
def solve_sudoku(board):
    """Solve a 9x9 Sudoku puzzle"""
    def find_empty():
        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    return (i, j)
        return None

    def is_valid(num, pos):
        row, col = pos
        # Row check
        if num in board[row]:
            return False
        # Column check
        if num in [board[i][col] for i in range(9)]:
            return False
        # 3x3 box check
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                if board[i][j] == num:
                    return False
        return True

    empty = find_empty()
    if not empty:
        return True  # All cells filled → complete

    row, col = empty
    for num in range(1, 10):
        if is_valid(num, (row, col)):
            board[row][col] = num       # Choose
            if solve_sudoku(board):     # Explore
                return True
            board[row][col] = 0         # Backtrack

    return False  # No solution from this state
```

### 2.7 Problem-Solving Framework

```
Problem → Algorithm → Code Thought Process:

  Step 1: Understand the Problem
  ─────────────────────────────
  - What is the input? What is the output?
  - What are the constraints? (data size, time limits)
  - What are the edge cases?

  Step 2: Solve Examples by Hand
  ─────────────────────────────
  - Work through small, concrete examples manually
  - Look for patterns
  - Be conscious of the procedure you are following

  Step 3: Design the Algorithm
  ─────────────────────────────
  - Write out the steps in words
  - Estimate the computational complexity
  - Consider if there is a better approach

  Step 4: Convert to Code
  ─────────────────────────────
  - Pseudocode → actual code
  - Add edge case handling

  Step 5: Test and Verify
  ─────────────────────────────
  - Normal cases: basic inputs
  - Boundary values: empty, single element, maximum values
  - Error cases: invalid inputs
```

```
Design Strategy Selection Flowchart:

  When you see a problem → First consider brute force (O(n!)? O(2^n)?)

  Check the constraints:
  +-- n <= 10      → Brute force is fine
  +-- n <= 20      → Bitmask enumeration, backtracking
  +-- n <= 500     → O(n^3) DP
  +-- n <= 5000    → O(n^2) DP, all pairs
  +-- n <= 10^6    → O(n log n) sort + binary search, divide & conquer
  +-- n <= 10^8    → O(n) linear scan, greedy

  Check the nature of the problem:
  +-- Optimization problem (max/min)
  |   +-- Decomposable into subproblems → DP
  |   +-- Local optimum = global optimum (provable) → Greedy
  |   +-- Binary search on the answer is possible → Binary search + predicate
  +-- Search problem (existence/enumeration)
  |   +-- Graph/tree structure → BFS/DFS
  |   +-- Constraint satisfaction → Backtracking
  |   +-- Enumerate all patterns → Recursion/bitmask enumeration
  +-- Transformation problem (convert A to B)
      +-- Minimum steps → BFS
      +-- Feasibility check → DFS/DP
```

### 2.8 Concrete Example: Finding the Maximum Value in an Array

```python
# Problem: Find the maximum value in an array
# Input: An array of integers (non-empty)
# Output: The maximum value

# Method 1: Brute force
def find_max_brute(arr):
    """Check every element one by one"""
    max_val = arr[0]
    for i in range(1, len(arr)):
        if arr[i] > max_val:
            max_val = arr[i]
    return max_val
# Complexity: O(n) — cannot be improved (every element must be examined)

# Method 2: Divide and conquer
def find_max_divide(arr, left, right):
    """Split the array in half and compare the max of each half"""
    if left == right:
        return arr[left]
    mid = (left + right) // 2
    left_max = find_max_divide(arr, left, mid)
    right_max = find_max_divide(arr, mid + 1, right)
    return max(left_max, right_max)
# Complexity: O(n) — same, but well-suited for parallelization

# Method 3: Python built-in
max_val = max(arr)  # Internally same as Method 1: O(n)

# For this problem, O(n) is optimal (lower bound)
# → Every element must be examined at least once to determine the maximum
# → This is called an "information-theoretic lower bound"
```

### 2.9 Concrete Example: Two Sum

```python
# Problem: Return indices of two elements that sum to target
# Input: nums = [2, 7, 11, 15], target = 9
# Output: [0, 1] (nums[0] + nums[1] = 2 + 7 = 9)

# Method 1: Brute force O(n^2)
def two_sum_brute(nums, target):
    """Try all pairs"""
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []

# Method 2: Hash map O(n)
def two_sum_hash(nums, target):
    """Record seen values and look for the complement"""
    seen = {}  # value → index
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

# Method 3: Sort + binary search O(n log n)
def two_sum_sort(nums, target):
    """Sort and search from both ends"""
    indexed = sorted(enumerate(nums), key=lambda x: x[1])
    left, right = 0, len(indexed) - 1
    while left < right:
        total = indexed[left][1] + indexed[right][1]
        if total == target:
            return [indexed[left][0], indexed[right][0]]
        elif total < target:
            left += 1
        else:
            right -= 1
    return []

# Comparison:
# +-------------+-----------+-----------+-------------------+
# | Method      | Time      | Space     | Notes             |
# +-------------+-----------+-----------+-------------------+
# | Brute force | O(n^2)    | O(1)      | Simplest          |
# | Hash map    | O(n)      | O(n)      | Fastest, trades   |
# |             |           |           | space for speed   |
# | Sort+search | O(n log n)| O(n)      | Middle ground     |
# +-------------+-----------+-----------+-------------------+

# → Choose the best method based on requirements
# - Speed priority → Hash map
# - Memory constraint → Brute force (or Method 3 if indices not needed)
# - Multiple queries → Sort (after preprocessing, each search is O(log n))
```

---

## 3. Proving Algorithm Correctness

### 3.1 Loop Invariants

```python
# Loop Invariant:
# A property that holds true before each iteration of a loop

def insertion_sort(arr):
    """Insertion Sort"""
    for i in range(1, len(arr)):
        # Loop invariant: arr[0:i] is sorted
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
        # Loop invariant: arr[0:i+1] is sorted
    # At termination: i = len(arr), so arr[0:len(arr)] is sorted

# Three properties of a loop invariant:
# 1. Initialization: Holds before the loop starts (arr[0:1] is trivially sorted)
# 2. Maintenance: Maintained at each iteration (sorted range expands each step)
# 3. Termination: When the loop ends, the desired property holds (entire array is sorted)
```

```python
# Another example: Proving correctness of linear search

def linear_search(arr, target):
    """Linear Search"""
    for i in range(len(arr)):
        # Loop invariant: target does not exist in arr[0:i]
        if arr[i] == target:
            return i
    # At termination: target does not exist in arr[0:len(arr)]
    return -1

# Proof:
# Initialization: When i=0, arr[0:0] (empty set) does not contain target → trivially true
# Maintenance: At iteration i, if arr[i] != target, then arr[0:i+1] also does not contain target
#              → invariant holds for i+1
# Termination: If i = len(arr), the entire arr does not contain target
#              → return -1 is correct
#              If arr[i] == target is found mid-loop, return i is also correct
```

### 3.2 Correctness of Recursion — Mathematical Induction

```python
# Proving algorithm correctness using mathematical induction

def factorial(n):
    """Compute n! recursively"""
    if n <= 1:    # Base case
        return 1
    return n * factorial(n - 1)  # Recursive case

# Proof:
# Base case: For n=0, n=1, returns 1 → 0!=1, 1!=1
# Inductive step:
#   Assume factorial(k) = k! is correct (inductive hypothesis)
#   factorial(k+1) = (k+1) * factorial(k) = (k+1) * k! = (k+1)!
# Therefore factorial(n) = n! for all n >= 0
```

```python
# More complex example: Proving correctness of merge sort

def merge_sort(arr):
    if len(arr) <= 1:
        return arr  # Base case
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

# Proof (strong induction):
#
# Proposition: For any array of length n, merge_sort(arr) returns a
#              sorted array containing the same elements as arr
#
# Base case: n=0 or n=1
#   → The array is returned as-is → trivially sorted
#
# Inductive step: Assume correctness for all inputs of size less than n, where n > 1
#   1. arr[:mid] has length n/2 < n → by inductive hypothesis, left is sorted
#   2. arr[mid:] has length n - n/2 < n → by inductive hypothesis, right is sorted
#   3. merge(left, right):
#      - Both left and right are sorted
#      - merge picks the smaller element at each step
#      - → the result is sorted
#      - No elements are added or removed
#      - → all original elements are included
#
# Therefore merge_sort is correct for all n >= 0
```

### 3.3 Proof by Contradiction

```python
# Proof by Contradiction: Assume incorrectness and derive a contradiction

# Example: Correctness of Dijkstra's algorithm (sketch)
#
# Proposition: Dijkstra's algorithm correctly finds shortest paths in
#              non-negative-weight graphs
#
# Proof (sketch):
# Assume, for contradiction, that there exists a node v for which
# Dijkstra's first assigns an incorrect shortest distance
#
# d[v] = distance computed by Dijkstra
# delta(v) = true shortest distance
# By assumption: d[v] > delta(v)
#
# Consider the true shortest path s → ... → u → v
# u is the node immediately before v on this path
#
# Since u was processed before v: d[u] = delta(u) (v is the "first" mistake)
# d[v] <= d[u] + w(u,v)  (by Dijkstra's relaxation step)
#       = delta(u) + w(u,v)   (since d[u] = delta(u))
#       = delta(v)            (since u→v is part of the shortest path)
#
# Therefore d[v] <= delta(v), but d[v] >= delta(v) is trivially true
# → d[v] = delta(v), which is a contradiction!
#
# Therefore Dijkstra's algorithm is correct
```

---

## 4. Classification of Algorithms

### 4.1 By Problem Type

```
Problem Classification and Algorithms:

  +-------------+------------------+-----------------------+
  | Problem Type| Representative   | Practical Examples    |
  |             | Algorithms       |                       |
  +-------------+------------------+-----------------------+
  | Search      | Binary search,   | DB queries, dictionary|
  |             | hashing          | lookups               |
  | Sorting     | Quicksort,       | Rankings,             |
  |             | merge sort       | data organization     |
  | Graph       | BFS, DFS,        | Route finding,        |
  |             | Dijkstra         | SNS friend suggestion |
  | String      | KMP, Rabin-Karp  | Full-text search,     |
  |             |                  | text editors          |
  | Optimization| DP, greedy       | Scheduling,           |
  |             |                  | resource allocation   |
  | Numerical   | Newton's method, | Scientific simulation,|
  |             | FFT              | signal processing     |
  | Cryptography| RSA, AES,        | HTTPS,                |
  |             | SHA-256          | digital signatures    |
  | Machine     | Gradient descent,| Recommendations,      |
  | Learning    | backpropagation  | image recognition     |
  +-------------+------------------+-----------------------+
```

### 4.2 From the Perspective of Computability

```
Classification of Problem Difficulty:

  +----------------------------------------------------------+
  | P (Polynomial): Problems solvable in polynomial time     |
  | Examples: Sorting O(n log n), shortest path O(V+E)       |
  |                                                          |
  | NP: Problems whose solutions can be verified in          |
  |     polynomial time                                      |
  | Example: A TSP solution is easy to verify (O(n)),        |
  |          but hard to "find"                              |
  |                                                          |
  | NP-hard: Problems at least as hard as NP                 |
  | NP-complete: Both NP and NP-hard                         |
  |                                                          |
  | Open problem: Is P = NP or P != NP?                      |
  | → The greatest unsolved problem in CS                    |
  |   ($1M Millennium Prize Problem)                         |
  +----------------------------------------------------------+

  Practical Implications:
  +----------+----------------+-----------------------------+
  | Class    | Meaning        | Approach                    |
  +----------+----------------+-----------------------------+
  | P        | Efficiently    | Implement directly          |
  |          | solvable       |                             |
  | NP-      | Efficient      | Approximation algorithms,   |
  | complete | solution       | heuristics                  |
  |          | likely doesn't |                             |
  |          | exist          |                             |
  | Undeci-  | Unsolvable     | Add constraints to solve    |
  | dable    |                | a restricted version        |
  +----------+----------------+-----------------------------+

  Real-world NP-complete problems:
  - Traveling Salesman Problem (TSP): Delivery route optimization
  - Graph coloring: Scheduling, register allocation
  - Boolean Satisfiability (SAT): Logic circuit verification
  - Knapsack problem: Resource allocation
  - Hamiltonian path: Circuit design

  When a problem is known to be NP-complete:
  1. If the input size is small, brute force is acceptable
  2. Use approximation algorithms (guaranteed within 1.5x of optimal, etc.)
  3. Use heuristics (genetic algorithms, simulated annealing)
  4. Find special cases (where adding constraints makes it P)
```

### 4.3 Common Algorithm Patterns in Practice

```
Algorithm Patterns Frequently Used in Practice:

  1. Binary Search Pattern
  +----------------------------------------------+
  | Context: Searching in sorted data            |
  | Examples: API rate limit threshold search     |
  |          Finding the bug-introducing commit  |
  |          Optimal parameter search             |
  +----------------------------------------------+

  2. Hash Map Pattern
  +----------------------------------------------+
  | Context: O(1) lookups needed                 |
  | Examples: Caching, deduplication             |
  |          Counting, grouping                  |
  |          Search optimization (like Two Sum)  |
  +----------------------------------------------+

  3. Stack/Queue Pattern
  +----------------------------------------------+
  | Context: Processing with ordering constraints|
  | Examples: Parenthesis matching               |
  |          Task scheduling                     |
  |          BFS/DFS                             |
  +----------------------------------------------+

  4. Two Pointers / Sliding Window
  +----------------------------------------------+
  | Context: Array/string subsequence processing |
  | Examples: Maximum subarray sum               |
  |          Longest substring without repeats   |
  |          Merge operations                    |
  +----------------------------------------------+

  5. Graph Search Pattern
  +----------------------------------------------+
  | Context: Exploring relationships             |
  | Examples: Dependency resolution (build order)|
  |          Social graph analysis               |
  |          Reachability determination           |
  +----------------------------------------------+

  6. DP Pattern
  +----------------------------------------------+
  | Context: Optimization problems               |
  | Examples: Longest Common Subsequence (diff)  |
  |          Edit distance (fuzzy search)        |
  |          Counting paths                      |
  +----------------------------------------------+
```

---

## 5. Writing Pseudocode

### 5.1 Pseudocode Notation

```
Pseudocode:
  A description of an algorithm's procedure in a form close to
  natural language, independent of any specific programming language.

Basic notation:
  +----------------------------------------------+
  | Assignment:   x <- value                     |
  | Conditional:  if condition then ... else ...  |
  | Loop:         for i <- 1 to n do ...         |
  |               while condition do ...          |
  | Array:        A[i] for the i-th element      |
  | Function:     function name(params) -> return |
  | Comment:      // This line is a comment      |
  | Input:        read(x)                        |
  | Output:       print(x)                       |
  | Return value: return value                   |
  +----------------------------------------------+

Example: Pseudocode for Binary Search

  function BINARY-SEARCH(A, target)
    left <- 0
    right <- length(A) - 1

    while left <= right do
      mid <- floor((left + right) / 2)

      if A[mid] = target then
        return mid
      else if A[mid] < target then
        left <- mid + 1
      else
        right <- mid - 1

    return -1  // not found
```

### 5.2 Converting Pseudocode to Code

```python
# Converting pseudocode to Python, Java, Go, and TypeScript

# Pseudocode:
# function GCD(a, b)
#   while b != 0 do
#     temp <- b
#     b <- a mod b
#     a <- temp
#   return a

# Python:
def gcd(a: int, b: int) -> int:
    while b != 0:
        a, b = b, a % b
    return a
```

```java
// Java:
public static int gcd(int a, int b) {
    while (b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}
```

```go
// Go:
func gcd(a, b int) int {
    for b != 0 {
        a, b = b, a%b
    }
    return a
}
```

```typescript
// TypeScript:
function gcd(a: number, b: number): number {
    while (b !== 0) {
        [a, b] = [b, a % b];
    }
    return a;
}
```

---

## 6. Evaluation Criteria for Algorithms

### 6.1 Evaluation Criteria Beyond Time Complexity

```
Comprehensive Evaluation Criteria for Algorithms:

  1. Time Complexity
     → Growth rate of execution time relative to input size
     → Worst / average / best case

  2. Space Complexity
     → Amount of additional memory required
     → Note whether the input data itself is included

  3. Correctness
     → Does it return the correct output for all valid inputs?
     → Proof methods: Loop invariants, induction, proof by contradiction

  4. Stability
     → Specific to sorting algorithms: Does it preserve the relative
        order of equal elements?

  5. Adaptivity
     → Does performance improve based on input characteristics
        (e.g., nearly sorted)?
     → Example: Insertion sort runs in O(n) on nearly sorted data

  6. Online Property
     → Can it process data as it arrives incrementally?
     → Example: Insertion sort is online; merge sort is offline

  7. Parallelizability
     → Can it be efficiently parallelized on multi-core/GPU?
     → Example: Merge sort is well-suited for parallelization

  8. Cache Efficiency
     → Does it use the CPU cache efficiently?
     → Example: Quicksort has good cache efficiency

  9. Implementation Simplicity
     → Code complexity, susceptibility to bugs
     → An important consideration in practice

  10. Constant Factor
      → Even with the same Big-O, actual speed differs by constant factors
      → Example: O(n) with factor 2n vs 100n is a 50x difference
```

### 6.2 Practical Judgment for Algorithm Selection

```
Guidelines for Algorithm Selection in Practice:

  +----------------------------------------------------+
  | "Make it work. Make it right. Make it fast."        |
  |                           — Kent Beck               |
  +----------------------------------------------------+

  1. Define "fast enough"
     - Web apps: Response within 200ms
     - Batch processing: Complete within SLA
     - Real-time: Within 16ms (60fps)

  2. The "optimal algorithm" is not always the best choice
     - For n=100, the difference between O(n^2) and O(n log n) is negligible
     - Readability and maintainability are also important criteria
     - Use standard library implementations when available

  3. Optimize after profiling
     - Make decisions based on measurement, not guessing
       (avoid premature optimization)
     - Identify the bottleneck before improving

  4. Be aware of trade-offs
     - Time vs. space
     - Accuracy vs. speed
     - Implementation cost vs. performance gain

  Decision flow:
  +----------------------------------------------+
  | 1. Is there a standard library implementation?|
  |    → Yes → Use it                            |
  |    → No → Go to 2                            |
  |                                              |
  | 2. Is the simplest implementation fast enough?|
  |    → Yes → Use it                            |
  |    → No → Go to 3                            |
  |                                              |
  | 3. Is the bottleneck asymptotic or constant? |
  |    → Asymptotic → Choose a better algorithm  |
  |    → Constant → Cache optimization, etc.     |
  +----------------------------------------------+
```

---

## 7. Exercises

### Exercise 1: Everyday Algorithms (Basic)
Describe the following everyday tasks as algorithms (satisfying all five properties):
1. Finding a specific book on a bookshelf
2. Sorting playing cards
3. The process of dispensing change from a vending machine

### Exercise 2: Problem-Solving Process (Intermediate)
Solve the problem "Remove all duplicate elements from an array" using three different methods, and compare the complexity of each.

### Exercise 3: Choosing a Design Strategy (Intermediate)
For each of the following problems, choose the most suitable algorithm design strategy and write pseudocode with justification:
1. Enumerate all combinations of choosing K people from N
2. Find the maximum sum of a contiguous subarray
3. Find the shortest route visiting all vertices in a graph (TSP)

### Exercise 4: Proof of Correctness (Advanced)
Prove the correctness of the binary search algorithm using loop invariants. Use the invariant: "If the target exists in the array, it exists within the range arr[left]...arr[right]."

### Exercise 5: Comparing Algorithms (Advanced)
Solve the same problem using three different design strategies (brute force, DP, greedy) and report the advantages and disadvantages of each with empirical data. Example problem: The coin problem (make a target amount with the minimum number of coins).


---

## Troubleshooting

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| Initialization error | Configuration file issues | Verify configuration file path and format |
| Timeout | Network latency / resource shortage | Adjust timeout values, add retry logic |
| Out of memory | Data volume growth | Introduce batch processing, implement pagination |
| Permission error | Insufficient access rights | Verify user permissions, review settings |
| Data inconsistency | Concurrency conflicts | Introduce locking mechanisms, transaction management |

### Debugging Procedure

1. **Check error messages**: Read the stack trace to identify the location
2. **Establish reproduction steps**: Reproduce the error with minimal code
3. **Formulate hypotheses**: List possible causes
4. **Verify incrementally**: Use log output or a debugger to test hypotheses
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
    """A decorator that logs function inputs and outputs"""
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
4. **Check concurrent connections**: Examine connection pool status

| Issue Type | Diagnostic Tools | Countermeasures |
|-----------|-----------------|-----------------|
| CPU load | cProfile, py-spy | Algorithm improvement, parallelization |
| Memory leak | tracemalloc, objgraph | Proper reference release |
| I/O bottleneck | strace, iostat | Async I/O, caching |
| DB latency | EXPLAIN, slow query log | Indexing, query optimization |

---

## Design Decision Guide

### Selection Criteria Matrix

The following summarizes the decision criteria for making technology choices.

| Criterion | Prioritize When | Acceptable to Compromise When |
|-----------|----------------|-------------------------------|
| Performance | Real-time processing, large-scale data | Admin panels, batch processing |
| Maintainability | Long-term operation, team development | Prototypes, short-term projects |
| Scalability | Growing services | Internal tools, fixed user base |
| Security | Personal data, financial data | Public data, internal use |
| Development speed | MVP, time-to-market | Quality-critical, mission-critical |

### Architecture Pattern Selection

```
+-----------------------------------------------------+
|          Architecture Selection Flow                 |
+-----------------------------------------------------+
|                                                     |
|  (1) Team size?                                     |
|    +-- Small (1-5) → Monolith                       |
|    +-- Large (10+) → Go to (2)                      |
|                                                     |
|  (2) Deployment frequency?                          |
|    +-- Once a week or less → Monolith + module split|
|    +-- Daily / multiple times → Go to (3)           |
|                                                     |
|  (3) Team independence?                             |
|    +-- High → Microservices                         |
|    +-- Medium → Modular monolith                    |
|                                                     |
+-----------------------------------------------------+
```

### Trade-off Analysis

Technical decisions always involve trade-offs. Analyze from the following perspectives:

**1. Short-term vs. Long-term Cost**
- A quick approach may become technical debt in the long run
- Conversely, over-engineering incurs high short-term costs and delays projects

**2. Consistency vs. Flexibility**
- A unified tech stack has lower learning costs
- Diverse technologies enable best-fit choices but increase operational costs

**3. Level of Abstraction**
- High abstraction offers reusability but can make debugging difficult
- Low abstraction is intuitive but prone to code duplication

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

## Practical Application Scenarios

### Scenario 1: MVP Development at a Startup

**Situation:** Need to release a product quickly with limited resources

**Approach:**
- Choose a simple architecture
- Focus on the minimum viable features
- Automated tests only for the critical path
- Introduce monitoring from the start

**Lessons Learned:**
- Do not pursue perfection (YAGNI principle)
- Get user feedback early
- Manage technical debt consciously

### Scenario 2: Modernizing a Legacy System

**Situation:** Incrementally modernize a system that has been running for 10+ years

**Approach:**
- Use the Strangler Fig pattern for incremental migration
- If existing tests are absent, create Characterization Tests first
- Use an API gateway to have old and new systems coexist
- Perform data migration in stages

| Phase | Work Content | Estimated Duration | Risk |
|-------|-------------|-------------------|------|
| 1. Investigation | Current state analysis, dependency mapping | 2-4 weeks | Low |
| 2. Foundation | CI/CD setup, test environment | 4-6 weeks | Low |
| 3. Migration start | Migrate peripheral features first | 3-6 months | Medium |
| 4. Core migration | Migrate core functionality | 6-12 months | High |
| 5. Completion | Decommission old system | 2-4 weeks | Medium |

### Scenario 3: Large-Scale Team Development

**Situation:** 50+ engineers developing the same product

**Approach:**
- Use domain-driven design to clarify boundaries
- Assign ownership per team
- Manage shared libraries using an Inner Source model
- Design API-first to minimize cross-team dependencies

```python
# API contract definition between teams
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class APIContract:
    """API contract between teams"""
    endpoint: str
    method: str
    owner_team: str
    consumers: List[str]
    sla_ms: int  # Response time SLA
    priority: Priority

    def validate_sla(self, actual_ms: int) -> bool:
        """Check SLA compliance"""
        return actual_ms <= self.sla_ms

    def to_openapi(self) -> dict:
        """Output in OpenAPI format"""
        return {
            'path': self.endpoint,
            'method': self.method,
            'x-owner': self.owner_team,
            'x-consumers': self.consumers,
            'x-sla-ms': self.sla_ms
        }

# Usage example
contracts = [
    APIContract(
        endpoint="/api/v1/users",
        method="GET",
        owner_team="user-team",
        consumers=["order-team", "notification-team"],
        sla_ms=200,
        priority=Priority.HIGH
    ),
    APIContract(
        endpoint="/api/v1/orders",
        method="POST",
        owner_team="order-team",
        consumers=["payment-team", "inventory-team"],
        sla_ms=500,
        priority=Priority.CRITICAL
    )
]
```

### Scenario 4: Performance-Critical Systems

**Situation:** A system requiring millisecond-level response times

**Optimization Points:**
1. Caching strategy (L1: in-memory, L2: Redis, L3: CDN)
2. Leverage asynchronous processing
3. Connection pooling
4. Query optimization and index design

| Optimization Technique | Effect | Implementation Cost | Applicable Scenario |
|----------------------|--------|-------------------|-------------------|
| In-memory cache | High | Low | Frequently accessed data |
| CDN | High | Low | Static content |
| Async processing | Medium | Medium | I/O-heavy processing |
| DB optimization | High | High | Slow queries |
| Code optimization | Low-Medium | High | CPU-bound cases |

---

## Leveraging in Team Development

### Code Review Checklist

Key points to check in code reviews related to this topic:

- [ ] Naming conventions are consistent
- [ ] Error handling is appropriate
- [ ] Test coverage is sufficient
- [ ] There is no performance impact
- [ ] There are no security issues
- [ ] Documentation has been updated

### Best Practices for Knowledge Sharing

| Method | Frequency | Target | Effect |
|--------|-----------|--------|--------|
| Pair programming | As needed | Complex tasks | Immediate feedback |
| Tech talks | Weekly | Entire team | Horizontal knowledge spread |
| ADR (Decision Records) | Per decision | Future team members | Decision transparency |
| Retrospectives | Biweekly | Entire team | Continuous improvement |
| Mob programming | Monthly | Important designs | Consensus building |

### Managing Technical Debt

```
Priority Matrix:

        High Impact
          |
    +-----+-----+
    | Plan | Act  |
    | for  | imme-|
    | later| diately|
    +-----+-----+
    | Record| Next |
    | only | Sprint|
    |      |      |
    +-----+-----+
          |
        Low Impact
    Low Frequency  High Frequency
```

---

## Security Considerations

### Common Vulnerabilities and Countermeasures

| Vulnerability | Risk Level | Countermeasure | Detection Method |
|--------------|-----------|----------------|-----------------|
| Injection attacks | High | Input validation, parameterized queries | SAST/DAST |
| Authentication flaws | High | MFA, session management hardening | Penetration testing |
| Sensitive data exposure | High | Encryption, access control | Security audit |
| Misconfiguration | Medium | Security headers, principle of least privilege | Configuration scan |
| Insufficient logging | Medium | Structured logging, audit trails | Log analysis |

### Secure Coding Best Practices

```python
# Secure coding examples
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
        """Sanitize input values"""
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
- [ ] Sensitive information is not output in logs
- [ ] HTTPS is enforced
- [ ] CORS policy is properly configured
- [ ] Dependency vulnerability scanning has been performed
- [ ] Error messages do not contain internal information

---

## Migration Guide

### Notes for Version Upgrades

| Version | Major Changes | Migration Work | Impact Scope |
|---------|--------------|---------------|-------------|
| v1.x → v2.x | API redesign | Endpoint changes | All clients |
| v2.x → v3.x | Authentication method change | Token format update | Auth-related |
| v3.x → v4.x | Data model change | Run migration scripts | DB-related |

### Incremental Migration Procedure

```python
# Migration script template
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Callable

logger = logging.getLogger(__name__)

class MigrationRunner:
    """Incremental migration execution engine"""

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
        """Execute migrations (upgrade)"""
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
2. **Test environment verification**: Pre-verify in an environment equivalent to production
3. **Staged rollout**: Deploy incrementally using canary releases
4. **Enhanced monitoring**: Shorten monitoring intervals during migration
5. **Clear decision criteria**: Define rollback criteria in advance
---

## FAQ

### Q1: What is the relationship between algorithms and data structures?
**A**: They are closely related. Choosing the right data structure can dramatically change algorithm efficiency. For example: Search → Array (O(n)) vs. Hash table (O(1)). "Algorithms + Data Structures = Programs" (Wirth, 1976). Data structures are "how information is organized," and algorithms are "procedures for processing information" — they are like two wheels of the same cart.

### Q2: How does studying algorithms benefit practical work?
**A**: Directly, it helps with performance optimization, system design, and coding interviews. Indirectly, it develops problem decomposition skills, logical thinking, and a "sense of computational complexity," improving the quality of everyday coding. Specific examples: B-Tree knowledge is essential for database index design, LRU algorithm knowledge for caching strategies, and string matching knowledge for search feature implementation.

### Q3: Do I need to memorize all algorithms?
**A**: No. What matters is developing an understanding of "design patterns" and a "sense of computational complexity." You can look up specific implementations when needed. However, you should understand basic sorting, search, and graph algorithms. The important skill is not memorization but the ability to judge "which pattern might apply to this problem."

### Q4: What should I do when I cannot solve an algorithm problem?
**A**: There are several approaches. (1) Try solving a small example by hand first. (2) Simplify the problem and solve that (relax constraints). (3) Consider whether it can be reduced to a similar problem. (4) Think in reverse (work backward from the output to the input). (5) Estimate the target complexity and narrow down applicable algorithms. (6) Take a break and do something else (let the brain process subconsciously).

### Q5: Are competitive programming algorithms different from practical algorithms?
**A**: They are fundamentally the same, but the emphasis differs. Competitive programming focuses on complexity optimization and implementation speed. In practice, readability, maintainability, and testability are also important. Additionally, leveraging existing libraries, team development, and incremental improvement are realistic approaches in practice. That said, the algorithmic thinking honed through competitive programming is highly valuable in real-world work.

---

## Summary

| Concept | Key Points |
|---------|-----------|
| Definition | A computational procedure that solves a problem in finite, well-defined steps |
| Five Properties | Input, output, definiteness, finiteness, effectiveness |
| Design Strategies | Brute force, divide & conquer, DP, greedy, backtracking, binary search, graph search |
| Correctness | Proven via loop invariants, mathematical induction, proof by contradiction |
| Thought Process | Understand → hand-calculate → design → implement → test |
| Classification | P, NP, NP-complete: classification of problem difficulty |
| Practice | Optimize in order: correctness → readability → performance |

---

## Recommended Next Guides

---

## References
1. Cormen, T. H. et al. "Introduction to Algorithms (CLRS)." 4th Edition, MIT Press, 2022.
2. Knuth, D. E. "The Art of Computer Programming." Addison-Wesley, 1968-2022.
3. Skiena, S. S. "The Algorithm Design Manual." 3rd Edition, Springer, 2020.
4. Sedgewick, R. & Wayne, K. "Algorithms." 4th Edition, Addison-Wesley, 2011.
5. Kleinberg, J. & Tardos, E. "Algorithm Design." Pearson, 2005.
6. Garey, M. R. & Johnson, D. S. "Computers and Intractability: A Guide to the Theory of NP-Completeness." W.H. Freeman, 1979.
