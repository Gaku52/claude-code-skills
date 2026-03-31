# Problem-Solving Methods

> Master systematic approaches to solving algorithmic problems through pattern recognition, constraint analysis, and iterative refinement

## What You Will Learn in This Chapter

1. **Pattern recognition** to classify problems into known algorithm categories
2. **Constraint analysis** to derive acceptable complexity bounds and select appropriate algorithms
3. **Iterative refinement** to efficiently progress from brute-force solutions to optimal solutions
4. **Stress testing** and brute-force comparison for robust debugging
5. **Mastery of 12+ canonical patterns** to build a repertoire for tackling unfamiliar problems


## Prerequisites

Before reading this guide, the following knowledge will help deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts

---

## 1. Overall Problem-Solving Framework

Problem-solving ability is more than just algorithmic knowledge. It is the entire thought process of reading a problem, grasping its structure, mapping it to known techniques, and incrementally constructing a solution. Extending George Polya's 4-step framework from *How to Solve It* for algorithmic problems yields the following 5 steps.

```
┌──────────────────────────────────────────────────────────────────────┐
│                    5 Steps to Problem Solving                        │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Step 1: Understand the Problem                                      │
│    ├─ Clarify the input/output format                                │
│    ├─ Check constraints (range of n, value ranges, time limit)       │
│    ├─ Identify edge cases (empty input, minimum values, max values)  │
│    └─ Rephrase the problem in your own words                         │
│                                                                      │
│  Step 2: Work Through Concrete Examples (Explore)                    │
│    ├─ Compute by hand with small inputs (n=3-5)                      │
│    ├─ Discover patterns through multiple examples                    │
│    ├─ Draw diagrams and tables for visualization                     │
│    └─ Check for counterexamples                                      │
│                                                                      │
│  Step 3: Analyze Constraints                                         │
│    ├─ Derive the acceptable O(?) from the range of n                 │
│    ├─ Also check space complexity constraints (ML = memory limit)    │
│    ├─ Consider constant-factor overhead of the language used         │
│    └─ Identify special constraints (coordinate compression, MOD, etc.)│
│                                                                      │
│  Step 4: Select and Design an Algorithm (Design)                     │
│    ├─ Map to known techniques via pattern recognition                │
│    ├─ Compare multiple candidates by complexity                      │
│    ├─ Select the required data structures                            │
│    └─ Validate the design with pseudocode                            │
│                                                                      │
│  Step 5: Implement, Verify, and Improve                              │
│    ├─ First implement a brute-force solution and verify correctness  │
│    ├─ Implement the optimal solution                                 │
│    ├─ Compare with brute-force via stress testing                    │
│    └─ Exhaustively test edge cases                                   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 1.1 Practicing Step 1: Understanding the Problem

Proceeding to implementation without accurately understanding the problem is one of the most common mistakes. When reading a problem statement, always verify the following points.

```python
# Problem understanding checklist (always verify before implementing)

class ProblemUnderstanding:
    """Framework for understanding a problem"""

    def __init__(self, problem_text: str):
        self.problem_text = problem_text
        self.input_format = None    # Input format
        self.output_format = None   # Output format
        self.constraints = {}       # Constraints
        self.edge_cases = []        # Edge cases
        self.examples = []          # Concrete examples

    def parse_constraints(self):
        """Extract and structure the constraints"""
        # Typical constraint patterns
        constraint_patterns = {
            'n_range': '1 <= n <= ?',
            'value_range': 'Value range',
            'time_limit': 'Execution time limit (typically 2 seconds)',
            'memory_limit': 'Memory limit (typically 256MB)',
            'special': 'MOD, coordinate range, character set, etc.',
        }
        return constraint_patterns

    def identify_edge_cases(self):
        """Systematically enumerate edge cases"""
        common_edges = [
            "n = 0 (empty input)",
            "n = 1 (minimum input)",
            "n = max constraint (TLE/MLE boundary)",
            "All elements have the same value",
            "Already sorted / completely reversed",
            "Contains negative values and zeros",
            "No valid answer exists",
            "Multiple valid answers exist",
        ]
        return common_edges

    def rephrase(self) -> str:
        """Rephrase the problem in your own words"""
        # "Find ..." → "Minimize/Maximize ..." → "Count those satisfying condition ..."
        # Verify this rephrasing is correct using sample cases
        pass
```

### 1.2 Practicing Step 2: Working Through Concrete Examples

Before thinking about algorithms, compute at least 3 concrete examples by hand. This reveals the essential structure of the problem.

```
Guide to creating concrete examples:

   Example 1: Trace through the sample input as given
              → Understand the exact meaning of the problem

   Example 2: Create your own small inputs (n=3-5)
              → Discover patterns through hand computation

   Example 3: Try edge cases
              → Empty input, minimum input, special values

   Example 4: Try larger inputs (n=10-20)
              → Verify the generalization of the algorithm

   ┌─────────────────────────────────────────────────┐
   │  Tips for Hand Computation                       │
   │                                                   │
   │  - Create tables to trace state transitions (→DP)│
   │  - Draw diagrams for spatial relationships        │
   │    (geometry/graphs)                              │
   │  - Note intermediate results (greedy validation)  │
   │  - Articulate "why this answer is optimal"        │
   └─────────────────────────────────────────────────┘
```

### 1.3 Connecting Steps 3-5

Steps 3-5 are not linear but form a feedback loop.

```
    ┌──────────────┐
    │ Step 3: Analyze│◄─────────────────────┐
    └──────┬───────┘                       │
           │ Complexity estimate            │ Complexity doesn't fit
           ▼                               │ → Try a different algorithm
    ┌──────────────┐                       │
    │ Step 4: Design│───────────────────────┘
    └──────┬───────┘
           │ Pseudocode
           ▼
    ┌──────────────┐
    │ Step 5: Impl  │───┐
    └──────┬───────┘   │ WA (Wrong Answer)
           │            │ → Compare with brute force
           ▼            │
    ┌──────────────┐   │
    │  Verified     │◄──┘
    └──────────────┘
```

---

## 2. Constraint Analysis: Deriving Complexity from n

### 2.1 Fundamental Principle

Constraint analysis is the starting point for algorithm selection. For many problems, the moment you see the constraints, the required complexity class becomes clear. This is expressed as the important principle that "the constraints tell you the answer."

```
Operations processable in 1 second ≈ 10^8 to 10^9 (C++ baseline)
For Python: approximately 10^6 to 10^7 (30-100x slower than C++)

Range of n → Acceptable complexity:

  n <= 8       → O(n! * n)    All permutations + processing per permutation
  n <= 10      → O(n!)        Full permutation search, backtracking
  n <= 20      → O(2^n * n)   Bitmask DP, subset enumeration
  n <= 25      → O(2^(n/2))   Meet in the Middle
  n <= 50      → O(n^4)       Quadruple loop (rare)
  n <= 300     → O(n^3)       Floyd-Warshall, interval DP
  n <= 3,000   → O(n^2 log n) Some optimized double loops
  n <= 5,000   → O(n^2)       DP (2D), all-pairs comparison
  n <= 100,000 → O(n*sqrt(n)) Square root decomposition, Mo's algorithm
  n <= 200,000 → O(n log n)   Sorting, segment tree, binary search
  n <= 10^6    → O(n)         Linear scan, two pointers, BFS
  n <= 10^7    → O(n)         Linear (watch constant factors)
  n <= 10^9    → O(sqrt(n)) or O(log n)  Prime factorization, binary search
  n <= 10^18   → O(log n) or O(1)        Matrix exponentiation, math formulas
```

### 2.2 Constraint Analysis in Practice

```python
# ============================================================
# Example: Find a pair in an array that sums to target
# Constraints: 1 <= n <= 10^5, -10^9 <= arr[i] <= 10^9
# ============================================================

# Constraint analysis:
# n = 10^5 → O(n^2) = 10^10 → TLE (Time Limit Exceeded)
# → Need O(n log n) or O(n)
# → Candidates: sort + binary search, sort + two pointers, hash map

# === Solution 1: O(n^2) brute force (TLE but useful for correctness verification) ===
def two_sum_brute(arr: list[int], target: int) -> tuple[int, int] | None:
    """Brute force: enumerate all pairs O(n^2)"""
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] + arr[j] == target:
                return (i, j)
    return None

# === Solution 2: O(n log n) sort + two pointers ===
def two_sum_sort(arr: list[int], target: int) -> tuple[int, int] | None:
    """Sort + two pointers O(n log n)"""
    indexed = sorted(enumerate(arr), key=lambda x: x[1])
    left, right = 0, len(arr) - 1

    while left < right:
        current_sum = indexed[left][1] + indexed[right][1]
        if current_sum == target:
            return (indexed[left][0], indexed[right][0])
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return None

# === Solution 3: O(n) hash map (optimal solution) ===
def two_sum_hash(arr: list[int], target: int) -> tuple[int, int] | None:
    """Hash map O(n)"""
    seen: dict[int, int] = {}
    for i, num in enumerate(arr):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return None

# === Stress test comparing all 3 solutions ===
import random

def stress_test_two_sum(n_tests: int = 1000, max_n: int = 100):
    """Random tests comparing the three solutions"""
    for test_id in range(n_tests):
        n = random.randint(2, max_n)
        arr = [random.randint(-100, 100) for _ in range(n)]
        target = random.randint(-200, 200)

        result_brute = two_sum_brute(arr, target)
        result_sort = two_sum_sort(arr, target)
        result_hash = two_sum_hash(arr, target)

        # Check existence agreement
        has_brute = result_brute is not None
        has_sort = result_sort is not None
        has_hash = result_hash is not None

        assert has_brute == has_sort == has_hash, (
            f"Test {test_id}: Mismatch! arr={arr}, target={target}"
        )

        # If a solution exists, verify the pair is correct
        if has_brute:
            i, j = result_hash
            assert arr[i] + arr[j] == target, (
                f"Test {test_id}: Hash solution is incorrect"
            )
    print(f"All {n_tests} tests passed")

# stress_test_two_sum()  # Run to verify
```

### 2.3 Space Complexity Constraints

Time complexity is not the only concern; space complexity must also be considered.

```
Memory limit guidelines (for 256MB):

  int array:       approximately 6.4 * 10^7 elements
  long long array: approximately 3.2 * 10^7 elements
  2D array:        n * m <= approximately 6.4 * 10^7

  For Python (less memory efficient):
    list[int]:   approximately 28 bytes per element → approximately 9 * 10^6 elements
    numpy array: 8 bytes per element → approximately 3.2 * 10^7 elements

  Common patterns:
    2D DP with n = 10^5 → O(n^2) = 10^10 → MLE
    → Reduce to O(n) by keeping "only the current row" (DP scrolling optimization)
```

### 2.4 Language-Specific Constant Factors and Countermeasures

```
┌─────────────┬──────────────┬────────────────────────────────────┐
│ Language    │ Speed Factor │ Countermeasure                      │
│             │ (vs C++)     │                                     │
├─────────────┼──────────────┼────────────────────────────────────┤
│ C++         │ 1x           │ Baseline                            │
│ Java        │ 2-3x         │ Use FastReader, avoid Scanner       │
│ Python      │ 30-100x      │ Use PyPy, list comprehensions,      │
│             │              │ sys.stdin                            │
│ Go          │ 1-2x         │ Use bufio.Scanner                   │
│ Rust        │ 0.8-1.2x     │ Comparable to C++                   │
│ JavaScript  │ 3-10x        │ Use TypedArray                      │
└─────────────┴──────────────┴────────────────────────────────────┘

Python optimization techniques:

  1. Speed up I/O
     import sys
     input = sys.stdin.readline  # 10x+ faster
     print = sys.stdout.write    # Effective for bulk output

  2. List comprehensions > for loops
     # Slow: for i in range(n): result.append(i*i)
     # Fast: result = [i*i for i in range(n)]

  3. Raise recursion limit
     sys.setrecursionlimit(300000)  # Default is 1000

  4. Use collections
     from collections import deque, defaultdict, Counter

  5. Local variables are faster than global variables
     def solve():
         # Write all processing here (local variable lookups are faster)
         pass
     solve()
```

---

## 3. Pattern Recognition Map

### 3.1 Keyword-to-Algorithm Mapping Table

Infer the appropriate algorithm from keywords in the problem statement. This is an experience-based heuristic, not an absolute rule, but it has a high success rate.

```
Problem keyword → Algorithm candidates:

┌──────────────────────────────────────────────────────────────────────┐
│ Keyword                        │ Candidate Algorithms                │
├──────────────────────────────────────────────────────────────────────┤
│ "shortest distance" "min steps"│ BFS (unweighted), Dijkstra (weighted)│
│ "shortest path" "minimum cost" │ Dijkstra, Bellman-Ford, Floyd-Warshall│
│ "find max/min"                 │ DP, greedy, binary search on answer │
│ "count the number of ..."      │ DP, combinatorics, inclusion-exclusion│
│ "all combinations"             │ Backtracking, bitmask DP            │
│ "subsequence" "substring"      │ DP (LIS, LCS), two pointers         │
│ "connected?" "reachable?"      │ Union-Find, BFS/DFS                 │
│ "interval" "range"             │ Segment tree, BIT, two pointers     │
│ "string matching"              │ KMP, Z-algorithm, Rabin-Karp        │
│ "lexicographically smallest"   │ Greedy, stack                       │
│ "bipartite graph"              │ Bipartite matching, 2-coloring      │
│ "assignment" "flow"            │ Network flow, bipartite matching    │
│ "MOD 10^9+7"                   │ DP + fast exponentiation + modular  │
│                                │ inverse                              │
│ "tree" "rooted tree"           │ DFS, Euler tour, LCA                │
│ "cycle"                        │ DFS (back edge), Union-Find         │
│ "coordinates" "points" "dist"  │ Geometry, sorting, sweep line       │
│ "game" "first/second player"   │ Grundy numbers, DP, game theory    │
│ "expected value" "probability" │ Probability DP, matrix exp          │
│ "palindrome"                   │ Manacher, DP, hashing               │
│ "inversion count" "crossing"   │ BIT, merge sort                     │
│ "XOR"                          │ Trie (bit sequence), linear basis   │
│ "GCD" "LCM" "prime"           │ Euclidean algorithm, Sieve of       │
│                                │ Eratosthenes                         │
│ "parentheses" "nesting"        │ Stack                               │
│ "k-th" "median"                │ Binary search, order statistics     │
│ "subset sum" "knapsack"        │ DP, Meet in the Middle              │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.2 Classification by Problem Structure

Algorithms can also be inferred from structural features of the problem, not just keywords.

```python
def classify_problem(problem_features: dict) -> list[str]:
    """
    Infer algorithm candidates from problem features

    problem_features example:
    {
        'optimization': True,          # Is it an optimization problem?
        'counting': False,             # Is it a counting problem?
        'decision': False,             # Is it a decision problem?
        'graph': True,                 # Does it involve graphs?
        'has_weights': True,           # Are there weights/costs?
        'constraint_n': 100000,        # Constraint on n
        'overlapping_subproblems': True,  # Overlapping subproblems
        'greedy_choice': False,        # Greedy choice property
        'monotonic': True,             # Is there monotonicity?
    }
    """
    candidates = []
    n = problem_features.get('constraint_n', 0)

    # Optimization problem branching
    if problem_features.get('optimization'):
        if problem_features.get('greedy_choice'):
            candidates.append('Greedy')
        if problem_features.get('overlapping_subproblems'):
            candidates.append('Dynamic programming')
        if problem_features.get('monotonic'):
            candidates.append('Binary search on the answer')
        if n <= 20:
            candidates.append('Bitmask DP / exhaustive search')

    # Graph problem branching
    if problem_features.get('graph'):
        if problem_features.get('has_weights'):
            candidates.append('Dijkstra / Bellman-Ford')
        else:
            candidates.append('BFS / DFS')

    # Counting problem
    if problem_features.get('counting'):
        candidates.append('DP')
        candidates.append('Combinatorics')

    # Decision problem
    if problem_features.get('decision'):
        if problem_features.get('monotonic'):
            candidates.append('Binary search')

    return candidates
```

### 3.3 Data Structure Selection Guide

```
Required operation → Appropriate data structure:

┌──────────────────────────────────────────────────────────────────────┐
│ Required Operation                │ Data Structure                    │
├──────────────────────────────────────────────────────────────────────┤
│ Add/remove at front/back          │ deque (double-ended queue)        │
│ Extract min/max                   │ heapq (priority queue)            │
│ Insert/search/delete in O(1)      │ dict / set (hash)                 │
│ Insert/search/delete in O(log n)  │ SortedList (balanced BST)         │
│ Range sum / max with updates      │ Segment tree / BIT                │
│ Set merge / identity check        │ Union-Find (disjoint set)         │
│ String prefix search              │ Trie                              │
│ LIFO (last in, first out)         │ stack (list)                      │
│ FIFO (first in, first out)        │ deque                             │
│ k-th element in sorted order      │ BIT / balanced BST                │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 4. Canonical Pattern 1: Two Pointers / Sliding Window

### 4.1 Basic Concept

The two pointers technique efficiently solves problems involving contiguous subarrays. It uses two pointers (left and right) to manage a window that slides while maintaining a condition.

```
Fixed-length window:
  ┌───┬───┬───┬───┬───┬───┬───┬───┐
  │ 1 │ 4 │ 2 │10 │23 │ 3 │ 1 │ 0 │  array
  └───┴───┴───┴───┴───┴───┴───┴───┘
  ├─── k=4 ────┤                      window sum = 17
      ├─── k=4 ────┤                  window sum = 39 ← max
          ├─── k=4 ────┤              window sum = 38
              ├─── k=4 ────┤          window sum = 37
                  ├─── k=4 ────┤      window sum = 27

  Add new element, remove old element → O(1) update

Variable-length window:
  ┌───┬───┬───┬───┬───┬───┐
  │ 2 │ 3 │ 1 │ 2 │ 4 │ 3 │  array, target=7
  └───┴───┴───┴───┴───┴───┘
   L       R                    sum=6 < 7 → advance R
   L           R                sum=8 >= 7 → record length 4, advance L
       L       R                sum=6 < 7 → advance R
       L           R            sum=10 >= 7 → record length 3, advance L
           L       R            sum=7 >= 7 → record length 2 ← shortest
```

### 4.2 Implementation Patterns

```python
# ============================================================
# Pattern 1: Fixed-length window
# ============================================================
def max_subarray_sum_k(arr: list[int], k: int) -> int:
    """Maximum sum of a subarray of length k"""
    if len(arr) < k:
        return 0

    # First window
    window_sum = sum(arr[:k])
    max_sum = window_sum

    # Slide the window
    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i - k]  # O(1) update
        max_sum = max(max_sum, window_sum)

    return max_sum


# ============================================================
# Pattern 2: Variable-length window (shortest interval satisfying condition)
# ============================================================
def min_subarray_len(arr: list[int], target: int) -> int:
    """Length of the shortest contiguous subarray with sum >= target"""
    n = len(arr)
    min_len = float('inf')
    left = 0
    current_sum = 0

    for right in range(n):
        current_sum += arr[right]

        # Advance left as long as the condition is satisfied (shrink the interval)
        while current_sum >= target:
            min_len = min(min_len, right - left + 1)
            current_sum -= arr[left]
            left += 1

    return min_len if min_len != float('inf') else 0


# ============================================================
# Pattern 3: Variable-length window (longest interval satisfying condition)
# ============================================================
def longest_substring_k_distinct(s: str, k: int) -> int:
    """Longest substring with at most k distinct characters"""
    from collections import defaultdict

    char_count: dict[str, int] = defaultdict(int)
    left = 0
    max_len = 0

    for right in range(len(s)):
        char_count[s[right]] += 1

        # Advance left when the condition is violated
        while len(char_count) > k:
            char_count[s[left]] -= 1
            if char_count[s[left]] == 0:
                del char_count[s[left]]
            left += 1

        max_len = max(max_len, right - left + 1)

    return max_len


# ============================================================
# Pattern 4: Two pointers on two arrays (merge operation)
# ============================================================
def merge_sorted_arrays(arr1: list[int], arr2: list[int]) -> list[int]:
    """Merge two sorted arrays O(n + m)"""
    result = []
    i, j = 0, 0

    while i < len(arr1) and j < len(arr2):
        if arr1[i] <= arr2[j]:
            result.append(arr1[i])
            i += 1
        else:
            result.append(arr2[j])
            j += 1

    # Append remaining elements
    result.extend(arr1[i:])
    result.extend(arr2[j:])
    return result


# ============================================================
# Tests
# ============================================================
print(max_subarray_sum_k([1, 4, 2, 10, 23, 3, 1, 0, 20], 4))  # 39
print(min_subarray_len([2, 3, 1, 2, 4, 3], 7))                  # 2
print(longest_substring_k_distinct("eceba", 2))                  # 3 ("ece")
print(merge_sorted_arrays([1, 3, 5], [2, 4, 6]))                # [1,2,3,4,5,6]
```

### 4.3 Applicability Conditions for Two Pointers

Understanding the conditions under which two pointers can be applied is crucial.

```
Conditions for applicability:

  1. Expanding the interval (advancing right) monotonically increases the "cost"
  2. Shrinking the interval (advancing left) monotonically decreases the "cost"
  3. In other words, the predicate function for interval [l, r] has monotonicity

  Example: interval sum >= target
    → Expanding the interval increases the sum (monotonically increasing)
    → Shrinking the interval decreases the sum (monotonically decreasing)
    → Two pointers is applicable

  Counterexample: max - min within interval <= k
    → Expanding the interval doesn't necessarily increase the difference
    → Simple two pointers is not applicable
    → However, with extensions like sortedcontainers, it becomes possible

Typical applicable problems:
  - Shortest/longest interval with sum >= or <= target
  - Longest interval with at most k distinct elements
  - Counting intervals satisfying a condition
  - Merging two sorted arrays
```

### 4.4 Complexity Analysis of Two Pointers

```
Why is it O(n)?

  Both left and right advance at most n times each.
  Each element is accessed by left once and by right once.
  → Total operations at most 2n → O(n)

  ┌──────────────────────────────────────────┐
  │  right: 0 → 1 → 2 → ... → n-1          │
  │  left:  0 → 0 → 1 → ... → n-1          │
  │                                          │
  │  Moves of right: n                       │
  │  Moves of left:  at most n               │
  │  Total: at most 2n = O(n)               │
  └──────────────────────────────────────────┘
```

---

## 5. Canonical Pattern 2: Binary Search on the Answer

### 5.1 The "Binary Search on the Answer" Concept

Ordinary binary search finds a value in a sorted array, but here the paradigm is "assume an answer value and check if it is feasible." This is a powerful technique that converts optimization problems into decision problems.

```
Ordinary binary search:
  "Does value x exist in the array?" → O(log n)

Binary search on the answer:
  "Is the answer achievable with x or less?" → O(log(answer range) * check complexity)

  Conditions for applicability:
    1. The answer has monotonicity (larger answers are easier to achieve, etc.)
    2. "Is the answer feasible at x?" can be efficiently checked

  ┌──────────────────────────────────────────────────────┐
  │  Answer space: [lo, hi]                              │
  │                                                      │
  │  lo ──────── mid ──────── hi                         │
  │  ← infeasible →│← feasible →→→→→→→→→│               │
  │                 ▲                                     │
  │           minimum answer                              │
  │                                                      │
  │  can_achieve(mid) == True  → hi = mid (move left)    │
  │  can_achieve(mid) == False → lo = mid + 1 (move right)│
  └──────────────────────────────────────────────────────┘
```

### 5.2 Implementation Patterns

```python
# ============================================================
# Example 1: Split array into k parts, minimize the max partition sum
# (Painter's Partition Problem / Split Array Largest Sum)
# ============================================================

def min_max_partition(arr: list[int], k: int) -> int:
    """Minimize the maximum partition sum when splitting array into k parts"""

    def can_partition(max_sum: int) -> bool:
        """Can we partition into k parts with each sum <= max_sum?"""
        count = 1       # Current number of partitions
        current_sum = 0
        for num in arr:
            if current_sum + num > max_sum:
                count += 1
                current_sum = num
                if count > k:
                    return False
            else:
                current_sum += num
        return True

    # Binary search on the answer
    lo = max(arr)      # Minimum: a partition with just the largest element
    hi = sum(arr)      # Maximum: entire array is one partition
    result = hi

    while lo <= hi:
        mid = (lo + hi) // 2
        if can_partition(mid):
            result = mid
            hi = mid - 1   # Search for a smaller answer
        else:
            lo = mid + 1   # Increase the answer

    return result


# ============================================================
# Example 2: Cut k logs of length L from n trees. Maximize L.
# ============================================================

def max_log_length(trees: list[int], k: int) -> int:
    """Maximum length L such that k or more logs of length L can be cut from n trees"""

    def count_logs(length: int) -> int:
        """Number of logs obtainable at the given length"""
        return sum(t // length for t in trees)

    lo, hi = 1, max(trees)
    result = 0

    while lo <= hi:
        mid = (lo + hi) // 2
        if count_logs(mid) >= k:
            result = mid
            lo = mid + 1    # Search for a larger L
        else:
            hi = mid - 1

    return result


# ============================================================
# Example 3: Binary search on real values (with precision)
# ============================================================

def sqrt_binary_search(x: float, eps: float = 1e-9) -> float:
    """Find the square root of x using binary search"""
    lo, hi = 0.0, max(1.0, x)

    while hi - lo > eps:
        mid = (lo + hi) / 2
        if mid * mid <= x:
            lo = mid
        else:
            hi = mid

    return lo


# Tests
print(min_max_partition([7, 2, 5, 10, 8], 2))   # 18 ([7,2,5] and [10,8])
print(max_log_length([10, 24, 15], 7))           # 6
print(f"{sqrt_binary_search(2):.10f}")           # 1.4142135624
```

### 5.3 Common Binary Search Bugs and Countermeasures

```
┌──────────────────────────────────────────────────────────────────────┐
│  Top 5 Binary Search Pitfalls                                        │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Infinite loop                                                    │
│     Cause: lo and hi updates don't converge                         │
│     Fix: Always narrow the range with lo = mid + 1 / hi = mid - 1  │
│                                                                      │
│  2. Off-by-one error                                                 │
│     Cause: Confusing lo <= hi vs lo < hi                            │
│     Fix: Use a consistent template for condition and return         │
│                                                                      │
│  3. Integer overflow                                                 │
│     Cause: mid = (lo + hi) / 2 overflows when lo + hi is too large │
│     Fix: mid = lo + (hi - lo) // 2                                  │
│                                                                      │
│  4. Wrong monotonicity direction                                     │
│     Cause: Using lo = mid + 1 for minimization (reversed)          │
│     Fix: Write out a table of True/False vs search direction        │
│                                                                      │
│  5. Insufficient precision for real-valued binary search             │
│     Cause: eps too large / too few iterations                       │
│     Fix: Use a fixed number of iterations (e.g., 100) for safety   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

Safe binary search template (integer version):

  # Finding the minimum (smallest answer satisfying the condition)
  lo, hi = lower_bound, upper_bound
  while lo < hi:
      mid = (lo + hi) // 2
      if is_feasible(mid):
          hi = mid        # mid is a candidate, search further left
      else:
          lo = mid + 1    # mid is infeasible, move right
  answer = lo             # lo == hi is the answer

  # Finding the maximum (largest answer satisfying the condition)
  lo, hi = lower_bound, upper_bound
  while lo < hi:
      mid = (lo + hi + 1) // 2  # Round up (important!)
      if is_feasible(mid):
          lo = mid        # mid is a candidate, search further right
      else:
          hi = mid - 1    # mid is infeasible, move left
  answer = lo             # lo == hi is the answer
```

---

## 6. Canonical Pattern 3: Prefix Sums

### 6.1 Basics of Prefix Sums

Prefix sums are a preprocessing technique that enables O(1) range sum queries. Preprocessing takes O(n), and each query is answered in O(1).

```
Original array:   [3, 1, 4, 1, 5, 9, 2, 6]
Prefix sum array: [0, 3, 4, 8, 9, 14, 23, 25, 31]
                   ↑  prefix[0] = 0 (sentinel)

Sum of range [2, 5] = prefix[6] - prefix[2]
                     = 23 - 4 = 19

  Verification: arr[2]+arr[3]+arr[4]+arr[5] = 4+1+5+9 = 19  ✓

  ┌───┬───┬───┬───┬───┬───┬───┬───┐
  │ 3 │ 1 │ 4 │ 1 │ 5 │ 9 │ 2 │ 6 │  original array
  └───┴───┴───┴───┴───┴───┴───┴───┘
  idx: 0   1   2   3   4   5   6   7

  prefix[i+1] = prefix[i] + arr[i]
  sum(arr[l..r]) = prefix[r+1] - prefix[l]
```

### 6.2 1D and 2D Implementations

```python
# ============================================================
# 1D Prefix Sum
# ============================================================
def prefix_sum_queries(arr: list[int], queries: list[tuple[int, int]]) -> list[int]:
    """Answer multiple range sum queries in O(1)"""
    n = len(arr)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + arr[i]

    results = []
    for l, r in queries:
        results.append(prefix[r + 1] - prefix[l])

    return results


# ============================================================
# 2D Prefix Sum
# ============================================================
def build_prefix_sum_2d(matrix: list[list[int]]) -> list[list[int]]:
    """Build 2D prefix sum O(rows * cols)"""
    if not matrix or not matrix[0]:

    rows, cols = len(matrix), len(matrix[0])
    prefix = [[0] * (cols + 1) for _ in range(rows + 1)]

    for i in range(rows):
        for j in range(cols):
            prefix[i + 1][j + 1] = (
                matrix[i][j]
                + prefix[i][j + 1]
                + prefix[i + 1][j]
                - prefix[i][j]         # Inclusion-exclusion
            )

    return prefix


def query_2d(prefix: list[list[int]], r1: int, c1: int, r2: int, c2: int) -> int:
    """Rectangle sum for (r1,c1)-(r2,c2) in O(1)"""
    return (
        prefix[r2 + 1][c2 + 1]
        - prefix[r1][c2 + 1]
        - prefix[r2 + 1][c1]
        + prefix[r1][c1]              # Inclusion-exclusion
    )


# ============================================================
# Prefix sum variant: imos method (difference array)
# ============================================================
def range_add_queries(n: int, queries: list[tuple[int, int, int]]) -> list[int]:
    """
    imos method: Efficiently process range addition queries
    queries: [(l, r, val), ...] → add val to arr[l..r]
    Returns the final array after all queries

    Complexity: O(n + Q)  (Q = number of queries)
    """
    diff = [0] * (n + 1)   # Difference array

    for l, r, val in queries:
        diff[l] += val
        if r + 1 < n:
            diff[r + 1] -= val

    # Prefix sum of the difference array = final array
    result = [0] * n
    result[0] = diff[0]
    for i in range(1, n):
        result[i] = result[i - 1] + diff[i]

    return result


# Tests
arr = [3, 1, 4, 1, 5, 9, 2, 6]
print(prefix_sum_queries(arr, [(0, 3), (2, 5), (0, 7)]))
# [9, 19, 31]

matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]
prefix = build_prefix_sum_2d(matrix)
print(query_2d(prefix, 0, 0, 1, 1))  # 1+2+4+5 = 12
print(query_2d(prefix, 1, 1, 2, 2))  # 5+6+8+9 = 28

print(range_add_queries(5, [(1, 3, 2), (0, 2, 1), (2, 4, 3)]))
# [1, 3, 6, 5, 3]
```

### 6.3 Prefix Sum Variants

```
┌────────────────────────┬───────────────────────────────────────────┐
│ Variant                │ Use Case                                  │
├────────────────────────┼───────────────────────────────────────────┤
│ Standard prefix sum    │ Range sum queries in O(1)                 │
│ 2D prefix sum          │ Rectangle sum queries in O(1)             │
│ imos method (diff arr) │ Range addition in O(1) + restore in O(n) │
│ Prefix XOR             │ Range XOR in O(1)                         │
│ Prefix GCD             │ Range GCD (also needs right-to-left)      │
│ Prefix max/min         │ Prefix max query in O(1)                  │
│                        │ ※ Sparse Table needed for arbitrary ranges│
└────────────────────────┴───────────────────────────────────────────┘
```

---

## 7. Canonical Pattern 4: Applying Dynamic Programming

### 7.1 Criteria for Applying DP

Dynamic programming (DP) is the most frequently tested paradigm in algorithmic problems. The criteria for applicability are the following two properties.

```
Two conditions for DP:

  1. Optimal Substructure
     → The optimal solution to the whole can be constructed from optimal solutions to subproblems
     → Example: A sub-path of a shortest path is also a shortest path

  2. Overlapping Subproblems
     → The same subproblem appears multiple times
     → Example: Computing fib(5) requires fib(3) multiple times

  DP design steps:
    Step 1: Define the state → What does dp[i] represent?
    Step 2: Derive the recurrence → dp[i] = f(dp[j], ...) form
    Step 3: Set initial conditions → dp[0] = ?
    Step 4: Determine computation order → From smaller to larger states
    Step 5: Identify the answer → dp[n]? max(dp)? dp[n][m]?
```

### 7.2 Collection of Canonical DP Patterns

```python
# ============================================================
# Pattern 1: 1D DP (Longest Increasing Subsequence - LIS)
# ============================================================
import bisect

def lis_length(arr: list[int]) -> int:
    """Length of the longest increasing subsequence O(n log n)"""
    # tails[i] = minimum tail element of an increasing subsequence of length (i+1)
    tails: list[int] = []

    for num in arr:
        pos = bisect.bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)   # New length subsequence
        else:
            tails[pos] = num    # Update to a smaller tail

    return len(tails)


# ============================================================
# Pattern 2: 2D DP (Knapsack Problem)
# ============================================================
def knapsack(weights: list[int], values: list[int], capacity: int) -> int:
    """0-1 Knapsack Problem O(n * capacity)"""
    n = len(weights)
    # dp[j] = maximum value achievable with capacity j (compressed to 1D)
    dp = [0] * (capacity + 1)

    for i in range(n):
        # Reverse loop (to use each item at most once)
        for j in range(capacity, weights[i] - 1, -1):
            dp[j] = max(dp[j], dp[j - weights[i]] + values[i])

    return dp[capacity]


# ============================================================
# Pattern 3: Interval DP (Matrix Chain Multiplication)
# ============================================================
def matrix_chain_order(dims: list[int]) -> int:
    """
    Minimum cost of matrix chain multiplication O(n^3)
    dims[i-1] x dims[i] is the size of the i-th matrix
    """
    n = len(dims) - 1  # Number of matrices
    # dp[i][j] = minimum cost of multiplying matrices i..j
    dp = [[0] * n for _ in range(n)]

    # Increase interval length from 2 to n
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            for k in range(i, j):
                cost = dp[i][k] + dp[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1]
                dp[i][j] = min(dp[i][j], cost)

    return dp[0][n - 1]


# ============================================================
# Pattern 4: Bitmask DP (Traveling Salesman Problem - TSP)
# ============================================================
def tsp(dist: list[list[int]]) -> int:
    """
    Traveling Salesman Problem O(2^n * n^2)
    dist[i][j] = distance from city i to city j
    """
    n = len(dist)
    INF = float('inf')
    # dp[S][i] = minimum cost having visited the set S, currently at city i
    dp = [[INF] * n for _ in range(1 << n)]
    dp[1][0] = 0  # Start from city 0

    for S in range(1 << n):
        for u in range(n):
            if dp[S][u] == INF:
                continue
            if not (S >> u & 1):
                continue
            for v in range(n):
                if S >> v & 1:
                    continue  # Already visited
                new_S = S | (1 << v)
                dp[new_S][v] = min(dp[new_S][v], dp[S][u] + dist[u][v])

    # Visit all cities and return to city 0
    full = (1 << n) - 1
    return min(dp[full][i] + dist[i][0] for i in range(n))


# Tests
print(lis_length([10, 9, 2, 5, 3, 7, 101, 18]))  # 4 ([2,3,7,101] or [2,5,7,101])
print(knapsack([2, 3, 4, 5], [3, 4, 5, 6], 8))    # 10
print(matrix_chain_order([10, 30, 5, 60]))         # 4500

dist_matrix = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0],
]
print(tsp(dist_matrix))  # 80 (0→1→3→2→0)
```

### 7.3 DP State Design Guide

```
┌───────────────────────────────────────────────────────────────────┐
│  DP State Design Cheat Sheet                                      │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Problem Type       │ Typical State Definition                    │
│  ───────────────────┼────────────────────────────────────────────│
│  Array/String       │ dp[i] = optimal value for first i elements │
│  Knapsack family    │ dp[i][w] = optimal for items 0..i, cap w   │
│  Interval           │ dp[l][r] = optimal for interval [l,r]      │
│  Tree               │ dp[v] = optimal for subtree rooted at v    │
│  On DAG             │ dp[v] = optimal path to vertex v           │
│  Bitmask            │ dp[S] = optimal with set S processed       │
│  Digit              │ dp[pos][tight][...] = digit DP states      │
│  Probability/EV     │ dp[state] = expected value at state        │
│  Game               │ dp[state] = win/loss or Grundy number      │
│                                                                   │
│  State reduction techniques:                                      │
│    - Scrolling array: if dp[i] depends only on dp[i-1] → 1D     │
│    - Coordinate compression: map large values to appeared values  │
│    - State symmetry: group rotationally/reflectively equivalent   │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

---

## 8. Iterative Refinement in Practice

### 8.1 Methodology of Iterative Refinement

Rather than aiming for the optimal solution immediately, start from a brute-force solution and improve incrementally. This is the most reliable approach to problem solving.

```
Iterative refinement flow:

  ┌──────────────────┐
  │  Stage 1: Brute  │  Ignore complexity
  │  Force           │  Focus solely on producing correct output
  └────────┬─────────┘
           │ ✓ Verify correctness
           ▼
  ┌──────────────────┐
  │  Stage 2: Observe│  Where is the bottleneck in brute force?
  │                  │  Where are redundant computations?
  └────────┬─────────┘
           │ Identify bottleneck
           ▼
  ┌──────────────────┐
  │  Stage 3: Optimize│ Apply data structures and algorithms
  │                  │  Improve complexity
  └────────┬─────────┘
           │ ✓ Verify with stress test
           ▼
  ┌──────────────────┐
  │  Stage 4: Fine-  │  Improve constant factors if needed
  │  tune            │  Language-specific optimizations
  └──────────────────┘
```

### 8.2 Practical Example: Closest Pair Problem

```
Problem: Given n 2D points, find the closest pair

Stage 1: Brute force O(n^2)
  → Compute distances between all pairs
  → Use as ground truth for stress testing

Stage 2: Sort-based O(n log n + alpha)
  → Sort by x-coordinate, compare only nearby points
  → Faster on average but worst case still O(n^2)

Stage 3: Divide and conquer O(n log n)
  → Split left/right + limited comparisons within strip
  → Optimal (deterministic algorithm)

Stage 4: Randomized O(n) expected
  → Grid method
  → Fastest in expectation but analysis is complex
```

```python
import math
import random

def dist(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Euclidean distance between two points"""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# === Stage 1: O(n^2) brute force ===
def closest_pair_brute(points: list[tuple[float, float]]) -> float:
    """Compute distances between all pairs"""
    n = len(points)
    min_dist = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            d = dist(points[i], points[j])
            min_dist = min(min_dist, d)
    return min_dist


# === Stage 2: Sort + pruning ===
def closest_pair_sorted(points: list[tuple[float, float]]) -> float:
    """Sort by x-coordinate and prune when x-distance exceeds min_dist"""
    points_sorted = sorted(points)  # Sort by x-coordinate
    n = len(points_sorted)
    min_dist = float('inf')

    for i in range(n):
        for j in range(i + 1, n):
            # If x-distance alone exceeds min_dist, skip all remaining
            if points_sorted[j][0] - points_sorted[i][0] >= min_dist:
                break
            d = dist(points_sorted[i], points_sorted[j])
            min_dist = min(min_dist, d)

    return min_dist


# === Stage 3: Divide and conquer O(n log n) ===
def closest_pair_dnc(points: list[tuple[float, float]]) -> float:
    """Closest pair using divide and conquer"""
    def solve(pts_x: list, pts_y: list) -> float:
        n = len(pts_x)
        if n <= 3:
            return closest_pair_brute(pts_x)

        mid = n // 2
        mid_x = pts_x[mid][0]

        # Split into left and right
        left_x = pts_x[:mid]
        right_x = pts_x[mid:]
        left_set = set(map(id, left_x))

        left_y = [p for p in pts_y if id(p) in left_set]  # Simplified implementation
        right_y = [p for p in pts_y if id(p) not in left_set]

        d_left = solve(left_x, left_y)
        d_right = solve(right_x, right_y)
        d = min(d_left, d_right)

        # Check points within the strip
        strip = [p for p in pts_y if abs(p[0] - mid_x) < d]

        for i in range(len(strip)):
            j = i + 1
            while j < len(strip) and strip[j][1] - strip[i][1] < d:
                d = min(d, dist(strip[i], strip[j]))
                j += 1

        return d

    pts_x = sorted(points, key=lambda p: p[0])
    pts_y = sorted(points, key=lambda p: p[1])
    return solve(pts_x, pts_y)


# === Stress test ===
def stress_test_closest_pair(n_tests: int = 200):
    """Stress test comparing brute force with optimized solutions"""
    for test_id in range(n_tests):
        n = random.randint(2, 50)
        points = [(random.uniform(-100, 100), random.uniform(-100, 100))
                   for _ in range(n)]

        d_brute = closest_pair_brute(points)
        d_sorted = closest_pair_sorted(points)
        d_dnc = closest_pair_dnc(points)

        assert abs(d_brute - d_sorted) < 1e-9, f"Test {test_id}: sorted mismatch"
        assert abs(d_brute - d_dnc) < 1e-9, f"Test {test_id}: dnc mismatch"

    print(f"All {n_tests} tests passed")

# stress_test_closest_pair()
```

### 8.3 Generic Stress Test Template

```python
import random
import time

def stress_test(brute_fn, optimized_fn, generator_fn,
                comparator=None, n_tests: int = 1000, verbose: bool = False):
    """
    Generic stress test framework

    brute_fn:      Brute force solution (correct answers)
    optimized_fn:  Optimized solution (function under test)
    generator_fn:  Function that generates test inputs
    comparator:    Result comparison function (None uses ==)
    """
    compare = comparator or (lambda a, b: a == b)

    for test_id in range(n_tests):
        test_input = generator_fn()
        # Compare results on the same input
        result_brute = brute_fn(*test_input)
        result_opt = optimized_fn(*test_input)

        if not compare(result_brute, result_opt):
            print(f"Mismatch! Test #{test_id}")
            print(f"  Input:     {test_input}")
            print(f"  Brute:     {result_brute}")
            print(f"  Optimized: {result_opt}")
            return False

        if verbose and test_id % 100 == 0:
            print(f"  Test #{test_id} passed")

    print(f"All {n_tests} tests passed")
    return True


# Usage example
def gen_two_sum_input():
    n = random.randint(2, 100)
    arr = [random.randint(-100, 100) for _ in range(n)]
    target = random.randint(-200, 200)
    return (arr, target)

# stress_test(two_sum_brute, two_sum_hash, gen_two_sum_input, n_tests=5000)
```

---

## 9. Edge Case Checklist and Problem Category Classification

### 9.1 Comprehensive Edge Case Checklist

Overlooking edge cases is the single biggest cause of WA (Wrong Answer). Check systematically by problem type.

```
┌──────────────────────────────────────────────────────────────────────┐
│  Universal Edge Case Checklist                                       │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Input boundaries:                                                   │
│    □ n = 0 (empty input) → Is a special return value needed?        │
│    □ n = 1 (minimum input) → Does the loop execute at all?          │
│    □ n = max constraint → Will it TLE/MLE?                          │
│                                                                      │
│  Value characteristics:                                              │
│    □ All elements identical → Sorting becomes meaningless            │
│    □ Already sorted / completely reversed → Best/worst case          │
│    □ Negative values / zero → Product sign, division by zero        │
│    □ Integer overflow → Watch out in non-Python languages           │
│    □ Very large / very small values → Precision issues              │
│                                                                      │
│  Graph-specific:                                                     │
│    □ Self-loops → Verify dist[v][v] = 0                             │
│    □ Multi-edges → Select the minimum cost edge?                    │
│    □ Disconnected graph → Return value for unreachable cases        │
│    □ Tree (edges = vertices - 1) → No cycles assumed                │
│    □ Graph with 1 vertex → 0 edges                                  │
│    □ Complete graph → O(n^2) edges, watch memory                    │
│                                                                      │
│  String-specific:                                                    │
│    □ Empty string → Processing at len=0                              │
│    □ Single character → Special handling for palindrome checks etc.  │
│    □ All characters the same → "aaaa" etc.                          │
│    □ Mixed case → Is normalization needed?                          │
│    □ Special characters → Handling of spaces and symbols            │
│                                                                      │
│  Numeric-specific:                                                   │
│    □ Division by zero → Cases where denominator becomes 0           │
│    □ Negative number MOD → Python: -1 % 3 = 2, C++: -1 % 3 = -1   │
│    □ Floating-point comparison → Use abs(a-b) < eps                 │
│    □ Large number MOD → Apply % MOD at each intermediate step      │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 9.2 Problem Category Classification Flowchart

```
Read the problem
    │
    ├─ Optimization problem? (find max/min)
    │   ├─ Greedy choice property?
    │   │   ├─ Yes → Greedy
    │   │   └─ Unclear → Look for counterexample. If none, try greedy
    │   ├─ Overlapping subproblems?
    │   │   ├─ Few states → DP
    │   │   └─ Many states → Memoized recursion or state compression
    │   ├─ Monotonicity in the answer? → Binary search on the answer
    │   └─ Small constraint (n <= 20)? → Bitmask DP / exhaustive search
    │
    ├─ Graph problem?
    │   ├─ Shortest path?
    │   │   ├─ Unweighted → BFS
    │   │   ├─ Non-negative weights → Dijkstra
    │   │   ├─ Negative edges → Bellman-Ford
    │   │   └─ All-pairs → Floyd-Warshall (n <= 300)
    │   ├─ Connectivity?
    │   │   ├─ Static → BFS/DFS
    │   │   └─ Dynamic (edge additions) → Union-Find
    │   ├─ Ordering? → Topological sort
    │   ├─ MST? → Kruskal / Prim
    │   ├─ Matching? → Bipartite matching / network flow
    │   └─ Strongly connected components? → Tarjan / Kosaraju
    │
    ├─ Interval/array problem?
    │   ├─ Range queries (no updates)? → Prefix sum / Sparse Table
    │   ├─ Range queries (with updates)? → Segment tree / BIT
    │   ├─ Subarray sum/product? → Prefix sum / two pointers
    │   ├─ Binary searchable? → Binary search on the answer
    │   └─ Offline queries? → Mo's algorithm
    │
    ├─ String problem?
    │   ├─ Pattern search? → KMP / Z-algorithm / Rabin-Karp
    │   ├─ Prefix search? → Trie
    │   ├─ Subsequence comparison? → DP (LCS / Edit Distance)
    │   ├─ Palindrome? → Manacher / DP
    │   └─ Suffix-related? → Suffix Array
    │
    ├─ Mathematics problem?
    │   ├─ Primality test / factorization? → Sieve of Eratosthenes / trial division
    │   ├─ GCD/LCM? → Euclidean algorithm
    │   ├─ Combinatorial numbers? → Pascal's triangle / modular inverse
    │   ├─ MOD arithmetic? → Fast exponentiation / Fermat's little theorem
    │   └─ Matrix? → Matrix exponentiation
    │
    └─ Game theory?
        ├─ Two-player zero-sum game? → Minimax / Alpha-Beta
        ├─ Nim-type? → Grundy numbers / XOR
        └─ General game? → DP + game theory
```

### 9.3 Algorithm Selection Comparison Table

| Problem Property | First Choice | Second Choice | Avoid |
|:---|:---|:---|:---|
| Shortest path (unweighted) | BFS O(V+E) | --- | Dijkstra (unnecessarily complex) |
| Shortest path (non-negative weights) | Dijkstra O(E log V) | A* | Bellman-Ford O(VE) (slow) |
| Shortest path (negative edges) | Bellman-Ford O(VE) | SPFA | Dijkstra (incorrect) |
| All-pairs shortest path | Floyd-Warshall O(V^3) | Dijkstra V times | BFS V times (wrong with weights) |
| Connected components (static) | Union-Find O(alpha(n)) | BFS/DFS O(V+E) | --- |
| Connected components (dynamic add) | Union-Find | --- | BFS each time (slow) |
| Range sum (no updates) | Prefix sum O(1)/query | --- | Segment tree (overkill) |
| Range sum (with updates) | BIT O(log n) | Segment tree | Recompute each time O(n) |
| Range min (no updates) | Sparse Table O(1)/query | Segment tree | --- |
| Range min (with updates) | Segment tree O(log n) | --- | BIT (not supported) |
| Search in sorted array | Binary search O(log n) | --- | Linear search O(n) |
| Insert/delete/search elements | Hash O(1) avg | Balanced BST O(log n) | Linear search in array |
| Dynamic min/max management | Heap O(log n) | --- | Sort each time O(n log n) |

### 9.4 Complexity Comparison Table

| Operation | Python estimate (n=10^6) | C++ estimate (n=10^6) | Notes |
|:---|:---|:---|:---|
| Simple loop | ~0.1s | ~0.003s | Python is 30-100x slower |
| Sort | ~0.3s | ~0.06s | TimSort is fast |
| dict/set operations | ~0.2s | ~0.05s | Hash constant factor |
| BFS/DFS | ~0.5s | ~0.01s | Watch Python recursion limit |
| Segment tree build | ~1s | ~0.02s | Can be tight in Python |
| Binary search (log n iterations) | ~0.00002s | ~0.000001s | Depends on check function cost |

---

## 10. Anti-Pattern Collection

### Anti-Pattern 1: Ignoring Constraints

```python
# ================================================================
# BAD: Implementing O(n^2) when n=10^5
# → 10^10 operations → TLE (Time Limit Exceeded)
# ================================================================

def find_duplicate_bad(arr: list[int]) -> int:
    """O(n^2): Compare all pairs to find duplicate"""
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] == arr[j]:
                return arr[i]
    return -1

# ================================================================
# GOOD: Check constraints first and estimate acceptable complexity
# n=10^5 → Need O(n log n) or better
# ================================================================

def find_duplicate_good(arr: list[int]) -> int:
    """O(n): Detect duplicates using a hash set"""
    seen: set[int] = set()
    for num in arr:
        if num in seen:
            return num
        seen.add(num)
    return -1
```

### Anti-Pattern 2: Jumping Straight to the Optimal Solution

```python
# ================================================================
# BAD: Trying to implement the complex optimal algorithm from scratch
# → Prone to bugs, hard to debug, wastes time
# ================================================================

# Starting to write segment tree + coordinate compression + offline queries immediately...

# ================================================================
# GOOD: First implement brute force, verify correct output, then optimize
# ================================================================

# Step 1: Verify correctness with O(n^2) brute force
# Step 2: Implement O(n log n) optimal solution
# Step 3: Stress test to compare brute force and optimal to detect bugs
# Step 4: Verify execution time on large inputs
```

### Anti-Pattern 3: Overusing Global Variables

```python
# ================================================================
# BAD: Heavy use of global variables causing state management chaos
# ================================================================

visited = set()       # Global → forgetting to reset between test cases
result = []           # Global → contamination across multiple calls

def dfs_bad(graph, v):
    visited.add(v)
    for u in graph[v]:
        if u not in visited:
            dfs_bad(graph, u)

# ================================================================
# GOOD: Manage state via parameters/return values or encapsulate in functions
# ================================================================

def dfs_good(graph: dict, start: int) -> set[int]:
    """State encapsulated within the function"""
    visited: set[int] = set()
    stack = [start]

    while stack:
        v = stack.pop()
        if v in visited:
            continue
        visited.add(v)
        for u in graph[v]:
            if u not in visited:
                stack.append(u)

    return visited
```

### Anti-Pattern 4: Direct Floating-Point Comparison

```python
# ================================================================
# BAD: Comparing floating-point numbers with ==
# ================================================================

def is_right_triangle_bad(a: float, b: float, c: float) -> bool:
    return a*a + b*b == c*c  # May return False due to floating-point error

# ================================================================
# GOOD: Comparison with tolerance
# ================================================================

EPS = 1e-9

def is_right_triangle_good(a: float, b: float, c: float) -> bool:
    return abs(a*a + b*b - c*c) < EPS
```

### Anti-Pattern 5: Forgetting Recursion Depth Limits

```python
# ================================================================
# BAD: Exceeding Python's default recursion limit (1000)
# ================================================================

def dfs_recursive_bad(graph, v, visited):
    visited.add(v)
    for u in graph[v]:
        if u not in visited:
            dfs_recursive_bad(graph, u, visited)
    # RecursionError on a linear graph with n=10000

# ================================================================
# GOOD: Raise the recursion limit or implement with a stack
# ================================================================

import sys
sys.setrecursionlimit(300000)  # Raise recursion limit

# Or, stack-based iterative implementation (recommended)
def dfs_iterative(graph: dict, start: int) -> list[int]:
    visited = set()
    stack = [start]
    order = []

    while stack:
        v = stack.pop()
        if v in visited:
            continue
        visited.add(v)
        order.append(v)
        for u in reversed(graph[v]):
            if u not in visited:
                stack.append(u)

    return order
```

---

## 11. Exercises (3 Levels)

### Level 1: Fundamentals (Constraint Analysis and Pattern Recognition)

**Exercise 1-1: Derive Complexity from Constraints**

For each of the following problems, estimate the acceptable complexity and the algorithm to use.

```
(a) An array with n <= 15 is given. Find the subset with the maximum sum.
    → Acceptable complexity: O(?)  → Candidate algorithm: ?

(b) An array with n <= 200,000 is given. Answer Q range sum queries. (Q <= 200,000)
    → Acceptable complexity: O(?)  → Candidate algorithm: ?

(c) A weighted graph with n <= 300 vertices. Find all-pairs shortest distances.
    → Acceptable complexity: O(?)  → Candidate algorithm: ?

(d) A string S of length n <= 10^6 and a pattern P of length m <= 10^6 are given.
    Find all positions where P appears in S.
    → Acceptable complexity: O(?)  → Candidate algorithm: ?
```

<details>
<summary>Solution</summary>

```
(a) n <= 15 → O(2^n) = O(32768) → Enumerate all subsets with bitmask.
    Computing the sum of each subset in O(2^n * n) is sufficient.

(b) n, Q <= 200,000 → Preprocess O(n), each query O(1) is ideal.
    → Prefix sum. Preprocessing O(n), query O(1), total O(n + Q).

(c) n <= 300 → O(n^3) = O(2.7 * 10^7) → Floyd-Warshall.
    Running Dijkstra n times for O(n^2 log n) also works, but Floyd-Warshall is simpler.

(d) n, m <= 10^6 → Need O(n + m).
    → KMP algorithm or Z-algorithm. Naive O(nm) will TLE.
```
</details>

**Exercise 1-2: Pattern Matching**

For each of the following problems, identify the algorithm to use from the keywords.

```
(a) "Find the shortest path from vertex s to vertex t in a graph. All edge weights are 1."
(b) "Find the shortest contiguous subarray whose sum is at least K."
(c) "Find the shortest route visiting all N cities and returning. N <= 20."
(d) "Find the longest palindromic substring in string S."
```

<details>
<summary>Solution</summary>

```
(a) Unweighted shortest path → BFS O(V + E)
(b) Contiguous subarray + sum condition + shortest → Two pointers O(n)
(c) Visit all cities + N <= 20 → Bitmask DP (TSP) O(2^n * n^2)
(d) Longest palindromic substring → Manacher O(n) or DP O(n^2)
```
</details>

### Level 2: Applied (Practicing Iterative Refinement)

**Exercise 2-1: Longest Subarray Problem**

Given an integer array arr of length n and a positive integer k, find the length of the longest contiguous subarray with at most k distinct elements.

Constraints: 1 <= n <= 10^5, 1 <= k <= n

```
Input example: arr = [1, 2, 1, 2, 3], k = 2
Output example: 4  (subarray [1, 2, 1, 2])
```

Build the solution incrementally:
1. Implement O(n^3) brute force
2. Improve to O(n^2)
3. Optimize to O(n) (two pointers)
4. Verify with stress testing

<details>
<summary>Solution</summary>

```python
from collections import defaultdict

# Stage 1: O(n^3) enumerate all subarrays, count distinct elements in each
def longest_k_distinct_brute(arr: list[int], k: int) -> int:
    n = len(arr)
    max_len = 0
    for i in range(n):
        for j in range(i, n):
            distinct = len(set(arr[i:j+1]))  # O(n) set construction
            if distinct <= k:
                max_len = max(max_len, j - i + 1)
    return max_len

# Stage 2: O(n^2) incrementally update set
def longest_k_distinct_n2(arr: list[int], k: int) -> int:
    n = len(arr)
    max_len = 0
    for i in range(n):
        count = defaultdict(int)
        distinct = 0
        for j in range(i, n):
            if count[arr[j]] == 0:
                distinct += 1
            count[arr[j]] += 1
            if distinct <= k:
                max_len = max(max_len, j - i + 1)
            else:
                break  # Extending further won't satisfy the condition
    return max_len

# Stage 3: O(n) two pointers
def longest_k_distinct_optimal(arr: list[int], k: int) -> int:
    n = len(arr)
    count = defaultdict(int)
    left = 0
    max_len = 0

    for right in range(n):
        count[arr[right]] += 1

        while len(count) > k:
            count[arr[left]] -= 1
            if count[arr[left]] == 0:
                del count[arr[left]]
            left += 1

        max_len = max(max_len, right - left + 1)

    return max_len
```
</details>

**Exercise 2-2: Binary Search Application**

Given an array arr of n positive integers and a positive integer m, split arr into m contiguous subarrays to minimize the maximum subarray sum.

Constraints: 1 <= m <= n <= 10^5, 1 <= arr[i] <= 10^4

```
Input example: arr = [7, 2, 5, 10, 8], m = 2
Output example: 18  (split: [7, 2, 5] and [10, 8], sums are 14 and 18)
```

<details>
<summary>Solution</summary>

Refer to the `min_max_partition` function in section "5.2 Implementation Patterns." Solve using binary search on the answer. The predicate `can_partition(max_sum)` greedily checks whether m partitions are possible with each sum at most max_sum.

Complexity: O(n log(sum(arr)))
</details>

### Level 3: Advanced (Combined Problems)

**Exercise 3-1: Comprehensive Problem**

There are n people, each with an ability value a[i]. Divide them into k teams to minimize the total difference (max - min) within each team. Each team must have at least one person.

Constraints: 1 <= k <= n <= 5000

```
Input example: n=5, k=2, a=[3, 1, 7, 5, 2]
        After sorting: [1, 2, 3, 5, 7]
        Split example: [1, 2, 3] and [5, 7] → differences: (3-1)+(7-5) = 2+2 = 4
Output example: 4
```

Hint: Sort first, then solve with DP. dp[i][j] = minimum cost of dividing the first i people into j teams.

<details>
<summary>Solution approach</summary>

```python
def min_team_diff(a: list[int], k: int) -> int:
    a.sort()
    n = len(a)
    INF = float('inf')

    # dp[i][j] = minimum cost of dividing a[0..i-1] into j teams
    dp = [[INF] * (k + 1) for _ in range(n + 1)]
    dp[0][0] = 0

    for j in range(1, k + 1):
        for i in range(j, n + 1):
            # Last team is a[m..i-1] (1 or more people)
            for m in range(j - 1, i):
                cost = a[i - 1] - a[m]  # Sorted, so max-min = last-first
                dp[i][j] = min(dp[i][j], dp[m][j - 1] + cost)

    return dp[n][k]

# Complexity: O(n^2 * k)
# Worst case O(n^3) when n=5000, k=5000, but fast enough for small k.
```
</details>

---

## 12. FAQ (Frequently Asked Questions)

### Q1: What should I do when nothing comes to mind for a problem?

**A:** Try the following 5 stages in order.

1. **Compute by hand with small examples.** Create at least 3 examples with n=3 to 5 and solve them on paper. Patterns often emerge.
2. **Derive complexity from constraints.** Just knowing that n <= 10^5 requires O(n log n) drastically narrows the candidates.
3. **Consult the pattern recognition map.** List algorithm candidates from keywords (shortest, maximum, counting, etc.).
4. **Write a brute-force solution.** Having code that produces correct answers provides a stepping stone for optimization. Identify the bottleneck (where is it O(n^2)?) and improve with data structures or algorithms.
5. **Recall similar problems.** Consider whether a past problem's solution can be adapted.

"Nothing comes to mind" usually means either "not enough concrete examples" or "haven't done constraint analysis."

### Q2: My brute force is correct but TLEs. How do I optimize?

**A:** Check the following list from top to bottom.

1. **Eliminate redundant computation**: Are you computing the same value multiple times? → Memoization/prefix sum
2. **Change data structures**: Can you replace linear search with hash lookup or binary search?
3. **Change algorithm**: Can you switch from O(n^2) to an O(n log n) algorithm?
4. **Apply binary search**: If the answer has monotonicity, can you use binary search on the answer?
5. **Use preprocessing**: Preprocess for O(1) query response (prefix sum, Sparse Table, etc.)
6. **Improve constant factors**: Python-specific optimizations (sys.stdin, list comprehensions, PyPy)

### Q3: I can't derive the DP recurrence. What should I do?

**A:** Follow these steps.

1. **Define the state.** Clearly define in plain language "what does dp[i] represent?" A vague definition leads to a vague recurrence.
2. **Trace the recurrence by hand.** Concretely compute dp[0], dp[1], dp[2], ... and write down "what was needed to compute dp[3]."
3. **Focus on the last element.** Case-splitting by "how do we handle the last element" often reveals the recurrence. E.g., select/don't select the last element.
4. **Add dimensions to the state.** If dp[i] alone lacks information, add dimensions like dp[i][j]. E.g., the knapsack problem requires a weight dimension.
5. **Look for similarity to known DP patterns.** Consider correspondences with canonical patterns like LIS, LCS, knapsack, interval DP, bitmask DP.

### Q4: How should I allocate time during contests?

**A:** Recommended time allocation for a typical contest (2-hour, 6-problem format):

```
Problem A: 5 min    (implementation only)
Problem B: 10 min   (simple algorithm)
Problem C: 20 min   (standard algorithm)
Problem D: 30 min   (applied algorithm)
Problem E: 30 min   (advanced algorithm)
Problem F: 25 min   (very advanced / acceptable to skip)

Important: Skip a problem if you can't find an approach after 20 minutes.
           Come back to it with remaining time.
```

### Q5: Should I use Python or C++?

**A:** It depends on the situation.

```
When Python is advantageous:
  - Big integers are needed (Python supports them natively)
  - Implementation is complex and you want to reduce bugs
  - n is around 10^5 and constant factors aren't a bottleneck
  - Libraries (itertools, collections, etc.) are useful

When C++ is advantageous:
  - n is 10^6 or larger and constant factors matter
  - Tight time limits (1-2 seconds)
  - STL data structures (set, map, priority_queue) are convenient

Compromise: Use PyPy
  - Python's ease of writing with 3-5x speedup
  - Some libraries may not be available
```

### Q6: How large should stress tests be?

**A:** Guidelines for test input size and number of runs:

```
Purpose-based test configurations:

  Correctness verification (comparison with brute force):
    - Input size: n = 1-50 (range where brute force runs fast)
    - Number of tests: 1000-10000
    - Time: Adjust to complete in seconds to tens of seconds

  Edge case verification:
    - Explicitly test n = 0, 1, 2
    - All elements identical, sorted, reversed
    - Maximum values, minimum values, zero

  Performance verification:
    - n = maximum constraint (10^5, 10^6, etc.)
    - Manually construct worst cases (sorted, reversed, random, etc.)
    - Measure execution time on 1-3 cases
```

---


## FAQ

### Q1: What is the most important point when studying this topic?

Gaining practical experience is the most important aspect. Understanding deepens not just through theory alone, but by actually writing code and verifying its behavior.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals and jumping into advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before proceeding to the next step.

### Q3: How is this used in practice?

Knowledge of this topic is frequently utilized in day-to-day development work. It becomes particularly important during code reviews and architecture design.

---

## 13. Summary

| Item | Key Points |
|:---|:---|
| 5-Step Problem Solving | Understand → Examples → Constraint analysis → Design → Implement & Verify |
| Constraint Analysis | Derive O(?) from the range of n to select the right algorithm |
| Pattern Recognition | Map keywords and problem structure to known techniques |
| Iterative Refinement | Progress in 3 stages: brute force → verify → optimize |
| Edge Cases | Always check: empty input, min/max, special values, graph-specific cases |
| Canonical Techniques | Two pointers, binary search on answer, prefix sum, DP, bitmask DP |
| Stress Testing | Comparing with brute force is the most powerful debugging method |
| Anti-Patterns | Ignoring constraints, jumping to optimal, global variable abuse, direct float comparison |
| Time Management | Have the courage to skip after 20 minutes without a plan |

---

## Recommended Next Guides

- [Competitive Programming](./01-competitive-programming.md) -- Sharpen problem-solving skills through competition
- [Dynamic Programming](../02-algorithms/04-dynamic-programming.md) -- The most frequently tested paradigm
- [Graph Traversal](../02-algorithms/02-graph-traversal.md) -- Graph problem fundamentals
- Sorting -- Frequently used as preprocessing
- Data Structure Fundamentals -- Selecting appropriate data structures

---

## References

1. Skiena, S. S. (2020). *The Algorithm Design Manual* (3rd ed.). Springer. -- Part I: Practical Algorithm Design details the overall problem-solving framework. Chapters 1-8 provide the foundation for pattern recognition and design techniques.

2. Polya, G. (1945). *How to Solve It*. Princeton University Press. -- A classic on mathematical problem solving. The 4-step framework of "Understand → Plan → Execute → Review" is directly applicable to algorithmic problem solving.

3. Halim, S. & Halim, F. (2013). *Competitive Programming 3*. -- Chapters 1-3 comprehensively cover practical problem-solving techniques. Pattern recognition, constraint analysis, and application of canonical algorithms are explained with abundant examples.

4. Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. -- Part I "Foundations" provides the theoretical basis for complexity analysis. All correctness proofs and complexity analyses of algorithms are grounded here.

5. Laaksonen, A. (2017). *Competitive Programmer's Handbook*. -- A freely available online resource. Canonical patterns including binary search, DP, and graph algorithms are concisely summarized, making it ideal for pattern recognition training. URL: https://cses.fi/book/book.pdf

6. Sedgewick, R. & Wayne, K. (2011). *Algorithms* (4th ed.). Addison-Wesley. -- An implementation-oriented textbook. Rich in Java code examples, covering data structure selection guidelines and practical applications of complexity analysis.
