# Complexity Analysis (Big-O Notation)

> The difference between code that "works" and code that is "fast" lies in computational complexity. The gap between O(n^2) and O(n log n) becomes critical as data size grows.

## Learning Objectives

- [ ] Express time complexity using Big-O notation
- [ ] Develop intuitive understanding of major complexity classes
- [ ] Analyze the complexity of code by inspection
- [ ] Evaluate space complexity (memory usage)
- [ ] Determine the complexity of recursive algorithms using the Master Theorem
- [ ] Understand and apply the concept of amortized complexity

## Prerequisites


---

## 1. Big-O Notation

### 1.1 Definition and Intuition

```
Big-O Notation: Expresses the "growth rate" of execution time relative to input size n

  Formal definition:
    f(n) = O(g(n)) ⟺ ∃c > 0, ∃n₀ > 0 such that
    ∀n ≥ n₀: f(n) ≤ c × g(n)

  Intuition:
  "For sufficiently large n, f(n) is bounded above by a constant multiple of g(n)"

  Example:
    3n² + 5n + 100 = O(n²)
    → As n grows, n² dominates
    → 5n and 100 become negligible

  Rules:
  1. Ignore constant factors: 5n → O(n)
  2. Ignore lower-order terms: n² + n → O(n²)
  3. Ignore base of logarithm: log₂n = log₁₀n × constant → O(log n)
```

```
Concrete examples of Big-O calculation:

  Example 1: f(n) = 3n² + 5n + 100 = O(n²)

    Proof: Let c = 5, n₀ = 100
    For n ≥ 100:
    3n² + 5n + 100 ≤ 3n² + n² + n² = 5n²
    c = 5 gives f(n) ≤ 5n² → O(n²) ✓

  Example 2: f(n) = log₂n = O(log n)

    Base conversion: log₂n = log₁₀n / log₁₀2 = log₁₀n × 3.32...
    → Differs only by a constant factor → Big-O does not distinguish
    → O(log₂n) = O(log₁₀n) = O(ln n) = O(log n)

  Example 3: f(n) = 2^(n+1) = O(2ⁿ)

    2^(n+1) = 2 × 2ⁿ
    c = 2 gives f(n) ≤ 2 × 2ⁿ → O(2ⁿ) ✓

  Example 4: f(n) = n! is NOT O(2ⁿ)

    n! = n × (n-1) × ... × 1 grows faster than 2ⁿ
    n! = Ω(2ⁿ) but n! ≠ O(2ⁿ) (for n≥5, n! > 2ⁿ×n)
    Precisely: n! = O(nⁿ) and n! = Ω((n/e)ⁿ)
```

### 1.2 Major Complexity Classes

```
Comparison of complexity classes (n = input size):

  ┌──────────────┬──────────────┬──────────────────────────┐
  │ Notation     │ Name         │ Operations for n=100     │
  ├──────────────┼──────────────┼──────────────────────────┤
  │ O(1)         │ Constant     │ 1                        │
  │ O(log n)     │ Logarithmic  │ 7                        │
  │ O(√n)        │ Square root  │ 10                       │
  │ O(n)         │ Linear       │ 100                      │
  │ O(n log n)   │ Linearithmic │ 700                      │
  │ O(n²)        │ Quadratic    │ 10,000                   │
  │ O(n³)        │ Cubic        │ 1,000,000                │
  │ O(2ⁿ)        │ Exponential  │ 1.27 × 10³⁰            │
  │ O(n!)        │ Factorial    │ 9.33 × 10¹⁵⁷           │
  └──────────────┴──────────────┴──────────────────────────┘

  Execution time for n=1,000,000 (one million) at 1 operation = 1 ns:
  O(1):        0.001 μs     Instantaneous
  O(log n):    0.02 μs      Instantaneous
  O(√n):       1 μs         Instantaneous
  O(n):        1 ms          Instantaneous
  O(n log n):  20 ms         A blink
  O(n²):       16 minutes    Time for a coffee
  O(n³):       31.7 years    A lifetime
  O(2ⁿ):       Exceeds the age of the universe

  → The gap between O(n²) and O(n log n) at n=1 million is 48,000x!
```

### 1.3 Growth Rate Graph (ASCII)

```
  Operations
  │
  │                                          ╱ O(2ⁿ)
  │                                        ╱
  │                                      ╱
  │                                    ╱
  │                               ╱──── O(n²)
  │                          ╱───
  │                    ╱────
  │              ╱────
  │         ╱───────────────── O(n log n)
  │    ╱──────────────────── O(n)
  │ ╱
  │──────────────────────── O(log n)
  │━━━━━━━━━━━━━━━━━━━━━━━ O(1)
  └───────────────────────────── n
```

### 1.4 Intuitive Understanding of Each Complexity Class

```
The "feel" of each complexity class:

  O(1) — Constant time:
  ┌──────────────────────────────────────┐
  │ "Pick the 3rd book from the shelf"   │
  │ Same time regardless of input size   │
  │                                      │
  │ Examples:                            │
  │ - Array index access arr[i]          │
  │ - Hash table lookup                  │
  │ - Stack push/pop                     │
  │ - Calculation using a formula        │
  └──────────────────────────────────────┘

  O(log n) — Logarithmic time:
  ┌──────────────────────────────────────┐
  │ "Look up a word in a dictionary"     │
  │ Halve the search space each time     │
  │ Only 30 steps even for n = 1 billion │
  │                                      │
  │ Examples:                            │
  │ - Binary search                      │
  │ - Balanced binary tree operations    │
  │ - Exponentiation by squaring         │
  └──────────────────────────────────────┘

  O(√n) — Square root time:
  ┌──────────────────────────────────────┐
  │ "Primality test: trial division      │
  │  up to √n"                           │
  │ Only 31,623 steps for n = 1 billion  │
  │                                      │
  │ Examples:                            │
  │ - Trial division for primality       │
  │ - Square root decomposition          │
  └──────────────────────────────────────┘

  O(n) — Linear time:
  ┌──────────────────────────────────────┐
  │ "Check every book on the shelf       │
  │  one by one"                         │
  │ Process each element exactly once    │
  │                                      │
  │ Examples:                            │
  │ - Array traversal (max, sum)         │
  │ - Linked list search                 │
  │ - Counting sort                      │
  └──────────────────────────────────────┘

  O(n log n) — Linearithmic time:
  ┌──────────────────────────────────────┐
  │ "Take all the books off the shelf    │
  │  and rearrange them"                 │
  │ Efficient processing via divide      │
  │ and conquer                          │
  │                                      │
  │ Examples:                            │
  │ - Merge sort, quick sort             │
  │ - FFT (Fast Fourier Transform)       │
  │ - Convex hull computation            │
  └──────────────────────────────────────┘

  O(n²) — Quadratic time:
  ┌──────────────────────────────────────┐
  │ "Shake hands with everyone"          │
  │ n people require n(n-1)/2 handshakes │
  │                                      │
  │ Examples:                            │
  │ - Bubble sort, insertion sort        │
  │ - All-pairs comparison               │
  │ - Simple matrix operations           │
  └──────────────────────────────────────┘

  O(2ⁿ) — Exponential time:
  ┌──────────────────────────────────────┐
  │ "Try every combination"              │
  │ All subsets of n elements = 2ⁿ       │
  │                                      │
  │ Examples:                            │
  │ - Naive recursive Fibonacci          │
  │ - Subset sum brute force             │
  │ - TSP brute force                    │
  └──────────────────────────────────────┘

  O(n!) — Factorial time:
  ┌──────────────────────────────────────┐
  │ "Try every permutation"              │
  │ All orderings of n elements = n!     │
  │                                      │
  │ Examples:                            │
  │ - TSP brute force (all permutations) │
  │ - Generating all permutations        │
  │ n=20 gives 2.4×10¹⁸ → practically   │
  │ impossible                           │
  └──────────────────────────────────────┘
```

---

## 2. Determining Complexity

### 2.1 Basic Patterns

```python
# Pattern 1: O(1) — Constant time
def get_first(arr):
    return arr[0]  # Index access is O(1)

# Pattern 2: O(n) — Linear time
def sum_array(arr):
    total = 0
    for x in arr:      # Loop n times
        total += x     # O(1) operation
    return total
# → O(n)

# Pattern 3: O(n²) — Nested loop
def has_duplicate(arr):
    n = len(arr)
    for i in range(n):       # n times
        for j in range(i+1, n):  # up to n-1 times
            if arr[i] == arr[j]:
                return True
    return False
# → O(n × n) = O(n²)

# Pattern 4: O(log n) — Halving
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1     # Search space halved
        else:
            right = mid - 1    # Search space halved
    return -1
# → Halved each time → log₂(n) iterations → O(log n)

# Pattern 5: O(n log n) — Divide and conquer
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])    # T(n/2)
    right = merge_sort(arr[mid:])   # T(n/2)
    return merge(left, right)       # O(n)
# T(n) = 2T(n/2) + O(n) → O(n log n)
```

### 2.2 Detailed Loop Complexity Analysis

```python
# Case 1: Independent loops → Addition
def example1(arr):
    n = len(arr)

    # Loop 1: O(n)
    for i in range(n):
        process(arr[i])

    # Loop 2: O(n)
    for i in range(n):
        process(arr[i])

    # Total: O(n) + O(n) = O(2n) = O(n)
    # → Constant factors are ignored

# Case 2: Nested loops → Multiplication
def example2(arr):
    n = len(arr)
    for i in range(n):        # n times
        for j in range(n):    # n times
            process(arr[i], arr[j])
    # Total: O(n × n) = O(n²)

# Case 3: Inner loop depends on i
def example3(arr):
    n = len(arr)
    for i in range(n):
        for j in range(i):    # i times (0, 1, 2, ..., n-1)
            process(arr[i], arr[j])
    # Total: 0 + 1 + 2 + ... + (n-1) = n(n-1)/2 = O(n²)

# Case 4: Loop variable doubles
def example4(n):
    i = 1
    while i < n:
        process(i)
        i *= 2  # 1, 2, 4, 8, 16, ...
    # How many iterations? 2^k = n → k = log₂(n)
    # Total: O(log n)

# Case 5: Loop variable takes square root
def example5(n):
    i = n
    while i > 1:
        process(i)
        i = int(i ** 0.5)  # n, √n, n^(1/4), n^(1/8), ...
    # 2^(2^k) = n → k = log₂(log₂(n))
    # Total: O(log log n)

# Case 6: Double loop but total is O(n)
def example6(arr):
    """Classic two-pointer example"""
    n = len(arr)
    j = 0
    for i in range(n):         # n times
        while j < n and arr[j] < arr[i]:
            j += 1             # j increases at most n times total
    # Outer loop: n times, j increments: n times total
    # Total: O(n + n) = O(n) ← NOT O(n²)!

# Case 7: O(2^n) with recursion
def example7(n):
    """Naive recursive Fibonacci"""
    if n <= 1:
        return n
    return example7(n-1) + example7(n-2)
    # T(n) = T(n-1) + T(n-2) + O(1)
    # T(n) ≈ 2T(n-1) → O(2^n)
    # More precisely O(φ^n) where φ = (1+√5)/2 ≈ 1.618
```

### 2.3 Recurrence Complexity — The Master Theorem

```
Master Theorem:
  For recurrences of the form T(n) = a × T(n/b) + O(n^d):

  Case 1: d < log_b(a)  → T(n) = O(n^(log_b(a)))
  Case 2: d = log_b(a)  → T(n) = O(n^d × log n)
  Case 3: d > log_b(a)  → T(n) = O(n^d)

  Intuition:
  - Case 1: Recursive "branching" dominates (many leaves)
  - Case 2: Work at each level is balanced
  - Case 3: The "combine" step dominates (heavy work at the root)

  Application examples:
  ─────────────────────────────────────────────────
  Merge sort: T(n) = 2T(n/2) + O(n)
    a=2, b=2, d=1 → log₂(2)=1=d → Case 2 → O(n log n) ✓

  Binary search: T(n) = T(n/2) + O(1)
    a=1, b=2, d=0 → log₂(1)=0=d → Case 2 → O(log n) ✓

  Strassen's matrix multiplication: T(n) = 7T(n/2) + O(n²)
    a=7, b=2, d=2 → log₂(7)≈2.81 > 2 → Case 1 → O(n^2.81) ✓

  Divide-and-conquer max: T(n) = 2T(n/2) + O(1)
    a=2, b=2, d=0 → log₂(2)=1 > 0 → Case 1 → O(n^1) = O(n) ✓

  Karatsuba multiplication: T(n) = 3T(n/2) + O(n)
    a=3, b=2, d=1 → log₂(3)≈1.585 > 1 → Case 1 → O(n^1.585) ✓

  Linear search (recursive): T(n) = T(n-1) + O(1)
    → Master Theorem does not apply (n-1, not n/b)
    → Direct expansion: T(n) = T(n-1) + 1 = T(n-2) + 2 = ... = O(n)
```

```
When the Master Theorem does not apply:

  1. T(n) = T(n-1) + O(1) → Direct expansion → O(n)
  2. T(n) = T(n-1) + O(n) → Direct expansion → O(n²)
     T(n) = n + (n-1) + ... + 1 = n(n+1)/2
  3. T(n) = 2T(n-1) + O(1) → O(2^n)
  4. T(n) = T(n/2) + O(n) → O(n) (also via Master Theorem Case 3)
  5. T(n) = T(√n) + O(1) → Solve via variable substitution
     Let m = log n, then T(2^m) = T(2^(m/2)) + O(1)
     S(m) = S(m/2) + O(1) → O(log m) = O(log log n)

  Recursion tree method:
  ┌──────────────────────────────────────┐
  │ For T(n) = 2T(n/2) + cn:            │
  │                                      │
  │ Level 0:        cn                   │
  │                / \                   │
  │ Level 1:    cn/2  cn/2     = cn      │
  │            / \    / \                │
  │ Level 2: cn/4 ... cn/4    = cn      │
  │          ...                        │
  │ Level k:   c × n leaves   = cn      │
  │                                      │
  │ Height: log₂(n) levels              │
  │ Work per level: cn                   │
  │ Total: cn × log₂(n) = O(n log n) ✓  │
  └──────────────────────────────────────┘
```

### 2.4 Amortized Analysis

```python
# Is the append operation of a dynamic array (Python list) O(1)?

# Actual behavior:
# - When capacity remains: O(1)
# - When capacity is exceeded: O(n) (copy to new array)

# Amortized analysis:
# Total cost of n appends:
# 1, 1, 1, ..., 1, n, 1, 1, ..., 1, 2n, ...
#                  ↑ resize          ↑ resize
#
# When capacity doubles:
# Total cost = n + (1 + 2 + 4 + ... + n) = n + 2n = 3n
# Per operation = 3n / n = O(1) (amortized)
#
# → Individual operations range from O(1) to O(n),
#   but n operations total O(n) → O(1) per operation

# Other examples of amortized O(1):
# - Hash table insertion (O(n) during rehash)
# - Union-Find operations (with path compression + union by rank)
# - Binary counter increment
```

```
Three techniques for amortized analysis:

  1. Aggregate Method
  ┌──────────────────────────────────────┐
  │ Compute total cost T for n           │
  │ operations, then amortized cost      │
  │ per operation is T/n                 │
  │                                      │
  │ Example: Dynamic array append        │
  │ Total for n operations: O(n)         │
  │ → O(1) per operation                 │
  └──────────────────────────────────────┘

  2. Accounting Method
  ┌──────────────────────────────────────┐
  │ Assign a "charge" to each operation  │
  │ Overcharge cheap operations,         │
  │ undercharge expensive ones           │
  │ Use prepaid "savings" to cover       │
  │ expensive operations                 │
  │                                      │
  │ Example: Dynamic array append        │
  │ - Normal append: charge 3            │
  │   1: actual insertion,               │
  │   2: savings for future copy         │
  │ - On resize: pay copy cost from      │
  │   savings                            │
  └──────────────────────────────────────┘

  3. Potential Method
  ┌──────────────────────────────────────┐
  │ Define a "potential function" Φ for  │
  │ the data structure                   │
  │ Amortized cost = actual cost + ΔΦ   │
  │                                      │
  │ Example: Dynamic array (size s,      │
  │ capacity c)                          │
  │ Φ = 2s - c (twice usage - capacity)  │
  │ - Normal append:                     │
  │   actual cost 1 + ΔΦ=2 → amortized 3│
  │ - On resize:                         │
  │   actual cost=s+1 + ΔΦ=-(s-1)       │
  │   → amortized 2                      │
  └──────────────────────────────────────┘
```

---

## 3. Space Complexity

### 3.1 Analyzing Memory Usage

```python
# Space complexity: Additional memory used by an algorithm

# O(1) space: Fixed amount of memory beyond input
def find_max(arr):
    max_val = arr[0]  # Only one variable
    for x in arr:
        if x > max_val:
            max_val = x
    return max_val
# Space: O(1) — only max_val

# O(n) space: Memory proportional to input
def reverse_array(arr):
    result = []        # New array
    for x in reversed(arr):
        result.append(x)
    return result
# Space: O(n) — result array

# O(n) space (recursive stack)
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
# Space: O(n) — recursion depth is n (n stack frames)

# O(log n) space
# Merge sort recursion depth: O(log n)
# But O(n) space is needed for array copies

# Time-space tradeoff:
# Example: Duplicate detection
# Method 1: O(n²) time, O(1) space — all-pairs comparison
# Method 2: O(n) time, O(n) space — using a hash set
# → Trading memory for speed
```

### 3.2 Detailed Space Complexity Analysis

```python
# Example 1: In-place vs out-of-place

# Out-of-place: O(n) additional space
def sorted_copy(arr):
    return sorted(arr)  # Creates a new list

# In-place: O(1) additional space
def sort_inplace(arr):
    arr.sort()  # Modifies the original list

# Example 2: Space complexity of recursion

# O(n) space — deep recursion
def sum_recursive(arr, n):
    if n == 0:
        return 0
    return arr[n-1] + sum_recursive(arr, n-1)
# Call stack: n frames

# O(log n) space — shallow recursion
def sum_divide(arr, left, right):
    if left == right:
        return arr[left]
    mid = (left + right) // 2
    return sum_divide(arr, left, mid) + sum_divide(arr, mid+1, right)
# Call stack: log₂(n) frames

# Example 3: Tail Call Optimization (TCO)
# Some languages (Scheme, Scala, etc.) optimize tail recursion to O(1) space
def factorial_tail(n, acc=1):
    if n <= 1:
        return acc
    return factorial_tail(n - 1, n * acc)  # Tail-position recursive call
# Python does NOT support tail call optimization
# → Converting to an iterative approach is standard practice

# Example 4: Space optimization in DP
# Standard DP: O(n × m)
def lcs_full(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]
# Space: O(m × n)

# Space-optimized DP: O(min(m, n))
def lcs_optimized(s1, s2):
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
# Space: O(n) — only two rows retained
```

### 3.3 Memory Considerations in Practice

```
Practical guidelines for memory usage:

  ┌──────────────────┬───────────────────────────────┐
  │ Data volume      │ Approximate memory usage       │
  ├──────────────────┼───────────────────────────────┤
  │ int array 10^6   │ ~4MB (32-bit) / 8MB (64-bit)  │
  │ int array 10^7   │ ~40MB / 80MB                   │
  │ int array 10^8   │ ~400MB / 800MB                 │
  │ 2D array 10^3×10^3│ ~4MB / 8MB                    │
  │ 2D array 10^4×10^4│ ~400MB / 800MB                │
  │ string 10^6 chars│ ~1MB (ASCII) / 4MB (UTF-32)    │
  └──────────────────┴───────────────────────────────┘

  Memory hierarchy and speed:
  ┌───────────┬─────────────┬──────────────┐
  │ Level     │ Size        │ Access time  │
  ├───────────┼─────────────┼──────────────┤
  │ L1 cache  │ 32-64 KB    │ 1 ns         │
  │ L2 cache  │ 256 KB-1MB  │ 4 ns         │
  │ L3 cache  │ 8-64 MB     │ 10 ns        │
  │ Main RAM  │ 8-128 GB    │ 100 ns       │
  │ SSD       │ 256GB-4TB   │ 100 μs       │
  │ HDD       │ 1-20 TB     │ 10 ms        │
  └───────────┴─────────────┴──────────────┘

  → Even an O(n) algorithm can see 10-100x difference
    in measured speed depending on whether data fits in cache!

  Memory-saving techniques:
  1. Streaming (avoid loading all data into memory)
  2. Use generators/iterators
  3. Use bitboards (instead of boolean arrays)
  4. Space-optimize DP tables (full table → 2 rows)
  5. External memory algorithms (leverage disk)
```

---

## 4. Complexity in Practice

### 4.1 Reverse-Engineering Complexity from Constraints

```
Guidelines for competitive programming / real-world applications:

  Operations per second: approximately 10^8 to 10^9

  ┌──────────┬──────────────────┬──────────────────┐
  │ Data size│ Allowable        │ Usable           │
  │          │ complexity       │ algorithms       │
  ├──────────┼──────────────────┼──────────────────┤
  │ n ≤ 10   │ O(n!) ← OK      │ Brute force      │
  │ n ≤ 20   │ O(2ⁿ) ← OK     │ Bitmask brute    │
  │ n ≤ 500  │ O(n³) ← OK     │ Triple loop      │
  │ n ≤ 5000 │ O(n²) ← OK     │ Double loop      │
  │ n ≤ 10⁶  │ O(n log n) ← OK│ Sort, binary     │
  │          │                  │ search           │
  │ n ≤ 10⁸  │ O(n) ← OK      │ Linear scan      │
  │ n ≤ 10¹⁸ │ O(log n) ← OK  │ Binary search,   │
  │          │                  │ math             │
  └──────────┴──────────────────┴──────────────────┘

  Real-world web applications:
  - API response: within 100ms → up to O(n log n) (n ~ tens of thousands)
  - Batch processing: minutes to hours → O(n²) may be acceptable
  - Real-time: within 16ms (60fps) → O(n) or better preferred
  - Database queries: O(log n) index-based access is standard
```

### 4.2 Common Optimization Patterns

```python
# Classic patterns for improving O(n²) → O(n)

# Pattern 1: Hash map for O(1) lookup
# Problem: Find a pair in an array that sums to target

# ❌ O(n²): Brute force all pairs
def two_sum_brute(nums, target):
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]

# ✅ O(n): Hash map
def two_sum_hash(nums, target):
    seen = {}  # value → index
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:  # O(1) lookup
            return [seen[complement], i]
        seen[num] = i

# Pattern 2: Sort then binary search
# Problem: Find the smallest value >= target in a sorted array

# ❌ O(n): Linear search
def find_ceiling_linear(arr, target):
    for x in arr:
        if x >= target:
            return x

# ✅ O(log n): Binary search
import bisect
def find_ceiling_binary(arr, target):
    idx = bisect.bisect_left(arr, target)
    return arr[idx] if idx < len(arr) else None

# Pattern 3: Sliding window
# Problem: Maximum sum of a contiguous subarray of length k

# ❌ O(nk): Recompute sum each time
def max_sum_brute(arr, k):
    max_sum = 0
    for i in range(len(arr) - k + 1):
        max_sum = max(max_sum, sum(arr[i:i+k]))
    return max_sum

# ✅ O(n): Slide the window
def max_sum_sliding(arr, k):
    window = sum(arr[:k])
    max_sum = window
    for i in range(k, len(arr)):
        window += arr[i] - arr[i-k]  # Add and remove
        max_sum = max(max_sum, window)
    return max_sum
```

### 4.3 Additional Optimization Patterns

```python
# Pattern 4: Prefix sum (precomputation)
# Problem: Answer Q range-sum queries over interval [l, r]

# ❌ O(n × Q): Recompute sum each time
def range_sum_brute(arr, queries):
    results = []
    for l, r in queries:
        results.append(sum(arr[l:r+1]))  # O(n)
    return results

# ✅ O(n + Q): Precompute for O(1) queries
def range_sum_prefix(arr, queries):
    n = len(arr)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i+1] = prefix[i] + arr[i]  # Precompute O(n)

    results = []
    for l, r in queries:
        results.append(prefix[r+1] - prefix[l])  # O(1)
    return results

# Pattern 5: Two pointers
# Problem: Find pairs satisfying a condition in a sorted array

# ❌ O(n²): All pairs
def count_pairs_brute(arr, target):
    count = 0
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            if arr[i] + arr[j] <= target:
                count += 1
    return count

# ✅ O(n): Two pointers (assumes sorted input)
def count_pairs_two_pointer(arr, target):
    count = 0
    left, right = 0, len(arr) - 1
    while left < right:
        if arr[left] + arr[right] <= target:
            count += right - left  # All pairs with left and left+1...right
            left += 1
        else:
            right -= 1
    return count

# Pattern 6: Running min/max precomputation
# Problem: Maximum profit from stock trading

# ❌ O(n²): Compare all buy-sell pairs
def max_profit_brute(prices):
    max_p = 0
    for i in range(len(prices)):
        for j in range(i+1, len(prices)):
            max_p = max(max_p, prices[j] - prices[i])
    return max_p

# ✅ O(n): Track minimum while scanning
def max_profit_optimal(prices):
    min_price = float('inf')
    max_profit = 0
    for price in prices:
        min_price = min(min_price, price)
        max_profit = max(max_profit, price - min_price)
    return max_profit

# Pattern 7: Monotonic stack
# Problem: Find the "next greater element" to the right of each element

# ❌ O(n²): Linear scan to the right for each element
def next_greater_brute(arr):
    n = len(arr)
    result = [-1] * n
    for i in range(n):
        for j in range(i+1, n):
            if arr[j] > arr[i]:
                result[i] = arr[j]
                break
    return result

# ✅ O(n): Monotonic stack
def next_greater_stack(arr):
    n = len(arr)
    result = [-1] * n
    stack = []  # Stack of indices
    for i in range(n):
        while stack and arr[i] > arr[stack[-1]]:
            result[stack.pop()] = arr[i]
        stack.append(i)
    return result
# Each element is pushed/popped from the stack at most once → O(n)
```

---

## 5. Asymptotic Notations Beyond Big-O

### 5.1 Omega and Theta Notations

```
Three asymptotic notations:

  O (Big-O):  Upper bound — "at most this much"
    f(n) = O(g(n)) → f(n) ≤ c × g(n)

  Ω (Big-Omega): Lower bound — "at least this much"
    f(n) = Ω(g(n)) → f(n) ≥ c × g(n)

  Θ (Big-Theta): Tight bound — "exactly this much"
    f(n) = Θ(g(n)) → f(n) = O(g(n)) and f(n) = Ω(g(n))

  Example:
  Lower bound for comparison-based sorting:  Ω(n log n)
  → No matter what tricks are used, it cannot be faster than O(n log n)

  Merge sort: Θ(n log n) — worst, average, and best are all the same
  Quick sort: O(n²) worst case, Θ(n log n) average

  In practice:
  - Big-O (upper bound) is most important → guarantees worst case
  - Average complexity is also important → predicts actual performance
  - Best-case complexity is rarely important → "lucky case"
```

### 5.2 Little-o and Little-omega Notations

```
Lowercase asymptotic notations (rarely used but helpful to know):

  o (Little-o):  Strict upper bound
    f(n) = o(g(n)) → lim(n→∞) f(n)/g(n) = 0
    "f(n) grows strictly slower than g(n)"

    Example: n = o(n²)  — n grows strictly slower than n²
             n² ≠ o(n²) — n² grows at the same rate as n²

  ω (Little-omega):  Strict lower bound
    f(n) = ω(g(n)) → lim(n→∞) f(n)/g(n) = ∞

    Example: n² = ω(n)  — n² grows strictly faster than n

  Summary of relationships:
  ┌───────┬────────────────────────────────┐
  │ Notn. │ Analogy                        │
  ├───────┼────────────────────────────────┤
  │ O     │ ≤ (at most)                    │
  │ Ω     │ ≥ (at least)                   │
  │ Θ     │ = (equal, up to constant)      │
  │ o     │ < (strictly less than)         │
  │ ω     │ > (strictly greater than)      │
  └───────┴────────────────────────────────┘
```

### 5.3 Worst, Average, and Best-Case Complexity

```
Meaning of the three cases:

  Worst-case complexity:
  ┌──────────────────────────────────────┐
  │ Execution time on the most           │
  │ unfavorable input                    │
  │ → Most important as a performance    │
  │   "guarantee"                        │
  │ → Big-O typically refers to this     │
  │                                      │
  │ Example: Quick sort O(n²)            │
  │    → Worst case occurs on already    │
  │      sorted arrays                   │
  └──────────────────────────────────────┘

  Average-case complexity:
  ┌──────────────────────────────────────┐
  │ Expected execution time on random    │
  │ input                                │
  │ → Useful for predicting actual       │
  │   performance                        │
  │ → Requires assumptions about input   │
  │   probability distribution           │
  │                                      │
  │ Example: Quick sort O(n log n)       │
  │    → Fast on random input            │
  └──────────────────────────────────────┘

  Best-case complexity:
  ┌──────────────────────────────────────┐
  │ Execution time on the most           │
  │ favorable input                      │
  │ → Rarely meaningful in practice      │
  │                                      │
  │ Example: Insertion sort O(n)         │
  │    → Best case on already sorted     │
  │      input                           │
  │    → Useful for "nearly sorted" data │
  └──────────────────────────────────────┘

  The three complexities for major algorithms:
  ┌────────────────┬─────────┬──────────┬─────────┐
  │ Algorithm      │ Best    │ Average  │ Worst   │
  ├────────────────┼─────────┼──────────┼─────────┤
  │ Binary search  │ O(1)    │ O(log n) │ O(log n)│
  │ Linear search  │ O(1)    │ O(n)     │ O(n)    │
  │ Insertion sort │ O(n)    │ O(n²)   │ O(n²)  │
  │ Merge sort     │ O(n logn)│O(n logn) │O(n logn)│
  │ Quick sort     │ O(n logn)│O(n logn) │ O(n²)  │
  │ Hash lookup    │ O(1)    │ O(1)     │ O(n)    │
  │ Heap insert    │ O(1)    │ O(log n) │ O(log n)│
  └────────────────┴─────────┴──────────┴─────────┘
```

---

## 6. Practical Techniques for Complexity Analysis

### 6.1 Review of Logarithm Properties

```
Important properties of logarithms (frequently used in complexity analysis):

  1. log(a × b) = log(a) + log(b)
     → Multiplication inside loops can be decomposed into addition

  2. log(a / b) = log(a) - log(b)

  3. log(a^k) = k × log(a)
     → log(n²) = 2 × log(n) = O(log n)

  4. log(n!) = n × log(n) - n + O(log n) ≈ n × log(n)
     → Consequence of Stirling's approximation

  5. Base conversion: log_a(n) = log_b(n) / log_b(a)
     → In Big-O, the base is irrelevant: O(log₂n) = O(log₁₀n) = O(ln n)

  Practical reference values:
  - log₂(10) ≈ 3.32
  - log₂(100) ≈ 6.64
  - log₂(1000) ≈ 10
  - log₂(10⁶) ≈ 20
  - log₂(10⁹) ≈ 30
  - log₂(10¹⁸) ≈ 60

  → Even 1 billion items can be searched in just 30 steps with binary search!
```

### 6.2 Common Mistakes in Complexity Analysis

```
Error-prone areas:

  1. Confusing input size
  ┌──────────────────────────────────────┐
  │ ❌ "sort() is O(n log n)"           │
  │ → What is n? Array length? String    │
  │   length?                            │
  │                                      │
  │ ✅ "sort() is O(n log n) where n is  │
  │    the array length"                 │
  └──────────────────────────────────────┘

  2. Hash operation complexity
  ┌──────────────────────────────────────┐
  │ ❌ "Hash table operations are O(1)"  │
  │ → Expected O(1), worst case O(n)     │
  │ → Key comparison cost should be      │
  │   included (O(L) for string keys,    │
  │   L = string length)                 │
  └──────────────────────────────────────┘

  3. Forgetting recursive stack space
  ┌──────────────────────────────────────┐
  │ ❌ "DFS space complexity is O(1)"    │
  │ → Stack space O(V) is required for   │
  │   recursion depth                    │
  └──────────────────────────────────────┘

  4. Cost of string operations
  ┌──────────────────────────────────────┐
  │ ❌ s += "a" repeated n times → O(n)  │
  │ → In Python, strings are immutable   │
  │   so a new string is created each    │
  │   time: O(n²)!                       │
  │                                      │
  │ ✅ parts = []; parts.append("a") ×n  │
  │   → ''.join(parts) at the end: O(n)  │
  └──────────────────────────────────────┘

  5. List operation costs
  ┌──────────────────────────────────────┐
  │ Operation            │ Python list   │
  │ ──────────────────── │ ───────────── │
  │ append               │ O(1) amortized│
  │ pop()                │ O(1)          │
  │ pop(0) / insert(0,x) │ O(n)!         │
  │ x in list            │ O(n)          │
  │ list[i]              │ O(1)          │
  │ list.sort()          │ O(n log n)    │
  │ len(list)            │ O(1)          │
  │ list.copy()          │ O(n)          │
  │ list + list          │ O(n + m)      │
  └──────────────────────────────────────┘
```

### 6.3 Complexity of Data Structures by Language

```
Summary of complexity for major data structures:

  Python:
  ┌──────────────┬───────┬──────────┬──────────┬──────────┐
  │ Operation    │ list  │ dict     │ set      │ deque    │
  ├──────────────┼───────┼──────────┼──────────┼──────────┤
  │ Access       │ O(1)  │ O(1) exp.│ -        │ O(1)     │
  │ Search       │ O(n)  │ O(1) exp.│ O(1) exp.│ O(n)     │
  │ Insert front │ O(n)  │ -        │ -        │ O(1)     │
  │ Insert back  │ O(1)* │ O(1) exp.│ O(1) exp.│ O(1)     │
  │ Delete front │ O(n)  │ -        │ -        │ O(1)     │
  │ Delete back  │ O(1)  │ O(1) exp.│ O(1) exp.│ O(1)     │
  │ Sort         │O(nlogn)│ -       │ -        │ -        │
  └──────────────┴───────┴──────────┴──────────┴──────────┘
  * Amortized O(1)

  Java:
  ┌──────────────┬─────────┬──────────┬──────────┬──────────┐
  │ Operation    │ArrayList│ HashMap  │ TreeMap  │LinkedList│
  ├──────────────┼─────────┼──────────┼──────────┼──────────┤
  │ Access       │ O(1)    │ O(1) exp.│ O(log n) │ O(n)     │
  │ Search       │ O(n)    │ O(1) exp.│ O(log n) │ O(n)     │
  │ Insert       │ O(1)*   │ O(1) exp.│ O(log n) │ O(1)     │
  │ Delete       │ O(n)    │ O(1) exp.│ O(log n) │ O(1)     │
  └──────────────┴─────────┴──────────┴──────────┴──────────┘

  → Data structure choice can dramatically change complexity!
  → Choosing the right data structure is half of algorithm design
```

---

## 7. Practice Exercises

### Exercise 1: Determining Complexity (Basics)
Determine the time and space complexity of the following:
1. Code that prints all 2-element combinations of an array
2. Recursive binary search
3. Recursive Fibonacci computation (without vs. with memoization)
4. Triple-nested loops where the inner loop variable depends on the outer ones

### Exercise 2: Improving Complexity (Applied)
Improve an O(n^3) algorithm (3Sum problem using three arrays) to O(n^2).

### Exercise 3: Applying the Master Theorem (Applied)
Use the Master Theorem or the recursion tree method to determine the complexity of:
1. T(n) = 4T(n/2) + O(n)
2. T(n) = T(n/3) + T(2n/3) + O(n)
3. T(n) = 2T(n/2) + O(n log n)

### Exercise 4: Amortized Analysis (Advanced)
Prove that each operation of a queue implemented with two stacks has amortized O(1) complexity. Use the potential method.

### Exercise 5: Comparing Measured and Theoretical Results (Advanced)
Implement the following algorithms, measure execution time for varying input sizes, and compare the results with theoretical complexity. If differences exist, analyze their causes:
1. Bubble sort vs. merge sort vs. quick sort
2. Linear search vs. binary search vs. hash table lookup


---

## Troubleshooting

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| Initialization error | Configuration file issues | Verify configuration file path and format |
| Timeout | Network latency/insufficient resources | Adjust timeout values, add retry logic |
| Out of memory | Data volume growth | Introduce batch processing, implement pagination |
| Permission error | Insufficient access rights | Verify execution user permissions, review settings |
| Data inconsistency | Concurrent processing conflicts | Introduce locking, implement transaction management |

### Debugging Procedure

1. **Check the error message**: Read the stack trace to identify the location of the error
2. **Establish reproduction steps**: Reproduce the error with minimal code
3. **Formulate hypotheses**: List possible causes
4. **Verify step by step**: Use logging or a debugger to test hypotheses
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

Steps to diagnose performance problems:

1. **Identify the bottleneck**: Measure with profiling tools
2. **Check memory usage**: Verify the presence of memory leaks
3. **Check I/O waits**: Examine disk and network I/O conditions
4. **Check connection count**: Inspect connection pool status

| Problem type | Diagnostic tools | Countermeasures |
|-------------|-----------------|-----------------|
| CPU load | cProfile, py-spy | Algorithm improvement, parallelization |
| Memory leak | tracemalloc, objgraph | Proper reference release |
| I/O bottleneck | strace, iostat | Async I/O, caching |
| DB latency | EXPLAIN, slow query log | Indexing, query optimization |

---

## Design Decision Guide

### Selection Criteria Matrix

A summary of decision criteria for technology selection:

| Criterion | Prioritize when | Acceptable to compromise when |
|-----------|----------------|------------------------------|
| Performance | Real-time processing, large-scale data | Admin panels, batch processing |
| Maintainability | Long-term operation, team development | Prototypes, short-term projects |
| Scalability | Growing services | Internal tools, fixed user base |
| Security | Personal data, financial data | Public data, internal use |
| Development speed | MVP, time-to-market | Quality-first, mission-critical |

### Architecture Pattern Selection

```
┌─────────────────────────────────────────────────┐
│           Architecture Selection Flow            │
├─────────────────────────────────────────────────┤
│                                                 │
│  (1) Team size?                                 │
│    ├─ Small (1-5) → Monolith                    │
│    └─ Large (10+) → Go to (2)                   │
│                                                 │
│  (2) Deployment frequency?                      │
│    ├─ Weekly or less → Monolith + modules        │
│    └─ Daily/multiple → Go to (3)                │
│                                                 │
│  (3) Team independence?                         │
│    ├─ High → Microservices                      │
│    └─ Moderate → Modular monolith               │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Tradeoff Analysis

Technical decisions always involve tradeoffs. Analyze from the following perspectives:

**1. Short-term vs. long-term cost**
- A quick short-term solution may become technical debt long-term
- Conversely, over-engineering raises short-term costs and delays the project

**2. Consistency vs. flexibility**
- A unified tech stack lowers learning costs
- Diverse technologies enable best-fit choices but increase operational costs

**3. Level of abstraction**
- Higher abstraction improves reusability but can complicate debugging
- Lower abstraction is more intuitive but leads to code duplication

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
        """Describe background and problem"""
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
            icon = "✅" if c['type'] == 'positive' else "⚠️"
            md += f"- {icon} {c['description']}\n"
        md += "\n## Rejected Alternatives\n"
        for a in self.alternatives:
            md += f"- **{a['name']}**: {a['reason_rejected']}\n"
        return md
```

---

## Team Development

### Code Review Checklist

Points to check in code reviews related to this topic:

- [ ] Naming conventions are consistent
- [ ] Error handling is appropriate
- [ ] Test coverage is sufficient
- [ ] No performance impact
- [ ] No security concerns
- [ ] Documentation is updated

### Knowledge Sharing Best Practices

| Method | Frequency | Target | Effect |
|--------|-----------|--------|--------|
| Pair programming | As needed | Complex tasks | Immediate feedback |
| Tech talks | Weekly | Entire team | Horizontal knowledge transfer |
| ADR (Decision records) | Per decision | Future members | Decision transparency |
| Retrospectives | Biweekly | Entire team | Continuous improvement |
| Mob programming | Monthly | Important designs | Consensus building |

### Managing Technical Debt

```
Priority matrix:

        Impact: High
          │
    ┌─────┼─────┐
    │ Plan │ Fix │
    │ for  │ imme│
    │ later│ dia-│
    │      │ tely│
    ├─────┼─────┤
    │Record│ Next│
    │ only │Sprint│
    │      │     │
    └─────┼─────┘
          │
        Impact: Low
    Frequency: Low  Frequency: High
```

---

## Security Considerations

### Common Vulnerabilities and Countermeasures

| Vulnerability | Risk level | Countermeasure | Detection method |
|--------------|-----------|----------------|-----------------|
| Injection attacks | High | Input validation, parameterized queries | SAST/DAST |
| Authentication flaws | High | MFA, session management hardening | Penetration testing |
| Sensitive data exposure | High | Encryption, access control | Security audit |
| Misconfiguration | Medium | Security headers, principle of least privilege | Configuration scanning |
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

# Usage
token = SecurityUtils.generate_token()
hashed, salt = SecurityUtils.hash_password("my_password")
is_valid = SecurityUtils.verify_password("my_password", hashed, salt)
```

### Security Checklist

- [ ] All input values are validated
- [ ] Sensitive information is not output to logs
- [ ] HTTPS is enforced
- [ ] CORS policy is properly configured
- [ ] Dependency vulnerability scanning has been performed
- [ ] Error messages do not contain internal information

---

## Migration Guide

### Notes for Version Upgrades

| Version | Major changes | Migration work | Impact scope |
|---------|--------------|----------------|-------------|
| v1.x → v2.x | API design overhaul | Endpoint changes | All clients |
| v2.x → v3.x | Authentication method change | Token format update | Auth-related |
| v3.x → v4.x | Data model change | Run migration scripts | DB-related |

### Step-by-Step Migration Procedure

```python
# Migration script template
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Callable

logger = logging.getLogger(__name__)

class MigrationRunner:
    """Step-by-step migration execution engine"""

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
3. **Staged rollout**: Deploy gradually with canary releases
4. **Enhanced monitoring**: Shorten metric monitoring intervals during migration
5. **Clear decision criteria**: Define rollback decision criteria in advance

---

## Glossary

| Term | English | Description |
|------|---------|-------------|
| Abstraction | Abstraction | Hiding complex implementation details and exposing only essential interfaces |
| Encapsulation | Encapsulation | Bundling data and operations into a single unit and controlling external access |
| Cohesion | Cohesion | A measure of how closely related elements within a module are |
| Coupling | Coupling | The degree of interdependence between modules |
| Refactoring | Refactoring | Improving the internal structure of code without changing its external behavior |
| Test-Driven Development | TDD (Test-Driven Development) | An approach where tests are written before implementation |
| Continuous Integration | CI (Continuous Integration) | A practice of frequently integrating code changes and verifying with automated tests |
| Continuous Delivery | CD (Continuous Delivery) | A practice of maintaining a release-ready state at all times |
| Technical Debt | Technical Debt | Additional future work caused by choosing a short-term solution |
| Domain-Driven Design | DDD (Domain-Driven Design) | An approach to software design based on business domain knowledge |
| Microservices | Microservices | An architecture that builds applications as a collection of small independent services |
| Circuit Breaker | Circuit Breaker | A design pattern to prevent cascading failures |
| Event-Driven | Event-Driven | An architecture pattern based on event generation and processing |
| Idempotency | Idempotency | The property where performing the same operation multiple times yields the same result |
| Observability | Observability | The ability to observe a system's internal state from the outside |
---

## FAQ

### Q1: Can Big-O be used to accurately predict execution time?
**A**: No. Big-O only represents "growth rate" and ignores constant factors. An O(n) algorithm with a large constant can be slower than O(n log n). Actual performance depends on many factors including cache efficiency, branch prediction, and memory access patterns. Theoretical complexity serves as a guide for "which strategy to choose," but final performance should always be verified through measurement.

### Q2: Is O(1) always fast?
**A**: Not necessarily. O(1) only means "independent of input size" -- O(1) could mean one million constant operations. Also, the O(1) of hash tables is an "expected value"; worst case is O(n). Measurement is essential. For example, if the hash function computation itself is expensive, linear search may be faster for small n.

### Q3: Should I improve the complexity class or the constant factor?
**A**: Start with improving the complexity class. Going from O(n^2) to O(n log n) is dramatic. Constant factor improvements (cache optimization, etc.) should be considered after the complexity is optimal. However, for small n, constant factors can dominate. In practice, C++'s std::sort switches to insertion sort (O(n^2) but with a small constant) when n is small.

### Q4: When is complexity analysis needed?
**A**: (1) When there is a performance problem, (2) when evaluating scalability during system design, (3) when judging the appropriateness of algorithms during code review. For everyday coding, being able to intuitively judge "will this get slow as n increases?" is sufficient. However, degradation patterns like the N+1 problem (O(n) → O(n^2)) should always be kept in mind.

---

## Summary

| Concept | Key Points |
|---------|-----------|
| Big-O | Growth rate of execution time. Ignores constants and lower-order terms |
| Major classes | O(1) < O(log n) < O(√n) < O(n) < O(n log n) < O(n²) < O(2ⁿ) |
| How to determine | Loop count, recursion depth, Master Theorem, recursion tree method |
| Space complexity | Additional memory. "Tradeoff" with time |
| Amortized complexity | Operations that are individually expensive but average to O(1) |
| Practice | n ≤ 10⁶ → O(n log n) or better; APIs within 100ms |
| Optimization | Hash map, precomputation, two pointers, sliding window |

---

## Recommended Next Reading

---

## References
1. Cormen, T. H. et al. "Introduction to Algorithms (CLRS)." Chapter 3: Growth of Functions.
2. Skiena, S. S. "The Algorithm Design Manual." Chapter 2: Algorithm Analysis.
3. Sedgewick, R. & Wayne, K. "Algorithms." 4th Edition, Chapter 1.4.
4. Bentley, J. "Programming Pearls." 2nd Edition, Addison-Wesley, 2000.
5. Tarjan, R. E. "Amortized Computational Complexity." SIAM Journal on Algebraic Discrete Methods, 1985.

Complexity analysis is the foundation of efficient algorithm design.
