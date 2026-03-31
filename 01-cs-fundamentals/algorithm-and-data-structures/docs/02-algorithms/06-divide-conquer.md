# Divide and Conquer

> Understand the design technique of dividing a problem into smaller subproblems, solving them recursively, and combining the results -- through merge sort, large number multiplication, closest pair of points, and Strassen matrix multiplication

## Learning Objectives

1. **Understand the 3-step framework of divide and conquer** (divide, conquer, combine) and design algorithms recursively
2. **Accurately analyze the complexity of divide-and-conquer algorithms** using the **Master Theorem**
3. **Implement advanced divide-and-conquer algorithms** including **Karatsuba multiplication, closest pair of points, and Strassen matrix multiplication**
4. **Understand the differences between divide and conquer, DP, and greedy algorithms** and choose the appropriate technique for a given problem
5. **Develop intuition for complexity analysis** through **recursion tree drawing** and **recurrence derivation**


## Prerequisites

Before reading this guide, familiarity with the following will deepen your understanding:

- Basic programming knowledge
- Understanding of related fundamental concepts
- Familiarity with the content in [Greedy Algorithm](./05-greedy.md)

---

## 1. Principles of Divide and Conquer

### 1.1 The 3-Step Framework

Divide and conquer is one of the most powerful paradigms in algorithm design. As its name suggests, it consists of three steps: "divide," "conquer," and "combine."

```
+--------------------------------------------------------------+
|             The 3 Steps of Divide and Conquer                |
+--------------------------------------------------------------+
|                                                              |
|  Step 1. Divide                                              |
|     Split the problem into smaller subproblems of the        |
|     same kind                                                |
|     - Typically split into two (bisection)                   |
|     - Sometimes into three or more (Karatsuba: 3-way split) |
|                                                              |
|  Step 2. Conquer                                             |
|     Solve the subproblems recursively                        |
|     - If the subproblem is small enough, solve it directly   |
|       (base case)                                            |
|     - The design of the base case directly determines the    |
|       correctness of the entire algorithm                    |
|                                                              |
|  Step 3. Combine                                             |
|     Merge the solutions of the subproblems to form the       |
|     solution to the original problem                         |
|     - The efficiency of this step governs the overall        |
|       complexity                                             |
|     - e.g., merge in merge sort, strip processing in         |
|       closest pair                                           |
|                                                              |
+--------------------------------------------------------------+
|  Key difference from DP:                                     |
|  DP           -> Subproblems overlap -> Cache and reuse      |
|  Divide & Conq -> Subproblems are independent -> Solve       |
|                   each recursively as-is                     |
|                                                              |
|  However, exceptions exist:                                  |
|  - Fibonacci recursion tree has massive overlap              |
|    -> Should use DP (memoized recursion), not divide & conq  |
|  - Matrix chain multiplication also overlaps -> DP is proper |
+--------------------------------------------------------------+
```

### 1.2 Visualizing the Recursive Structure

Divide-and-conquer algorithms can be visualized as a recursion tree. Each node represents one subproblem, and its children represent the divided subproblems.

```
Recursion tree of divide and conquer (merge sort example):

Depth 0:           [38, 27, 43, 3, 9, 82, 10]         <- Problem size n
                 /                          \
Depth 1:    [38, 27, 43]                [3, 9, 82, 10]  <- Two subproblems of size n/2
           /       \                   /          \
Depth 2:  [38]    [27, 43]          [3, 9]      [82, 10] <- Four subproblems of size n/4
                /     \           /    \       /     \
Depth 3:        [27]   [43]       [3]   [9]   [82]   [10] <- Base cases

                    v Combine (merge) phase v

Depth 3:        [27]   [43]       [3]   [9]   [82]   [10]
                \     /           \    /       \     /
Depth 2:       [27, 43]           [3, 9]       [10, 82]
           \       /                   \          /
Depth 1:    [27, 38, 43]           [3, 9, 10, 82]
                 \                          /
Depth 0:          [3, 9, 10, 27, 38, 43, 82]            <- Final result

Recursion depth: O(log n)
Total work at each depth: O(n) (merge operation)
Overall complexity: O(n) x O(log n) = O(n log n)
```

### 1.3 Historical Background of Divide and Conquer

Divide and conquer is an age-old concept. Its name derives from the political maxim "divide et impera" (divide and rule). In the context of algorithm design, it can be traced back to John von Neumann's invention of merge sort in 1945.

Systematic study of divide and conquer in algorithm theory began in the 1960s. Karatsuba published a fast algorithm for large number multiplication in 1962, and Strassen demonstrated faster matrix multiplication in 1969, establishing divide and conquer as a powerful tool for fundamentally improving computational complexity.

### 1.4 Divide-and-Conquer Template

The following Python template shows the common skeleton shared by all divide-and-conquer algorithms.

```python
from typing import TypeVar, List, Callable, Any

T = TypeVar('T')

def divide_and_conquer(
    problem: T,
    base_case: Callable[[T], bool],
    solve_base: Callable[[T], Any],
    divide: Callable[[T], List[T]],
) -> Any:
    """Generic template for divide and conquer

    Args:
        problem: The problem to solve
        base_case: Function to check for the base case
        solve_base: Function to directly solve the base case
        divide: Function to split the problem into subproblems
        combine: Function to merge subproblem solutions

    Returns:
        The solution to the problem
    """
    # Base case: solve directly if the problem is small enough
    if base_case(problem):
        return solve_base(problem)

    # Divide: split the problem into smaller subproblems
    subproblems = divide(problem)

    # Conquer: solve each subproblem recursively
    subsolutions = [
        divide_and_conquer(sub, base_case, solve_base, divide, combine)
        for sub in subproblems
    ]

    # Combine: merge the subproblem solutions
    return combine(subsolutions)


# --- Usage example: find the maximum of an array ---
def find_max(arr: list) -> int:
    """Find the maximum of an array using divide and conquer"""
    return divide_and_conquer(
        problem=arr,
        base_case=lambda a: len(a) <= 1,
        solve_base=lambda a: a[0] if a else float('-inf'),
        divide=lambda a: [a[:len(a)//2], a[len(a)//2:]],
        combine=lambda results: max(results)
    )


# Verification
data = [3, 7, 2, 9, 1, 8, 5, 4, 6]
print(find_max(data))  # 9
```

---

## 2. Merge Sort -- The Textbook Example of Divide and Conquer

### 2.1 Algorithm Details

Merge sort is the most important algorithm for learning divide and conquer. Invented by John von Neumann in 1945, it has been the canonical example of divide and conquer in computer science textbooks ever since.

**Key properties:**
- Stable sort (preserves the order of equal elements)
- Worst-case complexity guaranteed at O(n log n) (unlike quicksort's worst-case O(n^2))
- Well-suited for external sorting (sorting data that doesn't fit in memory)
- Well-suited for sorting linked lists (requires almost no extra memory)

```python
def merge_sort(arr: list) -> list:
    """Merge sort - O(n log n)

    Divide: split the array in half
    Conquer: recursively sort each half
    Combine: merge the two sorted arrays

    A stable sort that guarantees O(n log n) even in the worst case.
    Requires O(n) additional memory.
    """
    # Base case: an array of 0 or 1 elements is already sorted
    if len(arr) <= 1:
        return arr

    # Divide: split at the midpoint
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])     # Conquer (recursively sort left half)
    right = merge_sort(arr[mid:])    # Conquer (recursively sort right half)

    # Combine: merge the two sorted arrays
    return merge(left, right)


def merge(left: list, right: list) -> list:
    """Merge two sorted arrays - O(n)

    Compare the heads of both arrays and append the smaller one to the result.
    When one is exhausted, append the remainder of the other.
    """
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:  # <= guarantees stability
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    # Append remaining elements
    result.extend(left[i:])
    result.extend(right[j:])
    return result


# --- Verification ---
data = [38, 27, 43, 3, 9, 82, 10]
sorted_data = merge_sort(data)
print(f"Input: {data}")
print(f"Output: {sorted_data}")
# Input: [38, 27, 43, 3, 9, 82, 10]
# Output: [3, 9, 10, 27, 38, 43, 82]

# Stability check (order preservation of equal elements)
pairs = [(3, 'a'), (1, 'b'), (3, 'c'), (2, 'd'), (1, 'e')]
sorted_pairs = merge_sort(pairs)  # Uses lexicographic comparison of tuples
print(f"Stable sort: {sorted_pairs}")
# [(1, 'b'), (1, 'e'), (2, 'd'), (3, 'a'), (3, 'c')]
# (1,'b') before (1,'e'), (3,'a') before (3,'c') -> stable
```

### 2.2 In-place Merge Sort

Standard merge sort requires O(n) additional memory. The following version reduces memory usage by allocating only one auxiliary array.

```python
def merge_sort_inplace(arr: list) -> None:
    """Merge sort using a single auxiliary array (near in-place)

    Modifies the original array in place.
    Allocates the auxiliary array aux once and reuses it for every merge.
    """
    if len(arr) <= 1:
        return

    aux = [0] * len(arr)  # Allocate auxiliary array once

    def _sort(lo: int, hi: int) -> None:
        """Sort arr[lo..hi]"""
        if lo >= hi:
            return

        mid = (lo + hi) // 2
        _sort(lo, mid)       # Sort left half
        _sort(mid + 1, hi)   # Sort right half

        # Optimization: skip merge if already sorted
        if arr[mid] <= arr[mid + 1]:
            return

        _merge(lo, mid, hi)

    def _merge(lo: int, mid: int, hi: int) -> None:
        """Merge arr[lo..mid] and arr[mid+1..hi]"""
        # Copy to auxiliary array
        for k in range(lo, hi + 1):
            aux[k] = arr[k]

        i, j = lo, mid + 1
        for k in range(lo, hi + 1):
            if i > mid:
                arr[k] = aux[j]; j += 1
            elif j > hi:
                arr[k] = aux[i]; i += 1
            elif aux[j] < aux[i]:
                arr[k] = aux[j]; j += 1
            else:
                arr[k] = aux[i]; i += 1

    _sort(0, len(arr) - 1)


# Verification
data = [38, 27, 43, 3, 9, 82, 10]
merge_sort_inplace(data)
print(data)  # [3, 9, 10, 27, 38, 43, 82]
```

### 2.3 Bottom-up Merge Sort

A non-recursive merge sort also exists. It starts with blocks of size 1 and iteratively merges adjacent blocks.

```python
def merge_sort_bottom_up(arr: list) -> list:
    """Bottom-up merge sort (non-recursive version)

    Merges blocks of increasing size: 1 -> 2 -> 4 -> 8 -> ...
    No recursion overhead and no risk of stack overflow.
    """
    n = len(arr)
    if n <= 1:
        return arr[:]

    result = arr[:]
    width = 1  # Current block size

    while width < n:
        for start in range(0, n, 2 * width):
            mid = min(start + width, n)
            end = min(start + 2 * width, n)

            left = result[start:mid]
            right = result[mid:end]
            merged = merge(left, right)  # Uses the merge function defined above
            result[start:start + len(merged)] = merged

        width *= 2

    return result


# Verification
data = [5, 3, 8, 1, 9, 2, 7, 4, 6]
print(merge_sort_bottom_up(data))  # [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

### 2.4 Merge Sort Trace

A step-by-step trace of merge sort, showing the divide and combine phases.

```
Input: [38, 27, 43, 3, 9, 82, 10]

=== Divide Phase ===
[38, 27, 43, 3, 9, 82, 10]
+-- [38, 27, 43]
|   +-- [38]            <- Base case
|   +-- [27, 43]
|       +-- [27]        <- Base case
|       +-- [43]        <- Base case
+-- [3, 9, 82, 10]
    +-- [3, 9]
    |   +-- [3]         <- Base case
    |   +-- [9]         <- Base case
    +-- [82, 10]
        +-- [82]        <- Base case
        +-- [10]        <- Base case

=== Combine Phase (bottom-up) ===
merge([27], [43])       -> [27, 43]
merge([38], [27,43])    -> [27, 38, 43]
merge([3], [9])         -> [3, 9]
merge([82], [10])       -> [10, 82]
merge([3,9], [10,82])   -> [3, 9, 10, 82]
merge([27,38,43], [3,9,10,82]) -> [3, 9, 10, 27, 38, 43, 82]

Result: [3, 9, 10, 27, 38, 43, 82]
```

---

## 3. The Master Theorem -- The Definitive Tool for Complexity Analysis

### 3.1 Definition of the Theorem

The complexity of divide-and-conquer algorithms is expressed as a recurrence relation. The Master Theorem directly solves recurrences of a specific form.

```
+------------------------------------------------------------------+
|                    Master Theorem                                 |
+------------------------------------------------------------------+
|                                                                  |
|  Recurrence: T(n) = a * T(n/b) + O(n^d)                         |
|                                                                  |
|  Parameters:                                                     |
|    a : number of subproblems (a >= 1)                            |
|    b : division ratio / each subproblem size = n/b (b > 1)       |
|    d : exponent of the combine cost (d >= 0)                     |
|                                                                  |
|  Comparison value: c = log_b(a) (represents the "weight" of     |
|                                  the recursion)                  |
|                                                                  |
|  +------------------------------------------------------------+ |
|  | Case 1: d < log_b(a)                                       | |
|  |   -> T(n) = O(n^{log_b(a)})                                | |
|  |   -> Recursive calls dominate (leaves do more work)         | |
|  |                                                             | |
|  | Case 2: d = log_b(a)                                       | |
|  |   -> T(n) = O(n^d * log n)                                 | |
|  |   -> Recursion and combine are balanced (equal work at each | |
|  |      level)                                                 | |
|  |                                                             | |
|  | Case 3: d > log_b(a)                                       | |
|  |   -> T(n) = O(n^d)                                         | |
|  |   -> Combine step dominates (root does more work)           | |
|  +------------------------------------------------------------+ |
|                                                                  |
|  Intuitive understanding:                                        |
|    - log_b(a) represents the growth rate of the number of leaves |
|    - d represents the growth rate of work at each level          |
|    - Compare these two to determine which dominates              |
+------------------------------------------------------------------+
```

### 3.2 Intuitive Understanding via Recursion Trees

Understanding the three cases of the Master Theorem through recursion trees.

```
Recursion tree structure (T(n) = a * T(n/b) + O(n^d)):

Depth 0:   n^d of work              <- 1 problem
          |
Depth 1:   a copies of (n/b)^d      <- a subproblems
          |
Depth 2:   a^2 copies of (n/b^2)^d
          |
  ...     ...
          |
Depth k:   a^k copies of (n/b^k)^d
          |
Depth log_b(n): a^{log_b(n)} = n^{log_b(a)} copies of O(1) work

Total work at each depth:

  Total at depth k = a^k * (n/b^k)^d = n^d * (a / b^d)^k

  Consider the ratio r = a / b^d:
    - r > 1 (i.e., d < log_b(a)): work increases with depth -> leaves dominate
    - r = 1 (i.e., d = log_b(a)): same work at each depth -> all levels equal
    - r < 1 (i.e., d > log_b(a)): shallower levels have more work -> root dominates
```

### 3.3 Concrete Examples

```python
import math

def analyze_master_theorem(a: int, b: int, d: float, name: str = "") -> str:
    """Complexity analysis using the Master Theorem

    Args:
        a: Number of subproblems
        b: Division ratio
        d: Exponent of the combine cost
        name: Algorithm name (for display)

    Returns:
        Analysis result as a string
    """
    log_b_a = math.log(a) / math.log(b)

    header = f"=== {name} ===" if name else "=== Analysis Result ==="
    lines = [
        header,
        f"  Recurrence: T(n) = {a}T(n/{b}) + O(n^{d})",
        f"  a = {a}, b = {b}, d = {d}",
        f"  log_{b}({a}) = {log_b_a:.4f}",
    ]

    if abs(d - log_b_a) < 1e-9:
        lines.append(f"  Case 2: d = log_b(a) = {log_b_a:.4f}")
        lines.append(f"  => T(n) = O(n^{d} * log n)")
    elif d < log_b_a:
        lines.append(f"  Case 1: d = {d} < log_b(a) = {log_b_a:.4f}")
        lines.append(f"  => T(n) = O(n^{log_b_a:.4f})")
    else:
        lines.append(f"  Case 3: d = {d} > log_b(a) = {log_b_a:.4f}")
        lines.append(f"  => T(n) = O(n^{d})")

    return "\n".join(lines)


# Analysis of representative algorithms
print(analyze_master_theorem(2, 2, 1, "Merge Sort"))
# Case 2: T(n) = O(n log n)

print()
print(analyze_master_theorem(1, 2, 0, "Binary Search"))
# Case 2: T(n) = O(log n)

print()
print(analyze_master_theorem(3, 2, 1, "Karatsuba Multiplication"))
# Case 1: T(n) = O(n^1.585)

print()
print(analyze_master_theorem(7, 2, 2, "Strassen Matrix Multiplication"))
# Case 1: T(n) = O(n^2.807)

print()
print(analyze_master_theorem(4, 2, 2, "Special Case"))
# Case 2: T(n) = O(n^2 log n)
```

### 3.4 Master Theorem Application Summary

| Algorithm | Recurrence | a | b | d | log_b(a) | Case | Complexity |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| Binary search | T(n) = T(n/2) + O(1) | 1 | 2 | 0 | 0 | 2 | O(log n) |
| Merge sort | T(n) = 2T(n/2) + O(n) | 2 | 2 | 1 | 1 | 2 | O(n log n) |
| Karatsuba | T(n) = 3T(n/2) + O(n) | 3 | 2 | 1 | 1.585 | 1 | O(n^1.585) |
| Strassen | T(n) = 7T(n/2) + O(n^2) | 7 | 2 | 2 | 2.807 | 1 | O(n^2.807) |
| Closest pair | T(n) = 2T(n/2) + O(n) | 2 | 2 | 1 | 1 | 2 | O(n log n) |
| Selection algorithm | T(n) = T(n/2) + O(n) | 1 | 2 | 1 | 0 | 3 | O(n) |
| Naive matrix mult. | T(n) = 8T(n/2) + O(n^2) | 8 | 2 | 2 | 3 | 1 | O(n^3) |

### 3.5 Cases Where the Master Theorem Does Not Apply

The Master Theorem is not universal. It cannot be applied to the following types of recurrences.

```
Non-applicable recurrences:

1. Unequal subproblem sizes
   T(n) = T(n/3) + T(2n/3) + O(n)
   -> Subproblem sizes differ: n/3 and 2n/3
   -> Use the Akra-Bazzi theorem or the recursion tree method

2. Not in the form T(n) = aT(n/b) + f(n)
   T(n) = T(n-1) + T(n-2)
   -> Reduction by a constant, not division
   -> Solve using the characteristic equation method

3. f(n) is not polynomial
   T(n) = 2T(n/2) + n/log(n)
   -> f(n) cannot be expressed as n^d
   -> Use the recursion tree method or substitution method
```

---

## 4. Karatsuba Multiplication -- Speeding Up Large Number Multiplication

### 4.1 The Problem with Standard Multiplication

When multiplying two n-digit numbers, the grade-school long multiplication algorithm requires O(n^2) time. This becomes a bottleneck when dealing with numbers of thousands of digits or more, as in cryptography and high-precision arithmetic.

In 1962, 23-year-old graduate student Anatolii Karatsuba discovered a method to reduce the complexity of multiplication to O(n^{log_2 3}) ~= O(n^{1.585}). This was a groundbreaking result demonstrating the power of divide and conquer.

### 4.2 Karatsuba's Trick

```
+--------------------------------------------------------------+
|                 Karatsuba's Idea                              |
+--------------------------------------------------------------+
|                                                              |
|  Split two n-digit numbers x, y into upper and lower halves: |
|    x = a * B^m + b    (B is the base, m = n/2)              |
|    y = c * B^m + d                                           |
|                                                              |
|  Naive computation:                                          |
|    x * y = ac * B^{2m} + (ad + bc) * B^m + bd               |
|    -> 4 multiplications: ac, ad, bc, bd                      |
|                                                              |
|  Karatsuba's trick:                                          |
|    p1 = a * c                                                |
|    p2 = b * d                                                |
|    p3 = (a + b) * (c + d) = ac + ad + bc + bd                |
|                                                              |
|    ad + bc = p3 - p1 - p2                                    |
|                                                              |
|    x * y = p1 * B^{2m} + (p3 - p1 - p2) * B^m + p2          |
|                                                              |
|  -> Multiplications reduced from 4 to 3!                     |
|    (Additions/subtractions increase but are much cheaper)     |
|                                                              |
|  Recurrence: T(n) = 3T(n/2) + O(n)                           |
|  Master Theorem Case 1: T(n) = O(n^{log_2 3}) ~ O(n^{1.585})|
+--------------------------------------------------------------+
```

### 4.3 Implementation

```python
def karatsuba(x: int, y: int) -> int:
    """Karatsuba multiplication algorithm - O(n^1.585)

    Improves large integer multiplication from naive O(n^2) to O(n^{1.585}).
    Recursively decomposes into 3 multiplications.
    """
    # Base case: multiply small numbers directly
    if x < 10 or y < 10:
        return x * y

    # Handle signs
    sign = 1
    if x < 0:
        x, sign = -x, -sign
    if y < 0:
        y, sign = -y, -sign

    # Equalize digit counts
    n = max(len(str(x)), len(str(y)))
    m = n // 2

    # Divide: x = a * 10^m + b, y = c * 10^m + d
    power = 10 ** m
    a, b = divmod(x, power)
    c, d = divmod(y, power)

    # 3 recursive multiplications (not 4!)
    p1 = karatsuba(a, c)           # a * c
    p2 = karatsuba(b, d)           # b * d
    p3 = karatsuba(a + b, c + d)   # (a+b) * (c+d)

    # Combine: x*y = p1 * 10^{2m} + (p3 - p1 - p2) * 10^m + p2
    result = p1 * (10 ** (2 * m)) + (p3 - p1 - p2) * power + p2

    return sign * result


# --- Verification ---
# Small numbers
assert karatsuba(1234, 5678) == 1234 * 5678  # 7006652
print(f"1234 * 5678 = {karatsuba(1234, 5678)}")

# Large numbers
big_x = 3141592653589793238462643383279
big_y = 2718281828459045235360287471352
assert karatsuba(big_x, big_y) == big_x * big_y
print(f"Large number verification: OK")

# Negative numbers
assert karatsuba(-123, 456) == -123 * 456
assert karatsuba(-123, -456) == (-123) * (-456)
print(f"Negative number verification: OK")

# Zero
assert karatsuba(0, 12345) == 0
print(f"Zero verification: OK")

# Performance comparison (conceptual)
# For n-digit x n-digit multiplication:
#   Naive: n^2 single-digit multiplications
#   Karatsuba: n^1.585 single-digit multiplications
# For n = 1000:
#   Naive: 1,000,000 operations
#   Karatsuba: ~38,000 operations -> ~26x faster
```

### 4.4 Multiplication Algorithms Beyond Karatsuba

Karatsuba opened the door to faster large number multiplication. Even faster algorithms have been discovered since then.

| Algorithm | Complexity | Year | Notes |
|:---|:---|:---:|:---|
| Long multiplication | O(n^2) | - | The grade-school method |
| Karatsuba | O(n^{1.585}) | 1962 | 3-way divide and conquer |
| Toom-Cook 3 | O(n^{1.465}) | 1963 | 5-way divide and conquer |
| Schonhage-Strassen | O(n log n log log n) | 1971 | FFT-based |
| Harvey-van der Hoeven | O(n log n) | 2019 | Reaches the theoretical lower bound |

---

## 5. Closest Pair of Points

### 5.1 Problem Definition and Applications

Given n points in the plane, find the pair of points with the minimum Euclidean distance.

**Application domains:**
- Collision detection (games, robotics)
- Clustering initialization
- Geographic information systems (nearest facility search)
- Molecular simulation (finding the nearest atom pair)

A naive brute-force approach checks all pairs, requiring O(n^2), but divide and conquer solves it in O(n log n).

### 5.2 Algorithm Details

```
+------------------------------------------------------------------+
|        Closest Pair Divide-and-Conquer Algorithm                  |
+------------------------------------------------------------------+
|                                                                  |
|  Preprocessing: sort all points by x-coordinate                  |
|                                                                  |
|  1. Divide: split into left and right at the median x-coordinate |
|                                                                  |
|     Left half      |  Right half                                  |
|     *     *        |  *                                           |
|        *           |     *                                        |
|     *      *       |                                              |
|          *         |  *                                           |
|     *              | *                                            |
|                    |                                              |
|             Midline x = mid_x                                     |
|                                                                  |
|  2. Conquer: recursively find closest pair in each half          |
|     delta_L = closest distance in left half                       |
|     delta_R = closest distance in right half                      |
|     delta   = min(delta_L, delta_R)                               |
|                                                                  |
|  3. Combine: check pairs crossing the boundary                   |
|     - Only examine points within a "strip" of width delta from   |
|       the midline                                                |
|     - Sort points in the strip by y-coordinate                   |
|     - For each point, compare only with points whose y-distance  |
|       is less than delta                                         |
|     - Theoretically, at most 7 points need to be compared per    |
|       point                                                      |
|     - Therefore strip processing is O(n) (O(n log n) with sort)  |
|                                                                  |
|   <-delta->|<-delta->                                             |
|     *      |  *      <- Only examine points inside the strip      |
|        *   |     *                                                |
|     *      |*                                                     |
|     <- strip ->                                                   |
|                                                                  |
|  Recurrence: T(n) = 2T(n/2) + O(n log n) -> O(n log^2 n)        |
|  Optimization: pre-sorting by y yields O(n log n)                |
+------------------------------------------------------------------+
```

### 5.3 Implementation

```python
import math
from typing import List, Tuple, Optional

Point = Tuple[float, float]

def closest_pair(points: List[Point]) -> Tuple[float, Point, Point]:
    """Find the closest pair of points using divide and conquer - O(n log^2 n)

    Args:
        points: List of points on a 2D plane [(x, y), ...]

    Returns:
        Tuple of (minimum distance, point1, point2)
    """
    def dist(p1: Point, p2: Point) -> float:
        """Euclidean distance between two points"""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def brute_force(pts: List[Point]) -> Tuple[float, Point, Point]:
        """Brute force for 3 or fewer points"""
        min_d = float('inf')
        best_p1, best_p2 = pts[0], pts[1]
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = dist(pts[i], pts[j])
                if d < min_d:
                    min_d = d
                    best_p1, best_p2 = pts[i], pts[j]
        return min_d, best_p1, best_p2

    def closest_in_strip(
        strip: List[Point], delta: float
    ) -> Tuple[float, Optional[Point], Optional[Point]]:
        """Find the closest pair within the strip

        In a strip sorted by y-coordinate, search for pairs with distance
        less than delta. At most 7 points need to be compared for each,
        so this runs in O(n).
        """
        min_d = delta
        best_p1, best_p2 = None, None

        # Sort by y-coordinate
        strip.sort(key=lambda p: p[1])

        for i in range(len(strip)):
            j = i + 1
            while j < len(strip) and (strip[j][1] - strip[i][1]) < min_d:
                d = dist(strip[i], strip[j])
                if d < min_d:
                    min_d = d
                    best_p1, best_p2 = strip[i], strip[j]
                j += 1

        return min_d, best_p1, best_p2

    def solve(pts_sorted_x: List[Point]) -> Tuple[float, Point, Point]:
        """Main body of divide and conquer"""
        n = len(pts_sorted_x)

        # Base case: brute force for 3 or fewer points
        if n <= 3:
            return brute_force(pts_sorted_x)

        # Divide
        mid = n // 2
        mid_x = pts_sorted_x[mid][0]
        left_half = pts_sorted_x[:mid]
        right_half = pts_sorted_x[mid:]

        # Conquer (recursion)
        dl, pl1, pl2 = solve(left_half)
        dr, pr1, pr2 = solve(right_half)

        # Choose the smaller result from left and right
        if dl <= dr:
            delta, best_p1, best_p2 = dl, pl1, pl2
        else:
            delta, best_p1, best_p2 = dr, pr1, pr2

        # Combine: check within the strip
        strip = [p for p in pts_sorted_x if abs(p[0] - mid_x) < delta]
        ds, ps1, ps2 = closest_in_strip(strip, delta)

        if ps1 is not None and ds < delta:
            return ds, ps1, ps2
        return delta, best_p1, best_p2

    # Preprocessing: sort by x-coordinate
    sorted_by_x = sorted(points, key=lambda p: p[0])
    return solve(sorted_by_x)


# --- Verification ---
points = [
    (2, 3), (12, 30), (40, 50), (5, 1), (12, 10), (3, 4)
]
distance, p1, p2 = closest_pair(points)
print(f"Closest pair: {p1}, {p2}")
print(f"Distance: {distance:.4f}")
# Closest pair: (2, 3), (3, 4)
# Distance: 1.4142

# Test with more points
import random
random.seed(42)
many_points = [(random.uniform(0, 1000), random.uniform(0, 1000)) for _ in range(100)]
d, p1, p2 = closest_pair(many_points)
print(f"\nTest with 100 points:")
print(f"Closest pair: ({p1[0]:.2f}, {p1[1]:.2f}), ({p2[0]:.2f}, {p2[1]:.2f})")
print(f"Distance: {d:.4f}")
```

---

## 6. Other Important Divide-and-Conquer Algorithms

### 6.1 Strassen Matrix Multiplication

Multiplying two n x n matrices naively costs O(n^3). In 1969, Volker Strassen discovered a way to reduce this to O(n^{2.807}) using divide and conquer.

**Core idea:**
Standard 2x2 block matrix multiplication requires 8 multiplications, but Strassen cleverly reduced this to 7 through algebraic rearrangement.

```
Naive 2x2 block matrix multiplication:

+       +   +       +   +                   +
| A   B | x | E   F | = | AE+BG    AF+BH    |
| C   D |   | G   H |   | CE+DG    CF+DH    |
+       +   +       +   +                   +

-> 8 matrix multiplications: AE, BG, AF, BH, CE, DG, CF, DH

Strassen's 7 multiplications:

  M1 = (A + D)(E + H)
  M2 = (C + D) E
  M3 = A (F - H)
  M4 = D (G - E)
  M5 = (A + B) H
  M6 = (C - A)(E + F)
  M7 = (B - D)(G + H)

Result:
  +                          +
  | M1+M4-M5+M7    M3+M5    |
  | M2+M4        M1-M2+M3+M6|
  +                          +

Recurrence: T(n) = 7T(n/2) + O(n^2)
Master Theorem Case 1: O(n^{log_2 7}) ~ O(n^{2.807})
```

```python
import numpy as np
from typing import Tuple

def strassen(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Strassen matrix multiplication - O(n^2.807)

    Computes the product of square matrices A and B using the Strassen
    algorithm. Zero-pads if the size is not a power of 2.
    """
    n = A.shape[0]

    # Base case: compute small matrices naively
    if n <= 64:
        return A @ B

    # Pad if size is odd
    if n % 2 != 0:
        A = np.pad(A, ((0, 1), (0, 1)))
        B = np.pad(B, ((0, 1), (0, 1)))
        result = strassen(A, B)
        return result[:n, :n]

    mid = n // 2

    # Split into 4 blocks
    A11, A12 = A[:mid, :mid], A[:mid, mid:]
    A21, A22 = A[mid:, :mid], A[mid:, mid:]
    B11, B12 = B[:mid, :mid], B[:mid, mid:]
    B21, B22 = B[mid:, :mid], B[mid:, mid:]

    # Strassen's 7 multiplications
    M1 = strassen(A11 + A22, B11 + B22)
    M2 = strassen(A21 + A22, B11)
    M3 = strassen(A11, B12 - B22)
    M4 = strassen(A22, B21 - B11)
    M5 = strassen(A11 + A12, B22)
    M6 = strassen(A21 - A11, B11 + B12)
    M7 = strassen(A12 - A22, B21 + B22)

    # Combine the results
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6

    # Assemble blocks
    C = np.zeros((n, n))
    C[:mid, :mid] = C11
    C[:mid, mid:] = C12
    C[mid:, :mid] = C21
    C[mid:, mid:] = C22

    return C


# --- Verification ---
np.random.seed(42)
n = 128
A = np.random.randint(0, 10, (n, n)).astype(float)
B = np.random.randint(0, 10, (n, n)).astype(float)

C_naive = A @ B
C_strassen = strassen(A, B)

print(f"Matrix size: {n}x{n}")
print(f"Max difference from naive: {np.max(np.abs(C_naive - C_strassen)):.10f}")
# Matrix size: 128x128
# Max difference from naive: 0.0000000000 (within floating-point range)
```

### 6.2 Fast Exponentiation (Repeated Squaring)

```python
def fast_power(base: int, exp: int, mod: int = None) -> int:
    """Repeated squaring - O(log n)

    Divide-and-conquer idea:
      base^exp = (base^{exp/2})^2         (if exp is even)
      base^exp = base * (base^{(exp-1)/2})^2  (if exp is odd)

    The recursive version is easy to understand but consumes stack.
    """
    if exp == 0:
        return 1
    if exp == 1:
        return base % mod if mod else base

    if exp % 2 == 0:
        half = fast_power(base, exp // 2, mod)
        result = half * half
    else:
        half = fast_power(base, (exp - 1) // 2, mod)
        result = base * half * half

    return result % mod if mod else result


def fast_power_iterative(base: int, exp: int, mod: int = None) -> int:
    """Repeated squaring (iterative version) - O(log n)

    Preferred in practice. No risk of stack overflow.
    Uses the bit representation of exp.

    Example: base^13 = base^(1101_2) = base^8 * base^4 * base^1
    """
    result = 1
    if mod:
        base %= mod

    while exp > 0:
        # If the least significant bit of exp is 1, multiply into result
        if exp & 1:
            result *= base
            if mod:
                result %= mod
        # Square the base
        exp >>= 1
        base *= base
        if mod:
            base %= mod

    return result


# --- Verification ---
print(fast_power(2, 30))              # 1073741824
print(fast_power(2, 30, 10**9 + 7))   # 1073741824
print(fast_power(3, 100, 10**9 + 7))  # 981453966

# Consistency check with iterative version
for b in range(2, 10):
    for e in range(0, 50):
        assert fast_power(b, e, 997) == fast_power_iterative(b, e, 997)
print("All tests passed")
```

### 6.3 Maximum Subarray Sum (Divide-and-Conquer Version)

```python
def max_subarray_dc(arr: list) -> tuple:
    """Maximum subarray sum - divide-and-conquer version O(n log n)

    Divide: split the array into left and right halves
    Conquer: find the maximum subarray sum in each half
    Combine: find the maximum subarray sum crossing the midpoint,
             return the maximum of the three

    Kadane's algorithm is faster at O(n), but this is an important
    exercise in divide and conquer.

    Returns:
        (maximum sum, start index, end index)
    """
    def solve(lo: int, hi: int) -> tuple:
        # Base case
        if lo == hi:
            return arr[lo], lo, hi

        mid = (lo + hi) // 2

        # Maximum subarray sum in the left half
        left_max, ll, lr = solve(lo, mid)

        # Maximum subarray sum in the right half
        right_max, rl, rr = solve(mid + 1, hi)

        # Maximum subarray sum crossing the midpoint
        # Extend left from the center
        left_sum = float('-inf')
        total = 0
        cross_l = mid
        for i in range(mid, lo - 1, -1):
            total += arr[i]
            if total > left_sum:
                left_sum = total
                cross_l = i

        # Extend right from the center
        right_sum = float('-inf')
        total = 0
        cross_r = mid + 1
        for i in range(mid + 1, hi + 1):
            total += arr[i]
            if total > right_sum:
                right_sum = total
                cross_r = i

        cross_max = left_sum + right_sum

        # Choose the maximum among three candidates
        if left_max >= right_max and left_max >= cross_max:
            return left_max, ll, lr
        elif right_max >= left_max and right_max >= cross_max:
            return right_max, rl, rr
        else:
            return cross_max, cross_l, cross_r

    if not arr:
        return 0, -1, -1

    return solve(0, len(arr) - 1)


# --- Verification ---
data = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
max_sum, start, end = max_subarray_dc(data)
print(f"Maximum subarray sum: {max_sum}")
print(f"Subarray: {data[start:end+1]} (indices {start}..{end})")
# Maximum subarray sum: 6
# Subarray: [4, -1, 2, 1] (indices 3..6)
```

### 6.4 Median of Medians

An algorithm that finds the k-th smallest element in worst-case O(n). It uses divide and conquer for pivot selection to prevent the worst case of quickselect.

```python
def median_of_medians(arr: list, k: int) -> int:
    """Median of medians algorithm for selecting the k-th smallest element

    Worst-case complexity: O(n)
    Recurrence: T(n) = T(n/5) + T(7n/10) + O(n)

    Args:
        arr: List of numbers
        k: Desired rank (0-indexed)

    Returns:
        The k-th smallest element
    """
    if len(arr) <= 5:
        return sorted(arr)[k]

    # Step 1: Divide into groups of 5, find the median of each group
    medians = []
    for i in range(0, len(arr), 5):
        group = sorted(arr[i:i + 5])
        medians.append(group[len(group) // 2])

    # Step 2: Recursively find the median of medians (use as pivot)
    pivot = median_of_medians(medians, len(medians) // 2)

    # Step 3: 3-way partition around the pivot
    low = [x for x in arr if x < pivot]
    equal = [x for x in arr if x == pivot]
    high = [x for x in arr if x > pivot]

    # Step 4: Determine which partition k belongs to, recurse
    if k < len(low):
        return median_of_medians(low, k)
    elif k < len(low) + len(equal):
        return pivot
    else:
        return median_of_medians(high, k - len(low) - len(equal))


# --- Verification ---
data = [3, 7, 2, 9, 1, 8, 5, 4, 6, 10]
for k in range(len(data)):
    result = median_of_medians(data[:], k)
    expected = sorted(data)[k]
    assert result == expected, f"k={k}: got {result}, expected {expected}"
    print(f"  {k}-th smallest element: {result}")

# Retrieve the median
median = median_of_medians(data[:], len(data) // 2)
print(f"\nMedian: {median}")  # 6
```

---

## 7. When to Apply Divide and Conquer -- Comparing Design Paradigms

### 7.1 Characteristics of Problems Suited for Divide and Conquer

```
Divide-and-conquer applicability flowchart:

Can the problem be divided into smaller subproblems?
|
+-- NO -> Consider other techniques (greedy, search, mathematical methods, etc.)
|
+-- YES -> Are the subproblems independent (no overlap)?
          |
          +-- NO -> Consider dynamic programming (DP)
          |       e.g., Fibonacci sequence, LCS, matrix chain multiplication
          |
          +-- YES -> Is the combine step efficient (O(n) or less)?
                    |
                    +-- NO -> Re-examine whether division improves complexity
                    |       If combining is O(n^2), division may be pointless
                    |
                    +-- YES -> Divide and conquer is effective!
                              Is the division balanced?
                              |
                              +-- YES -> Analyzable with the Master Theorem
                              +-- NO  -> Analyze with the recursion tree method
```

### 7.2 Comprehensive Comparison of Design Paradigms

| Property | Divide & Conquer | Dynamic Programming (DP) | Greedy | Backtracking |
|:---|:---|:---|:---|:---|
| Subproblem relationship | Independent | Overlapping | Independent | Independent (search tree) |
| Solution construction | Recursion + combine | Table filling | Sequential decisions | Trial + undo |
| Guaranteed optimality | Problem-dependent | Always optimal | Optimal if greedy-choice property holds | Always optimal (exhaustive) |
| Typical complexity | O(n log n) | O(n^2) ~ O(nW) | O(n log n) | O(2^n) ~ O(n!) |
| Space complexity | O(log n) ~ O(n) | O(n) ~ O(n^2) | O(1) ~ O(n) | O(n) |
| Representative problem | Merge sort | Longest common subsequence | Activity selection | N-Queens |
| Backtracking | None | None | None | Yes |
| Subproblem size | n/b (shrinks by ratio) | Shrinks by 1 | Shrinks by 1 | Shrinks by 1 |

### 7.3 The Boundary Between Divide-and-Conquer and DP -- Understanding Through Concrete Examples

Even for the same problem, we choose between divide and conquer and DP depending on the structure of the subproblems.

```python
# --- Example: Matrix exponentiation ---
# A DP approach (memoization) is unnecessary -> divide and conquer is optimal

def matrix_power(M: list, n: int) -> list:
    """Compute the n-th power of a matrix using divide and conquer - O(k^3 log n)

    k is the matrix size. Subproblems are independent, so divide and
    conquer is appropriate.
    """
    size = len(M)

    def identity(size: int) -> list:
        return [[1 if i == j else 0 for j in range(size)] for i in range(size)]

    def mat_mult(A: list, B: list) -> list:
        size = len(A)
        C = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                for k_idx in range(size):
                    C[i][j] += A[i][k_idx] * B[k_idx][j]
        return C

    if n == 0:
        return identity(size)
    if n == 1:
        return [row[:] for row in M]

    if n % 2 == 0:
        half = matrix_power(M, n // 2)
        return mat_mult(half, half)
    else:
        return mat_mult(M, matrix_power(M, n - 1))


# --- Application: computing Fibonacci numbers in O(log n) ---
def fibonacci_matrix(n: int) -> int:
    """Compute Fibonacci numbers via matrix exponentiation - O(log n)

    [[F(n+1), F(n)], [F(n), F(n-1)]] = [[1,1],[1,0]]^n
    """
    if n <= 1:
        return n

    M = [[1, 1], [1, 0]]
    result = matrix_power(M, n)
    return result[0][1]


# Verification
for i in range(15):
    print(f"F({i}) = {fibonacci_matrix(i)}", end="  ")
# F(0)=0 F(1)=1 F(2)=1 F(3)=2 F(4)=3 F(5)=5 F(6)=8 ...

print()
print(f"F(50) = {fibonacci_matrix(50)}")   # 12586269025
print(f"F(100) = {fibonacci_matrix(100)}") # 354224848179261915075
```

---

## 8. Anti-patterns and Common Pitfalls

### Anti-pattern 1: Unbalanced Division Causing Complexity Degradation

The most important aspect of divide and conquer is "balanced division." If the division is skewed, the recursion tree depth degrades to O(n), and the expected complexity is not achieved.

```python
# ============================================================
# BAD: Unbalanced division -> worst-case O(n^2) degradation
# ============================================================
def bad_quicksort(arr: list) -> list:
    """Worst-case quicksort

    Choosing the minimum as the pivot leads to a 1 vs (n-1) split.
    Recursion depth becomes O(n), and overall is O(n^2).

    Recurrence: T(n) = T(n-1) + O(n) -> O(n^2)
    """
    if len(arr) <= 1:
        return arr
    pivot = min(arr)  # Always picks the minimum as pivot -> worst split
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return bad_quicksort(left) + middle + bad_quicksort(right)


# ============================================================
# GOOD: Merge sort guarantees balanced division
# ============================================================
def good_merge_sort(arr: list) -> list:
    """Merge sort always guarantees balanced division

    Always splits at the midpoint, so recursion depth is stable at O(log n).
    Recurrence: T(n) = 2T(n/2) + O(n) -> O(n log n) is always guaranteed.
    """
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = good_merge_sort(arr[:mid])
    right = good_merge_sort(arr[mid:])
    return merge(left, right)  # Uses the merge function defined earlier


# ============================================================
# BETTER: Randomization achieves balanced division on average
# ============================================================
import random

def randomized_quicksort(arr: list) -> list:
    """Randomized quicksort

    By choosing the pivot randomly, achieves expected O(n log n) complexity.
    Probability of the worst case is 1/n!, virtually impossible.
    """
    if len(arr) <= 1:
        return arr
    pivot = random.choice(arr)  # Randomly select the pivot
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return randomized_quicksort(left) + middle + randomized_quicksort(right)


# Verify the impact of unbalanced division
sorted_input = list(range(1000))
# bad_quicksort(sorted_input)  # Risk of RecursionError!
print(f"Randomized QS: {randomized_quicksort(sorted_input)[:10]}...")  # Works correctly
```

```
Recursion tree of unbalanced division (worst case):

T(n)
+-- T(0)     <- empty
+-- T(n-1)
    +-- T(0)
    +-- T(n-2)
        +-- T(0)
        +-- T(n-3)
            +-- ...
                +-- T(1)

Depth: n
Work at each level: O(n), O(n-1), O(n-2), ...
Total: O(n^2)  <- much worse than merge sort's O(n log n)


Recursion tree of balanced division (ideal case):

              T(n)
         /          \
     T(n/2)        T(n/2)
     /    \         /    \
  T(n/4) T(n/4) T(n/4) T(n/4)
  ...     ...    ...     ...

Depth: log n
Work at each level: O(n) (total)
Total: O(n log n)
```

### Anti-pattern 2: Applying Divide and Conquer to Overlapping Subproblems

```python
# ============================================================
# BAD: Naive divide and conquer for Fibonacci -> O(2^n) exponential explosion
# ============================================================
call_count = 0

def fib_bad(n: int) -> int:
    """Naive recursive Fibonacci - O(2^n)

    Computing fib(5) alone calls fib(2) three times and fib(3) twice.
    Subproblems overlap massively -- divide and conquer is inappropriate.
    """
    global call_count
    call_count += 1

    if n <= 1:
        return n
    return fib_bad(n - 1) + fib_bad(n - 2)


call_count = 0
result = fib_bad(20)
print(f"fib_bad(20) = {result}, function calls: {call_count}")
# fib_bad(20) = 6765, function calls: 21891


# ============================================================
# GOOD: Memoized recursion (DP) -> O(n)
# ============================================================
from functools import lru_cache

@lru_cache(maxsize=None)
def fib_dp(n: int) -> int:
    """Memoized Fibonacci - O(n)

    Never recomputes the same subproblem.
    """
    if n <= 1:
        return n
    return fib_dp(n - 1) + fib_dp(n - 2)

print(f"fib_dp(20) = {fib_dp(20)}")   # 6765
print(f"fib_dp(100) = {fib_dp(100)}") # 354224848179261915075


# ============================================================
# BEST: Matrix exponentiation -> O(log n) <- divide and conquer works properly
# ============================================================
print(f"fibonacci_matrix(20) = {fibonacci_matrix(20)}")   # 6765
print(f"fibonacci_matrix(100) = {fibonacci_matrix(100)}") # 354224848179261915075
```

### Anti-pattern 3: Incomplete Base Cases

```python
# ============================================================
# BAD: Incomplete base case -> infinite recursion
# ============================================================
def bad_binary_search(arr: list, target: int, lo: int, hi: int) -> int:
    """Binary search with a flawed base case"""
    mid = (lo + hi) // 2
    # BUG: Missing lo > hi check -> infinite recursion when element doesn't exist
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return bad_binary_search(arr, target, mid + 1, hi)
    else:
        return bad_binary_search(arr, target, lo, mid - 1)


# ============================================================
# GOOD: Correct base case
# ============================================================
def good_binary_search(arr: list, target: int, lo: int, hi: int) -> int:
    """Correct binary search"""
    if lo > hi:  # Base case: element not found
        return -1
    mid = (lo + hi) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return good_binary_search(arr, target, mid + 1, hi)
    else:
        return good_binary_search(arr, target, lo, mid - 1)


# Verification
arr = [1, 3, 5, 7, 9, 11, 13]
print(good_binary_search(arr, 7, 0, len(arr) - 1))   # 3
print(good_binary_search(arr, 6, 0, len(arr) - 1))   # -1 (not found)
```

### Anti-pattern Decision Criteria Summary

| Anti-pattern | Symptom | Cause | Remedy |
|:---|:---|:---|:---|
| Unbalanced division | Complexity degrades to O(n^2) | Poor pivot selection | Randomization or median of medians |
| Overlapping subproblems | Exponential complexity | Misidentifying subproblem independence | Switch to DP (memoization or bottom-up) |
| Incomplete base cases | Infinite recursion / stack overflow | Missing termination conditions | Enumerate all terminal cases |
| Overlooked combine cost | Slower than expected | Combine is O(n^2) | Optimize combining or consider alternative methods |

---

## 9. Exercises (3 Levels)

### Foundation Level

**Exercise 1: Inversion Count**

Count the number of inversions in an array using divide and conquer in O(n log n). An inversion is a pair (i, j) where i < j and arr[i] > arr[j].

```python
def count_inversions(arr: list) -> tuple:
    """Count inversions using divide and conquer - O(n log n)

    During the merge step of merge sort, the number of times an element
    from the right is merged before one from the left gives the
    inversion count.

    Returns:
        (sorted array, inversion count)
    """
    if len(arr) <= 1:
        return arr[:], 0

    mid = len(arr) // 2
    left, left_inv = count_inversions(arr[:mid])
    right, right_inv = count_inversions(arr[mid:])

    merged = []
    inversions = left_inv + right_inv
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            # right[j] is smaller than all remaining left[i..] -> inversions
            merged.append(right[j])
            inversions += len(left) - i
            j += 1

    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged, inversions


# --- Verification ---
test_cases = [
    ([1, 2, 3, 4, 5], 0),       # Already sorted -> 0 inversions
    ([5, 4, 3, 2, 1], 10),      # Reversed -> C(5,2) = 10
    ([2, 4, 1, 3, 5], 3),       # (2,1), (4,1), (4,3)
    ([1, 20, 6, 4, 5], 5),      # (20,6), (20,4), (20,5), (6,4), (6,5)
]

for arr, expected in test_cases:
    _, inv = count_inversions(arr)
    status = "OK" if inv == expected else "NG"
    print(f"  {arr} -> inversions {inv} (expected {expected}) [{status}]")
```

**Exercise 2: Power Sum**

Compute 1^k + 2^k + ... + n^k efficiently using divide-and-conquer thinking (hint: direct computation is O(n log k), but the divide-and-conquer perspective helps understand the problem structure).

```python
def power_sum(n: int, k: int, mod: int = 10**9 + 7) -> int:
    """Compute 1^k + 2^k + ... + n^k

    Each term uses repeated squaring O(log k),
    giving an overall complexity of O(n log k).

    Divide-and-conquer decomposition:
      S(n, k) = S(n/2, k) + sum(i^k for i in range(n/2+1, n+1))
      However, in this case direct computation is simpler.
    """
    total = 0
    for i in range(1, n + 1):
        total = (total + pow(i, k, mod)) % mod
    return total


# Verification
print(f"1^2 + 2^2 + ... + 10^2 = {power_sum(10, 2)}")  # 385
print(f"1^3 + 2^3 + ... + 10^3 = {power_sum(10, 3)}")  # 3025
# Check: sum(i**2 for i in range(1,11)) = 385
# Check: sum(i**3 for i in range(1,11)) = 3025
```

### Intermediate Level

**Exercise 3: Majority Element**

Return the majority element in an array (appearing more than half the time), or None if none exists. Solve it in O(n log n) using divide and conquer.

```python
def majority_element(arr: list) -> int:
    """Find the majority element using divide and conquer - O(n log n)

    The Boyer-Moore voting algorithm is O(n), but we implement this as
    a divide-and-conquer exercise.

    Idea:
    - Split the array into left and right halves
    - Recursively find the majority candidate in each half
    - Count each candidate in the full range to determine the answer
    """
    def solve(lo: int, hi: int):
        # Base case: single element
        if lo == hi:
            return arr[lo]

        mid = (lo + hi) // 2
        left_maj = solve(lo, mid)
        right_maj = solve(mid + 1, hi)

        # If both majorities agree, it's confirmed
        if left_maj == right_maj:
            return left_maj

        # If they differ, count each in the full range
        left_count = sum(1 for i in range(lo, hi + 1) if arr[i] == left_maj)
        right_count = sum(1 for i in range(lo, hi + 1) if arr[i] == right_maj)

        threshold = (hi - lo + 1) // 2 + 1
        if left_count >= threshold:
            return left_maj
        if right_count >= threshold:
            return right_maj
        return None

    if not arr:
        return None

    result = solve(0, len(arr) - 1)

    # Final verification
    if result is not None and arr.count(result) > len(arr) // 2:
        return result
    return None


# --- Verification ---
print(majority_element([3, 3, 4, 2, 3, 3, 3]))  # 3
print(majority_element([1, 2, 3, 4, 5]))          # None
print(majority_element([2, 2, 1, 1, 1, 2, 2]))    # 2
```

**Exercise 4: k-th Smallest Element from Two Sorted Arrays**

Given two sorted arrays A and B, find the k-th smallest element in O(log(min(m, n))) as if they were merged.

```python
def kth_of_two_sorted(A: list, B: list, k: int) -> int:
    """k-th smallest element from two sorted arrays - O(log(min(m,n)))

    Divide-and-conquer idea:
    - Consider taking i elements from A and j elements from B (i + j = k)
    - If A[i-1] <= B[j] and B[j-1] <= A[i],
      then max(A[i-1], B[j-1]) is the answer

    Args:
        A, B: Sorted arrays
        k: Desired rank (1-indexed)
    """
    # Ensure A is the shorter one
    if len(A) > len(B):
        return kth_of_two_sorted(B, A, k)

    m, n = len(A), len(B)

    # Range of i: take at least 0, at most min(m, k) from A
    lo = max(0, k - n)
    hi = min(m, k)

    while lo <= hi:
        i = (lo + hi) // 2  # Take i elements from A
        j = k - i            # Take j elements from B

        a_left = A[i - 1] if i > 0 else float('-inf')
        b_left = B[j - 1] if j > 0 else float('-inf')
        a_right = A[i] if i < m else float('inf')
        b_right = B[j] if j < n else float('inf')

        if a_left <= b_right and b_left <= a_right:
            return max(a_left, b_left)
        elif a_left > b_right:
            hi = i - 1
        else:
            lo = i + 1

    raise ValueError("Invalid input")


# --- Verification ---
A = [1, 3, 5, 7, 9]
B = [2, 4, 6, 8, 10]
for k in range(1, 11):
    result = kth_of_two_sorted(A, B, k)
    print(f"  k={k}: {result}", end="")
print()
# k=1: 1, k=2: 2, ..., k=10: 10
```

### Advanced Level

**Exercise 5: Fast Fourier Transform (FFT) for Polynomial Multiplication**

The FFT, which computes the product of two polynomials in O(n log n), represents the pinnacle of divide and conquer.

```python
import cmath
from typing import List

def fft(a: List[complex], invert: bool = False) -> List[complex]:
    """Fast Fourier Transform (Cooley-Tukey FFT) - O(n log n)

    Divide-and-conquer application:
    - Split into even-indexed and odd-indexed coefficients
    - Recursively apply FFT to each
    - Combine via butterfly operations

    Args:
        a: Polynomial coefficients (length must be a power of 2)
        invert: If True, compute the inverse FFT

    Returns:
        FFT-transformed coefficients
    """
    n = len(a)
    if n == 1:
        return a[:]

    # Split into even-indexed and odd-indexed
    even = fft(a[0::2], invert)
    odd = fft(a[1::2], invert)

    # Twiddle factor
    angle = 2 * cmath.pi / n * (-1 if invert else 1)
    w = 1
    wn = cmath.exp(1j * angle)

    result = [0] * n
    for i in range(n // 2):
        result[i] = even[i] + w * odd[i]
        result[i + n // 2] = even[i] - w * odd[i]
        if invert:
            result[i] /= 2
            result[i + n // 2] /= 2
        w *= wn

    return result


def polynomial_multiply(a: List[int], b: List[int]) -> List[int]:
    """Polynomial multiplication via FFT - O(n log n)

    Standard polynomial multiplication is O(n^2), but FFT reduces it to O(n log n).

    Procedure:
    1. Coefficient representation -> point-value representation (FFT)
    2. Pointwise multiplication O(n)
    3. Point-value representation -> coefficient representation (inverse FFT)

    Args:
        a, b: Coefficient lists of polynomials (a[i] is the coefficient of x^i)

    Returns:
        Coefficient list of the product polynomial
    """
    result_len = len(a) + len(b) - 1

    # Extend size to a power of 2
    n = 1
    while n < result_len:
        n <<= 1

    fa = [complex(x) for x in a] + [0] * (n - len(a))
    fb = [complex(x) for x in b] + [0] * (n - len(b))

    # Transform to point-value representation via FFT
    fa = fft(fa)
    fb = fft(fb)

    # Pointwise multiplication
    fc = [fa[i] * fb[i] for i in range(n)]

    # Transform back to coefficient representation via inverse FFT
    fc = fft(fc, invert=True)

    # Extract real parts and round to integers
    result = [round(c.real) for c in fc[:result_len]]
    return result


# --- Verification ---
# (1 + 2x + 3x^2) * (4 + 5x) = 4 + 13x + 22x^2 + 15x^3
a = [1, 2, 3]
b = [4, 5]
product = polynomial_multiply(a, b)
print(f"({a}) * ({b}) = {product}")
# [4, 13, 22, 15]

# Verification: 123 * 45 = 5535
# (3 + 2*10 + 1*100) * (5 + 4*10) = 5535
a2 = [3, 2, 1]
b2 = [5, 4]
p2 = polynomial_multiply(a2, b2)
value = sum(c * (10 ** i) for i, c in enumerate(p2))
print(f"123 * 45 = {value}")  # 5535
```

---

## 10. FAQ (Frequently Asked Questions)

### Q1: What is the difference between divide and conquer and recursion?

**A:** Recursion is an **implementation technique** (a function calling itself), while divide and conquer is a **design paradigm** (a strategy of dividing, conquering, and combining).

Divide and conquer is typically implemented using recursion. However, not every recursive algorithm is divide and conquer. For example, DFS (depth-first search) is implemented recursively but does not have the "divide the problem and combine the results" structure, so it is not called divide and conquer.

Conversely, the divide-and-conquer approach can also be implemented iteratively, as in bottom-up merge sort.

| | Recursion | Divide and Conquer |
|:---|:---|:---|
| Category | Implementation technique | Design paradigm |
| Definition | A function calls itself | 3-step process: divide, conquer, combine |
| Examples | DFS, factorial, Tower of Hanoi | Merge sort, Karatsuba, FFT |
| Relationship | Used to implement divide and conquer | Usually implemented with recursion |

### Q2: How do you handle recurrences that the Master Theorem cannot solve?

**A:** When the Master Theorem does not apply, there are three alternative methods.

1. **Recursion Tree Method:** Draw the recursion tree and sum the work at each level. Intuitive and versatile.

2. **Substitution Method:** Guess the solution and prove it by mathematical induction. Rigorous but requires a correct guess.

3. **Akra-Bazzi Theorem:** Applies to recurrences of the form T(n) = sum(a_i * T(n/b_i)) + f(n) with unequal subproblem sizes. A generalization of the Master Theorem.

```
Example: Analysis of T(n) = T(n/3) + T(2n/3) + O(n)

Recursion tree method:
  Depth 0: n              work = n
  Depth 1: n/3 + 2n/3     work = n
  Depth 2: n/9 + 2n/9 + 2n/9 + 4n/9  work = n
  ...
  Depth k: work = n (same at every depth)

  Max depth: log_{3/2}(n) ~ 1.71 log n (following the 2n/3 path)
  Min depth: log_3(n)

  -> T(n) = O(n log n)
```

### Q3: Is Strassen matrix multiplication practical?

**A:** Strassen's algorithm is theoretically O(n^{2.807}), which is asymptotically faster than the naive O(n^3). However, in practice its use is limited for the following reasons.

1. **Large constant factor:** Strassen requires 18 additions/subtractions, making the naive method faster for small matrices. Generally, Strassen only becomes advantageous for n > several hundred.

2. **Numerical stability:** The many additions/subtractions accumulate rounding errors in floating-point arithmetic. This is problematic in scientific computing.

3. **Cache efficiency:** Modern CPUs have cache hierarchies. Libraries like BLAS (Basic Linear Algebra Subprograms) implement cache-optimized block matrix multiplication that is often faster than Strassen.

4. **Rise of GPUs:** GPUs specialize in massive parallelism and can execute naive matrix multiplication extremely fast.

Conclusion: Strassen is theoretically important but in practice, BLAS libraries or GPUs are more commonly used. However, it can be useful for very large matrices (several thousand x several thousand or more) or integer matrices.

### Q4: Is divide and conquer well-suited for parallelization?

**A:** Divide and conquer is very well-suited for parallelization. Since the subproblems are independent, each can be processed simultaneously on different processors or threads.

```python
from concurrent.futures import ThreadPoolExecutor

def parallel_merge_sort(arr: list, threshold: int = 10000) -> list:
    """Parallel merge sort

    When the array exceeds the threshold, runs left and right recursions
    in separate threads. Falls back to sequential processing for small arrays.

    Note: Due to Python's GIL constraint, ProcessPoolExecutor is more
    effective than ThreadPoolExecutor for CPU-bound work.
    ThreadPoolExecutor is used here for simplicity.
    """
    if len(arr) <= 1:
        return arr

    if len(arr) < threshold:
        return merge_sort(arr)  # Sequential for small arrays

    mid = len(arr) // 2

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_left = executor.submit(parallel_merge_sort, arr[:mid], threshold)
        future_right = executor.submit(parallel_merge_sort, arr[mid:], threshold)
        left = future_left.result()
        right = future_right.result()

    return merge(left, right)


# Note: The above is a conceptual implementation. In production,
# use multiprocessing or C extensions (e.g., numpy.sort).
```

### Q5: What should you do when recursion gets too deep?

**A:** Python's default recursion limit is around 1000 calls. Divide-and-conquer recursion depth is O(log n), so unless n = 2^1000, this is rarely an issue. However, care is needed when unbalanced division or bugs cause O(n) depth.

Remedies:
1. **Raise the recursion limit:** Use `sys.setrecursionlimit()` temporarily (not a fundamental fix).
2. **Convert to iterative:** As in bottom-up merge sort, convert recursion to iteration.
3. **Tail recursion optimization:** Python does not support tail call optimization, but you can manually convert to loops.
4. **Guarantee balanced division:** Always splitting in half keeps the depth at O(log n).

### Q6: What are typical competitive programming problems solvable with divide and conquer?

**A:** The following are representative problem patterns.

| Pattern | Typical Problem | Complexity |
|:---|:---|:---|
| Merge sort application | Inversion count | O(n log n) |
| Binary search | Value search, minimum satisfying condition | O(log n) |
| Divide & conquer + computational geometry | Closest pair of points | O(n log n) |
| Segment tree (range queries) | Range maximum, range sum | O(n log n) build, O(log n) query |
| FFT / NTT | Polynomial multiplication, convolution | O(n log n) |
| CDQ divide and conquer | 3D partial order counting | O(n log^2 n) |

---

## 11. Comprehensive Complexity Comparison

### Complexity Summary of Divide-and-Conquer Algorithms

| Algorithm | Time Complexity | Space Complexity | Divisions | Division Ratio | Combine Cost |
|:---|:---|:---|:---:|:---:|:---|
| Binary search | O(log n) | O(log n) recursive / O(1) iterative | 1 | 1/2 | O(1) |
| Merge sort | O(n log n) | O(n) | 2 | 1/2 | O(n) |
| Quicksort (average) | O(n log n) | O(log n) | 2 | variable | O(n) |
| Quicksort (worst) | O(n^2) | O(n) | 2 | unbalanced | O(n) |
| Karatsuba multiplication | O(n^{1.585}) | O(n log n) | 3 | 1/2 | O(n) |
| Strassen matrix mult. | O(n^{2.807}) | O(n^2) | 7 | 1/2 | O(n^2) |
| Closest pair of points | O(n log n) | O(n) | 2 | 1/2 | O(n) |
| FFT | O(n log n) | O(n) | 2 | 1/2 | O(n) |
| Median of medians | O(n) | O(n) | 1 | 7/10 | O(n) |
| Repeated squaring | O(log n) | O(log n) recursive / O(1) iterative | 1 | 1/2 | O(1) |

---


## FAQ

### Q1: What is the most important point when studying this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory, but by actually writing code and verifying behavior.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving on.

### Q3: How is this used in practice?

Knowledge of this topic is frequently applied in everyday development work. It is particularly important during code reviews and architecture design.

---

## 12. Summary

### Core of Divide and Conquer

| Item | Key Point |
|:---|:---|
| 3 Steps | Recursive design of divide, conquer, combine. Correctness of the base case determines the whole |
| Master Theorem | T(n) = aT(n/b) + O(n^d) -> 3-case evaluation instantly yields complexity |
| Merge sort | Stable sort, worst-case O(n log n) guaranteed. The most fundamental and practical example |
| Karatsuba | Reduces 4 multiplications to 3. Dramatic improvement from O(n^2) to O(n^{1.585}) |
| Strassen | Reduces 8 matrix multiplications to 7. Theoretically significant but practically limited |
| Closest pair | Limited checking within the strip achieves O(n^2) to O(n log n) |
| FFT | Accelerates polynomial multiplication from O(n^2) to O(n log n). Foundation of signal processing |
| Applicability | Independent subproblems + efficient combining + balanced division are the keys to success |

### Checklist for Mastering Divide and Conquer

1. **Is the division balanced?** Unbalanced division degrades complexity
2. **Are the subproblems independent?** If they overlap, switch to DP
3. **Is the combine step efficient?** Combine cost governs overall complexity
4. **Is the base case correct?** Verify that all terminal cases are covered
5. **Can the Master Theorem be applied?** Set up the recurrence and analyze the complexity

---

## Recommended Next Readings

- [Sorting Algorithms](./00-sorting.md) -- Detailed implementation and comparison of merge sort and quicksort
- [Dynamic Programming](./04-dynamic-programming.md) -- Design technique for overlapping subproblems
- [Backtracking](./07-backtracking.md) -- Another recursive problem-solving paradigm
- [Search Algorithms](./01-searching.md) -- Details on binary search

---

## References

1. Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. -- Chapter 4, "Divide-and-Conquer," is a textbook treatment of divide and conquer, including the proof of the Master Theorem.
2. Karatsuba, A. & Ofman, Y. (1963). "Multiplication of multidigit numbers on automata." *Soviet Physics Doklady*, 7(7), 595-596. -- The historic paper first demonstrating that large number multiplication complexity can be sub-O(n^2).
3. Shamos, M. I. & Hoey, D. (1975). "Closest-point problems." *Proceedings of the 16th Annual Symposium on Foundations of Computer Science (FOCS)*. -- The original paper proposing the O(n log n) algorithm for the closest pair problem.
4. Kleinberg, J. & Tardos, E. (2005). *Algorithm Design*. Pearson. -- Chapter 5, "Divide and Conquer," offers rich application examples and carefully explains how to solve recurrences.
5. Strassen, V. (1969). "Gaussian elimination is not optimal." *Numerische Mathematik*, 13(4), 354-356. -- Showed that matrix multiplication complexity can be sub-O(n^3), opening the field of algebraic complexity theory.
6. Cooley, J. W. & Tukey, J. W. (1965). "An algorithm for the machine calculation of complex Fourier series." *Mathematics of Computation*, 19(90), 297-301. -- The original paper on the Fast Fourier Transform (FFT). One of the most influential applications of divide and conquer.
7. Sedgewick, R. & Wayne, K. (2011). *Algorithms* (4th ed.). Addison-Wesley. -- Provides detailed coverage of merge sort implementation variants (bottom-up, optimized versions).
