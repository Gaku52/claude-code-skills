# Sorting Algorithms

> Sorting is one of the oldest and most important problems in computer science. Comparison-based sorting has a theoretical lower bound of Omega(n log n), and understanding this limit provides deep insight into algorithm design in general. This chapter comprehensively covers sorting from basic to advanced algorithms, providing a complete picture of sorting from both theoretical and implementation perspectives.

---

## Learning Objectives

- [ ] Accurately explain the mechanics and computational complexity of 8 major sorting algorithms
- [ ] Understand and distinguish the stability, space complexity, and use cases of each sort
- [ ] Understand the proof of the O(n log n) comparison lower bound theorem and explain the difference from non-comparison sorts
- [ ] Explain the design philosophy of practical hybrid sorts such as Timsort and Introsort
- [ ] Survey sorting techniques for large-scale data processing, including external sort and parallel sort
- [ ] Identify correct usage patterns and anti-patterns for sorting in code

## Prerequisites

- Basic concepts of recursion and divide and conquer
- Basic array and list operations

---

## 1. Overview of Sorting Algorithms

Sorting algorithms are broadly classified into "comparison-based sorts" and "non-comparison-based sorts." Comparison-based sorts determine order through pairwise comparisons between elements, with a theoretical lower bound of Omega(n log n). Non-comparison-based sorts exploit the structure of element values themselves (digits, ranges, etc.) and can break this lower bound under certain conditions.

### 1.1 Overall Classification

```
                    Sorting Algorithms
                         |
            +------------+------------+
            |                         |
      Comparison-Based           Non-Comparison-Based
            |                         |
    +-------+-------+         +-------+-------+
    |       |       |         |       |       |
  O(n^2)  O(n logn) Hybrid   Counting Radix  Bucket
    |       |       |
  +--+--+ +--+--+ +--+--+
  |  |  | |  |  | |  |  |
 Bub Sel Ins Merge QS Heap Tim Intro pdq
 ble ect ert        Sort     sort sort sort
```

### 1.2 Comprehensive Sorting Algorithm Comparison Table

The following table summarizes the computational complexity and characteristics of all sorting algorithms covered in this chapter.

```
+================+============+============+============+=======+=======+====================+
| Algorithm      | Worst Case | Average    | Best Case  | Space | Stable| Primary Use        |
+================+============+============+============+=======+=======+====================+
| Bubble Sort    | O(n^2)     | O(n^2)     | O(n)       | O(1)  | Yes   | Educational         |
+----------------+------------+------------+------------+-------+-------+--------------------+
| Selection Sort | O(n^2)     | O(n^2)     | O(n^2)     | O(1)  | No    | Minimize swaps      |
+----------------+------------+------------+------------+-------+-------+--------------------+
| Insertion Sort | O(n^2)     | O(n^2)     | O(n)       | O(1)  | Yes   | Small/nearly sorted |
+----------------+------------+------------+------------+-------+-------+--------------------+
| Shell Sort     | O(n^1.5)   | Gap-dep.   | O(n log n) | O(1)  | No    | Medium-scale general|
+----------------+------------+------------+------------+-------+-------+--------------------+
| Merge Sort     | O(n log n) | O(n log n) | O(n log n) | O(n)  | Yes   | When stability req. |
+----------------+------------+------------+------------+-------+-------+--------------------+
| Quick Sort     | O(n^2)     | O(n log n) | O(n log n) | O(logn)| No   | General (fast)      |
+----------------+------------+------------+------------+-------+-------+--------------------+
| Heap Sort      | O(n log n) | O(n log n) | O(n log n) | O(1)  | No    | Worst-case + low mem|
+----------------+------------+------------+------------+-------+-------+--------------------+
| Counting Sort  | O(n + k)   | O(n + k)   | O(n + k)   | O(n+k)| Yes   | Integers/small range|
+----------------+------------+------------+------------+-------+-------+--------------------+
| Radix Sort     | O(d(n+k))  | O(d(n+k))  | O(d(n+k))  | O(n+k)| Yes   | Fixed-len int/string|
+----------------+------------+------------+------------+-------+-------+--------------------+
| Bucket Sort    | O(n^2)     | O(n + k)   | O(n + k)   | O(n+k)| Yes   | Uniformly dist. data|
+----------------+------------+------------+------------+-------+-------+--------------------+
| Timsort        | O(n log n) | O(n log n) | O(n)       | O(n)  | Yes   | Real-world data     |
+----------------+------------+------------+------------+-------+-------+--------------------+
| Introsort      | O(n log n) | O(n log n) | O(n log n) | O(logn)| No   | C++ STL             |
+----------------+------------+------------+------------+-------+-------+--------------------+
```

* k = range of values, d = number of digits

---

## 2. O(n^2) Sorts — Basic Sorting Algorithms

O(n^2) sorting algorithms suffer from rapid performance degradation as data size grows and are unsuitable for large-scale data. However, they are extremely important for learning fundamental algorithm design concepts (loop invariants, comparison and swap, worst-case analysis), and for small-scale data, their low overhead can make them the fastest option.

### 2.1 Bubble Sort

Bubble sort compares and swaps adjacent elements, causing the largest value to "bubble" up to the end of the array. In each pass, the maximum value in the unsorted portion moves to its final position.

#### Algorithm Steps

1. Compare adjacent elements from the beginning to the end of the array
2. If the left element is greater than the right, swap them
3. At the end of one pass, the maximum value has reached the end
4. Shrink the unsorted range by one and repeat
5. If no swaps occur during a pass, the array is sorted — terminate early

#### ASCII Diagram: Bubble Sort in Action

```
Array: [5, 3, 8, 1, 4]

=== Pass 1 ===
[5, 3, 8, 1, 4]   5 > 3 → swap
 ^  ^
[3, 5, 8, 1, 4]   5 < 8 → no swap
    ^  ^
[3, 5, 8, 1, 4]   8 > 1 → swap
       ^  ^
[3, 5, 1, 8, 4]   8 > 4 → swap
          ^  ^
[3, 5, 1, 4, 8]   ← 8 reaches its final position
                                    Settled: [8]

=== Pass 2 ===
[3, 5, 1, 4 | 8]  3 < 5 → no swap
 ^  ^
[3, 5, 1, 4 | 8]  5 > 1 → swap
    ^  ^
[3, 1, 5, 4 | 8]  5 > 4 → swap
       ^  ^
[3, 1, 4, 5 | 8]  ← 5 reaches its final position
                                    Settled: [5, 8]

=== Pass 3 ===
[3, 1, 4 | 5, 8]  3 > 1 → swap
 ^  ^
[1, 3, 4 | 5, 8]  3 < 4 → no swap
    ^  ^
[1, 3, 4 | 5, 8]  ← 4 reaches its final position
                                    Settled: [4, 5, 8]

=== Pass 4 ===
[1, 3 | 4, 5, 8]  1 < 3 → no swap (zero swaps)
 ^  ^
→ Zero swaps, early exit!

Result: [1, 3, 4, 5, 8]  ← Sort complete
```

#### Complete Implementation

```python
def bubble_sort(arr: list) -> list:
    """
    Bubble Sort: Compares and swaps adjacent elements, bubbling the
    maximum to the end.

    Loop Invariant:
        After the i-th pass, the last i elements of the array are sorted
        and in their final positions.

    Args:
        arr: List to sort (modified in-place)
    Returns:
        The sorted list (same object as input)
    """
    n = len(arr)
    for i in range(n):
        swapped = False
        # Scan the unsorted portion (last i elements are settled)
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        # If no swaps occurred, the array is already sorted
        if not swapped:
            break
    return arr


# --- Verification ---
assert bubble_sort([5, 3, 8, 1, 4]) == [1, 3, 4, 5, 8]
assert bubble_sort([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]   # Best: 1 pass
assert bubble_sort([5, 4, 3, 2, 1]) == [1, 3, 4, 5, 8]   # Worst: reverse order
assert bubble_sort([]) == []
assert bubble_sort([42]) == [42]
```

#### Complexity Analysis

- **Worst case O(n^2)**: When the array is in reverse order. All comparisons and swaps occur in every pass. The number of comparisons is n(n-1)/2.
- **Average case O(n^2)**: For random input, approximately n^2/4 comparisons and swaps on average.
- **Best case O(n)**: When the array is already sorted. Early exit terminates after 1 pass (n-1 comparisons).
- **Space complexity O(1)**: An in-place algorithm using only temporary variables.
- **Stability: Yes**: Equal elements are never swapped, so their relative order is preserved.

#### Bubble Sort Variants

**Cocktail Shaker Sort (Bidirectional Bubble Sort)**: A variant that alternates between left-to-right and right-to-left passes. This mitigates the "turtle" problem (small values near the end).

```python
def cocktail_shaker_sort(arr: list) -> list:
    """Bidirectional bubble sort. Alternates forward and backward passes."""
    n = len(arr)
    start = 0
    end = n - 1
    swapped = True

    while swapped:
        swapped = False
        # Forward pass: left → right
        for i in range(start, end):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swapped = True
        end -= 1

        if not swapped:
            break

        swapped = False
        # Backward pass: right → left
        for i in range(end, start, -1):
            if arr[i] < arr[i - 1]:
                arr[i], arr[i - 1] = arr[i - 1], arr[i]
                swapped = True
        start += 1

    return arr


assert cocktail_shaker_sort([5, 3, 8, 1, 4]) == [1, 3, 4, 5, 8]
```

---

### 2.2 Selection Sort

Selection sort finds the minimum value in the unsorted portion and swaps it with the first unsorted position. Its distinguishing characteristic is that the number of swaps is O(n), the minimum possible.

#### Algorithm Steps

1. Find the minimum value in the unsorted portion using linear search
2. Swap the minimum with the first element of the unsorted portion
3. Expand the sorted portion by one
4. Repeat until the unsorted portion is empty

#### ASCII Diagram: Selection Sort in Action

```
Array: [29, 10, 14, 37, 13]

=== Step 1: Search for minimum ===
[29, 10, 14, 37, 13]
      ^^                  ← Minimum = 10 (index 1)
 Swap with first (index 0)
[10, 29, 14, 37, 13]
 ~~                       ← Settled: [10]

=== Step 2: Search for minimum in remainder ===
[10 | 29, 14, 37, 13]
                  ^^      ← Minimum = 13 (index 4)
 Swap with index 1
[10, 13, 14, 37, 29]
 ~~  ~~                   ← Settled: [10, 13]

=== Step 3: Search for minimum in remainder ===
[10, 13 | 14, 37, 29]
          ^^              ← Minimum = 14 (index 2, itself)
 No swap needed
[10, 13, 14, 37, 29]
 ~~  ~~  ~~               ← Settled: [10, 13, 14]

=== Step 4: Search for minimum in remainder ===
[10, 13, 14 | 37, 29]
                  ^^      ← Minimum = 29 (index 4)
 Swap with index 3
[10, 13, 14, 29, 37]
 ~~  ~~  ~~  ~~  ~~       ← All elements settled

Result: [10, 13, 14, 29, 37]
```

#### Complete Implementation

```python
def selection_sort(arr: list) -> list:
    """
    Selection Sort: Finds the minimum in the unsorted portion and
    places it at the front.

    Loop Invariant:
        At the start of the i-th iteration, arr[0:i] is sorted and
        every element is less than or equal to all elements in arr[i:].

    Args:
        arr: List to sort (modified in-place)
    Returns:
        The sorted list
    """
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        # Swap only if the minimum is not already in place
        if min_idx != i:
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr


# --- Verification ---
assert selection_sort([29, 10, 14, 37, 13]) == [10, 13, 14, 29, 37]
assert selection_sort([1, 2, 3]) == [1, 2, 3]
assert selection_sort([3, 2, 1]) == [1, 2, 3]
```

#### Complexity Analysis

- **Worst/Average/Best case: All O(n^2)**: Regardless of input state, always performs n(n-1)/2 comparisons.
- **Space complexity O(1)**: In-place.
- **Stability: No**: Swapping distant elements can disrupt the relative order of equal elements.

Example: Sorting `[3a, 3b, 1]` produces `[1, 3b, 3a]`, reversing the order of 3a and 3b.

#### Concrete Example of Selection Sort Instability

```
Input: [3a, 3b, 1]
       * 3a and 3b have the same value 3, but a originally comes first

Step 1: Minimum = 1 (index 2)
        Swap index 0 (3a) with index 2 (1)
        → [1, 3b, 3a]
                  ^^  ← Relative order of 3a and 3b reversed!

Result: [1, 3b, 3a]  ← Unstable
```

#### Advantages and Disadvantages of Selection Sort

**Advantages**:
- At most n-1 swaps. Beneficial in environments where write cost is high (e.g., flash memory)
- Simple algorithm with low risk of implementation errors
- Low data movement

**Disadvantages**:
- Computation does not improve even for sorted input (no adaptivity)
- Unstable sort
- Unsuitable for large-scale data

---

### 2.3 Insertion Sort

Insertion sort works like inserting a card into a sorted hand: it takes the first element of the unsorted portion and inserts it at the correct position within the sorted portion. It is extremely fast for nearly sorted data and is used internally by many hybrid sort implementations.

#### Algorithm Steps

1. Start from the second element (the first element is considered sorted by itself)
2. Temporarily save the current element (key)
3. Scan the sorted portion from right to left, shifting elements greater than the key one position to the right
4. Insert the key at the vacated position
5. Repeat until the end of the array

#### ASCII Diagram: Insertion Sort in Action

```
Array: [7, 3, 5, 1, 9]

 Sorted    | Unsorted
-----------+---------
[7]        | [3, 5, 1, 9]     key = 3
 3 < 7 → shift 7 right
[_, 7]       insert key=3 at position 0
[3, 7]     | [5, 1, 9]

[3, 7]     | [5, 1, 9]        key = 5
 5 < 7 → shift 7 right
 5 > 3 → stop
[3, _, 7]    insert key=5 at position 1
[3, 5, 7]  | [1, 9]

[3, 5, 7]  | [1, 9]           key = 1
 1 < 7 → shift 7 right
 1 < 5 → shift 5 right
 1 < 3 → shift 3 right
 Reached front → stop
[_, 3, 5, 7] insert key=1 at position 0
[1, 3, 5, 7] | [9]

[1, 3, 5, 7] | [9]            key = 9
 9 > 7 → stop (no shifts)
[1, 3, 5, 7, 9]               ← Sort complete
```

#### Complete Implementation

```python
def insertion_sort(arr: list) -> list:
    """
    Insertion Sort: Inserts each element at the correct position
    within the sorted portion.

    Loop Invariant:
        At the start of the i-th iteration, arr[0:i] contains the
        elements of the original arr[0:i] in sorted order.

    Args:
        arr: List to sort (modified in-place)
    Returns:
        The sorted list
    """
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        # Shift elements greater than key to the right
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        # Insert key at the correct position
        arr[j + 1] = key
    return arr


# --- Verification ---
assert insertion_sort([7, 3, 5, 1, 9]) == [1, 3, 5, 7, 9]
assert insertion_sort([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]   # Best: O(n)
assert insertion_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]   # Worst: O(n^2)
```

#### Complexity Analysis

- **Worst case O(n^2)**: When the array is in reverse order. Each element must be moved to the front. Both the number of comparisons and shifts are n(n-1)/2.
- **Average case O(n^2)**: For random input, approximately n^2/4 comparisons on average.
- **Best case O(n)**: When already sorted. Each element requires only 1 comparison to confirm its position.
- **Space complexity O(1)**: In-place.
- **Stability: Yes**: The comparison `arr[j] > key` does not move equal elements, preserving relative order.

#### Important Properties of Insertion Sort

**Adaptivity**: Insertion sort's running time is proportional to the number of "inversions." An inversion is a pair (i, j) where i < j and arr[i] > arr[j]. A sorted array has 0 inversions; a reverse-sorted array has n(n-1)/2. For nearly sorted data (inversions on the order of O(n)), insertion sort runs in O(n).

**Online Property**: Can maintain sort order even when data arrives incrementally. Suitable for processing streaming data.

**Superiority for Small Data**: Due to low loop overhead and high cache efficiency, insertion sort is often faster than O(n log n) algorithms when n is small (roughly n < 50). Python's Timsort and C++'s Introsort use insertion sort for small subarrays.

#### Binary Insertion Sort

A variant that uses binary search to find the insertion position. This reduces the number of comparisons to O(n log n), but since the shift operation remains O(n^2), the overall complexity does not improve. However, it is effective when comparison cost is high (e.g., comparing complex objects).

```python
import bisect

def binary_insertion_sort(arr: list) -> list:
    """Insertion sort using binary search to determine the insertion position."""
    for i in range(1, len(arr)):
        key = arr[i]
        # Binary search for insertion position (O(log i))
        pos = bisect.bisect_left(arr, key, 0, i)
        # Shift elements from pos to i-1 one position right (O(i))
        for j in range(i, pos, -1):
            arr[j] = arr[j - 1]
        arr[pos] = key
    return arr


assert binary_insertion_sort([7, 3, 5, 1, 9]) == [1, 3, 5, 7, 9]
```

---

### 2.4 Shell Sort

Shell sort is a generalization of insertion sort. By comparing and swapping elements at distant positions first, it resolves large "inversions" early, reducing the workload of the final insertion sort.

#### Algorithm Steps

1. Determine a gap sequence (e.g., n/2, n/4, ..., 1)
2. For each gap, perform a gapped insertion sort
3. Repeat while shrinking the gap
4. The final pass with gap 1 (standard insertion sort) completes the sort

#### Complete Implementation

```python
def shell_sort(arr: list) -> list:
    """
    Shell Sort: Applies gapped insertion sorts progressively.

    The gap sequence follows Shell's original paper: n//2, n//4, ..., 1.
    Better gap sequences (Knuth, Sedgewick, etc.) improve
    the computational complexity.

    Args:
        arr: List to sort (modified in-place)
    Returns:
        The sorted list
    """
    n = len(arr)
    gap = n // 2

    while gap > 0:
        # Gapped insertion sort
        for i in range(gap, n):
            key = arr[i]
            j = i
            while j >= gap and arr[j - gap] > key:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = key
        gap //= 2

    return arr


# --- Verification ---
assert shell_sort([12, 34, 54, 2, 3]) == [2, 3, 12, 34, 54]
assert shell_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]
```

#### Gap Sequence Selection and Complexity

The complexity of shell sort depends heavily on the chosen gap sequence.

```
+============================+==========================+================+
| Gap Sequence               | Author                   | Worst Case     |
+============================+==========================+================+
| n/2, n/4, ..., 1           | Shell (1959)             | O(n^2)         |
+----------------------------+--------------------------+----------------+
| 2^k - 1: 1, 3, 7, 15, ... | Hibbard (1963)           | O(n^(3/2))     |
+----------------------------+--------------------------+----------------+
| (3^k-1)/2: 1, 4, 13, 40...| Knuth (1973)             | O(n^(3/2))     |
+----------------------------+--------------------------+----------------+
| Prime-based sequence       | Sedgewick (1986)         | O(n^(4/3))     |
+----------------------------+--------------------------+----------------+
| 1, 4, 10, 23, 57, 132, ...| Ciura (2001, empirical)  | Unproven (fast)|
+----------------------------+--------------------------+----------------+
```

The precise analysis of shell sort complexity is a combinatorial problem that depends on the gap sequence, and parts of it remain unsolved.

---

## 3. O(n log n) Sorts — Divide and Conquer and Priority Queues

### 3.1 Merge Sort

Merge sort is a representative application of the divide and conquer paradigm. It recursively splits the array in half, sorts each subarray, and then merges them back together. Its complexity is independent of the input state — always O(n log n) — and it is a stable sort, which is a major advantage.

#### Algorithm Steps

1. If the array length is 1 or less, return it as-is (base case)
2. Split the array at the midpoint
3. Recursively sort the left and right halves
4. Merge the two sorted subarrays

#### ASCII Diagram: Merge Sort Division and Merging

```
              [38, 27, 43, 3, 9, 82, 10]
                     /            \
             [38, 27, 43, 3]    [9, 82, 10]
              /         \        /       \
         [38, 27]    [43, 3]  [9, 82]   [10]
          /    \      /    \    /   \      |
        [38]  [27]  [43]  [3] [9]  [82]  [10]
          \    /      \    /    \   /      |
         [27, 38]    [3, 43]  [9, 82]   [10]   ← Merge (O(n) per level)
              \         /        \       /
          [3, 27, 38, 43]    [9, 10, 82]
                     \            /
           [3, 9, 10, 27, 38, 43, 82]              ← log n levels

  Total complexity = O(n) x O(log n) = O(n log n)
```

#### Detailed Merge Operation Diagram

```
Merge: combining [3, 27, 38, 43] and [9, 10, 82]

  Left: [3, 27, 38, 43]    Right: [9, 10, 82]    Result: []
        ^                         ^
        L                         R

  Step 1: 3 < 9  → add 3 to result
  Left: [3, 27, 38, 43]    Right: [9, 10, 82]    Result: [3]
           ^                       ^

  Step 2: 27 > 9 → add 9 to result
  Left: [3, 27, 38, 43]    Right: [9, 10, 82]    Result: [3, 9]
           ^                          ^

  Step 3: 27 > 10 → add 10 to result
  Left: [3, 27, 38, 43]    Right: [9, 10, 82]    Result: [3, 9, 10]
           ^                              ^

  Step 4: 27 < 82 → add 27 to result
  Left: [3, 27, 38, 43]    Right: [9, 10, 82]    Result: [3, 9, 10, 27]
               ^                          ^

  Step 5: 38 < 82 → add 38 to result
  Step 6: 43 < 82 → add 43 to result
  Left: exhausted            Right: [82] remaining

  Step 7: append remaining [82] from right

  Result: [3, 9, 10, 27, 38, 43, 82]  ← Merge complete
```

#### Complete Implementation

```python
def merge_sort(arr: list) -> list:
    """
    Merge Sort: A stable sort using the divide and conquer paradigm.

    Complexity:
        - Worst/Average/Best: O(n log n)
        - Space: O(n) (temporary array for merging)
        - Stable: Yes

    Args:
        arr: List to sort
    Returns:
        A new sorted list
    """
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return _merge(left, right)


def _merge(left: list, right: list) -> list:
    """Merge two sorted lists into one sorted list."""
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:   # <= ensures stability
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
assert merge_sort([38, 27, 43, 3, 9, 82, 10]) == [3, 9, 10, 27, 38, 43, 82]
assert merge_sort([]) == []
assert merge_sort([1]) == [1]
```

#### Rigorous Complexity Derivation

Recurrence relation for merge sort:

```
T(n) = 2 * T(n/2) + O(n)      (for n > 1)
T(1) = O(1)

Applying the Master Theorem:
  a = 2, b = 2, f(n) = O(n)
  n^(log_b(a)) = n^(log_2(2)) = n^1 = n
  f(n) = O(n) = O(n^(log_b(a)))

  → Case 2 applies: T(n) = O(n log n)

Intuitive understanding via recursion tree:
  Depth 0:  1 problem,  size n each     → work: n
  Depth 1:  2 problems, size n/2 each   → work: n
  Depth 2:  4 problems, size n/4 each   → work: n
  ...
  Depth k:  2^k problems, size n/2^k    → work: n
  ...
  Depth log n: n problems, size 1 each  → work: n

  Total = n * (log n + 1) = O(n log n)
```

#### Bottom-Up Merge Sort

A non-recursive implementation of merge sort. Avoids stack overflow from recursion and has a slightly smaller constant factor.

```python
def merge_sort_bottom_up(arr: list) -> list:
    """
    Bottom-up (non-recursive) merge sort.
    Starts with subarrays of size 1 and doubles the merge size iteratively.
    """
    n = len(arr)
    if n <= 1:
        return arr[:]

    # Working copy
    result = arr[:]
    width = 1

    while width < n:
        for start in range(0, n, 2 * width):
            mid = min(start + width, n)
            end = min(start + 2 * width, n)

            left = result[start:mid]
            right = result[mid:end]
            merged = _merge(left, right)
            result[start:start + len(merged)] = merged

        width *= 2

    return result


assert merge_sort_bottom_up([38, 27, 43, 3, 9, 82, 10]) == [3, 9, 10, 27, 38, 43, 82]
```

---

### 3.2 Quick Sort

Quick sort was devised by C.A.R. Hoare in 1960 and is based on the divide and conquer paradigm. It selects a pivot element, partitions the array into elements less than or equal to the pivot and elements greater than or equal to the pivot, then recursively sorts each partition. Its average complexity is O(n log n), and due to its small constant factor and excellent cache efficiency, it is adopted as the fastest sort in many implementations.

#### Algorithm Steps

1. Select a pivot element (multiple selection strategies exist)
2. Partition the array: move elements less than or equal to the pivot to the left, and elements greater than or equal to the right
3. Recursively sort the left and right partitions

#### ASCII Diagram: Quick Sort Partition Operation (Lomuto Scheme)

```
Array: [8, 3, 5, 1, 4, 2, 7, 6]    Pivot = arr[high] = 6

Initial state:
  i = -1 (right edge of the <= pivot region)
  j = 0  (scan position)

  [8, 3, 5, 1, 4, 2, 7, 6]
   j                    pivot
   i=-1

j=0: arr[0]=8 > 6 → do nothing
  [8, 3, 5, 1, 4, 2, 7, 6]
      j

j=1: arr[1]=3 <= 6 → i++, swap(arr[1], arr[1])  * i=0
  [3, 8, 5, 1, 4, 2, 7, 6]   ← swap 3 and 8
   i     j

j=2: arr[2]=5 <= 6 → i++, swap(arr[2], arr[2])  * i=1
  [3, 5, 8, 1, 4, 2, 7, 6]   ← swap 5 and 8
      i     j

j=3: arr[3]=1 <= 6 → i++, swap(arr[3], arr[3])  * i=2
  [3, 5, 1, 8, 4, 2, 7, 6]   ← swap 1 and 8
         i     j

j=4: arr[4]=4 <= 6 → i++, swap(arr[3], arr[4])  * i=3
  [3, 5, 1, 4, 8, 2, 7, 6]   ← swap 4 and 8
            i     j

j=5: arr[5]=2 <= 6 → i++, swap(arr[4], arr[5])  * i=4
  [3, 5, 1, 4, 2, 8, 7, 6]   ← swap 2 and 8
               i     j

j=6: arr[6]=7 > 6 → do nothing
  [3, 5, 1, 4, 2, 8, 7, 6]
               i        j

Final: swap(arr[i+1], arr[high])  * place pivot in correct position
  [3, 5, 1, 4, 2, 6, 7, 8]
                  ^
                  Pivot's final position (index 5)

Result:
  [3, 5, 1, 4, 2]  6  [7, 8]
   <= pivot        pivot  >= pivot
   → sort recursively     → sort recursively
```

#### Complete Implementation (Educational List Comprehension Version)

```python
def quicksort_simple(arr: list) -> list:
    """
    Quick Sort: Concise list comprehension version (educational).

    Note: This implementation uses O(n) additional memory.
    The in-place version is shown below.

    Args:
        arr: List to sort
    Returns:
        A new sorted list
    """
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort_simple(left) + middle + quicksort_simple(right)


assert quicksort_simple([8, 3, 5, 1, 4, 2, 7, 6]) == [1, 2, 3, 4, 5, 6, 7, 8]
```

#### Complete Implementation (In-Place Version + Pivot Strategy)

```python
import random

def quicksort_inplace(arr: list, low: int = 0, high: int = None) -> list:
    """
    Quick Sort: In-place version.
    Uses random pivot selection to probabilistically avoid worst case.

    Complexity:
        - Worst: O(n^2) — when the pivot is always the max/min
        - Average: O(n log n) — expected with random pivots
        - Space: O(log n) — recursion depth (in-place)
        - Stable: No

    Args:
        arr: List to sort (modified in-place)
        low: Start index of the sort range
        high: End index of the sort range
    Returns:
        The sorted list (same object as input)
    """
    if high is None:
        high = len(arr) - 1

    if low < high:
        pivot_idx = _partition_random(arr, low, high)
        quicksort_inplace(arr, low, pivot_idx - 1)
        quicksort_inplace(arr, pivot_idx + 1, high)

    return arr


def _partition_random(arr: list, low: int, high: int) -> int:
    """Random pivot selection + Lomuto partition."""
    # Select a random element as pivot and move to end
    rand_idx = random.randint(low, high)
    arr[rand_idx], arr[high] = arr[high], arr[rand_idx]

    pivot = arr[high]
    i = low - 1

    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


# --- Verification ---
test = [8, 3, 5, 1, 4, 2, 7, 6]
quicksort_inplace(test)
assert test == [1, 2, 3, 4, 5, 6, 7, 8]
```

#### Pivot Selection Strategy Comparison

The choice of pivot has a decisive impact on quick sort's performance.

```
+==========================+===================+============================+
| Strategy                 | Worst Case        | Characteristics            |
+==========================+===================+============================+
| First/last element       | Degrades to O(n^2)| Simplest but dangerous     |
|                          | on sorted arrays  |                            |
+--------------------------+-------------------+----------------------------+
| Random selection         | Probabilistically | Expected O(n log n)        |
|                          | avoided           | Worst case theoretically   |
|                          |                   | remains                    |
+--------------------------+-------------------+----------------------------+
| Median-of-Three         | Greatly improved   | Median of low, mid, high   |
|                          |                   | Adopted by many impls.     |
+--------------------------+-------------------+----------------------------+
| Ninther                  | Further improved   | Median of medians from     |
| (Median-of-Nine)         |                   | 3 groups. For large data   |
+--------------------------+-------------------+----------------------------+
| Median of Medians        | O(n log n)        | Theoretically optimal but  |
|                          | guaranteed        | large constant, impractical|
+--------------------------+-------------------+----------------------------+
```

#### Why Quick Sort Is Fastest in Practice

1. **Cache locality**: Sequential access to contiguous array regions yields high CPU cache hit rates
2. **Small constant factor**: Simple comparison and swap operations with low per-iteration overhead
3. **In-place operation**: No additional memory allocation or deallocation needed
4. **Branch prediction efficiency**: Conditional branches during partitioning are easy to predict
5. **Tail recursion optimization**: Processing the shorter partition first and using tail recursion for the longer one limits stack usage to O(log n)

---

### 3.3 Heap Sort

Heap sort uses the max-heap data structure. It guarantees O(n log n) worst-case complexity and operates in-place. However, due to poor cache efficiency, it is often outperformed by quick sort in practice.

#### Basic Heap Structure

```
Max-Heap: A complete binary tree where parent >= children

Array [16, 14, 10, 8, 7, 9, 3, 2, 4, 1] as a heap:

                    16          ← index 0
                   /  \
                 14    10       ← index 1, 2
                / \   / \
               8   7 9   3     ← index 3, 4, 5, 6
              / \  |
             2  4  1            ← index 7, 8, 9

  Parent index: i      → Children: 2i+1 (left), 2i+2 (right)
  Child index: i       → Parent: (i-1)//2
```

#### Algorithm Steps

1. Convert the array into a max-heap (Build-Max-Heap)
2. Swap the root (maximum) with the last element
3. Reduce the heap size by 1 and restore heap property (Heapify)
4. Repeat steps 2-3 until the heap size is 1

#### ASCII Diagram: Heap Sort in Action

```
Initial array: [4, 10, 3, 5, 1]

=== Phase 1: Build Max-Heap ===
Apply heapify bottom-up

index 1 (value 10) heapify: already satisfies heap property
index 0 (value 4) heapify:
  4 < 10 → swap → 10 becomes root
  4 < 5  → swap → 5 becomes left child

Max-heap complete:
         10
        /  \
       5    3
      / \
     4   1
Array: [10, 5, 3, 4, 1]

=== Phase 2: Sort ===

Step 1: Swap arr[0]=10 with arr[4]=1 → [1, 5, 3, 4, |10]
  heapify(0, size=4):
         1                    5
        / \        →         / \
       5   3                4   3
      /                    /
     4                    1
  Array: [5, 4, 3, 1, |10]

Step 2: Swap arr[0]=5 with arr[3]=1 → [1, 4, 3, |5, 10]
  heapify(0, size=3):
         1          →        4
        / \                  / \
       4   3                1   3
  Array: [4, 1, 3, |5, 10]

Step 3: Swap arr[0]=4 with arr[2]=3 → [3, 1, |4, 5, 10]
  heapify(0, size=2):
         3          →        3
        /                    /
       1                    1
  Array: [3, 1, |4, 5, 10]

Step 4: Swap arr[0]=3 with arr[1]=1 → [1, |3, 4, 5, 10]
  Array: [1, 3, 4, 5, 10]  ← Sort complete!
```

#### Complete Implementation

```python
def heapsort(arr: list) -> list:
    """
    Heap Sort: An in-place sort using a max-heap.

    Complexity:
        - Worst/Average/Best: O(n log n)
        - Space: O(1) — in-place
        - Stable: No

    Args:
        arr: List to sort (modified in-place)
    Returns:
        The sorted list
    """
    n = len(arr)

    # Phase 1: Build Max-Heap (bottom-up)
    # Heapify from the last non-leaf node (n//2 - 1) to root (0)
    for i in range(n // 2 - 1, -1, -1):
        _heapify(arr, n, i)

    # Phase 2: Extract max from heap and sort
    for i in range(n - 1, 0, -1):
        # Swap max (root) with last element
        arr[0], arr[i] = arr[i], arr[0]
        # Reduce heap size and restore heap property
        _heapify(arr, i, 0)

    return arr


def _heapify(arr: list, heap_size: int, root: int) -> None:
    """
    Restore the heap property of the subtree rooted at the given node.

    Args:
        arr: Array representing the heap
        heap_size: Number of elements treated as the heap
        root: Index of the subtree root to heapify
    """
    largest = root
    left = 2 * root + 1
    right = 2 * root + 2

    if left < heap_size and arr[left] > arr[largest]:
        largest = left
    if right < heap_size and arr[right] > arr[largest]:
        largest = right

    if largest != root:
        arr[root], arr[largest] = arr[largest], arr[root]
        # Recursively restore heap property in the affected subtree
        _heapify(arr, heap_size, largest)


# --- Verification ---
assert heapsort([4, 10, 3, 5, 1]) == [1, 3, 4, 5, 10]
assert heapsort([1]) == [1]
assert heapsort([]) == []
assert heapsort([5, 5, 5]) == [5, 5, 5]
```

#### Why Build-Max-Heap Is O(n)

Intuitively, it seems like O(n log n) since we apply O(log n) heapify to n/2 nodes, but the actual complexity is O(n).

```
Number of nodes at depth d: n / 2^(d+1)
Heapify cost at depth d: O(d)

Total = sum_{d=0}^{log n} (n / 2^(d+1)) * d
     = (n/2) * sum_{d=0}^{log n} d / 2^d
     <= (n/2) * sum_{d=0}^{inf} d / 2^d
     = (n/2) * 2
     = n
     = O(n)

  * Using the series sum_{d=0}^{inf} d * x^d = x / (1-x)^2 with x = 1/2 gives 2
```

Leaf nodes (roughly half the total) require no heapify, and deeper nodes require less heapify work, so the overall time is linear.

#### Heap Sort's Position Among Sorts

| Property | Quick Sort | Merge Sort | Heap Sort |
|----------|-----------|-----------|-----------|
| Worst case | O(n^2) | O(n log n) | O(n log n) |
| Space | O(log n) | O(n) | O(1) |
| Stability | No | Yes | No |
| Cache efficiency | High | Medium | Low |

Heap sort is the only algorithm that simultaneously guarantees "worst-case O(n log n)" and "O(1) space." It is valuable in memory-constrained environments or when worst-case complexity guarantees are required (e.g., real-time systems).

---

## 4. Non-Comparison-Based Sorts

Comparison-based sorts have a theoretical lower bound of Omega(n log n), but non-comparison-based sorts exploit the structure of element "values" to break this barrier. However, they have constraints on applicable data types and ranges.

### 4.1 Comparison Lower Bound Theorem

A rigorous proof of the theoretical lower bound for comparison-based sorts.

#### Decision Tree Model

The behavior of a comparison-based sorting algorithm can be represented as a "decision tree." Each internal node represents one comparison (a_i <= a_j?), with the left child for Yes and right child for No. Leaf nodes represent the sort result (a permutation).

```
Decision tree example for n = 3:

                      a1 <= a2 ?
                     /          \
                Yes               No
               /                    \
          a2 <= a3 ?             a1 <= a3 ?
         /        \             /         \
       Yes         No         Yes          No
       /             \         /             \
  a1,a2,a3      a1 <= a3 ?  a2,a1,a3    a2 <= a3 ?
                /        \               /        \
              Yes         No           Yes         No
              /             \           /             \
         a1,a3,a2      a3,a1,a2   a2,a3,a1      a3,a2,a1

  Number of leaves = 3! = 6 (all permutations)
  Tree height = 3 (worst-case number of comparisons)
```

#### Theorem and Proof

```
Theorem: The worst-case number of comparisons for any comparison-based
         sorting algorithm is Omega(n log n).

Proof:
  Sorting n elements can produce n! possible outputs.
  The number of leaves L in the decision tree must satisfy L >= n!.
  The maximum number of leaves in a binary tree of height h is 2^h, so:

    2^h >= n!
    h >= log_2(n!)

  By Stirling's approximation n! ~ sqrt(2*pi*n) * (n/e)^n:
    log_2(n!) = log_2(sqrt(2*pi*n)) + n * log_2(n/e)
              = Theta(n log n)

  Therefore:
    h >= Omega(n log n)

  Since the height of the decision tree equals the worst-case number
  of comparisons, the worst-case complexity of comparison-based sorting
  is Omega(n log n).  Q.E.D.

  More precise lower bound: log_2(n!) = n*log_2(n) - n*log_2(e) + O(log n)
                                       ~ n*log_2(n) - 1.443*n
```

This theorem implies that merge sort and heap sort are asymptotically optimal as comparison-based sorts.

### 4.2 Counting Sort

Counting sort counts the occurrences of each value and uses this information to place elements at their correct positions. It runs in O(n + k) when the range k of values is small relative to n.

#### Complete Implementation (Stable Version)

```python
def counting_sort(arr: list, max_val: int = None) -> list:
    """
    Counting Sort (stable version): Counts occurrences and places elements
    at their correct positions.

    Complexity:
        - Worst/Average/Best: O(n + k)  (k = max_val + 1 = range of values)
        - Space: O(n + k)
        - Stable: Yes

    Args:
        arr: List of non-negative integers
        max_val: Maximum value (auto-detected if omitted)
    Returns:
        A new sorted list
    """
    if not arr:
        return []

    if max_val is None:
        max_val = max(arr)

    # Step 1: Count occurrences of each value
    count = [0] * (max_val + 1)
    for x in arr:
        count[x] += 1

    # Step 2: Compute prefix sums (determines final position of each value)
    for i in range(1, len(count)):
        count[i] += count[i - 1]

    # Step 3: Traverse in reverse to ensure stability
    output = [0] * len(arr)
    for x in reversed(arr):
        count[x] -= 1
        output[count[x]] = x

    return output


# --- Verification ---
assert counting_sort([4, 2, 2, 8, 3, 3, 1]) == [1, 2, 2, 3, 3, 4, 8]
assert counting_sort([]) == []
assert counting_sort([5]) == [5]
assert counting_sort([1, 1, 1]) == [1, 1, 1]
```

#### Constraints of Counting Sort

- **Non-negative integers only**: Since values are used as indices, it cannot directly handle negative values or floating-point numbers (offset adjustment is needed)
- **Inefficient when the value range k is large**: When k >> n, both space and time are wasted. For example, with a range of 0-10^9 and 100 elements, counting sort is impractical
- **Lack of generality**: Cannot customize comparison functions, so it cannot handle complex sorting criteria

### 4.3 Radix Sort

Radix sort decomposes numbers digit by digit and repeatedly applies a stable sort (typically counting sort) to each digit. It sorts d-digit numbers in O(d(n + k)) where k is the radix (usually 10 or 256).

#### LSD (Least Significant Digit) Radix Sort

Applies stable sort from the least significant digit to the most significant digit.

```python
def radix_sort_lsd(arr: list) -> list:
    """
    LSD Radix Sort: Repeatedly applies stable sort from the least
    significant digit.

    Complexity:
        - O(d * (n + k))  (d = max digits, k = radix)
        - For radix 10: k = 10, d = log_10(max_val)
        - Space: O(n + k)
        - Stable: Yes

    Args:
        arr: List of non-negative integers
    Returns:
        A new sorted list
    """
    if not arr:
        return []

    max_val = max(arr)
    result = arr[:]
    exp = 1  # Current digit position (1, 10, 100, ...)

    while max_val // exp > 0:
        result = _counting_sort_by_digit(result, exp)
        exp *= 10

    return result


def _counting_sort_by_digit(arr: list, exp: int) -> list:
    """Stable counting sort based on a specific digit (position exp)."""
    n = len(arr)
    output = [0] * n
    count = [0] * 10  # Radix 10

    # Count occurrences of the relevant digit
    for x in arr:
        digit = (x // exp) % 10
        count[digit] += 1

    # Prefix sum
    for i in range(1, 10):
        count[i] += count[i - 1]

    # Traverse in reverse (stability guarantee)
    for x in reversed(arr):
        digit = (x // exp) % 10
        count[digit] -= 1
        output[count[digit]] = x

    return output


# --- Verification ---
assert radix_sort_lsd([170, 45, 75, 90, 802, 24, 2, 66]) == [2, 24, 45, 66, 75, 90, 170, 802]
assert radix_sort_lsd([]) == []
assert radix_sort_lsd([1]) == [1]
```

#### LSD Radix Sort Walkthrough

```
Input: [170, 45, 75, 90, 802, 24, 2, 66]

=== Sort by ones digit (exp=1) ===
  170 → 0, 45 → 5, 75 → 5, 90 → 0
  802 → 2, 24 → 4, 2 → 2, 66 → 6
  Result: [170, 90, 802, 2, 24, 45, 75, 66]

=== Sort by tens digit (exp=10) ===
  170 → 7, 90 → 9, 802 → 0, 2 → 0
  24 → 2, 45 → 4, 75 → 7, 66 → 6
  Result: [802, 2, 24, 45, 66, 170, 75, 90]

=== Sort by hundreds digit (exp=100) ===
  802 → 8, 2 → 0, 24 → 0, 45 → 0
  66 → 0, 170 → 1, 75 → 0, 90 → 0
  Result: [2, 24, 45, 66, 75, 90, 170, 802]

Sort complete!
```

### 4.4 Bucket Sort

Bucket sort divides the range of values into equal intervals (buckets), distributes each element into its corresponding bucket, individually sorts each bucket, and then concatenates them. When data is close to uniformly distributed, elements are distributed nearly evenly across buckets, yielding an expected O(n) complexity.

```python
def bucket_sort(arr: list, num_buckets: int = 10) -> list:
    """
    Bucket Sort: Sorts floating-point numbers in the range [0, 1).

    Complexity:
        - Average: O(n + k) (for uniform distribution)
        - Worst: O(n^2) (all elements in same bucket)
        - Space: O(n + k)
        - Stable: Yes (if internal sort is stable)

    Args:
        arr: List of floating-point numbers in [0, 1)
        num_buckets: Number of buckets
    Returns:
        A new sorted list
    """
    if not arr:
        return []

    # Initialize buckets
    buckets = [[] for _ in range(num_buckets)]

    # Distribute elements to corresponding buckets
    for x in arr:
        bucket_idx = int(x * num_buckets)
        # Handle boundary case (x == 1.0)
        if bucket_idx == num_buckets:
            bucket_idx -= 1
        buckets[bucket_idx].append(x)

    # Sort each bucket individually (insertion sort is appropriate)
    for bucket in buckets:
        bucket.sort()  # Uses Python's Timsort

    # Concatenate all buckets
    result = []
    for bucket in buckets:
        result.extend(bucket)

    return result


# --- Verification ---
import random
test_data = [random.random() for _ in range(20)]
assert bucket_sort(test_data) == sorted(test_data)
```

---

## 5. Hybrid Sorts and Practical Sorting Algorithms

The sorting algorithms used in modern programming language standard libraries are not single algorithms but "hybrid sorts" that combine the strengths of multiple algorithms.

### 5.1 Timsort

Timsort is an algorithm designed by Tim Peters in 2002 as Python's standard sort. It is a hybrid sort combining merge sort and insertion sort. It is now used in Python, Java (object types), JavaScript (V8), Rust (stable sort), and many other languages.

#### Design Philosophy

Based on the observation that real-world data frequently contains "already sorted subsequences." Timsort detects and exploits these "natural runs."

#### Algorithm Overview

1. **Run detection**: Scan the array to find ascending or (strictly) descending contiguous subsequences (runs). Descending runs are reversed to become ascending
2. **Minimum run length guarantee**: If a run is too short (below minrun), extend it by absorbing surrounding elements using insertion sort. minrun is typically in the range 32-64, determined by the array size
3. **Merge strategy**: Push runs onto a stack and merge adjacent runs according to specific conditions (merge policy). The merge policy maintains merge balance and guarantees efficient merge ordering
4. **Galloping mode**: During merging, if elements are consecutively selected from one run, switch to binary-search-based "galloping" for speedup

#### Timsort Performance Characteristics

```
+============+================+====================================+
| Case       | Complexity     | Condition                          |
+============+================+====================================+
| Best       | O(n)           | Already sorted (a single run)      |
+------------+----------------+------------------------------------+
| Average    | O(n log n)     | Random data                        |
+------------+----------------+------------------------------------+
| Worst      | O(n log n)     | Always guaranteed                  |
+------------+----------------+------------------------------------+
| Space      | O(n)           | Temporary space for merging        |
+------------+----------------+------------------------------------+
| Stability  | Yes            | Because it is merge-sort-based     |
+============+================+====================================+
```

### 5.2 Introsort (Introspective Sort)

Introsort was proposed by David Musser in 1997 and is used in C++ STL's `std::sort`. It is a hybrid sort based on quick sort that switches to heap sort when recursion becomes too deep, and uses insertion sort for small subarrays.

#### Algorithm Overview

```
function introsort(arr, maxdepth):
    n = length(arr)
    if n <= 16:
        insertion_sort(arr)         ← Use insertion sort for small inputs
    elif maxdepth == 0:
        heapsort(arr)               ← Fall back to heap sort if too deep
    else:
        pivot = partition(arr)
        introsort(arr[..pivot], maxdepth - 1)
        introsort(arr[pivot+1..], maxdepth - 1)

# Initial maxdepth = 2 * floor(log2(n))
```

#### Performance Characteristics

- **Worst case: O(n log n)** — Guaranteed by fallback to heap sort
- **Average case: O(n log n)** — Maintains quick sort's speed
- **Space: O(log n)** — Recursion stack
- **Stability: No**

### 5.3 pdqsort (Pattern-Defeating Quicksort)

pdqsort was published by Orson Peters in 2021 and is used in Rust's `sort_unstable()` and Go 1.19+'s standard sort. It is an improvement on Introsort that detects data patterns and automatically switches to the optimal strategy.

#### Features

- Sorted data: O(n) (via pattern detection)
- All identical values: O(n) (partition completes immediately)
- Random data: Performance equivalent to quick sort
- Worst case: Falls back to heap sort for O(n log n) guarantee
- Unstable sort

### 5.4 Built-in Sort Comparison Across Languages

```
+==============+===========================+=============================+
| Language/Env | Algorithm                 | Notes                       |
+==============+===========================+=============================+
| Python       | Timsort                   | list.sort(), sorted()       |
+--------------+---------------------------+-----------------------------+
| Java         | Timsort                   | Arrays.sort() for objects   |
+--------------+---------------------------+-----------------------------+
| Java         | Dual-Pivot Quicksort      | Arrays.sort() for primitives|
+--------------+---------------------------+-----------------------------+
| JavaScript   | Timsort (V8)              | Array.prototype.sort()      |
+--------------+---------------------------+-----------------------------+
| C++ (GCC)    | Introsort                 | std::sort()                 |
+--------------+---------------------------+-----------------------------+
| C++ (GCC)    | Timsort-based             | std::stable_sort()          |
+--------------+---------------------------+-----------------------------+
| Rust         | Timsort-based (merge sort)| slice::sort() (stable)      |
+--------------+---------------------------+-----------------------------+
| Rust         | pdqsort                   | slice::sort_unstable()      |
+--------------+---------------------------+-----------------------------+
| Go (1.19+)   | pdqsort                   | sort.Slice()                |
+--------------+---------------------------+-----------------------------+
| Swift        | Timsort-based             | Array.sorted()              |
+--------------+---------------------------+-----------------------------+
```

---

## 6. Sort Stability

### 6.1 What Is Stability?

A stable sort is one that preserves the relative order of elements with equal keys after sorting. This is an extremely important property when performing multi-key sorting or incremental sorting.

#### Concrete Example of Stability

```python
# --- Example demonstrating the importance of stability ---

students = [
    {"name": "Alice",   "score": 85, "class": "A"},
    {"name": "Bob",     "score": 92, "class": "B"},
    {"name": "Charlie", "score": 85, "class": "A"},
    {"name": "Diana",   "score": 92, "class": "B"},
    {"name": "Eve",     "score": 78, "class": "A"},
]

# Stable sort by score:
# Eve(78), Alice(85), Charlie(85), Bob(92), Diana(92)
# → Alice and Charlie with the same score 85 preserve their original order
# → Bob and Diana with the same score 92 also preserve their original order

# With an unstable sort:
# Eve(78), Charlie(85), Alice(85), Diana(92), Bob(92)
# → Order among elements with the same score is not guaranteed
```

### 6.2 Application to Multi-Key Sorting

The most practical application of stable sorts is multi-key sorting. With a stable sort, you can achieve correct multi-key sorting by simply sorting by the secondary key first, then by the primary key.

```python
# Multi-key sort: Sort by secondary key first, then primary key

records = [
    ("Engineering", "Bob"),
    ("Sales",       "Alice"),
    ("Engineering", "Alice"),
    ("Sales",       "Charlie"),
    ("Engineering", "Charlie"),
]

# Step 1: Stable sort by name (secondary key)
step1 = sorted(records, key=lambda x: x[1])
# [('Sales', 'Alice'), ('Engineering', 'Alice'),
#  ('Engineering', 'Bob'),
#  ('Sales', 'Charlie'), ('Engineering', 'Charlie')]

# Step 2: Stable sort by department (primary key)
step2 = sorted(step1, key=lambda x: x[0])
# [('Engineering', 'Alice'), ('Engineering', 'Bob'),
#  ('Engineering', 'Charlie'),
#  ('Sales', 'Alice'), ('Sales', 'Charlie')]

# → Within each department, names are in alphabetical order!
# This works precisely because the sort is stable.

# In Python, you can do this in one step with tuple keys:
result = sorted(records, key=lambda x: (x[0], x[1]))
assert result == step2
```

### 6.3 Classification of Stable and Unstable Sorts

```
Stable sorts:
  - Bubble sort
  - Insertion sort
  - Merge sort
  - Counting sort
  - Radix sort
  - Bucket sort (when internal sort is stable)
  - Timsort

Unstable sorts:
  - Selection sort
  - Quick sort
  - Heap sort
  - Shell sort
  - Introsort
  - pdqsort
```

---

## 7. Specialized Sorts and Applications

### 7.1 External Sort

A technique for sorting data too large to fit in main memory. Data on disk is read in chunks, sorted, and merged.

#### Multi-Way Merge Sort Steps

```
Phase 1: Generate sorted runs
  +----------------------------------------------+
  | Huge file on disk                            |
  | [.... 100GB of data ....]                    |
  +----------------------------------------------+
       ↓ Read in memory-sized chunks
  Chunk 1 (1GB) → In-memory sort → Write sorted run 1 to disk
  Chunk 2 (1GB) → In-memory sort → Write sorted run 2 to disk
  ...
  Chunk 100 (1GB) → In-memory sort → Write sorted run 100 to disk

Phase 2: Multi-way merge
  run 1  ─┐
  run 2  ─┤
  run 3  ─┼─→ k-way merge → Final sorted file
  ...    ─┤
  run 100 ─┘

  k-way merge uses a min-heap:
  - Insert the first element of each run into the heap
  - Extract the minimum from the heap and write to output
  - Insert the next element from the same run into the heap
  - Repeat until all runs are exhausted

  I/O complexity: O((N/B) * log_{M/B}(N/B))
    N = data size, B = block size, M = memory size
```

### 7.2 Parallel Sort

Techniques that leverage multiple processors or cores to accelerate sorting.

**Parallel merge sort**: Sort each subarray independently in separate threads/processes, then merge the results. The merge operation itself can also be parallelized.

**Bitonic sort**: A sorting network specialized for parallel computation. The comparison-swap pattern is data-independent, making it suitable for SIMD architectures such as GPUs. O(n log^2 n) complexity.

**Sample sort**: For large-scale parallel environments (distributed systems). Extracts samples from data to determine pivots and distributes data to processors. MapReduce sorting frameworks are based on this principle.

### 7.3 Partial Sort and Selection Algorithms

When you only need the k-th smallest element rather than sorting the entire array, use a selection algorithm.

```python
def quickselect(arr: list, k: int) -> int:
    """
    QuickSelect: Finds the k-th smallest element in expected O(n) time.
    (Applies quick sort's partition to only one side)

    Complexity:
        - Average: O(n)
        - Worst: O(n^2) (probabilistically avoided with random pivots)

    Args:
        arr: List to search
        k: The rank to find (0-indexed)
    Returns:
        The k-th smallest element
    """
    if len(arr) == 1:
        return arr[0]

    pivot = arr[random.randint(0, len(arr) - 1)]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    if k < len(left):
        return quickselect(left, k)
    elif k < len(left) + len(middle):
        return pivot
    else:
        return quickselect(right, k - len(left) - len(middle))


# --- Verification ---
assert quickselect([3, 1, 4, 1, 5, 9, 2, 6], 0) == 1   # Minimum
assert quickselect([3, 1, 4, 1, 5, 9, 2, 6], 3) == 3    # 4th smallest
```

---

## 8. Anti-Patterns and Pitfalls

### 8.1 Anti-Pattern 1: Choosing an Inappropriate Sorting Algorithm

**Problem**: Selecting a sorting algorithm without considering data characteristics.

```python
# BAD: Using bubble sort on 1 million records
def process_large_dataset(data):
    # Bubble sort is O(n^2) → ~10^12 operations for 1M records
    # Could take hours
    bubble_sort(data)  # Never do this

# GOOD: Use built-in sort
def process_large_dataset(data):
    data.sort()  # Timsort: O(n log n) → ~2 x 10^7 operations for 1M records
    # Completes in milliseconds

# BAD: Using comparison sort on integer data with range 0-255
def sort_byte_values(data):
    return sorted(data)  # O(n log n)

# GOOD: Use counting sort
def sort_byte_values(data):
    return counting_sort(data, max_val=255)  # Completes in O(n)

# BAD: Using quick sort on nearly sorted data
def sort_nearly_sorted(data):
    quicksort_inplace(data, 0, len(data) - 1)
    # May degrade to O(n^2) depending on pivot selection

# GOOD: Use insertion sort (O(n) for nearly sorted data)
def sort_nearly_sorted(data):
    insertion_sort(data)  # Fast because inversion count is low
```

**Guideline**: For large-scale data, using the language's built-in sort is the safest choice. Only consider specialized algorithms when special conditions exist (integers only, limited range, nearly sorted, etc.).

### 8.2 Anti-Pattern 2: Inconsistent Comparison Function

**Problem**: The comparison function for sorting does not satisfy total order conditions (reflexivity, transitivity, antisymmetry).

```python
# BAD: Comparison function that violates transitivity
import functools

# "Rock-paper-scissors" style cyclic comparison
def bad_compare(a, b):
    if a == "rock" and b == "scissors":
        return -1
    if a == "scissors" and b == "paper":
        return -1
    if a == "paper" and b == "rock":
        return -1
    return 1

# Using this with sorted() causes undefined behavior
# sorted(items, key=functools.cmp_to_key(bad_compare))
# → Results may differ between runs, risk of infinite loop

# BAD: Sorting data containing floating-point NaN
import math
data_with_nan = [3.0, float('nan'), 1.0, 2.0]
# sorted(data_with_nan) result is undefined
# NaN returns False for any comparison, so total order does not hold

# GOOD: Remove NaN beforehand or handle in key function
def safe_sort_with_nan(data):
    # Treat NaN as infinity
    return sorted(data, key=lambda x: (math.isnan(x), x if not math.isnan(x) else 0))
```

**Guideline**: Always design comparison functions to satisfy the three conditions of total order. Pay special attention to floating-point NaN and cyclic priority relations.

### 8.3 Anti-Pattern 3: Unnecessary Sorting

**Problem**: Using sorting in situations where it is not needed.

```python
# BAD: Sorting to find the minimum
def find_minimum(data):
    return sorted(data)[0]  # O(n log n) + O(n) space

# GOOD: Use min()
def find_minimum(data):
    return min(data)  # O(n)

# BAD: Sorting to find the k-th element
def find_kth_smallest(data, k):
    return sorted(data)[k]  # O(n log n)

# GOOD: Use QuickSelect or heapq.nsmallest
import heapq
def find_kth_smallest(data, k):
    return heapq.nsmallest(k + 1, data)[-1]  # O(n log k)

# BAD: Sorting for deduplication then comparing adjacent elements
def remove_duplicates(data):
    data.sort()
    return [data[i] for i in range(len(data)) if i == 0 or data[i] != data[i-1]]

# GOOD: Use a set (when order is not needed)
def remove_duplicates(data):
    return list(set(data))  # Expected O(n)

# GOOD: Use dict.fromkeys when order preservation is needed
def remove_duplicates_ordered(data):
    return list(dict.fromkeys(data))  # Expected O(n)
```

---

## 9. Exercises

### Level 1: Fundamentals (Implementation and Understanding)

#### Problem 1-1: Implement Merge Sort

Implement merge sort from scratch and verify its behavior on the following 3 types of input:

- Random array (1000 elements)
- Sorted array (1000 elements)
- Reverse-sorted array (1000 elements)

```python
# Test code
import random
import time

def test_merge_sort():
    sizes = [100, 1000, 5000]
    for n in sizes:
        # Random array
        random_arr = [random.randint(0, 10000) for _ in range(n)]
        assert merge_sort(random_arr) == sorted(random_arr)

        # Sorted array
        sorted_arr = list(range(n))
        assert merge_sort(sorted_arr) == sorted_arr

        # Reverse-sorted array
        reverse_arr = list(range(n, 0, -1))
        assert merge_sort(reverse_arr) == sorted(reverse_arr)

    print("All merge sort tests passed!")

test_merge_sort()
```

#### Problem 1-2: Verify Sort Stability

Implement tests to verify the stability of the following sorting algorithms:

- Insertion sort (confirm it is stable)
- Selection sort (find a counterexample showing instability)

```python
# Hint: Tag elements with identical keys and check their order after sorting
def test_stability():
    data = [(3, 'a'), (1, 'b'), (3, 'c'), (2, 'd'), (1, 'e')]
    # Sort by the first element of each tuple and check the order of the
    # second element for equal keys
    pass  # Implement this
```

### Level 2: Applied (Design Decisions and Optimization)

#### Problem 2-1: Choosing the Right Sort

Select the optimal sorting algorithm for each of the following situations and explain your reasoning.

1. Sort 1 million log entries by timestamp (stability required)
2. 1000 nearly sorted sensor data points
3. 100 million pixel values in the range 0-255
4. 100,000 records on an embedded system with extremely limited memory
5. A real-time system where worst-case response time must be guaranteed

**Model answer summary**:

```
1. Timsort (language built-in sorted())
   - Stable sort, O(n log n) guaranteed, strong on real data

2. Insertion sort
   - Runs in O(n) on nearly sorted data
   - For 1000 elements, even O(n^2) is fast enough

3. Counting sort
   - Value range k=256 is sufficiently small
   - O(n + k) = O(n) sorts 100M elements in linear time

4. Heap sort
   - O(1) additional memory (in-place)
   - O(n log n) guaranteed

5. Heap sort or Introsort
   - Worst-case O(n log n) is guaranteed
   - Quick sort is unsuitable due to worst-case O(n^2)
```

#### Problem 2-2: Custom Sort Implementation

Implement a function that sorts an array of strings such that their concatenation forms the largest number.

```python
# Example: ["3", "30", "34"] → ["34", "3", "30"] (34330 is largest)
# Example: ["9", "97", "972"] → ["9", "972", "97"] (997297 is largest)

from functools import cmp_to_key

def largest_number(nums: list) -> str:
    """
    Sorts an array of number strings so that their concatenation
    produces the largest number.

    Comparison function: Compare a+b with b+a; if a+b > b+a, a comes first.
    This comparison function can be proven to satisfy total order conditions.
    """
    str_nums = [str(n) for n in nums]

    def compare(a, b):
        if a + b > b + a:
            return -1
        elif a + b < b + a:
            return 1
        else:
            return 0

    str_nums.sort(key=cmp_to_key(compare))

    # Handle the case where the leading digit is 0 (all zeros)
    if str_nums[0] == '0':
        return '0'

    return ''.join(str_nums)


# --- Verification ---
assert largest_number([3, 30, 34]) == "34330"
assert largest_number([10, 2]) == "210"
assert largest_number([0, 0]) == "0"
```

### Level 3: Advanced (Theory and Advanced Implementation)

#### Problem 3-1: Merge k Sorted Arrays

Implement an algorithm that merges k sorted arrays into one sorted array. Achieve O(n log k) complexity where n is the total number of elements.

```python
import heapq

def merge_k_sorted_arrays(arrays: list) -> list:
    """
    Merge k sorted arrays in O(n log k).
    Uses a min-heap to always extract the minimum among the first elements
    of all arrays.

    Args:
        arrays: List of sorted arrays
    Returns:
        A single merged sorted array
    """
    result = []
    # Store (value, array index, element index) tuples in the heap
    heap = []

    for i, arr in enumerate(arrays):
        if arr:
            heapq.heappush(heap, (arr[0], i, 0))

    while heap:
        val, arr_idx, elem_idx = heapq.heappop(heap)
        result.append(val)

        # Add the next element from the same array to the heap
        if elem_idx + 1 < len(arrays[arr_idx]):
            next_val = arrays[arr_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, arr_idx, elem_idx + 1))

    return result


# --- Verification ---
arrays = [
    [1, 4, 5],
    [1, 3, 4],
    [2, 6],
]
assert merge_k_sorted_arrays(arrays) == [1, 1, 2, 3, 4, 4, 5, 6]
```

#### Problem 3-2: Counting Inversions

Implement an algorithm that computes the inversion count of an array in O(n log n) using a merge sort variant.

```python
def count_inversions(arr: list) -> int:
    """
    Count inversions in O(n log n).
    During the merge operation, when an element from the right array is
    selected, the count of remaining elements in the left array is added
    to the inversion count.

    Inversion: A pair (i, j) where i < j and arr[i] > arr[j]

    Args:
        arr: List of integers
    Returns:
        The number of inversions
    """
    if len(arr) <= 1:
        return 0

    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]

    # Recursively count inversions in left and right halves
    inv_count = count_inversions(left) + count_inversions(right)

    # Count split inversions during merge
    i = j = k = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            arr[k] = left[i]
            i += 1
        else:
            arr[k] = right[j]
            # All elements in left[i:] are greater than right[j] → inversions
            inv_count += len(left) - i
            j += 1
        k += 1

    while i < len(left):
        arr[k] = left[i]
        i += 1
        k += 1

    while j < len(right):
        arr[k] = right[j]
        j += 1
        k += 1

    return inv_count


# --- Verification ---
assert count_inversions([1, 2, 3, 4, 5]) == 0      # Sorted: no inversions
assert count_inversions([5, 4, 3, 2, 1]) == 10     # Reverse: n(n-1)/2 = 10
assert count_inversions([2, 4, 1, 3, 5]) == 3      # (2,1), (4,1), (4,3)
```

#### Problem 3-3: Simplified Timsort Implementation

Implement the core parts of Timsort (run detection + short run extension with insertion sort + merging) in a simplified form.

```python
# Hints:
# 1. find_runs(): Detect ascending/descending runs (reverse descending ones)
# 2. ensure_min_run(): If a run is shorter than minrun, extend with insertion sort
# 3. merge_runs(): Push runs onto a stack and merge according to policy
# minrun calculation: top 6 bits of n + OR of remaining bits
```

---

## 10. FAQ (Frequently Asked Questions)

### Q1: Why is bubble sort taught in educational settings?

**A**: Bubble sort is an excellent teaching tool for learning fundamental algorithm concepts. It provides the simplest form of experiencing comparison, swapping, loop invariants, worst-case analysis, and optimization through early exit. Furthermore, contrasting bubble sort's inefficiency with O(n log n) algorithms allows students to viscerally experience the dramatic impact of algorithm selection on performance. However, bubble sort should never be used in practice. It should be emphasized that it exists solely for educational purposes. Donald Knuth himself stated that "bubble sort seems to have nothing to recommend it, except a catchy name."

### Q2: Is quick sort's worst case O(n^2) a problem in practice?

**A**: With proper pivot selection strategies, the probability of encountering the worst case is extremely low. With random pivot selection, the probability is 1/n!. Furthermore, practical implementations are hybridized. C++'s Introsort switches to heap sort when recursion depth exceeds 2 * log(n), and Rust's pdqsort detects patterns and automatically selects the optimal strategy. Therefore, while caution is needed when using a custom quick sort implementation, this problem is resolved when using language built-in sorts.

### Q3: How do you sort 1 billion records?

**A**: The strategy differs based on whether the data fits in main memory. If it fits in memory, language-standard sorts (Timsort, etc.) are sufficient. If it does not fit, use external sort. Divide the data into memory-sized chunks, sort each chunk and write to disk, then combine via multi-way merge. In distributed environments, frameworks like MapReduce or Apache Spark's sortBy can be used. These internally distribute data to nodes based on sample sort principles and sort in parallel.

### Q4: What is the difference between Python's sorted() and list.sort()?

**A**: Both use Timsort, but their behavior differs. `sorted()` is a non-destructive function that returns a new list and can be applied to any iterable. `list.sort()` is a destructive method that modifies the list in-place and can only be applied to list objects. Use `list.sort()` for memory efficiency, and `sorted()` when you want to preserve the original data. Note that `sorted()` internally creates a list and then calls `list.sort()`, so it has slightly more overhead.

### Q5: When is sort "stability" important?

**A**: Typical situations where stability is important are: (1) When sorting database query results by multiple columns. For example, "first by name, then by score" can be achieved by applying a stable sort twice. (2) In UI table sorting, when users click column headers sequentially for multi-key sorting. (3) When radix sort's internal per-digit sort requires a stable sort. Unstable sorts may destroy previously sorted results. Python's `sorted()` and `list.sort()` are stable sorts, so they can be used safely in these situations.

### Q6: Why aren't O(n) sorts always used?

**A**: O(n) sorts (counting sort, radix sort, bucket sort) have applicability conditions. Counting sort is efficient only when the value range k is sufficiently small relative to n; when k >> n, both space and time are wasted. Radix sort is applicable only to fixed-length integers or strings, and cannot be directly applied to floating-point numbers or variable-length objects. Bucket sort is efficient only when data is close to uniformly distributed. Additionally, these algorithms cannot customize comparison functions, so they cannot handle complex sorting criteria. From a generality standpoint, comparison-based sorts are widely used as the standard.

---

## 11. Intuitive Understanding of Complexity — Estimated Sort Times

In practical programming, estimating sort duration for a given data size is important. The following table shows approximate operation counts for each complexity class (assuming ~10^8 to 10^9 basic operations per second).

```
+==========+==============+==============+==============+==============+
| n        | O(n)         | O(n log n)   | O(n^2)       | O(n^2) ratio |
|          | operations   | operations   | operations   | vs O(n logn) |
+==========+==============+==============+==============+==============+
| 10       | 10           | ~33          | 100          | 3x           |
+----------+--------------+--------------+--------------+--------------+
| 100      | 100          | ~664         | 10,000       | 15x          |
+----------+--------------+--------------+--------------+--------------+
| 1,000    | 1,000        | ~9,966       | 1,000,000    | 100x         |
+----------+--------------+--------------+--------------+--------------+
| 10,000   | 10,000       | ~132,877     | 100,000,000  | 752x         |
+----------+--------------+--------------+--------------+--------------+
| 100,000  | 100,000      | ~1,660,964   | 10^10        | 6,020x       |
+----------+--------------+--------------+--------------+--------------+
| 1,000,000| 1,000,000    | ~19,931,569  | 10^12        | 50,172x      |
+----------+--------------+--------------+--------------+--------------+
```

For n = 1 million, O(n^2) requires about 50,000 times more operations than O(n log n). This is why "you must never use bubble sort on 1 million records."

---

## 12. Historical Background of Sorting Algorithms

The history of sorting algorithms is deeply intertwined with the history of computer science itself.

```
1945    John von Neumann described merge sort as an EDVAC program
        (one of the first computer programs)

1959    Donald Shell published shell sort
        (introduced the concept of gapped insertion sort)

1960    C.A.R. Hoare published quick sort
        (considered "one of the most important algorithms of the 20th century")

1964    J.W.J. Williams published heap sort
        (simultaneously introducing the heap data structure)

1969    Harold H. Seward formalized counting sort and radix sort

1973    Donald Knuth published "The Art of Computer Programming" Vol.3
        (an encyclopedic work on sorting and searching)

1993    Ingo Wegener proved the lower bound on worst-case comparisons for heap sort

1997    David Musser proposed Introsort
        (adopted by C++ STL)

2002    Tim Peters designed Timsort
        (adopted as Python's standard sort)

2009    Java 7 adopted Vladimir Yaroslavskiy's Dual-Pivot Quicksort

2021    Orson Peters published pdqsort
        (adopted by Rust, Go)
```

---


## FAQ

### Q1: What is the most important point in learning this topic?

Gaining practical experience is most important. Understanding deepens not just through theory, but by actually writing code and observing its behavior.

### Q2: What common mistakes do beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving on.

### Q3: How is this applied in practice?

Knowledge of this topic is frequently used in everyday development work. It is particularly important during code reviews and architecture design.

---

## Summary

This chapter systematically covered sorting algorithms from fundamentals to applications. The key points are summarized below.

### Core Understanding

1. **Comparison-based sorting has a theoretical lower bound of Omega(n log n)**. Merge sort, heap sort, and Timsort are optimal algorithms that asymptotically reach this bound.

2. **Non-comparison-based sorts can break this lower bound** but have applicability conditions (integers, fixed range, etc.). Counting sort runs in O(n + k); radix sort runs in O(d(n + k)).

3. **Using language built-in sorts is the best practice**. Hybrid sorts like Timsort, Introsort, and pdqsort combine theoretical guarantees with practical performance.

4. **Algorithm selection depends on data characteristics**. Consider data volume, distribution, existing sort order, memory constraints, and stability requirements.

### Algorithm Selection Guidelines

```
Small data (n < 50)                → Insertion sort
Nearly sorted data                 → Insertion sort or Timsort
Stability required                 → Merge sort or Timsort
Strict memory constraints          → Heap sort
Worst-case guarantee required      → Heap sort or Introsort
Integers with small value range    → Counting sort
Fixed-length integers/strings      → Radix sort
General (no special constraints)   → Language built-in (Timsort, etc.)
Data too large for memory          → External sort
```

---

## 13. Important Theorems and Supplementary Notes on Sorting

### 13.1 Optimal Number of Comparisons

The "minimum number of comparisons" to sort n elements is given by the information-theoretic lower bound ceil(log_2(n!)). This is the lower bound on the height of the decision tree itself.

```
n = 1:  0 comparisons (none needed)
n = 2:  1 comparison   ceil(log_2(2!))  = ceil(1)    = 1
n = 3:  3 comparisons  ceil(log_2(3!))  = ceil(2.58) = 3
n = 4:  5 comparisons  ceil(log_2(4!))  = ceil(4.58) = 5
n = 5:  7 comparisons  ceil(log_2(5!))  = ceil(6.91) = 7
n = 10: 22 comparisons ceil(log_2(10!)) = ceil(21.79)= 22
n = 12: 29 comparisons ceil(log_2(12!)) = ceil(28.84)= 29

Ford-Johnson algorithm (merge-insertion sort, 1959):
  A sorting algorithm that achieves the minimum number of comparisons
  for small n. Proven optimal for n <= 12.
  Not practical but theoretically important.
```

### 13.2 Sorting Networks

A sorting network is a computational model that sorts using a fixed, data-independent comparison-swap pattern. It has high affinity with parallel computation.

```
Sorting network for n = 4 (optimal: 5 comparators)

Input   ──*──────*──────── Output (min)
         |      |
         *──*──────*───── Output
            |      |
         *──*──────*───── Output
         |      |
Input   ──*──────*──────── Output (max)

Each * represents a comparator (compare and swap if necessary).
Signals flow left to right; comparators in each column can execute in parallel.
Depth (number of columns) = number of parallel steps

AKS Network (Ajtai-Komlos-Szemeredi, 1983):
  Proved existence of a sorting network with O(n log n) comparators
  and O(log n) depth.
  However, the constant factor is enormous, making it impractical.

Batcher's Odd-Even Merge Network:
  O(n log^2 n) comparators, O(log^2 n) depth.
  Used in practical parallel sorting.
```

### 13.3 Breaking the Lower Bound — Information-Theoretic Interpretation of Non-Comparison Sorts

In comparison-based sorting, each comparison yields at most 1 bit of information. Distinguishing among n! permutations requires log_2(n!) bits, which is the essence of the Omega(n log n) lower bound.

Non-comparison sorts acquire more than 1 bit per operation. For example, counting sort reads an element's value to obtain log_2(k) bits of information at once, enabling it to break the comparison lower bound.

### 13.4 Sorting Lower Bounds and Computational Models

```
+========================+=============================+===============+
| Computational Model    | Sorting Lower Bound         | Representative|
+========================+=============================+===============+
| Comparison model       | Omega(n log n)              | Merge, QS     |
+------------------------+-----------------------------+---------------+
| Integer RAM model      | Omega(n * sqrt(log log n))  | Han-Thorup    |
| (word size w = O(logn))|                             | (2002, theory)|
+------------------------+-----------------------------+---------------+
| External memory model  | Omega((n/B)*log_{M/B}(n/B)) | Multi-way     |
|                        |                             | merge         |
+------------------------+-----------------------------+---------------+
| Parallel comparison    | Omega(log n) depth          | AKS network   |
| model (p = n procs)    |                             |               |
+------------------------+-----------------------------+---------------+
```

---

## 14. Implementation Tricks and Optimization Techniques

### 14.1 Sorting Pre-processing and Post-processing

```python
# --- Implementation technique: Sentinel values ---
# Using sentinels in the merge operation eliminates boundary checks

def merge_with_sentinel(arr, left, mid, right):
    """Merge operation using sentinels. Eliminates boundary checks."""
    L = arr[left:mid + 1] + [float('inf')]    # Sentinel (infinity)
    R = arr[mid + 1:right + 1] + [float('inf')]

    i = j = 0
    for k in range(left, right + 1):
        if L[i] <= R[j]:
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1
```

### 14.2 Tail Call Elimination for Quick Sort Optimization

```python
def quicksort_optimized(arr: list, low: int, high: int) -> None:
    """
    Quick sort with tail call elimination.
    Recursively processes the shorter subarray and loops on the longer one.
    → Guarantees O(log n) stack usage.
    """
    while low < high:
        pivot_idx = _partition_random(arr, low, high)

        # Recurse on the shorter side, loop on the longer side
        if pivot_idx - low < high - pivot_idx:
            quicksort_optimized(arr, low, pivot_idx - 1)
            low = pivot_idx + 1   # Convert tail recursion to loop
        else:
            quicksort_optimized(arr, pivot_idx + 1, high)
            high = pivot_idx - 1  # Convert tail recursion to loop
```

### 14.3 3-Way Partition (Dutch National Flag)

Improves quick sort performance on data with many duplicates. Groups elements equal to the pivot in the middle.

```python
def quicksort_3way(arr: list, low: int, high: int) -> None:
    """
    3-way partition quick sort.
    Uses Dijkstra's Dutch National Flag algorithm.
    For data with many duplicates, may improve from O(n log n) to O(n).
    """
    if low >= high:
        return

    # 3-way partition: [< pivot | == pivot | > pivot]
    lt = low      # arr[low..lt-1] < pivot
    gt = high     # arr[gt+1..high] > pivot
    i = low       # arr[lt..i-1] == pivot
    pivot = arr[low]

    while i <= gt:
        if arr[i] < pivot:
            arr[lt], arr[i] = arr[i], arr[lt]
            lt += 1
            i += 1
        elif arr[i] > pivot:
            arr[gt], arr[i] = arr[i], arr[gt]
            gt -= 1
            # Do not advance i (need to check the swapped element)
        else:
            i += 1

    # arr[lt..gt] are all equal to pivot → no sorting needed
    quicksort_3way(arr, low, lt - 1)
    quicksort_3way(arr, gt + 1, high)


# --- Verification ---
test = [4, 2, 4, 1, 4, 3, 4, 2]
quicksort_3way(test, 0, len(test) - 1)
assert test == [1, 2, 2, 3, 4, 4, 4, 4]
```

---

## 15. Applied Sorting Problems

Here are some applied problems that use sorting as a tool. Beyond knowledge of sorting algorithms themselves, the perspective of "using sorting to efficiently solve other problems" is important.

### 15.1 Interval Overlap Detection

```python
def has_overlap(intervals: list) -> bool:
    """
    Determine if any intervals in the list overlap.
    Sort intervals by start point and check adjacent intervals for overlap.

    Complexity: O(n log n) (dominated by sorting)

    Args:
        intervals: List of [(start, end), ...]
    Returns:
        True if overlap exists
    """
    if len(intervals) <= 1:
        return False

    # Sort by start point (by end point if start is the same)
    sorted_intervals = sorted(intervals, key=lambda x: (x[0], x[1]))

    for i in range(1, len(sorted_intervals)):
        # If the previous interval's end > current interval's start → overlap
        if sorted_intervals[i - 1][1] > sorted_intervals[i][0]:
            return True

    return False


assert has_overlap([(1, 5), (3, 7), (8, 10)]) == True   # (1,5) and (3,7) overlap
assert has_overlap([(1, 3), (4, 6), (7, 9)]) == False    # No overlap
```

### 15.2 Application to the Closest Pair Problem

```
Finding the closest pair among n points on a 2D plane.

Naive: O(n^2) — compute distance for all pairs
Divide and conquer + sort: O(n log n)

Steps:
  1. Sort points by x-coordinate: O(n log n)
  2. Split into left and right halves at the center
  3. Recursively find the closest pair in each half
  4. Check pairs spanning the dividing line (scan strip region sorted by y)
  5. Return the overall minimum distance

An excellent example of sorting playing an essential role in structuring the problem.
```

### 15.3 Basic Computational Geometry Operations Using Sort

```python
def closest_pair_1d(points: list) -> tuple:
    """
    Find the closest pair on a 1D line in O(n log n).
    After sorting, simply check differences between adjacent elements.
    """
    if len(points) < 2:
        return None

    sorted_pts = sorted(points)
    min_dist = float('inf')
    closest = None

    for i in range(1, len(sorted_pts)):
        dist = sorted_pts[i] - sorted_pts[i - 1]
        if dist < min_dist:
            min_dist = dist
            closest = (sorted_pts[i - 1], sorted_pts[i])

    return closest


assert closest_pair_1d([7, 1, 3, 10, 4]) == (3, 4)
```

---

## 16. Sorting Benchmark Guidelines

Guidelines for correctly evaluating sorting algorithm performance.

### 16.1 Input Patterns to Test

```
+=============================+==========================================+
| Pattern                     | Purpose                                  |
+=============================+==========================================+
| Random (uniform dist.)      | Measure average performance              |
+-----------------------------+------------------------------------------+
| Sorted (ascending)          | Verify best case                         |
+-----------------------------+------------------------------------------+
| Reverse (descending)        | Candidate for worst case                 |
+-----------------------------+------------------------------------------+
| All identical values        | Duplicate handling performance            |
+-----------------------------+------------------------------------------+
| Nearly sorted               | Verify adaptivity                        |
| (few random swaps)          |                                          |
+-----------------------------+------------------------------------------+
| Pipe organ                  | [1,2,...,n,...,2,1]                       |
| (mountain shape)            | Handling partially sorted data           |
+-----------------------------+------------------------------------------+
| Few unique values           | Handling data with many duplicates       |
+-----------------------------+------------------------------------------+
| Disorder at head/tail only  | Handling localized disorder              |
+-----------------------------+------------------------------------------+
```

### 16.2 Benchmarking Considerations

1. **Warm-up**: Perform warm-up runs before measurement to eliminate JIT compilation and cache effects
2. **Multiple measurements**: Run the same condition multiple times and take the median for statistically significant results
3. **Input copies**: For in-place sorts, create copies to avoid reusing the same input
4. **GC impact**: Account for garbage collection impact (in Python, temporarily disable with `gc.disable()`)
5. **Gradual data size variation**: Vary n as 10, 100, 1000, ... to confirm complexity trends

---

## Recommended Next Guides


---

## References

1. Cormen, T. H., Leiserson, C. E., Rivest, R. L., and Stein, C. "Introduction to Algorithms." 4th Edition, MIT Press, 2022. Chapters 6-9. (Standard textbook comprehensively covering theoretical foundations and proofs of sorting algorithms)

2. Sedgewick, R. and Wayne, K. "Algorithms." 4th Edition, Addison-Wesley, 2011. Chapter 2. (Implementation-oriented sorting algorithm explanation with extensive Java code examples)

3. Knuth, D. E. "The Art of Computer Programming, Volume 3: Sorting and Searching." 2nd Edition, Addison-Wesley, 1998. (Encyclopedic reference on sorting and searching with detailed historical background and mathematical analysis)

4. Peters, T. "Timsort description." CPython Developer Documentation, 2002. https://github.com/python/cpython/blob/main/Objects/listsort.txt (Timsort design document with detailed implementation insights)

5. Musser, D. "Introspective Sorting and Selection Algorithms." Software: Practice and Experience, 27(8), pp. 983-993, 1997. (Original Introsort paper on avoiding quick sort's worst case)

6. Peters, O. "Pattern-defeating Quicksort." arXiv:2106.05123, 2021. (pdqsort paper on data pattern detection and adaptive strategy switching)
