# Sorting Algorithms

> Systematically understand fundamental algorithms for rearranging data into a specific order, from the perspectives of computational complexity, stability, and implementation

## What You Will Learn

1. Compare the principles, implementations, and complexities of **7 sorting algorithms**
2. Select the optimal sort for a given scenario based on differences in **stability, in-place property, and adaptivity**
3. Understand how different paradigms such as **divide and conquer, heaps, and counting** are applied to sorting
4. Grasp the design philosophy behind modern practical sorts like **TimSort and Introsort**
5. Understand the fundamentals of advanced topics such as **external sorting and parallel sorting**


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts

---

## Table of Contents

1. [Overview of Sorting](#1-overview-of-sorting)
2. [Bubble Sort](#2-bubble-sort)
3. [Selection Sort](#3-selection-sort)
4. [Insertion Sort](#4-insertion-sort)
5. [Merge Sort](#5-merge-sort)
6. [Quick Sort](#6-quick-sort)
7. [Heap Sort](#7-heap-sort)
8. [Non-Comparison-Based Sorts](#8-non-comparison-based-sorts)
9. [Advanced Sorting Algorithms](#9-advanced-sorting-algorithms)
10. [Complexity Comparison Table and Use-Case Selection Guide](#10-complexity-comparison-table-and-use-case-selection-guide)
11. [Anti-Patterns](#11-anti-patterns)
12. [Exercises](#12-exercises)
13. [FAQ](#13-faq)
14. [Summary](#14-summary)
15. [References](#15-references)

---

## 1. Overview of Sorting

### 1.1 What Is Sorting?

Sorting is the operation of rearranging a collection of data according to a specific ordering relation. It is one of the most fundamental and important problems in computer science, used everywhere from search preprocessing, data visualization, and duplicate removal to statistical processing.

Donald Knuth estimated in *The Art of Computer Programming* that "more than 25% of all computer computation time is spent on sorting." Even in modern systems, sorting is ubiquitous -- from database index construction and file system directory listings to search engine rankings.

### 1.2 Classification of Sorting Algorithms

```
+----------------------------------------------------------------------+
|              Classification of Sorting Algorithms                     |
+----------------------------------------------------------------------+
|                                                                      |
|  +-------------------------------------------------------------+    |
|  |          Comparison-Based Sorts                              |    |
|  |          Theoretical lower bound: Omega(n log n)             |    |
|  +-----------------------+-------------------------------------+    |
|  |  Simple: O(n^2)       |  Efficient: O(n log n)              |    |
|  |  - Bubble Sort        |  - Merge Sort                       |    |
|  |  - Selection Sort     |  - Quick Sort                       |    |
|  |  - Insertion Sort     |  - Heap Sort                        |    |
|  |  - Shell Sort         |  - TimSort (hybrid)                 |    |
|  |                       |  - Introsort (hybrid)               |    |
|  +-----------------------+-------------------------------------+    |
|                                                                      |
|  +-------------------------------------------------------------+    |
|  |          Non-Comparison-Based Sorts                          |    |
|  |          Can achieve O(n) under certain conditions            |    |
|  +-------------------------------------------------------------+    |
|  |  - Counting Sort    -- integers, small range                 |    |
|  |  - Radix Sort       -- fixed-length keys                    |    |
|  |  - Bucket Sort      -- uniform distribution                 |    |
|  +-------------------------------------------------------------+    |
+----------------------------------------------------------------------+
```

### 1.3 Important Properties of Sorting

When evaluating sorting algorithms, it is necessary to consider not only time complexity but also the following properties.

**Stability**

A stable sort guarantees that the relative order of elements with equal keys is preserved before and after sorting.

```
Example of stable sort:
Input:  [(3,"Alice"), (1,"Bob"), (3,"Charlie"), (2,"Dave")]
        Sort by key
Output: [(1,"Bob"), (2,"Dave"), (3,"Alice"), (3,"Charlie")]
        ^ Relative order of Alice and Charlie is preserved

Example of unstable sort:
Output: [(1,"Bob"), (2,"Dave"), (3,"Charlie"), (3,"Alice")]
        ^ Relative order of Alice and Charlie may be reversed
```

Stability becomes important when sorting by multiple keys. For example, when first sorting by name and then by grade, a stable sort ensures that students with the same grade remain in name order.

**In-Place Property**

A sort is called in-place when its additional memory usage is O(1) or O(log n). The in-place property is important in embedded systems and environments with tight memory constraints.

**Adaptivity**

A sort is called adaptive when its complexity improves for nearly sorted input. Insertion Sort is a typical example, completing in O(n) for nearly sorted data.

### 1.4 Theoretical Lower Bound of Comparison-Based Sorts

Comparison-based sorting algorithms have a theoretical lower bound of **O(n log n)**. This is proven using the decision tree model.

```
Decision tree for n=3 (elements a, b, c):

                    a < b ?
                   /       \
                yes         no
               /               \
          b < c ?             a < c ?
         /     \             /     \
       yes     no          yes     no
       /         \         /         \
   [a,b,c]    a < c ?  [b,a,c]   b < c ?
              /     \            /     \
            yes     no         yes     no
            /         \        /         \
        [a,c,b]    [c,a,b] [b,c,a]   [c,b,a]

Number of leaves = n! = 6 (all permutations of 3 elements)
Height of tree >= log2(n!) = Omega(n log n)   (by Stirling's approximation)
```

There are n! permutations of n elements, and each leaf of the decision tree corresponds to one of them. The height of the tree (= worst-case number of comparisons) is at least log2(n!), and from Stirling's approximation n! ~ (n/e)^n, we derive log2(n!) = Omega(n log n).

---

## 2. Bubble Sort

### 2.1 Algorithm Principle

Repeatedly compare and swap adjacent elements, "bubbling" the largest value toward the end. The name comes from the way large values move toward the end of the array like rising bubbles.

### 2.2 Visualization

```
Initial array: [5, 3, 8, 1, 2]

Pass 1: Unsorted portion [5, 3, 8, 1, 2]
  Compare 5>3 -> swap  [3, 5, 8, 1, 2]
  Compare 5<8 -> keep   [3, 5, 8, 1, 2]
  Compare 8>1 -> swap  [3, 5, 1, 8, 2]
  Compare 8>2 -> swap  [3, 5, 1, 2, 8]  <- 8 reaches its final position
                                  ~~~~~~~~

Pass 2: Unsorted portion [3, 5, 1, 2] | Settled [8]
  Compare 3<5 -> keep   [3, 5, 1, 2, 8]
  Compare 5>1 -> swap  [3, 1, 5, 2, 8]
  Compare 5>2 -> swap  [3, 1, 2, 5, 8]  <- 5 reaches its final position
                             ~~~~~

Pass 3: Unsorted portion [3, 1, 2] | Settled [5, 8]
  Compare 3>1 -> swap  [1, 3, 2, 5, 8]
  Compare 3>2 -> swap  [1, 2, 3, 5, 8]  <- 3 reaches its final position
                          ~~~~~~~~

Pass 4: Unsorted portion [1, 2] | Settled [3, 5, 8]
  Compare 1<2 -> keep   [1, 2, 3, 5, 8]  <- No swaps -> done!

Result: [1, 2, 3, 5, 8]
```

### 2.3 Implementation (Python)

```python
def bubble_sort(arr: list) -> list:
    """Bubble Sort - stable, in-place

    Compares and swaps adjacent elements, moving the maximum to the end repeatedly.
    Optimization: Early termination if no swaps occur in a single pass.

    Args:
        arr: List to sort (modified in place)

    Returns:
        Sorted list (same object as input)

    Complexity:
        Best:    O(n)   -- input is already sorted
        Average: O(n^2)
        Worst:   O(n^2)
        Space:   O(1)
    """
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(n - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr


# --- Test ---
if __name__ == "__main__":
    data = [64, 34, 25, 12, 22, 11, 90]
    print(f"Input: {data}")
    result = bubble_sort(data)
    print(f"Output: {result}")
    # Input: [64, 34, 25, 12, 22, 11, 90]
    # Output: [11, 12, 22, 25, 34, 64, 90]

    # Already sorted data -> completes in O(n)
    sorted_data = [1, 2, 3, 4, 5]
    bubble_sort(sorted_data)
    print(f"Already sorted: {sorted_data}")  # [1, 2, 3, 4, 5]
```

### 2.4 Complexity Analysis

| Case | Comparisons | Swaps | Description |
|:---|:---|:---|:---|
| Best | O(n) | 0 | Input is already sorted (early termination via swapped flag) |
| Average | O(n^2) | O(n^2) | Random input |
| Worst | O(n^2) | O(n^2) | Reverse-sorted input |

The worst-case number of comparisons for Bubble Sort is n(n-1)/2, and the number of swaps equals the inversion count. An inversion is a pair where i < j and arr[i] > arr[j].

### 2.5 Variation: Cocktail Shaker Sort

Bubble Sort scans in only one direction, whereas Cocktail Shaker Sort alternates scanning forward and backward. This mitigates the "turtle problem" (slow movement of small values at the end of the array).

```python
def cocktail_shaker_sort(arr: list) -> list:
    """Cocktail Shaker Sort (Bidirectional Bubble Sort)

    An improved version of Bubble Sort that alternates forward and backward scans.
    Movement of small values near the end is accelerated.

    Worst-case complexity remains O(n^2), but the constant factor may improve.
    """
    n = len(arr)
    start = 0
    end = n - 1
    swapped = True

    while swapped:
        swapped = False

        # Forward scan: move maximum to the end
        for i in range(start, end):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swapped = True
        end -= 1

        if not swapped:
            break

        swapped = False

        # Backward scan: move minimum to the front
        for i in range(end, start, -1):
            if arr[i] < arr[i - 1]:
                arr[i], arr[i - 1] = arr[i - 1], arr[i]
                swapped = True
        start += 1

    return arr


# --- Test ---
if __name__ == "__main__":
    data = [5, 1, 4, 2, 8, 0, 2]
    print(cocktail_shaker_sort(data))  # [0, 1, 2, 2, 4, 5, 8]
```

---

## 3. Selection Sort

### 3.1 Algorithm Principle

Find the minimum value in the unsorted portion and swap it with the element at the beginning of the unsorted portion. By repeating this operation, the sorted portion expands from the beginning.

### 3.2 Visualization

```
Initial array: [29, 10, 14, 37, 13]

Step 1: Unsorted [29, 10, 14, 37, 13]
  Minimum = 10 (index 1)
  Swap 29 and 10
  Result: [10 | 29, 14, 37, 13]
         ~~

Step 2: Unsorted [29, 14, 37, 13]
  Minimum = 13 (index 4)
  Swap 29 and 13
  Result: [10, 13 | 14, 37, 29]
             ~~

Step 3: Unsorted [14, 37, 29]
  Minimum = 14 (index 2) -- itself
  No swap needed
  Result: [10, 13, 14 | 37, 29]
                 ~~

Step 4: Unsorted [37, 29]
  Minimum = 29 (index 4)
  Swap 37 and 29
  Result: [10, 13, 14, 29 | 37]
                     ~~

Done: [10, 13, 14, 29, 37]
```

### 3.3 Implementation

```python
def selection_sort(arr: list) -> list:
    """Selection Sort - unstable, in-place

    Finds the minimum in the unsorted portion and places it at the front, repeatedly.
    The number of swaps is always O(n), which is advantageous when write cost is high.

    Args:
        arr: List to sort

    Returns:
        Sorted list

    Complexity:
        Best/Average/Worst: O(n^2) -- always the same regardless of input
        Space: O(1)
    """
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        if min_idx != i:
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr


# --- Test ---
if __name__ == "__main__":
    data = [29, 10, 14, 37, 13]
    print(selection_sort(data))  # [10, 13, 14, 29, 37]
```

### 3.4 Why Selection Sort Is Unstable

Let us confirm with a concrete example that Selection Sort is unstable.

```
Input: [3a, 2, 3b, 1]    (3a and 3b have the same key value 3)

Step 1: Minimum = 1, swap 3a and 1
  [1, 2, 3b, 3a]    <- Relative order of 3a and 3b is reversed!

Result: [1, 2, 3b, 3a]    <- Unstable
```

Because the swap operation exchanges elements at distant positions, the relative order of elements with equal keys can be disrupted.

### 3.5 Characteristics and Use Cases of Selection Sort

Selection Sort is not outstanding in terms of complexity, but it is useful in specific situations due to the following characteristics.

- **O(n) swaps**: While the number of comparisons is O(n^2), at most one swap occurs per step. This can be advantageous for media with high write costs (e.g., flash memory).
- **Simplicity of implementation**: Small code size, less prone to bugs.
- **Predictable performance**: Always the same complexity regardless of input data.

---

## 4. Insertion Sort

### 4.1 Algorithm Principle

Like sorting a hand of playing cards, take the first element from the unsorted portion and insert it into the correct position within the sorted portion. This is the sorting technique closest to how humans naturally sort.

### 4.2 Visualization

```
Initial: [5, 2, 4, 6, 1, 3]
       Sorted|Unsorted

i=1: key=2
  Insert 2 into sorted [5]
  5 > 2 -> shift 5 right
  Insert 2 at position 0
  Result: [2, 5 | 4, 6, 1, 3]
         ~~~~

i=2: key=4
  Insert 4 into sorted [2, 5]
  5 > 4 -> shift 5 right
  2 < 4 -> stop
  Insert 4 at position 1
  Result: [2, 4, 5 | 6, 1, 3]
         ~~~~~~~

i=3: key=6
  Insert 6 into sorted [2, 4, 5]
  5 < 6 -> stop (no shift needed)
  Result: [2, 4, 5, 6 | 1, 3]
         ~~~~~~~~~~

i=4: key=1
  Insert 1 into sorted [2, 4, 5, 6]
  6 > 1 -> shift right
  5 > 1 -> shift right
  4 > 1 -> shift right
  2 > 1 -> shift right
  Insert 1 at position 0
  Result: [1, 2, 4, 5, 6 | 3]
         ~~~~~~~~~~~~~

i=5: key=3
  Insert 3 into sorted [1, 2, 4, 5, 6]
  6 > 3 -> shift right
  5 > 3 -> shift right
  4 > 3 -> shift right
  2 < 3 -> stop
  Insert 3 at position 2
  Result: [1, 2, 3, 4, 5, 6]
         ~~~~~~~~~~~~~~~~

Done!
```

### 4.3 Implementation

```python
def insertion_sort(arr: list) -> list:
    """Insertion Sort - stable, in-place, adaptive

    Inserts unsorted elements one by one into the sorted portion.
    Very fast for nearly sorted data (close to O(n)).

    Args:
        arr: List to sort

    Returns:
        Sorted list

    Complexity:
        Best:    O(n)   -- input is already sorted
        Average: O(n^2)
        Worst:   O(n^2) -- input is in reverse order
        Space:   O(1)
    """
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr


# --- Test ---
if __name__ == "__main__":
    # Random data
    data = [5, 2, 4, 6, 1, 3]
    print(insertion_sort(data))  # [1, 2, 3, 4, 5, 6]

    # Nearly sorted data -> close to O(n)
    nearly_sorted = [1, 3, 2, 4, 6, 5]
    print(insertion_sort(nearly_sorted))  # [1, 2, 3, 4, 5, 6]
```

### 4.4 Binary Insertion Sort

By using binary search to find the insertion position, the number of comparisons can be reduced to O(n log n). However, since the element shifting operations remain O(n^2), the overall complexity stays at O(n^2).

```python
import bisect

def binary_insertion_sort(arr: list) -> list:
    """Binary Insertion Sort - stable, in-place

    An improved version that determines the insertion position via binary search.
    Comparisons are reduced to O(n log n), but
    shift operations remain O(n^2).

    Combining with a linked list would make shifts O(1),
    but binary search requires O(n) access, canceling out the benefit.
    """
    for i in range(1, len(arr)):
        key = arr[i]
        # Find insertion position via binary search using bisect_left
        pos = bisect.bisect_left(arr, key, 0, i)
        # Shift elements from pos to i-1 to the right
        for j in range(i, pos, -1):
            arr[j] = arr[j - 1]
        arr[pos] = key
    return arr


# --- Test ---
if __name__ == "__main__":
    data = [37, 23, 0, 17, 12, 72, 31]
    print(binary_insertion_sort(data))  # [0, 12, 17, 23, 31, 37, 72]
```

### 4.5 The Importance of Insertion Sort

Insertion Sort is the most practical among simple O(n^2) algorithms and is widely used as a building block in modern sorting algorithms for the following reasons.

1. **Optimal for small data**: When n is small, the overhead of O(n log n) algorithms (recursive calls, function call costs) becomes relatively large. Many library implementations switch to Insertion Sort when n < 16-64.
2. **Adaptive**: Operates in O(n) on nearly sorted data.
3. **Stable**: Preserves the relative order of equal keys.
4. **Online**: Can handle data arriving incrementally.
5. **Cache-efficient**: Primarily sequential access to adjacent memory.

---

## 5. Merge Sort

### 5.1 Algorithm Principle

A classic example of divide and conquer. Recursively divide the array in half, sort each half, and then merge the sorted subarrays. Invented by John von Neumann in 1945.

**Three steps:**
1. **Divide**: Split the array in half
2. **Conquer**: Recursively sort each half
3. **Combine**: Merge the two sorted arrays

### 5.2 Visualization

```
Division phase (top-down):
                [38, 27, 43, 3, 9, 82, 10]
                     /                \
            [38, 27, 43]         [3, 9, 82, 10]
              /      \             /         \
          [38]    [27, 43]     [3, 9]     [82, 10]
                   /   \        / \         /   \
                [27]  [43]   [3]  [9]    [82]  [10]

Merge phase (bottom-up):
                [27]  [43]   [3]  [9]    [82]  [10]
                   \   /        \ /         \   /
                [27, 43]     [3, 9]      [10, 82]
              \      /             \         /
          [27, 38, 43]       [3, 9, 10, 82]
                     \                /
              [3, 9, 10, 27, 38, 43, 82]

Merge detail ([27, 38, 43] and [3, 9, 10, 82]):
  L = [27, 38, 43]    R = [3, 9, 10, 82]
       ^                    ^
  Compare: 27 > 3  -> output 3    Result: [3]
  Compare: 27 > 9  -> output 9    Result: [3, 9]
  Compare: 27 > 10 -> output 10   Result: [3, 9, 10]
  Compare: 27 < 82 -> output 27   Result: [3, 9, 10, 27]
  Compare: 38 < 82 -> output 38   Result: [3, 9, 10, 27, 38]
  Compare: 43 < 82 -> output 43   Result: [3, 9, 10, 27, 38, 43]
  L is empty -> output remainder of R  Result: [3, 9, 10, 27, 38, 43, 82]
```

### 5.3 Implementation

```python
def merge_sort(arr: list) -> list:
    """Merge Sort - stable, O(n) additional memory

    Recursively divides and merges the array using divide and conquer.
    Combines stability with guaranteed O(n log n).

    Args:
        arr: List to sort

    Returns:
        New sorted list

    Complexity:
        Best/Average/Worst: O(n log n)
        Space: O(n) -- temporary array during merge
    """
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return _merge(left, right)


def _merge(left: list, right: list) -> list:
    """Merge two sorted arrays

    Compares from the front of both arrays and appends the smaller element to the result.
    Using <= for comparison guarantees stability.
    """
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


# --- Test ---
if __name__ == "__main__":
    data = [38, 27, 43, 3, 9, 82, 10]
    print(merge_sort(data))  # [3, 9, 10, 27, 38, 43, 82]
```

### 5.4 Bottom-Up Merge Sort

An iterative version that does not use recursion. No risk of stack overflow and smaller constant overhead.

```python
def merge_sort_bottom_up(arr: list) -> list:
    """Bottom-Up Merge Sort (iterative)

    Without using recursion, starts from subarrays of size 1
    and progressively doubles the merge size.
    Same O(n log n) as the recursive version, but without
    function call overhead.
    """
    n = len(arr)
    if n <= 1:
        return arr[:]

    result = arr[:]
    width = 1

    while width < n:
        for i in range(0, n, 2 * width):
            left = result[i:i + width]
            right = result[i + width:i + 2 * width]
            merged = _merge(left, right)
            result[i:i + len(merged)] = merged
        width *= 2

    return result


# --- Test ---
if __name__ == "__main__":
    data = [5, 2, 4, 7, 1, 3, 2, 6]
    print(merge_sort_bottom_up(data))  # [1, 2, 2, 3, 4, 5, 6, 7]
```

### 5.5 Characteristics of Merge Sort

- **Guaranteed stability**: Using `<=` for comparison preserves the relative order of equal keys
- **Guaranteed O(n log n)**: Always O(n log n) regardless of input data
- **Affinity with external sorting**: Forms the basis for external sorting that reads large-scale data from disk in chunks and merges them
- **Affinity with parallelism**: Divided subarrays can be sorted independently, making it well-suited for parallel processing
- **Drawback**: Requires O(n) additional memory

---

## 6. Quick Sort

### 6.1 Algorithm Principle

Devised by Tony Hoare in 1959. Select a pivot (reference value), partition the array into elements "less than or equal to the pivot" and "greater than the pivot," and recursively sort each partition. It is a form of divide and conquer, but unlike Merge Sort, no "combine" step is needed -- the computation is concentrated in the partition step.

### 6.2 Partition Visualization

```
Lomuto Partition (pivot = last element):

Array: [10, 80, 30, 90, 40, 50, 70]   Pivot = 70
       i
       j

j=0: arr[0]=10 <= 70  -> i++, swap(arr[0],arr[0])
  [10, 80, 30, 90, 40, 50, 70]
        i

j=1: arr[1]=80 > 70   -> do nothing
  [10, 80, 30, 90, 40, 50, 70]
        i

j=2: arr[2]=30 <= 70  -> i++, swap(arr[1],arr[2])
  [10, 30, 80, 90, 40, 50, 70]
            i

j=3: arr[3]=90 > 70   -> do nothing

j=4: arr[4]=40 <= 70  -> i++, swap(arr[2],arr[4])
  [10, 30, 40, 90, 80, 50, 70]
                i

j=5: arr[5]=50 <= 70  -> i++, swap(arr[3],arr[5])
  [10, 30, 40, 50, 80, 90, 70]
                    i

Final: swap(arr[i+1], arr[high]) = swap(arr[4], arr[6])
  [10, 30, 40, 50, 70, 90, 80]
                    ^^
                  Pivot in final position

Left  [10, 30, 40, 50] are all <= 70
Right [90, 80] are all > 70
```

### 6.3 Implementation

```python
def quick_sort(arr: list, low: int = 0, high: int = None) -> list:
    """Quick Sort - unstable, in-place

    Performs partitioning by pivot and recursive sorting.
    On average, the fastest general-purpose sorting algorithm.

    Args:
        arr: List to sort
        low: Start index of sort range
        high: End index of sort range

    Returns:
        Sorted list

    Complexity:
        Best:    O(n log n)
        Average: O(n log n)
        Worst:   O(n^2) -- sorted array + last-element pivot
        Space:   O(log n) -- recursion stack (average)
    """
    if high is None:
        high = len(arr) - 1
    if low < high:
        pivot_idx = _partition(arr, low, high)
        quick_sort(arr, low, pivot_idx - 1)
        quick_sort(arr, pivot_idx + 1, high)
    return arr


def _partition(arr: list, low: int, high: int) -> int:
    """Lomuto partition"""
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


# --- Test ---
if __name__ == "__main__":
    data = [10, 7, 8, 9, 1, 5]
    print(quick_sort(data[:]))  # [1, 5, 7, 8, 9, 10]
```

### 6.4 Pivot Selection Strategies

Pivot selection has a decisive impact on Quick Sort's performance.

```python
import random


def quick_sort_randomized(arr: list, low: int = 0, high: int = None) -> list:
    """Randomized Quick Sort

    By selecting the pivot randomly,
    worst-case scenarios for specific input patterns are probabilistically avoided.
    Expected complexity is O(n log n).
    """
    if high is None:
        high = len(arr) - 1
    if low < high:
        rand_idx = random.randint(low, high)
        arr[rand_idx], arr[high] = arr[high], arr[rand_idx]
        pivot_idx = _partition(arr, low, high)
        quick_sort_randomized(arr, low, pivot_idx - 1)
        quick_sort_randomized(arr, pivot_idx + 1, high)
    return arr


def partition_median_of_three(arr: list, low: int, high: int) -> int:
    """Median-of-Three Partition

    Selects the median of the first, middle, and last elements as the pivot.
    Avoids worst-case behavior on sorted and reverse-sorted data.
    """
    mid = (low + high) // 2
    if arr[low] > arr[mid]:
        arr[low], arr[mid] = arr[mid], arr[low]
    if arr[low] > arr[high]:
        arr[low], arr[high] = arr[high], arr[low]
    if arr[mid] > arr[high]:
        arr[mid], arr[high] = arr[high], arr[mid]
    # Move median (mid) to high-1 as pivot
    arr[mid], arr[high - 1] = arr[high - 1], arr[mid]
    pivot = arr[high - 1]

    i = low
    j = high - 1
    while True:
        i += 1
        while arr[i] < pivot:
            i += 1
        j -= 1
        while arr[j] > pivot:
            j -= 1
        if i >= j:
            break
        arr[i], arr[j] = arr[j], arr[i]
    arr[i], arr[high - 1] = arr[high - 1], arr[i]
    return i


# --- Test ---
if __name__ == "__main__":
    data = [10, 7, 8, 9, 1, 5]
    print(quick_sort_randomized(data[:]))  # [1, 5, 7, 8, 9, 10]
```

### 6.5 Three-Way Partitioning (Dutch National Flag)

When there are many duplicate elements, standard Quick Sort becomes inefficient. By applying Dijkstra's "Dutch National Flag" problem, elements equal to the pivot can be handled collectively.

```python
def quick_sort_three_way(arr: list, low: int = 0, high: int = None) -> list:
    """Three-Way Partition Quick Sort

    Partitions the array into "less than pivot," "equal to pivot,"
    and "greater than pivot." Particularly effective when there are many duplicates.

    Complexity:
        No duplicates: O(n log n)
        Many duplicates: approaches O(n) (equal elements are skipped)
    """
    if high is None:
        high = len(arr) - 1
    if low >= high:
        return arr

    pivot = arr[low]
    lt = low      # arr[low..lt-1]   < pivot
    i = low + 1   # arr[lt..i-1]    == pivot
    gt = high     # arr[gt+1..high]  > pivot

    while i <= gt:
        if arr[i] < pivot:
            arr[lt], arr[i] = arr[i], arr[lt]
            lt += 1
            i += 1
        elif arr[i] > pivot:
            arr[i], arr[gt] = arr[gt], arr[i]
            gt -= 1
        else:
            i += 1

    quick_sort_three_way(arr, low, lt - 1)
    quick_sort_three_way(arr, gt + 1, high)
    return arr


# --- Test ---
if __name__ == "__main__":
    data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    print(quick_sort_three_way(data))  # [1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9]
```

### 6.6 Worst-Case Analysis of Quick Sort

| Input Pattern | Fixed Pivot (last) | Random Pivot | Median-of-Three |
|:---|:---|:---|:---|
| Sorted | O(n^2) | O(n log n) expected | O(n log n) |
| Reverse | O(n^2) | O(n log n) expected | O(n log n) |
| All same values | O(n^2) | O(n^2) | O(n^2) |
| Random | O(n log n) | O(n log n) | O(n log n) |

For the case where all values are identical, three-way partitioning achieves O(n).

---

## 7. Heap Sort

### 7.1 Algorithm Principle

Build a max-heap (a complete binary tree where each parent is greater than or equal to its children), then repeatedly move the root (maximum) to the end and reduce the heap size. Devised by J. W. J. Williams in 1964.

### 7.2 Heap Structure and Array Representation

```
Array representation of a max-heap:

Index:         0   1   2   3   4   5   6
Array:        [90, 70, 80, 30, 50, 40, 20]

Tree representation:
                 90 (i=0)
               /    \
           70 (i=1)   80 (i=2)
           / \         / \
      30(i=3) 50(i=4) 40(i=5) 20(i=6)

Parent-child relationships (0-indexed):
  Parent:      (i - 1) // 2
  Left child:  2 * i + 1
  Right child: 2 * i + 2

Heap Sort procedure:
  1. Build a max-heap from the entire array (bottom-up)
  2. Swap the root (maximum) with the last element
  3. Reduce heap size by 1 and heapify the root
  4. Repeat steps 2-3 until heap size is 1

Sorting process:
  [90, 70, 80, 30, 50, 40, 20]  <- Max-heap
   Swap 90 and 20 -> heapify
  [80, 70, 40, 30, 50, 20 | 90]
   Swap 80 and 20 -> heapify
  [70, 50, 40, 30, 20 | 80, 90]
   ...repeat...
  [20, 30, 40, 50, 70, 80, 90]  <- Sort complete
```

### 7.3 Implementation

```python
def heap_sort(arr: list) -> list:
    """Heap Sort - unstable, in-place, guaranteed O(n log n)

    Builds a max-heap and repeatedly moves the root to the end.
    Guarantees O(n log n) even in the worst case, using O(1) additional memory.

    Args:
        arr: List to sort

    Returns:
        Sorted list

    Complexity:
        Best/Average/Worst: O(n log n)
        Space: O(1)
    """
    n = len(arr)

    # Build max-heap (bottom-up)
    for i in range(n // 2 - 1, -1, -1):
        _heapify(arr, n, i)

    # Move root to end and reduce heap size
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        _heapify(arr, i, 0)

    return arr


def _heapify(arr: list, n: int, i: int) -> None:
    """Max-heapify the subtree rooted at node i"""
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[left] > arr[largest]:
        largest = left
    if right < n and arr[right] > arr[largest]:
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        _heapify(arr, n, largest)


# --- Test ---
if __name__ == "__main__":
    data = [12, 11, 13, 5, 6, 7]
    print(heap_sort(data))  # [5, 6, 7, 11, 12, 13]

    data2 = [4, 10, 3, 5, 1]
    print(heap_sort(data2))  # [1, 3, 4, 5, 10]
```

### 7.4 Complexity of Heap Construction

Heap construction may intuitively seem like O(n log n), but it is actually **O(n)**. This is because in bottom-up construction, nodes closer to the leaves have lower heapify costs.

```
Number of nodes and heapify cost at each height h:

Height 0 (leaves):   n/2 nodes  x O(0) = 0
Height 1:            n/4 nodes  x O(1) = n/4
Height 2:            n/8 nodes  x O(2) = n/4
Height 3:            n/16 nodes x O(3) = 3n/16
  ...
Height log n (root): 1 node     x O(log n) = log n

Total = Sum_{h=0}^{log n} ceil(n / 2^{h+1}) * h
      = n * Sum_{h=0}^{inf} h / 2^{h+1}
      = n * 1
      = O(n)
```

### 7.5 Characteristics of Heap Sort

| Property | Details |
|:---|:---|
| Worst-case guarantee | Always O(n log n) -- no O(n^2) degradation like Quick Sort |
| Memory efficiency | O(1) additional memory -- no O(n) required like Merge Sort |
| Unstable | Relative order of equal keys is not guaranteed |
| Cache efficiency | Poor -- heap parent-child accesses are far apart in memory |
| Practical performance | Large constant factor; often slower than Quick Sort at the same O(n log n) |

---

## 8. Non-Comparison-Based Sorts

### 8.1 Overview

Comparison-based sorts have a theoretical lower bound of O(n log n), but non-comparison-based sorts are not subject to this constraint. By exploiting properties of the data (integers, fixed-length, uniform distribution, etc.), they can achieve O(n) sorting under certain conditions.

### 8.2 Counting Sort

Count the occurrences of each element and use cumulative sums to determine final positions.

```
Input:  [4, 2, 2, 8, 3, 3, 1]
Range:  1..8 (min=1, max=8, range=8)

Step 1: Create count array
  Value:     1  2  3  4  5  6  7  8
  count:   [ 1, 2, 2, 1, 0, 0, 0, 1 ]

Step 2: Compute cumulative sums
  Value:     1  2  3  4  5  6  7  8
  count:   [ 1, 3, 5, 6, 6, 6, 6, 7 ]
  Meaning: There are count[k] elements with value <= k

Step 3: Build output array (traverse from end for stability)
  i=6: arr[6]=1 -> output[0] = 1
  i=5: arr[5]=3 -> output[4] = 3
  i=4: arr[4]=3 -> output[3] = 3
  i=3: arr[3]=8 -> output[6] = 8
  i=2: arr[2]=2 -> output[2] = 2
  i=1: arr[1]=2 -> output[1] = 2
  i=0: arr[0]=4 -> output[5] = 4

Output: [1, 2, 2, 3, 3, 4, 8]
```

```python
def counting_sort(arr: list) -> list:
    """Counting Sort - stable, O(n + k)

    Counts occurrences of elements and uses cumulative sums to determine positions.
    Optimal for integer data with a small range of values k.

    Args:
        arr: List of integers

    Returns:
        New sorted list

    Complexity:
        Time:  O(n + k) -- k is the range of values
        Space: O(n + k)
    """
    if not arr:
        return arr

    max_val = max(arr)
    min_val = min(arr)
    range_val = max_val - min_val + 1

    count = [0] * range_val
    output = [0] * len(arr)

    for num in arr:
        count[num - min_val] += 1

    for i in range(1, range_val):
        count[i] += count[i - 1]

    for i in range(len(arr) - 1, -1, -1):
        output[count[arr[i] - min_val] - 1] = arr[i]
        count[arr[i] - min_val] -= 1

    return output


# --- Test ---
if __name__ == "__main__":
    data = [4, 2, 2, 8, 3, 3, 1]
    print(counting_sort(data))  # [1, 2, 2, 3, 3, 4, 8]
```

### 8.3 Radix Sort

Apply a stable sort (typically Counting Sort) for each digit to sort multi-digit integers. The LSD (Least Significant Digit) approach, which processes from the least significant digit, is the most common.

```
LSD Radix Sort operation (radix=10):

Input: [170, 45, 75, 90, 802, 24, 2, 66]

Sort by ones digit:
  17[0], 9[0]  -> bucket 0
  80[2], [2]   -> bucket 2
  2[4]         -> bucket 4
  4[5], 7[5]   -> bucket 5
  6[6]         -> bucket 6
  Result: [170, 90, 802, 2, 24, 45, 75, 66]

Sort by tens digit:
  8[0]2, [0]2  -> bucket 0
  [2]4         -> bucket 2
  [4]5         -> bucket 4
  [6]6         -> bucket 6
  1[7]0, [7]5  -> bucket 7
  [9]0         -> bucket 9
  Result: [802, 2, 24, 45, 66, 170, 75, 90]

Sort by hundreds digit:
  [0]02, [0]24, [0]45, [0]66, [0]75, [0]90  -> 0
  [1]70                                      -> 1
  [8]02                                      -> 8
  Result: [2, 24, 45, 66, 75, 90, 170, 802]
```

```python
def radix_sort(arr: list) -> list:
    """Radix Sort (LSD) - stable, O(d * (n + k))

    Applies Counting Sort from the least significant digit upward.
    d is the maximum number of digits, k is the radix (typically 10).

    Args:
        arr: List of non-negative integers

    Returns:
        New sorted list

    Complexity:
        Time:  O(d * (n + k))
        Space: O(n + k)
    """
    if not arr:
        return arr

    result = arr[:]
    max_val = max(result)

    exp = 1
    while max_val // exp > 0:
        result = _counting_sort_by_digit(result, exp)
        exp *= 10

    return result


def _counting_sort_by_digit(arr: list, exp: int) -> list:
    """Counting Sort based on a specific digit"""
    n = len(arr)
    output = [0] * n
    count = [0] * 10

    for num in arr:
        digit = (num // exp) % 10
        count[digit] += 1

    for i in range(1, 10):
        count[i] += count[i - 1]

    for i in range(n - 1, -1, -1):
        digit = (arr[i] // exp) % 10
        output[count[digit] - 1] = arr[i]
        count[digit] -= 1

    return output


# --- Test ---
if __name__ == "__main__":
    data = [170, 45, 75, 90, 802, 24, 2, 66]
    print(radix_sort(data))  # [2, 24, 45, 66, 75, 90, 170, 802]
```

### 8.4 Bucket Sort

Effective when input follows a uniform distribution. Divide the value range into equally spaced buckets and sort each bucket individually.

```python
def bucket_sort(arr: list, bucket_count: int = 10) -> list:
    """Bucket Sort - stable (if internal sort is stable)

    Divides the value range into equally spaced buckets
    and sorts each bucket with Insertion Sort.
    Optimal for uniformly distributed floating-point data.

    Args:
        arr: List of numbers
        bucket_count: Number of buckets

    Returns:
        New sorted list

    Complexity:
        Average: O(n + n^2/k + k) -- O(n) when k=n
        Worst:   O(n^2) -- all elements fall into one bucket
        Space:   O(n + k)
    """
    if not arr:
        return arr

    min_val = min(arr)
    max_val = max(arr)

    if min_val == max_val:
        return arr[:]

    range_val = max_val - min_val

    buckets: list[list] = [[] for _ in range(bucket_count)]

    for num in arr:
        idx = int((num - min_val) / range_val * (bucket_count - 1))
        buckets[idx].append(num)

    result = []
    for bucket in buckets:
        insertion_sort(bucket)
        result.extend(bucket)

    return result


# --- Test ---
if __name__ == "__main__":
    data = [0.897, 0.565, 0.656, 0.1234, 0.665, 0.3434]
    print(bucket_sort(data, 5))
    # [0.1234, 0.3434, 0.565, 0.656, 0.665, 0.897]
```

### 8.5 Comparison of Non-Comparison-Based Sorts

| Algorithm | Complexity | Space | Stable | Applicable Conditions |
|:---|:---|:---|:---|:---|
| Counting Sort | O(n + k) | O(n + k) | Yes | Integers, small value range k |
| Radix Sort | O(d(n + k)) | O(n + k) | Yes | Fixed-length keys, d digits |
| Bucket Sort | O(n) average | O(n + k) | Yes* | Uniformly distributed data |

*When a stable sort is used internally

---

## 9. Advanced Sorting Algorithms

### 9.1 Shell Sort

Devised by Donald Shell in 1959. An improved version of Insertion Sort that compares and swaps elements at a distance, enabling large movements of elements efficiently. Insertion Sort is repeated while gradually reducing the gap (interval).

```
Shell Sort operation (gap sequence: 4, 2, 1):

Initial: [35, 33, 42, 10, 14, 19, 27, 44, 26, 31]

Gap 4:
  Group 1: [35, 14, 26]  -> [14, 26, 35]
  Group 2: [33, 19, 31]  -> [19, 31, 33]
  Group 3: [42, 27]      -> [27, 42]
  Group 4: [10, 44]      -> [10, 44]
  Result: [14, 19, 27, 10, 26, 31, 42, 44, 35, 33]

Gap 2:
  Group 1: [14, 27, 26, 42, 35]  -> [14, 26, 27, 35, 42]
  Group 2: [19, 10, 31, 44, 33]  -> [10, 19, 31, 33, 44]
  Result: [14, 10, 26, 19, 27, 31, 35, 33, 42, 44]

Gap 1 (standard Insertion Sort):
  Result: [10, 14, 19, 26, 27, 31, 33, 35, 42, 44]
```

```python
def shell_sort(arr: list) -> list:
    """Shell Sort - unstable, in-place

    Repeats Insertion Sort while reducing the gap.
    Complexity varies depending on the gap sequence chosen.

    Knuth's gap sequence: 1, 4, 13, 40, 121, ...  (3h + 1)
    Complexity with this sequence is O(n^{3/2}).

    Args:
        arr: List to sort

    Returns:
        Sorted list

    Complexity:
        Knuth sequence:    O(n^{3/2})
        Sedgewick sequence: O(n^{4/3})
        Space: O(1)
    """
    n = len(arr)

    # Compute Knuth's gap sequence
    gap = 1
    while gap < n // 3:
        gap = gap * 3 + 1  # 1, 4, 13, 40, 121, ...

    while gap >= 1:
        # Gap-based Insertion Sort
        for i in range(gap, n):
            key = arr[i]
            j = i
            while j >= gap and arr[j - gap] > key:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = key
        gap //= 3

    return arr


# --- Test ---
if __name__ == "__main__":
    data = [35, 33, 42, 10, 14, 19, 27, 44, 26, 31]
    print(shell_sort(data))  # [10, 14, 19, 26, 27, 31, 33, 35, 42, 44]
```

### 9.2 TimSort

A hybrid sorting algorithm used by Python's `list.sort()` and `sorted()`, as well as Java's `Arrays.sort()`. Designed by Tim Peters in 2002. Combines Merge Sort and Insertion Sort, leveraging patterns in real-world data for acceleration.

```
TimSort strategy:

1. Divide the array into "runs" (already sorted subsequences)
   Input: [1, 3, 5, 2, 4, 6, 8, 7, 9]
         |--run1--|  |---run2---|  |-run3-|
         [1, 3, 5]  [2, 4, 6, 8]  [7, 9]

2. Short runs are extended to at least minrun length using Insertion Sort
   (minrun is typically 32-64, computed from the top 6 bits of n)

3. Runs are pushed onto a stack and merged according to merge rules
   Merge rules (invariants):
     A > B + C  (third is larger than sum of second and first)
     B > C      (second is larger than first)
   When these conditions are violated, adjacent runs are merged

4. Galloping mode: when elements are consecutively selected from one
   array during merging, binary search is used for batch processing
```

**TimSort's design philosophy:**

- **Real-world data is not completely random**: Most real data contains partial order (runs). TimSort detects and exploits this.
- **Insertion Sort is optimal for small subarrays**: Short runs are processed with Insertion Sort. Cache-efficient with small overhead.
- **Stability guaranteed**: Stability is maintained because it is merge-based.
- **O(n log n) in the worst case**: Maintains the worst-case guarantee of Merge Sort.

### 9.3 Introsort (Introspective Sort)

Devised by David Musser in 1997. A hybrid algorithm that uses Quick Sort as its base, switches to Heap Sort when recursion depth exceeds 2 * log(n), and uses Insertion Sort for small subarrays. Used by C++'s `std::sort`.

```
Introsort operation:

1. Start with Quick Sort for partitioning
2. Monitor recursion depth
   - depth < 2*log(n) -> continue Quick Sort
   - depth >= 2*log(n) -> switch to Heap Sort (avoid worst case)
3. Subarray size below threshold -> switch to Insertion Sort

Result:
  - Average: Quick Sort's speed O(n log n)
  - Worst: Heap Sort's guarantee O(n log n)
  - Small: Insertion Sort's low overhead
```

```python
import math


def introsort(arr: list) -> list:
    """Introsort (Introspective Sort)

    Hybrid of Quick Sort + Heap Sort + Insertion Sort.
    The foundational algorithm of C++'s std::sort.

    Complexity:
        Best/Average/Worst: O(n log n)
        Space: O(log n)
    """
    max_depth = 2 * math.floor(math.log2(max(len(arr), 1)))
    _introsort_impl(arr, 0, len(arr) - 1, max_depth)
    return arr


def _introsort_impl(arr: list, low: int, high: int, depth_limit: int) -> None:
    """Internal implementation of Introsort"""
    INSERTION_THRESHOLD = 16

    while high - low + 1 > INSERTION_THRESHOLD:
        if depth_limit == 0:
            # Depth exceeded -> switch to Heap Sort
            sub = arr[low:high + 1]
            heap_sort(sub)
            arr[low:high + 1] = sub
            return

        depth_limit -= 1

        # Median-of-three pivot selection
        mid = (low + high) // 2
        if arr[mid] < arr[low]:
            arr[low], arr[mid] = arr[mid], arr[low]
        if arr[high] < arr[low]:
            arr[low], arr[high] = arr[high], arr[low]
        if arr[high] < arr[mid]:
            arr[mid], arr[high] = arr[high], arr[mid]

        pivot = arr[mid]
        arr[mid], arr[high - 1] = arr[high - 1], arr[mid]

        # Partition
        i = low
        j = high - 1
        while True:
            i += 1
            while arr[i] < pivot:
                i += 1
            j -= 1
            while arr[j] > pivot:
                j -= 1
            if i >= j:
                break
            arr[i], arr[j] = arr[j], arr[i]

        arr[i], arr[high - 1] = arr[high - 1], arr[i]

        # Recurse on the shorter side, loop on the longer side (tail recursion optimization)
        if i - low < high - i:
            _introsort_impl(arr, low, i - 1, depth_limit)
            low = i + 1
        else:
            _introsort_impl(arr, i + 1, high, depth_limit)
            high = i - 1

    # Below threshold: use Insertion Sort
    for i in range(low + 1, high + 1):
        key = arr[i]
        j = i - 1
        while j >= low and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


# --- Test ---
if __name__ == "__main__":
    data = [5, 3, 8, 4, 2, 7, 1, 10, 6, 9]
    print(introsort(data))  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

### 9.4 External Sort

A technique for sorting large-scale data that does not fit in main memory. The key is to minimize I/O between disk and memory.

```
External Merge Sort procedure:

1. Split phase:
   Huge file (e.g., 100GB)
   -> Split into chunks that fit in memory (e.g., 1GB)
   -> Sort each chunk in memory and write to disk
   -> 100 sorted chunk files are produced

2. Merge phase:
   +----------+  +----------+       +----------+
   | Chunk 1  |  | Chunk 2  |  ...  | Chunk 100|
   +----+-----+  +----+-----+       +----+-----+
        |             |                   |
        v             v                   v
   +------------------------------------------+
   |     k-way merge (using min-heap)          |
   |     Store head element of each chunk      |
   |     Extract minimum -> output file        |
   |     Replenish from the extracted chunk    |
   +--------------------+---------------------+
                        |
                        v
                +--------------+
                | Sorted       |
                | output file  |
                +--------------+
```

---

## 10. Complexity Comparison Table and Use-Case Selection Guide

### 10.1 Full Algorithm Complexity Comparison Table

| Algorithm | Best | Average | Worst | Space | Stable | In-Place | Adaptive |
|:---|:---|:---|:---|:---|:---|:---|:---|
| Bubble Sort | O(n) | O(n^2) | O(n^2) | O(1) | Yes | Yes | Yes |
| Selection Sort | O(n^2) | O(n^2) | O(n^2) | O(1) | No | Yes | No |
| Insertion Sort | O(n) | O(n^2) | O(n^2) | O(1) | Yes | Yes | Yes |
| Shell Sort | O(n log n) | O(n^{3/2}) | O(n^{3/2}) | O(1) | No | Yes | Yes |
| Merge Sort | O(n log n) | O(n log n) | O(n log n) | O(n) | Yes | No | No |
| Quick Sort | O(n log n) | O(n log n) | O(n^2) | O(log n) | No | Yes | No |
| Heap Sort | O(n log n) | O(n log n) | O(n log n) | O(1) | No | Yes | No |
| TimSort | O(n) | O(n log n) | O(n log n) | O(n) | Yes | No | Yes |
| Introsort | O(n log n) | O(n log n) | O(n log n) | O(log n) | No | Yes | No |
| Counting Sort | O(n+k) | O(n+k) | O(n+k) | O(n+k) | Yes | No | -- |
| Radix Sort | O(d(n+k)) | O(d(n+k)) | O(d(n+k)) | O(n+k) | Yes | No | -- |
| Bucket Sort | O(n+k) | O(n+k) | O(n^2) | O(n+k) | Yes* | No | -- |

### 10.2 Use-Case Selection Guide

| Situation | Recommended Algorithm | Reason |
|:---|:---|:---|
| Small data (n < 50) | Insertion Sort | Minimal overhead |
| Nearly sorted | Insertion Sort / TimSort | Adaptive, close to O(n) |
| General-purpose (library) | TimSort | Python/Java standard, stable + fast |
| Memory constrained | Heap Sort | O(1) additional memory, O(n log n) guaranteed |
| Integers, small range | Counting Sort | Fastest at O(n+k) |
| Average performance focus | Quick Sort / Introsort | Small constant factor |
| Stability required | Merge Sort / TimSort | O(n log n) and stable |
| External sort (large data) | External Merge Sort | Strong with sequential access |
| C++ standard library | Introsort | Basis of std::sort |
| Data with many duplicates | Three-way Quick Sort | Efficiently skips duplicate elements |
| Fixed-length strings/numbers | Radix Sort | O(d*n) without comparisons |

---

## 11. Anti-Patterns

### Anti-Pattern 1: Fixed Pivot Selection

```python
# BAD: Always using the last element as pivot
# Degrades to O(n^2) on sorted arrays
def bad_quicksort(arr, low, high):
    pivot = arr[high]  # Worst case on sorted input
    # ...partition processing...


# GOOD: Median-of-three or random selection
import random

def good_quicksort(arr, low, high):
    if low < high:
        # Random pivot selection
        rand_idx = random.randint(low, high)
        arr[rand_idx], arr[high] = arr[high], arr[rand_idx]
        # ...partition processing...
```

**Problem**: Degrades to O(n^2) on sorted, reverse-sorted, and nearly sorted data. In production systems, input data patterns cannot be predicted in advance, making fixed pivots dangerous.

### Anti-Pattern 2: Ignoring Value Range for Counting Sort

```python
# BAD: Using Counting Sort when the value range is huge
data = [1, 1000000000, 2]  # count array would be 10^9 in size!
counting_sort(data)         # Crashes with out-of-memory error

# GOOD: Check value range before choosing the algorithm
def smart_sort(data: list) -> list:
    if not data:
        return data
    range_val = max(data) - min(data) + 1
    if range_val <= len(data) * 10:
        return counting_sort(data)  # Small range -> Counting Sort
    else:
        return sorted(data)  # Large range -> comparison sort
```

**Problem**: Counting Sort's space complexity is O(n + k), and when k (value range) is huge, it consumes massive memory. Using it without checking the value range is dangerous.

### Anti-Pattern 3: Ignoring Stability Requirements

```python
# BAD: Using an unstable sort for records, breaking order
students = [(3, "Alice"), (1, "Bob"), (3, "Charlie")]
# Using Heap Sort does not guarantee the order of Alice and Charlie with the same key

# GOOD: Use a stable sort (Python's sort is TimSort, which is stable)
students.sort(key=lambda x: x[0])
# [(1, "Bob"), (3, "Alice"), (3, "Charlie")]  <- Original order is preserved
```

**Problem**: When performing multi-key sorting (e.g., first by name, then by grade), using an unstable sort breaks the previously established order. Stability is essential for operations equivalent to database ORDER BY.

### Anti-Pattern 4: Unlimited Recursion Depth

```python
# BAD: Quick Sort without recursion depth limit
# Worst case leads to O(n) recursion depth, causing stack overflow
def bad_quicksort(arr, low, high):
    if low < high:
        p = partition(arr, low, high)
        bad_quicksort(arr, low, p - 1)   # Recurse left
        bad_quicksort(arr, p + 1, high)  # Recurse right
        # RecursionError on sorted data with n=10000!

# GOOD: Introsort pattern with depth limit + tail recursion optimization
import math

def good_quicksort(arr, low, high, depth=None):
    if depth is None:
        depth = 2 * math.floor(math.log2(max(len(arr), 1)))
    while low < high:
        if depth == 0:
            # Fall back to Heap Sort
            sub = arr[low:high + 1]
            heap_sort(sub)
            arr[low:high + 1] = sub
            return
        depth -= 1
        p = _partition(arr, low, high)
        # Recurse on the shorter side, loop on the longer side (tail recursion optimization)
        if p - low < high - p:
            good_quicksort(arr, low, p - 1, depth)
            low = p + 1
        else:
            good_quicksort(arr, p + 1, high, depth)
            high = p - 1
```

---

## 12. Exercises

### Basic Level

**Problem B1: Manual Trace of a Sorting Algorithm**

Manually trace each step of Insertion Sort on the array `[6, 3, 8, 2, 7, 4]`. Write out the state of the array at each step.

<details>
<summary>Solution</summary>

```
Initial: [6, 3, 8, 2, 7, 4]

i=1: key=3, 6>3 -> shift
  [3, 6, 8, 2, 7, 4]

i=2: key=8, 6<8 -> no shift
  [3, 6, 8, 2, 7, 4]

i=3: key=2, 8>2 -> shift, 6>2 -> shift, 3>2 -> shift
  [2, 3, 6, 8, 7, 4]

i=4: key=7, 8>7 -> shift, 6<7 -> stop
  [2, 3, 6, 7, 8, 4]

i=5: key=4, 8>4 -> shift, 7>4 -> shift, 6>4 -> shift, 3<4 -> stop
  [2, 3, 4, 6, 7, 8]
```

</details>

**Problem B2: Identifying Stability**

From the following sorting algorithms, select all that are stable.

(a) Bubble Sort  (b) Selection Sort  (c) Insertion Sort  (d) Merge Sort  (e) Quick Sort  (f) Heap Sort  (g) Counting Sort

<details>
<summary>Solution</summary>

Stable sorts: **(a) Bubble Sort, (c) Insertion Sort, (d) Merge Sort, (g) Counting Sort**

Unstable sorts: (b) Selection Sort, (e) Quick Sort, (f) Heap Sort

Selection Sort is unstable because it swaps elements at distant positions. Quick Sort's partition operation disrupts relative order. Heap Sort's heap reconstruction disrupts relative order.

</details>

**Problem B3: Complexity Comparison**

For random data with n = 1,000,000, estimate the approximate number of computational steps for Bubble Sort O(n^2) and Merge Sort O(n log n).

<details>
<summary>Solution</summary>

```
Bubble Sort: n^2 = (10^6)^2 = 10^12 steps
Merge Sort: n * log2(n) = 10^6 * 20 = 2 * 10^7 steps

Ratio: 10^12 / (2 * 10^7) = 50,000x

Assuming 1 step = 1 nanosecond:
  Bubble Sort: 10^12 ns = 1000 seconds (approx. 16.7 minutes)
  Merge Sort: 2 * 10^7 ns = 0.02 seconds

This difference demonstrates the importance of algorithmic complexity classes.
```

</details>

### Applied Level

**Problem A1: Finding the k-th Smallest Element**

Implement an algorithm that finds the k-th smallest element in an array in average O(n) without sorting (Quickselect algorithm).

<details>
<summary>Solution</summary>

```python
import random


def quickselect(arr: list, k: int) -> int:
    """Quickselect - finds the k-th smallest element in average O(n)

    Uses Quick Sort's partitioning but only recurses into the
    needed side, reducing complexity.

    Args:
        arr: List of numbers
        k: The k-th smallest element to find (1-indexed)

    Returns:
        The k-th smallest element
    """
    if k < 1 or k > len(arr):
        raise ValueError(f"k={k} is out of range for array of size {len(arr)}")

    arr = arr[:]  # Copy to avoid modifying the original
    return _quickselect(arr, 0, len(arr) - 1, k - 1)


def _quickselect(arr: list, low: int, high: int, k: int) -> int:
    if low == high:
        return arr[low]

    # Random pivot
    pivot_idx = random.randint(low, high)
    arr[pivot_idx], arr[high] = arr[high], arr[pivot_idx]

    # Partition
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    pivot_pos = i + 1

    if k == pivot_pos:
        return arr[pivot_pos]
    elif k < pivot_pos:
        return _quickselect(arr, low, pivot_pos - 1, k)
    else:
        return _quickselect(arr, pivot_pos + 1, high, k)


# --- Test ---
if __name__ == "__main__":
    data = [7, 10, 4, 3, 20, 15]
    print(quickselect(data, 3))  # 7 (3rd smallest element)
    print(quickselect(data, 1))  # 3 (minimum)
    print(quickselect(data, 6))  # 20 (maximum)
```

</details>

**Problem A2: Counting Inversions Using Merge Sort**

Implement a function that computes the inversion count (the number of pairs where i < j and arr[i] > arr[j]) of an array in O(n log n) by modifying Merge Sort.

<details>
<summary>Solution</summary>

```python
def count_inversions(arr: list) -> tuple[list, int]:
    """Compute inversion count in O(n log n) using Merge Sort

    Counts the number of times a right-side element is output first during merging.
    When a right-side element is chosen, the remaining count of left-side elements
    contributes to the inversion count.

    Args:
        arr: List of numbers

    Returns:
        Tuple of (sorted list, inversion count)
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
            merged.append(right[j])
            # All remaining left-side elements form inversion pairs
            inversions += len(left) - i
            j += 1

    merged.extend(left[i:])
    merged.extend(right[j:])

    return merged, inversions


# --- Test ---
if __name__ == "__main__":
    data = [2, 4, 1, 3, 5]
    sorted_arr, inv_count = count_inversions(data)
    print(f"Sorted: {sorted_arr}")  # [1, 2, 3, 4, 5]
    print(f"Inversions: {inv_count}")  # 3
    # Inversion pairs: (2,1), (4,1), (4,3)

    data2 = [5, 4, 3, 2, 1]
    _, inv_count2 = count_inversions(data2)
    print(f"Inversions in reverse: {inv_count2}")  # 10 = 5*4/2
```

</details>

### Advanced Level

**Problem C1: Lower Bound Proof for Comparison-Based Sorts**

Prove using the decision tree model that the worst-case complexity of comparison-based sorting is Omega(n log n).

<details>
<summary>Key Points of the Proof</summary>

**Proof outline:**

1. Sorting n elements is equivalent to the problem of determining which of the n! permutations is the input.

2. A comparison-based algorithm can be modeled as a binary decision tree. Each internal node represents a comparison "a_i < a_j?" with two children YES/NO.

3. Each leaf of the decision tree represents output corresponding to a specific permutation. To sort correctly, distinct leaves are needed for all n! permutations.

4. Therefore, the number of leaves L >= n!.

5. The maximum number of leaves in a binary tree of height h is 2^h, so:
   2^h >= L >= n!
   h >= log2(n!)

6. By Stirling's approximation n! >= (n/e)^n:
   h >= log2((n/e)^n) = n * log2(n/e) = n * (log2(n) - log2(e))
   h = Omega(n log n)

7. Since the height of the decision tree corresponds to the worst-case number of comparisons, the worst-case complexity of comparison-based sorting is Omega(n log n).

This lower bound is an information-theoretic argument and is a universal result independent of any specific algorithm.

</details>

---

## 13. FAQ

### Q1: What algorithm do Python's `sort()` and `sorted()` use?

**A:** They use TimSort (a hybrid of Merge Sort + Insertion Sort). It is stable, with best-case O(n), and average/worst-case O(n log n). It detects and exploits patterns in real data (runs: already sorted subsequences). `sort()` is in-place and modifies the original list, while `sorted()` returns a new list. Java's `Arrays.sort()` also uses TimSort for object arrays (it uses Dual-Pivot Quicksort for primitive types).

### Q2: Does a sort faster than O(n log n) exist?

**A:** For comparison-based sorts, O(n log n) is the theoretical lower bound (proven via decision tree height). However, non-comparison-based sorts (Counting Sort, Radix Sort, Bucket Sort) are not subject to this constraint and can achieve O(n) depending on data characteristics. These, however, require preconditions on the input data (being integers, having a limited value range, etc.).

### Q3: Should you implement your own sort in practice?

**A:** Usually NO. Language-standard sorts (Python's TimSort, C++'s Introsort, Java's Dual-Pivot Quicksort, etc.) are highly optimized and are faster and more robust than custom implementations. Custom implementation is justified only in the following limited scenarios:
- External sorting (large-scale data on disk)
- When special key functions or custom comparisons are needed
- Embedded systems with extremely tight memory constraints
- Educational or research purposes

### Q4: When should you choose Quick Sort vs. Merge Sort?

**A:** Quick Sort is on average faster (good cache efficiency, small constant factor) but has the risk of worst-case O(n^2). Merge Sort guarantees O(n log n) and is stable, but requires O(n) additional memory. Practical guidelines:
- **Stability required** -> Merge Sort / TimSort
- **Memory constrained** -> Quick Sort (avoid worst case with Introsort)
- **External sorting** -> Merge Sort (sequential access pattern suits disk I/O)
- **General-purpose choice** -> Use the language's standard sort function

### Q5: Why does C++ use an unstable sort (std::sort) as the default?

**A:** `std::sort` is based on Introsort and operates in-place (O(log n) additional memory). Guaranteeing stability requires a merge-based algorithm with O(n) additional memory. Due to the performance and memory efficiency trade-off, an unstable sort was chosen as the default. When stability is required, use `std::stable_sort` (merge sort-based).

### Q6: How is sorting parallelized?

**A:** The main approaches are as follows:
- **Parallel Merge Sort**: Sort divided subarrays in independent threads and merge. The division phase naturally parallelizes, and parallel merge algorithms exist for the merge phase.
- **Parallel Quick Sort**: Process left and right partitions in separate threads. However, parallelizing the partition itself is difficult.
- **Sorting Networks**: Algorithms like Bitonic Sort and Odd-Even Merge Sort where the comparison order is independent of input. Suitable for parallel sorting on GPUs.
- **Sample Sort**: Take samples from the data to determine a set of pivots, distribute data evenly to processors. Designed for distributed environments.

---


## FAQ

### Q1: What is the most important point when studying this topic?

Building practical experience is the most important thing. Understanding deepens not only through theory but also by actually writing code and verifying behavior.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before proceeding to the next step.

### Q3: How is this applied in practice?

Knowledge of this topic is frequently used in day-to-day development work. It becomes particularly important during code reviews and architecture design.

---

## 14. Summary

### 14.1 Key Points

| Item | Key Point |
|:---|:---|
| O(n^2) sorts | Bubble, Selection, and Insertion are educational; useful for small data or partial sorting |
| O(n log n) sorts | Merge, Quick, and Heap are at the core of practical use |
| Non-comparison sorts | Counting, Radix, and Bucket achieve O(n) under certain conditions |
| Hybrid sorts | TimSort (Merge + Insertion) and Introsort (Quick + Heap + Insertion) are the modern standard |
| Stability | Whether relative order of records is preserved. Merge, Insertion, Counting, and TimSort are stable |
| Practical choice | Language-standard sorts (TimSort / Introsort) are the first choice; custom implementation only for special requirements |
| Pivot strategy | Randomization or median-of-three to avoid worst-case behavior |
| Theoretical lower bound | O(n log n) is the lower bound for comparison-based sorting (proven by decision tree model) |

### 14.2 Algorithm Selection Flowchart

```
Sorting algorithm selection flow:

                        Nature of data?
                       /              \
                Integers (small range)  General
                     |                  |
               Counting/Radix Sort  Data size?
                                  /     |      \
                              Small(n<50) Med    Large(external)
                                |       |        |
                          Insertion Sort |    External Merge Sort
                                       |
                                Is stability needed?
                               /            \
                             Yes             No
                              |               |
                         TimSort /      Memory constraint?
                       Merge Sort      /          \
                                    Tight        Plenty
                                      |            |
                                 Heap Sort     Introsort /
                                              Quick Sort
```

### 14.3 Learning Roadmap

```
Level 1 (Beginner):
  Insertion Sort -> Bubble Sort -> Selection Sort
  |
Level 2 (Intermediate):
  Merge Sort -> Quick Sort -> Heap Sort
  |
Level 3 (Advanced):
  Counting/Radix/Bucket Sort -> Shell Sort
  |
Level 4 (Expert):
  TimSort -> Introsort -> External Sort -> Parallel Sort
  |
Level 5 (Theory):
  Lower bound proofs -> Information-theoretic optimality -> Adaptive sorting theory
```

---

## Recommended Next Guides

- [Search Algorithms](./01-searching.md) -- Efficient searching on sorted data
- [Divide and Conquer](./06-divide-conquer.md) -- The design paradigm behind Merge Sort and Quick Sort
- [Dynamic Programming](./04-dynamic-programming.md) -- Another paradigm leveraging optimal substructure

---

## 15. References

1. Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. -- Chapter 2 (Insertion Sort), Chapter 6 (Heap Sort), Chapter 7 (Quick Sort), Chapter 8 (Linear-Time Sorting)
2. Sedgewick, R. & Wayne, K. (2011). *Algorithms* (4th ed.). Addison-Wesley. -- Part 2: Sorting (Merge Sort, Quick Sort, Priority Queues, Applications)
3. Knuth, D. E. (1998). *The Art of Computer Programming, Volume 3: Sorting and Searching* (2nd ed.). Addison-Wesley. -- Comprehensive analysis and history of sorting algorithms
4. Python Documentation. "Sorting HOW TO." https://docs.python.org/3/howto/sorting.html -- Official guide to Python's sorting features
5. McIlroy, P. (1993). "Optimistic Sorting and Information Theoretic Complexity." *Proceedings of the Fourth Annual ACM-SIAM Symposium on Discrete Algorithms (SODA)*. -- Theoretical foundation of TimSort
6. Musser, D. R. (1997). "Introspective Sorting and Selection Algorithms." *Software: Practice and Experience*, 27(8), 983-993. -- Original paper on Introsort
7. Peters, T. (2002). "[Python-Dev] Sorting." https://mail.python.org/pipermail/python-dev/2002-July/026837.html -- TimSort design document

---

---

## References

- [MDN Web Docs](https://developer.mozilla.org/) - Web technology reference
- [Wikipedia](https://en.wikipedia.org/) - Overview of technical concepts
