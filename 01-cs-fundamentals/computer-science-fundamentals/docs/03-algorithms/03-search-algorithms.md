# Search Algorithms — A Comprehensive Guide to Theory, Implementation, and Applications

> "Finding data" is the most fundamental operation in computing, and the efficiency of search directly determines the overall performance of a system.
> Choosing the right search algorithm can improve response times by orders of magnitude, directly impacting both user experience and infrastructure costs.

---

## What You Will Learn in This Chapter

- [ ] Understand the essence of linear search and the conditions under which it is the optimal solution
- [ ] Implement the basic form, boundary search, and application patterns of binary search with ease
- [ ] Understand the internal structure and collision resolution of hash-based search and use them appropriately
- [ ] Understand the principles of graph search (BFS / DFS) and solve typical application problems
- [ ] Understand the theoretical background of A* search and apply it to pathfinding problems
- [ ] Compare the performance characteristics of each search algorithm and make optimal choices for different scenarios

## Prerequisites

- Basic data structures (arrays, linked lists, stacks, queues) -> Reference: Chapter 02-data-structures
- Basic Python syntax (lists, dictionaries, class definitions)

---

## Part 1: Sequential Search

---

## 1. Linear Search

### 1.1 Essence of the Algorithm

Linear search is the most intuitive search method, examining elements one by one from the beginning to the end of a data structure. It requires no preconditions and has the versatility to be applicable to any data structure.

The computational complexity is as follows:

| Case | Time Complexity | Description |
|------|----------------|-------------|
| Best | O(1) | Target is the first element |
| Average | O(n) | Average n/2 comparisons |
| Worst | O(n) | Target is the last element or absent |
| Space | O(1) | No additional memory required |

### 1.2 Basic Implementation

```python
def linear_search(arr: list, target) -> int:
    """
    Linear search: scans elements sequentially from the beginning.

    Args:
        arr: List to search (sorting not required)
        target: Value to search for

    Returns:
        Index if found, -1 if not found

    Complexity: O(n) time, O(1) space
    """
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1


# --- Usage example ---
data = [4, 2, 7, 1, 9, 3, 8, 5]

print(linear_search(data, 9))   # => 4  (exists at index 4)
print(linear_search(data, 6))   # => -1 (does not exist)
```

### 1.3 Variations

#### 1.3.1 Sentinel Linear Search

A technique that eliminates the boundary check `i < len(arr)` performed on every loop iteration, halving the number of comparisons.

```python
def sentinel_linear_search(arr: list, target) -> int:
    """
    Linear search using a sentinel.
    Appends target as a sentinel to the end of the array,
    eliminating the need for boundary checks.

    Note: Temporarily modifies the original array, so it is not thread-safe.

    Complexity: O(n) time (constant factor improvement), O(1) space
    """
    n = len(arr)
    if n == 0:
        return -1

    # Save the last element and place the sentinel
    last = arr[n - 1]
    arr[n - 1] = target

    i = 0
    while arr[i] != target:
        i += 1

    # Restore the last element
    arr[n - 1] = last

    # Determine whether the found position was the original last element
    if i < n - 1 or arr[n - 1] == target:
        return i
    return -1


# --- Visualization of operation ---
# arr = [4, 2, 7, 1, 9], target = 7
#
# After sentinel placement: [4, 2, 7, 1, 7]  <- 7 placed at the end
#
# i=0: arr[0]=4 != 7 -> next
# i=1: arr[1]=2 != 7 -> next
# i=2: arr[2]=7 == 7 -> loop ends!
#
# i=2 < n-1=4, so determined to be found in the original array
# Restored: [4, 2, 7, 1, 9]
# return 2
```

#### 1.3.2 Find All

```python
def find_all(arr: list, target) -> list[int]:
    """
    Returns all indices matching target.

    Returns:
        List of matching indices (empty list means not found)

    Complexity: O(n) time, O(k) space (k = number of matches)
    """
    return [i for i, val in enumerate(arr) if val == target]


# --- Usage example ---
data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
print(find_all(data, 5))   # => [4, 8, 10]
print(find_all(data, 7))   # => []
```

#### 1.3.3 Conditional Search

```python
def find_first_match(arr: list, predicate) -> int:
    """
    Returns the index of the first element satisfying the given condition.

    Args:
        predicate: Function that takes an element and returns bool

    Complexity: O(n) time, O(1) space
    """
    for i, val in enumerate(arr):
        if predicate(val):
            return i
    return -1


# --- Usage example ---
students = [
    {"name": "Alice", "score": 72},
    {"name": "Bob", "score": 95},
    {"name": "Charlie", "score": 88},
]
# First student with a score of 90 or above
idx = find_first_match(students, lambda s: s["score"] >= 90)
print(students[idx]["name"])  # => "Bob"
```

### 1.4 Scenarios Where Linear Search Is Optimal

Linear search is not "inferior because it is simple." It is the optimal solution in the following scenarios:

1. **Small number of elements (approximately n < 50)**: The overhead of binary search (maintaining sort order, function calls, branch prediction misses) becomes relatively large, and linear search may be faster
2. **One-time search**: Since sorting costs O(n log n), a single search at O(n) has a lower total cost
3. **Frequently modified data**: The cost of maintaining sorted order on every insertion/deletion exceeds the search cost
4. **Non-random-access data structures such as linked lists**: Binary search requires O(1) random access, so it cannot be used on linked lists

### 1.5 ASCII Diagram: Linear Search in Action

```
Linear search: arr = [4, 2, 7, 1, 9, 3, 8, 5], target = 9

Step 1: [4] 2  7  1  9  3  8  5     4 != 9 -> next
         ^
Step 2:  4 [2] 7  1  9  3  8  5     2 != 9 -> next
            ^
Step 3:  4  2 [7] 1  9  3  8  5     7 != 9 -> next
               ^
Step 4:  4  2  7 [1] 9  3  8  5     1 != 9 -> next
                  ^
Step 5:  4  2  7  1 [9] 3  8  5     9 == 9 -> found! return 4
                     ^

Number of comparisons: 5 (worst case is n=8)
```

---

## Part 2: Divide-and-Conquer Search

---

## 2. Binary Search

### 2.1 Essence of the Algorithm

Binary search is an algorithm that achieves logarithmic-time search by halving the search range each time on **sorted** data. It has the remarkable efficiency that even when the data size doubles, only one additional comparison is needed.

| Data size n | Binary search comparisons | Linear search worst-case comparisons |
|-------------|--------------------------|--------------------------------------|
| 100 | 7 | 100 |
| 10,000 | 14 | 10,000 |
| 1,000,000 | 20 | 1,000,000 |
| 1,000,000,000 | 30 | 1,000,000,000 |

As this table shows, even with 1 billion records, the search completes in just 30 comparisons.

### 2.2 Basic Implementation (Iterative Version)

```python
def binary_search(arr: list, target) -> int:
    """
    Binary search: finds target in a sorted array.

    Args:
        arr: Sorted list (ascending order)
        target: Value to search for

    Returns:
        Index if found, -1 if not found

    Complexity: O(log n) time, O(1) space
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = left + (right - left) // 2  # Standard technique to prevent overflow

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
```

### 2.3 ASCII Diagram: Binary Search in Action

```
Binary search: arr = [1, 3, 5, 7, 9, 11, 13, 15, 17], target = 11

Step 1: left=0, right=8
        [1  3  5  7 [9] 11  13  15  17]
                     ^mid=4
        arr[4]=9 < 11 -> left = 5

Step 2: left=5, right=8
         1  3  5  7  9 [11  13 [15] 17]
                                ^mid=6...
        wait: mid = 5 + (8-5)//2 = 6
        arr[6]=13 > 11 -> right = 5

Step 3: left=5, right=5
         1  3  5  7  9 [11] 13  15  17
                         ^mid=5
        arr[5]=11 == 11 -> found! return 5

Number of comparisons: 3 (log2(9) ~ 3.17)

Search range reduction:
  Step 1: |*********| 9 elements
  Step 2: |    **** | 4 elements
  Step 3: |    *    | 1 element
```

### 2.4 Recursive Implementation

```python
def binary_search_recursive(arr: list, target, left: int = 0,
                             right: int = None) -> int:
    """
    Recursive version of binary search.

    Complexity: O(log n) time, O(log n) space (call stack)
    """
    if right is None:
        right = len(arr) - 1

    if left > right:
        return -1

    mid = left + (right - left) // 2

    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)
```

> **Iterative vs Recursive**: The iterative version requires only O(1) space complexity, so it is recommended for production use. The recursive version carries the risk of stack overflow and may hit Python's default recursion limit (1000). However, the recursive version excels in code readability and conveys the essence of the algorithm more clearly.

### 2.5 Common Bugs and Countermeasures (Top 5)

Binary search is notoriously bug-prone -- it is said that "fewer than 10% of programmers can implement it correctly" (Jon Bentley, "Programming Pearls").

```python
# ================================================================
# Bug 1: Overflow in midpoint calculation
# ================================================================
# This is problematic with fixed-length integers in C/C++/Java.
# Python has arbitrary-precision integers so there's no practical
# issue, but use the safe form for portability to other languages.

mid = (left + right) // 2           # Not recommended: left + right may overflow
mid = left + (right - left) // 2    # Recommended: safe calculation

# ================================================================
# Bug 2: Infinite loop
# ================================================================
# When left = 3, right = 4, mid = 3:
#   arr[mid] < target -> left = mid   <- does not advance! Infinite loop
#   Correct: left = mid + 1

# ================================================================
# Bug 3: Off-by-one error (loop condition)
# ================================================================
# while left < right   -> does not include the case left == right in search range
# while left <= right  -> includes the case left == right in search range
# Which one to use depends on the update expressions for correctness

# ================================================================
# Bug 4: Initial value of right
# ================================================================
# right = len(arr)     -> risk of out-of-bounds access
# right = len(arr) - 1 -> use this for exact match search
# right = len(arr)     -> use this for lower_bound / upper_bound (half-open interval)

# ================================================================
# Bug 5: Incorrect handling of equality
# ================================================================
# arr[mid] < target  vs  arr[mid] <= target
# The only difference between lower_bound and upper_bound is here!
# Getting it wrong produces different results when duplicate elements exist
```

### 2.6 Two Key Templates for Binary Search

To implement binary search correctly in practice, it is effective to master two templates: the "exact match template" and the "boundary search template."

```python
# ================================================================
# Template A: Exact Match Search
# ================================================================
# Purpose: Return the index of an element equal to target
# Loop condition: while left <= right
# Update: left = mid + 1, right = mid - 1
# Initial values: left = 0, right = len(arr) - 1
# Termination: Return mid if target is found. Return -1 if not found

def search_exact(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# ================================================================
# Template B: Boundary Search (lower_bound / upper_bound)
# ================================================================
# Purpose: Return the smallest (or largest) index satisfying a condition
# Loop condition: while left < right
# Update: left = mid + 1 or right = mid (depends on direction)
# Initial values: left = 0, right = len(arr) (half-open interval [left, right))
# Termination: The position where left == right is the answer
```

### 2.7 Boundary Search Implementation

```python
import bisect


def lower_bound(arr: list, target) -> int:
    """
    Returns the smallest index of an element >= target (leftmost insertion point).

    Returns len(arr) if all elements are less than target.
    Equivalent to bisect.bisect_left in the Python standard library.

    Example: arr = [1, 3, 3, 3, 5, 7], target = 3
        -> return 1 (index of the first 3)

    Complexity: O(log n) time, O(1) space
    """
    left, right = 0, len(arr)
    while left < right:
        mid = left + (right - left) // 2
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left


def upper_bound(arr: list, target) -> int:
    """
    Returns the smallest index of an element > target (rightmost insertion point).

    Returns len(arr) if all elements are <= target.
    Equivalent to bisect.bisect_right in the Python standard library.

    Example: arr = [1, 3, 3, 3, 5, 7], target = 3
        -> return 4 (index after the last 3)

    Complexity: O(log n) time, O(1) space
    """
    left, right = 0, len(arr)
    while left < right:
        mid = left + (right - left) // 2
        if arr[mid] <= target:    # <- the only difference from lower_bound
            left = mid + 1
        else:
            right = mid
    return left


# --- Relationship between lower_bound and upper_bound ---
# arr = [1, 3, 3, 3, 5, 7]
#
#  idx:  0  1  2  3  4  5
#  val:  1  3  3  3  5  7
#           ^        ^
#           |        |
#     lower_bound=1  upper_bound=4
#
# Count of target=3: upper_bound - lower_bound = 4 - 1 = 3

def count_occurrences(arr: list, target) -> int:
    """Count occurrences of target in a sorted array in O(log n)."""
    return upper_bound(arr, target) - lower_bound(arr, target)


# --- Using the bisect module ---
arr = [1, 3, 3, 3, 5, 7]

# lower_bound
print(bisect.bisect_left(arr, 3))   # => 1

# upper_bound
print(bisect.bisect_right(arr, 3))  # => 4

# Count of target
print(bisect.bisect_right(arr, 3) - bisect.bisect_left(arr, 3))  # => 3

# Insert while maintaining sorted order
bisect.insort(arr, 4)
print(arr)  # => [1, 3, 3, 3, 4, 5, 7]
```

### 2.8 Application Patterns for Binary Search

#### 2.8.1 Binary Search on Answer

A powerful technique for problems of the form "What is the minimum (or maximum) value that satisfies a condition?" by binary searching over the candidate space of answers.

The conditions for applicability are:
- The answer exists in a continuous (or discrete but ordered) space
- The predicate "Does the condition hold when the answer is >= x?" is monotonic (switches from Yes to No at some threshold)

```python
def max_rope_length(ropes: list[int], k: int) -> int:
    """
    Cut n ropes to make k or more ropes of equal length.
    What is the maximum length in cm?

    Example: ropes = [802, 743, 457, 539], k = 11
    -> answer = 200 (can obtain a total of 11 ropes of 200cm)

    Approach:
    - The range of the answer (rope length) is [1, max(ropes)]
    - Determine whether cutting at length L yields k or more pieces
    - The predicate is monotonically decreasing (larger L yields fewer pieces)
    - -> Binary search for the maximum L satisfying the condition

    Complexity: O(n * log(max(ropes))) time, O(1) space
    """
    def can_cut(length: int) -> bool:
        """Can we obtain k or more pieces when cutting at this length?"""
        return sum(r // length for r in ropes) >= k

    left, right = 1, max(ropes)
    result = 0

    while left <= right:
        mid = left + (right - left) // 2
        if can_cut(mid):
            result = mid        # Condition satisfied; try larger
            left = mid + 1
        else:
            right = mid - 1     # Condition not satisfied; try smaller

    return result


# --- Worked example ---
ropes = [802, 743, 457, 539]
print(max_rope_length(ropes, 11))  # => 200

# Verification of the predicate:
# L=401: 802//401 + 743//401 + 457//401 + 539//401 = 2+1+1+1 = 5 < 11 -> NG
# L=200: 802//200 + 743//200 + 457//200 + 539//200 = 4+3+2+2 = 11 >= 11 -> OK
# L=201: 802//201 + 743//201 + 457//201 + 539//201 = 3+3+2+2 = 10 < 11 -> NG
# -> answer = 200
```

#### 2.8.2 Binary Search in a Rotated Sorted Array

```python
def search_rotated(arr: list, target) -> int:
    """
    Search for target in a rotated sorted array.

    A rotated sorted array is formed by splitting a sorted array at
    some position and swapping the two halves.
    Example: [0,1,2,4,5,6,7] -> [4,5,6,7,0,1,2] (rotated at index 3)

    Key insight: When split at mid, at least one half is sorted.
    Determine whether target falls within the sorted half.

    Complexity: O(log n) time, O(1) space
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if arr[mid] == target:
            return mid

        # Determine if the left half is sorted
        if arr[left] <= arr[mid]:
            # Is target within the left half's range?
            if arr[left] <= target < arr[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            # The right half is sorted
            # Is target within the right half's range?
            if arr[mid] < target <= arr[right]:
                left = mid + 1
            else:
                right = mid - 1

    return -1


# --- Worked example ---
# arr = [4, 5, 6, 7, 0, 1, 2], target = 0
#
# Step 1: left=0, right=6, mid=3
#   arr[0]=4 <= arr[3]=7 -> left half [4,5,6,7] is sorted
#   4 <= 0 < 7 ? No -> target is in the right half -> left = 4
#
# Step 2: left=4, right=6, mid=5
#   arr[4]=0 <= arr[5]=1 -> left half [0,1] is sorted
#   0 <= 0 < 1 ? Yes -> right = 4
#
# Step 3: left=4, right=4, mid=4
#   arr[4]=0 == 0 -> found! return 4
```

#### 2.8.3 Finding a Peak Element

```python
def find_peak_element(arr: list) -> int:
    """
    Returns the index of a peak element (an element greater than both neighbors).

    Assumes arr[-1] = arr[n] = -infinity.
    If multiple peak elements exist, returning any one is acceptable.

    Key insight: If arr[mid] < arr[mid+1], a peak exists to the right.
    (There must be a summit beyond an uphill slope.)

    Complexity: O(log n) time, O(1) space
    """
    left, right = 0, len(arr) - 1

    while left < right:
        mid = left + (right - left) // 2
        if arr[mid] < arr[mid + 1]:
            left = mid + 1     # Peak is to the right
        else:
            right = mid        # Peak is to the left (including mid)

    return left


# --- Worked examples ---
print(find_peak_element([1, 3, 5, 4, 2]))  # => 2 (arr[2]=5 is a peak)
print(find_peak_element([1, 2, 3, 4, 5]))  # => 4 (monotonically increasing, so the last element)
```

#### 2.8.4 Integer Square Root via Binary Search

```python
def integer_sqrt(n: int) -> int:
    """
    Computes the integer square root of a non-negative integer n (floor).

    Binary search for the largest x such that x * x <= n.

    Complexity: O(log n) time, O(1) space
    """
    if n < 2:
        return n

    left, right = 1, n // 2
    result = 1

    while left <= right:
        mid = left + (right - left) // 2
        if mid * mid == n:
            return mid
        elif mid * mid < n:
            result = mid
            left = mid + 1
        else:
            right = mid - 1

    return result


# --- Verification ---
print(integer_sqrt(16))   # => 4
print(integer_sqrt(27))   # => 5 (5*5=25 <= 27 < 36=6*6)
print(integer_sqrt(100))  # => 10
```

### 2.9 Decision Flowchart for Binary Search

```
Should you use binary search? -- Decision flowchart

Check the problem
    |
    +-- Is the data sorted or sortable?
    |       |
    |       +-- Yes -> Exact match search?
    |       |            |
    |       |            +-- Yes -> Use Template A (exact match)
    |       |            |
    |       |            +-- No --> Searching for boundaries or ranges?
    |       |                          |
    |       |                          +-- Yes -> Use Template B (lower/upper_bound)
    |       |                          |
    |       |                          +-- No --> Consider other approaches
    |       |
    |       +-- No --> Is the answer space monotonic?
    |                    |
    |                    +-- Yes -> Use "binary search on answer" pattern
    |                    |
    |                    +-- No --> Binary search is not applicable. Consider other methods
    |
    +-- Is it a special structure? (rotated array, mountain array, etc.)
             |
             +-- Yes -> Use a modified binary search tailored to the structure
             |
             +-- No --> Consider linear search or graph search
```

---

## Part 3: Hash-Based Search

---

## 3. Hash-Based Search

### 3.1 Principles of Hash Tables

A hash table is a data structure that uses a **hash function** to convert keys into array indices, achieving average O(1) search, insertion, and deletion.

#### Requirements for Hash Functions

| Requirement | Description | Consequence of Violation |
|-------------|-------------|--------------------------|
| Deterministic | Always produces the same output for the same input | Search results become unstable |
| Uniform distribution | Output values are spread evenly without bias | Concentration in specific slots degrades performance |
| Fast computation | Low cost to compute the hash value | The O(1) advantage becomes meaningless |

### 3.2 ASCII Diagram: Hash Table Structure

```
Hash table (chaining method)

  Key "apple"  ->  hash("apple") % 8 = 3
  Key "banana" ->  hash("banana") % 8 = 6
  Key "cherry" ->  hash("cherry") % 8 = 3  <- collision!
  Key "date"   ->  hash("date") % 8 = 1

  Bucket array (size 8):
  +-----+-------------------------------------+
  |  0  | (empty)                             |
  +-----+-------------------------------------+
  |  1  | -> ["date", value] -> None          |
  +-----+-------------------------------------+
  |  2  | (empty)                             |
  +-----+-------------------------------------+
  |  3  | -> ["apple", value] -> ["cherry", value] -> None  <- chain
  +-----+-------------------------------------+
  |  4  | (empty)                             |
  +-----+-------------------------------------+
  |  5  | (empty)                             |
  +-----+-------------------------------------+
  |  6  | -> ["banana", value] -> None        |
  +-----+-------------------------------------+
  |  7  | (empty)                             |
  +-----+-------------------------------------+

  Searching for "cherry":
  1. hash("cherry") % 8 = 3
  2. Traverse the chain in bucket 3
  3. "apple" != "cherry" -> next
  4. "cherry" == "cherry" -> found!
```

### 3.3 Comparison of Collision Resolution Methods

No matter how good a hash function is, collisions are unavoidable due to the pigeonhole principle. There are broadly two approaches to collision resolution.

```
Comparison of collision resolution methods

  +------------------+---------------------+---------------------+
  | Property         | Chaining            | Open Addressing     |
  +------------------+---------------------+---------------------+
  | Collision        | Append to linked    | Probe for another   |
  | handling         | list                | slot                |
  | Memory           | Overhead from       | Self-contained      |
  | efficiency       | pointers            | within the table    |
  | Cache            | Low (pointer        | High (contiguous    |
  | efficiency       | chasing)            | memory)             |
  | Load factor      | Can exceed 1.0      | Must be below 1.0   |
  | Ease of          | Easy                | Complex (tombstones |
  | deletion         |                     | required)           |
  | Worst case       | O(n) (all collide)  | O(n) (all collide)  |
  | Used in          | Java HashMap        | Python dict         |
  +------------------+---------------------+---------------------+

  Probing strategies for open addressing:
  - Linear probing: h(k)+1, h(k)+2, h(k)+3, ...
    -> Prone to clustering
  - Quadratic probing: h(k)+1^2, h(k)+2^2, h(k)+3^2, ...
    -> Mitigates primary clustering
  - Double hashing: h(k)+i*h2(k) using a second hash function
    -> Most uniform but higher computation cost
```

### 3.4 Hash Table Implementation

```python
class HashTable:
    """
    Hash table implementation using the chaining method.

    Features:
    - Dynamic resizing (doubles capacity when load factor exceeds 0.75)
    - Supports insertion, search, and deletion
    """

    def __init__(self, initial_capacity: int = 16):
        self._capacity = initial_capacity
        self._size = 0
        self._buckets: list[list] = [[] for _ in range(self._capacity)]
        self._load_factor_threshold = 0.75

    def _hash(self, key) -> int:
        """Converts the hash value of a key to a bucket index."""
        return hash(key) % self._capacity

    def put(self, key, value) -> None:
        """Inserts a key-value pair. Updates the value if the key already exists."""
        if self._size / self._capacity > self._load_factor_threshold:
            self._resize()

        idx = self._hash(key)
        bucket = self._buckets[idx]

        # Check for existing key update
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return

        # New insertion
        bucket.append((key, value))
        self._size += 1

    def get(self, key, default=None):
        """Returns the value for the key. Returns default if not found."""
        idx = self._hash(key)
        for k, v in self._buckets[idx]:
            if k == key:
                return v
        return default

    def remove(self, key) -> bool:
        """Removes a key. Returns True if deletion was successful."""
        idx = self._hash(key)
        bucket = self._buckets[idx]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket.pop(i)
                self._size -= 1
                return True
        return False

    def __contains__(self, key) -> bool:
        """Supports 'key in table'."""
        idx = self._hash(key)
        return any(k == key for k, _ in self._buckets[idx])

    def __len__(self) -> int:
        return self._size

    def _resize(self) -> None:
        """Doubles the number of buckets and rehashes all elements."""
        old_buckets = self._buckets
        self._capacity *= 2
        self._buckets = [[] for _ in range(self._capacity)]
        self._size = 0

        for bucket in old_buckets:
            for key, value in bucket:
                self.put(key, value)


# --- Usage example ---
ht = HashTable()
ht.put("name", "Alice")
ht.put("age", 30)
ht.put("city", "Tokyo")

print(ht.get("name"))     # => "Alice"
print(ht.get("age"))      # => 30
print("city" in ht)        # => True
print(len(ht))             # => 3

ht.remove("age")
print(ht.get("age"))       # => None
print(len(ht))             # => 2
```

### 3.5 Internal Implementation of Python's dict / set

Python's `dict` is a highly optimized hash table using open addressing.

| Property | Python dict Implementation |
|----------|--------------------------|
| Collision resolution | Open addressing (close to random probing) |
| Load factor | Rehash at 2/3 (approximately 66.7%) |
| Hash function | SipHash (Hash DoS mitigation) |
| Memory layout | Compact dict (Python 3.6+, preserves insertion order) |
| Initial size | 8 slots |
| Growth factor | 3x (expands when used slots > 2/3 * capacity) |

```python
# Complexity of Python dict/set

# dict
d = {}
d[key] = value       # Insertion:   average O(1), worst O(n)
value = d[key]        # Lookup:      average O(1), worst O(n)
del d[key]            # Deletion:    average O(1), worst O(n)
key in d              # Membership:  average O(1), worst O(n)

# set
s = set()
s.add(elem)           # Add:         average O(1)
elem in s             # Membership:  average O(1)
s.remove(elem)        # Remove:      average O(1)
s1 & s2               # Intersection: O(min(len(s1), len(s2)))
s1 | s2               # Union:       O(len(s1) + len(s2))

# frozenset is hashable -> can be used as dict keys or set elements
fs = frozenset([1, 2, 3])
d = {fs: "value"}     # OK
```

### 3.6 Binary Search vs Hash Table -- Selection Criteria

```
Search method selection guide

  +------------------------+--------------+------------------+
  | Comparison             | Binary Search| Hash Table       |
  +------------------------+--------------+------------------+
  | Average search time    | O(log n)     | O(1)             |
  | Worst-case search time | O(log n)     | O(n)             |
  | Prerequisite           | Sorted       | Hash function    |
  | Additional memory      | O(1)         | O(n)             |
  | Range queries          | Efficient    | Requires full    |
  |                        |              | scan             |
  | Ordered enumeration    | Easy         | Not possible     |
  | Cache efficiency       | Good         | Inferior         |
  | Worst-case             | High         | Low              |
  | predictability         |              |                  |
  | Implementation         | Moderate     | Low              |
  | complexity             |              |                  |
  | Dynamic data updates   | Requires     | O(1) update      |
  |                        | maintaining  |                  |
  |                        | sort (costly)|                  |
  +------------------------+--------------+------------------+

  Selection guidelines:
  - Exact match only & large data -> Hash table
  - Range queries needed -> Binary search or B-Tree
  - Tight memory constraints -> Binary search (on array, O(1) extra memory)
  - Worst-case guarantees needed -> Binary search
  - Frequently modified data -> Hash table
  - Ordered enumeration needed -> Sorted array + binary search
```

---

## Part 4: Graph Search

---

## 4. Breadth-First Search (BFS)

### 4.1 Essence of the Algorithm

BFS is a search method that explores vertices in order of their distance from the starting point, implemented using a **queue**. It is the fundamental algorithm for solving shortest path problems on unweighted graphs.

Key properties are as follows:

| Property | Value |
|----------|-------|
| Time complexity | O(V + E) (V: vertices, E: edges) |
| Space complexity | O(V) (queue + visited set) |
| Shortest path | Guaranteed on unweighted graphs |
| Completeness | Guaranteed to find a solution on finite graphs if one exists |

### 4.2 Basic Implementation

```python
from collections import deque


def bfs(graph: dict[str, list[str]], start: str) -> list[str]:
    """
    Breadth-first search: visits all vertices in order of distance from start.

    Args:
        graph: Graph in adjacency list representation
        start: Starting node

    Returns:
        List of nodes in visitation order

    Complexity: O(V + E) time, O(V) space
    """
    visited = set([start])
    queue = deque([start])
    order = []

    while queue:
        node = queue.popleft()       # Dequeue from the front
        order.append(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return order


# --- Usage example ---
graph = {
    "A": ["B", "C"],
    "B": ["A", "D", "E"],
    "C": ["A", "F"],
    "D": ["B"],
    "E": ["B", "F"],
    "F": ["C", "E"],
}

print(bfs(graph, "A"))
# => ['A', 'B', 'C', 'D', 'E', 'F']
```

### 4.3 ASCII Diagram: BFS in Action

```
Graph structure:
        A
       / \
      B   C
     / \   \
    D   E - F

BFS visitation order (starting from A):

  Level 0:  [A]
             | A's neighbors: B, C
  Level 1:  [B, C]
             | B's neighbors: D, E  /  C's neighbors: F
  Level 2:  [D, E, F]

  Queue transitions:
  Initial:   Queue=[A],     Visited={A}
  Step 1:    Queue=[B,C],   Visited={A,B,C}       <- dequeue A
  Step 2:    Queue=[C,D,E], Visited={A,B,C,D,E}   <- dequeue B
  Step 3:    Queue=[D,E,F], Visited={A,B,C,D,E,F} <- dequeue C
  Step 4:    Queue=[E,F],   Visited={A,B,C,D,E,F} <- dequeue D
  Step 5:    Queue=[F],     Visited={A,B,C,D,E,F} <- dequeue E
  Step 6:    Queue=[],      Visited={A,B,C,D,E,F} <- dequeue F

  Visitation order: A -> B -> C -> D -> E -> F
```

### 4.4 BFS Shortest Path Recovery

```python
from collections import deque


def bfs_shortest_path(graph: dict, start: str,
                      goal: str) -> list[str] | None:
    """
    Finds the shortest path in an unweighted graph using BFS.

    Returns:
        List of nodes on the shortest path. None if unreachable.

    Complexity: O(V + E) time, O(V) space
    """
    if start == goal:
        return [start]

    visited = set([start])
    queue = deque([(start, [start])])   # (current node, path)

    while queue:
        node, path = queue.popleft()

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                new_path = path + [neighbor]

                if neighbor == goal:
                    return new_path

                visited.add(neighbor)
                queue.append((neighbor, new_path))

    return None  # Unreachable


# --- Usage example ---
graph = {
    "A": ["B", "C"],
    "B": ["A", "D", "E"],
    "C": ["A", "F"],
    "D": ["B"],
    "E": ["B", "F"],
    "F": ["C", "E"],
}

print(bfs_shortest_path(graph, "A", "F"))  # => ['A', 'C', 'F']
print(bfs_shortest_path(graph, "D", "F"))  # => ['D', 'B', 'E', 'F']
```

### 4.5 Typical Applications of BFS

| Application | Description |
|-------------|-------------|
| Shortest path (unweighted) | Shortest path on graphs where all edge weights are equal |
| Level-order traversal | Traversal of tree structures level by level (by depth) |
| Connected component detection | Enumerating connected components in a graph |
| Bipartite graph detection | Determining whether a graph can be 2-colored |
| Shortest solution for mazes | Shortest path problems on grids |
| Social graphs | Computing "friend of friend" distances |

```python
from collections import deque


def shortest_path_in_maze(maze: list[list[int]],
                          start: tuple[int, int],
                          goal: tuple[int, int]) -> int:
    """
    Finds the shortest path length in a 2D grid maze using BFS.

    maze[r][c] = 0: passable, 1: wall
    Movement is allowed in 4 directions (up, down, left, right).

    Returns:
        Shortest number of steps. -1 if unreachable.
    """
    rows, cols = len(maze), len(maze[0])
    if maze[start[0]][start[1]] == 1 or maze[goal[0]][goal[1]] == 1:
        return -1

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    visited = set([start])
    queue = deque([(start[0], start[1], 0)])  # (row, column, steps)

    while queue:
        r, c, steps = queue.popleft()

        if (r, c) == goal:
            return steps

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols
                    and maze[nr][nc] == 0
                    and (nr, nc) not in visited):
                visited.add((nr, nc))
                queue.append((nr, nc, steps + 1))

    return -1


# --- Usage example ---
maze = [
    [0, 0, 0, 0, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
]
print(shortest_path_in_maze(maze, (0, 0), (4, 4)))  # => 8
```

---

## 5. Depth-First Search (DFS)

### 5.1 Essence of the Algorithm

DFS is a search method that explores as deep as possible in one direction, then backtracks to explore other directions when a dead end is reached. It is implemented using a **stack** (or recursive calls).

| Property | Value |
|----------|-------|
| Time complexity | O(V + E) |
| Space complexity | O(V) (recursive: call stack, iterative: explicit stack) |
| Shortest path | **Not guaranteed** |
| Completeness | Guaranteed on finite graphs (not guaranteed on infinite graphs) |

### 5.2 Recursive and Iterative Implementations

```python
def dfs_recursive(graph: dict[str, list[str]], start: str,
                  visited: set = None) -> list[str]:
    """
    Depth-first search (recursive version).

    Complexity: O(V + E) time, O(V) space (call stack)
    """
    if visited is None:
        visited = set()

    visited.add(start)
    order = [start]

    for neighbor in graph[start]:
        if neighbor not in visited:
            order.extend(dfs_recursive(graph, neighbor, visited))

    return order


def dfs_iterative(graph: dict[str, list[str]], start: str) -> list[str]:
    """
    Depth-first search (iterative version).
    Uses an explicit stack to avoid stack overflow from recursion.

    Complexity: O(V + E) time, O(V) space
    """
    visited = set()
    stack = [start]
    order = []

    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)

        # Push neighbors in reverse order onto the stack
        # (to explore in original order)
        for neighbor in reversed(graph[node]):
            if neighbor not in visited:
                stack.append(neighbor)

    return order


# --- Usage example ---
graph = {
    "A": ["B", "C"],
    "B": ["A", "D", "E"],
    "C": ["A", "F"],
    "D": ["B"],
    "E": ["B", "F"],
    "F": ["C", "E"],
}

print(dfs_recursive(graph, "A"))   # => ['A', 'B', 'D', 'E', 'F', 'C']
print(dfs_iterative(graph, "A"))   # => ['A', 'B', 'D', 'E', 'F', 'C']
```

### 5.3 ASCII Diagram: DFS in Action

```
Graph structure:
        A
       / \
      B   C
     / \   \
    D   E - F

DFS visitation order (starting from A, recursive version):

  Stack transitions (recursive call stack):

  (1) visit(A)  ->  Stack: [A]
  (2) visit(B)  ->  Stack: [A, B]       <- A -> B
  (3) visit(D)  ->  Stack: [A, B, D]    <- B -> D
      D is a leaf -> backtrack
  (4) visit(E)  ->  Stack: [A, B, E]    <- B -> E
  (5) visit(F)  ->  Stack: [A, B, E, F] <- E -> F
      F's neighbor C is unvisited, but...
  (6) visit(C)  ->  Stack: [A, B, E, F, C] <- F -> C
      All of C's neighbors (A, F) are visited -> backtrack

  Visitation order: A -> B -> D -> E -> F -> C

  Comparison with BFS:
  BFS: A -> B -> C -> D -> E -> F (expands breadth-first)
  DFS: A -> B -> D -> E -> F -> C (dives depth-first)
```

### 5.4 Typical Applications of DFS

| Application | Description |
|-------------|-------------|
| Topological sort | Orders nodes of a DAG (directed acyclic graph) by dependency |
| Cycle detection | Determines whether a cycle exists in a graph |
| Connected component detection | Enumerates connected components in an undirected graph |
| Strongly connected components (SCC) | Decomposes a directed graph into SCCs (Tarjan / Kosaraju) |
| Backtracking | Combinatorial search for N-Queens, Sudoku, puzzles, etc. |
| Enumerating all paths | Lists all paths from start to goal |

#### 5.4.1 Topological Sort

```python
def topological_sort(graph: dict[str, list[str]]) -> list[str]:
    """
    Computes the topological sort of a directed acyclic graph (DAG) using DFS.

    For every edge (u, v), u comes before v in the result list.

    Use cases:
    - Task dependency resolution in build systems
    - Dependency resolution in package managers
    - Instruction scheduling in compilers

    Complexity: O(V + E) time, O(V) space
    """
    visited = set()
    result = []

    def dfs(node):
        visited.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor)
        result.append(node)  # Append in post-order

    for node in graph:
        if node not in visited:
            dfs(node)

    return result[::-1]  # Reverse of post-order is the topological order


# --- Usage example: compilation dependencies ---
# A -> B, A -> C, B -> D, C -> D dependency order
dependencies = {
    "main.c":    ["utils.h", "math.h"],
    "utils.h":   ["types.h"],
    "math.h":    ["types.h"],
    "types.h":   [],
}

order = topological_sort(dependencies)
print(order)
# => ['types.h', 'utils.h', 'math.h', 'main.c']
# types.h should be compiled first
```

#### 5.4.2 Cycle Detection

```python
def has_cycle(graph: dict[str, list[str]]) -> bool:
    """
    Determines whether a cycle exists in a directed graph using DFS.

    Three-color marking:
    - WHITE (unvisited): not yet visited
    - GRAY  (in progress): on the current DFS path
    - BLACK (completed): all descendants have been explored

    Revisiting a GRAY node indicates a cycle.

    Complexity: O(V + E) time, O(V) space
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in graph}

    def dfs(node) -> bool:
        color[node] = GRAY
        for neighbor in graph.get(node, []):
            if color.get(neighbor, WHITE) == GRAY:
                return True    # Cycle detected!
            if color.get(neighbor, WHITE) == WHITE:
                if dfs(neighbor):
                    return True
        color[node] = BLACK
        return False

    for node in graph:
        if color[node] == WHITE:
            if dfs(node):
                return True
    return False


# --- Usage examples ---
# Cycle present: A -> B -> C -> A
graph_with_cycle = {"A": ["B"], "B": ["C"], "C": ["A"]}
print(has_cycle(graph_with_cycle))  # => True

# No cycle (DAG)
dag = {"A": ["B", "C"], "B": ["D"], "C": ["D"], "D": []}
print(has_cycle(dag))  # => False
```

### 5.5 BFS vs DFS -- Comparison Table

```
BFS vs DFS comparison

  +----------------------+--------------------+--------------------+
  | Property             | BFS                | DFS                |
  +----------------------+--------------------+--------------------+
  | Data structure       | Queue (FIFO)       | Stack (LIFO)       |
  | Search order         | Level-order        | Depth-first        |
  |                      | (breadth-first)    | (depth-first)      |
  | Shortest path        | Guaranteed         | Not guaranteed     |
  | (unweighted)         |                    |                    |
  | Space complexity     | O(b^d)             | O(b*d)             |
  |                      | b=branching factor,| b=branching factor,|
  |                      | d=depth            | d=depth            |
  | Memory usage         | High (stores       | Low                |
  |                      | entire level)      |                    |
  | Completeness         | Guaranteed on      | Guaranteed on      |
  |                      | finite graphs      | finite graphs      |
  | Optimality           | Optimal for        | Not optimal        |
  |                      | uniform costs      |                    |
  +----------------------+--------------------+--------------------+
  | Well-suited problems | Shortest path      | Topological sort   |
  |                      | Level-order        | Cycle detection    |
  |                      | traversal          | Backtracking       |
  |                      | Bipartite graph    | Strongly connected |
  |                      | detection          | components         |
  |                      | Shortest maze      | Puzzle solving     |
  |                      | solution           |                    |
  |                      | Network distance   |                    |
  +----------------------+--------------------+--------------------+

  Selection guidelines:
  - Shortest path needed -> BFS
  - Memory conservation desired -> DFS
  - Enumerate all solutions -> DFS (backtracking)
  - Dependency resolution -> DFS (topological sort)
  - Reachability only -> Either works (DFS tends to be simpler to implement)
```

---

## Part 5: Heuristic Search

---

## 6. A* Search (A-star Search)

### 6.1 Essence of the Algorithm

A* is an algorithm that combines Dijkstra's algorithm with greedy best-first search to efficiently find the **optimal path**. It uses the evaluation function f(n) = g(n) + h(n) to prioritize the most promising nodes for exploration.

| Symbol | Meaning |
|--------|---------|
| g(n) | Actual cost from the start to the current node n |
| h(n) | **Estimated cost** from node n to the goal (heuristic function) |
| f(n) | g(n) + h(n) = total estimated cost |

#### Conditions for the Heuristic Function

For A* to guarantee an optimal solution, h(n) must be **admissible**. That is, h(n) must never exceed the true cost (h(n) <= actual cost).

Representative heuristic functions are as follows:

| Function | Definition | Movement Direction | Use Case |
|----------|-----------|-------------------|----------|
| Manhattan distance | \|x1-x2\| + \|y1-y2\| | 4-directional (up/down/left/right) | Grid maps |
| Euclidean distance | sqrt((x1-x2)^2 + (y1-y2)^2) | Any direction | Free-movement maps |
| Chebyshev distance | max(\|x1-x2\|, \|y1-y2\|) | 8-directional (including diagonals) | Chessboards |
| Zero function | 0 | - | Degenerates to Dijkstra's algorithm |

### 6.2 Implementation

```python
import heapq
from typing import Callable


def a_star(graph: dict, start, goal,
           h: Callable, get_neighbors: Callable) -> tuple[list, float]:
    """
    A* search algorithm.

    Args:
        graph: Graph data (depends on get_neighbors implementation)
        start: Starting node
        goal: Goal node
        h: Heuristic function h(node, goal) -> float
        get_neighbors: Function returning adjacent nodes and costs
                       get_neighbors(graph, node) -> [(neighbor, cost), ...]

    Returns:
        Tuple of (path list, total cost). ([], float('inf')) if unreachable.

    Complexity: Worst case O(b^d) time and space (b=branching factor, d=depth)
                Significantly reduced with a good heuristic
    """
    # Priority queue: (f-value, node)
    open_set = [(h(start, goal), 0, start)]  # (f, g, node)
    came_from = {}
    g_score = {start: 0}
    closed_set = set()

    while open_set:
        f, g, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct the path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1], g

        if current in closed_set:
            continue
        closed_set.add(current)

        for neighbor, cost in get_neighbors(graph, current):
            if neighbor in closed_set:
                continue

            tentative_g = g + cost

            if tentative_g < g_score.get(neighbor, float('inf')):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                f_score = tentative_g + h(neighbor, goal)
                heapq.heappush(open_set, (f_score, tentative_g, neighbor))

    return [], float('inf')  # Unreachable


# --- Usage example on a grid map ---

def manhattan_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Manhattan distance (heuristic for 4-directional movement)."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def grid_neighbors(grid: list[list[int]],
                   node: tuple[int, int]) -> list[tuple]:
    """Returns adjacent cells (up/down/left/right, non-wall) on a grid."""
    rows, cols = len(grid), len(grid[0])
    r, c = node
    neighbors = []
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
            neighbors.append(((nr, nc), 1))  # Cost 1
    return neighbors


# Grid map (0: passable, 1: wall)
grid = [
    [0, 0, 0, 0, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
]

path, cost = a_star(
    grid, (0, 0), (4, 4),
    h=manhattan_distance,
    get_neighbors=grid_neighbors
)
print(f"Path: {path}")  # => [(0,0), (0,1), (0,2), (1,2), (2,2), ...]
print(f"Cost: {cost}")  # => 8
```

### 6.3 ASCII Diagram: A* Search Process

```
A* search: grid map (S=start, G=goal, #=wall, .=passable)

  Initial map:
    S . . . .
    # # . # .
    . . . . .
    . # # # .
    . . . . G

  Search process (numbers are f-values = g + h):

  Step 1: Expand S
    [S] 1  2  3  4       f(S) = 0 + 8 = 8
     #  #  .  #  .
     .  .  .  .  .
     .  #  #  #  .
     .  .  .  .  G

  Around Step 5:
     S  1  2  .  .       <- explored upward
     #  #  3  #  .
     .  .  4  .  .       <- progressing toward lower-right
     .  #  #  #  .
     .  .  .  .  G

  Final result:
     *  *  *  .  .       * = shortest path
     #  #  *  #  .
     .  .  *  *  *
     .  #  #  #  *
     .  .  .  .  *

  Path length: 8 steps

  A* vs BFS comparison (on this map):
  - BFS: Explores uniformly in all directions -> more nodes expanded
  - A* : Prioritizes the goal direction -> fewer nodes expanded (more efficient)
```

### 6.4 Relationship to Special Cases of A*

```
A* and its special cases

  When h(n) = 0:
    f(n) = g(n) + 0 = g(n)
    -> Degenerates to Dijkstra's algorithm (explores uniformly in all directions)

  When g(n) = 0:
    f(n) = 0 + h(n) = h(n)
    -> Greedy best-first search (does not guarantee optimality)

  When h(n) = true cost:
    -> Expands only nodes on the optimal path (ideal but usually impossible to compute)

  Relationship diagram:
  +-------------------------------------------+
  |                A*                          |
  |    f(n) = g(n) + h(n)                     |
  |                                           |
  |  +--------------+  +------------------+   |
  |  | h(n) = 0     |  | g(n) = 0         |   |
  |  | -> Dijkstra  |  | -> Greedy best-  |   |
  |  |  (optimal,   |  |   first          |   |
  |  |   slow)      |  |  (non-optimal,   |   |
  |  |              |  |   fast)           |   |
  |  +--------------+  +------------------+   |
  |                                           |
  |  h(n) admissible -> guarantees optimal    |
  |  h(n) consistent -> guarantees efficient  |
  |                      search               |
  +-------------------------------------------+
```

### 6.5 Heuristic Function Selection and Impact

```python
import math


# --- Examples of heuristic functions ---

def manhattan(a: tuple, b: tuple) -> float:
    """Manhattan distance: optimal for 4-directional movement."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a: tuple, b: tuple) -> float:
    """Euclidean distance: optimal for free movement."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def chebyshev(a: tuple, b: tuple) -> float:
    """Chebyshev distance: optimal for 8-directional movement."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def zero_heuristic(a: tuple, b: tuple) -> float:
    """Zero heuristic: equivalent to Dijkstra's algorithm."""
    return 0


# --- Relationship between heuristic strength and search efficiency ---
#
# Larger h(n):
# + Fewer nodes explored (more efficient)
# - Higher risk of losing optimality (if h(n) > true cost)
#
# Smaller h(n):
# + Stronger guarantee of optimal solution
# - More nodes explored (less efficient)
#
# Ideal: h(n) = true cost (no search needed if computable)
# Reality: Maximize h(n) within the admissibility constraint
```

---

## Part 6: Search in Practice

---

## 7. Search in Databases

### 7.1 Types and Characteristics of Indexes

```
Comparison of database indexes

  +---------------+---------------+-------------+--------------+
  | Index         | Search        | Range       | Use case     |
  |               | complexity    | queries     |              |
  +---------------+---------------+-------------+--------------+
  | B-Tree        | O(log n)      | Efficient   | General      |
  |               |               |             | purpose      |
  | Hash          | O(1) average  | Not         | Exact match  |
  |               |               | supported   |              |
  | GiST          | O(log n)      | Efficient   | Geometric /  |
  |               |               |             | full-text    |
  | GIN           | O(1)~O(k)    | Partially   | Full-text /  |
  |               |               | supported   | arrays       |
  | BRIN          | O(1)          | Efficient   | Large        |
  |               |               |             | sequential   |
  +---------------+---------------+-------------+--------------+
```

### 7.2 Structure of a B-Tree Index

```
B-Tree index (simplified)

  Goal: Search for email = 'test@example.com'

                    +---------------------+
                    |   [M]               |    <- root node
                    | < M    |    >= M    |
                    +----+----------+-----+
                         |          |
            +------------+          +------------+
            v                                    v
  +-------------------+              +-------------------+
  | [D]  [H]         |              | [R]  [V]         |
  | <D | D-H | >H    |              | <R | R-V | >V    |
  +--+---+----+------+              +--+---+----+------+
     |   |    |                       |   |    |
     v   v    v                       v   v    v
   [leaf][leaf][leaf]              [leaf][leaf][leaf]
   a-c  d-g   h-l                  m-q  r-u   v-z
                                        ^
                                   test@ is here!

  Characteristics:
  - Each node fits within one disk page (4KB - 16KB)
  - Fanout: typically 100 - 500
  - Even 1 million records: height 3-4 (= 3-4 disk I/Os)
  - Leaf nodes are linked -> range queries are efficient

  Without vs with index:
    SELECT * FROM users WHERE email = 'test@example.com';
    - Without index: Full table scan O(n) -> seconds for 1 million rows
    - B-Tree: O(log n) -> milliseconds for 1 million rows
```

### 7.3 Full-Text Search and Inverted Indexes

```
Inverted Index

  Document data:
    Doc1: "The quick brown fox jumps"
    Doc2: "The lazy brown dog sleeps"
    Doc3: "Quick fox runs fast"

  Inverted index:
    +-----------+------------------+
    | Token     | Posting list     |
    +-----------+------------------+
    | brown     | [Doc1:3, Doc2:3] |
    | dog       | [Doc2:4]         |
    | fast      | [Doc3:4]         |
    | fox       | [Doc1:4, Doc3:2] |
    | jumps     | [Doc1:5]         |
    | lazy      | [Doc2:2]         |
    | quick     | [Doc1:2, Doc3:1] |
    | runs      | [Doc3:3]         |
    | sleeps    | [Doc2:5]         |
    | the       | [Doc1:1, Doc2:1] |
    +-----------+------------------+

  Search "quick AND fox":
    quick -> {Doc1, Doc3}
    fox   -> {Doc1, Doc3}
    AND   -> {Doc1, Doc3}  <- intersection

  Search "brown OR dog":
    brown -> {Doc1, Doc2}
    dog   -> {Doc2}
    OR    -> {Doc1, Doc2}  <- union

  Representative technologies:
  - Elasticsearch / OpenSearch: Distributed full-text search engine based on Lucene
  - PostgreSQL: tsvector + GIN index
  - SQLite FTS5: Lightweight full-text search extension
  - Apache Solr: Enterprise search based on Lucene
```

---

## Part 7: Comprehensive Comparison and Selection Guidelines

---

## 8. Comprehensive Comparison of Search Algorithms

```
Comparison of all search algorithms

+------------------+-----------+-----------+----------+--------------------+
| Algorithm        | Avg time  | Worst time| Space    | Prerequisites      |
+------------------+-----------+-----------+----------+--------------------+
| Linear search    | O(n)      | O(n)      | O(1)     | None               |
| Binary search    | O(log n)  | O(log n)  | O(1)     | Sorted             |
| Hash search      | O(1)      | O(n)      | O(n)     | Hash function      |
| BFS              | O(V+E)   | O(V+E)    | O(V)     | Graph structure    |
| DFS              | O(V+E)   | O(V+E)    | O(V)     | Graph structure    |
| A*               | O(b^d)*  | O(b^d)    | O(b^d)   | Heuristic          |
| B-Tree search    | O(log n)  | O(log n)  | O(n)     | Prebuilt tree      |
| Inverted index   | O(1)~O(k)| O(n)      | O(n+m)   | Prebuilt index     |
+------------------+-----------+-----------+----------+--------------------+

  * A*'s average complexity depends on the quality of h(n). Greatly improved
    with a good h(n).
  b = branching factor, d = solution depth, V = vertices, E = edges,
  k = result count, m = total tokens
```

### 8.1 Recommended Algorithms by Problem Type

| Problem Type | Recommended Algorithm | Reason |
|-------------|----------------------|--------|
| Small array search | Linear search | No sorting required, minimal overhead |
| Sorted large array | Binary search | O(log n) with no extra memory |
| Key-value lookup | Hash table | O(1) average search time |
| Shortest path (unweighted) | BFS | Guarantees shortest path |
| Shortest path (weighted) | A* / Dijkstra | Efficiency through heuristics |
| Enumerate all solutions | DFS + backtracking | Memory efficient |
| Dependency resolution | DFS (topological sort) | Ordering of DAGs |
| Database search | B-Tree index | Optimized for disk I/O |
| Text search | Inverted index | Specialized for full-text search |

---

## Part 8: Practical Pattern Collection and Performance Tuning

---

## 8.5 Practical Search Patterns

### 8.5.1 Two-Sum Problem (Typical Application of Hash Search)

Finding two elements in an array whose sum equals target is one of the most frequently asked search problems in interviews and coding assessments.

```python
def two_sum_brute_force(nums: list[int], target: int) -> list[int]:
    """
    Brute force: try all pairs.
    Complexity: O(n^2) time, O(1) space
    """
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []


def two_sum_hash(nums: list[int], target: int) -> list[int]:
    """
    Hash table approach: solve in a single pass.

    Idea: If target - nums[i] already exists in the hash map,
    that element forms the answer pair.

    Complexity: O(n) time, O(n) space
    """
    seen = {}  # value -> index mapping
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []


def two_sum_sorted(nums: list[int], target: int) -> list[int]:
    """
    For sorted arrays: two-pointer technique.

    Complexity: O(n) time, O(1) space
    (Assumes sorted input. Add O(n log n) sort if unsorted.)
    """
    left, right = 0, len(nums) - 1
    while left < right:
        current_sum = nums[left] + nums[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return []


# --- Comparison ---
nums = [2, 7, 11, 15]
target = 9
print(two_sum_brute_force(nums, target))  # => [0, 1]
print(two_sum_hash(nums, target))          # => [0, 1]
print(two_sum_sorted(nums, target))        # => [0, 1] (assumes sorted)
```

### 8.5.2 Duplicate Detection Patterns

```python
def has_duplicate_linear(arr: list) -> bool:
    """
    Duplicate detection via linear search: O(n^2)
    """
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] == arr[j]:
                return True
    return False


def has_duplicate_sort(arr: list) -> bool:
    """
    Sort and compare adjacent elements: O(n log n)
    Modifies the original array, so copy beforehand if needed.
    """
    sorted_arr = sorted(arr)
    for i in range(len(sorted_arr) - 1):
        if sorted_arr[i] == sorted_arr[i + 1]:
            return True
    return False


def has_duplicate_hash(arr: list) -> bool:
    """
    Duplicate detection via hash set: O(n)
    """
    seen = set()
    for val in arr:
        if val in seen:
            return True
        seen.add(val)
    return False


# --- Selection criteria ---
# n < 50 and want to save space -> sort approach
# Large n -> hash approach (fastest)
# Tight memory constraints -> sort approach
```

### 8.5.3 Interval Search Pattern (Event Scheduling)

```python
import bisect


def find_available_slots(events: list[tuple[int, int]],
                         query_start: int,
                         query_end: int) -> bool:
    """
    Determines whether a new event can be added to an existing
    sorted event list.

    events: [(start1, end1), (start2, end2), ...] sorted
    query_start, query_end: time range of the event to add

    Approach: Use binary search to find the insertion position,
    then check for overlap with adjacent events.

    Complexity: O(log n) time
    """
    starts = [e[0] for e in events]

    # Index of the smallest event with start >= query_start
    idx = bisect.bisect_left(starts, query_start)

    # Check overlap with the right neighbor
    if idx < len(events) and events[idx][0] < query_end:
        return False

    # Check overlap with the left neighbor
    if idx > 0 and events[idx - 1][1] > query_start:
        return False

    return True


# --- Usage example ---
events = [(1, 3), (5, 8), (10, 15), (18, 20)]
print(find_available_slots(events, 3, 5))    # => True (available)
print(find_available_slots(events, 4, 6))    # => False (overlaps with [5,8])
print(find_available_slots(events, 15, 18))  # => True (available)
```

### 8.5.4 Multi-Dimensional Search (k-d Tree Concept)

```
k-d Tree: efficient search for multi-dimensional data

  2D example: nearest neighbor search in a set of points on a plane

  Naive search: O(n) -- compute distance to all points
  k-d Tree:     O(log n) average -- recursively partition the space

  Partitioning mechanism:
  Depth 0: split by x-coordinate
  Depth 1: split by y-coordinate
  Depth 2: split by x-coordinate (alternating)

       (7, 2)          <- split at x=7
      /      \
   (5, 4)   (9, 6)     <- split at y=4, y=6
   /   \      \
 (2,3) (4,7) (8,1)

  Nearest neighbor search for "query=(6, 3)":
  1. Root (7,2): x=6 < 7 -> go left
  2. (5,4): y=3 < 4 -> go left
  3. (2,3): distance = sqrt(16+0) = 4.0
  4. Backtrack to check other candidates
  5. (5,4): distance = sqrt(1+1) = 1.41 -> update!
  6. Result: nearest neighbor is (5, 4), distance 1.41

  Applications:
  - Map apps: "nearest restaurant" search
  - Image processing: searching for similar-colored pixels
  - Machine learning: k-NN (k-nearest neighbors)
  - Games: accelerating collision detection
```

### 8.5.5 Bloom Filter (Probabilistic Data Structure for Search)

```python
import hashlib
from typing import Any


class BloomFilter:
    """
    Bloom filter: a probabilistic data structure that quickly determines
    whether an element is "definitely not" in a set.

    Properties:
    - False positive: possible
    - False negative: impossible
    - Space efficiency: much smaller than a hash table

    Use cases:
    - Spell checkers: fast identification of words not in the dictionary
    - Caching: avoiding unnecessary DB accesses for non-existent keys
    - Networking: routing table filtering
    """

    def __init__(self, size: int = 1000, num_hashes: int = 3):
        self._size = size
        self._num_hashes = num_hashes
        self._bit_array = [False] * size

    def _hashes(self, item: Any) -> list[int]:
        """Generates multiple hash values for an item."""
        result = []
        for i in range(self._num_hashes):
            h = hashlib.sha256(f"{item}:{i}".encode()).hexdigest()
            result.append(int(h, 16) % self._size)
        return result

    def add(self, item: Any) -> None:
        """Adds an element."""
        for idx in self._hashes(item):
            self._bit_array[idx] = True

    def might_contain(self, item: Any) -> bool:
        """
        Determines whether an element might be in the set.

        Returns:
            True: Possibly contained (false positive possible)
            False: Definitely not contained (no false negatives)
        """
        return all(self._bit_array[idx] for idx in self._hashes(item))


# --- Usage example: web crawler filtering ---
visited = BloomFilter(size=10000, num_hashes=5)

urls = [
    "https://example.com/page1",
    "https://example.com/page2",
    "https://example.com/page3",
]

for url in urls:
    visited.add(url)

# Check
print(visited.might_contain("https://example.com/page1"))  # => True
print(visited.might_contain("https://example.com/page4"))  # => False (definitely not added)
```

### 8.6 Search Performance Tuning

#### 8.6.1 Cache-Aware Search

```
Memory hierarchy and search performance

  L1 cache: ~1 ns     (approximately 64 KB)
  L2 cache: ~3 ns     (approximately 256 KB)
  L3 cache: ~10 ns    (several MB)
  Main memory: ~100 ns (several GB - tens of GB)
  SSD:      ~100 us    (hundreds of GB - several TB)
  HDD:      ~10 ms     (several TB)

  Impact:
  1. Binary search on arrays is cache-friendly
     -> Sequential memory access leverages L1/L2 caches
  2. Search on linked lists is cache-inefficient
     -> Random memory access causes frequent cache misses
  3. Hash tables are prone to cache misses
     -> Irregular jumps to buckets

  Practical guidelines:
  - Small data (n < tens): Linear search may win due to cache efficiency
  - Sorted array + binary search > Balanced BST (cache efficiency)
  - Keeping the load factor low in hash tables improves cache efficiency
```

#### 8.6.2 Optimizing Search in Python

```python
# --- Python search optimization techniques ---

# 1. Choosing the right 'in' operator
# my_list = [1, 2, 3, ..., 100000]  # List: 'in' is O(n)
# my_set = set(my_list)              # Set: 'in' is O(1)

# Use set for bulk membership tests
# 5 in my_list   # O(n) -- slow
# 5 in my_set    # O(1) -- fast

# 2. dict.get() vs try/except
d = {"key": "value"}

# get() does not raise KeyError (lower search overhead)
result = d.get("missing_key", "default")

# try/except is fast when the key exists but slow when it doesn't
# (exception handling cost is high)

# 3. bisect is implemented in C for speed
import bisect
# bisect.bisect_left() is implemented in C and is
# several times faster than a binary search written in Python

# 4. collections.deque is ideal for BFS
from collections import deque
# deque.popleft() is O(1)
# list.pop(0) is O(n) (shifts all elements on front removal)

# 5. sorted() + bisect vs set
# Range queries needed -> sorted list + bisect
# Membership test only -> set
# Both needed -> SortedList (sortedcontainers package)
```

#### 8.6.3 Preprocessing and Amortized Complexity of Search Algorithms

```
Trade-off between preprocessing cost and search cost

+------------------+--------------+--------------+--------------------+
| Method           | Preprocessing| Per search   | Total cost for     |
|                  | cost         |              | m searches         |
+------------------+--------------+--------------+--------------------+
| Linear search    | O(1)         | O(n)         | O(mn)              |
| Sort + binary    | O(n log n)   | O(log n)     | O(n log n + m log n)|
| search           |              |              |                    |
| Hash table build | O(n)         | O(1) expected| O(n + m)           |
| B-Tree build     | O(n log n)   | O(log n)     | O(n log n + m log n)|
| Inverted index   | O(n * L)     | O(1)~O(k)   | O(nL + mk)         |
+------------------+--------------+--------------+--------------------+

  n: data size, m: number of searches, L: average document length, k: result count

  Break-even points:
  - Sort + binary search: advantageous over linear search when m > n / log(n)
    Example: n=10000 -> advantageous when m > 769 searches
  - Hash table build: may have lower total cost than linear search even for m > 1
    (build O(n) + m * O(1) vs m * O(n))
```

---

## Part 8.5: Historical Background of Search Algorithms

---

## 8.7 Brief History of Search Algorithms

The development of search algorithms overlaps with the history of computer science itself.

| Era | Event | Significance |
|-----|-------|-------------|
| 1946 | John Mauchly presents the concept of binary search | Era of ENIAC, the first programmable computer |
| 1953 | Hans Peter Luhn develops hashing at IBM | Foundation of information retrieval |
| 1956 | IBM 305 RAMAC -- the first disk drive | Birth of random access |
| 1968 | Hart, Nilsson, Raphael publish A* | Theoretical foundation of heuristic search |
| 1970 | Bayer & McCreight publish the B-Tree | Foundation of database search |
| 1972 | Tarjan establishes linear algorithms for DFS | Complexity analysis of graph search |
| 1979 | Comer publishes "The Ubiquitous B-Tree" | Systematization of B-Tree's wide-ranging applications |
| 1997 | Google's PageRank (application of graph search) | Revolution in web search |
| 2004 | Compass, predecessor of Elasticsearch, appears | Era of distributed full-text search |

As this history shows, research on search algorithms has always been closely tied to hardware evolution. Hash tables became practical when random access to main memory became possible, B-Trees became essential with the proliferation of disk storage, and graph search and full-text search grew in importance with the explosive growth of the internet.

---

## Part 9: Anti-Patterns and Caveats

---

## 9. Anti-Patterns

### 9.1 Anti-Pattern 1: "Just Use Linear Search" Syndrome

**Symptom**: Always using linear search without analyzing data volume or access patterns.

```python
# --- Bad pattern ---
# Linear search through 100,000 users on every request
def find_user_by_email_bad(users: list[dict], email: str) -> dict | None:
    """
    Problem: O(n) search runs on every request.
    1000 requests/second * 100,000 items = 100 million comparisons/second
    """
    for user in users:
        if user["email"] == email:
            return user
    return None


# --- Good pattern ---
# Build an index at startup and search in O(1)
def build_user_index(users: list[dict]) -> dict[str, dict]:
    """Build an index once in O(n)."""
    return {user["email"]: user for user in users}

user_index = build_user_index(users)

def find_user_by_email_good(email: str) -> dict | None:
    """Search in O(1)."""
    return user_index.get(email)

# Decision criteria:
# - Search performed only once -> linear search is sufficient
# - Search performed repeatedly -> build index + O(1) search
# - Data changes frequently -> balance update cost
```

### 9.2 Anti-Pattern 2: Choosing an Inappropriate Hash Function

**Symptom**: Using a hash function with a high collision rate without verifying its quality.

```python
# --- Bad pattern ---
class BadHashTable:
    """Example of a bad hash function: all strings may end up in the same bucket."""

    def _hash(self, key: str) -> int:
        # Hash by string length -> all strings of the same length collide
        return len(key) % self._capacity


# --- Another bad pattern ---
class AnotherBadHashTable:
    """Another bad example: hash by first character only."""

    def _hash(self, key: str) -> int:
        # ASCII value of first character -> "apple", "avocado", "apricot" all collide
        return ord(key[0]) % self._capacity


# --- Good pattern ---
class GoodHashTable:
    """Use Python's built-in hash() (SipHash-based)."""

    def _hash(self, key) -> int:
        return hash(key) % self._capacity

# Lessons learned:
# - Do not write custom hash functions (use the language's standard one)
# - Hash functions should use all information from the key
# - Use cryptographic hash functions for security-critical scenarios
```

### 9.3 Anti-Pattern 3: Using BFS for Shortest Path on Weighted Graphs

**Symptom**: Applying BFS to a graph with weighted edges, obtaining incorrect shortest paths.

```python
# --- Bad pattern ---
# BFS guarantees shortest paths only on unweighted graphs.
# For graphs with different edge weights, BFS does not return the optimal solution.
#
# Example: A->B(cost 1), A->C(cost 5), B->C(cost 1)
# BFS: May determine A->C (cost 5) as the shortest
# Correct: A->B->C (cost 2) is the shortest

# --- Good pattern ---
# Use Dijkstra's algorithm or A* for weighted graphs.
import heapq

def dijkstra(graph: dict, start: str) -> dict[str, float]:
    """Dijkstra's algorithm: shortest path on weighted graphs."""
    dist = {start: 0}
    pq = [(0, start)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float('inf')):
            continue
        for v, w in graph.get(u, []):
            nd = d + w
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))

    return dist
```

### 9.4 Anti-Pattern 4: Not Verifying Binary Search Prerequisites

**Symptom**: Applying binary search to an unsorted array and obtaining incorrect results.

```python
# --- Bad pattern ---
data = [3, 1, 4, 1, 5, 9, 2, 6]  # Not sorted!
result = binary_search(data, 5)    # Highly likely to return incorrect results

# --- Good pattern ---
# Method 1: Sort beforehand
data.sort()  # O(n log n)
result = binary_search(data, 5)    # Correct result

# Method 2: Use linear search when sorted order cannot be assumed
result = linear_search(data, 5)

# Method 3: Maintain a sorted list for frequent searches
import bisect
sorted_data = sorted(data)  # O(n log n) only once
bisect.insort(sorted_data, new_element)  # O(n) on insertion (shift occurs)
# For frequent insertions, consider SortedList (sortedcontainers)
```

---

## Part 9: Exercises

---

## 10. Exercises (3 Levels)

### 10.1 Basic Level

#### Exercise B1: Variations of Linear Search

Implement the following functions:

```
1. find_min(arr): Return the index of the minimum value in the list (without using built-in functions)
2. find_last(arr, target): Return the index of the last occurrence of target
3. count_if(arr, predicate): Return the count of elements satisfying the condition
```

**Test cases**:
```python
assert find_min([5, 3, 8, 1, 9, 2]) == 3
assert find_last([1, 3, 5, 3, 7, 3], 3) == 5
assert count_if([1, 2, 3, 4, 5, 6], lambda x: x % 2 == 0) == 3
```

#### Exercise B2: Binary Search Basics

Implement the following functions:

```
1. binary_search(arr, target): Exact match search
2. lower_bound(arr, target): Smallest index with element >= target
3. upper_bound(arr, target): Smallest index with element > target
4. count_in_range(arr, lo, hi): Count of elements satisfying lo <= x <= hi
   (Hint: combine upper_bound and lower_bound)
```

**Test cases**:
```python
arr = [1, 2, 2, 3, 3, 3, 4, 5, 5]
assert binary_search(arr, 3) in [3, 4, 5]  # any index of a 3
assert lower_bound(arr, 3) == 3
assert upper_bound(arr, 3) == 6
assert count_in_range(arr, 2, 4) == 6  # 6 elements: [2,2,3,3,3,4]
```

### 10.2 Applied Level

#### Exercise A1: Binary Search on Answer

**Problem**: Assign M tasks with consecutive numbers to N workers. The time required for each task is given. When assigning tasks with consecutive numbers to each worker, minimize the maximum total time of the most heavily loaded worker.

```
Input: tasks = [7, 2, 5, 10, 8], workers = 2
Output: 18

Explanation: Split into [7, 2, 5] and [10, 8] -> maximum is max(14, 18) = 18
             [7, 2, 5, 10] and [8] -> max(24, 8) = 24 -> not optimal
```

**Hint**: Use "Can the maximum total time be divided to be at most X?" as the predicate function and binary search over X.

#### Exercise A2: Shortest Transformation via BFS

**Problem**: Using a given word list, find the shortest number of steps to transform a start word into an end word, changing one character at a time (Word Ladder).

```
Input: begin = "hit", end = "cog",
      words = ["hot", "dot", "dog", "lot", "log", "cog"]
Output: 5  ("hit" -> "hot" -> "dot" -> "dog" -> "cog")
```

### 10.3 Advanced Level

#### Exercise C1: 15-Puzzle with A*

**Problem**: Implement a program that solves the 4x4 15-puzzle (sliding puzzle) using A*.

```
Initial state:        Goal state:
 1  2  3  4           1  2  3  4
 5  6  _  8           5  6  7  8
 9 10  7 11           9 10 11 12
13 14 15 12          13 14 15  _
```

**Requirements**:
1. Use the sum of Manhattan distances as the heuristic function
2. Output the solution steps (sequence of move directions)
3. Also output the number of expanded nodes to verify the heuristic's effectiveness

#### Exercise C2: Complete Hash Table Implementation

**Problem**: Implement a hash table using open addressing (double hashing).

**Requirements**:
1. Support insertion, search, deletion, and resizing
2. Use tombstones for deletion
3. Double capacity when load factor exceeds 0.5
4. Support iteration (`__iter__`)
5. Implement a debug mode that records collision counts

---

## Part 10: FAQ and References

---

## 11. FAQ (Frequently Asked Questions)

### Q1: Can binary search be used on data other than sorted arrays?

**A**: Yes. "Binary Search on Answer" is an extremely powerful pattern. If a condition has "monotonicity where it switches between satisfied/unsatisfied at a threshold," binary search can efficiently find that threshold. This is a technique that reduces optimization problems to decision problems, and it appears frequently in both competitive programming and production systems. Specifically, it can be applied to problems like "What is the minimum number of trucks to deliver all packages within N days?" or "What is the minimum pizza size to satisfy everyone?"

### Q2: Is the worst-case O(n) of hash tables a problem in practice?

**A**: Typically not. With appropriate hash functions (such as Python's SipHash) and load factor management (automatic rehashing below 2/3), collisions are statistically minimized. However, **Hash DoS attacks** require caution. Attackers can send a large number of inputs that intentionally produce the same hash value, forcing O(n) searches and bringing down a service. Python has mitigated this attack since version 3.3 with hash randomization (PYTHONHASHSEED) and the adoption of SipHash.

### Q3: Which is better, BFS or DFS?

**A**: Choose based on the nature of the problem. Use BFS when shortest paths are needed, DFS when memory conservation is desired, DFS (backtracking) for enumerating all solutions, and DFS (topological sort) for dependency resolution. When only determining reachability, either works, but DFS often results in simpler implementations. Note that BFS is advantageous when the solution is close to the root, while DFS is advantageous when the solution is deep or when pruning is effective.

### Q4: How should A*'s heuristic function be designed?

**A**: Follow these three principles: (1) **It must be admissible**: h(n) must be less than or equal to the true cost. If this is violated, optimal solutions are not guaranteed. (2) **It should be consistent**: For any node n and its neighbor m, h(n) <= cost(n, m) + h(m) must hold. When this is satisfied, A* expands each node at most once. (3) **Return as large a value as possible**: Within the bounds of admissibility, a larger h(n) results in fewer nodes explored and greater efficiency. Manhattan distance is commonly used for 4-directional movement on grid maps, Chebyshev distance for 8-directional movement, and Euclidean distance for free movement.

### Q5: How many database indexes should be created?

**A**: Create the minimum necessary based on query patterns. Indexes speed up reads but slow down writes (INSERT / UPDATE / DELETE). Each index also consumes additional storage. General guidelines: (1) Create indexes on columns frequently used in WHERE clauses, (2) Create indexes on JOIN key columns, (3) Indexes are less effective on columns with low selectivity (cardinality) (e.g., boolean), (4) For composite indexes, the selectivity of the leading column is important, (5) Use EXPLAIN ANALYZE to examine the actual query plan before deciding.

### Q6: What is an effective learning order for search algorithms?

**A**: The following order is recommended: (1) Linear search (confirm the basics) -> (2) Binary search (foundation of divide and conquer) -> (3) Hash search (understanding hash tables) -> (4) BFS/DFS (foundation of graph search) -> (5) A* (heuristic search). Solving exercises at each stage before moving on deepens understanding. In particular, mastering the "binary search on answer" pattern for binary search greatly expands your problem-solving capabilities.

---

## 12. References

### 12.1 Books

1. **Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C.** (2022). *Introduction to Algorithms* (4th ed.). MIT Press. -- Known as CLRS. Comprehensive coverage of the theoretical foundations of search algorithms. Chapters 11 (Hash Tables), 12 (Binary Search Trees), and 22 (BFS/DFS) are particularly relevant.

2. **Knuth, D. E.** (1998). *The Art of Computer Programming, Vol. 3: Sorting and Searching* (2nd ed.). Addison-Wesley. -- The definitive mathematical analysis of search algorithms. The detailed analysis of hashing methods is exceptional.

3. **Sedgewick, R., & Wayne, K.** (2011). *Algorithms* (4th ed.). Addison-Wesley. -- A textbook focused on implementation. Rich in code examples in Java, with visualization tools available online.

4. **Bentley, J.** (2000). *Programming Pearls* (2nd ed.). Addison-Wesley. -- Contains the famous anecdote illustrating how difficult it is to correctly implement binary search. Column 4 is particularly relevant.

5. **Skiena, S. S.** (2020). *The Algorithm Design Manual* (3rd ed.). Springer. -- A practical textbook focused on real-world applications of search algorithms.

### 12.2 Papers

6. **Comer, D.** (1979). "The Ubiquitous B-Tree." *ACM Computing Surveys*, 11(2), 121-137. -- A classic survey paper covering the concepts and applications of B-Trees.

7. **Hart, P. E., Nilsson, N. J., & Raphael, B.** (1968). "A Formal Basis for the Heuristic Determination of Minimum Cost Paths." *IEEE Transactions on Systems Science and Cybernetics*, SSC-4(2), 100-107. -- The original paper on the A* algorithm.

8. **Tarjan, R. E.** (1972). "Depth-First Search and Linear Graph Algorithms." *SIAM Journal on Computing*, 1(2), 146-160. -- The seminal paper establishing the theoretical foundation of DFS.

### 12.3 Online Resources

9. **VisuAlgo** (https://visualgo.net/) -- An interactive visualization tool for various search algorithms. Useful for understanding their operation.

10. **Python bisect Module Official Documentation** -- https://docs.python.org/3/library/bisect.html -- The Python standard library's binary search implementation.

---


## FAQ

### Q1: What is the most important point when learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory, but by actually writing code and verifying how it works.

### Q2: What are common mistakes that beginners make?

Skipping the fundamentals and jumping to applications. We recommend thoroughly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this knowledge applied in practice?

Knowledge of this topic is frequently used in everyday development work. It becomes particularly important during code reviews and architecture design.

---

## 13. Summary

### 13.1 Search Algorithm Quick Reference

| Search Method | Complexity | Prerequisites | Best Use Case |
|--------------|-----------|---------------|---------------|
| Linear search | O(n) | None | Small-scale / unsorted / one-time search |
| Binary search | O(log n) | Sorted | Large-scale / range queries / boundary search |
| Hash search | O(1) expected | Hash function | Exact match / high-frequency search |
| BFS | O(V+E) | Graph structure | Shortest path (unweighted) / level-order traversal |
| DFS | O(V+E) | Graph structure | Topological sort / cycle detection / enumerating all solutions |
| A* | O(b^d)* | Heuristic | Shortest path (weighted) / pathfinding |
| B-Tree | O(log n) | Prebuilt | DB / file systems |
| Inverted index | O(1)~O(k) | Prebuilt | Full-text search |

### 13.2 Key Points

1. **Search is inseparable from data structures**: Choosing the right data structure is a prerequisite for efficient search
2. **Verify prerequisites**: Binary search requires sorted data, hash search requires a hash function, A* requires a heuristic
3. **Choose based on the nature of the problem**: Exact match or range query, whether shortest path is needed, whether data is static or dynamic
4. **Distinguish between average and worst-case complexity**: The O(1) of hashing is an "expected value," and the worst case can be O(n)
5. **Leverage libraries in practice**: Use Python's bisect, dict/set, collections.deque, and similar tools appropriately

---

## Recommended Next Guides


---

---

## References

- [MDN Web Docs](https://developer.mozilla.org/) - Web technology reference
- [Wikipedia](https://en.wikipedia.org/) - Overview of technical concepts
