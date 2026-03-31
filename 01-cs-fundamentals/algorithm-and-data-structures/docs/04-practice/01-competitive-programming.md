# Competitive Programming

> Systematically master the essential techniques, strategies, and practical patterns of competitive programming, with a focus on AtCoder, LeetCode, and Codeforces

## What You Will Learn in This Chapter

1. **AtCoder and LeetCode** level systems and efficient learning roadmaps
2. **Frequently used classic techniques** (coordinate compression, bit brute force, two pointers, MOD arithmetic, etc.) with practical implementations
3. **Contest strategies** (time management, template usage, debugging methods)
4. **Interview preparation applications** (LeetCode pattern classification, solution frameworks for common problems)
5. **Phased growth plans** designed for steady rating improvement


## Prerequisites

Before reading this guide, familiarity with the following will deepen your understanding:

- Basic programming knowledge
- Understanding of fundamental concepts
- The content of the [Problem Solving](./00-problem-solving.md) guide

---

## Table of Contents

1. [Overview of Competitive Programming](#1-overview-of-competitive-programming)
2. [Python Templates and Optimization Techniques](#2-python-templates-and-optimization-techniques)
3. [Classic Technique: Bit Brute Force and Meet in the Middle](#3-classic-technique-bit-brute-force-and-meet-in-the-middle)
4. [Classic Technique: Coordinate Compression and Inversion Count](#4-classic-technique-coordinate-compression-and-inversion-count)
5. [Classic Technique: Two Pointers (Sliding Window)](#5-classic-technique-two-pointers-sliding-window)
6. [Classic Technique: MOD Arithmetic and Combinatorics](#6-classic-technique-mod-arithmetic-and-combinatorics)
7. [Graph Algorithm Implementations for Competitive Programming](#7-graph-algorithm-implementations-for-competitive-programming)
8. [Prefix Sums, imos Method, and Binary Search](#8-prefix-sums-imos-method-and-binary-search)
9. [AtCoder / LeetCode / Codeforces Pattern Analysis](#9-atcoder--leetcode--codeforces-pattern-analysis)
10. [Contest Strategy and Mental Models](#10-contest-strategy-and-mental-models)
11. [Platform Comparison and Learning Roadmap](#11-platform-comparison-and-learning-roadmap)
12. [Exercises (3 Levels)](#12-exercises-3-levels)
13. [Anti-patterns](#13-anti-patterns)
14. [FAQ](#14-faq)
15. [Summary](#15-summary)
16. [References](#16-references)

---

## 1. Overview of Competitive Programming

### 1.1 What Is Competitive Programming?

Competitive programming (often abbreviated as "CP") is an intellectual sport where participants compete to solve algorithmic problems within a time limit. The goal is to write programs that produce correct outputs for given problems while satisfying constraints on execution time and memory usage, solving as many problems as quickly as possible.

The essence of competitive programming lies in the process of "reading a problem, modeling it mathematically and algorithmically, and implementing an efficient solution." This demands not just coding ability but a comprehensive combination of problem analysis skills, algorithm design skills, implementation skills, and debugging skills.

### 1.2 Positioning of Major Platforms

```
+----------------------------------------------------------------------+
|              The World Map of Competitive Programming                 |
+---------------------+-------------------+----------------------------+
|  AtCoder (Japan)    |  LeetCode (USA)   |  Codeforces (Russia)       |
|  Contest-focused    |  Interview-focused|  Contest-focused            |
+---------------------+-------------------+----------------------------+
| ABC: Beginner-Mid   | Easy:   Basic     | Div.4: Beginner            |
| ARC: Mid-Advanced   | Medium: Applied   | Div.3: Beginner-Mid        |
| AGC: Adv-Expert     | Hard:   Advanced  | Div.2: Mid-Advanced        |
| AHC: Heuristic      |                   | Div.1: Advanced-Expert     |
|                     |                   | Global: Expert             |
+---------------------+-------------------+----------------------------+
| Rating:             | Rating:           | Rating:                    |
| Gray  <400  (Entry) | Assigned upon     | Newbie   <1200             |
| Brown 400-799(Beg.) | contest           | Pupil    1200-1399         |
| Green 800-1199(Mid) | participation     | Specialist 1400-1599       |
| Cyan  1200-1599(Adv)|                   | Expert   1600-1899         |
| Blue  1600-1999(S-A)| Primary purposes: | CM       1900-2099         |
| Yel.  2000-2399(Exp)| - FAANG interview | Master   2100-2299         |
| Org.  2400-2799(Ult)|   preparation     | IM       2300-2399         |
| Red   2800+  (Top)  | - Algorithmic     | GM       2400-2599         |
|                     |   skill proof     | IGM      2600-2999         |
|                     | - Skill visibility| LGM      3000+             |
+---------------------+-------------------+----------------------------+
```

### 1.3 AtCoder Contest System

AtCoder is Japan's largest competitive programming platform, known for high-quality problems and thorough editorial explanations.

**ABC (AtCoder Beginner Contest)** is held every Saturday at 21:00 (JST). It consists of 7 problems (A through G) with a 100-minute time limit. Problems A and B test fundamental programming ability, C and D test knowledge of standard algorithms, and E, F, and G require advanced data structures and algorithmic knowledge.

**ARC (AtCoder Regular Contest)** is an intermediate-to-advanced contest held 1-2 times per month, featuring problems that heavily demand mathematical reasoning.

**AGC (AtCoder Grand Contest)** is an irregularly held expert-level contest featuring world-class difficult problems.

**AHC (AtCoder Heuristic Contest)** is a marathon-style (optimization) contest where participants compete on the quality of approximate solutions rather than exact solutions.

### 1.4 Rating and Skill Level Correspondence

```
AtCoder Rating and Approximate Difficulty to Reach:

  Rating   Color    Approx. Time       Required Knowledge Level
  ----------------------------------------------------------------
  < 400    Gray     Immediately        Programming basics
  400-799  Brown    1-3 months         Basic algorithms (sorting, searching)
  800-1199 Green    3-6 months         Classic techniques (DP, BFS, prefix sums)
  1200-1599 Cyan    6 months - 1 year  Segment trees, Union-Find, advanced DP
  1600-1999 Blue    1-2 years          Number theory, flow, advanced graph theory
  2000-2399 Yellow  2-3 years          Mathematical insight, advanced construction
  2400-2799 Orange  3+ years           World-class problem-solving ability
  2800+    Red      Extremely hard     International Olympiad in Informatics medalist level

  * Timeframes assume 1-2 hours of daily practice
  * Varies significantly based on mathematical background and programming experience
```

### 1.5 Competitive Programming and Professional Development

Skills developed through competitive programming overlap with foundational software engineering skills in some areas, but not others.

**Directly Transferable Skills:**
- Designing with computational complexity in mind (the habit of considering O(N log N) when O(N^2) causes TLE)
- Sensitivity to edge cases (empty arrays, single-element inputs, values near the maximum)
- Appropriate selection of data structures (hash maps vs. sorted arrays vs. heaps)
- A broad repertoire of algorithms (recognizing when graph traversal, DP, or binary search is applicable)

**Skills Requiring Separate Development:**
- Maintainable code design (variable naming, function decomposition, testability)
- Team development (code reviews, documentation, communication)
- System design (distributed systems, database design, API design)

---

## 2. Python Templates and Optimization Techniques

### 2.1 Basic Template

In competitive programming, preparing boilerplate patterns for input and output as templates lets you focus on the essential problem-solving. Below is a standard Python template.

```python
import sys
from collections import defaultdict, deque, Counter
from itertools import combinations, permutations, accumulate, product
from heapq import heappush, heappop, heapify
from bisect import bisect_left, bisect_right
from functools import lru_cache
from math import gcd, lcm, isqrt, inf

# Fast input (sys.stdin.readline is about 3x faster than input())
input = sys.stdin.readline

def main():
    # ============================
    # Input patterns
    # ============================

    # Single integer
    N = int(input())

    # Two integers
    # N, M = map(int, input().split())

    # List of integers
    A = list(map(int, input().split()))

    # String
    # S = input().strip()  # strip() removes trailing newline

    # N lines of input (e.g., graph edges)
    # edges = [tuple(map(int, input().split())) for _ in range(M)]

    # Grid input
    # grid = [input().strip() for _ in range(H)]

    # ============================
    # Write solution here
    # ============================
    ans = 0

    # ============================
    # Output
    # ============================
    print(ans)

if __name__ == "__main__":
    main()
```

### 2.2 I/O Optimization

Python often becomes I/O-bound. The following techniques can dramatically improve speed.

```python
import sys

# Method 1: Replace input with sys.stdin.readline
input = sys.stdin.readline

# Method 2: Bulk read all input at once (fastest)
def solve():
    data = sys.stdin.read().split()
    idx = 0
    def rd():
        nonlocal idx
        idx += 1
        return data[idx - 1]

    N = int(rd())
    A = [int(rd()) for _ in range(N)]

    # Solution...
    ans = sum(A)
    print(ans)

solve()

# Method 3: Fast output for large amounts of data
def fast_output(results):
    """Output each element of a list separated by newlines"""
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

# Method 4: Extend recursion limit (for deep recursion in DFS)
sys.setrecursionlimit(10**6)

# Method 5: Use a thread to ensure adequate recursion stack size
import threading
def main():
    sys.setrecursionlimit(10**6)
    # Processing with deep recursion like DFS
    pass

threading.Thread(target=main, daemon=True).start()
```

### 2.3 C++ Template (Reference)

For problems that demand speed, switching to C++ may be necessary. Below is a minimal C++ template.

```cpp
#include <bits/stdc++.h>
using namespace std;

// Type aliases
using ll = long long;
using pii = pair<int, int>;
using vi = vector<int>;
using vll = vector<ll>;

// Constants
const int INF = 1e9;
const ll LINF = 1e18;
const int MOD = 1e9 + 7;

// Macros
#define rep(i, n) for (int i = 0; i < (n); i++)
#define all(v) (v).begin(), (v).end()

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int N;
    cin >> N;
    vi A(N);
    rep(i, N) cin >> A[i];

    ll ans = 0;
    // Write solution here

    cout << ans << endl;
    return 0;
}
```

### 2.4 Python Optimization Techniques Overview

```
+-------------------------------------------------------------+
|          Python Optimization Techniques Overview              |
+-----------------+-------------------------------------------+
| I/O             | sys.stdin.readline, sys.stdin.read()       |
|                 | sys.stdout.write(), suppress print flush   |
+-----------------+-------------------------------------------+
| Loops           | List comprehensions (2-3x faster than for) |
|                 | Use map/filter                             |
|                 | Localize loop variables                    |
+-----------------+-------------------------------------------+
| Data Structures | set/dict lookup O(1)                       |
|                 | deque O(1) operations on both ends         |
|                 | heapq for priority queues                  |
|                 | SortedContainers (use BIT if pip unavail.) |
+-----------------+-------------------------------------------+
| Arithmetic      | Reduce branching with bit operations       |
|                 | Leverage C implementations: math.isqrt,    |
|                 | math.gcd                                   |
|                 | numpy (available on AtCoder)               |
+-----------------+-------------------------------------------+
| Submission Lang | PyPy (3-10x faster than CPython)           |
|                 | Cython (compiled, fast)                    |
+-----------------+-------------------------------------------+
| Memoization     | @lru_cache (essential for recursive DP)    |
|                 | Manual memoization (dict-based)            |
+-----------------+-------------------------------------------+
```

---

## 3. Classic Technique: Bit Brute Force and Meet in the Middle

### 3.1 Bit Brute Force Basics

Bit brute force is a technique for enumerating all subsets of n elements. When n is small (n <= 20 or so), all 2^n combinations are managed using the bit representation of integers. A bit of 1 means the element is selected; 0 means it is not.

```
Elements: [A, B, C, D]   (n = 4)
Bits:     0000 -> {}
          0001 -> {A}
          0010 -> {B}
          0011 -> {A, B}
          0100 -> {C}
          ...
          1111 -> {A, B, C, D}

Total: 2^4 = 16 combinations
```

### 3.2 Bit Brute Force Implementation

```python
def bit_bruteforce(n: int, items: list) -> list:
    """Enumerate all subsets using bit brute force

    Time complexity: O(2^n * n)
    Constraint:      n <= 20 or so (2^20 = approx. 1 million)
    """
    results = []
    for mask in range(1 << n):       # 0 to 2^n - 1
        subset = []
        for i in range(n):
            if mask & (1 << i):      # Check if the i-th bit is set
                subset.append(items[i])
        results.append(subset)
    return results

# Usage example
items = ['A', 'B', 'C']
subsets = bit_bruteforce(3, items)
# [[], ['A'], ['B'], ['A','B'], ['C'], ['A','C'], ['B','C'], ['A','B','C']]
```

### 3.3 Application to the Subset Sum Problem

The subset sum problem is a classic application of bit brute force. It determines whether "some elements selected from an array can sum to a given target."

```python
def subset_sum(arr: list, target: int) -> bool:
    """Solve the subset sum problem using bit brute force

    Time complexity: O(2^n * n)
    Constraint:      n <= 20
    """
    n = len(arr)
    for mask in range(1 << n):
        total = 0
        for i in range(n):
            if mask & (1 << i):
                total += arr[i]
        if total == target:
            return True
    return False

# Tests
print(subset_sum([3, 7, 1, 8, 4], 12))   # True  (3 + 1 + 8 = 12)
print(subset_sum([3, 7, 1, 8, 4], 2))    # False
print(subset_sum([3, 7, 1, 8, 4], 23))   # True  (3 + 7 + 1 + 8 + 4 = 23)

# More efficient implementation (skip masks where popcount doesn't match)
def subset_sum_optimized(arr: list, target: int) -> list:
    """Return all subsets that sum to the target"""
    n = len(arr)
    results = []
    for mask in range(1 << n):
        total = sum(arr[i] for i in range(n) if mask & (1 << i))
        if total == target:
            subset = [arr[i] for i in range(n) if mask & (1 << i)]
            results.append(subset)
    return results

print(subset_sum_optimized([3, 7, 1, 8, 4], 12))
# [[3, 1, 8], [8, 4]]
```

### 3.4 Meet in the Middle

When n extends to around 40, split the array into two halves, enumerate all 2^(n/2) subsets for each half, and match them.

```
Full brute force: O(2^40) = approx. 1 trillion -> TLE
Meet in the Middle: O(2^20 * 20) = approx. 20 million -> feasible

+--------------------------------------------+
|  Split array A into left half L and        |
|  right half R                              |
|                                            |
|  Enumerate all subset sums of L:           |
|    {s_1, s_2, ..., s_k}                   |
|                   |                        |
|  For each subset sum t of R,               |
|  check if target - t exists in L's set     |
|                                            |
|  Time complexity: O(2^(n/2) * n)           |
+--------------------------------------------+
```

```python
def meet_in_the_middle(arr: list, target: int) -> bool:
    """Solve the subset sum problem using Meet in the Middle

    Time complexity: O(2^(n/2) * n)
    Constraint:      n <= 40
    """
    n = len(arr)
    half = n // 2

    # Store all subset sums of the left half in a set
    left_sums = set()
    for mask in range(1 << half):
        s = sum(arr[i] for i in range(half) if mask & (1 << i))
        left_sums.add(s)

    # For each subset sum of the right half, check if the complement exists in the left
    for mask in range(1 << (n - half)):
        s = sum(arr[half + i] for i in range(n - half) if mask & (1 << i))
        if target - s in left_sums:
            return True

    return False

# Test: Subset sum with n = 30 can be solved efficiently
import random
arr = [random.randint(1, 10**9) for _ in range(30)]
target = sum(arr[:5])  # Use the sum of the first 5 elements as the target
print(meet_in_the_middle(arr, target))  # True
```

### 3.5 Bit Manipulation Tips

```python
# Essential bit manipulation techniques (frequently used in competitive programming)

# 1. Check if the i-th bit is set
def is_set(mask, i):
    return bool(mask & (1 << i))

# 2. Set the i-th bit
def set_bit(mask, i):
    return mask | (1 << i)

# 3. Clear the i-th bit
def clear_bit(mask, i):
    return mask & ~(1 << i)

# 4. Get the lowest set bit
def lowest_bit(mask):
    return mask & (-mask)

# 5. Enumerate subsets (enumerate subsets of mask in descending order)
def enumerate_subsets(mask):
    """Enumerate all subsets of the set bits in mask"""
    sub = mask
    while sub > 0:
        yield sub
        sub = (sub - 1) & mask
    yield 0  # Empty set

# 6. popcount (number of set bits)
def popcount(mask):
    return bin(mask).count('1')

# Usage example: Enumerate all combinations of choosing 3 elements
n = 5
for mask in range(1 << n):
    if popcount(mask) == 3:
        selected = [i for i in range(n) if mask & (1 << i)]
        print(selected)
# [0,1,2], [0,1,3], [0,1,4], [0,2,3], [0,2,4], ...
```

---

## 4. Classic Technique: Coordinate Compression and Inversion Count

### 4.1 Coordinate Compression Overview

Coordinate compression is a technique that maps large coordinate values to 0, 1, 2, ... while preserving their relative order. It is used in problems where only the relative ordering of values matters, not their absolute magnitudes.

```
Original coordinates: [100, 5000, 300, 100, 10000]

Step 1: Remove duplicates and sort
  sorted_unique = [100, 300, 5000, 10000]

Step 2: Assign 0-indexed numbers to each value
  100   -> 0
  300   -> 1
  5000  -> 2
  10000 -> 3

Step 3: Transform the original array
  Compressed: [0, 2, 1, 0, 3]

Memory efficiency:
  Original coordinate range: 0 - 10000 -> requires array of size 10001
  Compressed range:          0 - 3     -> array of size 4 is sufficient
```

### 4.2 Coordinate Compression Implementation

```python
def coordinate_compress(arr: list) -> tuple:
    """Perform coordinate compression

    Returns: (compressed array, mapping array for restoration)
    Time complexity: O(n log n)
    """
    sorted_unique = sorted(set(arr))
    compress = {v: i for i, v in enumerate(sorted_unique)}
    compressed = [compress[v] for v in arr]
    return compressed, sorted_unique

# Basic usage example
data = [100, 5000, 300, 100, 10000]
compressed, mapping = coordinate_compress(data)
print(compressed)  # [0, 2, 1, 0, 3]
print(mapping)     # [100, 300, 5000, 10000]

# Restoration
original = [mapping[c] for c in compressed]
print(original)    # [100, 5000, 300, 100, 10000]
```

### 4.3 Application of Coordinate Compression: Inversion Count

The inversion count is the number of pairs (i, j) in an array where i < j and A[i] > A[j]. It can be computed in O(n log n) by combining coordinate compression with a BIT (Binary Indexed Tree).

```python
def count_inversions(arr: list) -> int:
    """Compute inversion count using coordinate compression + BIT

    Time complexity: O(n log n)
    """
    # Coordinate compression
    compressed, _ = coordinate_compress(arr)
    n = len(compressed)
    max_val = max(compressed) + 1

    # BIT (Binary Indexed Tree)
    bit = [0] * (max_val + 2)

    def bit_update(i, delta=1):
        i += 1  # 1-indexed
        while i <= max_val + 1:
            bit[i] += delta
            i += i & (-i)

    def bit_query(i):
        """Prefix sum over [0, i]"""
        i += 1
        s = 0
        while i > 0:
            s += bit[i]
            i -= i & (-i)
        return s

    # Scan from right to left, counting how many smaller values have already appeared
    inversions = 0
    for i in range(n - 1, -1, -1):
        if compressed[i] > 0:
            inversions += bit_query(compressed[i] - 1)
        bit_update(compressed[i])

    return inversions

# Tests
print(count_inversions([3, 1, 2]))      # 2  (3>1, 3>2)
print(count_inversions([1, 2, 3]))      # 0  (already sorted)
print(count_inversions([3, 2, 1]))      # 3  (3>2, 3>1, 2>1)
print(count_inversions([5, 2, 6, 1]))   # 4  (5>2, 5>1, 2>1, 6>1)
```

### 4.4 Typical Problem Patterns Requiring Coordinate Compression

| Pattern | Description | Role of Coordinate Compression |
|:---|:---|:---|
| Interval scheduling | Interval endpoints are huge values | Compress endpoints to apply imos method |
| Inversion count | Value range is wide | Reduce BIT size to the number of distinct values |
| Rectangle area union | Coordinates are huge (e.g., 10^9) | Discretize the coordinate plane |
| Interval coverage detection | Many interval endpoints | Preprocessing for event sorting |
| Rank statistics | Values are not discrete | Convert to rank-based queries |

---

## 5. Classic Technique: Two Pointers (Sliding Window)

### 5.1 Two Pointers Overview

The two pointers technique (also known as the sliding window or "shakutori method" in Japanese) manages two pointers (left, right) on an array to efficiently search for intervals satisfying a condition. While a brute-force approach checking all intervals is O(N^2), the two pointers technique solves it in O(N).

The technique applies when the following monotonicity holds: "In an interval [l, r) satisfying the condition, extending r to the right makes the condition easier to satisfy, and advancing l to the right makes it harder to satisfy."

```
Two Pointers in action:

Array:     [2, 5, 1, 3, 7, 2, 4]
Condition: Subarray sum <= 10

step 1:  [2]               sum=2   -> OK, extend right
step 2:  [2, 5]            sum=7   -> OK, extend right
step 3:  [2, 5, 1]         sum=8   -> OK, extend right
step 4:  [2, 5, 1, 3]      sum=11  -> NG, shrink left
step 5:     [5, 1, 3]      sum=9   -> OK, extend right
step 6:     [5, 1, 3, 7]   sum=16  -> NG, shrink left
step 7:        [1, 3, 7]   sum=11  -> NG, shrink left
step 8:           [3, 7]   sum=10  -> OK, extend right
...

Both left and right advance at most N times each, so O(N) overall
```

### 5.2 Basic Two Pointers Patterns

```python
def two_pointers_max_length(arr: list, threshold: int) -> int:
    """Find the length of the longest contiguous subarray with sum <= threshold

    Time complexity: O(N)
    """
    n = len(arr)
    left = 0
    current_sum = 0
    max_len = 0

    for right in range(n):
        current_sum += arr[right]

        # Advance left until the condition is satisfied
        while current_sum > threshold:
            current_sum -= arr[left]
            left += 1

        # At this point, [left, right] is the longest interval satisfying the condition
        max_len = max(max_len, right - left + 1)

    return max_len

# Test
print(two_pointers_max_length([2, 5, 1, 3, 7, 2, 4], 10))  # 3 ([2,5,1] or [5,1,3])

def two_pointers_count(arr: list, threshold: int) -> int:
    """Count the number of contiguous subarrays with sum <= threshold

    Time complexity: O(N)
    Key insight: If [l, r] satisfies the condition, then all intervals
                 [l, l], [l, l+1], ..., [l, r] (a total of r - l + 1)
                 also satisfy it
    """
    n = len(arr)
    left = 0
    current_sum = 0
    count = 0

    for right in range(n):
        current_sum += arr[right]
        while current_sum > threshold:
            current_sum -= arr[left]
            left += 1
        count += right - left + 1  # Number of valid intervals ending at right

    return count

print(two_pointers_count([2, 5, 1, 3, 7, 2, 4], 10))  # 16
```

### 5.3 Two Pointers Application: Managing Distinct Value Counts

This technique also works for conditions like "the number of distinct values in the interval is at most K."

```python
from collections import defaultdict

def at_most_k_distinct(arr: list, k: int) -> int:
    """Length of the longest contiguous subarray with at most k distinct values

    Time complexity: O(N)
    """
    n = len(arr)
    left = 0
    freq = defaultdict(int)
    distinct = 0
    max_len = 0

    for right in range(n):
        if freq[arr[right]] == 0:
            distinct += 1
        freq[arr[right]] += 1

        while distinct > k:
            freq[arr[left]] -= 1
            if freq[arr[left]] == 0:
                distinct -= 1
            left += 1

        max_len = max(max_len, right - left + 1)

    return max_len

# Test: At most 2 distinct values
print(at_most_k_distinct([1, 2, 1, 2, 3, 3, 4], 2))  # 4 ([1,2,1,2])

def exactly_k_distinct(arr: list, k: int) -> int:
    """Count contiguous subarrays with exactly k distinct values

    Technique: exactly(k) = at_most(k) - at_most(k-1)
    """
    def at_most(k):
        if k < 0:
            return 0
        left = 0
        freq = defaultdict(int)
        distinct = 0
        count = 0
        for right in range(len(arr)):
            if freq[arr[right]] == 0:
                distinct += 1
            freq[arr[right]] += 1
            while distinct > k:
                freq[arr[left]] -= 1
                if freq[arr[left]] == 0:
                    distinct -= 1
                left += 1
            count += right - left + 1
        return count

    return at_most(k) - at_most(k - 1)

print(exactly_k_distinct([1, 2, 1, 2, 3], 2))  # 7
```

### 5.4 Two Pointers Applicability Flowchart

```
When reading a problem:
  |
  +-- Does it mention "contiguous subarray" or "interval"?
  |   +-- No  -> Two pointers is likely not applicable
  |   +-- Yes
  |       |
  |       +-- Does extending the interval "relax" the condition?
  |       |   (e.g., sum increases, distinct count increases)
  |       |   +-- Yes -> Two pointers is applicable
  |       |   +-- No  -> Consider other approaches
  |       |
  |       +-- Does the condition exhibit "monotonicity"?
  |           +-- Yes -> Two pointers
  |           +-- No  -> Segment tree or other methods
  |
  +-- Do you need to scan two arrays simultaneously?
      +-- Yes -> Two Pointers (merge operations, leveraging sorted arrays)
```

---

## 6. Classic Technique: MOD Arithmetic and Combinatorics

### 6.1 Why MOD Arithmetic?

Competitive programming very frequently features problems of the form "find the answer modulo 10^9 + 7." This is for the following reasons:

1. To prevent overflow when answers become astronomically large (e.g., the 10^18-th Fibonacci number)
2. To avoid the overhead of arbitrary-precision integer arithmetic and reduce computation
3. To guarantee uniqueness of the answer (10^9 + 7 is prime, so modular inverses exist)

```
Commonly used MOD values:
  998244353 = 119 * 2^23 + 1  (a prime suitable for NTT)
  10^9 + 7  = 1000000007      (the most common)
  10^9 + 9  = 1000000009      (used for hashing)
```

### 6.2 Basic MOD Arithmetic Implementation

```python
MOD = 10**9 + 7

# Basic operations (addition and multiplication simply take MOD)
def mod_add(a, b, mod=MOD):
    return (a + b) % mod

def mod_sub(a, b, mod=MOD):
    return (a - b) % mod  # Python correctly handles negative remainders

def mod_mul(a, b, mod=MOD):
    return (a * b) % mod

# Fast exponentiation (repeated squaring)
def mod_pow(base, exp, mod=MOD):
    """Compute base^exp mod mod in O(log exp)"""
    result = 1
    base %= mod
    while exp > 0:
        if exp & 1:              # exp is odd
            result = result * base % mod
        exp >>= 1               # Halve exp
        base = base * base % mod
    return result

# Python's built-in pow(base, exp, mod) performs the same computation
# pow(2, 10, MOD) == mod_pow(2, 10, MOD) == 1024

# Modular inverse (Fermat's little theorem: a^(-1) = a^(p-2) mod p, where p is prime)
def mod_inv(a, mod=MOD):
    """Compute the modular inverse of a (when mod is prime)"""
    return mod_pow(a, mod - 2, mod)

# Division: a / b mod p = a * b^(-1) mod p
def mod_div(a, b, mod=MOD):
    return a * mod_inv(b, mod) % mod
```

### 6.3 Combinatorics (nCr mod p)

To efficiently compute the binomial coefficient nCr with MOD, precompute factorials and their inverses.

```python
class Combinatorics:
    """Class for O(1) combinatorial calculations

    Preprocessing: O(max_n)
    Per query:     O(1)
    """
    def __init__(self, max_n: int, mod: int = MOD):
        self.mod = mod
        self.fact = [1] * (max_n + 1)       # fact[i] = i! mod p
        self.inv_fact = [1] * (max_n + 1)   # inv_fact[i] = (i!)^(-1) mod p

        # Precompute factorials
        for i in range(1, max_n + 1):
            self.fact[i] = self.fact[i - 1] * i % mod

        # Precompute inverse factorials (backtrack from the maximum)
        self.inv_fact[max_n] = pow(self.fact[max_n], mod - 2, mod)
        for i in range(max_n - 1, -1, -1):
            self.inv_fact[i] = self.inv_fact[i + 1] * (i + 1) % mod

    def comb(self, n: int, r: int) -> int:
        """Return nCr mod p in O(1)"""
        if r < 0 or r > n:
            return 0
        return self.fact[n] * self.inv_fact[r] % self.mod * self.inv_fact[n - r] % self.mod

    def perm(self, n: int, r: int) -> int:
        """Return nPr mod p in O(1)"""
        if r < 0 or r > n:
            return 0
        return self.fact[n] * self.inv_fact[n - r] % self.mod

    def homo(self, n: int, r: int) -> int:
        """Combinations with repetition: nHr = (n+r-1)Cr mod p"""
        return self.comb(n + r - 1, r)

    def catalan(self, n: int) -> int:
        """Catalan number C_n = (2n)Cn / (n+1)"""
        return self.comb(2 * n, n) * pow(n + 1, self.mod - 2, self.mod) % self.mod

# Usage example
comb = Combinatorics(200000)
print(comb.comb(10, 3))         # 120
print(comb.comb(100000, 50000)) # Large value mod 10^9+7
print(comb.perm(5, 3))          # 60
print(comb.catalan(5))          # 42
```

### 6.4 Common Pitfalls in MOD Arithmetic

```
+--------------------------------------------------------------+
|           MOD Arithmetic Pitfalls and Countermeasures          |
+--------------------+-----------------------------------------+
| Pitfall            | Countermeasure                           |
+--------------------+-----------------------------------------+
| Subtraction yields | (a - b % MOD + MOD) % MOD               |
| a negative value   | * Python handles negative remainders     |
|                    |   correctly, so this is unnecessary      |
+--------------------+-----------------------------------------+
| Applying MOD       | Multiply by the inverse:                 |
| directly to        | a * mod_inv(b) % MOD                     |
| division           | Never use % with division                |
+--------------------+-----------------------------------------+
| Intermediate value | Take % MOD at each step                  |
| overflow (C++)     | Not an issue in Python (arbitrary prec.) |
+--------------------+-----------------------------------------+
| Inverse when MOD   | Fermat's little theorem is unusable      |
| is not prime       | Use the Extended Euclidean Algorithm      |
+--------------------+-----------------------------------------+
| Confusing          | Read the problem statement carefully      |
| 998244353 with     | Define MOD as a constant at the start    |
| 10^9+7             |                                          |
+--------------------+-----------------------------------------+
```

---

## 7. Graph Algorithm Implementations for Competitive Programming

### 7.1 Graph Input Patterns

Competitive programming has several standard patterns for graph input.

```python
# Pattern 1: Adjacency list (unweighted)
# Input:
# N M
# a1 b1
# a2 b2
# ...

def read_graph_unweighted():
    N, M = map(int, input().split())
    graph = [[] for _ in range(N)]
    for _ in range(M):
        a, b = map(int, input().split())
        a -= 1; b -= 1   # Convert to 0-indexed
        graph[a].append(b)
        graph[b].append(a)  # For undirected graphs
    return N, graph

# Pattern 2: Adjacency list (weighted)
# Input:
# N M
# a1 b1 c1
# ...

def read_graph_weighted():
    N, M = map(int, input().split())
    graph = [[] for _ in range(N)]
    for _ in range(M):
        a, b, c = map(int, input().split())
        a -= 1; b -= 1
        graph[a].append((b, c))
        graph[b].append((a, c))
    return N, graph

# Pattern 3: Tree (parent specification)
# Input: Tree with N vertices where vertex i (2 <= i <= N) has parent p_i
# p_2 p_3 ... p_N

def read_tree_parent():
    N = int(input())
    parent = [-1] + [-1] + list(map(int, input().split()))
    # parent[i] = parent of vertex i (1-indexed)
    children = [[] for _ in range(N + 1)]
    for i in range(2, N + 1):
        children[parent[i]].append(i)
    return N, parent, children
```

### 7.2 Dijkstra's Algorithm (Competitive Programming Version)

```python
import heapq

def dijkstra(graph: list, start: int) -> list:
    """Dijkstra's algorithm (priority queue version)

    graph[u] = [(v, cost), ...]
    Time complexity: O((V + E) log V)
    """
    INF = float('inf')
    n = len(graph)
    dist = [INF] * n
    dist[start] = 0
    pq = [(0, start)]  # (distance, vertex)

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:     # A shorter path was already found
            continue
        for v, cost in graph[u]:
            nd = d + cost
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))

    return dist

# When path reconstruction is needed
def dijkstra_with_path(graph: list, start: int, goal: int) -> tuple:
    """Dijkstra's algorithm + path reconstruction"""
    INF = float('inf')
    n = len(graph)
    dist = [INF] * n
    prev_node = [-1] * n
    dist[start] = 0
    pq = [(0, start)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == goal:
            break
        for v, cost in graph[u]:
            nd = d + cost
            if nd < dist[v]:
                dist[v] = nd
                prev_node[v] = u
                heapq.heappush(pq, (nd, v))

    # Path reconstruction
    if dist[goal] == INF:
        return INF, []
    path = []
    v = goal
    while v != -1:
        path.append(v)
        v = prev_node[v]
    path.reverse()
    return dist[goal], path
```

### 7.3 BFS / DFS Templates for Competitive Programming

```python
from collections import deque

def bfs(graph: list, start: int) -> list:
    """BFS (Breadth-First Search) - Shortest distance in unweighted graphs

    Time complexity: O(V + E)
    """
    n = len(graph)
    dist = [-1] * n
    dist[start] = 0
    queue = deque([start])

    while queue:
        u = queue.popleft()
        for v in graph[u]:
            if dist[v] == -1:   # Not yet visited
                dist[v] = dist[u] + 1
                queue.append(v)

    return dist

def dfs_iterative(graph: list, start: int) -> list:
    """DFS (Depth-First Search) - Stack-based (avoids recursion limit)

    Time complexity: O(V + E)
    """
    n = len(graph)
    visited = [False] * n
    order = []
    stack = [start]

    while stack:
        u = stack.pop()
        if visited[u]:
            continue
        visited[u] = True
        order.append(u)
        for v in graph[u]:
            if not visited[v]:
                stack.append(v)

    return order

# Grid BFS (e.g., shortest path in a maze)
def grid_bfs(grid: list, start: tuple, goal: tuple) -> int:
    """BFS on a grid

    grid[i][j] = '.' (passable) or '#' (wall)
    """
    H, W = len(grid), len(grid[0])
    dist = [[-1] * W for _ in range(H)]
    sy, sx = start
    gy, gx = goal
    dist[sy][sx] = 0
    queue = deque([(sy, sx)])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up

    while queue:
        y, x = queue.popleft()
        if (y, x) == (gy, gx):
            return dist[y][x]
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and grid[ny][nx] != '#' and dist[ny][nx] == -1:
                dist[ny][nx] = dist[y][x] + 1
                queue.append((ny, nx))

    return -1  # Unreachable
```

### 7.4 Union-Find (Disjoint Set Union)

```python
class UnionFind:
    """Union-Find (Disjoint Set Union)

    With path compression + union by rank,
    both find and union run in amortized O(alpha(N)) ~ O(1)
    """
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n
        self.num_groups = n

    def find(self, x: int) -> int:
        """Return the root of x (with path compression)"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """Merge x and y into the same group. Returns False if already in the same group"""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        self.size[rx] += self.size[ry]
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        self.num_groups -= 1
        return True

    def connected(self, x: int, y: int) -> bool:
        """Check if x and y are in the same group"""
        return self.find(x) == self.find(y)

    def group_size(self, x: int) -> int:
        """Return the size of the group containing x"""
        return self.size[self.find(x)]

# Usage example: Count the number of connected components
def count_connected_components():
    N, M = map(int, input().split())
    uf = UnionFind(N)
    for _ in range(M):
        a, b = map(int, input().split())
        uf.union(a - 1, b - 1)
    print(uf.num_groups)
```

### 7.5 Graph Algorithm Selection Chart

```
Select the appropriate algorithm based on the problem's properties:

  Want to find shortest paths
  |
  +-- Unweighted? -> BFS  O(V+E)
  |
  +-- All weights non-negative?
  |   +-- Single source? -> Dijkstra  O((V+E) log V)
  |   +-- All pairs?     -> Floyd-Warshall  O(V^3)
  |
  +-- Negative weights present?
      +-- No negative cycles -> Bellman-Ford  O(VE)
      +-- Need negative cycle detection -> Bellman-Ford (check for update on N-th iteration)

  Want to determine connectivity
  |
  +-- Static (edge additions only) -> Union-Find  O(alpha(N))
  +-- Dynamic (edge deletions too) -> Offline processing or Link-Cut Tree

  Tree problems
  |
  +-- LCA (Lowest Common Ancestor) -> Binary lifting  O(N log N) preprocessing, O(log N) per query
  +-- Path weights -> Euler Tour + Segment Tree
  +-- Subtree information -> DFS order + BIT/Segment Tree
```

---

## 8. Prefix Sums, imos Method, and Binary Search

### 8.1 Prefix Sums

Prefix sums are a preprocessing technique that enables O(1) answers to range sum queries. With O(N) preprocessing and O(1) per query, this is the go-to method for problems involving range sums.

```python
from itertools import accumulate

def prefix_sum_demo():
    """Basic 1D prefix sum"""
    A = [3, 1, 4, 1, 5, 9, 2, 6]

    # Build prefix sum array (prepend 0)
    prefix = [0] + list(accumulate(A))
    # prefix = [0, 3, 4, 8, 9, 14, 23, 25, 31]

    # Sum of range [l, r) = prefix[r] - prefix[l]
    # Example: A[1] + A[2] + A[3] + A[4] = 1+4+1+5 = 11
    print(prefix[5] - prefix[1])  # 11

    # Sum of entire array
    print(prefix[8] - prefix[0])  # 31

# 2D prefix sums
def prefix_sum_2d(grid: list) -> list:
    """Build a 2D prefix sum table

    O(HW) preprocessing for an H x W grid
    Retrieve rectangular range sums in O(1)
    """
    H = len(grid)
    W = len(grid[0])
    # (H+1) x (W+1) prefix sum table
    ps = [[0] * (W + 1) for _ in range(H + 1)]

    for i in range(H):
        for j in range(W):
            ps[i + 1][j + 1] = (
                grid[i][j]
                + ps[i][j + 1]
                + ps[i + 1][j]
                - ps[i][j]
            )
    return ps

def query_2d(ps, r1, c1, r2, c2):
    """Retrieve the sum of rectangle [r1, r2) x [c1, c2) in O(1)"""
    return ps[r2][c2] - ps[r1][c2] - ps[r2][c1] + ps[r1][c1]

# Test
grid = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]
ps = prefix_sum_2d(grid)
print(query_2d(ps, 0, 0, 2, 2))  # 1+2+4+5 = 12
print(query_2d(ps, 1, 1, 3, 3))  # 5+6+8+9 = 28
```

### 8.2 imos Method (Difference Arrays)

The imos method is a technique for efficiently processing many range addition operations. It records +1 and -1 at the start and end of each interval, then takes a prefix sum at the end to obtain the cumulative result of all additions.

```
imos method in action:

Add +1 to interval [1, 4), +1 to [2, 6), +1 to [3, 5)

Step 1: Record in difference array
  index:  0  1  2  3  4  5  6  7
  diff:  [0, +1, +1, +1, -1, -1, -1, 0]
                        ^ -1 from [1,4)  ^ -1 from [2,6)
                   ^ +1 from [3,5)

Step 2: Take prefix sum
  result: [0, 1, 2, 3, 2, 1, 0, 0]

  Interval overlaps are computed automatically
```

```python
def imos_1d(n: int, intervals: list) -> list:
    """1D imos method

    intervals: [(l, r), ...] Add +1 to each interval [l, r)
    Time complexity: O(N + Q)  (N: array size, Q: number of intervals)
    """
    diff = [0] * (n + 1)
    for l, r in intervals:
        diff[l] += 1
        diff[r] -= 1

    # Restore via prefix sum
    result = [0] * n
    result[0] = diff[0]
    for i in range(1, n):
        result[i] = result[i - 1] + diff[i]

    return result

# Test
intervals = [(1, 4), (2, 6), (3, 5)]
print(imos_1d(7, intervals))  # [0, 1, 2, 3, 2, 1, 0]

def imos_2d(H: int, W: int, rectangles: list) -> list:
    """2D imos method

    rectangles: [(r1, c1, r2, c2), ...] Add +1 to rectangle [r1,r2) x [c1,c2)
    Time complexity: O(HW + Q)
    """
    diff = [[0] * (W + 1) for _ in range(H + 1)]
    for r1, c1, r2, c2 in rectangles:
        diff[r1][c1] += 1
        diff[r1][c2] -= 1
        diff[r2][c1] -= 1
        diff[r2][c2] += 1

    # Horizontal prefix sum
    for i in range(H):
        for j in range(1, W):
            diff[i][j] += diff[i][j - 1]
    # Vertical prefix sum
    for j in range(W):
        for i in range(1, H):
            diff[i][j] += diff[i - 1][j]

    return [row[:W] for row in diff[:H]]
```

### 8.3 Binary Search

Binary search is a technique that halves the search space to find an answer in O(log N) when the answer exhibits monotonicity.

```python
from bisect import bisect_left, bisect_right

# Pattern 1: Searching in a sorted array
def binary_search_in_sorted(arr: list):
    """Using the bisect module"""
    # arr = [1, 1, 2, 3, 4, 5, 6, 9]

    # Smallest index where value >= x
    idx = bisect_left(arr, 3)   # 3

    # Smallest index where value > x
    idx = bisect_right(arr, 3)  # 4

    # Count of elements >= x
    count_ge = len(arr) - bisect_left(arr, 3)  # 5

    # Count of elements <= x
    count_le = bisect_right(arr, 3)             # 4

    # Check if x exists
    idx = bisect_left(arr, 3)
    exists = idx < len(arr) and arr[idx] == 3   # True

# Pattern 2: Binary search on the answer (generalized binary search)
def binary_search_on_answer():
    """For problems of the form "find the minimum (maximum) value satisfying a condition" """

    def is_ok(mid: int) -> bool:
        """Function to check if mid satisfies the condition"""
        # Implement according to the problem
        return True

    # Find the minimum value satisfying the condition
    lo, hi = 0, 10**18  # Search range
    while lo < hi:
        mid = (lo + hi) // 2
        if is_ok(mid):
            hi = mid       # Condition satisfied -> search the left half
        else:
            lo = mid + 1   # Condition not satisfied -> search the right half
    # lo == hi is the answer

    # Binary search on real numbers (floating-point case)
    lo, hi = 0.0, 1e18
    for _ in range(100):   # Loop a sufficient number of times
        mid = (lo + hi) / 2
        if is_ok(int(mid)):
            hi = mid
        else:
            lo = mid
    # lo ~ hi is the answer

# Pattern 3: Minimizing the maximum (classic problem type)
def minimize_maximum_distance(positions: list, k: int) -> int:
    """Add K relay points to N positions on a number line.
    Minimize the maximum distance between adjacent points.

    Binary search on "Can we achieve max distance <= d?"
    """
    positions.sort()

    def can_achieve(d):
        """Number of relay points needed to keep max distance <= d"""
        count = 0
        for i in range(len(positions) - 1):
            gap = positions[i + 1] - positions[i]
            count += (gap - 1) // d  # Relay points needed for this gap
        return count <= k

    lo, hi = 1, positions[-1] - positions[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if can_achieve(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo
```

### 8.4 Choosing Between Prefix Sums, imos Method, and Binary Search

| Technique | Preprocessing | Per Query | Use Case |
|:---|:---|:---|:---|
| Prefix sums | O(N) | O(1) | Static range sum queries |
| 2D prefix sums | O(HW) | O(1) | Rectangular range sum queries |
| imos method | O(N+Q) | - | Batch processing of many range additions |
| 2D imos method | O(HW+Q) | - | Batch processing of many rectangular additions |
| Binary search | O(N log N) | O(log N) | Searching in sorted arrays |
| Binary search on answer | - | O(log X * f(N)) | Optimization with monotonicity |

---

## 9. AtCoder / LeetCode / Codeforces Pattern Analysis

### 9.1 AtCoder ABC Frequent Patterns

ABC (AtCoder Beginner Contest) consists of 7 problems (A through G), with the following difficulty levels and typical patterns.

```
Problem  Points   Expected Rank   Typical Patterns
-------------------------------------------------------------
A        100 pts  Gray            Arithmetic, conditionals, string output
B        200 pts  Gray-Brown      Loops, string manipulation, simulation
C        300 pts  Brown-Green     Brute force, sorting, prefix sums, greedy
D        400 pts  Green-Cyan      DP, binary search, BFS/DFS, Union-Find
E        500 pts  Cyan-Blue       Segment tree, advanced DP, number theory
F        500 pts  Blue-Yellow     Flow, matrix exponentiation, advanced combinatorics
G        600 pts  Yellow-Orange   Constructive problems, combining advanced data structures
```

**Level-specific Learning Goals and Recommended Patterns:**

| Goal | Problems to Solve Consistently | Patterns to Master |
|:---|:---|:---|
| Gray -> Brown | A, B reliably; C about half the time | Loops, conditionals, sorting |
| Brown -> Green | A, B, C reliably; D about half | Basic DP, BFS, prefix sums |
| Green -> Cyan | A-D reliably; E about half | Segment tree, Union-Find, advanced DP |
| Cyan -> Blue | A-E reliably; attempt F | Number theory, flow, inclusion-exclusion |

### 9.2 LeetCode Frequent Patterns

LeetCode is primarily used for interview preparation, and problems can be classified by pattern. Curated lists such as Blind 75 and NeetCode 150 are widely known.

```
Frequency and Interview Importance by Category:

  Category                    Frequency          Interview Importance
  ---------------------------------------------------------
  Array / HashMap             ████████████       Extremely high
  Binary Search               ██████████         High
  Sliding Window              █████████          High
  Two Pointers                ████████           High
  Trees / Graphs              ████████           High
  Dynamic Programming         ██████████         Very high
  Backtracking                ██████             Medium
  Stack / Queue               ██████             Medium
  Heap                        █████              Medium
  Greedy                      █████              Medium
  Strings                     ████               Medium
  Trie / Segment Tree         ███                Low (advanced)
```

**Frequently Asked Problems in Interviews (by Difficulty):**

- **Easy:** Two Sum, Valid Parentheses, Merge Two Sorted Lists, Best Time to Buy and Sell Stock, Valid Palindrome, Linked List Cycle, Invert Binary Tree
- **Medium:** 3Sum, Container With Most Water, LRU Cache, Course Schedule, Word Break, Coin Change, Number of Islands, Group Anagrams, Longest Substring Without Repeating Characters
- **Hard:** Merge K Sorted Lists, Trapping Rain Water, Word Ladder II, Sliding Window Maximum, Minimum Window Substring

### 9.3 Codeforces Characteristics and Patterns

Codeforces has an overwhelmingly large number of problems and features even more mathematically demanding problems than AtCoder. The typical format is problems A through E in Div.2 contests.

```
Characteristics of Each Div.2 Problem:

  A (800-1000):  Basic math, implementation
  B (1000-1300): Greedy, constructive, case analysis
  C (1300-1600): Binary search, DP, basic graph theory
  D (1600-2000): Advanced DP, number theory, segment tree
  E (2000-2400): Advanced combinatorial problems, constructive

Notable Tendencies:
  - Constructive algorithm problems are very frequent
  - Number theory (GCD, LCM, prime factorization) appears often
  - Interactive problems appear regularly
  - Game theory (Nim, Sprague-Grundy) appears occasionally
```

### 9.4 Cross-Platform Comparison

| Attribute | AtCoder | LeetCode | Codeforces |
|:---|:---|:---|:---|
| Primary purpose | Competition / skill improvement | Interview preparation | Competition / skill improvement |
| Language | Japanese | English | English |
| Difficulty display | Color rating | Easy/Medium/Hard | *A-*F + Rating |
| Contest frequency | Weekly (Saturday) | Twice weekly | 2-3 times weekly |
| Input format | Standard input | Function arguments | Standard input |
| Python support | PyPy available | Python3 | PyPy available |
| Editorials | Official editorial | Discussion | Community editorials |
| Problem count | ~5000+ | ~3000+ | ~10000+ |
| Strengths | High quality, thorough editorials | Company-specific problem sets | Overwhelming volume of problems |
| Community | Japanese-language | English-language | English / Russian-language |

---

## 10. Contest Strategy and Mental Models

### 10.1 Time Allocation Strategy

```
Optimal Time Allocation for AtCoder ABC (100 minutes):

  0-5 min:    Solve problem A reliably (warm-up)
  5-15 min:   Solve problem B reliably (implement carefully)
  15-35 min:  Work on problem C (this is the turning point)
  35-65 min:  Focus on problem D (highest points-per-minute efficiency)
  65-90 min:  Attempt problem E (solving it means a major score boost)
  90-100 min: Review / revisit unsolved problems

  Key Points:
  - Do not spend more than 30 minutes on a single problem (knowing when to move on is crucial)
  - Do not submit anything in the last 5 minutes (avoid penalty)
  - If WA occurs, first recheck sample cases

Optimal Time Allocation for LeetCode Weekly Contest (90 minutes):

  0-5 min:    Solve Q1 (Easy) quickly
  5-20 min:   Solve Q2 (Medium)
  20-55 min:  Work on Q3 (Medium-Hard)
  55-90 min:  Attempt Q4 (Hard)
```

### 10.2 Debugging Strategy

A systematic debugging methodology for when WA (Wrong Answer) or RE (Runtime Error) occurs during a contest.

```python
# Debugging Strategy 1: Stress testing against a brute force solution
import random

def brute_force(arr):
    """Brute force O(N^2) solution"""
    # ...correct but slow solution
    pass

def optimized(arr):
    """Optimized O(N log N) solution"""
    # ...fast but correctness unverified
    pass

def stress_test(n_tests=10000, max_n=10, max_val=100):
    """Compare brute force and optimized solution with random tests"""
    for _ in range(n_tests):
        n = random.randint(1, max_n)
        arr = [random.randint(1, max_val) for _ in range(n)]
        expected = brute_force(arr)
        actual = optimized(arr)
        if expected != actual:
            print(f"MISMATCH: arr={arr}")
            print(f"  expected={expected}, actual={actual}")
            return
    print("All tests passed!")

# Debugging Strategy 2: Edge case checklist
EDGE_CASES = """
  N = 0 (empty input)
  N = 1 (single element)
  N = 2 (minimum pair)
  All elements are the same value
  All elements at maximum value (10^9, etc.)
  Already sorted (ascending)
  Already sorted (descending)
  Negative values present
  Calculations that may overflow
  0-indexed vs 1-indexed conversion mistakes
"""
```

### 10.3 Problem Analysis Framework

Use the following procedure to narrow down the solution approach when reading a problem.

```
Step 1: Check the constraints
  |
  +-- N <= 8        -> Bit brute force or permutation brute force
  +-- N <= 20       -> Bit brute force
  +-- N <= 40       -> Meet in the Middle
  +-- N <= 300      -> O(N^3) is feasible (DP, Floyd-Warshall)
  +-- N <= 3000     -> O(N^2) is feasible
  +-- N <= 10^5     -> O(N log N) or O(N sqrt(N))
  +-- N <= 10^6     -> O(N) or O(N log N)
  +-- N <= 10^18    -> O(log N) or O(sqrt(N)) (mathematical methods)

Step 2: Classify the problem type
  |
  +-- Shortest path       -> BFS / Dijkstra / Bellman-Ford
  +-- Max/min optimization -> DP / Greedy / Binary search
  +-- Counting            -> DP / Inclusion-exclusion / MOD arithmetic
  +-- Connectivity        -> Union-Find / BFS / DFS
  +-- Range queries       -> Segment tree / BIT / Prefix sums
  +-- String matching     -> KMP / Z-algorithm / Rolling hash

Step 3: Estimate the time complexity and proceed to implementation
```

### 10.4 Mental Management

```
Mental management rules during a contest:

  1. If stuck on a problem, move to the next one
     -> A later problem may be easier (especially when E is easier than D)

  2. Do not panic on WA (Wrong Answer)
     -> First verify sample cases by hand, then examine edge cases

  3. Do not check the standings
     -> Viewing the leaderboard mid-contest causes anxiety

  4. Review for 1 minute before submitting
     -> Check for variable name typos, off-by-one errors, output format

  5. Always do a post-contest review
     -> Read editorials for unsolved problems and re-implement them within one week
```

---

## 11. Platform Comparison and Learning Roadmap

### 11.1 Technique Quick Reference

| Technique | Time Complexity | Use Case | Frequency |
|:---|:---|:---|:---|
| Prefix sums | O(N) preprocessing | Range sum queries | Very high |
| Binary search | O(log N) | Monotonic search | Very high |
| Two pointers | O(N) | Finding intervals satisfying a condition | High |
| Bit brute force | O(2^N) | Full enumeration for N<=20 | Medium |
| Coordinate compression | O(N log N) | Discretizing large values | Medium |
| MOD arithmetic | O(N) | Computation with huge numbers | Very high |
| Union-Find | O(alpha(N)) | Connected component management | High |
| Dijkstra | O((V+E) log V) | Weighted shortest paths | High |
| Binary lifting | O(N log N) | LCA, repeated transitions | Medium |
| Segment tree | O(N) build, O(log N) query | General range queries | High (Cyan+) |
| Matrix exponentiation | O(K^3 log N) | Fast evaluation of linear recurrences | Low |
| Lazy segment tree | O(log N) | Range update + range query | Medium (Blue+) |

### 11.2 Phased Learning Roadmap

**Phase 1: Beginner (Gray -> Brown, approx. 1-3 months)**

Topics:
- Basic syntax of your chosen language (Python or C++)
- I/O processing, loops, conditionals
- Using sorting algorithms
- Linear search and basic brute force

Recommended Problems:
- 50+ A and B problems from AtCoder ABC
- AtCoder Beginners Selection (10 official recommended problems)

**Phase 2: Foundations (Brown -> Green, approx. 3-6 months)**

Topics:
- Prefix sums, binary search
- BFS / DFS basics
- DP basics (knapsack, LCS, LIS)
- Basic greedy patterns
- Bit brute force

Recommended Problems:
- 100+ C and D problems from AtCoder ABC
- Typical 90 Competitive Programming Problems (by E869120), 2-3 star difficulty
- Educational DP Contest (EDPC) problems A through K

**Phase 3: Intermediate (Green -> Cyan, approx. 6 months - 1 year)**

Topics:
- Segment tree and BIT
- Union-Find
- Dijkstra's algorithm, Floyd-Warshall algorithm
- Advanced DP (digit DP, tree DP, bitmask DP)
- Basic number theory (prime factorization, modular inverse, Chinese Remainder Theorem)

Recommended Problems:
- E and F problems from AtCoder ABC
- Typical 90 Competitive Programming Problems, 4-5 star difficulty
- AtCoder Library Practice Contest

**Phase 4: Advanced (Cyan -> Blue and above, approx. 1+ year)**

Topics:
- Lazy propagation segment tree
- Maximum flow, minimum cut
- Matrix exponentiation, inclusion-exclusion principle
- String algorithms (SA, LCP, Aho-Corasick)
- Sweep line, convex hull

Recommended Problems:
- Past ARC and AGC problems
- Codeforces Div.1 problems
- IOI / ICPC past problems

### 11.3 Learning Resources

```
+--------------------------------------------------------------+
|                    Learning Resource Map                       |
+---------------+----------------------------------------------+
| Books         | - Programming Contest Challenge Book          |
|               |   (aka "Ant Book", Akiba et al.)              |
|               | - Problem Solving with Algorithms and         |
|               |   Data Structures (Otsuki)                    |
|               | - Competitive Programming 3 (Halim)           |
|               | - Introduction to Algorithms (CLRS)           |
+---------------+----------------------------------------------+
| Websites      | - AtCoder Problems (kenkoooo)                 |
|               | - NeetCode 150                                |
|               | - Typical 90 CP Problems                      |
|               | - EDPC (Educational DP Contest)               |
|               | - algo-method                                 |
+---------------+----------------------------------------------+
| Libraries     | - AtCoder Library (ACL)                        |
|               | - Python: sortedcontainers, networkx          |
|               | - C++: bits/stdc++.h, pb_ds                  |
+---------------+----------------------------------------------+
| Communities   | - AtCoder Official Discord                    |
|               | - Twitter/X #competitive_programming tag      |
|               | - Codeforces Blog                             |
|               | - LeetCode Discussion                         |
+---------------+----------------------------------------------+
```

---

## 12. Exercises (3 Levels)

### 12.1 Beginner Exercises (Gray-Brown Level)

**Exercise 1-1: Subset Sum Check**

> Given the array A = [2, 7, 3, 5, 11] and a target value of 15, determine whether some elements of A can be selected so that their sum equals the target.

Hint: Since N <= 5, bit brute force (2^5 = 32 combinations) is sufficient.

```python
# Sample solution
def solve_1_1():
    A = [2, 7, 3, 5, 11]
    target = 15
    n = len(A)
    for mask in range(1 << n):
        total = sum(A[i] for i in range(n) if mask & (1 << i))
        if total == target:
            subset = [A[i] for i in range(n) if mask & (1 << i)]
            print(f"Yes: {subset}")
            return
    print("No")

solve_1_1()  # Yes: [7, 3, 5] (sum = 15)
```

**Exercise 1-2: Longest Contiguous Subarray**

> Given the array A = [1, 3, 2, 5, 4, 7, 6, 8], find the maximum length of a contiguous subarray whose sum is at most 12.

Hint: Use the two pointers technique.

**Exercise 1-3: Grid Shortest Path**

> In the following 5x5 grid, find the length of the shortest path from the top-left (0,0) to the bottom-right (4,4). '.' is passable and '#' is a wall.

```
.....
.#.#.
.#...
...#.
.#...
```

Hint: Use BFS.

### 12.2 Intermediate Exercises (Green-Cyan Level)

**Exercise 2-1: Computing Inversion Count**

> Compute the inversion count of the array A = [5, 3, 1, 4, 2].
> Solve it in O(N log N) using coordinate compression + BIT.

Expected output: 7 (pairs: (5,3), (5,1), (5,4), (5,2), (3,1), (3,2), (4,2))

**Exercise 2-2: Minimizing the Maximum**

> Given positions [1, 5, 12, 23, 37, 50] on a number line, add 2 relay points to minimize the maximum distance between adjacent points.

Hint: Binary search on "Can we achieve max distance <= d?"

**Exercise 2-3: Distinct Value Count in Intervals**

> Given the array A = [1, 2, 1, 3, 2, 3, 1, 4], count the number of contiguous subarrays containing exactly 3 distinct values.

Hint: Use the identity exactly(k) = at_most(k) - at_most(k-1).

### 12.3 Advanced Exercises (Cyan-Blue Level)

**Exercise 3-1: Combinatorial Counting**

> Find the number of ways to divide N people into K groups (each group having at least 1 person), modulo 10^9 + 7.
> Compute the answer for N = 10, K = 3.

Hint: Compute the Stirling numbers of the second kind S(N, K) using DP.

**Exercise 3-2: Tree Diameter**

> Given a tree with N vertices, find the distance between the two farthest vertices (the tree diameter).

Hint: Run BFS from any vertex to find the farthest vertex, then BFS again from there (two BFS passes).

**Exercise 3-3: Weighted Interval Scheduling**

> Given N jobs, each with a start time s_i, end time e_i, and reward w_i, select non-overlapping jobs to maximize the total reward.

Hint: Sort by end time, then use binary search + DP.

---

## 13. Anti-patterns

### Anti-pattern 1: Overestimating Python's Speed

Python (CPython) is 50-100x slower than C++. The practical upper limit is about 10^7 loop iterations; 10^8 iterations will certainly cause TLE (Time Limit Exceeded).

```python
# BAD: 10^8 loop iterations in Python -> about 10 seconds, guaranteed TLE
result = 0
for i in range(10**8):
    result += i

# GOOD: Solution 1 - Submit with PyPy (3-10x faster)
# Just change the submission language to "PyPy3"

# GOOD: Solution 2 - List comprehensions (2-3x faster than for loops)
# BAD
result = []
for i in range(n):
    result.append(i * i)

# GOOD
result = [i * i for i in range(n)]

# GOOD: Solution 3 - Use built-in functions
# BAD
total = 0
for x in arr:
    total += x

# GOOD
total = sum(arr)

# GOOD: Solution 4 - Localize variables (global access is slow)
def solve():
    # Local variables use LOAD_FAST instruction for fast access
    n = len(arr)
    for i in range(n):
        pass
```

### Anti-pattern 2: Overlooking Corner Cases

Most WA results in contests are caused by bugs related to special inputs (corner cases).

```python
# BAD: Forgetting the N=1 case
def solve_bad():
    N = int(input())
    A = list(map(int, input().split()))
    # IndexError when N=1 due to A[1]!
    print(A[0] + A[1])

# GOOD: Handle edge cases first
def solve_good():
    N = int(input())
    A = list(map(int, input().split()))
    if N == 1:
        print(A[0])
        return
    # From here on, N >= 2 is assumed
    print(A[0] + A[1])
```

### Anti-pattern 3: Floating-Point Comparison

```python
# BAD: Equality comparison with floating-point numbers
if 0.1 + 0.2 == 0.3:  # Evaluates to False!
    print("equal")

# GOOD: Compare with a sufficiently small epsilon
EPS = 1e-9
if abs((0.1 + 0.2) - 0.3) < EPS:
    print("equal")

# BETTER: Convert to integers when possible
# Decimal coordinates -> multiply by 10^6 and use integers
# Probabilities -> compute as fractions (use modular inverse for division)
```

### Anti-pattern 4: Forgetting the Recursion Depth Limit

```python
# BAD: Python's default recursion limit is 1000
def dfs(v, graph, visited):
    visited[v] = True
    for u in graph[v]:
        if not visited[u]:
            dfs(u, graph, visited)
# RecursionError on a tree with N = 10^5

# GOOD: Extend the recursion limit
import sys
sys.setrecursionlimit(10**6)

# BETTER: Rewrite as iterative DFS using a stack
def dfs_iterative(start, graph):
    visited = set()
    stack = [start]
    while stack:
        v = stack.pop()
        if v in visited:
            continue
        visited.add(v)
        for u in graph[v]:
            if u not in visited:
                stack.append(u)
```

---

## 14. FAQ

### Q1: What is the best language to start competitive programming with?

**A:** C++ is the most recommended language. Its execution speed is fast, and the STL (Standard Template Library) is extremely powerful for competitive programming (set, map, priority_queue, lower_bound, etc.). However, Python excels in ease of writing, and it is the dominant language on LeetCode in particular. A practical path for beginners is to start with Python and switch to C++ when speed becomes necessary. On AtCoder, PyPy allows Python solutions to pass for most problems.

### Q2: What is an effective practice method?

**A:** The following 4-step approach is recommended:
1. **1-2 problems daily**: Use AtCoder Problems (kenkoooo) to solve problems near your current difficulty rating
2. **Read editorials thoroughly**: After spending 30 minutes on problems you could not solve, read the editorial, understand it, and re-implement it yourself
3. **Cycle through classic problems**: Repeatedly solve the Typical 90 CP Problems and EDPC to internalize patterns
4. **Participate in weekly contests**: Join ABC every week to gain experience solving problems under real-time pressure and time constraints

Quality over quantity; depth of understanding matters. Fully understanding one problem is more valuable than superficially solving ten.

### Q3: Is competitive programming useful for professional work?

**A:** The direct benefits are limited, but the indirect benefits are significant.

**Where it helps:**
- A broader repertoire of algorithms that enables writing more efficient code
- A coding habit that is conscious of computational complexity
- Heightened sensitivity to edge cases, leading to fewer bugs
- Improved ability to verify code correctness (test design skills)
- A major advantage in technical interviews

**Skills that require separate development:**
- Maintainable code design (variable naming, function decomposition, testability)
- Team development skills (code reviews, documentation)
- Overall system design capability (distributed systems, DB design)

### Q4: How long does it take to reach Green on AtCoder?

**A:** Individual variance is large, but for someone with programming experience practicing 1-2 hours daily, 3-6 months is a rough estimate. Mathematical aptitude accelerates progress. The key is to consistently do the following three things:
1. Solve at least 100 past ABC problems
2. Solidly master classic patterns (DP, BFS, prefix sums, binary search)
3. Participate in contests every week to build practical experience

### Q5: Should I prioritize LeetCode or AtCoder?

**A:** It depends on your goal. If you are targeting FAANG or other international tech company positions, prioritize LeetCode. LeetCode Medium-level problems are commonly asked in interviews. If you want to purely improve your algorithmic skills or prepare for Japanese company technical interviews, AtCoder is more suitable. Ideally, work on both in parallel, but if time is limited, choose based on your objective.

### Q6: What should I do when I cannot solve a problem during a contest?

**A:** Use the following decision criteria:
1. **No approach after 15 minutes** -> Move to the next problem. A later problem may be easier
2. **Approach is clear but implementation won't finish in time** -> Check remaining time. If 30+ minutes remain, attempt it; otherwise, review other problems
3. **Continuous WA** -> Hand-trace sample cases -> Check edge cases -> After 3 WAs, move on

After the contest, always read editorials for unsolved problems and make a habit of getting AC on them within one week.

---


## FAQ

### Q1: What is the most important point for learning this topic?

Gaining practical experience is the most important factor. Understanding deepens not through theory alone, but by actually writing and running code to confirm behavior.

### Q2: What common mistakes do beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this knowledge applied in professional practice?

The knowledge from this topic is frequently applied in day-to-day development work. It becomes especially important during code reviews and architecture design.

---

## 15. Summary

| Topic | Key Points |
|:---|:---|
| AtCoder | Japanese, weekly contests, color rating makes skill level clear |
| LeetCode | English, the go-to for interview preparation, practice by pattern |
| Codeforces | English, overwhelming problem volume, rich in mathematical problems |
| Templates | Prepare fast I/O, recursion limit extension, and MOD arithmetic as boilerplate |
| Classic Techniques | Bit brute force, coordinate compression, two pointers, prefix sums, and MOD arithmetic are frequent |
| Graphs | BFS / DFS / Dijkstra / Union-Find are essential from intermediate level onward |
| Contest Strategy | Time management, debugging methods, and mental management determine performance |
| Practice Method | 1-2 problems daily, thorough editorial reading, consistent contest participation |
| Key to Growth | Quality over quantity, pattern recognition, continuous reflection |

---

## Recommended Next Guides

- [Problem Solving](./00-problem-solving.md) -- A systematic approach to algorithmic problems
- [Dynamic Programming](../02-algorithms/04-dynamic-programming.md) -- The most frequently tested algorithm paradigm
- [Segment Tree](../03-advanced/01-segment-tree.md) -- An essential data structure from intermediate level onward

---

## 16. References

1. Akiba, T., Iwata, Y., & Kitagawa, Y. (2012). *Programming Contest Challenge Book, 2nd Edition*. Mynavi Publishing. -- Known as the "Ant Book." The most acclaimed introductory book for competitive programming.
2. Otsuki, K. (2020). *Sharpen Your Problem-Solving Skills! Algorithms and Data Structures*. Kodansha. -- A modern algorithm textbook written in Japanese. Features abundant diagrams and thorough explanations.
3. E869120. "Typical 90 Competitive Programming Problems." https://github.com/E869120/kyopro-tenkei-90 -- A problem set for systematically learning 90 classic techniques. Covers beginner to advanced levels.
4. kenkoooo. "AtCoder Problems." https://kenkoooo.com/atcoder/ -- Displays past AtCoder problems with difficulty ratings. An indispensable tool for tracking progress.
5. NeetCode. "NeetCode 150." https://neetcode.io/ -- Organizes 150 frequently asked LeetCode problems by pattern. A staple resource for interview preparation.
6. Halim, S. & Halim, F. (2013). *Competitive Programming 3*. -- A world-renowned competitive programming textbook. Comprehensive algorithm coverage.
7. Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms, 4th Edition*. MIT Press. -- Known as CLRS. The most authoritative textbook for learning the theoretical foundations of algorithms.
8. AtCoder Library (ACL). https://github.com/atcoder/ac-library -- AtCoder's official algorithm library. Provides high-quality implementations of segment trees, lazy propagation, flow, and more.

---

## References

- [MDN Web Docs](https://developer.mozilla.org/) - Web technology reference
- [Wikipedia](https://en.wikipedia.org/) - Overview of technical concepts
