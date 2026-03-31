# Arrays and Strings

> An array is a contiguous region of memory, and its simplicity makes it the data structure most compatible with CPU caches.
> It is the most fundamental building block that every programmer learns first and continues to use throughout their career.

## Learning Objectives

- [ ] Understand the relationship between array memory layout and CPU caches
- [ ] Explain the internal implementation of static and dynamic arrays
- [ ] Understand the differences and practical implications of string encodings (UTF-8/UTF-16/UTF-32)
- [ ] Implement classic array techniques (binary search, two pointers, sliding window)
- [ ] Explain the principles of string search algorithms (KMP, Rabin-Karp, Boyer-Moore)
- [ ] Recognize anti-patterns and avoid performance pitfalls

## Prerequisites


---

## 1. Why Arrays and Strings Matter

### 1.1 The Most Primitive Data Structure in Computing

An array is a data structure that stores elements of the same type in a contiguous region of memory.
This "contiguity" is the most important characteristic that distinguishes arrays from all other data structures,
and it provides a decisive performance advantage in modern CPU architectures.

To understand why arrays are so important, you need to think from a hardware perspective.

### 1.2 Memory Contiguity and Cache Locality

Modern CPUs access cache orders of magnitude faster than main memory (DRAM).
Typical access times are as follows:

```
CPU Cache Hierarchy and Access Times:

  +-------------------------------------------------------------+
  |                      CPU Core                                |
  |  +-----------------------------------------------------+    |
  |  |  Register           : ~0.3 ns    (1 cycle)          |    |
  |  +-----------------------------------------------------+    |
  |  +-----------------------------------------------------+    |
  |  |  L1 Cache (64KB)    : ~1 ns      (3-4 cycles)       |    |
  |  +-----------------------------------------------------+    |
  |  +-----------------------------------------------------+    |
  |  |  L2 Cache (256KB)   : ~4 ns      (12 cycles)        |    |
  |  +-----------------------------------------------------+    |
  +-------------------------------------------------------------+
  +-------------------------------------------------------------+
  |  L3 Cache (8MB)        : ~12 ns     (40 cycles)             |
  +-------------------------------------------------------------+
  +-------------------------------------------------------------+
  |  Main Memory (DRAM)    : ~100 ns    (300+ cycles)           |
  +-------------------------------------------------------------+
  +-------------------------------------------------------------+
  |  SSD                   : ~100,000 ns                        |
  +-------------------------------------------------------------+

  -> L1 cache is about 100x faster than DRAM
  -> Sequential array access is prefetched in cache line (64B) units
  -> Linked lists cause frequent cache misses due to pointer chasing
```

When the CPU loads data from memory, it fetches not just a single byte but a 64-byte chunk called a "cache line."
Because array elements are contiguous, accessing one element simultaneously loads neighboring elements into the cache.
This is known as "spatial locality."

Furthermore, sequential array traversal patterns are detected by the CPU's hardware prefetcher,
which preloads the next needed data into the cache. In some cases, this effectively reduces
memory access latency to near zero.

### 1.3 Array vs Linked List: The Impact of Caching

Theoretical computational complexity alone cannot predict actual performance.
The following C program compares traversal speeds of arrays and linked lists:

```c
/* cache_benchmark.c - Array vs linked list traversal speed comparison */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 10000000

typedef struct Node {
    int value;
    struct Node *next;
} Node;

int main(void) {
    /* --- Array traversal --- */
    int *arr = malloc(sizeof(int) * N);
    for (int i = 0; i < N; i++) arr[i] = i;

    clock_t start = clock();
    long long sum_arr = 0;
    for (int i = 0; i < N; i++) {
        sum_arr += arr[i];
    }
    clock_t end = clock();
    double time_arr = (double)(end - start) / CLOCKS_PER_SEC;

    /* --- Linked list traversal --- */
    Node *head = NULL;
    for (int i = N - 1; i >= 0; i--) {
        Node *node = malloc(sizeof(Node));
        node->value = i;
        node->next = head;
        head = node;
    }

    start = clock();
    long long sum_list = 0;
    Node *cur = head;
    while (cur) {
        sum_list += cur->value;
        cur = cur->next;
    }
    end = clock();
    double time_list = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Array traversal      : %.4f sec (sum = %lld)\n", time_arr, sum_arr);
    printf("Linked list traversal: %.4f sec (sum = %lld)\n", time_list, sum_list);
    printf("Ratio                : %.1fx\n", time_list / time_arr);

    /* Free memory */
    free(arr);
    while (head) {
        Node *tmp = head;
        head = head->next;
        free(tmp);
    }
    return 0;
}
```

Typical results:

```
Array traversal      : 0.0123 sec (sum = 49999995000000)
Linked list traversal: 0.0891 sec (sum = 49999995000000)
Ratio                : 7.2x
```

Despite both being O(n) traversals in theory, the array is approximately 5-10x faster than the linked list.
This difference is due to cache miss frequency and becomes more pronounced as data size increases.

---

## 2. Static Arrays

### 2.1 Memory Model

Static arrays have their size determined at compile time and are placed on the stack or data segment.

```
Memory layout of a static array:

  Declaration: int arr[5] = {10, 20, 30, 40, 50};
  Type: int (4 bytes)

  Base address: 0x7ffc00001000

  Address         Value    Index
  -----------------------------------
  0x7ffc00001000   10       arr[0]
  0x7ffc00001004   20       arr[1]
  0x7ffc00001008   30       arr[2]
  0x7ffc0000100C   40       arr[3]
  0x7ffc00001010   50       arr[4]
  -----------------------------------

  Address computation formula:
    Address of arr[i] = base_address + i * sizeof(int)
                      = 0x7ffc00001000 + i * 4

  Example: arr[3] = 0x7ffc00001000 + 3 * 4 = 0x7ffc0000100C

  -> One addition and one multiplication determine the address -> O(1) random access
```

This simple address computation guarantees O(1) random access.
While linked lists require traversing i nodes to reach the i-th element (O(n)),
arrays can directly compute the memory address from the index.

### 2.2 Multidimensional Arrays: Row-Major vs Column-Major

There are two approaches for mapping a 2D array into 1D memory:

```
Memory layout of 2D array matrix[3][4]:

  Logical representation:
       Col0 Col1 Col2 Col3
  Row0 [  1,   2,   3,   4 ]
  Row1 [  5,   6,   7,   8 ]
  Row2 [  9,  10,  11,  12 ]

  Row-major order — C, C++, Python (numpy default), Rust
  Memory: [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 ]
           <-- Row0 --> <-- Row1 --> <--- Row2  --->

  matrix[i][j] address = base + (i * num_cols + j) * sizeof(element)
  Example: matrix[1][2] = base + (1 * 4 + 2) * 4 = base + 24

  Column-major order — Fortran, MATLAB, Julia, R
  Memory: [ 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12 ]
           <-Col0-> <-Col1--> <--Col2--> <--Col3-->

  matrix[i][j] address = base + (j * num_rows + i) * sizeof(element)
  Example: matrix[1][2] = base + (2 * 3 + 1) * 4 = base + 28
```

Traversing a row-major array along rows accesses data along cache lines and is fast,
while column-wise traversal crosses cache lines, increasing cache misses.

The following C program demonstrates this difference:

```c
/* row_vs_col.c - Row-major vs column-major traversal speed comparison */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ROWS 4096
#define COLS 4096

int main(void) {
    int (*matrix)[COLS] = malloc(sizeof(int[ROWS][COLS]));

    /* Initialization */
    for (int i = 0; i < ROWS; i++)
        for (int j = 0; j < COLS; j++)
            matrix[i][j] = i + j;

    /* Row-major traversal (cache-friendly) */
    clock_t start = clock();
    long long sum1 = 0;
    for (int i = 0; i < ROWS; i++)
        for (int j = 0; j < COLS; j++)
            sum1 += matrix[i][j];
    clock_t end = clock();
    double time_row = (double)(end - start) / CLOCKS_PER_SEC;

    /* Column-major traversal (cache-unfriendly) */
    start = clock();
    long long sum2 = 0;
    for (int j = 0; j < COLS; j++)
        for (int i = 0; i < ROWS; i++)
            sum2 += matrix[i][j];
    end = clock();
    double time_col = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Row-major traversal: %.4f sec (sum = %lld)\n", time_row, sum1);
    printf("Col-major traversal: %.4f sec (sum = %lld)\n", time_col, sum2);
    printf("Ratio              : %.1fx\n", time_col / time_row);

    free(matrix);
    return 0;
}
```

Typical result: column-major traversal is 3-8x slower than row-major traversal.
This demonstrates that cache effects alone can produce such a large difference,
even though both have O(n^2) computational complexity.

### 2.3 Practical Considerations for Multidimensional Arrays

In numerical computing and data science, matrix operation performance is particularly important.
NumPy defaults to row-major (C order), but when interfacing with Fortran,
column-major (F order) may need to be specified.

```python
import numpy as np

# C order (row-major, default)
a = np.array([[1, 2, 3], [4, 5, 6]], order='C')
print(a.strides)  # (24, 8) — 24 bytes to advance one row, 8 bytes for one column

# Fortran order (column-major)
b = np.array([[1, 2, 3], [4, 5, 6]], order='F')
print(b.strides)  # (8, 16) — 8 bytes to advance one row, 16 bytes for one column

# Performance difference
import timeit

n = 2000
c_arr = np.random.rand(n, n)               # C order
f_arr = np.asfortranarray(np.random.rand(n, n))  # Fortran order

# Row-wise sum (C order advantageous)
t1 = timeit.timeit(lambda: c_arr.sum(axis=1), number=100)
t2 = timeit.timeit(lambda: f_arr.sum(axis=1), number=100)
print(f"Row sum: C order={t1:.4f}s, F order={t2:.4f}s")

# Column-wise sum (Fortran order advantageous)
t3 = timeit.timeit(lambda: c_arr.sum(axis=0), number=100)
t4 = timeit.timeit(lambda: f_arr.sum(axis=0), number=100)
print(f"Col sum: C order={t3:.4f}s, F order={t4:.4f}s")
```

---

## 3. Dynamic Arrays

### 3.1 Basic Concepts of Dynamic Arrays

A dynamic array automatically expands its size as needed, without requiring
a predetermined upper bound on the number of elements.
Python's `list`, Java's `ArrayList`, C++'s `std::vector`, and Rust's `Vec<T>` all fall into this category.

Internally, it maintains a fixed-size array and, when capacity is exhausted,
allocates a larger array and copies all elements over.

```
Dynamic array resize behavior:

  Initial state (capacity=4, size=4):
  +---+---+---+---+
  | A | B | C | D |
  +---+---+---+---+

  Adding element 'E' -> capacity exceeded!

  Step 1: Allocate a new array (capacity=8)
  +---+---+---+---+---+---+---+---+
  |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+

  Step 2: Copy existing elements
  +---+---+---+---+---+---+---+---+
  | A | B | C | D |   |   |   |   |
  +---+---+---+---+---+---+---+---+

  Step 3: Add the new element
  +---+---+---+---+---+---+---+---+
  | A | B | C | D | E |   |   |   |
  +---+---+---+---+---+---+---+---+
  (size=5, capacity=8)

  Step 4: Free the old array

  -> Resize cost: O(n) (full element copy)
  -> However, by controlling resize frequency,
     the "amortized complexity" of append can be O(1)
```

### 3.2 Amortized Complexity Analysis

For the dynamic array `append` operation, suppose we adopt a "doubling strategy"
that allocates twice the capacity when capacity is exhausted.

Let us compute the total cost for n append operations:

```
Amortized analysis of the doubling strategy:

  Total cost for n append operations:

  Normal append:     cost 1 x n times    = n
  Copy cost at
  resize:            1 + 2 + 4 + 8 + ... + n/2 + n
                     = 2n - 1 (geometric series sum)

  Total cost = n + (2n - 1) = 3n - 1

  Amortized cost per operation = (3n - 1) / n ~ 3 = O(1)

  Alternative proof: Potential method
  Potential function: Phi(i) = 2 * size - capacity
  * Phi is 0 right after resize and equals size right before resize

  Normal append:
    Actual cost = 1
    Delta Phi = 2  (size increases by 1)
    Amortized cost = 1 + 2 = 3

  Append with resize:
    Actual cost = size + 1  (copy + add)
    Delta Phi = 2 - size  (capacity doubles, size increases by 1)
    Amortized cost = (size + 1) + (2 - size) = 3

  -> In both cases, the amortized cost is 3 = O(1)
```

### 3.3 Choosing the Growth Factor

Doubling (growth factor 2.0) is theoretically simple, but actual implementations sometimes use different growth factors.

```
Growth factors in various languages/libraries:

  +--------------------+---------+-----------------------------+
  | Implementation     | Growth  | Reason                      |
  |                    | Factor  |                             |
  +--------------------+---------+-----------------------------+
  | CPython list       | ~1.125  | Conservative at small sizes,|
  |                    |         | gradually increasing        |
  | Java ArrayList     | 1.5     | Balance between memory      |
  |                    |         | efficiency and copy cost     |
  | C++ std::vector    | 2.0*    | Implementation-dependent.   |
  |                    |         | GCC=2, MSVC=1.5             |
  | Rust Vec           | 2.0     | Simplicity-first            |
  | Go slice           | ~1.25-2 | Varies by size              |
  | C# List<T>        | 2.0     |                             |
  +--------------------+---------+-----------------------------+

  Growth factor trade-offs:
  - Larger growth factor (e.g., 2.0):
    -> Fewer resizes -> append is faster
    -> Up to 50% memory waste
    -> Old regions cannot be reused (discussed below)

  - Smaller growth factor (e.g., 1.5):
    -> More resizes -> append is slightly slower
    -> Up to 33% memory waste
    -> Old regions easier to reuse
```

With a growth factor of 2.0, the total size of freed old regions is always less than the size of the new array.
This means it is fundamentally impossible to reuse old regions to place the new array.
With a growth factor of 1.5, after several resizes the total of old regions becomes large enough to accommodate the new array,
giving the memory allocator an opportunity to reuse them.
This is one reason Java and MSVC chose a growth factor of 1.5.

### 3.4 CPython list Internal Implementation

CPython's list is an array of pointers to PyObjects.

```python
# Observing the internal structure of CPython list
import sys

lst = []
prev_size = sys.getsizeof(lst)
print(f"{'len':>4s}  {'sizeof':>8s}  {'est. capacity':>13s}  {'resize':>8s}")
print("-" * 50)

for i in range(65):
    lst.append(i)
    current_size = sys.getsizeof(lst)
    # Pointer size (64bit) = 8 bytes
    # Overhead = size of empty list
    overhead = sys.getsizeof([])
    capacity = (current_size - overhead) // 8
    resized = "*" if current_size != prev_size else ""
    if resized or i < 10 or i % 10 == 0:
        print(f"{len(lst):4d}  {current_size:8d}  {capacity:13d}  {resized:>8s}")
    prev_size = current_size

# CPython growth pattern (from cpython/Objects/listobject.c):
# new_allocated = new_size + (new_size >> 3) + (6 if new_size < 9 else 3)
# That is: new_allocated = new_size + new_size/8 + constant
# Growth rate increases by about 12.5% each time (= growth factor ~1.125)
```

CPython's list is an "array of pointers," so the elements themselves reside elsewhere in memory.
This is not a true contiguous array, so for numerical computing, NumPy's ndarray should be used.

```
CPython list vs NumPy ndarray memory layout:

  CPython list (array of pointers):
  list object
  +------------------+
  | ob_refcnt        |
  | ob_type          |
  | ob_size (length) |
  | ob_item ---------+--> +-----+-----+-----+-----+ pointer array
  | allocated        |    | ptr | ptr | ptr | ptr |
  +------------------+    +--+--+--+--+--+--+--+--+
                             |     |     |     |
                             v     v     v     v
                          int(1) int(2) int(3) int(4)  <- individual PyObjects
                          (28B)  (28B)  (28B)  (28B)     scattered on the heap

  -> Per element: pointer 8B + PyObject 28B = 36B (for int)
  -> Cache efficiency: poor (cache misses due to pointer indirection)

  NumPy ndarray (contiguous array):
  ndarray object
  +------------------+
  | (header)         |
  | data ------------+--> +---------+---------+---------+---------+
  | shape            |    | 1 (8B)  | 2 (8B)  | 3 (8B)  | 4 (8B)  |
  | strides          |    +---------+---------+---------+---------+
  | dtype            |    <- contiguous memory region, values stored directly ->
  +------------------+

  -> Per element: 8B (for int64)
  -> Cache efficiency: extremely high
```

---

## 4. Strings

### 4.1 The Nature of Strings

A string is an "array of characters." However, the definition of "character" varies depending on the encoding,
making string processing far more complex than it appears.

### 4.2 Encodings: UTF-8, UTF-16, UTF-32

Let us compare the three major encodings for representing Unicode characters:

```
Unicode encoding comparison:

  Character "A" (U+0041):
    UTF-8:  [0x41]                          — 1 byte
    UTF-16: [0x0041]                        — 2 bytes
    UTF-32: [0x00000041]                    — 4 bytes

  Character "あ" (U+3042):
    UTF-8:  [0xE3, 0x81, 0x82]             — 3 bytes
    UTF-16: [0x3042]                        — 2 bytes
    UTF-32: [0x00003042]                    — 4 bytes

  Character "𠀋" (U+2000B, CJK Extension B):
    UTF-8:  [0xF0, 0xA0, 0x80, 0x8B]       — 4 bytes
    UTF-16: [0xD840, 0xDC0B]               — 4 bytes (surrogate pair)
    UTF-32: [0x0002000B]                    — 4 bytes

  Emoji "👨‍👩‍👧‍👦" (family):
    -> Composed of 7 code points
    U+1F468 U+200D U+1F469 U+200D U+1F467 U+200D U+1F466
    UTF-8:  25 bytes
    UTF-16: 18 bytes (including surrogate pairs)
    UTF-32: 28 bytes
```

Characteristics of each encoding:

```
+-------------+------------+------------+------------------------+
| Property    | UTF-8      | UTF-16     | UTF-32                 |
+-------------+------------+------------+------------------------+
| Code unit   | 1 byte     | 2 bytes    | 4 bytes                |
| Variable    | 1-4 bytes  | 2 or 4B    | Fixed 4 bytes          |
| length      |            |            |                        |
| ASCII       | Yes        | No         | No                     |
| compatible  |            |            |                        |
| Random      | No         | No*        | Yes                    |
| access      | (variable) | (surrogate)| (fixed length)         |
| ASCII doc   | Smallest   | 2x         | 4x                     |
| size        |            |            |                        |
| Japanese    | 1.5x       | Smallest   | 2x                     |
| doc size    |            |            |                        |
| Primary     | Files      | Windows    | Internal               |
| use         | Network    | Java, JS   | processing (rare)      |
| Null safe   | Yes        | No**       | No**                   |
+-------------+------------+------------+------------------------+

* Not strictly random access due to surrogate pairs
** 0x00 may appear within strings
```

### 4.3 The Elegance of UTF-8 Design

UTF-8 is an encoding designed by Ken Thompson and Rob Pike in 1992,
with the following excellent properties:

```
UTF-8 byte patterns:

  Range              Bytes  Bit pattern
  -----------------------------------------------
  U+0000..U+007F    1      0xxxxxxx
  U+0080..U+07FF    2      110xxxxx 10xxxxxx
  U+0800..U+FFFF    3      1110xxxx 10xxxxxx 10xxxxxx
  U+10000..U+10FFFF 4      11110xxx 10xxxxxx 10xxxxxx 10xxxxxx

  Design elegance:
  1. Fully compatible with ASCII (U+0000-U+007F)
     -> Existing ASCII text is valid UTF-8 without modification
  2. The number of bytes can be determined from the leading byte alone
     -> 0xxxxxxx: 1 byte
     -> 110xxxxx: 2 bytes
     -> 1110xxxx: 3 bytes
     -> 11110xxx: 4 bytes
  3. Continuation bytes always start with 10xxxxxx
     -> Character boundaries can be identified from any position in the byte stream
     -> Self-synchronizing property
  4. Byte sort order = code point sort order
     -> memcmp can compare in Unicode order
  5. NULL byte (0x00) only represents U+0000
     -> C string functions work directly
```

### 4.4 String Immutability

In many languages, strings are designed as immutable (unmodifiable).

Benefits of immutable strings:
- Thread-safe: safely readable from multiple threads
- Hash caching: hash value can be computed once and reused
- Security: guarantees that string content will not change
- String interning: strings with the same content can be shared

Caveats of immutable strings:
- "Modifying" a string creates a new object -> O(n)
- Concatenation in loops easily becomes O(n^2)

```python
# Anti-pattern: string concatenation in a loop — O(n^2)
def build_string_bad(n):
    result = ""
    for i in range(n):
        result += str(i) + ","  # Creates a new string each time
    return result

# Correct pattern: use join — O(n)
def build_string_good(n):
    parts = [str(i) for i in range(n)]
    return ",".join(parts)

# Benchmark
import timeit
n = 50000
t1 = timeit.timeit(lambda: build_string_bad(n), number=5)
t2 = timeit.timeit(lambda: build_string_good(n), number=5)
print(f"Concat (+=): {t1:.3f}s")
print(f"join:        {t2:.3f}s")
print(f"Ratio:       {t1/t2:.1f}x")
# Typical result: join is 10x+ faster than += (n=50000)
```

### 4.5 String Interning

String interning is an optimization technique that shares a single instance
when multiple strings with the same content exist.

```python
# CPython string interning
a = "hello"
b = "hello"
print(a is b)       # True — sharing the same object

# Conditions for compile-time interning in CPython:
# - Strings valid as identifiers (alphanumeric and underscore only)
# - Length below a certain threshold

c = "hello world"
d = "hello world"
print(c is d)       # True (CPython 3.x interns compile-time constants)

# Dynamically generated strings at runtime are usually not interned
e = "hello" + " " + "world"
f = "hello" + " " + "world"
# Results may vary depending on CPython optimization level

# Explicit interning
import sys
g = sys.intern("hello world 123")
h = sys.intern("hello world 123")
print(g is h)       # True — always the same object

# In Java:
# String s1 = "hello";    // Retrieved from literal pool
# String s2 = "hello";    // Same reference
# s1 == s2                // true (reference comparison)
# new String("hello").intern() == s1  // true
```

---

## 5. String Algorithms

String search is the problem of finding pattern P within text T.
The naive approach is O(nm), but advanced algorithms can achieve O(n+m) or even O(n/m).

### 5.1 Naive String Search

```python
def naive_search(text: str, pattern: str) -> list[int]:
    """Naive string search — O(nm)"""
    n, m = len(text), len(pattern)
    positions = []
    for i in range(n - m + 1):
        if text[i:i+m] == pattern:
            positions.append(i)
    return positions

# Test
text = "AABAACAADAABAABA"
pattern = "AABA"
print(naive_search(text, pattern))  # [0, 9, 12]
```

Worst case: patterns like `text = "AAAAAAAAB"`, `pattern = "AAAAB"`,
where there is near-match but mismatch at the pattern end, result in O(nm).

### 5.2 KMP (Knuth-Morris-Pratt) Algorithm

The KMP algorithm precomputes prefix-suffix match information for the pattern and
efficiently shifts the pattern when a mismatch occurs, achieving O(n+m).

```python
def kmp_search(text: str, pattern: str) -> list[int]:
    """KMP algorithm — O(n + m)"""
    n, m = len(text), len(pattern)
    if m == 0:
        return []

    # Build the failure function — O(m)
    # fail[i] = length of the longest proper prefix of pattern[0:i+1]
    #           that is also a suffix
    fail = [0] * m
    j = 0
    for i in range(1, m):
        while j > 0 and pattern[i] != pattern[j]:
            j = fail[j - 1]
        if pattern[i] == pattern[j]:
            j += 1
        fail[i] = j

    # Search — O(n)
    positions = []
    j = 0
    for i in range(n):
        while j > 0 and text[i] != pattern[j]:
            j = fail[j - 1]
        if text[i] == pattern[j]:
            j += 1
        if j == m:
            positions.append(i - m + 1)
            j = fail[j - 1]

    return positions

# Test
text = "AABAACAADAABAABA"
pattern = "AABA"
print(kmp_search(text, pattern))  # [0, 9, 12]

# Failure function example:
# pattern = "AABA"
# fail = [0, 1, 0, 1]
#
# A  A  B  A
# 0  1  0  1
# ^  ^  ^  ^
# -  A  -  A   <- longest prefix = suffix match
```

### 5.3 Rabin-Karp Algorithm

Uses rolling hash to detect patterns by hash value matching.
Average O(n+m), worst case O(nm), but excels at simultaneous multi-pattern search.

```python
def rabin_karp_search(text: str, pattern: str, base: int = 256,
                      mod: int = 101) -> list[int]:
    """Rabin-Karp algorithm — average O(n + m)"""
    n, m = len(text), len(pattern)
    if m > n:
        return []

    positions = []

    # Compute pattern hash value
    p_hash = 0
    t_hash = 0
    h = pow(base, m - 1, mod)  # base^(m-1) mod mod

    for i in range(m):
        p_hash = (p_hash * base + ord(pattern[i])) % mod
        t_hash = (t_hash * base + ord(text[i])) % mod

    # Slide and compare
    for i in range(n - m + 1):
        if p_hash == t_hash:
            # Hash match -> actual comparison (eliminate false positives)
            if text[i:i+m] == pattern:
                positions.append(i)

        # Rolling hash update
        if i < n - m:
            t_hash = (t_hash - ord(text[i]) * h) * base + ord(text[i + m])
            t_hash %= mod

    return positions

# Test
text = "AABAACAADAABAABA"
pattern = "AABA"
print(rabin_karp_search(text, pattern))  # [0, 9, 12]
```

### 5.4 Boyer-Moore Algorithm (Simplified)

Boyer-Moore compares from the pattern end and uses mismatch character information
to skip large distances, achieving practical performance close to O(n/m).
This simplified version implements only the Bad Character rule.

```python
def boyer_moore_search(text: str, pattern: str) -> list[int]:
    """Boyer-Moore (Bad Character rule only) — practically O(n/m)"""
    n, m = len(text), len(pattern)
    if m == 0 or m > n:
        return []

    # Bad Character table: last occurrence of each character in the pattern
    bad_char = {}
    for i in range(m):
        bad_char[pattern[i]] = i

    positions = []
    s = 0  # Shift amount on the text
    while s <= n - m:
        j = m - 1  # Compare from pattern end

        while j >= 0 and pattern[j] == text[s + j]:
            j -= 1

        if j < 0:
            # Full match
            positions.append(s)
            s += 1  # Could skip more with the Good Suffix rule
        else:
            # Mismatch
            char = text[s + j]
            skip = j - bad_char.get(char, -1)
            s += max(1, skip)

    return positions

# Test
text = "AABAACAADAABAABA"
pattern = "AABA"
print(boyer_moore_search(text, pattern))  # [0, 9, 12]
```

### 5.5 String Search Algorithm Comparison

```
+--------------+------------+------------+--------------------------+
| Algorithm    | Best       | Worst      | Characteristics          |
+--------------+------------+------------+--------------------------+
| Naive        | O(n)       | O(nm)      | Sufficient for short     |
|              |            |            | patterns                 |
| KMP          | O(n+m)     | O(n+m)     | Worst-case guaranteed.   |
|              |            |            | Good for streaming       |
| Rabin-Karp   | O(n+m)     | O(nm)      | Multi-pattern search.    |
|              |            |            | Plagiarism detection     |
| Boyer-Moore  | O(n/m)     | O(nm)*     | Fastest in practice.     |
|              |            |            | Especially effective for |
|              |            |            | long patterns            |
| Aho-Corasick | O(n+m+z)   | O(n+m+z)   | Multiple patterns.       |
|              |            |            | Dictionary search.       |
|              |            |            | z = occurrence count     |
+--------------+------------+------------+--------------------------+

* Boyer-Moore worst case can be improved to O(n+m) with Galil's improvement
```

---

## 6. Classic Array Techniques

### 6.1 Binary Search

The most important algorithm for sorted arrays. Can search elements in O(log n).

```python
def binary_search(arr: list[int], target: int) -> int:
    """Binary search — O(log n)
    Returns the index if found, -1 otherwise
    """
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2  # Overflow prevention
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

# lower_bound: smallest index where value >= target
def lower_bound(arr: list[int], target: int) -> int:
    """Equivalent to C++ std::lower_bound — O(log n)"""
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo

# upper_bound: smallest index where value > target
def upper_bound(arr: list[int], target: int) -> int:
    """Equivalent to C++ std::upper_bound — O(log n)"""
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if arr[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    return lo

# Test
arr = [1, 3, 3, 3, 5, 7, 9]
print(binary_search(arr, 3))    # 2 (first found)
print(lower_bound(arr, 3))      # 1 (smallest index >= 3)
print(upper_bound(arr, 3))      # 4 (smallest index > 3)
print(lower_bound(arr, 4))      # 4 (smallest >= 4 -> arr[4]=5)
```

**Key points to avoid off-by-one errors in binary search:**

1. `lo <= hi` vs `lo < hi`: depends on whether the search range is [lo, hi] or [lo, hi)
2. `mid = lo + (hi - lo) // 2`: `(lo + hi) // 2` risks overflow
3. Clarify whether you need lower_bound or upper_bound

### 6.2 Two Pointers

A technique using two pointers on a sorted array to find pairs satisfying a condition.

```python
def two_sum_sorted(arr: list[int], target: int) -> tuple[int, int] | None:
    """Find a pair in a sorted array that sums to target — O(n)"""
    left, right = 0, len(arr) - 1
    while left < right:
        s = arr[left] + arr[right]
        if s == target:
            return (left, right)
        elif s < target:
            left += 1
        else:
            right -= 1
    return None

# Test
arr = [1, 2, 4, 6, 8, 10]
print(two_sum_sorted(arr, 10))  # (1, 4) -> arr[1]+arr[4] = 2+8 = 10

# Application: 3Sum (enumerate triplets that sum to 0)
def three_sum(arr: list[int]) -> list[tuple[int, int, int]]:
    """Enumerate combinations of three elements summing to 0 — O(n^2)"""
    arr.sort()
    n = len(arr)
    result = []
    for i in range(n - 2):
        if i > 0 and arr[i] == arr[i - 1]:
            continue  # Skip duplicates
        left, right = i + 1, n - 1
        while left < right:
            s = arr[i] + arr[left] + arr[right]
            if s == 0:
                result.append((arr[i], arr[left], arr[right]))
                while left < right and arr[left] == arr[left + 1]:
                    left += 1  # Skip duplicates
                while left < right and arr[right] == arr[right - 1]:
                    right -= 1  # Skip duplicates
                left += 1
                right -= 1
            elif s < 0:
                left += 1
            else:
                right -= 1
    return result

print(three_sum([-1, 0, 1, 2, -1, -4]))
# [(-1, -1, 2), (-1, 0, 1)]
```

### 6.3 Sliding Window

A technique that slides a fixed-length or variable-length window over an array.

```python
# Fixed-length window: maximum sum of a contiguous subarray of length k
def max_sum_subarray(arr: list[int], k: int) -> int:
    """Fixed-length sliding window — O(n)"""
    n = len(arr)
    if n < k:
        return 0
    window_sum = sum(arr[:k])
    best = window_sum
    for i in range(k, n):
        window_sum += arr[i] - arr[i - k]
        best = max(best, window_sum)
    return best

print(max_sum_subarray([1, 4, 2, 10, 2, 3, 1, 0, 20], 4))  # 24

# Variable-length window: shortest subarray with sum >= target
def min_subarray_len(arr: list[int], target: int) -> int:
    """Variable-length sliding window — O(n)"""
    n = len(arr)
    left = 0
    current_sum = 0
    min_len = float('inf')
    for right in range(n):
        current_sum += arr[right]
        while current_sum >= target:
            min_len = min(min_len, right - left + 1)
            current_sum -= arr[left]
            left += 1
    return min_len if min_len != float('inf') else 0

print(min_subarray_len([2, 3, 1, 2, 4, 3], 7))  # 2 (subarray [4, 3])

# Variable-length window: longest substring without repeating characters
def longest_unique_substring(s: str) -> int:
    """Length of the longest substring without repeating characters — O(n)"""
    seen = {}
    left = 0
    max_len = 0
    for right, char in enumerate(s):
        if char in seen and seen[char] >= left:
            left = seen[char] + 1
        seen[char] = right
        max_len = max(max_len, right - left + 1)
    return max_len

print(longest_unique_substring("abcabcbb"))  # 3 ("abc")
print(longest_unique_substring("pwwkew"))    # 3 ("wke")
```

### 6.4 Prefix Sum (Cumulative Sum)

A preprocessing technique to answer range sum queries in O(1).

```python
class PrefixSum:
    """Prefix sum — build O(n), query O(1)"""

    def __init__(self, arr: list[int]):
        n = len(arr)
        self.prefix = [0] * (n + 1)
        for i in range(n):
            self.prefix[i + 1] = self.prefix[i] + arr[i]

    def range_sum(self, left: int, right: int) -> int:
        """Return sum of arr[left:right+1] in O(1)"""
        return self.prefix[right + 1] - self.prefix[left]

# Test
arr = [3, 1, 4, 1, 5, 9, 2, 6]
ps = PrefixSum(arr)
print(ps.range_sum(2, 5))  # arr[2]+arr[3]+arr[4]+arr[5] = 4+1+5+9 = 19
print(ps.range_sum(0, 7))  # Total sum = 31

# 2D Prefix Sum
class PrefixSum2D:
    """2D prefix sum — build O(nm), query O(1)"""

    def __init__(self, matrix: list[list[int]]):
        if not matrix:
            self.prefix = [[]]
            return
        rows, cols = len(matrix), len(matrix[0])
        self.prefix = [[0] * (cols + 1) for _ in range(rows + 1)]
        for i in range(rows):
            for j in range(cols):
                self.prefix[i+1][j+1] = (matrix[i][j]
                    + self.prefix[i][j+1]
                    + self.prefix[i+1][j]
                    - self.prefix[i][j])

    def range_sum(self, r1: int, c1: int, r2: int, c2: int) -> int:
        """Return sum of matrix[r1:r2+1][c1:c2+1] in O(1)"""
        return (self.prefix[r2+1][c2+1]
                - self.prefix[r1][c2+1]
                - self.prefix[r2+1][c1]
                + self.prefix[r1][c1])

# Test
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
ps2d = PrefixSum2D(matrix)
print(ps2d.range_sum(0, 0, 1, 1))  # 1+2+4+5 = 12
print(ps2d.range_sum(1, 1, 2, 2))  # 5+6+8+9 = 28
```

---

## 7. Language-Specific Implementations

### 7.1 CPython list

CPython's list is implemented in `Objects/listobject.c`.

```
CPython list main data structure:

  typedef struct {
      PyObject_VAR_HEAD          // ob_refcnt, ob_type, ob_size
      PyObject **ob_item;        // Pointer to the pointer array
      Py_ssize_t allocated;      // Number of allocated slots
  } PyListObject;

  ob_size    = current number of elements (len)
  allocated  = number of allocated slots (capacity)

  Growth policy (listobject.c, list_resize):
  new_allocated = new_size + (new_size >> 3) + (new_size < 9 ? 3 : 6)

  Specific allocated transitions:
  len=0 -> allocated=0
  len=1 -> allocated=4     (0 + 0 + 3 + 1 = 4)
  len=5 -> allocated=8     (5 + 0 + 3 = 8)
  len=9 -> allocated=16    (9 + 1 + 6 = 16)
  len=17 -> allocated=25   (17 + 2 + 6 = 25)
  len=26 -> allocated=35   (26 + 3 + 6 = 35)
  ...

  Characteristics:
  - Saves extra memory for small lists
  - Grows by about 12.5% for large lists
  - More memory-efficient than doubling (x2), but slightly more frequent resizes
```

### 7.2 Java ArrayList

```
Java ArrayList<E> internal structure:

  public class ArrayList<E> {
      private Object[] elementData;   // Element array
      private int size;               // Current number of elements

      // Default initial capacity
      private static final int DEFAULT_CAPACITY = 10;
  }

  Growth policy:
  private void grow(int minCapacity) {
      int oldCapacity = elementData.length;
      int newCapacity = oldCapacity + (oldCapacity >> 1);  // 1.5x
      // ...
  }

  Allocated transitions:
  10 -> 15 -> 22 -> 33 -> 49 -> 73 -> 109 -> ...

  Characteristics:
  - Default initial capacity 10
  - Growth factor 1.5 (50% increase)
  - trimToSize() can release excess capacity
  - ensureCapacity() can pre-allocate capacity
  - Type-safe via generics
  - However, primitive types cannot be stored directly (boxing required)
    -> Use raw arrays when int[] is needed
```

### 7.3 Rust Vec\<T>

```
Rust Vec<T> internal structure:

  pub struct Vec<T> {
      ptr: NonNull<T>,    // Pointer to the heap array
      len: usize,         // Current number of elements
      cap: usize,         // Allocated element count
  }

  // Vec memory layout (24 bytes on the stack)
  // +------+------+------+
  // | ptr  | len  | cap  |  <- On the stack (8 bytes each)
  // +--+---+------+------+
  //    |
  //    v Heap
  //    +-----+-----+-----+-----+-----+-----+
  //    | T   | T   | T   | T   |     |     |
  //    +-----+-----+-----+-----+-----+-----+
  //    <- len=4 ->  <-- unused -->
  //    <-------- cap=6 -------->

  Growth policy:
  - Growth factor: 2.0
  - Zero-cost abstraction: Vec implicitly converts to &[T] (slice)
  - Drop trait: automatically freed when going out of scope

  Characteristics:
  - Ownership system prevents dangling pointers
  - No buffer overflows without unsafe
  - Iterators optimized to the same speed as for loops
  - Vec<u8> can be used as a string byte sequence
```

### 7.4 Go slice

```
Go slice internal structure:

  type slice struct {
      array unsafe.Pointer  // Pointer to the underlying array
      len   int             // Current number of elements
      cap   int             // Capacity
  }

  // Slice 3-element structure:
  // +--------+------+------+
  // | array  | len  | cap  |
  // +---+----+------+------+
  //     |
  //     v
  //     +---+---+---+---+---+---+
  //     | 1 | 2 | 3 |   |   |   |
  //     +---+---+---+---+---+---+
  //     <- len=3 ->
  //     <---- cap=6 ---->

  Growth policy (Go 1.18+):
  - cap < 256:      x2 (doubling)
  - cap >= 256:     x(1.25 + 192/cap)  (gradually approaches 1.25x)

  Characteristics:
  - Slices are reference types (copies share the underlying array)
  - append returns a new slice (new array allocated when cap exceeded)
  - re-slicing: s[1:3] does not create a copy
  - make([]int, len, cap) specifies length and capacity separately
```

---

## 8. Memory Alignment and SIMD

### 8.1 Memory Alignment

CPUs efficiently handle memory accesses from addresses aligned to specific byte boundaries.

```
Memory alignment example:

  struct Example {
      char   a;    // 1 byte
      int    b;    // 4 bytes
      char   c;    // 1 byte
      double d;    // 8 bytes
  };

  Without padding (theoretical):
  Offset: 0  1  2  3  4  5  6  7  8  9  10 11 12 13
          [a][b  b  b  b][c][d  d  d  d  d  d  d  d]
          Total: 14 bytes

  With padding (actual):
  Offset: 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
          [a][..padding..][b  b  b  b][c][..padding........]
  Offset: 16 17 18 19 20 21 22 23
          [d  d  d  d  d  d  d  d]
          Total: 24 bytes (14 + 10 bytes of padding)

  Optimized field ordering:
  struct ExampleOptimized {
      double d;    // 8 bytes (offset 0)
      int    b;    // 4 bytes (offset 8)
      char   a;    // 1 byte  (offset 12)
      char   c;    // 1 byte  (offset 13)
      // 2 bytes of padding
  };
  Total: 16 bytes (14 + 2 bytes of padding)

  -> Just reordering fields reduces 24B -> 16B
  -> 33% memory savings when holding large numbers of structs in arrays
```

### 8.2 SIMD (Single Instruction, Multiple Data)

SIMD is a CPU feature that processes multiple data elements with a single instruction.
Array contiguity is a prerequisite for SIMD utilization.

```
SIMD concept:

  Normal addition (scalar):
  a[0] + b[0] -> c[0]     1st instruction
  a[1] + b[1] -> c[1]     2nd instruction
  a[2] + b[2] -> c[2]     3rd instruction
  a[3] + b[3] -> c[3]     4th instruction
  -> 4 instructions needed

  SIMD addition (vector):
  +---------+---------+---------+---------+
  | a[0]    | a[1]    | a[2]    | a[3]    |  128-bit register
  +----+----+----+----+----+----+----+----+
       +         +         +         +       <- 1 instruction
  +----+----+----+----+----+----+----+----+
  | b[0]    | b[1]    | b[2]    | b[3]    |  128-bit register
  +----+----+----+----+----+----+----+----+
       =         =         =         =
  +----+----+----+----+----+----+----+----+
  | c[0]    | c[1]    | c[2]    | c[3]    |  Result
  +---------+---------+---------+---------+
  -> Completed in 1 instruction (theoretically 4x faster)

  Major SIMD instruction sets:
  +-------------+----------+----------------------+
  | Name        | Bit Width| Supported Processors |
  +-------------+----------+----------------------+
  | SSE         | 128-bit  | x86 (2000~)          |
  | AVX2        | 256-bit  | x86 (2013~)          |
  | AVX-512     | 512-bit  | x86 (2017~)          |
  | NEON        | 128-bit  | ARM                  |
  | SVE/SVE2    | Variable | ARM (2020~)          |
  +-------------+----------+----------------------+

  Simultaneous elements for float (32-bit):
  SSE:     128 / 32 =  4 elements
  AVX2:    256 / 32 =  8 elements
  AVX-512: 512 / 32 = 16 elements
```

NumPy and BLAS libraries internally leverage SIMD,
making numerical operations on contiguous arrays extremely fast.

```python
import numpy as np
import timeit

n = 10_000_000

# Sum with Python list
py_list = list(range(n))
t1 = timeit.timeit(lambda: sum(py_list), number=10)

# Sum with NumPy array (SIMD utilized)
np_arr = np.arange(n, dtype=np.int64)
t2 = timeit.timeit(lambda: np_arr.sum(), number=10)

print(f"Python sum : {t1:.4f}s")
print(f"NumPy sum  : {t2:.4f}s")
print(f"Speed ratio: {t1/t2:.0f}x")
# Typical result: NumPy is 50-100x faster than Python
```

---

## 9. Trade-offs and Comparative Analysis

### 9.1 Array vs Linked List

```
+------------------+--------------+--------------+------------------+
| Operation        | Array        | Linked List  | Winner           |
|                  | (dynamic)    |              |                  |
+------------------+--------------+--------------+------------------+
| Index access     | O(1)         | O(n)         | Array            |
+------------------+--------------+--------------+------------------+
| Insert at head   | O(n)         | O(1)         | Linked list      |
+------------------+--------------+--------------+------------------+
| Insert at tail   | O(1) amort.  | O(1)*        | Tie              |
|                  |              | *w/ tail ref  |                  |
+------------------+--------------+--------------+------------------+
| Insert in middle | O(n)         | O(1)**       | Linked list      |
|                  |              | **pos. known  |                  |
+------------------+--------------+--------------+------------------+
| Search           | O(n)         | O(n)         | Array            |
| (unsorted)       | cache-       | cache-       | (better constant)|
|                  | efficient    | inefficient  |                  |
+------------------+--------------+--------------+------------------+
| Search           | O(log n)     | O(n)         | Array            |
| (sorted)         | binary search|              |                  |
+------------------+--------------+--------------+------------------+
| Memory usage     | Elements     | Elements +   | Array            |
|                  | (+ excess    | pointers     |                  |
|                  | capacity)    | (8-16B/node) |                  |
+------------------+--------------+--------------+------------------+
| Cache efficiency | Extremely    | Low          | Array            |
|                  | high         |              |                  |
+------------------+--------------+--------------+------------------+
| Memory           | None         | Possible     | Array            |
| fragmentation    |              |              |                  |
+------------------+--------------+--------------+------------------+

Conclusion: arrays (dynamic arrays) are superior in most use cases.
Linked lists are advantageous when:
1. Frequent insertion/deletion at the head (deque, stack)
2. Moving elements is expensive (large structs)
3. Memory fragmentation is acceptable and real-time guarantees are needed
   (array resize incurs a one-time O(n) copy)
```

### 9.2 String Encoding Comparison (Practical Perspective)

```
+-------------+--------------+--------------+------------------+
| Criterion   | UTF-8        | UTF-16       | UTF-32           |
+-------------+--------------+--------------+------------------+
| ASCII doc   | 1B/char best | 2B/char      | 4B/char          |
| size        |              |              |                  |
+-------------+--------------+--------------+------------------+
| Japanese    | 3B/char      | 2B/char best | 4B/char          |
| doc size    |              |              |                  |
+-------------+--------------+--------------+------------------+
| Emoji       | 4B/char      | 4B/char      | 4B/char          |
| size        |              | (surrogate)  |                  |
+-------------+--------------+--------------+------------------+
| Random      | No (variable)| Mostly       | Yes (fixed)      |
| access      |              | (nearly      |                  |
|             |              | fixed)       |                  |
+-------------+--------------+--------------+------------------+
| ASCII       | Yes          | No           | No               |
| compatible  |              |              |                  |
+-------------+--------------+--------------+------------------+
| Byte Order  | Not needed   | Required     | Required (BOM)   |
| Mark (BOM)  |              | (BOM)        |                  |
+-------------+--------------+--------------+------------------+
| Network     | Standard     | Rare         | Very rare        |
+-------------+--------------+--------------+------------------+
| File        | Standard     | Windows      | Almost never     |
| storage     |              |              |                  |
+-------------+--------------+--------------+------------------+
| Primary     | Linux, Web   | Windows      | Internal         |
| adoption    | macOS, Go    | Java, JS     | processing only  |
|             | Rust, Python | C# (.NET)    |                  |
+-------------+--------------+--------------+------------------+

Practical recommendations:
- External data exchange: UTF-8 exclusively
- Windows API: UTF-16 (WideChar)
- Internal processing: follow the language's default
- Performance-critical: UTF-8 + byte-level processing
```

### 9.3 Dynamic Array Comparison Across Languages

```
+--------------+-------------+-----------+----------+------------+
| Property     | Python list | Java      | C++      | Rust Vec   |
|              |             | ArrayList | vector   |            |
+--------------+-------------+-----------+----------+------------+
| Element type | Any         | Object    | Any type | Any type   |
|              | (PyObject*) | (ref type)| (template| (generics) |
|              |             |           | )        |            |
+--------------+-------------+-----------+----------+------------+
| Contiguity   | Pointers    | References| Values   | Values     |
|              | only        | only      | contig.  | contig.    |
+--------------+-------------+-----------+----------+------------+
| Growth factor| ~1.125      | 1.5       | 2 (GCC)  | 2          |
+--------------+-------------+-----------+----------+------------+
| Initial cap  | 0           | 10        | 0        | 0          |
+--------------+-------------+-----------+----------+------------+
| Bounds check | Yes         | Yes       | No*      | Yes        |
|              |             |           | *at()    |            |
|              |             |           | available|            |
+--------------+-------------+-----------+----------+------------+
| Null safety  | N/A         | Elements  | N/A      | Option<T>  |
|              | (None ok)   | can be    |          | for safety |
|              |             | null      |          |            |
+--------------+-------------+-----------+----------+------------+
| Thread safety| GIL         | No*       | No       | Ownership  |
|              |             |           |          | compile-   |
|              |             |           |          | time       |
|              |             |           |          | guarantee  |
+--------------+-------------+-----------+----------+------------+
| Memory mgmt  | GC (ref     | GC        | Manual/  | Ownership  |
|              | counting)   |           | RAII     | (Drop)     |
+--------------+-------------+-----------+----------+------------+

* Java: Use Collections.synchronizedList() or CopyOnWriteArrayList
```

---

## 10. Anti-Patterns

### Anti-Pattern 1: String Concatenation in Loops

**Problem:** Concatenating immutable strings with `+=` in a loop creates
a new string every time. For n concatenations, the cost is O(n^2).

```python
# Bad: O(n^2) — copies the entire string each time
def build_csv_bad(rows: list[list[str]]) -> str:
    result = ""
    for row in rows:
        result += ",".join(row) + "\n"  # Creates a new string each time
    return result

# Good: O(n) — accumulate in a list and join at the end
def build_csv_good(rows: list[list[str]]) -> str:
    lines = []
    for row in rows:
        lines.append(",".join(row))
    return "\n".join(lines)

# Even better: generator expression
def build_csv_best(rows: list[list[str]]) -> str:
    return "\n".join(",".join(row) for row in rows)
```

**Scale of impact:**
- n=1,000: Bad version is about 2x slower
- n=10,000: Bad version is about 20x slower
- n=100,000: Bad version is about 200x slower

Same issue in Java:

```java
// Bad: O(n^2)
String result = "";
for (String s : list) {
    result += s;  // Creates a new String object each time
}

// Good: O(n)
StringBuilder sb = new StringBuilder();
for (String s : list) {
    sb.append(s);
}
String result = sb.toString();
```

### Anti-Pattern 2: Frequent Insertion/Deletion at the Array Head

**Problem:** Inserting/deleting at the beginning of an array requires shifting all elements.
When repeated, this becomes O(n^2).

```python
# Bad: O(n^2) — shifts all elements each time
def build_reversed_bad(items: list) -> list:
    result = []
    for item in items:
        result.insert(0, item)  # O(n) x n times = O(n^2)
    return result

# Good: O(n) — append to end and reverse
def build_reversed_good(items: list) -> list:
    result = []
    for item in items:
        result.append(item)  # O(1) amortized
    result.reverse()  # O(n) — only once
    return result

# Even better: use collections.deque
from collections import deque
def build_reversed_deque(items: list) -> deque:
    result = deque()
    for item in items:
        result.appendleft(item)  # O(1)
    return result

# Benchmark
import timeit
n = 50000
items = list(range(n))
t1 = timeit.timeit(lambda: build_reversed_bad(items), number=5)
t2 = timeit.timeit(lambda: build_reversed_good(items), number=5)
t3 = timeit.timeit(lambda: build_reversed_deque(items), number=5)
print(f"insert(0,x): {t1:.3f}s")
print(f"append+rev : {t2:.3f}s")
print(f"deque      : {t3:.3f}s")
```

### Anti-Pattern 3: Unnecessary Copy Creation

```python
# Bad: slicing creates a new list (unintentional copy)
def process_subarray_bad(arr: list, start: int, end: int) -> int:
    sub = arr[start:end]  # O(end-start) copy occurs
    return sum(sub)

# Good: process by specifying index range
def process_subarray_good(arr: list, start: int, end: int) -> int:
    total = 0
    for i in range(start, end):
        total += arr[i]  # No copy
    return total

# NumPy: views vs copies
import numpy as np
arr = np.array([1, 2, 3, 4, 5])

view = arr[1:4]      # View (no copy, shares underlying array)
view[0] = 99         # arr is also modified!
print(arr)            # [1, 99, 3, 4, 5]

copy = arr[1:4].copy()  # Explicit copy
copy[0] = 0           # arr is not modified
print(arr)            # [1, 99, 3, 4, 5]
```

---

## 11. Edge Case Analysis

### Edge Case 1: Empty Arrays and Single-Element Arrays

The most commonly overlooked cases in array operations are empty arrays and single-element arrays.

```python
def find_max_bad(arr: list[int]) -> int:
    """Bad: exception on empty array"""
    max_val = arr[0]  # IndexError if arr is empty!
    for x in arr[1:]:
        max_val = max(max_val, x)
    return max_val

def find_max_good(arr: list[int]) -> int | None:
    """Good: properly handles empty arrays"""
    if not arr:
        return None
    max_val = arr[0]
    for x in arr[1:]:
        max_val = max(max_val, x)
    return max_val

# Test
print(find_max_good([]))         # None
print(find_max_good([42]))       # 42
print(find_max_good([3, 1, 4]))  # 4

# Binary search edge case
def binary_search_safe(arr: list[int], target: int) -> int:
    """Safe binary search even with empty arrays"""
    if not arr:
        return -1
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

# Palindrome check edge case
def is_palindrome(s: str) -> bool:
    """Empty strings and single characters are palindromes"""
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True  # Empty string and single character return True

assert is_palindrome("") == True
assert is_palindrome("a") == True
assert is_palindrome("ab") == False
assert is_palindrome("aba") == True
```

### Edge Case 2: Integer Overflow

In many languages, integer overflow can occur in array index calculations or element summation.

```c
/* Integer overflow in binary search in C */

/* Bad: lo + hi may overflow */
int mid_bad = (lo + hi) / 2;
/* If lo = 2,000,000,000 and hi = 2,000,000,000
   lo + hi = 4,000,000,000 > INT_MAX (2,147,483,647)
   -> Becomes negative, leading to an invalid index */

/* Good: overflow-safe calculation */
int mid_good = lo + (hi - lo) / 2;
/* hi - lo is always non-negative, and lo + (hi-lo)/2 is between lo and hi */

/* The same problem exists in Java:
   java.util.Arrays.binarySearch() was fixed in JDK 6
   to (lo + hi) >>> 1
   (unsigned right shift to avoid overflow) */
```

```python
# Python has arbitrary-precision integers so overflow does not occur,
# but array sums can strain memory

# Sum of a huge array
import sys
huge_list = [10**100] * 1000000
total = sum(huge_list)
# Python's int is arbitrary-precision so no overflow,
# but each element as a PyObject consumes significant memory

# NumPy can overflow
import numpy as np
arr = np.array([2**62, 2**62], dtype=np.int64)
print(arr.sum())  # Overflow! -> becomes negative
# Workaround: no np.int128, so convert to Python int
print(sum(int(x) for x in arr))  # Correct result
```

### Edge Case 3: Unicode String "Character Count"

```python
# "Character count" depends on what you mean by "character"

text = "👨‍👩‍👧‍👦"  # Family emoji

# Byte count (UTF-8)
print(len(text.encode('utf-8')))    # 25

# Code point count
print(len(text))                     # 7 (Python counts Unicode code points)

# Grapheme cluster count — the number of "characters" humans perceive
# Difficult with the standard library alone.
# pip install grapheme
# import grapheme
# print(grapheme.length(text))       # 1

# Japanese example
text_ja = "が"  # U+304C (が) — 1 code point
text_ja2 = "が"  # U+304B (か) + U+3099 (dakuten) — 2 code points
# They look the same but len() may differ

# Normalization
import unicodedata
nfc = unicodedata.normalize('NFC', text_ja2)   # Composed form: 1 code point
nfd = unicodedata.normalize('NFD', text_ja)    # Decomposed form: 2 code points
print(len(nfc), len(nfd))  # 1, 2

# Safe string comparison: compare after normalization
def safe_compare(s1: str, s2: str) -> bool:
    return unicodedata.normalize('NFC', s1) == unicodedata.normalize('NFC', s2)
```

---

## 12. Practice Exercises

### Exercise 1: Basic Level

**Problem 1-1: Array Rotation**

Implement a function that rotates an array to the right by k positions.
Example: `[1,2,3,4,5,6,7]`, k=3 -> `[5,6,7,1,2,3,4]`

Hint: can be achieved with three reversals.

```python
def rotate_array(arr: list, k: int) -> None:
    """Rotate array in-place by k positions to the right — O(n) time, O(1) space"""
    n = len(arr)
    if n == 0:
        return
    k = k % n  # Handle k > n

    def reverse(start: int, end: int) -> None:
        while start < end:
            arr[start], arr[end] = arr[end], arr[start]
            start += 1
            end -= 1

    # Step 1: Reverse entire array  [1,2,3,4,5,6,7] -> [7,6,5,4,3,2,1]
    reverse(0, n - 1)
    # Step 2: Reverse first half    [7,6,5,4,3,2,1] -> [5,6,7,4,3,2,1]
    reverse(0, k - 1)
    # Step 3: Reverse second half   [5,6,7,4,3,2,1] -> [5,6,7,1,2,3,4]
    reverse(k, n - 1)

# Test
arr = [1, 2, 3, 4, 5, 6, 7]
rotate_array(arr, 3)
print(arr)  # [5, 6, 7, 1, 2, 3, 4]
```

**Problem 1-2: Palindrome Check**

Implement a function that checks if a string is a palindrome, ignoring non-alphanumeric characters and case.

```python
def is_palindrome_alnum(s: str) -> bool:
    """Alphanumeric-only palindrome check — O(n) time, O(1) space"""
    left, right = 0, len(s) - 1
    while left < right:
        # Skip non-alphanumeric characters
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        if s[left].lower() != s[right].lower():
            return False
        left += 1
        right -= 1
    return True

# Test
print(is_palindrome_alnum("A man, a plan, a canal: Panama"))  # True
print(is_palindrome_alnum("race a car"))                       # False
print(is_palindrome_alnum(""))                                 # True
```

### Exercise 2: Applied Level

**Problem 2-1: Longest Palindromic Substring**

Find the longest palindromic substring within a given string.

```python
def longest_palindrome_substring(s: str) -> str:
    """Longest palindromic substring — O(n^2) time, O(1) space
    (O(n) is possible with Manacher's algorithm)
    """
    if not s:
        return ""

    start, max_len = 0, 1

    def expand_around_center(left: int, right: int) -> tuple[int, int]:
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return left + 1, right - left - 1

    for i in range(len(s)):
        # Odd-length palindrome
        l1, len1 = expand_around_center(i, i)
        if len1 > max_len:
            start, max_len = l1, len1

        # Even-length palindrome
        if i + 1 < len(s):
            l2, len2 = expand_around_center(i, i + 1)
            if len2 > max_len:
                start, max_len = l2, len2

    return s[start:start + max_len]

# Test
print(longest_palindrome_substring("babad"))     # "bab" or "aba"
print(longest_palindrome_substring("cbbd"))      # "bb"
print(longest_palindrome_substring("racecar"))   # "racecar"
```

**Problem 2-2: 90-Degree Matrix Rotation**

Implement a function that rotates an n x n matrix 90 degrees clockwise in-place.

```python
def rotate_matrix(matrix: list[list[int]]) -> None:
    """Rotate n x n matrix 90 degrees clockwise — O(n^2) time, O(1) space"""
    n = len(matrix)

    # Step 1: Transpose (swap rows and columns)
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

    # Step 2: Reverse each row
    for i in range(n):
        matrix[i].reverse()

# Test
matrix = [
    [1,  2,  3,  4],
    [5,  6,  7,  8],
    [9,  10, 11, 12],
    [13, 14, 15, 16]
]
rotate_matrix(matrix)
for row in matrix:
    print(row)
# [13, 9,  5, 1]
# [14, 10, 6, 2]
# [15, 11, 7, 3]
# [16, 12, 8, 4]
```

### Exercise 3: Advanced Level

**Problem 3-1: Spiral Order Matrix Output**

Implement a function that outputs the elements of an m x n matrix in spiral order.

```python
def spiral_order(matrix: list[list[int]]) -> list[int]:
    """Return matrix elements in spiral order — O(mn) time"""
    if not matrix or not matrix[0]:
        return []

    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1

    while top <= bottom and left <= right:
        # Top row: left -> right
        for j in range(left, right + 1):
            result.append(matrix[top][j])
        top += 1

        # Right column: top -> bottom
        for i in range(top, bottom + 1):
            result.append(matrix[i][right])
        right -= 1

        # Bottom row: right -> left
        if top <= bottom:
            for j in range(right, left - 1, -1):
                result.append(matrix[bottom][j])
            bottom -= 1

        # Left column: bottom -> top
        if left <= right:
            for i in range(bottom, top - 1, -1):
                result.append(matrix[i][left])
            left += 1

    return result

# Test
matrix = [
    [1,  2,  3,  4],
    [5,  6,  7,  8],
    [9,  10, 11, 12]
]
print(spiral_order(matrix))
# [1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]
```

**Problem 3-2: Minimum Window Substring**

Given strings s and t, find the minimum substring of s that contains all characters of t.

```python
from collections import Counter

def min_window(s: str, t: str) -> str:
    """Minimum window substring — O(|s| + |t|)"""
    if not s or not t:
        return ""

    need = Counter(t)
    missing = len(t)  # Number of characters still unmatched
    best_start, best_len = 0, float('inf')
    left = 0

    for right, char in enumerate(s):
        if need[char] > 0:
            missing -= 1
        need[char] -= 1

        # Once all characters are included, shrink the window
        while missing == 0:
            window_len = right - left + 1
            if window_len < best_len:
                best_start, best_len = left, window_len

            need[s[left]] += 1
            if need[s[left]] > 0:
                missing += 1
            left += 1

    return "" if best_len == float('inf') else s[best_start:best_start + best_len]

# Test
print(min_window("ADOBECODEBANC", "ABC"))  # "BANC"
print(min_window("a", "a"))                 # "a"
print(min_window("a", "aa"))                # "" (impossible)
```

---

## 13. FAQ

### Q1: How should Python's list and tuple be used differently?

**A:** They should be used with distinct semantic meanings.

- **list**: Variable-length collection of homogeneous elements. E.g., user list, score history
- **tuple**: Fixed-length record of heterogeneous elements. E.g., (name, age), (x-coord, y-coord)

Performance-wise:
- tuples are immutable and therefore hashable, usable as dictionary keys or set elements
- tuple creation is slightly faster than list (simpler memory allocation)
- CPython caches and reuses small tuples (free list)

```python
import sys
print(sys.getsizeof([1, 2, 3]))   # 120 (includes excess capacity)
print(sys.getsizeof((1, 2, 3)))   # 64  (minimum necessary)
```

### Q2: When should you use an array vs a hashmap (dictionary)?

**A:** Decide based on these criteria:

- **Use arrays when**: keys are consecutive integers starting from 0 and dense (most positions have values)
  - Example: counting 26 letters -> `counts = [0] * 26`
  - Example: DP table -> `dp = [0] * (n + 1)`
  - Benefits: O(1) access, high cache efficiency, minimal memory overhead

- **Use hashmaps when**: keys are sparse, non-integer, or have a wide range
  - Example: word frequency -> `Counter(words)`
  - Example: coordinates -> `grid = {(x, y): value}`
  - Benefits: arbitrary keys, dynamic insertion/deletion

```python
# Array is appropriate: frequency count of lowercase letters
def count_letters_array(s: str) -> list[int]:
    counts = [0] * 26
    for c in s:
        counts[ord(c) - ord('a')] += 1
    return counts

# Dictionary is appropriate: frequency count of arbitrary characters
from collections import Counter
def count_chars_dict(s: str) -> dict[str, int]:
    return Counter(s)
```

### Q3: What is the difference between `==` and `is` for string comparison?

**A:** `==` checks value equality, `is` checks object identity.
Always use `==` for string comparison.

```python
a = "hello"
b = "hello"
c = "hel" + "lo"

print(a == b)    # True  — same value
print(a is b)    # True  — CPython interns it (implementation-dependent!)

# Dynamically generated string
d = "".join(["h", "e", "l", "l", "o"])
print(a == d)    # True  — same value
print(a is d)    # False — different objects (not interned)

# Important: 'is' for string comparison is implementation-dependent; never use it
# PyPy, Jython, IronPython may produce different results from CPython
```

### Q4: Which sorting algorithm is used for arrays?

**A:** Sort implementations in major languages:

- **Python (Timsort)**: Hybrid of merge sort + insertion sort. O(n log n) worst case, O(n) best case (already sorted). Stable sort.
- **Java (Arrays.sort)**: Dual Pivot Quicksort for primitives, Timsort for objects.
- **C++ (std::sort)**: Introsort (quicksort + heapsort + insertion sort). Unstable sort. Use `std::stable_sort` when stability is needed.
- **Rust (sort)**: Timsort-based. Stable sort. `sort_unstable` is unstable but slightly faster.

### Q5: Why are strings immutable in many languages?

**A:** Immutable design is considered rational for the following reasons:

1. **Thread safety**: safely readable from multiple threads without locks
2. **Hash caching**: hash value, once computed, never changes and can be reused. Can serve as dictionary keys
3. **Security**: prevents unexpected modification of security-critical strings like file paths and URLs
4. **Optimization**: compilers can leverage immutability for optimizations (interning, constant propagation, etc.)
5. **API simplification**: no need for defensive copying since the string cannot be modified after passing

However, for cases requiring frequent modification, each language provides mutable string builders:
- Python: `io.StringIO`, `list` + `join`
- Java: `StringBuilder`, `StringBuffer`
- C#: `StringBuilder`
- Rust: `String` (Rust's String is mutable)

---


## FAQ

### Q1: What is the most important point in learning this topic?

Gaining practical experience is the most important thing. Understanding deepens when you not only study theory but also write and run actual code.

### Q2: What common mistakes do beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend firmly understanding the basic concepts explained in this guide before moving to the next step.

### Q3: How is this applied in practice?

Knowledge of this topic is frequently used in everyday development work, especially during code reviews and architecture design.

---

## 14. Summary

| Concept | Key Point |
|---------|-----------|
| Nature of arrays | Sequence of same-type elements in contiguous memory. O(1) random access |
| Cache efficiency | Arrays align with cache lines, delivering speeds beyond theoretical values |
| Static arrays | Size determined at compile time. Can be stack-allocated |
| Dynamic arrays | Amortized O(1) append via doubling strategy. Growth factors range from 1.125 to 2.0 |
| Multidimensional arrays | Row-major vs column-major. Traversal order directly impacts cache efficiency |
| Strings | Immutable in many languages. UTF-8 is the de facto standard |
| Encoding | UTF-8: ASCII-compatible/variable-length, UTF-16: Windows/Java, UTF-32: fixed-length |
| String search | KMP: worst-case O(n+m) guaranteed, Boyer-Moore: practically fastest O(n/m) |
| Classic techniques | Binary search, two pointers, sliding window, prefix sum |
| SIMD | Contiguous arrays are a prerequisite for SIMD. NumPy etc. use it internally |
| Anti-patterns | Loop string concatenation (O(n^2)), head insertion (O(n^2)), unnecessary copies |
| Edge cases | Empty arrays, integer overflow, Unicode normalization |

**Most important lesson:** Arrays are the simplest data structure, but that very simplicity
creates excellent compatibility with modern hardware, making them the optimal choice in many scenarios.
When choosing a data structure, consider not only Big-O notation but also the impact of caching.

---

## Recommended Next Guides


---

## References

1. Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 2: Getting Started, Chapter 32: String Matching.
   — The most authoritative textbook on algorithms. Covers everything from array basics to string search algorithms (KMP, Rabin-Karp, Boyer-Moore).

2. Sedgewick, R., & Wayne, K. (2011). *Algorithms* (4th ed.). Addison-Wesley. Chapter 1.3: Bags, Queues, and Stacks; Chapter 5: Strings.
   — An implementation-focused algorithms textbook. Features concrete Java implementations and rich illustrations. Covers amortized analysis of dynamic arrays and practical string processing algorithms.

3. Bryant, R. E., & O'Hallaron, D. R. (2015). *Computer Systems: A Programmer's Perspective* (3rd ed.). Pearson. Chapter 6: The Memory Hierarchy.
   — Essential reading for deeply understanding the relationship between cache mechanisms and array access patterns. Provides detailed coverage of spatial/temporal locality concepts and cache line behavior.

4. Pike, R., & Thompson, K. (2003). "Hello World" or Καλημέρα κόσμε or こんにちは 世界. *Proceedings of the USENIX Annual Technical Conference*.
   — Unicode encoding explained by the designers of UTF-8. The original source for understanding why UTF-8 is an excellent design.

5. Knuth, D. E. (1997). *The Art of Computer Programming, Volume 1: Fundamental Algorithms* (3rd ed.). Addison-Wesley.
   — Mathematical foundations for arrays and basic data structures. Deeply explores information density analysis and the theoretical background of address computation.

6. CPython Source Code: `Objects/listobject.c`, `Objects/unicodeobject.c`.
   — Primary source for list and string implementation in CPython. Growth policies and interning code can be examined directly. URL: https://github.com/python/cpython
