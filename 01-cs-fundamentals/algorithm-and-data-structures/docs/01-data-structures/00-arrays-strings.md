# Arrays and Strings — A Complete Guide to Dynamic Arrays, String Algorithms, and 2D Arrays

> Gain a deep understanding of arrays and strings — the most fundamental data structures in programming — covering dynamic array internals, string operation complexities, 2D array traversal patterns, and frequently tested techniques such as Two Pointers and Sliding Window, all presented systematically.

---


## What You Will Learn in This Chapter

- [ ] Understanding of fundamental concepts and terminology
- [ ] Mastery of implementation patterns and best practices
- [ ] Comprehension of practical application methods
- [ ] Basics of troubleshooting

---

## Prerequisites

Understanding the following will help you get the most from this guide:

- Basic programming knowledge
- Understanding of related foundational concepts

---

## Table of Contents

1. [Array Fundamentals — Memory Model and Address Calculation](#1-array-fundamentals--memory-model-and-address-calculation)
2. [Dynamic Array Internals — Growth Strategy and Amortized Analysis](#2-dynamic-array-internals--growth-strategy-and-amortized-analysis)
3. [Multidimensional Arrays — Row-Major and Column-Major Memory Layout](#3-multidimensional-arrays--row-major-and-column-major-memory-layout)
4. [String Fundamentals — Immutability, Encoding, and Internal Representation](#4-string-fundamentals--immutability-encoding-and-internal-representation)
5. [Fundamental Array Algorithms — Rotation, Merge, and Partition](#5-fundamental-array-algorithms--rotation-merge-and-partition)
6. [String Algorithms — Palindrome, Anagram, and Pattern Search](#6-string-algorithms--palindrome-anagram-and-pattern-search)
7. [Two Pointers and Sliding Window](#7-two-pointers-and-sliding-window)
8. [2D Array Traversal Patterns — Spiral, Diagonal, and Rotation](#8-2d-array-traversal-patterns--spiral-diagonal-and-rotation)
9. [Comparison Tables and Complexity Cheat Sheet](#9-comparison-tables-and-complexity-cheat-sheet)
10. [Anti-Patterns and Correct Approaches](#10-anti-patterns-and-correct-approaches)
11. [Exercises — Three Levels: Beginner, Intermediate, and Advanced](#11-exercises--three-levels-beginner-intermediate-and-advanced)
12. [FAQ — Frequently Asked Questions](#12-faq--frequently-asked-questions)
13. [Summary and Next Steps](#13-summary-and-next-steps)
14. [References](#14-references)

---

## 1. Array Fundamentals — Memory Model and Address Calculation

### 1.1 What Is an Array?

An array is a data structure that stores elements of the same type in a contiguous region of memory.
It is one of the most fundamental and important data structures in computer science.

Why arrays are important:

- **O(1) random access**: Addresses can be computed directly from the index
- **Cache efficiency**: Contiguous memory layout results in high CPU cache hit rates
- **Foundation for other data structures**: Many structures like hash tables, heaps, and dynamic arrays are built on top of arrays
- **Language-level optimization**: Nearly all programming languages provide native support

### 1.2 Memory Layout and Address Calculation

The defining characteristic of arrays is that elements are placed at contiguous memory addresses.
This enables O(1) access to any index.

```
Memory layout of static array int arr[5] (4 bytes per element):

  Base address: 0x1000

  Address:  0x1000  0x1004  0x1008  0x100C  0x1010
            +-------+-------+-------+-------+-------+
            |  10   |  20   |  30   |  40   |  50   |
            +-------+-------+-------+-------+-------+
  index:      [0]     [1]     [2]     [3]     [4]

  Address calculation formula:
    Address of arr[i] = base_address + i * sizeof(element)
    Address of arr[3] = 0x1000 + 3 * 4 = 0x100C

  This calculation completes in 1 instruction at the hardware level -> O(1)
```

### 1.3 Static Arrays vs. Dynamic Arrays

Static arrays have their size determined at compile time and are often placed on the stack.
Dynamic arrays can change size at runtime and are placed on the heap.

```
Static array (C):
  int arr[5];                   // Allocate 20 bytes on the stack
  +-------------------------+
  |  Stack area (fixed)      |
  |  arr: [_][_][_][_][_]    |   Size cannot change
  +-------------------------+

Dynamic array (C):
  int *arr = malloc(5 * sizeof(int));  // Allocate on the heap
  +-------------------------+
  |  Stack area             |
  |  arr: pointer ----------+--+
  +-------------------------+  |
                                v
  +---------------------------------+
  |  Heap area                      |
  |  [_][_][_][_][_][...spare...]   |   Can be extended with realloc
  +---------------------------------+
```

### 1.4 Array Implementations in Various Languages

Array implementations vary significantly across languages:

| Language | Static Array | Dynamic Array | Characteristics |
|----------|-------------|---------------|-----------------|
| C | `int arr[N]` | `malloc` + `realloc` | No bounds checking, fastest |
| C++ | `std::array<T,N>` | `std::vector<T>` | Automatic memory management via RAII |
| Java | `int[]` | `ArrayList<Integer>` | Bounds checking, primitive types stored directly |
| Python | None | `list` | All elements are object pointers |
| Go | `[N]T` | Slice `[]T` | Value-type arrays + reference-type slices |
| Rust | `[T; N]` | `Vec<T>` | Safety guaranteed by the ownership system |

### 1.5 Code Example: Basic Array Operations and Complexity Verification

```python
"""
Program to verify basic array operations and the complexity of each operation
"""
import time
import sys


def demonstrate_array_basics():
    """Demo of basic array (Python list) operations"""

    # === Creation ===
    arr = [10, 20, 30, 40, 50]
    print(f"Array: {arr}")
    print(f"Length: {len(arr)}")
    print(f"Memory size: {sys.getsizeof(arr)} bytes")
    print()

    # === O(1) Random Access ===
    print(f"arr[0] = {arr[0]}")   # First element
    print(f"arr[2] = {arr[2]}")   # Middle element
    print(f"arr[-1] = {arr[-1]}") # Last element
    print()

    # === O(1) Amortized Tail Append ===
    arr.append(60)
    print(f"After append(60): {arr}")

    # === O(n) Front Insertion ===
    arr.insert(0, 5)
    print(f"After insert(0, 5): {arr}")
    # All elements must be shifted right by one, hence O(n)

    # === O(n) Middle Deletion ===
    arr.pop(3)
    print(f"After pop(3): {arr}")
    # Elements after the deletion point must shift left, hence O(n)

    # === O(1) Tail Deletion ===
    arr.pop()
    print(f"After pop(): {arr}")
    print()

    # === Slice Operations ===
    print(f"arr[1:4] = {arr[1:4]}")     # O(k) where k=slice length
    print(f"arr[::2] = {arr[::2]}")     # O(n/2)
    print(f"arr[::-1] = {arr[::-1]}")   # O(n) reversed copy


def measure_append_performance():
    """Measure dynamic array append performance"""
    sizes = [10_000, 100_000, 1_000_000]

    for n in sizes:
        start = time.perf_counter()
        arr = []
        for i in range(n):
            arr.append(i)
        elapsed = time.perf_counter() - start
        print(f"n={n:>10,}: {n} appends = {elapsed:.4f}s")
        # Results are roughly proportional to n (append is amortized O(1) per call)


def measure_insert_front_performance():
    """Measure front insertion performance (O(n) x n = O(n^2))"""
    sizes = [10_000, 50_000, 100_000]

    for n in sizes:
        start = time.perf_counter()
        arr = []
        for i in range(n):
            arr.insert(0, i)
        elapsed = time.perf_counter() - start
        print(f"n={n:>10,}: {n} insert(0) calls = {elapsed:.4f}s")
        # When n grows 5x, time grows ~25x (characteristic of O(n^2))


if __name__ == "__main__":
    print("=== Basic Array Operations ===")
    demonstrate_array_basics()
    print()
    print("=== append Performance ===")
    measure_append_performance()
    print()
    print("=== insert(0) Performance ===")
    measure_insert_front_performance()
```

---

## 2. Dynamic Array Internals — Growth Strategy and Amortized Analysis

### 2.1 How Dynamic Arrays Work

A dynamic array maintains a fixed-size internal array and allocates a larger array
when capacity is exceeded, copying data over. This "resize" operation costs
O(n), but since it occurs infrequently, append is amortized O(1) overall.

```
Growth process of a dynamic array (growth factor 2x):

Step 1: Initial state (capacity=1)
  [A]            Used: 1/1 (100%)

Step 2: append(B) -> Capacity exceeded! Resize (capacity 1->2)
  Allocate new array -> Copy [A] -> Add [B]
  [A][B]          Used: 2/2 (100%)

Step 3: append(C) -> Capacity exceeded! Resize (capacity 2->4)
  Allocate new array -> Copy [A][B] -> Add [C]
  [A][B][C][ ]    Used: 3/4 (75%)

Step 4: append(D) -> Space available, add directly
  [A][B][C][D]    Used: 4/4 (100%)

Step 5: append(E) -> Capacity exceeded! Resize (capacity 4->8)
  Allocate new array -> Copy [A][B][C][D] -> Add [E]
  [A][B][C][D][E][ ][ ][ ]    Used: 5/8 (62.5%)

Step 6-8: append(F), append(G), append(H) -> Space available
  [A][B][C][D][E][F][G][H]    Used: 8/8 (100%)

Step 9: append(I) -> Capacity exceeded! Resize (capacity 8->16)
  [A][B][C][D][E][F][G][H][I][ ][ ][ ][ ][ ][ ][ ]  Used: 9/16 (56.25%)
```

### 2.2 Growth Factor Selection and Language Implementations

The growth factor is a tradeoff between space efficiency and time efficiency:

| Implementation | Growth Factor | Characteristics |
|---------------|---------------|-----------------|
| Python `list` | ~1.125x | Memory-efficient, grows in fine steps |
| Java `ArrayList` | 1.5x | Balanced |
| C++ `std::vector` (GCC) | 2x | Speed-oriented, memory usage up to 2x |
| C++ `std::vector` (MSVC) | 1.5x | Balanced |
| Go slice | 2x (small), 1.25x (large) | Growth rate varies with size |
| Rust `Vec` | 2x | Speed-oriented |

### 2.3 Amortized Analysis — Why Is append O(1)?

Amortized Analysis analyzes the average cost across a sequence of operations
rather than individual operations.

**Aggregate Method**:

Consider n appends (growth factor 2x):
- Resizes occur at sizes 1, 2, 4, 8, ..., 2^k (2^k <= n)
- Total copy cost: 1 + 2 + 4 + ... + 2^k = 2^(k+1) - 1 < 2n
- Total cost of n appends: n (additions) + 2n (copies) = 3n
- Amortized cost per operation: 3n / n = O(1)

**Accounting (Banker's) Method**:

Charge "3 coins" for each append:
- 1 coin: Used for adding the element
- 2 coins: Saved (banked for future resizes)

When resizing from capacity m to 2m:
- m elements need to be copied
- The m elements added since the last resize each banked 2 coins
- Total 2m coins available, but copying only costs m coins

Therefore, the amortized cost per operation is O(1).

### 2.4 Code Example: Custom Dynamic Array Implementation

```python
"""
Custom implementation of a dynamic array (DynamicArray)
Educational code to understand the internal workings of Python's list
"""
import ctypes


class DynamicArray:
    """Dynamic array with 2x growth factor"""

    def __init__(self):
        self._size = 0          # Actual number of elements
        self._capacity = 1      # Internal array capacity
        self._array = self._make_array(self._capacity)
        self._resize_count = 0  # Track resize count

    def _make_array(self, capacity):
        """Allocate an internal array of the specified capacity"""
        return (capacity * ctypes.py_object)()

    def __len__(self):
        """Return element count — O(1)"""
        return self._size

    def __getitem__(self, index):
        """Index access — O(1)"""
        if not 0 <= index < self._size:
            raise IndexError(f"index {index} out of range [0, {self._size})")
        return self._array[index]

    def __setitem__(self, index, value):
        """Assignment by index — O(1)"""
        if not 0 <= index < self._size:
            raise IndexError(f"index {index} out of range [0, {self._size})")
        self._array[index] = value

    def append(self, value):
        """Append element to end — amortized O(1)"""
        if self._size == self._capacity:
            self._resize(2 * self._capacity)
        self._array[self._size] = value
        self._size += 1

    def insert(self, index, value):
        """Insert element at specified position — O(n)"""
        if not 0 <= index <= self._size:
            raise IndexError(f"index {index} out of range [0, {self._size}]")
        if self._size == self._capacity:
            self._resize(2 * self._capacity)
        # Shift elements from index onward to the right
        for i in range(self._size, index, -1):
            self._array[i] = self._array[i - 1]
        self._array[index] = value
        self._size += 1

    def pop(self, index=None):
        """Remove and return element — O(1) for tail, O(n) otherwise"""
        if self._size == 0:
            raise IndexError("pop from empty array")
        if index is None:
            index = self._size - 1
        if not 0 <= index < self._size:
            raise IndexError(f"index {index} out of range [0, {self._size})")

        value = self._array[index]
        # Shift elements from index onward to the left
        for i in range(index, self._size - 1):
            self._array[i] = self._array[i + 1]
        self._size -= 1

        # Shrink if usage drops to 25% or below
        if self._size > 0 and self._size <= self._capacity // 4:
            self._resize(self._capacity // 2)

        return value

    def _resize(self, new_capacity):
        """Resize internal array — O(n)"""
        new_array = self._make_array(new_capacity)
        for i in range(self._size):
            new_array[i] = self._array[i]
        self._array = new_array
        self._capacity = new_capacity
        self._resize_count += 1

    def __repr__(self):
        items = [str(self._array[i]) for i in range(self._size)]
        return (f"DynamicArray([{', '.join(items)}], "
                f"size={self._size}, capacity={self._capacity}, "
                f"resizes={self._resize_count})")


def demo_dynamic_array():
    """DynamicArray behavior demo"""
    da = DynamicArray()
    print(f"Initial state: {da}")

    # Observe the resize process while adding elements
    for i in range(1, 17):
        da.append(i * 10)
        print(f"append({i*10:>3}): size={len(da):>2}, "
              f"capacity={da._capacity:>2}, resizes={da._resize_count}")

    print(f"\nFinal state: {da}")
    print(f"da[5] = {da[5]}")

    # Element deletion
    removed = da.pop()
    print(f"\npop() = {removed}")
    removed = da.pop(0)
    print(f"pop(0) = {removed}")
    print(f"After deletion: {da}")


if __name__ == "__main__":
    demo_dynamic_array()
```

### 2.5 Dynamic Array Shrinking Strategy

Dynamic arrays not only grow but also shrink when the element count drops significantly.
However, the shrink threshold must not be the inverse of the growth threshold.

```
Dangerous design (thrashing problem):
  Growth factor 2x, shrink threshold 50%

  Capacity 8, 4 elements: [A][B][C][D][ ][ ][ ][ ]   Usage 50%
  append(E) -> Usage 62.5%, no resize needed
  pop() -> Usage below 50% -> Shrink to capacity 4
  append(E) -> Capacity exceeded -> Grow to capacity 8
  pop() -> Shrink to capacity 4
  ...resizing repeats forever!

  Safe design:
  Growth factor 2x, shrink threshold 25%
  -> Sufficient buffer between growth and shrink thresholds
```

---

## 3. Multidimensional Arrays — Row-Major and Column-Major Memory Layout

### 3.1 Memory Layout Methods

A 2D array is logically a grid of rows and columns, but it is laid out in 1D in memory.
The layout method differs by language and has a significant impact on performance.

```
Logical 2D array (3x4):
          Col0 Col1 Col2 Col3
  Row0  [  1    2    3    4  ]
  Row1  [  5    6    7    8  ]
  Row2  [  9   10   11   12  ]

Row-Major — C, C++, Python, Java:
  Memory: [1][2][3][4][5][6][7][8][9][10][11][12]
           <--- Row0 ---><--- Row1 ---><--- Row2 --->

  Address calculation: arr[i][j] = base + (i * cols + j) * sizeof(element)
  arr[1][2] = base + (1 * 4 + 2) * 4 = base + 24

Column-Major — Fortran, MATLAB, Julia, R:
  Memory: [1][5][9][2][6][10][3][7][11][4][8][12]
           <-- Col0 --><-- Col1 ---><-- Col2 ---><-- Col3 -->

  Address calculation: arr[i][j] = base + (j * rows + i) * sizeof(element)
```

### 3.2 Cache Efficiency and Traversal Order

CPUs exploit "spatial locality" in their caches — when an address is accessed,
nearby addresses are also loaded into the cache line. Therefore, choosing a
traversal order that matches the memory layout is crucial.

```
Cache efficiency for row-major storage:

  Fast (row-wise traversal — high cache hit rate):
    for i in range(rows):
        for j in range(cols):     <- Access contiguous memory in order
            process(arr[i][j])

  Slow (column-wise traversal — frequent cache misses):
    for j in range(cols):
        for i in range(rows):     <- Jump around in memory
            process(arr[i][j])

  Expected performance difference: For large arrays (e.g., 10000x10000),
  a 2-10x difference can occur
```

### 3.3 Code Example: Row-Major vs. Column-Major Traversal Performance

```python
"""
Program comparing row-major and column-major traversal speeds
Experience the difference in cache efficiency
"""
import time


def create_matrix(rows, cols):
    """Generate a rows x cols matrix"""
    return [[i * cols + j for j in range(cols)] for i in range(rows)]


def row_major_sum(matrix, rows, cols):
    """Calculate sum with row-major traversal — cache-friendly"""
    total = 0
    for i in range(rows):
        for j in range(cols):
            total += matrix[i][j]
    return total


def col_major_sum(matrix, rows, cols):
    """Calculate sum with column-major traversal — cache-unfriendly"""
    total = 0
    for j in range(cols):
        for i in range(rows):
            total += matrix[i][j]
    return total


def benchmark():
    """Measure performance difference from traversal order"""
    sizes = [(1000, 1000), (3000, 3000), (5000, 5000)]

    for rows, cols in sizes:
        matrix = create_matrix(rows, cols)

        # Row-major traversal
        start = time.perf_counter()
        s1 = row_major_sum(matrix, rows, cols)
        t_row = time.perf_counter() - start

        # Column-major traversal
        start = time.perf_counter()
        s2 = col_major_sum(matrix, rows, cols)
        t_col = time.perf_counter() - start

        assert s1 == s2
        ratio = t_col / t_row if t_row > 0 else float('inf')
        print(f"{rows}x{cols}: row-major={t_row:.4f}s, "
              f"col-major={t_col:.4f}s, ratio={ratio:.2f}x")


if __name__ == "__main__":
    benchmark()
```

### 3.4 Multidimensional Array Representations

In Python, there are multiple ways to represent multidimensional arrays, used for different purposes:

| Method | Code Example | Memory Layout | Use Case |
|--------|-------------|---------------|----------|
| List of lists | `[[0]*n for _ in range(m)]` | Non-contiguous | Small-scale, general |
| numpy ndarray | `np.zeros((m, n))` | Contiguous (row-major) | Numerical computation |
| array module | `array.array('i', [...])` | Contiguous (1D only) | Memory efficiency |
| ctypes array | `(c_int * n * m)()` | Contiguous | C library integration |

---

## 4. String Fundamentals — Immutability, Encoding, and Internal Representation

### 4.1 String Memory Representation

Strings are stored as "arrays of characters," but how characters are encoded into byte sequences
differs by language and platform.

```
Memory layout of ASCII string "Hello":

  C (NULL-terminated):
  +-----+-----+-----+-----+-----+-----+
  | 'H' | 'e' | 'l' | 'l' | 'o' | '\0'|
  | 0x48| 0x65| 0x6C| 0x6C| 0x6F| 0x00|
  +-----+-----+-----+-----+-----+-----+
  6 bytes (including NULL terminator)

  Python 3 (compact ASCII):
  +------------------------------+
  | PyObject header (16 bytes)    |
  | hash: cached (8 bytes)        |
  | length: 5 (8 bytes)           |
  | +-----+-----+-----+-----+-----+
  | | 'H' | 'e' | 'l' | 'l' | 'o' |
  | +-----+-----+-----+-----+-----+
  +------------------------------+
  ~54 bytes (including object header)

  Java (UTF-16, Java 9+ compact):
  +------------------------------+
  | Object header (12 bytes)      |
  | hash: 0 (not computed) (4 B)  |
  | value: byte[] reference (8 B) |
  | coder: LATIN1 or UTF16        |
  +------------------------------+
```

### 4.2 String Immutability

In many languages, strings are immutable. Modification operations always create a new string object.

```
What actually happens when "modifying" a string in Python:

  s = "Hello"
  id(s) -> 0x7f8001000  <- Address A

  s = s + " World"
  id(s) -> 0x7f8002000  <- Address B (a new object!)

  Memory changes:
  Before:
    s --> [H][e][l][l][o]                   (Address A)

  After:
    s --> [H][e][l][l][o][ ][W][o][r][l][d] (Address B, newly created)
           [H][e][l][l][o]                   (Address A, eligible for GC)

  Benefits of immutability:
  1. Hash values can be cached -> usable as dict keys with O(1) lookup
  2. Thread-safe -> can be shared without locks
  3. String interning -> save memory by sharing identical content
```

Languages with and without immutability:

| Immutable | Mutable |
|-----------|---------|
| Python, Java, C#, Go, JavaScript | C (char[]), C++ (std::string), Rust (String) |

### 4.3 Encoding Basics

```
Unicode encoding comparison:

  Character "A" (U+0041):
    UTF-8:  [0x41]                    — 1 byte
    UTF-16: [0x00][0x41]              — 2 bytes
    UTF-32: [0x00][0x00][0x00][0x41]  — 4 bytes

  Character "あ" (U+3042):
    UTF-8:  [0xE3][0x81][0x82]        — 3 bytes
    UTF-16: [0x30][0x42]              — 2 bytes
    UTF-32: [0x00][0x00][0x30][0x42]  — 4 bytes

  Emoji "😀" (U+1F600):
    UTF-8:  [0xF0][0x9F][0x98][0x80]  — 4 bytes
    UTF-16: [0xD83D][0xDE00]          — 4 bytes (surrogate pair)
    UTF-32: [0x00][0x01][0xF6][0x00]  — 4 bytes
```

| Encoding | Bytes/Character | Characteristics | Primary Use |
|----------|----------------|-----------------|-------------|
| UTF-8 | 1-4 bytes/char | ASCII compatible, variable-length | Web, Linux, files |
| UTF-16 | 2-4 bytes/char | BMP is 2 bytes | Windows, Java, JavaScript |
| UTF-32 | 4 bytes/char | Fixed-length, O(1) access | Internal processing |

### 4.4 Python 3 String Internal Representation (PEP 393)

Since Python 3.3, PEP 393 (Flexible String Representation) automatically selects
the optimal encoding based on the string content:

| Character Range | Internal Encoding | Size per Character |
|----------------|-------------------|-------------------|
| U+0000 - U+00FF (Latin-1) | Latin-1 | 1 byte |
| U+0000 - U+FFFF (BMP) | UCS-2 | 2 bytes |
| U+0000 - U+10FFFF (full range) | UCS-4 | 4 bytes |

### 4.5 Code Example: Complexity-Aware String Operations

```python
"""
Collection of implementation patterns for complexity-aware string operations
Learn the correct way to build strings in languages with immutable strings
"""
import sys
import time


def string_building_comparison():
    """String building: concatenation vs. join performance comparison"""
    n = 50000
    words = [f"word{i}" for i in range(n)]

    # Method 1: += concatenation — O(n^2)
    start = time.perf_counter()
    result1 = ""
    for w in words:
        result1 += w
    t1 = time.perf_counter() - start

    # Method 2: join — O(n)
    start = time.perf_counter()
    result2 = "".join(words)
    t2 = time.perf_counter() - start

    # Method 3: io.StringIO — O(n)
    import io
    start = time.perf_counter()
    buf = io.StringIO()
    for w in words:
        buf.write(w)
    result3 = buf.getvalue()
    t3 = time.perf_counter() - start

    assert result1 == result2 == result3
    print(f"+= concat:  {t1:.4f}s")
    print(f"join:       {t2:.4f}s")
    print(f"StringIO:   {t3:.4f}s")
    print(f"join is ~{t1/t2:.1f}x faster than +=")


def unicode_details():
    """Observe Python's internal string representation"""
    strings = [
        "Hello",           # ASCII only -> Latin-1 (1 byte/char)
        "Bonjour cafe\u0301",  # Latin-1 range -> Latin-1
        "\u3053\u3093\u306b\u3061\u306f",       # BMP -> UCS-2 (2 bytes/char)
        "Hello \U0001f600",        # Contains emoji -> UCS-4 (4 bytes/char)
    ]

    for s in strings:
        size = sys.getsizeof(s)
        # Estimate data size excluding header
        overhead = sys.getsizeof("") # Size of empty string
        data_size = size - overhead
        per_char = data_size / len(s) if len(s) > 0 else 0
        print(f"'{s}': len={len(s)}, size={size}B, "
              f"data={data_size}B, ~{per_char:.1f}B/char")


if __name__ == "__main__":
    print("=== String Building Performance Comparison ===")
    string_building_comparison()
    print()
    print("=== Unicode Internal Representation ===")
    unicode_details()
```

### 4.6 String Operation Complexity Summary

| Operation | Python | Java | C++ (std::string) | Notes |
|-----------|--------|------|--------------------|-------|
| Index access `s[i]` | O(1) | O(1) | O(1) | Assumes fixed-width encoding |
| Concatenation `s + t` | O(n+m) | O(n+m) | O(n+m) | Creates a new string |
| Substring `s[i:j]` | O(j-i) | O(j-i) | O(j-i) | Copy occurs |
| Search `s.find(t)` | O(n*m) worst | O(n*m) worst | O(n*m) worst | Worst case |
| Comparison `s == t` | O(min(n,m)) | O(min(n,m)) | O(min(n,m)) | Compare from start |
| Hash `hash(s)` | O(n) first time | O(n) first time | O(n) | Cached after first computation |
| Length `len(s)` | O(1) | O(1) | O(1) | Pre-stored |
| Replace `s.replace(a, b)` | O(n*m) | O(n*m) | O(n*m) | Replace all occurrences |

---

## 5. Fundamental Array Algorithms — Rotation, Merge, and Partition

### 5.1 Merging Sorted Arrays

Merging two sorted arrays into one sorted array is the core of merge sort
and a fundamental array algorithm.

```
Visualization of the merge process:

  a = [1, 3, 5, 7]    b = [2, 4, 6, 8]
       ^                    ^
       i=0                  j=0

  Compare: a[0]=1 < b[0]=2 -> Add 1 to result, i++
  Compare: a[1]=3 > b[0]=2 -> Add 2 to result, j++
  Compare: a[1]=3 < b[1]=4 -> Add 3 to result, i++
  Compare: a[2]=5 > b[1]=4 -> Add 4 to result, j++
  ...

  result = [1, 2, 3, 4, 5, 6, 7, 8]
```

```python
"""
Merging sorted arrays — fully working version
"""


def merge_sorted(a: list, b: list) -> list:
    """
    Merge two sorted arrays into one sorted array

    Time complexity: O(n + m)  n = len(a), m = len(b)
    Space complexity: O(n + m)  for the result array
    """
    result = []
    i = j = 0

    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1

    # Add remaining elements
    result.extend(a[i:])
    result.extend(b[j:])
    return result


def merge_sorted_inplace(arr: list, mid: int) -> None:
    """
    If arr[0:mid] and arr[mid:] are each sorted,
    sort the entire arr (additional memory O(n))
    """
    left = arr[:mid]
    right = arr[mid:]
    i = j = k = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            arr[k] = left[i]
            i += 1
        else:
            arr[k] = right[j]
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


# Test
if __name__ == "__main__":
    # Basic tests
    assert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]
    assert merge_sorted([], [1, 2, 3]) == [1, 2, 3]
    assert merge_sorted([1], []) == [1]
    assert merge_sorted([], []) == []
    assert merge_sorted([1, 1, 1], [1, 1]) == [1, 1, 1, 1, 1]

    # In-place test
    arr = [1, 3, 5, 2, 4, 6]
    merge_sorted_inplace(arr, 3)
    assert arr == [1, 2, 3, 4, 5, 6]
    print("All tests passed")
```

### 5.2 Array Rotation

Array rotation is the operation of cyclically shifting elements by a specified number of positions
left or right. It can be achieved in O(n) time and O(1) space using three reversals.

```
Reversal algorithm for left rotation by k=2:

  Original array:  [1, 2, 3, 4, 5, 6, 7]

  Step 1: Reverse the first k elements
           [2, 1, | 3, 4, 5, 6, 7]

  Step 2: Reverse the remaining n-k elements
           [2, 1, | 7, 6, 5, 4, 3]

  Step 3: Reverse the entire array
           [3, 4, 5, 6, 7, 1, 2]   <- Left-rotated by 2!

  Why this works:
    Original: [a1, a2, ..., ak, b1, b2, ..., bn-k]
    Goal:     [b1, b2, ..., bn-k, a1, a2, ..., ak]

    Step 1: [ak, ..., a2, a1, b1, b2, ..., bn-k]   = rev(A) + B
    Step 2: [ak, ..., a2, a1, bn-k, ..., b2, b1]   = rev(A) + rev(B)
    Step 3: [b1, b2, ..., bn-k, a1, a2, ..., ak]   = rev(rev(A) + rev(B))
           = B + A                                   <- Goal achieved!
```

```python
"""
Array rotation — comparison of 3 methods
"""


def rotate_left_reversal(arr: list, k: int) -> list:
    """
    Left rotation using the reversal algorithm
    Time: O(n), Space: O(1) — in-place
    """
    if not arr:
        return arr
    n = len(arr)
    k = k % n
    if k == 0:
        return arr

    def reverse(lo: int, hi: int) -> None:
        while lo < hi:
            arr[lo], arr[hi] = arr[hi], arr[lo]
            lo += 1
            hi -= 1

    reverse(0, k - 1)      # Reverse first k elements
    reverse(k, n - 1)      # Reverse remaining n-k elements
    reverse(0, n - 1)      # Reverse entire array
    return arr


def rotate_left_slice(arr: list, k: int) -> list:
    """
    Left rotation using slicing
    Time: O(n), Space: O(n)
    """
    if not arr:
        return arr
    k = k % len(arr)
    return arr[k:] + arr[:k]


def rotate_right(arr: list, k: int) -> list:
    """
    Right rotation — equivalent to left rotation by (n - k)
    Time: O(n), Space: O(1)
    """
    if not arr:
        return arr
    n = len(arr)
    k = k % n
    return rotate_left_reversal(arr, n - k)


# Test
if __name__ == "__main__":
    # Left rotation test
    assert rotate_left_slice([1, 2, 3, 4, 5, 6, 7], 2) == [3, 4, 5, 6, 7, 1, 2]

    arr = [1, 2, 3, 4, 5, 6, 7]
    rotate_left_reversal(arr, 2)
    assert arr == [3, 4, 5, 6, 7, 1, 2]

    # Right rotation test
    arr2 = [1, 2, 3, 4, 5, 6, 7]
    rotate_right(arr2, 2)
    assert arr2 == [6, 7, 1, 2, 3, 4, 5]

    # Edge cases
    assert rotate_left_slice([], 3) == []
    assert rotate_left_slice([1], 5) == [1]
    assert rotate_left_slice([1, 2], 4) == [1, 2]  # k % n = 0
    print("All tests passed")
```

### 5.3 Dutch National Flag (Three-Way Partition)

An algorithm that partitions an array into three regions. It forms the basis of quicksort's 3-way partition.

```python
"""
Dutch National Flag problem — sorting an array with 3 values
Equivalent to LeetCode 75: Sort Colors
"""


def dutch_national_flag(arr: list, pivot: int = 1) -> list:
    """
    Partition the array into [< pivot | == pivot | > pivot] three regions

    Time complexity: O(n) — completes in one pass
    Space complexity: O(1) — in-place

    Three pointers:
      lo:  Next position to place an element < pivot
      mid: Currently examined position
      hi:  Next position to place an element > pivot
    """
    lo = mid = 0
    hi = len(arr) - 1

    while mid <= hi:
        if arr[mid] < pivot:
            arr[lo], arr[mid] = arr[mid], arr[lo]
            lo += 1
            mid += 1
        elif arr[mid] > pivot:
            arr[mid], arr[hi] = arr[hi], arr[mid]
            hi -= 1
            # Don't advance mid (the swapped element hasn't been examined yet)
        else:
            mid += 1

    return arr


# Test
if __name__ == "__main__":
    assert dutch_national_flag([2, 0, 1, 2, 0, 1, 0]) == [0, 0, 0, 1, 1, 2, 2]
    assert dutch_national_flag([1, 1, 1]) == [1, 1, 1]
    assert dutch_national_flag([2, 1, 0]) == [0, 1, 2]
    assert dutch_national_flag([]) == []
    assert dutch_national_flag([0]) == [0]
    print("All tests passed")
```

---

## 6. String Algorithms — Palindrome, Anagram, and Pattern Search

### 6.1 Palindrome Detection

A palindrome reads the same forwards and backwards.
It can be detected in O(n) time and O(1) space using Two Pointers.

```python
"""
Palindrome detection — 3 variations
"""


def is_palindrome(s: str) -> bool:
    """
    Palindrome detection considering only alphanumeric characters
    Time: O(n), Space: O(n) — generates a filtered string
    """
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]


def is_palindrome_two_pointers(s: str) -> bool:
    """
    Palindrome detection using Two Pointers (O(1) space)
    Considers only alphanumeric characters, case-insensitive
    """
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


def is_almost_palindrome(s: str) -> bool:
    """
    Determine if a palindrome can be formed by deleting at most 1 character
    LeetCode 680: Valid Palindrome II
    """
    def check(lo: int, hi: int) -> bool:
        while lo < hi:
            if s[lo] != s[hi]:
                return False
            lo += 1
            hi -= 1
        return True

    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            # Two choices: skip left or skip right
            return check(left + 1, right) or check(left, right - 1)
        left += 1
        right -= 1
    return True


# Test
if __name__ == "__main__":
    # Basic tests
    assert is_palindrome("A man, a plan, a canal: Panama") is True
    assert is_palindrome("race a car") is False
    assert is_palindrome("") is True

    # Two Pointers version
    assert is_palindrome_two_pointers("A man, a plan, a canal: Panama") is True
    assert is_palindrome_two_pointers("race a car") is False

    # Almost Palindrome
    assert is_almost_palindrome("abca") is True   # Remove 'c' -> "aba"
    assert is_almost_palindrome("abc") is False
    assert is_almost_palindrome("aba") is True     # No deletion needed
    assert is_almost_palindrome("a") is True
    print("All tests passed")
```

### 6.2 Anagram Detection and Search

An anagram is a pair of strings where one can be formed by rearranging the characters of the other.
Detection uses character frequency counting.

```python
"""
Anagram detection and substring anagram search
"""
from collections import Counter


def is_anagram(s: str, t: str) -> bool:
    """
    Determine if two strings are anagrams
    Time: O(n), Space: O(k) where k = number of distinct characters
    """
    if len(s) != len(t):
        return False
    return Counter(s) == Counter(t)


def is_anagram_sort(s: str, t: str) -> bool:
    """
    Anagram detection by sorting (alternative)
    Time: O(n log n), Space: O(n)
    """
    return sorted(s) == sorted(t)


def find_anagrams(s: str, p: str) -> list:
    """
    Return all starting positions of substrings of s that are anagrams of p
    LeetCode 438: Find All Anagrams in a String

    Achieved in O(n) using Sliding Window + character frequency counting

    Time: O(n), Space: O(k) where k = number of distinct characters
    """
    if len(p) > len(s):
        return []

    result = []
    p_count = Counter(p)
    window = Counter(s[:len(p)])

    if window == p_count:
        result.append(0)

    for i in range(len(p), len(s)):
        # Add right end of window
        window[s[i]] += 1
        # Remove left end of window
        left_char = s[i - len(p)]
        window[left_char] -= 1
        if window[left_char] == 0:
            del window[left_char]

        if window == p_count:
            result.append(i - len(p) + 1)

    return result


# Test
if __name__ == "__main__":
    # Anagram detection
    assert is_anagram("listen", "silent") is True
    assert is_anagram("hello", "world") is False
    assert is_anagram("", "") is True
    assert is_anagram("a", "ab") is False

    # Anagram search
    assert find_anagrams("cbaebabacd", "abc") == [0, 6]
    assert find_anagrams("abab", "ab") == [0, 1, 2]
    assert find_anagrams("a", "ab") == []
    print("All tests passed")
```

### 6.3 Longest Common Prefix

```python
"""
Longest Common Prefix of a string array
LeetCode 14
"""


def longest_common_prefix(strs: list) -> str:
    """
    Return the longest common prefix of a list of strings

    Method: Vertical scanning — compare the character at each position across all strings
    Time: O(S) where S = total number of characters across all strings
    Space: O(1) (excluding result)
    """
    if not strs:
        return ""

    for i in range(len(strs[0])):
        char = strs[0][i]
        for j in range(1, len(strs)):
            if i >= len(strs[j]) or strs[j][i] != char:
                return strs[0][:i]

    return strs[0]


def longest_common_prefix_binary_search(strs: list) -> str:
    """
    Longest common prefix using binary search

    Time: O(S * log m) where S = shortest string length, m = shortest string length
    """
    if not strs:
        return ""

    min_len = min(len(s) for s in strs)

    def is_common_prefix(length: int) -> bool:
        prefix = strs[0][:length]
        return all(s[:length] == prefix for s in strs)

    lo, hi = 0, min_len
    while lo <= hi:
        mid = (lo + hi) // 2
        if is_common_prefix(mid):
            lo = mid + 1
        else:
            hi = mid - 1

    return strs[0][:(lo + hi) // 2 + (1 if lo > hi else 0)]


# Test
if __name__ == "__main__":
    assert longest_common_prefix(["flower", "flow", "flight"]) == "fl"
    assert longest_common_prefix(["dog", "racecar", "car"]) == ""
    assert longest_common_prefix(["alone"]) == "alone"
    assert longest_common_prefix([]) == ""
    assert longest_common_prefix(["", "b"]) == ""
    assert longest_common_prefix(["ab", "ab", "ab"]) == "ab"
    print("All tests passed")
```

---

## 7. Two Pointers and Sliding Window

### 7.1 Overview of Two Pointers

Two Pointers is a technique that uses two pointers on arrays or strings to improve
O(n^2) brute force to O(n).

```
Three patterns of Two Pointers:

Pattern 1: Opposite Direction
  Use: Pair search in sorted arrays, palindrome detection
  +-----------------------------+
  |  [1] [3] [5] [7] [9] [11]  |
  |   L->              <-R      |
  +-----------------------------+
  L moves right, R moves left

Pattern 2: Same Direction
  Use: Duplicate removal, subsequences satisfying conditions
  +-----------------------------+
  |  [1] [1] [2] [2] [3] [4]   |
  |   S->                       |
  |       F->                   |
  +-----------------------------+
  Slow and Fast move in the same direction

Pattern 3: Fast and Slow (Different Speed)
  Use: Cycle detection, finding the middle element
  +-----------------------------+
  |  [a]->[b]->[c]->[d]->[e]->  |
  |   S->                       |
  |       F-->                  |
  +-----------------------------+
  Slow moves 1 step, Fast moves 2 steps
```

### 7.2 Code Example: Representative Two Pointers Problems

```python
"""
Two Pointers pattern collection — 5 representative problems
"""


def two_sum_sorted(arr: list, target: int) -> list:
    """
    Find indices of a pair summing to target in a sorted array
    LeetCode 167: Two Sum II

    Pattern: Opposite direction
    Time: O(n), Space: O(1)
    """
    left, right = 0, len(arr) - 1

    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1

    return []  # No pair found


def remove_duplicates(arr: list) -> int:
    """
    Remove duplicates from a sorted array and return the count of unique elements
    LeetCode 26: Remove Duplicates from Sorted Array

    Pattern: Same direction (Slow/Fast)
    Time: O(n), Space: O(1)
    """
    if not arr:
        return 0

    slow = 0
    for fast in range(1, len(arr)):
        if arr[fast] != arr[slow]:
            slow += 1
            arr[slow] = arr[fast]

    return slow + 1  # Number of unique elements


def three_sum(nums: list) -> list:
    """
    Return all unique triplets that sum to 0
    LeetCode 15: 3Sum

    Sort + fix one + Two Pointers
    Time: O(n^2), Space: O(1) (excluding result)
    """
    nums.sort()
    result = []

    for i in range(len(nums) - 2):
        # Skip duplicates
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        left, right = i + 1, len(nums) - 1

        while left < right:
            total = nums[i] + nums[left] + nums[right]

            if total == 0:
                result.append([nums[i], nums[left], nums[right]])
                # Skip duplicates
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1

    return result


def container_with_most_water(height: list) -> int:
    """
    Find the container that holds the most water
    LeetCode 11: Container With Most Water

    Pattern: Opposite direction
    Time: O(n), Space: O(1)
    """
    left, right = 0, len(height) - 1
    max_area = 0

    while left < right:
        width = right - left
        h = min(height[left], height[right])
        max_area = max(max_area, width * h)

        if height[left] < height[right]:
            left += 1
        else:
            right -= 1

    return max_area


def move_zeroes(nums: list) -> None:
    """
    Move zeros to the end while maintaining order of non-zero elements
    LeetCode 283: Move Zeroes

    Pattern: Same direction
    Time: O(n), Space: O(1)
    """
    slow = 0
    for fast in range(len(nums)):
        if nums[fast] != 0:
            nums[slow], nums[fast] = nums[fast], nums[slow]
            slow += 1


# Test
if __name__ == "__main__":
    # Two Sum Sorted
    assert two_sum_sorted([1, 3, 5, 7, 9, 11], 12) == [0, 5]  # 1+11=12
    assert two_sum_sorted([2, 7, 11, 15], 9) == [0, 1]

    # Remove Duplicates
    arr = [1, 1, 2, 2, 3, 4, 4]
    k = remove_duplicates(arr)
    assert arr[:k] == [1, 2, 3, 4]

    # 3Sum
    assert three_sum([-1, 0, 1, 2, -1, -4]) == [[-1, -1, 2], [-1, 0, 1]]

    # Container With Most Water
    assert container_with_most_water([1, 8, 6, 2, 5, 4, 8, 3, 7]) == 49

    # Move Zeroes
    nums = [0, 1, 0, 3, 12]
    move_zeroes(nums)
    assert nums == [1, 3, 12, 0, 0]

    print("All tests passed")
```

### 7.3 Overview of Sliding Window

Sliding Window is a technique for efficiently solving problems involving contiguous subarrays
(or substrings). It works by appropriately moving the left and right ends of a window
while updating the state within the window.

```
Sliding Window operation (fixed size k=3):

  Array: [2, 1, 5, 1, 3, 2]

  Window 1: [2, 1, 5]          sum = 8
  Window 2:    [1, 5, 1]       sum = 8 - 2 + 1 = 7
  Window 3:       [5, 1, 3]    sum = 7 - 1 + 3 = 9  <- maximum
  Window 4:          [1, 3, 2] sum = 9 - 5 + 2 = 6

  Add right end (+) and remove left end (-) -> O(1) update
  Overall: O(n)

Sliding Window operation (variable size):

  Goal: Shortest subarray with sum >= target
  Array: [2, 3, 1, 2, 4, 3], target = 7

  [2, 3, 1, 2]              sum=8 >= 7 -> length 4, shrink left
     [3, 1, 2]              sum=6 <  7 -> expand right
     [3, 1, 2, 4]           sum=10>= 7 -> length 4, shrink left
        [1, 2, 4]           sum=7 >= 7 -> length 3, shrink left
           [2, 4]           sum=6 <  7 -> expand right
           [2, 4, 3]        sum=9 >= 7 -> length 3, shrink left
              [4, 3]        sum=7 >= 7 -> length 2 <- shortest!
```

### 7.4 Code Example: Representative Sliding Window Problems

```python
"""
Sliding Window pattern collection
"""


def max_sum_subarray(arr: list, k: int) -> int:
    """
    Return the maximum sum of a contiguous subarray of size k
    Fixed-size Sliding Window

    Time: O(n), Space: O(1)
    """
    if len(arr) < k:
        return 0

    # Sum of the first window
    window_sum = sum(arr[:k])
    max_sum = window_sum

    # Slide the window
    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i - k]
        max_sum = max(max_sum, window_sum)

    return max_sum


def min_subarray_len(target: int, nums: list) -> int:
    """
    Return the length of the shortest subarray with sum >= target
    LeetCode 209: Minimum Size Subarray Sum

    Variable-size Sliding Window
    Time: O(n), Space: O(1)
    """
    min_len = float('inf')
    window_sum = 0
    left = 0

    for right in range(len(nums)):
        window_sum += nums[right]

        while window_sum >= target:
            min_len = min(min_len, right - left + 1)
            window_sum -= nums[left]
            left += 1

    return min_len if min_len != float('inf') else 0


def longest_substring_without_repeating(s: str) -> int:
    """
    Return the length of the longest substring without repeating characters
    LeetCode 3: Longest Substring Without Repeating Characters

    Variable-size Sliding Window + HashSet
    Time: O(n), Space: O(k) where k = number of distinct characters
    """
    char_set = set()
    max_len = 0
    left = 0

    for right in range(len(s)):
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        char_set.add(s[right])
        max_len = max(max_len, right - left + 1)

    return max_len


# Test
if __name__ == "__main__":
    # Fixed size
    assert max_sum_subarray([2, 1, 5, 1, 3, 2], 3) == 9
    assert max_sum_subarray([1, 2], 3) == 0

    # Shortest subarray
    assert min_subarray_len(7, [2, 3, 1, 2, 4, 3]) == 2
    assert min_subarray_len(100, [1, 2, 3]) == 0
    assert min_subarray_len(4, [1, 4, 4]) == 1

    # Longest substring
    assert longest_substring_without_repeating("abcabcbb") == 3
    assert longest_substring_without_repeating("bbbbb") == 1
    assert longest_substring_without_repeating("pwwkew") == 3
    assert longest_substring_without_repeating("") == 0

    print("All tests passed")
```

---

## 8. 2D Array Traversal Patterns — Spiral, Diagonal, and Rotation

### 8.1 Overview of Major Traversal Patterns

```
Each traversal pattern on a 4x4 matrix:

  Matrix:
  +----+----+----+----+
  |  1 |  2 |  3 |  4 |
  +----+----+----+----+
  |  5 |  6 |  7 |  8 |
  +----+----+----+----+
  |  9 | 10 | 11 | 12 |
  +----+----+----+----+
  | 13 | 14 | 15 | 16 |
  +----+----+----+----+

  (a) Row-major:    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
  (b) Column-major: 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16
  (c) Spiral:       1, 2, 3, 4, 8, 12, 16, 15, 14, 13, 9, 5, 6, 7, 11, 10
  (d) Diagonal:     1 | 2, 5 | 3, 6, 9 | 4, 7, 10, 13 | 8, 11, 14 | 12, 15 | 16
  (e) Zigzag:       1 | 5, 2 | 3, 6, 9 | 13, 10, 7, 4 | 8, 11, 14 | 15, 12 | 16

Traversal direction visualization:

  Spiral traversal:          Zigzag (diagonal) traversal:
  -> -> -> v                 /   /   /
  ^  -> v  v               \   \   \
  ^  ^  <- v                 /   /   /
  ^  <- <- <               \   \   \
```

### 8.2 Code Example: Spiral Traversal

```python
"""
Spiral traversal of a 2D array — fully working version
"""


def spiral_order(matrix: list) -> list:
    """
    Spiral traversal of an m x n matrix
    LeetCode 54: Spiral Matrix

    Manage 4 boundaries (top, bottom, left, right) and
    traverse from outside to inside in a spiral pattern

    Time: O(m * n), Space: O(1) (excluding result)
    """
    if not matrix or not matrix[0]:
        return []

    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1

    while top <= bottom and left <= right:
        # Top edge: left to right
        for j in range(left, right + 1):
            result.append(matrix[top][j])
        top += 1

        # Right edge: top to bottom
        for i in range(top, bottom + 1):
            result.append(matrix[i][right])
        right -= 1

        # Bottom edge: right to left (only if rows remain)
        if top <= bottom:
            for j in range(right, left - 1, -1):
                result.append(matrix[bottom][j])
            bottom -= 1

        # Left edge: bottom to top (only if columns remain)
        if left <= right:
            for i in range(bottom, top - 1, -1):
                result.append(matrix[i][left])
            left += 1

    return result


def generate_spiral_matrix(n: int) -> list:
    """
    Generate an n x n spiral matrix
    LeetCode 59: Spiral Matrix II

    Place numbers 1 through n^2 in spiral order
    """
    matrix = [[0] * n for _ in range(n)]
    top, bottom = 0, n - 1
    left, right = 0, n - 1
    num = 1

    while top <= bottom and left <= right:
        for j in range(left, right + 1):
            matrix[top][j] = num
            num += 1
        top += 1

        for i in range(top, bottom + 1):
            matrix[i][right] = num
            num += 1
        right -= 1

        if top <= bottom:
            for j in range(right, left - 1, -1):
                matrix[bottom][j] = num
                num += 1
            bottom -= 1

        if left <= right:
            for i in range(bottom, top - 1, -1):
                matrix[i][left] = num
                num += 1
            left += 1

    return matrix


# Test
if __name__ == "__main__":
    # Spiral traversal
    m1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert spiral_order(m1) == [1, 2, 3, 6, 9, 8, 7, 4, 5]

    m2 = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    assert spiral_order(m2) == [1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]

    assert spiral_order([]) == []

    # Spiral matrix generation
    assert generate_spiral_matrix(3) == [
        [1, 2, 3],
        [8, 9, 4],
        [7, 6, 5]
    ]
    print("All tests passed")
```

### 8.3 Code Example: 90-Degree Matrix Rotation

```python
"""
Rotation of an N x N matrix — O(1) space solution via transpose + reverse
LeetCode 48: Rotate Image
"""


def rotate_90_clockwise(matrix: list) -> list:
    """
    Rotate an N x N matrix 90 degrees clockwise (in-place)

    Method: Transpose -> Reverse each row
    Time: O(n^2), Space: O(1)
    """
    n = len(matrix)

    # Step 1: Transpose (swap rows and columns)
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

    # Step 2: Reverse each row
    for row in matrix:
        row.reverse()

    return matrix


def rotate_90_counterclockwise(matrix: list) -> list:
    """
    Rotate an N x N matrix 90 degrees counterclockwise (in-place)

    Method: Transpose -> Reverse each column (= reverse top-bottom)
    Time: O(n^2), Space: O(1)
    """
    n = len(matrix)

    # Step 1: Transpose
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

    # Step 2: Reverse top-bottom
    matrix.reverse()

    return matrix


def rotate_180(matrix: list) -> list:
    """
    Rotate an N x N matrix 180 degrees (in-place)

    Method: Reverse top-bottom -> Reverse each row
    Time: O(n^2), Space: O(1)
    """
    matrix.reverse()
    for row in matrix:
        row.reverse()
    return matrix


# Test
if __name__ == "__main__":
    # Clockwise 90 degrees
    m1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    rotate_90_clockwise(m1)
    assert m1 == [[7, 4, 1], [8, 5, 2], [9, 6, 3]]

    # Counterclockwise 90 degrees
    m2 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    rotate_90_counterclockwise(m2)
    assert m2 == [[3, 6, 9], [2, 5, 8], [1, 4, 7]]

    # 180 degrees
    m3 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    rotate_180(m3)
    assert m3 == [[9, 8, 7], [6, 5, 4], [3, 2, 1]]

    print("All tests passed")
```

### 8.4 Code Example: Diagonal Traversal

```python
"""
Diagonal traversal of a matrix
LeetCode 498: Diagonal Traverse
"""


def diagonal_traverse(matrix: list) -> list:
    """
    Traverse an m x n matrix in zigzag diagonal order

    Time: O(m * n), Space: O(1) (excluding result)
    """
    if not matrix or not matrix[0]:
        return []

    m, n = len(matrix), len(matrix[0])
    result = []
    row, col = 0, 0
    going_up = True

    for _ in range(m * n):
        result.append(matrix[row][col])

        if going_up:
            if col == n - 1:          # Reached right edge
                row += 1
                going_up = False
            elif row == 0:            # Reached top edge
                col += 1
                going_up = False
            else:                     # Move upper-right
                row -= 1
                col += 1
        else:
            if row == m - 1:          # Reached bottom edge
                col += 1
                going_up = True
            elif col == 0:            # Reached left edge
                row += 1
                going_up = True
            else:                     # Move lower-left
                row += 1
                col -= 1

    return result


# Test
if __name__ == "__main__":
    m1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert diagonal_traverse(m1) == [1, 2, 4, 7, 5, 3, 6, 8, 9]

    m2 = [[1, 2], [3, 4]]
    assert diagonal_traverse(m2) == [1, 2, 3, 4]

    assert diagonal_traverse([]) == []

    print("All tests passed")
```

---

## 9. Comparison Tables and Complexity Cheat Sheet

### Table 1: Array Operation Complexity Comparison

| Operation | Static Array (C) | Dynamic Array (Python list) | Linked List | deque |
|-----------|------------------|-----------------------------|-------------|-------|
| Access `[i]` | O(1) | O(1) | O(n) | O(n) |
| Front insert | O(n) | O(n) | O(1) | O(1) |
| Tail insert | N/A | O(1) amortized | O(1)* | O(1) |
| Middle insert | O(n) | O(n) | O(1)** | O(n) |
| Front delete | O(n) | O(n) | O(1) | O(1) |
| Tail delete | O(1) | O(1) | O(n)* | O(1) |
| Middle delete | O(n) | O(n) | O(1)** | O(n) |
| Search (unsorted) | O(n) | O(n) | O(n) | O(n) |
| Search (sorted) | O(log n) | O(log n) | O(n) | O(n) |
| Memory efficiency | Best | Good (excess capacity) | Poor (pointers) | Good |
| Cache efficiency | Best | Good | Poor | Medium |

`*` When a tail pointer exists
`**` When the insertion/deletion position is already known (finding the position is O(n))

### Table 2: String Operation Complexity (Cross-Language Comparison)

| Operation | Python (str) | Java (String) | C++ (std::string) | Go (string) |
|-----------|-------------|---------------|-------------------|-------------|
| Index `s[i]` | O(1) | O(1) | O(1) | O(1)*** |
| Concatenation `s + t` | O(n+m) | O(n+m) | O(n+m) | O(n+m) |
| Substring | O(k) copy | O(k) copy | O(k) copy | O(k) copy |
| Search `find` | O(nm) worst | O(nm) worst | O(nm) worst | O(nm) worst |
| Length | O(1) | O(1) | O(1) | O(1) |
| Comparison | O(n) | O(n) | O(n) | O(n) |
| Hash | O(n) first only | O(n) first only | O(n) | O(n) |
| Immutability | Immutable | Immutable | Mutable | Immutable |
| Mutable version | `list` + `join` | `StringBuilder` | `std::string` itself | `[]byte` |

`***` Go's string is a byte sequence; `[]rune` conversion is needed for multi-byte characters

### Table 3: Array Implementation Comparison (Memory Characteristics)

| Implementation | Element Storage | Overhead | Cache Line Efficiency |
|---------------|----------------|----------|----------------------|
| C `int[]` | Values stored directly | None | Best |
| C++ `vector<int>` | Values stored directly | 24B (ptr+size+cap) | Best |
| Java `int[]` | Values stored directly | 16B (header) | High |
| Java `ArrayList<Integer>` | Boxing + pointers | Large | Low |
| Python `list` | Array of pointers | 56B + 8B/element | Low |
| NumPy `ndarray` | Values stored directly | Fixed header | Best |

### Table 4: Algorithm Pattern Selection Guide

| Pattern | Time Complexity | Applicable Conditions | Representative Problems |
|---------|----------------|----------------------|------------------------|
| Brute force | O(n^2) | Small constraints (n <= 1000) | All-pair examination |
| Sort + Two Pointers | O(n log n) | Sortable, pair search | Two Sum, 3Sum |
| Hash map | O(n) | Existence check, frequency counting | Anagram detection |
| Fixed Sliding Window | O(n) | Fixed-size subarrays | Maximum sum subarray |
| Variable Sliding Window | O(n) | Shortest/longest subarray satisfying condition | Shortest subarray |
| Binary search | O(log n) | Sorted array | Value search |
| Dutch National Flag | O(n) | Partition into 3 values | Sort Colors |
| Prefix Sum | O(n) preprocess + O(1) query | Range sum queries | Subarray Sum |

---

## 10. Anti-Patterns and Correct Approaches

### Anti-Pattern 1: String Concatenation in Loops

Using `+=` in a loop with immutable strings results in O(n^2).
This is one of the most commonly pointed out anti-patterns in interviews and code reviews.

```python
"""
Anti-pattern 1: String concatenation in loops
"""


def build_string_bad(words: list) -> str:
    """BAD: O(n^2) — creates a new string and copies every time"""
    result = ""
    for w in words:
        result += w  # Copies len(result) bytes every time
    return result
    # For n strings of length L:
    # Copy volume = L + 2L + 3L + ... + nL = L * n(n+1)/2 = O(n^2 * L)


def build_string_good(words: list) -> str:
    """GOOD: O(n) — join internally calculates total length then allocates once"""
    return "".join(words)
    # 1. Calculate total length: O(n)
    # 2. Allocate result buffer once: O(total length)
    # 3. Copy each string: O(total length)
    # Total: O(n + total length) = O(n)


def build_string_also_good(words: list) -> str:
    """ALSO GOOD: Accumulate in a list then join"""
    parts = []
    for w in words:
        # Process if needed before appending
        parts.append(w.upper())
    return " ".join(parts)
```

### Anti-Pattern 2: Frequent Insert/Delete at the Front of an Array

Repeatedly using `insert(0, x)` or `pop(0)` on a Python `list` costs O(n) each time
(all elements shift), making the overall complexity O(n^2).

```python
"""
Anti-pattern 2: Frequent operations at the front of an array
"""
from collections import deque


def queue_bad(operations: list) -> list:
    """BAD: Using list as a queue — dequeue is O(n)"""
    queue = []
    results = []
    for op, val in operations:
        if op == "enqueue":
            queue.append(val)       # O(1) — this is fine
        elif op == "dequeue":
            results.append(queue.pop(0))  # O(n) — shifts all elements every time!
    return results


def queue_good(operations: list) -> list:
    """GOOD: Using deque — O(1) at both ends"""
    queue = deque()
    results = []
    for op, val in operations:
        if op == "enqueue":
            queue.append(val)       # O(1)
        elif op == "dequeue":
            results.append(queue.popleft())  # O(1)
    return results
```

### Anti-Pattern 3: Shallow Copy Trap with 2D Arrays

```python
"""
Anti-pattern 3: Shallow copy trap with 2D arrays
"""


def create_matrix_bad(rows: int, cols: int) -> list:
    """BAD: All rows reference the same list object"""
    return [[0] * cols] * rows
    # matrix[0] is matrix[1] is ... -> True
    # Setting matrix[0][0] = 1 makes [0] of every row become 1!


def create_matrix_good(rows: int, cols: int) -> list:
    """GOOD: Each row is an independent list object"""
    return [[0] * cols for _ in range(rows)]
    # matrix[0] is matrix[1] -> False
    # Each row can be modified independently


# Demo
if __name__ == "__main__":
    bad = create_matrix_bad(3, 3)
    bad[0][0] = 99
    print(f"BAD:  {bad}")    # [[99, 0, 0], [99, 0, 0], [99, 0, 0]] <- affects all rows!

    good = create_matrix_good(3, 3)
    good[0][0] = 99
    print(f"GOOD: {good}")   # [[99, 0, 0], [0, 0, 0], [0, 0, 0]] <- as intended
```

### Anti-Pattern 4: Misuse of the `in` Operator

```python
"""
Anti-pattern 4: The `in` operator on a list is O(n)
"""


def contains_bad(data: list, targets: list) -> list:
    """BAD: Using `in` on a list — O(n) x m = O(nm)"""
    return [t for t in targets if t in data]  # O(n) when data is a list


def contains_good(data: list, targets: list) -> list:
    """GOOD: Convert to set then use `in` — O(1) x m = O(n+m)"""
    data_set = set(data)  # O(n) conversion
    return [t for t in targets if t in data_set]  # O(1) lookup
```

---

## 11. Exercises — Three Levels: Beginner, Intermediate, and Advanced

### Beginner Level (Basic Array and String Operations)

**Problem B1: Find the maximum and minimum of an array in a single pass**

Implement a function that finds both the maximum and minimum simultaneously.
Do not use sorting; complete it in a single O(n) pass.

```python
def find_min_max(arr: list) -> tuple:
    """
    Find the minimum and maximum of an array in a single pass
    Return: (min_val, max_val)
    Constraint: arr is non-empty
    """
    # Implement here
    pass


# Test cases
assert find_min_max([3, 1, 4, 1, 5, 9, 2, 6]) == (1, 9)
assert find_min_max([42]) == (42, 42)
assert find_min_max([-5, -1, -10, -3]) == (-10, -1)
```

<details>
<summary>Solution (click to expand)</summary>

```python
def find_min_max(arr: list) -> tuple:
    min_val = max_val = arr[0]
    for x in arr[1:]:
        if x < min_val:
            min_val = x
        elif x > max_val:
            max_val = x
    return (min_val, max_val)
```

</details>

**Problem B2: Count character frequencies in a string**

Implement a function that returns a dictionary of character occurrence counts in a string.
Case-insensitive. Exclude spaces and symbols.

```python
def char_frequency(s: str) -> dict:
    """
    Return the frequency of alphanumeric characters (unified to lowercase)
    """
    # Implement here
    pass


# Test cases
assert char_frequency("Hello, World!") == {
    'h': 1, 'e': 1, 'l': 3, 'o': 2, 'w': 1, 'r': 1, 'd': 1
}
assert char_frequency("") == {}
```

<details>
<summary>Solution (click to expand)</summary>

```python
def char_frequency(s: str) -> dict:
    freq = {}
    for c in s.lower():
        if c.isalnum():
            freq[c] = freq.get(c, 0) + 1
    return freq
```

</details>

**Problem B3: Determine if two strings are rotations of each other**

Determine if string t can be obtained by rotating string s some number of times to the left.
Hint: Is t contained in s+s?

```python
def is_rotation(s: str, t: str) -> bool:
    """
    Determine if t can be obtained by rotating s
    Example: "waterbottle" is a rotation of "erbottlewat" -> True
    """
    # Implement here
    pass


# Test cases
assert is_rotation("waterbottle", "erbottlewat") is True
assert is_rotation("abc", "cab") is True
assert is_rotation("abc", "acb") is False
assert is_rotation("", "") is True
```

<details>
<summary>Solution (click to expand)</summary>

```python
def is_rotation(s: str, t: str) -> bool:
    if len(s) != len(t):
        return False
    if len(s) == 0:
        return True
    return t in (s + s)
```

</details>

---

### Intermediate Level (Combining Algorithms)

**Problem A1: Product of array except self**

Given an array nums, return an array where `result[i]` is the product of all elements
except `nums[i]`. Solve in O(n) without using division.

```python
def product_except_self(nums: list) -> list:
    """
    LeetCode 238: Product of Array Except Self
    No division, O(n) time, O(1) space (excluding result array)
    """
    # Implement here
    pass


# Test cases
assert product_except_self([1, 2, 3, 4]) == [24, 12, 8, 6]
assert product_except_self([-1, 1, 0, -3, 3]) == [0, 0, 9, 0, 0]
```

<details>
<summary>Solution (click to expand)</summary>

```python
def product_except_self(nums: list) -> list:
    n = len(nums)
    result = [1] * n

    # Left cumulative product
    left_product = 1
    for i in range(n):
        result[i] = left_product
        left_product *= nums[i]

    # Multiply by right cumulative product
    right_product = 1
    for i in range(n - 1, -1, -1):
        result[i] *= right_product
        right_product *= nums[i]

    return result
```

</details>

**Problem A2: Maximum subarray sum (Kadane's Algorithm)**

Find the contiguous subarray with the maximum sum.

```python
def max_subarray_sum(nums: list) -> int:
    """
    LeetCode 53: Maximum Subarray
    Kadane's Algorithm — O(n) time, O(1) space
    """
    # Implement here
    pass


# Test cases
assert max_subarray_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6  # [4,-1,2,1]
assert max_subarray_sum([1]) == 1
assert max_subarray_sum([-1, -2, -3]) == -1
```

<details>
<summary>Solution (click to expand)</summary>

```python
def max_subarray_sum(nums: list) -> int:
    max_sum = current_sum = nums[0]
    for x in nums[1:]:
        current_sum = max(x, current_sum + x)
        max_sum = max(max_sum, current_sum)
    return max_sum
```

</details>

**Problem A3: Longest palindromic substring**

Return the longest palindromic substring within string s.

```python
def longest_palindrome_substring(s: str) -> str:
    """
    LeetCode 5: Longest Palindromic Substring
    Center expansion method — O(n^2) time, O(1) space
    """
    # Implement here
    pass


# Test cases
assert longest_palindrome_substring("babad") in ("bab", "aba")
assert longest_palindrome_substring("cbbd") == "bb"
assert longest_palindrome_substring("a") == "a"
```

<details>
<summary>Solution (click to expand)</summary>

```python
def longest_palindrome_substring(s: str) -> str:
    if len(s) < 2:
        return s

    start, max_len = 0, 1

    def expand(left: int, right: int) -> None:
        nonlocal start, max_len
        while left >= 0 and right < len(s) and s[left] == s[right]:
            if right - left + 1 > max_len:
                start = left
                max_len = right - left + 1
            left -= 1
            right += 1

    for i in range(len(s)):
        expand(i, i)      # Odd-length palindrome
        expand(i, i + 1)  # Even-length palindrome

    return s[start:start + max_len]
```

</details>

---

### Advanced Level (Sophisticated Algorithm Design)

**Problem E1: Trapping Rain Water**

Given an array of heights, find the total amount of water that can be trapped after rain.

```python
def trap_rainwater(height: list) -> int:
    """
    LeetCode 42: Trapping Rain Water
    Two Pointers — O(n) time, O(1) space
    """
    # Implement here
    pass


# Test cases
assert trap_rainwater([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]) == 6
assert trap_rainwater([4, 2, 0, 3, 2, 5]) == 9
assert trap_rainwater([]) == 0
```

<details>
<summary>Solution (click to expand)</summary>

```python
def trap_rainwater(height: list) -> int:
    if not height:
        return 0

    left, right = 0, len(height) - 1
    left_max = right_max = 0
    water = 0

    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                water += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                water += right_max - height[right]
            right -= 1

    return water
```

</details>

**Problem E2: Minimum window substring**

Given strings s and t, return the minimum substring of s that contains all characters of t.

```python
def min_window(s: str, t: str) -> str:
    """
    LeetCode 76: Minimum Window Substring
    Sliding Window — O(n + m) time
    """
    # Implement here
    pass


# Test cases
assert min_window("ADOBECODEBANC", "ABC") == "BANC"
assert min_window("a", "a") == "a"
assert min_window("a", "aa") == ""
```

<details>
<summary>Solution (click to expand)</summary>

```python
from collections import Counter

def min_window(s: str, t: str) -> str:
    if not s or not t or len(s) < len(t):
        return ""

    t_count = Counter(t)
    required = len(t_count)       # Number of unique characters in t
    formed = 0                     # Number of character types satisfying condition
    window_counts = {}

    ans = (float('inf'), 0, 0)     # (length, left, right)
    left = 0

    for right in range(len(s)):
        char = s[right]
        window_counts[char] = window_counts.get(char, 0) + 1

        if char in t_count and window_counts[char] == t_count[char]:
            formed += 1

        while left <= right and formed == required:
            if right - left + 1 < ans[0]:
                ans = (right - left + 1, left, right)

            left_char = s[left]
            window_counts[left_char] -= 1
            if left_char in t_count and window_counts[left_char] < t_count[left_char]:
                formed -= 1
            left += 1

    return "" if ans[0] == float('inf') else s[ans[1]:ans[2] + 1]
```

</details>

**Problem E3: String permutations**

List all permutations of a string without duplicates.

```python
def string_permutations(s: str) -> list:
    """
    Return all permutations of the string sorted lexicographically
    Handle duplicate characters without producing duplicate permutations
    """
    # Implement here
    pass


# Test cases
assert string_permutations("abc") == ["abc", "acb", "bac", "bca", "cab", "cba"]
assert string_permutations("aab") == ["aab", "aba", "baa"]
assert string_permutations("a") == ["a"]
```

<details>
<summary>Solution (click to expand)</summary>

```python
def string_permutations(s: str) -> list:
    result = []
    chars = sorted(s)

    def backtrack(path: list, remaining: list) -> None:
        if not remaining:
            result.append("".join(path))
            return
        for i in range(len(remaining)):
            # Skip duplicates
            if i > 0 and remaining[i] == remaining[i - 1]:
                continue
            backtrack(
                path + [remaining[i]],
                remaining[:i] + remaining[i + 1:]
            )

    backtrack([], chars)
    return result
```

</details>

---

## 12. FAQ — Frequently Asked Questions

### Q1: How much extra memory does Python's list use?

**A:** Python's `list` internally holds an array of element pointers. Each element
counts as an 8-byte pointer, plus the list itself has a header of about 56 bytes.
The growth rate is approximately 1.125x (precisely `new_size = old_size + (old_size >> 3) + 6`),
which is more conservative than Java's `ArrayList` (1.5x) or C++'s `vector` (2x).

The memory usage of a list with capacity n is approximately `56 + 8n` bytes (not including
the size of the objects the pointers point to). For an integer list `[0, 1, ..., 99]`,
the list body = 56 + 800 = 856 bytes, int objects (28 bytes each) = 2800 bytes,
for a total of about 3.6 KB. Storing the same data in a `numpy.array` gives
100 * 8 + header = about 900 bytes, roughly 4x more memory-efficient.

### Q2: What are the advantages and disadvantages of string immutability?

**A:**

**Advantages:**
1. **Hash value caching**: When using strings as dictionary keys, the hash value can be
   computed once and reused. This is why Python's `dict` is fast with string keys.
2. **Thread safety**: Safe to read from multiple threads simultaneously. No locks needed.
3. **String interning**: Automatically share strings with identical content to save memory.
   Python automatically interns short strings and identifier-like strings.
4. **Elimination of side effects**: Guarantee that the original string is not modified
   when passed to a function.

**Disadvantages:**
1. **A new object is created with every modification**: Particularly noticeable with
   concatenation in loops.
2. **Memory fragmentation**: GC load increases from creating and destroying many short strings.

### Q3: Selection criteria between numpy arrays and Python lists?

**A:** Use the following criteria:

| Criterion | Python list | numpy ndarray |
|-----------|------------|---------------|
| Element types | Mixed allowed | Single type only |
| Resizing | append/pop are fast | Resizing is awkward |
| Numerical operations | Slow (loop required) | Fast (vectorized) |
| Memory efficiency | Low (pointer array) | High (values stored directly) |
| Broadcasting | None | Powerful support |
| Slicing | Returns a copy | Returns a view (shared memory) |

**Conclusion**: Use numpy for homogeneous numeric arrays with arithmetic operations.
Use list when types are mixed, sizes change frequently, or data is non-numeric.

### Q4: Does array size affect sorting algorithm selection?

**A:** It has a significant impact. Python's `sorted()` / `list.sort()` uses Tim Sort,
which switches to insertion sort for small n (typically 64 elements or fewer). Insertion sort
is O(n^2) but has good cache efficiency and low overhead, making it faster than
O(n log n) algorithms for small data. C++'s `std::sort` similarly falls back to
insertion sort for small sizes.

### Q5: How to choose between Two Pointers and Sliding Window?

**A:**

- **Two Pointers**: Primarily used with sorted arrays. Two pointers move independently
  based on conditions. Typical examples: pair search in sorted arrays, palindrome detection.
- **Sliding Window**: Targets contiguous subarrays/substrings. Both left and right
  window ends move in the same direction. Typical examples: maximum sum subarray,
  longest/shortest substring satisfying a condition.

Decision criteria:
1. If the problem is about "contiguous subarrays/substrings" -> Sliding Window
2. If the problem is about "combinations of two elements" -> Two Pointers
3. If the array is sorted and you're searching for pairs -> Opposite Two Pointers

### Q6: Tips for avoiding boundary condition mistakes in matrix traversal problems?

**A:** Three techniques:

1. **Manage 4 boundary variables explicitly**: Use `top`, `bottom`, `left`, `right`,
   and update the corresponding boundary after traversing each edge.
2. **Check remaining before traversal**: Before traversing the bottom edge, check
   `top <= bottom`; before the left edge, check `left <= right`. Forgetting this
   causes double-counting of elements.
3. **Trace with small examples**: Manually trace with 1x1, 1xN, Nx1, 2x2 matrices
   to verify boundary conditions are correct.

---


## FAQ

### Q1: What is the most important point in learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory, but by actually writing code and verifying behavior.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this applied in practice?

Knowledge of this topic is frequently used in everyday development work. It becomes particularly important during code reviews and architecture design.

---

## 13. Summary and Next Steps

### What You Learned in This Chapter

| Topic | Key Points |
|-------|------------|
| Static arrays | Stored in contiguous memory, O(1) random access, fixed size |
| Dynamic arrays | append is amortized O(1), growth rate affects space efficiency, shrinking strategy matters too |
| Multidimensional arrays | Cache efficiency differs between row-major/column-major, be aware of traversal order |
| Strings | Immutable in many languages, use join for concatenation, be aware of encoding |
| Two Pointers | 3 patterns: opposite, same direction, different speed; improves O(n^2) to O(n) |
| Sliding Window | 2 types: fixed-size and variable-size; applies to contiguous subarray problems |
| 2D array traversal | Spiral, diagonal, zigzag, etc.; boundary management is key |

### Frequently Tested LeetCode Problems

| Difficulty | Number | Problem Name | Pattern |
|------------|--------|-------------|---------|
| Easy | 1 | Two Sum | Hash Map |
| Easy | 26 | Remove Duplicates | Two Pointers |
| Easy | 14 | Longest Common Prefix | Vertical scan |
| Easy | 283 | Move Zeroes | Two Pointers |
| Medium | 3 | Longest Substring Without Repeating | Sliding Window |
| Medium | 15 | 3Sum | Sort + Two Pointers |
| Medium | 48 | Rotate Image | Transpose + Reverse |
| Medium | 54 | Spiral Matrix | Boundary management |
| Medium | 238 | Product of Array Except Self | Prefix/Suffix |
| Medium | 438 | Find All Anagrams | Sliding Window |
| Hard | 42 | Trapping Rain Water | Two Pointers |
| Hard | 76 | Minimum Window Substring | Sliding Window |

### Recommended Next Guides

- [Linked Lists — Singly/Doubly Linked and Floyd's Algorithm](./01-linked-lists.md)
- [Stacks and Queues — LIFO/FIFO and Monotonic Stacks](./02-stacks-queues.md)
- [Hash Tables — Hash Functions and Collision Resolution](./03-hash-tables.md)

---

## 14. References

1. Cormen, T.H., Leiserson, C.E., Rivest, R.L. & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. — Chapter 2 "Getting Started" covers the basics of array merging and sorting; Chapter 17 "Amortized Analysis" provides detailed analysis of dynamic array amortization.

2. Sedgewick, R. & Wayne, K. (2011). *Algorithms* (4th ed.). Addison-Wesley. — Detailed implementation of arrays and sorting using Java. Rich visualizations available on the website (algs4.cs.princeton.edu).

3. Skiena, S.S. (2020). *The Algorithm Design Manual* (3rd ed.). Springer. — Practical design patterns for array and string algorithms, including interview-relevant "War Stories."

4. Python Software Foundation. "Time Complexity." Python Wiki. https://wiki.python.org/moin/TimeComplexity — Official reference for operation-specific complexity of Python's built-in data structures.

5. Knuth, D.E. (1997). *The Art of Computer Programming, Volume 1: Fundamental Algorithms* (3rd ed.). Addison-Wesley. — Mathematical foundations of arrays and linear lists.

6. CPython Source Code. `Objects/listobject.c`. https://github.com/python/cpython — The actual source code of Python list's dynamic array implementation. The growth rate formula `new_allocated = (size_t)newsize + (newsize >> 3) + (newsize < 9 ? 3 : 6)` can be verified here.

7. Bentley, J. (1986). *Programming Pearls*. Addison-Wesley. — The classic book that introduced the array rotation algorithm (reversal technique). Presented in Chapter 2 "Aha! Algorithms."

---

> **Disclaimer**: All code examples in this guide are created for educational purposes. For production code, using optimized implementations provided by each language's standard library is recommended.

---

## Recommended Next Guides

- [Linked Lists — Singly, Doubly, Circular, and Floyd's Algorithm](./01-linked-lists.md) - Proceed to the next topic

---

## References

- [MDN Web Docs](https://developer.mozilla.org/) - Web technology reference
- [Wikipedia](https://en.wikipedia.org/) - Overview of technical concepts
