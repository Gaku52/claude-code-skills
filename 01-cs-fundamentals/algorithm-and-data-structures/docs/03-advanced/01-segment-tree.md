# Segment Tree

> Systematically understand tree structures that handle range queries and point updates in O(log n), through basic implementation, lazy propagation, and BIT

## What You Will Learn in This Chapter

1. **Segment tree structure and basic operations** (construction, range query, point update) with O(log n) implementation
2. **Lazy Propagation** to extend range updates to O(log n)
3. **BIT (Binary Indexed Tree / Fenwick Tree)** comparison to choose the right tool for the job
4. **Abstract segment tree** to support arbitrary monoid operations
5. **Persistent segment tree, merge sort tree**, and other advanced variations


## Prerequisites

The following knowledge will help deepen your understanding before reading this guide:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding of [Union-Find (Disjoint Set Data Structure)](./00-union-find.md)

---

## 1. Concept of Segment Trees

A segment tree is a complete binary tree designed to efficiently handle "range queries" and "element updates" on an array. Each element of the array is placed at a leaf, and each internal node holds the result of an operation (sum, minimum, maximum, GCD, etc.) over the range covered by its child nodes.

```
Array: [2, 1, 5, 3, 4, 2, 1, 6]

Segment tree (range sum):
                  [24]              <- total sum
               /        \
          [11]             [13]     <- first half / second half sums
         /    \           /    \
      [3]     [8]     [6]     [7]
     / \     / \     / \     / \
   [2] [1] [5] [3] [4] [2] [1] [6]  <- leaves = original array

Query for sum of range [1, 5):
  -> [1] + [5,3] + [4] = 1 + 8 + 4 = 13
  -> answered by accessing 3 nodes (O(log n))

Naive array approach:
  -> 1 + 5 + 3 + 4 = 13
  -> scans 4 elements (O(n))
```

### Why Segment Trees Matter

```
Scenarios where segment trees are needed:

1. Range queries on dynamic arrays
   -> Values are frequently updated, and aggregate values (sum, min, etc.) over ranges must be computed each time

2. Comparison with naive approaches:
   Operation        | Array    | Prefix Sum | Segment Tree
   Point update     | O(1)     | O(n)       | O(log n)
   Range query      | O(n)     | O(1)       | O(log n)

   -> Segment trees are optimal when both updates and queries are frequent

3. Concrete use cases:
   - Real-time min/max queries on stock prices over intervals
   - Game score rankings (update + rank query)
   - Internal implementation of database range queries
   - Range-based problems in competitive programming
```

---

## 2. Basic Implementation (Range Sum)

```python
class SegmentTree:
    """Segment tree (range sum query + point update)"""

    def __init__(self, data: list):
        self.n = len(data)
        self.tree = [0] * (4 * self.n)  # sufficient size
        self._build(data, 1, 0, self.n - 1)

    def _build(self, data, node, start, end):
        """Build in O(n)"""
        if start == end:
            self.tree[node] = data[start]
            return
        mid = (start + end) // 2
        self._build(data, 2 * node, start, mid)
        self._build(data, 2 * node + 1, mid + 1, end)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def update(self, idx: int, val: int):
        """Point update - O(log n)"""
        self._update(1, 0, self.n - 1, idx, val)

    def _update(self, node, start, end, idx, val):
        if start == end:
            self.tree[node] = val
            return
        mid = (start + end) // 2
        if idx <= mid:
            self._update(2 * node, start, mid, idx, val)
        else:
            self._update(2 * node + 1, mid + 1, end, idx, val)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def query(self, l: int, r: int) -> int:
        """Return sum of range [l, r] - O(log n)"""
        return self._query(1, 0, self.n - 1, l, r)

    def _query(self, node, start, end, l, r):
        if r < start or end < l:
            return 0  # out of range
        if l <= start and end <= r:
            return self.tree[node]  # fully contained
        mid = (start + end) // 2
        left_sum = self._query(2 * node, start, mid, l, r)
        right_sum = self._query(2 * node + 1, mid + 1, end, l, r)
        return left_sum + right_sum

# Usage example
data = [2, 1, 5, 3, 4, 2, 1, 6]
st = SegmentTree(data)
print(st.query(1, 4))  # 13 (1+5+3+4)
print(st.query(0, 7))  # 24 (total sum)

st.update(2, 10)        # data[2] = 5 -> 10
print(st.query(1, 4))  # 18 (1+10+3+4)
```

### Iterative Segment Tree (Faster Version)

The non-recursive implementation has a smaller constant factor and is practically faster in competitive programming.

```python
class SegmentTreeIterative:
    """Iterative segment tree (range sum) - smaller constant factor"""

    def __init__(self, data: list):
        self.n = len(data)
        self.size = 1
        while self.size < self.n:
            self.size <<= 1
        self.tree = [0] * (2 * self.size)

        # Place data at the leaves
        for i in range(self.n):
            self.tree[self.size + i] = data[i]

        # Build bottom-up
        for i in range(self.size - 1, 0, -1):
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]

    def update(self, idx: int, val: int):
        """Point update - O(log n)"""
        idx += self.size
        self.tree[idx] = val
        idx >>= 1
        while idx >= 1:
            self.tree[idx] = self.tree[2 * idx] + self.tree[2 * idx + 1]
            idx >>= 1

    def query(self, l: int, r: int) -> int:
        """Sum of range [l, r) - O(log n)"""
        result = 0
        l += self.size
        r += self.size
        while l < r:
            if l & 1:
                result += self.tree[l]
                l += 1
            if r & 1:
                r -= 1
                result += self.tree[r]
            l >>= 1
            r >>= 1
        return result

# Usage example (note: half-open interval [l, r))
data = [2, 1, 5, 3, 4, 2, 1, 6]
st = SegmentTreeIterative(data)
print(st.query(1, 5))  # 13 (1+5+3+4) <- [1, 5) = index 1,2,3,4
print(st.query(0, 8))  # 24 (total)
```

### C++ Implementation (Iterative Version)

```cpp
#include <vector>
#include <functional>

template <typename T>
class SegmentTree {
    int n;
    std::vector<T> tree;
    T identity;
    std::function<T(T, T)> op;

public:
    SegmentTree(int n, T identity, std::function<T(T, T)> op)
        : n(n), tree(2 * n, identity), identity(identity), op(op) {}

    SegmentTree(const std::vector<T>& data, T identity, std::function<T(T, T)> op)
        : SegmentTree(data.size(), identity, op) {
        for (int i = 0; i < (int)data.size(); i++)
            tree[n + i] = data[i];
        for (int i = n - 1; i > 0; i--)
            tree[i] = op(tree[2 * i], tree[2 * i + 1]);
    }

    void update(int idx, T val) {
        tree[idx += n] = val;
        for (idx >>= 1; idx >= 1; idx >>= 1)
            tree[idx] = op(tree[2 * idx], tree[2 * idx + 1]);
    }

    T query(int l, int r) {  // [l, r)
        T left_result = identity, right_result = identity;
        for (l += n, r += n; l < r; l >>= 1, r >>= 1) {
            if (l & 1) left_result = op(left_result, tree[l++]);
            if (r & 1) right_result = op(tree[--r], right_result);
        }
        return op(left_result, right_result);
    }
};

// Usage examples
// SegmentTree<int> st(data, 0, { return a + b; });  // range sum
// SegmentTree<int> st(data, INT_MAX, { return min(a, b); });  // range min
```

---

## 3. Range Minimum Query (RMQ)

```python
class SegmentTreeMin:
    """Segment tree (range minimum query)"""

    def __init__(self, data: list):
        self.n = len(data)
        self.tree = [float('inf')] * (4 * self.n)
        self._build(data, 1, 0, self.n - 1)

    def _build(self, data, node, start, end):
        if start == end:
            self.tree[node] = data[start]
            return
        mid = (start + end) // 2
        self._build(data, 2 * node, start, mid)
        self._build(data, 2 * node + 1, mid + 1, end)
        self.tree[node] = min(self.tree[2 * node], self.tree[2 * node + 1])

    def query(self, l: int, r: int) -> int:
        return self._query(1, 0, self.n - 1, l, r)

    def _query(self, node, start, end, l, r):
        if r < start or end < l:
            return float('inf')
        if l <= start and end <= r:
            return self.tree[node]
        mid = (start + end) // 2
        return min(
            self._query(2 * node, start, mid, l, r),
            self._query(2 * node + 1, mid + 1, end, l, r)
        )

    def update(self, idx: int, val: int):
        self._update(1, 0, self.n - 1, idx, val)

    def _update(self, node, start, end, idx, val):
        if start == end:
            self.tree[node] = val
            return
        mid = (start + end) // 2
        if idx <= mid:
            self._update(2 * node, start, mid, idx, val)
        else:
            self._update(2 * node + 1, mid + 1, end, idx, val)
        self.tree[node] = min(self.tree[2 * node], self.tree[2 * node + 1])

data = [5, 2, 8, 1, 9, 3, 7, 4]
st_min = SegmentTreeMin(data)
print(st_min.query(0, 3))  # 1
print(st_min.query(4, 7))  # 3
```

### Range Minimum with Index

```python
class SegmentTreeMinIndex:
    """Segment tree that returns the range minimum and its index"""

    def __init__(self, data: list):
        self.n = len(data)
        # tree[i] = (value, index) pair
        self.tree = [(float('inf'), -1)] * (4 * self.n)
        self._build(data, 1, 0, self.n - 1)

    def _merge(self, a, b):
        """Return the smaller one (prefer smaller index on tie)"""
        if a[0] < b[0]:
            return a
        elif a[0] > b[0]:
            return b
        else:
            return a if a[1] <= b[1] else b

    def _build(self, data, node, start, end):
        if start == end:
            self.tree[node] = (data[start], start)
            return
        mid = (start + end) // 2
        self._build(data, 2 * node, start, mid)
        self._build(data, 2 * node + 1, mid + 1, end)
        self.tree[node] = self._merge(self.tree[2 * node], self.tree[2 * node + 1])

    def query(self, l: int, r: int) -> tuple:
        """Return the minimum value and its position in range [l, r]"""
        return self._query(1, 0, self.n - 1, l, r)

    def _query(self, node, start, end, l, r):
        if r < start or end < l:
            return (float('inf'), -1)
        if l <= start and end <= r:
            return self.tree[node]
        mid = (start + end) // 2
        left = self._query(2 * node, start, mid, l, r)
        right = self._query(2 * node + 1, mid + 1, end, l, r)
        return self._merge(left, right)

data = [5, 2, 8, 1, 9, 3, 7, 4]
st = SegmentTreeMinIndex(data)
val, idx = st.query(0, 7)
print(f"Min: {val}, Position: {idx}")  # Min: 1, Position: 3
val, idx = st.query(4, 7)
print(f"Min: {val}, Position: {idx}")  # Min: 3, Position: 5
```

---

## 4. Abstract Segment Tree (Monoid)

A generic segment tree that supports any associative binary operation. Any monoid (an associative binary operation + identity element) can be placed on a segment tree.

```python
class AbstractSegmentTree:
    """Abstract segment tree - supports arbitrary monoid operations
    op: binary operation (must satisfy associativity)
    e: identity element (op(a, e) = op(e, a) = a)
    """

    def __init__(self, data: list, op, e):
        self.n = len(data)
        self.op = op
        self.e = e
        self.size = 1
        while self.size < self.n:
            self.size <<= 1
        self.tree = [e] * (2 * self.size)

        # Place data at the leaves
        for i in range(self.n):
            self.tree[self.size + i] = data[i]

        # Build bottom-up
        for i in range(self.size - 1, 0, -1):
            self.tree[i] = self.op(self.tree[2 * i], self.tree[2 * i + 1])

    def update(self, idx: int, val):
        """Point update"""
        idx += self.size
        self.tree[idx] = val
        idx >>= 1
        while idx >= 1:
            self.tree[idx] = self.op(self.tree[2 * idx], self.tree[2 * idx + 1])
            idx >>= 1

    def query(self, l: int, r: int):
        """Query over range [l, r)"""
        left_result = self.e
        right_result = self.e
        l += self.size
        r += self.size
        while l < r:
            if l & 1:
                left_result = self.op(left_result, self.tree[l])
                l += 1
            if r & 1:
                r -= 1
                right_result = self.op(self.tree[r], right_result)
            l >>= 1
            r >>= 1
        return self.op(left_result, right_result)

# --- Usage examples with various monoids ---

data = [2, 1, 5, 3, 4, 2, 1, 6]

# Range sum (sum, 0)
st_sum = AbstractSegmentTree(data, lambda a, b: a + b, 0)
print(st_sum.query(1, 5))  # 13

# Range minimum (min, inf)
st_min = AbstractSegmentTree(data, min, float('inf'))
print(st_min.query(0, 8))  # 1

# Range maximum (max, -inf)
st_max = AbstractSegmentTree(data, max, float('-inf'))
print(st_max.query(0, 8))  # 6

# Range GCD (gcd, 0)
from math import gcd
data_gcd = [12, 18, 24, 36]
st_gcd = AbstractSegmentTree(data_gcd, gcd, 0)
print(st_gcd.query(0, 4))  # 6

# Range XOR (xor, 0)
st_xor = AbstractSegmentTree(data, lambda a, b: a ^ b, 0)
print(st_xor.query(0, 8))  # 2^1^5^3^4^2^1^6 = 0

# Range product (multiplication, 1) + MOD
MOD = 10**9 + 7
st_prod = AbstractSegmentTree(data, lambda a, b: (a * b) % MOD, 1)
print(st_prod.query(0, 4))  # 2*1*5*3 = 30
```

---

## 5. Lazy Propagation

Handles range updates (e.g., adding v to all elements in range [l,r]) in O(log n). The core idea of lazy propagation is "defer updates until they are actually needed."

```
Lazy propagation for range addition:

Before addition:   [24]
                  /    \
              [11]      [13]
             /  \      /   \
           [3]  [8] [6]   [7]

Add +3 to range [2,5]:
  -> Record +3 in the lazy values of affected nodes
  -> Actual propagation is deferred until needed

          [24+12=36]
            /      \
      [11+6=17]   [13+6=19]
       /    \      /     \
     [3]  [8+6] [6+6]  [7]
           lazy   lazy
           =+3   =+3
```

```python
class LazySegmentTree:
    """Segment tree with lazy propagation (range addition + range sum query)"""

    def __init__(self, data: list):
        self.n = len(data)
        self.tree = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        self._build(data, 1, 0, self.n - 1)

    def _build(self, data, node, start, end):
        if start == end:
            self.tree[node] = data[start]
            return
        mid = (start + end) // 2
        self._build(data, 2 * node, start, mid)
        self._build(data, 2 * node + 1, mid + 1, end)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def _push_down(self, node, start, end):
        """Propagate lazy values to children"""
        if self.lazy[node] != 0:
            mid = (start + end) // 2
            left_len = mid - start + 1
            right_len = end - mid

            self.tree[2 * node] += self.lazy[node] * left_len
            self.tree[2 * node + 1] += self.lazy[node] * right_len

            self.lazy[2 * node] += self.lazy[node]
            self.lazy[2 * node + 1] += self.lazy[node]

            self.lazy[node] = 0

    def range_update(self, l: int, r: int, val: int):
        """Add val to range [l, r] - O(log n)"""
        self._range_update(1, 0, self.n - 1, l, r, val)

    def _range_update(self, node, start, end, l, r, val):
        if r < start or end < l:
            return
        if l <= start and end <= r:
            self.tree[node] += val * (end - start + 1)
            self.lazy[node] += val
            return
        self._push_down(node, start, end)
        mid = (start + end) // 2
        self._range_update(2 * node, start, mid, l, r, val)
        self._range_update(2 * node + 1, mid + 1, end, l, r, val)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def query(self, l: int, r: int) -> int:
        """Sum of range [l, r] - O(log n)"""
        return self._query(1, 0, self.n - 1, l, r)

    def _query(self, node, start, end, l, r):
        if r < start or end < l:
            return 0
        if l <= start and end <= r:
            return self.tree[node]
        self._push_down(node, start, end)
        mid = (start + end) // 2
        return (self._query(2 * node, start, mid, l, r) +
                self._query(2 * node + 1, mid + 1, end, l, r))

# Usage example
data = [1, 3, 5, 7, 9, 11]
lst = LazySegmentTree(data)
print(lst.query(1, 3))     # 15 (3+5+7)
lst.range_update(1, 4, 10)  # add +10 to range [1,4]
print(lst.query(1, 3))     # 45 (13+15+17)
```

### Lazy Propagation: Range Assignment + Range Sum Query

```python
class LazySegmentTreeAssign:
    """Segment tree with lazy propagation (range assignment + range sum query)
    Replace all elements in range [l, r] with val
    """

    def __init__(self, data: list):
        self.n = len(data)
        self.tree = [0] * (4 * self.n)
        self.lazy = [None] * (4 * self.n)  # None = not propagated
        self._build(data, 1, 0, self.n - 1)

    def _build(self, data, node, start, end):
        if start == end:
            self.tree[node] = data[start]
            return
        mid = (start + end) // 2
        self._build(data, 2 * node, start, mid)
        self._build(data, 2 * node + 1, mid + 1, end)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def _push_down(self, node, start, end):
        if self.lazy[node] is not None:
            mid = (start + end) // 2
            left_len = mid - start + 1
            right_len = end - mid

            self.tree[2 * node] = self.lazy[node] * left_len
            self.tree[2 * node + 1] = self.lazy[node] * right_len

            self.lazy[2 * node] = self.lazy[node]
            self.lazy[2 * node + 1] = self.lazy[node]

            self.lazy[node] = None

    def range_assign(self, l: int, r: int, val: int):
        """Assign val to range [l, r] - O(log n)"""
        self._range_assign(1, 0, self.n - 1, l, r, val)

    def _range_assign(self, node, start, end, l, r, val):
        if r < start or end < l:
            return
        if l <= start and end <= r:
            self.tree[node] = val * (end - start + 1)
            self.lazy[node] = val
            return
        self._push_down(node, start, end)
        mid = (start + end) // 2
        self._range_assign(2 * node, start, mid, l, r, val)
        self._range_assign(2 * node + 1, mid + 1, end, l, r, val)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def query(self, l: int, r: int) -> int:
        return self._query(1, 0, self.n - 1, l, r)

    def _query(self, node, start, end, l, r):
        if r < start or end < l:
            return 0
        if l <= start and end <= r:
            return self.tree[node]
        self._push_down(node, start, end)
        mid = (start + end) // 2
        return (self._query(2 * node, start, mid, l, r) +
                self._query(2 * node + 1, mid + 1, end, l, r))

# Usage example
data = [1, 2, 3, 4, 5]
lst = LazySegmentTreeAssign(data)
print(lst.query(0, 4))     # 15 (1+2+3+4+5)
lst.range_assign(1, 3, 10)  # [1, 10, 10, 10, 5]
print(lst.query(0, 4))     # 36 (1+10+10+10+5)
```

### Lazy Propagation: Range Addition + Range Minimum Query

```python
class LazySegmentTreeMinAdd:
    """Segment tree with lazy propagation (range addition + range minimum query)"""

    def __init__(self, data: list):
        self.n = len(data)
        self.tree = [float('inf')] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        self._build(data, 1, 0, self.n - 1)

    def _build(self, data, node, start, end):
        if start == end:
            self.tree[node] = data[start]
            return
        mid = (start + end) // 2
        self._build(data, 2 * node, start, mid)
        self._build(data, 2 * node + 1, mid + 1, end)
        self.tree[node] = min(self.tree[2 * node], self.tree[2 * node + 1])

    def _push_down(self, node):
        if self.lazy[node] != 0:
            for child in [2 * node, 2 * node + 1]:
                self.tree[child] += self.lazy[node]
                self.lazy[child] += self.lazy[node]
            self.lazy[node] = 0

    def range_add(self, l: int, r: int, val: int):
        self._range_add(1, 0, self.n - 1, l, r, val)

    def _range_add(self, node, start, end, l, r, val):
        if r < start or end < l:
            return
        if l <= start and end <= r:
            self.tree[node] += val
            self.lazy[node] += val
            return
        self._push_down(node)
        mid = (start + end) // 2
        self._range_add(2 * node, start, mid, l, r, val)
        self._range_add(2 * node + 1, mid + 1, end, l, r, val)
        self.tree[node] = min(self.tree[2 * node], self.tree[2 * node + 1])

    def query_min(self, l: int, r: int):
        return self._query_min(1, 0, self.n - 1, l, r)

    def _query_min(self, node, start, end, l, r):
        if r < start or end < l:
            return float('inf')
        if l <= start and end <= r:
            return self.tree[node]
        self._push_down(node)
        mid = (start + end) // 2
        return min(
            self._query_min(2 * node, start, mid, l, r),
            self._query_min(2 * node + 1, mid + 1, end, l, r)
        )

data = [5, 2, 8, 1, 9, 3]
lst = LazySegmentTreeMinAdd(data)
print(lst.query_min(0, 5))     # 1
lst.range_add(2, 4, -5)        # [5, 2, 3, -4, 4, 3]
print(lst.query_min(0, 5))     # -4
```

---

## 6. BIT (Binary Indexed Tree / Fenwick Tree)

Performs prefix sum computation and point updates in O(log n). Simpler to implement and more memory-efficient than a segment tree.

```
Array:  [3, 2, 5, 1, 4, 7, 2, 6]
index:  1  2  3  4  5  6  7  8

BIT structure (1-indexed):
tree[1] = a[1]           = 3
tree[2] = a[1]+a[2]      = 5
tree[3] = a[3]           = 5
tree[4] = a[1]+...+a[4]  = 11
tree[5] = a[5]           = 4
tree[6] = a[5]+a[6]      = 11
tree[7] = a[7]           = 2
tree[8] = a[1]+...+a[8]  = 30

Range covered by tree[i] = lowbit(i) = i & (-i) elements
  tree[4]: lowbit(4)=4 -> a[1]~a[4]
  tree[6]: lowbit(6)=2 -> a[5]~a[6]
  tree[7]: lowbit(7)=1 -> a[7]

Parent-child relationship via bit operations:
  index   binary   lowbit  covered range
    1     0001      1      [1,1]
    2     0010      2      [1,2]
    3     0011      1      [3,3]
    4     0100      4      [1,4]
    5     0101      1      [5,5]
    6     0110      2      [5,6]
    7     0111      1      [7,7]
    8     1000      8      [1,8]
```

```python
class BIT:
    """Binary Indexed Tree (Fenwick Tree) - 1-indexed"""

    def __init__(self, n: int):
        self.n = n
        self.tree = [0] * (n + 1)

    @classmethod
    def from_array(cls, data: list):
        """Build from array in O(n)"""
        bit = cls(len(data))
        for i, val in enumerate(data):
            bit.update(i + 1, val)  # 1-indexed
        return bit

    @classmethod
    def from_array_fast(cls, data: list):
        """Build from array in O(n) (fast version)"""
        n = len(data)
        bit = cls(n)
        for i in range(1, n + 1):
            bit.tree[i] += data[i - 1]
            j = i + (i & (-i))
            if j <= n:
                bit.tree[j] += bit.tree[i]
        return bit

    def update(self, i: int, delta: int):
        """Add delta to a[i] - O(log n)"""
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)  # next node

    def prefix_sum(self, i: int) -> int:
        """a[1] + a[2] + ... + a[i] - O(log n)"""
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)  # parent node
        return s

    def range_sum(self, l: int, r: int) -> int:
        """a[l] + ... + a[r] - O(log n)"""
        return self.prefix_sum(r) - self.prefix_sum(l - 1)

    def lower_bound(self, target: int) -> int:
        """Return the smallest i such that prefix_sum(i) >= target
        (binary search on BIT) - O(log n)
        """
        pos = 0
        total = 0
        k = 1
        while k <= self.n:
            k <<= 1
        k >>= 1

        while k > 0:
            if pos + k <= self.n and total + self.tree[pos + k] < target:
                total += self.tree[pos + k]
                pos += k
            k >>= 1

        return pos + 1

# Usage example
data = [3, 2, 5, 1, 4, 7, 2, 6]
bit = BIT.from_array(data)
print(bit.range_sum(2, 5))  # 12 (2+5+1+4)
bit.update(3, 3)             # data[3] += 3 (5->8)
print(bit.range_sum(2, 5))  # 15 (2+8+1+4)
```

### Range Update BIT

```python
class RangeUpdateBIT:
    """BIT supporting range addition + point query / range sum query
    Uses two BITs to achieve range addition
    """

    def __init__(self, n: int):
        self.n = n
        self.bit1 = BIT(n)  # difference array
        self.bit2 = BIT(n)  # correction array

    def range_add(self, l: int, r: int, val: int):
        """Add val to range [l, r]"""
        self.bit1.update(l, val)
        self.bit1.update(r + 1, -val)
        self.bit2.update(l, val * (l - 1))
        self.bit2.update(r + 1, -val * r)

    def prefix_sum(self, i: int) -> int:
        """Sum of a[1] + ... + a[i]"""
        return self.bit1.prefix_sum(i) * i - self.bit2.prefix_sum(i)

    def range_sum(self, l: int, r: int) -> int:
        """Sum of a[l] + ... + a[r]"""
        return self.prefix_sum(r) - self.prefix_sum(l - 1)

    def point_query(self, i: int) -> int:
        """Value of a[i]"""
        return self.bit1.prefix_sum(i)

# Usage example
rbit = RangeUpdateBIT(8)
rbit.range_add(2, 5, 3)  # [0, 3, 3, 3, 3, 0, 0, 0]
print(rbit.point_query(3))   # 3
print(rbit.range_sum(1, 8))  # 12
```

### 2D BIT

```python
class BIT2D:
    """2D Binary Indexed Tree"""

    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.tree = [[0] * (cols + 1) for _ in range(rows + 1)]

    def update(self, r: int, c: int, delta: int):
        """Add delta to (r, c) - O(log R * log C)"""
        i = r
        while i <= self.rows:
            j = c
            while j <= self.cols:
                self.tree[i][j] += delta
                j += j & (-j)
            i += i & (-i)

    def prefix_sum(self, r: int, c: int) -> int:
        """Rectangle sum from (1,1) to (r,c) - O(log R * log C)"""
        s = 0
        i = r
        while i > 0:
            j = c
            while j > 0:
                s += self.tree[i][j]
                j -= j & (-j)
            i -= i & (-i)
        return s

    def range_sum(self, r1: int, c1: int, r2: int, c2: int) -> int:
        """Rectangle sum from (r1,c1) to (r2,c2)"""
        return (self.prefix_sum(r2, c2)
                - self.prefix_sum(r1 - 1, c2)
                - self.prefix_sum(r2, c1 - 1)
                + self.prefix_sum(r1 - 1, c1 - 1))

# Usage example
bit2d = BIT2D(4, 4)
bit2d.update(1, 1, 3)
bit2d.update(2, 3, 5)
bit2d.update(3, 2, 7)
print(bit2d.range_sum(1, 1, 3, 3))  # 15 (3+5+7)
```

---

## 7. Segment Tree vs BIT Comparison

| Property | Segment Tree | BIT |
|:---|:---|:---|
| Space complexity | O(4n) | O(n) |
| Construction | O(n) | O(n log n) (O(n) also possible) |
| Point update | O(log n) | O(log n) |
| Range query | O(log n) | O(log n) |
| Range update | O(log n) (with lazy propagation) | O(log n) (range update BIT) |
| Supported queries | Sum, min, max, GCD, etc. | Primarily sum (invertible operations) |
| Implementation complexity | Moderately complex | Concise |
| Constant factor | Somewhat large | Small |

## Selection Guide by Use Case

| Use Case | Recommended | Reason |
|:---|:---|:---|
| Range sum + point update | BIT | Simple implementation, fast |
| Range min/max | Segment tree | Not supported by BIT |
| Range addition + range sum | Lazy segment tree | Range update required |
| Inversion count | BIT | Coordinate compression + point update |
| 2D range queries | 2D BIT or segment tree | Easy to extend |
| k-th element | BIT (with binary search) | Efficient lower_bound |

---

## 8. Inversion Count (BIT Application)

```python
def count_inversions(arr: list) -> int:
    """Count inversions using BIT in O(n log n)
    Inversion count = number of pairs (i, j) where i < j and arr[i] > arr[j]
    """
    # Coordinate compression
    sorted_unique = sorted(set(arr))
    rank = {v: i + 1 for i, v in enumerate(sorted_unique)}

    bit = BIT(len(sorted_unique))
    inversions = 0

    for i in range(len(arr) - 1, -1, -1):
        # Count elements smaller than arr[i] that appear to the right of arr[i]
        inversions += bit.prefix_sum(rank[arr[i]] - 1)
        bit.update(rank[arr[i]], 1)

    return inversions

data = [5, 3, 2, 4, 1]
print(count_inversions(data))  # 7
# (5,3),(5,2),(5,4),(5,1),(3,2),(3,1),(4,1) = 7
```

### k-th Smallest Element Search (Binary Search on BIT)

```python
def kth_smallest(bit: BIT, k: int) -> int:
    """Return the index of the k-th smallest element on BIT
    BIT[i] = whether value i exists (0 or 1)
    """
    return bit.lower_bound(k)

# Usage example: dynamic k-th element
class DynamicKthElement:
    """Find the k-th element while supporting insertion and deletion"""

    def __init__(self, max_val: int):
        self.bit = BIT(max_val)
        self.count = 0

    def add(self, val: int):
        self.bit.update(val, 1)
        self.count += 1

    def remove(self, val: int):
        self.bit.update(val, -1)
        self.count -= 1

    def kth(self, k: int) -> int:
        """Return the k-th smallest element (1-indexed)"""
        return self.bit.lower_bound(k)

dke = DynamicKthElement(100)
dke.add(10)
dke.add(30)
dke.add(20)
dke.add(50)
print(dke.kth(1))  # 10 (smallest)
print(dke.kth(3))  # 30 (3rd smallest)
dke.remove(20)
print(dke.kth(2))  # 30 (2nd smallest)
```

---

## 9. Segment Tree Applications

### Binary Search on Segment Tree

```python
class SegmentTreeWithSearch(SegmentTree):
    """Binary search on segment tree
    Find the smallest r such that the sum of range [0, r] >= target in O(log n)
    """

    def find_first(self, target: int) -> int:
        """Smallest r where prefix_sum(r) >= target"""
        return self._find_first(1, 0, self.n - 1, target)

    def _find_first(self, node, start, end, target):
        if self.tree[node] < target:
            return -1  # not enough in this subtree
        if start == end:
            return start  # found
        mid = (start + end) // 2
        # If the left child is sufficient, go left
        if self.tree[2 * node] >= target:
            return self._find_first(2 * node, start, mid, target)
        else:
            # Subtract the left portion and search right
            return self._find_first(
                2 * node + 1, mid + 1, end,
                target - self.tree[2 * node]
            )
```

### Longest Increasing Subsequence (LIS) via Segment Tree

```python
def lis_segtree(arr: list) -> int:
    """Compute LIS length using segment tree in O(n log n)"""
    # Coordinate compression
    sorted_unique = sorted(set(arr))
    compress = {v: i for i, v in enumerate(sorted_unique)}
    m = len(sorted_unique)

    # Segment tree for range maximum
    st = AbstractSegmentTree([0] * m, max, 0)

    for val in arr:
        idx = compress[val]
        # Maximum LIS length ending with an element smaller than val
        if idx > 0:
            best = st.query(0, idx)  # max over [0, idx)
        else:
            best = 0
        # Update LIS length ending with val
        st.update(idx, best + 1)

    return st.query(0, m)

arr = [10, 9, 2, 5, 3, 7, 101, 18]
print(lis_segtree(arr))  # 4 (2, 3, 7, 101)
```

### Interval Scheduling (Segment Tree)

```python
def max_events_attended(events: list) -> int:
    """Maximum number of events attended (at most one event per day)
    events: [(start_day, end_day), ...]
    Uses a segment tree to efficiently search for available days
    """
    if not events:
        return 0

    # Sort by end day (greedy approach)
    events.sort(key=lambda x: x[1])
    max_day = max(e[1] for e in events)

    # Segment tree (range minimum): 0=available, 1=reserved
    tree = [0] * (4 * (max_day + 1))
    n = max_day + 1

    def update(node, start, end, idx, val):
        if start == end:
            tree[node] = val
            return
        mid = (start + end) // 2
        if idx <= mid:
            update(2 * node, start, mid, idx, val)
        else:
            update(2 * node + 1, mid + 1, end, idx, val)
        tree[node] = min(tree[2 * node], tree[2 * node + 1])

    def find_empty(node, start, end, l, r):
        """Return the first available day in [l, r] (-1 = none available)"""
        if r < start or end < l or tree[node] >= 1:
            return -1
        if start == end:
            return start if tree[node] == 0 else -1
        mid = (start + end) // 2
        left = find_empty(2 * node, start, mid, l, r)
        if left != -1:
            return left
        return find_empty(2 * node + 1, mid + 1, end, l, r)

    count = 0
    for s, e in events:
        day = find_empty(1, 0, n - 1, s, e)
        if day != -1:
            update(1, 0, n - 1, day, 1)
            count += 1

    return count
```

---

## 10. Persistent Segment Tree

Allows access to any past version of the segment tree. Only the modified nodes are newly created at each update (shared structure).

```python
class PersistentSegmentTree:
    """Persistent segment tree (range sum)
    A new version is created on each update, and past versions remain accessible
    Space complexity: O(n + q * log n) (q = number of updates)
    """

    class Node:
        __slots__ = ['left', 'right', 'val']

        def __init__(self, val=0, left=None, right=None):
            self.val = val
            self.left = left
            self.right = right

    def __init__(self, data: list):
        self.n = len(data)
        self.roots = []
        self.roots.append(self._build(data, 0, self.n - 1))

    def _build(self, data, start, end):
        if start == end:
            return self.Node(data[start])
        mid = (start + end) // 2
        left = self._build(data, start, mid)
        right = self._build(data, mid + 1, end)
        return self.Node(left.val + right.val, left, right)

    def update(self, version: int, idx: int, val: int) -> int:
        """Create a new version by updating idx to val based on the given version
        Returns: new version number
        """
        new_root = self._update(self.roots[version], 0, self.n - 1, idx, val)
        self.roots.append(new_root)
        return len(self.roots) - 1

    def _update(self, node, start, end, idx, val):
        if start == end:
            return self.Node(val)
        mid = (start + end) // 2
        if idx <= mid:
            new_left = self._update(node.left, start, mid, idx, val)
            return self.Node(new_left.val + node.right.val, new_left, node.right)
        else:
            new_right = self._update(node.right, mid + 1, end, idx, val)
            return self.Node(node.left.val + new_right.val, node.left, new_right)

    def query(self, version: int, l: int, r: int) -> int:
        """Compute the sum of range [l, r] on the given version"""
        return self._query(self.roots[version], 0, self.n - 1, l, r)

    def _query(self, node, start, end, l, r):
        if node is None or r < start or end < l:
            return 0
        if l <= start and end <= r:
            return node.val
        mid = (start + end) // 2
        return (self._query(node.left, start, mid, l, r) +
                self._query(node.right, mid + 1, end, l, r))

# Usage example
data = [1, 2, 3, 4, 5]
pst = PersistentSegmentTree(data)  # version 0

v1 = pst.update(0, 2, 10)  # version 1: [1, 2, 10, 4, 5]
v2 = pst.update(1, 4, 20)  # version 2: [1, 2, 10, 4, 20]

print(pst.query(0, 0, 4))  # 15 (version 0: 1+2+3+4+5)
print(pst.query(1, 0, 4))  # 22 (version 1: 1+2+10+4+5)
print(pst.query(2, 0, 4))  # 37 (version 2: 1+2+10+4+20)
```

---

## 11. Sparse Table (Static RMQ)

A structure specialized for range minimum queries when there are no updates. Preprocessing O(n log n), query O(1).

```python
import math

class SparseTable:
    """Sparse Table - static RMQ (Range Minimum Query)
    Preprocessing: O(n log n), Query: O(1), Update: not supported
    """

    def __init__(self, data: list):
        self.n = len(data)
        self.LOG = max(1, math.floor(math.log2(self.n))) + 1
        self.table = [[float('inf')] * self.n for _ in range(self.LOG)]

        # Initialization
        for i in range(self.n):
            self.table[0][i] = data[i]

        # Build via DP
        for j in range(1, self.LOG):
            for i in range(self.n - (1 << j) + 1):
                self.table[j][i] = min(
                    self.table[j-1][i],
                    self.table[j-1][i + (1 << (j-1))]
                )

    def query(self, l: int, r: int) -> int:
        """Minimum of range [l, r] - O(1)"""
        length = r - l + 1
        k = math.floor(math.log2(length))
        return min(self.table[k][l], self.table[k][r - (1 << k) + 1])

# Usage example
data = [5, 2, 8, 1, 9, 3, 7, 4]
sp = SparseTable(data)
print(sp.query(0, 3))  # 1
print(sp.query(4, 7))  # 3
print(sp.query(1, 6))  # 1

# RMQ method comparison
# | Method         | Preprocessing | Query     | Update   |
# |:--------------|:-------------|:---------|:---------|
# | Naive          | O(1)         | O(n)     | O(1)     |
# | Segment tree   | O(n)         | O(log n) | O(log n) |
# | Sparse Table   | O(n lg n)    | O(1)     | N/A      |
# | Sqrt decomp.   | O(n)         | O(sqrt n)| O(1)     |
```

---

## 12. Sqrt Decomposition

A technique that divides the array into blocks of sqrt(n). Simpler to implement than segment trees and provides sufficient performance for some problems.

```python
import math

class SqrtDecomposition:
    """Sqrt decomposition - range sum query + point update
    Construction: O(n), Query: O(sqrt n), Update: O(1)
    """

    def __init__(self, data: list):
        self.n = len(data)
        self.block_size = max(1, int(math.sqrt(self.n)))
        self.num_blocks = (self.n + self.block_size - 1) // self.block_size
        self.data = data[:]
        self.blocks = [0] * self.num_blocks

        for i in range(self.n):
            self.blocks[i // self.block_size] += data[i]

    def update(self, idx: int, val: int):
        """Point update - O(1)"""
        block = idx // self.block_size
        self.blocks[block] += val - self.data[idx]
        self.data[idx] = val

    def query(self, l: int, r: int) -> int:
        """Sum of range [l, r] - O(sqrt n)"""
        result = 0
        bl = l // self.block_size
        br = r // self.block_size

        if bl == br:
            # Within the same block
            for i in range(l, r + 1):
                result += self.data[i]
        else:
            # Left partial block
            for i in range(l, (bl + 1) * self.block_size):
                result += self.data[i]
            # Complete middle blocks
            for b in range(bl + 1, br):
                result += self.blocks[b]
            # Right partial block
            for i in range(br * self.block_size, r + 1):
                result += self.data[i]

        return result

# Usage example
data = [1, 3, 5, 7, 9, 11, 13, 15, 17]
sd = SqrtDecomposition(data)
print(sd.query(2, 6))  # 45 (5+7+9+11+13)
sd.update(4, 100)       # data[4] = 9 -> 100
print(sd.query(2, 6))  # 136 (5+7+100+11+13)
```

---

## 13. Anti-patterns

### Anti-pattern 1: Incorrect Array Size Estimation

```python
# BAD: Segment tree size is insufficient
class BadSegTree:
    def __init__(self, data):
        self.tree = [0] * (2 * len(data))  # insufficient!
        # When n is not a power of 2, 2n is not enough

# GOOD: Allocate sufficient size with 4n
class GoodSegTree:
    def __init__(self, data):
        self.tree = [0] * (4 * len(data))  # safe
```

### Anti-pattern 2: Forgetting push_down in Lazy Propagation

```python
# BAD: Not propagating lazy values during query
def bad_query(self, node, start, end, l, r):
    if l <= start and end <= r:
        return self.tree[node]  # lazy not reflected!
    mid = (start + end) // 2
    # Accessing children without push_down -> incorrect results
    return (self._query(2*node, start, mid, l, r) +
            self._query(2*node+1, mid+1, end, l, r))

# GOOD: Always push_down before accessing children
def good_query(self, node, start, end, l, r):
    if l <= start and end <= r:
        return self.tree[node]
    self._push_down(node, start, end)  # propagate lazy values!
    mid = (start + end) // 2
    return (self._query(2*node, start, mid, l, r) +
            self._query(2*node+1, mid+1, end, l, r))
```

### Anti-pattern 3: Using 0-indexed BIT

```python
# BAD: Using BIT with 0-indexed
# i & (-i) when i=0 is 0 -> infinite loop!
class BadBIT:
    def prefix_sum(self, i):
        s = 0
        while i > 0:  # when i=0, i & (-i) = 0 so it exits but...
            s += self.tree[i]
            i -= i & (-i)  # 0 - 0 = 0 -> no problem here
        return s

    def update(self, i, delta):
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)  # 0 + 0 = 0 -> infinite loop!

# GOOD: Always use 1-indexed for BIT
class GoodBIT:
    def update(self, i, delta):
        # Ensure i is at least 1
        assert i >= 1
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)
```

### Anti-pattern 4: Mixing Closed and Half-open Intervals in Segment Tree

```python
# BAD: Mixing closed interval [l, r] and half-open interval [l, r)
# Maintain consistency between segment tree using closed intervals and BIT using 1-indexed

# GOOD: Use one convention consistently throughout the implementation
# Recursive segment tree: closed interval [l, r] is common
# Iterative segment tree: half-open interval [l, r) is common
# BIT: 1-indexed + closed interval [l, r]
```


---

## Hands-on Exercises

### Exercise 1: Basic Implementation

Implement code that meets the following requirements.

**Requirements:**
- Validate input data
- Implement proper error handling
- Include test code

```python
# Exercise 1: Basic implementation template
class Exercise1:
    """Exercise for basic implementation patterns"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """Validate input value"""
        if value is None:
            raise ValueError("Input value is None")
        return True

    def process(self, value):
        """Main processing logic"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """Get processing results"""
        return {
            'count': len(self.data),
            'data': self.data
        }

# Tests
def test_exercise1():
    ex = Exercise1()
    assert ex.process(1) == [1]
    assert ex.process(2) == [1, 2]
    assert ex.get_results()['count'] == 2

    try:
        ex.process(None)
        assert False, "Should have raised an exception"
    except ValueError:
        pass

    print("All tests passed!")

test_exercise1()
```

### Exercise 2: Advanced Patterns

Extend the basic implementation with the following features.

```python
# Exercise 2: Advanced patterns
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """Exercise for advanced patterns"""

    def __init__(self, max_size: int = 100):
        self._items: List[Dict] = []
        self._max_size = max_size
        self._created_at = datetime.now()

    def add(self, key: str, value: any) -> bool:
        """Add an item (with size limit)"""
        if len(self._items) >= self._max_size:
            return False
        self._items.append({
            'key': key,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        return True

    def find(self, key: str) -> Optional[Dict]:
        """Search by key"""
        for item in reversed(self._items):
            if item['key'] == key:
                return item
        return None

    def remove(self, key: str) -> bool:
        """Delete by key"""
        for i, item in enumerate(self._items):
            if item['key'] == key:
                self._items.pop(i)
                return True
        return False

    def stats(self) -> Dict:
        """Statistics"""
        return {
            'total_items': len(self._items),
            'max_size': self._max_size,
            'usage_percent': len(self._items) / self._max_size * 100,
            'uptime': str(datetime.now() - self._created_at)
        }

# Tests
def test_advanced():
    ex = AdvancedExercise(max_size=3)
    assert ex.add("a", 1) == True
    assert ex.add("b", 2) == True
    assert ex.add("c", 3) == True
    assert ex.add("d", 4) == False  # size limit
    assert ex.find("b")['value'] == 2
    assert ex.remove("b") == True
    assert ex.find("b") is None
    stats = ex.stats()
    assert stats['total_items'] == 2
    print("All advanced tests passed!")

test_advanced()
```

### Exercise 3: Performance Optimization

Improve the performance of the following code.

```python
# Exercise 3: Performance optimization
import time
from functools import lru_cache

# Before optimization (O(n^2))
def slow_search(data: list, target: int) -> int:
    """Inefficient search"""
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] + data[j] == target:
                return (i, j)
    return (-1, -1)

# After optimization (O(n))
def fast_search(data: list, target: int) -> tuple:
    """Efficient search using a hash map"""
    seen = {}
    for i, num in enumerate(data):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return (-1, -1)

# Benchmark
def benchmark():
    import random
    data = list(range(5000))
    random.shuffle(data)
    target = data[100] + data[4000]

    start = time.time()
    result1 = slow_search(data, target)
    slow_time = time.time() - start

    start = time.time()
    result2 = fast_search(data, target)
    fast_time = time.time() - start

    print(f"Inefficient version: {slow_time:.4f}s")
    print(f"Efficient version:   {fast_time:.6f}s")
    print(f"Speedup: {slow_time/fast_time:.0f}x")

benchmark()
```

**Key points:**
- Be mindful of algorithmic time complexity
- Choose appropriate data structures
- Measure effectiveness with benchmarks
---

## 14. FAQ

### Q1: Why is the segment tree size 4n?

**A:** A segment tree is built as a complete binary tree, and when n is not a power of 2, the number of leaves is rounded up to the next power of 2. When using 1-indexed node numbering (children at 2*node and 2*node+1), up to approximately 4n nodes are needed. 4n is the standard safe margin. The iterative version can allocate exactly 2 * (next power of 2).

### Q2: Can BIT compute range minimum?

**A:** It is difficult with a standard BIT. BIT is suited for "invertible binary operations on prefixes," and since min/max lack inverses, they cannot be decomposed into range [l, r] queries. Use a segment tree for range minimum. However, there is a special case where BIT can compute range minimum when updates are monotonically increasing (values never decrease).

### Q3: What about 2D segment trees?

**A:** A "2D segment tree" where each node of the outer segment tree holds an inner segment tree is possible. Time complexity is O(log^2 n), space is O(n^2 log^2 n). In practice, a 2D BIT is simpler to implement. For more advanced needs, KD-Tree or R-Tree can be used.

### Q4: How do you combine multiple operations in lazy propagation?

**A:** For example, when mixing "range assignment" and "range addition," the lazy values must be managed as (add, assign) pairs with correctly defined composition rules. Typically: "if assign exists, reset add; if only add, accumulate." Such compound lazy propagation is hard to implement and prone to bugs.

### Q5: What if the segment tree has too many nodes and causes MLE (Memory Limit Exceeded)?

**A:** (1) Use a dynamic segment tree (create nodes on demand). Effective when coordinates are large but only a few indices are actually used. (2) Use coordinate compression to reduce the value range. (3) If the problem can use BIT, switch to BIT (approximately 1/4 the memory).

---


## FAQ

### Q1: What is the most important point when learning this topic?

Building practical experience is the most important aspect. Understanding deepens not only through theory but also by actually writing code and verifying its behavior.

### Q2: What common mistakes do beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving to the next step.

### Q3: How is this used in practice?

Knowledge of this topic is frequently applied in day-to-day development work. It becomes especially important during code reviews and architecture design.

---

## 15. Summary

| Topic | Key Points |
|:---|:---|
| Segment tree | Handles range queries + point updates in O(log n) |
| Lazy propagation | Extends range updates to O(log n). push_down is the key |
| BIT | Specialized for prefix sums. Simpler and faster than segment trees |
| Abstract segment tree | Generalizes monoid operations. Supports any associative operation |
| Persistent segment tree | Enables access to past versions |
| Sparse Table | Answers static RMQ in O(1). No update support |
| Sqrt decomposition | Simple implementation. O(sqrt n) queries |
| Use cases | Dynamic queries for range sum, min, max, GCD |

---

## Recommended Next Guides

- [Union-Find](./00-union-find.md) -- Another advanced data structure
- [String Algorithms](./02-string-algorithms.md) -- Tree structure applications such as Trie
- [Competitive Programming](../04-practice/01-competitive-programming.md) -- Practical use of segment trees

---

## References

1. Cormen, T. H. et al. (2022). *Introduction to Algorithms* (4th ed.). MIT Press.
2. Fenwick, P. M. (1994). "A New Data Structure for Cumulative Frequency Tables." *Software: Practice and Experience*.
3. Halim, S. & Halim, F. (2013). *Competitive Programming 3*. -- Chapter 2: Data Structures
4. cp-algorithms. "Segment Tree." https://cp-algorithms.com/data_structures/segment_tree.html
5. Bender, M. A. & Farach-Colton, M. (2000). "The LCA Problem Revisited." *LATIN*. -- Sparse Table
6. Akiba, T. et al. (2012). *Programming Contest Challenge Book* (2nd ed.). Mynavi Publishing.
