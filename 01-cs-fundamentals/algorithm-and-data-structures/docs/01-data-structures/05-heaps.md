# Heaps — Binary Heaps, Heapsort, and Priority Queue Implementation

> Learn the structure of heaps that enable efficient retrieval of maximum/minimum values, heapsort, and priority queue implementation.

---

## What You Will Learn in This Chapter

1. **Binary heap** structure and array representation
2. **Heapsort** mechanism and complexity analysis
3. **Priority queue** implementation using heaps and practical applications
4. **Various heap types** — d-ary heaps, Fibonacci heaps, indexed heaps
5. **Practical applications** — Top-K, median finding, task schedulers, merging K sorted lists

## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Familiarity with the content of [Trees — Binary Trees, BST, AVL/Red-Black Trees, B-Trees, Tries](./04-trees.md)

---

## 1. Binary Heap Structure

### 1.1 Heap Definition

```
Min-Heap:
  parent <= child holds for all nodes

         [1]            Index:
        /   \              0: 1
      [3]   [2]           1: 3,  2: 2
     / \   /              3: 5,  4: 8,  5: 7
   [5] [8] [7]

Array representation: [1, 3, 2, 5, 8, 7]

  Parent:       i -> (i-1) // 2
  Left child:   i -> 2*i + 1
  Right child:  i -> 2*i + 2
  Leaf node:    i >= n // 2

Max-Heap:
  parent >= child holds for all nodes

         [9]
        /   \
      [7]   [8]
     / \   /
   [3] [5] [2]

Array representation: [9, 7, 8, 3, 5, 2]
```

### 1.2 Important Properties of Heaps

```python
# Properties of a binary heap:
# 1. Complete binary tree: all levels except the last are fully filled
# 2. Heap order property: parent <= child (min-heap) or parent >= child (max-heap)
# 3. Can be efficiently represented as an array (no pointers needed)
# 4. Root is the minimum (min-heap) or maximum (max-heap)

# Properties of a heap with n nodes:
# - Height: floor(log2(n))
# - Number of leaf nodes: ceil(n/2)
# - Number of internal nodes: floor(n/2)
# - Last internal node: index n//2 - 1

def heap_properties(n):
    """Properties of a heap with n nodes"""
    import math
    height = math.floor(math.log2(n)) if n > 0 else 0
    leaves = math.ceil(n / 2)
    internal = n // 2
    last_internal = n // 2 - 1
    print(f"Number of nodes: {n}")
    print(f"Height: {height}")
    print(f"Number of leaf nodes: {leaves}")
    print(f"Number of internal nodes: {internal}")
    print(f"Last internal node: index {last_internal}")

heap_properties(10)
# Number of nodes: 10, Height: 3, Leaf nodes: 5, Internal nodes: 5
```

---

## 2. Heap Operation Implementation

### 2.1 Min-Heap

```python
class MinHeap:
    """Min-Heap: root is the minimum value

    Use cases: priority queues, Dijkstra's algorithm, Huffman coding
    """
    def __init__(self):
        self.heap = []

    def __len__(self):
        return len(self.heap)

    def __bool__(self):
        return len(self.heap) > 0

    def push(self, val):
        """O(log n) — append to the end and sift up"""
        self.heap.append(val)
        self._sift_up(len(self.heap) - 1)

    def pop(self):
        """O(log n) — extract root and sift down"""
        if not self.heap:
            raise IndexError("heap is empty")
        self._swap(0, len(self.heap) - 1)
        val = self.heap.pop()
        if self.heap:
            self._sift_down(0)
        return val

    def peek(self):
        """O(1) — return minimum value (without removing)"""
        if not self.heap:
            raise IndexError("heap is empty")
        return self.heap[0]

    def push_pop(self, val):
        """Optimized push + pop — O(log n)
        More efficient than pushing then popping separately
        """
        if self.heap and self.heap[0] < val:
            val, self.heap[0] = self.heap[0], val
            self._sift_down(0)
        return val

    def replace(self, val):
        """Optimized pop + push — O(log n)
        More efficient than popping then pushing separately
        """
        if not self.heap:
            raise IndexError("heap is empty")
        old = self.heap[0]
        self.heap[0] = val
        self._sift_down(0)
        return old

    def _sift_up(self, i):
        """Sift up: swap with parent if smaller"""
        while i > 0:
            parent = (i - 1) // 2
            if self.heap[i] < self.heap[parent]:
                self._swap(i, parent)
                i = parent
            else:
                break

    def _sift_down(self, i):
        """Sift down: swap with smallest child if larger"""
        n = len(self.heap)
        while True:
            smallest = i
            left = 2 * i + 1
            right = 2 * i + 2
            if left < n and self.heap[left] < self.heap[smallest]:
                smallest = left
            if right < n and self.heap[right] < self.heap[smallest]:
                smallest = right
            if smallest == i:
                break
            self._swap(i, smallest)
            i = smallest

    def _swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    @classmethod
    def heapify(cls, arr):
        """Build a heap from an array — O(n)
        Achieves O(n) by performing bottom-up sift_down
        """
        heap = cls()
        heap.heap = list(arr)
        n = len(heap.heap)
        # Sift down from the last internal node toward the root
        for i in range(n // 2 - 1, -1, -1):
            heap._sift_down(i)
        return heap

# Usage example
h = MinHeap()
h.push(5)
h.push(3)
h.push(8)
h.push(1)
print(h.peek())  # 1
print(h.pop())   # 1
print(h.pop())   # 3

# heapify
h2 = MinHeap.heapify([5, 3, 8, 1, 2, 7])
print(h2.pop())  # 1
print(h2.pop())  # 2
```

### 2.2 Max-Heap

```python
class MaxHeap:
    """Max-Heap: root is the maximum value

    Use cases: retrieving highest-priority tasks, descending-order streaming
    """
    def __init__(self):
        self.heap = []

    def __len__(self):
        return len(self.heap)

    def push(self, val):
        """O(log n)"""
        self.heap.append(val)
        self._sift_up(len(self.heap) - 1)

    def pop(self):
        """O(log n)"""
        if not self.heap:
            raise IndexError("heap is empty")
        self._swap(0, len(self.heap) - 1)
        val = self.heap.pop()
        if self.heap:
            self._sift_down(0)
        return val

    def peek(self):
        """O(1)"""
        if not self.heap:
            raise IndexError("heap is empty")
        return self.heap[0]

    def _sift_up(self, i):
        while i > 0:
            parent = (i - 1) // 2
            if self.heap[i] > self.heap[parent]:  # Changed to >
                self._swap(i, parent)
                i = parent
            else:
                break

    def _sift_down(self, i):
        n = len(self.heap)
        while True:
            largest = i
            left = 2 * i + 1
            right = 2 * i + 2
            if left < n and self.heap[left] > self.heap[largest]:
                largest = left
            if right < n and self.heap[right] > self.heap[largest]:
                largest = right
            if largest == i:
                break
            self._swap(i, largest)
            i = largest

    def _swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
```

### 2.3 Visual Illustration of Operations

```
push(2) example (sift-up):

  [1, 5, 3, 8, 7]  <- push(2)
  [1, 5, 3, 8, 7, 2]

         [1]                  [1]
        /   \                /   \
      [5]   [3]    ->      [5]   [2]  <- swap because 2 < 3
     / \   /              / \   /
   [8] [7] [2]         [8] [7] [3]

pop() example (sift-down):

  [1, 5, 2, 8, 7, 3]  -> return head (1)

  Move tail (3) to head:
  [3, 5, 2, 8, 7]

         [3]                  [2]
        /   \                /   \
      [5]   [2]    ->      [5]   [3]  <- swap because 3 > 2
     / \                  / \
   [8] [7]             [8] [7]

heapify process (bottom-up):

  Input: [4, 10, 3, 5, 1]

  Step 0: Initial state
         [4]
        /   \
      [10]  [3]
     / \
    [5] [1]

  Step 1: i=1, sift_down(10)
         [4]
        /   \
      [1]   [3]     <- swap because 10 > 1
     / \
    [5] [10]

  Step 2: i=0, sift_down(4)
         [1]
        /   \
      [4]   [3]     <- swap because 4 > 1; no swap needed between 4 and 5
     / \
    [5] [10]

  Result: [1, 4, 3, 5, 10] — valid heap
```

---

## 3. Heapsort

### 3.1 Basic Implementation

```python
def heapsort(arr):
    """Heapsort — O(n log n) time, O(1) space

    Step 1: Build a max-heap — O(n)
    Step 2: Swap root with tail and shrink heap size — O(n log n)
    """
    n = len(arr)

    # Step 1: Build max-heap — O(n)
    for i in range(n // 2 - 1, -1, -1):
        _sift_down_max(arr, n, i)

    # Step 2: Extract one by one — O(n log n)
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]  # Move maximum to tail
        _sift_down_max(arr, i, 0)        # Re-heapify remainder

    return arr

def _sift_down_max(arr, n, i):
    """Max-heap sift down"""
    while True:
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        if left < n and arr[left] > arr[largest]:
            largest = left
        if right < n and arr[right] > arr[largest]:
            largest = right
        if largest == i:
            break
        arr[i], arr[largest] = arr[largest], arr[i]
        i = largest

# Usage example
data = [12, 11, 13, 5, 6, 7, 3, 1, 9, 2, 4, 8, 10, 14, 15]
print(heapsort(data))
# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
```

### 3.2 Heapsort Process

```
Heapsort process:

Step 1: Build max-heap
  [4, 1, 3, 2, 5]  ->  [5, 4, 3, 2, 1]

         [4]               [5]
        /   \      ->      /   \
      [1]   [3]         [4]   [3]
     / \               / \
    [2] [5]           [2] [1]

Step 2: Swap head and tail + sift-down
  Round 1: swap(5,1)
  [5, 4, 3, 2, 1] -> [1, 4, 3, 2, |5]
  sift-down -> [4, 2, 3, 1, |5]

  Round 2: swap(4,1)
  [4, 2, 3, 1, |5] -> [1, 2, 3, |4, 5]
  sift-down -> [3, 2, 1, |4, 5]

  Round 3: swap(3,1)
  [3, 2, 1, |4, 5] -> [1, 2, |3, 4, 5]

  Round 4: swap(2,1)
  [2, 1, |3, 4, 5] -> [1, |2, 3, 4, 5]

  Result: [1, 2, 3, 4, 5]
```

### 3.3 Partial Sort (Top-K) Optimization

```python
def partial_sort_top_k(arr, k):
    """Return only the top k elements sorted — O(n + k log n)

    More efficient than full O(n log n) sort when k << n
    """
    n = len(arr)
    # Step 1: Build max-heap — O(n)
    for i in range(n // 2 - 1, -1, -1):
        _sift_down_max(arr, n, i)

    # Step 2: Extract only top k elements — O(k log n)
    result = []
    heap_size = n
    for _ in range(min(k, n)):
        result.append(arr[0])
        arr[0] = arr[heap_size - 1]
        heap_size -= 1
        _sift_down_max(arr, heap_size, 0)

    return result

data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
print(partial_sort_top_k(data[:], 3))  # [9, 6, 5]
```

---

## 4. The heapq Module

### 4.1 Basic Operations

```python
import heapq

# Min-heap operations
nums = [5, 3, 8, 1, 2]
heapq.heapify(nums)          # O(n) heapify
print(nums)                   # [1, 2, 8, 5, 3]

heapq.heappush(nums, 0)      # O(log n)
print(heapq.heappop(nums))   # 0 — O(log n)

# heappushpop: optimized push + pop
result = heapq.heappushpop(nums, 4)  # push(4) then pop
print(result)  # 1

# heapreplace: optimized pop + push
result = heapq.heapreplace(nums, 10)  # pop then push(10)
print(result)  # 2

# Top-K (smallest K elements)
top3 = heapq.nsmallest(3, [5, 3, 8, 1, 2])  # [1, 2, 3]
# Top-K (largest K elements)
top3 = heapq.nlargest(3, [5, 3, 8, 1, 2])   # [8, 5, 3]

# Max-heap (sign negation trick)
max_heap = []
for x in [5, 3, 8, 1, 2]:
    heapq.heappush(max_heap, -x)
print(-heapq.heappop(max_heap))  # 8

# Keyed heap (using tuples)
tasks = [(3, "low priority"), (1, "high priority"), (2, "medium")]
heapq.heapify(tasks)
print(heapq.heappop(tasks))  # (1, 'high priority')
```

### 4.2 Internal Implementation of heapq

```python
# Performance characteristics of heapq:
# - C implementation (CPython): very fast
# - heapify: O(n)
# - heappush/heappop: O(log n)
# - nsmallest/nlargest:
#   - When k is small: uses heap O(n + k log n)
#   - When k is close to n: uses sort O(n log n)
#   - Automatically selects the optimal method internally

# Choosing between nsmallest/nlargest
import heapq

data = list(range(1000000))

# When k << n: nsmallest is efficient
top10 = heapq.nsmallest(10, data)  # O(n)

# When k is close to n: sorted is more efficient
# heapq.nsmallest(999990, data)  # internally switches to sorted

# When k = 1: min/max is fastest
minimum = min(data)  # O(n)
maximum = max(data)  # O(n)
```

### 4.3 heapq.merge — Merging K Sorted Iterables

```python
import heapq

# Merging sorted lists — O(n log k)
list1 = [1, 4, 7, 10]
list2 = [2, 5, 8, 11]
list3 = [3, 6, 9, 12]

merged = list(heapq.merge(list1, list2, list3))
print(merged)  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# Custom merge using the key argument
records1 = [(1, "a"), (3, "c"), (5, "e")]
records2 = [(2, "b"), (4, "d"), (6, "f")]
merged = list(heapq.merge(records1, records2, key=lambda x: x[0]))
print(merged)  # [(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd'), (5, 'e'), (6, 'f')]

# Merge phase of external sort (sorting large files)
# Sort each chunk and write to file
# Merge K chunks using heapq.merge
def external_sort_merge(sorted_files, output_file):
    """Merge phase of external sort"""
    import itertools
    file_iters = [open(f) for f in sorted_files]
    with open(output_file, 'w') as out:
        for line in heapq.merge(*file_iters, key=lambda x: int(x.strip())):
            out.write(line)
    for f in file_iters:
        f.close()
```

---

## 5. Priority Queue Implementation

### 5.1 Basic Priority Queue

```python
import heapq

class PriorityQueue:
    """Priority Queue: dequeues the highest-priority element first

    heapq-based implementation.
    Uses tuples (priority, counter, item) to achieve:
    1. FIFO ordering for equal priorities
    2. Support for non-comparable objects
    """
    def __init__(self):
        self._queue = []
        self._counter = 0  # For maintaining FIFO order

    def push(self, item, priority=0):
        """O(log n)"""
        heapq.heappush(self._queue, (priority, self._counter, item))
        self._counter += 1

    def pop(self):
        """O(log n) — returns the element with highest priority (lowest value)"""
        if not self._queue:
            raise IndexError("queue is empty")
        return heapq.heappop(self._queue)[2]

    def peek(self):
        """O(1)"""
        if not self._queue:
            raise IndexError("queue is empty")
        return self._queue[0][2]

    def __len__(self):
        return len(self._queue)

    def __bool__(self):
        return len(self._queue) > 0

# Usage example
pq = PriorityQueue()
pq.push("low priority task", priority=3)
pq.push("high priority task", priority=1)
pq.push("medium priority task", priority=2)
pq.push("another high priority", priority=1)

while pq:
    print(pq.pop())
# high priority task
# another high priority  (FIFO for equal priorities)
# medium priority task
# low priority task
```

### 5.2 Priority Queue with Lazy Deletion

```python
import heapq

class LazyDeletionPQ:
    """Priority queue with lazy deletion support

    Instead of directly removing elements from the heap,
    marks them as deleted and skips them on pop.
    Commonly used in Dijkstra's algorithm implementations.
    """
    def __init__(self):
        self._queue = []
        self._counter = 0
        self._deleted = set()  # Counter IDs of deleted entries

    def push(self, item, priority=0):
        """O(log n)"""
        entry_id = self._counter
        heapq.heappush(self._queue, (priority, entry_id, item))
        self._counter += 1
        return entry_id

    def pop(self):
        """O(log n) amortized — skips deleted entries"""
        while self._queue:
            priority, entry_id, item = heapq.heappop(self._queue)
            if entry_id not in self._deleted:
                return item
            self._deleted.discard(entry_id)
        raise IndexError("queue is empty")

    def delete(self, entry_id):
        """O(1) — lazy deletion mark"""
        self._deleted.add(entry_id)

    def __len__(self):
        return len(self._queue) - len(self._deleted)

# Usage example: priority update (delete old entry and add new one)
pq = LazyDeletionPQ()
id1 = pq.push("task A", priority=5)
id2 = pq.push("task B", priority=3)

# Update priority of task A (5 -> 1)
pq.delete(id1)
id1_new = pq.push("task A", priority=1)

print(pq.pop())  # "task A" (priority 1)
print(pq.pop())  # "task B" (priority 3)
```

### 5.3 Indexed Heap

```python
class IndexedMinHeap:
    """Indexed min-heap

    Supports O(log n) priority updates for each key.
    Ideal for Dijkstra's and Prim's algorithms.

    - insert(key, priority): O(log n)
    - pop(): O(log n)
    - decrease_key(key, new_priority): O(log n)
    - contains(key): O(1)
    """
    def __init__(self, capacity=100):
        self.heap = []          # List of (priority, key)
        self.key_to_idx = {}    # key -> index in heap
        self.key_to_priority = {}

    def __len__(self):
        return len(self.heap)

    def __contains__(self, key):
        return key in self.key_to_idx

    def insert(self, key, priority):
        """O(log n)"""
        if key in self.key_to_idx:
            raise ValueError(f"Key {key} already exists")
        idx = len(self.heap)
        self.heap.append((priority, key))
        self.key_to_idx[key] = idx
        self.key_to_priority[key] = priority
        self._sift_up(idx)

    def pop(self):
        """O(log n) — returns the element with minimum priority"""
        if not self.heap:
            raise IndexError("heap is empty")
        priority, key = self.heap[0]
        self._swap(0, len(self.heap) - 1)
        self.heap.pop()
        del self.key_to_idx[key]
        del self.key_to_priority[key]
        if self.heap:
            self._sift_down(0)
        return key, priority

    def decrease_key(self, key, new_priority):
        """O(log n) — decrease the priority of a key"""
        if key not in self.key_to_idx:
            raise KeyError(key)
        if new_priority >= self.key_to_priority[key]:
            return  # Do nothing if new priority is not lower
        idx = self.key_to_idx[key]
        self.heap[idx] = (new_priority, key)
        self.key_to_priority[key] = new_priority
        self._sift_up(idx)

    def _sift_up(self, i):
        while i > 0:
            parent = (i - 1) // 2
            if self.heap[i][0] < self.heap[parent][0]:
                self._swap(i, parent)
                i = parent
            else:
                break

    def _sift_down(self, i):
        n = len(self.heap)
        while True:
            smallest = i
            left = 2 * i + 1
            right = 2 * i + 2
            if left < n and self.heap[left][0] < self.heap[smallest][0]:
                smallest = left
            if right < n and self.heap[right][0] < self.heap[smallest][0]:
                smallest = right
            if smallest == i:
                break
            self._swap(i, smallest)
            i = smallest

    def _swap(self, i, j):
        ki = self.heap[i][1]
        kj = self.heap[j][1]
        self.key_to_idx[ki] = j
        self.key_to_idx[kj] = i
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

# Usage example: Dijkstra's algorithm
ipq = IndexedMinHeap()
ipq.insert("A", 0)   # Source: distance 0
ipq.insert("B", 10)
ipq.insert("C", 5)
ipq.insert("D", float('inf'))

# Update distances
ipq.decrease_key("B", 3)  # Distance to B improved from 10 to 3

key, dist = ipq.pop()
print(f"{key}: {dist}")  # A: 0
key, dist = ipq.pop()
print(f"{key}: {dist}")  # B: 3
key, dist = ipq.pop()
print(f"{key}: {dist}")  # C: 5
```

---

## 6. Various Heap Types

### 6.1 d-ary Heap

```python
class DaryHeap:
    """d-ary heap: each node has d children

    d=2: binary heap (standard)
    d=4: quaternary heap (can be more cache-efficient)

    - sift_up: O(log_d n) — lower height with larger d
    - sift_down: O(d * log_d n) — more child comparisons with larger d
    - Larger d is advantageous when decrease_key is frequent
      (e.g., Dijkstra on dense graphs)
    """
    def __init__(self, d=4):
        self.d = d
        self.heap = []

    def push(self, val):
        self.heap.append(val)
        self._sift_up(len(self.heap) - 1)

    def pop(self):
        if not self.heap:
            raise IndexError("heap is empty")
        self.heap[0], self.heap[-1] = self.heap[-1], self.heap[0]
        val = self.heap.pop()
        if self.heap:
            self._sift_down(0)
        return val

    def _parent(self, i):
        return (i - 1) // self.d

    def _children(self, i):
        start = self.d * i + 1
        return range(start, min(start + self.d, len(self.heap)))

    def _sift_up(self, i):
        while i > 0:
            parent = self._parent(i)
            if self.heap[i] < self.heap[parent]:
                self.heap[i], self.heap[parent] = self.heap[parent], self.heap[i]
                i = parent
            else:
                break

    def _sift_down(self, i):
        while True:
            smallest = i
            for child in self._children(i):
                if self.heap[child] < self.heap[smallest]:
                    smallest = child
            if smallest == i:
                break
            self.heap[i], self.heap[smallest] = self.heap[smallest], self.heap[i]
            i = smallest

# Guidelines for choosing d:
# - d=2: standard binary heap. Best overall balance
# - d=4: when decrease_key is frequent. Advantageous for Dijkstra on dense graphs
# - d=8+: can be optimized for cache line alignment
```

### 6.2 Fibonacci Heap (Concept)

```python
# The Fibonacci heap is the theoretically most efficient heap:
#
# | Operation      | Binary Heap | Fibonacci Heap  |
# |----------------|-------------|-----------------|
# | insert         | O(log n)    | O(1) amortized  |
# | peek           | O(1)        | O(1)            |
# | pop            | O(log n)    | O(log n) amort. |
# | decrease_key   | O(log n)    | O(1) amortized  |
# | merge          | O(n)        | O(1)            |
#
# When used with Dijkstra's, achieves O(V log V + E) (vs O((V+E) log V) with binary heap)
#
# However, the implementation is very complex and the constant factor is large,
# so binary heaps are often faster in practice.
# Primarily used for theoretical complexity analysis.

# Simplified mergeable heap (Pairing Heap)
class PairingHeapNode:
    def __init__(self, val):
        self.val = val
        self.children = []

class PairingHeap:
    """Pairing heap: simplified version of Fibonacci heap

    Simple to implement with good practical performance.
    Merge is possible in O(1).
    """
    def __init__(self):
        self.root = None

    def push(self, val):
        """O(1)"""
        new_node = PairingHeapNode(val)
        self.root = self._merge(self.root, new_node)

    def peek(self):
        """O(1)"""
        if not self.root:
            raise IndexError("heap is empty")
        return self.root.val

    def pop(self):
        """O(log n) amortized"""
        if not self.root:
            raise IndexError("heap is empty")
        val = self.root.val
        children = self.root.children

        # Two-pass pairing
        if not children:
            self.root = None
        else:
            # Pass 1: merge adjacent pairs
            merged = []
            for i in range(0, len(children), 2):
                if i + 1 < len(children):
                    merged.append(self._merge(children[i], children[i + 1]))
                else:
                    merged.append(children[i])

            # Pass 2: merge from right to left
            result = merged[-1]
            for i in range(len(merged) - 2, -1, -1):
                result = self._merge(result, merged[i])
            self.root = result

        return val

    def _merge(self, h1, h2):
        """Merge two heaps — O(1)"""
        if not h1:
            return h2
        if not h2:
            return h1
        if h1.val <= h2.val:
            h1.children.append(h2)
            return h1
        else:
            h2.children.append(h1)
            return h2

    def merge_with(self, other):
        """Merge with another heap — O(1)"""
        self.root = self._merge(self.root, other.root)

# Usage example
ph = PairingHeap()
for x in [5, 3, 8, 1, 7]:
    ph.push(x)
print(ph.pop())  # 1
print(ph.pop())  # 3
```

---

## 7. Practical Applications

### 7.1 K-th Largest Element

```python
def kth_largest(nums, k):
    """K-th largest element — O(n log k)

    Maintain a min-heap of size k.
    The root of the heap is always the k-th largest element.
    """
    import heapq
    # Maintain a min-heap of size k
    heap = nums[:k]
    heapq.heapify(heap)
    for num in nums[k:]:
        if num > heap[0]:
            heapq.heapreplace(heap, num)
    return heap[0]

print(kth_largest([3, 2, 1, 5, 6, 4], 2))  # 5
print(kth_largest([3, 2, 3, 1, 2, 4, 5, 5, 6], 4))  # 4
```

### 7.2 Streaming Median

```python
import heapq

class MedianFinder:
    """Streaming median — O(log n) insertion, O(1) retrieval

    Uses two heaps:
    - max_heap: lower half (max-heap)
    - min_heap: upper half (min-heap)

    Root of max_heap <= root of min_heap
    Size difference is at most 1
    """
    def __init__(self):
        self.max_heap = []  # Lower half (max-heap via sign negation)
        self.min_heap = []  # Upper half

    def add_num(self, num):
        """O(log n)"""
        # First add to max_heap
        heapq.heappush(self.max_heap, -num)
        # Move max of max_heap to min_heap
        heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))
        # If min_heap is too large, move back to max_heap
        if len(self.min_heap) > len(self.max_heap):
            heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))

    def find_median(self):
        """O(1)"""
        if len(self.max_heap) > len(self.min_heap):
            return -self.max_heap[0]
        return (-self.max_heap[0] + self.min_heap[0]) / 2

# Usage example
mf = MedianFinder()
mf.add_num(1)
print(mf.find_median())  # 1.0
mf.add_num(2)
print(mf.find_median())  # 1.5
mf.add_num(3)
print(mf.find_median())  # 2.0
mf.add_num(4)
print(mf.find_median())  # 2.5
mf.add_num(5)
print(mf.find_median())  # 3.0
```

### 7.3 Merging K Sorted Lists

```python
import heapq

def merge_k_sorted_lists(lists):
    """Merge K sorted lists — O(N log K)

    N = total number of elements, K = number of lists
    Insert the head element of each list into the heap and extract the minimum.
    """
    heap = []
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))

    result = []
    while heap:
        val, list_idx, elem_idx = heapq.heappop(heap)
        result.append(val)
        if elem_idx + 1 < len(lists[list_idx]):
            next_val = lists[list_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))

    return result

# Usage example
lists = [
    [1, 4, 5],
    [1, 3, 4],
    [2, 6],
]
print(merge_k_sorted_lists(lists))  # [1, 1, 2, 3, 4, 4, 5, 6]
```

### 7.4 Task Scheduler

```python
import heapq
from collections import Counter

def least_interval(tasks, n):
    """Task scheduler: minimum n intervals between identical tasks

    Example: tasks = ["A","A","A","B","B","B"], n = 2
    Result: A B _ A B _ A B -> length 8

    Greedy + max-heap: execute higher-frequency tasks first
    """
    freq = Counter(tasks)
    max_heap = [-count for count in freq.values()]
    heapq.heapify(max_heap)

    time = 0
    cooldown = []  # (available_time, remaining_count)

    while max_heap or cooldown:
        time += 1

        if max_heap:
            count = heapq.heappop(max_heap) + 1  # Negated, so +1 decrements count
            if count != 0:
                cooldown.append((time + n, count))
        # else: idle

        # Return tasks whose cooldown has ended back to the heap
        if cooldown and cooldown[0][0] == time:
            _, count = cooldown.pop(0)
            heapq.heappush(max_heap, count)

    return time

print(least_interval(["A","A","A","B","B","B"], 2))  # 8
print(least_interval(["A","A","A","B","B","B"], 0))  # 6
```

### 7.5 Sliding Window Maximum

```python
from collections import deque

def max_sliding_window(nums, k):
    """Sliding window maximum — O(n)

    Uses a monotonic decreasing deque.
    The heap version is O(n log n), but the deque version is O(n).
    """
    if not nums or k == 0:
        return []

    dq = deque()  # Stores indices (monotonically decreasing)
    result = []

    for i in range(len(nums)):
        # Remove elements outside the window
        while dq and dq[0] < i - k + 1:
            dq.popleft()

        # Remove elements smaller than the new element
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()

        dq.append(i)

        # Add to result once the window is complete
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result

print(max_sliding_window([1, 3, -1, -3, 5, 3, 6, 7], 3))
# [3, 3, 5, 5, 6, 7]

# Heap version (for comparison)
import heapq

def max_sliding_window_heap(nums, k):
    """Heap version — O(n log n)"""
    if not nums or k == 0:
        return []

    heap = []  # (-value, index)
    result = []

    for i in range(len(nums)):
        heapq.heappush(heap, (-nums[i], i))

        if i >= k - 1:
            # Skip elements outside the window
            while heap[0][1] < i - k + 1:
                heapq.heappop(heap)
            result.append(-heap[0][0])

    return result
```

### 7.6 Maximum Performance Team Selection

```python
import heapq

def max_performance(n, speed, efficiency, k):
    """Maximum performance: maximize sum(speed) * min(efficiency)
    with a team of at most k members

    Sort by efficiency in descending order and manage
    the speed sum using a min-heap.
    """
    # Sort by efficiency in descending order
    engineers = sorted(zip(efficiency, speed), reverse=True)
    max_perf = 0
    speed_sum = 0
    min_heap = []

    for eff, spd in engineers:
        heapq.heappush(min_heap, spd)
        speed_sum += spd

        if len(min_heap) > k:
            speed_sum -= heapq.heappop(min_heap)

        max_perf = max(max_perf, speed_sum * eff)

    return max_perf

# Usage example
print(max_performance(6, [2,10,3,1,5,8], [5,4,3,9,7,2], 2))  # 60
```

### 7.7 Dijkstra's Algorithm

```python
import heapq

def dijkstra(graph, start):
    """Dijkstra's shortest path algorithm — O((V+E) log V)

    graph: {node: [(neighbor, weight), ...]}
    """
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    prev = {node: None for node in graph}
    heap = [(0, start)]

    while heap:
        dist, u = heapq.heappop(heap)

        if dist > distances[u]:
            continue  # Skip stale entry (lazy deletion)

        for v, weight in graph[u]:
            new_dist = dist + weight
            if new_dist < distances[v]:
                distances[v] = new_dist
                prev[v] = u
                heapq.heappush(heap, (new_dist, v))

    return distances, prev

def reconstruct_path(prev, start, end):
    """Reconstruct shortest path"""
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = prev[current]
    return path[::-1] if path[-1] == start else []

# Usage example
graph = {
    'A': [('B', 4), ('C', 2)],
    'B': [('D', 3), ('C', 1)],
    'C': [('B', 1), ('D', 5)],
    'D': [],
}
distances, prev = dijkstra(graph, 'A')
print(distances)  # {'A': 0, 'B': 3, 'C': 2, 'D': 6}
print(reconstruct_path(prev, 'A', 'D'))  # ['A', 'C', 'B', 'D']
```

### 7.8 Huffman Coding

```python
import heapq
from collections import Counter

class HuffmanNode:
    def __init__(self, char=None, freq=0, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq

def huffman_encoding(text):
    """Huffman coding — O(n log n)

    Assigns variable-length codes based on character frequency.
    Higher-frequency characters get shorter codes, improving overall compression.
    """
    if not text:
        return {}, ""

    # Frequency count
    freq = Counter(text)

    # Add leaf nodes to the priority queue
    heap = [HuffmanNode(char=ch, freq=f) for ch, f in freq.items()]
    heapq.heapify(heap)

    # Special case: only one unique character
    if len(heap) == 1:
        return {heap[0].char: "0"}, "0" * len(text)

    # Build the Huffman tree
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, merged)

    # Generate codes
    codes = {}
    def build_codes(node, code=""):
        if node.char is not None:
            codes[node.char] = code
            return
        build_codes(node.left, code + "0")
        build_codes(node.right, code + "1")

    build_codes(heap[0])

    # Encode
    encoded = "".join(codes[ch] for ch in text)
    return codes, encoded

# Usage example
text = "this is an example of huffman encoding"
codes, encoded = huffman_encoding(text)
print("Code table:")
for char, code in sorted(codes.items(), key=lambda x: len(x[1])):
    print(f"  '{char}': {code}")
print(f"\nOriginal: {len(text) * 8} bits (ASCII)")
print(f"Compressed: {len(encoded)} bits")
print(f"Compression ratio: {len(encoded) / (len(text) * 8) * 100:.1f}%")
```

---

## 8. Comparison Tables

### Table 1: Heap Operation Complexity

| Operation | Binary Heap | d-ary Heap | Fibonacci | Pairing |
|-----------|-------------|------------|-----------|---------|
| peek | O(1) | O(1) | O(1) | O(1) |
| push | O(log n) | O(log_d n) | O(1) amort. | O(1) |
| pop | O(log n) | O(d log_d n) | O(log n) amort. | O(log n) amort. |
| decrease_key | O(log n) | O(log_d n) | O(1) amort. | O(1) amort. |
| merge | O(n) | O(n) | O(1) | O(1) |
| heapify | O(n) | O(n) | - | - |

### Table 2: Comparison with Sorting Algorithms

| Algorithm | Average | Worst | Space | Stable | Characteristics |
|-----------|---------|-------|-------|--------|-----------------|
| Heapsort | O(n log n) | O(n log n) | O(1) | Unstable | In-place, worst-case guarantee |
| Merge sort | O(n log n) | O(n log n) | O(n) | Stable | Suited for external sort |
| Quicksort | O(n log n) | O(n^2) | O(log n) | Unstable | Fastest in practice |
| Timsort | O(n log n) | O(n log n) | O(n) | Stable | Python/Java standard |
| Introsort | O(n log n) | O(n log n) | O(log n) | Unstable | C++ standard |

### Table 3: Priority Queue Implementations by Language

| Language | Class/Module | Internal Implementation | Notes |
|----------|-------------|------------------------|-------|
| Python | heapq | Binary heap (array) | Min-heap only |
| Java | PriorityQueue | Binary heap | Min-heap, configurable via Comparator |
| C++ | priority_queue | Binary heap | Max-heap by default |
| Go | container/heap | Interface | Implement Push/Pop/Len/Less/Swap |
| Rust | BinaryHeap | Binary heap | Max-heap, use Reverse for min |
| C# | PriorityQueue<T, P> | .NET 6+ | Min-heap |

---

## 9. Anti-Patterns

### Anti-Pattern 1: Sequential push instead of heapify

```python
import heapq

# BAD: n pushes — O(n log n)
heap = []
for x in data:
    heapq.heappush(heap, x)

# GOOD: heapify — O(n)
heap = list(data)
heapq.heapify(heap)

# Performance difference:
# For n = 1,000,000
# BAD:  ~1.2 seconds
# GOOD: ~0.05 seconds (approximately 24x faster)
```

### Anti-Pattern 2: Frequent non-min/max lookups in a heap

```python
# BAD: searching for a specific value in a heap — O(n)
def find_in_heap(heap, target):
    for item in heap:
        if item == target:
            return True
    return False

# Heaps are specialized for min/max retrieval
# If arbitrary element lookup is needed, use a set or dict alongside

# GOOD: combining heap with set
class HeapWithSet:
    def __init__(self):
        self.heap = []
        self.members = set()

    def push(self, val):
        if val not in self.members:
            heapq.heappush(self.heap, val)
            self.members.add(val)

    def pop(self):
        val = heapq.heappop(self.heap)
        self.members.discard(val)
        return val

    def contains(self, val):
        return val in self.members  # O(1)
```

### Anti-Pattern 3: Inappropriate use of nsmallest/nlargest

```python
import heapq

# BAD: using nsmallest when all elements are needed
sorted_data = heapq.nsmallest(len(data), data)  # O(n log n)

# GOOD: use sorted()
sorted_data = sorted(data)  # O(n log n) but with smaller constant factor

# BAD: using nsmallest when only 1 element is needed
minimum = heapq.nsmallest(1, data)[0]  # O(n)

# GOOD: use min/max
minimum = min(data)  # O(n) but with less overhead

# GOOD: nsmallest/nlargest when k is small
top10 = heapq.nsmallest(10, data)  # O(n + 10 log n) ~ O(n)
```

### Anti-Pattern 4: Assuming heap array is sorted

```python
import heapq

# BAD: assuming the heap array is sorted
heap = [1, 3, 2, 5, 8, 7]
heapq.heapify(heap)
print(heap)  # [1, 3, 2, 5, 8, 7] — NOT sorted!

# Heap order property: parent <= child, but sibling order is not guaranteed
# heap[0] is the minimum, but heap[1] is not necessarily the second smallest

# GOOD: if sorted order is needed, repeatedly pop
sorted_result = []
temp = list(heap)
heapq.heapify(temp)
while temp:
    sorted_result.append(heapq.heappop(temp))
print(sorted_result)  # [1, 2, 3, 5, 7, 8]
```

### Anti-Pattern 5: Substituting priority updates with re-insertion

```python
import heapq

# BAD: updating priority by just adding a new entry
# -> old entry remains in the heap, causing a memory leak
heap = []
heapq.heappush(heap, (5, "task_A"))
heapq.heappush(heap, (3, "task_A"))  # Intended as priority update
# -> (5, "task_A") remains in the heap

# GOOD: use the lazy deletion pattern
# -> LazyDeletionPQ (see section above)
#
# BETTER: use an indexed heap with decrease_key
# -> IndexedMinHeap (see section above)
```


---

## Practical Exercises

### Exercise 1: Basic Implementation

Implement code that satisfies the following requirements.

**Requirements:**
- Validate input data
- Implement proper error handling
- Write test code

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
        assert False, "Exception should have been raised"
    except ValueError:
        pass

    print("All tests passed!")

test_exercise1()
```

### Exercise 2: Advanced Patterns

Extend the basic implementation by adding the following features.

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
    assert ex.add("d", 4) == False  # Size limit
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
    """Efficient search using hash map"""
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

**Key Points:**
- Be mindful of algorithm complexity
- Choose appropriate data structures
- Measure the effect with benchmarks
---

## 10. FAQ

### Q1: Why is heapify O(n)?

**A:** It performs sift-down from the bottom up. Leaf nodes (approximately n/2) require no sift-down. The number of nodes at height h is n/2^(h+1), and the sift-down cost is O(h). The total is Sigma(h * n/2^(h+1)) = O(n). Intuitively, most nodes are at low heights, so their sift-down distances are short.

### Q2: Does Python's heapq have a max-heap?

**A:** The standard library provides only a min-heap. Ways to implement a max-heap:
1. **Sign negation**: insert with `heappush(h, -val)`, retrieve with `-heappop(h)`. Most common approach
2. **Tuple negation**: `heappush(h, (-priority, item))` for custom objects
3. **Third-party**: `heapdict` package or custom class
4. **Wrapper class**: wrap with a class that reverses `__lt__`

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass(order=True)
class MaxHeapItem:
    priority: int = field(compare=True)
    item: Any = field(compare=False)

    def __post_init__(self):
        self.priority = -self.priority  # Sign negation
```

### Q3: When to use a heap vs. a BST?

**A:**
- **Heap**: when only the min/max value is needed -- O(1) lookup, O(log n) insert/delete
- **BST**: when range queries, in-order traversal, or k-th element are needed -- O(log n) for various operations
- **Heap advantages**: implemented as an array, memory-efficient, cache-friendly
- **BST advantages**: O(log n) search/delete of arbitrary elements; heaps require O(n)

### Q4: What is the optimal solution for the Top-K problem?

**A:** It depends on the size of K:
- **k = 1**: `min()` / `max()` in O(n)
- **k is small**: heap of size k in O(n log k)
- **k ~ n/2**: Quick Select in O(n) average
- **k ~ n**: `sorted()` in O(n log n)

### Q5: Why is heapsort rarely used in practice?

**A:** Heapsort has excellent properties with O(n log n) worst-case guarantee and O(1) extra memory, but it is slower than quicksort and Timsort in practice. Reasons:
1. **Poor cache efficiency**: the heap's access pattern is non-local (parent-child relationships are far apart in memory)
2. **Difficult branch prediction**: comparison patterns in sift-down are hard to predict
3. **Unstable**: relative order of equal keys is not preserved
However, it is effective when memory constraints are strict or worst-case guarantees are required. C++'s `std::sort` internally uses introsort (a hybrid of quicksort and heapsort).

### Q6: How to perform efficient external sorting with heaps?

**A:** K-way merge:
1. Split the large file into chunks that fit in RAM
2. Sort each chunk in memory and write to file
3. Manage the head elements of K chunks using a min-heap of size K
4. Extract the minimum from the heap and write to the output file
5. Add the next element from the extracted chunk to the heap

Python's `heapq.merge()` provides exactly this operation.

---


## FAQ

### Q1: What is the most important point when learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory, but by actually writing code and verifying its behavior.

### Q2: What are common mistakes that beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving to the next step.

### Q3: How is this used in practice?

Knowledge of this topic is frequently applied in daily development work. It becomes particularly important during code reviews and architecture design.

---

## 11. Summary

| Item | Key Point |
|------|-----------|
| Binary heap | Complete binary tree. Efficiently represented as an array. No pointers needed |
| Min/max heap | Root is the min/max value. O(1) lookup |
| sift-up/down | Restoring heap property after insertion/deletion -- O(log n) |
| heapify | Bottom-up construction in O(n). Faster than sequential push O(n log n) |
| Heapsort | O(n log n), in-place, unstable. Worst-case guarantee |
| Priority queue | Implemented with heaps. Lazy deletion and indexed heaps are also important |
| d-ary heap | Larger d is advantageous when decrease_key is frequent |
| Pairing heap | O(1) merge. Simplified version of Fibonacci heap |
| Practical applications | Top-K, median, Dijkstra, Huffman, task scheduler |

---

## Recommended Next Guides

- [Graphs -- Representations and Weighted Graphs](./06-graphs.md)
- [Shortest Paths -- Dijkstra and Heap Utilization](../02-algorithms/03-shortest-path.md)

---

## References

1. Cormen, T.H. et al. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. -- Chapter 6 "Heapsort", Chapter 19 "Fibonacci Heaps"
2. Williams, J.W.J. (1964). "Algorithm 232: Heapsort." *Communications of the ACM*, 7(6), 347-348.
3. Fredman, M.L. & Tarjan, R.E. (1987). "Fibonacci heaps and their uses in improved network optimization algorithms." *Journal of the ACM*, 34(3), 596-615.
4. Python Documentation. "heapq --- Heap queue algorithm." --- https://docs.python.org/3/library/heapq.html
5. Fredman, M.L. et al. (1986). "The pairing heap: A new form of self-adjusting heap." *Algorithmica*, 1(1), 111-129.
