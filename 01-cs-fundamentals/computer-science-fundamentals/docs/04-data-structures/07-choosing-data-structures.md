# How to Choose Data Structures

> Choosing the right data structure is as important as choosing the right algorithm.
> --- Niklaus Wirth, "Algorithms + Data Structures = Programs" (1976)

## Learning Objectives

- [ ] Select the optimal data structure based on requirements
- [ ] Understand the trade-offs of each data structure quantitatively
- [ ] Organize selection criteria systematically and develop a decision framework
- [ ] Immediately identify the appropriate structure for common real-world scenarios
- [ ] Recognize and avoid anti-patterns

## Prerequisites


---

## 1. The Importance of Data Structure Selection

### 1.1 Why Data Structure Choice Is Decisive

In software engineering, the choice of data structure fundamentally affects the performance, maintainability, and extensibility of an entire system. As Niklaus Wirth conveyed through his book title "Algorithms + Data Structures = Programs," algorithms and data structures are two sides of the same coin, and optimizing only one side yields insufficient results.

The impact of data structure selection can be organized along the following four axes.

```
Impact scope of data structure selection:

  +------------------------------------------------------+
  |              Data Structure Selection                  |
  +----------+-----------+-----------+----------+---------+
             |           |           |          |
             v           v           v          v
  +--------------+ +----------+ +----------+ +----------+
  | Time         | | Space    | |Readability| |Extensi-  |
  | Complexity   | |Complexity| |          | |bility    |
  |              | |          | |          | |          |
  | Execution    | | Memory   | | Code     | | Adapta-  |
  | speed per    | | usage &  | | clarity  | | bility to|
  | operation    | | locality | |          | | changes  |
  +--------------+ +----------+ +----------+ +----------+
```

#### Time Complexity Perspective

Even for the same "value lookup" operation, complexity varies dramatically depending on the data structure used. Linear search on an unsorted array is O(n), but a hash table achieves expected O(1), and a balanced BST completes in O(log n). Assuming 100,000 searches on a dataset of n=100,000 items, the theoretical number of comparisons is as follows:

| Data Structure | 1 Search | 100,000 Searches |
|---|---|---|
| Unsorted array (linear search) | Average 50,000 | ~5 billion |
| Sorted array (binary search) | ~17 | ~1.7 million |
| Hash table | Expected 1 | ~100,000 |
| Balanced BST | ~17 | ~1.7 million |

This difference is often imperceptible with small-scale data, but it manifests exponentially as data volume increases.

#### Space Complexity Perspective

Memory usage is another factor that cannot be ignored. While hash tables achieve fast lookups, they carry additional memory overhead for hash value storage, chaining pointers, and unused slots to maintain load factor. Arrays, on the other hand, store elements in contiguous memory regions, making them memory-efficient and more likely to benefit from CPU cache.

In environments with limited memory, such as embedded systems or mobile applications, there are many situations where data structures must be chosen from the perspective of space efficiency.

#### Readability Perspective

Data structure selection directly affects code readability. A lookup table using Python's `dict` is more intent-revealing and maintainable than a chain of if-elif statements. Similarly, deduplication using `set` expresses the intent more concisely than manual loops over lists.

#### Extensibility Perspective

The data structures chosen in the initial design are very costly to change later. Once they become embedded in API interfaces or database schemas, changes to internal structures propagate throughout the entire system. Therefore, it is important to make selections that anticipate future requirement changes.

### 1.2 What Happens When You Choose Wrong

Data structure selection mistakes manifest in the following ways:

1. **Performance collapse**: Response time degrades sharply as data volume increases. A typical example is a system that relies heavily on list membership checks, which could be solved with a hash set.
2. **Memory waste**: Selecting structures with unnecessary redundancy. For example, preparing a huge hash table for a small number of fixed keys.
3. **Code complexity**: "Workaround code" proliferates to compensate for inappropriate data structures, increasing maintenance costs.
4. **Concurrency difficulties**: Choosing non-thread-safe data structures, later requiring the addition of locking mechanisms.

### 1.3 Approach of This Chapter

This chapter organizes data structure selection as a "systematic decision process" rather than relying on "intuition and experience." Specifically, we proceed with the following steps:

1. Clarify the axes of selection criteria (Section 2)
2. Compare the characteristics of major data structures quantitatively (Section 3)
3. Provide recommendations per use case (Section 4)
4. Validate with implementation examples and benchmarks (Section 5)
5. Systematize with a decision flowchart (Section 6)
6. Present common anti-patterns (Section 7)

---

## 2. Selection Criteria

When selecting data structures, multiple criteria must be considered simultaneously. Here we explain six major criteria axes.

### 2.1 Operation Frequency and Types

The most fundamental criterion is "which operations are performed most frequently." Each data structure has strengths and weaknesses in different operations, so the complexity of the dominant operations becomes the deciding factor.

```
Operation classification matrix:

  +--------------------------------------------------------+
  |                    Operation Types                       |
  +--------------+--------------+--------------------------+
  |  Read        |  Write       |     Special Operations   |
  +--------------+--------------+--------------------------+
  | - Index      | - Insert     | - Sort                   |
  |   access     | - Delete     | - Range search           |
  | - Search     | - Update     | - Prefix/Substring match |
  | - Min/Max    | - Prepend    | - Set ops (union/        |
  | - Ordered    | - Append     |   intersection/diff)     |
  |   traversal  | - Mid insert | - Merge                  |
  | - Random     |              | - Top-K / Median         |
  |   access     |              | - Rank                   |
  +--------------+--------------+--------------------------+
```

For example, if the requirement is "insertions are rare but searches are frequent," you should choose a data structure with fast searches even if insertion cost is high (sorted array + binary search, hash table, etc.). Conversely, if "insertions and deletions are frequent but searches are almost never performed," linked lists or queues become candidates.

The following code example demonstrates the basic thinking behind structure selection based on operation patterns.

```python
"""
Code Example 1: Comparing data structure selection by operation pattern

Requirement: Frequently check existence against a large set of integer data
"""
import time
from typing import List, Set


def benchmark_membership_test(data_list: List[int], data_set: Set[int],
                               queries: List[int]) -> None:
    """Demonstrate the performance difference between list and set membership tests"""

    # --- Membership test with list: O(n) ---
    start = time.perf_counter()
    count_list = 0
    for q in queries:
        if q in data_list:
            count_list += 1
    elapsed_list = time.perf_counter() - start

    # --- Membership test with set: Expected O(1) ---
    start = time.perf_counter()
    count_set = 0
    for q in queries:
        if q in data_set:
            count_set += 1
    elapsed_set = time.perf_counter() - start

    print(f"Data count:   {len(data_list):>10,}")
    print(f"Query count:  {len(queries):>10,}")
    print(f"List search:  {elapsed_list:.4f} sec  (hits: {count_list})")
    print(f"Set search:   {elapsed_set:.4f} sec  (hits: {count_set})")
    print(f"Speed ratio:  {elapsed_list / max(elapsed_set, 1e-9):.1f}x")
    print()


if __name__ == "__main__":
    import random
    random.seed(42)

    for size in [1_000, 10_000, 100_000]:
        data = list(range(size))
        data_set = set(data)
        queries = [random.randint(0, size * 2) for _ in range(10_000)]
        benchmark_membership_test(data, data_set, queries)
```

Running this code confirms that the performance gap between list and set widens as data count increases. At around 1,000 items, the difference is only a few times, but at 100,000 items, it becomes several hundred times or more.

### 2.2 Data Size and Growth Patterns

The scale and growth pattern of data have a significant impact on selection.

| Data Scale | Characteristics | Recommended Strategy |
|---|---|---|
| Small (~1,000) | Constant factors dominate | Simple structures (arrays) suffice |
| Medium (1,000~100,000) | Order of complexity begins to matter | Choose appropriate structure based on requirements |
| Large (100,000~) | Order difference dominates | O(1) or O(log n) structures are essential |
| Very large (hundreds of millions~) | May not fit in memory | Consider external memory structures (B+ tree, etc.) |

An important caveat: **for small-scale data, simpler data structures may actually be faster**. This is due to constant factors and cache efficiency. Array linear search is O(n), but when n is small, the benefit of contiguous memory access fitting in the CPU cache can make it faster than O(1) hash table lookups.

### 2.3 Memory Usage and Locality

Memory usage characteristics differ significantly across data structures. Below is a summary of typical memory overhead in Python.

```
Memory characteristics of each structure in Python (approximate):

  Structure       Additional overhead per element      Memory locality
  ----------      --------------------------------      ---------------
  list            8 bytes (pointer array)                High
  tuple           8 bytes (pointer array)                High (immutable)
  set             Hash value stored per element          Moderate
  dict            Key + value + hash value               Moderate
  deque           Block-based linked list                Moderate
  Linked list     Node object + pointer                  Low

  Note: Since all Python objects are heap-allocated,
  the locality is not as good as C/C++ arrays
```

Modern processors have hierarchical cache mechanisms, and memory access locality significantly affects performance. Arrays use contiguous memory regions, efficiently utilizing cache lines. In contrast, linked lists and tree structures have scattered nodes, leading to frequent cache misses.

### 2.4 Order Preservation

Whether data order needs to be preserved is an important selection criterion.

- **Order not required**: Hash table, hash set are optimal. Fastest when even insertion order is unnecessary.
- **Insertion order preserved**: Python 3.7+ `dict` preserves insertion order. `collections.OrderedDict` is equivalent.
- **Sorted order maintained**: `sortedcontainers.SortedList`, balanced BST (`TreeMap`, etc.), B+ tree.
- **Custom order**: Heap (priority queue) manages order based on priority.

### 2.5 Concurrency and Thread Safety

In multi-threaded environments, thread safety of data structures becomes important.

| Strategy | Characteristics | Application |
|---|---|---|
| Lock-free (immutable) | Use immutable data structures | Sharing read-only data |
| Global lock | Acquire lock for all operations | Low-contention simple sharing |
| Fine-grained locking | Minimal locking per operation | High-contention read-write mix |
| Lock-free structures | Non-lock sync via CAS, etc. | Ultra-high concurrency scenarios |
| Thread-local | Independent copy per thread | Write-heavy scenarios |

In Python, due to the GIL (Global Interpreter Lock), multi-threading benefits are limited for CPU-bound operations, but it remains an important consideration for I/O-bound processing or when using `multiprocessing`.

`queue.Queue`, `queue.PriorityQueue`, and `collections.deque` (append/pop on both ends are thread-safe) can be used as thread-safe data structures.

### 2.6 Persistence and Serialization

When saving data to disk or transmitting it over a network, ease of serialization also becomes a selection criterion.

- **Arrays/Dictionaries**: Easily serializable with JSON, MessagePack, Protocol Buffers, etc.
- **Trees/Graphs**: Require creative representation of inter-node references.
- **Custom structures**: May require custom serialization logic.

Additionally, the concept of persistent data structures exists. These create new versions on modification while keeping past versions accessible. They are commonly used in functional programming, and Git's internal data model is also based on this concept.

---

## 3. Characteristics Comparison of Major Data Structures

### 3.1 Comprehensive Complexity Comparison Table

The following is a comprehensive table of operation complexities for major data structures. It can be used as a basic reference for selection.

```
Complexity overview of major data structures (average / worst):

  +------------------+--------------+--------------+--------------+--------------+----------+
  | Data Structure   | Access       | Search       | Insert       | Delete       | Space    |
  +------------------+--------------+--------------+--------------+--------------+----------+
  | Static array     | O(1)/O(1)    | O(n)/O(n)    | O(n)/O(n)    | O(n)/O(n)    | O(n)     |
  | Dynamic array    | O(1)/O(1)    | O(n)/O(n)    | O(1)*/O(n)   | O(n)/O(n)    | O(n)     |
  | (list)           |              |              |              |              |          |
  | Singly linked    | O(n)/O(n)    | O(n)/O(n)    | O(1)/O(1)    | O(1)/O(1)    | O(n)     |
  | list             |              |              |              |              |          |
  | Doubly linked    | O(n)/O(n)    | O(n)/O(n)    | O(1)/O(1)    | O(1)/O(1)    | O(n)     |
  | list             |              |              |              |              |          |
  | Stack            | O(n)/O(n)    | O(n)/O(n)    | O(1)/O(1)    | O(1)/O(1)    | O(n)     |
  | Queue            | O(n)/O(n)    | O(n)/O(n)    | O(1)/O(1)    | O(1)/O(1)    | O(n)     |
  | Hash table       | N/A          | O(1)/O(n)    | O(1)/O(n)    | O(1)/O(n)    | O(n)     |
  | Hash set         | N/A          | O(1)/O(n)    | O(1)/O(n)    | O(1)/O(n)    | O(n)     |
  | BST (unbalanced) | O(log n)/O(n)| O(log n)/O(n)| O(log n)/O(n)| O(log n)/O(n)| O(n)     |
  | Balanced BST     | O(log n)     | O(log n)     | O(log n)     | O(log n)     | O(n)     |
  | Binary heap      | O(1)^        | O(n)/O(n)    | O(log n)     | O(log n)     | O(n)     |
  | Trie             | N/A          | O(m)/O(m)    | O(m)/O(m)    | O(m)/O(m)    | O(SIGMA) |
  | B-tree / B+ tree | O(log n)     | O(log n)     | O(log n)     | O(log n)     | O(n)     |
  | Skip list        | O(log n)     | O(log n)     | O(log n)     | O(log n)     | O(n)     |
  +------------------+--------------+--------------+--------------+--------------+----------+

  * Amortized complexity for tail append  ^ O(1) for min or max only
  m = key length  SIGMA = total characters across all keys
```

### 3.2 Optimal Structure by Operation

The optimal data structure for each operation is as follows.

**Index access (get the i-th element)**
- Best: Array / Dynamic array -> O(1)
- Runner-up: Skip list -> O(log n) (with extended index operations)
- Unsuitable: Linked list -> O(n)

**Value search (check if a value exists)**
- Best: Hash table / Hash set -> Expected O(1)
- Runner-up: Balanced BST / Sorted array -> O(log n)
- Unsuitable: Unsorted array / Linked list -> O(n)

**Min/Max retrieval**
- Best: Binary heap -> O(1) (one side only)
- Runner-up: Balanced BST -> O(log n) (can retrieve both)
- Unsuitable: Hash table -> O(n)

**Range search (get all elements between a and b)**
- Best: Balanced BST / B+ tree -> O(log n + k) (k is the result count)
- Runner-up: Sorted array + binary search -> O(log n + k)
- Unsuitable: Hash table -> O(n)

**Insertion/Deletion at the front**
- Best: Linked list / deque -> O(1)
- Runner-up: None
- Unsuitable: Array -> O(n) (requires shifting all elements)

**Dynamic maintenance of sorted order**
- Best: Balanced BST / Skip list -> O(log n)
- Runner-up: Sorted array (insertion is O(n) but search is O(log n))
- Unsuitable: Hash table (has no order)

### 3.3 Correspondence in Python Standard Library

Python's standard library provides many data structures as built-in types or modules.

```
Data structure correspondence in Python standard library:

  Abstract Structure            Python Implementation
  -----------------------       ----------------------------------
  Dynamic array                 list
  Immutable array               tuple
  Hash table                    dict
  Hash set                      set / frozenset
  Double-ended queue (deque)    collections.deque
  Heap (priority queue)         heapq module (list-based)
  Stack                         list (append/pop at end)
  FIFO queue                    collections.deque (or queue.Queue)
  Ordered dictionary            dict (3.7+) / collections.OrderedDict
  Default dictionary            collections.defaultdict
  Counter                       collections.Counter
  Named tuple                   collections.namedtuple / typing.NamedTuple
  Frozen set                    frozenset
  Bit array                     int bit operations / array.array
  Typed array                   array.array / numpy.ndarray (external)

  Note: Balanced BST, Trie, Skip list, etc. are not included in the
  standard library, requiring sortedcontainers (external) or custom
  implementations
```

### 3.4 Cross-Language Correspondence

When programming in multiple languages, note that equivalent structures are provided under different names.

| Abstract Structure | Python | Java | C++ | Go | Rust |
|---|---|---|---|---|---|
| Dynamic array | `list` | `ArrayList` | `std::vector` | `slice` | `Vec<T>` |
| Linked list | (custom) | `LinkedList` | `std::list` | `list.List` | `LinkedList<T>` |
| Hash map | `dict` | `HashMap` | `std::unordered_map` | `map` | `HashMap<K,V>` |
| Hash set | `set` | `HashSet` | `std::unordered_set` | (use map) | `HashSet<T>` |
| Sorted map | (external) | `TreeMap` | `std::map` | (external) | `BTreeMap<K,V>` |
| Heap | `heapq` | `PriorityQueue` | `std::priority_queue` | `heap` | `BinaryHeap<T>` |
| Double-ended queue | `deque` | `ArrayDeque` | `std::deque` | (external) | `VecDeque<T>` |
| Stack | `list` | `Stack`/`Deque` | `std::stack` | `slice` | `Vec<T>` |

---

## 4. Use Case Guide

### 4.1 Search Optimization

Search is one of the most frequently performed operations, and the optimal structure differs depending on the use case.

#### Exact Match Search

Check whether an element exists, or retrieve the value corresponding to a key.

- **Recommended**: Hash table (`dict`) / Hash set (`set`)
- **Complexity**: Expected O(1)
- **Caveat**: Depends on hash function quality. Can degrade to O(n) in the worst case, but Python's built-in hash function is sufficiently high quality.

```python
"""
Code Example 2: Optimal structures for different search patterns
"""


# --- Exact match search: dict / set are optimal ---
def build_user_lookup(users: list[dict]) -> dict[int, dict]:
    """Build an ID -> user info lookup table from a user list"""
    return {user["id"]: user for user in users}


# Usage example
users = [
    {"id": 1, "name": "Alice", "email": "alice@example.com"},
    {"id": 2, "name": "Bob", "email": "bob@example.com"},
    {"id": 3, "name": "Charlie", "email": "charlie@example.com"},
]
lookup = build_user_lookup(users)
print(lookup[2])  # O(1) retrieval: {'id': 2, 'name': 'Bob', ...}


# --- Multi-criteria search: Building compound indexes ---
from collections import defaultdict


def build_multi_index(users: list[dict]) -> dict:
    """Build indexes on multiple keys"""
    index = {
        "by_id": {},
        "by_email": {},
        "by_name_prefix": defaultdict(list),
    }
    for user in users:
        index["by_id"][user["id"]] = user
        index["by_email"][user["email"]] = user
        # Index by each prefix of the name (simplified Trie-like approach)
        name = user["name"].lower()
        for i in range(1, len(name) + 1):
            index["by_name_prefix"][name[:i]].append(user)
    return index


multi_idx = build_multi_index(users)
print(multi_idx["by_email"]["bob@example.com"])     # O(1)
print(multi_idx["by_name_prefix"]["ch"])             # O(1) + result count
```

#### Range Search

Search based on a range of values, such as "between 10 and 20" or "dates within last month."

- **Recommended**: Sorted array + `bisect` / Balanced BST
- **Complexity**: O(log n + k) (k is the result count)
- **Not possible with hash tables**: Hash tables have no order and cannot handle range searches.

```python
"""
Code Example 3: Range search implementation (using bisect module)
"""
import bisect
from datetime import datetime, timedelta


class TimeSeriesIndex:
    """An index for efficient timestamp-based range queries"""

    def __init__(self):
        self._timestamps: list[float] = []
        self._values: list[dict] = []

    def insert(self, timestamp: datetime, value: dict) -> None:
        """Insert data while maintaining timestamp order"""
        ts = timestamp.timestamp()
        pos = bisect.bisect_left(self._timestamps, ts)
        self._timestamps.insert(pos, ts)
        self._values.insert(pos, value)

    def range_query(self, start: datetime, end: datetime) -> list[dict]:
        """Retrieve data within the specified time range - O(log n + k)"""
        start_ts = start.timestamp()
        end_ts = end.timestamp()
        left = bisect.bisect_left(self._timestamps, start_ts)
        right = bisect.bisect_right(self._timestamps, end_ts)
        return self._values[left:right]

    def __len__(self) -> int:
        return len(self._timestamps)


# Usage example
index = TimeSeriesIndex()
base_time = datetime(2025, 1, 1)

# Insert data
for i in range(1000):
    t = base_time + timedelta(hours=i)
    index.insert(t, {"event_id": i, "time": t.isoformat()})

# Range search: Events in the first 24 hours
results = index.range_query(
    base_time,
    base_time + timedelta(hours=24)
)
print(f"Events in the first 24 hours: {len(results)}")  # 25
print(f"First event: {results[0]}")
print(f"Last event: {results[-1]}")
```

#### Prefix Search

- **Recommended**: Trie
- **Complexity**: O(m) (m is the prefix length)
- **Use cases**: Autocomplete, dictionary prefix matching, IP address routing tables

### 4.2 Ordered Data Management

Scenarios where data must be kept sorted at all times with dynamic additions and deletions.

| Requirement | Recommended Structure | Reason |
|---|---|---|
| Sort static data | Array + sort | O(n log n) for sorting, then O(log n) for search |
| Dynamically maintain sorted order | Balanced BST / SortedList | Insert/delete at O(log n) |
| Dynamic Top-K management | Heap | Insert O(log K), get min/max O(1) |
| Rank query | Order-statistic tree | Get k-th element at O(log n) |

### 4.3 FIFO / LIFO Patterns

#### Stack (LIFO: Last In, First Out)

Typical use cases:
- Function call management
- Undo/Redo functionality
- Parenthesis matching
- DFS (Depth-First Search)
- Expression evaluation (Reverse Polish Notation)

Python implementation: `list`'s `append()` / `pop()` is simplest.

#### Queue (FIFO: First In, First Out)

Typical use cases:
- Task queue / Job queue
- BFS (Breadth-First Search)
- Event processing
- Buffering

Python implementation: `collections.deque` is optimal. `list`'s `pop(0)` is O(n) and thus inappropriate.

#### Priority Queue

Typical use cases:
- Dijkstra's algorithm
- Task scheduling (with priorities)
- Real-time Top-K
- Event-driven simulation

Python implementation: `heapq` module.

```python
"""
Code Example 4: Queue selection based on use case
"""
import heapq
from collections import deque


# --- Standard queue (FIFO): deque ---
class TaskQueue:
    """Simple first-in-first-out task queue"""

    def __init__(self):
        self._queue = deque()

    def enqueue(self, task: str) -> None:
        self._queue.append(task)

    def dequeue(self) -> str:
        if not self._queue:
            raise IndexError("Queue is empty")
        return self._queue.popleft()  # O(1)

    def __len__(self) -> int:
        return len(self._queue)


# --- Priority queue: heapq ---
class PriorityTaskQueue:
    """Priority task queue (lower number = higher priority)"""

    def __init__(self):
        self._heap: list[tuple[int, int, str]] = []
        self._counter = 0  # Tiebreaker (stability for same priority)

    def enqueue(self, task: str, priority: int) -> None:
        heapq.heappush(self._heap, (priority, self._counter, task))
        self._counter += 1

    def dequeue(self) -> tuple[int, str]:
        if not self._heap:
            raise IndexError("Queue is empty")
        priority, _, task = heapq.heappop(self._heap)  # O(log n)
        return priority, task

    def peek(self) -> tuple[int, str]:
        if not self._heap:
            raise IndexError("Queue is empty")
        priority, _, task = self._heap[0]  # O(1)
        return priority, task

    def __len__(self) -> int:
        return len(self._heap)


# Usage example
print("=== Standard Queue ===")
tq = TaskQueue()
for task in ["Send email", "Write log", "Update DB"]:
    tq.enqueue(task)
while len(tq) > 0:
    print(f"  Processing: {tq.dequeue()}")

print("\n=== Priority Queue ===")
pq = PriorityTaskQueue()
pq.enqueue("Bug fix", priority=1)              # Highest priority
pq.enqueue("New feature development", priority=3)  # Low priority
pq.enqueue("Security fix", priority=1)         # Highest priority
pq.enqueue("Documentation update", priority=5) # Lowest priority
pq.enqueue("Performance improvement", priority=2)

while len(pq) > 0:
    pri, task = pq.dequeue()
    print(f"  Priority {pri}: {task}")
```

### 4.4 Set Operations

Scenarios requiring deduplication, union, intersection, difference, and other set operations.

- **Recommended**: `set` / `frozenset`
- **Use cases**: Tag filtering, permission management, data deduplication, extracting common elements

```python
# Set operation usage examples
admin_permissions = {"read", "write", "delete", "admin"}
editor_permissions = {"read", "write"}
viewer_permissions = {"read"}

# Union: All permissions a user has
user_roles = ["editor", "viewer"]
all_permissions: set[str] = set()
role_map = {
    "admin": admin_permissions,
    "editor": editor_permissions,
    "viewer": viewer_permissions,
}
for role in user_roles:
    all_permissions |= role_map[role]  # O(len(smaller_set))
print(f"All permissions: {all_permissions}")  # {'read', 'write'}

# Intersection: Permissions common to all roles
common = admin_permissions & editor_permissions & viewer_permissions
print(f"Common permissions: {common}")  # {'read'}

# Difference: Permissions only admin has
admin_only = admin_permissions - editor_permissions
print(f"Admin-only: {admin_only}")  # {'delete', 'admin'}
```

### 4.5 Caching

Scenarios requiring data retention and eviction based on access frequency.

- **LRU cache**: `functools.lru_cache` or `collections.OrderedDict`
- **Internal structure**: Hash table + Doubly linked list
- **Complexity**: get/put both O(1)

### 4.6 Graph Representation

Scenarios requiring representation of relationships between entities.

| Representation | Applicable Scenario | Space Complexity |
|---|---|---|
| Adjacency matrix | Dense graphs, frequent edge existence checks | O(V^2) |
| Adjacency list | Sparse graphs, frequent neighbor enumeration | O(V + E) |
| Edge list | Kruskal's algorithm, etc. | O(E) |
| Adjacency map | Weighted graphs, both edge existence and enumeration | O(V + E) |

Most practical graphs (social networks, web link structures, etc.) are sparse, so adjacency lists or adjacency maps are the default choice.

---

## 5. Implementation Examples and Benchmark Comparisons

### 5.1 Applying Different Data Structures to the Same Problem

Here we compare solutions using different data structures for the specific problem of "deduplication."

```python
"""
Code Example 5: Deduplication implementation comparison and benchmark
"""
import time
import random
from typing import Callable


def deduplicate_with_list(data: list[int]) -> list[int]:
    """Deduplication using list - O(n^2)"""
    result: list[int] = []
    for item in data:
        if item not in result:  # O(n) search occurs each time
            result.append(item)
    return result


def deduplicate_with_set(data: list[int]) -> list[int]:
    """Deduplication using set (preserves insertion order) - O(n)"""
    seen: set[int] = set()
    result: list[int] = []
    for item in data:
        if item not in seen:  # O(1) search
            seen.add(item)
            result.append(item)
    return result


def deduplicate_with_dict(data: list[int]) -> list[int]:
    """Deduplication using dict (Python 3.7+ preserves insertion order) - O(n)"""
    return list(dict.fromkeys(data))


def benchmark(func: Callable, data: list[int], label: str) -> float:
    """Measure function execution time"""
    start = time.perf_counter()
    result = func(data)
    elapsed = time.perf_counter() - start
    return elapsed


if __name__ == "__main__":
    random.seed(42)

    print("Deduplication Benchmark")
    print("=" * 60)

    for size in [100, 1_000, 10_000]:
        # Generate data with 50% duplicates
        data = [random.randint(0, size // 2) for _ in range(size)]

        print(f"\nData count: {size:>10,}  (unique: ~{size // 2:,})")
        print("-" * 60)

        # List approach
        t_list = benchmark(deduplicate_with_list, data, "list")
        print(f"  list approach:  {t_list:.6f} sec")

        # Set approach
        t_set = benchmark(deduplicate_with_set, data, "set")
        print(f"  set approach:   {t_set:.6f} sec")

        # Dict approach
        t_dict = benchmark(deduplicate_with_dict, data, "dict")
        print(f"  dict approach:  {t_dict:.6f} sec")

        if t_set > 0:
            print(f"  list/set ratio: {t_list / t_set:.1f}x")
```

Key takeaways from this example:

1. **Small scale (100 items)**: Virtually no noticeable difference across approaches.
2. **Medium scale (1,000 items)**: The list approach begins to show clear slowdown.
3. **Large scale (10,000 items)**: The list approach is catastrophically slow due to O(n^2), showing hundreds of times difference from set/dict approaches.

### 5.2 LRU Cache Implementation Comparison

Caching is an important component in many systems. Here we implement an LRU (Least Recently Used) cache using different approaches and compare performance.

```python
"""
Code Example 6: LRU cache implementation
O(1) implementation using hash table + doubly linked list
"""
from collections import OrderedDict
from typing import Optional, Hashable


class LRUCache:
    """
    LRU cache using OrderedDict

    OrderedDict internally maintains a doubly linked list,
    enabling O(1) element order changes.
    Combined with hash table O(1) lookup,
    this achieves O(1) for both get and put.
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer")
        self._capacity = capacity
        self._cache: OrderedDict[Hashable, object] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: Hashable) -> Optional[object]:
        """
        Retrieve value for key (O(1))
        Accessed element is moved to the end (most recent)
        """
        if key in self._cache:
            self._cache.move_to_end(key)  # O(1): Mark as most recent
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, key: Hashable, value: object) -> None:
        """
        Store key-value pair (O(1))
        Evicts the oldest element when capacity is exceeded
        """
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._capacity:
            self._cache.popitem(last=False)  # O(1): Remove oldest element

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def __len__(self) -> int:
        return len(self._cache)

    def __repr__(self) -> str:
        return (f"LRUCache(capacity={self._capacity}, size={len(self)}, "
                f"hit_rate={self.hit_rate:.2%})")


# Usage example
cache = LRUCache(capacity=3)
cache.put("a", 1)
cache.put("b", 2)
cache.put("c", 3)
print(cache.get("a"))   # 1 ("a" becomes most recent)
cache.put("d", 4)       # Capacity exceeded -> "b" is evicted
print(cache.get("b"))   # None (evicted)
print(cache.get("c"))   # 3
print(cache)            # LRUCache(capacity=3, size=3, hit_rate=66.67%)
```

### 5.3 Insert/Search Benchmark by Data Structure

The following is a benchmark comparing insertion and search performance across major data structures.

```python
"""
Code Example 7: Insert/search benchmark for major data structures
"""
import time
import random
import bisect
from collections import deque


def benchmark_insert_search(n: int) -> dict:
    """Measure insertion and search time for each data structure"""
    random.seed(42)
    data = [random.randint(0, n * 10) for _ in range(n)]
    queries = [random.randint(0, n * 10) for _ in range(min(n, 10_000))]

    results = {}

    # --- list ---
    start = time.perf_counter()
    lst: list[int] = []
    for item in data:
        lst.append(item)
    insert_time = time.perf_counter() - start

    start = time.perf_counter()
    for q in queries:
        _ = q in lst
    search_time = time.perf_counter() - start
    results["list"] = {"insert": insert_time, "search": search_time}

    # --- set ---
    start = time.perf_counter()
    s: set[int] = set()
    for item in data:
        s.add(item)
    insert_time = time.perf_counter() - start

    start = time.perf_counter()
    for q in queries:
        _ = q in s
    search_time = time.perf_counter() - start
    results["set"] = {"insert": insert_time, "search": search_time}

    # --- dict ---
    start = time.perf_counter()
    d: dict[int, bool] = {}
    for item in data:
        d[item] = True
    insert_time = time.perf_counter() - start

    start = time.perf_counter()
    for q in queries:
        _ = q in d
    search_time = time.perf_counter() - start
    results["dict"] = {"insert": insert_time, "search": search_time}

    # --- Sorted list + bisect ---
    start = time.perf_counter()
    sorted_lst: list[int] = []
    for item in data:
        bisect.insort(sorted_lst, item)
    insert_time = time.perf_counter() - start

    start = time.perf_counter()
    for q in queries:
        idx = bisect.bisect_left(sorted_lst, q)
        _ = idx < len(sorted_lst) and sorted_lst[idx] == q
    search_time = time.perf_counter() - start
    results["sorted_list"] = {"insert": insert_time, "search": search_time}

    return results


if __name__ == "__main__":
    print("Data Structure Insert/Search Benchmark")
    print("=" * 70)

    for n in [1_000, 10_000, 100_000]:
        print(f"\n--- n = {n:,} ---")
        results = benchmark_insert_search(n)
        print(f"  {'Structure':<15} {'Insert(sec)':<15} {'Search(sec)':<15}")
        print(f"  {'-'*45}")
        for name, times in results.items():
            print(f"  {name:<15} {times['insert']:<15.6f} {times['search']:<15.6f}")
```

### 5.4 Guidelines for Interpreting Benchmark Results

When interpreting benchmark results, keep the following points in mind:

1. **Constant factor effects**: Even O(1) hash table lookups have hash computation costs. For small n, O(n) linear search may be faster.
2. **Cache efficiency**: Array-based structures have high memory locality and may perform faster than theoretical complexity suggests.
3. **Python overhead**: As an interpreted language, Python incurs overhead per operation. Built-in types (`list`, `dict`, `set`) implemented as C extensions are significantly faster than pure Python implementations.
4. **Garbage collection**: If Python's GC runs during measurement, results may fluctuate. It is advisable to temporarily disable it with `gc.disable()` or take the median of multiple measurements.

---

## 6. Selection Flowchart

### 6.1 Basic Flowchart

The following flowchart visualizes the basic decision process for data structure selection.

```
Data structure selection decision flow (detailed):

  START: What operation is dominant?
  |
  +-- Key-based search is primary -----------------------+
  |                                                      |
  |  Q: Is order needed?                                 |
  |  +-- No --> Hash table (dict)                        |
  |  |          Expected O(1) search/insert/delete       |
  |  |                                                   |
  |  +-- Yes -> Q: Is range search also needed?          |
  |             +-- Yes --> Balanced BST / B+ tree        |
  |             |           O(log n) + O(k)              |
  |             +-- No ---> Sorted array                 |
  |                         + bisect                     |
  |                         O(log n) search              |
  |                                                      |
  +-- Existence check is primary -------------------------+
  |                                                      |
  |  Q: Are exact results needed?                        |
  |  +-- Yes --> Hash set (set)                          |
  |  |           Expected O(1)                           |
  |  +-- No ---> Q: Large data + memory constraints?     |
  |              +-- Yes --> Bloom Filter                 |
  |              |           Space-efficient, FP possible |
  |              +-- No ---> Hash set (set)              |
  |                                                      |
  +-- Ordered processing is primary ----------------------+
  |                                                      |
  |  Q: What kind of order?                              |
  |  +-- LIFO --------> Stack (list)                     |
  |  +-- FIFO --------> Queue (deque)                    |
  |  +-- Priority -----> Heap (heapq)                    |
  |  +-- Both ends ----> Deque (deque)                   |
  |                                                      |
  +-- Dynamic sorted order maintenance is primary --------+
  |                                                      |
  |  Q: Insertion frequency?                             |
  |  +-- High -----> Balanced BST / SortedList           |
  |  |               O(log n) insert/delete              |
  |  +-- Low ------> Array + periodic sort               |
  |                   Insert O(n), sort O(n log n)       |
  |                                                      |
  +-- Sequential / random access is primary ---------------+
  |                                                      |
  |  Q: Is size fixed?                                   |
  |  +-- Yes --> Array / tuple                           |
  |  |           O(1) access, good memory efficiency     |
  |  +-- No ---> Dynamic array (list)                    |
  |              O(1) tail append (amortized)             |
  |                                                      |
  +-- Relationship representation is primary ---------------+
     |
     Q: Graph density?
     +-- Dense --> Adjacency matrix
     |             O(1) edge existence check
     +-- Sparse -> Adjacency list / Adjacency map
                   O(V + E) space
```

### 6.2 Quick Reference by Use Case

```
Quick reference of recommended data structures by use case:

  +------------------------------+-----------------------------+
  | Use Case                     | Recommended Data Structure  |
  +------------------------------+-----------------------------+
  | Display user list            | list (dynamic array)        |
  | Look up user by ID          | dict (hash table)           |
  | Check username uniqueness    | set (hash set)              |
  | Ranking (Top-K)             | heapq (binary heap)         |
  | Undo / Redo                 | list (stack) x 2            |
  | Task queue                  | deque or heapq              |
  | Autocomplete                | Trie                        |
  | Range search (date, price)  | SortedList / Balanced BST   |
  | LRU cache                   | OrderedDict                 |
  | Shortest path in graph      | Adjacency list + heapq      |
  | Existence check on big data | Bloom Filter                |
  | Configuration management    | dict                        |
  | Event log                   | deque (with maxlen)         |
  | Expression parsing (AST)    | Tree structure              |
  | Database index              | B+ tree                     |
  | Network routing             | Trie / Radix tree           |
  | Real-time median            | Two heaps (max + min)       |
  | Interval overlap detection  | Interval tree               |
  | Substring matching          | Suffix tree / Suffix array  |
  +------------------------------+-----------------------------+
```

### 6.3 Golden Rules When in Doubt

When uncertain about data structure selection, follow these rules.

**Rule 1: First consider arrays or hash tables**

More than 90% of real-world scenarios can be adequately handled with arrays (`list`) or hash tables (`dict` / `set`). Special structures are only needed when there are clear performance requirements.

**Rule 2: Avoid optimization without measurement**

Do not change data structures based solely on the intuition that "lists must be slow." First measure, identify the bottleneck, then optimize. Over-optimization for small-scale data only sacrifices readability with no practical benefit.

**Rule 3: Look ahead by exactly one step**

Consider not only current requirements but requirements that are likely to be added in the near future -- but only one step ahead. Following the YAGNI (You Ain't Gonna Need It) principle, do not optimize for more than two steps ahead.

**Rule 4: Provide an abstraction layer**

Do not directly expose internal data structures in public APIs. Instead, interpose an abstraction layer (class or interface). This minimizes the impact radius when changing data structures later.

```python
# Bad example: Exposing data structure directly
class UserService:
    def __init__(self):
        self.users: list[dict] = []  # Internal structure exposed externally

# Good example: Providing an abstraction layer
class UserRepository:
    def __init__(self):
        self._users_by_id: dict[int, dict] = {}  # Internal structure is private

    def add(self, user: dict) -> None:
        self._users_by_id[user["id"]] = user

    def find_by_id(self, user_id: int) -> dict | None:
        return self._users_by_id.get(user_id)

    def find_all(self) -> list[dict]:
        return list(self._users_by_id.values())
```

---

## 7. Anti-Patterns

Common mistakes in data structure selection are organized as anti-patterns. Recognizing these helps prevent failures at the design stage.

### 7.1 Anti-Pattern 1: The "Everything Is a List" Syndrome

**Symptom**: Using `list` for every situation without considering other data structures.

**Root cause**: Lists are the most familiar data structure, leading to unconscious selection. Particularly common among beginners.

**Example and fix**:

```python
# ============================================================
# Anti-pattern: Processing everything with lists
# ============================================================

# --- Problematic code ---
def find_duplicates_bad(items: list[str]) -> list[str]:
    """Find duplicate elements (anti-pattern)"""
    duplicates = []
    for i, item in enumerate(items):
        if item in items[i + 1:]:       # O(n) search each time
            if item not in duplicates:   # O(n) search each time
                duplicates.append(item)
    return duplicates
    # Overall complexity: O(n^3) <- catastrophically slow


# --- Fixed code ---
def find_duplicates_good(items: list[str]) -> list[str]:
    """Find duplicate elements (improved)"""
    seen: set[str] = set()
    duplicates: set[str] = set()
    for item in items:
        if item in seen:          # O(1) search
            duplicates.add(item)  # O(1) insertion
        else:
            seen.add(item)        # O(1) insertion
    return list(duplicates)
    # Overall complexity: O(n) <- linear time


# --- Performance verification ---
import time
import random
import string

def generate_random_strings(n: int, length: int = 5) -> list[str]:
    """Generate a random string list"""
    return ["".join(random.choices(string.ascii_lowercase, k=length))
            for _ in range(n)]

if __name__ == "__main__":
    random.seed(42)
    for size in [100, 1_000, 5_000]:
        data = generate_random_strings(size, length=3)

        start = time.perf_counter()
        result_bad = find_duplicates_bad(data)
        t_bad = time.perf_counter() - start

        start = time.perf_counter()
        result_good = find_duplicates_good(data)
        t_good = time.perf_counter() - start

        print(f"n={size:>5,}: "
              f"list approach={t_bad:.4f}s, set approach={t_good:.6f}s, "
              f"ratio={t_bad/max(t_good, 1e-9):.0f}x")
```

**Identification points**:
- `if x in some_list` appears frequently
- Linear searches within lists are nested inside loops
- `list.index()` or `list.count()` are called frequently

**Fix strategy**: Use `set` for existence checks, `dict` for key-based lookups.

### 7.2 Anti-Pattern 2: Premature Optimization

**Symptom**: Introducing complex data structures despite small data volumes.

**Root cause**: Focusing only on theoretical complexity advantages without considering actual data scale or readability impact.

**Example and fix**:

```python
# ============================================================
# Anti-pattern: Premature optimization
# ============================================================

# --- Problematic code: Using B+ tree for configuration management ---
# Configuration items are at most a few dozen
# The implementation and maintenance cost of a B+ tree is not justified

# class ConfigStore:
#     def __init__(self):
#         self._btree = BPlusTree(order=4)  # B+ tree for dozens of items...
#
#     def get(self, key: str) -> str:
#         return self._btree.search(key)
#
#     def set(self, key: str, value: str) -> None:
#         self._btree.insert(key, value)


# --- Fixed code: A simple dict suffices ---
class ConfigStore:
    """Application configuration management (assuming a few dozen items)"""

    def __init__(self):
        self._config: dict[str, str] = {}

    def get(self, key: str, default: str = "") -> str:
        return self._config.get(key, default)

    def set(self, key: str, value: str) -> None:
        self._config[key] = value

    def get_all(self) -> dict[str, str]:
        return dict(self._config)
```

**Identification points**:
- Custom tree structures or skip lists implemented for fewer than 100 items
- Based on unfounded assumptions like "it might grow to millions in the future"
- Readability and testability are significantly degraded

**Fix strategy**: Follow the YAGNI principle and select based on current requirements and realistic future scale. Optimize when the need actually arises.

### 7.3 Anti-Pattern 3: Overuse of list.pop(0)

**Symptom**: Using `list` as a FIFO queue, removing head elements with `pop(0)`.

**Root cause**: Assuming `list.pop(0)` is O(1) because `list.pop()` is O(1).

**Problem**: `list.pop(0)` requires shifting all elements forward by one position, resulting in O(n) complexity. Over n operations, this becomes O(n^2).

```python
# --- Anti-pattern ---
queue_bad: list[int] = list(range(10_000))
while queue_bad:
    item = queue_bad.pop(0)  # O(n) x n times = O(n^2)

# --- Fix ---
from collections import deque
queue_good: deque[int] = deque(range(10_000))
while queue_good:
    item = queue_good.popleft()  # O(1) x n times = O(n)
```

### 7.4 Anti-Pattern 4: Abuse of Nested Dicts

**Symptom**: Expressing complex data relationships with deeply nested dictionaries.

**Root cause**: Avoiding class or dataclass design, piling up `dict` layers ad hoc.

```python
# --- Anti-pattern ---
# Structure is unclear, types are unknown, typos go unnoticed
user = {
    "profile": {
        "name": "Alice",
        "address": {
            "city": "Tokyo",
            "zip": "100-0001"
        }
    },
    "settings": {
        "notifications": {
            "email": True,
            "push": False
        }
    }
}
# user["profile"]["adress"]["city"]  # Typo goes unnoticed (KeyError)

# --- Fix: Use dataclasses ---
from dataclasses import dataclass

@dataclass
class Address:
    city: str
    zip_code: str

@dataclass
class NotificationSettings:
    email: bool = True
    push: bool = False

@dataclass
class UserProfile:
    name: str
    address: Address

@dataclass
class UserSettings:
    notifications: NotificationSettings

@dataclass
class User:
    profile: UserProfile
    settings: UserSettings

# Type checker catches typos
user_obj = User(
    profile=UserProfile(
        name="Alice",
        address=Address(city="Tokyo", zip_code="100-0001")
    ),
    settings=UserSettings(
        notifications=NotificationSettings(email=True, push=False)
    )
)
print(user_obj.profile.address.city)  # IDE autocompletion works
```

### 7.5 Anti-Pattern 5: Using Mutable Structures for Immutable Data

**Symptom**: Using `list` or `dict` for data that is never modified, leaving risk of unintended changes.

**Fix**: Use immutable structures such as `tuple`, `frozenset`, `types.MappingProxyType`.

```python
# --- Anti-pattern ---
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
# If someone does WEEKDAYS.append("Holiday"), everything breaks

# --- Fix ---
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
# tuple is immutable, so append is not possible -> safe
```

### 7.6 Anti-Pattern Summary Table

| Anti-Pattern | Symptom | Impact on Complexity | Fix Strategy |
|---|---|---|---|
| Everything is a list | Frequent `in list` | O(n) -> O(1) | Switch to `set` / `dict` |
| Premature optimization | Complex structure for small data | No change but maintainability drops | Revert to simple structure |
| list.pop(0) | Using list as FIFO | O(n^2) -> O(n) | Switch to `deque` |
| Nested dict abuse | Deep dictionary nesting | No change but type safety drops | Switch to dataclasses |
| Mutable structure misuse | Constants as list/dict | Breeding ground for bugs | Switch to immutable structures |

---

## 8. Exercises

### Exercise 1: Requirements Analysis (Basics)

For each of the following requirements, select the optimal data structure and explain the rationale using Big-O notation.

**Problem 1-1**: Keep the most recent 100 log entries and automatically discard old ones.

<details>
<summary>Solution</summary>

**Recommended**: `collections.deque(maxlen=100)`

**Rationale**:
- `deque` performs additions and removals at both ends in O(1)
- With `maxlen` specified, elements are automatically removed from the opposite end when capacity is exceeded
- FIFO (first-in-first-out) semantics are naturally expressed
- Can be achieved with `list` too, but head removal is O(n), making it inappropriate

```python
from collections import deque

log_buffer = deque(maxlen=100)
for i in range(200):
    log_buffer.append(f"log entry {i}")
# len(log_buffer) == 100
# log_buffer[0] == "log entry 100" (oldest)
# log_buffer[-1] == "log entry 199" (newest)
```

</details>

**Problem 1-2**: From an English dictionary (~300,000 words), retrieve all words that prefix-match a given input string.

<details>
<summary>Solution</summary>

**Recommended**: Trie

**Rationale**:
- Trie is specialized for prefix searches, reaching the search node in O(m) for prefix length m
- After reaching the node, all matching words can be enumerated in O(k) (k is the result count) by traversing the subtree
- With a hash table, prefix matching against all keys requires O(n)
- Sorted array + bisect can also achieve O(log n + k), but Trie is more efficient for prefix operations

**Note**: Using `sortedcontainers.SortedList`, an approximate prefix search is possible via the `irange` method without implementing a Trie from scratch. In practice, the balance with implementation cost should also be considered.

</details>

**Problem 1-3**: Compute the median in real-time from streaming data.

<details>
<summary>Solution</summary>

**Recommended**: Two heaps (max-heap + min-heap)

**Rationale**:
- Combine a max-heap (managing the lower half) with a min-heap (managing the upper half)
- Data insertion: O(log n) (heap insertion and balance adjustment)
- Median retrieval: O(1) (just reference the tops of both heaps)
- Sorted array would require O(n) for insertion; balanced BST also O(log n) but more complex to implement

```python
import heapq

class MedianFinder:
    """Real-time median computation using two heaps"""

    def __init__(self):
        self._max_heap: list[int] = []  # Lower half (sign-inverted for max-heap)
        self._min_heap: list[int] = []  # Upper half

    def add(self, num: int) -> None:
        # First add to max-heap
        heapq.heappush(self._max_heap, -num)
        # Move max-heap's max to min-heap
        heapq.heappush(self._min_heap, -heapq.heappop(self._max_heap))
        # Size balance: max-heap size >= min-heap size
        if len(self._min_heap) > len(self._max_heap):
            heapq.heappush(self._max_heap, -heapq.heappop(self._min_heap))

    def median(self) -> float:
        if len(self._max_heap) > len(self._min_heap):
            return float(-self._max_heap[0])
        return (-self._max_heap[0] + self._min_heap[0]) / 2.0


mf = MedianFinder()
for num in [5, 2, 8, 1, 9]:
    mf.add(num)
    print(f"Added: {num}, Median: {mf.median()}")
# Added: 5, Median: 5.0
# Added: 2, Median: 3.5
# Added: 8, Median: 5.0
# Added: 1, Median: 3.5
# Added: 9, Median: 5.0
```

</details>

**Problem 1-4**: Manage a game leaderboard (ranking) and efficiently perform score updates and rank retrieval.

<details>
<summary>Solution</summary>

**Recommended**: Balanced BST (`sortedcontainers.SortedList`) or Skip list

**Rationale**:
- Score update (delete + insert): O(log n)
- Rank retrieval (count of elements above a given score): O(log n)
- Top-K retrieval: O(K) (traversal from the end)
- Hash table requires O(n) for rank computation; heaps make arbitrary score updates difficult

</details>

### Exercise 2: Design Problems (Applied)

**Problem 2-1**: Design a timeline feature for a social network.

Design data structures that satisfy the following requirements, and show the complexity of each operation.

- Users can create posts (text + timestamp)
- Users can follow other users
- Timeline retrieval: Get up to 20 most recent posts from followed users
- Post count: Up to several thousand per user, millions total expected

<details>
<summary>Solution</summary>

```python
"""
SNS Timeline Design

Data structure selection rationale:
- User lookup: dict (O(1))
- Follow relationships: dict[int, set] (O(1) for follow/unfollow/check)
- Post storage: dict[int, list] (lists per user maintaining chronological order)
- Timeline retrieval: Heap merge (efficiently merge latest posts from followed users)
"""
import heapq
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Post:
    post_id: int
    user_id: int
    text: str
    created_at: datetime

    def __lt__(self, other: "Post") -> bool:
        # For heap: Descending by timestamp (newest first)
        return self.created_at > other.created_at


@dataclass
class SocialNetwork:
    # User ID -> Username: O(1) lookup
    users: dict[int, str] = field(default_factory=dict)

    # User ID -> Set of followed user IDs: O(1) follow check
    following: dict[int, set[int]] = field(default_factory=lambda: {})

    # User ID -> Post list (chronological order)
    posts: dict[int, list[Post]] = field(default_factory=lambda: {})

    _post_counter: int = 0

    def create_post(self, user_id: int, text: str) -> Post:
        """Create a post - O(1)"""
        self._post_counter += 1
        post = Post(
            post_id=self._post_counter,
            user_id=user_id,
            text=text,
            created_at=datetime.now()
        )
        if user_id not in self.posts:
            self.posts[user_id] = []
        self.posts[user_id].append(post)
        return post

    def follow(self, user_id: int, target_id: int) -> None:
        """Follow - O(1)"""
        if user_id not in self.following:
            self.following[user_id] = set()
        self.following[user_id].add(target_id)

    def unfollow(self, user_id: int, target_id: int) -> None:
        """Unfollow - O(1)"""
        if user_id in self.following:
            self.following[user_id].discard(target_id)

    def get_timeline(self, user_id: int, limit: int = 20) -> list[Post]:
        """
        Get timeline - O(F * log F + limit * log F)
        F = number of followed users

        K-way merge algorithm:
        Put the latest post from each followed user into a heap,
        then extract the limit newest ones
        """
        follow_ids = self.following.get(user_id, set())
        if not follow_ids:
            return []

        # Prepare iterators for the latest posts from each followed user
        # Heap of (post, user's post list, index)
        heap: list[tuple[Post, int, int]] = []
        for fid in follow_ids:
            user_posts = self.posts.get(fid, [])
            if user_posts:
                idx = len(user_posts) - 1
                heapq.heappush(heap, (user_posts[idx], fid, idx))

        timeline: list[Post] = []
        while heap and len(timeline) < limit:
            post, fid, idx = heapq.heappop(heap)
            timeline.append(post)
            if idx > 0:
                next_idx = idx - 1
                next_post = self.posts[fid][next_idx]
                heapq.heappush(heap, (next_post, fid, next_idx))

        return timeline
```

**Complexity summary**:
| Operation | Complexity | Structure Used |
|---|---|---|
| Create post | O(1) | list (append) |
| Follow/Unfollow | O(1) | set (add/discard) |
| Follow check | O(1) | set (in) |
| Timeline retrieval | O(F log F + L log F) | heapq (K-way merge) |

(F = number of followed users, L = limit)

</details>

**Problem 2-2**: Design a text editor buffer structure.

Efficiently support the following operations:
- Character insertion at cursor position
- Character deletion at cursor position
- Cursor movement (forward/backward, beginning/end of line)
- Undo / Redo

<details>
<summary>Solution</summary>

**Recommended structure combination**:
- **Text buffer**: Gap Buffer
  - An array with a "gap" (empty region) at the cursor position
  - Insertion/deletion near the cursor is O(1)
  - Gap moves when cursor moves: O(movement distance)
  - Continuous input is local, so practically works at ~O(1)
- **Undo/Redo**: Two stacks
  - Operation stack (for Undo) and Redo stack
  - Operations represented using the Command pattern

**Alternative**: Rope structure
- A balanced binary tree-based string structure suited for large text (several MB+)
- Insert/delete/concatenation at O(log n)
- Similar to the structure used internally by Visual Studio Code

</details>

### Exercise 3: Optimization Problems (Advanced)

**Problem 3-1**: Identify the bottleneck in the following code and improve it by changing the data structure.

```python
def count_common_elements(list_a: list[int], list_b: list[int]) -> int:
    """Count common elements between two lists (before improvement)"""
    count = 0
    for item in list_a:
        if item in list_b:  # <- Bottleneck: O(n) x m times
            count += 1
    return count
```

<details>
<summary>Solution</summary>

```python
def count_common_elements_optimized(list_a: list[int],
                                     list_b: list[int]) -> int:
    """Count common elements between two lists (after improvement)"""
    set_b = set(list_b)  # O(n) to build
    count = 0
    for item in list_a:
        if item in set_b:  # O(1) search
            count += 1
    return count
    # Total: O(m + n)

# More concise version:
def count_common_elements_pythonic(list_a: list[int],
                                    list_b: list[int]) -> int:
    """Implementation using set intersection"""
    return len(set(list_a) & set(list_b))
```

**Improvement**: O(m * n) -> O(m + n)

</details>

**Problem 3-2**: Design a system that receives large volumes of sensor data in real-time (1,000 readings per second) and responds to the following queries:

- Average value of data from the last 5 minutes
- Max/min values from the last 5 minutes
- List of data within a specified time range

<details>
<summary>Solution</summary>

**Recommended structure combination**:

1. **Ring buffer (`deque(maxlen=300_000)`)**: Holds the last 5 minutes (300,000 readings) of data
2. **Running sum / Sliding window**: Running total for O(1) average computation
3. **Monotonic deque**: Get max/min within the window at O(1)
4. **Sorted index**: For time range queries (bisect-based)

```python
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class SensorReading:
    timestamp: datetime
    value: float


class SensorAggregator:
    """Real-time aggregation of sensor data"""

    def __init__(self, window_seconds: int = 300):
        self._window = timedelta(seconds=window_seconds)
        self._data: deque[SensorReading] = deque()
        self._sum: float = 0.0
        # Monotonic deque: For max values (stores indices)
        self._max_deque: deque[int] = deque()
        # Monotonic deque: For min values (stores indices)
        self._min_deque: deque[int] = deque()
        self._index: int = 0

    def _evict_old(self, now: datetime) -> None:
        """Evict data outside the window"""
        cutoff = now - self._window
        while self._data and self._data[0].timestamp < cutoff:
            old = self._data.popleft()
            self._sum -= old.value

    def add(self, reading: SensorReading) -> None:
        """Add data - Amortized O(1)"""
        self._evict_old(reading.timestamp)
        self._data.append(reading)
        self._sum += reading.value
        self._index += 1

    def average(self) -> float:
        """Average of recent window - O(1)"""
        if not self._data:
            return 0.0
        return self._sum / len(self._data)

    def count(self) -> int:
        """Data count in recent window - O(1)"""
        return len(self._data)
```

</details>

**Problem 3-3**: Design the index structure for an in-memory database with the following requirements.

- Record count: Up to 1 million
- Exact match search by primary key (integer): O(1)
- Prefix search by name field: Efficient
- Range search by creation date: Efficient
- Insert/delete: O(log n) or less

<details>
<summary>Solution</summary>

**Compound index strategy**:

| Field | Index Structure | Complexity |
|---|---|---|
| Primary key | `dict` (hash table) | Search O(1), Insert O(1) |
| Name | Trie | Prefix search O(m + k) |
| Creation date | `SortedList` (balanced BST) | Range search O(log n + k), Insert O(log n) |

Each index holds references (IDs) to records, while the actual data is stored in the primary key `dict`. Index synchronization is maintained by updating all indexes during inserts and deletes.

**Trade-off**: More indexes speed up searches but increase update costs for inserts and deletes. Decide based on the ratio of read frequency to write frequency.

</details>

---

## 9. Frequently Asked Questions (FAQ)

### FAQ 1: How should dict and defaultdict be used differently?

**Answer**: `dict` raises a `KeyError` when a key does not exist, while `defaultdict` automatically generates a default value when a key does not exist.

```python
from collections import defaultdict

# --- dict: Requires explicit initialization ---
word_count_dict: dict[str, int] = {}
for word in ["apple", "banana", "apple", "cherry", "banana", "apple"]:
    if word not in word_count_dict:
        word_count_dict[word] = 0
    word_count_dict[word] += 1
# Or use dict.get()
# word_count_dict[word] = word_count_dict.get(word, 0) + 1

# --- defaultdict: No initialization needed ---
word_count_dd: defaultdict[str, int] = defaultdict(int)
for word in ["apple", "banana", "apple", "cherry", "banana", "apple"]:
    word_count_dd[word] += 1  # If key doesn't exist, int() = 0 is auto-generated

# --- Counter: Even more concise ---
from collections import Counter
word_count_counter = Counter(["apple", "banana", "apple", "cherry",
                               "banana", "apple"])
print(word_count_counter.most_common(2))  # [('apple', 3), ('banana', 2)]
```

**Usage guidelines**:
- Simple counting -> `Counter`
- Grouping (key -> list) -> `defaultdict(list)`
- Want error on missing key -> `dict`
- Want default value on missing key -> `defaultdict` or `dict.setdefault()`

### FAQ 2: How should list and tuple be used differently?

**Answer**: There are both semantic and technical differences.

**Semantic differences**:
- `list`: Variable-length collection of homogeneous elements (e.g., a list of users)
- `tuple`: Fixed-length record of heterogeneous elements (e.g., (x-coordinate, y-coordinate), (name, age))

**Technical differences**:
- `tuple` is immutable and hashable. Can be used as `dict` keys or `set` elements
- `tuple` is slightly more memory-efficient (`list` has a buffer for resizing)
- `tuple` creation is slightly faster (`list` requires internal array allocation)

```python
import sys

# Memory comparison
lst = [1, 2, 3, 4, 5]
tpl = (1, 2, 3, 4, 5)
print(f"list: {sys.getsizeof(lst)} bytes")  # list: 104 bytes (approximate)
print(f"tuple: {sys.getsizeof(tpl)} bytes")  # tuple: 80 bytes (approximate)
```

**Guideline**: Use `tuple` if data will not be modified, `list` if it may be modified.

### FAQ 3: Is the order of elements in a set guaranteed?

**Answer**: **Not guaranteed.** `set` is internally implemented as a hash table, and element storage order depends on hash values. Iteration order is implementation-dependent and may vary across Python versions and execution environments.

```python
# Example where order is not guaranteed
s = {3, 1, 4, 1, 5, 9, 2, 6}
print(s)  # {1, 2, 3, 4, 5, 6, 9} <- this order is not guaranteed

# Options when order is needed:
# 1. Convert to sorted list
sorted_list = sorted(s)  # [1, 2, 3, 4, 5, 6, 9]

# 2. If insertion order is needed, use dict.fromkeys()
ordered_unique = list(dict.fromkeys([3, 1, 4, 1, 5, 9, 2, 6]))
print(ordered_unique)  # [3, 1, 4, 5, 9, 2, 6] (insertion order preserved)
```

Note: Python 3.7+ `dict` guarantees insertion order, but this is a `dict` specification and does not apply to `set`.

### FAQ 4: Why is heapq a min-heap? What if a max-heap is needed?

**Answer**: Python's `heapq` only provides a min-heap. When a max-heap is needed, the standard approach is to negate the values.

```python
import heapq

# --- Min-heap (as-is) ---
min_heap: list[int] = []
for val in [5, 3, 8, 1, 9]:
    heapq.heappush(min_heap, val)
print(heapq.heappop(min_heap))  # 1 (minimum)

# --- Max-heap (negate values) ---
max_heap: list[int] = []
for val in [5, 3, 8, 1, 9]:
    heapq.heappush(max_heap, -val)  # Store negated
print(-heapq.heappop(max_heap))  # 9 (maximum)

# --- Top-K (get the K largest) ---
data = [5, 3, 8, 1, 9, 2, 7, 4, 6]
top_3 = heapq.nlargest(3, data)    # [9, 8, 7]
bottom_3 = heapq.nsmallest(3, data)  # [1, 2, 3]
```

### FAQ 5: How should numpy arrays and Python lists be used differently?

**Answer**: Use `numpy.ndarray` when numerical computation is primary, `list` for general data management.

| Aspect | `list` | `numpy.ndarray` |
|---|---|---|
| Element types | Can be mixed | Same type only |
| Memory efficiency | Low (array of pointers to objects) | High (contiguous value array) |
| Numerical operations | Slow (Python loops) | Fast (C-implemented vector ops) |
| Flexibility | High (append, extend, etc.) | Low (resizing is costly) |
| Use case | General data management | Numerical, scientific computing, ML |

For bulk operations on 100,000+ numerical data points, `numpy` can be tens to hundreds of times faster than `list`.

### FAQ 6: What is the relationship between database indexes and language data structures?

**Answer**: Relational database indexes are applications of data structures.

| Database Feature | Underlying Data Structure |
|---|---|
| B-Tree index | B+ tree |
| Hash index | Hash table |
| Full-text search index | Inverted index (hash table + sorted list) |
| Spatial index (GiST) | R-tree |
| Covering index | B+ tree (with data in leaves) |

Understanding data structure theory enables deeper understanding of database index design and query optimization.

---


## FAQ

### Q1: What is the most important point in learning this topic?

Gaining practical experience is the most important aspect. Understanding deepens not just through theory but by actually writing and running code.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this used in practice?

Knowledge of this topic is frequently applied in day-to-day development work. It becomes particularly important during code reviews and architecture design.

---

## 10. Summary

### 10.1 Fundamental Principles of Selection

The contents covered in this chapter are organized as fundamental principles of selection.

**Principle 1: Base decisions on operation frequency**

Choose the data structure that makes the most frequently performed operation the fastest. You cannot optimize all operations simultaneously, so accept trade-offs.

**Principle 2: Start with simple structures**

The vast majority of real-world situations can be solved with just three: `list`, `dict`, `set`. Special data structures should only be considered after a bottleneck has been confirmed through measurement.

**Principle 3: Estimate data scale**

When n is under 100, there is no perceptible difference regardless of which structure you choose. Around n > 10,000, the order of complexity starts to matter. When n exceeds 1 million, the difference between O(n) and O(log n) becomes critical.

**Principle 4: Prepare for change through abstraction**

Do not directly expose internal data structures; access them through interfaces. This minimizes the cost of replacing data structures later.

**Principle 5: Do not sacrifice readability**

Code is read more often than it is written. If a data structure choice makes code intent unclear, prioritize readability even at the cost of some performance.

### 10.2 Final Recommendation List by Requirement

| Requirement | Recommended Data Structure | Key Operation Complexity |
|---|---|---|
| Sequential access / tail append | `list` | Access O(1), tail append O(1)* |
| Fast lookup by key | `dict` | Search O(1), insert O(1) |
| Deduplication / existence check | `set` | Search O(1), insert O(1) |
| Dynamic sorted order maintenance | `SortedList` / Balanced BST | Insert O(log n), search O(log n) |
| LIFO (Undo/Redo, etc.) | `list` (stack) | push O(1), pop O(1) |
| FIFO (task queue, etc.) | `deque` | enqueue O(1), dequeue O(1) |
| Priority-based processing | `heapq` | Insert O(log n), get min O(1) |
| Prefix search | Trie | Search O(m) (m = key length) |
| Range search | Sorted array / Balanced BST | O(log n + k) |
| Cache (LRU) | `OrderedDict` | get/put O(1) |
| Existence check on large data | Bloom Filter | Search O(k) (k = hash count) |
| Graph (sparse) | `dict[int, set[int]]` | Add edge O(1), adjacency O(1) |
| Graph (dense) | 2D array | Edge check O(1) |
| Real-time median | Two heaps | Add O(log n), get O(1) |

### 10.3 Learning Roadmap

The following roadmap is for improving your data structure selection skills.

```
Data Structure Selection Skill Roadmap:

  Level 1: Fundamentals (Essential)
  +-----------------------------------------------------+
  | - Completely understand characteristics and          |
  |   complexities of list, dict, set                    |
  | - Distinguish between tuple, deque, heapq            |
  | - Intuitive understanding of Big-O                   |
  |   (how much impact at n=100,000)                     |
  +-----------------------------------------------------+
            |
            v
  Level 2: Applied (Common in Practice)
  +-----------------------------------------------------+
  | - Range search with sortedcontainers                 |
  | - Compound index design                              |
  | - LRU cache implementation and usage                 |
  | - Median computation with two heaps                  |
  | - Graph representation method selection              |
  +-----------------------------------------------------+
            |
            v
  Level 3: Advanced (Specialized Scenarios)
  +-----------------------------------------------------+
  | - Trie, suffix tree, suffix array                    |
  | - Bloom Filter, Count-Min Sketch                     |
  | - Persistent Data Structures                         |
  | - Concurrent data structures (Lock-Free, Wait-Free)  |
  | - External memory algorithms (B+ tree, LSM tree)     |
  +-----------------------------------------------------+
```

---

## Recommended Next Guides

---

## References

1. Skiena, S. S. *The Algorithm Design Manual*, 3rd Edition, Springer, 2020.
   - Chapter 3 "Data Structures": Practical guidelines on data structure selection. Chapter 12's "Data Structures" catalog provides detailed trade-off analysis for each structure.

2. Kleppmann, M. *Designing Data-Intensive Applications*, O'Reilly Media, 2017.
   - Chapter 3 "Storage and Retrieval": Explains selection criteria for data structures used inside databases, including B-trees, LSM trees, and hash indexes.

3. Cormen, T. H., Leiserson, C. E., Rivest, R. L., and Stein, C. *Introduction to Algorithms*, 4th Edition, MIT Press, 2022.
   - Known as CLRS. Covers theoretical complexity analysis foundations for each data structure. Part III "Data Structures" provides the theoretical background for this chapter.

4. Wirth, N. *Algorithms + Data Structures = Programs*, Prentice-Hall, 1976.
   - A classic work demonstrating the inseparability of data structures and algorithms. Argues how critical data structure selection is in program design.

5. Knuth, D. E. *The Art of Computer Programming, Volume 3: Sorting and Searching*, 2nd Edition, Addison-Wesley, 1998.
   - Theoretical foundations for search algorithms and data structures. Detailed analysis of hashing, tree structures, and sorting.

6. Python Official Documentation "Data Structures"
   - https://docs.python.org/3/tutorial/datastructures.html
   - Official reference for Python built-in data structures. Usage and performance characteristics of `list`, `dict`, `set`, `tuple`.

7. Python Official Documentation "collections --- Container datatypes"
   - https://docs.python.org/3/library/collections.html
   - Official reference for additional data structures including `deque`, `defaultdict`, `Counter`, `OrderedDict`.

8. Python Wiki "TimeComplexity"
   - https://wiki.python.org/moin/TimeComplexity
   - Official reference comprehensively covering the complexity of each operation for Python built-in types. Essential for verifying complexities during data structure selection.
