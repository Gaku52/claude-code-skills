# Advanced Data Structures

> A Bloom Filter guarantees 100% that an element "does not exist," and a Skip List gives linked lists O(log n) search.

## Learning Objectives

- [ ] Understand the mechanism and use cases of Bloom Filters
- [ ] Explain the structure of Skip Lists
- [ ] Implement range queries with Segment Trees and Fenwick Trees
- [ ] Understand optimization techniques for Union-Find
- [ ] Explain the design of LRU/LFU caches
- [ ] Know specialized data structures such as Rope and Merkle Tree
- [ ] Know practical use cases for each data structure

## Prerequisites


---

## 1. Bloom Filter

### 1.1 Mechanism

```
Bloom Filter: Rapidly determines whether an element is "not contained" in a set

  Structure: An array of m bits + k hash functions

  Insertion: Set the bits at positions hash1(x), hash2(x), ..., hashk(x) to 1
  Lookup:   All hash positions are 1 -> "probably exists"
            At least one is 0 -> "definitely does not exist"

  False positive rate: (1 - e^(-kn/m))^k
  n=10 million, m=100 million bits (12.5MB), k=7 -> false positive rate ≈ 0.008 (0.8%)

  Optimal number of hash functions: k = (m/n) × ln(2)

  Use cases:
  - Chrome: Checking for malicious URLs
  - Cassandra/HBase: Skipping SST file lookups
  - Medium: Deduplication in article recommendations
  - Bitcoin: Transaction verification in SPV nodes
  - Redis: Pre-filtering for large-scale caches
```

### 1.2 Basic Implementation

```python
import hashlib
import math

class BloomFilter:
    """Bloom Filter: A probabilistic data structure"""

    def __init__(self, expected_items, fp_rate=0.01):
        """
        expected_items: Expected number of elements
        fp_rate: Acceptable false positive rate
        """
        # Calculate optimal bit array size
        self.size = self._optimal_size(expected_items, fp_rate)
        # Calculate optimal number of hash functions
        self.num_hashes = self._optimal_hashes(self.size, expected_items)
        # Bit array (represented as a list of integers)
        self.bit_array = [0] * self.size
        self.count = 0

    def _optimal_size(self, n, p):
        """Optimal bit array size: m = -(n × ln(p)) / (ln(2))²"""
        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return int(m)

    def _optimal_hashes(self, m, n):
        """Optimal number of hash functions: k = (m/n) × ln(2)"""
        k = (m / n) * math.log(2)
        return max(1, int(k))

    def _hashes(self, item):
        """Generate k independent hash values"""
        indices = []
        for i in range(self.num_hashes):
            h = hashlib.sha256(f"{item}:{i}".encode()).hexdigest()
            indices.append(int(h, 16) % self.size)
        return indices

    def add(self, item):
        """Add an element"""
        for idx in self._hashes(item):
            self.bit_array[idx] = 1
        self.count += 1

    def contains(self, item):
        """Check for element existence"""
        return all(self.bit_array[idx] == 1 for idx in self._hashes(item))

    def estimated_fp_rate(self):
        """Estimate the current false positive rate"""
        ones = sum(self.bit_array)
        if ones == 0:
            return 0.0
        return (ones / self.size) ** self.num_hashes

    def __contains__(self, item):
        return self.contains(item)

    def __len__(self):
        return self.count


# Usage example
bf = BloomFilter(expected_items=100000, fp_rate=0.01)
print(f"Bit array size: {bf.size:,} bits ({bf.size // 8:,} bytes)")
print(f"Number of hash functions: {bf.num_hashes}")

# Add elements
for i in range(100000):
    bf.add(f"user:{i}")

# Test
print(f"Existence check (exists): {'user:42' in bf}")    # True
print(f"Existence check (does not exist): {'user:999999' in bf}")  # Probably False

# Measure actual false positive rate
false_positives = sum(
    1 for i in range(100000, 200000)
    if f"user:{i}" in bf
)
actual_fp_rate = false_positives / 100000
print(f"Measured false positive rate: {actual_fp_rate:.4f}")  # Close to 0.01
```

### 1.3 Optimized Bit Array Implementation

```python
import array

class OptimizedBloomFilter:
    """Memory-efficient Bloom Filter (bitwise operations)"""

    def __init__(self, size, num_hashes):
        self.size = size
        self.num_hashes = num_hashes
        # Implement bit array as a byte array
        self.bit_array = bytearray((size + 7) // 8)

    def _set_bit(self, pos):
        """Set a bit to 1"""
        self.bit_array[pos >> 3] |= (1 << (pos & 7))

    def _get_bit(self, pos):
        """Get a bit"""
        return (self.bit_array[pos >> 3] >> (pos & 7)) & 1

    def _hashes(self, item):
        """Generate k hash values efficiently using Double Hashing"""
        # Derive k hash values from h1 and h2
        data = str(item).encode()
        h1 = int(hashlib.md5(data).hexdigest(), 16)
        h2 = int(hashlib.sha1(data).hexdigest(), 16)
        return [(h1 + i * h2) % self.size for i in range(self.num_hashes)]

    def add(self, item):
        for idx in self._hashes(item):
            self._set_bit(idx)

    def contains(self, item):
        return all(self._get_bit(idx) for idx in self._hashes(item))

    def memory_usage(self):
        """Return memory usage in bytes"""
        return len(self.bit_array)
```

### 1.4 Counting Bloom Filter

```python
class CountingBloomFilter:
    """Bloom Filter that supports deletion"""

    def __init__(self, size, num_hashes, counter_bits=4):
        self.size = size
        self.num_hashes = num_hashes
        self.max_count = (1 << counter_bits) - 1  # 4 bits -> max 15
        self.counters = [0] * size

    def _hashes(self, item):
        indices = []
        for i in range(self.num_hashes):
            h = hashlib.sha256(f"{item}:{i}".encode()).hexdigest()
            indices.append(int(h, 16) % self.size)
        return indices

    def add(self, item):
        """Add an element"""
        for idx in self._hashes(item):
            if self.counters[idx] < self.max_count:
                self.counters[idx] += 1

    def remove(self, item):
        """Remove an element (impossible with a standard Bloom Filter)"""
        indices = self._hashes(item)
        # First check existence
        if not all(self.counters[idx] > 0 for idx in indices):
            return False  # Cannot delete a non-existent element

        for idx in indices:
            self.counters[idx] -= 1
        return True

    def contains(self, item):
        return all(self.counters[idx] > 0 for idx in self._hashes(item))

    def memory_usage(self):
        """Requires 4x the memory of a standard Bloom Filter"""
        return self.size * 4  # 4 bits per counter


# Use cases for Counting Bloom Filter:
# - Cache invalidation in CDNs
# - Deduplication of streaming data (requires add and delete)
# - Routing tables in P2P networks
```

### 1.5 Cuckoo Filter

```python
import hashlib
import random

class CuckooFilter:
    """Cuckoo Filter: An improved version of Bloom Filter
    - Supports deletion
    - Better space efficiency than Bloom Filter (at low false positive rates)
    - Fast lookups (only checks 2 locations)"""

    MAX_KICKS = 500

    def __init__(self, capacity, fingerprint_size=8):
        self.capacity = capacity
        self.fp_size = fingerprint_size
        self.buckets = [[] for _ in range(capacity)]
        self.bucket_size = 4  # Number of entries per bucket
        self.count = 0

    def _fingerprint(self, item):
        """Generate a fingerprint (partial hash)"""
        h = hashlib.sha256(str(item).encode()).hexdigest()
        fp = int(h[:self.fp_size], 16)
        return fp if fp != 0 else 1  # Avoid 0 since it means empty

    def _hash1(self, item):
        h = hashlib.md5(str(item).encode()).hexdigest()
        return int(h, 16) % self.capacity

    def _hash2(self, idx1, fingerprint):
        """Alternate index = idx1 XOR hash(fingerprint)"""
        h = hashlib.md5(str(fingerprint).encode()).hexdigest()
        return (idx1 ^ (int(h, 16) % self.capacity)) % self.capacity

    def insert(self, item):
        """Insert an element"""
        fp = self._fingerprint(item)
        idx1 = self._hash1(item)
        idx2 = self._hash2(idx1, fp)

        # Insert into bucket 1 if there is space
        if len(self.buckets[idx1]) < self.bucket_size:
            self.buckets[idx1].append(fp)
            self.count += 1
            return True

        # Insert into bucket 2 if there is space
        if len(self.buckets[idx2]) < self.bucket_size:
            self.buckets[idx2].append(fp)
            self.count += 1
            return True

        # Eviction (Cuckoo-style)
        idx = random.choice([idx1, idx2])
        for _ in range(self.MAX_KICKS):
            # Randomly select and evict an entry
            victim_idx = random.randrange(len(self.buckets[idx]))
            fp, self.buckets[idx][victim_idx] = self.buckets[idx][victim_idx], fp
            idx = self._hash2(idx, fp)

            if len(self.buckets[idx]) < self.bucket_size:
                self.buckets[idx].append(fp)
                self.count += 1
                return True

        return False  # Table is full

    def lookup(self, item):
        """Look up an element"""
        fp = self._fingerprint(item)
        idx1 = self._hash1(item)
        idx2 = self._hash2(idx1, fp)

        return fp in self.buckets[idx1] or fp in self.buckets[idx2]

    def delete(self, item):
        """Delete an element"""
        fp = self._fingerprint(item)
        idx1 = self._hash1(item)
        idx2 = self._hash2(idx1, fp)

        if fp in self.buckets[idx1]:
            self.buckets[idx1].remove(fp)
            self.count -= 1
            return True

        if fp in self.buckets[idx2]:
            self.buckets[idx2].remove(fp)
            self.count -= 1
            return True

        return False

    def __contains__(self, item):
        return self.lookup(item)

    def __len__(self):
        return self.count


# Bloom Filter vs Cuckoo Filter comparison:
# +--------------------+--------------+--------------+
# | Property           | Bloom Filter | Cuckoo Filter|
# +--------------------+--------------+--------------+
# | Deletion support   | No           | Yes          |
# | Lookup speed       | k hashes     | Only 2       |
# | Insertion speed    | O(k)         | O(1) expected|
# | FP rate < 3%       | Less efficient| Good space  |
# | FP rate > 3%       | Good space   | Less efficient|
# | Multiple insertion | Possible     | Limited      |
# | of same element    |              |              |
# +--------------------+--------------+--------------+
```

---

## 2. Skip List

### 2.1 Basic Structure

```
Skip List: A probabilistically balanced ordered list

  Level 3: head --------------------------------- 50 ---------- tail
  Level 2: head ---------- 20 ------------------- 50 ---------- tail
  Level 1: head -- 10 -- 20 -- 30 -- 40 -- 50 -- 60 -- tail

  Search: Start from the top level, drop down when unable to proceed
  -> Average O(log n)

  Advantages:
  - Performance equivalent to red-black trees (O(log n) search/insert/delete)
  - Much simpler to implement (far simpler than red-black trees)
  - Suitable for concurrent processing (lock-free implementation possible)

  Use case: Redis Sorted Set
```

### 2.2 Skip List Implementation

```python
import random

class SkipNode:
    def __init__(self, key=None, value=None, level=0):
        self.key = key
        self.value = value
        # forward[i] is the next node at level i
        self.forward = [None] * (level + 1)

class SkipList:
    """Skip List: A probabilistically balanced ordered list"""

    MAX_LEVEL = 16  # Maximum number of levels
    P = 0.5         # Probability of level promotion

    def __init__(self):
        self.header = SkipNode(level=self.MAX_LEVEL)
        self.level = 0  # Current maximum level
        self.size = 0

    def _random_level(self):
        """Generate a random level (geometric distribution)"""
        level = 0
        while random.random() < self.P and level < self.MAX_LEVEL:
            level += 1
        return level

    def search(self, key):
        """Search: Average O(log n), Worst O(n)"""
        current = self.header

        # From the top level downward
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]

        current = current.forward[0]

        if current and current.key == key:
            return current.value
        return None

    def insert(self, key, value):
        """Insert: Average O(log n)"""
        update = [None] * (self.MAX_LEVEL + 1)
        current = self.header

        # Record insertion positions at each level
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current

        current = current.forward[0]

        # Update existing key
        if current and current.key == key:
            current.value = value
            return

        # Determine the new level
        new_level = self._random_level()

        if new_level > self.level:
            for i in range(self.level + 1, new_level + 1):
                update[i] = self.header
            self.level = new_level

        # Create a new node
        new_node = SkipNode(key, value, new_level)

        # Add links at each level
        for i in range(new_level + 1):
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node

        self.size += 1

    def delete(self, key):
        """Delete: Average O(log n)"""
        update = [None] * (self.MAX_LEVEL + 1)
        current = self.header

        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current

        current = current.forward[0]

        if current and current.key == key:
            for i in range(self.level + 1):
                if update[i].forward[i] != current:
                    break
                update[i].forward[i] = current.forward[i]

            # Remove empty levels
            while self.level > 0 and self.header.forward[self.level] is None:
                self.level -= 1

            self.size -= 1
            return True

        return False

    def range_query(self, start, end):
        """Range query: O(log n + k), where k is the number of results"""
        result = []
        current = self.header

        # Find the first node >= start
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < start:
                current = current.forward[i]

        current = current.forward[0]

        # Traverse up to end
        while current and current.key <= end:
            result.append((current.key, current.value))
            current = current.forward[0]

        return result

    def __len__(self):
        return self.size

    def __contains__(self, key):
        return self.search(key) is not None

    def display(self):
        """Visualize the Skip List structure"""
        for level in range(self.level, -1, -1):
            nodes = []
            current = self.header.forward[level]
            while current:
                nodes.append(str(current.key))
                current = current.forward[level]
            print(f"Level {level}: {' -> '.join(nodes)}")


# Usage example
sl = SkipList()
for val in [3, 6, 7, 9, 12, 19, 17, 26, 21, 25]:
    sl.insert(val, val * 10)

sl.display()
# Level 2: 7 -> 19
# Level 1: 3 -> 7 -> 12 -> 19 -> 25
# Level 0: 3 -> 6 -> 7 -> 9 -> 12 -> 17 -> 19 -> 21 -> 25 -> 26

print(sl.search(12))         # 120
print(sl.range_query(10, 20))  # [(12, 120), (17, 170), (19, 190)]
sl.delete(12)
print(sl.search(12))         # None


# Skip List time complexity:
# +------------+----------+----------+
# | Operation  | Average  | Worst    |
# +------------+----------+----------+
# | Search     | O(log n) | O(n)     |
# | Insert     | O(log n) | O(n)     |
# | Delete     | O(log n) | O(n)     |
# | Range query| O(log n + k)| O(n)  |
# +------------+----------+----------+
# Space: O(n) (expected)
# Average level of a node: 1/(1-p) = 2 (when p=0.5)
```

### 2.3 Redis Sorted Set

```
Redis Sorted Set internal implementation:

  A hybrid of Skip List + Hash Table
  - Skip List: Sorted by score, range queries
  - Hash Table: O(1) lookup from member name to score

  Usage examples (Redis commands):
  ZADD leaderboard 100 "Alice"
  ZADD leaderboard 85 "Bob"
  ZADD leaderboard 92 "Charlie"

  ZRANK leaderboard "Alice"          # 2 (0-indexed, sorted by score)
  ZRANGE leaderboard 0 -1 WITHSCORES # Get all members sorted by score
  ZRANGEBYSCORE leaderboard 85 95    # Members with scores 85-95
  ZREVRANK leaderboard "Alice"       # 0 (rank in descending order)

  Why Skip List was chosen over red-black trees (explained by Redis creator antirez):
  1. Simpler implementation
  2. Better range query performance
  3. Easier to extend for concurrent processing
  4. Equivalent performance to red-black trees (constant factor difference)
```

---

## 3. Segment Tree

### 3.1 Basic Concept

```
Segment Tree: Processes range queries in O(log n)

  Array: [2, 1, 5, 3, 4, 2]

  Segment Tree (range minimum):
              1           <- Overall minimum
            /   \
          1       2       <- Minimum of first/second half
         / \     / \
        1   5   3   2     <- Further halves
       / \ / \ / \ / \
      2  1 5  3 4  2      <- Original array

  Use cases:
  - Range minimum/maximum queries (RMQ)
  - Range sum queries
  - Range updates (lazy propagation)
  - Combination with coordinate compression
  - A staple in competitive programming
```

### 3.2 Segment Tree Implementation

```python
class SegmentTree:
    """Segment Tree (range minimum query + point update)"""

    def __init__(self, data):
        self.n = len(data)
        self.tree = [0] * (4 * self.n)  # Allocate sufficient size
        self._build(data, 1, 0, self.n - 1)

    def _build(self, data, node, start, end):
        """Build the tree: O(n)"""
        if start == end:
            self.tree[node] = data[start]
            return

        mid = (start + end) // 2
        self._build(data, 2 * node, start, mid)
        self._build(data, 2 * node + 1, mid + 1, end)
        self.tree[node] = min(self.tree[2 * node], self.tree[2 * node + 1])

    def query(self, left, right):
        """Get the minimum value in range [left, right]: O(log n)"""
        return self._query(1, 0, self.n - 1, left, right)

    def _query(self, node, start, end, left, right):
        # Out of range
        if right < start or end < left:
            return float('inf')

        # Completely within range
        if left <= start and end <= right:
            return self.tree[node]

        # Partially within range
        mid = (start + end) // 2
        left_min = self._query(2 * node, start, mid, left, right)
        right_min = self._query(2 * node + 1, mid + 1, end, left, right)
        return min(left_min, right_min)

    def update(self, index, value):
        """Update the value at index to value: O(log n)"""
        self._update(1, 0, self.n - 1, index, value)

    def _update(self, node, start, end, index, value):
        if start == end:
            self.tree[node] = value
            return

        mid = (start + end) // 2
        if index <= mid:
            self._update(2 * node, start, mid, index, value)
        else:
            self._update(2 * node + 1, mid + 1, end, index, value)

        self.tree[node] = min(self.tree[2 * node], self.tree[2 * node + 1])


# Usage example
data = [2, 1, 5, 3, 4, 2]
st = SegmentTree(data)

print(st.query(0, 5))  # 1 (overall minimum)
print(st.query(2, 4))  # 3 (minimum of [5,3,4])
print(st.query(0, 2))  # 1 (minimum of [2,1,5])

st.update(1, 10)       # Update data[1] to 10
print(st.query(0, 2))  # 2 (minimum of [2,10,5])
```

### 3.3 Range Sum Segment Tree

```python
class SumSegmentTree:
    """Range sum query + point update"""

    def __init__(self, data):
        self.n = len(data)
        self.tree = [0] * (4 * self.n)
        self._build(data, 1, 0, self.n - 1)

    def _build(self, data, node, start, end):
        if start == end:
            self.tree[node] = data[start]
            return
        mid = (start + end) // 2
        self._build(data, 2 * node, start, mid)
        self._build(data, 2 * node + 1, mid + 1, end)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def query(self, left, right):
        """Sum of range [left, right]: O(log n)"""
        return self._query(1, 0, self.n - 1, left, right)

    def _query(self, node, start, end, left, right):
        if right < start or end < left:
            return 0
        if left <= start and end <= right:
            return self.tree[node]
        mid = (start + end) // 2
        return (self._query(2 * node, start, mid, left, right) +
                self._query(2 * node + 1, mid + 1, end, left, right))

    def update(self, index, value):
        """Update the value at index to value: O(log n)"""
        self._update(1, 0, self.n - 1, index, value)

    def _update(self, node, start, end, index, value):
        if start == end:
            self.tree[node] = value
            return
        mid = (start + end) // 2
        if index <= mid:
            self._update(2 * node, start, mid, index, value)
        else:
            self._update(2 * node + 1, mid + 1, end, index, value)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

# Usage example
data = [1, 3, 5, 7, 9, 11]
st = SumSegmentTree(data)
print(st.query(1, 3))  # 15 (3 + 5 + 7)
st.update(2, 10)       # data[2] = 10
print(st.query(1, 3))  # 20 (3 + 10 + 7)
```

### 3.4 Lazy Propagation

```python
class LazySegmentTree:
    """Segment Tree with lazy propagation: Range update + range query O(log n)"""

    def __init__(self, data):
        self.n = len(data)
        self.tree = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)  # Lazy update values
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
        """Propagate lazy updates to child nodes"""
        if self.lazy[node] != 0:
            mid = (start + end) // 2
            # Left child
            self.tree[2 * node] += self.lazy[node] * (mid - start + 1)
            self.lazy[2 * node] += self.lazy[node]
            # Right child
            self.tree[2 * node + 1] += self.lazy[node] * (end - mid)
            self.lazy[2 * node + 1] += self.lazy[node]
            # Clear own lazy value
            self.lazy[node] = 0

    def range_update(self, left, right, value):
        """Add value to range [left, right]: O(log n)"""
        self._range_update(1, 0, self.n - 1, left, right, value)

    def _range_update(self, node, start, end, left, right, value):
        if right < start or end < left:
            return
        if left <= start and end <= right:
            self.tree[node] += value * (end - start + 1)
            self.lazy[node] += value
            return
        self._push_down(node, start, end)
        mid = (start + end) // 2
        self._range_update(2 * node, start, mid, left, right, value)
        self._range_update(2 * node + 1, mid + 1, end, left, right, value)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def query(self, left, right):
        """Sum of range [left, right]: O(log n)"""
        return self._query(1, 0, self.n - 1, left, right)

    def _query(self, node, start, end, left, right):
        if right < start or end < left:
            return 0
        if left <= start and end <= right:
            return self.tree[node]
        self._push_down(node, start, end)
        mid = (start + end) // 2
        return (self._query(2 * node, start, mid, left, right) +
                self._query(2 * node + 1, mid + 1, end, left, right))


# Usage example
data = [1, 3, 5, 7, 9, 11]
lst = LazySegmentTree(data)
print(lst.query(1, 4))       # 24 (3 + 5 + 7 + 9)
lst.range_update(1, 3, 10)   # [1, 13, 15, 17, 9, 11]
print(lst.query(1, 4))       # 54 (13 + 15 + 17 + 9)
```

---

## 4. Fenwick Tree (Binary Indexed Tree / BIT)

### 4.1 Basic Concept and Implementation

```python
class FenwickTree:
    """Fenwick Tree (BIT): Range sum query and point update
    Simpler than Segment Tree but limited to range sums"""

    def __init__(self, n):
        self.n = n
        self.tree = [0] * (n + 1)  # 1-indexed

    def update(self, i, delta):
        """Add delta to index i: O(log n)"""
        i += 1  # Convert to 1-indexed
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)  # Add lowest set bit

    def prefix_sum(self, i):
        """Prefix sum of [0, i]: O(log n)"""
        i += 1  # Convert to 1-indexed
        total = 0
        while i > 0:
            total += self.tree[i]
            i -= i & (-i)  # Remove lowest set bit
        return total

    def range_sum(self, left, right):
        """Range sum of [left, right]: O(log n)"""
        if left == 0:
            return self.prefix_sum(right)
        return self.prefix_sum(right) - self.prefix_sum(left - 1)

    @classmethod
    def from_array(cls, arr):
        """Build from array in O(n)"""
        ft = cls(len(arr))
        for i, val in enumerate(arr):
            ft.update(i, val)
        return ft


# Usage example
data = [1, 3, 5, 7, 9, 11]
ft = FenwickTree.from_array(data)

print(ft.prefix_sum(3))    # 16 (1 + 3 + 5 + 7)
print(ft.range_sum(2, 4))  # 21 (5 + 7 + 9)

ft.update(2, 5)            # data[2] += 5 -> [1, 3, 10, 7, 9, 11]
print(ft.range_sum(2, 4))  # 26 (10 + 7 + 9)


# Fenwick Tree vs Segment Tree:
# +--------------------+--------------+--------------+
# | Property           | Fenwick Tree | Segment Tree |
# +--------------------+--------------+--------------+
# | Implementation     | Very concise | Somewhat     |
# | simplicity         |              | complex      |
# | Memory             | O(n)         | O(4n)        |
# | Constant factor    | Fast         | Somewhat slow|
# | Supported queries  | Range sum    | Arbitrary    |
# |                    | only         |              |
# | Range update       | Requires     | Lazy         |
# |                    | workarounds  | propagation  |
# | Min/Max            | No           | Yes          |
# +--------------------+--------------+--------------+
```

### 4.2 2D Fenwick Tree

```python
class FenwickTree2D:
    """2D Fenwick Tree: 2D range sum query"""

    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.tree = [[0] * (cols + 1) for _ in range(rows + 1)]

    def update(self, row, col, delta):
        """Add delta to (row, col): O(log(R) * log(C))"""
        r = row + 1
        while r <= self.rows:
            c = col + 1
            while c <= self.cols:
                self.tree[r][c] += delta
                c += c & (-c)
            r += r & (-r)

    def prefix_sum(self, row, col):
        """Prefix sum from (0,0) to (row, col)"""
        total = 0
        r = row + 1
        while r > 0:
            c = col + 1
            while c > 0:
                total += self.tree[r][c]
                c -= c & (-c)
            r -= r & (-r)
        return total

    def range_sum(self, r1, c1, r2, c2):
        """Rectangle range sum from (r1,c1) to (r2,c2)"""
        total = self.prefix_sum(r2, c2)
        if r1 > 0:
            total -= self.prefix_sum(r1 - 1, c2)
        if c1 > 0:
            total -= self.prefix_sum(r2, c1 - 1)
        if r1 > 0 and c1 > 0:
            total += self.prefix_sum(r1 - 1, c1 - 1)
        return total

# Usage example: Range sum of a 2D matrix
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
ft2d = FenwickTree2D(3, 3)
for r in range(3):
    for c in range(3):
        ft2d.update(r, c, matrix[r][c])

print(ft2d.range_sum(0, 0, 1, 1))  # 12 (1+2+4+5)
print(ft2d.range_sum(1, 1, 2, 2))  # 28 (5+6+8+9)
```

---

## 5. Rope

### 5.1 Concept and Implementation

```python
class RopeNode:
    """Rope: Efficient operations on long strings (for text editors)"""

    def __init__(self, text=None, left=None, right=None):
        if text is not None:
            # Leaf node
            self.text = text
            self.weight = len(text)
            self.left = None
            self.right = None
        else:
            # Internal node
            self.text = None
            self.left = left
            self.right = right
            self.weight = self._total_length(left) if left else 0

    def _total_length(self, node):
        if node is None:
            return 0
        if node.text is not None:
            return len(node.text)
        return node.weight + self._total_length(node.right)

class Rope:
    """Rope: A string data structure for text editors

    Operation complexities:
    - Concatenation: O(1) (just create a new root node)
    - Split: O(log n)
    - Index access: O(log n)
    - Insert: O(log n)
    - Delete: O(log n)

    Comparison with regular strings:
    - Concatenation: O(1) vs O(n) (Rope is overwhelmingly faster)
    - Access: O(log n) vs O(1) (strings are faster)
    """

    def __init__(self, text=""):
        self.root = RopeNode(text=text) if text else None

    def concat(self, other):
        """Concatenate two Ropes: O(1)"""
        if not self.root:
            self.root = other.root
        elif other.root:
            self.root = RopeNode(left=self.root, right=other.root)
        return self

    def index(self, i):
        """Get the i-th character: O(log n)"""
        return self._index(self.root, i)

    def _index(self, node, i):
        if node is None:
            raise IndexError("index out of range")

        if node.text is not None:
            # Leaf node
            return node.text[i]

        if i < node.weight:
            return self._index(node.left, i)
        else:
            return self._index(node.right, i - node.weight)

    def to_string(self):
        """Get the full string: O(n)"""
        result = []
        self._collect(self.root, result)
        return "".join(result)

    def _collect(self, node, result):
        if node is None:
            return
        if node.text is not None:
            result.append(node.text)
            return
        self._collect(node.left, result)
        self._collect(node.right, result)

    def __len__(self):
        return self._length(self.root)

    def _length(self, node):
        if node is None:
            return 0
        if node.text is not None:
            return len(node.text)
        return node.weight + self._length(node.right)


# Usage example (internal representation of a text editor)
rope1 = Rope("Hello, ")
rope2 = Rope("World!")
rope1.concat(rope2)
print(rope1.to_string())    # "Hello, World!"
print(rope1.index(7))       # 'W'
print(len(rope1))            # 13

# Where Rope is used:
# - Xi Editor (a Rust text editor)
# - JetBrains IDE text buffer
# - Parts of Visual Studio Code
# - CLion text management
```

---

## 6. Merkle Tree

### 6.1 Concept and Implementation

```python
import hashlib

class MerkleNode:
    def __init__(self, data=None, left=None, right=None):
        if data is not None:
            self.hash = hashlib.sha256(data.encode()).hexdigest()
        else:
            combined = left.hash + right.hash
            self.hash = hashlib.sha256(combined.encode()).hexdigest()
        self.left = left
        self.right = right
        self.data = data

class MerkleTree:
    """Merkle Tree: Efficient verification of data integrity

    Use cases:
    - Git: Integrity of commits and files
    - Bitcoin/Ethereum: Transaction verification
    - Amazon DynamoDB: Data synchronization between replicas
    - IPFS: Content addressing
    """

    def __init__(self, data_list):
        leaves = [MerkleNode(data=d) for d in data_list]

        # If odd number, duplicate the last element
        if len(leaves) % 2 == 1:
            leaves.append(MerkleNode(data=data_list[-1]))

        self.root = self._build(leaves)

    def _build(self, nodes):
        """Build the tree bottom-up: O(n)"""
        while len(nodes) > 1:
            next_level = []
            for i in range(0, len(nodes), 2):
                if i + 1 < len(nodes):
                    parent = MerkleNode(left=nodes[i], right=nodes[i + 1])
                else:
                    parent = MerkleNode(left=nodes[i], right=nodes[i])
                next_level.append(parent)
            nodes = next_level
        return nodes[0] if nodes else None

    def get_root_hash(self):
        """Get the root hash"""
        return self.root.hash if self.root else None

    def get_proof(self, index, data_list):
        """Merkle Proof: Generate a proof of existence for specific data
        Can prove overall integrity with only O(log n) hash values"""
        # Reconstruct leaf nodes
        leaves = [MerkleNode(data=d) for d in data_list]
        if len(leaves) % 2 == 1:
            leaves.append(MerkleNode(data=data_list[-1]))

        proof = []
        nodes = leaves
        target_index = index

        while len(nodes) > 1:
            next_level = []
            for i in range(0, len(nodes), 2):
                if i + 1 < len(nodes):
                    parent = MerkleNode(left=nodes[i], right=nodes[i + 1])
                    # Record the hash of the sibling node needed for the proof
                    if i == target_index or i + 1 == target_index:
                        sibling_idx = i + 1 if i == target_index else i
                        side = "right" if sibling_idx > target_index else "left"
                        proof.append((side, nodes[sibling_idx].hash))
                        target_index = len(next_level)
                else:
                    parent = MerkleNode(left=nodes[i], right=nodes[i])
                next_level.append(parent)
            nodes = next_level

        return proof

    @staticmethod
    def verify_proof(data, proof, root_hash):
        """Verify a Merkle Proof: O(log n)"""
        current_hash = hashlib.sha256(data.encode()).hexdigest()

        for side, sibling_hash in proof:
            if side == "right":
                combined = current_hash + sibling_hash
            else:
                combined = sibling_hash + current_hash
            current_hash = hashlib.sha256(combined.encode()).hexdigest()

        return current_hash == root_hash


# Usage example
data = ["tx1", "tx2", "tx3", "tx4"]
mt = MerkleTree(data)
print(f"Root Hash: {mt.get_root_hash()[:16]}...")

# Generate proof of existence for tx2 (only O(log n) data)
proof = mt.get_proof(1, data)
print(f"Proof size: {len(proof)} hashes")  # log2(4) = 2

# Verification
is_valid = MerkleTree.verify_proof("tx2", proof, mt.get_root_hash())
print(f"Valid: {is_valid}")  # True

# Verification with tampered data
is_valid = MerkleTree.verify_proof("tx_fake", proof, mt.get_root_hash())
print(f"Valid: {is_valid}")  # False
```

---

## 7. LFU Cache

### 7.1 Implementation

```python
from collections import defaultdict, OrderedDict

class LFUCache:
    """Least Frequently Used Cache
    All operations O(1)"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}            # key -> value
        self.freq = {}             # key -> frequency
        self.freq_to_keys = defaultdict(OrderedDict)  # freq -> OrderedDict of keys
        self.min_freq = 0

    def get(self, key):
        if key not in self.cache:
            return -1

        # Increment frequency
        self._update_freq(key)
        return self.cache[key]

    def put(self, key, value):
        if self.capacity <= 0:
            return

        if key in self.cache:
            self.cache[key] = value
            self._update_freq(key)
            return

        if len(self.cache) >= self.capacity:
            # Evict the oldest key with the lowest frequency
            evict_key, _ = self.freq_to_keys[self.min_freq].popitem(last=False)
            del self.cache[evict_key]
            del self.freq[evict_key]

        self.cache[key] = value
        self.freq[key] = 1
        self.freq_to_keys[1][key] = True
        self.min_freq = 1

    def _update_freq(self, key):
        """Update the frequency of a key"""
        old_freq = self.freq[key]
        new_freq = old_freq + 1
        self.freq[key] = new_freq

        # Remove from old frequency group
        del self.freq_to_keys[old_freq][key]
        if not self.freq_to_keys[old_freq]:
            del self.freq_to_keys[old_freq]
            if self.min_freq == old_freq:
                self.min_freq += 1

        # Add to new frequency group
        self.freq_to_keys[new_freq][key] = True


# Usage example
cache = LFUCache(3)
cache.put("a", 1)
cache.put("b", 2)
cache.put("c", 3)
cache.get("a")       # 1 -> freq("a") = 2
cache.get("a")       # 1 -> freq("a") = 3
cache.get("b")       # 2 -> freq("b") = 2
cache.put("d", 4)    # "c" is evicted (freq=1 and oldest)
print(cache.get("c"))  # -1 (already evicted)
print(cache.get("a"))  # 1
print(cache.get("d"))  # 4
```

---

## 8. Disjoint Set Applications

### 8.1 Weighted Union-Find

```python
class WeightedUnionFind:
    """Weighted Union-Find: Manages relative weights from each node to the root"""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.weight = [0] * n  # Relative weight from parent

    def find(self, x):
        """Find with path compression (also updates weights)"""
        if self.parent[x] != x:
            root = self.find(self.parent[x])
            self.weight[x] += self.weight[self.parent[x]]
            self.parent[x] = root
        return self.parent[x]

    def union(self, x, y, w):
        """Merge x and y, where weight(y) - weight(x) = w"""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return self.weight[x] - self.weight[y] == w  # Consistency check

        # w = weight(y) - weight(x)
        # -> weight(root_y) = weight(x) - weight(y) + w
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
            self.weight[root_x] = self.weight[y] - self.weight[x] + w
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
            self.weight[root_y] = self.weight[x] - self.weight[y] - w
        else:
            self.parent[root_y] = root_x
            self.weight[root_y] = self.weight[x] - self.weight[y] - w
            self.rank[root_x] += 1

        return True

    def diff(self, x, y):
        """Return weight(y) - weight(x)"""
        if self.find(x) != self.find(y):
            return None  # Not in the same set
        return self.weight[x] - self.weight[y]


# Usage example: Relative rating problem
# A is 3 higher than B, B is 2 higher than C -> How much higher is A than C?
wuf = WeightedUnionFind(3)
wuf.union(0, 1, 3)   # score[1] - score[0] = 3
wuf.union(1, 2, 2)   # score[2] - score[1] = 2
print(wuf.diff(0, 2)) # 5 (score[2] - score[0] = 5)
```

---

## 9. Other Advanced Data Structures

### 9.1 Overview

```
+------------------+------------------+-------------------+
| Data Structure   | Use Case         | Complexity        |
+------------------+------------------+-------------------+
| Segment Tree     | Range queries    | O(log n) update/  |
|                  |                  | query             |
| Fenwick Tree(BIT)| Range sum queries| O(log n)          |
| Disjoint Set     | Set merging      | O(alpha(n)) ~ O(1)|
| LRU Cache        | Cache management | O(1) all ops      |
| LFU Cache        | Frequency-based  | O(1) all ops      |
|                  | cache            |                   |
| Rope             | Long string ops  | O(log n) concat   |
| Merkle Tree      | Data integrity   | O(log n) verify   |
|                  | verification     |                   |
| Bloom Filter     | Existence check  | O(k) search/insert|
| Cuckoo Filter    | Existence check  | O(1) search       |
|                  | + deletion       |                   |
| Skip List        | Ordered data     | O(log n) search   |
| Trie             | String search    | O(m) search       |
| Suffix Array     | Substring search | O(m log n) search |
| Suffix Tree      | Substring search | O(m) search       |
| Wavelet Tree     | Range frequency  | O(log sigma) query|
|                  | queries          |                   |
| Persistent DS    | Version control  | O(log n) per op   |
| Van Emde Boas    | Integer sets     | O(log log U)      |
| Splay Tree       | Self-adjusting   | Amortized O(log n)|
|                  | BST              |                   |
| Treap            | Probabilistic BST| Expected O(log n) |
+------------------+------------------+-------------------+
```

### 9.2 Treap (Tree + Heap)

```python
import random

class TreapNode:
    def __init__(self, key):
        self.key = key
        self.priority = random.random()  # Random priority
        self.left = None
        self.right = None
        self.size = 1  # Subtree size

class Treap:
    """Treap: BST property (key) + Heap property (priority)
    Random priorities ensure probabilistic balance"""

    def __init__(self):
        self.root = None

    def _size(self, node):
        return node.size if node else 0

    def _update(self, node):
        if node:
            node.size = 1 + self._size(node.left) + self._size(node.right)

    def _split(self, node, key):
        """Split the tree by key: left < key <= right"""
        if not node:
            return None, None

        if key <= node.key:
            left, node.left = self._split(node.left, key)
            self._update(node)
            return left, node
        else:
            node.right, right = self._split(node.right, key)
            self._update(node)
            return node, right

    def _merge(self, left, right):
        """Merge two trees (all keys in left < all keys in right)"""
        if not left or not right:
            return left or right

        if left.priority > right.priority:
            left.right = self._merge(left.right, right)
            self._update(left)
            return left
        else:
            right.left = self._merge(left, right.left)
            self._update(right)
            return right

    def insert(self, key):
        """Insert: Expected O(log n)"""
        left, right = self._split(self.root, key)
        node = TreapNode(key)
        self.root = self._merge(self._merge(left, node), right)

    def delete(self, key):
        """Delete: Expected O(log n)"""
        self.root = self._delete(self.root, key)

    def _delete(self, node, key):
        if not node:
            return None
        if key < node.key:
            node.left = self._delete(node.left, key)
        elif key > node.key:
            node.right = self._delete(node.right, key)
        else:
            node = self._merge(node.left, node.right)
        if node:
            self._update(node)
        return node

    def kth(self, k):
        """k-th smallest element: O(log n)"""
        return self._kth(self.root, k)

    def _kth(self, node, k):
        if not node:
            return None
        left_size = self._size(node.left)
        if k == left_size + 1:
            return node.key
        elif k <= left_size:
            return self._kth(node.left, k)
        else:
            return self._kth(node.right, k - left_size - 1)


# Advantages of Treap:
# - Implementation is significantly simpler than red-black trees
# - Split/Merge based operations enable flexible manipulation
# - Expected complexity is O(log n)
# - Can be extended for range operations (reversal, shifting)
```

### 9.3 Persistent Data Structure

```python
class PersistentNode:
    """Persistent node: Copy-on-write on update"""
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class PersistentBST:
    """Persistent BST: All versions of the tree are accessible
    Creates only O(log n) new nodes per operation"""

    def __init__(self):
        self.versions = [None]  # Version 0: empty tree

    def insert(self, val, version=-1):
        """Insert into the specified version and create a new version"""
        root = self.versions[version]
        new_root = self._insert(root, val)
        self.versions.append(new_root)
        return len(self.versions) - 1  # New version number

    def _insert(self, node, val):
        if not node:
            return PersistentNode(val)

        if val < node.val:
            return PersistentNode(node.val,
                                  self._insert(node.left, val),
                                  node.right)  # Right subtree is shared
        elif val > node.val:
            return PersistentNode(node.val,
                                  node.left,  # Left subtree is shared
                                  self._insert(node.right, val))
        return node  # Ignore duplicates

    def search(self, val, version=-1):
        """Search in the specified version"""
        return self._search(self.versions[version], val)

    def _search(self, node, val):
        if not node:
            return False
        if val == node.val:
            return True
        if val < node.val:
            return self._search(node.left, val)
        return self._search(node.right, val)

    def inorder(self, version=-1):
        """In-order traversal of the specified version"""
        result = []
        self._inorder(self.versions[version], result)
        return result

    def _inorder(self, node, result):
        if not node:
            return
        self._inorder(node.left, result)
        result.append(node.val)
        self._inorder(node.right, result)


# Usage example
pbst = PersistentBST()
v1 = pbst.insert(5)   # v1: {5}
v2 = pbst.insert(3)   # v2: {3, 5}
v3 = pbst.insert(7)   # v3: {3, 5, 7}
v4 = pbst.insert(4, version=v2)  # v4: Based on v2 -> {3, 4, 5}

print(pbst.inorder(v1))  # [5]
print(pbst.inorder(v2))  # [3, 5]
print(pbst.inorder(v3))  # [3, 5, 7]
print(pbst.inorder(v4))  # [3, 4, 5] (independent of v3)

# Where persistent data structures are used:
# - Git: Version control of file trees
# - Clojure/Haskell: Immutable data structures
# - React: Diff detection in Virtual DOM
# - Databases: MVCC (Multi-Version Concurrency Control)
```

---

## 10. Practice Exercises

### Exercise 1: Bloom Filter (Basics)
Implement a Bloom Filter and measure the false positive rate by varying parameters (m, k). Also implement a feature that automatically calculates the optimal parameters.

### Exercise 2: Segment Tree (Applied)
Implement a Segment Tree that supports the following:
- Range minimum query and point update
- Range sum query
- Range update with lazy propagation
- Range maximum and its position retrieval

### Exercise 3: Skip List (Applied)
Implement a Skip List with the following features:
- Insert, search, delete
- Range search (get keys from start to end)
- Rank retrieval (k-th smallest element)
- Level structure visualization

### Exercise 4: LFU Cache (Advanced)
Implement an LFU cache that supports all operations in O(1). Also add TTL (time-to-live) support.

### Exercise 5: Merkle Tree (Advanced)
Implement a Merkle Tree with the following features:
- Root hash computation
- Merkle Proof generation and verification
- Data tampering detection demo
- Diff detection between two Merkle Trees

### Exercise 6: Cuckoo Filter (Advanced)
Implement a Cuckoo Filter and benchmark its performance against Bloom Filter:
- Benchmarks for insert, lookup, and delete
- False positive rate comparison
- Memory usage comparison

---


## FAQ

### Q1: What is the most important point in learning this topic?

Gaining practical experience is the most important aspect. Understanding deepens not just through theory but by actually writing and running code.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this used in practice?

Knowledge of this topic is frequently applied in day-to-day development work. It becomes particularly important during code reviews and architecture design.

---

## Summary

| Data Structure | Characteristic | Primary Use Case |
|-----------|------|---------|
| Bloom Filter | False positives possible, no false negatives | Existence checks, caching |
| Cuckoo Filter | Bloom Filter + deletion support | Existence checks requiring deletion |
| Skip List | Probabilistic balance O(log n) | Redis Sorted Set |
| Segment Tree | Range query O(log n) | Competitive programming, DB |
| Fenwick Tree | Range sum O(log n), concise | Dynamic prefix sum updates |
| Union-Find | Near O(1) set merging | Clustering, MST |
| Rope | O(1) concatenation | Text editors |
| Merkle Tree | O(log n) integrity verification | Git, blockchain |
| LFU Cache | Frequency-based O(1) | CDN, DB cache |
| Treap | Probabilistic BST + Split/Merge | Flexible ordered sets |
| Persistent DS | Version control | Functional programming |

---

## Recommended Next Guides

---

## References
1. Bloom, B. H. "Space/Time Trade-offs in Hash Coding with Allowable Errors." 1970.
2. Pugh, W. "Skip Lists: A Probabilistic Alternative to Balanced Trees." 1990.
3. Fan, B. et al. "Cuckoo Filter: Practically Better Than Bloom." CoNEXT 2014.
4. Merkle, R. C. "A Digital Signature Based on a Conventional Encryption Function." CRYPTO 1987.
5. Driscoll, J. R. et al. "Making Data Structures Persistent." STOC 1986.
6. Aragon, C. R., Seidel, R. "Randomized Search Trees." FOCS 1989.
7. De Berg, M. et al. "Computational Geometry." Chapter 10: Segment Trees.
