# Hash Tables

> Hash tables are the most frequently used data structure in practice, achieving expected O(1) lookups.

## Learning Objectives

- [ ] Understand hash functions and collision resolution mechanisms
- [ ] Explain the performance characteristics of hash tables
- [ ] Understand practical usage patterns (dict/set/Map, etc.)
- [ ] Recognize security considerations for hash tables
- [ ] Understand hash table internal implementations across different languages
- [ ] Master hashing techniques for distributed systems

## Prerequisites


---

## 1. How Hash Tables Work

### 1.1 Basic Structure

```
Hash Table: Key -> Hash Function -> Index -> Value

  Key "Alice" -> hash("Alice") = 0x7A3B...
              -> 0x7A3B % 8 = 3  (table size 8)
              -> table[3] = "Alice: 100"

  +-----+
  |  0  | -> (empty)
  |  1  | -> ("Bob", 85)
  |  2  | -> (empty)
  |  3  | -> ("Alice", 100)
  |  4  | -> ("Charlie", 92) -> ("Eve", 78)  <- collision (chaining)
  |  5  | -> (empty)
  |  6  | -> ("Diana", 88)
  |  7  | -> (empty)
  +-----+
```

The fundamental concept of a hash table is remarkably simple. An arbitrary key is converted to an integer value through a hash function, and the remainder of dividing that integer by the table size serves as the index. This mechanism enables direct access from key to value, achieving expected O(1) lookup time.

### 1.2 Hash Function Design Principles

A good hash function must satisfy the following properties:

```
Properties of a Good Hash Function:

  1. Determinism: Always returns the same output for the same input
  2. Uniform Distribution: Output is evenly distributed across the hash space
  3. Efficiency: Computation is fast (close to O(1))
  4. Avalanche Effect: A 1-bit change in input alters approximately half the output bits

Major Hash Functions:

  Division Method:
    h(k) = k mod m
    Avoid powers of 2 for m (primes are preferred)
    Example: m = 997 (prime)

  Multiplication Method:
    h(k) = floor(m * (k * A mod 1))
    A = (sqrt(5) - 1) / 2 ~ 0.6180339887 (golden ratio)
    Works well even when m is a power of 2

  Universal Hashing:
    h(k) = ((a * k + b) mod p) mod m
    a, b chosen randomly
    Collision probability bounded by 1/m

  MurmurHash3:
    Fast non-cryptographic hash function
    Used in Redis, Hadoop, Spark, etc.

  SipHash:
    PRF with Hash DoS resistance
    Standard in Python 3.4+, Rust, Perl
```

### 1.3 String Hashing

```python
# Examples of string hash functions

# 1. Naive approach (poor uniformity)
def bad_hash(s, m):
    """Sum of ASCII codes of each character"""
    return sum(ord(c) for c in s) % m
# "abc" and "bca" produce the same hash -> anagrams collide

# 2. Polynomial hash (commonly used)
def polynomial_hash(s, m, base=31):
    """Polynomial hash: s[0]*base^(n-1) + s[1]*base^(n-2) + ... + s[n-1]"""
    h = 0
    for c in s:
        h = (h * base + ord(c)) % m
    return h
# "abc" -> 31^2*97 + 31*98 + 99 = 96262
# "bca" -> 31^2*98 + 31*99 + 97 = 97168  -> different hash values

# 3. Equivalent to Java's String.hashCode()
def java_string_hash(s):
    """s[0]*31^(n-1) + s[1]*31^(n-2) + ... + s[n-1]"""
    h = 0
    for c in s:
        h = h * 31 + ord(c)
        h &= 0xFFFFFFFF  # Constrain to 32 bits
    return h

# 4. FNV-1a hash (fast, uniform distribution)
def fnv1a_hash(data, m):
    """FNV-1a: offset_basis XOR byte -> multiply by prime"""
    FNV_OFFSET = 0xcbf29ce484222325
    FNV_PRIME = 0x100000001b3
    h = FNV_OFFSET
    for byte in data.encode():
        h ^= byte
        h *= FNV_PRIME
        h &= 0xFFFFFFFFFFFFFFFF  # Constrain to 64 bits
    return h % m
```

### 1.4 Collision Resolution

```
Two Major Collision Resolution Strategies:

  1. Separate Chaining:
     -> Store colliding entries in a linked list at the same slot
     -> Simple implementation, load factor can exceed 1
     -> Java HashMap, Go map

  2. Open Addressing:
     -> Probe for an alternative slot upon collision
     -> Better memory efficiency, cache-friendly
     -> Python dict, Rust HashMap

     Probing strategies:
     - Linear probing: h(k)+1, h(k)+2, ... (clustering problem)
     - Quadratic probing: h(k)+1^2, h(k)+2^2, ...
     - Double hashing: h(k)+i*h2(k)

  Load Factor = number of elements / table size
  -> Chaining: resize at 0.75 (Java)
  -> Open addressing: resize at 2/3 (Python)
```

### 1.5 Detailed Chaining Implementation

```python
class HashTableChaining:
    """Hash table using separate chaining"""

    def __init__(self, capacity=16, load_factor_threshold=0.75):
        self.capacity = capacity
        self.load_factor_threshold = load_factor_threshold
        self.size = 0
        self.buckets = [[] for _ in range(capacity)]

    def _hash(self, key):
        """Compute hash value"""
        return hash(key) % self.capacity

    def put(self, key, value):
        """Insert a key-value pair"""
        idx = self._hash(key)
        bucket = self.buckets[idx]

        # Update existing key
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return

        # New insertion
        bucket.append((key, value))
        self.size += 1

        # Check load factor
        if self.size / self.capacity > self.load_factor_threshold:
            self._resize()

    def get(self, key, default=None):
        """Retrieve value by key"""
        idx = self._hash(key)
        for k, v in self.buckets[idx]:
            if k == key:
                return v
        return default

    def remove(self, key):
        """Remove a key"""
        idx = self._hash(key)
        bucket = self.buckets[idx]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket.pop(i)
                self.size -= 1
                return True
        return False

    def __contains__(self, key):
        """Support for the `in` operator"""
        return self.get(key) is not None

    def _resize(self):
        """Double the table size"""
        old_buckets = self.buckets
        self.capacity *= 2
        self.buckets = [[] for _ in range(self.capacity)]
        self.size = 0
        for bucket in old_buckets:
            for key, value in bucket:
                self.put(key, value)

    def load_factor(self):
        """Return the current load factor"""
        return self.size / self.capacity

    def __len__(self):
        return self.size

    def __repr__(self):
        items = []
        for bucket in self.buckets:
            for k, v in bucket:
                items.append(f"{k!r}: {v!r}")
        return "{" + ", ".join(items) + "}"


# Usage example
ht = HashTableChaining()
ht.put("name", "Alice")
ht.put("age", 30)
ht.put("email", "alice@example.com")
print(ht.get("name"))   # "Alice"
print("age" in ht)      # True
print(ht.load_factor()) # 0.1875 (3/16)
ht.remove("age")
print(len(ht))          # 2
```

### 1.6 Detailed Open Addressing Implementation

```python
class HashTableOpenAddressing:
    """Hash table using open addressing (linear probing)"""

    EMPTY = object()    # Sentinel for empty slots
    DELETED = object()  # Sentinel for deleted slots (tombstone)

    def __init__(self, capacity=16, load_factor_threshold=0.67):
        self.capacity = capacity
        self.load_factor_threshold = load_factor_threshold
        self.size = 0
        self.keys = [self.EMPTY] * capacity
        self.values = [None] * capacity

    def _hash(self, key):
        return hash(key) % self.capacity

    def _probe(self, key):
        """Find a slot using linear probing"""
        idx = self._hash(key)
        first_deleted = None

        for i in range(self.capacity):
            pos = (idx + i) % self.capacity

            if self.keys[pos] is self.EMPTY:
                # Reached an empty slot -> key does not exist
                return (first_deleted if first_deleted is not None else pos, False)

            if self.keys[pos] is self.DELETED:
                if first_deleted is None:
                    first_deleted = pos
                continue

            if self.keys[pos] == key:
                return (pos, True)  # Key found

        # Table is full (normally prevented by resizing)
        return (first_deleted if first_deleted is not None else -1, False)

    def put(self, key, value):
        pos, found = self._probe(key)
        if found:
            self.values[pos] = value  # Update
        else:
            self.keys[pos] = key
            self.values[pos] = value
            self.size += 1
            if self.size / self.capacity > self.load_factor_threshold:
                self._resize()

    def get(self, key, default=None):
        pos, found = self._probe(key)
        if found:
            return self.values[pos]
        return default

    def remove(self, key):
        """Delete: place a DELETED tombstone marker"""
        pos, found = self._probe(key)
        if found:
            self.keys[pos] = self.DELETED
            self.values[pos] = None
            self.size -= 1
            return True
        return False

    def _resize(self):
        old_keys = self.keys
        old_values = self.values
        self.capacity *= 2
        self.keys = [self.EMPTY] * self.capacity
        self.values = [None] * self.capacity
        self.size = 0
        for k, v in zip(old_keys, old_values):
            if k is not self.EMPTY and k is not self.DELETED:
                self.put(k, v)


# Visualization of the clustering problem with linear probing
# Slots: [A][B][C][_][_][E][F][G][_][_]
#          ^^^^^^^^          ^^^^^^^^
#          Cluster 1         Cluster 2
# When a new element hashes near a cluster, the cluster grows
# -> Search time degrades from O(1) to O(n)
```

### 1.7 Robin Hood Hashing

```python
class RobinHoodHashTable:
    """Robin Hood hashing: prioritizes 'poor' elements during insertion"""

    EMPTY = None

    def __init__(self, capacity=16):
        self.capacity = capacity
        self.size = 0
        self.keys = [self.EMPTY] * capacity
        self.values = [None] * capacity
        self.distances = [0] * capacity  # Distance from ideal position

    def _hash(self, key):
        return hash(key) % self.capacity

    def put(self, key, value):
        if self.size / self.capacity > 0.8:
            self._resize()

        idx = self._hash(key)
        dist = 0

        while True:
            pos = (idx + dist) % self.capacity

            if self.keys[pos] is self.EMPTY:
                # Insert into empty slot
                self.keys[pos] = key
                self.values[pos] = value
                self.distances[pos] = dist
                self.size += 1
                return

            if self.keys[pos] == key:
                # Update existing key
                self.values[pos] = value
                return

            # Robin Hood: swap if the current element has a shorter probe distance
            if self.distances[pos] < dist:
                # Displace the "rich" element (shorter distance)
                key, self.keys[pos] = self.keys[pos], key
                value, self.values[pos] = self.values[pos], value
                dist, self.distances[pos] = self.distances[pos], dist

            dist += 1

    def get(self, key, default=None):
        idx = self._hash(key)
        dist = 0

        while True:
            pos = (idx + dist) % self.capacity

            if self.keys[pos] is self.EMPTY:
                return default

            if self.distances[pos] < dist:
                # Should have been found before reaching this position
                return default

            if self.keys[pos] == key:
                return self.values[pos]

            dist += 1

    def _resize(self):
        old_keys = self.keys
        old_values = self.values
        self.capacity *= 2
        self.keys = [self.EMPTY] * self.capacity
        self.values = [None] * self.capacity
        self.distances = [0] * self.capacity
        self.size = 0
        for k, v in zip(old_keys, old_values):
            if k is not self.EMPTY:
                self.put(k, v)


# Advantages of Robin Hood hashing:
# - Worst-case probe distance improved to O(log log n)
# - Low variance in probe distances (close to uniform)
# - Used by Rust's HashMap (2015-2021, then migrated to hashbrown/Swiss Table)
```

### 1.8 Time Complexity

```
Hash Table Time Complexity:

  +----------+----------+----------+
  | Operation| Expected | Worst    |
  +----------+----------+----------+
  | Insert   | O(1)     | O(n)     |
  | Search   | O(1)     | O(n)     |
  | Delete   | O(1)     | O(n)     |
  | Resize   | O(n)     | O(n)     |
  +----------+----------+----------+

  Conditions for worst-case O(n):
  - All keys collide into the same slot
  - Deliberate attack (Hash DoS)

  Countermeasures:
  - Randomized hashing (SipHash: Python, Rust)
  - Fallback to red-black trees (Java 8+ HashMap)

  Amortized analysis of resizing:
  - When the table size is n, the resize cost is O(n)
  - At least n/2 insertions occur between resizes
  - Amortized cost = O(n) / (n/2) = O(1)
  - Thus, the amortized cost per insertion is O(1)
```

---

## 2. Hash Table Internal Implementations Across Languages

### 2.1 Python dict Internals

```python
# Python 3.6+ dict uses a compact dictionary (order-preserving)
# Internal structure:
#   - indices: hash table (array of indices)
#   - entries: compact array of (hash, key, value)

# Conceptual diagram:
# indices = [None, 1, None, None, 0, None, None, 2]
#                  |                |              |
# entries = [(hash_a, "age", 30),      # index 0
#            (hash_n, "name", "Alice"), # index 1
#            (hash_e, "email", "x@y")] # index 2

# Benefits:
# 1. Memory efficiency: indices use 1-byte entries (when element count < 256)
# 2. Order preservation: entries array maintains insertion order
# 3. Fast iteration: sequential traversal of the entries array

# Special optimizations in Python dict
d = {}

# 1. Key-Sharing dictionaries (keys shared across instances of the same class)
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# p1.__dict__ and p2.__dict__ share the key array
# -> Memory savings (especially with many class instances)

# 2. Special handling of strings
# Short strings are interned (same object is reused)
a = "hello"
b = "hello"
print(a is b)  # True (same object)

# 3. __hash__ and __eq__ protocol
class CustomKey:
    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        return isinstance(other, CustomKey) and self.value == other.value

# Mutable objects are not hashable
# list, dict, set cannot be used as keys (TypeError)
```

### 2.2 Java HashMap Internals

```java
// Java 8+ HashMap internal structure
// - Initial capacity: 16
// - Load factor: 0.75
// - Separate chaining
// - Converts to red-black tree when bucket size >= 8 (Treeification)
// - Reverts to linked list when bucket size <= 6

// Flow of HashMap's put method
public V put(K key, V value) {
    // 1. Get the hashCode() of key
    int hash = hash(key);

    // 2. Mix upper bits into lower bits (spread)
    // static int hash(Object key) {
    //     int h;
    //     return (key == null) ? 0 : (h = key.hashCode()) ^ (h >>> 16);
    // }

    // 3. index = hash & (capacity - 1)
    //    Since capacity is a power of 2, bitwise AND provides fast modulo

    // 4. Insert into bucket
    //    - Empty -> new node
    //    - Linked list -> linear search, append to tail
    //    - Red-black tree -> insert into tree

    // 5. Resize if size > threshold (capacity * loadFactor)
    return putVal(hash, key, value, false, true);
}

// Treeification in Java 8+ (conversion to red-black tree)
// Linked list O(n) -> Red-black tree O(log n)
// Countermeasure against Hash DoS attacks

// ConcurrentHashMap (thread-safe version)
// - Java 8+: CAS + synchronized (segment locks removed)
// - Reads are lock-free (volatile reads)
// - Writes use per-bucket synchronized

import java.util.concurrent.ConcurrentHashMap;
ConcurrentHashMap<String, Integer> map = new ConcurrentHashMap<>();
map.put("key", 42);
map.computeIfAbsent("key2", k -> expensiveComputation(k));
```

### 2.3 Go map Internals

```go
// Go's map is implemented in the runtime package
// - Bucket-based (each bucket holds 8 key/value pairs)
// - Incremental resizing
// - Randomized iteration order

package main

import "fmt"

func main() {
    // Basic operations
    m := make(map[string]int)
    m["Alice"] = 100
    m["Bob"] = 85

    // Existence check (two-value return)
    if v, ok := m["Alice"]; ok {
        fmt.Printf("Alice: %d\n", v)
    }

    // Deletion
    delete(m, "Bob")

    // Iteration (order is non-deterministic)
    for k, v := range m {
        fmt.Printf("%s: %d\n", k, v)
    }
}

// Go map internal structure:
// type hmap struct {
//     count     int     // Number of elements
//     flags     uint8
//     B         uint8   // Number of buckets = 2^B
//     noverflow uint16  // Number of overflow buckets
//     hash0     uint32  // Hash seed (randomized)
//     buckets   unsafe.Pointer
//     oldbuckets unsafe.Pointer // Old buckets during resize
//     ...
// }

// Go map is NOT thread-safe
// Use sync.Map for concurrent access
// import "sync"
// var m sync.Map
// m.Store("key", "value")
// v, ok := m.Load("key")
```

### 2.4 Rust HashMap Internals

```rust
use std::collections::HashMap;

fn main() {
    // Rust HashMap is based on the hashbrown crate (Swiss Table)
    // - Uses SipHash-1-3 as the default hash function (Hash DoS resistant)
    // - Leverages SIMD for fast probing

    let mut map = HashMap::new();
    map.insert("Alice", 100);
    map.insert("Bob", 85);

    // Safe access via pattern matching
    match map.get("Alice") {
        Some(&score) => println!("Alice: {}", score),
        None => println!("Not found"),
    }

    // Entry API (insert only if absent)
    map.entry("Charlie").or_insert(90);

    // Entry API (update existing value)
    let count = map.entry("Alice").or_insert(0);
    *count += 10; // Alice: 110

    // Iteration
    for (key, value) in &map {
        println!("{}: {}", key, value);
    }
}

// Using a custom hash function (fast but not DoS-resistant)
// use std::collections::HashMap;
// use std::hash::BuildHasherDefault;
// use ahash::AHasher;
//
// type AHashMap<K, V> = HashMap<K, V, BuildHasherDefault<AHasher>>;
// let mut map: AHashMap<String, i32> = AHashMap::default();
```

---

## 3. Practical Usage

### 3.1 Hash Tables Across Languages

```python
# Python: dict (the most frequently used data structure)
d = {"name": "Alice", "age": 30}
d["email"] = "alice@example.com"  # O(1)
"name" in d  # O(1)

# Python: set (deduplication, set operations)
s = {1, 2, 3}
s.add(4)          # O(1)
2 in s            # O(1)
s & {2, 3, 4}     # Intersection: {2, 3}
s | {4, 5}        # Union: {1, 2, 3, 4, 5}

# Counter (frequency counting)
from collections import Counter
words = ["apple", "banana", "apple", "cherry", "banana", "apple"]
count = Counter(words)
# Counter({'apple': 3, 'banana': 2, 'cherry': 1})
count.most_common(2)  # [('apple', 3), ('banana', 2)]

# defaultdict (dictionary with default values)
from collections import defaultdict
graph = defaultdict(list)
graph["A"].append("B")  # No KeyError
```

### 3.2 Hash Table Design Patterns

```python
# Pattern 1: Memoization (caching)
cache = {}
def expensive_computation(key):
    if key not in cache:
        cache[key] = compute(key)  # Computed only on first call
    return cache[key]

# Memoization using functools.lru_cache
from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Pattern 2: Grouping
from collections import defaultdict
def group_anagrams(words):
    groups = defaultdict(list)
    for word in words:
        key = tuple(sorted(word))
        groups[key].append(word)
    return list(groups.values())

# Usage example
print(group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))
# [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]

# Pattern 3: Two Sum (O(n) with a hash map)
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i

# Pattern 4: Sliding window + hash map
def length_of_longest_substring(s):
    """Length of the longest substring without repeating characters"""
    char_index = {}
    max_len = 0
    start = 0

    for i, c in enumerate(s):
        if c in char_index and char_index[c] >= start:
            start = char_index[c] + 1
        char_index[c] = i
        max_len = max(max_len, i - start + 1)

    return max_len

# Usage examples
print(length_of_longest_substring("abcabcbb"))  # 3 ("abc")
print(length_of_longest_substring("bbbbb"))     # 1 ("b")
print(length_of_longest_substring("pwwkew"))    # 3 ("wke")

# Pattern 5: Frequency map for validation
def is_anagram(s, t):
    """Check whether two strings are anagrams"""
    if len(s) != len(t):
        return False
    count = {}
    for c in s:
        count[c] = count.get(c, 0) + 1
    for c in t:
        count[c] = count.get(c, 0) - 1
        if count[c] < 0:
            return False
    return True

# Pattern 6: Graph representation using hash maps
def build_graph(edges):
    """Build an adjacency list from an edge list"""
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    return graph

edges = [("A", "B"), ("A", "C"), ("B", "C"), ("C", "D")]
graph = build_graph(edges)
# {'A': ['B', 'C'], 'B': ['A', 'C'], 'C': ['A', 'B', 'D'], 'D': ['C']}

# Pattern 7: String pattern matching (bijection verification)
def word_pattern(pattern, s):
    """Verify a bijection between pattern characters and words"""
    words = s.split()
    if len(pattern) != len(words):
        return False

    char_to_word = {}
    word_to_char = {}

    for c, w in zip(pattern, words):
        if c in char_to_word:
            if char_to_word[c] != w:
                return False
        else:
            if w in word_to_char:
                return False
            char_to_word[c] = w
            word_to_char[w] = c

    return True

print(word_pattern("abba", "dog cat cat dog"))  # True
print(word_pattern("abba", "dog cat cat fish")) # False
```

### 3.3 Hash Tables in JavaScript/TypeScript

```typescript
// Map (order-preserving, arbitrary key types)
const map = new Map<string, number>();
map.set("Alice", 100);
map.set("Bob", 85);
console.log(map.get("Alice")); // 100
console.log(map.has("Charlie")); // false
console.log(map.size); // 2

// Object vs Map: when to use which
// Object: keys are strings/Symbols only, prototype chain exists
// Map: arbitrary key types, O(1) size retrieval, guaranteed iteration order

// WeakMap (GC-friendly, keys must be objects)
const weakMap = new WeakMap<object, string>();
let obj = { id: 1 };
weakMap.set(obj, "metadata");
// When there are no more references to obj, the WeakMap entry is also GC'd

// Set
const set = new Set<number>([1, 2, 3, 2, 1]);
console.log(set.size); // 3 (duplicates removed)
set.add(4);
set.delete(2);
console.log([...set]); // [1, 3, 4]

// Practical example: API response cache
class APICache {
    private cache: Map<string, { data: unknown; timestamp: number }>;
    private ttlMs: number;

    constructor(ttlMs: number = 60_000) {
        this.cache = new Map();
        this.ttlMs = ttlMs;
    }

    get(key: string): unknown | null {
        const entry = this.cache.get(key);
        if (!entry) return null;

        if (Date.now() - entry.timestamp > this.ttlMs) {
            this.cache.delete(key);
            return null;
        }

        return entry.data;
    }

    set(key: string, data: unknown): void {
        this.cache.set(key, { data, timestamp: Date.now() });
    }

    clear(): void {
        this.cache.clear();
    }
}
```

---

## 4. LRU Cache Implementation

### 4.1 Implementation Using OrderedDict

```python
from collections import OrderedDict

class LRUCache:
    """Least Recently Used cache"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key not in self.cache:
            return -1
        # Move accessed key to the end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            # Remove the oldest entry (front)
            self.cache.popitem(last=False)

# Usage example
cache = LRUCache(3)
cache.put("a", 1)
cache.put("b", 2)
cache.put("c", 3)
print(cache.get("a"))  # 1 ("a" becomes most recently used)
cache.put("d", 4)      # "b" is evicted (oldest)
print(cache.get("b"))   # -1 (evicted)
```

### 4.2 Implementation with Hash Map + Doubly Linked List

```python
class DLinkedNode:
    """Node for a doubly linked list"""
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCacheManual:
    """LRU cache using a doubly linked list + hash map"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.size = 0
        self.cache = {}  # key -> DLinkedNode

        # Sentinel nodes (simplify implementation)
        self.head = DLinkedNode()  # Dummy head (most recently used side)
        self.tail = DLinkedNode()  # Dummy tail (oldest side)
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._move_to_head(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            self._move_to_head(node)
        else:
            node = DLinkedNode(key, value)
            self.cache[key] = node
            self._add_node(node)
            self.size += 1

            if self.size > self.capacity:
                tail = self._pop_tail()
                del self.cache[tail.key]
                self.size -= 1

    def _add_node(self, node):
        """Add immediately after the head"""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node):
        """Remove a node from the linked list"""
        prev = node.prev
        nxt = node.next
        prev.next = nxt
        nxt.prev = prev

    def _move_to_head(self, node):
        """Move a node to just after the head"""
        self._remove_node(node)
        self._add_node(node)

    def _pop_tail(self):
        """Remove the node just before the tail (oldest)"""
        node = self.tail.prev
        self._remove_node(node)
        return node

# LRU cache time complexity:
# get: O(1) - hash map lookup + linked list operation
# put: O(1) - hash map insertion + linked list operation
# Space: O(capacity)
```

### 4.3 TTL (Time-To-Live) Cache

```python
import time
import threading
from collections import OrderedDict

class TTLCache:
    """Cache with expiration"""

    def __init__(self, capacity: int, ttl_seconds: float):
        self.capacity = capacity
        self.ttl = ttl_seconds
        self.cache = OrderedDict()  # key -> (value, expire_time)
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key not in self.cache:
                return None

            value, expire_time = self.cache[key]

            if time.time() > expire_time:
                # Expired
                del self.cache[key]
                return None

            # LRU: move to most recently used
            self.cache.move_to_end(key)
            return value

    def put(self, key, value):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)

            self.cache[key] = (value, time.time() + self.ttl)

            # Capacity limit
            while len(self.cache) > self.capacity:
                self.cache.popitem(last=False)

    def cleanup(self):
        """Bulk removal of expired entries"""
        with self.lock:
            now = time.time()
            expired_keys = [
                k for k, (_, exp) in self.cache.items()
                if now > exp
            ]
            for k in expired_keys:
                del self.cache[k]

# Usage example
cache = TTLCache(capacity=100, ttl_seconds=300)  # 5-minute cache
cache.put("user:123", {"name": "Alice", "age": 30})
user = cache.get("user:123")  # Retrievable within 5 minutes
```

---

## 5. Consistent Hashing

### 5.1 Basic Concept

```
Consistent Hashing: Data distribution in distributed systems

  Traditional hashing:
    server = hash(key) % num_servers
    -> All data must be redistributed when servers are added/removed
    -> Going from 3 to 4 servers: 75% of data must move

  Consistent hashing:
    Servers and keys are placed on the same hash ring
    -> Only ~1/N of data moves when servers are added/removed

    Hash ring (0 ~ 2^32-1):
                   0
                 /   \
               S1     S3
              /         \
            K1           K4
           |               |
          K2               S2
            \             /
             K3         K5
               \       /
                S4---K6
                 2^32

    K1 -> assigned to first server clockwise: S1
    K4 -> S2
    K5 -> S2
    K6 -> S4

    If S2 goes down -> K4, K5 move to S3 (K1, K2, K3, K6 are unaffected)
```

### 5.2 Implementation with Virtual Nodes

```python
import hashlib
from bisect import bisect_right

class ConsistentHash:
    """Consistent hashing with virtual nodes"""

    def __init__(self, num_replicas=150):
        self.num_replicas = num_replicas  # Number of virtual nodes
        self.ring = {}       # hash -> node
        self.sorted_keys = []  # Sorted hash values

    def _hash(self, key: str) -> int:
        """Achieve uniform distribution using MD5 hash"""
        digest = hashlib.md5(key.encode()).hexdigest()
        return int(digest, 16)

    def add_node(self, node: str):
        """Add a node (server)"""
        for i in range(self.num_replicas):
            virtual_key = f"{node}:{i}"
            h = self._hash(virtual_key)
            self.ring[h] = node
            self.sorted_keys.append(h)
        self.sorted_keys.sort()

    def remove_node(self, node: str):
        """Remove a node"""
        for i in range(self.num_replicas):
            virtual_key = f"{node}:{i}"
            h = self._hash(virtual_key)
            del self.ring[h]
            self.sorted_keys.remove(h)

    def get_node(self, key: str) -> str:
        """Get the node assigned to a key"""
        if not self.ring:
            return None

        h = self._hash(key)
        idx = bisect_right(self.sorted_keys, h)

        # Wrap around to the beginning if past the end of the ring
        if idx == len(self.sorted_keys):
            idx = 0

        return self.ring[self.sorted_keys[idx]]

    def get_nodes(self, key: str, n: int = 3) -> list:
        """For replicas: get n distinct nodes for a key"""
        if not self.ring or n > len(set(self.ring.values())):
            return list(set(self.ring.values()))

        h = self._hash(key)
        idx = bisect_right(self.sorted_keys, h)

        result = []
        seen = set()

        while len(result) < n:
            if idx == len(self.sorted_keys):
                idx = 0
            node = self.ring[self.sorted_keys[idx]]
            if node not in seen:
                result.append(node)
                seen.add(node)
            idx += 1

        return result


# Usage example: distributing cache servers
ch = ConsistentHash(num_replicas=150)
ch.add_node("cache-server-1")
ch.add_node("cache-server-2")
ch.add_node("cache-server-3")

# Key assignment
print(ch.get_node("user:1001"))    # "cache-server-2"
print(ch.get_node("session:abc"))  # "cache-server-1"

# Adding a server (minimal impact)
ch.add_node("cache-server-4")
print(ch.get_node("user:1001"))    # Most keys remain on the same server

# Retrieving replicas
print(ch.get_nodes("user:1001", n=2))  # 2 different servers
```

### 5.3 Real-World Applications of Consistent Hashing

```
Real-world applications of consistent hashing:

  1. Amazon DynamoDB:
     - Distributes keys across partitions
     - Virtual nodes mitigate data skew
     - Preference lists determine replica destinations

  2. Apache Cassandra:
     - Determines nodes via hash of partition key
     - Murmur3Partitioner is the default
     - vnodes (virtual nodes) for load balancing

  3. Memcached / Redis Cluster:
     - Consistent hashing implemented on the client side
     - Minimizes cache misses when servers are added/removed

  4. CDN (Akamai, etc.):
     - Hashes request URLs to assign edge servers
     - Localizes the impact of server failures

  5. Load Balancers:
     - Session affinity (same user to same server)
     - Nginx upstream hash module
```

---

## 6. Hash DoS Attacks and Countermeasures

### 6.1 Attack Principle

```
Hash DoS (Hash Collision Attack):

  Principle:
  - When hash function output is predictable
  - Attacker generates a large number of keys that collide into the same bucket
  - Hash table operations degrade from O(1) to O(n)
  - n keys cause O(n^2) processing time

  History:
  - 2003: Hash collision attack on Perl reported
  - 2011: Attacks on PHP, Python, Ruby, Java, etc. presented at
          CCC (Chaos Communication Congress)
  - Sending JSON/XML requests with many colliding keys
  - A small request can consume 100% of a server's CPU

  Concrete example (Python 3.2 and earlier):
  # The following keys are crafted to have the same hash value
  # {"key1": 1, "key2": 2, ..., "key100000": 100000}
  # -> Insertion takes O(n^2) time, causing the server to hang
```

### 6.2 Countermeasures

```python
# Countermeasure 1: Randomized hashing (Python 3.3+)
# PYTHONHASHSEED environment variable controls the seed
# Default is a random seed at process startup
import sys
print(sys.hash_info)
# sys.hash_info(width=64, modulus=2305843009213693951,
#               inf=314159, nan=0, imag=1000003,
#               algorithm='siphash24', ...)

# Countermeasure 2: SipHash (cryptographic PRF)
# Standard in Python 3.4+, Rust, Perl 5.18+
# Uses a 128-bit secret key
# Secure against Hash DoS while remaining sufficiently fast

# Countermeasure 3: Fallback to red-black trees (Java 8+)
# When a HashMap bucket exceeds 8 elements,
# it converts from linked list to red-black tree
# Worst case improves from O(n) to O(log n)

# Countermeasure 4: Request throttling
# Application-level countermeasures
# - Limit request size
# - Set maximum number of JSON parameters
# - Rate limiting

# Countermeasure 5: Using custom hash functions
import hmac
import hashlib

def secure_hash(key: str, secret: bytes) -> int:
    """HMAC-based secure hash"""
    h = hmac.new(secret, key.encode(), hashlib.sha256)
    return int.from_bytes(h.digest()[:8], 'big')
```

---

## 7. Hash Table Derived Data Structures

### 7.1 Counting Bloom Filter

```python
class CountingBloomFilter:
    """Bloom Filter with deletion support"""

    def __init__(self, size, num_hashes):
        self.size = size
        self.num_hashes = num_hashes
        self.counters = [0] * size  # Counters instead of bits

    def _hashes(self, item):
        """Generate k hash values"""
        import hashlib
        indices = []
        for i in range(self.num_hashes):
            h = hashlib.sha256(f"{item}:{i}".encode()).hexdigest()
            indices.append(int(h, 16) % self.size)
        return indices

    def add(self, item):
        for idx in self._hashes(item):
            self.counters[idx] += 1

    def remove(self, item):
        """Deletion operation impossible with standard Bloom Filters"""
        indices = self._hashes(item)
        if all(self.counters[idx] > 0 for idx in indices):
            for idx in indices:
                self.counters[idx] -= 1

    def contains(self, item):
        return all(self.counters[idx] > 0 for idx in self._hashes(item))
```

### 7.2 Cuckoo Hashing

```python
class CuckooHashTable:
    """Cuckoo hashing: worst-case O(1) lookup with two hash functions"""

    MAX_KICKS = 500  # Maximum number of displacements

    def __init__(self, capacity=16):
        self.capacity = capacity
        self.table1 = [None] * capacity
        self.table2 = [None] * capacity
        self.size = 0

    def _hash1(self, key):
        return hash(key) % self.capacity

    def _hash2(self, key):
        return hash(key * 2654435761) % self.capacity  # Different hash

    def get(self, key):
        """O(1) worst-case lookup"""
        pos1 = self._hash1(key)
        if self.table1[pos1] and self.table1[pos1][0] == key:
            return self.table1[pos1][1]

        pos2 = self._hash2(key)
        if self.table2[pos2] and self.table2[pos2][0] == key:
            return self.table2[pos2][1]

        return None  # Not found

    def put(self, key, value):
        """Insert (using cuckoo-style displacement)"""
        # Check for existing key update
        pos1 = self._hash1(key)
        if self.table1[pos1] and self.table1[pos1][0] == key:
            self.table1[pos1] = (key, value)
            return

        pos2 = self._hash2(key)
        if self.table2[pos2] and self.table2[pos2][0] == key:
            self.table2[pos2] = (key, value)
            return

        # New insertion
        current = (key, value)
        for _ in range(self.MAX_KICKS):
            # Try inserting into table1
            pos = self._hash1(current[0])
            if self.table1[pos] is None:
                self.table1[pos] = current
                self.size += 1
                return

            # Displace
            current, self.table1[pos] = self.table1[pos], current

            # Try inserting into table2
            pos = self._hash2(current[0])
            if self.table2[pos] is None:
                self.table2[pos] = current
                self.size += 1
                return

            # Displace
            current, self.table2[pos] = self.table2[pos], current

        # Displacement loop -> resize needed
        self._resize()
        self.put(current[0], current[1])

    def _resize(self):
        old_items = []
        for entry in self.table1:
            if entry:
                old_items.append(entry)
        for entry in self.table2:
            if entry:
                old_items.append(entry)

        self.capacity *= 2
        self.table1 = [None] * self.capacity
        self.table2 = [None] * self.capacity
        self.size = 0

        for key, value in old_items:
            self.put(key, value)


# Advantages of Cuckoo hashing:
# - Worst-case O(1) lookup (only 2 locations to check)
# - Foundation for Cuckoo Filter (alternative to Bloom Filter)
# - Expected O(1) insertion (expected constant number of displacements)
```

### 7.3 Swiss Table (Google's High-Performance Hash Table)

```
How Swiss Table Works:

  Adopted by Rust (hashbrown), C++ (Abseil), Go map

  Structure:
  +----------------+
  | Control Bytes  |  16-byte groups (compared in bulk using SIMD)
  | [H2|H2|..|H2] |  H2 = upper 7 bits of hash
  +----------------+
  |     Slots      |  Actual key/value pairs
  | [KV|KV|..|KV]  |
  +----------------+

  Search flow:
  1. H1 = lower bits of hash -> group index
  2. H2 = upper 7 bits of hash
  3. SIMD instruction compares 16 control bytes at once
  4. Only matching slots require key comparison

  Advantages:
  - SIMD compares 16 slots simultaneously
  - High cache line efficiency
  - 2-3x faster than traditional open addressing
  - Memory overhead is only 1 byte/entry
```

---

## 8. Hash Table Benchmarks and Performance Comparison

### 8.1 Performance Measurement in Python

```python
import time
import random
import string

def benchmark_dict_operations(n):
    """Benchmark of dict operations"""

    # Generate random keys
    keys = [''.join(random.choices(string.ascii_lowercase, k=10)) for _ in range(n)]

    # Insertion
    d = {}
    start = time.perf_counter()
    for k in keys:
        d[k] = random.randint(0, 1000000)
    insert_time = time.perf_counter() - start

    # Search (existing keys)
    start = time.perf_counter()
    for k in keys:
        _ = d[k]
    search_hit_time = time.perf_counter() - start

    # Search (non-existing keys)
    missing = [k + "x" for k in keys]
    start = time.perf_counter()
    for k in missing:
        _ = d.get(k)
    search_miss_time = time.perf_counter() - start

    # Deletion
    start = time.perf_counter()
    for k in keys:
        del d[k]
    delete_time = time.perf_counter() - start

    print(f"n={n:>10,}")
    print(f"  Insert:       {insert_time:.4f}s ({insert_time/n*1e6:.2f}us/op)")
    print(f"  Search(hit):  {search_hit_time:.4f}s ({search_hit_time/n*1e6:.2f}us/op)")
    print(f"  Search(miss): {search_miss_time:.4f}s ({search_miss_time/n*1e6:.2f}us/op)")
    print(f"  Delete:       {delete_time:.4f}s ({delete_time/n*1e6:.2f}us/op)")

# Typical results:
# n=    10,000
#   Insert:       0.0034s (0.34us/op)
#   Search(hit):  0.0012s (0.12us/op)
#   Search(miss): 0.0014s (0.14us/op)
#   Delete:       0.0010s (0.10us/op)
# n= 1,000,000
#   Insert:       0.4521s (0.45us/op)
#   Search(hit):  0.1523s (0.15us/op)
#   Search(miss): 0.1842s (0.18us/op)
#   Delete:       0.1234s (0.12us/op)
```

### 8.2 Performance Comparison Across Data Structures

```python
import time
from sortedcontainers import SortedDict

def compare_dict_vs_sorted(n):
    """Search performance comparison: dict vs SortedDict vs list"""

    import random
    data = [(random.randint(0, n*10), random.randint(0, 1000)) for _ in range(n)]
    search_keys = [random.randint(0, n*10) for _ in range(10000)]

    # dict
    d = dict(data)
    start = time.perf_counter()
    for k in search_keys:
        _ = k in d
    dict_time = time.perf_counter() - start

    # SortedDict (equivalent to a balanced BST)
    sd = SortedDict(data)
    start = time.perf_counter()
    for k in search_keys:
        _ = k in sd
    sorted_time = time.perf_counter() - start

    # list (linear search)
    lst = data
    start = time.perf_counter()
    for k in search_keys:
        _ = any(kk == k for kk, _ in lst)
    list_time = time.perf_counter() - start

    print(f"n={n:>10,} (10000 searches)")
    print(f"  dict:       {dict_time:.4f}s")
    print(f"  SortedDict: {sorted_time:.4f}s")
    print(f"  list:       {list_time:.4f}s")
    print(f"  dict/sorted: {sorted_time/dict_time:.1f}x slower")
    print(f"  dict/list:   {list_time/dict_time:.1f}x slower")

# Typical results:
# n=   100,000 (10000 searches)
#   dict:       0.0009s
#   SortedDict: 0.0213s
#   list:       12.4523s
#   dict/sorted: 23.7x slower
#   dict/list:   13836.x slower
```

---

## 9. Common Practical Patterns

### 9.1 Frequency Counting and Statistics

```python
from collections import Counter, defaultdict

# Pattern: Log analysis
def analyze_access_log(log_entries):
    """Statistical analysis of access logs"""

    # Access count by endpoint
    endpoint_count = Counter(entry["path"] for entry in log_entries)

    # Count by status code
    status_count = Counter(entry["status"] for entry in log_entries)

    # Access count by hour
    hourly_count = Counter(entry["timestamp"].hour for entry in log_entries)

    # Access count by IP address (top 10)
    ip_count = Counter(entry["ip"] for entry in log_entries)

    return {
        "top_endpoints": endpoint_count.most_common(10),
        "status_distribution": dict(status_count),
        "peak_hour": hourly_count.most_common(1)[0],
        "top_ips": ip_count.most_common(10),
        "total_requests": len(log_entries),
        "unique_ips": len(ip_count),
        "error_rate": status_count.get(500, 0) / len(log_entries)
    }


# Pattern: Permission management using set operations
def check_permissions(user_roles: set, required_roles: set) -> bool:
    """Check user permissions"""
    return required_roles.issubset(user_roles)

def common_permissions(user_a_roles: set, user_b_roles: set) -> set:
    """Permissions common to two users"""
    return user_a_roles & user_b_roles

def exclusive_permissions(user_a_roles: set, user_b_roles: set) -> set:
    """Permissions held only by user A"""
    return user_a_roles - user_b_roles

# Pattern: Building an inverted index
def build_inverted_index(documents):
    """Build an inverted index (foundation for full-text search)"""
    index = defaultdict(set)

    for doc_id, text in enumerate(documents):
        words = text.lower().split()
        for word in words:
            # Normalization (remove punctuation, etc.)
            word = word.strip(".,!?;:")
            index[word].add(doc_id)

    return index

def search(index, query):
    """AND search"""
    words = query.lower().split()
    if not words:
        return set()

    result = index.get(words[0], set())
    for word in words[1:]:
        result &= index.get(word, set())

    return result

# Usage example
docs = [
    "Python is a great programming language",
    "Java is also a programming language",
    "Python and Java are both popular",
    "Rust is a systems programming language"
]
idx = build_inverted_index(docs)
print(search(idx, "programming language"))  # {0, 1, 3}
print(search(idx, "Python"))                # {0, 2}
```

### 9.2 Data Validation and Transformation

```python
# Pattern: Schema validation
def validate_schema(data: dict, schema: dict) -> list:
    """Simple schema validation"""
    errors = []

    for field, rules in schema.items():
        # Required check
        if rules.get("required") and field not in data:
            errors.append(f"Missing required field: {field}")
            continue

        if field not in data:
            continue

        value = data[field]

        # Type check
        expected_type = rules.get("type")
        if expected_type and not isinstance(value, expected_type):
            errors.append(f"Invalid type for {field}: expected {expected_type.__name__}")

        # Range check
        if "min" in rules and value < rules["min"]:
            errors.append(f"{field} must be >= {rules['min']}")
        if "max" in rules and value > rules["max"]:
            errors.append(f"{field} must be <= {rules['max']}")

        # Allowed values check
        if "choices" in rules and value not in rules["choices"]:
            errors.append(f"{field} must be one of {rules['choices']}")

    return errors

# Usage example
schema = {
    "name": {"required": True, "type": str},
    "age": {"required": True, "type": int, "min": 0, "max": 150},
    "role": {"type": str, "choices": ["admin", "user", "guest"]},
}

errors = validate_schema(
    {"name": "Alice", "age": 30, "role": "admin"},
    schema
)
print(errors)  # []

errors = validate_schema(
    {"age": -5, "role": "superuser"},
    schema
)
print(errors)
# ['Missing required field: name', 'age must be >= 0',
#  'role must be one of ["admin", "user", "guest"]']


# Pattern: Data mapping and transformation
def transform_records(records, field_mapping):
    """Field name mapping and transformation"""
    transformed = []
    for record in records:
        new_record = {}
        for old_key, (new_key, converter) in field_mapping.items():
            if old_key in record:
                new_record[new_key] = converter(record[old_key])
        transformed.append(new_record)
    return transformed

# Usage example: CSV data transformation
mapping = {
    "full_name": ("name", str.strip),
    "birth_year": ("age", lambda y: 2026 - int(y)),
    "salary_str": ("salary", lambda s: float(s.replace(",", ""))),
}

raw_data = [
    {"full_name": " Alice ", "birth_year": "1990", "salary_str": "85,000.00"},
    {"full_name": " Bob ", "birth_year": "1985", "salary_str": "92,500.50"},
]

print(transform_records(raw_data, mapping))
# [{'name': 'Alice', 'age': 36, 'salary': 85000.0},
#  {'name': 'Bob', 'age': 41, 'salary': 92500.5}]
```

### 9.3 Configuration Management and Environment Variables

```python
import os
from typing import Any, Optional

class Config:
    """Hash table-based configuration management"""

    def __init__(self, defaults: dict = None):
        self._config = {}
        self._defaults = defaults or {}

    def load_from_env(self, prefix: str = "APP_"):
        """Load configuration from environment variables"""
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                self._config[config_key] = value

    def load_from_dict(self, data: dict):
        """Load configuration from a dictionary (flatten nested structures)"""
        def flatten(d, parent_key=""):
            for k, v in d.items():
                new_key = f"{parent_key}.{k}" if parent_key else k
                if isinstance(v, dict):
                    flatten(v, new_key)
                else:
                    self._config[new_key] = v
        flatten(data)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value (priority: config > defaults > argument)"""
        if key in self._config:
            return self._config[key]
        if key in self._defaults:
            return self._defaults[key]
        return default

    def get_int(self, key: str, default: int = 0) -> int:
        return int(self.get(key, default))

    def get_bool(self, key: str, default: bool = False) -> bool:
        val = self.get(key, default)
        if isinstance(val, str):
            return val.lower() in ("true", "1", "yes")
        return bool(val)

# Usage example
config = Config(defaults={
    "database.host": "localhost",
    "database.port": 5432,
    "debug": False,
})
config.load_from_dict({
    "database": {"host": "db.example.com", "name": "myapp"},
    "cache": {"ttl": 300},
})
print(config.get("database.host"))  # "db.example.com" (overridden)
print(config.get("database.port"))  # 5432 (default value)
print(config.get("cache.ttl"))      # 300
```

---

## 10. Exercises

### Exercise 1: Basic Operations (Fundamentals)
Implement a hash table from scratch (using chaining with resizing). It must satisfy the following requirements:
- `put(key, value)`, `get(key)`, `remove(key)`, `contains(key)` operations
- Resize at load factor 0.75
- Implement `__len__`, `__iter__`, `__repr__`
- Collision counting functionality

### Exercise 2: LRU Cache (Intermediate)
Implement an LRU cache using OrderedDict or dict + doubly linked list. It must satisfy the following requirements:
- Both `get(key)` and `put(key, value)` operations run in O(1)
- Automatically evict the oldest entry when capacity is exceeded
- TTL (time-to-live) support
- Thread safety (using threading.Lock)

### Exercise 3: Consistent Hashing (Advanced)
Implement consistent hashing as used in distributed systems. It must satisfy the following requirements:
- Load distribution via virtual nodes
- Node addition and removal
- Replica retrieval (returning N distinct nodes)
- Benchmark testing load distribution uniformity

### Exercise 4: Inverted Index (Intermediate)
Implement an inverted index as the foundation for a full-text search engine:
- Document addition and search (AND search, OR search)
- TF-IDF scoring
- Prefix search support

### Exercise 5: Cuckoo Filter (Advanced)
Implement a Cuckoo Filter as an improvement over Bloom Filters:
- Support insert, lookup, and delete operations
- Configurable false positive rate
- Performance comparison benchmark against Bloom Filters

---


## FAQ

### Q1: What is the most important point when learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory, but by actually writing code and verifying its behavior.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this used in practice?

Knowledge of this topic is frequently applied in day-to-day development work. It becomes especially important during code reviews and architecture design.

---

## Summary

| Concept | Key Points |
|---------|-----------|
| Hash Functions | Key -> integer. Uniform distribution is ideal. SipHash/MurmurHash3 are practical |
| Collision Resolution | Chaining or open addressing. Robin Hood/Cuckoo as well |
| Time Complexity | Expected O(1), worst O(n). Amortized O(1) for resizing |
| Practical Usage | dict/set/Map/Counter/defaultdict |
| Internal Implementations | Python: compact dict, Java: red-black tree fallback, Rust: Swiss Table |
| Security | SipHash for Hash DoS protection. Request throttling is also important |
| Distributed Systems | Consistent hashing for load distribution. Virtual nodes for uniformity |
| Ordering | Python 3.7+ guarantees insertion order. Others do not |

---

## Recommended Next Guides

---

## References
1. Cormen, T. H. "Introduction to Algorithms." Chapter 11: Hash Tables.
2. Sedgewick, R. "Algorithms." Chapter 3.4: Hash Tables.
3. Kleppmann, M. "Designing Data-Intensive Applications." Chapter 6: Partitioning.
4. Karger, D. et al. "Consistent Hashing and Random Trees." STOC 1997.
5. Aumasson, J-P., Bernstein, D. J. "SipHash: a fast short-input PRF." 2012.
6. Abseil Team. "Swiss Tables Design Notes." Google, 2017.
7. Pagh, R., Rodler, F. F. "Cuckoo Hashing." ESA 2001.
8. Fan, B. et al. "Cuckoo Filter: Practically Better Than Bloom." CoNEXT 2014.
