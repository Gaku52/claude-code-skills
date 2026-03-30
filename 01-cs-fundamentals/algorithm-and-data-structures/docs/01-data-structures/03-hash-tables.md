# Hash Tables — Hash Functions, Collision Resolution, and Load Factor

> Learn the internal structure, collision resolution strategies, and performance tuning of hash tables that achieve O(1) average key lookup.

---

## Learning Objectives

1. **Hash function** design principles and characteristics of good functions
2. **Collision resolution** — Separate chaining and open addressing
3. **Load factor** and rehashing mechanisms
4. **Language-specific implementations** — Internal structures of Python dict, Java HashMap, C++ unordered_map
5. **Practical applications** — Caching, uniqueness checks, set operations


## Prerequisites

Before reading this guide, the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Familiarity with the content of [Stacks and Queues — A Complete Guide to Implementation, Applications, and Priority Queues](./02-stacks-queues.md)

---

## 1. Basic Structure of Hash Tables

A hash table is a data structure that maps "keys" to "values" with O(1) average complexity. Internally, it maintains a bucket array (slot array) and computes the bucket index from the key using a hash function.

```
Key "apple" -> Hash Function -> Index 3

  Bucket Array:
  [0] -> null
  [1] -> ("banana", 2)
  [2] -> null
  [3] -> ("apple", 5)     <- h("apple") = 3
  [4] -> null
  [5] -> ("cherry", 8)
  [6] -> null
  [7] -> ("date", 1)

  Search "apple":
  1. h("apple") = 3
  2. Access bucket[3]
  3. Key matches -> return value 5
  -> O(1) average
```

### 1.1 Complexity of Basic Operations

| Operation | Average | Worst | Notes |
|-----------|---------|-------|-------|
| Search (get) | O(1) | O(n) | When all keys collide in the same bucket |
| Insert (put) | O(1) | O(n) | O(n) bulk cost when rehashing occurs |
| Delete (delete) | O(1) | O(n) | DELETED marker needed for open addressing |
| Key enumeration | O(n + m) | O(n + m) | m = number of buckets |

### 1.2 Typical Use Cases for Hash Tables

```python
# 1. Frequency counting
from collections import Counter
words = ["apple", "banana", "apple", "cherry", "banana", "apple"]
freq = Counter(words)
print(freq)  # Counter({'apple': 3, 'banana': 2, 'cherry': 1})

# 2. Duplicate check (completes in O(n))
def has_duplicate(arr):
    seen = set()
    for x in arr:
        if x in seen:
            return True
        seen.add(x)
    return False

# 3. Two Sum — O(n)
def two_sum(nums, target):
    lookup = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in lookup:
            return [lookup[complement], i]
        lookup[num] = i
    return []

# 4. Grouping
from collections import defaultdict
def group_anagrams(strs):
    groups = defaultdict(list)
    for s in strs:
        key = tuple(sorted(s))
        groups[key].append(s)
    return list(groups.values())

print(group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))
# [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]
```

---

## 2. Hash Functions

### 2.1 Characteristics of a Good Hash Function

```python
# Requirements:
# 1. Deterministic: same input -> same output
# 2. Uniform distribution: output spreads evenly across all buckets
# 3. Fast: computation is approximately O(key length)
# 4. Avalanche effect: small input changes produce large output changes

# Example string hash function (polynomial hash)
def polynomial_hash(key, table_size, base=31):
    """Polynomial hash — O(len(key))"""
    h = 0
    for char in key:
        h = (h * base + ord(char)) % table_size
    return h

# Python's built-in hash()
print(hash("hello"))    # Returns an integer
print(hash(42))         # Integers are (roughly) identity-mapped
print(hash((1, 2, 3)))  # Tuples are hashable
# hash([1, 2, 3])       # Lists are not hashable (mutable)
```

### 2.2 Implementations of Various Hash Functions

```python
# === Division Method ===
def hash_division(key, table_size):
    """Division method: h(k) = k mod m
    Choose table size m as a prime number not close to a power of 2
    """
    return key % table_size

# === Multiplication Method ===
def hash_multiplication(key, table_size):
    """Multiplication method: h(k) = floor(m * (k * A mod 1))
    A = (sqrt(5) - 1) / 2 ~ 0.6180339887 is recommended
    No constraint on table size m
    """
    import math
    A = (math.sqrt(5) - 1) / 2  # Reciprocal of the golden ratio
    return int(table_size * ((key * A) % 1))

# === FNV-1a Hash ===
def fnv1a_hash(data, table_size):
    """FNV-1a: Widely used for string hashing
    Fast with excellent uniform distribution
    """
    FNV_OFFSET = 2166136261
    FNV_PRIME = 16777619
    h = FNV_OFFSET
    for byte in data.encode('utf-8'):
        h ^= byte
        h = (h * FNV_PRIME) & 0xFFFFFFFF  # Limit to 32 bits
    return h % table_size

# === MurmurHash3 Simplified ===
def murmur3_32(key, seed=0):
    """Simplified 32-bit implementation of MurmurHash3
    Most widely used non-cryptographic hash function
    """
    c1 = 0xcc9e2d51
    c2 = 0x1b873593

    data = key.encode('utf-8') if isinstance(key, str) else key
    length = len(data)
    h = seed

    # Body: process 4 bytes at a time
    nblocks = length // 4
    for i in range(nblocks):
        k = int.from_bytes(data[i*4:(i+1)*4], 'little')
        k = (k * c1) & 0xFFFFFFFF
        k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
        k = (k * c2) & 0xFFFFFFFF
        h ^= k
        h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
        h = (h * 5 + 0xe6546b64) & 0xFFFFFFFF

    # Tail: remaining bytes
    tail = data[nblocks * 4:]
    k = 0
    for i in range(len(tail) - 1, -1, -1):
        k = (k << 8) | tail[i]
    if k:
        k = (k * c1) & 0xFFFFFFFF
        k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
        k = (k * c2) & 0xFFFFFFFF
        h ^= k

    # Finalization
    h ^= length
    h ^= (h >> 16)
    h = (h * 0x85ebca6b) & 0xFFFFFFFF
    h ^= (h >> 13)
    h = (h * 0xc2b2ae35) & 0xFFFFFFFF
    h ^= (h >> 16)

    return h

# Test
print(polynomial_hash("hello", 16))      # Polynomial hash
print(hash_division(42, 17))              # Division method
print(hash_multiplication(42, 16))        # Multiplication method
print(fnv1a_hash("hello", 16))           # FNV-1a
print(murmur3_32("hello"))               # MurmurHash3
```

### 2.3 Cryptographic vs Non-cryptographic Hashes

```python
import hashlib

# === Cryptographic Hashes (for security purposes) ===
# SHA-256: Password storage, data integrity verification
data = "hello world"
sha256_hash = hashlib.sha256(data.encode()).hexdigest()
print(f"SHA-256: {sha256_hash}")

# MD5: Checksums (not recommended for security)
md5_hash = hashlib.md5(data.encode()).hexdigest()
print(f"MD5: {md5_hash}")

# === Comparison of Properties ===
# | Property          | Cryptographic Hash       | Non-cryptographic Hash |
# |-------------------|--------------------------|------------------------|
# | Speed             | Slow (intentionally)     | Fast                   |
# | Collision resist. | High (computationally hard)| Low (sufficient but no guarantee)|
# | Use case          | Security, signatures     | Hash tables            |
# | Examples          | SHA-256, bcrypt          | MurmurHash, FNV        |
# | Preimage resist.  | Yes                      | Not required           |
```

### 2.4 Universal Hashing

```python
import random

class UniversalHashFamily:
    """Universal Hash Family
    For any two distinct keys x, y:
    Pr[h(x) = h(y)] <= 1/m  (m = table size)

    Effective as a countermeasure against hash DoS attacks
    """
    def __init__(self, table_size, prime=2147483647):
        self.m = table_size
        self.p = prime  # Prime number larger than table size
        self.a = random.randint(1, self.p - 1)
        self.b = random.randint(0, self.p - 1)

    def hash(self, key):
        """h(k) = ((a*k + b) mod p) mod m"""
        return ((self.a * key + self.b) % self.p) % self.m

    def regenerate(self):
        """Randomly select a new hash function"""
        self.a = random.randint(1, self.p - 1)
        self.b = random.randint(0, self.p - 1)

# Usage
uhf = UniversalHashFamily(16)
print(uhf.hash(42))
print(uhf.hash(100))
uhf.regenerate()  # Switch to a different hash function
print(uhf.hash(42))  # Highly likely to produce a different result
```

---

## 3. Collision Resolution

### 3.1 Separate Chaining

```
Bucket Array + Linked Lists:

  [0] -> null
  [1] -> ("banana",2) -> ("fig",7) -> null
  [2] -> null
  [3] -> ("apple",5) -> ("grape",3) -> null
  [4] -> null

  h("banana") = h("fig") = 1  -> Stored via chaining
```

```python
class HashTableChaining:
    """Hash table using separate chaining

    Each bucket holds a linked list (Python list in this implementation),
    storing collided keys within the same bucket.
    """
    def __init__(self, size=16):
        self.size = size
        self.buckets = [[] for _ in range(size)]
        self.count = 0

    def _hash(self, key):
        return hash(key) % self.size

    def put(self, key, value):
        """O(1) average"""
        idx = self._hash(key)
        for i, (k, v) in enumerate(self.buckets[idx]):
            if k == key:
                self.buckets[idx][i] = (key, value)
                return
        self.buckets[idx].append((key, value))
        self.count += 1
        if self.count / self.size > 0.75:
            self._rehash()

    def get(self, key):
        """O(1) average"""
        idx = self._hash(key)
        for k, v in self.buckets[idx]:
            if k == key:
                return v
        raise KeyError(key)

    def delete(self, key):
        """O(1) average"""
        idx = self._hash(key)
        for i, (k, v) in enumerate(self.buckets[idx]):
            if k == key:
                self.buckets[idx].pop(i)
                self.count -= 1
                return v
        raise KeyError(key)

    def contains(self, key):
        """Key existence check — O(1) average"""
        idx = self._hash(key)
        return any(k == key for k, _ in self.buckets[idx])

    def keys(self):
        """Enumerate all keys — O(n + m)"""
        result = []
        for bucket in self.buckets:
            for k, v in bucket:
                result.append(k)
        return result

    def items(self):
        """Enumerate all key-value pairs — O(n + m)"""
        result = []
        for bucket in self.buckets:
            for k, v in bucket:
                result.append((k, v))
        return result

    def _rehash(self):
        """Double the table size and relocate all elements"""
        old = self.buckets
        self.size *= 2
        self.buckets = [[] for _ in range(self.size)]
        self.count = 0
        for bucket in old:
            for key, value in bucket:
                self.put(key, value)

    def load_factor(self):
        """Return the current load factor"""
        return self.count / self.size

    def __repr__(self):
        items = []
        for bucket in self.buckets:
            for k, v in bucket:
                items.append(f"{k!r}: {v!r}")
        return "{" + ", ".join(items) + "}"

# Usage
ht = HashTableChaining()
ht.put("name", "Alice")
ht.put("age", 30)
ht.put("city", "Tokyo")
print(ht.get("name"))     # "Alice"
print(ht.contains("age")) # True
print(ht.keys())          # ["name", "age", "city"]
ht.delete("age")
print(ht.contains("age")) # False
print(ht.load_factor())   # 0.125
```

### 3.2 Improved Chaining: Red-Black Tree Chaining (Java 8+ HashMap)

```python
class HashTableTreeChaining:
    """Mimics the Java 8+ HashMap strategy:
    - When bucket elements are few (< 8): linked list
    - When bucket elements are many (>= 8): convert to a balanced tree (red-black tree)

    Worst case improves from O(n) to O(log n)
    """
    TREEIFY_THRESHOLD = 8
    UNTREEIFY_THRESHOLD = 6

    def __init__(self, size=16):
        self.size = size
        self.buckets = [[] for _ in range(size)]
        self.count = 0

    def _hash(self, key):
        # Mix upper bits into lower bits, similar to Java
        h = hash(key)
        return ((h >> 16) ^ h) % self.size

    def put(self, key, value):
        idx = self._hash(key)
        bucket = self.buckets[idx]

        # Update existing key
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return

        bucket.append((key, value))
        self.count += 1

        # Convert to tree when bucket exceeds threshold (simplified: use sorted list)
        if len(bucket) >= self.TREEIFY_THRESHOLD:
            self.buckets[idx] = sorted(bucket, key=lambda x: hash(x[0]))

        if self.count / self.size > 0.75:
            self._rehash()

    def get(self, key):
        idx = self._hash(key)
        bucket = self.buckets[idx]

        if len(bucket) >= self.TREEIFY_THRESHOLD:
            # Treeified bucket: binary search (O(log n))
            return self._tree_search(bucket, key)

        # List bucket: linear search
        for k, v in bucket:
            if k == key:
                return v
        raise KeyError(key)

    def _tree_search(self, sorted_bucket, key):
        """Binary search on a sorted bucket"""
        target_hash = hash(key)
        lo, hi = 0, len(sorted_bucket) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            mid_hash = hash(sorted_bucket[mid][0])
            if mid_hash == target_hash and sorted_bucket[mid][0] == key:
                return sorted_bucket[mid][1]
            elif mid_hash < target_hash:
                lo = mid + 1
            else:
                hi = mid - 1
        raise KeyError(key)

    def _rehash(self):
        old = self.buckets
        self.size *= 2
        self.buckets = [[] for _ in range(self.size)]
        self.count = 0
        for bucket in old:
            for key, value in bucket:
                self.put(key, value)
```

### 3.3 Open Addressing (Linear Probing)

```
h("apple") = 3, h("grape") = 3 -> Collision!

Linear Probing: Find the next empty slot
  [0] -> null
  [1] -> null
  [2] -> null
  [3] -> ("apple", 5)   <- h("apple") = 3
  [4] -> ("grape", 3)   <- h("grape") = 3 -> collision -> 3+1=4
  [5] -> null
```

```python
class HashTableLinearProbing:
    """Open-address hash table using Linear Probing

    Probe sequence: h(k), h(k)+1, h(k)+2, ...
    Advantage: Good cache efficiency (scans contiguous memory regions)
    Disadvantage: Primary clustering tends to occur
    """
    DELETED = object()

    def __init__(self, size=16):
        self.size = size
        self.keys = [None] * size
        self.values = [None] * size
        self.count = 0

    def _hash(self, key):
        return hash(key) % self.size

    def put(self, key, value):
        if self.count / self.size > 0.5:
            self._rehash()
        idx = self._hash(key)
        while self.keys[idx] is not None and self.keys[idx] is not self.DELETED:
            if self.keys[idx] == key:
                self.values[idx] = value
                return
            idx = (idx + 1) % self.size
        self.keys[idx] = key
        self.values[idx] = value
        self.count += 1

    def get(self, key):
        idx = self._hash(key)
        while self.keys[idx] is not None:
            if self.keys[idx] == key:
                return self.values[idx]
            idx = (idx + 1) % self.size
        raise KeyError(key)

    def delete(self, key):
        idx = self._hash(key)
        while self.keys[idx] is not None:
            if self.keys[idx] == key:
                self.keys[idx] = self.DELETED
                self.values[idx] = None
                self.count -= 1
                return
            idx = (idx + 1) % self.size
        raise KeyError(key)

    def _rehash(self):
        old_keys, old_values = self.keys, self.values
        self.size *= 2
        self.keys = [None] * self.size
        self.values = [None] * self.size
        self.count = 0
        for k, v in zip(old_keys, old_values):
            if k is not None and k is not self.DELETED:
                self.put(k, v)
```

### 3.4 Quadratic Probing

```python
class HashTableQuadraticProbing:
    """Quadratic Probing: Mitigates primary clustering

    Probe sequence: h(k), h(k)+1^2, h(k)+2^2, h(k)+3^2, ...

    When the table size is prime and alpha < 0.5,
    the first m/2 probe positions are guaranteed to be distinct
    """
    DELETED = object()

    def __init__(self, size=17):  # Prime recommended
        self.size = size
        self.keys = [None] * size
        self.values = [None] * size
        self.count = 0

    def _hash(self, key):
        return hash(key) % self.size

    def _probe(self, key):
        """Generator for quadratic probing"""
        idx = self._hash(key)
        for i in range(self.size):
            yield (idx + i * i) % self.size

    def put(self, key, value):
        if self.count / self.size > 0.5:
            self._rehash()

        first_deleted = None
        for idx in self._probe(key):
            if self.keys[idx] is None:
                target = first_deleted if first_deleted is not None else idx
                self.keys[target] = key
                self.values[target] = value
                self.count += 1
                return
            elif self.keys[idx] is self.DELETED:
                if first_deleted is None:
                    first_deleted = idx
            elif self.keys[idx] == key:
                self.values[idx] = value
                return

    def get(self, key):
        for idx in self._probe(key):
            if self.keys[idx] is None:
                raise KeyError(key)
            if self.keys[idx] == key:
                return self.values[idx]
        raise KeyError(key)

    def _rehash(self):
        old_keys, old_values = self.keys, self.values
        # Expand to the next prime size
        self.size = self._next_prime(self.size * 2)
        self.keys = [None] * self.size
        self.values = [None] * self.size
        self.count = 0
        for k, v in zip(old_keys, old_values):
            if k is not None and k is not self.DELETED:
                self.put(k, v)

    @staticmethod
    def _next_prime(n):
        """Return the smallest prime >= n"""
        if n <= 2:
            return 2
        candidate = n if n % 2 != 0 else n + 1
        while True:
            if all(candidate % i != 0 for i in range(3, int(candidate**0.5) + 1, 2)):
                return candidate
            candidate += 2
```

### 3.5 Double Hashing

```python
class HashTableDoubleHashing:
    """Double Hashing: Eliminates secondary clustering as well

    Probe sequence: h1(k), h1(k)+h2(k), h1(k)+2*h2(k), ...

    h2(k) must be coprime with the table size
    -> Make the table size prime, or restrict h2's range
    """
    DELETED = object()

    def __init__(self, size=17):
        self.size = size
        self.keys = [None] * size
        self.values = [None] * size
        self.count = 0

    def _h1(self, key):
        """Primary hash function"""
        return hash(key) % self.size

    def _h2(self, key):
        """Secondary hash function (must never be 0)
        h2(k) = prime - (k mod prime) where prime < size
        """
        prime = self.size - 2  # If size is prime, size-2 is likely prime
        return prime - (hash(key) % prime)

    def _probe(self, key):
        idx = self._h1(key)
        step = self._h2(key)
        for i in range(self.size):
            yield (idx + i * step) % self.size

    def put(self, key, value):
        if self.count / self.size > 0.5:
            self._rehash()
        for idx in self._probe(key):
            if self.keys[idx] is None or self.keys[idx] is self.DELETED:
                self.keys[idx] = key
                self.values[idx] = value
                self.count += 1
                return
            elif self.keys[idx] == key:
                self.values[idx] = value
                return

    def get(self, key):
        for idx in self._probe(key):
            if self.keys[idx] is None:
                raise KeyError(key)
            if self.keys[idx] == key:
                return self.values[idx]
        raise KeyError(key)

    def _rehash(self):
        old_keys, old_values = self.keys, self.values
        self.size = self._next_prime(self.size * 2)
        self.keys = [None] * self.size
        self.values = [None] * self.size
        self.count = 0
        for k, v in zip(old_keys, old_values):
            if k is not None and k is not self.DELETED:
                self.put(k, v)

    @staticmethod
    def _next_prime(n):
        if n <= 2:
            return 2
        candidate = n if n % 2 != 0 else n + 1
        while True:
            if all(candidate % i != 0 for i in range(3, int(candidate**0.5) + 1, 2)):
                return candidate
            candidate += 2
```

### 3.6 Robin Hood Hashing

```python
class RobinHoodHashTable:
    """Robin Hood Hashing: Adopted by Rust's HashMap

    Principle: "Steal from the rich (elements with short probe distances)
               and give to the poor (elements with long probe distances)"

    - During insertion, swap with existing elements when probe distance is longer
    - Maximum probe distance is averaged, improving worst-case performance
    - Expected maximum probe distance: O(log n)
    """
    EMPTY = None

    def __init__(self, capacity=16):
        self.capacity = capacity
        self.size = 0
        self.keys = [self.EMPTY] * capacity
        self.values = [None] * capacity
        self.distances = [0] * capacity  # Distance from ideal position for each element

    def _hash(self, key):
        return hash(key) % self.capacity

    def put(self, key, value):
        if self.size >= self.capacity * 7 // 8:  # alpha < 7/8
            self._rehash()

        idx = self._hash(key)
        dist = 0

        while True:
            if self.keys[idx] is self.EMPTY:
                # Insert into empty slot
                self.keys[idx] = key
                self.values[idx] = value
                self.distances[idx] = dist
                self.size += 1
                return

            if self.keys[idx] == key:
                # Update existing key
                self.values[idx] = value
                return

            # Robin Hood: Swap if probe distance is longer than existing element
            if dist > self.distances[idx]:
                key, self.keys[idx] = self.keys[idx], key
                value, self.values[idx] = self.values[idx], value
                dist, self.distances[idx] = self.distances[idx], dist

            idx = (idx + 1) % self.capacity
            dist += 1

    def get(self, key):
        idx = self._hash(key)
        dist = 0

        while True:
            if self.keys[idx] is self.EMPTY:
                raise KeyError(key)
            if dist > self.distances[idx]:
                # Robin Hood property: no need to search further
                raise KeyError(key)
            if self.keys[idx] == key:
                return self.values[idx]
            idx = (idx + 1) % self.capacity
            dist += 1

    def _rehash(self):
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
```

### 3.7 Cuckoo Hashing

```python
class CuckooHashTable:
    """Cuckoo Hashing: Guarantees O(1) worst-case lookup

    Principle:
    - 2 hash functions h1, h2 and 2 tables T1, T2
    - Each key is stored in either T1[h1(k)] or T2[h2(k)]
    - Lookup: Check only 2 locations -> O(1) worst case
    - Insert: "Evict" existing elements and relocate (like a cuckoo's nest takeover)

    Rehashing is needed when cycles occur
    """
    def __init__(self, size=16):
        self.size = size
        self.table1 = [None] * size
        self.table2 = [None] * size
        self.count = 0
        self._max_kicks = 500  # Prevent infinite loops

    def _h1(self, key):
        return hash(key) % self.size

    def _h2(self, key):
        return hash(key * 2654435761) % self.size  # A different hash function

    def get(self, key):
        """O(1) worst case — just check 2 locations"""
        idx1 = self._h1(key)
        if self.table1[idx1] is not None and self.table1[idx1][0] == key:
            return self.table1[idx1][1]

        idx2 = self._h2(key)
        if self.table2[idx2] is not None and self.table2[idx2][0] == key:
            return self.table2[idx2][1]

        raise KeyError(key)

    def put(self, key, value):
        # Check for existing key update
        idx1 = self._h1(key)
        if self.table1[idx1] is not None and self.table1[idx1][0] == key:
            self.table1[idx1] = (key, value)
            return
        idx2 = self._h2(key)
        if self.table2[idx2] is not None and self.table2[idx2][0] == key:
            self.table2[idx2] = (key, value)
            return

        # New insertion
        entry = (key, value)
        for _ in range(self._max_kicks):
            # Try inserting into table 1
            idx1 = self._h1(entry[0])
            if self.table1[idx1] is None:
                self.table1[idx1] = entry
                self.count += 1
                return

            # Evict existing element
            entry, self.table1[idx1] = self.table1[idx1], entry

            # Try inserting into table 2
            idx2 = self._h2(entry[0])
            if self.table2[idx2] is None:
                self.table2[idx2] = entry
                self.count += 1
                return

            # Evict existing element
            entry, self.table2[idx2] = self.table2[idx2], entry

        # Cycle detected -> rehash and retry
        self._rehash()
        self.put(entry[0], entry[1])

    def _rehash(self):
        old_t1 = self.table1
        old_t2 = self.table2
        self.size *= 2
        self.table1 = [None] * self.size
        self.table2 = [None] * self.size
        self.count = 0
        for entry in old_t1 + old_t2:
            if entry is not None:
                self.put(entry[0], entry[1])
```

### 3.8 Detailed Comparison of Collision Resolution Methods

```
Clustering characteristics of each probing method:

  Linear Probing:
  +---+---+---+---+---+---+---+---+
  |   | X | X | X | X | X |   |   |  <- Primary cluster
  +---+---+---+---+---+---+---+---+
  Contiguous occupied slots grow -> new collisions are absorbed into the cluster

  Quadratic Probing:
  +---+---+---+---+---+---+---+---+
  |   | X |   | X |   |   | X |   |  <- Scattered
  +---+---+---+---+---+---+---+---+
  Primary clustering is resolved. Keys with the same hash follow the same path (secondary clustering)

  Double Hashing:
  +---+---+---+---+---+---+---+---+
  | X |   |   | X |   | X |   |   |  <- Nearly random
  +---+---+---+---+---+---+---+---+
  Step size differs per key -> no clustering
```

---

## 4. Load Factor

```
Load factor alpha = number of elements / number of buckets

Effect of alpha:
  +------------------------------------+
  |                                    |
  | Search time                        |
  | ^                                  |
  | |                        /         |
  | |                     /            |
  | |                  /   Chaining    |
  | |               /                  |
  | |          ///  Open Addressing    |
  | |    ///                           |
  | |//                                |
  | +-------------------------------> alpha |
  | 0   0.25  0.5  0.75  1.0          |
  |                                    |
  | Recommended alpha:                 |
  |   Chaining: alpha < 0.75          |
  |   Open Addressing: alpha < 0.5    |
  +------------------------------------+
```

### 4.1 Theoretical Search Cost

```python
def expected_probes_chaining(alpha):
    """Expected number of probes for chaining
    Successful: 1 + alpha/2
    Unsuccessful: 1 + alpha
    """
    return {
        "successful": 1 + alpha / 2,
        "unsuccessful": 1 + alpha
    }

def expected_probes_linear(alpha):
    """Expected number of probes for linear probing
    Successful: (1/2)(1 + 1/(1-alpha))
    Unsuccessful: (1/2)(1 + 1/(1-alpha)^2)
    """
    if alpha >= 1:
        return {"successful": float('inf'), "unsuccessful": float('inf')}
    return {
        "successful": 0.5 * (1 + 1 / (1 - alpha)),
        "unsuccessful": 0.5 * (1 + 1 / (1 - alpha) ** 2)
    }

def expected_probes_double(alpha):
    """Expected number of probes for double hashing
    Successful: (1/alpha) * ln(1/(1-alpha))
    Unsuccessful: 1/(1-alpha)
    """
    import math
    if alpha >= 1:
        return {"successful": float('inf'), "unsuccessful": float('inf')}
    if alpha == 0:
        return {"successful": 1, "unsuccessful": 1}
    return {
        "successful": (1 / alpha) * math.log(1 / (1 - alpha)),
        "unsuccessful": 1 / (1 - alpha)
    }

# Comparison by alpha
for alpha in [0.25, 0.5, 0.75, 0.9]:
    print(f"\n--- alpha = {alpha} ---")
    chain = expected_probes_chaining(alpha)
    linear = expected_probes_linear(alpha)
    double = expected_probes_double(alpha)
    print(f"Chaining:       success {chain['successful']:.2f}, fail {chain['unsuccessful']:.2f}")
    print(f"Linear Probing: success {linear['successful']:.2f}, fail {linear['unsuccessful']:.2f}")
    print(f"Double Hashing: success {double['successful']:.2f}, fail {double['unsuccessful']:.2f}")

# Output:
# --- alpha = 0.25 ---
# Chaining:       success 1.12, fail 1.25
# Linear Probing: success 1.17, fail 1.39
# Double Hashing: success 1.15, fail 1.33
#
# --- alpha = 0.5 ---
# Chaining:       success 1.25, fail 1.50
# Linear Probing: success 1.50, fail 2.50
# Double Hashing: success 1.39, fail 2.00
#
# --- alpha = 0.75 ---
# Chaining:       success 1.38, fail 1.75
# Linear Probing: success 2.50, fail 8.50
# Double Hashing: success 1.85, fail 4.00
#
# --- alpha = 0.9 ---
# Chaining:       success 1.45, fail 1.90
# Linear Probing: success 5.50, fail 50.50
# Double Hashing: success 2.56, fail 10.00
```

### 4.2 Amortized Analysis of Rehashing

```python
class AmortizedRehashDemo:
    """Amortized cost analysis of rehashing

    Total cost of n insertions:
    - Normal insertions: n x O(1) = O(n)
    - Rehashing: 1 + 2 + 4 + ... + n = O(n) (geometric series sum)

    Total cost O(n) / n operations = O(1) amortized

    In other words, even though individual rehashing costs O(n),
    amortized analysis shows each insertion is O(1)
    """
    def __init__(self):
        self.size = 2
        self.count = 0
        self.total_cost = 0
        self.rehash_count = 0

    def insert_simulation(self, n):
        """Simulate n insertions"""
        for i in range(n):
            self.count += 1
            self.total_cost += 1  # Normal insertion cost

            if self.count > self.size * 0.75:
                # Rehash: cost of reinserting all elements
                self.total_cost += self.count
                self.size *= 2
                self.rehash_count += 1

        print(f"Insert count: {n}")
        print(f"Rehash count: {self.rehash_count}")
        print(f"Final table size: {self.size}")
        print(f"Total cost: {self.total_cost}")
        print(f"Amortized cost (average): {self.total_cost / n:.2f}")

demo = AmortizedRehashDemo()
demo.insert_simulation(1000000)
# Insert count: 1000000
# Rehash count: 20
# Final table size: 2097152
# Total cost: ~3000000
# Amortized cost (average): ~3.00  -> O(1) constant
```

---

## 5. Language-Specific Hash Table Implementation Details

### Table 1: Collision Resolution Comparison

| Property | Chaining | Open Addressing |
|----------|----------|-----------------|
| Structure | Array + Lists | Array only |
| Memory | Extra pointer overhead | Densely packed |
| Worst-case search | O(n) | O(n) |
| Deletion | Easy | DELETED marker needed |
| Cache efficiency | Low | High |
| Recommended alpha | < 0.75 | < 0.5 |
| Implementation | Simple | Somewhat complex |

### Table 2: Hash Table Implementations by Language

| Language | Type Name | Collision Resolution | Initial Capacity | Max alpha |
|----------|-----------|---------------------|-----------------|-----------|
| Python | dict | Open addressing | 8 | 2/3 |
| Java | HashMap | Chaining (+tree) | 16 | 0.75 |
| C++ | unordered_map | Chaining | Implementation-dependent | 1.0 |
| Go | map | Chaining (buckets) | Implementation-dependent | 6.5 |
| Rust | HashMap | Robin Hood | Implementation-dependent | 7/8 |
| C# | Dictionary | Chaining | 3 | 1.0 |
| Ruby | Hash | Open addressing | 8 | Implementation-dependent |

### 5.1 Internal Structure of Python dict

```python
# Python 3.7+ dict uses a compact 2-layer structure:
#
# 1. Hash index array (sparse):
#    [_, _, 0, _, 1, _, 2, _]  <- Indices into actual entries
#
# 2. Entry array (dense, insertion order):
#    [("apple", 5), ("banana", 2), ("cherry", 8)]
#
# Benefits:
# - Insertion order is preserved (guaranteed from 3.7+)
# - Good memory efficiency (20-25% reduction from previous implementation)
# - Fast iteration (just scanning a dense array)

# Observing internal size changes of dict
import sys

d = {}
print(f"Empty dict: {sys.getsizeof(d)} bytes")  # 64 bytes

for i in range(10):
    d[f"key_{i}"] = i
    print(f"{i+1} elements: {sys.getsizeof(d)} bytes")

# Python dict hash probing method:
# - Open addressing (similar to quadratic probing)
# - Probe sequence: j = ((5*j) + 1 + perturb) % size
#   perturb is the initial hash value, right-shifted on each iteration
# - This method can traverse the entire table
```

### 5.2 Internal Structure of Java HashMap

```java
// Java 8+ HashMap characteristics:
//
// 1. Initial capacity: 16, load factor: 0.75
// 2. Capacity is always a power of 2
// 3. When bucket elements >= 8: convert to red-black tree (treeify)
// 4. When bucket elements <= 6: revert to list (untreeify)
// 5. Mix upper bits of hash into lower bits (perturbation)
//
// Key method implementation overview:

// static final int hash(Object key) {
//     int h;
//     return (key == null) ? 0 : (h = key.hashCode()) ^ (h >>> 16);
// }
//
// // Index computation: (n - 1) & hash
// // Since n is a power of 2, & operation speeds up mod
//
// // During resize: just check one bit of each element
// // to determine the new position (same or +oldCapacity)
```

### 5.3 Usage Examples by Language

```python
# === Python ===
# dict: Hash table
d = {"name": "Alice", "age": 30}
d["city"] = "Tokyo"          # O(1) insertion
print(d.get("name", "N/A"))  # O(1) lookup (with default value)
del d["age"]                  # O(1) deletion

# defaultdict: With default values
from collections import defaultdict
word_count = defaultdict(int)
for w in "hello world hello".split():
    word_count[w] += 1
print(dict(word_count))  # {'hello': 2, 'world': 1}

# Counter: Frequency counting
from collections import Counter
freq = Counter("mississippi")
print(freq.most_common(3))  # [('s', 4), ('i', 4), ('p', 2)]

# OrderedDict: Preserves insertion order (same as 3.7+ dict but supports order comparison)
from collections import OrderedDict
od = OrderedDict()
od["b"] = 2
od["a"] = 1
od.move_to_end("b")  # Move to end (useful for LRU cache)
```

```go
// === Go ===
package main

import "fmt"

func main() {
    // map: Hash table
    m := map[string]int{
        "apple":  5,
        "banana": 2,
    }

    // Insert
    m["cherry"] = 8

    // Lookup (two-value return)
    if val, ok := m["apple"]; ok {
        fmt.Println(val) // 5
    }

    // Delete
    delete(m, "banana")

    // Iteration (order is non-deterministic)
    for k, v := range m {
        fmt.Printf("%s: %d\n", k, v)
    }
}
```

```rust
// === Rust ===
use std::collections::HashMap;

fn main() {
    let mut scores = HashMap::new();

    // Insert
    scores.insert("Alice", 10);
    scores.insert("Bob", 20);

    // Lookup
    if let Some(score) = scores.get("Alice") {
        println!("Alice: {}", score);
    }

    // entry API: Insert if key doesn't exist
    scores.entry("Charlie").or_insert(30);

    // Update value
    let count = scores.entry("Alice").or_insert(0);
    *count += 5;

    // Iteration
    for (name, score) in &scores {
        println!("{}: {}", name, score);
    }
}
```

```typescript
// === TypeScript ===
// Object: String keys only
const obj: Record<string, number> = { apple: 5, banana: 2 };

// Map: Any type can be used as key
const map = new Map<string, number>();
map.set("apple", 5);
map.set("banana", 2);
console.log(map.get("apple"));  // 5
console.log(map.has("cherry")); // false
console.log(map.size);          // 2

// Set: Collection of values
const set = new Set<number>([1, 2, 3, 2, 1]);
console.log(set.size);  // 3

// WeakMap: Garbage collection compatible
const weakMap = new WeakMap<object, string>();
let key = {};
weakMap.set(key, "value");
// key = null; enables GC
```

---

## 6. Practical Application Patterns

### 6.1 LRU Cache Implementation

```python
from collections import OrderedDict

class LRUCache:
    """Least Recently Used Cache

    Uses OrderedDict to achieve O(1) get/put.
    Moves accessed elements to the end,
    and removes the oldest (head) element when capacity is exceeded.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        """O(1)"""
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)  # Mark as recently used
        return self.cache[key]

    def put(self, key, value):
        """O(1)"""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # Remove oldest element

# Usage
cache = LRUCache(3)
cache.put("a", 1)
cache.put("b", 2)
cache.put("c", 3)
print(cache.get("a"))  # 1 ("a" becomes most recently used)
cache.put("d", 4)      # Capacity exceeded -> "b" is removed
print(cache.get("b"))   # -1 (already removed)
```

### 6.2 Finding Matching Pairs

```python
def find_pairs_with_sum(arr, target):
    """Return all pairs whose sum equals target — O(n)

    Manage complements using a hash table
    """
    seen = {}
    pairs = []
    for num in arr:
        complement = target - num
        if complement in seen:
            pairs.append((complement, num))
        seen[num] = True
    return pairs

print(find_pairs_with_sum([1, 5, 7, -1, 5], 6))
# [(1, 5), (-1, 7), (1, 5)]
```

### 6.3 Isomorphic Strings

```python
def is_isomorphic(s: str, t: str) -> bool:
    """Determine whether two strings are isomorphic — O(n)

    "egg" and "add" -> True (e->a, g->d)
    "foo" and "bar" -> False
    """
    if len(s) != len(t):
        return False

    s_to_t = {}
    t_to_s = {}

    for cs, ct in zip(s, t):
        if cs in s_to_t:
            if s_to_t[cs] != ct:
                return False
        else:
            s_to_t[cs] = ct

        if ct in t_to_s:
            if t_to_s[ct] != cs:
                return False
        else:
            t_to_s[ct] = cs

    return True

print(is_isomorphic("egg", "add"))   # True
print(is_isomorphic("foo", "bar"))   # False
print(is_isomorphic("paper", "title"))  # True
```

### 6.4 Count Subarrays with Sum Equal to k

```python
def subarray_sum(nums, k):
    """Count of contiguous subarrays with sum equal to k — O(n)

    Combine prefix sums with a hash table
    """
    count = 0
    prefix_sum = 0
    prefix_count = {0: 1}  # Prefix sum 0 occurs once (empty prefix)

    for num in nums:
        prefix_sum += num
        # If prefix_sum - k existed in the past,
        # the sum of that interval is k
        if prefix_sum - k in prefix_count:
            count += prefix_count[prefix_sum - k]
        prefix_count[prefix_sum] = prefix_count.get(prefix_sum, 0) + 1

    return count

print(subarray_sum([1, 1, 1], 2))      # 2
print(subarray_sum([1, 2, 3], 3))      # 2 ([1,2] and [3])
print(subarray_sum([1, -1, 1, 1], 2))  # 3
```

### 6.5 Longest Consecutive Sequence

```python
def longest_consecutive(nums):
    """Return the length of the longest consecutive integer sequence — O(n)

    Example: [100, 4, 200, 1, 3, 2] -> 4 ([1, 2, 3, 4])

    Use a set for O(1) lookup, and count only from the starting points of sequences
    """
    num_set = set(nums)
    max_length = 0

    for num in num_set:
        # If num-1 doesn't exist -> this num is the start of a sequence
        if num - 1 not in num_set:
            current = num
            length = 1
            while current + 1 in num_set:
                current += 1
                length += 1
            max_length = max(max_length, length)

    return max_length

print(longest_consecutive([100, 4, 200, 1, 3, 2]))  # 4
print(longest_consecutive([0, 3, 7, 2, 5, 8, 4, 6, 0, 1]))  # 9
```

### 6.6 Bloom Filter

```python
import hashlib

class BloomFilter:
    """Bloom Filter: A space-efficient probabilistic data structure

    - "Element does not exist" is 100% accurate (no false negatives)
    - "Element exists" may have false positives
    - Deletion is not possible (use Counting Bloom Filter for that)

    Use cases: Spell checking, cache filtering, DB query optimization
    """
    def __init__(self, size=1000, num_hashes=3):
        self.size = size
        self.num_hashes = num_hashes
        self.bit_array = [False] * size
        self.count = 0

    def _hashes(self, item):
        """Generate multiple hash values"""
        results = []
        for i in range(self.num_hashes):
            h = hashlib.sha256(f"{item}:{i}".encode()).hexdigest()
            results.append(int(h, 16) % self.size)
        return results

    def add(self, item):
        """Add an element — O(k), k = number of hash functions"""
        for idx in self._hashes(item):
            self.bit_array[idx] = True
        self.count += 1

    def might_contain(self, item):
        """True if the element might exist
        False means it definitely does not exist
        """
        return all(self.bit_array[idx] for idx in self._hashes(item))

    def false_positive_rate(self):
        """Theoretical false positive probability
        p ~ (1 - e^(-kn/m))^k
        k: number of hash functions, n: number of elements, m: bit array size
        """
        import math
        k = self.num_hashes
        n = self.count
        m = self.size
        if n == 0:
            return 0.0
        return (1 - math.exp(-k * n / m)) ** k

# Usage
bf = BloomFilter(size=10000, num_hashes=5)
for word in ["apple", "banana", "cherry", "date"]:
    bf.add(word)

print(bf.might_contain("apple"))   # True
print(bf.might_contain("fig"))     # False (definitely does not exist)
print(bf.might_contain("grape"))   # False (or possible false positive)
print(f"False positive rate: {bf.false_positive_rate():.6f}")
```

### 6.7 Consistent Hashing

```python
import hashlib
from bisect import bisect_right

class ConsistentHash:
    """Consistent Hashing: Used for data distribution in distributed systems

    Minimizes the number of keys that need to be relocated when nodes are added/removed.
    - Normal hashing: Node change -> nearly all keys are relocated
    - Consistent hashing: Node change -> only K/N keys are relocated
      (K: total keys, N: number of nodes)

    Use cases: CDN, distributed caching (Memcached, Redis Cluster),
               distributed DB partitioning
    """
    def __init__(self, nodes=None, replicas=100):
        self.replicas = replicas  # Number of virtual nodes
        self.ring = []            # Sorted hash values
        self.hash_to_node = {}    # Hash value -> node name

        if nodes:
            for node in nodes:
                self.add_node(node)

    def _hash(self, key):
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def add_node(self, node):
        """Add a node to the ring"""
        for i in range(self.replicas):
            virtual_key = f"{node}:replica:{i}"
            h = self._hash(virtual_key)
            self.ring.append(h)
            self.hash_to_node[h] = node
        self.ring.sort()

    def remove_node(self, node):
        """Remove a node from the ring"""
        for i in range(self.replicas):
            virtual_key = f"{node}:replica:{i}"
            h = self._hash(virtual_key)
            self.ring.remove(h)
            del self.hash_to_node[h]

    def get_node(self, key):
        """Return the node to which a key is assigned"""
        if not self.ring:
            return None
        h = self._hash(key)
        idx = bisect_right(self.ring, h)
        if idx == len(self.ring):
            idx = 0
        return self.hash_to_node[self.ring[idx]]

# Usage
ch = ConsistentHash(["server-1", "server-2", "server-3"])
for key in ["user:100", "user:200", "user:300", "user:400", "user:500"]:
    print(f"{key} -> {ch.get_node(key)}")

# Check impact of adding a node
print("\n--- Adding server-4 ---")
ch.add_node("server-4")
for key in ["user:100", "user:200", "user:300", "user:400", "user:500"]:
    print(f"{key} -> {ch.get_node(key)}")
# Only some keys are relocated
```

---

## 7. Hash Table Security

### 7.1 Hash DoS Attacks and Countermeasures

```python
# === Hash DoS Attack ===
# An attacker sends a large number of keys that intentionally cause hash collisions
# -> O(n^2) processing time to DoS the server
#
# Countermeasures:
# 1. Python 3.3+: Hash randomization (PYTHONHASHSEED)
# 2. SipHash: Default hash function from Python 3.4+
# 3. Universal hashing: Use random hash functions
# 4. Request size limits

# Checking Python's hash randomization
import sys
print(f"Hash randomization flag: {sys.flags.hash_randomization}")

# Controlled via PYTHONHASHSEED environment variable
# PYTHONHASHSEED=0  -> Disable randomization (for tests requiring reproducibility)
# PYTHONHASHSEED=42 -> Fixed seed
# Not set            -> Random seed (default, recommended)

# SipHash characteristics:
# - Not cryptographically secure, but resistant to collision attacks
# - Fast even for short inputs (uses a 128-bit key)
# - Adopted by Python, Rust, Ruby, etc.
```

### 7.2 Countermeasures Against Timing Attacks on Hash Tables

```python
# Constant-time comparison (timing attack countermeasure)
import hmac

def safe_compare(a: str, b: str) -> bool:
    """Constant-time string comparison

    Normal == returns immediately when a mismatch is found,
    so timing differences can be exploited to guess passwords.
    """
    return hmac.compare_digest(a.encode(), b.encode())

# BAD: Timing information leaks
# if user_token == stored_token: ...

# GOOD: Constant-time comparison
# if safe_compare(user_token, stored_token): ...
```

---

## 8. Performance Benchmark

```python
import time
import random
import string

def benchmark_hash_tables():
    """Performance comparison of implementations"""
    n = 100000
    keys = [''.join(random.choices(string.ascii_letters, k=10)) for _ in range(n)]
    values = list(range(n))

    # === Python dict ===
    start = time.perf_counter()
    d = {}
    for k, v in zip(keys, values):
        d[k] = v
    dict_insert = time.perf_counter() - start

    start = time.perf_counter()
    for k in keys:
        _ = d[k]
    dict_lookup = time.perf_counter() - start

    # === Chaining ===
    start = time.perf_counter()
    ht = HashTableChaining(size=n * 2)
    for k, v in zip(keys[:10000], values[:10000]):  # Compare at smaller scale
        ht.put(k, v)
    chain_insert = time.perf_counter() - start

    print(f"Python dict - Insert {n} items: {dict_insert:.4f}s")
    print(f"Python dict - Lookup {n} items: {dict_lookup:.4f}s")
    print(f"Chaining    - Insert 10000 items: {chain_insert:.4f}s")
    print(f"\nPython dict is highly optimized with a C implementation")

# benchmark_hash_tables()
```

---

## 9. Anti-patterns

### Anti-pattern 1: Using Mutable Objects as Keys

```python
# BAD: Lists are not hashable
d = {}
key = [1, 2, 3]
# d[key] = "value"  # TypeError: unhashable type: 'list'

# GOOD: Convert to tuple
d[tuple(key)] = "value"

# BAD: Defining __hash__ with mutable fields in a custom class
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __hash__(self):
        return hash((self.x, self.y))
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

p = Point(1, 2)
d = {p: "origin"}
p.x = 10  # Hash value changes -> d[p] can no longer be found!

# GOOD: Make it frozen (immutable)
from dataclasses import dataclass

@dataclass(frozen=True)
class FrozenPoint:
    x: int
    y: int
    # frozen=True auto-generates __hash__ and __eq__
    # Fields cannot be modified

fp = FrozenPoint(1, 2)
d = {fp: "origin"}
# fp.x = 10  # FrozenInstanceError is raised
```

### Anti-pattern 2: Hash Functions with Many Collisions

```python
# BAD: All keys go to the same bucket -> O(n) search
def terrible_hash(key, size):
    return 0  # Always index 0

# BAD: Using only lower bits
def bad_hash(key, size):
    return key & 0xF  # Only 16 possibilities

# BAD: Even-sized table with only even keys
def biased_hash(key, size):
    return key % size  # size=16, key=even -> only even indices used

# GOOD: Python's hash() + prime size
def good_hash(key, size):
    return hash(key) % size

# GOOD: Utilize upper bits as well (Java approach)
def better_hash(key, size):
    h = hash(key)
    h ^= (h >> 16)  # Mix upper bits into lower bits
    return h % size
```

### Anti-pattern 3: Unnecessary Hash Table Usage

```python
# BAD: Using a hash table for small-scale data
# -> Linear search on a list is faster (smaller constant factor)
small_data = {"a": 1, "b": 2, "c": 3}  # 3 elements

# For about 3 elements, linear search on a list is sufficient
# The overhead of hash computation is larger
small_list = [("a", 1), ("b", 2), ("c", 3)]

# BAD: Using a hash table when key range is small
# -> Consider direct addressing
counts = {}
for x in data:  # If data is in range 0-255
    counts[x] = counts.get(x, 0) + 1

# GOOD: Direct address table
counts = [0] * 256
for x in data:
    counts[x] += 1
```

### Anti-pattern 4: Modifying dict During Iteration

```python
# BAD: Adding/removing elements during iteration
d = {"a": 1, "b": 2, "c": 3}
# for k in d:
#     if d[k] < 2:
#         del d[k]  # RuntimeError: dictionary changed size during iteration

# GOOD: Make a copy before iterating
d = {"a": 1, "b": 2, "c": 3}
for k in list(d.keys()):
    if d[k] < 2:
        del d[k]
print(d)  # {'b': 2, 'c': 3}

# GOOD: Create a new dict with a dict comprehension
d = {"a": 1, "b": 2, "c": 3}
d = {k: v for k, v in d.items() if v >= 2}
print(d)  # {'b': 2, 'c': 3}
```

---

## 10. Common Interview and Competitive Programming Patterns

### 10.1 Sliding Window + HashMap

```python
def length_of_longest_substring(s: str) -> int:
    """Length of the longest substring without repeating characters — O(n)

    Example: "abcabcbb" -> 3 ("abc")
    """
    char_index = {}
    max_length = 0
    start = 0

    for i, char in enumerate(s):
        if char in char_index and char_index[char] >= start:
            start = char_index[char] + 1
        char_index[char] = i
        max_length = max(max_length, i - start + 1)

    return max_length

print(length_of_longest_substring("abcabcbb"))  # 3
print(length_of_longest_substring("bbbbb"))      # 1
print(length_of_longest_substring("pwwkew"))     # 3
```

### 10.2 Frequency Count + Top-K

```python
import heapq
from collections import Counter

def top_k_frequent(nums, k):
    """Top k most frequent elements — O(n log k)

    Example: [1,1,1,2,2,3], k=2 -> [1, 2]
    """
    freq = Counter(nums)
    return heapq.nlargest(k, freq.keys(), key=freq.get)

print(top_k_frequent([1, 1, 1, 2, 2, 3], 2))  # [1, 2]

# Bucket sort version — O(n)
def top_k_frequent_bucket(nums, k):
    freq = Counter(nums)
    # freq[num] -> add num to bucket[freq[num]]
    buckets = [[] for _ in range(len(nums) + 1)]
    for num, count in freq.items():
        buckets[count].append(num)

    result = []
    for i in range(len(buckets) - 1, -1, -1):
        result.extend(buckets[i])
        if len(result) >= k:
            return result[:k]
    return result
```

### 10.3 O(1) Data Structure Design Using HashMap

```python
import random

class RandomizedSet:
    """Achieves O(1) for insert, remove, and getRandom

    Combines a dict (value -> index) + list (array of values)
    """
    def __init__(self):
        self.val_to_idx = {}
        self.vals = []

    def insert(self, val) -> bool:
        """O(1)"""
        if val in self.val_to_idx:
            return False
        self.val_to_idx[val] = len(self.vals)
        self.vals.append(val)
        return True

    def remove(self, val) -> bool:
        """O(1) — Swap with the last element and delete"""
        if val not in self.val_to_idx:
            return False
        idx = self.val_to_idx[val]
        last = self.vals[-1]
        # Swap with last
        self.vals[idx] = last
        self.val_to_idx[last] = idx
        # Delete last
        self.vals.pop()
        del self.val_to_idx[val]
        return True

    def get_random(self):
        """O(1)"""
        return random.choice(self.vals)

rs = RandomizedSet()
rs.insert(1)
rs.insert(2)
rs.insert(3)
rs.remove(2)
print(rs.get_random())  # 1 or 3
```

---

## 11. FAQ

### Q1: Why does Python dict preserve insertion order?

**A:** Since Python 3.7, dict uses a compact array structure that preserves insertion order. Internally, it has a 2-layer structure: a hash index array and a dense array in insertion order. This design improved memory efficiency by 20-25%, and iteration is fast since it just scans a dense array. Implemented in CPython 3.6, it was guaranteed by the language specification in 3.7.

### Q2: How can we avoid the O(n) worst case of hash tables?

**A:** Multiple strategies exist:
1. **Java 8+ HashMap**: Converts high-collision buckets to red-black trees, improving to O(log n)
2. **Universal hashing**: Random hash functions to counter adversarial inputs
3. **Cuckoo hashing**: Guarantees O(1) lookup even in the worst case
4. **Robin Hood hashing**: Averages probe distances
5. **Proper load factor management**: Rehash when threshold is exceeded

### Q3: Do set and dict have the same internal structure?

**A:** In Python, set and dict have nearly the same hash table structure. set is more memory-efficient since it doesn't store values. Operations (in, add, remove) are the same O(1) average. However, set additionally supports fast set operations (union, intersection, difference).

### Q4: How should the initial size of a hash table be determined?

**A:** The guideline is the expected number of elements divided by the load factor threshold. For example, 1000 elements with alpha=0.75 requires at least about 1334 buckets. Since rehashing costs O(n), pre-allocate a larger size when the size is known in advance. Python dict doesn't support pre-allocation via `dict.fromkeys(range(n))`, but Java HashMap allows specifying via `new HashMap<>(initialCapacity)`.

### Q5: When should you use a hash table vs a balanced BST?

**A:**
- **Hash table**: O(1) average search. Fastest when ordering is not needed.
- **Balanced BST (TreeMap, etc.)**: O(log n) but maintains key ordering. Use when range queries, min/max, or ordered traversal are needed.
- Practical guideline: Use hash tables for simple key lookups, BSTs when ordering-related operations are needed.

### Q6: How about hash tables in concurrent processing?

**A:** Regular hash tables are not thread-safe. Countermeasures:
- Java: `ConcurrentHashMap` (segment-level locking)
- Python: Wrap with `threading.Lock`, or use `multiprocessing.Manager().dict()`
- Go: `sync.Map` (optimized for read-heavy scenarios)
- General: Striped locks (locking per group of buckets) to improve concurrency

---


## FAQ

### Q1: What is the most important point for learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not only through theory, but by actually writing code and verifying behavior.

### Q2: What common mistakes do beginners make?

Skipping the basics and jumping to advanced topics. We recommend thoroughly understanding the fundamental concepts explained in this guide before moving to the next step.

### Q3: How is this applied in practice?

Knowledge of this topic is frequently utilized in day-to-day development work. It becomes particularly important during code reviews and architecture design.

---

## 12. Summary

| Item | Key Point |
|------|-----------|
| Hash Function | Requires uniform distribution, determinism, and speed. SipHash is the standard |
| Chaining | Simple to implement. Deletion is easy. Adopted by Java HashMap |
| Open Addressing | Good cache efficiency. Adopted by Python dict |
| Robin Hood | Averages probe distances. Adopted by Rust HashMap |
| Cuckoo Hashing | O(1) worst-case search. Theoretically important |
| Load Factor | Key to maintaining performance. Rehash when threshold is exceeded |
| Rehashing | Double table size + reinsert all elements. Amortized O(1) |
| Key Requirements | Immutable with consistent __hash__ and __eq__ |
| Security | Hash randomization to defend against DoS attacks |
| Practical Applications | LRU cache, consistent hashing, Bloom filter |

---

## Recommended Next Guides

- [Tree Structures — BST and Balanced Trees](./04-trees.md)
- [Time-Space Tradeoff — Bloom Filters](../00-complexity/02-space-time-tradeoff.md)

---

## References

1. Cormen, T.H. et al. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. -- Chapter 11 "Hash Tables"
2. Knuth, D.E. (1998). *The Art of Computer Programming, Volume 3*. Addison-Wesley. -- Theory of hashing
3. Python Developer's Guide. "Dictionaries." -- https://docs.python.org/3/c-api/dict.html
4. Pagh, R. & Rodler, F.F. (2004). "Cuckoo hashing." *Journal of Algorithms*, 51(2), 122-144.
5. Celis, P. (1986). *Robin Hood Hashing*. Technical Report CS-86-14, University of Waterloo.
6. Bloom, B.H. (1970). "Space/time trade-offs in hash coding with allowable errors." *Communications of the ACM*, 13(7), 422-426.
