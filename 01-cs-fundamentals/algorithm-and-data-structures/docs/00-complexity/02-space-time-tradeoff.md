# Space-Time Tradeoff — Memoization, Tables, Caching, and Bloom Filters

> A systematic study of techniques for reducing computation time by using additional memory (or, conversely, saving space at the expense of time).
> This is one of the most fundamental decision axes in algorithm design, and the knowledge directly impacts performance tuning in all software.

---

## What You Will Learn in This Chapter

1. **Theoretical Foundations of Tradeoffs** — Why time and space are interchangeable
2. **Memoization** — Speeding up recursive algorithms and analyzing their costs
3. **Caching Strategies** — Memory management with LRU, LFU, and TTL, and practical patterns
4. **Lookup Tables** — Design techniques for O(1) reference via precomputation
5. **Bloom Filters** — Dramatic space efficiency improvement through probabilistic data structures
6. **Practical Decision Criteria** — A decision-making framework for choosing the right technique at the right time

### Prerequisites

- Understanding of basic complexity notation (O, Theta, Omega)
- Ability to read basic Python syntax (recursion, dictionaries, list comprehensions)
- Familiarity with the concept of hash functions (details not required)

---

## 1. Theoretical Foundations of Tradeoffs

### 1.1 Why Are Time and Space Interchangeable?

The essence of computation is "state transformation." Consider a scenario where a computation result needs to be reused. There are two options:

1. **Recompute every time** — No space required, but computation takes time
2. **Store and reuse the result** — Time is reduced, but space is needed for storage

This relationship can be expressed mathematically. Consider a scenario where function f(x) is evaluated n times, with each computation taking time T:

- **No storage**: Time = O(n * T), Space = O(1)
- **Full storage**: Time = O(T + n) (initial computation + n lookups), Space = O(result size)

Between these two extremes lies an infinite number of intermediate points. This is what "tradeoff" means.

### 1.2 The Big Picture of Tradeoffs

```
The Big Picture of Space-Time Tradeoffs:

  Time  ^
  (Complexity)
       |  * Naive recomputation
       |     (high time, low space)
       |     e.g., Fibonacci via naive recursion every time
       |
       |     * Partial caching
       |        (cache only frequent items with LRU)
       |
       |        * Memoization
       |           (store all computed results)
       |
       |           * Lookup table
       |              (precompute all patterns)
       |                 low time, high space
       +---------------------------------------> Space
                                          (Memory usage)

  The design decision is choosing which point on the "Pareto frontier"
```

### 1.3 Relationship with Cobham's Thesis

In computational complexity theory, **PSPACE** is the class of problems solvable in polynomial space, and **P** is the class of problems solvable in polynomial time. The inclusion P ⊆ PSPACE holds, meaning "problems that can be solved efficiently in time can also be solved efficiently in space." The converse is unresolved (whether PSPACE = P is unknown), and space efficiency and time efficiency are not necessarily equivalent.

With this theoretical background, the following practical guidelines emerge:

| Situation | Recommended Direction | Reason |
|-----------|----------------------|--------|
| Ample memory, latency matters | Prioritize time (consume space) | Directly impacts user experience |
| Embedded / IoT | Prioritize space (sacrifice time) | Physical memory constraints are severe |
| Batch processing | Depends on situation | Balance between throughput and cost |
| Cloud environment | Cost optimization | Compare memory billing vs. compute time billing |

---

## 2. Memoization

### 2.1 Principles of Memoization

Memoization is a technique that stores function computation results using arguments as keys and returns the stored result when the function is called again with the same arguments. Two conditions must be met for this technique to be effective:

1. **Overlapping subproblems**: The function must be called multiple times with the same arguments
2. **Referential transparency**: The function must always return the same result for the same arguments (no side effects)

If condition 1 is not met, cache hits never occur and space is wasted. If condition 2 is not met, cached values become incorrect.

### 2.2 Naive Fibonacci vs. Memoized Fibonacci

Below is fully working code that can measure the effect of memoization.

```python
"""
Fibonacci sequence: Naive recursion vs. memoization comparison
- Naive: Time O(2^n), Space O(n) (call stack)
- Memoized: Time O(n), Space O(n) (cache + call stack)
"""

import time
import sys


def fib_naive(n: int) -> int:
    """Fibonacci via naive recursion.

    Why O(2^n):
    fib(n) calls fib(n-1) and fib(n-2).
    The recursion tree has height n and up to 2 branches at each level.
    Therefore, the number of calls is close to 2^n in the worst case.
    In reality, the number of calls to fib(n) is fib(n+1) - 1,
    which is proportional to phi^n using the golden ratio
    phi = (1+sqrt(5))/2 ~ 1.618.
    """
    if n <= 1:
        return n
    return fib_naive(n - 1) + fib_naive(n - 2)


def fib_memo(n: int, memo: dict = None) -> int:
    """Fibonacci with memoization.

    Why O(n):
    Each fib(k) (k=0..n) is "actually computed" at most once.
    On subsequent calls, it is retrieved from memo in O(1).
    Therefore, the number of computations is n+1, each O(1),
    giving an overall O(n).

    Note: Using a mutable object as a default argument causes
    it to be shared between function calls. Here we intentionally
    use None as default and create the dictionary on the first call.
    """
    if memo is None:
        memo = {}
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib_memo(n - 1, memo) + fib_memo(n - 2, memo)
    return memo[n]


def fib_bottom_up(n: int) -> int:
    """Fibonacci via bottom-up DP.

    Why does this approach exist:
    Memoization (top-down) relies on recursive calls, so it hits
    Python's default recursion limit (typically 1000).
    Bottom-up uses a loop for computation and is not constrained
    by the recursion limit.

    Time O(n), Space O(1) (only retains the previous two values)
    """
    if n <= 1:
        return n
    prev2, prev1 = 0, 1
    for _ in range(2, n + 1):
        prev2, prev1 = prev1, prev2 + prev1
    return prev1


def benchmark_fibonacci():
    """Compare performance of the three approaches."""
    print("=" * 60)
    print("Fibonacci Sequence: Benchmark by Method")
    print("=" * 60)

    # The naive version is limited to about n=35 (beyond that, wait time is too long)
    for n in [10, 20, 30, 35]:
        start = time.perf_counter()
        result = fib_naive(n)
        elapsed = time.perf_counter() - start
        print(f"Naive      fib({n:2d}) = {result:>15,d}  "
              f"Time: {elapsed:.6f}s")

    print("-" * 60)

    # Memoized and bottom-up versions are fast even for large n
    sys.setrecursionlimit(10000)
    for n in [10, 100, 1000, 5000]:
        memo = {}
        start = time.perf_counter()
        result_memo = fib_memo(n, memo)
        elapsed_memo = time.perf_counter() - start

        start = time.perf_counter()
        result_bu = fib_bottom_up(n)
        elapsed_bu = time.perf_counter() - start

        assert result_memo == result_bu, "Results do not match"
        digits = len(str(result_memo))
        print(f"Memoized   fib({n:4d}): {digits:>5d} digits  "
              f"Time: {elapsed_memo:.6f}s  "
              f"Cache size: {len(memo)}")
        print(f"Bottom-up  fib({n:4d}): {digits:>5d} digits  "
              f"Time: {elapsed_bu:.6f}s  "
              f"Extra space: O(1)")


if __name__ == "__main__":
    benchmark_fibonacci()
```

### 2.3 Understanding Call Reduction with Diagrams

```
Naive Fibonacci fib(5) recursion tree — many duplicate computations:

               fib(5)
              /      \
          fib(4)      fib(3)        <- fib(3) appears 2 times
          /    \       /   \
      fib(3)  fib(2) fib(2) fib(1)  <- fib(2) appears 3 times
      /  \    /  \    /  \
   f(2) f(1) f(1) f(0) f(1) f(0)   <- fib(1) appears 5 times
   / \
 f(1) f(0)

Number of calls: 15 (already many for n=5)
For n=40, this reaches about 330 million calls

---------------------------------------------------

Memoized fib(5) — each value is "truly computed" only once:

  Call order:
  fib(5) -> fib(4) -> fib(3) -> fib(2) -> fib(1): return 1  <- compute
                                       fib(0): return 0  <- compute
                              fib(2) = 1       <- compute(1+0)
                     fib(3):  fib(1) -> cache hit
                     fib(3) = 2                <- compute(1+1)
            fib(4):  fib(2) -> cache hit
            fib(4) = 3                         <- compute(2+1)
   fib(5):  fib(3) -> cache hit
   fib(5) = 5                                  <- compute(3+2)

  Actual computations: 6 (fib(0)..fib(5), once each)
  Cache lookups: 3

  State changes of memo:
  {} -> {1:1} -> {1:1, 0:0} -> {1:1, 0:0, 2:1}
     -> {..., 3:2} -> {..., 4:3} -> {..., 5:5}
```

### 2.4 Memoization with Python Decorators

`functools.lru_cache` is a memoization decorator provided by the Python standard library. Internally, it implements an LRU (Least Recently Used) cache using a combination of a doubly linked list and a hash map.

```python
"""
Usage patterns and considerations for functools.lru_cache

What lru_cache does internally:
1. Converts arguments to a tuple to generate a hash key
2. If the key exists in the cache, moves that entry to "most recently used" and returns it
3. If it doesn't exist, executes the function and adds the result to the cache
4. If maxsize is exceeded, removes the "oldest" entry

Why distinguish between maxsize=None and maxsize=128:
- maxsize=None: Unlimited cache. Stores all results. Memory usage grows indefinitely.
  -> Suitable when the number of subproblems is finite and predictable (e.g., DP)
- maxsize=128 (default): Maximum 128 entries. Evicts old entries via LRU.
  -> Suitable when input varieties are vast but access has locality
"""

from functools import lru_cache
import time


@lru_cache(maxsize=None)
def fib_cached(n: int) -> int:
    """Memoized Fibonacci — lru_cache version

    Why maxsize=None:
    The number of subproblems for fib(n) is n+1, which is finite.
    Caching all of them stays within O(n) space.
    If LRU eviction occurs, recomputation becomes necessary and is inefficient.
    """
    if n <= 1:
        return n
    return fib_cached(n - 1) + fib_cached(n - 2)


@lru_cache(maxsize=256)
def expensive_api_simulation(user_id: int, query: str) -> dict:
    """Example of caching an expensive operation

    Why maxsize=256:
    The combination of user ID x query can be enormous.
    Caching everything would exhaust memory.
    Since access patterns have temporal locality,
    retaining only recent entries yields sufficient hit rates.
    """
    # Simulate heavy processing
    time.sleep(0.001)
    return {"user_id": user_id, "query": query, "result": "data"}


def demonstrate_cache_info():
    """Examine lru_cache statistics."""
    # Clear cache
    fib_cached.cache_clear()

    # Execute computation
    result = fib_cached(100)
    info = fib_cached.cache_info()

    print(f"fib(100) = {result}")
    print(f"Cache info:")
    print(f"  Hits:         {info.hits}")
    print(f"  Misses:       {info.misses}")
    print(f"  Max size:     {info.maxsize}")
    print(f"  Current size: {info.currsize}")
    print(f"  Hit rate:     {info.hits / (info.hits + info.misses) * 100:.1f}%")


if __name__ == "__main__":
    demonstrate_cache_info()
```

### 2.5 Space Cost Analysis of Memoization

Memoization is a technique that "buys time with space," so analyzing the space cost is crucial.

```
Space cost breakdown of memoization:

1. Cache body
   +--------------------------------------------+
   |  Dictionary (dict) overhead                 |
   |  - Empty slots in the hash table            |
   |  - Key and value for each entry             |
   |  - Python object headers                    |
   |                                             |
   |  Example: Memoizing fib(1000)               |
   |  - Number of entries: 1001                  |
   |  - Per entry: ~100 bytes (approximate)      |
   |  - Total: ~100KB                            |
   +--------------------------------------------+

2. Call stack (for recursion)
   +--------------------------------------------+
   |  Stack frames from recursive calls          |
   |  - Maximum depth: O(n)                      |
   |  - Per frame: ~several hundred bytes        |
   |                                             |
   |  Python default recursion limit: 1000       |
   |  -> Can be changed with                     |
   |     sys.setrecursionlimit()                 |
   |  -> However, changing it carries the risk   |
   |    of stack overflow                        |
   +--------------------------------------------+

3. Space comparison with bottom-up DP
   +--------------------------------------------+
   |  Problem         Memoization  Bottom-up     |
   |  --------------- ----------   -----------   |
   |  Fibonacci       O(n)         O(1)*         |
   |  LCS             O(m*n)       O(min(m,n))   |
   |  Knapsack        O(n*W)       O(W)*         |
   |                                             |
   |  * In bottom-up, rows that are no longer    |
   |    needed can be discarded to reduce space  |
   +--------------------------------------------+
```

### 2.6 When Memoization Is Effective vs. Ineffective

| Condition | Memoization Effective | Memoization Ineffective |
|-----------|----------------------|------------------------|
| Overlapping subproblems | Present (Fibonacci, shortest path) | Absent (binary search, merge sort) |
| Referential transparency | Present (pure functions) | Absent (random, depends on current time) |
| Number of subproblems | Polynomial (n^k) | Exponential (2^n) — cache itself explodes |
| Argument hashing | Possible (integers, strings) | Difficult (lists, mutable objects) |

---

## 3. Systematic Caching Strategies

### 3.1 Comparison of Cache Eviction Policies

Memoization is a simple strategy of "storing all results," but in practice, memory is limited, so **cache eviction policies** — deciding which entries to keep and which to evict — become important.

```python
"""
Implementation and comparison of major cache eviction policies

Why do multiple policies exist:
The optimal policy differs depending on the access pattern.
- LRU: Strong against temporal locality (recently accessed items are likely to be accessed again)
- LFU: Strong against frequency locality (frequently used items are likely to be used again)
- FIFO: Simplest implementation. Default choice when locality cannot be assumed.
"""

from collections import OrderedDict, defaultdict
import time


class LRUCache:
    """Least Recently Used cache

    Why use OrderedDict:
    LRU evicts the "least recently used entry."
    This requires tracking the "last access time" of each entry.
    OrderedDict preserves insertion order and can move entries to the
    end in O(1) with move_to_end(), making it ideal for LRU.

    Time complexity: get/put both O(1) average
    Space complexity: O(capacity)
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # Remove the first (oldest) entry

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class LFUCache:
    """Least Frequently Used cache

    Why choose LFU over LRU in some cases:
    When the access pattern has "popular items,"
    LFU keeps high-frequency items cached, making it more efficient.
    Example: Content caching on CDNs (popular videos are always cached)

    Weaknesses of LFU:
    - Items that were popular in the past but are no longer needed persist
    - New items have difficulty establishing in the cache (due to low frequency)
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}            # key -> value
        self.freq = defaultdict(int)  # key -> usage frequency
        self.hits = 0
        self.misses = 0

    def get(self, key):
        if key in self.cache:
            self.hits += 1
            self.freq[key] += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key, value):
        if self.capacity <= 0:
            return
        if key in self.cache:
            self.cache[key] = value
            self.freq[key] += 1
            return
        if len(self.cache) >= self.capacity:
            # Find and remove the key with the lowest frequency
            min_freq_key = min(self.freq, key=lambda k: self.freq[k])
            del self.cache[min_freq_key]
            del self.freq[min_freq_key]
        self.cache[key] = value
        self.freq[key] = 1

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class TTLCache:
    """Time-To-Live cache

    Why TTL is necessary:
    External data (API responses, DB query results) may change over
    time. By setting a TTL, stale cache entries are automatically
    invalidated, ensuring data freshness.

    Difference from LRU/LFU:
    - LRU/LFU: Eviction based on capacity constraints
    - TTL: Invalidation based on time constraints
    - In practice, LRU + TTL are often combined
    """

    def __init__(self, capacity: int, ttl_seconds: float):
        self.capacity = capacity
        self.ttl = ttl_seconds
        self.cache = OrderedDict()  # key -> (value, timestamp)
        self.hits = 0
        self.misses = 0

    def _is_expired(self, key) -> bool:
        if key not in self.cache:
            return True
        _, timestamp = self.cache[key]
        return (time.time() - timestamp) > self.ttl

    def get(self, key):
        if key in self.cache and not self._is_expired(key):
            self.hits += 1
            value, _ = self.cache[key]
            # Update timestamp
            self.cache[key] = (value, time.time())
            self.cache.move_to_end(key)
            return value
        # Delete expired entry if present
        if key in self.cache:
            del self.cache[key]
        self.misses += 1
        return None

    def put(self, key, value):
        self.cache[key] = (value, time.time())
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


def compare_cache_policies():
    """Demo comparing different cache policies.

    Scenario: Cache capacity of 5, skewed access pattern
    """
    import random
    random.seed(42)

    # Zipf-like access pattern: concentrated on a few keys
    # Keys 0-2 are high frequency, keys 3-19 are low frequency
    access_pattern = []
    for _ in range(1000):
        if random.random() < 0.7:
            access_pattern.append(random.randint(0, 2))   # 70%: popular keys
        else:
            access_pattern.append(random.randint(3, 19))   # 30%: others

    lru = LRUCache(capacity=5)
    lfu = LFUCache(capacity=5)

    for key in access_pattern:
        # get and put on miss
        for cache in [lru, lfu]:
            if cache.get(key) is None:
                cache.put(key, f"value_{key}")

    print("Cache Policy Comparison (capacity=5, accesses=1000)")
    print(f"  LRU hit rate: {lru.hit_rate():.1%}")
    print(f"  LFU hit rate: {lfu.hit_rate():.1%}")


if __name__ == "__main__":
    compare_cache_policies()
```

### 3.2 Cache Policy Selection Criteria

| Policy | Best For | Worst For | Implementation Complexity |
|--------|----------|-----------|--------------------------|
| LRU | Strong temporal locality | Scan pollution (massive one-time accesses) | O(1) operations, moderate |
| LFU | Fixed popular items | Rapidly shifting popularity | O(1) requires effort, somewhat high |
| FIFO | Weak/unpredictable locality | Cases with locality (inferior to LRU) | Simplest |
| TTL | Data where freshness matters | When freshness is irrelevant and capacity is the concern | Requires time management, moderate |
| Random | Minimizing computation cost | Cases with locality | Simplest |

---

## 4. Lookup Tables

### 4.1 Principles of Precomputation

A lookup table is a technique that precomputes the output for all inputs and stores them in a table, applicable when "the input space is finite and small." At query time, no computation is performed; results are obtained in O(1) through table lookup alone.

Why this technique is powerful: Even if the original function has complexity O(f(n)), a table lookup is always O(1). By paying the precomputation cost once, unlimited O(1) queries become possible thereafter.

### 4.2 popcount (Bit Count) Table

```python
"""
Implement and compare three methods for popcount (counting 1-bits in an integer).

Why popcount is important:
- Hamming distance computation (error correction codes)
- Bit vector implementation of set operations
- Bitboard evaluation in chess engines
- Implementation of Bloom filters
"""

import time


def popcount_naive(n: int) -> int:
    """Naive method: examine bits one by one.

    Time: O(log n) — proportional to the number of bits in n
    Space: O(1)

    Why n & 1 reveals the bit:
    n & 1 extracts the least significant bit of n.
    n >>= 1 right-shifts so the next bit becomes the LSB.
    Repeat until n becomes 0.
    """
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count


def popcount_kernighan(n: int) -> int:
    """Brian Kernighan's algorithm.

    Time: O(number of set bits) — proportional to the number of 1-bits
    Space: O(1)

    Why n & (n-1) clears the lowest set bit:
    When n = ...10...0 (the lowest 1-bit and zeros to its right),
    n-1 = ...01...1 (that 1-bit becomes 0 and all bits to the right become 1).
    Therefore n & (n-1) clears only the lowest set bit.

    Example: n = 12 = 1100
             n-1 = 11 = 1011
             n & (n-1) = 1000 = 8  -> the lowest 1 (position 2) was cleared
    """
    count = 0
    while n:
        n &= n - 1
        count += 1
    return count


# Build the 8-bit lookup table
# Why 8 bits:
# - 256 entries cover all patterns, requiring only 256 bytes of memory
# - 16 bits (65536 entries) is also feasible, but a size that fits
#   in L1 cache is preferable
# - 32 bits (~4 billion entries) is impractical
POPCOUNT_TABLE_8BIT = [0] * 256
for i in range(256):
    # Bit count of i = LSB + bit count of (i >> 1)
    # This recurrence builds from 0 upward
    POPCOUNT_TABLE_8BIT[i] = (i & 1) + POPCOUNT_TABLE_8BIT[i >> 1]


def popcount_table(n: int) -> int:
    """popcount via 8-bit table lookup.

    Time: O(1) — always 4 table lookups for a 32-bit integer
    Space: O(256) — table size

    Why split into 8-bit chunks:
    A 32-bit integer is split into 4 eight-bit chunks,
    and the popcount of each chunk is obtained in O(1) from the table
    and summed.
    """
    return (POPCOUNT_TABLE_8BIT[n & 0xFF] +
            POPCOUNT_TABLE_8BIT[(n >> 8) & 0xFF] +
            POPCOUNT_TABLE_8BIT[(n >> 16) & 0xFF] +
            POPCOUNT_TABLE_8BIT[(n >> 24) & 0xFF])


def benchmark_popcount():
    """Performance comparison of the three methods."""
    import random
    random.seed(42)
    test_values = [random.randint(0, 2**32 - 1) for _ in range(100000)]

    # Verify correctness
    for v in test_values[:100]:
        assert popcount_naive(v) == popcount_kernighan(v) == popcount_table(v), \
            f"Mismatch: {v}"

    methods = [
        ("Naive (bit scan)",       popcount_naive),
        ("Kernighan's method",     popcount_kernighan),
        ("Table lookup (8-bit)",   popcount_table),
    ]

    print("popcount Benchmark (100,000 iterations)")
    print("-" * 50)
    for name, func in methods:
        start = time.perf_counter()
        for v in test_values:
            func(v)
        elapsed = time.perf_counter() - start
        print(f"  {name:25s}: {elapsed:.4f}s")


if __name__ == "__main__":
    benchmark_popcount()
```

### 4.3 Trigonometric Function Table — A Classic Pattern in Game Development

```python
"""
Trigonometric function lookup table

Why trigonometric tables are used in game development:
1. math.sin/cos internally uses Taylor expansion or CORDIC,
   taking tens to hundreds of nanoseconds per call
2. Games require thousands of trigonometric calculations per frame (1/60 second)
3. A table lookup takes only one array access (a few nanoseconds)
4. Angular precision of 1 degree or 0.1 degrees is often sufficient

Tradeoff:
- Precision: Controlled by table granularity (1 deg vs. 0.1 deg vs. 0.01 deg)
- Space: Finer granularity means larger tables
- Interpolation: Values between table entries can be improved with linear interpolation
"""

import math
import time


class TrigTable:
    """Trigonometric function lookup table (with linear interpolation)"""

    def __init__(self, resolution: float = 1.0):
        """
        resolution: Step size in degrees. Smaller means higher precision but more space.

        Why compute everything in the constructor:
        The table is immutable and only needs to be built once at application startup.
        It can then be safely shared across all threads.
        """
        self.resolution = resolution
        self.size = int(360 / resolution)
        self.sin_table = [0.0] * self.size
        self.cos_table = [0.0] * self.size

        for i in range(self.size):
            angle_rad = math.radians(i * resolution)
            self.sin_table[i] = math.sin(angle_rad)
            self.cos_table[i] = math.cos(angle_rad)

        # Calculate space usage
        self.memory_bytes = self.size * 8 * 2  # float64 x 2 tables

    def sin(self, degrees: float) -> float:
        """sin approximation via table lookup + linear interpolation."""
        degrees = degrees % 360
        idx_f = degrees / self.resolution
        idx = int(idx_f)
        frac = idx_f - idx  # Fractional part (for interpolation)

        if frac < 1e-9:
            return self.sin_table[idx % self.size]

        # Linear interpolation: f(x) ~ f(a) + (f(b) - f(a)) * t
        val_a = self.sin_table[idx % self.size]
        val_b = self.sin_table[(idx + 1) % self.size]
        return val_a + (val_b - val_a) * frac

    def cos(self, degrees: float) -> float:
        """cos approximation via table lookup + linear interpolation."""
        degrees = degrees % 360
        idx_f = degrees / self.resolution
        idx = int(idx_f)
        frac = idx_f - idx

        if frac < 1e-9:
            return self.cos_table[idx % self.size]

        val_a = self.cos_table[idx % self.size]
        val_b = self.cos_table[(idx + 1) % self.size]
        return val_a + (val_b - val_a) * frac


def benchmark_trig():
    """Compare precision and speed."""
    import random
    random.seed(42)

    table_1deg = TrigTable(resolution=1.0)    # 360 entries
    table_01deg = TrigTable(resolution=0.1)   # 3600 entries

    test_angles = [random.uniform(0, 360) for _ in range(100000)]

    # Precision comparison
    max_error_1deg = 0
    max_error_01deg = 0
    for angle in test_angles[:1000]:
        exact = math.sin(math.radians(angle))
        err_1 = abs(table_1deg.sin(angle) - exact)
        err_01 = abs(table_01deg.sin(angle) - exact)
        max_error_1deg = max(max_error_1deg, err_1)
        max_error_01deg = max(max_error_01deg, err_01)

    print("Trig Table: Precision and Speed Comparison")
    print("=" * 55)
    print(f"  1-degree:   Max error = {max_error_1deg:.8f}  "
          f"Space = {table_1deg.memory_bytes:,} bytes")
    print(f"  0.1-degree: Max error = {max_error_01deg:.8f}  "
          f"Space = {table_01deg.memory_bytes:,} bytes")
    print()

    # Speed comparison
    methods = [
        ("math.sin",          lambda a: math.sin(math.radians(a))),
        ("Table (1 deg)",     table_1deg.sin),
        ("Table (0.1 deg)",   table_01deg.sin),
    ]
    print("Speed Comparison (100,000 iterations)")
    print("-" * 45)
    for name, func in methods:
        start = time.perf_counter()
        for a in test_angles:
            func(a)
        elapsed = time.perf_counter() - start
        print(f"  {name:20s}: {elapsed:.4f}s")


if __name__ == "__main__":
    benchmark_trig()
```

### 4.4 Table Size vs. Precision Tradeoff

```
Relationship between table size and precision (trigonometric example):

  Max error ^
           |
  1e-2     |  *  10-degree steps (36 entries, 576B)
           |
  1e-3     |     *  1-degree steps (360 entries, 5.6KB)
           |
  1e-5     |        *  0.1-degree steps (3600 entries, 56KB)
           |
  1e-7     |           *  0.01-degree steps (36000 entries, 562KB)
           |
  1e-9     |              *  0.001-degree steps (360000 entries, 5.6MB)
           |
           +---------------------------------------------> Table size
                                                         (Memory usage)

  For every order of magnitude improvement in precision, the table size grows 10x.
  For most applications, 0.1-degree steps (56KB) are sufficient.
  56KB is a size that barely fits in a modern CPU's L1 cache
  (typically 32-64KB), enabling fast access.
```

---

## 5. Bloom Filters

### 5.1 Why Are Bloom Filters Necessary?

"Membership testing" against huge datasets is needed in many systems:

- Web browsers: Whether a URL is on a malicious site list (Google Safe Browsing)
- Databases: Whether a key exists in an SSTable (LevelDB, RocksDB, Cassandra)
- Networking: Whether a packet fingerprint matches a known one

Using a "complete hash set" for these scenarios requires memory proportional to the data size. For example, storing 10 million URLs in a hash set requires several hundred MB. A Bloom filter can represent the same data in just a few MB, and false negatives (reporting "does not exist" for something that actually exists) never occur.

### 5.2 Detailed Mechanism

```
How a Bloom filter works:

[Initial state] Bit array of m=12 bits, k=3 hash functions

  Position: 0  1  2  3  4  5  6  7  8  9  10 11
  Value:   [0][0][0][0][0][0][0][0][0][0][0][0]

[Adding "apple"]
  h1("apple") = 1
  h2("apple") = 5    -> Set positions 1, 5, 9 to 1
  h3("apple") = 9

  Position: 0  1  2  3  4  5  6  7  8  9  10 11
  Value:   [0][1][0][0][0][1][0][0][0][1][0][0]

[Adding "banana"]
  h1("banana") = 3
  h2("banana") = 5   -> Set positions 3, 5, 11 to 1
  h3("banana") = 11     (position 5 is already 1)

  Position: 0  1  2  3  4  5  6  7  8  9  10 11
  Value:   [0][1][0][1][0][1][0][0][0][1][0][1]

[Querying "cherry" — true negative]
  h1("cherry") = 2   -> Position 2: 0 -> Immediately "does not exist"
  h2("cherry") = 7       (if any bit is 0, definitely absent)
  h3("cherry") = 9

[Querying "date" — false positive!]
  h1("date") = 1    -> Position 1: 1 check
  h2("date") = 3    -> Position 3: 1 check  (set by banana)
  h3("date") = 9    -> Position 9: 1 check  (set by apple)

  All 1 -> Answers "probably exists"
  But "date" was never added -> This is a false positive

[Why false negatives never occur]
  When element x is added, all positions h1(x), h2(x), ..., hk(x) are set to 1.
  Bits only change from 0->1, never from 1->0 (since there is no deletion).
  Therefore, querying an already-added element always finds all bits set to 1.
```

### 5.3 Mathematical Analysis — Derivation of False Positive Rate

The false positive rate of a Bloom filter is determined by the following parameters:

- **m**: Size of the bit array
- **n**: Number of elements to be added
- **k**: Number of hash functions

After adding n elements, the probability that a given bit is still 0:

```
P(bit=0) = (1 - 1/m)^(k*n) ~ e^(-kn/m)
```

A false positive occurs when all k hash positions are 1:

```
False positive rate ~ (1 - e^(-kn/m))^k
```

The optimal number of hash functions is:

```
k_opt = (m/n) * ln(2) ~ 0.693 * (m/n)
```

At this point, the false positive rate is:

```
FPR_opt = (1/2)^k = (0.6185)^(m/n)
```

### 5.4 Complete Python Implementation

```python
"""
Complete Bloom filter implementation — with parameter optimization and
false positive rate verification

This code includes:
1. A basic Bloom filter class
2. Automatic calculation of optimal parameters
3. Comparison of theoretical and expected measured false positive rates
"""

import hashlib
import math


class BloomFilter:
    """Complete implementation of a Bloom filter.

    Why use md5:
    What is required of a Bloom filter's hash functions is not
    cryptographic security but uniform distribution of output.
    md5 has been broken for cryptographic use, but its property
    of uniformly distributing 128-bit output remains intact,
    which is sufficient for Bloom filters.

    In practice, non-cryptographic hash functions like
    mmh3 (MurmurHash3) or xxHash are faster and recommended.
    """

    def __init__(self, expected_items: int, false_positive_rate: float = 0.01):
        """
        expected_items: Expected number of elements to be added
        false_positive_rate: Acceptable false positive rate (0 to 1)

        Why this constructor automatically computes m and k:
        It is tedious and error-prone for users to manually calculate
        optimal m and k. Automatically deriving optimal values from
        expected_items and false_positive_rate is safer and more usable.
        """
        # Optimal bit array size: m = -n*ln(p) / (ln2)^2
        self.size = self._optimal_size(expected_items, false_positive_rate)
        # Optimal number of hash functions: k = (m/n) * ln2
        self.num_hashes = self._optimal_hash_count(
            self.size, expected_items
        )
        self.bit_array = bytearray(
            (self.size + 7) // 8  # Convert bits to bytes (round up)
        )
        self.count = 0  # Number of added elements

        # Record parameters
        self.expected_items = expected_items
        self.target_fpr = false_positive_rate

    @staticmethod
    def _optimal_size(n: int, p: float) -> int:
        """Calculate the optimal bit array size.

        Derivation:
        Minimize m for the false positive rate p = (1 - e^(-kn/m))^k.
        Substituting k with the optimal value k = (m/n)*ln2:
        m = -n * ln(p) / (ln2)^2
        """
        m = -n * math.log(p) / (math.log(2) ** 2)
        return int(math.ceil(m))

    @staticmethod
    def _optimal_hash_count(m: int, n: int) -> int:
        """Calculate the optimal number of hash functions."""
        k = (m / n) * math.log(2)
        return max(1, int(round(k)))

    def _get_bit(self, index: int) -> bool:
        """Get the value at the specified position of the bit array."""
        byte_index = index // 8
        bit_offset = index % 8
        return bool(self.bit_array[byte_index] & (1 << bit_offset))

    def _set_bit(self, index: int):
        """Set the specified position of the bit array to 1."""
        byte_index = index // 8
        bit_offset = index % 8
        self.bit_array[byte_index] |= (1 << bit_offset)

    def _hashes(self, item: str) -> list:
        """Generate k hash values from an element.

        Why double hashing is used:
        Preparing k independent hash functions is cumbersome.
        Instead, from two hash values h1 and h2,
        gi(x) = h1(x) + i*h2(x) (mod m)  (i = 0, 1, ..., k-1)
        generates k hash values.
        Kirsch & Mitzenmacher (2006) proved that this method
        achieves the same false positive rate as k independent
        hash functions.
        """
        h = hashlib.md5(str(item).encode()).hexdigest()
        h1 = int(h[:16], 16)
        h2 = int(h[16:], 16)
        return [(h1 + i * h2) % self.size for i in range(self.num_hashes)]

    def add(self, item: str):
        """Add an element."""
        for pos in self._hashes(item):
            self._set_bit(pos)
        self.count += 1

    def might_contain(self, item: str) -> bool:
        """Check for element existence.

        True: "Probably exists" (possibility of false positive)
        False: "Definitely does not exist" (no false negatives)
        """
        return all(self._get_bit(pos) for pos in self._hashes(item))

    def theoretical_fpr(self) -> float:
        """Theoretical false positive rate for the current state."""
        if self.count == 0:
            return 0.0
        exponent = -self.num_hashes * self.count / self.size
        return (1 - math.exp(exponent)) ** self.num_hashes

    def memory_usage_bytes(self) -> int:
        """Memory usage (bytes)."""
        return len(self.bit_array)

    def info(self) -> dict:
        """Return filter state information."""
        return {
            "Bit array size (m)": self.size,
            "Number of hash functions (k)": self.num_hashes,
            "Elements added": self.count,
            "Memory usage": f"{self.memory_usage_bytes():,} bytes",
            "Theoretical FPR": f"{self.theoretical_fpr():.6f}",
            "Target FPR": f"{self.target_fpr:.6f}",
        }


def verify_bloom_filter():
    """Verify Bloom filter behavior."""
    print("=" * 60)
    print("Bloom Filter Verification")
    print("=" * 60)

    # Build with 10000 elements, 1% false positive rate
    bf = BloomFilter(expected_items=10000, false_positive_rate=0.01)

    # Filter info
    for key, val in bf.info().items():
        print(f"  {key}: {val}")
    print()

    # Add 10000 elements
    added = set()
    for i in range(10000):
        word = f"word_{i}"
        bf.add(word)
        added.add(word)

    # Verify no false negatives (added elements must always return True)
    false_negatives = 0
    for word in added:
        if not bf.might_contain(word):
            false_negatives += 1
    print(f"False negatives: {false_negatives} (theoretically 0)")

    # Measure false positive rate
    false_positives = 0
    test_count = 100000
    for i in range(test_count):
        word = f"test_{i}"
        if word not in added and bf.might_contain(word):
            false_positives += 1

    actual_fpr = false_positives / test_count
    print(f"Expected FPR: {actual_fpr:.4f} "
          f"(theoretical: {bf.theoretical_fpr():.4f})")

    # Space comparison with hash set
    import sys
    hash_set_size = sys.getsizeof(added)
    bloom_size = bf.memory_usage_bytes()
    print(f"\nSpace comparison:")
    print(f"  Hash set:      {hash_set_size:>10,} bytes")
    print(f"  Bloom filter:  {bloom_size:>10,} bytes")
    print(f"  Reduction:     {(1 - bloom_size / hash_set_size) * 100:.1f}%")


if __name__ == "__main__":
    verify_bloom_filter()
```

### 5.5 Bloom Filter Variations

```
Bloom filter variation comparison:

+---------------------+----------+----------+------------------+
| Variation           | Deletion | Counting | Primary Use      |
+---------------------+----------+----------+------------------+
| Standard Bloom      | No       | No       | General membership|
| Counting Bloom      | Yes      | Yes      | Dynamic sets     |
| Cuckoo Filter       | Yes      | No       | When deletion is |
|                     |          |          | needed           |
| Quotient Filter     | Yes      | No       | SSD-friendly     |
| Scalable Bloom      | No       | No       | Unknown element  |
|                     |          |          | count            |
+---------------------+----------+----------+------------------+

Counting Bloom Filter bit array:
  Standard: 1 bit per position -> [0][1][1][0][1][0]...
  Counting: 4 bits per position -> [0][3][2][0][1][0]...
  -> Deletion is possible by decrementing counters
  -> However, space is 4x larger

Cuckoo Filter:
  Can be more space-efficient than Bloom filters in some cases.
  Supports deletion and has fast lookups.
  However, insertion requires relocation and worst cases exist.
```

---

## 6. Representative Tradeoff Pattern Collection

### 6.1 Hash Table vs. Linear Search — Duplicate Detection

```python
"""
Duplicate detection in arrays: Space vs. time tradeoff

This pattern is one of the most frequently asked in coding interviews.
It demonstrates the classic tradeoff: "Using O(n) space can improve
time from O(n^2) to O(n)."
"""

import time
import random


def has_duplicate_brute_force(arr: list) -> bool:
    """Duplicate detection by comparing all pairs.

    Time: O(n^2) — compare all pairs
    Space: O(1)   — no additional memory

    Why O(n^2):
    The outer loop runs n times, the inner loop averages n/2 times.
    Total: n*(n-1)/2 comparisons = O(n^2).
    """
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] == arr[j]:
                return True
    return False


def has_duplicate_sort(arr: list) -> bool:
    """Sort then compare adjacent elements.

    Time: O(n log n) — sorting time
    Space: O(n)       — for sorting (Timsort)

    Why sorting makes adjacent comparison sufficient:
    After sorting, identical values are adjacent. Therefore,
    checking only adjacent pairs detects all duplicates.
    """
    sorted_arr = sorted(arr)
    for i in range(len(sorted_arr) - 1):
        if sorted_arr[i] == sorted_arr[i + 1]:
            return True
    return False


def has_duplicate_hash_set(arr: list) -> bool:
    """Duplicate detection using a hash set.

    Time: O(n) — process each element once
    Space: O(n) — hash set size

    Why O(n):
    The set's `in` and `add` operations run in average O(1)
    due to the hash table. Processing n elements once each
    gives O(n) overall.
    """
    seen = set()
    for x in arr:
        if x in seen:
            return True
        seen.add(x)
    return False


def benchmark_duplicate_detection():
    """Performance comparison of the three methods."""
    random.seed(42)

    print("Duplicate Detection Benchmark")
    print("=" * 65)

    for n in [1000, 5000, 10000]:
        arr = list(range(n))
        arr[-1] = 0  # Make the last element a duplicate (worst case)

        results = {}
        for name, func in [
            ("All pairs O(n^2)",        has_duplicate_brute_force),
            ("Sort + compare O(n log n)", has_duplicate_sort),
            ("Hash set O(n)",           has_duplicate_hash_set),
        ]:
            start = time.perf_counter()
            result = func(arr[:])  # Pass a copy
            elapsed = time.perf_counter() - start
            results[name] = elapsed
            assert result is True

        print(f"\nn = {n:,}")
        for name, elapsed in results.items():
            print(f"  {name:30s}: {elapsed:.6f}s")


if __name__ == "__main__":
    benchmark_duplicate_detection()
```

### 6.2 Two Sum Problem — A Classic Interview Tradeoff Example

```python
"""
Two Sum: Find a pair in an array that sums to target

This problem is LeetCode #1 and one of the most frequently asked
in interviews. Three approaches exist, each demonstrating a
different tradeoff.
"""


def two_sum_brute_force(nums: list, target: int) -> tuple:
    """All-pair search.
    Time: O(n^2), Space: O(1)
    """
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return (i, j)
    return None


def two_sum_sort(nums: list, target: int) -> tuple:
    """Sort + binary search / two pointers.
    Time: O(n log n), Space: O(n) (storing original indices)

    Note: Sorting changes indices, so original indices must
    be stored separately.
    """
    indexed = sorted(enumerate(nums), key=lambda x: x[1])
    left, right = 0, len(indexed) - 1
    while left < right:
        current_sum = indexed[left][1] + indexed[right][1]
        if current_sum == target:
            return (indexed[left][0], indexed[right][0])
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return None


def two_sum_hash(nums: list, target: int) -> tuple:
    """Hash map.
    Time: O(n), Space: O(n)

    Why a single pass suffices:
    For each element nums[i], if target - nums[i] is already
    registered in the map, the pair is found.
    If not, register nums[i] and move on.
    Completed in a single scan.
    """
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return None


def test_two_sum():
    """Verify correctness and performance of the three methods."""
    test_cases = [
        ([2, 7, 11, 15], 9, {(0, 1)}),
        ([3, 2, 4], 6, {(1, 2)}),
        ([3, 3], 6, {(0, 1)}),
    ]

    for nums, target, expected in test_cases:
        for name, func in [
            ("brute_force", two_sum_brute_force),
            ("sort",        two_sum_sort),
            ("hash",        two_sum_hash),
        ]:
            result = func(nums, target)
            assert result is not None, f"{name}: No solution found"
            pair = tuple(sorted(result))
            assert pair in expected, \
                f"{name}: {pair} is not in expected {expected}"

    print("All test cases passed")


if __name__ == "__main__":
    test_two_sum()
```

---

## 7. Comparison Tables

### Table 1: Comprehensive Comparison of Tradeoff Techniques

| Technique | Time Improvement | Space Cost | Prerequisites | Application | Risk |
|-----------|-----------------|------------|---------------|-------------|------|
| Memoization | Exponential -> Polynomial | O(subproblem count) | Overlapping subproblems, referential transparency | DP, recursive computations | Stack overflow |
| LRU Cache | Query O(1) | O(capacity) | Temporal locality of access | API responses, DB queries | Cache pollution |
| Lookup Table | O(f) -> O(1) | O(input space) | Input space is finite and small | Trig functions, bit operations | L1 cache overflow |
| Hash Set | O(n^2) -> O(n) | O(n) | Hashable elements | Duplicate detection, membership testing | Hash collisions |
| Bloom Filter | Membership test O(k) | O(m), m << n | False positives are acceptable | Large-scale membership testing | False positives, no deletion |
| Sort Preprocessing | Query O(log n) | O(1) to O(n) | Static data | Repeated searches | Sort cost O(n log n) |
| Inverted Index | Search O(1) | O(n) | Key extraction is possible | Full-text search, DB indexing | Update cost |

### Table 2: Bloom Filter vs. Other Data Structures

| Property | Hash Set | Bloom Filter | Cuckoo Filter | Sorted Array |
|----------|----------|-------------|---------------|-------------|
| Space Complexity | O(n) | O(m), m << n | O(n) but small constant | O(n) |
| Add | O(1) average | O(k) | O(1) average | O(n) |
| Lookup | O(1) average | O(k) | O(1) average | O(log n) |
| Delete | O(1) average | Not possible (standard) | O(1) average | O(n) |
| False Positives | None | Yes (controllable) | Yes (controllable) | None |
| False Negatives | None | None | None | None |
| Element Retrieval | Possible | Not possible | Fingerprint only | Possible |
| Space for 10^7 elements | ~400MB | ~12MB (FPR=1%) | ~80MB | ~80MB |

### Table 3: Cache Strategy Selection Guide

| Situation | Recommended Strategy | Reason |
|-----------|---------------------|--------|
| Few subproblems, all needed | Memoization (maxsize=None) | Space is small even when all results are retained |
| Vast input patterns but with locality | LRU Cache | Access is biased toward recent items |
| Fixed popular items | LFU Cache | Prioritize retaining high-frequency items |
| Data freshness is important | TTL Cache | Automatically invalidate stale data |
| Small, fixed input space | Lookup Table | O(1) reference is guaranteed |
| Large-scale membership testing | Bloom Filter | Space efficiency is orders of magnitude better |

---

## 8. Anti-Patterns

### Anti-Pattern 1: Applying Memoization to Recursion Without Overlapping Subproblems

```python
"""
Anti-pattern: Applying memoization to binary search

Why this is pointless:
Binary search has a different search range (lo, hi pair) for each
recursive call. This means the function is never called twice with
the same arguments. Even if results are stored in the cache, there
are zero hits, making it a waste of space.

General principle: Memoization is effective only when "the same
subproblem appears multiple times." Even for recursive algorithms,
divide-and-conquer approaches (binary search, merge sort, quicksort)
where each subproblem is called only once do not need memoization.
"""

from functools import lru_cache


# BAD: Pointless memoization (all calls are cache misses)
@lru_cache(maxsize=None)
def binary_search_bad(arr_tuple: tuple, target: int,
                      lo: int, hi: int) -> int:
    """Binary search with memoization — a pointless example.

    Problems:
    1. All combinations of (arr_tuple, target, lo, hi) are unique
       -> Cache hit rate = 0%
    2. Hashing arr_tuple takes O(n)
       -> Slower than without memoization
    3. Cache wastes memory for O(log n) entries
    """
    if lo > hi:
        return -1
    mid = (lo + hi) // 2
    if arr_tuple[mid] == target:
        return mid
    elif arr_tuple[mid] < target:
        return binary_search_bad(arr_tuple, target, mid + 1, hi)
    else:
        return binary_search_bad(arr_tuple, target, lo, mid - 1)


# GOOD: Straightforward implementation without memoization
def binary_search_good(arr: list, target: int) -> int:
    """Straightforward binary search — iterative version.

    Why the iterative version is better:
    1. No recursion overhead
    2. Stack space O(1) (recursive version is O(log n))
    3. Especially important in Python which lacks tail call optimization
    """
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def demonstrate_useless_memoization():
    """Demonstrate the cost of useless memoization."""
    import time

    arr = list(range(100000))
    arr_tuple = tuple(arr)
    target = 99999

    # Clear cache
    binary_search_bad.cache_clear()

    start = time.perf_counter()
    for _ in range(100):
        binary_search_bad.cache_clear()
        binary_search_bad(arr_tuple, target, 0, len(arr) - 1)
    elapsed_bad = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(100):
        binary_search_good(arr, target)
    elapsed_good = time.perf_counter() - start

    info = binary_search_bad.cache_info()
    print("Useless Memoization Example")
    print(f"  Memoized:       {elapsed_bad:.4f}s")
    print(f"  Straightforward: {elapsed_good:.4f}s")
    print(f"  Cache hits: {info.hits} "
          f"(misses: {info.misses})")
    print(f"  -> Hit rate 0%, memoization is completely wasteful")


if __name__ == "__main__":
    demonstrate_useless_memoization()
```

### Anti-Pattern 2: Underestimating Table Size

```python
"""
Anti-pattern: Lookup table with overestimated input space

Why this is dangerous:
If the table is too large:
1. Out of memory (OOM) crashes the process
2. The table doesn't fit in L1/L2 cache, making lookups slow
   -> What should be a table lookup becomes a main memory access every time
   -> A reversal can occur where it becomes slower than the original computation
3. Build time becomes long, delaying application startup
"""

import sys


def demonstrate_table_size_problem():
    """Example of table size estimation failure."""
    print("Table Size Estimation")
    print("=" * 50)

    sizes = [
        ("8-bit",  2**8,   "Optimal for popcount etc."),
        ("16-bit", 2**16,  "Still within acceptable range"),
        ("20-bit", 2**20,  "May fit in L2 cache"),
        ("24-bit", 2**24,  "16MB — L3 cache limit"),
        ("32-bit", 2**32,  "16GB — exceeds typical PC RAM"),
    ]

    for name, size, comment in sizes:
        # For a list of ints (Python int is 28+ bytes)
        memory_mb = size * 28 / (1024 * 1024)
        feasible = "OK" if memory_mb < 100 else "DANGER" if memory_mb < 10000 else "IMPOSSIBLE"
        print(f"  {name:10s}: {size:>15,} entries  "
              f"~{memory_mb:>10,.0f}MB  [{feasible}] {comment}")

    print()
    print("Countermeasure: Split tables")
    print("  To compute popcount of a 32-bit value:")
    print("  BAD:  table[2^32] = 16GB table")
    print("  GOOD: table[2^8] = 256 entries x 4 lookups = effectively O(1)")


# BAD: Table that consumes massive memory
def create_bad_table():
    """DO NOT run this — it would consume 16GB+ of memory."""
    # table = [0] * (2**32)  # ~16GB — running this may crash
    print("This code is dangerous, so execution is skipped")


# GOOD: Split table achieving the same result
SMALL_TABLE = [0] * 256
for i in range(256):
    SMALL_TABLE[i] = (i & 1) + SMALL_TABLE[i >> 1]


def popcount_safe(n: int) -> int:
    """popcount using a safe split table.

    Why 4 table lookups are equivalent to 1 giant table lookup:
    A 32-bit integer can be split into 4 eight-bit chunks.
    The popcount of each chunk can be computed independently
    and summed to get the total popcount. Four O(1) lookups
    are still O(1). Moreover, the table is guaranteed to fit
    in L1 cache, giving better cache efficiency than a giant table.
    """
    return (SMALL_TABLE[n & 0xFF] +
            SMALL_TABLE[(n >> 8) & 0xFF] +
            SMALL_TABLE[(n >> 16) & 0xFF] +
            SMALL_TABLE[(n >> 24) & 0xFF])


if __name__ == "__main__":
    demonstrate_table_size_problem()
```

### Anti-Pattern 3: Forgetting to Invalidate the Cache

```python
"""
Anti-pattern: Not invalidating the cache for changing data

Why this is dangerous:
When cached values become stale, "incorrect results" are returned.
This is a silent bug (quietly returns wrong results without errors),
making debugging extremely difficult.

Phil Karlton's famous quote:
"There are only two hard things in Computer Science:
 cache invalidation and naming things."
"""

from functools import lru_cache
import time


# BAD: Caching a function that depends on external state
EXCHANGE_RATE = {"USD_JPY": 150.0}  # Exchange rates constantly fluctuate


@lru_cache(maxsize=128)
def convert_usd_to_jpy_bad(amount: float) -> float:
    """BAD: Returns stale cached value even when the exchange rate changes."""
    return amount * EXCHANGE_RATE["USD_JPY"]


# GOOD: TTL cache to guarantee freshness
class CurrencyConverter:
    """Currency converter — TTL cache version.

    Why TTL is necessary:
    Exchange rates fluctuate on a per-second basis. However,
    calling the API every time is costly (rate limits, latency).
    By setting a TTL, you use "a rate at most N seconds old,"
    reducing API calls while ensuring a degree of freshness.
    """

    def __init__(self, ttl_seconds: float = 60.0):
        self.ttl = ttl_seconds
        self._cache = {}
        self._timestamps = {}

    def get_rate(self, pair: str) -> float:
        """Get exchange rate (with cache)."""
        now = time.time()
        if pair in self._cache:
            age = now - self._timestamps[pair]
            if age < self.ttl:
                return self._cache[pair]

        # Cache miss or expired -> API call (simulated)
        rate = self._fetch_rate_from_api(pair)
        self._cache[pair] = rate
        self._timestamps[pair] = now
        return rate

    def _fetch_rate_from_api(self, pair: str) -> float:
        """Fetch rate from API (simulated)."""
        # In practice, this would call an external API
        return EXCHANGE_RATE.get(pair, 1.0)

    def convert(self, amount: float, pair: str = "USD_JPY") -> float:
        """Convert currency."""
        return amount * self.get_rate(pair)


def demonstrate_cache_invalidation():
    """Demonstrate the importance of cache invalidation."""
    print("Cache Invalidation Problem")
    print("=" * 50)

    # BAD version
    convert_usd_to_jpy_bad.cache_clear()
    result1 = convert_usd_to_jpy_bad(100.0)
    print(f"BAD: $100 = JPY {result1:.0f} (rate: {EXCHANGE_RATE['USD_JPY']})")

    EXCHANGE_RATE["USD_JPY"] = 140.0  # Rate changes
    result2 = convert_usd_to_jpy_bad(100.0)
    print(f"BAD: $100 = JPY {result2:.0f} (stale value returned after rate change!)")

    # GOOD version
    converter = CurrencyConverter(ttl_seconds=0.1)  # TTL 0.1s
    result3 = converter.convert(100.0)
    print(f"GOOD: $100 = JPY {result3:.0f}")

    EXCHANGE_RATE["USD_JPY"] = 130.0
    time.sleep(0.15)  # Wait for TTL to expire
    result4 = converter.convert(100.0)
    print(f"GOOD: $100 = JPY {result4:.0f} (fetched new rate after TTL expiry)")


if __name__ == "__main__":
    demonstrate_cache_invalidation()
```

---

## 9. Edge Case Analysis

### Edge Case 1: Memoization and Python's Recursion Limit

```python
"""
Edge case: Hitting the recursion limit with memoization

Python's default recursion limit is 1000.
Calling fib(1000) directly with memoized Fibonacci
will raise a RecursionError.

Why this is a problem:
Memoization relies on recursion, so problems requiring deep
recursion hit Python's recursion limit as a bottleneck.

Solutions:
1. Raise the limit with sys.setrecursionlimit() (not recommended)
   -> Risk of stack overflow
2. Rewrite as bottom-up DP (recommended)
   -> Uses a loop so is not constrained by the recursion limit
3. Iterative deepening: call with small values first (compromise)
   -> Cache warms up so deep recursion doesn't occur
"""

import sys


def fib_memo_recursive(n: int, memo: dict = None) -> int:
    """Memoized Fibonacci (recursive version)."""
    if memo is None:
        memo = {}
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib_memo_recursive(n - 1, memo) + \
              fib_memo_recursive(n - 2, memo)
    return memo[n]


def fib_iterative_warmup(n: int) -> int:
    """Iteratively warm up the cache, then compute large values.

    Why this works:
    Calling fib(500) first caches fib(0) through fib(500).
    When fib(1000) is then called, fib(499) and fib(500)
    are retrieved from the cache during computation of fib(501) through fib(1000).
    The maximum recursion depth is 500, which fits within the limit.
    """
    memo = {}
    step = 400  # Step size well below the recursion limit
    for start in range(0, n + 1, step):
        target = min(start + step, n)
        fib_memo_recursive(target, memo)
    return memo.get(n, fib_memo_recursive(n, memo))


def fib_bottom_up_space_optimized(n: int) -> int:
    """Bottom-up DP (space-optimized version).

    Why O(1) space suffices:
    Computing fib(i) requires only fib(i-1) and fib(i-2).
    Values before those are not needed, so two variables suffice.
    """
    if n <= 1:
        return n
    prev2, prev1 = 0, 1
    for _ in range(2, n + 1):
        prev2, prev1 = prev1, prev2 + prev1
    return prev1


def demonstrate_recursion_limit():
    """Demonstrate the recursion limit problem and solutions."""
    print("Recursion Limit Edge Case")
    print("=" * 50)

    # Current recursion limit
    print(f"Current recursion limit: {sys.getrecursionlimit()}")

    # Small values are fine
    memo = {}
    result = fib_memo_recursive(100, memo)
    print(f"fib(100) = {len(str(result))} digits [OK]")

    # Large values trigger RecursionError
    try:
        memo2 = {}
        fib_memo_recursive(5000, memo2)
        print("fib(5000) [OK — recursion limit is large enough in this environment]")
    except RecursionError:
        print("fib(5000) [RecursionError occurred]")

    # Avoid with iterative warmup
    result_warmup = fib_iterative_warmup(5000)
    print(f"fib(5000) (warmup version) = {len(str(result_warmup))} digits [OK]")

    # Avoid with bottom-up
    result_bu = fib_bottom_up_space_optimized(5000)
    print(f"fib(5000) (bottom-up version) = {len(str(result_bu))} digits [OK]")

    assert result_warmup == result_bu, "Results do not match"


if __name__ == "__main__":
    demonstrate_recursion_limit()
```

### Edge Case 2: Bloom Filter Saturation

```python
"""
Edge case: Adding more elements than expected to a Bloom filter causes
the false positive rate to spike

Why this matters:
A Bloom filter's size is determined in advance based on the expected
element count. Adding more than expected causes almost all bits in the
bit array to become 1 ("saturation"), and the false positive rate
approaches 100%. This effectively makes the filter useless.

Countermeasures:
1. Estimate the element count in advance and size with margin
2. Use a Scalable Bloom Filter (auto-expands with element count)
3. Detect saturation and switch to a new filter
"""

import math
import hashlib


class MonitoredBloomFilter:
    """Bloom filter with saturation monitoring."""

    def __init__(self, expected_items: int, false_positive_rate: float = 0.01):
        self.expected_items = expected_items
        self.target_fpr = false_positive_rate
        m = int(-expected_items * math.log(false_positive_rate) /
                (math.log(2) ** 2))
        self.size = max(m, 64)
        self.num_hashes = max(1, int(round(
            (self.size / expected_items) * math.log(2)
        )))
        self.bit_array = bytearray((self.size + 7) // 8)
        self.count = 0
        self._set_bits = 0  # Number of bits set to 1

    def _hashes(self, item: str) -> list:
        h = hashlib.md5(str(item).encode()).hexdigest()
        h1 = int(h[:16], 16)
        h2 = int(h[16:], 16)
        return [(h1 + i * h2) % self.size for i in range(self.num_hashes)]

    def _get_bit(self, idx: int) -> bool:
        return bool(self.bit_array[idx // 8] & (1 << (idx % 8)))

    def _set_bit(self, idx: int):
        byte_idx = idx // 8
        bit_off = idx % 8
        if not (self.bit_array[byte_idx] & (1 << bit_off)):
            self._set_bits += 1
            self.bit_array[byte_idx] |= (1 << bit_off)

    def add(self, item: str):
        for pos in self._hashes(item):
            self._set_bit(pos)
        self.count += 1

    def might_contain(self, item: str) -> bool:
        return all(self._get_bit(pos) for pos in self._hashes(item))

    def saturation(self) -> float:
        """Return the saturation rate (0.0 to 1.0)."""
        return self._set_bits / self.size

    def estimated_fpr(self) -> float:
        """Estimated false positive rate based on current saturation."""
        sat = self.saturation()
        return sat ** self.num_hashes

    def is_saturated(self, threshold: float = 0.5) -> bool:
        """Determine whether the filter is saturated.

        Why 0.5 as the threshold:
        With the optimal number of hash functions, the probability
        of each bit being 1 is exactly 0.5. Beyond this, the false
        positive rate rises rapidly.
        """
        return self.saturation() > threshold


def demonstrate_saturation():
    """Demonstrate the progression of saturation and FPR increase."""
    print("Bloom Filter Saturation")
    print("=" * 65)
    print(f"{'Added':>10s}  {'Saturation':>10s}  {'Est. FPR':>10s}  "
          f"{'Expected Act. FPR':>17s}  {'Status':>8s}")
    print("-" * 65)

    bf = MonitoredBloomFilter(expected_items=1000, false_positive_rate=0.01)

    checkpoints = [100, 500, 1000, 2000, 5000, 10000]

    for target in checkpoints:
        while bf.count < target:
            bf.add(f"item_{bf.count}")

        # Measure false positive rate
        fp = 0
        tests = 10000
        for i in range(tests):
            probe = f"probe_{i}"
            if bf.might_contain(probe):
                fp += 1
        actual_fpr = fp / tests

        status = "Normal" if not bf.is_saturated() else "Saturated!"
        print(f"{bf.count:>10,d}  {bf.saturation():>10.1%}  "
              f"{bf.estimated_fpr():>10.4f}  "
              f"{actual_fpr:>17.4f}  {status:>8s}")


if __name__ == "__main__":
    demonstrate_saturation()
```

### Edge Case 3: Hash Table Resizing and Memory Spikes

```
Hash table (set/dict) memory usage is not constant:

  Memory ^
         |
  256KB  |              +-------------  <- Resize occurs
         |              |                 (table size doubles at 2/3 fill)
  128KB  |      +-------+
         |      |
   64KB  |  +---+
         |  |
   32KB  |--+
         |
         +-------------------------------> Element count
         0     1000   2000   3000   4000

  Problem: During resize, both old and new tables exist temporarily
  -> At worst, 3x the steady-state memory is consumed momentarily
  -> In memory-constrained environments, this spike needs attention

  Countermeasures:
  1. Predict initial size and pre-allocate (e.g., dict.fromkeys())
  2. If memory is tight, consider sorted array + binary search
  3. If the use case can substitute a Bloom filter, switch to one
```

---

## 10. Practical Design Decision Framework

### 10.1 Decision Flowchart

```
Space-time tradeoff selection flow:

  [Start]
    |
    v
  How many times is the same computation repeated?
    |
    +-- Only once -> No tradeoff needed (compute straightforwardly)
    |
    +-- Few times (< 10) -> Recomputation is cheap, keep as is
    |
    +-- Many times (10+) --+
                            v
                  How large is the input space?
                            |
                  +-- Small (< 10^6)
                  |     |
                  |     v
                  |   Can all patterns be precomputed?
                  |     +-- Yes -> Lookup table
                  |     +-- No  -> Memoization / LRU cache
                  |
                  +-- Medium (10^6 to 10^9)
                  |     |
                  |     v
                  |   Is there locality in access?
                  |     +-- Yes -> LRU / LFU cache
                  |     +-- No  -> Bloom filter (membership testing only)
                  |
                  +-- Huge (> 10^9)
                        |
                        v
                  Can false positives be tolerated?
                        +-- Yes -> Bloom filter
                        +-- No  -> External storage (DB, Redis)
```

### 10.2 Practical Decision Criteria

| Decision Axis | Choose Time Priority When | Choose Space Priority When |
|---------------|--------------------------|---------------------------|
| Latency requirements | Strict (e.g., p99 < 10ms) | Batch processing with delay tolerance |
| Memory cost per unit | Memory is cheap in the cloud | RAM constraints in embedded systems |
| Data volatility | Static data (reference tables) | Dynamic data (frequently updated) |
| Accuracy requirements | Exact results needed | Approximate/probabilistic is fine (Bloom) |
| Scale | Self-contained on a single machine | Distributed system with consistency challenges |

---

## 11. Exercises

### Beginner Level

**Exercise 1: Experience the Effect of Memoization**

The function `grid_paths(m, n)` below computes the number of paths from the top-left to the bottom-right of an m x n grid. Run the version without memoization, then implement the memoized version yourself and verify the performance difference.

```python
"""
Exercise 1: Apply memoization to grid path counting

Problem:
In an m x n grid, find the number of paths from the top-left (0,0) to
the bottom-right (m-1,n-1) when only moving right or down.

Hint:
- grid_paths(m, n) = grid_paths(m-1, n) + grid_paths(m, n-1)
- Base case: When m == 1 or n == 1, there is exactly 1 path
"""

import time


def grid_paths_naive(m: int, n: int) -> int:
    """Version without memoization. What is the time complexity? Think about it."""
    if m == 1 or n == 1:
        return 1
    return grid_paths_naive(m - 1, n) + grid_paths_naive(m, n - 1)


# TODO: Implement grid_paths_memo(m, n) with memoization
# def grid_paths_memo(m, n, memo=None):
#     ...


# Verification
if __name__ == "__main__":
    # Verify with small inputs
    print(f"grid_paths(3, 3) = {grid_paths_naive(3, 3)}")  # 6
    print(f"grid_paths(4, 4) = {grid_paths_naive(4, 4)}")  # 20

    # Experience the slowness of the naive version at m=15, n=15 (should take seconds)
    start = time.perf_counter()
    result = grid_paths_naive(15, 15)
    elapsed = time.perf_counter() - start
    print(f"grid_paths(15, 15) = {result}  ({elapsed:.3f}s)")
    # The memoized version should complete instantly
```

**Exercise 2: Building a Lookup Table**

Build a table that returns the "bit-reversed value" for each byte value 0-255, and implement a 32-bit integer bit-reversal function.

```python
"""
Exercise 2: Bit reversal table

Problem:
Build an 8-bit bit reversal lookup table.
Example: 0b11010010 -> 0b01001011  (swap high and low bits)

Hint:
reverse_bits_8(n) reverses the 8 bits of n.
0b10110000 -> 0b00001101

Precompute all 256 patterns and store them in a table.
"""


def reverse_bits_8(n: int) -> int:
    """Bit reversal of an 8-bit value (naive version)."""
    result = 0
    for _ in range(8):
        result = (result << 1) | (n & 1)
        n >>= 1
    return result


# TODO: Build the table
# REVERSE_TABLE = [reverse_bits_8(i) for i in range(256)]

# TODO: Implement 32-bit integer bit reversal using table lookup
# def reverse_bits_32(n):
#     ...
```

### Intermediate Level

**Exercise 3: Build Your Own LRU Cache**

Implement an LRU cache using a doubly linked list and a hash map, without using Python's `OrderedDict`. Both `get(key)` and `put(key, value)` must operate in O(1).

```python
"""
Exercise 3: Custom LRU cache implementation

Requirements:
1. get(key): Retrieve from cache in O(1). Return -1 on miss.
2. put(key, value): Add/update in O(1). Evict oldest on capacity overflow.

Hints:
- Hash map: key -> reference to node (O(1) access)
- Doubly linked list: manage access order (O(1) move/delete)
- Using dummy head and tail nodes simplifies boundary condition handling
"""


class Node:
    """Doubly linked list node."""
    def __init__(self, key: int = 0, value: int = 0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCacheManual:
    """LRU cache — custom implementation."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # key -> Node

        # Dummy nodes (sentinels)
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    # TODO: Implement _remove(node), _add_to_front(node), get(key), put(key, value)


# Test
if __name__ == "__main__":
    cache = LRUCacheManual(2)
    cache.put(1, 1)
    cache.put(2, 2)
    assert cache.get(1) == 1      # Access key=1 -> becomes most recent
    cache.put(3, 3)               # Capacity exceeded -> key=2 is evicted
    assert cache.get(2) == -1     # key=2 has been evicted
    cache.put(4, 4)               # Capacity exceeded -> key=1 is evicted
    assert cache.get(1) == -1
    assert cache.get(3) == 3
    assert cache.get(4) == 4
    print("All tests passed")
```

### Advanced Level

**Exercise 4: Implementing a Counting Bloom Filter**

Implement a Counting Bloom Filter that adds "deletion" capability to a standard Bloom filter. Use 4-bit counters at each position, incrementing on add and decrementing on delete.

```python
"""
Exercise 4: Counting Bloom Filter

A standard Bloom filter cannot delete (resetting a bit to 0 would lose
information about other elements). A Counting Bloom Filter solves this
by maintaining counters at each position.

Requirements:
1. add(item): Increment counters at each hash position
2. remove(item): Decrement counters at each hash position
3. might_contain(item): Check if all counters are greater than 0
4. Handle counter overflow (> 15)

Note:
- Removing an element that was never added makes counters inconsistent
- This is an inherent limitation of Bloom filters that the Counting
  Bloom Filter cannot fully resolve
"""


class CountingBloomFilter:
    """Counting Bloom Filter skeleton."""

    def __init__(self, size: int = 1000, num_hashes: int = 3):
        self.size = size
        self.num_hashes = num_hashes
        # 4-bit counters -> could pack 2 per byte
        # Here simplified as a list
        self.counters = [0] * size

    # TODO: Implement _hashes(item), add(item), remove(item),
    #       might_contain(item)
    # Hint: Ensure counter doesn't go below 0 on remove


if __name__ == "__main__":
    cbf = CountingBloomFilter(size=10000, num_hashes=5)
    cbf.add("apple")
    cbf.add("banana")
    assert cbf.might_contain("apple") is True
    assert cbf.might_contain("banana") is True

    cbf.remove("apple")
    assert cbf.might_contain("apple") is False  # Deletion successful
    assert cbf.might_contain("banana") is True   # Unaffected
    print("Counting Bloom Filter test passed")
```

**Exercise 5: Space-Optimized DP — Longest Common Subsequence**

Optimize the space of the Longest Common Subsequence (LCS) DP table. Normally O(m*n) space is required, but it can be reduced to O(min(m,n)).

```python
"""
Exercise 5: Space optimization of LCS

Standard LCS DP table:
     ""  a  b  c  d  e
  "" [0, 0, 0, 0, 0, 0]
  a  [0, 1, 1, 1, 1, 1]
  c  [0, 1, 1, 2, 2, 2]
  e  [0, 1, 1, 2, 2, 3]

Space O(m*n) -> memory issues for long strings

Hints:
- dp[i][j] depends only on dp[i-1][j], dp[i][j-1], and dp[i-1][j-1]
- Keeping only 2 rows suffices -> space O(n)
- With further optimization, 1 row + 1 variable gives O(n)
"""


def lcs_full_table(s1: str, s2: str) -> int:
    """Standard LCS — space O(m*n)."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


# TODO: Implement lcs_space_optimized(s1, s2) with space O(min(m,n))
# def lcs_space_optimized(s1, s2):
#     ...

if __name__ == "__main__":
    assert lcs_full_table("abcde", "ace") == 3
    assert lcs_full_table("abc", "def") == 0
    # assert lcs_space_optimized("abcde", "ace") == 3
    print("LCS test passed")
```

---

## 12. FAQ

### Q1: Should I Use Top-Down DP (Memoization) or Bottom-Up DP?

**A:** Both can solve the same problems, but they have different characteristics.

| Property | Top-Down (Memoization) | Bottom-Up |
|----------|----------------------|-----------|
| Subproblems computed | Only those needed | All of them |
| Implementation intuitiveness | Written naturally with recursion | Requires managing table indices |
| Recursion overhead | Present | None |
| Python recursion limit | Affected | Unaffected |
| Space optimization | Difficult | Possible (discard unneeded rows) |

**Recommendation:** Use top-down when first understanding and implementing a problem (it's easier to think about). Convert to bottom-up when performance is needed. In Python especially, bottom-up is safer due to the recursion limit.

### Q2: How Do You Control the False Positive Rate of a Bloom Filter?

**A:** It is controlled by the relationship between three parameters (m, n, k).

- **Increase m (bit array size)**: Reduces the false positive rate. Cost is increased memory.
- **Optimize k (number of hash functions)**: k = (m/n) * ln(2) is optimal. Too few or too many increases the false positive rate.
- **Estimate n (element count) in advance**: Underestimation leads to filter saturation.

Practical guidelines:

| False Positive Rate | Required bits/element (m/n) | Optimal k |
|--------------------|---------------------------|-----------|
| 10% | 4.8 | 3 |
| 1% | 9.6 | 7 |
| 0.1% | 14.4 | 10 |
| 0.01% | 19.2 | 13 |

Example: To achieve a 1% false positive rate for 10 million data items, m = 9.6 * 10^7 ~ 12 million bytes (~12MB) of memory is needed. Storing the same data in a hash set would require several hundred MB, so this represents roughly a 25x space reduction.

### Q3: Should Time or Space Be Prioritized?

**A:** General guidelines are as follows.

1. **First, check the time constraints**: Latency that directly impacts users (API response time, UI responsiveness) is often the most important. Memory can be purchased, but user patience cannot.

2. **Next, check the memory constraints**: When physical or economic memory constraints exist — embedded systems, mobile devices, container Memory Limits, cloud memory billing — consider prioritizing space.

3. **Consider scalability**: Think about which approach scales as data volume grows. An O(n) space technique requires 10x memory when n grows 10x, but an O(1) space technique is data-volume-independent.

4. **Perform cost estimation**: In cloud environments, compare the cost of memory increases with compute time costs (CPU time billing) to select the optimal point.

### Q4: Should the Default maxsize=128 of lru_cache Be Changed?

**A:** It depends on the use case.

- **DP problems (finite subproblems)**: Set `maxsize=None`. If all results are not retained, evicted values require recomputation, degrading complexity.
- **API caching (vast input patterns)**: Default 128 or `maxsize=256-1024` or so. Making it too large puts pressure on memory.
- **Numerical computation (float arguments)**: Due to floating-point hashing issues, intended cache hits may not occur. Consider rounding to integers or switching to table lookup.

### Q5: Should a Cuckoo Filter Be Used Instead of a Bloom Filter?

**A:** A Cuckoo Filter is advantageous in the following cases:

1. **Deletion is needed**: Bloom filters cannot delete (except the Counting variant), but Cuckoo Filters natively support deletion.
2. **Space efficiency**: When the false positive rate is 3% or less, Cuckoo Filters can be more space-efficient than Bloom filters.
3. **Lookup speed**: Cuckoo Filters complete lookups in at most 2 memory accesses and are cache-friendly.

However, Cuckoo Filters may require relocation (moving existing elements) during insertion, and in the worst case, insertions can fail. If insertions are frequent and failures are unacceptable, Bloom filters are safer.

---

## 13. Real-World Application Examples

### 13.1 Tradeoffs in Databases

```
Database read/write tradeoffs:

  B-Tree Index:
  +----------------------------------------------+
  |  Space: Index adds 10-30% extra space to data |
  |  Read: O(log n) — improved from O(n) without  |
  |        an index                                |
  |  Write: Index must also be updated on each     |
  |         INSERT/UPDATE                          |
  |                                                |
  |  Tradeoff:                                     |
  |  More indexes -> faster reads / slower writes  |
  |  Fewer indexes -> slower reads / faster writes |
  +----------------------------------------------+

  LSM-Tree (LevelDB, RocksDB):
  +----------------------------------------------+
  |  Write: In-memory MemTable -> fast             |
  |  Read: Scan multiple SSTables -> somewhat slow |
  |  Bloom filter: Attached to each SSTable        |
  |    -> Quickly determines "this key is not in   |
  |       this SSTable"                            |
  |    -> Skips unnecessary SSTable reads          |
  |    -> Dramatically improves read performance   |
  +----------------------------------------------+
```

### 13.2 Safe Browsing in Web Browsers

Google Chrome's Safe Browsing feature stores a list of malicious URLs as a Bloom filter within the browser. When a user accesses a URL, the local Bloom filter is checked first, and only if it determines "probably dangerous" is a query sent to Google's servers.

- Without Bloom filter: Every URL is queried to the server -> latency + privacy issues
- With Bloom filter: 99%+ of accesses need no server query -> fast + privacy protection

### 13.3 Cache Patterns in Redis

```
Cache-Aside pattern (most common):

  [Application]
       |
       +-- 1. get(key) --> [Redis Cache]
       |                      |
       |                      +-- Hit -> Return result (fast)
       |                      |
       |                      +-- Miss
       |                           |
       +-- 2. query(key) --> [Database]
       |                      |
       |                      +-- Result
       |
       +-- 3. set(key, result, TTL) --> [Redis Cache]

  Tradeoffs of this pattern:
  - Space: Additional cost for Redis memory
  - Time: On cache hit, DB access is completely skipped
  - Consistency: Stale data may be returned until TTL expires after DB update
  - Availability: On Redis failure, falls back to direct DB access
```

---


## FAQ

### Q1: What is the most important point in learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory, but by actually writing code and verifying behavior.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this applied in practice?

Knowledge of this topic is frequently used in everyday development work. It becomes particularly important during code reviews and architecture design.

---

## 14. Summary

### Core Principles

| Principle | Description |
|-----------|-------------|
| Space and time are interchangeable | Additional memory can reduce computation time, and vice versa |
| Detection of overlapping subproblems | The most important criterion for whether memoization/DP is effective |
| Table size estimation | Lookup tables are practical only when the input space is small |
| Tolerance for false positives | Bloom filters dramatically reduce space by tolerating false positives |
| Cache invalidation | Designing invalidation strategy is more important than choosing the caching strategy itself |
| Pareto frontier | No perfect tradeoff exists; choose the optimal point based on requirements |

### Quick Reference for Technique Selection

| Technique | When to Use | When to Avoid |
|-----------|-------------|---------------|
| Memoization | When overlapping subproblems exist | When each subproblem is called only once |
| lru_cache | When you want quick caching in Python | When arguments are mutable |
| Lookup table | When the input space is small and fixed | When the input space is huge |
| Bloom filter | Large-scale membership testing where false positives are acceptable | When false positives are fatal |
| Hash set | General membership testing and duplicate detection | When memory constraints are severe |
| TTL cache | When freshness of external data matters | When data is immutable |

---

## Recommended Next Guides

- [Hash Tables — Collision Resolution and Load Factor](../01-data-structures/03-hash-tables.md) — The data structure underlying hash sets and Bloom filters
- [Dynamic Programming — Memoization and Tables in Detail](../02-algorithms/04-dynamic-programming.md) — A deeper dive into memoization
- Cache Architecture — From CPU Cache to CDN — How caching works at the hardware level

---

## References

1. Bloom, B.H. (1970). "Space/time trade-offs in hash coding with allowable errors." *Communications of the ACM*, 13(7), 422-426. — The original paper on Bloom filters. A historic paper proposing a probabilistic data structure that dramatically improves space efficiency by tolerating false positives.

2. Cormen, T.H., Leiserson, C.E., Rivest, R.L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. — Chapter 14 "Dynamic Programming" systematically explains the tradeoffs between memoization and bottom-up DP. Chapter 11 "Hash Tables" provides foundational theory on hash functions and caching.

3. Mitzenmacher, M. & Upfal, E. (2017). *Probability and Computing: Randomization and Probabilistic Techniques in Algorithms and Data Analysis* (2nd ed.). Cambridge University Press. — Rigorous probabilistic analysis of Bloom filter false positive rates, and the theoretical foundation for variants including Counting Bloom Filters.

4. Kirsch, A. & Mitzenmacher, M. (2006). "Less Hashing, Same Performance: Building a Better Bloom Filter." *Proceedings of ESA 2006*, 456-467. — Paper proving that the number of hash functions needed for a Bloom filter can be reduced to 2. The theoretical basis for the double hashing implementation in this guide.

5. Fan, B., Andersen, D.G., Kaminsky, M., & Mitzenmacher, M. (2014). "Cuckoo Filter: Practically Better Than Bloom." *Proceedings of ACM CoNEXT 2014*. — Paper demonstrating cases where Cuckoo Filters outperform Bloom filters in practice. Proposes deletion support and space efficiency improvements.

6. Knuth, D.E. (1997). *The Art of Computer Programming, Volume 3: Sorting and Searching* (2nd ed.). Addison-Wesley. — A comprehensive reference on search algorithms including lookup tables and precomputation techniques.

---

*All code examples in this guide can be tested with Python 3.8 or later.*
*Benchmark results will vary depending on the execution environment (CPU, memory, Python version).*
