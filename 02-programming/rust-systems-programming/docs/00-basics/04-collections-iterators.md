# Collections and Iterators -- Rust's Functional Data Processing Pipelines

> By combining collections such as Vec and HashMap with the Iterator trait, you can build type-safe, zero-cost data processing pipelines.

---

## What You Will Learn in This Chapter

1. **Major collections** -- Understand when to use Vec, HashMap, HashSet, BTreeMap, and others
2. **The Iterator trait** -- Master the iterator protocol and the mechanics of lazy evaluation
3. **Iterator adapters** -- Learn data transformation through chaining map/filter/fold/collect
4. **Practical pipelines** -- Acquire techniques for building complex data processing in a type-safe way
5. **Performance characteristics** -- Understand the computational complexity of each collection and how to choose appropriately


## Prerequisites

Reading this guide will be more rewarding if you have the following knowledge:

- Basic programming knowledge
- Understanding of related foundational concepts
- A grasp of the contents of [Error Handling -- Type-Safe Error Handling Patterns in Rust](./03-error-handling.md)

---

## 1. Vec<T> -- Dynamic Array

### Example 1: Basic Operations on Vec

```rust
fn main() {
    // Creation
    let mut v: Vec<i32> = Vec::new();
    let v2 = vec![1, 2, 3, 4, 5]; // Initialize with a macro

    // Append
    v.push(10);
    v.push(20);
    v.push(30);

    // Access
    println!("Index: {}", v[0]);                 // 10 (may panic)
    println!("Safe: {:?}", v.get(99));           // None

    // Iteration
    for item in &v {
        println!("{}", item);
    }

    // Mutable iteration
    for item in &mut v {
        *item *= 2;
    }
    println!("{:?}", v); // [20, 40, 60]

    // Convenient methods
    println!("Length: {}, Empty?: {}", v.len(), v.is_empty());
    println!("Contains?: {}", v.contains(&20));

    let last = v.pop(); // Some(60)
    println!("pop: {:?}", last);
}
```

### Vec Memory Layout

```
  Stack                        Heap
  ┌──────────────┐           ┌────┬────┬────┬────┬────┐
  │ ptr ──────────────────── │ 20 │ 40 │ 60 │    │    │
  │ len = 3      │           └────┴────┴────┴────┴────┘
  │ capacity = 5 │                used          unused
  └──────────────┘

  When push exceeds capacity, a new region is allocated and
  all elements are copied (amortized O(1))
```

### 1.1 Advanced Operations on Vec

```rust
fn main() {
    let mut v = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];

    // Sort
    v.sort();
    println!("Sorted: {:?}", v); // [1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9]

    // Deduplicate (only effective when sorted)
    v.dedup();
    println!("Deduplicated: {:?}", v);   // [1, 2, 3, 4, 5, 6, 9]

    // retain: keep only elements that satisfy the condition
    v.retain(|&x| x % 2 == 0);
    println!("Even only: {:?}", v);   // [2, 4, 6]

    // extend: append from another iterator
    v.extend([8, 10, 12].iter());
    println!("After extend: {:?}", v);     // [2, 4, 6, 8, 10, 12]

    // split_off: split the vector into two
    let tail = v.split_off(3);
    println!("First half: {:?}", v);       // [2, 4, 6]
    println!("Second half: {:?}", tail);    // [8, 10, 12]

    // windows: sliding window
    let data = vec![1, 2, 3, 4, 5];
    for window in data.windows(3) {
        println!("Window: {:?}", window);
    }
    // [1, 2, 3], [2, 3, 4], [3, 4, 5]

    // chunks: fixed-size chunks
    for chunk in data.chunks(2) {
        println!("Chunk: {:?}", chunk);
    }
    // [1, 2], [3, 4], [5]
}
```

### 1.2 Capacity Management for Vec

```rust
fn main() {
    // Reserve capacity in advance (avoids reallocation)
    let mut v = Vec::with_capacity(1000);
    println!("len={}, capacity={}", v.len(), v.capacity());
    // len=0, capacity=1000

    for i in 0..1000 {
        v.push(i);
    }
    // No reallocation occurs

    // Release excess capacity
    v.shrink_to_fit();
    println!("After shrink: len={}, capacity={}", v.len(), v.capacity());

    // Capacity growth strategy:
    // Vec reserves 2x the current capacity
    // Cost of push: amortized O(1) (mostly O(1), O(n) only on reallocation)

    // Benchmark: with_capacity vs push only
    use std::time::Instant;

    let n = 1_000_000;

    let start = Instant::now();
    let mut v1 = Vec::new();
    for i in 0..n {
        v1.push(i);
    }
    let t1 = start.elapsed();

    let start = Instant::now();
    let mut v2 = Vec::with_capacity(n);
    for i in 0..n {
        v2.push(i);
    }
    let t2 = start.elapsed();

    println!("without capacity: {:?}", t1);
    println!("with capacity:    {:?}", t2);
    // with_capacity is faster (no reallocation)
}
```

### 1.3 Slices -- References to Vec

```rust
fn sum_slice(data: &[i32]) -> i32 {
    data.iter().sum()
}

fn find_max(data: &[i32]) -> Option<&i32> {
    data.iter().max()
}

fn main() {
    let v = vec![10, 20, 30, 40, 50];

    // Vec → slice (implicit conversion)
    let total = sum_slice(&v);
    println!("Total: {}", total); // 150

    // Sub-slice
    let middle = &v[1..4]; // [20, 30, 40]
    println!("Sub-slice: {:?}", middle);
    println!("Partial total: {}", sum_slice(middle)); // 90

    // Arrays can also be passed as slices
    let arr = [1, 2, 3, 4, 5];
    println!("Array total: {}", sum_slice(&arr)); // 15

    // Using find_max
    if let Some(max) = find_max(&v) {
        println!("Max: {}", max); // 50
    }

    // Empty slice
    let empty: &[i32] = &[];
    println!("Max of empty: {:?}", find_max(empty)); // None

    // Binary search on a slice (when sorted)
    let sorted = vec![1, 3, 5, 7, 9, 11, 13];
    match sorted.binary_search(&7) {
        Ok(index) => println!("7 is at position {}", index),
        Err(index) => println!("7 not found (insertion position: {})", index),
    }
}
```

---

## 2. HashMap<K, V>

### Example 2: Basic Operations on HashMap

```rust
use std::collections::HashMap;

fn main() {
    let mut scores: HashMap<String, u32> = HashMap::new();

    // Insert
    scores.insert("Tanaka".to_string(), 85);
    scores.insert("Suzuki".to_string(), 92);
    scores.insert("Sato".to_string(), 78);

    // Access
    if let Some(score) = scores.get("Tanaka") {
        println!("Tanaka's score: {}", score);
    }

    // entry API (insert if the key does not exist)
    scores.entry("Yamada".to_string()).or_insert(0);
    *scores.entry("Tanaka".to_string()).or_insert(0) += 10;

    // Iteration
    for (name, score) in &scores {
        println!("{}: {}", name, score);
    }

    // The classic word-count pattern
    let text = "hello world hello rust hello";
    let mut word_count = HashMap::new();
    for word in text.split_whitespace() {
        *word_count.entry(word).or_insert(0) += 1;
    }
    println!("{:?}", word_count);
    // {"hello": 3, "world": 1, "rust": 1}
}
```

### 2.1 Advanced HashMap Patterns

```rust
use std::collections::HashMap;

fn main() {
    // Build from an iterator
    let teams: HashMap<&str, u32> = vec![
        ("Red", 10),
        ("Blue", 20),
        ("Green", 15),
    ].into_iter().collect();

    println!("{:?}", teams);

    // Advanced use of the entry API
    let mut cache: HashMap<String, Vec<String>> = HashMap::new();

    // or_insert_with: lazy initialization
    cache.entry("users".to_string())
        .or_insert_with(Vec::new)
        .push("Alice".to_string());

    cache.entry("users".to_string())
        .or_insert_with(Vec::new)
        .push("Bob".to_string());

    println!("users: {:?}", cache.get("users"));
    // Some(["Alice", "Bob"])

    // and_modify + or_insert: update if it exists, insert if it does not
    let mut counter: HashMap<&str, i32> = HashMap::new();
    let words = vec!["hello", "world", "hello", "rust", "hello"];

    for word in &words {
        counter.entry(word)
            .and_modify(|count| *count += 1)
            .or_insert(1);
    }
    println!("Counts: {:?}", counter);

    // Merging HashMaps
    let mut map1: HashMap<&str, i32> = [("a", 1), ("b", 2)].into();
    let map2: HashMap<&str, i32> = [("b", 10), ("c", 3)].into();

    for (key, value) in map2 {
        map1.entry(key)
            .and_modify(|v| *v += value)
            .or_insert(value);
    }
    println!("Merged: {:?}", map1); // {"a": 1, "b": 12, "c": 3}
}
```

### 2.2 HashMap with Custom Keys

```rust
use std::collections::HashMap;

// HashMap keys require Eq + Hash
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Coordinate {
    x: i32,
    y: i32,
}

impl Coordinate {
    fn new(x: i32, y: i32) -> Self {
        Coordinate { x, y }
    }
}

fn main() {
    let mut grid: HashMap<Coordinate, char> = HashMap::new();

    grid.insert(Coordinate::new(0, 0), '.');
    grid.insert(Coordinate::new(1, 0), '#');
    grid.insert(Coordinate::new(0, 1), '.');
    grid.insert(Coordinate::new(1, 1), '#');

    // Display the grid
    for y in 0..2 {
        for x in 0..2 {
            let cell = grid.get(&Coordinate::new(x, y)).unwrap_or(&' ');
            print!("{}", cell);
        }
        println!();
    }

    // Be aware of key ownership
    // For HashMap<String, V>, you can access keys via &str (Borrow trait)
    let mut names: HashMap<String, u32> = HashMap::new();
    names.insert("Tanaka".to_string(), 85);

    // Accessible via &str (String implements Borrow<str>)
    if let Some(score) = names.get("Tanaka") {
        println!("Score: {}", score);
    }
}
```

### 2.3 Customizing the Hash Function

```rust
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

// A simple hasher (for educational purposes)
#[derive(Default)]
struct SimpleHasher {
    hash: u64,
}

impl Hasher for SimpleHasher {
    fn finish(&self) -> u64 {
        self.hash
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.hash = self.hash.wrapping_mul(31).wrapping_add(byte as u64);
        }
    }
}

type SimpleHashMap<K, V> = HashMap<K, V, BuildHasherDefault<SimpleHasher>>;

fn main() {
    // Default SipHash (DoS-resistant, slightly slower)
    let mut default_map: HashMap<String, i32> = HashMap::new();
    default_map.insert("key".to_string(), 42);

    // Custom hasher
    let mut custom_map: SimpleHashMap<String, i32> = SimpleHashMap::default();
    custom_map.insert("key".to_string(), 42);

    // When fast hashing is required, use external crates:
    // - rustc-hash (FxHashMap): used inside the Rust compiler, fast
    // - ahash: fast hashing using AES instructions
    // - fnv: FNV-1a hash, optimal for small keys

    println!("default: {:?}", default_map);
    println!("custom:  {:?}", custom_map);
}
```

---

## 3. Other Collections

```
┌────────────────┬──────────────────────────────────────┐
│ Collection      │ Use case / characteristics           │
├────────────────┼──────────────────────────────────────┤
│ Vec<T>         │ Dynamic array. Append O(1), search O(n)│
│ VecDeque<T>    │ Double-ended queue. Push front/back O(1)│
│ LinkedList<T>  │ Doubly linked list. Rarely needed in practice│
│ HashMap<K,V>   │ Hash map. Average O(1) access        │
│ BTreeMap<K,V>  │ B-tree map. Keeps sort order. O(logn)│
│ HashSet<T>     │ Set without duplicates. Avg O(1) lookup│
│ BTreeSet<T>    │ Sorted set. O(logn)                  │
│ BinaryHeap<T>  │ Max heap. Priority queue             │
└────────────────┴──────────────────────────────────────┘
```

### 3.1 HashSet -- Set Operations

```rust
use std::collections::HashSet;

fn main() {
    let mut set_a: HashSet<i32> = [1, 2, 3, 4, 5].into();
    let set_b: HashSet<i32> = [3, 4, 5, 6, 7].into();

    // Basic operations
    set_a.insert(6);
    set_a.remove(&6);
    println!("Contains 3?: {}", set_a.contains(&3)); // true

    // Set operations
    // Union
    let union: HashSet<_> = set_a.union(&set_b).collect();
    println!("Union: {:?}", union); // {1, 2, 3, 4, 5, 6, 7}

    // Intersection
    let intersection: HashSet<_> = set_a.intersection(&set_b).collect();
    println!("Intersection: {:?}", intersection); // {3, 4, 5}

    // Difference
    let difference: HashSet<_> = set_a.difference(&set_b).collect();
    println!("Difference (A-B): {:?}", difference); // {1, 2}

    // Symmetric difference
    let sym_diff: HashSet<_> = set_a.symmetric_difference(&set_b).collect();
    println!("Symmetric difference: {:?}", sym_diff); // {1, 2, 6, 7}

    // Subset check
    let subset: HashSet<i32> = [3, 4].into();
    println!("subset ⊂ set_a?: {}", subset.is_subset(&set_a)); // true
    println!("set_a ⊃ subset?: {}", set_a.is_superset(&subset)); // true

    // Deduplication pattern
    let with_dups = vec![1, 2, 2, 3, 3, 3, 4, 4, 4, 4];
    let unique: HashSet<_> = with_dups.iter().collect();
    println!("Number of unique elements: {}", unique.len()); // 4

    // Deduplicate while preserving order
    let mut seen = HashSet::new();
    let unique_ordered: Vec<_> = with_dups.iter()
        .filter(|x| seen.insert(*x))
        .collect();
    println!("Deduplicated preserving order: {:?}", unique_ordered); // [1, 2, 3, 4]
}
```

### 3.2 BTreeMap -- Sorted Map

```rust
use std::collections::BTreeMap;

fn main() {
    let mut scores = BTreeMap::new();
    scores.insert("Charlie", 85);
    scores.insert("Alice", 92);
    scores.insert("Bob", 78);
    scores.insert("David", 95);

    // Iteration is always in sorted key order
    for (name, score) in &scores {
        println!("{}: {}", name, score);
    }
    // Alice: 92, Bob: 78, Charlie: 85, David: 95

    // Range query
    for (name, score) in scores.range("Bob"..="David") {
        println!("Range: {} = {}", name, score);
    }
    // Bob: 78, Charlie: 85, David: 95

    // Min/max key
    if let Some((first, _)) = scores.iter().next() {
        println!("Min key: {}", first); // Alice
    }
    if let Some((last, _)) = scores.iter().next_back() {
        println!("Max key: {}", last);  // David
    }

    // Unlike HashMap, BTreeMap requires the Ord trait
    // Keys like f64 cannot be used (only PartialOrd)
    // However, the ordered-float crate can work around this
}
```

### 3.3 VecDeque -- Double-Ended Queue

```rust
use std::collections::VecDeque;

fn main() {
    let mut deque = VecDeque::new();

    // Push to both ends
    deque.push_back(2);
    deque.push_back(3);
    deque.push_front(1);
    deque.push_front(0);
    println!("{:?}", deque); // [0, 1, 2, 3]

    // Pop from both ends
    println!("front: {:?}", deque.pop_front()); // Some(0)
    println!("back:  {:?}", deque.pop_back());  // Some(3)
    println!("{:?}", deque); // [1, 2]

    // Use as a FIFO queue
    let mut queue: VecDeque<&str> = VecDeque::new();
    queue.push_back("Task 1");
    queue.push_back("Task 2");
    queue.push_back("Task 3");

    while let Some(task) = queue.pop_front() {
        println!("Processing: {}", task);
    }

    // Sliding window (fixed size)
    let mut window: VecDeque<i32> = VecDeque::with_capacity(3);
    let data = [1, 2, 3, 4, 5, 6, 7];

    for &value in &data {
        if window.len() == 3 {
            window.pop_front();
        }
        window.push_back(value);
        if window.len() == 3 {
            let sum: i32 = window.iter().sum();
            println!("Window {:?} sum: {}", window, sum);
        }
    }
}
```

### 3.4 BinaryHeap -- Priority Queue

```rust
use std::collections::BinaryHeap;
use std::cmp::Reverse;

fn main() {
    // Max heap (default)
    let mut max_heap = BinaryHeap::new();
    max_heap.push(3);
    max_heap.push(1);
    max_heap.push(4);
    max_heap.push(1);
    max_heap.push(5);

    // pop always returns the maximum value
    while let Some(value) = max_heap.pop() {
        print!("{} ", value); // 5 4 3 1 1
    }
    println!();

    // Min heap (Reverse wrapper)
    let mut min_heap: BinaryHeap<Reverse<i32>> = BinaryHeap::new();
    min_heap.push(Reverse(3));
    min_heap.push(Reverse(1));
    min_heap.push(Reverse(4));

    while let Some(Reverse(value)) = min_heap.pop() {
        print!("{} ", value); // 1 3 4
    }
    println!();

    // Task scheduler example
    #[derive(Debug, Eq, PartialEq)]
    struct Task {
        priority: u32,
        name: String,
    }

    impl Ord for Task {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.priority.cmp(&other.priority)
        }
    }

    impl PartialOrd for Task {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    let mut scheduler = BinaryHeap::new();
    scheduler.push(Task { priority: 1, name: "Low priority task".into() });
    scheduler.push(Task { priority: 10, name: "High priority task".into() });
    scheduler.push(Task { priority: 5, name: "Medium priority task".into() });

    while let Some(task) = scheduler.pop() {
        println!("Run: {} (priority={})", task.name, task.priority);
    }
    // High priority task (10), Medium priority task (5), Low priority task (1)
}
```

---

## 4. The Iterator Trait

### Definition of the Iterator Trait

```rust
// Standard library Iterator trait (simplified)
trait Iterator {
    type Item;  // Associated type: type of elements produced by the iterator

    fn next(&mut self) -> Option<Self::Item>;

    // The following are default implementations based on next() (75+ methods)
    // fn map<B, F>(self, f: F) -> Map<Self, F> { ... }
    // fn filter<P>(self, predicate: P) -> Filter<Self, P> { ... }
    // fn fold<B, F>(self, init: B, f: F) -> B { ... }
    // fn collect<B: FromIterator<Self::Item>>(self) -> B { ... }
    // ...
}
```

### Example 3: Custom Iterator

```rust
struct Counter {
    count: u32,
    max: u32,
}

impl Counter {
    fn new(max: u32) -> Self {
        Counter { count: 0, max }
    }
}

impl Iterator for Counter {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.count < self.max {
            self.count += 1;
            Some(self.count)
        } else {
            None
        }
    }
}

fn main() {
    let counter = Counter::new(5);
    let v: Vec<u32> = counter.collect();
    println!("{:?}", v); // [1, 2, 3, 4, 5]

    // Since Counter implements Iterator, all adapters are available
    let sum: u32 = Counter::new(10).sum();
    println!("Sum of 1..10: {}", sum); // 55

    let doubled: Vec<u32> = Counter::new(5).map(|x| x * 2).collect();
    println!("Doubled: {:?}", doubled); // [2, 4, 6, 8, 10]

    let evens: Vec<u32> = Counter::new(10).filter(|x| x % 2 == 0).collect();
    println!("Evens: {:?}", evens); // [2, 4, 6, 8, 10]
}
```

### Example 4: Fibonacci Iterator

```rust
struct Fibonacci {
    a: u64,
    b: u64,
}

impl Fibonacci {
    fn new() -> Self {
        Fibonacci { a: 0, b: 1 }
    }
}

impl Iterator for Fibonacci {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        let value = self.a;
        let new_b = self.a.checked_add(self.b)?; // None on overflow
        self.a = self.b;
        self.b = new_b;
        Some(value)
    }
}

fn main() {
    // First 10 Fibonacci numbers
    let fibs: Vec<u64> = Fibonacci::new().take(10).collect();
    println!("Fibonacci: {:?}", fibs);
    // [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

    // Fibonacci numbers less than 100
    let small_fibs: Vec<u64> = Fibonacci::new()
        .take_while(|&x| x < 100)
        .collect();
    println!("Less than 100: {:?}", small_fibs);
    // [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]

    // Sum of Fibonacci numbers (first 20)
    let sum: u64 = Fibonacci::new().take(20).sum();
    println!("Sum of the first 20: {}", sum);
}
```

### Lazy Evaluation of Iterators

```
  map / filter / take are "adapters": evaluated lazily
  collect / sum / for_each are "consumers": actually drive iteration

  vec.iter()
    .map(|x| x * 2)      ← adapter (nothing executes)
    .filter(|x| *x > 5)  ← adapter (nothing executes)
    .collect::<Vec<_>>()  ← consumer (this is when all elements are processed)

  ┌──────┐   ┌──────┐   ┌────────┐   ┌─────────┐
  │ iter │──>│ map  │──>│ filter │──>│ collect │
  │      │   │*2    │   │ >5     │   │Vec<_>   │
  └──────┘   └──────┘   └────────┘   └─────────┘
   elem 1 ──── 2 ──────── skip ─────── (skipped)
   elem 2 ──── 4 ──────── skip ─────── (skipped)
   elem 3 ──── 6 ──────── pass ─────── append 6
   elem 4 ──── 8 ──────── pass ─────── append 8
   elem 5 ──── 10 ─────── pass ─────── append 10

  Important: each element passes through the entire pipeline at once
  (it is not batch processing).
  This avoids intermediate collections and is memory efficient.
```

### Demonstrating Lazy Evaluation

```rust
fn main() {
    let v = vec![1, 2, 3, 4, 5];

    // Adapters alone do nothing
    let _lazy = v.iter()
        .map(|x| {
            println!("map: {}", x); // This line is not executed!
            x * 2
        })
        .filter(|x| {
            println!("filter: {}", x); // This line is not executed either!
            *x > 5
        });
    println!("Only the adapter has been built. Nothing has run.");

    // It runs once a consumer is called
    let result: Vec<_> = v.iter()
        .map(|x| {
            println!("map: {}", x);
            x * 2
        })
        .filter(|x| {
            println!("filter: {}", x);
            *x > 5
        })
        .collect();

    println!("Result: {:?}", result);
    // Output:
    // map: 1, filter: 2     ← element 1 passes through map → filter
    // map: 2, filter: 4     ← element 2 passes through map → filter
    // map: 3, filter: 6     ← element 3 passes through map → filter
    // map: 4, filter: 8
    // map: 5, filter: 10
    // Result: [6, 8, 10]
}
```

---

## 5. Iterator Adapters in Detail

### Example 5: Major Adapters and Chaining

```rust
fn main() {
    let numbers = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    // map: transform each element
    let doubled: Vec<i32> = numbers.iter().map(|x| x * 2).collect();
    println!("doubled: {:?}", doubled);

    // filter: keep only matching elements
    let evens: Vec<&i32> = numbers.iter().filter(|x| *x % 2 == 0).collect();
    println!("evens: {:?}", evens);

    // fold: reduce
    let sum = numbers.iter().fold(0, |acc, x| acc + x);
    println!("sum: {}", sum); // 55

    // enumerate: with indices
    for (i, val) in numbers.iter().enumerate() {
        print!("[{}]={} ", i, val);
    }
    println!();

    // zip: combine two iterators
    let names = vec!["Alice", "Bob", "Charlie"];
    let ages = vec![30, 25, 35];
    let people: Vec<_> = names.iter().zip(ages.iter()).collect();
    println!("{:?}", people); // [("Alice", 30), ("Bob", 25), ("Charlie", 35)]

    // chain: concatenate two iterators
    let first = vec![1, 2, 3];
    let second = vec![4, 5, 6];
    let combined: Vec<_> = first.iter().chain(second.iter()).collect();
    println!("{:?}", combined); // [1, 2, 3, 4, 5, 6]

    // take / skip
    let first_three: Vec<_> = numbers.iter().take(3).collect();
    let after_three: Vec<_> = numbers.iter().skip(3).collect();
    println!("take(3): {:?}", first_three);
    println!("skip(3): {:?}", after_three);

    // flat_map: flatten nested iterators
    let words = vec!["hello world", "foo bar"];
    let chars: Vec<&str> = words.iter().flat_map(|s| s.split_whitespace()).collect();
    println!("{:?}", chars); // ["hello", "world", "foo", "bar"]
}
```

### 5.1 Additional Adapters

```rust
fn main() {
    let numbers = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    // take_while / skip_while: condition-based take/skip
    let small: Vec<_> = numbers.iter().take_while(|&&x| x < 5).collect();
    println!("take_while(<5): {:?}", small); // [1, 2, 3, 4]

    let large: Vec<_> = numbers.iter().skip_while(|&&x| x < 5).collect();
    println!("skip_while(<5): {:?}", large); // [5, 6, 7, 8, 9, 10]

    // scan: stateful map
    let running_sum: Vec<i32> = numbers.iter()
        .scan(0, |state, &x| {
            *state += x;
            Some(*state)
        })
        .collect();
    println!("Running sum: {:?}", running_sum);
    // [1, 3, 6, 10, 15, 21, 28, 36, 45, 55]

    // inspect: for debugging (observe values without modifying them)
    let result: Vec<_> = numbers.iter()
        .inspect(|x| print!("before filter: {} ", x))
        .filter(|&&x| x % 2 == 0)
        .inspect(|x| print!("after filter: {} ", x))
        .collect();
    println!("\nResult: {:?}", result);

    // peekable: peek at the next element without consuming it
    let mut iter = numbers.iter().peekable();
    while let Some(&&next) = iter.peek() {
        if next % 3 == 0 {
            println!("Multiple of 3 found: {}", iter.next().unwrap());
        } else {
            iter.next(); // Skip
        }
    }

    // step_by: take every Nth element
    let every_third: Vec<_> = numbers.iter().step_by(3).collect();
    println!("Every third: {:?}", every_third); // [1, 4, 7, 10]

    // unzip: split a pair iterator into two collections
    let pairs = vec![(1, 'a'), (2, 'b'), (3, 'c')];
    let (nums, chars): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
    println!("nums: {:?}, chars: {:?}", nums, chars);
    // nums: [1, 2, 3], chars: ['a', 'b', 'c']
}
```

### 5.2 partition and group_by Patterns

```rust
fn main() {
    let numbers = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    // partition: split into two by a condition
    let (evens, odds): (Vec<_>, Vec<_>) = numbers.iter().partition(|&&x| x % 2 == 0);
    println!("Evens: {:?}", evens); // [2, 4, 6, 8, 10]
    println!("Odds: {:?}", odds);  // [1, 3, 5, 7, 9]

    // group_by pattern (using HashMap)
    use std::collections::HashMap;

    let words = vec!["apple", "banana", "avocado", "blueberry", "cherry", "apricot"];
    let grouped: HashMap<char, Vec<&&str>> = words.iter()
        .fold(HashMap::new(), |mut acc, word| {
            let first_char = word.chars().next().unwrap();
            acc.entry(first_char).or_default().push(word);
            acc
        });

    for (letter, group) in &grouped {
        println!("{}: {:?}", letter, group);
    }
    // a: ["apple", "avocado", "apricot"]
    // b: ["banana", "blueberry"]
    // c: ["cherry"]
}
```

### Example 6: Practical Iterator Pipeline

```rust
#[derive(Debug)]
struct Student {
    name: String,
    score: u32,
}

fn main() {
    let students = vec![
        Student { name: "Tanaka".into(), score: 85 },
        Student { name: "Suzuki".into(), score: 92 },
        Student { name: "Sato".into(), score: 67 },
        Student { name: "Yamada".into(), score: 78 },
        Student { name: "Watanabe".into(), score: 95 },
    ];

    // Get the names of students with 80 or higher and sort by score in descending order
    let mut honor_roll: Vec<_> = students
        .iter()
        .filter(|s| s.score >= 80)
        .collect();
    honor_roll.sort_by(|a, b| b.score.cmp(&a.score));

    println!("Honor roll:");
    for s in &honor_roll {
        println!("  {} ({} points)", s.name, s.score);
    }

    // Average score
    let avg = students.iter().map(|s| s.score).sum::<u32>() as f64
        / students.len() as f64;
    println!("Average: {:.1} points", avg);

    // Group by grade
    use std::collections::HashMap;
    let grouped: HashMap<&str, Vec<&Student>> = students.iter().fold(
        HashMap::new(),
        |mut acc, s| {
            let grade = if s.score >= 90 { "A" }
                       else if s.score >= 80 { "B" }
                       else if s.score >= 70 { "C" }
                       else { "D" };
            acc.entry(grade).or_default().push(s);
            acc
        },
    );
    for (grade, group) in &grouped {
        let names: Vec<_> = group.iter().map(|s| s.name.as_str()).collect();
        println!("Grade {}: {:?}", grade, names);
    }
}
```

### Example 7: Complex Data Pipeline

```rust
use std::collections::HashMap;

#[derive(Debug, Clone)]
struct Sale {
    product: String,
    category: String,
    amount: f64,
    quantity: u32,
}

fn main() {
    let sales = vec![
        Sale { product: "Apple".into(), category: "Fruit".into(), amount: 150.0, quantity: 3 },
        Sale { product: "Banana".into(), category: "Fruit".into(), amount: 100.0, quantity: 5 },
        Sale { product: "Carrot".into(), category: "Vegetable".into(), amount: 80.0, quantity: 2 },
        Sale { product: "Tomato".into(), category: "Vegetable".into(), amount: 200.0, quantity: 4 },
        Sale { product: "Apple".into(), category: "Fruit".into(), amount: 150.0, quantity: 2 },
        Sale { product: "Broccoli".into(), category: "Vegetable".into(), amount: 180.0, quantity: 1 },
    ];

    // Aggregate sales by category
    let category_totals: HashMap<&str, f64> = sales.iter()
        .fold(HashMap::new(), |mut acc, sale| {
            *acc.entry(sale.category.as_str()).or_insert(0.0) += sale.amount * sale.quantity as f64;
            acc
        });

    println!("Sales by category:");
    for (category, total) in &category_totals {
        println!("  {}: {:.0} yen", category, total);
    }

    // Top 3 products by quantity sold
    let mut product_quantities: HashMap<&str, u32> = HashMap::new();
    for sale in &sales {
        *product_quantities.entry(&sale.product).or_insert(0) += sale.quantity;
    }

    let mut top_products: Vec<_> = product_quantities.iter().collect();
    top_products.sort_by(|a, b| b.1.cmp(a.1));

    println!("\nTop 3 by sales quantity:");
    for (product, qty) in top_products.iter().take(3) {
        println!("  {}: {} units", product, qty);
    }

    // Average unit price of fruits with unit price >= 100 yen
    let expensive_fruits: Vec<f64> = sales.iter()
        .filter(|s| s.category == "Fruit" && s.amount >= 100.0)
        .map(|s| s.amount)
        .collect();

    if !expensive_fruits.is_empty() {
        let avg = expensive_fruits.iter().sum::<f64>() / expensive_fruits.len() as f64;
        println!("\nAverage unit price of fruits priced 100 yen or more: {:.0} yen", avg);
    }
}
```

---

## 6. Differences Between into_iter / iter / iter_mut

```
┌───────────────┬───────────────────┬────────────┬───────────────┐
│ Method         │ Element type       │ Ownership  │ Collection     │
├───────────────┼───────────────────┼────────────┼───────────────┤
│ .iter()       │ &T                │ Borrow     │ Remains as is  │
│ .iter_mut()   │ &mut T            │ Mut borrow │ Remains as is  │
│ .into_iter()  │ T                 │ Move       │ Consumed       │
├───────────────┼───────────────────┼────────────┼───────────────┤
│ for x in &v   │ &T  (iter)        │ Borrow     │ Remains as is  │
│ for x in &mut v│ &mut T (iter_mut)│ Mut borrow │ Remains as is  │
│ for x in v    │ T    (into_iter)  │ Move       │ Consumed       │
└───────────────┴───────────────────┴────────────┴───────────────┘
```

### 6.1 Choosing the Right Iterator

```rust
fn main() {
    let v = vec![String::from("hello"), String::from("world")];

    // iter(): borrow -- collection remains usable
    for s in v.iter() {
        println!("Borrow: {}", s); // s is &String
    }
    println!("v is still usable: {:?}", v);

    // iter_mut(): mutable borrow -- elements can be modified
    let mut v2 = vec![1, 2, 3, 4, 5];
    for n in v2.iter_mut() {
        *n *= 2; // Double each element
    }
    println!("After modification: {:?}", v2); // [2, 4, 6, 8, 10]

    // into_iter(): move ownership -- collection is consumed
    let v3 = vec![String::from("a"), String::from("b")];
    let uppercased: Vec<String> = v3.into_iter()
        .map(|s| s.to_uppercase()) // s is String (owned)
        .collect();
    println!("Uppercased: {:?}", uppercased);
    // println!("{:?}", v3); // Compile error! v3 has been moved

    // How to obtain ownership from a reference iterator
    let v4 = vec!["hello", "world"];
    let owned: Vec<String> = v4.iter()
        .map(|s| s.to_string()) // &str → clone to String
        .collect();
    println!("Owned: {:?}", owned);
    println!("Original v4: {:?}", v4); // Still usable

    // Use cloned() / copied() to copy/clone
    let v5 = vec![1, 2, 3, 4, 5];
    let copied: Vec<i32> = v5.iter().copied().collect(); // &i32 → i32
    println!("copied: {:?}", copied);

    let v6 = vec!["a".to_string(), "b".to_string()];
    let cloned: Vec<String> = v6.iter().cloned().collect(); // &String → String
    println!("cloned: {:?}", cloned);
}
```

### 6.2 The IntoIterator Trait

```rust
// The IntoIterator trait makes for-loop iteration possible
// trait IntoIterator {
//     type Item;
//     type IntoIter: Iterator<Item = Self::Item>;
//     fn into_iter(self) -> Self::IntoIter;
// }

// Implement IntoIterator for a custom type
struct Matrix {
    data: Vec<Vec<f64>>,
}

impl Matrix {
    fn new(data: Vec<Vec<f64>>) -> Self {
        Matrix { data }
    }
}

// Owning version
impl IntoIterator for Matrix {
    type Item = Vec<f64>;
    type IntoIter = std::vec::IntoIter<Vec<f64>>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

// Borrowing version
impl<'a> IntoIterator for &'a Matrix {
    type Item = &'a Vec<f64>;
    type IntoIter = std::slice::Iter<'a, Vec<f64>>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

fn main() {
    let matrix = Matrix::new(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ]);

    // Usable in a for loop
    for row in &matrix {
        println!("Row: {:?}", row);
    }

    // Consume rows via into_iter
    for row in matrix {
        let sum: f64 = row.iter().sum();
        println!("Row sum: {}", sum);
    }
}
```

---

## 7. How FromIterator and collect Work

### 7.1 The Many Conversions of collect

```rust
use std::collections::{HashMap, HashSet, BTreeSet, VecDeque};

fn main() {
    let numbers = vec![1, 2, 3, 4, 5, 3, 2, 1];

    // collect into Vec
    let v: Vec<i32> = numbers.iter().copied().collect();
    println!("Vec: {:?}", v);

    // collect into HashSet (deduplication)
    let set: HashSet<i32> = numbers.iter().copied().collect();
    println!("HashSet: {:?}", set);

    // collect into BTreeSet (sorted set)
    let bset: BTreeSet<i32> = numbers.iter().copied().collect();
    println!("BTreeSet: {:?}", bset);

    // collect into VecDeque
    let deque: VecDeque<i32> = numbers.iter().copied().collect();
    println!("VecDeque: {:?}", deque);

    // collect into HashMap (from a pair iterator)
    let map: HashMap<&str, i32> = vec![("a", 1), ("b", 2), ("c", 3)]
        .into_iter()
        .collect();
    println!("HashMap: {:?}", map);

    // collect into String
    let chars = vec!['H', 'e', 'l', 'l', 'o'];
    let s: String = chars.into_iter().collect();
    println!("String: {}", s);

    // collect into Result<Vec<T>, E> (returns Err immediately on any error)
    let strings = vec!["1", "2", "abc", "4"];
    let result: Result<Vec<i32>, _> = strings.iter()
        .map(|s| s.parse::<i32>())
        .collect();
    println!("Result: {:?}", result); // Err(ParseIntError)

    // Successful case
    let valid = vec!["1", "2", "3", "4"];
    let result: Result<Vec<i32>, _> = valid.iter()
        .map(|s| s.parse::<i32>())
        .collect();
    println!("Result: {:?}", result); // Ok([1, 2, 3, 4])
}
```

### 7.2 Custom FromIterator Implementation

```rust
use std::iter::FromIterator;

#[derive(Debug)]
struct Histogram {
    bins: std::collections::HashMap<i32, usize>,
}

impl FromIterator<i32> for Histogram {
    fn from_iter<I: IntoIterator<Item = i32>>(iter: I) -> Self {
        let mut bins = std::collections::HashMap::new();
        for value in iter {
            *bins.entry(value).or_insert(0) += 1;
        }
        Histogram { bins }
    }
}

impl Histogram {
    fn display(&self) {
        let mut entries: Vec<_> = self.bins.iter().collect();
        entries.sort_by_key(|&(k, _)| k);
        for (value, count) in entries {
            println!("{:>3}: {}", value, "#".repeat(*count));
        }
    }
}

fn main() {
    let data = vec![1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5];

    // Convert into a histogram via collect
    let hist: Histogram = data.into_iter().collect();
    hist.display();
    //   1: #
    //   2: ##
    //   3: ###
    //   4: ####
    //   5: #####
}
```

---

## 8. Advanced Iterator Patterns

### 8.1 Recursive Flattening

```rust
fn flatten_nested(nested: &[Vec<Vec<i32>>]) -> Vec<i32> {
    nested.iter()
        .flat_map(|inner| inner.iter())
        .flat_map(|v| v.iter())
        .copied()
        .collect()
}

fn main() {
    let nested = vec![
        vec![vec![1, 2], vec![3, 4]],
        vec![vec![5, 6], vec![7, 8]],
    ];

    let flat = flatten_nested(&nested);
    println!("Flat: {:?}", flat); // [1, 2, 3, 4, 5, 6, 7, 8]

    // Using Iterator::flatten()
    let nested2 = vec![vec![1, 2, 3], vec![4, 5], vec![6]];
    let flat2: Vec<i32> = nested2.into_iter().flatten().collect();
    println!("flatten: {:?}", flat2); // [1, 2, 3, 4, 5, 6]

    // flatten an iterator of Options
    let options = vec![Some(1), None, Some(3), None, Some(5)];
    let values: Vec<i32> = options.into_iter().flatten().collect();
    println!("flatten Option: {:?}", values); // [1, 3, 5]
}
```

### 8.2 Window Operations and Adjacent Comparisons

```rust
fn main() {
    let data = vec![1, 3, 2, 5, 4, 7, 6, 8];

    // Compare adjacent pairs (using windows)
    let increasing_pairs: Vec<_> = data.windows(2)
        .filter(|w| w[1] > w[0])
        .map(|w| (w[0], w[1]))
        .collect();
    println!("Increasing pairs: {:?}", increasing_pairs);
    // [(1, 3), (2, 5), (4, 7), (6, 8)]

    // Moving average
    let values = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0];
    let window_size = 3;
    let moving_avg: Vec<f64> = values.windows(window_size)
        .map(|w| w.iter().sum::<f64>() / window_size as f64)
        .collect();
    println!("Moving average (size=3): {:?}", moving_avg);
    // [20.0, 30.0, 40.0, 50.0, 60.0]

    // Difference series
    let diffs: Vec<f64> = values.windows(2)
        .map(|w| w[1] - w[0])
        .collect();
    println!("Differences: {:?}", diffs); // [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]

    // Group consecutive identical values (run-length encoding)
    let seq = vec![1, 1, 2, 2, 2, 3, 1, 1];
    let mut rle: Vec<(i32, usize)> = Vec::new();
    for &value in &seq {
        if let Some(last) = rle.last_mut() {
            if last.0 == value {
                last.1 += 1;
                continue;
            }
        }
        rle.push((value, 1));
    }
    println!("RLE: {:?}", rle); // [(1, 2), (2, 3), (3, 1), (1, 2)]
}
```

### 8.3 Building a Custom Iterator Adapter

```rust
// Custom adapter: returns each element with its index (a simplified enumerate)
struct Indexed<I> {
    iter: I,
    index: usize,
}

impl<I: Iterator> Iterator for Indexed<I> {
    type Item = (usize, I::Item);

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.iter.next()?;
        let index = self.index;
        self.index += 1;
        Some((index, item))
    }
}

// Integrate into method chains via an extension trait
trait IteratorExt: Iterator + Sized {
    fn indexed(self) -> Indexed<Self> {
        Indexed { iter: self, index: 0 }
    }
}

impl<I: Iterator> IteratorExt for I {}

fn main() {
    let words = vec!["hello", "world", "rust"];
    for (i, word) in words.iter().indexed() {
        println!("[{}] = {}", i, word);
    }

    // Usable inside a chain
    let result: Vec<_> = (0..5)
        .map(|x| x * x)
        .indexed()
        .filter(|(_, v)| *v > 4)
        .collect();
    println!("{:?}", result); // [(3, 9), (4, 16)]
}
```

---

## 9. Performance and Zero-Cost Abstractions

### 9.1 Iterator vs Loop Benchmark

```rust
fn main() {
    let n = 10_000_000;
    let data: Vec<i32> = (0..n).collect();

    use std::time::Instant;

    // for-loop version
    let start = Instant::now();
    let mut sum1: i64 = 0;
    for &x in &data {
        if x % 2 == 0 {
            sum1 += (x as i64) * (x as i64);
        }
    }
    let t1 = start.elapsed();

    // Iterator version
    let start = Instant::now();
    let sum2: i64 = data.iter()
        .filter(|&&x| x % 2 == 0)
        .map(|&x| (x as i64) * (x as i64))
        .sum();
    let t2 = start.elapsed();

    assert_eq!(sum1, sum2);
    println!("for loop:   {:?} (sum={})", t1, sum1);
    println!("iterator:   {:?} (sum={})", t2, sum2);
    // In an optimized build (-O / --release), performance is equivalent
    // LLVM optimizes both to the same machine code
}
```

### 9.2 How Zero-Cost Abstraction Works

```
  Iterator chain:
    data.iter().map(|x| x * 2).filter(|x| *x > 10).sum()

  After compiler optimization (conceptual equivalent):
    let mut sum = 0;
    for &x in data {
        let doubled = x * 2;
        if doubled > 10 {
            sum += doubled;
        }
    }

  No intermediate collections are produced
  Closures in map/filter are inlined
  The loop runs only once

  Furthermore, with automatic SIMD vectorization:
  ┌──────────────────────────────────────┐
  │  Hand-written loop  → SIMD optimized │
  │  Iterator chain     → SIMD optimized │
  │  Performance gap = zero (same code)  │
  └──────────────────────────────────────┘
```

### 9.3 Cases Where Iterators Are Faster

```rust
fn main() {
    let data: Vec<i32> = (0..1_000_000).collect();

    // Cases where iterators end up faster:
    // 1. Eliminating bounds checks
    //    Iterators internally track length, so no
    //    bounds check is needed for index access

    // BAD: bounds check every iteration
    let mut sum1 = 0i64;
    for i in 0..data.len() {
        sum1 += data[i] as i64; // data[i] triggers a bounds check
    }

    // GOOD: no bounds check needed
    let sum2: i64 = data.iter().map(|&x| x as i64).sum();

    assert_eq!(sum1, sum2);

    // 2. Encouraging automatic vectorization
    //    Iterators are a pattern that the compiler finds easy to optimize

    // 3. Optimization of branch prediction
    //    The compiler can place filter conditions optimally
}
```

---

## 10. Comparison Tables

### 10.1 Collection Selection Guide

| Operation | Vec | VecDeque | HashMap | BTreeMap | HashSet |
|------|-----|----------|---------|----------|---------|
| Append to back | O(1)* | O(1)* | - | - | - |
| Push to front | O(n) | O(1)* | - | - | - |
| Key lookup | O(n) | O(n) | O(1)* | O(log n) | O(1)* |
| Order preserved | Insertion | Insertion | None | Sorted | None |
| Memory | Contiguous | Contiguous | Hash | Tree | Hash |

*= amortized complexity

### 10.2 Detailed Collection Comparison

| Collection | push/insert | remove | search | Memory efficiency | Cache efficiency |
|-------------|-------------|--------|--------|-----------|--------------|
| Vec | O(1)* back | O(n) | O(n) | Highest | Highest |
| VecDeque | O(1)* both ends | O(n) | O(n) | High | High |
| LinkedList | O(1) | O(1) | O(n) | Low | Low |
| HashMap | O(1)* | O(1)* | O(1)* | Medium | Medium |
| BTreeMap | O(log n) | O(log n) | O(log n) | Medium | Medium |
| HashSet | O(1)* | O(1)* | O(1)* | Medium | Medium |
| BTreeSet | O(log n) | O(log n) | O(log n) | Medium | Medium |
| BinaryHeap | O(log n) | O(log n) | O(n) | High | High |

### 10.3 Iterator Consumer Methods

| Method | Returns | Description |
|----------|--------|------|
| `collect()` | Collection | Convert iterator into a collection |
| `sum()` | Numeric | Compute the sum |
| `product()` | Numeric | Compute the product |
| `count()` | usize | Count the elements |
| `any(f)` | bool | Whether any element matches |
| `all(f)` | bool | Whether all elements match |
| `find(f)` | Option | First element matching the condition |
| `position(f)` | Option | Position of first matching element |
| `min()` / `max()` | Option | Minimum/maximum value |
| `min_by_key(f)` / `max_by_key(f)` | Option | Min/max by a key function |
| `for_each(f)` | () | Apply a side effect to each element |
| `reduce(f)` | Option | Reduce without an initial value |
| `fold(init, f)` | B | Reduce with an initial value |
| `last()` | Option | The last element |
| `nth(n)` | Option | The nth element |
| `unzip()` | (B, C) | Split a pair iterator into two collections |

### 10.4 List of Iterator Adapters

| Adapter | Description | Example |
|---------|------|-----|
| `map(f)` | Transform each element | `.map(\|x\| x * 2)` |
| `filter(f)` | Select matching elements | `.filter(\|x\| *x > 0)` |
| `filter_map(f)` | Transform + filter | `.filter_map(\|x\| x.parse().ok())` |
| `flat_map(f)` | Transform + flatten | `.flat_map(\|s\| s.chars())` |
| `flatten()` | Flatten nesting | `.flatten()` |
| `enumerate()` | Add indices | `.enumerate()` |
| `zip(iter)` | Combine two | `.zip(other.iter())` |
| `chain(iter)` | Concatenate two | `.chain(other.iter())` |
| `take(n)` | First n elements | `.take(5)` |
| `skip(n)` | Skip first n elements | `.skip(3)` |
| `take_while(f)` | While condition holds | `.take_while(\|x\| *x < 10)` |
| `skip_while(f)` | Skip while condition holds | `.skip_while(\|x\| *x < 5)` |
| `peekable()` | Allows peeking | `.peekable()` |
| `scan(state, f)` | Stateful map | `.scan(0, \|s, x\| { ... })` |
| `inspect(f)` | Debug observation | `.inspect(\|x\| println!("{}", x))` |
| `step_by(n)` | Every n elements | `.step_by(2)` |
| `rev()` | Reverse | `.rev()` |
| `cloned()` | &T → T (Clone) | `.cloned()` |
| `copied()` | &T → T (Copy) | `.copied()` |

---

## 11. Anti-patterns

### Anti-pattern 1: Unnecessary collect

```rust
// BAD: no need to create an intermediate collection
fn sum_of_squares(v: &[i32]) -> i32 {
    let squared: Vec<i32> = v.iter().map(|x| x * x).collect(); // wasted Vec
    squared.iter().sum()
}

// GOOD: consume the iterator chain directly
fn sum_of_squares_good(v: &[i32]) -> i32 {
    v.iter().map(|x| x * x).sum()
}
```

### Anti-pattern 2: Index-based Loops

```rust
// BAD: C-style index loop
fn print_all(v: &[String]) {
    for i in 0..v.len() {
        println!("{}: {}", i, v[i]); // bounds check runs every iteration
    }
}

// GOOD: use an iterator
fn print_all_good(v: &[String]) {
    for (i, item) in v.iter().enumerate() {
        println!("{}: {}", i, item); // safe and fast
    }
}
```

### Anti-pattern 3: Unnecessary Clones

```rust
// BAD: unnecessarily cloning
fn find_longest(strings: &[String]) -> Option<String> {
    strings.iter()
        .cloned()  // clones every element (expensive)
        .max_by_key(|s| s.len())
}

// GOOD: process via references
fn find_longest_good(strings: &[String]) -> Option<&String> {
    strings.iter()
        .max_by_key(|s| s.len())
}
```

### Anti-pattern 4: Splitting filter and map

```rust
// BAD: filter and map separately
fn parse_valid_numbers(items: &[&str]) -> Vec<i32> {
    items.iter()
        .filter(|s| s.parse::<i32>().is_ok())
        .map(|s| s.parse::<i32>().unwrap()) // parsing twice!
        .collect()
}

// GOOD: use filter_map
fn parse_valid_numbers_good(items: &[&str]) -> Vec<i32> {
    items.iter()
        .filter_map(|s| s.parse::<i32>().ok())
        .collect()
}
```

### Anti-pattern 5: collect Followed by len/is_empty

```rust
// BAD: gather all elements before counting
fn count_evens(v: &[i32]) -> usize {
    let evens: Vec<_> = v.iter().filter(|&&x| x % 2 == 0).collect();
    evens.len()
}

// GOOD: use count()
fn count_evens_good(v: &[i32]) -> usize {
    v.iter().filter(|&&x| x % 2 == 0).count()
}

// BAD: gather all elements before checking emptiness
fn has_evens(v: &[i32]) -> bool {
    let evens: Vec<_> = v.iter().filter(|&&x| x % 2 == 0).collect();
    !evens.is_empty()
}

// GOOD: use any() (returns immediately on the first match)
fn has_evens_good(v: &[i32]) -> bool {
    v.iter().any(|&x| x % 2 == 0)
}
```

---

## 12. FAQ

### Q1: How do I specify the type for `collect()`?

**A:** There are three ways:
```rust
// Method 1: type annotation on the variable
let v: Vec<i32> = (0..10).collect();

// Method 2: turbofish syntax
let v = (0..10).collect::<Vec<i32>>();

// Method 3: partial type annotation
let v = (0..10).collect::<Vec<_>>(); // element type is inferred
```

### Q2: Which is faster, iterators or for loops?

**A:** They are equivalent. Rust iterators are zero-cost abstractions, and the compiler optimizes them into the same machine code. Iterators may even be faster in some cases thanks to elimination of bounds checks and automatic vectorization. Comparing in `--release` builds (with optimizations) is what matters; iterators may be slower in `debug` builds because inlining is disabled.

### Q3: How can I use my own type as a HashMap key?

**A:** You need to implement the `Eq + Hash` traits. You can derive them automatically with `#[derive(PartialEq, Eq, Hash)]`:
```rust
#[derive(PartialEq, Eq, Hash)]
struct Point {
    x: i32,
    y: i32,
}
```
Note: `f64` does not implement `Eq`, so types containing floating-point numbers cannot be used as HashMap keys. You can work around this with `ordered_float::OrderedFloat<f64>`.

### Q4: When should I use `iter()` versus `into_iter()`?

**A:** Use `iter()` (borrow) when you need the collection later, and `into_iter()` (move ownership) when you no longer need it. `into_iter()` can reuse heap allocations directly, which is efficient when transforming owned types like strings:
```rust
// iter() → cloning is required
let v = vec!["hello".to_string(), "world".to_string()];
let upper: Vec<String> = v.iter().map(|s| s.to_uppercase()).collect();
println!("{:?}", v); // still usable

// into_iter() → no clone, reuses original memory
let v = vec!["hello".to_string(), "world".to_string()];
let upper: Vec<String> = v.into_iter().map(|s| s.to_uppercase()).collect();
// println!("{:?}", v); // compile error
```

### Q5: When processing large amounts of data, are iterators sufficient?

**A:** They are sufficient on a single thread. If you need parallel processing, use `par_iter()` from the `rayon` crate:
```rust
use rayon::prelude::*;

let data: Vec<i32> = (0..10_000_000).collect();

// Automatically parallelized over a thread pool
let sum: i64 = data.par_iter()
    .map(|&x| (x as i64) * (x as i64))
    .sum();
```
With `rayon`, you can parallelize an existing iterator chain simply by changing `iter()` to `par_iter()`.

### Q6: How do I choose a collection?

**A:** Use the following criteria:

| Requirement | Recommended collection |
|------|----------------|
| Ordered list | Vec |
| FIFO queue | VecDeque |
| Fast lookup by key | HashMap |
| Sorted lookup by key | BTreeMap |
| Deduplication | HashSet |
| Sorted set | BTreeSet |
| Priority queue | BinaryHeap |
| 99% of cases | Vec |

When in doubt, use `Vec` and switch to a more appropriate collection only after a bottleneck has been identified -- this is the typical Rust approach.

---


## FAQ

### Q1: What is the most important point when learning this topic?

Gaining practical experience is most important. Beyond theory, your understanding deepens when you actually write code and verify how it behaves.

### Q2: What mistakes do beginners commonly make?

Skipping the basics and jumping into advanced topics. We recommend solidly understanding the fundamental concepts described in this guide before moving on to the next step.

### Q3: How is this used in practice?

Knowledge of this topic is used frequently in day-to-day development work. It is especially important during code reviews and in architectural design.

---

## 13. Summary

| Concept | Key point |
|------|------|
| Vec<T> | The most common collection. Contiguous memory, O(1) append |
| HashMap<K,V> | Key-value pairs. O(1) lookup. Insert/update via the entry API |
| HashSet<T> | Set without duplicates. Supports set operations |
| BTreeMap<K,V> | Preserves sort order. Supports range queries |
| VecDeque<T> | Double-ended queue. O(1) at both ends |
| BinaryHeap<T> | Priority queue. Efficient access to max/min |
| Iterator | Trait that implements next(). Lazy evaluation |
| Adapters | map/filter/take and others. Lazy and chainable |
| Consumers | collect/sum/for_each and others. Drive iteration |
| Zero-cost | Iterators perform on par with hand-written loops |
| into_iter vs iter | The difference between consuming ownership and borrowing |
| FromIterator | Trait that determines the target type of collect |
| Extension traits | Integrate custom adapters into method chains |

---

## Recommended Next Reading

- [../01-advanced/02-closures-fn-traits.md](../01-advanced/02-closures-fn-traits.md) -- Closures and Fn traits
- [../01-advanced/00-lifetimes.md](../01-advanced/00-lifetimes.md) -- Lifetimes in detail
- [../02-async/02-async-patterns.md](../02-async/02-async-patterns.md) -- Stream (asynchronous iterators)

---

## References

1. **The Rust Programming Language - Ch.8 Common Collections, Ch.13 Iterators** -- https://doc.rust-lang.org/book/
2. **std::iter Module Documentation** -- https://doc.rust-lang.org/std/iter/
3. **std::collections Module Documentation** -- https://doc.rust-lang.org/std/collections/
4. **Rust by Example - Iterator** -- https://doc.rust-lang.org/rust-by-example/trait/iter.html
5. **Iterator Performance (Rust Blog)** -- https://blog.rust-lang.org/2017/02/02/Rust-1.15.html
6. **rayon: Data-parallelism library** -- https://docs.rs/rayon/
7. **Rust Performance Book** -- https://nnethercote.github.io/perf-book/
