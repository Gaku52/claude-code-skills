# Iterators and Generators

> An iterator is an abstraction for "retrieving elements from a collection one at a time," enabling infinite sequences and memory-efficient processing through lazy evaluation. A generator is "a function whose execution can be suspended and resumed," serving as syntactic sugar for conveniently creating iterators. This chapter provides a systematic explanation of these concepts from fundamentals to advanced applications.

---

## What You Will Learn in This Chapter

- [ ] Understand the design intent and mechanism of the iterator pattern
- [ ] Grasp the differences in iterator protocols across languages
- [ ] Understand how generators work (relationship with coroutines)
- [ ] Make informed tradeoff decisions between lazy and eager evaluation
- [ ] Build declarative data processing pipelines by combining iterator adapters
- [ ] Understand the use cases for async iterators
- [ ] Master techniques for safely working with infinite sequences


## Prerequisites

Before reading this guide, the following knowledge will help deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Familiarity with the content of [Error Handling](./02-error-handling.md)

---

## Table of Contents

1. [The Essence of the Iterator Pattern](#1-the-essence-of-the-iterator-pattern)
2. [Iterator Protocols Across Languages](#2-iterator-protocols-across-languages)
3. [Iterator Adapters and Lazy Transformation Chains](#3-iterator-adapters-and-lazy-transformation-chains)
4. [How Generators Work](#4-how-generators-work)
5. [Lazy Evaluation vs Eager Evaluation](#5-lazy-evaluation-vs-eager-evaluation)
6. [Async Iterators and Streams](#6-async-iterators-and-streams)
7. [Practical Patterns](#7-practical-patterns)
8. [Anti-Patterns and Pitfalls](#8-anti-patterns-and-pitfalls)
9. [Performance Characteristics and Optimization](#9-performance-characteristics-and-optimization)
10. [Exercises](#10-exercises)
11. [FAQ (Frequently Asked Questions)](#11-faqfrequently-asked-questions)
12. [Summary](#12-summary)
13. [References](#13-references)

---

## 1. The Essence of the Iterator Pattern

### 1.1 The Iterator as a Design Pattern

In the GoF (Gang of Four) design patterns, the iterator pattern is classified as a **Behavioral Pattern**. Its purpose is to "provide a means to sequentially access elements of a collection without exposing its internal representation."

```
+-------------------------------------------------------------+
|  Position of the Iterator Pattern                            |
+-------------------------------------------------------------+
|                                                             |
|  Client Code                                                |
|       |                                                     |
|       | next() / hasNext()                                  |
|       v                                                     |
|  +------------------+                                       |
|  | Iterator         |  <--- Unified Interface               |
|  | - next()         |                                       |
|  | - hasNext()      |                                       |
|  +------------------+                                       |
|       ^        ^        ^                                   |
|       |        |        |                                   |
|  +--------+ +--------+ +--------+                           |
|  | Array  | | Tree   | | Graph  |   <--- Different internal |
|  |Iterator| |Iterator| |Iterator|        structures         |
|  +--------+ +--------+ +--------+                           |
|                                                             |
|  Key point: Clients can traverse without knowing internals  |
+-------------------------------------------------------------+
```

This abstraction provides four benefits.

| Benefit | Description | Example |
|---------|-------------|---------|
| **Encapsulation** | Hides the internal structure of a collection | Traverse arrays, linked lists, and trees through the same interface |
| **Uniform Access** | Use the same loop syntax for different data structures | `for x in collection` works for any collection |
| **Lazy Evaluation** | Generate elements one at a time as needed | Process a million-line file one line at a time |
| **Composability** | Combine iterators to build pipelines | `filter -> map -> take -> collect` |

### 1.2 Internal Iterators and External Iterators

There are two types of iterator designs, depending on which side holds control.

```
+-----------------------------------------------+
|  External Iterator (Pull-based)                |
+-----------------------------------------------+
|                                               |
|  Client:   "Give me the next" --> Iterator    |
|  Iterator: <-- Returns an element             |
|                                               |
|  Control: Client side                         |
|  Examples: Python's next(), Rust's .next()    |
|  Advantage: Client controls traversal timing  |
|             Can alternate between two iterators|
+-----------------------------------------------+

+-----------------------------------------------+
|  Internal Iterator (Push-based)                |
+-----------------------------------------------+
|                                               |
|  Client:     Passes a closure --> Collection  |
|  Collection: Applies closure to each element  |
|                                               |
|  Control: Collection side                     |
|  Examples: Ruby's each, JS's forEach          |
|  Advantage: Simpler implementation,           |
|             easier to parallelize             |
+-----------------------------------------------+
```

**Comparison: Internal vs External Iterators**

| Property | External Iterator (Pull) | Internal Iterator (Push) |
|----------|-------------------------|--------------------------|
| Control | Client | Collection |
| Interrupting traversal | Freely possible | Interrupt via break/exception |
| Alternating multiple iterators | Easy | Difficult |
| Implementation complexity | Requires state management | Relatively simple |
| Parallelization | Manual implementation | Collection can optimize |
| Representative examples | Python `__next__`, Rust `Iterator` | Ruby `each`, JS `forEach` |
| Affinity with lazy evaluation | Naturally lazy | Basically eager |

```python
# External iterator example (Pull-based)
it = iter([1, 2, 3, 4, 5])
a = next(it)  # 1 --- Client drives
b = next(it)  # 2 --- Retrieves at desired timing
# Remaining 3, 4, 5 have not been "pulled out" yet

# Internal iterator example (Push-based)
[1, 2, 3, 4, 5].forEach(x => console.log(x))
# The collection traverses all elements
# The client only specifies "what to do with each element"
```

### 1.3 Common Structure of Iterator Protocols

All iterator protocols across languages share the same fundamental structure.

```
+-----------------------------------------------------------+
|  Common Structure of Iterator Protocols                    |
+-----------------------------------------------------------+
|                                                           |
|  State: current_position, underlying_collection           |
|                                                           |
|  Operation:                                               |
|  +-----------------------+                                |
|  | next()                |                                |
|  | +------------------+  |                                |
|  | | Has elements?    |  |                                |
|  | |   Yes -> (value, |  |  <- Indicates "more to come"  |
|  | |          continue)|  |                               |
|  | |   No  -> end     |  |  <- Indicates "no more"       |
|  | |          signal   |  |                               |
|  | +------------------+  |                                |
|  +-----------------------+                                |
|                                                           |
|  How end is signaled:                                     |
|    Python:  StopIteration exception                       |
|    Rust:    Option<T>'s None                              |
|    Java:    Two-method approach: hasNext() + next()       |
|    JS:      { value, done: true/false } object            |
|    C++:     Comparison with end() iterator                |
+-----------------------------------------------------------+
```

---

## 2. Iterator Protocols Across Languages

### 2.1 Python: `__iter__` / `__next__` Protocol

Python's iterator protocol consists of two dunder methods:

- `__iter__()`: Returns the iterator object itself
- `__next__()`: Returns the next element, or raises `StopIteration` if there are no more elements

```python
class Countdown:
    """Implementation example of a countdown iterator"""

    def __init__(self, start):
        self.current = start

    def __iter__(self):
        return self

    def __next__(self):
        if self.current <= 0:
            raise StopIteration
        self.current -= 1
        return self.current + 1

# Usage
for n in Countdown(5):
    print(n)  # 5, 4, 3, 2, 1

# Expanding the internal behavior of the for loop:
it = iter(Countdown(5))   # Calls __iter__()
while True:
    try:
        n = next(it)      # Calls __next__()
        print(n)
    except StopIteration:
        break             # End
```

Python distinguishes between iterables (which have `__iter__`) and iterators (which have both `__iter__` and `__next__`). A list is iterable but not an iterator. You obtain an iterator by calling `iter()`.

```python
# Difference between iterable and iterator
lst = [1, 2, 3]
print(type(lst))         # <class 'list'> --- Iterable
it = iter(lst)
print(type(it))          # <class 'list_iterator'> --- Iterator

# An iterable can generate iterators any number of times
for x in lst: pass  # 1st time
for x in lst: pass  # 2nd time (works without issues)

# An iterator becomes empty once exhausted
it = iter(lst)
list(it)  # [1, 2, 3]
list(it)  # [] --- Empty after exhaustion
```

### 2.2 Rust: The `Iterator` Trait

Rust iterators are defined by implementing the `Iterator` trait. Termination is represented by `None` of `Option<Self::Item>`.

```rust
struct Countdown {
    current: u32,
}

impl Countdown {
    fn new(start: u32) -> Self {
        Countdown { current: start }
    }
}

impl Iterator for Countdown {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        if self.current == 0 {
            None
        } else {
            self.current -= 1;
            Some(self.current + 1)
        }
    }
}

fn main() {
    // Using with a for loop
    for n in Countdown::new(5) {
        println!("{}", n);  // 5, 4, 3, 2, 1
    }

    // IntoIterator trait: syntactic sugar for the for loop
    // for item in collection { ... }
    // is equivalent to:
    // let mut iter = collection.into_iter();
    // while let Some(item) = iter.next() { ... }
}
```

Rust's iterator system has three ownership models.

```rust
let v = vec![1, 2, 3];

// 1. iter(): Iterator of immutable references (&T)
for val in v.iter() {
    // val is of type &i32
    println!("{}", val);
}
// v is still usable

// 2. iter_mut(): Iterator of mutable references (&mut T)
let mut v2 = vec![1, 2, 3];
for val in v2.iter_mut() {
    // val is of type &mut i32
    *val *= 2;
}
// v2 has been modified to [2, 4, 6]

// 3. into_iter(): Iterator that moves ownership (T)
for val in v.into_iter() {
    // val is of type i32 (ownership moved)
    println!("{}", val);
}
// v can no longer be used (ownership has been moved)
```

### 2.3 JavaScript: `Symbol.iterator` Protocol

JavaScript's iterator protocol is composed of the `Symbol.iterator` method and `{ value, done }` objects.

```javascript
class Countdown {
    constructor(start) {
        this.start = start;
    }

    [Symbol.iterator]() {
        let current = this.start;
        return {
            next() {
                if (current <= 0) {
                    return { value: undefined, done: true };
                }
                return { value: current--, done: false };
            }
        };
    }
}

// Using with for-of loop
for (const n of new Countdown(5)) {
    console.log(n);  // 5, 4, 3, 2, 1
}

// Also works with spread syntax
const arr = [...new Countdown(5)];  // [5, 4, 3, 2, 1]

// Also works with destructuring
const [a, b, c] = new Countdown(5);  // a=5, b=4, c=3
```

### 2.4 Java: `Iterator<T>` / `Iterable<T>` Interfaces

```java
import java.util.Iterator;

class Countdown implements Iterable<Integer> {
    private final int start;

    Countdown(int start) {
        this.start = start;
    }

    @Override
    public Iterator<Integer> iterator() {
        return new Iterator<Integer>() {
            int current = start;

            @Override
            public boolean hasNext() {
                return current > 0;
            }

            @Override
            public Integer next() {
                return current--;
            }
        };
    }
}

// Using with enhanced for loop
for (int n : new Countdown(5)) {
    System.out.println(n);  // 5, 4, 3, 2, 1
}
```

### 2.5 Protocol Comparison Across Languages

| Language | Iterable | Iterator | End Signaling | for Loop |
|----------|----------|----------|---------------|----------|
| **Python** | `__iter__()` | `__next__()` | `StopIteration` exception | `for x in iterable:` |
| **Rust** | `IntoIterator` | `Iterator::next()` | `Option::None` | `for x in iterable {}` |
| **JavaScript** | `[Symbol.iterator]()` | `next()` | `{ done: true }` | `for (x of iterable)` |
| **Java** | `Iterable<T>` | `Iterator<T>` | `hasNext() == false` | `for (T x : iterable)` |
| **C++** | `begin()` / `end()` | `operator++` / `operator*` | `iter == end()` | `for (auto x : container)` |
| **C#** | `IEnumerable<T>` | `IEnumerator<T>` | `MoveNext() == false` | `foreach (var x in collection)` |
| **Go** | None (channels as substitute) | None | channel close | `for x := range ch` |

---

## 3. Iterator Adapters and Lazy Transformation Chains

### 3.1 The Concept of Adapter Patterns

Iterator adapters are "functions that take an iterator and return a transformed iterator." The important point is that adapters are **lazily evaluated** -- nothing executes when the chain is assembled. Only when a terminal operation (`collect`, `sum`, `for` loop, etc.) is called does the entire pipeline execute element by element.

```
+-------------------------------------------------------------+
|  Execution Model of Iterator Adapter Chains                  |
+-------------------------------------------------------------+
|                                                             |
|  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]                           |
|       |                                                     |
|       v                                                     |
|  filter(n % 2 == 0)  --- Pass only even numbers             |
|       |                                                     |
|       v                                                     |
|  map(n * n)           --- Square                             |
|       |                                                     |
|       v                                                     |
|  take(3)              --- Stop after first 3                 |
|       |                                                     |
|       v                                                     |
|  collect()            --- Collect results                    |
|                                                             |
|  Per-element execution order:                                |
|  1 -> filter(odd=reject)                                     |
|  2 -> filter(even=pass) -> map(4) -> take(1st) -> [4]       |
|  3 -> filter(odd=reject)                                     |
|  4 -> filter(even=pass) -> map(16) -> take(2nd) -> [4,16]   |
|  5 -> filter(odd=reject)                                     |
|  6 -> filter(even=pass) -> map(36) -> take(3rd) -> [4,16,36]|
|  7 onward is not processed (take(3) satisfied)               |
|                                                             |
|  => Result: [4, 16, 36]                                     |
|  => Only 6 out of 10 elements examined (short-circuit)       |
+-------------------------------------------------------------+
```

### 3.2 Rust's Iterator Adapters

Rust is one of the languages with the most extensive iterator adapter support. All adapters are zero-cost abstractions, inlined at compile time.

```rust
// Basic adapter chain
let result: Vec<i32> = (1..=100)
    .filter(|n| n % 3 == 0)      // Multiples of 3
    .map(|n| n * n)               // Square
    .take(5)                      // First 5
    .collect();
// => [9, 36, 81, 144, 225]

// Important: Nothing executes until collect() is called (lazy evaluation)
// Each element passes through the filter -> map -> take pipeline one at a time
```

**Key transformation adapters (lazy):**

```rust
let v = vec![1, 2, 3, 4, 5];

// map: Transform each element
v.iter().map(|x| x * 2);                    // [2, 4, 6, 8, 10]

// filter: Keep only elements satisfying the condition
v.iter().filter(|x| **x > 2);               // [3, 4, 5]

// filter_map: Simultaneous filter and map (removes None values)
v.iter().filter_map(|x| {
    if *x > 2 { Some(x * 10) } else { None }
}); // [30, 40, 50]

// take / skip: Take/skip the first N elements
v.iter().take(3);                            // [1, 2, 3]
v.iter().skip(2);                            // [3, 4, 5]

// take_while / skip_while: Take/skip based on condition
v.iter().take_while(|x| **x < 4);           // [1, 2, 3]
v.iter().skip_while(|x| **x < 4);           // [4, 5]

// enumerate: Add indices
v.iter().enumerate();                        // [(0,1), (1,2), (2,3), ...]

// zip: Combine two iterators
v.iter().zip(vec![10, 20, 30].iter());       // [(1,10), (2,20), (3,30)]

// chain: Concatenate iterators
v.iter().chain(vec![6, 7].iter());           // [1,2,3,4,5,6,7]

// flat_map: Expand elements into iterators and flatten
v.iter().flat_map(|x| vec![*x, *x * 10]);   // [1,10,2,20,3,30,4,40,5,50]

// flatten: Flatten nested iterators
vec![vec![1,2], vec![3,4]].into_iter().flatten(); // [1,2,3,4]

// peekable: Peek at the next element without consuming
let mut it = v.iter().peekable();
assert_eq!(it.peek(), Some(&&1));            // Just peek
assert_eq!(it.next(), Some(&1));             // Consume

// scan: Stateful transformation (e.g., cumulative sum)
v.iter().scan(0, |acc, x| {
    *acc += x;
    Some(*acc)
}); // [1, 3, 6, 10, 15]

// inspect: For debugging (execute side effects without modifying elements)
v.iter()
    .inspect(|x| println!("before filter: {}", x))
    .filter(|x| **x > 2)
    .inspect(|x| println!("after filter: {}", x))
    .collect::<Vec<_>>();
```

**Key consuming adapters (eager/terminal operations):**

```rust
let v = vec![1, 2, 3, 4, 5];

// collect: Convert to a collection
let vec: Vec<_> = v.iter().collect();
let set: HashSet<_> = v.iter().collect();
let s: String = vec!['a', 'b', 'c'].into_iter().collect();

// sum / product: Total / product
let total: i32 = v.iter().sum();             // 15
let product: i32 = v.iter().product();       // 120

// count: Number of elements
v.iter().count();                             // 5

// any / all: Condition checking
v.iter().any(|x| *x > 3);                   // true
v.iter().all(|x| *x > 0);                   // true

// find: First element satisfying the condition
v.iter().find(|x| **x > 3);                 // Some(&4)

// position: Index of the first element satisfying the condition
v.iter().position(|x| *x > 3);              // Some(3)

// min / max: Minimum/maximum
v.iter().min();                               // Some(&1)
v.iter().max();                               // Some(&5)

// fold: Fold (generalization of reduce)
v.iter().fold(0, |acc, x| acc + x);          // 15

// reduce: Fold without initial value
v.iter().copied().reduce(|a, b| a + b);      // Some(15)

// for_each: Execute side effects for each element
v.iter().for_each(|x| println!("{}", x));
```

### 3.3 Python's Iterator Tools

Python provides iterator operations through built-in functions and the `itertools` module.

```python
import itertools

data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Iterator operations via built-in functions
list(map(lambda x: x ** 2, data))              # [1, 4, 9, 16, 25, ...]
list(filter(lambda x: x % 2 == 0, data))       # [2, 4, 6, 8, 10]
list(zip(data, range(10, 20)))                  # [(1,10), (2,11), ...]
list(enumerate(data))                           # [(0,1), (1,2), ...]
list(reversed(data))                            # [10, 9, 8, ..., 1]
sum(data)                                       # 55
min(data)                                       # 1
max(data)                                       # 10
any(x > 5 for x in data)                       # True
all(x > 0 for x in data)                       # True

# Advanced operations via itertools
list(itertools.chain([1,2], [3,4], [5,6]))      # [1,2,3,4,5,6]
list(itertools.islice(range(100), 5, 10))       # [5,6,7,8,9]
list(itertools.takewhile(lambda x: x < 5, data))  # [1,2,3,4]
list(itertools.dropwhile(lambda x: x < 5, data))  # [5,6,...,10]
list(itertools.accumulate(data))                 # [1,3,6,10,15,...]

# groupby: Group consecutive elements with the same key
sorted_data = sorted(data, key=lambda x: x % 3)
for key, group in itertools.groupby(sorted_data, key=lambda x: x % 3):
    print(f"key={key}: {list(group)}")

# product / permutations / combinations: Combinatorial generation
list(itertools.product('AB', '12'))         # [('A','1'),('A','2'),('B','1'),('B','2')]
list(itertools.permutations('ABC', 2))      # [('A','B'),('A','C'),('B','A'),...]
list(itertools.combinations('ABCD', 2))     # [('A','B'),('A','C'),...]

# tee: Duplicate an iterator
it1, it2 = itertools.tee(iter(data), 2)
# it1 and it2 can traverse independently
```

---

## 4. How Generators Work

### 4.1 What is a Generator?

A generator is "a function that can suspend (yield) and later resume execution." While a regular function follows a one-way path of "call -> execute -> return value," a generator repeats the cycle of "call -> yield value and suspend -> resume -> yield -> ... -> end."

```
+-------------------------------------------------------------+
|  Regular Function vs Generator Function                      |
+-------------------------------------------------------------+
|                                                             |
|  Regular function:                                           |
|  call() --> [start] --> [process] --> return value --> end   |
|         Control returns                                     |
|         to caller                                           |
|                                                             |
|  Generator function:                                         |
|  call() --> Creates a generator object (not yet executed)    |
|                                                             |
|  next() --> [start] --> yield value1 --> suspend (state kept)|
|         Control returns              ^                      |
|         to caller                    |                      |
|                                      |                      |
|  next() --> [resume] --> [process] --> yield value2 --> suspend|
|         Control returns              ^                      |
|         to caller                    |                      |
|                                      |                      |
|  next() --> [resume] --> [process] --> return --> StopIteration|
|                                                             |
|  Preserved state: local variables, execution position       |
|  (program counter)                                          |
+-------------------------------------------------------------+
```

### 4.2 Python Generators

```python
# Basic generator
def fibonacci():
    """Generator that infinitely produces Fibonacci numbers"""
    a, b = 0, 1
    while True:
        yield a           # Return value and suspend
        a, b = b, a + b   # Continue from here on resume

# Usage: Infinite sequence, but only generates what's needed
fib = fibonacci()
for _ in range(10):
    print(next(fib))  # 0, 1, 1, 2, 3, 5, 8, 13, 21, 34

# Generator expression (generator version of comprehension)
# Comparison with list comprehension
squares_list = [x**2 for x in range(1_000_000)]  # Holds all elements in memory
squares_gen  = (x**2 for x in range(1_000_000))  # Generates one element at a time (memory-efficient)

# Generator expressions have an extremely small memory footprint
import sys
print(sys.getsizeof(squares_list))  # Approximately 8.5 MB
print(sys.getsizeof(squares_gen))   # Approximately 200 bytes (constant)
```

### 4.3 yield from: Sub-Generator Delegation

```python
# Without yield from
def chain_v1(*iterables):
    for it in iterables:
        for item in it:
            yield item

# With yield from (equivalent but more efficient)
def chain_v2(*iterables):
    for it in iterables:
        yield from it

list(chain_v2([1, 2], [3, 4], [5, 6]))  # [1, 2, 3, 4, 5, 6]

# yield from is particularly useful for recursive generators
def flatten(nested):
    """Flatten a nested list"""
    for item in nested:
        if isinstance(item, (list, tuple)):
            yield from flatten(item)  # Recursively delegate
        else:
            yield item

list(flatten([1, [2, [3, 4], 5], [6, 7]]))  # [1, 2, 3, 4, 5, 6, 7]
```

### 4.4 Bidirectional Communication with send()

Generators can also receive values from the caller using the `send()` method.

```python
def accumulator():
    """Generator that accumulates sent values"""
    total = 0
    while True:
        value = yield total    # Return total and receive the sent value into value
        if value is None:
            break
        total += value

acc = accumulator()
next(acc)            # 0 (Advance the generator to the first yield)
acc.send(10)         # 10 (total = 0 + 10)
acc.send(20)         # 30 (total = 10 + 20)
acc.send(5)          # 35 (total = 30 + 5)

# Practical example: Generator as a coroutine (moving average calculation)
def moving_average(window_size):
    """Coroutine that calculates the moving average"""
    window = []
    average = None
    while True:
        value = yield average
        window.append(value)
        if len(window) > window_size:
            window.pop(0)
        average = sum(window) / len(window)

ma = moving_average(3)
next(ma)         # None (initialization)
ma.send(10)      # 10.0
ma.send(20)      # 15.0
ma.send(30)      # 20.0
ma.send(40)      # 30.0 (window: [20, 30, 40])
```

### 4.5 JavaScript Generators

```javascript
// Define a generator function with function*
function* fibonacci() {
    let a = 0, b = 1;
    while (true) {
        yield a;
        [a, b] = [b, a + b];
    }
}

const fib = fibonacci();
console.log(fib.next());  // { value: 0, done: false }
console.log(fib.next());  // { value: 1, done: false }
console.log(fib.next());  // { value: 1, done: false }

// Helper generator: take
function* take(iter, n) {
    let count = 0;
    for (const item of iter) {
        if (count++ >= n) break;
        yield item;
    }
}

// Consume with for-of
for (const n of take(fibonacci(), 10)) {
    console.log(n);  // 0, 1, 1, 2, 3, 5, 8, 13, 21, 34
}

// Delegate with yield*
function* flatten(arr) {
    for (const item of arr) {
        if (Array.isArray(item)) {
            yield* flatten(item);  // Recursively delegate
        } else {
            yield item;
        }
    }
}

console.log([...flatten([1, [2, [3, 4], 5], [6, 7]])]);
// [1, 2, 3, 4, 5, 6, 7]

// Equivalent of send: next(value)
function* accumulator() {
    let total = 0;
    while (true) {
        const value = yield total;
        total += value;
    }
}

const acc = accumulator();
acc.next();        // { value: 0, done: false } -- Initialization
acc.next(10);      // { value: 10, done: false }
acc.next(20);      // { value: 30, done: false }
acc.next(5);       // { value: 35, done: false }
```

### 4.6 Relationship Between Generators and Coroutines

Generators are also called "semi-coroutines." While full coroutines can yield control to any other coroutine, generators can only return control to their caller.

```
+-------------------------------------------------------------+
|  Classification of Coroutines                                |
+-------------------------------------------------------------+
|                                                             |
|  Symmetric Coroutine                                        |
|    - Can yield control to any coroutine                     |
|    - Free transitions like A -> B -> C -> A                 |
|    - Examples: Lua coroutines, Go goroutines (similar)      |
|                                                             |
|  Asymmetric Coroutine (Semi-Coroutine)                      |
|    - Can only return control to the caller                  |
|    - Only round-trips between caller <-> generator          |
|    - Examples: Python / JavaScript generators               |
|                                                             |
|  +------------------+    +------------------+               |
|  | Caller           |    | Generator        |               |
|  |                  |--->|  Executing...    |               |
|  |                  |    |  yield value     |               |
|  |  Receives value  |<---|                  |               |
|  |  ...             |    |  (suspended)     |               |
|  |  next()          |--->|  Resumes...      |               |
|  |                  |    |  yield value     |               |
|  |  Receives value  |<---|                  |               |
|  +------------------+    +------------------+               |
+-------------------------------------------------------------+
```

Python's `async/await` syntax is actually built on top of generators. Historically, before Python 3.4, coroutines were implemented using the `@asyncio.coroutine` decorator and `yield from`. Python 3.5 introduced `async def` / `await`, separating them syntactically, but internally they use the same suspend/resume mechanism as generators.

---

## 5. Lazy Evaluation vs Eager Evaluation

### 5.1 Fundamental Concepts of Evaluation Strategies

There are broadly two strategies for when expressions in a programming language are evaluated.

```
+-------------------------------------------------------------+
|  Eager / Strict Evaluation                                   |
+-------------------------------------------------------------+
|                                                             |
|  Expressions are evaluated immediately when defined          |
|                                                             |
|  Example: result = [x*2 for x in range(1000000)]           |
|                                                             |
|  1. range(1000000) generates 0 to 999999                    |
|  2. Applies x*2 to each element                             |
|  3. Holds the resulting 1 million elements in memory         |
|                                                             |
|  Memory usage: O(N)                                          |
|  Computation cost: All elements processed upfront            |
|  Advantage: Predictable, easier to debug                     |
+-------------------------------------------------------------+

+-------------------------------------------------------------+
|  Lazy Evaluation                                             |
+-------------------------------------------------------------+
|                                                             |
|  Evaluation is deferred until the value is actually needed   |
|                                                             |
|  Example: result = (x*2 for x in range(1000000))           |
|                                                             |
|  1. Creates a generator object (nothing computed yet)        |
|  2. Computes one element at a time when next() is called     |
|  3. Unused elements are never computed or allocated           |
|                                                             |
|  Memory usage: O(1)                                          |
|  Computation cost: Only processes what's needed              |
|  Advantage: Infinite data structures, memory-efficient,      |
|             avoids unnecessary computation                   |
+-------------------------------------------------------------+
```

### 5.2 Evaluation Strategies by Language

| Language | Default Evaluation | Lazy Evaluation Mechanism | Eager Evaluation Mechanism |
|----------|-------------------|--------------------------|---------------------------|
| **Haskell** | Lazy | Default | `seq`, `deepseq`, `BangPatterns` |
| **Rust** | Eager | `Iterator` chains | `collect()`, `sum()`, `for` |
| **Python** | Eager | Generators, `itertools` | `list()`, `tuple()` |
| **JavaScript** | Eager | Generators | `Array.from()`, spread `[...]` |
| **Scala** | Eager | `LazyList`, `View` | `.toList`, `.toVector` |
| **C#** | Eager | `IEnumerable<T>` (LINQ) | `.ToList()`, `.ToArray()` |
| **Kotlin** | Eager | `Sequence<T>` | `.toList()` |
| **Java** | Eager | `Stream<T>` | `.collect()`, `.toList()` |

### 5.3 Advantages and Caveats of Lazy Evaluation

**Advantage 1: Representing Infinite Sequences**

```python
def natural_numbers():
    """Infinitely generates natural numbers"""
    n = 1
    while True:
        yield n
        n += 1

def primes():
    """Infinitely generates primes (variant of the Sieve of Eratosthenes)"""
    yield 2
    composites = {}
    n = 3
    while True:
        if n not in composites:
            yield n
            composites[n * n] = [n]
        else:
            for prime in composites[n]:
                composites.setdefault(prime + n, []).append(prime)
            del composites[n]
        n += 2

# Get the first 20 primes
from itertools import islice
print(list(islice(primes(), 20)))
# [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
```

**Advantage 2: Efficient Pipeline Processing**

```rust
// Due to lazy evaluation, processing stops once the take(5) condition is met
// No need to process all 100,000 elements
let result: Vec<i32> = (1..=100_000)
    .filter(|n| is_prime(*n))    // Primes only
    .map(|n| n * n)              // Square
    .take(5)                     // Stop at the first 5!
    .collect();
// is_prime is only called until the first 5 primes are found
// => [4, 9, 25, 49, 121]
```

**Caveat 1: Interaction with Side Effects**

```python
# Danger: Combining lazy evaluation with side effects
def log_and_double(x):
    print(f"Processing: {x}")  # Side effect
    return x * 2

gen = (log_and_double(x) for x in range(5))
# Nothing is output at this point!

# Side effects only occur when the generator is consumed
result = list(gen)
# Processing: 0
# Processing: 1
# Processing: 2
# Processing: 3
# Processing: 4
```

**Caveat 2: Multiple Traversals**

```python
gen = (x ** 2 for x in range(5))

# First traversal
print(sum(gen))   # 30 (0 + 1 + 4 + 9 + 16)

# Second traversal (empty! Generators are single-use)
print(sum(gen))   # 0 --- A common bug
```

### 5.4 Haskell's Lazy Evaluation

Haskell is the only major language that adopts lazy evaluation by default.

```haskell
-- Infinite lists can be written naturally
naturals = [1..]                    -- [1, 2, 3, 4, ...]
evens    = [2, 4..]                 -- [2, 4, 6, 8, ...]
fibs     = 0 : 1 : zipWith (+) fibs (tail fibs)  -- Fibonacci sequence

-- Take only what's needed
take 10 fibs    -- [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
take 5 naturals -- [1, 2, 3, 4, 5]

-- Elegant expressions enabled by infinite lists
-- "The first 10 primes"
take 10 [n | n <- [2..], isPrime n]

-- Definitions possible only because of lazy evaluation
-- ones = 1 : ones  -- [1, 1, 1, 1, ...] An infinite list of 1s
```

---

## 6. Async Iterators and Streams

### 6.1 Motivation for Async Iterators

Regular iterators produce values synchronously. However, values arrive asynchronously in cases like:

- Data streams from a network
- Line-by-line file reading
- Row fetching from a database cursor
- WebSocket message reception
- Responses from paginated APIs

Async iterators are needed to handle these cases.

```
+-------------------------------------------------------------+
|  Synchronous Iterator vs Async Iterator                      |
+-------------------------------------------------------------+
|                                                             |
|  Synchronous iterator:                                       |
|  next() --> Value returned immediately (blocking)            |
|  next() --> Value returned immediately                       |
|  next() --> End                                              |
|                                                             |
|  Async iterator:                                             |
|  next() --> Returns a Future/Promise                         |
|         --> Await the value (non-blocking)                   |
|  next() --> Returns a Future/Promise                         |
|         --> Await the value                                  |
|  next() --> End                                              |
|                                                             |
|  Other tasks can execute while waiting                       |
+-------------------------------------------------------------+
```

### 6.2 JavaScript Async Iterators

```javascript
// Async generator with async function*
async function* fetchPages(baseUrl) {
    let page = 1;
    while (true) {
        const response = await fetch(`${baseUrl}?page=${page}`);
        const data = await response.json();
        if (data.items.length === 0) break;
        yield data.items;       // Yield each page's data
        page++;
    }
}

// Consume with for await...of
async function processAllUsers(url) {
    for await (const users of fetchPages(url)) {
        for (const user of users) {
            console.log(user.name);
        }
    }
}

// Symbol.asyncIterator protocol
class EventStream {
    constructor(eventSource) {
        this.source = eventSource;
    }

    [Symbol.asyncIterator]() {
        const source = this.source;
        return {
            next() {
                return new Promise((resolve) => {
                    source.once('data', (data) => {
                        resolve({ value: data, done: false });
                    });
                    source.once('end', () => {
                        resolve({ value: undefined, done: true });
                    });
                });
            }
        };
    }
}

// Async generator utilities
async function* asyncMap(asyncIter, fn) {
    for await (const item of asyncIter) {
        yield fn(item);
    }
}

async function* asyncFilter(asyncIter, predicate) {
    for await (const item of asyncIter) {
        if (predicate(item)) {
            yield item;
        }
    }
}

async function* asyncTake(asyncIter, n) {
    let count = 0;
    for await (const item of asyncIter) {
        if (count++ >= n) break;
        yield item;
    }
}

// Combined usage
const activeUsers = asyncFilter(
    asyncMap(
        fetchPages("/api/users"),
        page => page.filter(u => u.active)
    ),
    page => page.length > 0
);
```

### 6.3 Python Async Generators

```python
import aiohttp
import asyncio

# async def + yield = async generator
async def fetch_pages(base_url):
    """Asynchronously fetch data from a paginated API"""
    page = 1
    async with aiohttp.ClientSession() as session:
        while True:
            async with session.get(f"{base_url}?page={page}") as resp:
                data = await resp.json()
                if not data["items"]:
                    break
                yield data["items"]
                page += 1

# Consume with async for
async def process_all_users():
    async for users in fetch_pages("https://api.example.com/users"):
        for user in users:
            print(user["name"])

asyncio.run(process_all_users())

# Async comprehension
async def get_active_users():
    return [
        user
        async for page in fetch_pages("/api/users")
        for user in page
        if user.get("active")
    ]

# Explicit cleanup with aclose() for async generators
async def resource_stream():
    resource = await acquire_resource()
    try:
        while True:
            data = await resource.read()
            if data is None:
                break
            yield data
    finally:
        await resource.release()  # Cleanup is guaranteed

# Cleanup executes even on early termination with aclose()
stream = resource_stream()
first_item = await stream.__anext__()
await stream.aclose()  # The finally block executes
```

### 6.4 Rust Streams (Async Iterators)

```rust
use tokio_stream::{self, StreamExt};
use tokio::time::{self, Duration};

// Stream is the async version of Iterator
// poll_next() returns a Future

// Example using tokio_stream
#[tokio::main]
async fn main() {
    // Creating and consuming a basic Stream
    let mut stream = tokio_stream::iter(vec![1, 2, 3, 4, 5])
        .filter(|n| *n % 2 == 0)
        .map(|n| n * 10);

    while let Some(value) = stream.next().await {
        println!("{}", value);  // 20, 40
    }

    // Interval stream
    let mut interval = tokio_stream::wrappers::IntervalStream::new(
        time::interval(Duration::from_secs(1))
    );

    // Process only the first 5 ticks
    let mut count = 0;
    while let Some(_tick) = interval.next().await {
        count += 1;
        println!("Tick {}", count);
        if count >= 5 { break; }
    }
}

// Stream generation using the async-stream crate
// (Rust does not yet have native async generator syntax)
use async_stream::stream;

fn countdown(from: u32) -> impl tokio_stream::Stream<Item = u32> {
    stream! {
        for i in (1..=from).rev() {
            tokio::time::sleep(Duration::from_millis(100)).await;
            yield i;
        }
    }
}
```

### 6.5 Async Iterator Comparison

| Property | JavaScript | Python | Rust |
|----------|-----------|--------|------|
| Syntax | `async function*` | `async def` + `yield` | `async-stream` crate |
| Consumption | `for await...of` | `async for` | `while let Some(x) = stream.next().await` |
| Protocol | `Symbol.asyncIterator` | `__aiter__` / `__anext__` | `Stream` trait |
| Delegation | `yield*` | `async for` + `yield` | None (manual) |
| Native support | ES2018 | Python 3.6 | Unstable (nightly) |
| Error handling | `try-catch` | `try-except` | `Result<Option<T>>` |

---

## 7. Practical Patterns

### 7.1 Pipeline Pattern

Build data processing pipelines with iterators, implementing the Unix pipe concept within a program.

```python
import csv
from typing import Iterator, Dict, Any

def read_csv(filename: str) -> Iterator[Dict[str, str]]:
    """Generator that reads a CSV file one row at a time"""
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row
    # File is automatically closed

def parse_numbers(rows: Iterator[Dict]) -> Iterator[Dict]:
    """Convert numeric fields"""
    for row in rows:
        row["age"] = int(row["age"])
        row["salary"] = float(row["salary"])
        yield row

def filter_by_age(rows: Iterator[Dict], min_age: int) -> Iterator[Dict]:
    """Filter by age"""
    for row in rows:
        if row["age"] >= min_age:
            yield row

def top_n(rows: Iterator[Dict], n: int, key: str) -> list:
    """Get the top N entries (memory-efficient using a heap)"""
    import heapq
    return heapq.nlargest(n, rows, key=lambda r: r[key])

# Building and executing the pipeline
# Memory usage remains constant even for a million-row CSV
pipeline = read_csv("employees.csv")
pipeline = parse_numbers(pipeline)
pipeline = filter_by_age(pipeline, 30)
top_earners = top_n(pipeline, 10, "salary")

for emp in top_earners:
    print(f"{emp['name']}: ${emp['salary']:,.0f}")
```

### 7.2 Windowing Pattern

Process time-series or continuous data by sliding a fixed-width window.

```python
from collections import deque
from typing import Iterator, TypeVar, Tuple
import itertools

T = TypeVar('T')

def sliding_window(iterable, size: int) -> Iterator[Tuple]:
    """Sliding window"""
    it = iter(iterable)
    window = deque(itertools.islice(it, size), maxlen=size)
    if len(window) == size:
        yield tuple(window)
    for item in it:
        window.append(item)
        yield tuple(window)

# Usage: Moving average
data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
for window in sliding_window(data, 3):
    avg = sum(window) / len(window)
    print(f"window={window}, avg={avg:.1f}")
# window=(10, 20, 30), avg=20.0
# window=(20, 30, 40), avg=30.0
# window=(30, 40, 50), avg=40.0
# ...

def chunked(iterable, size: int) -> Iterator[list]:
    """Split an iterable into fixed-size chunks"""
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk

# Usage: Batch processing
for batch in chunked(range(100), 10):
    process_batch(batch)  # Process 10 at a time
```

### 7.3 Tree Traversal Pattern

Traverse recursive data structures flat using iterators/generators.

```python
class TreeNode:
    def __init__(self, value, children=None):
        self.value = value
        self.children = children or []

def dfs_preorder(node: TreeNode) -> Iterator:
    """Depth-first search (preorder) implemented with a generator"""
    yield node.value
    for child in node.children:
        yield from dfs_preorder(child)

def dfs_postorder(node: TreeNode) -> Iterator:
    """Depth-first search (postorder)"""
    for child in node.children:
        yield from dfs_postorder(child)
    yield node.value

def bfs(node: TreeNode) -> Iterator:
    """Breadth-first search implemented with a generator"""
    from collections import deque
    queue = deque([node])
    while queue:
        current = queue.popleft()
        yield current.value
        queue.extend(current.children)

# Build tree
tree = TreeNode(1, [
    TreeNode(2, [TreeNode(4), TreeNode(5)]),
    TreeNode(3, [TreeNode(6), TreeNode(7)])
])

print(list(dfs_preorder(tree)))   # [1, 2, 4, 5, 3, 6, 7]
print(list(dfs_postorder(tree)))  # [4, 5, 2, 6, 7, 3, 1]
print(list(bfs(tree)))            # [1, 2, 3, 4, 5, 6, 7]
```

### 7.4 State Machine Pattern

Use generators to express state machines simply.

```python
def lexer(text: str):
    """Simple tokenizer implemented as a state machine"""
    i = 0
    while i < len(text):
        # Skip whitespace
        if text[i].isspace():
            i += 1
            continue

        # Number token
        if text[i].isdigit():
            start = i
            while i < len(text) and text[i].isdigit():
                i += 1
            yield ("NUMBER", text[start:i])
            continue

        # Identifier token
        if text[i].isalpha():
            start = i
            while i < len(text) and text[i].isalnum():
                i += 1
            yield ("IDENT", text[start:i])
            continue

        # Operator token
        if text[i] in "+-*/=<>":
            yield ("OP", text[i])
            i += 1
            continue

        raise ValueError(f"Unexpected character: {text[i]}")

# Usage
for token_type, value in lexer("x = 42 + y * 3"):
    print(f"{token_type}: {value}")
# IDENT: x
# OP: =
# NUMBER: 42
# OP: +
# IDENT: y
# OP: *
# NUMBER: 3
```

### 7.5 Resource Management Pattern

Combine generators with context managers to guarantee reliable resource cleanup.

```python
from contextlib import contextmanager

@contextmanager
def managed_cursor(connection):
    """Manage the lifecycle of a database cursor"""
    cursor = connection.cursor()
    try:
        yield cursor
    finally:
        cursor.close()

def query_rows(connection, sql, params=None):
    """Generator that fetches large result sets one row at a time"""
    with managed_cursor(connection) as cursor:
        cursor.execute(sql, params or ())
        while True:
            row = cursor.fetchone()
            if row is None:
                break
            yield row
    # cursor.close() is called when exiting the with block

# Process a million-row query result memory-efficiently
for row in query_rows(conn, "SELECT * FROM large_table"):
    process_row(row)
# Even if you break mid-iteration, the cursor is reliably closed
```

---

## 8. Anti-Patterns and Pitfalls

### 8.1 Anti-Pattern 1: Double Consumption of Generators

```python
# =========================================
# ANTI-PATTERN: Double consumption of a generator
# =========================================

def get_numbers():
    yield from range(10)

numbers = get_numbers()

# First time: works correctly
total = sum(numbers)          # 45

# Second time: empty generator! Value becomes 0
average = total / sum(numbers) if sum(numbers) > 0 else 0
# ZeroDivisionError or unexpected result

# -----------------------------------------
# Fix 1: Convert to list for reuse
# -----------------------------------------
numbers = list(get_numbers())  # Hold all elements in memory
total = sum(numbers)           # 45
average = total / len(numbers) # 4.5

# -----------------------------------------
# Fix 2: Duplicate with itertools.tee
# -----------------------------------------
import itertools
nums1, nums2 = itertools.tee(get_numbers(), 2)
total = sum(nums1)
count = sum(1 for _ in nums2)
average = total / count  # 4.5
# Note: tee uses an internal buffer, so it's not suitable for large iterators

# -----------------------------------------
# Fix 3: Define an iterable class (recommended)
# -----------------------------------------
class NumberRange:
    """An iterable that can be iterated any number of times"""
    def __iter__(self):
        yield from range(10)

numbers = NumberRange()
total = sum(numbers)           # 45 (1st time)
average = total / sum(1 for _ in numbers)  # 4.5 (2nd time works too)
```

### 8.2 Anti-Pattern 2: Lazy Evaluation and Closure Variable Trap

```python
# =========================================
# ANTI-PATTERN: Closure capture of loop variables
# =========================================

# Expected: A list of generators producing [0, 1, 4, 9, 16]
generators = [lambda: i ** 2 for i in range(5)]

# All return 16! (i references the final value 4)
results = [g() for g in generators]
print(results)  # [16, 16, 16, 16, 16]

# -----------------------------------------
# Fix 1: Capture the value with a default argument
# -----------------------------------------
generators = [lambda i=i: i ** 2 for i in range(5)]
results = [g() for g in generators]
print(results)  # [0, 1, 4, 9, 16]

# -----------------------------------------
# Fix 2: Use a generator expression
# -----------------------------------------
gen = (i ** 2 for i in range(5))
print(list(gen))  # [0, 1, 4, 9, 16]
```

### 8.3 Anti-Pattern 3: Unnecessary Intermediate List Creation

```python
# =========================================
# ANTI-PATTERN: Unnecessary intermediate list creation
# =========================================

# Bad example: Full list generated at each step
data = list(range(1_000_000))
filtered = [x for x in data if x % 2 == 0]        # 500K element list
mapped   = [x ** 2 for x in filtered]              # 500K element list
result   = [x for x in mapped if x > 1000]         # Another list
final    = sum(result)
# Memory: original data + filtered + mapped + result = ~4x

# -----------------------------------------
# Fix: Pipeline with generator expressions
# -----------------------------------------
data = range(1_000_000)                             # range is lazy
filtered = (x for x in data if x % 2 == 0)         # Generator
mapped   = (x ** 2 for x in filtered)               # Generator
result   = (x for x in mapped if x > 1000)          # Generator
final    = sum(result)                               # Computation happens here
# Memory: O(1) --- no intermediate lists
```

### 8.4 Anti-Pattern 4: Missing Exception Handling in Generators

```python
# =========================================
# ANTI-PATTERN: Generator's finally not executing
# =========================================

def file_lines(path):
    """Generator that reads a file line by line"""
    f = open(path)
    try:
        for line in f:
            yield line.strip()
    finally:
        f.close()
        print("File closed")

# Problem: When a generator is abandoned midway...
gen = file_lines("data.txt")
first_line = next(gen)
# finally is not executed until gen is garbage collected
# CPython's reference counting GCs immediately, but
# PyPy and other implementations may delay

# -----------------------------------------
# Fix: Call close() explicitly, or use a context manager
# -----------------------------------------

# Method 1: Call close() explicitly
gen = file_lines("data.txt")
first_line = next(gen)
gen.close()  # The finally block executes

# Method 2: Use contextlib.closing
from contextlib import closing
with closing(file_lines("data.txt")) as lines:
    first_line = next(lines)
# close() is automatically called when exiting the with block
```

---

## 9. Performance Characteristics and Optimization

### 9.1 Zero-Cost Abstraction of Iterators (Rust)

Rust iterator chains are inlined at compile time, generating machine code equivalent to hand-written loops. This is called "Zero-Cost Abstraction."

```rust
// Iterator chain version
let sum: i32 = (0..1000)
    .filter(|n| n % 3 == 0 || n % 5 == 0)
    .sum();

// Hand-written loop version (equivalent performance)
let mut sum = 0;
for n in 0..1000 {
    if n % 3 == 0 || n % 5 == 0 {
        sum += n;
    }
}

// The compiler generates nearly identical machine code
// The iterator version may even be faster due to eliding bounds checks
```

```
+-------------------------------------------------------------+
|  Rust Iterator Compilation Process                           |
+-------------------------------------------------------------+
|                                                             |
|  Source code                                                 |
|    (0..1000).filter(...).map(...).sum()                      |
|         |                                                   |
|    Monomorphization (type specialization)                    |
|         |                                                   |
|    Inlining (expand function calls)                          |
|         |                                                   |
|    LLVM optimization passes                                  |
|         |                                                   |
|    Machine code equivalent to hand-written loops             |
|         |                                                   |
|  Important: No heap allocations at runtime                   |
|  Important: No virtual function calls (vtable) either        |
+-------------------------------------------------------------+
```

### 9.2 Memory Usage Comparison

The following compares memory usage when processing 1 million data items.

| Approach | Memory Usage | Description |
|----------|-------------|-------------|
| Hold all elements in a list | O(N) -- ~8MB | `list(range(1_000_000))` |
| Generator pipeline | O(1) -- ~200B | `(x for x in range(1_000_000))` |
| `itertools` chain | O(1) -- ~400B | `itertools.chain(...)` |
| `map`/`filter` (Python 3) | O(1) -- ~200B | `map(fn, range(1_000_000))` |
| Rust iterator | O(1) -- on stack | `(0..1_000_000).filter(...)` |

### 9.3 Computational Complexity of Iterator Operations

```
+-------------------------------------------------------------+
|  Computational Complexity of Key Iterator Operations         |
+-------------------------------------------------------------+
|                                                             |
|  Operation        | Time        | Space       | Notes       |
|  -----------------|-------------|-------------|-------------|
|  map              | O(1)/elem   | O(1)        | Lazy        |
|  filter           | O(1)/elem   | O(1)        | Lazy        |
|  take(n)          | O(1)/elem   | O(1)        | Lazy,       |
|                   |             |             | short-circuit|
|  skip(n)          | O(n) first  | O(1)        | Lazy        |
|  zip              | O(1)/elem   | O(1)        | Lazy        |
|  chain            | O(1)/elem   | O(1)        | Lazy        |
|  enumerate        | O(1)/elem   | O(1)        | Lazy        |
|  collect          | O(N)        | O(N)        | Eagerly     |
|                   |             |             | materializes|
|  sum              | O(N)        | O(1)        | Terminal    |
|  count            | O(N)        | O(1)        | Terminal    |
|  find             | O(N) worst  | O(1)        | Can         |
|                   |             |             | short-circuit|
|  any/all          | O(N) worst  | O(1)        | Can         |
|                   |             |             | short-circuit|
|  sort_by          | O(N log N)  | O(N)        | Needs all   |
|                   |             |             | elements    |
|  group_by         | O(N)        | O(N)        | Needs all   |
|                   |             |             | elements    |
|  tee (Python)     | O(1)/elem   | O(N) worst  | Buffers     |
|                   |             |             | differences |
+-------------------------------------------------------------+
```

### 9.4 Performance Optimization Best Practices

```python
# Optimization 1: Generator expressions vs list comprehensions
# Use generator expressions when you don't need all results

# Bad example (unnecessary intermediate list)
has_error = any([line.startswith("ERROR") for line in open("log.txt")])
# Even if "ERROR" is found, all remaining lines are processed

# Good example (generator expression + short-circuit evaluation)
has_error = any(line.startswith("ERROR") for line in open("log.txt"))
# Stops as soon as "ERROR" is found

# Optimization 2: itertools is implemented in C and is fast
import itertools

# Slow (Python loop)
def my_chain(*iterables):
    for it in iterables:
        yield from it

# Fast (C implementation)
itertools.chain(*iterables)

# Optimization 3: Use heapq.nlargest() instead of sorted() for large data
import heapq

# Full sort is O(N log N)
top_10 = sorted(huge_data, key=extract_key, reverse=True)[:10]

# Heap is O(N log K), faster when K is small
top_10 = heapq.nlargest(10, huge_data, key=extract_key)
```

```rust
// Optimization techniques in Rust

// Optimization 1: Hints for collect
// Pre-allocate Vec capacity when size is known
let result: Vec<i32> = Vec::with_capacity(1000);
let result: Vec<i32> = (0..1000).collect();  // Auto-allocates using size_hint

// Optimization 2: Avoid unnecessary clones
// Bad example
let names: Vec<String> = people.iter()
    .map(|p| p.name.clone())  // Clone every time
    .collect();

// Good example (when references suffice)
let names: Vec<&str> = people.iter()
    .map(|p| p.name.as_str())  // References only
    .collect();

// Optimization 3: flat_map vs flatten
// flat_map performs map + flatten in one step, skipping intermediate iterators
let result: Vec<i32> = data.iter()
    .flat_map(|row| row.iter().copied())  // Efficient
    .collect();
```

---

## 10. Exercises

### 10.1 Beginner Level

**Exercise B-1: Implementing a Custom Iterator**

Implement an iterator that generates only even numbers within a specified range.

```python
class EvenRange:
    """Iterator that generates only even numbers"""
    def __init__(self, start, end):
        # TODO: Implement
        pass

    def __iter__(self):
        # TODO: Implement
        pass

    def __next__(self):
        # TODO: Implement
        pass

# Test
assert list(EvenRange(1, 10)) == [2, 4, 6, 8]
assert list(EvenRange(0, 6)) == [0, 2, 4]
assert list(EvenRange(7, 8)) == [8]
```

<details>
<summary>Solution (click to expand)</summary>

```python
class EvenRange:
    def __init__(self, start, end):
        # Adjust start to the first even number
        self.current = start if start % 2 == 0 else start + 1
        self.end = end

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.end:
            raise StopIteration
        value = self.current
        self.current += 2
        return value

# Alternative: Generator function (simpler)
def even_range(start, end):
    n = start if start % 2 == 0 else start + 1
    while n < end:
        yield n
        n += 2
```

</details>

**Exercise B-2: FizzBuzz with Generators**

Generate a FizzBuzz sequence using generators.

```python
def fizzbuzz(n):
    """Generator that produces FizzBuzz from 1 to n"""
    # TODO: Implement
    pass

# Test
result = list(fizzbuzz(15))
assert result == [
    1, 2, "Fizz", 4, "Buzz", "Fizz", 7, 8, "Fizz", "Buzz",
    11, "Fizz", 13, 14, "FizzBuzz"
]
```

<details>
<summary>Solution (click to expand)</summary>

```python
def fizzbuzz(n):
    for i in range(1, n + 1):
        if i % 15 == 0:
            yield "FizzBuzz"
        elif i % 3 == 0:
            yield "Fizz"
        elif i % 5 == 0:
            yield "Buzz"
        else:
            yield i
```

</details>

### 10.2 Intermediate Level

**Exercise I-1: Building Custom Iterator Adapters**

Implement Rust-style iterator adapters in Python.

```python
class LazyIter:
    """Lazy evaluation iterator wrapper"""

    def __init__(self, iterable):
        self._iter = iter(iterable)

    def __iter__(self):
        return self._iter

    def __next__(self):
        return next(self._iter)

    def map(self, fn):
        # TODO: Return a new LazyIter
        pass

    def filter(self, pred):
        # TODO: Return a new LazyIter
        pass

    def take(self, n):
        # TODO: Return a new LazyIter
        pass

    def enumerate(self):
        # TODO: Return a new LazyIter
        pass

    def collect(self):
        # TODO: Convert to list
        pass

    def sum(self):
        # TODO: Calculate total
        pass

    def any(self, pred):
        # TODO: Check if any element satisfies the condition
        pass

# Test
result = (LazyIter(range(100))
    .filter(lambda x: x % 3 == 0)
    .map(lambda x: x * x)
    .take(5)
    .collect())

assert result == [0, 9, 36, 81, 144]
```

<details>
<summary>Solution (click to expand)</summary>

```python
class LazyIter:
    def __init__(self, iterable):
        self._iter = iter(iterable)

    def __iter__(self):
        return self._iter

    def __next__(self):
        return next(self._iter)

    def map(self, fn):
        def _map():
            for item in self._iter:
                yield fn(item)
        return LazyIter(_map())

    def filter(self, pred):
        def _filter():
            for item in self._iter:
                if pred(item):
                    yield item
        return LazyIter(_filter())

    def take(self, n):
        def _take():
            count = 0
            for item in self._iter:
                if count >= n:
                    break
                yield item
                count += 1
        return LazyIter(_take())

    def enumerate(self):
        def _enumerate():
            for i, item in __builtins__'enumerate':
                yield (i, item)
        return LazyIter(_enumerate())

    def collect(self):
        return list(self._iter)

    def sum(self):
        return sum(self._iter)

    def any(self, pred):
        return any(pred(item) for item in self._iter)
```

</details>

**Exercise I-2: Composing Infinite Sequences**

Implement the following infinite sequence with generators.

```python
def collatz(n):
    """Generate the Collatz sequence.
    If n is even, divide by 2; if odd, compute 3n+1. Stop when 1 is reached.
    """
    # TODO: Implement
    pass

# Test
assert list(collatz(6)) == [6, 3, 10, 5, 16, 8, 4, 2, 1]
assert list(collatz(1)) == [1]
```

<details>
<summary>Solution (click to expand)</summary>

```python
def collatz(n):
    yield n
    while n != 1:
        if n % 2 == 0:
            n //= 2
        else:
            n = 3 * n + 1
        yield n
```

</details>

### 10.3 Advanced Level

**Exercise A-1: Async Pipeline**

Build a pipeline that merges inputs from multiple data sources using async generators.

```python
import asyncio

async def merge(*async_iterables):
    """Merge multiple async iterables and yield in arrival order"""
    # TODO: Implement
    # Hint: Use asyncio.Queue and asyncio.create_task
    pass

# Test async generators
async def delayed_range(name, start, end, delay):
    for i in range(start, end):
        await asyncio.sleep(delay)
        yield (name, i)

# Usage
async def main():
    async for source, value in merge(
        delayed_range("A", 0, 5, 0.3),
        delayed_range("B", 10, 15, 0.5),
    ):
        print(f"{source}: {value}")
    # Output in arrival order

asyncio.run(main())
```

<details>
<summary>Solution (click to expand)</summary>

```python
import asyncio

async def merge(*async_iterables):
    queue = asyncio.Queue()
    sentinel = object()  # End marker
    active = len(async_iterables)

    async def producer(ait):
        nonlocal active
        async for item in ait:
            await queue.put(item)
        active -= 1
        if active == 0:
            await queue.put(sentinel)

    # Start all producers
    tasks = [asyncio.create_task(producer(ait)) for ait in async_iterables]

    # Read from the queue
    while True:
        item = await queue.get()
        if item is sentinel:
            break
        yield item

    # Wait for tasks to complete
    await asyncio.gather(*tasks)
```

</details>

---

## 11. FAQ (Frequently Asked Questions)

### Q1: What is the difference between iterators and generators?

**A:** An iterator is an "interface/protocol for retrieving elements one at a time," and a generator is "syntactic sugar" for easily creating that iterator.

- **Iterator**: An object that explicitly implements `__iter__` / `__next__` (Python) or the `Iterator` trait (Rust). State management must be done manually.
- **Generator**: A function containing the `yield` keyword. Calling it automatically produces an iterator object. State (local variables and execution position) is managed by the runtime.

```python
# Comparison of the same functionality implemented as an iterator and a generator
# Iterator version: 13 lines
class RangeIterator:
    def __init__(self, n):
        self.n = n
        self.current = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self.current >= self.n:
            raise StopIteration
        value = self.current
        self.current += 1
        return value

# Generator version: 3 lines
def range_generator(n):
    for i in range(n):
        yield i
```

### Q2: When should generators be used?

**A:** Generators are particularly useful in the following cases.

| Case | Reason |
|------|--------|
| Sequential processing of large data | No need to hold all elements in memory |
| Representing infinite sequences | Impossible with lists |
| Complex traversal logic | Can be written more simply than state machines |
| Pipeline processing | Stream without accumulating intermediate results |
| Coroutine-like processing | Bidirectional communication via `send()` / `yield` |

Conversely, regular lists are appropriate when:

- Multiple accesses to elements are needed
- Random access (index-based) is needed
- The number of elements is small and memory is not a concern
- Sequence operations like `len()` or `reversed()` are needed

### Q3: Why doesn't Rust have `yield`?

**A:** As of 2024, Rust does not have native generator syntax (`yield`) in stable releases. There are several reasons for this:

1. **Ownership and borrowing**: While a generator is suspended at `yield`, references to local variables must remain valid. This interacts complexly with Rust's borrow checker (the self-referential struct problem).
2. **Pin and Unpin**: A suspended generator cannot be moved in memory (due to self-references). This requires the `Pin<T>` concept, the same problem encountered in the `async/await` implementation.
3. **Availability of alternatives**: In Rust, you can implement the Iterator trait directly or use crates like `async-stream` as substitutes.

```rust
// On Rust nightly, gen blocks are experimentally available
#![feature(gen_blocks)]

fn fibonacci() -> impl Iterator<Item = u64> {
    gen {
        let (mut a, mut b) = (0, 1);
        loop {
            yield a;
            (a, b) = (b, a + b);
        }
    }
}
```

### Q4: Why does `itertools.tee()` consume memory?

**A:** `tee()` retains values obtained from the original iterator in an internal buffer. When the two duplicated iterators consume at different rates, the one that has advanced further accumulates values in the buffer for the one that lags behind.

```
tee(iter, 2) -> (it1, it2)

If it1 has advanced 1000 elements and it2 is still at 0:
-> 1000 elements worth of buffer held in memory

If both consume at the same pace:
-> Buffer remains minimal
```

When using `tee()` with large data, make sure to consume both iterators "alternately," or consider converting to a list instead.

### Q5: What happens when yield is used inside a for loop?

**A:** A function containing `yield` becomes a generator function. By using `yield` inside a `for` loop, you create a generator that returns one value per loop iteration. This is a very common pattern.

```python
def filtered_lines(filename, keyword):
    """Yield only lines containing the keyword from a file"""
    with open(filename) as f:
        for line in f:        # for loop
            if keyword in line:
                yield line    # Return matching lines per iteration

# Consumer side
for line in filtered_lines("access.log", "ERROR"):
    print(line)
```

### Q6: Can generators implement synchronous sleep in JavaScript?

**A:** Generators themselves do not implement synchronous sleep. However, it is possible to make async processing "appear" synchronous using generators, which was the approach taken by libraries (`co`, `bluebird`, etc.) before `async/await`. Today, `async/await` should be used instead.

---


## FAQ

### Q1: What is the most important point in learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory, but by actually writing code and verifying behavior.

### Q2: What are common mistakes beginners make?

Skipping the basics and jumping to advanced topics. We recommend thoroughly understanding the fundamental concepts explained in this guide before moving to the next step.

### Q3: How is this applied in practice?

Knowledge of this topic is frequently used in everyday development work. It becomes particularly important during code reviews and architecture design.

---

## 12. Summary

### 12.1 Concept Map

```
+-------------------------------------------------------------+
|  Concept Map of Iterators and Generators                     |
+-------------------------------------------------------------+
|                                                             |
|  Data Structures                                             |
|    |                                                        |
|    +-- Iterable                                             |
|         |                                                   |
|         +-- Iterator                                        |
|         |    |                                               |
|         |    +-- External (Pull): Retrieve with next()      |
|         |    |                                               |
|         |    +-- Internal (Push): each/forEach              |
|         |                                                   |
|         +-- Generator                                       |
|              |                                               |
|              +-- Suspend/resume with yield                   |
|              |                                               |
|              +-- Bidirectional communication with send()     |
|              |                                               |
|              +-- Delegate with yield from / yield*           |
|                                                             |
|  Evaluation Strategies                                       |
|    |                                                        |
|    +-- Eager: Compute everything immediately                |
|    |                                                        |
|    +-- Lazy: Compute only when needed                       |
|                                                             |
|  Adapters                                                    |
|    |                                                        |
|    +-- Transform: map, filter, take, skip, zip, chain, ...  |
|    |                                                        |
|    +-- Terminal: collect, sum, count, any, all, find, fold   |
|                                                             |
|  Async                                                       |
|    |                                                        |
|    +-- Async Iterator: await + next()                       |
|    |                                                        |
|    +-- Stream (Rust): poll_next() -> Poll<Option<T>>        |
|    |                                                        |
|    +-- for await...of / async for                           |
+-------------------------------------------------------------+
```

### 12.2 Key Points Summary

| Concept | Core Idea | Typical Use Case | Representative Languages |
|---------|-----------|------------------|-------------------------|
| Iterator | Unified interface for retrieving elements one at a time | Collection traversal | All languages |
| Adapter | Chain of lazy transformations | Data processing pipelines | Rust, Python, JS |
| Generator | Function that suspends/resumes with yield | Infinite sequences, state machines | Python, JS |
| Lazy Evaluation | Compute only when needed | Large data processing, infinite sequences | Haskell, Rust iter |
| Async Iterator | await + yield | Network streams | Python, JS, Rust |
| send/Bidirectional | Exchange values between caller and generator | Coroutines, state control | Python, JS |

### 12.3 Language Selection Guide

| Purpose | Recommended Language/Approach | Reason |
|---------|-------------------------------|--------|
| High-performance iterator processing | Rust | Zero-cost abstraction |
| Easy generators | Python | Simplest syntax |
| Functional iterator operations | Haskell / Scala | Lazy evaluation by default |
| Frontend stream processing | JavaScript | async generator + for await |
| Large-scale data pipelines | Python + itertools | Rich combination functions |
| Type-safe stream processing | Rust + tokio_stream | Compile-time guarantees |

---

## Suggested Next Reading


---

## 13. References

### Books

1. **Gamma, E., Helm, R., Johnson, R., & Vlissides, J.** "Design Patterns: Elements of Reusable Object-Oriented Software." Addison-Wesley, 1994. -- The original source for the iterator pattern in GoF design patterns.
2. **Klabnik, S. & Nichols, C.** "The Rust Programming Language." No Starch Press, 2019. -- Chapter 13 "Functional Language Features: Iterators and Closures" provides detailed coverage of iterators.
3. **Beazley, D. & Jones, B.K.** "Python Cookbook, 3rd Edition." O'Reilly Media, 2013. -- Chapter 4 "Iterators and Generators" contains numerous practical recipes.

### Official Documentation and Specifications

4. **"Iterator trait - Rust Standard Library Documentation."** doc.rust-lang.org. -- Official reference for the Rust Iterator trait. Over 75 methods documented.
5. **"PEP 255 -- Simple Generators."** python.org, 2001. -- The proposal that introduced generators to Python. Contains design rationale and background.
6. **"PEP 380 -- Syntax for Delegating to a Subgenerator."** python.org, 2009. -- The proposal that introduced the `yield from` syntax.
7. **"PEP 525 -- Asynchronous Generators."** python.org, 2016. -- The proposal that introduced async generators (`async def` + `yield`).
8. **"MDN Web Docs: Iterators and generators."** developer.mozilla.org. -- Comprehensive guide to JavaScript iterators and generators.

### Papers and Technical Articles

9. **Hutton, G.** "A Tutorial on the Universality and Expressiveness of Fold." Journal of Functional Programming, 1999. -- A classic paper on the mathematical foundations and expressiveness of fold.
10. **Kiselyov, O., Shan, C., Friedman, D., & Sabry, A.** "Backtracking, Interleaving, and Terminating Monad Transformers." ICFP, 2005. -- Theoretical foundations of lazy evaluation and stream processing.

---

> **Key Message of This Chapter:** Iterators and generators are powerful tools for abstracting "the flow of data." Through lazy evaluation, they compute only what is needed, conserve memory, and can handle infinite data structures. Understanding these concepts enables you to build declarative, composable data processing pipelines.

---

## References

- [MDN Web Docs](https://developer.mozilla.org/) - Web technology reference
- [Wikipedia](https://en.wikipedia.org/) - Overview of technical concepts
