# Garbage Collection (GC) Complete Guide

> **GC (Garbage Collection) is a mechanism that "automatically reclaims memory that is no longer needed."**
> While it dramatically reduces the burden on programmers, it comes with the trade-off of
> uncontrollable pause times (STW: Stop-The-World). This guide systematically covers everything
> from the theoretical foundations of GC to implementation details in major languages,
> tuning strategies, and anti-pattern avoidance.

---

## What You Will Learn in This Chapter

- [ ] Understand the problems GC solves and the new challenges it introduces
- [ ] Grasp the operating principles of major algorithms such as Mark & Sweep, Copying GC, and Reference Counting
- [ ] Understand the rationale and applications of the Generational Hypothesis
- [ ] Compare GC implementations across Java (G1/ZGC), Go, Python, and JavaScript (V8)
- [ ] Master basic GC tuning strategies
- [ ] Diagnose and resolve performance issues caused by GC
- [ ] Explain the fundamental differences from ownership systems (Rust) and manual management (C/C++)


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related fundamental concepts
- Understanding of the contents of [Stack and Heap](./00-stack-and-heap.md)

---

## Table of Contents

1. [Why GC Is Needed](#1-why-gc-is-needed)
2. [Basic GC Concepts and Root Sets](#2-basic-gc-concepts-and-root-sets)
3. [Mark & Sweep](#3-mark--sweep)
4. [Reference Counting](#4-reference-counting)
5. [Copying GC](#5-copying-gc)
6. [Generational GC](#6-generational-gc)
7. [Concurrent and Incremental GC](#7-concurrent-and-incremental-gc)
8. [GC Implementation Comparison Across Major Languages](#8-gc-implementation-comparison-across-major-languages)
9. [GC Tuning Strategies](#9-gc-tuning-strategies)
10. [Anti-Patterns and Avoidance Strategies](#10-anti-patterns-and-avoidance-strategies)
11. [Exercises (3 Levels)](#11-exercises-3-levels)
12. [FAQ (Frequently Asked Questions)](#12-faq-frequently-asked-questions)
13. [Summary and Next Steps](#13-summary-and-next-steps)
14. [References](#14-references)

---

## 1. Why GC Is Needed

### 1.1 Three Major Problems with Manual Memory Management

When a program dynamically allocates memory, determining when to free that memory is a fundamentally
difficult problem. In manual management languages like C/C++, the following three bugs have
repeatedly occurred.

```
Three Major Problems with Manual Memory Management:

  1. Memory Leak
     ─────────────────────────────────────────────────────
     malloc() to allocate → use → forget to free() → memory exhaustion

     Fatal for long-running servers. Memory usage increases monotonically,
     and the process is ultimately killed by the OOM Killer.

     Example: 100B leak per request × 1 million requests/day
              = approximately 95MB/day of memory leaks

  2. Dangling Pointer
     ─────────────────────────────────────────────────────
     free(ptr) → ... → access *ptr → undefined behavior (UB)

     The contents of freed memory may accidentally appear correct,
     delaying the discovery of the problem. A leading cause of security
     vulnerabilities. Frequently appears in CVEs as Use-After-Free (UAF).

  3. Double Free
     ─────────────────────────────────────────────────────
     free(ptr) → ... → free(ptr) → heap corruption / crash

     The internal data structures of the memory allocator are corrupted,
     causing subsequent malloc/free calls to behave unpredictably.
```

### 1.2 Concrete Examples in C

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// --- Problem 1: Memory Leak ---
char* create_greeting(const char* name) {
    // The caller is responsible for freeing (easily forgotten)
    char* buf = (char*)malloc(256);
    if (!buf) return NULL;
    snprintf(buf, 256, "Hello, %s!", name);
    return buf;  // Ownership transfers to the caller
}

void process_request(const char* name) {
    char* greeting = create_greeting(name);
    printf("%s\n", greeting);
    // free(greeting);  ← Forgetting this causes a leak!
}

// --- Problem 2: Dangling Pointer ---
int* get_local_ptr() {
    int local_val = 42;
    return &local_val;  // Returning a pointer to a local variable (UB)
}

// --- Problem 3: Double Free ---
void double_free_example() {
    int* p = (int*)malloc(sizeof(int));
    *p = 100;
    free(p);
    // ... subsequent code ...
    free(p);   // Double free! Heap corruption
}

// --- GC solves all of these ---
// In Java/Go/Python, the above three problems structurally cannot occur
```

### 1.3 GC Trade-offs

GC automatically solves the above problems, but at the cost of the following overhead.

```
GC Trade-offs:

  ┌─────────────────────────────────────────────────────────────┐
  │                     GC Advantages                            │
  │  + Memory leaks, dangling pointers, and double frees are     │
  │    structurally eliminated                                   │
  │  + Improved development speed (no memory management code)    │
  │  + Memory safety guarantee                                   │
  ├─────────────────────────────────────────────────────────────┤
  │                     GC Disadvantages                          │
  │  - Unpredictable pauses due to STW (Stop-The-World)          │
  │  - Increased memory usage (GC metadata + delayed reclamation)│
  │  - Increased CPU usage (GC thread execution)                 │
  │  - Reduced cache efficiency (object relocation)              │
  │  - Difficulty guaranteeing real-time performance              │
  └─────────────────────────────────────────────────────────────┘

  Performance Characteristics Comparison:

  Manual     ████████████████████████████  Best performance (but low safety)
  With GC    ████████████████████░░░░░░░░  Typically 5-20% overhead
  Ownership  ██████████████████████████░░  Near-manual performance + safety
```

### 1.4 Historical Background

GC was invented in 1959 by John McCarthy for Lisp. This is one of the oldest and most important
inventions in the history of programming languages.

```
GC Historical Timeline:

  1959  McCarthy invents Mark & Sweep GC for Lisp
  1960  Collins proposes reference counting
  1963  Minsky invents Copying GC
  1969  Fenichel & Yochelson develop Semi-space Copying GC
  1984  Lieberman & Hewitt propose generational GC
  1992  Boehm develops conservative GC (for C/C++)
  2004  Bacon et al. publish research on Real-time GC
  2006  Java 6 makes Parallel GC the default
  2014  G1 GC matures in Java 8
  2017  Go 1.8 achieves STW < 100μs
  2018  Java 11 introduces ZGC (experimental)
  2021  Java 17 LTS makes ZGC production-ready
  2023  Java 21 LTS introduces generational ZGC
```

---

## 2. Basic GC Concepts and Root Sets

### 2.1 Reachability

The core concept of GC is "reachability." Objects that can be reached by following chains of
references from the root set are considered "alive," while objects that cannot be reached
are considered "garbage."

```
Reachability Determination:

  Root Set (GC Roots)
  ┌──────────────────────────────────────┐
  │  Local variables on the stack         │
  │  Global / static variables            │
  │  JNI references (in Java)             │
  │  Stack frames of active threads       │
  │  Class loader references (Java)       │
  │  Pointers stored in registers         │
  └──────────────────────────────────────┘
         │          │           │
         ▼          ▼           ▼
      [Obj A] ──→ [Obj B] ──→ [Obj C]    ← Reachable (alive)
                    │
                    ▼
                  [Obj D]                  ← Reachable (alive)

      [Obj E] ──→ [Obj F]                 ← Unreachable (garbage → subject to collection)
         ↑           │
         └───────────┘  Circular reference, but
                        unreachable from roots so subject to collection
```

### 2.2 Root Set Details

The components of the root set vary by language and runtime. Below is a classification of major roots.

```
Root Set Classification:

  Type             │ Description                       │ Example
  ─────────────────┼──────────────────────────────────┼────────────────────
  Stack roots      │ Method/function local variables    │ int x = new ...
  Global roots     │ Static fields, global variables    │ static Map cache
  Register roots   │ Pointers in CPU registers          │ Used by JIT optimization
  JNI roots        │ References from native code        │ JNI GlobalRef
  Finalizer        │ Objects awaiting finalization       │ Weak/Phantom Ref
  Thread roots     │ Thread objects themselves           │ Thread.currentThread
```

### 2.3 Object Graphs and Pointer Analysis

GC traverses the object graph at runtime. The structure of this graph significantly
affects GC efficiency.

```java
// Java: Object graph construction example
public class ObjectGraphDemo {

    // Object graph traceable from roots
    public static void main(String[] args) {
        // root1: Stack variable (root)
        List<String> list = new ArrayList<>();
        list.add("Hello");   // list → "Hello"
        list.add("World");   // list → "World"

        // root2: Local variable (root)
        Map<String, List<String>> map = new HashMap<>();
        map.put("greetings", list);  // map → list → {"Hello", "World"}

        // Object that becomes unreachable
        {
            byte[] temp = new byte[1024 * 1024]; // Allocate 1MB
            // ... use temp ...
        }
        // ← At the point of leaving this block, temp becomes unreachable
        //   It will be collected at the next GC

        // map and list are still reachable from roots
        System.out.println(map.get("greetings"));
    }
}
```

---

## 3. Mark & Sweep

### 3.1 Algorithm Overview

Mark & Sweep is the first GC algorithm invented by John McCarthy in 1959 for Lisp.
It is conceptually the simplest and forms the foundation for all other GC algorithms.

```
Mark & Sweep Two Phases:

  ┌──────────────────────────────────────────────────────────┐
  │  Phase 1: Mark — Searching for reachable objects          │
  │                                                          │
  │  1. Reset the mark bit of all objects to 0               │
  │  2. Follow references from the root set using DFS        │
  │  3. Set the mark bit of reachable objects to 1           │
  │                                                          │
  │  Phase 2: Sweep — Collecting garbage                     │
  │                                                          │
  │  1. Linearly scan the entire heap                        │
  │  2. Return objects with mark bit 0 to the free list      │
  │  3. Reset the mark bit of objects with mark bit 1 to 0   │
  └──────────────────────────────────────────────────────────┘
```

### 3.2 Detailed Visualization of Operation

```
Step-by-step Mark & Sweep Operation:

  === Initial State ===

  Roots: [R1] ─→ [A] ─→ [B]
         [R2] ─→ [C] ─→ [D]
                        ↗
  Isolated: [E] ─→ [F] ─┘     ← E,F are unreachable (appear to be, but via D...)

  Actual graph:

  [R1]──→[A]──→[B]
              ↗
  [R2]──→[C]──→[D]

  [E]──→[F]           ← Unreachable from roots

  === Phase 1: Mark ===

  Step 1: Start exploration from R1
    R1 → A (mark=1) → B (mark=1)

  Step 2: Start exploration from R2
    R2 → C (mark=1) → D (mark=1)

  Result:
    A(1) B(1) C(1) D(1) E(0) F(0)

  === Phase 2: Sweep ===

  Scan heap:
    A(1) → reset mark=0, retain
    B(1) → reset mark=0, retain
    C(1) → reset mark=0, retain
    D(1) → reset mark=0, retain
    E(0) → add to free list (collected)  ★
    F(0) → add to free list (collected)  ★

  Memory collected: size of E + F
```

### 3.3 Pseudocode Implementation

```python
# Mark & Sweep GC pseudocode implementation

class Object:
    def __init__(self, name, size):
        self.name = name
        self.size = size
        self.marked = False
        self.references = []  # References to other objects

class MarkSweepGC:
    def __init__(self, heap_size):
        self.heap = []          # List of all objects
        self.roots = []         # Root set
        self.free_list = []     # Free list
        self.heap_size = heap_size
        self.used = 0

    def allocate(self, name, size):
        """Allocate memory. Run GC if no space available"""
        if self.used + size > self.heap_size:
            self.collect()
            if self.used + size > self.heap_size:
                raise MemoryError("Out of memory after GC")

        obj = Object(name, size)
        self.heap.append(obj)
        self.used += size
        return obj

    def collect(self):
        """Execute GC: Mark → Sweep"""
        print("=== GC Start ===")

        # Phase 1: Mark
        for obj in self.heap:
            obj.marked = False    # Reset all bits

        for root in self.roots:
            self._mark(root)      # Mark objects reachable from roots

        # Phase 2: Sweep
        alive = []
        freed = 0
        for obj in self.heap:
            if obj.marked:
                obj.marked = False
                alive.append(obj)
            else:
                freed += obj.size
                print(f"  Freed: {obj.name} ({obj.size} bytes)")

        self.heap = alive
        self.used -= freed
        print(f"=== GC End: freed {freed} bytes ===")

    def _mark(self, obj):
        """Recursively mark reachable objects (DFS)"""
        if obj is None or obj.marked:
            return
        obj.marked = True
        for ref in obj.references:
            self._mark(ref)

# --- Usage example ---
gc = MarkSweepGC(heap_size=1024)

# Allocate objects
a = gc.allocate("A", 100)
b = gc.allocate("B", 200)
c = gc.allocate("C", 150)
d = gc.allocate("D", 100)

# Build reference graph
a.references.append(b)    # A → B
b.references.append(c)    # B → C

# Set roots (only A is reachable from roots)
gc.roots = [a]

# Execute GC → D is unreachable so it gets collected
gc.collect()
# Output:
#   === GC Start ===
#     Freed: D (100 bytes)
#   === GC End: freed 100 bytes ===
```

### 3.4 Advantages and Disadvantages of Mark & Sweep

```
┌────────────────────────────────────────────────────────────────┐
│  Mark & Sweep Evaluation                                        │
├───────────────────────┬────────────────────────────────────────┤
│  Advantages            │  Details                               │
├───────────────────────┼────────────────────────────────────────┤
│  Circular reference    │  Unlike reference counting, can         │
│  handling              │  correctly detect and collect cycles    │
├───────────────────────┼────────────────────────────────────────┤
│  Simplicity of         │  Composed of two phases                │
│  implementation        │  Fundamentally easy to understand       │
├───────────────────────┼────────────────────────────────────────┤
│  Zero allocation cost  │  No additional management cost          │
│                       │  when allocating objects                 │
├───────────────────────┼────────────────────────────────────────┤
│  Disadvantages         │  Details                               │
├───────────────────────┼────────────────────────────────────────┤
│  STW (full pause)     │  The entire application stops during    │
│                       │  mark and sweep phases                   │
├───────────────────────┼────────────────────────────────────────┤
│  Memory fragmentation │  Holes form in memory after collection, │
│                       │  making large contiguous allocation hard │
├───────────────────────┼────────────────────────────────────────┤
│  Full heap scan       │  Sweep phase scans the entire heap      │
│                       │  Takes time for large heaps              │
└───────────────────────┴────────────────────────────────────────┘
```

### 3.5 Fragmentation Problem Illustrated

```
Memory Fragmentation Problem:

  Before GC:
  ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
  │  A   │  B   │  C   │  D   │  E   │  F   │  G   │  H   │
  │ 100B │ 200B │ 150B │ 100B │ 300B │ 50B  │ 200B │ 100B │
  └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘

  After GC (B, D, F collected):
  ┌──────┬ ─ ─ ┬──────┬ ─ ─ ┬──────┬ ─ ─ ┬──────┬──────┐
  │  A   │(free)│  C   │(free)│  E   │(free)│  G   │  H   │
  │ 100B │200B  │ 150B │100B  │ 300B │ 50B  │ 200B │ 100B │
  └──────┴ ─ ─ ┴──────┴ ─ ─ ┴──────┴ ─ ─ ┴──────┴──────┘

  Total free: 350B, but the largest contiguous free region is 200B
  → Cannot allocate a 250B object! (external fragmentation)

  Solution: Compaction
  ┌──────┬──────┬──────┬──────┬──────┬─────────────────────┐
  │  A   │  C   │  E   │  G   │  H   │      free (350B)     │
  │ 100B │ 150B │ 300B │ 200B │ 100B │                      │
  └──────┴──────┴──────┴──────┴──────┴─────────────────────┘

  → A contiguous 350B free region can now be obtained
  → However, compaction requires rewriting pointers (high cost)
```

---

## 4. Reference Counting

### 4.1 Algorithm Overview

Reference counting is a method where each object records "how many pointers reference it."
When the reference count reaches 0, the object is immediately collected.

```
Basic Reference Counting Operation:

  Operation                      Count Change
  ─────────────────────────────────────────────
  a = new Obj()                 Obj.rc = 1
  b = a                         Obj.rc = 2
  a = null                      Obj.rc = 1
  b = null                      Obj.rc = 0 → Immediately collected!

  Timeline:

  Time  Operation   a      b      Obj.rc   Status
  ────────────────────────────────────────────────
  T1    a=new()     →Obj   -      1        Alive
  T2    b=a         →Obj   →Obj   2        Alive
  T3    a=null      null   →Obj   1        Alive
  T4    b=null      null   null   0        ★Collected★
```

### 4.2 Python's Reference Counting Implementation

Python is the only major language that uses reference counting as its main GC mechanism.

```python
import sys
import gc

# === Observing Reference Counting ===

class MyObject:
    def __init__(self, name):
        self.name = name
    def __del__(self):
        print(f"  Destructor called: collecting {self.name}")

print("--- Reference Counting Basics ---")
obj = MyObject("Alpha")
print(f"Reference count: {sys.getrefcount(obj) - 1}")
# Subtract 1 for the temporary reference from getrefcount itself

ref2 = obj
print(f"After adding reference: {sys.getrefcount(obj) - 1}")

del ref2
print(f"After deleting reference: {sys.getrefcount(obj) - 1}")

del obj  # rc=0 → destructor is called immediately
print("Right after del obj")

# Output:
#   --- Reference Counting Basics ---
#   Reference count: 1
#   After adding reference: 2
#   After deleting reference: 1
#     Destructor called: collecting Alpha
#   Right after del obj

print("\n--- Circular Reference Problem ---")
# Case where reference counting alone cannot collect
a = MyObject("CycleA")
b = MyObject("CycleB")
a.partner = b   # a → b
b.partner = a   # b → a (circular reference)

# External references are deleted but rc never reaches 0
del a  # CycleA.rc: 2→1 (still referenced by b.partner)
del b  # CycleB.rc: 2→1 (still referenced by a.partner)

# → Destructors are NOT called!
# → Python's generational GC (cycle collector) will collect them

print("Calling gc.collect() manually:")
collected = gc.collect()
print(f"Number of collected objects: {collected}")

# Output:
#   --- Circular Reference Problem ---
#   Calling gc.collect() manually:
#     Destructor called: collecting CycleA
#     Destructor called: collecting CycleB
#   Number of collected objects: 2
```

### 4.3 Circular Reference Problem Illustrated

```
Circular Reference Problem:

  === State with External References ===

  Roots
  ┌──────┐
  │ a ───┼──→ [ObjA  rc=2] ──→ [ObjB  rc=2]
  │ b ───┼──→       ↑          │
  └──────┘          └──────────┘  Circular reference

  rc=2 breakdown:
    ObjA: root a (1) + ObjB.partner (1) = 2
    ObjB: root b (1) + ObjA.partner (1) = 2

  === After del a, del b ===

  Roots
  ┌──────┐
  │      │     [ObjA  rc=1] ──→ [ObjB  rc=1]
  │      │           ↑          │
  └──────┘           └──────────┘  Circular reference

  rc=1 breakdown:
    ObjA: ObjB.partner (1) = 1  ← Never reaches 0!
    ObjB: ObjA.partner (1) = 1  ← Never reaches 0!

  → Cannot be collected by reference counting alone
  → A separate "cycle collector" is needed
```

### 4.4 Advantages and Disadvantages of Reference Counting

```
Reference Counting Evaluation:

  ┌────────────────────────────────────────────────────────────┐
  │  Advantages                                                │
  ├────────────────────────────────────────────────────────────┤
  │  1. Immediate collection: Memory is freed the moment rc=0  │
  │     → Destructor execution timing is predictable           │
  │     → Peak memory usage is kept low                        │
  │                                                            │
  │  2. No STW: No batch pauses caused by GC                   │
  │     → Latency is predictable                               │
  │                                                            │
  │  3. Locality: Cost is incurred only on reference changes    │
  │     → Cost is distributed compared to GC                   │
  ├────────────────────────────────────────────────────────────┤
  │  Disadvantages                                             │
  ├────────────────────────────────────────────────────────────┤
  │  1. Circular references: Cannot be collected by RC alone    │
  │     → A separate cycle collector is needed (Python's impl.) │
  │                                                            │
  │  2. Count update overhead                                   │
  │     → Atomic inc/dec required for every reference operation │
  │     → Especially costly in multithreaded environments       │
  │                                                            │
  │  3. Cascading deallocation: Freeing large data structures  │
  │     cascades, potentially causing a temporarily long pause  │
  │     Example: Freeing a tree with 1M nodes → 1M recursive   │
  │     free calls                                              │
  │                                                            │
  │  4. Memory overhead: Each object needs an rc field          │
  │     (typically 4-8 bytes)                                   │
  └────────────────────────────────────────────────────────────┘
```

### 4.5 Swift's ARC (Automatic Reference Counting)

Swift uses ARC, which automates reference counting at the compiler level.
Programmers don't need to manually write retain/release, but must explicitly break
circular references using the `weak` / `unowned` keywords.

```swift
// Swift: ARC operation and circular reference avoidance

class Person {
    let name: String
    var apartment: Apartment?  // Strong reference

    init(name: String) {
        self.name = name
        print("\(name) allocated")
    }
    deinit {
        print("\(name) deallocated")
    }
}

class Apartment {
    let unit: String
    weak var tenant: Person?   // Weak reference (circular reference avoidance)

    init(unit: String) {
        self.unit = unit
        print("Apt \(unit) allocated")
    }
    deinit {
        print("Apt \(unit) deallocated")
    }
}

// Usage example
var john: Person? = Person(name: "John")       // John.rc = 1
var apt: Apartment? = Apartment(unit: "101")   // Apt101.rc = 1

john!.apartment = apt    // Apt101.rc = 2 (strong reference)
apt!.tenant = john       // John.rc = 1 (weak, so rc does not increase)

john = nil   // John.rc = 0 → deallocated, Apt101.rc = 1
apt = nil    // Apt101.rc = 0 → deallocated

// Without weak:
// john = nil → John.rc=1 (apt.tenant holds a strong reference) → NOT deallocated!
// apt = nil  → Apt101.rc=1 (john.apartment holds a strong reference) → NOT deallocated!
// → Memory leak
```

### 4.6 Reference Counting vs Mark & Sweep Comparison

```
┌───────────────────┬────────────────────┬────────────────────┐
│  Property          │  Reference Counting │  Mark & Sweep       │
├───────────────────┼────────────────────┼────────────────────┤
│  Collection timing │  Immediate (rc=0)  │  At GC execution    │
│  STW              │  None              │  Yes (full pause)    │
│  Circular refs    │  Cannot handle     │  Can handle          │
│  CPU overhead     │  Distributed       │  Concentrated (GC)   │
│  Memory efficiency│  High (immediate)  │  Slightly low (delay)│
│  Impl. complexity │  Low               │  Moderate            │
│  Thread safety    │  Atomics needed    │  Batch by GC thread  │
│  Cascading free   │  Occurs            │  Does not occur      │
│  Languages using  │  Python, Swift,    │  Java, Go, JS,       │
│                   │  Objective-C, Perl │  Ruby, C#            │
└───────────────────┴────────────────────┴────────────────────┘
```

---

## 5. Copying GC

### 5.1 Semi-space Principle

Copying GC divides the heap into two equally-sized spaces (semi-spaces) and copies
surviving objects from one to the other. It was proposed by Marvin Minsky in 1963
and refined into its current form by Fenichel and Yochelson in 1969.

```
Semi-space Copying GC Operation:

  === Before GC ===

  From-space (active)               To-space (empty)
  ┌─────────────────────────┐       ┌─────────────────────────┐
  │ [A 100B] [dead] [B 80B] │       │                         │
  │ [dead] [C 60B] [dead]   │       │        (unused)          │
  │ [D 40B] [dead] [dead]   │       │                         │
  └─────────────────────────┘       └─────────────────────────┘
   Alive: A,B,C,D  Dead: 5

  === GC Execution (Copying) ===

  Copy only objects reachable from roots to To-space:

  From-space                          To-space
  ┌─────────────────────────┐       ┌─────────────────────────┐
  │                         │       │ [A' 100B][B' 80B]       │
  │    (discard entirely)    │  ←→   │ [C' 60B] [D' 40B]       │
  │                         │       │       (free space)       │
  └─────────────────────────┘       └─────────────────────────┘
   Swap From and To

  === After GC ===

  From-space (old To, now active)    To-space (old From, empty)
  ┌─────────────────────────┐       ┌─────────────────────────┐
  │ [A' 100B][B' 80B]       │       │                         │
  │ [C' 60B] [D' 40B]       │       │        (unused)          │
  │       (free space)       │       │                         │
  └─────────────────────────┘       └─────────────────────────┘

  Characteristics:
  - Compaction happens automatically (no fragmentation)
  - Cost proportional to number of surviving objects (dead objects ignored)
  - Trade-off: only half the memory is usable
```

### 5.2 Forwarding Pointers

In Copying GC, a "forwarding pointer" is placed at the original location of a copied object
to redirect references from other objects to the new address.

```
Forwarding Pointer Mechanism:

  Step 1: Discover A from roots, copy to To-space
  From: [A] ──(forwarding)──→ To: [A']
        ↑
  Step 2: Discover B (references A)
  From: [B] → [A's fwd] → To: [A']
        ↓ copy
  To:   [B'] → [A']    ← Reference automatically updated

  Step 3: Discover C (references A)
  From: [C] → [A's fwd] → To: [A'] ← Already copied, not re-copied
        ↓ copy
  To:   [C'] → [A']

  → All references are automatically rewritten to point to the new addresses
```

### 5.3 Cheney's Algorithm

The BFS-based Copying GC algorithm proposed by C.J. Cheney in 1970 does not use a stack,
making its implementation efficient without recursion.

```python
# Cheney's Copying GC Algorithm (pseudocode)

class CheneyGC:
    def __init__(self, space_size):
        self.space_size = space_size
        # From-space and To-space
        self.from_space = bytearray(space_size)
        self.to_space = bytearray(space_size)
        self.alloc_ptr = 0  # Allocation position in From-space
        self.scan = 0       # Scan pointer in To-space
        self.free = 0       # Free pointer in To-space

    def collect(self, roots):
        """Execute GC: Cheney's BFS copy"""
        self.scan = 0
        self.free = 0

        # Step 1: Copy objects directly referenced from roots
        new_roots = []
        for root in roots:
            new_roots.append(self._copy(root))

        # Step 2: BFS — also copy references of already-copied objects
        while self.scan < self.free:
            obj = self._object_at(self.to_space, self.scan)
            for i, ref in enumerate(obj.references):
                obj.references[i] = self._copy(ref)
            self.scan += obj.size

        # Step 3: Swap From and To
        self.from_space, self.to_space = self.to_space, self.from_space
        self.alloc_ptr = self.free

        return new_roots

    def _copy(self, obj):
        """Copy object to To-space (only if not already copied)"""
        if obj.forwarding is not None:
            return obj.forwarding  # Already copied

        # Copy to To-space
        new_obj = self._copy_bytes(obj, self.to_space, self.free)
        self.free += obj.size

        # Install forwarding pointer
        obj.forwarding = new_obj
        return new_obj
```

### 5.4 Performance Characteristics of Copying GC

```
Copying GC Performance Analysis:

  Relationship between survival rate (surviving objects / total objects) and GC cost:

  Cost
  High│                                     ╱
     │                                   ╱
     │                                ╱
     │                             ╱
     │                          ╱
     │                       ╱
     │                    ╱
     │                 ╱           ← Copying GC: proportional to survival rate
     │              ╱
     │           ╱
     │        ╱
     │─────╱──────────────────────── ← Mark & Sweep: roughly constant
     │  ╱                               (due to full heap scan)
  Low│╱
     └──────────────────────────────→ Survival rate
     0%                           100%

  Conclusion:
  - Low survival rate (<50%): Copying GC is advantageous
  - High survival rate (>50%): Mark & Sweep is advantageous
  - Young generation has low survival rate → Copying GC is optimal
  - Old generation has high survival rate → Mark & Sweep is optimal
```

---

## 6. Generational GC

### 6.1 Generational Hypothesis

Generational GC is an optimization technique based on the empirical rule that
"most objects die young" (Infant Mortality). This hypothesis was systematized by
David Ungar in 1984 through his research on Smalltalk.

```
Basis for the Generational Hypothesis:

  Object lifetime distribution (typical application):

  Number of objects
  Many│██
     │██
     │██
     │██░░
     │██░░
     │██░░░░
     │██░░░░░░
     │██░░░░░░░░
     │██░░░░░░░░░░░░░░
     │██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  Few │██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
     └──────────────────────────────────────────────────────→ Lifetime
     Short                                                  Long

  █ = Die in Young generation (80-98%)
  ░ = Survive to Old generation (2-20%)

  Concrete examples:
  - Method local variables, temporary objects → die quickly
  - Iterators, lambda closures → die quickly
  - Caches, configuration objects → long-lived
  - Singletons → alive until application exit
```

### 6.2 Java HotSpot Generational Layout

```
Java HotSpot VM Heap Layout:

  ┌─────────────────────────────────────────────────────────────────┐
  │                         Java Heap                                │
  │                                                                 │
  │  Young Generation                    Old Generation              │
  │  ┌────────┬──────┬──────┐            ┌──────────────────────┐   │
  │  │  Eden  │  S0  │  S1  │            │                      │   │
  │  │        │(from)│ (to) │  Promotion │    Tenured Space     │   │
  │  │  New   │      │      │ ────────→  │                      │   │
  │  │ objects│Survi-│Survi-│            │  Long-lived objects   │   │
  │  │        │vor 0 │vor 1 │            │                      │   │
  │  └────────┴──────┴──────┘            └──────────────────────┘   │
  │  ←── Minor GC (frequent) ──→         ←── Major GC (infrequent)→ │
  │                                                                 │
  │  Metaspace (Java 8+: native memory)                             │
  │  ┌──────────────────────────────────────────────────────────┐   │
  │  │ Class metadata, method info, constant pool                │   │
  │  └──────────────────────────────────────────────────────────┘   │
  └─────────────────────────────────────────────────────────────────┘

  Default ratios:
  - Young : Old = 1 : 2 (-XX:NewRatio=2)
  - Eden : S0 : S1 = 8 : 1 : 1 (-XX:SurvivorRatio=8)
```

### 6.3 Detailed Minor GC Operation

```
Minor GC Step by Step:

  === Step 1: Eden is full ===
  Eden            S0(from)     S1(to)       Old
  ┌──────────┐   ┌────┐      ┌────┐      ┌──────┐
  │A B C D E │   │ F  │      │    │      │  Z   │
  │(all new)  │   │age1│      │    │      │      │
  └──────────┘   └────┘      └────┘      └──────┘
  * B, D are already unreachable

  === Step 2: Minor GC executes ===
  - Copy surviving objects from Eden (A,C,E) to S1(to)
  - Copy surviving objects from S0 (F, age=1) to S1(to) (age+1)
  - B, D are unreachable so they are not copied (collected)

  Eden            S0(from)     S1(to)       Old
  ┌──────────┐   ┌────┐      ┌────┐      ┌──────┐
  │  (empty)  │   │(empty)│   │A   │      │  Z   │
  │          │   │    │      │C   │      │      │
  │          │   │    │      │E   │age0  │      │
  │          │   │    │      │F   │age2  │      │
  └──────────┘   └────┘      └────┘      └──────┘

  === Step 3: Swap S0 and S1 roles ===
  Eden            S0(=old S1)  S1(=old S0)  Old
  ┌──────────┐   ┌────┐      ┌────┐      ┌──────┐
  │  (empty)  │   │A   │      │    │      │  Z   │
  │          │   │C   │      │(empty)│   │      │
  │          │   │E   │age0  │    │      │      │
  │          │   │F   │age2  │    │      │      │
  └──────────┘   └────┘      └────┘      └──────┘

  === Step 4: In the next Minor GC, promote F (age>=threshold) to Old ===
  - Promoted when MaxTenuringThreshold (default 15) is reached
  - Also promoted directly to Old if Survivor overflows
```

### 6.4 Write Barrier

In generational GC, references from Old to Young need to be tracked. This is achieved
through write barriers.

```
Necessity of Write Barriers:

  Problem: Old → Young reference

  Old                    Young
  ┌──────────┐          ┌──────────┐
  │  OldObj  │ ──ref──→ │ YoungObj │
  │          │          │          │
  └──────────┘          └──────────┘

  Minor GC only scans the Young generation.
  However, if the Old → Young reference is missed,
  YoungObj would be erroneously collected!

  Solution: Write Barrier + Remembered Set

  1. A barrier triggers on writes to Old objects
  2. Old → Young references are recorded in the Remembered Set (card table)
  3. During Minor GC, the Remembered Set is also treated as roots

  Card Table:
  ┌───┬───┬───┬───┬───┬───┬───┬───┐
  │ 0 │ 0 │ 1 │ 0 │ 0 │ 1 │ 0 │ 0 │  ← Each card corresponds to a 512B region
  └───┴───┴───┴───┴───┴───┴───┴───┘
            ↑               ↑
        This card's region  This card's region
        has Young refs      has Young refs

  → Only dirty card regions need to be scanned
```

### 6.5 V8 (JavaScript) Generational GC

```
V8 Engine GC Architecture (Orinoco):

  ┌─────────────────────────────────────────────────────────────┐
  │  Young Generation (Scavenger)                               │
  │                                                             │
  │  Semi-space method (Copying GC)                             │
  │  ┌──────────────┐  ┌──────────────┐                        │
  │  │  From-space   │  │   To-space    │                        │
  │  │              │  │              │                        │
  │  │  New objects  │  │  (Surviving   │                        │
  │  │  allocated   │  │   objects     │                        │
  │  │  here        │  │   after GC)   │                        │
  │  └──────────────┘  └──────────────┘                        │
  │  - Size: 1-8MB (default)                                    │
  │  - Promoted to Old after surviving 2 scavenges              │
  │  - Pause time: 1-2ms                                        │
  ├─────────────────────────────────────────────────────────────┤
  │  Old Generation (Mark-Sweep-Compact)                        │
  │                                                             │
  │  ┌──────────────────────────────────────────────────┐       │
  │  │                                                  │       │
  │  │  Incremental marking:                             │       │
  │  │    Splits GC into small steps to minimize         │       │
  │  │    main thread pauses                             │       │
  │  │                                                  │       │
  │  │  Concurrent marking:                              │       │
  │  │    Marking runs concurrently on worker threads    │       │
  │  │                                                  │       │
  │  │  Concurrent sweeping:                             │       │
  │  │    Sweeping in background while main thread runs  │       │
  │  │                                                  │       │
  │  │  Compaction (only when needed):                   │       │
  │  │    Executes when fragmentation exceeds threshold  │       │
  │  └──────────────────────────────────────────────────┘       │
  │  - Size: hundreds of MB to several GB                       │
  │  - Idle-Time GC: Runs GC during browser idle time           │
  └─────────────────────────────────────────────────────────────┘
```

### 6.6 V8 GC Tuning in Node.js

```javascript
// Node.js: Observing and tuning V8 GC behavior

// --- Monitoring GC events ---
// Launch option: node --expose-gc --trace-gc app.js

// Get heap statistics
const v8 = require('v8');

function printHeapStats() {
    const stats = v8.getHeapStatistics();
    console.log('=== V8 Heap Statistics ===');
    console.log(`  Total heap size:      ${(stats.total_heap_size / 1024 / 1024).toFixed(2)} MB`);
    console.log(`  Used heap size:       ${(stats.used_heap_size / 1024 / 1024).toFixed(2)} MB`);
    console.log(`  Heap size limit:      ${(stats.heap_size_limit / 1024 / 1024).toFixed(2)} MB`);
    console.log(`  External memory:      ${(stats.external_memory / 1024 / 1024).toFixed(2)} MB`);
    console.log(`  Malloced memory:      ${(stats.malloced_memory / 1024 / 1024).toFixed(2)} MB`);
    console.log(`  Number of contexts:   ${stats.number_of_native_contexts}`);
}

// --- Memory leak detection patterns ---
// Problematic code: Leak through closures
const leakyCache = [];
function processRequest(data) {
    // Closure keeps capturing data → GC cannot collect it
    const handler = () => {
        return data.length; // Holds reference to data
    };
    leakyCache.push(handler); // Accumulates in array → memory leak
}

// Improvement: Use WeakRef (Node.js 14.6+)
const cache = new Map();
function processRequestFixed(id, data) {
    // WeakRef: GC can collect when needed
    cache.set(id, new WeakRef(data));
}

function getCachedData(id) {
    const ref = cache.get(id);
    if (ref) {
        const data = ref.deref(); // null if collected by GC
        if (data) return data;
        cache.delete(id); // Remove collected entry
    }
    return null;
}

// --- Tuning options ---
// node --max-old-space-size=4096 app.js   # Old generation to 4GB
// node --max-semi-space-size=64 app.js    # Young generation to 64MB
// node --expose-gc app.js                 # Enable global.gc()
// node --trace-gc app.js                  # Log GC events
// node --gc-interval=100 app.js           # Adjust GC interval

printHeapStats();
```

---

## 7. Concurrent and Incremental GC

### 7.1 The STW Problem and Solution Approaches

The biggest challenge of Mark & Sweep and Generational GC is STW (Stop-The-World).
All application threads stop during GC execution, causing unpredictable spikes in
response time.

```
Impact of STW:

  Timeline ──────────────────────────────────────────→

  App     ████████████│          │████████████████████
  threads             │  STW !   │
                     │  GC runs │
                     │ (50ms)   │
                     │          │
  Response           │          │
  time    2ms  3ms   │ 53ms!!   │ 2ms  2ms  3ms
                     ↑
                  If a request arrives here,
                  a 50ms latency spike occurs

  Solution Approaches:
  ┌──────────────────────────────────────────────────┐
  │  1. Incremental GC: Run GC in small increments   │
  │  2. Concurrent GC: Run in the background          │
  │  3. Parallel GC: Run GC on multiple threads       │
  │  4. Combination of the above                      │
  └──────────────────────────────────────────────────┘
```

### 7.2 Tri-color Marking

The core technique that guarantees correctness in concurrent GC is tri-color marking,
proposed by Dijkstra et al. in 1975.

```
Tri-color Marking:

  Color definitions:
  ■ Black: Scanning complete. Self and all children are marked
  ░ Gray:  Self is marked, but children not yet scanned
  □ White: Not yet scanned. White at GC end means subject to collection

  Marking progression:

  Step 0 (initial state): All objects white, direct references from roots turned gray
  ┌──────┐
  │ root │─→ ░A ─→ □B ─→ □C
  └──────┘    │
              └─→ □D ─→ □E

  Step 1: Process A (gray→black), turn A's children gray
  ┌──────┐
  │ root │─→ ■A ─→ ░B ─→ □C
  └──────┘    │
              └─→ ░D ─→ □E

  Step 2: Process B (gray→black), turn B's children gray
  ┌──────┐
  │ root │─→ ■A ─→ ■B ─→ ░C
  └──────┘    │
              └─→ ░D ─→ □E

  Step 3: Process C (gray→black), no children
  ┌──────┐
  │ root │─→ ■A ─→ ■B ─→ ■C
  └──────┘    │
              └─→ ░D ─→ □E

  Step 4: Process D (gray→black), turn D's children gray
  ┌──────┐
  │ root │─→ ■A ─→ ■B ─→ ■C
  └──────┘    │
              └─→ ■D ─→ ░E

  Step 5: Process E (gray→black)
  ┌──────┐
  │ root │─→ ■A ─→ ■B ─→ ■C
  └──────┘    │
              └─→ ■D ─→ ■E

  Complete: Marking ends when there are no more gray objects
  → White objects are unreachable from roots → collect
```

### 7.3 Concurrent GC Invariants and Write Barriers

In concurrent GC, the application (Mutator) and GC (Collector) run simultaneously,
requiring correct tracking of reference changes.

```
Lost Object Problem in Concurrent GC:

  If the Mutator changes references during GC, live objects may be
  erroneously collected (lost object problem).

  === Problematic Scenario ===

  1. GC has finished scanning ■A, currently scanning ░B
     ■A ─→ ░B ─→ □C

  2. Mutator simultaneously performs:
     A.ref = C    (add reference from A to C)
     B.ref = null  (remove reference from B to C)

  3. Result:
     ■A ─→ □C    A is already black so it won't be re-scanned
     ■B           B's reference to C has been removed

  → C is reachable but remains white → erroneously collected!

  === Solutions: Write Barriers ===

  Dijkstra Barrier (snapshot-at-the-beginning):
    On write, turn the new reference target gray
    → "Newly referenced objects are definitely scanned"

  Yuasa Barrier (deletion barrier):
    On reference deletion, turn the deleted target gray
    → "Deleted reference targets are definitely scanned"

  Steele Barrier (incremental update):
    When a new reference is added to a black object, revert black→gray
    → "Black objects with new references are re-scanned"
```

### 7.4 Go's Goroutines and Concurrent GC

```go
// Go: Concurrent GC in practice

package main

import (
    "fmt"
    "runtime"
    "runtime/debug"
    "time"
)

func main() {
    // === Get GC statistics ===
    var stats debug.GCStats
    debug.ReadGCStats(&stats)
    fmt.Printf("GC count: %d\n", stats.NumGC)
    fmt.Printf("Last GC: %v\n", stats.LastGC)

    // === Get memory statistics ===
    var m runtime.MemStats
    runtime.ReadMemStats(&m)
    fmt.Printf("Heap in use: %.2f MB\n", float64(m.HeapAlloc)/1024/1024)
    fmt.Printf("Heap allocated: %.2f MB\n", float64(m.HeapSys)/1024/1024)
    fmt.Printf("GC count: %d\n", m.NumGC)
    fmt.Printf("Total GC pause time: %v\n", time.Duration(m.PauseTotalNs))

    // === GOGC configuration ===
    // GOGC=100 (default): GC when heap grows by 100%
    // GOGC=50: More frequent GC → saves memory, more CPU
    // GOGC=200: Less frequent GC → better throughput, more memory
    // GOGC=off: Disable GC (special use cases only)
    old := debug.SetGCPercent(50)
    fmt.Printf("Old GOGC: %d, New GOGC: 50\n", old)

    // === GOMEMLIMIT configuration (Go 1.19+) ===
    // Set a soft memory limit
    // Recommended to use in combination with GOGC
    debug.SetMemoryLimit(512 * 1024 * 1024) // 512MB

    // === Manual GC execution ===
    runtime.GC() // Execute GC immediately

    // === Wait for GC completion ===
    // runtime.GC() blocks until GC completes

    // === Setting finalizers ===
    type Resource struct {
        name string
    }
    r := &Resource{name: "database-connection"}
    runtime.SetFinalizer(r, func(res *Resource) {
        fmt.Printf("Finalizer: releasing %s\n", res.name)
    })
    // When r becomes unreachable, the finalizer runs at the next GC
    // Note: Finalizer execution timing is not guaranteed
}
```

---

## 8. GC Implementation Comparison Across Major Languages

### 8.1 Java GC Collector List

```
Evolution of Java GC Collectors:

  ┌──────────────┬──────────┬────────────┬─────────────────────────┐
  │ Collector     │ Intro    │ Target      │ Characteristics          │
  ├──────────────┼──────────┼────────────┼─────────────────────────┤
  │ Serial GC    │ JDK 1.0  │ Small-scale│ Single-threaded          │
  │              │          │ Client     │ Has STW                  │
  ├──────────────┼──────────┼────────────┼─────────────────────────┤
  │ Parallel GC  │ JDK 1.4  │ Throughput │ Multi-threaded GC        │
  │ (Throughput) │          │ focused    │ Has STW                  │
  ├──────────────┼──────────┼────────────┼─────────────────────────┤
  │ CMS          │ JDK 1.4  │ Low latency│ Concurrent Mark & Sweep  │
  │ (deprecated) │          │            │ Removed in Java 14       │
  ├──────────────┼──────────┼────────────┼─────────────────────────┤
  │ G1 GC        │ JDK 7    │ Balanced   │ Region-based             │
  │ (default)    │ (Java 9) │            │ Configurable pause target│
  ├──────────────┼──────────┼────────────┼─────────────────────────┤
  │ ZGC          │ JDK 11   │ Ultra-low  │ Pause time < 1ms         │
  │              │ (JDK 15) │ latency,   │ Up to 16TB heap          │
  │              │          │ large heap │ Colored pointers         │
  ├──────────────┼──────────┼────────────┼─────────────────────────┤
  │ Shenandoah   │ JDK 12   │ Ultra-low  │ Low latency like ZGC     │
  │              │          │ latency    │ Brooks pointer           │
  │              │          │            │ Led by Red Hat            │
  ├──────────────┼──────────┼────────────┼─────────────────────────┤
  │ Generational │ JDK 21   │ Latest     │ ZGC + Generational       │
  │ ZGC          │          │ recommended│ Separates Young/Old      │
  └──────────────┴──────────┴────────────┴─────────────────────────┘
```

### 8.2 G1 GC Region-Based Architecture

```
G1 GC Heap Structure:

  ┌────┬────┬────┬────┬────┬────┬────┬────┐
  │ E  │ E  │ S  │ O  │ O  │ H  │ O  │ E  │
  ├────┼────┼────┼────┼────┼────┼────┼────┤
  │ O  │ E  │ O  │ O  │ E  │ H  │ S  │ O  │
  ├────┼────┼────┼────┼────┼────┼────┼────┤
  │ E  │ O  │ O  │ S  │ O  │ O  │ E  │ O  │
  ├────┼────┼────┼────┼────┼────┼────┼────┤
  │ O  │ O  │ E  │ O  │ O  │ E  │ O  │ O  │
  └────┴────┴────┴────┴────┴────┴────┴────┘

  E = Eden Region      S = Survivor Region
  O = Old Region       H = Humongous Region (for large objects)

  Operation:
  1. Heap is divided into 1-32MB Regions (typically 2048)
  2. Each Region has a role of E/S/O/H
  3. Mixed GC: Prioritizes regions with highest collection efficiency
     → Origin of the name "Garbage First"
  4. Adjusts the number of regions collected to stay within
     the pause time target (-XX:MaxGCPauseMillis=200)

  Java command line:
  java -XX:+UseG1GC \
       -XX:MaxGCPauseMillis=200 \
       -XX:G1HeapRegionSize=4m \
       -Xms4g -Xmx4g \
       -Xlog:gc*:file=gc.log \
       MyApplication
```

### 8.3 ZGC Innovative Technology

```
ZGC (Z Garbage Collector) Architecture:

  Core technology: Colored Pointer

  64-bit pointer structure:
  ┌──────────────────────────────────────────────────────────────┐
  │ 63    47│46│45│44│43│42  │41                              0│
  │ (unused)│M │R │F │Md│Met │        Object address            │
  │         │a │e │i │  │a   │        (42 bits = 4TB)           │
  │         │r │m │n │  │    │                                 │
  │         │k │a │a │  │    │                                 │
  │         │e │p │l │  │    │                                 │
  │         │d │  │  │  │    │                                 │
  └──────────────────────────────────────────────────────────────┘

  - Marked:     Marked flag
  - Remapped:   Relocated flag
  - Finalizable: Finalizer pending flag

  ZGC Phases:
  1. Pause Mark Start     ← STW (extremely short: < 1ms)
     Root scanning only
  2. Concurrent Mark      ← Concurrent with application
     Object graph traversal
  3. Pause Mark End       ← STW (extremely short: < 1ms)
     Reference processing completion
  4. Concurrent Relocate  ← Concurrent with application
     Object relocation

  Java command line:
  java -XX:+UseZGC \
       -XX:+ZGenerational \    # Generational ZGC (recommended for JDK 21+)
       -Xms8g -Xmx8g \
       -Xlog:gc*:file=gc.log \
       MyApplication
```

### 8.4 Go's Concurrent Mark & Sweep

```
Go GC Characteristics:

  Design Philosophy:
  - Not generational (prioritizes simplicity)
  - Concurrent Mark & Sweep
  - Minimizes STW (target < 500μs)
  - Latency-focused (prioritizes latency over throughput)

  GC Cycle:
  ┌────────────────────────────────────────────────────────────┐
  │                                                            │
  │  ┌─STW─┐                              ┌─STW─┐            │
  │  │Mark │  Concurrent Mark & Sweep      │Mark │            │
  │  │Start│                               │Term │            │
  │  │     │  ┌──────────────────────────┐ │     │            │
  │  │<1ms │  │ Mutator and GC run       │ │<1ms │            │
  │  │     │  │ concurrently             │ │     │            │
  │  └─────┘  └──────────────────────────┘ └─────┘            │
  │                                                            │
  │  STW1       Concurrent marking          STW2   Sweeping    │
  │  (short)    (app running)               (short) (concurrent)│
  └────────────────────────────────────────────────────────────┘

  GC Pacing:
  - GOGC=100: Start GC when heap reaches 2x post-GC size
  - GOMEMLIMIT: Soft memory limit (Go 1.19+)
  - GC adjusts start timing based on heap growth rate

  Tuning Parameters:
  ┌────────────────────────┬───────────────────────────────────┐
  │ Parameter               │ Description                       │
  ├────────────────────────┼───────────────────────────────────┤
  │ GOGC=100               │ Start GC when heap grows by 100%  │
  │ GOGC=50                │ Frequent GC → saves memory, more CPU│
  │ GOGC=200               │ Less frequent GC → better throughput│
  │ GOGC=off               │ Disable GC                        │
  │ GOMEMLIMIT=512MiB      │ Soft memory limit                 │
  │ GODEBUG=gctrace=1      │ Enable GC trace                   │
  └────────────────────────┴───────────────────────────────────┘
```

### 8.5 Python's Hybrid GC

```
Python (CPython) GC Architecture:

  ┌─────────────────────────────────────────────────────────────┐
  │  Layer 1: Reference Counting (Main mechanism)               │
  │                                                             │
  │  - Reference count (ob_refcnt) on every object              │
  │  - Incremented/decremented immediately on reference add/del │
  │  - Immediately freed when rc=0                              │
  │  - GIL eliminates need for atomic operations (single-thread)│
  │                                                             │
  │  Problem: Cannot collect circular references                 │
  ├─────────────────────────────────────────────────────────────┤
  │  Layer 2: Generational Cycle Collector (Auxiliary mechanism) │
  │                                                             │
  │  Gen 0: New objects (threshold: 700)                        │
  │  Gen 1: Objects that survived Gen 0 once (threshold: 10)    │
  │  Gen 2: Objects that survived Gen 1 once (threshold: 10)    │
  │                                                             │
  │  Operation:                                                 │
  │  1. Run GC when Gen 0 object count exceeds threshold        │
  │  2. Detect and collect circular references                   │
  │  3. Only container objects are targeted                      │
  │     (int, str, etc. cannot form cycles so are excluded)     │
  └─────────────────────────────────────────────────────────────┘
```

```python
# Python: Detailed GC control

import gc
import sys

# === Check GC state ===
print("GC enabled:", gc.isenabled())
print("Generational thresholds:", gc.get_threshold())
# Output: (700, 10, 10)
# Gen 0: Triggers when alloc-dealloc difference reaches 700
# Gen 1: Triggers after Gen 0 runs 10 times
# Gen 2: Triggers after Gen 1 runs 10 times

# === GC statistics ===
stats = gc.get_stats()
for i, gen in enumerate(stats):
    print(f"Gen {i}: collections={gen['collections']}, "
          f"collected={gen['collected']}, "
          f"uncollectable={gen['uncollectable']}")

# === GC tuning ===
# Adjust thresholds (performance-oriented)
gc.set_threshold(1000, 15, 15)  # Reduce GC frequency

# Temporarily disable GC (for benchmarks, etc.)
gc.disable()
# ... measurement code ...
gc.enable()

# === Detecting uncollectable objects ===
# Circular references with __del__ may be uncollectable
gc.set_debug(gc.DEBUG_SAVEALL)
gc.collect()
print("Uncollectable:", gc.garbage)
```

### 8.6 Major Language GC Comparison Table

```
┌──────────┬────────────┬────────────┬──────────┬───────────────────┐
│ Language  │ GC Method   │ STW Time   │ Generat. │ Notes              │
├──────────┼────────────┼────────────┼──────────┼───────────────────┤
│ Java     │ G1/ZGC     │ <1ms(ZGC)  │ Yes      │ Multiple collectors│
│ (HotSpot)│            │ <200ms(G1) │          │ JFR for analysis   │
├──────────┼────────────┼────────────┼──────────┼───────────────────┤
│ Go       │ Concurrent │ <500μs     │ No       │ GOGC/GOMEMLIMIT   │
│          │ M&S        │            │          │ Simplicity focused │
├──────────┼────────────┼────────────┼──────────┼───────────────────┤
│ Python   │ Ref counting│ ms to tens │ Yes      │ GIL interaction    │
│ (CPython)│ + Cycle GC │ of ms      │ (3 gen)  │ RC is primary      │
├──────────┼────────────┼────────────┼──────────┼───────────────────┤
│ JS (V8)  │ Orinoco    │ <2ms(Young)│ Yes      │ Idle-Time GC      │
│          │ M&S+Copy   │ <10ms(Old) │ (2 gen)  │ Incremental       │
├──────────┼────────────┼────────────┼──────────┼───────────────────┤
│ C#       │ Generational│ <10ms     │ Yes      │ LOH, POH(.NET5+)  │
│ (.NET)   │ M&S+Compact│            │ (3 gen)  │ Server/Workstation│
├──────────┼────────────┼────────────┼──────────┼───────────────────┤
│ Ruby     │ Generational│ <10ms     │ Yes      │ Incremental M&S   │
│ (CRuby)  │ M&S        │            │ (2 gen)  │ RUBY_GC_* env vars│
├──────────┼────────────┼────────────┼──────────┼───────────────────┤
│ Swift    │ ARC        │ None       │ No       │ weak/unowned needed│
│          │ (ref count)│            │          │ Compile-time mgmt  │
├──────────┼────────────┼────────────┼──────────┼───────────────────┤
│ Rust     │ None       │ None       │ No       │ Ownership+borrowing│
│          │ (ownership)│            │          │ Zero GC overhead   │
└──────────┴────────────┴────────────┴──────────┴───────────────────┘
```

---

## 9. GC Tuning Strategies

### 9.1 Basic Tuning Principles

GC tuning is a domain where no "universal setting" exists. Strategy varies depending on
application characteristics (latency-focused vs throughput-focused vs memory-efficiency-focused).

```
GC Tuning Triangle (Trade-offs):

              Latency
              (Low pause time)
                 ╱╲
                ╱  ╲
               ╱    ╲
              ╱  GC   ╲
             ╱ Tuning   ╲
            ╱  Triangle   ╲
           ╱________________╲
   Throughput           Memory Efficiency
  (High processing     (Low memory usage)
   capacity)

  It is impossible to optimize all three simultaneously:
  - Latency priority   → More frequent GC → Lower throughput
  - Throughput priority → Less frequent GC → Higher memory usage
  - Memory priority    → Small heap → More frequent GC → Worse latency
```

### 9.2 Java GC Tuning in Practice

```java
// Java: Step-by-step approach to GC tuning

// === Step 1: Enable GC logging ===
// JDK 9+ unified logging:
// java -Xlog:gc*:file=gc.log:time,uptime,level,tags -jar app.jar

// === Step 2: Analyze GC logs ===
// How to read GC logs:
// [0.234s][info][gc] GC(0) Pause Young (Normal) (G1 Evacuation Pause)
//                        12M->8M(256M) 3.456ms
//                        ↑    ↑  ↑      ↑
//                    Pre-GC Post-GC Heap  Pause time

// === Step 3: Set heap size ===
// Set -Xms and -Xmx to the same value (avoid heap resizing)
// java -Xms4g -Xmx4g -jar app.jar

// === Step 4: Select GC collector ===
// Latency-focused:
//   java -XX:+UseZGC -XX:+ZGenerational -jar app.jar
// Balanced:
//   java -XX:+UseG1GC -XX:MaxGCPauseMillis=100 -jar app.jar
// Throughput-focused:
//   java -XX:+UseParallelGC -jar app.jar

// === Java Flight Recorder (JFR) for analysis ===
// java -XX:StartFlightRecording=filename=recording.jfr,
//       duration=60s,settings=profile -jar app.jar

// === Getting GC info from program ===
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.util.List;

public class GCMonitor {
    public static void printGCInfo() {
        List<GarbageCollectorMXBean> gcBeans =
            ManagementFactory.getGarbageCollectorMXBeans();

        for (GarbageCollectorMXBean gcBean : gcBeans) {
            System.out.printf("GC name: %s%n", gcBean.getName());
            System.out.printf("  Count: %d%n", gcBean.getCollectionCount());
            System.out.printf("  Cumulative time: %d ms%n", gcBean.getCollectionTime());
        }
    }

    // Cache using Weak References
    // GC automatically collects when memory is insufficient
    private static final java.util.Map<String, java.lang.ref.WeakReference<byte[]>> cache
        = new java.util.concurrent.ConcurrentHashMap<>();

    public static void cacheData(String key, byte[] data) {
        cache.put(key, new java.lang.ref.WeakReference<>(data));
    }

    public static byte[] getCachedData(String key) {
        java.lang.ref.WeakReference<byte[]> ref = cache.get(key);
        if (ref != null) {
            byte[] data = ref.get();
            if (data != null) return data;
            cache.remove(key); // Already collected by GC
        }
        return null;
    }
}
```

### 9.3 GC Tuning Checklist

```
GC Tuning Checklist:

  □ 1. Clarify goals
     ├── What is the max pause time target? (e.g., under 100ms)
     ├── What is the throughput target? (e.g., <5% time spent in GC)
     └── What are the memory constraints? (e.g., max 4GB)

  □ 2. Understand current state
     ├── Enable and analyze GC logs
     ├── Understand trends in GC frequency, pause time, heap usage
     └── Check for memory leaks

  □ 3. Optimize heap size
     ├── -Xms = -Xmx (avoid resizing)
     ├── Heap too small → excessive GC
     └── Heap too large → longer GC pauses

  □ 4. Select GC collector
     ├── G1: Appropriate for most cases (Java 9+ default)
     ├── ZGC: When ultra-low latency is needed
     ├── Parallel: Batch processing, throughput-focused
     └── Serial: Small apps / containers

  □ 5. Adjust generational sizes
     ├── Young generation too small → frequent Minor GCs
     ├── Young generation too large → longer Minor GC pauses
     └── Adjust with -XX:NewRatio, -XX:SurvivorRatio

  □ 6. Continuous monitoring
     ├── Real-time monitoring with JFR / JMX
     ├── Trend analysis with dashboards (Grafana, etc.)
     └── Set alerts (GC pause time > threshold)
```

### 9.4 Memory Management Patterns: GC vs Manual Management vs Ownership

```
Comprehensive Comparison of Three Memory Management Paradigms:

  ┌──────────────┬──────────────┬──────────────┬──────────────┐
  │ Aspect        │ GC           │ Manual Mgmt   │ Ownership    │
  │              │              │              │ (Rust)       │
  ├──────────────┼──────────────┼──────────────┼──────────────┤
  │ Safety        │ High         │ Low          │ Highest      │
  │ Memory leaks  │ Logical leaks│ Frequent     │ Structurally │
  │              │ possible     │              │ prevented    │
  │ Dangling ptrs │ Cannot occur │ Frequent     │ Compile error│
  │ Double free   │ Cannot occur │ Frequent     │ Compile error│
  │ Predictability│ Low (STW)    │ High         │ High         │
  │ Dev speed     │ High         │ Low          │ Med-High     │
  │ Runtime cost  │ 5-20%        │ 0%           │ 0-3%         │
  │ Compile-time  │ None         │ None         │ Borrow check │
  │ Learning curve│ Low          │ Moderate     │ Steep        │
  │ Real-time     │ Difficult    │ Possible     │ Possible     │
  │ Large-scale   │ Suitable     │ Difficult    │ Suitable     │
  │ Representative│ Java,Go,     │ C,C++        │ Rust         │
  │ languages    │ Python,JS,C# │              │              │
  └──────────────┴──────────────┴──────────────┴──────────────┘

  Selection Guidelines:
  - Web apps, business logic → GC (Java, Go, C#)
  - OS, embedded, game engines → Manual management (C, C++)
  - Systems programming → Ownership (Rust)
  - Scripting, data analysis → GC (Python, JS)
```

---

## 10. Anti-Patterns and Avoidance Strategies

### 10.1 Anti-Pattern 1: Hidden Memory Leaks (Logical Leaks)

"Logical memory leaks" can occur even in GC languages. Since GC does not collect objects
that are reachable from roots, memory grows indefinitely when unnecessary references are retained.

```java
// Java: Typical examples of logical memory leaks

// === Anti-pattern: Unbounded cache ===
public class LeakyCache {
    // Problem: Added entries are never GC'd
    private static final Map<String, byte[]> cache = new HashMap<>();

    public static void addToCache(String key, byte[] data) {
        cache.put(key, data);  // Accumulates without limit → memory leak
    }

    // Caller:
    // for (Request req : requests) {
    //     addToCache(req.getId(), req.getPayload());
    //     // cache keeps growing forever → OOM
    // }
}

// === Fix 1: Size-limited cache (LRU) ===
public class BoundedCache {
    private static final int MAX_SIZE = 1000;
    private static final Map<String, byte[]> cache =
        new LinkedHashMap<String, byte[]>(MAX_SIZE, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<String, byte[]> eldest) {
                return size() > MAX_SIZE;  // Remove oldest when max size exceeded
            }
        };

    public static void addToCache(String key, byte[] data) {
        cache.put(key, data);
    }
}

// === Fix 2: Use WeakHashMap ===
public class WeakCache {
    // Entries are automatically removed when keys are no longer referenced elsewhere
    private static final Map<Object, byte[]> cache = new WeakHashMap<>();

    public static void addToCache(Object key, byte[] data) {
        cache.put(key, data);
    }
}

// === Fix 3: Caffeine library (recommended) ===
// Caffeine: High-performance Java caching library
//
// Cache<String, byte[]> cache = Caffeine.newBuilder()
//     .maximumSize(10_000)
//     .expireAfterWrite(Duration.ofMinutes(5))
//     .build();
```

### 10.2 Anti-Pattern 2: Finalizer Abuse

```java
// Java: Finalizer anti-pattern and correct alternatives

// === Anti-pattern: Resource release via finalize() ===
public class BadResourceHandler {
    private java.io.InputStream stream;

    public BadResourceHandler(String path) throws Exception {
        this.stream = new java.io.FileInputStream(path);
    }

    // Problems:
    // 1. Timing of finalize() execution is indefinite
    // 2. No guarantee that finalize() will execute
    // 3. GC is delayed by finalize() execution
    // 4. Exceptions in finalize() are silently ignored
    // 5. Deprecated in Java 9+, scheduled for removal in Java 18+
    @Override
    protected void finalize() throws Throwable {
        try {
            if (stream != null) stream.close();
        } finally {
            super.finalize();
        }
    }
}

// === Fix: try-with-resources + AutoCloseable ===
public class GoodResourceHandler implements AutoCloseable {
    private final java.io.InputStream stream;

    public GoodResourceHandler(String path) throws Exception {
        this.stream = new java.io.FileInputStream(path);
    }

    @Override
    public void close() throws Exception {
        if (stream != null) stream.close();
    }

    // Usage:
    // try (GoodResourceHandler handler = new GoodResourceHandler("data.txt")) {
    //     // use handler
    // }  ← close() is automatically called when the block exits
}

// === Java 9+: Cleaner API (replacement for finalize) ===
import java.lang.ref.Cleaner;

public class ModernResourceHandler implements AutoCloseable {
    private static final Cleaner cleaner = Cleaner.create();

    private final Cleaner.Cleanable cleanable;
    private final ResourceState state;

    // Internal state class (implements Runnable)
    // Important: Must not hold a reference to the outer class
    private static class ResourceState implements Runnable {
        private java.io.InputStream stream;

        ResourceState(java.io.InputStream stream) {
            this.stream = stream;
        }

        @Override
        public void run() {
            // Fallback cleanup at GC time
            try {
                if (stream != null) stream.close();
            } catch (Exception e) {
                // Log the error
            }
        }
    }

    public ModernResourceHandler(String path) throws Exception {
        this.state = new ResourceState(new java.io.FileInputStream(path));
        this.cleanable = cleaner.register(this, state);
    }

    @Override
    public void close() {
        cleanable.clean();  // Explicit cleanup
    }
}
```

### 10.3 Anti-Pattern 3: Mass Generation of Short-Lived Objects

```
Problem with Mass Temporary Object Generation:

  Problematic pattern:
  ┌────────────────────────────────────────────────────────────┐
  │  for (int i = 0; i < 1_000_000; i++) {                    │
  │      String result = "prefix_" + i + "_suffix";  // ★     │
  │      // Multiple String objects generated per iteration     │
  │      // "prefix_" + i → new String                         │
  │      // new String + "_suffix" → another new String         │
  │      process(result);                                      │
  │  }                                                         │
  │                                                            │
  │  → 2M+ temporary Strings generated → GC pressure surges    │
  └────────────────────────────────────────────────────────────┘

  Improvement:
  ┌────────────────────────────────────────────────────────────┐
  │  StringBuilder sb = new StringBuilder(64);                 │
  │  for (int i = 0; i < 1_000_000; i++) {                    │
  │      sb.setLength(0);  // Clear buffer (reuse)             │
  │      sb.append("prefix_").append(i).append("_suffix");     │
  │      process(sb.toString());                               │
  │  }                                                         │
  │                                                            │
  │  → Reuse StringBuilder → dramatically fewer temp objects    │
  └────────────────────────────────────────────────────────────┘
```

### 10.4 Anti-Pattern 4: Calling System.gc()

```java
// Anti-pattern: Explicitly calling System.gc()

// Problematic code:
public void processLargeData(byte[] data) {
    // ... data processing ...
    data = null;
    System.gc();  // ★ Trying to force GC execution
}

// Why this is bad:
// 1. System.gc() is merely a "hint"; GC execution is not guaranteed
// 2. May trigger a Full GC causing a long STW
// 3. Interferes with GC timing optimization (pacing)
// 4. Often disabled in production with -XX:+DisableExplicitGC

// Correct approach:
// - Let the JVM's GC handle it
// - Adjust heap size or GC parameters if needed
// - Set unnecessary references to null (only effective for large objects)
```

---

## 11. Exercises (3 Levels)

### Level 1: Fundamentals (Understanding GC Algorithms)

**Problem 1-1: Manual Trace of Mark & Sweep**

Execute Mark & Sweep on the following object graph and list all objects
that are collected.

```
Roots: R1 → A, R2 → B

A → C, A → D
B → D
C → E
D → (none)
E → (none)
F → G
G → F (circular reference)
H → (none)
```

<details>
<summary>Answer (click to expand)</summary>

```
Marking phase:
  R1 → A(mark) → C(mark) → E(mark)
                → D(mark)
  R2 → B(mark) → D(already marked)

Marked objects: A, B, C, D, E
Unmarked objects: F, G, H

Collection targets: F, G, H

Note: F and G have circular references, but they are collected because
they are unreachable from roots. This is the major difference from
reference counting. With reference counting, F and G would have
rc=1 and could not be collected.
```

</details>

**Problem 1-2: Manual Trace of Reference Counting**

Track the reference count after each line of the following Python code.

```python
a = [1, 2, 3]      # (1) What is a's reference count?
b = a               # (2) What is a's reference count?
c = [a, b]          # (3) What is a's reference count?
del b               # (4) What is a's reference count?
c.pop()             # (5) What is a's reference count?
c.pop()             # (6) What is a's reference count?
del a               # (7) What happens to the list object?
```

<details>
<summary>Answer (click to expand)</summary>

```
(1) a=[1,2,3] → list rc=1 (reference from a)
(2) b=a       → list rc=2 (references from a, b)
(3) c=[a,b]   → list rc=4 (references from a, b, c[0], c[1])
(4) del b     → list rc=3 (references from a, c[0], c[1])
(5) c.pop()   → list rc=2 (references from a, c[0])
               * c[1] was removed
(6) c.pop()   → list rc=1 (reference from a only)
               * c[0] was removed
(7) del a     → list rc=0 → immediately freed!

Note: When checking with sys.getrefcount(a), it shows +1 because
getrefcount temporarily increases the reference as its argument.
```

</details>

### Level 2: Applied (GC Tuning and Diagnosis)

**Problem 2-1: GC Log Analysis**

Diagnose the problem from the following Java GC log and propose improvements.

```
[2.001s][info][gc] GC(0) Pause Young 128M->64M(512M) 5.2ms
[2.503s][info][gc] GC(1) Pause Young 192M->72M(512M) 6.1ms
[3.012s][info][gc] GC(2) Pause Young 200M->80M(512M) 7.3ms
[3.498s][info][gc] GC(3) Pause Young 208M->96M(512M) 8.5ms
[4.002s][info][gc] GC(4) Pause Young 224M->112M(512M) 10.2ms
[4.503s][info][gc] GC(5) Pause Full 240M->48M(512M) 350.1ms  ★
[5.001s][info][gc] GC(6) Pause Young 176M->64M(512M) 5.0ms
...(repeating pattern)
```

<details>
<summary>Answer (click to expand)</summary>

```
Diagnosis:
1. Surviving amount after Minor GC increases monotonically: 64→72→80→96→112MB
   → Promotions to Old generation are rapidly increasing
2. Full GC occurs at GC(5): 350.1ms STW
   → Old generation became full
3. After Full GC it returns to 48MB
   → Few truly long-lived objects (many temporary promotions)

Estimated cause:
- Many objects with moderate lifetimes (don't die in Young but
  get promoted to Old, yet become unnecessary soon in Old)
- Young generation may be too small

Improvements:
1. Enlarge Young generation: -XX:NewRatio=1 (Young:Old=1:1)
   → Increase probability of objects dying within Young
2. Adjust Survivor: -XX:MaxTenuringThreshold=15
   → Raise the promotion threshold
3. For G1 GC: -XX:MaxGCPauseMillis=100
   → Set pause time target to avoid Full GC
4. Consider switching to ZGC: -XX:+UseZGC
   → Fundamentally eliminates long STW from Full GC
```

</details>

**Problem 2-2: Identifying Memory Leaks**

The following Node.js code has a memory leak. Identify the cause and fix it.

```javascript
const EventEmitter = require('events');

class DataProcessor extends EventEmitter {
    constructor() {
        super();
        this.cache = new Map();
    }

    process(data) {
        const id = data.id;
        const result = this.transform(data);
        this.cache.set(id, result);

        const handler = (event) => {
            console.log(`Event for ${id}: ${event}`);
            console.log(`Cache size: ${this.cache.size}`);
        };
        this.on('update', handler);

        return result;
    }

    transform(data) {
        return { ...data, processed: true, timestamp: Date.now() };
    }
}

const processor = new DataProcessor();
// Process new data every second
setInterval(() => {
    const data = { id: Math.random().toString(36), payload: 'x'.repeat(1024) };
    processor.process(data);
}, 1000);
```

<details>
<summary>Answer (click to expand)</summary>

```
There are two causes of memory leaks:

1. cache grows without bound
   - process() calls cache.set() but there is no deletion logic
   - → Map entries accumulate indefinitely

2. Event listeners are added without limit
   - Each call to process() adds a new listener via this.on('update', handler)
   - Listeners continue to reference id, this.cache through closures
   - → Listener count increases indefinitely

Fixed version:
class DataProcessor extends EventEmitter {
    constructor(maxCacheSize = 1000) {
        super();
        this.cache = new Map();
        this.maxCacheSize = maxCacheSize;
    }

    process(data) {
        const id = data.id;
        const result = this.transform(data);

        // Fix 1: Limit cache size
        if (this.cache.size >= this.maxCacheSize) {
            const oldestKey = this.cache.keys().next().value;
            this.cache.delete(oldestKey);
        }
        this.cache.set(id, result);

        // Fix 2: Don't add listeners (manage separately)
        // Or use once() to execute only once

        return result;
    }
}
```

</details>

### Level 3: Advanced (GC Algorithm Implementation and Analysis)

**Problem 3-1: Generational GC Simulator Design**

Design a generational GC simulator in Python that meets the following specifications.
(Full implementation is not required. Provide class structure, method signatures, and key pseudocode.)

Requirements:
- Has Young generation (Eden + Survivor x 2) and Old generation
- Minor GC: Copy surviving objects from Eden to Survivor
- Promotion: Move objects to Old that survived the threshold number of Minor GCs
- Major GC: Mark & Sweep of the Old generation
- Write Barrier: Record Old → Young references in a Remembered Set

<details>
<summary>Answer (click to expand)</summary>

```python
class GenerationalGC:
    def __init__(self, eden_size, survivor_size, old_size,
                 tenuring_threshold=15):
        self.eden = Space("Eden", eden_size)
        self.survivor_from = Space("S0", survivor_size)
        self.survivor_to = Space("S1", survivor_size)
        self.old = Space("Old", old_size)
        self.tenuring_threshold = tenuring_threshold
        self.remembered_set = set()  # Records Old → Young references
        self.roots = []

    def allocate(self, size):
        """Allocate object in Eden. Minor GC if full"""
        if not self.eden.can_allocate(size):
            self.minor_gc()
            if not self.eden.can_allocate(size):
                self.major_gc()  # Major GC if Old is also full
        obj = self.eden.allocate(size)
        obj.age = 0
        return obj

    def minor_gc(self):
        """Young generation GC (Copying GC)"""
        # Root scanning
        # 1. Copy Young objects reachable from stack/global roots
        # 2. Also copy Young objects reachable from Remembered Set entries
        # 3. Promote to Old if age >= threshold
        # 4. Otherwise copy to survivor_to (age+1)
        # 5. Clear Eden and survivor_from
        # 6. Swap survivor_from and survivor_to
        for root in self.roots + list(self.remembered_set):
            self._copy_reachable(root)
        self.eden.clear()
        self.survivor_from.clear()
        self.survivor_from, self.survivor_to = (
            self.survivor_to, self.survivor_from
        )

    def major_gc(self):
        """Old generation GC (Mark & Sweep)"""
        # 1. Reset mark bits for all objects
        # 2. Mark objects reachable from roots
        # 3. Free unmarked Old objects
        self._mark_all()
        self._sweep_old()

    def write_barrier(self, src, dst):
        """Detect and record Old → Young references"""
        if self.old.contains(src) and not self.old.contains(dst):
            self.remembered_set.add(src)
```

</details>

---

## 12. FAQ (Frequently Asked Questions)

### Q1: Do memory leaks never occur if there is GC?

**A:** No. GC only collects "unreachable objects." Objects that are reachable but unnecessary
(logical leaks) are not collected. Typical examples include:

- Caches that grow without limit (HashMap, etc.)
- Event listeners that are registered but never unregistered
- ThreadLocal variables not cleared (when used with thread pools)
- Collections held in static fields
- Closures that continue to capture unnecessary variables

Countermeasures include cache size limits (LRU, TTL), use of WeakReferences,
and regular inspection with memory profilers (VisualVM, Chrome DevTools, pprof, etc.).

### Q2: Can GC pause time be reduced to zero?

**A:** Making it completely zero is theoretically difficult, but it can be made extremely short.

- **Java ZGC**: STW < 1ms (independent of heap size). Uses colored pointers and
  load barriers to perform nearly all work concurrently.
- **Go**: Targets STW < 500μs, and often completes in tens of μs in practice.
- **Azul C4 GC** (commercial): Claims to be "Pauseless GC," operating without STW.

An important note: "no STW" and "no impact on latency" are different things.
Concurrent GC consumes CPU in the background, which may indirectly affect
application throughput and latency.

### Q3: Why is Rust safe without GC?

**A:** Instead of GC, Rust uses the "ownership system" and "borrow checker" to
guarantee memory safety at compile time.

- **Ownership**: Each value has a unique owner, and the value is automatically freed when the owner goes out of scope.
- **Borrowing**: References (borrows) to a value are either immutable (multiple allowed) or mutable (only one), statically verified by the compiler.
- **Lifetimes**: The compiler tracks the validity period of references, structurally preventing dangling pointers.

Trade-offs compared to GC:
- Advantages: Zero runtime overhead, no STW, deterministic resource release
- Disadvantages: Steep learning curve, expressing circular data structures is somewhat complex (requires Rc/Arc + RefCell)

### Q4: Which language should I choose (from a memory management perspective)?

**A:** It depends on project requirements.

| Requirement | Recommended Languages | Reason |
|------|---------|------|
| Web backend | Java, Go, C# | Mature GC, rich ecosystem |
| Low latency | Go, Java(ZGC), Rust | Short STW, or no STW |
| Embedded systems | C, Rust | GC overhead is not acceptable |
| Data science | Python | Development without worrying about GC |
| Frontend | JavaScript/TypeScript | V8's GC auto-optimizes |
| Game engines | C++, Rust | Real-time performance is essential |
| Mobile apps | Swift(iOS), Kotlin(Android) | ARC/GC is the platform standard |

### Q5: How much is the GC overhead?

**A:** It varies greatly by workload, but general guidelines are as follows.

- **CPU overhead**: 2-10% of total (cost of GC thread execution)
- **Memory overhead**: 20-50% (GC metadata + excess usage from delayed collection)
- **Latency impact**: Depends on GC collector. < 1ms for ZGC, potentially hundreds of ms for G1

To minimize GC overhead:
1. Reduce object allocation rate (object pooling, buffer reuse)
2. Break unnecessary references from long-lived objects
3. Select appropriate GC collector and parameters
4. Set heap size appropriately

---


## FAQ

### Q1: What is the most important point in learning this topic?

Building practical experience is most important. Understanding deepens not just from theory, but from actually writing code and observing the behavior.

### Q2: What mistakes do beginners commonly make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before proceeding to the next step.

### Q3: How is this applied in practice?

Knowledge of this topic is frequently applied in day-to-day development work. It becomes particularly important during code reviews and architecture design.

---

## 13. Summary and Next Steps

### Overall Map of GC Algorithms

```
GC Algorithm Classification:

  Garbage Collection
  │
  ├── Tracing GC (reachability-based)
  │   ├── Mark & Sweep ──── Basic form. Handles cycles, fragmentation issues
  │   ├── Mark & Compact ── Eliminates fragmentation, high cost
  │   ├── Copying GC ────── Automatic compaction, uses 50% of memory
  │   ├── Generational GC ── Fast collection of young objects
  │   ├── Incremental GC ── Splits STW into smaller pieces
  │   └── Concurrent GC ─── Runs GC in the background
  │
  └── Reference Counting (count-based)
      ├── Simple RC ──────── Immediate collection, cannot handle cycles
      ├── Deferred RC ────── Delays root updates
      └── ARC ────────────── Compiler auto-inserts (Swift)
```

### Key Points Summary

| Algorithm | Core Idea | Main Advantage | Main Disadvantage | Usage |
|------------|------------|---------|---------|-------|
| Mark & Sweep | Reachability-based | Handles cycles | STW, fragmentation | Foundation for many languages |
| Reference Counting | Count-based | Immediate collection | Cannot handle cycles | Python, Swift |
| Copying GC | Copy surviving objects | No fragmentation | 2x memory | Young generation |
| Generational GC | Optimized by generational hypothesis | Fast Minor GC | Implementation complexity | Java, JS, C# |
| Concurrent GC | Background execution | Minimizes STW | Requires write barriers | Go, Java(ZGC) |

---

## Recommended Next Guides


---

## 14. References

1. Jones, R., Hosking, A. & Moss, E. *The Garbage Collection Handbook: The Art of Automatic Memory Management.* 2nd Edition, CRC Press, 2023. -- The definitive text comprehensively covering GC theory and implementation. Provides detailed explanations of Mark & Sweep, Copying GC, Generational GC, and Concurrent GC.

2. McCarthy, J. "Recursive Functions of Symbolic Expressions and Their Computation by Machine, Part I." *Communications of the ACM*, Vol.3, No.4, pp.184-195, 1960. -- The original paper on Lisp and GC. One of the most important papers in programming language history, containing the first proposal of the Mark & Sweep method.

3. Dijkstra, E.W., Lamport, L., Martin, A.J., Scholten, C.S. & Steffens, E.F.M. "On-the-Fly Garbage Collection: An Exercise in Cooperation." *Communications of the ACM*, Vol.21, No.11, pp.966-975, 1978. -- A classic paper that established the theoretical foundations of tri-color marking and concurrent GC.

4. Bacon, D.F., Cheng, P. & Rajan, V.T. "A Unified Theory of Garbage Collection." *Proceedings of the 19th ACM SIGPLAN Conference on Object-Oriented Programming, Systems, Languages, and Applications (OOPSLA)*, pp.50-68, 2004. -- A groundbreaking paper demonstrating that tracing GC and reference counting are mathematically dual.

5. Oracle. *Java Platform, Standard Edition HotSpot Virtual Machine Garbage Collection Tuning Guide.* https://docs.oracle.com/en/java/javase/21/gctuning/ -- Official GC tuning guide for Java 21 LTS. Explains configuration and optimization methods for G1 GC, ZGC, Serial GC, and Parallel GC.

6. Go Team. *A Guide to the Go Garbage Collector.* https://tip.golang.org/doc/gc-guide -- Official Go GC guide. Explains how to use GOGC and GOMEMLIMIT, and the GC pacing algorithm.

7. V8 Team. *Trash Talk: The Orinoco Garbage Collector.* https://v8.dev/blog/trash-talk -- Official blog post explaining the design and implementation of V8's GC (Orinoco). Includes details on Scavenger, concurrent marking, and incremental marking.

8. Tene, G., Iyengar, B. & Wolf, M. "C4: The Continuously Concurrent Compacting Collector." *Proceedings of the International Symposium on Memory Management (ISMM)*, ACM, 2011. -- Paper explaining the design of Azul Systems' Pauseless GC (C4). The cutting edge of commercial GC.

---

## References

- [MDN Web Docs](https://developer.mozilla.org/) - Web technology reference
- [Wikipedia](https://en.wikipedia.org/) - Overview of technical concepts
