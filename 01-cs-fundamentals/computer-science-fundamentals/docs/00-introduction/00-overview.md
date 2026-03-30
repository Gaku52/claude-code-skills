# The Big Picture of Computer Science

> CS is a discipline that explores "what computation is," and programming is merely one aspect of it.

## What You Will Learn in This Chapter

- [ ] Explain the definition and major fields of computer science
- [ ] Understand how the various fields of CS relate to one another
- [ ] Grasp the overall structure of this Skill and how to approach your studies
- [ ] Apply the four elements of Computational Thinking in practice
- [ ] Explain how CS intersects with other academic disciplines

## Prerequisites

- None (this guide is the starting point for CS study)

---

## 1. What Is Computer Science?

### 1.1 Definition

**Computer Science (CS)** is the academic discipline that studies the theory and practice of "computation." It is not merely the study of "how to use computers" but rather the exploration of **"what is computable and how can we compute it efficiently."**

The definition by ACM (Association for Computing Machinery) and IEEE Computer Society:

> "Computer Science is the study of computers and computational systems. Unlike electrical and computer engineers, computer scientists deal mostly with software and software systems; this includes their theory, design, development, and application."
> — ACM/IEEE Computing Curricula 2020

More fundamentally, CS is a discipline that answers three root questions:

```
┌─────────────────────────────────────────────────────────┐
│         The Three Grand Questions of Computer Science    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. What is computable? (Computability Theory)          │
│     → The Halting Problem, Turing's incompleteness      │
│                                                         │
│  2. How efficiently can we compute? (Complexity Theory)  │
│     → P vs NP problem, algorithm design                 │
│                                                         │
│  3. How do we compute correctly? (Software Engineering)  │
│     → Formal verification, testing, design patterns     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 1.2 CS Is Not "Programming"

Many people confuse CS with programming, but the two are fundamentally different.

A famous quote by **Edsger Dijkstra**:

> "Computer Science is no more about computers than astronomy is about telescopes."

| Perspective | Programming | Computer Science |
|------------|-------------|------------------|
| **Essence** | The craft of writing code | The theory and practice of computation |
| **Focus** | "How to build it" (How) | "Why it should be built that way" (Why) |
| **Rate of change** | Frameworks change every 3-5 years | Foundational theory has remained unchanged for 50+ years |
| **Examples** | React, Django, SwiftUI | Algorithms, data structures, complexity |
| **Skills** | Language syntax, APIs, tools | Problem analysis, abstraction, proofs |
| **Learning approach** | Practice (coding) | Theory + practice (mathematics + implementation) |
| **Lifespan** | Depends on the technology generation | Universal (Turing's theory dates to 1936) |

```
The relationship between programming and CS, using an architecture analogy:

  Programming ≈ Construction worker (the skill of laying bricks)
  CS          ≈ Architecture (structural mechanics + material science + design theory)

  A construction worker can build a building,
  but without knowledge of architecture,
  they cannot explain why steel beams are H-shaped
  or why the foundation must be a certain depth.
```

### 1.3 Why "Computer" Science?

The name CS is a historical artifact. In reality, "Computation Science" would be a more accurate name.

The subjects of CS research are not limited to physical computers:
- **Turing machines**: Abstract computational models that do not physically exist
- **Lambda calculus**: A mathematical theory of functions that requires no computer
- **Algorithms**: Euclid's algorithm for GCD has existed since antiquity
- **Information theory**: A theory dealing with channel capacity

### 1.4 Computational Thinking

The essential value of CS lies in a problem-solving methodology called "computational thinking." Proposed by Jeannette Wing in 2006, this concept can be broadly applied to fields beyond CS:

```
The Four Elements of Computational Thinking:

┌─────────────────────────────────────────────────────────────┐
│                 Computational Thinking                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Decomposition                                           │
│     Break complex problems into smaller sub-problems        │
│     Example: Web app → Frontend + Backend + DB              │
│                                                             │
│  2. Pattern Recognition                                     │
│     Identify common patterns across different problems      │
│     Example: Shortest path and schedule optimization        │
│             are both graph problems                         │
│                                                             │
│  3. Abstraction                                             │
│     Ignore non-essential details and focus on what matters  │
│     Example: Each TCP/IP layer hides the details of the     │
│             layer below                                     │
│                                                             │
│  4. Algorithmic Thinking                                    │
│     Express solutions as clear, reproducible steps          │
│     Example: Cooking recipes, software specifications       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Practical Applications of Computational Thinking**:

```python
# Improving API performance using computational thinking

# 1. Decomposition: Profile a slow API endpoint to break it down
# → DB query: 800ms / Business logic: 50ms / Serialization: 150ms
# → Identified DB query as the bottleneck

# 2. Pattern Recognition: Other slow APIs exhibit the same issue
# → Discovered a pattern: all caused by N+1 queries

# 3. Abstraction: Focus on ORM usage patterns rather than individual API details
# → Systematic application of prefetch_related / select_related

# 4. Algorithmic Thinking: Define a clear improvement procedure
def optimize_api():
    """Standard procedure for API optimization"""
    # Step 1: Identify bottleneck through profiling
    profile = measure_endpoint_performance()

    # Step 2: Automatically detect N+1 problems
    n_plus_one = detect_n_plus_one_queries(profile)

    # Step 3: Apply JOINs or prefetching
    for query in n_plus_one:
        apply_prefetch(query)

    # Step 4: Measure effectiveness
    assert measure_improvement() > 0.5  # At least 50% improvement
```

### 1.5 CS and Its Relationship with Adjacent Fields

CS intersects with many fields, giving rise to new academic disciplines:

```
Intersections of CS with Adjacent Fields:

  Mathematics ────────┬── Computation theory, cryptography
                      │
  Physics ────────────┤── Quantum computing, simulation
                      │
  Biology ────────────┤── Bioinformatics, computational biology
                      │
  Economics ──────────┤── Algorithmic trading, computational economics
                      │
  Linguistics ────────┤── Natural language processing, computational linguistics
                      │
  Psychology ─────────┤── HCI, cognitive science
                      │
  Medicine ───────────┤── Medical imaging AI, electronic health records
                      │
  Art ────────────────┤── Computer graphics, generative AI
                      │
  Law ────────────────┴── AI regulation, data privacy
```

This interdisciplinary nature is one of CS's greatest strengths — studying CS enables you to apply computational methods to other fields.

---

## 2. The 10 Major Fields of CS

CS covers an extremely broad range of academic areas. Based on the ACM Computing Classification System, here is an overview of the 10 major fields.

```
┌──────────────────────────── CS Academic Discipline Map ────────────────────────────┐
│                                                                                    │
│                          ┌──────────────────────┐                                  │
│                          │  Theory of Computation│ ← Mathematical foundation       │
│                          └──────────┬───────────┘                                  │
│                                     │                                              │
│               ┌─────────────────────┼─────────────────────┐                        │
│               ▼                     ▼                     ▼                        │
│    ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐               │
│    │ Algorithms       │  │ Data Structures  │  │ Programming      │               │
│    │                  │  │                  │  │ Languages (PL)   │               │
│    └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘               │
│             │                     │                     │                          │
│             └─────────────────────┼─────────────────────┘                          │
│                                   │                                                │
│               ┌───────────────────┼───────────────────┐                            │
│               ▼                   ▼                   ▼                            │
│    ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐               │
│    │ Operating        │  │ Networks         │  │ Databases        │               │
│    │ Systems (OS)     │  │                  │  │                  │               │
│    └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘               │
│             │                     │                     │                          │
│             └─────────────────────┼─────────────────────┘                          │
│                                   │                                                │
│               ┌───────────────────┼───────────────────┐                            │
│               ▼                   ▼                   ▼                            │
│    ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐               │
│    │ Software         │  │ Artificial       │  │ Security         │               │
│    │ Engineering (SE) │  │ Intelligence     │  │                  │               │
│    └──────────────────┘  │ (AI/ML)          │  └──────────────────┘               │
│                          └──────────────────┘                                      │
│                                   │                                                │
│                                   ▼                                                │
│                          ┌──────────────────┐                                      │
│                          │ HCI / UX Design  │ ← Human interface                   │
│                          └──────────────────┘                                      │
│                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
```

### Field 1: Theory of Computation

Deals with the mathematical foundations of computation. Rigorously defines what is computable and what is not.

- **Automata theory**: Finite state machines, pushdown automata
- **Formal languages**: Regular languages, context-free languages, Chomsky hierarchy
- **Computability**: Turing machines, the Halting Problem, undecidability
- **Complexity theory**: P, NP, NP-complete, PSPACE

**Impact on practice**: Enables you to understand why regular expressions cannot match recursive patterns and why a perfect bug detection tool is impossible to build.

**Concrete practical example**:

```python
# Example illustrating the limits of regular expressions
import re

# ✅ Regular language: expressible with regex
email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
re.match(email_pattern, 'user@example.com')  # Matches

# ❌ Context-free language: fundamentally impossible with regex
# "Checking balanced parentheses"
# ((()))  → OK
# (()     → NG
# Regex cannot track nesting depth of parentheses
# → A parser (pushdown automaton) is required

# ❌ Halting Problem: undecidable by any algorithm
# "Determining whether an arbitrary program enters an infinite loop"
# is fundamentally impossible
# → A perfect static analysis tool cannot exist (approximation is possible)
```

### Field 2: Algorithms

The design and analysis of efficient methods for solving problems.

- **Sorting**: Quicksort O(n log n) vs Bubble sort O(n²)
- **Searching**: Binary search O(log n), hash lookup O(1)
- **Graphs**: Shortest path (Dijkstra), minimum spanning tree (Kruskal)
- **Dynamic programming**: Exploiting optimal substructure and overlapping subproblems

**Impact on practice**: Improving from O(n²) to O(n log n) when processing 1 million records reduces time from 11.5 days to 20 seconds.

**Practical application of algorithms**:

```python
# Dijkstra's algorithm — the foundation of GPS navigation and network routing
import heapq

def dijkstra(graph, start):
    """Find shortest paths: O((V+E) log V)"""
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]  # Priority queue of (distance, node)

    while pq:
        current_dist, current = heapq.heappop(pq)
        if current_dist > distances[current]:
            continue

        for neighbor, weight in graph[current].items():
            distance = current_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))

    return distances

# Usage: Finding the lowest-latency path between servers
network = {
    'tokyo': {'osaka': 5, 'singapore': 30},
    'osaka': {'tokyo': 5, 'singapore': 25},
    'singapore': {'tokyo': 30, 'osaka': 25, 'sydney': 40},
    'sydney': {'singapore': 40}
}
print(dijkstra(network, 'tokyo'))
# {'tokyo': 0, 'osaka': 5, 'singapore': 30, 'sydney': 70}
```

### Field 3: Data Structures

Methods for efficiently storing and manipulating data.

- **Linear**: Arrays, linked lists, stacks, queues
- **Trees**: Binary search trees, AVL trees, B+ trees, tries
- **Hashing**: Hash tables, Bloom filters
- **Graphs**: Adjacency matrices, adjacency lists, Union-Find

**Impact on practice**: Simply changing `Array.includes()` at O(n) to `Set.has()` at O(1) can make an API 100x faster.

**Practical guide to choosing data structures**:

```python
# Selecting the optimal data structure for each scenario

# Scenario 1: Checking user ID existence (frequent lookups)
# → Hash set O(1)
active_users = set()
active_users.add(user_id)
if user_id in active_users:  # O(1)
    pass

# Scenario 2: Displaying rankings (ordered, range queries)
# → Sorted list or balanced BST
import sortedcontainers
ranking = sortedcontainers.SortedList(key=lambda x: -x['score'])
ranking.add({'name': 'Alice', 'score': 950})
top_10 = ranking[:10]  # Top 10 entries

# Scenario 3: Undo/Redo functionality
# → Stack
undo_stack = []
redo_stack = []
def execute_action(action):
    undo_stack.append(action)
    redo_stack.clear()
    action.execute()

# Scenario 4: Task queue (FIFO processing)
# → Queue (deque)
from collections import deque
task_queue = deque()
task_queue.append(task)      # Enqueue O(1)
next_task = task_queue.popleft()  # Dequeue O(1)

# Scenario 5: String autocomplete
# → Trie (Prefix Tree)
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
```

### Field 4: Operating Systems

The bridge between hardware and applications.

- **Process management**: Scheduling, multithreading, deadlocks
- **Memory management**: Virtual memory, paging, garbage collection
- **File systems**: inodes, journaling, COW
- **I/O**: Interrupts, DMA, epoll/kqueue

**Impact on practice**: Enables deep understanding of Node.js's event loop, Docker's cgroups/namespaces, and io_uring.

### Field 5: Computer Networks

How computers communicate with each other.

- **Protocols**: TCP/IP, UDP, HTTP/2/3, gRPC
- **Security**: TLS, certificate chains, HTTPS
- **Architecture**: DNS, CDN, load balancing
- **Emerging technologies**: QUIC, WebRTC, WebTransport

**Impact on practice**: Enables fundamental understanding of "why HTTP/3 is faster" and "why WebSocket is needed."

### Field 6: Databases

Persisting structured data and enabling efficient retrieval.

- **Relational**: SQL, normalization, transactions (ACID)
- **NoSQL**: Document, key-value, graph DB, columnar
- **Indexes**: B+ trees, hash indexes, GiST
- **Distributed databases**: CAP theorem, replication, sharding

**Impact on practice**: Proper index design can improve a query on 1 million rows from 30 seconds to 0.01 seconds.

### Field 7: Artificial Intelligence (AI / Machine Learning)

Realizing intelligent behavior with computers.

- **Classical AI**: Search, planning, expert systems
- **Machine learning**: Supervised, unsupervised, reinforcement learning
- **Deep learning**: CNN, RNN, Transformer
- **Generative AI**: LLMs (GPT, Claude), diffusion models (Stable Diffusion)

**Impact on practice**: Engineering in the AI era (RAG, fine-tuning, agent design).

### Field 8: Software Engineering

Methodologies for building large-scale software correctly and efficiently.

- **Development methodologies**: Agile, Scrum, XP, DevOps
- **Design**: SOLID principles, design patterns, clean architecture
- **Testing**: Unit testing, integration testing, TDD, BDD
- **Quality**: Code review, CI/CD, refactoring

**Impact on practice**: Fundamentally improves productivity and quality in team development.

### Field 9: Computer Security

Protecting systems and data from threats.

- **Cryptography**: Symmetric encryption (AES), public-key cryptography (RSA), hashing (SHA-256)
- **Web**: XSS, SQL injection, CSRF, OWASP Top 10
- **Authentication**: OAuth 2.0, JWT, passkeys
- **Infrastructure**: Firewalls, IDS/IPS, zero trust

**Impact on practice**: Secure system design and vulnerability prevention.

### Field 10: Human-Computer Interaction (HCI)

Designing the interface between humans and computers.

- **UI design**: Fitts's Law, Hick's Law, Gestalt principles
- **UX**: Usability testing, personas, journey maps
- **Accessibility**: WCAG, screen reader support
- **Emerging interfaces**: VR/AR, voice UI, brain-computer interfaces

**Impact on practice**: Provides scientific foundations for designing usable products.

---

## 3. Understanding the Difference Between CS and Programming — Through Concrete Examples

### Example 1: Searching an Array

Programming-oriented thinking:
```python
# The "just make it work" approach
def find_user(users, target_id):
    for user in users:
        if user['id'] == target_id:
            return user
    return None
```

CS-oriented thinking:
```python
# An approach based on understanding *why* it's slow
# O(n) linear search → Convert to O(1) hash table
def build_user_index(users):
    """Preprocessing: Build a dictionary in O(n)"""
    return {user['id']: user for user in users}

def find_user(user_index, target_id):
    """Lookup: O(1) access"""
    return user_index.get(target_id)

# When searching among 100,000 users:
# Linear search: Average 50,000 comparisons
# Hash: 1 hash computation + 1 access
```

### Example 2: Fibonacci Sequence

Programming-oriented thinking:
```python
# Intuitive recursive implementation
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)
# fib(50) → Takes minutes to hours (O(2^n))
```

CS-oriented thinking:
```python
# Apply dynamic programming (recognizing overlapping subproblems)
def fib(n):
    if n <= 1:
        return n
    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr
    return curr
# fib(50) → Completes instantly (O(n))
# Can also achieve O(log n) with matrix exponentiation
```

### Example 3: Character Encoding Issues in a Web App

Programming-oriented thinking: "Characters are garbled → Google it and add charset=utf-8"

CS-oriented thinking:
- UTF-8 is a variable-length encoding (1-4 bytes)
- An ingeniously ASCII-compatible design (leading bits determine length)
- Presence or absence of BOM, surrogate pairs, normalization (NFC/NFD)
- Consistency with the database's collation settings

→ Because you understand the root cause, the same problem never occurs again.

### Example 4: Designing a Cache Strategy

Programming-oriented thinking: "Just add Redis and it'll be faster"

CS-oriented thinking:
```python
# Deriving cache design from CS principles

# 1. Principle of Locality (memory hierarchy knowledge)
# → Temporal locality: Recently accessed data is likely to be accessed again
# → LRU (Least Recently Used) cache is effective

from collections import OrderedDict

class LRUCache:
    """LRU cache implementation — leveraging OrderedDict"""
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)  # Move to most-recently-used O(1)
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # Evict oldest O(1)

# 2. Cache Consistency (distributed systems knowledge)
# → Write-Through: Update both cache and DB on writes
# → Write-Back: Update only cache, flush to DB later
# → Cache-Aside: On cache miss during reads, fetch from DB

# 3. Cache Invalidation (computation theory knowledge)
# → "There are only two hard things in CS:
#    cache invalidation and naming things." — Phil Karlton
# → Choose among TTL, event-driven invalidation, and versioning
```

### Example 5: Correct Design of Concurrent Processing

Programming-oriented thinking: "Just use multithreading and it'll be faster"

CS-oriented thinking:
```python
# Amdahl's Law: Calculate the limits of parallelization in advance

def amdahl_speedup(parallel_fraction: float, num_processors: int) -> float:
    """Calculate theoretical maximum speedup using Amdahl's Law

    Args:
        parallel_fraction: Fraction that can be parallelized (0.0 - 1.0)
        num_processors: Number of processors

    Returns:
        Speedup factor
    """
    serial_fraction = 1 - parallel_fraction
    return 1 / (serial_fraction + parallel_fraction / num_processors)

# Example: If 80% of the program is parallelizable
print(f"2 cores: {amdahl_speedup(0.8, 2):.2f}x")     # 1.67x
print(f"4 cores: {amdahl_speedup(0.8, 4):.2f}x")     # 2.50x
print(f"8 cores: {amdahl_speedup(0.8, 8):.2f}x")     # 3.33x
print(f"∞ cores: {amdahl_speedup(0.8, 10000):.2f}x")  # 5.00x (upper limit!)
# → Even with infinite cores, 5x is the maximum (the 20% serial portion is the bottleneck)
# → The CS lesson: "Optimize the serial portion before parallelizing"
```

---

## 4. Overview of CS Degree Curricula

### Comparison: MIT / Stanford / CMU

| Area | MIT (6-3) | Stanford (BS CS) | CMU (SCS) |
|------|-----------|-------------------|-----------|
| **Math foundations** | Linear algebra + Calculus + Probability & statistics | Same + Discrete math | Same + Mathematical logic |
| **Programming** | Python (6.100A) | Java (CS106A) | SML/C (15-150) |
| **Algorithms** | 6.006 + 6.046 | CS161 | 15-451 |
| **Systems** | 6.004 (Computation Structures) + 6.033 | CS107 + CS110 | 15-213 (CS:APP) |
| **AI/ML** | 6.034 + 6.036 | CS221 + CS229 | 10-301 + 10-315 |
| **Theory** | 6.045 (Theory of Computation) | CS154 | 15-251 (Great Ideas) |
| **Electives** | Numerous elective courses | Track system | Specialization-based |
| **Distinguishing feature** | Balance of theory and practice | Entrepreneurship & industry ties | Systems emphasis |
| **Graduation requirement** | Research project | Thesis | Research or industry collaboration |

### CS Education at Major Japanese Universities

| University | Distinguishing Feature | Strong Areas | Introductory Course |
|-----------|----------------------|-------------|-------------------|
| University of Tokyo | Balance of theory and practice | Computation theory, AI | Introduction to Information Science |
| Kyoto University | Theory-oriented | Algorithms, mathematics | Introduction to Computer Science |
| Tokyo Institute of Technology | System implementation focus | OS, networks | Computer Science I |
| University of Tsukuba | Broad coverage across information sciences | HCI, media | Introduction to Information Science |
| University of Aizu | All courses taught in English | Embedded systems, compilers | Introduction to CS |

### Common Required Subjects

```
Fields required across all top CS programs:

  ■■■■■■■■■■ Algorithms and Data Structures (100% of programs)
  ■■■■■■■■■■ Discrete Mathematics / Mathematical Logic (100%)
  ■■■■■■■■■□ Computer Systems (90%)
  ■■■■■■■■■□ Operating Systems (90%)
  ■■■■■■■■□□ Programming Language Theory (80%)
  ■■■■■■■□□□ Theory of Computation (70%)
  ■■■■■■■□□□ Networks (70%)
  ■■■■■■□□□□ Databases (60%)
  ■■■■■□□□□□ AI / Machine Learning (50% — increasing in recent years)
  ■■■■□□□□□□ Software Engineering (40%)
```

→ **Algorithms/data structures and mathematics are required at every university.** This Skill fully covers these common foundations.

---

## 5. How CS Applies to Real-World Practice — 10 Concrete Scenarios

### Scenario 1: Slow API Responses
**CS knowledge**: Complexity analysis → Discover nested loop O(n²) → Improve to O(n) with hash maps

**Detailed improvement process**:
```python
# Step 1: Identify bottleneck through profiling
import cProfile

def slow_endpoint():
    users = get_all_users()          # 0.1s
    orders = get_all_orders()        # 0.2s
    # ↓ This consumes 99% of the time
    result = []
    for user in users:               # 10,000 users
        user_orders = [o for o in orders if o['user_id'] == user['id']]
        # ↑ Full scan of orders (100,000 records) every iteration → O(10,000 × 100,000) = O(10^9)
        result.append({**user, 'orders': user_orders})
    return result

# Step 2: Apply CS knowledge
def fast_endpoint():
    users = get_all_users()
    orders = get_all_orders()
    # Group using a hash map: O(n)
    orders_by_user = {}
    for order in orders:
        orders_by_user.setdefault(order['user_id'], []).append(order)
    # Join: O(m)
    return [{**user, 'orders': orders_by_user.get(user['id'], [])}
            for user in users]
    # Total: O(n + m) = O(110,000) — 9,000x faster
```

### Scenario 2: App Crashing Due to Out of Memory
**CS knowledge**: Understanding memory hierarchy and GC → Discover retained unnecessary references → Resolve with WeakRef

### Scenario 3: 0.1 + 0.2 ≠ 0.3 Causing Calculation Errors in Currency
**CS knowledge**: IEEE 754 floating-point representation → Use a decimal library (Decimal)

### Scenario 4: Extremely Slow Database Queries
**CS knowledge**: Understanding B+ tree indexes → Optimal composite index design

**Detailed improvement process**:
```sql
-- Before: Full table scan 30 seconds
SELECT * FROM orders
WHERE user_id = 12345
  AND status = 'completed'
  AND created_at > '2025-01-01'
ORDER BY created_at DESC
LIMIT 20;

-- CS knowledge: Understanding B+ tree structure
-- B+ trees are "sorted" data structures
-- → The column order in composite indexes matters
-- → Place equality conditions (=) first, range conditions (>) last

-- After: Add composite index 0.01 seconds
CREATE INDEX idx_orders_user_status_created
  ON orders(user_id, status, created_at DESC);

-- Why this order:
-- 1. user_id = 12345 → Narrow down with equality condition
-- 2. status = 'completed' → Further narrow down with equality condition
-- 3. created_at DESC → Range condition + sort (retrieved in index order)
-- → Completes with Index Scan Only (no table access required)
```

### Scenario 5: Mysterious Bugs in Multithreaded Code
**CS knowledge**: Race conditions and deadlocks → Unify lock ordering, leverage CAS

### Scenario 6: Unable to Follow System Design Discussions
**CS knowledge**: CAP theorem, consistency models → Understand the rationale behind architectural choices

### Scenario 7: Regular Expressions Not Working as Expected
**CS knowledge**: Automata theory → Understand the limits of regular languages, decide when to use a parser

### Scenario 8: Introducing Security Vulnerabilities
**CS knowledge**: Cryptography, input validation → Fundamentally prevent SQL injection/XSS

### Scenario 9: Unable to Make Appropriate Technology Choices
**CS knowledge**: Trade-off analysis → CAP theorem, ACID vs BASE, synchronous vs asynchronous

### Scenario 10: Unsure How to Leverage AI/LLMs
**CS knowledge**: Probability and statistics, Transformer → RAG design, prompt engineering optimization

---

## 6. The Mathematical Foundations of CS — A Complete Overview

CS requires certain mathematics to be deeply understood, but not everything is needed from the start:

```
Mathematics Required for CS (by stage):

  Stage 1 — Essential for CS fundamentals (high school math level)
  ├── Logic: AND, OR, NOT, implication, contrapositive
  ├── Set theory: Union, intersection, subsets
  ├── Exponents & logarithms: Understanding O(log n), O(2^n)
  └── Basic probability: Expected value, conditional probability

  Stage 2 — Intermediate CS (university year 1-2 level)
  ├── Discrete mathematics: Graph theory, combinatorics, induction
  ├── Linear algebra basics: Vectors, matrix multiplication
  ├── Probability & statistics: Distributions, estimation, hypothesis testing
  └── Basic number theory: Primes, modular arithmetic (needed for cryptography)

  Stage 3 — Advanced CS (university year 3-4 / graduate level)
  ├── Information theory: Entropy, mutual information
  ├── Optimization: Gradient descent, convex optimization
  ├── Calculus: Partial derivatives (needed for ML/AI)
  └── Abstract algebra: Group theory (theoretical foundation of cryptography)
```

```python
# Concrete examples of mathematics used in CS

import math

# Logarithms (fundamental to algorithm analysis)
# Number of steps to search 1 million items with binary search
n = 1_000_000
steps = math.ceil(math.log2(n))  # 20 steps
print(f"Binary search steps: {steps}")

# Combinatorics (calculating password strength)
# Number of combinations for an 8-character alphanumeric (62 chars) password
chars = 62
length = 8
combinations = chars ** length  # 218 trillion
time_to_crack = combinations / 10_000_000_000  # At 10 billion attempts per second
print(f"Brute force time: {time_to_crack / 3600:.0f} hours")

# Probability (calculating hash collisions — the Birthday Paradox)
# Probability that among n people, at least two share a birthday
def birthday_collision_probability(n, d=365):
    """Probability of collision among n items from d possible values"""
    prob_no_collision = 1.0
    for i in range(n):
        prob_no_collision *= (d - i) / d
    return 1 - prob_no_collision

print(f"Collision probability with 23 people: {birthday_collision_probability(23):.1%}")  # 50.7%
# → Even with a hash space of 2^128, there's a 50% collision probability at 2^64 items
```

---

## 7. Structure and Usage of This Skill

### Section Structure

```
computer-science-fundamentals/
├── docs/
│   ├── 00-introduction/     ← You are here (CS overview, history, learning path)
│   ├── 01-hardware-basics/  ← How hardware works (CPU, memory, GPU)
│   ├── 02-data-representation/ ← Internal data representation (binary, character encoding, floating-point)
│   ├── 03-algorithms-basics/  ← Algorithms (sorting, searching, DP, graphs)
│   ├── 04-data-structures/    ← Data structures (arrays, trees, hashing, graphs)
│   ├── 05-computation-theory/ ← Computation theory (automata, Turing machines)
│   ├── 06-programming-paradigms/ ← Paradigms (imperative, functional, OOP)
│   ├── 07-software-engineering-basics/ ← SE fundamentals (methodologies, testing, debugging)
│   └── 08-advanced-topics/    ← Advanced (distributed, concurrent, security, AI)
├── checklists/               ← Mastery checklists for each section
├── templates/                ← Exercise templates
└── references/               ← References and resource lists
```

### Detailed Contents of Each Section

| Section | Files | Key Topics | Estimated Study Time |
|---------|-------|------------|---------------------|
| 00-introduction | 4 | CS overview, history, motivation, roadmap | 4-6 hours |
| 01-hardware-basics | 4 | CPU, memory hierarchy, storage, GPU | 8-12 hours |
| 02-data-representation | 5 | Binary, integers, floating-point, character encoding | 10-15 hours |
| 03-algorithms-basics | 8 | Complexity, sorting, searching, recursion, DP, graphs | 20-30 hours |
| 04-data-structures | 8 | Arrays, lists, trees, hashing, graphs, heaps | 20-30 hours |
| 05-computation-theory | 6 | Automata, formal languages, Turing machines | 15-20 hours |
| 06-programming-paradigms | 5 | Imperative, OOP, functional, logic | 10-15 hours |
| 07-software-engineering | 5 | Methodologies, testing, debugging, version control | 10-15 hours |
| 08-advanced-topics | 10 | Distributed, concurrent, security, AI/ML | 25-35 hours |

### Recommended Study Order

```
Beginner (new to CS):
  00 → 02 → 03 → 04 → 01 → 05 → 06 → 07 → 08
  Rationale: Mathematical foundations → implementation → theory → applications

Intermediate (has programming experience):
  00 → 03 → 04 → 01 → 02 → 05 → 06 → 07 → 08
  Rationale: Algorithms → systems → theory, filling knowledge gaps

Advanced (knowledge consolidation):
  05 → 08 → any section of interest
  Rationale: Prioritize theory and cutting-edge topics, then deep-dive as needed
```

---

## 8. Hands-On Exercises

### Exercise 1: CS Field Mapping (Basic)

For code you've recently written (or a service you use), fill in the following table:

| Feature/Code | Related CS Field | Specific Concept |
|-------------|-----------------|------------------|
| Example: Login | Security + DB | Hash functions, session management |
| 1. | | |
| 2. | | |
| 3. | | |

### Exercise 2: Complexity Quiz (Applied)

Estimate the time complexity (Big-O) of each operation below and explain your reasoning:

1. JavaScript's `Array.push()` → ?
2. JavaScript's `Array.unshift()` → ?
3. Python's `dict[key]` → ?
4. SQL `SELECT * FROM users WHERE email = ?` (no index) → ?
5. SQL `SELECT * FROM users WHERE email = ?` (with index) → ?

<details>
<summary>Answers</summary>

1. `O(1)` — Amortized constant time. Appending to the end of a dynamic array.
2. `O(n)` — Every element must be shifted one position to the right.
3. `O(1)` — Hash table lookup. Worst case O(n) with collisions, but typically O(1).
4. `O(n)` — Full table scan. Every row is examined.
5. `O(log n)` — B+ tree index lookup.

</details>

### Exercise 3: Practicing Computational Thinking (Applied)

Apply the four elements of computational thinking (decomposition, pattern recognition, abstraction, algorithmic thinking) to the following everyday problems:

1. "Scheduling a meeting for 100 team members"
2. "Product recommendations for an e-commerce site"
3. "Identifying the root cause of errors from massive log files"

<details>
<summary>Sample Answer (Problem 1)</summary>

**Decomposition**: List candidate dates / Check each member's availability / Select the optimal date
**Pattern Recognition**: This is a type of "constraint satisfaction problem" — finding a solution that satisfies everyone's constraints
**Abstraction**: Individual schedule details are unnecessary. Abstract to an "available/busy" bitmap
**Algorithm**: Count available members for each candidate date and select the maximum (greedy approach)

```python
def find_best_date(members, candidate_dates):
    """Find the date when the most members are available"""
    best_date = None
    max_available = 0
    for date in candidate_dates:
        available = sum(1 for m in members if date in m.free_dates)
        if available > max_available:
            max_available = available
            best_date = date
    return best_date, max_available
```

</details>

### Exercise 4: CS Crossword (Advanced)

Identify the CS term matching each description:

1. The notation representing "the upper bound for the worst case" of complexity → ___
2. The architecture that "stores both programs (instructions) and data in the same memory" → ___
3. The law proposed by Gordon Moore in 1965 → ___
4. The property in the memory hierarchy that "recently accessed data is likely to be accessed again" → ___
5. The theorem stating "it is impossible to simultaneously satisfy consistency, availability, and partition tolerance" → ___

<details>
<summary>Answers</summary>

1. Big-O notation (O notation)
2. Von Neumann architecture (stored-program architecture)
3. Moore's Law
4. Temporal Locality
5. CAP theorem (Brewer's Theorem)

</details>


---

## Troubleshooting

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|---------|
| Initialization error | Configuration file issues | Verify the path and format of the configuration file |
| Timeout | Network latency / insufficient resources | Adjust timeout values, add retry logic |
| Out of memory | Growing data volume | Introduce batch processing, implement pagination |
| Permission error | Insufficient access rights | Verify execution user permissions, review settings |
| Data inconsistency | Concurrent processing conflicts | Introduce locking mechanisms, manage transactions |

### Debugging Steps

1. **Check the error message**: Read the stack trace to identify the point of failure
2. **Establish reproduction steps**: Reproduce the error with minimal code
3. **Formulate hypotheses**: List possible causes
4. **Incremental verification**: Use log output or a debugger to verify hypotheses
5. **Fix and regression test**: After fixing, run tests on related areas as well

```python
# Debugging utilities
import logging
import traceback
from functools import wraps

# Logger configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def debug_decorator(func):
    """Decorator that logs function input/output"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Calling: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Return value: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"Exception in: {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

@debug_decorator
def process_data(items):
    """Data processing (debug target)"""
    if not items:
        raise ValueError("Empty data")
    return [item * 2 for item in items]
```

### Diagnosing Performance Issues

Steps for diagnosing performance problems:

1. **Identify the bottleneck**: Measure with profiling tools
2. **Check memory usage**: Look for memory leaks
3. **Check I/O waits**: Assess disk and network I/O status
4. **Check concurrent connections**: Review the state of connection pools

| Problem Type | Diagnostic Tool | Solution |
|-------------|----------------|----------|
| CPU load | cProfile, py-spy | Algorithm improvement, parallelization |
| Memory leak | tracemalloc, objgraph | Proper reference cleanup |
| I/O bottleneck | strace, iostat | Async I/O, caching |
| DB latency | EXPLAIN, slow query log | Indexes, query optimization |

---

## Design Decision Guide

### Selection Criteria Matrix

The following summarizes the criteria for making technology choices.

| Criterion | When to prioritize | When to compromise |
|----------|-------------------|-------------------|
| Performance | Real-time processing, large-scale data | Admin panels, batch processing |
| Maintainability | Long-term operation, team development | Prototypes, short-term projects |
| Scalability | Services expected to grow | Internal tools, fixed user base |
| Security | Personal data, financial data | Public data, internal use |
| Development speed | MVP, time-to-market | Quality-critical, mission-critical |

### Architecture Pattern Selection

```
┌─────────────────────────────────────────────────┐
│          Architecture Selection Flow             │
├─────────────────────────────────────────────────┤
│                                                 │
│  (1) What is the team size?                     │
│    ├─ Small (1-5 people) → Monolith             │
│    └─ Large (10+ people) → Go to (2)            │
│                                                 │
│  (2) What is the deployment frequency?          │
│    ├─ Once a week or less → Monolith + modules  │
│    └─ Daily / multiple times → Go to (3)        │
│                                                 │
│  (3) How independent are the teams?             │
│    ├─ High → Microservices                      │
│    └─ Moderate → Modular monolith               │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Trade-off Analysis

Technical decisions always involve trade-offs. Analyze from the following perspectives:

**1. Short-term vs Long-term Cost**
- A method that is faster in the short term may become technical debt in the long term
- Conversely, over-engineering increases short-term costs and delays the project

**2. Consistency vs Flexibility**
- A unified tech stack has lower learning costs
- Adopting diverse technologies enables best-fit selection but increases operational costs

**3. Level of Abstraction**
- High abstraction offers great reusability but can make debugging more difficult
- Low abstraction is intuitive but tends to produce code duplication

```python
# Design decision recording template
class ArchitectureDecisionRecord:
    """Creating an ADR (Architecture Decision Record)"""

    def __init__(self, title: str):
        self.title = title
        self.context = ""
        self.decision = ""
        self.consequences = []
        self.alternatives = []

    def set_context(self, context: str):
        """Describe the background and problem"""
        self.context = context
        return self

    def set_decision(self, decision: str):
        """Describe the decision"""
        self.decision = decision
        return self

    def add_consequence(self, consequence: str, positive: bool = True):
        """Add a consequence"""
        self.consequences.append({
            'description': consequence,
            'type': 'positive' if positive else 'negative'
        })
        return self

    def add_alternative(self, name: str, reason_rejected: str):
        """Add a rejected alternative"""
        self.alternatives.append({
            'name': name,
            'reason_rejected': reason_rejected
        })
        return self

    def to_markdown(self) -> str:
        """Output in Markdown format"""
        md = f"# ADR: {self.title}\n\n"
        md += f"## Context\n{self.context}\n\n"
        md += f"## Decision\n{self.decision}\n\n"
        md += "## Consequences\n"
        for c in self.consequences:
            icon = "✅" if c['type'] == 'positive' else "⚠️"
            md += f"- {icon} {c['description']}\n"
        md += "\n## Rejected Alternatives\n"
        for a in self.alternatives:
            md += f"- **{a['name']}**: {a['reason_rejected']}\n"
        return md
```

---

## Real-World Application Scenarios

### Scenario 1: MVP Development at a Startup

**Situation:** Need to release a product quickly with limited resources

**Approach:**
- Choose a simple architecture
- Focus on the minimum viable set of features
- Automated tests only for the critical path
- Introduce monitoring from the start

**Lessons learned:**
- Don't strive for perfection (YAGNI principle)
- Obtain user feedback early
- Manage technical debt consciously

### Scenario 2: Modernizing a Legacy System

**Situation:** Incrementally overhauling a system that has been running for 10+ years

**Approach:**
- Migrate incrementally using the Strangler Fig pattern
- If existing tests are lacking, create Characterization Tests first
- Use an API gateway to allow old and new systems to coexist
- Perform data migration in stages

| Phase | Tasks | Estimated Duration | Risk |
|-------|------|--------------------|------|
| 1. Investigation | Current state analysis, dependency mapping | 2-4 weeks | Low |
| 2. Foundation | CI/CD setup, test environment | 4-6 weeks | Low |
| 3. Migration start | Migrate peripheral features first | 3-6 months | Medium |
| 4. Core migration | Migrate core functionality | 6-12 months | High |
| 5. Completion | Decommission legacy system | 2-4 weeks | Medium |

### Scenario 3: Development with a Large Team

**Situation:** 50+ engineers working on a single product

**Approach:**
- Clarify boundaries using Domain-Driven Design
- Assign ownership per team
- Manage shared libraries using the Inner Source model
- Design API-first to minimize inter-team dependencies

```python
# Defining API contracts between teams
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class APIContract:
    """API contract between teams"""
    endpoint: str
    method: str
    owner_team: str
    consumers: List[str]
    sla_ms: int  # Response time SLA
    priority: Priority

    def validate_sla(self, actual_ms: int) -> bool:
        """Verify SLA compliance"""
        return actual_ms <= self.sla_ms

    def to_openapi(self) -> dict:
        """Output in OpenAPI format"""
        return {
            'path': self.endpoint,
            'method': self.method,
            'x-owner': self.owner_team,
            'x-consumers': self.consumers,
            'x-sla-ms': self.sla_ms
        }

# Usage example
contracts = [
    APIContract(
        endpoint="/api/v1/users",
        method="GET",
        owner_team="user-team",
        consumers=["order-team", "notification-team"],
        sla_ms=200,
        priority=Priority.HIGH
    ),
    APIContract(
        endpoint="/api/v1/orders",
        method="POST",
        owner_team="order-team",
        consumers=["payment-team", "inventory-team"],
        sla_ms=500,
        priority=Priority.CRITICAL
    )
]
```

### Scenario 4: Performance-Critical Systems

**Situation:** A system requiring millisecond-level response times

**Optimization points:**
1. Cache strategy (L1: in-memory, L2: Redis, L3: CDN)
2. Leveraging asynchronous processing
3. Connection pooling
4. Query optimization and index design

| Optimization Method | Impact | Implementation Cost | Applicable Scenario |
|--------------------|--------|--------------------|--------------------|
| In-memory cache | High | Low | Frequently accessed data |
| CDN | High | Low | Static content |
| Async processing | Medium | Medium | I/O-heavy processing |
| DB optimization | High | High | When queries are slow |
| Code optimization | Low-Medium | High | CPU-bound cases |

---

## Leveraging in Team Development

### Code Review Checklist

Points to verify in code reviews related to this topic:

- [ ] Naming conventions are consistent
- [ ] Error handling is appropriate
- [ ] Test coverage is sufficient
- [ ] No negative performance impact
- [ ] No security issues
- [ ] Documentation is updated

### Best Practices for Knowledge Sharing

| Method | Frequency | Audience | Benefit |
|--------|-----------|----------|---------|
| Pair programming | As needed | Complex tasks | Immediate feedback |
| Tech talks | Weekly | Entire team | Horizontal knowledge transfer |
| ADR (Decision Records) | Per decision | Future members | Transparency of decision-making |
| Retrospectives | Biweekly | Entire team | Continuous improvement |
| Mob programming | Monthly | Major design decisions | Building consensus |

### Managing Technical Debt

```
Priority Matrix:

        Impact: High
          │
    ┌─────┼─────┐
    │ Plan│ Fix  │
    │ for │ imme-│
    │ it  │diately│
    ├─────┼─────┤
    │ Log │ Next │
    │ only│Sprint│
    │     │      │
    └─────┼─────┘
          │
        Impact: Low
  Frequency: Low  Frequency: High
```

---

## Security Considerations

### Common Vulnerabilities and Countermeasures

| Vulnerability | Risk Level | Countermeasure | Detection Method |
|--------------|-----------|---------------|-----------------|
| Injection attacks | High | Input validation, parameterized queries | SAST/DAST |
| Authentication flaws | High | Multi-factor authentication, strengthen session management | Penetration testing |
| Sensitive data exposure | High | Encryption, access control | Security audits |
| Misconfiguration | Medium | Security headers, principle of least privilege | Configuration scanning |
| Insufficient logging | Medium | Structured logging, audit trails | Log analysis |

### Secure Coding Best Practices

```python
# Secure coding examples
import hashlib
import secrets
import hmac
from typing import Optional

class SecurityUtils:
    """Security utilities"""

    @staticmethod
    def generate_token(length: int = 32) -> str:
        """Generate a cryptographically secure token"""
        return secrets.token_urlsafe(length)

    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> tuple:
        """Hash a password"""
        if salt is None:
            salt = secrets.token_hex(16)
        hashed = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            iterations=100000
        )
        return hashed.hex(), salt

    @staticmethod
    def verify_password(password: str, hashed: str, salt: str) -> bool:
        """Verify a password"""
        new_hash, _ = SecurityUtils.hash_password(password, salt)
        return hmac.compare_digest(new_hash, hashed)

    @staticmethod
    def sanitize_input(value: str) -> str:
        """Sanitize input"""
        dangerous_chars = ['<', '>', '"', "'", '&', '\\']
        result = value
        for char in dangerous_chars:
            result = result.replace(char, '')
        return result.strip()

# Usage example
token = SecurityUtils.generate_token()
hashed, salt = SecurityUtils.hash_password("my_password")
is_valid = SecurityUtils.verify_password("my_password", hashed, salt)
```

### Security Checklist

- [ ] All input values are validated
- [ ] Sensitive information is not output to logs
- [ ] HTTPS is enforced
- [ ] CORS policy is properly configured
- [ ] Dependency vulnerability scanning has been performed
- [ ] Error messages do not contain internal information

---

## Migration Guide

### Notes on Version Upgrades

| Version | Major Changes | Migration Tasks | Scope of Impact |
|---------|-------------|----------------|----------------|
| v1.x → v2.x | API design overhaul | Endpoint changes | All clients |
| v2.x → v3.x | Authentication method change | Token format update | Authentication-related |
| v3.x → v4.x | Data model change | Run migration scripts | DB-related |

### Step-by-Step Migration Procedure

```python
# Migration script template
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Callable

logger = logging.getLogger(__name__)

class MigrationRunner:
    """Step-by-step migration execution engine"""

    def __init__(self, migration_dir: str):
        self.migration_dir = Path(migration_dir)
        self.migrations: List[Dict] = []
        self.completed: List[str] = []

    def register(self, version: str, description: str,
                 up: Callable, down: Callable):
        """Register a migration"""
        self.migrations.append({
            'version': version,
            'description': description,
            'up': up,
            'down': down,
            'registered_at': datetime.now().isoformat()
        })

    def run_up(self, target_version: str = None):
        """Run migrations (upgrade)"""
        for migration in self.migrations:
            if migration['version'] in self.completed:
                continue
            logger.info(f"Running: {migration['version']} - "
                       f"{migration['description']}")
            try:
                migration['up']()
                self.completed.append(migration['version'])
                logger.info(f"Completed: {migration['version']}")
            except Exception as e:
                logger.error(f"Failed: {migration['version']}: {e}")
                raise
            if target_version and migration['version'] == target_version:
                break

    def run_down(self, target_version: str):
        """Rollback migrations"""
        for migration in reversed(self.migrations):
            if migration['version'] not in self.completed:
                continue
            if migration['version'] == target_version:
                break
            logger.info(f"Rolling back: {migration['version']}")
            migration['down']()
            self.completed.remove(migration['version'])

    def status(self) -> Dict:
        """Check migration status"""
        return {
            'total': len(self.migrations),
            'completed': len(self.completed),
            'pending': len(self.migrations) - len(self.completed),
            'versions': {
                m['version']: 'completed'
                if m['version'] in self.completed else 'pending'
                for m in self.migrations
            }
        }
```

### Rollback Plan

Always prepare a rollback plan for migration work:

1. **Data backup**: Take a complete backup before migration
2. **Verification in test environment**: Verify in advance in an environment equivalent to production
3. **Gradual rollout**: Roll out incrementally using canary releases
4. **Enhanced monitoring**: Shorten monitoring intervals during migration
5. **Clear decision criteria**: Define criteria for triggering a rollback in advance
---

## FAQ

### Q1: Is CS impossible without being good at math?

**A**: The vast majority of CS fundamentals can be understood with high school-level math. The mathematics primarily needed are:
- **Discrete mathematics**: Logic (AND/OR/NOT), sets, graphs
- **Basic algebra**: Equations, exponents & logarithms (to understand O(log n))
- **Basic probability & statistics**: If you proceed to AI/ML

Calculus and linear algebra become necessary for advanced topics (AI/ML, computer graphics), but they are not required to begin studying CS fundamentals.

### Q2: Do I need programming experience to learn CS fundamentals?

**A**: It is not required, but having it dramatically accelerates understanding. This Skill uses code examples extensively, so having basics in Python or JavaScript is most effective. If you have no coding experience, we recommend starting with "02-data-representation" (it contains much content that can be understood without programming).

### Q3: Are there really situations where CS is used in practice?

**A**: You are using it every day, even if you don't realize it. Here are typical examples:
- Choosing between `Array` and `Set` → Knowledge of data structures
- Caring about API response time → Knowledge of complexity
- Using `async/await` → Knowledge of concurrency
- Hashing passwords → Knowledge of cryptography
- Merging branches in Git → Knowledge of graph theory

You can write code without CS fundamentals, but **it won't scale**. When your user base grows from 100 to 1,000,000, the presence or absence of CS fundamentals makes a critical difference.

### Q4: Can I fully learn CS with this Skill alone?

**A**: This Skill provides "an entry point and a comprehensive overview of CS fundamentals." To delve deeper into each field, refer to the advanced Skills.

### Q5: Will doing LeetCode teach me CS fundamentals?

**A**: LeetCode is "algorithm pattern practice" and covers only a portion of CS fundamentals. There is also the risk of falling into pattern memorization. If you study CS fundamentals systematically and then tackle LeetCode, you can solve problems while understanding *why* each algorithm is correct, dramatically increasing effectiveness.

### Q6: Does age matter for learning CS?

**A**: Not at all. CS foundational concepts depend on logical reasoning ability, which is unrelated to age. In fact, having real-world experience makes it easier to appreciate *why* each concept is important, deepening your understanding. Linus Torvalds, the leader of the Linux kernel, continues to work at the forefront in his 50s, and CS foundational knowledge retains its value throughout your entire career.

### Q7: Is CS knowledge still necessary in the AI era?

**A**: The more AI advances, the more important CS fundamentals become:

1. **Ability to evaluate AI output**: CS knowledge is essential to judge whether code generated by LLMs has appropriate complexity or security issues
2. **Ability to use AI correctly**: CS knowledge is needed for RAG design, embedding selection, and understanding token limits
3. **Abilities AI cannot replace**: System-wide architecture design and distributed systems trade-off decisions remain human work
4. **Ability to understand AI's limitations**: Understanding the undecidability of the Halting Problem reveals that AI also has fundamental limitations

---

## Summary

| Concept | Key Point |
|---------|-----------|
| Definition of CS | A discipline exploring "what computation is." Programming is merely one aspect |
| 10 major fields | Theory, algorithms, data structures, OS, networks, DB, AI, SE, security, HCI |
| Value of CS | Frameworks change, but CS fundamentals have remained universal for 50+ years |
| Computational thinking | Four elements: decomposition, pattern recognition, abstraction, algorithmic thinking |
| Learning approach | Alternate between theory and practice, deepen gradually |
| Mathematical foundations | You can start from high school math level. Acquire necessary math incrementally |
| Scope of this Skill | CS fundamentals from entry to intermediate. Advanced topics lead to specialized Skills |

---

## Recommended Next Readings


---

## References

1. ACM/IEEE. "Computing Curricula 2020: Paradigms for Global Computing Education." ACM, 2020.
2. MIT OpenCourseWare. "6.0001 Introduction to CS and Programming Using Python." https://ocw.mit.edu/
3. Sipser, M. "Introduction to the Theory of Computation." 3rd Edition, Cengage, 2012.
4. Cormen, T. H. et al. "Introduction to Algorithms (CLRS)." 4th Edition, MIT Press, 2022.
5. Abelson, H. & Sussman, G. J. "Structure and Interpretation of Computer Programs (SICP)." 2nd Edition, MIT Press, 1996.
6. Wing, J. M. "Computational Thinking." Communications of the ACM, Vol. 49, No. 3, 2006.
7. Denning, P. J. "The Profession of IT: Beyond Computational Thinking." Communications of the ACM, 2009.
8. Knuth, D. E. "Computer Science and its Relation to Mathematics." The American Mathematical Monthly, 1974.
9. Dijkstra, E. W. "On the cruelty of really teaching computing science." EWD1036, 1988.
10. Patterson, D. A. & Hennessy, J. L. "Computer Organization and Design." 6th Edition, Morgan Kaufmann, 2020.
11. Feynman, R. P. "Feynman Lectures on Computation." CRC Press, 1996.
12. Sedgewick, R. & Wayne, K. "Algorithms." 4th Edition, Addison-Wesley, 2011.
