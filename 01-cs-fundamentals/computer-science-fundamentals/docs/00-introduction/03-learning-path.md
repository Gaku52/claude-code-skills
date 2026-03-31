# CS Learning Path -- Roadmap, Recommended Order, and Interdisciplinary Relationships

> The shortest path is not "learning everything" but "grasping the essentials in the right order."
> -- The most important discovery in computer science education is not the content itself, but the order in which it should be learned.

## Learning Objectives

- [ ] Select a learning path suited to your level and goals
- [ ] Understand the dependency relationships between CS disciplines and design an efficient learning order
- [ ] Understand the characteristics and usage of learning resources
- [ ] Practice learning techniques grounded in scientific evidence
- [ ] Understand how to effectively use this Skill

## Prerequisites

- None (this guide serves as the blueprint for your learning)

---

## 1. Why a Learning Path Matters

### 1.1 The Pitfalls of Self-Study

Many people who attempt to self-study computer science fall into one of the following patterns.

```
Three common traps for self-learners:

Trap 1: Scattershot Learning
  +-------+  +-------+  +-------+  +-------+
  |  AI   |  | Web   |  |Crypto |  |  DB   |   <- Jumping around randomly
  |       |  |       |  |       |  |       |
  +---+---+  +---+---+  +---+---+  +---+---+
      |          |          |          |
      v          v          v          v
  Scattered shallow knowledge -> Understanding nothing deeply

Trap 2: Perfectionist Learning
  +---------------------------------------+
  | Spending months on a single topic     |
  | "Must perfect binary first..."        |
  | -> No big picture, motivation drops   |
  +---------------------------------------+

Trap 3: Ignoring Dependencies
  Starting with distributed systems
  -> Can't understand CAP
  -> No networking fundamentals
  -> Process concept itself is vague
  -> Give up
```

The root cause common to all these traps is **the absence of a designed learning path (what to learn and in what order)**.

### 1.2 Learning from University Curricula

Comparing CS curricula at the world's top universities (MIT, Stanford, CMU, University of Tokyo) reveals a remarkably common structure.

```
Comparison of core CS curricula at major universities:

                MIT       Stanford    CMU      U of Tokyo
Year 1, S1 | 6.001     | CS106A    | 15-112  | CS Fundamentals
           | (SICP)    | (Java)    | (Python)| (Discrete Math)
           |           |           |         |
Year 1, S2 | 6.004     | CS106B    | 15-122  | Algorithms
           | (Comp.    | (Data     | (C/DS)  | and Data
           |  Struct.) | Struct.)  |         | Structures
           |           |           |         |
Year 2, S1 | 6.006     | CS107     | 15-213  | Computer
           | (Algo-    | (Systems) | (CS:APP)| Architecture
           |  rithms)  |           |         |
Year 2, S2 | 6.046     | CS110     | 15-210  | OS
           | (Algo     | (Concur-  | (Para-  | Networking
           |  Design)  |  rency)   |  llel   |
           |           |           |  DS)    |
Year 3+    | Electives | Electives | Electives| Electives
           | (AI, DB,  | (AI, NLP, | (AI, PL,| (AI, DB,
           |  NW, PL)  |  DB, HCI) |  SE)    |  Dist.)

Common pattern:
  Programming fundamentals -> Data structures -> Algorithms
  -> Systems (HW/OS) -> Applied fields
```

This common pattern is no coincidence. Clear **prerequisite dependencies** exist between fields, and an order that reflects these dependencies is the most efficient.

### 1.3 Prerequisite Dependency Graph

CS fields have clear prerequisite dependencies. Ignoring them imposes unnecessary difficulty on the learner.

```
Dependency graph of CS fields (overview):

  +-------------------------------------------------------------------+
  |                                                                   |
  |   +----------+                                                    |
  |   | Binary   |--+                                                 |
  |   | Logic    |  |                                                 |
  |   +----------+  |    +--------------+    +----------------+       |
  |                  +--->| Hardware     |--->| OS             |       |
  |   +----------+  |    | (CPU, Memory)|    | (Processes,    |       |
  |   | Basic    |--+    +--------------+    | Memory Mgmt)   |       |
  |   | Program- |                           +------+---------+       |
  |   | ming     |                                  |                 |
  |   +----+-----+                                  v                 |
  |        |         +--------------+    +----------------+           |
  |        +-------->| Algorithms   |--->| Databases      |           |
  |        |         | + Data       |    | (Indexes,      |           |
  |        |         | Structures   |    | Transactions)  |           |
  |        |         +------+-------+    +----------------+           |
  |        |                |                                         |
  |        |                v                                         |
  |        |         +--------------+    +----------------+           |
  |        +-------->| Theory of    |    | Security       |           |
  |        |         | Computation  |--->| (Crypto, Auth) |           |
  |        |         | (Automata,   |    +----------------+           |
  |        |         |  Turing)     |                                 |
  |        |         +--------------+                                 |
  |        |                                                          |
  |        |         +--------------+    +----------------+           |
  |        +-------->| Networking   |--->| Distributed    |           |
  |                  | (TCP/IP,     |    | Systems (CAP,  |           |
  |                  |  HTTP)       |    | Consensus,     |           |
  |                  +--------------+    | Microservices) |           |
  |                                      +----------------+           |
  |                                                                   |
  +-------------------------------------------------------------------+

  Legend: A --> B means "learning A before B is more efficient"
```

This dependency can be expressed programmatically as follows.

```python
# Representing CS field dependencies as a DAG (Directed Acyclic Graph)
from collections import defaultdict, deque

class CSCurriculum:
    """A class to manage CS learning curriculum dependencies"""

    def __init__(self):
        self.prerequisites = defaultdict(list)  # prerequisites
        self.topics = {}

    def add_topic(self, topic_id, name, estimated_hours):
        self.topics[topic_id] = {
            "name": name,
            "hours": estimated_hours
        }

    def add_prerequisite(self, topic, prerequisite):
        """prerequisite is required before learning topic"""
        self.prerequisites[topic].append(prerequisite)

    def topological_sort(self):
        """Determine the optimal learning order via topological sort"""
        in_degree = defaultdict(int)
        graph = defaultdict(list)

        all_topics = set(self.topics.keys())
        for topic, prereqs in self.prerequisites.items():
            for prereq in prereqs:
                graph[prereq].append(topic)
                in_degree[topic] += 1

        # Start with topics that have no prerequisites
        queue = deque([t for t in all_topics if in_degree[t] == 0])
        order = []

        while queue:
            current = queue.popleft()
            order.append(current)
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return order

    def get_learning_path(self, target_topics):
        """Compute the minimal path to learn the specified set of topics"""
        required = set()

        def collect_prereqs(topic):
            if topic in required:
                return
            required.add(topic)
            for prereq in self.prerequisites[topic]:
                collect_prereqs(prereq)

        for topic in target_topics:
            collect_prereqs(topic)

        # Topological sort over only the required topics
        full_order = self.topological_sort()
        return [t for t in full_order if t in required]


# Curriculum definition
curriculum = CSCurriculum()

# Register topics (ID, name, estimated hours)
curriculum.add_topic("binary",        "Binary & Logic",         20)
curriculum.add_topic("programming",   "Programming Basics",     80)
curriculum.add_topic("hardware",      "Hardware",               40)
curriculum.add_topic("algorithms",    "Algorithms",             80)
curriculum.add_topic("data_struct",   "Data Structures",        60)
curriculum.add_topic("os",            "OS",                     50)
curriculum.add_topic("network",       "Networking",             40)
curriculum.add_topic("database",      "Databases",              40)
curriculum.add_topic("computation",   "Theory of Computation",  40)
curriculum.add_topic("security",      "Security",               30)
curriculum.add_topic("distributed",   "Distributed Systems",    50)

# Register dependencies
curriculum.add_prerequisite("hardware",    "binary")
curriculum.add_prerequisite("hardware",    "programming")
curriculum.add_prerequisite("os",          "hardware")
curriculum.add_prerequisite("algorithms",  "programming")
curriculum.add_prerequisite("data_struct", "algorithms")
curriculum.add_prerequisite("database",    "data_struct")
curriculum.add_prerequisite("computation", "algorithms")
curriculum.add_prerequisite("security",    "computation")
curriculum.add_prerequisite("network",     "programming")
curriculum.add_prerequisite("distributed", "network")
curriculum.add_prerequisite("distributed", "os")

# Compute the minimal path for a web developer
web_targets = ["algorithms", "data_struct", "network", "database", "security"]
path = curriculum.get_learning_path(web_targets)
print("Learning order for web developers:")
for i, topic_id in enumerate(path, 1):
    t = curriculum.topics[topic_id]
    print(f"  {i}. {t['name']} ({t['hours']} hours)")

# Example output:
# Learning order for web developers:
#   1. Binary & Logic (20 hours)
#   2. Programming Basics (80 hours)
#   3. Hardware (40 hours)
#   4. Algorithms (80 hours)
#   5. Data Structures (60 hours)
#   6. Networking (40 hours)
#   7. Theory of Computation (40 hours)
#   8. Databases (40 hours)
#   9. Security (30 hours)
```

This code itself uses important CS concepts (DAGs, topological sort, DFS). As your CS study progresses, you will naturally understand the meaning and efficiency of such code.

---

## 2. Five Principles of Learning Path Design

### Principle 1: Alternate Between Theory and Practice

After 30 minutes of theory, always write code to verify your understanding. "Knowing" and "doing" are entirely different things; academic research also shows that Active Learning yields more than twice the retention rate of passive learning.

```
The golden ratio of theory and practice:

  +-----------------------------------------------------+
  |                                                       |
  |   Theory (30%)      Practice (50%)   Review (20%)    |
  |   +-----------+  +-----------------+  +----------+   |
  |   | Textbooks |  | Code            |  | Anki     |   |
  |   | Lectures  |  | LeetCode        |  | Blog     |   |
  |   | This Skill|  | Personal project|  | Explain  |   |
  |   +-----------+  +-----------------+  +----------+   |
  |                                                       |
  |   Example: Learning hash tables (2-hour session)     |
  |   [0:00-0:30] Theory: Learn the mechanism and        |
  |               complexity                              |
  |   [0:30-0:45] Practice 1: Simple hash map with array |
  |   [0:45-1:15] Practice 2: Implement collision        |
  |               resolution                              |
  |   [1:15-1:40] Practice 3: Solve LeetCode "Two Sum"  |
  |   [1:40-2:00] Review: Organize notes on what you     |
  |               learned                                 |
  |                                                       |
  +-----------------------------------------------------+
```

### Principle 2: Move On at 80% Understanding (Spiral Learning)

Perfectionism kills learning. Once you understand 80%, move on to the next topic and return later for deeper study. This approach is known as "Spiral Learning," a concept proposed by education theorist Jerome Bruner.

```
Spiral learning model:

  Understanding
  100% |                                    *--- 3rd pass
      |                              *---*
  80% |                        *---*           <- Move on here
      |                  *---*
  60% |            *---*
      |      *---*                      *--- 2nd pass
  40% |*---*                       *---*
      |                       *---*
  20% |                  *---*              *--- 1st pass
      |            *---*
   0% |------+------+------+------+------
             M1     M2     M3     M4

  1st pass: Broad and shallow overview -> 2nd pass: Focus on weak areas -> 3rd pass: Deep understanding
```

### Principle 3: Connect to Real-World Practice

Always consider where the concepts you learn are used in your own projects.

```python
# Practical examples of Principle 3: Finding CS concepts in everyday code

# --- Example 1: Python dictionary operations ---
user = {"name": "Alice", "age": 30}
user["email"] = "alice@example.com"  # O(1) -- hash table

# Question: Why is dictionary access O(1)?
# -> Hash table internals (docs/04-data-structures/02)

# --- Example 2: List search ---
numbers = [3, 1, 4, 1, 5, 9, 2, 6]
if 5 in numbers:       # O(n) -- linear search
    print("found!")

sorted_numbers = sorted(numbers)  # O(n log n) -- TimSort
# -> Sorting algorithms (docs/03-algorithms-basics/02)

# --- Example 3: Web framework routing ---
# Flask routing
# @app.route("/users/<int:user_id>")
# -> URL parsing uses regex/finite automata (docs/05-computation-theory/01)
# -> Request handling uses processes/threads (OS concepts)

# --- Example 4: Database queries ---
# SELECT * FROM users WHERE email = 'alice@example.com';
# -> Without an index: O(n) full table scan
# -> With a B-tree index: O(log n)
# -> With a hash index: O(1)
# -> Data structure choice determines performance
```

### Principle 4: Learn Through Output

Input-only learning has low retention. Based on the "Learning Pyramid" below, prioritize output.

| Learning Method | Avg. Retention Rate | Category |
|----------------|--------------------| ---------|
| Lecture | 5% | Passive |
| Reading | 10% | Passive |
| Audiovisual | 20% | Passive |
| Demonstration | 30% | Passive |
| Discussion | 50% | Active |
| Practice by Doing | 75% | Active |
| Teaching Others | 90% | Active |

Concrete examples of output include writing blog posts about algorithms you learned, presenting at study groups, contributing to OSS projects, and writing LeetCode solutions with detailed comments.

### Principle 5: Develop Metacognition

The ability to accurately gauge "what you understand and what you don't" (metacognition) is the key to efficient learning.

```
Four stages of metacognition:

  +------------------------------------------------------------+
  |                                                            |
  |  Stage 1: Unconscious Incompetence                        |
  |  "You don't know what you don't know"                     |
  |  Example: Not even aware hash tables exist                |
  |       |                                                    |
  |  Stage 2: Conscious Incompetence                          |
  |  "You know what you don't know"                           |
  |  Example: Realizing you can't explain how hash tables work|
  |       |                                                    |
  |  Stage 3: Conscious Competence                            |
  |  "You can do it with effort"                              |
  |  Example: Implementing a hash table while thinking through|
  |           each step                                        |
  |       |                                                    |
  |  Stage 4: Unconscious Competence                          |
  |  "You can do it naturally without thinking"               |
  |  Example: Instantly recognizing a hash map is optimal for |
  |           a given problem                                  |
  |                                                            |
  +------------------------------------------------------------+
```

---

## 3. Level-Based Learning Paths

### 3.1 Level Assessment Checklist

First, assess your current level. You can determine it by how many items you can answer "Yes" to.

```
Level assessment checklist (answer Yes/No for each item):

[Fundamentals Level] -- 4 or more Yes out of 6 means fundamentals are OK
  [ ] Can write a program using variables, conditionals, and loops
  [ ] Can define functions and understand parameters and return values
  [ ] Can distinguish between arrays (lists) and dictionaries (maps)
  [ ] Can read and write files programmatically
  [ ] Can use the command line (terminal) at a basic level
  [ ] Can perform basic Git operations (commit, push, pull)

[Intermediate Level] -- 5 or more Yes out of 8 means intermediate
  [ ] Can explain the difference between O(n) and O(n^2) with examples
  [ ] Can explain the difference between stacks and queues and when to use each
  [ ] Comfortable writing recursive functions
  [ ] Can implement binary search from scratch
  [ ] Can explain the difference between TCP and UDP
  [ ] Can explain the difference between processes and threads
  [ ] Can write basic SQL JOINs
  [ ] Know the meaning of HTTP status codes (200, 404, 500)

[Advanced Level] -- 5 or more Yes out of 8 means advanced
  [ ] Can solve dynamic programming problems independently
  [ ] Can explain the structure of a B-tree or red-black tree
  [ ] Can explain the CAP theorem with specific database examples
  [ ] Can explain how TLS/SSL works (handshake)
  [ ] Can explain two or more garbage collection mechanisms
  [ ] Can explain the four conditions for deadlock
  [ ] Can explain the relationship between P, NP, and NP-complete
  [ ] Can explain the overview of Raft or Paxos

Assessment results:
  Fundamentals not met -> Beginner path (Section 3.2)
  Fundamentals OK, intermediate not met -> Intermediate path (Section 3.3)
  Intermediate OK, advanced not met -> Advanced path (Section 3.4)
  Advanced OK -> Expert path (Section 3.5)
```

### 3.2 Beginner Path (CS Newcomer) -- 6-Month Plan

Target: Little programming experience, no systematic CS study

**Goal**: Acquire an overview of CS fundamentals and become capable of tackling intermediate-level problems

```
Beginner path -- 6-month plan:

Month 1: Foundation of Foundations [Data Representation]
  +-- Week 1-2: Binary and data representation (docs/02-data-representation/00-01)
  |    - Binary, hexadecimal conversion
  |    - Bitwise operations (AND, OR, XOR, shift)
  +-- Week 3: Character encoding and integer representation
  |    (docs/02-data-representation/01-02)
  |    - ASCII, Unicode, UTF-8
  |    - Signed integers (two's complement)
  +-- Week 4: Floating-point numbers (docs/02-data-representation/03)
       - IEEE 754
       - Why 0.1 + 0.2 !== 0.3
  [Milestone] Can accurately explain the 0.1 + 0.2 problem
  [Practice] 10 decimal-binary conversion problems, 5 bitwise operation problems

Month 2: Understanding Hardware [How Computers Work]
  +-- Week 1-2: CPU basics (docs/01-hardware-basics/00)
  |    - Fetch-decode-execute cycle
  |    - Registers, ALU, control unit
  +-- Week 3: Memory hierarchy (docs/01-hardware-basics/01)
  |    - Register -> L1 -> L2 -> L3 -> RAM -> SSD -> HDD
  |    - Cache locality (temporal and spatial)
  +-- Week 4: Storage and I/O (docs/01-hardware-basics/02)
       - Types and characteristics of storage
       - How I/O works
  [Milestone] Can explain how a program is executed
  [Practice] Draw a diagram of the "Hello World" execution flow

Month 3-4: Introduction to Algorithms [Problem-Solving Toolkit]
  +-- Week 1: What are algorithms (docs/03-algorithms-basics/00)
  |    - Definition and importance of algorithms
  |    - How to read pseudocode
  +-- Week 2-3: Complexity and sorting (docs/03-algorithms-basics/01-02)
  |    - Big-O notation (O(1), O(log n), O(n), O(n log n), O(n^2))
  |    - Bubble sort, selection sort, merge sort
  +-- Week 4-5: Search and recursion (docs/03-algorithms-basics/03-04)
  |    - Linear search, binary search
  |    - Recursive thinking, call stack
  +-- Week 6: Divide and conquer (docs/03-algorithms-basics/05)
  |    - Merge sort in detail
  |    - The divide and conquer approach
  +-- Week 7: Introduction to dynamic programming (docs/03-algorithms-basics/06)
  |    - Memoized recursion
  |    - Fibonacci sequence, knapsack problem
  +-- Week 8: Graph basics (docs/03-algorithms-basics/07)
       - Graph representations (adjacency list, adjacency matrix)
       - BFS, DFS
  [Milestone] Can judge fast/slow from Big-O notation
  [Practice] Implement each sorting algorithm from scratch, solve 10 LeetCode Easy problems

Month 5: Data Structures [Organizing Information]
  +-- Week 1: Arrays and lists (docs/04-data-structures/00)
  |    - Static and dynamic arrays
  |    - Linked lists (singly, doubly)
  +-- Week 2: Hash tables (docs/04-data-structures/02)
  |    - Hash functions
  |    - Collision resolution (chaining, open addressing)
  +-- Week 3: Tree structures (docs/04-data-structures/03)
  |    - Binary search trees
  |    - Heaps
  +-- Week 4: Practical exercises
       - Solve 20 LeetCode Easy problems intensively
       - Internalize data structure selection
  [Milestone] Can choose the appropriate data structure for a given problem
  [Practice] At least 30 LeetCode Easy problems total

Month 6: Integration and Application [Completing the Big Picture]
  +-- Week 1-2: Programming paradigms (docs/06)
  |    - Procedural, object-oriented, functional
  |    - Strengths and weaknesses of each paradigm
  +-- Week 3: Software engineering basics (docs/07)
  |    - Version control, testing, design principles
  +-- Week 4: Review and strengthen weak areas
       - Self-assess understanding of all topics on a 5-point scale
       - Review weak areas intensively
  [Milestone] Have a comprehensive overview of CS fundamentals
  [Final Test] Score 5+ Yes on the intermediate level assessment checklist
```

### 3.3 Intermediate Path (3+ Years of Programming Experience) -- 6-Month Plan

Target: Has practical experience but feels uncertain about CS fundamentals

**Goal**: Able to directly apply CS fundamentals to practice and handle technical interviews

```
Intermediate path -- 6-month plan:

Month 1: Identify Weaknesses and Solidify Foundations [Assessment]
  +-- Week 1: Self-diagnostic test (run the level assessment checklist)
  |    - Identify weak areas
  |    - Customize a 6-month learning plan
  +-- Week 2: Review complexity analysis (docs/03-algorithms-basics/01)
  |    - Amortized complexity
  |    - Worst-case vs. average-case analysis
  +-- Week 3: Deep dive into hash tables (docs/04-data-structures/02)
  |    - Load factor and rehashing
  |    - Read the internal implementation of your language's hash table
  +-- Week 4: Tree structures and balanced trees (docs/04-data-structures/03-04)
       - AVL tree and red-black tree rotation operations
       - Relationship between B-trees and database indexes
  [Milestone] Can instantly answer the time complexity of all data structures

Month 2: Understanding Systems [Computer Internals]
  +-- Week 1: CPU + Memory (docs/01-hardware-basics/00-01)
  |    - Pipeline, branch prediction
  |    - Cache lines, false sharing
  +-- Week 2: OS basics (processes, memory management)
  |    - Virtual memory, page tables
  |    - Context switches
  +-- Week 3: Networking basics (TCP/IP, HTTP)
  |    - 3-way handshake
  |    - HTTP/1.1 vs HTTP/2 vs HTTP/3
  +-- Week 4: Introduction to concurrency (docs/08-advanced-topics/01)
       - mutex, semaphore, monitor
       - Understanding async/await internals
  [Milestone] Can explain async/await internals

Month 3-4: Algorithm Strengthening [Improving Problem-Solving]
  +-- Week 1-2: Mastering dynamic programming (docs/03-algorithms-basics/06)
  |    - State design patterns
  |    - Top-down vs. bottom-up
  |    - Interval DP, tree DP, bitmask DP
  +-- Week 3-4: Graph algorithms (docs/03-algorithms-basics/07)
  |    - Shortest paths (Dijkstra, Bellman-Ford)
  |    - Minimum spanning trees (Kruskal, Prim)
  |    - Topological sort
  +-- Week 5-6: 50 LeetCode Medium problems
  |    - Arrays/strings: 10 problems
  |    - Trees/graphs: 10 problems
  |    - DP: 10 problems
  |    - Stacks/queues: 10 problems
  |    - Other: 10 problems
  +-- Week 7-8: System design basics
       - Design a URL shortening service
       - Design a chat application
  [Milestone] Can solve coding problems in technical interviews

Month 5: Theory and Design [Deep Understanding]
  +-- Week 1-2: Theory of computation (docs/05-computation-theory/00-04)
  |    - Regular languages and finite automata
  |    - Context-free grammars and pushdown automata
  |    - Turing machines and computability
  |    - The P vs NP problem
  +-- Week 3: Distributed systems (docs/08-advanced-topics/00)
  |    - Deep understanding of the CAP theorem
  |    - Distributed consensus (overview of Raft)
  |    - Eventual consistency vs. strong consistency
  +-- Week 4: Security basics (docs/08-advanced-topics/02)
       - Symmetric and asymmetric encryption
       - Hash functions (SHA-256)
       - Digital signatures
  [Milestone] Can explain the CAP theorem and the P vs NP problem

Month 6: Practical Integration [Applying Your Learning]
  +-- Week 1-2: Optimize your own projects
  |    - Conduct profiling
  |    - Identify and resolve bottlenecks using CS knowledge
  +-- Week 3: Mock interviews (system design)
  |    - Twitter timeline design
  |    - Rate limiter design
  +-- Week 4: Comprehensive review and plan next steps
       - Reflect on the entire journey
       - Transition to the advanced path
  [Milestone] Can directly apply CS fundamentals to practical work
```

### 3.4 Advanced Path (CS Knowledge Inventory) -- 3-Month Plan

Target: Has a CS degree or 5+ years of practical experience

**Goal**: Gain deep theoretical understanding and the ability to transfer knowledge to teams

```
Advanced path -- 3-month plan:

Month 1: Deep Theory [Rebuilding the Theoretical Foundation]
  +-- Week 1: All sections on theory of computation (docs/05)
  |    - Church-Turing thesis
  |    - Halting problem and diagonal argument
  |    - Decidability and semi-decidability
  +-- Week 2: Information theory (docs/05-computation-theory/05)
  |    - Shannon entropy
  |    - Information content and optimal coding
  |    - Kolmogorov complexity
  +-- Week 3: Advanced data structures (docs/04-data-structures/07)
  |    - Skip lists
  |    - Bloom filters
  |    - Tries and suffix trees
  +-- Week 4: Advanced algorithms
       - Approximation algorithms
       - Randomized algorithms
       - String algorithms (KMP, Rabin-Karp)
  [Milestone] Can reproduce the proof of the halting problem

Month 2: Deep Systems [Implementation-Level Understanding]
  +-- Week 1: Distributed systems (CAP, Raft, CRDT)
  |    - Raft leader election and log replication
  |    - CRDTs (Conflict-free Replicated Data Types)
  |    - Lamport clocks and vector clocks
  +-- Week 2: Concurrency (lock-free, CAS, memory models)
  |    - Compare-And-Swap (CAS) operations
  |    - Lock-free data structures
  |    - Java Memory Model / C++ Memory Model
  +-- Week 3: Performance optimization
  |    - CPU cache-aware data structures
  |    - Using SIMD instructions
  |    - How memory allocators work
  +-- Week 4: Database internals
       - WAL (Write-Ahead Logging)
       - MVCC (Multi-Version Concurrency Control)
       - How query optimizers work
  [Milestone] Can explain distributed consensus algorithms

Month 3: Cutting-Edge Topics and Education [Knowledge Synthesis and Sharing]
  +-- Week 1: AI/ML fundamentals
  |    - Mathematical foundations of neural networks
  |    - Transformer architecture
  |    - How LLMs work
  +-- Week 2: Quantum computing
  |    - Qubits, superposition, quantum entanglement
  |    - Shor's algorithm (overview)
  |    - Impact of quantum computers on existing cryptography
  +-- Week 3: Knowledge transfer to teams
  |    - Deepening understanding by teaching (Feynman Technique)
  |    - Designing internal study sessions
  |    - Writing technical blog posts
  +-- Week 4: Future learning roadmap
       - How to read research papers
       - Following conferences (SIGMOD, OSDI, SOSP, etc.)
       - Choosing and deepening a specialization
  [Milestone] Can teach CS fundamentals to a team
```

### 3.5 Expert Path (Research-Oriented) -- Continuous Learning

Target: Has cleared the advanced level and seeks deep expertise in a specific field

```
Expert path -- continuous learning:

Direction 1: Systems Research
  +-- Papers: Key publications from OSDI, SOSP, NSDI, EuroSys
  +-- Implementation: Build a small OS, distributed KVS, or database from scratch
  +-- Goal: Understand "why it was designed that way" at the paper level

Direction 2: Algorithms Research
  +-- Papers: Key publications from STOC, FOCS, SODA
  +-- Implementation: Top-tier competitive programming (AtCoder Yellow-Orange)
  +-- Goal: Able to design and analyze new algorithms

Direction 3: AI/ML Research
  +-- Papers: Key publications from NeurIPS, ICML, ICLR, ACL
  +-- Implementation: Reproduce models, propose new architectures
  +-- Goal: Understand and contribute to state-of-the-art research

Direction 4: Security Research
  +-- Papers: Key publications from CCS, S&P, USENIX Security
  +-- Implementation: Top-tier CTF (Capture The Flag) performance
  +-- Goal: Able to discover vulnerabilities and design defenses
```

---

## 4. Goal-Oriented Learning Paths

### 4.1 For Web Developers

```
CS fields web developers should prioritize:

Priority: *** Essential  ** Important  * Reference

*** Algorithms + Data Structures (complexity, hashing, trees)
  -> Directly impacts API performance improvement and data modeling
  -> Example: Solving N+1 queries, designing cache strategies

*** Networking (HTTP, TCP/IP, DNS, TLS)
  -> The foundation of the web. You cannot understand the web without this
  -> Example: Solving CORS issues, CDN configuration, HTTP/2 optimization

*** Databases (indexes, transactions, SQL optimization)
  -> The core of backend development. Most performance is determined by the DB
  -> Example: Improving slow queries, schema design

**  OS Basics (processes, memory management, I/O)
  -> Needed for Node.js event loop and memory leak investigation
  -> Example: Understanding pm2 cluster mode, monitoring memory usage

**  Security (XSS, CSRF, SQLi, authentication)
  -> Vulnerability mitigation for web applications is an essential skill
  -> Example: JWT vs Session, OAuth 2.0 flows

**  Concurrency (event loop, async/await, Workers)
  -> Directly relates to how Node.js/browsers work
  -> Example: Optimizing parallel requests with Promise.all

*   Theory of Computation (to understand the limits of regular expressions)
  -> Understand why you should not parse HTML with regex

*   Information Theory (compression, encoding)
  -> How gzip and Brotli work, principles of image optimization
```

### 4.2 For iOS/Mobile Developers

```
CS fields iOS/mobile developers should prioritize:

*** Algorithms + Data Structures
  -> Essential for collection operations and performance optimization
  -> Example: TableView diff update algorithms

*** Memory Management (ARC, no GC, retain cycles)
  -> iOS-specific memory management. Memory leaks are frequent without this
  -> Example: Proper use of weak/unowned, debugging deinit

*** Concurrency (GCD, Swift Concurrency, Actor)
  -> async/await and Actors are essential for modern iOS development
  -> Example: UI updates with MainActor, parallel processing with Task Group

**  OS Basics (processes, threads, sandboxing)
  -> Needed to understand the app lifecycle
  -> Example: Background Tasks, App Extensions

**  Networking (HTTP, WebSocket, Protocol Buffers)
  -> API communication, real-time feature implementation
  -> Example: Optimal URLSession configuration, gRPC support

**  Security (Keychain, App Transport Security, encryption)
  -> Protecting user data is also a legal requirement
  -> Example: Keychain Services, CryptoKit

*   Hardware (CPU, GPU, Neural Engine)
  -> Performance optimization and ML feature utilization
  -> Example: Metal Shaders, Core ML optimization

*   Databases (Core Data, SQLite, Realm)
  -> Options for local data persistence
  -> Example: Core Data migration strategies
```

### 4.3 For AI/ML Developers

```
CS fields AI/ML developers should prioritize:

*** Algorithms (especially optimization, graphs, probability)
  -> Foundations for gradient descent, computational graphs, stochastic methods
  -> Example: Understanding how the Adam optimizer works

*** Linear Algebra + Calculus + Probability & Statistics (mathematical foundations)
  -> Math is the core of ML: matrix operations, probability distributions,
     gradient computation
  -> Example: Attention is expressed as matrix products and softmax

*** Data Structures (tensors, sparse matrices, graphs)
  -> Efficient data representation determines ML scalability
  -> Example: Sparse matrices for recommendation system efficiency

**  Hardware (GPU, TPU, memory bandwidth)
  -> Understanding GPU programming directly impacts performance
  -> Example: CUDA kernel optimization, mixed-precision arithmetic

**  Concurrency (data parallelism, model parallelism, pipeline parallelism)
  -> Parallelization is essential for large-scale model training
  -> Example: PyTorch DDP, DeepSpeed, Megatron-LM

**  Distributed Systems (distributed training, parameter servers)
  -> Design and implementation of multi-node training
  -> Example: Ring-AllReduce, Federated Learning

*   Theory of Computation (computability, approximation algorithms)
  -> Learn how to approach NP-hard problems

*   Information Theory (entropy, cross-entropy)
  -> Understand the theoretical background of loss functions
```

### 4.4 For SRE/Infrastructure Engineers

```
CS fields SRE/infrastructure engineers should prioritize:

*** OS (process management, cgroups, namespaces, file systems)
  -> The foundation of container technology. Docker = cgroups + namespaces
  -> Example: How OOM Killer works, file descriptor limits

*** Networking (all TCP/IP layers, DNS, load balancing)
  -> 80% of incident investigations involve networking
  -> Example: Troubleshooting with tcpdump, choosing L4/L7 LB

*** Distributed Systems (CAP, consistency models, Raft/Paxos)
  -> All modern infrastructure is a distributed system
  -> Example: How etcd (Raft) works, split-brain mitigation

**  Algorithms (hashing, consistent hashing, Bloom filters)
  -> Needed for cache distribution, routing, and filtering
  -> Example: Cache node addition with consistent hashing

**  Security (TLS, firewalls, zero trust)
  -> Infrastructure security is the most critical responsibility
  -> Example: mTLS, network policy design

**  Databases (replication, sharding, backup)
  -> Ensuring data availability and durability
  -> Example: PostgreSQL streaming replication setup

*   Hardware (CPU, SSD, network equipment)
  -> Capacity planning and hardware failure response
```

### 4.5 Goal-Oriented Path Comparison Table

| CS Field | Web | iOS | AI/ML | SRE |
|----------|-----|-----|-------|-----|
| Algorithms | *** | *** | *** | ** |
| Data Structures | *** | *** | *** | ** |
| Networking | *** | ** | * | *** |
| OS | ** | ** | * | *** |
| Databases | *** | * | * | ** |
| Security | ** | ** | * | ** |
| Concurrency | ** | *** | ** | ** |
| Distributed Systems | * | * | ** | *** |
| Theory of Computation | * | * | * | * |
| Hardware | * | * | ** | * |
| Mathematics | * | * | *** | * |

**Legend**: *** = Top priority, ** = Learn if time permits, * = Learn as needed

---

## 5. Section Dependencies and Recommended Order for This Skill

### 5.1 Inter-Section Dependencies

```
Section dependency map for this Skill:

  00-introduction (you are here)
  |
  +--> 02-data-representation --> 01-hardware-basics
  |    (Internal data             (How HW works)
  |     representation)
  |         |
  |         v
  |    03-algorithms-basics <--- Can be studied independently
  |    (Algorithms)
  |         |
  |         v
  |    04-data-structures
  |    (Data Structures)
  |         |
  |    +----+----+
  |    v         v
  |  05-computation    06-programming
  |  -theory           -paradigms
  |  (Computation      (Paradigms)
  |   Theory)
  |    |                |
  |    +----+-----------|
  |         v
  |    07-software-engineering-basics
  |    (SE Basics)
  |         |
  |         v
  |    08-advanced-topics
  |    (Advanced: Distributed, Concurrency, Security, AI)
  |
  +--> Each section can also be read independently
       (but following the dependencies is most efficient)
```

### 5.2 Section Overview and Estimated Time

| Section | Number of Topics | Estimated Time | Prerequisites |
|---------|-----------------|----------------|---------------|
| 00-introduction | 4 files | 4 hours | None |
| 01-hardware-basics | 5 files | 15 hours | Basics of 02 |
| 02-data-representation | 6 files | 12 hours | None |
| 03-algorithms-basics | 9 files | 30 hours | Programming basics |
| 04-data-structures | 8 files | 25 hours | Basics of 03 |
| 05-computation-theory | 7 files | 20 hours | Basics of 03-04 |
| 06-programming-paradigms | 6 files | 15 hours | Basics of 03-04 |
| 07-software-engineering | 5 files | 12 hours | Basics of 05-06 |
| 08-advanced-topics | 5 files | 20 hours | Through 07 |
| **Total** | **55 files** | **~153 hours** | -- |

### 5.3 Cross-Reference Map Between Sections

This shows how content learned in each section is utilized in other sections.

```
Cross-reference map:

  02-data-representation          01-hardware-basics
  +- Binary ------------------> CPU instruction set
  +- Floating point ----------> FPU, SIMD
  +- Character encoding ------> Data representation in memory

  03-algorithms-basics            04-data-structures
  +- Sorting -----------------> Heaps (heap sort)
  +- Searching ---------------> BST (binary search tree)
  +- Graphs ------------------> Adjacency list/matrix
  +- DP ----------------------> Memoization tables

  04-data-structures              05-computation-theory
  +- Stacks ------------------> PDA (pushdown automata)
  +- Hashing -----------------> Cryptographic hashing
  +- Tree structures ---------> Parse trees

  05-computation-theory           08-advanced-topics
  +- Finite automata ---------> Network protocols
  +- Computability -----------> Impossibility theorems in distributed systems
  +- NP-completeness ---------> Approximation algorithms

  06-programming-paradigms        07-software-engineering
  +- OOP ---------------------> Design patterns
  +- Functional --------------> Immutable design
  +- Concurrent programming --> Microservices
```

---

## 6. Learning Schedule Templates

### 6.1 Weekly Schedule (Recommended)

```
Weekly schedule for full-time workers:

  Monday    Tuesday   Wednesday Thursday  Friday    Saturday  Sunday
  +--------++--------++--------++--------++--------++--------++--------+
  |Theory  ||Practice||Theory  ||Practice||Review  ||Practice||Rest    |
  |30min   ||60min   ||30min   ||60min   ||30min   ||2-3h    ||        |
  |        ||        ||        ||        ||        ||        ||(opt.)  |
  |Textbook||Coding  ||Textbook||Coding  ||Anki    ||LeetCode||        |
  |Video   ||        ||Video   ||        ||Summary ||Project ||        |
  +--------++--------++--------++--------++--------++--------++--------+

  Weekly study time: ~6.5 hours
  Monthly study time: ~26 hours
  6-month total: ~156 hours ~ Total estimated time for this Skill (153 hours)
```

### 6.2 Intensive Learning Schedule (For Job Transition, etc.)

```
Intensive learning schedule (4 hours/day, complete in 3 months):

  Morning (2h)            Afternoon (2h)
  +------------------+    +------------------+
  | Theory study     |    | Practical        |
  | 9:00-11:00       |    | exercises        |
  |                  |    | 14:00-16:00      |
  | - Textbooks      |    | - LeetCode       |
  | - This Skill     |    | - Implementation |
  | - Note-taking    |    | - Mock interviews|
  +------------------+    +------------------+

  Weekly study time: ~20 hours (weekday 4h x 5 days)
  3-month total: ~240 hours (plenty of margin)
```

---

## 7. Evidence-Based Learning Techniques

### 7.1 Feynman Technique

Physicist Richard Feynman's learning method is extremely effective for CS study.

**Steps:**
1. **Choose a concept**: e.g., "Hash table"
2. **Explain it as if teaching a child**: Simple, without jargon
3. **When you get stuck, go back**: The parts you can't explain are the gaps
4. **Simplify and re-explain**: Use analogies

```
Example: Explaining hash tables with the Feynman Technique

"A hash table is like a library bookshelf.
 There's a magic formula (hash function) that calculates a shelf number
 from a book's title.
 You can instantly find which shelf any book is on with the formula.
 But sometimes two books end up on the same shelf (collision).
 When that happens, you either make a list on that shelf (chaining)
 or put it on the next shelf (open addressing)."

-> Parts where you struggled to explain = gaps in understanding
-> "What's a concrete example of a hash function?" -> Research to deepen understanding
```

**Feynman Technique in Practice -- Analogy Examples by CS Topic:**

| CS Topic | Analogy |
|----------|---------|
| Stack | A pile of plates (the last plate placed is the first one taken) |
| Queue | A line of people (first in line gets served first) |
| Binary search | Dictionary lookup (open the middle, decide first/second half) |
| Recursion | Facing mirrors (a mirror within a mirror within a mirror...) |
| Cache | Papers on your desk (keep frequently used items close) |
| TCP/IP | Postal system (address, envelope, delivery confirmation) |
| Encryption | Padlock (public key = open lock, private key = the key) |
| Deadlock | Two cars facing each other on a bridge (neither can reverse) |
| GC | Garbage truck (collects memory that's no longer in use) |
| Virtual memory | Book catalog (not all books on the shelf; fetch as needed) |

### 7.2 Spaced Repetition

A review schedule based on Ebbinghaus's forgetting curve is the most effective method for long-term memory retention.

```
Spaced repetition schedule:

  Memory
  retention
  100% |*
      | \
  80% |  *---- Review 1 (1 day later)
      |    \      *---- Review 2 (3 days later)
  60% |     \       \      *---- Review 3 (7 days later)
      |      \       \       \      *---- Review 4 (14 days later)
  40% |       \       \       \       \      *---- Review 5 (30 days later)
      |        \       \       \       \       \
  20% |    Forgetting  Forgetting slows with each review
      |    curve
   0% +------+------+------+------+------+------
         Day 0     3d     7d     14d    30d    60d

  How to practice:
  1. Create Anki cards for CS concepts
  2. Review 5-10 minutes daily following Anki's schedule
  3. Card front: "What is the average search time for a hash table?"
  4. Card back: "O(1). Because the address is computed directly by the hash function"
```

**Anki Card Examples:**

```
--- Card 1 (Basic) ---
Front: Name three algorithms with O(log n) time complexity in Big-O notation
Back: - Binary search
      - Balanced BST lookup
      - Heap insertion/deletion

--- Card 2 (Comprehension Check) ---
Front: Why is the worst-case complexity of a hash table O(n)?
Back: When all keys collide into the same bucket,
      it degrades to a linked list.
      Countermeasures: good hash function selection, load factor management

--- Card 3 (Application) ---
Front: From which Python version does dict preserve insertion order?
       What was the internal implementation change?
Back: From Python 3.7 (in 3.6 it was an implementation detail).
      With the introduction of compact dicts,
      an insertion-order array is maintained separately from the hash table.
```

### 7.3 Active Recall

Not just reading text, but **remembering with the material closed** dramatically increases long-term retention.

**How to practice:**
- After reading each section, close the material and ask yourself "List three things I just learned"
- **Writing on a blank page** is more effective than reviewing notes
- Ask yourself "Name three collision resolution methods for hash tables"

```python
# Self-test script to support Active Recall

import random
import json
from datetime import datetime

class SelfTestSystem:
    """A self-test system to support active recall"""

    def __init__(self):
        self.questions = []
        self.results = []

    def add_question(self, topic, question, expected_keywords):
        """Add a test question"""
        self.questions.append({
            "topic": topic,
            "question": question,
            "keywords": expected_keywords
        })

    def run_test(self, num_questions=5):
        """Quiz num_questions randomly"""
        selected = random.sample(
            self.questions,
            min(num_questions, len(self.questions))
        )

        score = 0
        for i, q in enumerate(selected, 1):
            print(f"\nQ{i}: [{q['topic']}]")
            print(f"  {q['question']}")
            answer = input("  Your answer: ")

            # Keyword check (simplified)
            matched = [kw for kw in q["keywords"]
                       if kw.lower() in answer.lower()]
            ratio = len(matched) / len(q["keywords"])

            if ratio >= 0.7:
                print("  Correct!")
                score += 1
            elif ratio >= 0.3:
                print(f"  Partial. Missing keywords: "
                      f"{set(q['keywords']) - set(matched)}")
            else:
                print(f"  Needs review. Keywords: {q['keywords']}")

        print(f"\nResult: {score}/{len(selected)} "
              f"({score/len(selected)*100:.0f}%)")

        self.results.append({
            "date": datetime.now().isoformat(),
            "score": score,
            "total": len(selected)
        })

# Usage example
test = SelfTestSystem()
test.add_question(
    "Data Structures",
    "Name three collision resolution methods for hash tables",
    ["chaining", "open addressing", "double hashing"]
)
test.add_question(
    "Algorithms",
    "Name three stable sorting algorithms",
    ["merge sort", "insertion sort", "bubble sort", "TimSort"]
)
test.add_question(
    "Complexity",
    "What is the time complexity and prerequisite for binary search?",
    ["O(log n)", "sorted"]
)
```

### 7.4 Interleaving

Research has shown that **alternating between different topics** yields better long-term learning outcomes than studying the same topic repeatedly (block practice).

```
Block practice vs. interleaving:

Block practice (not recommended):
  Monday: Sort Sort Sort Sort
  Tuesday: Search Search Search Search
  Wednesday: Tree Tree Tree Tree
  -> High accuracy during practice, but low long-term retention

Interleaving (recommended):
  Monday: Sort -> Search -> Tree -> Sort
  Tuesday: Search -> Tree -> Sort -> Search
  Wednesday: Tree -> Sort -> Search -> Tree
  -> More confusion during practice, but higher long-term retention

Reason: Interleaving requires deciding "which technique fits this problem"
        every time, training metacognition
```

---

## 8. Recommended Resources

### 8.1 Books (In Recommended Order)

| # | Title | Field | Level | Language | Features |
|---|-------|-------|-------|----------|----------|
| 1 | Introduction to Algorithms (CLRS) | Algorithms | Mid-Adv | EN | The bible of algorithms. Highest comprehensiveness |
| 2 | Computer Systems: A Programmer's Perspective (CS:APP) | Systems | Mid | EN | Understanding systems from a programmer's perspective |
| 3 | Structure and Interpretation of Computer Programs (SICP) | CS Basics | Mid | EN | A classic for developing CS thinking |
| 4 | Designing Data-Intensive Applications (DDIA) | Distributed | Mid-Adv | EN/JP | The definitive book on modern distributed systems |
| 5 | Operating Systems: Three Easy Pieces (OSTEP) | OS | Mid | EN (free) | Best OS introduction. Freely available |
| 6 | The Algorithm Design Manual (Skiena) | Algorithms | Mid | EN | Practical algorithm design |
| 7 | Computer Networking: A Top-Down Approach | Networking | Mid | EN/JP | Learn networking top-down |
| 8 | Introduction to the Theory of Computation (Sipser) | Computation | Mid-Adv | EN | Standard textbook for theory of computation |
| 9 | Clean Code (Robert C. Martin) | SE | Beg-Mid | EN/JP | How to write readable code |
| 10 | The Art of Computer Programming (Knuth) | Algorithms | Adv | EN | The ultimate algorithms reference |
| 11 | Introduction to Algorithms (CLRS, Japanese edition) | Algorithms | Mid-Adv | JP | Japanese translation of CLRS |
| 12 | Programming Contest Challenge Book | Competitive | Mid | JP | Known as "Ant Book." Competitive programming classic |
| 13 | Programming Pearls | Algorithms | Mid | EN/JP | Sharpening algorithmic thinking |
| 14 | Computer Organization and Design (Patterson & Hennessy) | HW | Mid | EN/JP | Known as "P&H." The hardware classic |
| 15 | Advanced Programming in the UNIX Environment (APUE) | OS/UNIX | Mid-Adv | EN/JP | The encyclopedia of UNIX/Linux programming |

### 8.2 Online Courses

| # | Course | Provider | Target | Features |
|---|--------|----------|--------|----------|
| 1 | CS50 | Harvard/edX | Beginner | The definitive CS introduction. Prof. David Malan |
| 2 | 6.006 Introduction to Algorithms | MIT OCW | Intermediate | Systematic algorithm fundamentals |
| 3 | 6.046 Design and Analysis of Algorithms | MIT OCW | Advanced | Algorithm design and analysis |
| 4 | 15-213 (CS:APP) | CMU | Intermediate | Deep understanding of computer systems |
| 5 | Nand2Tetris | Coursera | Intermediate | Build everything from NAND gates to an OS |
| 6 | Algorithms I & II | Princeton/Coursera | Intermediate | Prof. Sedgewick's renowned lectures |
| 7 | Stanford CS229 | Stanford Online | Advanced | The standard machine learning course |
| 8 | Operating Systems: Three Easy Pieces | OSTEP | Intermediate | Free OS textbook |

### 8.3 Practice Sites

| # | Site | Features | Recommended Level | Recommended Usage |
|---|------|----------|-------------------|-------------------|
| 1 | LeetCode | 2500+ problems, company-specific lists | Intermediate | 5 Medium problems per week |
| 2 | AtCoder | Japanese, weekly contests | Beg-Adv | Participate in ABC weekly |
| 3 | HackerRank | Problems across many domains | Beg-Mid | In parallel with language learning |
| 4 | Project Euler | Mathematical problems | Mid-Adv | Fusion of math and CS |
| 5 | Exercism | Language-specific mentoring | Beg-Mid | When learning a new language |

### 8.4 Resource Selection Flowchart

```
Resource selection flowchart:

  "I want to systematically learn CS"
    |
    +-- No programming experience?
    |     +-- Yes -> Start with CS50 (Harvard)
    |     +-- No  |
    |             v
    +-- Uncomfortable with English materials?
    |     +-- Yes -> This Skill + Japanese books (Ant Book, P&H)
    |     +-- No  |
    |             v
    +-- What is your goal?
    |     +-- Job change/interview prep -> LeetCode + CLRS + System Design Primer
    |     +-- Practical skill improvement -> CS:APP + DDIA + This Skill
    |     +-- Academic research -> Sipser + CLRS + Field-specific papers
    |     +-- General education -> This Skill + CS50 + SICP
    |
    +-- What is your learning style?
          +-- Video -> MIT OCW + Coursera
          +-- Reading -> CLRS + CS:APP + OSTEP
          +-- Hands-on -> LeetCode + Nand2Tetris + Personal projects
```

---

## 9. Tracking Learning Progress

### 9.1 How to Track Progress

Visualizing learning progress is extremely important for maintaining motivation.

```python
# Learning progress tracking system (simple version)

import json
from datetime import datetime, timedelta
from collections import defaultdict

class LearningTracker:
    """A system to manage CS learning progress"""

    def __init__(self, filepath="learning_progress.json"):
        self.filepath = filepath
        self.sessions = []
        self.goals = {}

    def log_session(self, topic, duration_min, notes=""):
        """Record a learning session"""
        session = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "topic": topic,
            "duration": duration_min,
            "notes": notes
        }
        self.sessions.append(session)
        self._save()

        total = sum(s["duration"] for s in self.sessions
                    if s["topic"] == topic)
        print(f"Logged: {topic} {duration_min}min "
              f"(total: {total}min = {total/60:.1f}h)")

    def weekly_report(self):
        """Generate a weekly report"""
        today = datetime.now()
        week_ago = today - timedelta(days=7)

        weekly = [s for s in self.sessions
                  if datetime.strptime(s["date"], "%Y-%m-%d") >= week_ago]

        by_topic = defaultdict(int)
        for s in weekly:
            by_topic[s["topic"]] += s["duration"]

        total = sum(by_topic.values())

        print(f"\n=== Weekly Report ({week_ago.strftime('%m/%d')}"
              f"-{today.strftime('%m/%d')}) ===")
        print(f"Total study time: {total}min ({total/60:.1f}h)")
        print(f"\nBy topic:")
        for topic, minutes in sorted(by_topic.items(),
                                      key=lambda x: -x[1]):
            bar = "█" * (minutes // 10)
            print(f"  {topic:20s} {minutes:4d}min {bar}")

    def streak_count(self):
        """Calculate consecutive study days"""
        if not self.sessions:
            return 0

        dates = sorted(set(s["date"] for s in self.sessions), reverse=True)
        streak = 1
        for i in range(len(dates) - 1):
            d1 = datetime.strptime(dates[i], "%Y-%m-%d")
            d2 = datetime.strptime(dates[i+1], "%Y-%m-%d")
            if (d1 - d2).days == 1:
                streak += 1
            else:
                break
        return streak

    def _save(self):
        with open(self.filepath, 'w') as f:
            json.dump({"sessions": self.sessions}, f, indent=2)


# Usage example
tracker = LearningTracker()
tracker.log_session("Algorithms", 60, "Implemented binary search")
tracker.log_session("Data Structures", 45, "Hash table collision resolution")
tracker.weekly_report()
print(f"\nConsecutive study days: {tracker.streak_count()} days")
```

### 9.2 Milestone Checklist

Define clear milestones to achieve at each level.

```
Beginner milestones (by month):

Month 1 completion check:
  [ ] Can convert between decimal and binary
  [ ] Can explain the basic structure of IEEE 754
  [ ] Can predict the results of bitwise operations (AND, OR, XOR)

Month 2 completion check:
  [ ] Can explain the fetch-decode-execute cycle
  [ ] Can explain two types of cache locality
  [ ] Can diagram the memory hierarchy

Month 3-4 completion check:
  [ ] Can express complexity in Big-O notation
  [ ] Can implement bubble sort and merge sort
  [ ] Can implement binary search
  [ ] Can write recursive functions
  [ ] Can implement BFS/DFS

Month 5 completion check:
  [ ] Can explain the difference between arrays and linked lists
  [ ] Can implement a hash table
  [ ] Can implement BST insertion and search
  [ ] Have solved 30+ LeetCode Easy problems

Month 6 completion check:
  [ ] Can explain the differences between OOP, functional, and procedural
  [ ] Score 5+ Yes on the intermediate level assessment
  [ ] Have reviewed the 6 months of study and have a plan for what's next
```

---

## 10. Common Anti-Patterns and Countermeasures

### 10.1 Anti-Pattern 1: Tutorial Hell

**Symptom**: Completing tutorials one after another but unable to build anything independently.

```
The tutorial hell cycle:

  +------------------+
  | Start a tutorial |
  +--------+---------+
           |
           v
  +------------------+
  | Follow along,    |
  | feel "I get it"  |
  +--------+---------+
           |
           v
  +------------------+
  | Try to apply     |
  | it on your own   |
  +--------+---------+
           |
           v
  +------------------+     +------------------+
  | Can't write      |---->| Next tutorial    |--- Loop!
  | anything, give up|     |                  |
  +------------------+     +------------------+
```

**Countermeasures:**
1. After every tutorial, set a "variation problem" for yourself
2. Example: After a sorting tutorial, try "reverse sort" or "check if it's stable"
3. Don't just copy; **understand, then close the tutorial and write it yourself**
4. Target ratio: "tutorial time : self-implementation time = 1:2"

### 10.2 Anti-Pattern 2: Theory Paralysis (Analysis Paralysis)

**Symptom**: Study only theory without writing code. Can't move forward until you understand everything.

```
The theory paralysis cycle:

  +---------------------+
  | Start reading a     |
  | textbook chapter    |
  +--------+------------+
           |
           v
  +---------------------+
  | A detail catches    |
  | your eye, chase     |
  | references          |
  +--------+------------+
           |
           v
  +---------------------+
  | References of       |
  | references...       |--- Infinite rabbit hole!
  +--------+------------+
           |
           v
  +---------------------+
  | 1 month has passed  |
  | Still on chapter 1  |
  | 0 lines of code     |
  +---------------------+
```

**Countermeasures:**
1. **80% Rule**: Move on once you understand 80%
2. **Time limits**: 2 hours max per topic. Beyond that, add to a "revisit later" list
3. **Implementation first**: Before reading theory, try writing "working code" first
4. **TODO list**: Record items to deep-dive into and come back later

### 10.3 Anti-Pattern 3: Comparison Trap

**Symptom**: Comparing your progress with others on social media and feeling behind.

**Countermeasures:**
1. Learning is a solo endeavor. Your only comparison target is "yourself yesterday"
2. Record small daily progress (learning log)
3. Mute/unfollow tech accounts on social media
4. Being "slow" is not the problem. "Stopping" is

### 10.4 Anti-Pattern 4: Tool Worship

**Symptom**: Endlessly searching for the optimal editor, note-taking app, or study method, and never starting.

**Countermeasures:**
1. **Start now**: You can optimize tools later
2. **Minimal set**: Text editor + terminal + browser is sufficient
3. **30-minute rule**: Don't spend more than 30 minutes on tool selection

---

## 11. Practical Exercises

### Exercise 1: Design Your Own Learning Path (Fundamentals)

**Goal**: Create a concrete study plan tailored to your level and objectives

**Steps:**
1. Assess your level using the level assessment checklist (Section 3.1)
2. Choose the appropriate goal-oriented path from Section 4
3. Fill in the template below and create a 3-month plan

```
Learning plan template:

Name: _______________
Current level: [ ] Beginner  [ ] Intermediate  [ ] Advanced
Goal: _______________
Available study time: Weekdays ___ hours/day, Weekends ___ hours/day

Month 1 goal: _______________
  Week 1: _______________
  Week 2: _______________
  Week 3: _______________
  Week 4: _______________
  Milestone: _______________

Month 2 goal: _______________
  Week 1: _______________
  Week 2: _______________
  Week 3: _______________
  Week 4: _______________
  Milestone: _______________

Month 3 goal: _______________
  Week 1: _______________
  Week 2: _______________
  Week 3: _______________
  Week 4: _______________
  Milestone: _______________

Target by end of 3 months: _______________
Resources to use:
  [ ] This Skill
  [ ] Book: _______________
  [ ] Online course: _______________
  [ ] Practice site: _______________
```

**Evaluation criteria:**
- Is your level correctly assessed?
- Are goals specific and measurable?
- Are there weekly milestones?
- Are resources appropriately selected?

### Exercise 2: Feynman Technique Practice (Application)

**Goal**: Express a deep understanding of concepts in a form that can be explained to others

Choose **2** from the following concepts and write an explanation of each in **under 3 minutes**, as if explaining to a middle school student. You may use technical terms but must always provide explanations.

- Binary search
- Stack
- Recursion
- Cache
- Hash table
- TCP/IP three-way handshake

**Evaluation criteria:**
- Are all technical terms accompanied by explanations?
- Are analogies appropriate?
- Does it capture the essence (not just surface-level)?
- Could a middle school student read and understand it?

### Exercise 3: Knowledge Inventory (Advanced)

**Goal**: Objectively evaluate the overall picture of your CS knowledge and determine learning priorities

Browse the titles of all 55 files in this Skill and fill in the following for each topic.

```
Knowledge inventory table:

| # | Topic                | Under-  | Used in | Learning | Priority |
|   |                      |standing | work?   | Difficulty|         |
|   |                      | (1-5)   |         |          |          |
|---|----------------------|---------|---------|----------|----------|
| 1 | CPU Architecture     |   ?     |   ?     |   ?      |   ?      |
| 2 | Memory Hierarchy     |   ?     |   ?     |   ?      |   ?      |
| 3 | Storage              |   ?     |   ?     |   ?      |   ?      |
| 4 | Binary/Data Repr.    |   ?     |   ?     |   ?      |   ?      |
| 5 | Floating Point       |   ?     |   ?     |   ?      |   ?      |
| 6 | Character Encoding   |   ?     |   ?     |   ?      |   ?      |
| 7 | Complexity Analysis  |   ?     |   ?     |   ?      |   ?      |
| 8 | Sorting Algorithms   |   ?     |   ?     |   ?      |   ?      |
| 9 | Search Algorithms    |   ?     |   ?     |   ?      |   ?      |
|10 | Dynamic Programming  |   ?     |   ?     |   ?      |   ?      |
|...|  ...                 |  ...    |  ...    |  ...     | ...      |

Understanding: 1=Only know the name 2=Understand the overview 3=Can explain
              4=Can implement 5=Can teach
Work usage: Daily / Weekly+ / Occasionally / Never
Learning difficulty: 1=Easy to 5=Hard
Priority = (5 - understanding) x work usage weight x difficulty adjustment
```

-> Study in order of highest priority. This becomes your own optimal learning path.

---

## 12. Cross-Disciplinary Learning Projects

To develop an integrated understanding of CS fundamentals, it is strongly recommended to work on projects that span multiple fields.

### 12.1 Project Examples

```
Integrated project list (by difficulty):

Level 1: Simple Hash Map Implementation
  Related fields: Data Structures, Algorithms, Memory Management
  Estimated time: 4-8 hours
  What you'll learn:
    - Hash function design
    - Collision resolution (chaining, open addressing)
    - Dynamic resizing (based on load factor)
    - Complexity analysis (average vs. worst case)

Level 2: Simple HTTP Server Implementation
  Related fields: Networking, OS (processes/threads), String processing
  Estimated time: 8-16 hours
  What you'll learn:
    - TCP socket programming
    - HTTP protocol parsing
    - Concurrent request handling (multithreaded or event-driven)
    - File I/O

Level 3: Simple Database Engine Implementation
  Related fields: Data Structures (B-tree), Algorithms, OS (file I/O), Concurrency
  Estimated time: 20-40 hours
  What you'll learn:
    - B-tree index implementation
    - Page-based storage management
    - SQL parser (simplified)
    - Transactions (WAL)

Level 4: Distributed KVS Implementation
  Related fields: Distributed Systems, Networking, Concurrency, Data Structures
  Estimated time: 40-80 hours
  What you'll learn:
    - Raft consensus algorithm
    - RPC (Remote Procedure Call)
    - Consistent hashing
    - Replication
```

### 12.2 Project Implementation Example: Simple Hash Map

Here is an implementation of "Level 1: Simple Hash Map" from the integrated projects. This code applies knowledge across data structures, algorithms, and memory management.

```python
# Simple hash map implementation example
# Related fields: Data Structures (hash tables), Algorithms (hash functions)

class SimpleHashMap:
    """A simple hash map using chaining"""

    def __init__(self, initial_capacity=16, load_factor=0.75):
        self.capacity = initial_capacity
        self.load_factor = load_factor
        self.size = 0
        # Each bucket is a list of (key, value) pairs (chaining)
        self.buckets = [[] for _ in range(self.capacity)]

    def _hash(self, key):
        """Hash function: compute bucket index from key"""
        # Use Python's built-in hash() and take modulo with capacity
        return hash(key) % self.capacity

    def put(self, key, value):
        """Insert a key-value pair (overwrite if key exists)"""
        index = self._hash(key)
        bucket = self.buckets[index]

        # Check for existing key update
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)  # Overwrite
                return

        # New insertion
        bucket.append((key, value))
        self.size += 1

        # Load factor check -> resize if exceeded
        if self.size / self.capacity > self.load_factor:
            self._resize()

    def get(self, key, default=None):
        """Get the value for a key (O(1) average)"""
        index = self._hash(key)
        for k, v in self.buckets[index]:
            if k == key:
                return v
        return default

    def delete(self, key):
        """Delete a key"""
        index = self._hash(key)
        bucket = self.buckets[index]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket.pop(i)
                self.size -= 1
                return True
        return False

    def _resize(self):
        """Double the capacity and rehash all entries"""
        old_buckets = self.buckets
        self.capacity *= 2
        self.buckets = [[] for _ in range(self.capacity)]
        self.size = 0

        for bucket in old_buckets:
            for key, value in bucket:
                self.put(key, value)  # Rehash with new capacity

    def __repr__(self):
        items = []
        for bucket in self.buckets:
            for k, v in bucket:
                items.append(f"{k!r}: {v!r}")
        return "SimpleHashMap({" + ", ".join(items) + "})"


# Usage example and complexity verification
hm = SimpleHashMap()
hm.put("name", "Alice")      # O(1) average
hm.put("age", 30)             # O(1) average
hm.put("city", "Tokyo")       # O(1) average

print(hm.get("name"))         # "Alice" -- O(1) average
print(hm.get("missing"))      # None -- O(1) average

hm.delete("age")              # O(1) average
print(hm)                     # SimpleHashMap({'name': 'Alice', 'city': 'Tokyo'})

# Complexity summary:
# | Operation | Average | Worst  |
# |-----------|---------|--------|
# | put       | O(1)    | O(n)   |  <- When all keys collide
# | get       | O(1)    | O(n)   |  <- When all keys collide
# | delete    | O(1)    | O(n)   |  <- When all keys collide
# | resize    | O(n)    | O(n)   |  <- Rehash all entries
```

Through this kind of implementation, you can experientially understand "why Python's dict operates in O(1)," "what happens when the load factor is too high," and "why hash function quality matters."

### 12.3 How to Approach Projects

```
Recommended project workflow:

  +------------------------------------------------------+
  |                                                        |
  |  1. Understand the specification (study existing       |
  |     implementations)                                   |
  |     |                                                  |
  |  2. Build a minimum viable version (MVP)               |
  |     |                                                  |
  |  3. Write tests to verify correctness                  |
  |     |                                                  |
  |  4. Analyze time complexity and memory usage            |
  |     |                                                  |
  |  5. Optimize (based on profiling results)              |
  |     |                                                  |
  |  6. Write documentation (organize what you learned)    |
  |     |                                                  |
  |  7. Write a blog post (Feynman Technique)              |
  |                                                        |
  +------------------------------------------------------+
```

---

## 13. FAQ

### Q1: Where should I start learning CS?

**A**: The optimal entry point depends on your level.

- **No programming experience**: First learn Python or JavaScript, then proceed to this Skill. Harvard CS50 is the ideal entry point.
- **Programming beginner**: Follow `00-introduction` -> `02-data-representation` -> `03-algorithms-basics` in order.
- **Practical experience (no CS background)**: Run the level assessment checklist (Section 3.1) and start with your weak areas. Prioritize filling in fields where you don't yet meet the intermediate level.
- **CS degree holder**: Refresh your knowledge with `05-computation-theory` -> `08-advanced-topics` and find connections to your practical work.

### Q2: How many hours a day should I study?

**A**: Quality matters more than quantity, and consistency matters most of all.

| Situation | Recommended Time | Breakdown |
|-----------|-----------------|-----------|
| Minimum | 30 min/day | Read an article during commute, etc. |
| Recommended | 1-2 hours/day | 30 min theory + 30 min practice + review |
| Intensive | 3-4 hours/day | Job transition, interview prep |

The key point is that a little bit every day is more effective than cramming on weekends (the spaced repetition effect). "1.5 hours every day" yields higher retention than "10 hours on the weekend."

### Q3: The learning materials are only in English...

**A**: English is unavoidable in CS, but you can approach it gradually.

1. **Start in Japanese** to understand the concepts (this Skill, Japanese textbooks)
2. **Watch videos with English subtitles** to get used to listening (CS50, MIT OCW)
3. **Challenge English textbooks** (technical English uses a more limited vocabulary than everyday English)
4. **Write/speak in English** (write LeetCode solutions on a blog, etc.)

Technical English can be largely covered with 2,000-3,000 words, making it far more efficient than general English study. Here are some frequently used technical English terms.

```
Frequently used CS English (knowing these makes reading materials easier):

Basic verbs:
  allocate, traverse, invoke,
  implement, initialize, iterate,
  propagate, terminate, instantiate

Basic nouns:
  complexity, overhead, throughput,
  latency, concurrency, consistency,
  invariant, bottleneck, trade-off

Basic adjectives:
  deterministic, idempotent, immutable,
  scalable, amortized, asymptotic
```

### Q4: How do I maintain motivation when self-studying?

**A**: The keys to maintaining motivation are "accumulating small successes" and "connecting with others."

1. **Set small goals**: "Implement 3 sorting algorithms this week"
2. **Produce output**: Share what you learn on blogs, social media, and study groups
3. **Find peers**: Join programming learning communities on Discord/Slack
4. **Connect to practice**: Ask yourself "How can I apply what I learned today to my project?"
5. **LeetCode Streak**: Build a habit of solving 1 problem per day (track consecutive days)
6. **Visualize progress**: GitHub contribution graph, learning logs
7. **Reflect regularly**: Once a month, confirm "a problem I couldn't solve a month ago"

### Q5: Can I learn CS even if I'm bad at math?

**A**: Many CS fields do not require advanced mathematics. However, the required math level varies by field.

| CS Field | Required Math Level | Specific Math Needed |
|----------|--------------------|--------------------|
| Programming basics | Middle school math | Arithmetic, logic (AND/OR) |
| Data structures | High school math | Logarithms, exponents |
| Algorithms | HS-University intro | Asymptotic analysis, basic probability |
| Theory of computation | University intro | Set theory, logic, induction |
| AI/ML | University math | Linear algebra, calculus, probability & statistics |
| Cryptography | University math | Number theory, group theory |

High school math is sufficient to handle complexity analysis of algorithms. If you're uneasy about math, "just-in-time learning" -- studying it when the need arises -- is the most efficient approach.

### Q6: How should I study CS for job transitions and interviews?

**A**: Studying aligned with the components of technical interviews is most effective.

```
Typical structure of a technical interview:

  +-------------------------------------------+
  | 1. Coding Interview (45 min x 2-3 rounds) |
  |    -> Algorithms + Data Structures are key |
  |    -> Solve 50+ LeetCode Medium problems   |
  |                                            |
  | 2. System Design Interview (45-60 min x    |
  |    1-2 rounds)                             |
  |    -> Distributed systems + DB + Networking|
  |    -> Design URL shortener, chat, Twitter  |
  |                                            |
  | 3. Behavioral Interview (30-45 min x 1-2   |
  |    rounds)                                 |
  |    -> CS knowledge not needed, but CS      |
  |      terminology is needed to describe     |
  |      technical experience                  |
  |                                            |
  | Recommended schedule (3 months):           |
  |   Month 1: Review algorithms + data struct.|
  |   Month 2: Intensive LeetCode Medium       |
  |   Month 3: System design + mock interviews |
  +-------------------------------------------+
```

### Q7: Is this Skill alone sufficient?

**A**: This Skill aims to provide a **comprehensive overview of CS fundamentals**. It is sufficient for understanding "what you need to know" and "why it matters" in each field, but additional resources are required for deep specialization.

Positioning of this Skill:
- **Understanding the big picture**: Sufficient with this Skill
- **Ability to explain concepts**: Sufficient with this Skill
- **Basic implementation skills**: Sufficient with this Skill + exercises
- **Interview preparation level**: Requires this Skill + LeetCode + CLRS
- **Research level**: Requires this Skill + specialized books + papers

---

## 14. Long-Term Learning Strategy

### 14.1 Building T-Shaped Skills

An effective approach to CS learning is aiming for "T-shaped skills."

```
T-shaped skill model:

  <---------- Broad foundational knowledge ---------->
  +-------+-------+-------+-------+-------+-------+
  | Algo  | Data  |  OS   |  NW   |  DB   | Secu- | <- Shallow and broad
  | rithms| Struct|       |       |       | rity  |    (all fields)
  +-------+-------+-------+-------+-------+-------+
  |       |       |       |       |       |       |
  |       |       |       |       |       |       |
  |       |   v   |       |       |       |       | <- Deep in 1-2 fields
  |       |       |       |       |       |       |    (specialization)
  |       |       |       |       |       |       |
  |       |       |       |       |       |       |
  +-------+-------+-------+-------+-------+-------+

  Phase 1 (Year 1): Extend the horizontal bar
    -> Broadly study fundamentals of all fields (this Skill)

  Phase 2 (Year 2+): Extend the vertical bar
    -> Choose 1-2 fields and dive deep (specialized books, papers)

  Phase 3 (Year 3+): Evolve into Pi-shaped
    -> Acquire a second area of specialization
```

### 14.2 The Compound Effect of Learning

CS learning has a compound effect. Even though it starts slow, learning accelerates as knowledge accumulates.

```
The compound effect of CS learning:

  Learning
  speed
    |                                        *
    |                                    *
    |                                *
    |                            *
    |                        *
    |                    *         <- Acceleration phase
    |               *                (knowledge interconnections increase)
    |          *
    |     *
    |  *                           <- Slow at first
    |*                               (building foundational concepts)
    +-------------------------------------------
                  Study time

  Example: Learning databases after studying algorithms
    -> B-tree is understood as "tree structure + extension of binary search"
    -> Index complexity can be analyzed immediately
    -> Knowledge connects exponentially
```

### 14.3 Systems for Continuous Learning

CS learning is not a one-time endeavor. Technology continues to evolve, requiring systems for continuous learning.

```
Systems for continuous learning:

  Daily (5-10 min):
    +-- Review with Anki (spaced repetition)
    +-- Read one tech blog/news article
    +-- Solve 1 LeetCode problem (Easy or Medium)

  Weekly (2-3 hours):
    +-- One section of this Skill or one textbook chapter
    +-- Implementation exercises
    +-- Weekly review

  Monthly (half day):
    +-- Monthly progress review
    +-- Adjust learning plan
    +-- Attend a study group or write a blog post

  Quarterly (1 day):
    +-- Knowledge inventory (redo Exercise 3)
    +-- Explore new fields
    +-- Reset learning goals

  Annually:
    +-- Attend a conference (YAPC, RubyKaigi, PyCon, etc.)
    +-- Review annual learning outcomes
    +-- Create next year's learning plan
```

---


## FAQ

### Q1: What is the most important point when studying this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory but by actually writing code and verifying behavior.

### Q2: What are common mistakes beginners make?

Skipping fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving to the next step.

### Q3: How is this knowledge applied in practice?

The knowledge from this topic is frequently applied in daily development work. It becomes particularly important during code reviews and architecture design.

---

## Summary

| Aspect | Recommendation |
|--------|---------------|
| Learning order | Progress step by step following the dependency graph |
| Level assessment | Objectively gauge your level with the checklist |
| Pace | 30 min to 2 hours daily, review with spaced repetition |
| Study allocation | Theory 30% + Practice 50% + Review 20% |
| Resources | Combine this Skill + CLRS/CS:APP + LeetCode |
| Output | Share your learning through blogs, OSS contributions, study groups |
| Anti-patterns | Avoid tutorial hell, theory paralysis, and the comparison trap |
| Long-term strategy | Build T-shaped skills and leverage the compound effect |
| Goal setting | Create a 3-month plan and check progress weekly |
| Consistency | Make learning a habit with daily, weekly, and monthly systems |

**The most important thing**: Don't aim for perfection -- start today. The greatest enemy of CS learning is the procrastination of "I'll start when I'm ready." The very moment you finish reading this guide is the best time to begin.

---

## Recommended Next Guides


---

## References

1. Oz, B. "Teach Yourself Computer Science." https://teachyourselfcs.com/ -- The definitive guide for self-taught CS curricula
2. MIT OpenCourseWare. https://ocw.mit.edu/ -- A platform offering all MIT lectures for free
3. Harvard CS50. https://cs50.harvard.edu/ -- The world-standard introductory CS course
4. Ebbinghaus, H. "Memory: A Contribution to Experimental Psychology." 1885. -- The original source on the forgetting curve
5. Feynman, R. P. "Surely You're Joking, Mr. Feynman!" 1985. -- Background on the Feynman Technique
6. Roediger, H. L. & Butler, A. C. "The Critical Role of Retrieval Practice in Long-Term Retention." Trends in Cognitive Sciences, 2011. -- Academic evidence for active recall
7. Brown, P. C. et al. "Make It Stick: The Science of Successful Learning." Harvard University Press, 2014. -- A comprehensive guide to evidence-based learning
8. Bruner, J. S. "The Process of Education." Harvard University Press, 1960. -- The proposal of spiral learning
9. Bjork, R. A. "Memory and Metamemory Considerations in the Training of Human Beings." In Metacognition: Knowing about Knowing, MIT Press, 1994. -- The concept of Desirable Difficulties
10. Cormen, T. H. et al. "Introduction to Algorithms (CLRS)." MIT Press, 4th Edition, 2022. -- The standard algorithms textbook
11. Bryant, R. E. & O'Hallaron, D. R. "Computer Systems: A Programmer's Perspective (CS:APP)." Pearson, 3rd Edition, 2015. -- The standard systems programming textbook
12. Kleppmann, M. "Designing Data-Intensive Applications." O'Reilly Media, 2017. -- A seminal work on distributed systems and data processing
