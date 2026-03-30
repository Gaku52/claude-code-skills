---
name: computer-science-fundamentals
description: A comprehensive guide covering the fundamentals of computer science. From hardware internals and data representation to algorithms, data structures, computation theory, programming paradigms, and software engineering basics — a systematic guide to all the CS foundations every engineer needs.
---

[日本語版](../../ja/01-cs-fundamentals/computer-science-fundamentals/SKILL.md)

# Computer Science Fundamentals

## Table of Contents

1. [Overview](#overview)
2. [When to Use](#when-to-use)
3. [Learning Roadmap](#learning-roadmap)
4. [Section Index](#section-index)
5. [Best Practices](#best-practices)
6. [Detailed Documentation](#detailed-documentation)

---

## Overview

This Skill provides **comprehensive coverage of computer science fundamentals**. Frameworks and libraries change every five years, but the CS foundations covered here are timeless knowledge that has remained relevant for over 50 years — they represent the core strength of an engineer.

### Coverage

- Hardware internals (CPU, memory, storage, GPU, I/O)
- Data representation (binary, character encoding, integers, floating point, compression)
- Algorithms (complexity analysis, sorting, searching, recursion, DP, graphs)
- Data structures (arrays, lists, stacks, queues, hash tables, trees, graphs)
- Computation theory (automata, formal languages, Turing machines, computability, P vs NP)
- Programming paradigms (imperative, functional, OOP, logic, multi-paradigm)
- Software engineering basics (development methodologies, version control, testing, debugging)
- Advanced topics (distributed systems, concurrency, security, AI/ML introduction)

### Prerequisites

- **None**. This Skill is designed as an entry point for learning computer science from scratch.
- Prior programming experience helps deepen understanding but is not required.

### Target Audience

- **Beginners**: Those who have just started programming and want to systematically learn CS fundamentals
- **Intermediate**: Those with practical experience who feel uncertain about CS fundamentals or are preparing for job interviews
- **Advanced**: Those looking to audit their knowledge, gain deeper theoretical understanding, or use as a reference for team education

---

## When to Use

### Automatically Referenced Cases

- Investigating performance issues (complexity analysis, memory usage analysis)
- Choosing data structures (deciding between arrays vs hash tables vs trees)
- Designing and optimizing algorithms
- Low-level issues (character encoding errors, floating-point precision, integer overflow)
- System design discussions (CAP theorem, consistency models)

### Cases for Manual Reference

- Starting systematic study of CS fundamentals
- Preparing for technical interviews
- Educating and mentoring team members
- When deep understanding of a specific CS concept is needed

---

## Learning Roadmap

### Phase 1: The Very Basics (1-2 months)

```
00-introduction → 01-hardware-basics → 02-data-representation
Understand "how a computer physically works"
```

### Phase 2: Algorithms and Data Structures (2-3 months)

```
03-algorithms-basics → 04-data-structures
Learn "how to solve problems efficiently"
```

### Phase 3: Theory and Application (2-3 months)

```
05-computation-theory → 06-programming-paradigms → 07-software-engineering-basics → 08-advanced-topics
Understand the "why" theoretically and apply it in practice
```

---

## Section Index

### 00 - Introduction

| # | File | Description |
|---|------|-------------|
| 00 | [overview.md](docs/00-introduction/00-overview.md) | Overview of Computer Science — definition, major fields, and why you should study it |
| 01 | [history-of-computing.md](docs/00-introduction/01-history-of-computing.md) | History of Computing — from the abacus to quantum computers |
| 02 | [why-learn-cs.md](docs/00-introduction/02-why-learn-cs.md) | Why Learn CS — concrete benefits and real-world failure cases from lack of CS knowledge |
| 03 | [learning-path.md](docs/00-introduction/03-learning-path.md) | CS Learning Roadmap — customized paths by goal with resource lists |

An introductory section that provides a bird's-eye view of what computer science is and its overall landscape. While tracing the history of CS, it explains with concrete examples why CS knowledge is indispensable for engineers. It also presents optimal learning paths tailored to the reader's level and goals, providing a roadmap for the entire Skill.

CS is not merely "programming." It is a discipline that mathematically addresses computability, efficiency, and correctness, with foundations spanning from hardware to software, theory to application. This section provides that big picture and serves as a signpost for the sections that follow.

### 01 - Hardware Basics

| # | File | Description |
|---|------|-------------|
| 00 | [cpu-architecture.md](docs/01-hardware-basics/00-cpu-architecture.md) | CPU Architecture — instruction cycle, pipelining, CISC vs RISC |
| 01 | [memory-hierarchy.md](docs/01-hardware-basics/01-memory-hierarchy.md) | Memory Hierarchy — cache, RAM, locality principles, virtual memory |
| 02 | [storage-systems.md](docs/01-hardware-basics/02-storage-systems.md) | Storage — HDD, SSD, NVMe, file systems, RAID |
| 03 | [motherboard-and-bus.md](docs/01-hardware-basics/03-motherboard-and-bus.md) | Motherboard and Bus — PCIe, USB, chipset, boot process |
| 04 | [gpu-and-parallel.md](docs/01-hardware-basics/04-gpu-and-parallel.md) | GPU and Parallel Computing — CUDA, OpenCL, AI training engines |
| 05 | [io-systems.md](docs/01-hardware-basics/05-io-systems.md) | I/O Systems — interrupts, DMA, device drivers |
| 06 | [pcb-and-circuits.md](docs/01-hardware-basics/06-pcb-and-circuits.md) | Electronic Circuits — transistors, logic gates, semiconductor manufacturing |
| 07 | [capacity-limits.md](docs/01-hardware-basics/07-capacity-limits.md) | Performance Limits and the Future — Moore's Law, quantum computing |

Software ultimately runs on hardware. Understanding how a CPU executes instructions, how memory is organized into hierarchies, and how storage persists data is the first step toward performance-conscious programming.

Understanding CPU pipelining reveals the cost of branch misprediction. Knowing the memory hierarchy enables writing cache-friendly code. Understanding GPU architecture clarifies how AI training acceleration works, and knowing I/O mechanisms reveals the significance of io_uring and DPDK. This section explains hardware from a programmer's perspective, providing knowledge directly applicable to real-world practice.

### 02 - Data Representation

| # | File | Description |
|---|------|-------------|
| 00 | [binary-and-number-systems.md](docs/02-data-representation/00-binary-and-number-systems.md) | Binary and Number Systems — bitwise operations, base conversion |
| 01 | [character-encoding.md](docs/02-data-representation/01-character-encoding.md) | Character Encoding — ASCII, Unicode, UTF-8, handling encoding issues |
| 02 | [integer-representation.md](docs/02-data-representation/02-integer-representation.md) | Integer Representation — two's complement, overflow, endianness |
| 03 | [floating-point.md](docs/02-data-representation/03-floating-point.md) | Floating-Point Numbers — IEEE 754, rounding errors, the 0.1+0.2 problem |
| 04 | [compression-algorithms.md](docs/02-data-representation/04-compression-algorithms.md) | Compression Algorithms — Huffman, LZ77, DEFLATE, JPEG/MP3 |
| 05 | [storage-capacity.md](docs/02-data-representation/05-storage-capacity.md) | Developing Intuition for Data Sizes — units, back-of-the-envelope calculations |
| 06 | [brain-vs-computer.md](docs/02-data-representation/06-brain-vs-computer.md) | Brain vs Computer — fundamental differences in information processing |

Inside a computer, all data is represented as 0s and 1s. Text, numbers, images, audio — they are all just bit sequences interpreted differently. This section covers everything about data representation.

Many real-world problems stem from misunderstanding data representation: why 0.1 + 0.2 does not equal 0.3, what causes character encoding errors, and integer overflow incidents (such as the Ariane 5 rocket explosion). By understanding IEEE 754 at the bit level and grasping the byte structure of UTF-8, you can understand and prevent these issues at their root.

### 03 - Algorithms Basics

| # | File | Description |
|---|------|-------------|
| 00 | what-is-algorithm.md | What Is an Algorithm — definition, representation methods, design approaches overview |
| 01 | complexity-analysis.md | Complexity Analysis — Big-O, Big-Omega, Big-Theta, amortized complexity |
| 02 | sorting-algorithms.md | Sorting Algorithms — from Bubble Sort to TimSort, lower bound of comparison sorts |
| 03 | searching-algorithms.md | Searching Algorithms — linear, binary, hash-based, string search |
| 04 | recursion-and-divide.md | Recursion and Divide-and-Conquer — call stack, Master Theorem |
| 05 | greedy-algorithms.md | Greedy Algorithms — activity selection, Huffman coding, Dijkstra's algorithm |
| 06 | dynamic-programming.md | Dynamic Programming — knapsack, LCS, edit distance |
| 07 | graph-algorithms.md | Graph Algorithms — BFS/DFS, shortest paths, MST, topological sort |

Algorithms are the heart of CS. The difference between O(n) and O(n^2) translates to "1 second vs 11.5 days" when data reaches 1 million entries. Whether you understand this difference and can choose the right algorithm is what separates engineers by skill level.

This section covers the five major topics — sorting, searching, recursion, dynamic programming, and graph algorithms — from both theoretical (complexity proofs) and practical (working code) perspectives. Each algorithm is explained internally in terms of "why this approach is efficient," promoting deep understanding rather than mere memorization.

### 04 - Data Structures

| # | File | Description |
|---|------|-------------|
| 00 | arrays-and-lists.md | Arrays and Linked Lists — dynamic arrays, skip lists |
| 01 | stacks-and-queues.md | Stacks and Queues — LIFO/FIFO, deque, priority queue |
| 02 | hash-tables.md | Hash Tables — collision resolution, Bloom filters, consistent hashing |
| 03 | trees-basics.md | Tree Basics — binary search trees, traversal, tries |
| 04 | balanced-trees.md | Balanced Trees — AVL trees, red-black trees, B-trees, B+ trees |
| 05 | heaps-and-priority.md | Heaps and Priority Queues — binary heaps, Fibonacci heaps |
| 06 | graphs.md | Graphs — adjacency matrix/list, Union-Find |
| 07 | advanced-structures.md | Advanced Data Structures — Bloom filters, LRU cache, ropes |

"Choosing the right data structure" is one of the most important decisions in programming. Arrays, hash tables, trees, graphs — each has its strengths and weaknesses, and whether you can make the optimal choice for a given problem determines the quality of your code.

Each data structure is covered in depth, including internal implementation details, time complexity, memory usage, and cache efficiency. Additionally, standard library implementations across multiple programming languages (Python, JavaScript, Java, Rust) are compared, providing knowledge immediately applicable in practice.

### 05 - Computation Theory

| # | File | Description |
|---|------|-------------|
| 00 | automata-theory.md | Automata Theory — DFA/NFA, regex engines |
| 01 | formal-languages.md | Formal Languages — Chomsky hierarchy, BNF, parsing |
| 02 | turing-machines.md | Turing Machines — mathematical definition of computation |
| 03 | computability.md | Computability — halting problem, undecidability |
| 04 | complexity-classes.md | Complexity Classes — P, NP, NP-complete, the P vs NP problem |
| 05 | information-theory.md | Information Theory — entropy, Shannon's theorem |

Computation theory is the deepest layer of CS, mathematically revealing "what is computable and what is not." Why regular expressions cannot match recursive patterns, why a perfect bug detector cannot be built — understanding these "impossibilities" fundamentally changes the depth of an engineer's thinking.

### 06 - Programming Paradigms

| # | File | Description |
|---|------|-------------|
| 00 | [imperative.md](docs/06-programming-paradigms/00-imperative.md) | Imperative Programming — procedural, structured, C language |
| 01 | functional.md | Functional Programming — pure functions, immutability, monads |
| 02 | object-oriented.md | Object-Oriented Programming — SOLID, design patterns |
| 03 | logic.md | Logic Programming — Prolog, declarative programming |
| 04 | multi-paradigm.md | Multi-Paradigm — Rust, Kotlin, TypeScript |

### 07 - Software Engineering Basics

| # | File | Description |
|---|------|-------------|
| 00 | development-lifecycle.md | Development Lifecycle — Waterfall, Agile, DevOps |
| 01 | version-control.md | Version Control — Git internals, branching strategies |
| 02 | testing-fundamentals.md | Testing Fundamentals — test pyramid, TDD, BDD |
| 03 | debugging-techniques.md | Debugging Techniques — scientific debugging, profiling |
| 04 | documentation-practices.md | Documentation — Docs as Code, ADR |

### 08 - Advanced Topics

| # | File | Description |
|---|------|-------------|
| 00 | distributed-systems-intro.md | Introduction to Distributed Systems — CAP theorem, Raft, microservices |
| 01 | concurrency-intro.md | Introduction to Concurrency — threads, deadlocks, async/await |
| 02 | security-intro.md | Introduction to Security — cryptography, TLS, OWASP Top 10 |
| 03 | ai-ml-intro.md | Introduction to AI/ML — machine learning taxonomy, neural networks, LLMs |

---

## Best Practices

### Studying CS

1. **Alternate between theory and practice** — Always verify with code after learning theory
2. **Always be conscious of complexity** — Build the habit of asking "O(?)'' for every piece of code
3. **Pursue the "why"** — Understand not just the technique but why it is correct
4. **Get hands-on** — Implement all major data structures and algorithms yourself at least once
5. **Connect to real-world practice** — Consider how the concepts you learn are used in actual products
6. **Deepen progressively** — First grasp the overview, then dive into deeper theory as needed
7. **Implement in multiple languages** — Writing the same algorithm in Python, C, and Rust deepens understanding
8. **Visualize** — Draw diagrams of data structures and algorithm behavior to aid comprehension
9. **Teach** — Aim for the level of understanding where you can explain what you have learned to others
10. **Keep going** — CS fundamentals are not a one-time study; their depth is infinite

### Anti-Patterns

1. **"CS is math, so it's impossible for me"** — The vast majority of CS fundamentals can be understood with high school-level math
2. **"I can program, so I don't need CS"** — This is the root cause of code that does not scale
3. **"I just need to keep up with the latest tech"** — Application without fundamentals is a house built on sand
4. **"Read the textbook from cover to cover"** — Start with what you need and learn practically
5. **"Just grind LeetCode"** — Fundamental understanding matters more than pattern memorization
6. **"Algorithms are not used in real work"** — You use them without realizing it
7. **"Theory is unnecessary"** — Without theory, you waste time attempting impossible problems
8. **"Once learned, never forgotten"** — Regular review and practice are essential
9. **"Understand perfectly before moving on"** — Move forward at 80% understanding and deepen later
10. **"Memorize everything"** — Understanding the principles eliminates the need for memorization

---

## Detailed Documentation

| Directory | Description | File Count |
|-----------|-------------|------------|
| `docs/00-introduction/` | Introduction, history, learning paths | 4 |
| `docs/01-hardware-basics/` | How hardware works | 8 |
| `docs/02-data-representation/` | Internal data representation | 7 |
| `docs/03-algorithms-basics/` | Algorithm fundamentals | 8 |
| `docs/04-data-structures/` | Data structures | 8 |
| `docs/05-computation-theory/` | Computation theory | 6 |
| `docs/06-programming-paradigms/` | Programming paradigms | 5 |
| `docs/07-software-engineering-basics/` | SE basics | 5 |
| `docs/08-advanced-topics/` | Advanced topics | 4 |
| **Total** | | **55** |

---

## Related Skills

| Skill | Relationship |
|-------|--------------|

---

## References

1. Cormen, T. H. et al. "Introduction to Algorithms" (CLRS). MIT Press, 4th Edition, 2022.
2. Bryant, R. E. & O'Hallaron, D. R. "Computer Systems: A Programmer's Perspective" (CS:APP). Pearson, 3rd Edition, 2015.
3. Patterson, D. A. & Hennessy, J. L. "Computer Organization and Design" (COD). Morgan Kaufmann, 6th Edition, 2020.
4. Sipser, M. "Introduction to the Theory of Computation". Cengage, 3rd Edition, 2012.
5. Abelson, H. & Sussman, G. J. "Structure and Interpretation of Computer Programs" (SICP). MIT Press, 2nd Edition, 1996.
6. Tanenbaum, A. S. "Modern Operating Systems". Pearson, 4th Edition, 2014.
7. Knuth, D. E. "The Art of Computer Programming". Addison-Wesley, Volumes 1-4A.
8. Shannon, C. E. "A Mathematical Theory of Communication". Bell System Technical Journal, 1948.
9. Turing, A. M. "On Computable Numbers, with an Application to the Entscheidungsproblem". 1936.
10. ACM/IEEE. "Computing Curricula 2020". ACM, 2020.
11. MIT OpenCourseWare. "6.006 Introduction to Algorithms". https://ocw.mit.edu/
12. Stanford CS Library. https://cs.stanford.edu/
