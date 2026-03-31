# Computational Complexity Theory

> The P vs NP problem is the greatest unsolved problem in CS, and its answer has the potential to fundamentally change the foundations of cryptography, optimization, and AI.

## What You Will Learn in This Chapter

- [ ] Explain the definitions of P, NP, NP-Complete, and NP-Hard
- [ ] Understand the meaning and consequences of the P vs NP problem
- [ ] Know practical approaches for dealing with NP-Complete problems
- [ ] Understand the outline of the proof of the Cook-Levin theorem
- [ ] Grasp the major complexity classes (PSPACE, BPP, BQP, etc.)
- [ ] Understand approximation algorithms and inapproximability
- [ ] Know the basics of parameterized complexity (FPT)
- [ ] Understand the concept of space complexity and important theorems


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding of the content in [Computability](./01-computability.md)

---

## 1. Foundations of Computational Complexity Theory

### 1.1 Why Consider Complexity

```
Difference between computability and computational complexity:

  Computability theory:
  - Question: "Can this problem be solved in principle?"
  - Answer: Decidable / Undecidable
  - Does not consider constraints on time or memory

  Computational complexity theory:
  - Question: "Can this problem be solved efficiently?"
  - Answer: Polynomial time / Exponential time / etc.
  - Considers constraints on computational resources (time, space)

  Practical importance:
  ┌────────────────────────────────────────────┐
  │ For problem size n = 100:                  │
  │                                            │
  │ O(n)      = 100        → Instantaneous     │
  │ O(n²)     = 10,000     → Instantaneous     │
  │ O(n³)     = 1,000,000  → About 0.001 sec   │
  │ O(n⁵)     = 10^10      → About 10 sec      │
  │ O(2^n)    = 10^30      → Exceeds the age   │
  │                          of the universe    │
  │ O(n!)     = 10^158     → Completely         │
  │                          impossible         │
  └────────────────────────────────────────────┘

  → The difference between polynomial and exponential time
    is practically the difference between "solvable" and "unsolvable"
```

### 1.2 Formal Definition of Time Complexity

```
Formal definition of time complexity:

  Definition: Time complexity f(n) of a Turing machine M:
  M halts within f(n) steps for any input of length n

  Asymptotic evaluation:
  - O notation (upper bound): f(n) = O(g(n)) ⟺ ∃c,n₀: f(n) ≤ c·g(n) for n ≥ n₀
  - Ω notation (lower bound): f(n) = Ω(g(n)) ⟺ ∃c,n₀: f(n) ≥ c·g(n) for n ≥ n₀
  - Θ notation (tight):       f(n) = Θ(g(n)) ⟺ f(n) = O(g(n)) and f(n) = Ω(g(n))

  Time complexity classes:
  TIME(f(n)) = { L | there exists a TM that decides L in O(f(n)) time }

  P = ∪_{k≥0} TIME(n^k)
    = TIME(n) ∪ TIME(n²) ∪ TIME(n³) ∪ ...

  EXPTIME = ∪_{k≥0} TIME(2^{n^k})

  Note: The time complexity of a multi-tape TM is within the square
  of a single-tape TM
  → The choice of computational model does not affect class P
    (absorbed within polynomial bounds)
```

---

## 2. Complexity Classes

### 2.1 Class P

```
P (Polynomial time):
  The class of problems that can be "solved" in polynomial time

  Formal definition:
  P = { L | for some k ≥ 0, L ∈ TIME(n^k) }

  Examples of problems in P:

  ┌─────────────────────────────────────────────────┐
  │ Problem             │ Algorithm          │ Complexity  │
  ├───────────────────┼──────────────────┼──────────┤
  │ Sorting             │ Merge sort         │ O(n log n) │
  │ Shortest path       │ Dijkstra's         │ O(V² + E)  │
  │ Maximum flow        │ Ford-Fulkerson     │ O(VE²)     │
  │ Primality testing   │ AKS algorithm      │ O(n^6)     │
  │ Linear programming  │ Ellipsoid method   │ Polynomial │
  │ Maximum matching    │ Edmonds' algorithm │ O(V³)      │
  │ 2-SAT              │ Implication        │ O(n + m)   │
  │                     │ graph + SCC        │            │
  │ 2-colorability      │ BFS/DFS           │ O(V + E)   │
  │ Connectivity        │ BFS/DFS           │ O(V + E)   │
  │ Minimum spanning    │ Kruskal / Prim    │ O(E log V) │
  │ tree                │                    │            │
  │ Determinant         │ Gaussian           │ O(n³)      │
  │ computation         │ elimination        │            │
  │ Pattern matching    │ KMP algorithm      │ O(n + m)   │
  └─────────────────────────────────────────────────┘

  Meaning of P:
  - Theoretical definition of "efficiently solvable"
  - In practice, even O(n⁵) is often too slow
  - However, theoretically the boundary between polynomial
    and exponential is essential
```

```python
# Implementation examples of problems in P

# 1. 2-SAT (a satisfiability problem solvable in polynomial time)
from collections import defaultdict

class TwoSAT:
    """
    Solution for the 2-SAT problem
    Time complexity: O(n + m) (SCC decomposition on the implication graph)

    2-SAT belongs to P (solvable in polynomial time)
    whereas 3-SAT is NP-Complete
    """

    def __init__(self, n):
        """n: number of variables"""
        self.n = n
        self.graph = defaultdict(list)
        self.reverse_graph = defaultdict(list)

    def _neg(self, x):
        """Return the negation of variable x"""
        return x + self.n if x < self.n else x - self.n

    def add_clause(self, x, y):
        """
        Add clause (x ∨ y)
        Implication: ¬x → y, ¬y → x
        """
        neg_x = self._neg(x)
        neg_y = self._neg(y)
        self.graph[neg_x].append(y)
        self.graph[neg_y].append(x)
        self.reverse_graph[y].append(neg_x)
        self.reverse_graph[x].append(neg_y)

    def solve(self):
        """
        2-SAT solution using SCC decomposition

        Returns: List of truth values for each variable if satisfiable,
                 None if unsatisfiable
        """
        # Use Kosaraju's algorithm to find topological order
        visited = set()
        order = []

        def dfs1(v):
            visited.add(v)
            for u in self.graph[v]:
                if u not in visited:
                    dfs1(u)
            order.append(v)

        for v in range(2 * self.n):
            if v not in visited:
                dfs1(v)

        # SCC decomposition on the reverse graph
        comp = [-1] * (2 * self.n)
        comp_id = 0

        def dfs2(v, c):
            comp[v] = c
            for u in self.reverse_graph[v]:
                if comp[u] == -1:
                    dfs2(u, c)

        for v in reversed(order):
            if comp[v] == -1:
                dfs2(v, comp_id)
                comp_id += 1

        # Satisfiability check
        for i in range(self.n):
            if comp[i] == comp[i + self.n]:
                return None  # x and ¬x in the same SCC → unsatisfiable

        # Solution construction
        result = []
        for i in range(self.n):
            result.append(comp[i] > comp[i + self.n])

        return result


# Usage example
sat = TwoSAT(3)  # 3 variables: x₀, x₁, x₂
# (x₀ ∨ x₁) ∧ (¬x₀ ∨ x₂) ∧ (¬x₁ ∨ ¬x₂)
sat.add_clause(0, 1)      # x₀ ∨ x₁
sat.add_clause(3, 2)      # ¬x₀ ∨ x₂  (3 = ¬0)
sat.add_clause(4, 5)      # ¬x₁ ∨ ¬x₂  (4 = ¬1, 5 = ¬2)

solution = sat.solve()
print(f"Solution: {solution}")  # [True, True, False] etc.


# 2. Maximum bipartite matching (Hopcroft-Karp)
class BipartiteMatching:
    """
    Maximum matching in a bipartite graph
    Hopcroft-Karp algorithm: O(E√V)
    → Belongs to P
    """

    def __init__(self, n, m):
        """n: number of left vertices, m: number of right vertices"""
        self.n = n
        self.m = m
        self.graph = defaultdict(list)
        self.match_left = [-1] * n
        self.match_right = [-1] * m

    def add_edge(self, u, v):
        """Add an edge from left vertex u to right vertex v"""
        self.graph[u].append(v)

    def bfs(self):
        """Compute augmenting path lengths using BFS"""
        from collections import deque
        dist = [-1] * self.n
        queue = deque()

        for u in range(self.n):
            if self.match_left[u] == -1:
                dist[u] = 0
                queue.append(u)

        found = False
        while queue:
            u = queue.popleft()
            for v in self.graph[u]:
                w = self.match_right[v]
                if w == -1:
                    found = True
                elif dist[w] == -1:
                    dist[w] = dist[u] + 1
                    queue.append(w)

        self.dist = dist
        return found

    def dfs(self, u):
        """Find augmenting paths using DFS and update the matching"""
        for v in self.graph[u]:
            w = self.match_right[v]
            if w == -1 or (self.dist[w] == self.dist[u] + 1 and self.dfs(w)):
                self.match_left[u] = v
                self.match_right[v] = u
                return True
        self.dist[u] = -1
        return False

    def solve(self):
        """Find the maximum matching"""
        matching = 0
        while self.bfs():
            for u in range(self.n):
                if self.match_left[u] == -1:
                    if self.dfs(u):
                        matching += 1
        return matching
```

### 2.2 Class NP

```
NP (Nondeterministic Polynomial time):
  The class of problems whose solutions can be "verified in polynomial time"

  Formal definition (verifier-based):
  L ∈ NP ⟺ There exists a TM V (verifier) and polynomial p such that:
    - w ∈ L ⟺ there exists c (|c| ≤ p(|w|)) such that V(w, c) = accept
    - V runs in polynomial time
    - c is called a "certificate", "proof", or "witness"

  Formal definition (nondeterministic TM-based):
  NP = ∪_{k≥0} NTIME(n^k)
  NTIME(f(n)) = set of languages accepted by a nondeterministic TM
                within f(n) steps

  Intuitive understanding of NP:
  ┌────────────────────────────────────────────────────┐
  │ "Finding a solution may be hard, but given a       │
  │  solution, verifying its correctness is quick"     │
  │                                                    │
  │  Sudoku: Solving is hard, but verifying a          │
  │          completed grid is easy                    │
  │  Factoring: Finding factors is hard, but           │
  │            verifying multiplication is easy         │
  │  Puzzles: Solving is hard, but checking            │
  │           the answer is easy                       │
  └────────────────────────────────────────────────────┘

  Examples of problems in NP:

  Problem            │ Witness (certificate)    │ Verification method
  ─────────────────┼────────────────────────┼──────────────
  SAT               │ Variable assignment      │ Evaluate each clause
  Hamiltonian cycle  │ Permutation of vertices  │ Check all edges exist
  Graph coloring     │ Color for each vertex    │ No adjacent vertices
                     │                          │ share a color
  Knapsack           │ Set of chosen items      │ Sum weights and values
  Clique             │ Vertex subset            │ All pairs connected
  TSP                │ City visit order         │ Compute total distance
  Subset sum         │ Element subset           │ Compute sum
  Integer program    │ Integer assignment to    │ Check constraints
                     │ variables                │
```

```python
# Implementation examples of NP problem verifiers

class NPVerifier:
    """Collection of verifiers for NP problems"""

    @staticmethod
    def verify_sat(formula, assignment):
        """
        Verifier for SAT (satisfiability problem)
        formula: CNF formula [[1, -2, 3], [-1, 3], ...]
                 positive numbers = variables, negative = negation
        assignment: {variable_number: True/False}

        Verification complexity: O(n × m) (n=variables, m=clauses)
        → Verifiable in polynomial time → belongs to NP
        """
        for clause in formula:
            satisfied = False
            for literal in clause:
                var = abs(literal)
                value = assignment.get(var, False)
                if literal > 0 and value:
                    satisfied = True
                    break
                if literal < 0 and not value:
                    satisfied = True
                    break
            if not satisfied:
                return False
        return True

    @staticmethod
    def verify_hamiltonian_cycle(graph, cycle):
        """
        Verifier for Hamiltonian cycle
        graph: adjacency list {vertex: [adjacent vertices, ...]}
        cycle: list of vertices

        Verification: O(V)
        """
        n = len(graph)

        # Check that all vertices are visited exactly once
        if len(cycle) != n:
            return False
        if set(cycle) != set(graph.keys()):
            return False

        # Check that each edge exists
        for i in range(n):
            u = cycle[i]
            v = cycle[(i + 1) % n]
            if v not in graph[u]:
                return False

        return True

    @staticmethod
    def verify_graph_coloring(graph, coloring, k):
        """
        Verifier for k-coloring
        graph: adjacency list
        coloring: {vertex: color}
        k: number of available colors

        Verification: O(V + E)
        """
        # Check that the number of colors is at most k
        if len(set(coloring.values())) > k:
            return False

        # Check that all vertices are assigned a color
        if set(coloring.keys()) != set(graph.keys()):
            return False

        # Check that no adjacent vertices share the same color
        for u in graph:
            for v in graph[u]:
                if coloring[u] == coloring[v]:
                    return False

        return True

    @staticmethod
    def verify_subset_sum(numbers, target, subset):
        """
        Verifier for the subset sum problem
        numbers: set of numbers
        target: target sum
        subset: set of indices of chosen elements

        Verification: O(n)
        """
        # Check that subset is a valid subset of numbers
        if not all(0 <= i < len(numbers) for i in subset):
            return False

        # Check that the sum equals the target
        return sum(numbers[i] for i in subset) == target

    @staticmethod
    def verify_clique(graph, clique, k):
        """
        Verifier for k-clique
        graph: adjacency list
        clique: set of vertices
        k: clique size

        Verification: O(k²)
        """
        if len(clique) != k:
            return False

        # Check that all pairs are connected by an edge
        clique_list = list(clique)
        for i in range(len(clique_list)):
            for j in range(i + 1, len(clique_list)):
                u, v = clique_list[i], clique_list[j]
                if v not in graph.get(u, []):
                    return False
        return True


# Verification demonstration
verifier = NPVerifier()

# SAT verification
formula = [[1, -2, 3], [-1, 3], [2, -3]]  # (x₁ ∨ ¬x₂ ∨ x₃) ∧ ...
assignment = {1: True, 2: False, 3: True}
print(f"SAT verification: {verifier.verify_sat(formula, assignment)}")  # True

# Subset sum verification
numbers = [3, 7, 1, 8, 4]
target = 12
subset = {0, 2, 3}  # 3 + 1 + 8 = 12
print(f"Subset sum verification: {verifier.verify_subset_sum(numbers, target, subset)}")
```

### 2.3 NP-Complete and NP-Hard

```
NP-Complete:
  The class of "hardest" problems in NP

  Formal definition:
  L is NP-Complete ⟺
    1. L ∈ NP
    2. For every L' ∈ NP, L' ≤ₚ L
       (all NP problems are polynomial-time reducible to L)

  If one NP-Complete problem can be solved in polynomial time:
  → All NP problems can be solved in polynomial time → P = NP

  NP-Hard:
  L is NP-Hard ⟺
    For every L' ∈ NP, L' ≤ₚ L

  NP-Complete = NP ∩ NP-Hard

  ┌──────────────────────────────────────────────────────┐
  │                                                      │
  │  Venn diagram assuming P ≠ NP:                       │
  │                                                      │
  │  ┌─────────────── NP-Hard ──────────────────────┐   │
  │  │                                              │   │
  │  │     ┌──────────── NP ────────────────┐      │   │
  │  │     │                                │      │   │
  │  │     │  ┌── P ──┐  ┌─── NP-Complete ─┐│     │   │
  │  │     │  │Sorting │  │ SAT            ││     │   │
  │  │     │  │Shortest│  │ TSP (decision) ││     │   │
  │  │     │  │path    │  │ Graph coloring ││     │   │
  │  │     │  └────────┘  └────────────────┘│     │   │
  │  │     │     NP-Intermediate (if exists) │      │   │
  │  │     │     - Graph isomorphism         │      │   │
  │  │     │     - Integer factorization     │      │   │
  │  │     └────────────────────────────────┘      │   │
  │  │  ┌─── NP-Hard not in NP ─┐                 │   │
  │  │  │  Halting problem       │                 │   │
  │  │  │  QBF (PSPACE-Complete) │                 │   │
  │  │  └───────────────────────┘                  │   │
  │  └──────────────────────────────────────────────┘   │
  │                                                      │
  └──────────────────────────────────────────────────────┘

  NP-Intermediate:
  If P ≠ NP, there exist problems that are neither NP-Complete
  nor in P (Ladner's theorem)
  Candidates:
  - Graph isomorphism: unknown whether NP-Complete or in P
  - Integer factorization: not believed to be NP-Complete
    (in P for quantum computers)
  - Discrete logarithm: similar status to integer factorization
```

### 2.4 Cook-Levin Theorem

```
Cook-Levin Theorem:
  SAT (satisfiability problem) is NP-Complete

  Significance:
  - First proof of an NP-Complete problem
  - Afterward, NP-Completeness of other problems is proved by
    reduction from SAT

  Proof outline:

  1. SAT ∈ NP: Given an assignment, it can be verified in
     polynomial time ✓

  2. Show L ≤ₚ SAT for every NP problem L:

  Idea:
  - Since L ∈ NP, a polynomial-time verifier V exists
  - Encode the computation of V(w, c) as a Boolean formula
  - Represent tape contents, head position, and state at each
    step as variables
  - Express transition function constraints as clauses

  Variables to construct:
  - x_{i,j,s}: symbol s is at tape position j at time i
  - h_{i,j}: head is at position j at time i
  - q_{i,s}: state is s at time i

  Clauses to construct:
  - Initial conditions: input w is written on the tape
  - Transition conditions: each step follows the transition function
  - Acceptance condition: an accepting state is eventually reached

  Size of the resulting formula: O(t(n)² × |Σ|) ← polynomial

  → Any NP problem can be transformed to SAT in polynomial time
  → SAT is NP-Complete ∎
```

### 2.5 Representative NP-Complete Problems and the Chain of Reductions

```
Chain of reductions among NP-Complete problems:

  SAT (Cook-Levin Theorem)
   │
   ├──→ 3-SAT
   │     │
   │     ├──→ Independent Set ──→ Vertex Cover ──→ Set Cover
   │     │
   │     ├──→ 3-Coloring ──→ k-Coloring (k ≥ 3)
   │     │
   │     ├──→ Clique
   │     │
   │     ├──→ Hamiltonian Cycle ──→ TSP (decision version)
   │     │
   │     └──→ Subset Sum ──→ Knapsack ──→ Partition
   │
   └──→ Circuit Satisfiability (Circuit-SAT)

  Key points about reductions:
  - A ≤ₚ B (A reduces to B): if B is solvable, then A is also solvable
  - To show NP-Completeness of a new problem X:
    1. Show X ∈ NP
    2. Show Y ≤ₚ X for a known NP-Complete problem Y
```

```python
# Implementation examples of reductions between NP-Complete problems

# Reduction 1: 3-SAT → Independent Set
def reduce_3sat_to_independent_set(clauses, num_vars):
    """
    Reduce 3-SAT to the independent set problem

    Input: 3-CNF formula (each clause has at most 3 literals)
    Output: Graph G and integer k

    (formula ∈ 3-SAT) ⟺ (G has independent set of size k)
    """
    # Create one vertex for each literal in each clause
    vertices = []
    for i, clause in enumerate(clauses):
        for j, literal in enumerate(clause):
            vertices.append((i, j, literal))

    k = len(clauses)  # Independent set size = number of clauses

    # Edge construction
    edges = []
    for idx1, v1 in enumerate(vertices):
        for idx2, v2 in enumerate(vertices):
            if idx1 >= idx2:
                continue
            # Add edge between literals in the same clause
            # (choose only one from each clause)
            if v1[0] == v2[0]:
                edges.append((idx1, idx2))
            # Add edge between contradictory literals (x and ¬x)
            elif v1[2] == -v2[2]:
                edges.append((idx1, idx2))

    return vertices, edges, k


# Reduction 2: Independent Set → Vertex Cover
def reduce_independent_set_to_vertex_cover(graph, n, k):
    """
    Reduce independent set to vertex cover

    G has an independent set of size k ⟺ G has a vertex cover of size (n-k)

    Proof:
    S is an independent set of G → V\S is a vertex cover
    (S has no edge connecting two vertices in S → every edge
     has at least one endpoint in V\S)
    """
    # Graph stays the same, transform the target size
    return graph, n - k


# Reduction 3: Vertex Cover → Set Cover
def reduce_vertex_cover_to_set_cover(graph, n, k):
    """
    Reduce vertex cover to set cover

    Universe U = set of edges
    For each vertex v, define Sv as the set of edges incident to v
    Cover all edges with at most k sets Sv ⟺ vertex cover of size ≤ k exists
    """
    # Universe = set of edges
    universe = set()
    for u in graph:
        for v in graph[u]:
            edge = (min(u, v), max(u, v))
            universe.add(edge)

    # Set for each vertex
    sets = {}
    for u in graph:
        sets[u] = set()
        for v in graph[u]:
            edge = (min(u, v), max(u, v))
            sets[u].add(edge)

    return universe, sets, k


# Reduction 4: Subset Sum → Partition
def reduce_subset_sum_to_partition(numbers, target):
    """
    Reduce subset sum to the partition problem

    Partition problem: Can set S be divided into two subsets
    such that both have equal sum?

    Reduction: S = numbers ∪ {|sum(numbers) - 2*target|}
    """
    total = sum(numbers)
    diff = abs(total - 2 * target)
    new_numbers = numbers + [diff]

    # new_numbers can be partitioned into two equal-sum subsets
    # ⟺ a subset of numbers sums to target
    return new_numbers
```

---

## 3. The P vs NP Problem

### 3.1 Meaning and Consequences of the Problem

```
The P vs NP Problem:

  Question: Does P = NP?

  Clay Mathematics Institute Millennium Prize Problem ($1,000,000)

  If P = NP were proved:

  ┌─────────────────────────────────────────────┐
  │ Collapse of cryptography                     │
  │ - RSA, elliptic curve cryptography broken    │
  │ - All online security becomes invalid        │
  │ - New cryptographic systems needed            │
  ├─────────────────────────────────────────────┤
  │ Revolution in optimization                   │
  │ - Scheduling, route optimization perfectly   │
  │   solvable                                   │
  │ - Drug discovery: perfect protein structure  │
  │   prediction                                 │
  │ - Complete optimization of logistics and     │
  │   manufacturing                              │
  ├─────────────────────────────────────────────┤
  │ Breakthrough in AI/mathematics               │
  │ - Automatic theorem proving becomes feasible │
  │ - NP search = finding optimal solutions      │
  │   becomes easy                               │
  │ - The definition of creativity is shaken     │
  └─────────────────────────────────────────────┘

  If P ≠ NP were proved:

  ┌─────────────────────────────────────────────┐
  │ Reassuring confirmation                      │
  │ - Cryptography is (theoretically) secure     │
  │ - NP-Complete problems have inherent         │
  │   difficulty                                 │
  │ - Research on approximation and heuristics   │
  │   becomes important                          │
  ├─────────────────────────────────────────────┤
  │ New theoretical tools                        │
  │ - Proof techniques applicable to other       │
  │   unsolved problems                          │
  │ - Theory of computational complexity         │
  │   advances greatly                           │
  └─────────────────────────────────────────────┘

  Current status:
  - Most researchers expect P ≠ NP (Gasarch's survey: about 83%)
  - However, there is no clear path to a proof
  - Proving P ≠ NP requires new mathematical techniques
  - Existing techniques (diagonalization, relativization) are insufficient
    → Baker-Gill-Solovay theorem (1975)
  - Natural proofs are also insufficient
    → Razborov-Rudich theorem (1997)
```

### 3.2 Barriers to Proving P ≠ NP

```
Why proving P ≠ NP is so difficult:

  1. Relativization Barrier
     Baker-Gill-Solovay (1975):
     - For some oracle A, P^A = NP^A
     - For another oracle B, P^B ≠ NP^B
     → "Relativizing" techniques like diagonalization cannot
       prove or disprove P ≠ NP

  2. Natural Proofs Barrier
     Razborov-Rudich (1997):
     - If one-way functions exist (cryptographic assumption)
     - "Natural proofs" cannot show circuit lower bounds
     → Many conventional techniques cannot be used

  3. Algebrization Barrier
     Aaronson-Wigderson (2009):
     - Even extensions of diagonalization (allowing algebraic
       extensions) are insufficient
     → Even newer techniques are needed

  Promising approaches:
  - Geometric Complexity Theory (GCT): uses algebraic geometry
  - Circuit lower bounds research: lower bounds for specific
    circuit models
  - Communication complexity: proofs of related lower bounds
  - However, a complete proof is still far away

  Current partial results:
  - P ≠ EXPTIME (time hierarchy theorem)
  - NP ⊄ SIZE(n^k) for any fixed k (Kannan, 1982)
  - Circuit lower bounds for ACC⁰ (Williams, 2011)
```

---

## 4. Other Complexity Classes

### 4.1 Space Complexity

```
Space complexity:

  SPACE(f(n)) = { L | there exists a TM that decides L using O(f(n)) space }
  NSPACE(f(n)) = space complexity for nondeterministic TMs

  Major classes:
  ┌────────────────────────────────────────────────────┐
  │ L (LOGSPACE) = SPACE(log n)                        │
  │   Example: reachability in directed graphs          │
  │   Example: pattern matching                         │
  │                                                    │
  │ NL = NSPACE(log n)                                 │
  │   Example: reachability in directed graphs          │
  │   NL = co-NL (Immerman-Szelepcsényi theorem)      │
  │                                                    │
  │ PSPACE = ∪_{k≥0} SPACE(n^k)                       │
  │   Example: QBF (quantified Boolean formulas)        │
  │   Example: generalized chess, Go                    │
  │   PSPACE-Complete problem: QBF                      │
  │                                                    │
  │ EXPSPACE = ∪_{k≥0} SPACE(2^{n^k})                 │
  └────────────────────────────────────────────────────┘

  Containment relationships (known):

  L ⊆ NL ⊆ P ⊆ NP ⊆ PSPACE ⊆ EXPTIME ⊆ EXPSPACE

  Unknown:
  - L = NL? (open)
  - P = NP? (open)
  - NP = PSPACE? (open)
  - However, L ≠ PSPACE and P ≠ EXPTIME have been proved

  Savitch's Theorem:
  NSPACE(f(n)) ⊆ SPACE(f(n)²)
  → Nondeterminism only squares the space requirement
  → PSPACE = NPSPACE
```

```python
# Examples of space-efficient algorithms

# Reachability test based on Savitch's theorem
def reachability_savitch(graph, start, end, n):
    """
    Reachability test based on Savitch's theorem
    Space complexity: O(log²n) (recursion depth × space per level)

    Idea: "Can we reach from start to end in at most 2^i steps?"
    Recursively check by trying all intermediate points mid
    """
    def can_reach(u, v, steps):
        """Can we reach from u to v in at most steps steps?"""
        if steps == 0:
            return u == v
        if steps == 1:
            return u == v or v in graph.get(u, [])

        half = steps // 2
        # Try all intermediate points
        for mid in range(n):
            if can_reach(u, mid, half) and can_reach(mid, v, steps - half):
                return True
        return False

    return can_reach(start, end, n)


# PSPACE-Complete: QBF (Quantified Boolean Formula)
def solve_qbf(formula):
    """
    Evaluation of quantified Boolean formulas
    PSPACE-Complete problem

    Example: ∀x ∃y (x ∨ y) ∧ (¬x ∨ ¬y)

    Can be solved in polynomial space (space-efficient)
    but may take exponential time
    """
    if not formula.quantifiers:
        # When no quantifiers remain, evaluate the Boolean expression
        return evaluate_boolean(formula.expression, formula.assignment)

    quantifier, variable = formula.quantifiers[0]
    remaining = formula.without_first_quantifier()

    if quantifier == 'forall':
        # ∀x: must be true for both x=True and x=False
        remaining.assignment[variable] = True
        result_true = solve_qbf(remaining)
        remaining.assignment[variable] = False
        result_false = solve_qbf(remaining)
        return result_true and result_false

    elif quantifier == 'exists':
        # ∃x: must be true for either x=True or x=False
        remaining.assignment[variable] = True
        result_true = solve_qbf(remaining)
        if result_true:
            return True
        remaining.assignment[variable] = False
        return solve_qbf(remaining)
```

### 4.2 Probabilistic Complexity Classes

```
Probabilistic complexity classes:

  BPP (Bounded-Error Probabilistic Polynomial time):
  - Solvable by a randomized TM in polynomial time
  - Correct answer probability ≥ 2/3 (can be improved exponentially
    by repeating a constant number of times)
  - P ⊆ BPP ⊆ PSPACE

  RP (Randomized Polynomial time):
  - YES: accept with probability ≥ 1/2
  - NO: reject with probability 1 (no one-sided error)
  - Example: Miller-Rabin primality test (correctly identifies
    composite numbers)

  ZPP (Zero-Error Probabilistic Polynomial time):
  - No errors, expected running time is polynomial
  - ZPP = RP ∩ co-RP

  PP (Probabilistic Polynomial time):
  - Correct answer probability > 1/2 (even barely is OK)
  - PP is very powerful (#P ⊆ P^{PP})

  Relationship diagram:
  P ⊆ ZPP ⊆ RP ⊆ BPP ⊆ PP ⊆ PSPACE

  Practical importance:
  - Many practical algorithms belong to BPP
  - Randomness increases "apparent computational power"
  - It is conjectured that BPP = P (derandomization hypothesis)
```

```python
# Examples of probabilistic algorithms

import random

# RP example: Miller-Rabin primality test
def miller_rabin(n, k=20):
    """
    Miller-Rabin primality test
    Belongs to RP:
    - If n is composite → answers "composite" with probability ≥ 1 - 4^{-k}
    - If n is prime → answers "prime" with probability 1
    """
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False

    # n - 1 = 2^r × d (d is odd)
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)

        if x == 1 or x == n - 1:
            continue

        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False  # Composite

    return True  # Probably prime


# BPP example: Randomized minimum cut (Karger's algorithm)
def karger_min_cut(graph, n):
    """
    Karger's randomized minimum cut algorithm
    Belongs to BPP

    Probability of finding the minimum cut in one run: ≥ 2/n²
    Repeating n²ln(n) times gives high probability of success
    """
    import copy

    # Edge contraction
    vertices = list(range(n))
    edges = copy.deepcopy(graph)  # List of (u, v)

    # Union-Find
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        parent[px] = py

    remaining = n
    while remaining > 2:
        # Randomly select an edge and contract
        while True:
            u, v = random.choice(edges)
            if find(u) != find(v):
                break

        union(u, v)
        remaining -= 1

    # Count edges between the remaining 2 super-vertices
    cut_size = 0
    for u, v in edges:
        if find(u) != find(v):
            cut_size += 1

    return cut_size


# ZPP example: Randomized quicksort
def randomized_quicksort(arr):
    """
    Randomized quicksort
    Belongs to ZPP: always correct result, expected O(n log n) time
    Worst case O(n²) but with very low probability
    """
    if len(arr) <= 1:
        return arr

    pivot = random.choice(arr)
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return randomized_quicksort(left) + middle + randomized_quicksort(right)
```

### 4.3 Quantum Complexity Classes

```
Quantum complexity classes:

  BQP (Bounded-Error Quantum Polynomial time):
  - Polynomial time on a quantum computer, error probability ≤ 1/3
  - P ⊆ BPP ⊆ BQP ⊆ PSPACE

  Problems efficiently solvable in BQP:
  - Integer factorization (Shor's algorithm)
  - Discrete logarithm
  - Simulation of unitary matrices
  - Certain algebraic problems

  Problems conjectured to be intractable even in BQP:
  - NP-Complete problems (SAT, TSP, etc.)
  - However, this has not been proved

  Relationship diagram:
  ┌──────────────────────────────────────────┐
  │                PSPACE                     │
  │  ┌────────────────────────────────────┐  │
  │  │              BQP                    │  │
  │  │  ┌──────────────────────────────┐  │  │
  │  │  │           BPP                 │  │  │
  │  │  │  ┌──────────────────────┐    │  │  │
  │  │  │  │         P             │    │  │  │
  │  │  │  └──────────────────────┘    │  │  │
  │  │  └──────────────────────────────┘  │  │
  │  └────────────────────────────────────┘  │
  │                                          │
  │    NP is conjectured to be in a different │
  │    position from BQP                      │
  │    NP ⊄ BQP and BQP ⊄ NP (conjectured)  │
  └──────────────────────────────────────────┘

  Quantum supremacy:
  - 2019: Google claimed quantum supremacy with 53 qubits
  - Experimental demonstration of outperforming classical
    computers on a specific task
  - However, still limited for practical problems
```

### 4.4 Counting Problem Classes

```
Counting problem classes:

  #P (Sharp-P):
  - Not the decision version of NP, but counting the "number" of solutions
  - Example: #SAT = number of satisfying assignments
  - Example: number of perfect matchings (permanent computation)

  #P-Complete problems:
  - Permanent computation (Valiant, 1979)
  - #SAT
  - Graph coloring polynomial (evaluation of chromatic polynomial)

  Important relationships:
  - P ⊆ NP ⊆ P^{#P}
  - Toda's theorem: PH ⊆ P^{#P}
    → #P contains the entire polynomial hierarchy (very powerful)

  Approximate counting:
  - Even when exact counting is hard, approximate counting can
    sometimes be done efficiently
  - FPRAS (Fully Polynomial Randomized Approximation Scheme)
  - Example: approximate counting of DNF satisfying assignments
```

---

## 5. Practical Approaches

### 5.1 Strategies for Dealing with NP-Complete Problems

```
Decision flow when facing an NP-Complete/NP-Hard problem:

  ┌──────────────────────────────┐
  │ Problem identified as        │
  │ NP-Complete/NP-Hard          │
  └──────────┬───────────────────┘
             ↓
  ┌──────────────────────────────┐
  │ Is the input size small?     │
  │ (n ≤ 20~25 or so)           │
  └────┬──────────┬──────────────┘
       │ YES      │ NO
       ↓          ↓
  ┌─────────┐  ┌──────────────────────────┐
  │ Find    │  │ Is there special         │
  │ exact   │  │ structure?               │
  │ solution│  └────┬──────────┬───────────┘
  └─────────┘       │ YES      │ NO
                     ↓          ↓
              ┌───────────┐  ┌──────────────────────┐
              │ Efficient │  │ Is an approximation  │
              │ algorithm │  │ guarantee needed?     │
              │ for       │  └────┬──────────┬────────┘
              │ special   │      │ YES      │ NO
              │ cases     │      ↓          ↓
              └───────────┘  ┌───────────┐  ┌──────────┐
                             │Approxima- │  │ Heuristic│
                             │tion algo- │  │          │
                             │rithm      │  │          │
                             └───────────┘  └──────────┘
```

### 5.2 Exact Solutions (for small-scale instances)

```python
# 1. Brute-force bit enumeration (n ≤ 20)
def knapsack_brute_force(weights, values, capacity):
    """
    Knapsack problem via brute-force bit enumeration
    O(2^n): practical for n ≤ 20 or so
    """
    n = len(weights)
    best_value = 0
    best_items = 0

    for mask in range(1 << n):
        total_weight = 0
        total_value = 0
        for i in range(n):
            if mask & (1 << i):
                total_weight += weights[i]
                total_value += values[i]

        if total_weight <= capacity and total_value > best_value:
            best_value = total_value
            best_items = mask

    return best_value, best_items


# 2. Dynamic programming (exponential time but faster)
def tsp_dp(dist, n):
    """
    TSP DP solution (Held-Karp)
    O(n² × 2^n): significantly faster than brute-force O(n! × n)
    Practical for n ≤ 25 or so
    """
    INF = float('inf')
    # dp[S][i] = minimum cost of visiting the set of cities S,
    #            ending at city i
    dp = [[INF] * n for _ in range(1 << n)]
    dp[1][0] = 0  # Start from city 0

    for S in range(1 << n):
        for u in range(n):
            if dp[S][u] == INF:
                continue
            if not (S & (1 << u)):
                continue
            for v in range(n):
                if S & (1 << v):
                    continue
                new_S = S | (1 << v)
                dp[new_S][v] = min(dp[new_S][v], dp[S][u] + dist[u][v])

    # Visit all cities and return to 0
    full = (1 << n) - 1
    result = min(dp[full][i] + dist[i][0] for i in range(1, n))

    return result


# 3. Branch and Bound
def branch_and_bound_tsp(dist, n):
    """
    Branch and bound for TSP
    Worst case O(n!), but often faster than DP on average
    Pruning significantly reduces the search space
    """
    import heapq

    INF = float('inf')
    best = INF

    def lower_bound(visited, current, cost):
        """Lower bound computation (simplified: sum of minimum edges)"""
        lb = cost
        for i in range(n):
            if i not in visited:
                min_edge = min(dist[i][j] for j in range(n) if j != i)
                lb += min_edge
        return lb

    # Priority queue: (lower bound, cost, current city, visited set)
    pq = [(lower_bound({0}, 0, 0), 0, 0, frozenset({0}))]

    while pq:
        lb, cost, current, visited = heapq.heappop(pq)

        if lb >= best:
            continue  # Pruning

        if len(visited) == n:
            total = cost + dist[current][0]
            best = min(best, total)
            continue

        for next_city in range(n):
            if next_city in visited:
                continue
            new_cost = cost + dist[current][next_city]
            new_visited = visited | {next_city}
            lb = lower_bound(new_visited, next_city, new_cost)
            if lb < best:
                heapq.heappush(pq, (lb, new_cost, next_city, new_visited))

    return best
```

### 5.3 Approximation Algorithms

```python
# Implementation examples of approximation algorithms

# 1. 2-approximation for vertex cover
def vertex_cover_2approx(graph):
    """
    2-approximation algorithm for vertex cover

    Approximation ratio: 2 (guaranteed within 2x of optimal)
    Time complexity: O(V + E)

    Algorithm:
    1. Select an edge
    2. Add both endpoints to the cover
    3. Remove all edges incident to the added vertices
    4. Repeat until no edges remain
    """
    cover = set()
    edges = set()

    for u in graph:
        for v in graph[u]:
            if u < v:
                edges.add((u, v))

    remaining_edges = set(edges)

    while remaining_edges:
        # Select an edge
        u, v = next(iter(remaining_edges))
        cover.add(u)
        cover.add(v)

        # Remove edges incident to both endpoints
        remaining_edges = {
            (a, b) for (a, b) in remaining_edges
            if a != u and a != v and b != u and b != v
        }

    return cover

# Proof of 2-approximation:
# - Let M be the set of selected edges (|M| edges)
# - Edges in M share no endpoints (a matching)
# - cover = 2|M| vertices
# - The optimal solution must include at least one endpoint
#   of each edge → OPT ≥ |M|
# - cover = 2|M| ≤ 2 × OPT ∎


# 2. Greedy approximation for set cover (O(log n)-approximation)
def greedy_set_cover(universe, sets):
    """
    Greedy algorithm for set cover

    Approximation ratio: H(n) = Σ_{i=1}^{n} 1/i ≈ ln(n)
    → This is the best possible (if P ≠ NP, below (1-ε)ln(n)
      is impossible)

    Time complexity: O(|universe| × |sets|)
    """
    uncovered = set(universe)
    selected = []
    remaining_sets = list(sets.items())

    while uncovered:
        # Select the set covering the most uncovered elements
        best_set = max(
            remaining_sets,
            key=lambda s: len(s[1] & uncovered)
        )

        if len(best_set[1] & uncovered) == 0:
            break  # Cannot cover any more

        selected.append(best_set[0])
        uncovered -= best_set[1]
        remaining_sets.remove(best_set)

    return selected


# 3. Christofides' approximation for TSP (3/2-approximation)
def christofides_tsp(dist, n):
    """
    Christofides' algorithm (conceptual implementation)

    Guarantees 3/2-approximation for TSP satisfying the
    triangle inequality
    This is the best approximation ratio achievable in
    polynomial time

    Procedure:
    1. Find the minimum spanning tree T
    2. Find the minimum weight perfect matching M on
       odd-degree vertices of T
    3. Find an Euler circuit on T ∪ M
    4. Shortcut to obtain a Hamiltonian cycle
    """
    # 1. Minimum spanning tree
    mst = compute_mst(dist, n)

    # 2. Minimum perfect matching on odd-degree vertices
    odd_vertices = [v for v in range(n) if degree(mst, v) % 2 == 1]
    matching = min_weight_perfect_matching(dist, odd_vertices)

    # 3. Euler graph from MST + matching
    euler_graph = combine(mst, matching)
    euler_tour = find_euler_tour(euler_graph)

    # 4. Shortcut (skip duplicate vertices)
    visited = set()
    hamiltonian = []
    for v in euler_tour:
        if v not in visited:
            visited.add(v)
            hamiltonian.append(v)

    return hamiltonian

# Proof of approximation ratio:
# MST ≤ OPT (removing one edge from optimal TSP gives an upper
#             bound for MST)
# Matching ≤ OPT/2 (at most half of optimal TSP on odd-degree vertices)
# Therefore: Christofides ≤ MST + Matching ≤ OPT + OPT/2 = 3/2 × OPT


# 4. FPTAS for the knapsack problem
#    (Fully Polynomial-Time Approximation Scheme)
def knapsack_fptas(weights, values, capacity, epsilon):
    """
    FPTAS for the knapsack problem

    Guarantees (1-ε)-approximation for any ε > 0
    Time complexity: O(n² / ε)

    Existence of FPTAS → not strongly NP-Hard
    """
    n = len(weights)
    v_max = max(values)

    # Scaling
    K = epsilon * v_max / n
    scaled_values = [int(v / K) for v in values]
    V_total = sum(scaled_values)

    # DP with scaled values
    INF = float('inf')
    # dp[v] = minimum weight to achieve exactly value sum v
    dp = [INF] * (V_total + 1)
    dp[0] = 0

    for i in range(n):
        for v in range(V_total, scaled_values[i] - 1, -1):
            if dp[v - scaled_values[i]] + weights[i] < dp[v]:
                dp[v] = dp[v - scaled_values[i]] + weights[i]

    # Find the maximum value within capacity
    best_v = 0
    for v in range(V_total + 1):
        if dp[v] <= capacity:
            best_v = v

    return best_v * K  # Restore to original scale
```

### 5.4 Inapproximability

```
Inapproximability:

  PCP Theorem (Probabilistically Checkable Proofs, 1992):
  NP = PCP(log n, 1)
  → Proofs of NP problems can be verified with high probability
    by randomly checking only a constant number of bits

  Consequences of the PCP Theorem (inapproximability):

  ┌────────────────────────────────────────────────────────┐
  │ Problem              │ Approximable       │ Inapproximable     │
  ├───────────────────┼──────────────────┼──────────────────┤
  │ Vertex cover        │ 2-approx           │ 2-ε is NP-Hard    │
  │                     │ (achieved)         │ (under UGC)        │
  │ Set cover           │ ln(n)-approx       │ (1-ε)ln(n) is     │
  │                     │ (achieved)         │ NP-Hard            │
  │ MAX-3SAT           │ 7/8-approx         │ 7/8+ε is NP-Hard  │
  │                     │ (achieved)         │                    │
  │ MAX-CLIQUE         │ n^{1-ε}-approx     │ Below n^{1-ε} is  │
  │                     │                    │ NP-Hard            │
  │ TSP (general)       │ Inapproximable     │ Constant approx    │
  │                     │                    │ is NP-Hard         │
  │ TSP (triangle ineq.)│ 3/2-approx        │ 220/219 is NP-Hard │
  │                     │ (achieved)         │                    │
  │ Knapsack           │ FPTAS exists        │ ─                  │
  └────────────────────────────────────────────────────────┘

  Meaning:
  - MAX-3SAT: approximation beyond 7/8 is impossible (if P ≠ NP)
    → Random assignment already achieves 7/8
    → Going beyond this is fundamentally impossible
  - MAX-CLIQUE: even n^{1-ε}-factor approximation is impossible
    → Extremely hard to approximate
```

### 5.5 Heuristics

```python
# Implementation examples of heuristics

import random
import math

# 1. Simulated Annealing
def simulated_annealing_tsp(dist, n, initial_temp=10000,
                             cooling_rate=0.9995, min_temp=1e-8):
    """
    Simulated annealing for TSP

    Guarantee: none (but practically yields very good solutions)
    Running time: user-specified (controlled by iterations or temperature)
    """
    # Initial solution: random permutation
    current = list(range(n))
    random.shuffle(current)

    def tour_cost(tour):
        return sum(dist[tour[i]][tour[(i+1) % n]] for i in range(n))

    current_cost = tour_cost(current)
    best = current[:]
    best_cost = current_cost
    temp = initial_temp

    while temp > min_temp:
        # Neighborhood: 2-opt (swap 2 edges)
        i = random.randint(0, n - 2)
        j = random.randint(i + 1, n - 1)
        neighbor = current[:i] + current[i:j+1][::-1] + current[j+1:]
        neighbor_cost = tour_cost(neighbor)

        # Acceptance criterion
        delta = neighbor_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current = neighbor
            current_cost = neighbor_cost

            if current_cost < best_cost:
                best = current[:]
                best_cost = current_cost

        temp *= cooling_rate

    return best, best_cost


# 2. Genetic Algorithm
def genetic_algorithm_tsp(dist, n, pop_size=100, generations=1000,
                          mutation_rate=0.02):
    """
    Genetic algorithm for TSP

    Components:
    - Individual: permutation of cities (chromosome)
    - Fitness: inverse of tour length
    - Crossover: Order crossover (OX)
    - Mutation: 2-opt
    - Selection: Tournament selection
    """
    def tour_cost(tour):
        return sum(dist[tour[i]][tour[(i+1) % n]] for i in range(n))

    def tournament_select(population, fitnesses, tournament_size=5):
        indices = random.sample(range(len(population)), tournament_size)
        best_idx = max(indices, key=lambda i: fitnesses[i])
        return population[best_idx]

    def order_crossover(parent1, parent2):
        """Order Crossover (OX)"""
        start = random.randint(0, n - 2)
        end = random.randint(start + 1, n - 1)

        child = [-1] * n
        child[start:end+1] = parent1[start:end+1]

        fill_pos = (end + 1) % n
        parent2_pos = (end + 1) % n
        while -1 in child:
            if parent2[parent2_pos] not in child:
                child[fill_pos] = parent2[parent2_pos]
                fill_pos = (fill_pos + 1) % n
            parent2_pos = (parent2_pos + 1) % n

        return child

    def mutate(tour):
        """2-opt mutation"""
        if random.random() < mutation_rate:
            i = random.randint(0, n - 2)
            j = random.randint(i + 1, n - 1)
            tour[i:j+1] = reversed(tour[i:j+1])
        return tour

    # Initial population
    population = [random.sample(range(n), n) for _ in range(pop_size)]

    best_ever = None
    best_cost_ever = float('inf')

    for gen in range(generations):
        # Fitness computation
        costs = [tour_cost(ind) for ind in population]
        fitnesses = [1.0 / c for c in costs]

        # Update best individual
        best_idx = min(range(pop_size), key=lambda i: costs[i])
        if costs[best_idx] < best_cost_ever:
            best_cost_ever = costs[best_idx]
            best_ever = population[best_idx][:]

        # Generate next generation
        new_population = [best_ever[:]]  # Elitism

        while len(new_population) < pop_size:
            p1 = tournament_select(population, fitnesses)
            p2 = tournament_select(population, fitnesses)
            child = order_crossover(p1, p2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

    return best_ever, best_cost_ever


# 3. Tabu Search
def tabu_search_tsp(dist, n, max_iter=10000, tabu_size=20):
    """
    Tabu search for TSP

    Forbids the last tabu_size moves (tabu list)
    → Enables escape from local optima
    """
    current = list(range(n))
    random.shuffle(current)

    def tour_cost(tour):
        return sum(dist[tour[i]][tour[(i+1) % n]] for i in range(n))

    current_cost = tour_cost(current)
    best = current[:]
    best_cost = current_cost
    tabu_list = []

    for _ in range(max_iter):
        # Enumerate all 2-opt neighborhoods
        best_neighbor = None
        best_neighbor_cost = float('inf')
        best_move = None

        for i in range(n - 1):
            for j in range(i + 1, n):
                move = (i, j)
                neighbor = current[:i] + current[i:j+1][::-1] + current[j+1:]
                nc = tour_cost(neighbor)

                # Accept if not tabu, or if it improves the best known solution
                if (move not in tabu_list or nc < best_cost):
                    if nc < best_neighbor_cost:
                        best_neighbor = neighbor
                        best_neighbor_cost = nc
                        best_move = move

        if best_neighbor is None:
            break

        current = best_neighbor
        current_cost = best_neighbor_cost

        # Update tabu list
        tabu_list.append(best_move)
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

        if current_cost < best_cost:
            best = current[:]
            best_cost = current_cost

    return best, best_cost
```

### 5.6 Parameterized Complexity (FPT)

```
Parameterized Complexity (Fixed-Parameter Tractability):

  Idea: Consider a parameter k in addition to input size n

  FPT: Problems solvable in O(f(k) × n^c)
  - f(k): depends only on k (may be exponential)
  - n^c: polynomial in input size
  - Practically solvable when k is small

  Example: Vertex cover (parameter k = cover size)
  - Brute force: O(n^k) — impractical for large k
  - FPT algorithm: O(2^k × n) — fast when k is small

  FPT hierarchy:
  ┌─────────────────────────────────────────────┐
  │ FPT ⊆ W[1] ⊆ W[2] ⊆ ... ⊆ XP             │
  │                                             │
  │ FPT: f(k) × n^c                            │
  │ XP: n^{f(k)} (parameter appears in exponent)│
  │                                             │
  │ FPT ≠ W[1] is conjectured                  │
  │ → The parameterized version of P ≠ NP       │
  └─────────────────────────────────────────────┘

  Examples of W[1]-Complete problems:
  - k-Clique
  - Independent set (parameter k)

  Examples of FPT:
  - Vertex cover (parameter k): O(1.2738^k + kn)
  - k-Path problem: O(2^k × n) (Color Coding)
  - SAT on graphs of treewidth k: O(2^k × n)
```

```python
# Examples of FPT algorithms

# FPT algorithm for vertex cover (branch and bound)
def vertex_cover_fpt(graph, k):
    """
    FPT algorithm for vertex cover
    Time complexity: O(2^k × (V + E))

    Idea:
    Select an edge (u,v) → two choices: include u or include v
    in the cover → search a binary tree of depth k
    """
    def solve(edges, cover, remaining_k):
        # Success if no edges remain
        if not edges:
            return cover

        # Failure if budget is exhausted
        if remaining_k == 0:
            return None

        # Select an edge
        u, v = next(iter(edges))

        # Choice 1: include u in the cover
        new_edges_u = {(a, b) for (a, b) in edges if a != u and b != u}
        result = solve(new_edges_u, cover | {u}, remaining_k - 1)
        if result is not None:
            return result

        # Choice 2: include v in the cover
        new_edges_v = {(a, b) for (a, b) in edges if a != v and b != v}
        result = solve(new_edges_v, cover | {v}, remaining_k - 1)
        if result is not None:
            return result

        return None  # No vertex cover of size ≤ k exists

    # Build the edge set
    edges = set()
    for u in graph:
        for v in graph[u]:
            if u < v:
                edges.add((u, v))

    return solve(edges, set(), k)


# Color Coding: FPT algorithm for the k-path problem
def color_coding_k_path(graph, n, k):
    """
    k-Path problem: Does the graph contain a path of length k?
    FPT algorithm: O(2^k × E) (randomized)

    Idea:
    1. Randomly assign k colors to each vertex
    2. Find a colorful path (using all colors) via DP
    3. Success probability ≥ e^{-k} → repeat O(e^k) times
       for high probability
    """
    for trial in range(int(2.72 ** k) * 2):  # e^k × 2 trials
        # Random coloring
        colors = [random.randint(0, k - 1) for _ in range(n)]

        # DP: dp[v][S] = does a colorful path ending at vertex v
        #                using color set S exist?
        dp = [[False] * (1 << k) for _ in range(n)]

        for v in range(n):
            dp[v][1 << colors[v]] = True

        for S in range(1, 1 << k):
            for v in range(n):
                if not dp[v][S]:
                    continue
                for u in graph.get(v, []):
                    c = colors[u]
                    if not (S & (1 << c)):  # New color
                        dp[u][S | (1 << c)] = True

        # Was a path using all colors found?
        full_set = (1 << k) - 1
        for v in range(n):
            if dp[v][full_set]:
                return True

    return False
```

---

## 6. Important Theorems

### 6.1 Time Hierarchy Theorem and Space Hierarchy Theorem

```
Time Hierarchy Theorem:
  If f(n) × log(f(n)) = o(g(n)) then
  DTIME(f(n)) ⊊ DTIME(g(n))

  Consequence:
  - P ⊊ EXPTIME (P ≠ EXPTIME)
  - With sufficiently more time, more problems become solvable

Space Hierarchy Theorem:
  If f(n) = o(g(n)) then
  SPACE(f(n)) ⊊ SPACE(g(n))

  Consequence:
  - L ⊊ PSPACE
  - With more space, more problems become solvable

Note:
  - P ≠ NP cannot be derived from the time hierarchy theorem
  - The hierarchy theorems are valid only within the "same
    computational model"
  - Cannot be used to compare deterministic vs nondeterministic
```

### 6.2 Savitch's Theorem

```
Savitch's Theorem:
  NSPACE(f(n)) ⊆ SPACE(f(n)²)

  Consequences:
  - NPSPACE = PSPACE
  - Nondeterminism only squares the space requirement
  - In contrast, the effect of nondeterminism on time is
    unknown (P vs NP)

  Proof idea:
  Simulate the accepting computation of the NTM using a
  deterministic TM in O(f(n)²) space by trying all
  intermediate configurations

  Practical meaning:
  - Optimal strategies for games (PSPACE-Complete) can be
    computed in polynomial space without nondeterminism
  - QBF (quantified Boolean formulas) can also be solved
    in polynomial space
```

---

## 7. Practice Exercises

### Exercise 1: NP-Complete Reduction (Basic)

```
Problem: Show that the independent set problem can be reduced
to the vertex cover problem.

Definitions:
- Independent set: a subset of vertices where no two vertices
  are connected by an edge
- Vertex cover: a subset of vertices that includes at least
  one endpoint of every edge

Proof:
  S is an independent set ⟺ V \ S is a vertex cover

  (→) Suppose S is an independent set.
  For any edge (u,v), at least one of u,v is not in S
  (if both were in S, they would not be independent).
  Therefore, at least one is in V \ S → V \ S is a vertex cover.

  (←) Suppose V \ S is a vertex cover.
  For any two vertices u,v in S, (u,v) is not an edge
  (if it were, then u,v ∈ S means neither is in V \ S,
   so it wouldn't be a cover).
  Therefore, S is an independent set.

  Reduction: An independent set of size k exists ⟺
             a vertex cover of size (n-k) exists

  This reduction takes O(1) time → polynomial-time reduction ∎
```

### Exercise 2: Approximation Algorithm (Applied)

```python
"""
Exercise: Implement the greedy approximation algorithm for set cover
and empirically measure the approximation ratio.
"""

import random

def generate_set_cover_instance(n, m, density=0.3):
    """Generate a random instance of the set cover problem"""
    universe = set(range(n))
    sets = {}
    for i in range(m):
        size = max(1, int(n * density * random.random()))
        sets[i] = set(random.sample(list(universe), min(size, n)))

    # Ensure all elements can be covered
    for elem in universe:
        random_set = random.choice(list(sets.keys()))
        sets[random_set].add(elem)

    return universe, sets


def optimal_set_cover(universe, sets):
    """Exact solution (for small instances, exponential time)"""
    n = len(sets)
    keys = list(sets.keys())
    best = None

    for mask in range(1, 1 << n):
        covered = set()
        selected = []
        for i in range(n):
            if mask & (1 << i):
                covered |= sets[keys[i]]
                selected.append(keys[i])
        if covered >= universe:
            if best is None or len(selected) < len(best):
                best = selected

    return best


# Experiment
for n in [10, 15, 20]:
    total_ratio = 0
    trials = 50
    for _ in range(trials):
        universe, sets = generate_set_cover_instance(n, n * 2)
        greedy = greedy_set_cover(universe, sets)
        optimal = optimal_set_cover(universe, sets)
        ratio = len(greedy) / len(optimal)
        total_ratio += ratio

    avg_ratio = total_ratio / trials
    print(f"n={n}: Average approximation ratio = {avg_ratio:.3f} "
          f"(theoretical upper bound: {math.log(n):.3f})")
```

### Exercise 3: SAT Solver (Implementation)

```python
"""
Exercise: Implement the DPLL algorithm (the foundation of SAT solvers).
"""

def dpll(clauses, assignment=None):
    """
    DPLL Algorithm — standard approach for solving SAT

    Worst case O(2^n) but very efficient in practice
    Foundation of modern SAT solvers (MiniSat, Z3, etc.)
    """
    if assignment is None:
        assignment = {}

    # Satisfaction check
    clauses = simplify(clauses, assignment)

    # An empty clause exists → unsatisfiable
    if any(len(c) == 0 for c in clauses):
        return None

    # All clauses eliminated → satisfiable
    if len(clauses) == 0:
        return assignment

    # Unit Propagation
    unit_clauses = [c for c in clauses if len(c) == 1]
    while unit_clauses:
        literal = unit_clauses[0][0]
        var = abs(literal)
        value = literal > 0
        assignment[var] = value
        clauses = simplify(clauses, {var: value})

        if any(len(c) == 0 for c in clauses):
            return None
        if len(clauses) == 0:
            return assignment

        unit_clauses = [c for c in clauses if len(c) == 1]

    # Pure Literal Elimination
    all_literals = set()
    for clause in clauses:
        for lit in clause:
            all_literals.add(lit)

    for lit in list(all_literals):
        if -lit not in all_literals:
            var = abs(lit)
            assignment[var] = (lit > 0)
            clauses = simplify(clauses, {var: lit > 0})

    if len(clauses) == 0:
        return assignment

    # Branching
    var = abs(clauses[0][0])

    # Try True
    result = dpll(clauses, {**assignment, var: True})
    if result is not None:
        return result

    # Try False
    result = dpll(clauses, {**assignment, var: False})
    return result


def simplify(clauses, assignment):
    """Simplify clauses based on the assignment"""
    new_clauses = []
    for clause in clauses:
        new_clause = []
        satisfied = False
        for lit in clause:
            var = abs(lit)
            if var in assignment:
                if (lit > 0) == assignment[var]:
                    satisfied = True
                    break
                # If the literal is False, remove it from the clause
            else:
                new_clause.append(lit)
        if not satisfied:
            new_clauses.append(new_clause)
    return new_clauses
```

---


## FAQ

### Q1: What is the most important point when studying this topic?

Gaining practical experience is the most important. Understanding deepens not only through theory but also by actually writing code and verifying its behavior.

### Q2: What common mistakes do beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving to the next step.

### Q3: How is this used in practice?

Knowledge of this topic is frequently applied in everyday development work. It becomes particularly important during code reviews and architecture design.

---

## Summary

| Concept | Key Points |
|---------|-----------|
| P | Solvable in polynomial time. Sorting, shortest path, 2-SAT, etc. |
| NP | Verifiable in polynomial time. P ⊆ NP |
| NP-Complete | Hardest in NP. Solving one solves all NP problems |
| NP-Hard | At least as hard as NP-Complete. May not belong to NP |
| P vs NP? | Greatest unsolved problem in CS. Directly tied to cryptographic security |
| PSPACE | Polynomial space. QBF is the complete problem. PSPACE = NPSPACE |
| BPP/BQP | Probabilistic/quantum computation classes. BPP = P is conjectured |
| Approximation algorithms | Guarantee a constant factor of optimal. PCP theorem reveals limits |
| FPT | Efficient when parameter k is small. f(k)×n^c |
| Heuristics | No guarantees but practical. SA, GA, tabu search, etc. |

---

## Recommended Next Reading

---

## References
1. Sipser, M. "Introduction to the Theory of Computation." Chapters 7-8.
2. Arora, S. & Barak, B. "Computational Complexity: A Modern Approach." Cambridge, 2009.
3. Cook, S. A. "The Complexity of Theorem-Proving Procedures." STOC, 1971.
4. Karp, R. M. "Reducibility Among Combinatorial Problems." 1972.
5. Garey, M. R. & Johnson, D. S. "Computers and Intractability." W. H. Freeman, 1979.
6. Vazirani, V. V. "Approximation Algorithms." Springer, 2001.
7. Downey, R. G. & Fellows, M. R. "Parameterized Complexity." Springer, 1999.
8. Arora, S., Lund, C., Motwani, R., Sudan, M., & Szegedy, M. "Proof Verification and the Hardness of Approximation Problems." JACM, 1998.
9. Williamson, D. P. & Shmoys, D. B. "The Design of Approximation Algorithms." Cambridge, 2011.
10. Christofides, N. "Worst-Case Analysis of a New Heuristic for the Travelling Salesman Problem." 1976.
