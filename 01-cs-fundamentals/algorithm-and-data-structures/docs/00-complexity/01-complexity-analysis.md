# Complexity Analysis — Worst/Average/Best Case, Amortized Analysis, Recurrence Analysis, Master Theorem

> Learn the systematic methods for accurately evaluating algorithm efficiency. This guide covers the differences between worst, average, and best cases, the concepts behind amortized analysis, how to formulate recurrence relations for recursive algorithms, and their solution methods (Master Theorem, recursion tree method, substitution method).

---

## What You Will Learn in This Chapter

1. **Asymptotic notation** (O, Omega, Theta): rigorous definitions and usage
2. **Worst case, average case, best case**: distinguishing and analyzing each
3. **Amortized analysis** (aggregate, accounting, potential methods): theory and practice
4. **Recurrence relations** for recursion: formulation and three solution methods (Master Theorem, recursion tree, substitution)
5. **Master Theorem**: the three cases, applicability conditions, and handling inapplicable cases
6. **Common pitfalls** in complexity analysis and how to avoid them


## Prerequisites

The following knowledge will help deepen your understanding of this guide:

- Basic programming knowledge
- Understanding of related foundational concepts
- Familiarity with the content of [Big-O Notation and Complexity Basics](./00-big-o-notation.md)

---

## Table of Contents

1. [Fundamentals of Asymptotic Notation](#1-fundamentals-of-asymptotic-notation)
2. [Worst Case, Average Case, and Best Case Analysis](#2-worst-case-average-case-and-best-case-analysis)
3. [Amortized Analysis](#3-amortized-analysis)
4. [Methods for Analyzing Recursive Complexity](#4-methods-for-analyzing-recursive-complexity)
5. [Formulating Recurrence Relations](#5-formulating-recurrence-relations)
6. [Master Theorem](#6-master-theorem)
7. [Recursion Tree Method](#7-recursion-tree-method)
8. [Substitution Method (Induction)](#8-substitution-method-induction)
9. [Common Recursion Patterns](#9-common-recursion-patterns)
10. [Comparison Tables](#10-comparison-tables)
11. [Anti-patterns](#11-anti-patterns)
12. [Edge Case Analysis](#12-edge-case-analysis)
13. [Practice Problems](#13-practice-problems)
14. [FAQ](#14-faq)
15. [Summary](#15-summary)
16. [References](#16-references)

---

## 1. Fundamentals of Asymptotic Notation

Complexity analysis discusses the behavior of algorithms as the input size n becomes sufficiently large.
To do this, we use **asymptotic notation**, which ignores constant factors and lower-order terms and captures only the dominant growth rate.

### 1.1 Why Use Asymptotic Notation?

Concrete execution times (e.g., 3.2 seconds) vary depending on hardware, implementation language, compiler optimizations, and many other factors.
The reasons for using asymptotic notation are as follows:

- **Hardware independence**: Enables evaluation independent of CPU speed or memory bandwidth
- **Scalability prediction**: Allows estimation of how much execution time increases when input size grows by 10x
- **Algorithm comparison**: Enables discussion of the fundamental efficiency differences between different algorithms solving the same problem

### 1.2 The Three Asymptotic Notations

```
Relationship diagram of asymptotic notations:

              O(g(n))        -- Upper bound (at most this much)
             /      \
     Actual growth rate f(n)
             \      /
              Omega(g(n))    -- Lower bound (at least this much)

     When upper and lower bounds coincide:
              Theta(g(n))    -- Tight bound (exactly this much)
```

**O notation (upper bound)**: There exist constants c > 0 and n_0 > 0 such that for all n >= n_0, f(n) <= c * g(n). We write f(n) = O(g(n)).

**Omega notation (lower bound)**: There exist constants c > 0 and n_0 > 0 such that for all n >= n_0, f(n) >= c * g(n). We write f(n) = Omega(g(n)).

**Theta notation (tight bound)**: When f(n) = O(g(n)) and f(n) = Omega(g(n)), we write f(n) = Theta(g(n)).

### 1.3 Rules for Asymptotic Notation

The following rules hold when working with asymptotic notation. These are properties backed by proofs, not mere conventions.

| Rule | Content | Reason |
|------|---------|--------|
| Ignore constant factors | O(5n) = O(n) | The 5 can be absorbed into the constant c in the definition |
| Ignore lower-order terms | O(n^2 + n) = O(n^2) | For sufficiently large n, n^2 dominates n |
| Ignore logarithm base | O(log_2 n) = O(log_10 n) | log_a(n) = log_b(n) / log_b(a), so they differ only by a constant factor |
| Addition rule | O(f) + O(g) = O(max(f, g)) | The dominant term determines the overall order |
| Multiplication rule | O(f) * O(g) = O(f * g) | Applies when each step is independently repeated |

### 1.4 Code Example: Building Intuition for Asymptotic Notation

The following code numerically demonstrates how quickly each complexity class grows.

```python
"""
A program that numerically compares growth rates of asymptotic notation.
Why this comparison matters: To intuitively understand how input size
growth affects execution time when choosing algorithms.
"""

import math


def compare_growth_rates(sizes: list[int]) -> None:
    """Display growth rates of different complexity classes in tabular form.

    Args:
        sizes: List of input sizes to compare
    """
    header = f"{'n':>10} | {'log n':>10} | {'n':>10} | {'n log n':>12} | {'n^2':>12} | {'2^n':>15}"
    separator = "-" * len(header)

    print(header)
    print(separator)

    for n in sizes:
        log_n = math.log2(n) if n > 0 else 0
        n_log_n = n * log_n
        n_squared = n * n
        # 2^n becomes astronomically large for large n, so we cap it
        two_to_n = 2 ** n if n <= 30 else float("inf")

        print(
            f"{n:>10} | "
            f"{log_n:>10.2f} | "
            f"{n:>10} | "
            f"{n_log_n:>12.1f} | "
            f"{n_squared:>12} | "
            f"{two_to_n:>15}"
        )


def main() -> None:
    sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    compare_growth_rates(sizes)
    print()

    # Feel the "wall" between complexity classes
    print("=== Values of each function at n=20 ===")
    n = 20
    print(f"  log n    = {math.log2(n):.2f}")
    print(f"  n        = {n}")
    print(f"  n log n  = {n * math.log2(n):.1f}")
    print(f"  n^2      = {n**2}")
    print(f"  n^3      = {n**3}")
    print(f"  2^n      = {2**n}")
    print(f"  n!       = {math.factorial(n)}")
    # 2^20 = 1,048,576 vs 20! = 2,432,902,008,176,640,000
    # This gap illustrates the "wall" between exponential and factorial time


if __name__ == "__main__":
    main()
```

Expected output (first few lines):

```
         n |      log n |          n |      n log n |          n^2 |             2^n
------------------------------------------------------------------------------------
         1 |       0.00 |          1 |          0.0 |            1 |               2
         2 |       1.00 |          2 |          2.0 |            4 |               4
         4 |       2.00 |          4 |          8.0 |           16 |              16
         8 |       3.00 |          8 |         24.0 |           64 |             256
        16 |       4.00 |         16 |         64.0 |          256 |           65536
```

---

## 2. Worst Case, Average Case, and Best Case Analysis

Even for the same algorithm, execution time can vary significantly depending on the input data.
Complexity analysis captures this variation from three perspectives.

### 2.1 Definition of the Three Cases

```
Relationship between input space and execution time:

  Execution
  Time
  ^
  |    x                              -- Worst case W(n)
  |         x        x
  |    x         x        x    x
  |         x        x              x -- Average case A(n)
  |    x              x        x
  |                        x
  |              x                    -- Best case B(n)
  |
  +-----------------------------------> Input patterns
           (all inputs of size n)

  W(n) = max { T(I) | |I| = n }    Maximum execution time over all inputs I of size n
  B(n) = min { T(I) | |I| = n }    Minimum execution time over all inputs I of size n
  A(n) = Sum T(I) * Pr(I)          Expected value of execution time over each input I
```

**Why worst case is emphasized**: In systems with response time guarantees (SLAs) or real-time systems, "fast most of the time" is insufficient -- "guaranteed response within a fixed time for any input" is required.

**Why average case also matters**: When the worst case rarely occurs (e.g., quicksort's O(n^2) is extremely rare), the average case provides a more practical performance evaluation.

### 2.2 Code Example: Three Cases of Linear Search

```python
"""
Analysis of worst, average, and best cases in linear search.
Demonstrates that the same algorithm can have vastly different
comparison counts depending on the target's position.
"""

import random
import statistics


def linear_search(arr: list[int], target: int) -> tuple[int, int]:
    """Perform linear search and return the found index and comparison count.

    Args:
        arr: Array to search
        target: Value to search for

    Returns:
        Tuple of (found index, comparison count).
        Returns -1 for the index if not found.
    """
    comparisons = 0
    for i, val in enumerate(arr):
        comparisons += 1
        if val == target:
            return i, comparisons
    return -1, comparisons


def analyze_linear_search(n: int, trials: int = 10000) -> None:
    """Experimentally analyze worst, average, and best cases of linear search.

    Why perform experimental analysis: To verify the validity of asymptotic
    analysis by confirming agreement with theoretical values.

    Args:
        n: Array size
        trials: Number of trials
    """
    arr = list(range(n))

    # Best case: target is at the beginning
    _, best_comparisons = linear_search(arr, 0)

    # Worst case: target does not exist
    _, worst_comparisons = linear_search(arr, n + 1)

    # Average case: target position is random
    comparison_counts = []
    for _ in range(trials):
        target = random.randint(0, n - 1)
        _, comps = linear_search(arr, target)
        comparison_counts.append(comps)

    avg_comparisons = statistics.mean(comparison_counts)

    print(f"=== Linear Search Analysis (n={n}) ===")
    print(f"  Best case:    {best_comparisons} comparisons   (theoretical: 1)")
    print(f"  Worst case:   {worst_comparisons} comparisons   (theoretical: {n})")
    print(f"  Average case: {avg_comparisons:.1f} comparisons (theoretical: {(n + 1) / 2:.1f})")
    print()

    # Check error against theoretical value
    theoretical_avg = (n + 1) / 2
    error_pct = abs(avg_comparisons - theoretical_avg) / theoretical_avg * 100
    print(f"  Error from theoretical average: {error_pct:.2f}%")
    # With sufficient trials, this error should be below 1%


def main() -> None:
    for n in [100, 1000, 10000]:
        analyze_linear_search(n)
        print()


if __name__ == "__main__":
    main()
```

### 2.3 Code Example: Three Cases of Quicksort

Quicksort is a prime example of an algorithm whose performance changes dramatically depending on pivot selection.

```python
"""
Concrete examples of worst, average, and best cases in quicksort.
Demonstrates how pivot selection affects algorithm performance.
"""

import random
import time


def quicksort_count(arr: list[int]) -> tuple[list[int], int]:
    """Perform quicksort (first-element pivot) and return comparison count.

    Why use the first element as pivot: To demonstrate an implementation
    prone to worst-case behavior, highlighting the importance of pivot selection.

    Args:
        arr: List to sort

    Returns:
        Tuple of (sorted list, comparison count)
    """
    if len(arr) <= 1:
        return arr[:], 0

    pivot = arr[0]  # Intentionally simple pivot selection
    left = []
    right = []
    comparisons = 0

    for x in arr[1:]:
        comparisons += 1
        if x <= pivot:
            left.append(x)
        else:
            right.append(x)

    sorted_left, left_comps = quicksort_count(left)
    sorted_right, right_comps = quicksort_count(right)

    return sorted_left + [pivot] + sorted_right, comparisons + left_comps + right_comps


def demonstrate_quicksort_cases(n: int) -> None:
    """Experimentally demonstrate the three cases of quicksort.

    Args:
        n: Array size
    """
    print(f"=== Quicksort Analysis (n={n}) ===")

    # Worst case: already sorted array (worst for first-element pivot)
    sorted_arr = list(range(n))
    _, worst_comps = quicksort_count(sorted_arr)
    print(f"  Worst case (sorted):      {worst_comps} comparisons")
    print(f"    -> Expected complexity: O(n^2) = {n * (n - 1) // 2}")

    # Best case: median always selected as pivot
    # (constructing such an array is complex, so we show the theoretical value)
    import math
    theoretical_best = n * math.log2(max(n, 1))
    print(f"  Best case theoretical:    O(n log n) ~ {theoretical_best:.0f}")

    # Average case: random array
    total_comps = 0
    trials = 100
    for _ in range(trials):
        random_arr = list(range(n))
        random.shuffle(random_arr)
        _, comps = quicksort_count(random_arr)
        total_comps += comps
    avg_comps = total_comps / trials
    # Average case theoretical value: 2n ln n ~ 1.39 n log_2 n
    theoretical_avg = 1.39 * n * math.log2(max(n, 1))
    print(f"  Average case ({trials}-trial mean): {avg_comps:.0f} comparisons")
    print(f"    -> Theoretical 1.39 n log n ~ {theoretical_avg:.0f}")
    print()


def main() -> None:
    # n=500 or so avoids stack overflow even in worst case
    for n in [50, 100, 500]:
        demonstrate_quicksort_cases(n)


if __name__ == "__main__":
    main()
```

### 2.4 Mathematical Derivation of Average Case Analysis

Average case analysis computes the expected value of execution time under an assumed probability distribution of inputs.

**Average case of linear search (when target exists in the array)**:

When target is at each position i (1 <= i <= n) with equal probability 1/n:

```
A(n) = Sum_{i=1}^{n} i * (1/n) = (1/n) * n(n+1)/2 = (n+1)/2
```

Therefore A(n) = Theta((n+1)/2) = Theta(n).

**Average case of quicksort**:

When the pivot is the k-th smallest element with probability 1/n each, the expected number of comparisons C(n) is:

```
C(n) = (n - 1) + (1/n) Sum_{k=0}^{n-1} [C(k) + C(n - 1 - k)]
     = (n - 1) + (2/n) Sum_{k=0}^{n-1} C(k)
```

Solving this recurrence yields C(n) = 2n ln n + O(n) ~ 1.39 n log_2 n. This derivation means that for random inputs, quicksort achieves "nearly optimal" O(n log n) performance.

---

## 3. Amortized Analysis

### 3.1 What Is Amortized Analysis?

Amortized analysis is a technique for evaluating the cost of a sequence of operations as a whole. It is used when looking at the worst case of individual operations is overly pessimistic, to accurately determine the "average cost per operation" across the entire sequence.

**Why is "simply summing worst cases" insufficient?**: For example, the append operation on a dynamic array is usually O(1), but occasionally costs O(n) when the array needs to be expanded (copied). Simply summing "O(n) per operation" yields O(n^2), but the actual cost is much less. Amortized analysis correctly handles these "occasionally expensive operations."

```
Intuitive diagram of amortized analysis:

  Cost
  ^
  |
n |              *                              *
  |
  |
  |
  |        *                         *
  |
  |    *                    *
  |  *              *
1 | * * * * * * * * * * * * * * * * * * * * * * * *
  +-----------------------------------------------> Operation number
      1 2 3 4 5 6 7 8 9 ...

  Most operations are O(1), but rare O(n) spikes occur.
  Amortized analysis accurately estimates the total cost across all operations.

  -> Total cost of n operations = O(n)
  -> Amortized cost per operation = O(1)
```

### 3.2 Three Amortized Analysis Methods

| Method | Concept | When to Apply |
|--------|---------|---------------|
| Aggregate | Directly compute total cost of all operations and divide by the number of operations | When computation can be done directly |
| Accounting | Overcharge cheap operations as "savings" and spend them on expensive ones | When you want to assign uniform costs per operation |
| Potential | Define a "potential function" on the data structure and track cost changes | Suitable for analyzing complex data structures |

### 3.3 Code Example: Amortized Analysis of Dynamic Arrays

```python
"""
Amortized analysis of dynamic arrays (doubling strategy).
Demonstrates the three methods (aggregate, accounting, potential) through implementation.
"""


class DynamicArray:
    """Dynamic array implementation using the doubling strategy.

    Why use the doubling strategy:
    When the array is full, capacity is doubled. A strategy of increasing
    by 1 each time would cost O(n^2) for n appends, but the doubling
    strategy costs only O(n). This difference stems from the frequency
    of capacity expansion.
    """

    def __init__(self) -> None:
        self._capacity = 1
        self._size = 0
        self._data = [None] * self._capacity
        self._total_cost = 0  # For cost tracking
        self._operation_costs: list[int] = []  # Record cost of each operation

    def append(self, value: int) -> int:
        """Add element to end and return cost (copy count + 1) of this operation.

        Args:
            value: Value to add

        Returns:
            Cost incurred by this operation
        """
        cost = 1  # Write cost for the element

        if self._size == self._capacity:
            # Expansion: copying all elements incurs cost
            cost += self._size  # Copy cost
            new_data = [None] * (self._capacity * 2)
            for i in range(self._size):
                new_data[i] = self._data[i]
            self._data = new_data
            self._capacity *= 2

        self._data[self._size] = value
        self._size += 1
        self._total_cost += cost
        self._operation_costs.append(cost)
        return cost

    @property
    def size(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def total_cost(self) -> int:
        return self._total_cost


def demonstrate_aggregate_method(n: int) -> None:
    """Demonstrate amortized analysis using the aggregate method.

    Aggregate method concept:
    Directly compute total cost of n appends.
    Expansion occurs at i = 1, 2, 4, 8, ..., 2^k,
    expansion cost is 1 + 2 + 4 + ... + 2^k <= 2n.
    Write cost is n.
    Total cost <= 3n -> O(1) per operation.
    """
    arr = DynamicArray()
    print(f"=== Aggregate Method Analysis (n={n}) ===")
    print(f"{'Op #':>8} | {'Cost of this op':>16} | {'Cumulative':>10} | {'Capacity':>6}")
    print("-" * 50)

    for i in range(1, n + 1):
        cost = arr.append(i)
        if cost > 1 or i <= 5 or i == n:  # Show expansions plus first/last
            print(f"{i:>8} | {cost:>16} | {arr.total_cost:>10} | {arr.capacity:>6}")

    amortized = arr.total_cost / n
    print(f"\n  Total cost: {arr.total_cost}")
    print(f"  Number of operations: {n}")
    print(f"  Amortized cost per operation: {amortized:.2f}")
    print(f"  -> Theoretical upper bound: 3.0 (3n/n)")
    print()


def demonstrate_accounting_method(n: int) -> None:
    """Demonstrate amortized analysis using the accounting method.

    Accounting method concept:
    Charge each append "amortized cost 3."
      - 1: for writing the element
      - 2: "savings" for future expansion
    During expansion, the saved coins pay for all copies.
    """
    print(f"=== Accounting Method Analysis (n={n}) ===")
    balance = 0  # Savings balance
    amortized_cost_per_op = 3  # Amortized cost charged per operation

    capacity = 1
    size = 0
    all_balanced = True

    for i in range(1, n + 1):
        balance += amortized_cost_per_op  # Charge 3 coins
        balance -= 1  # Use 1 coin for writing

        if size == capacity:
            # Expansion: use coins for copying size elements
            balance -= size
            capacity *= 2
            if balance < 0:
                all_balanced = False

        size += 1

    print(f"  Charge per operation: {amortized_cost_per_op}")
    print(f"  Final savings balance: {balance}")
    print(f"  Balance never went negative: {'Yes' if all_balanced else 'No'}")
    print(f"  -> If balance is always non-negative, amortized cost {amortized_cost_per_op} is justified")
    print()


def demonstrate_potential_method(n: int) -> None:
    """Demonstrate amortized analysis using the potential method.

    Potential method concept:
    Define potential function Phi(D) = 2 * size - capacity
    (where D is the data structure state)

    Amortized cost = actual cost + Phi(D_after) - Phi(D_before)

    Without expansion:
      Actual cost = 1
      Phi change = 2(size+1) - cap - (2*size - cap) = 2
      Amortized cost = 1 + 2 = 3

    With expansion (when size == capacity):
      Actual cost = 1 + size (copying)
      New capacity = 2 * capacity
      Phi_after = 2(size+1) - 2*capacity = 2*size + 2 - 2*size = 2
      Phi_before = 2*size - capacity = 2*size - size = size
      Phi change = 2 - size
      Amortized cost = (1 + size) + (2 - size) = 3
    """
    print(f"=== Potential Method Analysis (n={n}) ===")

    capacity = 1
    size = 0

    for i in range(1, min(n, 20) + 1):
        phi_before = 2 * size - capacity

        actual_cost = 1
        expanded = False
        if size == capacity:
            actual_cost += size
            capacity *= 2
            expanded = True

        size += 1
        phi_after = 2 * size - capacity
        amortized = actual_cost + phi_after - phi_before

        if expanded or i <= 5:
            print(
                f"  Op {i:>3}: "
                f"actual_cost={actual_cost:>4}, "
                f"Phi_change={phi_after - phi_before:>4}, "
                f"amortized_cost={amortized:>2}"
                f"{'  <- expansion' if expanded else ''}"
            )

    print(f"  -> Amortized cost = 3 (constant) for all operations")
    print()


def main() -> None:
    n = 64
    demonstrate_aggregate_method(n)
    demonstrate_accounting_method(n)
    demonstrate_potential_method(n)


if __name__ == "__main__":
    main()
```

### 3.4 Application of Amortized Analysis: Stack with MultiPop

```python
"""
Amortized analysis of a stack with MultiPop.
push is O(1), multi_pop(k) is O(min(k, size)), but the amortized
cost per operation over a sequence of n operations is O(1).
"""


class StackWithMultiPop:
    """Stack with MultiPop operation.

    Why amortized analysis is needed:
    multi_pop(k) is worst-case O(n), but only previously pushed elements
    can be popped. In a sequence of n operations, the total number of pops
    cannot exceed the number of pushes, so total cost is bounded by O(n).
    """

    def __init__(self) -> None:
        self._stack: list[int] = []
        self._total_cost = 0

    def push(self, value: int) -> int:
        """Push an element. Cost: 1."""
        self._stack.append(value)
        self._total_cost += 1
        return 1

    def multi_pop(self, k: int) -> tuple[list[int], int]:
        """Pop up to k elements. Cost: min(k, size).

        Args:
            k: Maximum number of elements to pop

        Returns:
            (list of popped elements, cost)
        """
        actual_pops = min(k, len(self._stack))
        popped = []
        for _ in range(actual_pops):
            popped.append(self._stack.pop())
        self._total_cost += actual_pops
        return popped, actual_pops

    @property
    def size(self) -> int:
        return len(self._stack)

    @property
    def total_cost(self) -> int:
        return self._total_cost


def demonstrate_multipop_amortized() -> None:
    """Demonstrate amortized analysis of MultiPop stack."""
    stack = StackWithMultiPop()
    operations = []

    # Operation sequence: many pushes with occasional bulk multi_pops
    import random
    random.seed(42)

    n = 100
    for i in range(n):
        if random.random() < 0.7 or stack.size == 0:
            cost = stack.push(i)
            operations.append(("push", cost))
        else:
            k = random.randint(1, stack.size)
            _, cost = stack.multi_pop(k)
            operations.append((f"multi_pop({k})", cost))

    print(f"=== MultiPop Stack Amortized Analysis ===")
    print(f"  Number of operations: {n}")
    print(f"  Total cost: {stack.total_cost}")
    print(f"  Amortized cost per operation: {stack.total_cost / n:.2f}")
    print(f"  -> Theoretical amortized cost: O(1)")
    print()

    # Check max cost of any individual operation
    max_cost = max(cost for _, cost in operations)
    print(f"  Maximum individual operation cost: {max_cost}")
    print(f"  -> Individual worst case is O(n), but amortized is O(1)")


if __name__ == "__main__":
    demonstrate_multipop_amortized()
```

### 3.5 Amortized Complexities of Common Data Structures

| Data Structure | Operation | Worst Case | Amortized Cost | Reason |
|----------------|-----------|------------|----------------|--------|
| Dynamic array | append | O(n) | O(1) | Doubling strategy makes expansion frequency decrease exponentially |
| Dynamic array | pop (end) | O(1) | O(1) | Always O(1) when no shrinking |
| Binomial heap | insert | O(log n) | O(1) | Total carry cost is bounded |
| Splay tree | any operation | O(n) | O(log n) | Splay operations balance the tree |
| Union-Find | Union + Find | O(log n) | O(alpha(n)) ~ O(1) | Path compression and union by rank |

---

## 4. Methods for Analyzing Recursive Complexity

Analyzing the complexity of recursive algorithms proceeds in two stages: first formulate the recurrence relation, then solve it.

```
Methods for analyzing recursive complexity:

+----------------------------------+
| Step 1: Formulate the recurrence |
|   Derive the cost relation from  |
|   the recursive structure        |
+----------+-----------------------+
           v
+----------------------------------+
| Step 2: Solve the recurrence     |
|                                  |
|  +- Master Theorem              |
|  |    For T(n)=aT(n/b)+f(n)    |
|  |    -> 3 cases, immediate     |
|  |                               |
|  +- Recursion Tree Method        |
|  |    Sum costs at each level    |
|  |    -> Visually intuitive      |
|  |                               |
|  +- Substitution Method          |
|       Guess solution, prove by   |
|       induction -> Most rigorous |
+----------------------------------+
```

### Why Go Through Recurrence Relations?

The running time of a recursive algorithm is expressed in terms of the running time of smaller instances of itself. This directly corresponds to the structure of a recurrence relation. By solving the recurrence, we obtain a closed-form (non-recursive) expression in terms of input size n, allowing discussion of asymptotic growth rates.

---

## 5. Formulating Recurrence Relations

### 5.1 Steps for Formulating Recurrences

1. **Identify the base case**: The condition under which recursion stops and its cost
2. **Identify the recursive call structure**: How many calls are made and what is the problem size of each
3. **Identify non-recursive costs**: Cost of dividing, merging, or other processing

### 5.2 Example: Merge Sort

```python
"""
Derivation of merge sort's recurrence relation.
Each line's cost is annotated in comments to make
the components of the recurrence explicit.
"""


def merge_sort(arr: list[int]) -> list[int]:
    """Merge sort -- T(n) = 2T(n/2) + O(n)

    Recurrence derivation:
    - Base case: O(1) when len(arr) <= 1
    - Recursive calls: 2 calls, each of size n/2 -> 2T(n/2)
    - Merge cost: merge examines each element once -> O(n)
    """
    if len(arr) <= 1:                  # O(1) -- base case
        return arr

    mid = len(arr) // 2                # O(1) -- compute split point
    left = merge_sort(arr[:mid])       # T(n/2) -- recursively sort left half
    right = merge_sort(arr[mid:])      # T(n/2) -- recursively sort right half
    return merge(left, right)          # O(n) -- merge


def merge(left: list[int], right: list[int]) -> list[int]:
    """Merge two sorted lists. Cost: O(n).

    Why O(n): Each element is compared and copied at most once,
    and the total number of elements across both lists is n.
    """
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result


def verify_merge_sort() -> None:
    """Verify correctness and complexity of merge sort."""
    import random
    import time

    sizes = [1000, 2000, 4000, 8000, 16000]
    print("=== Merge Sort Complexity Verification ===")
    print(f"{'n':>8} | {'Time (ms)':>10} | {'Ratio':>8} | {'Expected ratio':>14}")
    print("-" * 50)

    prev_time = None
    for n in sizes:
        arr = list(range(n))
        random.shuffle(arr)

        start = time.perf_counter()
        merge_sort(arr)
        elapsed = (time.perf_counter() - start) * 1000

        if prev_time and prev_time > 0:
            ratio = elapsed / prev_time
            # For O(n log n), doubling n gives
            # 2n log(2n) / (n log n) = 2(1 + log2/logn) ~ 2+ (for large n)
            print(f"{n:>8} | {elapsed:>10.2f} | {ratio:>8.2f} | ~2.0-2.3")
        else:
            print(f"{n:>8} | {elapsed:>10.2f} | {'---':>8} | ---")

        prev_time = elapsed


if __name__ == "__main__":
    verify_merge_sort()
```

Recurrence: `T(n) = 2T(n/2) + cn` (c is a constant)

### 5.3 Example: Strassen's Matrix Multiplication

Standard matrix multiplication performs 8 submatrix multiplications of size n/2, giving T(n) = 8T(n/2) + O(n^2) -> O(n^3).
Strassen reduces the multiplication count to 7: T(n) = 7T(n/2) + O(n^2) -> O(n^{log_2 7}) ~ O(n^{2.807}).

```
Standard matrix multiplication vs Strassen:

  Standard:                    Strassen:
  a = 8, b = 2, f(n) = n^2    a = 7, b = 2, f(n) = n^2
  n^(log_2 8) = n^3            n^(log_2 7) ~ n^2.807
  Case 1 -> Theta(n^3)         Case 1 -> Theta(n^2.807)

  -> Reducing multiplications by just 1 yields asymptotic speedup
```

---

## 6. Master Theorem

### 6.1 General Form

The Master Theorem solves divide-and-conquer recurrences of the form T(n) = aT(n/b) + f(n) by simply comparing f(n) with n^{log_b(a)}.

```
T(n) = aT(n/b) + f(n)

  a : number of recursive calls (a >= 1)
  b : factor by which problem size shrinks (b > 1)
  f(n) : cost of dividing/merging (non-negative)

  Key value: n^(log_b(a))
    -> This corresponds to the "total number of leaves in the recursion tree"
    -> The recursion tree has depth log_b(n), branching by factor a at each level
    -> Number of leaves = a^(log_b(n)) = n^(log_b(a))
```

### 6.2 The Three Cases

```
Case decision flowchart:

  Compare f(n) with n^(log_b(a))
         |
    +----+----------------+
    v    v                v
 Case 1  Case 2         Case 3
 f(n) is f(n) is        f(n) is
 smaller  comparable     larger

Case 1: f(n) = O(n^(log_b(a) - epsilon))  (epsilon > 0)
  -> Leaf cost dominates
  -> T(n) = Theta(n^(log_b(a)))
  -> Intuition: recursion "fans out" fast, leaf count determines cost

Case 2: f(n) = Theta(n^(log_b(a)))
  -> Cost is uniform across levels
  -> T(n) = Theta(n^(log_b(a)) * log n)
  -> Intuition: equal cost at each level x number of levels = x log n

Case 3: f(n) = Omega(n^(log_b(a) + epsilon))  (epsilon > 0)
  and regularity condition: a*f(n/b) <= c*f(n) (c < 1, sufficiently large n)
  -> Root cost dominates
  -> T(n) = Theta(f(n))
  -> Intuition: "merge cost" decays rapidly at each level
```

**Why is the regularity condition (Case 3) needed?**: Even if f(n) is asymptotically larger than n^{log_b(a)}, the total may diverge if f decreases irregularly. The regularity condition guarantees that "f's cost decreases by a constant fraction at each level," ensuring convergence of the geometric series.

### 6.3 Application Examples of the Master Theorem

```python
"""
Systematically demonstrate applications of the Master Theorem.
For each example, identify a, b, f(n), and determine which case applies.
"""

import math


def master_theorem_analyze(
    a: int, b: int, f_desc: str, f_degree: float, algorithm: str
) -> None:
    """Perform case determination using the Master Theorem.

    Args:
        a: Number of recursive calls
        b: Problem size reduction factor
        f_desc: Description string of f(n)
        f_degree: Degree when f(n) is Theta(n^f_degree)
        algorithm: Algorithm name
    """
    critical_exp = math.log(a) / math.log(b)  # log_b(a)
    print(f"--- {algorithm} ---")
    print(f"  Recurrence: T(n) = {a}T(n/{b}) + {f_desc}")
    print(f"  a={a}, b={b}, n^(log_{b}({a})) = n^{critical_exp:.3f}")
    print(f"  f(n) = {f_desc} -> degree {f_degree}")

    if f_degree < critical_exp:
        print(f"  f(n) degree {f_degree} < {critical_exp:.3f} -> Case 1")
        print(f"  T(n) = Theta(n^{critical_exp:.3f})")
    elif abs(f_degree - critical_exp) < 0.001:
        print(f"  f(n) degree {f_degree} ~ {critical_exp:.3f} -> Case 2")
        print(f"  T(n) = Theta(n^{critical_exp:.3f} * log n)")
    else:
        print(f"  f(n) degree {f_degree} > {critical_exp:.3f} -> Case 3")
        print(f"  T(n) = Theta({f_desc})")
    print()


def main() -> None:
    print("=== Master Theorem Application Examples ===\n")

    # Example 1: Merge sort
    master_theorem_analyze(2, 2, "n", 1.0, "Merge Sort")

    # Example 2: Binary search
    master_theorem_analyze(1, 2, "1", 0.0, "Binary Search")

    # Example 3: Karatsuba multiplication
    master_theorem_analyze(3, 2, "n", 1.0, "Karatsuba Multiplication")

    # Example 4: Strassen matrix multiplication
    master_theorem_analyze(7, 2, "n^2", 2.0, "Strassen Matrix Multiplication")

    # Example 5: Binary tree traversal
    master_theorem_analyze(2, 2, "1", 0.0, "Binary Tree Traversal")

    # Example 6: T(n) = 4T(n/2) + n^2
    master_theorem_analyze(4, 2, "n^2", 2.0, "4T(n/2) + n^2")

    # Example 7: T(n) = 4T(n/2) + n^3
    master_theorem_analyze(4, 2, "n^3", 3.0, "4T(n/2) + n^3")

    # Example 8: Quickselect (average)
    master_theorem_analyze(1, 2, "n", 1.0, "Quickselect (average)")


if __name__ == "__main__":
    main()
```

### 6.4 Cases Where the Master Theorem Cannot Be Applied

The Master Theorem has clear limitations. It cannot be applied in the following cases:

**1. When f(n) does not differ polynomially (gap case)**

```
T(n) = 2T(n/2) + n log n

  a=2, b=2 -> n^(log_2(2)) = n
  f(n) = n log n

  n log n is larger than n^1 but smaller than n^(1+epsilon) (for any epsilon > 0).
  -> Falls between Case 2 and Case 3; standard Master Theorem is inapplicable.

  Solution: Apply the Akra-Bazzi theorem or solve directly with the recursion tree method.
  Result: T(n) = Theta(n log^2 n)
```

**2. When the problem size split is uneven**

```
T(n) = T(n/3) + T(2n/3) + n

  Recursive call sizes differ -> Not in the form T(n) = aT(n/b).
  -> Master Theorem is inapplicable.

  Solution: Solve with the recursion tree method.
  The deepest path is n -> (2/3)n -> (2/3)^2 n -> ... -> 1,
  with depth log_{3/2}(n). Cost at each level is O(n).
  Result: T(n) = O(n log n)
```

**3. When a < 1 or b <= 1**

The Master Theorem assumes a >= 1 and b > 1. It cannot be applied when these conditions are not met.

---

## 7. Recursion Tree Method

The recursion tree method visualizes the expansion of a recurrence as a tree structure and sums the costs at each level. It is also the most powerful tool for understanding the intuition behind the Master Theorem.

### 7.1 Recursion Tree for T(n) = 2T(n/2) + n

```
Level 0:              n                           -> Cost: n
                    /     \
Level 1:        n/2       n/2                     -> Cost: n/2 + n/2 = n
                / \        / \
Level 2:    n/4  n/4   n/4  n/4                  -> Cost: 4*(n/4) = n
              / \  / \  / \  / \
Level 3: n/8 ...                                  -> Cost: 8*(n/8) = n
             :
             :
Level k:  2^k nodes, each of size n/2^k          -> Cost: n

Level log_2(n): n leaves, each of size 1          -> Cost: n

Total: n * (log_2(n) + 1) = Theta(n log n)

Why cost at each level is n:
  Level k has 2^k nodes, each of size n/2^k.
  The "merge cost" at each node is proportional to n/2^k.
  2^k * (n/2^k) = n -> independent of level.
```

### 7.2 Recursion Tree for T(n) = 3T(n/4) + cn^2

```
Level 0:                cn^2                          -> Cost: cn^2
                      /   |   \
Level 1:      c(n/4)^2  c(n/4)^2  c(n/4)^2          -> Cost: 3c(n/4)^2 = (3/16)cn^2
               / | \    / | \    / | \
Level 2:    each c(n/16)^2                            -> Cost: 9c(n/16)^2 = (3/16)^2 cn^2
              :
Level k:    3^k nodes, each of size n/4^k            -> Cost: (3/16)^k * cn^2

  Total: cn^2 * Sum_{k=0}^{inf} (3/16)^k
       = cn^2 * 1/(1 - 3/16)
       = cn^2 * 16/13
       = Theta(n^2)

  Why the geometric series converges: Because the common ratio 3/16 < 1.
  This corresponds to Case 3: f(n) = n^2 dominates, and cost
  decays rapidly at deeper levels.
```

### 7.3 Recursion Tree for T(n) = T(n/3) + T(2n/3) + n (Uneven Split)

```
Level 0:                    n                           -> Cost: n
                          /     \
Level 1:             n/3       2n/3                     -> Cost: n/3 + 2n/3 = n
                     / \        / \
Level 2:        n/9  2n/9  2n/9  4n/9                  -> Cost: n
                 :                  :
                 :                  :

  Shortest path: n -> n/3 -> n/9 -> ... -> 1   depth log_3 n
  Longest path:  n -> 2n/3 -> 4n/9 -> ... -> 1  depth log_{3/2} n

  Cost at each level is at most n (slightly less near the leaves).
  Number of levels is up to log_{3/2} n.

  Total: O(n * log_{3/2} n) = O(n log n)

  -> Even with uneven splits, if the total cost per level is O(n),
    the overall result is O(n log n).
```

---

## 8. Substitution Method (Induction)

The substitution method "guesses" the solution to a recurrence and proves correctness by mathematical induction.

### 8.1 Procedure

1. **Guess the solution** (from recursion tree results or experience)
2. **State the inductive hypothesis**: Assume T(k) <= c * g(k) for all k < n
3. **Prove the inductive step**: Derive T(n) <= c * g(n)
4. **Verify the base case**: Confirm there exists a c such that T(n_0) <= c * g(n_0)

### 8.2 Example: Prove T(n) = 2T(n/2) + n is O(n log n)

```
Guess: T(n) <= c * n * log(n)  (c is an appropriate constant)

Inductive hypothesis: For all k < n, T(k) <= c * k * log(k)

Inductive step:
  T(n) = 2T(n/2) + n
       <= 2 * c * (n/2) * log(n/2) + n     (applying inductive hypothesis)
       = c * n * (log(n) - log(2)) + n      (using log(n/2) = log(n) - 1)
       = c * n * log(n) - c * n + n
       = c * n * log(n) - (c - 1) * n

  If c >= 1, then -(c-1)*n <= 0, so:
       <= c * n * log(n)

  Therefore T(n) <= c * n * log(n) holds. QED

Base case: Let T(1) = d (constant), then
  c * 1 * log(1) = 0, so T(1) <= c * 1 * log(1) does not hold.
  -> Take n >= 2 as base case: T(2) <= c * 2 * log(2) = 2c.
  T(2) = 2T(1) + 2 = 2d + 2, so setting c >= d + 1 suffices.
```

### 8.3 Pitfalls of the Substitution Method

```python
"""
Common mistakes with the substitution method.
"""

# Mistake 1: Setting the inductive hypothesis too weak
#
# Guess: T(n) = O(n) for T(n) = 2T(n/2) + n
#
# T(n) = 2T(n/2) + n
#      <= 2 * c * (n/2) + n
#      = cn + n
#      = (c+1)n
#      != cn   <- c "grows," so the proof fails
#
# -> This is correct. T(n) = Theta(n log n), not O(n).

# Mistake 2: Ignoring lower-order terms and falsely concluding success
#
# Guess: T(n) <= cn for T(n) = T(n/2) + T(n/2) + 1
#
# T(n) <= c(n/2) + c(n/2) + 1 = cn + 1  <- "almost cn" but not <= cn!
#
# Correct approach: Use a stronger hypothesis T(n) <= cn - d
# T(n) <= c(n/2) - d + c(n/2) - d + 1 = cn - 2d + 1 <= cn - d (if d >= 1)
```

---

## 9. Common Recursion Patterns

### Pattern 1: Linear Recursion T(n) = T(n-1) + O(1) -> O(n)

```python
def factorial(n: int) -> int:
    """Compute factorial recursively. T(n) = T(n-1) + O(1) -> O(n)

    Why O(n): Recursion depth is n, and cost at each level is O(1).
    """
    if n <= 1:
        return 1
    return n * factorial(n - 1)


# Verification
assert factorial(0) == 1
assert factorial(1) == 1
assert factorial(5) == 120
assert factorial(10) == 3628800
```

### Pattern 2: Linear Recursion (O(n) work per level) T(n) = T(n-1) + O(n) -> O(n^2)

```python
def selection_sort(arr: list[int]) -> list[int]:
    """Selection sort. T(n) = T(n-1) + O(n) -> O(n^2)

    Why O(n^2): Each step finds the minimum among remaining elements in O(n),
    repeated n times. Expanding the recurrence:
    T(n) = n + (n-1) + (n-2) + ... + 1 = n(n+1)/2 = Theta(n^2).
    """
    arr = arr[:]
    for i in range(len(arr)):
        min_idx = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr


# Verification
assert selection_sort([3, 1, 4, 1, 5]) == [1, 1, 3, 4, 5]
assert selection_sort([]) == []
assert selection_sort([1]) == [1]
```

### Pattern 3: Binary Recursion (divide only) T(n) = T(n/2) + O(1) -> O(log n)

```python
def binary_search(arr: list[int], target: int) -> int:
    """Binary search. T(n) = T(n/2) + O(1) -> O(log n)

    Why O(log n): The search range halves at each step,
    requiring log_2(n) comparisons until the range becomes 1.
    """
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1


# Verification
arr = [1, 3, 5, 7, 9, 11, 13]
assert binary_search(arr, 7) == 3
assert binary_search(arr, 1) == 0
assert binary_search(arr, 13) == 6
assert binary_search(arr, 4) == -1
```

### Pattern 4: Exponentiation by Squaring T(n) = T(n/2) + O(1) -> O(log n)

```python
def power(x: float, n: int) -> float:
    """Exponentiation by squaring. T(n) = T(n/2) + O(1) -> O(log n)

    Why O(log n): The exponent is halved at each step, so recursion
    depth is log_2(n). Each level performs only one multiplication (O(1)).

    x^n = (x^(n/2))^2       (n is even)
    x^n = x * (x^(n-1/2))^2 (n is odd)
    """
    if n == 0:
        return 1
    if n < 0:
        return 1 / power(x, -n)
    if n % 2 == 0:
        half = power(x, n // 2)
        return half * half
    else:
        return x * power(x, n - 1)


# Verification
assert power(2, 10) == 1024
assert power(3, 0) == 1
assert abs(power(2, -1) - 0.5) < 1e-10
```

### Pattern 5: Multi-branch Recursion T(n) = T(n-1) + T(n-2) + O(1) -> O(phi^n)

```python
def fibonacci_naive(n: int) -> int:
    """Naive Fibonacci. T(n) = T(n-1) + T(n-2) + O(1) -> O(phi^n)

    Why exponential time: Each node in the recursion tree has 2 children,
    and the tree extends to depth n, forming a "nearly" complete binary tree.
    More precisely, growth follows the golden ratio
    phi = (1 + sqrt(5)) / 2 ~ 1.618 raised to the power n.
    """
    if n <= 1:
        return n
    return fibonacci_naive(n - 1) + fibonacci_naive(n - 2)


def fibonacci_dp(n: int) -> int:
    """Dynamic programming Fibonacci. O(n) time, O(1) space.

    Eliminating redundant computation via memoization or DP table
    improves from O(phi^n) to O(n).
    """
    if n <= 1:
        return n
    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr
    return curr


# Verification
for i in range(10):
    assert fibonacci_naive(i) == fibonacci_dp(i)
```

### Pattern 6: Exponential Recursion T(n) = 2T(n-1) + O(1) -> O(2^n)

```python
def hanoi(n: int, source: str = "A", target: str = "C", auxiliary: str = "B") -> int:
    """Tower of Hanoi. T(n) = 2T(n-1) + O(1) -> O(2^n)

    Why O(2^n): Moving n disks requires moving the top n-1 disks twice
    and the largest disk once.
    T(n) = 2T(n-1) + 1 -> T(n) = 2^n - 1.

    Returns:
        Number of moves
    """
    if n == 0:
        return 0
    moves = 0
    moves += hanoi(n - 1, source, auxiliary, target)
    moves += 1  # Move the largest disk
    moves += hanoi(n - 1, auxiliary, target, source)
    return moves


# Verification: T(n) = 2^n - 1
for n in range(1, 15):
    assert hanoi(n) == 2**n - 1, f"n={n}: {hanoi(n)} != {2**n - 1}"

print("Tower of Hanoi move counts:")
for n in [1, 5, 10, 15, 20]:
    print(f"  n={n:>2}: {2**n - 1:>8} moves")
```

---

## 10. Comparison Tables

### Table 1: Detailed Comparison of the Master Theorem's Three Cases

| Case | Condition | Result | Intuitive Explanation | Understanding via Recursion Tree |
|------|-----------|--------|-----------------------|----------------------------------|
| Case 1 | f(n) = O(n^{log_b(a) - epsilon}) | Theta(n^{log_b(a)}) | Leaf cost dominates | Cost increases toward lower levels; total at leaves determines overall cost |
| Case 2 | f(n) = Theta(n^{log_b(a)}) | Theta(n^{log_b(a)} * log n) | Uniform across levels | Equal cost at all levels, multiplied by number of levels |
| Case 3 | f(n) = Omega(n^{log_b(a) + epsilon}) + regularity condition | Theta(f(n)) | Root dominates | Cost increases toward upper levels; root's contribution dominates |

### Table 2: Representative Recurrences and Their Solutions

| Recurrence | Solution | Algorithm Example | Basis |
|------------|----------|-------------------|-------|
| T(n) = T(n-1) + O(1) | O(n) | Linear scan, factorial | Sum = 1+1+...+1 = n |
| T(n) = T(n-1) + O(n) | O(n^2) | Selection sort, insertion sort | Sum = n+(n-1)+...+1 |
| T(n) = T(n/2) + O(1) | O(log n) | Binary search | Master Theorem Case 2 |
| T(n) = T(n/2) + O(n) | O(n) | Quickselect (average) | Master Theorem Case 3 |
| T(n) = 2T(n/2) + O(1) | O(n) | Binary tree traversal | Master Theorem Case 1 |
| T(n) = 2T(n/2) + O(n) | O(n log n) | Merge sort | Master Theorem Case 2 |
| T(n) = 2T(n/2) + O(n^2) | O(n^2) | Inefficient divide-and-conquer | Master Theorem Case 3 |
| T(n) = 3T(n/2) + O(n) | O(n^{1.585}) | Karatsuba multiplication | Master Theorem Case 1 |
| T(n) = 7T(n/2) + O(n^2) | O(n^{2.807}) | Strassen matrix multiplication | Master Theorem Case 1 |
| T(n) = 2T(n-1) + O(1) | O(2^n) | Tower of Hanoi | Expansion: 2^n - 1 |
| T(n) = T(n-1) + T(n-2) + O(1) | O(phi^n) | Naive Fibonacci | Characteristic equation solution |

### Table 3: Growth Rate Comparison of Complexity Classes

| n | O(1) | O(log n) | O(n) | O(n log n) | O(n^2) | O(2^n) |
|---|------|----------|------|------------|--------|--------|
| 1 | 1 | 0 | 1 | 0 | 1 | 2 |
| 10 | 1 | 3.3 | 10 | 33 | 100 | 1,024 |
| 100 | 1 | 6.6 | 100 | 664 | 10,000 | 1.27 x 10^30 |
| 1,000 | 1 | 10.0 | 1,000 | 9,966 | 1,000,000 | -- |
| 10,000 | 1 | 13.3 | 10,000 | 132,877 | 100,000,000 | -- |
| 100,000 | 1 | 16.6 | 100,000 | 1,660,964 | 10,000,000,000 | -- |

"--" indicates values so astronomically large as to be computationally infeasible.

---

## 11. Anti-patterns

### Anti-pattern 1: Applying the Master Theorem Without Checking Conditions

```python
"""
Example of falling into the "gap case" of the Master Theorem.
"""

# BAD: Trying to directly apply Master Theorem to T(n) = 2T(n/2) + n log n
#
# a=2, b=2 -> n^(log_2(2)) = n^1 = n
# f(n) = n log n
#
# Does Case 2 apply? -> Case 2 requires f(n) = Theta(n^(log_b(a))).
#   n log n != Theta(n), so Case 2 does not apply.
#
# Does Case 3 apply? -> Case 3 requires f(n) = Omega(n^(1+epsilon)).
#   n log n = O(n^(1+epsilon)) for any epsilon > 0, so Case 3 does not apply either.
#
# -> Standard Master Theorem is inapplicable (gap case)
# -> Apply Extended Master Theorem: for f(n) = Theta(n log^k n) with k=1,
#   T(n) = Theta(n log^(k+1) n) = Theta(n log^2 n)
#
# GOOD: Always verify which of the 3 cases applies before using it,
#        and use recursion tree or Akra-Bazzi theorem when none applies.
```

### Anti-pattern 2: Confusing Recursion Depth with Total Call Count

```python
"""
Recursion "depth" and "total call count" are distinct concepts.
"""

# BAD: Mistakenly thinking "binary recursion means O(log n)"
def count_all(n: int) -> int:
    """T(n) = T(n-1) + T(n-2) + O(1)

    Wrong reasoning: "Recursion splits into 2, so O(log n)"
    Correct analysis: Depth is O(n), call count is O(phi^n)
    """
    if n <= 0:
        return 0
    return 1 + count_all(n - 1) + count_all(n - 2)


# GOOD: Distinguish between depth and call count
#
#   Recursion tree depth: Length of deepest path -> affects stack usage
#   Call count: Total number of nodes -> affects time complexity
#
#   Example: Fibonacci recursion
#     Depth: O(n) -- leftmost path goes n -> n-1 -> n-2 -> ... -> 0
#     Call count: O(phi^n) -- nearly complete binary tree structure
```

### Anti-pattern 3: Equating Average Case with "Typical Case"

```python
"""
Average case is the expected value based on a probability distribution,
which does not necessarily match performance on "common inputs."
"""

# BAD: Assuming "quicksort averages O(n log n), so nearly all
#        inputs will be O(n log n)"

# Correct understanding:
# 1. Average case is the expected value under the assumption that
#    "all size-n inputs appear with equal probability"
# 2. In real applications, input may be biased
#    Example: a log processing system frequently receives nearly-sorted data
# 3. In environments where adversarial input exists,
#    worst-case guarantees are necessary

# GOOD: Choose algorithms considering input distribution
# - Random input is guaranteed -> Average case evaluation is valid
# - Adversarial input possible -> Worst-case guarantees needed
# - Specific patterns are common -> Evaluate performance on those patterns individually
```

### Anti-pattern 4: Ignoring Recursion Overhead

```python
"""
Even when asymptotically equivalent, recursion overhead
affects the constant factor.
"""

import time


def sum_recursive(n: int) -> int:
    """Recursive sum. Time complexity: O(n)."""
    if n <= 0:
        return 0
    return n + sum_recursive(n - 1)


def sum_iterative(n: int) -> int:
    """Iterative sum. Time complexity: O(n)."""
    total = 0
    for i in range(1, n + 1):
        total += i
    return total


def compare_overhead() -> None:
    """Measure overhead difference between recursion and iteration.

    Both are O(n) asymptotically, but the recursive version has a larger
    constant factor due to function call overhead (stack frame creation/destruction).
    """
    import sys
    sys.setrecursionlimit(20000)

    n = 10000
    start = time.perf_counter()
    result_rec = sum_recursive(n)
    time_rec = time.perf_counter() - start

    start = time.perf_counter()
    result_iter = sum_iterative(n)
    time_iter = time.perf_counter() - start

    assert result_rec == result_iter

    print(f"n = {n}")
    print(f"  Recursive: {time_rec * 1000:.2f} ms")
    print(f"  Iterative: {time_iter * 1000:.2f} ms")
    print(f"  Ratio: {time_rec / time_iter:.1f}x")
    print(f"  -> Asymptotically the same O(n), but constant factors differ")


if __name__ == "__main__":
    compare_overhead()
```

---

## 12. Edge Case Analysis

### Edge Case 1: Limitations of Asymptotic Analysis for Very Small Input Sizes

Asymptotic notation describes behavior as n -> infinity, so for small n, theory and reality can diverge.

```python
"""
Demonstrates cases where an asymptotically "inferior" algorithm
outperforms an asymptotically "superior" one on small inputs.

Why this happens:
When the constant factor of an O(n^2) algorithm is much smaller
than that of an O(n log n) algorithm, the constant factor difference
dominates for small n.
"""

import time
import random


def insertion_sort(arr: list[int]) -> list[int]:
    """Insertion sort. Worst case O(n^2) but small constant factor."""
    arr = arr[:]
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr


def merge_sort_full(arr: list[int]) -> list[int]:
    """Merge sort. O(n log n) but somewhat larger constant factor."""
    if len(arr) <= 1:
        return arr[:]
    mid = len(arr) // 2
    left = merge_sort_full(arr[:mid])
    right = merge_sort_full(arr[mid:])
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def find_crossover_point() -> None:
    """Experimentally find the crossover point between insertion sort and merge sort.

    Expected result: Crossover occurs around n = 20-50.
    This is why Tim Sort (Python's built-in sort) uses insertion sort
    for small sub-arrays.
    """
    print("=== Small Input Size Comparison ===")
    print(f"{'n':>6} | {'Insertion (ms)':>14} | {'Merge sort (ms)':>14} | {'Faster':>8}")
    print("-" * 55)

    trials = 1000
    for n in [5, 10, 15, 20, 30, 50, 100, 200, 500]:
        # Measure insertion sort
        total_ins = 0
        for _ in range(trials):
            arr = random.sample(range(n * 10), n)
            start = time.perf_counter()
            insertion_sort(arr)
            total_ins += time.perf_counter() - start

        # Measure merge sort
        total_merge = 0
        for _ in range(trials):
            arr = random.sample(range(n * 10), n)
            start = time.perf_counter()
            merge_sort_full(arr)
            total_merge += time.perf_counter() - start

        ins_ms = total_ins / trials * 1000
        merge_ms = total_merge / trials * 1000
        winner = "Insert" if ins_ms < merge_ms else "Merge"
        print(f"{n:>6} | {ins_ms:>14.4f} | {merge_ms:>14.4f} | {winner:>8}")

    print()
    print("  -> For small n, O(n^2) insertion sort beats O(n log n) merge sort.")
    print("    This is a practical consequence of asymptotic notation ignoring constant factors.")


if __name__ == "__main__":
    find_crossover_point()
```

**Lesson**: Asymptotic notation is a tool for evaluating algorithm "scalability," not for guaranteeing absolute performance on small n. Practical sorting algorithms (Tim Sort, Introsort, etc.) switch to insertion sort for small sub-arrays, combining asymptotic superiority with constant-factor superiority.

### Edge Case 2: When Base Case Cost Cannot Be Ignored in Recurrences

```python
"""
Demonstrates that when the base case cost is not O(1),
the recurrence solution changes.
"""


def matrix_multiply_recursive(
    A: list[list[float]], B: list[list[float]], n: int
) -> list[list[float]]:
    """Recursive matrix multiplication (simple divide and conquer).

    When base case is 1x1 matrix: T(1) = O(1)
    -> T(n) = 8T(n/2) + O(n^2) -> O(n^3)

    If base case is k x k (k constant) with direct computation:
    T(k) = O(k^3) ~ O(1)  (since k is constant)
    The asymptotic result stays the same, but constant factor improves.

    Why this matters: In implementation, setting an appropriate recursion
    base case reduces function call overhead. Strassen's algorithm also
    switches to standard O(n^3) matrix multiplication when n becomes small enough.
    """
    if n == 1:
        return [[A[0][0] * B[0][0]]]

    mid = n // 2

    # Extract sub-matrices (simplified using slicing)
    A11 = [row[:mid] for row in A[:mid]]
    A12 = [row[mid:] for row in A[:mid]]
    A21 = [row[:mid] for row in A[mid:]]
    A22 = [row[mid:] for row in A[mid:]]

    B11 = [row[:mid] for row in B[:mid]]
    B12 = [row[mid:] for row in B[:mid]]
    B21 = [row[:mid] for row in B[mid:]]
    B22 = [row[mid:] for row in B[mid:]]

    # 8 recursive calls
    C11 = matrix_add(
        matrix_multiply_recursive(A11, B11, mid),
        matrix_multiply_recursive(A12, B21, mid),
    )
    C12 = matrix_add(
        matrix_multiply_recursive(A11, B12, mid),
        matrix_multiply_recursive(A12, B22, mid),
    )
    C21 = matrix_add(
        matrix_multiply_recursive(A21, B11, mid),
        matrix_multiply_recursive(A22, B21, mid),
    )
    C22 = matrix_add(
        matrix_multiply_recursive(A21, B12, mid),
        matrix_multiply_recursive(A22, B22, mid),
    )

    # Combine results
    result = []
    for i in range(mid):
        result.append(C11[i] + C12[i])
    for i in range(mid):
        result.append(C21[i] + C22[i])
    return result


def matrix_add(
    A: list[list[float]], B: list[list[float]]
) -> list[list[float]]:
    """Matrix addition. O(n^2)."""
    n = len(A)
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]


def verify_recursive_matrix_multiply() -> None:
    """Verify correctness of recursive matrix multiplication."""
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    C = matrix_multiply_recursive(A, B, 2)
    # Expected: [[19, 22], [43, 50]]
    assert C == [[19, 22], [43, 50]], f"Got {C}"

    # 4x4 matrix test
    A4 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    B4 = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    C4 = matrix_multiply_recursive(A4, B4, 4)
    assert C4 == B4, "Product with identity matrix should equal original"

    print("Recursive matrix multiplication: all tests passed")


if __name__ == "__main__":
    verify_recursive_matrix_multiply()
```

### Edge Case 3: Scenarios Where "Negative Savings" Occur in Amortized Analysis

```
When amortized cost is set inappropriately, "savings" can go negative.

Example: Dynamic array with 1.5x expansion instead of doubling

  Expansion frequency is higher than the doubling strategy,
  so an amortized cost of 3 per operation may not suffice.

  To correctly analyze with the accounting method:
  - Compute the required "prepayment" based on expansion factor alpha
  - alpha = 2: amortized cost 3 is sufficient
  - alpha = 1.5: amortized cost needs to be higher

  Correct computation:
  With expansion factor alpha, each element is copied at most 1/(alpha-1) times.
  alpha = 2 -> each element copied at most 1/(2-1) = 1 time -> amortized cost ~ 3
  alpha = 1.5 -> each element copied at most 1/(1.5-1) = 2 times -> amortized cost ~ 5

  Lesson: Amortized analysis results depend on the data structure's implementation strategy.
        Changing the expansion factor changes the amortized cost.
```

---

## 13. Practice Problems

### Basic Level

**Problem B1: Formulating Recurrences**

For each of the following code snippets, formulate the recurrence relation and determine the complexity.

```python
# (a) Finding maximum recursively
def find_max(arr: list[int], n: int) -> int:
    if n == 1:
        return arr[0]
    return max(arr[n - 1], find_max(arr, n - 1))


# (b) Sum via divide and conquer
def divide_sum(arr: list[int], left: int, right: int) -> int:
    if left == right:
        return arr[left]
    mid = (left + right) // 2
    return divide_sum(arr, left, mid) + divide_sum(arr, mid + 1, right)


# (c) Naive recursive exponentiation
def power_naive(x: float, n: int) -> float:
    if n == 0:
        return 1
    return x * power_naive(x, n - 1)
```

**Solution**:

```
(a) T(n) = T(n-1) + O(1)
    Expansion: T(n) = T(n-1) + c = T(n-2) + 2c = ... = T(1) + (n-1)c
    -> T(n) = O(n)

(b) T(n) = 2T(n/2) + O(1)
    Master Theorem: a=2, b=2, f(n)=O(1), n^(log_2 2) = n
    f(n) = O(n^(1-epsilon)) -> Case 1 -> T(n) = Theta(n)

(c) T(n) = T(n-1) + O(1)
    -> T(n) = O(n)
    Note: Can be improved to O(log n) using exponentiation by squaring
```

---

**Problem B2: Applying the Master Theorem**

Apply the Master Theorem to each of the following recurrences and determine the complexity.

```
(a) T(n) = 9T(n/3) + n
(b) T(n) = T(2n/3) + 1
(c) T(n) = 3T(n/4) + n log n
(d) T(n) = 2T(n/4) + sqrt(n)
```

**Solution**:

```
(a) a=9, b=3, f(n)=n, n^(log_3 9) = n^2
    f(n) = n = O(n^(2-epsilon)) with epsilon=1 -> Case 1
    T(n) = Theta(n^2)

(b) a=1, b=3/2, f(n)=1, n^(log_{3/2} 1) = n^0 = 1
    f(n) = 1 = Theta(n^0) -> Case 2
    T(n) = Theta(log n)

(c) a=3, b=4, f(n)=n log n, n^(log_4 3) ~ n^0.793
    f(n) = n log n = Omega(n^(0.793+epsilon)) -> Case 3 candidate
    Regularity condition: 3 * (n/4) * log(n/4) <= c * n * log n
              (3/4) * n * (log n - log 4) <= c * n * log n
              -> c = 3/4 < 1, holds for sufficiently large n
    T(n) = Theta(n log n)

(d) a=2, b=4, f(n)=sqrt(n), n^(log_4 2) = n^(1/2) = sqrt(n)
    f(n) = sqrt(n) = Theta(n^(1/2)) -> Case 2
    T(n) = Theta(sqrt(n) * log n)
```

---

**Problem B3: Identifying Worst and Best Cases**

For the following algorithm, provide example inputs for the worst and best cases and determine the complexity of each.

```python
def linear_search_first_even(arr: list[int]) -> int:
    """Find and return the first even number in the array. Return -1 if not found."""
    for i, val in enumerate(arr):
        if val % 2 == 0:
            return val
    return -1
```

**Solution**:

```
Best case: arr = [2, 1, 3, 5, ...] (first element is even)
  -> Terminates after 1 comparison. B(n) = O(1)

Worst case: arr = [1, 3, 5, 7, ...] (all odd)
  -> Requires n comparisons. W(n) = O(n)

Average case: Assuming each element is even with probability p = 1/2,
  Probability that first even is at position k = (1/2)^k * (1/2) = (1/2)^(k+1)
  Expected comparisons = Sum_{k=0}^{n-1} (k+1) * (1/2)^(k+1) ~ 2 - (n+2)/2^n
  -> A(n) = O(1) (expected value converges to a constant)
```

---

### Advanced Level

**Problem A1: Analysis Using the Recursion Tree Method**

Determine the complexity of T(n) = T(n/4) + T(3n/4) + cn using the recursion tree method.

**Hint**: Find the depth of the shortest and longest paths, and analyze the cost at each level.

**Solution**:

```
Recursion tree structure:
                        cn
                      /     \
                 cn/4       3cn/4           -> Total: cn
                /    \      /     \
           cn/16  3cn/16 3cn/16  9cn/16     -> Total: cn
              :                     :

  Shortest path: n -> n/4 -> n/16 -> ... -> 1  depth = log_4 n
  Longest path:  n -> 3n/4 -> 9n/16 -> ... -> 1  depth = log_{4/3} n

  Total cost at each level:
  Level 0: cn
  Level 1: c(n/4) + c(3n/4) = cn
  Level 2: cn (because the sum of all node sizes equals n)

  Upper bound on number of levels: log_{4/3} n

  -> T(n) = Theta(n log n)

  This result is intuitive: total cost O(n) per level x log n levels.
  Even with uneven splits, the property that "total size does not
  exceed n at each level" is maintained, yielding the same complexity as merge sort.
```

---

**Problem A2: Amortized Analysis in Practice**

For the following "binary counter," determine the amortized complexity of n INCREMENT operations using all three methods.

```python
class BinaryCounter:
    """k-bit binary counter."""

    def __init__(self, k: int) -> None:
        self.bits = [0] * k
        self.k = k

    def increment(self) -> int:
        """Increment counter by 1. Return number of bit flips."""
        flips = 0
        i = 0
        while i < self.k and self.bits[i] == 1:
            self.bits[i] = 0  # Carry
            flips += 1
            i += 1
        if i < self.k:
            self.bits[i] = 1
            flips += 1
        return flips
```

**Solution**:

```
Aggregate method:
  Count how many times each bit flips over n INCREMENTs.
  - Bit 0: flips every time -> n times
  - Bit 1: flips every 2nd time -> n/2 times
  - Bit 2: flips every 4th time -> n/4 times
  - Bit i: flips every 2^i times -> n/2^i times

  Total flips = Sum_{i=0}^{k-1} n/2^i < n * Sum_{i=0}^{inf} 1/2^i = 2n

  -> Total cost for n operations is O(n) -> O(1) per operation

Accounting method:
  Charge amortized cost 2 per INCREMENT.
  - Setting a bit to 1: cost 1 + savings 1
  - Resetting a bit to 0: spend 1 from savings

  Every reset was "prepaid" by a past set operation,
  so savings is always non-negative. -> Amortized cost O(1)

Potential method:
  Phi(D) = number of 1-bits in the counter

  When INCREMENT resets t bits and sets 1 bit:
  Actual cost = t + 1
  Phi change = (1 bit set - t bits reset) = 1 - t
  Amortized cost = (t + 1) + (1 - t) = 2

  -> Amortized cost per operation is O(1)
```

---

**Problem A3: Proof by Substitution**

Prove by the substitution method that T(n) = T(n/2) + T(n/4) + n is O(n).

**Solution**:

```
Guess: T(n) <= cn  (c is an appropriate constant)

Inductive hypothesis: For all k < n, T(k) <= ck

Inductive step:
  T(n) = T(n/2) + T(n/4) + n
       <= c(n/2) + c(n/4) + n     (inductive hypothesis)
       = cn/2 + cn/4 + n
       = (3/4)cn + n
       = cn - cn/4 + n
       = cn - (c/4 - 1)n

  When c/4 - 1 >= 0, i.e., c >= 4:
       <= cn

  Therefore with c >= 4, T(n) <= cn = O(n). QED

  Base case: T(1) = d -> c >= d suffices.
  Setting c = max(4, d) satisfies the entire proof.
```

---

### Advanced Level

**Problem D1: Applying the Akra-Bazzi Theorem**

Solve T(n) = T(n/3) + T(2n/3) + n using the Akra-Bazzi theorem.

**Hint**: The Akra-Bazzi theorem handles T(n) = Sum a_i T(n/b_i) + g(n),
finding p such that Sum a_i / b_i^p = 1, then T(n) = Theta(n^p (1 + integral_1^n g(u)/u^{p+1} du)).

**Solution**:

```
T(n) = T(n/3) + T(2n/3) + n

Akra-Bazzi condition: Find p such that (1/3)^p + (2/3)^p = 1.

  Try p = 1: 1/3 + 2/3 = 1  checkmark

  g(n) = n, so:
  T(n) = Theta(n^1 * (1 + integral_1^n u / u^2 du))
       = Theta(n * (1 + integral_1^n 1/u du))
       = Theta(n * (1 + ln n))
       = Theta(n log n)

  -> Matches the result obtained by the recursion tree method.
```

---

**Problem D2: Extended Master Theorem**

Determine the complexity of T(n) = 4T(n/2) + n^2 log n (standard Master Theorem is inapplicable).

**Hint**: Extended Master Theorem: When f(n) = Theta(n^{log_b a} * log^k n), T(n) = Theta(n^{log_b a} * log^{k+1} n).

**Solution**:

```
a = 4, b = 2, n^(log_2 4) = n^2
f(n) = n^2 log n = Theta(n^2 * log^1 n)

This is the case f(n) = Theta(n^(log_b a) * log^k n) with k = 1.

By Extended Master Theorem:
T(n) = Theta(n^2 * log^(1+1) n) = Theta(n^2 log^2 n)

Verification (recursion tree):
  Cost at level l: 4^l * (n/2^l)^2 * log(n/2^l)
                  = n^2 * (log n - l)
  Total: Sum_{l=0}^{log n} n^2 * (log n - l)
       = n^2 * Sum_{j=0}^{log n} j
       = n^2 * (log n)(log n + 1)/2
       = Theta(n^2 log^2 n)  checkmark
```

---

**Problem D3: Lower Bound Proof for Amortized Complexity**

Consider a dynamic array strategy that expands capacity by a factor of alpha > 1.
Prove that the total cost of n append operations is Omega(n),
and show that the amortized cost per operation is proportional to Theta(1/(alpha - 1)).

**Solution**:

```
Over n appends, expansion occurs at the following times:
  Capacity 1 -> alpha -> alpha^2 -> ... -> alpha^k (until alpha^k >= n)

  Number of expansions: k = ceil(log_alpha n)

  Copy cost at each expansion: number of elements just before expansion
  Total copy cost = 1 + alpha + alpha^2 + ... + alpha^{k-1}
                  = (alpha^k - 1) / (alpha - 1)
                  ~ n / (alpha - 1)

  Write cost: n

  Total cost = n + n/(alpha-1) = n * (1 + 1/(alpha-1)) = n * alpha/(alpha-1)

  Amortized cost per operation = alpha/(alpha-1)

  alpha = 2 -> 2/1 = 2   (2 per operation)
  alpha = 1.5 -> 1.5/0.5 = 3   (3 per operation)
  alpha -> 1 -> infinity   (diverges as expansion factor approaches 1)

  -> Reducing expansion factor alpha improves memory efficiency
    but increases copy frequency and time cost.
    This is a classic example of the space-time tradeoff.
```

---

## 14. FAQ

### Q1: What should I do when the Master Theorem cannot be applied?

**A:** There are mainly three alternative methods:

1. **Recursion tree method**: The most versatile. Can handle cases like T(n) = T(n/3) + T(2n/3) + n where splits are uneven. Compute the cost at each level and sum them.

2. **Akra-Bazzi theorem**: A generalization of the Master Theorem that handles T(n) = Sum a_i T(n/b_i) + g(n). However, g(n) must be polynomially bounded.

3. **Substitution method**: Guess and prove by induction. Used to rigorize guesses obtained from the recursion tree method. Its primary role is as a "final verification" that the guess is correct.

Practical approach: First use the recursion tree to get an estimate, then confirm with the Master Theorem (if applicable) or Akra-Bazzi theorem, and rigorously prove with the substitution method if needed.

### Q2: Does the base of a logarithm affect complexity?

**A:** Within asymptotic notation (O, Theta, Omega), it does not. This is based on the mathematical property:

```
log_a(n) = log_b(n) / log_b(a)
```

Since log_b(a) is a constant, differences in base are only constant factor differences. Asymptotic notation ignores constant factors, so O(log_2 n) = O(log_10 n) = O(ln n) = O(log n).

**However, note**: Outside asymptotic notation, the base matters. For example:
- Binary search performs exactly log_2 n comparisons
- Ternary search performs log_3 n comparisons
- When discussing actual comparison counts, the base should be specified

Also, different exponents of log do matter: O(log n) != O(log^2 n).

### Q3: Does converting recursion to iteration change the complexity?

**A:** Time complexity usually does not change. However, differences arise in the following aspects:

| Aspect | Recursive Version | Iterative Version |
|--------|-------------------|-------------------|
| Time complexity | Same | Same |
| Space complexity | O(recursion depth) stack | Controllable with explicit stack |
| Constant factor | Function call overhead present | No overhead |
| Tail recursion optimization | Language-dependent (Python: not supported) | Not needed |
| Stack overflow | Can occur with deep recursion | Does not occur |

In Python especially, recursion depth is limited to 1000 by default, so deep recursion should be converted to iteration or `sys.setrecursionlimit()` must be explicitly set.

### Q4: Is amortized complexity the same as average complexity?

**A:** They are different. This confusion is very common but should be clearly distinguished.

| Aspect | Amortized Complexity | Average Complexity |
|--------|---------------------|-------------------|
| Subject | Cost of the entire operation sequence | Expected value based on input probability distribution |
| Probability | Not used at all (deterministic) | Assumes input probability distribution |
| Guarantee | Worst-case guarantee | Probabilistic guarantee only |
| Example | Dynamic array: guarantees O(n) for n appends | Quicksort: expects O(n log n) for random input |

Amortized complexity is a guarantee that "the total of n operations is definitely at most O(n)." Even if a specific operation is slow, the entire sequence balances out. In contrast, average complexity is the "expected running time for typical input" and is not guaranteed in the worst case.

### Q5: Why is complexity analysis important? Isn't profiling sufficient?

**A:** Complexity analysis and profiling are complementary tools; neither alone is sufficient.

Advantages of complexity analysis:
- **Can predict performance before implementation**: Identify bottlenecks at design stage before writing code
- **Scalability prediction**: Can answer "what happens when input grows 10x?"
- **Hardware independent**: Differences negligible on today's fast machines can become critical at 10x input

Advantages of profiling:
- **Reflects constant factors and cache efficiency**: Measures actual performance invisible in asymptotic notation
- **Bottleneck identification**: Precisely identifies which parts of code are actually slow

Ideally, combine both: First use complexity analysis to decide the "big picture" (O(n log n) vs O(n^2) etc.), then use profiling for constant factor and implementation optimization.

### Q6: How does space complexity relate to time complexity?

**A:** In general, space complexity is at most equal to time complexity. This is because writing to each memory cell takes at least O(1) time.

```
S(n) <= T(n) always holds

However, the reverse does not hold:
  Example: Binary search -- T(n) = O(log n), S(n) = O(1)
  Example: Merge sort -- T(n) = O(n log n), S(n) = O(n)

Classic examples of time-space tradeoffs:
  Naive Fibonacci: T(n) = O(phi^n), S(n) = O(n) (recursion stack)
  Memoized Fibonacci: T(n) = O(n), S(n) = O(n) (cache)
  Iterative Fibonacci: T(n) = O(n), S(n) = O(1) (two variables)
```

---


## FAQ

### Q1: What is the most important point when learning this topic?

Gaining practical experience is paramount. Understanding deepens not just through theory, but by actually writing and running code to verify behavior.

### Q2: What common mistakes do beginners make?

Skipping fundamentals and jumping to applications. We recommend thoroughly understanding the basic concepts explained in this guide before moving to the next step.

### Q3: How is this knowledge applied in practice?

Knowledge of this topic is frequently used in everyday development work, particularly during code reviews and architecture design.

---

## 15. Summary

### 15.1 Overall Knowledge Map

```
Overall map of complexity analysis:

+--------------------------------------------------+
|                  Complexity Analysis              |
+--------------+---------------+--------------------+
|  Case Analysis| Amortized    | Recursion Analysis |
|              | Analysis      |                    |
| - Worst case | - Aggregate   | - Formulating      |
| - Average    |   method      |   recurrences      |
|   case       | - Accounting  | - Master Theorem   |
| - Best case  |   method      | - Recursion tree   |
|              | - Potential    | - Substitution     |
|              |   method      | - Akra-Bazzi       |
+--------------+---------------+--------------------+
         |                           |
  +------------------+    +------------------------+
  | Criteria for     |    | Criteria for evaluating|
  | algorithm        |    | data structures        |
  | selection        |    |                        |
  +------------------+    +------------------------+
```

### 15.2 Key Points

| Item | Key Point |
|------|-----------|
| Asymptotic notation | Distinguish among O (upper bound), Omega (lower bound), and Theta (tight bound) |
| Worst case | Emphasized when response time guarantees are needed. Essential for SLAs and real-time systems |
| Average case | Expected value assuming input probability distribution. Useful for practical performance evaluation |
| Amortized analysis | Evaluates cost of the entire operation sequence. Correctly handles "occasionally expensive operations" |
| Recurrences | Formulate cost relations from recursive structure. Starting point for analysis |
| Master Theorem | Standard solution for T(n) = aT(n/b) + f(n). Classified into 3 cases |
| Recursion tree | Visualize and sum costs at each level. Most intuitive method |
| Substitution | Guess and prove by induction. Used when rigor is required |
| Akra-Bazzi | Generalization of Master Theorem. Handles uneven splits |
| Practical connection | Asymptotic analysis provides the "big picture," profiling provides "details." Both are needed |

### 15.3 Checklist for Performing Complexity Analysis

1. **Was the recurrence formulated correctly?**: Verify base case, number of recursive calls, size of each call, and non-recursive costs
2. **Are the conditions for the applied theorem satisfied?**: Master Theorem's 3 cases, regularity condition
3. **Are cases distinguished?**: Make explicit whether worst, average, or best case is being discussed
4. **Have you missed situations requiring amortized analysis?**: Verify that individual operation worst case is not "overly pessimistic"
5. **Are constant factors practically important?**: Also consider constant factors in scenarios where small-n performance matters

---

## Recommended Next Guides

- [Space-Time Tradeoff -- Memoization and Bloom Filters](./02-space-time-tradeoff.md)
- [Sorting -- Quick/Merge/Heap Complexity Comparison](../02-algorithms/00-sorting.md)

---

## 16. References

1. Cormen, T.H., Leiserson, C.E., Rivest, R.L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. -- Chapter 3 "Growth of Functions," Chapter 4 "Divide-and-Conquer," Chapter 16 "Amortized Analysis." Provides rigorous definitions of asymptotic notation, proof of the Master Theorem, and detailed treatment of the three amortized analysis methods.

2. Akra, M. & Bazzi, L. (1998). "On the solution of linear recurrence equations." *Computational Optimization and Applications*, 10(2), 195-210. -- The original paper presenting a theorem for generally solving recurrences with uneven splits that the Master Theorem cannot handle.

3. Levitin, A. (2012). *Introduction to the Design and Analysis of Algorithms* (3rd ed.). Pearson. -- Explains recursion analysis techniques with abundant examples. One of the most accessible textbooks for beginners.

4. Sedgewick, R. & Flajolet, P. (2013). *An Introduction to the Analysis of Algorithms* (2nd ed.). Addison-Wesley. -- An advanced textbook that systematically covers mathematical methods for average case analysis. Particularly detailed on quicksort's average case derivation.

5. Tarjan, R.E. (1985). "Amortized Computational Complexity." *SIAM Journal on Algebraic and Discrete Methods*, 6(2), 306-318. -- The pioneering paper that formalized the concept of amortized analysis. Presents the prototype of the potential method.

6. Knuth, D.E. (1997). *The Art of Computer Programming, Volume 1: Fundamental Algorithms* (3rd ed.). Addison-Wesley. -- The classic that treats the mathematical foundations of complexity analysis most rigorously. Contains the origins and precise definitions of asymptotic notation.

7. MIT OpenCourseWare. "6.006 Introduction to Algorithms" and "6.046J Design and Analysis of Algorithms." -- MIT lecture materials on complexity analysis. Lecture videos and notes on recursion analysis, Master Theorem, and amortized analysis are freely available.

---

## References

- [MDN Web Docs](https://developer.mozilla.org/) - Web technology reference
- [Wikipedia](https://en.wikipedia.org/) - Technical concept overviews
