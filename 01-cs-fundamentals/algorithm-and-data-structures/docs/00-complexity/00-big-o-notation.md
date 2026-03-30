# Big-O Notation and the Foundations of Computational Complexity

> A systematic study of O notation, Omega notation, and Theta notation for mathematically expressing algorithm efficiency, along with space complexity and amortized complexity. This guide covers proof techniques using induction and common pitfalls.

---

## What You Will Learn in This Chapter

1. **O / Omega / Theta notation** — mathematical definitions and when to use each
2. **Little-o and little-omega notation** — meaning and applications
3. **Proof techniques using induction** for computational complexity
4. **Major complexity classes** — growth rates and concrete examples
5. **Space complexity** — evaluation methods and stack consumption in recursive calls
6. **Amortized complexity** — concepts and application to dynamic arrays
7. **Anti-patterns and edge cases** to be aware of
8. **Three tiers of exercises** for reinforcement

### Prerequisites

- Basic programming skills (ability to read and write Python)
- High school-level understanding of functions, inequalities, and limits
- Basic understanding of loops and recursion

### After Completing This Guide

- You will be able to accurately describe the complexity of any algorithm using O / Omega / Theta
- You will be able to rigorously prove complexity using reduction and induction
- You will be able to confidently discuss complexity in interviews and code reviews

---

## 1. Why Complexity Analysis Is Necessary

When discussing the "speed" of an algorithm, measuring execution time in seconds alone is insufficient. The reasons are:

1. **Hardware dependence**: The same algorithm runs at very different speeds on different machines
2. **Input dependence**: Even for the same input size, processing time varies depending on the data arrangement
3. **Invisible scalability issues**: An algorithm that is fast for n=100 may break down at n=1,000,000

Complexity analysis solves these problems. It provides a hardware-independent mathematical framework for describing how an algorithm behaves as the input size n grows large.

```
Concrete example: Enumerating all pairs from n elements

Method A: Nested loops      -> Operations ≈ n²/2
Method B: Optimized method  -> Operations ≈ n log n

n=100:       Method A ≈ 5,000       Method B ≈ 664       (Ratio: ~7.5x)
n=10,000:    Method A ≈ 50,000,000  Method B ≈ 132,877   (Ratio: ~376x)
n=1,000,000: Method A ≈ 5×10¹¹     Method B ≈ 19,931,568 (Ratio: ~25,000x)

-> The gap grows dramatically as n increases
```

The tool used to describe this "behavior as n grows large" is **Asymptotic Notation**.

---

## 2. Mathematical Definitions of Asymptotic Notation

### 2.1 Big-O Notation (Upper Bound)

**Definition:**

```
f(n) = O(g(n))
⟺ ∃ c > 0, ∃ n₀ > 0 such that ∀ n ≥ n₀: f(n) ≤ c · g(n)
```

In plain terms: "There exist a constant c and a threshold n₀ such that for all sufficiently large n, f(n) is bounded above by c·g(n)."

**Why is the constant c needed?** Asymptotic notation ignores constant factors and focuses only on the order of growth. To treat both 3n² and 100n² as the same O(n²), we need the flexibility to adjust via the constant c.

**Why is n₀ needed?** Even if f(n) exceeds c·g(n) for small n, only the asymptotic (large-scale) behavior matters. The upper bound need only hold from n₀ onward.

```
  f(n)
  ▲
  │            ╱ c·g(n)
  │           ╱
  │      ×  ╱     <- Before n₀, f(n) may exceed c·g(n)
  │     ╱×╱
  │    ╱╱  <- From n₀ onward, f(n) ≤ c·g(n) always holds
  │  ╱╱
  │ ╱╱  f(n)
  │╱╱
  ┼──────────────────────► n
       n₀

  Legend: × marks points where f(n) exceeds c·g(n)
          From n₀ onward, f(n) ≤ c·g(n) always holds
```

**Proof example: 3n² + 5n + 2 = O(n²)**

Let c = 4, n₀ = 6. For n ≥ 6:
- 3n² + 5n + 2 ≤ 3n² + 5n² + 2n² = 10n² (since n ≤ n² and 1 ≤ n² when n ≥ 1)
- More precisely: when n ≥ 6, 5n ≤ n² (since n ≥ 5)
- Therefore 3n² + 5n + 2 ≤ 3n² + n² + n² = 5n² ≤ 5 · n² (when n ≥ 6, 2 ≤ n²)

A simpler approach: when n ≥ 1, 5n ≤ 5n² and 2 ≤ 2n², so
3n² + 5n + 2 ≤ 3n² + 5n² + 2n² = 10n². Thus c = 10, n₀ = 1 also works.

**Fully working Python code to verify the definition:**

```python
#!/usr/bin/env python3
"""
Program to numerically verify the definition of Big-O notation.
Confirms that f(n) = 3n² + 5n + 2 is O(n²).
"""


def f(n: int) -> int:
    """The function under analysis: f(n) = 3n² + 5n + 2"""
    return 3 * n * n + 5 * n + 2


def g(n: int) -> int:
    """The bounding function: g(n) = n²"""
    return n * n


def verify_big_o(c: float, n0: int, upper: int = 1000) -> bool:
    """
    Verify that f(n) ≤ c * g(n) holds for all n ≥ n₀.

    Why numerical verification is useful:
    - It builds intuition for forming hypotheses before a proof
    - It confirms whether the chosen c, n₀ are reasonable
    - It provides concrete understanding of the definition for educational purposes

    Note: This is NOT a substitute for a mathematical proof,
    since it can only verify a finite range.
    """
    for n in range(n0, upper + 1):
        if f(n) > c * g(n):
            print(f"  Counterexample found: n={n}, f(n)={f(n)}, c*g(n)={c * g(n)}")
            return False
    return True


def find_minimum_c(n0: int, precision: float = 0.01) -> float:
    """
    Find the minimum c that satisfies f(n) ≤ c * g(n) for a given n₀.

    Why find the minimum c:
    - The Big-O definition only requires "some c exists," but knowing
      the minimum c deepens intuition about the actual growth rate.
    """
    c = precision
    while c <= 1000:
        if verify_big_o(c, n0, upper=10000):
            return c
        c += precision
    return float('inf')


if __name__ == "__main__":
    print("=" * 60)
    print("Big-O Definition Verification: f(n) = 3n² + 5n + 2 = O(n²)")
    print("=" * 60)

    # Verification with c=10, n₀=1
    print("\n[Test 1] c=10, n₀=1:")
    result = verify_big_o(c=10, n0=1)
    print(f"  Result: {'Holds' if result else 'Does not hold'}")

    # Verification with c=4, n₀=1
    print("\n[Test 2] c=4, n₀=1:")
    result = verify_big_o(c=4, n0=1)
    print(f"  Result: {'Holds' if result else 'Does not hold'}")

    # Verification with c=3.5, n₀=1 (borderline case)
    print("\n[Test 3] c=3.5, n₀=1:")
    result = verify_big_o(c=3.5, n0=1)
    print(f"  Result: {'Holds' if result else 'Does not hold'}")

    # Verification with c=3.5, n₀=10
    print("\n[Test 4] c=3.5, n₀=10:")
    result = verify_big_o(c=3.5, n0=10)
    print(f"  Result: {'Holds' if result else 'Does not hold'}")

    # Search for minimum c
    print("\n[Search] Minimum c for n₀=1:")
    min_c = find_minimum_c(n0=1, precision=0.01)
    print(f"  Minimum c ≈ {min_c}")

    # Observe convergence of f(n)/g(n)
    print("\n[Convergence] Progression of f(n)/g(n):")
    print(f"  {'n':>8} | {'f(n)':>15} | {'g(n)':>15} | {'f(n)/g(n)':>10}")
    print(f"  {'-'*8}-+-{'-'*15}-+-{'-'*15}-+-{'-'*10}")
    for n in [1, 5, 10, 50, 100, 500, 1000, 10000]:
        fn = f(n)
        gn = g(n)
        ratio = fn / gn if gn > 0 else float('inf')
        print(f"  {n:>8} | {fn:>15,} | {gn:>15,} | {ratio:>10.4f}")

    print("\n-> As n grows, f(n)/g(n) converges to 3.0.")
    print("   This is consistent with the dominant term of f(n) being 3n².")
```

### 2.2 Big-Omega Notation (Lower Bound)

**Definition:**

```
f(n) = Ω(g(n))
⟺ ∃ c > 0, ∃ n₀ > 0 such that ∀ n ≥ n₀: f(n) ≥ c · g(n)
```

While Big-O indicates "at worst, this is how bad it gets," Big-Omega indicates "at minimum, it takes at least this much."

**Why lower bounds matter:** If an algorithm can be proven to be Omega(n log n), it means that no matter how clever the optimization, it cannot be faster than n log n. The fact that comparison-based sorting is Omega(n log n) imposes a fundamental constraint on sorting algorithm design.

**Proof example: 3n² + 5n + 2 = Omega(n²)**

Let c = 3, n₀ = 1. For n ≥ 1:
- 3n² + 5n + 2 ≥ 3n² (since 5n + 2 ≥ 0)
- 3n² = 3 · n²

Thus c = 3, n₀ = 1 satisfies the definition.

### 2.3 Big-Theta Notation (Tight Bound)

**Definition:**

```
f(n) = Θ(g(n))
⟺ f(n) = O(g(n)) and f(n) = Ω(g(n))

Equivalent definition:
⟺ ∃ c₁ > 0, c₂ > 0, n₀ > 0 such that
   ∀ n ≥ n₀: c₁ · g(n) ≤ f(n) ≤ c₂ · g(n)
```

When the upper and lower bounds are of the same order, this provides the most precise characterization.

```
  f(n)
  ▲
  │         ╱ c₂·g(n)    <- Upper bound
  │        ╱
  │      ╱╱ f(n)         <- f(n) is sandwiched between the two lines
  │    ╱╱╱
  │  ╱╱╱   c₁·g(n)      <- Lower bound
  │╱╱╱
  ┼──────────────────────► n
       n₀

  From n₀ onward, f(n) always lies between c₁·g(n) and c₂·g(n).
  This is the meaning of a "tight bound."
```

**Proof example: 3n² + 5n + 2 = Θ(n²)**

From sections 2.1 and 2.2:
- O(n²): Holds with c₂ = 10, n₀ = 1
- Ω(n²): Holds with c₁ = 3, n₀ = 1

Therefore Θ(n²) holds with c₁ = 3, c₂ = 10, n₀ = 1.

### 2.4 Little-o and Little-omega Notation

**Little-o notation (strict upper bound):**

```
f(n) = o(g(n))
⟺ ∀ c > 0, ∃ n₀ > 0 such that ∀ n ≥ n₀: f(n) < c · g(n)

Equivalent condition: lim(n→∞) f(n)/g(n) = 0
```

Difference from Big-O: Big-O requires "some c exists" (∃c), whereas Little-o requires "for all c" (∀c). This means f(n) is asymptotically negligible compared to g(n).

Example: n = o(n²) holds, but n² = o(n²) does not.

**Little-omega notation (strict lower bound):**

```
f(n) = ω(g(n))
⟺ ∀ c > 0, ∃ n₀ > 0 such that ∀ n ≥ n₀: f(n) > c · g(n)

Equivalent condition: lim(n→∞) f(n)/g(n) = ∞
```

Example: n² = ω(n) holds, but n² = ω(n²) does not.

**Intuitive understanding through analogy with inequalities:**

```
Analogy between asymptotic notation and inequalities:

  f(n) = O(g(n))    <->  a ≤ b      (at most)
  f(n) = Ω(g(n))    <->  a ≥ b      (at least)
  f(n) = Θ(g(n))    <->  a = b      (equal, in the order sense)
  f(n) = o(g(n))    <->  a < b      (strictly less than)
  f(n) = ω(g(n))    <->  a > b      (strictly greater than)

  Note: This analogy is for intuitive understanding only;
    strictly speaking, it means "within a constant factor."
```

### 2.5 Important Properties of Asymptotic Notation

**Transitivity:**
- f(n) = O(g(n)) and g(n) = O(h(n)) ⟹ f(n) = O(h(n))
- The same holds for Ω, Θ, o, ω

**Reflexivity:**
- f(n) = O(f(n))
- f(n) = Ω(f(n))
- f(n) = Θ(f(n))
- However, f(n) = o(f(n)) does NOT hold (a function is not strictly smaller than itself)

**Symmetry:**
- f(n) = Θ(g(n)) ⟺ g(n) = Θ(f(n))

**Transpose Symmetry:**
- f(n) = O(g(n)) ⟺ g(n) = Ω(f(n))
- f(n) = o(g(n)) ⟺ g(n) = ω(f(n))

**Sum Rule:**
- f(n) = O(h(n)) and g(n) = O(h(n)) ⟹ f(n) + g(n) = O(h(n))
- More generally: O(f(n)) + O(g(n)) = O(max(f(n), g(n)))

**Product Rule:**
- O(f(n)) · O(g(n)) = O(f(n) · g(n))

---

## 3. Proofs of Complexity Using Induction

Mathematical induction is a powerful tool for rigorously proving algorithm complexity. It is particularly effective for recursive algorithms.

### 3.1 Basic Structure of Induction

```
Steps for induction-based complexity proofs:

Step 1: State the proposition P(n) clearly
        Example: "T(n) ≤ c · n log n holds for all n ≥ n₀"

Step 2: Base Case
        Directly verify that P(n) holds for small n

Step 3: Inductive Step
        Assume P(k) holds for all k < n (inductive hypothesis),
        and show that P(n) holds

Step 4: Conclusion
        By induction, P(n) holds for all n ≥ n₀
```

### 3.2 Example: Proving Merge Sort is O(n log n)

Merge sort recurrence:
```
T(n) = 2T(n/2) + cn    (n > 1)
T(1) = c               (constant)
```

**Why this recurrence:**
- Split the array in half and recursively sort each half -> 2T(n/2)
- Merge the two sorted arrays -> cn (examining each element once)

**Proof: Show T(n) ≤ cn log₂ n (assuming n is a power of 2)**

**Base case:** For n = 2:
- T(2) = 2T(1) + 2c = 2c + 2c = 4c
- cn log₂ n = 2c · 1 = 2c
- T(2) = 4c > 2c, so this form does not hold

Correction: Show T(n) ≤ cn log₂ n + cn instead.

**Base case:** For n = 1:
- T(1) = c
- cn log₂ n + cn = 0 + c = c ✓

**Inductive step:** Assume the bound holds for all sizes up to n/2.

```
T(n) = 2T(n/2) + cn
     ≤ 2[c(n/2)log₂(n/2) + c(n/2)] + cn    (by inductive hypothesis)
     = cn[log₂(n/2)] + cn + cn
     = cn[log₂ n - 1] + 2cn
     = cn·log₂ n - cn + 2cn
     = cn·log₂ n + cn ✓
```

Therefore T(n) = O(n log n) is proven.

**Fully working Python code for verification:**

```python
#!/usr/bin/env python3
"""
Verify the O(n log n) complexity of merge sort against
the induction result. Counts actual operations and compares
with theoretical values.
"""

import math


class MergeSortCounter:
    """
    A class that accurately counts comparison operations in merge sort.

    Why use a class:
    - Mutable state is needed to update counters inside recursive functions
    - Encapsulation avoids global variables
    """

    def __init__(self):
        self.comparisons = 0  # Number of comparisons
        self.assignments = 0  # Number of assignments

    def reset(self):
        """Reset counters"""
        self.comparisons = 0
        self.assignments = 0

    def merge_sort(self, arr: list) -> list:
        """
        Execute merge sort and count comparisons/assignments.

        Returns: Sorted array
        """
        if len(arr) <= 1:
            return arr

        mid = len(arr) // 2
        left = self.merge_sort(arr[:mid])
        right = self.merge_sort(arr[mid:])
        return self._merge(left, right)

    def _merge(self, left: list, right: list) -> list:
        """Merge two sorted arrays"""
        result = []
        i = j = 0

        while i < len(left) and j < len(right):
            self.comparisons += 1
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
            self.assignments += 1

        while i < len(left):
            result.append(left[i])
            i += 1
            self.assignments += 1

        while j < len(right):
            result.append(right[j])
            j += 1
            self.assignments += 1

        return result


def theoretical_upper_bound(n: int) -> float:
    """
    Compute the theoretical upper bound c * n * log₂(n).

    Why log₂: Merge sort splits in half each time, so the recursion
    depth is log₂(n). Each level performs O(n) work, yielding a
    total of n * log₂(n).
    """
    if n <= 1:
        return 0
    return n * math.log2(n)


if __name__ == "__main__":
    import random

    counter = MergeSortCounter()

    print("=" * 70)
    print("Merge Sort Complexity Verification: Comparisons vs. Theoretical n·log₂(n)")
    print("=" * 70)
    print(f"{'n':>8} | {'Comparisons':>10} | {'n·log₂n':>12} | {'Ratio':>8} | {'Status':>6}")
    print(f"{'-'*8}-+-{'-'*10}-+-{'-'*12}-+-{'-'*8}-+-{'-'*6}")

    for exp in range(1, 16):
        n = 2 ** exp
        arr = list(range(n))
        random.seed(42)  # Fixed seed for reproducibility
        random.shuffle(arr)

        counter.reset()
        sorted_arr = counter.merge_sort(arr[:])

        # Verify sort correctness
        assert sorted_arr == sorted(arr), "Sort result is incorrect"

        theory = theoretical_upper_bound(n)
        ratio = counter.comparisons / theory if theory > 0 else 0
        within = "OK" if ratio <= 1.5 else "WARN"

        print(f"{n:>8,} | {counter.comparisons:>10,} | {theory:>12,.1f} | {ratio:>8.4f} | {within:>6}")

    print()
    print("-> The ratio converging to a constant confirms that comparisons = Θ(n log n).")
    print("   The ratio not going below 1.0 is due to the constant term in the merge step.")
```

### 3.3 Induction Pitfall: The Importance of the Base Case

A common mistake when proving complexity by induction is neglecting to verify the base case.

```
[Example of an Incorrect Proof]

Claim: T(n) = O(n) for T(n) = 2T(⌊n/2⌋) + n

Inductive step:
  T(n) = 2T(⌊n/2⌋) + n
       ≤ 2 · c · ⌊n/2⌋ + n     (inductive hypothesis)
       ≤ 2 · c · (n/2) + n
       = cn + n
       = (c+1)n                  <- This does not stay within cn!

This "proof" breaks down at the inductive step.
Trying to show T(n) ≤ cn fails because cn + n ≤ cn cannot hold.
Thus T(n) = O(n) does not hold (in fact T(n) = Θ(n log n)).
```

### 3.4 The Substitution Method

The substitution method is a general technique for proving recursive complexity.

```
Three steps of the substitution method:

1. Guess: Guess the form of the answer
   Example: Guess T(n) = O(n log n)

2. Substitute: Plug the guess into the recurrence and prove by induction
   Use the inductive hypothesis T(k) ≤ ck log k (k < n) to show T(n) ≤ cn log n

3. Determine constants: Find constants that satisfy the entire proof including the base case
```

**Tips for guessing:**
- Draw the recursion tree and estimate costs at each level
- Use known solutions to similar recurrences as reference
- Get an initial estimate using the Master Theorem (detailed in the next chapter)

### 3.5 Example: Complexity Proof for Recursive Binary Search

```python
#!/usr/bin/env python3
"""
Prove the O(log n) complexity of binary search by induction
and verify numerically.

Recurrence: T(n) = T(n/2) + c  (n > 1)
            T(1) = c

Claim: T(n) ≤ c · (log₂ n + 1) = O(log n)
"""

import math
import random
import time


def binary_search_recursive(arr: list, target: int,
                            lo: int, hi: int,
                            depth: int = 0) -> tuple:
    """
    Recursive binary search. Returns the search result and recursion depth.

    Why return depth:
    - To compare the theoretical O(log n) with actual recursion depth
    - To numerically verify the induction proof

    Returns: (found index or -1, recursion depth)
    """
    if lo > hi:
        return (-1, depth)

    mid = (lo + hi) // 2

    if arr[mid] == target:
        return (mid, depth + 1)
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, hi, depth + 1)
    else:
        return binary_search_recursive(arr, target, lo, mid - 1, depth + 1)


if __name__ == "__main__":
    print("=" * 60)
    print("Binary Search Complexity Verification: Recursion Depth vs. log₂(n)")
    print("=" * 60)
    print(f"{'n':>10} | {'Max Depth':>8} | {'log₂(n)+1':>10} | {'Status':>6}")
    print(f"{'-'*10}-+-{'-'*8}-+-{'-'*10}-+-{'-'*6}")

    for exp in range(1, 21):
        n = 2 ** exp
        arr = list(range(n))

        # Search for several elements and record maximum depth
        max_depth = 0
        for target in [0, n // 4, n // 2, 3 * n // 4, n - 1, n]:
            _, depth = binary_search_recursive(arr, target, 0, n - 1)
            max_depth = max(max_depth, depth)

        theory = math.log2(n) + 1
        ok = "OK" if max_depth <= theory + 1 else "WARN"

        print(f"{n:>10,} | {max_depth:>8} | {theory:>10.1f} | {ok:>6}")

    print()
    print("-> The maximum depth always stays within log₂(n) + 1.")
    print("   This is consistent with T(n) = O(log n).")
```

### 3.6 Proving Lower Bounds Using Induction

Induction can be used to prove lower bounds as well, not just upper bounds.

**Example: Outline of the Ω(n log n) lower bound proof for comparison-based sorting**

A comparison-based sorting algorithm needs at least log₂(n!) comparisons to determine the permutation of n elements.

```
Why log₂(n!):

There are n! possible permutations of n elements.
Each comparison has two outcomes (≤ or >), so k comparisons
can distinguish at most 2^k permutations.
To distinguish all permutations:
  2^k ≥ n!
  k ≥ log₂(n!)

By Stirling's approximation n! ≈ (n/e)^n:
  log₂(n!) ≈ n log₂(n/e) = n log₂ n - n log₂ e ≈ n log₂ n - 1.443n

Therefore k = Ω(n log n)

  Decision tree concept:

                    a₁ ≤ a₂?
                   /         \
              a₂ ≤ a₃?     a₁ ≤ a₃?
              /     \       /     \
          [1,2,3]  ...   ...    ...

  Since the number of leaves ≥ n!,
  the height of the tree ≥ log₂(n!) = Ω(n log n)
```

---

## 4. Major Complexity Classes in Detail

### 4.1 O(1) — Constant Time

Completes in a fixed number of operations regardless of input size.

**Typical operations:**
- Array index access: `arr[i]`
- Average-case hash table lookup/insertion
- Stack push/pop
- Variable assignment

```python
#!/usr/bin/env python3
"""
Examples of O(1) operations vs. operations that appear O(1) but are not.

Why this distinction matters:
- Even a single line of code may not be O(1)
- Understanding what happens internally is essential
"""


def constant_time_access(arr: list, index: int) -> int:
    """
    Array index access — O(1)

    Why O(1): Arrays are stored in contiguous memory, so the address
    is computed directly as base_address + index × element_size.
    This computation does not depend on input size.
    """
    return arr[index]


def hash_table_lookup(d: dict, key: str) -> object:
    """
    Hash table lookup — Average O(1), Worst O(n)

    Why average O(1): The hash function distributes keys uniformly,
    so with few collisions, a single hash computation and memory
    access suffices.

    Why worst O(n): If all keys collide to the same hash value,
    a linear search through the chain is required.
    """
    return d.get(key)


def looks_constant_but_not(s: str) -> str:
    """
    Appears O(1) but is actually O(n) — string concatenation

    Why O(n): Python strings are immutable, so s + "x" creates
    a new string of length len(s)+1. This creation requires
    copying O(len(s)) characters.
    """
    return s + "x"


if __name__ == "__main__":
    import time

    print("=" * 60)
    print("O(1) Operations vs. Hidden O(n) Operations")
    print("=" * 60)

    # Array access is O(1): constant time regardless of size
    print("\n[Array Index Access (O(1))]")
    for size in [100, 10_000, 1_000_000, 100_000_000]:
        arr = list(range(size))
        start = time.perf_counter_ns()
        for _ in range(10000):
            _ = arr[size // 2]
        elapsed = (time.perf_counter_ns() - start) / 10000
        print(f"  Size {size:>12,}: {elapsed:>8.1f} ns per access")

    # String concatenation is O(n): slows proportionally with size
    print("\n[String Concatenation (looks O(1) but is O(n))]")
    for size in [100, 1_000, 10_000, 100_000]:
        s = "a" * size
        start = time.perf_counter_ns()
        for _ in range(100):
            _ = s + "x"
        elapsed = (time.perf_counter_ns() - start) / 100
        print(f"  Length {size:>12,}: {elapsed:>10.1f} ns per operation")
```

### 4.2 O(log n) — Logarithmic Time

The problem size is halved (or reduced by a constant fraction) at each step.

**Typical operations:**
- Binary search
- Balanced BST lookup/insertion/deletion
- Euclidean algorithm

**Why log n:**
How many times can n be halved to reach 1? -> n / 2^k = 1 -> k = log₂ n

```
Binary search with n = 1024:

Step  0: Search range 1024 elements  [0 .......... 1023]
Step  1: Search range  512 elements  [0 ..... 511]
Step  2: Search range  256 elements  [0 .. 255]
Step  3: Search range  128 elements  [0 . 127]
Step  4: Search range   64 elements
Step  5: Search range   32 elements
Step  6: Search range   16 elements
Step  7: Search range    8 elements
Step  8: Search range    4 elements
Step  9: Search range    2 elements
Step 10: Search range    1 element   -> Found or absent

log₂(1024) = 10 steps. Even 1 million elements takes only ~20 steps.
```

### 4.3 O(n) — Linear Time

Processes every element of the input a constant number of times.

**Typical operations:**
- Linear search
- Finding the max/min of an array
- Counting sort
- Linked list traversal

### 4.4 O(n log n) — Linearithmic Time

The typical complexity of efficient sorting algorithms.

**Typical algorithms:**
- Merge sort
- Heap sort
- Quicksort (average)
- Fast Fourier Transform (FFT)

**Why n log n appears in many problems:**
- Divide-and-conquer splits the problem in half (log n levels), with O(n) work at each level
- The theoretical lower bound for comparison-based sorting is Ω(n log n)

### 4.5 O(n²) — Quadratic Time

Appears when every pair of elements needs to be examined.

**Typical algorithms:**
- Bubble sort, selection sort, insertion sort
- Substeps of naive matrix multiplication
- Naive closest pair problem

### 4.6 O(2ⁿ) and O(n!) — Exponential and Factorial Time

Appears in problems with combinatorial explosion.

**Typical problems:**
- O(2ⁿ): Enumerating all subsets, naive dynamic programming
- O(n!): Enumerating all permutations, Traveling Salesman Problem (naive approach)

### 4.7 Growth Rate Comparison Table

| Complexity | Name | n=10 | n=20 | n=50 | n=100 | n=1000 | Example |
|-----------|------|------|------|------|-------|--------|---------|
| O(1) | Constant | 1 | 1 | 1 | 1 | 1 | Array index access |
| O(log n) | Logarithmic | 3.3 | 4.3 | 5.6 | 6.6 | 10.0 | Binary search |
| O(√n) | Square root | 3.2 | 4.5 | 7.1 | 10.0 | 31.6 | Trial division for primality |
| O(n) | Linear | 10 | 20 | 50 | 100 | 1,000 | Linear search |
| O(n log n) | Linearithmic | 33 | 86 | 282 | 664 | 9,966 | Merge sort |
| O(n²) | Quadratic | 100 | 400 | 2,500 | 10,000 | 10⁶ | Bubble sort |
| O(n³) | Cubic | 1,000 | 8,000 | 125,000 | 10⁶ | 10⁹ | Naive matrix multiplication |
| O(2ⁿ) | Exponential | 1,024 | 10⁶ | 10¹⁵ | 10³⁰ | 10³⁰¹ | Subset enumeration |
| O(n!) | Factorial | 3.6×10⁶ | 2.4×10¹⁸ | 3×10⁶⁴ | 9×10¹⁵⁷ | - | Permutation enumeration |

**Estimated processing time (assuming 10⁸ operations per second):**

| Complexity | Time required for n = 10⁶ |
|-----------|--------------------------|
| O(n) | ~0.01 seconds |
| O(n log n) | ~0.2 seconds |
| O(n²) | ~2.8 hours |
| O(n³) | ~31.7 years |
| O(2ⁿ) | Far exceeds the age of the universe |

---

## 5. Space Complexity

### 5.1 What Is Space Complexity?

Space complexity expresses the amount of memory an algorithm requires as a function of the input size n.

**Why space complexity matters:**
- Memory is a finite resource and a constraint just like time
- Embedded systems and mobile devices have particularly limited memory
- When considering cache efficiency, algorithms that use less memory may actually be faster
- In distributed systems, it also affects the amount of data transferred between nodes

### 5.2 Total Space vs. Auxiliary Space

| Type | Definition | Example |
|------|-----------|---------|
| **Total space complexity** | Input data itself + additional memory used by the algorithm | Sort's input array + working array |
| **Auxiliary space complexity** | Only the additional memory used by the algorithm | Working array only (excluding input) |

In general, when people say "space complexity" they usually mean **auxiliary space complexity**. However, since this varies by context, it is preferable to be explicit about which one is meant.

```
Example: Space complexity of merge sort

Input array: [3, 1, 4, 1, 5, 9, 2, 6]   <- Size n (included in total space)

Working array during merge: [_, _, _, _, _, _, _, _]  <- Size n (auxiliary space)

Total space: O(n) + O(n) = O(n)
Auxiliary space: O(n)

Comparison: Heap sort operates with O(1) auxiliary space (in-place sort)
```

### 5.3 Stack Consumption in Recursion

Each recursive call consumes a stack frame. This consumption is easy to overlook but is an important component of space complexity.

```
Call stack for fib(5) (at the deepest point):

┌─────────────────────────┐
│ fib(1)  <- Currently executing │ Stack frame 5
├─────────────────────────┤
│ fib(2)  <- Called fib(1)       │ Stack frame 4
├─────────────────────────┤
│ fib(3)  <- Called fib(2)       │ Stack frame 3
├─────────────────────────┤
│ fib(4)  <- Called fib(3)       │ Stack frame 2
├─────────────────────────┤
│ fib(5)  <- Initial call        │ Stack frame 1
└─────────────────────────┘

Maximum stack depth = n = 5 -> Space complexity O(n)

Note: Naive recursive Fibonacci has time O(2ⁿ) but space O(n).
This is because at most n frames exist on the stack simultaneously.
The left child's computation completes and the stack unwinds before
the right child's computation begins.
```

**Fully working Python code to visualize stack depth:**

```python
#!/usr/bin/env python3
"""
Program to measure and visualize the stack depth of recursive calls.
Compares stack consumption of various recursive algorithms.
"""

import sys


class StackDepthTracker:
    """
    Tracks the stack depth of recursive calls.

    Why tracking is needed:
    - Python's default recursion limit is 1000
    - Writing recursive code without awareness of depth
      can cause RecursionError
    - Stack depth information is essential for space complexity analysis
    """

    def __init__(self):
        self.max_depth = 0
        self.current_depth = 0

    def enter(self):
        """Call when entering a recursive call"""
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)

    def leave(self):
        """Call when leaving a recursive call"""
        self.current_depth -= 1

    def reset(self):
        """Reset counters"""
        self.max_depth = 0
        self.current_depth = 0


def fib_naive(n: int, tracker: StackDepthTracker) -> int:
    """
    Naive recursive Fibonacci.
    Time: O(2ⁿ), Space: O(n) (stack depth)
    """
    tracker.enter()
    try:
        if n <= 1:
            return n
        result = fib_naive(n - 1, tracker) + fib_naive(n - 2, tracker)
        return result
    finally:
        tracker.leave()


def factorial_recursive(n: int, tracker: StackDepthTracker) -> int:
    """
    Recursive factorial.
    Time: O(n), Space: O(n) (stack depth)
    """
    tracker.enter()
    try:
        if n <= 1:
            return 1
        return n * factorial_recursive(n - 1, tracker)
    finally:
        tracker.leave()


def binary_search_rec(arr: list, target: int,
                      lo: int, hi: int,
                      tracker: StackDepthTracker) -> int:
    """
    Recursive binary search.
    Time: O(log n), Space: O(log n) (stack depth)
    """
    tracker.enter()
    try:
        if lo > hi:
            return -1
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            return binary_search_rec(arr, target, mid + 1, hi, tracker)
        else:
            return binary_search_rec(arr, target, lo, mid - 1, tracker)
    finally:
        tracker.leave()


if __name__ == "__main__":
    tracker = StackDepthTracker()

    print("=" * 60)
    print("Stack Depth Comparison of Recursive Algorithms")
    print("=" * 60)

    # Fibonacci (naive recursion) — Space O(n)
    print("\n[Fibonacci (naive recursion)] Space O(n)")
    for n in [5, 10, 15, 20]:
        tracker.reset()
        fib_naive(n, tracker)
        print(f"  fib({n:>2}): Max stack depth = {tracker.max_depth}")

    # Factorial (recursion) — Space O(n)
    print("\n[Factorial (recursion)] Space O(n)")
    for n in [5, 10, 50, 100]:
        tracker.reset()
        factorial_recursive(n, tracker)
        print(f"  {n:>3}!: Max stack depth = {tracker.max_depth}")

    # Binary search (recursion) — Space O(log n)
    print("\n[Binary Search (recursion)] Space O(log n)")
    for exp in [4, 8, 12, 16, 20]:
        n = 2 ** exp
        arr = list(range(n))
        tracker.reset()
        binary_search_rec(arr, n - 1, 0, n - 1, tracker)
        print(f"  n={n:>8,}: Max stack depth = {tracker.max_depth:>3}"
              f"  (log₂ n = {exp})")

    print()
    print("-> The recursive version of binary search uses O(log n) space.")
    print("   Converting to an iterative version (while loop) reduces space to O(1).")
    print("   When space efficiency matters, the iterative version should be preferred.")
```

### 5.4 Tail Recursion and Space Optimization

```
Tail Recursion:
When the recursive call is the last operation of the function,
the compiler/interpreter may be able to reuse the stack frame.

Normal recursion (factorial):
  def factorial(n):
      if n <= 1: return 1
      return n * factorial(n-1)  <- Multiplication after recursion -> NOT tail recursive

Tail recursive version:
  def factorial_tail(n, acc=1):
      if n <= 1: return acc
      return factorial_tail(n-1, n*acc)  <- Recursive call is last -> tail recursive

Note: Python does NOT perform tail call optimization (by design decision).
Languages like Scheme, Haskell, and Scala guarantee tail call optimization.
In Python, when deep recursion is needed, converting to iteration (loops)
is the standard approach.
```

### 5.5 Time-Space Tradeoffs

In many cases, time complexity and space complexity are in a tradeoff relationship.

| Approach | Time | Space | Example |
|---------|------|-------|---------|
| Recompute every time | Long | Small | Recompute Fibonacci each time |
| Cache results | Short | Large | Memoize Fibonacci results |
| Lookup table | Shortest | Largest | Pre-compute and store all results |

```python
#!/usr/bin/env python3
"""
Fibonacci computation: Demonstrating the time-space tradeoff.

Compares three implementations and numerically confirms the tradeoff.
"""

import time
from functools import lru_cache


def fib_naive(n: int) -> int:
    """
    Naive recursion: Time O(2ⁿ), Space O(n)

    Why it is slow:
    The same subproblems are computed repeatedly. To compute fib(30),
    fib(1) is called 832,040 times.
    """
    if n <= 1:
        return n
    return fib_naive(n - 1) + fib_naive(n - 2)


def fib_memoized(n: int, memo: dict = None) -> int:
    """
    Memoized recursion: Time O(n), Space O(n)

    Why it is fast:
    Each subproblem is computed only once and the result is stored
    in a dictionary. Subsequent calls use dictionary lookup (O(1)).
    """
    if memo is None:
        memo = {}
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib_memoized(n - 1, memo) + fib_memoized(n - 2, memo)
    return memo[n]


def fib_iterative(n: int) -> int:
    """
    Iterative (bottom-up): Time O(n), Space O(1)

    Why Space O(1):
    Only the two most recent values need to be retained.
    There is no need to store the entire array.
    """
    if n <= 1:
        return n
    prev2, prev1 = 0, 1
    for _ in range(2, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current
    return prev1


if __name__ == "__main__":
    print("=" * 65)
    print("Fibonacci Computation: Time-Space Tradeoff")
    print("=" * 65)

    # Correctness check
    for n in range(20):
        assert fib_iterative(n) == fib_memoized(n)

    # Performance comparison (including naive version for small n)
    print(f"\n{'Method':>14} | {'n':>6} | {'Result':>15} | {'Time':>12}")
    print(f"{'-'*14}-+-{'-'*6}-+-{'-'*15}-+-{'-'*12}")

    for n in [10, 20, 30, 35]:
        # Naive recursion
        start = time.perf_counter()
        result = fib_naive(n)
        elapsed_naive = time.perf_counter() - start

        # Memoized recursion
        start = time.perf_counter()
        result_memo = fib_memoized(n)
        elapsed_memo = time.perf_counter() - start

        # Iterative
        start = time.perf_counter()
        result_iter = fib_iterative(n)
        elapsed_iter = time.perf_counter() - start

        print(f"{'Naive':>14} | {n:>6} | {result:>15,} | {elapsed_naive:>10.6f} s")
        print(f"{'Memoized':>14} | {n:>6} | {result_memo:>15,} | {elapsed_memo:>10.6f} s")
        print(f"{'Iterative':>14} | {n:>6} | {result_iter:>15,} | {elapsed_iter:>10.6f} s")
        print(f"{'-'*14}-+-{'-'*6}-+-{'-'*15}-+-{'-'*12}")

    # Large n (naive version is too slow, excluded)
    print(f"\nComparison for large n (naive recursion omitted):")
    print(f"{'Method':>14} | {'n':>6} | {'Time':>12}")
    print(f"{'-'*14}-+-{'-'*6}-+-{'-'*12}")

    for n in [100, 500, 1000, 5000]:
        start = time.perf_counter()
        fib_memoized(n, {})
        elapsed_memo = time.perf_counter() - start

        start = time.perf_counter()
        fib_iterative(n)
        elapsed_iter = time.perf_counter() - start

        print(f"{'Memoized':>14} | {n:>6} | {elapsed_memo:>10.6f} s")
        print(f"{'Iterative':>14} | {n:>6} | {elapsed_iter:>10.6f} s")
        print(f"{'-'*14}-+-{'-'*6}-+-{'-'*12}")

    print()
    print("-> Both memoized and iterative have the same time complexity O(n),")
    print("   but the iterative version uses O(1) space, giving it an advantage for large n.")
```

---

## 6. Amortized Complexity

### 6.1 What Is Amortized Complexity?

Amortized complexity is the **total cost** of a sequence of operations divided by the number of operations. Individual operations may have high cost, but the average cost per operation across the entire sequence is low.

**How is it different from "average complexity":**
- **Average complexity**: Expected value for random inputs. Assumes a probability distribution
- **Amortized complexity**: Cost per operation for the worst-case sequence. No probability involved

Amortized complexity provides a "guarantee." Regardless of the operation sequence, if the total cost of n operations is O(n·f(n)), then the amortized cost per operation is O(f(n)).

### 6.2 Append to Dynamic Array (Aggregate Method)

```
How a dynamic array (Python's list.append) works:

Initial capacity = 1. When full, double the capacity.

Op    Capacity  Size   Cost    Description
─────────────────────────────────────────────
 1     1         1      1     Normal append
 2     2         2      1+1   Expand (copy 1 element) + append
 3     4         3      2+1   Expand (copy 2 elements) + append
 4     4         4      1     Normal append
 5     8         5      4+1   Expand (copy 4 elements) + append
 6     8         6      1     Normal append
 7     8         7      1     Normal append
 8     8         8      1     Normal append
 9    16         9      8+1   Expand (copy 8 elements) + append
─────────────────────────────────────────────

Total cost for n operations:
  Normal appends:   n operations × 1 = n
  Copies on expansion: 1 + 2 + 4 + 8 + ... + 2^⌊log₂n⌋ ≤ 2n

  Total ≤ n + 2n = 3n = O(n)

Amortized cost = O(n) / n = O(1)
```

### 6.3 Three Analysis Techniques

#### Technique 1: Aggregate Method

Compute the total cost of n operations and divide by n. The dynamic array example above uses this technique.

#### Technique 2: Accounting Method

Assign "credits" to each operation. Save extra credits on cheap operations and spend them on expensive ones.

```
Accounting method for dynamic arrays:

Assign 3 credits to each append operation:
  - 1: Cost of the append itself
  - 1: Savings for copying this element during future expansion
  - 1: Savings for copying one existing element during expansion

Op    Credits Paid  Actual Cost  Balance  Description
──────────────────────────────────────────
 1      3            1           2       Normal append, save 2
 2      3            2           3       Expand (cost 2), use savings
 3      3            3           3       Expand (cost 3), use savings
 4      3            1           5       Normal append, save 2
 5      3            5           3       Expand (cost 5), use savings
 ...
──────────────────────────────────────────

Balance is always non-negative -> Amortized cost per operation ≤ 3 = O(1)
```

#### Technique 3: Potential Method

Define a "potential energy" for the data structure and track changes with each operation.

```
Potential method for dynamic arrays:

Potential function: Φ(D) = 2 × size - capacity

Amortized cost of operation i:
  ĉᵢ = cᵢ + Φ(Dᵢ) - Φ(Dᵢ₋₁)

Without expansion (cᵢ = 1):
  ĉᵢ = 1 + (2(s+1) - cap) - (2s - cap)
     = 1 + 2 = 3

With expansion (cᵢ = s + 1, old capacity = s, new capacity = 2s):
  ĉᵢ = (s+1) + (2(s+1) - 2s) - (2s - s)
     = (s+1) + 2 - s = 3

In both cases, amortized cost = 3 = O(1) ✓
```

### 6.4 Verifying Python's Dynamic Array

```python
#!/usr/bin/env python3
"""
Observe the internal capacity changes of Python's list and verify amortized O(1).

Uses sys.getsizeof() to track the memory size of the list object.
"""

import sys


def observe_list_growth(max_elements: int = 100) -> list:
    """
    Add elements one by one to a list and record memory size changes.

    Why use sys.getsizeof:
    - It returns the size of the list object itself (pointer array capacity)
    - It does not include element sizes, but is sufficient for observing capacity changes
    """
    records = []
    lst = []
    prev_size = sys.getsizeof(lst)

    for i in range(max_elements):
        lst.append(i)
        curr_size = sys.getsizeof(lst)
        if curr_size != prev_size:
            records.append({
                'index': i,
                'elements': i + 1,
                'size_bytes': curr_size,
                'growth': curr_size - prev_size,
                'resized': True
            })
        else:
            records.append({
                'index': i,
                'elements': i + 1,
                'size_bytes': curr_size,
                'growth': 0,
                'resized': False
            })
        prev_size = curr_size

    return records


def calculate_amortized_cost(records: list) -> None:
    """
    Calculate amortized cost using the aggregate method.

    Each resize is counted as a "high-cost operation," and the
    average cost across all operations is computed.
    """
    total_cost = 0
    resize_count = 0

    for r in records:
        if r['resized']:
            # Resize cost = copying existing elements + append
            total_cost += r['elements']
            resize_count += 1
        else:
            # Normal append
            total_cost += 1

    n = len(records)
    amortized = total_cost / n if n > 0 else 0

    print(f"\n  Total of {n} operations:")
    print(f"    Total cost: {total_cost}")
    print(f"    Resize count: {resize_count}")
    print(f"    Amortized cost: {amortized:.2f} = O(1)")


if __name__ == "__main__":
    print("=" * 65)
    print("Behavior of Python list as a Dynamic Array")
    print("=" * 65)

    records = observe_list_growth(100)

    print("\nResize events:")
    print(f"  {'Elements':>6} | {'Size (bytes)':>12} | {'Growth (bytes)':>13}")
    print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*13}")

    for r in records:
        if r['resized']:
            print(f"  {r['elements']:>6} | {r['size_bytes']:>12} | +{r['growth']:>12}")

    calculate_amortized_cost(records)

    # Large-scale verification
    print("\n" + "=" * 65)
    print("Amortized Cost Verification at Scale")
    print("=" * 65)

    for n in [1_000, 10_000, 100_000, 1_000_000]:
        lst = []
        resize_count = 0
        prev_size = sys.getsizeof(lst)

        for i in range(n):
            lst.append(i)
            curr_size = sys.getsizeof(lst)
            if curr_size != prev_size:
                resize_count += 1
                prev_size = curr_size

        print(f"  n={n:>10,}: Resize count = {resize_count:>4}"
              f"  (log₂ n ≈ {n.bit_length():>3})")

    print()
    print("-> Resize count is O(log n), and the total cost of all operations is O(n).")
    print("   Therefore the amortized cost per operation is O(1).")
```

### 6.5 Other Data Structures Using Amortized Complexity

| Data Structure | Operation | Worst Case | Amortized |
|---------------|-----------|-----------|-----------|
| Dynamic array | append | O(n) | O(1) |
| Splay tree | search/insert/delete | O(n) | O(log n) |
| Fibonacci heap | decrease-key | O(n) | O(1) |
| Union-Find | union/find | O(log n) | O(α(n)) ≈ O(1) |
| Hash table | insert (with rehash) | O(n) | O(1) |

---

## 7. Recognizing Common Complexity Patterns

### 7.1 Pattern Recognition Framework

A systematic guide for determining complexity by examining code.

```
Complexity pattern recognition chart:

[No loops]──────────────────────── O(1)
     │
[Single loop, n iterations]─────── O(n)
     │
[Loop halves the size]──────────── O(log n)
     │
[Nested loops (independent)]────── O(n²)
     │
[Loop + sort inside]────────────── O(n² log n) or O(n log n)
     │
[Divide and conquer + linear merge] O(n log n)
     │
[Enumerate all subsets]─────────── O(2ⁿ)
     │
[Enumerate all permutations]────── O(n!)
```

### 7.2 Common Pitfall: Hidden Complexity

```python
#!/usr/bin/env python3
"""
Typical cases where the apparent complexity differs from the actual complexity.

Why this matters:
- To catch complexity issues in code reviews, you must understand
  the internal cost of each operation
- The assumption "one line = O(1)" is dangerous
"""


def hidden_quadratic_string(n: int) -> str:
    """
    [Pitfall 1] Repeated string concatenation — Appears O(n), actually O(n²)

    Why O(n²):
    Python strings are immutable. Each += creates a new string.
    The i-th concatenation copies a string of length i, so
    total = 1 + 2 + ... + n = n(n+1)/2 = O(n²)

    Correct approach: Append to a list and join at the end -> O(n)
    """
    result = ""
    for i in range(n):
        result += "a"  # O(i) copy each time
    return result


def correct_string_building(n: int) -> str:
    """O(n) string construction"""
    parts = []
    for i in range(n):
        parts.append("a")  # Amortized O(1)
    return "".join(parts)  # O(n) join


def hidden_quadratic_list(arr: list) -> list:
    """
    [Pitfall 2] Inserting at the front of a list — Appears O(n), actually O(n²)

    Why O(n²):
    list.insert(0, x) must shift the entire list back by one position.
    The i-th insertion shifts i elements, so the total is O(n²).

    Correct approach: Use collections.deque -> appendleft is O(1)
    """
    result = []
    for x in arr:
        result.insert(0, x)  # O(len(result)) shift each time
    return result


def hidden_quadratic_membership(arr: list, targets: list) -> list:
    """
    [Pitfall 3] Using `in` on a list — Appears O(n), actually O(n²)

    Why O(n²):
    The `in` operator on a list is O(n) linear search.
    For n targets, each requiring O(n), the total is O(n²).

    Correct approach: Convert to a set first -> O(n) total
    """
    found = []
    for target in targets:
        if target in arr:  # O(n) linear search
            found.append(target)
    return found


def correct_membership(arr: list, targets: list) -> list:
    """Membership check in O(n) using a set"""
    arr_set = set(arr)  # O(n) to build set
    found = []
    for target in targets:
        if target in arr_set:  # O(1) hash lookup
            found.append(target)
    return found


if __name__ == "__main__":
    import time

    print("=" * 65)
    print("Detecting and Fixing Hidden O(n²) Patterns")
    print("=" * 65)

    # Pattern 1: String concatenation
    print("\n[Pattern 1: String Concatenation]")
    for n in [1000, 5000, 10000, 50000]:
        start = time.perf_counter()
        hidden_quadratic_string(n)
        t_bad = time.perf_counter() - start

        start = time.perf_counter()
        correct_string_building(n)
        t_good = time.perf_counter() - start

        speedup = t_bad / t_good if t_good > 0 else float('inf')
        print(f"  n={n:>6}: += {t_bad:.4f}s | join {t_good:.6f}s"
              f" | Speedup: {speedup:.0f}x")

    # Pattern 3: Membership test
    print("\n[Pattern 3: Membership Test]")
    for n in [1000, 5000, 10000, 50000]:
        arr = list(range(n))
        targets = list(range(0, n, 2))  # Even numbers only

        start = time.perf_counter()
        hidden_quadratic_membership(arr, targets)
        t_bad = time.perf_counter() - start

        start = time.perf_counter()
        correct_membership(arr, targets)
        t_good = time.perf_counter() - start

        speedup = t_bad / t_good if t_good > 0 else float('inf')
        print(f"  n={n:>6}: list {t_bad:.4f}s | set {t_good:.6f}s"
              f" | Speedup: {speedup:.0f}x")

    print()
    print("-> The speedup ratio increases as n grows.")
    print("   This is the O(n²) vs O(n) gap becoming apparent.")
```

### 7.3 Loop Variable Change Patterns and Complexity

| Loop Pattern | Code Example | Complexity | Reason |
|-------------|-------------|-----------|--------|
| Linear increment | `for i in range(n)` | O(n) | i increments by 1 |
| Constant step | `for i in range(0, n, k)` | O(n/k) = O(n) | k is a constant |
| Doubling | `while i < n: i *= 2` | O(log n) | i doubles each time |
| Square root decrease | `while n > 1: n = sqrt(n)` | O(log log n) | Double logarithm |
| Independent nested loops | `for i in range(n): for j in range(n)` | O(n²) | n × n |
| Dependent nested loops | `for i in range(n): for j in range(i)` | O(n²) | Σi = n(n-1)/2 |
| Triple nested loops | `for i,j,k in range(n)³` | O(n³) | n × n × n |
| Outer loop × inner log | `for i: while j<n: j*=2` | O(n log n) | n × log n |

---

## 8. Anti-Patterns

### Anti-Pattern 1: Including Constant Factors in Asymptotic Notation

```python
# BAD: "There are 2 loops so it's O(2n)" -> Incorrect
def double_pass(arr: list) -> int:
    """
    A function that performs two linear passes.

    Incorrect analysis: O(2n)
    Correct analysis: O(n)

    Why constant factors are ignored:
    Asymptotic notation expresses the "growth rate" as input size
    increases. Both 2n and 100n grow linearly with n, so they are
    the same O(n). Constant factors depend on hardware and language
    implementation, which is separate from the algorithm's essential
    efficiency.
    """
    total = 0
    for x in arr:     # Pass 1: O(n)
        total += x
    for x in arr:     # Pass 2: O(n)
        total += x * 2
    return total
    # O(n) + O(n) = O(n) <- Constant factors are absorbed


# Similar mistakes:
# × O(n/2) -> ○ O(n)   (Half the loop is still linear)
# × O(3n²) -> ○ O(n²)  (Constant factors are ignored)
# × O(n² + n) -> ○ O(n²) (Lower-order terms are also ignored)
```

### Anti-Pattern 2: Stating Complexity Based Only on the Best Case

```python
# BAD: "Quicksort is O(n log n)" -> Incomplete
def quicksort(arr: list) -> list:
    """
    Stating quicksort's complexity accurately:

    - Best case:    O(n log n)  <- Pivot is always the median
    - Average case: O(n log n)  <- Random input
    - Worst case:   O(n²)      <- Already sorted input + first-element pivot

    Why distinguishing cases matters:
    Saying only "O(n log n)" hides the worst case O(n²).
    In a security context, an attacker can intentionally trigger the
    worst case (e.g., hash table collision attacks).

    Correct description:
    "Quicksort has average complexity O(n log n) and worst-case O(n²).
     Randomized pivot selection makes the probability of the worst case
     negligibly small."
    """
    if len(arr) <= 1:
        return arr
    pivot = arr[0]  # First element as pivot (cause of worst case)
    left = [x for x in arr[1:] if x <= pivot]
    right = [x for x in arr[1:] if x > pivot]
    return quicksort(left) + [pivot] + quicksort(right)


# GOOD: Avoid worst case with randomized pivot
import random

def quicksort_randomized(arr: list) -> list:
    """
    Randomized quicksort.
    Average/expected complexity: O(n log n)
    Worst-case complexity: O(n²) (but extremely unlikely)
    """
    if len(arr) <= 1:
        return arr
    pivot = random.choice(arr)  # Choose pivot randomly
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort_randomized(left) + middle + quicksort_randomized(right)
```

### Anti-Pattern 3: Completely Ignoring Space Complexity

```python
# BAD: Considering only time complexity and ignoring space
def get_all_pairs(arr: list) -> list:
    """
    Generates all pairs.
    Time: O(n²)
    Space: O(n²) <- Ignoring this is dangerous

    Why dangerous:
    For n = 100,000, the number of pairs is 10^10.
    At 8 bytes per pair, ~80GB of memory is needed.
    This exceeds the memory of most machines.

    Countermeasures:
    1. Use a generator so all pairs are not held in memory simultaneously
    2. Change to an algorithm that generates only the needed pairs
    """
    return [(a, b) for a in arr for b in arr]  # O(n²) space


# GOOD: Generator for O(1) space
def get_all_pairs_generator(arr: list):
    """Generate pairs one at a time — Time O(n²), Space O(1)"""
    for a in arr:
        for b in arr:
            yield (a, b)
```

### Anti-Pattern 4: Confusing Big-O Equals Sign with Mathematical Equality

```
× Incorrect: "O(n) = O(n²), therefore n = n²"

The equals sign in Big-O is NOT mathematical equality.
It should properly be read as "∈ (belongs to)."

f(n) = O(g(n)) should be read as "f(n) ∈ O(g(n))."
That is, f(n) belongs to the set O(g(n)).

O(n) ⊂ O(n²) is correct. Since n is of order at most n².
But writing "O(n) = O(n²)" is imprecise because the reverse does not hold.

Correctly:
  O(n) ⊂ O(n²)    n ∈ O(n²) is true
  O(n²) ⊄ O(n)    n² ∈ O(n) is false
```

---

## 9. Edge Case Analysis

### Edge Case 1: Input Sizes 0 and 1

```python
#!/usr/bin/env python3
"""
Verify behavior at edge cases for complexity.

Why edge cases matter:
- Asymptotic notation describes behavior for "sufficiently large n,"
  but real programs must handle n=0 and n=1 as well
- Forgetting constant-time handling at edge cases can lead to
  division by zero or index errors
"""


def binary_search(arr: list, target: int) -> int:
    """
    Binary search: O(log n), but what happens at n=0 or n=1?

    n = 0: Does not enter the loop, immediately returns -1 -> O(1)
    n = 1: Completes with one comparison -> O(1)

    Asymptotically, these are included in O(log n) (since O(1) ⊂ O(log n)),
    but in implementation, it is safe to handle these cases explicitly.
    """
    if not arr:  # Handle n = 0
        return -1
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


def merge_sort(arr: list) -> list:
    """
    Merge sort: O(n log n), but does not recurse for n ≤ 1.

    Many implementations switch to insertion sort for small n.
    This is because insertion sort has a smaller constant factor
    and is faster for small n. A typical threshold is n = 16-64.
    """
    if len(arr) <= 1:
        return arr[:]
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    # Merge step
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


if __name__ == "__main__":
    # Edge case testing
    print("=" * 50)
    print("Edge Case Verification")
    print("=" * 50)

    # n = 0
    assert binary_search([], 5) == -1
    assert merge_sort([]) == []
    print("n=0: All tests passed")

    # n = 1
    assert binary_search([5], 5) == 0
    assert binary_search([5], 3) == -1
    assert merge_sort([42]) == [42]
    print("n=1: All tests passed")

    # n = 2
    assert binary_search([1, 3], 3) == 1
    assert merge_sort([3, 1]) == [1, 3]
    print("n=2: All tests passed")

    print("\n-> Confirmed correct behavior at edge cases.")
```

### Edge Case 2: Integer Overflow and Complexity

```
Problem: Overflow in (lo + hi) // 2

When lo = 2,000,000,000 and hi = 2,000,000,000:
  lo + hi = 4,000,000,000  <- Exceeds the 32-bit integer limit (2^31 - 1 ≈ 2.1×10⁹)

This is not a problem in Python (integers have no upper limit),
but it is a serious bug in C, Java, and C++.

Fix:
  mid = lo + (hi - lo) // 2

Why this is safe:
  hi - lo is always non-negative and at most hi.
  lo + (a non-negative value) is at most lo + hi.
  Therefore overflow does not occur.

Impact on complexity:
  The complexity itself remains O(log n), but overflow can
  cause infinite loops.
  "Correct complexity analysis" presumes "correct implementation."
```

### Edge Case 3: Worst Case of Hash Tables

```
Typically: Hash table lookup is said to be O(1)

But in the worst case:
  - All keys collide to the same bucket -> O(n) linear search
  - Rehashing occurs -> O(n) copy
  - Hash function computation itself can be O(k) where k is key length
    (e.g., for string keys)

Potential impact:
  - Inserting n keys: Usually O(n), worst case O(n²)
  - Attacker sends keys that intentionally cause collisions -> DoS attack

Countermeasures:
  1. Introduce randomness via universal hash functions
  2. Python 3.3+ has hash randomization enabled by default
  3. Switch to a balanced tree when collision count exceeds a threshold
     (Java 8+ HashMap)
```

---

## 10. Exercises

### 10.1 Basic Level

**Problem B1: Determining Complexity**

Determine the time complexity of each function below using O notation.

```python
def func_a(n: int) -> int:
    total = 0
    for i in range(n):
        for j in range(10):
            total += i * j
    return total


def func_b(n: int) -> int:
    total = 0
    i = 1
    while i < n:
        total += i
        i *= 2
    return total


def func_c(arr: list) -> list:
    result = []
    for x in arr:
        if x not in result:
            result.append(x)
    return result
```

<details>
<summary>Solution B1</summary>

- `func_a`: O(n). The inner loop runs a constant number of times (10), so O(10n) = O(n).
- `func_b`: O(log n). i doubles as 1, 2, 4, 8, ..., so the loop runs log₂ n times.
- `func_c`: O(n²). `x not in result` is O(n) linear search in the worst case, repeated n times, yielding O(n²). Using a set for deduplication gives O(n).

</details>

**Problem B2: Applying the Big-O Definition**

For f(n) = 5n³ + 3n² + 7, prove the following:
1. f(n) = O(n³)
2. f(n) = Ω(n³)
3. f(n) = Θ(n³)

<details>
<summary>Solution B2</summary>

1. **Proof of O(n³):** For n ≥ 1, 5n³ + 3n² + 7 ≤ 5n³ + 3n³ + 7n³ = 15n³. Thus with c=15, n₀=1, f(n) ≤ 15n³ holds.

2. **Proof of Ω(n³):** For all n ≥ 1, 5n³ + 3n² + 7 ≥ 5n³. Thus with c=5, n₀=1, f(n) ≥ 5n³ holds.

3. **Proof of Θ(n³):** From 1. and 2., with c₁=5, c₂=15, n₀=1, we have 5n³ ≤ f(n) ≤ 15n³. Therefore f(n) = Θ(n³).

</details>

**Problem B3: Comparing Complexities**

For each pair of functions below, determine whether f(n) = O(g(n)), f(n) = Ω(g(n)), or f(n) = Θ(g(n)) holds.

1. f(n) = n², g(n) = n³
2. f(n) = 2ⁿ, g(n) = 3ⁿ
3. f(n) = log₂ n, g(n) = log₁₀ n
4. f(n) = n log n, g(n) = n√n

<details>
<summary>Solution B3</summary>

1. f(n) = O(g(n)). n² ≤ n³ (n ≥ 1). However, not Θ (since n² = o(n³)).
2. f(n) = O(g(n)). 2ⁿ ≤ 3ⁿ. Also 2ⁿ = o(3ⁿ), so not Θ.
3. f(n) = Θ(g(n)). log₂ n = log₁₀ n / log₁₀ 2 = (1/log₁₀ 2) · log₁₀ n. Base conversion is a constant factor, so they are the same order.
4. f(n) = O(g(n)). n log n = o(n^1.5), so not Θ (since log n = o(√n)).

</details>

### 10.2 Intermediate Level

**Problem A1: Complexity of a Recursive Function**

Determine the time and space complexity of the following recursive function.

```python
def mystery(n: int) -> int:
    if n <= 0:
        return 0
    return mystery(n - 1) + mystery(n - 1)
```

<details>
<summary>Solution A1</summary>

**Time complexity: O(2ⁿ)**

Recurrence: T(n) = 2T(n-1) + O(1)
Expanding: T(n) = 2T(n-1) = 2·2T(n-2) = 4T(n-2) = ... = 2ⁿ·T(0) = O(2ⁿ)

**Space complexity: O(n)**

Maximum stack depth is n. The left `mystery(n-1)` completes and the stack unwinds before the right `mystery(n-1)` executes. At most n+1 frames exist on the stack simultaneously.

Note: This function is equivalent to `2 * mystery(n-1)`, but the two recursive calls make the time exponential.

</details>

**Problem A2: Amortized Complexity Analysis**

Determine the amortized complexity of the `push` and `multipop(k)` operations for the following `MultiStack` class.

```python
class MultiStack:
    def __init__(self):
        self.stack = []

    def push(self, x):
        """Add one element — Cost 1"""
        self.stack.append(x)

    def multipop(self, k):
        """Pop up to k elements — Cost min(k, len(stack))"""
        count = min(k, len(self.stack))
        for _ in range(count):
            self.stack.pop()
        return count
```

<details>
<summary>Solution A2</summary>

**Analysis using the accounting method:**

Assign 2 credits to each push operation:
- 1: Cost of push itself
- 1: Savings for a future pop

multipop(k) pops at most k elements, but each pop's cost was prepaid by the corresponding push.

Total cost of n operations (mix of push and multipop):
- Each element is pushed at most once and popped at most once
- If the total number of pushes is m, then total pops is also at most m
- Total cost ≤ 2m ≤ 2n

**Conclusion:**
- Amortized cost of push: O(1)
- Amortized cost of multipop(k): O(1) (regardless of k!)

Intuitive explanation: To pop k elements via multipop, k pushes must have occurred beforehand. The cost is "prepaid" by the pushes.

</details>

**Problem A3: Discovering Hidden Complexity**

Determine the complexity of the following code. Note that the apparent and actual complexities differ.

```python
def process(data: list) -> list:
    result = []
    for item in data:                  # n iterations
        result = result + [item]       # <- Pay attention to this line
    return result
```

<details>
<summary>Solution A3</summary>

**Time complexity: O(n²)**

`result = result + [item]` is different from `result.append(item)`.
- `result + [item]` creates a new list and copies all elements of result
- At the i-th iteration, a list of length i is copied, costing O(i)
- Total: 1 + 2 + ... + n = n(n+1)/2 = O(n²)

Using `result.append(item)` gives amortized O(1) × n = O(n).

**Lesson:** Python's `+` for list concatenation creates a new list. For list building inside a loop, use `append`.

</details>

### 10.3 Advanced Level

**Problem E1: Proof Using the Substitution Method**

For the recurrence T(n) = 4T(n/2) + n, prove T(n) = O(n²) using the substitution method.

<details>
<summary>Solution E1</summary>

**Claim:** T(n) ≤ cn² (for some constant c > 0)

**Inductive hypothesis:** T(k) ≤ ck² (for all k < n)

**Inductive step:**
```
T(n) = 4T(n/2) + n
     ≤ 4c(n/2)² + n          (inductive hypothesis)
     = 4c · n²/4 + n
     = cn² + n
```

Here we need cn² + n ≤ cn², which is impossible since n > 0.

**Correction:** Guess T(n) ≤ cn² - dn (subtract a lower-order term).

```
T(n) = 4T(n/2) + n
     ≤ 4[c(n/2)² - d(n/2)] + n
     = 4c·n²/4 - 4d·n/2 + n
     = cn² - 2dn + n
     = cn² - dn - (dn - n)
     = cn² - dn - n(d - 1)
     ≤ cn² - dn                (when d ≥ 1)
```

Thus the inductive step holds when d ≥ 1. The base case also holds with an appropriate c.

**Conclusion:** T(n) = O(n²)

(Verification via Master Theorem: a=4, b=2, f(n)=n. n^{log_b a} = n^{log₂ 4} = n². f(n) = n = O(n^{2-ε}) (ε=1), so Case 1: T(n) = Θ(n²))

</details>

**Problem E2: Lower Bound Proof**

Prove that a comparison-based search algorithm on a sorted array requires at worst Ω(log n) comparisons.

<details>
<summary>Solution E2</summary>

**Proof (decision tree argument):**

A comparison-based search algorithm performs one comparison (≤ or >) at each step and branches based on the result. This can be modeled as a binary decision tree.

1. The search target is one of n elements or "not present" — n+1 possible outcomes.
2. A binary decision tree of height h has at most 2^h leaves.
3. To distinguish all outcomes: 2^h ≥ n + 1
4. Therefore h ≥ log₂(n + 1) = Ω(log n)

**Conclusion:** Search on a sorted array requires Ω(log n) comparisons in the worst case. Since binary search runs in O(log n), it is an optimal algorithm.

</details>

**Problem E3: Practical Complexity Improvement**

Improve the following O(n³) code to O(n²).

```python
def count_triples(arr: list, target: int) -> int:
    """Count triplets in arr whose sum equals target"""
    n = len(arr)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if arr[i] + arr[j] + arr[k] == target:
                    count += 1
    return count
```

<details>
<summary>Solution E3</summary>

```python
def count_triples_optimized(arr: list, target: int) -> int:
    """
    Improved to O(n²) using Sort + Two Pointers.

    Why O(n²):
    - Outer loop: O(n)
    - Inner Two Pointers: O(n) (approaching from both ends is linear)
    - Total: O(n²)
    - Sorting: O(n log n) < O(n²), so the overall complexity is O(n²)
    """
    arr_sorted = sorted(arr)  # O(n log n)
    n = len(arr_sorted)
    count = 0

    for i in range(n - 2):                    # O(n)
        left = i + 1
        right = n - 1
        while left < right:                   # Amortized O(n)
            s = arr_sorted[i] + arr_sorted[left] + arr_sorted[right]
            if s == target:
                count += 1
                left += 1
                right -= 1
            elif s < target:
                left += 1
            else:
                right -= 1

    return count
```

Note: This solution simplifies the handling of duplicate elements. A complete implementation may need logic to skip elements with the same value.

</details>

---

## 11. Summary Comparison Tables

### Table 1: Complete Comparison of Asymptotic Notations

| Notation | Meaning | Mathematical Definition | Intuitive Explanation | Inequality Analogy |
|---------|---------|------------------------|----------------------|-------------------|
| f = O(g) | Upper bound | ∃c>0, n₀: f(n) ≤ cg(n) | At worst, on the order of g | a ≤ b |
| f = Ω(g) | Lower bound | ∃c>0, n₀: f(n) ≥ cg(n) | At least on the order of g | a ≥ b |
| f = Θ(g) | Tight bound | c₁g(n) ≤ f(n) ≤ c₂g(n) | Exactly on the order of g | a = b |
| f = o(g) | Strict upper bound | ∀c>0, ∃n₀: f(n) < cg(n) | Strictly smaller than g | a < b |
| f = ω(g) | Strict lower bound | ∀c>0, ∃n₀: f(n) > cg(n) | Strictly larger than g | a > b |

### Table 2: Complexity of Data Structure Operations

| Data Structure | Access | Insert | Delete | Search | Space |
|---------------|--------|--------|--------|--------|-------|
| Array | O(1) | O(n) | O(n) | O(n) | O(n) |
| Dynamic array | O(1) | O(1)* | O(n) | O(n) | O(n) |
| Linked list | O(n) | O(1)** | O(1)** | O(n) | O(n) |
| Hash table | - | O(1)* | O(1)* | O(1)* | O(n) |
| BST (balanced) | - | O(log n) | O(log n) | O(log n) | O(n) |
| Heap | O(1)*** | O(log n) | O(log n) | O(n) | O(n) |

\* Amortized or average
\*\* When the position of insertion/deletion is known
\*\*\* O(1) for the minimum (or maximum) only

---

## 12. FAQ

### Q1: What is the difference between O(n) and Θ(n)?

**A:** O(n) means "of order at most n" and specifies only an upper bound. Θ(n) means "exactly of order n" and specifies both upper and lower bounds.

Concrete example for f(n) = 5n + 3:
- f(n) = O(n) ✓ (n is an upper bound for the order)
- f(n) = O(n²) ✓ (n² is also an upper bound, but a "loose" one)
- f(n) = Θ(n) ✓ (tight upper and lower bound)
- f(n) = Θ(n²) ✗ (the lower bound is n, so it is not of order n²)

**Practical guideline:** Use Θ when possible, as it conveys more information. However, use O when only the worst-case upper bound is known.

### Q2: Can complexity be O(n²) even without nested loops?

**A:** Yes. Here are three representative examples:

```python
# Example 1: Repeated string concatenation
s = ""
for i in range(n):
    s += str(i)  # Creates a new string each time -> O(1+2+...+n) = O(n²)

# Example 2: Inserting at the front of a list
result = []
for x in data:
    result.insert(0, x)  # Shifts all elements each time -> O(n²)

# Example 3: Recursive expansion
# T(n) = T(n-1) + n -> T(n) = n + (n-1) + ... + 1 = O(n²)
```

### Q3: How do amortized complexity and average complexity differ?

**A:**

| Aspect | Average Complexity | Amortized Complexity |
|--------|-------------------|---------------------|
| Subject | A single operation | A sequence of operations |
| Assumption | Assumes a probability distribution on inputs | No probability assumptions |
| Guarantee | Holds as an expected value | Holds for any sequence of operations |
| Example | Average O(n log n) for quicksort | Amortized O(1) for dynamic array append |

The amortized O(1) for dynamic array append is **guaranteed**: no matter the order of operations, the total cost of n operations is O(n). Quicksort's O(n log n), on the other hand, is an "expected value for random input" and may be O(n²) for specific inputs.

### Q4: Does Big-O represent the worst case?

**A:** No. Big-O is a mathematical notation for upper bounds and can be applied to any case — best, average, or worst.

```
Misconception: "Big-O = worst case"
Correct: Big-O is a notation for the upper bound of any function

Examples of correct usage:
  "Quicksort's worst-case complexity is O(n²)"
  "Quicksort's average-case complexity is O(n log n)"
  "Quicksort's best-case complexity is O(n log n)"

All use Big-O, but each refers to a different case.
Cases and notation are orthogonal concepts.
```

### Q5: Why do algorithms with the same O(n log n) have different performance?

**A:** Because Big-O notation hides constant factors and lower-order terms.

```
Example: Merge sort vs. Quicksort

Merge sort: T(n) ≈ 1.44 · n · log₂ n (expected number of comparisons)
Quicksort:  T(n) ≈ 1.39 · n · log₂ n (expected number of comparisons)

Both are O(n log n), but:
1. Constant factors differ (quicksort's is slightly smaller)
2. Cache efficiency differs (quicksort has better memory locality)
3. Additional memory differs (merge sort uses O(n), quicksort uses O(log n))

-> Even with the same Big-O, actual performance can differ significantly.
   Big-O is a tool for "comparing orders of growth" and is not well-suited
   for comparing within the same order.
```

### Q6: Does the base of the logarithm affect complexity?

**A:** Not in asymptotic notation, because a change of base is just a constant factor.

```
log_a(n) = log_b(n) / log_b(a)

Since log_b(a) is a constant:
  O(log_a n) = O(log_b n / log_b a) = O(log_b n)

Concrete example:
  log₂(1024) = 10
  log₁₀(1024) ≈ 3.01
  log₃(1024) ≈ 6.29

All are constant multiples of each other -> The same O(log n)

However, in actual algorithms:
  - Binary search -> halves at each step -> operations = log₂ n
  - Ternary search -> reduces by 1/3 at each step -> operations = log₃ n ≈ 0.63 · log₂ n
  Asymptotically the same, but constant factors differ.
```

---


## FAQ

### Q1: What is the most important point when studying this topic?

Building practical experience is the most important thing. Understanding deepens not just through theory, but by actually writing and running code to observe behavior.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals to jump to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving on.

### Q3: How is this knowledge applied in practice?

This knowledge is frequently applied in day-to-day development work. It is particularly important during code reviews and architecture design.

---

## 13. Summary

| Topic | Key Point |
|-------|----------|
| O notation | Indicates an upper bound. "At worst, on the order of this." Constants c and threshold n₀ exist |
| Ω notation | Indicates a lower bound. "Takes at least this much." Used for lower bound proofs |
| Θ notation | Tight bound. Precise order when upper bound = lower bound |
| o / ω notation | Strict upper/lower bounds. Indicates "strictly smaller/larger" |
| Induction | Essential for rigorous proofs of recursive complexity. Do not forget the base case |
| Space complexity | Consider auxiliary memory + recursion stack. Tradeoff with time |
| Amortized complexity | Total cost of consecutive operations / number of operations. Guarantee independent of probability |
| Constant factors | Ignored in asymptotic notation. Not suitable for comparisons within the same order |
| Specify the case | Best/average/worst should be distinguished and stated |
| Hidden complexity | Watch out for string concatenation, list concatenation, and `in` operations |

---

## Suggested Next Readings

- [Complexity Analysis — Recurrences and the Master Theorem](./01-complexity-analysis.md)
- [Space-Time Tradeoffs — Memoization and Bloom Filters](./02-space-time-tradeoff.md)

---

## References

1. Cormen, T.H., Leiserson, C.E., Rivest, R.L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. — Chapter 3 "Characterizing Running Times" covers the rigorous definitions and properties of O/Ω/Θ. Chapter 4 "Divide-and-Conquer" details the Master Theorem and substitution method. Chapter 16 "Amortized Analysis" systematically covers the aggregate, accounting, and potential methods.
2. Knuth, D.E. (1997). *The Art of Computer Programming, Volume 1: Fundamental Algorithms* (3rd ed.). Addison-Wesley. — The original source of asymptotic notation. Knuth introduced Big-O notation to computer science and also advocated for the need for Ω and Θ notations.
3. Skiena, S.S. (2020). *The Algorithm Design Manual* (3rd ed.). Springer. — Chapter 2 "Algorithm Analysis" covers practical analysis techniques for complexity. The pattern recognition for reading complexity from code is particularly useful.
4. Sedgewick, R. & Wayne, K. (2011). *Algorithms* (4th ed.). Addison-Wesley. — Excels in practical explanation of complexity, with Java code examples and abundant illustrations.
5. Sipser, M. (2012). *Introduction to the Theory of Computation* (3rd ed.). Cengage Learning. — Provides the theoretical foundation for complexity classes (P, NP, etc.). Recommended for deeper understanding of computational complexity theory.
6. MIT OpenCourseWare. *6.006 Introduction to Algorithms*. Massachusetts Institute of Technology. https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/ — MIT's lecture materials on complexity analysis. Freely available.
7. Roughgarden, T. (2017). *Algorithms Illuminated* (Part 1). Soundlikeyourself Publishing. — Ideal for an introduction to asymptotic notation and algorithm design. Strikes a good balance between intuitive explanation and rigorous discussion.

