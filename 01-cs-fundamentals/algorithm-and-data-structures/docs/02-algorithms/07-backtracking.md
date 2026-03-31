# Backtracking

> Deeply understand the technique of systematically exploring solution candidates and pruning branches that violate constraints early -- through diverse examples including N-Queens, Sudoku, permutation generation, graph coloring, and Knight's tour

## Learning Objectives

1. **Understand the principles of backtracking** (state space tree exploration and pruning) and implement them using a generic template
2. **Efficiently solve N-Queens and Sudoku** with multiple pruning strategies
3. **Systematically implement enumeration of permutations, combinations, and subsets**, including handling of duplicate elements
4. **Apply backtracking to problems** such as graph coloring, Knight's tour, and parenthesis generation
5. **Design pruning strategies** and analyze computational complexity to plan optimization approaches


## Prerequisites

Before reading this guide, familiarity with the following will deepen your understanding:

- Basic programming knowledge
- Understanding of related fundamental concepts
- Familiarity with the content in [Divide and Conquer](./06-divide-conquer.md)

---

## Table of Contents

1. [Principles of Backtracking](#1-principles-of-backtracking)
2. [N-Queens Problem](#2-n-queens-problem)
3. [Sudoku Solver](#3-sudoku-solver)
4. [Permutations, Combinations, and Subsets](#4-permutations-combinations-and-subsets)
5. [Applied Problem: Graph Coloring](#5-applied-problem-graph-coloring)
6. [Applied Problems: Knight's Tour and Parenthesis Generation](#6-applied-problems-knights-tour-and-parenthesis-generation)
7. [State Space Tree Structure and Pruning Strategies](#7-state-space-tree-structure-and-pruning-strategies)
8. [Complexity Analysis and Optimization Techniques](#8-complexity-analysis-and-optimization-techniques)
9. [Anti-patterns](#9-anti-patterns)
10. [Exercises](#10-exercises)
11. [FAQ](#11-faq)
12. [Summary](#12-summary)
13. [References](#13-references)

---

## 1. Principles of Backtracking

### 1.1 Overview and Intuitive Understanding

Backtracking is a systematic approach to solving search problems. It is easiest to understand by imagining how you solve a maze: when you reach a fork, choose one path and proceed; if you hit a dead end, return to the last fork and try a different path. This cycle of "advance -> get stuck -> retreat -> try another path" is the essence of backtracking.

```
Backtracking = Depth-First Search (DFS) + Pruning

Understanding backtracking through a maze example:

  S -> -> v         S -> -> v         S -> -> v
           v              v              v
       v <- <          x <- <          v <- <
       v                               v
       x (dead end)                    -> -> G (goal!)

  Step 1: Advance    Step 2: Retreat   Step 3: Try another path
```

The core of the algorithm consists of three steps:

1. **Choose**: Select one from the available choices
2. **Explore**: Continue searching recursively under that choice
3. **Unchoose**: Undo the choice after exploration is complete

### 1.2 State Space Tree

The exploration process of backtracking can be visualized as a **state space tree**. The root node represents the initial state, each edge represents a choice, and leaf nodes represent solutions or dead ends.

```
State space tree (choosing 2 from 3 elements):

                    root (initial state)
                  /   |   \
                 1    2    3         <- 1st choice
                / \   |
               2   3  3             <- 2nd choice
               |   |  |
             [1,2][1,3][2,3]        <- Solutions (leaf nodes)


Exhaustive search without pruning:

                    root
                  /   |   \
                 1    2    3
               / | \ / | \ / | \
              1 2 3 1 2 3 1 2 3      <- Exhaustive search including duplicates/invalids
              Nodes explored: 1 + 3 + 9 = 13


With pruning:

                    root
                  /   |   \
                 1    2    3
                / \   |    X         <- Pruned: "only numbers greater than self"
               2   3  3
               |   |  |
             [1,2][1,3][2,3]         <- Only valid solutions explored
              Nodes explored: 1 + 3 + 3 = 7 (~46% reduction)
```

The effect of pruning becomes dramatically greater as problem size increases. In the N-Queens problem, without pruning, O(N^N) nodes are explored, but appropriate pruning reduces this to approximately O(N!). For N=8, 16,777,216 nodes are reduced to 40,320 or fewer.

### 1.3 Basic Template

Backtracking has a generic template applicable to virtually all problems.

```python
def backtrack(state, choices, result):
    """
    Basic backtracking template

    Parameters:
        state   : Current partial solution (mutable object)
        choices : Available choices
        result  : List to store found solutions
    """
    # ---- Base condition: check if solution is complete ----
    if is_solution(state):
        result.append(state.copy())  # Save a copy of the solution
        return

    # ---- Try each choice ----
    for choice in choices:
        # Pruning: does this choice satisfy the constraints?
        if is_valid(state, choice):
            # 1. Apply the choice (Choose)
            apply(state, choice)

            # 2. Explore recursively (Explore)
            backtrack(state, next_choices(choice), result)

            # 3. Undo the choice (Unchoose / Backtrack)
            undo(state, choice)


def backtrack_single(state, choices):
    """
    Variation for finding only one solution

    Returns:
        bool: True if a solution is found
    """
    if is_solution(state):
        return True

    for choice in choices:
        if is_valid(state, choice):
            apply(state, choice)

            if backtrack_single(state, next_choices(choice)):
                return True  # Stop immediately upon finding a solution

            undo(state, choice)

    return False  # No solution in this branch
```

**Choosing the right template:**

| Objective | Template | Return value | Solution storage |
|:---|:---|:---|:---|
| Enumerate all solutions | `backtrack` | None (void) | Append to `result` list |
| Find one solution | `backtrack_single` | `bool` | Reference state directly |
| Count solutions | Counter variant | `int` | Global variable or return value |
| Find optimal solution | Optimization variant | None or value | Update best solution |

### 1.4 Characteristics of Problems Where Backtracking is Effective

Backtracking is particularly effective for problems with the following properties:

1. **Sequential selectivity**: The solution can be constructed as a sequence of choices
2. **Early constraint detection**: Constraint violations can be detected at the partial solution stage
3. **Search space reducibility**: Pruning can significantly reduce the number of nodes

```
Backtracking applicability flowchart:

  Receive problem
       |
       v
  Can the solution be
  constructed as a sequence ----No----> Consider other approaches
  of choices?
       |
      Yes
       |
       v
  Can constraint violations
  be detected at the         ----No----> Consider exhaustive search or DP
  partial solution stage?
       |
      Yes
       |
       v
  Can pruning significantly
  reduce the search space?   ----No----> Exhaustive search + pruning (limited effect)
       |
      Yes
       |
       v
  Backtracking is effective!
```

---

## 2. N-Queens Problem

### 2.1 Problem Definition

Place N queens on an N x N chessboard such that no two queens attack each other. A chess queen can attack any piece on the same row, column, or diagonal.

```
Two solutions for 4-Queens:

  Solution 1:           Solution 2:
  . Q . .               . . Q .
  . . . Q               Q . . .
  Q . . .               . . . Q
  . . Q .               . Q . .

  board = [1,3,0,2]     board = [2,0,3,1]


Queen's attack range (placed at center):

  \ . | . /           Three conditions for attack detection:
  . \ | / .
  ----Q----           1. Same column: col == prev_col
  . / | \ .           2. Top-left to bottom-right diagonal: row - col == prev_row - prev_col
  / . | . \           3. Top-right to bottom-left diagonal: row + col == prev_row + prev_col

                      => Conditions 2,3 are equivalent to |row-prev_row| == |col-prev_col|
```

### 2.2 Basic Implementation

We implement a row-by-row placement approach. Since each row is processed in order, the row constraint is automatically satisfied.

```python
def solve_nqueens(n: int) -> list:
    """
    Solve the N-Queens problem and return all solutions

    Parameters:
        n: Board size (number of queens)

    Returns:
        List of solutions. Each solution is in the format board[row] = col
    """
    solutions = []
    board = [-1] * n  # board[row] = col (column position of the queen in that row)

    def is_safe(row: int, col: int) -> bool:
        """Check if a queen can be placed at row, col"""
        for prev_row in range(row):
            prev_col = board[prev_row]
            # Same column?
            if prev_col == col:
                return False
            # Same diagonal? (equal distance = on diagonal)
            if abs(prev_row - row) == abs(prev_col - col):
                return False
        return True

    def backtrack(row: int):
        """Place a queen in the given row"""
        if row == n:
            solutions.append(board[:])
            return

        for col in range(n):
            if is_safe(row, col):
                board[row] = col
                backtrack(row + 1)
                board[row] = -1  # Backtrack

    backtrack(0)
    return solutions


# ---- Example ----
solutions = solve_nqueens(4)
print(f"Number of 4-Queens solutions: {len(solutions)}")
for i, sol in enumerate(solutions):
    print(f"  Solution {i+1}: {sol}")

# Output:
# Number of 4-Queens solutions: 2
#   Solution 1: [1, 3, 0, 2]
#   Solution 2: [2, 0, 3, 1]

solutions_8 = solve_nqueens(8)
print(f"Number of 8-Queens solutions: {len(solutions_8)}")
# Output: Number of 8-Queens solutions: 92
```

### 2.3 Optimization Using Sets

Instead of checking all previous rows in `is_safe` each time, we manage occupied columns and diagonals using sets for O(1) checking.

```python
def solve_nqueens_fast(n: int) -> list:
    """N-Queens - fast version using sets"""
    solutions = []
    board = [-1] * n
    cols = set()       # Occupied columns
    diag1 = set()      # Occupied top-left to bottom-right diagonals (row - col)
    diag2 = set()      # Occupied top-right to bottom-left diagonals (row + col)

    def backtrack(row: int):
        if row == n:
            solutions.append(board[:])
            return

        for col in range(n):
            if col in cols:
                continue
            d1 = row - col
            if d1 in diag1:
                continue
            d2 = row + col
            if d2 in diag2:
                continue

            # Apply choice
            board[row] = col
            cols.add(col)
            diag1.add(d1)
            diag2.add(d2)

            backtrack(row + 1)

            # Undo choice
            board[row] = -1
            cols.remove(col)
            diag1.remove(d1)
            diag2.remove(d2)

    backtrack(0)
    return solutions


# ---- Example ----
solutions = solve_nqueens_fast(8)
print(f"Number of 8-Queens solutions: {len(solutions)}")  # 92
```

### 2.4 Optimization Using Bitmasks

Using bit operations instead of sets achieves a constant-factor speedup. Each bit corresponds to a column on the board.

```python
def count_nqueens_bitmask(n: int) -> int:
    """
    Count N-Queens solutions quickly using bitmasks

    cols:  Bitmask of occupied columns
    diag1: Top-left to bottom-right diagonal (shifted left)
    diag2: Top-right to bottom-left diagonal (shifted right)

    Bit operation mechanics:
      available & (-available) -> extract the lowest set bit
      available ^= pos         -> clear that bit
    """
    count = 0
    all_cols = (1 << n) - 1  # Mask with all n bits set to 1

    def backtrack(row: int, cols: int, diag1: int, diag2: int):
        nonlocal count
        if row == n:
            count += 1
            return

        # Compute available columns
        available = all_cols & ~(cols | diag1 | diag2)

        while available:
            pos = available & (-available)  # Extract lowest set bit
            available ^= pos               # Clear that bit
            backtrack(
                row + 1,
                cols | pos,
                (diag1 | pos) << 1,  # Diagonal shifts by one column at the next row
                (diag2 | pos) >> 1
            )

    backtrack(0, 0, 0, 0)
    return count


# ---- Example ----
for n in range(1, 13):
    print(f"N={n:2d}: {count_nqueens_bitmask(n)} solutions")

# Output:
# N= 1: 1 solutions
# N= 2: 0 solutions
# N= 3: 0 solutions
# N= 4: 2 solutions
# N= 5: 10 solutions
# N= 6: 4 solutions
# N= 7: 40 solutions
# N= 8: 92 solutions
# N= 9: 352 solutions
# N=10: 724 solutions
# N=11: 2680 solutions
# N=12: 14200 solutions
```

```
Bitmask operation example (N=4, placing queen at col=1 in row=0):

  row=0:  cols=0010  diag1=0010  diag2=0010
                      v <<1       v >>1
  row=1:  cols=0010  diag1=0100  diag2=0001
          blocked = cols | diag1 | diag2 = 0111
          available = 1111 & ~0111 = 1000
          -> Only col=3 is available

  Board:
  row 0:  . Q . .    (0010 in cols)
  row 1:  . . . Q    (only col=3 available)
```

### 2.5 Solution Visualization

```python
def print_board(solution: list) -> None:
    """Display an N-Queens board as ASCII art"""
    n = len(solution)
    border = "+" + "---+" * n
    print(border)
    for row in range(n):
        line = "|"
        for col in range(n):
            if solution[row] == col:
                line += " Q |"
            else:
                line += "   |"
        print(line)
        print(border)
    print()


# ---- Example ----
solutions = solve_nqueens_fast(4)
for i, sol in enumerate(solutions):
    print(f"Solution {i + 1}:")
    print_board(sol)

# Output:
# Solution 1:
# +---+---+---+---+
# |   | Q |   |   |
# +---+---+---+---+
# |   |   |   | Q |
# +---+---+---+---+
# | Q |   |   |   |
# +---+---+---+---+
# |   |   | Q |   |
# +---+---+---+---+
```

---

## 3. Sudoku Solver

### 3.1 Problem Definition and Constraints

Fill a 9x9 Sudoku grid satisfying all row, column, and 3x3 box constraints. Each row, column, and 3x3 box must contain exactly one of each digit from 1 to 9.

```
Input:                         Output:
5 3 . | . 7 . | . . .        5 3 4 | 6 7 8 | 9 1 2
6 . . | 1 9 5 | . . .        6 7 2 | 1 9 5 | 3 4 8
. 9 8 | . . . | . 6 .        1 9 8 | 3 4 2 | 5 6 7
------+-------+------        ------+-------+------
8 . . | . 6 . | . . 3        8 5 9 | 7 6 1 | 4 2 3
4 . . | 8 . 3 | . . 1        4 2 6 | 8 5 3 | 7 9 1
7 . . | . 2 . | . . 6        7 1 3 | 9 2 4 | 8 5 6
------+-------+------        ------+-------+------
. 6 . | . . . | 2 8 .        9 6 1 | 5 3 7 | 2 8 4
. . . | 4 1 9 | . . 5        2 8 7 | 4 1 9 | 6 3 5
. . . | . 8 . | . 7 9        3 4 5 | 2 8 6 | 1 7 9

Triple constraint check:

  Row constraint      Column constraint   Box constraint
  -> -> -> -> ->       v               +-------+
  5 3 4 6 7 8 9 1 2   5              | 5 3 4 |
  (each row has 1-9     6              | 6 7 2 |
   exactly once)        1              | 1 9 8 |
                        ...            +-------+
                       (each col has   (each 3x3 box has
                        1-9 once)       1-9 once)
```

### 3.2 Basic Implementation

A straightforward approach that fills empty cells from top-left to bottom-right.

```python
def solve_sudoku(board: list) -> bool:
    """
    Sudoku solver - basic backtracking

    Parameters:
        board: 9x9 2D list. Empty cells are 0

    Returns:
        True if a solution is found (board is overwritten with the solution)
    """

    def find_empty() -> tuple:
        """Find the first empty cell from top-left"""
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0:
                    return r, c
        return None

    def is_valid(row: int, col: int, num: int) -> bool:
        """Check if num can be placed at (row, col)"""
        # Row check
        if num in board[row]:
            return False

        # Column check
        for r in range(9):
            if board[r][col] == num:
                return False

        # 3x3 box check
        box_r, box_c = 3 * (row // 3), 3 * (col // 3)
        for r in range(box_r, box_r + 3):
            for c in range(box_c, box_c + 3):
                if board[r][c] == num:
                    return False

        return True

    cell = find_empty()
    if cell is None:
        return True  # All cells filled -> solution found

    row, col = cell
    for num in range(1, 10):
        if is_valid(row, col, num):
            board[row][col] = num

            if solve_sudoku(board):
                return True

            board[row][col] = 0  # Backtrack

    return False  # No solution in this branch
```

### 3.3 Speedup Using Candidate Sets

By managing candidate digits with sets, the `is_valid` check becomes O(1).

```python
def solve_sudoku_fast(board: list) -> bool:
    """
    Sudoku solver - fast version using candidate sets

    Manages digits in use per row, column, and box using sets,
    enabling O(1) placement feasibility checking.
    """
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]
    empty_cells = []

    # Build initial state
    for r in range(9):
        for c in range(9):
            if board[r][c] != 0:
                num = board[r][c]
                rows[r].add(num)
                cols[c].add(num)
                boxes[3 * (r // 3) + c // 3].add(num)
            else:
                empty_cells.append((r, c))

    def backtrack(idx: int) -> bool:
        if idx == len(empty_cells):
            return True

        r, c = empty_cells[idx]
        box_id = 3 * (r // 3) + c // 3

        for num in range(1, 10):
            if (num not in rows[r] and
                num not in cols[c] and
                num not in boxes[box_id]):

                board[r][c] = num
                rows[r].add(num)
                cols[c].add(num)
                boxes[box_id].add(num)

                if backtrack(idx + 1):
                    return True

                board[r][c] = 0
                rows[r].discard(num)
                cols[c].discard(num)
                boxes[box_id].discard(num)

        return False

    return backtrack(0)


# ---- Example ----
board = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]

solve_sudoku_fast(board)
print("Sudoku solution:")
for row in board:
    print(row)

# Output:
# [5, 3, 4, 6, 7, 8, 9, 1, 2]
# [6, 7, 2, 1, 9, 5, 3, 4, 8]
# [1, 9, 8, 3, 4, 2, 5, 6, 7]
# [8, 5, 9, 7, 6, 1, 4, 2, 3]
# [4, 2, 6, 8, 5, 3, 7, 9, 1]
# [7, 1, 3, 9, 2, 4, 8, 5, 6]
# [9, 6, 1, 5, 3, 7, 2, 8, 4]
# [2, 8, 7, 4, 1, 9, 6, 3, 5]
# [3, 4, 5, 2, 8, 6, 1, 7, 9]
```

### 3.4 Further Speedup with MRV Heuristic

**MRV (Minimum Remaining Values)** is a heuristic that fills cells with the fewest candidates first. Cells with fewer choices reach dead ends sooner, avoiding wasted exploration.

```python
def solve_sudoku_mrv(board: list) -> bool:
    """
    Sudoku solver - MRV heuristic version

    By choosing the cell with the minimum number of candidates at each step,
    the search space is dramatically reduced.
    """
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]
    empty_cells = set()

    for r in range(9):
        for c in range(9):
            if board[r][c] != 0:
                num = board[r][c]
                rows[r].add(num)
                cols[c].add(num)
                boxes[3 * (r // 3) + c // 3].add(num)
            else:
                empty_cells.add((r, c))

    def get_candidates(r: int, c: int) -> set:
        """Set of digits that can be placed at (r, c)"""
        return set(range(1, 10)) - rows[r] - cols[c] - boxes[3 * (r // 3) + c // 3]

    def backtrack() -> bool:
        if not empty_cells:
            return True

        # MRV: select the cell with the fewest candidates
        min_cell = min(empty_cells, key=lambda rc: len(get_candidates(*rc)))
        r, c = min_cell
        candidates = get_candidates(r, c)

        if not candidates:
            return False  # No candidates -> backtrack

        empty_cells.remove((r, c))
        box_id = 3 * (r // 3) + c // 3

        for num in candidates:
            board[r][c] = num
            rows[r].add(num)
            cols[c].add(num)
            boxes[box_id].add(num)

            if backtrack():
                return True

            board[r][c] = 0
            rows[r].discard(num)
            cols[c].discard(num)
            boxes[box_id].discard(num)

        empty_cells.add((r, c))
        return False

    return backtrack()
```

### 3.5 Sudoku Difficulty and Search Node Count

```
Relationship between Sudoku difficulty and search:

  Difficulty   Empty cells   Search nodes (approx.)   Effect of candidate sets
  ---------------------------------------------------------------------------
  Easy         30-35         Less than 100            High (many uniquely determined)
  Medium       40-50         Less than 1,000          Medium
  Hard         50-55         Less than 10,000         Low (many branches)
  Expert       55-60         100,000 or more          MRV is critical

  * Theoretical max search space is 9^81 ~ 1.97 x 10^77,
    but proper pruning keeps it to hundreds to tens of thousands of nodes
```

---

## 4. Permutations, Combinations, and Subsets

### 4.1 Generating Permutations

Arrange n elements in every possible order.

```python
def permutations(nums: list) -> list:
    """
    Generate all permutations

    Complexity: O(n * n!)
    Space: O(n) (recursion stack) + O(n * n!) (result storage)
    """
    result = []

    def backtrack(path: list, remaining: list):
        if not remaining:
            result.append(path[:])
            return

        for i in range(len(remaining)):
            path.append(remaining[i])
            backtrack(path, remaining[:i] + remaining[i + 1:])
            path.pop()

    backtrack([], nums)
    return result


# ---- Example ----
print(permutations([1, 2, 3]))
# [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]
```

**Permutations with duplicate elements:**

```python
def permutations_unique(nums: list) -> list:
    """
    Generate permutations with duplicate elements (duplicates removed)

    Strategy: sort the array, then ensure that a duplicate value is not
    used before the previous identical element has been used
    """
    result = []
    nums.sort()
    used = [False] * len(nums)

    def backtrack(path: list):
        if len(path) == len(nums):
            result.append(path[:])
            return

        for i in range(len(nums)):
            # Already used
            if used[i]:
                continue
            # Skip duplicates: if the previous identical element is unused, skip
            if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:
                continue

            used[i] = True
            path.append(nums[i])
            backtrack(path)
            path.pop()
            used[i] = False

    backtrack([])
    return result


# ---- Example ----
print(permutations_unique([1, 1, 2]))
# [[1, 1, 2], [1, 2, 1], [2, 1, 1]]  -- no duplicates (3!/2! = 3 patterns)
```

### 4.2 Generating Combinations

Choose k items from n (order does not matter).

```python
def combinations(nums: list, k: int) -> list:
    """
    Generate all nCk combinations

    Complexity: O(nCk)
    Pruning: skip exploration when remaining elements are insufficient
    """
    result = []

    def backtrack(start: int, path: list):
        if len(path) == k:
            result.append(path[:])
            return

        # Pruning: skip if not enough elements remain
        remaining_needed = k - len(path)
        for i in range(start, len(nums) - remaining_needed + 1):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    backtrack(0, [])
    return result


# ---- Example ----
print(combinations([1, 2, 3, 4], 2))
# [[1,2], [1,3], [1,4], [2,3], [2,4], [3,4]]

print(f"C(4,2) = {len(combinations([1,2,3,4], 2))}")
# C(4,2) = 6
```

### 4.3 Enumerating Subsets

```python
def subsets(nums: list) -> list:
    """
    Generate all subsets (power set)

    Complexity: O(n * 2^n)
    Feature: every exploration point is recorded as a solution
    """
    result = []

    def backtrack(start: int, path: list):
        result.append(path[:])  # Every partial solution is an answer
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    backtrack(0, [])
    return result


# ---- Example ----
print(subsets([1, 2, 3]))
# [[], [1], [1,2], [1,2,3], [1,3], [2], [2,3], [3]]
```

### 4.4 Comparison of Enumeration Patterns

```
Differences in state space trees for permutations, combinations, and subsets:

Permutations {1,2,3}:        Solution condition: path length equals n
              {}
           /  |  \
          1   2   3           <- Choose from all elements
         / \ / \ / \
       12 13 21 23 31 32      <- All elements except used ones
       |  |  |  |  |  |
      123 132 213 231 312 321 <- Solutions (all leaf nodes)
      Number of solutions: 3! = 6


Combinations C(4,2):          Solution condition: path length equals k
              {}
          / | \  \
         1  2  3  4           <- Choose from start onward
        /|\ /\  |
      12 13 14 23 24 34       <- Solutions (when length k is reached)
      Number of solutions: C(4,2) = 6


Subsets {1,2,3}:              Solution condition: every node is a solution
              {}              <- Solution
           /  |  \
          1   2   3           <- Solutions x3
         / \  |
       12  13 23              <- Solutions x3
       |
      123                    <- Solution
      Number of solutions: 2^3 = 8
```

| Enumeration Pattern | Solution Condition | Choice Constraint | Complexity | Number of Solutions |
|:---|:---|:---|:---|:---|
| Permutations | path length = n | All unused elements | O(n * n!) | n! |
| Permutations with repetition | path length = n | All elements (repeats allowed) | O(n^n) | n^n |
| Combinations | path length = k | Elements from start onward | O(nCk) | nCk |
| Subsets | Always a solution | Elements from start onward | O(n * 2^n) | 2^n |
| Combinations with repetition | path length = k | Current element onward | O(n+k-1 C k) | (n+k-1)Ck |

---

## 5. Applied Problem: Graph Coloring

### 5.1 Problem Definition

The graph coloring problem assigns colors to each vertex of a graph such that no two adjacent vertices share the same color. Minimizing the number of colors is NP-hard, but determining whether a graph is colorable with a given number of colors m can be solved by backtracking.

```
Graph coloring example (is 3-coloring possible?):

  Input graph:              Coloring result:
      0 --- 1                R --- G
      |   / |                |   / |
      |  /  |                |  /  |
      | /   |                | /   |
      2 --- 3                B --- R

  Adjacency list:            Color assignment:
  0: [1, 2]                  0: Red (R)
  1: [0, 2, 3]               1: Green (G)
  2: [0, 1, 3]               2: Blue (B)
  3: [1, 2]                  3: Red (R)
```

### 5.2 Implementation

```python
def graph_coloring(adj: dict, m: int) -> list:
    """
    Solve the graph coloring problem using backtracking

    Parameters:
        adj: Adjacency list {vertex: [list of adjacent vertices]}
        m  : Number of available colors

    Returns:
        Color assignment dictionary if colorable, None otherwise
    """
    vertices = sorted(adj.keys())
    n = len(vertices)
    colors = [0] * n  # 0 = uncolored, 1..m = colors

    def is_safe(vertex_idx: int, color: int) -> bool:
        """Check if color can be assigned to vertex_idx"""
        vertex = vertices[vertex_idx]
        for neighbor in adj[vertex]:
            neighbor_idx = vertices.index(neighbor)
            if colors[neighbor_idx] == color:
                return False
        return True

    def backtrack(vertex_idx: int) -> bool:
        if vertex_idx == n:
            return True  # All vertices colored

        for color in range(1, m + 1):
            if is_safe(vertex_idx, color):
                colors[vertex_idx] = color

                if backtrack(vertex_idx + 1):
                    return True

                colors[vertex_idx] = 0  # Backtrack

        return False

    if backtrack(0):
        return {vertices[i]: colors[i] for i in range(n)}
    return None


# ---- Example ----
graph = {
    0: [1, 2],
    1: [0, 2, 3],
    2: [0, 1, 3],
    3: [1, 2],
}

# Try 2-coloring
result = graph_coloring(graph, 2)
print(f"2 colors: {result}")
# Output: 2 colors: None (impossible)

# Try 3-coloring
result = graph_coloring(graph, 3)
print(f"3 colors: {result}")
# Output: 3 colors: {0: 1, 1: 2, 2: 3, 3: 1}

color_names = {1: "Red", 2: "Green", 3: "Blue"}
if result:
    for vertex, color in result.items():
        print(f"  Vertex {vertex}: {color_names[color]}")
```

### 5.3 Graph Coloring Variations

| Variation | Description | Application |
|:---|:---|:---|
| Vertex coloring | Adjacent vertices must differ in color | Map coloring, timetable scheduling |
| Edge coloring | Edges sharing an endpoint must differ in color | Network frequency assignment |
| List coloring | Available colors vary per vertex | Register allocation |
| Chromatic number minimization | Find the minimum number of colors | Compiler optimization |

---

## 6. Applied Problems: Knight's Tour and Parenthesis Generation

### 6.1 Knight's Tour

Find a path for a chess knight to visit every square of the board exactly once.

```
Knight's movement pattern:          A Knight's tour solution on a 5x5 board (example):

    . 2 . 1 .                  1 14  9 20  3
    3 . . . 8                  24 19  2 15 10
    . . N . .                   13  8 23  4 21
    4 . . . 7                   18 25  6 11 16
    . 5 . 6 .                    7 12 17 22  5

  N can move to positions 1-8
  (L-shaped moves)
```

```python
def solve_knight_tour(n: int, start_row: int = 0, start_col: int = 0) -> list:
    """
    Solve the Knight's tour problem

    Parameters:
        n: Board size
        start_row, start_col: Starting position

    Returns:
        Board with visit order (2D list) if a solution exists, None otherwise
    """
    board = [[-1] * n for _ in range(n)]

    # 8 possible knight moves
    moves = [
        (-2, -1), (-2, 1), (-1, -2), (-1, 2),
        (1, -2), (1, 2), (2, -1), (2, 1)
    ]

    def count_onward_moves(r: int, c: int) -> int:
        """Count available moves from (r, c) (for Warnsdorff's Rule)"""
        count = 0
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and board[nr][nc] == -1:
                count += 1
        return count

    def backtrack(r: int, c: int, move_count: int) -> bool:
        board[r][c] = move_count

        if move_count == n * n - 1:
            return True  # All squares visited

        # Warnsdorff's Rule: prioritize moves with fewer onward options
        next_moves = []
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and board[nr][nc] == -1:
                next_moves.append((count_onward_moves(nr, nc), nr, nc))

        next_moves.sort()  # Sort by fewest onward moves

        for _, nr, nc in next_moves:
            if backtrack(nr, nc, move_count + 1):
                return True

        board[r][c] = -1  # Backtrack
        return False

    if backtrack(start_row, start_col, 0):
        return board
    return None


# ---- Example ----
n = 6
result = solve_knight_tour(n)
if result:
    print(f"{n}x{n} Knight's tour solution:")
    for row in result:
        print(" ".join(f"{x:3d}" for x in row))

# Example output (6x6):
#   0 15 22  3 14 25
#  23  4  1 26  9 20
#  16 31 24 21  2 13
#   5 28 33 10 19  8
#  32 17 30 35 12 27
#  29  6 11 34  7 18
```

### 6.2 Parenthesis Generation

Generate all patterns of n pairs of properly matched parentheses.

```
All patterns for n=3:

  ((()))    (()())    (())()    ()(())    ()()()

  Part of the state space tree (with pruning):

                    ""
                  /    \
                "("    X (")" cannot come first)
               /    \
            "(("    "()"
           /    \       \
        "((("  "(()""  "()("
         |      / \      |
       ...    ...  ...  ...
```

```python
def generate_parentheses(n: int) -> list:
    """
    Generate all valid parentheses patterns with n pairs

    Parameters:
        n: Number of pairs

    Returns:
        List of valid parenthesis strings

    Pruning conditions:
      - Open parentheses: up to n can be used
      - Close parentheses: must not exceed open parentheses count
    """
    result = []

    def backtrack(s: str, open_count: int, close_count: int):
        if len(s) == 2 * n:
            result.append(s)
            return

        # Can still add an open parenthesis
        if open_count < n:
            backtrack(s + "(", open_count + 1, close_count)

        # Can add a close parenthesis (if fewer than open)
        if close_count < open_count:
            backtrack(s + ")", open_count, close_count + 1)

    backtrack("", 0, 0)
    return result


# ---- Example ----
for n in range(1, 5):
    parens = generate_parentheses(n)
    print(f"n={n}: {len(parens)} patterns")
    if n <= 3:
        for p in parens:
            print(f"  {p}")

# Output:
# n=1: 1 patterns
#   ()
# n=2: 2 patterns
#   (())
#   ()()
# n=3: 5 patterns
#   ((()))
#   (()())
#   (())()
#   ()(())
#   ()()()
# n=4: 14 patterns
```

The number of parenthesis patterns is given by the Catalan number C_n = (2n)! / ((n+1)! * n!).

### 6.3 Combination Sum

Find all combinations that sum exactly to a target value.

```python
def combination_sum(candidates: list, target: int) -> list:
    """
    Find all combinations from candidates (with reuse) that sum to target

    Parameters:
        candidates: List of positive integers (no duplicates)
        target: Target sum

    Returns:
        List of combinations that sum to target
    """
    result = []
    candidates.sort()

    def backtrack(start: int, path: list, remaining: int):
        if remaining == 0:
            result.append(path[:])
            return

        for i in range(start, len(candidates)):
            # Pruning: if candidate exceeds remaining, all subsequent will too
            if candidates[i] > remaining:
                break

            path.append(candidates[i])
            # Same element can be reused, so start = i
            backtrack(i, path, remaining - candidates[i])
            path.pop()

    backtrack(0, [], target)
    return result


# ---- Example ----
print(combination_sum([2, 3, 6, 7], 7))
# [[2, 2, 3], [7]]

print(combination_sum([2, 3, 5], 8))
# [[2, 2, 2, 2], [2, 3, 3], [3, 5]]
```

---

## 7. State Space Tree Structure and Pruning Strategies

### 7.1 Quantitative Analysis of State Space Trees

```
N-Queens state space tree node count comparison (N=8):

  No pruning (try all columns):
    Nodes ~ 8^8 = 16,777,216

  Column duplicate elimination only:
    Nodes ~ 8! = 40,320

  Column + diagonal pruning:
    Nodes ~ 2,057 (dramatic reduction)

  Bitmask optimization:
    Same number of nodes, but O(1) processing per node

  +----------------------------------------------+
  | Node count by pruning level (N=8)            |
  |                                              |
  | 16,777,216  ======================== Full    |
  |     40,320  =                        Column  |
  |      2,057  |                        Col+Diag|
  |                                              |
  | -> Proper pruning eliminates 99.99% of nodes |
  +----------------------------------------------+
```

### 7.2 Taxonomy of Pruning Strategies

| Strategy Category | Strategy Name | Description | Application | Reduction Effect |
|:---|:---|:---|:---|:---|
| **Constraint propagation** | Constraint check | Check if current choice satisfies constraints | N-Queens (col/diagonal) | High |
| **Constraint propagation** | Forward checking | Pre-filter candidates of unassigned variables | Sudoku (candidate sets) | High |
| **Constraint propagation** | Arc consistency | Maintain consistency between two variables | CSP in general | Very high |
| **Variable ordering** | MRV | Choose variable with fewest candidates first | Sudoku | Very high |
| **Variable ordering** | Degree heuristic | Choose variable with most constraints first | Graph coloring | High |
| **Value ordering** | LCV | Choose value that least restricts other variables | CSP in general | Medium |
| **Symmetry** | Symmetry breaking | Eliminate equivalent solutions (rotation/reflection) | N-Queens | Medium |
| **Bounding** | Branch and bound | Reduce search range using upper/lower bounds | Knapsack, TSP | High |
| **Ordering** | Order constraint | Eliminate duplicates via ascending order constraint | Combination enumeration | High |

### 7.3 Pruning Implementation Patterns

```python
# ---- Pattern 1: Filtering-based pruning ----
# Pre-filter choices that violate constraints
def backtrack_filter(state, all_choices, result):
    if is_solution(state):
        result.append(state.copy())
        return
    # Filter to valid choices only
    valid_choices = [c for c in all_choices if is_valid(state, c)]
    for choice in valid_choices:
        state.append(choice)
        backtrack_filter(state, all_choices, result)
        state.pop()


# ---- Pattern 2: Bound-based pruning (branch and bound) ----
# Estimate best case from current partial solution, compare with known best
best_value = float('inf')

def backtrack_bound(state, choices, cost_so_far):
    global best_value
    if is_solution(state):
        best_value = min(best_value, cost_so_far)
        return

    for choice in choices:
        if is_valid(state, choice):
            # Lower bound estimate: current cost + estimated minimum remaining cost
            lower_bound = cost_so_far + estimate_remaining(state, choice)
            if lower_bound >= best_value:
                continue  # This branch cannot improve the best solution

            apply(state, choice)
            backtrack_bound(state, next_choices(choice),
                          cost_so_far + cost(choice))
            undo(state, choice)


# ---- Pattern 3: Symmetry breaking pruning ----
# Restrict first row placement to half in N-Queens
def nqueens_symmetry(n):
    """Halve the search space using symmetry"""
    solutions = []

    def backtrack(row, board, cols, diag1, diag2):
        if row == n:
            solutions.append(board[:])
            return

        # First row: only left half (right half is a mirror image)
        limit = (n + 1) // 2 if row == 0 else n
        for col in range(limit):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue
            board[row] = col
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)
            backtrack(row + 1, board, cols, diag1, diag2)
            board[row] = -1
            cols.discard(col)
            diag1.discard(row - col)
            diag2.discard(row + col)

    backtrack(0, [-1] * n, set(), set(), set())
    return solutions
```

---

## 8. Complexity Analysis and Optimization Techniques

### 8.1 Backtracking Complexity Summary

| Problem | Without pruning | With pruning | Number of solutions | Notes |
|:---|:---|:---|:---|:---|
| N-Queens | O(N^N) | ~O(N!) | N=8: 92 | Can be halved with symmetry breaking |
| Sudoku | O(9^81) | Hundreds to tens of thousands of nodes | 1 (typically) | MRV yields dramatic speedup |
| All permutations | O(N * N!) | O(N!) | N! | Optimal (no pruning needed) |
| All combinations | O(2^N) | O(NCk) | NCk | Order constraint eliminates duplicates |
| All subsets | O(N * 2^N) | O(2^N) | 2^N | Optimal (no pruning needed) |
| Graph coloring | O(m^V) | Problem-dependent | Problem-dependent | MRV + LCV effective |
| Knight's tour | O(8^(N^2)) | Near-linear with Warnsdorff | Problem-dependent | Heuristic essential |
| Parenthesis gen. | O(4^n) | O(C_n) | Catalan number | Pruning is very effective |
| Combination sum | Exponential | Problem-dependent | Problem-dependent | Sort + upper bound pruning |

### 8.2 Collection of Optimization Techniques

```
Optimization technique effectiveness comparison:

  Technique                        Effectiveness    Implementation complexity
  -------------------------------------------------------------------
  Well-designed pruning conditions      *****          ***
  Variable ordering (MRV)              *****          **
  Data structure choice (set/bitmask)  ****           **
  Symmetry breaking                    ***            ****
  Value ordering (LCV)                 ***            ***
  Memoization (avoiding duplicate      ****           **
    states)
  Iterative deepening (depth-limited)  **             *
  Parallelization                      ***            *****
```

### 8.3 Combining with Memoization

Combining backtracking with **memoization** avoids recomputing identical states. This also serves as a bridge to dynamic programming (DP).

```python
def can_partition(nums: list) -> bool:
    """
    Determine if an array can be partitioned into two subsets with equal sums

    Example of backtracking + memoization
    (speeds up cases that would TLE with pure backtracking)
    """
    total = sum(nums)
    if total % 2 != 0:
        return False

    target = total // 2
    n = len(nums)
    memo = {}

    def backtrack(idx: int, remaining: int) -> bool:
        if remaining == 0:
            return True
        if remaining < 0 or idx >= n:
            return False

        key = (idx, remaining)
        if key in memo:
            return memo[key]

        # Include or exclude nums[idx]
        result = (backtrack(idx + 1, remaining - nums[idx]) or
                  backtrack(idx + 1, remaining))

        memo[key] = result
        return result

    return backtrack(0, target)


# ---- Example ----
print(can_partition([1, 5, 11, 5]))   # True: [1,5,5] and [11]
print(can_partition([1, 2, 3, 5]))    # False
print(can_partition([3, 3, 3, 4, 5])) # True: [3,3,3] and [4,5]
```

```
Progressive transformation from backtracking to DP:

  Step 1: Naive backtracking
    -> Recursively explore all branches
    -> May recompute the same state many times

  Step 2: Backtracking + memoization (top-down DP)
    -> Cache computed states
    -> Return cached states in O(1)

  Step 3: Bottom-up DP (tabulation)
    -> Eliminate recursion, fill table iteratively
    -> No risk of stack overflow

  Conditions for transformation:
    - State is uniquely determined by a finite number of parameters
    - Subproblems overlap (overlapping subproblems)
    - Optimal substructure exists
```

### 8.4 Memory Optimization via Iterative Deepening

A technique that repeats depth-limited backtracking, increasing the depth by 1 each iteration. Effective when the optimal solution depth is shallow, reducing memory usage from O(bd) to O(d) (b: branching factor, d: depth).

```python
def iterative_deepening_backtrack(initial_state, max_depth: int):
    """
    Iterative deepening backtracking

    Parameters:
        initial_state: Initial state
        max_depth: Maximum search depth

    Returns:
        State if solution found, None otherwise
    """
    def depth_limited_search(state, depth_limit: int) -> bool:
        if is_solution(state):
            return True
        if depth_limit <= 0:
            return False  # Depth limit reached

        for choice in get_choices(state):
            if is_valid(state, choice):
                apply(state, choice)
                if depth_limited_search(state, depth_limit - 1):
                    return True
                undo(state, choice)

        return False

    for depth in range(max_depth + 1):
        if depth_limited_search(initial_state, depth):
            return initial_state
    return None
```

---

## 9. Anti-patterns

### 9.1 Anti-pattern 1: Forgetting to Undo the Backtrack

The most frequent bug in backtracking. If the state is not restored after modification, corrupted state propagates to subsequent explorations, causing solutions to be missed or invalid solutions to be included.

```python
# ============================================================
# BAD: Forgetting to undo the choice -> state corruption
# ============================================================
def bad_nqueens(board, row, n, solutions):
    if row == n:
        solutions.append(board[:])
        return
    for col in range(n):
        if is_safe(board, row, col):
            board[row] = col
            bad_nqueens(board, row + 1, n, solutions)
            # Forgot board[row] = -1!
            # -> For N=4, finds only 1 solution instead of 2,
            #   or invalid solutions may be included


# ============================================================
# GOOD: Always undo
# ============================================================
def good_nqueens(board, row, n, solutions):
    if row == n:
        solutions.append(board[:])
        return
    for col in range(n):
        if is_safe(board, row, col):
            board[row] = col
            good_nqueens(board, row + 1, n, solutions)
            board[row] = -1  # Always undo!
```

**Techniques to prevent forgetting to undo:**

1. **Pass immutable state**: Pass copies of state so undoing is unnecessary (increases memory usage)
2. **Context manager pattern**: Use Python's `with` statement to guarantee automatic undo
3. **Tuple concatenation**: Create new tuples like `path + (choice,)` (equivalent to copying)

```python
# Immutable version: undo-forgetting is structurally impossible
def safe_backtrack(path: tuple, remaining: tuple, result: list):
    """Safely backtrack using immutable data"""
    if not remaining:
        result.append(list(path))
        return
    for i in range(len(remaining)):
        # Creates a new tuple, so the original path is never modified
        safe_backtrack(
            path + (remaining[i],),
            remaining[:i] + remaining[i + 1:],
            result
        )

result = []
safe_backtrack((), (1, 2, 3), result)
print(result)  # [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]
```

### 9.2 Anti-pattern 2: Exhaustive Search Without Pruning

Performing constraint checks only when a solution is complete results in exploring a vast number of invalid branches.

```python
# ============================================================
# BAD: Check only after completion -> exponentially slow
# ============================================================
def bad_nqueens_late_check(n):
    """Generate all placements then validate (very slow)"""
    solutions = []

    def generate(row, board):
        if row == n:
            # Check all constraints only here
            if all_queens_safe(board, n):
                solutions.append(board[:])
            return
        for col in range(n):
            board[row] = col
            generate(row + 1, board)  # Explores all branches!

    generate(0, [-1] * n)
    return solutions
    # N=8: explores 16,777,216 nodes (8^8)


# ============================================================
# GOOD: Check incrementally at each step -> fast
# ============================================================
def good_nqueens_early_check(n):
    """Check constraints at each row with early pruning"""
    solutions = []

    def backtrack(row, board, cols, diag1, diag2):
        if row == n:
            solutions.append(board[:])
            return
        for col in range(n):
            if col not in cols and (row - col) not in diag1 and (row + col) not in diag2:
                board[row] = col
                cols.add(col)
                diag1.add(row - col)
                diag2.add(row + col)
                backtrack(row + 1, board, cols, diag1, diag2)
                board[row] = -1
                cols.discard(col)
                diag1.discard(row - col)
                diag2.discard(row + col)

    backtrack(0, [-1] * n, set(), set(), set())
    return solutions
    # N=8: explores ~2,057 nodes (99.99% reduction)
```

### 9.3 Anti-pattern 3: Forgetting to Copy Solutions

```python
# ============================================================
# BAD: Storing references directly -> all entries point to the same list
# ============================================================
def bad_subsets(nums):
    result = []
    path = []

    def backtrack(start):
        result.append(path)  # Appends reference -> all entries are the same object!
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1)
            path.pop()

    backtrack(0)
    return result
    # Result: [[], [], [], [], [], [], [], []] (all empty lists)


# ============================================================
# GOOD: Store copies
# ============================================================
def good_subsets(nums):
    result = []
    path = []

    def backtrack(start):
        result.append(path[:])  # Append a copy!
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1)
            path.pop()

    backtrack(0)
    return result
    # Result: [[], [1], [1,2], [1,2,3], [1,3], [2], [2,3], [3]]
```

### 9.4 Anti-pattern 4: Continuing Search Unnecessarily

Exploring all solutions when only one is needed.

```python
# ============================================================
# BAD: Searching for all solutions when one suffices
# ============================================================
def bad_sudoku(board):
    """Explores all solutions (Sudoku typically needs only one)"""
    solutions = []

    def backtrack(idx):
        if idx == len(empty_cells):
            solutions.append([row[:] for row in board])
            return  # Not returning True -> search continues

        r, c = empty_cells[idx]
        for num in range(1, 10):
            if is_valid(r, c, num):
                board[r][c] = num
                backtrack(idx + 1)
                board[r][c] = 0

    backtrack(0)
    return solutions  # List of all solutions (typically 1 for Sudoku)


# ============================================================
# GOOD: Stop immediately at the first solution
# ============================================================
def good_sudoku(board):
    """Stop immediately upon finding the first solution"""
    def backtrack(idx):
        if idx == len(empty_cells):
            return True  # Solution found -> return True immediately

        r, c = empty_cells[idx]
        for num in range(1, 10):
            if is_valid(r, c, num):
                board[r][c] = num
                if backtrack(idx + 1):
                    return True  # Propagate upward to halt all exploration
                board[r][c] = 0

        return False

    return backtrack(0)
```

---

## 10. Exercises

### 10.1 Foundation Level

**Problem 1: All Permutations of a String**

Generate all permutations of a given string in lexicographic order. Remove duplicates when the string contains repeated characters.

```python
def string_permutations(s: str) -> list:
    """
    Return all permutations of the string in lexicographic order

    >>> string_permutations("abc")
    ['abc', 'acb', 'bac', 'bca', 'cab', 'cba']
    >>> string_permutations("aba")
    ['aab', 'aba', 'baa']
    """
    result = []
    chars = sorted(s)
    used = [False] * len(chars)

    def backtrack(path: list):
        if len(path) == len(chars):
            result.append("".join(path))
            return

        for i in range(len(chars)):
            if used[i]:
                continue
            if i > 0 and chars[i] == chars[i - 1] and not used[i - 1]:
                continue

            used[i] = True
            path.append(chars[i])
            backtrack(path)
            path.pop()
            used[i] = False

    backtrack([])
    return result


# Test
assert string_permutations("abc") == ['abc', 'acb', 'bac', 'bca', 'cab', 'cba']
assert string_permutations("aba") == ['aab', 'aba', 'baa']
assert string_permutations("a") == ['a']
print("Problem 1: All tests passed")
```

**Problem 2: Filtered Power Set**

Return only those subsets of an integer list whose sum does not exceed a specified value.

```python
def filtered_subsets(nums: list, max_sum: int) -> list:
    """
    Return all subsets whose sum is at most max_sum

    >>> filtered_subsets([1, 2, 3], 3)
    [[], [1], [1, 2], [2], [3]]
    """
    result = []

    def backtrack(start: int, path: list, current_sum: int):
        result.append(path[:])

        for i in range(start, len(nums)):
            # Pruning: skip if sum would exceed max_sum
            if current_sum + nums[i] > max_sum:
                continue
            path.append(nums[i])
            backtrack(i + 1, path, current_sum + nums[i])
            path.pop()

    nums.sort()
    backtrack(0, [], 0)
    return result


# Test
result = filtered_subsets([1, 2, 3], 3)
assert [] in result
assert [1] in result
assert [1, 2] in result
assert [2] in result
assert [3] in result
assert [1, 2, 3] not in result  # Sum 6 > 3
print("Problem 2: All tests passed")
```

### 10.2 Intermediate Level

**Problem 3: Word Search**

Determine whether a specified word can be formed by traversing adjacent cells in a 2D grid. Each cell may be used only once.

```python
def word_search(board: list, word: str) -> bool:
    """
    Search for a word in a 2D grid

    >>> board = [
    ...     ['A','B','C','E'],
    ...     ['S','F','C','S'],
    ...     ['A','D','E','E']
    ... ]
    >>> word_search(board, "ABCCED")
    True
    >>> word_search(board, "SEE")
    True
    >>> word_search(board, "ABCB")
    False
    """
    if not board or not board[0] or not word:
        return False

    rows, cols = len(board), len(board[0])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def backtrack(r: int, c: int, idx: int) -> bool:
        if idx == len(word):
            return True

        if (r < 0 or r >= rows or c < 0 or c >= cols or
                board[r][c] != word[idx]):
            return False

        # Mark cell as used
        original = board[r][c]
        board[r][c] = '#'

        for dr, dc in directions:
            if backtrack(r + dr, c + dc, idx + 1):
                return True

        # Backtrack
        board[r][c] = original
        return False

    for r in range(rows):
        for c in range(cols):
            if backtrack(r, c, 0):
                return True
    return False


# Test
board = [
    ['A', 'B', 'C', 'E'],
    ['S', 'F', 'C', 'S'],
    ['A', 'D', 'E', 'E']
]
assert word_search([row[:] for row in board], "ABCCED") is True
assert word_search([row[:] for row in board], "SEE") is True
assert word_search([row[:] for row in board], "ABCB") is False
print("Problem 3: All tests passed")
```

**Problem 4: Palindrome Partitioning**

Find all ways to partition a string such that every part is a palindrome.

```python
def palindrome_partition(s: str) -> list:
    """
    Return all palindrome partitions of a string

    >>> palindrome_partition("aab")
    [['a', 'a', 'b'], ['aa', 'b']]
    """
    result = []

    def is_palindrome(sub: str) -> bool:
        return sub == sub[::-1]

    def backtrack(start: int, path: list):
        if start == len(s):
            result.append(path[:])
            return

        for end in range(start + 1, len(s) + 1):
            substring = s[start:end]
            if is_palindrome(substring):
                path.append(substring)
                backtrack(end, path)
                path.pop()

    backtrack(0, [])
    return result


# Test
assert palindrome_partition("aab") == [['a', 'a', 'b'], ['aa', 'b']]
assert palindrome_partition("aba") == [['a', 'b', 'a'], ['aba']]
print("Problem 4: All tests passed")
```

### 10.3 Advanced Level

**Problem 5: Hard Sudoku**

Solve the following hard Sudoku using the MRV heuristic solver.

```python
def solve_hard_sudoku():
    """
    Solve one of the world's hardest Sudoku puzzles (by Arto Inkala)

    With many empty cells, basic backtracking requires exploring
    many nodes. Verify the effectiveness of the MRV heuristic.
    """
    hard_board = [
        [8, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 6, 0, 0, 0, 0, 0],
        [0, 7, 0, 0, 9, 0, 2, 0, 0],
        [0, 5, 0, 0, 0, 7, 0, 0, 0],
        [0, 0, 0, 0, 4, 5, 7, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 3, 0],
        [0, 0, 1, 0, 0, 0, 0, 6, 8],
        [0, 0, 8, 5, 0, 0, 0, 1, 0],
        [0, 9, 0, 0, 0, 0, 4, 0, 0],
    ]

    # Solve using solve_sudoku_mrv (see Section 3.4)
    solve_sudoku_mrv(hard_board)

    # Validate
    for r in range(9):
        assert sorted(hard_board[r]) == list(range(1, 10)), f"Row {r} is invalid"
    for c in range(9):
        col_vals = [hard_board[r][c] for r in range(9)]
        assert sorted(col_vals) == list(range(1, 10)), f"Column {c} is invalid"

    print("Hard Sudoku solution:")
    for row in hard_board:
        print(" ".join(str(x) for x in row))

solve_hard_sudoku()
# Output:
# 8 1 2 7 5 3 6 4 9
# 9 4 3 6 8 2 1 7 5
# 6 7 5 4 9 1 2 8 3
# 1 5 4 2 3 7 8 9 6
# 3 6 9 8 4 5 7 2 1
# 2 8 7 1 6 9 5 3 4
# 5 2 1 9 7 4 3 6 8
# 4 3 8 5 2 6 9 1 7
# 7 9 6 3 1 8 4 5 2
```

**Problem 6: N-Queens Symmetry Analysis**

From all N-Queens solutions, count the number of "essentially distinct solutions" by removing those equivalent under rotation and reflection.

```python
def count_unique_nqueens(n: int) -> int:
    """
    Count the number of essentially distinct N-Queens solutions

    Group equivalent solutions under 8 symmetry operations
    (4 rotations x 2 reflections) and count only representatives.
    """
    all_solutions = solve_nqueens_fast(n)  # See Section 2.3

    def rotate_90(sol):
        """90-degree clockwise rotation"""
        return [sol.index(n - 1 - i) for i in range(n)]

    def reflect(sol):
        """Left-right reflection"""
        return [n - 1 - col for col in sol]

    def canonical(sol):
        """Return the lexicographically smallest form among 8 symmetry operations"""
        variants = []
        current = sol[:]
        for _ in range(4):
            variants.append(tuple(current))
            variants.append(tuple(reflect(current)))
            current = rotate_90(current)
        return min(variants)

    unique = set()
    for sol in all_solutions:
        unique.add(canonical(sol))

    return len(unique)


# Test
for n in range(1, 11):
    total = len(solve_nqueens_fast(n))
    unique = count_unique_nqueens(n)
    print(f"N={n:2d}: total solutions={total:5d}, essentially distinct={unique:4d}")

# Output:
# N= 1: total solutions=    1, essentially distinct=   1
# N= 2: total solutions=    0, essentially distinct=   0
# N= 3: total solutions=    0, essentially distinct=   0
# N= 4: total solutions=    2, essentially distinct=   1
# N= 5: total solutions=   10, essentially distinct=   2
# N= 6: total solutions=    4, essentially distinct=   1
# N= 7: total solutions=   40, essentially distinct=   6
# N= 8: total solutions=   92, essentially distinct=  12
# N= 9: total solutions=  352, essentially distinct=  46
# N=10: total solutions=  724, essentially distinct=  92
```

---

## 11. FAQ

### Q1: What is the difference between backtracking and DFS?

**A:** DFS (depth-first search) is a concrete algorithm for traversing graphs or trees. Backtracking, on the other hand, is a **problem-solving design pattern** based on DFS, with the added strategy of "pruning branches that violate constraints." Not all DFS is backtracking, but backtracking is always a form of DFS.

```
         DFS (graph traversal)         Backtracking
  +------------------------+    +-----------------------------+
  | Visit all nodes        |    | Search for constrained sols |
  | No pruning             |    | Prune on constraint violation|
  | Visited check          |    | Choose->Validate->Recurse   |
  | Explore graph structure|    |  ->Undo                     |
  +------------------------+    | Explore solution space       |
                                +-----------------------------+
```

### Q2: How can the complexity of backtracking be improved?

**A:** The five main improvement techniques are:

1. **Strong pruning conditions**: Detect constraint violations as early as possible. In N-Queens, check columns and diagonals row by row.
2. **Variable ordering optimization (MRV)**: Process variables with the fewest choices first. In Sudoku, fill cells with the minimum number of candidates first.
3. **Data structure improvements**: Use sets or bitmasks to make constraint checking O(1).
4. **Symmetry breaking**: Avoid exploring equivalent solutions redundantly. In N-Queens, restrict the first row to the left half.
5. **Branch and bound**: Prune branches that cannot improve the best known solution using lower bound estimation. Effective for optimization problems.

### Q3: How does the implementation differ between finding all solutions and finding one?

**A:** The control flow differs.

```python
# Finding all solutions: add to list and continue exploring
def find_all(state, result):
    if is_solution(state):
        result.append(state.copy())
        return          # <- void: exploration continues in the upper loop

    for choice in choices:
        if is_valid(state, choice):
            apply(state, choice)
            find_all(state, result)  # Return value not used
            undo(state, choice)

# Finding one solution: return True immediately and stop
def find_one(state):
    if is_solution(state):
        return True     # <- Exit immediately

    for choice in choices:
        if is_valid(state, choice):
            apply(state, choice)
            if find_one(state):
                return True   # <- Propagate True upward
            undo(state, choice)

    return False
```

### Q4: How do you choose between backtracking and dynamic programming (DP)?

**A:** The criteria are "subproblem overlap" and "state space structure."

| Property | Backtracking | Dynamic Programming |
|:---|:---|:---|
| Subproblem overlap | None (each state visited once) | Present (same state computed many times) |
| Solution structure | Combinatorial (enumeration) | Optimal value (max/min/count) |
| Effect of memoization | Low (states don't recur) | High (dramatic speedup) |
| Suitable problems | N-Queens, Sudoku, permutation enum. | Knapsack, longest subsequence |

Backtracking with memoization added is equivalent to "top-down DP." If the problem has overlapping subproblems, choose DP; otherwise, choose backtracking.

### Q5: How do you prevent stack overflow in backtracking?

**A:** Three approaches:

1. **Recursion depth limit**: Adjust the limit in Python with `sys.setrecursionlimit()`. Not a fundamental solution.
2. **Iterative deepening**: Gradually deepen the depth-limited search to control memory usage (see Section 8.4).
3. **Explicit stack for non-recursive implementation**: Rewrite recursion using a loop and stack.

```python
# Non-recursive backtracking (explicit stack)
def iterative_permutations(nums):
    result = []
    # Stack elements: (path, remaining)
    stack = [([], list(nums))]

    while stack:
        path, remaining = stack.pop()
        if not remaining:
            result.append(path)
            continue
        # Add in reverse order (to maintain DFS order)
        for i in range(len(remaining) - 1, -1, -1):
            new_path = path + [remaining[i]]
            new_remaining = remaining[:i] + remaining[i + 1:]
            stack.append((new_path, new_remaining))

    return result

print(iterative_permutations([1, 2, 3]))
# [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]
```

### Q6: What is the relationship between constraint satisfaction problems (CSP) and backtracking?

**A:** Backtracking is the standard algorithm for solving CSPs. A CSP is defined by three components: a set of variables, a domain for each variable, and a set of constraints. The goal is to find an assignment of values to variables that satisfies all constraints. Sudoku, graph coloring, and N-Queens can all be formulated as CSPs, and are efficiently solved by backtracking combined with constraint propagation.

---


## FAQ

### Q1: What is the most important point when studying this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory, but by actually writing code and verifying behavior.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving on.

### Q3: How is this used in practice?

Knowledge of this topic is frequently applied in everyday development work. It is particularly important during code reviews and architecture design.

---

## 12. Summary

### 12.1 Key Concepts

| Item | Key Point |
|:---|:---|
| Definition of backtracking | Systematic search technique using DFS + pruning |
| Basic cycle | Choose -> Validate -> Explore (recurse) -> Unchoose |
| N-Queens | Row, column, and diagonal constraints. Speed up with sets or bitmasks |
| Sudoku | Row, column, and box constraints. MRV heuristic is dramatically effective |
| Permutations, combinations, subsets | Three fundamental enumeration patterns. Distinguish between start parameter and used array |
| Graph coloring | Adjacent vertex color constraint. Representative CSP example |
| Knight's tour | Warnsdorff's rule for speedup. Demonstrates the importance of heuristics |
| Effect of pruning | Proper pruning can reduce search space by over 99% |
| Combination with memoization | Can be transformed to top-down DP when duplicate states exist |

### 12.2 Backtracking Implementation Checklist

```
Verification items when implementing backtracking:

  [ ] Is the base condition (solution completeness check) correct?
  [ ] Are choices enumerated completely and without duplication?
  [ ] Are pruning conditions correct and sufficiently strong?
  [ ] Are state changes properly undone?
  [ ] Are solution copies being saved (not references)?
  [ ] Is the control flow appropriate for all-solutions vs. one-solution?
  [ ] Is duplicate element handling correct?
  [ ] Is recursion depth sufficient (stack overflow risk)?
```

### 12.3 Technique Selection Guide

```
Algorithm selection by problem type:

  Enumeration problem (find all solutions)
    -> Backtracking

  Optimization problem (find the best solution)
    -> Branch and bound (backtracking + upper/lower bounds)
    -> If subproblems overlap -> Dynamic programming

  Decision problem (confirm solution existence)
    -> Backtracking (stop at first solution)

  Constraint satisfaction problem (find assignment satisfying all constraints)
    -> Backtracking + constraint propagation (AC-3, etc.)
    -> MRV + LCV heuristics
```

---

## Recommended Next Readings

- [Graph Traversal](./02-graph-traversal.md) -- Understanding DFS, the foundation of backtracking
- [Dynamic Programming](./04-dynamic-programming.md) -- Transitioning from backtracking + memoization to DP
- [Problem-Solving Methods](../04-practice/00-problem-solving.md) -- Deciding when to apply backtracking and choosing approaches

---

## 13. References

1. Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. -- Chapter 34: Theoretical foundations of NP-completeness and search algorithms
2. Knuth, D. E. (2000). "Dancing Links." *arXiv preprint cs/0011047*. -- Application to exact cover problems and efficient Sudoku solving (Algorithm X + DLX)
3. Skiena, S. S. (2020). *The Algorithm Design Manual* (3rd ed.). Springer. -- Chapter 9: Combinatorial Search and Heuristic Methods. Systematic treatment of backtracking
4. Wirth, N. (1976). *Algorithms + Data Structures = Programs*. Prentice-Hall. -- Classic treatment of N-Queens and backtracking
5. Russell, S. J., & Norvig, P. (2021). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson. -- Chapter 6: Constraint Satisfaction Problems. Systematic positioning of CSP theory and backtracking
6. Sedgewick, R., & Wayne, K. (2011). *Algorithms* (4th ed.). Addison-Wesley. -- Subset enumeration and backtracking implementation patterns
7. LeetCode. "Backtracking Problems." https://leetcode.com/tag/backtracking/ -- Practice problem collection for backtracking (N-Queens, Sudoku Solver, Word Search, etc.)

---

## References

- [MDN Web Docs](https://developer.mozilla.org/) - Web technology reference
- [Wikipedia](https://en.wikipedia.org/) - Overview of technical concepts
