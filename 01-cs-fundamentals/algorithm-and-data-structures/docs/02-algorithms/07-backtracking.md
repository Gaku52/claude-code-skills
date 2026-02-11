# バックトラッキング

> 解の候補を体系的に探索し、制約を満たさない分岐を早期に枝刈りして効率化する手法を、N-Queens・数独・順列生成で理解する

## この章で学ぶこと

1. **バックトラッキングの原理**（状態空間木の探索と枝刈り）を理解し、テンプレートを使って実装できる
2. **N-Queens問題・数独ソルバー**を枝刈り戦略とともに効率的に解ける
3. **順列・組み合わせ・部分集合**の列挙を体系的に実装できる

---

## 1. バックトラッキングの原理

```
バックトラッキング = DFS + 枝刈り

状態空間木（3つの中から2つ選ぶ例）:
                root
              / | \
             1  2  3     ← 1番目の選択
            / \  |
           2   3 3       ← 2番目の選択
           |   | |
          [1,2][1,3][2,3] ← 解

枝刈りなしの全探索:
                root
              / | \
             1  2  3
           /|\ /|\ /|\
          1 2 3 1 2 3 1 2 3    ← 重複・無効を含む全探索

枝刈りあり:
                root
              / | \
             1  2  3
            / \  |             ← 「自分以下の数字のみ」で枝刈り
           2   3 3
           |   | |
          [1,2][1,3][2,3]     ← 有効な解のみ探索
```

### 基本テンプレート

```python
def backtrack(state, choices, result):
    """バックトラッキングの基本テンプレート"""
    # 基底条件: 解が見つかった
    if is_solution(state):
        result.append(state.copy())
        return

    for choice in choices:
        # 枝刈り: この選択が有効か？
        if is_valid(state, choice):
            # 選択を適用
            state.append(choice)

            # 再帰的に探索
            backtrack(state, choices, result)

            # 選択を取り消し（バックトラック）
            state.pop()
```

---

## 2. N-Queens 問題

N x N のチェス盤に N 個のクイーンを、互いに攻撃し合わないように配置する。

```
4-Queens の解:

解1:             解2:
. Q . .          . . Q .
. . . Q          Q . . .
Q . . .          . . . Q
. . Q .          . Q . .

クイーンの攻撃範囲:        枝刈りの判定:
. . * . * . .              行: 各行に1つずつ配置
. . . Q . . .              列: used_cols で管理
. . * . * . .              対角線: row-col で管理（左上→右下）
. * . . . * .              反対角線: row+col で管理（右上→左下）
* . . . . . *
```

```python
def solve_nqueens(n: int) -> list:
    """N-Queens - バックトラッキング"""
    solutions = []
    board = [-1] * n  # board[row] = col

    def is_safe(row, col):
        for prev_row in range(row):
            prev_col = board[prev_row]
            # 同じ列 or 同じ対角線
            if (prev_col == col or
                abs(prev_row - row) == abs(prev_col - col)):
                return False
        return True

    def backtrack(row):
        if row == n:
            solutions.append(board[:])
            return

        for col in range(n):
            if is_safe(row, col):
                board[row] = col
                backtrack(row + 1)
                board[row] = -1  # バックトラック

    backtrack(0)
    return solutions

# 最適化版: ビットマスクで高速判定
def solve_nqueens_optimized(n: int) -> int:
    """N-Queens (解の数) - ビットマスク最適化"""
    count = 0

    def backtrack(row, cols, diag1, diag2):
        nonlocal count
        if row == n:
            count += 1
            return

        available = ((1 << n) - 1) & ~(cols | diag1 | diag2)
        while available:
            pos = available & (-available)  # 最下位ビット
            available ^= pos
            backtrack(row + 1,
                      cols | pos,
                      (diag1 | pos) << 1,
                      (diag2 | pos) >> 1)

    backtrack(0, 0, 0, 0)
    return count

# 4-Queens
solutions = solve_nqueens(4)
for sol in solutions:
    print(sol)
# [1, 3, 0, 2] と [2, 0, 3, 1]

# 8-Queens の解の数
print(solve_nqueens_optimized(8))   # 92
print(solve_nqueens_optimized(12))  # 14200
```

### 解の可視化

```python
def print_board(solution: list):
    """N-Queens の盤面を表示"""
    n = len(solution)
    for row in range(n):
        line = ""
        for col in range(n):
            line += "Q " if solution[row] == col else ". "
        print(line)
    print()

solutions = solve_nqueens(4)
for sol in solutions:
    print_board(sol)
```

---

## 3. 数独ソルバー

9x9 の数独を、行・列・3x3ボックスの制約を満たすように埋める。

```
入力:                     出力:
5 3 . | . 7 . | . . .    5 3 4 | 6 7 8 | 9 1 2
6 . . | 1 9 5 | . . .    6 7 2 | 1 9 5 | 3 4 8
. 9 8 | . . . | . 6 .    1 9 8 | 3 4 2 | 5 6 7
------+-------+------    ------+-------+------
8 . . | . 6 . | . . 3    8 5 9 | 7 6 1 | 4 2 3
4 . . | 8 . 3 | . . 1    4 2 6 | 8 5 3 | 7 9 1
7 . . | . 2 . | . . 6    7 1 3 | 9 2 4 | 8 5 6
------+-------+------    ------+-------+------
. 6 . | . . . | 2 8 .    9 6 1 | 5 3 7 | 2 8 4
. . . | 4 1 9 | . . 5    2 8 7 | 4 1 9 | 6 3 5
. . . | . 8 . | . 7 9    3 4 5 | 2 8 6 | 1 7 9
```

```python
def solve_sudoku(board: list) -> bool:
    """数独ソルバー - バックトラッキング"""

    def find_empty():
        """空のセルを見つける"""
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0:
                    return r, c
        return None

    def is_valid(row, col, num):
        """num を (row, col) に置けるか"""
        # 行チェック
        if num in board[row]:
            return False
        # 列チェック
        if num in [board[r][col] for r in range(9)]:
            return False
        # 3x3ボックスチェック
        box_r, box_c = 3 * (row // 3), 3 * (col // 3)
        for r in range(box_r, box_r + 3):
            for c in range(box_c, box_c + 3):
                if board[r][c] == num:
                    return False
        return True

    cell = find_empty()
    if cell is None:
        return True  # 全て埋まった → 解発見

    row, col = cell
    for num in range(1, 10):
        if is_valid(row, col, num):
            board[row][col] = num

            if solve_sudoku(board):
                return True

            board[row][col] = 0  # バックトラック

    return False  # この分岐に解なし

# 高速版: 候補集合を事前計算
def solve_sudoku_fast(board: list) -> bool:
    """数独ソルバー（候補集合による高速化）"""
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]
    empty = []

    # 初期化
    for r in range(9):
        for c in range(9):
            if board[r][c] != 0:
                num = board[r][c]
                rows[r].add(num)
                cols[c].add(num)
                boxes[3 * (r // 3) + c // 3].add(num)
            else:
                empty.append((r, c))

    def backtrack(idx):
        if idx == len(empty):
            return True

        r, c = empty[idx]
        box_id = 3 * (r // 3) + c // 3

        for num in range(1, 10):
            if num not in rows[r] and num not in cols[c] and num not in boxes[box_id]:
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

# 使用例
board = [
    [5,3,0,0,7,0,0,0,0],
    [6,0,0,1,9,5,0,0,0],
    [0,9,8,0,0,0,0,6,0],
    [8,0,0,0,6,0,0,0,3],
    [4,0,0,8,0,3,0,0,1],
    [7,0,0,0,2,0,0,0,6],
    [0,6,0,0,0,0,2,8,0],
    [0,0,0,4,1,9,0,0,5],
    [0,0,0,0,8,0,0,7,9],
]
solve_sudoku_fast(board)
for row in board:
    print(row)
```

---

## 4. 順列・組み合わせ・部分集合

### 順列の生成

```python
def permutations(nums: list) -> list:
    """全順列の生成 - O(n!)"""
    result = []

    def backtrack(path, remaining):
        if not remaining:
            result.append(path[:])
            return
        for i in range(len(remaining)):
            path.append(remaining[i])
            backtrack(path, remaining[:i] + remaining[i+1:])
            path.pop()

    backtrack([], nums)
    return result

print(permutations([1, 2, 3]))
# [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]
```

### 組み合わせの生成

```python
def combinations(nums: list, k: int) -> list:
    """nCk の全組み合わせ - O(nCk)"""
    result = []

    def backtrack(start, path):
        if len(path) == k:
            result.append(path[:])
            return
        # 枝刈り: 残り要素数が足りない場合スキップ
        for i in range(start, len(nums) - (k - len(path)) + 1):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    backtrack(0, [])
    return result

print(combinations([1, 2, 3, 4], 2))
# [[1,2], [1,3], [1,4], [2,3], [2,4], [3,4]]
```

### 部分集合の列挙

```python
def subsets(nums: list) -> list:
    """全部分集合（べき集合）- O(2^n)"""
    result = []

    def backtrack(start, path):
        result.append(path[:])  # 全ての時点で解
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    backtrack(0, [])
    return result

print(subsets([1, 2, 3]))
# [[], [1], [1,2], [1,2,3], [1,3], [2], [2,3], [3]]
```

---

## 5. 状態空間木の構造

```
順列 {1,2,3} の状態空間木:

              {}
           /  |  \
         1    2    3
        / \  / \  / \
      12  13 21 23 31 32
      |   |  |  |  |  |
     123 132 213 231 312 321

  ノード数: 1 + 3 + 6 + 6 = 16
  解の数: 3! = 6

組み合わせ C(4,2) の状態空間木:

              {}
          /  |  \  \
         1   2   3  4
        /|\ /\  |
      12 13 14 23 24 34

  ノード数: 1 + 4 + 6 = 11
  解の数: C(4,2) = 6
```

---

## 6. 枝刈り戦略比較表

| 戦略 | 説明 | 適用例 |
|:---|:---|:---|
| 制約チェック | 現在の選択が制約を満たすか | N-Queens（同列/対角線チェック） |
| 候補削減 | 有効な候補を事前に絞る | 数独（候補集合の管理） |
| 対称性除去 | 対称な解を重複して探索しない | N-Queens（回転・鏡像の排除） |
| 上界/下界 | 分枝限定法で探索範囲を削減 | ナップサック・TSP |
| 順序制約 | 昇順等の制約で重複を排除 | 組み合わせ（start パラメータ） |

## バックトラッキングの計算量

| 問題 | 枝刈りなし | 枝刈りあり | 解の数 |
|:---|:---|:---|:---|
| N-Queens | O(n^n) | O(n!) 程度 | n=8: 92 |
| 数独 | O(9^81) | 実測数百〜数千ノード | 1（通常） |
| 全順列 | O(n * n!) | O(n!) | n! |
| 全組み合わせ | O(2^n) | O(nCk) | nCk |
| 全部分集合 | O(2^n) | O(2^n) | 2^n |

---

## 7. アンチパターン

### アンチパターン1: バックトラックの取り消し忘れ

```python
# BAD: 選択の取り消しを忘れる → 状態が汚染される
def bad_backtrack(board, row, n):
    if row == n:
        solutions.append(board[:])
        return
    for col in range(n):
        if is_safe(board, row, col):
            board[row] = col
            bad_backtrack(board, row + 1, n)
            # board[row] = -1 を忘れている!
            # → 次の反復で前の値が残る

# GOOD: 必ず取り消す
def good_backtrack(board, row, n):
    if row == n:
        solutions.append(board[:])
        return
    for col in range(n):
        if is_safe(board, row, col):
            board[row] = col
            good_backtrack(board, row + 1, n)
            board[row] = -1  # 取り消し!
```

### アンチパターン2: 枝刈りなしの全探索

```python
# BAD: 制約チェックを解の完成時にだけ行う
def bad_nqueens(board, row, n):
    if row == n:
        if is_valid_complete(board):  # 完成後にチェック → 遅い
            solutions.append(board[:])
        return
    for col in range(n):
        board[row] = col
        bad_nqueens(board, row + 1, n)  # 全分岐を探索

# GOOD: 各ステップで制約チェック（早期枝刈り）
def good_nqueens(board, row, n):
    if row == n:
        solutions.append(board[:])
        return
    for col in range(n):
        if is_safe(board, row, col):  # 逐次チェック → 高速
            board[row] = col
            good_nqueens(board, row + 1, n)
            board[row] = -1
```

---

## 8. FAQ

### Q1: バックトラッキングとDFSの違いは？

**A:** DFS はグラフ/木を走査する具体的なアルゴリズム。バックトラッキングは DFS を使った問題解決の設計パターンで、「制約を満たさない分岐を枝刈りする」という戦略が加わる。バックトラッキングは DFS の一種だが、DFS が全てバックトラッキングではない。

### Q2: バックトラッキングの計算量を改善する方法は？

**A:** (1) 強力な枝刈り条件を追加する。(2) 変数の順序を工夫する（最も制約の厳しい変数を先に選択 = MRV ヒューリスティック）。(3) 分枝限定法（Branch and Bound）で上界/下界を用いて探索空間を削減する。(4) 対称性を利用して重複探索を排除する。

### Q3: バックトラッキングで全解を求めるか、一つの解を求めるかで実装はどう変わる？

**A:** 一つの解を求める場合は、解が見つかった時点で `return True` して探索を打ち切る。全解を求める場合は、解をリストに追加して探索を続行する。数独は通常一つの解、N-Queens は全解を求めることが多い。

---

## 9. まとめ

| 項目 | 要点 |
|:---|:---|
| バックトラッキング | DFS + 枝刈りによる体系的探索 |
| 基本構造 | 選択→検証→再帰→取り消し のサイクル |
| N-Queens | 行・列・対角線の制約で枝刈り。ビットマスクで高速化 |
| 数独 | 行・列・ボックスの制約。候補集合管理で高速化 |
| 列挙問題 | 順列 O(n!)、組み合わせ O(nCk)、部分集合 O(2^n) |
| 枝刈りの重要性 | 適切な枝刈りで探索空間を桁違いに削減 |

---

## 次に読むべきガイド

- [グラフ走査](./02-graph-traversal.md) -- バックトラッキングの基盤となるDFS
- [動的計画法](./04-dynamic-programming.md) -- バックトラッキング+メモ化で DP に移行
- [問題解決法](../04-practice/00-problem-solving.md) -- バックトラッキングの適用判断

---

## 参考文献

1. Cormen, T. H. et al. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. -- 第34章（NP完全問題）
2. Knuth, D. E. (2000). "Dancing Links." *arXiv preprint*. -- 数独の効率的解法
3. Skiena, S. S. (2020). *The Algorithm Design Manual* (3rd ed.). Springer. -- 第9章: Combinatorial Search
4. Wirth, N. (1976). *Algorithms + Data Structures = Programs*. Prentice-Hall.
