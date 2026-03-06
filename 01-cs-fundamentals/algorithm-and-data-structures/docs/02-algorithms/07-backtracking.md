# バックトラッキング

> 解の候補を体系的に探索し、制約を満たさない分岐を早期に枝刈りして効率化する手法を、N-Queens・数独・順列生成・グラフ彩色・ナイトツアーなど多角的な例題で深く理解する

## この章で学ぶこと

1. **バックトラッキングの原理**（状態空間木の探索と枝刈り）を理解し、汎用テンプレートを使って実装できる
2. **N-Queens 問題・数独ソルバー**を複数の枝刈り戦略とともに効率的に解ける
3. **順列・組み合わせ・部分集合**の列挙を体系的に実装し、重複要素への対応もできる
4. **グラフ彩色・ナイトツアー・括弧列挙**など応用問題にバックトラッキングを適用できる
5. **枝刈り戦略の設計**と計算量分析を行い、最適化の方針を立てられる

---

## 目次

1. [バックトラッキングの原理](#1-バックトラッキングの原理)
2. [N-Queens 問題](#2-n-queens-問題)
3. [数独ソルバー](#3-数独ソルバー)
4. [順列・組み合わせ・部分集合](#4-順列組み合わせ部分集合)
5. [応用問題: グラフ彩色](#5-応用問題-グラフ彩色)
6. [応用問題: ナイトツアーと括弧生成](#6-応用問題-ナイトツアーと括弧生成)
7. [状態空間木の構造と枝刈り戦略](#7-状態空間木の構造と枝刈り戦略)
8. [計算量分析と最適化テクニック](#8-計算量分析と最適化テクニック)
9. [アンチパターン](#9-アンチパターン)
10. [演習問題](#10-演習問題)
11. [FAQ](#11-faq)
12. [まとめ](#12-まとめ)
13. [参考文献](#13-参考文献)

---

## 1. バックトラッキングの原理

### 1.1 概要と直感的理解

バックトラッキング（backtracking）は、探索問題を解くための体系的なアプローチである。迷路を解くときの行動を思い浮かべるとわかりやすい。分岐点に到達したら一つの道を選んで進み、行き止まりに当たったら最後の分岐点まで戻って別の道を試す。この「進む→行き詰まる→戻る→別の道を試す」というサイクルが、バックトラッキングの本質である。

```
バックトラッキング = 深さ優先探索（DFS） + 枝刈り（Pruning）

迷路の例で理解するバックトラッキング:

  S → → ↓         S → → ↓         S → → ↓
           ↓              ↓              ↓
       ↓ ← ←          × ← ←          ↓ ← ←
       ↓                               ↓
       × (行き止まり)                   → → G (ゴール!)

  Step 1: 進む      Step 2: 戻る      Step 3: 別の道
```

アルゴリズムの核心は以下の3つのステップにある:

1. **選択（Choose）**: 利用可能な選択肢の中から一つを選ぶ
2. **探索（Explore）**: その選択のもとで再帰的に探索を続ける
3. **取り消し（Unchoose）**: 探索が終わったら選択を元に戻す

### 1.2 状態空間木

バックトラッキングの探索過程は、**状態空間木（State Space Tree）** として視覚化できる。根ノードは初期状態、各辺は一つの選択、葉ノードは解または行き止まりを表す。

```
状態空間木（3つの中から2つ選ぶ例）:

                    root (初期状態)
                  /   |   \
                 1    2    3         ← 1番目の選択
                / \   |
               2   3  3             ← 2番目の選択
               |   |  |
             [1,2][1,3][2,3]        ← 解（葉ノード）


枝刈りなしの全探索:

                    root
                  /   |   \
                 1    2    3
               / | \ / | \ / | \
              1 2 3 1 2 3 1 2 3      ← 重複・無効を含む全探索
              探索ノード数: 1 + 3 + 9 = 13


枝刈りあり:

                    root
                  /   |   \
                 1    2    3
                / \   |    X         ← 「自分より大きい数字のみ」で枝刈り
               2   3  3
               |   |  |
             [1,2][1,3][2,3]         ← 有効な解のみ探索
              探索ノード数: 1 + 3 + 3 = 7（約46%削減）
```

枝刈りの効果は問題のサイズが大きくなるほど劇的に増大する。N-Queens 問題では、枝刈りなしの場合 O(N^N) のノードを探索するが、適切な枝刈りにより O(N!) 程度まで削減される。N=8 の場合、16,777,216 ノードが 40,320 ノード以下になる。

### 1.3 基本テンプレート

バックトラッキングには汎用的なテンプレートが存在し、ほぼ全ての問題に適用できる。

```python
def backtrack(state, choices, result):
    """
    バックトラッキングの基本テンプレート

    Parameters:
        state   : 現在の部分解（可変オブジェクト）
        choices : 利用可能な選択肢
        result  : 見つかった解を格納するリスト
    """
    # ---- 基底条件: 解が完成したか判定 ----
    if is_solution(state):
        result.append(state.copy())  # 解のコピーを保存
        return

    # ---- 各選択肢を試す ----
    for choice in choices:
        # 枝刈り: この選択が制約を満たすか？
        if is_valid(state, choice):
            # 1. 選択を適用（Choose）
            apply(state, choice)

            # 2. 再帰的に探索（Explore）
            backtrack(state, next_choices(choice), result)

            # 3. 選択を取り消し（Unchoose / Backtrack）
            undo(state, choice)


def backtrack_single(state, choices):
    """
    一つの解だけを求めるバリエーション

    Returns:
        bool: 解が見つかったら True
    """
    if is_solution(state):
        return True

    for choice in choices:
        if is_valid(state, choice):
            apply(state, choice)

            if backtrack_single(state, next_choices(choice)):
                return True  # 解が見つかったら即座に打ち切り

            undo(state, choice)

    return False  # この分岐には解がない
```

**テンプレートの使い分け:**

| 目的 | テンプレート | return 値 | 解の格納 |
|:---|:---|:---|:---|
| 全ての解を列挙 | `backtrack` | なし (void) | `result` リストに追加 |
| 一つの解を発見 | `backtrack_single` | `bool` | state を直接参照 |
| 解の数を数える | カウンタ版 | `int` | グローバル変数 or 戻り値 |
| 最適解を求める | 最適化版 | なし or 値 | 最良解を更新 |

### 1.4 バックトラッキングが有効な問題の特徴

バックトラッキングが特に有効なのは、以下の性質を持つ問題である:

1. **逐次選択性**: 解が一連の選択の列として構成できる
2. **制約の早期検出**: 部分解の段階で制約違反を検出できる
3. **探索空間の削減可能性**: 枝刈りにより大幅にノード数を減らせる

```
バックトラッキングの適用判断フローチャート:

  問題を受け取る
       |
       v
  解は選択の列として
  構成できるか？ ──No──> 他の手法を検討
       |
      Yes
       |
       v
  部分解の段階で
  制約違反を検出  ──No──> 全探索 or DP を検討
  できるか？
       |
      Yes
       |
       v
  枝刈りで探索空間を
  大幅削減できるか？ ──No──> 全探索 + 枝刈り（効果は限定的）
       |
      Yes
       |
       v
  バックトラッキングが有効!
```

---

## 2. N-Queens 問題

### 2.1 問題定義

N x N のチェス盤に N 個のクイーンを、互いに攻撃し合わないように配置する問題である。チェスのクイーンは、同じ行・同じ列・同じ対角線上にある駒を攻撃できる。

```
4-Queens の2つの解:

  解1:                解2:
  . Q . .             . . Q .
  . . . Q             Q . . .
  Q . . .             . . . Q
  . . Q .             . Q . .

  board = [1,3,0,2]   board = [2,0,3,1]


クイーンの攻撃範囲（中央に配置した場合）:

  \ . | . /           攻撃判定に使う3つの条件:
  . \ | / .
  ----Q----           1. 同じ列: col == prev_col
  . / | \ .           2. 左上-右下対角線: row - col == prev_row - prev_col
  / . | . \           3. 右上-左下対角線: row + col == prev_row + prev_col

                      => 条件2,3は |row-prev_row| == |col-prev_col| と等価
```

### 2.2 基本実装

各行に一つずつクイーンを配置していく方式で実装する。行ごとに処理するため、行の制約は自動的に満たされる。

```python
def solve_nqueens(n: int) -> list:
    """
    N-Queens 問題を解き、全ての解を返す

    Parameters:
        n: 盤面のサイズ（クイーンの数）

    Returns:
        解のリスト。各解は board[row] = col の形式
    """
    solutions = []
    board = [-1] * n  # board[row] = col（その行のクイーンの列位置）

    def is_safe(row: int, col: int) -> bool:
        """row 行 col 列にクイーンを置けるか判定"""
        for prev_row in range(row):
            prev_col = board[prev_row]
            # 同じ列にあるか
            if prev_col == col:
                return False
            # 同じ対角線上にあるか（距離が等しい = 対角線上）
            if abs(prev_row - row) == abs(prev_col - col):
                return False
        return True

    def backtrack(row: int):
        """row 行目にクイーンを配置する"""
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


# ---- 実行例 ----
solutions = solve_nqueens(4)
print(f"4-Queens の解の数: {len(solutions)}")
for i, sol in enumerate(solutions):
    print(f"  解{i+1}: {sol}")

# 出力:
# 4-Queens の解の数: 2
#   解1: [1, 3, 0, 2]
#   解2: [2, 0, 3, 1]

solutions_8 = solve_nqueens(8)
print(f"8-Queens の解の数: {len(solutions_8)}")
# 出力: 8-Queens の解の数: 92
```

### 2.3 集合を使った高速化

`is_safe` で毎回全ての前の行をチェックする代わりに、使用中の列と対角線を集合で管理すると O(1) で判定できる。

```python
def solve_nqueens_fast(n: int) -> list:
    """N-Queens - 集合による高速判定版"""
    solutions = []
    board = [-1] * n
    cols = set()       # 使用中の列
    diag1 = set()      # 使用中の左上-右下対角線 (row - col)
    diag2 = set()      # 使用中の右上-左下対角線 (row + col)

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

            # 選択を適用
            board[row] = col
            cols.add(col)
            diag1.add(d1)
            diag2.add(d2)

            backtrack(row + 1)

            # 選択を取り消し
            board[row] = -1
            cols.remove(col)
            diag1.remove(d1)
            diag2.remove(d2)

    backtrack(0)
    return solutions


# ---- 実行例 ----
solutions = solve_nqueens_fast(8)
print(f"8-Queens の解の数: {len(solutions)}")  # 92
```

### 2.4 ビットマスクによる最適化

集合の代わりにビット演算を用いると、定数倍の高速化が実現できる。各ビットが盤面の列に対応する。

```python
def count_nqueens_bitmask(n: int) -> int:
    """
    N-Queens の解の数をビットマスクで高速に数える

    cols:  使用中の列を表すビットマスク
    diag1: 左上→右下対角線（左にシフト）
    diag2: 右上→左下対角線（右にシフト）

    ビット操作の仕組み:
      available & (-available) → 最下位の立っているビットを取得
      available ^= pos         → そのビットを消す
    """
    count = 0
    all_cols = (1 << n) - 1  # n ビット全てが 1 のマスク

    def backtrack(row: int, cols: int, diag1: int, diag2: int):
        nonlocal count
        if row == n:
            count += 1
            return

        # 配置可能な列を計算
        available = all_cols & ~(cols | diag1 | diag2)

        while available:
            pos = available & (-available)  # 最下位ビットを取得
            available ^= pos               # そのビットを消す
            backtrack(
                row + 1,
                cols | pos,
                (diag1 | pos) << 1,  # 対角線は次の行で1列ずれる
                (diag2 | pos) >> 1
            )

    backtrack(0, 0, 0, 0)
    return count


# ---- 実行例 ----
for n in range(1, 13):
    print(f"N={n:2d}: {count_nqueens_bitmask(n)} 解")

# 出力:
# N= 1: 1 解
# N= 2: 0 解
# N= 3: 0 解
# N= 4: 2 解
# N= 5: 10 解
# N= 6: 4 解
# N= 7: 40 解
# N= 8: 92 解
# N= 9: 352 解
# N=10: 724 解
# N=11: 2680 解
# N=12: 14200 解
```

```
ビットマスクの動作例（N=4, row=0 で col=1 にクイーンを配置）:

  row=0:  cols=0010  diag1=0010  diag2=0010
                      ↓ <<1       ↓ >>1
  row=1:  cols=0010  diag1=0100  diag2=0001
          blocked = cols | diag1 | diag2 = 0111
          available = 1111 & ~0111 = 1000
          → col=3 のみ配置可能

  盤面:
  row 0:  . Q . .    (cols に 0010)
  row 1:  . . . Q    (col=3 のみ可能)
```

### 2.5 解の可視化

```python
def print_board(solution: list) -> None:
    """N-Queens の盤面をASCIIアートで表示"""
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


# ---- 実行例 ----
solutions = solve_nqueens_fast(4)
for i, sol in enumerate(solutions):
    print(f"解 {i + 1}:")
    print_board(sol)

# 出力:
# 解 1:
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

## 3. 数独ソルバー

### 3.1 問題定義と制約

9x9 の数独を、行・列・3x3 ボックスの制約を全て満たすように埋める。各行、各列、各 3x3 ボックスに 1-9 の数字が一つずつ入る。

```
入力:                         出力:
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

制約の3重チェック:

  行の制約          列の制約         ボックスの制約
  → → → → →       ↓               +-------+
  5 3 4 6 7 8 9 1 2 5              | 5 3 4 |
  (各行に1-9が       6              | 6 7 2 |
   1つずつ)          1              | 1 9 8 |
                     ...            +-------+
                    (各列に1-9が    (各3x3に1-9が
                     1つずつ)        1つずつ)
```

### 3.2 基本実装

空きマスを左上から順に埋めていく素朴なアプローチ。

```python
def solve_sudoku(board: list) -> bool:
    """
    数独ソルバー - 基本的なバックトラッキング

    Parameters:
        board: 9x9 の2次元リスト。空きマスは 0

    Returns:
        解が見つかったら True（board は解で上書きされる）
    """

    def find_empty() -> tuple:
        """左上から順に空のセルを見つける"""
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0:
                    return r, c
        return None

    def is_valid(row: int, col: int, num: int) -> bool:
        """num を (row, col) に置けるか"""
        # 行チェック
        if num in board[row]:
            return False

        # 列チェック
        for r in range(9):
            if board[r][col] == num:
                return False

        # 3x3 ボックスチェック
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
```

### 3.3 候補集合による高速化

候補数字を集合で管理することで、`is_valid` の判定を O(1) に高速化する。

```python
def solve_sudoku_fast(board: list) -> bool:
    """
    数独ソルバー - 候補集合による高速版

    行・列・ボックスごとに使用中の数字を集合で管理し、
    O(1) で配置可能性を判定する。
    """
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]
    empty_cells = []

    # 初期状態の構築
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


# ---- 実行例 ----
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
print("数独の解:")
for row in board:
    print(row)

# 出力:
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

### 3.4 MRV ヒューリスティックによるさらなる高速化

**MRV（Minimum Remaining Values）** は、候補数が最も少ないマスから先に埋めるヒューリスティックである。選択肢が少ないマスほど早く行き詰まり、無駄な探索を回避できる。

```python
def solve_sudoku_mrv(board: list) -> bool:
    """
    数独ソルバー - MRV ヒューリスティック版

    各ステップで候補数が最小のマスを選んで埋めることで、
    探索空間を劇的に削減する。
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
        """(r, c) に配置可能な数字の集合"""
        return set(range(1, 10)) - rows[r] - cols[c] - boxes[3 * (r // 3) + c // 3]

    def backtrack() -> bool:
        if not empty_cells:
            return True

        # MRV: 候補数が最小のセルを選択
        min_cell = min(empty_cells, key=lambda rc: len(get_candidates(*rc)))
        r, c = min_cell
        candidates = get_candidates(r, c)

        if not candidates:
            return False  # 候補なし → バックトラック

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

### 3.5 数独の難易度と探索ノード数

```
数独の難易度と探索の関係:

  難易度     空きマス数    探索ノード数（目安）     候補集合の効果
  ─────────────────────────────────────────────────────────
  簡単       30-35個      100 以下                  大（多くが一意決定）
  普通       40-50個      1,000 以下                中
  難しい     50-55個      10,000 以下               小（分岐が多い）
  極難       55-60個      100,000 以上              MRV が重要

  ※ 理論上の最大探索空間は 9^81 ≒ 1.97 x 10^77 だが、
    適切な枝刈りにより数百〜数万ノードに収まる
```

---

## 4. 順列・組み合わせ・部分集合

### 4.1 順列の生成

n 個の要素を全ての順番で並べる。

```python
def permutations(nums: list) -> list:
    """
    全順列の生成

    計算量: O(n * n!)
    空間量: O(n) （再帰スタック）+ O(n * n!)（結果格納）
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


# ---- 実行例 ----
print(permutations([1, 2, 3]))
# [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]
```

**重複要素がある場合の順列:**

```python
def permutations_unique(nums: list) -> list:
    """
    重複要素を含む順列の生成（重複排除）

    戦略: ソートして、同じ値の要素が前の要素を使い切る前に
    使われないようにする
    """
    result = []
    nums.sort()
    used = [False] * len(nums)

    def backtrack(path: list):
        if len(path) == len(nums):
            result.append(path[:])
            return

        for i in range(len(nums)):
            # 既に使用済み
            if used[i]:
                continue
            # 重複スキップ: 同じ値の前の要素が未使用なら飛ばす
            if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:
                continue

            used[i] = True
            path.append(nums[i])
            backtrack(path)
            path.pop()
            used[i] = False

    backtrack([])
    return result


# ---- 実行例 ----
print(permutations_unique([1, 1, 2]))
# [[1, 1, 2], [1, 2, 1], [2, 1, 1]]  -- 重複なし（3!/2! = 3通り）
```

### 4.2 組み合わせの生成

n 個の中から k 個を選ぶ（順序を問わない）。

```python
def combinations(nums: list, k: int) -> list:
    """
    nCk の全組み合わせを生成

    計算量: O(nCk)
    枝刈り: 残り要素が足りない場合は探索を打ち切る
    """
    result = []

    def backtrack(start: int, path: list):
        if len(path) == k:
            result.append(path[:])
            return

        # 枝刈り: 残り要素数が不足していたらスキップ
        remaining_needed = k - len(path)
        for i in range(start, len(nums) - remaining_needed + 1):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    backtrack(0, [])
    return result


# ---- 実行例 ----
print(combinations([1, 2, 3, 4], 2))
# [[1,2], [1,3], [1,4], [2,3], [2,4], [3,4]]

print(f"C(4,2) = {len(combinations([1,2,3,4], 2))}")
# C(4,2) = 6
```

### 4.3 部分集合の列挙

```python
def subsets(nums: list) -> list:
    """
    全部分集合（べき集合）の生成

    計算量: O(n * 2^n)
    特徴: 全ての探索時点で解として記録する
    """
    result = []

    def backtrack(start: int, path: list):
        result.append(path[:])  # 全ての部分解が答え
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    backtrack(0, [])
    return result


# ---- 実行例 ----
print(subsets([1, 2, 3]))
# [[], [1], [1,2], [1,2,3], [1,3], [2], [2,3], [3]]
```

### 4.4 列挙パターンの比較

```
順列・組み合わせ・部分集合の状態空間木の違い:

順列 {1,2,3}:                 解の条件: path の長さが n
              {}
           /  |  \
          1   2   3           ← 全要素から選択
         / \ / \ / \
       12 13 21 23 31 32      ← 使用済みを除く全要素
       |  |  |  |  |  |
      123 132 213 231 312 321 ← 解（葉ノード全て）
      解の数: 3! = 6


組み合わせ C(4,2):            解の条件: path の長さが k
              {}
          / | \  \
         1  2  3  4           ← start 以降から選択
        /|\ /\  |
      12 13 14 23 24 34       ← 解（長さ k に達した時点）
      解の数: C(4,2) = 6


部分集合 {1,2,3}:             解の条件: 全てのノードが解
              {}              ← 解
           /  |  \
          1   2   3           ← 解 x3
         / \  |
       12  13 23              ← 解 x3
       |
      123                    ← 解
      解の数: 2^3 = 8
```

| 列挙パターン | 解の条件 | 選択肢の制約 | 計算量 | 解の数 |
|:---|:---|:---|:---|:---|
| 順列 | path長 = n | 未使用の全要素 | O(n * n!) | n! |
| 重複順列 | path長 = n | 全要素（重複可） | O(n^n) | n^n |
| 組み合わせ | path長 = k | start以降の要素 | O(nCk) | nCk |
| 部分集合 | 常に解 | start以降の要素 | O(n * 2^n) | 2^n |
| 重複組み合わせ | path長 = k | 自身以降の要素 | O(n+k-1 C k) | (n+k-1)Ck |

---

## 5. 応用問題: グラフ彩色

### 5.1 問題定義

グラフ彩色問題（Graph Coloring）は、グラフの各頂点に色を割り当て、隣接する頂点が同じ色にならないようにする問題である。使用する色の数を最小化する問題は NP 困難だが、与えられた色数 m で彩色可能かどうかを判定する問題はバックトラッキングで解ける。

```
グラフ彩色の例（3色で彩色可能か？）:

  入力グラフ:              彩色結果:
      0 --- 1                R --- G
      |   / |                |   / |
      |  /  |                |  /  |
      | /   |                | /   |
      2 --- 3                B --- R

  隣接リスト:              色の割り当て:
  0: [1, 2]                0: 赤(R)
  1: [0, 2, 3]             1: 緑(G)
  2: [0, 1, 3]             2: 青(B)
  3: [1, 2]                3: 赤(R)
```

### 5.2 実装

```python
def graph_coloring(adj: dict, m: int) -> list:
    """
    グラフ彩色問題をバックトラッキングで解く

    Parameters:
        adj: 隣接リスト {頂点: [隣接頂点のリスト]}
        m  : 使用可能な色の数

    Returns:
        彩色が可能なら色の割り当てリスト、不可能なら None
    """
    vertices = sorted(adj.keys())
    n = len(vertices)
    colors = [0] * n  # 0 = 未彩色、1..m = 色

    def is_safe(vertex_idx: int, color: int) -> bool:
        """vertex_idx に color を割り当てられるか"""
        vertex = vertices[vertex_idx]
        for neighbor in adj[vertex]:
            neighbor_idx = vertices.index(neighbor)
            if colors[neighbor_idx] == color:
                return False
        return True

    def backtrack(vertex_idx: int) -> bool:
        if vertex_idx == n:
            return True  # 全頂点に色を割り当て完了

        for color in range(1, m + 1):
            if is_safe(vertex_idx, color):
                colors[vertex_idx] = color

                if backtrack(vertex_idx + 1):
                    return True

                colors[vertex_idx] = 0  # バックトラック

        return False

    if backtrack(0):
        return {vertices[i]: colors[i] for i in range(n)}
    return None


# ---- 実行例 ----
graph = {
    0: [1, 2],
    1: [0, 2, 3],
    2: [0, 1, 3],
    3: [1, 2],
}

# 2色で彩色を試みる
result = graph_coloring(graph, 2)
print(f"2色: {result}")
# 出力: 2色: None（不可能）

# 3色で彩色を試みる
result = graph_coloring(graph, 3)
print(f"3色: {result}")
# 出力: 3色: {0: 1, 1: 2, 2: 3, 3: 1}

color_names = {1: "赤", 2: "緑", 3: "青"}
if result:
    for vertex, color in result.items():
        print(f"  頂点 {vertex}: {color_names[color]}")
```

### 5.3 グラフ彩色のバリエーション

| バリエーション | 説明 | 応用例 |
|:---|:---|:---|
| 頂点彩色 | 隣接頂点が同色でない | 地図の塗り分け、時間割作成 |
| 辺彩色 | 共通端点の辺が同色でない | ネットワーク周波数割り当て |
| リスト彩色 | 頂点ごとに使用可能色が異なる | レジスタ割り当て |
| 彩色数の最小化 | 最少の色数を求める | コンパイラ最適化 |

---

## 6. 応用問題: ナイトツアーと括弧生成

### 6.1 ナイトツアー

チェスのナイトが全てのマスをちょうど1回ずつ訪問する経路を見つける問題。

```
ナイトの移動パターン:          5x5 盤面でのナイトツアーの解（一例）:

    . 2 . 1 .                  1 14  9 20  3
    3 . . . 8                  24 19  2 15 10
    . . N . .                   13  8 23  4 21
    4 . . . 7                   18 25  6 11 16
    . 5 . 6 .                    7 12 17 22  5

  N から 1-8 の位置に
  移動可能（L字型）
```

```python
def solve_knight_tour(n: int, start_row: int = 0, start_col: int = 0) -> list:
    """
    ナイトツアー問題を解く

    Parameters:
        n: 盤面のサイズ
        start_row, start_col: 開始位置

    Returns:
        解が見つかったら盤面（訪問順の2次元リスト）、なければ None
    """
    board = [[-1] * n for _ in range(n)]

    # ナイトの8方向の移動
    moves = [
        (-2, -1), (-2, 1), (-1, -2), (-1, 2),
        (1, -2), (1, 2), (2, -1), (2, 1)
    ]

    def count_onward_moves(r: int, c: int) -> int:
        """(r, c) から移動可能なマス数（Warnsdorff's Rule 用）"""
        count = 0
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and board[nr][nc] == -1:
                count += 1
        return count

    def backtrack(r: int, c: int, move_count: int) -> bool:
        board[r][c] = move_count

        if move_count == n * n - 1:
            return True  # 全マス訪問完了

        # Warnsdorff's Rule: 移動先の選択肢が少ない方を優先
        next_moves = []
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and board[nr][nc] == -1:
                next_moves.append((count_onward_moves(nr, nc), nr, nc))

        next_moves.sort()  # 選択肢が少ない順にソート

        for _, nr, nc in next_moves:
            if backtrack(nr, nc, move_count + 1):
                return True

        board[r][c] = -1  # バックトラック
        return False

    if backtrack(start_row, start_col, 0):
        return board
    return None


# ---- 実行例 ----
n = 6
result = solve_knight_tour(n)
if result:
    print(f"{n}x{n} ナイトツアーの解:")
    for row in result:
        print(" ".join(f"{x:3d}" for x in row))

# 出力例（6x6）:
#   0 15 22  3 14 25
#  23  4  1 26  9 20
#  16 31 24 21  2 13
#   5 28 33 10 19  8
#  32 17 30 35 12 27
#  29  6 11 34  7 18
```

### 6.2 括弧の生成

n 組の正しく対応する括弧の全パターンを生成する。

```
n=3 の場合の全パターン:

  ((()))    (()())    (())()    ()(())    ()()()

  状態空間木の一部（枝刈りあり）:

                    ""
                  /    \
                "("    X (")" は最初に来れない)
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
    n 組の正しく対応する括弧を全て生成

    Parameters:
        n: 括弧の組数

    Returns:
        有効な括弧文字列のリスト

    枝刈り条件:
      - 開き括弧は n 個まで使用可能
      - 閉じ括弧は開き括弧の数以下
    """
    result = []

    def backtrack(s: str, open_count: int, close_count: int):
        if len(s) == 2 * n:
            result.append(s)
            return

        # 開き括弧をまだ追加できる
        if open_count < n:
            backtrack(s + "(", open_count + 1, close_count)

        # 閉じ括弧を追加できる（開き括弧より少ない場合）
        if close_count < open_count:
            backtrack(s + ")", open_count, close_count + 1)

    backtrack("", 0, 0)
    return result


# ---- 実行例 ----
for n in range(1, 5):
    parens = generate_parentheses(n)
    print(f"n={n}: {len(parens)} パターン")
    if n <= 3:
        for p in parens:
            print(f"  {p}")

# 出力:
# n=1: 1 パターン
#   ()
# n=2: 2 パターン
#   (())
#   ()()
# n=3: 5 パターン
#   ((()))
#   (()())
#   (())()
#   ()(())
#   ()()()
# n=4: 14 パターン
```

括弧の数はカタラン数 C_n = (2n)! / ((n+1)! * n!) で与えられる。

### 6.3 組み合わせ和（Combination Sum）

目標値 target に合計がちょうど等しくなる組み合わせを見つける。

```python
def combination_sum(candidates: list, target: int) -> list:
    """
    candidates の要素を重複使用して target を作る全組み合わせ

    Parameters:
        candidates: 正の整数のリスト（重複なし）
        target: 目標合計値

    Returns:
        合計が target になる組み合わせのリスト
    """
    result = []
    candidates.sort()

    def backtrack(start: int, path: list, remaining: int):
        if remaining == 0:
            result.append(path[:])
            return

        for i in range(start, len(candidates)):
            # 枝刈り: 候補が残りを超えたら以降の候補も全て超える
            if candidates[i] > remaining:
                break

            path.append(candidates[i])
            # 同じ要素を再利用可能なので start = i
            backtrack(i, path, remaining - candidates[i])
            path.pop()

    backtrack(0, [], target)
    return result


# ---- 実行例 ----
print(combination_sum([2, 3, 6, 7], 7))
# [[2, 2, 3], [7]]

print(combination_sum([2, 3, 5], 8))
# [[2, 2, 2, 2], [2, 3, 3], [3, 5]]
```

---

## 7. 状態空間木の構造と枝刈り戦略

### 7.1 状態空間木の定量分析

```
N-Queens の状態空間木ノード数比較（N=8）:

  枝刈りなし（全列を試す）:
    ノード数 ≒ 8^8 = 16,777,216

  列の重複排除のみ:
    ノード数 ≒ 8! = 40,320

  列 + 対角線の枝刈り:
    ノード数 ≒ 2,057 （大幅に削減）

  ビットマスク最適化:
    ノード数は同じだが、各ノードの処理が O(1)

  ┌──────────────────────────────────────────────┐
  │ 枝刈りの段階別ノード数（N=8）                │
  │                                              │
  │ 16,777,216  ████████████████████████ 全探索  │
  │     40,320  █                       列制約   │
  │      2,057  ▏                       列+対角  │
  │                                              │
  │ → 適切な枝刈りで 99.99% のノードを削減      │
  └──────────────────────────────────────────────┘
```

### 7.2 枝刈り戦略の体系

| 戦略カテゴリ | 戦略名 | 説明 | 適用例 | 削減効果 |
|:---|:---|:---|:---|:---|
| **制約伝播** | 制約チェック | 現在の選択が制約を満たすか判定 | N-Queens（列/対角線） | 高 |
| **制約伝播** | 前方チェック | 未割当変数の候補を事前に絞る | 数独（候補集合） | 高 |
| **制約伝播** | アーク整合性 | 2変数間の整合性を維持 | CSP 全般 | 非常に高 |
| **変数順序** | MRV | 候補数最小の変数から選択 | 数独 | 非常に高 |
| **変数順序** | 次数ヒューリスティック | 制約数最大の変数から選択 | グラフ彩色 | 高 |
| **値順序** | LCV | 他変数の選択肢を最も減らさない値 | CSP 全般 | 中 |
| **対称性** | 対称性除去 | 回転・鏡像などの等価解を排除 | N-Queens | 中 |
| **限定** | 分枝限定法 | 上界/下界で探索範囲を削減 | ナップサック、TSP | 高 |
| **順序** | 順序制約 | 昇順等の制約で重複排除 | 組み合わせ列挙 | 高 |

### 7.3 枝刈りの実装パターン

```python
# ---- パターン1: フィルタリング型枝刈り ----
# 制約を満たさない選択を事前に除外
def backtrack_filter(state, all_choices, result):
    if is_solution(state):
        result.append(state.copy())
        return
    # 有効な選択肢のみをフィルタリング
    valid_choices = [c for c in all_choices if is_valid(state, c)]
    for choice in valid_choices:
        state.append(choice)
        backtrack_filter(state, all_choices, result)
        state.pop()


# ---- パターン2: 上界・下界型枝刈り（分枝限定法）----
# 現在の部分解から最良ケースを推定し、既知の最良解と比較
best_value = float('inf')

def backtrack_bound(state, choices, cost_so_far):
    global best_value
    if is_solution(state):
        best_value = min(best_value, cost_so_far)
        return

    for choice in choices:
        if is_valid(state, choice):
            # 下界推定: 現在コスト + 残りの最小コスト推定
            lower_bound = cost_so_far + estimate_remaining(state, choice)
            if lower_bound >= best_value:
                continue  # この分岐は最良解を改善できない

            apply(state, choice)
            backtrack_bound(state, next_choices(choice),
                          cost_so_far + cost(choice))
            undo(state, choice)


# ---- パターン3: 対称性除去型枝刈り ----
# N-Queens で最初の行の配置を半分に制限
def nqueens_symmetry(n):
    """対称性を利用して探索空間を半減"""
    solutions = []

    def backtrack(row, board, cols, diag1, diag2):
        if row == n:
            solutions.append(board[:])
            return

        # 最初の行は左半分のみ（対称性で右半分は鏡像）
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

## 8. 計算量分析と最適化テクニック

### 8.1 バックトラッキングの計算量一覧

| 問題 | 枝刈りなし | 枝刈りあり | 解の数 | 備考 |
|:---|:---|:---|:---|:---|
| N-Queens | O(N^N) | O(N!) 程度 | N=8: 92 | 対称性除去で半減可能 |
| 数独 | O(9^81) | 数百〜数万ノード | 1（通常） | MRV で劇的に高速化 |
| 全順列 | O(N * N!) | O(N!) | N! | 最適（枝刈り不要） |
| 全組み合わせ | O(2^N) | O(NCk) | NCk | 順序制約で重複排除 |
| 全部分集合 | O(N * 2^N) | O(2^N) | 2^N | 最適（枝刈り不要） |
| グラフ彩色 | O(m^V) | 問題依存 | 問題依存 | MRV + LCV が有効 |
| ナイトツアー | O(8^(N^2)) | Warnsdorff で準線形 | 問題依存 | ヒューリスティック必須 |
| 括弧生成 | O(4^n) | O(C_n) | カタラン数 | 枝刈りが非常に有効 |
| 組み合わせ和 | 指数的 | 問題依存 | 問題依存 | ソート + 上界枝刈り |

### 8.2 最適化テクニック集

```
最適化テクニックの効果比較:

  テクニック                     効果の大きさ    実装の複雑さ
  ────────────────────────────────────────────────────────
  適切な枝刈り条件の設計         ★★★★★          ★★★
  変数順序の最適化（MRV）        ★★★★★          ★★
  データ構造の選択（集合/ビット） ★★★★            ★★
  対称性の排除                   ★★★              ★★★★
  値順序の最適化（LCV）          ★★★              ★★★
  メモ化（重複状態の回避）       ★★★★            ★★
  反復深化（深さ制限付き）       ★★                ★
  並列化                         ★★★              ★★★★★
```

### 8.3 メモ化との組み合わせ

バックトラッキングに**メモ化（Memoization）** を組み合わせると、同じ状態を再計算することを避けられる。これは動的計画法（DP）への橋渡しでもある。

```python
def can_partition(nums: list) -> bool:
    """
    配列を和が等しい2つの部分集合に分割可能か判定

    バックトラッキング + メモ化 の例
    （純粋なバックトラッキングでは TLE になるケースを高速化）
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

        # nums[idx] を選ぶ or 選ばない
        result = (backtrack(idx + 1, remaining - nums[idx]) or
                  backtrack(idx + 1, remaining))

        memo[key] = result
        return result

    return backtrack(0, target)


# ---- 実行例 ----
print(can_partition([1, 5, 11, 5]))   # True: [1,5,5] と [11]
print(can_partition([1, 2, 3, 5]))    # False
print(can_partition([3, 3, 3, 4, 5])) # True: [3,3,3] と [4,5]
```

```
バックトラッキングからDPへの段階的変換:

  Step 1: 素朴なバックトラッキング
    → 全ての分岐を再帰的に探索
    → 同じ状態を何度も再計算する可能性

  Step 2: バックトラッキング + メモ化（トップダウンDP）
    → 計算済みの状態をキャッシュ
    → 同じ状態は O(1) で返す

  Step 3: ボトムアップ DP（テーブル）
    → 再帰を排除し、テーブルを順に埋める
    → スタックオーバーフローのリスクなし

  変換が可能な条件:
    - 状態が有限個のパラメータで一意に決まる
    - 部分問題に重複がある（overlapping subproblems）
    - 最適部分構造がある（optimal substructure）
```

### 8.4 反復深化によるメモリ最適化

深さ制限付きのバックトラッキングを、深さを1ずつ増やしながら繰り返す手法。最適解の深さが浅い場合に有効で、メモリ使用量を O(bd) から O(d) に削減できる（b: 分岐因子、d: 深さ）。

```python
def iterative_deepening_backtrack(initial_state, max_depth: int):
    """
    反復深化バックトラッキング

    Parameters:
        initial_state: 初期状態
        max_depth: 探索する最大深さ

    Returns:
        解が見つかったら状態、なければ None
    """
    def depth_limited_search(state, depth_limit: int) -> bool:
        if is_solution(state):
            return True
        if depth_limit <= 0:
            return False  # 深さ制限に到達

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

## 9. アンチパターン

### 9.1 アンチパターン1: バックトラックの取り消し忘れ

バックトラッキングで最も頻繁に発生するバグ。状態を変更した後に元に戻さないと、後続の探索で不正な状態が伝播し、解が見つからなかったり、不正な解が含まれたりする。

```python
# ============================================================
# BAD: 選択の取り消しを忘れる → 状態が汚染される
# ============================================================
def bad_nqueens(board, row, n, solutions):
    if row == n:
        solutions.append(board[:])
        return
    for col in range(n):
        if is_safe(board, row, col):
            board[row] = col
            bad_nqueens(board, row + 1, n, solutions)
            # board[row] = -1 を忘れている!
            # → N=4 で本来2解なのに、1解しか見つからない、
            #   または不正な解が混入する


# ============================================================
# GOOD: 必ず取り消す
# ============================================================
def good_nqueens(board, row, n, solutions):
    if row == n:
        solutions.append(board[:])
        return
    for col in range(n):
        if is_safe(board, row, col):
            board[row] = col
            good_nqueens(board, row + 1, n, solutions)
            board[row] = -1  # 必ず取り消す!
```

**取り消し忘れを防ぐ技法:**

1. **イミュータブルな状態を渡す**: 状態のコピーを渡すことで取り消し不要にする（ただしメモリ使用量が増加）
2. **with 文パターン**: Python のコンテキストマネージャで自動取り消しを保証する
3. **タプルの連結**: `path + (choice,)` のように新しいタプルを作る（コピーと同等）

```python
# イミュータブル版: 取り消し忘れが原理的に発生しない
def safe_backtrack(path: tuple, remaining: tuple, result: list):
    """イミュータブルなデータで安全にバックトラッキング"""
    if not remaining:
        result.append(list(path))
        return
    for i in range(len(remaining)):
        # 新しいタプルを作るので元の path は変更されない
        safe_backtrack(
            path + (remaining[i],),
            remaining[:i] + remaining[i + 1:],
            result
        )

result = []
safe_backtrack((), (1, 2, 3), result)
print(result)  # [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]
```

### 9.2 アンチパターン2: 枝刈りなしの全探索

制約チェックを解の完成時にのみ行うと、無効な分岐を大量に探索してしまう。

```python
# ============================================================
# BAD: 完成後にまとめてチェック → 指数的に遅い
# ============================================================
def bad_nqueens_late_check(n):
    """全配置を生成してから検証する（非常に遅い）"""
    solutions = []

    def generate(row, board):
        if row == n:
            # ここで初めて全制約をチェック
            if all_queens_safe(board, n):
                solutions.append(board[:])
            return
        for col in range(n):
            board[row] = col
            generate(row + 1, board)  # 全分岐を探索!

    generate(0, [-1] * n)
    return solutions
    # N=8: 16,777,216 ノードを探索（8^8）


# ============================================================
# GOOD: 各ステップで逐次チェック → 高速
# ============================================================
def good_nqueens_early_check(n):
    """各行で制約チェックして早期枝刈り"""
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
    # N=8: 約 2,057 ノードを探索（99.99% 削減）
```

### 9.3 アンチパターン3: 解のコピーを忘れる

```python
# ============================================================
# BAD: 参照をそのまま保存 → 全て同じリストを指す
# ============================================================
def bad_subsets(nums):
    result = []
    path = []

    def backtrack(start):
        result.append(path)  # 参照を追加 → 全エントリが同一オブジェクト!
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1)
            path.pop()

    backtrack(0)
    return result
    # 結果: [[], [], [], [], [], [], [], []]（全て空リスト）


# ============================================================
# GOOD: コピーを保存
# ============================================================
def good_subsets(nums):
    result = []
    path = []

    def backtrack(start):
        result.append(path[:])  # コピーを追加!
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1)
            path.pop()

    backtrack(0)
    return result
    # 結果: [[], [1], [1,2], [1,2,3], [1,3], [2], [2,3], [3]]
```

### 9.4 アンチパターン4: 不要な探索の継続

一つの解が必要なのに全解を探索してしまう。

```python
# ============================================================
# BAD: 1つ見つかれば十分なのに全解を探索
# ============================================================
def bad_sudoku(board):
    """全解を探索（数独は通常1つの解で十分）"""
    solutions = []

    def backtrack(idx):
        if idx == len(empty_cells):
            solutions.append([row[:] for row in board])
            return  # return True しない → 探索が続行

        r, c = empty_cells[idx]
        for num in range(1, 10):
            if is_valid(r, c, num):
                board[r][c] = num
                backtrack(idx + 1)
                board[r][c] = 0

    backtrack(0)
    return solutions  # 全解のリスト（数独では通常1つ）


# ============================================================
# GOOD: 最初の解で探索を打ち切り
# ============================================================
def good_sudoku(board):
    """最初の解で即座に打ち切り"""
    def backtrack(idx):
        if idx == len(empty_cells):
            return True  # 解発見 → 即座に True を返す

        r, c = empty_cells[idx]
        for num in range(1, 10):
            if is_valid(r, c, num):
                board[r][c] = num
                if backtrack(idx + 1):
                    return True  # 上位に伝播して全探索を停止
                board[r][c] = 0

        return False

    return backtrack(0)
```

---

## 10. 演習問題

### 10.1 基礎レベル

**問題1: 文字列の全順列**

与えられた文字列の全順列を辞書順で生成せよ。重複文字がある場合は重複を排除すること。

```python
def string_permutations(s: str) -> list:
    """
    文字列の全順列を辞書順で返す

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


# テスト
assert string_permutations("abc") == ['abc', 'acb', 'bac', 'bca', 'cab', 'cba']
assert string_permutations("aba") == ['aab', 'aba', 'baa']
assert string_permutations("a") == ['a']
print("問題1: 全テスト通過")
```

**問題2: べき集合のフィルタリング**

整数リストの部分集合のうち、合計が指定値以下のものだけを返せ。

```python
def filtered_subsets(nums: list, max_sum: int) -> list:
    """
    合計が max_sum 以下の部分集合を全て返す

    >>> filtered_subsets([1, 2, 3], 3)
    [[], [1], [1, 2], [2], [3]]
    """
    result = []

    def backtrack(start: int, path: list, current_sum: int):
        result.append(path[:])

        for i in range(start, len(nums)):
            # 枝刈り: 合計が max_sum を超えるならスキップ
            if current_sum + nums[i] > max_sum:
                continue
            path.append(nums[i])
            backtrack(i + 1, path, current_sum + nums[i])
            path.pop()

    nums.sort()
    backtrack(0, [], 0)
    return result


# テスト
result = filtered_subsets([1, 2, 3], 3)
assert [] in result
assert [1] in result
assert [1, 2] in result
assert [2] in result
assert [3] in result
assert [1, 2, 3] not in result  # 合計6 > 3
print("問題2: 全テスト通過")
```

### 10.2 応用レベル

**問題3: 単語探索（Word Search）**

2次元グリッドの中から、隣接セルを辿って指定の単語を構成できるか判定せよ。同じセルは1回のみ使用可能。

```python
def word_search(board: list, word: str) -> bool:
    """
    2次元グリッドで単語を探す

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

        # セルを使用済みとしてマーク
        original = board[r][c]
        board[r][c] = '#'

        for dr, dc in directions:
            if backtrack(r + dr, c + dc, idx + 1):
                return True

        # バックトラック
        board[r][c] = original
        return False

    for r in range(rows):
        for c in range(cols):
            if backtrack(r, c, 0):
                return True
    return False


# テスト
board = [
    ['A', 'B', 'C', 'E'],
    ['S', 'F', 'C', 'S'],
    ['A', 'D', 'E', 'E']
]
assert word_search([row[:] for row in board], "ABCCED") is True
assert word_search([row[:] for row in board], "SEE") is True
assert word_search([row[:] for row in board], "ABCB") is False
print("問題3: 全テスト通過")
```

**問題4: 分割回文列挙**

文字列を、全てのパーツが回文になるように分割する全パターンを求めよ。

```python
def palindrome_partition(s: str) -> list:
    """
    文字列を回文のパーツに分割する全パターンを返す

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


# テスト
assert palindrome_partition("aab") == [['a', 'a', 'b'], ['aa', 'b']]
assert palindrome_partition("a") == [['a']]
assert palindrome_partition("aba") == [['a', 'b', 'a'], ['aba']]
print("問題4: 全テスト通過")
```

### 10.3 発展レベル

**問題5: 数独の難問**

以下の難易度の高い数独をMRVヒューリスティック付きソルバーで解け。

```python
def solve_hard_sudoku():
    """
    世界最難数独の一つ（Arto Inkala 作）を解く

    空きマスが多く、通常のバックトラッキングでは
    多くのノードを探索する必要がある。
    MRV ヒューリスティックの効果を確認する。
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

    # solve_sudoku_mrv を使って解く（セクション3.4参照）
    solve_sudoku_mrv(hard_board)

    # 検証
    for r in range(9):
        assert sorted(hard_board[r]) == list(range(1, 10)), f"行 {r} が不正"
    for c in range(9):
        col_vals = [hard_board[r][c] for r in range(9)]
        assert sorted(col_vals) == list(range(1, 10)), f"列 {c} が不正"

    print("難問数独の解:")
    for row in hard_board:
        print(" ".join(str(x) for x in row))

solve_hard_sudoku()
# 出力:
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

**問題6: N-Queens の対称性解析**

N-Queens の全解から、回転・反転で等価な解を除いた「本質的に異なる解」の数を数えよ。

```python
def count_unique_nqueens(n: int) -> int:
    """
    N-Queens の本質的に異なる解の数を数える

    8つの対称操作（4回転 x 2反転）で等価な解をグループ化し、
    代表元のみを数える。
    """
    all_solutions = solve_nqueens_fast(n)  # セクション2.3参照

    def rotate_90(sol):
        """90度時計回り回転"""
        return [sol.index(n - 1 - i) for i in range(n)]

    def reflect(sol):
        """左右反転"""
        return [n - 1 - col for col in sol]

    def canonical(sol):
        """8つの対称操作の中で辞書順最小の形を返す"""
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


# テスト
for n in range(1, 11):
    total = len(solve_nqueens_fast(n))
    unique = count_unique_nqueens(n)
    print(f"N={n:2d}: 全解={total:5d}, 本質的に異なる解={unique:4d}")

# 出力:
# N= 1: 全解=    1, 本質的に異なる解=   1
# N= 2: 全解=    0, 本質的に異なる解=   0
# N= 3: 全解=    0, 本質的に異なる解=   0
# N= 4: 全解=    2, 本質的に異なる解=   1
# N= 5: 全解=   10, 本質的に異なる解=   2
# N= 6: 全解=    4, 本質的に異なる解=   1
# N= 7: 全解=   40, 本質的に異なる解=   6
# N= 8: 全解=   92, 本質的に異なる解=  12
# N= 9: 全解=  352, 本質的に異なる解=  46
# N=10: 全解=  724, 本質的に異なる解=  92
```

---

## 11. FAQ

### Q1: バックトラッキングと DFS の違いは何か？

**A:** DFS（深さ優先探索）は、グラフや木を走査するための具体的なアルゴリズムである。一方、バックトラッキングは DFS をベースにした**問題解決の設計パターン**であり、「制約を満たさない分岐を枝刈りする」という戦略が加わる。全ての DFS がバックトラッキングではないが、バックトラッキングは常に DFS の一種である。

```
         DFS（グラフ走査）              バックトラッキング
  ┌──────────────────────┐    ┌─────────────────────────────┐
  │ 全ノードを訪問する   │    │ 制約を満たす解を探す        │
  │ 枝刈りなし           │    │ 制約違反で枝刈り            │
  │ 訪問済みチェック     │    │ 選択→検証→再帰→取り消し    │
  │ グラフの構造を探索   │    │ 解の空間を探索              │
  └──────────────────────┘    └─────────────────────────────┘
```

### Q2: バックトラッキングの計算量を改善するにはどうすればよいか？

**A:** 主な改善手法は以下の5つである:

1. **強力な枝刈り条件**: 制約違反をできるだけ早い段階で検出する。N-Queens では行ごとに配置して列と対角線をチェックする。
2. **変数順序の最適化（MRV）**: 選択肢が最も少ない変数から先に処理する。数独では候補数が最小のマスから埋める。
3. **データ構造の改善**: 集合やビットマスクを使い、制約チェックを O(1) にする。
4. **対称性の排除**: 等価な解を重複して探索しない。N-Queens では最初の行を左半分に制限する。
5. **分枝限定法（Branch and Bound）**: 下界推定により、最良解を改善できない分岐を刈る。最適化問題に有効。

### Q3: 全解を求める場合と一つの解を求める場合で実装はどう変わるか？

**A:** 制御フローが異なる。

```python
# 全解を求める場合: 解を見つけてもリストに追加して探索続行
def find_all(state, result):
    if is_solution(state):
        result.append(state.copy())
        return          # ← void: 探索は上位ループで続行

    for choice in choices:
        if is_valid(state, choice):
            apply(state, choice)
            find_all(state, result)  # 戻り値を使わない
            undo(state, choice)

# 一つの解を求める場合: 見つかったら True を返して即座に打ち切り
def find_one(state):
    if is_solution(state):
        return True     # ← 即座に脱出

    for choice in choices:
        if is_valid(state, choice):
            apply(state, choice)
            if find_one(state):
                return True   # ← True を上位に伝播
            undo(state, choice)

    return False
```

### Q4: バックトラッキングと動的計画法（DP）はどう使い分けるか？

**A:** 判断基準は「部分問題の重複」と「状態空間の構造」にある。

| 特性 | バックトラッキング | 動的計画法 |
|:---|:---|:---|
| 部分問題の重複 | ない（各状態は一度だけ訪問） | ある（同じ状態を何度も計算） |
| 解の構造 | 組み合わせ的（列挙） | 最適値（最大/最小/個数） |
| メモ化の効果 | 低い（状態が再出現しない） | 高い（劇的に高速化） |
| 適する問題 | N-Queens、数独、順列列挙 | ナップサック、最長部分列 |

メモ化を追加したバックトラッキングは「トップダウン DP」と等価になる。問題に部分問題の重複があるなら DP を、なければバックトラッキングを選ぶのが基本方針。

### Q5: バックトラッキングで stack overflow を防ぐにはどうするか？

**A:** 3つの対策がある:

1. **再帰の深さ制限**: Python では `sys.setrecursionlimit()` で上限を調整する。ただし根本的解決ではない。
2. **反復深化**: 深さ制限付き探索を段階的に深くすることで、メモリ使用量を制限する（セクション8.4参照）。
3. **明示的スタックによる非再帰化**: 再帰をループとスタックで書き換える。

```python
# 非再帰版バックトラッキング（明示的スタック）
def iterative_permutations(nums):
    result = []
    # スタック要素: (path, remaining)
    stack = [([], list(nums))]

    while stack:
        path, remaining = stack.pop()
        if not remaining:
            result.append(path)
            continue
        # 逆順に追加（DFS の順序を維持するため）
        for i in range(len(remaining) - 1, -1, -1):
            new_path = path + [remaining[i]]
            new_remaining = remaining[:i] + remaining[i + 1:]
            stack.append((new_path, new_remaining))

    return result

print(iterative_permutations([1, 2, 3]))
# [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]
```

### Q6: 制約充足問題（CSP）とバックトラッキングの関係は？

**A:** バックトラッキングは CSP を解くための標準的なアルゴリズムである。CSP は「変数の集合」「各変数の定義域」「制約の集合」の3要素で定義され、全制約を満たす変数割り当てを求める問題。数独・グラフ彩色・N-Queens は全て CSP として定式化でき、バックトラッキング + 制約伝播で効率的に解ける。

---

## 12. まとめ

### 12.1 重要概念の整理

| 項目 | 要点 |
|:---|:---|
| バックトラッキングの定義 | DFS + 枝刈りによる体系的な探索手法 |
| 基本サイクル | 選択（Choose）→ 検証 → 再帰（Explore）→ 取り消し（Unchoose） |
| N-Queens | 行・列・対角線の制約。集合やビットマスクで高速化 |
| 数独 | 行・列・ボックスの制約。MRV ヒューリスティックが劇的に有効 |
| 順列・組み合わせ・部分集合 | 列挙問題の3基本パターン。start パラメータと used 配列の使い分け |
| グラフ彩色 | 隣接頂点の色制約。CSP の代表例 |
| ナイトツアー | Warnsdorff のルールで高速化。ヒューリスティックの重要性 |
| 枝刈りの効果 | 適切な枝刈りで探索空間を 99% 以上削減可能 |
| メモ化との組み合わせ | 重複状態がある場合はトップダウン DP に変換可能 |

### 12.2 バックトラッキング実装チェックリスト

```
バックトラッキング実装時の確認事項:

  [ ] 基底条件（解の完成判定）は正しいか
  [ ] 選択肢の列挙は漏れなくダブりなく行われているか
  [ ] 枝刈り条件は正しく、十分に強力か
  [ ] 状態の変更を正しく取り消しているか
  [ ] 解のコピーを保存しているか（参照ではなく）
  [ ] 全解 vs 一つの解で制御フローは適切か
  [ ] 重複要素がある場合の処理は正しいか
  [ ] 再帰の深さは十分か（stack overflow のリスク）
```

### 12.3 手法選択ガイド

```
問題のタイプ別アルゴリズム選択:

  列挙問題（全ての解を求める）
    → バックトラッキング

  最適化問題（最良の解を求める）
    → 分枝限定法（バックトラッキング + 上界/下界）
    → 部分問題の重複あり → 動的計画法

  判定問題（解の存在を確認）
    → バックトラッキング（一つの解で打ち切り）

  制約充足問題（全制約を満たす割り当て）
    → バックトラッキング + 制約伝播（AC-3 等）
    → MRV + LCV ヒューリスティック
```

---

## 次に読むべきガイド

- [グラフ走査](./02-graph-traversal.md) -- バックトラッキングの基盤となる DFS の理解
- [動的計画法](./04-dynamic-programming.md) -- バックトラッキング + メモ化から DP への移行
- [問題解決法](../04-practice/00-problem-solving.md) -- バックトラッキングの適用判断とアプローチ選択

---

## 13. 参考文献

1. Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. -- 第34章: NP 完全性と探索アルゴリズムの理論的基盤
2. Knuth, D. E. (2000). "Dancing Links." *arXiv preprint cs/0011047*. -- 正確被覆問題への応用と数独の効率的解法（Algorithm X + DLX）
3. Skiena, S. S. (2020). *The Algorithm Design Manual* (3rd ed.). Springer. -- 第9章: Combinatorial Search and Heuristic Methods。バックトラッキングの体系的解説
4. Wirth, N. (1976). *Algorithms + Data Structures = Programs*. Prentice-Hall. -- N-Queens とバックトラッキングの古典的解説
5. Russell, S. J., & Norvig, P. (2021). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson. -- 第6章: Constraint Satisfaction Problems。CSP の理論とバックトラッキングの体系的位置づけ
6. Sedgewick, R., & Wayne, K. (2011). *Algorithms* (4th ed.). Addison-Wesley. -- 部分集合列挙とバックトラッキングの実装パターン
7. LeetCode. "Backtracking Problems." https://leetcode.com/tag/backtracking/ -- バックトラッキングの練習問題集（N-Queens, Sudoku Solver, Word Search 等）
