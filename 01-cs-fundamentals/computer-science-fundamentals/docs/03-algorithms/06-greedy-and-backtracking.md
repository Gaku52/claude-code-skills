# 貪欲法とバックトラック

> 貪欲法は「今この瞬間の最善手」を選び続ける楽観的戦略。バックトラックは「全ての可能性を試し、失敗したら引き返す」慎重な戦略。

## この章で学ぶこと

- [ ] 貪欲法が最適解を与える条件を理解する
- [ ] 典型的な貪欲法アルゴリズムを実装できる
- [ ] バックトラックの仕組みと枝刈りを理解する

## 前提知識

- 計算量解析 → 参照: [[01-complexity-analysis.md]]
- 動的計画法 → 参照: [[05-dynamic-programming.md]]

---

## 1. 貪欲法（Greedy Algorithm）

### 1.1 基本概念

```
貪欲法: 各ステップで局所的に最適な選択をする

  特徴:
  - 一度選んだら変更しない（後戻りなし）
  - 高速（通常 O(n log n) 以下）
  - 最適解の保証には証明が必要

  貪欲法が最適になる2つの条件:
  1. 貪欲選択性質: 局所最適な選択が全体最適に含まれる
  2. 最適部分構造: 残りの部分問題も最適に解ける
```

### 1.2 典型的な貪欲法

```python
# 1. 活動選択問題（区間スケジューリング）
def activity_selection(activities):
    """重ならない最大数の活動を選ぶ"""
    # 終了時刻でソート
    activities.sort(key=lambda x: x[1])
    selected = [activities[0]]
    last_end = activities[0][1]

    for start, end in activities[1:]:
        if start >= last_end:
            selected.append((start, end))
            last_end = end

    return selected

# 例: [(1,4), (3,5), (0,6), (5,7), (3,9), (5,9), (6,10), (8,11)]
# → [(1,4), (5,7), (8,11)] — 3つの活動が最大

# 2. ハフマン符号（最適前置符号）
import heapq

def huffman(freq):
    """出現頻度から最適なハフマン木を構築"""
    heap = [[f, [char, ""]] for char, f in freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    return sorted(heap[0][1:], key=lambda x: (len(x[1]), x[0]))

# 3. お釣り問題（特定の硬貨セットのみ最適）
def coin_greedy(coins, amount):
    """大きい硬貨から貪欲に使う"""
    coins.sort(reverse=True)
    result = []
    for coin in coins:
        while amount >= coin:
            result.append(coin)
            amount -= coin
    return result if amount == 0 else None

# 注意: coins=[1,5,10,25] では最適
# coins=[1,3,4] で amount=6 → 貪欲: [4,1,1]=3枚 ≠ 最適: [3,3]=2枚
```

### 1.3 貪欲法 vs DP の判断

```
いつ貪欲法が使えるか:

  ✅ 貪欲法が最適:
  - 区間スケジューリング（終了時刻でソート）
  - ハフマン符号
  - クラスカル法（最小全域木）
  - ダイクストラ法（最短経路）
  - 米国の硬貨での釣り銭

  ❌ 貪欲法が最適でない（DPが必要）:
  - 0-1 ナップサック
  - コイン問題（一般の硬貨セット）
  - 最長共通部分列
  - 編集距離

  判断のヒント:
  - 「各ステップで後悔しない選択」ができるか？
  - 反例が見つからないか？
  - マトロイド構造を持つか？（数学的に厳密な判定）
```

---

## 2. バックトラック

### 2.1 基本概念

```
バックトラック: 解の候補を構築し、制約違反で引き返す

  全探索との違い:
  - 全探索: 全ての組み合わせを生成してからチェック
  - バックトラック: 構築途中で制約違反を検出→枝刈り

  探索木のイメージ:
          root
        /  |  \
       a   b   c     ← 1文字目の選択
      /|\ /|\ /|\
     a b c a b c ...  ← 2文字目の選択
     ↑     ↑
     OK    制約違反→戻る（バックトラック）
```

### 2.2 典型的なバックトラック

```python
# 1. N-Queens問題
def solve_n_queens(n):
    """N×Nのボードにクイーンを互いに攻撃しないように配置"""
    solutions = []

    def is_safe(board, row, col):
        for i in range(row):
            if board[i] == col:  # 同じ列
                return False
            if abs(board[i] - col) == abs(i - row):  # 対角線
                return False
        return True

    def backtrack(board, row):
        if row == n:
            solutions.append(board[:])
            return
        for col in range(n):
            if is_safe(board, row, col):
                board[row] = col
                backtrack(board, row + 1)
                # board[row] は次のイテレーションで上書きされるので
                # 明示的な「元に戻す」操作は不要

    backtrack([0] * n, 0)
    return solutions

# 2. 順列の生成
def permutations(nums):
    result = []
    def backtrack(path, remaining):
        if not remaining:
            result.append(path[:])
            return
        for i in range(len(remaining)):
            path.append(remaining[i])
            backtrack(path, remaining[:i] + remaining[i+1:])
            path.pop()  # バックトラック（元に戻す）
    backtrack([], nums)
    return result

# 3. 数独ソルバー
def solve_sudoku(board):
    def is_valid(board, row, col, num):
        for i in range(9):
            if board[row][i] == num: return False
            if board[i][col] == num: return False
        box_r, box_c = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_r, box_r + 3):
            for j in range(box_c, box_c + 3):
                if board[i][j] == num: return False
        return True

    def backtrack():
        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    for num in range(1, 10):
                        if is_valid(board, i, j, num):
                            board[i][j] = num
                            if backtrack():
                                return True
                            board[i][j] = 0  # バックトラック
                    return False  # どの数字も入らない
        return True  # 全マス埋まった
    backtrack()
```

### 2.3 枝刈りの技法

```
枝刈り（Pruning）: 不要な探索を事前に切り捨てる

  1. 実行可能性枝刈り: 制約違反が確定したら探索打ち切り
     → N-Queens: 同じ列/対角線にクイーンがあったら即終了

  2. 最適性枝刈り: 現時点で最適解に届かないなら打ち切り
     → 分岐限定法: 残りの最良見積もりが暫定最良解以下なら切る

  3. 対称性枝刈り: 対称な解を1つだけ探索
     → N-Queens: 最初のクイーンを上半分に限定

  枝刈りの効果:
  - N-Queens (N=8): 全探索 16,777,216通り → 枝刈り 15,720通り
  - 数独: 全探索 6.67×10²¹ → 枝刈りで瞬時
```

---

## 3. 全探索のテクニック

### 3.1 ビット全探索

```python
# ビット全探索: 2^n 通りの部分集合を列挙

def subsets(nums):
    """全ての部分集合を列挙"""
    n = len(nums)
    result = []
    for mask in range(1 << n):  # 0 から 2^n - 1
        subset = []
        for i in range(n):
            if mask & (1 << i):  # i番目のビットが立っているか
                subset.append(nums[i])
        result.append(subset)
    return result

# nums = [1, 2, 3]
# mask=000 → []
# mask=001 → [1]
# mask=010 → [2]
# mask=011 → [1, 2]
# mask=100 → [3]
# mask=101 → [1, 3]
# mask=110 → [2, 3]
# mask=111 → [1, 2, 3]

# 適用条件: n ≤ 20 程度（2^20 ≈ 100万）
```

---

## 4. 実践演習

### 演習1: 貪欲法（基礎）
分数ナップサック問題（品物を切り分けてよい場合）を貪欲法で解け。0-1ナップサックとの違いを述べよ。

### 演習2: バックトラック（応用）
与えられた数字の配列から、和が target になる全ての組み合わせを求めよ（同じ要素は1回のみ使用可能）。

### 演習3: 最適化（発展）
巡回セールスマン問題を、バックトラック+枝刈り（分岐限定法）で解くプログラムを実装し、都市数を増やした時の実行時間の変化を計測せよ。

---

## FAQ

### Q1: 貪欲法の正しさをどう証明しますか？
**A**: 3つの方法: (1)交換論法: 最適解を貪欲解に変換しても悪くならないことを示す (2)帰納法: 各ステップで最適性が維持されることを示す (3)マトロイド理論: 問題がマトロイド構造を持つことを示す

### Q2: バックトラックと動的計画法の使い分けは？
**A**: 部分問題が重複するならDP。重複がなく全パターン列挙が必要ならバックトラック。「全ての解を列挙する」問題はバックトラック。「最適値だけ求める」問題はDP向き。

### Q3: NP困難な問題に実務でどう対処しますか？
**A**: (1)近似アルゴリズム（最適解の定数倍以内を保証）(2)ヒューリスティック（焼きなまし法、遺伝的アルゴリズム）(3)問題サイズの制限（n≤20ならビット全探索）(4)特殊ケースへの帰着

---

## まとめ

| 手法 | 計算量 | 最適性 | 用途 |
|------|--------|--------|------|
| 貪欲法 | O(n log n)〜 | 条件付き最適 | 区間スケジューリング、MST |
| バックトラック | O(指数)〜 | 完全探索（枝刈りで高速化）| N-Queens、数独、組合せ列挙 |
| ビット全探索 | O(2^n × n) | 完全 | n≤20の部分集合問題 |

---

## 次に読むべきガイド
→ [[07-string-algorithms.md]] — 文字列アルゴリズム

---

## 参考文献
1. Cormen, T. H. et al. "Introduction to Algorithms." Chapters 16-17.
2. Skiena, S. S. "The Algorithm Design Manual." Chapters 8-9.
