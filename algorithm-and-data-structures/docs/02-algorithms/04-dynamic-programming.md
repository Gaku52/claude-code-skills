# 動的計画法（Dynamic Programming）

> 重複する部分問題を効率的に解くための設計手法を、メモ化・ボトムアップ・代表的問題を通じて体系的に理解する

## この章で学ぶこと

1. **メモ化（トップダウン）とボトムアップ（テーブル法）**の2つのアプローチを使い分けられる
2. **最適部分構造と重複部分問題**という DP が適用できる条件を見抜ける
3. **ナップサック問題・LCS・その他の典型 DP** を正確に実装できる

---

## 1. 動的計画法の原理

```
DP が適用可能な2条件:

1. 最適部分構造（Optimal Substructure）
   → 問題の最適解が部分問題の最適解から構成できる

2. 重複部分問題（Overlapping Subproblems）
   → 同じ部分問題が何度も現れる

   fib(5) の再帰木:
                    fib(5)
                   /      \
              fib(4)       fib(3)      ← fib(3)が重複!
             /     \       /    \
         fib(3)  fib(2) fib(2) fib(1)  ← fib(2)が重複!
        /    \
    fib(2)  fib(1)

   メモ化なし: O(2^n) → メモ化あり: O(n)
```

---

## 2. メモ化（トップダウン）

再帰 + 結果のキャッシュ。自然な再帰構造をそのまま使える。

```python
from functools import lru_cache

# 方法1: 辞書でメモ化
def fib_memo(n: int, memo: dict = None) -> int:
    if memo is None:
        memo = {}
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib_memo(n - 1, memo) + fib_memo(n - 2, memo)
    return memo[n]

# 方法2: lru_cache デコレータ（Pythonic）
@lru_cache(maxsize=None)
def fib_cached(n: int) -> int:
    if n <= 1:
        return n
    return fib_cached(n - 1) + fib_cached(n - 2)

print(fib_memo(50))    # 12586269025
print(fib_cached(50))  # 12586269025
```

---

## 3. ボトムアップ（テーブル法）

小さい部分問題から順に解いてテーブルを埋める。再帰のオーバーヘッドがない。

```python
def fib_bottom_up(n: int) -> int:
    """ボトムアップ DP - O(n) 時間、O(n) 空間"""
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]

def fib_optimized(n: int) -> int:
    """空間最適化 - O(n) 時間、O(1) 空間"""
    if n <= 1:
        return n
    prev2, prev1 = 0, 1
    for _ in range(2, n + 1):
        prev2, prev1 = prev1, prev2 + prev1
    return prev1

print(fib_bottom_up(50))   # 12586269025
print(fib_optimized(50))   # 12586269025
```

```
メモ化 vs ボトムアップ:

トップダウン（メモ化）:           ボトムアップ:
  fib(5)                           dp[0]=0
    → fib(4)                       dp[1]=1
      → fib(3)                     dp[2]=1
        → fib(2)                   dp[3]=2
          → fib(1) = 1             dp[4]=3
          → fib(0) = 0             dp[5]=5
        → 結果=1 (キャッシュ)      答え: dp[5]
      → fib(2) → キャッシュヒット!
    → fib(3) → キャッシュヒット!
  答え: 5
```

---

## 4. 0/1 ナップサック問題

重さ制限 W のナップサックに、各アイテム（重さ w, 価値 v）を最大価値で詰める。

```
アイテム: [(重さ=2, 価値=3), (重さ=3, 価値=4), (重さ=4, 価値=5), (重さ=5, 価値=6)]
容量: W = 8

DPテーブル dp[i][w] = i番目までのアイテムで容量wの最大価値:

       w: 0  1  2  3  4  5  6  7  8
item 0:   0  0  3  3  3  3  3  3  3
item 1:   0  0  3  4  4  7  7  7  7
item 2:   0  0  3  4  5  7  8  9  9
item 3:   0  0  3  4  5  7  8  9  10

答え: dp[3][8] = 10（アイテム1,2,3を選択: 3+4+5=12 > 10?）
実際: アイテム0(w=2,v=3) + アイテム2(w=4,v=5) + ... を最適化
```

```python
def knapsack_01(weights: list, values: list, W: int) -> int:
    """0/1 ナップサック - O(nW)"""
    n = len(weights)
    dp = [[0] * (W + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(W + 1):
            # アイテム i-1 を入れない
            dp[i][w] = dp[i - 1][w]
            # アイテム i-1 を入れる（容量に余裕がある場合）
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i][w],
                               dp[i - 1][w - weights[i - 1]] + values[i - 1])

    return dp[n][W]

# 空間最適化版（1次元DP）
def knapsack_01_optimized(weights: list, values: list, W: int) -> int:
    """0/1 ナップサック（空間最適化）- O(nW) 時間、O(W) 空間"""
    dp = [0] * (W + 1)

    for i in range(len(weights)):
        # 逆順に更新（同じアイテムを2回使わないため）
        for w in range(W, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])

    return dp[W]

weights = [2, 3, 4, 5]
values = [3, 4, 5, 6]
print(knapsack_01(weights, values, 8))            # 10
print(knapsack_01_optimized(weights, values, 8))  # 10
```

---

## 5. 最長共通部分列（LCS）

2つの文字列の最長共通部分列を求める。diff コマンドの基礎。

```
X = "ABCBDAB"
Y = "BDCAB"

DPテーブル:
     ""  B  D  C  A  B
  ""  0  0  0  0  0  0
  A   0  0  0  0  1  1
  B   0  1  1  1  1  2
  C   0  1  1  2  2  2
  B   0  1  1  2  2  3
  D   0  1  2  2  2  3
  A   0  1  2  2  3  3
  B   0  1  2  2  3  4

LCS = "BCAB" (長さ 4)
```

```python
def lcs(X: str, Y: str) -> tuple:
    """最長共通部分列 - O(mn)"""
    m, n = len(X), len(Y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # 復元
    result = []
    i, j = m, n
    while i > 0 and j > 0:
        if X[i - 1] == Y[j - 1]:
            result.append(X[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return dp[m][n], ''.join(reversed(result))

length, subseq = lcs("ABCBDAB", "BDCAB")
print(f"長さ: {length}, LCS: {subseq}")  # 長さ: 4, LCS: BCAB
```

---

## 6. その他の典型 DP

### コイン問題（最小枚数）

```python
def coin_change(coins: list, amount: int) -> int:
    """コイン問題 - O(n * amount)"""
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i and dp[i - coin] + 1 < dp[i]:
                dp[i] = dp[i - coin] + 1

    return dp[amount] if dp[amount] != float('inf') else -1

print(coin_change([1, 5, 10, 25], 30))  # 2 (25+5)
print(coin_change([3, 7], 5))            # -1 (不可能)
```

### 最長増加部分列（LIS）

```python
import bisect

def lis_dp(arr: list) -> int:
    """LIS (DP版) - O(n²)"""
    n = len(arr)
    dp = [1] * n
    for i in range(1, n):
        for j in range(i):
            if arr[j] < arr[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)

def lis_binary_search(arr: list) -> int:
    """LIS (二分探索版) - O(n log n)"""
    tails = []  # tails[i] = 長さ i+1 の IS の最小末尾
    for num in arr:
        pos = bisect.bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    return len(tails)

data = [10, 9, 2, 5, 3, 7, 101, 18]
print(lis_dp(data))             # 4 ([2, 3, 7, 18] or [2, 5, 7, 18])
print(lis_binary_search(data))  # 4
```

### 編集距離（レーベンシュタイン距離）

```python
def edit_distance(s1: str, s2: str) -> int:
    """編集距離 - O(mn)"""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # 削除
                    dp[i][j - 1],      # 挿入
                    dp[i - 1][j - 1],  # 置換
                )

    return dp[m][n]

print(edit_distance("kitten", "sitting"))  # 3
```

---

## 7. DP 設計フレームワーク

```
┌─────────────────────────────────────────────┐
│          DP 問題を解く5ステップ               │
├─────────────────────────────────────────────┤
│ 1. 状態の定義                                │
│    → dp[i] / dp[i][j] が何を表すか明確にする │
│                                              │
│ 2. 遷移式の導出                              │
│    → dp[i] = f(dp[i-1], dp[i-2], ...)       │
│                                              │
│ 3. 基底条件の設定                             │
│    → dp[0] = ?, dp[1] = ?                   │
│                                              │
│ 4. 計算順序の決定                             │
│    → ボトムアップの充填順                     │
│                                              │
│ 5. 答えの抽出                                 │
│    → dp[n] / max(dp) / 復元処理              │
└─────────────────────────────────────────────┘
```

---

## 8. メモ化 vs ボトムアップ 比較表

| 特性 | メモ化（トップダウン） | ボトムアップ |
|:---|:---|:---|
| 実装スタイル | 再帰 + キャッシュ | ループ + テーブル |
| 計算する部分問題 | 必要な分だけ | 全ての部分問題 |
| スタックオーバーフロー | 起こりうる | 起こらない |
| 空間最適化 | 困難 | 可能（次元削減） |
| コーディングの容易さ | 再帰的思考が自然 | 遷移順序を考える必要 |
| デバッグ | やや困難 | テーブルを確認しやすい |

## 典型DPパターン

| パターン | 代表問題 | 状態 | 計算量 |
|:---|:---|:---|:---|
| 1次元 DP | フィボナッチ、階段 | dp[i] | O(n) |
| 2次元 DP | LCS、編集距離 | dp[i][j] | O(mn) |
| ナップサック | 0/1ナップサック | dp[i][w] | O(nW) |
| 区間 DP | 行列連鎖積 | dp[l][r] | O(n³) |
| ビット DP | 巡回セールスマン | dp[S][v] | O(2^n * n) |
| 木 DP | 木上の最大独立集合 | dp[v][0/1] | O(V) |

---

## 9. アンチパターン

### アンチパターン1: 再帰のみでメモ化を忘れる

```python
# BAD: メモ化なし → O(2^n) で爆発
def fib_bad(n):
    if n <= 1:
        return n
    return fib_bad(n-1) + fib_bad(n-2)
# fib_bad(40) で数十秒かかる

# GOOD: メモ化で O(n) に
@lru_cache(maxsize=None)
def fib_good(n):
    if n <= 1:
        return n
    return fib_good(n-1) + fib_good(n-2)
# fib_good(1000) も一瞬
```

### アンチパターン2: 0/1 ナップサックで順方向に更新

```python
# BAD: 1次元DPで順方向に更新 → 同じアイテムを複数回使ってしまう
def bad_knapsack(weights, values, W):
    dp = [0] * (W + 1)
    for i in range(len(weights)):
        for w in range(weights[i], W + 1):  # 順方向 → 完全ナップサックになる
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    return dp[W]

# GOOD: 逆方向に更新
def good_knapsack(weights, values, W):
    dp = [0] * (W + 1)
    for i in range(len(weights)):
        for w in range(W, weights[i] - 1, -1):  # 逆方向 → 各アイテム最大1回
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    return dp[W]
```

### アンチパターン3: DP の状態定義が曖昧

```python
# BAD: dp[i] が何を表すか不明確なまま実装
dp = [0] * n
for i in range(n):
    dp[i] = ???  # 何を計算しているのか...

# GOOD: 状態を明確に定義してから実装
# dp[i] = 「i番目の要素で終わる最長増加部分列の長さ」
dp = [1] * n
for i in range(n):
    for j in range(i):
        if arr[j] < arr[i]:
            dp[i] = max(dp[i], dp[j] + 1)
```

---

## 10. FAQ

### Q1: DP と分割統治法の違いは？

**A:** 両方とも問題を分割して解くが、核心的な違いは「部分問題の重複」。分割統治法（マージソート等）は部分問題が独立しており重複しない。DP は同じ部分問題が何度も現れるため、結果をキャッシュして再利用する。重複がなければ分割統治、あれば DP を使う。

### Q2: DP の次元（状態数）をどう決める？

**A:** 問題を一意に表現するために必要な最小限のパラメータ数が次元になる。フィボナッチは n の1つ（1次元）、LCS は 2文字列の位置 i,j の2つ（2次元）。状態を増やすと表現力は上がるが計算量も増えるため、必要十分な次元を見極める。

### Q3: メモ化再帰でスタックオーバーフローが起きたら？

**A:** 3つの対策がある。(1) `sys.setrecursionlimit()` を増やす（応急処置）。(2) ボトムアップ DP に書き換える（推奨）。(3) 末尾再帰最適化が可能な場合はループに変換する。Python では (2) が最も安全。

---

## 11. まとめ

| 項目 | 要点 |
|:---|:---|
| DP の2条件 | 最適部分構造 + 重複部分問題 |
| メモ化 | トップダウン、再帰+キャッシュ、必要分だけ計算 |
| ボトムアップ | テーブル法、ループ、空間最適化が可能 |
| ナップサック | 0/1 は逆方向更新、完全は順方向更新 |
| LCS | 2次元 DP の代表問題。diff/スペルチェックに応用 |
| 設計手順 | 状態定義→遷移式→基底条件→計算順序→答え抽出 |

---

## 次に読むべきガイド

- [貪欲法](./05-greedy.md) -- DP が不要な場合の効率的な手法
- [分割統治法](./06-divide-conquer.md) -- 部分問題が重複しない場合の設計
- [問題解決法](../04-practice/00-problem-solving.md) -- DP 問題を見抜くパターン認識

---

## 参考文献

1. Cormen, T. H. et al. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. -- 第14章
2. Bellman, R. (1957). *Dynamic Programming*. Princeton University Press.
3. Skiena, S. S. (2020). *The Algorithm Design Manual* (3rd ed.). Springer. -- 第10章
4. Halim, S. & Halim, F. (2013). *Competitive Programming 3*. -- Chapter 3: Dynamic Programming
