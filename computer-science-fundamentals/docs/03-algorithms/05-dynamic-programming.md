# 動的計画法（DP）

> 動的計画法の本質は「同じ計算を2度しない」。重複する部分問題の結果を記憶することで、指数時間を多項式時間に削減する。

## この章で学ぶこと

- [ ] メモ化再帰とボトムアップDPの違いを理解する
- [ ] DPが適用できる問題の特徴を見抜ける
- [ ] 典型的なDPパターン（ナップサック、LCS等）を実装できる

## 前提知識

- 再帰 → 参照: [[00-what-is-algorithm.md]]
- 計算量解析 → 参照: [[01-complexity-analysis.md]]

---

## 1. DPの基本概念

### 1.1 フィボナッチ数列で理解するDP

```python
# ❌ 素朴な再帰: O(2^n) — 指数時間
def fib_naive(n):
    if n <= 1:
        return n
    return fib_naive(n - 1) + fib_naive(n - 2)

# fib(5) の呼び出し木:
#              fib(5)
#            /        \
#       fib(4)        fib(3)
#       /    \        /    \
#   fib(3)  fib(2)  fib(2) fib(1)
#   /   \    / \     / \
# f(2) f(1) f(1) f(0) f(1) f(0)
# → fib(2)が3回、fib(3)が2回計算される（無駄！）

# ✅ メモ化再帰（トップダウンDP）: O(n)
def fib_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib_memo(n - 1, memo) + fib_memo(n - 2, memo)
    return memo[n]

# ✅ ボトムアップDP（テーブル法）: O(n)
def fib_dp(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]

# ✅ 空間最適化: O(1)
def fib_optimized(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
```

### 1.2 DPが適用できる2つの条件

```
DPが使える問題の特徴:

  1. 最適部分構造（Optimal Substructure）
     → 問題の最適解が部分問題の最適解から構成される
     → 例: 最短経路の部分経路も最短経路

  2. 重複する部分問題（Overlapping Subproblems）
     → 同じ部分問題が何度も出現する
     → 例: fib(n) = fib(n-1) + fib(n-2) → fib(n-2)が複数箇所で必要

  DPが使えない例:
  - 最長単純パス → 最適部分構造がない（部分パスが干渉する）
  - 部分問題が重複しない分割統治 → メモ化の意味がない
```

### 1.3 メモ化 vs ボトムアップ

```
トップダウン（メモ化再帰）:
  利点: 自然な再帰的思考、必要な部分問題のみ計算
  欠点: 再帰のスタックオーバーフロー（深い再帰）
  実装: @functools.lru_cache

ボトムアップ（テーブル法）:
  利点: スタックオーバーフローなし、空間最適化しやすい
  欠点: 計算順序を事前に決める必要がある
  実装: forループ + 配列
```

---

## 2. 典型的なDPパターン

### 2.1 0-1 ナップサック問題

```python
def knapsack(weights, values, capacity):
    """重さ制限内で価値を最大化"""
    n = len(weights)
    # dp[i][w] = i個目までの品物で重さw以下での最大価値
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            dp[i][w] = dp[i-1][w]  # i番目を入れない
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w],
                    dp[i-1][w - weights[i-1]] + values[i-1])  # 入れる

    return dp[n][capacity]

# 例: weights=[2,3,4,5], values=[3,4,5,6], capacity=8
# → 最大価値 = 10（品物1+品物3: 重さ6, 価値8... 実際は品物2+品物4: 重さ8, 価値10）

# 計算量: O(n × W)  (W = 容量)
# 空間: O(n × W) → O(W) に最適化可能

# 1次元DPに空間最適化:
def knapsack_optimized(weights, values, capacity):
    dp = [0] * (capacity + 1)
    for i in range(len(weights)):
        for w in range(capacity, weights[i] - 1, -1):  # 逆順が重要！
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    return dp[capacity]
```

### 2.2 最長共通部分列（LCS）

```python
def lcs(s1, s2):
    """2つの文字列の最長共通部分列の長さ"""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]

# 例: lcs("ABCBDAB", "BDCAB") = 4 ("BCAB")
# 計算量: O(m × n)
# 用途: diff コマンド、DNA配列比較、バージョン管理

# LCS テーブルの可視化:
#     ""  B  D  C  A  B
# ""   0  0  0  0  0  0
#  A   0  0  0  0  1  1
#  B   0  1  1  1  1  2
#  C   0  1  1  2  2  2
#  B   0  1  1  2  2  3
#  D   0  1  2  2  2  3
#  A   0  1  2  2  3  3
#  B   0  1  2  2  3  4
```

### 2.3 コイン問題（完全ナップサック）

```python
def coin_change(coins, amount):
    """最小枚数でamountを作る"""
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i and dp[i - coin] + 1 < dp[i]:
                dp[i] = dp[i - coin] + 1

    return dp[amount] if dp[amount] != float('inf') else -1

# 例: coins=[1,5,10,25], amount=36
# → 25 + 10 + 1 = 3枚（貪欲法でも最適だが、coins=[1,3,4], amount=6では貪欲法は失敗）
# 貪欲法: 4+1+1 = 3枚 ❌
# DP:     3+3   = 2枚 ✅

# 計算量: O(amount × len(coins))
```

### 2.4 最長増加部分列（LIS）

```python
import bisect

def lis(arr):
    """最長増加部分列の長さ（O(n log n)）"""
    tails = []  # tails[i] = 長さ i+1 のLISの最小末尾値

    for x in arr:
        pos = bisect.bisect_left(tails, x)
        if pos == len(tails):
            tails.append(x)
        else:
            tails[pos] = x

    return len(tails)

# 例: arr = [10, 9, 2, 5, 3, 7, 101, 18]
# LIS = [2, 3, 7, 18] or [2, 5, 7, 18] → 長さ4

# O(n²) 版: dp[i] = arr[i]で終わるLISの長さ
# O(n log n) 版: 二分探索で tails 配列を管理
```

---

## 3. DP の状態設計

### 3.1 状態の定義方法

```
DP設計の思考プロセス:

  1. 状態を定義する
     → dp[i] = 「i番目まで見た時の最適値」
     → dp[i][j] = 「i番目まで見て、残り容量jの時の最適値」

  2. 遷移式を立てる
     → dp[i] = f(dp[i-1], dp[i-2], ...)
     → 現在の状態を、以前の状態の組み合わせで表現

  3. 基底条件（ベースケース）を決める
     → dp[0] = ? dp[1] = ?

  4. 計算順序を決める
     → 依存関係に従って小→大の順

  5. 答えを特定する
     → dp[n] or max(dp) or dp[n][W]

  よくある状態の取り方:
  ┌──────────────────┬──────────────────────────────┐
  │ パターン          │ 状態の例                      │
  ├──────────────────┼──────────────────────────────┤
  │ 線形DP          │ dp[i] = i番目までの最適値      │
  │ 区間DP          │ dp[i][j] = 区間[i,j]の最適値  │
  │ ナップサックDP   │ dp[i][w] = i個でw容量の最適値 │
  │ 木DP            │ dp[v] = 部分木vでの最適値     │
  │ 桁DP            │ dp[pos][tight] = pos桁目まで  │
  │ ビットDP        │ dp[mask] = 集合maskの状態     │
  └──────────────────┴──────────────────────────────┘
```

---

## 4. 実務でのDP

### 4.1 DP が活躍する場面

```
実務でのDP応用:

  1. テキスト処理
     - diff アルゴリズム (LCS)
     - スペルチェック (編集距離)
     - 自然言語処理 (CYK構文解析)

  2. 機械学習
     - Viterbiアルゴリズム（隠れマルコフモデル）
     - CTC (Connectionist Temporal Classification)
     - ビームサーチ（近似的なDP）

  3. 最適化
     - リソース割り当て
     - スケジューリング
     - 在庫管理

  4. ゲーム
     - ゲーム木の評価（ミニマックス + メモ化）
     - 最適戦略の計算
```

---

## 5. 実践演習

### 演習1: 基本DP（基礎）
階段の上り方問題: N段の階段を1段または2段ずつ上る方法の総数を求めよ。

### 演習2: 2次元DP（応用）
編集距離（レーベンシュタイン距離）を計算する関数を実装せよ。"kitten" → "sitting" の編集距離を求めよ。

### 演習3: ビットDP（発展）
巡回セールスマン問題（TSP）を、ビットマスクDPで O(n² × 2^n) で解くプログラムを実装せよ。

---

## FAQ

### Q1: DPと分割統治の違いは？
**A**: 分割統治は部分問題が独立（マージソート: 左右が独立）。DPは部分問題が重複（フィボナッチ: fib(n-1)とfib(n-2)がfib(n-3)を共有）。分割統治 + メモ化 = DP とも言える。

### Q2: DPの状態設計のコツは？
**A**: まず「何を決めたら残りが決まるか」を考える。状態は「残りの問題を解くのに必要な最小限の情報」。状態が多すぎたら次元削減を検討。まずは素朴なDPを書いてから最適化する。

### Q3: 貪欲法とDPの使い分けは？
**A**: 貪欲法は局所最適が全体最適になる場合のみ有効（証明が必要）。DPは全ての選択肢を考慮するので常に正しい。迷ったらDPが安全。貪欲法が使えるなら計算量は小さい。

---

## まとめ

| 概念 | ポイント |
|------|---------|
| DP条件 | 最適部分構造 + 重複部分問題 |
| メモ化 | トップダウン。再帰+キャッシュ |
| ボトムアップ | テーブル法。forループ+配列 |
| 典型パターン | ナップサック、LCS、コイン、LIS |
| 状態設計 | 「残りを解くのに必要な最小限の情報」を状態にする |

---

## 次に読むべきガイド
→ [[06-greedy-and-backtracking.md]] — 貪欲法とバックトラック

---

## 参考文献
1. Cormen, T. H. et al. "Introduction to Algorithms." Chapter 15: Dynamic Programming.
2. Bellman, R. "Dynamic Programming." Princeton University Press, 1957.
3. Skiena, S. S. "The Algorithm Design Manual." Chapter 10.
