# 動的計画法（DP）

> 動的計画法の本質は「同じ計算を2度しない」。重複する部分問題の結果を記憶することで、指数時間を多項式時間に削減する。

## この章で学ぶこと

- [ ] メモ化再帰とボトムアップDPの違いを理解する
- [ ] DPが適用できる問題の特徴を見抜ける
- [ ] 典型的なDPパターン（ナップサック、LCS等）を実装できる
- [ ] 状態設計の手法を体系的に習得する
- [ ] 空間最適化・次元削減のテクニックを使いこなせる
- [ ] 区間DP、木DP、桁DP、ビットDPなど高度なDPを理解する

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

### 1.4 Python での lru_cache を使ったメモ化

```python
from functools import lru_cache

# lru_cache を使うと、メモ化を自分で実装する必要がない
@lru_cache(maxsize=None)
def fib_cached(n):
    if n <= 1:
        return n
    return fib_cached(n - 1) + fib_cached(n - 2)

# Python 3.9+ では cache デコレータも使える
from functools import cache

@cache
def fib_py39(n):
    if n <= 1:
        return n
    return fib_py39(n - 1) + fib_py39(n - 2)

# lru_cache の注意点:
# 1. 引数はハッシュ可能でなければならない（list不可、tuple可）
# 2. maxsize=None で無制限キャッシュ（メモリに注意）
# 3. 再帰の深さ制限に注意（sys.setrecursionlimit で変更可能）
# 4. グローバルな辞書を汚染しない（関数ごとに独立）

# 使用例: グリッド上のパス数
@lru_cache(maxsize=None)
def grid_paths(m, n):
    """m×n グリッドの左上から右下へのパス数（右と下のみ移動）"""
    if m == 1 or n == 1:
        return 1
    return grid_paths(m - 1, n) + grid_paths(m, n - 1)

print(grid_paths(10, 10))  # 48620
```

### 1.5 DPテーブルの可視化テクニック

```python
def visualize_dp_table(dp, row_labels=None, col_labels=None, title="DP Table"):
    """DPテーブルを見やすく表示するユーティリティ"""
    print(f"\n--- {title} ---")

    if isinstance(dp[0], list):
        # 2次元テーブル
        rows = len(dp)
        cols = len(dp[0])

        # ヘッダー
        if col_labels:
            header = "     " + "".join(f"{l:>5}" for l in col_labels)
            print(header)
            print("     " + "-" * (cols * 5))

        for i in range(rows):
            label = f"{row_labels[i]:>3} |" if row_labels else f"{i:>3} |"
            row = "".join(f"{dp[i][j]:>5}" for j in range(cols))
            print(label + row)
    else:
        # 1次元テーブル
        if col_labels:
            header = "".join(f"{l:>5}" for l in col_labels)
            print(header)
        row = "".join(f"{v:>5}" for v in dp)
        print(row)

# 使用例: LCSテーブルの可視化
s1, s2 = "ABCB", "BDCAB"
m, n = len(s1), len(s2)
dp = [[0] * (n + 1) for _ in range(m + 1)]
for i in range(1, m + 1):
    for j in range(1, n + 1):
        if s1[i-1] == s2[j-1]:
            dp[i][j] = dp[i-1][j-1] + 1
        else:
            dp[i][j] = max(dp[i-1][j], dp[i][j-1])

visualize_dp_table(
    dp,
    row_labels=['""'] + list(s1),
    col_labels=['""'] + list(s2),
    title="LCS Table"
)
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

#### ナップサック問題の選択物復元

```python
def knapsack_with_items(weights, values, capacity):
    """最大価値だけでなく、選んだ品物も復元する"""
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            dp[i][w] = dp[i-1][w]
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w],
                    dp[i-1][w - weights[i-1]] + values[i-1])

    # 選択した品物を復元（逆順にたどる）
    selected = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            selected.append(i - 1)  # 0-indexed
            w -= weights[i-1]

    selected.reverse()
    return dp[n][capacity], selected

# 使用例
weights = [2, 3, 4, 5]
values  = [3, 4, 5, 6]
capacity = 8
max_val, items = knapsack_with_items(weights, values, capacity)
print(f"最大価値: {max_val}")        # 10
print(f"選んだ品物: {items}")        # [1, 3]（0-indexed）
print(f"重さ合計: {sum(weights[i] for i in items)}")  # 8
print(f"価値合計: {sum(values[i] for i in items)}")   # 10
```

#### 個数制限なしナップサック（完全ナップサック）

```python
def unbounded_knapsack(weights, values, capacity):
    """各品物を何個でも使える場合のナップサック"""
    dp = [0] * (capacity + 1)

    for w in range(1, capacity + 1):
        for i in range(len(weights)):
            if weights[i] <= w:
                dp[w] = max(dp[w], dp[w - weights[i]] + values[i])

    return dp[capacity]

# 0-1ナップサックとの違い:
# - 0-1: 各品物は1回のみ → 内側ループを逆順に回す
# - 完全: 各品物は何回でも → 内側ループを順方向に回す
# この違いが生じる理由:
#   逆順にすると dp[w - weights[i]] は「品物iを使う前の状態」を参照
#   順方向だと dp[w - weights[i]] は「品物iを使った後の状態」を含む可能性がある

# 個数制限付きナップサック
def bounded_knapsack(weights, values, counts, capacity):
    """各品物に使用回数制限がある場合"""
    dp = [0] * (capacity + 1)

    for i in range(len(weights)):
        # 二進数分解テクニック: count を 1, 2, 4, ... のグループに分解
        count = counts[i]
        k = 1
        while count > 0:
            actual = min(k, count)
            w_group = weights[i] * actual
            v_group = values[i] * actual
            # 0-1ナップサックとして処理
            for w in range(capacity, w_group - 1, -1):
                dp[w] = max(dp[w], dp[w - w_group] + v_group)
            count -= actual
            k *= 2

    return dp[capacity]

# 例: 品物A(w=2,v=3)を5個、品物B(w=3,v=5)を3個使える
# → 二進分解: A: 1+2+2個のグループ、B: 1+2個のグループ
# → 計算量: O(W × Σ(log count_i)) ← O(W × Σ count_i) より高速
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

#### LCS の文字列復元

```python
def lcs_string(s1, s2):
    """LCSの実際の文字列を復元する"""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    # 復元（逆順にたどる）
    result = []
    i, j = m, n
    while i > 0 and j > 0:
        if s1[i-1] == s2[j-1]:
            result.append(s1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1

    return "".join(reversed(result))

# 使用例
print(lcs_string("ABCBDAB", "BDCAB"))  # "BCAB"

# diff コマンド風の出力
def print_diff(s1, s2):
    """2つの文字列（行のリスト）のdiffを表示"""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    # 逆順に辿ってdiff操作を収集
    ops = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s1[i-1] == s2[j-1]:
            ops.append((' ', s1[i-1]))
            i -= 1
            j -= 1
        elif j > 0 and (i == 0 or dp[i][j-1] >= dp[i-1][j]):
            ops.append(('+', s2[j-1]))
            j -= 1
        else:
            ops.append(('-', s1[i-1]))
            i -= 1

    ops.reverse()
    for op, line in ops:
        print(f"{op} {line}")

# 使用例
lines1 = ["apple", "banana", "cherry", "date"]
lines2 = ["apple", "blueberry", "cherry", "elderberry"]
print_diff(lines1, lines2)
# 出力:
#   apple
# - banana
# + blueberry
#   cherry
# - date
# + elderberry
```

#### 空間最適化LCS

```python
def lcs_space_optimized(s1, s2):
    """O(min(m,n)) の空間でLCS長を計算"""
    # 短い方を内側のループにする
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    m, n = len(s1), len(s2)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev, curr = curr, [0] * (n + 1)

    return prev[n]

# 注意: この最適化では文字列の復元はできない
# 復元も空間 O(n) でやりたい場合は Hirschberg のアルゴリズムを使う
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

#### コインの組み合わせ数

```python
def coin_combinations(coins, amount):
    """amountを作る方法の総数（順序を区別しない）"""
    dp = [0] * (amount + 1)
    dp[0] = 1

    # 外側ループがコイン → 順序を区別しない（組み合わせ）
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]

    return dp[amount]

def coin_permutations(coins, amount):
    """amountを作る方法の総数（順序を区別する）"""
    dp = [0] * (amount + 1)
    dp[0] = 1

    # 外側ループが金額 → 順序を区別する（順列）
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] += dp[i - coin]

    return dp[amount]

# 例: coins=[1,2,3], amount=4
# 組み合わせ: {1+1+1+1, 1+1+2, 1+3, 2+2} = 4通り
# 順列:       上記 + {2+1+1, 3+1, 2+1+1, 1+2+1, ...} = 7通り

# この違いはDPのループ順序の違いから生じる
# 非常によく面接で聞かれるポイント
```

#### 使用コインの復元

```python
def coin_change_with_trace(coins, amount):
    """最小枚数とその内訳を返す"""
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    parent = [-1] * (amount + 1)  # どのコインを使ったか

    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i and dp[i - coin] + 1 < dp[i]:
                dp[i] = dp[i - coin] + 1
                parent[i] = coin

    if dp[amount] == float('inf'):
        return -1, []

    # 復元
    used_coins = []
    current = amount
    while current > 0:
        used_coins.append(parent[current])
        current -= parent[current]

    return dp[amount], used_coins

# 使用例
min_coins, coins_used = coin_change_with_trace([1, 3, 4], 6)
print(f"最小枚数: {min_coins}")    # 2
print(f"使用コイン: {coins_used}")  # [3, 3]
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

#### LIS の O(n^2) 版と復元

```python
def lis_quadratic_with_recovery(arr):
    """O(n^2) のLIS + 実際の部分列を復元"""
    n = len(arr)
    if n == 0:
        return 0, []

    dp = [1] * n
    parent = [-1] * n

    for i in range(1, n):
        for j in range(i):
            if arr[j] < arr[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j

    # 最大値の位置を見つける
    max_len = max(dp)
    max_idx = dp.index(max_len)

    # 復元
    result = []
    idx = max_idx
    while idx != -1:
        result.append(arr[idx])
        idx = parent[idx]

    result.reverse()
    return max_len, result

# 使用例
length, subsequence = lis_quadratic_with_recovery([10, 9, 2, 5, 3, 7, 101, 18])
print(f"長さ: {length}")          # 4
print(f"部分列: {subsequence}")   # [2, 3, 7, 18] or [2, 5, 7, 101] etc.
```

#### 最長非減少部分列（非狭義）

```python
def longest_non_decreasing_subsequence(arr):
    """等しい要素も許す場合（非狭義単調増加）"""
    tails = []
    for x in arr:
        # bisect_right を使う（等しい要素を許す）
        pos = bisect.bisect_right(tails, x)
        if pos == len(tails):
            tails.append(x)
        else:
            tails[pos] = x
    return len(tails)

# 狭義 vs 非狭義の違い:
# 狭義増加: bisect_left → [2,3,7,18] (等しい要素は不可)
# 非減少:   bisect_right → [2,3,3,7,18] (等しい要素も可)
```

### 2.5 編集距離（レーベンシュタイン距離）

```python
def edit_distance(s1, s2):
    """2つの文字列の編集距離を計算"""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 基底条件
    for i in range(m + 1):
        dp[i][0] = i  # s1の先頭i文字を全削除
    for j in range(n + 1):
        dp[0][j] = j  # 空文字列にs2の先頭j文字を全挿入

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]  # 一致 → コストなし
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],      # 削除
                    dp[i][j-1],      # 挿入
                    dp[i-1][j-1]     # 置換
                )

    return dp[m][n]

# 例:
# edit_distance("kitten", "sitting") = 3
#   kitten → sitten (置換 k→s)
#   sitten → sittin (置換 e→i)
#   sittin → sitting (挿入 g)

# 用途:
# - スペルチェッカー
# - DNA配列のアラインメント
# - ファジーマッチング（あいまい検索）
# - 自然言語処理（単語の類似度）

# 計算量: O(m × n)
# 空間: O(m × n) → O(min(m,n)) に最適化可能
```

#### 編集距離の空間最適化と操作復元

```python
def edit_distance_optimized(s1, s2):
    """O(min(m,n)) の空間で編集距離を計算"""
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    m, n = len(s1), len(s2)
    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                curr[j] = prev[j-1]
            else:
                curr[j] = 1 + min(prev[j], curr[j-1], prev[j-1])
        prev, curr = curr, [0] * (n + 1)

    return prev[n]

def edit_operations(s1, s2):
    """編集距離と具体的な操作列を返す"""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    # 操作の復元
    ops = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s1[i-1] == s2[j-1]:
            ops.append(('MATCH', s1[i-1], i-1))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            ops.append(('REPLACE', f"{s1[i-1]}→{s2[j-1]}", i-1))
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            ops.append(('INSERT', s2[j-1], i))
            j -= 1
        else:
            ops.append(('DELETE', s1[i-1], i-1))
            i -= 1

    ops.reverse()
    return dp[m][n], ops

# 使用例
dist, ops = edit_operations("kitten", "sitting")
print(f"編集距離: {dist}")
for op, char, pos in ops:
    if op != 'MATCH':
        print(f"  {op}: '{char}' at position {pos}")
```

### 2.6 部分和問題

```python
def subset_sum(nums, target):
    """numsの要素の部分集合でtargetを作れるか"""
    dp = [False] * (target + 1)
    dp[0] = True

    for num in nums:
        # 逆順に処理（各要素を1回のみ使用）
        for j in range(target, num - 1, -1):
            dp[j] = dp[j] or dp[j - num]

    return dp[target]

# 応用: 配列を2つのグループに分けて差を最小化
def min_subset_difference(nums):
    """2つのグループの和の差を最小にする"""
    total = sum(nums)
    half = total // 2

    dp = [False] * (half + 1)
    dp[0] = True

    for num in nums:
        for j in range(half, num - 1, -1):
            dp[j] = dp[j] or dp[j - num]

    # dp[j]がTrueの最大のjを見つける
    for j in range(half, -1, -1):
        if dp[j]:
            return total - 2 * j

# 例: nums = [1, 6, 11, 5]
# グループ1: {1, 5, 6} = 12, グループ2: {11} = 11
# 差 = 1

# 応用: 配列をk個の等和グループに分割可能か
def can_partition_k(nums, k):
    """k個の等しい和のグループに分割できるか"""
    total = sum(nums)
    if total % k != 0:
        return False
    target = total // k

    nums.sort(reverse=True)
    if nums[0] > target:
        return False

    buckets = [0] * k

    def backtrack(idx):
        if idx == len(nums):
            return all(b == target for b in buckets)

        seen = set()
        for i in range(k):
            if buckets[i] + nums[idx] <= target and buckets[i] not in seen:
                seen.add(buckets[i])
                buckets[i] += nums[idx]
                if backtrack(idx + 1):
                    return True
                buckets[i] -= nums[idx]

        return False

    return backtrack(0)
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

### 3.2 状態設計の実践例

```python
# 例1: 最大部分配列和（カダネのアルゴリズム）
# 状態: dp[i] = arr[i]で終わる連続部分配列の最大和
def max_subarray(arr):
    dp = arr[0]
    best = arr[0]
    for i in range(1, len(arr)):
        dp = max(arr[i], dp + arr[i])  # 新しく始めるか、続けるか
        best = max(best, dp)
    return best

# arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
# dp:   -2  1  -2  4   3  5  6   1  5
# best: -2  1   1  4   4  5  6   6  6
# 答え: 6（部分配列 [4, -1, 2, 1]）

# 例2: ペンキ塗り問題（色のコスト最小化）
# 状態: dp[i][c] = i番目の家を色cで塗った時の最小コスト
def paint_houses(costs):
    """n軒の家を3色で塗る。隣接する家は異なる色"""
    if not costs:
        return 0

    n = len(costs)
    # dp[i][c] を 前の家の結果のみで計算可能 → O(1) 空間
    prev = costs[0][:]

    for i in range(1, n):
        curr = [0, 0, 0]
        curr[0] = costs[i][0] + min(prev[1], prev[2])
        curr[1] = costs[i][1] + min(prev[0], prev[2])
        curr[2] = costs[i][2] + min(prev[0], prev[1])
        prev = curr

    return min(prev)

# 例: costs = [[17,2,17],[16,16,5],[14,3,19]]
# 家0を緑(2) + 家1を青(5) + 家2を緑(3) = 10

# 例3: ワイン問題（区間DPの一種）
# n本のワインを年に1本ずつ売る。左端か右端からのみ選べる。
# 売値 = ワインの価格 × 年数
def max_wine_profit(prices):
    """左端か右端から年に1本ずつ売り、利益を最大化"""
    n = len(prices)

    @lru_cache(maxsize=None)
    def dp(left, right):
        year = n - (right - left)  # 何年目か (1-indexed)
        if left > right:
            return 0
        if left == right:
            return prices[left] * year

        sell_left = prices[left] * year + dp(left + 1, right)
        sell_right = prices[right] * year + dp(left, right - 1)
        return max(sell_left, sell_right)

    return dp(0, n - 1)
```

---

## 4. 高度なDPパターン

### 4.1 区間DP

```python
# 区間DP: dp[i][j] = 区間 [i, j] の最適解
# 典型: 行列の連鎖積、最適二分探索木、回文分割

def matrix_chain_multiplication(dims):
    """行列の連鎖積の最小乗算回数
    dims[i-1] × dims[i] が i番目の行列のサイズ
    """
    n = len(dims) - 1  # 行列の数
    dp = [[0] * n for _ in range(n)]

    # 区間の長さを2から順に
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            for k in range(i, j):
                cost = dp[i][k] + dp[k+1][j] + dims[i] * dims[k+1] * dims[j+1]
                dp[i][j] = min(dp[i][j], cost)

    return dp[0][n-1]

# 例: 4つの行列 A(10×30), B(30×5), C(5×60), D(60×10)
# dims = [10, 30, 5, 60, 10]
# 最適な括弧付け: (A × B) × (C × D)
# 乗算回数: 10*30*5 + 5*60*10 + 10*5*10 = 1500 + 3000 + 500 = 5000
# 最悪の括弧付け: A × (B × (C × D)) = 30*5*60 + 30*60*10 + 10*30*10 = 9000+18000+3000 = 30000

# 括弧付けの復元
def matrix_chain_with_paren(dims):
    n = len(dims) - 1
    dp = [[0] * n for _ in range(n)]
    split = [[0] * n for _ in range(n)]

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            for k in range(i, j):
                cost = dp[i][k] + dp[k+1][j] + dims[i] * dims[k+1] * dims[j+1]
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    split[i][j] = k

    def build_paren(i, j):
        if i == j:
            return f"A{i+1}"
        k = split[i][j]
        left = build_paren(i, k)
        right = build_paren(k+1, j)
        return f"({left} × {right})"

    return dp[0][n-1], build_paren(0, n-1)
```

#### 回文分割

```python
def min_palindrome_partitions(s):
    """文字列を回文に分割する最小カット数"""
    n = len(s)

    # is_pal[i][j] = s[i:j+1] が回文かどうか
    is_pal = [[False] * n for _ in range(n)]
    for i in range(n):
        is_pal[i][i] = True
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if length == 2:
                is_pal[i][j] = (s[i] == s[j])
            else:
                is_pal[i][j] = (s[i] == s[j]) and is_pal[i+1][j-1]

    # dp[i] = s[0:i+1] を回文に分割する最小カット数
    dp = list(range(n))  # 最悪: 各文字が1つの回文

    for i in range(1, n):
        if is_pal[0][i]:
            dp[i] = 0
            continue
        for j in range(1, i + 1):
            if is_pal[j][i]:
                dp[i] = min(dp[i], dp[j-1] + 1)

    return dp[n-1]

# 例: "aab" → ["aa", "b"] → 1カット
# 例: "abcba" → 0カット（全体が回文）
# 例: "abcde" → 4カット（各文字がそれぞれ回文）
```

### 4.2 木DP

```python
# 木DP: 木構造上でDPを行う
# 典型: 木の直径、木の最大独立集合、木の重心分解

class TreeNode:
    def __init__(self, val=0):
        self.val = val
        self.children = []

def max_independent_set(root):
    """木の最大独立集合（隣接するノードを同時に選べない）の最大値和"""

    def dp(node):
        if not node:
            return 0, 0

        # include: このノードを含む場合の最大値
        # exclude: このノードを含まない場合の最大値
        include = node.val
        exclude = 0

        for child in node.children:
            child_inc, child_exc = dp(child)
            include += child_exc      # 子を含まない
            exclude += max(child_inc, child_exc)  # 子は含んでも含まなくてもよい

        return include, exclude

    inc, exc = dp(root)
    return max(inc, exc)

# 木の直径（最長パス）
def tree_diameter(adj, n):
    """隣接リスト表現の木の直径を求める"""
    diameter = [0]

    def dfs(node, parent):
        max1 = max2 = 0  # 子からの最長と2番目

        for neighbor in adj[node]:
            if neighbor != parent:
                depth = dfs(neighbor, node)
                if depth > max1:
                    max2 = max1
                    max1 = depth
                elif depth > max2:
                    max2 = depth

        diameter[0] = max(diameter[0], max1 + max2)
        return max1 + 1

    dfs(0, -1)
    return diameter[0]

# 使用例（隣接リスト）
# 木:    0
#       / \
#      1   2
#     / \   \
#    3   4   5
adj = [
    [1, 2],    # 0の隣接
    [0, 3, 4], # 1の隣接
    [0, 5],    # 2の隣接
    [1],       # 3の隣接
    [1],       # 4の隣接
    [2],       # 5の隣接
]
print(tree_diameter(adj, 6))  # 4 (3→1→0→2→5)
```

### 4.3 桁DP

```python
def count_numbers_with_digit(n, d):
    """1からnまでの整数の中で、数字dが現れる個数の合計"""
    digits = list(map(int, str(n)))
    num_digits = len(digits)

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def dp(pos, count, tight, started):
        """
        pos:     現在の桁位置
        count:   これまでに数字dが出現した回数
        tight:   上界制約があるか
        started: 数が始まっているか（先頭の0を除外）
        """
        if pos == num_digits:
            return count if started else 0

        limit = digits[pos] if tight else 9
        result = 0

        for digit in range(0, limit + 1):
            new_started = started or (digit > 0)
            new_count = count + (1 if digit == d and new_started else 0)
            new_tight = tight and (digit == limit)
            result += dp(pos + 1, new_count, new_tight, new_started)

        return result

    return dp(0, 0, True, False)

# 例: 1から100までに「1」が何回出現するか
# 1, 10, 11(2回), 12, ..., 19, 21, 31, ..., 91, 100 = 21回

# 桁DPの応用例:
# - N以下の「ラッキーナンバー」（4と7のみ含む数）の個数
# - 隣接する桁の差がk以下の数の個数
# - 桁の和がSの倍数である数の個数
```

#### 桁DPのもう一つの典型例

```python
def count_step_numbers(low, high):
    """low以上high以下のステップ数（隣接桁の差が1）の個数"""

    def count_up_to(n):
        if n < 0:
            return 0
        digits = list(map(int, str(n)))
        num_digits = len(digits)

        @lru_cache(maxsize=None)
        def dp(pos, prev_digit, tight, started):
            if pos == num_digits:
                return 1 if started else 0

            limit = digits[pos] if tight else 9
            result = 0

            for digit in range(0, limit + 1):
                if not started and digit == 0:
                    result += dp(pos + 1, -1, False, False)
                elif not started or abs(digit - prev_digit) == 1:
                    new_tight = tight and (digit == limit)
                    result += dp(pos + 1, digit, new_tight, True)

            return result

        return dp(0, -1, True, False)

    return count_up_to(high) - count_up_to(low - 1)

# 例: 10から100の間のステップ数
# 10, 12, 21, 23, 32, 34, 43, 45, 54, 56, 65, 67, 76, 78, 87, 89, 98 = 17個
```

### 4.4 ビットDP（ビットマスクDP）

```python
def tsp(dist):
    """巡回セールスマン問題（TSP）をビットDPで解く"""
    n = len(dist)
    # dp[mask][i] = 集合maskの都市を訪問し、最後に都市iにいる場合の最小コスト
    INF = float('inf')
    dp = [[INF] * n for _ in range(1 << n)]
    dp[1][0] = 0  # 都市0からスタート

    for mask in range(1 << n):
        for u in range(n):
            if dp[mask][u] == INF:
                continue
            if not (mask & (1 << u)):
                continue
            for v in range(n):
                if mask & (1 << v):
                    continue  # 訪問済み
                new_mask = mask | (1 << v)
                new_cost = dp[mask][u] + dist[u][v]
                dp[new_mask][v] = min(dp[new_mask][v], new_cost)

    # 全都市を訪問して都市0に戻る
    full_mask = (1 << n) - 1
    result = min(dp[full_mask][i] + dist[i][0] for i in range(n))

    return result

# 計算量: O(n^2 × 2^n)
# 空間: O(n × 2^n)
# n=20 まで現実的（2^20 × 20^2 ≈ 4億）

# 経路の復元
def tsp_with_path(dist):
    n = len(dist)
    INF = float('inf')
    dp = [[INF] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]
    dp[1][0] = 0

    for mask in range(1 << n):
        for u in range(n):
            if dp[mask][u] == INF or not (mask & (1 << u)):
                continue
            for v in range(n):
                if mask & (1 << v):
                    continue
                new_mask = mask | (1 << v)
                new_cost = dp[mask][u] + dist[u][v]
                if new_cost < dp[new_mask][v]:
                    dp[new_mask][v] = new_cost
                    parent[new_mask][v] = u

    full_mask = (1 << n) - 1
    last = min(range(n), key=lambda i: dp[full_mask][i] + dist[i][0])
    min_cost = dp[full_mask][last] + dist[last][0]

    # 経路復元
    path = []
    mask = full_mask
    node = last
    while node != -1:
        path.append(node)
        prev = parent[mask][node]
        mask ^= (1 << node)
        node = prev
    path.reverse()
    path.append(0)  # 出発点に戻る

    return min_cost, path

# 使用例
dist = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]
cost, path = tsp_with_path(dist)
print(f"最小コスト: {cost}")    # 80
print(f"経路: {path}")          # [0, 1, 3, 2, 0]
```

#### ビットDPの応用: 仕事割り当て問題

```python
def min_cost_assignment(cost):
    """n人にn個の仕事を1対1で割り当てるとき、総コストを最小化"""
    n = len(cost)
    INF = float('inf')
    dp = [INF] * (1 << n)
    dp[0] = 0

    for mask in range(1 << n):
        person = bin(mask).count('1')  # 何人に割り当て済みか
        if person >= n:
            continue
        for job in range(n):
            if mask & (1 << job):
                continue
            new_mask = mask | (1 << job)
            dp[new_mask] = min(dp[new_mask], dp[mask] + cost[person][job])

    return dp[(1 << n) - 1]

# 例:
cost = [
    [9, 2, 7, 8],
    [6, 4, 3, 7],
    [5, 8, 1, 8],
    [7, 6, 9, 4]
]
print(min_cost_assignment(cost))  # 13 (人0→仕事1, 人1→仕事2, 人2→仕事0 ではなく最適解)
```

### 4.5 確率DP

```python
def expected_dice_rolls(target):
    """サイコロを振って合計がtarget以上になるまでの期待回数"""
    dp = [0.0] * (target + 7)

    # dp[i] = 現在の合計がiの時、target以上になるまでの期待回数
    for i in range(target - 1, -1, -1):
        dp[i] = 1  # サイコロを1回振る
        for face in range(1, 7):
            dp[i] += dp[i + face] / 6

    return dp[0]

# 例: target=10 → 約3.77回

# ランダムウォークの到達確率
def random_walk_probability(n, target, steps):
    """1次元ランダムウォークでsteps歩後にtargetにいる確率"""
    # dp[step][pos] = step歩後に位置posにいる確率
    # 位置は -steps から +steps の範囲
    offset = steps
    dp = [[0.0] * (2 * steps + 1) for _ in range(steps + 1)]
    dp[0][offset] = 1.0  # 初期位置 0

    for step in range(steps):
        for pos in range(2 * steps + 1):
            if dp[step][pos] == 0:
                continue
            # 左に移動
            if pos > 0:
                dp[step + 1][pos - 1] += dp[step][pos] * 0.5
            # 右に移動
            if pos < 2 * steps:
                dp[step + 1][pos + 1] += dp[step][pos] * 0.5

    target_idx = target + offset
    if 0 <= target_idx < 2 * steps + 1:
        return dp[steps][target_idx]
    return 0.0
```

### 4.6 文字列DP

```python
# 正規表現マッチング（'.'と'*'のみ）
def regex_match(text, pattern):
    """正規表現マッチング: '.'は任意の1文字、'*'は直前の文字の0回以上"""
    m, n = len(text), len(pattern)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True

    # パターンの先頭が "X*Y*..." の場合、空文字列にマッチ
    for j in range(2, n + 1):
        if pattern[j-1] == '*':
            dp[0][j] = dp[0][j-2]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pattern[j-1] == '*':
                # '*'の前の文字を0回使う
                dp[i][j] = dp[i][j-2]
                # '*'の前の文字を1回以上使う
                if pattern[j-2] == '.' or pattern[j-2] == text[i-1]:
                    dp[i][j] = dp[i][j] or dp[i-1][j]
            elif pattern[j-1] == '.' or pattern[j-1] == text[i-1]:
                dp[i][j] = dp[i-1][j-1]

    return dp[m][n]

# テスト
assert regex_match("aa", "a") == False
assert regex_match("aa", "a*") == True
assert regex_match("ab", ".*") == True
assert regex_match("aab", "c*a*b") == True

# ワイルドカードマッチング（'?'と'*'）
def wildcard_match(text, pattern):
    """'?'は任意の1文字、'*'は任意の文字列（0文字以上）"""
    m, n = len(text), len(pattern)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True

    for j in range(1, n + 1):
        if pattern[j-1] == '*':
            dp[0][j] = dp[0][j-1]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pattern[j-1] == '*':
                dp[i][j] = dp[i-1][j] or dp[i][j-1]
            elif pattern[j-1] == '?' or pattern[j-1] == text[i-1]:
                dp[i][j] = dp[i-1][j-1]

    return dp[m][n]
```

### 4.7 ゲーム理論DP

```python
def stone_game(piles):
    """石取りゲーム: 2人のプレイヤーが交互に左端か右端の石山を取る
    先手が最適にプレイした場合の得点差を返す"""
    n = len(piles)
    # dp[i][j] = 区間[i,j]で先手が後手より多く取れる量
    dp = [[0] * n for _ in range(n)]

    for i in range(n):
        dp[i][i] = piles[i]

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = max(
                piles[i] - dp[i+1][j],  # 左端を取る
                piles[j] - dp[i][j-1]   # 右端を取る
            )

    return dp[0][n-1]

# 例: piles = [5, 3, 4, 5]
# 先手は最適にプレイすると 5+4=9、後手は 3+5=8 → 差は1
# stone_game([5,3,4,5]) = 1

# Nim Game
def nim_game(piles):
    """Nim: 各山からいくつでも石を取れる。最後の石を取ったら勝ち。
    先手が勝てるかどうかを返す"""
    xor_sum = 0
    for p in piles:
        xor_sum ^= p
    return xor_sum != 0  # XOR和が0でなければ先手必勝

# Sprague-Grundy定理:
# 任意の不偏ゲーム（impartial game）はNimに帰着できる
# Grundy数 = mex(後続状態のGrundy数の集合)
# mex(S) = Sに含まれない最小の非負整数

def grundy_number(pos, moves, memo={}):
    """ポジションposからのGrundy数を計算"""
    if pos in memo:
        return memo[pos]

    reachable = set()
    for m in moves:
        if pos >= m:
            reachable.add(grundy_number(pos - m, moves, memo))

    # mex を計算
    g = 0
    while g in reachable:
        g += 1

    memo[pos] = g
    return g
```

---

## 5. 実務でのDP

### 5.1 DP が活躍する場面

```
実務でのDP応用:

  1. テキスト処理
     - diff アルゴリズム (LCS)
     - スペルチェック (編集距離)
     - 自然言語処理 (CYK構文解析)
     - 文字列のアラインメント

  2. 機械学習
     - Viterbiアルゴリズム（隠れマルコフモデル）
     - CTC (Connectionist Temporal Classification)
     - ビームサーチ（近似的なDP）
     - 強化学習の価値反復法

  3. 最適化
     - リソース割り当て
     - スケジューリング
     - 在庫管理
     - 金融工学（オプション価格計算）

  4. ゲーム
     - ゲーム木の評価（ミニマックス + メモ化）
     - 最適戦略の計算
     - 勝敗判定

  5. バイオインフォマティクス
     - DNA/RNA配列のアラインメント
     - タンパク質の構造予測
     - 系統樹の構築

  6. 画像処理
     - シームカービング（画像のリサイズ）
     - ステレオマッチング
     - 文字認識（CTC）
```

### 5.2 実務的なDP最適化テクニック

```python
# 1. 空間最適化: 2行だけ使う（ローリング配列）
def lcs_rolling(s1, s2):
    m, n = len(s1), len(s2)
    if m < n:
        s1, s2 = s2, s1
        m, n = n, m

    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev, curr = curr, [0] * (n + 1)

    return prev[n]

# 2. 遷移の高速化: 累積和・セグメント木
# dp[i] = max(dp[j] + f(j, i)) for j < i
# → f(j, i)が加法的なら累積和で O(1) に
# → 範囲最大値なら セグメント木 で O(log n) に

# 3. Knuth's optimization
# dp[i][j] = min(dp[i][k] + dp[k+1][j] + C[i][j]) for k in [i, j)
# 分割点が単調なら O(n^3) → O(n^2)

# 4. Divide and Conquer DP
# dp[i] = min(dp[j] + C[j+1][i]) で C が Concave Monge 条件を満たす場合
# O(n^2) → O(n log n)

# 5. Convex Hull Trick（凸包テクニック）
def convex_hull_trick_example():
    """
    dp[i] = min(dp[j] + (a[i] - a[j])^2) の形の遷移を高速化
    O(n^2) → O(n) or O(n log n)

    考え方:
    dp[i] = min(dp[j] + a[i]^2 - 2*a[i]*a[j] + a[j]^2)
          = a[i]^2 + min(-2*a[j]*a[i] + (dp[j] + a[j]^2))

    y = mx + b の形:
    m = -2*a[j], b = dp[j] + a[j]^2, x = a[i]
    → 直線の集合の最小値 → 凸包で管理
    """
    pass

# 6. SOS DP (Sum over Subsets)
def sos_dp(values, n):
    """全ての部分集合の和を高速に計算
    dp[mask] = sum(values[sub]) for all sub ⊆ mask
    """
    dp = values[:]
    for bit in range(n):
        for mask in range(1 << n):
            if mask & (1 << bit):
                dp[mask] += dp[mask ^ (1 << bit)]
    return dp
# 計算量: O(n × 2^n) — 素朴にやると O(3^n)
```

### 5.3 DPのデバッグテクニック

```python
# 1. 小さいケースで手計算と照合
def debug_knapsack(weights, values, capacity):
    """DPテーブルを表示してデバッグ"""
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            dp[i][w] = dp[i-1][w]
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w], dp[i-1][w - weights[i-1]] + values[i-1])

    # テーブル表示
    print("DP Table:")
    print("     ", end="")
    for w in range(capacity + 1):
        print(f"w={w:2d} ", end="")
    print()

    for i in range(n + 1):
        if i == 0:
            print("none ", end="")
        else:
            print(f"i={i:2d} ", end="")
        for w in range(capacity + 1):
            print(f"{dp[i][w]:4d} ", end="")
        print()

    return dp[n][capacity]

# 2. ブルートフォースとの比較テスト
import random

def test_dp_correctness():
    """ランダムテストでDPの正しさを検証"""
    def brute_force_knapsack(weights, values, capacity):
        n = len(weights)
        best = 0
        for mask in range(1 << n):
            total_w = sum(weights[i] for i in range(n) if mask & (1 << i))
            total_v = sum(values[i] for i in range(n) if mask & (1 << i))
            if total_w <= capacity:
                best = max(best, total_v)
        return best

    for _ in range(1000):
        n = random.randint(1, 15)
        weights = [random.randint(1, 20) for _ in range(n)]
        values = [random.randint(1, 100) for _ in range(n)]
        capacity = random.randint(1, 50)

        dp_result = knapsack_optimized(weights, values, capacity)
        bf_result = brute_force_knapsack(weights, values, capacity)

        assert dp_result == bf_result, \
            f"Mismatch: w={weights}, v={values}, cap={capacity}"

    print("All tests passed!")

# 3. 遷移式の検証
# DPの遷移式が正しいか確認するチェックリスト:
# □ 基底条件は正しいか？
# □ 遷移の方向は正しいか？（依存する値が既に計算済みか？）
# □ 境界条件は正しいか？（配列の範囲外アクセスはないか？）
# □ 答えはどこにあるか？（dp[n]? max(dp)? dp[0]?）
# □ 逆順ループは正しいか？（0-1ナップサックの空間最適化）
```

---

## 6. よくあるDPの問題パターン集

### 6.1 階段の上り方問題

```python
def climb_stairs(n, steps=[1, 2]):
    """n段の階段をstepsで指定された歩幅で上る方法の総数"""
    dp = [0] * (n + 1)
    dp[0] = 1

    for i in range(1, n + 1):
        for step in steps:
            if i >= step:
                dp[i] += dp[i - step]

    return dp[n]

# 例: climb_stairs(5) = 8
# climb_stairs(5, [1, 2, 3]) = 13

# 変形: コストを考慮した階段問題
def min_cost_climbing(cost):
    """各段にコストがあり、最小コストで頂上に到達する"""
    n = len(cost)
    dp = [0] * (n + 1)

    for i in range(2, n + 1):
        dp[i] = min(dp[i-1] + cost[i-1], dp[i-2] + cost[i-2])

    return dp[n]
```

### 6.2 経路数の問題

```python
def unique_paths(m, n):
    """m×nグリッドの左上から右下への経路数（右と下のみ）"""
    dp = [[1] * n for _ in range(m)]

    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]

    return dp[m-1][n-1]

# 障害物ありの場合
def unique_paths_with_obstacles(grid):
    """障害物(1)がある場合の経路数"""
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]

    # 最初のセルが障害物なら0
    dp[0][0] = 1 if grid[0][0] == 0 else 0

    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                dp[i][j] = 0
                continue
            if i > 0:
                dp[i][j] += dp[i-1][j]
            if j > 0:
                dp[i][j] += dp[i][j-1]

    return dp[m-1][n-1]

# 最小コストパス
def min_path_sum(grid):
    """左上から右下への最小コストパス"""
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = grid[0][0]

    for j in range(1, n):
        dp[0][j] = dp[0][j-1] + grid[0][j]
    for i in range(1, m):
        dp[i][0] = dp[i-1][0] + grid[i][0]

    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])

    return dp[m-1][n-1]
```

### 6.3 文字列のインターリーブ

```python
def is_interleave(s1, s2, s3):
    """s3がs1とs2のインターリーブ（交互織り）かどうか"""
    m, n = len(s1), len(s2)
    if m + n != len(s3):
        return False

    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True

    for i in range(1, m + 1):
        dp[i][0] = dp[i-1][0] and s1[i-1] == s3[i-1]
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j-1] and s2[j-1] == s3[j-1]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = (
                (dp[i-1][j] and s1[i-1] == s3[i+j-1]) or
                (dp[i][j-1] and s2[j-1] == s3[i+j-1])
            )

    return dp[m][n]

# 例: s1="aab", s2="axy", s3="aaxaby" → True
# "aaxaby" = a(s1) + a(s2) + x(s2) + a(s1) + b(s1) + y(s2)
```

### 6.4 株の売買問題

```python
def max_profit_k_transactions(prices, k):
    """最大k回の取引で得られる最大利益"""
    n = len(prices)
    if n <= 1:
        return 0

    # k >= n//2 なら制限なし
    if k >= n // 2:
        return sum(max(0, prices[i+1] - prices[i]) for i in range(n-1))

    # dp[t][i] = t回目までの取引でi日目までの最大利益
    dp = [[0] * n for _ in range(k + 1)]

    for t in range(1, k + 1):
        max_diff = -prices[0]
        for i in range(1, n):
            dp[t][i] = max(dp[t][i-1], prices[i] + max_diff)
            max_diff = max(max_diff, dp[t-1][i] - prices[i])

    return dp[k][n-1]

# 計算量: O(k × n)

# 変形: クールダウン期間付き
def max_profit_with_cooldown(prices):
    """売った翌日は買えない（1日のクールダウン）"""
    n = len(prices)
    if n <= 1:
        return 0

    # hold: 株を持っている状態の最大利益
    # sold: 今日売った状態の最大利益
    # rest: 何も持っていない&クールダウン中/待機中の最大利益
    hold = -prices[0]
    sold = 0
    rest = 0

    for i in range(1, n):
        prev_hold = hold
        hold = max(hold, rest - prices[i])
        rest = max(rest, sold)
        sold = prev_hold + prices[i]

    return max(sold, rest)

# 変形: 取引手数料付き
def max_profit_with_fee(prices, fee):
    """各取引に手数料がかかる"""
    n = len(prices)
    hold = -prices[0]  # 株を持っている
    cash = 0           # 株を持っていない

    for i in range(1, n):
        hold = max(hold, cash - prices[i])
        cash = max(cash, hold + prices[i] - fee)

    return cash
```

---

## 7. 実践演習

### 演習1: 基本DP（基礎）
階段の上り方問題: N段の階段を1段または2段ずつ上る方法の総数を求めよ。

### 演習2: 2次元DP（応用）
編集距離（レーベンシュタイン距離）を計算する関数を実装せよ。"kitten" → "sitting" の編集距離を求めよ。

### 演習3: ビットDP（発展）
巡回セールスマン問題（TSP）を、ビットマスクDPで O(n^2 x 2^n) で解くプログラムを実装せよ。

### 演習4: 区間DP（発展）
行列の連鎖積問題を解き、最適な括弧付けを復元せよ。

### 演習5: 実務応用（発展）
LCSを使ったdiffツールを実装し、2つのテキストファイルの差分を表示せよ。

### 演習6: 桁DP（発展）
1からNまでの整数において、各数字(0-9)が何回出現するかを数えよ。

### 演習7: ゲーム理論DP（発展）
石取りゲーム（左端か右端から石を取り、合計を最大化する）の最適戦略を求めよ。

### 演習8: 確率DP（発展）
すごろくで、6面サイコロを振ってゴール（マスN）にちょうど止まるまでの期待試行回数を計算せよ。

---

## FAQ

### Q1: DPと分割統治の違いは？
**A**: 分割統治は部分問題が独立（マージソート: 左右が独立）。DPは部分問題が重複（フィボナッチ: fib(n-1)とfib(n-2)がfib(n-3)を共有）。分割統治 + メモ化 = DP とも言える。

### Q2: DPの状態設計のコツは？
**A**: まず「何を決めたら残りが決まるか」を考える。状態は「残りの問題を解くのに必要な最小限の情報」。状態が多すぎたら次元削減を検討。まずは素朴なDPを書いてから最適化する。

### Q3: 貪欲法とDPの使い分けは？
**A**: 貪欲法は局所最適が全体最適になる場合のみ有効（証明が必要）。DPは全ての選択肢を考慮するので常に正しい。迷ったらDPが安全。貪欲法が使えるなら計算量は小さい。

### Q4: メモ化再帰とボトムアップ、どちらを使うべき？
**A**: 一般的にはメモ化再帰の方が書きやすく、不要な部分問題を計算しない利点がある。ただしPythonでは再帰の深さ制限（デフォルト1000）があるため、大きな入力ではボトムアップが安全。空間最適化もボトムアップの方がやりやすい。競技プログラミングではボトムアップが主流。

### Q5: DPのテーブルサイズが大きすぎる場合はどうする？
**A**: (1) 空間最適化（ローリング配列） (2) 状態の再定義（次元削減） (3) メモ化再帰で必要な状態のみ計算 (4) ビットマスクの圧縮 (5) ハッシュマップベースのメモ化。それでも無理なら問題を近似解法に切り替える。

### Q6: 「DPを知っていると実務のどこで役立つか」を具体的に教えてください
**A**: (1) テキストエディタのdiff機能（LCS）はバージョン管理の根幹技術 (2) スペルチェッカーの候補提示（編集距離） (3) ネットワークルーティング（最短経路のDP的アプローチ） (4) 機械学習のViterbiアルゴリズム（音声認識・品詞タグ付け） (5) データベースのクエリ最適化（結合順序の決定） (6) コンパイラの命令選択（木DPの一種）

### Q7: 計算量が擬多項式（pseudo-polynomial）とはどういう意味ですか？
**A**: ナップサック問題の O(nW) は入力サイズではなく入力の値（W）に依存する。Wをバイナリ表現すると log W ビットなので、入力のビット数に対しては指数時間。これを「擬多項式時間」と呼ぶ。NP困難な問題でも入力値が小さければ実用的に高速に解ける場合がある。

---

## まとめ

| 概念 | ポイント |
|------|---------|
| DP条件 | 最適部分構造 + 重複部分問題 |
| メモ化 | トップダウン。再帰+キャッシュ |
| ボトムアップ | テーブル法。forループ+配列 |
| 典型パターン | ナップサック、LCS、コイン、LIS、編集距離 |
| 状態設計 | 「残りを解くのに必要な最小限の情報」を状態にする |
| 空間最適化 | ローリング配列、1次元化 |
| 高度なDP | 区間DP、木DP、桁DP、ビットDP、確率DP |
| 遷移高速化 | 累積和、Convex Hull Trick、Knuth最適化 |
| デバッグ | 小ケース手計算、ブルートフォース比較、テーブル表示 |

---

## 次に読むべきガイド
→ [[06-greedy-and-backtracking.md]] -- 貪欲法とバックトラック

---

## 参考文献
1. Cormen, T. H. et al. "Introduction to Algorithms." Chapter 15: Dynamic Programming.
2. Bellman, R. "Dynamic Programming." Princeton University Press, 1957.
3. Skiena, S. S. "The Algorithm Design Manual." Chapter 10.
4. Halim, S. "Competitive Programming 3." Chapter 3.5: Dynamic Programming.
5. Knuth, D. E. "The Art of Computer Programming, Volume 3." Sorting and Searching.
6. Dasgupta, S., Papadimitriou, C., Vazirani, U. "Algorithms." Chapter 6: Dynamic Programming.
