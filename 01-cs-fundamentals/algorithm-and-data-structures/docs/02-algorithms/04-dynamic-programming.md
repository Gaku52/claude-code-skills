# 動的計画法（Dynamic Programming）

> 重複する部分問題を効率的に解くための設計手法を、メモ化・ボトムアップ・代表的問題を通じて体系的に理解する

## この章で学ぶこと

1. **メモ化（トップダウン）とボトムアップ（テーブル法）**の2つのアプローチを使い分けられる
2. **最適部分構造と重複部分問題**という DP が適用できる条件を見抜ける
3. **ナップサック問題・LCS・LIS・編集距離・区間DP・ビットDP** を正確に実装できる
4. **DP の状態設計・遷移式導出・空間最適化**のフレームワークを使いこなせる

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

### DP が適用できる問題の見分け方

```
以下のキーワードが問題文に含まれていたら DP を疑う:

  - 「最大」「最小」「最長」「最短」「最適」
  - 「方法の数」「場合の数」「組み合わせ数」
  - 「可能かどうか」（Yes/No の判定問題）
  - 「部分列」「部分文字列」「部分集合」
  - 「コスト最小化」「利益最大化」

判断フロー:
  1. 再帰的に解ける? → YES なら次へ
  2. 部分問題が重複する? → YES なら DP
  3. 重複しない? → 分割統治法（メモ化不要）
  4. 局所最適 = 全体最適? → 貪欲法を先に検討
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

# 方法3: 汎用メモ化デコレータ
def memoize(func):
    """汎用メモ化デコレータ（hashable な引数に対応）"""
    cache = {}
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    wrapper.cache = cache
    return wrapper

@memoize
def fib_memoized(n: int) -> int:
    if n <= 1:
        return n
    return fib_memoized(n - 1) + fib_memoized(n - 2)

print(fib_memo(50))      # 12586269025
print(fib_cached(50))    # 12586269025
print(fib_memoized(50))  # 12586269025
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

答え: dp[3][8] = 10
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

# 選択されたアイテムの復元
def knapsack_01_with_items(weights: list, values: list, W: int) -> tuple:
    """0/1 ナップサック + 選択アイテムの復元"""
    n = len(weights)
    dp = [[0] * (W + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(W + 1):
            dp[i][w] = dp[i - 1][w]
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i][w],
                               dp[i - 1][w - weights[i - 1]] + values[i - 1])

    # 選択アイテムの復元（バックトラック）
    selected = []
    w = W
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected.append(i - 1)  # アイテム i-1 を選択
            w -= weights[i - 1]

    return dp[n][W], selected[::-1]

weights = [2, 3, 4, 5]
values = [3, 4, 5, 6]
print(knapsack_01(weights, values, 8))            # 10
print(knapsack_01_optimized(weights, values, 8))  # 10
max_val, items = knapsack_01_with_items(weights, values, 8)
print(f"最大価値: {max_val}, 選択: {items}")       # 最大価値: 10, 選択: [0, 1, 2]
```

### 完全ナップサック（各アイテム無制限使用可）

```python
def knapsack_unbounded(weights: list, values: list, W: int) -> int:
    """完全ナップサック - O(nW)
    各アイテムを何個でも使える
    """
    dp = [0] * (W + 1)

    for i in range(len(weights)):
        # 順方向に更新（同じアイテムを何度でも使える）
        for w in range(weights[i], W + 1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])

    return dp[W]

# 0/1 は逆順、完全は順方向 — この違いが重要!
print(knapsack_unbounded([2, 3, 4, 5], [3, 4, 5, 6], 8))  # 12 (重さ2×4)
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

遷移式:
  X[i] == Y[j] の場合: dp[i][j] = dp[i-1][j-1] + 1
  X[i] != Y[j] の場合: dp[i][j] = max(dp[i-1][j], dp[i][j-1])
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

# 空間最適化版（長さのみ）
def lcs_length_optimized(X: str, Y: str) -> int:
    """LCS の長さのみを求める - O(mn) 時間、O(min(m,n)) 空間"""
    if len(X) < len(Y):
        X, Y = Y, X  # 短い方を Y にする

    m, n = len(X), len(Y)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)

    return prev[n]

length, subseq = lcs("ABCBDAB", "BDCAB")
print(f"長さ: {length}, LCS: {subseq}")  # 長さ: 4, LCS: BCAB
```

### LCS の実務応用: diff の計算

```python
def compute_diff(original: list, modified: list) -> list:
    """LCS を使って2つのテキストの差分を計算"""
    m, n = len(original), len(modified)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if original[i-1] == modified[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    # 差分の生成
    diff = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and original[i-1] == modified[j-1]:
            diff.append(('  ', original[i-1]))  # 変更なし
            i -= 1
            j -= 1
        elif j > 0 and (i == 0 or dp[i][j-1] >= dp[i-1][j]):
            diff.append(('+ ', modified[j-1]))   # 追加
            j -= 1
        else:
            diff.append(('- ', original[i-1]))   # 削除
            i -= 1

    return diff[::-1]

original = ["def hello():", "    print('hello')", "    return True"]
modified = ["def hello():", "    print('hello, world')", "    return True", "    # comment"]
for prefix, line in compute_diff(original, modified):
    print(f"{prefix}{line}")
```

---

## 6. コイン問題（最小枚数）

```python
def coin_change(coins: list, amount: int) -> int:
    """コイン問題（最小枚数）- O(n * amount)"""
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i and dp[i - coin] + 1 < dp[i]:
                dp[i] = dp[i - coin] + 1

    return dp[amount] if dp[amount] != float('inf') else -1

def coin_change_ways(coins: list, amount: int) -> int:
    """コイン問題（場合の数）- O(n * amount)"""
    dp = [0] * (amount + 1)
    dp[0] = 1

    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]

    return dp[amount]

# 使用例
print(coin_change([1, 5, 10, 25], 30))       # 2 (25+5)
print(coin_change([3, 7], 5))                  # -1 (不可能)
print(coin_change_ways([1, 5, 10, 25], 30))   # 18通り
```

---

## 7. 最長増加部分列（LIS）

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

def lis_with_reconstruction(arr: list) -> tuple:
    """LIS + 実際の部分列の復元 - O(n log n)"""
    n = len(arr)
    if n == 0:
        return 0, []

    tails = []
    tails_idx = []      # tails の各位置に対応する元配列のインデックス
    prev_idx = [-1] * n  # 各要素の LIS 内での前の要素のインデックス

    for i, num in enumerate(arr):
        pos = bisect.bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
            tails_idx.append(i)
        else:
            tails[pos] = num
            tails_idx[pos] = i

        if pos > 0:
            prev_idx[i] = tails_idx[pos - 1]

    # 復元
    length = len(tails)
    result = []
    idx = tails_idx[-1]
    while idx != -1:
        result.append(arr[idx])
        idx = prev_idx[idx]

    return length, result[::-1]

data = [10, 9, 2, 5, 3, 7, 101, 18]
print(lis_dp(data))                          # 4
print(lis_binary_search(data))               # 4
length, subseq = lis_with_reconstruction(data)
print(f"長さ: {length}, LIS: {subseq}")     # 長さ: 4, LIS: [2, 3, 7, 18]
```

---

## 8. 編集距離（レーベンシュタイン距離）

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

def edit_distance_with_operations(s1: str, s2: str) -> tuple:
    """編集距離 + 操作列の復元"""
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
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    # 操作列の復元
    operations = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s1[i-1] == s2[j-1]:
            operations.append(('keep', s1[i-1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            operations.append(('replace', s1[i-1], s2[j-1]))
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            operations.append(('insert', s2[j-1]))
            j -= 1
        else:
            operations.append(('delete', s1[i-1]))
            i -= 1

    return dp[m][n], operations[::-1]

print(edit_distance("kitten", "sitting"))  # 3
dist, ops = edit_distance_with_operations("kitten", "sitting")
print(f"距離: {dist}")
for op in ops:
    print(f"  {op}")
# ('replace', 'k', 's'), ('keep', 'i'), ('keep', 't'), ('keep', 't'),
# ('replace', 'e', 'i'), ('keep', 'n'), ('insert', 'g')
```

### 編集距離の実務応用: あいまい検索

```python
def fuzzy_search(query: str, dictionary: list, max_distance: int = 2) -> list:
    """あいまい検索: 編集距離が閾値以内の単語を返す"""
    results = []
    for word in dictionary:
        dist = edit_distance(query.lower(), word.lower())
        if dist <= max_distance:
            results.append((word, dist))
    results.sort(key=lambda x: x[1])
    return results

dictionary = ["python", "pytorch", "pycharm", "piton", "prism", "prison"]
print(fuzzy_search("pyton", dictionary))
# [('piton', 1), ('python', 1), ('prism', 2), ('prison', 2)]
```

---

## 9. 区間DP

区間 [l, r] に関する最適解を、より小さな区間の解から求める手法。

### 行列連鎖積問題

```python
def matrix_chain_order(dims: list) -> tuple:
    """行列連鎖積の最小乗算回数 - O(n³)
    dims: 行列の次元リスト（n+1個の要素）
    行列 A_i は dims[i] × dims[i+1]
    """
    n = len(dims) - 1
    dp = [[0] * n for _ in range(n)]
    split = [[0] * n for _ in range(n)]

    # l: 区間の長さ（2以上）
    for l in range(2, n + 1):
        for i in range(n - l + 1):
            j = i + l - 1
            dp[i][j] = float('inf')
            for k in range(i, j):
                cost = dp[i][k] + dp[k+1][j] + dims[i] * dims[k+1] * dims[j+1]
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    split[i][j] = k

    return dp[0][n-1], split

def print_optimal_parens(split: list, i: int, j: int) -> str:
    """最適な括弧付けを出力"""
    if i == j:
        return f"A{i}"
    k = split[i][j]
    left = print_optimal_parens(split, i, k)
    right = print_optimal_parens(split, k + 1, j)
    return f"({left} × {right})"

# 行列: A0(30×35), A1(35×15), A2(15×5), A3(5×10), A4(10×20), A5(20×25)
dims = [30, 35, 15, 5, 10, 20, 25]
min_ops, split = matrix_chain_order(dims)
print(f"最小乗算回数: {min_ops}")  # 15125
print(f"最適括弧: {print_optimal_parens(split, 0, 5)}")
# ((A0 × (A1 × A2)) × ((A3 × A4) × A5))
```

### 回文分割

```python
def min_palindrome_cuts(s: str) -> int:
    """文字列を回文に分割する最小カット数 - O(n²)"""
    n = len(s)
    if n <= 1:
        return 0

    # is_pal[i][j] = s[i..j] が回文か
    is_pal = [[False] * n for _ in range(n)]
    for i in range(n):
        is_pal[i][i] = True
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if length == 2:
                is_pal[i][j] = (s[i] == s[j])
            else:
                is_pal[i][j] = (s[i] == s[j] and is_pal[i+1][j-1])

    # dp[i] = s[0..i] を回文に分割する最小カット数
    dp = list(range(n))  # 最悪: 1文字ずつ分割
    for i in range(1, n):
        if is_pal[0][i]:
            dp[i] = 0
            continue
        for j in range(1, i + 1):
            if is_pal[j][i]:
                dp[i] = min(dp[i], dp[j-1] + 1)

    return dp[n-1]

print(min_palindrome_cuts("aab"))       # 1 ("aa" + "b")
print(min_palindrome_cuts("abcba"))     # 0 (全体が回文)
print(min_palindrome_cuts("abcdef"))    # 5 (各文字で分割)
```

---

## 10. ビットDP

状態を整数のビットで表現し、部分集合を効率的に管理する手法。

### 巡回セールスマン問題（TSP）

```python
def tsp(dist_matrix: list) -> tuple:
    """巡回セールスマン問題 - O(2^n * n²)
    dist_matrix[i][j]: 都市 i から j への距離
    返り値: (最小距離, 経路)
    """
    n = len(dist_matrix)
    INF = float('inf')

    # dp[S][v] = 集合 S の都市を訪問し、現在 v にいるときの最小距離
    # S はビットマスク: bit i が立つ = 都市 i を訪問済み
    dp = [[INF] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]
    dp[1][0] = 0  # 都市 0 から出発

    for S in range(1 << n):
        for u in range(n):
            if dp[S][u] == INF:
                continue
            if not (S & (1 << u)):
                continue
            for v in range(n):
                if S & (1 << v):
                    continue  # 既に訪問済み
                new_S = S | (1 << v)
                new_dist = dp[S][u] + dist_matrix[u][v]
                if new_dist < dp[new_S][v]:
                    dp[new_S][v] = new_dist
                    parent[new_S][v] = u

    # 全都市訪問後、出発点に戻る
    full = (1 << n) - 1
    min_dist = INF
    last = -1
    for v in range(n):
        total = dp[full][v] + dist_matrix[v][0]
        if total < min_dist:
            min_dist = total
            last = v

    # 経路復元
    path = [0]
    S = full
    v = last
    while v != 0:
        path.append(v)
        u = parent[S][v]
        S ^= (1 << v)
        v = u
    path.append(0)
    path.reverse()

    return min_dist, path

# 4都市の例
dist_matrix = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0],
]
min_dist, path = tsp(dist_matrix)
print(f"最短巡回距離: {min_dist}")  # 80
print(f"経路: {path}")              # [0, 1, 3, 2, 0]
```

### ビットDP: 集合に対する最適割り当て

```python
def min_cost_assignment(cost: list) -> int:
    """最小コスト割り当て問題 - O(2^n * n)
    cost[i][j]: 人 i にタスク j を割り当てるコスト
    各人に1つのタスクを割り当て、全タスクをカバー
    """
    n = len(cost)
    INF = float('inf')
    dp = [INF] * (1 << n)
    dp[0] = 0

    for mask in range(1 << n):
        person = bin(mask).count('1')  # 何人目まで割り当てたか
        if person >= n:
            continue
        for task in range(n):
            if mask & (1 << task):
                continue  # このタスクは割り当て済み
            new_mask = mask | (1 << task)
            dp[new_mask] = min(dp[new_mask], dp[mask] + cost[person][task])

    return dp[(1 << n) - 1]

cost_matrix = [
    [9, 2, 7, 8],
    [6, 4, 3, 7],
    [5, 8, 1, 8],
    [7, 6, 9, 4],
]
print(min_cost_assignment(cost_matrix))  # 13 (2+3+1+7? or 2+4+1+4=11)
```

---

## 11. 木DP

木構造のグラフに対する DP。各頂点の値を子の値から計算する。

```python
def tree_dp_max_independent_set(tree: dict, root: int) -> int:
    """木の最大独立集合のサイズ - O(V)
    独立集合: 隣接する頂点を含まない頂点部分集合
    tree: {node: [children]}
    """
    # dp[v][0] = v を含まない場合の部分木の最大独立集合サイズ
    # dp[v][1] = v を含む場合の部分木の最大独立集合サイズ
    dp = {}

    def dfs(v, parent):
        dp[v] = [0, 1]  # [含まない, 含む]

        for child in tree.get(v, []):
            if child == parent:
                continue
            dfs(child, v)
            dp[v][0] += max(dp[child][0], dp[child][1])  # 子は含んでも含まなくても良い
            dp[v][1] += dp[child][0]  # v を含むなら子は含まない

    dfs(root, -1)
    return max(dp[root][0], dp[root][1])

# 木:      1
#         / \
#        2   3
#       / \
#      4   5
tree = {1: [2, 3], 2: [1, 4, 5], 3: [1], 4: [2], 5: [2]}
print(tree_dp_max_independent_set(tree, 1))  # 3 (頂点 3, 4, 5)


def tree_diameter(tree: dict, root: int) -> int:
    """木の直径（最長パスの長さ）- O(V)"""
    diameter = [0]

    def dfs(v, parent) -> int:
        """v の部分木における最長の根からの距離を返す"""
        max1 = max2 = 0  # 最大と2番目の最大

        for child in tree.get(v, []):
            if child == parent:
                continue
            depth = dfs(child, v) + 1
            if depth > max1:
                max2 = max1
                max1 = depth
            elif depth > max2:
                max2 = depth

        diameter[0] = max(diameter[0], max1 + max2)
        return max1

    dfs(root, -1)
    return diameter[0]

print(tree_diameter(tree, 1))  # 3 (4→2→1→3 or 5→2→1→3)
```

---

## 12. 確率DP・期待値DP

```python
def expected_coin_flips(target_heads: int) -> float:
    """公平なコインを投げて、target_heads 回表が出るまでの期待投げ回数
    dp[i] = あと i 回表を出すのに必要な期待投げ回数
    """
    dp = [0.0] * (target_heads + 1)
    for i in range(1, target_heads + 1):
        # 表: dp[i-1] に遷移（確率 1/2）
        # 裏: dp[i] に遷移（確率 1/2）→ 1回無駄
        # dp[i] = 1 + 0.5 * dp[i-1] + 0.5 * dp[i]
        # → dp[i] = 2 + dp[i-1]
        dp[i] = 2 + dp[i - 1]
    return dp[target_heads]

print(expected_coin_flips(3))  # 6.0（3回の表を出すのに平均6回投げる）


def dice_probability(n_dice: int, target: int) -> float:
    """n個のサイコロの目の合計が target になる確率"""
    # dp[i][j] = i個のサイコロで合計 j になる場合の数
    dp = [[0] * (target + 1) for _ in range(n_dice + 1)]
    dp[0][0] = 1

    for i in range(1, n_dice + 1):
        for j in range(i, min(6 * i, target) + 1):
            for face in range(1, 7):
                if j - face >= 0:
                    dp[i][j] += dp[i-1][j-face]

    total_outcomes = 6 ** n_dice
    return dp[n_dice][target] / total_outcomes if target <= 6 * n_dice else 0

print(f"2個のサイコロで7: {dice_probability(2, 7):.4f}")  # 0.1667
print(f"3個のサイコロで10: {dice_probability(3, 10):.4f}")  # 0.1250
```

---

## 13. DP 設計フレームワーク

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

### 設計例: 階段の上り方問題

```python
# 問題: n段の階段を1段または2段ずつ上る方法は何通り?
#
# Step 1. 状態の定義: dp[i] = i段目に到達する方法の数
# Step 2. 遷移式:     dp[i] = dp[i-1] + dp[i-2]
#                     （1段前から1段 or 2段前から2段）
# Step 3. 基底条件:   dp[0] = 1（地面にいる: 1通り）
#                     dp[1] = 1（1段目: 1通り）
# Step 4. 計算順序:   i = 2, 3, ..., n（小→大）
# Step 5. 答え:       dp[n]

def climb_stairs(n: int) -> int:
    if n <= 1:
        return 1
    dp = [0] * (n + 1)
    dp[0] = dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]

print(climb_stairs(10))  # 89
```

---

## 14. メモ化 vs ボトムアップ 比較表

| 特性 | メモ化（トップダウン） | ボトムアップ |
|:---|:---|:---|
| 実装スタイル | 再帰 + キャッシュ | ループ + テーブル |
| 計算する部分問題 | 必要な分だけ | 全ての部分問題 |
| スタックオーバーフロー | 起こりうる | 起こらない |
| 空間最適化 | 困難 | 可能（次元削減） |
| コーディングの容易さ | 再帰的思考が自然 | 遷移順序を考える必要 |
| デバッグ | やや困難 | テーブルを確認しやすい |
| 定数係数 | 関数呼び出しオーバーヘッド | ループなので高速 |

## 典型DPパターン

| パターン | 代表問題 | 状態 | 計算量 |
|:---|:---|:---|:---|
| 1次元 DP | フィボナッチ、階段 | dp[i] | O(n) |
| 2次元 DP | LCS、編集距離 | dp[i][j] | O(mn) |
| ナップサック | 0/1ナップサック | dp[i][w] | O(nW) |
| 区間 DP | 行列連鎖積 | dp[l][r] | O(n^3) |
| ビット DP | 巡回セールスマン | dp[S][v] | O(2^n * n) |
| 木 DP | 木上の最大独立集合 | dp[v][0/1] | O(V) |
| 確率 DP | 期待値計算 | dp[state] | 問題依存 |
| 桁 DP | N以下の条件を満たす数の個数 | dp[pos][tight][...] | O(D * states) |

---

## 15. 桁DP

N 以下で特定の条件を満たす整数の個数を数える。

```python
def count_numbers_with_digit_sum(N: int, target_sum: int) -> int:
    """N 以下の非負整数で、各桁の和が target_sum となるものの個数"""
    digits = [int(d) for d in str(N)]
    n = len(digits)

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def dp(pos, remaining_sum, tight, started):
        """
        pos: 現在の桁位置
        remaining_sum: 残りの桁和
        tight: 上限制約があるか
        started: 先頭のゼロを過ぎたか
        """
        if remaining_sum < 0:
            return 0
        if pos == n:
            return 1 if remaining_sum == 0 and started else 0

        limit = digits[pos] if tight else 9
        count = 0

        for d in range(0, limit + 1):
            count += dp(
                pos + 1,
                remaining_sum - d,
                tight and (d == limit),
                started or (d > 0),
            )

        return count

    return dp(0, target_sum, True, False)

# 1000以下で桁和が10の数の個数
print(count_numbers_with_digit_sum(1000, 10))  # 63
```

---

## 16. アンチパターン

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

### アンチパターン4: 不要な次元を持つ状態設計

```python
# BAD: ナップサックで3次元（アイテム × 容量 × 選択数）
# → 選択数は不要な場合が多い

# GOOD: 必要最小限の次元で設計
# 0/1ナップサックなら dp[w] の1次元で十分（空間最適化後）
```

### アンチパターン5: 浮動小数点の DP

```python
# BAD: 浮動小数点をキーにする → 精度問題
memo = {}
def bad_dp(x):
    if x in memo:  # 0.1 + 0.2 != 0.3 問題
        return memo[x]
    ...

# GOOD: 整数に変換するか、適切な丸めを行う
def good_dp(x_cents):  # セント単位の整数
    ...
```

---

## 17. FAQ

### Q1: DP と分割統治法の違いは？

**A:** 両方とも問題を分割して解くが、核心的な違いは「部分問題の重複」。分割統治法（マージソート等）は部分問題が独立しており重複しない。DP は同じ部分問題が何度も現れるため、結果をキャッシュして再利用する。重複がなければ分割統治、あれば DP を使う。

### Q2: DP の次元（状態数）をどう決める？

**A:** 問題を一意に表現するために必要な最小限のパラメータ数が次元になる。フィボナッチは n の1つ（1次元）、LCS は 2文字列の位置 i,j の2つ（2次元）。状態を増やすと表現力は上がるが計算量も増えるため、必要十分な次元を見極める。

### Q3: メモ化再帰でスタックオーバーフローが起きたら？

**A:** 3つの対策がある。(1) `sys.setrecursionlimit()` を増やす（応急処置）。(2) ボトムアップ DP に書き換える（推奨）。(3) 末尾再帰最適化が可能な場合はループに変換する。Python では (2) が最も安全。

### Q4: DP テーブルのデバッグ方法は？

**A:** 小さな入力でテーブルを手計算し、プログラムの出力と照合する。2次元DPなら `for row in dp: print(row)` でテーブル全体を表示。遷移式が正しいか、基底条件が正しいか、計算順序が正しいかを順にチェックする。

### Q5: DP の計算量を改善するには？

**A:** (1) 状態数の削減（不要な次元の除去）。(2) 遷移の高速化（単調性やConvex Hull Trickの利用）。(3) 空間最適化（前の行/列のみ保持）。(4) 行列累乗による高速化（線形漸化式の場合）。

---

## 18. まとめ

| 項目 | 要点 |
|:---|:---|
| DP の2条件 | 最適部分構造 + 重複部分問題 |
| メモ化 | トップダウン、再帰+キャッシュ、必要分だけ計算 |
| ボトムアップ | テーブル法、ループ、空間最適化が可能 |
| ナップサック | 0/1 は逆方向更新、完全は順方向更新 |
| LCS | 2次元 DP の代表問題。diff/スペルチェックに応用 |
| 区間DP | dp[l][r] で区間を管理。行列連鎖積が代表例 |
| ビットDP | ビットマスクで集合を表現。TSP が代表例 |
| 木DP | 子の結果から親の値を計算 |
| 設計手順 | 状態定義→遷移式→基底条件→計算順序→答え抽出 |

---

## 19. 実務応用パターン集

### 19.1 テキストエディタの自動補完（編集距離ベース）

```python
def autocomplete_with_edit_distance(prefix: str, dictionary: list, max_suggestions: int = 5) -> list:
    """編集距離ベースの自動補完候補を返す"""
    candidates = []

    for word in dictionary:
        # 接頭辞との編集距離を計算（単語の先頭部分のみ比較）
        min_len = min(len(prefix), len(word))
        partial_dist = edit_distance(prefix, word[:min_len])

        # 完全一致の接頭辞は最優先
        if word.startswith(prefix):
            candidates.append((word, 0))
        else:
            candidates.append((word, partial_dist))

    candidates.sort(key=lambda x: (x[1], len(x[0])))
    return [word for word, _ in candidates[:max_suggestions]]
```

### 19.2 DNA配列のアラインメント

```python
def sequence_alignment(seq1: str, seq2: str,
                       match_score: int = 2,
                       mismatch_penalty: int = -1,
                       gap_penalty: int = -2) -> tuple:
    """Needleman-Wunsch アルゴリズム（グローバルアラインメント）
    DNA/タンパク質配列の比較に使用
    """
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i * gap_penalty
    for j in range(n + 1):
        dp[0][j] = j * gap_penalty

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = dp[i-1][j-1] + (match_score if seq1[i-1] == seq2[j-1] else mismatch_penalty)
            delete = dp[i-1][j] + gap_penalty
            insert = dp[i][j-1] + gap_penalty
            dp[i][j] = max(match, delete, insert)

    # アラインメントの復元
    align1, align2 = [], []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            score = match_score if seq1[i-1] == seq2[j-1] else mismatch_penalty
            if dp[i][j] == dp[i-1][j-1] + score:
                align1.append(seq1[i-1])
                align2.append(seq2[j-1])
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i-1][j] + gap_penalty:
            align1.append(seq1[i-1])
            align2.append('-')
            i -= 1
        else:
            align1.append('-')
            align2.append(seq2[j-1])
            j -= 1

    return dp[m][n], ''.join(reversed(align1)), ''.join(reversed(align2))

score, a1, a2 = sequence_alignment("AGTACG", "ACATAG")
print(f"スコア: {score}")
print(f"配列1: {a1}")
print(f"配列2: {a2}")
```

### 19.3 正規表現マッチング

```python
def regex_match(text: str, pattern: str) -> bool:
    """正規表現マッチング（'.' は任意1文字、'*' は直前の文字の0回以上の繰り返し）
    LeetCode #10 相当
    """
    m, n = len(text), len(pattern)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True

    # パターンの先頭が "a*b*c*" のような場合の初期化
    for j in range(2, n + 1):
        if pattern[j - 1] == '*':
            dp[0][j] = dp[0][j - 2]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pattern[j - 1] == '*':
                # '*' の0回マッチ
                dp[i][j] = dp[i][j - 2]
                # '*' の1回以上マッチ
                if pattern[j - 2] == '.' or pattern[j - 2] == text[i - 1]:
                    dp[i][j] = dp[i][j] or dp[i - 1][j]
            elif pattern[j - 1] == '.' or pattern[j - 1] == text[i - 1]:
                dp[i][j] = dp[i - 1][j - 1]

    return dp[m][n]

print(regex_match("aab", "c*a*b"))     # True
print(regex_match("mississippi", "mis*is*p*."))  # False
print(regex_match("ab", ".*"))          # True
```

### 19.4 株式売買の最大利益

```python
def max_profit_k_transactions(prices: list, k: int) -> int:
    """最大 k 回の売買で得られる最大利益 - O(nk)
    dp[j][0] = j回目の取引で株を持っていない状態の最大利益
    dp[j][1] = j回目の取引で株を持っている状態の最大利益
    """
    n = len(prices)
    if n <= 1 or k <= 0:
        return 0

    # k が十分大きい場合は無制限売買
    if k >= n // 2:
        return sum(max(prices[i+1] - prices[i], 0) for i in range(n - 1))

    dp = [[[0, 0] for _ in range(k + 1)] for _ in range(n)]

    for j in range(k + 1):
        dp[0][j][1] = -prices[0]

    for i in range(1, n):
        for j in range(1, k + 1):
            dp[i][j][0] = max(dp[i-1][j][0], dp[i-1][j][1] + prices[i])
            dp[i][j][1] = max(dp[i-1][j][1], dp[i-1][j-1][0] - prices[i])

    return max(dp[n-1][j][0] for j in range(k + 1))

prices = [3, 2, 6, 5, 0, 3]
print(max_profit_k_transactions(prices, 2))  # 7 (2で買い6で売り+0で買い3で売り)
```

### 19.5 最長回文部分文字列

```python
def longest_palindrome_substring(s: str) -> str:
    """最長回文部分文字列 - O(n²)
    dp[i][j] = s[i..j] が回文かどうか
    """
    n = len(s)
    if n < 2:
        return s

    dp = [[False] * n for _ in range(n)]
    start = 0
    max_len = 1

    # 長さ1は全て回文
    for i in range(n):
        dp[i][i] = True

    # 長さ2
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            dp[i][i + 1] = True
            start = i
            max_len = 2

    # 長さ3以上
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j] and dp[i + 1][j - 1]:
                dp[i][j] = True
                if length > max_len:
                    start = i
                    max_len = length

    return s[start:start + max_len]

print(longest_palindrome_substring("babad"))    # "bab" or "aba"
print(longest_palindrome_substring("cbbd"))     # "bb"
print(longest_palindrome_substring("racecar"))  # "racecar"
```

---

## 20. 行列累乗による DP の高速化

線形漸化式を持つ DP は、行列累乗で O(k^3 log n) に高速化できる。

```python
import numpy as np

def matrix_power(M, n, mod=None):
    """行列の n 乗を繰り返し二乗法で計算 - O(k³ log n)"""
    result = [[0] * len(M) for _ in range(len(M))]
    for i in range(len(M)):
        result[i][i] = 1  # 単位行列

    base = [row[:] for row in M]

    while n > 0:
        if n % 2 == 1:
            result = matrix_multiply(result, base, mod)
        base = matrix_multiply(base, base, mod)
        n //= 2

    return result

def matrix_multiply(A, B, mod=None):
    """行列の積"""
    n = len(A)
    m = len(B[0])
    k = len(B)
    C = [[0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            for l in range(k):
                C[i][j] += A[i][l] * B[l][j]
                if mod:
                    C[i][j] %= mod
    return C

def fib_matrix(n: int, mod: int = 10**9 + 7) -> int:
    """フィボナッチ数を行列累乗で計算 - O(log n)
    [F(n+1)]   [1, 1]^n   [1]
    [F(n)  ] = [1, 0]   * [0]
    """
    if n <= 1:
        return n
    M = [[1, 1], [1, 0]]
    result = matrix_power(M, n, mod)
    return result[0][1] % mod

print(fib_matrix(10))     # 55
print(fib_matrix(100))    # 782204094 (mod 10^9+7)
print(fib_matrix(10**18)) # O(log n) で計算可能
```

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
5. Knuth, D. E. (1997). *The Art of Computer Programming, Volume 1*. Addison-Wesley.
