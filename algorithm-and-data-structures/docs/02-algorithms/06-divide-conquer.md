# 分割統治法（Divide and Conquer）

> 問題を小さな部分問題に分割し、再帰的に解いて統合する設計手法を、マージソート・大数乗算・最近接点対を通じて理解する

## この章で学ぶこと

1. **分割統治法の3ステップ**（分割・統治・統合）を理解し、再帰的に設計できる
2. **マスター定理**で分割統治アルゴリズムの計算量を解析できる
3. **Karatsuba乗算・最近接点対**など実用的な分割統治アルゴリズムを実装できる

---

## 1. 分割統治法の原理

```
┌─────────────────────────────────────────────────┐
│           分割統治法の3ステップ                    │
├─────────────────────────────────────────────────┤
│                                                  │
│  1. 分割 (Divide)                                │
│     → 問題を同じ種類の小さな部分問題に分割        │
│                                                  │
│  2. 統治 (Conquer)                               │
│     → 部分問題を再帰的に解く                      │
│     → 十分小さければ直接解く（基底条件）           │
│                                                  │
│  3. 統合 (Combine)                               │
│     → 部分問題の解を組み合わせて元の問題の解を得る │
│                                                  │
├─────────────────────────────────────────────────┤
│  DP との違い:                                     │
│  DP     → 部分問題が重複する → キャッシュして再利用 │
│  分割統治 → 部分問題が独立 → そのまま再帰         │
└─────────────────────────────────────────────────┘
```

```
分割統治の再帰構造:

        問題 (サイズ n)
       /              \
  部分問題            部分問題
  (サイズ n/2)       (サイズ n/2)
   /      \           /      \
 n/4     n/4        n/4     n/4
  ...    ...        ...     ...
  |       |          |       |
 基底   基底        基底    基底

  ↑ 統治（再帰）    ↓ 統合（マージ）
```

---

## 2. マージソート（分割統治の典型例）

```python
def merge_sort(arr: list) -> list:
    """マージソート - O(n log n)
    分割: 配列を半分に分割
    統治: 各半分を再帰的にソート
    統合: 2つのソート済み配列をマージ
    """
    # 基底条件
    if len(arr) <= 1:
        return arr

    # 分割
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])     # 統治（左半分）
    right = merge_sort(arr[mid:])    # 統治（右半分）

    # 統合
    return merge(left, right)

def merge(left: list, right: list) -> list:
    """2つのソート済み配列をマージ - O(n)"""
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

data = [38, 27, 43, 3, 9, 82, 10]
print(merge_sort(data))  # [3, 9, 10, 27, 38, 43, 82]
```

---

## 3. マスター定理

分割統治アルゴリズムの漸化式 T(n) = aT(n/b) + O(n^d) の解を求める。

```
T(n) = a * T(n/b) + O(n^d)

  a: 部分問題の数
  b: 分割比率（各部分問題のサイズ = n/b）
  d: 統合コストの指数

┌─────────────────────────────────────────────┐
│  ケース1: d < log_b(a)  →  T(n) = O(n^log_b(a))  │
│  ケース2: d = log_b(a)  →  T(n) = O(n^d log n)    │
│  ケース3: d > log_b(a)  →  T(n) = O(n^d)          │
└─────────────────────────────────────────────┘

例:
  マージソート: T(n) = 2T(n/2) + O(n)
    a=2, b=2, d=1, log_2(2)=1 → d=log_b(a) → O(n log n)

  二分探索: T(n) = T(n/2) + O(1)
    a=1, b=2, d=0, log_2(1)=0 → d=log_b(a) → O(log n)

  Karatsuba: T(n) = 3T(n/2) + O(n)
    a=3, b=2, d=1, log_2(3)≈1.585 → d<log_b(a) → O(n^1.585)
```

---

## 4. Karatsuba 乗算（大数乗算の高速化）

通常の筆算は O(n^2) だが、分割統治で O(n^1.585) に削減できる。

```
通常の乗算 (x * y):
  x = a * 10^(n/2) + b    (上位桁 a, 下位桁 b)
  y = c * 10^(n/2) + d

  x * y = ac * 10^n + (ad + bc) * 10^(n/2) + bd
  → 4回の乗算が必要: ac, ad, bc, bd

Karatsuba のトリック:
  p1 = ac
  p2 = bd
  p3 = (a+b)(c+d) = ac + ad + bc + bd
  ad + bc = p3 - p1 - p2

  → 3回の乗算で済む! (p1, p2, p3)
```

```python
def karatsuba(x: int, y: int) -> int:
    """Karatsuba 乗算 - O(n^1.585)"""
    # 基底条件
    if x < 10 or y < 10:
        return x * y

    # 桁数
    n = max(len(str(abs(x))), len(str(abs(y))))
    half = n // 2

    # 分割
    power = 10 ** half
    a, b = divmod(x, power)  # x = a * 10^half + b
    c, d = divmod(y, power)  # y = c * 10^half + d

    # 3回の再帰的乗算
    p1 = karatsuba(a, c)           # ac
    p2 = karatsuba(b, d)           # bd
    p3 = karatsuba(a + b, c + d)   # (a+b)(c+d)

    # 統合
    return p1 * (10 ** (2 * half)) + (p3 - p1 - p2) * (10 ** half) + p2

# 検証
x, y = 1234, 5678
print(karatsuba(x, y))  # 7006652
print(x * y)            # 7006652 (一致)

# 大きな数
big_x = 3141592653589793238462643383279
big_y = 2718281828459045235360287471352
print(karatsuba(big_x, big_y) == big_x * big_y)  # True
```

---

## 5. 最近接点対問題

平面上の n 個の点から、最も近い2点を見つける。ナイーブは O(n^2) だが、分割統治で O(n log n)。

```
点の集合:
  *     *
     *        *
  *      *
       *    *
  *       *

1. x座標でソートして左右に分割
2. 左半分・右半分それぞれで最近接点対を求める
3. 境界をまたぐペアをチェック（ストリップ内の点のみ）

  左半分  |  右半分
  *     * | *
     *    |    *
  *      *|
       *  | *
  *       |*

  δ = min(左の最近接, 右の最近接)
  中央線から δ 以内の「ストリップ」内で
  より近いペアがないか確認
```

```python
import math

def closest_pair(points: list) -> tuple:
    """最近接点対 - O(n log n)
    points: [(x, y), ...]
    返り値: (最小距離, 点1, 点2)
    """
    def dist(p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def brute_force(pts):
        min_d = float('inf')
        pair = (None, None)
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                d = dist(pts[i], pts[j])
                if d < min_d:
                    min_d = d
                    pair = (pts[i], pts[j])
        return min_d, pair

    def closest_strip(strip, d):
        """ストリップ内の最近接点対"""
        min_d = d
        pair = (None, None)
        strip.sort(key=lambda p: p[1])  # y座標でソート

        for i in range(len(strip)):
            j = i + 1
            while j < len(strip) and strip[j][1] - strip[i][1] < min_d:
                dd = dist(strip[i], strip[j])
                if dd < min_d:
                    min_d = dd
                    pair = (strip[i], strip[j])
                j += 1
        return min_d, pair

    def solve(pts_x):
        n = len(pts_x)
        if n <= 3:
            return brute_force(pts_x)

        mid = n // 2
        mid_x = pts_x[mid][0]

        # 分割
        left = pts_x[:mid]
        right = pts_x[mid:]

        # 統治
        dl, pl = solve(left)
        dr, pr = solve(right)

        d = min(dl, dr)
        pair = pl if dl <= dr else pr

        # 統合: ストリップ内のチェック
        strip = [p for p in pts_x if abs(p[0] - mid_x) < d]
        ds, ps = closest_strip(strip, d)

        if ds < d:
            return ds, ps
        return d, pair

    sorted_x = sorted(points, key=lambda p: p[0])
    return solve(sorted_x)

# 使用例
points = [(2,3), (12,30), (40,50), (5,1), (12,10), (3,4)]
d, (p1, p2) = closest_pair(points)
print(f"最近接点対: {p1}, {p2}")
print(f"距離: {d:.4f}")
# 最近接点対: (2, 3), (3, 4)
# 距離: 1.4142
```

---

## 6. その他の分割統治アルゴリズム

### べき乗の高速化

```python
def fast_power(base: int, exp: int, mod: int = None) -> int:
    """繰り返し二乗法 - O(log n)"""
    if exp == 0:
        return 1
    if exp % 2 == 0:
        half = fast_power(base, exp // 2, mod)
        result = half * half
    else:
        result = base * fast_power(base, exp - 1, mod)

    return result % mod if mod else result

# 反復版（実用的）
def fast_power_iterative(base: int, exp: int, mod: int = None) -> int:
    result = 1
    base = base % mod if mod else base
    while exp > 0:
        if exp % 2 == 1:
            result = result * base
            if mod:
                result %= mod
        exp >>= 1
        base = base * base
        if mod:
            base %= mod
    return result

print(fast_power(2, 30))           # 1073741824
print(fast_power(2, 30, 10**9+7))  # 1073741824
```

### 最大部分配列和（Kadane は O(n) だが分割統治版）

```python
def max_subarray_dc(arr: list, low: int = 0, high: int = None) -> int:
    """最大部分配列和 - 分割統治版 O(n log n)"""
    if high is None:
        high = len(arr) - 1

    if low == high:
        return arr[low]

    mid = (low + high) // 2

    # 左半分の最大、右半分の最大
    left_max = max_subarray_dc(arr, low, mid)
    right_max = max_subarray_dc(arr, mid + 1, high)

    # 中央をまたぐ最大
    left_sum = float('-inf')
    total = 0
    for i in range(mid, low - 1, -1):
        total += arr[i]
        left_sum = max(left_sum, total)

    right_sum = float('-inf')
    total = 0
    for i in range(mid + 1, high + 1):
        total += arr[i]
        right_sum = max(right_sum, total)

    cross_max = left_sum + right_sum

    return max(left_max, right_max, cross_max)

data = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
print(max_subarray_dc(data))  # 6 (部分配列 [4, -1, 2, 1])
```

---

## 7. 分割統治の適用判断

```
分割統治が有効な問題の特徴:

  [YES] 問題が同種の小問題に分割可能
  [YES] 部分問題が独立（重複しない）
  [YES] 統合ステップが効率的（O(n) 以下）
  [YES] 分割により計算量が減少する

  [NO]  部分問題が重複する → DPを使う
  [NO]  逐次的な処理が必要 → 貪欲法を検討
  [NO]  統合コストが O(n²) → 分割の意味がない
```

## 計算量比較表

| アルゴリズム | 漸化式 | 計算量 | マスター定理 |
|:---|:---|:---|:---|
| 二分探索 | T(n) = T(n/2) + O(1) | O(log n) | a=1,b=2,d=0 ケース2 |
| マージソート | T(n) = 2T(n/2) + O(n) | O(n log n) | a=2,b=2,d=1 ケース2 |
| Karatsuba | T(n) = 3T(n/2) + O(n) | O(n^1.585) | a=3,b=2,d=1 ケース1 |
| Strassen | T(n) = 7T(n/2) + O(n²) | O(n^2.807) | a=7,b=2,d=2 ケース1 |
| 最近接点対 | T(n) = 2T(n/2) + O(n) | O(n log n) | a=2,b=2,d=1 ケース2 |

## 設計パラダイムの使い分け

| 特徴 | 分割統治 | DP | 貪欲法 |
|:---|:---|:---|:---|
| 部分問題の関係 | 独立 | 重複 | 独立 |
| 解の構築方法 | 再帰+統合 | テーブル埋め | 逐次決定 |
| 後戻り | なし | なし | なし |
| 典型計算量 | O(n log n) | O(n²)〜O(nW) | O(n log n) |
| 代表例 | マージソート | LCS | 活動選択 |

---

## 8. アンチパターン

### アンチパターン1: 不均等な分割

```python
# BAD: 分割が極端に不均等 → O(n²)に退化
def bad_sort(arr):
    if len(arr) <= 1:
        return arr
    # ピボットが最小値 → 1個 vs n-1個の分割
    pivot = min(arr)  # 最悪の分割
    left = [x for x in arr if x < pivot]
    right = [x for x in arr if x > pivot]
    return bad_sort(left) + [pivot] + bad_sort(right)

# GOOD: 均等分割を保証
def good_merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2  # 常に半分に分割
    left = good_merge_sort(arr[:mid])
    right = good_merge_sort(arr[mid:])
    return merge(left, right)
```

### アンチパターン2: 重複部分問題に分割統治を適用

```python
# BAD: フィボナッチに分割統治（部分問題が重複するためDP向き）
def fib_dc(n):
    if n <= 1:
        return n
    return fib_dc(n-1) + fib_dc(n-2)  # O(2^n) — 重複だらけ

# GOOD: DPで解く（メモ化 or ボトムアップ）
from functools import lru_cache

@lru_cache(maxsize=None)
def fib_dp(n):
    if n <= 1:
        return n
    return fib_dp(n-1) + fib_dp(n-2)  # O(n)
```

---

## 9. FAQ

### Q1: 分割統治と再帰の違いは？

**A:** 再帰は実装技法（関数が自身を呼び出す）で、分割統治は設計パラダイム（問題を分割→統治→統合）。分割統治は再帰で実装されることが多いが、再帰が全て分割統治ではない。例えばDFSは再帰だが分割統治ではない。

### Q2: マスター定理で解けない漸化式は？

**A:** T(n) = aT(n/b) + f(n) の形でない場合（例: T(n) = T(n-1) + T(n-2)）や、部分問題サイズが不均等な場合はマスター定理が適用できない。その場合は再帰木法や置換法を使う。

### Q3: Strassen 行列乗算は実用的か？

**A:** Strassen法は理論的に O(n^2.807) で通常の O(n^3) より速いが、定数係数が大きく、数値の安定性も劣る。実用的には n > 数百程度の行列で初めて有利になる。現代のライブラリ（BLAS等）はハードウェア最適化された通常の乗算を使うことが多い。

---

## 10. まとめ

| 項目 | 要点 |
|:---|:---|
| 3ステップ | 分割→統治→統合の再帰的設計 |
| マスター定理 | T(n)=aT(n/b)+O(n^d) の計算量を解析 |
| マージソート | 分割統治の最も基本的な例。O(n log n) |
| Karatsuba | 4回→3回の乗算で O(n^1.585) |
| 最近接点対 | ストリップ内の限定的チェックで O(n log n) |
| 適用条件 | 独立な部分問題 + 効率的な統合 |

---

## 次に読むべきガイド

- [ソートアルゴリズム](./00-sorting.md) -- マージソート・クイックソートの詳細
- [動的計画法](./04-dynamic-programming.md) -- 部分問題が重複する場合の手法
- [バックトラッキング](./07-backtracking.md) -- 別の再帰的問題解決パラダイム

---

## 参考文献

1. Cormen, T. H. et al. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. -- 第4章
2. Karatsuba, A. & Ofman, Y. (1963). "Multiplication of multidigit numbers on automata." *Soviet Physics Doklady*.
3. Shamos, M. I. & Hoey, D. (1975). "Closest-point problems." *16th Annual Symposium on Foundations of Computer Science*.
4. Kleinberg, J. & Tardos, E. (2005). *Algorithm Design*. Pearson. -- Chapter 5: Divide and Conquer
