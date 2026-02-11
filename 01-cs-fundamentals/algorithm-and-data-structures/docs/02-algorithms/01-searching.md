# 探索アルゴリズム

> データ集合から目的の要素を効率的に見つけ出す手法を、線形探索・二分探索・補間探索の3段階で理解する

## この章で学ぶこと

1. **線形探索・二分探索・補間探索**の原理と計算量を比較し、適切な場面で使い分けられる
2. **二分探索の変形パターン**（lower_bound, upper_bound, 条件探索）を正確に実装できる
3. **探索とデータ構造の関係**を理解し、前処理としてのソートやインデックスの意義を把握する

---

## 1. 探索アルゴリズムの全体像

```
┌──────────────────────────────────────────────────┐
│              探索アルゴリズム                      │
├────────────────┬─────────────────┬───────────────┤
│  線形探索       │  二分探索        │  補間探索      │
│  O(n)          │  O(log n)       │  O(log log n)  │
│  前処理不要     │  ソート済み必須   │  均一分布必須   │
│  任意データ     │  ランダムアクセス  │  数値データ    │
└────────────────┴─────────────────┴───────────────┘
```

---

## 2. 線形探索（Linear Search）

先頭から順番に1つずつ要素を比較する最も単純な探索。

```
探索: key = 7
配列: [3, 8, 1, 7, 5, 2]
       ↑        見つからない (3≠7)
          ↑     見つからない (8≠7)
             ↑  見つからない (1≠7)
                ↑ 発見! (7=7) → インデックス 3
```

### 実装

```python
def linear_search(arr: list, target) -> int:
    """線形探索 - O(n)"""
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1

# 番兵付き線形探索（比較回数を半減）
def sentinel_search(arr: list, target) -> int:
    """番兵法 - ループ内の境界チェックを省略"""
    n = len(arr)
    last = arr[n - 1]
    arr[n - 1] = target  # 番兵を設置

    i = 0
    while arr[i] != target:
        i += 1

    arr[n - 1] = last  # 元に戻す
    if i < n - 1 or arr[n - 1] == target:
        return i
    return -1

data = [15, 23, 8, 42, 16, 4]
print(linear_search(data, 42))  # 3
print(linear_search(data, 99))  # -1
```

---

## 3. 二分探索（Binary Search）

ソート済み配列を半分に分割しながら探索する。毎回探索範囲が半減するため O(log n)。

```
探索: key = 23
配列: [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]

ステップ1: low=0, high=9, mid=4
  [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]
                   ↑mid
  16 < 23 → 右半分へ (low=5)

ステップ2: low=5, high=9, mid=7
  [_, _, _, _, _, 23, 38, 56, 72, 91]
                          ↑mid
  56 > 23 → 左半分へ (high=6)

ステップ3: low=5, high=6, mid=5
  [_, _, _, _, _, 23, 38, _, _, _]
                  ↑mid
  23 == 23 → 発見! インデックス 5
```

### 基本実装

```python
def binary_search(arr: list, target) -> int:
    """二分探索 - O(log n), ソート済み配列が前提"""
    low, high = 0, len(arr) - 1

    while low <= high:
        mid = low + (high - low) // 2  # オーバーフロー防止
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1

    return -1

# 再帰版
def binary_search_recursive(arr: list, target, low: int = 0, high: int = None) -> int:
    if high is None:
        high = len(arr) - 1
    if low > high:
        return -1

    mid = low + (high - low) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, high)
    else:
        return binary_search_recursive(arr, target, low, mid - 1)

data = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]
print(binary_search(data, 23))  # 5
print(binary_search(data, 99))  # -1
```

### 二分探索の変形: lower_bound / upper_bound

```python
def lower_bound(arr: list, target) -> int:
    """target 以上の最小インデックスを返す（C++ の lower_bound 相当）"""
    low, high = 0, len(arr)
    while low < high:
        mid = low + (high - low) // 2
        if arr[mid] < target:
            low = mid + 1
        else:
            high = mid
    return low

def upper_bound(arr: list, target) -> int:
    """target より大きい最小インデックスを返す"""
    low, high = 0, len(arr)
    while low < high:
        mid = low + (high - low) // 2
        if arr[mid] <= target:
            low = mid + 1
        else:
            high = mid
    return low

# 要素の出現回数
def count_occurrences(arr: list, target) -> int:
    return upper_bound(arr, target) - lower_bound(arr, target)

data = [1, 2, 2, 2, 3, 4, 5]
print(lower_bound(data, 2))      # 1 (最初の2のインデックス)
print(upper_bound(data, 2))      # 4 (最後の2の次のインデックス)
print(count_occurrences(data, 2)) # 3
```

### 二分探索の応用: 条件で探索

```python
def binary_search_condition(low: int, high: int, condition) -> int:
    """条件を満たす最小値を二分探索で求める
    condition(x) が False, False, ..., True, True, ... となる境界を探す
    """
    while low < high:
        mid = low + (high - low) // 2
        if condition(mid):
            high = mid
        else:
            low = mid + 1
    return low

# 例: x^2 >= 100 となる最小の正整数
result = binary_search_condition(1, 100, lambda x: x * x >= 100)
print(result)  # 10
```

---

## 4. 補間探索（Interpolation Search）

データが均一分布している場合、値の位置を「補間」で推定する。電話帳で "T" を探すとき、後半を開くのと同じ発想。

```
配列: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
target = 70

通常の二分探索: mid = (0+9)/2 = 4 → arr[4]=50

補間探索の推定:
  pos = low + (target - arr[low]) * (high - low) / (arr[high] - arr[low])
      = 0 + (70 - 10) * (9 - 0) / (100 - 10)
      = 0 + 60 * 9 / 90
      = 6 → arr[6] = 70  ← 一発で発見!
```

```python
def interpolation_search(arr: list, target) -> int:
    """補間探索 - 均一分布で O(log log n)"""
    low, high = 0, len(arr) - 1

    while (low <= high and
           arr[low] <= target <= arr[high]):

        if low == high:
            return low if arr[low] == target else -1

        # 位置を補間で推定
        pos = low + int(
            (target - arr[low]) * (high - low) /
            (arr[high] - arr[low])
        )

        if arr[pos] == target:
            return pos
        elif arr[pos] < target:
            low = pos + 1
        else:
            high = pos - 1

    return -1

# 均一分布データ
data = list(range(10, 1001, 10))  # [10, 20, 30, ..., 1000]
print(interpolation_search(data, 700))  # 69
```

---

## 5. 探索アルゴリズム比較表

| アルゴリズム | 最良 | 平均 | 最悪 | 前提条件 | 空間計算量 |
|:---|:---|:---|:---|:---|:---|
| 線形探索 | O(1) | O(n) | O(n) | なし | O(1) |
| 二分探索 | O(1) | O(log n) | O(log n) | ソート済み | O(1) |
| 補間探索 | O(1) | O(log log n) | O(n) | ソート済み+均一分布 | O(1) |

## 実装言語の標準ライブラリ

| 言語 | 二分探索関数 | 備考 |
|:---|:---|:---|
| Python | `bisect.bisect_left()`, `bisect.bisect_right()` | lower_bound, upper_bound 相当 |
| C++ | `std::lower_bound()`, `std::upper_bound()` | イテレータベース |
| Java | `Arrays.binarySearch()`, `Collections.binarySearch()` | 見つからない場合 -(挿入位置)-1 |
| JavaScript | なし（自前実装） | `Array.prototype.findIndex()` は線形 |
| Go | `sort.Search()` | 条件関数ベース |

---

## 6. Python bisect モジュールの活用

```python
import bisect

data = [1, 3, 5, 7, 9, 11, 13]

# 挿入位置を探す（ソート順を維持）
print(bisect.bisect_left(data, 7))   # 3 (7の左側)
print(bisect.bisect_right(data, 7))  # 4 (7の右側)

# ソート順を維持しながら挿入
bisect.insort(data, 6)
print(data)  # [1, 3, 5, 6, 7, 9, 11, 13]

# 要素の存在確認
def binary_contains(arr, target):
    idx = bisect.bisect_left(arr, target)
    return idx < len(arr) and arr[idx] == target

print(binary_contains(data, 7))   # True
print(binary_contains(data, 8))   # False
```

---

## 7. アンチパターン

### アンチパターン1: 未ソートデータに二分探索

```python
# BAD: ソートされていないデータに二分探索
unsorted = [3, 1, 4, 1, 5, 9, 2, 6]
result = binary_search(unsorted, 5)  # 不正な結果!

# GOOD: 事前にソートするか、線形探索を使う
sorted_data = sorted(unsorted)
result = binary_search(sorted_data, 5)  # 正しい結果
# あるいは探索頻度が低いなら
result = linear_search(unsorted, 5)
```

### アンチパターン2: 二分探索の off-by-one エラー

```python
# BAD: low <= high の条件と mid ± 1 の整合性ミス
def bad_binary_search(arr, target):
    low, high = 0, len(arr) - 1
    while low < high:  # <= ではなく < にしてしまう
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1
# → low == high のケースを見逃す（要素が1つの場合など）

# GOOD: low <= high を使い、mid ± 1 で確実に範囲を縮小
def good_binary_search(arr, target):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = low + (high - low) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1
```

### アンチパターン3: 整数オーバーフロー

```python
# BAD: (low + high) がオーバーフローする可能性（C/C++/Java）
mid = (low + high) // 2

# GOOD: オーバーフロー安全な計算
mid = low + (high - low) // 2
# Python は任意精度整数なので問題ないが、他言語では必須
```

---

## 8. FAQ

### Q1: 二分探索はリンクリストに使えるか？

**A:** 理論的には可能だが実用的でない。リンクリストはランダムアクセスが O(n) のため、mid への到達に O(n) かかり、全体で O(n log n) となって線形探索 O(n) より遅くなる。二分探索にはランダムアクセス可能なデータ構造（配列）が必要。

### Q2: 探索を O(1) にする方法は？

**A:** ハッシュテーブル（辞書）を使えば平均 O(1) で探索できる。ただし最悪 O(n)、順序を保持しない、追加メモリが必要といったトレードオフがある。順序付き探索が必要なら二分探索木（O(log n)）を使う。

### Q3: 浮動小数点数の二分探索はどうする？

**A:** `low <= high` の代わりに `high - low > epsilon` を使う。または反復回数を固定（例: 100回）する方法が安全。

```python
def binary_search_float(low: float, high: float, f, target: float, eps: float = 1e-9) -> float:
    for _ in range(100):  # 十分な精度を保証
        mid = (low + high) / 2
        if f(mid) < target:
            low = mid
        else:
            high = mid
    return (low + high) / 2
```

---

## 9. まとめ

| 項目 | 要点 |
|:---|:---|
| 線形探索 | 前処理不要で万能だが O(n)。小規模データや未ソートデータに |
| 二分探索 | ソート済みなら O(log n)。最も頻出する探索アルゴリズム |
| 補間探索 | 均一分布なら O(log log n) だが、偏りがあると O(n) に退化 |
| lower_bound/upper_bound | 二分探索の最重要変形。範囲探索・出現回数に不可欠 |
| 条件二分探索 | 単調性のある判定関数に適用可能。最適化問題に頻出 |
| 標準ライブラリ | Python の bisect、C++ の STL を活用すべき |

---

## 次に読むべきガイド

- [ソートアルゴリズム](./00-sorting.md) -- 二分探索の前提となるソートの理解
- [グラフ走査](./02-graph-traversal.md) -- グラフ上の探索（BFS/DFS）
- [動的計画法](./04-dynamic-programming.md) -- 二分探索 + DP の組み合わせ技法

---

## 参考文献

1. Cormen, T. H. et al. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. -- 第2章 二分探索
2. Knuth, D. E. (1998). *The Art of Computer Programming, Vol. 3: Sorting and Searching*. Addison-Wesley.
3. Python Documentation. "bisect --- Array bisection algorithm." https://docs.python.org/3/library/bisect.html
4. Perl, Y., Itai, A., & Avni, H. (1978). "Interpolation search -- a log log n search." *Communications of the ACM*.
