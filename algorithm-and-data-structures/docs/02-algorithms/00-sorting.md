# ソートアルゴリズム

> データを特定の順序に並べ替える基本アルゴリズム群を、計算量・安定性・実装の観点から体系的に理解する

## この章で学ぶこと

1. **7種のソートアルゴリズム**の原理・実装・計算量を比較できるようになる
2. **安定性・in-place性・適応性**の違いから、場面に応じた最適なソートを選択できる
3. **分割統治・ヒープ・計数**といった異なるパラダイムがソートにどう適用されるかを理解する

---

## 1. ソートの全体像

```
┌─────────────────────────────────────────────────────────┐
│                   ソートアルゴリズム                      │
├─────────────────┬───────────────────┬───────────────────┤
│  比較ベース      │  比較ベース        │  非比較ベース      │
│  O(n²)          │  O(n log n)       │  O(n + k)         │
├─────────────────┼───────────────────┼───────────────────┤
│ バブルソート     │ マージソート       │ 計数ソート         │
│ 選択ソート       │ クイックソート     │ 基数ソート         │
│ 挿入ソート       │ ヒープソート       │ バケットソート      │
└─────────────────┴───────────────────┴───────────────────┘
```

---

## 2. バブルソート（Bubble Sort）

隣接する要素を比較・交換し、最大値を末尾に「泡」のように浮かせる。

```
パス1: [5, 3, 8, 1, 2]
        ↕↕
       [3, 5, 8, 1, 2]  → 5>3 なので交換
       [3, 5, 8, 1, 2]  → 5<8 そのまま
       [3, 5, 1, 8, 2]  → 8>1 なので交換
       [3, 5, 1, 2, 8]  → 8>2 なので交換  ← 8が確定

パス2: [3, 5, 1, 2 | 8]
       [3, 1, 2, 5 | 8]  ← 5が確定

パス3: [1, 2, 3 | 5, 8]  ← 3が確定
       完了!
```

### 実装（Python）

```python
def bubble_sort(arr: list) -> list:
    """バブルソート - 安定・in-place"""
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(n - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:  # 最適化: 交換がなければ終了
            break
    return arr

# 使用例
data = [64, 34, 25, 12, 22, 11, 90]
print(bubble_sort(data))  # [11, 12, 22, 25, 34, 64, 90]
```

---

## 3. 選択ソート（Selection Sort）

未ソート部分から最小値を見つけ、先頭に配置する。

```python
def selection_sort(arr: list) -> list:
    """選択ソート - 不安定・in-place"""
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

# 交換回数は常に O(n) — 書き込みコストが高い場合に有利
data = [29, 10, 14, 37, 13]
print(selection_sort(data))  # [10, 13, 14, 29, 37]
```

---

## 4. 挿入ソート（Insertion Sort）

トランプの手札を整列するように、一枚ずつ適切な位置に挿入する。

```
初期: [5, 2, 4, 6, 1, 3]

i=1: key=2  →  [2, 5, 4, 6, 1, 3]
i=2: key=4  →  [2, 4, 5, 6, 1, 3]
i=3: key=6  →  [2, 4, 5, 6, 1, 3]  (移動なし)
i=4: key=1  →  [1, 2, 4, 5, 6, 3]
i=5: key=3  →  [1, 2, 3, 4, 5, 6]
```

```python
def insertion_sort(arr: list) -> list:
    """挿入ソート - 安定・in-place・適応的"""
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

# ほぼ整列済みデータなら O(n) — TimSort の内部で使用
data = [1, 3, 2, 4, 6, 5]
print(insertion_sort(data))  # [1, 2, 3, 4, 5, 6]
```

---

## 5. マージソート（Merge Sort）

配列を再帰的に半分に分割し、ソート済みの部分配列をマージする。

```
分割フェーズ:
[38, 27, 43, 3, 9, 82, 10]
       /                \
[38, 27, 43]      [3, 9, 82, 10]
  /      \          /         \
[38]  [27, 43]   [3, 9]    [82, 10]
       / \        / \        /   \
     [27] [43]  [3] [9]   [82]  [10]

マージフェーズ:
     [27] [43]  [3] [9]   [82]  [10]
       \ /        \ /        \   /
     [27, 43]   [3, 9]    [10, 82]
  \      /          \         /
[27, 38, 43]    [3, 9, 10, 82]
       \                /
[3, 9, 10, 27, 38, 43, 82]
```

```python
def merge_sort(arr: list) -> list:
    """マージソート - 安定・O(n)追加メモリ"""
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left: list, right: list) -> list:
    """2つのソート済み配列をマージ"""
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:  # <= で安定性を保証
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

## 6. クイックソート（Quick Sort）

ピボットを選び、配列を「ピボット以下」「ピボット以上」に分割して再帰的にソートする。

```python
def quick_sort(arr: list, low: int = 0, high: int = None) -> list:
    """クイックソート - 不安定・in-place"""
    if high is None:
        high = len(arr) - 1
    if low < high:
        pivot_idx = partition(arr, low, high)
        quick_sort(arr, low, pivot_idx - 1)
        quick_sort(arr, pivot_idx + 1, high)
    return arr

def partition(arr: list, low: int, high: int) -> int:
    """Lomutoのパーティション"""
    pivot = arr[high]  # 末尾をピボットに
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

import random

def quick_sort_randomized(arr: list, low: int = 0, high: int = None) -> list:
    """ランダム化クイックソート - 最悪ケースを回避"""
    if high is None:
        high = len(arr) - 1
    if low < high:
        rand_idx = random.randint(low, high)
        arr[rand_idx], arr[high] = arr[high], arr[rand_idx]
        pivot_idx = partition(arr, low, high)
        quick_sort_randomized(arr, low, pivot_idx - 1)
        quick_sort_randomized(arr, pivot_idx + 1, high)
    return arr

data = [10, 7, 8, 9, 1, 5]
print(quick_sort(data[:]))  # [1, 5, 7, 8, 9, 10]
```

---

## 7. ヒープソート（Heap Sort）

最大ヒープを構築し、ルート（最大値）を末尾に移動して繰り返す。

```
最大ヒープ構築:          ヒープから取り出し:
       9                    取り出し順: 9, 8, 7, 6, 5
      / \                   → ソート結果: [5, 6, 7, 8, 9]
     8   7
    / \
   5   6
```

```python
def heap_sort(arr: list) -> list:
    """ヒープソート - 不安定・in-place・O(n log n)保証"""
    n = len(arr)

    # 最大ヒープ構築 (ボトムアップ)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # ルートを末尾に移動してヒープサイズを縮小
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 0)

    return arr

def heapify(arr: list, n: int, i: int) -> None:
    """ノードiを根とする部分木をヒープ化"""
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[left] > arr[largest]:
        largest = left
    if right < n and arr[right] > arr[largest]:
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

data = [12, 11, 13, 5, 6, 7]
print(heap_sort(data))  # [5, 6, 7, 11, 12, 13]
```

---

## 8. 計数ソート（Counting Sort）

要素の出現回数をカウントし、累積和で位置を決定する。

```
入力:  [4, 2, 2, 8, 3, 3, 1]
範囲:  0..8

カウント: [0, 1, 2, 2, 1, 0, 0, 0, 1]
          idx: 0  1  2  3  4  5  6  7  8

累積和:   [0, 1, 3, 5, 6, 6, 6, 6, 7]

出力:  [1, 2, 2, 3, 3, 4, 8]
```

```python
def counting_sort(arr: list) -> list:
    """計数ソート - 安定・O(n+k)"""
    if not arr:
        return arr

    max_val = max(arr)
    min_val = min(arr)
    range_val = max_val - min_val + 1

    count = [0] * range_val
    output = [0] * len(arr)

    # カウント
    for num in arr:
        count[num - min_val] += 1

    # 累積和
    for i in range(1, range_val):
        count[i] += count[i - 1]

    # 出力配列構築（安定性のため末尾から）
    for i in range(len(arr) - 1, -1, -1):
        output[count[arr[i] - min_val] - 1] = arr[i]
        count[arr[i] - min_val] -= 1

    return output

data = [4, 2, 2, 8, 3, 3, 1]
print(counting_sort(data))  # [1, 2, 2, 3, 3, 4, 8]
```

---

## 9. 計算量比較表

| アルゴリズム | 最良 | 平均 | 最悪 | 空間 | 安定 | in-place |
|:---|:---|:---|:---|:---|:---|:---|
| バブルソート | O(n) | O(n²) | O(n²) | O(1) | Yes | Yes |
| 選択ソート | O(n²) | O(n²) | O(n²) | O(1) | No | Yes |
| 挿入ソート | O(n) | O(n²) | O(n²) | O(1) | Yes | Yes |
| マージソート | O(n log n) | O(n log n) | O(n log n) | O(n) | Yes | No |
| クイックソート | O(n log n) | O(n log n) | O(n²) | O(log n) | No | Yes |
| ヒープソート | O(n log n) | O(n log n) | O(n log n) | O(1) | No | Yes |
| 計数ソート | O(n+k) | O(n+k) | O(n+k) | O(n+k) | Yes | No |

## 10. 用途別選択ガイド

| 状況 | 推奨アルゴリズム | 理由 |
|:---|:---|:---|
| 小規模データ（n < 50） | 挿入ソート | オーバーヘッドが小さい |
| ほぼ整列済み | 挿入ソート | 適応的で O(n) に近い |
| 汎用（ライブラリ） | TimSort（マージ+挿入） | Python/Java標準 |
| メモリ制約あり | ヒープソート | O(1) 追加メモリ |
| 整数・範囲が小さい | 計数ソート | O(n+k) で高速 |
| 平均性能重視 | クイックソート | 定数係数が小さい |
| 安定性必須 | マージソート | O(n log n) かつ安定 |
| 外部ソート（大容量） | マージソート | 逐次アクセスに強い |

---

## 11. アンチパターン

### アンチパターン1: ピボット選択の固定化

```python
# BAD: 常に末尾をピボットにする
# → ソート済み配列で O(n²) に退化
def bad_quicksort(arr, low, high):
    pivot = arr[high]  # ソート済みで最悪ケース
    ...

# GOOD: 三値の中央値 or ランダム選択
def good_partition(arr, low, high):
    mid = (low + high) // 2
    # 三値の中央値
    if arr[low] > arr[mid]:
        arr[low], arr[mid] = arr[mid], arr[low]
    if arr[low] > arr[high]:
        arr[low], arr[high] = arr[high], arr[low]
    if arr[mid] > arr[high]:
        arr[mid], arr[high] = arr[high], arr[mid]
    pivot = arr[mid]
    ...
```

### アンチパターン2: 計数ソートの値域無視

```python
# BAD: 値域が巨大な場合に計数ソートを使う
data = [1, 1000000000, 2]  # → count配列が 10^9 サイズ！

# GOOD: 値域が小さい場合のみ使用
if max(data) - min(data) < len(data) * 10:
    counting_sort(data)
else:
    data.sort()  # 比較ベースソートにフォールバック
```

### アンチパターン3: 安定性を無視した選択

```python
# BAD: レコードのソートに不安定ソートを使い順序が崩れる
students = [(3, "Alice"), (1, "Bob"), (3, "Charlie")]
# ヒープソートだと同一キーの Alice, Charlie の順序が保証されない

# GOOD: 安定ソートを使う
students.sort(key=lambda x: x[0])  # Python の sort は安定
# [(1, "Bob"), (3, "Alice"), (3, "Charlie")]  ← 元の順序保持
```

---

## 12. FAQ

### Q1: Python の `sort()` と `sorted()` は何のアルゴリズム？

**A:** TimSort（マージソート + 挿入ソートのハイブリッド）。安定で、最良 O(n)、平均・最悪 O(n log n)。実データのパターン（run）を検出して効率化する。Java の `Arrays.sort()` もオブジェクト配列に TimSort を使用する。

### Q2: O(n log n) より速いソートは存在するか？

**A:** 比較ベースのソートでは O(n log n) が理論的下限（決定木の高さ）。ただし非比較ベース（計数・基数・バケット）はこの制約を受けず、条件次第で O(n) を達成できる。

### Q3: 実務で自前ソートを実装すべきか？

**A:** 通常はNO。言語標準のソート（TimSort等）は高度に最適化されており、自前実装より高速。自前が必要なのは、外部ソート、特殊なキー関数、教育目的など限定的な場面のみ。

### Q4: クイックソートとマージソートの使い分けは？

**A:** クイックソートは平均的に高速（キャッシュ効率が良い）だが最悪 O(n²)。マージソートは O(n log n) 保証だが O(n) 追加メモリ。メモリ潤沢なら安定性のあるマージソート、メモリ制約ありならクイックソートが定番。

---

## 13. まとめ

| 項目 | 要点 |
|:---|:---|
| O(n²) ソート | バブル・選択・挿入は教育的、小規模データや部分ソートに有用 |
| O(n log n) ソート | マージ・クイック・ヒープが実用の中心 |
| 非比較ソート | 計数・基数・バケットは条件付きで O(n) |
| 安定性 | レコードの相対順序を保持するかどうか。マージ・挿入・計数は安定 |
| 実務選択 | 言語標準ソート（TimSort）が第一選択、特殊要件時のみ自前実装 |
| ピボット戦略 | ランダム化 or 三値中央値で最悪ケースを回避 |

---

## 次に読むべきガイド

- [探索アルゴリズム](./01-searching.md) -- ソート済みデータに対する効率的な探索
- [分割統治法](./06-divide-conquer.md) -- マージソート・クイックソートの設計パラダイム
- [動的計画法](./04-dynamic-programming.md) -- 最適部分構造を活用する別のパラダイム

---

## 参考文献

1. Cormen, T. H. et al. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. -- 第2章・第6-8章
2. Sedgewick, R. & Wayne, K. (2011). *Algorithms* (4th ed.). Addison-Wesley. -- Part 2: Sorting
3. Python Documentation. "Sorting HOW TO." https://docs.python.org/3/howto/sorting.html
4. McIlroy, P. (1993). "Optimistic Sorting and Information Theoretic Complexity." *SODA*. -- TimSort の理論的基盤
