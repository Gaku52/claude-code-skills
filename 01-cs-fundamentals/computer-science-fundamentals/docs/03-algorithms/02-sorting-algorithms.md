# ソートアルゴリズム

> ソートはCS最古にして最重要の問題の一つ。比較ベースのソートには O(n log n) の理論的下限が存在する。

## この章で学ぶこと

- [ ] 主要なソートアルゴリズム6種類の仕組みと計算量を説明できる
- [ ] 各ソートの使い分けを理解する
- [ ] O(n log n) の下限の証明を理解する

## 前提知識

- 計算量解析 → 参照: [[01-complexity-analysis.md]]

---

## 1. O(n²) のソート

### 1.1 バブルソート

```python
def bubble_sort(arr):
    """隣接する要素を比較・交換して「泡」のように浮かべる"""
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:  # 交換がなければソート済み
            break
    return arr

# 計算量:
# 最悪: O(n²) — 逆順
# 平均: O(n²)
# 最良: O(n) — ソート済み（early exit あり）
# 空間: O(1)
# 安定: Yes

# 動作例: [5, 3, 1, 4, 2]
# Pass 1: [3,1,4,2,5] — 5が最後に浮かぶ
# Pass 2: [1,3,2,4,5] — 4が所定位置に
# Pass 3: [1,2,3,4,5] — ソート完了
# Pass 4: 交換なし → 終了
```

### 1.2 選択ソート

```python
def selection_sort(arr):
    """未ソート部分から最小値を選んで先頭に配置"""
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

# 計算量:
# 最悪/平均/最良: 全て O(n²) — 常に全探索
# 空間: O(1)
# 安定: No（交換で順序が崩れる）
# 交換回数: O(n) — バブルソートより交換が少ない
```

### 1.3 挿入ソート

```python
def insertion_sort(arr):
    """カードを手札に挿入するように、正しい位置に挿入"""
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

# 計算量:
# 最悪: O(n²) — 逆順
# 平均: O(n²)
# 最良: O(n) — ソート済み
# 空間: O(1)
# 安定: Yes

# 利点:
# - ほぼソート済みのデータに非常に速い
# - 小さいデータ(n < 50)ではオーバーヘッドが小さく最速
# - オンライン: データを受信しながらソート可能
# - Python の Timsort は小さい部分配列に挿入ソートを使用
```

---

## 2. O(n log n) のソート

### 2.1 マージソート

```python
def merge_sort(arr):
    """分割→ソート→結合（分割統治法）"""
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    """ソート済みの2つの配列を結合"""
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

# 計算量:
# 最悪/平均/最良: 全て O(n log n) — 入力に依存しない
# 空間: O(n) — マージ用の一時配列
# 安定: Yes

# 再帰の展開（n=8の場合）:
# [5,3,1,4,2,8,7,6]
#      /          \
# [5,3,1,4]    [2,8,7,6]
#   /    \       /    \
# [5,3] [1,4]  [2,8] [7,6]
# / \   / \    / \   / \
# 5  3  1  4  2  8  7  6   ← 分割 O(log n) 段
# \ /   \ /    \ /   \ /
# [3,5] [1,4]  [2,8] [6,7]  ← 各段で O(n) のマージ
#   \    /       \    /
# [1,3,4,5]    [2,6,7,8]
#      \          /
# [1,2,3,4,5,6,7,8]
```

### 2.2 クイックソート

```python
def quicksort(arr):
    """ピボットで分割→再帰（分割統治法）"""
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]  # ピボット選択
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

# インプレース版（省メモリ）:
def quicksort_inplace(arr, low, high):
    if low < high:
        pivot_idx = partition(arr, low, high)
        quicksort_inplace(arr, low, pivot_idx - 1)
        quicksort_inplace(arr, pivot_idx + 1, high)

def partition(arr, low, high):
    pivot = arr[high]  # 最後の要素をピボットに
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

# 計算量:
# 最悪: O(n²) — ピボットが常に最大/最小（ソート済み配列）
# 平均: O(n log n) — ランダムなピボット選択で期待値
# 最良: O(n log n)
# 空間: O(log n) — 再帰の深さ（インプレース版）
# 安定: No

# クイックソートが実務で最速な理由:
# 1. キャッシュ効率が良い（連続メモリアクセス）
# 2. 定数倍が小さい
# 3. ランダムピボットで最悪ケースを確率的に回避
# 4. インプレースでメモリ割り当てなし
```

### 2.3 ヒープソート

```python
def heapsort(arr):
    """最大ヒープを構築→最大値を取り出す"""
    n = len(arr)

    # 最大ヒープを構築（ボトムアップ）
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # 最大値を末尾に移動し、ヒープサイズを縮小
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 0)

def heapify(arr, n, i):
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

# 計算量:
# 最悪/平均/最良: O(n log n)
# 空間: O(1) — インプレース
# 安定: No

# ヒープソートの位置づけ:
# - 最悪 O(n log n) が保証（クイックソートと異なり）
# - インプレース（マージソートと異なり）
# - しかしキャッシュ効率が悪い（ランダムアクセス）
# → 実務ではクイックソートに劣ることが多い
```

---

## 3. 非比較ソート

### 3.1 O(n) のソート

```python
# 計数ソート（Counting Sort）
def counting_sort(arr, max_val):
    """値の出現回数を数えて復元"""
    count = [0] * (max_val + 1)
    for x in arr:
        count[x] += 1

    result = []
    for i in range(max_val + 1):
        result.extend([i] * count[i])
    return result

# 計算量: O(n + k)  (k = 値の範囲)
# 空間: O(n + k)
# 安定: Yes（安定版の実装では）
# 制約: 非負整数のみ、k が n に対して小さい必要あり

# 基数ソート（Radix Sort）
def radix_sort(arr):
    """各桁ごとに安定ソートを繰り返す"""
    max_val = max(arr)
    exp = 1
    while max_val // exp > 0:
        arr = counting_sort_by_digit(arr, exp)
        exp *= 10
    return arr

# 計算量: O(d × (n + k))  (d = 桁数, k = 基数)
# 用途: 整数のソート、文字列のソート

# なぜ O(n) ソートが常に使われないのか:
# - 比較ベースでない → 汎用性がない
# - 整数/固定長文字列のみ対応
# - 値の範囲 k が大きいと非効率
# - 追加メモリが必要
```

### 3.2 ソートの比較下限定理

```
比較ベースソートの下限: Ω(n log n)

  証明の概要:
  n 個の要素のソートには n! 通りの結果がありえる

  比較ベースのソートは「決定木」で表現できる:
  - 各内部ノード = 1回の比較 (a < b ?)
  - 各葉 = 1つの順列（結果）
  - 木の高さ = 最悪の比較回数

  n! 個の葉を持つ二分木の高さ h:
    2^h ≥ n!
    h ≥ log₂(n!)

  スターリングの近似: n! ≈ (n/e)^n
    h ≥ log₂((n/e)^n) = n × log₂(n/e) = Ω(n log n)

  → 比較ベースのソートでは O(n log n) が最善

  → 計数ソートや基数ソートは「比較」を使わないので
    この下限に縛られない
```

---

## 4. 実務でのソート

### 4.1 各言語の組み込みソート

```
各言語の内部実装:

  ┌──────────────┬──────────────────┬──────────────────────┐
  │ 言語         │ アルゴリズム      │ 特徴                 │
  ├──────────────┼──────────────────┼──────────────────────┤
  │ Python       │ Timsort          │ マージ+挿入の混合    │
  │ Java         │ Dual-Pivot QS    │ プリミティブ型       │
  │ Java         │ Timsort          │ オブジェクト型       │
  │ JavaScript   │ Timsort (V8)     │ エンジンにより異なる │
  │ C++ (STL)   │ Introsort        │ QS+ヒープ+挿入      │
  │ Rust         │ Timsort (stable) │ sort()               │
  │ Rust         │ Pattern-defeating│ sort_unstable()      │
  │ Go           │ pdqsort          │ 1.19以降             │
  └──────────────┴──────────────────┴──────────────────────┘

  Timsort（Tim Peters, 2002）:
  - Python/Java/JavaScript の標準
  - 実データによく現れる「すでにソートされた部分列(run)」を利用
  - 短い run には挿入ソート（キャッシュ効率が良い）
  - run をマージソートで結合
  - 最悪 O(n log n)、ソート済みデータに O(n)

  Introsort (C++ STL):
  - クイックソート + ヒープソート + 挿入ソート
  - QS の再帰が深すぎたらヒープソートに切替
  - → 最悪 O(n log n) を保証
```

### 4.2 ソートの安定性

```python
# 安定ソート: 同じキーの要素の相対順序を保持する

data = [
    ("Alice", 85),
    ("Bob", 92),
    ("Charlie", 85),
    ("Diana", 92),
]

# 安定ソート（点数順）:
# Alice(85), Charlie(85), Bob(92), Diana(92)
# → 同じ85のAlice, Charlieの順序が保持される ✓

# 不安定ソート:
# Charlie(85), Alice(85), Diana(92), Bob(92)
# → 同じ85のAlice, Charlieの順序が保証されない

# Pythonのsorted()は安定ソート（Timsort）
# → 複数キーのソートに安心して使える

# 多重キーソート（安定ソートの応用）:
# まず副キーでソートし、次に主キーでソートすると
# 主キーが同じ場合に副キーの順序が保持される
sorted_data = sorted(data, key=lambda x: x[0])  # 名前順
sorted_data = sorted(sorted_data, key=lambda x: x[1])  # 点数順
# → 点数順、同点は名前順
```

---

## 5. 実践演習

### 演習1: ソートの実装（基礎）
マージソートとクイックソートをゼロから実装し、ランダム配列・ソート済み配列・逆順配列での実行時間を計測せよ。

### 演習2: ソートの使い分け（応用）
以下の状況でどのソートが最適か判断し、理由を述べよ:
1. 100万件のログをタイムスタンプ順にソート
2. ほぼソート済みの1000件のデータ
3. 0-255の値を持つ1億件のデータ

### 演習3: カスタムソート（発展）
文字列の配列を「文字列を連結した時に最大の数になる順」にソートする関数を実装せよ。
例: ["3", "30", "34"] → ["34", "3", "30"] (34330が最大)

---

## FAQ

### Q1: なぜバブルソートは教えるのですか？
**A**: 歴史的・教育的な価値。アルゴリズムの基本概念（比較、交換、不変条件）を学ぶのに最適。実務では使わないが、計算量の違いを実感するための比較対象として有用。

### Q2: クイックソートの最悪ケース O(n²) は問題にならないのですか？
**A**: ランダムピボット選択や「3つの中央値」戦略で最悪ケースを確率的に回避できる。C++のIntrosortは深さの閾値を超えるとヒープソートに切り替えて O(n log n) を保証する。

### Q3: 10億件のデータをソートするには？
**A**: メモリに収まらない場合は外部ソート（External Sort）を使う。データをチャンクに分割し、各チャンクをソートしてディスクに書き出し、マージする（多方向マージ）。MapReduceもこの原理。

---

## まとめ

| アルゴリズム | 最悪 | 平均 | 空間 | 安定 | 用途 |
|------------|------|------|------|------|------|
| バブル | O(n²) | O(n²) | O(1) | Yes | 教育用 |
| 挿入 | O(n²) | O(n²) | O(1) | Yes | 少量/ほぼソート済み |
| マージ | O(n log n) | O(n log n) | O(n) | Yes | 安定性が必要な場合 |
| クイック | O(n²) | O(n log n) | O(log n) | No | 汎用（最速） |
| ヒープ | O(n log n) | O(n log n) | O(1) | No | メモリ制約あり |
| 計数 | O(n+k) | O(n+k) | O(n+k) | Yes | 整数/範囲が小さい |

---

## 次に読むべきガイド
→ [[03-search-algorithms.md]] — 探索アルゴリズム

---

## 参考文献
1. Cormen, T. H. et al. "Introduction to Algorithms." Chapters 6-9.
2. Sedgewick, R. "Algorithms." 4th Edition, Chapter 2.
3. Peters, T. "Timsort description." Python Developer Documentation, 2002.
