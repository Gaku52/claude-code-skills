# ヒープ — 二分ヒープ・ヒープソート・優先度キュー実装

> 効率的な最大/最小値の取得を可能にするヒープの構造、ヒープソート、優先度キューの実装を学ぶ。

---

## この章で学ぶこと

1. **二分ヒープ** の構造と配列表現
2. **ヒープソート** の仕組みと計算量
3. **優先度キュー** のヒープによる実装と応用

---

## 1. 二分ヒープの構造

```
最小ヒープ (Min-Heap):
  親 ≤ 子 が全ノードで成立

         [1]            インデックス:
        /   \              0: 1
      [3]   [2]           1: 3,  2: 2
     / \   /              3: 5,  4: 8,  5: 7
   [5] [8] [7]

配列表現: [1, 3, 2, 5, 8, 7]

  親:       i → (i-1) // 2
  左の子:   i → 2*i + 1
  右の子:   i → 2*i + 2

最大ヒープ (Max-Heap):
  親 ≥ 子 が全ノードで成立

         [9]
        /   \
      [7]   [8]
     / \   /
   [3] [5] [2]

配列表現: [9, 7, 8, 3, 5, 2]
```

---

## 2. ヒープ操作の実装

### 2.1 最小ヒープ

```python
class MinHeap:
    def __init__(self):
        self.heap = []

    def push(self, val):
        """O(log n)"""
        self.heap.append(val)
        self._sift_up(len(self.heap) - 1)

    def pop(self):
        """O(log n)"""
        if not self.heap:
            raise IndexError("heap is empty")
        self._swap(0, len(self.heap) - 1)
        val = self.heap.pop()
        if self.heap:
            self._sift_down(0)
        return val

    def peek(self):
        """O(1)"""
        if not self.heap:
            raise IndexError("heap is empty")
        return self.heap[0]

    def _sift_up(self, i):
        while i > 0:
            parent = (i - 1) // 2
            if self.heap[i] < self.heap[parent]:
                self._swap(i, parent)
                i = parent
            else:
                break

    def _sift_down(self, i):
        n = len(self.heap)
        while True:
            smallest = i
            left = 2 * i + 1
            right = 2 * i + 2
            if left < n and self.heap[left] < self.heap[smallest]:
                smallest = left
            if right < n and self.heap[right] < self.heap[smallest]:
                smallest = right
            if smallest == i:
                break
            self._swap(i, smallest)
            i = smallest

    def _swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
```

### 2.2 操作の図解

```
push(2) の例 (sift-up):

  [1, 5, 3, 8, 7]  ← push(2)
  [1, 5, 3, 8, 7, 2]

         [1]                  [1]
        /   \                /   \
      [5]   [3]    →      [5]   [2]  ← 2 < 3 なので swap
     / \   /              / \   /
   [8] [7] [2]         [8] [7] [3]

pop() の例 (sift-down):

  [1, 5, 2, 8, 7, 3]  → 先頭(1)を返す

  末尾(3)を先頭に移動:
  [3, 5, 2, 8, 7]

         [3]                  [2]
        /   \                /   \
      [5]   [2]    →      [5]   [3]  ← 3 > 2 なので swap
     / \                  / \
   [8] [7]             [8] [7]
```

---

## 3. ヒープソート

```python
def heapsort(arr):
    """ヒープソート — O(n log n) 時間, O(1) 空間"""
    n = len(arr)

    # Step 1: 最大ヒープを構築 — O(n)
    for i in range(n // 2 - 1, -1, -1):
        _sift_down_max(arr, n, i)

    # Step 2: 1つずつ取り出し — O(n log n)
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        _sift_down_max(arr, i, 0)

    return arr

def _sift_down_max(arr, n, i):
    while True:
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        if left < n and arr[left] > arr[largest]:
            largest = left
        if right < n and arr[right] > arr[largest]:
            largest = right
        if largest == i:
            break
        arr[i], arr[largest] = arr[largest], arr[i]
        i = largest
```

```
ヒープソートの過程:

Step 1: 最大ヒープ構築
  [4, 1, 3, 2, 5]  →  [5, 4, 3, 2, 1]

Step 2: 先頭と末尾を swap + sift-down
  [5, 4, 3, 2, 1] → swap(5,1) → [1, 4, 3, 2, |5]
  → sift-down → [4, 2, 3, 1, |5]
  → swap(4,1) → [1, 2, 3, |4, 5]
  → sift-down → [3, 2, 1, |4, 5]
  → ... → [1, 2, 3, 4, 5]
```

---

## 4. heapq モジュール

```python
import heapq

# 最小ヒープ操作
nums = [5, 3, 8, 1, 2]
heapq.heapify(nums)          # O(n) でヒープ化
print(nums)                   # [1, 2, 8, 5, 3]

heapq.heappush(nums, 0)      # O(log n)
print(heapq.heappop(nums))   # 0 — O(log n)

# Top-K（最小 K 個）
top3 = heapq.nsmallest(3, [5, 3, 8, 1, 2])  # [1, 2, 3]
# Top-K（最大 K 個）
top3 = heapq.nlargest(3, [5, 3, 8, 1, 2])   # [8, 5, 3]

# 最大ヒープ（符号反転のトリック）
max_heap = []
for x in [5, 3, 8, 1, 2]:
    heapq.heappush(max_heap, -x)
print(-heapq.heappop(max_heap))  # 8
```

---

## 5. 応用: K番目に大きい要素

```python
def kth_largest(nums, k):
    """K番目に大きい要素 — O(n log k)"""
    import heapq
    # サイズ k の最小ヒープを維持
    heap = nums[:k]
    heapq.heapify(heap)
    for num in nums[k:]:
        if num > heap[0]:
            heapq.heapreplace(heap, num)
    return heap[0]

# kth_largest([3, 2, 1, 5, 6, 4], 2) → 5
```

---

## 6. 比較表

### 表1: ヒープ操作の計算量

| 操作 | 計算量 | 説明 |
|------|--------|------|
| peek (最小/最大) | O(1) | 根を参照 |
| push | O(log n) | 末尾追加 + sift-up |
| pop | O(log n) | 根削除 + sift-down |
| heapify | O(n) | ボトムアップ構築 |
| 任意要素の探索 | O(n) | ヒープは探索に向かない |
| 任意要素の削除 | O(n) | 探索 O(n) + sift O(log n) |

### 表2: ソートアルゴリズムとの比較

| アルゴリズム | 平均 | 最悪 | 空間 | 安定 | 特徴 |
|-------------|------|------|------|------|------|
| ヒープソート | O(n log n) | O(n log n) | O(1) | 不安定 | in-place |
| マージソート | O(n log n) | O(n log n) | O(n) | 安定 | 外部ソート向き |
| クイックソート | O(n log n) | O(n²) | O(log n) | 不安定 | 実測最速 |
| Tim ソート | O(n log n) | O(n log n) | O(n) | 安定 | Python 標準 |

---

## 7. アンチパターン

### アンチパターン1: heapify の代わりに逐次 push

```python
import heapq

# BAD: n 回の push — O(n log n)
heap = []
for x in data:
    heapq.heappush(heap, x)

# GOOD: heapify — O(n)
heap = list(data)
heapq.heapify(heap)
```

### アンチパターン2: ヒープで最小/最大以外を頻繁に検索

```python
# BAD: ヒープで特定値を探索 — O(n)
def find_in_heap(heap, target):
    for item in heap:
        if item == target:
            return True
    return False

# ヒープは最小値/最大値の取得に特化
# 任意の要素検索が必要なら set や dict を併用する
```

---

## 8. FAQ

### Q1: heapify がなぜ O(n) で済むのか？

**A:** ボトムアップに sift-down する。葉ノード（約 n/2 個）は sift-down 不要。高さ h のノード数は n/2^(h+1) で、sift-down コストは O(h)。合計 Σ(h × n/2^(h+1)) = O(n)。

### Q2: Python の heapq に最大ヒープはないのか？

**A:** 標準では最小ヒープのみ。最大ヒープは値を符号反転して挿入する（`heappush(h, -val)`）。サードパーティの `heapdict` や自作クラスで対応も可能。

### Q3: ヒープと BST の使い分けは？

**A:** 最小/最大値だけが必要ならヒープ（O(1) 参照）。範囲検索や順序走査が必要なら BST。ヒープは配列で実装できるためメモリ効率が良い。

---

## 9. まとめ

| 項目 | ポイント |
|------|---------|
| 二分ヒープ | 完全二分木。配列で効率的に表現 |
| 最小/最大ヒープ | 根が最小/最大値。O(1) で参照 |
| sift-up/down | 挿入/削除後のヒープ性質の復元 — O(log n) |
| heapify | ボトムアップ構築で O(n) |
| ヒープソート | O(n log n)、in-place、不安定 |
| 優先度キュー | ヒープで実装。Dijkstra 等に使用 |

---

## 次に読むべきガイド

- [グラフ — 表現方法と重み付きグラフ](./06-graphs.md)
- [最短経路 — Dijkstra とヒープの活用](../02-algorithms/03-shortest-path.md)

---

## 参考文献

1. Cormen, T.H. et al. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. — 第6章「Heapsort」
2. Williams, J.W.J. (1964). "Algorithm 232: Heapsort." *Communications of the ACM*, 7(6), 347-348.
3. Python Documentation. "heapq — Heap queue algorithm." — https://docs.python.org/3/library/heapq.html
