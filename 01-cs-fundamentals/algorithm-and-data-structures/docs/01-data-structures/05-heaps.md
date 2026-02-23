# ヒープ — 二分ヒープ・ヒープソート・優先度キュー実装

> 効率的な最大/最小値の取得を可能にするヒープの構造、ヒープソート、優先度キューの実装を学ぶ。

---

## この章で学ぶこと

1. **二分ヒープ** の構造と配列表現
2. **ヒープソート** の仕組みと計算量
3. **優先度キュー** のヒープによる実装と応用
4. **各種ヒープ** — d-ary ヒープ、フィボナッチヒープ、インデックス付きヒープ
5. **実務応用** — Top-K、中央値、タスクスケジューラ、マージ K ソート済みリスト

---

## 1. 二分ヒープの構造

### 1.1 ヒープの定義

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
  葉ノード: i >= n // 2

最大ヒープ (Max-Heap):
  親 ≥ 子 が全ノードで成立

         [9]
        /   \
      [7]   [8]
     / \   /
   [3] [5] [2]

配列表現: [9, 7, 8, 3, 5, 2]
```

### 1.2 ヒープの重要な性質

```python
# 二分ヒープの性質:
# 1. 完全二分木: 最後のレベル以外は完全に埋まっている
# 2. ヒープ順序性: 親 ≤ 子（最小ヒープ）or 親 ≥ 子（最大ヒープ）
# 3. 配列で効率的に表現可能（ポインタ不要）
# 4. 根が最小値（最小ヒープ）or 最大値（最大ヒープ）

# n ノードのヒープの性質:
# - 高さ: floor(log2(n))
# - 葉ノード数: ceil(n/2)
# - 内部ノード数: floor(n/2)
# - 最後の内部ノード: インデックス n//2 - 1

def heap_properties(n):
    """n ノードのヒープの性質"""
    import math
    height = math.floor(math.log2(n)) if n > 0 else 0
    leaves = math.ceil(n / 2)
    internal = n // 2
    last_internal = n // 2 - 1
    print(f"ノード数: {n}")
    print(f"高さ: {height}")
    print(f"葉ノード数: {leaves}")
    print(f"内部ノード数: {internal}")
    print(f"最後の内部ノード: インデックス {last_internal}")

heap_properties(10)
# ノード数: 10, 高さ: 3, 葉ノード数: 5, 内部ノード数: 5
```

---

## 2. ヒープ操作の実装

### 2.1 最小ヒープ

```python
class MinHeap:
    """最小ヒープ: 根が最小値

    用途: 優先度キュー、Dijkstra、ハフマン符号化
    """
    def __init__(self):
        self.heap = []

    def __len__(self):
        return len(self.heap)

    def __bool__(self):
        return len(self.heap) > 0

    def push(self, val):
        """O(log n) — 末尾に追加して上方向に修正"""
        self.heap.append(val)
        self._sift_up(len(self.heap) - 1)

    def pop(self):
        """O(log n) — 根を取り出して下方向に修正"""
        if not self.heap:
            raise IndexError("heap is empty")
        self._swap(0, len(self.heap) - 1)
        val = self.heap.pop()
        if self.heap:
            self._sift_down(0)
        return val

    def peek(self):
        """O(1) — 最小値を返す（削除しない）"""
        if not self.heap:
            raise IndexError("heap is empty")
        return self.heap[0]

    def push_pop(self, val):
        """push + pop を最適化 — O(log n)
        push してから pop するより効率的
        """
        if self.heap and self.heap[0] < val:
            val, self.heap[0] = self.heap[0], val
            self._sift_down(0)
        return val

    def replace(self, val):
        """pop + push を最適化 — O(log n)
        pop してから push するより効率的
        """
        if not self.heap:
            raise IndexError("heap is empty")
        old = self.heap[0]
        self.heap[0] = val
        self._sift_down(0)
        return old

    def _sift_up(self, i):
        """上方向修正: 親より小さければ交換"""
        while i > 0:
            parent = (i - 1) // 2
            if self.heap[i] < self.heap[parent]:
                self._swap(i, parent)
                i = parent
            else:
                break

    def _sift_down(self, i):
        """下方向修正: 子の最小値より大きければ交換"""
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

    @classmethod
    def heapify(cls, arr):
        """配列からヒープを構築 — O(n)
        ボトムアップに sift_down することで O(n) を達成
        """
        heap = cls()
        heap.heap = list(arr)
        n = len(heap.heap)
        # 最後の内部ノードから根に向かって sift_down
        for i in range(n // 2 - 1, -1, -1):
            heap._sift_down(i)
        return heap

# 使用例
h = MinHeap()
h.push(5)
h.push(3)
h.push(8)
h.push(1)
print(h.peek())  # 1
print(h.pop())   # 1
print(h.pop())   # 3

# heapify
h2 = MinHeap.heapify([5, 3, 8, 1, 2, 7])
print(h2.pop())  # 1
print(h2.pop())  # 2
```

### 2.2 最大ヒープ

```python
class MaxHeap:
    """最大ヒープ: 根が最大値

    用途: 優先度の高いタスクの取得、降順のストリーミング処理
    """
    def __init__(self):
        self.heap = []

    def __len__(self):
        return len(self.heap)

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
            if self.heap[i] > self.heap[parent]:  # > に変更
                self._swap(i, parent)
                i = parent
            else:
                break

    def _sift_down(self, i):
        n = len(self.heap)
        while True:
            largest = i
            left = 2 * i + 1
            right = 2 * i + 2
            if left < n and self.heap[left] > self.heap[largest]:
                largest = left
            if right < n and self.heap[right] > self.heap[largest]:
                largest = right
            if largest == i:
                break
            self._swap(i, largest)
            i = largest

    def _swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
```

### 2.3 操作の図解

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

heapify の過程 (ボトムアップ):

  入力: [4, 10, 3, 5, 1]

  Step 0: 初期状態
         [4]
        /   \
      [10]  [3]
     / \
    [5] [1]

  Step 1: i=1, sift_down(10)
         [4]
        /   \
      [1]   [3]     ← 10 > 1 なので swap
     / \
    [5] [10]

  Step 2: i=0, sift_down(4)
         [1]
        /   \
      [4]   [3]     ← 4 > 1 なので swap, さらに 4 と 5 は交換不要
     / \
    [5] [10]

  結果: [1, 4, 3, 5, 10] — 正しいヒープ
```

---

## 3. ヒープソート

### 3.1 基本実装

```python
def heapsort(arr):
    """ヒープソート — O(n log n) 時間, O(1) 空間

    Step 1: 最大ヒープを構築 — O(n)
    Step 2: 根と末尾を交換し、ヒープサイズを縮小 — O(n log n)
    """
    n = len(arr)

    # Step 1: 最大ヒープを構築 — O(n)
    for i in range(n // 2 - 1, -1, -1):
        _sift_down_max(arr, n, i)

    # Step 2: 1つずつ取り出し — O(n log n)
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]  # 最大値を末尾に
        _sift_down_max(arr, i, 0)        # 残りをヒープ化

    return arr

def _sift_down_max(arr, n, i):
    """最大ヒープの下方向修正"""
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

# 使用例
data = [12, 11, 13, 5, 6, 7, 3, 1, 9, 2, 4, 8, 10, 14, 15]
print(heapsort(data))
# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
```

### 3.2 ヒープソートの過程

```
ヒープソートの過程:

Step 1: 最大ヒープ構築
  [4, 1, 3, 2, 5]  →  [5, 4, 3, 2, 1]

         [4]               [5]
        /   \      →      /   \
      [1]   [3]         [4]   [3]
     / \               / \
    [2] [5]           [2] [1]

Step 2: 先頭と末尾を swap + sift-down
  Round 1: swap(5,1)
  [5, 4, 3, 2, 1] → [1, 4, 3, 2, |5]
  sift-down → [4, 2, 3, 1, |5]

  Round 2: swap(4,1)
  [4, 2, 3, 1, |5] → [1, 2, 3, |4, 5]
  sift-down → [3, 2, 1, |4, 5]

  Round 3: swap(3,1)
  [3, 2, 1, |4, 5] → [1, 2, |3, 4, 5]

  Round 4: swap(2,1)
  [2, 1, |3, 4, 5] → [1, |2, 3, 4, 5]

  結果: [1, 2, 3, 4, 5]
```

### 3.3 部分ソート（Top-K）の最適化

```python
def partial_sort_top_k(arr, k):
    """配列の上位 k 個のみをソートして返す — O(n + k log n)

    全体をソートする O(n log n) より効率的（k << n の場合）
    """
    n = len(arr)
    # Step 1: 最大ヒープ構築 — O(n)
    for i in range(n // 2 - 1, -1, -1):
        _sift_down_max(arr, n, i)

    # Step 2: 上位 k 個だけ取り出し — O(k log n)
    result = []
    heap_size = n
    for _ in range(min(k, n)):
        result.append(arr[0])
        arr[0] = arr[heap_size - 1]
        heap_size -= 1
        _sift_down_max(arr, heap_size, 0)

    return result

data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
print(partial_sort_top_k(data[:], 3))  # [9, 6, 5]
```

---

## 4. heapq モジュール

### 4.1 基本操作

```python
import heapq

# 最小ヒープ操作
nums = [5, 3, 8, 1, 2]
heapq.heapify(nums)          # O(n) でヒープ化
print(nums)                   # [1, 2, 8, 5, 3]

heapq.heappush(nums, 0)      # O(log n)
print(heapq.heappop(nums))   # 0 — O(log n)

# heappushpop: push + pop を最適化
result = heapq.heappushpop(nums, 4)  # push(4) して pop
print(result)  # 1

# heapreplace: pop + push を最適化
result = heapq.heapreplace(nums, 10)  # pop して push(10)
print(result)  # 2

# Top-K（最小 K 個）
top3 = heapq.nsmallest(3, [5, 3, 8, 1, 2])  # [1, 2, 3]
# Top-K（最大 K 個）
top3 = heapq.nlargest(3, [5, 3, 8, 1, 2])   # [8, 5, 3]

# 最大ヒープ（符号反転のトリック）
max_heap = []
for x in [5, 3, 8, 1, 2]:
    heapq.heappush(max_heap, -x)
print(-heapq.heappop(max_heap))  # 8

# キー付きヒープ（タプルを使用）
tasks = [(3, "low priority"), (1, "high priority"), (2, "medium")]
heapq.heapify(tasks)
print(heapq.heappop(tasks))  # (1, 'high priority')
```

### 4.2 heapq の内部実装

```python
# heapq のパフォーマンス特性:
# - C実装（CPython）: 非常に高速
# - heapify: O(n)
# - heappush/heappop: O(log n)
# - nsmallest/nlargest:
#   - k が小さい場合: ヒープを使用 O(n + k log n)
#   - k が n に近い場合: ソートを使用 O(n log n)
#   - 内部で自動的に最適な方法を選択

# nsmallest/nlargest の使い分け
import heapq

data = list(range(1000000))

# k << n の場合: nsmallest が効率的
top10 = heapq.nsmallest(10, data)  # O(n)

# k が n に近い場合: sorted の方が効率的
# heapq.nsmallest(999990, data)  # 内部で sorted に切り替え

# k = 1 の場合: min/max が最速
minimum = min(data)  # O(n)
maximum = max(data)  # O(n)
```

### 4.3 heapq.merge — K個のソート済みイテラブルのマージ

```python
import heapq

# ソート済みリストのマージ — O(n log k)
list1 = [1, 4, 7, 10]
list2 = [2, 5, 8, 11]
list3 = [3, 6, 9, 12]

merged = list(heapq.merge(list1, list2, list3))
print(merged)  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# key 引数を使ったカスタムマージ
records1 = [(1, "a"), (3, "c"), (5, "e")]
records2 = [(2, "b"), (4, "d"), (6, "f")]
merged = list(heapq.merge(records1, records2, key=lambda x: x[0]))
print(merged)  # [(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd'), (5, 'e'), (6, 'f')]

# 外部ソート（大容量ファイルのソート）のマージフェーズ
# 各チャンクをソートしてファイルに書き出し
# heapq.merge で K 個のチャンクをマージ
def external_sort_merge(sorted_files, output_file):
    """外部ソートのマージフェーズ"""
    import itertools
    file_iters = [open(f) for f in sorted_files]
    with open(output_file, 'w') as out:
        for line in heapq.merge(*file_iters, key=lambda x: int(x.strip())):
            out.write(line)
    for f in file_iters:
        f.close()
```

---

## 5. 優先度キューの実装

### 5.1 基本的な優先度キュー

```python
import heapq

class PriorityQueue:
    """優先度キュー: 優先度の高い要素を先に取り出す

    heapq ベースの実装。
    タプル (priority, counter, item) を使用して:
    1. 同一優先度での FIFO 順序
    2. 比較不可能なオブジェクトへの対応
    を実現する。
    """
    def __init__(self):
        self._queue = []
        self._counter = 0  # FIFO 順序保持用

    def push(self, item, priority=0):
        """O(log n)"""
        heapq.heappush(self._queue, (priority, self._counter, item))
        self._counter += 1

    def pop(self):
        """O(log n) — 最も優先度の高い（値が小さい）要素を返す"""
        if not self._queue:
            raise IndexError("queue is empty")
        return heapq.heappop(self._queue)[2]

    def peek(self):
        """O(1)"""
        if not self._queue:
            raise IndexError("queue is empty")
        return self._queue[0][2]

    def __len__(self):
        return len(self._queue)

    def __bool__(self):
        return len(self._queue) > 0

# 使用例
pq = PriorityQueue()
pq.push("low priority task", priority=3)
pq.push("high priority task", priority=1)
pq.push("medium priority task", priority=2)
pq.push("another high priority", priority=1)

while pq:
    print(pq.pop())
# high priority task
# another high priority  (同一優先度は FIFO)
# medium priority task
# low priority task
```

### 5.2 削除可能な優先度キュー（遅延削除）

```python
import heapq

class LazyDeletionPQ:
    """遅延削除対応の優先度キュー

    ヒープから要素を直接削除する代わりに、
    削除マークを付けて pop 時にスキップする。
    Dijkstra アルゴリズムの実装で頻出。
    """
    def __init__(self):
        self._queue = []
        self._counter = 0
        self._deleted = set()  # 削除済みのカウンタID

    def push(self, item, priority=0):
        """O(log n)"""
        entry_id = self._counter
        heapq.heappush(self._queue, (priority, entry_id, item))
        self._counter += 1
        return entry_id

    def pop(self):
        """O(log n) 償却 — 削除済み要素をスキップ"""
        while self._queue:
            priority, entry_id, item = heapq.heappop(self._queue)
            if entry_id not in self._deleted:
                return item
            self._deleted.discard(entry_id)
        raise IndexError("queue is empty")

    def delete(self, entry_id):
        """O(1) — 遅延削除マーク"""
        self._deleted.add(entry_id)

    def __len__(self):
        return len(self._queue) - len(self._deleted)

# 使用例: 優先度の更新（古いエントリを削除して新しく追加）
pq = LazyDeletionPQ()
id1 = pq.push("task A", priority=5)
id2 = pq.push("task B", priority=3)

# task A の優先度を更新（5 → 1）
pq.delete(id1)
id1_new = pq.push("task A", priority=1)

print(pq.pop())  # "task A" (優先度 1)
print(pq.pop())  # "task B" (優先度 3)
```

### 5.3 インデックス付きヒープ

```python
class IndexedMinHeap:
    """インデックス付き最小ヒープ

    各キーの優先度を O(log n) で更新可能。
    Dijkstra、Prim のアルゴリズムに最適。

    - insert(key, priority): O(log n)
    - pop(): O(log n)
    - decrease_key(key, new_priority): O(log n)
    - contains(key): O(1)
    """
    def __init__(self, capacity=100):
        self.heap = []          # (priority, key) のリスト
        self.key_to_idx = {}    # key → ヒープ内のインデックス
        self.key_to_priority = {}

    def __len__(self):
        return len(self.heap)

    def __contains__(self, key):
        return key in self.key_to_idx

    def insert(self, key, priority):
        """O(log n)"""
        if key in self.key_to_idx:
            raise ValueError(f"Key {key} already exists")
        idx = len(self.heap)
        self.heap.append((priority, key))
        self.key_to_idx[key] = idx
        self.key_to_priority[key] = priority
        self._sift_up(idx)

    def pop(self):
        """O(log n) — 最小優先度の要素を返す"""
        if not self.heap:
            raise IndexError("heap is empty")
        priority, key = self.heap[0]
        self._swap(0, len(self.heap) - 1)
        self.heap.pop()
        del self.key_to_idx[key]
        del self.key_to_priority[key]
        if self.heap:
            self._sift_down(0)
        return key, priority

    def decrease_key(self, key, new_priority):
        """O(log n) — キーの優先度を下げる"""
        if key not in self.key_to_idx:
            raise KeyError(key)
        if new_priority >= self.key_to_priority[key]:
            return  # 新しい優先度が高くない場合は何もしない
        idx = self.key_to_idx[key]
        self.heap[idx] = (new_priority, key)
        self.key_to_priority[key] = new_priority
        self._sift_up(idx)

    def _sift_up(self, i):
        while i > 0:
            parent = (i - 1) // 2
            if self.heap[i][0] < self.heap[parent][0]:
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
            if left < n and self.heap[left][0] < self.heap[smallest][0]:
                smallest = left
            if right < n and self.heap[right][0] < self.heap[smallest][0]:
                smallest = right
            if smallest == i:
                break
            self._swap(i, smallest)
            i = smallest

    def _swap(self, i, j):
        ki = self.heap[i][1]
        kj = self.heap[j][1]
        self.key_to_idx[ki] = j
        self.key_to_idx[kj] = i
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

# 使用例: Dijkstra での使用
ipq = IndexedMinHeap()
ipq.insert("A", 0)   # 始点: 距離 0
ipq.insert("B", 10)
ipq.insert("C", 5)
ipq.insert("D", float('inf'))

# 距離の更新
ipq.decrease_key("B", 3)  # B への距離が 10 → 3 に改善

key, dist = ipq.pop()
print(f"{key}: {dist}")  # A: 0
key, dist = ipq.pop()
print(f"{key}: {dist}")  # B: 3
key, dist = ipq.pop()
print(f"{key}: {dist}")  # C: 5
```

---

## 6. 各種ヒープ

### 6.1 d-ary ヒープ

```python
class DaryHeap:
    """d-ary ヒープ: 各ノードが d 個の子を持つ

    d=2: 二分ヒープ（標準）
    d=4: 四分ヒープ（キャッシュ効率が良い場合がある）

    - sift_up: O(log_d n) — d が大きいほど高さが低い
    - sift_down: O(d * log_d n) — d が大きいほど子の比較が多い
    - decrease_key が多い場合は d を大きくすると有利
      （Dijkstra の密グラフなど）
    """
    def __init__(self, d=4):
        self.d = d
        self.heap = []

    def push(self, val):
        self.heap.append(val)
        self._sift_up(len(self.heap) - 1)

    def pop(self):
        if not self.heap:
            raise IndexError("heap is empty")
        self.heap[0], self.heap[-1] = self.heap[-1], self.heap[0]
        val = self.heap.pop()
        if self.heap:
            self._sift_down(0)
        return val

    def _parent(self, i):
        return (i - 1) // self.d

    def _children(self, i):
        start = self.d * i + 1
        return range(start, min(start + self.d, len(self.heap)))

    def _sift_up(self, i):
        while i > 0:
            parent = self._parent(i)
            if self.heap[i] < self.heap[parent]:
                self.heap[i], self.heap[parent] = self.heap[parent], self.heap[i]
                i = parent
            else:
                break

    def _sift_down(self, i):
        while True:
            smallest = i
            for child in self._children(i):
                if self.heap[child] < self.heap[smallest]:
                    smallest = child
            if smallest == i:
                break
            self.heap[i], self.heap[smallest] = self.heap[smallest], self.heap[i]
            i = smallest

# d の選択指針:
# - d=2: 標準的な二分ヒープ。最もバランスが良い
# - d=4: decrease_key が多い場合。Dijkstra の密グラフに有利
# - d=8+: キャッシュラインに合わせた最適化が可能
```

### 6.2 フィボナッチヒープ（概念）

```python
# フィボナッチヒープは理論的に最も効率的なヒープ:
#
# | 操作           | 二分ヒープ  | フィボナッチヒープ |
# |---------------|-----------|----------------|
# | insert        | O(log n)  | O(1) 償却      |
# | peek          | O(1)      | O(1)           |
# | pop           | O(log n)  | O(log n) 償却   |
# | decrease_key  | O(log n)  | O(1) 償却      |
# | merge         | O(n)      | O(1)           |
#
# Dijkstra に使うと O(V log V + E) を達成（二分ヒープでは O((V+E) log V)）
#
# ただし実装が非常に複雑で、定数係数が大きいため、
# 実務では二分ヒープの方が高速な場合が多い。
# 主に理論的な計算量の分析で使用される。

# 簡易的なマージ可能ヒープ（Pairing Heap）
class PairingHeapNode:
    def __init__(self, val):
        self.val = val
        self.children = []

class PairingHeap:
    """ペアリングヒープ: フィボナッチヒープの簡易版

    実装が簡単で実測性能も良い。
    merge が O(1) で可能。
    """
    def __init__(self):
        self.root = None

    def push(self, val):
        """O(1)"""
        new_node = PairingHeapNode(val)
        self.root = self._merge(self.root, new_node)

    def peek(self):
        """O(1)"""
        if not self.root:
            raise IndexError("heap is empty")
        return self.root.val

    def pop(self):
        """O(log n) 償却"""
        if not self.root:
            raise IndexError("heap is empty")
        val = self.root.val
        children = self.root.children

        # 二段階マージ (two-pass pairing)
        if not children:
            self.root = None
        else:
            # 1段目: 隣接ペアをマージ
            merged = []
            for i in range(0, len(children), 2):
                if i + 1 < len(children):
                    merged.append(self._merge(children[i], children[i + 1]))
                else:
                    merged.append(children[i])

            # 2段目: 右から左にマージ
            result = merged[-1]
            for i in range(len(merged) - 2, -1, -1):
                result = self._merge(result, merged[i])
            self.root = result

        return val

    def _merge(self, h1, h2):
        """2つのヒープをマージ — O(1)"""
        if not h1:
            return h2
        if not h2:
            return h1
        if h1.val <= h2.val:
            h1.children.append(h2)
            return h1
        else:
            h2.children.append(h1)
            return h2

    def merge_with(self, other):
        """別のヒープとマージ — O(1)"""
        self.root = self._merge(self.root, other.root)

# 使用例
ph = PairingHeap()
for x in [5, 3, 8, 1, 7]:
    ph.push(x)
print(ph.pop())  # 1
print(ph.pop())  # 3
```

---

## 7. 実務応用

### 7.1 K番目に大きい要素

```python
def kth_largest(nums, k):
    """K番目に大きい要素 — O(n log k)

    サイズ k の最小ヒープを維持。
    ヒープの根が常に k 番目に大きい要素になる。
    """
    import heapq
    # サイズ k の最小ヒープを維持
    heap = nums[:k]
    heapq.heapify(heap)
    for num in nums[k:]:
        if num > heap[0]:
            heapq.heapreplace(heap, num)
    return heap[0]

print(kth_largest([3, 2, 1, 5, 6, 4], 2))  # 5
print(kth_largest([3, 2, 3, 1, 2, 4, 5, 5, 6], 4))  # 4
```

### 7.2 ストリーミング中央値

```python
import heapq

class MedianFinder:
    """ストリーミング中央値 — O(log n) 挿入, O(1) 取得

    2つのヒープを使用:
    - max_heap: 小さい方の半分（最大ヒープ）
    - min_heap: 大きい方の半分（最小ヒープ）

    max_heap の根 ≤ min_heap の根
    サイズ差は最大1
    """
    def __init__(self):
        self.max_heap = []  # 小さい方の半分（符号反転で最大ヒープ）
        self.min_heap = []  # 大きい方の半分

    def add_num(self, num):
        """O(log n)"""
        # まず max_heap に追加
        heapq.heappush(self.max_heap, -num)
        # max_heap の最大値を min_heap に移動
        heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))
        # min_heap が大きすぎたら max_heap に戻す
        if len(self.min_heap) > len(self.max_heap):
            heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))

    def find_median(self):
        """O(1)"""
        if len(self.max_heap) > len(self.min_heap):
            return -self.max_heap[0]
        return (-self.max_heap[0] + self.min_heap[0]) / 2

# 使用例
mf = MedianFinder()
mf.add_num(1)
print(mf.find_median())  # 1.0
mf.add_num(2)
print(mf.find_median())  # 1.5
mf.add_num(3)
print(mf.find_median())  # 2.0
mf.add_num(4)
print(mf.find_median())  # 2.5
mf.add_num(5)
print(mf.find_median())  # 3.0
```

### 7.3 K個のソート済みリストのマージ

```python
import heapq

def merge_k_sorted_lists(lists):
    """K個のソート済みリストをマージ — O(N log K)

    N = 全要素数, K = リスト数
    ヒープに各リストの先頭要素を入れ、最小値を取り出す。
    """
    heap = []
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))

    result = []
    while heap:
        val, list_idx, elem_idx = heapq.heappop(heap)
        result.append(val)
        if elem_idx + 1 < len(lists[list_idx]):
            next_val = lists[list_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))

    return result

# 使用例
lists = [
    [1, 4, 5],
    [1, 3, 4],
    [2, 6],
]
print(merge_k_sorted_lists(lists))  # [1, 1, 2, 3, 4, 4, 5, 6]
```

### 7.4 タスクスケジューラ

```python
import heapq
from collections import Counter

def least_interval(tasks, n):
    """タスクスケジューラ: 同一タスク間に最低 n 個の間隔

    例: tasks = ["A","A","A","B","B","B"], n = 2
    結果: A B _ A B _ A B → 長さ 8

    貪欲法 + 最大ヒープ: 頻度の高いタスクから優先的に実行
    """
    freq = Counter(tasks)
    max_heap = [-count for count in freq.values()]
    heapq.heapify(max_heap)

    time = 0
    cooldown = []  # (再利用可能時刻, 残り回数)

    while max_heap or cooldown:
        time += 1

        if max_heap:
            count = heapq.heappop(max_heap) + 1  # -値なので +1 で回数減少
            if count != 0:
                cooldown.append((time + n, count))
        # else: アイドル

        # クールダウン終了のタスクをヒープに戻す
        if cooldown and cooldown[0][0] == time:
            _, count = cooldown.pop(0)
            heapq.heappush(max_heap, count)

    return time

print(least_interval(["A","A","A","B","B","B"], 2))  # 8
print(least_interval(["A","A","A","B","B","B"], 0))  # 6
```

### 7.5 スライディングウィンドウの最大値

```python
from collections import deque

def max_sliding_window(nums, k):
    """スライディングウィンドウの最大値 — O(n)

    単調減少のデック（monotonic deque）を使用。
    ヒープ版は O(n log n) だが、デック版は O(n)。
    """
    if not nums or k == 0:
        return []

    dq = deque()  # インデックスを格納（単調減少）
    result = []

    for i in range(len(nums)):
        # ウィンドウ外の要素を除去
        while dq and dq[0] < i - k + 1:
            dq.popleft()

        # 新しい要素より小さい要素を除去
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()

        dq.append(i)

        # ウィンドウが完成したら結果に追加
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result

print(max_sliding_window([1, 3, -1, -3, 5, 3, 6, 7], 3))
# [3, 3, 5, 5, 6, 7]

# ヒープ版（比較用）
import heapq

def max_sliding_window_heap(nums, k):
    """ヒープ版 — O(n log n)"""
    if not nums or k == 0:
        return []

    heap = []  # (-値, インデックス)
    result = []

    for i in range(len(nums)):
        heapq.heappush(heap, (-nums[i], i))

        if i >= k - 1:
            # ウィンドウ外の要素をスキップ
            while heap[0][1] < i - k + 1:
                heapq.heappop(heap)
            result.append(-heap[0][0])

    return result
```

### 7.6 最大スコアの仕事選択

```python
import heapq

def max_performance(n, speed, efficiency, k):
    """最大パフォーマンス: k人以下のチームで
    sum(speed) * min(efficiency) を最大化

    効率降順にソートし、速度の最小ヒープで
    合計速度を管理する。
    """
    # 効率の降順にソート
    engineers = sorted(zip(efficiency, speed), reverse=True)
    max_perf = 0
    speed_sum = 0
    min_heap = []

    for eff, spd in engineers:
        heapq.heappush(min_heap, spd)
        speed_sum += spd

        if len(min_heap) > k:
            speed_sum -= heapq.heappop(min_heap)

        max_perf = max(max_perf, speed_sum * eff)

    return max_perf

# 使用例
print(max_performance(6, [2,10,3,1,5,8], [5,4,3,9,7,2], 2))  # 60
```

### 7.7 Dijkstra のアルゴリズム

```python
import heapq

def dijkstra(graph, start):
    """Dijkstra の最短経路アルゴリズム — O((V+E) log V)

    graph: {node: [(neighbor, weight), ...]}
    """
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    prev = {node: None for node in graph}
    heap = [(0, start)]

    while heap:
        dist, u = heapq.heappop(heap)

        if dist > distances[u]:
            continue  # 古いエントリをスキップ（遅延削除）

        for v, weight in graph[u]:
            new_dist = dist + weight
            if new_dist < distances[v]:
                distances[v] = new_dist
                prev[v] = u
                heapq.heappush(heap, (new_dist, v))

    return distances, prev

def reconstruct_path(prev, start, end):
    """最短経路を復元"""
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = prev[current]
    return path[::-1] if path[-1] == start else []

# 使用例
graph = {
    'A': [('B', 4), ('C', 2)],
    'B': [('D', 3), ('C', 1)],
    'C': [('B', 1), ('D', 5)],
    'D': [],
}
distances, prev = dijkstra(graph, 'A')
print(distances)  # {'A': 0, 'B': 3, 'C': 2, 'D': 6}
print(reconstruct_path(prev, 'A', 'D'))  # ['A', 'C', 'B', 'D']
```

### 7.8 ハフマン符号化

```python
import heapq
from collections import Counter

class HuffmanNode:
    def __init__(self, char=None, freq=0, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq

def huffman_encoding(text):
    """ハフマン符号化 — O(n log n)

    出現頻度に基づいて可変長符号を割り当てる。
    頻度の高い文字ほど短い符号 → 全体の圧縮率が向上。
    """
    if not text:
        return {}, ""

    # 頻度カウント
    freq = Counter(text)

    # 優先度キューに葉ノードを追加
    heap = [HuffmanNode(char=ch, freq=f) for ch, f in freq.items()]
    heapq.heapify(heap)

    # 1文字しかない場合の特殊処理
    if len(heap) == 1:
        return {heap[0].char: "0"}, "0" * len(text)

    # ハフマン木の構築
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, merged)

    # 符号の生成
    codes = {}
    def build_codes(node, code=""):
        if node.char is not None:
            codes[node.char] = code
            return
        build_codes(node.left, code + "0")
        build_codes(node.right, code + "1")

    build_codes(heap[0])

    # エンコード
    encoded = "".join(codes[ch] for ch in text)
    return codes, encoded

# 使用例
text = "this is an example of huffman encoding"
codes, encoded = huffman_encoding(text)
print("符号表:")
for char, code in sorted(codes.items(), key=lambda x: len(x[1])):
    print(f"  '{char}': {code}")
print(f"\n原文: {len(text) * 8} bits (ASCII)")
print(f"圧縮後: {len(encoded)} bits")
print(f"圧縮率: {len(encoded) / (len(text) * 8) * 100:.1f}%")
```

---

## 8. 比較表

### 表1: ヒープ操作の計算量

| 操作 | 二分ヒープ | d-ary ヒープ | フィボナッチ | ペアリング |
|------|-----------|-------------|------------|----------|
| peek | O(1) | O(1) | O(1) | O(1) |
| push | O(log n) | O(log_d n) | O(1) 償却 | O(1) |
| pop | O(log n) | O(d log_d n) | O(log n) 償却 | O(log n) 償却 |
| decrease_key | O(log n) | O(log_d n) | O(1) 償却 | O(1) 償却 |
| merge | O(n) | O(n) | O(1) | O(1) |
| heapify | O(n) | O(n) | - | - |

### 表2: ソートアルゴリズムとの比較

| アルゴリズム | 平均 | 最悪 | 空間 | 安定 | 特徴 |
|-------------|------|------|------|------|------|
| ヒープソート | O(n log n) | O(n log n) | O(1) | 不安定 | in-place、最悪保証 |
| マージソート | O(n log n) | O(n log n) | O(n) | 安定 | 外部ソート向き |
| クイックソート | O(n log n) | O(n^2) | O(log n) | 不安定 | 実測最速 |
| Tim ソート | O(n log n) | O(n log n) | O(n) | 安定 | Python/Java 標準 |
| イントロソート | O(n log n) | O(n log n) | O(log n) | 不安定 | C++ 標準 |

### 表3: 優先度キューの言語別実装

| 言語 | クラス/モジュール | 内部実装 | 備考 |
|------|----------------|---------|------|
| Python | heapq | 二分ヒープ（配列） | 最小ヒープのみ |
| Java | PriorityQueue | 二分ヒープ | 最小ヒープ、Comparator で変更可能 |
| C++ | priority_queue | 二分ヒープ | 最大ヒープがデフォルト |
| Go | container/heap | インタフェース | Push/Pop/Len/Less/Swap を実装 |
| Rust | BinaryHeap | 二分ヒープ | 最大ヒープ、Reverse で最小 |
| C# | PriorityQueue<T, P> | .NET 6+ | 最小ヒープ |

---

## 9. アンチパターン

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

# パフォーマンス差:
# n = 1,000,000 の場合
# BAD:  ~1.2 秒
# GOOD: ~0.05 秒（約 24 倍高速）
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

# GOOD: ヒープ + セットの併用
class HeapWithSet:
    def __init__(self):
        self.heap = []
        self.members = set()

    def push(self, val):
        if val not in self.members:
            heapq.heappush(self.heap, val)
            self.members.add(val)

    def pop(self):
        val = heapq.heappop(self.heap)
        self.members.discard(val)
        return val

    def contains(self, val):
        return val in self.members  # O(1)
```

### アンチパターン3: nsmallest/nlargest の不適切な使用

```python
import heapq

# BAD: 全要素が必要なのに nsmallest を使う
sorted_data = heapq.nsmallest(len(data), data)  # O(n log n)

# GOOD: sorted() を使う
sorted_data = sorted(data)  # O(n log n) だが定数係数が小さい

# BAD: 最小/最大の1つだけが必要なのに nsmallest を使う
minimum = heapq.nsmallest(1, data)[0]  # O(n)

# GOOD: min/max を使う
minimum = min(data)  # O(n) だがオーバーヘッドが少ない

# GOOD: k が小さい場合は nsmallest/nlargest
top10 = heapq.nsmallest(10, data)  # O(n + 10 log n) ≈ O(n)
```

### アンチパターン4: ヒープのソート済み保証を誤解

```python
import heapq

# BAD: ヒープ配列がソート済みだと思い込む
heap = [1, 3, 2, 5, 8, 7]
heapq.heapify(heap)
print(heap)  # [1, 3, 2, 5, 8, 7] — ソートされていない!

# ヒープ順序性: 親 ≤ 子 だが、兄弟間の順序は保証されない
# heap[0] は最小値だが、heap[1] は2番目に小さい値とは限らない

# GOOD: ソート順が必要なら pop を繰り返す
sorted_result = []
temp = list(heap)
heapq.heapify(temp)
while temp:
    sorted_result.append(heapq.heappop(temp))
print(sorted_result)  # [1, 2, 3, 5, 7, 8]
```

### アンチパターン5: 優先度キューの優先度更新を再挿入で代用

```python
import heapq

# BAD: 優先度の更新を新しいエントリの追加だけで対応
# → 古いエントリがヒープに残り続けメモリリーク
heap = []
heapq.heappush(heap, (5, "task_A"))
heapq.heappush(heap, (3, "task_A"))  # 優先度更新のつもり
# → (5, "task_A") がヒープに残る

# GOOD: 遅延削除パターンを使う
# → LazyDeletionPQ（上記セクション参照）
#
# BETTER: インデックス付きヒープで decrease_key を使う
# → IndexedMinHeap（上記セクション参照）
```

---

## 10. FAQ

### Q1: heapify がなぜ O(n) で済むのか？

**A:** ボトムアップに sift-down する。葉ノード（約 n/2 個）は sift-down 不要。高さ h のノード数は n/2^(h+1) で、sift-down コストは O(h)。合計 Sigma(h * n/2^(h+1)) = O(n)。直感的には、ほとんどのノードが低い高さにあり、sift-down 距離が短いため。

### Q2: Python の heapq に最大ヒープはないのか？

**A:** 標準では最小ヒープのみ。最大ヒープを実現する方法:
1. **符号反転**: `heappush(h, -val)` で挿入、`-heappop(h)` で取得。最も一般的
2. **タプルの反転**: `heappush(h, (-priority, item))` でカスタムオブジェクトに対応
3. **サードパーティ**: `heapdict` パッケージや自作クラス
4. **ラッパークラス**: `__lt__` を反転させたクラスでラップ

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass(order=True)
class MaxHeapItem:
    priority: int = field(compare=True)
    item: Any = field(compare=False)

    def __post_init__(self):
        self.priority = -self.priority  # 符号反転
```

### Q3: ヒープと BST の使い分けは？

**A:**
- **ヒープ**: 最小/最大値だけが必要 → O(1) 参照、O(log n) 挿入/削除
- **BST**: 範囲検索、順序走査、k番目の要素が必要 → O(log n) 各種操作
- **ヒープの利点**: 配列で実装でき、メモリ効率が良い。キャッシュフレンドリー
- **BST の利点**: 任意の要素の探索/削除が O(log n)。ヒープは O(n)

### Q4: Top-K 問題の最適解は？

**A:** K の大きさによって変わる:
- **k = 1**: `min()` / `max()` で O(n)
- **k が小さい**: サイズ k のヒープで O(n log k)
- **k ≈ n/2**: Quick Select で O(n) 平均
- **k ≈ n**: `sorted()` で O(n log n)

### Q5: ヒープソートはなぜ実務であまり使われないか？

**A:** ヒープソートは O(n log n) 最悪保証、O(1) 追加メモリという優れた性質を持つが、実測ではクイックソートや TimSort に劣る。理由:
1. **キャッシュ効率が悪い**: ヒープの参照パターンが非局所的（親子関係がメモリ上で遠い）
2. **分岐予測が困難**: sift-down での比較パターンが予測しにくい
3. **不安定**: 同一キーの相対順序が保持されない
ただしメモリ制約が厳しい場合や、最悪ケース保証が必要な場合は有効。C++ の `std::sort` は内部でイントロソート（クイック+ヒープのハイブリッド）を使用。

### Q6: ヒープを使った効率的な外部ソートの方法は？

**A:** K-way マージ:
1. 大きなファイルを RAM に収まるチャンクに分割
2. 各チャンクをメモリ内でソートしてファイルに書き出す
3. サイズ K の最小ヒープで K 個のチャンクの先頭要素を管理
4. ヒープから最小値を取り出し、出力ファイルに書き込む
5. 取り出したチャンクの次の要素をヒープに追加

Python の `heapq.merge()` がまさにこの操作を提供する。

---

## 11. まとめ

| 項目 | ポイント |
|------|---------|
| 二分ヒープ | 完全二分木。配列で効率的に表現。ポインタ不要 |
| 最小/最大ヒープ | 根が最小/最大値。O(1) で参照 |
| sift-up/down | 挿入/削除後のヒープ性質の復元 — O(log n) |
| heapify | ボトムアップ構築で O(n)。逐次 push の O(n log n) より高速 |
| ヒープソート | O(n log n)、in-place、不安定。最悪保証あり |
| 優先度キュー | ヒープで実装。遅延削除やインデックス付きヒープも重要 |
| d-ary ヒープ | decrease_key が多い場合に d を大きくすると有利 |
| ペアリングヒープ | マージが O(1)。フィボナッチヒープの簡易版 |
| 実務応用 | Top-K、中央値、Dijkstra、ハフマン、タスクスケジューラ |

---

## 次に読むべきガイド

- [グラフ — 表現方法と重み付きグラフ](./06-graphs.md)
- [最短経路 — Dijkstra とヒープの活用](../02-algorithms/03-shortest-path.md)

---

## 参考文献

1. Cormen, T.H. et al. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. — 第6章「Heapsort」、第19章「Fibonacci Heaps」
2. Williams, J.W.J. (1964). "Algorithm 232: Heapsort." *Communications of the ACM*, 7(6), 347-348.
3. Fredman, M.L. & Tarjan, R.E. (1987). "Fibonacci heaps and their uses in improved network optimization algorithms." *Journal of the ACM*, 34(3), 596-615.
4. Python Documentation. "heapq --- Heap queue algorithm." --- https://docs.python.org/3/library/heapq.html
5. Fredman, M.L. et al. (1986). "The pairing heap: A new form of self-adjusting heap." *Algorithmica*, 1(1), 111-129.
