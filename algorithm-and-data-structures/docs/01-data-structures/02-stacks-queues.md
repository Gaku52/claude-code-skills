# スタックとキュー — 実装・応用・優先度キュー

> LIFO / FIFO の原理に基づくスタックとキューを深く理解し、括弧マッチ、BFS、優先度キューなどの応用を学ぶ。

---

## この章で学ぶこと

1. **スタック（LIFO）とキュー（FIFO）** の実装と基本操作
2. **応用問題** — 括弧マッチ、逆ポーランド記法、BFS
3. **優先度キュー** の概念とヒープとの関係

---

## 1. スタックとキューの概念

```
スタック (LIFO):            キュー (FIFO):
  ┌─────┐                    front          rear
  │  C  │ ← top (push/pop)   │               │
  ├─────┤                     ▼               ▼
  │  B  │                   ┌───┬───┬───┬───┐
  ├─────┤                   │ A │ B │ C │ D │
  │  A  │                   └───┴───┴───┴───┘
  └─────┘                    dequeue →  ← enqueue

  push(D):                  enqueue(E):
  ┌─────┐                   ┌───┬───┬───┬───┬───┐
  │  D  │ ← new top         │ A │ B │ C │ D │ E │
  ├─────┤                   └───┴───┴───┴───┴───┘
  │  C  │
  ├─────┤
  │  B  │
  ├─────┤
  │  A  │
  └─────┘
```

---

## 2. 実装

### 2.1 スタック（配列ベース）

```python
class Stack:
    def __init__(self):
        self._data = []

    def push(self, val):
        """O(1) 償却"""
        self._data.append(val)

    def pop(self):
        """O(1)"""
        if self.is_empty():
            raise IndexError("stack is empty")
        return self._data.pop()

    def peek(self):
        """O(1)"""
        if self.is_empty():
            raise IndexError("stack is empty")
        return self._data[-1]

    def is_empty(self):
        return len(self._data) == 0

    def __len__(self):
        return len(self._data)
```

### 2.2 キュー（collections.deque）

```python
from collections import deque

class Queue:
    def __init__(self):
        self._data = deque()

    def enqueue(self, val):
        """O(1)"""
        self._data.append(val)

    def dequeue(self):
        """O(1)"""
        if self.is_empty():
            raise IndexError("queue is empty")
        return self._data.popleft()

    def front(self):
        """O(1)"""
        if self.is_empty():
            raise IndexError("queue is empty")
        return self._data[0]

    def is_empty(self):
        return len(self._data) == 0
```

### 2.3 2つのスタックでキューを実装

```python
class QueueWithStacks:
    """償却 O(1) の enqueue/dequeue"""
    def __init__(self):
        self.in_stack = []
        self.out_stack = []

    def enqueue(self, val):
        self.in_stack.append(val)

    def dequeue(self):
        if not self.out_stack:
            while self.in_stack:
                self.out_stack.append(self.in_stack.pop())
        return self.out_stack.pop()
```

```
動作例:
  enqueue(1), enqueue(2), enqueue(3):
    in_stack:  [1, 2, 3]
    out_stack: []

  dequeue():
    移動 → in_stack:  []
           out_stack: [3, 2, 1]
    pop → 1 を返す
    out_stack: [3, 2]
```

---

## 3. 応用

### 例1: 括弧マッチ

```python
def is_valid_parentheses(s):
    """括弧の対応チェック — O(n)"""
    stack = []
    pairs = {')': '(', '}': '{', ']': '['}
    for c in s:
        if c in '({[':
            stack.append(c)
        elif c in ')}]':
            if not stack or stack[-1] != pairs[c]:
                return False
            stack.pop()
    return len(stack) == 0

# is_valid_parentheses("({[]})") → True
# is_valid_parentheses("({[})") → False
```

### 例2: 逆ポーランド記法の評価

```python
def eval_rpn(tokens):
    """逆ポーランド記法を評価 — O(n)"""
    stack = []
    ops = {
        '+': lambda a, b: a + b,
        '-': lambda a, b: a - b,
        '*': lambda a, b: a * b,
        '/': lambda a, b: int(a / b),
    }
    for token in tokens:
        if token in ops:
            b, a = stack.pop(), stack.pop()
            stack.append(ops[token](a, b))
        else:
            stack.append(int(token))
    return stack[0]

# eval_rpn(["2", "3", "+", "4", "*"]) → 20
```

### 例3: 単調スタック（次に大きい要素）

```python
def next_greater_element(nums):
    """各要素の右側で最初に大きい要素 — O(n)"""
    result = [-1] * len(nums)
    stack = []  # インデックスを格納
    for i, num in enumerate(nums):
        while stack and nums[stack[-1]] < num:
            idx = stack.pop()
            result[idx] = num
        stack.append(i)
    return result

# next_greater_element([2, 1, 4, 3]) → [4, 4, -1, -1]
```

---

## 4. 優先度キュー

```
通常のキュー:      先着順（FIFO）
優先度キュー:      優先度が高い要素が先に出る

  enqueue(3), enqueue(1), enqueue(4), enqueue(2)

  通常キュー:    dequeue → 3, 1, 4, 2
  優先度キュー:  dequeue → 1, 2, 3, 4 (最小優先の場合)
```

```python
import heapq

class PriorityQueue:
    """最小ヒープベースの優先度キュー"""
    def __init__(self):
        self._heap = []

    def push(self, priority, item):
        """O(log n)"""
        heapq.heappush(self._heap, (priority, item))

    def pop(self):
        """O(log n)"""
        return heapq.heappop(self._heap)

    def peek(self):
        """O(1)"""
        return self._heap[0]

    def is_empty(self):
        return len(self._heap) == 0
```

---

## 5. 比較表

### 表1: スタック・キュー・優先度キューの操作計算量

| 操作 | スタック | キュー (deque) | 優先度キュー (ヒープ) |
|------|---------|---------------|---------------------|
| 挿入 | O(1) | O(1) | O(log n) |
| 取り出し | O(1) | O(1) | O(log n) |
| 先頭/トップ参照 | O(1) | O(1) | O(1) |
| 探索 | O(n) | O(n) | O(n) |
| 順序 | LIFO | FIFO | 優先度順 |

### 表2: キューの実装方法比較

| 実装 | enqueue | dequeue | 空間 | 備考 |
|------|---------|---------|------|------|
| 配列 (list) | O(1) 償却 | O(n) | O(n) | dequeue が遅い |
| deque | O(1) | O(1) | O(n) | 推奨 |
| 連結リスト | O(1) | O(1) | O(n) + ポインタ | メモリオーバーヘッド |
| 2つのスタック | O(1) | O(1) 償却 | O(n) | 面接頻出 |
| 循環バッファ | O(1) | O(1) | O(n) 固定 | 組み込み向け |

---

## 6. アンチパターン

### アンチパターン1: list を dequeue に使う

```python
# BAD: list.pop(0) は O(n)
queue = [1, 2, 3, 4, 5]
while queue:
    item = queue.pop(0)  # 毎回 O(n) のシフト

# GOOD: deque.popleft() は O(1)
from collections import deque
queue = deque([1, 2, 3, 4, 5])
while queue:
    item = queue.popleft()
```

### アンチパターン2: スタックの中身を走査するために pop し続ける

```python
# BAD: 探索のために破壊的に pop
def find_in_stack_bad(stack, target):
    temp = []
    found = False
    while stack:
        val = stack.pop()
        temp.append(val)
        if val == target:
            found = True
            break
    while temp:
        stack.append(temp.pop())  # 復元
    return found

# GOOD: 内部リストを直接走査
def find_in_stack_good(stack, target):
    return target in stack._data
```

---

## 7. FAQ

### Q1: スタックとキューは再帰とどう関係するか？

**A:** 再帰は暗黙のスタック（コールスタック）を使う。全ての再帰は明示的なスタックで反復に変換可能。DFS は再帰（スタック）、BFS はキューを使う。

### Q2: deque はスレッドセーフか？

**A:** Python の `collections.deque` は GIL の下で `append` と `popleft` がアトミック。マルチスレッド環境では `queue.Queue`（内部で Lock を使用）の方が安全。

### Q3: 優先度キューを配列で実装しない理由は？

**A:** ソート済み配列の挿入は O(n)、未ソート配列の最小値取り出しは O(n)。ヒープなら両方 O(log n) で済む。大量の挿入・取り出しが混在する場合はヒープが圧倒的に有利。

---

## 8. まとめ

| 項目 | ポイント |
|------|---------|
| スタック | LIFO。DFS、括弧マッチ、逆ポーランド記法 |
| キュー | FIFO。BFS、タスクスケジューリング |
| deque | 両端 O(1)。Python ではキューの推奨実装 |
| 単調スタック | 次の大きい/小さい要素の問題に O(n) |
| 優先度キュー | ヒープで実装。Dijkstra、スケジューリング |

---

## 次に読むべきガイド

- [ヒープ — 二分ヒープと優先度キュー実装](./05-heaps.md)
- [グラフ走査 — BFS/DFS](../02-algorithms/02-graph-traversal.md)

---

## 参考文献

1. Cormen, T.H. et al. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. — 第10章
2. Sedgewick, R. & Wayne, K. (2011). *Algorithms* (4th ed.). Addison-Wesley. — スタックとキューの実装
3. Python Documentation. "collections.deque." — https://docs.python.org/3/library/collections.html#collections.deque
