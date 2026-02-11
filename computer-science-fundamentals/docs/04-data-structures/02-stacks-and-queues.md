# スタックとキュー

> スタック（LIFO）は関数呼び出しと括弧の対応を、キュー（FIFO）はタスクスケジューリングとBFSを支える。

## この章で学ぶこと

- [ ] スタック（LIFO）とキュー（FIFO）の違いを説明できる
- [ ] 各データ構造の実務的な用途を理解する
- [ ] 優先度キュー（ヒープ）の仕組みを理解する

## 前提知識

- 配列 → 参照: [[00-arrays-and-strings.md]]

---

## 1. スタック（Stack）— LIFO

### 1.1 基本操作

```python
# スタック: Last In, First Out（後入れ先出し）
# Python では list をスタックとして使用

stack = []
stack.append(1)    # push: O(1)
stack.append(2)
stack.append(3)
top = stack[-1]    # peek: O(1) → 3
val = stack.pop()  # pop: O(1)  → 3
len(stack)         # size: O(1) → 2

# 用途:
# - 関数呼び出しスタック（コールスタック）
# - Undo/Redo
# - 括弧の対応チェック
# - DFS（深さ優先探索）
# - 式の評価（後置記法）
# - ブラウザの「戻る」ボタン
```

### 1.2 典型問題: 括弧の対応

```python
def is_valid_parentheses(s):
    """括弧が正しく対応しているか"""
    stack = []
    pairs = {')': '(', ']': '[', '}': '{'}
    for char in s:
        if char in '([{':
            stack.append(char)
        elif char in ')]}':
            if not stack or stack[-1] != pairs[char]:
                return False
            stack.pop()
    return len(stack) == 0

# "([{}])" → True
# "([)]"   → False
```

### 1.3 単調スタック

```python
def next_greater_element(arr):
    """各要素の右側にある最初の大きい要素を求める"""
    n = len(arr)
    result = [-1] * n
    stack = []  # インデックスを格納

    for i in range(n):
        while stack and arr[stack[-1]] < arr[i]:
            idx = stack.pop()
            result[idx] = arr[i]
        stack.append(i)

    return result

# arr = [4, 2, 3, 5, 1]
# result = [5, 3, 5, -1, -1]
# 計算量: O(n) — 各要素は最大1回pushと1回pop
```

---

## 2. キュー（Queue）— FIFO

### 2.1 基本操作

```python
from collections import deque

# キュー: First In, First Out（先入れ先出し）
queue = deque()
queue.append(1)      # enqueue: O(1)
queue.append(2)
queue.append(3)
front = queue[0]     # peek: O(1) → 1
val = queue.popleft() # dequeue: O(1) → 1

# 注意: list.pop(0) は O(n)! deque を使うこと

# 用途:
# - BFS（幅優先探索）
# - タスクキュー（ジョブスケジューラ）
# - メッセージキュー（RabbitMQ, SQS）
# - プリンタキュー
# - 順番待ちシステム
```

### 2.2 デキュー（Deque）— 両端キュー

```python
from collections import deque

dq = deque()
dq.append(1)      # 右に追加: O(1)
dq.appendleft(0)  # 左に追加: O(1)
dq.pop()           # 右から取出: O(1)
dq.popleft()       # 左から取出: O(1)

# スライディングウィンドウの最大値（単調デキュー）
def max_sliding_window(arr, k):
    dq = deque()  # インデックスを格納（降順に値を管理）
    result = []
    for i in range(len(arr)):
        while dq and dq[0] < i - k + 1:
            dq.popleft()  # ウィンドウ外を除去
        while dq and arr[dq[-1]] < arr[i]:
            dq.pop()  # 現在の値より小さい要素を除去
        dq.append(i)
        if i >= k - 1:
            result.append(arr[dq[0]])
    return result
# 計算量: O(n)
```

---

## 3. 優先度キュー（Priority Queue）

### 3.1 ヒープ

```python
import heapq

# 最小ヒープ（Python の heapq）
heap = []
heapq.heappush(heap, 3)  # O(log n)
heapq.heappush(heap, 1)
heapq.heappush(heap, 2)
min_val = heapq.heappop(heap)  # O(log n) → 1
peek = heap[0]  # O(1) → 2

# 最大ヒープ（値を反転）
max_heap = []
heapq.heappush(max_heap, -3)
heapq.heappush(max_heap, -1)
max_val = -heapq.heappop(max_heap)  # → 3

# ヒープの内部構造:
#       1           配列表現: [1, 2, 3, 5, 4]
#      / \          親: i → 子: 2i+1, 2i+2
#     2   3         子: i → 親: (i-1)//2
#    / \
#   5   4

# 用途:
# - ダイクストラ法の優先度キュー
# - スケジューラ（優先度付きタスク処理）
# - ストリームの中央値
# - Top-K問題
# - イベント駆動シミュレーション
```

### 3.2 Top-K問題

```python
import heapq

def top_k_frequent(nums, k):
    """出現頻度Top-Kの要素"""
    from collections import Counter
    count = Counter(nums)
    return heapq.nlargest(k, count.keys(), key=count.get)

# heapq.nlargest: O(n log k) — 全ソート O(n log n) より効率的
```

---

## 4. 実践演習

### 演習1: スタック活用（基礎）
逆ポーランド記法（後置記法）の式を評価する関数を実装せよ。

### 演習2: キュー活用（応用）
スタック2つだけを使ってキューを実装せよ。各操作が償却O(1)であることを確認せよ。

### 演習3: ヒープ活用（発展）
データストリームから中央値をリアルタイムで求めるクラスを実装せよ（2つのヒープを使用）。

---

## まとめ

| データ構造 | 原則 | 主要操作 | 用途 |
|-----------|------|---------|------|
| スタック | LIFO | push/pop O(1) | コールスタック、括弧チェック |
| キュー | FIFO | enqueue/dequeue O(1) | BFS、タスクキュー |
| デキュー | 両端 | 両端の追加/削除 O(1) | スライディングウィンドウ |
| 優先度キュー | 最小/最大優先 | push/pop O(log n) | ダイクストラ、Top-K |

---

## 次に読むべきガイド
→ [[03-hash-tables.md]] — ハッシュテーブル

---

## 参考文献
1. Cormen, T. H. "Introduction to Algorithms." Chapters 6, 10.
2. Sedgewick, R. "Algorithms." Chapter 2.4: Priority Queues.
