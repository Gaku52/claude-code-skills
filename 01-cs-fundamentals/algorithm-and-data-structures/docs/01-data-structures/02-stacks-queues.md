# スタックとキュー — 実装・応用・優先度キュー 完全ガイド

> LIFO / FIFO の原理に基づくスタックとキューを深く理解し、括弧マッチ、BFS、単調スタック、優先度キューなどの応用を体系的に学ぶ。

---

## 目次

1. [スタックとキューの基本概念](#1-スタックとキューの基本概念)
2. [スタックの実装と内部構造](#2-スタックの実装と内部構造)
3. [キューの実装と内部構造](#3-キューの実装と内部構造)
4. [スタックの応用アルゴリズム](#4-スタックの応用アルゴリズム)
5. [キューの応用アルゴリズム](#5-キューの応用アルゴリズム)
6. [優先度キューとヒープ](#6-優先度キューとヒープ)
7. [特殊なスタック・キュー構造](#7-特殊なスタックキュー構造)
8. [比較表と選定ガイド](#8-比較表と選定ガイド)
9. [アンチパターンとベストプラクティス](#9-アンチパターンとベストプラクティス)
10. [演習問題（3段階）](#10-演習問題3段階)
11. [FAQ](#11-faq)
12. [まとめ](#12-まとめ)
13. [参考文献](#13-参考文献)

---

## 1. スタックとキューの基本概念

### 1.1 抽象データ型としての定義

スタックとキューはコンピュータサイエンスにおける最も基本的な**抽象データ型 (ADT: Abstract Data Type)** である。
どちらも要素の集合を管理するが、要素を取り出す順序が根本的に異なる。

| 特性 | スタック | キュー |
|------|---------|-------|
| **原理** | LIFO (Last In, First Out) | FIFO (First In, First Out) |
| **日常の例** | 皿の積み重ね、Undo 操作 | レジの行列、印刷ジョブ |
| **主要操作** | push, pop, peek | enqueue, dequeue, front |
| **挿入位置** | トップ（上端） | リア（末尾） |
| **取り出し位置** | トップ（上端） | フロント（先頭） |

### 1.2 動作の視覚的理解

```
スタック (LIFO — Last In, First Out):

  初期状態:          push(A):          push(B):          push(C):
  ┌─────────┐      ┌─────────┐      ┌─────────┐      ┌─────────┐
  │  (empty) │      │         │      │         │      │    C    │ ← top
  │         │      │         │      │    B    │ ← top├─────────┤
  │         │      │    A    │ ← top├─────────┤      │    B    │
  └─────────┘      └─────────┘      │    A    │      ├─────────┤
                                     └─────────┘      │    A    │
                                                       └─────────┘

  pop() → C:        pop() → B:        pop() → A:
  ┌─────────┐      ┌─────────┐      ┌─────────┐
  │         │      │         │      │  (empty) │
  │    B    │ ← top│    A    │ ← top└─────────┘
  ├─────────┤      └─────────┘
  │    A    │
  └─────────┘


キュー (FIFO — First In, First Out):

  初期状態:
  front                                         rear
   │                                              │
   ▼                                              ▼
  ┌──── empty ────┐

  enqueue(A), enqueue(B), enqueue(C), enqueue(D):
  front                                         rear
   │                                              │
   ▼                                              ▼
  ┌─────┬─────┬─────┬─────┐
  │  A  │  B  │  C  │  D  │
  └─────┴─────┴─────┴─────┘

  dequeue() → A:
        front                                   rear
         │                                       │
         ▼                                       ▼
  ┌─────┬─────┬─────┐
  │  B  │  C  │  D  │
  └─────┴─────┴─────┘

  enqueue(E):
        front                                         rear
         │                                              │
         ▼                                              ▼
  ┌─────┬─────┬─────┬─────┐
  │  B  │  C  │  D  │  E  │
  └─────┴─────┴─────┴─────┘
```

### 1.3 コールスタックとの関係

プログラムの実行そのものがスタックに支えられている。関数呼び出しのたびにスタックフレームが積まれ、
return で取り除かれる。この仕組みを**コールスタック**と呼ぶ。

```
関数呼び出し: main() → funcA() → funcB() → funcC()

コールスタックの変遷:

  main()呼出   funcA()呼出   funcB()呼出   funcC()呼出   funcC() return
  ┌────────┐  ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐
  │ main() │  │ funcA()│   │ funcB()│   │ funcC()│   │ funcB()│
  └────────┘  ├────────┤   ├────────┤   ├────────┤   ├────────┤
              │ main() │   │ funcA()│   │ funcB()│   │ funcA()│
              └────────┘   ├────────┤   ├────────┤   ├────────┤
                           │ main() │   │ funcA()│   │ main() │
                           └────────┘   ├────────┤   └────────┘
                                        │ main() │
                                        └────────┘
```

再帰が深すぎると `RecursionError` (Python) や `StackOverflowError` (Java) が発生する。
これは OS が割り当てたスタック領域を超えたことを意味する。

### 1.4 スタックとキューの利用場面マップ

```
┌──────────────────────────────────────────────────────┐
│               利用場面マップ                          │
├──────────────────────────────────────────────────────┤
│                                                      │
│  スタック (LIFO)              キュー (FIFO)           │
│  ├─ DFS (深さ優先探索)        ├─ BFS (幅優先探索)     │
│  ├─ 括弧マッチ               ├─ タスクスケジューリング │
│  ├─ Undo/Redo               ├─ メッセージキュー      │
│  ├─ 逆ポーランド記法          ├─ プリンタキュー        │
│  ├─ ブラウザの戻る/進む       ├─ イベントループ        │
│  ├─ 関数コールスタック        ├─ キャッシュ (FIFO)     │
│  ├─ 構文解析 (パーサ)         ├─ ストリーム処理        │
│  └─ 単調スタック              └─ 公平なリソース割当    │
│                                                      │
│  優先度キュー                  デック (両端キュー)      │
│  ├─ Dijkstra最短経路          ├─ スライディングウィンドウ│
│  ├─ ハフマン符号化            ├─ ワークスティーリング   │
│  ├─ A* 探索                   ├─ 回文チェック          │
│  ├─ イベント駆動シミュレーション├─ ブラウザ履歴          │
│  └─ OS プロセススケジューリング └─ 両方向 BFS           │
│                                                      │
└──────────────────────────────────────────────────────┘
```

---

## 2. スタックの実装と内部構造

### 2.1 配列ベースのスタック（Python リスト）

Python の `list` は動的配列であり、末尾への追加・削除が O(1) 償却で行える。
スタックの実装に最適である。

```python
class ArrayStack:
    """
    配列ベースのスタック実装。
    Python の list を内部に使用し、全操作が O(1) 償却で動作する。

    使用例:
        >>> stack = ArrayStack()
        >>> stack.push(10)
        >>> stack.push(20)
        >>> stack.push(30)
        >>> len(stack)
        3
        >>> stack.peek()
        30
        >>> stack.pop()
        30
        >>> stack.pop()
        20
        >>> stack.is_empty()
        False
        >>> stack.pop()
        10
        >>> stack.is_empty()
        True
    """

    def __init__(self):
        self._data = []

    def push(self, val):
        """要素をトップに追加する。O(1) 償却。"""
        self._data.append(val)

    def pop(self):
        """トップの要素を取り出して返す。O(1)。
        空のスタックに対して呼ぶと IndexError を送出する。
        """
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._data.pop()

    def peek(self):
        """トップの要素を取り出さずに参照する。O(1)。"""
        if self.is_empty():
            raise IndexError("peek from empty stack")
        return self._data[-1]

    def is_empty(self):
        """スタックが空なら True を返す。O(1)。"""
        return len(self._data) == 0

    def __len__(self):
        """スタックの要素数を返す。O(1)。"""
        return len(self._data)

    def __repr__(self):
        return f"ArrayStack({self._data})"

    def __iter__(self):
        """トップから順にイテレートする（非破壊的）。"""
        return reversed(self._data)

    def clear(self):
        """全要素を削除する。O(1)。"""
        self._data.clear()


# === 動作確認 ===
if __name__ == "__main__":
    s = ArrayStack()
    for v in [10, 20, 30, 40, 50]:
        s.push(v)
    print(f"Stack: {s}")            # ArrayStack([10, 20, 30, 40, 50])
    print(f"Top: {s.peek()}")       # 50
    print(f"Pop: {s.pop()}")        # 50
    print(f"Pop: {s.pop()}")        # 40
    print(f"Size: {len(s)}")        # 3
    print(f"Contents (top first): {list(s)}")  # [30, 20, 10]
```

### 2.2 連結リストベースのスタック

連結リストを使えば、動的配列のリサイズに伴う一時的な O(n) コストを完全に回避できる。
ただしノードごとにポインタのメモリオーバーヘッドが生じる。

```python
class _Node:
    """単方向連結リストのノード。"""
    __slots__ = ('value', 'next')

    def __init__(self, value, next_node=None):
        self.value = value
        self.next = next_node


class LinkedStack:
    """
    連結リストベースのスタック実装。
    全操作が最悪 O(1) で動作する（償却ではなく真の O(1)）。

    使用例:
        >>> stack = LinkedStack()
        >>> stack.push(1)
        >>> stack.push(2)
        >>> stack.push(3)
        >>> stack.pop()
        3
        >>> stack.peek()
        2
        >>> len(stack)
        2
    """

    def __init__(self):
        self._top = None
        self._size = 0

    def push(self, val):
        """新しいノードをトップに追加する。O(1)。"""
        self._top = _Node(val, self._top)
        self._size += 1

    def pop(self):
        """トップのノードを取り除き、その値を返す。O(1)。"""
        if self.is_empty():
            raise IndexError("pop from empty stack")
        val = self._top.value
        self._top = self._top.next
        self._size -= 1
        return val

    def peek(self):
        """トップの値を参照する。O(1)。"""
        if self.is_empty():
            raise IndexError("peek from empty stack")
        return self._top.value

    def is_empty(self):
        return self._size == 0

    def __len__(self):
        return self._size

    def __iter__(self):
        """トップから順にイテレートする。"""
        node = self._top
        while node is not None:
            yield node.value
            node = node.next


# === 動作確認 ===
if __name__ == "__main__":
    s = LinkedStack()
    for v in ["alpha", "beta", "gamma"]:
        s.push(v)
    print(f"Top: {s.peek()}")        # gamma
    print(f"Pop: {s.pop()}")         # gamma
    print(f"Remaining: {list(s)}")   # ['beta', 'alpha']
```

### 2.3 固定容量スタック（組み込み向け）

組み込みシステムなど、メモリ量が厳密に制約される環境では、固定容量のスタックが使われる。

```python
class BoundedStack:
    """
    固定容量のスタック。容量超過時は OverflowError を送出する。
    組み込みシステムやリアルタイムシステムで使用される。

    使用例:
        >>> stack = BoundedStack(3)
        >>> stack.push(1)
        >>> stack.push(2)
        >>> stack.push(3)
        >>> stack.is_full()
        True
        >>> stack.push(4)  # OverflowError
    """

    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._data = [None] * capacity
        self._capacity = capacity
        self._top = -1

    def push(self, val):
        """O(1)。容量超過時は OverflowError。"""
        if self.is_full():
            raise OverflowError(
                f"stack is full (capacity={self._capacity})"
            )
        self._top += 1
        self._data[self._top] = val

    def pop(self):
        """O(1)。"""
        if self.is_empty():
            raise IndexError("pop from empty stack")
        val = self._data[self._top]
        self._data[self._top] = None  # 参照を解放
        self._top -= 1
        return val

    def peek(self):
        if self.is_empty():
            raise IndexError("peek from empty stack")
        return self._data[self._top]

    def is_empty(self):
        return self._top == -1

    def is_full(self):
        return self._top == self._capacity - 1

    def __len__(self):
        return self._top + 1


# === 動作確認 ===
if __name__ == "__main__":
    bs = BoundedStack(4)
    for v in [10, 20, 30, 40]:
        bs.push(v)
    print(f"Full? {bs.is_full()}")    # True
    print(f"Pop: {bs.pop()}")         # 40
    print(f"Full? {bs.is_full()}")    # False
```

### 2.4 配列 vs 連結リスト — スタック実装の比較

| 観点 | 配列ベース (list) | 連結リストベース |
|------|-------------------|----------------|
| push の計算量 | O(1) 償却 | O(1) 最悪 |
| pop の計算量 | O(1) | O(1) |
| メモリ効率 | 高い（連続領域） | 低い（ノード+ポインタ） |
| キャッシュ性能 | 良好（局所性が高い） | 低い（散在する可能性） |
| 容量制限 | 動的に拡張 | 動的に拡張 |
| リサイズコスト | たまに O(n) | なし |
| 実装の簡潔さ | 非常にシンプル | やや複雑 |
| **推奨場面** | **一般用途（推奨）** | **厳密なリアルタイム要件** |

---

## 3. キューの実装と内部構造

### 3.1 collections.deque によるキュー

Python の `collections.deque` は双方向連結リストで実装されており、
両端の操作が O(1) で行える。キューの実装には `deque` が最も推奨される。

```python
from collections import deque


class Queue:
    """
    deque ベースのキュー実装。
    enqueue / dequeue ともに O(1) で動作する。

    使用例:
        >>> q = Queue()
        >>> q.enqueue("task_1")
        >>> q.enqueue("task_2")
        >>> q.enqueue("task_3")
        >>> q.dequeue()
        'task_1'
        >>> q.front()
        'task_2'
        >>> len(q)
        2
    """

    def __init__(self):
        self._data = deque()

    def enqueue(self, val):
        """末尾に要素を追加する。O(1)。"""
        self._data.append(val)

    def dequeue(self):
        """先頭の要素を取り出して返す。O(1)。"""
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        return self._data.popleft()

    def front(self):
        """先頭の要素を取り出さずに参照する。O(1)。"""
        if self.is_empty():
            raise IndexError("front from empty queue")
        return self._data[0]

    def rear(self):
        """末尾の要素を取り出さずに参照する。O(1)。"""
        if self.is_empty():
            raise IndexError("rear from empty queue")
        return self._data[-1]

    def is_empty(self):
        return len(self._data) == 0

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f"Queue({list(self._data)})"

    def __iter__(self):
        """先頭から順にイテレートする（非破壊的）。"""
        return iter(self._data)


# === 動作確認 ===
if __name__ == "__main__":
    q = Queue()
    for task in ["print_report", "send_email", "backup_db", "notify_user"]:
        q.enqueue(task)
    print(f"Queue: {q}")
    print(f"Front: {q.front()}")     # print_report
    print(f"Rear: {q.rear()}")       # notify_user
    print(f"Dequeue: {q.dequeue()}")  # print_report
    print(f"Dequeue: {q.dequeue()}")  # send_email
    print(f"Remaining: {list(q)}")    # ['backup_db', 'notify_user']
```

### 3.2 循環バッファ（リングバッファ）によるキュー

固定サイズの配列上でキューを実現する手法。メモリ割り当てが一度だけで済むため、
組み込みシステムやカーネルのバッファリングで広く使われる。

```python
class CircularQueue:
    """
    固定容量の循環バッファキュー。
    配列のインデックスを mod 演算で循環させる。

    使用例:
        >>> cq = CircularQueue(5)
        >>> cq.enqueue(10)
        >>> cq.enqueue(20)
        >>> cq.enqueue(30)
        >>> cq.dequeue()
        10
        >>> cq.enqueue(40)
        >>> cq.enqueue(50)
        >>> cq.enqueue(60)
        >>> cq.is_full()
        True
    """

    def __init__(self, capacity):
        self._data = [None] * capacity
        self._capacity = capacity
        self._front = 0
        self._size = 0

    def enqueue(self, val):
        """O(1)。容量超過時は OverflowError。"""
        if self.is_full():
            raise OverflowError("circular queue is full")
        rear = (self._front + self._size) % self._capacity
        self._data[rear] = val
        self._size += 1

    def dequeue(self):
        """O(1)。"""
        if self.is_empty():
            raise IndexError("dequeue from empty circular queue")
        val = self._data[self._front]
        self._data[self._front] = None
        self._front = (self._front + 1) % self._capacity
        self._size -= 1
        return val

    def front(self):
        if self.is_empty():
            raise IndexError("front from empty circular queue")
        return self._data[self._front]

    def is_empty(self):
        return self._size == 0

    def is_full(self):
        return self._size == self._capacity

    def __len__(self):
        return self._size

    def __repr__(self):
        items = []
        for i in range(self._size):
            idx = (self._front + i) % self._capacity
            items.append(self._data[idx])
        return f"CircularQueue({items})"


# === 動作確認 ===
if __name__ == "__main__":
    cq = CircularQueue(4)
    cq.enqueue("A")
    cq.enqueue("B")
    cq.enqueue("C")
    print(f"Dequeue: {cq.dequeue()}")   # A
    print(f"Dequeue: {cq.dequeue()}")   # B
    cq.enqueue("D")
    cq.enqueue("E")
    cq.enqueue("F")
    print(f"Queue: {cq}")  # CircularQueue(['C', 'D', 'E', 'F'])
    print(f"Full? {cq.is_full()}")      # True
```

```
循環バッファの内部動作:

  容量 = 4

  初期状態:          enqueue A,B,C:       dequeue() → A:
  ┌───┬───┬───┬───┐ ┌───┬───┬───┬───┐   ┌───┬───┬───┬───┐
  │   │   │   │   │ │ A │ B │ C │   │   │   │ B │ C │   │
  └───┴───┴───┴───┘ └───┴───┴───┴───┘   └───┴───┴───┴───┘
   ↑f               ↑f          ↑r          ↑f      ↑r
                                  (rear)     (front)

  enqueue D:          enqueue E:          結果:
  ┌───┬───┬───┬───┐  ┌───┬───┬───┬───┐  front=1, size=4
  │   │ B │ C │ D │  │ E │ B │ C │ D │  配列は循環して
  └───┴───┴───┴───┘  └───┴───┴───┴───┘  再利用される
      ↑f          ↑r  ↑r  ↑f
```

### 3.3 2つのスタックによるキューの実装

面接で頻出のテクニック。2つのスタック（`in_stack` と `out_stack`）を使って FIFO を実現する。
償却計算量は O(1) となる。

```python
class QueueWithTwoStacks:
    """
    2つのスタックを使ったキュー実装。
    - in_stack: enqueue 用
    - out_stack: dequeue 用
    - out_stack が空のときだけ in_stack から移し替える

    償却分析:
      各要素は in_stack に1回 push、in_stack から1回 pop、
      out_stack に1回 push、out_stack から1回 pop される。
      合計4回の操作 → 1要素あたり O(1) 償却。

    使用例:
        >>> q = QueueWithTwoStacks()
        >>> q.enqueue(1)
        >>> q.enqueue(2)
        >>> q.enqueue(3)
        >>> q.dequeue()
        1
        >>> q.dequeue()
        2
        >>> q.enqueue(4)
        >>> q.dequeue()
        3
        >>> q.dequeue()
        4
    """

    def __init__(self):
        self._in_stack = []
        self._out_stack = []

    def enqueue(self, val):
        """O(1)。"""
        self._in_stack.append(val)

    def dequeue(self):
        """O(1) 償却。"""
        if not self._out_stack:
            if not self._in_stack:
                raise IndexError("dequeue from empty queue")
            while self._in_stack:
                self._out_stack.append(self._in_stack.pop())
        return self._out_stack.pop()

    def front(self):
        """O(1) 償却。"""
        if not self._out_stack:
            if not self._in_stack:
                raise IndexError("front from empty queue")
            while self._in_stack:
                self._out_stack.append(self._in_stack.pop())
        return self._out_stack[-1]

    def is_empty(self):
        return not self._in_stack and not self._out_stack

    def __len__(self):
        return len(self._in_stack) + len(self._out_stack)


# === 動作確認 ===
if __name__ == "__main__":
    q = QueueWithTwoStacks()
    # 交互に enqueue / dequeue しても正しく FIFO 順になる
    q.enqueue("A")
    q.enqueue("B")
    print(q.dequeue())  # A
    q.enqueue("C")
    q.enqueue("D")
    print(q.dequeue())  # B
    print(q.dequeue())  # C
    print(q.dequeue())  # D
```

```
2つのスタックによるキューの動作:

  enqueue(1), enqueue(2), enqueue(3):

    in_stack:          out_stack:
    ┌───┐
    │ 3 │ ← top       (empty)
    ├───┤
    │ 2 │
    ├───┤
    │ 1 │
    └───┘

  dequeue() 呼び出し → out_stack が空なので移し替え:

    in_stack:          out_stack:
                       ┌───┐
    (empty)            │ 1 │ ← top  ← 次の dequeue で取得
                       ├───┤
                       │ 2 │
                       ├───┤
                       │ 3 │
                       └───┘

  dequeue() → 1 を返す:

    in_stack:          out_stack:
                       ┌───┐
    (empty)            │ 2 │ ← top
                       ├───┤
                       │ 3 │
                       └───┘
```

---

## 4. スタックの応用アルゴリズム

### 4.1 括弧マッチ（バランスチェック）

コンパイラの構文解析やコードエディタのハイライト機能で使われる基本アルゴリズム。
開き括弧をスタックに積み、閉じ括弧が来たらペアを確認する。

```python
def is_valid_parentheses(s: str) -> bool:
    """
    括弧の対応が正しいかチェックする。
    対応する括弧: (), {}, []

    計算量: O(n) 時間、O(n) 空間

    使用例:
        >>> is_valid_parentheses("({[]})")
        True
        >>> is_valid_parentheses("({[})")
        False
        >>> is_valid_parentheses("")
        True
        >>> is_valid_parentheses("((")
        False
        >>> is_valid_parentheses("[{()}]")
        True
    """
    stack = []
    pairs = {')': '(', '}': '{', ']': '['}

    for ch in s:
        if ch in '({[':
            stack.append(ch)
        elif ch in ')}]':
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()

    return len(stack) == 0


# === テスト ===
if __name__ == "__main__":
    test_cases = [
        ("({[]})", True),
        ("({[})", False),
        ("", True),
        ("((", False),
        ("))", False),
        ("[{()}]", True),
        ("a(b{c[d]e}f)g", True),  # 括弧以外の文字を含む
    ]
    for expr, expected in test_cases:
        result = is_valid_parentheses(expr)
        status = "PASS" if result == expected else "FAIL"
        print(f"[{status}] is_valid_parentheses('{expr}') = {result}")
```

### 4.2 逆ポーランド記法（RPN: Reverse Polish Notation）の評価

逆ポーランド記法は演算子が被演算子の後に来る記法で、括弧が不要である。
HP の電卓や PostScript 言語で採用されている。

```python
import operator
from typing import List


def eval_rpn(tokens: List[str]) -> float:
    """
    逆ポーランド記法（後置記法）の式を評価する。

    アルゴリズム:
      1. トークンを左から順に読む
      2. 数値ならスタックに push
      3. 演算子なら2つ pop して計算し、結果を push
      4. 最後にスタックに残った値が答え

    計算量: O(n) 時間、O(n) 空間

    使用例:
        >>> eval_rpn(["2", "3", "+", "4", "*"])
        20.0
        >>> eval_rpn(["5", "1", "2", "+", "4", "*", "+", "3", "-"])
        14.0
        >>> eval_rpn(["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"])
        22.0
    """
    ops = {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '/': lambda a, b: int(a / b) if isinstance(a, int) else a / b,
    }

    stack = []
    for token in tokens:
        if token in ops:
            b = stack.pop()
            a = stack.pop()
            stack.append(ops[token](a, b))
        else:
            stack.append(int(token))

    return float(stack[0])


def infix_to_rpn(expression: str) -> List[str]:
    """
    中置記法を逆ポーランド記法に変換する（シャンティングヤード法）。
    演算子の優先順位と結合性を正しく処理する。

    使用例:
        >>> infix_to_rpn("3 + 4 * 2")
        ['3', '4', '2', '*', '+']
        >>> infix_to_rpn("( 1 + 2 ) * 3")
        ['1', '2', '+', '3', '*']
    """
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
    output = []
    op_stack = []

    tokens = expression.split()
    for token in tokens:
        if token.lstrip('-').isdigit():
            output.append(token)
        elif token == '(':
            op_stack.append(token)
        elif token == ')':
            while op_stack and op_stack[-1] != '(':
                output.append(op_stack.pop())
            op_stack.pop()  # '(' を捨てる
        elif token in precedence:
            while (op_stack and
                   op_stack[-1] != '(' and
                   op_stack[-1] in precedence and
                   precedence[op_stack[-1]] >= precedence[token]):
                output.append(op_stack.pop())
            op_stack.append(token)

    while op_stack:
        output.append(op_stack.pop())

    return output


# === 動作確認 ===
if __name__ == "__main__":
    # 中置記法 → 逆ポーランド記法 → 評価
    infix = "( 2 + 3 ) * 4"
    rpn = infix_to_rpn(infix)
    result = eval_rpn(rpn)
    print(f"Infix:  {infix}")
    print(f"RPN:    {' '.join(rpn)}")
    print(f"Result: {result}")  # 20.0

    # 直接 RPN を評価
    print(f"\neval_rpn(['5','1','2','+','4','*','+','3','-']) = "
          f"{eval_rpn(['5','1','2','+','4','*','+','3','-'])}")  # 14.0
```

### 4.3 単調スタック（Monotonic Stack）

単調スタックは「次に大きい要素」「次に小さい要素」を O(n) で求める強力なテクニックである。
スタック内の要素が単調増加（または単調減少）になるよう維持する。

```python
from typing import List


def next_greater_element(nums: List[int]) -> List[int]:
    """
    各要素の右側で最初に見つかる、自分より大きい要素を返す。
    存在しなければ -1。

    アルゴリズム:
      スタックにインデックスを格納し、単調減少を維持する。
      新しい要素がスタックトップより大きければ、
      スタックトップの「次に大きい要素」が見つかったことになる。

    計算量: O(n) 時間、O(n) 空間
    各要素は最大1回 push、1回 pop されるため線形。

    使用例:
        >>> next_greater_element([2, 1, 4, 3])
        [4, 4, -1, -1]
        >>> next_greater_element([1, 3, 2, 4])
        [3, 4, 4, -1]
        >>> next_greater_element([5, 4, 3, 2, 1])
        [-1, -1, -1, -1, -1]
    """
    n = len(nums)
    result = [-1] * n
    stack = []  # インデックスを格納

    for i in range(n):
        while stack and nums[stack[-1]] < nums[i]:
            idx = stack.pop()
            result[idx] = nums[i]
        stack.append(i)

    return result


def daily_temperatures(temperatures: List[int]) -> List[int]:
    """
    各日から、それより暖かい日が何日後に来るかを返す。
    LeetCode 739 の典型問題。

    使用例:
        >>> daily_temperatures([73, 74, 75, 71, 69, 72, 76, 73])
        [1, 1, 4, 2, 1, 1, 0, 0]
    """
    n = len(temperatures)
    answer = [0] * n
    stack = []

    for i in range(n):
        while stack and temperatures[stack[-1]] < temperatures[i]:
            prev_idx = stack.pop()
            answer[prev_idx] = i - prev_idx
        stack.append(i)

    return answer


def largest_rectangle_in_histogram(heights: List[int]) -> int:
    """
    ヒストグラムの最大長方形面積を求める。
    LeetCode 84 の典型問題。単調スタックの代表的応用。

    使用例:
        >>> largest_rectangle_in_histogram([2, 1, 5, 6, 2, 3])
        10
        >>> largest_rectangle_in_histogram([2, 4])
        4
    """
    stack = []
    max_area = 0
    heights = heights + [0]  # 番兵を追加

    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)

    return max_area


# === 動作確認 ===
if __name__ == "__main__":
    print("=== Next Greater Element ===")
    nums = [2, 7, 4, 3, 5]
    print(f"Input:  {nums}")
    print(f"Result: {next_greater_element(nums)}")
    # [7, -1, 5, 5, -1]

    print("\n=== Daily Temperatures ===")
    temps = [73, 74, 75, 71, 69, 72, 76, 73]
    print(f"Input:  {temps}")
    print(f"Result: {daily_temperatures(temps)}")
    # [1, 1, 4, 2, 1, 1, 0, 0]

    print("\n=== Largest Rectangle in Histogram ===")
    hist = [2, 1, 5, 6, 2, 3]
    print(f"Input:  {hist}")
    print(f"Result: {largest_rectangle_in_histogram(hist)}")
    # 10
```

```
単調スタックの動作例 — next_greater_element([2, 7, 4, 3, 5]):

  i=0, num=2:  stack=[]          → push 0        stack=[0]
  i=1, num=7:  stack=[0]         → 2<7 → pop 0, result[0]=7
                                  → push 1        stack=[1]
  i=2, num=4:  stack=[1]         → 7>4 → push 2   stack=[1,2]
  i=3, num=3:  stack=[1,2]       → 4>3 → push 3   stack=[1,2,3]
  i=4, num=5:  stack=[1,2,3]     → 3<5 → pop 3, result[3]=5
                                  → 4<5 → pop 2, result[2]=5
                                  → 7>5 → push 4   stack=[1,4]

  最終 result: [7, -1, 5, 5, -1]
  (スタックに残った 1,4 → 右に大きい要素なし → -1)
```

### 4.4 ブラウザの戻る/進む機能の実装

2つのスタックを使った実用的な例。

```python
class BrowserHistory:
    """
    ブラウザの戻る/進む機能を2つのスタックで実装する。

    使用例:
        >>> browser = BrowserHistory("google.com")
        >>> browser.visit("youtube.com")
        >>> browser.visit("github.com")
        >>> browser.back()
        'youtube.com'
        >>> browser.back()
        'google.com'
        >>> browser.forward()
        'youtube.com'
        >>> browser.visit("twitter.com")
        >>> browser.forward()  # 進む履歴はクリアされている
        'twitter.com'
    """

    def __init__(self, homepage: str):
        self._current = homepage
        self._back_stack = []
        self._forward_stack = []

    def visit(self, url: str):
        """新しいページを訪問。進む履歴はクリアされる。"""
        self._back_stack.append(self._current)
        self._current = url
        self._forward_stack.clear()

    def back(self) -> str:
        """1つ前のページに戻る。"""
        if self._back_stack:
            self._forward_stack.append(self._current)
            self._current = self._back_stack.pop()
        return self._current

    def forward(self) -> str:
        """1つ先のページに進む。"""
        if self._forward_stack:
            self._back_stack.append(self._current)
            self._current = self._forward_stack.pop()
        return self._current

    @property
    def current_page(self) -> str:
        return self._current


# === 動作確認 ===
if __name__ == "__main__":
    browser = BrowserHistory("google.com")
    browser.visit("youtube.com")
    browser.visit("github.com")
    browser.visit("stackoverflow.com")

    print(f"Current: {browser.current_page}")  # stackoverflow.com
    print(f"Back: {browser.back()}")           # github.com
    print(f"Back: {browser.back()}")           # youtube.com
    print(f"Forward: {browser.forward()}")     # github.com
    browser.visit("twitter.com")               # 進む履歴クリア
    print(f"Current: {browser.current_page}")  # twitter.com
    print(f"Forward: {browser.forward()}")     # twitter.com (進めない)
    print(f"Back: {browser.back()}")           # youtube.com
```

---

## 5. キューの応用アルゴリズム

### 5.1 幅優先探索（BFS: Breadth-First Search）

BFS はグラフや木の探索において、始点に近い頂点から順に探索する手法である。
キューを使って実装する。最短経路問題（辺の重みが等しい場合）の基本となる。

```python
from collections import deque
from typing import Dict, List, Optional


def bfs(graph: Dict[str, List[str]], start: str) -> List[str]:
    """
    グラフの幅優先探索。訪問順を返す。

    計算量: O(V + E)（V: 頂点数、E: 辺数）

    使用例:
        >>> graph = {
        ...     'A': ['B', 'C'],
        ...     'B': ['A', 'D', 'E'],
        ...     'C': ['A', 'F'],
        ...     'D': ['B'],
        ...     'E': ['B', 'F'],
        ...     'F': ['C', 'E'],
        ... }
        >>> bfs(graph, 'A')
        ['A', 'B', 'C', 'D', 'E', 'F']
    """
    visited = set()
    queue = deque([start])
    visited.add(start)
    order = []

    while queue:
        vertex = queue.popleft()
        order.append(vertex)
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return order


def bfs_shortest_path(
    graph: Dict[str, List[str]], start: str, goal: str
) -> Optional[List[str]]:
    """
    BFS で最短経路を求める（重みなしグラフ）。

    使用例:
        >>> graph = {
        ...     'A': ['B', 'C'],
        ...     'B': ['A', 'D', 'E'],
        ...     'C': ['A', 'F'],
        ...     'D': ['B'],
        ...     'E': ['B', 'F'],
        ...     'F': ['C', 'E'],
        ... }
        >>> bfs_shortest_path(graph, 'A', 'F')
        ['A', 'C', 'F']
    """
    if start == goal:
        return [start]

    visited = {start}
    queue = deque([(start, [start])])

    while queue:
        vertex, path = queue.popleft()
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                new_path = path + [neighbor]
                if neighbor == goal:
                    return new_path
                visited.add(neighbor)
                queue.append((neighbor, new_path))

    return None  # 到達不可能


# === 動作確認 ===
if __name__ == "__main__":
    graph = {
        'A': ['B', 'C'],
        'B': ['A', 'D', 'E'],
        'C': ['A', 'F'],
        'D': ['B'],
        'E': ['B', 'F'],
        'F': ['C', 'E'],
    }
    print(f"BFS order: {bfs(graph, 'A')}")
    # ['A', 'B', 'C', 'D', 'E', 'F']

    print(f"Shortest A->F: {bfs_shortest_path(graph, 'A', 'F')}")
    # ['A', 'C', 'F']

    print(f"Shortest A->D: {bfs_shortest_path(graph, 'A', 'D')}")
    # ['A', 'B', 'D']
```

```
BFS の動作（グラフ例）:

  グラフ構造:
       A
      / \
     B   C
    / \   \
   D   E - F

  BFS キューの変遷 (start = A):

  Step 0: queue=[A]          visited={A}
  Step 1: dequeue A → queue=[B,C]      visited={A,B,C}
  Step 2: dequeue B → queue=[C,D,E]    visited={A,B,C,D,E}
  Step 3: dequeue C → queue=[D,E,F]    visited={A,B,C,D,E,F}
  Step 4: dequeue D → queue=[E,F]      (D の隣接 B は訪問済み)
  Step 5: dequeue E → queue=[F]        (E の隣接 B,F は訪問済み)
  Step 6: dequeue F → queue=[]         (F の隣接 C,E は訪問済み)

  訪問順: A → B → C → D → E → F  (レベル順)
```

### 5.2 二分木のレベル順走査

```python
from collections import deque
from typing import List, Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def level_order_traversal(root: Optional[TreeNode]) -> List[List[int]]:
    """
    二分木をレベル順（BFS）で走査し、各レベルの値をリストで返す。
    LeetCode 102 の典型問題。

    計算量: O(n) 時間、O(n) 空間

    使用例:
        >>> #       3
        >>> #      / \
        >>> #     9   20
        >>> #        /  \
        >>> #       15   7
        >>> root = TreeNode(3,
        ...     TreeNode(9),
        ...     TreeNode(20, TreeNode(15), TreeNode(7))
        ... )
        >>> level_order_traversal(root)
        [[3], [9, 20], [15, 7]]
    """
    if root is None:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        level_values = []

        for _ in range(level_size):
            node = queue.popleft()
            level_values.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(level_values)

    return result


# === 動作確認 ===
if __name__ == "__main__":
    root = TreeNode(3,
        TreeNode(9),
        TreeNode(20, TreeNode(15), TreeNode(7))
    )
    print(f"Level order: {level_order_traversal(root)}")
    # [[3], [9, 20], [15, 7]]
```

### 5.3 タスクスケジューリング（ラウンドロビン）

OS のプロセススケジューリングで使われるラウンドロビン方式をキューで実装する。

```python
from collections import deque
from typing import List, Tuple


def round_robin_scheduling(
    tasks: List[Tuple[str, int]], time_quantum: int
) -> List[str]:
    """
    ラウンドロビン方式のタスクスケジューリング。
    各タスクに均等な CPU 時間（タイムクォンタム）を割り当てる。

    Args:
        tasks: (タスク名, 実行時間) のリスト
        time_quantum: 1回あたりの最大実行時間

    Returns:
        実行ログのリスト

    使用例:
        >>> tasks = [("P1", 10), ("P2", 4), ("P3", 6)]
        >>> log = round_robin_scheduling(tasks, 3)
    """
    queue = deque()
    for name, burst_time in tasks:
        queue.append([name, burst_time])

    log = []
    current_time = 0

    while queue:
        task = queue.popleft()
        name, remaining = task

        exec_time = min(remaining, time_quantum)
        current_time += exec_time
        remaining -= exec_time

        log.append(
            f"t={current_time-exec_time:3d}-{current_time:3d}: "
            f"{name} (remaining={remaining})"
        )

        if remaining > 0:
            queue.append([name, remaining])
        else:
            log.append(f"  >>> {name} completed at t={current_time}")

    return log


# === 動作確認 ===
if __name__ == "__main__":
    tasks = [("P1", 10), ("P2", 4), ("P3", 6)]
    log = round_robin_scheduling(tasks, 3)
    print("=== Round Robin Scheduling (quantum=3) ===")
    for entry in log:
        print(entry)
    # t=  0-  3: P1 (remaining=7)
    # t=  3-  6: P2 (remaining=1)
    # t=  6-  9: P3 (remaining=3)
    # t=  9- 12: P1 (remaining=4)
    # t= 12- 13: P2 (remaining=0)
    #   >>> P2 completed at t=13
    # t= 13- 16: P3 (remaining=0)
    #   >>> P3 completed at t=16
    # t= 16- 19: P1 (remaining=1)
    # t= 19- 20: P1 (remaining=0)
    #   >>> P1 completed at t=20
```

### 5.4 スネーク＆ラダー（BFS による最短手数）

盤面ゲームの最短解をBFSで求める例。キューを使ったBFSが最短経路に最適であることの好例。

```python
from collections import deque
from typing import Dict


def snakes_and_ladders(
    board_size: int,
    snakes: Dict[int, int],
    ladders: Dict[int, int]
) -> int:
    """
    スネーク＆ラダーで最短何手でゴールに到達できるかをBFSで求める。

    Args:
        board_size: 盤面のマス数 (1-indexed でゴールは board_size)
        snakes: {頭の位置: 尾の位置} 蛇で降下
        ladders: {底の位置: 頂の位置} 梯子で上昇

    Returns:
        最短手数。到達不可能なら -1。

    使用例:
        >>> snakes_and_ladders(
        ...     30,
        ...     snakes={17: 7, 27: 1},
        ...     ladders={3: 22, 5: 8, 11: 26, 20: 29}
        ... )
        3
    """
    goal = board_size
    shortcuts = {**snakes, **ladders}

    visited = set([1])
    queue = deque([(1, 0)])  # (現在位置, 手数)

    while queue:
        position, moves = queue.popleft()

        for dice in range(1, 7):  # サイコロ 1-6
            next_pos = position + dice
            if next_pos > goal:
                continue
            # 蛇や梯子があれば移動
            next_pos = shortcuts.get(next_pos, next_pos)
            if next_pos == goal:
                return moves + 1
            if next_pos not in visited:
                visited.add(next_pos)
                queue.append((next_pos, moves + 1))

    return -1


# === 動作確認 ===
if __name__ == "__main__":
    result = snakes_and_ladders(
        30,
        snakes={17: 7, 27: 1},
        ladders={3: 22, 5: 8, 11: 26, 20: 29}
    )
    print(f"Minimum moves: {result}")  # 3
    # 最短経路: 1 → (dice=2) → 3 → (ladder) → 22 → (dice=4) → 26
    #         → ... → (dice=4) → 29 → (dice=1) → 30 = goal
```

---

## 6. 優先度キューとヒープ

### 6.1 優先度キューの概念

通常のキューが FIFO（先着順）で要素を処理するのに対し、
優先度キューは**優先度が最も高い要素**を先に取り出す。

```
通常のキュー vs 優先度キュー:

  通常のキュー (FIFO):
    enqueue 順: A(pri=3), B(pri=1), C(pri=4), D(pri=2)
    dequeue 順: A → B → C → D  (先着順)

  優先度キュー (最小優先):
    enqueue 順: A(pri=3), B(pri=1), C(pri=4), D(pri=2)
    dequeue 順: B → D → A → C  (優先度が低い=重要度が高い順)

  内部のヒープ構造（最小ヒープ）:

    push A(3):    push B(1):    push C(4):    push D(2):
       3             1             1             1
                    / \           / \           / \
                   3             3   4         2   4
                                              /
                                             3
```

### 6.2 heapq を使った優先度キュー

```python
import heapq
from typing import Any, List, Tuple


class PriorityQueue:
    """
    最小ヒープベースの優先度キュー。
    同一優先度の場合は挿入順を保持する（FIFO tiebreaker）。

    使用例:
        >>> pq = PriorityQueue()
        >>> pq.push(3, "low priority task")
        >>> pq.push(1, "high priority task")
        >>> pq.push(2, "medium priority task")
        >>> pq.pop()
        (1, 'high priority task')
        >>> pq.pop()
        (2, 'medium priority task')
        >>> pq.pop()
        (3, 'low priority task')
    """

    def __init__(self):
        self._heap: List[Tuple[Any, int, Any]] = []
        self._counter = 0  # FIFO tiebreaker

    def push(self, priority: Any, item: Any):
        """優先度付きで要素を追加する。O(log n)。"""
        heapq.heappush(self._heap, (priority, self._counter, item))
        self._counter += 1

    def pop(self) -> Tuple[Any, Any]:
        """最も優先度の高い要素を取り出す。O(log n)。"""
        if self.is_empty():
            raise IndexError("pop from empty priority queue")
        priority, _, item = heapq.heappop(self._heap)
        return priority, item

    def peek(self) -> Tuple[Any, Any]:
        """最も優先度の高い要素を参照する。O(1)。"""
        if self.is_empty():
            raise IndexError("peek from empty priority queue")
        priority, _, item = self._heap[0]
        return priority, item

    def is_empty(self) -> bool:
        return len(self._heap) == 0

    def __len__(self) -> int:
        return len(self._heap)


# === 動作確認 ===
if __name__ == "__main__":
    pq = PriorityQueue()

    # 優先度の低い数値 = より優先
    pq.push(3, "backup database")
    pq.push(1, "fix critical bug")
    pq.push(2, "review PR")
    pq.push(1, "deploy hotfix")  # 同一優先度 → FIFO

    print("=== Processing tasks by priority ===")
    while not pq.is_empty():
        priority, task = pq.pop()
        print(f"  Priority {priority}: {task}")
    # Priority 1: fix critical bug
    # Priority 1: deploy hotfix
    # Priority 2: review PR
    # Priority 3: backup database
```

### 6.3 優先度キューの応用: k 個の最小要素

```python
import heapq
from typing import List


def k_smallest_elements(nums: List[int], k: int) -> List[int]:
    """
    リストから k 個の最小要素を返す。

    方法1: heapq.nsmallest — O(n log k)
    方法2: ソート — O(n log n)
    方法3: 最大ヒープ（サイズ k）— O(n log k)

    使用例:
        >>> k_smallest_elements([7, 10, 4, 3, 20, 15], 3)
        [3, 4, 7]
    """
    return heapq.nsmallest(k, nums)


def k_largest_elements(nums: List[int], k: int) -> List[int]:
    """
    リストから k 個の最大要素を返す。

    使用例:
        >>> k_largest_elements([7, 10, 4, 3, 20, 15], 3)
        [20, 15, 10]
    """
    return heapq.nlargest(k, nums)


def merge_k_sorted_lists(lists: List[List[int]]) -> List[int]:
    """
    k 個のソート済みリストをマージして1つのソート済みリストを返す。
    LeetCode 23 の典型問題。

    計算量: O(N log k) — N: 全要素数、k: リスト数

    使用例:
        >>> merge_k_sorted_lists([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
        [1, 2, 3, 4, 5, 6, 7, 8, 9]
    """
    result = []
    heap = []

    # 各リストの先頭要素をヒープに入れる
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))

    while heap:
        val, list_idx, elem_idx = heapq.heappop(heap)
        result.append(val)

        # 同じリストの次の要素があれば追加
        if elem_idx + 1 < len(lists[list_idx]):
            next_val = lists[list_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))

    return result


# === 動作確認 ===
if __name__ == "__main__":
    nums = [7, 10, 4, 3, 20, 15]
    print(f"3 smallest: {k_smallest_elements(nums, 3)}")  # [3, 4, 7]
    print(f"3 largest:  {k_largest_elements(nums, 3)}")    # [20, 15, 10]

    sorted_lists = [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
    print(f"Merged: {merge_k_sorted_lists(sorted_lists)}")
    # [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

### 6.4 優先度キューの実装方法比較

| 実装方法 | 挿入 | 最小値取り出し | 最小値参照 | 備考 |
|---------|------|-------------|-----------|------|
| 未ソート配列 | O(1) | O(n) | O(n) | 挿入は速いが取り出しが遅い |
| ソート済み配列 | O(n) | O(1) | O(1) | 取り出しは速いが挿入が遅い |
| 二分ヒープ | O(log n) | O(log n) | O(1) | **バランスが良い（標準）** |
| フィボナッチヒープ | O(1) 償却 | O(log n) 償却 | O(1) | 理論的に優れるが実装が複雑 |
| バランス BST | O(log n) | O(log n) | O(log n) | 汎用性は高いがオーバーヘッド |

---

## 7. 特殊なスタック・キュー構造

### 7.1 最小値取得 O(1) のスタック（MinStack）

LeetCode 155 の典型問題。push / pop / top に加え、最小値取得も O(1) で行う。

```python
class MinStack:
    """
    全操作 O(1) のスタック（最小値取得を含む）。

    アイデア:
      各要素とともに「その時点での最小値」をペアで記録する。

    使用例:
        >>> ms = MinStack()
        >>> ms.push(5)
        >>> ms.push(3)
        >>> ms.push(7)
        >>> ms.get_min()
        3
        >>> ms.pop()  # 7 を除去
        7
        >>> ms.get_min()
        3
        >>> ms.pop()  # 3 を除去
        3
        >>> ms.get_min()
        5
    """

    def __init__(self):
        self._stack = []  # (value, current_min) のペア

    def push(self, val: int):
        """O(1)。"""
        current_min = min(val, self._stack[-1][1]) if self._stack else val
        self._stack.append((val, current_min))

    def pop(self) -> int:
        """O(1)。"""
        if not self._stack:
            raise IndexError("pop from empty MinStack")
        return self._stack.pop()[0]

    def top(self) -> int:
        """O(1)。"""
        if not self._stack:
            raise IndexError("top from empty MinStack")
        return self._stack[-1][0]

    def get_min(self) -> int:
        """O(1) で現在の最小値を返す。"""
        if not self._stack:
            raise IndexError("get_min from empty MinStack")
        return self._stack[-1][1]

    def __len__(self) -> int:
        return len(self._stack)


# === 動作確認 ===
if __name__ == "__main__":
    ms = MinStack()
    operations = [
        ("push", 5), ("push", 2), ("push", 8),
        ("push", 1), ("pop", None), ("pop", None),
    ]
    for op, val in operations:
        if op == "push":
            ms.push(val)
            print(f"push({val}) → min={ms.get_min()}")
        else:
            popped = ms.pop()
            min_val = ms.get_min() if len(ms) > 0 else "N/A"
            print(f"pop() → {popped}, min={min_val}")
    # push(5) → min=5
    # push(2) → min=2
    # push(8) → min=2
    # push(1) → min=1
    # pop() → 1, min=2
    # pop() → 8, min=2
```

### 7.2 デック（Deque: 両端キュー）

デック（ダブルエンドキュー）は両端からの挿入・取り出しが O(1) で行える構造。
スタックとキューの両方の機能を兼ね備える。

```python
from collections import deque
from typing import List


def sliding_window_maximum(nums: List[int], k: int) -> List[int]:
    """
    スライディングウィンドウの最大値を求める。
    LeetCode 239 の典型問題。単調減少デックを使用。

    計算量: O(n) 時間、O(k) 空間

    アルゴリズム:
      デックにインデックスを格納し、デック内の値が単調減少になるよう維持。
      デックの先頭が常にウィンドウ内の最大値のインデックス。

    使用例:
        >>> sliding_window_maximum([1, 3, -1, -3, 5, 3, 6, 7], 3)
        [3, 3, 5, 5, 6, 7]
    """
    if not nums or k == 0:
        return []

    dq = deque()  # インデックスを格納（単調減少）
    result = []

    for i in range(len(nums)):
        # ウィンドウ外のインデックスを先頭から除去
        while dq and dq[0] < i - k + 1:
            dq.popleft()

        # 現在の要素より小さい要素のインデックスを末尾から除去
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()

        dq.append(i)

        # ウィンドウが完成したら最大値を記録
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result


# === 動作確認 ===
if __name__ == "__main__":
    nums = [1, 3, -1, -3, 5, 3, 6, 7]
    k = 3
    print(f"Input: {nums}, k={k}")
    print(f"Max values: {sliding_window_maximum(nums, k)}")
    # [3, 3, 5, 5, 6, 7]

    # 各ウィンドウの確認:
    # [1, 3, -1] → max=3
    # [3, -1, -3] → max=3
    # [-1, -3, 5] → max=5
    # [-3, 5, 3] → max=5
    # [5, 3, 6] → max=6
    # [3, 6, 7] → max=7
```

```
スライディングウィンドウ最大値のデック動作:

  nums = [1, 3, -1, -3, 5, 3, 6, 7],  k = 3

  i=0, num=1:  dq=[]       → append 0        dq=[0]
  i=1, num=3:  dq=[0]      → 1<3 → pop 0
                             → append 1        dq=[1]
  i=2, num=-1: dq=[1]      → 3>-1 → append 2  dq=[1,2]
               ★ window [0..2]: max = nums[dq[0]] = nums[1] = 3

  i=3, num=-3: dq=[1,2]    → -1>-3 → append 3 dq=[1,2,3]
               ★ window [1..3]: max = nums[1] = 3

  i=4, num=5:  dq=[1,2,3]  → dq[0]=1 < 4-3+1=2 → popleft
                             → nums[2]=-1<5 → pop, nums[3]=-3<5 → pop
                             → append 4        dq=[4]
               ★ window [2..4]: max = nums[4] = 5

  i=5, num=3:  dq=[4]      → 5>3 → append 5   dq=[4,5]
               ★ window [3..5]: max = nums[4] = 5

  i=6, num=6:  dq=[4,5]    → nums[5]=3<6 → pop, nums[4]=5<6 → pop
                             → append 6        dq=[6]
               ★ window [4..6]: max = nums[6] = 6

  i=7, num=7:  dq=[6]      → nums[6]=6<7 → pop
                             → append 7        dq=[7]
               ★ window [5..7]: max = nums[7] = 7

  結果: [3, 3, 5, 5, 6, 7]
```

### 7.3 スタックを使った DFS と再帰の相互変換

再帰は暗黙のスタック（コールスタック）を使用する。すべての再帰は明示的なスタックで
反復に変換できる。スタックオーバーフローのリスクを回避するために重要なテクニック。

```python
from typing import Dict, List


def dfs_recursive(
    graph: Dict[str, List[str]], start: str,
    visited: set = None
) -> List[str]:
    """再帰による DFS。"""
    if visited is None:
        visited = set()
    visited.add(start)
    order = [start]
    for neighbor in graph[start]:
        if neighbor not in visited:
            order.extend(dfs_recursive(graph, neighbor, visited))
    return order


def dfs_iterative(
    graph: Dict[str, List[str]], start: str
) -> List[str]:
    """明示的なスタックによる DFS（再帰を使わない）。"""
    visited = set()
    stack = [start]
    order = []

    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            order.append(vertex)
            # 逆順に push することで、隣接リストの先頭から探索する
            for neighbor in reversed(graph[vertex]):
                if neighbor not in visited:
                    stack.append(neighbor)

    return order


# === 動作確認 ===
if __name__ == "__main__":
    graph = {
        'A': ['B', 'C'],
        'B': ['A', 'D', 'E'],
        'C': ['A', 'F'],
        'D': ['B'],
        'E': ['B', 'F'],
        'F': ['C', 'E'],
    }
    print(f"DFS recursive: {dfs_recursive(graph, 'A')}")
    print(f"DFS iterative: {dfs_iterative(graph, 'A')}")
    # どちらも A → B → D → E → F → C の順（隣接リスト順による）
```

---

## 8. 比較表と選定ガイド

### 表1: スタック・キュー・デック・優先度キューの操作計算量

| 操作 | スタック | キュー (deque) | デック (deque) | 優先度キュー (ヒープ) |
|------|---------|---------------|---------------|---------------------|
| 先頭に挿入 | N/A | N/A | O(1) | N/A |
| 末尾に挿入 | O(1) push | O(1) enqueue | O(1) | O(log n) push |
| 先頭から取出 | N/A | O(1) dequeue | O(1) | O(log n) pop |
| 末尾から取出 | O(1) pop | N/A | O(1) | N/A |
| 先頭参照 | N/A | O(1) front | O(1) | O(1) peek |
| 末尾参照 | O(1) peek | O(1) rear | O(1) | N/A |
| 任意位置参照 | O(n) | O(n) | O(n) | O(n) |
| 探索 | O(n) | O(n) | O(n) | O(n) |
| 順序 | LIFO | FIFO | 両端 | 優先度順 |

### 表2: キューの実装方法と選定基準

| 実装方法 | enqueue | dequeue | 空間計算量 | メモリ特性 | 最適な場面 |
|---------|---------|---------|-----------|-----------|-----------|
| Python list | O(1) 償却 | O(n) | O(n) 動的 | 連続領域 | **使わない（非推奨）** |
| collections.deque | O(1) | O(1) | O(n) 動的 | ブロック連結 | **一般用途（推奨）** |
| 連結リスト | O(1) | O(1) | O(n)+ポインタ | 散在 | ノードの再利用がある場合 |
| 循環バッファ | O(1) | O(1) | O(n) 固定 | 連続領域 | 組み込み・固定容量 |
| 2つのスタック | O(1) | O(1) 償却 | O(n) | 連続領域 x 2 | 面接・制約付き問題 |
| queue.Queue | O(1) | O(1) | O(n) 動的 | ロック付き | **マルチスレッド** |
| asyncio.Queue | O(1) | O(1) | O(n) 動的 | 非同期対応 | **asyncio 環境** |

### 表3: データ構造の選定フローチャート

```
要素の取り出し順序は？
 │
 ├─ 最後に入れたものを先に → スタック
 │   └─ 最小値も O(1) で必要？ → MinStack
 │
 ├─ 最初に入れたものを先に → キュー
 │   ├─ スレッドセーフが必要？ → queue.Queue
 │   ├─ 非同期処理？ → asyncio.Queue
 │   ├─ 固定容量？ → CircularQueue
 │   └─ 一般用途 → collections.deque
 │
 ├─ 優先度の高いものを先に → 優先度キュー
 │   └─ Python 標準 → heapq + ラッパークラス
 │
 └─ 両端から操作したい → デック (collections.deque)
     └─ スライディングウィンドウ？ → 単調デック
```
