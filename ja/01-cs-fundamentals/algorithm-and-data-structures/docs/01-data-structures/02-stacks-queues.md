# スタックとキュー — 実装・応用・優先度キュー 完全ガイド

> LIFO / FIFO の原理に基づくスタックとキューを深く理解し、括弧マッチ、BFS、単調スタック、優先度キューなどの応用を体系的に学ぶ。

---


## この章で学ぶこと

- [ ] 基本概念と用語の理解
- [ ] 実装パターンとベストプラクティスの習得
- [ ] 実務での適用方法の把握
- [ ] トラブルシューティングの基本

---

## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解
- [連結リスト — 単方向・双方向・循環・フロイドのアルゴリズム](./01-linked-lists.md) の内容を理解していること

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
            stack.append(opstoken)
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

---

## 9. アンチパターンとベストプラクティス

### アンチパターン1: list.pop(0) をキューの dequeue に使う

Python の `list.pop(0)` は先頭要素を削除した後、残りの全要素を1つずつ前にシフトする。
これは O(n) の操作であり、要素数が多いと深刻なパフォーマンス劣化を引き起こす。

```python
import time
from collections import deque


def benchmark_queue_pop():
    """list.pop(0) vs deque.popleft() のベンチマーク。"""

    sizes = [10_000, 50_000, 100_000]

    for n in sizes:
        # --- list.pop(0) ---
        data_list = list(range(n))
        start = time.perf_counter()
        while data_list:
            data_list.pop(0)
        list_time = time.perf_counter() - start

        # --- deque.popleft() ---
        data_deque = deque(range(n))
        start = time.perf_counter()
        while data_deque:
            data_deque.popleft()
        deque_time = time.perf_counter() - start

        ratio = list_time / deque_time if deque_time > 0 else float('inf')
        print(
            f"n={n:>7,d}: "
            f"list.pop(0)={list_time:.4f}s, "
            f"deque.popleft()={deque_time:.4f}s, "
            f"ratio={ratio:.1f}x"
        )


# === 実行例 ===
if __name__ == "__main__":
    benchmark_queue_pop()
    # n= 10,000: list.pop(0)=0.0121s, deque.popleft()=0.0004s, ratio=30.3x
    # n= 50,000: list.pop(0)=0.2890s, deque.popleft()=0.0020s, ratio=144.5x
    # n=100,000: list.pop(0)=1.1520s, deque.popleft()=0.0040s, ratio=288.0x
    # ※ n が大きくなるほど差が顕著に開く
```

**教訓:** FIFO キューには必ず `collections.deque` を使う。`list.pop(0)` は O(n) であり、
要素数 100,000 程度で数百倍の速度差が生じる。

### アンチパターン2: スタックの中身を探索するために pop し続ける

スタックは本来「トップの要素のみにアクセスする」データ構造だが、
内部に特定の値が存在するか調べたい場面がある。このとき pop して復元するのは非効率かつバグの温床。

```python
# === BAD: 破壊的な探索 ===
def find_in_stack_bad(stack_data: list, target) -> bool:
    """pop して探索し、復元する。O(n) だが2倍の操作量+バグのリスク。"""
    temp = []
    found = False
    while stack_data:
        val = stack_data.pop()
        temp.append(val)
        if val == target:
            found = True
            break
    # 復元（break で途中終了した場合も全要素を戻す）
    while temp:
        stack_data.append(temp.pop())
    return found


# === GOOD: 内部データを直接走査 ===
def find_in_stack_good(stack_data: list, target) -> bool:
    """内部リストを直接走査する。O(n) で操作量は半分。"""
    return target in stack_data


# === 動作確認 ===
if __name__ == "__main__":
    stack = [10, 20, 30, 40, 50]

    # BAD
    result_bad = find_in_stack_bad(stack, 30)
    print(f"BAD: found={result_bad}, stack={stack}")
    # found=True, stack=[10, 20, 30, 40, 50]（復元されている）

    # GOOD
    result_good = find_in_stack_good(stack, 30)
    print(f"GOOD: found={result_good}, stack={stack}")
    # found=True, stack=[10, 20, 30, 40, 50]
```

**教訓:** スタック・キューの ADT としてのインターフェースと、実装の内部構造を区別する。
探索が必要なら、内部データ構造に直接アクセスするか、別のデータ構造（セットなど）を併用する。

### アンチパターン3: 再帰の深さを考慮しない DFS

```python
import sys


def dfs_naive(graph, start, visited=None):
    """再帰 DFS — 大規模グラフでスタックオーバーフローの危険。"""
    if visited is None:
        visited = set()
    visited.add(start)
    for neighbor in graph.get(start, []):
        if neighbor not in visited:
            dfs_naive(graph, neighbor, visited)  # 深い再帰！
    return visited


def dfs_safe(graph, start):
    """明示的スタックによる DFS — スタックオーバーフローを回避。"""
    visited = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    stack.append(neighbor)
    return visited


# === 動作確認 ===
if __name__ == "__main__":
    # Python のデフォルト再帰制限
    print(f"Default recursion limit: {sys.getrecursionlimit()}")
    # 通常は 1000

    # 線形グラフ（深さ 2000）を作成
    linear_graph = {i: [i + 1] for i in range(2000)}
    linear_graph[2000] = []

    # 再帰版: RecursionError が発生する可能性
    # dfs_naive(linear_graph, 0)  # RecursionError!

    # 反復版: 問題なく動作
    result = dfs_safe(linear_graph, 0)
    print(f"Iterative DFS visited {len(result)} nodes")  # 2001
```

**教訓:** 大規模グラフの DFS では明示的スタックを使う。Python の再帰制限（デフォルト 1000）を
超えるグラフでは `RecursionError` が発生する。`sys.setrecursionlimit()` で上限を上げる
方法もあるが、OS のスタック領域を消費するため推奨されない。

### アンチパターン4: 優先度キューで同一優先度の比較エラー

```python
import heapq


# === BAD: 比較不能なオブジェクトをヒープに入れる ===
class Task:
    def __init__(self, name, priority):
        self.name = name
        self.priority = priority

# heapq.heappush(heap, (1, Task("a", 1)))
# heapq.heappush(heap, (1, Task("b", 1)))
# → TypeError: '<' not supported between instances of 'Task' and 'Task'
# 同一優先度の場合、タプルの2番目の要素で比較しようとしてエラーになる


# === GOOD: カウンタで tiebreaker を入れる ===
class SafePriorityQueue:
    """同一優先度でも TypeError にならない優先度キュー。"""

    def __init__(self):
        self._heap = []
        self._counter = 0

    def push(self, priority, item):
        # counter が tiebreaker になり、item の比較は発生しない
        heapq.heappush(self._heap, (priority, self._counter, item))
        self._counter += 1

    def pop(self):
        priority, _, item = heapq.heappop(self._heap)
        return priority, item


# === 動作確認 ===
if __name__ == "__main__":
    pq = SafePriorityQueue()
    pq.push(1, Task("critical_fix", 1))
    pq.push(1, Task("urgent_deploy", 1))  # 同一優先度でもOK
    pq.push(2, Task("code_review", 2))

    while pq._heap:
        pri, task = pq.pop()
        print(f"Priority {pri}: {task.name}")
    # Priority 1: critical_fix
    # Priority 1: urgent_deploy
    # Priority 2: code_review
```

**教訓:** `heapq` にタプルを入れるとき、同一優先度の場合に2番目の要素が比較される。
比較不能なオブジェクトが入っていると `TypeError` になる。カウンタを tiebreaker として挟む。

---

## 10. 演習問題（3段階）

### 基礎レベル（Essential）

**演習 1-1: スタックで文字列を逆順にする**

```python
def reverse_string_with_stack(s: str) -> str:
    """
    スタックを使って文字列を逆順にする。
    ヒント: 全文字を push した後、全文字を pop する。

    >>> reverse_string_with_stack("hello")
    'olleh'
    >>> reverse_string_with_stack("abcde")
    'edcba'
    """
    stack = []
    for ch in s:
        stack.append(ch)

    result = []
    while stack:
        result.append(stack.pop())

    return ''.join(result)


# テスト
assert reverse_string_with_stack("hello") == "olleh"
assert reverse_string_with_stack("") == ""
assert reverse_string_with_stack("a") == "a"
print("Exercise 1-1: All tests passed!")
```

**演習 1-2: キューでホットポテトゲームをシミュレーション**

```python
from collections import deque
from typing import List


def hot_potato(names: List[str], num_passes: int) -> str:
    """
    ホットポテトゲーム（ジョセファス問題の変形）。
    円形に並んだ人が順番にポテトを渡し、
    num_passes 回目にポテトを持っている人が脱落する。
    最後に残った人が勝者。

    ヒント: dequeue して enqueue を繰り返す。

    >>> hot_potato(["Alice", "Bob", "Charlie", "David"], 3)
    'David'  # 脱落順: Charlie → Alice → Bob → David が勝者
    """
    queue = deque(names)

    while len(queue) > 1:
        for _ in range(num_passes):
            # ポテトを渡す = dequeue して enqueue
            queue.append(queue.popleft())
        eliminated = queue.popleft()
        print(f"  Eliminated: {eliminated}")

    winner = queue[0]
    print(f"  Winner: {winner}")
    return winner


# テスト
print("=== Hot Potato Game ===")
result = hot_potato(["Alice", "Bob", "Charlie", "David", "Eve"], 3)
```

**演習 1-3: 2つのキューでスタックを実装する**

```python
from collections import deque


class StackWithTwoQueues:
    """
    2つのキューを使ってスタックを実装する。
    （QueueWithTwoStacks の逆パターン）

    ヒント: push のたびに要素の順序を反転させる。

    >>> s = StackWithTwoQueues()
    >>> s.push(1)
    >>> s.push(2)
    >>> s.push(3)
    >>> s.pop()
    3
    >>> s.pop()
    2
    """

    def __init__(self):
        self._q1 = deque()
        self._q2 = deque()

    def push(self, val):
        """O(n) — push のたびに全要素を移し替える。"""
        self._q2.append(val)
        while self._q1:
            self._q2.append(self._q1.popleft())
        self._q1, self._q2 = self._q2, self._q1

    def pop(self):
        """O(1)。"""
        if not self._q1:
            raise IndexError("pop from empty stack")
        return self._q1.popleft()

    def top(self):
        """O(1)。"""
        if not self._q1:
            raise IndexError("top from empty stack")
        return self._q1[0]

    def is_empty(self):
        return len(self._q1) == 0


# テスト
s = StackWithTwoQueues()
for v in [10, 20, 30]:
    s.push(v)
assert s.pop() == 30
assert s.pop() == 20
assert s.top() == 10
assert not s.is_empty()
assert s.pop() == 10
assert s.is_empty()
print("Exercise 1-3: All tests passed!")
```

### 応用レベル（Intermediate）

**演習 2-1: 有効な括弧の最小削除数**

```python
def min_remove_to_make_valid(s: str) -> str:
    """
    無効な括弧を最小限削除して、有効な文字列を返す。
    LeetCode 1249 の問題。

    アルゴリズム:
      1. スタックに不正な括弧のインデックスを記録
      2. 記録されたインデックスの文字を除外

    >>> min_remove_to_make_valid("lee(t(c)o)de)")
    'lee(t(c)o)de'
    >>> min_remove_to_make_valid("a)b(c)d")
    'ab(c)d'
    >>> min_remove_to_make_valid("))((")
    ''
    """
    chars = list(s)
    stack = []  # 不正な括弧のインデックス

    for i, ch in enumerate(chars):
        if ch == '(':
            stack.append(i)
        elif ch == ')':
            if stack and chars[stack[-1]] == '(':
                stack.pop()
            else:
                stack.append(i)

    # 不正な括弧を除去
    remove_set = set(stack)
    return ''.join(ch for i, ch in enumerate(chars) if i not in remove_set)


# テスト
assert min_remove_to_make_valid("lee(t(c)o)de)") == "lee(t(c)o)de"
assert min_remove_to_make_valid("a)b(c)d") == "ab(c)d"
assert min_remove_to_make_valid("))((") == ""
assert min_remove_to_make_valid("(a(b(c)d)") == "a(b(c)d)"
print("Exercise 2-1: All tests passed!")
```

**演習 2-2: キューを使った壁と門の問題**

```python
from collections import deque
from typing import List

INF = float('inf')


def walls_and_gates(rooms: List[List[int]]) -> List[List[int]]:
    """
    -1 = 壁、0 = 門、INF = 空き部屋。
    各空き部屋から最も近い門までの距離を求める。
    LeetCode 286 の問題。多始点 BFS で解く。

    >>> rooms = [
    ...     [INF, -1,  0, INF],
    ...     [INF, INF, INF, -1],
    ...     [INF, -1,  INF, -1],
    ...     [0,   -1,  INF, INF],
    ... ]
    >>> walls_and_gates(rooms)
    [[3, -1, 0, 1], [2, 2, 1, -1], [1, -1, 2, -1], [0, -1, 3, 4]]
    """
    if not rooms or not rooms[0]:
        return rooms

    rows, cols = len(rooms), len(rooms[0])
    queue = deque()

    # 全ての門を起点としてキューに入れる（多始点 BFS）
    for r in range(rows):
        for c in range(cols):
            if rooms[r][c] == 0:
                queue.append((r, c))

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    while queue:
        r, c = queue.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols
                    and rooms[nr][nc] == INF):
                rooms[nr][nc] = rooms[r][c] + 1
                queue.append((nr, nc))

    return rooms


# テスト
rooms = [
    [INF, -1,  0, INF],
    [INF, INF, INF, -1],
    [INF, -1,  INF, -1],
    [0,   -1,  INF, INF],
]
result = walls_and_gates(rooms)
expected = [
    [3, -1, 0, 1],
    [2,  2, 1, -1],
    [1, -1, 2, -1],
    [0, -1, 3,  4],
]
assert result == expected
print("Exercise 2-2: All tests passed!")
```

**演習 2-3: 最大頻度スタック**

```python
from collections import defaultdict


class FreqStack:
    """
    頻度が最も高い要素を pop するスタック。
    同一頻度の場合は、最も最近に push された要素を優先する。
    LeetCode 895 の問題。

    アルデア:
      - freq[val]: 各値の現在の頻度
      - group[freq]: その頻度で push された値のスタック
      - max_freq: 現在の最大頻度

    >>> fs = FreqStack()
    >>> for v in [5, 7, 5, 7, 4, 5]:
    ...     fs.push(v)
    >>> fs.pop()  # 5 (freq=3, 最も頻出)
    5
    >>> fs.pop()  # 7 (freq=2, 5 と同率だが 7 の方が最近)
    7
    >>> fs.pop()  # 5 (freq=2)
    5
    >>> fs.pop()  # 4 (freq=1, 最も最近)
    4
    """

    def __init__(self):
        self.freq = defaultdict(int)
        self.group = defaultdict(list)
        self.max_freq = 0

    def push(self, val: int):
        self.freq[val] += 1
        f = self.freq[val]
        self.group[f].append(val)
        self.max_freq = max(self.max_freq, f)

    def pop(self) -> int:
        val = self.group[self.max_freq].pop()
        self.freq[val] -= 1
        if not self.group[self.max_freq]:
            self.max_freq -= 1
        return val


# テスト
fs = FreqStack()
for v in [5, 7, 5, 7, 4, 5]:
    fs.push(v)
assert fs.pop() == 5  # freq=3
assert fs.pop() == 7  # freq=2
assert fs.pop() == 5  # freq=2
assert fs.pop() == 4  # freq=1
print("Exercise 2-3: All tests passed!")
```

### 発展レベル（Advanced）

**演習 3-1: 全てのビルが見える方向の数**

```python
from typing import List


def count_visible_people(heights: List[int]) -> List[int]:
    """
    一列に並んだビルの各位置から、右方向に見える人の数を求める。
    背の高いビルが遮るため、自分より高いビルの向こう側は見えない。
    LeetCode 1944 の問題。単調スタックの発展問題。

    >>> count_visible_people([10, 6, 8, 5, 11, 9])
    [3, 1, 2, 1, 1, 0]
    """
    n = len(heights)
    result = [0] * n
    stack = []  # インデックスのスタック（単調減少）

    for i in range(n - 1, -1, -1):
        # 自分より低いビルは見える＆スタックから除去
        while stack and heights[stack[-1]] < heights[i]:
            stack.pop()
            result[i] += 1
        # 自分以上のビルも1つ見える（ただし遮られる）
        if stack:
            result[i] += 1
        stack.append(i)

    return result


# テスト
assert count_visible_people([10, 6, 8, 5, 11, 9]) == [3, 1, 2, 1, 1, 0]
assert count_visible_people([5, 1, 2, 3, 10]) == [4, 1, 1, 1, 0]
print("Exercise 3-1: All tests passed!")
```

**演習 3-2: 0-1 BFS（デックを使った最短経路）**

```python
from collections import deque
from typing import List, Tuple

INF = float('inf')


def zero_one_bfs(
    n: int, edges: List[Tuple[int, int, int]], start: int
) -> List[int]:
    """
    辺の重みが 0 か 1 のグラフで最短距離を求める。
    通常の BFS と Dijkstra の中間に位置するアルゴリズム。
    デックを使って O(V + E) で解く。

    アルゴリズム:
      - 重み 0 の辺 → デックの先頭に追加（すぐに処理）
      - 重み 1 の辺 → デックの末尾に追加（後で処理）

    Args:
        n: 頂点数 (0-indexed)
        edges: (from, to, weight) のリスト。weight は 0 か 1。
        start: 始点

    Returns:
        各頂点への最短距離のリスト

    >>> zero_one_bfs(5, [(0,1,1),(0,2,0),(1,3,1),(2,3,0),(3,4,1)], 0)
    [0, 1, 0, 0, 1]
    """
    # 隣接リストを構築
    graph = [[] for _ in range(n)]
    for u, v, w in edges:
        graph[u].append((v, w))

    dist = [INF] * n
    dist[start] = 0
    dq = deque([start])

    while dq:
        u = dq.popleft()
        for v, w in graph[u]:
            new_dist = dist[u] + w
            if new_dist < dist[v]:
                dist[v] = new_dist
                if w == 0:
                    dq.appendleft(v)  # 重み0 → 先頭
                else:
                    dq.append(v)      # 重み1 → 末尾

    return [d if d != INF else -1 for d in dist]


# テスト
edges = [(0,1,1), (0,2,0), (1,3,1), (2,3,0), (3,4,1)]
result = zero_one_bfs(5, edges, 0)
assert result == [0, 1, 0, 0, 1]
print("Exercise 3-2: All tests passed!")
```

**演習 3-3: 中央値を O(log n) で取得するデータストリーム**

```python
import heapq


class MedianFinder:
    """
    データストリームから中央値を効率的に取得する。
    LeetCode 295 の問題。2つのヒープ（最大ヒープ+最小ヒープ）を使用。

    アイデア:
      - max_heap: 小さい方の半分（最大ヒープ）
      - min_heap: 大きい方の半分（最小ヒープ）
      - 常に |max_heap| == |min_heap| or |max_heap| == |min_heap| + 1 を維持

    >>> mf = MedianFinder()
    >>> mf.add_num(1)
    >>> mf.find_median()
    1.0
    >>> mf.add_num(2)
    >>> mf.find_median()
    1.5
    >>> mf.add_num(3)
    >>> mf.find_median()
    2.0
    """

    def __init__(self):
        self._max_heap = []  # 小さい半分（符号反転で最大ヒープ化）
        self._min_heap = []  # 大きい半分

    def add_num(self, num: int):
        """O(log n)。"""
        # まず max_heap に追加
        heapq.heappush(self._max_heap, -num)
        # max_heap の最大値を min_heap に移す
        heapq.heappush(self._min_heap, -heapq.heappop(self._max_heap))
        # サイズのバランスを取る
        if len(self._min_heap) > len(self._max_heap):
            heapq.heappush(
                self._max_heap, -heapq.heappop(self._min_heap)
            )

    def find_median(self) -> float:
        """O(1)。"""
        if len(self._max_heap) > len(self._min_heap):
            return float(-self._max_heap[0])
        return (-self._max_heap[0] + self._min_heap[0]) / 2.0


# テスト
mf = MedianFinder()
mf.add_num(1)
assert mf.find_median() == 1.0
mf.add_num(2)
assert mf.find_median() == 1.5
mf.add_num(3)
assert mf.find_median() == 2.0
mf.add_num(4)
assert mf.find_median() == 2.5
mf.add_num(5)
assert mf.find_median() == 3.0
print("Exercise 3-3: All tests passed!")
```

---

## 11. FAQ

### Q1: スタックとキューは再帰とどう関係するか？

**A:** 再帰は暗黙のスタック（コールスタック）を使用する。関数呼び出しのたびにスタックフレーム
（ローカル変数、戻りアドレス等）が積まれ、`return` で取り除かれる。したがって、
すべての再帰アルゴリズムは明示的なスタックを使って反復に書き換えることができる。

典型的な対応関係:
- **DFS（深さ優先探索）** → 再帰 or 明示的スタック
- **BFS（幅優先探索）** → キュー（再帰では自然に表現しにくい）
- **木の前順・中順・後順走査** → 再帰 or 明示的スタック

### Q2: Python の collections.deque はスレッドセーフか？

**A:** CPython の GIL（Global Interpreter Lock）の下では、`deque.append()` と
`deque.popleft()` は個別にアトミックに実行される。しかし、複数の操作を組み合わせた
処理（例: 「サイズをチェックしてから pop する」）はアトミックではない。

マルチスレッド環境では以下を推奨する:
- **`queue.Queue`**: 内部でロックを使用。ブロッキング機能あり
- **`queue.LifoQueue`**: スレッドセーフなスタック
- **`queue.PriorityQueue`**: スレッドセーフな優先度キュー

```python
from queue import Queue, LifoQueue, PriorityQueue

# スレッドセーフなキュー
q = Queue(maxsize=100)
q.put("task1")         # ブロッキング put
item = q.get()         # ブロッキング get
q.task_done()          # タスク完了通知

# スレッドセーフなスタック
stack = LifoQueue()
stack.put("item1")
stack.get()  # LIFO 順

# スレッドセーフな優先度キュー
pq = PriorityQueue()
pq.put((1, "high"))
pq.put((3, "low"))
pq.get()  # (1, "high")
```

### Q3: 優先度キューを配列ソートで実装しない理由は？

**A:** 各実装方法の計算量を比較すると:

| 方式 | 挿入 | 最小値取出 | N回の挿入+取出 |
|------|------|-----------|---------------|
| 未ソート配列 | O(1) | O(n) | O(n^2) |
| ソート済み配列 | O(n) | O(1) | O(n^2) |
| 二分ヒープ | O(log n) | O(log n) | **O(n log n)** |

N = 1,000,000 の場合:
- 配列方式: 約 10^12 回の操作
- ヒープ方式: 約 2 x 10^7 回の操作（約 50,000 倍高速）

したがって、挿入と取り出しが混在する一般的なユースケースでは、ヒープベースの
優先度キューが圧倒的に有利である。

### Q4: deque と list はどちらが速いのか？

**A:** 操作の種類による。

| 操作 | list | deque | 勝者 |
|------|------|-------|------|
| 末尾追加 (append) | O(1) 償却 | O(1) | ほぼ同等 |
| 末尾削除 (pop) | O(1) | O(1) | ほぼ同等 |
| 先頭追加 (appendleft) | O(n) | O(1) | **deque** |
| 先頭削除 (popleft) | O(n) | O(1) | **deque** |
| ランダムアクセス [i] | O(1) | O(n) | **list** |
| スライス [a:b] | O(k) | O(n) | **list** |
| メモリ効率 | 高い | やや低い | **list** |

結論: キュー用途（先頭からの削除が頻繁）なら `deque`、ランダムアクセスが頻繁なら `list`。

### Q5: 実務でスタック・キューが使われる場面は？

**A:** 代表的な実務での利用場面:

1. **Web サーバーのリクエストキュー** — 受信リクエストを FIFO で処理
2. **メッセージキュー（RabbitMQ, SQS, Kafka）** — 非同期処理の基盤
3. **Undo/Redo 機能** — テキストエディタ、画像編集ソフトでスタックを使用
4. **ブラウザの戻る/進む** — 2つのスタックで履歴管理
5. **コンパイラの構文解析** — 式の評価、括弧マッチにスタックを使用
6. **OS のプロセススケジューラ** — ラウンドロビンにキューを使用
7. **BFS/DFS によるグラフ探索** — ソーシャルグラフ、経路探索
8. **Dijkstra 法の最短経路** — 優先度キューが不可欠

---


## FAQ

### Q1: このトピックを学ぶ上で最も重要なポイントは何ですか？

実践的な経験を積むことが最も重要です。理論だけでなく、実際にコードを書いて動作を確認することで理解が深まります。

### Q2: 初心者がよく陥る間違いは何ですか？

基礎を飛ばして応用に進むことです。このガイドで説明している基本概念をしっかり理解してから、次のステップに進むことをお勧めします。

### Q3: 実務ではどのように活用されていますか？

このトピックの知識は、日常的な開発業務で頻繁に活用されます。特にコードレビューやアーキテクチャ設計の際に重要になります。

---

## 12. まとめ

### 本ガイドの要点

| 項目 | ポイント |
|------|---------|
| スタック (LIFO) | DFS、括弧マッチ、逆ポーランド記法、Undo/Redo。配列ベースが推奨 |
| キュー (FIFO) | BFS、タスクスケジューリング、メッセージキュー。`deque` が推奨 |
| 循環バッファ | 固定容量キュー。組み込みシステム、カーネルバッファリング |
| 2つのスタックでキュー | 面接頻出。償却 O(1) の証明が重要 |
| 単調スタック | 「次に大きい/小さい要素」系の問題に O(n) で対応 |
| 優先度キュー | ヒープで実装。Dijkstra、A*、k-way merge に不可欠 |
| MinStack | ペア格納で最小値取得を O(1) に |
| デック | 両端 O(1)。スライディングウィンドウ、0-1 BFS に活用 |
| スレッドセーフ | `queue.Queue` / `asyncio.Queue` を使用 |

### 学習ロードマップ

```
レベル1: 基本操作を理解
  └→ スタック push/pop、キュー enqueue/dequeue を実装

レベル2: 典型的な応用を習得
  └→ 括弧マッチ、BFS、RPN 評価

レベル3: 発展的なテクニック
  └→ 単調スタック、スライディングウィンドウ最大値、0-1 BFS

レベル4: 実務・設計レベル
  └→ メッセージキュー設計、スレッドセーフ、優先度キューの応用
```

---

## 次に読むべきガイド

- [ヒープ — 二分ヒープと優先度キュー実装](./05-heaps.md)
- [グラフ走査 — BFS/DFS](../02-algorithms/02-graph-traversal.md)
- 連結リスト — ノードベースの基本構造

---

## 13. 参考文献

1. Cormen, T.H., Leiserson, C.E., Rivest, R.L. & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. — 第10章「基本データ構造」: スタック、キュー、連結リストの形式的定義と解析
2. Sedgewick, R. & Wayne, K. (2011). *Algorithms* (4th ed.). Addison-Wesley. — 第1.3節「バッグ、キュー、スタック」: 配列・連結リストベースの実装を詳解
3. Skiena, S.S. (2020). *The Algorithm Design Manual* (3rd ed.). Springer. — 第3章「データ構造」: スタック・キューの実務での活用パターン
4. Python Software Foundation. "collections.deque." Python 3 Documentation. https://docs.python.org/3/library/collections.html#collections.deque — Python の deque 実装仕様と API リファレンス
5. Python Software Foundation. "queue --- A synchronized queue class." Python 3 Documentation. https://docs.python.org/3/library/queue.html — スレッドセーフなキュー実装
6. Goodrich, M.T., Tamassia, R. & Goldwasser, M.H. (2013). *Data Structures and Algorithms in Python*. Wiley. — 第6章「スタック、キュー、デック」: Python での実装に特化
7. Kleinberg, J. & Tardos, E. (2005). *Algorithm Design*. Pearson. — 第3章「グラフ」: BFS のキューベース実装と最短経路への応用

---

## 参考文献

- [MDN Web Docs](https://developer.mozilla.org/) - Web技術のリファレンス
- [Wikipedia](https://ja.wikipedia.org/) - 技術概念の概要
