# スタックとキュー

> スタック（LIFO）は関数呼び出しと括弧の対応を、キュー（FIFO）はタスクスケジューリングとBFSを支える。

## この章で学ぶこと

- [ ] スタック（LIFO）とキュー（FIFO）の概念・操作・計算量を完全に説明できる
- [ ] 配列実装と連結リスト実装の両方をゼロから書ける
- [ ] コールスタック・括弧検証・逆ポーランド記法など典型問題を解ける
- [ ] 循環バッファによるキュー実装を理解し、利点を説明できる
- [ ] 両端キュー（Deque）の仕組みと用途を把握する
- [ ] 優先度キュー（ヒープ）の内部構造を理解し、ダイクストラ法に適用できる
- [ ] 単調スタック・単調キューで O(n) の最適解を導ける
- [ ] Undo/Redo、BFS/DFS、タスクスケジューラ等の実務応用を設計できる
- [ ] CPython の list や Java の ArrayDeque の内部実装を説明できる
- [ ] 各データ構造のトレードオフを比較し、場面に応じて最適な選択ができる

## 前提知識

- 配列の基本操作（インデックスアクセス、挿入、削除） → 参照: [[00-arrays-and-strings.md]]
- ポインタ・参照の基本概念
- 計算量表記（ビッグO）の読み方 → 参照: [[../03-algorithms/00-big-o-notation.md]]

---

## 1. なぜスタックとキューが必要か

配列やリストは汎用的なデータ格納手段だが、「どの順序でデータを取り出すか」に制約を課すことで、アルゴリズムの意図を明確にし、バグを減らし、パフォーマンスを保証できる。

スタックとキューは **制約付きの線形データ構造** である。操作を限定することで以下の恩恵を得る。

1. **意図の明示**: push/pop だけを許すことで「最後に入れたものを最初に取り出す」という不変条件をコードレベルで保証する
2. **バグの防止**: 任意位置へのアクセスを禁止することで、不正な操作を型レベル・インタフェースレベルで排除する
3. **計算量の保証**: 主要操作がすべて O(1) であることを構造的に担保する
4. **アルゴリズムとの対応**: DFS にはスタック、BFS にはキューという対応関係が、探索の正しさを保証する

### 1.1 日常のアナロジー

```
スタック（LIFO: Last In, First Out）        キュー（FIFO: First In, First Out）
┌─────────────────────┐                     ┌─────────────────────────────────┐
│ 本の積み重ね         │                     │ レジの行列                       │
│                     │                     │                                 │
│  ┌───┐ ← 最後に置いた │                     │  先頭 → [A] [B] [C] [D] ← 末尾  │
│  │ C │   本を最初に取る│                     │  最初に並んだ人が最初にサービス   │
│  ├───┤              │                     │  を受ける                        │
│  │ B │              │                     │                                 │
│  ├───┤              │                     │  enqueue(E):                     │
│  │ A │              │                     │  [A] [B] [C] [D] [E]            │
│  └───┘              │                     │                                 │
│  pop → C を取り出し  │                     │  dequeue → A を取り出し           │
└─────────────────────┘                     └─────────────────────────────────┘
```

スタックは「皿の山」「ブラウザの戻るボタン」「Ctrl+Z の Undo 履歴」に対応する。キューは「プリンタの印刷待ち」「コールセンターの待ち行列」「メッセージキュー」に対応する。

### 1.2 抽象データ型（ADT）としての定義

スタックとキューは **抽象データ型（Abstract Data Type）** である。つまり、内部の実装方法は問わず、外部から見たインタフェース（操作の集合）で定義される。

**スタック ADT**:
- `push(x)`: 要素 x をトップに追加する
- `pop()`: トップの要素を削除して返す
- `peek()` / `top()`: トップの要素を削除せずに返す
- `is_empty()`: スタックが空かどうかを返す
- `size()`: 要素数を返す

**キュー ADT**:
- `enqueue(x)`: 要素 x を末尾に追加する
- `dequeue()`: 先頭の要素を削除して返す
- `peek()` / `front()`: 先頭の要素を削除せずに返す
- `is_empty()`: キューが空かどうかを返す
- `size()`: 要素数を返す

これらの ADT を実現する具体的なデータ構造（配列、連結リスト、循環バッファ等）は複数存在し、それぞれに長所と短所がある。以下のセクションで順に見ていく。

---

## 2. スタック（Stack）

### 2.1 配列によるスタック実装

最も単純なスタック実装は、動的配列（Python の list、Java の ArrayList）の末尾を利用するものである。

#### Python 実装

```python
class ArrayStack:
    """動的配列ベースのスタック実装"""

    def __init__(self):
        self._data = []

    def push(self, value):
        """要素をトップに追加する: O(1) 償却"""
        self._data.append(value)

    def pop(self):
        """トップの要素を削除して返す: O(1) 償却"""
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._data.pop()

    def peek(self):
        """トップの要素を参照する: O(1)"""
        if self.is_empty():
            raise IndexError("peek from empty stack")
        return self._data[-1]

    def is_empty(self):
        """空かどうか: O(1)"""
        return len(self._data) == 0

    def size(self):
        """要素数: O(1)"""
        return len(self._data)

    def __repr__(self):
        return f"ArrayStack({self._data})"


# 動作確認
if __name__ == "__main__":
    s = ArrayStack()
    s.push(10)
    s.push(20)
    s.push(30)
    print(s)            # ArrayStack([10, 20, 30])
    print(s.peek())     # 30
    print(s.pop())      # 30
    print(s.pop())      # 20
    print(s.size())     # 1
    print(s.is_empty()) # False
    print(s.pop())      # 10
    print(s.is_empty()) # True
```

**計算量分析**:
- `push`: O(1) 償却。内部配列の容量が足りないときだけ O(n) のリサイズが発生するが、倍々に拡張するため償却 O(1)。
- `pop`: O(1) 償却。同様の理由。
- `peek`: O(1)。末尾要素のインデックスアクセスのみ。
- 空間計算量: O(n)。ただし、動的配列の未使用領域分のオーバーヘッドがある。

#### C 言語実装（固定サイズ配列）

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define MAX_SIZE 1024

typedef struct {
    int data[MAX_SIZE];
    int top;  /* 次に push する位置（=要素数） */
} Stack;

void stack_init(Stack *s) {
    s->top = 0;
}

bool stack_is_empty(const Stack *s) {
    return s->top == 0;
}

bool stack_is_full(const Stack *s) {
    return s->top == MAX_SIZE;
}

void stack_push(Stack *s, int value) {
    if (stack_is_full(s)) {
        fprintf(stderr, "Error: stack overflow\n");
        exit(EXIT_FAILURE);
    }
    s->data[s->top++] = value;
}

int stack_pop(Stack *s) {
    if (stack_is_empty(s)) {
        fprintf(stderr, "Error: stack underflow\n");
        exit(EXIT_FAILURE);
    }
    return s->data[--s->top];
}

int stack_peek(const Stack *s) {
    if (stack_is_empty(s)) {
        fprintf(stderr, "Error: peek from empty stack\n");
        exit(EXIT_FAILURE);
    }
    return s->data[s->top - 1];
}

int main(void) {
    Stack s;
    stack_init(&s);

    stack_push(&s, 10);
    stack_push(&s, 20);
    stack_push(&s, 30);

    printf("peek: %d\n", stack_peek(&s));  /* 30 */
    printf("pop:  %d\n", stack_pop(&s));   /* 30 */
    printf("pop:  %d\n", stack_pop(&s));   /* 20 */
    printf("empty: %s\n", stack_is_empty(&s) ? "true" : "false"); /* false */

    return 0;
}
```

C 言語版では固定サイズ配列を使っているため、スタックオーバーフローのチェックが必須である。組み込みシステムやカーネルモジュールでは、この固定サイズ方式がメモリアロケーションのオーバーヘッドを回避するために好まれる。

### 2.2 連結リストによるスタック実装

連結リストの先頭をスタックのトップとして使うと、push/pop が常に O(1)（最悪計算量も O(1)）で実現できる。動的配列のように容量のリサイズが不要という利点がある。

```python
class Node:
    """単方向連結リストのノード"""
    __slots__ = ('value', 'next')

    def __init__(self, value, next_node=None):
        self.value = value
        self.next = next_node


class LinkedStack:
    """連結リストベースのスタック実装"""

    def __init__(self):
        self._top = None
        self._size = 0

    def push(self, value):
        """新しいノードを先頭に追加: O(1) 最悪"""
        self._top = Node(value, self._top)
        self._size += 1

    def pop(self):
        """先頭ノードを削除して値を返す: O(1) 最悪"""
        if self.is_empty():
            raise IndexError("pop from empty stack")
        value = self._top.value
        self._top = self._top.next
        self._size -= 1
        return value

    def peek(self):
        """先頭ノードの値を参照: O(1)"""
        if self.is_empty():
            raise IndexError("peek from empty stack")
        return self._top.value

    def is_empty(self):
        return self._size == 0

    def size(self):
        return self._size

    def __iter__(self):
        """トップからボトムへの走査"""
        current = self._top
        while current is not None:
            yield current.value
            current = current.next

    def __repr__(self):
        return f"LinkedStack([{', '.join(str(v) for v in self)}])"


# 動作確認
if __name__ == "__main__":
    s = LinkedStack()
    for v in [10, 20, 30]:
        s.push(v)
    print(s)            # LinkedStack([30, 20, 10])
    print(s.peek())     # 30
    print(s.pop())      # 30
    print(s.size())     # 2
```

**連結リスト方式の図解**:

```
push(10) → push(20) → push(30):

  top
   │
   v
 ┌────┬───┐    ┌────┬───┐    ┌────┬──────┐
 │ 30 │ ──┼───>│ 20 │ ──┼───>│ 10 │ None │
 └────┴───┘    └────┴───┘    └────┴──────┘

pop() → 30 を返し、top を次のノードに移動:

  top
   │
   v
 ┌────┬───┐    ┌────┬──────┐
 │ 20 │ ──┼───>│ 10 │ None │
 └────┴───┘    └────┴──────┘
```

### 2.3 配列実装 vs 連結リスト実装の比較

| 観点 | 配列（動的配列）ベース | 連結リストベース |
|------|----------------------|----------------|
| push の計算量 | O(1) 償却（リサイズ時 O(n)） | O(1) 最悪 |
| pop の計算量 | O(1) 償却 | O(1) 最悪 |
| メモリ効率 | 連続メモリ、キャッシュフレンドリー | ノードごとにポインタのオーバーヘッド |
| メモリ割り当て | リサイズ時にまとめて割り当て | 要素ごとに割り当て |
| キャッシュ性能 | 優秀（空間的局所性が高い） | 劣る（ノードが散在しうる） |
| 最大サイズ | 動的に拡張可能（連続メモリが必要） | メモリの断片があっても利用可能 |
| 実装の簡潔さ | 非常に簡潔（言語組み込みを利用） | やや複雑（ノード管理が必要） |
| リアルタイム性 | リサイズ時にスパイクあり | 完全に一定時間 |

**推奨**: 一般的な用途では配列ベースが最適（キャッシュ効率が高く、実装が簡潔）。リアルタイムシステムや最悪計算量の保証が必要な場面では連結リストベースを検討する。

### 2.4 コールスタック — プログラムの根幹

プログラムの実行において、関数呼び出しはスタックで管理される。これを **コールスタック（Call Stack）** と呼ぶ。

```python
def factorial(n):
    """再帰的に階乗を計算する"""
    if n <= 1:
        return 1
    return n * factorial(n - 1)

print(factorial(5))  # 120
```

`factorial(5)` を呼び出すと、コールスタックは以下のように変化する。

```
呼び出し時（スタックが成長する方向）:

  factorial(5) を呼び出し
  ┌─────────────────┐
  │ factorial(1)     │ ← n=1, return 1
  ├─────────────────┤
  │ factorial(2)     │ ← n=2, 待機中: 2 * factorial(1)
  ├─────────────────┤
  │ factorial(3)     │ ← n=3, 待機中: 3 * factorial(2)
  ├─────────────────┤
  │ factorial(4)     │ ← n=4, 待機中: 4 * factorial(3)
  ├─────────────────┤
  │ factorial(5)     │ ← n=5, 待機中: 5 * factorial(4)
  ├─────────────────┤
  │ main()           │ ← 最初の呼び出し元
  └─────────────────┘

戻り時（スタックが縮小する方向）:

  factorial(1) → 1 を返す
  factorial(2) → 2 * 1 = 2 を返す
  factorial(3) → 3 * 2 = 6 を返す
  factorial(4) → 4 * 6 = 24 を返す
  factorial(5) → 5 * 24 = 120 を返す
```

各フレームには、関数のローカル変数、戻りアドレス、引数の情報が含まれる。再帰が深すぎると **スタックオーバーフロー** が発生する（Python のデフォルト再帰上限は 1000）。

```python
import sys
print(sys.getrecursionlimit())  # 1000

# 再帰上限の変更（推奨しない。末尾再帰最適化か反復に書き換えるべき）
# sys.setrecursionlimit(10000)
```

### 2.5 括弧の対応検証

括弧の対応チェックは、スタックの代表的な応用問題である。

```python
def is_valid_parentheses(s: str) -> bool:
    """
    括弧が正しく対応しているかを検証する。
    対応する括弧: (), [], {}

    アルゴリズム:
    1. 開き括弧が来たらスタックに push
    2. 閉じ括弧が来たら:
       a. スタックが空なら False（対応する開き括弧がない）
       b. トップの開き括弧と対応しなければ False
       c. 対応すれば pop
    3. 最後にスタックが空なら True（すべて対応した）

    計算量: O(n) 時間、O(n) 空間
    """
    stack = []
    pairs = {')': '(', ']': '[', '}': '{'}

    for char in s:
        if char in '([{':
            stack.append(char)
        elif char in ')]}':
            if not stack:
                return False  # 対応する開き括弧がない
            if stack[-1] != pairs[char]:
                return False  # 括弧の種類が不一致
            stack.pop()
        # 括弧以外の文字は無視

    return len(stack) == 0  # 未対応の開き括弧が残っていないか


# テストケース
assert is_valid_parentheses("()") == True
assert is_valid_parentheses("()[]{}") == True
assert is_valid_parentheses("(]") == False
assert is_valid_parentheses("([)]") == False
assert is_valid_parentheses("{[()]}") == True
assert is_valid_parentheses("") == True
assert is_valid_parentheses("(") == False
assert is_valid_parentheses(")") == False
print("All tests passed!")
```

### 2.6 逆ポーランド記法（後置記法）の評価

逆ポーランド記法（Reverse Polish Notation, RPN）は、演算子を被演算子の後に置く表記法である。括弧が不要になるため、計算機の内部処理に適している。

```python
def eval_rpn(tokens: list) -> int:
    """
    逆ポーランド記法の式を評価する。

    例: ["2", "1", "+", "3", "*"] → (2 + 1) * 3 = 9

    アルゴリズム:
    1. 数値ならスタックに push
    2. 演算子なら、スタックから2つ pop して計算し、結果を push
    3. 最後にスタックに残った1つの値が答え

    計算量: O(n) 時間、O(n) 空間
    """
    stack = []
    operators = {'+', '-', '*', '/'}

    for token in tokens:
        if token in operators:
            b = stack.pop()  # 2番目のオペランド（後に push されたもの）
            a = stack.pop()  # 1番目のオペランド（先に push されたもの）
            if token == '+':
                stack.append(a + b)
            elif token == '-':
                stack.append(a - b)
            elif token == '*':
                stack.append(a * b)
            elif token == '/':
                # C 言語風のゼロ方向への切り捨て
                stack.append(int(a / b))
        else:
            stack.append(int(token))

    return stack[0]


# テストケース
assert eval_rpn(["2", "1", "+", "3", "*"]) == 9
assert eval_rpn(["4", "13", "5", "/", "+"]) == 6
assert eval_rpn(["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]) == 22
print("All RPN tests passed!")
```

**処理の流れの図解**:

```
式: ["2", "1", "+", "3", "*"]

ステップ1: "2" → push(2)     スタック: [2]
ステップ2: "1" → push(1)     スタック: [2, 1]
ステップ3: "+" → pop 1, 2    スタック: [3]       (2+1=3)
           push(3)
ステップ4: "3" → push(3)     スタック: [3, 3]
ステップ5: "*" → pop 3, 3    スタック: [9]       (3*3=9)
           push(9)

結果: 9
```

---

## 3. キュー（Queue）

### 3.1 配列によるキュー実装（素朴な方法とその問題点）

Python の list で素朴にキューを実装すると、dequeue が O(n) になる問題がある。

```python
class NaiveArrayQueue:
    """
    素朴な配列ベースキュー（問題あり）
    dequeue が O(n) であり、大量のデータには不適切
    """

    def __init__(self):
        self._data = []

    def enqueue(self, value):
        """末尾に追加: O(1) 償却"""
        self._data.append(value)

    def dequeue(self):
        """先頭から削除: *** O(n) *** — 全要素のシフトが発生"""
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        return self._data.pop(0)  # これが O(n) のボトルネック！

    def peek(self):
        if self.is_empty():
            raise IndexError("peek from empty queue")
        return self._data[0]

    def is_empty(self):
        return len(self._data) == 0

    def size(self):
        return len(self._data)
```

`pop(0)` は先頭要素を削除した後、残りの全要素を1つ前にシフトする必要があるため O(n) である。これは 10 万件規模のデータで著しく遅くなる。

### 3.2 循環バッファによるキュー実装

循環バッファ（Ring Buffer / Circular Buffer）を使えば、固定サイズの配列で enqueue/dequeue の両方を O(1) で実現できる。

```python
class CircularQueue:
    """
    循環バッファベースのキュー実装。
    固定サイズの配列で enqueue/dequeue を O(1) で実現する。
    """

    def __init__(self, capacity=16):
        self._data = [None] * capacity
        self._capacity = capacity
        self._front = 0   # 先頭要素のインデックス
        self._size = 0     # 現在の要素数

    def enqueue(self, value):
        """末尾に追加: O(1)"""
        if self._size == self._capacity:
            self._resize(self._capacity * 2)
        rear = (self._front + self._size) % self._capacity
        self._data[rear] = value
        self._size += 1

    def dequeue(self):
        """先頭から削除: O(1)"""
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        value = self._data[self._front]
        self._data[self._front] = None  # メモリリーク防止
        self._front = (self._front + 1) % self._capacity
        self._size -= 1
        return value

    def peek(self):
        """先頭要素を参照: O(1)"""
        if self.is_empty():
            raise IndexError("peek from empty queue")
        return self._data[self._front]

    def is_empty(self):
        return self._size == 0

    def size(self):
        return self._size

    def _resize(self, new_capacity):
        """配列を拡張し、要素を再配置"""
        new_data = [None] * new_capacity
        for i in range(self._size):
            new_data[i] = self._data[(self._front + i) % self._capacity]
        self._data = new_data
        self._front = 0
        self._capacity = new_capacity

    def __repr__(self):
        items = []
        for i in range(self._size):
            items.append(str(self._data[(self._front + i) % self._capacity]))
        return f"CircularQueue([{', '.join(items)}])"


# 動作確認
if __name__ == "__main__":
    q = CircularQueue(capacity=4)
    q.enqueue(10)
    q.enqueue(20)
    q.enqueue(30)
    print(q)           # CircularQueue([10, 20, 30])
    print(q.dequeue()) # 10
    print(q.dequeue()) # 20
    q.enqueue(40)
    q.enqueue(50)
    print(q)           # CircularQueue([30, 40, 50])
```

**循環バッファの図解**:

```
capacity=6 の循環バッファ:

初期状態: enqueue(A), enqueue(B), enqueue(C)
  ┌───┬───┬───┬───┬───┬───┐
  │ A │ B │ C │   │   │   │
  └───┴───┴───┴───┴───┴───┘
    ^front          ^rear(次の挿入位置)
    size=3

dequeue() → A:
  ┌───┬───┬───┬───┬───┬───┐
  │   │ B │ C │   │   │   │
  └───┴───┴───┴───┴───┴───┘
        ^front      ^rear
    size=2

enqueue(D), enqueue(E), enqueue(F):
  ┌───┬───┬───┬───┬───┬───┐
  │   │ B │ C │ D │ E │ F │
  └───┴───┴───┴───┴───┴───┘
        ^front              ^rear(=0、配列の先頭に巻き戻る)
    size=5

enqueue(G) → rear は 0 の位置に書き込む（循環!）:
  ┌───┬───┬───┬───┬───┬───┐
  │ G │ B │ C │ D │ E │ F │
  └───┴───┴───┴───┴───┴───┘
    ^rear ^front
    size=6 (満杯 → 次の enqueue でリサイズ)
```

### 3.3 連結リストによるキュー実装

```python
class Node:
    __slots__ = ('value', 'next')

    def __init__(self, value, next_node=None):
        self.value = value
        self.next = next_node


class LinkedQueue:
    """
    連結リストベースのキュー実装。
    head が先頭（dequeue 側）、tail が末尾（enqueue 側）。
    """

    def __init__(self):
        self._head = None
        self._tail = None
        self._size = 0

    def enqueue(self, value):
        """末尾に追加: O(1)"""
        new_node = Node(value)
        if self.is_empty():
            self._head = new_node
        else:
            self._tail.next = new_node
        self._tail = new_node
        self._size += 1

    def dequeue(self):
        """先頭から削除: O(1)"""
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        value = self._head.value
        self._head = self._head.next
        if self._head is None:
            self._tail = None  # キューが空になった
        self._size -= 1
        return value

    def peek(self):
        if self.is_empty():
            raise IndexError("peek from empty queue")
        return self._head.value

    def is_empty(self):
        return self._size == 0

    def size(self):
        return self._size


# 動作確認
if __name__ == "__main__":
    q = LinkedQueue()
    q.enqueue(10)
    q.enqueue(20)
    q.enqueue(30)
    print(q.dequeue())  # 10
    print(q.peek())     # 20
    print(q.size())     # 2
```

### 3.4 キュー実装方式の比較

| 観点 | 素朴な配列 | 循環バッファ | 連結リスト |
|------|-----------|-------------|-----------|
| enqueue | O(1) 償却 | O(1) 償却 | O(1) 最悪 |
| dequeue | **O(n)** | O(1) 償却 | O(1) 最悪 |
| メモリ効率 | 高い | 高い | ノード分オーバーヘッド |
| キャッシュ効率 | 高い | 高い | 低い |
| 実装の複雑さ | 最も簡単 | 中程度 | 中程度 |
| 最悪計算量 | O(n) | O(n)（リサイズ時） | O(1) |
| 使うべき場面 | 使わない | 一般的な用途 | リアルタイムシステム |

---

## 4. 両端キュー（Deque）

### 4.1 Deque の概念

Deque（Double-Ended Queue、デック/デキュー）は、両端からの挿入・削除を O(1) で行えるデータ構造である。スタックとキューの両方の機能を兼ね備える。

```
         appendleft        append
              │                │
              v                v
  ┌───┬───┬───┬───┬───┬───┬───┐
  │   │ A │ B │ C │ D │ E │   │
  └───┴───┴───┴───┴───┴───┴───┘
              ^                ^
              │                │
          popleft            pop
```

### 4.2 Python の collections.deque

Python の `collections.deque` は、内部的にブロック連結リスト（doubly-linked list of fixed-size blocks）で実装されており、両端操作が O(1) で行える。

```python
from collections import deque

# 基本操作
dq = deque()

# 右端の操作（スタック的・キュー的）
dq.append(1)       # 右に追加: O(1)
dq.append(2)
dq.append(3)
print(dq)           # deque([1, 2, 3])

val = dq.pop()      # 右から取出: O(1) → 3
print(dq)           # deque([1, 2])

# 左端の操作
dq.appendleft(0)    # 左に追加: O(1)
print(dq)           # deque([0, 1, 2])

val = dq.popleft()  # 左から取出: O(1) → 0
print(dq)           # deque([1, 2])

# 固定長 deque（古い要素を自動で捨てる）
history = deque(maxlen=3)
history.append("page1")
history.append("page2")
history.append("page3")
history.append("page4")  # "page1" が自動的に消える
print(history)      # deque(['page2', 'page3', 'page4'], maxlen=3)

# 回転
dq2 = deque([1, 2, 3, 4, 5])
dq2.rotate(2)       # 右に2つ回転
print(dq2)          # deque([4, 5, 1, 2, 3])
dq2.rotate(-2)      # 左に2つ回転
print(dq2)          # deque([1, 2, 3, 4, 5])
```

**注意点**: `deque` はインデックスアクセス `dq[i]` が O(n) である（内部がブロック連結リストのため）。ランダムアクセスが頻繁なら `list` を使うべきである。

### 4.3 Deque の用途

1. **スライディングウィンドウの最大値/最小値**: 単調 Deque を用いて O(n) で解ける（後述のセクション 6 で詳述）
2. **回文チェック**: 両端から文字を取り出して比較する
3. **仕事の窃取（Work Stealing）**: マルチスレッドのタスクスケジューリングで、自分のキューの片端からタスクを取り、他人のキューの反対端からタスクを盗む
4. **ブラウザの戻る/進む**: 2つのスタックの代わりに1つの Deque で管理可能

---

## 5. 優先度キュー（Priority Queue）

### 5.1 概念と動機

通常のキューは FIFO（先入れ先出し）だが、**優先度キュー** は「最も優先度の高い要素」を最初に取り出す。救急外来のトリアージに似ている。到着順ではなく、重症度の高い患者が先に診察される。

優先度キューの ADT:
- `insert(element, priority)`: 要素を優先度とともに追加する
- `extract_min()` / `extract_max()`: 最も優先度の高い要素を取り出す
- `peek()`: 最も優先度の高い要素を参照する

### 5.2 ヒープによる実装

優先度キューの最も効率的な実装は **二分ヒープ（Binary Heap）** である。

```python
class MinHeap:
    """
    最小ヒープの完全な実装。
    配列で二分木を表現する。

    親: i → 子: 2i+1, 2i+2
    子: i → 親: (i-1)//2
    """

    def __init__(self):
        self._data = []

    def insert(self, value):
        """要素を追加: O(log n)"""
        self._data.append(value)
        self._sift_up(len(self._data) - 1)

    def extract_min(self):
        """最小要素を取り出す: O(log n)"""
        if not self._data:
            raise IndexError("extract from empty heap")
        min_val = self._data[0]
        last = self._data.pop()
        if self._data:
            self._data[0] = last
            self._sift_down(0)
        return min_val

    def peek(self):
        """最小要素を参照: O(1)"""
        if not self._data:
            raise IndexError("peek from empty heap")
        return self._data[0]

    def _sift_up(self, idx):
        """追加した要素を適切な位置まで上に移動"""
        while idx > 0:
            parent = (idx - 1) // 2
            if self._data[idx] < self._data[parent]:
                self._data[idx], self._data[parent] = self._data[parent], self._data[idx]
                idx = parent
            else:
                break

    def _sift_down(self, idx):
        """ルートの要素を適切な位置まで下に移動"""
        n = len(self._data)
        while True:
            smallest = idx
            left = 2 * idx + 1
            right = 2 * idx + 2

            if left < n and self._data[left] < self._data[smallest]:
                smallest = left
            if right < n and self._data[right] < self._data[smallest]:
                smallest = right

            if smallest != idx:
                self._data[idx], self._data[smallest] = self._data[smallest], self._data[idx]
                idx = smallest
            else:
                break

    def size(self):
        return len(self._data)

    def is_empty(self):
        return len(self._data) == 0


# 動作確認
if __name__ == "__main__":
    heap = MinHeap()
    for v in [5, 3, 8, 1, 2, 7]:
        heap.insert(v)

    result = []
    while not heap.is_empty():
        result.append(heap.extract_min())
    print(result)  # [1, 2, 3, 5, 7, 8] — ヒープソート
```

**ヒープの内部構造（図解）**:

```
insert の過程: [5, 3, 8, 1, 2, 7]

(1) insert(5):          (2) insert(3):          (3) insert(8):
       5                      3                       3
                             / \                     / \
                            5                       5   8

(4) insert(1):          (5) insert(2):          (6) insert(7):
       1                      1                       1
      / \                    / \                     / \
     3   8                  2   8                   2   7
    /                      / \                     / \ /
   5                      5   3                   5  3 8

配列表現: [1, 2, 7, 5, 3, 8]

インデックス:  0  1  2  3  4  5
               1  2  7  5  3  8
               │
               └─ 最小値は常にインデックス 0
```

### 5.3 Python の heapq モジュール

```python
import heapq

# 最小ヒープ
heap = []
heapq.heappush(heap, 3)
heapq.heappush(heap, 1)
heapq.heappush(heap, 4)
heapq.heappush(heap, 1)
heapq.heappush(heap, 5)

print(heapq.heappop(heap))  # 1
print(heapq.heappop(heap))  # 1
print(heap[0])               # 3 (peek)

# 最大ヒープ（値を反転して格納）
max_heap = []
for val in [3, 1, 4, 1, 5]:
    heapq.heappush(max_heap, -val)

print(-heapq.heappop(max_heap))  # 5
print(-heapq.heappop(max_heap))  # 4

# n個の最大/最小要素
data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
print(heapq.nlargest(3, data))   # [9, 6, 5]
print(heapq.nsmallest(3, data))  # [1, 1, 2]

# 既存のリストをヒープ化: O(n)
data = [5, 3, 8, 1, 2, 7]
heapq.heapify(data)
print(data)  # [1, 2, 7, 5, 3, 8]（ヒープ性質を満たす）
```

### 5.4 ダイクストラ法との連携

優先度キューの最も有名な応用がダイクストラ法（最短経路アルゴリズム）である。

```python
import heapq
from collections import defaultdict

def dijkstra(graph: dict, start: str) -> dict:
    """
    ダイクストラ法: 単一始点最短経路。
    graph: {node: [(neighbor, weight), ...]}
    戻り値: {node: 最短距離}

    計算量: O((V + E) log V)  （二分ヒープ使用時）
    """
    distances = {start: 0}
    pq = [(0, start)]  # (距離, ノード)
    visited = set()

    while pq:
        dist, node = heapq.heappop(pq)

        if node in visited:
            continue
        visited.add(node)

        for neighbor, weight in graph.get(node, []):
            new_dist = dist + weight
            if neighbor not in distances or new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))

    return distances


# テスト用グラフ
#
#     A ---2--- B ---3--- D
#     |         |         |
#     4         1         1
#     |         |         |
#     C ---5--- E ---2--- F
#
graph = {
    'A': [('B', 2), ('C', 4)],
    'B': [('A', 2), ('D', 3), ('E', 1)],
    'C': [('A', 4), ('E', 5)],
    'D': [('B', 3), ('F', 1)],
    'E': [('B', 1), ('C', 5), ('F', 2)],
    'F': [('D', 1), ('E', 2)],
}

result = dijkstra(graph, 'A')
print(result)
# {'A': 0, 'B': 2, 'E': 3, 'C': 4, 'D': 5, 'F': 5}
```

### 5.5 Top-K 問題

```python
import heapq
from collections import Counter

def top_k_frequent(nums: list, k: int) -> list:
    """
    出現頻度が高い上位 K 個の要素を返す。

    方法: サイズ K のミニヒープを維持する。
    計算量: O(n log k) — 全ソート O(n log n) より効率的（k << n の場合）
    """
    count = Counter(nums)
    return heapq.nlargest(k, count.keys(), key=count.get)


# テスト
nums = [1, 1, 1, 2, 2, 3, 3, 3, 3, 4]
print(top_k_frequent(nums, 2))  # [3, 1] — 3が4回、1が3回
```

### 5.6 優先度キューの操作計算量

| 操作 | 二分ヒープ | ソート済み配列 | 未ソート配列 | フィボナッチヒープ |
|------|-----------|-------------|-------------|----------------|
| insert | O(log n) | O(n) | O(1) | O(1) 償却 |
| extract_min | O(log n) | O(1) | O(n) | O(log n) 償却 |
| peek | O(1) | O(1) | O(n) | O(1) |
| decrease_key | O(log n) | O(n) | O(1) | O(1) 償却 |
| 構築 | O(n) | O(n log n) | O(n) | O(n) |

実用上は、二分ヒープが最もバランスが良く、実装も比較的容易であるため広く使われている。フィボナッチヒープは理論的に優れるが、定数倍が大きく実装も複雑なため、実務ではほとんど使われない。

---

## 6. 単調スタック・単調キュー

### 6.1 単調スタック（Monotonic Stack）

単調スタックは、スタック内の要素が常に単調増加（または単調減少）になるように管理するテクニックである。「各要素について、右（左）にある最初の大きい（小さい）要素を見つける」問題を O(n) で解ける。

```python
def next_greater_element(arr: list) -> list:
    """
    各要素について、右側にある最初の「より大きい要素」を求める。
    見つからなければ -1。

    例: [4, 2, 3, 5, 1] → [5, 3, 5, -1, -1]

    アルゴリズム:
    - スタックにインデックスを格納
    - スタックのトップより大きい要素が来たら、それがトップの NGE
    - 各要素は最大1回 push、1回 pop されるため O(n)
    """
    n = len(arr)
    result = [-1] * n
    stack = []  # インデックスのスタック（値は単調減少に維持）

    for i in range(n):
        # 現在の要素がスタックのトップより大きい間、答えを確定
        while stack and arr[stack[-1]] < arr[i]:
            idx = stack.pop()
            result[idx] = arr[i]
        stack.append(i)

    return result


# テスト
print(next_greater_element([4, 2, 3, 5, 1]))
# [5, 3, 5, -1, -1]

print(next_greater_element([1, 3, 2, 4]))
# [3, 4, 4, -1]
```

**処理の流れ（arr = [4, 2, 3, 5, 1]）**:

```
i=0, arr[0]=4: stack=[]     → push 0       stack=[0]
i=1, arr[1]=2: 2<4          → push 1       stack=[0,1]
i=2, arr[2]=3: 3>arr[1]=2   → pop 1, result[1]=3
               3<arr[0]=4   → push 2       stack=[0,2]
i=3, arr[3]=5: 5>arr[2]=3   → pop 2, result[2]=5
               5>arr[0]=4   → pop 0, result[0]=5
                             → push 3       stack=[3]
i=4, arr[4]=1: 1<arr[3]=5   → push 4       stack=[3,4]

残ったインデックス 3,4 は -1 のまま
result = [5, 3, 5, -1, -1]
```

### 6.2 単調スタックの応用: ヒストグラムの最大長方形

LeetCode 84 "Largest Rectangle in Histogram" は単調スタックの代表的な応用問題である。

```python
def largest_rectangle_area(heights: list) -> int:
    """
    ヒストグラムにおける最大長方形の面積を求める。

    アルゴリズム: 単調増加スタック
    - 各バーについて、そのバーを高さとした長方形がどこまで
      左右に伸びるかを効率的に求める
    - スタックには単調増加のインデックスを維持
    - 現在のバーがスタックのトップより低い場合、トップを pop して
      面積を計算する

    計算量: O(n)
    """
    stack = []  # インデックスの単調増加スタック
    max_area = 0
    # 番兵として末尾に 0 を追加
    heights_ext = heights + [0]

    for i, h in enumerate(heights_ext):
        while stack and heights_ext[stack[-1]] > h:
            height = heights_ext[stack.pop()]
            # 幅: スタックが空なら左端から i まで、そうでなければ stack[-1]+1 から i まで
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)

    return max_area


# テスト
print(largest_rectangle_area([2, 1, 5, 6, 2, 3]))  # 10 (5x2)
print(largest_rectangle_area([2, 4]))               # 4
```

### 6.3 単調キュー（Monotonic Deque）

単調キュー（単調 Deque）は、スライディングウィンドウの最大値/最小値を O(n) で求めるテクニックである。

```python
from collections import deque

def max_sliding_window(nums: list, k: int) -> list:
    """
    サイズ k のスライディングウィンドウの最大値を求める。

    アルゴリズム:
    - Deque にインデックスを格納し、値が単調減少になるよう維持
    - Deque の先頭が常にウィンドウ内の最大値のインデックス
    - ウィンドウから外れたインデックスは先頭から除去
    - 現在の値より小さい値は末尾から除去（もう最大値にならない）

    計算量: O(n) 時間、O(k) 空間
    """
    if not nums or k == 0:
        return []

    dq = deque()  # インデックスを格納（値は単調減少）
    result = []

    for i in range(len(nums)):
        # ウィンドウから外れたインデックスを先頭から除去
        while dq and dq[0] < i - k + 1:
            dq.popleft()

        # 現在の値より小さい値を末尾から除去
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()

        dq.append(i)

        # ウィンドウが完成したら結果に追加
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result


# テスト
print(max_sliding_window([1, 3, -1, -3, 5, 3, 6, 7], 3))
# [3, 3, 5, 5, 6, 7]
```

**処理の流れ（nums=[1,3,-1,-3,5,3,6,7], k=3）**:

```
i=0, num=1:  dq=[0]                        ウィンドウ未完成
i=1, num=3:  3>1 → pop 0, dq=[1]           ウィンドウ未完成
i=2, num=-1: -1<3 → dq=[1,2]               result=[3]  (nums[1]=3)
i=3, num=-3: -3<-1 → dq=[1,2,3]            result=[3,3]
i=4, num=5:  5>-3 → pop 3; 5>-1 → pop 2;
             5>3 → pop 1; dq=[4]           result=[3,3,5]
i=5, num=3:  3<5 → dq=[4,5]                result=[3,3,5,5]
i=6, num=6:  6>3 → pop 5; 6>5 → pop 4;
             dq=[6]                         result=[3,3,5,5,6]
i=7, num=7:  7>6 → pop 6; dq=[7]           result=[3,3,5,5,6,7]
```

---

## 7. 実務応用

### 7.1 Undo/Redo 機能

テキストエディタの Undo/Redo は、2つのスタックで実装できる。

```python
class UndoRedoEditor:
    """
    2つのスタックによる Undo/Redo の実装。

    undo_stack: 実行した操作の履歴
    redo_stack: Undo した操作の履歴（Redo 用）

    新しい操作を実行すると redo_stack はクリアされる。
    """

    def __init__(self):
        self.text = ""
        self.undo_stack = []
        self.redo_stack = []

    def type_text(self, new_text: str):
        """テキストを入力する"""
        self.undo_stack.append(self.text)
        self.redo_stack.clear()  # 新操作で Redo 履歴をクリア
        self.text = new_text

    def undo(self):
        """直前の状態に戻す"""
        if not self.undo_stack:
            print("Nothing to undo")
            return
        self.redo_stack.append(self.text)
        self.text = self.undo_stack.pop()

    def redo(self):
        """Undo を取り消す"""
        if not self.redo_stack:
            print("Nothing to redo")
            return
        self.undo_stack.append(self.text)
        self.text = self.redo_stack.pop()


# デモ
editor = UndoRedoEditor()
editor.type_text("Hello")
editor.type_text("Hello World")
editor.type_text("Hello World!")
print(editor.text)   # "Hello World!"
editor.undo()
print(editor.text)   # "Hello World"
editor.undo()
print(editor.text)   # "Hello"
editor.redo()
print(editor.text)   # "Hello World"
```

### 7.2 BFS（幅優先探索）

キューは BFS の中核データ構造である。

```python
from collections import deque

def bfs(graph: dict, start: str) -> list:
    """
    幅優先探索。グラフの頂点を距離順（レベル順）に走査する。

    graph: {node: [neighbors]}
    戻り値: 訪問順のリスト

    計算量: O(V + E)
    """
    visited = {start}
    queue = deque([start])
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return order


# テスト
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E'],
}
print(bfs(graph, 'A'))  # ['A', 'B', 'C', 'D', 'E', 'F']
```

```
BFS の探索順序（グラフの図解）:

        A (レベル 0)
       / \
      B   C (レベル 1)
     / \   \
    D   E   F (レベル 2)
         \ /
    (E-F は辺で接続)

キューの変化:
  初期:    [A]
  A 処理:  [B, C]           → A を訪問
  B 処理:  [C, D, E]        → B を訪問
  C 処理:  [D, E, F]        → C を訪問
  D 処理:  [E, F]           → D を訪問
  E 処理:  [F]              → E を訪問（F は既に visited）
  F 処理:  []               → F を訪問
```

### 7.3 DFS（深さ優先探索）— スタックによる反復実装

再帰の代わりにスタックを使って DFS を実装できる。再帰が深い場合のスタックオーバーフローを回避できる。

```python
def dfs_iterative(graph: dict, start: str) -> list:
    """
    スタックを使った反復的 DFS。

    注意: 再帰 DFS とは訪問順が微妙に異なる場合がある
    （隣接ノードの処理順が逆になるため）。

    計算量: O(V + E)
    """
    visited = set()
    stack = [start]
    order = []

    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)

        # 逆順で push すると、元の順序で処理される
        for neighbor in reversed(graph.get(node, [])):
            if neighbor not in visited:
                stack.append(neighbor)

    return order


graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E'],
}
print(dfs_iterative(graph, 'A'))  # ['A', 'B', 'D', 'E', 'F', 'C']
```

### 7.4 タスクスケジューラ

```python
import heapq
from collections import Counter

def least_interval(tasks: list, n: int) -> int:
    """
    タスクスケジューラ: 同種のタスク間に最低 n インターバル必要。
    最短の総実行時間を求める。

    例: tasks=["A","A","A","B","B","B"], n=2
    → A B _ A B _ A B → 8

    アルゴリズム:
    1. 最大ヒープで残りカウントが多いタスクから処理
    2. クールダウン中のタスクは待機キューに入れる
    3. 待機キューからクールダウンが終わったタスクをヒープに戻す
    """
    count = Counter(tasks)
    max_heap = [-cnt for cnt in count.values()]
    heapq.heapify(max_heap)

    time = 0
    cooldown = []  # (再利用可能な時刻, 残りカウント)

    while max_heap or cooldown:
        time += 1

        if max_heap:
            cnt = heapq.heappop(max_heap) + 1  # 1つ消費（負数なので +1）
            if cnt < 0:  # まだ残っている
                cooldown.append((time + n, cnt))

        if cooldown and cooldown[0][0] == time:
            _, cnt = cooldown.pop(0)
            heapq.heappush(max_heap, cnt)

    return time


# テスト
print(least_interval(["A","A","A","B","B","B"], 2))  # 8
print(least_interval(["A","A","A","B","B","B"], 0))  # 6
```

### 7.5 メッセージキューの概念

実務のメッセージキュー（RabbitMQ, Apache Kafka, Amazon SQS 等）は、キューの概念を分散システムに拡張したものである。

```
プロデューサ                     コンシューマ
┌──────────┐                   ┌──────────┐
│ Web App  │──enqueue──>       │ Worker 1 │
└──────────┘            │      └──────────┘
                        v          ^
┌──────────┐    ┌───────────┐     │
│ API Srv  │──> │ Message   │──dequeue──>
└──────────┘    │ Queue     │     │
                │ (Broker)  │     v
┌──────────┐    └───────────┘  ┌──────────┐
│ Cron Job │──enqueue──>       │ Worker 2 │
└──────────┘                   └──────────┘

特徴:
- 非同期処理: プロデューサはコンシューマの完了を待たない
- 負荷分散: 複数のコンシューマが並行してメッセージを処理
- 信頼性: キューがメッセージを永続化し、コンシューマ障害時に再配送
- スケーラビリティ: コンシューマを追加するだけで処理能力を向上
```

---

## 8. 各言語の標準ライブラリ実装

### 8.1 言語別スタック/キュー/優先度キュー

| 言語 | スタック | キュー | Deque | 優先度キュー |
|------|---------|-------|-------|-------------|
| Python | `list` (append/pop) | `collections.deque` | `collections.deque` | `heapq` |
| Java | `ArrayDeque` | `ArrayDeque` / `LinkedList` | `ArrayDeque` | `PriorityQueue` |
| C++ | `std::stack` | `std::queue` | `std::deque` | `std::priority_queue` |
| Go | スライス (append/pop) | `container/list` | なし（自作） | `container/heap` |
| Rust | `Vec` (push/pop) | `VecDeque` | `VecDeque` | `BinaryHeap` |
| JavaScript | `Array` (push/pop) | なし（自作） | なし（自作） | なし（自作） |
| C# | `Stack<T>` | `Queue<T>` | `LinkedList<T>` | `PriorityQueue<T,P>` (.NET 6+) |

### 8.2 Java での使用例

```java
import java.util.*;

public class StackQueueExample {
    public static void main(String[] args) {
        // スタック: ArrayDeque を推奨（Stack クラスは旧式で同期コストあり）
        Deque<Integer> stack = new ArrayDeque<>();
        stack.push(10);
        stack.push(20);
        stack.push(30);
        System.out.println(stack.peek()); // 30
        System.out.println(stack.pop());  // 30

        // キュー: ArrayDeque を推奨
        Queue<Integer> queue = new ArrayDeque<>();
        queue.offer(10);
        queue.offer(20);
        queue.offer(30);
        System.out.println(queue.peek()); // 10
        System.out.println(queue.poll()); // 10

        // 優先度キュー（最小ヒープ）
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        pq.offer(5);
        pq.offer(1);
        pq.offer(3);
        System.out.println(pq.poll()); // 1

        // 優先度キュー（最大ヒープ）
        PriorityQueue<Integer> maxPq = new PriorityQueue<>(
            Comparator.reverseOrder()
        );
        maxPq.offer(5);
        maxPq.offer(1);
        maxPq.offer(3);
        System.out.println(maxPq.poll()); // 5
    }
}
```

### 8.3 C++ での使用例

```cpp
#include <iostream>
#include <stack>
#include <queue>
#include <deque>

int main() {
    // スタック（デフォルトは deque ベース）
    std::stack<int> st;
    st.push(10);
    st.push(20);
    st.push(30);
    std::cout << st.top() << std::endl;  // 30
    st.pop();  // void を返す（値は返さない）

    // キュー（デフォルトは deque ベース）
    std::queue<int> q;
    q.push(10);
    q.push(20);
    q.push(30);
    std::cout << q.front() << std::endl;  // 10
    q.pop();

    // 優先度キュー（デフォルトは最大ヒープ）
    std::priority_queue<int> pq;
    pq.push(5);
    pq.push(1);
    pq.push(3);
    std::cout << pq.top() << std::endl;  // 5
    pq.pop();

    // 最小ヒープ
    std::priority_queue<int, std::vector<int>, std::greater<int>> min_pq;
    min_pq.push(5);
    min_pq.push(1);
    min_pq.push(3);
    std::cout << min_pq.top() << std::endl;  // 1

    return 0;
}
```

---

## 9. 内部実装の仕組み

### 9.1 CPython の list

CPython の `list` は、**ポインタの動的配列** として実装されている。

```
CPython list の内部構造:

PyListObject:
┌──────────────────────┐
│ ob_refcnt             │  参照カウント
│ ob_type               │  型オブジェクトへのポインタ
│ ob_size (Py_ssize_t)  │  現在の要素数
│ allocated             │  確保済みスロット数
│ ob_item (PyObject**)  │──────> ┌──────────┐
└──────────────────────┘        │ ptr[0]   │──> PyObject (要素0)
                                │ ptr[1]   │──> PyObject (要素1)
                                │ ptr[2]   │──> PyObject (要素2)
                                │ ...      │
                                │ (空き)   │
                                └──────────┘

リサイズ戦略:
- 新しいサイズ = (現在のサイズ + 現在のサイズ >> 3) + (3 if 現在のサイズ < 9 else 6)
- 概ね 1.125 倍ずつ成長（リストが小さいときは大きめに成長）
- これにより append() の償却計算量が O(1) になる
```

`list.pop()` が O(1) なのは、末尾のポインタを無効化し `ob_size` を 1 減らすだけで済むためである。一方 `list.pop(0)` が O(n) なのは、残りの全ポインタを1つ前にシフトする必要があるためである。

### 9.2 CPython の collections.deque

`collections.deque` は、**固定サイズのブロック（64要素）の双方向連結リスト** として実装されている。

```
deque の内部構造:

dequeobject:
┌────────────────────┐
│ leftblock          │──> ┌─────────────────────┐
│ rightblock         │    │ Block (64 slots)    │
│ leftindex          │    │ ┌───┬───┬...┬───┐   │
│ rightindex         │    │ │   │ A │   │ B │   │ ←── leftindex, rightindex
│ length             │    │ └───┴───┴...┴───┘   │
│ maxlen             │    │ prev ──> NULL        │
└────────────────────┘    │ next ──> Block2      │
                          └─────────────────────┘
                                    │
                                    v
                          ┌─────────────────────┐
                          │ Block (64 slots)    │
                          │ ┌───┬───┬...┬───┐   │
                          │ │ C │ D │   │   │   │
                          │ └───┴───┴...┴───┘   │
                          │ prev ──> Block1      │
                          │ next ──> NULL        │
                          └─────────────────────┘

- 各ブロックは 64 個のスロットを持つ固定サイズ配列
- ブロック間は双方向連結リストでつながる
- appendleft/popleft は leftblock の leftindex を操作
- append/pop は rightblock の rightindex を操作
- インデックスアクセス dq[i] は O(n/64) = O(n)（ブロックを辿る必要がある）
```

この設計により、両端操作は O(1) でありながら、ブロック内はキャッシュフレンドリーな連続メモリである。純粋な連結リストより高速に動作する。

### 9.3 Java の ArrayDeque

Java の `ArrayDeque` は **循環バッファ** で実装されている。

```
ArrayDeque の内部構造:

┌───────────────────────────────────┐
│ Object[] elements (2のべき乗サイズ) │
│                                   │
│  [  ] [E] [F] [G] [  ] [  ] [C] [D]
│   0    1   2   3   4    5    6   7
│                              ^head
│              ^tail(次の挿入位置)
│                                   │
│ head = 6  (先頭要素のインデックス)   │
│ tail = 4  (次に書き込むインデックス) │
└───────────────────────────────────┘

キューとして使用:
  offer(x) → elements[tail] = x; tail = (tail+1) & (len-1)
  poll()   → val = elements[head]; head = (head+1) & (len-1)

スタックとして使用:
  push(x) → head = (head-1) & (len-1); elements[head] = x
  pop()   → val = elements[head]; head = (head+1) & (len-1)

- サイズは常に 2 のべき乗 → ビット AND でモジュロ演算を高速化
- リサイズ時は 2 倍に拡張
- null 要素は許可されない（null を番兵として使用するため）
```

Java では `Stack` クラスは `Vector` を継承しているため、すべてのメソッドが `synchronized`（スレッドセーフだがオーバーヘッドあり）である。シングルスレッド環境では `ArrayDeque` を使うのが推奨される。

---

## 10. トレードオフと比較分析

### 10.1 スタック vs キューの使い分け

| 基準 | スタック（LIFO） | キュー（FIFO） |
|------|--------------|--------------|
| データ順序 | 後入れ先出し | 先入れ先出し |
| 探索アルゴリズム | DFS（深さ優先） | BFS（幅優先） |
| 最短経路 | 保証しない | 重みなしグラフで最短経路を保証 |
| メモリ使用量（探索時） | O(最大深さ) | O(最大幅) |
| 典型的な用途 | 式の評価、括弧チェック、Undo | タスク処理、レベル走査、最短路 |
| 再帰との関係 | 再帰は暗黙のスタック | 再帰では自然に表現しにくい |
| 空間効率（グラフ探索） | 深いが狭い木で有利 | 幅広い木で有利 |

### 10.2 いつ何を使うか — 判断フローチャート

```
データを格納・取り出したい
│
├─ 最後に入れたものを最初に取り出す？
│  └─ YES → スタック
│     ├─ 括弧のマッチング
│     ├─ DFS
│     ├─ Undo/Redo
│     └─ 式の評価（後置記法）
│
├─ 最初に入れたものを最初に取り出す？
│  └─ YES → キュー
│     ├─ BFS
│     ├─ タスクスケジューリング
│     └─ メッセージ処理
│
├─ 両端から出し入れしたい？
│  └─ YES → Deque
│     ├─ スライディングウィンドウ
│     └─ スタック+キューの機能が必要
│
├─ 優先度に基づいて取り出したい？
│  └─ YES → 優先度キュー（ヒープ）
│     ├─ 最短経路（ダイクストラ）
│     ├─ Top-K 問題
│     └─ イベントシミュレーション
│
└─ ランダムアクセスも必要？
   └─ YES → 配列/リストを使うべき（スタック/キューは不適切）
```

### 10.3 データ構造の総合比較表

| データ構造 | push/enqueue | pop/dequeue | peek | ランダムアクセス | 主な実装 | メモリ |
|-----------|-------------|-------------|------|--------------|---------|-------|
| スタック (配列) | O(1)* | O(1)* | O(1) | O(1) | 動的配列 | 連続 |
| スタック (連結リスト) | O(1) | O(1) | O(1) | O(n) | 単方向リスト | 分散 |
| キュー (循環バッファ) | O(1)* | O(1)* | O(1) | O(1) | 固定配列 | 連続 |
| キュー (連結リスト) | O(1) | O(1) | O(1) | O(n) | 双方向リスト | 分散 |
| Deque (ブロックリスト) | O(1) | O(1) | O(1) | O(n) | ブロック連結 | 半連続 |
| 優先度キュー (二分ヒープ) | O(log n) | O(log n) | O(1) | - | 配列 | 連続 |

\* 償却計算量

---

## 11. アンチパターン

### 11.1 アンチパターン1: list.pop(0) でキューを実装する

```python
# NG: O(n) の dequeue
queue = [1, 2, 3, 4, 5]
val = queue.pop(0)  # O(n) — 全要素のシフトが発生

# 10万要素で pop(0) を繰り返すと:
# list.pop(0): 約 3.5 秒
# deque.popleft(): 約 0.01 秒
# 350倍の速度差!
```

**正しいアプローチ**:

```python
from collections import deque

queue = deque([1, 2, 3, 4, 5])
val = queue.popleft()  # O(1)
```

**なぜこのバグが起きるか**: Python 入門書やチュートリアルで「list でキューを作れます」と紹介されることがあるため。小規模データでは速度差が目立たないが、数万件を超えると致命的なパフォーマンス問題になる。

### 11.2 アンチパターン2: 再帰を深くしすぎてスタックオーバーフロー

```python
# NG: 深い再帰で RecursionError
def sum_list(lst, index=0):
    if index == len(lst):
        return 0
    return lst[index] + sum_list(lst, index + 1)

# 1000 要素以下なら動くが、10000 要素で RecursionError
# sum_list(list(range(10000)))  # RecursionError!
```

**正しいアプローチ**:

```python
# 方法1: 反復に書き換える
def sum_list_iterative(lst):
    total = 0
    for val in lst:
        total += val
    return total

# 方法2: 明示的なスタックを使う（複雑な再帰の場合）
def sum_list_stack(lst):
    stack = list(range(len(lst)))  # インデックスをスタックに
    total = 0
    while stack:
        i = stack.pop()
        total += lst[i]
    return total

# 方法3: Python の組み込み関数を使う
total = sum(range(10000))
```

**教訓**: Python はデフォルトの再帰上限が 1000 であり、末尾再帰の最適化も行わない。深い再帰が想定される場合は、反復や明示的なスタックに書き換えるべきである。

### 11.3 アンチパターン3: Java の Stack クラスを使う

```java
// NG: Stack は Vector を継承しており、不要な同期コストがかかる
Stack<Integer> stack = new Stack<>();
stack.push(1);       // synchronized — シングルスレッドでも同期コスト発生
int val = stack.pop();

// さらに、Stack は Vector のメソッド（get(i), set(i, v) 等）を
// 継承しているため、スタック以外の操作ができてしまう
stack.add(0, 999);   // スタックの「底」に要素を挿入できてしまう（意味がない）
```

**正しいアプローチ**:

```java
// OK: ArrayDeque をスタックとして使う
Deque<Integer> stack = new ArrayDeque<>();
stack.push(1);
int val = stack.pop();
// ArrayDeque はスレッドセーフではないが、シングルスレッドでは高速
// また、任意位置へのアクセスメソッドを持たないため、抽象データ型として適切
```

### 11.4 アンチパターン4: 空のスタック/キューから pop する

```python
# NG: チェックなしの pop
stack = []
val = stack.pop()  # IndexError: pop from empty list

# NG: キューでも同様
from collections import deque
queue = deque()
val = queue.popleft()  # IndexError: pop from an empty deque
```

**正しいアプローチ**:

```python
# 方法1: 事前チェック
if stack:
    val = stack.pop()

# 方法2: 例外処理
try:
    val = stack.pop()
except IndexError:
    val = None  # デフォルト値

# 方法3: ラッパークラスで安全な操作を提供
class SafeStack:
    def __init__(self):
        self._data = []

    def pop(self, default=None):
        return self._data.pop() if self._data else default
```

---

## 12. エッジケース分析

### 12.1 エッジケース1: スタックに1要素しかない場合の操作

```python
stack = [42]

# peek と pop が同じ値を返すことを確認
assert stack[-1] == 42   # peek
val = stack.pop()         # pop
assert val == 42
assert len(stack) == 0    # 空になっている

# 空になった後の peek/pop はエラーになるべき
# stack[-1]  → IndexError
# stack.pop() → IndexError
```

括弧検証で問題になるケース:

```python
# 閉じ括弧だけの入力
assert is_valid_parentheses(")") == False   # スタックが空の状態で pop を試みる
assert is_valid_parentheses("(") == False   # 開き括弧が残ったまま終了

# 空文字列
assert is_valid_parentheses("") == True     # 空は valid とみなす
```

### 12.2 エッジケース2: 循環バッファの容量境界

```python
# 容量 1 のキュー
q = CircularQueue(capacity=1)
q.enqueue(10)
# q.enqueue(20)  → リサイズが発生する
print(q.dequeue())  # 10

# 容量ちょうどまで埋めてから dequeue/enqueue を繰り返す
q2 = CircularQueue(capacity=4)
for i in range(4):
    q2.enqueue(i)
# 内部状態: [0, 1, 2, 3], front=0, size=4

q2.dequeue()  # 0 を取り出し → front=1, size=3
q2.dequeue()  # 1 を取り出し → front=2, size=2

q2.enqueue(4)  # rear = (2+2) % 4 = 0 に書き込み → 循環!
q2.enqueue(5)  # rear = (2+3) % 4 = 1 に書き込み

# 内部状態: [4, 5, 2, 3], front=2, size=4
# 論理的な順序: 2, 3, 4, 5
```

### 12.3 エッジケース3: ヒープに同じ優先度の要素がある場合

```python
import heapq

# 同じ優先度の要素があるとき、Python の heapq はタプルの2番目以降を比較する
# 比較不可能なオブジェクトだとエラーになる

class Task:
    def __init__(self, name):
        self.name = name

# NG: Task オブジェクトは比較不可能
pq = []
heapq.heappush(pq, (1, Task("A")))
heapq.heappush(pq, (1, Task("B")))  # TypeError: '<' not supported

# 正しいアプローチ: 一意なカウンタを挟む
counter = 0
pq = []

def push_task(priority, task):
    global counter
    heapq.heappush(pq, (priority, counter, task))
    counter += 1

push_task(1, Task("A"))
push_task(1, Task("B"))  # 同じ優先度でもカウンタで区別される（FIFO 順）
priority, _, task = heapq.heappop(pq)
print(task.name)  # "A"（先に入れた方）
```

### 12.4 エッジケース4: 2つのスタックでキューを実装する際の償却分析

```python
class QueueFromStacks:
    """
    2つのスタックでキューを実装する。

    エッジケース: dequeue 時に in_stack から out_stack への
    移し替えが発生する。最悪 O(n) だが償却 O(1)。
    """

    def __init__(self):
        self.in_stack = []   # enqueue 用
        self.out_stack = []  # dequeue 用

    def enqueue(self, value):
        """常に O(1)"""
        self.in_stack.append(value)

    def dequeue(self):
        """償却 O(1)、最悪 O(n)"""
        if not self.out_stack:
            if not self.in_stack:
                raise IndexError("dequeue from empty queue")
            # in_stack の全要素を out_stack に移す（順序が反転する）
            while self.in_stack:
                self.out_stack.append(self.in_stack.pop())
        return self.out_stack.pop()

    def peek(self):
        if not self.out_stack:
            if not self.in_stack:
                raise IndexError("peek from empty queue")
            while self.in_stack:
                self.out_stack.append(self.in_stack.pop())
        return self.out_stack[-1]


# エッジケースのテスト
q = QueueFromStacks()

# 1要素だけの場合
q.enqueue(1)
assert q.dequeue() == 1

# enqueue と dequeue を交互に
q.enqueue(1)
q.enqueue(2)
assert q.dequeue() == 1  # in_stack → out_stack 移し替え発生
q.enqueue(3)
assert q.dequeue() == 2  # out_stack にまだ 2 がある
assert q.dequeue() == 3  # in_stack → out_stack 移し替え発生
```

**償却計算量の証明**: 各要素は、in_stack に1回 push され、in_stack から1回 pop され、out_stack に1回 push され、out_stack から1回 pop される。合計4回の操作が各要素に対して行われるため、n 回の enqueue/dequeue 全体で O(4n) = O(n)。よって1回あたりの償却計算量は O(1)。

---

## 13. 演習問題

### 演習1（基礎）: 逆ポーランド記法の評価

**問題**: 逆ポーランド記法（後置記法）の式を評価する関数を実装せよ。演算子は `+`, `-`, `*`, `/` の4種類。除算はゼロ方向への切り捨て。

**入力例**: `["2", "1", "+", "3", "*"]`
**出力例**: `9` （(2+1)*3）

**ヒント**: セクション 2.6 を参照。数値ならスタックに push、演算子なら2つ pop して計算して push。

<details>
<summary>解答例</summary>

```python
def eval_rpn(tokens: list) -> int:
    stack = []
    for token in tokens:
        if token in {'+', '-', '*', '/'}:
            b, a = stack.pop(), stack.pop()
            if token == '+': stack.append(a + b)
            elif token == '-': stack.append(a - b)
            elif token == '*': stack.append(a * b)
            elif token == '/': stack.append(int(a / b))
        else:
            stack.append(int(token))
    return stack[0]

# テスト
assert eval_rpn(["2", "1", "+", "3", "*"]) == 9
assert eval_rpn(["4", "13", "5", "/", "+"]) == 6
assert eval_rpn(["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]) == 22
```

</details>

### 演習2（応用）: 2つのスタックでキューを実装

**問題**: スタック（push/pop のみ）を2つだけ使ってキューを実装せよ。enqueue と dequeue の償却計算量がそれぞれ O(1) であることを説明せよ。

**要件**:
- `enqueue(x)`: 要素 x をキューに追加
- `dequeue()`: 先頭の要素を削除して返す
- `peek()`: 先頭の要素を参照
- `is_empty()`: 空かどうか

<details>
<summary>解答例</summary>

```python
class MyQueue:
    def __init__(self):
        self.in_stack = []
        self.out_stack = []

    def enqueue(self, x):
        self.in_stack.append(x)

    def dequeue(self):
        self._move_if_needed()
        return self.out_stack.pop()

    def peek(self):
        self._move_if_needed()
        return self.out_stack[-1]

    def is_empty(self):
        return not self.in_stack and not self.out_stack

    def _move_if_needed(self):
        if not self.out_stack:
            while self.in_stack:
                self.out_stack.append(self.in_stack.pop())

# 償却分析:
# 各要素は in_stack に1回 push、1回 pop、out_stack に1回 push、1回 pop
# 計4回の O(1) 操作 → 1要素あたり O(1) → 償却 O(1)

q = MyQueue()
q.enqueue(1)
q.enqueue(2)
q.enqueue(3)
assert q.dequeue() == 1
assert q.peek() == 2
q.enqueue(4)
assert q.dequeue() == 2
assert q.dequeue() == 3
assert q.dequeue() == 4
assert q.is_empty()
```

</details>

### 演習3（発展）: データストリームの中央値

**問題**: データストリームからリアルタイムで中央値を求めるクラスを実装せよ。2つのヒープ（最大ヒープと最小ヒープ）を使用すること。

**要件**:
- `add_num(num)`: 数値を追加する
- `find_median()`: 現在の中央値を返す

**制約**: `add_num` は O(log n)、`find_median` は O(1) で実現すること。

<details>
<summary>解答例</summary>

```python
import heapq

class MedianFinder:
    """
    2つのヒープで中央値を管理する。

    max_heap: 小さい方の半分（最大ヒープ、値を反転して格納）
    min_heap: 大きい方の半分（最小ヒープ）

    不変条件:
    1. max_heap の全要素 <= min_heap の全要素
    2. len(max_heap) == len(min_heap) or len(max_heap) == len(min_heap) + 1

    中央値:
    - 要素数が奇数: max_heap のトップ
    - 要素数が偶数: (max_heap のトップ + min_heap のトップ) / 2
    """

    def __init__(self):
        self.max_heap = []  # 小さい方の半分（値を反転）
        self.min_heap = []  # 大きい方の半分

    def add_num(self, num: int):
        # まず max_heap に追加
        heapq.heappush(self.max_heap, -num)

        # max_heap のトップが min_heap のトップより大きければ移す
        if self.min_heap and (-self.max_heap[0]) > self.min_heap[0]:
            val = -heapq.heappop(self.max_heap)
            heapq.heappush(self.min_heap, val)

        # サイズのバランスを維持（max_heap が最大1つ多い）
        if len(self.max_heap) > len(self.min_heap) + 1:
            val = -heapq.heappop(self.max_heap)
            heapq.heappush(self.min_heap, val)
        elif len(self.min_heap) > len(self.max_heap):
            val = heapq.heappop(self.min_heap)
            heapq.heappush(self.max_heap, -val)

    def find_median(self) -> float:
        if len(self.max_heap) > len(self.min_heap):
            return -self.max_heap[0]
        return (-self.max_heap[0] + self.min_heap[0]) / 2


# テスト
mf = MedianFinder()
mf.add_num(1)
assert mf.find_median() == 1
mf.add_num(2)
assert mf.find_median() == 1.5
mf.add_num(3)
assert mf.find_median() == 2
mf.add_num(4)
assert mf.find_median() == 2.5
mf.add_num(5)
assert mf.find_median() == 3
print("All median tests passed!")
```

</details>

### 演習4（発展）: 最小値取得 O(1) のスタック

**問題**: push、pop、top に加えて、O(1) でスタック内の最小値を取得できるスタック（MinStack）を実装せよ。

<details>
<summary>解答例</summary>

```python
class MinStack:
    """
    各操作が O(1) のスタック。
    メインスタックと並行して、各時点での最小値を記録する
    補助スタックを管理する。
    """

    def __init__(self):
        self.stack = []
        self.min_stack = []  # 各時点での最小値を記録

    def push(self, val):
        self.stack.append(val)
        # min_stack のトップと比較して小さい方を push
        current_min = min(val, self.min_stack[-1] if self.min_stack else val)
        self.min_stack.append(current_min)

    def pop(self):
        self.stack.pop()
        self.min_stack.pop()

    def top(self):
        return self.stack[-1]

    def get_min(self):
        return self.min_stack[-1]


# テスト
ms = MinStack()
ms.push(5)
ms.push(3)
ms.push(7)
assert ms.get_min() == 3
ms.pop()  # 7 を削除
assert ms.get_min() == 3
ms.pop()  # 3 を削除
assert ms.get_min() == 5
print("MinStack tests passed!")
```

</details>

---

## 14. FAQ

### Q1: Python で list をスタックとして使って問題ないのか?

**A**: はい、問題ありません。Python の `list.append()` と `list.pop()` はどちらも償却 O(1) であり、スタック操作として適切です。ただし以下の点に注意してください。

1. **pop(0) は使わない**: 先頭からの削除は O(n) です。キューが必要なら `collections.deque` を使いましょう。
2. **メモリの縮小**: `pop()` で要素を減らしても、内部の配列サイズはすぐには縮小されません。メモリが重要な場面では注意が必要です。
3. **スレッドセーフティ**: `list` はスレッドセーフではありません。マルチスレッド環境では `queue.LifoQueue` を使いましょう。

### Q2: deque と list、どちらを使うべきか?

**A**: 用途によって異なります。

- **スタック（片端のみ）**: `list` で十分。`append()` / `pop()` は O(1) で、キャッシュ効率も良い。
- **キュー（両端）**: `deque` を使う。`popleft()` が O(1) だが、`list.pop(0)` は O(n)。
- **ランダムアクセスが必要**: `list` を使う。`deque[i]` は O(n) だが、`list[i]` は O(1)。
- **固定長の履歴**: `deque(maxlen=N)` が便利。古い要素が自動で消える。

### Q3: 優先度キューとソート、どちらが効率的か?

**A**: 取り出す要素数によります。

- **全要素をソート順に取り出す**: ソート O(n log n) とヒープソート O(n log n) は同等。ただし、Python の `sorted()` は Timsort で定数倍が小さいため実測では速い。
- **上位 K 個だけ取り出す（K << n）**: `heapq.nlargest(k, data)` が O(n log k) で、全ソート O(n log n) より効率的。
- **動的にデータが追加される**: 優先度キューが適切。ソート済み配列への挿入は O(n) だが、ヒープへの挿入は O(log n)。
- **一括処理**: データがすべて揃ってから処理するなら、ソートの方がシンプルで高速なことが多い。

### Q4: なぜ BFS にはキュー、DFS にはスタックを使うのか?

**A**: 探索の順序がデータ構造の取り出し順序と一致するためです。

- **BFS（幅優先）**: レベル順に探索する。先に発見したノード（近いノード）を先に処理するため、FIFO のキューが必要。
- **DFS（深さ優先）**: できるだけ深く潜ってから戻る。最後に発見したノード（最も深いノード）を先に処理するため、LIFO のスタックが必要。

再帰関数による DFS は、コールスタック（暗黙のスタック）を利用している。明示的なスタックに書き換えることで、深い再帰によるスタックオーバーフローを回避できる。

### Q5: メッセージキュー（RabbitMQ, Kafka）とデータ構造のキューは何が違うのか?

**A**: 概念は同じ（FIFO で要素を処理する）だが、メッセージキューは分散システム向けの追加機能を持っています。

| 特徴 | データ構造のキュー | メッセージキュー |
|------|----------------|--------------|
| 永続性 | メモリ上のみ | ディスクに永続化可能 |
| 耐障害性 | プロセス終了で消失 | ブローカー障害時も復旧可能 |
| 分散 | 単一プロセス内 | ネットワーク越しに複数プロセス |
| 確認応答 | なし | ACK/NACK によるメッセージ管理 |
| スケーラビリティ | 単一マシン | 水平スケール可能 |

### Q6: スタックオーバーフローが発生したらどうすればいいか?

**A**: 以下の順に対処を検討してください。

1. **反復に書き換える**: 最も確実。スタックを明示的に管理する。
2. **末尾再帰に変換する**: 言語がサポートしていれば（Python は非サポート）。
3. **再帰の深さを減らす**: アルゴリズムの改善（例: メモ化、分割統治の改善）。
4. **スタックサイズを増やす**: 最終手段。`sys.setrecursionlimit()` や JVM の `-Xss` オプション。
5. **トランポリン**: 再帰呼び出しを遅延評価に変換するテクニック。

---

## 15. まとめ

### 主要データ構造の一覧

| データ構造 | 原則 | 主要操作 | 典型的な用途 |
|-----------|------|---------|-------------|
| スタック | LIFO（後入れ先出し） | push/pop: O(1) | コールスタック、括弧チェック、DFS、Undo/Redo |
| キュー | FIFO（先入れ先出し） | enqueue/dequeue: O(1) | BFS、タスクキュー、メッセージキュー |
| 両端キュー（Deque） | 両端操作 | 両端の追加/削除: O(1) | スライディングウィンドウ、Work Stealing |
| 優先度キュー | 優先度順 | push: O(log n), pop: O(log n) | ダイクストラ法、Top-K、タスクスケジューラ |
| 単調スタック | スタック内が単調 | push/pop: 償却 O(1) | Next Greater Element、ヒストグラム最大長方形 |
| 単調キュー | Deque 内が単調 | 各操作: 償却 O(1) | スライディングウィンドウの最大値/最小値 |

### 設計判断のまとめ

1. **「最後に入れたものを最初に取り出す」ならスタック** — DFS、式の評価、Undo
2. **「最初に入れたものを最初に取り出す」ならキュー** — BFS、タスク処理、メッセージング
3. **「両端から操作したい」なら Deque** — スライディングウィンドウ
4. **「優先度に基づいて取り出す」なら優先度キュー** — 最短経路、Top-K
5. **実装は配列ベースが基本** — キャッシュ効率が良く、実装も簡潔
6. **Python のキューは collections.deque を使う** — list.pop(0) は O(n) で致命的
7. **Java のスタックは ArrayDeque を使う** — Stack クラスは旧式で非推奨

### 学習ロードマップ

```
基礎: スタックとキューの操作を理解する
  │
  ├─ 括弧の対応検証（スタック）
  ├─ BFS（キュー）
  └─ 逆ポーランド記法（スタック）
  │
中級: 応用テクニックを習得する
  │
  ├─ 2つのスタックでキュー実装
  ├─ 単調スタック（Next Greater Element）
  ├─ 優先度キュー（ヒープ）
  └─ ダイクストラ法
  │
上級: 発展的な問題を解く
  │
  ├─ 単調キュー（スライディングウィンドウ）
  ├─ ヒストグラムの最大長方形
  ├─ データストリームの中央値
  └─ タスクスケジューラ
```

---

## 16. 次に読むべきガイド

→ [[03-hash-tables.md]] — ハッシュテーブル

---

## 17. 参考文献

1. Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapters 6 (Heapsort) and 10 (Elementary Data Structures).
   - スタック、キュー、ヒープの理論的基礎を網羅。計算量の証明が厳密。

2. Sedgewick, R., & Wayne, K. (2011). *Algorithms* (4th ed.). Addison-Wesley. Chapter 1.3 (Bags, Queues, and Stacks) and Chapter 2.4 (Priority Queues).
   - 実装寄りの解説が充実。Java による実装例が豊富。

3. Skiena, S. S. (2020). *The Algorithm Design Manual* (3rd ed.). Springer. Chapter 3: Data Structures.
   - 実務でのデータ構造の選び方に焦点。問題解決の戦略としてのスタック/キューの使い方。

4. CPython ソースコード — `Objects/listobject.c` および `Modules/_collectionsmodule.c`.
   - https://github.com/python/cpython
   - list と deque の内部実装の詳細。リサイズ戦略やブロック構造を確認できる。

5. Knuth, D. E. (1997). *The Art of Computer Programming, Volume 1: Fundamental Algorithms* (3rd ed.). Addison-Wesley. Section 2.2: Linear Lists.
   - スタックとキューの数学的基礎。歴史的経緯も含めた包括的な解説。

6. LeetCode Problems:
   - [20. Valid Parentheses](https://leetcode.com/problems/valid-parentheses/) — 括弧の対応検証
   - [150. Evaluate Reverse Polish Notation](https://leetcode.com/problems/evaluate-reverse-polish-notation/) — 逆ポーランド記法
   - [232. Implement Queue using Stacks](https://leetcode.com/problems/implement-queue-using-stacks/) — 2つのスタックでキュー
   - [239. Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/) — 単調キュー
   - [84. Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/) — 単調スタック
   - [295. Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/) — 2つのヒープ
   - [155. Min Stack](https://leetcode.com/problems/min-stack/) — 最小値取得 O(1) のスタック
