# Stacks and Queues — Implementation, Applications, and Priority Queues: A Complete Guide

> Gain a deep understanding of stacks and queues based on the LIFO / FIFO principles, and systematically learn their applications including parenthesis matching, BFS, monotonic stacks, and priority queues.

---


## What You Will Learn in This Chapter

- [ ] Understanding fundamental concepts and terminology
- [ ] Mastering implementation patterns and best practices
- [ ] Grasping practical application methods
- [ ] Learning the basics of troubleshooting

---

## Prerequisites

Before reading this guide, the following knowledge will help deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding of the content in [Linked Lists — Singly Linked, Doubly Linked, Circular, and Floyd's Algorithm](./01-linked-lists.md)

---

## Table of Contents

1. [Fundamental Concepts of Stacks and Queues](#1-fundamental-concepts-of-stacks-and-queues)
2. [Stack Implementation and Internal Structure](#2-stack-implementation-and-internal-structure)
3. [Queue Implementation and Internal Structure](#3-queue-implementation-and-internal-structure)
4. [Stack Application Algorithms](#4-stack-application-algorithms)
5. [Queue Application Algorithms](#5-queue-application-algorithms)
6. [Priority Queues and Heaps](#6-priority-queues-and-heaps)
7. [Specialized Stack and Queue Structures](#7-specialized-stack-and-queue-structures)
8. [Comparison Tables and Selection Guide](#8-comparison-tables-and-selection-guide)
9. [Anti-Patterns and Best Practices](#9-anti-patterns-and-best-practices)
10. [Exercises (3 Levels)](#10-exercises-3-levels)
11. [FAQ](#11-faq)
12. [Summary](#12-summary)
13. [References](#13-references)

---

## 1. Fundamental Concepts of Stacks and Queues

### 1.1 Definition as Abstract Data Types

Stacks and queues are among the most fundamental **abstract data types (ADT)** in computer science.
Both manage collections of elements, but they differ fundamentally in the order in which elements are retrieved.

| Property | Stack | Queue |
|----------|-------|-------|
| **Principle** | LIFO (Last In, First Out) | FIFO (First In, First Out) |
| **Everyday analogy** | Stacking plates, Undo operations | Checkout line, print jobs |
| **Primary operations** | push, pop, peek | enqueue, dequeue, front |
| **Insertion point** | Top | Rear (back) |
| **Removal point** | Top | Front |

### 1.2 Visual Understanding of Operations

```
Stack (LIFO — Last In, First Out):

  Initial state:        push(A):          push(B):          push(C):
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


Queue (FIFO — First In, First Out):

  Initial state:
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

### 1.3 Relationship with the Call Stack

Program execution itself is supported by a stack. Each function call pushes a stack frame,
and each `return` removes it. This mechanism is known as the **call stack**.

```
Function calls: main() → funcA() → funcB() → funcC()

Call stack progression:

  main() called  funcA() called  funcB() called  funcC() called  funcC() return
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

When recursion goes too deep, a `RecursionError` (Python) or `StackOverflowError` (Java) is raised.
This means the stack space allocated by the OS has been exceeded.

### 1.4 Use Case Map for Stacks and Queues

```
┌──────────────────────────────────────────────────────┐
│                   Use Case Map                        │
├──────────────────────────────────────────────────────┤
│                                                      │
│  Stack (LIFO)                Queue (FIFO)            │
│  ├─ DFS (Depth-First Search) ├─ BFS (Breadth-First) │
│  ├─ Parenthesis matching     ├─ Task scheduling      │
│  ├─ Undo/Redo               ├─ Message queues       │
│  ├─ Reverse Polish Notation  ├─ Printer queues       │
│  ├─ Browser back/forward     ├─ Event loops          │
│  ├─ Function call stack      ├─ Cache (FIFO)         │
│  ├─ Syntax parsing           ├─ Stream processing    │
│  └─ Monotonic stack          └─ Fair resource alloc.  │
│                                                      │
│  Priority Queue              Deque (Double-Ended Q)  │
│  ├─ Dijkstra shortest path   ├─ Sliding window       │
│  ├─ Huffman encoding         ├─ Work stealing        │
│  ├─ A* search                ├─ Palindrome check     │
│  ├─ Event-driven simulation  ├─ Browser history      │
│  └─ OS process scheduling    └─ Bidirectional BFS    │
│                                                      │
└──────────────────────────────────────────────────────┘
```

---

## 2. Stack Implementation and Internal Structure

### 2.1 Array-Based Stack (Python List)

Python's `list` is a dynamic array that supports amortized O(1) appends and removals at the end.
It is ideal for implementing stacks.

```python
class ArrayStack:
    """
    Array-based stack implementation.
    Uses a Python list internally; all operations run in amortized O(1).

    Usage:
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
        """Push an element onto the top. Amortized O(1)."""
        self._data.append(val)

    def pop(self):
        """Remove and return the top element. O(1).
        Raises IndexError if called on an empty stack.
        """
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._data.pop()

    def peek(self):
        """Return the top element without removing it. O(1)."""
        if self.is_empty():
            raise IndexError("peek from empty stack")
        return self._data[-1]

    def is_empty(self):
        """Return True if the stack is empty. O(1)."""
        return len(self._data) == 0

    def __len__(self):
        """Return the number of elements in the stack. O(1)."""
        return len(self._data)

    def __repr__(self):
        return f"ArrayStack({self._data})"

    def __iter__(self):
        """Iterate from top to bottom (non-destructive)."""
        return reversed(self._data)

    def clear(self):
        """Remove all elements. O(1)."""
        self._data.clear()


# === Demo ===
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

### 2.2 Linked List-Based Stack

Using a linked list completely avoids the occasional O(n) cost of dynamic array resizing.
However, each node incurs pointer memory overhead.

```python
class _Node:
    """Node for a singly linked list."""
    __slots__ = ('value', 'next')

    def __init__(self, value, next_node=None):
        self.value = value
        self.next = next_node


class LinkedStack:
    """
    Linked list-based stack implementation.
    All operations run in worst-case O(1) (true O(1), not amortized).

    Usage:
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
        """Add a new node to the top. O(1)."""
        self._top = _Node(val, self._top)
        self._size += 1

    def pop(self):
        """Remove the top node and return its value. O(1)."""
        if self.is_empty():
            raise IndexError("pop from empty stack")
        val = self._top.value
        self._top = self._top.next
        self._size -= 1
        return val

    def peek(self):
        """Return the top value without removing it. O(1)."""
        if self.is_empty():
            raise IndexError("peek from empty stack")
        return self._top.value

    def is_empty(self):
        return self._size == 0

    def __len__(self):
        return self._size

    def __iter__(self):
        """Iterate from top to bottom."""
        node = self._top
        while node is not None:
            yield node.value
            node = node.next


# === Demo ===
if __name__ == "__main__":
    s = LinkedStack()
    for v in ["alpha", "beta", "gamma"]:
        s.push(v)
    print(f"Top: {s.peek()}")        # gamma
    print(f"Pop: {s.pop()}")         # gamma
    print(f"Remaining: {list(s)}")   # ['beta', 'alpha']
```

### 2.3 Fixed-Capacity Stack (For Embedded Systems)

In environments with strictly constrained memory, such as embedded systems, fixed-capacity stacks are used.

```python
class BoundedStack:
    """
    Fixed-capacity stack. Raises OverflowError when capacity is exceeded.
    Used in embedded systems and real-time systems.

    Usage:
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
        """O(1). Raises OverflowError if full."""
        if self.is_full():
            raise OverflowError(
                f"stack is full (capacity={self._capacity})"
            )
        self._top += 1
        self._data[self._top] = val

    def pop(self):
        """O(1)."""
        if self.is_empty():
            raise IndexError("pop from empty stack")
        val = self._data[self._top]
        self._data[self._top] = None  # Release reference
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


# === Demo ===
if __name__ == "__main__":
    bs = BoundedStack(4)
    for v in [10, 20, 30, 40]:
        bs.push(v)
    print(f"Full? {bs.is_full()}")    # True
    print(f"Pop: {bs.pop()}")         # 40
    print(f"Full? {bs.is_full()}")    # False
```

### 2.4 Array vs. Linked List — Stack Implementation Comparison

| Aspect | Array-Based (list) | Linked List-Based |
|--------|-------------------|-------------------|
| push complexity | O(1) amortized | O(1) worst-case |
| pop complexity | O(1) | O(1) |
| Memory efficiency | High (contiguous memory) | Low (node + pointer overhead) |
| Cache performance | Good (high locality) | Poor (potentially scattered) |
| Capacity limit | Dynamically expandable | Dynamically expandable |
| Resize cost | Occasional O(n) | None |
| Implementation simplicity | Very simple | Slightly more complex |
| **Recommended use** | **General purpose (recommended)** | **Strict real-time requirements** |

---

## 3. Queue Implementation and Internal Structure

### 3.1 Queue Using collections.deque

Python's `collections.deque` is implemented as a doubly linked list,
providing O(1) operations at both ends. `deque` is the most recommended choice for implementing queues.

```python
from collections import deque


class Queue:
    """
    deque-based queue implementation.
    Both enqueue and dequeue operate in O(1).

    Usage:
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
        """Append an element to the rear. O(1)."""
        self._data.append(val)

    def dequeue(self):
        """Remove and return the front element. O(1)."""
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        return self._data.popleft()

    def front(self):
        """Return the front element without removing it. O(1)."""
        if self.is_empty():
            raise IndexError("front from empty queue")
        return self._data[0]

    def rear(self):
        """Return the rear element without removing it. O(1)."""
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
        """Iterate from front to rear (non-destructive)."""
        return iter(self._data)


# === Demo ===
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

### 3.2 Queue Using a Circular Buffer (Ring Buffer)

A technique for implementing a queue on a fixed-size array. Since memory allocation occurs only once,
it is widely used in embedded systems and kernel buffering.

```python
class CircularQueue:
    """
    Fixed-capacity circular buffer queue.
    Array indices wrap around using modular arithmetic.

    Usage:
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
        """O(1). Raises OverflowError if full."""
        if self.is_full():
            raise OverflowError("circular queue is full")
        rear = (self._front + self._size) % self._capacity
        self._data[rear] = val
        self._size += 1

    def dequeue(self):
        """O(1)."""
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


# === Demo ===
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
Internal operation of a circular buffer:

  capacity = 4

  Initial state:        enqueue A,B,C:       dequeue() → A:
  ┌───┬───┬───┬───┐ ┌───┬───┬───┬───┐   ┌───┬───┬───┬───┐
  │   │   │   │   │ │ A │ B │ C │   │   │   │ B │ C │   │
  └───┴───┴───┴───┘ └───┴───┴───┴───┘   └───┴───┴───┴───┘
   ↑f               ↑f          ↑r          ↑f      ↑r
                                  (rear)     (front)

  enqueue D:          enqueue E:          Result:
  ┌───┬───┬───┬───┐  ┌───┬───┬───┬───┐  front=1, size=4
  │   │ B │ C │ D │  │ E │ B │ C │ D │  The array wraps around
  └───┴───┴───┴───┘  └───┴───┴───┴───┘  and is reused
      ↑f          ↑r  ↑r  ↑f
```

### 3.3 Queue Implementation Using Two Stacks

A frequently asked interview technique. Two stacks (`in_stack` and `out_stack`) are used to achieve FIFO behavior.
The amortized complexity is O(1).

```python
class QueueWithTwoStacks:
    """
    Queue implementation using two stacks.
    - in_stack: for enqueue
    - out_stack: for dequeue
    - Elements are transferred from in_stack to out_stack only when out_stack is empty

    Amortized analysis:
      Each element is pushed to in_stack once, popped from in_stack once,
      pushed to out_stack once, and popped from out_stack once.
      Total 4 operations → amortized O(1) per element.

    Usage:
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
        """O(1)."""
        self._in_stack.append(val)

    def dequeue(self):
        """Amortized O(1)."""
        if not self._out_stack:
            if not self._in_stack:
                raise IndexError("dequeue from empty queue")
            while self._in_stack:
                self._out_stack.append(self._in_stack.pop())
        return self._out_stack.pop()

    def front(self):
        """Amortized O(1)."""
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


# === Demo ===
if __name__ == "__main__":
    q = QueueWithTwoStacks()
    # Interleaved enqueue / dequeue still maintains correct FIFO order
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
Operation of a queue using two stacks:

  enqueue(1), enqueue(2), enqueue(3):

    in_stack:          out_stack:
    ┌───┐
    │ 3 │ ← top       (empty)
    ├───┤
    │ 2 │
    ├───┤
    │ 1 │
    └───┘

  dequeue() called → out_stack is empty, so transfer:

    in_stack:          out_stack:
                       ┌───┐
    (empty)            │ 1 │ ← top  ← retrieved on next dequeue
                       ├───┤
                       │ 2 │
                       ├───┤
                       │ 3 │
                       └───┘

  dequeue() → returns 1:

    in_stack:          out_stack:
                       ┌───┐
    (empty)            │ 2 │ ← top
                       ├───┤
                       │ 3 │
                       └───┘
```

---

## 4. Stack Application Algorithms

### 4.1 Parenthesis Matching (Balance Check)

A fundamental algorithm used in compiler syntax analysis and code editor highlighting features.
Opening brackets are pushed onto the stack, and when a closing bracket appears, the pair is verified.

```python
def is_valid_parentheses(s: str) -> bool:
    """
    Check whether parentheses are correctly balanced.
    Matching pairs: (), {}, []

    Complexity: O(n) time, O(n) space

    Usage:
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


# === Tests ===
if __name__ == "__main__":
    test_cases = [
        ("({[]})", True),
        ("({[})", False),
        ("", True),
        ("((", False),
        ("))", False),
        ("[{()}]", True),
        ("a(b{c[d]e}f)g", True),  # Contains non-bracket characters
    ]
    for expr, expected in test_cases:
        result = is_valid_parentheses(expr)
        status = "PASS" if result == expected else "FAIL"
        print(f"[{status}] is_valid_parentheses('{expr}') = {result}")
```

### 4.2 Reverse Polish Notation (RPN) Evaluation

Reverse Polish Notation is a notation where operators follow their operands, eliminating the need for parentheses.
It is used in HP calculators and the PostScript language.

```python
import operator
from typing import List


def eval_rpn(tokens: List[str]) -> float:
    """
    Evaluate an expression in Reverse Polish Notation (postfix notation).

    Algorithm:
      1. Read tokens from left to right
      2. If it's a number, push it onto the stack
      3. If it's an operator, pop two operands, compute, and push the result
      4. The final value remaining on the stack is the answer

    Complexity: O(n) time, O(n) space

    Usage:
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
    Convert infix notation to Reverse Polish Notation (Shunting-Yard algorithm).
    Correctly handles operator precedence and associativity.

    Usage:
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
            op_stack.pop()  # Discard '('
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


# === Demo ===
if __name__ == "__main__":
    # Infix → Reverse Polish Notation → Evaluate
    infix = "( 2 + 3 ) * 4"
    rpn = infix_to_rpn(infix)
    result = eval_rpn(rpn)
    print(f"Infix:  {infix}")
    print(f"RPN:    {' '.join(rpn)}")
    print(f"Result: {result}")  # 20.0

    # Evaluate RPN directly
    print(f"\neval_rpn(['5','1','2','+','4','*','+','3','-']) = "
          f"{eval_rpn(['5','1','2','+','4','*','+','3','-'])}")  # 14.0
```

### 4.3 Monotonic Stack

A monotonic stack is a powerful technique for finding the "next greater element" or "next smaller element" in O(n).
Elements in the stack are maintained in monotonically increasing (or decreasing) order.

```python
from typing import List


def next_greater_element(nums: List[int]) -> List[int]:
    """
    For each element, return the first element to its right that is greater.
    Returns -1 if no such element exists.

    Algorithm:
      Store indices in the stack and maintain monotonic decreasing order.
      When a new element is greater than the stack top,
      the "next greater element" for the stack top has been found.

    Complexity: O(n) time, O(n) space
    Each element is pushed at most once and popped at most once, hence linear.

    Usage:
        >>> next_greater_element([2, 1, 4, 3])
        [4, 4, -1, -1]
        >>> next_greater_element([1, 3, 2, 4])
        [3, 4, 4, -1]
        >>> next_greater_element([5, 4, 3, 2, 1])
        [-1, -1, -1, -1, -1]
    """
    n = len(nums)
    result = [-1] * n
    stack = []  # Stores indices

    for i in range(n):
        while stack and nums[stack[-1]] < nums[i]:
            idx = stack.pop()
            result[idx] = nums[i]
        stack.append(i)

    return result


def daily_temperatures(temperatures: List[int]) -> List[int]:
    """
    For each day, return how many days until a warmer temperature.
    Classic LeetCode 739 problem.

    Usage:
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
    Find the largest rectangular area in a histogram.
    Classic LeetCode 84 problem. A representative application of monotonic stacks.

    Usage:
        >>> largest_rectangle_in_histogram([2, 1, 5, 6, 2, 3])
        10
        >>> largest_rectangle_in_histogram([2, 4])
        4
    """
    stack = []
    max_area = 0
    heights = heights + [0]  # Add sentinel

    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)

    return max_area


# === Demo ===
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
Monotonic stack walkthrough — next_greater_element([2, 7, 4, 3, 5]):

  i=0, num=2:  stack=[]          → push 0        stack=[0]
  i=1, num=7:  stack=[0]         → 2<7 → pop 0, result[0]=7
                                  → push 1        stack=[1]
  i=2, num=4:  stack=[1]         → 7>4 → push 2   stack=[1,2]
  i=3, num=3:  stack=[1,2]       → 4>3 → push 3   stack=[1,2,3]
  i=4, num=5:  stack=[1,2,3]     → 3<5 → pop 3, result[3]=5
                                  → 4<5 → pop 2, result[2]=5
                                  → 7>5 → push 4   stack=[1,4]

  Final result: [7, -1, 5, 5, -1]
  (Indices 1,4 remain on the stack → no greater element to the right → -1)
```

### 4.4 Implementing Browser Back/Forward Navigation

A practical example using two stacks.

```python
class BrowserHistory:
    """
    Implement browser back/forward navigation using two stacks.

    Usage:
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
        >>> browser.forward()  # Forward history has been cleared
        'twitter.com'
    """

    def __init__(self, homepage: str):
        self._current = homepage
        self._back_stack = []
        self._forward_stack = []

    def visit(self, url: str):
        """Visit a new page. Forward history is cleared."""
        self._back_stack.append(self._current)
        self._current = url
        self._forward_stack.clear()

    def back(self) -> str:
        """Go back to the previous page."""
        if self._back_stack:
            self._forward_stack.append(self._current)
            self._current = self._back_stack.pop()
        return self._current

    def forward(self) -> str:
        """Go forward to the next page."""
        if self._forward_stack:
            self._back_stack.append(self._current)
            self._current = self._forward_stack.pop()
        return self._current

    @property
    def current_page(self) -> str:
        return self._current


# === Demo ===
if __name__ == "__main__":
    browser = BrowserHistory("google.com")
    browser.visit("youtube.com")
    browser.visit("github.com")
    browser.visit("stackoverflow.com")

    print(f"Current: {browser.current_page}")  # stackoverflow.com
    print(f"Back: {browser.back()}")           # github.com
    print(f"Back: {browser.back()}")           # youtube.com
    print(f"Forward: {browser.forward()}")     # github.com
    browser.visit("twitter.com")               # Forward history cleared
    print(f"Current: {browser.current_page}")  # twitter.com
    print(f"Forward: {browser.forward()}")     # twitter.com (cannot go forward)
    print(f"Back: {browser.back()}")           # youtube.com
```

---

## 5. Queue Application Algorithms

### 5.1 Breadth-First Search (BFS)

BFS is a graph and tree traversal method that explores vertices in order of proximity to the starting point.
It is implemented using a queue and serves as the foundation for shortest path problems (when edge weights are equal).

```python
from collections import deque
from typing import Dict, List, Optional


def bfs(graph: Dict[str, List[str]], start: str) -> List[str]:
    """
    Breadth-first search on a graph. Returns the visit order.

    Complexity: O(V + E) (V: vertices, E: edges)

    Usage:
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
    Find the shortest path using BFS (unweighted graph).

    Usage:
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

    return None  # Unreachable


# === Demo ===
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
BFS walkthrough (graph example):

  Graph structure:
       A
      / \
     B   C
    / \   \
   D   E - F

  BFS queue progression (start = A):

  Step 0: queue=[A]          visited={A}
  Step 1: dequeue A → queue=[B,C]      visited={A,B,C}
  Step 2: dequeue B → queue=[C,D,E]    visited={A,B,C,D,E}
  Step 3: dequeue C → queue=[D,E,F]    visited={A,B,C,D,E,F}
  Step 4: dequeue D → queue=[E,F]      (D's neighbor B already visited)
  Step 5: dequeue E → queue=[F]        (E's neighbors B,F already visited)
  Step 6: dequeue F → queue=[]         (F's neighbors C,E already visited)

  Visit order: A → B → C → D → E → F  (level order)
```

### 5.2 Level-Order Traversal of a Binary Tree

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
    Traverse a binary tree in level order (BFS) and return values grouped by level.
    Classic LeetCode 102 problem.

    Complexity: O(n) time, O(n) space

    Usage:
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


# === Demo ===
if __name__ == "__main__":
    root = TreeNode(3,
        TreeNode(9),
        TreeNode(20, TreeNode(15), TreeNode(7))
    )
    print(f"Level order: {level_order_traversal(root)}")
    # [[3], [9, 20], [15, 7]]
```

### 5.3 Task Scheduling (Round Robin)

Implementing the round-robin scheduling method used in OS process scheduling with a queue.

```python
from collections import deque
from typing import List, Tuple


def round_robin_scheduling(
    tasks: List[Tuple[str, int]], time_quantum: int
) -> List[str]:
    """
    Round-robin task scheduling.
    Allocates equal CPU time (time quantum) to each task.

    Args:
        tasks: List of (task_name, execution_time) tuples
        time_quantum: Maximum execution time per turn

    Returns:
        List of execution log entries

    Usage:
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


# === Demo ===
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

### 5.4 Snakes and Ladders (Minimum Moves via BFS)

An example of finding the shortest solution for a board game using BFS. A perfect illustration of how queue-based BFS is optimal for shortest path problems.

```python
from collections import deque
from typing import Dict


def snakes_and_ladders(
    board_size: int,
    snakes: Dict[int, int],
    ladders: Dict[int, int]
) -> int:
    """
    Find the minimum number of moves to reach the goal in Snakes and Ladders using BFS.

    Args:
        board_size: Number of squares on the board (1-indexed, goal is board_size)
        snakes: {head_position: tail_position} — descend via snake
        ladders: {bottom_position: top_position} — ascend via ladder

    Returns:
        Minimum number of moves. Returns -1 if unreachable.

    Usage:
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
    queue = deque([(1, 0)])  # (current_position, move_count)

    while queue:
        position, moves = queue.popleft()

        for dice in range(1, 7):  # Dice roll 1-6
            next_pos = position + dice
            if next_pos > goal:
                continue
            # Move if there is a snake or ladder
            next_pos = shortcuts.get(next_pos, next_pos)
            if next_pos == goal:
                return moves + 1
            if next_pos not in visited:
                visited.add(next_pos)
                queue.append((next_pos, moves + 1))

    return -1


# === Demo ===
if __name__ == "__main__":
    result = snakes_and_ladders(
        30,
        snakes={17: 7, 27: 1},
        ladders={3: 22, 5: 8, 11: 26, 20: 29}
    )
    print(f"Minimum moves: {result}")  # 3
    # Shortest path: 1 → (dice=2) → 3 → (ladder) → 22 → (dice=4) → 26
    #         → ... → (dice=4) → 29 → (dice=1) → 30 = goal
```

---

## 6. Priority Queues and Heaps

### 6.1 The Concept of Priority Queues

While a regular queue processes elements in FIFO (first-come, first-served) order,
a priority queue retrieves the element with the **highest priority** first.

```
Regular Queue vs. Priority Queue:

  Regular Queue (FIFO):
    enqueue order: A(pri=3), B(pri=1), C(pri=4), D(pri=2)
    dequeue order: A → B → C → D  (first-come, first-served)

  Priority Queue (min-priority):
    enqueue order: A(pri=3), B(pri=1), C(pri=4), D(pri=2)
    dequeue order: B → D → A → C  (lower number = higher importance)

  Internal heap structure (min-heap):

    push A(3):    push B(1):    push C(4):    push D(2):
       3             1             1             1
                    / \           / \           / \
                   3             3   4         2   4
                                              /
                                             3
```

### 6.2 Priority Queue Using heapq

```python
import heapq
from typing import Any, List, Tuple


class PriorityQueue:
    """
    Min-heap-based priority queue.
    Preserves insertion order for equal priorities (FIFO tiebreaker).

    Usage:
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
        """Add an element with a priority. O(log n)."""
        heapq.heappush(self._heap, (priority, self._counter, item))
        self._counter += 1

    def pop(self) -> Tuple[Any, Any]:
        """Remove and return the highest-priority element. O(log n)."""
        if self.is_empty():
            raise IndexError("pop from empty priority queue")
        priority, _, item = heapq.heappop(self._heap)
        return priority, item

    def peek(self) -> Tuple[Any, Any]:
        """Return the highest-priority element without removing it. O(1)."""
        if self.is_empty():
            raise IndexError("peek from empty priority queue")
        priority, _, item = self._heap[0]
        return priority, item

    def is_empty(self) -> bool:
        return len(self._heap) == 0

    def __len__(self) -> int:
        return len(self._heap)


# === Demo ===
if __name__ == "__main__":
    pq = PriorityQueue()

    # Lower numeric priority = higher importance
    pq.push(3, "backup database")
    pq.push(1, "fix critical bug")
    pq.push(2, "review PR")
    pq.push(1, "deploy hotfix")  # Same priority → FIFO

    print("=== Processing tasks by priority ===")
    while not pq.is_empty():
        priority, task = pq.pop()
        print(f"  Priority {priority}: {task}")
    # Priority 1: fix critical bug
    # Priority 1: deploy hotfix
    # Priority 2: review PR
    # Priority 3: backup database
```

### 6.3 Priority Queue Application: k Smallest Elements

```python
import heapq
from typing import List


def k_smallest_elements(nums: List[int], k: int) -> List[int]:
    """
    Return the k smallest elements from a list.

    Method 1: heapq.nsmallest — O(n log k)
    Method 2: Sort — O(n log n)
    Method 3: Max-heap of size k — O(n log k)

    Usage:
        >>> k_smallest_elements([7, 10, 4, 3, 20, 15], 3)
        [3, 4, 7]
    """
    return heapq.nsmallest(k, nums)


def k_largest_elements(nums: List[int], k: int) -> List[int]:
    """
    Return the k largest elements from a list.

    Usage:
        >>> k_largest_elements([7, 10, 4, 3, 20, 15], 3)
        [20, 15, 10]
    """
    return heapq.nlargest(k, nums)


def merge_k_sorted_lists(lists: List[List[int]]) -> List[int]:
    """
    Merge k sorted lists into a single sorted list.
    Classic LeetCode 23 problem.

    Complexity: O(N log k) — N: total elements, k: number of lists

    Usage:
        >>> merge_k_sorted_lists([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
        [1, 2, 3, 4, 5, 6, 7, 8, 9]
    """
    result = []
    heap = []

    # Push the first element of each list onto the heap
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))

    while heap:
        val, list_idx, elem_idx = heapq.heappop(heap)
        result.append(val)

        # Push the next element from the same list if available
        if elem_idx + 1 < len(lists[list_idx]):
            next_val = lists[list_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))

    return result


# === Demo ===
if __name__ == "__main__":
    nums = [7, 10, 4, 3, 20, 15]
    print(f"3 smallest: {k_smallest_elements(nums, 3)}")  # [3, 4, 7]
    print(f"3 largest:  {k_largest_elements(nums, 3)}")    # [20, 15, 10]

    sorted_lists = [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
    print(f"Merged: {merge_k_sorted_lists(sorted_lists)}")
    # [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

### 6.4 Comparison of Priority Queue Implementations

| Implementation | Insert | Extract-Min | Peek-Min | Notes |
|----------------|--------|-------------|----------|-------|
| Unsorted array | O(1) | O(n) | O(n) | Fast insert but slow extraction |
| Sorted array | O(n) | O(1) | O(1) | Fast extraction but slow insert |
| Binary heap | O(log n) | O(log n) | O(1) | **Well-balanced (standard)** |
| Fibonacci heap | O(1) amortized | O(log n) amortized | O(1) | Theoretically superior but complex to implement |
| Balanced BST | O(log n) | O(log n) | O(log n) | Versatile but has overhead |

---

## 7. Specialized Stack and Queue Structures

### 7.1 Stack with O(1) Minimum Retrieval (MinStack)

Classic LeetCode 155 problem. Supports push / pop / top plus O(1) minimum retrieval.

```python
class MinStack:
    """
    Stack with all O(1) operations (including minimum retrieval).

    Idea:
      Store each element as a pair along with the "minimum at that point in time."

    Usage:
        >>> ms = MinStack()
        >>> ms.push(5)
        >>> ms.push(3)
        >>> ms.push(7)
        >>> ms.get_min()
        3
        >>> ms.pop()  # Remove 7
        7
        >>> ms.get_min()
        3
        >>> ms.pop()  # Remove 3
        3
        >>> ms.get_min()
        5
    """

    def __init__(self):
        self._stack = []  # Pairs of (value, current_min)

    def push(self, val: int):
        """O(1)."""
        current_min = min(val, self._stack[-1][1]) if self._stack else val
        self._stack.append((val, current_min))

    def pop(self) -> int:
        """O(1)."""
        if not self._stack:
            raise IndexError("pop from empty MinStack")
        return self._stack.pop()[0]

    def top(self) -> int:
        """O(1)."""
        if not self._stack:
            raise IndexError("top from empty MinStack")
        return self._stack[-1][0]

    def get_min(self) -> int:
        """Return the current minimum in O(1)."""
        if not self._stack:
            raise IndexError("get_min from empty MinStack")
        return self._stack[-1][1]

    def __len__(self) -> int:
        return len(self._stack)


# === Demo ===
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

### 7.2 Deque (Double-Ended Queue)

A deque (double-ended queue) supports O(1) insertion and removal at both ends.
It combines the capabilities of both stacks and queues.

```python
from collections import deque
from typing import List


def sliding_window_maximum(nums: List[int], k: int) -> List[int]:
    """
    Find the maximum value in each sliding window.
    Classic LeetCode 239 problem. Uses a monotonically decreasing deque.

    Complexity: O(n) time, O(k) space

    Algorithm:
      Store indices in the deque, maintaining monotonically decreasing values.
      The front of the deque always holds the index of the maximum value in the window.

    Usage:
        >>> sliding_window_maximum([1, 3, -1, -3, 5, 3, 6, 7], 3)
        [3, 3, 5, 5, 6, 7]
    """
    if not nums or k == 0:
        return []

    dq = deque()  # Stores indices (monotonically decreasing)
    result = []

    for i in range(len(nums)):
        # Remove indices outside the window from the front
        while dq and dq[0] < i - k + 1:
            dq.popleft()

        # Remove indices of elements smaller than the current from the back
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()

        dq.append(i)

        # Record the maximum once the window is complete
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result


# === Demo ===
if __name__ == "__main__":
    nums = [1, 3, -1, -3, 5, 3, 6, 7]
    k = 3
    print(f"Input: {nums}, k={k}")
    print(f"Max values: {sliding_window_maximum(nums, k)}")
    # [3, 3, 5, 5, 6, 7]

    # Window verification:
    # [1, 3, -1] → max=3
    # [3, -1, -3] → max=3
    # [-1, -3, 5] → max=5
    # [-3, 5, 3] → max=5
    # [5, 3, 6] → max=6
    # [3, 6, 7] → max=7
```

```
Sliding window maximum — deque walkthrough:

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

  Result: [3, 3, 5, 5, 6, 7]
```

### 7.3 Converting Between DFS with Stack and Recursion

Recursion uses an implicit stack (the call stack). All recursive algorithms can be converted to iterative
form using an explicit stack. This is an important technique for avoiding stack overflow risks.

```python
from typing import Dict, List


def dfs_recursive(
    graph: Dict[str, List[str]], start: str,
    visited: set = None
) -> List[str]:
    """DFS using recursion."""
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
    """DFS using an explicit stack (no recursion)."""
    visited = set()
    stack = [start]
    order = []

    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            order.append(vertex)
            # Push in reverse order so the first neighbor is explored first
            for neighbor in reversed(graph[vertex]):
                if neighbor not in visited:
                    stack.append(neighbor)

    return order


# === Demo ===
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
    # Both produce A → B → D → E → F → C (depending on adjacency list order)
```

---

## 8. Comparison Tables and Selection Guide

### Table 1: Operation Complexity for Stack, Queue, Deque, and Priority Queue

| Operation | Stack | Queue (deque) | Deque (deque) | Priority Queue (Heap) |
|-----------|-------|---------------|---------------|-----------------------|
| Insert at front | N/A | N/A | O(1) | N/A |
| Insert at back | O(1) push | O(1) enqueue | O(1) | O(log n) push |
| Remove from front | N/A | O(1) dequeue | O(1) | O(log n) pop |
| Remove from back | O(1) pop | N/A | O(1) | N/A |
| Peek front | N/A | O(1) front | O(1) | O(1) peek |
| Peek back | O(1) peek | O(1) rear | O(1) | N/A |
| Random access | O(n) | O(n) | O(n) | O(n) |
| Search | O(n) | O(n) | O(n) | O(n) |
| Ordering | LIFO | FIFO | Both ends | Priority order |

### Table 2: Queue Implementation Methods and Selection Criteria

| Implementation | enqueue | dequeue | Space Complexity | Memory Characteristics | Best Use Case |
|----------------|---------|---------|-----------------|----------------------|---------------|
| Python list | O(1) amortized | O(n) | O(n) dynamic | Contiguous | **Do not use (not recommended)** |
| collections.deque | O(1) | O(1) | O(n) dynamic | Block-linked | **General purpose (recommended)** |
| Linked list | O(1) | O(1) | O(n) + pointers | Scattered | When node reuse is needed |
| Circular buffer | O(1) | O(1) | O(n) fixed | Contiguous | Embedded / fixed capacity |
| Two stacks | O(1) | O(1) amortized | O(n) | Contiguous x 2 | Interviews / constrained problems |
| queue.Queue | O(1) | O(1) | O(n) dynamic | With locks | **Multi-threaded** |
| asyncio.Queue | O(1) | O(1) | O(n) dynamic | Async-compatible | **asyncio environments** |

### Table 3: Data Structure Selection Flowchart

```
What is the element retrieval order?
 │
 ├─ Last in, first out → Stack
 │   └─ Need O(1) minimum too? → MinStack
 │
 ├─ First in, first out → Queue
 │   ├─ Need thread safety? → queue.Queue
 │   ├─ Async processing? → asyncio.Queue
 │   ├─ Fixed capacity? → CircularQueue
 │   └─ General purpose → collections.deque
 │
 ├─ Highest priority first → Priority Queue
 │   └─ Python standard → heapq + wrapper class
 │
 └─ Operations at both ends → Deque (collections.deque)
     └─ Sliding window? → Monotonic deque
```

---

## 9. Anti-Patterns and Best Practices

### Anti-Pattern 1: Using list.pop(0) for Queue Dequeue

Python's `list.pop(0)` removes the first element and then shifts all remaining elements forward by one.
This is an O(n) operation that causes severe performance degradation when the number of elements is large.

```python
import time
from collections import deque


def benchmark_queue_pop():
    """Benchmark: list.pop(0) vs deque.popleft()."""

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


# === Example output ===
if __name__ == "__main__":
    benchmark_queue_pop()
    # n= 10,000: list.pop(0)=0.0121s, deque.popleft()=0.0004s, ratio=30.3x
    # n= 50,000: list.pop(0)=0.2890s, deque.popleft()=0.0020s, ratio=144.5x
    # n=100,000: list.pop(0)=1.1520s, deque.popleft()=0.0040s, ratio=288.0x
    # * The gap grows more pronounced as n increases
```

**Lesson:** Always use `collections.deque` for FIFO queues. `list.pop(0)` is O(n),
and at around 100,000 elements, the speed difference can reach several hundred times.

### Anti-Pattern 2: Popping Elements to Search Through a Stack

A stack is designed for "accessing only the top element," but sometimes you need to check
whether a specific value exists inside it. Popping and restoring is inefficient and error-prone.

```python
# === BAD: Destructive search ===
def find_in_stack_bad(stack_data: list, target) -> bool:
    """Pop to search, then restore. O(n) but twice the operations + risk of bugs."""
    temp = []
    found = False
    while stack_data:
        val = stack_data.pop()
        temp.append(val)
        if val == target:
            found = True
            break
    # Restore (return all elements even if we broke early)
    while temp:
        stack_data.append(temp.pop())
    return found


# === GOOD: Direct traversal of internal data ===
def find_in_stack_good(stack_data: list, target) -> bool:
    """Directly traverse the internal list. O(n) with half the operations."""
    return target in stack_data


# === Demo ===
if __name__ == "__main__":
    stack = [10, 20, 30, 40, 50]

    # BAD
    result_bad = find_in_stack_bad(stack, 30)
    print(f"BAD: found={result_bad}, stack={stack}")
    # found=True, stack=[10, 20, 30, 40, 50] (restored)

    # GOOD
    result_good = find_in_stack_good(stack, 30)
    print(f"GOOD: found={result_good}, stack={stack}")
    # found=True, stack=[10, 20, 30, 40, 50]
```

**Lesson:** Distinguish between the ADT interface of a stack/queue and its underlying implementation.
If you need search capability, access the internal data structure directly or use an auxiliary data structure (such as a set).

### Anti-Pattern 3: DFS Without Considering Recursion Depth

```python
import sys


def dfs_naive(graph, start, visited=None):
    """Recursive DFS — risk of stack overflow on large graphs."""
    if visited is None:
        visited = set()
    visited.add(start)
    for neighbor in graph.get(start, []):
        if neighbor not in visited:
            dfs_naive(graph, neighbor, visited)  # Deep recursion!
    return visited


def dfs_safe(graph, start):
    """DFS using explicit stack — avoids stack overflow."""
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


# === Demo ===
if __name__ == "__main__":
    # Python's default recursion limit
    print(f"Default recursion limit: {sys.getrecursionlimit()}")
    # Typically 1000

    # Create a linear graph (depth 2000)
    linear_graph = {i: [i + 1] for i in range(2000)}
    linear_graph[2000] = []

    # Recursive version: may raise RecursionError
    # dfs_naive(linear_graph, 0)  # RecursionError!

    # Iterative version: works without issues
    result = dfs_safe(linear_graph, 0)
    print(f"Iterative DFS visited {len(result)} nodes")  # 2001
```

**Lesson:** Use an explicit stack for DFS on large graphs. On graphs exceeding Python's recursion limit
(default 1000), `RecursionError` will be raised. While `sys.setrecursionlimit()` can increase the limit,
it consumes OS stack space and is not recommended.

### Anti-Pattern 4: Comparison Errors with Equal Priorities in Priority Queues

```python
import heapq


# === BAD: Pushing non-comparable objects onto the heap ===
class Task:
    def __init__(self, name, priority):
        self.name = name
        self.priority = priority

# heapq.heappush(heap, (1, Task("a", 1)))
# heapq.heappush(heap, (1, Task("b", 1)))
# → TypeError: '<' not supported between instances of 'Task' and 'Task'
# When priorities are equal, the tuple tries to compare the second element, causing an error


# === GOOD: Use a counter as a tiebreaker ===
class SafePriorityQueue:
    """Priority queue that avoids TypeError on equal priorities."""

    def __init__(self):
        self._heap = []
        self._counter = 0

    def push(self, priority, item):
        # Counter serves as tiebreaker, so item comparison never occurs
        heapq.heappush(self._heap, (priority, self._counter, item))
        self._counter += 1

    def pop(self):
        priority, _, item = heapq.heappop(self._heap)
        return priority, item


# === Demo ===
if __name__ == "__main__":
    pq = SafePriorityQueue()
    pq.push(1, Task("critical_fix", 1))
    pq.push(1, Task("urgent_deploy", 1))  # Same priority — no error
    pq.push(2, Task("code_review", 2))

    while pq._heap:
        pri, task = pq.pop()
        print(f"Priority {pri}: {task.name}")
    # Priority 1: critical_fix
    # Priority 1: urgent_deploy
    # Priority 2: code_review
```

**Lesson:** When inserting tuples into `heapq`, the second element is compared when priorities are equal.
If that element is a non-comparable object, a `TypeError` is raised. Insert a counter as a tiebreaker.

---

## 10. Exercises (3 Levels)

### Essential Level

**Exercise 1-1: Reverse a String Using a Stack**

```python
def reverse_string_with_stack(s: str) -> str:
    """
    Reverse a string using a stack.
    Hint: Push all characters, then pop all characters.

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


# Tests
assert reverse_string_with_stack("hello") == "olleh"
assert reverse_string_with_stack("") == ""
assert reverse_string_with_stack("a") == "a"
print("Exercise 1-1: All tests passed!")
```

**Exercise 1-2: Simulate a Hot Potato Game Using a Queue**

```python
from collections import deque
from typing import List


def hot_potato(names: List[str], num_passes: int) -> str:
    """
    Hot Potato game (a variation of the Josephus problem).
    People standing in a circle pass a potato in turn;
    the person holding it after num_passes passes is eliminated.
    The last person remaining wins.

    Hint: Repeatedly dequeue and enqueue.

    >>> hot_potato(["Alice", "Bob", "Charlie", "David"], 3)
    'David'  # Elimination order: Charlie → Alice → Bob → David wins
    """
    queue = deque(names)

    while len(queue) > 1:
        for _ in range(num_passes):
            # Pass the potato = dequeue and enqueue
            queue.append(queue.popleft())
        eliminated = queue.popleft()
        print(f"  Eliminated: {eliminated}")

    winner = queue[0]
    print(f"  Winner: {winner}")
    return winner


# Test
print("=== Hot Potato Game ===")
result = hot_potato(["Alice", "Bob", "Charlie", "David", "Eve"], 3)
```

**Exercise 1-3: Implement a Stack Using Two Queues**

```python
from collections import deque


class StackWithTwoQueues:
    """
    Implement a stack using two queues.
    (The reverse pattern of QueueWithTwoStacks)

    Hint: Reverse the element order on each push.

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
        """O(n) — Transfer all elements on each push."""
        self._q2.append(val)
        while self._q1:
            self._q2.append(self._q1.popleft())
        self._q1, self._q2 = self._q2, self._q1

    def pop(self):
        """O(1)."""
        if not self._q1:
            raise IndexError("pop from empty stack")
        return self._q1.popleft()

    def top(self):
        """O(1)."""
        if not self._q1:
            raise IndexError("top from empty stack")
        return self._q1[0]

    def is_empty(self):
        return len(self._q1) == 0


# Tests
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

### Intermediate Level

**Exercise 2-1: Minimum Removals to Make Valid Parentheses**

```python
def min_remove_to_make_valid(s: str) -> str:
    """
    Remove the minimum number of invalid parentheses and return a valid string.
    LeetCode 1249.

    Algorithm:
      1. Record indices of invalid parentheses on a stack
      2. Exclude characters at recorded indices

    >>> min_remove_to_make_valid("lee(t(c)o)de)")
    'lee(t(c)o)de'
    >>> min_remove_to_make_valid("a)b(c)d")
    'ab(c)d'
    >>> min_remove_to_make_valid("))((")
    ''
    """
    chars = list(s)
    stack = []  # Indices of invalid parentheses

    for i, ch in enumerate(chars):
        if ch == '(':
            stack.append(i)
        elif ch == ')':
            if stack and chars[stack[-1]] == '(':
                stack.pop()
            else:
                stack.append(i)

    # Remove invalid parentheses
    remove_set = set(stack)
    return ''.join(ch for i, ch in enumerate(chars) if i not in remove_set)


# Tests
assert min_remove_to_make_valid("lee(t(c)o)de)") == "lee(t(c)o)de"
assert min_remove_to_make_valid("a)b(c)d") == "ab(c)d"
assert min_remove_to_make_valid("))((") == ""
assert min_remove_to_make_valid("(a(b(c)d)") == "a(b(c)d)"
print("Exercise 2-1: All tests passed!")
```

**Exercise 2-2: Walls and Gates Problem Using a Queue**

```python
from collections import deque
from typing import List

INF = float('inf')


def walls_and_gates(rooms: List[List[int]]) -> List[List[int]]:
    """
    -1 = wall, 0 = gate, INF = empty room.
    Find the distance from each empty room to the nearest gate.
    LeetCode 286. Solved with multi-source BFS.

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

    # Add all gates as starting points to the queue (multi-source BFS)
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


# Tests
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

**Exercise 2-3: Maximum Frequency Stack**

```python
from collections import defaultdict


class FreqStack:
    """
    A stack that pops the most frequently occurring element.
    When frequencies are tied, the most recently pushed element is preferred.
    LeetCode 895.

    Idea:
      - freq[val]: current frequency of each value
      - group[freq]: stack of values pushed at that frequency
      - max_freq: current maximum frequency

    >>> fs = FreqStack()
    >>> for v in [5, 7, 5, 7, 4, 5]:
    ...     fs.push(v)
    >>> fs.pop()  # 5 (freq=3, most frequent)
    5
    >>> fs.pop()  # 7 (freq=2, tied with 5 but 7 was pushed more recently)
    7
    >>> fs.pop()  # 5 (freq=2)
    5
    >>> fs.pop()  # 4 (freq=1, most recent)
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


# Tests
fs = FreqStack()
for v in [5, 7, 5, 7, 4, 5]:
    fs.push(v)
assert fs.pop() == 5  # freq=3
assert fs.pop() == 7  # freq=2
assert fs.pop() == 5  # freq=2
assert fs.pop() == 4  # freq=1
print("Exercise 2-3: All tests passed!")
```

### Advanced Level

**Exercise 3-1: Count Visible People in a Line of Buildings**

```python
from typing import List


def count_visible_people(heights: List[int]) -> List[int]:
    """
    For each position in a line of buildings, count the number of people visible to the right.
    A taller building blocks the view of everything beyond it.
    LeetCode 1944. An advanced monotonic stack problem.

    >>> count_visible_people([10, 6, 8, 5, 11, 9])
    [3, 1, 2, 1, 1, 0]
    """
    n = len(heights)
    result = [0] * n
    stack = []  # Index stack (monotonically decreasing)

    for i in range(n - 1, -1, -1):
        # Buildings shorter than current are visible and removed from stack
        while stack and heights[stack[-1]] < heights[i]:
            stack.pop()
            result[i] += 1
        # One building at least as tall is also visible (but blocks further view)
        if stack:
            result[i] += 1
        stack.append(i)

    return result


# Tests
assert count_visible_people([10, 6, 8, 5, 11, 9]) == [3, 1, 2, 1, 1, 0]
assert count_visible_people([5, 1, 2, 3, 10]) == [4, 1, 1, 1, 0]
print("Exercise 3-1: All tests passed!")
```

**Exercise 3-2: 0-1 BFS (Shortest Path Using a Deque)**

```python
from collections import deque
from typing import List, Tuple

INF = float('inf')


def zero_one_bfs(
    n: int, edges: List[Tuple[int, int, int]], start: int
) -> List[int]:
    """
    Find shortest distances in a graph where edge weights are 0 or 1.
    An algorithm positioned between standard BFS and Dijkstra's.
    Solved in O(V + E) using a deque.

    Algorithm:
      - Weight-0 edge → add to front of deque (process immediately)
      - Weight-1 edge → add to back of deque (process later)

    Args:
        n: Number of vertices (0-indexed)
        edges: List of (from, to, weight) tuples. Weight is 0 or 1.
        start: Source vertex

    Returns:
        List of shortest distances to each vertex

    >>> zero_one_bfs(5, [(0,1,1),(0,2,0),(1,3,1),(2,3,0),(3,4,1)], 0)
    [0, 1, 0, 0, 1]
    """
    # Build adjacency list
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
                    dq.appendleft(v)  # Weight 0 → front
                else:
                    dq.append(v)      # Weight 1 → back

    return [d if d != INF else -1 for d in dist]


# Tests
edges = [(0,1,1), (0,2,0), (1,3,1), (2,3,0), (3,4,1)]
result = zero_one_bfs(5, edges, 0)
assert result == [0, 1, 0, 0, 1]
print("Exercise 3-2: All tests passed!")
```

**Exercise 3-3: Retrieve Median in O(log n) from a Data Stream**

```python
import heapq


class MedianFinder:
    """
    Efficiently retrieve the median from a data stream.
    LeetCode 295. Uses two heaps (max-heap + min-heap).

    Idea:
      - max_heap: the smaller half (max-heap)
      - min_heap: the larger half (min-heap)
      - Always maintain |max_heap| == |min_heap| or |max_heap| == |min_heap| + 1

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
        self._max_heap = []  # Smaller half (negated values for max-heap behavior)
        self._min_heap = []  # Larger half

    def add_num(self, num: int):
        """O(log n)."""
        # First add to max_heap
        heapq.heappush(self._max_heap, -num)
        # Move the max of max_heap to min_heap
        heapq.heappush(self._min_heap, -heapq.heappop(self._max_heap))
        # Rebalance sizes
        if len(self._min_heap) > len(self._max_heap):
            heapq.heappush(
                self._max_heap, -heapq.heappop(self._min_heap)
            )

    def find_median(self) -> float:
        """O(1)."""
        if len(self._max_heap) > len(self._min_heap):
            return float(-self._max_heap[0])
        return (-self._max_heap[0] + self._min_heap[0]) / 2.0


# Tests
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

### Q1: How are stacks and queues related to recursion?

**A:** Recursion uses an implicit stack (the call stack). Each function call pushes a stack frame
(local variables, return address, etc.), and `return` pops it. Therefore,
every recursive algorithm can be rewritten as iteration using an explicit stack.

Typical correspondences:
- **DFS (Depth-First Search)** → recursion or explicit stack
- **BFS (Breadth-First Search)** → queue (difficult to express naturally with recursion)
- **Pre-order, in-order, post-order tree traversal** → recursion or explicit stack

### Q2: Is Python's collections.deque thread-safe?

**A:** Under CPython's GIL (Global Interpreter Lock), `deque.append()` and
`deque.popleft()` are individually executed atomically. However, combinations of
multiple operations (e.g., "check size then pop") are not atomic.

For multi-threaded environments, the following are recommended:
- **`queue.Queue`**: Uses internal locks. Supports blocking
- **`queue.LifoQueue`**: Thread-safe stack
- **`queue.PriorityQueue`**: Thread-safe priority queue

```python
from queue import Queue, LifoQueue, PriorityQueue

# Thread-safe queue
q = Queue(maxsize=100)
q.put("task1")         # Blocking put
item = q.get()         # Blocking get
q.task_done()          # Task completion notification

# Thread-safe stack
stack = LifoQueue()
stack.put("item1")
stack.get()  # LIFO order

# Thread-safe priority queue
pq = PriorityQueue()
pq.put((1, "high"))
pq.put((3, "low"))
pq.get()  # (1, "high")
```

### Q3: Why not implement a priority queue with a sorted array?

**A:** Comparing the complexity of each implementation:

| Approach | Insert | Extract-Min | N inserts + extractions |
|----------|--------|-------------|------------------------|
| Unsorted array | O(1) | O(n) | O(n^2) |
| Sorted array | O(n) | O(1) | O(n^2) |
| Binary heap | O(log n) | O(log n) | **O(n log n)** |

For N = 1,000,000:
- Array approach: approximately 10^12 operations
- Heap approach: approximately 2 x 10^7 operations (roughly 50,000x faster)

Therefore, for typical use cases where inserts and extractions are interleaved,
a heap-based priority queue is overwhelmingly advantageous.

### Q4: Which is faster — deque or list?

**A:** It depends on the type of operation.

| Operation | list | deque | Winner |
|-----------|------|-------|--------|
| Append to end (append) | O(1) amortized | O(1) | Roughly equal |
| Remove from end (pop) | O(1) | O(1) | Roughly equal |
| Prepend (appendleft) | O(n) | O(1) | **deque** |
| Remove from front (popleft) | O(n) | O(1) | **deque** |
| Random access [i] | O(1) | O(n) | **list** |
| Slicing [a:b] | O(k) | O(n) | **list** |
| Memory efficiency | High | Slightly lower | **list** |

Conclusion: Use `deque` for queue use cases (frequent removal from front); use `list` for frequent random access.

### Q5: Where are stacks and queues used in practice?

**A:** Representative real-world use cases:

1. **Web server request queues** — Process incoming requests in FIFO order
2. **Message queues (RabbitMQ, SQS, Kafka)** — Foundation for asynchronous processing
3. **Undo/Redo functionality** — Text editors and image editing software use stacks
4. **Browser back/forward** — History management with two stacks
5. **Compiler syntax analysis** — Stacks used for expression evaluation and parenthesis matching
6. **OS process schedulers** — Queues used for round-robin scheduling
7. **BFS/DFS graph traversal** — Social graphs, pathfinding
8. **Dijkstra's shortest path** — Priority queues are essential


## FAQ

### Q1: What is the most important point when learning this topic?

Gaining practical experience is the most important aspect. Understanding deepens not just through theory, but by actually writing and running code.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving to the next step.

### Q3: How is this applied in real-world development?

The knowledge from this topic is frequently applied in everyday development work. It becomes particularly important during code reviews and architecture design.

---

## 12. Summary

### Key Points of This Guide

| Topic | Key Points |
|-------|------------|
| Stack (LIFO) | DFS, parenthesis matching, Reverse Polish Notation, Undo/Redo. Array-based is recommended |
| Queue (FIFO) | BFS, task scheduling, message queues. `deque` is recommended |
| Circular buffer | Fixed-capacity queue. Embedded systems, kernel buffering |
| Queue with two stacks | Interview favorite. Proving amortized O(1) is important |
| Monotonic stack | Solves "next greater/smaller element" problems in O(n) |
| Priority queue | Implemented with heaps. Essential for Dijkstra, A*, k-way merge |
| MinStack | Store pairs to achieve O(1) minimum retrieval |
| Deque | O(1) at both ends. Used for sliding windows, 0-1 BFS |
| Thread safety | Use `queue.Queue` / `asyncio.Queue` |

### Learning Roadmap

```
Level 1: Understand basic operations
  └→ Implement stack push/pop, queue enqueue/dequeue

Level 2: Master typical applications
  └→ Parenthesis matching, BFS, RPN evaluation

Level 3: Advanced techniques
  └→ Monotonic stack, sliding window maximum, 0-1 BFS

Level 4: Production and design level
  └→ Message queue design, thread safety, priority queue applications
```

---

## Recommended Next Guides

- [Heaps — Binary Heaps and Priority Queue Implementation](./05-heaps.md)
- [Graph Traversal — BFS/DFS](../02-algorithms/02-graph-traversal.md)
- Linked Lists — Node-based fundamental structures

---

## 13. References

1. Cormen, T.H., Leiserson, C.E., Rivest, R.L. & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. — Chapter 10 "Elementary Data Structures": Formal definitions and analysis of stacks, queues, and linked lists
2. Sedgewick, R. & Wayne, K. (2011). *Algorithms* (4th ed.). Addison-Wesley. — Section 1.3 "Bags, Queues, and Stacks": Detailed array and linked list-based implementations
3. Skiena, S.S. (2020). *The Algorithm Design Manual* (3rd ed.). Springer. — Chapter 3 "Data Structures": Practical use patterns for stacks and queues
4. Python Software Foundation. "collections.deque." Python 3 Documentation. https://docs.python.org/3/library/collections.html#collections.deque — Python deque implementation specifications and API reference
5. Python Software Foundation. "queue --- A synchronized queue class." Python 3 Documentation. https://docs.python.org/3/library/queue.html — Thread-safe queue implementations
6. Goodrich, M.T., Tamassia, R. & Goldwasser, M.H. (2013). *Data Structures and Algorithms in Python*. Wiley. — Chapter 6 "Stacks, Queues, and Deques": Python-specific implementations
7. Kleinberg, J. & Tardos, E. (2005). *Algorithm Design*. Pearson. — Chapter 3 "Graphs": Queue-based BFS implementation and shortest path applications

---

## References

- [MDN Web Docs](https://developer.mozilla.org/) - Web technology reference
- [Wikipedia](https://en.wikipedia.org/) - Overview of technical concepts
