# Stacks and Queues

> Stacks (LIFO) underpin function calls and parenthesis matching, while queues (FIFO) underpin task scheduling and BFS.

## Learning Objectives

- [ ] Fully explain the concepts, operations, and time complexities of stacks (LIFO) and queues (FIFO)
- [ ] Implement both array-based and linked-list-based versions from scratch
- [ ] Solve classic problems such as the call stack, parenthesis validation, and Reverse Polish Notation
- [ ] Understand the circular buffer queue implementation and explain its advantages
- [ ] Grasp the mechanism and use cases of the double-ended queue (Deque)
- [ ] Understand the internal structure of a priority queue (heap) and apply it to Dijkstra's algorithm
- [ ] Derive O(n) optimal solutions using monotonic stacks and monotonic queues
- [ ] Design practical applications such as Undo/Redo, BFS/DFS, and task schedulers
- [ ] Explain the internal implementations of CPython's list and Java's ArrayDeque
- [ ] Compare the trade-offs of each data structure and choose the optimal one for each scenario

## Prerequisites

- Basic concepts of pointers and references

---

## 1. Why Stacks and Queues Are Necessary

Arrays and lists are general-purpose data storage, but by imposing constraints on the order in which data is retrieved, we can clarify algorithmic intent, reduce bugs, and guarantee performance.

Stacks and queues are **constrained linear data structures**. By limiting operations, we gain the following benefits:

1. **Explicit intent**: By allowing only push/pop, we guarantee the invariant "the last item inserted is the first to be removed" at the code level
2. **Bug prevention**: By prohibiting random-position access, we eliminate invalid operations at the type or interface level
3. **Complexity guarantees**: All primary operations are structurally guaranteed to be O(1)
4. **Algorithmic correspondence**: The mapping of stacks to DFS and queues to BFS guarantees the correctness of the traversal

### 1.1 Everyday Analogies

```
Stack (LIFO: Last In, First Out)          Queue (FIFO: First In, First Out)
+-----------------------+                 +---------------------------------+
| A stack of books      |                 | A line at a register            |
|                       |                 |                                 |
|  +---+ <- Last book   |                 |  Front -> [A] [B] [C] [D] <- Rear
|  | C |    placed is    |                 |  The first person in line gets  |
|  +---+    removed first|                 |  served first                   |
|  | B |                |                 |                                 |
|  +---+                |                 |  enqueue(E):                    |
|  | A |                |                 |  [A] [B] [C] [D] [E]           |
|  +---+                |                 |                                 |
|  pop -> Removes C     |                 |  dequeue -> Removes A           |
+-----------------------+                 +---------------------------------+
```

Stacks correspond to "a stack of plates", "the browser's back button", and "the Ctrl+Z undo history". Queues correspond to "a printer's print queue", "a call center's waiting line", and "a message queue".

### 1.2 Definition as an Abstract Data Type (ADT)

Stacks and queues are **Abstract Data Types (ADTs)**. That is, they are defined by the set of operations visible from the outside (their interface), regardless of the internal implementation.

**Stack ADT**:
- `push(x)`: Add element x to the top
- `pop()`: Remove and return the top element
- `peek()` / `top()`: Return the top element without removing it
- `is_empty()`: Return whether the stack is empty
- `size()`: Return the number of elements

**Queue ADT**:
- `enqueue(x)`: Add element x to the rear
- `dequeue()`: Remove and return the front element
- `peek()` / `front()`: Return the front element without removing it
- `is_empty()`: Return whether the queue is empty
- `size()`: Return the number of elements

Multiple concrete data structures (arrays, linked lists, circular buffers, etc.) can realize these ADTs, each with its own strengths and weaknesses. We will examine them in the following sections.

---

## 2. Stack

### 2.1 Array-Based Stack Implementation

The simplest stack implementation uses the tail end of a dynamic array (Python's list, Java's ArrayList).

#### Python Implementation

```python
class ArrayStack:
    """Dynamic array-based stack implementation"""

    def __init__(self):
        self._data = []

    def push(self, value):
        """Add element to top: O(1) amortized"""
        self._data.append(value)

    def pop(self):
        """Remove and return top element: O(1) amortized"""
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._data.pop()

    def peek(self):
        """View top element: O(1)"""
        if self.is_empty():
            raise IndexError("peek from empty stack")
        return self._data[-1]

    def is_empty(self):
        """Check if empty: O(1)"""
        return len(self._data) == 0

    def size(self):
        """Number of elements: O(1)"""
        return len(self._data)

    def __repr__(self):
        return f"ArrayStack({self._data})"


# Verification
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

**Complexity Analysis**:
- `push`: O(1) amortized. An O(n) resize occurs only when the internal array runs out of capacity, but since it doubles in size, the amortized cost is O(1).
- `pop`: O(1) amortized. Same reasoning.
- `peek`: O(1). Only an index access to the last element.
- Space complexity: O(n). However, there is overhead from unused capacity in the dynamic array.

#### C Implementation (Fixed-Size Array)

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define MAX_SIZE 1024

typedef struct {
    int data[MAX_SIZE];
    int top;  /* Next push position (= element count) */
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

The C implementation uses a fixed-size array, so a stack overflow check is essential. In embedded systems and kernel modules, this fixed-size approach is preferred to avoid the overhead of memory allocation.

### 2.2 Linked-List-Based Stack Implementation

By using the head of a linked list as the stack top, push/pop can always be achieved in O(1) (worst-case O(1)). Unlike dynamic arrays, no capacity resizing is needed.

```python
class Node:
    """Node for a singly linked list"""
    __slots__ = ('value', 'next')

    def __init__(self, value, next_node=None):
        self.value = value
        self.next = next_node


class LinkedStack:
    """Linked-list-based stack implementation"""

    def __init__(self):
        self._top = None
        self._size = 0

    def push(self, value):
        """Add new node at head: O(1) worst case"""
        self._top = Node(value, self._top)
        self._size += 1

    def pop(self):
        """Remove head node and return its value: O(1) worst case"""
        if self.is_empty():
            raise IndexError("pop from empty stack")
        value = self._top.value
        self._top = self._top.next
        self._size -= 1
        return value

    def peek(self):
        """View head node's value: O(1)"""
        if self.is_empty():
            raise IndexError("peek from empty stack")
        return self._top.value

    def is_empty(self):
        return self._size == 0

    def size(self):
        return self._size

    def __iter__(self):
        """Traverse from top to bottom"""
        current = self._top
        while current is not None:
            yield current.value
            current = current.next

    def __repr__(self):
        return f"LinkedStack([{', '.join(str(v) for v in self)}])"


# Verification
if __name__ == "__main__":
    s = LinkedStack()
    for v in [10, 20, 30]:
        s.push(v)
    print(s)            # LinkedStack([30, 20, 10])
    print(s.peek())     # 30
    print(s.pop())      # 30
    print(s.size())     # 2
```

**Diagram of the linked list approach**:

```
push(10) -> push(20) -> push(30):

  top
   |
   v
 +----+---+    +----+---+    +----+------+
 | 30 | --+--->| 20 | --+--->| 10 | None |
 +----+---+    +----+---+    +----+------+

pop() -> Returns 30 and moves top to the next node:

  top
   |
   v
 +----+---+    +----+------+
 | 20 | --+--->| 10 | None |
 +----+---+    +----+------+
```

### 2.3 Array-Based vs. Linked-List-Based Comparison

| Aspect | Array (Dynamic Array) Based | Linked List Based |
|--------|---------------------------|-------------------|
| push complexity | O(1) amortized (O(n) on resize) | O(1) worst case |
| pop complexity | O(1) amortized | O(1) worst case |
| Memory efficiency | Contiguous memory, cache-friendly | Pointer overhead per node |
| Memory allocation | Bulk allocation on resize | Per-element allocation |
| Cache performance | Excellent (high spatial locality) | Poor (nodes may be scattered) |
| Maximum size | Dynamically expandable (requires contiguous memory) | Can use fragmented memory |
| Implementation simplicity | Very simple (uses language built-ins) | Somewhat complex (node management required) |
| Real-time suitability | Spikes on resize | Perfectly constant time |

**Recommendation**: For general use, the array-based approach is optimal (high cache efficiency and simple implementation). Consider the linked-list approach for real-time systems or scenarios that require worst-case complexity guarantees.

### 2.4 The Call Stack -- Foundation of Program Execution

In program execution, function calls are managed on a stack. This is called the **call stack**.

```python
def factorial(n):
    """Compute factorial recursively"""
    if n <= 1:
        return 1
    return n * factorial(n - 1)

print(factorial(5))  # 120
```

When `factorial(5)` is called, the call stack changes as follows:

```
During invocation (stack grows upward):

  Calling factorial(5)
  +-------------------+
  | factorial(1)      | <- n=1, return 1
  +-------------------+
  | factorial(2)      | <- n=2, waiting: 2 * factorial(1)
  +-------------------+
  | factorial(3)      | <- n=3, waiting: 3 * factorial(2)
  +-------------------+
  | factorial(4)      | <- n=4, waiting: 4 * factorial(3)
  +-------------------+
  | factorial(5)      | <- n=5, waiting: 5 * factorial(4)
  +-------------------+
  | main()            | <- Original caller
  +-------------------+

During return (stack shrinks):

  factorial(1) -> returns 1
  factorial(2) -> returns 2 * 1 = 2
  factorial(3) -> returns 3 * 2 = 6
  factorial(4) -> returns 4 * 6 = 24
  factorial(5) -> returns 5 * 24 = 120
```

Each frame contains the function's local variables, return address, and argument information. If the recursion is too deep, a **stack overflow** occurs (Python's default recursion limit is 1000).

```python
import sys
print(sys.getrecursionlimit())  # 1000

# Changing the recursion limit (not recommended; rewrite to tail recursion or iteration instead)
# sys.setrecursionlimit(10000)
```

### 2.5 Parenthesis Matching Validation

Parenthesis matching is a classic stack application problem.

```python
def is_valid_parentheses(s: str) -> bool:
    """
    Validate whether parentheses are correctly matched.
    Matching pairs: (), [], {}

    Algorithm:
    1. Push opening brackets onto the stack
    2. When a closing bracket appears:
       a. If the stack is empty, return False (no matching opening bracket)
       b. If the top of the stack doesn't match, return False
       c. If it matches, pop
    3. At the end, if the stack is empty, return True (all matched)

    Complexity: O(n) time, O(n) space
    """
    stack = []
    pairs = {')': '(', ']': '[', '}': '{'}

    for char in s:
        if char in '([{':
            stack.append(char)
        elif char in ')]}':
            if not stack:
                return False  # No matching opening bracket
            if stack[-1] != pairs[char]:
                return False  # Bracket type mismatch
            stack.pop()
        # Ignore non-bracket characters

    return len(stack) == 0  # Check for unmatched opening brackets


# Test cases
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

### 2.6 Evaluating Reverse Polish Notation (Postfix Notation)

Reverse Polish Notation (RPN) is a notation system where operators are placed after their operands. Since parentheses are unnecessary, it is well-suited for internal processing in calculators.

```python
def eval_rpn(tokens: list) -> int:
    """
    Evaluate a Reverse Polish Notation expression.

    Example: ["2", "1", "+", "3", "*"] -> (2 + 1) * 3 = 9

    Algorithm:
    1. If the token is a number, push it onto the stack
    2. If it's an operator, pop two operands, compute, and push the result
    3. The last remaining value on the stack is the answer

    Complexity: O(n) time, O(n) space
    """
    stack = []
    operators = {'+', '-', '*', '/'}

    for token in tokens:
        if token in operators:
            b = stack.pop()  # Second operand (pushed later)
            a = stack.pop()  # First operand (pushed earlier)
            if token == '+':
                stack.append(a + b)
            elif token == '-':
                stack.append(a - b)
            elif token == '*':
                stack.append(a * b)
            elif token == '/':
                # C-style truncation toward zero
                stack.append(int(a / b))
        else:
            stack.append(int(token))

    return stack[0]


# Test cases
assert eval_rpn(["2", "1", "+", "3", "*"]) == 9
assert eval_rpn(["4", "13", "5", "/", "+"]) == 6
assert eval_rpn(["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]) == 22
print("All RPN tests passed!")
```

**Step-by-step diagram**:

```
Expression: ["2", "1", "+", "3", "*"]

Step 1: "2" -> push(2)     Stack: [2]
Step 2: "1" -> push(1)     Stack: [2, 1]
Step 3: "+" -> pop 1, 2    Stack: [3]       (2+1=3)
         push(3)
Step 4: "3" -> push(3)     Stack: [3, 3]
Step 5: "*" -> pop 3, 3    Stack: [9]       (3*3=9)
         push(9)

Result: 9
```

---

## 3. Queue

### 3.1 Array-Based Queue Implementation (Naive Approach and Its Problems)

Implementing a queue naively with Python's list results in O(n) dequeue operations.

```python
class NaiveArrayQueue:
    """
    Naive array-based queue (problematic).
    dequeue is O(n) and unsuitable for large volumes of data.
    """

    def __init__(self):
        self._data = []

    def enqueue(self, value):
        """Add to rear: O(1) amortized"""
        self._data.append(value)

    def dequeue(self):
        """Remove from front: *** O(n) *** -- shifts all elements"""
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        return self._data.pop(0)  # This is the O(n) bottleneck!

    def peek(self):
        if self.is_empty():
            raise IndexError("peek from empty queue")
        return self._data[0]

    def is_empty(self):
        return len(self._data) == 0

    def size(self):
        return len(self._data)
```

`pop(0)` is O(n) because after removing the front element, all remaining elements must be shifted forward by one position. This becomes critically slow at around 100,000 elements.

### 3.2 Circular Buffer Queue Implementation

Using a circular buffer (ring buffer), both enqueue and dequeue can be performed in O(1) with a fixed-size array.

```python
class CircularQueue:
    """
    Circular buffer-based queue implementation.
    Achieves O(1) enqueue/dequeue with a fixed-size array.
    """

    def __init__(self, capacity=16):
        self._data = [None] * capacity
        self._capacity = capacity
        self._front = 0   # Index of the front element
        self._size = 0     # Current number of elements

    def enqueue(self, value):
        """Add to rear: O(1)"""
        if self._size == self._capacity:
            self._resize(self._capacity * 2)
        rear = (self._front + self._size) % self._capacity
        self._data[rear] = value
        self._size += 1

    def dequeue(self):
        """Remove from front: O(1)"""
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        value = self._data[self._front]
        self._data[self._front] = None  # Prevent memory leak
        self._front = (self._front + 1) % self._capacity
        self._size -= 1
        return value

    def peek(self):
        """View front element: O(1)"""
        if self.is_empty():
            raise IndexError("peek from empty queue")
        return self._data[self._front]

    def is_empty(self):
        return self._size == 0

    def size(self):
        return self._size

    def _resize(self, new_capacity):
        """Expand the array and rearrange elements"""
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


# Verification
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

**Circular buffer diagram**:

```
Circular buffer with capacity=6:

Initial state: enqueue(A), enqueue(B), enqueue(C)
  +---+---+---+---+---+---+
  | A | B | C |   |   |   |
  +---+---+---+---+---+---+
    ^front          ^rear (next insertion position)
    size=3

dequeue() -> A:
  +---+---+---+---+---+---+
  |   | B | C |   |   |   |
  +---+---+---+---+---+---+
        ^front      ^rear
    size=2

enqueue(D), enqueue(E), enqueue(F):
  +---+---+---+---+---+---+
  |   | B | C | D | E | F |
  +---+---+---+---+---+---+
        ^front              ^rear (=0, wraps to start of array)
    size=5

enqueue(G) -> rear writes to position 0 (circular!):
  +---+---+---+---+---+---+
  | G | B | C | D | E | F |
  +---+---+---+---+---+---+
    ^rear ^front
    size=6 (full -> resize on next enqueue)
```

### 3.3 Linked-List-Based Queue Implementation

```python
class Node:
    __slots__ = ('value', 'next')

    def __init__(self, value, next_node=None):
        self.value = value
        self.next = next_node


class LinkedQueue:
    """
    Linked-list-based queue implementation.
    head is the front (dequeue side), tail is the rear (enqueue side).
    """

    def __init__(self):
        self._head = None
        self._tail = None
        self._size = 0

    def enqueue(self, value):
        """Add to rear: O(1)"""
        new_node = Node(value)
        if self.is_empty():
            self._head = new_node
        else:
            self._tail.next = new_node
        self._tail = new_node
        self._size += 1

    def dequeue(self):
        """Remove from front: O(1)"""
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        value = self._head.value
        self._head = self._head.next
        if self._head is None:
            self._tail = None  # Queue became empty
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


# Verification
if __name__ == "__main__":
    q = LinkedQueue()
    q.enqueue(10)
    q.enqueue(20)
    q.enqueue(30)
    print(q.dequeue())  # 10
    print(q.peek())     # 20
    print(q.size())     # 2
```

### 3.4 Queue Implementation Comparison

| Aspect | Naive Array | Circular Buffer | Linked List |
|--------|-------------|----------------|-------------|
| enqueue | O(1) amortized | O(1) amortized | O(1) worst case |
| dequeue | **O(n)** | O(1) amortized | O(1) worst case |
| Memory efficiency | High | High | Node overhead |
| Cache efficiency | High | High | Low |
| Implementation complexity | Simplest | Moderate | Moderate |
| Worst-case complexity | O(n) | O(n) (on resize) | O(1) |
| When to use | Don't use this | General purpose | Real-time systems |

---

## 4. Double-Ended Queue (Deque)

### 4.1 Deque Concept

A Deque (Double-Ended Queue, pronounced "deck") is a data structure that supports O(1) insertion and deletion at both ends. It combines the functionality of both stacks and queues.

```
         appendleft        append
              |                |
              v                v
  +---+---+---+---+---+---+---+
  |   | A | B | C | D | E |   |
  +---+---+---+---+---+---+---+
              ^                ^
              |                |
          popleft            pop
```

### 4.2 Python's collections.deque

Python's `collections.deque` is internally implemented as a doubly-linked list of fixed-size blocks, enabling O(1) operations at both ends.

```python
from collections import deque

# Basic operations
dq = deque()

# Right-end operations (stack-like / queue-like)
dq.append(1)       # Add to right: O(1)
dq.append(2)
dq.append(3)
print(dq)           # deque([1, 2, 3])

val = dq.pop()      # Remove from right: O(1) -> 3
print(dq)           # deque([1, 2])

# Left-end operations
dq.appendleft(0)    # Add to left: O(1)
print(dq)           # deque([0, 1, 2])

val = dq.popleft()  # Remove from left: O(1) -> 0
print(dq)           # deque([1, 2])

# Fixed-length deque (automatically discards old elements)
history = deque(maxlen=3)
history.append("page1")
history.append("page2")
history.append("page3")
history.append("page4")  # "page1" is automatically removed
print(history)      # deque(['page2', 'page3', 'page4'], maxlen=3)

# Rotation
dq2 = deque([1, 2, 3, 4, 5])
dq2.rotate(2)       # Rotate right by 2
print(dq2)          # deque([4, 5, 1, 2, 3])
dq2.rotate(-2)      # Rotate left by 2
print(dq2)          # deque([1, 2, 3, 4, 5])
```

**Note**: Index access `dq[i]` on a `deque` is O(n) (because the internal structure is a block-linked list). If random access is frequent, use `list` instead.

### 4.3 Deque Use Cases

1. **Sliding window max/min**: Can be solved in O(n) using a monotonic deque (detailed in Section 6)
2. **Palindrome check**: Compare characters removed from both ends
3. **Work stealing**: In multi-threaded task scheduling, a thread takes tasks from one end of its own queue and steals from the opposite end of another thread's queue
4. **Browser back/forward**: Can be managed with a single deque instead of two stacks

---

## 5. Priority Queue

### 5.1 Concept and Motivation

A regular queue is FIFO (first-in, first-out), but a **priority queue** retrieves the element with the highest priority first. It is similar to triage in an emergency room: patients are seen by severity, not arrival order.

Priority Queue ADT:
- `insert(element, priority)`: Add an element with a priority
- `extract_min()` / `extract_max()`: Remove and return the element with the highest priority
- `peek()`: View the element with the highest priority

### 5.2 Heap-Based Implementation

The most efficient implementation of a priority queue is the **binary heap**.

```python
class MinHeap:
    """
    Complete implementation of a min-heap.
    Represents a binary tree using an array.

    Parent: i -> Children: 2i+1, 2i+2
    Child: i -> Parent: (i-1)//2
    """

    def __init__(self):
        self._data = []

    def insert(self, value):
        """Add element: O(log n)"""
        self._data.append(value)
        self._sift_up(len(self._data) - 1)

    def extract_min(self):
        """Remove and return minimum element: O(log n)"""
        if not self._data:
            raise IndexError("extract from empty heap")
        min_val = self._data[0]
        last = self._data.pop()
        if self._data:
            self._data[0] = last
            self._sift_down(0)
        return min_val

    def peek(self):
        """View minimum element: O(1)"""
        if not self._data:
            raise IndexError("peek from empty heap")
        return self._data[0]

    def _sift_up(self, idx):
        """Move added element up to the correct position"""
        while idx > 0:
            parent = (idx - 1) // 2
            if self._data[idx] < self._data[parent]:
                self._data[idx], self._data[parent] = self._data[parent], self._data[idx]
                idx = parent
            else:
                break

    def _sift_down(self, idx):
        """Move root element down to the correct position"""
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


# Verification
if __name__ == "__main__":
    heap = MinHeap()
    for v in [5, 3, 8, 1, 2, 7]:
        heap.insert(v)

    result = []
    while not heap.is_empty():
        result.append(heap.extract_min())
    print(result)  # [1, 2, 3, 5, 7, 8] -- Heapsort
```

**Heap internal structure (diagram)**:

```
Insertion process: [5, 3, 8, 1, 2, 7]

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

Array representation: [1, 2, 7, 5, 3, 8]

Index:  0  1  2  3  4  5
        1  2  7  5  3  8
        |
        +- Minimum is always at index 0
```

### 5.3 Python's heapq Module

```python
import heapq

# Min-heap
heap = []
heapq.heappush(heap, 3)
heapq.heappush(heap, 1)
heapq.heappush(heap, 4)
heapq.heappush(heap, 1)
heapq.heappush(heap, 5)

print(heapq.heappop(heap))  # 1
print(heapq.heappop(heap))  # 1
print(heap[0])               # 3 (peek)

# Max-heap (store negated values)
max_heap = []
for val in [3, 1, 4, 1, 5]:
    heapq.heappush(max_heap, -val)

print(-heapq.heappop(max_heap))  # 5
print(-heapq.heappop(max_heap))  # 4

# Top n largest/smallest elements
data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
print(heapq.nlargest(3, data))   # [9, 6, 5]
print(heapq.nsmallest(3, data))  # [1, 1, 2]

# Heapify an existing list: O(n)
data = [5, 3, 8, 1, 2, 7]
heapq.heapify(data)
print(data)  # [1, 2, 7, 5, 3, 8] (satisfies heap property)
```

### 5.4 Integration with Dijkstra's Algorithm

The most famous application of priority queues is Dijkstra's algorithm (shortest path algorithm).

```python
import heapq
from collections import defaultdict

def dijkstra(graph: dict, start: str) -> dict:
    """
    Dijkstra's algorithm: Single-source shortest paths.
    graph: {node: [(neighbor, weight), ...]}
    Returns: {node: shortest_distance}

    Complexity: O((V + E) log V)  (with binary heap)
    """
    distances = {start: 0}
    pq = [(0, start)]  # (distance, node)
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


# Test graph
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

### 5.5 Top-K Problem

```python
import heapq
from collections import Counter

def top_k_frequent(nums: list, k: int) -> list:
    """
    Return the top K most frequent elements.

    Method: Maintain a min-heap of size K.
    Complexity: O(n log k) -- more efficient than full sort O(n log n) when k << n
    """
    count = Counter(nums)
    return heapq.nlargest(k, count.keys(), key=count.get)


# Test
nums = [1, 1, 1, 2, 2, 3, 3, 3, 3, 4]
print(top_k_frequent(nums, 2))  # [3, 1] -- 3 appears 4 times, 1 appears 3 times
```

### 5.6 Priority Queue Operation Complexities

| Operation | Binary Heap | Sorted Array | Unsorted Array | Fibonacci Heap |
|-----------|------------|--------------|----------------|----------------|
| insert | O(log n) | O(n) | O(1) | O(1) amortized |
| extract_min | O(log n) | O(1) | O(n) | O(log n) amortized |
| peek | O(1) | O(1) | O(n) | O(1) |
| decrease_key | O(log n) | O(n) | O(1) | O(1) amortized |
| Build | O(n) | O(n log n) | O(n) | O(n) |

In practice, the binary heap is the most balanced and relatively easy to implement, making it widely used. The Fibonacci heap is theoretically superior but has large constant factors and complex implementation, so it is rarely used in practice.

---

## 6. Monotonic Stacks and Monotonic Queues

### 6.1 Monotonic Stack

A monotonic stack is a technique that maintains elements in the stack in monotonically increasing (or decreasing) order. It solves problems like "for each element, find the first larger (or smaller) element to its right (or left)" in O(n).

```python
def next_greater_element(arr: list) -> list:
    """
    For each element, find the first "greater element" to its right.
    Return -1 if not found.

    Example: [4, 2, 3, 5, 1] -> [5, 3, 5, -1, -1]

    Algorithm:
    - Store indices on the stack
    - When an element greater than the stack top arrives, it is the NGE for the top
    - Each element is pushed at most once and popped at most once, so O(n)
    """
    n = len(arr)
    result = [-1] * n
    stack = []  # Stack of indices (values maintained in monotonically decreasing order)

    for i in range(n):
        # While the current element is greater than the stack top, finalize answers
        while stack and arr[stack[-1]] < arr[i]:
            idx = stack.pop()
            result[idx] = arr[i]
        stack.append(i)

    return result


# Test
print(next_greater_element([4, 2, 3, 5, 1]))
# [5, 3, 5, -1, -1]

print(next_greater_element([1, 3, 2, 4]))
# [3, 4, 4, -1]
```

**Step-by-step trace (arr = [4, 2, 3, 5, 1])**:

```
i=0, arr[0]=4: stack=[]     -> push 0       stack=[0]
i=1, arr[1]=2: 2<4          -> push 1       stack=[0,1]
i=2, arr[2]=3: 3>arr[1]=2   -> pop 1, result[1]=3
               3<arr[0]=4   -> push 2       stack=[0,2]
i=3, arr[3]=5: 5>arr[2]=3   -> pop 2, result[2]=5
               5>arr[0]=4   -> pop 0, result[0]=5
                             -> push 3       stack=[3]
i=4, arr[4]=1: 1<arr[3]=5   -> push 4       stack=[3,4]

Remaining indices 3,4 stay at -1
result = [5, 3, 5, -1, -1]
```

### 6.2 Monotonic Stack Application: Largest Rectangle in Histogram

LeetCode 84 "Largest Rectangle in Histogram" is a classic monotonic stack application problem.

```python
def largest_rectangle_area(heights: list) -> int:
    """
    Find the largest rectangular area in a histogram.

    Algorithm: Monotonically increasing stack
    - For each bar, efficiently determine how far the rectangle extends
      left and right using that bar's height
    - Maintain a stack of indices in monotonically increasing order
    - When the current bar is shorter than the stack top, pop the top
      and compute the area

    Complexity: O(n)
    """
    stack = []  # Monotonically increasing stack of indices
    max_area = 0
    # Append 0 as a sentinel at the end
    heights_ext = heights + [0]

    for i, h in enumerate(heights_ext):
        while stack and heights_ext[stack[-1]] > h:
            height = heights_ext[stack.pop()]
            # Width: if stack is empty, from left edge to i; otherwise from stack[-1]+1 to i
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)

    return max_area


# Test
print(largest_rectangle_area([2, 1, 5, 6, 2, 3]))  # 10 (5x2)
print(largest_rectangle_area([2, 4]))               # 4
```

### 6.3 Monotonic Queue (Monotonic Deque)

A monotonic queue (monotonic deque) is a technique to find the sliding window maximum/minimum in O(n).

```python
from collections import deque

def max_sliding_window(nums: list, k: int) -> list:
    """
    Find the maximum value in a sliding window of size k.

    Algorithm:
    - Store indices in a deque, maintaining values in monotonically decreasing order
    - The front of the deque always holds the index of the window's maximum
    - Remove indices that have fallen outside the window from the front
    - Remove values smaller than the current value from the rear (they can never be the max)

    Complexity: O(n) time, O(k) space
    """
    if not nums or k == 0:
        return []

    dq = deque()  # Stores indices (values in monotonically decreasing order)
    result = []

    for i in range(len(nums)):
        # Remove indices outside the window from the front
        while dq and dq[0] < i - k + 1:
            dq.popleft()

        # Remove values smaller than the current value from the rear
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()

        dq.append(i)

        # Add to result once the window is complete
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result


# Test
print(max_sliding_window([1, 3, -1, -3, 5, 3, 6, 7], 3))
# [3, 3, 5, 5, 6, 7]
```

**Step-by-step trace (nums=[1,3,-1,-3,5,3,6,7], k=3)**:

```
i=0, num=1:  dq=[0]                        Window not yet complete
i=1, num=3:  3>1 -> pop 0, dq=[1]          Window not yet complete
i=2, num=-1: -1<3 -> dq=[1,2]              result=[3]  (nums[1]=3)
i=3, num=-3: -3<-1 -> dq=[1,2,3]           result=[3,3]
i=4, num=5:  5>-3 -> pop 3; 5>-1 -> pop 2;
             5>3 -> pop 1; dq=[4]          result=[3,3,5]
i=5, num=3:  3<5 -> dq=[4,5]               result=[3,3,5,5]
i=6, num=6:  6>3 -> pop 5; 6>5 -> pop 4;
             dq=[6]                         result=[3,3,5,5,6]
i=7, num=7:  7>6 -> pop 6; dq=[7]          result=[3,3,5,5,6,7]
```

---

## 7. Practical Applications

### 7.1 Undo/Redo Functionality

A text editor's Undo/Redo can be implemented with two stacks.

```python
class UndoRedoEditor:
    """
    Undo/Redo implementation using two stacks.

    undo_stack: History of executed operations
    redo_stack: History of undone operations (for Redo)

    Executing a new operation clears the redo_stack.
    """

    def __init__(self):
        self.text = ""
        self.undo_stack = []
        self.redo_stack = []

    def type_text(self, new_text: str):
        """Type text"""
        self.undo_stack.append(self.text)
        self.redo_stack.clear()  # Clear Redo history on new operation
        self.text = new_text

    def undo(self):
        """Revert to previous state"""
        if not self.undo_stack:
            print("Nothing to undo")
            return
        self.redo_stack.append(self.text)
        self.text = self.undo_stack.pop()

    def redo(self):
        """Cancel an Undo"""
        if not self.redo_stack:
            print("Nothing to redo")
            return
        self.undo_stack.append(self.text)
        self.text = self.redo_stack.pop()


# Demo
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

### 7.2 BFS (Breadth-First Search)

The queue is the core data structure for BFS.

```python
from collections import deque

def bfs(graph: dict, start: str) -> list:
    """
    Breadth-first search. Traverses graph vertices in distance order (level order).

    graph: {node: [neighbors]}
    Returns: List of nodes in visit order

    Complexity: O(V + E)
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


# Test
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
BFS traversal order (graph diagram):

        A (Level 0)
       / \
      B   C (Level 1)
     / \   \
    D   E   F (Level 2)
         \ /
    (E-F are connected by an edge)

Queue changes:
  Initial: [A]
  Process A:  [B, C]           -> Visit A
  Process B:  [C, D, E]        -> Visit B
  Process C:  [D, E, F]        -> Visit C
  Process D:  [E, F]           -> Visit D
  Process E:  [F]              -> Visit E (F already visited)
  Process F:  []               -> Visit F
```

### 7.3 DFS (Depth-First Search) -- Iterative Implementation Using a Stack

DFS can be implemented iteratively using a stack instead of recursion. This avoids stack overflow for deep recursion.

```python
def dfs_iterative(graph: dict, start: str) -> list:
    """
    Iterative DFS using a stack.

    Note: The visit order may differ slightly from recursive DFS
    (because adjacent nodes are processed in reverse order).

    Complexity: O(V + E)
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

        # Push in reverse order so they are processed in original order
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

### 7.4 Task Scheduler

```python
import heapq
from collections import Counter

def least_interval(tasks: list, n: int) -> int:
    """
    Task scheduler: A minimum of n intervals is required between same-type tasks.
    Find the minimum total execution time.

    Example: tasks=["A","A","A","B","B","B"], n=2
    -> A B _ A B _ A B -> 8

    Algorithm:
    1. Use a max-heap to process tasks with the highest remaining count first
    2. Tasks in cooldown are placed in a waiting queue
    3. Tasks whose cooldown has expired are moved back from the waiting queue to the heap
    """
    count = Counter(tasks)
    max_heap = [-cnt for cnt in count.values()]
    heapq.heapify(max_heap)

    time = 0
    cooldown = []  # (time when available again, remaining count)

    while max_heap or cooldown:
        time += 1

        if max_heap:
            cnt = heapq.heappop(max_heap) + 1  # Consume one (negative, so +1)
            if cnt < 0:  # Still remaining
                cooldown.append((time + n, cnt))

        if cooldown and cooldown[0][0] == time:
            _, cnt = cooldown.pop(0)
            heapq.heappush(max_heap, cnt)

    return time


# Test
print(least_interval(["A","A","A","B","B","B"], 2))  # 8
print(least_interval(["A","A","A","B","B","B"], 0))  # 6
```

### 7.5 Message Queue Concepts

Production message queues (RabbitMQ, Apache Kafka, Amazon SQS, etc.) extend the queue concept to distributed systems.

```
Producer                        Consumer
+----------+                   +----------+
| Web App  |--enqueue-->       | Worker 1 |
+----------+            |      +----------+
                        v          ^
+----------+    +-----------+     |
| API Srv  |--> | Message   |--dequeue-->
+----------+    | Queue     |     |
                | (Broker)  |     v
+----------+    +-----------+  +----------+
| Cron Job |--enqueue-->       | Worker 2 |
+----------+                   +----------+

Characteristics:
- Asynchronous processing: Producers don't wait for consumers to finish
- Load balancing: Multiple consumers process messages in parallel
- Reliability: The queue persists messages and redelivers on consumer failure
- Scalability: Increase throughput simply by adding more consumers
```

---

## 8. Standard Library Implementations by Language

### 8.1 Stack/Queue/Priority Queue by Language

| Language | Stack | Queue | Deque | Priority Queue |
|----------|-------|-------|-------|----------------|
| Python | `list` (append/pop) | `collections.deque` | `collections.deque` | `heapq` |
| Java | `ArrayDeque` | `ArrayDeque` / `LinkedList` | `ArrayDeque` | `PriorityQueue` |
| C++ | `std::stack` | `std::queue` | `std::deque` | `std::priority_queue` |
| Go | Slice (append/pop) | `container/list` | None (DIY) | `container/heap` |
| Rust | `Vec` (push/pop) | `VecDeque` | `VecDeque` | `BinaryHeap` |
| JavaScript | `Array` (push/pop) | None (DIY) | None (DIY) | None (DIY) |
| C# | `Stack<T>` | `Queue<T>` | `LinkedList<T>` | `PriorityQueue<T,P>` (.NET 6+) |

### 8.2 Java Usage Examples

```java
import java.util.*;

public class StackQueueExample {
    public static void main(String[] args) {
        // Stack: ArrayDeque is recommended (Stack class is legacy with synchronization overhead)
        Deque<Integer> stack = new ArrayDeque<>();
        stack.push(10);
        stack.push(20);
        stack.push(30);
        System.out.println(stack.peek()); // 30
        System.out.println(stack.pop());  // 30

        // Queue: ArrayDeque is recommended
        Queue<Integer> queue = new ArrayDeque<>();
        queue.offer(10);
        queue.offer(20);
        queue.offer(30);
        System.out.println(queue.peek()); // 10
        System.out.println(queue.poll()); // 10

        // Priority Queue (min-heap)
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        pq.offer(5);
        pq.offer(1);
        pq.offer(3);
        System.out.println(pq.poll()); // 1

        // Priority Queue (max-heap)
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

### 8.3 C++ Usage Examples

```cpp
#include <iostream>
#include <stack>
#include <queue>
#include <deque>

int main() {
    // Stack (default is deque-based)
    std::stack<int> st;
    st.push(10);
    st.push(20);
    st.push(30);
    std::cout << st.top() << std::endl;  // 30
    st.pop();  // Returns void (does not return the value)

    // Queue (default is deque-based)
    std::queue<int> q;
    q.push(10);
    q.push(20);
    q.push(30);
    std::cout << q.front() << std::endl;  // 10
    q.pop();

    // Priority queue (default is max-heap)
    std::priority_queue<int> pq;
    pq.push(5);
    pq.push(1);
    pq.push(3);
    std::cout << pq.top() << std::endl;  // 5
    pq.pop();

    // Min-heap
    std::priority_queue<int, std::vector<int>, std::greater<int>> min_pq;
    min_pq.push(5);
    min_pq.push(1);
    min_pq.push(3);
    std::cout << min_pq.top() << std::endl;  // 1

    return 0;
}
```

---

## 9. Internal Implementation Details

### 9.1 CPython's list

CPython's `list` is implemented as a **dynamic array of pointers**.

```
CPython list internal structure:

PyListObject:
+----------------------+
| ob_refcnt             |  Reference count
| ob_type               |  Pointer to type object
| ob_size (Py_ssize_t)  |  Current element count
| allocated             |  Number of allocated slots
| ob_item (PyObject**)  |------> +----------+
+----------------------+        | ptr[0]   |--> PyObject (element 0)
                                | ptr[1]   |--> PyObject (element 1)
                                | ptr[2]   |--> PyObject (element 2)
                                | ...      |
                                | (unused) |
                                +----------+

Resize strategy:
- new_size = (current_size + current_size >> 3) + (3 if current_size < 9 else 6)
- Grows roughly by a factor of 1.125 (grows more for small lists)
- This ensures append() has O(1) amortized complexity
```

`list.pop()` is O(1) because it simply invalidates the last pointer and decrements `ob_size` by 1. In contrast, `list.pop(0)` is O(n) because all remaining pointers must be shifted forward by one position.

### 9.2 CPython's collections.deque

`collections.deque` is implemented as a **doubly-linked list of fixed-size blocks (64 elements each)**.

```
deque internal structure:

dequeobject:
+--------------------+
| leftblock          |--> +---------------------+
| rightblock         |    | Block (64 slots)    |
| leftindex          |    | +---+---+...+---+   |
| rightindex         |    | |   | A |   | B |   | <-- leftindex, rightindex
| length             |    | +---+---+...+---+   |
| maxlen             |    | prev --> NULL        |
+--------------------+    | next --> Block2      |
                          +---------------------+
                                    |
                                    v
                          +---------------------+
                          | Block (64 slots)    |
                          | +---+---+...+---+   |
                          | | C | D |   |   |   |
                          | +---+---+...+---+   |
                          | prev --> Block1      |
                          | next --> NULL        |
                          +---------------------+

- Each block is a fixed-size array of 64 slots
- Blocks are connected via a doubly-linked list
- appendleft/popleft operate on leftblock's leftindex
- append/pop operate on rightblock's rightindex
- Index access dq[i] is O(n/64) = O(n) (must traverse blocks)
```

This design ensures both-end operations are O(1) while memory within blocks is contiguous and cache-friendly. It runs faster than a pure linked list.

### 9.3 Java's ArrayDeque

Java's `ArrayDeque` is implemented as a **circular buffer**.

```
ArrayDeque internal structure:

+-----------------------------------+
| Object[] elements (power-of-2 size)|
|                                   |
|  [  ] [E] [F] [G] [  ] [  ] [C] [D]
|   0    1   2   3   4    5    6   7
|                              ^head
|              ^tail (next insertion position)
|                                   |
| head = 6  (index of front element)|
| tail = 4  (index for next write)  |
+-----------------------------------+

Queue usage:
  offer(x) -> elements[tail] = x; tail = (tail+1) & (len-1)
  poll()   -> val = elements[head]; head = (head+1) & (len-1)

Stack usage:
  push(x) -> head = (head-1) & (len-1); elements[head] = x
  pop()   -> val = elements[head]; head = (head+1) & (len-1)

- Size is always a power of 2 -> bitwise AND for fast modulo operation
- Doubles in size on resize
- Null elements are not allowed (null is used as a sentinel)
```

In Java, the `Stack` class extends `Vector`, so all methods are `synchronized` (thread-safe but with overhead). In single-threaded environments, `ArrayDeque` is recommended.

---

## 10. Trade-offs and Comparative Analysis

### 10.1 When to Use Stack vs. Queue

| Criterion | Stack (LIFO) | Queue (FIFO) |
|-----------|-------------|-------------|
| Data ordering | Last in, first out | First in, first out |
| Search algorithm | DFS (depth-first) | BFS (breadth-first) |
| Shortest path | Not guaranteed | Guarantees shortest path in unweighted graphs |
| Memory usage (search) | O(max depth) | O(max width) |
| Typical uses | Expression evaluation, parenthesis check, Undo | Task processing, level traversal, shortest path |
| Relationship to recursion | Recursion is an implicit stack | Difficult to express naturally with recursion |
| Space efficiency (graph search) | Favorable for deep but narrow trees | Favorable for wide trees |

### 10.2 Decision Flowchart -- When to Use What

```
Need to store and retrieve data
|
+-- Retrieve the last item inserted first?
|   +-- YES -> Stack
|      +-- Parenthesis matching
|      +-- DFS
|      +-- Undo/Redo
|      +-- Expression evaluation (postfix notation)
|
+-- Retrieve the first item inserted first?
|   +-- YES -> Queue
|      +-- BFS
|      +-- Task scheduling
|      +-- Message processing
|
+-- Need insertion/deletion at both ends?
|   +-- YES -> Deque
|      +-- Sliding window
|      +-- Need both stack and queue functionality
|
+-- Need retrieval based on priority?
|   +-- YES -> Priority Queue (Heap)
|      +-- Shortest path (Dijkstra)
|      +-- Top-K problems
|      +-- Event simulation
|
+-- Also need random access?
    +-- YES -> Use array/list (stack/queue is inappropriate)
```

### 10.3 Comprehensive Data Structure Comparison

| Data Structure | push/enqueue | pop/dequeue | peek | Random Access | Primary Implementation | Memory |
|---------------|-------------|-------------|------|--------------|----------------------|--------|
| Stack (array) | O(1)* | O(1)* | O(1) | O(1) | Dynamic array | Contiguous |
| Stack (linked list) | O(1) | O(1) | O(1) | O(n) | Singly linked list | Scattered |
| Queue (circular buffer) | O(1)* | O(1)* | O(1) | O(1) | Fixed array | Contiguous |
| Queue (linked list) | O(1) | O(1) | O(1) | O(n) | Doubly linked list | Scattered |
| Deque (block list) | O(1) | O(1) | O(1) | O(n) | Block linked list | Semi-contiguous |
| Priority Queue (binary heap) | O(log n) | O(log n) | O(1) | - | Array | Contiguous |

\* Amortized complexity

---

## 11. Anti-Patterns

### 11.1 Anti-Pattern 1: Implementing a Queue with list.pop(0)

```python
# BAD: O(n) dequeue
queue = [1, 2, 3, 4, 5]
val = queue.pop(0)  # O(n) -- shifts all elements

# Repeating pop(0) on 100,000 elements:
# list.pop(0): ~3.5 seconds
# deque.popleft(): ~0.01 seconds
# 350x speed difference!
```

**Correct approach**:

```python
from collections import deque

queue = deque([1, 2, 3, 4, 5])
val = queue.popleft()  # O(1)
```

**Why this bug occurs**: Some Python tutorials introduce "you can make a queue with a list". With small datasets the speed difference is not noticeable, but beyond tens of thousands of elements it becomes a critical performance problem.

### 11.2 Anti-Pattern 2: Causing Stack Overflow with Deep Recursion

```python
# BAD: Deep recursion causes RecursionError
def sum_list(lst, index=0):
    if index == len(lst):
        return 0
    return lst[index] + sum_list(lst, index + 1)

# Works below 1000 elements, but RecursionError at 10000
# sum_list(list(range(10000)))  # RecursionError!
```

**Correct approaches**:

```python
# Method 1: Rewrite as iteration
def sum_list_iterative(lst):
    total = 0
    for val in lst:
        total += val
    return total

# Method 2: Use an explicit stack (for complex recursion)
def sum_list_stack(lst):
    stack = list(range(len(lst)))  # Indices on the stack
    total = 0
    while stack:
        i = stack.pop()
        total += lst[i]
    return total

# Method 3: Use Python built-ins
total = sum(range(10000))
```

**Lesson**: Python's default recursion limit is 1000, and it does not perform tail-call optimization. When deep recursion is expected, rewrite to iteration or an explicit stack.

### 11.3 Anti-Pattern 3: Using Java's Stack Class

```java
// BAD: Stack inherits from Vector with unnecessary synchronization overhead
Stack<Integer> stack = new Stack<>();
stack.push(1);       // synchronized -- sync overhead even in single-threaded code
int val = stack.pop();

// Furthermore, Stack inherits Vector's methods (get(i), set(i, v), etc.)
// allowing non-stack operations
stack.add(0, 999);   // Can insert at the "bottom" of the stack (meaningless)
```

**Correct approach**:

```java
// GOOD: Use ArrayDeque as a stack
Deque<Integer> stack = new ArrayDeque<>();
stack.push(1);
int val = stack.pop();
// ArrayDeque is not thread-safe but faster in single-threaded contexts
// Also, it lacks random-access methods, making it proper as an ADT
```

### 11.4 Anti-Pattern 4: Popping from an Empty Stack/Queue

```python
# BAD: Popping without checking
stack = []
val = stack.pop()  # IndexError: pop from empty list

# BAD: Same issue with queues
from collections import deque
queue = deque()
val = queue.popleft()  # IndexError: pop from an empty deque
```

**Correct approaches**:

```python
# Method 1: Pre-check
if stack:
    val = stack.pop()

# Method 2: Exception handling
try:
    val = stack.pop()
except IndexError:
    val = None  # Default value

# Method 3: Provide safe operations via a wrapper class
class SafeStack:
    def __init__(self):
        self._data = []

    def pop(self, default=None):
        return self._data.pop() if self._data else default
```

---

## 12. Edge Case Analysis

### 12.1 Edge Case 1: Operations with Only One Element in the Stack

```python
stack = [42]

# Verify that peek and pop return the same value
assert stack[-1] == 42   # peek
val = stack.pop()         # pop
assert val == 42
assert len(stack) == 0    # Now empty

# peek/pop on an empty stack should raise an error
# stack[-1]  -> IndexError
# stack.pop() -> IndexError
```

Cases that are problematic in parenthesis validation:

```python
# Input with only a closing bracket
assert is_valid_parentheses(")") == False   # Attempts to pop from an empty stack
assert is_valid_parentheses("(") == False   # Opening bracket remains at the end

# Empty string
assert is_valid_parentheses("") == True     # Empty is considered valid
```

### 12.2 Edge Case 2: Circular Buffer Capacity Boundaries

```python
# Queue with capacity 1
q = CircularQueue(capacity=1)
q.enqueue(10)
# q.enqueue(20)  -> Resize occurs
print(q.dequeue())  # 10

# Fill to exactly capacity, then repeat dequeue/enqueue
q2 = CircularQueue(capacity=4)
for i in range(4):
    q2.enqueue(i)
# Internal state: [0, 1, 2, 3], front=0, size=4

q2.dequeue()  # Removes 0 -> front=1, size=3
q2.dequeue()  # Removes 1 -> front=2, size=2

q2.enqueue(4)  # rear = (2+2) % 4 = 0 writes here -> circular!
q2.enqueue(5)  # rear = (2+3) % 4 = 1 writes here

# Internal state: [4, 5, 2, 3], front=2, size=4
# Logical order: 2, 3, 4, 5
```

### 12.3 Edge Case 3: Elements with the Same Priority in a Heap

```python
import heapq

# When elements have the same priority, Python's heapq compares subsequent tuple elements
# This causes an error for non-comparable objects

class Task:
    def __init__(self, name):
        self.name = name

# BAD: Task objects are not comparable
pq = []
heapq.heappush(pq, (1, Task("A")))
heapq.heappush(pq, (1, Task("B")))  # TypeError: '<' not supported

# Correct approach: Insert a unique counter as a tiebreaker
counter = 0
pq = []

def push_task(priority, task):
    global counter
    heapq.heappush(pq, (priority, counter, task))
    counter += 1

push_task(1, Task("A"))
push_task(1, Task("B"))  # Same priority but distinguished by counter (FIFO order)
priority, _, task = heapq.heappop(pq)
print(task.name)  # "A" (the one pushed first)
```

### 12.4 Edge Case 4: Amortized Analysis of Queue Implemented with Two Stacks

```python
class QueueFromStacks:
    """
    Implements a queue using two stacks.

    Edge case: During dequeue, a transfer from in_stack to out_stack
    occurs. Worst case O(n), but amortized O(1).
    """

    def __init__(self):
        self.in_stack = []   # For enqueue
        self.out_stack = []  # For dequeue

    def enqueue(self, value):
        """Always O(1)"""
        self.in_stack.append(value)

    def dequeue(self):
        """Amortized O(1), worst case O(n)"""
        if not self.out_stack:
            if not self.in_stack:
                raise IndexError("dequeue from empty queue")
            # Transfer all elements from in_stack to out_stack (reverses order)
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


# Edge case tests
q = QueueFromStacks()

# Single element
q.enqueue(1)
assert q.dequeue() == 1

# Interleaved enqueue and dequeue
q.enqueue(1)
q.enqueue(2)
assert q.dequeue() == 1  # in_stack -> out_stack transfer occurs
q.enqueue(3)
assert q.dequeue() == 2  # out_stack still has 2
assert q.dequeue() == 3  # in_stack -> out_stack transfer occurs
```

**Amortized complexity proof**: Each element is pushed once onto in_stack, popped once from in_stack, pushed once onto out_stack, and popped once from out_stack. That is 4 operations per element across n enqueue/dequeue operations, giving O(4n) = O(n) total. Therefore, the amortized cost per operation is O(1).

---

## 13. Exercises

### Exercise 1 (Basic): Evaluate Reverse Polish Notation

**Problem**: Implement a function to evaluate a Reverse Polish Notation (postfix) expression. Operators are `+`, `-`, `*`, `/`. Division truncates toward zero.

**Input example**: `["2", "1", "+", "3", "*"]`
**Output example**: `9` ((2+1)*3)

**Hint**: Refer to Section 2.6. Push numbers onto the stack; on an operator, pop two values, compute, and push the result.

<details>
<summary>Solution</summary>

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

# Test
assert eval_rpn(["2", "1", "+", "3", "*"]) == 9
assert eval_rpn(["4", "13", "5", "/", "+"]) == 6
assert eval_rpn(["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]) == 22
```

</details>

### Exercise 2 (Intermediate): Implement a Queue Using Two Stacks

**Problem**: Implement a queue using only two stacks (push/pop only). Explain why the amortized complexity of enqueue and dequeue is O(1) each.

**Requirements**:
- `enqueue(x)`: Add element x to the queue
- `dequeue()`: Remove and return the front element
- `peek()`: View the front element
- `is_empty()`: Check if empty

<details>
<summary>Solution</summary>

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

# Amortized analysis:
# Each element is pushed once to in_stack, popped once from in_stack,
# pushed once to out_stack, and popped once from out_stack.
# 4 O(1) operations per element -> O(1) amortized per element

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

### Exercise 3 (Advanced): Finding the Median from a Data Stream

**Problem**: Implement a class that computes the median in real-time from a data stream. Use two heaps (a max-heap and a min-heap).

**Requirements**:
- `add_num(num)`: Add a number
- `find_median()`: Return the current median

**Constraint**: `add_num` must be O(log n) and `find_median` must be O(1).

<details>
<summary>Solution</summary>

```python
import heapq

class MedianFinder:
    """
    Manages the median using two heaps.

    max_heap: The smaller half (max-heap, stored as negated values)
    min_heap: The larger half (min-heap)

    Invariants:
    1. All elements in max_heap <= all elements in min_heap
    2. len(max_heap) == len(min_heap) or len(max_heap) == len(min_heap) + 1

    Median:
    - Odd number of elements: top of max_heap
    - Even number of elements: (top of max_heap + top of min_heap) / 2
    """

    def __init__(self):
        self.max_heap = []  # Smaller half (negated values)
        self.min_heap = []  # Larger half

    def add_num(self, num: int):
        # First add to max_heap
        heapq.heappush(self.max_heap, -num)

        # If max_heap's top > min_heap's top, move it
        if self.min_heap and (-self.max_heap[0]) > self.min_heap[0]:
            val = -heapq.heappop(self.max_heap)
            heapq.heappush(self.min_heap, val)

        # Maintain size balance (max_heap can be at most 1 larger)
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


# Test
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

### Exercise 4 (Advanced): Min Stack with O(1) getMin

**Problem**: Implement a stack (MinStack) that supports push, pop, top, and retrieving the minimum value in O(1).

<details>
<summary>Solution</summary>

```python
class MinStack:
    """
    Stack where all operations are O(1).
    Maintains an auxiliary stack that records the minimum at each point
    alongside the main stack.
    """

    def __init__(self):
        self.stack = []
        self.min_stack = []  # Records the minimum at each point

    def push(self, val):
        self.stack.append(val)
        # Push the smaller of val and the current min_stack top
        current_min = min(val, self.min_stack[-1] if self.min_stack else val)
        self.min_stack.append(current_min)

    def pop(self):
        self.stack.pop()
        self.min_stack.pop()

    def top(self):
        return self.stack[-1]

    def get_min(self):
        return self.min_stack[-1]


# Test
ms = MinStack()
ms.push(5)
ms.push(3)
ms.push(7)
assert ms.get_min() == 3
ms.pop()  # Remove 7
assert ms.get_min() == 3
ms.pop()  # Remove 3
assert ms.get_min() == 5
print("MinStack tests passed!")
```

</details>

---

## 14. FAQ

### Q1: Is it safe to use Python's list as a stack?

**A**: Yes, it is fine. Python's `list.append()` and `list.pop()` are both amortized O(1) and appropriate for stack operations. However, keep the following in mind:

1. **Don't use pop(0)**: Removing from the front is O(n). Use `collections.deque` if you need a queue.
2. **Memory shrinkage**: Removing elements with `pop()` does not immediately shrink the internal array. Be cautious in memory-sensitive scenarios.
3. **Thread safety**: `list` is not thread-safe. Use `queue.LifoQueue` in multi-threaded environments.

### Q2: Should I use deque or list?

**A**: It depends on the use case.

- **Stack (single end only)**: `list` is sufficient. `append()` / `pop()` are O(1) with good cache efficiency.
- **Queue (both ends)**: Use `deque`. `popleft()` is O(1), while `list.pop(0)` is O(n).
- **Random access needed**: Use `list`. `deque[i]` is O(n), while `list[i]` is O(1).
- **Fixed-length history**: `deque(maxlen=N)` is convenient. Old elements are automatically discarded.

### Q3: Which is more efficient -- a priority queue or sorting?

**A**: It depends on how many elements you need to extract.

- **Extracting all elements in sorted order**: Sorting O(n log n) and heapsort O(n log n) are equivalent. However, Python's `sorted()` uses Timsort with smaller constant factors and is faster in practice.
- **Extracting only the top K (K << n)**: `heapq.nlargest(k, data)` is O(n log k), more efficient than full sort O(n log n).
- **Data is added dynamically**: A priority queue is appropriate. Insertion into a sorted array is O(n), while insertion into a heap is O(log n).
- **Batch processing**: If all data is available upfront, sorting is often simpler and faster.

### Q4: Why does BFS use a queue and DFS use a stack?

**A**: Because the retrieval order of the data structure matches the exploration order.

- **BFS (breadth-first)**: Explores level by level. Since nodes discovered first (closer nodes) should be processed first, a FIFO queue is needed.
- **DFS (depth-first)**: Dives as deep as possible before backtracking. Since the most recently discovered node (deepest node) should be processed first, a LIFO stack is needed.

Recursive DFS uses the call stack (an implicit stack). Rewriting with an explicit stack avoids stack overflow from deep recursion.

### Q5: How do message queues (RabbitMQ, Kafka) differ from the queue data structure?

**A**: The concept is the same (FIFO element processing), but message queues include additional features for distributed systems.

| Feature | Queue Data Structure | Message Queue |
|---------|---------------------|---------------|
| Persistence | In memory only | Can persist to disk |
| Fault tolerance | Lost on process termination | Recoverable after broker failure |
| Distribution | Within a single process | Across multiple processes over the network |
| Acknowledgment | None | ACK/NACK for message management |
| Scalability | Single machine | Horizontally scalable |

### Q6: What should I do when a stack overflow occurs?

**A**: Consider the following in order:

1. **Rewrite as iteration**: The most reliable approach. Manage the stack explicitly.
2. **Convert to tail recursion**: If the language supports it (Python does not).
3. **Reduce recursion depth**: Improve the algorithm (e.g., memoization, better divide-and-conquer).
4. **Increase the stack size**: Last resort. `sys.setrecursionlimit()` or JVM's `-Xss` option.
5. **Trampoline**: A technique that transforms recursive calls into lazy evaluation.

---


## FAQ

### Q1: What is the most important point when studying this topic?

Gaining practical experience is the most important thing. Understanding deepens not just from theory, but by actually writing code and verifying its behavior.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts covered in this guide before moving on.

### Q3: How is this applied in practice?

Knowledge of this topic is frequently used in everyday development work. It is particularly important during code reviews and architecture design.

---

## 15. Summary

### Overview of Key Data Structures

| Data Structure | Principle | Key Operations | Typical Uses |
|---------------|-----------|---------------|--------------|
| Stack | LIFO (Last In, First Out) | push/pop: O(1) | Call stack, parenthesis check, DFS, Undo/Redo |
| Queue | FIFO (First In, First Out) | enqueue/dequeue: O(1) | BFS, task queue, message queue |
| Deque | Both-end operations | Both-end add/remove: O(1) | Sliding window, work stealing |
| Priority Queue | Priority ordering | push: O(log n), pop: O(log n) | Dijkstra's algorithm, Top-K, task scheduler |
| Monotonic Stack | Monotonic invariant in stack | push/pop: amortized O(1) | Next Greater Element, largest rectangle in histogram |
| Monotonic Queue | Monotonic invariant in deque | Each operation: amortized O(1) | Sliding window max/min |

### Design Decision Summary

1. **"Retrieve the last item inserted first" -> Stack** -- DFS, expression evaluation, Undo
2. **"Retrieve the first item inserted first" -> Queue** -- BFS, task processing, messaging
3. **"Need operations at both ends" -> Deque** -- Sliding window
4. **"Retrieve based on priority" -> Priority Queue** -- Shortest path, Top-K
5. **Array-based is the default implementation** -- Good cache efficiency, simple implementation
6. **Use collections.deque for queues in Python** -- list.pop(0) is O(n) and critically slow
7. **Use ArrayDeque for stacks in Java** -- The Stack class is legacy and deprecated

### Learning Roadmap

```
Fundamentals: Understand stack and queue operations
  |
  +-- Parenthesis matching validation (Stack)
  +-- BFS (Queue)
  +-- Reverse Polish Notation (Stack)
  |
Intermediate: Master applied techniques
  |
  +-- Implement queue with two stacks
  +-- Monotonic stack (Next Greater Element)
  +-- Priority queue (Heap)
  +-- Dijkstra's algorithm
  |
Advanced: Solve advanced problems
  |
  +-- Monotonic queue (Sliding window)
  +-- Largest rectangle in histogram
  +-- Median from data stream
  +-- Task scheduler
```

---

## 16. Recommended Next Reading


---

## 17. References

1. Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapters 6 (Heapsort) and 10 (Elementary Data Structures).
   - Comprehensive theoretical foundations for stacks, queues, and heaps. Rigorous complexity proofs.

2. Sedgewick, R., & Wayne, K. (2011). *Algorithms* (4th ed.). Addison-Wesley. Chapter 1.3 (Bags, Queues, and Stacks) and Chapter 2.4 (Priority Queues).
   - Rich implementation-oriented explanations. Abundant Java implementation examples.

3. Skiena, S. S. (2020). *The Algorithm Design Manual* (3rd ed.). Springer. Chapter 3: Data Structures.
   - Focuses on choosing data structures in practice. Stacks/queues as problem-solving strategies.

4. CPython source code -- `Objects/listobject.c` and `Modules/_collectionsmodule.c`.
   - https://github.com/python/cpython
   - Details of the internal implementations of list and deque. Resize strategy and block structure can be examined here.

5. Knuth, D. E. (1997). *The Art of Computer Programming, Volume 1: Fundamental Algorithms* (3rd ed.). Addison-Wesley. Section 2.2: Linear Lists.
   - Mathematical foundations of stacks and queues. Comprehensive treatment including historical context.

6. LeetCode Problems:
   - [20. Valid Parentheses](https://leetcode.com/problems/valid-parentheses/) -- Parenthesis matching validation
   - [150. Evaluate Reverse Polish Notation](https://leetcode.com/problems/evaluate-reverse-polish-notation/) -- Reverse Polish Notation
   - [232. Implement Queue using Stacks](https://leetcode.com/problems/implement-queue-using-stacks/) -- Queue with two stacks
   - [239. Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/) -- Monotonic queue
   - [84. Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/) -- Monotonic stack
   - [295. Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/) -- Two heaps
   - [155. Min Stack](https://leetcode.com/problems/min-stack/) -- Min stack with O(1) getMin

---

## Recommended Next Reading

- [Hash Tables](./03-hash-tables.md) - Proceed to the next topic

---

## References

- [MDN Web Docs](https://developer.mozilla.org/) - Web technology reference
- [Wikipedia](https://en.wikipedia.org/) - Overview of technical concepts
