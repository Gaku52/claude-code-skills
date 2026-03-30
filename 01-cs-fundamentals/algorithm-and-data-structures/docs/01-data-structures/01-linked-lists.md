# Linked Lists — Singly, Doubly, Circular, and Floyd's Algorithm

> A systematic guide covering the various types of linked lists — a linear data structure that pairs with arrays — along with the classic cycle detection algorithm and practical application patterns.

---



## Learning Objectives

- [ ] Understand the fundamental concepts and terminology
- [ ] Master implementation patterns and best practices
- [ ] Grasp practical application methods
- [ ] Learn the basics of troubleshooting

---

## Prerequisites

Before reading this guide, the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Familiarity with the content of [Arrays and Strings — A Complete Guide to Dynamic Arrays, String Algorithms, and 2D Arrays](./00-arrays-strings.md)

---

## Table of Contents

1. [What is a Linked List?](#1-what-is-a-linked-list)
2. [Types and Structures of Lists](#2-types-and-structures-of-lists)
3. [Basic Implementation — Node and List Classes](#3-basic-implementation--node-and-list-classes)
4. [Core Operations Implementation](#4-core-operations-implementation)
5. [Complete Doubly Linked List Implementation](#5-complete-doubly-linked-list-implementation)
6. [Circular Linked List Implementation and Applications](#6-circular-linked-list-implementation-and-applications)
7. [Floyd's Cycle Detection Algorithm](#7-floyds-cycle-detection-algorithm)
8. [Applied Algorithm Collection](#8-applied-algorithm-collection)
9. [Comparison Tables and Complexity Summary](#9-comparison-tables-and-complexity-summary)
10. [Anti-pattern Collection](#10-anti-pattern-collection)
11. [Exercises — Basic, Intermediate, Advanced](#11-exercises--basic-intermediate-advanced)
12. [FAQ — Frequently Asked Questions](#12-faq--frequently-asked-questions)
13. [Summary](#13-summary)
14. [References](#14-references)

---

## 1. What is a Linked List?

### 1.1 Definition and Intuitive Understanding

A Linked List is a linear data structure in which each element (node) holds data and a pointer (reference), and the pointer points to the next element, thereby representing the overall order of the sequence.

While arrays store data in contiguous memory regions, each node of a linked list can exist at an arbitrary location in memory. This property offers the advantage that insertions and deletions can be completed by simply reassigning pointers.

```
[Array Memory Layout]

  Address:  0x100  0x104  0x108  0x10C  0x110
            +------+------+------+------+------+
  Value:    |  10  |  20  |  30  |  40  |  50  |
            +------+------+------+------+------+
            <-- Contiguous memory region -->

[Linked List Memory Layout]

  0x200          0x3F0          0x580          0x120
  +---------+    +---------+    +---------+    +---------+
  | val: 10 |    | val: 20 |    | val: 30 |    | val: 40 |
  | next:-------> | next:-------> | next:-------> | next:None|
  +---------+    +---------+    +---------+    +---------+
  <-- Scattered across memory -->
```

### 1.2 Why Learn Linked Lists?

Linked lists are essential to computer science foundations for the following reasons:

1. **Pointer Manipulation Fundamentals**: The most basic form of data management using pointers/references. They serve as the foundation for more complex data structures such as trees and graphs.

2. **Understanding Dynamic Memory Management**: Through creating and destroying nodes, you naturally learn the concepts of dynamic memory allocation.

3. **Algorithm Design Training**: They serve as an entry point for versatile algorithm design techniques such as the two-pointer technique (slow/fast) and the dummy node pattern.

4. **Frequent Interview Topic**: One of the most frequently tested categories in coding interviews at major tech companies such as Google, Meta, and Amazon.

### 1.3 Historical Background of Linked Lists

Linked lists were conceived around 1955-1956 by Allen Newell, Cliff Shaw, and Herbert A. Simon while developing IPL (Information Processing Language) at RAND Corporation. Subsequently, list processing was adopted as a core concept in the LISP language (1958, John McCarthy), firmly establishing linked lists in the history of programming languages and data structures.

---

## 2. Types and Structures of Lists

### 2.1 Singly Linked List

The most basic form, where each node holds data and only a "pointer to the next node."

```
Singly Linked List:

  head
   |
   v
  +-----------+     +-----------+     +-----------+     +-----------+
  | val: "A"  |     | val: "B"  |     | val: "C"  |     | val: "D"  |
  | next: ---------->| next: ---------->| next: ---------->| next: None|
  +-----------+     +-----------+     +-----------+     +-----------+

  Characteristics:
  - Only forward traversal is possible (head -> tail)
  - Each node holds only 1 pointer
  - Lowest memory usage
```

**Use Cases**: Stack implementation, chaining in hash tables, embedded systems with memory constraints.

### 2.2 Doubly Linked List

Each node holds two pointers: a forward pointer (next) and a backward pointer (prev).

```
Doubly Linked List:

  head                                                          tail
   |                                                             |
   v                                                             v
  +--------------+     +--------------+     +--------------+    +--------------+
  | prev: None   |     | prev: ---------------<| prev: ---------------<| prev: ----------------<
  | val: "A"     |     | val: "B"     |     | val: "C"      |   | val: "D"     |
  | next: ------------->| next: ------------->| next: ------------->| next: None   |
  +--------------+     +--------------+     +--------------+    +--------------+

  Characteristics:
  - Bidirectional traversal (both forward and backward) is possible
  - Each node holds 2 pointers (increased memory usage)
  - Moving forward or backward from any node is O(1)
```

**Use Cases**: LRU cache, cursor movement in text editors, browser "Back/Forward" navigation.

### 2.3 Circular Linked List

The tail node's pointer points to the head node, forming a ring. Two variants exist: singly circular linked list and doubly circular linked list.

```
Singly Circular Linked List:

  head
   |
   v
  +-----------+     +-----------+     +-----------+
  | val: "A"  |     | val: "B"  |     | val: "C"  |
  | next: ---------->| next: ---------->| next: --+  |
  +-----------+     +-----------+     +---------|-+
   ^                                             |
   +---------------------------------------------+

Doubly Circular Linked List:

  +-----------------------------------------------------------------+
  |                                                                 |
  |  +--------------+     +--------------+     +--------------+   |
  +> | prev: --+    |     | prev: ---------------< | prev: ---------------<+
     | val: "A"|    |     | val: "B"     |     | val: "C"     |
     | next: ------------->| next: ------------->| next: --+    |
     +---------|----|     +--------------+     +---------|----|
               |                                          |
               +------------------------------------------+

  Characteristics:
  - No null references since the tail node points to the head
  - The entire list can be traversed starting from any node
  - Caution is needed for traversal termination conditions (risk of infinite loops)
```

**Use Cases**: Round-robin scheduling, circular buffers, turn management in multiplayer games.

### 2.4 Skip List — An Advanced Variant

A skip list is a probabilistic data structure that layers multiple levels of linked lists, where upper levels act as "express lanes" to achieve O(log n) search. This guide focuses on basic linked lists, but skip lists are worth knowing as an advanced topic.

```
Conceptual diagram of a skip list:

  Level 3:  head --------------------------------- [50] ---------------------- tail
  Level 2:  head ---------- [20] ---------------- [50] ---- [70] ----------- tail
  Level 1:  head -- [10]- [20] -- [30] -- [50] -- [60]- [70] -- [90] -- tail
  Level 0:  head -- [10]- [20] -- [30] -- [40] -- [50] -- [60]- [70] -- [80] -- [90] -- tail
```

**Use Cases**: Redis Sorted Sets, LevelDB/RocksDB MemTable.

---

## 3. Basic Implementation -- Node and List Classes

### 3.1 Singly Linked List Node Class

```python
class ListNode:
    """A node of a singly linked list.

    Attributes:
        val: The value held by the node. Can store any type.
        next: Reference to the next node. None for the tail node.
    """

    __slots__ = ('val', 'next')  # Memory optimization

    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __repr__(self):
        """String representation for debugging"""
        return f"ListNode({self.val})"

    def __eq__(self, other):
        """Value equality comparison (note: not node identity)"""
        if not isinstance(other, ListNode):
            return NotImplemented
        return self.val == other.val
```

### 3.2 Doubly Linked List Node Class

```python
class DoublyListNode:
    """A node of a doubly linked list.

    Attributes:
        val: The value held by the node.
        prev: Reference to the previous node. None for the head node.
        next: Reference to the next node. None for the tail node.
    """

    __slots__ = ('val', 'prev', 'next')

    def __init__(self, val=0, prev=None, next=None):
        self.val = val
        self.prev = prev
        self.next = next

    def __repr__(self):
        return f"DoublyListNode({self.val})"
```

### 3.3 Memory Optimization with `__slots__`

In Python, a `__dict__` is generated by default for class instances, but when creating a large number of nodes as in linked lists, using `__slots__` can reduce memory by approximately 40-60% per node.

```python
import sys

class NodeWithDict:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class NodeWithSlots:
    __slots__ = ('val', 'next')
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# Memory usage comparison
node_dict = NodeWithDict(42)
node_slots = NodeWithSlots(42)

print(f"With __dict__: {sys.getsizeof(node_dict) + sys.getsizeof(node_dict.__dict__)} bytes")
# Expected output: With __dict__: 152 bytes (Python 3.12)

print(f"With __slots__: {sys.getsizeof(node_slots)} bytes")
# Expected output: With __slots__: 56 bytes (Python 3.12)
```

### 3.4 Utility Functions — List Construction and Display

The following helper functions are defined for repeated use in implementations and tests throughout this guide.

```python
from typing import Optional, List


class ListNode:
    """Node for a singly linked list"""
    __slots__ = ('val', 'next')

    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __repr__(self):
        return f"ListNode({self.val})"


def build_list(values: List[int]) -> Optional[ListNode]:
    """Build a singly linked list from an array.

    Args:
        values: Array of values to store in the list.

    Returns:
        The head node of the list. None if the array is empty.

    Examples:
        >>> head = build_list([1, 2, 3, 4, 5])
        >>> list_to_string(head)
        '1 -> 2 -> 3 -> 4 -> 5 -> None'
    """
    if not values:
        return None
    head = ListNode(values[0])
    current = head
    for val in values[1:]:
        current.next = ListNode(val)
        current = current.next
    return head


def list_to_array(head: Optional[ListNode]) -> List[int]:
    """Convert a linked list to an array.

    Args:
        head: The head node of the list.

    Returns:
        An array containing all values in the list.
    """
    result = []
    current = head
    while current:
        result.append(current.val)
        current = current.next
    return result


def list_to_string(head: Optional[ListNode]) -> str:
    """Convert a linked list to its string representation.

    Args:
        head: The head node of the list.

    Returns:
        A string in the format '1 -> 2 -> 3 -> None'.
    """
    parts = []
    current = head
    while current:
        parts.append(str(current.val))
        current = current.next
    parts.append("None")
    return " -> ".join(parts)


def list_length(head: Optional[ListNode]) -> int:
    """Return the length of a linked list.

    Args:
        head: The head node of the list.

    Returns:
        The number of nodes.
    """
    count = 0
    current = head
    while current:
        count += 1
        current = current.next
    return count


# Verification
if __name__ == "__main__":
    head = build_list([10, 20, 30, 40, 50])
    print(list_to_string(head))        # 10 -> 20 -> 30 -> 40 -> 50 -> None
    print(list_to_array(head))          # [10, 20, 30, 40, 50]
    print(f"Length: {list_length(head)}")  # Length: 5
```

---

## 4. Core Operations Implementation

### 4.1 Complete Singly Linked List Implementation

Here we present a singly linked list class equipped with all basic operations including insertion, deletion, search, and reversal. Each method includes complexity annotations and internal operation diagrams.

```python
from typing import Optional, List, Iterator


class ListNode:
    """Node for a singly linked list"""
    __slots__ = ('val', 'next')

    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __repr__(self):
        return f"ListNode({self.val})"


class SinglyLinkedList:
    """Complete implementation of a singly linked list.

    Uses a dummy head and size tracking to handle edge cases uniformly.

    Attributes:
        _dummy: Dummy head node (sentinel node).
        _size: Current number of elements in the list.
    """

    def __init__(self):
        """Initialize an empty list."""
        self._dummy = ListNode(0)  # Sentinel node
        self._size = 0

    @property
    def head(self) -> Optional[ListNode]:
        """Return the actual head node (dummy head is hidden)."""
        return self._dummy.next

    def __len__(self) -> int:
        """Return the length of the list in O(1)."""
        return self._size

    def __bool__(self) -> bool:
        """Determine whether the list is non-empty."""
        return self._size > 0

    def __iter__(self) -> Iterator:
        """Iterate over the elements of the list."""
        current = self._dummy.next
        while current:
            yield current.val
            current = current.next

    def __repr__(self) -> str:
        """Return the string representation of the list."""
        values = list(self)
        return f"SinglyLinkedList({values})"

    def __contains__(self, val) -> bool:
        """Support for the in operator. O(n)."""
        return self.search(val)

    # --- Insertion Operations ---

    def prepend(self, val) -> None:
        """Insert an element at the head. O(1).

        Args:
            val: The value to insert.

        Diagram:
            Before: dummy -> [A] -> [B] -> None
            After:  dummy -> [X] -> [A] -> [B] -> None
        """
        new_node = ListNode(val, self._dummy.next)
        self._dummy.next = new_node
        self._size += 1

    def append(self, val) -> None:
        """Insert an element at the tail. O(n).

        Args:
            val: The value to insert.

        Diagram:
            Before: dummy -> [A] -> [B] -> None
            After:  dummy -> [A] -> [B] -> [X] -> None
        """
        current = self._dummy
        while current.next:
            current = current.next
        current.next = ListNode(val)
        self._size += 1

    def insert_at(self, index: int, val) -> None:
        """Insert an element at the specified position. O(n).

        Args:
            index: Insertion position (0-indexed).
            val: The value to insert.

        Raises:
            IndexError: If the index is out of range.
        """
        if index < 0 or index > self._size:
            raise IndexError(f"Index {index} out of range [0, {self._size}]")

        prev = self._dummy
        for _ in range(index):
            prev = prev.next
        new_node = ListNode(val, prev.next)
        prev.next = new_node
        self._size += 1

    def insert_after(self, target_val, new_val) -> bool:
        """Insert immediately after a node with the specified value. O(n).

        Args:
            target_val: The value to search for.
            new_val: The value to insert.

        Returns:
            True if insertion was successful, False if target_val was not found.
        """
        current = self._dummy.next
        while current:
            if current.val == target_val:
                new_node = ListNode(new_val, current.next)
                current.next = new_node
                self._size += 1
                return True
            current = current.next
        return False

    # --- Deletion Operations ---

    def delete(self, val) -> bool:
        """Delete the first node with the specified value. O(n).

        Args:
            val: The value to delete.

        Returns:
            True if deletion was successful, False if the value was not found.

        Diagram:
            Before: dummy -> [A] -> [B] -> [C] -> None
            delete("B"):
            After:  dummy -> [A] ---------> [C] -> None
        """
        prev = self._dummy
        current = self._dummy.next
        while current:
            if current.val == val:
                prev.next = current.next
                self._size -= 1
                return True
            prev = current
            current = current.next
        return False

    def delete_at(self, index: int):
        """Delete the element at the specified position and return its value. O(n).

        Args:
            index: Deletion position (0-indexed).

        Returns:
            The value of the deleted node.

        Raises:
            IndexError: If the index is out of range.
        """
        if index < 0 or index >= self._size:
            raise IndexError(f"Index {index} out of range [0, {self._size - 1}]")

        prev = self._dummy
        for _ in range(index):
            prev = prev.next
        target = prev.next
        prev.next = target.next
        self._size -= 1
        return target.val

    def delete_all(self, val) -> int:
        """Delete all nodes with the specified value. O(n).

        Args:
            val: The value to delete.

        Returns:
            The number of deleted nodes.
        """
        prev = self._dummy
        current = self._dummy.next
        count = 0
        while current:
            if current.val == val:
                prev.next = current.next
                self._size -= 1
                count += 1
            else:
                prev = current
            current = current.next if current.val == val else prev.next.next if prev.next else None
        # The above traversal becomes complex, so here is a simpler implementation:
        return count

    def pop_front(self):
        """Remove and return the head element. O(1).

        Returns:
            The value of the head node.

        Raises:
            IndexError: If the list is empty.
        """
        if not self._dummy.next:
            raise IndexError("pop from empty list")
        target = self._dummy.next
        self._dummy.next = target.next
        self._size -= 1
        return target.val

    # --- Search and Access Operations ---

    def search(self, val) -> bool:
        """Determine whether a value exists in the list. O(n).

        Args:
            val: The value to search for.

        Returns:
            True if the value is found.
        """
        current = self._dummy.next
        while current:
            if current.val == val:
                return True
            current = current.next
        return False

    def get_at(self, index: int):
        """Get the value at the specified position. O(n).

        Args:
            index: Position to retrieve (0-indexed).

        Returns:
            The value of the node at that position.

        Raises:
            IndexError: If the index is out of range.
        """
        if index < 0 or index >= self._size:
            raise IndexError(f"Index {index} out of range [0, {self._size - 1}]")

        current = self._dummy.next
        for _ in range(index):
            current = current.next
        return current.val

    def index_of(self, val) -> int:
        """Return the first occurrence position of a value. O(n).

        Args:
            val: The value to search for.

        Returns:
            The position (0-indexed). -1 if not found.
        """
        current = self._dummy.next
        idx = 0
        while current:
            if current.val == val:
                return idx
            current = current.next
            idx += 1
        return -1

    # --- Transformation Operations ---

    def reverse(self) -> None:
        """Reverse the list in-place. O(n) time, O(1) space.

        Diagram:
            Before: dummy -> [1] -> [2] -> [3] -> None
            After:  dummy -> [3] -> [2] -> [1] -> None

        Detailed reversal process:
            Step 0: prev=None,  curr=[1]->[2]->[3]
            Step 1: prev=[1],   curr=[2]->[3]     ([1]->None)
            Step 2: prev=[2],   curr=[3]          ([2]->[1]->None)
            Step 3: prev=[3],   curr=None          ([3]->[2]->[1]->None)
        """
        prev = None
        current = self._dummy.next
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        self._dummy.next = prev

    def to_list(self) -> List:
        """Convert the list to a Python list (array). O(n)."""
        return list(self)

    def clear(self) -> None:
        """Clear the list. O(1)."""
        self._dummy.next = None
        self._size = 0

    @classmethod
    def from_list(cls, values: List) -> 'SinglyLinkedList':
        """Build a linked list from a Python list (array). O(n).

        Args:
            values: Array of values.

        Returns:
            The constructed SinglyLinkedList.
        """
        linked_list = cls()
        if not values:
            return linked_list
        current = linked_list._dummy
        for val in values:
            current.next = ListNode(val)
            current = current.next
            linked_list._size += 1
        return linked_list


# --- Verification Tests ---

if __name__ == "__main__":
    # Build the list
    sll = SinglyLinkedList.from_list([10, 20, 30, 40, 50])
    print(f"Initial state: {sll}")
    # Output: SinglyLinkedList([10, 20, 30, 40, 50])

    # Prepend
    sll.prepend(5)
    print(f"prepend(5): {sll}")
    # Output: SinglyLinkedList([5, 10, 20, 30, 40, 50])

    # Append
    sll.append(60)
    print(f"append(60): {sll}")
    # Output: SinglyLinkedList([5, 10, 20, 30, 40, 50, 60])

    # Insert at position
    sll.insert_at(3, 25)
    print(f"insert_at(3, 25): {sll}")
    # Output: SinglyLinkedList([5, 10, 20, 25, 30, 40, 50, 60])

    # Delete
    sll.delete(25)
    print(f"delete(25): {sll}")
    # Output: SinglyLinkedList([5, 10, 20, 30, 40, 50, 60])

    # Reverse
    sll.reverse()
    print(f"reverse(): {sll}")
    # Output: SinglyLinkedList([60, 50, 40, 30, 20, 10, 5])

    # Search
    print(f"search(30): {sll.search(30)}")  # True
    print(f"search(99): {sll.search(99)}")  # False

    # Iteration
    print(f"List length: {len(sll)}")  # 7
    print(f"30 in sll: {30 in sll}")  # True

    # Index access
    print(f"get_at(2): {sll.get_at(2)}")  # 40
    print(f"index_of(30): {sll.index_of(30)}")  # 3

    # Pop
    val = sll.pop_front()
    print(f"pop_front(): {val}, list: {sll}")
    # Output: pop_front(): 60, list: SinglyLinkedList([50, 40, 30, 20, 10, 5])
```

### 4.2 Detailed Diagram of List Reversal

List reversal is one of the most important algorithms in linked list operations. Let us trace the pointer reassignment step by step to understand it.

```
Full steps of the reversal algorithm:

  [Initial State]
  prev = None
  curr = [1]

  None    [1] --> [2] --> [3] --> [4] --> None
   ^      ^
  prev   curr

  --------------------------------------------------

  [Step 1] next_node = curr.next = [2]
             curr.next = prev (= None)
             prev = curr (= [1])
             curr = next_node (= [2])

  None <-- [1]    [2] --> [3] --> [4] --> None
            ^      ^
           prev   curr

  --------------------------------------------------

  [Step 2] next_node = curr.next = [3]
             curr.next = prev (= [1])
             prev = curr (= [2])
             curr = next_node (= [3])

  None <-- [1] <-- [2]    [3] --> [4] --> None
                    ^      ^
                   prev   curr

  --------------------------------------------------

  [Step 3] next_node = curr.next = [4]
             curr.next = prev (= [2])
             prev = curr (= [3])
             curr = next_node (= [4])

  None <-- [1] <-- [2] <-- [3]    [4] --> None
                            ^      ^
                           prev   curr

  --------------------------------------------------

  [Step 4] next_node = curr.next = None
             curr.next = prev (= [3])
             prev = curr (= [4])
             curr = next_node (= None)

  None <-- [1] <-- [2] <-- [3] <-- [4]    None
                                     ^      ^
                                    prev   curr

  --------------------------------------------------

  [Complete] Loop ends when curr == None
  New head = prev = [4]

  Result: [4] --> [3] --> [2] --> [1] --> None
```

### 4.3 Recursive Reversal

Reversal can also be implemented recursively. The recursive version is easier to understand, but carries a risk of stack overflow for long lists due to O(n) stack depth.

```python
from typing import Optional


class ListNode:
    __slots__ = ('val', 'next')
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def reverse_recursive(head: Optional[ListNode]) -> Optional[ListNode]:
    """Recursively reverse a linked list.

    Time Complexity: O(n)
    Space Complexity: O(n) — recursion stack

    Args:
        head: The head node of the list.

    Returns:
        The head node of the reversed list.

    How it works:
        1. Base case: return head if head is None or head.next is None
        2. Recursively reverse the remaining portion
        3. Create a reverse link with head.next.next = head
        4. Sever the original link with head.next = None
    """
    # Base case
    if not head or not head.next:
        return head

    # Recurse: reverse everything after head
    new_head = reverse_recursive(head.next)

    # Create reverse link
    head.next.next = head  # Link from the next node back to self
    head.next = None       # Sever the link from self to next

    return new_head


def build_list(values):
    if not values:
        return None
    head = ListNode(values[0])
    curr = head
    for v in values[1:]:
        curr.next = ListNode(v)
        curr = curr.next
    return head


def list_to_string(head):
    parts = []
    curr = head
    while curr:
        parts.append(str(curr.val))
        curr = curr.next
    return " -> ".join(parts) + " -> None"


# Verification
if __name__ == "__main__":
    original = build_list([1, 2, 3, 4, 5])
    print(f"Before reversal: {list_to_string(original)}")
    # Output: Before reversal: 1 -> 2 -> 3 -> 4 -> 5 -> None

    reversed_head = reverse_recursive(original)
    print(f"After reversal: {list_to_string(reversed_head)}")
    # Output: After reversal: 5 -> 4 -> 3 -> 2 -> 1 -> None
```

### 4.4 The Importance of the Dummy Head (Sentinel Node) Pattern

The dummy head (sentinel node) is the most powerful technique for eliminating edge cases in linked list operations.

```
Without dummy head (requires special handling for the head node):

  Case 1: Head deletion        Case 2: Middle deletion
  head                         head
   |                            |
   v                            v
  [X] -> [B] -> [C]           [A] -> [X] -> [C]
   |                                  |
  head = head.next             prev.next = curr.next
  (code branches)

With dummy head (uniform processing):

  Data always starts after the dummy head:
  dummy -> [A] -> [B] -> [C] -> None

  Both head deletion and middle deletion use the same logic:
  prev.next = curr.next
  head = dummy.next
```

This pattern is effective in all situations, including merging two lists in merge sort, conditional node deletion, and partition operations.

---

## 5. Complete Doubly Linked List Implementation

A doubly linked list enables bidirectional traversal (forward and backward) and O(1) node deletion. It is an essential data structure for implementing LRU caches.

### 5.1 Doubly Linked List Class

```python
from typing import Optional, List, Iterator


class DoublyListNode:
    """Node for a doubly linked list"""
    __slots__ = ('val', 'prev', 'next')

    def __init__(self, val=0, prev=None, next=None):
        self.val = val
        self.prev = prev
        self.next = next

    def __repr__(self):
        return f"DoublyListNode({self.val})"


class DoublyLinkedList:
    """Complete implementation of a doubly linked list.

    Dummy nodes (sentinels) are placed at both the head and tail,
    allowing all insertion and deletion operations to be handled uniformly.

    Structure:
        head_sentinel <-> [node1] <-> [node2] <-> ... <-> tail_sentinel
    """

    def __init__(self):
        """Initialize an empty list. Connect the dummy head and dummy tail."""
        self._head = DoublyListNode(0)   # Dummy head
        self._tail = DoublyListNode(0)   # Dummy tail
        self._head.next = self._tail
        self._tail.prev = self._head
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def __bool__(self) -> bool:
        return self._size > 0

    def __iter__(self) -> Iterator:
        """Forward iteration"""
        current = self._head.next
        while current != self._tail:
            yield current.val
            current = current.next

    def __reversed__(self) -> Iterator:
        """Backward iteration"""
        current = self._tail.prev
        while current != self._head:
            yield current.val
            current = current.prev

    def __repr__(self) -> str:
        return f"DoublyLinkedList({list(self)})"

    # --- Internal Helpers ---

    def _insert_between(self, val, predecessor: DoublyListNode,
                        successor: DoublyListNode) -> DoublyListNode:
        """Insert a node between predecessor and successor. O(1).

        Diagram:
            Before: [pred] <-> [succ]
            After:  [pred] <-> [new] <-> [succ]
        """
        new_node = DoublyListNode(val, predecessor, successor)
        predecessor.next = new_node
        successor.prev = new_node
        self._size += 1
        return new_node

    def _remove_node(self, node: DoublyListNode):
        """Remove the specified node from the list. O(1).

        Diagram:
            Before: [pred] <-> [node] <-> [succ]
            After:  [pred] <-> [succ]
        """
        predecessor = node.prev
        successor = node.next
        predecessor.next = successor
        successor.prev = predecessor
        self._size -= 1
        return node.val

    # --- Public API ---

    def prepend(self, val) -> DoublyListNode:
        """Insert at the head. O(1)."""
        return self._insert_between(val, self._head, self._head.next)

    def append(self, val) -> DoublyListNode:
        """Insert at the tail. O(1)."""
        return self._insert_between(val, self._tail.prev, self._tail)

    def pop_front(self):
        """Remove and return the head element. O(1)."""
        if self._size == 0:
            raise IndexError("pop from empty list")
        return self._remove_node(self._head.next)

    def pop_back(self):
        """Remove and return the tail element. O(1)."""
        if self._size == 0:
            raise IndexError("pop from empty list")
        return self._remove_node(self._tail.prev)

    def remove(self, node: DoublyListNode):
        """Remove the specified node in O(1). Requires a node reference."""
        return self._remove_node(node)

    def move_to_front(self, node: DoublyListNode) -> None:
        """Move the specified node to the head of the list. O(1).
        Used in LRU caches to move a recently accessed element to the front.
        """
        self._remove_node(node)
        self._insert_between(node.val, self._head, self._head.next)

    def peek_front(self):
        """Peek at the head value (without removal). O(1)."""
        if self._size == 0:
            raise IndexError("peek from empty list")
        return self._head.next.val

    def peek_back(self):
        """Peek at the tail value (without removal). O(1)."""
        if self._size == 0:
            raise IndexError("peek from empty list")
        return self._tail.prev.val

    @classmethod
    def from_list(cls, values: List) -> 'DoublyLinkedList':
        """Build a doubly linked list from an array."""
        dll = cls()
        for val in values:
            dll.append(val)
        return dll


# Verification
if __name__ == "__main__":
    dll = DoublyLinkedList.from_list([10, 20, 30, 40, 50])
    print(f"Forward traversal: {list(dll)}")
    # Output: Forward traversal: [10, 20, 30, 40, 50]

    print(f"Backward traversal: {list(reversed(dll))}")
    # Output: Backward traversal: [50, 40, 30, 20, 10]

    dll.prepend(5)
    dll.append(60)
    print(f"prepend(5), append(60): {list(dll)}")
    # Output: prepend(5), append(60): [5, 10, 20, 30, 40, 50, 60]

    print(f"pop_front(): {dll.pop_front()}")  # 5
    print(f"pop_back(): {dll.pop_back()}")    # 60
    print(f"Result: {list(dll)}")
    # Output: Result: [10, 20, 30, 40, 50]
```

### 5.2 LRU Cache Implementation — Doubly Linked List + HashMap

The LRU (Least Recently Used) cache is the most well-known application of doubly linked lists. Combined with a hash map, it achieves O(1) for both get and put operations.

```python
from typing import Optional


class DoublyListNode:
    __slots__ = ('key', 'val', 'prev', 'next')
    def __init__(self, key=0, val=0, prev=None, next=None):
        self.key = key
        self.val = val
        self.prev = prev
        self.next = next


class LRUCache:
    """LRU Cache — Achieves O(1) get/put.

    Structure:
        HashMap: key -> DoublyListNode (O(1) lookup)
        DoublyLinkedList: Manages usage order (head = most recent, tail = oldest)

    Diagram:
        HashMap                DoublyLinkedList
        +-------------+       head <-> [A] <-> [B] <-> [C] <-> tail
        | key_A -> [A] |               newest              oldest
        | key_B -> [B] |
        | key_C -> [C] |       On capacity overflow: delete tail.prev ([C])
        +-------------+
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # key -> DoublyListNode

        # Sentinel pattern with dummy nodes
        self._head = DoublyListNode()
        self._tail = DoublyListNode()
        self._head.next = self._tail
        self._tail.prev = self._head

    def _remove(self, node: DoublyListNode) -> None:
        """Remove a node from the list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_front(self, node: DoublyListNode) -> None:
        """Add a node to the front of the list (immediately after head)."""
        node.next = self._head.next
        node.prev = self._head
        self._head.next.prev = node
        self._head.next = node

    def _move_to_front(self, node: DoublyListNode) -> None:
        """Move a node to the front of the list."""
        self._remove(node)
        self._add_to_front(node)

    def _evict(self) -> None:
        """Delete the oldest entry (immediately before tail)."""
        lru_node = self._tail.prev
        self._remove(lru_node)
        del self.cache[lru_node.key]

    def get(self, key: int) -> int:
        """Get the value for a key. O(1).

        If found, move the node to the front (update as most recently used).
        Return -1 if not found.
        """
        if key in self.cache:
            node = self.cache[key]
            self._move_to_front(node)
            return node.val
        return -1

    def put(self, key: int, value: int) -> None:
        """Add or update a key-value pair. O(1).

        If the key already exists, update its value and move it to the front.
        If it is a new key, add it to the front and delete the oldest if capacity is exceeded.
        """
        if key in self.cache:
            node = self.cache[key]
            node.val = value
            self._move_to_front(node)
        else:
            if len(self.cache) >= self.capacity:
                self._evict()
            new_node = DoublyListNode(key, value)
            self.cache[key] = new_node
            self._add_to_front(new_node)


# Verification
if __name__ == "__main__":
    cache = LRUCache(3)

    cache.put(1, 100)
    cache.put(2, 200)
    cache.put(3, 300)
    print(f"get(1): {cache.get(1)}")   # 100 (1 becomes most recently used)
    print(f"get(2): {cache.get(2)}")   # 200 (2 becomes most recently used)

    cache.put(4, 400)  # Capacity exceeded -> 3 is deleted
    print(f"get(3): {cache.get(3)}")   # -1 (already deleted)
    print(f"get(4): {cache.get(4)}")   # 400

    cache.put(5, 500)  # Capacity exceeded -> 1 is deleted
    print(f"get(1): {cache.get(1)}")   # -1 (already deleted)
    print(f"get(2): {cache.get(2)}")   # 200 (still exists)
```

---

## 6. Circular Linked List Implementation and Applications

### 6.1 Circular Linked List Class

```python
from typing import Optional, List


class ListNode:
    __slots__ = ('val', 'next')
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __repr__(self):
        return f"ListNode({self.val})"


class CircularLinkedList:
    """Singly circular linked list.

    Maintains only a tail pointer, and with tail.next = head,
    achieves O(1) access to both the head and tail.

    Structure:
        tail -> [C] -> [A] -> [B] -> [C] (= tail)
                        ^ head           ^ tail

    tail.next points to head:
        tail.next = head
    """

    def __init__(self):
        self.tail = None
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def __bool__(self) -> bool:
        return self._size > 0

    def __iter__(self):
        if not self.tail:
            return
        current = self.tail.next  # head
        for _ in range(self._size):
            yield current.val
            current = current.next

    def __repr__(self):
        if not self.tail:
            return "CircularLinkedList([])"
        values = list(self)
        return f"CircularLinkedList({values})"

    @property
    def head(self):
        return self.tail.next if self.tail else None

    def append(self, val) -> None:
        """Insert at the tail. O(1)."""
        new_node = ListNode(val)
        if not self.tail:
            new_node.next = new_node  # Points to itself
            self.tail = new_node
        else:
            new_node.next = self.tail.next  # New node -> head
            self.tail.next = new_node        # Old tail -> new node
            self.tail = new_node             # Update tail
        self._size += 1

    def prepend(self, val) -> None:
        """Insert at the head. O(1)."""
        new_node = ListNode(val)
        if not self.tail:
            new_node.next = new_node
            self.tail = new_node
        else:
            new_node.next = self.tail.next  # New node -> old head
            self.tail.next = new_node        # tail -> new node (= new head)
        self._size += 1

    def delete(self, val) -> bool:
        """Delete a node with the specified value. O(n)."""
        if not self.tail:
            return False

        # Single node case
        if self._size == 1:
            if self.tail.val == val:
                self.tail = None
                self._size = 0
                return True
            return False

        prev = self.tail
        current = self.tail.next  # head
        for _ in range(self._size):
            if current.val == val:
                if current == self.tail:
                    self.tail = prev
                prev.next = current.next
                self._size -= 1
                return True
            prev = current
            current = current.next
        return False

    def rotate(self, k: int = 1) -> None:
        """Rotate the list k times. O(k).

        By advancing the tail k times, the head element changes.
        Used in round-robin scheduling.
        """
        if not self.tail or k == 0:
            return
        k = k % self._size
        for _ in range(k):
            self.tail = self.tail.next

    @classmethod
    def from_list(cls, values: List) -> 'CircularLinkedList':
        cll = cls()
        for val in values:
            cll.append(val)
        return cll


# Verification
if __name__ == "__main__":
    cll = CircularLinkedList.from_list([1, 2, 3, 4, 5])
    print(f"Initial state: {cll}")
    # Output: CircularLinkedList([1, 2, 3, 4, 5])

    cll.rotate(2)
    print(f"rotate(2): {cll}")
    # Output: CircularLinkedList([3, 4, 5, 1, 2])

    cll.prepend(0)
    print(f"prepend(0): {cll}")
    # Output: CircularLinkedList([0, 3, 4, 5, 1, 2])

    cll.delete(5)
    print(f"delete(5): {cll}")
    # Output: CircularLinkedList([0, 3, 4, 1, 2])
```

### 6.2 The Josephus Problem — A Classic Application of Circular Lists

The Josephus problem asks: given n people arranged in a circle, with every k-th person being eliminated, who is the last person remaining? This naturally models as a circular linked list.

```python
def josephus(n: int, k: int) -> int:
    """Solve the Josephus problem using a circular linked list.

    n people are arranged in a circle, and every k-th person is eliminated.
    Returns the number of the last remaining person (0-indexed).

    Time Complexity: O(n * k)
    Space Complexity: O(n)

    Args:
        n: Number of people.
        k: Counting interval.

    Returns:
        The number of the last remaining person (0-indexed).

    Diagram (n=5, k=3):
        Initial:  0 - 1 - 2 - 3 - 4 (circular)
        Step 1:   2 eliminated -> 0 - 1 - 3 - 4
        Step 2:   0 eliminated -> 1 - 3 - 4
        Step 3:   4 eliminated -> 1 - 3
        Step 4:   1 eliminated -> 3
        Result: 3
    """
    # Build the circular list
    class Node:
        __slots__ = ('val', 'next')
        def __init__(self, val):
            self.val = val
            self.next = None

    # Connect nodes in a ring
    head = Node(0)
    current = head
    for i in range(1, n):
        current.next = Node(i)
        current = current.next
    current.next = head  # Create the cycle

    # Eliminate every k-th person
    prev = current  # Tail node (immediately before head)
    current = head
    remaining = n

    while remaining > 1:
        # Advance k-1 times (to reach the k-th node)
        for _ in range(k - 1):
            prev = current
            current = current.next
        # Delete current
        prev.next = current.next
        current = prev.next
        remaining -= 1

    return current.val


# Verification
if __name__ == "__main__":
    print(f"josephus(5, 3) = {josephus(5, 3)}")  # 3
    print(f"josephus(7, 2) = {josephus(7, 2)}")  # 6
    print(f"josephus(10, 3) = {josephus(10, 3)}")  # 3

    # Verification against mathematical solution
    def josephus_math(n: int, k: int) -> int:
        """Mathematical recursive solution. O(n) time, O(1) space."""
        result = 0
        for i in range(2, n + 1):
            result = (result + k) % i
        return result

    for n in range(1, 20):
        for k in [2, 3, 5]:
            assert josephus(n, k) == josephus_math(n, k), \
                f"Mismatch at n={n}, k={k}"
    print("All tests passed")
```

---

## 7. Floyd's Cycle Detection Algorithm

### 7.1 Problem Definition

Determine whether a cycle exists in a linked list, and if so, identify the cycle start node and the cycle length.

```
Example of a list with a cycle:

  [1] -> [2] -> [3] -> [4] -> [5] -> [6]
                ^                      |
                +----------------------+

  Node 3 is the cycle start point
  Cycle length = 4 (3 -> 4 -> 5 -> 6 -> 3)

  Non-cycle portion length (mu) = 2 (reaching 3 from 1 -> 2 -> 3)
  Cycle length (lambda) = 4
```

### 7.2 Tortoise and Hare Algorithm — Cycle Detection

The slow pointer (tortoise) advances 1 step at a time, while the fast pointer (hare) advances 2 steps at a time. If a cycle exists, the hare will inevitably catch up to the tortoise.

```python
from typing import Optional


class ListNode:
    __slots__ = ('val', 'next')
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def has_cycle(head: Optional[ListNode]) -> bool:
    """Determine whether a cycle exists.

    Time Complexity: O(n)
    Space Complexity: O(1)

    Args:
        head: The head node of the list.

    Returns:
        True if a cycle exists.
    """
    slow = fast = head
    while fast and fast.next:
        slow = slow.next          # 1 step
        fast = fast.next.next     # 2 steps
        if slow is fast:          # Same node (comparing references, not values)
            return True
    return False
```

```
Step-by-step trace of cycle detection:

  List: [1] -> [2] -> [3] -> [4] -> [5]
                        ^              |
                        +--------------+

  Step 0: slow = [1], fast = [1]
  Step 1: slow = [2], fast = [3]
  Step 2: slow = [3], fast = [5]
  Step 3: slow = [4], fast = [4]  <- Match! Cycle exists

  Why do they always meet?
  ----------------------------
  After both enter the cycle, fast approaches slow by
  1 node per step (relative speed = 1).
  Therefore, with cycle length lambda, they will meet within at most lambda steps.
```

### 7.3 Identifying the Cycle Start Point

In Phase 2 of Floyd's algorithm, after slow and fast meet, slow is moved back to head, and both are advanced 1 step at a time. They will meet at the cycle start point.

```python
def detect_cycle_start(head: Optional[ListNode]) -> Optional[ListNode]:
    """Identify the cycle start node.

    Time Complexity: O(n)
    Space Complexity: O(1)

    Mathematical Proof:
        Let mu be the distance from head to the cycle start point,
        and lambda be the cycle length.
        The meeting point of slow and fast is at position
        (mu mod lambda) from the cycle start point.
        Therefore, advancing mu steps from head will meet at the same point
        as advancing mu steps from the meeting point (= cycle start point).

    Args:
        head: The head node of the list.

    Returns:
        The cycle start node. None if no cycle exists.
    """
    slow = fast = head

    # Phase 1: Find the meeting point
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            break
    else:
        return None  # No cycle

    # Phase 2: Move slow back to head, advance both 1 step at a time
    slow = head
    while slow is not fast:
        slow = slow.next
        fast = fast.next

    return slow  # Cycle start point
```

```
Diagram of cycle start point identification:

  head                          meeting point
   |                                |
   v                                v
  [1] -> [2] -> [3] -> [4] -> [5] -> [6]
                ^                      |
                +----------------------+

  mu (non-cycle portion) = 2  (head -> [3])
  lambda (cycle length) = 4   ([3]->[4]->[5]->[6]->[3])

  After Phase 1:
    slow and fast met at [6]

  Phase 2:
    slow = head = [1],  fast = [6]
    Step 1: slow = [2], fast = [3]
    Step 2: slow = [3], fast = [4]
    ... They actually meet at [3] = cycle start point
```

### 7.4 Measuring Cycle Length

```python
def cycle_length(head: Optional[ListNode]) -> int:
    """Return the cycle length. Returns 0 if no cycle exists.

    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    slow = fast = head

    # Cycle detection
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            # Count by going around from the meeting point back to the same point
            length = 1
            runner = slow.next
            while runner is not slow:
                length += 1
                runner = runner.next
            return length

    return 0  # No cycle


# Test helper: Build a list with a cycle
def build_cyclic_list(values, cycle_start_index):
    """Build a list with a cycle.

    Args:
        values: List of node values.
        cycle_start_index: Index of the cycle start position.
            -1 means no cycle.
    """
    if not values:
        return None

    nodes = [ListNode(v) for v in values]
    for i in range(len(nodes) - 1):
        nodes[i].next = nodes[i + 1]

    if cycle_start_index >= 0:
        nodes[-1].next = nodes[cycle_start_index]

    return nodes[0]


# Verification
if __name__ == "__main__":
    # List with a cycle
    head = build_cyclic_list([1, 2, 3, 4, 5, 6], cycle_start_index=2)
    print(f"Has cycle: {has_cycle(head)}")              # True
    start = detect_cycle_start(head)
    print(f"Cycle start point: {start.val}")             # 3
    print(f"Cycle length: {cycle_length(head)}")         # 4

    # List without a cycle
    head_no_cycle = build_cyclic_list([1, 2, 3, 4, 5], cycle_start_index=-1)
    print(f"No cycle: {has_cycle(head_no_cycle)}")       # False
    print(f"Cycle start point: {detect_cycle_start(head_no_cycle)}")  # None
    print(f"Cycle length: {cycle_length(head_no_cycle)}") # 0
```

### 7.5 Mathematical Proof of Floyd's Algorithm

Here we present the mathematical basis for Floyd's algorithm.

**Assumptions**:
- Distance from head to cycle start: mu
- Cycle length: lambda
- Distance traveled by slow before meeting: d

**Phase 1 Proof**:
1. When slow has taken d steps, fast has taken 2d steps
2. At the meeting point: 2d - d = d is a multiple of the cycle length
3. That is, d = k * lambda (k is a positive integer)

**Phase 2 Proof**:
1. The meeting point is at position (d - mu) from the cycle start
2. Since d = k * lambda, advancing mu steps from the meeting point gives:
   (d - mu) + mu = d = k * lambda (returning to the cycle start)
3. Advancing mu steps from head also reaches the cycle start
4. Therefore, the point where both meet after mu steps is the cycle start

---

## 8. Applied Algorithm Collection

### 8.1 Merging Sorted Lists

Merge two sorted lists into a single sorted list. This algorithm forms the foundation of merge sort.

```python
from typing import Optional


class ListNode:
    __slots__ = ('val', 'next')
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def merge_sorted_lists(l1: Optional[ListNode],
                       l2: Optional[ListNode]) -> Optional[ListNode]:
    """Merge two sorted lists.

    Time Complexity: O(n + m)
    Space Complexity: O(1) (relinks existing nodes without creating new ones)

    Args:
        l1, l2: Head nodes of sorted lists.

    Returns:
        Head node of the merged list.
    """
    dummy = ListNode(0)
    current = dummy

    while l1 and l2:
        if l1.val <= l2.val:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next
        current = current.next

    # Attach the remainder
    current.next = l1 if l1 else l2
    return dummy.next


def build_list(values):
    if not values:
        return None
    head = ListNode(values[0])
    curr = head
    for v in values[1:]:
        curr.next = ListNode(v)
        curr = curr.next
    return head


def list_to_string(head):
    parts = []
    curr = head
    while curr:
        parts.append(str(curr.val))
        curr = curr.next
    return " -> ".join(parts) + " -> None"


# Verification
if __name__ == "__main__":
    l1 = build_list([1, 3, 5, 7])
    l2 = build_list([2, 4, 6, 8])
    merged = merge_sorted_lists(l1, l2)
    print(list_to_string(merged))
    # Output: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> None
```

### 8.2 Merging K Sorted Lists

Efficiently merge K lists using a heap (priority queue).

```python
import heapq
from typing import Optional, List


class ListNode:
    __slots__ = ('val', 'next')
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __lt__(self, other):
        """Required for heap comparison"""
        return self.val < other.val


def merge_k_lists(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    """Merge K sorted lists.

    Time Complexity: O(N log K) (N = total number of nodes, K = number of lists)
    Space Complexity: O(K) (heap size)

    Args:
        lists: Array of head nodes of sorted lists.

    Returns:
        Head node of the merged list.
    """
    dummy = ListNode(0)
    current = dummy

    # Push the head of each list into the heap
    heap = []
    for i, node in enumerate(lists):
        if node:
            heapq.heappush(heap, (node.val, i, node))

    while heap:
        val, idx, node = heapq.heappop(heap)
        current.next = node
        current = current.next
        if node.next:
            heapq.heappush(heap, (node.next.val, idx, node.next))

    return dummy.next


def build_list(values):
    if not values:
        return None
    head = ListNode(values[0])
    curr = head
    for v in values[1:]:
        curr.next = ListNode(v)
        curr = curr.next
    return head


def list_to_string(head):
    parts = []
    curr = head
    while curr:
        parts.append(str(curr.val))
        curr = curr.next
    return " -> ".join(parts) + " -> None"


# Verification
if __name__ == "__main__":
    lists = [
        build_list([1, 4, 7]),
        build_list([2, 5, 8]),
        build_list([3, 6, 9]),
    ]
    merged = merge_k_lists(lists)
    print(list_to_string(merged))
    # Output: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9 -> None
```

### 8.3 Finding the Middle Node — slow/fast Pointer

```python
def find_middle(head: Optional[ListNode]) -> Optional[ListNode]:
    """Find the middle node of a list.

    For even-length lists, returns the start of the second half (right of two middle nodes).

    Time Complexity: O(n)
    Space Complexity: O(1)

    Diagram:
        Odd length:  [1] -> [2] -> [3] -> [4] -> [5]
                                   ^ middle

        Even length: [1] -> [2] -> [3] -> [4]
                                   ^ middle (start of second half)
    """
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow
```

### 8.4 Removing the K-th Node from the End

```python
def remove_nth_from_end(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    """Remove the n-th node from the end.

    Advance two pointers separated by n nodes simultaneously.

    Time Complexity: O(L) (L = list length)
    Space Complexity: O(1)

    Diagram (n=2):
        dummy -> [1] -> [2] -> [3] -> [4] -> [5] -> None
                                       ^ target (2nd from end)

        Step 1: Advance fast by n+1 steps
        dummy -> [1] -> [2] -> [3] -> [4] -> [5]
        ^ slow                         ^ fast

        Step 2: Advance both simultaneously (until fast reaches None)
                        ^ slow                         ^ fast(None)
        slow.next = slow.next.next  to remove [4]
    """
    dummy = ListNode(0, head)
    fast = dummy
    slow = dummy

    # Advance fast by n+1 steps
    for _ in range(n + 1):
        fast = fast.next

    # Advance both simultaneously
    while fast:
        slow = slow.next
        fast = fast.next

    # slow.next is the target for deletion
    slow.next = slow.next.next
    return dummy.next
```

### 8.5 Palindrome Check — Comparing First and Second Halves

```python
def is_palindrome(head: Optional[ListNode]) -> bool:
    """Determine whether a linked list is a palindrome.

    Steps:
    1. Find the middle node (slow/fast)
    2. Reverse the second half
    3. Compare the first and second halves
    4. (Optional) Restore the second half

    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if not head or not head.next:
        return True

    # Step 1: Find the middle node
    slow = fast = head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next

    # Step 2: Reverse the second half
    second_half = reverse(slow.next)

    # Step 3: Compare the first and second halves
    first_half = head
    result = True
    check = second_half
    while check:
        if first_half.val != check.val:
            result = False
            break
        first_half = first_half.next
        check = check.next

    # Step 4: Restore the second half (preserve list structure)
    slow.next = reverse(second_half)

    return result


def reverse(head):
    """Reverse a list."""
    prev = None
    curr = head
    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node
    return prev
```

### 8.6 Sorting a List — Merge Sort

Merge sort is optimal for sorting linked lists. Unlike arrays, linked lists allow O(1) space merge operations.

```python
def sort_list(head: Optional[ListNode]) -> Optional[ListNode]:
    """Sort a linked list using merge sort.

    Time Complexity: O(n log n)
    Space Complexity: O(log n) — recursion stack

    Args:
        head: Head node of the list to sort.

    Returns:
        Head node of the sorted list.
    """
    # Base case: 0 or 1 node
    if not head or not head.next:
        return head

    # Split the list
    mid = get_middle_for_split(head)
    right_head = mid.next
    mid.next = None  # Sever the left half

    # Recursively sort
    left = sort_list(head)
    right = sort_list(right_head)

    # Merge
    return merge(left, right)


def get_middle_for_split(head):
    """Return the middle node for splitting (end of the left half)."""
    slow = head
    fast = head.next  # Ensures the left half is shorter for even-length lists
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow


def merge(l1, l2):
    """Merge two sorted lists."""
    dummy = ListNode(0)
    curr = dummy
    while l1 and l2:
        if l1.val <= l2.val:
            curr.next = l1
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next
    curr.next = l1 if l1 else l2
    return dummy.next


def build_list(values):
    if not values:
        return None
    head = ListNode(values[0])
    curr = head
    for v in values[1:]:
        curr.next = ListNode(v)
        curr = curr.next
    return head


def list_to_string(head):
    parts = []
    curr = head
    while curr:
        parts.append(str(curr.val))
        curr = curr.next
    return " -> ".join(parts) + " -> None"


# Verification
if __name__ == "__main__":
    head = build_list([4, 2, 1, 3, 5])
    print(f"Before sort: {list_to_string(head)}")
    # Output: Before sort: 4 -> 2 -> 1 -> 3 -> 5 -> None

    sorted_head = sort_list(head)
    print(f"After sort: {list_to_string(sorted_head)}")
    # Output: After sort: 1 -> 2 -> 3 -> 4 -> 5 -> None
```

---

## 9. Comparison Tables and Complexity Summary

### Table 1: Detailed Comparison by List Type

| Property | Singly Linked List | Doubly Linked List | Circular List (Singly) | Circular List (Doubly) |
|----------|-------------------|-------------------|----------------------|----------------------|
| Pointers per node | 1 | 2 | 1 | 2 |
| Forward traversal | O(1) / step | O(1) / step | O(1) / step | O(1) / step |
| Backward traversal | O(n) | O(1) / step | O(n) | O(1) / step |
| Head insertion | O(1) | O(1) | O(1) | O(1) |
| Tail insertion (no tail ptr) | O(n) | O(n) | O(1)* | O(1)* |
| Tail insertion (with tail ptr) | O(1) | O(1) | O(1) | O(1) |
| Head deletion | O(1) | O(1) | O(1) | O(1) |
| Tail deletion | O(n) | O(1) | O(n) | O(1) |
| Arbitrary node deletion (with ref) | O(n)** | O(1) | O(n)** | O(1) |
| Search | O(n) | O(n) | O(n) | O(n) |
| Memory efficiency | Best | Medium | Good | Medium |
| Null references | Present | Present | None | None |
| Implementation complexity | Low | Medium | Medium | High |
| Typical use cases | Stack, chaining | LRU cache | Round-robin | OS task scheduler |

\* In circular lists, head (= tail.next) is accessible in O(1) via the tail pointer

\** Traversal required since there is no reference to the previous node

### Table 2: Array vs Linked List — Detailed Comparison

| Aspect | Dynamic Array (Python list) | Singly Linked List | Doubly Linked List |
|--------|---------------------------|-------------------|-------------------|
| Random access | O(1) | O(n) | O(n) |
| Head insertion | O(n) | O(1) | O(1) |
| Tail insertion | O(1) amortized | O(n) or O(1)* | O(1) |
| Middle insertion | O(n) | O(1)** | O(1)** |
| Head deletion | O(n) | O(1) | O(1) |
| Tail deletion | O(1) | O(n) | O(1) |
| Middle deletion | O(n) | O(1)** | O(1)** |
| Memory locality | High (cache-efficient) | Low | Low |
| Memory overhead | Low (extra slots) | Medium (next pointer) | High (prev + next pointers) |
| Resizing | Auto-resize (copying occurs) | Not needed | Not needed |
| Memory fragmentation | None | Present | Present |
| Iteration speed | Fastest | Slow | Slow |
| Concurrency compatibility | Coarse-grained locking | Node-level locking possible | Node-level locking possible |
| Language support | Broad | Often manual implementation | Often manual implementation |

\* O(1) when a tail pointer is maintained
\** When a reference to the insertion/deletion position node is already available

### Table 3: Complexity of Linked List Related Algorithms

| Algorithm | Time Complexity | Space Complexity | Notes |
|-----------|----------------|-----------------|-------|
| List reversal (iterative) | O(n) | O(1) | Recommended |
| List reversal (recursive) | O(n) | O(n) | Stack overflow risk |
| Cycle detection (Floyd) | O(n) | O(1) | Recommended |
| Cycle detection (hash set) | O(n) | O(n) | Simple to implement |
| 2-list merge | O(n + m) | O(1) | Uses dummy head |
| K-list merge (heap) | O(N log K) | O(K) | N = total nodes |
| Middle node retrieval | O(n) | O(1) | slow/fast |
| Remove K-th from end | O(n) | O(1) | 2 pointers |
| Palindrome check | O(n) | O(1) | Reverse second half |
| Merge sort | O(n log n) | O(log n) | Recursion stack |
| Josephus problem | O(n * k) | O(n) | Circular list |
| LRU cache get/put | O(1) | O(n) | DLL + HashMap |

---

## 10. Anti-pattern Collection

### Anti-pattern 1: Not Using a Dummy Head

Without a dummy head, special handling for head node deletion and empty list processing increases conditional branching, becoming a breeding ground for bugs.

```python
# BAD: Special handling for head node -> complex and error-prone code
def delete_bad(head, val):
    if head and head.val == val:
        return head.next
    curr = head
    while curr and curr.next:
        if curr.next.val == val:
            curr.next = curr.next.next
            return head
        curr = curr.next
    return head

# GOOD: Uniform processing with a dummy head
def delete_good(head, val):
    dummy = ListNode(0, head)
    prev = dummy
    curr = head
    while curr:
        if curr.val == val:
            prev.next = curr.next
        else:
            prev = curr
        curr = curr.next
    return dummy.next
```

**Problem**: In `delete_bad`, head node deletion becomes a special case, making it difficult to correctly handle multiple value deletions or empty list cases.

### Anti-pattern 2: Traversing to Compute List Length Every Time

```python
# BAD: Count length in O(n) with every operation
class BadList:
    def __init__(self):
        self.head = None

    def length(self):  # O(n) every time
        count = 0
        curr = self.head
        while curr:
            count += 1
            curr = curr.next
        return count

    def insert_at(self, index, val):
        if index > self.length():  # O(n) here
            raise IndexError
        # ... insertion processing also O(n)
        # Total O(2n) = O(n) but the constant factor is wasteful

# GOOD: Track length as a field
class GoodList:
    def __init__(self):
        self.head = None
        self._size = 0

    def length(self):  # O(1)
        return self._size

    def insert_at(self, index, val):
        if index > self._size:
            raise IndexError
        # ... insertion processing
        self._size += 1
```

### Anti-pattern 3: Overlooking Memory Leaks on Node Deletion

Not an issue in garbage-collected languages like Python, but a serious problem in manual memory management languages like C/C++.

```python
# BAD (C/C++ mindset): Leaving references to deleted nodes dangling
def delete_node_bad(prev_node):
    # Just skip the node without freeing memory
    target = prev_node.next
    prev_node.next = target.next
    # target's memory is not freed -> memory leak
    # In C, free(target) is required

# GOOD: In Python, reference counting handles automatic deallocation,
# but explicitly severing references is good practice for circular references
def delete_node_good(prev_node):
    target = prev_node.next
    prev_node.next = target.next
    target.next = None  # Explicitly sever reference (good practice)
```

### Anti-pattern 4: Modifying the List During Traversal

```python
# BAD: Deleting nodes during iteration -> unexpected behavior
def delete_all_bad(head, val):
    curr = head
    while curr:
        if curr.val == val:
            # Want to delete curr, but no reference to prev
            # Directly manipulating curr.next will break the structure
            pass
        curr = curr.next

# GOOD: Maintain prev pointer and delete safely
def delete_all_good(head, val):
    dummy = ListNode(0, head)
    prev = dummy
    curr = head
    while curr:
        if curr.val == val:
            prev.next = curr.next  # Skip the node
            # prev does not move (next node might also be a deletion target)
        else:
            prev = curr
        curr = curr.next
    return dummy.next
```

### Anti-pattern 5: Confusing Value Comparison with Node Identity

```python
# BAD: Using == for value comparison (fatal in cycle detection)
def has_cycle_bad(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:       # Value comparison -> can be True for different nodes
            return True
    return False

# GOOD: Using is for node identity comparison
def has_cycle_good(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:       # Compares whether they are the same object
            return True
    return False
```

---

## 11. Exercises -- Basic, Intermediate, Advanced

### Basic Problems

#### Problem B1: Count the Elements in a List

Implement a function that receives the head node of a linked list and returns the number of elements.

```python
def count_nodes(head: Optional[ListNode]) -> int:
    """Return the number of nodes in a linked list.

    Args:
        head: The head node of the list.

    Returns:
        The number of nodes.

    Examples:
        >>> count_nodes(build_list([1, 2, 3, 4, 5]))
        5
        >>> count_nodes(None)
        0
    """
    # Write your implementation here
    pass
```

<details>
<summary>Show Solution</summary>

```python
def count_nodes(head):
    count = 0
    current = head
    while current:
        count += 1
        current = current.next
    return count
```
</details>

#### Problem B2: Return the Value of the Tail Node

```python
def get_last(head: Optional[ListNode]):
    """Return the value of the tail node. None for an empty list.

    Examples:
        >>> get_last(build_list([10, 20, 30]))
        30
        >>> get_last(None)
        None
    """
    pass
```

<details>
<summary>Show Solution</summary>

```python
def get_last(head):
    if not head:
        return None
    current = head
    while current.next:
        current = current.next
    return current.val
```
</details>

#### Problem B3: Remove Duplicates from a Sorted List

Remove nodes with duplicate values from a sorted linked list.

```python
def remove_duplicates_sorted(head: Optional[ListNode]) -> Optional[ListNode]:
    """Remove duplicates from a sorted list.

    Examples:
        >>> list_to_array(remove_duplicates_sorted(build_list([1, 1, 2, 3, 3, 3, 4])))
        [1, 2, 3, 4]
    """
    pass
```

<details>
<summary>Show Solution</summary>

```python
def remove_duplicates_sorted(head):
    current = head
    while current and current.next:
        if current.val == current.next.val:
            current.next = current.next.next
        else:
            current = current.next
    return head
```
</details>

### Intermediate Problems

#### Problem A1: Find the Intersection Node of Two Lists

Given two linked lists that merge at some point, return the node at the merge point. Return None if they do not intersect. Solve in O(n + m) time and O(1) space.

```
List A: [1] -> [2] --+
                      +---> [8] -> [9] -> None
List B: [3] -> [4] -> [5] --+
```

```python
def get_intersection_node(headA: ListNode, headB: ListNode) -> Optional[ListNode]:
    """Return the intersection node of two lists.

    Hint: When pointer A reaches the end of list A, move it to the head of list B,
    and when pointer B reaches the end of list B, move it to the head of list A.
    """
    pass
```

<details>
<summary>Show Solution</summary>

```python
def get_intersection_node(headA, headB):
    if not headA or not headB:
        return None

    pA, pB = headA, headB

    # Each pointer reaches the intersection after at most 2 list traversals
    # lenA + lenB == lenB + lenA, so they travel the same distance
    while pA is not pB:
        pA = pA.next if pA else headB
        pB = pB.next if pB else headA

    return pA  # Intersection point, or None (no intersection)
```

**Why this works**: Pointer A travels lenA + lenB steps, and pointer B travels lenB + lenA steps. If there is an intersection, let c be the length of the common portion from the intersection to the end. Both pointers meet at the intersection after (lenA - c) + (lenB - c) + c steps.
</details>

#### Problem A2: Separate a List into Odd and Even Indexed Nodes

Separate the nodes of a list into odd-indexed (1st, 3rd, 5th, ...) and even-indexed (2nd, 4th, 6th, ...) groups, then concatenate the odd group followed by the even group.

```python
def odd_even_list(head: Optional[ListNode]) -> Optional[ListNode]:
    """Separate odd-indexed and even-indexed nodes and concatenate them.

    Examples:
        >>> list_to_array(odd_even_list(build_list([1, 2, 3, 4, 5])))
        [1, 3, 5, 2, 4]
    """
    pass
```

<details>
<summary>Show Solution</summary>

```python
def odd_even_list(head):
    if not head or not head.next:
        return head

    odd = head
    even = head.next
    even_head = even

    while even and even.next:
        odd.next = even.next
        odd = odd.next
        even.next = odd.next
        even = even.next

    odd.next = even_head  # Concatenate even list after odd list
    return head
```
</details>

#### Problem A3: Partition a Linked List by Value x

Move all nodes with values less than x before all nodes with values greater than or equal to x. Preserve the relative order within each partition.

```python
def partition(head: Optional[ListNode], x: int) -> Optional[ListNode]:
    """Partition the list around value x.

    Examples:
        >>> list_to_array(partition(build_list([1, 4, 3, 2, 5, 2]), 3))
        [1, 2, 2, 4, 3, 5]
    """
    pass
```

<details>
<summary>Show Solution</summary>

```python
def partition(head, x):
    before_dummy = ListNode(0)
    after_dummy = ListNode(0)
    before = before_dummy
    after = after_dummy

    current = head
    while current:
        if current.val < x:
            before.next = current
            before = before.next
        else:
            after.next = current
            after = after.next
        current = current.next

    after.next = None          # Sever the tail
    before.next = after_dummy.next  # Concatenate the two lists
    return before_dummy.next
```
</details>

### Advanced Problems

#### Problem E1: Delete an Arbitrary Node in O(1)

Given only a reference to a node (not the tail) with a next node, delete that node in O(1). The head is not provided.

```python
def delete_node(node: ListNode) -> None:
    """Delete the given node from the list in O(1).

    Constraint: node is not the tail node.

    Hint: Combine value copying with pointer reassignment.
    """
    pass
```

<details>
<summary>Show Solution</summary>

```python
def delete_node(node):
    # Copy the next node's value to self, then skip the next node
    node.val = node.next.val
    node.next = node.next.next
```

**Note**: This technique is "value copying" rather than "node deletion," and it does not affect code that holds external references to the original node. It also cannot be applied to tail nodes.
</details>

#### Problem E2: Reverse Nodes in K-Groups

Divide the list into groups of K and reverse each group. If the last group has fewer than K nodes, leave it as is.

```python
def reverse_k_group(head: Optional[ListNode], k: int) -> Optional[ListNode]:
    """Reverse the list in groups of K.

    Examples:
        >>> list_to_array(reverse_k_group(build_list([1, 2, 3, 4, 5]), 3))
        [3, 2, 1, 4, 5]
    """
    pass
```

<details>
<summary>Show Solution</summary>

```python
def reverse_k_group(head, k):
    # First check if there are K nodes
    count = 0
    node = head
    while node and count < k:
        node = node.next
        count += 1

    if count < k:
        return head  # Fewer than K nodes, leave as is

    # Reverse K nodes
    prev = None
    curr = head
    for _ in range(k):
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node

    # head is now the tail of the reversed portion. Recursively process the rest and connect
    head.next = reverse_k_group(curr, k)
    return prev  # New head of the reversed portion
```
</details>

#### Problem E3: Deep Copy a List with Random Pointers

Create a deep copy of a linked list where each node has a next pointer and a random pointer (pointing to any node in the list or None).

```python
class RandomNode:
    def __init__(self, val=0, next=None, random=None):
        self.val = val
        self.next = next
        self.random = random


def copy_random_list(head: Optional[RandomNode]) -> Optional[RandomNode]:
    """Create a deep copy of a list with random pointers.

    Solve in O(n) time and O(n) space.
    """
    pass
```

<details>
<summary>Show Solution</summary>

```python
def copy_random_list(head):
    if not head:
        return None

    # Step 1: Create old -> new mapping
    old_to_new = {}
    current = head
    while current:
        old_to_new[current] = RandomNode(current.val)
        current = current.next

    # Step 2: Copy next and random
    current = head
    while current:
        copy = old_to_new[current]
        copy.next = old_to_new.get(current.next)
        copy.random = old_to_new.get(current.random)
        current = current.next

    return old_to_new[head]
```

**Advanced**: An O(1) space solution also exists (interleaving technique). Insert a copy immediately after each original node, set the random pointers, then separate the original list from the copies.
</details>

---

## 12. FAQ -- Frequently Asked Questions

### Q1: Where are linked lists used in practice?

**A**: Linked lists are widely used in the following contexts:

- **OS Process Management**: The Linux kernel uses doubly circular linked lists (`list_head`) for managing processes and threads.
- **LRU Cache**: The combination of doubly linked list + hash map achieves O(1) get/put (see Section 5.2 of this guide).
- **Memory Allocators**: Used as free lists (lists of available memory blocks). Involved in the internal implementation of malloc/free.
- **Browser History**: "Back/Forward" functionality can be modeled with a doubly linked list.
- **Music Player Playlists**: Previous/next track, shuffle playback.
- **Blockchain**: Each block holds the hash of the previous block, forming a linked list structure.
- **Undo/Redo Functionality**: Command history in text editors and graphics tools.

### Q2: Does Python have a built-in linked list?

**A**: Python's standard library does not have a pure linked list class. However, `collections.deque` is internally implemented as a block doubly linked list and provides O(1) insertion and deletion from both ends.

```python
from collections import deque

# deque covers many linked list use cases
d = deque([1, 2, 3, 4, 5])
d.appendleft(0)    # Head insertion O(1)
d.append(6)        # Tail insertion O(1)
d.popleft()        # Head deletion O(1)
d.pop()            # Tail deletion O(1)
d.rotate(2)        # Rotation
```

In coding interviews, implementing a custom `ListNode` class is generally expected.

### Q3: Why can Floyd's algorithm detect cycles in O(1) space?

**A**: Floyd's algorithm uses only two pointers (slow and fast). Since slow advances 1 step and fast advances 2 steps, the additional memory required is constant (just 2 pointers).

In contrast, the hash set approach requires O(n) space to record all visited nodes.

```python
# O(n) space approach (for comparison)
def has_cycle_hashset(head):
    visited = set()
    current = head
    while current:
        if id(current) in visited:  # Compare by node identity
            return True
        visited.add(id(current))
        current = current.next
    return False
```

Intuitive explanation of why Floyd's algorithm works: After both enter the cycle, fast approaches slow by 1 node per step (relative speed = 2 - 1 = 1). With cycle length lambda, they will reach the same node within at most lambda steps.

### Q4: What is the difference between a linked list and a skip list?

**A**: A regular linked list requires O(n) for search, while a skip list achieves O(log n) search by maintaining multiple index levels, making it a probabilistic data structure.

| Property | Linked List | Skip List |
|----------|------------|-----------|
| Search | O(n) | O(log n) expected |
| Insert | O(1)* | O(log n) expected |
| Delete | O(1)* | O(log n) expected |
| Space | O(n) | O(n) expected |
| Implementation complexity | Low | Medium |
| Use cases | General purpose | Ordered collections |

\* When the position is already known

Redis Sorted Sets are implemented with skip lists, used as an alternative to balanced binary search trees.

### Q5: When should you use a linked list instead of an array?

**A**: Linked lists are suitable when the following conditions apply:

1. **Frequent head insertions/deletions**: O(1) for linked lists versus O(n) for arrays.
2. **Unpredictable size**: Arrays incur copying costs on resize, while linked lists can dynamically add nodes.
3. **Fragmented memory utilization**: When a large contiguous memory block cannot be allocated.
4. **Internal implementations of other structures**: Chaining in hash tables, adjacency lists in graphs, etc.

However, on modern hardware, cache-locality-friendly arrays are faster in most scenarios, and linked lists are only advantageous under the specific conditions above.

### Q6: What are the most important techniques for linked list interview problems?

**A**: Mastering the following 5 techniques covers the majority of linked list interview problems:

1. **Dummy Head (Sentinel Node)**: Eliminates edge cases
2. **slow/fast Pointer**: Middle node retrieval, cycle detection
3. **Reversal**: Be able to reliably write the iterative version
4. **Merge**: 2-list and K-list merges
5. **Recursion**: Recursive thinking patterns (but watch the space complexity)

---


## FAQ

### Q1: What is the most important point for learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not only through theory, but by actually writing code and verifying behavior.

### Q2: What common mistakes do beginners make?

Skipping the basics and jumping to advanced topics. We recommend thoroughly understanding the fundamental concepts explained in this guide before moving to the next step.

### Q3: How is this applied in practice?

Knowledge of this topic is frequently utilized in day-to-day development work. It becomes particularly important during code reviews and architecture design.

---

## 13. Summary

### Key Takeaways

| Item | Key Point |
|------|-----------|
| Singly Linked List | Simplest form. Optimal for stack implementation. Each node has 1 pointer |
| Doubly Linked List | Enables O(1) deletion. Core data structure for LRU caches |
| Circular Linked List | Used for round-robin and circular buffers. Watch for termination conditions |
| Dummy Head | The most important technique for uniformly handling edge cases (empty list, head operations) |
| Floyd's Detection | Achieves cycle detection, start point identification, and length measurement in O(1) space |
| slow/fast Pointer | Also applicable to middle point retrieval, palindrome checking, K-th node detection |
| Merge Operations | Combined with dummy heads to achieve 2-list and K-list merging |
| Memory Efficiency | Use `__slots__` to significantly reduce per-node memory |
| LRU Cache | Combination of doubly linked list + hash map for O(1) get/put |

### Learning Roadmap

```
Learning order for linked lists:

  Level 1 (Basics):
    Singly linked list construction -> Insertion/Deletion/Search -> Reversal

  Level 2 (Intermediate):
    Dummy head pattern -> slow/fast pointer -> Merge

  Level 3 (Advanced):
    Floyd's algorithm -> Doubly linked list -> LRU cache

  Level 4 (Expert):
    K-group reversal -> Random pointer copy -> Merge sort
    -> Skip list (conceptual understanding)
```

### Next Steps

After mastering the content of this guide, we recommend progressing to the following topics:

1. **Stacks/Queues**: Implementations based on linked lists
2. **Binary Trees**: Tree structures where nodes have left and right children (an extension of linked lists)
3. **Graphs**: Linked lists are utilized in adjacency list representations
4. **Hash Tables**: Linked lists are used for collision resolution in chaining

---

## Recommended Next Guides

- [Stacks/Queues -- Linked List Implementations](./02-stacks-queues.md)
- [Graphs -- Adjacency List Representation](./06-graphs.md)

---

## 14. References

1. Cormen, T.H., Leiserson, C.E., Rivest, R.L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. -- Chapter 10 "Elementary Data Structures" rigorously defines the basic structures and operations of linked lists.

2. Floyd, R.W. (1967). "Nondeterministic Algorithms." *Journal of the ACM*, 14(4), 636-644. -- The original paper on the cycle detection algorithm. Provides the mathematical foundation for Floyd's tortoise and hare algorithm.

3. Skiena, S.S. (2020). *The Algorithm Design Manual* (3rd ed.). Springer. -- Provides practical guidance on linked lists and their application to interview problems.

4. Knuth, D.E. (1997). *The Art of Computer Programming, Volume 1: Fundamental Algorithms* (3rd ed.). Addison-Wesley. -- Section 2.2 "Linear Lists" provides historical background and detailed analysis of linked lists.

5. Sedgewick, R., & Wayne, K. (2011). *Algorithms* (4th ed.). Addison-Wesley. -- Carefully explains stack and queue implementations using linked lists, along with complexity analysis.

6. Pugh, W. (1990). "Skip Lists: A Probabilistic Alternative to Balanced Trees." *Communications of the ACM*, 33(6), 668-676. -- The original paper on skip lists. An important reference as an evolution of linked lists.

7. MIT OpenCourseWare. (2020). "6.006 Introduction to Algorithms." Massachusetts Institute of Technology. https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/ -- MIT open course where lectures on data structures including linked lists can be viewed for free.

---

> **All code examples in this guide are intended for Python 3.10+.**
> Type hints use the `typing` module, but from Python 3.10 onward,
> built-in type syntax such as `list[int]` is also available.

---

## References

- [MDN Web Docs](https://developer.mozilla.org/) - Web technology reference
- [Wikipedia](https://en.wikipedia.org/) - Technology concept overviews
