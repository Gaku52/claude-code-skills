# Linked Lists

> Linked lists are the best teaching material for learning the concept of "pointers" and serve as the foundation for many data structures.
> By understanding a memory model that contrasts with arrays, you gain insight into the essence of data structure design.

## What You Will Learn in This Chapter

- [ ] Understand the mechanics of singly, doubly, and circular linked lists
- [ ] Accurately grasp the time complexity of each operation
- [ ] Make informed decisions about when to use linked lists versus arrays
- [ ] Solve classic linked list problems (reversal, cycle detection, merging, etc.)
- [ ] Implement practical data structures such as LRU caches

## Prerequisites

- Basic Python class definitions (`__init__`, `self`)
- Concept of pointers/references (experience with C helps deepen understanding but is not required)

---

## 1. Why Linked Lists Are Needed -- The Limitations of Arrays

### 1.1 Structural Constraints of Arrays

Arrays are the most fundamental data structure and deliver excellent performance in many scenarios. However, arrays have several inherent constraints.

**Constraint 1: Requirement for Contiguous Memory**

Arrays place elements contiguously in memory. This property is essential for achieving O(1) random access, but allocating a large array requires a single contiguous free region of the same size. In environments where memory is fragmented, it may be impossible to allocate an array even when sufficient total free space exists.

```
Array memory layout (contiguous region required):

Address: 0x100  0x104  0x108  0x10C  0x110  0x114
         +------+------+------+------+------+------+
         |  10  |  20  |  30  |  40  |  50  |  60  |
         +------+------+------+------+------+------+
         <---------- 24 contiguous bytes needed ---------->

Fragmented memory may cause allocation failure:

         +------+      +------+      +------+
  Free:  | 8B   | Used | 12B  | Used | 8B   | ...
         +------+      +------+      +------+
         28B total free, but cannot obtain 24 contiguous bytes
```

**Constraint 2: Cost of Insertion and Deletion**

Inserting into or deleting from the middle of an array requires shifting all subsequent elements. With n elements, this takes O(n) time in the worst case.

```
Middle insertion in an array (inserting "25" at index 2):

Before: [10, 20, 30, 40, 50]
                ^ Want to insert 25 here

Step 1: Shift trailing elements one position to the right
        [10, 20, 30, 30, 40, 50]  <- Shift 50 right
        [10, 20, 30, 30, 40, 50]  <- Shift 40 right
        [10, 20, 30, 30, 40, 50]  <- Shift 30 right (gap created)

Step 2: Insert into the vacated position
        [10, 20, 25, 30, 40, 50]

-> Up to n-1 copies for an array of n=5 elements
```

**Constraint 3: Overhead of Resizing**

Static arrays have a fixed size. Dynamic arrays (Python's `list`, Java's `ArrayList`) automatically expand, but internally they allocate a new memory region and copy all existing data. The amortized complexity is O(1), but at the moment resizing occurs, an O(n) delay is incurred. In systems requiring real-time guarantees, this transient delay can be problematic.

### 1.2 Linked Lists as a Solution

Linked lists solve these constraints through a fundamentally different approach.

```
Linked list memory layout (non-contiguous is fine):

Address 0x200        Address 0x350        Address 0x120
+-------------+      +-------------+      +-------------+
| val: 10     |      | val: 20     |      | val: 30     |
| next: 0x350 |----->| next: 0x120 |----->| next: None  |
+-------------+      +-------------+      +-------------+
Nodes can be scattered across memory without issue
```

Each element (node) holds "data" and a "reference (pointer) to the next node." Since nodes are connected via pointers, they do not need to be contiguous in memory. Insertions and deletions are completed simply by reassigning pointers, eliminating the need to shift elements.

### 1.3 Fundamental Strengths and Weaknesses of Linked Lists

Linked lists are not a panacea. They are complementary to arrays, with each excelling in different scenarios.

**Strengths:**
- Insertion/deletion at the head in O(1)
- Insertion/deletion at any position (when the pointer is known) in O(1)
- Dynamic resizing without resize-copy overhead
- No contiguous memory region required

**Weaknesses:**
- Random access is O(n) (no direct access by index)
- Each node requires extra memory for the pointer
- Poor cache locality (less benefit from CPU caches)
- Overhead from pointer indirection

This chapter covers the structure and operations of each linked list variant (singly, doubly, circular) in detail, aiming to equip you with the ability to choose appropriately between linked lists and arrays.

---

## 2. Singly Linked List

### 2.1 Basic Structure

A singly linked list is the simplest form of linked list. Each node has two fields: "data" and a "pointer to the next node."

```
Structure of a singly linked list:

  head
   |
   v
+-----------+    +-----------+    +-----------+    +-----------+
| val: "A"  |    | val: "B"  |    | val: "C"  |    | val: "D"  |
| next: -----+-->| next: -----+-->| next: -----+-->| next: None|
+-----------+    +-----------+    +-----------+    +-----------+

Characteristics:
- Each node holds only a next pointer (traversal in one direction only)
- The head pointer serves as the entry point to the list
- The last node's next is None (NULL)
- Reverse traversal is impossible
```

**Node class definition:**

```python
class ListNode:
    """Node for a singly linked list"""
    __slots__ = ('val', 'next')  # Use slots for memory efficiency

    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __repr__(self):
        return f"ListNode({self.val})"
```

By specifying `__slots__`, we suppress Python's default `__dict__`-based attribute management, reducing memory usage per node. This optimization is effective for linked lists that generate large numbers of nodes.

### 2.2 Basic Operations

#### 2.2.1 Insertion at the Head (prepend)

```
Head insertion operation (O(1)):

Before:
  head
   |
   v
  [B] -> [C] -> [D] -> None

Insert "A" at the head:

Step 1: Create a new node and set its next to the current head
  new_node        head
     |              |
     v              v
    [A] ----------> [B] -> [C] -> [D] -> None

Step 2: Update head to the new node
  head
   |
   v
  [A] -> [B] -> [C] -> [D] -> None

-> Only 2 pointer reassignments. O(1), independent of list length
```

#### 2.2.2 Insertion at the Tail (append)

```
Tail insertion operation (O(n)):

Before:
  head
   |
   v
  [A] -> [B] -> [C] -> None

Insert "D" at the tail:

Step 1: Traverse to the last node (O(n))
  head
   |
   v
  [A] -> [B] -> [C] -> None
                ^
              curr (last node found)

Step 2: Set the last node's next to the new node
  head
   |
   v
  [A] -> [B] -> [C] -> [D] -> None

-> O(n) traversal to reach the tail
-> Can be improved to O(1) by maintaining a separate tail pointer
```

#### 2.2.3 Deletion by Value (delete)

```
Middle node deletion operation (O(n)):

Before:
  head
   |
   v
  [A] -> [B] -> [C] -> [D] -> None

Delete "C":

Step 1: Traverse to the node immediately before the target
  head
   |
   v
  [A] -> [B] -> [C] -> [D] -> None
          ^     ^
         prev  target

Step 2: Reassign prev.next to target.next
  head
   |
   v                +-----------+
  [A] -> [B] -------+-> [D] -> None
                    |
          [C] -----+  <- Reference broken; eligible for GC

-> O(n) for traversal; the pointer reassignment itself is O(1)
```

#### 2.2.4 Dummy Head (Sentinel Node) Technique

Deleting the head node is a common edge case. By introducing a dummy head (sentinel node), every node has a "preceding node," eliminating the need for special-case handling.

```
Using a dummy head:

  dummy    head
   |        |
   v        v
  [0] --> [A] -> [B] -> [C] -> None
   ^
 Value unused (sentinel)

Deleting "A" (the head) requires no special treatment:

  dummy
   |
   v           +----------+
  [0] ---------+-> [B] -> [C] -> None
               |
     [A] -----+

Simply return dummy.next at the end
```

### 2.3 Complete Implementation

```python
class ListNode:
    """Node for a singly linked list"""
    __slots__ = ('val', 'next')

    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __repr__(self):
        return f"ListNode({self.val})"


class SinglyLinkedList:
    """Complete implementation of a singly linked list"""

    def __init__(self):
        self.head = None
        self._size = 0

    def __len__(self):
        return self._size

    def __iter__(self):
        """Implementation of the iterator protocol"""
        curr = self.head
        while curr:
            yield curr.val
            curr = curr.next

    def __repr__(self):
        values = []
        curr = self.head
        count = 0
        while curr and count < 20:  # Prevent infinite loops
            values.append(str(curr.val))
            curr = curr.next
            count += 1
        chain = " -> ".join(values)
        if curr:
            chain += " -> ..."
        return f"SinglyLinkedList([{chain}])"

    def is_empty(self):
        """Check if the list is empty: O(1)"""
        return self.head is None

    def prepend(self, val):
        """Insert at the head: O(1)"""
        self.head = ListNode(val, self.head)
        self._size += 1

    def append(self, val):
        """Insert at the tail: O(n)"""
        new_node = ListNode(val)
        if not self.head:
            self.head = new_node
        else:
            curr = self.head
            while curr.next:
                curr = curr.next
            curr.next = new_node
        self._size += 1

    def insert_at(self, index, val):
        """Insert at a specified index: O(n)

        Args:
            index: Insertion position (0-based). 0 for head, size for tail.
            val: Value to insert

        Raises:
            IndexError: If the index is out of range
        """
        if index < 0 or index > self._size:
            raise IndexError(f"Index {index} out of range [0, {self._size}]")
        if index == 0:
            self.prepend(val)
            return
        curr = self.head
        for _ in range(index - 1):
            curr = curr.next
        curr.next = ListNode(val, curr.next)
        self._size += 1

    def delete(self, val):
        """Delete the first occurrence of a value: O(n)

        Args:
            val: Value to delete

        Returns:
            bool: True if deletion succeeded, False if value not found
        """
        # Use the dummy head technique to avoid the special case of head deletion
        dummy = ListNode(0, self.head)
        curr = dummy
        while curr.next:
            if curr.next.val == val:
                curr.next = curr.next.next
                self.head = dummy.next
                self._size -= 1
                return True
            curr = curr.next
        return False

    def delete_at(self, index):
        """Delete the element at a specified index: O(n)

        Args:
            index: Deletion position (0-based)

        Returns:
            The deleted value

        Raises:
            IndexError: If the index is out of range
        """
        if index < 0 or index >= self._size:
            raise IndexError(f"Index {index} out of range [0, {self._size - 1}]")
        if index == 0:
            val = self.head.val
            self.head = self.head.next
            self._size -= 1
            return val
        curr = self.head
        for _ in range(index - 1):
            curr = curr.next
        val = curr.next.val
        curr.next = curr.next.next
        self._size -= 1
        return val

    def search(self, val):
        """Search for a value: O(n)

        Returns:
            int: The index where the value is found, or -1 if not found
        """
        curr = self.head
        index = 0
        while curr:
            if curr.val == val:
                return index
            curr = curr.next
            index += 1
        return -1

    def get(self, index):
        """Access by index: O(n)

        Args:
            index: Position to retrieve (0-based)

        Returns:
            The value at the specified position

        Raises:
            IndexError: If the index is out of range
        """
        if index < 0 or index >= self._size:
            raise IndexError(f"Index {index} out of range [0, {self._size - 1}]")
        curr = self.head
        for _ in range(index):
            curr = curr.next
        return curr.val

    def reverse(self):
        """Reverse the list (in-place): O(n)"""
        prev = None
        curr = self.head
        while curr:
            next_node = curr.next
            curr.next = prev
            prev = curr
            curr = next_node
        self.head = prev

    def to_list(self):
        """Convert to a Python list: O(n)"""
        return list(self)

    @classmethod
    def from_list(cls, values):
        """Build a linked list from a Python list: O(n)

        Args:
            values: An iterable sequence of values

        Returns:
            SinglyLinkedList: The constructed linked list
        """
        ll = cls()
        for val in reversed(values):
            ll.prepend(val)
        return ll


# --- Verification ---
if __name__ == "__main__":
    # Construction
    ll = SinglyLinkedList.from_list([1, 2, 3, 4, 5])
    print(ll)                   # SinglyLinkedList([1 -> 2 -> 3 -> 4 -> 5])
    print(f"Length: {len(ll)}")  # Length: 5

    # Insertion
    ll.prepend(0)
    ll.append(6)
    ll.insert_at(3, 99)
    print(ll)  # SinglyLinkedList([0 -> 1 -> 2 -> 99 -> 3 -> 4 -> 5 -> 6])

    # Search
    print(f"Search 99: index {ll.search(99)}")  # Search 99: index 3
    print(f"Get index 3: {ll.get(3)}")          # Get index 3: 99

    # Deletion
    ll.delete(99)
    deleted = ll.delete_at(0)
    print(f"Deleted: {deleted}")  # Deleted: 0
    print(ll)                     # SinglyLinkedList([1 -> 2 -> 3 -> 4 -> 5 -> 6])

    # Reversal
    ll.reverse()
    print(ll)  # SinglyLinkedList([6 -> 5 -> 4 -> 3 -> 2 -> 1])

    # Iteration
    for val in ll:
        print(val, end=" ")  # 6 5 4 3 2 1
    print()
```

### 2.4 Complexity Summary

| Operation | Time Complexity | Space Complexity | Notes |
|-----------|----------------|-----------------|-------|
| `prepend` | O(1) | O(1) | Head insertion. The most efficient operation |
| `append` | O(n) | O(1) | Requires tail traversal. Can be improved to O(1) with a tail pointer |
| `insert_at(i)` | O(i) | O(1) | Traversal to insertion point. O(1) if inserting at head |
| `delete(val)` | O(n) | O(1) | O(n) for searching the value |
| `delete_at(i)` | O(i) | O(1) | O(i) for traversal to the position |
| `search` | O(n) | O(1) | Linear search |
| `get(i)` | O(i) | O(1) | No random access |
| `reverse` | O(n) | O(1) | In-place reversal |
| `from_list` | O(n) | O(n) | Construction from a list |

### 2.5 Important Techniques

#### 2.5.1 List Reversal

Reversing a linked list is one of the most frequently asked interview topics. It is important to understand both the iterative and recursive versions.

```python
def reverse_iterative(head):
    """Reverse a linked list (iterative): O(n) time, O(1) space

    Uses three pointers (prev, curr, next_node) to
    reassign each node's next to point to the previous node.
    """
    prev = None
    curr = head
    while curr:
        next_node = curr.next   # Save the next node
        curr.next = prev        # Reverse the pointer
        prev = curr             # Advance prev
        curr = next_node        # Advance curr
    return prev  # New head


def reverse_recursive(head):
    """Reverse a linked list (recursive): O(n) time, O(n) space

    Recurse to the tail of the list and reverse pointers on the way back.
    Note that this uses O(n) space on the call stack.
    """
    # Base case: empty list or single node
    if not head or not head.next:
        return head
    # Recurse: reverse everything after head.next
    new_head = reverse_recursive(head.next)
    # Make head.next's node point back to head
    head.next.next = head
    head.next = None
    return new_head
```

```
Tracing the iterative version:

Initial state: 1 -> 2 -> 3 -> None
         prev=None, curr=1

Step 1: next_node=2, 1.next=None, prev=1, curr=2
        None <- 1    2 -> 3 -> None

Step 2: next_node=3, 2.next=1, prev=2, curr=3
        None <- 1 <- 2    3 -> None

Step 3: next_node=None, 3.next=2, prev=3, curr=None
        None <- 1 <- 2 <- 3

Result: 3 -> 2 -> 1 -> None (prev is the new head)
```

#### 2.5.2 Floyd's Cycle Detection (Tortoise and Hare)

An algorithm for detecting whether a cycle exists in a linked list. By advancing two pointers at different speeds, it detects cycles using O(1) extra space.

```python
def has_cycle(head):
    """Determine if a cycle exists: O(n) time, O(1) space

    slow (tortoise) advances 1 step at a time, fast (hare) advances 2 steps.
    If a cycle exists, they will eventually meet; otherwise, fast reaches the tail.
    """
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False


def detect_cycle_start(head):
    """Identify the starting node of a cycle: O(n) time, O(1) space

    Phase 1: Find the meeting point of slow and fast
    Phase 2: Advance from head and the meeting point at equal speed;
             the convergence point is the cycle start
    """
    slow = fast = head

    # Phase 1: Find the meeting point
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    else:
        return None  # No cycle

    # Phase 2: Identify the cycle start node
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    return slow
```

```
Floyd's cycle detection -- Why it works correctly:

With a cycle:
  head -> [1] -> [2] -> [3] -> [4] -> [5]
                         ^              |
                         +--------------+

  slow: 1, 2, 3, 4, 5, 3, 4, ...
  fast: 1, 3, 5, 4, 3, 5, 4, ...

  Step 1: slow=2, fast=3
  Step 2: slow=3, fast=5
  Step 3: slow=4, fast=4  <- They meet! (cycle exists)

Without a cycle:
  head -> [1] -> [2] -> [3] -> [4] -> None

  Step 1: slow=2, fast=3
  Step 2: slow=3, fast=None <- fast reaches None (no cycle)

Mathematical proof for identifying the cycle start node:
  F = distance from head to cycle start
  C = cycle length
  Let the meeting point be a nodes past the cycle start

  Distance traveled by slow: F + a
  Distance traveled by fast: F + a + kC (k = number of full cycles)

  Since fast travels twice as far as slow:
  2(F + a) = F + a + kC
  F + a = kC
  F = kC - a

  Thus, advancing F steps from head arrives at the same point
  as advancing F steps from the meeting point
  (= kC - a steps = the cycle start).
```

#### 2.5.3 Fast/Slow Pointer (Finding the Middle Node)

```python
def find_middle(head):
    """Find the middle node: O(n) time, O(1) space

    slow advances 1 step, fast advances 2 steps.
    When fast reaches the end, slow is at the middle.
    For even-length lists, returns the second of the two middle nodes.
    """
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow


def find_middle_first(head):
    """Find the middle node (returns the first middle for even-length lists)

    For an even-length list [1, 2, 3, 4]:
    - find_middle returns 3 (the second middle)
    - find_middle_first returns 2 (the first middle)
    """
    slow = fast = head
    while fast and fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
    return slow
```

```
Middle node detection examples:

Odd length: 1 -> 2 -> 3 -> 4 -> 5
  Step 0: slow=1, fast=1
  Step 1: slow=2, fast=3
  Step 2: slow=3, fast=5  <- fast.next=None, loop ends
  Result: slow=3 (exact middle)

Even length: 1 -> 2 -> 3 -> 4
  Step 0: slow=1, fast=1
  Step 1: slow=2, fast=3
  Step 2: slow=3, fast=None <- fast is None, loop ends
  Result: slow=3 (second middle)

  With find_middle_first:
  Step 0: slow=1, fast=1
  Step 1: slow=2, fast=3  <- fast.next.next=None, loop ends
  Result: slow=2 (first middle)
```

---

## 3. Doubly Linked List

### 3.1 Basic Structure

In a doubly linked list, each node holds both a "pointer to the previous node (prev)" and a "pointer to the next node (next)." This allows traversal in both directions from any node.

```
Structure of a doubly linked list:

  head                                                    tail
   |                                                       |
   v                                                       v
+----------+    +----------+    +----------+    +----------+
| prev:None|    | prev: <--+----| prev: <--+----| prev: <--|
| val: "A" |    | val: "B" |    | val: "C" |    | val: "D" |
| next: ---+--> | next: ---+--> | next: ---+--> | next:None|
+----------+    +----------+    +----------+    +----------+

  None <- [A] <=> [B] <=> [C] <=> [D] -> None

Characteristics:
- Each node holds two pointers: prev and next
- Accessible from both ends via head and tail
- Bidirectional traversal is possible
- Uses more memory per node than a singly linked list due to two pointers
```

### 3.2 Comparison with Singly Linked Lists

| Property | Singly Linked List | Doubly Linked List |
|----------|-------------------|-------------------|
| Pointers per node | 1 (next only) | 2 (prev + next) |
| Reverse traversal | Not possible (requires O(n) reversal) | O(1) by following prev |
| Deletion from the tail | O(n) (must find predecessor) | O(1) (access predecessor via tail.prev) |
| Deletion of an arbitrary node | O(n) (must find predecessor) | O(1) (if you have the node reference, access predecessor via prev) |
| Memory usage | Less | One extra pointer per node |
| Implementation complexity | Simple | More complex due to prev management |
| Primary use cases | Stacks, simple queues | LRU caches, text editors, browser history |

The greatest advantage of doubly linked lists is the ability to delete any node in O(1). In a singly linked list, deleting a node requires knowing the predecessor, which necessitates an O(n) traversal from the head. In a doubly linked list, the predecessor is immediately accessible via `node.prev`.

### 3.3 Complete Implementation

```python
class DListNode:
    """Node for a doubly linked list"""
    __slots__ = ('val', 'prev', 'next')

    def __init__(self, val=0, prev=None, next=None):
        self.val = val
        self.prev = prev
        self.next = next

    def __repr__(self):
        return f"DListNode({self.val})"


class DoublyLinkedList:
    """Complete implementation of a doubly linked list

    Uses sentinel nodes to simplify the implementation.
    head_sentinel and tail_sentinel always exist and hold no actual data.

    Structure:
    head_sentinel <=> [data1] <=> [data2] <=> ... <=> tail_sentinel
    """

    def __init__(self):
        # Initialize sentinel nodes
        self._head = DListNode(0)  # head sentinel
        self._tail = DListNode(0)  # tail sentinel
        self._head.next = self._tail
        self._tail.prev = self._head
        self._size = 0

    def __len__(self):
        return self._size

    def __iter__(self):
        """Forward iterator"""
        curr = self._head.next
        while curr != self._tail:
            yield curr.val
            curr = curr.next

    def __reversed__(self):
        """Reverse iterator"""
        curr = self._tail.prev
        while curr != self._head:
            yield curr.val
            curr = curr.prev

    def __repr__(self):
        values = list(self)
        return f"DoublyLinkedList({values})"

    def is_empty(self):
        """Check if the list is empty: O(1)"""
        return self._size == 0

    def _insert_between(self, val, predecessor, successor):
        """Insert a new node between two nodes: O(1)

        Internal helper method. All insertion operations reduce to this.
        """
        new_node = DListNode(val, predecessor, successor)
        predecessor.next = new_node
        successor.prev = new_node
        self._size += 1
        return new_node

    def _remove_node(self, node):
        """Remove a specified node: O(1)

        Internal helper method. All deletion operations reduce to this.
        Sentinel nodes are never removed.
        """
        predecessor = node.prev
        successor = node.next
        predecessor.next = successor
        successor.prev = predecessor
        self._size -= 1
        node.prev = node.next = None  # Clear references
        return node.val

    def prepend(self, val):
        """Insert at the head: O(1)"""
        return self._insert_between(val, self._head, self._head.next)

    def append(self, val):
        """Insert at the tail: O(1)

        In a doubly linked list, simply insert before the tail sentinel.
        A significant improvement over the O(n) of singly linked lists.
        """
        return self._insert_between(val, self._tail.prev, self._tail)

    def insert_at(self, index, val):
        """Insert at a specified index: O(min(i, n-i))

        Chooses the traversal direction based on proximity to head or tail.
        """
        if index < 0 or index > self._size:
            raise IndexError(f"Index {index} out of range [0, {self._size}]")

        # Determine traversal direction based on proximity to head or tail
        if index <= self._size // 2:
            curr = self._head.next
            for _ in range(index):
                curr = curr.next
            self._insert_between(val, curr.prev, curr)
        else:
            curr = self._tail.prev
            for _ in range(self._size - index - 1):
                curr = curr.prev
            self._insert_between(val, curr, curr.next)

    def pop_front(self):
        """Remove and return the head value: O(1)

        Raises:
            IndexError: If the list is empty
        """
        if self.is_empty():
            raise IndexError("pop from empty list")
        return self._remove_node(self._head.next)

    def pop_back(self):
        """Remove and return the tail value: O(1)

        Raises:
            IndexError: If the list is empty
        """
        if self.is_empty():
            raise IndexError("pop from empty list")
        return self._remove_node(self._tail.prev)

    def delete(self, val):
        """Delete the first occurrence of a value: O(n)

        Returns:
            bool: True if deletion succeeded
        """
        curr = self._head.next
        while curr != self._tail:
            if curr.val == val:
                self._remove_node(curr)
                return True
            curr = curr.next
        return False

    def get(self, index):
        """Access by index: O(min(i, n-i))

        Leverages bidirectional traversal, choosing direction based on
        proximity to head or tail.
        Worst case improves to O(n/2) compared to O(i) in singly linked lists.
        """
        if index < 0 or index >= self._size:
            raise IndexError(f"Index {index} out of range [0, {self._size - 1}]")

        if index <= self._size // 2:
            curr = self._head.next
            for _ in range(index):
                curr = curr.next
        else:
            curr = self._tail.prev
            for _ in range(self._size - 1 - index):
                curr = curr.prev
        return curr.val

    def front(self):
        """Peek at the head value (without removal): O(1)"""
        if self.is_empty():
            raise IndexError("front from empty list")
        return self._head.next.val

    def back(self):
        """Peek at the tail value (without removal): O(1)"""
        if self.is_empty():
            raise IndexError("back from empty list")
        return self._tail.prev.val

    def reverse(self):
        """Reverse the list (in-place): O(n)

        Swaps prev and next for every node.
        """
        curr = self._head
        while curr:
            curr.prev, curr.next = curr.next, curr.prev
            curr = curr.prev  # prev and next are swapped, so advance via prev
        # Swap sentinel nodes
        self._head, self._tail = self._tail, self._head

    def to_list(self):
        """Convert to a Python list: O(n)"""
        return list(self)

    @classmethod
    def from_list(cls, values):
        """Build from a Python list: O(n)"""
        dll = cls()
        for val in values:
            dll.append(val)
        return dll


# --- Verification ---
if __name__ == "__main__":
    dll = DoublyLinkedList.from_list([1, 2, 3, 4, 5])
    print(dll)                     # DoublyLinkedList([1, 2, 3, 4, 5])
    print(f"Length: {len(dll)}")    # Length: 5

    # Operations at both ends
    dll.prepend(0)
    dll.append(6)
    print(dll)                     # DoublyLinkedList([0, 1, 2, 3, 4, 5, 6])
    print(f"Front: {dll.front()}")  # Front: 0
    print(f"Back: {dll.back()}")    # Back: 6

    # Deletion from both ends
    dll.pop_front()
    dll.pop_back()
    print(dll)  # DoublyLinkedList([1, 2, 3, 4, 5])

    # Reverse traversal
    print("Reversed:", list(reversed(dll)))  # Reversed: [5, 4, 3, 2, 1]

    # Reversal
    dll.reverse()
    print(dll)  # DoublyLinkedList([5, 4, 3, 2, 1])
```

### 3.4 Design Intent of Sentinel Nodes

The design using sentinel nodes has clear advantages.

```
Without sentinels:

  head                    tail
   |                       |
   v                       v
  [A] <=> [B] <=> [C] <=> [D]

Inserting at the head:
  if self.head is None:       # Empty list case
      self.head = self.tail = new_node
  else:                        # Non-empty list case
      new_node.next = self.head
      self.head.prev = new_node
      self.head = new_node

With sentinels:

  head_sentinel              tail_sentinel
       |                          |
       v                          v
      [S] <=> [A] <=> [B] <=> [C] <=> [S]

Inserting at the head (always the same code):
  self._insert_between(val, self._head, self._head.next)

-> No need to handle the empty list case separately
-> Eliminates edge cases that are breeding grounds for bugs
-> Trade-off: Memory overhead of 2 extra nodes (usually negligible)
```

### 3.5 Complexity Summary

| Operation | Time Complexity | Comparison with Singly Linked List |
|-----------|----------------|-----------------------------------|
| `prepend` | O(1) | Same |
| `append` | O(1) | Singly: O(n) -> Major improvement |
| `insert_at(i)` | O(min(i, n-i)) | Singly: O(i) -> Improved |
| `pop_front` | O(1) | Same |
| `pop_back` | O(1) | Singly: O(n) -> Major improvement |
| `delete(node)` | O(1) | Singly: O(n) -> Major improvement |
| `get(i)` | O(min(i, n-i)) | Singly: O(i) -> Improved |
| `reverse` | O(n) | Same |

---

## 4. Circular Linked List

### 4.1 Basic Structure

In a circular linked list, the last node's `next` points to the head node. This enables seamless transition from the tail back to the head. Both singly circular and doubly circular linked lists exist.

```
Singly circular linked list:

  head
   |
   v
  [A] -> [B] -> [C] -> [D] --+
   ^                           |
   +---------------------------+

  The last node D's next points to the head A
  -> None does not exist
  -> The entire list is traversable from any node

Doubly circular linked list:

  head
   |
   v
  [A] <=> [B] <=> [C] <=> [D]
   ^                        |
   |    <------------------ | (prev)
   +--------------------->    (next)

  A.prev = D, D.next = A
  -> The distinction between head and tail becomes ambiguous
  -> Well-suited for ring buffer-like applications
```

### 4.2 Use Cases for Circular Lists

Here are scenarios where circular lists are particularly effective.

**1. Round-Robin Scheduling**
OS process schedulers commonly use round-robin scheduling to allocate equal CPU time to each process. With a circular list, the transition from the last process naturally returns to the first.

**2. Circular Buffer (Ring Buffer)**
A fixed-size buffer that wraps around to the beginning when the end is reached, overwriting old data. Used for streaming data processing and log retention.

**3. Multiplayer Games**
In turn-based games where multiple players take actions in sequence, the turn naturally returns to the first player after the last.

**4. Josephus Problem**
A classic problem where n people stand in a circle and every m-th person is eliminated in sequence. It can be naturally modeled with a circular list.

### 4.3 Complete Implementation

```python
class CircularLinkedList:
    """Singly circular linked list

    Design that maintains only a tail pointer.
    head is accessible in O(1) via tail.next.

    Empty list: tail = None
    1 node:     tail -> [A] -> (points back to itself)
    Multiple:   tail -> [C] -> [A] -> [B] -> [C] (returns to tail)

    Reason for maintaining tail:
    - head is obtainable via tail.next
    - Tail insertion is O(1) (insert right after tail and update tail)
    - If only head is maintained, tail insertion requires O(n) traversal
    """

    def __init__(self):
        self.tail = None
        self._size = 0

    def __len__(self):
        return self._size

    def __iter__(self):
        if not self.tail:
            return
        curr = self.tail.next  # Start from head
        for _ in range(self._size):
            yield curr.val
            curr = curr.next

    def __repr__(self):
        if not self.tail:
            return "CircularLinkedList([])"
        values = list(self)
        return f"CircularLinkedList({values}, circular)"

    def is_empty(self):
        """Check if the list is empty: O(1)"""
        return self.tail is None

    def prepend(self, val):
        """Insert at the head: O(1)"""
        new_node = ListNode(val)
        if not self.tail:
            new_node.next = new_node  # Points to itself
            self.tail = new_node
        else:
            new_node.next = self.tail.next  # New node -> old head
            self.tail.next = new_node       # tail -> new node (new head)
        self._size += 1

    def append(self, val):
        """Insert at the tail: O(1)

        Insert right after tail and update tail.
        """
        self.prepend(val)       # Insert at the head
        self.tail = self.tail.next  # Move tail to the new node -> effectively a tail insertion
        self._size += 0  # Already incremented in prepend

    def pop_front(self):
        """Remove and return the head value: O(1)

        Raises:
            IndexError: If the list is empty
        """
        if self.is_empty():
            raise IndexError("pop from empty circular list")
        head = self.tail.next
        val = head.val
        if self.tail == head:
            # Only one node
            self.tail = None
        else:
            self.tail.next = head.next
        self._size -= 1
        return val

    def rotate(self, k=1):
        """Rotate the list k times (head moves to tail): O(k)

        Corresponds to moving to the next process in round-robin scheduling.
        """
        if self.is_empty() or k == 0:
            return
        for _ in range(k % self._size):
            self.tail = self.tail.next

    def search(self, val):
        """Search for a value: O(n)

        Returns:
            bool: True if the value is found
        """
        if self.is_empty():
            return False
        curr = self.tail.next
        for _ in range(self._size):
            if curr.val == val:
                return True
            curr = curr.next
        return False

    def josephus(self, step):
        """Solve the Josephus problem

        Args:
            step: Elimination interval (every m-th person is eliminated)

        Returns:
            list: Elimination order
        """
        if self.is_empty():
            return []

        elimination_order = []
        curr = self.tail
        remaining = self._size

        while remaining > 0:
            # Advance step - 1 times (move to just before the step-th node)
            for _ in range(step - 1):
                curr = curr.next
            # Eliminate curr.next
            eliminated = curr.next
            elimination_order.append(eliminated.val)
            if curr == eliminated:
                # Last person
                curr = None
            else:
                curr.next = eliminated.next
            remaining -= 1

        self.tail = None
        self._size = 0
        return elimination_order


# --- Verification ---
if __name__ == "__main__":
    # Basic operations
    cl = CircularLinkedList()
    for i in range(1, 6):
        cl.append(i)
    print(cl)  # CircularLinkedList([1, 2, 3, 4, 5], circular)

    # Rotation
    cl.rotate(2)
    print(cl)  # CircularLinkedList([3, 4, 5, 1, 2], circular)

    # Josephus problem: 7 people in a circle, every 3rd person eliminated
    josephus = CircularLinkedList()
    for i in range(1, 8):
        josephus.append(i)
    order = josephus.josephus(3)
    print(f"Josephus elimination order: {order}")
    # [3, 6, 2, 7, 5, 1, 4]
```

### 4.4 Special Considerations for Circular Lists

The most dangerous aspect of working with circular lists is **infinite loops**. In a normal linked list, you can terminate a loop with `while curr:` or `while curr is not None:`, but in a circular list, `None` is never reached, so such conditions loop forever.

```
Danger of infinite loops:

  X Incorrect code:
    curr = head
    while curr:          # <- Always True in a circular list
        process(curr)
        curr = curr.next

  O Correct code (counter-based):
    curr = head
    for _ in range(size):  # <- Iterate exactly size times
        process(curr)
        curr = curr.next

  O Correct code (comparison with starting point):
    curr = head
    while True:
        process(curr)
        curr = curr.next
        if curr == head:   # <- Stop when back at the starting point
            break
```

### 4.5 Comparison of the Three List Types

| Property | Singly Linked List | Doubly Linked List | Circular Linked List |
|----------|-------------------|-------------------|---------------------|
| Node pointers | next only | prev + next | next (+ prev) |
| Tail -> Head | Not possible (O(n)) | Not possible (O(n)) | O(1) |
| Head -> Tail | O(n) | O(1) (with tail) | O(1) (with tail) |
| Memory/node | Low | Medium | Low to Medium |
| Traversal direction | Forward only | Bidirectional | Forward (circular) |
| End detection | next == None | next == sentinel | next == head |
| Infinite loop risk | Low | Low | High (caution required) |
| Typical use cases | Stacks, simple lists | LRU caches, Deques | Round-robin, Josephus |
| Implementation difficulty | Easy | Medium | Medium |

---

## 5. Implementation Patterns and Classic Algorithms

### 5.1 Merging Two Sorted Lists

Merging two sorted linked lists into a single sorted list. Used as a building block for merge sort and also frequently appears as a standalone interview problem.

```python
def merge_sorted_lists(l1, l2):
    """Merge two sorted linked lists: O(n + m) time, O(1) space

    Uses a dummy head for a concise implementation.
    Does not create new nodes; reconnects existing nodes.

    Args:
        l1: Head of the first sorted list
        l2: Head of the second sorted list

    Returns:
        Head of the merged list
    """
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

    # Attach the remaining nodes
    curr.next = l1 if l1 else l2
    return dummy.next
```

```
Merge operation trace:

l1: 1 -> 3 -> 5 -> None
l2: 2 -> 4 -> 6 -> None

dummy -> ?

Step 1: 1 <= 2 -> dummy -> 1, l1 = 3
Step 2: 3 > 2  -> dummy -> 1 -> 2, l2 = 4
Step 3: 3 <= 4 -> dummy -> 1 -> 2 -> 3, l1 = 5
Step 4: 5 > 4  -> dummy -> 1 -> 2 -> 3 -> 4, l2 = 6
Step 5: 5 <= 6 -> dummy -> 1 -> 2 -> 3 -> 4 -> 5, l1 = None
Remaining:      -> dummy -> 1 -> 2 -> 3 -> 4 -> 5 -> 6

Result: 1 -> 2 -> 3 -> 4 -> 5 -> 6
```

### 5.2 Sorting a Linked List (Merge Sort)

For sorting linked lists, merge sort is optimal. Quicksort, which performs well on arrays, suffers from the lack of random access in linked lists. Merge sort, on the other hand, operates using only sequential access, making it a natural fit for linked lists.

```python
def sort_linked_list(head):
    """Merge sort for a linked list: O(n log n) time, O(log n) space

    1. Split the list in half at the midpoint
    2. Recursively sort each half
    3. Merge the two sorted lists

    The O(log n) space complexity comes from the recursion stack depth.
    Unlike array merge sort, no additional arrays are needed.
    """
    # Base case: 0 or 1 node
    if not head or not head.next:
        return head

    # Split at the midpoint
    mid = find_middle_first(head)  # Get the first middle
    second_half = mid.next
    mid.next = None  # Cut the list

    # Recursively sort
    left = sort_linked_list(head)
    right = sort_linked_list(second_half)

    # Merge
    return merge_sorted_lists(left, right)


# --- Verification ---
if __name__ == "__main__":
    # Sort test
    def build_list(values):
        dummy = ListNode(0)
        curr = dummy
        for v in values:
            curr.next = ListNode(v)
            curr = curr.next
        return dummy.next

    def print_list(head):
        values = []
        while head:
            values.append(str(head.val))
            head = head.next
        print(" -> ".join(values))

    unsorted = build_list([4, 2, 1, 3, 5])
    print("Before sort:", end=" ")
    print_list(unsorted)  # 4 -> 2 -> 1 -> 3 -> 5

    sorted_head = sort_linked_list(unsorted)
    print("After sort: ", end=" ")
    print_list(sorted_head)  # 1 -> 2 -> 3 -> 4 -> 5
```

### 5.3 Palindrome Detection

Determine whether a linked list is a palindrome (reads the same forward and backward).

```python
def is_palindrome(head):
    """Palindrome check for a linked list: O(n) time, O(1) space

    1. Find the midpoint using Fast/Slow pointers
    2. Reverse the second half
    3. Compare the first half with the second half
    4. Restore the second half (avoid destroying the list)
    """
    if not head or not head.next:
        return True

    # Step 1: Find the midpoint
    slow = fast = head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next

    # Step 2: Reverse the second half
    second_half = reverse_iterative(slow.next)

    # Step 3: Compare
    first_half = head
    second_half_copy = second_half  # Save for restoration later
    is_palin = True

    while second_half:
        if first_half.val != second_half.val:
            is_palin = False
            break
        first_half = first_half.next
        second_half = second_half.next

    # Step 4: Restore the second half
    slow.next = reverse_iterative(second_half_copy)

    return is_palin
```

```
Palindrome detection example:

List: 1 -> 2 -> 3 -> 2 -> 1

Step 1: Midpoint = 3 (slow points to 3)
  First half:  1 -> 2 -> 3
  Second half: 2 -> 1

Step 2: Reverse the second half
  Second half: 1 -> 2

Step 3: Compare
  First half: 1, 2  <->  Second half: 1, 2  Match

Result: True (palindrome)
```

### 5.4 Finding the Intersection of Two Lists

When two singly linked lists merge at some point, find the merge node.

```python
def find_intersection(headA, headB):
    """Detect the intersection of two lists: O(n + m) time, O(1) space

    Uses pointers A and B. When A reaches the end of list A, it moves
    to the head of list B, and vice versa for B.
    Since both pointers travel the same total distance, they meet at the intersection.

    If there is no intersection, both reach None and the loop terminates.
    """
    if not headA or not headB:
        return None

    a, b = headA, headB

    # Continue until both pointers meet or both become None
    while a != b:
        a = a.next if a else headB
        b = b.next if b else headA

    return a  # Intersection node, or None (no intersection)
```

```
Intersection detection trace:

List A: 1 -> 2 -+
                 +-> 8 -> 9 -> None
List B: 3 -> 4 -> 5 -+

Length: A = 4, B = 5

Pointer A's path: 1->2->8->9->(B start)->3->4->5->8
Pointer B's path: 3->4->5->8->9->(A start)->1->2->8

Both meet at 8 (the intersection).
Total distance traveled: A = 4 + 4 = 8, B = 5 + 3 = 8 (equal!)

Mathematical basis:
  Length of A's unique part: a
  Length of B's unique part: b
  Length of the shared part: c

  Total distance for pointer A: a + c + b
  Total distance for pointer B: b + c + a
  -> Always equal, so they meet at the intersection
```

---

## 6. Applications: Practical Data Structures

### 6.1 LRU Cache (Least Recently Used Cache)

An LRU cache is a practical data structure combining a doubly linked list with a hash map, and is a frequent interview topic. When the cache exceeds its capacity, it evicts the least recently used entry.

```
LRU Cache structure:

  +---------------------------------------------+
  |  HashMap: key -> DListNode                    |
  |  +------+------+------+------+              |
  |  | k1-> | k2-> | k3-> | k4-> |              |
  |  +--+---+--+---+--+---+--+---+              |
  |     |      |      |      |                   |
  |     v      v      v      v                   |
  |  [head] <=> [k1] <=> [k2] <=> [k3] <=> [k4] <=> [tail]  |
  |  sentinel  <- most recent --- oldest ->  sentinel |
  |                                               |
  |  - On every get/put, move the accessed node to the front |
  |  - On capacity overflow, remove from the tail side (oldest) |
  +---------------------------------------------+

  All operations O(1):
  - get: O(1) lookup via HashMap + O(1) move node to front
  - put: O(1) insertion via HashMap + O(1) add node at front
  - evict: O(1) removal from tail side + O(1) deletion from HashMap
```

```python
class LRUCache:
    """LRU Cache: Doubly linked list + Hash map

    All operations (get, put) run in O(1).

    Attributes:
        capacity: Maximum cache capacity
        cache: Mapping from key to DListNode
        _head: Sentinel node (most recent side)
        _tail: Sentinel node (oldest side)
    """

    class _Node:
        __slots__ = ('key', 'val', 'prev', 'next')

        def __init__(self, key=0, val=0):
            self.key = key
            self.val = val
            self.prev = None
            self.next = None

    def __init__(self, capacity):
        """
        Args:
            capacity: Maximum cache capacity (must be >= 1)

        Raises:
            ValueError: If capacity is less than 1
        """
        if capacity < 1:
            raise ValueError("Capacity must be at least 1")
        self.capacity = capacity
        self.cache = {}  # key -> Node

        # Sentinel nodes
        self._head = self._Node()
        self._tail = self._Node()
        self._head.next = self._tail
        self._tail.prev = self._head

    def _remove(self, node):
        """Remove a node from the list: O(1)"""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_front(self, node):
        """Add a node to the front of the list (right after head): O(1)"""
        node.next = self._head.next
        node.prev = self._head
        self._head.next.prev = node
        self._head.next = node

    def _move_to_front(self, node):
        """Move a node to the front of the list: O(1)"""
        self._remove(node)
        self._add_to_front(node)

    def get(self, key):
        """Retrieve the value for a key: O(1)

        If found, the entry is moved to the front as "most recently used."

        Args:
            key: The lookup key

        Returns:
            The value. Returns -1 if the key does not exist.
        """
        if key in self.cache:
            node = self.cache[key]
            self._move_to_front(node)
            return node.val
        return -1

    def put(self, key, value):
        """Insert or update a key-value pair: O(1)

        For existing keys, updates the value and moves to front.
        For new keys, inserts at the front and evicts the oldest if over capacity.

        Args:
            key: The key
            value: The value
        """
        if key in self.cache:
            # Existing key: update value and move to front
            node = self.cache[key]
            node.val = value
            self._move_to_front(node)
        else:
            # New key
            if len(self.cache) >= self.capacity:
                # Over capacity: evict the oldest entry (just before tail)
                lru_node = self._tail.prev
                self._remove(lru_node)
                del self.cache[lru_node.key]

            # Add new node at the front
            new_node = self._Node(key, value)
            self.cache[key] = new_node
            self._add_to_front(new_node)

    def __repr__(self):
        items = []
        node = self._head.next
        while node != self._tail:
            items.append(f"{node.key}:{node.val}")
            node = node.next
        return f"LRUCache({' -> '.join(items)}, cap={self.capacity})"


# --- Verification ---
if __name__ == "__main__":
    lru = LRUCache(3)

    lru.put("a", 1)
    lru.put("b", 2)
    lru.put("c", 3)
    print(lru)  # LRUCache(c:3 -> b:2 -> a:1, cap=3)

    # Access "a" -> moves to front
    print(lru.get("a"))  # 1
    print(lru)  # LRUCache(a:1 -> c:3 -> b:2, cap=3)

    # Add new key "d" -> oldest "b" is evicted
    lru.put("d", 4)
    print(lru)  # LRUCache(d:4 -> a:1 -> c:3, cap=3)

    # Access evicted "b" -> -1
    print(lru.get("b"))  # -1
```

### 6.2 Polynomial Representation and Arithmetic

Linked lists are also suitable for representing polynomials. Each term (coefficient and exponent) is held as a node, and adding or removing terms can be performed via pointer operations.

```python
class TermNode:
    """Node representing a polynomial term"""
    __slots__ = ('coef', 'exp', 'next')

    def __init__(self, coef, exp, next=None):
        self.coef = coef  # Coefficient
        self.exp = exp    # Exponent
        self.next = next

    def __repr__(self):
        if self.exp == 0:
            return f"{self.coef}"
        elif self.exp == 1:
            return f"{self.coef}x"
        else:
            return f"{self.coef}x^{self.exp}"


class Polynomial:
    """Polynomial class (implemented with a linked list)

    Terms are sorted in descending order of exponents.
    Example: 3x^3 + 2x^2 + 5x + 1
        -> [3,3] -> [2,2] -> [5,1] -> [1,0] -> None
    """

    def __init__(self):
        self.head = None

    def add_term(self, coef, exp):
        """Add a term (maintaining descending exponent order)"""
        if coef == 0:
            return

        new_node = TermNode(coef, exp)

        # Empty list or insert at the head
        if not self.head or self.head.exp < exp:
            new_node.next = self.head
            self.head = new_node
            return

        # If a term with the same exponent exists, add coefficients
        if self.head.exp == exp:
            self.head.coef += coef
            if self.head.coef == 0:
                self.head = self.head.next
            return

        # Search for the insertion position
        curr = self.head
        while curr.next and curr.next.exp > exp:
            curr = curr.next

        if curr.next and curr.next.exp == exp:
            curr.next.coef += coef
            if curr.next.coef == 0:
                curr.next = curr.next.next
        else:
            new_node.next = curr.next
            curr.next = new_node

    def __add__(self, other):
        """Polynomial addition: O(n + m)"""
        result = Polynomial()
        p1, p2 = self.head, other.head

        while p1 and p2:
            if p1.exp > p2.exp:
                result.add_term(p1.coef, p1.exp)
                p1 = p1.next
            elif p1.exp < p2.exp:
                result.add_term(p2.coef, p2.exp)
                p2 = p2.next
            else:
                total_coef = p1.coef + p2.coef
                if total_coef != 0:
                    result.add_term(total_coef, p1.exp)
                p1 = p1.next
                p2 = p2.next

        while p1:
            result.add_term(p1.coef, p1.exp)
            p1 = p1.next
        while p2:
            result.add_term(p2.coef, p2.exp)
            p2 = p2.next

        return result

    def evaluate(self, x):
        """Evaluate the polynomial by substituting x: O(n)"""
        result = 0
        curr = self.head
        while curr:
            result += curr.coef * (x ** curr.exp)
            curr = curr.next
        return result

    def __repr__(self):
        if not self.head:
            return "0"
        terms = []
        curr = self.head
        while curr:
            terms.append(repr(curr))
            curr = curr.next
        return " + ".join(terms)


# --- Verification ---
if __name__ == "__main__":
    # p1 = 3x^3 + 2x^2 + 5
    p1 = Polynomial()
    p1.add_term(3, 3)
    p1.add_term(2, 2)
    p1.add_term(5, 0)
    print(f"p1 = {p1}")  # p1 = 3x^3 + 2x^2 + 5

    # p2 = x^3 + 4x + 2
    p2 = Polynomial()
    p2.add_term(1, 3)
    p2.add_term(4, 1)
    p2.add_term(2, 0)
    print(f"p2 = {p2}")  # p2 = 1x^3 + 4x + 2

    # Addition
    p3 = p1 + p2
    print(f"p1 + p2 = {p3}")  # p1 + p2 = 4x^3 + 2x^2 + 4x + 7

    # Evaluation
    print(f"p3(2) = {p3.evaluate(2)}")  # p3(2) = 4*8 + 2*4 + 4*2 + 7 = 55
```

### 6.3 Skip List Concept

A skip list is a data structure that adds multiple levels of "express lanes" to a linked list, speeding up search to O(log n). It is used as an alternative to balanced binary search trees and is employed in implementations such as Redis's Sorted Set.

```
Skip list structure (conceptual diagram):

Level 3: head ------------------------------------------- 9 -> None
Level 2: head ---------- 3 ------------------- 7 --- 9 -> None
Level 1: head ---- 2 -- 3 ---- 5 ---- 7 ---- 9 -> None
Level 0: head -> 1, 2, 3, 4, 5, 6, 7, 8, 9 -> None
         (a regular linked list containing all elements)

Search example: searching for value 6
  Level 3: head -> 9 (too large; go down)
  Level 2: head -> 3 (OK) -> 7 (too large; go down)
  Level 1: 3 -> 5 (OK) -> 7 (too large; go down)
  Level 0: 5 -> 6 (found!)

-> O(log n) via skipping, rather than O(n) sequential traversal
-> Each node's level is determined randomly (probabilistic data structure)
```

A detailed skip list implementation is beyond the scope of this chapter, but it is important to be aware of it as an extension of linked lists. The idea of overcoming the weakness of linked lists -- slow random access -- through a probabilistic hierarchical structure is highly instructive.

### 6.4 XOR Linked List (Memory Optimization)

An XOR linked list is an optimization to reduce the memory usage of doubly linked lists. Instead of storing two pointers (prev and next) per node, each node stores only a single value: `prev XOR next`.

```
XOR linked list principle:

Regular doubly linked list (2 pointers per node):
  [A] <=> [B] <=> [C] <=> [D]
  B.prev = addr(A), B.next = addr(C)

XOR linked list (1 pointer per node):
  [A] - [B] - [C] - [D]
  B.npx = addr(A) XOR addr(C)

Traversal mechanism (moving from A toward C):
  Previous node's address: addr(A) is known
  B.npx = addr(A) XOR addr(C)
  addr(C) = B.npx XOR addr(A)  <- XOR property: a XOR b XOR a = b

Memory savings: 1 pointer per node
Trade-off: Poor compatibility with garbage collectors (not practical in Python, Java)
    -> Only practical in C/C++
```

XOR linked lists are difficult to implement in languages like Python (since direct address manipulation is unavailable), but they are a practical technique in C/C++. Knowing the concept broadens your thinking about memory optimization.

### 6.5 Self-Organizing Lists

A self-organizing list dynamically changes the order of nodes based on access patterns, reducing search time for frequently accessed elements.

```
Primary strategies for self-organizing lists:

1. Move-to-Front (MTF): Move the accessed node to the front
   Before: [D] -> [B] -> [A] -> [C]
   Access A:
   After:  [A] -> [D] -> [B] -> [C]
   -> Same idea as an LRU cache

2. Transpose: Move the accessed node one position forward
   Before: [D] -> [B] -> [A] -> [C]
   Access A:
   After:  [D] -> [A] -> [B] -> [C]
   -> Gentler reorganization than MTF

3. Count: Sort by access frequency
   Before: [D:3] -> [B:1] -> [A:5] -> [C:2]
   Access A (count 5->6):
   After:  [A:6] -> [D:3] -> [C:2] -> [B:1]
   -> Most accurate but requires counter management
```

The LRU cache is an application of the Move-to-Front strategy and is the most widely used strategy in practice.

---

## 7. Comparison Analysis with Arrays

### 7.1 Comprehensive Complexity Comparison

```
+---------------------+--------------+--------------+----------------+
| Operation           | Array        | Singly       | Doubly         |
|                     | (dynamic)    | Linked List  | Linked List    |
+---------------------+--------------+--------------+----------------+
| Insert at head      | O(n)         | O(1)  *      | O(1)  *        |
| Insert at tail      | O(1) amort * | O(n)         | O(1)  *        |
| Insert at middle    | O(n)         | O(n)         | O(n)           |
| Insert at middle    | O(n)         | O(1)*        | O(1)*  *       |
| (position known)    |              |              |                |
| Delete from head    | O(n)         | O(1)  *      | O(1)  *        |
| Delete from tail    | O(1) *       | O(n)         | O(1)  *        |
| Delete arbitrary    | O(n)         | O(n)         | O(1)*  *       |
| node (ref known)    |              |              |                |
| Index access        | O(1)  *      | O(n)         | O(min(i,n-i))  |
| Search by value     | O(n)         | O(n)         | O(n)           |
| Search (sorted)     | O(log n) *   | O(n)         | O(n)           |
| Memory efficiency   | Good *       | Somewhat poor| Poor           |
| Cache locality      | Good *       | Poor         | Poor           |
+---------------------+--------------+--------------+----------------+

* = superior operation
* = when the pointer to the insertion/deletion position is already known
```

### 7.2 Detailed Memory Usage Comparison

Analyzing memory usage based on data type and element count.

```
Memory required to store n integer values on a 64-bit system:

Array (Python list):
  Overhead: 56 bytes (list object)
  Per element: 8 bytes (pointer) + 28 bytes (int object) = 36 bytes
  Total: 56 + 36n bytes
  * Actual implementation includes over-allocation (approximately 1.125x)

Singly linked list:
  Per node: 28 bytes (int) + 8 bytes (next pointer)
           + 56 bytes object header (less with __slots__)
  Total: approximately 72n bytes (more without __slots__)

Doubly linked list:
  Per node: 28 bytes (int) + 16 bytes (prev + next pointers)
           + object header
  Total: approximately 80n bytes + 2 sentinel nodes

For n = 1,000,000 (1 million elements):
  Array:              approximately 36 MB
  Singly linked list: approximately 72 MB (about 2x the array)
  Doubly linked list:  approximately 80 MB (about 2.2x the array)
```

### 7.3 Impact of Cache Locality

In modern processors, CPU cache utilization has a significant impact on performance. Arrays are contiguous in memory, so a single cache line load provides access to multiple elements. Linked list nodes, on the other hand, are scattered throughout memory, resulting in a high probability of cache misses for each node access.

```
Cache locality comparison:

Array (contiguous memory):
  A cache line (64 bytes) can hold 8 ints
  +------------------------------------------+
  | [1] [2] [3] [4] [5] [6] [7] [8]         | <- 8 elements in one load
  +------------------------------------------+
  Next cache line is also contiguous -> prefetchable

Linked list (scattered memory):
  Cache line 1: [Node A: val=1, next=0x...]  <- Only 1 node per load
  Cache line 2: [other data]                  <- Cache miss
  Cache line 3: [Node B: val=2, next=0x...]  <- Cache miss
  Cache line 4: [other data]                  <- Cache miss
  ...

  -> Even for traversing the same n elements, measured performance
     can differ by 5x to 10x or more
  -> O(n) vs O(n) but with vastly different constant factors

Exception: Using a memory pool (allocating nodes from the same region
      consecutively) can improve cache locality to some extent
```

### 7.4 Guidelines for Choosing Between Arrays and Linked Lists

```
When to use arrays:
  V Frequent random access
  V Append/delete at the tail is the primary operation
  V Data size is known or varies little
  V Frequent search operations (especially on sorted data)
  V Cache efficiency is important
  V Minimizing memory usage
  -> Arrays are the optimal solution in most cases

When to use linked lists:
  V Frequent insertion/deletion at the head
  V Frequent insertion/deletion at known middle positions
  V Worst-case time guarantees are needed (avoiding dynamic array resizing)
  V As an internal structure for other data structures (LRU cache, hash table chaining, etc.)
  V Frequent reordering of elements
  -> Advantageous only in specific use cases

Practical rule of thumb:
  -> "Start with an array, and consider a linked list only if profiling reveals a problem"
  -> Linked lists are optimal in roughly 5-10% of cases
  -> In those cases, however, linked lists offer overwhelming advantages
```

### 7.5 Linked Lists in Standard Libraries of Various Languages

```
Availability of linked lists across languages:

+----------+-----------------+------------------------------+
| Language | Class Name      | Notes                        |
+----------+-----------------+------------------------------+
| Python   | collections.deque| Doubly linked. list is dynamic array |
| Java     | LinkedList      | Doubly linked. Implements List/Deque |
| C++      | std::list       | Doubly linked                |
|          | std::forward_list| Singly linked (C++11+)       |
| C#       | LinkedList<T>   | Doubly linked                |
| Go       | container/list  | Doubly linked                |
| Rust     | std::collections| LinkedList (doubly linked)   |
|          | ::LinkedList    | * Practically discouraged    |
| Haskell  | [ ] (list)      | Singly linked. Core language feature |
+----------+-----------------+------------------------------+

Notable points:
- Python does not provide an explicit LinkedList class
  -> deque uses a doubly linked list internally
  -> list (dynamic array) is recommended in most scenarios
- Rust discourages the use of LinkedList, favoring Vec
  -> Due to cache locality considerations
- In Haskell, lists (singly linked lists) are the most fundamental data structure
  -> Immutable lists are a natural choice in functional programming
```

---

## 8. Anti-Patterns

### 8.1 Anti-Pattern 1: Unnecessary Use of Linked Lists

**Problem:**
Using a linked list in scenarios where an array would suffice. This is a misguided choice based on the superficial understanding that "linked lists are faster for insertion and deletion."

```python
# X Anti-pattern: An array is sufficient for tail appends and traversals
class BadLogBuffer:
    """Using a linked list as a log buffer (not recommended)"""
    def __init__(self):
        self.head = None
        self.tail = None

    def add_log(self, message):
        """Append a log to the tail"""
        node = ListNode(message)
        if not self.tail:
            self.head = self.tail = node
        else:
            self.tail.next = node
            self.tail = node

    def get_all_logs(self):
        """Retrieve all logs"""
        logs = []
        curr = self.head
        while curr:
            logs.append(curr.val)
            curr = curr.next
        return logs


# O Correct approach: Use an array (list)
class GoodLogBuffer:
    """Using an array as a log buffer (recommended)"""
    def __init__(self):
        self.logs = []

    def add_log(self, message):
        """Append a log to the tail: O(1) amortized"""
        self.logs.append(message)

    def get_all_logs(self):
        """Retrieve all logs: O(n), good cache efficiency"""
        return list(self.logs)

    def get_log(self, index):
        """Retrieve a specific log: O(1)"""
        return self.logs[index]
```

**Why an array is more appropriate:**
- The primary operations are "tail append" and "full traversal" -- both are efficient with arrays
- Random access (referencing a specific log) is O(1)
- Cache locality makes traversal fast
- Less memory usage
- Simpler code

**Criteria for choosing a linked list:**
1. Are insertions/deletions at the head or middle frequent?
2. Do you have a pointer to the insertion/deletion position?
3. Do you need worst-case time guarantees?

If the answer to all three is "no," choose an array.

### 8.2 Anti-Pattern 2: Memory Leaks and Dangling Pointers

**Problem:**
Failing to clean up pointers when deleting nodes, leaving references that prevent garbage collection. In Python, the garbage collector covers most cases, but caution is needed with circular references.

```python
# X Anti-pattern: Memory leak from circular references
class BadNode:
    def __init__(self, val):
        self.val = val
        self.next = None
        self.parent_list = None  # Back-reference to the list

class BadLinkedList:
    def __init__(self):
        self.head = None

    def append(self, val):
        node = BadNode(val)
        node.parent_list = self  # Creates a circular reference!
        if not self.head:
            self.head = node
        else:
            curr = self.head
            while curr.next:
                curr = curr.next
            curr.next = node

    def clear(self):
        """Clear the list (potential memory leak)"""
        self.head = None
        # <- Each node still references self (the list)
        # <- The node <-> list circular reference may not be collected by GC


# O Correct approach: Avoid circular references, or explicitly clear
class GoodLinkedList:
    def __init__(self):
        self.head = None

    def clear(self):
        """Clear with memory leak prevention"""
        curr = self.head
        while curr:
            next_node = curr.next
            curr.next = None       # Clear reference
            curr = next_node
        self.head = None
```

**Lessons:**
- Back-references from nodes to the list object create circular references
- `prev`/`next` in doubly linked lists are technically circular references, but Python's generational GC normally collects them
- For large lists that are frequently created and destroyed, explicitly clearing references is safer
- In C/C++, where memory is manually managed, this problem becomes even more severe

### 8.3 Anti-Pattern 3: O(n^2) from Repeated O(n) Operations

**Problem:**
Forgetting that `get(i)` is O(n) on a linked list and repeatedly performing index-based access in a loop.

```python
# X Anti-pattern: O(n^2) from index-based access
def print_all_bad(linked_list):
    """Print all elements (not recommended: O(n^2))"""
    for i in range(len(linked_list)):
        # get(i) traverses from head i times each call -> O(i)
        # Total: 0 + 1 + 2 + ... + (n-1) = O(n^2)
        print(linked_list.get(i))


# O Correct approach: Use an iterator for O(n) traversal
def print_all_good(linked_list):
    """Print all elements (recommended: O(n))"""
    for val in linked_list:  # Sequential traversal via __iter__
        print(val)


# O Correct approach: Use node pointers directly
def print_all_direct(head):
    """Print all elements (recommended: O(n))"""
    curr = head
    while curr:
        print(curr.val)
        curr = curr.next
```

**Lessons:**
- Every index-based access on a linked list incurs an O(n) traversal
- Using `for i in range(n): list.get(i)` with array thinking results in O(n^2)
- Traverse linked lists using iterators or direct pointer traversal
- This same trap exists in Java's `LinkedList`: use `for (T item : list)` instead of `for (int i = 0; i < list.size(); i++)`

### 8.4 Anti-Pattern 4: Lack of Thread Safety

**Problem:**
Sharing a linked list across multiple threads without proper synchronization mechanisms.

```python
import threading

# X Anti-pattern: Non-thread-safe list operations
class UnsafeSharedList:
    def __init__(self):
        self.head = None

    def prepend(self, val):
        # If threads A and B prepend simultaneously:
        # A: new_node.next = self.head  (head = [1])
        # B: new_node.next = self.head  (head = [1])  <- Both read the same head
        # A: self.head = new_node_A     (head = [A, 1])
        # B: self.head = new_node_B     (head = [B, 1])  <- A's node is lost!
        self.head = ListNode(val, self.head)


# O Correct approach: Protect with a lock
class ThreadSafeList:
    def __init__(self):
        self.head = None
        self._lock = threading.Lock()

    def prepend(self, val):
        with self._lock:
            self.head = ListNode(val, self.head)

    def pop_front(self):
        with self._lock:
            if not self.head:
                raise IndexError("pop from empty list")
            val = self.head.val
            self.head = self.head.next
            return val
```

---

## 9. Exercises

### 9.1 Fundamentals

**Exercise 1: Basic Linked List Operations**

Implement a singly linked list with the following operations:

- `prepend(val)`: Insert a value at the head
- `append(val)`: Insert a value at the tail
- `delete(val)`: Delete the first occurrence of a value
- `search(val)`: Determine if a value exists
- `__len__()`: Return the length of the list
- `to_list()`: Convert to a Python list

```python
# Test cases
ll = SinglyLinkedList()
ll.append(1)
ll.append(2)
ll.append(3)
ll.prepend(0)
assert ll.to_list() == [0, 1, 2, 3]
assert len(ll) == 4
assert ll.search(2) == 2  # index 2
assert ll.search(99) == -1
ll.delete(2)
assert ll.to_list() == [0, 1, 3]
```

**Hint:** Using a dummy head (sentinel node) eliminates the need to treat head node deletion as a special case.

---

**Exercise 2: Linked List Reversal**

Implement a function that reverses a given singly linked list, in both iterative and recursive versions.

```python
# Test cases
head = build_list([1, 2, 3, 4, 5])
reversed_head = reverse_iterative(head)
assert list_to_array(reversed_head) == [5, 4, 3, 2, 1]

head2 = build_list([1, 2, 3, 4, 5])
reversed_head2 = reverse_recursive(head2)
assert list_to_array(reversed_head2) == [5, 4, 3, 2, 1]

# Edge cases
assert reverse_iterative(None) is None  # Empty list
assert reverse_iterative(ListNode(42)).val == 42  # Single element
```

**Hint (iterative):** Use three pointers `prev`, `curr`, `next_node`, and at each step reassign `curr.next` to `prev`.

**Hint (recursive):** The base case is "empty list or single node." Recurse to the tail and reverse pointers on the way back.

---

**Exercise 3: Remove the N-th Node from the End**

Implement a function that removes the N-th node from the end of a singly linked list. Solve it in a single pass (do not traverse the list twice).

```python
# Test cases
head = build_list([1, 2, 3, 4, 5])
# Remove 2nd from end (= 4)
new_head = remove_nth_from_end(head, 2)
assert list_to_array(new_head) == [1, 2, 3, 5]

# Edge case: Remove the head
head2 = build_list([1, 2])
new_head2 = remove_nth_from_end(head2, 2)
assert list_to_array(new_head2) == [2]
```

**Hint:** Use two pointers `fast` and `slow`. First advance `fast` by N steps, then advance both `fast` and `slow` simultaneously. When `fast` reaches the end, `slow` is just before the node to be removed.

### 9.2 Intermediate

**Exercise 4: Merge Two Sorted Lists**

Implement a function that merges two sorted singly linked lists into a single sorted list. Do not create new nodes; reconnect the existing ones.

```python
# Test cases
l1 = build_list([1, 3, 5, 7])
l2 = build_list([2, 4, 6, 8])
merged = merge_sorted(l1, l2)
assert list_to_array(merged) == [1, 2, 3, 4, 5, 6, 7, 8]

# Edge case
l3 = build_list([1, 2, 3])
merged2 = merge_sorted(l3, None)
assert list_to_array(merged2) == [1, 2, 3]
```

---

**Exercise 5: Cycle Detection and Cycle Start Identification**

Implement the following two functions:
1. `has_cycle(head)`: Determine whether a cycle exists
2. `find_cycle_start(head)`: Return the starting node of the cycle (or None if no cycle)

Use O(1) extra space (hash set usage is not allowed).

```python
# Test cases
# 1 -> 2 -> 3 -> 4 -> 5 -> 3 (cycle)
nodes = [ListNode(i) for i in range(1, 6)]
for i in range(4):
    nodes[i].next = nodes[i + 1]
nodes[4].next = nodes[2]  # 5 -> 3 creates a cycle

assert has_cycle(nodes[0]) is True
assert find_cycle_start(nodes[0]).val == 3

# No cycle
normal = build_list([1, 2, 3])
assert has_cycle(normal) is False
assert find_cycle_start(normal) is None
```

**Hint:** Use Floyd's cycle detection algorithm (tortoise and hare). To identify the cycle start node, use the technique of advancing from both the meeting point and head at equal speed.

---

**Exercise 6: Linked List Palindrome Check**

Implement a function that determines whether a singly linked list is a palindrome. Solve it in O(n) time and O(1) extra space.

```python
# Test cases
assert is_palindrome(build_list([1, 2, 3, 2, 1])) is True
assert is_palindrome(build_list([1, 2, 2, 1])) is True
assert is_palindrome(build_list([1, 2, 3])) is False
assert is_palindrome(build_list([1])) is True
assert is_palindrome(None) is True
```

**Hint:** (1) Find the midpoint, (2) reverse the second half, (3) compare the first and second halves. After comparison, re-reverse the second half to restore the list.

### 9.3 Advanced

**Exercise 7: LRU Cache Implementation**

Using a doubly linked list and hash map, implement an LRU cache where all operations run in O(1).

- `get(key)`: Return the value for the key. Return -1 if the key does not exist. Mark the accessed entry as "most recently used."
- `put(key, value)`: Insert a key-value pair. If the key already exists, update the value. If the capacity is exceeded, evict the oldest entry.

```python
# Test cases
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
assert cache.get(1) == 1       # Access 1 (becomes most recently used)
cache.put(3, 3)                 # Over capacity -> 2 is evicted
assert cache.get(2) == -1       # 2 has been evicted
cache.put(4, 4)                 # Over capacity -> 1 is evicted
assert cache.get(1) == -1       # 1 has been evicted
assert cache.get(3) == 3
assert cache.get(4) == 4
```

---

**Exercise 8: Merge K Sorted Lists**

Merge K sorted lists into a single sorted list. Given N as the total number of elements across all lists, solve it in O(N log K) time.

```python
# Test cases
lists = [
    build_list([1, 4, 7]),
    build_list([2, 5, 8]),
    build_list([3, 6, 9]),
]
merged = merge_k_sorted(lists)
assert list_to_array(merged) == [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

**Hint:** Use a min-heap (priority queue). Insert the head element of each list into the heap, extract the minimum, add it to the result, and insert the next element from that list into the heap.

---

**Exercise 9: Text Editor Buffer Using a Linked List**

Using a doubly linked list, implement a simple text editor buffer with the following operations:

- `insert(char)`: Insert a character at the cursor position and move the cursor right
- `delete()`: Delete the character to the left of the cursor
- `move_left()`: Move the cursor one position left
- `move_right()`: Move the cursor one position right
- `get_text()`: Return all text in the buffer as a string

All operations must be O(1).

```python
# Test cases
editor = TextEditor()
editor.insert('H')
editor.insert('l')
editor.insert('l')
editor.insert('o')
assert editor.get_text() == "Hllo"
editor.move_left()
editor.move_left()
editor.move_left()
editor.insert('e')
assert editor.get_text() == "Hello"
editor.move_right()
editor.move_right()
editor.move_right()
editor.insert('!')
assert editor.get_text() == "Hello!"
```

**Hint:** Manage the cursor position as "immediately after a certain node." Using a sentinel node as the cursor's reference point simplifies the implementation.

---

## 10. FAQ (Frequently Asked Questions)

### Q1: Is Python's `list` internally a linked list?

**A:** No. Python's `list` is implemented as a **dynamic array**. Internally, it is a C array (an array of pointers) stored contiguously in memory. As a result, index access is O(1) and appending to the tail is O(1) amortized.

When linked list-like operations are needed in Python, `collections.deque` serves as an alternative. `deque` is based on a doubly linked list implementation (more precisely, a block-linked list) and provides O(1) additions and removals at both ends.

```python
from collections import deque

d = deque([1, 2, 3])
d.appendleft(0)  # Add to front: O(1)
d.append(4)      # Add to back: O(1)
d.popleft()      # Remove from front: O(1)
d.pop()          # Remove from back: O(1)

# However, middle index access is O(n)
# d[i] is also O(n) for deque
```

### Q2: Are there tips for solving linked list problems in interviews?

**A:** Knowing the following techniques will help you handle a wide variety of problems.

**1. Dummy Head (Sentinel Node)**
Avoids edge cases when operating on the head node. Effective for merging, deletion, insertion, and many other problems.

**2. Fast/Slow Pointer**
- Finding the middle node
- Cycle detection
- Palindrome check
- N-th node from the end

**3. Recursive Thinking**
Linked lists have a recursive structure (head + rest of the list), making them naturally amenable to recursive processing. However, beware of stack overflow (O(n) space usage).

**4. Draw Diagrams**
Pointer manipulations are error-prone when tracked only mentally. Always draw diagrams on paper to verify pointer reassignments.

**5. Common Edge Cases**
- Empty list (head = None)
- Single-node list
- Two-node list
- When the head node is the target
- When the tail node is the target

### Q3: How do linked lists affect garbage collection?

**A:** Linked lists have several impacts on garbage collection (GC).

**1. Cost of Reference Tracing**
GC follows pointers to mark reachable objects (Mark & Sweep). Since linked list nodes are scattered in memory, tracing them causes frequent cache misses, leading to longer GC pause times.

**2. Circular Reference Problem**
Nodes in a doubly linked list mutually reference each other via `prev` and `next`. Python's reference counting alone cannot collect these, but the generational GC detects and collects circular references. However, if objects with `__del__` methods are part of the circular reference chain, collection may be delayed or fail.

**3. Memory Fragmentation**
Repeatedly allocating and deallocating many small node objects causes memory fragmentation. Arrays use large contiguous regions, making them less susceptible to fragmentation.

**Countermeasures:**
- Use `__slots__` to reduce the memory footprint of nodes
- Explicitly clear lists that are no longer needed (set references to None)
- Consider array-based data structures for large-scale data

### Q4: What role do linked lists play in functional programming?

**A:** In functional programming, **immutable singly linked lists** are used as the most fundamental data structure. Haskell's list `[1, 2, 3]` is internally `1 : (2 : (3 : []))`, a linked list.

The advantage of immutable lists is "structural sharing." Even when you prepend an element to an existing list, the original list is unchanged, and the new list shares the tail of the original.

```
Structural sharing example:

xs = [2, 3, 4]
ys = 1 : xs     <- Prepend 1 to xs

In memory:
ys: [1] --+
          v
xs:      [2] -> [3] -> [4] -> []

xs and ys share [2, 3, 4]
-> Zero copy cost
-> Safe for concurrent processing (no data races)
```

### Q5: Where is the boundary between arrays and linked lists?

**A:** The decision must consider not only theoretical complexity but also the implementation context.

**Boundary conditions favoring arrays:**
- For small element counts (up to a few hundred), arrays are overwhelmingly faster due to cache efficiency
- When insertions/deletions are less than 10-20% of all operations, arrays are advantageous
- In memory-constrained embedded systems, arrays are preferred

**Boundary conditions favoring linked lists:**
- When insertions/deletions at the head dominate operations
- When node transfers between lists are frequent (O(1) splice operations)
- In real-time systems requiring worst-case time guarantees
- When combined with hash maps to achieve O(1) deletion (LRU caches, etc.)

On modern hardware, the impact of CPU caches is enormous. Even when both are theoretically O(n), array traversal can be 5x to 20x faster than linked list traversal. Considering this constant factor difference, it can be said that "linked lists are advantageous only when their theoretical complexity is clearly better than that of arrays."

### Q6: Why are linked list problems so common in interviews?

**A:** There are several reasons why linked list problems frequently appear in interviews.

1. **They test understanding of pointer operations**: Pointer reassignment is error-prone, and the ability to perform it accurately is an indicator of fundamental programming skill.

2. **They test edge case handling ability**: Empty lists, single nodes, head/tail operations, and other cases require careful attention to branching.

3. **They test algorithmic knowledge**: Many algorithmic concepts -- Fast/Slow pointers, recursion, divide and conquer -- can be tested through linked list problems.

4. **Code is short**: Solutions can be written in a length suitable for whiteboard or time-constrained interview settings.

5. **They test awareness of space complexity**: By imposing constraints like "solve with O(1) extra space," interviewers can evaluate awareness of space efficiency.

---

## 11. Linked Lists in Practice

### 11.1 Real-World Usage

While linked lists are often seen as primarily theoretical learning material, they play important roles in actual software systems.

```
Real-world applications of linked lists:

1. OS Kernels:
   - Linux kernel's list.h: Macros for doubly circular linked lists
   - Process lists, device driver lists
   - Free list management for memory

2. Cache Systems:
   - LRU caches (Redis, Memcached, browser caches)
   - LFU (Least Frequently Used) caches

3. Text Editors:
   - Emacs: Manages text with gap buffers (array-based) + linked lists
   - VS Code: Manages line lists with piece tables

4. Hash Tables:
   - Chaining method (collision resolution) uses lists for each bucket
   - Java's HashMap converts from list to red-black tree when element count is high

5. Memory Allocators:
   - Free lists: Manage free memory blocks as linked lists
   - Buddy allocators: Free lists for each size class

6. Blockchain:
   - Each block holds the hash of the previous block
   - Conceptually a singly linked list

7. Functional Languages:
   - Haskell, Clojure, Erlang: Immutable linked lists as fundamental language structures
   - Efficient data operations through structural sharing

8. Networking:
   - Linux packet queues (sk_buff): Doubly linked lists
   - TCP send/receive buffer management
```

### 11.2 Practical Choices in Python

When linked list-like functionality is needed in Python, `collections.deque` from the standard library should be considered first. Implementing a linked list from scratch should be limited to the following cases:

```
Decision guidelines for Python:

Use list (dynamic array) when:
  - Random access is needed
  - Tail appends are the primary operation
  - Slice operations are needed
  - Most use cases -> try this first

Use collections.deque when:
  - Additions/removals at both ends are needed
  - Using as a FIFO queue
  - Fixed-length sliding window (maxlen parameter)

Implement a custom linked list when:
  - Combining with a hash map (LRU cache, etc.)
  - Node-level operations are needed (node movement, splicing)
  - For educational purposes or algorithm problem solutions
  - As the foundation for specialized data structures
```

---


## FAQ

### Q1: What is the most important point for learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory but by actually writing code and verifying its behavior.

### Q2: What mistakes do beginners commonly make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this used in practice?

Knowledge of this topic is frequently applied in everyday development work. It becomes especially important during code reviews and architecture design.

---

## 12. Summary

### 12.1 Key Concept Review

| Concept | Key Points |
|---------|-----------|
| Singly linked list | next pointer only. Head insertion O(1). No reverse traversal. Simplest form |
| Doubly linked list | prev + next. Both-end operations O(1). Arbitrary node deletion O(1). Foundation for LRU caches |
| Circular linked list | Tail -> Head is circular. Round-robin, Josephus problem. Watch out for infinite loops |
| vs Arrays | Arrays dominate in cache efficiency. Linked lists excel at insertion/deletion and pointer operations |
| Dummy head | Sentinel nodes eliminate edge cases. Contribute to code simplicity and correctness |
| Fast/Slow | Middle node detection, cycle detection, palindrome check. The quintessential two-pointer technique |
| LRU Cache | Doubly linked list + hash map. All operations O(1). Interview favorite |
| Reversal | Iterative: O(1) space. Recursive: O(n) space. The most frequently asked interview problem |

### 12.2 Checklist

After completing this chapter, verify the following items:

- [ ] Can explain the structure and differences of singly, doubly, and circular linked lists
- [ ] Can accurately state the complexity of each operation (insertion, deletion, search, reversal)
- [ ] Can implement insertion and deletion using a dummy head
- [ ] Can detect middle nodes and cycles using Fast/Slow pointers
- [ ] Can implement an LRU cache using a doubly linked list + hash map
- [ ] Can make appropriate decisions about when to use arrays versus linked lists
- [ ] Can recognize and avoid linked list anti-patterns

---

## Recommended Next Guides


---

## References

1. Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). "Introduction to Algorithms." 4th Edition, MIT Press. Chapter 10: Elementary Data Structures -- Theoretical foundations and operations of linked lists.
2. Sedgewick, R. & Wayne, K. (2011). "Algorithms." 4th Edition, Addison-Wesley. Chapter 1.3: Bags, Queues, and Stacks -- Stack and queue implementations using linked lists.
3. Skiena, S. S. (2020). "The Algorithm Design Manual." 3rd Edition, Springer. Chapter 3: Data Structures -- Practical usage and design decisions for linked lists.
4. Knuth, D. E. (1997). "The Art of Computer Programming, Volume 1: Fundamental Algorithms." 3rd Edition, Addison-Wesley. Section 2.2: Linear Lists -- Classic exposition and mathematical analysis of linked lists.
5. Python Documentation. "collections.deque." https://docs.python.org/3/library/collections.html#collections.deque -- Linked list-based data structures in the Python standard library.
6. Pugh, W. (1990). "Skip Lists: A Probabilistic Alternative to Balanced Trees." Communications of the ACM, 33(6), 668-676. -- The original skip list paper. Important as an extension of linked lists.
