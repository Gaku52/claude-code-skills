# Tree Structures — Binary Trees, BST, AVL/Red-Black Trees, B-Trees, and Trie

> Learn the various tree structure variants that represent hierarchical data. This guide provides a systematic explanation ranging from the basics of binary search trees to balanced trees, B-trees, and Tries.

---

## What You Will Learn in This Chapter

1. **Binary trees and traversals** — Preorder, inorder, postorder, and level-order
2. **Binary Search Trees (BST)** and balanced trees (AVL, red-black trees)
3. **B-trees and Trie** — Disk access optimization and string search
4. **Segment trees and Fenwick trees** — Fast range query processing
5. **Practical applications** — DB indexes, autocomplete, syntax parsing

## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related fundamental concepts
- Familiarity with the content of [Hash Tables — Hash Functions, Collision Resolution, and Load Factor](./03-hash-tables.md)

---

## 1. Basic Tree Terminology

```
        [A]         ← Root, depth 0
       /   \
     [B]   [C]      ← Depth 1
    / \      \
  [D] [E]   [F]     ← Depth 2 (leaves: D, E, F)

  Terminology:
  - Root: The topmost node (A)
  - Leaf: A node with no children (D, E, F)
  - Internal node: A node with children (A, B, C)
  - Depth: Distance from the root
  - Height: Maximum distance to a leaf (tree height = 2)
  - Subtree: A tree rooted at any node
  - Degree: Number of children of a node
  - Level: Synonymous with depth (root is level 0)
  - Ancestor: Nodes on the path to the root
  - Descendant: All nodes within a subtree
  - Sibling: Nodes sharing the same parent (D and E are siblings)
```

### 1.1 Types of Trees

```
Complete Binary Tree:              Full/Perfect Binary Tree:
       [A]                           [A]
      /   \                         /   \
    [B]   [C]                     [B]   [C]
   / \   /                       / \   / \
  [D] [E][F]                   [D][E][F][G]
  Last level is left-filled     All levels are completely filled

Binary Tree:                    N-ary Tree:
       [A]                           [A]
      /   \                        / | \
    [B]   [C]                    [B][C][D]
   /        \                    |    / \
  [D]       [F]                 [E] [F][G]
  Each node has at most 2       Each node has at most N
  children                      children
```

### 1.2 Properties of Trees

```python
# Important properties of binary trees:
# - Number of edges in a binary tree with n nodes = n - 1
# - Node count for a binary tree of height h: min h+1, max 2^(h+1) - 1
# - Height of a complete binary tree with n nodes: floor(log2(n))
# - Node count of a full binary tree of height h: 2^(h+1) - 1
# - Number of leaves = number of internal nodes with 2 children + 1 (full binary tree)

def tree_properties(n):
    """Calculate properties of a complete binary tree with n nodes"""
    import math
    height = math.floor(math.log2(n)) if n > 0 else 0
    leaves = (n + 1) // 2  # For a complete binary tree
    internal = n - leaves
    edges = n - 1
    print(f"Node count: {n}")
    print(f"Height: {height}")
    print(f"Leaf count: {leaves}")
    print(f"Internal node count: {internal}")
    print(f"Edge count: {edges}")
    return height, leaves, internal, edges

tree_properties(15)
# Node count: 15, Height: 3, Leaf count: 8, Internal node count: 7, Edge count: 14
```

---

## 2. Binary Tree Traversals

### 2.1 Recursive Traversals

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def preorder(node):
    """Preorder traversal (root -> left -> right) — O(n)
    Use cases: Tree serialization, prefix notation of expressions
    """
    if not node:
        return []
    return [node.val] + preorder(node.left) + preorder(node.right)

def inorder(node):
    """Inorder traversal (left -> root -> right) — O(n)
    Use cases: Retrieving elements in sorted order from a BST
    """
    if not node:
        return []
    return inorder(node.left) + [node.val] + inorder(node.right)

def postorder(node):
    """Postorder traversal (left -> right -> root) — O(n)
    Use cases: Subtree deletion, postfix notation of expressions
    """
    if not node:
        return []
    return postorder(node.left) + postorder(node.right) + [node.val]

def levelorder(root):
    """Level-order traversal (BFS) — O(n)
    Use cases: Shortest path, level-by-level processing
    """
    from collections import deque
    if not root:
        return []
    result, queue = [], deque([root])
    while queue:
        node = queue.popleft()
        result.append(node.val)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return result
```

### 2.2 Iterative Traversals (Using a Stack)

```python
def preorder_iterative(root):
    """Iterative preorder traversal — O(n) time, O(h) space"""
    if not root:
        return []
    result, stack = [], [root]
    while stack:
        node = stack.pop()
        result.append(node.val)
        # Push right first so that left is processed first
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    return result

def inorder_iterative(root):
    """Iterative inorder traversal — O(n) time, O(h) space"""
    result, stack = [], []
    current = root
    while current or stack:
        # Go to the leftmost node
        while current:
            stack.append(current)
            current = current.left
        current = stack.pop()
        result.append(current.val)
        current = current.right
    return result

def postorder_iterative(root):
    """Iterative postorder traversal — O(n) time, O(h) space
    The reverse of preorder (root -> left -> right) is (right -> left -> root) = reverse of postorder
    """
    if not root:
        return []
    result, stack = [], [root]
    while stack:
        node = stack.pop()
        result.append(node.val)
        if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)
    return result[::-1]  # Reverse the result
```

### 2.3 Morris Traversal (O(1) Space)

```python
def morris_inorder(root):
    """Morris inorder traversal — O(n) time, O(1) space
    Uses the concept of threaded binary trees.
    Requires neither a stack nor recursion.
    """
    result = []
    current = root
    while current:
        if not current.left:
            result.append(current.val)
            current = current.right
        else:
            # Find the rightmost node of the left subtree
            predecessor = current.left
            while predecessor.right and predecessor.right != current:
                predecessor = predecessor.right

            if not predecessor.right:
                # Create thread
                predecessor.right = current
                current = current.left
            else:
                # Remove thread
                predecessor.right = None
                result.append(current.val)
                current = current.right
    return result
```

### 2.4 Traversal Illustration and Use Cases

```
Traversal order example:
       [4]
      /   \
    [2]   [6]
   / \   / \
  [1] [3] [5] [7]

  Preorder (Pre):   4, 2, 1, 3, 6, 5, 7   ← Tree copying, serialization
  Inorder (In):     1, 2, 3, 4, 5, 6, 7   ← Sorted order in BST
  Postorder (Post): 1, 3, 2, 5, 7, 6, 4   ← Tree deletion, expression evaluation
  Level-order:      4, 2, 6, 1, 3, 5, 7   ← Shortest distance, breadth-first

Level-by-level traversal (frequently used in practice):
  Level 0: [4]
  Level 1: [2, 6]
  Level 2: [1, 3, 5, 7]
```

```python
def levelorder_by_level(root):
    """Return elements grouped by level"""
    from collections import deque
    if not root:
        return []
    result = []
    queue = deque([root])
    while queue:
        level_size = len(queue)
        level = []
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
    return result

# [[4], [2, 6], [1, 3, 5, 7]]
```

---

## 3. Binary Search Tree (BST)

### 3.1 Complete Implementation

```python
class BST:
    """Binary Search Tree: A binary tree with the property left < root < right

    Average complexity: O(log n)
    Worst-case complexity: O(n) — when inserting sorted data
    """
    def __init__(self):
        self.root = None

    def insert(self, val):
        """O(h), h = height of the tree"""
        self.root = self._insert(self.root, val)

    def _insert(self, node, val):
        if not node:
            return TreeNode(val)
        if val < node.val:
            node.left = self._insert(node.left, val)
        elif val > node.val:
            node.right = self._insert(node.right, val)
        return node

    def search(self, val):
        """O(h)"""
        return self._search(self.root, val)

    def _search(self, node, val):
        if not node or node.val == val:
            return node
        if val < node.val:
            return self._search(node.left, val)
        return self._search(node.right, val)

    def delete(self, val):
        """O(h) — handles 3 cases"""
        self.root = self._delete(self.root, val)

    def _delete(self, node, val):
        if not node:
            return None
        if val < node.val:
            node.left = self._delete(node.left, val)
        elif val > node.val:
            node.right = self._delete(node.right, val)
        else:
            # Case 1: Leaf node
            if not node.left and not node.right:
                return None
            # Case 2: One child
            if not node.left:
                return node.right
            if not node.right:
                return node.left
            # Case 3: Two children — replace with minimum value of right subtree
            successor = self._min_node(node.right)
            node.val = successor.val
            node.right = self._delete(node.right, successor.val)
        return node

    def _min_node(self, node):
        """Return the minimum node in a subtree"""
        while node.left:
            node = node.left
        return node

    def find_min(self):
        """Minimum value — O(h)"""
        if not self.root:
            return None
        return self._min_node(self.root).val

    def find_max(self):
        """Maximum value — O(h)"""
        if not self.root:
            return None
        node = self.root
        while node.right:
            node = node.right
        return node.val

    def inorder_successor(self, val):
        """Inorder successor — O(h)"""
        successor = None
        node = self.root
        while node:
            if val < node.val:
                successor = node
                node = node.left
            elif val > node.val:
                node = node.right
            else:
                if node.right:
                    return self._min_node(node.right).val
                return successor.val if successor else None
        return None

    def kth_smallest(self, k):
        """k-th smallest element — O(h + k)"""
        stack = []
        current = self.root
        count = 0
        while current or stack:
            while current:
                stack.append(current)
                current = current.left
            current = stack.pop()
            count += 1
            if count == k:
                return current.val
            current = current.right
        return None

    def range_query(self, low, high):
        """All elements in [low, high] range — O(h + k), k = number of results"""
        result = []
        self._range_query(self.root, low, high, result)
        return result

    def _range_query(self, node, low, high, result):
        if not node:
            return
        if low < node.val:
            self._range_query(node.left, low, high, result)
        if low <= node.val <= high:
            result.append(node.val)
        if node.val < high:
            self._range_query(node.right, low, high, result)

    def is_valid_bst(self):
        """Validate the BST property"""
        return self._validate(self.root, float('-inf'), float('inf'))

    def _validate(self, node, min_val, max_val):
        if not node:
            return True
        if node.val <= min_val or node.val >= max_val:
            return False
        return (self._validate(node.left, min_val, node.val) and
                self._validate(node.right, node.val, max_val))

    def height(self):
        """Height of the tree — O(n)"""
        return self._height(self.root)

    def _height(self, node):
        if not node:
            return -1
        return 1 + max(self._height(node.left), self._height(node.right))

    def size(self):
        """Node count — O(n)"""
        return self._size(self.root)

    def _size(self, node):
        if not node:
            return 0
        return 1 + self._size(node.left) + self._size(node.right)

# Usage example
bst = BST()
for val in [5, 3, 7, 1, 4, 6, 8]:
    bst.insert(val)

print(bst.search(4))            # TreeNode(val=4)
print(bst.find_min())           # 1
print(bst.find_max())           # 8
print(bst.kth_smallest(3))      # 4
print(bst.range_query(3, 7))    # [3, 4, 5, 6, 7]
print(bst.inorder_successor(5)) # 6
print(bst.height())             # 2
print(bst.is_valid_bst())       # True
```

### 3.2 BST Deletion Illustrated

```
Deletion cases:

Case 1: Deleting a leaf node (delete value 1)
     5                5
    / \     ->       / \
   3   7            3   7
  / \              / \
 1   4            _   4

Case 2: Deleting a node with one child (delete value 3, left child only)
     5                5
    / \     ->       / \
   3   7            1   7
  /
 1

Case 3: Deleting a node with two children (delete value 5)
     5                6         <- Replace with the minimum of the right subtree (6)
    / \     ->       / \
   3   7            3   7
  / \  /           / \
 1  4 6           1   4
```

---

## 4. AVL Tree (Self-Balancing BST)

### 4.1 Rotation Operations Illustrated

```
AVL Tree: |left height - right height| <= 1 for all nodes

4 imbalance patterns and rotations:

=== LL Case (Right Rotation) ===
      z(3)                 y(2)
     /    \               /    \
    y(2)   T4    ->      x(1)  z(3)
   /    \               / \   / \
  x(1)  T3            T1  T2 T3 T4
 / \
T1  T2

=== RR Case (Left Rotation) ===
  z(1)                   y(2)
 /    \                 /    \
T1   y(2)     ->      z(1)  x(3)
    /    \            / \   / \
   T2   x(3)        T1 T2 T3 T4
       / \
      T3  T4

=== LR Case (Left Rotation -> Right Rotation) ===
      z(3)            z(3)               x(2)
     /    \          /    \             /    \
    y(1)   T4  ->   x(2)  T4    ->    y(1)  z(3)
   /    \          /    \            / \   / \
  T1   x(2)      y(1)  T3         T1 T2 T3 T4
      / \        / \
     T2  T3     T1  T2

=== RL Case (Right Rotation -> Left Rotation) ===
  z(1)             z(1)                  x(2)
 /    \           /    \                /    \
T1   y(3)   ->   T1   x(2)    ->      z(1)  y(3)
    /    \           /    \          / \   / \
   x(2)  T4        T2   y(3)       T1 T2 T3 T4
  / \                   / \
 T2  T3                T3  T4
```

### 4.2 Complete AVL Tree Implementation

```python
class AVLNode:
    def __init__(self, val):
        self.val = val
        self.left = self.right = None
        self.height = 1

class AVLTree:
    """AVL Tree: Balance factor of every node is within [-1, 0, 1]

    Guarantees O(log n) for all operations
    """
    def __init__(self):
        self.root = None

    def height(self, node):
        return node.height if node else 0

    def balance_factor(self, node):
        return self.height(node.left) - self.height(node.right) if node else 0

    def update_height(self, node):
        node.height = 1 + max(self.height(node.left), self.height(node.right))

    def rotate_right(self, z):
        """Right rotation: fixes LL case"""
        y = z.left
        T3 = y.right
        y.right = z
        z.left = T3
        self.update_height(z)
        self.update_height(y)
        return y

    def rotate_left(self, z):
        """Left rotation: fixes RR case"""
        y = z.right
        T2 = y.left
        y.left = z
        z.right = T2
        self.update_height(z)
        self.update_height(y)
        return y

    def _balance(self, node):
        """Restore balance of a node"""
        self.update_height(node)
        bf = self.balance_factor(node)

        # LL case
        if bf > 1 and self.balance_factor(node.left) >= 0:
            return self.rotate_right(node)

        # LR case
        if bf > 1 and self.balance_factor(node.left) < 0:
            node.left = self.rotate_left(node.left)
            return self.rotate_right(node)

        # RR case
        if bf < -1 and self.balance_factor(node.right) <= 0:
            return self.rotate_left(node)

        # RL case
        if bf < -1 and self.balance_factor(node.right) > 0:
            node.right = self.rotate_right(node.right)
            return self.rotate_left(node)

        return node

    def insert(self, val):
        """O(log n) — restores balance after insertion"""
        self.root = self._insert(self.root, val)

    def _insert(self, node, val):
        if not node:
            return AVLNode(val)
        if val < node.val:
            node.left = self._insert(node.left, val)
        elif val > node.val:
            node.right = self._insert(node.right, val)
        else:
            return node  # Ignore duplicates
        return self._balance(node)

    def delete(self, val):
        """O(log n) — restores balance after deletion"""
        self.root = self._delete(self.root, val)

    def _delete(self, node, val):
        if not node:
            return None
        if val < node.val:
            node.left = self._delete(node.left, val)
        elif val > node.val:
            node.right = self._delete(node.right, val)
        else:
            if not node.left:
                return node.right
            if not node.right:
                return node.left
            # Replace with minimum of right subtree
            successor = node.right
            while successor.left:
                successor = successor.left
            node.val = successor.val
            node.right = self._delete(node.right, successor.val)
        return self._balance(node)

    def search(self, val):
        """O(log n)"""
        node = self.root
        while node:
            if val == node.val:
                return node
            elif val < node.val:
                node = node.left
            else:
                node = node.right
        return None

    def inorder(self):
        """Inorder traversal — sorted order"""
        result = []
        self._inorder(self.root, result)
        return result

    def _inorder(self, node, result):
        if node:
            self._inorder(node.left, result)
            result.append(node.val)
            self._inorder(node.right, result)

# Usage example
avl = AVLTree()
for val in [10, 20, 30, 40, 50, 25]:
    avl.insert(val)
print(avl.inorder())  # [10, 20, 25, 30, 40, 50]
print(f"Root: {avl.root.val}")  # 30 (balance is maintained)
print(f"Height: {avl.height(avl.root)}")  # 2 or 3
```

---

## 5. Red-Black Tree

### 5.1 Properties of Red-Black Trees

```
Five properties of red-black trees:
1. Each node is either red or black
2. The root is black
3. All leaves (NIL) are black
4. Children of a red node are black (no consecutive reds)
5. For any node, all paths to its descendant leaves have the same number of black nodes (black-height)

Example:
         [7:Black]
        /      \
    [3:Red]    [18:Red]
    /   \     /    \
  [1:Black] [5:Black] [10:Black] [22:Black]
                  \
                 [15:Red]

Black-height = 2 (2 black nodes on every path)

Important theorem:
- Height h of a red-black tree with n nodes <= 2 log2(n+1)
- Therefore all operations are guaranteed O(log n)
```

### 5.2 Red-Black Tree Implementation

```python
class RBNode:
    RED = True
    BLACK = False

    def __init__(self, val, color=RED):
        self.val = val
        self.color = color
        self.left = self.right = self.parent = None

class RedBlackTree:
    """Red-Black Tree: Fewer rotations than AVL tree, more efficient insertions/deletions

    Adopted by Java's TreeMap, C++ std::map
    Also used in the Linux kernel scheduler
    """
    def __init__(self):
        self.NIL = RBNode(None, RBNode.BLACK)
        self.root = self.NIL

    def _rotate_left(self, x):
        y = x.right
        x.right = y.left
        if y.left != self.NIL:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

    def _rotate_right(self, x):
        y = x.left
        x.left = y.right
        if y.right != self.NIL:
            y.right.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y

    def insert(self, val):
        """O(log n) — fixes colors after insertion"""
        node = RBNode(val)
        node.left = self.NIL
        node.right = self.NIL

        # Standard BST insertion
        parent = None
        current = self.root
        while current != self.NIL:
            parent = current
            if val < current.val:
                current = current.left
            elif val > current.val:
                current = current.right
            else:
                return  # Ignore duplicates

        node.parent = parent
        if parent is None:
            self.root = node
        elif val < parent.val:
            parent.left = node
        else:
            parent.right = node

        # Case 1: Node is the root
        if node.parent is None:
            node.color = RBNode.BLACK
            return

        # Case 2: Parent is black — no violation
        if node.parent.color == RBNode.BLACK:
            return

        # Fix colors
        self._fix_insert(node)

    def _fix_insert(self, node):
        """Fix red-black tree properties after insertion"""
        while node.parent and node.parent.color == RBNode.RED:
            if node.parent == node.parent.parent.left:
                uncle = node.parent.parent.right
                if uncle.color == RBNode.RED:
                    # Case 3: Uncle is red -> color flip
                    node.parent.color = RBNode.BLACK
                    uncle.color = RBNode.BLACK
                    node.parent.parent.color = RBNode.RED
                    node = node.parent.parent
                else:
                    if node == node.parent.right:
                        # Case 4: LR -> left rotate to convert to LL
                        node = node.parent
                        self._rotate_left(node)
                    # Case 5: LL -> right rotate
                    node.parent.color = RBNode.BLACK
                    node.parent.parent.color = RBNode.RED
                    self._rotate_right(node.parent.parent)
            else:
                uncle = node.parent.parent.left
                if uncle.color == RBNode.RED:
                    node.parent.color = RBNode.BLACK
                    uncle.color = RBNode.BLACK
                    node.parent.parent.color = RBNode.RED
                    node = node.parent.parent
                else:
                    if node == node.parent.left:
                        node = node.parent
                        self._rotate_right(node)
                    node.parent.color = RBNode.BLACK
                    node.parent.parent.color = RBNode.RED
                    self._rotate_left(node.parent.parent)

            if node == self.root:
                break

        self.root.color = RBNode.BLACK

    def search(self, val):
        """O(log n)"""
        node = self.root
        while node != self.NIL:
            if val == node.val:
                return node
            elif val < node.val:
                node = node.left
            else:
                node = node.right
        return None

    def inorder(self):
        result = []
        self._inorder(self.root, result)
        return result

    def _inorder(self, node, result):
        if node != self.NIL:
            self._inorder(node.left, result)
            result.append(node.val)
            self._inorder(node.right, result)

# Usage example
rbt = RedBlackTree()
for val in [7, 3, 18, 10, 22, 8, 11, 26, 2, 6]:
    rbt.insert(val)
print(rbt.inorder())  # [2, 3, 6, 7, 8, 10, 11, 18, 22, 26]
```

---

## 6. B-Tree

### 6.1 B-Tree Structure and Properties

```
B-Tree (order m = 4, 2-3-4 tree):

              [17]
            /      \
      [5, 13]      [21, 30]
     /   |   \     /   |   \
  [1,3] [7,11] [14,16] [18,20] [22,25] [31,40]

B-Tree properties (order m):
- Each node holds at most m-1 keys
- Each node has at most m children
- Each node (except the root) holds at least ceil(m/2) - 1 keys
- All leaves are at the same level
- Keys within a node are sorted

Why B-Trees are optimal for databases:
1. Node size can be aligned to disk block size (4KB-16KB)
2. Tree height is very low (3-4 levels can store millions of records)
3. Number of disk I/O operations = tree height = O(log_m n)
4. Example: m=1000, n=10^9 -> height ~ 3 disk accesses
```

### 6.2 B-Tree Implementation

```python
class BTreeNode:
    def __init__(self, leaf=True):
        self.keys = []
        self.children = []
        self.leaf = leaf

class BTree:
    """B-Tree — a disk-based data structure

    Widely used as database indexes
    """
    def __init__(self, order=4):
        self.root = BTreeNode()
        self.order = order  # Maximum number of children
        self.min_keys = order // 2 - 1  # Minimum number of keys per node

    def search(self, key, node=None):
        """O(log_m n) — m = order"""
        if node is None:
            node = self.root

        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1

        if i < len(node.keys) and key == node.keys[i]:
            return (node, i)

        if node.leaf:
            return None

        return self.search(key, node.children[i])

    def insert(self, key):
        """O(log_m n)"""
        root = self.root
        if len(root.keys) == self.order - 1:
            # Root is full -> split and create a new root
            new_root = BTreeNode(leaf=False)
            new_root.children.append(self.root)
            self._split_child(new_root, 0)
            self.root = new_root

        self._insert_non_full(self.root, key)

    def _insert_non_full(self, node, key):
        i = len(node.keys) - 1

        if node.leaf:
            # Insert into a leaf node
            node.keys.append(None)
            while i >= 0 and key < node.keys[i]:
                node.keys[i + 1] = node.keys[i]
                i -= 1
            node.keys[i + 1] = key
        else:
            # Internal node: descend to the appropriate child
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1

            if len(node.children[i].keys) == self.order - 1:
                # Split the child if it is full
                self._split_child(node, i)
                if key > node.keys[i]:
                    i += 1

            self._insert_non_full(node.children[i], key)

    def _split_child(self, parent, i):
        """Split a child node"""
        order = self.order
        child = parent.children[i]
        mid = order // 2 - 1

        # Move the right half to a new node
        new_node = BTreeNode(leaf=child.leaf)
        new_node.keys = child.keys[mid + 1:]
        if not child.leaf:
            new_node.children = child.children[mid + 1:]

        # Promote the middle key to the parent
        parent.keys.insert(i, child.keys[mid])
        parent.children.insert(i + 1, new_node)

        # Keep the left half
        child.keys = child.keys[:mid]
        if not child.leaf:
            child.children = child.children[:mid + 1]

    def traverse(self, node=None):
        """Return all keys in sorted order"""
        if node is None:
            node = self.root
        result = []
        for i in range(len(node.keys)):
            if not node.leaf:
                result.extend(self.traverse(node.children[i]))
            result.append(node.keys[i])
        if not node.leaf:
            result.extend(self.traverse(node.children[-1]))
        return result

# Usage example
btree = BTree(order=4)
for key in [10, 20, 5, 6, 12, 30, 7, 17]:
    btree.insert(key)
print(btree.traverse())  # [5, 6, 7, 10, 12, 17, 20, 30]
```

### 6.3 B+ Tree

```python
class BPlusLeafNode:
    """B+ Tree leaf node: all data is stored in leaves"""
    def __init__(self):
        self.keys = []
        self.values = []  # Actual data or pointers to data
        self.next_leaf = None  # Pointer to the next leaf on the right

class BPlusInternalNode:
    """B+ Tree internal node: keys only (index)"""
    def __init__(self):
        self.keys = []
        self.children = []

# Characteristics of B+ Trees:
# 1. All data resides in leaf nodes -> fast range queries
# 2. Leaf nodes are connected as a linked list -> O(n) sequential access
# 3. Internal nodes hold only keys -> can store more keys
# 4. MySQL's InnoDB and PostgreSQL indexes use B+ Trees
#
# B-Tree vs B+ Tree:
# | Property         | B-Tree           | B+ Tree              |
# |-----------------|-----------------|----------------------|
# | Data location    | All nodes        | Leaf nodes only       |
# | Range search     | Requires inorder | Fast via leaf linked  |
# |                  | traversal        | list                  |
# | Duplicate keys   | In internal too  | Leaves only           |
# | Fan-out          | Slightly lower   | Higher (internal has  |
# |                  |                  | keys only)            |
# | Use case         | General purpose  | Standard for DB       |
# |                  |                  | indexes               |
```

---

## 7. Trie (Prefix Tree)

### 7.1 Basic Structure and Implementation

```
Storing "app", "apple", "apt", "bat":

      (root)
      /    \
    [a]    [b]
     |      |
    [p]    [a]
    / \     |
  [p] [t*] [t*]
   |
  [l]
   |
  [e*]

  * = end of word
```

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.count = 0  # Number of words with this prefix

class Trie:
    """Trie (Prefix Tree) — fast string search

    Use cases: Autocomplete, spell checking, IP routing
    """
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        """O(m), m = length of word"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.count += 1
        node.is_end = True

    def search(self, word):
        """Exact match search — O(m)"""
        node = self._find(word)
        return node is not None and node.is_end

    def starts_with(self, prefix):
        """Prefix search — O(m)"""
        return self._find(prefix) is not None

    def count_prefix(self, prefix):
        """Number of words with the given prefix — O(m)"""
        node = self._find(prefix)
        return node.count if node else 0

    def _find(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node

    def delete(self, word):
        """Delete a word — O(m)"""
        self._delete(self.root, word, 0)

    def _delete(self, node, word, depth):
        if depth == len(word):
            if not node.is_end:
                return False
            node.is_end = False
            return len(node.children) == 0

        char = word[depth]
        if char not in node.children:
            return False

        child = node.children[char]
        child.count -= 1
        should_delete = self._delete(child, word, depth + 1)

        if should_delete:
            del node.children[char]
            return not node.is_end and len(node.children) == 0

        return False

    def autocomplete(self, prefix, limit=10):
        """Autocomplete — return words that follow the prefix"""
        node = self._find(prefix)
        if not node:
            return []
        results = []
        self._collect_words(node, prefix, results, limit)
        return results

    def _collect_words(self, node, prefix, results, limit):
        if len(results) >= limit:
            return
        if node.is_end:
            results.append(prefix)
        for char in sorted(node.children.keys()):
            self._collect_words(node.children[char], prefix + char, results, limit)

    def get_all_words(self):
        """Return all words"""
        return self.autocomplete("", limit=float('inf'))

# Usage example
trie = Trie()
words = ["apple", "app", "application", "apply", "apt", "bat", "batch", "bath"]
for w in words:
    trie.insert(w)

print(trie.search("apple"))       # True
print(trie.search("app"))         # True
print(trie.search("ap"))          # False
print(trie.starts_with("ap"))     # True
print(trie.count_prefix("app"))   # 4 (apple, app, application, apply)
print(trie.autocomplete("app"))   # ['app', 'apple', 'application', 'apply']
```

### 7.2 Compressed Trie (Patricia Trie / Radix Tree)

```python
class CompressedTrieNode:
    """Compressed Trie: Collapses non-branching paths into a single node

    Standard Trie:          Compressed Trie:
        (root)              (root)
          |                 /    \
         [t]           ["test"]  ["toast"]
         / \
        [e] [o]
        |    |
       [s]  [a]
        |    |
       [t*] [s]
              |
             [t*]

    Greatly improves memory efficiency
    """
    def __init__(self, label=""):
        self.label = label
        self.children = {}
        self.is_end = False

class CompressedTrie:
    def __init__(self):
        self.root = CompressedTrieNode()

    def insert(self, word):
        node = self.root
        while word:
            # Find a child with a common prefix
            found = False
            for key, child in node.children.items():
                common = self._common_prefix(word, key)
                if common:
                    found = True
                    if common == key:
                        # Entire key is common -> proceed to child node
                        node = child
                        word = word[len(common):]
                    else:
                        # Partially common -> split the node
                        self._split(node, key, common)
                        node = node.children[common]
                        word = word[len(common):]
                    break
            if not found:
                # Add a new child node
                new_node = CompressedTrieNode(word)
                new_node.is_end = True
                node.children[word] = new_node
                return

        node.is_end = True

    def _common_prefix(self, s1, s2):
        i = 0
        while i < len(s1) and i < len(s2) and s1[i] == s2[i]:
            i += 1
        return s1[:i]

    def _split(self, parent, old_key, common):
        old_child = parent.children.pop(old_key)
        new_mid = CompressedTrieNode(common)
        remaining = old_key[len(common):]
        old_child.label = remaining
        new_mid.children[remaining] = old_child
        parent.children[common] = new_mid

    def search(self, word):
        node = self.root
        while word:
            found = False
            for key, child in node.children.items():
                if word.startswith(key):
                    node = child
                    word = word[len(key):]
                    found = True
                    break
            if not found:
                return False
        return node.is_end
```

### 7.3 Bit Trie (XOR Trie)

```python
class BitTrie:
    """Bit Trie: Builds a Trie on the bit representation of integers

    Use cases: Finding maximum XOR pairs, IP routing
    """
    def __init__(self, max_bits=32):
        self.root = {}
        self.max_bits = max_bits

    def insert(self, num):
        node = self.root
        for i in range(self.max_bits - 1, -1, -1):
            bit = (num >> i) & 1
            if bit not in node:
                node[bit] = {}
            node = node[bit]

    def find_max_xor(self, num):
        """Find the value that maximizes XOR with num — O(max_bits)"""
        node = self.root
        xor_result = 0
        for i in range(self.max_bits - 1, -1, -1):
            bit = (num >> i) & 1
            opposite = 1 - bit
            if opposite in node:
                xor_result |= (1 << i)
                node = node[opposite]
            elif bit in node:
                node = node[bit]
            else:
                break
        return xor_result

# Usage example: Maximum XOR pair
def max_xor_pair(nums):
    trie = BitTrie()
    max_xor = 0
    for num in nums:
        trie.insert(num)
        max_xor = max(max_xor, trie.find_max_xor(num))
    return max_xor

print(max_xor_pair([3, 10, 5, 25, 2, 8]))  # 28 (5 XOR 25)
```

---

## 8. Practical Application Patterns

### 8.1 Expression Tree

```python
class ExprNode:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def evaluate(node):
    """Evaluate an expression tree — postorder traversal"""
    if not node.left and not node.right:
        return float(node.val)

    left_val = evaluate(node.left)
    right_val = evaluate(node.right)

    ops = {'+': lambda a, b: a + b,
           '-': lambda a, b: a - b,
           '*': lambda a, b: a * b,
           '/': lambda a, b: a / b}

    return ops[node.val](left_val, right_val)

# Expression tree for (3 + 4) * 2
expr = ExprNode('*',
    ExprNode('+', ExprNode('3'), ExprNode('4')),
    ExprNode('2'))
print(evaluate(expr))  # 14.0
```

### 8.2 Lowest Common Ancestor (LCA)

```python
def lowest_common_ancestor(root, p, q):
    """Lowest Common Ancestor of a binary tree — O(n)

    LCA(p, q): The deepest node that has both p and q as descendants
    """
    if not root or root.val == p or root.val == q:
        return root

    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)

    if left and right:
        return root  # p and q are in different subtrees -> root is the LCA
    return left if left else right

# For BST, this can be optimized to O(log n)
def lca_bst(root, p, q):
    """Lowest Common Ancestor of a BST — O(h)"""
    while root:
        if p < root.val and q < root.val:
            root = root.left
        elif p > root.val and q > root.val:
            root = root.right
        else:
            return root
    return None
```

### 8.3 Tree Serialization/Deserialization

```python
def serialize(root):
    """Convert a tree to a string — preorder traversal"""
    if not root:
        return "null"
    return f"{root.val},{serialize(root.left)},{serialize(root.right)}"

def deserialize(data):
    """Reconstruct a tree from a string"""
    values = iter(data.split(","))

    def build():
        val = next(values)
        if val == "null":
            return None
        node = TreeNode(int(val))
        node.left = build()
        node.right = build()
        return node

    return build()

# Usage example
root = TreeNode(1, TreeNode(2), TreeNode(3, TreeNode(4), TreeNode(5)))
s = serialize(root)
print(s)  # "1,2,null,null,3,4,null,null,5,null,null"
restored = deserialize(s)
print(preorder(restored))  # [1, 2, 3, 4, 5]
```

### 8.4 Diameter of a Binary Tree

```python
def diameter_of_binary_tree(root):
    """Diameter of a binary tree (number of edges in the longest path) — O(n)"""
    max_diameter = [0]

    def height(node):
        if not node:
            return 0
        left_h = height(node.left)
        right_h = height(node.right)
        # Length of the path passing through this node
        max_diameter[0] = max(max_diameter[0], left_h + right_h)
        return 1 + max(left_h, right_h)

    height(root)
    return max_diameter[0]
```

### 8.5 Side View of a Tree

```python
def right_side_view(root):
    """Right side view of a binary tree — O(n)

    Returns the value of the rightmost node at each level
    """
    from collections import deque
    if not root:
        return []
    result = []
    queue = deque([root])
    while queue:
        level_size = len(queue)
        for i in range(level_size):
            node = queue.popleft()
            if i == level_size - 1:  # Last node at this level
                result.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    return result
```

---

## 9. Comparison Tables

### Table 1: Operation Complexity by Tree Type

| Tree Type | Search | Insert | Delete | Space |
|---------|------|------|------|------|
| BST (average) | O(log n) | O(log n) | O(log n) | O(n) |
| BST (worst) | O(n) | O(n) | O(n) | O(n) |
| AVL Tree | O(log n) | O(log n) | O(log n) | O(n) |
| Red-Black Tree | O(log n) | O(log n) | O(log n) | O(n) |
| B-Tree (order m) | O(log n) | O(log n) | O(log n) | O(n) |
| B+ Tree | O(log n) | O(log n) | O(log n) | O(n) |
| Trie | O(m) | O(m) | O(m) | O(SIGMA * L * N) |
| Splay Tree | O(log n) amortized | O(log n) amortized | O(log n) amortized | O(n) |

### Table 2: Balanced Tree Comparison

| Property | AVL Tree | Red-Black Tree | B-Tree | Splay Tree |
|------|--------|--------|------|----------|
| Balance condition | Height diff <= 1 | Color rules | Node fill rate | None (amortized) |
| Search speed | Fast | Slightly slower | Disk-optimized | High locality |
| Insert/Delete | More rotations | Fewer rotations | Split/Merge | Zig-zag |
| Primary use | In-memory search | General (TreeMap) | DB indexes | Cache |
| Height | <= 1.44 log n | <= 2 log n | <= log_t n | O(n) worst |
| Rotations (insert) | At most 2 | At most 2 | 0 | O(log n) |
| Rotations (delete) | O(log n) | At most 3 | 0 | O(log n) |

### Table 3: Tree Structures in Language Standard Libraries

| Language | Ordered Map | Internal Implementation | Unordered Map |
|------|-------------|---------|-------------|
| Java | TreeMap | Red-Black Tree | HashMap |
| C++ | std::map | Red-Black Tree | std::unordered_map |
| Python | SortedDict (sortedcontainers) | B-Tree based | dict |
| Rust | BTreeMap | B-Tree | HashMap |
| Go | None (standard) | - | map |
| C# | SortedDictionary | Red-Black Tree | Dictionary |

---

## 10. Anti-Patterns

### Anti-Pattern 1: Inserting Sorted Data into a BST

```python
# BAD: Sorted data -> skewed tree -> O(n)
bst = BST()
for i in range(1, 8):
    bst.insert(i)

# Result: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 (same as a linked list)
# Search: O(n)

# GOOD: Use a balanced tree (AVL/Red-Black Tree)
avl = AVLTree()
for i in range(1, 8):
    avl.insert(i)
# -> Always guarantees O(log n)

# GOOD: Randomly shuffle before inserting
import random
data = list(range(1, 8))
random.shuffle(data)
bst2 = BST()
for i in data:
    bst2.insert(i)
# -> On average produces a well-balanced tree

# GOOD: Build a balanced BST directly from a sorted array
def sorted_array_to_bst(arr):
    if not arr:
        return None
    mid = len(arr) // 2
    root = TreeNode(arr[mid])
    root.left = sorted_array_to_bst(arr[:mid])
    root.right = sorted_array_to_bst(arr[mid + 1:])
    return root

root = sorted_array_to_bst([1, 2, 3, 4, 5, 6, 7])
print(levelorder(root))  # [4, 2, 6, 1, 3, 5, 7]
```

### Anti-Pattern 2: Unnecessary Memory Consumption in Trie

```python
# BAD: Allocate an array for 26 characters at every node
class BadTrieNode:
    def __init__(self):
        self.children = [None] * 26  # 26 x 8bytes = 208bytes/node
        self.is_end = False

# GOOD: Use a dictionary for only the needed characters
class GoodTrieNode:
    def __init__(self):
        self.children = {}  # Only the characters that exist
        self.is_end = False

# BETTER: Use a compressed Trie to collapse non-branching paths
# "application" with 11 nodes -> compressed to 1-2 nodes
```

### Anti-Pattern 3: Not Considering Recursion Depth Limits

```python
# BAD: Recursion causes StackOverflow on deep trees
def bad_inorder(node):
    if not node:
        return []
    return bad_inorder(node.left) + [node.val] + bad_inorder(node.right)
# Python's default recursion limit is 1000

# GOOD: Use the iterative version
def good_inorder(root):
    result, stack = [], []
    current = root
    while current or stack:
        while current:
            stack.append(current)
            current = current.left
        current = stack.pop()
        result.append(current.val)
        current = current.right
    return result

# Or raise the recursion limit (not recommended)
# import sys
# sys.setrecursionlimit(100000)
```

### Anti-Pattern 4: Using Mutable Objects as BST Keys

```python
# BAD: Comparison results change after insertion
class MutableKey:
    def __init__(self, val):
        self.val = val
    def __lt__(self, other):
        return self.val < other.val

# Modifying the key value breaks the BST property

# GOOD: Use immutable keys
# Numbers, strings, tuples, etc.
```

---

## 11. FAQ

### Q1: Should I use an AVL tree or a red-black tree?

**A:** If searches are frequent, use an AVL tree (lower tree height, at most 1.44 log n). If insertions and deletions are frequent, use a red-black tree (fewer rotations: at most 2 for insertion, at most 3 for deletion). In practice, many language standard libraries adopt red-black trees (Java TreeMap, C++ std::map). However, Rust uses a B-tree as its standard (BTreeMap), which offers better cache efficiency.

### Q2: Why are B-trees used in databases?

**A:** B-tree nodes are large (hundreds to thousands of keys) and can be aligned to disk block sizes. The tree height is very low (3-4 levels can store millions of records), minimizing the number of disk I/O operations. B+ trees are further optimized for range queries, with leaf nodes connected as a linked list, making ORDER BY and BETWEEN clauses fast.

### Q3: How can I reduce the memory consumption of a Trie?

**A:** Several techniques are available:
1. **Compressed Trie (Patricia Trie / Radix Tree)**: Collapses non-branching paths into a single node
2. **Dictionary-based child management**: Using a dictionary instead of an array is efficient even for sparse alphabets
3. **Double-Array Trie**: Represents a Trie using two arrays, balancing memory efficiency and speed
4. **HAT-Trie**: A hybrid of hash tables and Tries

### Q4: What are some tips for solving binary tree problems?

**A:**
1. **Think recursively**: Most binary tree problems can be decomposed into "processing at the root + left subtree + right subtree"
2. **Clarify the return type**: Decide in advance what information the function should return (height, result, boolean, etc.)
3. **Check the base case**: Do not forget to handle null nodes
4. **Choose the right traversal**: Select the traversal that matches your goal (inorder for sorted order, preorder for copying, postorder for deletion)

### Q5: What is the difference between segment trees and Fenwick trees?

**A:**
- **Segment Tree**: Supports arbitrary range queries and range updates. Range updates are O(log n) with lazy propagation. Implementation is somewhat complex but highly versatile.
- **Fenwick Tree (BIT)**: Specialized for prefix sums. Easy to implement with good memory efficiency. However, range updates require additional techniques.
- Guideline: Use a Fenwick tree for simple cumulative sum queries, and a segment tree for complex range operations.

---


## FAQ

### Q1: What is the most important point when studying this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory but by actually writing code and verifying its behavior.

### Q2: What are common mistakes that beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend building a solid understanding of the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this knowledge applied in practice?

Knowledge of this topic is frequently utilized in day-to-day development work. It becomes especially important during code reviews and architecture design.

---

## 12. Summary

| Topic | Key Points |
|------|---------|
| Binary tree traversals | 4 types: preorder/inorder/postorder/level-order. Iterative and Morris traversal are also important |
| BST | Inorder traversal yields sorted order. Worst case O(n). Deletion handles 3 cases |
| AVL Tree | Height difference <= 1. Optimal for search. 4 rotation patterns |
| Red-Black Tree | Fewer rotations. At most 2 for insertion, 3 for deletion. General purpose |
| B-Tree | Disk I/O optimized. Used for DB indexes. High fan-out |
| B+ Tree | Data only in leaves. Leaves form a linked list. Optimal for range queries |
| Trie | O(m) for prefix search. Autocomplete. Memory-efficient with compression |
| Practical applications | Many common patterns including LCA, serialization, expression trees, diameter |

---

## Recommended Next Guides

- [Heaps — Binary Heaps and Heap Sort](./05-heaps.md)
- [Segment Trees — Range Queries and Lazy Propagation](../03-advanced/01-segment-tree.md)

---

## References

1. Cormen, T.H. et al. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. — Chapters 12-13 "Binary Search Trees" "Red-Black Trees", Chapter 18 "B-Trees"
2. Bayer, R. & McCreight, E. (1972). "Organization and maintenance of large ordered indexes." *Acta Informatica*, 1(3), 173-189.
3. Sedgewick, R. & Wayne, K. (2011). *Algorithms* (4th ed.). Addison-Wesley. — Trie implementation
4. Adelson-Velsky, G.M. & Landis, E.M. (1962). "An algorithm for the organization of information." *Soviet Mathematics Doklady*, 3, 1259-1263.
5. Guibas, L.J. & Sedgewick, R. (1978). "A dichromatic framework for balanced trees." *19th Annual Symposium on Foundations of Computer Science*.
