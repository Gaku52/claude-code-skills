# Tree Structures

> Trees are the most natural data structure for representing hierarchical relationships and serve as the foundation for file systems, the DOM, and database indexes.

## Learning Objectives

- [ ] Understand the fundamental terminology and traversal methods of trees
- [ ] Explain the operations and time complexities of Binary Search Trees (BST)
- [ ] Understand the necessity of balanced trees (AVL trees, Red-Black trees)
- [ ] Understand how B-trees/B+ trees optimize disk I/O
- [ ] Grasp the structure and use cases of Tries
- [ ] Know advanced tree structures such as segment trees and Fenwick trees

## Prerequisites


---

## 1. Tree Fundamentals

### 1.1 Terminology

```
Tree terminology:
         A          <- Root
        / \
       B   C        <- Children of A
      / \   \
     D   E   F      <- Leaves (nodes with no children)

  Node: Each element
  Edge: A connection between nodes
  Root: The topmost node (A)
  Leaf: A node with no children (D, E, F)
  Height: Number of edges from root to deepest leaf (= 2)
  Depth: Number of edges from root to a given node
  Subtree: A tree rooted at a given node
  Degree: Number of children a node has
  Level: Distance from the root (root = level 0)
  Ancestor: All nodes on the path from root to a given node
  Descendant: All nodes on paths from a given node toward the leaves
  Sibling: Nodes that share the same parent

Types of trees:
  Binary Tree: Each node has at most 2 children
  Complete Binary Tree: All levels except the last are fully filled
  Full Binary Tree: Every node has either 0 or 2 children
  Perfect Binary Tree: All leaves are at the same level
  N-ary Tree: Each node has at most N children

Node count for complete binary trees:
  Perfect binary tree of height h: 2^(h+1) - 1 nodes
  Complete binary tree of height h: 2^h to 2^(h+1) - 1 nodes
  Height of a complete binary tree with n nodes: floor(log2(n))
```

### 1.2 Tree Traversal

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# Pre-order: Root -> Left -> Right
def preorder(node):
    if not node: return []
    return [node.val] + preorder(node.left) + preorder(node.right)

# In-order: Left -> Root -> Right  <- Sorted order for BST
def inorder(node):
    if not node: return []
    return inorder(node.left) + [node.val] + inorder(node.right)

# Post-order: Left -> Right -> Root
def postorder(node):
    if not node: return []
    return postorder(node.left) + postorder(node.right) + [node.val]

# Level-order (BFS):
from collections import deque
def levelorder(root):
    if not root: return []
    result, queue = [], deque([root])
    while queue:
        node = queue.popleft()
        result.append(node.val)
        if node.left: queue.append(node.left)
        if node.right: queue.append(node.right)
    return result

#       4
#      / \
#     2   6
#    / \ / \
#   1  3 5  7
# Pre-order:   [4,2,1,3,6,5,7]
# In-order:    [1,2,3,4,5,6,7] <- Sorted!
# Post-order:  [1,3,2,5,7,6,4]
# Level-order: [4,2,6,1,3,5,7]
```

### 1.3 Iterative Traversal

Recursion carries the risk of stack overflow, so iterative traversal is necessary for deep trees.

```python
# Iterative in-order traversal (using a stack)
def inorder_iterative(root):
    """O(n) time, O(h) space (h is the tree height)"""
    result = []
    stack = []
    current = root

    while current or stack:
        # Go as far left as possible
        while current:
            stack.append(current)
            current = current.left

        # Process the leftmost node
        current = stack.pop()
        result.append(current.val)

        # Move to the right subtree
        current = current.right

    return result


# Iterative pre-order traversal
def preorder_iterative(root):
    if not root:
        return []
    result = []
    stack = [root]

    while stack:
        node = stack.pop()
        result.append(node.val)
        # Push right first so left is processed first
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)

    return result


# Iterative post-order traversal (using two stacks)
def postorder_iterative(root):
    if not root:
        return []
    result = []
    stack1 = [root]
    stack2 = []

    while stack1:
        node = stack1.pop()
        stack2.append(node)
        if node.left:
            stack1.append(node.left)
        if node.right:
            stack1.append(node.right)

    while stack2:
        result.append(stack2.pop().val)

    return result


# Morris traversal (O(1) space in-order traversal)
def morris_inorder(root):
    """O(1) space traversal using threaded binary trees"""
    result = []
    current = root

    while current:
        if current.left is None:
            result.append(current.val)
            current = current.right
        else:
            # Find the rightmost node in the left subtree
            predecessor = current.left
            while predecessor.right and predecessor.right != current:
                predecessor = predecessor.right

            if predecessor.right is None:
                # Create thread
                predecessor.right = current
                current = current.left
            else:
                # Remove thread (restore original structure)
                predecessor.right = None
                result.append(current.val)
                current = current.right

    return result
```

### 1.4 Level-Order Traversal Applications

```python
from collections import deque

# Group by level
def level_order_grouped(root):
    """Return nodes at each level as a list of lists"""
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

#       4
#      / \
#     2   6
#    / \ / \
#   1  3 5  7
# -> [[4], [2, 6], [1, 3, 5, 7]]


# Zigzag level-order traversal
def zigzag_level_order(root):
    """Traverse odd levels from right to left"""
    if not root:
        return []

    result = []
    queue = deque([root])
    left_to_right = True

    while queue:
        level_size = len(queue)
        level = deque()

        for _ in range(level_size):
            node = queue.popleft()
            if left_to_right:
                level.append(node.val)
            else:
                level.appendleft(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(list(level))
        left_to_right = not left_to_right

    return result

# -> [[4], [6, 2], [1, 3, 5, 7]]


# Right side view
def right_side_view(root):
    """Return the rightmost node at each level"""
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        for i in range(level_size):
            node = queue.popleft()
            if i == level_size - 1:
                result.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

    return result

# -> [4, 6, 7]
```

---

## 2. Binary Search Tree (BST)

### 2.1 Properties and Operations

```python
class BST:
    def __init__(self):
        self.root = None

    def insert(self, val):
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
        return self._search(self.root, val)

    def _search(self, node, val):
        if not node or node.val == val:
            return node
        if val < node.val:
            return self._search(node.left, val)
        return self._search(node.right, val)

# BST property: All nodes in left subtree < parent < All nodes in right subtree
# Complexity: Average O(log n), worst O(n) (skewed tree)
```

### 2.2 BST Deletion

BST deletion is more complex than insertion and search, requiring handling of three cases.

```python
class BST:
    # ... (in addition to insert, search above)

    def delete(self, val):
        self.root = self._delete(self.root, val)

    def _delete(self, node, val):
        if not node:
            return None

        if val < node.val:
            node.left = self._delete(node.left, val)
        elif val > node.val:
            node.right = self._delete(node.right, val)
        else:
            # Case 1: Leaf node (no children)
            if not node.left and not node.right:
                return None

            # Case 2: One child
            if not node.left:
                return node.right
            if not node.right:
                return node.left

            # Case 3: Two children
            # Replace with the minimum value in the right subtree (in-order successor)
            successor = self._find_min(node.right)
            node.val = successor.val
            node.right = self._delete(node.right, successor.val)

        return node

    def _find_min(self, node):
        """Return the node with the minimum value in the subtree"""
        current = node
        while current.left:
            current = current.left
        return current

    def _find_max(self, node):
        """Return the node with the maximum value in the subtree"""
        current = node
        while current.right:
            current = current.right
        return current

    # BST Utilities
    def kth_smallest(self, k):
        """Return the k-th smallest element (k-th in in-order traversal)"""
        self.count = 0
        self.result = None
        self._kth_smallest(self.root, k)
        return self.result

    def _kth_smallest(self, node, k):
        if not node or self.result is not None:
            return
        self._kth_smallest(node.left, k)
        self.count += 1
        if self.count == k:
            self.result = node.val
            return
        self._kth_smallest(node.right, k)

    def is_valid_bst(self):
        """Validate BST integrity"""
        return self._is_valid(self.root, float('-inf'), float('inf'))

    def _is_valid(self, node, min_val, max_val):
        if not node:
            return True
        if node.val <= min_val or node.val >= max_val:
            return False
        return (self._is_valid(node.left, min_val, node.val) and
                self._is_valid(node.right, node.val, max_val))

    def lca(self, p, q):
        """Lowest Common Ancestor"""
        node = self.root
        while node:
            if p < node.val and q < node.val:
                node = node.left
            elif p > node.val and q > node.val:
                node = node.right
            else:
                return node.val
        return None


# Usage example
bst = BST()
for val in [8, 3, 10, 1, 6, 14, 4, 7, 13]:
    bst.insert(val)

#         8
#        / \
#       3   10
#      / \    \
#     1   6   14
#        / \  /
#       4  7 13

print(inorder(bst.root))           # [1, 3, 4, 6, 7, 8, 10, 13, 14]
print(bst.kth_smallest(3))         # 4
print(bst.lca(4, 7))               # 6
print(bst.is_valid_bst())          # True

bst.delete(3)
print(inorder(bst.root))           # [1, 4, 6, 7, 8, 10, 13, 14]
```

### 2.3 The Need for Balanced Trees

```
Skewed tree vs. balanced tree:

  Inserting sorted data: 1, 2, 3, 4, 5
    1                 3
     \               / \
      2             2   4
       \           /     \
        3         1       5
         \
          4      Balanced tree -> O(log n)
           \
            5    Skewed tree -> O(n)

  Types of balanced trees:
  - AVL tree: Strictly balanced (height difference between left and right <= 1)
  - Red-Black tree: Loosely balanced (Java TreeMap, C++ std::map)
  - B-tree/B+ tree: Optimized for disk I/O (database indexes)
  - 2-3 tree: Educational (used to explain Red-Black trees)
```

---

## 3. AVL Tree

### 3.1 AVL Tree Basics

An AVL tree is the oldest self-balancing binary search tree. It guarantees that for every node, the height difference (balance factor) between the left and right subtrees is -1, 0, or +1.

```python
class AVLNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.height = 1  # Height of a new node is 1

class AVLTree:
    def __init__(self):
        self.root = None

    def height(self, node):
        return node.height if node else 0

    def balance_factor(self, node):
        """Balance factor = left height - right height"""
        return self.height(node.left) - self.height(node.right) if node else 0

    def update_height(self, node):
        """Update a node's height"""
        node.height = 1 + max(self.height(node.left), self.height(node.right))

    # Rotation operations
    def right_rotate(self, y):
        """Right rotation"""
        #     y              x
        #    / \            / \
        #   x   T3   ->   T1  y
        #  / \                / \
        # T1  T2            T2  T3

        x = y.left
        T2 = x.right

        x.right = y
        y.left = T2

        self.update_height(y)
        self.update_height(x)

        return x

    def left_rotate(self, x):
        """Left rotation"""
        #   x                y
        #  / \              / \
        # T1  y     ->     x   T3
        #    / \          / \
        #   T2  T3       T1  T2

        y = x.right
        T2 = y.left

        y.left = x
        x.right = T2

        self.update_height(x)
        self.update_height(y)

        return y

    def insert(self, val):
        self.root = self._insert(self.root, val)

    def _insert(self, node, val):
        # Standard BST insertion
        if not node:
            return AVLNode(val)

        if val < node.val:
            node.left = self._insert(node.left, val)
        elif val > node.val:
            node.right = self._insert(node.right, val)
        else:
            return node  # Ignore duplicates

        # Update height
        self.update_height(node)

        # Check balance factor
        bf = self.balance_factor(node)

        # Rotations for the 4 imbalance cases
        # LL: Left-Left -> Right rotation
        if bf > 1 and val < node.left.val:
            return self.right_rotate(node)

        # RR: Right-Right -> Left rotation
        if bf < -1 and val > node.right.val:
            return self.left_rotate(node)

        # LR: Left-Right -> Left rotation + Right rotation
        if bf > 1 and val > node.left.val:
            node.left = self.left_rotate(node.left)
            return self.right_rotate(node)

        # RL: Right-Left -> Right rotation + Left rotation
        if bf < -1 and val < node.right.val:
            node.right = self.right_rotate(node.right)
            return self.left_rotate(node)

        return node

    def delete(self, val):
        self.root = self._delete(self.root, val)

    def _delete(self, node, val):
        if not node:
            return None

        if val < node.val:
            node.left = self._delete(node.left, val)
        elif val > node.val:
            node.right = self._delete(node.right, val)
        else:
            # 0 or 1 child
            if not node.left:
                return node.right
            if not node.right:
                return node.left

            # 2 children: Replace with in-order successor
            successor = node.right
            while successor.left:
                successor = successor.left
            node.val = successor.val
            node.right = self._delete(node.right, successor.val)

        self.update_height(node)

        # Rebalancing
        bf = self.balance_factor(node)

        if bf > 1 and self.balance_factor(node.left) >= 0:
            return self.right_rotate(node)
        if bf > 1 and self.balance_factor(node.left) < 0:
            node.left = self.left_rotate(node.left)
            return self.right_rotate(node)
        if bf < -1 and self.balance_factor(node.right) <= 0:
            return self.left_rotate(node)
        if bf < -1 and self.balance_factor(node.right) > 0:
            node.right = self.right_rotate(node.right)
            return self.left_rotate(node)

        return node


# AVL tree complexities:
# Search: O(log n) guaranteed
# Insert: O(log n) (at most 2 rotations)
# Delete: O(log n) (at most O(log n) rotations)
# Space: O(n)

# Usage example
avl = AVLTree()
for val in [10, 20, 30, 40, 50, 25]:
    avl.insert(val)
# Inserting sorted data still maintains balance

print(avl.root.val)           # 30 (root)
print(avl.balance_factor(avl.root))  # 0 or +/-1
```

---

## 4. Red-Black Tree

### 4.1 Red-Black Tree Properties

```
Five properties of Red-Black trees:

  1. Each node is either red or black
  2. The root is black
  3. All leaves (NIL) are black
  4. Children of red nodes are all black (no consecutive reds)
  5. Every path from a node to its descendant leaves contains the same number
     of black nodes (Black Height)

  These properties ensure:
  - Longest path <= 2 x shortest path
  - Height <= 2 x log2(n+1)
  - Search/insert/delete all O(log n) guaranteed

  Comparison with AVL trees:
  +----------------+--------------+--------------+
  |                | AVL Tree     | Red-Black    |
  +----------------+--------------+--------------+
  | Balance        | Strict (+/-1)| Loose        |
  | Search speed   | Slightly     | Slightly     |
  |                | faster       | slower       |
  | Insert/Delete  | More         | Fewer        |
  |                | rotations    | rotations    |
  | Memory         | Height info  | 1 bit for    |
  |                | needed       | color        |
  | Use case       | Read-heavy   | Write-heavy  |
  | Implementations| -            | Java TreeMap |
  |                |              | C++ std::map |
  |                |              | Linux rbtree |
  +----------------+--------------+--------------+
```

### 4.2 Conceptual Red-Black Tree Implementation

```python
class RBColor:
    RED = True
    BLACK = False

class RBNode:
    def __init__(self, val, color=RBColor.RED):
        self.val = val
        self.left = None
        self.right = None
        self.parent = None
        self.color = color  # New nodes are red

class RedBlackTree:
    def __init__(self):
        self.NIL = RBNode(0, color=RBColor.BLACK)  # Sentinel node
        self.root = self.NIL

    def insert(self, val):
        """BST insertion + Red-Black tree fixup"""
        new_node = RBNode(val)
        new_node.left = self.NIL
        new_node.right = self.NIL

        # BST insertion
        parent = None
        current = self.root
        while current != self.NIL:
            parent = current
            if val < current.val:
                current = current.left
            else:
                current = current.right

        new_node.parent = parent
        if parent is None:
            self.root = new_node
        elif val < parent.val:
            parent.left = new_node
        else:
            parent.right = new_node

        # Fix Red-Black tree properties
        self._fix_insert(new_node)

    def _fix_insert(self, node):
        """Post-insertion fixup (3 cases)"""
        while node.parent and node.parent.color == RBColor.RED:
            if node.parent == node.parent.parent.left:
                uncle = node.parent.parent.right

                if uncle.color == RBColor.RED:
                    # Case 1: Uncle is red -> Color flip
                    node.parent.color = RBColor.BLACK
                    uncle.color = RBColor.BLACK
                    node.parent.parent.color = RBColor.RED
                    node = node.parent.parent
                else:
                    if node == node.parent.right:
                        # Case 2: Uncle is black, node is right child -> Left rotate
                        node = node.parent
                        self._left_rotate(node)
                    # Case 3: Uncle is black, node is left child -> Right rotate
                    node.parent.color = RBColor.BLACK
                    node.parent.parent.color = RBColor.RED
                    self._right_rotate(node.parent.parent)
            else:
                # Symmetric cases (swap left and right)
                uncle = node.parent.parent.left

                if uncle.color == RBColor.RED:
                    node.parent.color = RBColor.BLACK
                    uncle.color = RBColor.BLACK
                    node.parent.parent.color = RBColor.RED
                    node = node.parent.parent
                else:
                    if node == node.parent.left:
                        node = node.parent
                        self._right_rotate(node)
                    node.parent.color = RBColor.BLACK
                    node.parent.parent.color = RBColor.RED
                    self._left_rotate(node.parent.parent)

        self.root.color = RBColor.BLACK  # Property 2: Root is black

    def _left_rotate(self, x):
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

    def _right_rotate(self, y):
        x = y.left
        y.left = x.right
        if x.right != self.NIL:
            x.right.parent = y
        x.parent = y.parent
        if y.parent is None:
            self.root = x
        elif y == y.parent.left:
            y.parent.left = x
        else:
            y.parent.right = x
        x.right = y
        y.parent = x

    def search(self, val):
        return self._search(self.root, val)

    def _search(self, node, val):
        if node == self.NIL or node.val == val:
            return node if node != self.NIL else None
        if val < node.val:
            return self._search(node.left, val)
        return self._search(node.right, val)


# Where Red-Black trees are used (languages/libraries):
# Java: TreeMap, TreeSet
# C++: std::map, std::set, std::multimap, std::multiset
# C#: SortedDictionary, SortedSet
# Linux: Completely Fair Scheduler (CFS), memory management
# Nginx: Timer management
```

---

## 5. B-Trees and B+ Trees

### 5.1 B-Tree Basics

```
B-Tree: A multi-way search tree that minimizes disk I/O

  Properties of a B-tree of order t:
  - All leaves are at the same level
  - Each node (except root) has between t-1 and 2t-1 keys
  - The root has between 1 and 2t-1 keys
  - Number of children per node = number of keys + 1
  - Keys within each node are sorted

  Example: B-tree of order t=2 (also called a 2-3-4 tree)
          [10, 20]
         /    |    \
    [3,5]  [12,15] [25,30,35]

  Disk I/O optimization:
  - 1 node = 1 disk page (typically 4KB-16KB)
  - Node size is matched to page size
  - Very low height -> few I/O operations

  Example: t=1000 (up to 1999 keys per node)
  - Height 1: up to 1,999 keys
  - Height 2: up to ~4 million keys
  - Height 3: up to ~8 billion keys
  -> A database with 8 billion records can be searched in just 3 I/Os
```

### 5.2 B+ Tree Characteristics

```
B+ Tree: A variant of the B-tree (the standard for database indexes)

  Differences from B-tree:
  1. Data is stored only in leaf nodes (internal nodes store only keys)
  2. Leaf nodes are connected by a linked list (fast range queries)
  3. Internal nodes store no data, so they can hold more keys

  Structure:
  Internal node:   [  10  |  20  |  30  ]
                   / |    |     |    \
  Leaf nodes:    [3,5,8]->[10,12,15]->[20,22,25]->[30,35,40]

  Advantages:
  +--------------------+--------------+--------------+
  | Operation          | B-Tree       | B+ Tree      |
  +--------------------+--------------+--------------+
  | Point lookup       | O(log_B n)   | O(log_B n)   |
  | Range query        | O(log_B n+k) | O(log_B n+k) |
  |                    | (inefficient)| (leaf links) |
  | Full scan          | All nodes    | Leaves only  |
  | Internal node      | Keys + data  | Keys only    |
  | capacity           |              |              |
  +--------------------+--------------+--------------+

  Real-world usage:
  - PostgreSQL: B+ tree index (btree)
  - MySQL InnoDB: Clustered index (B+ tree)
  - SQLite: B+ tree-based page storage
  - File systems: NTFS, ext4, Btrfs
```

### 5.3 B+ Tree Operation Concepts

```python
class BPlusNode:
    def __init__(self, is_leaf=False):
        self.keys = []
        self.children = []  # Internal nodes: child nodes; Leaves: data
        self.is_leaf = is_leaf
        self.next = None    # Linked list for leaf nodes

class BPlusTree:
    def __init__(self, order=4):
        """order: Maximum number of children per node"""
        self.order = order
        self.root = BPlusNode(is_leaf=True)

    def search(self, key):
        """Search for a key and return the corresponding value"""
        node = self.root

        # Traverse internal nodes downward
        while not node.is_leaf:
            # Find the smallest index >= key
            i = 0
            while i < len(node.keys) and key >= node.keys[i]:
                i += 1
            node = node.children[i]

        # Search in the leaf node
        for i, k in enumerate(node.keys):
            if k == key:
                return node.children[i]  # children in leaves hold data

        return None  # Not found

    def range_search(self, start, end):
        """Range search: start <= key <= end"""
        # First find the position of start
        node = self.root
        while not node.is_leaf:
            i = 0
            while i < len(node.keys) and start >= node.keys[i]:
                i += 1
            node = node.children[i]

        # Traverse leaf nodes via linked list to collect results
        result = []
        while node:
            for i, k in enumerate(node.keys):
                if start <= k <= end:
                    result.append((k, node.children[i]))
                elif k > end:
                    return result
            node = node.next  # Move to the next leaf node

        return result

# Using a B+ tree as an index
# CREATE INDEX idx_users_age ON users(age);
# -> A B+ tree is built
# SELECT * FROM users WHERE age BETWEEN 20 AND 30;
# -> B+ tree range search O(log n + k) (k = number of results)
```

---

## 6. Trie

### 6.1 Basic Structure and Operations

```python
class TrieNode:
    def __init__(self):
        self.children = {}  # Character -> TrieNode
        self.is_end = False  # End of a word
        self.count = 0       # Number of words with this prefix

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str):
        """Insert a word: O(m), m is the word length"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.count += 1
        node.is_end = True

    def search(self, word: str) -> bool:
        """Exact match search: O(m)"""
        node = self._find_node(word)
        return node is not None and node.is_end

    def starts_with(self, prefix: str) -> bool:
        """Prefix match search: O(m)"""
        return self._find_node(prefix) is not None

    def count_prefix(self, prefix: str) -> int:
        """Count words with a given prefix: O(m)"""
        node = self._find_node(prefix)
        return node.count if node else 0

    def _find_node(self, prefix: str):
        """Return the node corresponding to a prefix"""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node

    def autocomplete(self, prefix: str, limit: int = 10) -> list:
        """Autocomplete: Return candidates from a prefix"""
        node = self._find_node(prefix)
        if not node:
            return []

        result = []
        self._collect_words(node, prefix, result, limit)
        return result

    def _collect_words(self, node, prefix, result, limit):
        if len(result) >= limit:
            return
        if node.is_end:
            result.append(prefix)
        for char in sorted(node.children):
            self._collect_words(node.children[char], prefix + char, result, limit)

    def delete(self, word: str) -> bool:
        """Delete a word"""
        return self._delete(self.root, word, 0)

    def _delete(self, node, word, depth):
        if depth == len(word):
            if not node.is_end:
                return False
            node.is_end = False
            node.count -= 1
            return len(node.children) == 0  # Can delete if no children

        char = word[depth]
        if char not in node.children:
            return False

        should_delete = self._delete(node.children[char], word, depth + 1)
        if should_delete:
            del node.children[char]
        node.count -= 1

        return len(node.children) == 0 and not node.is_end


# Usage example: Autocomplete
trie = Trie()
words = ["apple", "app", "application", "apply", "apt",
         "banana", "band", "bandwidth", "ban"]
for w in words:
    trie.insert(w)

print(trie.search("apple"))        # True
print(trie.search("appl"))         # False
print(trie.starts_with("app"))     # True
print(trie.count_prefix("app"))    # 4 (apple, app, application, apply)
print(trie.autocomplete("app"))    # ['app', 'apple', 'application', 'apply']
print(trie.autocomplete("ban"))    # ['ban', 'banana', 'band', 'bandwidth']
```

### 6.2 Compressed Trie (Radix Tree / Patricia Tree)

```python
class RadixNode:
    def __init__(self, prefix=""):
        self.prefix = prefix
        self.children = {}
        self.is_end = False
        self.value = None

class RadixTree:
    """Compressed Trie: Merges common prefixes onto edges"""

    def __init__(self):
        self.root = RadixNode()

    def insert(self, key, value=None):
        node = self.root
        remaining = key

        while remaining:
            # Look for a child with a common prefix
            match_found = False
            for char, child in node.children.items():
                common = self._common_prefix(remaining, child.prefix)

                if not common:
                    continue

                match_found = True

                if common == child.prefix:
                    # Child's prefix is a complete match
                    remaining = remaining[len(common):]
                    node = child
                    break
                else:
                    # Partial match -> Split the node
                    new_node = RadixNode(common)
                    child.prefix = child.prefix[len(common):]
                    new_node.children[child.prefix[0]] = child
                    node.children[common[0]] = new_node

                    remaining = remaining[len(common):]
                    node = new_node
                    break

            if not match_found:
                # Add a new edge
                new_node = RadixNode(remaining)
                new_node.is_end = True
                new_node.value = value
                node.children[remaining[0]] = new_node
                return

        node.is_end = True
        node.value = value

    def _common_prefix(self, s1, s2):
        i = 0
        while i < len(s1) and i < len(s2) and s1[i] == s2[i]:
            i += 1
        return s1[:i]


# Advantages of Radix Tree:
# - Memory efficient (shares common prefixes)
# - Fast search for long keys
# Used in:
# - Linux kernel routing table
# - HTTP routers (Go frameworks like gin, echo)
# - Longest prefix match for IP addresses

# Regular Trie vs Radix Tree:
# Trie:        [r]-[o]-[m]-[u]-[l]-[u]-[s]
#                           [a]-[n]-[e]
#                           [a]-[n]-[c]-[e]
# Radix Tree:  [rom]-[ulus]
#                    -[an]-[e]
#                         -[ce]
```

---

## 7. Heaps and Priority Queues

### 7.1 Binary Heap

```python
class MinHeap:
    """Min-heap implementation"""

    def __init__(self):
        self.heap = []

    def parent(self, i):
        return (i - 1) // 2

    def left_child(self, i):
        return 2 * i + 1

    def right_child(self, i):
        return 2 * i + 2

    def push(self, val):
        """Add element: O(log n)"""
        self.heap.append(val)
        self._sift_up(len(self.heap) - 1)

    def pop(self):
        """Remove minimum element: O(log n)"""
        if not self.heap:
            raise IndexError("empty heap")
        if len(self.heap) == 1:
            return self.heap.pop()

        min_val = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._sift_down(0)
        return min_val

    def peek(self):
        """View minimum element: O(1)"""
        if not self.heap:
            raise IndexError("empty heap")
        return self.heap[0]

    def _sift_up(self, i):
        """Move a node up (restore heap property)"""
        while i > 0 and self.heap[i] < self.heap[self.parent(i)]:
            p = self.parent(i)
            self.heap[i], self.heap[p] = self.heap[p], self.heap[i]
            i = p

    def _sift_down(self, i):
        """Move a node down (restore heap property)"""
        n = len(self.heap)
        while True:
            smallest = i
            left = self.left_child(i)
            right = self.right_child(i)

            if left < n and self.heap[left] < self.heap[smallest]:
                smallest = left
            if right < n and self.heap[right] < self.heap[smallest]:
                smallest = right

            if smallest == i:
                break

            self.heap[i], self.heap[smallest] = self.heap[smallest], self.heap[i]
            i = smallest

    def __len__(self):
        return len(self.heap)

    def __bool__(self):
        return len(self.heap) > 0


# Python's heapq module (min-heap)
import heapq

# Basic operations
h = []
heapq.heappush(h, 5)
heapq.heappush(h, 3)
heapq.heappush(h, 7)
heapq.heappush(h, 1)
print(heapq.heappop(h))  # 1 (minimum)

# Top-K problem
def top_k_largest(nums, k):
    """Get the k-th largest values from an array in O(n log k)"""
    return heapq.nlargest(k, nums)

def top_k_frequent(nums, k):
    """Get the k most frequent elements"""
    from collections import Counter
    count = Counter(nums)
    return heapq.nlargest(k, count.keys(), key=count.get)

# Streaming median
class MedianFinder:
    """Compute the median in O(log n) using two heaps"""

    def __init__(self):
        self.lo = []   # Max-heap (using negated values)
        self.hi = []   # Min-heap

    def add_num(self, num):
        heapq.heappush(self.lo, -num)

        # Ensure max of lo <= min of hi
        heapq.heappush(self.hi, -heapq.heappop(self.lo))

        # Size balance: size of lo >= size of hi
        if len(self.lo) < len(self.hi):
            heapq.heappush(self.lo, -heapq.heappop(self.hi))

    def find_median(self):
        if len(self.lo) > len(self.hi):
            return -self.lo[0]
        return (-self.lo[0] + self.hi[0]) / 2

# Usage example
mf = MedianFinder()
mf.add_num(1)
mf.add_num(2)
print(mf.find_median())  # 1.5
mf.add_num(3)
print(mf.find_median())  # 2.0
```

---

## 8. Tree Algorithm Problems

### 8.1 Recursive Approaches

```python
# Maximum depth of a tree
def max_depth(root):
    if not root:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))

# Diameter of a tree (longest path)
def diameter(root):
    """Longest path between any two nodes"""
    result = [0]

    def dfs(node):
        if not node:
            return 0
        left = dfs(node.left)
        right = dfs(node.right)
        result[0] = max(result[0], left + right)
        return 1 + max(left, right)

    dfs(root)
    return result[0]

# Invert a tree (mirror)
def invert_tree(root):
    if not root:
        return None
    root.left, root.right = invert_tree(root.right), invert_tree(root.left)
    return root

# Path sum
def has_path_sum(root, target_sum):
    """Check if any root-to-leaf path sums to target_sum"""
    if not root:
        return False
    if not root.left and not root.right:
        return root.val == target_sum
    return (has_path_sum(root.left, target_sum - root.val) or
            has_path_sum(root.right, target_sum - root.val))

# Enumerate all paths with a given sum
def path_sum_all(root, target_sum):
    """Return all paths that satisfy the condition"""
    result = []

    def dfs(node, remaining, path):
        if not node:
            return
        path.append(node.val)
        if not node.left and not node.right and remaining == node.val:
            result.append(path[:])  # Append a copy
        dfs(node.left, remaining - node.val, path)
        dfs(node.right, remaining - node.val, path)
        path.pop()  # Backtrack

    dfs(root, target_sum, [])
    return result

# Check if two trees are identical
def is_same_tree(p, q):
    if not p and not q:
        return True
    if not p or not q:
        return False
    return (p.val == q.val and
            is_same_tree(p.left, q.left) and
            is_same_tree(p.right, q.right))

# Check if one tree is a subtree of another
def is_subtree(root, sub_root):
    """Check whether sub_root is a subtree of root"""
    if not root:
        return False
    if is_same_tree(root, sub_root):
        return True
    return is_subtree(root.left, sub_root) or is_subtree(root.right, sub_root)

# Check if a tree is symmetric
def is_symmetric(root):
    def mirror(left, right):
        if not left and not right:
            return True
        if not left or not right:
            return False
        return (left.val == right.val and
                mirror(left.left, right.right) and
                mirror(left.right, right.left))

    return mirror(root.left, root.right) if root else True
```

### 8.2 Tree Construction

```python
# Build tree from pre-order + in-order traversals
def build_tree_from_preorder_inorder(preorder, inorder):
    """Reconstruct a tree from its pre-order and in-order traversals"""
    if not preorder or not inorder:
        return None

    # First element of pre-order is the root
    root_val = preorder[0]
    root = TreeNode(root_val)

    # Find the root's position in in-order
    mid = inorder.index(root_val)

    # Recursively build left and right subtrees
    root.left = build_tree_from_preorder_inorder(
        preorder[1:mid+1], inorder[:mid]
    )
    root.right = build_tree_from_preorder_inorder(
        preorder[mid+1:], inorder[mid+1:]
    )

    return root

# Sorted array -> BST
def sorted_array_to_bst(nums):
    """Build a height-balanced BST from a sorted array"""
    if not nums:
        return None

    mid = len(nums) // 2
    root = TreeNode(nums[mid])
    root.left = sorted_array_to_bst(nums[:mid])
    root.right = sorted_array_to_bst(nums[mid+1:])

    return root

# Usage example
nums = [1, 2, 3, 4, 5, 6, 7]
root = sorted_array_to_bst(nums)
# Result:
#       4
#      / \
#     2   6
#    / \ / \
#   1  3 5  7


# Tree serialization/deserialization
import json

def serialize(root):
    """Serialize a tree to a JSON string"""
    if not root:
        return "null"
    return json.dumps({
        "val": root.val,
        "left": json.loads(serialize(root.left)),
        "right": json.loads(serialize(root.right))
    })

def deserialize(data):
    """Deserialize a tree from a JSON string"""
    if data == "null":
        return None
    obj = json.loads(data)
    if obj is None:
        return None
    root = TreeNode(obj["val"])
    root.left = deserialize(json.dumps(obj["left"]))
    root.right = deserialize(json.dumps(obj["right"]))
    return root

# BFS-style serialization (LeetCode format)
def serialize_bfs(root):
    """BFS-style serialization: [4,2,6,1,3,5,7]"""
    if not root:
        return "[]"
    result = []
    queue = deque([root])
    while queue:
        node = queue.popleft()
        if node:
            result.append(node.val)
            queue.append(node.left)
            queue.append(node.right)
        else:
            result.append(None)

    # Remove trailing Nones
    while result and result[-1] is None:
        result.pop()

    return str(result)
```

---

## 9. Trees in Practice

### 9.1 Real-World Use Cases

```
Real-world examples where trees are used:

  1. File systems: Directories = tree structure
     /
     +-- home/
     |   +-- user/
     |   |   +-- documents/
     |   |   +-- downloads/
     |   +-- admin/
     +-- etc/
         +-- nginx/
         +-- ssh/

  2. DOM: Parent-child relationships of HTML elements
     html
     +-- head
     |   +-- title
     |   +-- meta
     +-- body
         +-- div#header
         +-- div#content

  3. AST: Syntax tree for source code (compilers)
     x = 3 + 5 * 2
         =
        / \
       x   +
          / \
         3   *
            / \
           5   2

  4. B+ tree: Database indexes (PostgreSQL, MySQL)

  5. JSON/XML: Nested data structures

  6. Decision tree: Machine learning classifiers

  7. Huffman tree: Data compression

  8. Trie: Dictionaries, autocomplete

  9. React Virtual DOM: Diff detection algorithm (Reconciliation)

  10. Git: Commit tree, Merkle tree
```

### 9.2 System Design with Tree Structures

```python
# 1. File system tree structure
class FileNode:
    def __init__(self, name, is_dir=False):
        self.name = name
        self.is_dir = is_dir
        self.children = {}  # name -> FileNode
        self.content = ""   # For files
        self.size = 0

class FileSystem:
    def __init__(self):
        self.root = FileNode("/", is_dir=True)

    def mkdir(self, path):
        """Create a directory"""
        parts = path.strip("/").split("/")
        node = self.root
        for part in parts:
            if part not in node.children:
                node.children[part] = FileNode(part, is_dir=True)
            node = node.children[part]

    def write(self, path, content):
        """Write to a file"""
        parts = path.strip("/").split("/")
        node = self.root
        for part in parts[:-1]:
            if part not in node.children:
                node.children[part] = FileNode(part, is_dir=True)
            node = node.children[part]
        filename = parts[-1]
        if filename not in node.children:
            node.children[filename] = FileNode(filename)
        node.children[filename].content = content
        node.children[filename].size = len(content)

    def read(self, path):
        """Read a file"""
        parts = path.strip("/").split("/")
        node = self.root
        for part in parts:
            if part not in node.children:
                return None
            node = node.children[part]
        return node.content if not node.is_dir else None

    def ls(self, path="/"):
        """List directory contents"""
        parts = path.strip("/").split("/")
        node = self.root
        if path != "/":
            for part in parts:
                if part and part in node.children:
                    node = node.children[part]
        return sorted(node.children.keys())

    def du(self, path="/"):
        """Calculate disk usage (recursive)"""
        parts = path.strip("/").split("/")
        node = self.root
        if path != "/":
            for part in parts:
                if part and part in node.children:
                    node = node.children[part]
        return self._calculate_size(node)

    def _calculate_size(self, node):
        if not node.is_dir:
            return node.size
        total = 0
        for child in node.children.values():
            total += self._calculate_size(child)
        return total

# Usage example
fs = FileSystem()
fs.mkdir("/home/user/documents")
fs.write("/home/user/documents/readme.txt", "Hello World")
fs.write("/home/user/documents/notes.txt", "Important notes here")
print(fs.ls("/home/user/documents"))  # ['notes.txt', 'readme.txt']
print(fs.read("/home/user/documents/readme.txt"))  # "Hello World"
print(fs.du("/home/user"))  # 31


# 2. Organization hierarchy management
class OrgNode:
    def __init__(self, name, title):
        self.name = name
        self.title = title
        self.reports = []  # Direct reports

class OrgChart:
    def __init__(self, ceo_name, ceo_title="CEO"):
        self.root = OrgNode(ceo_name, ceo_title)
        self.lookup = {ceo_name: self.root}

    def add_report(self, manager_name, employee_name, title):
        manager = self.lookup.get(manager_name)
        if not manager:
            raise ValueError(f"Manager {manager_name} not found")
        employee = OrgNode(employee_name, title)
        manager.reports.append(employee)
        self.lookup[employee_name] = employee

    def get_chain(self, name):
        """Return the chain of command (path to root)"""
        path = []
        self._find_path(self.root, name, path)
        return path

    def _find_path(self, node, target, path):
        path.append(node.name)
        if node.name == target:
            return True
        for report in node.reports:
            if self._find_path(report, target, path):
                return True
        path.pop()
        return False

    def count_reports(self, name):
        """Total number of direct and indirect reports"""
        node = self.lookup.get(name)
        if not node:
            return 0
        return self._count(node) - 1  # Exclude self

    def _count(self, node):
        return 1 + sum(self._count(r) for r in node.reports)
```

---

## 10. Practice Exercises

### Exercise 1: BST Operations (Basic)
Implement BST insertion, search, deletion, and in-order traversal. Additionally implement the following operations:
- Retrieve the k-th smallest element
- List nodes within a given range
- Validate BST integrity

### Exercise 2: Tree Recursion (Intermediate)
Implement the following operations recursively:
- Maximum depth, diameter, left-right inversion of a tree
- Check whether two trees are identical
- List all root-to-leaf paths where the path sum equals target_sum
- Compute the Lowest Common Ancestor (LCA)

### Exercise 3: Serialization (Advanced)
Implement functions to serialize a tree to a JSON string and deserialize it. Implement both BFS and DFS approaches.

### Exercise 4: AVL Tree Implementation (Intermediate)
Implement an AVL tree supporting insertion, deletion, and search. Verify the following:
- Balance is maintained after inserting sorted data
- Height is O(log n) compared to random data

### Exercise 5: Trie-Based Autocomplete (Intermediate)
Build an autocomplete system for English words using a Trie:
- Word insertion and search
- Listing candidates by prefix (sorted by frequency)
- Deletion support

### Exercise 6: Huffman Coding (Advanced)
Build a Huffman tree and implement text compression and decompression:
- Build the Huffman tree from character frequencies
- Encode text to a bit string
- Decode a bit string back to text
- Compute and display the compression ratio

---


## FAQ

### Q1: What is the most important point when studying this topic?

Gaining practical experience is the most important thing. Understanding deepens not just from theory, but by actually writing code and verifying its behavior.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts covered in this guide before moving on.

### Q3: How is this applied in practice?

Knowledge of this topic is frequently used in everyday development work. It is particularly important during code reviews and architecture design.

---

## Summary

| Data Structure | Operations | Use Cases |
|---------------|-----------|-----------|
| Binary Tree | Traversal O(n) | Expression evaluation, syntax trees |
| BST | Search/Insert O(log n) avg | Managing sorted data |
| AVL Tree | Search/Insert O(log n) guaranteed | Read-heavy scenarios |
| Red-Black Tree | Search/Insert O(log n) guaranteed | TreeMap, TreeSet, CFS |
| B+ Tree | Search O(log_B n) | DB indexes, file systems |
| Trie | Search O(m) | Dictionaries, prefix match, autocomplete |
| Heap | Min/Max retrieval O(1) | Priority queues, Top-K |
| Segment Tree | Range query O(log n) | Range minimum, range sum |

---

## Recommended Next Reading

---

## References
1. Cormen, T. H. "Introduction to Algorithms." Chapters 12-13, 18.
2. Sedgewick, R. "Algorithms." Chapter 3.2-3.3.
3. Knuth, D. E. "The Art of Computer Programming." Volume 3: Sorting and Searching.
4. Bayer, R., McCreight, E. "Organization and Maintenance of Large Ordered Indexes." 1972.
5. Fredkin, E. "Trie Memory." Communications of the ACM, 1960.
6. Guibas, L. J., Sedgewick, R. "A Dichromatic Framework for Balanced Trees." 1978.
7. Adelson-Velsky, G. M., Landis, E. M. "An Algorithm for the Organization of Information." 1962.
