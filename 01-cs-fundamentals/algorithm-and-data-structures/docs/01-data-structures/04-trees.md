# 木構造 — 二分木・BST・AVL/赤黒木・B木・Trie

> 階層的なデータを表現する木構造の各種バリエーションを学ぶ。二分探索木の基本から平衡木、B木、Trie まで体系的に解説する。

---

## この章で学ぶこと

1. **二分木と走査** — 前順・中順・後順・レベル順
2. **二分探索木（BST）** と平衡木（AVL・赤黒木）の原理
3. **B木と Trie** — ディスクアクセス最適化と文字列検索
4. **セグメント木と Fenwick 木** — 区間クエリの高速処理
5. **実務応用** — DB インデックス、オートコンプリート、構文解析

---

## 1. 木の基本用語

```
        [A]         ← 根 (root)、深さ 0
       /   \
     [B]   [C]      ← 深さ 1
    / \      \
  [D] [E]   [F]     ← 深さ 2 (葉: D, E, F)

  用語:
  - 根 (root): 最上位のノード (A)
  - 葉 (leaf): 子を持たないノード (D, E, F)
  - 内部ノード (internal): 子を持つノード (A, B, C)
  - 深さ (depth): 根からの距離
  - 高さ (height): 葉までの最大距離（木の高さ = 2）
  - 部分木 (subtree): 任意のノードを根とする木
  - 次数 (degree): ノードの子の数
  - レベル (level): 深さと同義（根がレベル 0）
  - 祖先 (ancestor): 根までのパス上のノード
  - 子孫 (descendant): 部分木内の全ノード
  - 兄弟 (sibling): 同じ親を持つノード (D, E は兄弟)
```

### 1.1 木の種類

```
完全二分木 (Complete):          満二分木 (Full/Perfect):
       [A]                           [A]
      /   \                         /   \
    [B]   [C]                     [B]   [C]
   / \   /                       / \   / \
  [D] [E][F]                   [D][E][F][G]
  最後のレベルは左詰め          全レベルが完全に埋まっている

二分木 (Binary):                N分木 (N-ary):
       [A]                           [A]
      /   \                        / | \
    [B]   [C]                    [B][C][D]
   /        \                    |    / \
  [D]       [F]                 [E] [F][G]
  各ノード最大2子               各ノード最大N子
```

### 1.2 木の性質

```python
# 二分木の重要な性質:
# - n ノードの二分木の辺の数 = n - 1
# - 高さ h の二分木のノード数: 最小 h+1, 最大 2^(h+1) - 1
# - n ノードの完全二分木の高さ: floor(log2(n))
# - 高さ h の満二分木のノード数: 2^(h+1) - 1
# - 葉の数 = 内部ノード（子が2つ）の数 + 1 （満二分木）

def tree_properties(n):
    """n ノードの完全二分木の性質を計算"""
    import math
    height = math.floor(math.log2(n)) if n > 0 else 0
    leaves = (n + 1) // 2  # 完全二分木の場合
    internal = n - leaves
    edges = n - 1
    print(f"ノード数: {n}")
    print(f"高さ: {height}")
    print(f"葉の数: {leaves}")
    print(f"内部ノード数: {internal}")
    print(f"辺の数: {edges}")
    return height, leaves, internal, edges

tree_properties(15)
# ノード数: 15, 高さ: 3, 葉の数: 8, 内部ノード数: 7, 辺の数: 14
```

---

## 2. 二分木の走査

### 2.1 再帰的走査

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def preorder(node):
    """前順走査 (根→左→右) — O(n)
    用途: 木のシリアライズ、式のプレフィックス表記
    """
    if not node:
        return []
    return [node.val] + preorder(node.left) + preorder(node.right)

def inorder(node):
    """中順走査 (左→根→右) — O(n)
    用途: BST でソート順に取得
    """
    if not node:
        return []
    return inorder(node.left) + [node.val] + inorder(node.right)

def postorder(node):
    """後順走査 (左→右→根) — O(n)
    用途: 部分木の削除、式の後置記法
    """
    if not node:
        return []
    return postorder(node.left) + postorder(node.right) + [node.val]

def levelorder(root):
    """レベル順走査 (BFS) — O(n)
    用途: 最短経路、レベルごとの処理
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

### 2.2 反復的走査（スタックを使用）

```python
def preorder_iterative(root):
    """前順走査の反復版 — O(n) 時間, O(h) 空間"""
    if not root:
        return []
    result, stack = [], [root]
    while stack:
        node = stack.pop()
        result.append(node.val)
        # 右を先にスタックに入れる（左を先に処理するため）
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    return result

def inorder_iterative(root):
    """中順走査の反復版 — O(n) 時間, O(h) 空間"""
    result, stack = [], []
    current = root
    while current or stack:
        # 左端まで進む
        while current:
            stack.append(current)
            current = current.left
        current = stack.pop()
        result.append(current.val)
        current = current.right
    return result

def postorder_iterative(root):
    """後順走査の反復版 — O(n) 時間, O(h) 空間
    前順（根→左→右）の逆は（右→左→根）＝後順の逆
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
    return result[::-1]  # 逆順にする
```

### 2.3 Morris 走査（O(1) 空間）

```python
def morris_inorder(root):
    """Morris 中順走査 — O(n) 時間, O(1) 空間
    スレッド化二分木の概念を使用。
    スタックも再帰も使わない。
    """
    result = []
    current = root
    while current:
        if not current.left:
            result.append(current.val)
            current = current.right
        else:
            # 左部分木の最右ノードを見つける
            predecessor = current.left
            while predecessor.right and predecessor.right != current:
                predecessor = predecessor.right

            if not predecessor.right:
                # スレッドを張る
                predecessor.right = current
                current = current.left
            else:
                # スレッドを除去
                predecessor.right = None
                result.append(current.val)
                current = current.right
    return result
```

### 2.4 走査の図解と用途

```
走査順序の例:
       [4]
      /   \
    [2]   [6]
   / \   / \
  [1] [3] [5] [7]

  前順 (Pre):  4, 2, 1, 3, 6, 5, 7   ← 木のコピー、シリアライズ
  中順 (In):   1, 2, 3, 4, 5, 6, 7   ← BST ではソート順
  後順 (Post): 1, 3, 2, 5, 7, 6, 4   ← 木の削除、式の評価
  レベル順:    4, 2, 6, 1, 3, 5, 7   ← 最短距離、幅優先

レベルごとの走査（実務で頻出）:
  Level 0: [4]
  Level 1: [2, 6]
  Level 2: [1, 3, 5, 7]
```

```python
def levelorder_by_level(root):
    """レベルごとにグループ化して返す"""
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

## 3. 二分探索木（BST）

### 3.1 完全実装

```python
class BST:
    """二分探索木: 左 < 根 < 右 の性質を持つ二分木

    平均計算量: O(log n)
    最悪計算量: O(n) — ソート済みデータ挿入時
    """
    def __init__(self):
        self.root = None

    def insert(self, val):
        """O(h), h = 木の高さ"""
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
        """O(h) — 3つのケースを処理"""
        self.root = self._delete(self.root, val)

    def _delete(self, node, val):
        if not node:
            return None
        if val < node.val:
            node.left = self._delete(node.left, val)
        elif val > node.val:
            node.right = self._delete(node.right, val)
        else:
            # ケース1: 葉ノード
            if not node.left and not node.right:
                return None
            # ケース2: 子が1つ
            if not node.left:
                return node.right
            if not node.right:
                return node.left
            # ケース3: 子が2つ — 右部分木の最小値で置換
            successor = self._min_node(node.right)
            node.val = successor.val
            node.right = self._delete(node.right, successor.val)
        return node

    def _min_node(self, node):
        """部分木の最小ノードを返す"""
        while node.left:
            node = node.left
        return node

    def find_min(self):
        """最小値 — O(h)"""
        if not self.root:
            return None
        return self._min_node(self.root).val

    def find_max(self):
        """最大値 — O(h)"""
        if not self.root:
            return None
        node = self.root
        while node.right:
            node = node.right
        return node.val

    def inorder_successor(self, val):
        """中順での後続ノード — O(h)"""
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
        """k番目に小さい要素 — O(h + k)"""
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
        """[low, high] 範囲内の全要素 — O(h + k), k = 結果数"""
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
        """BST の性質を検証"""
        return self._validate(self.root, float('-inf'), float('inf'))

    def _validate(self, node, min_val, max_val):
        if not node:
            return True
        if node.val <= min_val or node.val >= max_val:
            return False
        return (self._validate(node.left, min_val, node.val) and
                self._validate(node.right, node.val, max_val))

    def height(self):
        """木の高さ — O(n)"""
        return self._height(self.root)

    def _height(self, node):
        if not node:
            return -1
        return 1 + max(self._height(node.left), self._height(node.right))

    def size(self):
        """ノード数 — O(n)"""
        return self._size(self.root)

    def _size(self, node):
        if not node:
            return 0
        return 1 + self._size(node.left) + self._size(node.right)

# 使用例
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

### 3.2 BST の削除の図解

```
削除のケース:

ケース1: 葉ノードの削除 (値 1 を削除)
     5                5
    / \     →        / \
   3   7            3   7
  / \              / \
 1   4            _   4

ケース2: 子が1つのノードの削除 (値 3 を削除、左子のみ)
     5                5
    / \     →        / \
   3   7            1   7
  /
 1

ケース3: 子が2つのノードの削除 (値 5 を削除)
     5                6         ← 右部分木の最小値(6)で置換
    / \     →        / \
   3   7            3   7
  / \  /           / \
 1  4 6           1   4
```

---

## 4. AVL 木（自己平衡BST）

### 4.1 回転操作の図解

```
AVL木: 全ノードで |左の高さ - 右の高さ| ≤ 1

4つの不均衡パターンと回転:

=== LL ケース (右回転) ===
      z(3)                 y(2)
     /    \               /    \
    y(2)   T4    →      x(1)  z(3)
   /    \               / \   / \
  x(1)  T3            T1  T2 T3 T4
 / \
T1  T2

=== RR ケース (左回転) ===
  z(1)                   y(2)
 /    \                 /    \
T1   y(2)     →      z(1)  x(3)
    /    \            / \   / \
   T2   x(3)        T1 T2 T3 T4
       / \
      T3  T4

=== LR ケース (左回転 → 右回転) ===
      z(3)            z(3)               x(2)
     /    \          /    \             /    \
    y(1)   T4  →   x(2)  T4    →    y(1)  z(3)
   /    \          /    \            / \   / \
  T1   x(2)      y(1)  T3         T1 T2 T3 T4
      / \        / \
     T2  T3     T1  T2

=== RL ケース (右回転 → 左回転) ===
  z(1)             z(1)                  x(2)
 /    \           /    \                /    \
T1   y(3)   →   T1   x(2)    →      z(1)  y(3)
    /    \           /    \          / \   / \
   x(2)  T4        T2   y(3)       T1 T2 T3 T4
  / \                   / \
 T2  T3                T3  T4
```

### 4.2 AVL 木の完全実装

```python
class AVLNode:
    def __init__(self, val):
        self.val = val
        self.left = self.right = None
        self.height = 1

class AVLTree:
    """AVL 木: 全ノードのバランスファクターが [-1, 0, 1] の範囲内

    全操作が O(log n) を保証
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
        """右回転: LL ケースを修正"""
        y = z.left
        T3 = y.right
        y.right = z
        z.left = T3
        self.update_height(z)
        self.update_height(y)
        return y

    def rotate_left(self, z):
        """左回転: RR ケースを修正"""
        y = z.right
        T2 = y.left
        y.left = z
        z.right = T2
        self.update_height(z)
        self.update_height(y)
        return y

    def _balance(self, node):
        """ノードのバランスを復元"""
        self.update_height(node)
        bf = self.balance_factor(node)

        # LL ケース
        if bf > 1 and self.balance_factor(node.left) >= 0:
            return self.rotate_right(node)

        # LR ケース
        if bf > 1 and self.balance_factor(node.left) < 0:
            node.left = self.rotate_left(node.left)
            return self.rotate_right(node)

        # RR ケース
        if bf < -1 and self.balance_factor(node.right) <= 0:
            return self.rotate_left(node)

        # RL ケース
        if bf < -1 and self.balance_factor(node.right) > 0:
            node.right = self.rotate_right(node.right)
            return self.rotate_left(node)

        return node

    def insert(self, val):
        """O(log n) — 挿入後にバランス復元"""
        self.root = self._insert(self.root, val)

    def _insert(self, node, val):
        if not node:
            return AVLNode(val)
        if val < node.val:
            node.left = self._insert(node.left, val)
        elif val > node.val:
            node.right = self._insert(node.right, val)
        else:
            return node  # 重複は無視
        return self._balance(node)

    def delete(self, val):
        """O(log n) — 削除後にバランス復元"""
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
            # 右部分木の最小値で置換
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
        """中順走査 — ソート順"""
        result = []
        self._inorder(self.root, result)
        return result

    def _inorder(self, node, result):
        if node:
            self._inorder(node.left, result)
            result.append(node.val)
            self._inorder(node.right, result)

# 使用例
avl = AVLTree()
for val in [10, 20, 30, 40, 50, 25]:
    avl.insert(val)
print(avl.inorder())  # [10, 20, 25, 30, 40, 50]
print(f"根: {avl.root.val}")  # 30（バランスが保たれている）
print(f"高さ: {avl.height(avl.root)}")  # 2 or 3
```

---

## 5. 赤黒木

### 5.1 赤黒木の性質

```
赤黒木の5つの性質:
1. 各ノードは赤か黒
2. 根は黒
3. 全ての葉 (NIL) は黒
4. 赤ノードの子は黒（赤が連続しない）
5. 任意のノードからその子孫の葉までの全パスで、黒ノード数が同一（黒高さ）

例:
         [7:黒]
        /      \
    [3:赤]    [18:赤]
    /   \     /    \
  [1:黒] [5:黒] [10:黒] [22:黒]
                  \
                 [15:赤]

黒高さ = 2（任意のパスで黒ノードが2つ）

重要な定理:
- n ノードの赤黒木の高さ h ≤ 2 log2(n+1)
- したがって全操作が O(log n) を保証
```

### 5.2 赤黒木の実装

```python
class RBNode:
    RED = True
    BLACK = False

    def __init__(self, val, color=RED):
        self.val = val
        self.color = color
        self.left = self.right = self.parent = None

class RedBlackTree:
    """赤黒木: AVL 木より回転が少なく、挿入/削除が効率的

    Java の TreeMap, C++ の std::map が採用
    Linux カーネルのスケジューラでも使用
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
        """O(log n) — 挿入後に色の修正"""
        node = RBNode(val)
        node.left = self.NIL
        node.right = self.NIL

        # 通常の BST 挿入
        parent = None
        current = self.root
        while current != self.NIL:
            parent = current
            if val < current.val:
                current = current.left
            elif val > current.val:
                current = current.right
            else:
                return  # 重複は無視

        node.parent = parent
        if parent is None:
            self.root = node
        elif val < parent.val:
            parent.left = node
        else:
            parent.right = node

        # ケース1: 根の場合
        if node.parent is None:
            node.color = RBNode.BLACK
            return

        # ケース2: 親が黒ならOK
        if node.parent.color == RBNode.BLACK:
            return

        # 色の修正
        self._fix_insert(node)

    def _fix_insert(self, node):
        """挿入後の赤黒木性質の修正"""
        while node.parent and node.parent.color == RBNode.RED:
            if node.parent == node.parent.parent.left:
                uncle = node.parent.parent.right
                if uncle.color == RBNode.RED:
                    # ケース3: 叔父が赤 → 色の反転
                    node.parent.color = RBNode.BLACK
                    uncle.color = RBNode.BLACK
                    node.parent.parent.color = RBNode.RED
                    node = node.parent.parent
                else:
                    if node == node.parent.right:
                        # ケース4: LR → 左回転して LL に変換
                        node = node.parent
                        self._rotate_left(node)
                    # ケース5: LL → 右回転
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

# 使用例
rbt = RedBlackTree()
for val in [7, 3, 18, 10, 22, 8, 11, 26, 2, 6]:
    rbt.insert(val)
print(rbt.inorder())  # [2, 3, 6, 7, 8, 10, 11, 18, 22, 26]
```

---

## 6. B 木

### 6.1 B 木の構造と性質

```
B木 (order m = 4, 2-3-4 木):

              [17]
            /      \
      [5, 13]      [21, 30]
     /   |   \     /   |   \
  [1,3] [7,11] [14,16] [18,20] [22,25] [31,40]

B木の性質 (order m):
- 各ノードは最大 m-1 個のキーを持つ
- 各ノードは最大 m 個の子を持つ
- 根以外の各ノードは最低 ⌈m/2⌉ - 1 個のキーを持つ
- 全ての葉は同じレベルにある
- ノードのキーはソート済み

B木がデータベースに最適な理由:
1. ノードサイズをディスクブロック（4KB〜16KB）に合わせられる
2. 高さが非常に低い（3〜4レベルで数百万件を格納）
3. ディスク I/O 回数 = 木の高さ = O(log_m n)
4. 例: m=1000, n=10^9 → 高さ ≈ 3 回のディスクアクセス
```

### 6.2 B 木の実装

```python
class BTreeNode:
    def __init__(self, leaf=True):
        self.keys = []
        self.children = []
        self.leaf = leaf

class BTree:
    """B木 — ディスクベースのデータ構造

    データベースのインデックスとして広く使用される
    """
    def __init__(self, order=4):
        self.root = BTreeNode()
        self.order = order  # 最大子数
        self.min_keys = order // 2 - 1  # ノードの最小キー数

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
            # 根が満杯 → 分割して新しい根を作る
            new_root = BTreeNode(leaf=False)
            new_root.children.append(self.root)
            self._split_child(new_root, 0)
            self.root = new_root

        self._insert_non_full(self.root, key)

    def _insert_non_full(self, node, key):
        i = len(node.keys) - 1

        if node.leaf:
            # 葉ノードに挿入
            node.keys.append(None)
            while i >= 0 and key < node.keys[i]:
                node.keys[i + 1] = node.keys[i]
                i -= 1
            node.keys[i + 1] = key
        else:
            # 内部ノード: 適切な子に降りる
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1

            if len(node.children[i].keys) == self.order - 1:
                # 子が満杯なら分割
                self._split_child(node, i)
                if key > node.keys[i]:
                    i += 1

            self._insert_non_full(node.children[i], key)

    def _split_child(self, parent, i):
        """子ノードの分割"""
        order = self.order
        child = parent.children[i]
        mid = order // 2 - 1

        # 右半分を新しいノードに
        new_node = BTreeNode(leaf=child.leaf)
        new_node.keys = child.keys[mid + 1:]
        if not child.leaf:
            new_node.children = child.children[mid + 1:]

        # 中央のキーを親に昇格
        parent.keys.insert(i, child.keys[mid])
        parent.children.insert(i + 1, new_node)

        # 左半分を残す
        child.keys = child.keys[:mid]
        if not child.leaf:
            child.children = child.children[:mid + 1]

    def traverse(self, node=None):
        """全キーをソート順に返す"""
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

# 使用例
btree = BTree(order=4)
for key in [10, 20, 5, 6, 12, 30, 7, 17]:
    btree.insert(key)
print(btree.traverse())  # [5, 6, 7, 10, 12, 17, 20, 30]
```

### 6.3 B+ 木

```python
class BPlusLeafNode:
    """B+ 木の葉ノード: データは全て葉に格納"""
    def __init__(self):
        self.keys = []
        self.values = []  # 実データ or データへのポインタ
        self.next_leaf = None  # 右隣の葉へのポインタ

class BPlusInternalNode:
    """B+ 木の内部ノード: キーのみ（インデックス）"""
    def __init__(self):
        self.keys = []
        self.children = []

# B+ 木の特徴:
# 1. 全データが葉ノードにある → 範囲検索が高速
# 2. 葉ノードが連結リストで接続 → シーケンシャルアクセスが O(n)
# 3. 内部ノードはキーのみ → より多くのキーを格納できる
# 4. MySQL の InnoDB、PostgreSQL のインデックスが B+ 木
#
# B木 vs B+ 木:
# | 特性           | B木              | B+ 木             |
# |---------------|-----------------|-------------------|
# | データの位置    | 全ノード         | 葉ノードのみ       |
# | 範囲検索       | 中順走査が必要    | 葉の連結リストで高速 |
# | 重複キー       | 内部ノードにも    | 葉にのみ           |
# | ファンアウト    | やや少ない       | 多い（内部はキーのみ）|
# | 用途           | 一般的           | DB インデックスの標準 |
```

---

## 7. Trie（接頭辞木）

### 7.1 基本構造と実装

```
"app", "apple", "apt", "bat" を格納:

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

  * = 単語の終端
```

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.count = 0  # この接頭辞を持つ単語数

class Trie:
    """Trie（接頭辞木）— 文字列の高速検索

    用途: オートコンプリート、スペルチェック、IP ルーティング
    """
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        """O(m), m = 単語の長さ"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.count += 1
        node.is_end = True

    def search(self, word):
        """完全一致検索 — O(m)"""
        node = self._find(word)
        return node is not None and node.is_end

    def starts_with(self, prefix):
        """接頭辞検索 — O(m)"""
        return self._find(prefix) is not None

    def count_prefix(self, prefix):
        """指定接頭辞を持つ単語数 — O(m)"""
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
        """単語を削除 — O(m)"""
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
        """オートコンプリート — 接頭辞に続く単語を返す"""
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
        """全単語を返す"""
        return self.autocomplete("", limit=float('inf'))

# 使用例
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

### 7.2 圧縮 Trie（Patricia Trie / Radix Tree）

```python
class CompressedTrieNode:
    """圧縮 Trie: 分岐のないパスを1つのノードにまとめる

    通常の Trie:          圧縮 Trie:
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

    メモリ効率が大幅に改善される
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
            # 共通接頭辞を持つ子を探す
            found = False
            for key, child in node.children.items():
                common = self._common_prefix(word, key)
                if common:
                    found = True
                    if common == key:
                        # キー全体が共通 → 子ノードに進む
                        node = child
                        word = word[len(common):]
                    else:
                        # 部分的に共通 → ノードを分割
                        self._split(node, key, common)
                        node = node.children[common]
                        word = word[len(common):]
                    break
            if not found:
                # 新しい子ノードを追加
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

### 7.3 ビット Trie（XOR Trie）

```python
class BitTrie:
    """ビット Trie: 整数のビット表現で Trie を構築

    用途: 最大 XOR ペアの探索、IP ルーティング
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
        """num との XOR が最大になる値を探索 — O(max_bits)"""
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

# 使用例: 最大 XOR ペア
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

## 8. 実務応用パターン

### 8.1 式木（Expression Tree）

```python
class ExprNode:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def evaluate(node):
    """式木の評価 — 後順走査"""
    if not node.left and not node.right:
        return float(node.val)

    left_val = evaluate(node.left)
    right_val = evaluate(node.right)

    ops = {'+': lambda a, b: a + b,
           '-': lambda a, b: a - b,
           '*': lambda a, b: a * b,
           '/': lambda a, b: a / b}

    return ops[node.val](left_val, right_val)

# (3 + 4) * 2 の式木
expr = ExprNode('*',
    ExprNode('+', ExprNode('3'), ExprNode('4')),
    ExprNode('2'))
print(evaluate(expr))  # 14.0
```

### 8.2 最小共通祖先（LCA）

```python
def lowest_common_ancestor(root, p, q):
    """二分木の最小共通祖先 — O(n)

    LCA(p, q): p と q の両方を子孫に持つ最も深いノード
    """
    if not root or root.val == p or root.val == q:
        return root

    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)

    if left and right:
        return root  # p と q が左右に分かれている → root が LCA
    return left if left else right

# BST の場合は O(log n) に最適化可能
def lca_bst(root, p, q):
    """BST の最小共通祖先 — O(h)"""
    while root:
        if p < root.val and q < root.val:
            root = root.left
        elif p > root.val and q > root.val:
            root = root.right
        else:
            return root
    return None
```

### 8.3 木のシリアライズ/デシリアライズ

```python
def serialize(root):
    """木を文字列に変換 — 前順走査"""
    if not root:
        return "null"
    return f"{root.val},{serialize(root.left)},{serialize(root.right)}"

def deserialize(data):
    """文字列から木を復元"""
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

# 使用例
root = TreeNode(1, TreeNode(2), TreeNode(3, TreeNode(4), TreeNode(5)))
s = serialize(root)
print(s)  # "1,2,null,null,3,4,null,null,5,null,null"
restored = deserialize(s)
print(preorder(restored))  # [1, 2, 3, 4, 5]
```

### 8.4 二分木の直径

```python
def diameter_of_binary_tree(root):
    """二分木の直径（最長パスの辺の数）— O(n)"""
    max_diameter = [0]

    def height(node):
        if not node:
            return 0
        left_h = height(node.left)
        right_h = height(node.right)
        # このノードを経由するパスの長さ
        max_diameter[0] = max(max_diameter[0], left_h + right_h)
        return 1 + max(left_h, right_h)

    height(root)
    return max_diameter[0]
```

### 8.5 木の左側面ビュー

```python
def right_side_view(root):
    """二分木の右側面ビュー — O(n)

    各レベルで最も右にあるノードの値を返す
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
            if i == level_size - 1:  # レベルの最後のノード
                result.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    return result
```

---

## 9. 比較表

### 表1: 木構造の操作計算量

| 木の種類 | 探索 | 挿入 | 削除 | 空間 |
|---------|------|------|------|------|
| BST（平均） | O(log n) | O(log n) | O(log n) | O(n) |
| BST（最悪） | O(n) | O(n) | O(n) | O(n) |
| AVL 木 | O(log n) | O(log n) | O(log n) | O(n) |
| 赤黒木 | O(log n) | O(log n) | O(log n) | O(n) |
| B 木 (order m) | O(log n) | O(log n) | O(log n) | O(n) |
| B+ 木 | O(log n) | O(log n) | O(log n) | O(n) |
| Trie | O(m) | O(m) | O(m) | O(SIGMA * L * N) |
| Splay 木 | O(log n)償却 | O(log n)償却 | O(log n)償却 | O(n) |

### 表2: 平衡木の比較

| 特性 | AVL 木 | 赤黒木 | B 木 | Splay 木 |
|------|--------|--------|------|----------|
| 平衡条件 | 高さ差 <= 1 | 色条件 | ノード充填率 | なし（償却） |
| 探索速度 | 速い | やや遅い | ディスク最適 | 局所性高い |
| 挿入/削除 | 回転多い | 回転少ない | 分割/併合 | ジグザグ |
| 主用途 | メモリ内検索 | 汎用（TreeMap） | DB インデックス | キャッシュ |
| 高さ | <= 1.44 log n | <= 2 log n | <= log_t n | O(n) 最悪 |
| 回転数（挿入） | 最大2回 | 最大2回 | 0回 | O(log n) |
| 回転数（削除） | O(log n) | 最大3回 | 0回 | O(log n) |

### 表3: 言語標準ライブラリの木構造

| 言語 | 順序付きマップ | 内部実装 | 順序なしマップ |
|------|-------------|---------|-------------|
| Java | TreeMap | 赤黒木 | HashMap |
| C++ | std::map | 赤黒木 | std::unordered_map |
| Python | SortedDict (sortedcontainers) | B木ベース | dict |
| Rust | BTreeMap | B木 | HashMap |
| Go | なし（標準） | - | map |
| C# | SortedDictionary | 赤黒木 | Dictionary |

---

## 10. アンチパターン

### アンチパターン1: ソート済みデータをBSTに挿入

```python
# BAD: ソート済みデータ → 偏った木 → O(n)
bst = BST()
for i in range(1, 8):
    bst.insert(i)

# 結果: 1 → 2 → 3 → 4 → 5 → 6 → 7（連結リストと同じ）
# 探索: O(n)

# GOOD: 平衡木（AVL/赤黒木）を使う
avl = AVLTree()
for i in range(1, 8):
    avl.insert(i)
# → 常に O(log n) を保証

# GOOD: ランダムシャッフルしてから挿入
import random
data = list(range(1, 8))
random.shuffle(data)
bst2 = BST()
for i in data:
    bst2.insert(i)
# → 平均的にバランスの良い木になる

# GOOD: ソート済み配列から直接バランスBSTを構築
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

### アンチパターン2: Trie で不要なメモリ消費

```python
# BAD: 26文字分の配列を毎ノードに確保
class BadTrieNode:
    def __init__(self):
        self.children = [None] * 26  # 26 x 8byte = 208byte/node
        self.is_end = False

# GOOD: 辞書で必要な文字のみ
class GoodTrieNode:
    def __init__(self):
        self.children = {}  # 必要な文字だけ
        self.is_end = False

# BETTER: 圧縮 Trie で分岐のないパスをまとめる
# "application" の 11 ノード → 1-2 ノードに圧縮
```

### アンチパターン3: 再帰の深さ制限を考慮しない

```python
# BAD: 深い木で再帰が StackOverflow
def bad_inorder(node):
    if not node:
        return []
    return bad_inorder(node.left) + [node.val] + bad_inorder(node.right)
# Python のデフォルト再帰制限は 1000

# GOOD: 反復版を使う
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

# あるいは再帰制限を引き上げる（非推奨）
# import sys
# sys.setrecursionlimit(100000)
```

### アンチパターン4: BST のキーに可変オブジェクトを使用

```python
# BAD: 挿入後にキーの比較結果が変わる
class MutableKey:
    def __init__(self, val):
        self.val = val
    def __lt__(self, other):
        return self.val < other.val

# キーの値を変更すると BST の性質が壊れる

# GOOD: 不変のキーを使用する
# 数値、文字列、タプルなど
```

---

## 11. FAQ

### Q1: AVL 木と赤黒木はどちらを使うべきか？

**A:** 探索が多いなら AVL 木（木の高さが低い、最大 1.44 log n）、挿入・削除が多いなら赤黒木（回転が少ない、挿入は最大2回、削除は最大3回の回転）。実務では多くの言語の標準ライブラリが赤黒木を採用（Java TreeMap、C++ std::map）。ただし Rust は B木を標準（BTreeMap）に採用しており、キャッシュ効率が良い。

### Q2: B木はなぜデータベースに使われるか？

**A:** B木はノードが大きく（数百〜数千のキー）、ディスクブロックサイズに合わせられる。木の高さが非常に低い（3〜4レベルで数百万件を格納）ため、ディスクI/O回数が最小になる。B+木はさらに範囲検索に最適化されており、葉ノードが連結リストでつながっているため、ORDER BY や BETWEEN 句が高速。

### Q3: Trie のメモリ消費を減らすには？

**A:** 複数の手法がある:
1. **圧縮 Trie（Patricia Trie / Radix Tree）**: 分岐のないパスを1つのノードにまとめる
2. **辞書ベースの子管理**: 配列の代わりに辞書を使えばスパースなアルファベットでも効率的
3. **ダブルアレイ Trie**: 2つの配列で Trie を表現。メモリ効率と速度のバランスが良い
4. **HAT-Trie**: ハッシュテーブルと Trie のハイブリッド

### Q4: 二分木の問題を解く際のコツは？

**A:**
1. **再帰で考える**: ほとんどの二分木問題は「根での処理 + 左部分木 + 右部分木」に分解できる
2. **戻り値の型を明確に**: 関数が返すべき情報（高さ、結果、真偽値など）を事前に決める
3. **ベースケースを確認**: null ノードの処理を忘れない
4. **走査の選択**: 目的に合った走査を選ぶ（ソート順なら中順、コピーなら前順、削除なら後順）

### Q5: セグメント木と Fenwick 木の違いは？

**A:**
- **セグメント木**: 任意の区間クエリと区間更新に対応。遅延伝播で範囲更新が O(log n)。実装がやや複雑だが汎用性が高い。
- **Fenwick 木（BIT）**: 接頭辞和に特化。実装が簡単でメモリ効率が良い。ただし区間更新には追加の工夫が必要。
- 指針: 単純な累積和クエリなら Fenwick 木、複雑な区間操作ならセグメント木。

---

## 12. まとめ

| 項目 | ポイント |
|------|---------|
| 二分木走査 | 前順/中順/後順/レベル順の4種。反復版と Morris 走査も重要 |
| BST | 中順走査がソート順。最悪 O(n)。削除は3ケースの処理 |
| AVL 木 | 高さ差 <= 1。探索に最適。4パターンの回転 |
| 赤黒木 | 回転が少ない。挿入2回、削除3回が上限。汎用的 |
| B 木 | ディスク I/O 最適化。DB のインデックス。高ファンアウト |
| B+ 木 | データは葉のみ。葉が連結リスト。範囲検索に最適 |
| Trie | 接頭辞検索に O(m)。オートコンプリート。圧縮で省メモリ |
| 実務応用 | LCA、シリアライズ、式木、直径など多数の頻出パターン |

---

## 次に読むべきガイド

- [ヒープ — 二分ヒープとヒープソート](./05-heaps.md)
- [セグメント木 — 区間クエリと遅延伝播](../03-advanced/01-segment-tree.md)

---

## 参考文献

1. Cormen, T.H. et al. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. — 第12-13章「Binary Search Trees」「Red-Black Trees」、第18章「B-Trees」
2. Bayer, R. & McCreight, E. (1972). "Organization and maintenance of large ordered indexes." *Acta Informatica*, 1(3), 173-189.
3. Sedgewick, R. & Wayne, K. (2011). *Algorithms* (4th ed.). Addison-Wesley. — Trie の実装
4. Adelson-Velsky, G.M. & Landis, E.M. (1962). "An algorithm for the organization of information." *Soviet Mathematics Doklady*, 3, 1259-1263.
5. Guibas, L.J. & Sedgewick, R. (1978). "A dichromatic framework for balanced trees." *19th Annual Symposium on Foundations of Computer Science*.
