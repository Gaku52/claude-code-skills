# 木構造

> 木は階層関係を表現する最も自然なデータ構造であり、ファイルシステム、DOM、データベースインデックスの基盤である。

## この章で学ぶこと

- [ ] 木の基本用語と走査方法を理解する
- [ ] 二分探索木（BST）の操作と計算量を説明できる
- [ ] 平衡木（AVL木、赤黒木）の必要性を理解する
- [ ] B木/B+木のディスクI/O最適化の仕組みを理解する
- [ ] Trie（トライ木）の構造と用途を把握する
- [ ] セグメント木やフェニック木などの高度な木構造を知る

## 前提知識

- 再帰 → 参照: [[../03-algorithms/00-what-is-algorithm.md]]

---

## 1. 木の基礎

### 1.1 用語

```
木の用語:
         A          ← ルート（根）
        / \
       B   C        ← Aの子
      / \   \
     D   E   F      ← 葉（子を持たないノード）

  ノード(Node): 各要素
  エッジ(Edge): ノード間の接続
  ルート(Root): 最上位ノード（A）
  葉(Leaf): 子を持たないノード（D, E, F）
  高さ(Height): ルートから最も深い葉までのエッジ数（= 2）
  深さ(Depth): ルートからそのノードまでのエッジ数
  部分木(Subtree): あるノードを根とする木
  次数(Degree): ノードの子の数
  レベル(Level): ルートからの距離（ルート = レベル0）
  祖先(Ancestor): ルートからそのノードへのパス上のすべてのノード
  子孫(Descendant): そのノードから葉に向かうパス上のすべてのノード
  兄弟(Sibling): 同じ親を持つノード

木の種類:
  二分木(Binary Tree): 各ノードが最大2つの子を持つ
  完全二分木(Complete): 最後のレベル以外が完全に埋まっている
  満二分木(Full): すべてのノードが0個または2個の子を持つ
  完璧二分木(Perfect): すべての葉が同じレベルにある
  N分木(N-ary Tree): 各ノードが最大N個の子を持つ

完全二分木のノード数:
  高さ h の完璧二分木: 2^(h+1) - 1 ノード
  高さ h の完全二分木: 2^h ～ 2^(h+1) - 1 ノード
  n ノードの完全二分木の高さ: floor(log2(n))
```

### 1.2 木の走査（Traversal）

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# 前順（Pre-order）: ルート → 左 → 右
def preorder(node):
    if not node: return []
    return [node.val] + preorder(node.left) + preorder(node.right)

# 中順（In-order）: 左 → ルート → 右 ← BSTでソート順
def inorder(node):
    if not node: return []
    return inorder(node.left) + [node.val] + inorder(node.right)

# 後順（Post-order）: 左 → 右 → ルート
def postorder(node):
    if not node: return []
    return postorder(node.left) + postorder(node.right) + [node.val]

# レベル順（BFS）:
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
# 前順: [4,2,1,3,6,5,7]
# 中順: [1,2,3,4,5,6,7] ← ソート順！
# 後順: [1,3,2,5,7,6,4]
# レベル順: [4,2,6,1,3,5,7]
```

### 1.3 反復的走査（イテレーティブ）

再帰はスタックオーバーフローのリスクがあるため、深い木では反復的走査が必要になる。

```python
# 反復的中順走査（スタック使用）
def inorder_iterative(root):
    """O(n)時間、O(h)空間（hは木の高さ）"""
    result = []
    stack = []
    current = root

    while current or stack:
        # 左に進めるだけ進む
        while current:
            stack.append(current)
            current = current.left

        # 最も左のノードを処理
        current = stack.pop()
        result.append(current.val)

        # 右の部分木に移動
        current = current.right

    return result


# 反復的前順走査
def preorder_iterative(root):
    if not root:
        return []
    result = []
    stack = [root]

    while stack:
        node = stack.pop()
        result.append(node.val)
        # 右を先にスタックに入れる（左を先に処理するため）
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)

    return result


# 反復的後順走査（2つのスタック使用）
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


# Morris走査（O(1)空間の中順走査）
def morris_inorder(root):
    """スレッド化二分木を利用した O(1) 空間の走査"""
    result = []
    current = root

    while current:
        if current.left is None:
            result.append(current.val)
            current = current.right
        else:
            # 左部分木の最も右のノードを見つける
            predecessor = current.left
            while predecessor.right and predecessor.right != current:
                predecessor = predecessor.right

            if predecessor.right is None:
                # スレッドを作成
                predecessor.right = current
                current = current.left
            else:
                # スレッドを削除（元に戻す）
                predecessor.right = None
                result.append(current.val)
                current = current.right

    return result
```

### 1.4 レベル順走査の応用

```python
from collections import deque

# レベルごとにグループ化
def level_order_grouped(root):
    """各レベルのノードをリストのリストで返す"""
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
# → [[4], [2, 6], [1, 3, 5, 7]]


# ジグザグレベル順走査
def zigzag_level_order(root):
    """奇数レベルは右から左に走査"""
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

# → [[4], [6, 2], [1, 3, 5, 7]]


# 右側面ビュー
def right_side_view(root):
    """各レベルの最も右のノードを返す"""
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

# → [4, 6, 7]
```

---

## 2. 二分探索木（BST）

### 2.1 性質と操作

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

# BST性質: 左の全ノード < 親 < 右の全ノード
# 計算量: 平均 O(log n)、最悪 O(n)（偏った木）
```

### 2.2 BST の削除操作

BSTの削除は挿入・検索より複雑であり、3つのケースを扱う必要がある。

```python
class BST:
    # ... (上記の insert, search に加えて)

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
            # ケース1: 葉ノード（子なし）
            if not node.left and not node.right:
                return None

            # ケース2: 子が1つ
            if not node.left:
                return node.right
            if not node.right:
                return node.left

            # ケース3: 子が2つ
            # 右部分木の最小値（中順後継者）で置換
            successor = self._find_min(node.right)
            node.val = successor.val
            node.right = self._delete(node.right, successor.val)

        return node

    def _find_min(self, node):
        """部分木の最小値ノードを返す"""
        current = node
        while current.left:
            current = current.left
        return current

    def _find_max(self, node):
        """部分木の最大値ノードを返す"""
        current = node
        while current.right:
            current = current.right
        return current

    # BST のユーティリティ
    def kth_smallest(self, k):
        """k番目に小さい要素を返す（中順走査で k 番目）"""
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
        """BST の整合性を検証"""
        return self._is_valid(self.root, float('-inf'), float('inf'))

    def _is_valid(self, node, min_val, max_val):
        if not node:
            return True
        if node.val <= min_val or node.val >= max_val:
            return False
        return (self._is_valid(node.left, min_val, node.val) and
                self._is_valid(node.right, node.val, max_val))

    def lca(self, p, q):
        """最低共通祖先（Lowest Common Ancestor）"""
        node = self.root
        while node:
            if p < node.val and q < node.val:
                node = node.left
            elif p > node.val and q > node.val:
                node = node.right
            else:
                return node.val
        return None


# 使用例
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

### 2.3 平衡木の必要性

```
偏った木 vs 平衡木:

  ソート済みデータを挿入: 1, 2, 3, 4, 5
    1                 3
     \               / \
      2             2   4
       \           /     \
        3         1       5
         \
          4      平衡木 → O(log n)
           \
            5    偏った木 → O(n)

  平衡木の種類:
  - AVL木: 厳密に平衡（左右の高さの差 ≤ 1）
  - 赤黒木: ゆるく平衡（Java TreeMap, C++ std::map）
  - B木/B+木: ディスクI/O最適化（データベースインデックス）
  - 2-3木: 教育用（赤黒木の説明に使用）
```

---

## 3. AVL木

### 3.1 AVL木の基本

AVL木は最も古い自己平衡二分探索木であり、すべてのノードについて左右の部分木の高さの差（平衡係数）が-1、0、+1のいずれかであることを保証する。

```python
class AVLNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.height = 1  # 新規ノードの高さは1

class AVLTree:
    def __init__(self):
        self.root = None

    def height(self, node):
        return node.height if node else 0

    def balance_factor(self, node):
        """平衡係数 = 左の高さ - 右の高さ"""
        return self.height(node.left) - self.height(node.right) if node else 0

    def update_height(self, node):
        """ノードの高さを更新"""
        node.height = 1 + max(self.height(node.left), self.height(node.right))

    # 回転操作
    def right_rotate(self, y):
        """右回転"""
        #     y              x
        #    / \            / \
        #   x   T3   →    T1  y
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
        """左回転"""
        #   x                y
        #  / \              / \
        # T1  y     →     x   T3
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
        # 標準BST挿入
        if not node:
            return AVLNode(val)

        if val < node.val:
            node.left = self._insert(node.left, val)
        elif val > node.val:
            node.right = self._insert(node.right, val)
        else:
            return node  # 重複は無視

        # 高さ更新
        self.update_height(node)

        # 平衡係数チェック
        bf = self.balance_factor(node)

        # 4つの不均衡ケースに対する回転
        # LL: 左-左 → 右回転
        if bf > 1 and val < node.left.val:
            return self.right_rotate(node)

        # RR: 右-右 → 左回転
        if bf < -1 and val > node.right.val:
            return self.left_rotate(node)

        # LR: 左-右 → 左回転 + 右回転
        if bf > 1 and val > node.left.val:
            node.left = self.left_rotate(node.left)
            return self.right_rotate(node)

        # RL: 右-左 → 右回転 + 左回転
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
            # 子が0個または1個
            if not node.left:
                return node.right
            if not node.right:
                return node.left

            # 子が2個：中順後継者で置換
            successor = node.right
            while successor.left:
                successor = successor.left
            node.val = successor.val
            node.right = self._delete(node.right, successor.val)

        self.update_height(node)

        # 再平衡化
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


# AVL木の計算量:
# 検索: O(log n) 保証
# 挿入: O(log n)（最大2回の回転）
# 削除: O(log n)（最大O(log n)回の回転）
# 空間: O(n)

# 使用例
avl = AVLTree()
for val in [10, 20, 30, 40, 50, 25]:
    avl.insert(val)
# ソート済みデータを入れてもバランスが保たれる

print(avl.root.val)           # 30（ルート）
print(avl.balance_factor(avl.root))  # 0 or ±1
```

---

## 4. 赤黒木

### 4.1 赤黒木の性質

```
赤黒木の5つの性質:

  1. 各ノードは赤か黒
  2. ルートは黒
  3. すべての葉（NIL）は黒
  4. 赤ノードの子はすべて黒（赤が連続しない）
  5. 各ノードから子孫の葉までの全パスに含まれる黒ノード数が同じ
     （黒高さ: Black Height）

  これらの性質により:
  - 最長パス ≤ 2 × 最短パス
  - 高さ ≤ 2 × log2(n+1)
  - 検索/挿入/削除 すべて O(log n) 保証

  AVL木との比較:
  ┌──────────────┬──────────────┬──────────────┐
  │              │ AVL木        │ 赤黒木       │
  ├──────────────┼──────────────┼──────────────┤
  │ 平衡の厳密さ │ 厳密（±1）   │ ゆるい       │
  │ 検索速度     │ やや速い     │ やや遅い     │
  │ 挿入/削除    │ 回転が多い   │ 回転が少ない │
  │ メモリ       │ 高さ情報必要 │ 色1ビット    │
  │ 用途         │ 検索が多い   │ 挿入/削除多い│
  │ 実装例       │ -            │ Java TreeMap │
  │              │              │ C++ std::map │
  │              │              │ Linux rbtree │
  └──────────────┴──────────────┴──────────────┘
```

### 4.2 赤黒木の概念的な実装

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
        self.color = color  # 新規ノードは赤

class RedBlackTree:
    def __init__(self):
        self.NIL = RBNode(0, color=RBColor.BLACK)  # 番兵ノード
        self.root = self.NIL

    def insert(self, val):
        """BST挿入 + 赤黒木の修正"""
        new_node = RBNode(val)
        new_node.left = self.NIL
        new_node.right = self.NIL

        # BST挿入
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

        # 赤黒木の性質を修正
        self._fix_insert(new_node)

    def _fix_insert(self, node):
        """挿入後の修正（3つのケース）"""
        while node.parent and node.parent.color == RBColor.RED:
            if node.parent == node.parent.parent.left:
                uncle = node.parent.parent.right

                if uncle.color == RBColor.RED:
                    # ケース1: 叔父が赤 → 色の反転
                    node.parent.color = RBColor.BLACK
                    uncle.color = RBColor.BLACK
                    node.parent.parent.color = RBColor.RED
                    node = node.parent.parent
                else:
                    if node == node.parent.right:
                        # ケース2: 叔父が黒、ノードが右の子 → 左回転
                        node = node.parent
                        self._left_rotate(node)
                    # ケース3: 叔父が黒、ノードが左の子 → 右回転
                    node.parent.color = RBColor.BLACK
                    node.parent.parent.color = RBColor.RED
                    self._right_rotate(node.parent.parent)
            else:
                # 対称的なケース（左右を入れ替え）
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

        self.root.color = RBColor.BLACK  # 性質2: ルートは黒

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


# 赤黒木の利用先（言語・ライブラリ）:
# Java: TreeMap, TreeSet
# C++: std::map, std::set, std::multimap, std::multiset
# C#: SortedDictionary, SortedSet
# Linux: 完全公平スケジューラ（CFS）、メモリ管理
# Nginx: タイマー管理
```

---

## 5. B木とB+木

### 5.1 B木の基本

```
B木（B-Tree）: ディスクI/Oを最小化する多分岐探索木

  B木の性質（次数 t のB木）:
  - すべての葉は同じレベルにある
  - 各ノード（ルート以外）は t-1 以上 2t-1 以下のキーを持つ
  - ルートは1以上 2t-1 以下のキーを持つ
  - 各ノードの子の数 = キーの数 + 1
  - 各ノード内のキーはソート済み

  例: t=2 のB木（2-3-4木とも呼ばれる）
          [10, 20]
         /    |    \
    [3,5]  [12,15] [25,30,35]

  ディスクI/Oの最適化:
  - 1ノード = 1ディスクページ（通常 4KB-16KB）
  - ノードサイズをページサイズに合わせる
  - 高さが非常に低い → I/O回数が少ない

  例: t=1000（ノードあたり最大1999キー）の場合
  - 高さ1: 最大1999キー
  - 高さ2: 最大約400万キー
  - 高さ3: 最大約80億キー
  → 80億レコードのデータベースでも3回のI/Oで検索可能
```

### 5.2 B+木の特徴

```
B+木: B木の変種（データベースインデックスの標準）

  B木との違い:
  1. データは葉ノードにのみ格納（内部ノードはキーのみ）
  2. 葉ノードが連結リストで接続（範囲検索が高速）
  3. 内部ノードにデータがないため、より多くのキーを格納可能

  構造:
  内部ノード:  [  10  |  20  |  30  ]
              / |    |     |    \
  葉ノード: [3,5,8]→[10,12,15]→[20,22,25]→[30,35,40]

  利点:
  ┌──────────────────┬──────────────┬──────────────┐
  │ 操作              │ B木          │ B+木         │
  ├──────────────────┼──────────────┼──────────────┤
  │ 点検索            │ O(log_B n)   │ O(log_B n)   │
  │ 範囲検索          │ O(log_B n +k)│ O(log_B n +k)│
  │                   │ (非効率)     │ (葉の連結)   │
  │ フルスキャン      │ 全ノード走査 │ 葉のみ走査   │
  │ 内部ノードの容量  │ キー+データ  │ キーのみ     │
  └──────────────────┴──────────────┴──────────────┘

  実際の利用:
  - PostgreSQL: B+木インデックス（btree）
  - MySQL InnoDB: クラスタードインデックス（B+木）
  - SQLite: B+木ベースのページストレージ
  - ファイルシステム: NTFS, ext4, Btrfs
```

### 5.3 B+木の操作の概念

```python
class BPlusNode:
    def __init__(self, is_leaf=False):
        self.keys = []
        self.children = []  # 内部ノード: 子ノード、葉: データ
        self.is_leaf = is_leaf
        self.next = None    # 葉ノードの連結リスト

class BPlusTree:
    def __init__(self, order=4):
        """order: ノードあたりの最大子数"""
        self.order = order
        self.root = BPlusNode(is_leaf=True)

    def search(self, key):
        """キーを検索して対応する値を返す"""
        node = self.root

        # 内部ノードを下に辿る
        while not node.is_leaf:
            # key以上の最小のインデックスを見つける
            i = 0
            while i < len(node.keys) and key >= node.keys[i]:
                i += 1
            node = node.children[i]

        # 葉ノードで検索
        for i, k in enumerate(node.keys):
            if k == key:
                return node.children[i]  # 葉のchildrenはデータ

        return None  # 見つからない

    def range_search(self, start, end):
        """範囲検索: start <= key <= end"""
        # まず start の位置を見つける
        node = self.root
        while not node.is_leaf:
            i = 0
            while i < len(node.keys) and start >= node.keys[i]:
                i += 1
            node = node.children[i]

        # 葉ノードを連結リストで辿りながら結果を収集
        result = []
        while node:
            for i, k in enumerate(node.keys):
                if start <= k <= end:
                    result.append((k, node.children[i]))
                elif k > end:
                    return result
            node = node.next  # 次の葉ノードへ

        return result

# B+木のインデックスとしての利用
# CREATE INDEX idx_users_age ON users(age);
# → B+木が構築される
# SELECT * FROM users WHERE age BETWEEN 20 AND 30;
# → B+木の範囲検索 O(log n + k)（k は結果数）
```

---

## 6. Trie（トライ木）

### 6.1 基本構造と操作

```python
class TrieNode:
    def __init__(self):
        self.children = {}  # 文字 → TrieNode
        self.is_end = False  # 単語の終端
        self.count = 0       # この接頭辞を持つ単語数

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str):
        """単語を挿入: O(m)、m は単語の長さ"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.count += 1
        node.is_end = True

    def search(self, word: str) -> bool:
        """完全一致検索: O(m)"""
        node = self._find_node(word)
        return node is not None and node.is_end

    def starts_with(self, prefix: str) -> bool:
        """前方一致検索: O(m)"""
        return self._find_node(prefix) is not None

    def count_prefix(self, prefix: str) -> int:
        """接頭辞を持つ単語数: O(m)"""
        node = self._find_node(prefix)
        return node.count if node else 0

    def _find_node(self, prefix: str):
        """接頭辞に対応するノードを返す"""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node

    def autocomplete(self, prefix: str, limit: int = 10) -> list:
        """オートコンプリート: 接頭辞から候補を返す"""
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
        """単語の削除"""
        return self._delete(self.root, word, 0)

    def _delete(self, node, word, depth):
        if depth == len(word):
            if not node.is_end:
                return False
            node.is_end = False
            node.count -= 1
            return len(node.children) == 0  # 子がなければ削除可能

        char = word[depth]
        if char not in node.children:
            return False

        should_delete = self._delete(node.children[char], word, depth + 1)
        if should_delete:
            del node.children[char]
        node.count -= 1

        return len(node.children) == 0 and not node.is_end


# 使用例: オートコンプリート
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

### 6.2 圧縮Trie（Radix Tree / Patricia Tree）

```python
class RadixNode:
    def __init__(self, prefix=""):
        self.prefix = prefix
        self.children = {}
        self.is_end = False
        self.value = None

class RadixTree:
    """圧縮Trie: 共通接頭辞をエッジにまとめる"""

    def __init__(self):
        self.root = RadixNode()

    def insert(self, key, value=None):
        node = self.root
        remaining = key

        while remaining:
            # 共通接頭辞を持つ子を探す
            match_found = False
            for char, child in node.children.items():
                common = self._common_prefix(remaining, child.prefix)

                if not common:
                    continue

                match_found = True

                if common == child.prefix:
                    # 子のプレフィックスが完全に一致
                    remaining = remaining[len(common):]
                    node = child
                    break
                else:
                    # 部分一致 → ノードを分割
                    new_node = RadixNode(common)
                    child.prefix = child.prefix[len(common):]
                    new_node.children[child.prefix[0]] = child
                    node.children[common[0]] = new_node

                    remaining = remaining[len(common):]
                    node = new_node
                    break

            if not match_found:
                # 新しいエッジを追加
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


# Radix Tree の利点:
# - メモリ効率が良い（共通接頭辞を共有）
# - 長いキーの検索が高速
# 利用先:
# - Linux カーネルのルーティングテーブル
# - HTTP ルーター（gin, echo等のGoフレームワーク）
# - IPアドレスの最長一致検索

# 通常の Trie vs Radix Tree:
# Trie:        [r]-[o]-[m]-[u]-[l]-[u]-[s]
#                           [a]-[n]-[e]
#                           [a]-[n]-[c]-[e]
# Radix Tree:  [rom]-[ulus]
#                    -[an]-[e]
#                         -[ce]
```

---

## 7. ヒープと優先度キュー

### 7.1 二分ヒープ

```python
class MinHeap:
    """最小ヒープの実装"""

    def __init__(self):
        self.heap = []

    def parent(self, i):
        return (i - 1) // 2

    def left_child(self, i):
        return 2 * i + 1

    def right_child(self, i):
        return 2 * i + 2

    def push(self, val):
        """要素の追加: O(log n)"""
        self.heap.append(val)
        self._sift_up(len(self.heap) - 1)

    def pop(self):
        """最小要素の取り出し: O(log n)"""
        if not self.heap:
            raise IndexError("empty heap")
        if len(self.heap) == 1:
            return self.heap.pop()

        min_val = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._sift_down(0)
        return min_val

    def peek(self):
        """最小要素の参照: O(1)"""
        if not self.heap:
            raise IndexError("empty heap")
        return self.heap[0]

    def _sift_up(self, i):
        """ノードを上に移動（ヒープ性質の回復）"""
        while i > 0 and self.heap[i] < self.heap[self.parent(i)]:
            p = self.parent(i)
            self.heap[i], self.heap[p] = self.heap[p], self.heap[i]
            i = p

    def _sift_down(self, i):
        """ノードを下に移動（ヒープ性質の回復）"""
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


# Python の heapq モジュール（最小ヒープ）
import heapq

# 基本操作
h = []
heapq.heappush(h, 5)
heapq.heappush(h, 3)
heapq.heappush(h, 7)
heapq.heappush(h, 1)
print(heapq.heappop(h))  # 1（最小値）

# Top-K 問題
def top_k_largest(nums, k):
    """配列からk番目に大きい値を O(n log k) で取得"""
    return heapq.nlargest(k, nums)

def top_k_frequent(nums, k):
    """最も頻度が高いk個の要素"""
    from collections import Counter
    count = Counter(nums)
    return heapq.nlargest(k, count.keys(), key=count.get)

# ストリームの中央値
class MedianFinder:
    """2つのヒープで中央値を O(log n) で計算"""

    def __init__(self):
        self.lo = []   # 最大ヒープ（負の値で代用）
        self.hi = []   # 最小ヒープ

    def add_num(self, num):
        heapq.heappush(self.lo, -num)

        # lo の最大 ≤ hi の最小 を保証
        heapq.heappush(self.hi, -heapq.heappop(self.lo))

        # サイズバランス: lo のサイズ ≥ hi のサイズ
        if len(self.lo) < len(self.hi):
            heapq.heappush(self.lo, -heapq.heappop(self.hi))

    def find_median(self):
        if len(self.lo) > len(self.hi):
            return -self.lo[0]
        return (-self.lo[0] + self.hi[0]) / 2

# 使用例
mf = MedianFinder()
mf.add_num(1)
mf.add_num(2)
print(mf.find_median())  # 1.5
mf.add_num(3)
print(mf.find_median())  # 2.0
```

---

## 8. 木構造のアルゴリズム問題

### 8.1 再帰的アプローチ

```python
# 木の最大深さ
def max_depth(root):
    if not root:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))

# 木の直径（最長パス）
def diameter(root):
    """任意の2ノード間の最長パス長"""
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

# 木の反転（ミラー）
def invert_tree(root):
    if not root:
        return None
    root.left, root.right = invert_tree(root.right), invert_tree(root.left)
    return root

# パスの合計
def has_path_sum(root, target_sum):
    """ルートから葉までのパスの合計が target_sum のパスが存在するか"""
    if not root:
        return False
    if not root.left and not root.right:
        return root.val == target_sum
    return (has_path_sum(root.left, target_sum - root.val) or
            has_path_sum(root.right, target_sum - root.val))

# すべてのパスの合計を列挙
def path_sum_all(root, target_sum):
    """条件を満たすすべてのパスを返す"""
    result = []

    def dfs(node, remaining, path):
        if not node:
            return
        path.append(node.val)
        if not node.left and not node.right and remaining == node.val:
            result.append(path[:])  # コピーを追加
        dfs(node.left, remaining - node.val, path)
        dfs(node.right, remaining - node.val, path)
        path.pop()  # バックトラック

    dfs(root, target_sum, [])
    return result

# 同一の木の判定
def is_same_tree(p, q):
    if not p and not q:
        return True
    if not p or not q:
        return False
    return (p.val == q.val and
            is_same_tree(p.left, q.left) and
            is_same_tree(p.right, q.right))

# 部分木の判定
def is_subtree(root, sub_root):
    """sub_root が root の部分木かどうか"""
    if not root:
        return False
    if is_same_tree(root, sub_root):
        return True
    return is_subtree(root.left, sub_root) or is_subtree(root.right, sub_root)

# 対称木の判定
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

### 8.2 木の構築

```python
# 前順 + 中順 → 木の構築
def build_tree_from_preorder_inorder(preorder, inorder):
    """前順走査と中順走査から木を復元"""
    if not preorder or not inorder:
        return None

    # 前順の最初の要素がルート
    root_val = preorder[0]
    root = TreeNode(root_val)

    # 中順でルートの位置を見つける
    mid = inorder.index(root_val)

    # 左右の部分木を再帰的に構築
    root.left = build_tree_from_preorder_inorder(
        preorder[1:mid+1], inorder[:mid]
    )
    root.right = build_tree_from_preorder_inorder(
        preorder[mid+1:], inorder[mid+1:]
    )

    return root

# ソート済み配列 → BST
def sorted_array_to_bst(nums):
    """ソート済み配列から高さバランスのBSTを構築"""
    if not nums:
        return None

    mid = len(nums) // 2
    root = TreeNode(nums[mid])
    root.left = sorted_array_to_bst(nums[:mid])
    root.right = sorted_array_to_bst(nums[mid+1:])

    return root

# 使用例
nums = [1, 2, 3, 4, 5, 6, 7]
root = sorted_array_to_bst(nums)
# 結果:
#       4
#      / \
#     2   6
#    / \ / \
#   1  3 5  7


# 木のシリアライズ/デシリアライズ
import json

def serialize(root):
    """木をJSON文字列にシリアライズ"""
    if not root:
        return "null"
    return json.dumps({
        "val": root.val,
        "left": json.loads(serialize(root.left)),
        "right": json.loads(serialize(root.right))
    })

def deserialize(data):
    """JSON文字列から木をデシリアライズ"""
    if data == "null":
        return None
    obj = json.loads(data)
    if obj is None:
        return None
    root = TreeNode(obj["val"])
    root.left = deserialize(json.dumps(obj["left"]))
    root.right = deserialize(json.dumps(obj["right"]))
    return root

# BFS方式のシリアライズ（LeetCode形式）
def serialize_bfs(root):
    """BFS方式でシリアライズ: [4,2,6,1,3,5,7]"""
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

    # 末尾のNoneを除去
    while result and result[-1] is None:
        result.pop()

    return str(result)
```

---

## 9. 実務での木構造

### 9.1 実世界での利用例

```
木が使われる実世界の例:

  1. ファイルシステム: ディレクトリ = 木構造
     /
     ├── home/
     │   ├── user/
     │   │   ├── documents/
     │   │   └── downloads/
     │   └── admin/
     └── etc/
         ├── nginx/
         └── ssh/

  2. DOM: HTML要素の親子関係
     html
     ├── head
     │   ├── title
     │   └── meta
     └── body
         ├── div#header
         └── div#content

  3. AST: ソースコードの構文木（コンパイラ）
     x = 3 + 5 * 2
         =
        / \
       x   +
          / \
         3   *
            / \
           5   2

  4. B+木: データベースインデックス（PostgreSQL, MySQL）

  5. JSON/XML: ネストしたデータ構造

  6. 決定木: 機械学習の分類器

  7. ハフマン木: データ圧縮

  8. Trie: 辞書、オートコンプリート

  9. React Virtual DOM: 差分検出アルゴリズム（Reconciliation）

  10. Git: コミットツリー、マークルツリー
```

### 9.2 木構造を使ったシステム設計

```python
# 1. ファイルシステムの木構造
class FileNode:
    def __init__(self, name, is_dir=False):
        self.name = name
        self.is_dir = is_dir
        self.children = {}  # name -> FileNode
        self.content = ""   # ファイルの場合
        self.size = 0

class FileSystem:
    def __init__(self):
        self.root = FileNode("/", is_dir=True)

    def mkdir(self, path):
        """ディレクトリの作成"""
        parts = path.strip("/").split("/")
        node = self.root
        for part in parts:
            if part not in node.children:
                node.children[part] = FileNode(part, is_dir=True)
            node = node.children[part]

    def write(self, path, content):
        """ファイルの書き込み"""
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
        """ファイルの読み取り"""
        parts = path.strip("/").split("/")
        node = self.root
        for part in parts:
            if part not in node.children:
                return None
            node = node.children[part]
        return node.content if not node.is_dir else None

    def ls(self, path="/"):
        """ディレクトリの内容一覧"""
        parts = path.strip("/").split("/")
        node = self.root
        if path != "/":
            for part in parts:
                if part and part in node.children:
                    node = node.children[part]
        return sorted(node.children.keys())

    def du(self, path="/"):
        """ディスク使用量の計算（再帰）"""
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

# 使用例
fs = FileSystem()
fs.mkdir("/home/user/documents")
fs.write("/home/user/documents/readme.txt", "Hello World")
fs.write("/home/user/documents/notes.txt", "Important notes here")
print(fs.ls("/home/user/documents"))  # ['notes.txt', 'readme.txt']
print(fs.read("/home/user/documents/readme.txt"))  # "Hello World"
print(fs.du("/home/user"))  # 31


# 2. 組織階層の管理
class OrgNode:
    def __init__(self, name, title):
        self.name = name
        self.title = title
        self.reports = []  # 直属の部下

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
        """指揮系統（ルートまでのパス）を返す"""
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
        """直接・間接の部下の総数"""
        node = self.lookup.get(name)
        if not node:
            return 0
        return self._count(node) - 1  # 自分を除く

    def _count(self, node):
        return 1 + sum(self._count(r) for r in node.reports)
```

---

## 10. 実践演習

### 演習1: BST操作（基礎）
BSTの挿入・検索・削除・中順走査を実装せよ。さらに以下の操作も実装すること:
- k番目に小さい要素の取得
- 指定範囲のノードの列挙
- BST の妥当性検証

### 演習2: 木の再帰（応用）
以下の操作を再帰で実装せよ:
- 木の最大深さ、直径、左右反転
- 2つの木が同一かの判定
- ルートから葉へのパスの合計が target_sum になるパスの列挙
- 最低共通祖先（LCA）の計算

### 演習3: 直列化（発展）
木をJSON文字列にシリアライズし、デシリアライズする関数を実装せよ。BFS方式とDFS方式の両方を実装すること。

### 演習4: AVL木の実装（応用）
挿入・削除・検索をすべてサポートするAVL木を実装せよ。以下を検証すること:
- ソート済みデータの挿入後もバランスが保たれること
- ランダムデータと比較して高さが O(log n) であること

### 演習5: Trie によるオートコンプリート（応用）
英単語辞書のオートコンプリートシステムを Trie で実装せよ:
- 単語の挿入と検索
- 接頭辞による候補の列挙（頻度順）
- 削除のサポート

### 演習6: ハフマン符号化（発展）
ハフマン木を構築してテキストの圧縮・展開を実装せよ:
- 文字の出現頻度からハフマン木を構築
- テキストをビット列にエンコード
- ビット列からテキストにデコード
- 圧縮率の計算と表示

---

## まとめ

| データ構造 | 操作 | 用途 |
|-----------|------|------|
| 二分木 | 走査 O(n) | 式の評価、構文木 |
| BST | 検索/挿入 O(log n)平均 | ソート済みデータの管理 |
| AVL木 | 検索/挿入 O(log n)保証 | 検索が多い場面 |
| 赤黒木 | 検索/挿入 O(log n)保証 | TreeMap, TreeSet, CFS |
| B+木 | 検索 O(log_B n) | DBインデックス、ファイルシステム |
| Trie | 検索 O(m) | 辞書、前置一致、オートコンプリート |
| ヒープ | 最小/最大取得 O(1) | 優先度キュー、Top-K |
| セグメント木 | 区間クエリ O(log n) | 範囲最小値、範囲和 |

---

## 次に読むべきガイド
→ [[05-graphs.md]] -- グラフ

---

## 参考文献
1. Cormen, T. H. "Introduction to Algorithms." Chapters 12-13, 18.
2. Sedgewick, R. "Algorithms." Chapter 3.2-3.3.
3. Knuth, D. E. "The Art of Computer Programming." Volume 3: Sorting and Searching.
4. Bayer, R., McCreight, E. "Organization and Maintenance of Large Ordered Indexes." 1972.
5. Fredkin, E. "Trie Memory." Communications of the ACM, 1960.
6. Guibas, L. J., Sedgewick, R. "A Dichromatic Framework for Balanced Trees." 1978.
7. Adelson-Velsky, G. M., Landis, E. M. "An Algorithm for the Organization of Information." 1962.
