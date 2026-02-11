# 木構造 — 二分木・BST・AVL/赤黒木・B木・Trie

> 階層的なデータを表現する木構造の各種バリエーションを学ぶ。二分探索木の基本から平衡木、B木、Trie まで体系的に解説する。

---

## この章で学ぶこと

1. **二分木と走査** — 前順・中順・後順・レベル順
2. **二分探索木（BST）** と平衡木（AVL・赤黒木）の原理
3. **B木と Trie** — ディスクアクセス最適化と文字列検索

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
  - 深さ (depth): 根からの距離
  - 高さ (height): 葉までの最大距離（木の高さ = 2）
  - 部分木 (subtree): 任意のノードを根とする木
```

---

## 2. 二分木の走査

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def preorder(node):
    """前順走査 (根→左→右) — O(n)"""
    if not node:
        return []
    return [node.val] + preorder(node.left) + preorder(node.right)

def inorder(node):
    """中順走査 (左→根→右) — O(n)"""
    if not node:
        return []
    return inorder(node.left) + [node.val] + inorder(node.right)

def postorder(node):
    """後順走査 (左→右→根) — O(n)"""
    if not node:
        return []
    return postorder(node.left) + postorder(node.right) + [node.val]

def levelorder(root):
    """レベル順走査 (BFS) — O(n)"""
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

```
走査順序の例:
       [4]
      /   \
    [2]   [6]
   / \   / \
  [1] [3] [5] [7]

  前順 (Pre):  4, 2, 1, 3, 6, 5, 7
  中順 (In):   1, 2, 3, 4, 5, 6, 7  ← BST ではソート順
  後順 (Post): 1, 3, 2, 5, 7, 6, 4
  レベル順:    4, 2, 6, 1, 3, 5, 7
```

---

## 3. 二分探索木（BST）

```python
class BST:
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
        """O(h)"""
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
        return node
```

---

## 4. AVL 木（自己平衡BST）

```
AVL木: 全ノードで |左の高さ - 右の高さ| ≤ 1

  不均衡検出 → 回転で修正:

  右回転 (LL):          左回転 (RR):
      z                    x
     / \                  / \
    y   T4     →        T1   z
   / \                      / \
  x   T3                  T3  T4
 / \
T1  T2

  左右回転 (LR):        右左回転 (RL):
      z                    z
     / \                  / \
    y   T4    →         T1   y
   / \                      / \
  T1  x                    x  T4
     / \                  / \
    T2  T3               T2  T3
```

```python
class AVLNode:
    def __init__(self, val):
        self.val = val
        self.left = self.right = None
        self.height = 1

def height(node):
    return node.height if node else 0

def balance_factor(node):
    return height(node.left) - height(node.right) if node else 0

def rotate_right(z):
    y = z.left
    T3 = y.right
    y.right = z
    z.left = T3
    z.height = 1 + max(height(z.left), height(z.right))
    y.height = 1 + max(height(y.left), height(y.right))
    return y

def rotate_left(z):
    y = z.right
    T2 = y.left
    y.left = z
    z.right = T2
    z.height = 1 + max(height(z.left), height(z.right))
    y.height = 1 + max(height(y.left), height(y.right))
    return y
```

---

## 5. Trie（接頭辞木）

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

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        """O(m), m = 単語の長さ"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def search(self, word):
        """O(m)"""
        node = self._find(word)
        return node is not None and node.is_end

    def starts_with(self, prefix):
        """O(m)"""
        return self._find(prefix) is not None

    def _find(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node
```

---

## 6. 比較表

### 表1: 木構造の操作計算量

| 木の種類 | 探索 | 挿入 | 削除 | 空間 |
|---------|------|------|------|------|
| BST（平均） | O(log n) | O(log n) | O(log n) | O(n) |
| BST（最悪） | O(n) | O(n) | O(n) | O(n) |
| AVL 木 | O(log n) | O(log n) | O(log n) | O(n) |
| 赤黒木 | O(log n) | O(log n) | O(log n) | O(n) |
| B 木 (order m) | O(log n) | O(log n) | O(log n) | O(n) |
| Trie | O(m) | O(m) | O(m) | O(ΣLEN) |

### 表2: 平衡木の比較

| 特性 | AVL 木 | 赤黒木 | B 木 |
|------|--------|--------|------|
| 平衡条件 | 高さ差 ≤ 1 | 色条件 | ノード充填率 |
| 探索速度 | 速い | やや遅い | ディスク最適 |
| 挿入/削除 | 回転多い | 回転少ない | 分割/併合 |
| 主用途 | メモリ内検索 | 汎用（TreeMap） | DB インデックス |
| 高さ | ≤ 1.44 log n | ≤ 2 log n | ≤ log_t n |

---

## 7. アンチパターン

### アンチパターン1: ソート済みデータをBSTに挿入

```python
# BAD: ソート済みデータ → 偏った木 → O(n)
bst = BST()
for i in range(1, 8):
    bst.insert(i)

# 結果: 1 → 2 → 3 → 4 → 5 → 6 → 7（連結リストと同じ）
# 探索: O(n)

# GOOD: 平衡木（AVL/赤黒木）を使うか、ランダムシャッフルしてから挿入
import random
data = list(range(1, 8))
random.shuffle(data)
for i in data:
    bst.insert(i)
```

### アンチパターン2: Trie で不要なメモリ消費

```python
# BAD: 26文字分の配列を毎ノードに確保
class BadTrieNode:
    def __init__(self):
        self.children = [None] * 26  # 26 × 8byte = 208byte/node
        self.is_end = False

# GOOD: 辞書で必要な文字のみ
class GoodTrieNode:
    def __init__(self):
        self.children = {}  # 必要な文字だけ
        self.is_end = False
```

---

## 8. FAQ

### Q1: AVL 木と赤黒木はどちらを使うべきか？

**A:** 探索が多いなら AVL 木（木の高さが低い）、挿入・削除が多いなら赤黒木（回転が少ない）。実務では多くの言語の標準ライブラリが赤黒木を採用（Java TreeMap、C++ std::map）。

### Q2: B木はなぜデータベースに使われるか？

**A:** B木はノードが大きく（数百〜数千のキー）、ディスクブロックサイズに合わせられる。木の高さが非常に低い（3〜4レベルで数百万件を格納）ため、ディスクI/O回数が最小になる。

### Q3: Trie のメモリ消費を減らすには？

**A:** 圧縮 Trie（Patricia Trie / Radix Tree）で連鎖するノードを1つにまとめる。また、配列の代わりに辞書を使えばスパースなアルファベットでも効率的。

---

## 9. まとめ

| 項目 | ポイント |
|------|---------|
| 二分木走査 | 前順/中順/後順/レベル順の4種 |
| BST | 中順走査がソート順。最悪 O(n) |
| AVL 木 | 高さ差 ≤ 1。探索に最適 |
| 赤黒木 | 回転が少ない。汎用的 |
| B 木 | ディスク I/O 最適化。DB のインデックス |
| Trie | 接頭辞検索に O(m)。オートコンプリート |

---

## 次に読むべきガイド

- [ヒープ — 二分ヒープとヒープソート](./05-heaps.md)
- [セグメント木 — 区間クエリと遅延伝播](../03-advanced/01-segment-tree.md)

---

## 参考文献

1. Cormen, T.H. et al. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. — 第12-13章「Binary Search Trees」「Red-Black Trees」
2. Bayer, R. & McCreight, E. (1972). "Organization and maintenance of large ordered indexes." *Acta Informatica*, 1(3), 173-189.
3. Sedgewick, R. & Wayne, K. (2011). *Algorithms* (4th ed.). Addison-Wesley. — Trie の実装
