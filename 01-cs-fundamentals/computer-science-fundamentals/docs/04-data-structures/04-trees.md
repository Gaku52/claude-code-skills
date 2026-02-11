# 木構造

> 木は階層関係を表現する最も自然なデータ構造であり、ファイルシステム、DOM、データベースインデックスの基盤である。

## この章で学ぶこと

- [ ] 木の基本用語と走査方法を理解する
- [ ] 二分探索木（BST）の操作と計算量を説明できる
- [ ] 平衡木（AVL木、赤黒木）の必要性を理解する

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

### 2.2 平衡木の必要性

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

## 3. 実務での木構造

```
木が使われる実世界の例:

  1. ファイルシステム: ディレクトリ = 木構造
  2. DOM: HTML要素の親子関係
  3. AST: ソースコードの構文木（コンパイラ）
  4. B+木: データベースインデックス（PostgreSQL, MySQL）
  5. JSON/XML: ネストしたデータ構造
  6. 決定木: 機械学習の分類器
  7. ハフマン木: データ圧縮
  8. Trie: 辞書、オートコンプリート
```

---

## 4. 実践演習

### 演習1: BST操作（基礎）
BSTの挿入・検索・削除・中順走査を実装せよ。

### 演習2: 木の再帰（応用）
木の最大深さ、左右反転、2つの木が同一かの判定を再帰で実装せよ。

### 演習3: 直列化（発展）
木をJSON文字列にシリアライズし、デシリアライズする関数を実装せよ。

---

## まとめ

| データ構造 | 操作 | 用途 |
|-----------|------|------|
| 二分木 | 走査 O(n) | 式の評価、構文木 |
| BST | 検索/挿入 O(log n)平均 | ソート済みデータの管理 |
| AVL/赤黒木 | 検索/挿入 O(log n)保証 | TreeMap, TreeSet |
| B+木 | 検索 O(log_B n) | DBインデックス |
| Trie | 検索 O(m) | 辞書、前置一致 |

---

## 次に読むべきガイド
→ [[05-graphs.md]] — グラフ

---

## 参考文献
1. Cormen, T. H. "Introduction to Algorithms." Chapters 12-13.
2. Sedgewick, R. "Algorithms." Chapter 3.2-3.3.
