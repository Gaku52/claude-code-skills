# Union-Find（素集合データ構造）

> 要素のグループ化と所属判定を準定数時間で行うデータ構造を、経路圧縮・ランク付き合併・Kruskal応用を通じて理解する

## この章で学ぶこと

1. **Union-Find の2操作**（Find/Union）の原理と、経路圧縮・ランク付き合併による最適化を実装できる
2. **逆アッカーマン関数**α(n) による準定数時間の計算量を理解する
3. **Kruskal法・連結成分・等価クラス分け**など実用的な応用パターンを使いこなせる

---

## 1. Union-Find の概念

```
Union-Find = 素集合の森（Disjoint Set Forest）

初期状態（各要素が独立）:
  {0} {1} {2} {3} {4} {5} {6} {7}

Union(0,1), Union(2,3), Union(4,5):
  {0,1} {2,3} {4,5} {6} {7}

Union(0,2), Union(4,6):
  {0,1,2,3} {4,5,6} {7}

Union(0,4):
  {0,1,2,3,4,5,6} {7}

Find(5) → 0（代表元）
Find(7) → 7（代表元）
Find(5) == Find(3) → True（同じ集合）
Find(5) == Find(7) → False（異なる集合）
```

---

## 2. 木構造による表現

```
ナイーブな実装（木が偏る）:

Union(0,1): 1→0      Union(0,1,2,3,4):
Union(0,2): 2→0          0
Union(0,3): 3→0         /|\ \
Union(0,4): 4→0        1 2 3 4    ← バランスが良い

最悪ケース（チェーン状）:
  Union(0,1), Union(1,2), Union(2,3), Union(3,4)
    0 ← 1 ← 2 ← 3 ← 4    ← Find(4) が O(n)!

経路圧縮後:
         0
       / | \ \
      1  2  3  4    ← Find(4) が O(1)!
```

---

## 3. 基本実装

```python
class UnionFind:
    """Union-Find（素集合データ構造）
    経路圧縮 + ランク付き合併で O(α(n)) ≈ O(1)
    """

    def __init__(self, n: int):
        self.parent = list(range(n))  # 各要素の親
        self.rank = [0] * n           # 木の高さの上界
        self.size = [1] * n           # 各集合のサイズ
        self.count = n                # 集合の数

    def find(self, x: int) -> int:
        """x の代表元（根）を返す - 経路圧縮付き"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 経路圧縮
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """x と y の集合を合併 - ランク付き合併
        返り値: 合併が行われたか（既に同じ集合ならFalse）
        """
        px, py = self.find(x), self.find(y)
        if px == py:
            return False

        # ランクが低い方を高い方の子にする
        if self.rank[px] < self.rank[py]:
            px, py = py, px

        self.parent[py] = px
        self.size[px] += self.size[py]

        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

        self.count -= 1
        return True

    def connected(self, x: int, y: int) -> bool:
        """x と y が同じ集合に属するか"""
        return self.find(x) == self.find(y)

    def get_size(self, x: int) -> int:
        """x が属する集合のサイズ"""
        return self.size[self.find(x)]

# 使用例
uf = UnionFind(8)
uf.union(0, 1)
uf.union(2, 3)
uf.union(0, 2)
print(uf.connected(1, 3))  # True
print(uf.connected(0, 5))  # False
print(uf.get_size(0))      # 4
print(uf.count)             # 5 (集合数)
```

---

## 4. 経路圧縮の詳細

```
Find(7) の経路圧縮:

Before:           After:
    0                 0
    |              / | \ \
    1             1  3  5  7
    |
    3
    |
    5
    |
    7

Find(7): 7→5→3→1→0  (根を発見)
圧縮:   7→0, 5→0, 3→0, 1→0  (全ノードを根の直下に)
```

```python
# 経路圧縮の2つの方法

# 方法1: 再帰（上記の実装）
def find_recursive(self, x):
    if self.parent[x] != x:
        self.parent[x] = self.find(self.parent[x])
    return self.parent[x]

# 方法2: 反復（スタックオーバーフロー回避）
def find_iterative(self, x):
    root = x
    while self.parent[root] != root:
        root = self.parent[root]
    # 経路上の全ノードを根に直接接続
    while self.parent[x] != root:
        next_x = self.parent[x]
        self.parent[x] = root
        x = next_x
    return root

# 方法3: パス分割（Path Splitting）- 実装が簡潔
def find_splitting(self, x):
    while self.parent[x] != x:
        self.parent[x] = self.parent[self.parent[x]]  # 祖父に接続
        x = self.parent[x]
    return x
```

---

## 5. Kruskal法への応用

Union-Find は Kruskal の最小全域木アルゴリズムで不可欠。

```python
def kruskal_mst(n: int, edges: list) -> tuple:
    """Kruskal法 - Union-Find使用 - O(E log E)
    edges: [(weight, u, v), ...]
    """
    edges.sort()  # 重みでソート
    uf = UnionFind(n)
    mst_edges = []
    mst_weight = 0

    for weight, u, v in edges:
        if not uf.connected(u, v):  # サイクル判定
            uf.union(u, v)
            mst_edges.append((u, v, weight))
            mst_weight += weight

            if len(mst_edges) == n - 1:
                break  # MST完成

    return mst_edges, mst_weight

# 使用例
# 頂点: 0-4, 辺: (重み, 始点, 終点)
edges = [
    (1, 0, 1), (4, 0, 2), (3, 1, 2),
    (2, 1, 3), (5, 2, 3), (7, 2, 4), (6, 3, 4)
]
mst, total = kruskal_mst(5, edges)
print(f"MST辺: {mst}")        # [(0, 1, 1), (1, 3, 2), (1, 2, 3), (3, 4, 6)]
print(f"合計重み: {total}")    # 12
```

---

## 6. 応用パターン

### 連結成分の数（グラフ）

```python
def count_components(n: int, edges: list) -> int:
    """グラフの連結成分数"""
    uf = UnionFind(n)
    for u, v in edges:
        uf.union(u, v)
    return uf.count

print(count_components(5, [(0,1), (2,3)]))  # 3 ({0,1}, {2,3}, {4})
```

### 友達の友達（SNSグラフ）

```python
def friend_groups(n: int, friendships: list) -> list:
    """友達グループの一覧を返す"""
    uf = UnionFind(n)
    for a, b in friendships:
        uf.union(a, b)

    groups = {}
    for i in range(n):
        root = uf.find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    return list(groups.values())

friends = [(0,1), (1,2), (3,4)]
print(friend_groups(6, friends))
# [[0, 1, 2], [3, 4], [5]]
```

### 島の数（グリッド問題）

```python
def num_islands(grid: list) -> int:
    """2Dグリッドの島の数をUnion-Findで求める"""
    if not grid:
        return 0

    rows, cols = len(grid), len(grid[0])
    uf = UnionFind(rows * cols)
    water = 0

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '0':
                water += 1
                continue

            # 右と下の隣接セルとUnion
            for dr, dc in [(0, 1), (1, 0)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < rows and 0 <= nc < cols
                    and grid[nr][nc] == '1'):
                    uf.union(r * cols + c, nr * cols + nc)

    return uf.count - water

grid = [
    ['1','1','0','0','0'],
    ['1','1','0','0','0'],
    ['0','0','1','0','0'],
    ['0','0','0','1','1'],
]
print(num_islands(grid))  # 3
```

---

## 7. 最適化の効果比較表

| 最適化 | Find | Union | 備考 |
|:---|:---|:---|:---|
| ナイーブ（配列） | O(1) | O(n) | Quick-Find |
| ナイーブ（木） | O(n) | O(n) | 最悪ケース |
| 経路圧縮のみ | 償却 O(log n) | O(log n) | 大幅改善 |
| ランク付き合併のみ | O(log n) | O(log n) | 木の高さ制限 |
| 両方（最適） | 償却 O(α(n)) | 償却 O(α(n)) | 実質 O(1) |

## Union-Find vs 他の手法

| 手法 | 連結判定 | 合併 | 全成分列挙 | 用途 |
|:---|:---|:---|:---|:---|
| Union-Find | O(α(n)) | O(α(n)) | O(n) | 動的連結性 |
| BFS/DFS | O(V+E) | - | O(V+E) | 静的グラフ |
| 隣接行列 | O(1) | O(n) | O(n²) | 密グラフ |

---

## 8. 重み付き Union-Find

```python
class WeightedUnionFind:
    """重み付きUnion-Find
    要素間の相対的な重み（距離/差分）を管理
    """
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.weight = [0] * n  # 親との重みの差

    def find(self, x):
        if self.parent[x] != x:
            root = self.find(self.parent[x])
            self.weight[x] += self.weight[self.parent[x]]
            self.parent[x] = root
        return self.parent[x]

    def get_weight(self, x):
        """根からxまでの累積重み"""
        self.find(x)  # 経路圧縮
        return self.weight[x]

    def diff(self, x, y):
        """weight(y) - weight(x)"""
        if self.find(x) != self.find(y):
            return None  # 異なる集合
        return self.get_weight(y) - self.get_weight(x)

    def union(self, x, y, w):
        """weight(y) - weight(x) = w という関係を追加"""
        px, py = self.find(x), self.find(y)
        if px == py:
            return  # 既に同じ集合

        w = w + self.weight[x] - self.weight[y]

        if self.rank[px] < self.rank[py]:
            self.parent[px] = py
            self.weight[px] = -w
        else:
            self.parent[py] = px
            self.weight[py] = w
            if self.rank[px] == self.rank[py]:
                self.rank[px] += 1

# 使用例: A - B = 3, B - C = 5 → A - C = 8
wuf = WeightedUnionFind(3)
wuf.union(0, 1, 3)  # weight[1] - weight[0] = 3
wuf.union(1, 2, 5)  # weight[2] - weight[1] = 5
print(wuf.diff(0, 2))  # 8 (weight[2] - weight[0] = 3+5)
```

---

## 9. アンチパターン

### アンチパターン1: 経路圧縮・ランク付き合併を省略

```python
# BAD: ナイーブなUnion-Find → Find が O(n) に退化
class BadUnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            x = self.parent[x]  # 経路圧縮なし!
        return x

    def union(self, x, y):
        self.parent[self.find(x)] = self.find(y)  # ランク考慮なし!

# GOOD: 両方の最適化を適用
class GoodUnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 経路圧縮
        return self.parent[x]
    # ... ランク付き合併も実装
```

### アンチパターン2: 毎回 O(n) で連結判定

```python
# BAD: 辺の追加のたびに BFS/DFS で連結判定 → O(E * (V+E))
for u, v in edges:
    graph[u].append(v)
    if bfs_connected(graph, 0, target):  # O(V+E) が毎回
        ...

# GOOD: Union-Find で O(E * α(n)) ≈ O(E)
uf = UnionFind(n)
for u, v in edges:
    uf.union(u, v)
    if uf.connected(0, target):  # O(α(n)) ≈ O(1)
        ...
```

---

## 10. FAQ

### Q1: 逆アッカーマン関数 α(n) とは何か？

**A:** α(n) はアッカーマン関数の逆関数で、実用上の全ての入力（宇宙の原子数 ~10^80 まで）に対して 5 以下。つまり O(α(n)) は実質的に O(1)。経路圧縮とランク付き合併を併用した場合に得られる計算量。1975年にTarjanが証明した。

### Q2: Union-Find で要素の削除はできるか？

**A:** 標準的な Union-Find では要素の削除やグループの分割（Split）は効率的にできない。必要な場合は (1) 削除済みフラグで論理削除し新しい要素で代替、(2) Link-Cut Tree などの高度なデータ構造を使う、(3) 全体を再構築する、のいずれかを検討する。

### Q3: Union-Find はオンライン問題に強いか？

**A:** はい。辺が動的に追加される状況（オンラインクエリ）で連結性を管理するのに最適。BFS/DFS は辺が追加されるたびに再計算が必要だが、Union-Find は O(α(n)) で逐次的に処理できる。

---

## 11. まとめ

| 項目 | 要点 |
|:---|:---|
| 基本操作 | Find（代表元の取得）と Union（集合の合併） |
| 経路圧縮 | Find 時に全ノードを根に直接接続 → 木を平坦化 |
| ランク付き合併 | 低い木を高い木の子に → 木の高さを制限 |
| 計算量 | 両方の最適化で O(α(n)) ≈ 実質 O(1) |
| Kruskal応用 | サイクル判定に使用。O(E log E) で MST |
| 重み付き拡張 | 要素間の相対重みを管理 |

---

## 次に読むべきガイド

- [グラフ走査](../02-algorithms/02-graph-traversal.md) -- 連結成分の別アプローチ（BFS/DFS）
- [貪欲法](../02-algorithms/05-greedy.md) -- Kruskal法の詳細
- [セグメント木](./01-segment-tree.md) -- 別の高度データ構造

---

## 参考文献

1. Cormen, T. H. et al. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. -- 第19章
2. Tarjan, R. E. (1975). "Efficiency of a Good But Not Linear Set Union Algorithm." *JACM*.
3. Tarjan, R. E. & van Leeuwen, J. (1984). "Worst-case Analysis of Set Union Algorithms." *JACM*.
4. Sedgewick, R. & Wayne, K. (2011). *Algorithms* (4th ed.). Addison-Wesley. -- 1.5 Union-Find
