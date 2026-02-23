# グラフ — 表現方法・隣接リスト/行列・重み付きグラフ

> ネットワーク、依存関係、地図など多様な関係を表現するグラフの基本概念と、各表現方法の特徴を学ぶ。

---

## この章で学ぶこと

1. **グラフの基本用語** — 頂点、辺、有向/無向、重み
2. **隣接リストと隣接行列** の実装と使い分け
3. **重み付きグラフ** と特殊なグラフの表現
4. **Union-Find** — 素集合データ構造
5. **実務応用** — ソーシャルグラフ、依存関係解決、経路探索

---

## 1. グラフの基本概念

### 1.1 グラフの種類

```
無向グラフ:                  有向グラフ:
  A --- B                    A → B
  |   / |                    ↑   ↓
  |  /  |                    D ← C
  | /   |
  C --- D

重み付きグラフ:              DAG (有向非巡回グラフ):
  A -5- B                    A → B → D
  |     |                    ↓   ↓
  3     2                    C → E
  |     |
  C -1- D

完全グラフ K4:               二部グラフ:
  A --- B                    L1 --- R1
  |\ /|                     |  \ / |
  | X  |                     |   X  |
  |/ \ |                     |  / \ |
  C --- D                    L2 --- R2

多重グラフ:                  自己ループ:
  A ==== B                    A ⟲
  (2本の辺)                   (自分自身への辺)
```

### 1.2 用語

```
頂点 (Vertex/Node): グラフの点
辺 (Edge): 頂点間の接続
次数 (Degree): 頂点に接続する辺の数
  - 有向グラフ: 入次数 (in-degree) + 出次数 (out-degree)
パス (Path): 頂点の列で連続する辺が存在
  - 単純パス: 頂点の重複なし
サイクル (Cycle): 始点と終点が同じパス
  - DAG: サイクルを持たない有向グラフ
連結 (Connected): 全頂点間にパスが存在（無向グラフ）
強連結 (Strongly Connected): 全頂点間に有向パスが存在（有向グラフ）
連結成分 (Connected Component): 極大連結部分グラフ
木 (Tree): 連結でサイクルのないグラフ（E = V - 1）
森 (Forest): サイクルのないグラフ（複数の木の集合）
```

### 1.3 グラフの基本定理

```python
# グラフの重要な定理:
#
# 1. 握手定理 (Handshaking Lemma):
#    無向グラフの全頂点の次数の和 = 2 × 辺数
#    Σ deg(v) = 2|E|
#
# 2. 木の定理:
#    n 頂点の木は必ず n-1 本の辺を持つ
#    木に辺を1本追加するとサイクルが1つできる
#
# 3. オイラーの定理:
#    平面グラフについて: V - E + F = 2
#    (V: 頂点数, E: 辺数, F: 面数)
#
# 4. 辺数の上限:
#    無向単純グラフ: E ≤ V(V-1)/2
#    有向単純グラフ: E ≤ V(V-1)

def graph_info(vertices, edges, directed=False):
    """グラフの基本情報を表示"""
    v = len(vertices)
    e = len(edges)
    max_edges = v * (v - 1) if directed else v * (v - 1) // 2
    density = e / max_edges if max_edges > 0 else 0

    print(f"頂点数: {v}")
    print(f"辺数: {e}")
    print(f"最大辺数: {max_edges}")
    print(f"密度: {density:.4f}")
    print(f"疎/密: {'密' if density > 0.5 else '疎'}")
    print(f"木の可能性: {'Yes' if e == v - 1 else 'No'}")
```

---

## 2. 隣接リスト

### 2.1 辞書ベースの実装（最も一般的）

```python
class Graph:
    """辞書ベースの隣接リスト

    疎グラフに最適。動的な頂点/辺の追加が容易。
    空間計算量: O(V + E)
    """
    def __init__(self, directed=False):
        self.adj = {}
        self.directed = directed

    def add_vertex(self, v):
        """頂点を追加 — O(1)"""
        if v not in self.adj:
            self.adj[v] = []

    def add_edge(self, u, v, weight=1):
        """辺を追加 — O(1)"""
        self.add_vertex(u)
        self.add_vertex(v)
        self.adj[u].append((v, weight))
        if not self.directed:
            self.adj[v].append((u, weight))

    def remove_edge(self, u, v):
        """辺を削除 — O(degree)"""
        self.adj[u] = [(w, wt) for w, wt in self.adj.get(u, []) if w != v]
        if not self.directed:
            self.adj[v] = [(w, wt) for w, wt in self.adj.get(v, []) if w != u]

    def remove_vertex(self, v):
        """頂点とその辺を全て削除 — O(V + E)"""
        if v not in self.adj:
            return
        # v を参照する辺を削除
        for u in self.adj:
            self.adj[u] = [(w, wt) for w, wt in self.adj[u] if w != v]
        del self.adj[v]

    def neighbors(self, v):
        """隣接頂点を返す — O(1)"""
        return [(w, wt) for w, wt in self.adj.get(v, [])]

    def has_edge(self, u, v):
        """辺の存在確認 — O(degree(u))"""
        return any(w == v for w, _ in self.adj.get(u, []))

    def vertices(self):
        """全頂点 — O(V)"""
        return list(self.adj.keys())

    def edges(self):
        """全辺 — O(V + E)"""
        result = []
        visited = set()
        for u in self.adj:
            for v, w in self.adj[u]:
                edge = (min(u, v), max(u, v)) if not self.directed else (u, v)
                if edge not in visited:
                    result.append((u, v, w))
                    visited.add(edge)
        return result

    def degree(self, v):
        """次数 — O(1)"""
        return len(self.adj.get(v, []))

    def in_degree(self, v):
        """入次数（有向グラフ）— O(V + E)"""
        count = 0
        for u in self.adj:
            count += sum(1 for w, _ in self.adj[u] if w == v)
        return count

    def out_degree(self, v):
        """出次数（有向グラフ）— O(1)"""
        return len(self.adj.get(v, []))

    def __repr__(self):
        lines = []
        for v in sorted(self.adj.keys(), key=str):
            neighbors = [(w, wt) for w, wt in self.adj[v]]
            lines.append(f"  {v}: {neighbors}")
        return "Graph(\n" + "\n".join(lines) + "\n)"

# 使用例
g = Graph()
g.add_edge('A', 'B', 5)
g.add_edge('A', 'C', 3)
g.add_edge('B', 'D', 2)
g.add_edge('C', 'D', 1)
print(g)
print(f"A の隣接頂点: {g.neighbors('A')}")
print(f"B-D 辺の存在: {g.has_edge('B', 'D')}")
print(f"全辺: {g.edges()}")
```

### 2.2 defaultdict ベースの簡潔な実装

```python
from collections import defaultdict

class SimpleGraph:
    """defaultdict を使った簡潔な実装

    競技プログラミングや簡易的な用途向け
    """
    def __init__(self, directed=False):
        self.adj = defaultdict(list)
        self.directed = directed

    def add_edge(self, u, v, w=1):
        self.adj[u].append((v, w))
        if not self.directed:
            self.adj[v].append((u, w))

    def __getitem__(self, v):
        return self.adj[v]

# 重みなしグラフの場合はさらにシンプル
class UnweightedGraph:
    def __init__(self, directed=False):
        self.adj = defaultdict(set)
        self.directed = directed

    def add_edge(self, u, v):
        self.adj[u].add(v)
        if not self.directed:
            self.adj[v].add(u)

    def has_edge(self, u, v):
        return v in self.adj[u]  # O(1) average with set
```

### 2.3 隣接リスト表現の図解

```
隣接リスト表現:

  無向重み付きグラフ:
  A: [(B,5), (C,3)]
  B: [(A,5), (D,2)]
  C: [(A,3), (D,1)]
  D: [(B,2), (C,1)]

  有向グラフ:
  A: [B, C]
  B: [D]
  C: [D]
  D: []

メモリ: O(V + E)（無向は各辺が2回格納されるので O(V + 2E)）
```

---

## 3. 隣接行列

### 3.1 基本実装

```python
class GraphMatrix:
    """隣接行列: 密グラフや小規模グラフに最適

    辺の存在確認が O(1)
    空間計算量: O(V^2)
    """
    def __init__(self, n):
        self.n = n
        self.matrix = [[0] * n for _ in range(n)]

    def add_edge(self, u, v, weight=1):
        """辺を追加 — O(1)"""
        self.matrix[u][v] = weight
        self.matrix[v][u] = weight  # 無向グラフの場合

    def remove_edge(self, u, v):
        """辺を削除 — O(1)"""
        self.matrix[u][v] = 0
        self.matrix[v][u] = 0

    def has_edge(self, u, v):
        """辺の存在確認 — O(1)"""
        return self.matrix[u][v] != 0

    def neighbors(self, v):
        """隣接頂点を返す — O(V)"""
        return [u for u in range(self.n) if self.matrix[v][u] != 0]

    def degree(self, v):
        """次数 — O(V)"""
        return sum(1 for u in range(self.n) if self.matrix[v][u] != 0)

    def edge_weight(self, u, v):
        """辺の重みを返す — O(1)"""
        return self.matrix[u][v]

    def __repr__(self):
        header = "    " + " ".join(f"{i:3d}" for i in range(self.n))
        rows = []
        for i in range(self.n):
            row = f"{i:3d} " + " ".join(f"{self.matrix[i][j]:3d}" for j in range(self.n))
            rows.append(row)
        return header + "\n" + "\n".join(rows)

# 使用例
gm = GraphMatrix(4)  # A=0, B=1, C=2, D=3
gm.add_edge(0, 1, 5)  # A-B: 5
gm.add_edge(0, 2, 3)  # A-C: 3
gm.add_edge(1, 3, 2)  # B-D: 2
gm.add_edge(2, 3, 1)  # C-D: 1
print(gm)
print(f"A の隣接頂点: {gm.neighbors(0)}")  # [1, 2]
print(f"B-D の重み: {gm.edge_weight(1, 3)}")  # 2
```

### 3.2 NumPy を使った高速な隣接行列

```python
import numpy as np

class NumpyGraphMatrix:
    """NumPy ベースの隣接行列

    大規模グラフの行列演算（ワーシャル・フロイド等）に最適
    """
    def __init__(self, n):
        self.n = n
        self.matrix = np.zeros((n, n), dtype=np.float64)

    def add_edge(self, u, v, weight=1.0):
        self.matrix[u][v] = weight
        self.matrix[v][u] = weight

    def shortest_paths_floyd(self):
        """ワーシャル・フロイド法 — O(V^3)

        全ペアの最短距離を求める
        """
        dist = self.matrix.copy()
        dist[dist == 0] = np.inf
        np.fill_diagonal(dist, 0)

        for k in range(self.n):
            for i in range(self.n):
                for j in range(self.n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        return dist

    def transitive_closure(self):
        """推移的閉包: 到達可能性の行列"""
        reach = (self.matrix > 0).astype(int)
        np.fill_diagonal(reach, 1)
        for k in range(self.n):
            for i in range(self.n):
                for j in range(self.n):
                    reach[i][j] = reach[i][j] or (reach[i][k] and reach[k][j])
        return reach

    def degree_matrix(self):
        """次数行列"""
        degrees = np.sum(self.matrix > 0, axis=1)
        return np.diag(degrees)

    def laplacian_matrix(self):
        """ラプラシアン行列 = 次数行列 - 隣接行列

        グラフの連結成分数 = ラプラシアン行列の固有値 0 の重複度
        """
        adj = (self.matrix > 0).astype(float)
        return self.degree_matrix() - adj
```

### 3.3 隣接行列表現の図解

```
隣接行列表現 (A=0, B=1, C=2, D=3):

      A  B  C  D
  A [ 0  5  3  0 ]
  B [ 5  0  0  2 ]
  C [ 3  0  0  1 ]
  D [ 0  2  1  0 ]

メモリ: O(V^2)

行列演算が可能:
- A^k[i][j] = i から j への長さ k のパス数
- 固有値分析によるグラフのスペクトル解析
```

---

## 4. 辺リスト表現

```python
class EdgeListGraph:
    """辺リスト: 辺の集合としてグラフを表現

    Kruskal のアルゴリズムに最適（辺を重みでソート）
    空間計算量: O(E)
    """
    def __init__(self):
        self.edges = []
        self.vertices = set()

    def add_edge(self, u, v, weight=1):
        """O(1)"""
        self.edges.append((u, v, weight))
        self.vertices.add(u)
        self.vertices.add(v)

    def sorted_edges(self):
        """重みでソート — O(E log E)"""
        return sorted(self.edges, key=lambda e: e[2])

    def to_adjacency_list(self, directed=False):
        """隣接リストに変換"""
        adj = {v: [] for v in self.vertices}
        for u, v, w in self.edges:
            adj[u].append((v, w))
            if not directed:
                adj[v].append((u, w))
        return adj

# 使用例
g = EdgeListGraph()
g.add_edge('A', 'B', 5)
g.add_edge('A', 'C', 3)
g.add_edge('B', 'D', 2)
g.add_edge('C', 'D', 1)
print(f"辺（重み順）: {g.sorted_edges()}")
# [('C', 'D', 1), ('B', 'D', 2), ('A', 'C', 3), ('A', 'B', 5)]
```

---

## 5. Union-Find（素集合データ構造）

```python
class UnionFind:
    """Union-Find: 素集合の管理

    2つの最適化:
    1. Path Compression: find 時にノードを根に直接接続
    2. Union by Rank: 小さい木を大きい木に接続

    ほぼ O(1) の操作（逆アッカーマン関数 α(n) ≈ 5）
    """
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n
        self.components = n

    def find(self, x):
        """根を返す — O(α(n)) ≈ O(1)"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path Compression
        return self.parent[x]

    def union(self, x, y):
        """2つの集合を統合 — O(α(n)) ≈ O(1)

        Returns: True if merged, False if already in same set
        """
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False

        # Union by Rank
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        self.size[rx] += self.size[ry]
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        self.components -= 1
        return True

    def connected(self, x, y):
        """同じ集合に属するか — O(α(n))"""
        return self.find(x) == self.find(y)

    def component_size(self, x):
        """x が属する集合のサイズ"""
        return self.size[self.find(x)]

    def num_components(self):
        """連結成分の数"""
        return self.components

# 使用例
uf = UnionFind(7)
uf.union(0, 1)
uf.union(1, 2)
uf.union(3, 4)
print(uf.connected(0, 2))  # True
print(uf.connected(0, 3))  # False
print(uf.num_components())  # 4 (グループ: {0,1,2}, {3,4}, {5}, {6})
print(uf.component_size(0)) # 3
```

### 5.1 Union-Find の応用

```python
# === Kruskal の最小全域木 ===
def kruskal(vertices, edges):
    """Kruskal のアルゴリズム — O(E log E)

    辺を重みでソートし、サイクルを作らない辺を順に追加
    """
    n = len(vertices)
    uf = UnionFind(n)
    vertex_idx = {v: i for i, v in enumerate(vertices)}

    # 辺を重みでソート
    sorted_edges = sorted(edges, key=lambda e: e[2])

    mst = []
    total_weight = 0
    for u, v, w in sorted_edges:
        ui, vi = vertex_idx[u], vertex_idx[v]
        if uf.union(ui, vi):
            mst.append((u, v, w))
            total_weight += w
            if len(mst) == n - 1:
                break

    return mst, total_weight

# 使用例
vertices = ['A', 'B', 'C', 'D', 'E']
edges = [
    ('A', 'B', 4), ('A', 'C', 2), ('B', 'C', 1),
    ('B', 'D', 5), ('C', 'D', 8), ('C', 'E', 10),
    ('D', 'E', 2),
]
mst, weight = kruskal(vertices, edges)
print(f"最小全域木: {mst}")
print(f"総重み: {weight}")
# 最小全域木: [('B', 'C', 1), ('A', 'C', 2), ('D', 'E', 2), ('A', 'B', 4)]
# 総重み: 9

# === 連結成分の検出 ===
def count_connected_components(n, edges):
    """無向グラフの連結成分数"""
    uf = UnionFind(n)
    for u, v in edges:
        uf.union(u, v)
    return uf.num_components()

print(count_connected_components(5, [(0,1), (1,2), (3,4)]))  # 2

# === サイクル検出 ===
def has_cycle_undirected(n, edges):
    """無向グラフのサイクル検出"""
    uf = UnionFind(n)
    for u, v in edges:
        if uf.connected(u, v):
            return True  # 既に連結 → サイクル
        uf.union(u, v)
    return False

print(has_cycle_undirected(4, [(0,1), (1,2), (2,3)]))        # False
print(has_cycle_undirected(4, [(0,1), (1,2), (2,3), (3,0)])) # True
```

---

## 6. 特殊なグラフ

### 6.1 二部グラフ

```python
def is_bipartite(graph, n):
    """二部グラフ判定（BFS彩色）— O(V+E)

    2色で塗り分けられるかどうかを判定。
    マッチング問題の前提条件。
    """
    from collections import deque
    color = [-1] * n
    for start in range(n):
        if color[start] != -1:
            continue
        queue = deque([start])
        color[start] = 0
        while queue:
            u = queue.popleft()
            for v in graph[u]:
                if color[v] == -1:
                    color[v] = 1 - color[u]
                    queue.append(v)
                elif color[v] == color[u]:
                    return False, []
    return True, color

# 使用例
graph = {0: [1, 3], 1: [0, 2], 2: [1, 3], 3: [0, 2]}
is_bip, coloring = is_bipartite(graph, 4)
print(f"二部グラフ: {is_bip}, 彩色: {coloring}")
# 二部グラフ: True, 彩色: [0, 1, 0, 1]
```

### 6.2 グリッドをグラフとして扱う

```python
def grid_to_graph(grid):
    """2D グリッドの隣接関係

    多くのグラフ問題はグリッド上で出題される。
    迷路探索、島の数、最短経路など。
    """
    rows, cols = len(grid), len(grid[0])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def neighbors(r, c):
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                yield nr, nc

    return neighbors

# 島の数（Number of Islands）
def num_islands(grid):
    """島の数を DFS で数える — O(R × C)"""
    if not grid:
        return 0

    rows, cols = len(grid), len(grid[0])
    get_neighbors = grid_to_graph(grid)
    visited = set()
    count = 0

    def dfs(r, c):
        visited.add((r, c))
        for nr, nc in get_neighbors(r, c):
            if (nr, nc) not in visited and grid[nr][nc] == '1':
                dfs(nr, nc)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1' and (r, c) not in visited:
                dfs(r, c)
                count += 1

    return count

grid = [
    ['1', '1', '0', '0', '0'],
    ['1', '1', '0', '0', '0'],
    ['0', '0', '1', '0', '0'],
    ['0', '0', '0', '1', '1'],
]
print(f"島の数: {num_islands(grid)}")  # 3

# 8方向の場合
def grid_8dir_neighbors(grid):
    rows, cols = len(grid), len(grid[0])
    directions = [
        (0, 1), (0, -1), (1, 0), (-1, 0),
        (1, 1), (1, -1), (-1, 1), (-1, -1)
    ]
    def neighbors(r, c):
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                yield nr, nc
    return neighbors
```

### 6.3 暗黙的グラフ（Implicit Graph）

```python
# 暗黙的グラフ: 辺を明示的に持たず、
# 関数で隣接頂点を動的に生成するグラフ

# 例1: 数値パズル（状態空間グラフ）
def word_ladder(begin_word, end_word, word_list):
    """ワードラダー: 1文字ずつ変えて目的語に到達

    頂点: 各単語
    辺: 1文字だけ異なる単語ペア
    """
    from collections import deque
    word_set = set(word_list)
    if end_word not in word_set:
        return 0

    queue = deque([(begin_word, 1)])
    visited = {begin_word}

    while queue:
        word, steps = queue.popleft()
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                next_word = word[:i] + c + word[i+1:]
                if next_word == end_word:
                    return steps + 1
                if next_word in word_set and next_word not in visited:
                    visited.add(next_word)
                    queue.append((next_word, steps + 1))
    return 0

print(word_ladder("hit", "cog", ["hot","dot","dog","lot","log","cog"]))  # 5

# 例2: ナイトの最短移動
def min_knight_moves(x, y):
    """チェスのナイトが (0,0) から (x,y) への最短手数"""
    from collections import deque
    moves = [
        (2, 1), (2, -1), (-2, 1), (-2, -1),
        (1, 2), (1, -2), (-1, 2), (-1, -2)
    ]
    x, y = abs(x), abs(y)
    visited = {(0, 0)}
    queue = deque([(0, 0, 0)])

    while queue:
        cx, cy, steps = queue.popleft()
        if cx == x and cy == y:
            return steps
        for dx, dy in moves:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) not in visited and -2 <= nx <= x + 2 and -2 <= ny <= y + 2:
                visited.add((nx, ny))
                queue.append((nx, ny, steps + 1))
    return -1
```

### 6.4 トポロジカルソート

```python
from collections import deque

def topological_sort_kahn(graph, n):
    """カーンのアルゴリズム（BFS ベース）— O(V+E)

    入次数 0 のノードから順に処理。
    DAG でない場合（サイクルあり）は全ノードを処理できない。
    """
    in_degree = [0] * n
    for u in range(n):
        for v in graph.get(u, []):
            in_degree[v] += 1

    queue = deque([v for v in range(n) if in_degree[v] == 0])
    result = []

    while queue:
        u = queue.popleft()
        result.append(u)
        for v in graph.get(u, []):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    if len(result) != n:
        return None  # サイクルが存在
    return result

def topological_sort_dfs(graph, n):
    """DFS ベースのトポロジカルソート — O(V+E)"""
    visited = [False] * n
    stack = []
    has_cycle = [False]

    def dfs(u, in_stack):
        if has_cycle[0]:
            return
        visited[u] = True
        in_stack.add(u)
        for v in graph.get(u, []):
            if v in in_stack:
                has_cycle[0] = True
                return
            if not visited[v]:
                dfs(v, in_stack)
        in_stack.discard(u)
        stack.append(u)

    for v in range(n):
        if not visited[v]:
            dfs(v, set())

    if has_cycle[0]:
        return None
    return stack[::-1]

# 使用例: タスク依存関係
# 0→1→3
# 0→2→3
graph = {0: [1, 2], 1: [3], 2: [3], 3: []}
print(topological_sort_kahn(graph, 4))  # [0, 1, 2, 3] or [0, 2, 1, 3]
```

### 6.5 強連結成分（Kosaraju のアルゴリズム）

```python
def kosaraju_scc(graph, n):
    """Kosaraju のアルゴリズム: 強連結成分の分解 — O(V+E)

    Step 1: DFS で後行順序を求める
    Step 2: グラフを転置
    Step 3: 後行順序の逆順で転置グラフの DFS → 各 SCC
    """
    # Step 1: 後行順序
    visited = [False] * n
    order = []

    def dfs1(u):
        visited[u] = True
        for v in graph.get(u, []):
            if not visited[v]:
                dfs1(v)
        order.append(u)

    for v in range(n):
        if not visited[v]:
            dfs1(v)

    # Step 2: 転置グラフ
    reversed_graph = {i: [] for i in range(n)}
    for u in graph:
        for v in graph[u]:
            reversed_graph[v].append(u)

    # Step 3: 逆順で DFS
    visited = [False] * n
    sccs = []

    def dfs2(u, component):
        visited[u] = True
        component.append(u)
        for v in reversed_graph.get(u, []):
            if not visited[v]:
                dfs2(v, component)

    for u in reversed(order):
        if not visited[u]:
            component = []
            dfs2(u, component)
            sccs.append(component)

    return sccs

# 使用例
graph = {0: [1], 1: [2], 2: [0, 3], 3: [4], 4: [5], 5: [3]}
sccs = kosaraju_scc(graph, 6)
print(f"強連結成分: {sccs}")  # [[0, 2, 1], [3, 5, 4]]
```

---

## 7. グラフの表現変換

```python
def adj_list_to_matrix(adj, vertices):
    """隣接リストから隣接行列へ変換"""
    n = len(vertices)
    v_idx = {v: i for i, v in enumerate(vertices)}
    matrix = [[0] * n for _ in range(n)]
    for u in adj:
        for v, w in adj[u]:
            matrix[v_idx[u]][v_idx[v]] = w
    return matrix

def adj_matrix_to_list(matrix, vertices=None):
    """隣接行列から隣接リストへ変換"""
    n = len(matrix)
    if vertices is None:
        vertices = list(range(n))
    adj = {v: [] for v in vertices}
    for i in range(n):
        for j in range(n):
            if matrix[i][j] != 0:
                adj[vertices[i]].append((vertices[j], matrix[i][j]))
    return adj

def edge_list_to_adj_list(edges, directed=False):
    """辺リストから隣接リストへ変換"""
    from collections import defaultdict
    adj = defaultdict(list)
    for u, v, w in edges:
        adj[u].append((v, w))
        if not directed:
            adj[v].append((u, w))
    return dict(adj)
```

---

## 8. 実務応用パターン

### 8.1 ソーシャルグラフ分析

```python
from collections import deque

def bfs_shortest_path(graph, start, end):
    """2人のユーザー間の最短距離（6次の隔たり）"""
    if start == end:
        return 0, [start]

    visited = {start}
    queue = deque([(start, [start])])

    while queue:
        node, path = queue.popleft()
        for neighbor in graph.get(node, []):
            if neighbor == end:
                return len(path), path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return -1, []  # 到達不可能

def mutual_friends(graph, user_a, user_b):
    """共通の友人を返す — O(min(deg(A), deg(B)))"""
    friends_a = set(graph.get(user_a, []))
    friends_b = set(graph.get(user_b, []))
    return friends_a & friends_b

def friend_recommendations(graph, user, max_recs=10):
    """友達の友達（2ホップ先）をレコメンド

    既に友人でない人を、共通友人数でランク付け
    """
    from collections import Counter
    friends = set(graph.get(user, []))
    friends.add(user)

    candidates = Counter()
    for friend in graph.get(user, []):
        for fof in graph.get(friend, []):
            if fof not in friends:
                candidates[fof] += 1  # 共通友人数をカウント

    return candidates.most_common(max_recs)

# 使用例
social = {
    "Alice": ["Bob", "Charlie", "David"],
    "Bob": ["Alice", "Charlie", "Eve"],
    "Charlie": ["Alice", "Bob", "Frank"],
    "David": ["Alice", "Frank"],
    "Eve": ["Bob"],
    "Frank": ["Charlie", "David"],
}
print(mutual_friends(social, "Alice", "Bob"))  # {'Charlie'}
print(friend_recommendations(social, "Alice"))
# [('Frank', 2), ('Eve', 1)]  Frank は Charlie と David 経由
```

### 8.2 依存関係の解決

```python
def resolve_dependencies(packages):
    """パッケージの依存関係をトポロジカルソートで解決

    packages: {package: [dependencies]}
    """
    from collections import deque

    # 入次数の計算
    in_degree = {pkg: 0 for pkg in packages}
    for pkg, deps in packages.items():
        for dep in deps:
            if dep in in_degree:
                in_degree[dep] = in_degree.get(dep, 0)

    # 逆グラフの構築（依存先 → 依存元）
    reverse = {pkg: [] for pkg in packages}
    for pkg, deps in packages.items():
        for dep in deps:
            if dep in reverse:
                reverse[dep].append(pkg)
                in_degree[pkg] += 1

    # トポロジカルソート
    queue = deque([pkg for pkg, deg in in_degree.items() if deg == 0])
    install_order = []

    while queue:
        pkg = queue.popleft()
        install_order.append(pkg)
        for dependent in reverse.get(pkg, []):
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(install_order) != len(packages):
        return None  # 循環依存

    return install_order

# 使用例
packages = {
    "express": ["body-parser", "cookie-parser"],
    "body-parser": ["bytes", "content-type"],
    "cookie-parser": ["cookie"],
    "bytes": [],
    "content-type": [],
    "cookie": [],
}
order = resolve_dependencies(packages)
print(f"インストール順: {order}")
# ['bytes', 'content-type', 'cookie', 'body-parser', 'cookie-parser', 'express']
```

### 8.3 コースの順序（Course Schedule）

```python
def can_finish_courses(num_courses, prerequisites):
    """全コースを受講可能か判定（サイクル検出）

    prerequisites: [[course, prereq], ...]
    """
    from collections import defaultdict, deque

    graph = defaultdict(list)
    in_degree = [0] * num_courses

    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1

    # 入次数 0 のコースからスタート
    queue = deque([i for i in range(num_courses) if in_degree[i] == 0])
    count = 0

    while queue:
        course = queue.popleft()
        count += 1
        for next_course in graph[course]:
            in_degree[next_course] -= 1
            if in_degree[next_course] == 0:
                queue.append(next_course)

    return count == num_courses

print(can_finish_courses(4, [[1,0], [2,0], [3,1], [3,2]]))  # True
print(can_finish_courses(2, [[1,0], [0,1]]))  # False（循環依存）
```

### 8.4 グラフの彩色

```python
def graph_coloring(graph, n, max_colors=None):
    """貪欲法によるグラフ彩色 — O(V + E)

    各頂点に隣接頂点と異なる色を割り当てる。
    最適解の保証はないが、高々 max_degree + 1 色で彩色可能。
    """
    if max_colors is None:
        max_colors = n

    colors = [-1] * n

    for v in range(n):
        # 隣接頂点で使われている色を収集
        used_colors = set()
        for u in graph.get(v, []):
            if colors[u] != -1:
                used_colors.add(colors[u])

        # 最小の未使用色を割り当て
        for c in range(max_colors):
            if c not in used_colors:
                colors[v] = c
                break

    return colors

# 使用例: スケジューリング（時間帯の割り当て）
# 同時に行えない会議を隣接辺で表現
meetings = {0: [1, 2], 1: [0, 3], 2: [0, 3], 3: [1, 2]}
colors = graph_coloring(meetings, 4)
print(f"彩色結果: {colors}")  # [0, 1, 1, 0] — 2色で彩色可能
```

### 8.5 最短経路（BFS: 重みなし）

```python
from collections import deque

def shortest_path_bfs(graph, start, end):
    """重みなしグラフの最短経路 — O(V + E)"""
    visited = {start}
    queue = deque([(start, [start])])

    while queue:
        node, path = queue.popleft()
        if node == end:
            return path

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None  # 到達不可能

def all_shortest_paths_bfs(graph, start):
    """始点からの全頂点への最短距離 — O(V + E)"""
    dist = {start: 0}
    queue = deque([start])

    while queue:
        u = queue.popleft()
        for v in graph.get(u, []):
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)

    return dist
```

---

## 9. 比較表

### 表1: 表現方法の比較

| 操作 | 隣接リスト | 隣接行列 | 辺リスト |
|------|-----------|---------|---------|
| 空間 | O(V+E) | O(V^2) | O(E) |
| 辺の追加 | O(1) | O(1) | O(1) |
| 辺の存在確認 | O(degree) | O(1) | O(E) |
| 隣接頂点列挙 | O(degree) | O(V) | O(E) |
| 全辺列挙 | O(V+E) | O(V^2) | O(E) |
| 頂点の追加 | O(1) | O(V^2) | O(1) |
| 辺の削除 | O(degree) | O(1) | O(E) |
| 適するグラフ | 疎 | 密 | ソート前提 |
| メモリ効率 | 良い | 悪い（疎） | 最良 |

### 表2: グラフの種類と特徴

| 種類 | 特徴 | 例 | 辺数 |
|------|------|-----|------|
| 無向グラフ | 辺に方向なし | SNS の友人関係 | - |
| 有向グラフ | 辺に方向あり | Web のリンク構造 | - |
| 重み付きグラフ | 辺にコストあり | 道路ネットワーク | - |
| DAG | 有向 + サイクルなし | タスク依存関係 | - |
| 二部グラフ | 2色彩色可能 | マッチング問題 | - |
| 完全グラフ | 全頂点ペアに辺 | - | V(V-1)/2 |
| 木 | 連結 + サイクルなし | 階層構造 | V-1 |
| 平面グラフ | 辺が交差しない | V-E+F=2 | E <= 3V-6 |

### 表3: グラフアルゴリズムの計算量

| アルゴリズム | 計算量 | 用途 |
|-------------|--------|------|
| BFS | O(V+E) | 最短経路（重みなし）、連結性 |
| DFS | O(V+E) | サイクル検出、トポロジカルソート |
| Dijkstra | O((V+E) log V) | 最短経路（非負重み） |
| Bellman-Ford | O(VE) | 最短経路（負の重みあり） |
| Floyd-Warshall | O(V^3) | 全ペア最短経路 |
| Kruskal | O(E log E) | 最小全域木 |
| Prim | O((V+E) log V) | 最小全域木 |
| Tarjan SCC | O(V+E) | 強連結成分 |
| Kosaraju SCC | O(V+E) | 強連結成分 |

---

## 10. アンチパターン

### アンチパターン1: 疎グラフに隣接行列を使う

```python
# BAD: 頂点 10,000 で辺 20,000 の疎グラフ
# 隣接行列: 10,000 x 10,000 = 100,000,000 要素 (約 800MB)
matrix = [[0] * 10000 for _ in range(10000)]

# GOOD: 隣接リスト — O(V+E) = O(30,000) 程度
from collections import defaultdict
adj = defaultdict(list)
```

### アンチパターン2: 頂点の追加/削除を考慮しない

```python
# BAD: 固定サイズの隣接行列で頂点を動的に追加
class FixedGraph:
    def __init__(self):
        self.matrix = [[0] * 100 for _ in range(100)]
    # 100頂点を超えると破綻

# GOOD: 辞書ベースの隣接リスト — 動的にサイズ変更可能
class DynamicGraph:
    def __init__(self):
        self.adj = {}
    def add_vertex(self, v):
        if v not in self.adj:
            self.adj[v] = []
```

### アンチパターン3: BFS/DFS で visited を忘れる

```python
# BAD: visited チェックなし → 無限ループ
def bad_bfs(graph, start):
    queue = [start]
    while queue:
        node = queue.pop(0)
        for neighbor in graph[node]:
            queue.append(neighbor)  # 同じノードを何度も訪問!

# GOOD: visited set を使用
def good_bfs(graph, start):
    visited = {start}
    queue = [start]
    while queue:
        node = queue.pop(0)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

### アンチパターン4: 負の重みに Dijkstra を使う

```python
# BAD: 負の重みがある場合に Dijkstra → 不正な結果
# Dijkstra は貪欲法なので、負の重みがあると最短経路を見逃す

# GOOD: 負の重みがある場合は Bellman-Ford を使用
def bellman_ford(vertices, edges, start):
    dist = {v: float('inf') for v in vertices}
    dist[start] = 0

    for _ in range(len(vertices) - 1):
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w

    # 負のサイクル検出
    for u, v, w in edges:
        if dist[u] + w < dist[v]:
            return None  # 負のサイクルが存在
    return dist
```

---

## 11. FAQ

### Q1: 有向グラフと無向グラフの変換は？

**A:** 無向グラフは各辺を双方向の有向辺2本に置き換えれば有向グラフに変換できる。逆に、有向グラフから方向を無視すれば無向グラフになるが、情報が失われる。

### Q2: 密グラフと疎グラフの境界は？

**A:** E 約 V^2 なら密、E 約 V なら疎。実用上、E < V^2 / 10 程度なら隣接リストが有利。ソーシャルグラフは通常疎（ユーザー数は多いが友人数は限定的）、小規模な完全グラフは密。

### Q3: NetworkX と自前実装はどう使い分けるか？

**A:** プロトタイピングや分析には NetworkX が便利（豊富なアルゴリズムとVisualization）。競技プログラミングや性能要件が厳しいシステムでは自前実装。NetworkX は純 Python で大規模グラフ（数百万ノード）には遅い場合がある。大規模なら igraph や graph-tool（C++バックエンド）を検討。

### Q4: グラフの表現方法はどう選ぶべきか？

**A:**
- **隣接リスト**: ほとんどの場合のデフォルト。疎グラフ、動的なグラフに最適
- **隣接行列**: 密グラフ、辺の存在確認が頻繁、行列演算（Floyd-Warshall等）
- **辺リスト**: Kruskal のアルゴリズム、辺のソートが前提の問題
- **暗黙的グラフ**: 状態空間探索（パズル、ゲーム）

### Q5: Union-Find はどんな場面で使うか？

**A:** 「同じグループに属するか？」「グループを統合する」という操作が中心の問題に最適:
- 連結成分の管理
- Kruskal のアルゴリズム
- サイクル検出（無向グラフ）
- 等価クラスの管理
- 動的連結性の問題

### Q6: DAG の判定方法は？

**A:** トポロジカルソートが完了すれば DAG。完了しなければサイクルが存在する。カーンのアルゴリズム（入次数ベース）では、結果のノード数が全ノード数と一致すれば DAG。DFS ベースでは、バックエッジ（走査中のノードへの辺）が検出されればサイクルが存在。

---

## 12. まとめ

| 項目 | ポイント |
|------|---------|
| 隣接リスト | 疎グラフに最適。O(V+E) 空間。デフォルトの選択 |
| 隣接行列 | 密グラフ・辺の存在確認 O(1)。行列演算に便利 |
| 辺リスト | ソートが必要なアルゴリズム向け。Kruskal に最適 |
| Union-Find | 連結成分管理。ほぼ O(1) の find/union |
| トポロジカルソート | DAG の線形順序。依存関係の解決 |
| 強連結成分 | 有向グラフの分解。Kosaraju / Tarjan |
| 表現の選択 | グラフの密度と操作パターンで決定 |
| 有向/無向 | 問題の対称性で選択 |
| 重み | 辺にコストを付与。最短経路問題で使用 |
| 実務 | ソーシャルグラフ、依存関係、スケジューリング |

---

## 次に読むべきガイド

- [グラフ走査 — BFS/DFS](../02-algorithms/02-graph-traversal.md)
- [最短経路 — Dijkstra、Bellman-Ford](../02-algorithms/03-shortest-path.md)

---

## 参考文献

1. Cormen, T.H. et al. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. — 第20章「Elementary Graph Algorithms」、第21章「Minimum Spanning Trees」
2. Sedgewick, R. & Wayne, K. (2011). *Algorithms* (4th ed.). Addison-Wesley. — グラフの表現とアルゴリズム
3. Skiena, S.S. (2020). *The Algorithm Design Manual* (3rd ed.). Springer. — グラフの実践的設計
4. Tarjan, R.E. (1972). "Depth-first search and linear graph algorithms." *SIAM Journal on Computing*, 1(2), 146-160.
5. Kosaraju, S.R. (1978). Unpublished manuscript, referenced in Aho, Hopcroft, Ullman (1983).
