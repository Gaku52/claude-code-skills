# グラフ — 表現方法・隣接リスト/行列・重み付きグラフ

> ネットワーク、依存関係、地図など多様な関係を表現するグラフの基本概念と、各表現方法の特徴を学ぶ。

---

## この章で学ぶこと

1. **グラフの基本用語** — 頂点、辺、有向/無向、重み
2. **隣接リストと隣接行列** の実装と使い分け
3. **重み付きグラフ** と特殊なグラフの表現

---

## 1. グラフの基本概念

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
```

### 用語

```
頂点 (Vertex/Node): グラフの点
辺 (Edge): 頂点間の接続
次数 (Degree): 頂点に接続する辺の数
  - 有向グラフ: 入次数 (in-degree) + 出次数 (out-degree)
パス (Path): 頂点の列で連続する辺が存在
サイクル (Cycle): 始点と終点が同じパス
連結 (Connected): 全頂点間にパスが存在
```

---

## 2. 隣接リスト

```python
# 辞書ベースの隣接リスト（最も一般的）
class Graph:
    def __init__(self, directed=False):
        self.adj = {}
        self.directed = directed

    def add_vertex(self, v):
        if v not in self.adj:
            self.adj[v] = []

    def add_edge(self, u, v, weight=1):
        """辺を追加 — O(1)"""
        self.add_vertex(u)
        self.add_vertex(v)
        self.adj[u].append((v, weight))
        if not self.directed:
            self.adj[v].append((u, weight))

    def neighbors(self, v):
        """隣接頂点を返す — O(1)"""
        return self.adj.get(v, [])

    def has_edge(self, u, v):
        """辺の存在確認 — O(degree(u))"""
        return any(w == v for w, _ in self.adj.get(u, []))

# 使用例
g = Graph()
g.add_edge('A', 'B', 5)
g.add_edge('A', 'C', 3)
g.add_edge('B', 'D', 2)
g.add_edge('C', 'D', 1)
```

```
隣接リスト表現:

  A: [(B,5), (C,3)]
  B: [(A,5), (D,2)]
  C: [(A,3), (D,1)]
  D: [(B,2), (C,1)]

メモリ: O(V + E)
```

---

## 3. 隣接行列

```python
class GraphMatrix:
    def __init__(self, n):
        self.n = n
        self.matrix = [[0] * n for _ in range(n)]

    def add_edge(self, u, v, weight=1):
        """辺を追加 — O(1)"""
        self.matrix[u][v] = weight
        self.matrix[v][u] = weight  # 無向グラフの場合

    def has_edge(self, u, v):
        """辺の存在確認 — O(1)"""
        return self.matrix[u][v] != 0

    def neighbors(self, v):
        """隣接頂点を返す — O(V)"""
        return [u for u in range(self.n) if self.matrix[v][u] != 0]
```

```
隣接行列表現 (A=0, B=1, C=2, D=3):

      A  B  C  D
  A [ 0  5  3  0 ]
  B [ 5  0  0  2 ]
  C [ 3  0  0  1 ]
  D [ 0  2  1  0 ]

メモリ: O(V²)
```

---

## 4. 辺リスト表現

```python
class EdgeListGraph:
    def __init__(self):
        self.edges = []
        self.vertices = set()

    def add_edge(self, u, v, weight=1):
        self.edges.append((u, v, weight))
        self.vertices.add(u)
        self.vertices.add(v)

# 使用例
g = EdgeListGraph()
g.add_edge('A', 'B', 5)
g.add_edge('A', 'C', 3)
# edges = [('A','B',5), ('A','C',3)]
# Kruskal のアルゴリズムなどに便利
```

---

## 5. 特殊なグラフ

### 5.1 二部グラフ

```python
def is_bipartite(graph, n):
    """二部グラフ判定（BFS彩色）— O(V+E)"""
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
                    return False
    return True
```

### 5.2 グラフの構築パターン

```python
# グリッドをグラフとして扱う
def grid_to_graph(grid):
    """2D グリッドの隣接関係"""
    rows, cols = len(grid), len(grid[0])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def neighbors(r, c):
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                yield nr, nc

    return neighbors
```

---

## 6. 比較表

### 表1: 表現方法の比較

| 操作 | 隣接リスト | 隣接行列 | 辺リスト |
|------|-----------|---------|---------|
| 空間 | O(V+E) | O(V²) | O(E) |
| 辺の追加 | O(1) | O(1) | O(1) |
| 辺の存在確認 | O(degree) | O(1) | O(E) |
| 隣接頂点列挙 | O(degree) | O(V) | O(E) |
| 全辺列挙 | O(V+E) | O(V²) | O(E) |
| 適するグラフ | 疎 | 密 | ソート前提 |

### 表2: グラフの種類と特徴

| 種類 | 特徴 | 例 |
|------|------|-----|
| 無向グラフ | 辺に方向なし | SNS の友人関係 |
| 有向グラフ | 辺に方向あり | Web のリンク構造 |
| 重み付きグラフ | 辺にコストあり | 道路ネットワーク |
| DAG | 有向 + サイクルなし | タスク依存関係 |
| 二部グラフ | 2色彩色可能 | マッチング問題 |
| 完全グラフ | 全頂点ペアに辺 | E = V(V-1)/2 |
| 木 | 連結 + サイクルなし | E = V-1 |

---

## 7. アンチパターン

### アンチパターン1: 疎グラフに隣接行列を使う

```python
# BAD: 頂点 10,000 で辺 20,000 の疎グラフ
# 隣接行列: 10,000 × 10,000 = 100,000,000 要素 (約 800MB)
matrix = [[0] * 10000 for _ in range(10000)]

# GOOD: 隣接リスト — O(V+E) = O(30,000) 程度
adj = {i: [] for i in range(10000)}
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

---

## 8. FAQ

### Q1: 有向グラフと無向グラフの変換は？

**A:** 無向グラフは各辺を双方向の有向辺2本に置き換えれば有向グラフに変換できる。逆に、有向グラフから方向を無視すれば無向グラフになるが、情報が失われる。

### Q2: 密グラフと疎グラフの境界は？

**A:** E ≈ V² なら密、E ≈ V なら疎。実用上、E < V² / 10 程度なら隣接リストが有利。ソーシャルグラフは通常疎、小規模な完全グラフは密。

### Q3: NetworkX と自前実装はどう使い分けるか？

**A:** プロトタイピングや分析には NetworkX が便利。競技プログラミングや性能要件が厳しいシステムでは自前実装。NetworkX は純 Python で大規模グラフには遅い場合がある。

---

## 9. まとめ

| 項目 | ポイント |
|------|---------|
| 隣接リスト | 疎グラフに最適。O(V+E) 空間 |
| 隣接行列 | 密グラフ・辺の存在確認 O(1) |
| 辺リスト | ソートが必要なアルゴリズム向け |
| 表現の選択 | グラフの密度と操作パターンで決定 |
| 有向/無向 | 問題の対称性で選択 |
| 重み | 辺にコストを付与。最短経路問題で使用 |

---

## 次に読むべきガイド

- [グラフ走査 — BFS/DFS](../02-algorithms/02-graph-traversal.md)
- [最短経路 — Dijkstra、Bellman-Ford](../02-algorithms/03-shortest-path.md)

---

## 参考文献

1. Cormen, T.H. et al. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. — 第20章「Elementary Graph Algorithms」
2. Sedgewick, R. & Wayne, K. (2011). *Algorithms* (4th ed.). Addison-Wesley. — グラフの表現
3. Skiena, S.S. (2020). *The Algorithm Design Manual* (3rd ed.). Springer. — グラフの実践的設計
