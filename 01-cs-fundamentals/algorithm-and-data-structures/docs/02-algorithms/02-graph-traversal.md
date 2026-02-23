# グラフ走査アルゴリズム

> グラフの頂点と辺を体系的に訪問するBFS・DFS・トポロジカルソートを理解し、実装と応用パターンを習得する

## この章で学ぶこと

1. **BFS（幅優先探索）とDFS（深さ優先探索）**の動作原理・実装・計算量を正確に理解する
2. **グラフの表現方法**（隣接リスト・隣接行列）とその使い分けを把握する
3. **トポロジカルソート**の原理と応用（依存関係解決・ビルドシステム等）を実装できる
4. **強連結成分・二部グラフ判定・オイラー路**などの発展的な走査アルゴリズムを理解する

---

## 1. グラフの表現方法

### 1.1 隣接リストと隣接行列

```
グラフ G:
    0 --- 1
    |   / |
    |  /  |
    2 --- 3
        |
        4

隣接リスト:                     隣接行列:
0: [1, 2]                       0  1  2  3  4
1: [0, 2, 3]                 0 [0, 1, 1, 0, 0]
2: [0, 1, 3]                 1 [1, 0, 1, 1, 0]
3: [1, 2, 4]                 2 [1, 1, 0, 1, 0]
4: [3]                        3 [0, 1, 1, 0, 1]
                              4 [0, 0, 0, 1, 0]
```

### 1.2 表現方法の比較

| 特性 | 隣接リスト | 隣接行列 |
|:---|:---|:---|
| 空間計算量 | O(V + E) | O(V^2) |
| 辺の存在判定 | O(degree(v)) | O(1) |
| 全隣接頂点の列挙 | O(degree(v)) | O(V) |
| 辺の追加 | O(1) | O(1) |
| 辺の削除 | O(degree(v)) | O(1) |
| 適するグラフ | 疎グラフ (E << V^2) | 密グラフ (E ≈ V^2) |
| メモリ効率 | 高い | 低い（疎な場合） |

### 1.3 Python での各表現方法の実装

```python
from collections import defaultdict, deque

class Graph:
    """隣接リストによるグラフ表現"""
    def __init__(self, directed=False):
        self.adj = defaultdict(list)
        self.directed = directed

    def add_edge(self, u, v, weight=None):
        if weight is not None:
            self.adj[u].append((v, weight))
            if not self.directed:
                self.adj[v].append((u, weight))
        else:
            self.adj[u].append(v)
            if not self.directed:
                self.adj[v].append(u)

    def remove_edge(self, u, v):
        self.adj[u] = [x for x in self.adj[u] if x != v]
        if not self.directed:
            self.adj[v] = [x for x in self.adj[v] if x != u]

    def vertices(self):
        verts = set()
        for u in self.adj:
            verts.add(u)
            for v in self.adj[u]:
                if isinstance(v, tuple):
                    verts.add(v[0])
                else:
                    verts.add(v)
        return verts

    def degree(self, v):
        """頂点 v の次数を返す"""
        return len(self.adj[v])

    def has_edge(self, u, v):
        """辺 (u, v) が存在するか"""
        return v in self.adj[u]

    def __repr__(self):
        return '\n'.join(f'{u}: {neighbors}' for u, neighbors in self.adj.items())


class AdjacencyMatrix:
    """隣接行列によるグラフ表現"""
    def __init__(self, n, directed=False):
        self.n = n
        self.directed = directed
        self.matrix = [[0] * n for _ in range(n)]

    def add_edge(self, u, v, weight=1):
        self.matrix[u][v] = weight
        if not self.directed:
            self.matrix[v][u] = weight

    def remove_edge(self, u, v):
        self.matrix[u][v] = 0
        if not self.directed:
            self.matrix[v][u] = 0

    def has_edge(self, u, v):
        return self.matrix[u][v] != 0

    def neighbors(self, u):
        return [v for v in range(self.n) if self.matrix[u][v] != 0]

    def degree(self, u):
        return sum(1 for v in range(self.n) if self.matrix[u][v] != 0)
```

### 1.4 辺リスト表現

```python
class EdgeListGraph:
    """辺リストによるグラフ表現（Kruskal等で有用）"""
    def __init__(self, directed=False):
        self.edges = []
        self.directed = directed

    def add_edge(self, u, v, weight=1):
        self.edges.append((u, v, weight))
        if not self.directed:
            self.edges.append((v, u, weight))

    def vertices(self):
        verts = set()
        for u, v, _ in self.edges:
            verts.add(u)
            verts.add(v)
        return verts

    def neighbors(self, u):
        return [(v, w) for src, v, w in self.edges if src == u]
```

### 1.5 実務での使い分け指針

```
判断フロー:

  グラフは疎? (E << V²)
    ├─ YES → 隣接リスト（デフォルト選択）
    └─ NO  → 密グラフ?
              ├─ YES → 隣接行列
              └─ どちらとも → 隣接リストが安全

  辺の存在判定が頻繁?
    ├─ YES → 隣接行列 or set ベースの隣接リスト
    └─ NO  → 隣接リスト

  辺をソートして処理する?
    ├─ YES → 辺リスト（Kruskal等）
    └─ NO  → 隣接リスト
```

---

## 2. BFS（幅優先探索）

キューを使い、始点から近い頂点から順に訪問する。最短経路（重みなし）を保証。

```
始点: 0

レベル0:  [0]               キュー: [0]
レベル1:  [1, 2]            キュー: [1, 2]
レベル2:  [3]               キュー: [3]
レベル3:  [4]               キュー: [4]

訪問順: 0 → 1 → 2 → 3 → 4

    0 ─── 1
    |   / |       BFS は「波紋」のように広がる
    |  /  |       同じ距離の頂点を先に訪問
    2 ─── 3
          |
          4
```

### 2.1 基本実装

```python
def bfs(graph: dict, start) -> list:
    """幅優先探索 - O(V + E)"""
    visited = set([start])
    queue = deque([start])
    order = []

    while queue:
        vertex = queue.popleft()
        order.append(vertex)

        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return order

# 最短経路（重みなし）
def bfs_shortest_path(graph: dict, start, end) -> list:
    """BFS で最短経路を復元"""
    visited = {start}
    queue = deque([(start, [start])])

    while queue:
        vertex, path = queue.popleft()
        if vertex == end:
            return path

        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return []  # 到達不可能

# 最短距離（メモリ効率版）
def bfs_shortest_distance(graph: dict, start) -> dict:
    """全頂点への最短距離を計算（メモリ効率版）"""
    dist = {start: 0}
    queue = deque([start])

    while queue:
        vertex = queue.popleft()
        for neighbor in graph[vertex]:
            if neighbor not in dist:
                dist[neighbor] = dist[vertex] + 1
                queue.append(neighbor)

    return dist

# 使用例
g = {0: [1, 2], 1: [0, 2, 3], 2: [0, 1, 3], 3: [1, 2, 4], 4: [3]}
print(bfs(g, 0))                    # [0, 1, 2, 3, 4]
print(bfs_shortest_path(g, 0, 4))   # [0, 1, 3, 4]
print(bfs_shortest_distance(g, 0))  # {0: 0, 1: 1, 2: 1, 3: 2, 4: 3}
```

### 2.2 BFS の応用: レベル別走査

```python
def bfs_levels(graph: dict, start) -> list:
    """レベル（距離）ごとに頂点をグループ化"""
    visited = {start}
    queue = deque([start])
    levels = []

    while queue:
        level_size = len(queue)
        current_level = []

        for _ in range(level_size):
            vertex = queue.popleft()
            current_level.append(vertex)

            for neighbor in graph[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        levels.append(current_level)

    return levels

print(bfs_levels(g, 0))  # [[0], [1, 2], [3], [4]]
```

### 2.3 BFS の応用: 二部グラフ判定

二部グラフとは、頂点集合を2つのグループに分割でき、同じグループ内の頂点間に辺がないグラフのこと。BFS のレベル分けを利用して判定できる。

```python
def is_bipartite(graph: dict, vertices: set) -> tuple:
    """二部グラフ判定 - O(V + E)
    返り値: (二部グラフか, 2色の割当辞書)
    """
    color = {}

    for start in vertices:
        if start in color:
            continue

        # BFS で 2色塗り
        color[start] = 0
        queue = deque([start])

        while queue:
            v = queue.popleft()
            for neighbor in graph.get(v, []):
                if neighbor not in color:
                    color[neighbor] = 1 - color[v]
                    queue.append(neighbor)
                elif color[neighbor] == color[v]:
                    return False, {}

    return True, color

# 二部グラフの例（木は常に二部グラフ）
g_bipartite = {
    0: [1, 3],
    1: [0, 2],
    2: [1, 3],
    3: [0, 2],
}
result, coloring = is_bipartite(g_bipartite, {0, 1, 2, 3})
print(f"二部グラフ: {result}")  # True
print(f"色分け: {coloring}")    # {0: 0, 1: 1, 3: 1, 2: 0}

# 非二部グラフの例（奇数サイクル）
g_odd = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
result, _ = is_bipartite(g_odd, {0, 1, 2})
print(f"二部グラフ: {result}")  # False
```

### 2.4 BFS の応用: 複数始点 BFS（マルチソース BFS）

複数の始点から同時に BFS を行う手法。最寄りの始点からの距離を一括で求められる。

```python
def multi_source_bfs(graph: dict, sources: list) -> dict:
    """複数始点BFS - O(V + E)
    各頂点から最寄りのソースまでの距離を計算
    """
    dist = {}
    queue = deque()

    for s in sources:
        dist[s] = 0
        queue.append(s)

    while queue:
        v = queue.popleft()
        for neighbor in graph.get(v, []):
            if neighbor not in dist:
                dist[neighbor] = dist[v] + 1
                queue.append(neighbor)

    return dist

# 実務例: グリッド上で複数の施設から各セルへの最短距離
# ゲーム開発での「最寄りの敵/味方までの距離マップ」生成に使われる
grid_graph = {
    (0,0): [(0,1), (1,0)],
    (0,1): [(0,0), (0,2), (1,1)],
    (0,2): [(0,1), (1,2)],
    (1,0): [(0,0), (1,1), (2,0)],
    (1,1): [(1,0), (0,1), (1,2), (2,1)],
    (1,2): [(1,1), (0,2), (2,2)],
    (2,0): [(1,0), (2,1)],
    (2,1): [(2,0), (1,1), (2,2)],
    (2,2): [(2,1), (1,2)],
}
sources = [(0,0), (2,2)]
distances = multi_source_bfs(grid_graph, sources)
print(distances)
# {(0,0): 0, (2,2): 0, (0,1): 1, (1,0): 1, (2,1): 1, (1,2): 1,
#  (1,1): 2, (0,2): 2, (2,0): 2}
```

### 2.5 0-1 BFS

辺の重みが 0 か 1 のみのグラフで、Dijkstra の代わりに deque を使って O(V+E) で最短距離を求める手法。

```python
def bfs_01(graph: dict, start) -> dict:
    """0-1 BFS - O(V + E)
    graph: {u: [(v, weight), ...]}  weight は 0 or 1
    """
    dist = defaultdict(lambda: float('inf'))
    dist[start] = 0
    dq = deque([start])

    while dq:
        u = dq.popleft()
        for v, w in graph.get(u, []):
            new_dist = dist[u] + w
            if new_dist < dist[v]:
                dist[v] = new_dist
                if w == 0:
                    dq.appendleft(v)  # 重み0 → 先頭に追加
                else:
                    dq.append(v)      # 重み1 → 末尾に追加

    return dict(dist)

# 例: グリッドで壁の破壊コスト 1、通路の移動コスト 0
graph_01 = {
    'A': [('B', 0), ('C', 1)],
    'B': [('A', 0), ('D', 1)],
    'C': [('A', 1), ('D', 0)],
    'D': [('B', 1), ('C', 0)],
}
print(bfs_01(graph_01, 'A'))
# {'A': 0, 'B': 0, 'C': 1, 'D': 1}
```

---

## 3. DFS（深さ優先探索）

スタック（または再帰）を使い、行き止まりまで深く進んでからバックトラックする。

```
始点: 0

探索の流れ:
  0 → 1 → 2 (0は訪問済み) → 3 → 4 (行き止まり)
                                ↑ バックトラック
                             3 (隣接全訪問済み)
                             ↑ バックトラック
                          ...完了

訪問順: 0 → 1 → 2 → 3 → 4

    0 ─── 1
    |   / |       DFS は「一本道」を深く進む
    |  /  |       行き止まりで引き返す
    2 ─── 3
          |
          4
```

### 3.1 基本実装

```python
def dfs_recursive(graph: dict, start, visited=None) -> list:
    """DFS 再帰版 - O(V + E)"""
    if visited is None:
        visited = set()

    visited.add(start)
    order = [start]

    for neighbor in graph[start]:
        if neighbor not in visited:
            order.extend(dfs_recursive(graph, neighbor, visited))

    return order

def dfs_iterative(graph: dict, start) -> list:
    """DFS 反復版（スタック使用）"""
    visited = set()
    stack = [start]
    order = []

    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            order.append(vertex)
            # 逆順に積むと辞書順で訪問
            for neighbor in reversed(graph[vertex]):
                if neighbor not in visited:
                    stack.append(neighbor)

    return order

g = {0: [1, 2], 1: [0, 2, 3], 2: [0, 1, 3], 3: [1, 2, 4], 4: [3]}
print(dfs_recursive(g, 0))  # [0, 1, 2, 3, 4]
print(dfs_iterative(g, 0))  # [0, 1, 2, 3, 4]
```

### 3.2 DFS のタイムスタンプ（発見時刻と完了時刻）

```python
class DFSWithTimestamp:
    """タイムスタンプ付き DFS
    辺の分類やトポロジカルソートに必要な情報を収集する
    """
    def __init__(self, graph: dict):
        self.graph = graph
        self.discovery = {}   # 発見時刻
        self.finish = {}      # 完了時刻
        self.parent = {}      # DFS木における親
        self.time = 0

    def dfs(self):
        """全頂点から DFS を実行"""
        all_vertices = set(self.graph.keys())
        for neighbors in self.graph.values():
            all_vertices.update(neighbors)

        for v in all_vertices:
            if v not in self.discovery:
                self.parent[v] = None
                self._visit(v)

    def _visit(self, u):
        self.time += 1
        self.discovery[u] = self.time

        for v in self.graph.get(u, []):
            if v not in self.discovery:
                self.parent[v] = u
                self._visit(v)

        self.time += 1
        self.finish[u] = self.time

    def classify_edge(self, u, v):
        """辺 (u, v) を分類する"""
        if self.parent.get(v) == u:
            return "tree"      # 木辺
        elif (self.discovery[u] < self.discovery[v] and
              self.finish[u] > self.finish[v]):
            return "forward"   # 前方辺
        elif (self.discovery[u] > self.discovery[v] and
              self.finish[u] < self.finish[v]):
            return "back"      # 後退辺（サイクルを示す）
        else:
            return "cross"     # 交差辺

# 使用例
g_directed = {
    'A': ['B', 'C'],
    'B': ['D'],
    'C': ['D'],
    'D': ['E'],
    'E': [],
}
dfs_ts = DFSWithTimestamp(g_directed)
dfs_ts.dfs()
print("発見時刻:", dfs_ts.discovery)
print("完了時刻:", dfs_ts.finish)
```

### 3.3 辺の分類

```
有向グラフにおける DFS の辺分類:

  Tree Edge（木辺）     : DFS木の辺。新しい頂点を発見した辺
  Back Edge（後退辺）   : 祖先へ戻る辺。サイクルの存在を示す
  Forward Edge（前方辺）: 子孫への辺（木辺以外）
  Cross Edge（交差辺）  : 別の部分木への辺

  判定基準（タイムスタンプ d[u], f[u]）:
  - Tree/Forward: d[u] < d[v] < f[v] < f[u]
  - Back:         d[v] < d[u] < f[u] < f[v]
  - Cross:        d[v] < f[v] < d[u] < f[u]

  サイクル検出: Back Edge が存在 ⟺ サイクルが存在
```

### 3.4 DFS の応用: 連結成分の検出

```python
def find_connected_components(graph: dict, vertices: set) -> list:
    """無向グラフの連結成分を列挙"""
    visited = set()
    components = []

    for v in vertices:
        if v not in visited:
            component = []
            stack = [v]
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    component.append(node)
                    for neighbor in graph.get(node, []):
                        if neighbor not in visited:
                            stack.append(neighbor)
            components.append(component)

    return components

# 切断されたグラフ
g2 = {0: [1], 1: [0], 2: [3], 3: [2], 4: []}
print(find_connected_components(g2, {0,1,2,3,4}))
# [[0, 1], [2, 3], [4]]
```

### 3.5 DFS の応用: サイクル検出

```python
def has_cycle_directed(graph: dict) -> bool:
    """有向グラフのサイクル検出（3色法）"""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = defaultdict(int)  # 全頂点 WHITE

    def dfs(u):
        color[u] = GRAY  # 探索中
        for v in graph.get(u, []):
            if color[v] == GRAY:  # 探索中の頂点に戻った → サイクル
                return True
            if color[v] == WHITE and dfs(v):
                return True
        color[u] = BLACK  # 探索完了
        return False

    for vertex in graph:
        if color[vertex] == WHITE:
            if dfs(vertex):
                return True
    return False

def has_cycle_undirected(graph: dict) -> bool:
    """無向グラフのサイクル検出"""
    visited = set()

    def dfs(v, parent):
        visited.add(v)
        for neighbor in graph.get(v, []):
            if neighbor not in visited:
                if dfs(neighbor, v):
                    return True
            elif neighbor != parent:
                return True  # 親以外の訪問済み頂点 → サイクル
        return False

    for vertex in graph:
        if vertex not in visited:
            if dfs(vertex, None):
                return True
    return False

# サイクルあり: 0 → 1 → 2 → 0
g_cycle = {0: [1], 1: [2], 2: [0]}
print(has_cycle_directed(g_cycle))  # True

# サイクルなし: DAG
g_dag = {0: [1], 1: [2], 2: []}
print(has_cycle_directed(g_dag))  # False

# 無向グラフのサイクル
g_undirected_cycle = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
print(has_cycle_undirected(g_undirected_cycle))  # True
```

### 3.6 DFS の応用: サイクルの実際の経路を復元

```python
def find_cycle_directed(graph: dict) -> list:
    """有向グラフでサイクルを1つ見つけて経路を返す"""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = defaultdict(int)
    parent = {}
    cycle_start = None
    cycle_end = None

    def dfs(u):
        nonlocal cycle_start, cycle_end
        color[u] = GRAY
        for v in graph.get(u, []):
            if color[v] == GRAY:
                cycle_start = v
                cycle_end = u
                return True
            if color[v] == WHITE:
                parent[v] = u
                if dfs(v):
                    return True
        color[u] = BLACK
        return False

    for vertex in graph:
        if color[vertex] == WHITE:
            parent[vertex] = None
            if dfs(vertex):
                break

    if cycle_start is None:
        return []

    # サイクルの経路を復元
    cycle = [cycle_start]
    current = cycle_end
    while current != cycle_start:
        cycle.append(current)
        current = parent[current]
    cycle.reverse()
    return cycle

g_cycle = {0: [1], 1: [2], 2: [3], 3: [1]}
print(find_cycle_directed(g_cycle))  # [1, 2, 3]
```

---

## 4. トポロジカルソート

DAG（有向非巡回グラフ）の頂点を、辺の方向に沿った順序に並べる。

```
  課題の依存関係:
  数学 → 物理 → 量子力学
  数学 → 線形代数 → 量子力学
  プログラミング → アルゴリズム

  DAG:
  数学 ──→ 物理 ──────→ 量子力学
    │                     ↑
    └──→ 線形代数 ────────┘
  プログラミング ──→ アルゴリズム

  トポロジカル順序の一例:
  [数学, プログラミング, 物理, 線形代数, アルゴリズム, 量子力学]
```

### 4.1 DFS ベースの実装（Tarjan）

```python
def topological_sort_dfs(graph: dict) -> list:
    """DFS ベースのトポロジカルソート - O(V + E)"""
    visited = set()
    stack = []  # 結果を逆順に積む

    def dfs(v):
        visited.add(v)
        for neighbor in graph.get(v, []):
            if neighbor not in visited:
                dfs(neighbor)
        stack.append(v)  # 帰りがけに追加

    # 全頂点から DFS
    all_vertices = set(graph.keys())
    for v in graph.values():
        all_vertices.update(v)

    for vertex in all_vertices:
        if vertex not in visited:
            dfs(vertex)

    return stack[::-1]  # 逆順が答え

dag = {
    "数学": ["物理", "線形代数"],
    "物理": ["量子力学"],
    "線形代数": ["量子力学"],
    "プログラミング": ["アルゴリズム"],
    "量子力学": [],
    "アルゴリズム": [],
}
print(topological_sort_dfs(dag))
```

### 4.2 Kahn のアルゴリズム（BFS ベース）

```python
def topological_sort_kahn(graph: dict) -> list:
    """Kahn のアルゴリズム（入次数ベース）- O(V + E)"""
    # 全頂点を収集
    all_vertices = set(graph.keys())
    for neighbors in graph.values():
        all_vertices.update(neighbors)

    # 入次数を計算
    in_degree = {v: 0 for v in all_vertices}
    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1

    # 入次数 0 の頂点をキューに
    queue = deque([v for v in all_vertices if in_degree[v] == 0])
    result = []

    while queue:
        vertex = queue.popleft()
        result.append(vertex)

        for neighbor in graph.get(vertex, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(result) != len(all_vertices):
        raise ValueError("サイクルが存在します")

    return result

print(topological_sort_kahn(dag))
```

### 4.3 全てのトポロジカル順序を列挙

```python
def all_topological_sorts(graph: dict) -> list:
    """全トポロジカル順序を列挙（バックトラッキング）"""
    all_vertices = set(graph.keys())
    for neighbors in graph.values():
        all_vertices.update(neighbors)

    in_degree = {v: 0 for v in all_vertices}
    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1

    result = []
    current = []
    visited = set()

    def backtrack():
        if len(current) == len(all_vertices):
            result.append(current[:])
            return

        for v in sorted(all_vertices):
            if v not in visited and in_degree[v] == 0:
                # 選択
                visited.add(v)
                current.append(v)
                for neighbor in graph.get(v, []):
                    in_degree[neighbor] -= 1

                backtrack()

                # 取り消し
                visited.discard(v)
                current.pop()
                for neighbor in graph.get(v, []):
                    in_degree[neighbor] += 1

    backtrack()
    return result

# 小さなDAGの例
small_dag = {'A': ['C'], 'B': ['C'], 'C': []}
print(all_topological_sorts(small_dag))
# [['A', 'B', 'C'], ['B', 'A', 'C']]
```

### 4.4 トポロジカルソートの実務応用

```python
# 実務例1: ビルドシステムの依存関係解決
build_deps = {
    "utils.o": ["utils.c", "utils.h"],
    "main.o": ["main.c", "utils.h"],
    "app": ["main.o", "utils.o"],
    "utils.c": [],
    "utils.h": [],
    "main.c": [],
}

def build_order(deps: dict) -> list:
    """ビルド順序を決定"""
    # 依存関係を逆転（A depends on B → B → A）
    graph = defaultdict(list)
    all_files = set(deps.keys())
    for target, sources in deps.items():
        for src in sources:
            graph[src].append(target)
            all_files.add(src)

    # Kahn のアルゴリズム
    in_deg = {f: 0 for f in all_files}
    for u in graph:
        for v in graph[u]:
            in_deg[v] += 1

    queue = deque([f for f in all_files if in_deg[f] == 0])
    order = []
    while queue:
        f = queue.popleft()
        order.append(f)
        for dep in graph[f]:
            in_deg[dep] -= 1
            if in_deg[dep] == 0:
                queue.append(dep)

    return order

print(build_order(build_deps))
# ['utils.c', 'utils.h', 'main.c', 'utils.o', 'main.o', 'app']


# 実務例2: タスクスケジューリング（並列実行可能なタスクの識別）
def schedule_tasks_parallel(graph: dict) -> list:
    """並列実行可能なタスクをステージごとにまとめる"""
    all_vertices = set(graph.keys())
    for neighbors in graph.values():
        all_vertices.update(neighbors)

    in_degree = {v: 0 for v in all_vertices}
    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1

    queue = deque([v for v in all_vertices if in_degree[v] == 0])
    stages = []

    while queue:
        stage = []
        next_queue = deque()
        while queue:
            v = queue.popleft()
            stage.append(v)
            for neighbor in graph.get(v, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    next_queue.append(neighbor)
        stages.append(stage)
        queue = next_queue

    return stages

task_deps = {
    'compile_a': ['link'],
    'compile_b': ['link'],
    'compile_c': ['link'],
    'link': ['test'],
    'test': ['deploy'],
    'deploy': [],
}
print(schedule_tasks_parallel(task_deps))
# [['compile_a', 'compile_b', 'compile_c'], ['link'], ['test'], ['deploy']]
```

---

## 5. 強連結成分（SCC）

有向グラフにおいて、互いに到達可能な頂点の最大集合を強連結成分と呼ぶ。

### 5.1 Kosaraju のアルゴリズム

```python
def kosaraju_scc(graph: dict) -> list:
    """Kosaraju のアルゴリズム - O(V + E)
    1. 元グラフで DFS → 完了順に頂点を記録
    2. グラフの転置を作成
    3. 転置グラフ上で、完了順の逆順に DFS → 各 DFS で到達する頂点が SCC
    """
    # 全頂点を収集
    all_vertices = set(graph.keys())
    for neighbors in graph.values():
        all_vertices.update(neighbors)

    # Step 1: 元グラフで DFS、完了順を記録
    visited = set()
    finish_order = []

    def dfs1(v):
        visited.add(v)
        for neighbor in graph.get(v, []):
            if neighbor not in visited:
                dfs1(neighbor)
        finish_order.append(v)

    for v in all_vertices:
        if v not in visited:
            dfs1(v)

    # Step 2: 転置グラフの作成
    transpose = defaultdict(list)
    for u in graph:
        for v in graph[u]:
            transpose[v].append(u)

    # Step 3: 転置グラフ上で逆順に DFS
    visited.clear()
    sccs = []

    def dfs2(v, component):
        visited.add(v)
        component.append(v)
        for neighbor in transpose.get(v, []):
            if neighbor not in visited:
                dfs2(neighbor, component)

    for v in reversed(finish_order):
        if v not in visited:
            component = []
            dfs2(v, component)
            sccs.append(component)

    return sccs

# 使用例
g_scc = {
    0: [1],
    1: [2],
    2: [0, 3],  # 0→1→2→0 がSCC
    3: [4],
    4: [5],
    5: [3],     # 3→4→5→3 がSCC
}
print(kosaraju_scc(g_scc))
# [[0, 2, 1], [3, 5, 4]] （順序は異なりうる）
```

### 5.2 Tarjan の SCC アルゴリズム

```python
def tarjan_scc(graph: dict) -> list:
    """Tarjan の SCC アルゴリズム - O(V + E)
    DFS 1回で全 SCC を発見（Kosaraju より実用的）
    """
    index_counter = [0]
    stack = []
    lowlink = {}
    index = {}
    on_stack = set()
    sccs = []

    all_vertices = set(graph.keys())
    for neighbors in graph.values():
        all_vertices.update(neighbors)

    def strongconnect(v):
        index[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)

        for w in graph.get(v, []):
            if w not in index:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], index[w])

        # v がルートなら SCC を取り出す
        if lowlink[v] == index[v]:
            component = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                component.append(w)
                if w == v:
                    break
            sccs.append(component)

    for v in all_vertices:
        if v not in index:
            strongconnect(v)

    return sccs

print(tarjan_scc(g_scc))
```

---

## 6. BFS vs DFS 比較表

| 特性 | BFS | DFS |
|:---|:---|:---|
| データ構造 | キュー（FIFO） | スタック（LIFO）/ 再帰 |
| 訪問順 | 近い頂点から | 深い頂点から |
| 最短経路（重みなし） | 保証する | 保証しない |
| メモリ使用量 | O(V)（幅に比例） | O(V)（深さに比例） |
| 木の走査 | レベル順 | 前順/中順/後順 |
| 実装の容易さ | やや複雑 | 再帰で簡潔 |
| サイクル検出 | 可能 | 3色法で容易 |
| 辺の分類 | 不可 | 可能（4種類） |
| 完全性（無限グラフ） | 保証する | 保証しない |

## 走査アルゴリズムの用途

| 用途 | 推奨アルゴリズム | 理由 |
|:---|:---|:---|
| 最短経路（重みなし） | BFS | 最短距離を保証 |
| 連結成分 | DFS | 実装がシンプル |
| トポロジカルソート | DFS / Kahn | DAG の順序付け |
| サイクル検出 | DFS（3色法） | バックエッジ検出が容易 |
| 二部グラフ判定 | BFS | レベル別の2色塗り |
| 迷路の最短経路 | BFS | グリッド上の最短探索 |
| パズル解法 | DFS + バックトラック | 状態空間の全探索 |
| ウェブクローラ | BFS | 浅いページから順に |
| 強連結成分 | DFS（Tarjan/Kosaraju） | 1-2回のDFSで完了 |
| 関節点・橋の検出 | DFS | lowlink値で判定 |
| オイラー路 | DFS (Hierholzer) | 辺を1回ずつ通る経路 |

---

## 7. グリッド上の BFS

```python
def bfs_grid(grid: list, start: tuple, end: tuple) -> int:
    """2Dグリッド上の最短経路（BFS）"""
    rows, cols = len(grid), len(grid[0])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # 右左下上

    visited = {start}
    queue = deque([(start, 0)])  # (位置, 距離)

    while queue:
        (r, c), dist = queue.popleft()
        if (r, c) == end:
            return dist

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols and
                grid[nr][nc] != 1 and (nr, nc) not in visited):
                visited.add((nr, nc))
                queue.append(((nr, nc), dist + 1))

    return -1  # 到達不可能

# 0: 通路, 1: 壁
maze = [
    [0, 0, 0, 0],
    [1, 1, 0, 1],
    [0, 0, 0, 0],
    [0, 1, 1, 0],
]
print(bfs_grid(maze, (0, 0), (3, 3)))  # 6
```

### 8方向移動のグリッド BFS

```python
def bfs_grid_8dir(grid: list, start: tuple, end: tuple) -> int:
    """8方向移動の最短経路"""
    rows, cols = len(grid), len(grid[0])
    # 8方向: 上下左右 + 斜め4方向
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    visited = {start}
    queue = deque([(start, 0)])

    while queue:
        (r, c), dist = queue.popleft()
        if (r, c) == end:
            return dist

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols and
                grid[nr][nc] != 1 and (nr, nc) not in visited):
                visited.add((nr, nc))
                queue.append(((nr, nc), dist + 1))

    return -1
```

### グリッド BFS で経路を復元

```python
def bfs_grid_with_path(grid: list, start: tuple, end: tuple) -> list:
    """2Dグリッドで最短経路を復元"""
    rows, cols = len(grid), len(grid[0])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    visited = {start}
    parent = {start: None}
    queue = deque([start])

    while queue:
        r, c = queue.popleft()
        if (r, c) == end:
            # 経路復元
            path = []
            current = end
            while current is not None:
                path.append(current)
                current = parent[current]
            return path[::-1]

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols and
                grid[nr][nc] != 1 and (nr, nc) not in visited):
                visited.add((nr, nc))
                parent[(nr, nc)] = (r, c)
                queue.append((nr, nc))

    return []  # 到達不可能

path = bfs_grid_with_path(maze, (0, 0), (3, 3))
print(path)
# [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 3), (3, 3)]
```

---

## 8. 関節点と橋の検出

無向グラフにおいて、削除するとグラフが非連結になる頂点（関節点）と辺（橋）を検出する。ネットワークの脆弱性分析に使われる。

```python
def find_articulation_points_and_bridges(graph: dict, vertices: set):
    """関節点と橋の検出 - O(V + E)"""
    discovery = {}
    low = {}
    parent = {}
    ap = set()          # 関節点
    bridges = []        # 橋
    time_counter = [0]

    def dfs(u):
        discovery[u] = low[u] = time_counter[0]
        time_counter[0] += 1
        children = 0

        for v in graph.get(u, []):
            if v not in discovery:
                children += 1
                parent[v] = u
                dfs(v)
                low[u] = min(low[u], low[v])

                # u がルートで子が2つ以上 → 関節点
                if parent[u] is None and children > 1:
                    ap.add(u)

                # u がルートでなく、子の lowlink が u の discovery 以上 → 関節点
                if parent[u] is not None and low[v] >= discovery[u]:
                    ap.add(u)

                # 橋の条件: low[v] > discovery[u]
                if low[v] > discovery[u]:
                    bridges.append((u, v))
            elif v != parent.get(u):
                low[u] = min(low[u], discovery[v])

    for v in vertices:
        if v not in discovery:
            parent[v] = None
            dfs(v)

    return ap, bridges

# 使用例
g_bridge = {
    0: [1, 2],
    1: [0, 2],
    2: [0, 1, 3],
    3: [2, 4],
    4: [3],
}
ap, bridges = find_articulation_points_and_bridges(g_bridge, {0,1,2,3,4})
print(f"関節点: {ap}")     # {2, 3}
print(f"橋: {bridges}")    # [(2, 3), (3, 4)]
```

---

## 9. オイラー路とオイラー回路

全ての辺をちょうど1回ずつ通る経路（オイラー路）と、始点に戻る経路（オイラー回路）。

```
オイラー路の存在条件:
  無向グラフ: 奇数次数の頂点が 0個（回路）or 2個（路）
  有向グラフ: 各頂点の入次数=出次数（回路）
              or 1頂点で出次数=入次数+1、1頂点で入次数=出次数+1（路）
```

```python
def find_euler_path(graph: dict) -> list:
    """Hierholzer のアルゴリズムでオイラー路/回路を求める
    graph: 隣接リスト（辺のリスト）、辺は消費される
    """
    # グラフのコピー（辺を消費するため）
    adj = defaultdict(list)
    for u in graph:
        for v in graph[u]:
            adj[u].append(v)

    # 始点の決定
    odd_vertices = [v for v in adj if len(adj[v]) % 2 == 1]
    if len(odd_vertices) == 0:
        start = next(iter(adj))  # オイラー回路: 任意の頂点
    elif len(odd_vertices) == 2:
        start = odd_vertices[0]   # オイラー路: 奇数次数頂点から
    else:
        return []  # オイラー路/回路は存在しない

    stack = [start]
    path = []

    while stack:
        v = stack[-1]
        if adj[v]:
            u = adj[v].pop()
            # 無向グラフの場合、逆辺も削除
            adj[u].remove(v)
            stack.append(u)
        else:
            path.append(stack.pop())

    return path[::-1]

# ケーニヒスベルクの橋（オイラーが解けないことを証明した問題の変形）
g_euler = {
    'A': ['B', 'C'],
    'B': ['A', 'C', 'D', 'D'],
    'C': ['A', 'B', 'D'],
    'D': ['B', 'B', 'C'],
}
path = find_euler_path(g_euler)
print(f"オイラー路: {path}")
```

---

## 10. 実務応用パターン集

### 10.1 ソーシャルネットワーク分析

```python
def mutual_friends(graph: dict, u, v) -> set:
    """共通の友人を見つける"""
    friends_u = set(graph.get(u, []))
    friends_v = set(graph.get(v, []))
    return friends_u & friends_v

def degrees_of_separation(graph: dict, u, v) -> int:
    """2人の間の隔たりの次数（最短距離）"""
    if u == v:
        return 0
    visited = {u}
    queue = deque([(u, 0)])
    while queue:
        node, dist = queue.popleft()
        for neighbor in graph.get(node, []):
            if neighbor == v:
                return dist + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return -1  # 非連結

def influence_score(graph: dict, start, max_depth: int = 3) -> int:
    """影響力スコア: max_depth ホップ以内に到達可能な頂点数"""
    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
    return len(visited) - 1  # 自分を除く
```

### 10.2 ウェブクローラの基本構造

```python
def web_crawler_bfs(start_url: str, max_pages: int = 100) -> list:
    """BFS ベースのウェブクローラ（概念的な実装）"""
    visited = {start_url}
    queue = deque([start_url])
    crawled = []

    while queue and len(crawled) < max_pages:
        url = queue.popleft()
        crawled.append(url)

        # 実際には HTTP リクエストでページを取得し、リンクを抽出する
        links = extract_links(url)  # 仮想関数

        for link in links:
            if link not in visited:
                visited.add(link)
                queue.append(link)

    return crawled

def extract_links(url):
    """仮想的なリンク抽出（実際にはHTMLパースが必要）"""
    return []  # プレースホルダ
```

### 10.3 依存関係のデッドロック検出

```python
def detect_deadlock(resource_graph: dict) -> list:
    """リソース割り当てグラフでデッドロック（サイクル）を検出"""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = defaultdict(int)
    path = []
    cycle = []

    def dfs(u):
        color[u] = GRAY
        path.append(u)
        for v in resource_graph.get(u, []):
            if color[v] == GRAY:
                # サイクル発見 → デッドロック
                idx = path.index(v)
                cycle.extend(path[idx:])
                return True
            if color[v] == WHITE and dfs(v):
                return True
        path.pop()
        color[u] = BLACK
        return False

    for vertex in resource_graph:
        if color[vertex] == WHITE:
            if dfs(vertex):
                return cycle

    return []  # デッドロックなし

# プロセス P1 → リソース R1 → プロセス P2 → リソース R2 → P1
resource_graph = {
    'P1': ['R1'],
    'R1': ['P2'],
    'P2': ['R2'],
    'R2': ['P1'],
}
deadlock = detect_deadlock(resource_graph)
print(f"デッドロック: {deadlock}")
# デッドロック: ['P1', 'R1', 'P2', 'R2']
```

---

## 11. アンチパターン

### アンチパターン1: visited チェックのタイミング

```python
# BAD: キューから取り出した時に visited チェック
# → 同じ頂点が複数回キューに入り、メモリと時間を浪費
def bad_bfs(graph, start):
    queue = deque([start])
    visited = set()
    while queue:
        v = queue.popleft()
        if v in visited:  # ここでチェック → 遅い
            continue
        visited.add(v)
        for n in graph[v]:
            queue.append(n)  # 重複追加される

# GOOD: キューに追加する時に visited チェック
def good_bfs(graph, start):
    queue = deque([start])
    visited = {start}  # 追加時にマーク
    while queue:
        v = queue.popleft()
        for n in graph[v]:
            if n not in visited:
                visited.add(n)  # ここでマーク
                queue.append(n)
```

### アンチパターン2: 再帰 DFS のスタックオーバーフロー

```python
# BAD: 大きなグラフで再帰 DFS → RecursionError
import sys
sys.setrecursionlimit(10**6)  # 応急処置だが不安定

# GOOD: 反復版 DFS を使用
def safe_dfs(graph, start):
    visited = set()
    stack = [start]
    while stack:
        v = stack.pop()
        if v not in visited:
            visited.add(v)
            for n in graph.get(v, []):
                if n not in visited:
                    stack.append(n)
```

### アンチパターン3: 隣接行列を疎グラフに使用

```python
# BAD: 頂点数 100,000 の疎グラフに隣接行列
# → メモリ 100,000 × 100,000 = 10^10 要素 → メモリ不足
n = 100000
matrix = [[0] * n for _ in range(n)]  # MemoryError!

# GOOD: 隣接リストを使う
graph = defaultdict(list)
# 辺数 E << V^2 なので、メモリは O(V + E) で済む
```

### アンチパターン4: BFS でのパス記録方法

```python
# BAD: パスを毎回コピー → O(V^2) メモリ
def bad_bfs_path(graph, start, end):
    queue = deque([(start, [start])])  # パス全体をコピー
    visited = {start}
    while queue:
        v, path = queue.popleft()
        if v == end:
            return path
        for n in graph[v]:
            if n not in visited:
                visited.add(n)
                queue.append((n, path + [n]))  # O(V) のコピー

# GOOD: 前任者辞書を使って後から復元 → O(V) メモリ
def good_bfs_path(graph, start, end):
    prev = {start: None}
    queue = deque([start])
    while queue:
        v = queue.popleft()
        if v == end:
            path = []
            while v is not None:
                path.append(v)
                v = prev[v]
            return path[::-1]
        for n in graph[v]:
            if n not in prev:
                prev[n] = v
                queue.append(n)
    return []
```

---

## 12. 計算量のまとめ

| アルゴリズム | 時間計算量 | 空間計算量 | 備考 |
|:---|:---|:---|:---|
| BFS | O(V + E) | O(V) | キュー + visited |
| DFS（再帰） | O(V + E) | O(V) | コールスタック |
| DFS（反復） | O(V + E) | O(V) | 明示的スタック |
| トポロジカルソート（DFS） | O(V + E) | O(V) | DAG限定 |
| トポロジカルソート（Kahn） | O(V + E) | O(V) | DAG限定 |
| 強連結成分（Kosaraju） | O(V + E) | O(V) | DFS 2回 |
| 強連結成分（Tarjan） | O(V + E) | O(V) | DFS 1回 |
| 関節点・橋 | O(V + E) | O(V) | lowlink使用 |
| オイラー路（Hierholzer） | O(V + E) | O(E) | 辺の管理 |
| 二部グラフ判定 | O(V + E) | O(V) | BFS/DFS |

---

## 13. FAQ

### Q1: BFS と DFS の計算量は同じ？

**A:** はい、どちらも O(V + E)。全頂点と全辺を1回ずつ処理する。違いは訪問順序とメモリの使われ方。BFS は「幅」方向にメモリを消費し、DFS は「深さ」方向に消費する。木の場合、BFS は最大幅 O(V) のメモリを使い、DFS は最大深さ O(log V)〜O(V) のメモリを使う。

### Q2: トポロジカルソートの結果は一意か？

**A:** 一般に一意ではない。複数の有効な順序が存在しうる。一意になるのは、各レベルで入次数0の頂点が常に1つの場合（ハミルトンパスが存在する場合）。辞書順で最小のトポロジカル順序を求めたい場合は、Kahn のアルゴリズムで deque の代わりに heapq（最小ヒープ）を使う。

### Q3: 有向グラフの強連結成分はどう求める？

**A:** Tarjan のアルゴリズムまたは Kosaraju のアルゴリズムを使う。どちらも O(V + E)。Tarjan は DFS 1回、Kosaraju は DFS 2回（元グラフ + 転置グラフ）で求まる。実装の簡潔さでは Kosaraju、効率では Tarjan が優位。

### Q4: グラフが暗黙的に定義される場合は？

**A:** パズルの状態空間や迷路のグリッドなど、隣接リストを事前に構築せず、BFS/DFS の訪問時にオンデマンドで隣接頂点を生成する。メモリ効率が良い。例えばルービックキューブの解法では、各状態から可能な手の組み合わせで次の状態を生成する。

### Q5: 双方向BFSとは何か？

**A:** 始点と終点の両方から同時にBFSを行い、2つの探索が出会った時点で最短経路を確定する手法。通常の BFS が O(b^d) の頂点を探索するのに対し（b=分岐係数, d=距離）、双方向BFSは O(b^(d/2)) × 2 = O(2 × b^(d/2)) で済む。指数的に探索空間を削減できる。

```python
def bidirectional_bfs(graph: dict, start, end) -> int:
    """双方向BFS - 始点と終点から同時に探索"""
    if start == end:
        return 0

    front = {start}
    back = {end}
    visited_front = {start: 0}
    visited_back = {end: 0}
    depth = 0

    while front and back:
        depth += 1
        # 小さい方を拡張（効率化）
        if len(front) > len(back):
            front, back = back, front
            visited_front, visited_back = visited_back, visited_front

        next_front = set()
        for v in front:
            for neighbor in graph.get(v, []):
                if neighbor in visited_back:
                    return visited_front[v] + 1 + visited_back[neighbor]
                if neighbor not in visited_front:
                    visited_front[neighbor] = depth
                    next_front.add(neighbor)

        front = next_front

    return -1  # 到達不可能
```

### Q6: グラフの走査で注意すべきエッジケースは？

**A:** 主なエッジケース: (1) 空のグラフ（頂点も辺もない）、(2) 孤立頂点（辺のない頂点）、(3) 自己ループ（u→u の辺）、(4) 多重辺（同じ頂点間の複数辺）、(5) 非連結グラフ（全頂点を訪問するには全頂点からの開始が必要）、(6) 単一頂点のグラフ。堅牢な実装ではこれらを全て考慮する必要がある。

---

## 14. まとめ

| 項目 | 要点 |
|:---|:---|
| BFS | キュー使用、レベル順訪問、最短経路保証（重みなし） |
| DFS | スタック/再帰使用、深さ優先、バックトラックに適する |
| トポロジカルソート | DAG の依存順序。DFS(Tarjan) or BFS(Kahn) |
| 強連結成分 | Tarjan（DFS 1回）または Kosaraju（DFS 2回） |
| グラフ表現 | 疎→隣接リスト、密→隣接行列。ほとんどの場合隣接リスト |
| 計算量 | BFS/DFS ともに O(V + E) |
| 応用範囲 | 最短経路、連結成分、サイクル検出、二部判定、依存解決 |
| 実務応用 | SNS分析、ウェブクローラ、ビルドシステム、デッドロック検出 |

---

## 次に読むべきガイド

- [最短経路アルゴリズム](./03-shortest-path.md) -- 重み付きグラフの最短経路（Dijkstra等）
- [バックトラッキング](./07-backtracking.md) -- DFS を応用した全探索
- [Union-Find](../03-advanced/00-union-find.md) -- 連結成分管理の効率的データ構造

---

## 参考文献

1. Cormen, T. H. et al. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. -- 第20-22章
2. Sedgewick, R. & Wayne, K. (2011). *Algorithms* (4th ed.). Addison-Wesley. -- Part 5: Graphs
3. Kahn, A. B. (1962). "Topological sorting of large networks." *Communications of the ACM*.
4. Tarjan, R. E. (1972). "Depth-first search and linear graph algorithms." *SIAM Journal on Computing*.
5. Kosaraju, S. R. (1978). Unpublished manuscript. -- 強連結成分の2パスアルゴリズム
6. Hierholzer, C. (1873). "Über die Möglichkeit, einen Linienzug ohne Wiederholung und ohne Unterbrechung zu umfahren." *Mathematische Annalen*.
