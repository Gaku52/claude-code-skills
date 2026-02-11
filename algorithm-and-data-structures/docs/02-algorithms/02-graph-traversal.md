# グラフ走査アルゴリズム

> グラフの頂点と辺を体系的に訪問するBFS・DFS・トポロジカルソートを理解し、実装と応用パターンを習得する

## この章で学ぶこと

1. **BFS（幅優先探索）とDFS（深さ優先探索）**の動作原理・実装・計算量を正確に理解する
2. **グラフの表現方法**（隣接リスト・隣接行列）とその使い分けを把握する
3. **トポロジカルソート**の原理と応用（依存関係解決・ビルドシステム等）を実装できる

---

## 1. グラフの表現方法

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

```python
from collections import defaultdict, deque

class Graph:
    """隣接リストによるグラフ表現"""
    def __init__(self, directed=False):
        self.adj = defaultdict(list)
        self.directed = directed

    def add_edge(self, u, v):
        self.adj[u].append(v)
        if not self.directed:
            self.adj[v].append(u)

    def vertices(self):
        verts = set()
        for u in self.adj:
            verts.add(u)
            for v in self.adj[u]:
                verts.add(v)
        return verts
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

### 実装

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

# 使用例
g = {0: [1, 2], 1: [0, 2, 3], 2: [0, 1, 3], 3: [1, 2, 4], 4: [3]}
print(bfs(g, 0))                    # [0, 1, 2, 3, 4]
print(bfs_shortest_path(g, 0, 4))   # [0, 1, 3, 4]
```

### BFS の応用: レベル別走査

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

### 実装

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

### DFS の応用: 連結成分の検出

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

### DFS の応用: サイクル検出

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

# サイクルあり: 0 → 1 → 2 → 0
g_cycle = {0: [1], 1: [2], 2: [0]}
print(has_cycle_directed(g_cycle))  # True

# サイクルなし: DAG
g_dag = {0: [1], 1: [2], 2: []}
print(has_cycle_directed(g_dag))  # False
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

### DFS ベースの実装（Tarjan）

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

### Kahn のアルゴリズム（BFS ベース）

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

---

## 5. BFS vs DFS 比較表

| 特性 | BFS | DFS |
|:---|:---|:---|
| データ構造 | キュー（FIFO） | スタック（LIFO）/ 再帰 |
| 訪問順 | 近い頂点から | 深い頂点から |
| 最短経路（重みなし） | 保証する | 保証しない |
| メモリ使用量 | O(V)（幅に比例） | O(V)（深さに比例） |
| 木の走査 | レベル順 | 前順/中順/後順 |
| 実装の容易さ | やや複雑 | 再帰で簡潔 |
| サイクル検出 | 可能 | 3色法で容易 |

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

---

## 6. グリッド上の BFS

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

---

## 7. アンチパターン

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

---

## 8. FAQ

### Q1: BFS と DFS の計算量は同じ？

**A:** はい、どちらも O(V + E)。全頂点と全辺を1回ずつ処理する。違いは訪問順序とメモリの使われ方。BFS は「幅」方向にメモリを消費し、DFS は「深さ」方向に消費する。

### Q2: トポロジカルソートの結果は一意か？

**A:** 一般に一意ではない。複数の有効な順序が存在しうる。一意になるのは、各レベルで入次数0の頂点が常に1つの場合（ハミルトンパスが存在する場合）。

### Q3: 有向グラフの強連結成分はどう求める？

**A:** Tarjan のアルゴリズムまたは Kosaraju のアルゴリズムを使う。どちらも O(V + E)。Tarjan は DFS 1回、Kosaraju は DFS 2回（元グラフ + 転置グラフ）で求まる。

### Q4: グラフが暗黙的に定義される場合は？

**A:** パズルの状態空間や迷路のグリッドなど、隣接リストを事前に構築せず、BFS/DFS の訪問時にオンデマンドで隣接頂点を生成する。メモリ効率が良い。

---

## 9. まとめ

| 項目 | 要点 |
|:---|:---|
| BFS | キュー使用、レベル順訪問、最短経路保証（重みなし） |
| DFS | スタック/再帰使用、深さ優先、バックトラックに適する |
| トポロジカルソート | DAG の依存順序。DFS(Tarjan) or BFS(Kahn) |
| グラフ表現 | 疎→隣接リスト、密→隣接行列。ほとんどの場合隣接リスト |
| 計算量 | BFS/DFS ともに O(V + E) |
| 応用範囲 | 最短経路、連結成分、サイクル検出、二部判定、依存解決 |

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
