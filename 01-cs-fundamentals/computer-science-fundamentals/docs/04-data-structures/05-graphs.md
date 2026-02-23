# グラフ（データ構造としての）

> グラフはノードとエッジの集合であり、木、リンクリスト、さらには配列までもがグラフの特殊ケースである。

## この章で学ぶこと

- [ ] グラフの表現方法（隣接リスト、隣接行列）を実装できる
- [ ] 重み付き/有向/無向グラフの違いを理解する
- [ ] 実務でのグラフ表現パターンを知る
- [ ] BFS/DFSの実装と応用を習得する
- [ ] Union-Find（素集合データ構造）を理解する
- [ ] 最短経路・最小全域木のアルゴリズムを実装できる
- [ ] トポロジカルソートの仕組みと応用を把握する

## 前提知識

- グラフアルゴリズム → 参照: [[../03-algorithms/04-graph-algorithms.md]]

---

## 1. グラフの基礎概念

### 1.1 グラフの種類

```
グラフの分類:

  1. 方向による分類:
     無向グラフ: エッジに方向がない
       A ─── B    友人関係、道路ネットワーク
       │     │
       C ─── D

     有向グラフ（ダイグラフ）: エッジに方向がある
       A ──→ B    フォロー関係、依存関係
       ↑     ↓
       C ←── D

  2. 重みの有無:
     重みなし: エッジに重みがない（すべて等しい）
     重み付き: エッジに数値（距離、コスト等）が付く
       A ─(5)─ B
       │       │
      (3)     (2)
       │       │
       C ─(7)─ D

  3. その他の分類:
     単純グラフ: 自己ループ・多重辺がない
     多重グラフ: 2頂点間に複数の辺がある
     完全グラフ: すべての頂点間に辺がある（Kn）
     二部グラフ: 頂点を2つのグループに分けて同グループ間に辺がない
     DAG: 有向非巡回グラフ（Directed Acyclic Graph）
     連結グラフ: すべての頂点間にパスがある

  グラフの基本量:
     V: 頂点数（vertices/nodes）
     E: 辺数（edges/arcs）
     次数（degree）: ノードに接続する辺の数
     入次数（in-degree）: 有向グラフでそのノードに入る辺の数
     出次数（out-degree）: 有向グラフでそのノードから出る辺の数

  重要な定理:
     無向グラフ: Σ degree(v) = 2|E|（すべての次数の和 = 辺数の2倍）
     有向グラフ: Σ in-degree(v) = Σ out-degree(v) = |E|
     完全グラフ K_n の辺数: n(n-1)/2
     木の辺数: V - 1
```

### 1.2 グラフの用語

```
グラフの重要用語:

  パス(Path): 頂点の列 v1, v2, ..., vk（各隣接頂点間に辺がある）
  単純パス: 同じ頂点を2度通らないパス
  サイクル(Cycle): 始点と終点が同じパス
  DAG: サイクルのない有向グラフ

  連結(Connected): 無向グラフで任意の2頂点間にパスがある
  強連結(Strongly Connected): 有向グラフで任意の2頂点間に双方向のパスがある
  連結成分(Connected Component): 連結な部分グラフの最大のもの

  カット頂点(Articulation Point): 除去するとグラフが非連結になる頂点
  ブリッジ(Bridge): 除去するとグラフが非連結になる辺

  クリーク(Clique): 完全グラフである部分グラフ
  独立集合(Independent Set): 辺で接続されていない頂点の集合

  疎(Sparse): E ≪ V² のグラフ
  密(Dense): E ≈ V² のグラフ
```

---

## 2. グラフの表現

### 2.1 隣接リスト

```python
# 無向グラフ
from collections import defaultdict

class Graph:
    def __init__(self):
        self.adj = defaultdict(list)

    def add_edge(self, u, v, weight=1):
        self.adj[u].append((v, weight))
        self.adj[v].append((u, weight))  # 無向グラフ

    def neighbors(self, u):
        return self.adj[u]

# 空間: O(V + E)
# 辺の追加: O(1)
# 辺の存在確認: O(degree)
# 全辺の列挙: O(V + E)
# → 疎なグラフ（辺が少ない）に最適
```

### 2.2 有向グラフの隣接リスト

```python
class DirectedGraph:
    """有向グラフの隣接リスト表現"""

    def __init__(self):
        self.adj = defaultdict(list)        # 出辺
        self.reverse_adj = defaultdict(list) # 入辺（逆グラフ）
        self.vertices = set()

    def add_edge(self, u, v, weight=1):
        self.adj[u].append((v, weight))
        self.reverse_adj[v].append((u, weight))
        self.vertices.add(u)
        self.vertices.add(v)

    def out_neighbors(self, u):
        """出辺の隣接ノード"""
        return self.adj[u]

    def in_neighbors(self, v):
        """入辺の隣接ノード"""
        return self.reverse_adj[v]

    def out_degree(self, u):
        return len(self.adj[u])

    def in_degree(self, v):
        return len(self.reverse_adj[v])

    def has_edge(self, u, v):
        return any(node == v for node, _ in self.adj[u])

    def remove_edge(self, u, v):
        self.adj[u] = [(node, w) for node, w in self.adj[u] if node != v]
        self.reverse_adj[v] = [(node, w) for node, w in self.reverse_adj[v] if node != u]

    def get_all_edges(self):
        """全辺のリストを返す"""
        edges = []
        for u in self.adj:
            for v, w in self.adj[u]:
                edges.append((u, v, w))
        return edges

# 使用例
g = DirectedGraph()
g.add_edge("A", "B", 5)
g.add_edge("A", "C", 3)
g.add_edge("B", "D", 2)
g.add_edge("C", "D", 7)
g.add_edge("D", "A", 1)

print(g.out_neighbors("A"))  # [('B', 5), ('C', 3)]
print(g.in_neighbors("D"))   # [('B', 2), ('C', 7)]
print(g.out_degree("A"))     # 2
print(g.in_degree("D"))      # 2
```

### 2.3 隣接行列

```python
class GraphMatrix:
    def __init__(self, n):
        self.n = n
        self.matrix = [[0] * n for _ in range(n)]

    def add_edge(self, u, v, weight=1):
        self.matrix[u][v] = weight
        self.matrix[v][u] = weight  # 無向グラフ

    def has_edge(self, u, v):
        return self.matrix[u][v] != 0

# 空間: O(V²)
# 辺の追加: O(1)
# 辺の存在確認: O(1)
# → 密なグラフ（辺が多い）に最適
# → フロイドワーシャルに適する
```

### 2.4 エッジリスト

```python
class EdgeListGraph:
    """エッジリスト表現: (u, v, weight) のリスト"""

    def __init__(self):
        self.edges = []
        self.vertices = set()

    def add_edge(self, u, v, weight=1):
        self.edges.append((u, v, weight))
        self.vertices.add(u)
        self.vertices.add(v)

    def sort_by_weight(self):
        """重みでソート（クラスカル法に使用）"""
        self.edges.sort(key=lambda e: e[2])

# 空間: O(E)
# 辺の追加: O(1)
# 辺の存在確認: O(E)
# → クラスカル法、ベルマンフォード法に最適
```

### 2.5 どちらを選ぶか

```
選択基準:

  ┌────────────────┬──────────────┬──────────────┐
  │ 条件           │ 隣接リスト   │ 隣接行列     │
  ├────────────────┼──────────────┼──────────────┤
  │ 疎なグラフ     │ ✅ 省メモリ  │ ❌ 無駄多い  │
  │ 密なグラフ     │ △          │ ✅ 効率的    │
  │ 辺の存在確認   │ O(degree)   │ O(1) ✅     │
  │ 全隣接ノード   │ O(degree) ✅│ O(V)        │
  │ メモリ         │ O(V+E) ✅   │ O(V²)       │
  │ BFS/DFS        │ O(V+E) ✅   │ O(V²)       │
  │ フロイド       │ O(V³)       │ O(V³) ✅    │
  │ 辺の追加       │ O(1)        │ O(1)         │
  │ 辺の削除       │ O(E)        │ O(1) ✅     │
  └────────────────┴──────────────┴──────────────┘

  実務での目安:
  - ほとんどの場合: 隣接リスト（グラフは通常疎）
  - V < 1000 かつ密なグラフ: 隣接行列も検討
  - SNS(数億ノード): 隣接リスト一択
  - 最短経路（全頂点間）: 隣接行列 + フロイドワーシャル
```

---

## 3. グラフの走査

### 3.1 BFS（幅優先探索）

```python
from collections import deque

def bfs(graph, start):
    """幅優先探索: O(V + E)"""
    visited = set()
    queue = deque([start])
    visited.add(start)
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)

        for neighbor, _ in graph.adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return order


# BFS による最短経路（重みなし）
def bfs_shortest_path(graph, start, end):
    """重みなしグラフの最短経路: O(V + E)"""
    if start == end:
        return [start]

    visited = {start}
    queue = deque([(start, [start])])

    while queue:
        node, path = queue.popleft()
        for neighbor, _ in graph.adj[node]:
            if neighbor == end:
                return path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None  # パスが存在しない


# BFS による全最短距離
def bfs_distances(graph, start):
    """startから全ノードへの最短距離: O(V + E)"""
    distances = {start: 0}
    queue = deque([start])

    while queue:
        node = queue.popleft()
        for neighbor, _ in graph.adj[node]:
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)

    return distances


# レベルごとのBFS
def bfs_levels(graph, start):
    """各レベルのノードをグループ化"""
    visited = {start}
    queue = deque([start])
    levels = []

    while queue:
        level_size = len(queue)
        level = []
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node)
            for neighbor, _ in graph.adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        levels.append(level)

    return levels
```

### 3.2 DFS（深さ優先探索）

```python
# 再帰的DFS
def dfs_recursive(graph, start, visited=None):
    """深さ優先探索（再帰）: O(V + E)"""
    if visited is None:
        visited = set()

    visited.add(start)
    order = [start]

    for neighbor, _ in graph.adj[start]:
        if neighbor not in visited:
            order.extend(dfs_recursive(graph, neighbor, visited))

    return order


# 反復的DFS（スタック使用）
def dfs_iterative(graph, start):
    """深さ優先探索（反復）: O(V + E)"""
    visited = set()
    stack = [start]
    order = []

    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)

        # 隣接ノードを逆順でスタックに追加
        for neighbor, _ in reversed(graph.adj[node]):
            if neighbor not in visited:
                stack.append(neighbor)

    return order


# DFS によるパス検索
def dfs_all_paths(graph, start, end):
    """start から end への全パスを探索"""
    result = []

    def backtrack(node, path, visited):
        if node == end:
            result.append(path[:])
            return

        for neighbor, _ in graph.adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                backtrack(neighbor, path, visited)
                path.pop()
                visited.remove(neighbor)

    backtrack(start, [start], {start})
    return result


# DFS による連結成分の検出
def find_connected_components(graph, all_vertices):
    """全連結成分を検出: O(V + E)"""
    visited = set()
    components = []

    for v in all_vertices:
        if v not in visited:
            component = []
            stack = [v]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                for neighbor, _ in graph.adj[node]:
                    if neighbor not in visited:
                        stack.append(neighbor)
            components.append(component)

    return components
```

### 3.3 サイクル検出

```python
# 有向グラフのサイクル検出（DFS + 3色法）
def has_cycle_directed(graph, all_vertices):
    """有向グラフにサイクルがあるか判定: O(V + E)"""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {v: WHITE for v in all_vertices}

    def dfs(node):
        color[node] = GRAY  # 探索中

        for neighbor, _ in graph.adj[node]:
            if color[neighbor] == GRAY:
                return True  # 後退辺 → サイクル
            if color[neighbor] == WHITE and dfs(neighbor):
                return True

        color[node] = BLACK  # 探索完了
        return False

    for v in all_vertices:
        if color[v] == WHITE:
            if dfs(v):
                return True

    return False


# 無向グラフのサイクル検出
def has_cycle_undirected(graph, all_vertices):
    """無向グラフにサイクルがあるか判定: O(V + E)"""
    visited = set()

    def dfs(node, parent):
        visited.add(node)
        for neighbor, _ in graph.adj[node]:
            if neighbor not in visited:
                if dfs(neighbor, node):
                    return True
            elif neighbor != parent:
                return True  # 親以外の訪問済みノード → サイクル
        return False

    for v in all_vertices:
        if v not in visited:
            if dfs(v, None):
                return True

    return False


# サイクルの検出と復元
def find_cycle(graph, all_vertices):
    """サイクルを見つけて返す（有向グラフ）"""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {v: WHITE for v in all_vertices}
    parent = {}
    cycle_start = None
    cycle_end = None

    def dfs(node):
        nonlocal cycle_start, cycle_end
        color[node] = GRAY

        for neighbor, _ in graph.adj[node]:
            if color[neighbor] == GRAY:
                cycle_start = neighbor
                cycle_end = node
                return True
            if color[neighbor] == WHITE:
                parent[neighbor] = node
                if dfs(neighbor):
                    return True

        color[node] = BLACK
        return False

    for v in all_vertices:
        if color[v] == WHITE:
            if dfs(v):
                # サイクルを復元
                cycle = [cycle_start]
                current = cycle_end
                while current != cycle_start:
                    cycle.append(current)
                    current = parent[current]
                cycle.reverse()
                return cycle

    return None  # サイクルなし
```

---

## 4. トポロジカルソート

### 4.1 DFS ベース

```python
def topological_sort_dfs(graph, all_vertices):
    """DFS ベースのトポロジカルソート: O(V + E)"""
    visited = set()
    order = []

    def dfs(node):
        visited.add(node)
        for neighbor, _ in graph.adj[node]:
            if neighbor not in visited:
                dfs(neighbor)
        order.append(node)  # 後処理で追加

    for v in all_vertices:
        if v not in visited:
            dfs(v)

    order.reverse()
    return order
```

### 4.2 カーンのアルゴリズム（BFS ベース）

```python
from collections import deque

def topological_sort_kahn(graph, all_vertices):
    """カーンのアルゴリズム: O(V + E)
    サイクル検出も同時に行える"""

    # 入次数を計算
    in_degree = {v: 0 for v in all_vertices}
    for u in all_vertices:
        for v, _ in graph.adj[u]:
            in_degree[v] = in_degree.get(v, 0) + 1

    # 入次数0のノードをキューに追加
    queue = deque([v for v in all_vertices if in_degree[v] == 0])
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)

        for neighbor, _ in graph.adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(order) != len(all_vertices):
        raise ValueError("グラフにサイクルが存在します")

    return order


# 使用例: タスクの依存関係
g = DirectedGraph()
g.add_edge("compile", "link")
g.add_edge("compile", "test")
g.add_edge("link", "deploy")
g.add_edge("test", "deploy")
g.add_edge("init", "compile")

order = topological_sort_kahn(g, g.vertices)
print(order)  # ['init', 'compile', 'link', 'test', 'deploy'] など
```

### 4.3 トポロジカルソートの応用

```python
# 1. コースの履修順序
def find_course_order(num_courses, prerequisites):
    """
    prerequisites: [(course, prerequisite), ...]
    返り値: 履修順序のリスト（不可能な場合は空リスト）
    """
    graph = defaultdict(list)
    in_degree = [0] * num_courses

    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1

    queue = deque([i for i in range(num_courses) if in_degree[i] == 0])
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return order if len(order) == num_courses else []

# 使用例
print(find_course_order(4, [(1,0), (2,0), (3,1), (3,2)]))
# [0, 1, 2, 3] or [0, 2, 1, 3]


# 2. ビルドシステムの依存関係解決
def resolve_dependencies(packages):
    """
    packages: {name: [dependencies]}
    返り値: インストール順序
    """
    graph = defaultdict(list)
    all_pkgs = set()

    for pkg, deps in packages.items():
        all_pkgs.add(pkg)
        for dep in deps:
            graph[dep].append(pkg)
            all_pkgs.add(dep)

    in_degree = {pkg: 0 for pkg in all_pkgs}
    for pkg, deps in packages.items():
        in_degree[pkg] = len(deps)

    queue = deque([pkg for pkg in all_pkgs if in_degree[pkg] == 0])
    order = []

    while queue:
        pkg = queue.popleft()
        order.append(pkg)
        for dependent in graph[pkg]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(order) != len(all_pkgs):
        raise ValueError("循環依存を検出しました")

    return order

# 使用例
packages = {
    "app": ["framework", "database"],
    "framework": ["utils", "logging"],
    "database": ["utils"],
    "utils": [],
    "logging": [],
}
print(resolve_dependencies(packages))
# ['utils', 'logging', 'framework', 'database', 'app']
```

---

## 5. Union-Find（素集合データ構造）

### 5.1 基本実装

```python
class UnionFind:
    """Union-Find（素集合データ構造）
    経路圧縮 + ランクによる統合で O(α(n)) ≈ O(1)"""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = n  # 連結成分の数

    def find(self, x):
        """根を見つける（経路圧縮付き）"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 経路圧縮
        return self.parent[x]

    def union(self, x, y):
        """2つの集合を統合（ランクによる統合）"""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False  # 既に同じ集合

        # ランクの低い方を高い方の下に接続
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        self.count -= 1
        return True

    def connected(self, x, y):
        """同じ集合に属するか判定"""
        return self.find(x) == self.find(y)

    def get_count(self):
        """連結成分の数を返す"""
        return self.count


# 使用例
uf = UnionFind(10)
uf.union(0, 1)
uf.union(2, 3)
uf.union(1, 3)
print(uf.connected(0, 3))  # True（0-1-3-2 が同じ集合）
print(uf.connected(0, 5))  # False
print(uf.get_count())       # 7（{0,1,2,3}, {4}, {5}, ..., {9}）
```

### 5.2 Union-Find の応用

```python
# 1. 冗長な辺の検出
def find_redundant_connection(edges):
    """無向グラフからサイクルを作る辺を見つける"""
    n = len(edges)
    uf = UnionFind(n + 1)

    for u, v in edges:
        if not uf.union(u, v):
            return [u, v]  # この辺でサイクルが形成される

    return []


# 2. 島の数のカウント（2Dグリッド）
def num_islands(grid):
    """'1'の連結成分の数を数える"""
    if not grid:
        return 0

    rows, cols = len(grid), len(grid[0])
    uf = UnionFind(rows * cols)
    water_count = 0

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '0':
                water_count += 1
                continue
            # 右と下の隣接セルと統合
            for dr, dc in [(0, 1), (1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == '1':
                    uf.union(r * cols + c, nr * cols + nc)

    return uf.get_count() - water_count


# 3. アカウントの統合
def accounts_merge(accounts):
    """同じメールアドレスを持つアカウントを統合"""
    email_to_id = {}
    email_to_name = {}
    uf = UnionFind(len(accounts))

    for i, account in enumerate(accounts):
        name = account[0]
        for email in account[1:]:
            email_to_name[email] = name
            if email in email_to_id:
                uf.union(i, email_to_id[email])
            email_to_id[email] = i

    # 統合されたアカウントのメールをグループ化
    groups = defaultdict(set)
    for email, account_id in email_to_id.items():
        root = uf.find(account_id)
        groups[root].add(email)

    result = []
    for root, emails in groups.items():
        name = email_to_name[next(iter(emails))]
        result.append([name] + sorted(emails))

    return result


# 4. 最小全域木（クラスカル法）
def kruskal_mst(n, edges):
    """クラスカル法で最小全域木を求める: O(E log E)"""
    # 辺を重みでソート
    edges.sort(key=lambda e: e[2])

    uf = UnionFind(n)
    mst = []
    total_weight = 0

    for u, v, weight in edges:
        if not uf.connected(u, v):
            uf.union(u, v)
            mst.append((u, v, weight))
            total_weight += weight

            if len(mst) == n - 1:
                break  # 全頂点が接続

    return mst, total_weight

# 使用例
edges = [
    (0, 1, 4), (0, 7, 8),
    (1, 2, 8), (1, 7, 11),
    (2, 3, 7), (2, 5, 4),
    (2, 8, 2), (3, 4, 9),
    (3, 5, 14), (4, 5, 10),
    (5, 6, 2), (6, 7, 1),
    (6, 8, 6), (7, 8, 7),
]
mst, weight = kruskal_mst(9, edges)
print(f"MST重み: {weight}")  # 37
print(f"MST辺: {mst}")
```

---

## 6. 最短経路アルゴリズム

### 6.1 ダイクストラ法

```python
import heapq

def dijkstra(graph, start):
    """ダイクストラ法: O((V + E) log V)
    重みが非負のグラフの単一始点最短経路"""

    distances = {v: float('inf') for v in graph.adj}
    distances[start] = 0
    predecessors = {v: None for v in graph.adj}
    pq = [(0, start)]  # (距離, ノード)

    while pq:
        dist, node = heapq.heappop(pq)

        if dist > distances[node]:
            continue  # 古いエントリをスキップ

        for neighbor, weight in graph.adj[node]:
            new_dist = dist + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                predecessors[neighbor] = node
                heapq.heappush(pq, (new_dist, neighbor))

    return distances, predecessors


def reconstruct_path(predecessors, start, end):
    """最短経路を復元"""
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = predecessors[current]
    path.reverse()
    return path if path[0] == start else []


# 使用例
g = Graph()
g.add_edge("A", "B", 4)
g.add_edge("A", "C", 2)
g.add_edge("B", "D", 3)
g.add_edge("C", "D", 1)
g.add_edge("C", "B", 1)
g.add_edge("D", "E", 5)

distances, preds = dijkstra(g, "A")
print(distances)  # {'A': 0, 'B': 3, 'C': 2, 'D': 3, 'E': 8}
print(reconstruct_path(preds, "A", "E"))  # ['A', 'C', 'D', 'E']
```

### 6.2 ベルマンフォード法

```python
def bellman_ford(vertices, edges, start):
    """ベルマンフォード法: O(V * E)
    負の辺も扱える。負の閉路を検出可能"""

    distances = {v: float('inf') for v in vertices}
    distances[start] = 0
    predecessors = {v: None for v in vertices}

    # V-1 回の緩和
    for i in range(len(vertices) - 1):
        updated = False
        for u, v, w in edges:
            if distances[u] + w < distances[v]:
                distances[v] = distances[u] + w
                predecessors[v] = u
                updated = True
        if not updated:
            break  # 早期終了

    # 負の閉路の検出
    for u, v, w in edges:
        if distances[u] + w < distances[v]:
            raise ValueError("負の閉路が検出されました")

    return distances, predecessors

# 使用例
vertices = ["A", "B", "C", "D"]
edges = [
    ("A", "B", 4),
    ("A", "C", 2),
    ("B", "D", 3),
    ("C", "B", -1),  # 負の辺
    ("C", "D", 5),
]
distances, _ = bellman_ford(vertices, edges, "A")
print(distances)  # {'A': 0, 'B': 1, 'C': 2, 'D': 4}
```

### 6.3 フロイドワーシャル法

```python
def floyd_warshall(n, edges):
    """フロイドワーシャル法: O(V³)
    全頂点間の最短経路"""

    INF = float('inf')
    dist = [[INF] * n for _ in range(n)]
    next_node = [[None] * n for _ in range(n)]

    # 初期化
    for i in range(n):
        dist[i][i] = 0

    for u, v, w in edges:
        dist[u][v] = w
        next_node[u][v] = v

    # 動的計画法
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_node[i][j] = next_node[i][k]

    # 負の閉路の検出
    for i in range(n):
        if dist[i][i] < 0:
            raise ValueError("負の閉路が検出されました")

    return dist, next_node


def reconstruct_fw_path(next_node, u, v):
    """フロイドワーシャルのパス復元"""
    if next_node[u][v] is None:
        return []
    path = [u]
    while u != v:
        u = next_node[u][v]
        path.append(u)
    return path
```

### 6.4 A*アルゴリズム

```python
import heapq

def a_star(graph, start, goal, heuristic):
    """A*アルゴリズム: O((V + E) log V)
    ヒューリスティック関数で探索を効率化

    heuristic(node): ノードからゴールまでの推定コスト
    ヒューリスティックが admissible（過大評価しない）なら最適解を保証
    """

    open_set = [(0 + heuristic(start), 0, start)]  # (f, g, node)
    g_scores = {start: 0}
    came_from = {}

    while open_set:
        f, g, current = heapq.heappop(open_set)

        if current == goal:
            # パスを復元
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1], g

        for neighbor, weight in graph.adj[current]:
            tentative_g = g + weight

            if tentative_g < g_scores.get(neighbor, float('inf')):
                g_scores[neighbor] = tentative_g
                came_from[neighbor] = current
                f = tentative_g + heuristic(neighbor)
                heapq.heappush(open_set, (f, tentative_g, neighbor))

    return None, float('inf')  # パスが見つからない


# 使用例: 2Dグリッドでの経路探索
def grid_a_star(grid, start, goal):
    """2Dグリッドでの A* 経路探索"""

    def heuristic(pos):
        """マンハッタン距離"""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    rows, cols = len(grid), len(grid[0])
    open_set = [(heuristic(start), 0, start)]
    g_scores = {start: 0}
    came_from = {}

    while open_set:
        _, g, current = heapq.heappop(open_set)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = current[0] + dr, current[1] + dc
            neighbor = (nr, nc)

            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != 1:
                tentative_g = g + 1
                if tentative_g < g_scores.get(neighbor, float('inf')):
                    g_scores[neighbor] = tentative_g
                    came_from[neighbor] = current
                    f = tentative_g + heuristic(neighbor)
                    heapq.heappush(open_set, (f, tentative_g, neighbor))

    return None  # パスなし

# 使用例
grid = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0],
]
path = grid_a_star(grid, (0, 0), (4, 4))
print(path)  # [(0,0), (0,1), (0,2), (0,3), (1,3), (2,3), (2,4), (3,4), (4,4)]
```

---

## 7. 高度なグラフアルゴリズム

### 7.1 二部グラフ判定

```python
def is_bipartite(graph, all_vertices):
    """グラフが二部グラフかどうかをBFSで判定: O(V + E)"""
    color = {}

    for start in all_vertices:
        if start in color:
            continue

        queue = deque([start])
        color[start] = 0

        while queue:
            node = queue.popleft()
            for neighbor, _ in graph.adj[node]:
                if neighbor not in color:
                    color[neighbor] = 1 - color[node]
                    queue.append(neighbor)
                elif color[neighbor] == color[node]:
                    return False  # 同じ色の隣接ノード → 二部グラフでない

    return True

# 二部グラフの応用:
# - マッチング問題（求人と求職者の最適マッチング）
# - 2色塗り分け問題
# - 偶数長のサイクルのみを持つグラフの検出
```

### 7.2 強連結成分（Kosaraju のアルゴリズム）

```python
def kosaraju_scc(graph, all_vertices):
    """Kosaraju のアルゴリズムで強連結成分を求める: O(V + E)"""

    # パス1: 元のグラフでDFS、終了順を記録
    visited = set()
    finish_order = []

    def dfs1(node):
        visited.add(node)
        for neighbor, _ in graph.adj[node]:
            if neighbor not in visited:
                dfs1(neighbor)
        finish_order.append(node)

    for v in all_vertices:
        if v not in visited:
            dfs1(v)

    # 逆グラフを構築
    reverse_graph = defaultdict(list)
    for u in graph.adj:
        for v, w in graph.adj[u]:
            reverse_graph[v].append((u, w))

    # パス2: 逆グラフで終了順の逆順にDFS
    visited = set()
    sccs = []

    def dfs2(node, component):
        visited.add(node)
        component.append(node)
        for neighbor, _ in reverse_graph[node]:
            if neighbor not in visited:
                dfs2(neighbor, component)

    for v in reversed(finish_order):
        if v not in visited:
            component = []
            dfs2(v, component)
            sccs.append(component)

    return sccs

# 使用例: Webページのリンク構造分析
# 強連結成分 = 互いにリンクで到達可能なページのグループ
```

### 7.3 最小全域木（プリム法）

```python
def prim_mst(graph, start):
    """プリム法で最小全域木を求める: O((V + E) log V)"""
    visited = set()
    mst = []
    total_weight = 0

    # (重み, 現在のノード, 親ノード)
    pq = [(0, start, None)]

    while pq:
        weight, node, parent = heapq.heappop(pq)

        if node in visited:
            continue

        visited.add(node)
        if parent is not None:
            mst.append((parent, node, weight))
            total_weight += weight

        for neighbor, w in graph.adj[node]:
            if neighbor not in visited:
                heapq.heappush(pq, (w, neighbor, node))

    return mst, total_weight
```

---

## 8. 実務でのグラフ

### 8.1 実世界のグラフ問題

```
実務でのグラフ表現と応用:

  1. RDB: テーブル間のリレーション → 暗黙のグラフ
     users, follows テーブル → ソーシャルグラフ

  2. Neo4j: グラフDB
     Cypher: MATCH (a)-[:FOLLOWS]->(b) WHERE a.name = 'Alice'

  3. GraphQL: APIのグラフ構造

  4. npm/pip: パッケージ依存関係グラフ（DAG）

  5. Kubernetes: サービス間通信（サービスメッシュ）

  6. Google Maps: 道路ネットワーク → ダイクストラ/A*

  7. SNS: フォロー/フレンド関係 → ソーシャルグラフ

  8. 推薦システム: ユーザー×アイテムの二部グラフ

  9. コンパイラ: 制御フローグラフ（CFG）

  10. CI/CD: パイプラインのタスク依存関係（DAG）
      GitHub Actions, Airflow, Terraform
```

### 8.2 グラフデータベースとクエリ

```python
# NetworkX を使ったグラフ分析（Python の標準的なグラフライブラリ）
import networkx as nx

# グラフの作成
G = nx.DiGraph()  # 有向グラフ
G.add_weighted_edges_from([
    ("Alice", "Bob", 1),
    ("Alice", "Charlie", 1),
    ("Bob", "David", 1),
    ("Charlie", "David", 1),
    ("David", "Eve", 1),
])

# 基本的な分析
print(f"ノード数: {G.number_of_nodes()}")
print(f"辺数: {G.number_of_edges()}")
print(f"次数: {dict(G.degree())}")

# 最短経路
print(nx.shortest_path(G, "Alice", "Eve"))
# ['Alice', 'Bob', 'David', 'Eve']

# PageRank
pr = nx.pagerank(G)
print(f"PageRank: {pr}")

# 中心性分析
betweenness = nx.betweenness_centrality(G)
print(f"媒介中心性: {betweenness}")

# 連結成分
components = list(nx.weakly_connected_components(G))
print(f"弱連結成分数: {len(components)}")


# Neo4j Cypher クエリの例
"""
// フォロワーのフォロワーを検索（2ホップ）
MATCH (a:User {name: 'Alice'})-[:FOLLOWS]->(b)-[:FOLLOWS]->(c)
WHERE a <> c
RETURN c.name AS recommended_friend

// 最短経路
MATCH path = shortestPath(
  (a:User {name: 'Alice'})-[:FOLLOWS*..10]-(b:User {name: 'Eve'})
)
RETURN path

// コミュニティ検出
CALL gds.louvain.stream('social-graph')
YIELD nodeId, communityId
RETURN gds.util.asNode(nodeId).name AS name, communityId
"""
```

### 8.3 グラフを使ったシステム設計

```python
# 1. タスクスケジューラ（DAG ベース）
class TaskScheduler:
    """依存関係を考慮したタスクスケジューラ"""

    def __init__(self):
        self.tasks = {}      # task_name -> callable
        self.deps = defaultdict(list)  # task -> [dependencies]

    def add_task(self, name, func, dependencies=None):
        self.tasks[name] = func
        if dependencies:
            for dep in dependencies:
                self.deps[name].append(dep)

    def execute(self):
        """トポロジカル順序でタスクを実行"""
        # 入次数の計算
        in_degree = {task: 0 for task in self.tasks}
        graph = defaultdict(list)
        for task, deps in self.deps.items():
            for dep in deps:
                graph[dep].append(task)
                in_degree[task] += 1

        # BFS でトポロジカルソート
        queue = deque([t for t in self.tasks if in_degree[t] == 0])
        results = {}

        while queue:
            task = queue.popleft()
            print(f"実行中: {task}")
            results[task] = self.tasks[task]()
            for dependent in graph[task]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        return results


# 2. ソーシャルグラフの分析
class SocialGraph:
    """ソーシャルネットワークのグラフ分析"""

    def __init__(self):
        self.graph = defaultdict(set)  # user -> set of friends

    def add_friendship(self, user1, user2):
        self.graph[user1].add(user2)
        self.graph[user2].add(user1)

    def mutual_friends(self, user1, user2):
        """共通の友人を取得"""
        return self.graph[user1] & self.graph[user2]

    def friend_recommendations(self, user, top_n=5):
        """友人の友人から推薦（共通友人数でランク）"""
        scores = defaultdict(int)
        friends = self.graph[user]

        for friend in friends:
            for fof in self.graph[friend]:
                if fof != user and fof not in friends:
                    scores[fof] += 1  # 共通友人数をスコアに

        return sorted(scores.items(), key=lambda x: -x[1])[:top_n]

    def degrees_of_separation(self, user1, user2):
        """2人のユーザー間の隔たり（最短距離）"""
        if user1 == user2:
            return 0

        visited = {user1}
        queue = deque([(user1, 0)])

        while queue:
            node, dist = queue.popleft()
            for friend in self.graph[node]:
                if friend == user2:
                    return dist + 1
                if friend not in visited:
                    visited.add(friend)
                    queue.append((friend, dist + 1))

        return -1  # 到達不可能

    def clustering_coefficient(self, user):
        """クラスタリング係数: 友人同士の結びつきの度合い"""
        friends = list(self.graph[user])
        if len(friends) < 2:
            return 0.0

        # 友人間の辺の数を数える
        edges = 0
        for i in range(len(friends)):
            for j in range(i + 1, len(friends)):
                if friends[j] in self.graph[friends[i]]:
                    edges += 1

        # 可能な辺の最大数
        max_edges = len(friends) * (len(friends) - 1) / 2
        return edges / max_edges if max_edges > 0 else 0.0


# 使用例
sg = SocialGraph()
sg.add_friendship("Alice", "Bob")
sg.add_friendship("Alice", "Charlie")
sg.add_friendship("Bob", "Charlie")
sg.add_friendship("Bob", "David")
sg.add_friendship("Charlie", "Eve")
sg.add_friendship("David", "Eve")

print(sg.mutual_friends("Alice", "David"))    # {'Bob'}
print(sg.friend_recommendations("Alice"))     # [('David', 1), ('Eve', 1)]
print(sg.degrees_of_separation("Alice", "Eve"))  # 2
print(sg.clustering_coefficient("Alice"))      # 1.0（友人同士が全員結びついている）


# 3. ルートプランナー
class RoutePlanner:
    """重み付きグラフによる経路計画"""

    def __init__(self):
        self.graph = defaultdict(list)
        self.coordinates = {}  # ノード → (lat, lon)

    def add_road(self, city1, city2, distance):
        self.graph[city1].append((city2, distance))
        self.graph[city2].append((city1, distance))

    def set_coordinates(self, city, lat, lon):
        self.coordinates[city] = (lat, lon)

    def shortest_route(self, start, end):
        """ダイクストラ法で最短ルートを計算"""
        distances = {start: 0}
        predecessors = {start: None}
        pq = [(0, start)]

        while pq:
            dist, city = heapq.heappop(pq)
            if city == end:
                break
            if dist > distances.get(city, float('inf')):
                continue
            for neighbor, weight in self.graph[city]:
                new_dist = dist + weight
                if new_dist < distances.get(neighbor, float('inf')):
                    distances[neighbor] = new_dist
                    predecessors[neighbor] = city
                    heapq.heappush(pq, (new_dist, neighbor))

        # パス復元
        if end not in predecessors:
            return None, float('inf')

        path = []
        current = end
        while current is not None:
            path.append(current)
            current = predecessors[current]
        path.reverse()
        return path, distances[end]

# 使用例
planner = RoutePlanner()
planner.add_road("Tokyo", "Yokohama", 30)
planner.add_road("Tokyo", "Chiba", 40)
planner.add_road("Yokohama", "Nagoya", 350)
planner.add_road("Chiba", "Nagoya", 380)
planner.add_road("Nagoya", "Osaka", 180)

path, dist = planner.shortest_route("Tokyo", "Osaka")
print(f"ルート: {' → '.join(path)}")   # Tokyo → Yokohama → Nagoya → Osaka
print(f"総距離: {dist}km")              # 560km
```

---

## 9. グラフの視覚化

```python
# matplotlib + networkx によるグラフの描画
import matplotlib
matplotlib.use('Agg')  # 非GUIバックエンド
import matplotlib.pyplot as plt
import networkx as nx

def visualize_graph(edges, directed=False, weighted=True):
    """グラフをPNG画像として保存"""
    G = nx.DiGraph() if directed else nx.Graph()

    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(10, 8))
    nx.draw(G, pos,
            with_labels=True,
            node_color='lightblue',
            node_size=800,
            font_size=12,
            font_weight='bold',
            edge_color='gray',
            arrows=directed)

    if weighted:
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title("Graph Visualization")
    plt.tight_layout()
    plt.savefig("graph.png", dpi=150)
    plt.close()

# ASCII でのグラフ表示
def print_graph_ascii(graph):
    """隣接リストをASCIIで表示"""
    for node in sorted(graph.adj.keys()):
        neighbors = [(n, w) for n, w in graph.adj[node]]
        neighbor_str = ", ".join(f"{n}({w})" for n, w in sorted(neighbors))
        print(f"  {node} → [{neighbor_str}]")
```

---

## 10. 実践演習

### 演習1: グラフ構築（基礎）
隣接リストと隣接行列の両方でグラフを実装し、BFS/DFSを実行せよ。以下の操作を含むこと:
- 頂点と辺の追加・削除
- 辺の存在確認
- 全隣接ノードの取得
- BFS/DFS による走査

### 演習2: 二部グラフ判定（応用）
グラフが二部グラフかどうかをBFSで判定する関数を実装せよ。さらに、二部グラフであれば2つのグループを返すようにせよ。

### 演習3: 最小全域木（発展）
クラスカル法とプリム法の両方で最小全域木を求める関数を実装し、結果が一致することを検証せよ。

### 演習4: ダイクストラ法（応用）
重み付きグラフでダイクストラ法を実装し、最短経路と距離を返す関数を作成せよ。負の辺がある場合にベルマンフォード法に切り替える機能も実装すること。

### 演習5: トポロジカルソート（応用）
大学のカリキュラムを有向グラフで表現し、以下を実装せよ:
- 全科目の履修順序をトポロジカルソートで決定
- 循環依存の検出と報告
- 並列に履修可能な科目のグループ化

### 演習6: ソーシャルグラフ分析（発展）
ソーシャルネットワークのグラフを構築し、以下を実装せよ:
- 共通の友人の検索
- 友人推薦（友人の友人をスコアリング）
- 六次の隔たりの計算
- コミュニティ検出（連結成分の分析）

### 演習7: A*アルゴリズム（発展）
2Dグリッド上の障害物を避けた最短経路をA*アルゴリズムで求めよ:
- マンハッタン距離とユークリッド距離のヒューリスティック比較
- BFS/ダイクストラとの性能比較
- 結果のグリッド上への可視化

---

## まとめ

| 表現方法 | 空間 | 辺の確認 | 最適場面 |
|---------|------|---------|---------|
| 隣接リスト | O(V+E) | O(degree) | 疎グラフ、一般用途 |
| 隣接行列 | O(V^2) | O(1) | 密グラフ、小規模 |
| エッジリスト | O(E) | O(E) | クラスカル法 |

| アルゴリズム | 計算量 | 用途 |
|-------------|--------|------|
| BFS | O(V+E) | 最短経路（重みなし）、レベル走査 |
| DFS | O(V+E) | サイクル検出、トポロジカルソート |
| ダイクストラ | O((V+E)log V) | 最短経路（非負重み） |
| ベルマンフォード | O(VE) | 最短経路（負の辺あり） |
| フロイドワーシャル | O(V^3) | 全点間最短経路 |
| A* | O((V+E)log V) | ヒューリスティック付き最短経路 |
| クラスカル | O(E log E) | 最小全域木 |
| プリム | O((V+E)log V) | 最小全域木 |
| カーン | O(V+E) | トポロジカルソート |
| コサラジュ | O(V+E) | 強連結成分 |

---

## 次に読むべきガイド
→ [[06-advanced-structures.md]] -- 高度なデータ構造

---

## 参考文献
1. Cormen, T. H. "Introduction to Algorithms." Chapters 22-26.
2. Sedgewick, R. "Algorithms." Chapter 4.1-4.4.
3. Kleinberg, J., Tardos, E. "Algorithm Design." Chapters 3-7.
4. Skiena, S. S. "The Algorithm Design Manual." Chapter 5-7.
5. Hart, P. E., Nilsson, N. J., Raphael, B. "A Formal Basis for the Heuristic Determination of Minimum Cost Paths." 1968.
6. Dijkstra, E. W. "A Note on Two Problems in Connexion with Graphs." 1959.
7. Kruskal, J. B. "On the Shortest Spanning Subtree of a Graph." 1956.
