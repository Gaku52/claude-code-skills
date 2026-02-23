# 最短経路アルゴリズム

> 重み付きグラフにおける最短経路問題を、Dijkstra・Bellman-Ford・Floyd-Warshallの3大アルゴリズムで解く手法を体系的に理解する

## この章で学ぶこと

1. **Dijkstra法**の原理と優先度キュー実装で単一始点最短経路を効率的に求められる
2. **Bellman-Ford法**で負の辺重みを扱い、負閉路の検出ができる
3. **Floyd-Warshall法**で全点対最短経路を動的計画法で求められる
4. **Johnson法**や**0-1 BFS**、**A*探索**などの発展的アルゴリズムを理解する
5. 各アルゴリズムの適用条件と使い分けを正確に判断できる

---

## 1. 最短経路問題の分類

```
┌─────────────────────────────────────────────────────┐
│               最短経路問題                            │
├───────────────────┬─────────────────────────────────┤
│  単一始点          │  全点対間                        │
│  (SSSP)           │  (APSP)                         │
├───────────────────┼─────────────────────────────────┤
│ Dijkstra          │ Floyd-Warshall                   │
│ (負辺なし)        │ O(V³)                            │
│ O((V+E) log V)   │                                   │
├───────────────────┤                                   │
│ Bellman-Ford      │ Johnson                           │
│ (負辺あり)        │ (疎グラフ向け)                     │
│ O(VE)             │ O(V² log V + VE)                 │
├───────────────────┤                                   │
│ DAG 最短経路      │                                   │
│ (DAG限定)         │                                   │
│ O(V + E)          │                                   │
├───────────────────┤                                   │
│ A* 探索           │                                   │
│ (ヒューリスティック)│                                  │
│ O(E) 最良ケース   │                                   │
└───────────────────┴─────────────────────────────────┘
```

### 最短経路問題の前提知識

```
緩和（Relaxation）操作 — 全アルゴリズムに共通する基本操作:

  if dist[u] + weight(u, v) < dist[v]:
      dist[v] = dist[u] + weight(u, v)
      prev[v] = u

  「u を経由して v に行く方が近いなら、v の距離を更新する」

  この操作を:
  - Dijkstra  → 頂点を確定する度に実行
  - Bellman   → 全辺に対して V-1 回繰り返す
  - Floyd     → 中継点を増やしながら実行
  - DAG       → トポロジカル順に実行
```

---

## 2. Dijkstra法

非負の辺重みを持つグラフで、単一始点からの最短経路を求める。「確定していない頂点の中で最短距離の頂点を確定する」を繰り返す。

```
グラフ:
      2        3
  A ────→ B ────→ D
  │       ↑       ↑
  │1      │1      │1
  ↓       │       │
  C ────→ E ────→ F
      4        2

始点 A からの最短距離:
  Step 0: A=0, B=inf, C=inf, D=inf, E=inf, F=inf
  Step 1: A=0* → C=1, B=2
  Step 2: C=1* → E=5
  Step 3: B=2* → D=5, E=min(5,3)=3
  Step 4: E=3* → F=5
  Step 5: D=5* (B経由: 2+3=5)
  Step 6: F=5* (E経由: 3+2=5)

  最短距離: A=0, B=2, C=1, D=5, E=3, F=5
```

### 2.1 優先度キュー実装

```python
import heapq
from collections import defaultdict

def dijkstra(graph: dict, start: str) -> tuple:
    """Dijkstra法 - O((V+E) log V)
    graph: {u: [(v, weight), ...], ...}
    返り値: (距離辞書, 前任者辞書)
    """
    dist = defaultdict(lambda: float('inf'))
    prev = {}
    dist[start] = 0

    # (距離, 頂点) の最小ヒープ
    pq = [(0, start)]

    while pq:
        d, u = heapq.heappop(pq)

        # 古い情報はスキップ
        if d > dist[u]:
            continue

        for v, weight in graph.get(u, []):
            new_dist = dist[u] + weight
            if new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(pq, (new_dist, v))

    return dict(dist), prev

def reconstruct_path(prev: dict, start: str, end: str) -> list:
    """前任者辞書から経路を復元"""
    path = []
    current = end
    while current != start:
        if current not in prev:
            return []  # 到達不可能
        path.append(current)
        current = prev[current]
    path.append(start)
    return path[::-1]

# 使用例
graph = {
    'A': [('B', 2), ('C', 1)],
    'B': [('D', 3)],
    'C': [('E', 4)],
    'E': [('B', 1), ('F', 2)],
    'F': [('D', 1)],
}
dist, prev = dijkstra(graph, 'A')
print(dist)                            # {'A': 0, 'B': 2, 'C': 1, 'D': 5, 'E': 5, 'F': 7}
print(reconstruct_path(prev, 'A', 'D'))  # ['A', 'B', 'D']
```

### 2.2 Dijkstra法の正当性の証明（概要）

```
Dijkstra法の貪欲選択性質:

  定理: 非負辺グラフにおいて、未確定頂点の中で距離が最小の頂点 u を
        確定すると、dist[u] は u への真の最短距離である。

  証明の概要（帰納法・背理法）:
  1. dist[u] が最短でないと仮定する
  2. すると、別の未確定頂点 w を経由するより短い経路が存在する
  3. しかし w は未確定なので dist[w] >= dist[u]
  4. 辺の重みが非負なので、w を経由する経路は dist[u] 以上
  5. 矛盾 → dist[u] は最短距離

  この証明が成り立つ条件: 全辺の重みが非負
  負辺があると Step 4 が成立しなくなる
```

### 2.3 Dijkstra法のバリエーション

#### visitedフラグ版（メモリ効率版）

```python
def dijkstra_with_visited(graph: dict, start: str) -> dict:
    """visited フラグを使った Dijkstra（ヒープに古いエントリが溜まりにくい）"""
    dist = defaultdict(lambda: float('inf'))
    dist[start] = 0
    visited = set()
    pq = [(0, start)]

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)

        for v, weight in graph.get(u, []):
            if v not in visited:
                new_dist = d + weight
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    heapq.heappush(pq, (new_dist, v))

    return dict(dist)
```

#### k番目に短い経路を求めるDijkstra

```python
def dijkstra_kth_shortest(graph: dict, start: str, end: str, k: int) -> list:
    """k番目に短い経路の距離を返す
    各頂点に最大 k 回到達するまで探索を続ける
    """
    count = defaultdict(int)  # 各頂点への到達回数
    pq = [(0, start)]
    distances = []

    while pq:
        d, u = heapq.heappop(pq)
        count[u] += 1

        if u == end:
            distances.append(d)
            if len(distances) == k:
                return distances

        if count[u] > k:
            continue

        for v, weight in graph.get(u, []):
            heapq.heappush(pq, (d + weight, v))

    return distances

# 使用例
graph_multi = {
    'A': [('B', 1), ('C', 3)],
    'B': [('C', 1), ('D', 4)],
    'C': [('D', 1)],
    'D': [],
}
print(dijkstra_kth_shortest(graph_multi, 'A', 'D', 3))
# [3, 4, 5]  (A→B→C→D=3, A→C→D=4, A→B→D=5)
```

### 2.4 グリッド上の Dijkstra

```python
def dijkstra_grid(grid: list, start: tuple, end: tuple) -> int:
    """2Dグリッド上の Dijkstra（各セルにコストがある場合）"""
    rows, cols = len(grid), len(grid[0])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    dist = [[float('inf')] * cols for _ in range(rows)]
    dist[start[0]][start[1]] = grid[start[0]][start[1]]

    pq = [(grid[start[0]][start[1]], start[0], start[1])]

    while pq:
        d, r, c = heapq.heappop(pq)

        if (r, c) == end:
            return d

        if d > dist[r][c]:
            continue

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                new_dist = d + grid[nr][nc]
                if new_dist < dist[nr][nc]:
                    dist[nr][nc] = new_dist
                    heapq.heappush(pq, (new_dist, nr, nc))

    return -1

# 各セルの通過コスト
cost_grid = [
    [1, 3, 1, 2],
    [1, 5, 1, 1],
    [4, 2, 1, 3],
    [1, 1, 1, 1],
]
print(dijkstra_grid(cost_grid, (0, 0), (3, 3)))  # 7
```

---

## 3. Bellman-Ford法

負の辺重みを許容する単一始点最短経路アルゴリズム。全辺を V-1 回緩和し、V 回目の緩和で負閉路を検出する。

```
緩和（Relaxation）の概念:
  if dist[u] + weight(u,v) < dist[v]:
      dist[v] = dist[u] + weight(u,v)

  A ──(5)──→ B
  ↓           ↓
 (2)        (-3)
  ↓           ↓
  C ──(4)──→ D

  Step 0: A=0, B=inf, C=inf, D=inf
  Step 1: A→B: B=5, A→C: C=2
  Step 2: B→D: D=2, C→D: D=min(2,6)=2
  Step 3: 変化なし → 収束

なぜ V-1 回で十分か:
  最短経路は最大 V-1 本の辺を含む（サイクルがない場合）。
  各回のイテレーションで、最短経路の辺を少なくとも1本確定させる。
  したがって V-1 回のイテレーションで全ての最短経路が求まる。
```

### 3.1 基本実装

```python
def bellman_ford(vertices: list, edges: list, start) -> tuple:
    """Bellman-Ford法 - O(VE)
    edges: [(u, v, weight), ...]
    返り値: (距離辞書, 前任者辞書, 負閉路フラグ)
    """
    dist = {v: float('inf') for v in vertices}
    prev = {v: None for v in vertices}
    dist[start] = 0

    # V-1 回の緩和
    for _ in range(len(vertices) - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
                updated = True
        if not updated:  # 早期終了
            break

    # V 回目：負閉路の検出
    has_negative_cycle = False
    for u, v, w in edges:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            has_negative_cycle = True
            break

    return dist, prev, has_negative_cycle

# 使用例（負の辺あり）
vertices = ['A', 'B', 'C', 'D', 'E']
edges = [
    ('A', 'B', 4), ('A', 'C', 2),
    ('B', 'D', 3), ('C', 'B', -1),
    ('C', 'D', 5), ('D', 'E', 1),
]
dist, prev, neg_cycle = bellman_ford(vertices, edges, 'A')
print(dist)       # {'A': 0, 'B': 1, 'C': 2, 'D': 4, 'E': 5}
print(neg_cycle)  # False

# 負閉路の例
edges_neg = [('A', 'B', 1), ('B', 'C', -3), ('C', 'A', 1)]
_, _, neg = bellman_ford(['A','B','C'], edges_neg, 'A')
print(neg)  # True
```

### 3.2 負閉路に影響される頂点の特定

```python
def bellman_ford_with_negative_cycle_detection(vertices, edges, start):
    """負閉路の影響を受ける全頂点を特定"""
    dist = {v: float('inf') for v in vertices}
    prev = {v: None for v in vertices}
    dist[start] = 0

    # V-1 回の緩和
    for _ in range(len(vertices) - 1):
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u

    # V 回目で更新される頂点 → 負閉路の影響を受ける
    affected = set()
    for u, v, w in edges:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            affected.add(v)

    # 負閉路から到達可能な全頂点も影響を受ける
    # BFS で伝播
    queue = deque(affected)
    while queue:
        node = queue.popleft()
        for u, v, w in edges:
            if u == node and v not in affected:
                affected.add(v)
                queue.append(v)

    return dist, affected

vertices_nc = ['A', 'B', 'C', 'D', 'E']
edges_nc = [
    ('A', 'B', 1), ('B', 'C', -3), ('C', 'B', 1),  # B-C間に負閉路
    ('C', 'D', 2), ('D', 'E', 1),
]
dist, affected = bellman_ford_with_negative_cycle_detection(vertices_nc, edges_nc, 'A')
print(f"負閉路の影響を受ける頂点: {affected}")
# {'B', 'C', 'D', 'E'} — B,C が負閉路上、D,E は到達可能
```

### 3.3 SPFA（Shortest Path Faster Algorithm）

Bellman-Ford の改良版。更新された頂点のみをキューに入れて処理する。平均的には高速だが、最悪は O(VE)。

```python
from collections import deque

def spfa(graph: dict, start) -> tuple:
    """SPFA - Bellman-Ford の改良版
    graph: {u: [(v, weight), ...]}
    平均計算量は Bellman-Ford より大幅に速い
    """
    all_vertices = set(graph.keys())
    for neighbors in graph.values():
        for v, _ in neighbors:
            all_vertices.add(v)

    dist = {v: float('inf') for v in all_vertices}
    dist[start] = 0
    in_queue = {v: False for v in all_vertices}
    count = {v: 0 for v in all_vertices}  # 各頂点のキュー入り回数

    queue = deque([start])
    in_queue[start] = True
    count[start] = 1

    while queue:
        u = queue.popleft()
        in_queue[u] = False

        for v, w in graph.get(u, []):
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                if not in_queue[v]:
                    queue.append(v)
                    in_queue[v] = True
                    count[v] += 1
                    if count[v] >= len(all_vertices):
                        return dist, True  # 負閉路検出

    return dist, False

graph_spfa = {
    'A': [('B', 4), ('C', 2)],
    'B': [('D', 3)],
    'C': [('B', -1), ('D', 5)],
    'D': [('E', 1)],
    'E': [],
}
dist, has_neg = spfa(graph_spfa, 'A')
print(dist)      # {'A': 0, 'B': 1, 'C': 2, 'D': 4, 'E': 5}
print(has_neg)   # False
```

---

## 4. Floyd-Warshall法

全頂点対の最短経路を動的計画法で求める。中継点を1つずつ増やしながら距離行列を更新する。

```
初期距離行列:              k=1 (A経由):
     A    B    C    D         A    B    C    D
A [  0,   3, inf,   7]   A [  0,   3, inf,   7]
B [  8,   0,   2, inf]   B [  8,   0,   2,  15]
C [  5, inf,   0,   1]   C [  5,   8,   0,   1]
D [  2, inf, inf,   0]   D [  2,   5, inf,   0]

k=2 (B経由):              最終結果:
     A    B    C    D         A    B    C    D
A [  0,   3,   5,   7]   A [  0,   3,   5,   6]
B [  8,   0,   2,   3]   B [  5,   0,   2,   3]
C [  5,   8,   0,   1]   C [  3,   6,   0,   1]
D [  2,   5,   7,   0]   D [  2,   5,   7,   0]

DP の遷移式:
  dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

  「i から j への距離を、k を中継点として使った場合と使わなかった場合で比較」
```

### 4.1 基本実装

```python
def floyd_warshall(n: int, edges: list) -> list:
    """Floyd-Warshall法 - O(V³)
    edges: [(u, v, weight), ...]
    返り値: 距離行列
    """
    INF = float('inf')
    dist = [[INF] * n for _ in range(n)]

    # 初期化
    for i in range(n):
        dist[i][i] = 0
    for u, v, w in edges:
        dist[u][v] = w

    # DP: 中継点 k を順に追加
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    return dist
```

### 4.2 経路復元付き実装

```python
def floyd_warshall_with_path(n: int, edges: list) -> tuple:
    """Floyd-Warshall + 経路復元"""
    INF = float('inf')
    dist = [[INF] * n for _ in range(n)]
    nxt = [[None] * n for _ in range(n)]

    for i in range(n):
        dist[i][i] = 0
    for u, v, w in edges:
        dist[u][v] = w
        nxt[u][v] = v

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    nxt[i][j] = nxt[i][k]

    return dist, nxt

def get_path(nxt: list, u: int, v: int) -> list:
    """経路復元"""
    if nxt[u][v] is None:
        return []
    path = [u]
    while u != v:
        u = nxt[u][v]
        path.append(u)
    return path

# 使用例
edges = [(0,1,3), (0,3,7), (1,0,8), (1,2,2), (2,0,5), (2,3,1), (3,0,2)]
dist, nxt = floyd_warshall_with_path(4, edges)
print(dist[1][3])           # 3
print(get_path(nxt, 1, 3))  # [1, 2, 3]
```

### 4.3 Floyd-Warshall による負閉路検出

```python
def floyd_warshall_with_negative_cycle(n: int, edges: list) -> tuple:
    """Floyd-Warshall + 負閉路検出"""
    INF = float('inf')
    dist = [[INF] * n for _ in range(n)]

    for i in range(n):
        dist[i][i] = 0
    for u, v, w in edges:
        dist[u][v] = min(dist[u][v], w)

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] != INF and dist[k][j] != INF:
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]

    # 対角成分が負 → 負閉路が存在
    has_negative_cycle = any(dist[i][i] < 0 for i in range(n))

    return dist, has_negative_cycle
```

### 4.4 Floyd-Warshall の応用: 推移閉包

```python
def transitive_closure(n: int, edges: list) -> list:
    """推移閉包: i から j に到達可能かどうかの行列を求める
    Floyd-Warshall の変形（重みの代わりに到達可能性を管理）
    """
    reach = [[False] * n for _ in range(n)]

    for i in range(n):
        reach[i][i] = True
    for u, v in edges:
        reach[u][v] = True

    for k in range(n):
        for i in range(n):
            for j in range(n):
                reach[i][j] = reach[i][j] or (reach[i][k] and reach[k][j])

    return reach

edges_reach = [(0, 1), (1, 2), (2, 3)]
reach = transitive_closure(4, edges_reach)
print(reach[0][3])  # True (0→1→2→3)
print(reach[3][0])  # False (3から0へは到達不可)
```

---

## 5. DAG 最短経路

DAG（有向非巡回グラフ）では、トポロジカル順序で緩和すれば O(V+E) で求まる。

```python
from collections import defaultdict, deque

def dag_shortest_path(graph: dict, start) -> dict:
    """DAG上の最短経路 - O(V + E)"""
    # トポロジカルソート（Kahn）
    in_degree = defaultdict(int)
    all_verts = set(graph.keys())
    for u in graph:
        for v, _ in graph[u]:
            in_degree[v] += 1
            all_verts.add(v)

    queue = deque([v for v in all_verts if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for neighbor, _ in graph.get(v, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # トポロジカル順に緩和
    dist = {v: float('inf') for v in all_verts}
    dist[start] = 0
    for u in topo_order:
        if dist[u] != float('inf'):
            for v, w in graph.get(u, []):
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w

    return dist

dag = {
    'A': [('B', 2), ('C', 6)],
    'B': [('C', 1), ('D', 3)],
    'C': [('D', 1)],
    'D': [],
}
print(dag_shortest_path(dag, 'A'))  # {'A': 0, 'B': 2, 'C': 3, 'D': 4}
```

### DAG 最長経路

DAG では重みを反転させるか、max を使って最長経路も O(V+E) で求められる。クリティカルパス分析に使われる。

```python
def dag_longest_path(graph: dict, start) -> dict:
    """DAG上の最長経路 - O(V + E)
    プロジェクト管理のクリティカルパス分析に使用
    """
    in_degree = defaultdict(int)
    all_verts = set(graph.keys())
    for u in graph:
        for v, _ in graph[u]:
            in_degree[v] += 1
            all_verts.add(v)

    queue = deque([v for v in all_verts if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for neighbor, _ in graph.get(v, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    dist = {v: float('-inf') for v in all_verts}
    dist[start] = 0

    for u in topo_order:
        if dist[u] != float('-inf'):
            for v, w in graph.get(u, []):
                if dist[u] + w > dist[v]:
                    dist[v] = dist[u] + w

    return dist

# プロジェクトタスクのDAG（タスク名: [(次タスク, 所要日数)])
project = {
    '設計':     [('開発', 5), ('テスト設計', 3)],
    '開発':     [('テスト', 8)],
    'テスト設計': [('テスト', 2)],
    'テスト':   [('リリース', 3)],
    'リリース': [],
}
longest = dag_longest_path(project, '設計')
print(longest)
# {'設計': 0, '開発': 5, 'テスト設計': 3, 'テスト': 13, 'リリース': 16}
# クリティカルパス: 設計→開発→テスト→リリース = 16日
```

---

## 6. Johnson法（全点対最短経路 — 疎グラフ向け）

Bellman-Ford で辺の重みを非負に変換し、各頂点から Dijkstra を実行する。疎グラフでは Floyd-Warshall より高速。

```python
def johnson(n: int, edges: list) -> list:
    """Johnson法 - O(V² log V + VE)
    edges: [(u, v, weight), ...]
    """
    INF = float('inf')

    # Step 1: 仮想頂点 s を追加し、全頂点へ重み 0 の辺を張る
    new_edges = edges + [(n, v, 0) for v in range(n)]

    # Step 2: Bellman-Ford で s からの最短距離 h を求める
    h = [INF] * (n + 1)
    h[n] = 0
    for _ in range(n):
        for u, v, w in new_edges:
            if h[u] != INF and h[u] + w < h[v]:
                h[v] = h[u] + w

    # 負閉路チェック
    for u, v, w in new_edges:
        if h[u] != INF and h[u] + w < h[v]:
            raise ValueError("負閉路が存在します")

    # Step 3: 辺の重みを非負に変換: w'(u,v) = w(u,v) + h[u] - h[v]
    reweighted_graph = defaultdict(list)
    for u, v, w in edges:
        new_w = w + h[u] - h[v]
        reweighted_graph[u].append((v, new_w))

    # Step 4: 各頂点から Dijkstra
    dist = [[INF] * n for _ in range(n)]
    for s in range(n):
        d = dijkstra_array(reweighted_graph, s, n)
        for t in range(n):
            if d[t] != INF:
                # 元の重みに戻す
                dist[s][t] = d[t] - h[s] + h[t]

    return dist

def dijkstra_array(graph, start, n):
    """配列版 Dijkstra（Johnson用）"""
    INF = float('inf')
    dist = [INF] * n
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph.get(u, []):
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))
    return dist
```

---

## 7. A* 探索

Dijkstra にヒューリスティック関数を加えた探索。ゴールが明確な場合に Dijkstra より高速に最短経路を見つけられる。

```
Dijkstra: f(n) = g(n)        ← 始点からの実コスト
A*:       f(n) = g(n) + h(n) ← 実コスト + ゴールまでの推定コスト

ヒューリスティック h(n) の条件:
  - 許容性 (Admissible): h(n) <= 実際のコスト（過大評価しない）
  - 一貫性 (Consistent): h(n) <= cost(n, n') + h(n')

  許容的なら最適解が保証される
```

```python
def astar(graph: dict, start, goal, heuristic) -> tuple:
    """A* 探索
    graph: {u: [(v, weight), ...]}
    heuristic: h(node) → ゴールまでの推定コスト
    返り値: (距離, 経路)
    """
    open_set = [(heuristic(start), 0, start)]  # (f, g, node)
    came_from = {}
    g_score = defaultdict(lambda: float('inf'))
    g_score[start] = 0
    closed = set()

    while open_set:
        f, g, current = heapq.heappop(open_set)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return g, path[::-1]

        if current in closed:
            continue
        closed.add(current)

        for neighbor, weight in graph.get(current, []):
            if neighbor in closed:
                continue
            tentative_g = g + weight
            if tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                f = tentative_g + heuristic(neighbor)
                heapq.heappush(open_set, (f, tentative_g, neighbor))

    return float('inf'), []

# グリッド上の A* 探索例（マンハッタン距離をヒューリスティックに）
def manhattan_distance(node, goal=(9, 9)):
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

# グリッドグラフの構築
def build_grid_graph(rows, cols, blocked=set()):
    graph = {}
    for r in range(rows):
        for c in range(cols):
            if (r, c) in blocked:
                continue
            neighbors = []
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in blocked:
                    neighbors.append(((nr, nc), 1))
            graph[(r, c)] = neighbors
    return graph

grid_graph = build_grid_graph(10, 10, blocked={(3,3), (3,4), (3,5)})
dist, path = astar(grid_graph, (0, 0), (9, 9), manhattan_distance)
print(f"最短距離: {dist}")  # 18
print(f"経路長: {len(path)}")
```

### A* vs Dijkstra の比較

```
          Dijkstra                    A*
        ┌─────────┐             ┌─────────┐
        │ ○ ○ ○ ○ │             │ . . . . │
        │ ○ ○ ○ ○ │             │ . ○ ○ . │
     S → ○ ○ ○ ○ → G         S → . ○ ○ → G
        │ ○ ○ ○ ○ │             │ . ○ ○ . │
        │ ○ ○ ○ ○ │             │ . . . . │
        └─────────┘             └─────────┘
     探索頂点数: 多い          探索頂点数: 少ない

  ○ = 探索された頂点
  . = 探索されなかった頂点

  A* は h(n) によってゴール方向に探索を集中させるため、
  大規模なグラフで大幅に高速になる
```

---

## 8. 特殊な最短経路アルゴリズム

### 8.1 0-1 BFS

辺の重みが 0 か 1 のみの場合、deque を使って O(V+E) で最短距離を求める。

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
```

### 8.2 双方向 Dijkstra

始点と終点の両方から Dijkstra を行い、探索領域が重なった時点で最短距離を確定させる。大規模グラフでの1対1最短経路に有効。

```python
def bidirectional_dijkstra(graph: dict, reverse_graph: dict, start, end) -> int:
    """双方向 Dijkstra
    graph: 順方向の隣接リスト
    reverse_graph: 逆方向の隣接リスト
    """
    INF = float('inf')

    dist_f = defaultdict(lambda: INF)
    dist_b = defaultdict(lambda: INF)
    dist_f[start] = 0
    dist_b[end] = 0

    pq_f = [(0, start)]
    pq_b = [(0, end)]

    visited_f = set()
    visited_b = set()

    best = INF

    while pq_f or pq_b:
        # 前方から探索
        if pq_f:
            d, u = heapq.heappop(pq_f)
            if d <= best:
                visited_f.add(u)
                for v, w in graph.get(u, []):
                    new_d = d + w
                    if new_d < dist_f[v]:
                        dist_f[v] = new_d
                        heapq.heappush(pq_f, (new_d, v))
                    if v in visited_b:
                        best = min(best, dist_f[v] + dist_b[v])

        # 後方から探索
        if pq_b:
            d, u = heapq.heappop(pq_b)
            if d <= best:
                visited_b.add(u)
                for v, w in reverse_graph.get(u, []):
                    new_d = d + w
                    if new_d < dist_b[v]:
                        dist_b[v] = new_d
                        heapq.heappush(pq_b, (new_d, v))
                    if v in visited_f:
                        best = min(best, dist_f[v] + dist_b[v])

        # 両方の最小距離が best 以上なら終了
        min_f = pq_f[0][0] if pq_f else INF
        min_b = pq_b[0][0] if pq_b else INF
        if min_f + min_b >= best:
            break

    return best
```

### 8.3 ダイヤル法（Dial's Algorithm）

辺の重みが小さな非負整数の場合、バケットを使って O(V + E + W_max) で動作する。

```python
def dial_shortest_path(graph: dict, start, n: int, max_weight: int) -> list:
    """ダイヤル法 - O(V + E + W_max * V)
    辺の重みが [0, max_weight] の非負整数の場合に有効
    """
    INF = float('inf')
    dist = [INF] * n
    dist[start] = 0

    # バケット（循環バッファ）
    num_buckets = max_weight * n + 1
    buckets = [[] for _ in range(num_buckets)]
    buckets[0].append(start)

    idx = 0
    found = 0

    while found < n:
        while not buckets[idx % num_buckets]:
            idx += 1

        u = buckets[idx % num_buckets].pop()
        if dist[u] != idx:
            continue

        found += 1

        for v, w in graph.get(u, []):
            new_dist = dist[u] + w
            if new_dist < dist[v]:
                dist[v] = new_dist
                buckets[new_dist % num_buckets].append(v)

    return dist
```

---

## 9. 実務応用パターン

### 9.1 カーナビの経路検索

```python
# 実際のカーナビでは以下の手法が使われる:
#
# 1. Contraction Hierarchies (CH)
#    - 前処理で重要度の低い頂点を「収縮」し、ショートカット辺を追加
#    - クエリは双方向 Dijkstra で、前処理済みグラフ上で高速に探索
#    - 前処理: O(n log n) 〜 O(n²), クエリ: 数ミリ秒
#
# 2. ALT (A*, Landmarks, Triangle inequality)
#    - ランドマーク頂点への距離を前処理で計算
#    - 三角不等式を使ってヒューリスティックを計算
#    - A* のヒューリスティックとして使用

def travel_time_dijkstra(road_network, start, end, departure_time):
    """時間依存の経路検索（ラッシュアワーを考慮）"""
    dist = defaultdict(lambda: float('inf'))
    dist[start] = departure_time
    pq = [(departure_time, start)]

    while pq:
        time, u = heapq.heappop(pq)
        if u == end:
            return time - departure_time
        if time > dist[u]:
            continue
        for v, travel_time_func in road_network.get(u, []):
            # travel_time_func(t) は時刻 t における移動時間を返す
            arrival = time + travel_time_func(time)
            if arrival < dist[v]:
                dist[v] = arrival
                heapq.heappush(pq, (arrival, v))
    return float('inf')
```

### 9.2 ネットワークルーティング

```python
def ospf_routing(network: dict, router_id: str) -> dict:
    """OSPF（Open Shortest Path First）のルーティングテーブル計算
    実際の OSPF は Dijkstra 法をベースにしている
    """
    dist, prev = dijkstra(network, router_id)

    routing_table = {}
    for dest in dist:
        if dest == router_id:
            continue
        # 次ホップを求める
        next_hop = dest
        while prev.get(next_hop) != router_id:
            next_hop = prev.get(next_hop)
            if next_hop is None:
                break
        routing_table[dest] = {
            'next_hop': next_hop,
            'cost': dist[dest],
            'path': reconstruct_path(prev, router_id, dest),
        }

    return routing_table
```

### 9.3 為替レートの裁定取引検出

```python
import math

def detect_arbitrage(currencies: list, rates: dict) -> list:
    """為替レートの裁定取引を検出（Bellman-Ford で負閉路を検出）
    rates: {(from, to): rate, ...}

    裁定取引: A → B → C → A で利益が出る状態
    rates を -log(rate) に変換すると、負閉路 = 裁定取引機会
    """
    n = len(currencies)
    currency_idx = {c: i for i, c in enumerate(currencies)}

    # 辺リストの作成（重み = -log(rate)）
    edges = []
    for (src, dst), rate in rates.items():
        if rate > 0:
            edges.append((currency_idx[src], currency_idx[dst], -math.log(rate)))

    # Bellman-Ford
    dist = [float('inf')] * n
    prev = [None] * n
    dist[0] = 0

    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u

    # 負閉路の検出
    for u, v, w in edges:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            # 負閉路を見つけた → 裁定取引が可能
            # サイクルを復元
            cycle = []
            visited = set()
            x = v
            for _ in range(n):
                x = prev[x]
            start = x
            cycle.append(currencies[start])
            x = prev[start]
            while x != start:
                cycle.append(currencies[x])
                x = prev[x]
            cycle.append(currencies[start])
            return cycle[::-1]

    return []  # 裁定取引なし

currencies = ['USD', 'EUR', 'GBP', 'JPY']
rates = {
    ('USD', 'EUR'): 0.85, ('EUR', 'USD'): 1.20,  # 1.20 > 1/0.85
    ('EUR', 'GBP'): 0.86, ('GBP', 'EUR'): 1.17,
    ('GBP', 'USD'): 1.30, ('USD', 'GBP'): 0.78,
    ('USD', 'JPY'): 110.0, ('JPY', 'USD'): 0.0091,
}
cycle = detect_arbitrage(currencies, rates)
if cycle:
    print(f"裁定取引: {' → '.join(cycle)}")
```

---

## 10. アルゴリズム比較表

| アルゴリズム | 計算量 | 負辺 | 負閉路検出 | 種別 | 用途 |
|:---|:---|:---|:---|:---|:---|
| Dijkstra | O((V+E) log V) | 不可 | 不可 | 単一始点 | 非負辺グラフ |
| Bellman-Ford | O(VE) | 可 | 可 | 単一始点 | 負辺あり |
| SPFA | O(VE) 最悪 | 可 | 可 | 単一始点 | BFの改良版 |
| Floyd-Warshall | O(V³) | 可 | 可(対角負) | 全点対 | 小規模・密グラフ |
| Johnson | O(V² log V + VE) | 可 | 可 | 全点対 | 疎グラフ |
| DAG 最短経路 | O(V+E) | 可 | 不要(DAG) | 単一始点 | DAG限定 |
| BFS | O(V+E) | 不可(重み=1) | 不可 | 単一始点 | 重みなし |
| 0-1 BFS | O(V+E) | 不可(重み0/1) | 不可 | 単一始点 | 重み0or1 |
| A* | O(E) 最良 | 不可 | 不可 | 1対1 | ヒューリスティック可能 |

## 使い分けガイド

```
最短経路アルゴリズムの選択フロー:

  辺の重みは全て同じ(or なし)?
    ├─ YES → BFS  O(V+E)
    └─ NO  → 辺の重みは 0 or 1 のみ?
              ├─ YES → 0-1 BFS  O(V+E)
              └─ NO  → 負の辺がある?
                        ├─ NO  → DAG?
                        │        ├─ YES → DAGトポ順  O(V+E)
                        │        └─ NO  → ゴールが明確?
                        │                  ├─ YES → A*
                        │                  └─ NO  → Dijkstra  O((V+E)logV)
                        └─ YES → 全点対が必要?
                                  ├─ YES → 密? → Floyd  O(V³)
                                  │        疎? → Johnson  O(V²logV+VE)
                                  └─ NO  → Bellman-Ford  O(VE)
```

| 条件 | 推奨 | 理由 |
|:---|:---|:---|
| 非負辺 + 単一始点 | Dijkstra | 最速 |
| 負辺あり + 単一始点 | Bellman-Ford | 負辺対応 |
| 全頂点対の最短距離 | Floyd-Warshall | 簡潔な実装 |
| DAG | トポロジカル順緩和 | O(V+E)で最速 |
| 辺重み全て1 | BFS | O(V+E) |
| 疎グラフ + 全点対 | Johnson | Dijkstra V回 |
| 明確なゴール | A* | ヒューリスティックで高速化 |
| 辺重み 0/1 | 0-1 BFS | deque で O(V+E) |

---

## 11. アンチパターン

### アンチパターン1: 負の辺にDijkstraを使う

```python
# BAD: 負辺のあるグラフにDijkstra
graph_neg = {
    'A': [('B', 1), ('C', 4)],
    'B': [('C', -3)],   # 負辺!
    'C': [],
}
# Dijkstra は A→C=4 を確定してしまうが、
# 実際は A→B→C = 1+(-3) = -2 が最短

# GOOD: Bellman-Ford を使う
vertices = ['A', 'B', 'C']
edges = [('A','B',1), ('A','C',4), ('B','C',-3)]
dist, _, _ = bellman_ford(vertices, edges, 'A')
# dist['C'] = -2 (正しい)
```

### アンチパターン2: Floyd-Warshallを大規模グラフに使う

```python
# BAD: V=10000 のグラフに Floyd-Warshall
# → O(V³) = 10^12 回の演算 → 数時間かかる

# GOOD: 単一始点なら Dijkstra を使う
# 全点対が本当に必要か再検討する
# 疎グラフなら Johnson のアルゴリズムを検討
```

### アンチパターン3: 距離更新時にヒープの古いエントリを放置

```python
# Dijkstra で距離更新時、古いエントリがヒープに残る
# → heappop 時に dist[u] との比較でスキップが必要

# BAD: スキップ処理を忘れる
def bad_dijkstra(graph, start):
    dist = defaultdict(lambda: float('inf'))
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        # スキップなし → 古い情報で無駄な処理
        for v, w in graph[u]:
            ...

# GOOD: 古いエントリをスキップ
def good_dijkstra(graph, start):
    dist = defaultdict(lambda: float('inf'))
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:  # 古い情報 → スキップ
            continue
        for v, w in graph[u]:
            ...
```

### アンチパターン4: A* のヒューリスティックが非許容的

```python
# BAD: ヒューリスティックが実際のコストを過大評価 → 最適解が保証されない
def bad_heuristic(node, goal):
    # ユークリッド距離 × 2 → 過大評価
    return 2 * math.sqrt((node[0]-goal[0])**2 + (node[1]-goal[1])**2)

# GOOD: 許容的なヒューリスティック
def good_heuristic(node, goal):
    # マンハッタン距離（グリッドでは許容的）
    return abs(node[0]-goal[0]) + abs(node[1]-goal[1])
```

### アンチパターン5: 到達不可能な頂点の処理忘れ

```python
# BAD: 到達不可能な場合のチェックなし
def bad_path_reconstruction(prev, start, end):
    path = [end]
    current = end
    while current != start:
        current = prev[current]  # KeyError の可能性!
        path.append(current)
    return path[::-1]

# GOOD: 到達可能性をチェック
def good_path_reconstruction(prev, start, end):
    if end not in prev and end != start:
        return []  # 到達不可能
    path = [end]
    current = end
    while current != start:
        if current not in prev:
            return []
        current = prev[current]
        path.append(current)
    return path[::-1]
```

---

## 12. 計算量の詳細分析

### Dijkstra法の計算量の導出

```
優先度キュー（二分ヒープ）を使用した場合:

  操作                    回数          1回あたり    合計
  ─────────────────────────────────────────────────────
  heappush (初期化)       1             O(1)         O(1)
  heappop                 最大 V+E 回   O(log(V+E))  O((V+E) log V)
  heappush (緩和時)       最大 E 回     O(log(V+E))  O(E log V)
  ─────────────────────────────────────────────────────
  合計                                               O((V+E) log V)

  注: ヒープサイズは最大 V+E（古いエントリが残るため）
  log(V+E) = O(log V) （E ≤ V² なので log(V+E) ≤ log(V² + V) ≈ 2 log V）

  フィボナッチヒープを使った場合:
  - decrease-key が O(1)（償却）になるため
  - 合計 O(V log V + E)
  - 理論的には高速だが、実装が複雑で定数係数が大きい
```

### Floyd-Warshall の空間最適化

```python
# 標準版: O(V²) 空間（k の次元は不要 — 上書きしても正しい）
# 理由: dist[i][k] と dist[k][j] は k 番目の反復で変化しないため

# 注意: 経路復元が不要なら dist 行列だけで十分
# 経路復元が必要なら nxt 行列も必要で、合計 O(V²) 空間
```

---

## 13. FAQ

### Q1: Dijkstra法でなぜ負辺が扱えないのか？

**A:** Dijkstra法は「一度確定した最短距離は変わらない」という貪欲な仮定に基づく。負辺があると、後から見つかる経路が既に確定した距離より短くなりうるため、この仮定が崩れる。例: A→C(重み4)を確定した後に A→B→C(1+(-3)=-2) が見つかっても修正できない。

### Q2: ダイクストラ法の計算量 O((V+E) log V) はどこから来る？

**A:** 各頂点は最大1回ヒープから取り出され(V回のheappop = V log V)、各辺は最大1回の緩和でヒープへの挿入を引き起こす(E回のheappush = E log V)。合計 O((V+E) log V)。フィボナッチヒープを使うと O(V log V + E) に改善できるが、実装が複雑なため実用では二分ヒープが主流。

### Q3: 経路復元はどう実装する？

**A:** 「前任者（predecessor）」を記録する。各頂点について「どの頂点から来たか」を保存し、ゴールからスタートまで辿ることで経路を復元する。上記の `reconstruct_path` 関数を参照。

### Q4: 0-1 BFS とは何か？

**A:** 辺重みが0か1のグラフに特化した最短経路アルゴリズム。通常のキューの代わりにデック（両端キュー）を使い、重み0の辺は先頭に、重み1の辺は末尾に追加する。O(V+E)で動作する。壁を壊してグリッドを移動するような問題に有効。

### Q5: A*探索のヒューリスティック関数はどう選ぶ？

**A:** グリッド上の問題では、4方向移動ならマンハッタン距離、8方向移動ならチェビシェフ距離、自由移動ならユークリッド距離が一般的。重要なのは「許容性」（真の距離を過大評価しない）を満たすこと。許容的でないヒューリスティックは高速だが最適解を見逃す可能性がある。

### Q6: 負閉路があるとなぜ最短経路が定義できないのか？

**A:** 負閉路を何周もすれば距離を無限に小さくできるため、「最短」が定義できない。具体的には、始点から負閉路に到達でき、かつ負閉路から終点に到達できる場合、最短距離は -∞ になる。Bellman-Ford はこの状況を V 回目の緩和で検出する。

---

## 14. まとめ

| 項目 | 要点 |
|:---|:---|
| Dijkstra法 | 非負辺の単一始点最短経路。優先度キューで O((V+E) log V) |
| Bellman-Ford法 | 負辺を許容。全辺 V-1 回緩和で O(VE)。負閉路検出可能 |
| Floyd-Warshall法 | 全点対最短経路。DP で O(V³)。小規模グラフ向け |
| Johnson法 | 疎グラフの全点対最短経路。BF + Dijkstra V回 |
| DAG最短経路 | トポロジカル順の緩和で O(V+E)。DAG限定 |
| A*探索 | ヒューリスティック付き Dijkstra。1対1の最短経路に最適 |
| 緩和操作 | 全アルゴリズムに共通する基本操作 |
| 経路復元 | 前任者配列を使ってゴールから逆順に辿る |

---

## 次に読むべきガイド

- [グラフ走査](./02-graph-traversal.md) -- BFS/DFS の基礎（最短経路の前提知識）
- [動的計画法](./04-dynamic-programming.md) -- Floyd-Warshall の背景にある DP の考え方
- [貪欲法](./05-greedy.md) -- Dijkstra 法の背景にある貪欲戦略
- [ネットワークフロー](../03-advanced/03-network-flow.md) -- グラフ最適化の発展

---

## 参考文献

1. Cormen, T. H. et al. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. -- 第22-25章
2. Dijkstra, E. W. (1959). "A note on two problems in connexion with graphs." *Numerische Mathematik*.
3. Bellman, R. (1958). "On a routing problem." *Quarterly of Applied Mathematics*.
4. Floyd, R. W. (1962). "Algorithm 97: Shortest Path." *Communications of the ACM*.
5. Johnson, D. B. (1977). "Efficient algorithms for shortest paths in sparse networks." *Journal of the ACM*.
6. Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). "A Formal Basis for the Heuristic Determination of Minimum Cost Paths." *IEEE Transactions on Systems Science and Cybernetics*.
