# 最短経路アルゴリズム

> 重み付きグラフにおける最短経路問題を、Dijkstra・Bellman-Ford・Floyd-Warshallの3大アルゴリズムで解く手法を体系的に理解する

## この章で学ぶこと

1. **Dijkstra法**の原理と優先度キュー実装で単一始点最短経路を効率的に求められる
2. **Bellman-Ford法**で負の辺重みを扱い、負閉路の検出ができる
3. **Floyd-Warshall法**で全点対最短経路を動的計画法で求められる

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
└───────────────────┴─────────────────────────────────┘
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

### 優先度キュー実装

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
```

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
```

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

# 経路復元付き
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

---

## 6. アルゴリズム比較表

| アルゴリズム | 計算量 | 負辺 | 負閉路検出 | 種別 | 用途 |
|:---|:---|:---|:---|:---|:---|
| Dijkstra | O((V+E) log V) | 不可 | 不可 | 単一始点 | 非負辺グラフ |
| Bellman-Ford | O(VE) | 可 | 可 | 単一始点 | 負辺あり |
| Floyd-Warshall | O(V³) | 可 | 可(対角負) | 全点対 | 小規模・密グラフ |
| DAG 最短経路 | O(V+E) | 可 | 不要(DAG) | 単一始点 | DAG限定 |
| BFS | O(V+E) | 不可(重み=1) | 不可 | 単一始点 | 重みなし |

## 使い分けガイド

| 条件 | 推奨 | 理由 |
|:---|:---|:---|
| 非負辺 + 単一始点 | Dijkstra | 最速 |
| 負辺あり + 単一始点 | Bellman-Ford | 負辺対応 |
| 全頂点対の最短距離 | Floyd-Warshall | 簡潔な実装 |
| DAG | トポロジカル順緩和 | O(V+E)で最速 |
| 辺重み全て1 | BFS | O(V+E) |
| 疎グラフ + 全点対 | Johnson | Dijkstra V回 |

---

## 7. アンチパターン

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

---

## 8. FAQ

### Q1: Dijkstra法でなぜ負辺が扱えないのか？

**A:** Dijkstra法は「一度確定した最短距離は変わらない」という貪欲な仮定に基づく。負辺があると、後から見つかる経路が既に確定した距離より短くなりうるため、この仮定が崩れる。例: A→C(重み4)を確定した後に A→B→C(1+(-3)=-2) が見つかっても修正できない。

### Q2: ダイクストラ法の計算量 O((V+E) log V) はどこから来る？

**A:** 各頂点は最大1回ヒープから取り出され(V回のheappop = V log V)、各辺は最大1回の緩和でヒープへの挿入を引き起こす(E回のheappush = E log V)。合計 O((V+E) log V)。フィボナッチヒープを使うと O(V log V + E) に改善できるが、実装が複雑なため実用では二分ヒープが主流。

### Q3: 経路復元はどう実装する？

**A:** 「前任者（predecessor）」を記録する。各頂点について「どの頂点から来たか」を保存し、ゴールからスタートまで辿ることで経路を復元する。上記の `reconstruct_path` 関数を参照。

### Q4: 0-1 BFS とは何か？

**A:** 辺重みが0か1のグラフに特化した最短経路アルゴリズム。通常のキューの代わりにデック（両端キュー）を使い、重み0の辺は先頭に、重み1の辺は末尾に追加する。O(V+E)で動作する。

---

## 9. まとめ

| 項目 | 要点 |
|:---|:---|
| Dijkstra法 | 非負辺の単一始点最短経路。優先度キューで O((V+E) log V) |
| Bellman-Ford法 | 負辺を許容。全辺 V-1 回緩和で O(VE)。負閉路検出可能 |
| Floyd-Warshall法 | 全点対最短経路。DP で O(V³)。小規模グラフ向け |
| DAG最短経路 | トポロジカル順の緩和で O(V+E)。DAG限定 |
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
