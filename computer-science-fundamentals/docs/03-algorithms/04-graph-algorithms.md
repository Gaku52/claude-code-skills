# グラフアルゴリズム

> SNSの友達関係、地図の経路探索、Webのリンク構造——世界はグラフで満ちている。

## この章で学ぶこと

- [ ] グラフの基本用語と表現方法を理解する
- [ ] BFS/DFSの違いと使い分けを説明できる
- [ ] 最短経路アルゴリズム（ダイクストラ）を実装できる

## 前提知識

- 計算量解析 → 参照: [[01-complexity-analysis.md]]
- キューとスタック → 参照: [[../04-data-structures/02-stacks-and-queues.md]]

---

## 1. グラフの基礎

### 1.1 用語と表現

```
グラフの基本用語:

  グラフ G = (V, E)
  V = 頂点（vertex/node）の集合
  E = 辺（edge）の集合

  種類:
  ┌──────────────┬────────────────────────────────┐
  │ 無向グラフ    │ 辺に方向なし: A—B             │
  │ 有向グラフ    │ 辺に方向あり: A→B              │
  │ 重み付きグラフ │ 辺にコスト: A—(5)—B          │
  │ DAG          │ 有向非巡回グラフ（トポロジカルソート可能）│
  └──────────────┴────────────────────────────────┘

  データ構造での表現:

  1. 隣接リスト（Adjacency List）— 一般的
     graph = {
       'A': ['B', 'C'],
       'B': ['A', 'D'],
       'C': ['A', 'D'],
       'D': ['B', 'C']
     }
     空間: O(V + E)
     辺の存在確認: O(degree)

  2. 隣接行列（Adjacency Matrix）— 密なグラフ向け
     #    A  B  C  D
     # A [0, 1, 1, 0]
     # B [1, 0, 0, 1]
     # C [1, 0, 0, 1]
     # D [0, 1, 1, 0]
     空間: O(V²)
     辺の存在確認: O(1)
```

### 1.2 グラフの実世界での例

```
グラフの応用例:

  ┌──────────────────┬──────────────┬──────────────────┐
  │ 応用             │ 頂点          │ 辺              │
  ├──────────────────┼──────────────┼──────────────────┤
  │ SNS             │ ユーザー      │ 友達関係         │
  │ Web             │ ページ        │ ハイパーリンク   │
  │ 地図            │ 交差点        │ 道路             │
  │ ネットワーク    │ ルーター      │ 回線             │
  │ 依存関係        │ パッケージ    │ 依存             │
  │ スケジューリング │ タスク        │ 先行制約         │
  │ 推薦システム    │ ユーザー+商品 │ 購入/評価        │
  └──────────────────┴──────────────┴──────────────────┘
```

---

## 2. グラフ探索

### 2.1 BFS（幅優先探索）

```python
from collections import deque

def bfs(graph, start):
    """層ごとに探索（近い順）"""
    visited = set([start])
    queue = deque([start])
    order = []

    while queue:
        node = queue.popleft()  # FIFO
        order.append(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return order

# 計算量: O(V + E)
# 空間: O(V)

# BFS の特性:
# - 始点から近い順に探索（レベルオーダー）
# - 重みなしグラフの最短経路を求められる
# - キュー（FIFO）を使用

# 最短経路の復元:
def bfs_shortest_path(graph, start, end):
    visited = {start: None}  # node → 前のnode
    queue = deque([start])

    while queue:
        node = queue.popleft()
        if node == end:
            # 経路を復元
            path = []
            while node is not None:
                path.append(node)
                node = visited[node]
            return path[::-1]

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited[neighbor] = node
                queue.append(neighbor)

    return None  # 到達不可能
```

### 2.2 DFS（深さ優先探索）

```python
def dfs_iterative(graph, start):
    """できるだけ深く探索してから戻る"""
    visited = set()
    stack = [start]
    order = []

    while stack:
        node = stack.pop()  # LIFO
        if node not in visited:
            visited.add(node)
            order.append(node)
            for neighbor in reversed(graph[node]):
                if neighbor not in visited:
                    stack.append(neighbor)

    return order

def dfs_recursive(graph, node, visited=None):
    """再帰版DFS"""
    if visited is None:
        visited = set()
    visited.add(node)
    print(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited)

# 計算量: O(V + E)
# 空間: O(V) — スタック/再帰の深さ

# DFS の特性:
# - できるだけ深く進む
# - スタック（LIFO）を使用（再帰 = 暗黙のスタック）
# - サイクル検出、トポロジカルソート、連結成分に適する
```

### 2.3 BFS vs DFS

```
BFS vs DFS の使い分け:

  ┌──────────────────┬──────────────┬──────────────┐
  │ 用途             │ BFS          │ DFS          │
  ├──────────────────┼──────────────┼──────────────┤
  │ 最短経路（重みなし）│ ✅ 最適     │ ❌           │
  │ サイクル検出     │ 可能         │ ✅ 自然      │
  │ トポロジカルソート│ 可能(Kahn)  │ ✅ 自然      │
  │ 連結成分         │ 可能         │ ✅ 自然      │
  │ 二部グラフ判定   │ ✅ 自然      │ 可能         │
  │ メモリ使用量     │ O(幅)        │ O(深さ)      │
  │ 迷路の全探索     │ 可能         │ ✅ 自然      │
  └──────────────────┴──────────────┴──────────────┘

  メモリ比較:
  - 木の幅が広い場合: DFSが有利（深さ分のメモリ）
  - 木の深さが深い場合: BFSが有利（幅分のメモリ）
  - バランスの取れた木: ほぼ同等
```

---

## 3. 最短経路アルゴリズム

### 3.1 ダイクストラ法

```python
import heapq

def dijkstra(graph, start):
    """重み付きグラフの単一始点最短経路（非負の重みのみ）"""
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]  # (距離, ノード)

    while pq:
        dist, node = heapq.heappop(pq)

        if dist > distances[node]:
            continue  # 既により短い経路が見つかっている

        for neighbor, weight in graph[node]:
            new_dist = dist + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))

    return distances

# 使用例:
graph = {
    'A': [('B', 4), ('C', 2)],
    'B': [('D', 3), ('C', 1)],
    'C': [('B', 1), ('D', 5)],
    'D': []
}
print(dijkstra(graph, 'A'))
# {'A': 0, 'B': 3, 'C': 2, 'D': 6}
# A→C(2)→B(3)→D(6) が最短

# 計算量: O((V + E) log V) — 優先度キュー使用時
# 制約: 負の重みの辺があると正しく動作しない
```

### 3.2 その他の最短経路アルゴリズム

```
最短経路アルゴリズムの比較:

  ┌──────────────┬──────────────────┬──────────┬──────────┐
  │ アルゴリズム  │ 計算量           │ 負の辺   │ 用途     │
  ├──────────────┼──────────────────┼──────────┼──────────┤
  │ BFS          │ O(V + E)         │ 不可     │ 重みなし │
  │ ダイクストラ  │ O((V+E) log V)   │ 不可     │ 非負重み │
  │ ベルマンフォード│ O(V × E)       │ 可能     │ 負の辺あり│
  │ フロイドワーシャル│ O(V³)        │ 可能     │ 全対間   │
  │ A*           │ O(E) 期待        │ 不可     │ ヒューリスティック│
  └──────────────┴──────────────────┴──────────┴──────────┘

  実務での選択:
  - 地図アプリ: A*（ダイクストラ + ヒューリスティック）
  - ネットワーク: ダイクストラ（OSPF ルーティング）
  - 全頂点間: フロイドワーシャル
```

---

## 4. その他の重要なグラフアルゴリズム

### 4.1 トポロジカルソート

```python
def topological_sort(graph):
    """DAGの順序付け（全ての辺が左→右になる順序）"""
    in_degree = {node: 0 for node in graph}
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1

    queue = deque([n for n in in_degree if in_degree[n] == 0])
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(order) != len(graph):
        raise ValueError("グラフにサイクルあり")
    return order

# 用途:
# - ビルドシステム（Make, npm install）の依存解決
# - タスクスケジューリング
# - コンパイラの命令スケジューリング
# - 大学の科目履修順序
```

### 4.2 Union-Find（素集合データ構造）

```python
class UnionFind:
    """連結成分の管理 — 経路圧縮 + ランク付き"""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 経路圧縮
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

# 計算量: ほぼ O(1)（償却、逆アッカーマン関数 α(n)）
# 用途: クラスカル法(MST)、連結成分、等価クラス判定
```

---

## 5. 実践演習

### 演習1: BFS/DFS（基礎）
迷路を2次元配列で表現し、BFSで最短経路を、DFSで全ての経路を求めるプログラムを実装せよ。

### 演習2: ダイクストラ法（応用）
重み付きグラフで最短経路を求め、経路自体も復元するプログラムを実装せよ。

### 演習3: SNS分析（発展）
友達関係のグラフから「友達の友達」を推薦するアルゴリズム、連結成分（コミュニティ）を検出するアルゴリズムを実装せよ。

---

## FAQ

### Q1: Google Mapsの経路探索はダイクストラですか？
**A**: A*アルゴリズムがベース。A*はダイクストラにヒューリスティック（目的地への推定距離）を加えたもの。さらにContraction Hierarchies等の前処理で大幅に高速化。大陸規模の経路をミリ秒で計算。

### Q2: グラフのサイクル検出はどうしますか？
**A**: DFSで「訪問中」のノードに再度到達したらサイクル。有向グラフでは3色（未訪問/訪問中/完了）で管理。無向グラフでは親以外の訪問済みノードに到達したらサイクル。

### Q3: PageRankもグラフアルゴリズムですか？
**A**: はい。WebをWebページ（頂点）とリンク（辺）のグラフとみなし、各ページの「重要度」をリンク構造から計算する。ランダムウォーク（確率的遷移）の定常分布として定式化。Google検索の基盤となったアルゴリズム。

---

## まとめ

| アルゴリズム | 計算量 | 用途 |
|------------|--------|------|
| BFS | O(V+E) | 最短経路(重みなし)、レベル探索 |
| DFS | O(V+E) | サイクル検出、トポソート、連結成分 |
| ダイクストラ | O((V+E)log V) | 最短経路(非負重み) |
| トポロジカルソート | O(V+E) | 依存関係の解決 |
| Union-Find | O(α(n))≈O(1) | 連結成分、MST |

---

## 次に読むべきガイド
→ [[05-dynamic-programming.md]] — 動的計画法

---

## 参考文献
1. Cormen, T. H. et al. "Introduction to Algorithms." Chapters 22-26.
2. Skiena, S. S. "The Algorithm Design Manual." Chapters 5-7.
3. Sedgewick, R. "Algorithms." 4th Edition, Chapter 4.
