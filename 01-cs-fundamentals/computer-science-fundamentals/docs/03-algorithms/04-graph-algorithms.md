# グラフアルゴリズム

> SNSの友達関係、地図の経路探索、Webのリンク構造——世界はグラフで満ちている。
> グラフは離散数学の中心概念であり、現実世界のほぼ全ての「関係」をモデル化できる汎用的な構造である。

## この章で学ぶこと

- [ ] グラフの基本用語と表現方法（隣接リスト・隣接行列）を理解する
- [ ] BFS/DFSの違いと使い分けを説明できる
- [ ] 最短経路アルゴリズム（ダイクストラ、ベルマンフォード、フロイドワーシャル）を実装できる
- [ ] 最小全域木（クラスカル、プリム）を理解し実装できる
- [ ] トポロジカルソートを依存関係の解決に適用できる
- [ ] 強連結成分分解（Tarjan、Kosaraju）を理解する
- [ ] 二部グラフ判定・サイクル検出など応用問題を解ける

## 前提知識


---

## 1. グラフの基礎

### 1.1 グラフとは何か

グラフとは、**頂点（vertex/node）** の集合 V と、頂点間を結ぶ **辺（edge）** の集合 E の組 G = (V, E) として定義される数学的構造である。配列やツリーと異なり、グラフには「先頭」も「末尾」もない。任意の要素が任意の要素と関係を持てるため、現実世界の複雑な関係を自然にモデル化できる。

なぜグラフが重要なのか。それは、コンピュータサイエンスにおける多くの問題が「グラフ上の問題」として再定式化できるからである。たとえば：

- ソーシャルネットワークでの友達推薦 → グラフ上の距離計算
- カーナビの経路探索 → 重み付きグラフの最短経路問題
- コンパイラの依存解決 → 有向非巡回グラフ（DAG）のトポロジカルソート
- ネットワーク設計の最適化 → 最小全域木問題

### 1.2 基本用語

```
グラフの基本用語:

  グラフ G = (V, E)
  V = 頂点（vertex/node）の集合   |V| = 頂点数
  E = 辺（edge）の集合            |E| = 辺数

  ┌─────────────┬─────────────────────────────────────────────┐
  │ 用語         │ 説明                                        │
  ├─────────────┼─────────────────────────────────────────────┤
  │ 無向グラフ   │ 辺に方向なし: A—B（AからBもBからAも行ける） │
  │ 有向グラフ   │ 辺に方向あり: A→B（AからBのみ行ける）       │
  │ 重み付き     │ 辺にコスト: A—(5)—B（移動コスト5）          │
  │ DAG          │ 有向非巡回グラフ（サイクルなし）             │
  │ 次数(degree) │ 頂点に接続する辺の数                         │
  │ 入次数       │ 有向グラフで頂点に入る辺の数                 │
  │ 出次数       │ 有向グラフで頂点から出る辺の数               │
  │ パス         │ 辺で連結された頂点の列                       │
  │ サイクル     │ 始点と終点が同じパス                         │
  │ 連結         │ 任意の2頂点間にパスが存在する                │
  │ 疎グラフ     │ |E| が |V| に近い（辺が少ない）              │
  │ 密グラフ     │ |E| が |V|^2 に近い（辺が多い）              │
  └─────────────┴─────────────────────────────────────────────┘

  次数の性質（握手定理）:
    無向グラフでは、全頂点の次数の合計 = 2 × |E|
    なぜなら、1本の辺は必ず2つの頂点の次数に寄与するため。
```

### 1.3 グラフの表現方法

グラフをプログラムで扱うためのデータ構造は、大きく分けて2つある。どちらを選ぶかは、グラフの疎密と必要な操作によって決まる。

```
表現方法の比較:

  1. 隣接リスト（Adjacency List）
     各頂点について、隣接する頂点のリストを保持する

        A: [B, C]           A --- B
        B: [A, D]           |     |
        C: [A, D]           C --- D
        D: [B, C]

     空間計算量: O(V + E)   ← 辺の数に比例するため、疎グラフで効率的
     辺の存在確認: O(degree(v))  ← 隣接リストを走査する必要がある
     全隣接頂点の列挙: O(degree(v))  ← リストをそのまま走査

  2. 隣接行列（Adjacency Matrix）
     V×V の行列で、辺の有無を 0/1 で表す

        A  B  C  D
     A [0, 1, 1, 0]         A --- B
     B [1, 0, 0, 1]         |     |
     C [1, 0, 0, 1]         C --- D
     D [0, 1, 1, 0]

     空間計算量: O(V^2)     ← 頂点数の二乗。密グラフでは効率的
     辺の存在確認: O(1)     ← matrix[u][v] を参照するだけ
     全隣接頂点の列挙: O(V) ← 行全体を走査する必要がある

  どちらを選ぶべきか:
  ┌────────────────────────┬──────────────┬──────────────┐
  │ 条件                    │ 隣接リスト   │ 隣接行列     │
  ├────────────────────────┼──────────────┼──────────────┤
  │ 疎グラフ (E << V^2)    │ ★ 推奨      │ メモリ浪費   │
  │ 密グラフ (E ≈ V^2)     │ 使用可       │ ★ 推奨      │
  │ 辺の存在を頻繁に確認   │ O(degree)    │ ★ O(1)      │
  │ 隣接頂点の列挙が多い   │ ★ O(degree) │ O(V)         │
  │ 頂点数が10万以上       │ ★ 推奨      │ メモリ不足   │
  │ フロイドワーシャル使用  │ 変換必要     │ ★ そのまま  │
  └────────────────────────┴──────────────┴──────────────┘

  なぜ多くの場合に隣接リストが選ばれるのか:
  現実のグラフの多くは疎グラフであり（SNSで全員が友達にはならない）、
  隣接リストの方がメモリ効率が良い。頂点数10万のグラフを隣接行列で
  表現すると 10万×10万 = 100億要素が必要になり、現実的でない。
```

### 1.4 Pythonでのグラフ実装

```python
"""
グラフの基本実装 — 隣接リストと隣接行列の両方を提供する。
なぜ両方を実装するのか: 問題の性質によって最適な表現が異なるため、
相互変換できるようにしておくと柔軟に対応できる。
"""

from collections import defaultdict
from typing import Optional


class GraphAdjList:
    """隣接リストによるグラフ実装"""

    def __init__(self, directed: bool = False):
        """
        directed=True: 有向グラフ（辺に方向がある）
        directed=False: 無向グラフ（辺は双方向）
        なぜdefaultdict(list)を使うのか: 存在しないキーへのアクセスで
        自動的に空リストが作られるため、頂点追加の処理が簡潔になる。
        """
        self.graph = defaultdict(list)
        self.directed = directed

    def add_edge(self, u, v, weight: Optional[float] = None):
        """辺を追加する。weightがNoneなら重みなしグラフとして扱う"""
        if weight is not None:
            self.graph[u].append((v, weight))
            if not self.directed:
                self.graph[v].append((u, weight))
        else:
            self.graph[u].append(v)
            if not self.directed:
                self.graph[v].append(u)
        # 頂点だけ登録（辺の先が孤立頂点の可能性に備える）
        if v not in self.graph:
            self.graph[v] = []

    def get_vertices(self):
        """全頂点を返す"""
        return list(self.graph.keys())

    def get_neighbors(self, v):
        """頂点vの隣接頂点を返す"""
        return self.graph[v]

    def __str__(self):
        result = []
        for vertex in sorted(self.graph.keys(), key=str):
            neighbors = self.graph[vertex]
            result.append(f"  {vertex}: {neighbors}")
        return "Graph {\n" + "\n".join(result) + "\n}"


# === 使用例 ===
if __name__ == "__main__":
    # 無向・重みなしグラフ
    g1 = GraphAdjList(directed=False)
    g1.add_edge('A', 'B')
    g1.add_edge('A', 'C')
    g1.add_edge('B', 'D')
    g1.add_edge('C', 'D')
    print("=== 無向グラフ ===")
    print(g1)
    # 出力:
    # Graph {
    #   A: ['B', 'C']
    #   B: ['A', 'D']
    #   C: ['A', 'D']
    #   D: ['B', 'C']
    # }

    # 有向・重み付きグラフ
    g2 = GraphAdjList(directed=True)
    g2.add_edge('A', 'B', 4)
    g2.add_edge('A', 'C', 2)
    g2.add_edge('B', 'D', 3)
    g2.add_edge('C', 'B', 1)
    g2.add_edge('C', 'D', 5)
    print("\n=== 有向・重み付きグラフ ===")
    print(g2)
    # 出力:
    # Graph {
    #   A: [('B', 4), ('C', 2)]
    #   B: [('D', 3)]
    #   C: [('B', 1), ('D', 5)]
    #   D: []
    # }
```

### 1.5 グラフの実世界での例

```
グラフの応用マッピング:

  ┌──────────────────┬──────────────┬──────────────────┬──────────────┐
  │ 応用             │ 頂点          │ 辺              │ グラフの種類 │
  ├──────────────────┼──────────────┼──────────────────┼──────────────┤
  │ SNS             │ ユーザー      │ 友達関係         │ 無向         │
  │ Twitter/X       │ ユーザー      │ フォロー         │ 有向         │
  │ Web             │ ページ        │ ハイパーリンク   │ 有向         │
  │ 地図・道路      │ 交差点        │ 道路             │ 重み付き     │
  │ ネットワーク    │ ルーター      │ 回線             │ 重み付き     │
  │ 依存関係        │ パッケージ    │ 依存             │ 有向(DAG)    │
  │ スケジューリング│ タスク        │ 先行制約         │ 有向(DAG)    │
  │ 推薦システム    │ ユーザー+商品 │ 購入/評価        │ 二部グラフ   │
  │ 電力網          │ 変電所        │ 送電線           │ 重み付き     │
  │ 分子構造        │ 原子          │ 化学結合         │ 無向         │
  └──────────────────┴──────────────┴──────────────────┴──────────────┘
```

---

## 2. グラフ探索アルゴリズム

グラフ探索は、全てのグラフアルゴリズムの基盤となる。BFS（幅優先探索）とDFS（深さ優先探索）の2つは、最も基本的かつ強力な探索手法であり、多くの高度なアルゴリズムがこれらの上に構築されている。

### 2.1 BFS（幅優先探索: Breadth-First Search）

BFSは「始点に近い頂点から順に探索する」アルゴリズムである。なぜ近い順になるのか。それは、キュー（FIFO: 先入先出）を使って探索順序を管理するため、先に発見された（= 始点に近い）頂点が先に処理されるからである。

```
BFSの動作イメージ:

  始点 A からの BFS

  グラフ:          探索順序:
      A
     / \           レベル0: A
    B   C          レベル1: B, C
   / \   \         レベル2: D, E, F
  D   E   F

  キューの変化:
  [A]           → Aを処理、B,Cを追加
  [B, C]        → Bを処理、D,Eを追加
  [C, D, E]     → Cを処理、Fを追加
  [D, E, F]     → D,E,Fを順に処理
  []            → 完了

  なぜBFSが最短経路（重みなし）を求められるのか:
  BFSは始点からの距離（辺の本数）が小さい頂点を先に訪問する。
  ある頂点に初めて到達したとき、それが最短経路となる。
  なぜなら、もしより短い経路があれば、その経路上の頂点は
  先に訪問されているはずで、その頂点を経由して先に到達しているはずだからである。
```

```python
"""
BFS（幅優先探索）の完全実装
- 基本BFS: 全頂点の訪問順序を返す
- 最短経路BFS: 始点から終点への最短経路を復元する
- レベル別BFS: 各レベル（距離）ごとの頂点を返す
"""

from collections import deque


def bfs(graph: dict, start) -> list:
    """
    基本的なBFS。始点から到達可能な全頂点を近い順に返す。

    なぜ visited に追加するタイミングがキューに入れるときなのか:
    キューから取り出すときに visited チェックする方法もあるが、
    キューに入れるときにチェックする方が効率的。
    同じ頂点が複数回キューに入ることを防ぎ、メモリと時間を節約する。

    計算量: O(V + E) — 各頂点を1回、各辺を1回（無向なら2回）処理
    空間計算量: O(V) — visited集合とキューの最大サイズ
    """
    visited = set([start])
    queue = deque([start])
    order = []

    while queue:
        node = queue.popleft()  # O(1) — dequeの先頭取り出し
        order.append(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)    # キューに入れるときに訪問済みにする
                queue.append(neighbor)

    return order


def bfs_shortest_path(graph: dict, start, end) -> list:
    """
    重みなしグラフにおける始点から終点への最短経路を返す。

    なぜ visited を dict にするのか:
    各頂点の「親（どの頂点から来たか）」を記録することで、
    終点から始点へ逆順にたどって経路を復元できる。
    set では「訪問済みかどうか」しか分からないが、
    dict なら「どこから来たか」まで分かる。
    """
    if start == end:
        return [start]

    visited = {start: None}  # node → 前のnode（始点の前はNone）
    queue = deque([start])

    while queue:
        node = queue.popleft()
        if node == end:
            # 経路を復元: 終点から始点へ親をたどる
            path = []
            while node is not None:
                path.append(node)
                node = visited[node]
            return path[::-1]  # 逆順にして始点→終点の順にする

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited[neighbor] = node
                queue.append(neighbor)

    return []  # 到達不可能な場合は空リスト


def bfs_by_level(graph: dict, start) -> list:
    """
    レベル別BFS: 始点からの距離ごとにグループ化して返す。

    なぜレベル別が必要なのか:
    「友達の友達」の推薦や「感染の広がりのシミュレーション」など、
    距離ごとの処理が必要な場面で有用。通常のBFSでは距離情報が失われる。

    実装のポイント: 各レベルの処理開始時にキューの長さを記録し、
    その分だけ取り出すことで、明示的にレベルを区切る。
    """
    visited = set([start])
    queue = deque([start])
    levels = []

    while queue:
        level_size = len(queue)  # 現在のレベルの頂点数
        current_level = []
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        levels.append(current_level)

    return levels


# === 動作確認 ===
if __name__ == "__main__":
    graph = {
        'A': ['B', 'C'],
        'B': ['A', 'D', 'E'],
        'C': ['A', 'F'],
        'D': ['B'],
        'E': ['B', 'F'],
        'F': ['C', 'E']
    }

    print("BFS順序:", bfs(graph, 'A'))
    # 出力: BFS順序: ['A', 'B', 'C', 'D', 'E', 'F']

    print("最短経路 A→F:", bfs_shortest_path(graph, 'A', 'F'))
    # 出力: 最短経路 A→F: ['A', 'C', 'F']

    print("レベル別BFS:", bfs_by_level(graph, 'A'))
    # 出力: レベル別BFS: [['A'], ['B', 'C'], ['D', 'E', 'F']]
```

### 2.2 DFS（深さ優先探索: Depth-First Search）

DFSは「行き止まりに達するまで深く探索し、行き止まりになったら一つ前に戻って別の道を試す」アルゴリズムである。なぜ深く進むのか。それは、スタック（LIFO: 後入先出）を使うため、最後に発見された頂点が先に処理されるからである。再帰呼び出しは暗黙のスタックとして機能するため、DFSは再帰で自然に書ける。

```
DFSの動作イメージ:

  始点 A からの DFS

  グラフ:              探索順序（一例）:
      A
     / \               A → B → D → (戻る) → E → F → (戻る)
    B   C                                     → (戻る) → C
   / \   \
  D   E   F

  スタックの変化:
  [A]           → Aを処理、C,Bをpush（逆順にpushする理由は後述）
  [C, B]        → Bを処理、E,Dをpush
  [C, E, D]     → Dを処理（行き止まり）
  [C, E]        → Eを処理、Fをpush
  [C, F]        → Fを処理
  [C]           → Cを処理
  []            → 完了

  なぜ逆順にpushするのか:
  スタックはLIFOなので、graph[A] = [B, C] のとき
  C, B の順にpushすると B が先にpopされる。
  これにより、隣接リストの順序どおりに探索が進む。
```

```python
"""
DFS（深さ優先探索）の完全実装
- スタックを使った反復版
- 再帰版
- サイクル検出（有向グラフ）
- DFSの行きがけ順/帰りがけ順
"""

from typing import Optional


def dfs_iterative(graph: dict, start) -> list:
    """
    スタックを使ったDFS（反復版）

    なぜ反復版を使うのか:
    再帰版は Python のデフォルト再帰制限（1000）に引っかかる可能性がある。
    大きなグラフ（頂点数1000以上）では反復版が安全。
    sys.setrecursionlimit() で制限を上げることもできるが、
    スタックオーバーフローのリスクがある。

    計算量: O(V + E)
    空間計算量: O(V)
    """
    visited = set()
    stack = [start]
    order = []

    while stack:
        node = stack.pop()  # LIFO: 最後に追加された頂点を先に処理
        if node not in visited:
            visited.add(node)
            order.append(node)
            # 逆順にpushすることで、隣接リストの順序どおりに探索
            for neighbor in reversed(graph[node]):
                if neighbor not in visited:
                    stack.append(neighbor)

    return order


def dfs_recursive(graph: dict, node, visited: Optional[set] = None) -> list:
    """
    再帰版DFS

    なぜ visited のデフォルト値を None にするのか:
    Python の可変デフォルト引数の罠を避けるため。
    デフォルト値を set() にすると、関数呼び出し間で
    同じsetオブジェクトが共有されてしまう。
    None にして関数内で初期化するのが安全なパターン。
    """
    if visited is None:
        visited = set()

    visited.add(node)
    result = [node]

    for neighbor in graph[node]:
        if neighbor not in visited:
            result.extend(dfs_recursive(graph, neighbor, visited))

    return result


def dfs_with_timestamps(graph: dict, start) -> dict:
    """
    行きがけ順（discovery time）と帰りがけ順（finish time）を記録するDFS。

    なぜタイムスタンプが重要なのか:
    - 行きがけ順: トポロジカルソートの基盤
    - 帰りがけ順: 強連結成分分解（Tarjan, Kosaraju）の基盤
    - 区間の包含関係: u が v の先祖 ⟺ discovery[u] < discovery[v] かつ finish[v] < finish[u]

    戻り値: {node: (discovery_time, finish_time)} の辞書
    """
    visited = set()
    timestamps = {}
    time = [0]  # リストにする理由: クロージャ内で変更するため

    def _dfs(node):
        visited.add(node)
        time[0] += 1
        discovery = time[0]

        for neighbor in graph[node]:
            if neighbor not in visited:
                _dfs(neighbor)

        time[0] += 1
        finish = time[0]
        timestamps[node] = (discovery, finish)

    _dfs(start)

    # 未訪問の頂点も処理（非連結グラフ対応）
    for node in graph:
        if node not in visited:
            _dfs(node)

    return timestamps


def detect_cycle_directed(graph: dict) -> bool:
    """
    有向グラフのサイクル検出（3色法）

    なぜ3色が必要なのか:
    - WHITE（未訪問）: まだ探索していない
    - GRAY（訪問中）: 現在のDFSパス上にある（探索中の枝の途中）
    - BLACK（完了）: 全ての子孫の探索が完了した

    GRAYの頂点からGRAYの頂点への辺が見つかった場合、
    それはDFSパス上で「後戻り」していることを意味し、サイクルが存在する。
    BLACKの頂点への辺はサイクルではない（既に探索済みの別の枝）。

    なぜ2色（visited/not visited）では不十分なのか:
    有向グラフでは、A→B, A→C, C→B というグラフで、
    Bは2回訪問されるがサイクルではない。
    「訪問済み」だけでは「現在の探索パス上にある」かどうか区別できない。
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in graph}

    def _dfs(node):
        color[node] = GRAY
        for neighbor in graph[node]:
            if color[neighbor] == GRAY:
                return True   # サイクル検出!
            if color[neighbor] == WHITE:
                if _dfs(neighbor):
                    return True
        color[node] = BLACK
        return False

    # 全頂点から探索（非連結グラフ対応）
    for node in graph:
        if color[node] == WHITE:
            if _dfs(node):
                return True
    return False


# === 動作確認 ===
if __name__ == "__main__":
    # 無向グラフ
    graph = {
        'A': ['B', 'C'],
        'B': ['A', 'D', 'E'],
        'C': ['A', 'F'],
        'D': ['B'],
        'E': ['B', 'F'],
        'F': ['C', 'E']
    }

    print("DFS反復版:", dfs_iterative(graph, 'A'))
    # 出力: DFS反復版: ['A', 'B', 'D', 'E', 'F', 'C']

    print("DFS再帰版:", dfs_recursive(graph, 'A'))
    # 出力: DFS再帰版: ['A', 'B', 'D', 'E', 'F', 'C']

    # 有向グラフ（サイクルあり）
    directed_with_cycle = {
        'A': ['B'],
        'B': ['C'],
        'C': ['A'],  # A→B→C→A のサイクル
    }
    print("サイクルあり:", detect_cycle_directed(directed_with_cycle))
    # 出力: サイクルあり: True

    # 有向グラフ（サイクルなし = DAG）
    dag = {
        'A': ['B', 'C'],
        'B': ['D'],
        'C': ['D'],
        'D': [],
    }
    print("サイクルなし:", detect_cycle_directed(dag))
    # 出力: サイクルなし: False
```

### 2.3 BFS vs DFS 徹底比較

```
BFS vs DFS の使い分け:

  ┌───────────────────────┬──────────────────┬──────────────────┐
  │ 特性                   │ BFS              │ DFS              │
  ├───────────────────────┼──────────────────┼──────────────────┤
  │ データ構造             │ キュー（FIFO）   │ スタック（LIFO） │
  │ 探索順序               │ 近い順（層別）   │ 深い順           │
  │ 最短経路（重みなし）   │ ★ 保証される    │ 保証されない     │
  │ サイクル検出           │ 可能             │ ★ 自然（3色法） │
  │ トポロジカルソート     │ 可能（Kahn法）   │ ★ 自然          │
  │ 連結成分               │ 可能             │ ★ 自然          │
  │ 二部グラフ判定         │ ★ 自然          │ 可能             │
  │ 強連結成分分解         │ 不向き           │ ★ 必須          │
  │ メモリ使用量           │ O(最大レベル幅)  │ O(最大深さ)      │
  │ 完全探索（全経路列挙） │ 不向き           │ ★ 自然          │
  │ 実装の自然さ           │ 反復（キュー）   │ 再帰が自然       │
  └───────────────────────┴──────────────────┴──────────────────┘

  メモリ使用量の具体例:

  完全二分木（深さ d、頂点数 2^(d+1) - 1）の場合:
  - BFS: 最大レベル幅 = 2^d （最下層の頂点数）
  - DFS: 最大深さ = d

  深さ20の完全二分木（約100万頂点）:
  - BFS: 約50万頂点分のメモリ
  - DFS: 約20頂点分のメモリ  ← 圧倒的に省メモリ

  一方、星型グラフ（1つの中心と多数の葉）:
  - BFS: 中心処理後、全葉がキューに入る
  - DFS: スタックに1頂点ずつ → 常にO(1)

  結論: グラフの構造に応じて選択する。
  「迷ったらBFS」が安全だが、再帰で自然に書けるならDFSも有力。
```

---

## 3. 最短経路アルゴリズム

最短経路問題は、グラフアルゴリズムの中で最も実用的な問題の一つである。カーナビ、ネットワークルーティング、物流最適化など、応用範囲は極めて広い。

### 3.1 問題の分類

```
最短経路問題の分類:

  1. 単一始点最短経路（SSSP: Single-Source Shortest Path）
     「ある頂点から他の全頂点への最短距離」
     → ダイクストラ法、ベルマンフォード法

  2. 全対間最短経路（APSP: All-Pairs Shortest Path）
     「全ての頂点ペア間の最短距離」
     → フロイドワーシャル法

  3. 単一ペア最短経路
     「特定の始点から特定の終点への最短距離」
     → A*アルゴリズム（ヒューリスティック付き）

  重みの制約による分類:
  ┌──────────────────┬───────────────────────────────────────┐
  │ 重みの条件        │ 使えるアルゴリズム                      │
  ├──────────────────┼───────────────────────────────────────┤
  │ 重みなし（全て1） │ BFS（最も高速: O(V+E)）               │
  │ 非負の重み        │ ダイクストラ法: O((V+E) log V)        │
  │ 負の重みあり      │ ベルマンフォード法: O(V × E)          │
  │ 負の閉路なし      │ 上記全て使用可                         │
  │ 負の閉路あり      │ 最短経路は定義不能（無限に短くできる）  │
  └──────────────────┴───────────────────────────────────────┘
```

### 3.2 ダイクストラ法（Dijkstra's Algorithm）

ダイクストラ法は、**非負の重みを持つグラフ**において単一始点最短経路を求めるアルゴリズムである。1956年にエドガー・ダイクストラが考案した。

なぜダイクストラ法が正しいのか: 貪欲法の考え方に基づく。「現時点で確定していない頂点のうち、始点からの距離が最小の頂点」を選び、その距離を確定する。非負の重みという条件下では、一度確定した最短距離が後から更新されることはない。なぜなら、未確定の頂点を経由する経路は、その頂点までの距離（現時点の最小値以上）にさらに非負の重みが加算されるため、確定済みの距離を下回ることがないからである。

```
ダイクストラ法の動作ステップ:

  グラフ:
       A --4-- B
       |       |
       2       3
       |       |
       C --1-- B (C→Bは重み1)
       |
       5
       |
       D

  A を始点とする:

  ステップ1: A を確定 (距離0)
    距離: A=0, B=4, C=2, D=inf
    確定: {A}

  ステップ2: C を確定 (距離2)  ← 未確定の中で最小
    C経由でBを更新: 2+1=3 < 4 → B=3
    C経由でDを更新: 2+5=7 < inf → D=7
    距離: A=0, B=3, C=2, D=7
    確定: {A, C}

  ステップ3: B を確定 (距離3)
    B経由でDを更新: 3+3=6 < 7 → D=6
    距離: A=0, B=3, C=2, D=6
    確定: {A, C, B}

  ステップ4: D を確定 (距離6)
    確定: {A, C, B, D}

  最終結果: A=0, B=3, C=2, D=6
```

```python
"""
ダイクストラ法の完全実装
- 基本版（最短距離のみ）
- 経路復元版（最短経路も返す）
- 負の重みがある場合のエラー検出

なぜ優先度キュー（ヒープ）を使うのか:
「未確定頂点の中で距離最小のもの」を効率的に取得するため。
単純な配列だと最小値の探索に O(V) かかるが、
ヒープなら O(log V) で済む。結果的に全体の計算量が
O(V^2) から O((V+E) log V) に改善される。
"""

import heapq
from typing import Optional


def dijkstra(graph: dict, start) -> dict:
    """
    ダイクストラ法: 始点から全頂点への最短距離を返す。

    graph の形式: {node: [(neighbor, weight), ...], ...}
    戻り値: {node: distance, ...}

    計算量: O((V + E) log V)
      - 各頂点は最大1回確定される
      - 各辺について最大1回の緩和（relaxation）が行われる
      - 各緩和でヒープへのpush: O(log V)
    """
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    # 優先度キュー: (距離, ノード) のタプル
    # なぜタプルにするのか: heapqは第1要素で比較するため
    pq = [(0, start)]

    while pq:
        dist, node = heapq.heappop(pq)

        # この判定が重要: 既により短い距離が確定済みなら無視する
        # なぜ必要か: ヒープには古い（距離が大きい）エントリが残っている可能性がある。
        # decrease-key操作の代わりに、同じ頂点の新しいエントリをpushしているため。
        if dist > distances[node]:
            continue

        for neighbor, weight in graph[node]:
            new_dist = dist + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))

    return distances


def dijkstra_with_path(graph: dict, start, end) -> tuple:
    """
    経路復元付きダイクストラ法。

    戻り値: (最短距離, 最短経路のリスト)
    到達不可能な場合: (float('inf'), [])
    """
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    previous = {node: None for node in graph}  # 経路復元用
    pq = [(0, start)]

    while pq:
        dist, node = heapq.heappop(pq)

        if dist > distances[node]:
            continue

        if node == end:
            break  # 終点が確定したら終了（最適化）

        for neighbor, weight in graph[node]:
            new_dist = dist + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                previous[neighbor] = node
                heapq.heappush(pq, (new_dist, neighbor))

    # 経路復元
    if distances[end] == float('inf'):
        return float('inf'), []

    path = []
    node = end
    while node is not None:
        path.append(node)
        node = previous[node]
    path.reverse()

    return distances[end], path


# === 動作確認 ===
if __name__ == "__main__":
    graph = {
        'A': [('B', 4), ('C', 2)],
        'B': [('D', 3), ('C', 1)],
        'C': [('B', 1), ('D', 5)],
        'D': []
    }

    print("=== ダイクストラ法 ===")
    print("全頂点への最短距離:", dijkstra(graph, 'A'))
    # 出力: {'A': 0, 'B': 3, 'C': 2, 'D': 6}

    dist, path = dijkstra_with_path(graph, 'A', 'D')
    print(f"A→D: 距離={dist}, 経路={path}")
    # 出力: A→D: 距離=6, 経路=['A', 'C', 'B', 'D']
```

### 3.3 ベルマンフォード法（Bellman-Ford Algorithm）

ベルマンフォード法は、**負の重みを持つ辺が存在するグラフ**でも正しく動作する最短経路アルゴリズムである。さらに、**負の閉路の検出**も可能という利点がある。

なぜダイクストラ法では負の重みに対応できないのか: ダイクストラ法は「一度確定した最短距離は変わらない」という前提に基づく。しかし、負の重みの辺があると、確定済みの頂点をさらに短い距離で更新できる場合がある。ベルマンフォード法はこの問題を、全辺の緩和を V-1 回繰り返すことで解決する。

```python
"""
ベルマンフォード法の完全実装
- 負の重みに対応
- 負の閉路検出

なぜ V-1 回の反復で十分なのか:
最短経路に含まれる辺の数は最大 V-1 本（V 頂点を全て通る場合）。
各反復で少なくとも1つの頂点の最短距離が確定するため、
V-1 回の反復で全頂点の最短距離が確定する。

なぜ V 回目の反復で更新があれば負の閉路なのか:
V-1 回で全ての最短距離が確定しているはず。
それにもかかわらず更新が発生するということは、
「周回するたびに距離が減少する閉路」（負の閉路）が存在する。

計算量: O(V × E) — ダイクストラより遅いが、負の重みに対応可能
"""


def bellman_ford(vertices: list, edges: list, start) -> tuple:
    """
    ベルマンフォード法

    引数:
        vertices: 頂点のリスト ['A', 'B', 'C', ...]
        edges: 辺のリスト [(u, v, weight), ...]
        start: 始点

    戻り値:
        (distances, has_negative_cycle)
        distances: {node: distance} の辞書
        has_negative_cycle: 負の閉路が存在するか
    """
    # 初期化
    distances = {v: float('inf') for v in vertices}
    distances[start] = 0

    # V-1 回の反復
    for i in range(len(vertices) - 1):
        updated = False
        for u, v, weight in edges:
            if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                updated = True
        # 最適化: 更新がなければ早期終了
        # なぜ安全に終了できるのか: 更新がないということは、
        # 全ての辺について distances[u] + weight >= distances[v] が成り立ち、
        # これ以上の改善がないことを意味する。
        if not updated:
            break

    # 負の閉路チェック（V回目の反復）
    for u, v, weight in edges:
        if distances[u] != float('inf') and distances[u] + weight < distances[v]:
            return distances, True  # 負の閉路あり

    return distances, False


# === 動作確認 ===
if __name__ == "__main__":
    vertices = ['A', 'B', 'C', 'D', 'E']
    edges = [
        ('A', 'B', 4),
        ('A', 'C', 2),
        ('B', 'D', 3),
        ('C', 'B', -1),   # 負の重み!
        ('C', 'D', 5),
        ('D', 'E', 2),
    ]

    distances, has_neg_cycle = bellman_ford(vertices, edges, 'A')
    print("=== ベルマンフォード法 ===")
    print("最短距離:", distances)
    print("負の閉路:", has_neg_cycle)
    # 出力:
    # 最短距離: {'A': 0, 'B': 1, 'C': 2, 'D': 4, 'E': 6}
    # 負の閉路: False
    # A→C(2)→B(1) のように負の重みを活用した経路が最短

    # 負の閉路のあるグラフ
    vertices2 = ['A', 'B', 'C']
    edges2 = [
        ('A', 'B', 1),
        ('B', 'C', -3),
        ('C', 'A', 1),   # A→B→C→A のコスト: 1+(-3)+1 = -1（負の閉路）
    ]
    distances2, has_neg_cycle2 = bellman_ford(vertices2, edges2, 'A')
    print("\n負の閉路テスト:")
    print("負の閉路:", has_neg_cycle2)
    # 出力: 負の閉路: True
```

### 3.4 フロイドワーシャル法（Floyd-Warshall Algorithm）

フロイドワーシャル法は、**全頂点ペア間の最短距離**を一度に求めるアルゴリズムである。動的計画法に基づいており、隣接行列で表現されたグラフに対して自然に適用できる。

```python
"""
フロイドワーシャル法の完全実装

なぜ3重ループの順番が k, i, j なのか（最も重要な点）:
外側のループ変数 k は「中継頂点の候補」を表す。
dp[k][i][j] = 「頂点 0..k を中継点として使えるとき、i→j の最短距離」
k を外側にすることで、dp[k-1] の全値が確定した状態で dp[k] を計算できる。
i, j を外側にすると、依存関係が崩れて正しい結果が得られない。

計算量: O(V^3)
空間計算量: O(V^2) — in-placeで更新するため
"""


def floyd_warshall(n: int, edges: list) -> list:
    """
    フロイドワーシャル法

    引数:
        n: 頂点数（0-indexed: 0, 1, ..., n-1）
        edges: [(u, v, weight), ...] の辺リスト

    戻り値:
        dist[i][j] = 頂点i→jの最短距離 の2次元配列
        到達不可能な場合は float('inf')
    """
    INF = float('inf')

    # 初期化: 自分自身への距離は0、それ以外はINF
    dist = [[INF] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0

    # 直接の辺を反映
    for u, v, w in edges:
        dist[u][v] = min(dist[u][v], w)  # 多重辺がある場合に最小を採用

    # 本体: k を中継頂点として更新
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    return dist


def floyd_warshall_with_path(n: int, edges: list) -> tuple:
    """
    経路復元付きフロイドワーシャル法

    戻り値:
        (dist, next_node)
        dist[i][j]: i→jの最短距離
        next_node[i][j]: i→jの最短経路上でiの次の頂点
    """
    INF = float('inf')
    dist = [[INF] * n for _ in range(n)]
    next_node = [[None] * n for _ in range(n)]

    for i in range(n):
        dist[i][i] = 0
        next_node[i][i] = i

    for u, v, w in edges:
        if w < dist[u][v]:
            dist[u][v] = w
            next_node[u][v] = v

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_node[i][j] = next_node[i][k]

    return dist, next_node


def reconstruct_path(next_node: list, start: int, end: int) -> list:
    """next_node テーブルから経路を復元する"""
    if next_node[start][end] is None:
        return []  # 到達不可能
    path = [start]
    current = start
    while current != end:
        current = next_node[current][end]
        if current is None:
            return []
        path.append(current)
    return path


# === 動作確認 ===
if __name__ == "__main__":
    #   0 --2--> 1
    #   |        |
    #   6        3
    #   |        |
    #   v        v
    #   2 --1--> 3
    n = 4
    edges = [
        (0, 1, 2),
        (0, 2, 6),
        (1, 3, 3),
        (2, 3, 1),
        (1, 2, 1),  # 1→2 の近道
    ]

    dist, next_node = floyd_warshall_with_path(n, edges)

    print("=== フロイドワーシャル法 ===")
    print("距離行列:")
    for i in range(n):
        row = [str(dist[i][j]) if dist[i][j] != float('inf') else 'INF'
               for j in range(n)]
        print(f"  {i}: {row}")
    # 出力:
    #   0: ['0', '2', '3', '4']
    #   1: ['INF', '0', '1', '2']
    #   2: ['INF', 'INF', '0', '1']
    #   3: ['INF', 'INF', 'INF', '0']

    path = reconstruct_path(next_node, 0, 3)
    print(f"0→3の最短経路: {path}")
    # 出力: 0→3の最短経路: [0, 1, 2, 3]
    # 0→1(2) → 1→2(1) → 2→3(1) = 合計4
```

### 3.5 最短経路アルゴリズムの総合比較

```
最短経路アルゴリズム比較表:

  ┌────────────────┬─────────────────┬──────────┬────────┬─────────────────┐
  │ アルゴリズム    │ 計算量           │ 負の辺   │ 負の閉路│ 主な用途        │
  ├────────────────┼─────────────────┼──────────┼────────┼─────────────────┤
  │ BFS            │ O(V + E)        │ 不可     │ —      │ 重みなしグラフ  │
  │ ダイクストラ    │ O((V+E) log V)  │ 不可     │ —      │ 非負重みのSSP  │
  │ ベルマンフォード│ O(V × E)        │ 可能     │ 検出可 │ 負の重みあり    │
  │ フロイドワーシャル│ O(V^3)        │ 可能     │ 検出可 │ 全対間最短経路  │
  │ A*             │ O(E)（期待値）  │ 不可     │ —      │ 2点間+ヒュー    │
  │                │                 │          │        │ リスティック    │
  └────────────────┴─────────────────┴──────────┴────────┴─────────────────┘

  選択指針:
  - 重みなし → BFS（最速・最もシンプル）
  - 非負重み → ダイクストラ（標準的な選択）
  - 負の重みあり → ベルマンフォード
  - 全対間 → フロイドワーシャル（頂点数が少ない場合）
  - 2点間（大規模グラフ） → A*
  - ネットワークルーティング（OSPF） → ダイクストラ
  - 通貨アービトラージ検出 → ベルマンフォード（対数変換して負の閉路検出）
```

---

## 4. 最小全域木（MST: Minimum Spanning Tree）

### 4.1 最小全域木とは

最小全域木は、重み付き無向連結グラフにおいて、**全頂点を連結しつつ辺の重みの合計が最小となる木**（サイクルのない連結部分グラフ）である。

なぜ「木」なのか: N 頂点を連結するために最低限必要な辺の数は N-1 本であり、N-1 本の辺でサイクルなく全頂点を連結したものが木である。辺を1本でも増やすとサイクルができ、そのサイクル上で最も重い辺を取り除いても連結性は維持される。よって、最小コストの連結部分グラフは必ず木になる。

```
最小全域木の応用例:

  - ネットワーク設計: 全拠点を最小コストの回線で接続
  - 電力網設計: 全地域を最小コストの送電線で接続
  - クラスタリング: MSTの最も重い辺を削除すると、自然なクラスタに分割
  - 近似アルゴリズム: 巡回セールスマン問題の近似解にMSTを利用

  例: 5つの都市を結ぶ最小コストの道路網

     A ---3--- B              A ---3--- B
     |\ /|                         |
     6  2  5  7     MST →     2    7
     |/  \|                   |
     C ---4--- D              C        D
      \       /                \      /
       1                        1
        \   /                    \ /
         E                        E

  元のグラフの辺の重み合計: 3+6+2+5+7+4+1 = 28
  MSTの辺の重み合計: 3+2+7+1 = 13（4辺で5頂点を接続）
```

### 4.2 クラスカル法（Kruskal's Algorithm）

クラスカル法は、辺を重みの小さい順にソートし、サイクルを作らない辺を貪欲に追加していく方法である。Union-Find（素集合データ構造）を用いてサイクル判定を効率的に行う。

```python
"""
クラスカル法の完全実装

なぜ辺を重み順にソートするのか:
最小全域木の「カットの性質」に基づく。任意のカット（頂点集合の分割）において、
そのカットを横断する最小重みの辺は、必ず何らかのMSTに含まれる。
辺を小さい順に見て、異なる連結成分を結ぶ辺（= カットを横断する辺）を
採用することは、この性質を活用した貪欲選択である。

計算量: O(E log E) — ソートが支配的
  辺のソート: O(E log E)
  Union-Find操作: O(E × α(V)) ≈ O(E) — ほぼ定数時間
"""


class UnionFind:
    """
    Union-Find（素集合データ構造）

    なぜ経路圧縮とランクの両方を使うのか:
    - 経路圧縮: find() の際に、木を平らにして次回以降のfindを高速化
    - ランク: union() の際に、低い木を高い木にぶら下げて木の高さを抑制
    両方を組み合わせると、各操作がほぼ O(1)（正確には O(α(n))）になる。
    α(n) は逆アッカーマン関数で、実用的な n の範囲で 4 以下。
    """

    def __init__(self, n: int):
        self.parent = list(range(n))  # 各要素の親（初期状態は自分自身）
        self.rank = [0] * n           # 木の高さの上界

    def find(self, x: int) -> int:
        """xの根を返す（経路圧縮付き）"""
        if self.parent[x] != x:
            # 再帰的に根を見つけ、直接つなぎ直す（経路圧縮）
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """x, yを同じ集合に統合。既に同じ集合ならFalseを返す"""
        px, py = self.find(x), self.find(y)
        if px == py:
            return False  # 既に同じ集合
        # ランクが低い方を高い方にぶら下げる
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True


def kruskal(n: int, edges: list) -> tuple:
    """
    クラスカル法によるMST構築

    引数:
        n: 頂点数
        edges: [(u, v, weight), ...] — 無向辺のリスト

    戻り値:
        (mst_edges, total_weight)
        mst_edges: MSTに含まれる辺のリスト
        total_weight: MSTの総重み
    """
    # 辺を重みでソート
    sorted_edges = sorted(edges, key=lambda e: e[2])

    uf = UnionFind(n)
    mst_edges = []
    total_weight = 0

    for u, v, weight in sorted_edges:
        # u, v が異なる連結成分に属する場合のみ辺を追加
        # （同じ連結成分の場合、辺を追加するとサイクルが生まれる）
        if uf.union(u, v):
            mst_edges.append((u, v, weight))
            total_weight += weight
            # MSTの辺数は V-1 なので、十分な辺が集まったら終了
            if len(mst_edges) == n - 1:
                break

    return mst_edges, total_weight


# === 動作確認 ===
if __name__ == "__main__":
    # 頂点: 0=A, 1=B, 2=C, 3=D, 4=E
    n = 5
    edges = [
        (0, 1, 3),  # A-B: 3
        (0, 2, 6),  # A-C: 6
        (0, 2, 2),  # A-C: 2（多重辺、こちらが短い）
        (1, 3, 7),  # B-D: 7
        (0, 3, 5),  # A-D: 5
        (2, 3, 4),  # C-D: 4
        (2, 4, 1),  # C-E: 1
        (3, 4, 8),  # D-E: 8
    ]

    mst_edges, total = kruskal(n, edges)
    print("=== クラスカル法 ===")
    print(f"MST辺: {mst_edges}")
    print(f"総重み: {total}")
    # 出力:
    # MST辺: [(2, 4, 1), (0, 2, 2), (0, 1, 3), (1, 3, 7)]
    # 総重み: 13
    # 辺の選択順: C-E(1), A-C(2), A-B(3), B-D(7)
```

### 4.3 プリム法（Prim's Algorithm）

プリム法は、1つの頂点から始めて、MST に含まれる頂点集合から出る辺のうち最小重みの辺を貪欲に追加していく方法である。ダイクストラ法と非常に似た構造を持つ。

```python
"""
プリム法の完全実装

なぜプリム法がダイクストラ法に似ているのか:
- ダイクストラ: 「始点からの距離」が最小の頂点を選ぶ
- プリム: 「MST集合からの距離（辺の重み）」が最小の頂点を選ぶ
どちらも優先度キューを使って最小の候補を効率的に選択する。

クラスカル法との比較:
- クラスカル: 辺を全体ソート→サイクルチェック（Union-Find）
- プリム: 頂点を1つずつ追加→隣接辺で優先度キュー更新
- 疎グラフ（E << V^2）: クラスカルが有利
- 密グラフ（E ≈ V^2）: プリムが有利

計算量: O((V + E) log V) — 優先度キュー使用時
"""

import heapq


def prim(n: int, adj: dict) -> tuple:
    """
    プリム法によるMST構築

    引数:
        n: 頂点数
        adj: 隣接リスト {node: [(neighbor, weight), ...], ...}

    戻り値:
        (mst_edges, total_weight)
    """
    if not adj:
        return [], 0

    start = next(iter(adj))  # 任意の始点
    in_mst = set([start])
    mst_edges = []
    total_weight = 0

    # 始点の隣接辺を優先度キューに追加
    # (weight, from_node, to_node) のタプル
    pq = []
    for neighbor, weight in adj[start]:
        heapq.heappush(pq, (weight, start, neighbor))

    while pq and len(in_mst) < n:
        weight, u, v = heapq.heappop(pq)

        if v in in_mst:
            continue  # 既にMSTに含まれている頂点は無視

        # v をMSTに追加
        in_mst.add(v)
        mst_edges.append((u, v, weight))
        total_weight += weight

        # v の隣接辺をキューに追加
        for neighbor, w in adj[v]:
            if neighbor not in in_mst:
                heapq.heappush(pq, (w, v, neighbor))

    return mst_edges, total_weight


# === 動作確認 ===
if __name__ == "__main__":
    adj = {
        'A': [('B', 3), ('C', 2), ('D', 5)],
        'B': [('A', 3), ('D', 7)],
        'C': [('A', 2), ('D', 4), ('E', 1)],
        'D': [('A', 5), ('B', 7), ('C', 4), ('E', 8)],
        'E': [('C', 1), ('D', 8)],
    }

    mst_edges, total = prim(5, adj)
    print("=== プリム法 ===")
    print(f"MST辺: {mst_edges}")
    print(f"総重み: {total}")
    # 出力:
    # MST辺: [('A', 'C', 2), ('C', 'E', 1), ('A', 'B', 3), ('C', 'D', 4)]
    # 総重み: 10
```

---

## 5. トポロジカルソート

### 5.1 トポロジカルソートとは

トポロジカルソートは、有向非巡回グラフ（DAG）の頂点を「全ての辺が左から右に向かう」ように一列に並べる操作である。言い換えると、辺 u→v が存在するなら、並び順で u が v より前に来るような順序を求める。

なぜDAGでしか定義できないのか: サイクルが存在すると、A→B→C→A のような循環があり、AはBの前に、BはCの前に、CはAの前に来なければならない。これは矛盾であり、全ての辺が一方向に向かう順序は存在しない。

```
トポロジカルソートの応用:

  ビルドシステムの依存解決:

  main.c → utils.o → utils.c
  main.c → math.o → math.c
  main.c → main.o

  トポロジカル順序の一例:
  utils.c → utils.o → math.c → math.o → main.c → main.o

  この順序で処理すれば、各ファイルの依存先が
  必ず先に処理されていることが保証される。

  大学の履修計画:
  線形代数 → 微分方程式 → 制御工学
  微積分   → 微分方程式
  プログラミング基礎 → データ構造 → アルゴリズム

  トポロジカル順序:
  線形代数, 微積分, プログラミング基礎,
  微分方程式, データ構造, 制御工学, アルゴリズム
```

### 5.2 Kahn法（BFSベース）とDFSベース

```python
"""
トポロジカルソートの2つの実装
- Kahn法（BFSベース）: 入次数0の頂点を順に処理
- DFSベース: 帰りがけ順の逆順

なぜ2つの方法があるのか:
- Kahn法: サイクル検出が自然に組み込まれる。並列処理可能な頂点が分かる。
- DFS法: 実装がシンプル。他のDFSベースのアルゴリズムと組み合わせやすい。
"""

from collections import deque


def topological_sort_kahn(graph: dict) -> list:
    """
    Kahn法によるトポロジカルソート（BFSベース）

    アイデア: 入次数（入ってくる辺の数）が0の頂点は、
    他の頂点に依存しないため、最初に配置できる。
    その頂点を取り除くと、新たに入次数0になる頂点が現れる。
    これを繰り返す。

    計算量: O(V + E)
    """
    # 入次数を計算
    in_degree = {node: 0 for node in graph}
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] = in_degree.get(neighbor, 0) + 1

    # 入次数0の頂点をキューに入れる
    queue = deque([n for n in in_degree if in_degree[n] == 0])
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # サイクル検出: 全頂点が処理されなかったらサイクルあり
    if len(order) != len(graph):
        raise ValueError(
            f"グラフにサイクルが存在します。"
            f"処理できた頂点: {len(order)}/{len(graph)}"
        )
    return order


def topological_sort_dfs(graph: dict) -> list:
    """
    DFSベースのトポロジカルソート

    アイデア: DFSの帰りがけ順（全ての子孫を処理し終えた順）の逆順が
    トポロジカル順序になる。
    なぜなら、頂点 u から辺 u→v があるとき、DFSでは v の探索が
    u の探索より先に完了する（帰りがけ順で v が先）。
    したがって逆順にすると u が v の前に来る。

    計算量: O(V + E)
    """
    visited = set()
    finish_order = []  # 帰りがけ順

    def _dfs(node):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                _dfs(neighbor)
        finish_order.append(node)  # 帰りがけ（全子孫の処理後）

    for node in graph:
        if node not in visited:
            _dfs(node)

    return finish_order[::-1]  # 帰りがけ順の逆がトポロジカル順序


# === 動作確認 ===
if __name__ == "__main__":
    # 大学の履修依存関係
    courses = {
        '線形代数': ['微分方程式'],
        '微積分': ['微分方程式'],
        '微分方程式': ['制御工学'],
        'プログラミング基礎': ['データ構造'],
        'データ構造': ['アルゴリズム'],
        '制御工学': [],
        'アルゴリズム': [],
    }

    print("=== Kahn法 ===")
    print(topological_sort_kahn(courses))

    print("\n=== DFS法 ===")
    print(topological_sort_dfs(courses))

    # サイクルのあるグラフ（エラーになるはず）
    cyclic = {
        'A': ['B'],
        'B': ['C'],
        'C': ['A'],
    }
    try:
        topological_sort_kahn(cyclic)
    except ValueError as e:
        print(f"\nサイクル検出: {e}")
        # 出力: サイクル検出: グラフにサイクルが存在します。処理できた頂点: 0/3
```

---

## 6. 強連結成分分解（SCC: Strongly Connected Components）

### 6.1 強連結成分とは

有向グラフにおいて、頂点の集合 S が **強連結** であるとは、S 内の任意の2頂点 u, v について、u→v のパスと v→u のパスの両方が存在することである。**強連結成分**（SCC）とは、強連結な頂点集合のうち極大なものである。

```
強連結成分の直感的理解:

  有向グラフ:
    A → B → C → A     D → E → F → D     G

  このグラフには3つのSCCがある:
    SCC1: {A, B, C} — A→B→C→A で互いに到達可能
    SCC2: {D, E, F} — D→E→F→D で互いに到達可能
    SCC3: {G}       — 自分自身のみ（孤立頂点）

  SCCを1つの頂点に縮約すると、DAGが得られる:
    [ABC] → [DEF]
              ↓
             [G]

  なぜSCC分解が重要なのか:
  1. グラフの構造理解: 大きなグラフの「骨格」を把握できる
  2. DAGに帰着: SCC縮約でDAGになり、トポロジカルソートが適用可能
  3. 2-SAT問題: 充足可能性問題の解法に必須
  4. Webの構造分析: Webグラフの「ボウタイ構造」の解析
```

### 6.2 Kosarajuのアルゴリズム

```python
"""
Kosarajuのアルゴリズムによる強連結成分分解

なぜ2回のDFSで正しくSCCが求まるのか:
1回目のDFS: 帰りがけ順（finish time順）を記録
2回目のDFS: 転置グラフ（全辺を逆向きにしたグラフ）上で、
            帰りがけ順の逆順に探索

核心的な洞察:
- 同じSCC内の頂点は、元のグラフでも転置グラフでも互いに到達可能
- 異なるSCC間の辺は、転置すると向きが逆になる
- 帰りがけ順の逆順で処理することで、SCC間の辺を「逆方向」にたどることなく、
  各SCCの頂点のみを正確に拾い上げることができる

計算量: O(V + E) — DFSを2回行うだけ
"""

from collections import defaultdict


def kosaraju_scc(graph: dict) -> list:
    """
    Kosarajuのアルゴリズムによる強連結成分分解

    引数:
        graph: 有向グラフの隣接リスト {node: [neighbors], ...}

    戻り値:
        SCCのリスト [[scc1_nodes], [scc2_nodes], ...]
    """
    # ステップ1: 元のグラフでDFS、帰りがけ順を記録
    visited = set()
    finish_order = []

    def dfs1(node):
        visited.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs1(neighbor)
        finish_order.append(node)

    for node in graph:
        if node not in visited:
            dfs1(node)

    # ステップ2: 転置グラフを構築（全辺を逆向きに）
    reversed_graph = defaultdict(list)
    for node in graph:
        for neighbor in graph[node]:
            reversed_graph[neighbor].append(node)

    # ステップ3: 転置グラフ上で、帰りがけ順の逆順にDFS
    visited.clear()
    sccs = []

    def dfs2(node, component):
        visited.add(node)
        component.append(node)
        for neighbor in reversed_graph.get(node, []):
            if neighbor not in visited:
                dfs2(neighbor, component)

    for node in reversed(finish_order):
        if node not in visited:
            component = []
            dfs2(node, component)
            sccs.append(component)

    return sccs


# === 動作確認 ===
if __name__ == "__main__":
    graph = {
        'A': ['B'],
        'B': ['C', 'E'],
        'C': ['A', 'D'],    # A→B→C→A がサイクル（SCC）
        'D': ['E'],
        'E': ['F'],
        'F': ['D'],          # D→E→F→D がサイクル（SCC）
        'G': [],
    }

    sccs = kosaraju_scc(graph)
    print("=== Kosarajuの強連結成分分解 ===")
    for i, scc in enumerate(sccs):
        print(f"  SCC {i+1}: {scc}")
    # 想定される出力:
    # SCC 1: ['A', 'C', 'B']  （または順序が異なる）
    # SCC 2: ['D', 'F', 'E']
    # SCC 3: ['G']
```

### 6.3 Tarjanのアルゴリズム

Tarjanのアルゴリズムは、DFSを1回だけ行ってSCCを求める方法であり、Kosarajuのアルゴリズムよりも実装は複雑だが、定数倍が小さい。

```python
"""
Tarjanのアルゴリズムによる強連結成分分解

なぜDFS1回で済むのか:
DFS中に各頂点に「発見時刻」と「到達可能な最小発見時刻（low-link値）」を持たせ、
DFSの帰りがけ時に「自分のlow-link値 == 自分の発見時刻」であれば、
その頂点がSCCの「根」（最初に発見された頂点）であると判定する。
スタック上のその頂点より上にある全ての頂点が、同じSCCに属する。

low-link値の意味:
頂点 v の low-link 値 = v から DFS木の辺 + 後退辺を使って
到達可能な頂点のうち、最小の発見時刻。
low[v] == disc[v] のとき、v より上にはSCC外の頂点への後退辺がないため、
v を根とするSCCの境界が確定する。

計算量: O(V + E)
"""


def tarjan_scc(graph: dict) -> list:
    """
    Tarjanのアルゴリズムによる強連結成分分解

    戻り値: SCCのリスト（トポロジカル逆順）
    """
    index_counter = [0]
    stack = []
    on_stack = set()
    disc = {}     # 発見時刻
    low = {}      # low-link値
    sccs = []

    def strongconnect(v):
        disc[v] = low[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)

        for w in graph.get(v, []):
            if w not in disc:
                # 未訪問: DFSで再帰探索し、low値を伝播
                strongconnect(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                # スタック上にある = 現在のSCCの候補
                # なぜ disc[w] と比較するのか:
                # w はまだSCCが確定していない（スタック上）ので、
                # v から w へ到達可能であることを low 値に反映する
                low[v] = min(low[v], disc[w])
            # w がスタック上にない場合: 既にSCCが確定済みなので無視

        # SCCの根を発見（low[v] == disc[v]）
        if low[v] == disc[v]:
            component = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                component.append(w)
                if w == v:
                    break
            sccs.append(component)

    for v in graph:
        if v not in disc:
            strongconnect(v)

    return sccs


# === 動作確認 ===
if __name__ == "__main__":
    graph = {
        0: [1],
        1: [2, 4],
        2: [0, 3],
        3: [4],
        4: [5],
        5: [3],
        6: [],
    }

    sccs = tarjan_scc(graph)
    print("=== Tarjanの強連結成分分解 ===")
    for i, scc in enumerate(sccs):
        print(f"  SCC {i+1}: {scc}")
    # 想定される出力:
    # SCC 1: [0, 2, 1]   ← {0, 1, 2} は互いに到達可能
    # SCC 2: [3, 5, 4]   ← {3, 4, 5} は互いに到達可能
    # SCC 3: [6]          ← 孤立頂点
```

---

## 7. 二部グラフ判定

### 7.1 二部グラフとは

二部グラフとは、グラフの全頂点を2つのグループに分割し、同じグループ内の頂点間には辺がないようにできるグラフである。

```
二部グラフの直感:

  二部グラフの例:              二部グラフでない例:
  (2色で塗り分けできる)        (2色で塗り分けできない)

   ●---○                      ●---○
   |   |                      |   |
   ○---●                      ●---●  ← 同じ色が隣接!
                                \|
                                 ○

  ● = グループA, ○ = グループB

  二部グラフの応用:
  - マッチング問題: 仕事の割り当て、学生と研究室のマッチング
  - 推薦システム: ユーザー×商品の二部グラフ
  - スケジューリング: タスク×時間枠の割り当て

  判定法:
  奇数長の閉路が存在する ⟺ 二部グラフでない
  なぜなら、閉路を2色で交互に塗ると、奇数長なら始点と終点が同色になり矛盾。
```

```python
"""
二部グラフ判定（BFSによる2色塗り分け）

なぜBFSが適しているのか:
BFSは層（レベル）ごとに探索するため、
「偶数レベル→色A、奇数レベル→色B」と自然に2色を割り当てられる。
同じレベルの頂点間に辺があれば、それは奇数長の閉路を意味し、
二部グラフでないと判定できる。

計算量: O(V + E)
"""

from collections import deque


def is_bipartite(graph: dict) -> tuple:
    """
    二部グラフ判定

    戻り値:
        (is_bipartite, coloring)
        is_bipartite: 二部グラフならTrue
        coloring: {node: 0 or 1} の辞書（2色塗り分け）
    """
    color = {}

    for start in graph:
        if start in color:
            continue  # 既に塗り分け済み

        # BFSで2色塗り分け
        color[start] = 0
        queue = deque([start])

        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                if neighbor not in color:
                    # 隣接頂点に反対の色を塗る
                    color[neighbor] = 1 - color[node]
                    queue.append(neighbor)
                elif color[neighbor] == color[node]:
                    # 隣接頂点が同じ色 → 二部グラフでない
                    return False, {}

    return True, color


# === 動作確認 ===
if __name__ == "__main__":
    # 二部グラフ（4頂点のサイクル = 偶数長）
    bipartite_graph = {
        'A': ['B', 'D'],
        'B': ['A', 'C'],
        'C': ['B', 'D'],
        'D': ['C', 'A'],
    }
    result, coloring = is_bipartite(bipartite_graph)
    print(f"二部グラフ: {result}")
    print(f"塗り分け: {coloring}")
    # 出力: 二部グラフ: True
    # 塗り分け: {'A': 0, 'B': 1, 'C': 0, 'D': 1}

    # 非二部グラフ（3頂点のサイクル = 奇数長）
    non_bipartite = {
        'A': ['B', 'C'],
        'B': ['A', 'C'],
        'C': ['A', 'B'],
    }
    result2, _ = is_bipartite(non_bipartite)
    print(f"非二部グラフ: {result2}")
    # 出力: 非二部グラフ: False
```

---

## 8. A* アルゴリズム

### 8.1 A* とは

A* は、ダイクストラ法にヒューリスティック関数を加えたアルゴリズムである。「ゴールに近い方向を優先的に探索する」ことで、不要な探索を減らし、2点間の最短経路を効率的に求める。

なぜダイクストラではなくA*を使うのか: ダイクストラ法は始点から全方向に均等に探索を広げる。しかし、特定の終点への最短経路だけが必要な場合、ゴールと反対方向への探索は無駄である。A* はヒューリスティック（ゴールまでの推定距離）を使って、有望な方向を優先的に探索する。

```
A* の動作イメージ:

  始点Sから終点Gへの最短経路を探す

  ダイクストラ法:              A* アルゴリズム:
  (全方向に均等に広がる)       (ゴール方向を優先)

  . . . . . . G               . . . . . . G
  . . * * . . .               . . . * * / .
  . * * * * . .               . . . * / / .
  * * * S * * .               . . * S / . .
  . * * * * . .               . . . . . . .
  . . * * . . .               . . . . . . .
  . . . . . . .               . . . . . . .

  * = 探索した頂点              / = 探索した頂点
  A* の方が探索範囲が狭い → 高速

  優先度の計算:
  f(n) = g(n) + h(n)
  - g(n): 始点からnまでの実際のコスト（ダイクストラと同じ）
  - h(n): nからゴールまでの推定コスト（ヒューリスティック）
  - f(n): nを通る経路の推定総コスト

  ヒューリスティック h(n) の条件:
  - 許容的（admissible）: h(n) <= 実際の最短距離（過大評価しない）
  - 一貫的（consistent）: h(u) <= cost(u,v) + h(v)
  これらを満たせば、A* は最適解を保証する。
```

```python
"""
A* アルゴリズムの完全実装（グリッド上の経路探索）

なぜグリッドで例示するのか:
A* が最も頻繁に使われるのは、地図やゲームの経路探索である。
2次元グリッドは最も分かりやすい具体例。

ヒューリスティック関数としてマンハッタン距離を使用する。
なぜマンハッタン距離が許容的なのか:
グリッド上で斜め移動がない場合、最短距離は水平+垂直移動の合計。
マンハッタン距離はまさにこの距離であり、障害物がある場合は
実際の最短距離がマンハッタン距離以上になるため、過大評価しない。
"""

import heapq


def astar_grid(grid: list, start: tuple, goal: tuple) -> tuple:
    """
    グリッド上のA*アルゴリズム

    引数:
        grid: 2次元配列（0=通行可、1=壁）
        start: 始点 (row, col)
        goal: 終点 (row, col)

    戻り値:
        (cost, path) — 最短コストと経路
        到達不可能な場合は (float('inf'), [])
    """
    rows, cols = len(grid), len(grid[0])

    def heuristic(a, b):
        """マンハッタン距離"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # 4方向移動（上下左右）
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # 優先度キュー: (f値, g値, 現在位置)
    # なぜg値も入れるのか: f値が同じ場合、g値が大きい（ゴールに近い）方を優先
    open_set = [(heuristic(start, goal), 0, start)]
    g_score = {start: 0}
    came_from = {start: None}

    while open_set:
        f, g, current = heapq.heappop(open_set)

        if current == goal:
            # 経路復元
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            return g, path[::-1]

        if g > g_score.get(current, float('inf')):
            continue

        for dr, dc in directions:
            nr, nc = current[0] + dr, current[1] + dc
            neighbor = (nr, nc)

            # 範囲外チェック
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            # 壁チェック
            if grid[nr][nc] == 1:
                continue

            new_g = g + 1  # 移動コスト1
            if new_g < g_score.get(neighbor, float('inf')):
                g_score[neighbor] = new_g
                f_score = new_g + heuristic(neighbor, goal)
                came_from[neighbor] = current
                heapq.heappush(open_set, (f_score, new_g, neighbor))

    return float('inf'), []  # 到達不可能


# === 動作確認 ===
if __name__ == "__main__":
    grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ]
    start = (0, 0)
    goal = (4, 4)

    cost, path = astar_grid(grid, start, goal)
    print(f"=== A* アルゴリズム ===")
    print(f"最短コスト: {cost}")
    print(f"経路: {path}")

    # グリッド上に経路を表示
    display = [['.' if cell == 0 else '#' for cell in row] for row in grid]
    for r, c in path:
        display[r][c] = '*'
    display[start[0]][start[1]] = 'S'
    display[goal[0]][goal[1]] = 'G'
    print("\nグリッド:")
    for row in display:
        print('  ' + ' '.join(row))
    # 想定される出力:
    # グリッド:
    #   S * * * .
    #   . # # * .
    #   . . . * .
    #   . . # # *
    #   . . . . G
```

---

## 9. アンチパターンと注意点

### 9.1 アンチパターン1: visitedチェックの位置の間違い

```python
"""
アンチパターン: visited チェックをキューから取り出すときに行う
これにより同じ頂点が複数回キューに入り、性能が劣化する。
"""

# NG: キューから取り出すときに visited チェック
def bfs_bad(graph, start):
    """
    問題: 同じ頂点が複数回キューに入る可能性がある。
    最悪の場合、キューサイズが O(E) になる。
    密グラフ（E ≈ V^2）では O(V^2) のメモリを消費。
    """
    visited = set()
    queue = [start]  # さらに悪い例: dequeでなくlistを使用
    order = []

    while queue:
        node = queue.pop(0)  # NG: list.pop(0) は O(n)!
        if node in visited:  # NG: ここでチェックしても既にキューに重複あり
            continue
        visited.add(node)
        order.append(node)
        for neighbor in graph[node]:
            queue.append(neighbor)  # NG: visited チェックなしに追加

    return order


# OK: キューに入れるときに visited チェック
def bfs_good(graph, start):
    """
    正しい実装: キューに入れる前に visited チェック。
    各頂点は最大1回しかキューに入らない。
    キューサイズは O(V) に抑えられる。
    """
    from collections import deque
    visited = set([start])      # 始点を最初から visited に入れる
    queue = deque([start])      # deque の popleft() は O(1)
    order = []

    while queue:
        node = queue.popleft()  # O(1)
        order.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:  # キューに入れる前にチェック
                visited.add(neighbor)
                queue.append(neighbor)

    return order


"""
性能差のイメージ:

  完全グラフ K_1000 (1000頂点、全ペア間に辺) の場合:

  bfs_bad:
    - キューに入る回数: 最大 V×(V-1) ≈ 100万回
    - list.pop(0) の O(n) により、さらに遅い
    - 想定される処理時間: 数秒～数十秒

  bfs_good:
    - キューに入る回数: 正確に V = 1000回
    - deque.popleft() の O(1) で高速
    - 想定される処理時間: 数ミリ秒
"""
```

### 9.2 アンチパターン2: ダイクストラ法での負の重みの無視

```python
"""
アンチパターン: 負の重みがあるグラフにダイクストラ法を適用する
結果は不正確になるが、エラーは発生しないため、バグに気づきにくい。
"""

import heapq


def demonstrate_dijkstra_negative_weight_bug():
    """
    負の重みでダイクストラが失敗する具体例

    グラフ:
      A --1--> B --(-3)--> C
      A --2--> C

    正しい最短距離: A→B→C = 1 + (-3) = -2
    ダイクストラの結果: A→C = 2 (不正確!)

    なぜ失敗するのか:
    ダイクストラは A→C(距離2) を先に確定する。
    その後 A→B(距離1) を処理して B→C(距離-2) を発見しても、
    C は既に確定済みのため更新されない。
    """
    graph = {
        'A': [('B', 1), ('C', 2)],
        'B': [('C', -3)],
        'C': [],
    }

    # ダイクストラ（不正確な結果）
    distances = {'A': float('inf'), 'B': float('inf'), 'C': float('inf')}
    distances['A'] = 0
    pq = [(0, 'A')]

    while pq:
        dist, node = heapq.heappop(pq)
        if dist > distances[node]:
            continue
        for neighbor, weight in graph[node]:
            new_dist = dist + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))

    print("ダイクストラの結果:", distances)
    # 出力: {'A': 0, 'B': 1, 'C': -2}
    # この例ではたまたま正しい結果になることもあるが、
    # 「dist > distances[node]」のスキップにより、
    # 確定済みの頂点を経由する更新が失われるケースが存在する。

    # 正しくは ベルマンフォード を使うべき
    print("\n>>> 負の重みがある場合はベルマンフォード法を使用すること <<<")


# === 推奨: 入力検証を行うダイクストラ ===
def dijkstra_safe(graph: dict, start):
    """負の重みを検出して例外を投げるダイクストラ"""
    for node in graph:
        for neighbor, weight in graph[node]:
            if weight < 0:
                raise ValueError(
                    f"負の重みの辺が検出されました: {node}→{neighbor} (重み={weight})。"
                    f"ベルマンフォード法を使用してください。"
                )

    # 以降は通常のダイクストラ
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]

    while pq:
        dist, node = heapq.heappop(pq)
        if dist > distances[node]:
            continue
        for neighbor, weight in graph[node]:
            new_dist = dist + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))

    return distances


if __name__ == "__main__":
    demonstrate_dijkstra_negative_weight_bug()
```

### 9.3 アンチパターン3: 再帰DFSでのスタックオーバーフロー

```
再帰DFSの落とし穴:

  Python のデフォルト再帰制限: 1000
  頂点数が1000を超えるグラフでは RecursionError が発生する可能性がある。

  対策:
  1. sys.setrecursionlimit() で制限を上げる（推奨しない: OSのスタックに依存）
  2. 反復版DFS（スタックを明示的に管理）を使う（推奨）
  3. 末尾再帰最適化は Python では利用できない

  コンテスト/実務での推奨:
  - 頂点数 < 1000: 再帰DFSでOK
  - 頂点数 >= 1000: 反復版DFSを使う
  - 安全策: 常に反復版を使う習慣をつける
```

---

## 10. エッジケース分析

### 10.1 エッジケース1: 非連結グラフ

```python
"""
エッジケース: グラフが連結でない場合

多くのグラフアルゴリズムの実装は、暗黙に「グラフが連結」と仮定している。
非連結グラフでは、1つの始点からのBFS/DFSでは全頂点に到達できない。

対策: 全頂点をループして、未訪問の頂点から探索を開始する。
"""


def count_connected_components(graph: dict) -> int:
    """
    無向グラフの連結成分数を数える。

    なぜ重要か:
    - ネットワーク分断の検出（「全拠点が到達可能か」のチェック）
    - SNSのコミュニティ数の推定
    - 画像処理での領域分割
    """
    visited = set()
    components = 0

    def dfs(node):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)

    for node in graph:
        if node not in visited:
            dfs(node)
            components += 1

    return components


# === 動作確認 ===
if __name__ == "__main__":
    # 3つの連結成分: {A,B}, {C,D}, {E}
    graph = {
        'A': ['B'],
        'B': ['A'],
        'C': ['D'],
        'D': ['C'],
        'E': [],
    }
    print(f"連結成分数: {count_connected_components(graph)}")
    # 出力: 連結成分数: 3

    # BFSで A から到達可能な頂点
    from collections import deque
    visited = set(['A'])
    queue = deque(['A'])
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    print(f"Aから到達可能: {visited}")
    # 出力: Aから到達可能: {'A', 'B'} — C, D, E には到達不可能!
```

### 10.2 エッジケース2: 自己ループと多重辺

```python
"""
エッジケース: 自己ループ（self-loop）と多重辺（multi-edge）

自己ループ: 頂点から自分自身への辺 (v, v)
多重辺: 同じ頂点ペア間の複数の辺

これらが存在すると、多くのアルゴリズムで特別な処理が必要。
"""


def handle_self_loops_and_multi_edges():
    """自己ループと多重辺のデモ"""

    # 自己ループのあるグラフ
    graph_with_self_loop = {
        'A': ['A', 'B'],  # A→A（自己ループ）, A→B
        'B': ['C'],
        'C': [],
    }

    # 問題1: BFSの visited チェックが正しく機能するか
    from collections import deque
    visited = set(['A'])
    queue = deque(['A'])
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph_with_self_loop[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    print(f"自己ループありBFS: {order}")
    # 出力: ['A', 'B', 'C'] — 自己ループは visited により無視される（正常動作）

    # 問題2: サイクル検出で自己ループは検出されるか
    # 3色法（有向グラフ）: A を GRAY にした後、A→A で GRAY の頂点に到達
    # → サイクルとして正しく検出される

    # 問題3: 多重辺と最短経路
    # ダイクストラでは自然に最小重みの辺が選ばれる
    graph_multi_edge = {
        'A': [('B', 5), ('B', 3), ('B', 7)],  # A→Bに3本の辺
        'B': [],
    }
    import heapq
    distances = {'A': 0, 'B': float('inf')}
    pq = [(0, 'A')]
    while pq:
        dist, node = heapq.heappop(pq)
        if dist > distances[node]:
            continue
        for neighbor, weight in graph_multi_edge[node]:
            new_dist = dist + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))
    print(f"多重辺ダイクストラ: A→B = {distances['B']}")
    # 出力: 多重辺ダイクストラ: A→B = 3（最小重みの辺が自然に選ばれる）


if __name__ == "__main__":
    handle_self_loops_and_multi_edges()
```

### 10.3 エッジケース3: 巨大グラフでのメモリ管理

```
巨大グラフ（頂点数100万以上）でのエッジケース:

  問題1: メモリ不足
  - 隣接行列は使用不可（100万^2 = 1兆要素）
  - 隣接リストを使用する
  - 必要に応じてディスクベースの処理

  問題2: 再帰の深さ制限
  - 再帰DFSは使用不可（スタックオーバーフロー）
  - 反復版DFS/BFSを使用する

  問題3: 優先度キューのサイズ
  - ダイクストラで同じ頂点の複数エントリがヒープに溜まる
  - 対策: visited チェックで古いエントリをスキップ
  - IndexedPriorityQueue を使えば O(V) に抑えられる

  問題4: 到達不可能な頂点
  - 始点から到達できない頂点の距離は float('inf')
  - 「距離無限大」を正しく処理する（0除算やオーバーフローに注意）

  実務での対策:
  ┌──────────────────────┬────────────────────────────────┐
  │ 問題                  │ 対策                            │
  ├──────────────────────┼────────────────────────────────┤
  │ メモリ不足            │ 隣接リスト + ストリーム処理    │
  │ スタックオーバーフロー │ 反復版アルゴリズム             │
  │ 処理速度              │ ヒューリスティック (A*)        │
  │ 超大規模グラフ        │ Contraction Hierarchies 等     │
  │ 分散処理              │ Pregel / GraphX (Spark)        │
  └──────────────────────┴────────────────────────────────┘
```

---

## 11. グラフアルゴリズム総合比較表

```
全グラフアルゴリズムの比較表:

  ┌──────────────────┬─────────────────┬──────────┬──────────────────────────┐
  │ アルゴリズム      │ 計算量           │ 空間     │ 主な用途                 │
  ├──────────────────┼─────────────────┼──────────┼──────────────────────────┤
  │ BFS              │ O(V + E)        │ O(V)     │ 最短経路(重みなし)       │
  │ DFS              │ O(V + E)        │ O(V)     │ サイクル検出、連結成分   │
  │ ダイクストラ      │ O((V+E) log V)  │ O(V)     │ 最短経路(非負重み)       │
  │ ベルマンフォード  │ O(V × E)        │ O(V)     │ 最短経路(負の重みあり)   │
  │ フロイドワーシャル│ O(V^3)          │ O(V^2)   │ 全対間最短経路           │
  │ A*               │ O(E)（期待値）  │ O(V)     │ 2点間最短経路            │
  │ クラスカル        │ O(E log E)      │ O(V)     │ 最小全域木               │
  │ プリム            │ O((V+E) log V)  │ O(V)     │ 最小全域木               │
  │ トポロジカルソート│ O(V + E)        │ O(V)     │ 依存関係の解決           │
  │ Kosaraju SCC     │ O(V + E)        │ O(V + E) │ 強連結成分分解           │
  │ Tarjan SCC       │ O(V + E)        │ O(V)     │ 強連結成分分解           │
  │ Union-Find       │ O(α(n)) ≈ O(1) │ O(V)     │ 連結成分管理             │
  │ 二部グラフ判定   │ O(V + E)        │ O(V)     │ 2色塗り分け              │
  └──────────────────┴─────────────────┴──────────┴──────────────────────────┘

  MST アルゴリズムの選択基準:
  ┌────────────────────────┬──────────────┬──────────────┐
  │ 条件                    │ クラスカル   │ プリム       │
  ├────────────────────────┼──────────────┼──────────────┤
  │ 疎グラフ (E << V^2)    │ ★ 有利      │ 使用可       │
  │ 密グラフ (E ≈ V^2)     │ 使用可       │ ★ 有利      │
  │ 辺がソート済み          │ ★ 非常に有利│ 変わらない   │
  │ オンライン（辺が逐次追加）│ 使用可     │ ★ 有利      │
  │ 並列処理               │ ★ 可能      │ 難しい       │
  └────────────────────────┴──────────────┴──────────────┘
```

---

## 12. 実践演習

### 演習1: 迷路の最短経路（基礎）

```
問題:
2次元グリッドで表現された迷路が与えられる。
0 は通路、1 は壁を表す。
始点 (0,0) から終点 (rows-1, cols-1) への最短経路を求めよ。

入力例:
  grid = [
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 0]
  ]

期待される出力: 最短距離 = 6

ヒント:
- BFS を使う（重みなしグラフの最短経路）
- 4方向（上下左右）の移動を考慮
- visited の管理を忘れずに

解答のポイント:
- グリッドの各セルをグラフの頂点とみなす
- 隣接セル（壁でなく範囲内）への移動を辺とみなす
- BFS でレベル（距離）を管理しながら探索
```

```python
"""演習1の解答例"""
from collections import deque


def solve_maze(grid: list) -> int:
    """迷路のBFS最短経路"""
    rows, cols = len(grid), len(grid[0])
    if grid[0][0] == 1 or grid[rows-1][cols-1] == 1:
        return -1  # 始点or終点が壁

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    visited = set([(0, 0)])
    queue = deque([(0, 0, 0)])  # (row, col, distance)

    while queue:
        r, c, dist = queue.popleft()
        if r == rows - 1 and c == cols - 1:
            return dist
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols
                    and grid[nr][nc] == 0
                    and (nr, nc) not in visited):
                visited.add((nr, nc))
                queue.append((nr, nc, dist + 1))

    return -1  # 到達不可能


# テスト
grid = [
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 0]
]
print(f"最短距離: {solve_maze(grid)}")
# 出力: 最短距離: 6
```

### 演習2: 経路復元付きダイクストラ（応用）

```
問題:
都市間の移動コストが与えられる。始点から終点への最短経路と
そのコストを求め、経路を復元して表示せよ。

入力:
  cities = ['東京', '名古屋', '大阪', '京都', '福岡']
  routes = [
    ('東京', '名古屋', 350),
    ('東京', '大阪', 500),
    ('名古屋', '京都', 130),
    ('名古屋', '大阪', 180),
    ('京都', '大阪', 50),
    ('大阪', '福岡', 600),
    ('京都', '福岡', 650),
  ]

期待される出力:
  東京 → 福岡: コスト 1080
  経路: 東京 → 名古屋 → 京都 → 大阪 → 福岡

ヒント:
- ダイクストラ法 + 経路復元（previous 辞書）
- 有向グラフか無向グラフかを確認
- 到達不可能な場合のハンドリング
```

```python
"""演習2の解答例"""
import heapq
from collections import defaultdict


def solve_city_routes(cities, routes, start, end):
    """都市間最短経路（経路復元付き）"""
    # グラフ構築（双方向）
    graph = defaultdict(list)
    for u, v, w in routes:
        graph[u].append((v, w))
        graph[v].append((u, w))

    # ダイクストラ
    distances = {city: float('inf') for city in cities}
    distances[start] = 0
    previous = {city: None for city in cities}
    pq = [(0, start)]

    while pq:
        dist, node = heapq.heappop(pq)
        if dist > distances[node]:
            continue
        if node == end:
            break
        for neighbor, weight in graph[node]:
            new_dist = dist + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                previous[neighbor] = node
                heapq.heappush(pq, (new_dist, neighbor))

    # 経路復元
    if distances[end] == float('inf'):
        return None, []

    path = []
    node = end
    while node is not None:
        path.append(node)
        node = previous[node]
    path.reverse()

    return distances[end], path


# テスト
cities = ['東京', '名古屋', '大阪', '京都', '福岡']
routes = [
    ('東京', '名古屋', 350),
    ('東京', '大阪', 500),
    ('名古屋', '京都', 130),
    ('名古屋', '大阪', 180),
    ('京都', '大阪', 50),
    ('大阪', '福岡', 600),
    ('京都', '福岡', 650),
]

cost, path = solve_city_routes(cities, routes, '東京', '福岡')
print(f"東京 → 福岡: コスト {cost}")
print(f"経路: {' → '.join(path)}")
# 出力:
# 東京 → 福岡: コスト 1080
# 経路: 東京 → 名古屋 → 京都 → 大阪 → 福岡
```

### 演習3: SNSの友達推薦と影響力分析（発展）

```
問題:
SNSの友達関係グラフが与えられる。以下を実装せよ:

(a) 友達の友達推薦: ユーザーAに対し、直接の友達ではないが
    共通の友達が多い人を推薦する（共通友達数でランキング）

(b) 影響力の計算: 各ユーザーの「影響力」を、
    そのユーザーから距離2以内に到達可能な人数として定義し、
    最も影響力の高いユーザーを求める

(c) コミュニティ検出: 連結成分を求め、各コミュニティの
    メンバーを表示する

ヒント:
- (a): BFS で距離2の頂点を求め、共通友達数をカウント
- (b): BFS で距離2以内の頂点数を数える
- (c): DFS/BFS で連結成分を求める
```

```python
"""演習3の解答例"""
from collections import deque, defaultdict, Counter


class SocialNetwork:
    """SNS分析クラス"""

    def __init__(self):
        self.graph = defaultdict(set)

    def add_friendship(self, u, v):
        """双方向の友達関係を追加"""
        self.graph[u].add(v)
        self.graph[v].add(u)

    def recommend_friends(self, user, top_k=3):
        """
        友達の友達を推薦する。
        共通の友達が多い人ほど上位に推薦。
        """
        if user not in self.graph:
            return []

        direct_friends = self.graph[user]
        # 友達の友達をカウント（共通友達数）
        candidates = Counter()
        for friend in direct_friends:
            for fof in self.graph[friend]:
                if fof != user and fof not in direct_friends:
                    candidates[fof] += 1

        # 共通友達数の多い順に top_k 人を返す
        return candidates.most_common(top_k)

    def influence_score(self, user, max_distance=2):
        """距離 max_distance 以内の人数（影響力スコア）"""
        visited = set([user])
        queue = deque([(user, 0)])
        count = 0

        while queue:
            node, dist = queue.popleft()
            if dist > 0:
                count += 1
            if dist < max_distance:
                for neighbor in self.graph[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

        return count

    def most_influential(self, max_distance=2):
        """最も影響力の高いユーザーを返す"""
        scores = {user: self.influence_score(user, max_distance)
                  for user in self.graph}
        return max(scores.items(), key=lambda x: x[1])

    def find_communities(self):
        """連結成分（コミュニティ）を検出"""
        visited = set()
        communities = []

        for user in self.graph:
            if user not in visited:
                community = []
                queue = deque([user])
                visited.add(user)
                while queue:
                    node = queue.popleft()
                    community.append(node)
                    for neighbor in self.graph[node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                communities.append(sorted(community))

        return communities


# === テスト ===
if __name__ == "__main__":
    sn = SocialNetwork()
    # コミュニティ1
    sn.add_friendship('Alice', 'Bob')
    sn.add_friendship('Alice', 'Charlie')
    sn.add_friendship('Bob', 'Charlie')
    sn.add_friendship('Bob', 'David')
    sn.add_friendship('Charlie', 'Eve')
    sn.add_friendship('David', 'Eve')

    # コミュニティ2（独立）
    sn.add_friendship('Frank', 'Grace')
    sn.add_friendship('Grace', 'Heidi')

    # (a) 友達推薦
    print("=== 友達推薦 ===")
    recs = sn.recommend_friends('Alice')
    for person, common_count in recs:
        print(f"  {person}: 共通友達 {common_count}人")
    # 想定される出力:
    #   Eve: 共通友達 2人（Bob, Charlie経由）
    #   David: 共通友達 1人（Bob経由）

    # (b) 影響力
    print("\n=== 影響力スコア ===")
    for user in ['Alice', 'Bob', 'Frank']:
        score = sn.influence_score(user)
        print(f"  {user}: {score}人")
    user, score = sn.most_influential()
    print(f"  最も影響力が高い: {user} ({score}人)")

    # (c) コミュニティ
    print("\n=== コミュニティ ===")
    communities = sn.find_communities()
    for i, comm in enumerate(communities):
        print(f"  コミュニティ{i+1}: {comm}")
    # 想定される出力:
    # コミュニティ1: ['Alice', 'Bob', 'Charlie', 'David', 'Eve']
    # コミュニティ2: ['Frank', 'Grace', 'Heidi']
```

---

## 13. FAQ

### Q1: Google Mapsの経路探索はダイクストラですか?

**A**: 基本的にはA*アルゴリズムがベースだが、そのままでは大陸規模のグラフ（数十億の頂点）には遅すぎる。実際には、以下の高度な技法が組み合わされている:

- **Contraction Hierarchies（CH）**: 事前処理で「重要な頂点」のショートカット辺を作成し、実行時の探索範囲を大幅に削減する。前処理に数時間かかるが、クエリは数ミリ秒で応答可能。
- **ALT（A* with Landmarks and Triangle inequality）**: ランドマーク（地図上の特定の点）までの距離を事前計算し、ヒューリスティック関数として利用する。
- **Transit Node Routing**: 長距離の経路は少数の「中継ノード」（高速道路のIC等）を経由するという性質を利用し、中継ノード間の距離を事前計算する。

これらの前処理ベースの手法により、数千万～数十億頂点のグラフでも、2点間の最短経路をミリ秒オーダーで計算できる。

### Q2: グラフのサイクル検出で、無向グラフと有向グラフでなぜ方法が違うのですか?

**A**: 無向グラフと有向グラフでは「サイクル」の定義が微妙に異なるため。

- **無向グラフ**: A—B という辺があるとき、A→B→A は長さ2のサイクルとは見なさない（同じ辺の往復）。DFSで「親でない訪問済み頂点」に到達したらサイクル。Union-Findでも検出可能（辺を追加するとき、両端が同じ集合ならサイクル）。

- **有向グラフ**: A→B→C のグラフで、Cが訪問済みでもA→B→C→...→A というサイクルがあるとは限らない（Cの先がAに戻らないかもしれない）。3色法（WHITE/GRAY/BLACK）で「現在の探索パス上（GRAY）の頂点に到達したらサイクル」と判定する必要がある。BLACKの頂点への辺はサイクルではない。

### Q3: PageRankもグラフアルゴリズムですか?

**A**: PageRankはWebページのグラフ上で定義されるアルゴリズムである。各ページを頂点、ハイパーリンクを辺とするグラフ上で、ランダムウォーク（ユーザーがリンクをランダムにクリックする行動のモデル）の定常分布を計算する。

具体的には、ページ v の PageRank PR(v) は次の式で定義される:

```
PR(v) = (1-d)/N + d × Σ PR(u) / L(u)
```

ここで d はダンピングファクター（通常0.85）、N は全ページ数、L(u) はページ u からの出リンク数である。この式を全ページについて反復的に計算し、収束させる。行列の固有ベクトル問題としても定式化できる。Google検索の基盤となったアルゴリズムだが、現在は数百の要素と組み合わせて使われている。

### Q4: 二部マッチング問題とは何ですか?

**A**: 二部グラフにおいて、辺の部分集合で各頂点が高々1つの辺にのみ接するもの（マッチング）のうち、最大のもの（最大マッチング）を求める問題。仕事の割り当て（従業員×タスク）、研修医のマッチング、安定結婚問題などに応用される。

ハンガリアン法（O(V^3)）やホップクロフト-カープ法（O(E√V)）で効率的に解ける。最大フロー問題に帰着させる方法もある。

### Q5: グラフアルゴリズムを学ぶ際の優先順位は?

**A**: 以下の順序が推奨される:

1. **BFS/DFS**（最も基本。これなしには何も始まらない）
2. **ダイクストラ法**（最短経路の標準。面接でも頻出）
3. **トポロジカルソート**（依存解決。ビルドシステム等で頻出）
4. **Union-Find**（連結成分。クラスカル法の前提）
5. **最小全域木**（ネットワーク設計）
6. **ベルマンフォード/フロイドワーシャル**（負の重み対応）
7. **強連結成分分解**（高度な問題。2-SATの基盤）
8. **A***（ゲーム開発・地図アプリ）

### Q6: 実務でグラフアルゴリズムを使う場面はありますか?

**A**: 非常に多い。気づかないうちにグラフアルゴリズムに依存している場面がある:

- **パッケージマネージャ**（npm, pip, cargo）: 依存関係のDAGをトポロジカルソートしてインストール順を決定
- **CI/CDパイプライン**: ジョブの依存関係をDAGで管理（GitHub Actions, CircleCI）
- **データベース**: 外部キー制約の循環参照チェック（サイクル検出）
- **GCのマーク&スイープ**: オブジェクト参照グラフのBFS/DFS（到達可能オブジェクトの判定）
- **ネットワーク**: ルーティングプロトコル（OSPF = ダイクストラ、RIP = ベルマンフォード）
- **推薦エンジン**: ユーザー×アイテムの二部グラフでの協調フィルタリング

---


## FAQ

### Q1: このトピックを学ぶ上で最も重要なポイントは何ですか？

実践的な経験を積むことが最も重要です。理論だけでなく、実際にコードを書いて動作を確認することで理解が深まります。

### Q2: 初心者がよく陥る間違いは何ですか？

基礎を飛ばして応用に進むことです。このガイドで説明している基本概念をしっかり理解してから、次のステップに進むことをお勧めします。

### Q3: 実務ではどのように活用されていますか？

このトピックの知識は、日常的な開発業務で頻繁に活用されます。特にコードレビューやアーキテクチャ設計の際に重要になります。

---

## 14. まとめ

### 学習の振り返りチェックリスト

- [ ] グラフの隣接リスト/隣接行列の使い分けを説明できる
- [ ] BFSとDFSの動作原理と計算量を説明できる
- [ ] BFSで重みなしグラフの最短経路が求まる理由を説明できる
- [ ] ダイクストラ法が負の重みで正しく動作しない理由を説明できる
- [ ] ベルマンフォード法の「V-1回反復」の意味を説明できる
- [ ] フロイドワーシャル法の「k, i, j の順番」の理由を説明できる
- [ ] クラスカル法とプリム法の違いを説明できる
- [ ] トポロジカルソートが適用できる条件（DAG）を説明できる
- [ ] 強連結成分の定義と、Kosaraju/Tarjanの基本的な動作を説明できる
- [ ] 二部グラフ判定を2色塗り分けで実装できる
- [ ] 問題に応じて適切なグラフアルゴリズムを選択できる

### アルゴリズム選択フローチャート

```
問題の種類を判別:

  最短経路を求めたい
  ├── 重みなし → BFS
  ├── 非負の重み
  │   ├── 単一始点 → ダイクストラ
  │   └── 2点間（大規模） → A*
  ├── 負の重みあり
  │   ├── 単一始点 → ベルマンフォード
  │   └── 全対間 → フロイドワーシャル
  └── 負の閉路検出 → ベルマンフォード

  連結性に関する問題
  ├── 連結成分（無向） → BFS/DFS または Union-Find
  ├── 強連結成分（有向） → Kosaraju / Tarjan
  └── 二部グラフ判定 → BFS（2色塗り分け）

  順序に関する問題
  ├── 依存解決 → トポロジカルソート
  └── サイクル検出
      ├── 無向グラフ → DFS（親チェック）/ Union-Find
      └── 有向グラフ → DFS（3色法）

  最適化問題
  ├── 最小全域木
  │   ├── 疎グラフ → クラスカル
  │   └── 密グラフ → プリム
  └── 最大マッチング → ホップクロフト-カープ / 最大フロー
```

---

## 次に読むべきガイド


---

## 参考文献

1. Cormen, T. H., Leiserson, C. E., Rivest, R. L., and Stein, C. **"Introduction to Algorithms."** 4th Edition, MIT Press, 2022. Chapters 20-26. — グラフアルゴリズムの最も包括的な教科書。証明と擬似コードが充実。
2. Skiena, S. S. **"The Algorithm Design Manual."** 3rd Edition, Springer, 2020. Chapters 7-8. — 実践的な視点からグラフアルゴリズムを解説。「どの問題にどのアルゴリズムを使うか」のガイドが秀逸。
3. Sedgewick, R. and Wayne, K. **"Algorithms."** 4th Edition, Addison-Wesley, 2011. Chapter 4. — Javaによる実装例が豊富。可視化ツール付きで動作を直感的に理解できる。
4. Kleinberg, J. and Tardos, E. **"Algorithm Design."** Pearson, 2005. Chapters 3-7. — グラフアルゴリズムの設計原理（貪欲法、動的計画法との関連）を深く理解できる。
5. **Stanford CS161: Design and Analysis of Algorithms.** Online lecture notes. — ダイクストラ法の正当性証明やSCCの詳細な解説が無料で公開されている。
6. **Competitive Programmer's Handbook** by Antti Laaksonen. — 競技プログラミングの視点からのグラフアルゴリズム実装テクニック集。無料PDF公開。
