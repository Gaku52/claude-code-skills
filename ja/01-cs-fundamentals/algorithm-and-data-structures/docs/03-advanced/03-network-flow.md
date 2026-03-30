# ネットワークフロー

> グラフ上の最大流問題を Ford-Fulkerson法・Dinic法・二部マッチング・最小費用流を通じて理解し、実用的な応用パターンを習得する

## この章で学ぶこと

1. **最大流問題の定義**と残余グラフ・増加パスの概念を理解する
2. **Ford-Fulkerson法**と BFS による Edmonds-Karp 法を正確に実装できる
3. **Dinic法**で高速に最大流を求める手法を実装できる
4. **二部マッチング**を最大流に帰着させ、仕事割当・マッチング問題を解ける
5. **最大流最小カット定理・最小費用流**の理論と応用を理解する


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解
- [文字列アルゴリズム](./02-string-algorithms.md) の内容を理解していること

---

## 1. ネットワークフローの基本概念

ネットワークフロー問題は、パイプラインや通信ネットワークを流れる「もの」の最大量を求める問題として定式化される。

```
フローネットワーク:
  ・有向グラフ G = (V, E)
  ・容量関数 c(u,v) ≥ 0（各辺の最大流量）
  ・始点 s（ソース）、終点 t（シンク）

制約:
  1. 容量制約: 0 ≤ f(u,v) ≤ c(u,v)
  2. フロー保存: 各頂点で流入量 = 流出量（s,t を除く）
  3. 歪対称性: f(u,v) = -f(v,u)

例:
         10        10
    s ────→ A ────→ t
    │       ↑       ↑
    │5      │15     │10
    ↓       │       │
    B ────→ C ────→ D
         10        10

最大流 = 19（s→A→t:10, s→B→C→D→t:5, s→B→C→A→t:0, ...）
```

### フロー問題の実世界での例

```
1. 物流ネットワーク
   工場(s) → 倉庫 → 配送センター → 店舗(t)
   各辺の容量 = トラックの輸送能力
   最大流 = 最大輸送量

2. 通信ネットワーク
   送信元(s) → ルータ群 → 受信先(t)
   各辺の容量 = 回線の帯域幅
   最大流 = 最大データ転送速度

3. 水道管ネットワーク
   水源(s) → 配管 → 各家庭(t)
   各辺の容量 = 管の太さ
   最大流 = 最大配水量

4. スケジューリング
   開始(s) → 作業員 → タスク → 終了(t)
   容量 = 各作業員の処理能力
   最大流 = 最大処理量
```

---

## 2. 残余グラフと増加パス

```
元のグラフ:               現在のフロー:
    s ──(10)──→ A           s ──7/10──→ A
    │           │           │           │
   (5)       (10)         5/5        7/10
    ↓           ↓           ↓           ↓
    B ──(10)──→ t           B ──5/10──→ t

残余グラフ（residual graph）:
  ・順方向: 残り容量 = c(u,v) - f(u,v)
  ・逆方向: キャンセル可能量 = f(u,v)

    s ──(3)──→ A           残り容量
    │ ←(7)── A             逆方向（キャンセル）
    │           │
   (0)→      (3)→
   ←(5)      ←(7)
    ↓           ↓
    B ──(5)──→ t
    B ←(5)── t

増加パス = 残余グラフ上の s→t パス
ボトルネック = パス上の最小残余容量
```

### 残余グラフの重要性

```
なぜ逆辺（キャンセル）が必要か？

例:
    s ──(1)──→ A ──(1)──→ t
    │                       ↑
   (1)                    (1)
    ↓                       │
    B ──────(1)────────→ C

最適なフロー:
  s→A→t: 1
  s→B→C→t: 1
  合計: 2

逆辺なしで貪欲に選ぶと:
  s→A→C→t ではなく...（Cからtへの直接辺がないケース）
  → 逆辺があることで、一度流したフローを「取り消す」ことが可能
  → 最適解に到達できる
```

---

## 3. Ford-Fulkerson法

残余グラフ上で増加パスを見つけ、フローを送ることを繰り返す。

```python
from collections import defaultdict, deque

class MaxFlow:
    """Ford-Fulkerson法（Edmonds-Karp: BFSで増加パス探索）"""

    def __init__(self, n: int):
        self.n = n
        self.graph = defaultdict(lambda: defaultdict(int))  # 容量

    def add_edge(self, u: int, v: int, cap: int):
        """辺を追加（有向）"""
        self.graph[u][v] += cap

    def bfs(self, source: int, sink: int, parent: dict) -> int:
        """BFS で増加パスを探索し、ボトルネック容量を返す"""
        visited = {source}
        queue = deque([(source, float('inf'))])

        while queue:
            u, flow = queue.popleft()
            for v in self.graph[u]:
                if v not in visited and self.graph[u][v] > 0:
                    visited.add(v)
                    parent[v] = u
                    new_flow = min(flow, self.graph[u][v])
                    if v == sink:
                        return new_flow
                    queue.append((v, new_flow))

        return 0

    def max_flow(self, source: int, sink: int) -> int:
        """最大流を計算 - O(VE^2)"""
        total_flow = 0

        while True:
            parent = {}
            path_flow = self.bfs(source, sink, parent)

            if path_flow == 0:
                break  # 増加パスがない → 最大流に到達

            total_flow += path_flow

            # フローの更新（残余グラフの更新）
            v = sink
            while v != source:
                u = parent[v]
                self.graph[u][v] -= path_flow  # 順方向: 容量を減らす
                self.graph[v][u] += path_flow  # 逆方向: キャンセル可能量を増やす
                v = u

        return total_flow

# 使用例
mf = MaxFlow(6)
# s=0, A=1, B=2, C=3, D=4, t=5
mf.add_edge(0, 1, 10)  # s → A
mf.add_edge(0, 2, 10)  # s → B
mf.add_edge(1, 3, 4)   # A → C
mf.add_edge(1, 4, 8)   # A → D
mf.add_edge(2, 4, 9)   # B → D
mf.add_edge(3, 5, 10)  # C → t
mf.add_edge(4, 3, 6)   # D → C
mf.add_edge(4, 5, 10)  # D → t

print(mf.max_flow(0, 5))  # 19
```

### Ford-Fulkerson法の動作トレース

```
初期状態:
  s → A: 10, s → B: 10
  A → C: 4,  A → D: 8
  B → D: 9
  C → t: 10, D → C: 6, D → t: 10

Iteration 1: BFS で s→A→D→t を発見（ボトルネック = min(10,8,10) = 8）
  更新後: s→A: 2, A→D: 0, D→t: 2

Iteration 2: BFS で s→A→C→t を発見（ボトルネック = min(2,4,10) = 2）
  更新後: s→A: 0, A→C: 2, C→t: 8

Iteration 3: BFS で s→B→D→C→t を発見（ボトルネック = min(10,9,6,8) = 6）
  更新後: s→B: 4, B→D: 3, D→C: 0, C→t: 2

Iteration 4: BFS で s→B→D→t を発見（ボトルネック = min(4,3,2) = 2）
  更新後: s→B: 2, B→D: 1, D→t: 0

Iteration 5: 増加パスなし → 終了

最大流 = 8 + 2 + 6 + 2 + 1 = 19
```

---

## 4. Dinic法（高速版）

レベルグラフ（BFS で構築）上でブロッキングフローを求める。Edmonds-Karp より高速で、容量が整数の場合 O(V^2 E)、単位容量の場合 O(E sqrt(V))。

```python
class Dinic:
    """Dinic法 - O(V^2 E)
    二部マッチングでは O(E sqrt(V))
    """

    def __init__(self, n: int):
        self.n = n
        self.graph = [[] for _ in range(n)]

    def add_edge(self, u: int, v: int, cap: int):
        """辺の追加（逆辺も同時に追加）"""
        self.graph[u].append([v, cap, len(self.graph[v])])
        self.graph[v].append([u, 0, len(self.graph[u]) - 1])

    def bfs(self, s: int, t: int) -> bool:
        """レベルグラフを構築"""
        self.level = [-1] * self.n
        self.level[s] = 0
        queue = deque([s])

        while queue:
            u = queue.popleft()
            for v, cap, _ in self.graph[u]:
                if cap > 0 and self.level[v] < 0:
                    self.level[v] = self.level[u] + 1
                    queue.append(v)

        return self.level[t] >= 0

    def dfs(self, u: int, t: int, f: int) -> int:
        """ブロッキングフローを求める"""
        if u == t:
            return f
        while self.iter[u] < len(self.graph[u]):
            v, cap, rev = self.graph[u][self.iter[u]]
            if cap > 0 and self.level[v] == self.level[u] + 1:
                d = self.dfs(v, t, min(f, cap))
                if d > 0:
                    self.graph[u][self.iter[u]][1] -= d
                    self.graph[v][rev][1] += d
                    return d
            self.iter[u] += 1
        return 0

    def max_flow(self, s: int, t: int) -> int:
        """最大流を計算"""
        flow = 0
        while self.bfs(s, t):
            self.iter = [0] * self.n
            while True:
                f = self.dfs(s, t, float('inf'))
                if f == 0:
                    break
                flow += f
        return flow

    def min_cut(self, s: int) -> list:
        """最小カットを求める（max_flow 実行後に呼ぶ）
        返り値: s 側に属する頂点のリスト
        """
        visited = [False] * self.n
        queue = deque([s])
        visited[s] = True
        while queue:
            u = queue.popleft()
            for v, cap, _ in self.graph[u]:
                if cap > 0 and not visited[v]:
                    visited[v] = True
                    queue.append(v)
        return [i for i in range(self.n) if visited[i]]

# 使用例
dinic = Dinic(6)
dinic.add_edge(0, 1, 10)
dinic.add_edge(0, 2, 10)
dinic.add_edge(1, 3, 4)
dinic.add_edge(1, 4, 8)
dinic.add_edge(2, 4, 9)
dinic.add_edge(3, 5, 10)
dinic.add_edge(4, 3, 6)
dinic.add_edge(4, 5, 10)
print(dinic.max_flow(0, 5))  # 19
```

### Dinic法 vs Edmonds-Karp の違い

```
Edmonds-Karp:
  - BFS で最短の増加パスを1本見つける
  - フローを流す
  - 再び BFS → ... を繰り返す
  - 計算量: O(VE^2)

Dinic:
  - BFS でレベルグラフを構築
  - レベルグラフ上で DFS で複数の増加パスを一度に処理（ブロッキングフロー）
  - レベルグラフが変わるまで DFS を繰り返す
  - 再び BFS → ... を繰り返す
  - 計算量: O(V^2 E)

  キーポイント:
  - BFS の回数は高々 V-1 回（レベルが毎回少なくとも1増加）
  - 各 BFS フェーズ内で DFS によるブロッキングフローは O(VE)
  - iter 配列で同じ辺を再探索しない → DFS の効率化
```

---

## 5. 最大流最小カット定理

```
最大流 = 最小カット

最小カット = ネットワークを s 側と t 側に分断する辺の
            容量の最小和

例:
    s ──(3)──→ A ──(2)──→ t
    │                      ↑
   (5)                   (4)
    ↓                      │
    B ────────(6)────────→ C

最大流 = 7
最小カット: {(A,t): 2, (s,B): 5} = 7

→ 最大流の後、残余グラフで s から到達可能な頂点集合 S と
  到達不能な頂点集合 T を求めると、S→T の辺が最小カット
```

```python
def find_min_cut_edges(n: int, edges: list, source: int, sink: int) -> list:
    """最小カットの辺を求める"""
    dinic = Dinic(n)
    original_edges = []

    for u, v, cap in edges:
        edge_idx = len(dinic.graph[u])
        dinic.add_edge(u, v, cap)
        original_edges.append((u, v, cap, edge_idx))

    max_flow_value = dinic.max_flow(source, sink)

    # s 側の頂点を特定
    s_side = set(dinic.min_cut(source))

    # S→T の辺がカット辺
    cut_edges = []
    for u, v, cap, _ in original_edges:
        if u in s_side and v not in s_side:
            cut_edges.append((u, v, cap))

    return max_flow_value, cut_edges

edges = [(0, 1, 3), (0, 2, 5), (1, 3, 2), (2, 3, 6), (2, 1, 4)]
# s=0, A=1, B=2, t=3
flow, cuts = find_min_cut_edges(4, edges, 0, 3)
print(f"最大流: {flow}")
print(f"最小カット辺: {cuts}")
```

### 最大流最小カット定理の証明の概要

```
定理: 任意のフローネットワークにおいて、
      最大流の値 = 最小カットの容量

証明の概略:
1. 任意のフロー f と任意のカット (S, T) について、
   |f| ≤ c(S, T) （フロー値 ≤ カット容量）

2. Ford-Fulkerson が停止したとき、
   残余グラフに s→t パスが存在しない

3. s から到達可能な頂点集合を S、残りを T とすると、
   S→T のすべての辺は飽和（f(u,v) = c(u,v)）
   T→S のすべての辺はフロー 0（f(v,u) = 0）

4. よって |f| = c(S, T) = 最小カットの容量
```

---

## 6. 二部マッチング

二部グラフの最大マッチングを最大流に帰着する。

```
二部グラフ:                 フローネットワーク化:
  学生  ←→  プロジェクト       s → 学生 → プロジェクト → t
                              容量全て 1

  A ── P1                    s ──→ A ──→ P1 ──→ t
  A ── P2                    s ──→ A ──→ P2 ──→ t
  B ── P1                    s ──→ B ──→ P1 ──→ t
  B ── P3                    s ──→ B ──→ P3 ──→ t
  C ── P2                    s ──→ C ──→ P2 ──→ t
  C ── P3                    s ──→ C ──→ P3 ──→ t

  最大マッチング = 最大流 = 3
  例: A-P1, B-P3, C-P2
```

```python
def bipartite_matching(left: int, right: int, edges: list) -> tuple:
    """二部マッチング（最大流ベース）
    left: 左側頂点数, right: 右側頂点数
    edges: [(左頂点, 右頂点), ...]
    """
    n = left + right + 2
    source = 0
    sink = n - 1

    dinic = Dinic(n)

    # source → 左側頂点（容量1）
    for i in range(left):
        dinic.add_edge(source, i + 1, 1)

    # 左側 → 右側（容量1）
    for l, r in edges:
        dinic.add_edge(l + 1, left + r + 1, 1)

    # 右側頂点 → sink（容量1）
    for j in range(right):
        dinic.add_edge(left + j + 1, sink, 1)

    max_matching = dinic.max_flow(source, sink)
    return max_matching

# Hungarian アルゴリズム（直接実装版: DFSベース）
def hungarian(n: int, m: int, adj: list) -> tuple:
    """二部マッチング（ハンガリアン法）- O(VE)
    n: 左側頂点数, m: 右側頂点数
    adj[i]: 左頂点iに隣接する右頂点のリスト
    """
    match_l = [-1] * n  # 左側のマッチ相手
    match_r = [-1] * m  # 右側のマッチ相手

    def dfs(u, visited):
        for v in adj[u]:
            if visited[v]:
                continue
            visited[v] = True
            # v が未マッチ or v のマッチ相手を別に移せる
            if match_r[v] == -1 or dfs(match_r[v], visited):
                match_l[u] = v
                match_r[v] = u
                return True
        return False

    matching = 0
    for u in range(n):
        visited = [False] * m
        if dfs(u, visited):
            matching += 1

    return matching, match_l, match_r

# 使用例: 学生(0,1,2) → プロジェクト(0,1,2)
adj = [
    [0, 1],  # 学生0 → P0, P1
    [0, 2],  # 学生1 → P0, P2
    [1, 2],  # 学生2 → P1, P2
]
count, ml, mr = hungarian(3, 3, adj)
print(f"最大マッチング: {count}")  # 3
print(f"左→右: {ml}")             # [0, 2, 1] or similar
```

### Hopcroft-Karp法

二部マッチングの最速アルゴリズム。O(E sqrt(V))。

```python
def hopcroft_karp(n: int, m: int, adj: list) -> tuple:
    """Hopcroft-Karp法 - O(E sqrt(V))
    n: 左側頂点数, m: 右側頂点数
    adj[i]: 左頂点iに隣接する右頂点のリスト
    """
    INF = float('inf')
    match_l = [-1] * n
    match_r = [-1] * m
    dist = [0] * n

    def bfs():
        queue = deque()
        for u in range(n):
            if match_l[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = INF

        found = False
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                w = match_r[v]
                if w == -1:
                    found = True
                elif dist[w] == INF:
                    dist[w] = dist[u] + 1
                    queue.append(w)
        return found

    def dfs(u):
        for v in adj[u]:
            w = match_r[v]
            if w == -1 or (dist[w] == dist[u] + 1 and dfs(w)):
                match_l[u] = v
                match_r[v] = u
                return True
        dist[u] = INF
        return False

    matching = 0
    while bfs():
        for u in range(n):
            if match_l[u] == -1:
                if dfs(u):
                    matching += 1

    return matching, match_l, match_r

# 使用例
adj = [
    [0, 1],   # 左0 → 右0, 右1
    [0, 2],   # 左1 → 右0, 右2
    [1, 2],   # 左2 → 右1, 右2
]
count, ml, mr = hopcroft_karp(3, 3, adj)
print(f"最大マッチング: {count}")  # 3
```

---

## 7. 応用パターン

### 最小頂点被覆（Konig の定理）

```python
def minimum_vertex_cover(n: int, m: int, adj: list) -> list:
    """二部グラフの最小頂点被覆を求める（Konig の定理）
    最小頂点被覆 = 最大マッチング（二部グラフのみ）
    返り値: 被覆に含まれる頂点のリスト
    """
    _, match_l, match_r = hungarian(n, m, adj)

    # 未マッチの左頂点から交互道を辿る
    visited_l = [False] * n
    visited_r = [False] * m

    # 未マッチの左頂点をキューに入れる
    queue = deque()
    for u in range(n):
        if match_l[u] == -1:
            queue.append(u)
            visited_l[u] = True

    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if not visited_r[v]:
                visited_r[v] = True
                w = match_r[v]
                if w != -1 and not visited_l[w]:
                    visited_l[w] = True
                    queue.append(w)

    # 被覆 = 到達不能な左頂点 + 到達可能な右頂点
    cover = []
    for u in range(n):
        if not visited_l[u]:
            cover.append(('L', u))
    for v in range(m):
        if visited_r[v]:
            cover.append(('R', v))

    return cover
```

### プロジェクト割り当て問題

```python
def project_assignment(students: list, projects: list,
                       preferences: dict) -> dict:
    """学生をプロジェクトに最大数割り当て"""
    n_students = len(students)
    n_projects = len(projects)

    student_idx = {s: i for i, s in enumerate(students)}
    project_idx = {p: i for i, p in enumerate(projects)}

    adj = [[] for _ in range(n_students)]
    for student, prefs in preferences.items():
        for proj in prefs:
            adj[student_idx[student]].append(project_idx[proj])

    count, match_l, _ = hungarian(n_students, n_projects, adj)

    assignment = {}
    for i, j in enumerate(match_l):
        if j != -1:
            assignment[students[i]] = projects[j]

    return assignment

students = ["Alice", "Bob", "Charlie"]
projects = ["Web", "AI", "DB"]
prefs = {
    "Alice": ["Web", "AI"],
    "Bob": ["AI", "DB"],
    "Charlie": ["Web", "DB"],
}
result = project_assignment(students, projects, prefs)
print(result)  # {'Alice': 'Web', 'Bob': 'AI', 'Charlie': 'DB'} など
```

### 頂点素なパス数（頂点分割法）

```python
def vertex_disjoint_paths(n: int, edges: list, s: int, t: int) -> int:
    """s から t への頂点素なパスの最大数
    各頂点を v_in と v_out に分割し、v_in → v_out の容量を 1 にする
    """
    # 頂点 v → v_in = 2*v, v_out = 2*v + 1
    dinic = Dinic(2 * n)

    for v in range(n):
        if v == s or v == t:
            dinic.add_edge(2 * v, 2 * v + 1, float('inf'))  # s, t は無制限
        else:
            dinic.add_edge(2 * v, 2 * v + 1, 1)  # 各頂点は1回のみ通過

    for u, v in edges:
        dinic.add_edge(2 * u + 1, 2 * v, 1)  # u_out → v_in
        dinic.add_edge(2 * v + 1, 2 * u, 1)  # v_out → u_in（無向グラフの場合）

    return dinic.max_flow(2 * s, 2 * t + 1)

# 使用例
edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]
print(vertex_disjoint_paths(5, edges, 0, 4))  # 2
```

### 画像セグメンテーション（最小カット応用）

```python
def image_segmentation(rows: int, cols: int,
                        foreground_cost: list,
                        background_cost: list,
                        neighbor_penalty: float) -> list:
    """画像のピクセルを前景/背景に分割する最小カットベースの手法
    foreground_cost[i]: ピクセル i を前景にするコスト
    background_cost[i]: ピクセル i を背景にするコスト
    neighbor_penalty: 隣接ピクセルが異なるラベルの場合のペナルティ
    """
    n = rows * cols
    source = n      # 前景ソース
    sink = n + 1    # 背景シンク
    dinic = Dinic(n + 2)

    for i in range(n):
        # source → pixel: 前景コスト（背景にするとカットされる）
        dinic.add_edge(source, i, int(background_cost[i] * 100))
        # pixel → sink: 背景コスト（前景にするとカットされる）
        dinic.add_edge(i, sink, int(foreground_cost[i] * 100))

    # 隣接ペナルティ
    penalty = int(neighbor_penalty * 100)
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            for dr, dc in [(0, 1), (1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    nidx = nr * cols + nc
                    dinic.add_edge(idx, nidx, penalty)
                    dinic.add_edge(nidx, idx, penalty)

    dinic.max_flow(source, sink)

    # s 側 = 前景, t 側 = 背景
    s_side = set(dinic.min_cut(source))
    labels = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if r * cols + c in s_side:
                labels[r][c] = 1  # 前景

    return labels
```

---

## 8. 最小費用流

各辺にフローの単位コストが設定されたネットワークで、指定量のフローを最小コストで流す問題。

```python
class MinCostFlow:
    """最小費用流 (Primal-Dual / SPFA版)
    Successive Shortest Paths アルゴリズム
    """

    def __init__(self, n: int):
        self.n = n
        self.graph = [[] for _ in range(n)]

    def add_edge(self, u: int, v: int, cap: int, cost: int):
        """辺を追加 (容量 cap, 単位コスト cost)"""
        self.graph[u].append([v, cap, cost, len(self.graph[v])])
        self.graph[v].append([u, 0, -cost, len(self.graph[u]) - 1])

    def min_cost_flow(self, s: int, t: int, max_flow: int) -> tuple:
        """s から t に max_flow だけ流す最小コスト
        Returns: (actual_flow, total_cost) or (-1, -1) if infeasible
        """
        total_flow = 0
        total_cost = 0

        while total_flow < max_flow:
            # SPFA (Bellman-Ford の改良) で最短路を求める
            dist = [float('inf')] * self.n
            dist[s] = 0
            in_queue = [False] * self.n
            prev_node = [-1] * self.n
            prev_edge = [-1] * self.n

            queue = deque([s])
            in_queue[s] = True

            while queue:
                u = queue.popleft()
                in_queue[u] = False

                for i, (v, cap, cost, _) in enumerate(self.graph[u]):
                    if cap > 0 and dist[u] + cost < dist[v]:
                        dist[v] = dist[u] + cost
                        prev_node[v] = u
                        prev_edge[v] = i
                        if not in_queue[v]:
                            queue.append(v)
                            in_queue[v] = True

            if dist[t] == float('inf'):
                break  # t に到達不能

            # パス上のボトルネック容量
            path_flow = max_flow - total_flow
            v = t
            while v != s:
                u = prev_node[v]
                e = prev_edge[v]
                path_flow = min(path_flow, self.graph[u][e][1])
                v = u

            # フローの更新
            v = t
            while v != s:
                u = prev_node[v]
                e = prev_edge[v]
                self.graph[u][e][1] -= path_flow
                self.graph[v][self.graph[u][e][3]][1] += path_flow
                v = u

            total_flow += path_flow
            total_cost += path_flow * dist[t]

        return total_flow, total_cost

# 使用例: 最小費用で 2 単位のフローを流す
mcf = MinCostFlow(4)
# s=0, A=1, B=2, t=3
mcf.add_edge(0, 1, 2, 1)  # s→A: 容量2, コスト1
mcf.add_edge(0, 2, 2, 3)  # s→B: 容量2, コスト3
mcf.add_edge(1, 3, 1, 2)  # A→t: 容量1, コスト2
mcf.add_edge(2, 3, 2, 1)  # B→t: 容量2, コスト1
mcf.add_edge(1, 2, 1, 1)  # A→B: 容量1, コスト1

flow, cost = mcf.min_cost_flow(0, 3, 2)
print(f"フロー: {flow}, コスト: {cost}")
# フロー: 2, コスト: 6 (s→A→t: 1*3=3, s→A→B→t: 1*(1+1+1)=3 or similar)
```

### 最小費用流の応用: 仕事の割り当てコスト最小化

```python
def min_cost_assignment(workers: list, tasks: list,
                         costs: dict) -> tuple:
    """各ワーカーに1つのタスクを割り当て、総コストを最小化
    costs[(worker, task)] = コスト
    """
    n_workers = len(workers)
    n_tasks = len(tasks)
    worker_idx = {w: i for i, w in enumerate(workers)}
    task_idx = {t: i for i, t in enumerate(tasks)}

    # ネットワーク: s → workers → tasks → t
    n = n_workers + n_tasks + 2
    source = n - 2
    sink = n - 1
    mcf = MinCostFlow(n)

    for i in range(n_workers):
        mcf.add_edge(source, i, 1, 0)

    for (w, t), cost in costs.items():
        wi = worker_idx[w]
        ti = task_idx[t] + n_workers
        mcf.add_edge(wi, ti, 1, cost)

    for j in range(n_tasks):
        mcf.add_edge(n_workers + j, sink, 1, 0)

    flow, total_cost = mcf.min_cost_flow(source, sink, min(n_workers, n_tasks))

    return flow, total_cost

workers = ["Alice", "Bob", "Charlie"]
tasks = ["Task1", "Task2", "Task3"]
costs = {
    ("Alice", "Task1"): 5, ("Alice", "Task2"): 3, ("Alice", "Task3"): 7,
    ("Bob", "Task1"): 2, ("Bob", "Task2"): 6, ("Bob", "Task3"): 4,
    ("Charlie", "Task1"): 8, ("Charlie", "Task2"): 1, ("Charlie", "Task3"): 3,
}
flow, cost = min_cost_assignment(workers, tasks, costs)
print(f"割り当て数: {flow}, 総コスト: {cost}")
# 最適: Alice-Task2(3), Bob-Task1(2), Charlie-Task3(3) → 総コスト: 8
```

---

## 9. アルゴリズム比較表

| アルゴリズム | 計算量 | 特徴 |
|:---|:---|:---|
| Ford-Fulkerson (DFS) | O(E * max_flow) | 無理数容量で非停止の可能性 |
| Edmonds-Karp (BFS) | O(VE^2) | BFS で最短増加パス |
| Dinic | O(V^2 E) | レベルグラフ + ブロッキングフロー |
| Push-Relabel | O(V^2 E) or O(V^3) | 前置リレーベル法 |
| Hungarian | O(VE) | 二部マッチング特化 |
| Hopcroft-Karp | O(E sqrt(V)) | 二部マッチング最速 |
| MCMC/SPFA | O(VE * flow) | 最小費用流 |

## フロー問題の帰着関係

```
多くの組合せ最適化問題はフロー問題に帰着できる:

最大二部マッチング ←──── 最大流（容量1）
     │
     ↓ Konig の定理
最小頂点被覆 ←───── n - 最大独立集合
     │
     ↓ 補集合
最大独立集合

最小パスカバー ←──── n - 最大マッチング（DAG上）

エッジ素なパス数 ←── 最大流（辺容量1）
頂点素なパス数 ←──── 最大流（頂点分割）
```

## フロー問題の応用

| 問題 | 帰着先 | 容量設定 |
|:---|:---|:---|
| 二部マッチング | 最大流 | 全辺容量1 |
| 最小頂点被覆 | 最大マッチング | Konig の定理 |
| 最大独立集合 | n - 最小頂点被覆 | 補集合 |
| エッジ素なパス数 | 最大流 | 辺容量1 |
| 頂点素なパス数 | 最大流 | 頂点分割（容量1） |
| 最小費用流 | SPFA + 増加パス | 費用付き辺 |
| 最小パスカバー（DAG） | n - 最大マッチング | 入次数・出次数 |
| 画像セグメンテーション | 最小カット | ピクセル間ペナルティ |

---

## 10. 実務での応用例

### スケジューリング問題

```python
def schedule_tasks(n_workers: int, tasks: list) -> int:
    """各タスクに開始時刻・終了時刻・必要人数が設定されている
    全タスクを同時に満たすために必要な最小ワーカー数を求める
    （最大流の逆問題として解く場合もあるが、ここではフローで検証）
    """
    # 時刻を離散化
    times = set()
    for start, end, _ in tasks:
        times.add(start)
        times.add(end)
    times = sorted(times)
    time_idx = {t: i for i, t in enumerate(times)}

    n_times = len(times)
    source = n_times + len(tasks)
    sink = source + 1
    total_nodes = sink + 1
    dinic = Dinic(total_nodes)

    # 時間ノード間を繋ぐ（ワーカーの流れ）
    for i in range(n_times - 1):
        dinic.add_edge(i, i + 1, n_workers)

    # 各タスク: 開始時刻 → タスクノード → 終了時刻
    for task_id, (start, end, required) in enumerate(tasks):
        task_node = n_times + task_id
        si = time_idx[start]
        ei = time_idx[end]
        dinic.add_edge(si, task_node, required)
        dinic.add_edge(task_node, ei, required)

    # source → 最初の時刻, 最後の時刻 → sink
    dinic.add_edge(source, 0, n_workers)
    dinic.add_edge(n_times - 1, sink, n_workers)

    return dinic.max_flow(source, sink)
```

### ネットワークの信頼性分析

```python
def network_reliability(n: int, edges: list, s: int, t: int) -> dict:
    """ネットワークの信頼性指標を計算
    - 辺連結度: s-t 間を切断するために必要な最小辺数
    - 頂点連結度: s-t 間を切断するために必要な最小頂点数
    """
    # 辺連結度 = 最大流（全辺容量1）
    dinic_edge = Dinic(n)
    for u, v in edges:
        dinic_edge.add_edge(u, v, 1)
        dinic_edge.add_edge(v, u, 1)
    edge_connectivity = dinic_edge.max_flow(s, t)

    # 頂点連結度 = 頂点分割後の最大流
    vertex_connectivity = vertex_disjoint_paths(n, edges, s, t)

    return {
        'edge_connectivity': edge_connectivity,
        'vertex_connectivity': vertex_connectivity,
    }

edges = [(0, 1), (0, 2), (1, 3), (2, 3), (1, 2)]
result = network_reliability(4, edges, 0, 3)
print(f"辺連結度: {result['edge_connectivity']}")
print(f"頂点連結度: {result['vertex_connectivity']}")
```

---

## 11. アンチパターン

### アンチパターン1: 逆辺の追加忘れ

```python
# BAD: 残余グラフの逆辺を追加しない
def bad_add_edge(self, u, v, cap):
    self.graph[u][v] = cap
    # 逆辺 graph[v][u] = 0 を追加していない!
    # → フローのキャンセルができず、最大流が求まらない

# GOOD: 必ず逆辺を追加
def good_add_edge(self, u, v, cap):
    self.graph[u][v] += cap
    # 逆辺（初期容量0）を確保
    if v not in self.graph or u not in self.graph[v]:
        self.graph[v][u] += 0
```

### アンチパターン2: 二部マッチングで全探索

```python
# BAD: 全順列を試して最大マッチング → O(n!)
from itertools import permutations
def bad_matching(adj, n, m):
    max_match = 0
    for perm in permutations(range(m)):
        count = sum(1 for i in range(min(n,m)) if perm[i] in adj[i])
        max_match = max(max_match, count)
    return max_match  # n! は大きすぎる

# GOOD: ハンガリアン法 or 最大流 → O(VE) or O(V^2 E)
count, _, _ = hungarian(n, m, adj)
```

### アンチパターン3: Dinic法で iter 配列を初期化し忘れ

```python
# BAD: BFS のたびに iter を初期化しない
def bad_max_flow(self, s, t):
    flow = 0
    self.iter = [0] * self.n  # 一度だけ初期化
    while self.bfs(s, t):
        # iter が前回の値のまま → DFS が正しく動かない
        f = self.dfs(s, t, float('inf'))
        flow += f

# GOOD: BFS のたびに iter を再初期化
def good_max_flow(self, s, t):
    flow = 0
    while self.bfs(s, t):
        self.iter = [0] * self.n  # 毎回初期化
        while True:
            f = self.dfs(s, t, float('inf'))
            if f == 0:
                break
            flow += f
    return flow
```

### アンチパターン4: 無向グラフでの辺の追加ミス

```python
# BAD: 無向辺を有向辺として1回だけ追加
dinic.add_edge(u, v, cap)  # u→v のみ

# GOOD: 無向辺は両方向に追加
dinic.add_edge(u, v, cap)
dinic.add_edge(v, u, cap)
# ※ Dinic の add_edge は内部で逆辺（容量0）を追加しているので
# 無向辺の場合は手動で両方向の辺を追加する必要がある
```


---

## トラブルシューティング

### よくあるエラーと解決策

| エラー | 原因 | 解決策 |
|--------|------|--------|
| 初期化エラー | 設定ファイルの不備 | 設定ファイルのパスと形式を確認 |
| タイムアウト | ネットワーク遅延/リソース不足 | タイムアウト値の調整、リトライ処理の追加 |
| メモリ不足 | データ量の増大 | バッチ処理の導入、ページネーションの実装 |
| 権限エラー | アクセス権限の不足 | 実行ユーザーの権限確認、設定の見直し |
| データ不整合 | 並行処理の競合 | ロック機構の導入、トランザクション管理 |

### デバッグの手順

1. **エラーメッセージの確認**: スタックトレースを読み、発生箇所を特定する
2. **再現手順の確立**: 最小限のコードでエラーを再現する
3. **仮説の立案**: 考えられる原因をリストアップする
4. **段階的な検証**: ログ出力やデバッガを使って仮説を検証する
5. **修正と回帰テスト**: 修正後、関連する箇所のテストも実行する

```python
# デバッグ用ユーティリティ
import logging
import traceback
from functools import wraps

# ロガーの設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def debug_decorator(func):
    """関数の入出力をログ出力するデコレータ"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"呼び出し: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"戻り値: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"例外発生: {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

@debug_decorator
def process_data(items):
    """データ処理（デバッグ対象）"""
    if not items:
        raise ValueError("空のデータ")
    return [item * 2 for item in items]
```

### パフォーマンス問題の診断

パフォーマンス問題が発生した場合の診断手順:

1. **ボトルネックの特定**: プロファイリングツールで計測
2. **メモリ使用量の確認**: メモリリークの有無をチェック
3. **I/O待ちの確認**: ディスクやネットワークI/Oの状況を確認
4. **同時接続数の確認**: コネクションプールの状態を確認

| 問題の種類 | 診断ツール | 対策 |
|-----------|-----------|------|
| CPU負荷 | cProfile, py-spy | アルゴリズム改善、並列化 |
| メモリリーク | tracemalloc, objgraph | 参照の適切な解放 |
| I/Oボトルネック | strace, iostat | 非同期I/O、キャッシュ |
| DB遅延 | EXPLAIN, slow query log | インデックス、クエリ最適化 |

---

## 設計判断ガイド

### 選択基準マトリクス

技術選択を行う際の判断基準を以下にまとめます。

| 判断基準 | 重視する場合 | 妥協できる場合 |
|---------|------------|-------------|
| パフォーマンス | リアルタイム処理、大規模データ | 管理画面、バッチ処理 |
| 保守性 | 長期運用、チーム開発 | プロトタイプ、短期プロジェクト |
| スケーラビリティ | 成長が見込まれるサービス | 社内ツール、固定ユーザー |
| セキュリティ | 個人情報、金融データ | 公開データ、社内利用 |
| 開発速度 | MVP、市場投入スピード | 品質重視、ミッションクリティカル |

### アーキテクチャパターンの選択

```
┌─────────────────────────────────────────────────┐
│              アーキテクチャ選択フロー              │
├─────────────────────────────────────────────────┤
│                                                 │
│  ① チーム規模は？                                │
│    ├─ 小規模（1-5人）→ モノリス                   │
│    └─ 大規模（10人+）→ ②へ                       │
│                                                 │
│  ② デプロイ頻度は？                               │
│    ├─ 週1回以下 → モノリス + モジュール分割         │
│    └─ 毎日/複数回 → ③へ                          │
│                                                 │
│  ③ チーム間の独立性は？                            │
│    ├─ 高い → マイクロサービス                      │
│    └─ 中程度 → モジュラーモノリス                   │
│                                                 │
└─────────────────────────────────────────────────┘
```

### トレードオフの分析

技術的な判断には必ずトレードオフが伴います。以下の観点で分析を行いましょう:

**1. 短期 vs 長期のコスト**
- 短期的に速い方法が長期的には技術的負債になることがある
- 逆に、過剰な設計は短期的なコストが高く、プロジェクトの遅延を招く

**2. 一貫性 vs 柔軟性**
- 統一された技術スタックは学習コストが低い
- 多様な技術の採用は適材適所が可能だが、運用コストが増加

**3. 抽象化のレベル**
- 高い抽象化は再利用性が高いが、デバッグが困難になる場合がある
- 低い抽象化は直感的だが、コードの重複が発生しやすい

```python
# 設計判断の記録テンプレート
class ArchitectureDecisionRecord:
    """ADR (Architecture Decision Record) の作成"""

    def __init__(self, title: str):
        self.title = title
        self.context = ""
        self.decision = ""
        self.consequences = []
        self.alternatives = []

    def set_context(self, context: str):
        """背景と課題の記述"""
        self.context = context
        return self

    def set_decision(self, decision: str):
        """決定内容の記述"""
        self.decision = decision
        return self

    def add_consequence(self, consequence: str, positive: bool = True):
        """結果の追加"""
        self.consequences.append({
            'description': consequence,
            'type': 'positive' if positive else 'negative'
        })
        return self

    def add_alternative(self, name: str, reason_rejected: str):
        """却下した代替案の追加"""
        self.alternatives.append({
            'name': name,
            'reason_rejected': reason_rejected
        })
        return self

    def to_markdown(self) -> str:
        """Markdown形式で出力"""
        md = f"# ADR: {self.title}\n\n"
        md += f"## 背景\n{self.context}\n\n"
        md += f"## 決定\n{self.decision}\n\n"
        md += "## 結果\n"
        for c in self.consequences:
            icon = "✅" if c['type'] == 'positive' else "⚠️"
            md += f"- {icon} {c['description']}\n"
        md += "\n## 却下した代替案\n"
        for a in self.alternatives:
            md += f"- **{a['name']}**: {a['reason_rejected']}\n"
        return md
```
---

## 12. FAQ

### Q1: 最大流最小カット定理は何に使えるか？

**A:** ネットワークの「ボトルネック」の特定に使える。通信ネットワークで最も脆弱なリンク、道路網で渋滞の原因となる区間、パイプラインの最大輸送量など。また、画像のセグメンテーション（前景/背景分離）にも応用される。

### Q2: 最小費用流とは何か？

**A:** 各辺にフローの単位コストが設定されたネットワークで、指定量のフローを最小コストで流す問題。SPFA（Bellman-Ford の改良）で最短パスを見つけながら増加パスを送る。仕事の割り当てコスト最小化、輸送計画などに応用される。

### Q3: 二部マッチングの実際のユースケースは？

**A:** (1) 学生と研究室の配属、(2) タスクとワーカーの割り当て、(3) レビュアーと論文のマッチング、(4) 安定結婚問題（Gale-Shapley）、(5) コンパイラのレジスタ割り当て。Hall の結婚定理で完全マッチングの存在条件を判定できる。

### Q4: Dinic法とEdmonds-Karp法はどちらを使うべきか？

**A:** ほぼ常に Dinic法が推奨される。計算量が O(V^2 E) で Edmonds-Karp の O(VE^2) より良く、二部マッチングでは O(E sqrt(V)) になる。実装の複雑さもほぼ同等。

### Q5: 最大流問題で容量が実数（浮動小数点数）の場合は？

**A:** Ford-Fulkerson法（DFS版）は収束しない可能性がある（Zwick-Paterson の反例）。Edmonds-Karp法（BFS版）やDinic法は有理数容量でも正しく動作する。実用上は整数に丸めるか、有理数演算を使う。

### Q6: Hall の結婚定理とは？

**A:** 二部グラフ G = (L, R, E) において、L のすべての頂点をマッチできる（完全マッチングが存在する）ための必要十分条件は「L の任意の部分集合 S について、S の近傍 N(S) のサイズが |S| 以上」であること。フロー理論の最大流最小カット定理から導出できる。

---

## 13. Push-Relabel法（前置リレーベル法）

Ford-Fulkerson系のアルゴリズムがグローバルな増加パスを探索するのに対し、Push-Relabel法はローカルな操作（push と relabel）を繰り返して最大流を求める。大規模グラフで高い性能を発揮する。

### アルゴリズムの概要

```
Push-Relabel の基本概念:

1. 高さラベル h(v): 各頂点にラベルを割り当てる
   - h(s) = |V|, h(t) = 0
   - 辺 (u,v) に残余容量がある → h(u) ≤ h(v) + 1

2. 超過フロー e(v): 各頂点の流入量 - 流出量
   - s, t 以外で e(v) > 0 の頂点が「アクティブ」

3. Push 操作: アクティブ頂点 u から隣接頂点 v へフローを送る
   条件: h(u) = h(v) + 1 かつ残余容量 > 0
   送る量: min(e(u), 残余容量(u,v))

4. Relabel 操作: Push できない場合にラベルを引き上げる
   h(u) = min(h(v) + 1 | (u,v) に残余容量がある)

処理の流れ:
   s からすべての隣接辺を飽和させる（初期プッシュ）
   → アクティブ頂点が存在する限り Push or Relabel を繰り返す
   → 全頂点のラベルが安定したとき、e(t) が最大流

計算量: O(V^2 E)  ※FIFO選択則で O(V^3)
```

### Push-Relabel法のイメージ図

```
  高さラベルのイメージ（水が高い所から低い所へ流れる）:

  h=6(s)
   |
   | push
   v
  h=1(A) ──push──→ h=0(t)
   |
   | push
   v
  h=1(B) ──push──→ h=0(t)

  Push できない頂点は Relabel で高さを上げる:

  relabel前:          relabel後:
  h=1(A)              h=2(A)  ← 引き上げ
    ↓ push不可           ↓ push可能に
  h=1(B)              h=1(B)
    ↓                    ↓
  h=0(t)              h=0(t)
```

```python
class PushRelabel:
    """Push-Relabel法（FIFO選択則） - O(V^3)"""

    def __init__(self, n: int):
        self.n = n
        self.cap = [[0] * n for _ in range(n)]
        self.flow = [[0] * n for _ in range(n)]

    def add_edge(self, u: int, v: int, c: int):
        self.cap[u][v] += c

    def max_flow(self, s: int, t: int) -> int:
        n = self.n
        height = [0] * n
        excess = [0] * n
        height[s] = n

        # 初期プッシュ: s から全隣接辺を飽和
        for v in range(n):
            if self.cap[s][v] > 0:
                f = self.cap[s][v]
                self.flow[s][v] = f
                self.flow[v][s] = -f
                excess[v] = f
                excess[s] -= f

        # アクティブ頂点のキュー（FIFO）
        active = deque()
        in_queue = [False] * n
        for v in range(n):
            if v != s and v != t and excess[v] > 0:
                active.append(v)
                in_queue[v] = True

        while active:
            u = active.popleft()
            in_queue[u] = False
            self._discharge(u, s, t, height, excess, active, in_queue)

        return excess[t]

    def _discharge(self, u, s, t, height, excess, active, in_queue):
        n = self.n
        while excess[u] > 0:
            pushed = False
            for v in range(n):
                residual = self.cap[u][v] - self.flow[u][v]
                if residual > 0 and height[u] == height[v] + 1:
                    # Push
                    d = min(excess[u], residual)
                    self.flow[u][v] += d
                    self.flow[v][u] -= d
                    excess[u] -= d
                    excess[v] += d
                    if v != s and v != t and not in_queue[v] and excess[v] > 0:
                        active.append(v)
                        in_queue[v] = True
                    pushed = True
                    if excess[u] == 0:
                        break
            if not pushed:
                # Relabel
                min_height = float('inf')
                for v in range(n):
                    if self.cap[u][v] - self.flow[u][v] > 0:
                        min_height = min(min_height, height[v])
                height[u] = min_height + 1

        if excess[u] > 0 and not in_queue[u]:
            active.append(u)
            in_queue[u] = True

# 使用例
pr = PushRelabel(6)
pr.add_edge(0, 1, 10)
pr.add_edge(0, 2, 10)
pr.add_edge(1, 3, 4)
pr.add_edge(1, 4, 8)
pr.add_edge(2, 4, 9)
pr.add_edge(3, 5, 10)
pr.add_edge(4, 3, 6)
pr.add_edge(4, 5, 10)
print(pr.max_flow(0, 5))  # 19
```

### 各アルゴリズムの選択指針

```
使い分けの判断フロー:

問題の種類は？
├── 二部マッチング → Hopcroft-Karp  O(E sqrt(V))
├── 最小費用流     → SPFA + Successive Shortest Paths
├── 単純な最大流   → Dinic法  O(V^2 E)
│    ├── 密グラフ（E ≈ V^2）→ Push-Relabel  O(V^3)
│    └── 疎グラフ（E ≈ V）  → Dinic法
└── 小規模（V < 100）→ どれでも可（Edmonds-Karp が最も実装簡単）
```

---

## 14. 演習問題（3段階）

### 初級: 基本的なフロー計算

**問題 1:** 以下のグラフの最大流を手計算で求めよ。

```
         8         6
    s ────→ A ────→ t
    │               ↑
    │3              │5
    ↓               │
    B ─────────────→ C
           7
```

**ヒント:** 増加パスを1本ずつ見つけて残余グラフを更新する。

**解答例:**

```
パス1: s → A → t  ボトルネック = min(8, 6) = 6
  残余: s→A: 2, A→t: 0, t→A: 6

パス2: s → B → C → t  ボトルネック = min(3, 7, 5) = 3
  残余: s→B: 0, B→C: 4, C→t: 2

パス3: 増加パスなし（s からの辺: s→A: 2 だが A→t: 0、A→... 到達不能）
  → s → A → ... t に到達するパスを探す

  残余グラフで確認:
  s→A: 2, A→t: 0, t→A: 6
  s→B: 0, B→C: 4, C→t: 2
  （逆辺も含む）

  実際は s→A (容量2) を使って... A からは直接 t に行けないが
  逆辺等を考慮すると追加パスなし。

最大流 = 6 + 3 = 9

検証: 最小カットは {s→A: 8, s→B: 3} ではなく
      {A→t: 6, C→t: 5} = 11 でもない。
      実際の最小カット = {s→B: 3, A→t: 6} = 9  ← 一致
```

**問題 2:** Ford-Fulkerson法を使って、以下の 4 頂点グラフの最大流を計算するコードを書け。

```
s=0, A=1, B=2, t=3
辺: s→A(容量10), s→B(容量5), A→B(容量15), A→t(容量10), B→t(容量10)
```

### 中級: 二部マッチングと最小カット

**問題 3:** 5人の学生と5つの研究室がある。以下の志望リストから最大マッチングを求めよ。

```
学生0: 研究室 {0, 1, 3}
学生1: 研究室 {1, 2}
学生2: 研究室 {0, 3}
学生3: 研究室 {2, 4}
学生4: 研究室 {1, 3, 4}
```

**解答例:**

```python
adj = [
    [0, 1, 3],   # 学生0
    [1, 2],       # 学生1
    [0, 3],       # 学生2
    [2, 4],       # 学生3
    [1, 3, 4],    # 学生4
]
count, match_l, match_r = hopcroft_karp(5, 5, adj)
print(f"最大マッチング: {count}")  # 5（完全マッチング可能）
# 例: 学生0→研究室0, 学生1→研究室2, 学生2→研究室3,
#      学生3→研究室4, 学生4→研究室1
```

**問題 4:** ネットワークの最小カット辺を求め、最もボトルネックとなるリンクを特定せよ。

```
6頂点のネットワーク:
s(0)→A(1): 16, s(0)→B(2): 13
A(1)→B(2): 4,  A(1)→C(3): 12
B(2)→A(1): 10, B(2)→D(4): 14
C(3)→B(2): 9,  C(3)→t(5): 20
D(4)→C(3): 7,  D(4)→t(5): 4
```

### 上級: 最小費用流と複合問題

**問題 5:** 3つの工場と4つの店舗がある。各工場の供給量と各店舗の需要量、輸送コストが与えられている。総輸送コストを最小化する輸送計画を最小費用流で求めよ。

```
工場: F1(供給20), F2(供給30), F3(供給25)
店舗: S1(需要15), S2(需要20), S3(需要25), S4(需要15)

輸送コスト（単位あたり）:
      S1  S2  S3  S4
F1:    4   6   8   5
F2:    6   3   5   7
F3:    3   8   4   6
```

**ヒント:** 超過ソース s と超過シンク t を追加して、s→工場（容量=供給量, コスト=0）、店舗→t（容量=需要量, コスト=0）、工場→店舗（容量=十分大, コスト=輸送コスト）のネットワークを構築する。

```python
# 解法のスケルトン
mcf = MinCostFlow(2 + 3 + 4)  # s, t, 3工場, 4店舗
source, sink = 0, 1
factories = [2, 3, 4]      # ノード番号
stores = [5, 6, 7, 8]      # ノード番号
supply = [20, 30, 25]
demand = [15, 20, 25, 15]
cost_matrix = [
    [4, 6, 8, 5],
    [6, 3, 5, 7],
    [3, 8, 4, 6],
]

for i, f in enumerate(factories):
    mcf.add_edge(source, f, supply[i], 0)

for i, f in enumerate(factories):
    for j, s in enumerate(stores):
        mcf.add_edge(f, s, min(supply[i], demand[j]), cost_matrix[i][j])

for j, s in enumerate(stores):
    mcf.add_edge(s, sink, demand[j], 0)

total_demand = sum(demand)  # 75
flow, total_cost = mcf.min_cost_flow(source, sink, total_demand)
print(f"総輸送量: {flow}, 総コスト: {total_cost}")
```

**問題 6:** 有向非巡回グラフ（DAG）上の最小パスカバーを求めよ。最小パスカバー = n - 最大マッチング であることを利用せよ。

```
DAG (6頂点):
0 → 1, 0 → 2
1 → 3
2 → 3, 2 → 4
4 → 5
```

---

## 15. Push-Relabel vs Dinic: 性能比較

| 観点 | Dinic法 | Push-Relabel法 |
|:---|:---|:---|
| 計算量（一般） | O(V^2 E) | O(V^2 E) / O(V^3) |
| 計算量（単位容量） | O(E sqrt(V)) | O(E sqrt(V)) |
| 実装の容易さ | 中程度 | やや複雑 |
| 疎グラフでの性能 | 優れる | 標準的 |
| 密グラフでの性能 | 標準的 | 優れる |
| メモリ使用量 | 隣接リスト（軽量） | 隣接行列版は O(V^2) |
| 競技プログラミング | 最もよく使われる | 稀に使用 |
| 実務（大規模） | 適度 | 高い並列性で有利 |

### 問題規模別の推奨アルゴリズム

| グラフ規模 | 辺数 | 推奨アルゴリズム | 理由 |
|:---|:---|:---|:---|
| V < 100 | E < 1000 | Edmonds-Karp | 実装が最も簡単 |
| V < 1000 | E < 10000 | Dinic | バランスの良い性能 |
| V < 10000 | E < 100000 | Dinic | 疎グラフで高速 |
| V > 10000 | E ~ V^2 | Push-Relabel | 密グラフに強い |
| 二部グラフ | - | Hopcroft-Karp | O(E sqrt(V)) で最速 |
| コスト付き | - | SPFA + SSP | 唯一の選択肢 |

---

## 16. 追加のアンチパターン

### アンチパターン5: 最大流の値だけ求めて経路復元を忘れる

```python
# BAD: 最大流の値は求めたが、実際にどの辺にどれだけ流れたか不明
flow_value = dinic.max_flow(s, t)
# ここで「どの辺に何単位流れたか」を出力しようとしても
# 残余グラフから逆算する必要がある

# GOOD: フロー値と経路を同時に記録する設計
class DinicWithFlowRecovery(Dinic):
    def get_flow_on_edges(self):
        """各辺の実際のフロー量を復元"""
        result = []
        for u in range(self.n):
            for i, (v, cap, rev) in enumerate(self.graph[u]):
                # 元の辺（逆辺でない）のフロー量
                if i % 2 == 0:  # add_edge で追加された順辺
                    original_cap = cap + self.graph[v][rev][1]
                    flow = self.graph[v][rev][1]
                    if flow > 0:
                        result.append((u, v, flow, original_cap))
        return result
```

### アンチパターン6: 多重辺の処理ミス

```python
# BAD: 隣接行列で多重辺を上書き
cap[u][v] = 5   # 1本目
cap[u][v] = 3   # 2本目 → 上書き! 合計8ではなく3になる

# GOOD: 容量を加算する
cap[u][v] += 5   # 1本目
cap[u][v] += 3   # 2本目 → 合計8

# Dinic の隣接リスト版では自動的に別の辺として追加されるので
# この問題は発生しない（隣接行列版のみ注意）
```

---

## 17. 追加 FAQ

### Q7: フロー問題で負の容量はあり得るか？

**A:** 標準的なフロー問題では容量は非負（c(u,v) >= 0）。ただし、下限付きフロー（lower bound flow）問題では各辺に最小フロー量が設定される。この場合、変数変換によって標準的な最大流問題に帰着できる。具体的には、下限 l(u,v) の辺を「容量 c(u,v) - l(u,v) の辺」に変換し、超過ソース・超過シンクを追加する。

### Q8: 最大流問題は線形計画問題として定式化できるか？

**A:** できる。最大流問題は以下の線形計画問題（LP）と等価である:

```
maximize  Σ f(s,v)  （s からの総流出量）
subject to:
  0 ≤ f(u,v) ≤ c(u,v)           （容量制約）
  Σ f(u,v) = Σ f(v,w)  ∀v≠s,t  （フロー保存）
```

最大流最小カット定理は、この LP とその双対問題（最小カット）の強双対性から導かれる。整数容量の場合、LP 緩和の最適解が自動的に整数になる（完全単模行列性）。

### Q9: 動的にグラフが変化する場合の最大流はどう求めるか？

**A:** 辺の追加・削除が発生する動的フロー問題では、毎回ゼロから再計算するのは非効率である。辺追加の場合は、既存のフローを保持したまま追加辺を含む残余グラフで増加パスを探索すればよい（増分計算）。辺削除の場合は、削除辺にフローが流れていなければ何もしない。流れている場合は、そのフロー分を「取り消す」操作（逆方向に流す）が必要になり、やや複雑になる。

---


## FAQ

### Q1: このトピックを学ぶ上で最も重要なポイントは何ですか？

実践的な経験を積むことが最も重要です。理論だけでなく、実際にコードを書いて動作を確認することで理解が深まります。

### Q2: 初心者がよく陥る間違いは何ですか？

基礎を飛ばして応用に進むことです。このガイドで説明している基本概念をしっかり理解してから、次のステップに進むことをお勧めします。

### Q3: 実務ではどのように活用されていますか？

このトピックの知識は、日常的な開発業務で頻繁に活用されます。特にコードレビューやアーキテクチャ設計の際に重要になります。

---

## 18. まとめ

| 項目 | 要点 |
|:---|:---|
| 最大流問題 | 残余グラフ上の増加パスを繰り返し探索 |
| Ford-Fulkerson | BFS版(Edmonds-Karp)で O(VE^2) |
| Dinic法 | レベルグラフ + ブロッキングフローで O(V^2 E) |
| Push-Relabel法 | ローカル操作（push/relabel）で O(V^3)、密グラフに強い |
| 最大流最小カット | 最大流 = 最小カット（双対性） |
| 二部マッチング | 最大流に帰着（容量1）or ハンガリアン法 |
| Hopcroft-Karp | 二部マッチングを O(E sqrt(V)) で解く |
| 最小費用流 | 費用付き辺でコスト最小化 |
| 応用範囲 | 割り当て、被覆、独立集合、パス分離、画像処理、輸送計画 |

### 学習ロードマップ

```
Step 1: 基礎理解
  フローの定義 → 残余グラフ → 増加パス → Ford-Fulkerson
  ↓
Step 2: 高速化
  Edmonds-Karp → Dinic法 → Push-Relabel法
  ↓
Step 3: 応用
  二部マッチング → 最小カット → 頂点分割 → 最小費用流
  ↓
Step 4: 発展
  下限付きフロー → LP双対 → 動的フロー → 近似アルゴリズム
```

---

## 次に読むべきガイド

- [最短経路](../02-algorithms/03-shortest-path.md) -- フローアルゴリズムの前提知識
- [グラフ走査](../02-algorithms/02-graph-traversal.md) -- BFS/DFS の基礎
- [競技プログラミング](../04-practice/01-competitive-programming.md) -- フロー問題の実戦

---

## 参考文献

1. Cormen, T. H. et al. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. -- 第24-26章: ネットワークフローの包括的解説
2. Ford, L. R. & Fulkerson, D. R. (1956). "Maximal flow through a network." *Canadian Journal of Mathematics*. -- 最大流問題の原論文
3. Dinic, E. A. (1970). "Algorithm for solution of a problem of maximum flow in networks with power estimation." *Soviet Mathematics Doklady*. -- Dinic法の原論文
4. Kleinberg, J. & Tardos, E. (2005). *Algorithm Design*. Pearson. -- Chapter 7: Network Flow の応用を豊富に解説
5. Hopcroft, J. E. & Karp, R. M. (1973). "An n^{5/2} Algorithm for Maximum Matchings in Bipartite Graphs." *SIAM Journal on Computing*. -- Hopcroft-Karp法の原論文
6. Konig, D. (1931). "Grafok es matrixok." *Matematikai es Fizikai Lapok*. -- Konigの定理（最小頂点被覆 = 最大マッチング）
7. Goldberg, A. V. & Tarjan, R. E. (1988). "A new approach to the maximum-flow problem." *Journal of the ACM*. -- Push-Relabel法の原論文
8. Ahuja, R. K., Magnanti, T. L. & Orlin, J. B. (1993). *Network Flows: Theory, Algorithms, and Applications*. Prentice Hall. -- ネットワークフロー理論の決定版テキスト
9. Schrijver, A. (2003). *Combinatorial Optimization: Polyhedra and Efficiency*. Springer. -- 最適化理論からの視点でフロー問題を解説
