# ネットワークフロー

> グラフ上の最大流問題を Ford-Fulkerson法・Dinic法・二部マッチング・最小費用流を通じて理解し、実用的な応用パターンを習得する

## この章で学ぶこと

1. **最大流問題の定義**と残余グラフ・増加パスの概念を理解する
2. **Ford-Fulkerson法**と BFS による Edmonds-Karp 法を正確に実装できる
3. **Dinic法**で高速に最大流を求める手法を実装できる
4. **二部マッチング**を最大流に帰着させ、仕事割当・マッチング問題を解ける
5. **最大流最小カット定理・最小費用流**の理論と応用を理解する

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

## 13. まとめ

| 項目 | 要点 |
|:---|:---|
| 最大流問題 | 残余グラフ上の増加パスを繰り返し探索 |
| Ford-Fulkerson | BFS版(Edmonds-Karp)で O(VE^2) |
| Dinic法 | レベルグラフ + ブロッキングフローで O(V^2 E) |
| 最大流最小カット | 最大流 = 最小カット（双対性） |
| 二部マッチング | 最大流に帰着（容量1）or ハンガリアン法 |
| Hopcroft-Karp | 二部マッチングを O(E sqrt(V)) で解く |
| 最小費用流 | 費用付き辺でコスト最小化 |
| 応用範囲 | 割り当て、被覆、独立集合、パス分離、画像処理 |

---

## 次に読むべきガイド

- [最短経路](../02-algorithms/03-shortest-path.md) -- フローアルゴリズムの前提知識
- [グラフ走査](../02-algorithms/02-graph-traversal.md) -- BFS/DFS の基礎
- [競技プログラミング](../04-practice/01-competitive-programming.md) -- フロー問題の実戦

---

## 参考文献

1. Cormen, T. H. et al. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. -- 第24-26章
2. Ford, L. R. & Fulkerson, D. R. (1956). "Maximal flow through a network." *Canadian Journal of Mathematics*.
3. Dinic, E. A. (1970). "Algorithm for solution of a problem of maximum flow in networks with power estimation." *Soviet Mathematics Doklady*.
4. Kleinberg, J. & Tardos, E. (2005). *Algorithm Design*. Pearson. -- Chapter 7: Network Flow
5. Hopcroft, J. E. & Karp, R. M. (1973). "An n^{5/2} Algorithm for Maximum Matchings in Bipartite Graphs." *SIAM Journal on Computing*.
6. Konig, D. (1931). "Grafok es matrixok." *Matematikai es Fizikai Lapok*.
