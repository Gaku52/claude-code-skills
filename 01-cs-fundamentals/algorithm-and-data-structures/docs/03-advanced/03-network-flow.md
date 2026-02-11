# ネットワークフロー

> グラフ上の最大流問題を Ford-Fulkerson法と二部マッチングを通じて理解し、実用的な応用パターンを習得する

## この章で学ぶこと

1. **最大流問題の定義**と残余グラフ・増加パスの概念を理解する
2. **Ford-Fulkerson法**と BFS による Edmonds-Karp 法を正確に実装できる
3. **二部マッチング**を最大流に帰着させ、仕事割当・マッチング問題を解ける

---

## 1. ネットワークフローの基本概念

```
フローネットワーク:
  ・有向グラフ G = (V, E)
  ・容量関数 c(u,v) ≥ 0（各辺の最大流量）
  ・始点 s（ソース）、終点 t（シンク）

制約:
  1. 容量制約: 0 ≤ f(u,v) ≤ c(u,v)
  2. フロー保存: 各頂点で流入量 = 流出量（s,t を除く）

例:
         10        10
    s ────→ A ────→ t
    │       ↑       ↑
    │5      │15     │10
    ↓       │       │
    B ────→ C ────→ D
         10        10

最大流 = 19（s→A→t:10, s→B→C→A→t:0, s→B→C→D→t:5, ...）
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
        """最大流を計算 - O(VE²)"""
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

---

## 4. Dinic法（高速版）

レベルグラフ（BFS で構築）上でブロッキングフローを求める。Edmonds-Karp より高速。

```python
class Dinic:
    """Dinic法 - O(V²E)"""

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
最小カット: {(s,A): 3, (B,C): 6} ではなく
           {(A,t): 2, (s,B): 5} = 7

→ 最大流の後、残余グラフで s から到達可能な頂点集合 S と
  到達不能な頂点集合 T を求めると、S→T の辺が最小カット
```

```python
def min_cut(flow_network: MaxFlow, source: int, sink: int) -> list:
    """最小カットの辺を求める"""
    # まず最大流を計算
    flow_network.max_flow(source, sink)

    # 残余グラフで s から到達可能な頂点
    visited = set()
    queue = deque([source])
    visited.add(source)
    while queue:
        u = queue.popleft()
        for v in flow_network.graph[u]:
            if v not in visited and flow_network.graph[u][v] > 0:
                visited.add(v)
                queue.append(v)

    # S→T の辺（元の容量 > 0 かつ残余容量 = 0）がカット辺
    # visited が S 側、visited でない頂点が T 側
    return visited
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

---

## 7. 応用パターン

### 最小頂点被覆（Konig の定理）

```python
# 二部グラフにおいて: 最小頂点被覆 = 最大マッチング（Konig の定理）
# 頂点被覆: 全ての辺が少なくとも1つの選択頂点に接続

# 最大マッチング後、最小頂点被覆を構築:
# 1. マッチングに含まれない左頂点から出発
# 2. 交互道を辿って到達可能な頂点を求める
# 3. 被覆 = 到達不能な左頂点 + 到達可能な右頂点
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

---

## 8. アルゴリズム比較表

| アルゴリズム | 計算量 | 特徴 |
|:---|:---|:---|
| Ford-Fulkerson (DFS) | O(E * max_flow) | 無理数容量で非停止の可能性 |
| Edmonds-Karp (BFS) | O(VE²) | BFS で最短増加パス |
| Dinic | O(V²E) | レベルグラフ + ブロッキングフロー |
| Push-Relabel | O(V²E) or O(V³) | 前置リレーベル法 |
| Hungarian | O(VE) | 二部マッチング特化 |
| Hopcroft-Karp | O(E√V) | 二部マッチング最速 |

## フロー問題の応用

| 問題 | 帰着先 | 容量設定 |
|:---|:---|:---|
| 二部マッチング | 最大流 | 全辺容量1 |
| 最小頂点被覆 | 最大マッチング | Konig の定理 |
| 最大独立集合 | n - 最小頂点被覆 | 補集合 |
| エッジ素なパス数 | 最大流 | 辺容量1 |
| 頂点素なパス数 | 最大流 | 頂点分割（容量1） |
| 最小費用流 | SPFA + 増加パス | 費用付き辺 |

---

## 9. アンチパターン

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

# GOOD: ハンガリアン法 or 最大流 → O(VE) or O(V²E)
count, _, _ = hungarian(n, m, adj)
```

---

## 10. FAQ

### Q1: 最大流最小カット定理は何に使えるか？

**A:** ネットワークの「ボトルネック」の特定に使える。通信ネットワークで最も脆弱なリンク、道路網で渋滞の原因となる区間、パイプラインの最大輸送量など。また、画像のセグメンテーション（前景/背景分離）にも応用される。

### Q2: 最小費用流とは何か？

**A:** 各辺にフローの単位コストが設定されたネットワークで、指定量のフローを最小コストで流す問題。SPFA（Bellman-Ford の改良）で最短パスを見つけながら増加パスを送る。仕事の割り当てコスト最小化、輸送計画などに応用される。

### Q3: 二部マッチングの実際のユースケースは？

**A:** (1) 学生と研究室の配属、(2) タスクとワーカーの割り当て、(3) レビュアーと論文のマッチング、(4) 安定結婚問題（Gale-Shapley）、(5) コンパイラのレジスタ割り当て。Hall の結婚定理で完全マッチングの存在条件を判定できる。

---

## 11. まとめ

| 項目 | 要点 |
|:---|:---|
| 最大流問題 | 残余グラフ上の増加パスを繰り返し探索 |
| Ford-Fulkerson | BFS版(Edmonds-Karp)で O(VE²) |
| Dinic法 | レベルグラフ + ブロッキングフローで O(V²E) |
| 最大流最小カット | 最大流 = 最小カット（双対性） |
| 二部マッチング | 最大流に帰着（容量1）or ハンガリアン法 |
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
