# 計算複雑性理論

> P ≠ NP 問題はCS最大の未解決問題であり、その答えは暗号、最適化、AI の基盤を根本から変える可能性がある。

## この章で学ぶこと

- [ ] P, NP, NP完全, NP困難の定義を説明できる
- [ ] P ≠ NP 問題の意味と帰結を理解する
- [ ] NP完全問題への実務的な対処法を知る
- [ ] Cook-Levin定理の証明の概略を理解する
- [ ] 主要な複雑性クラス（PSPACE, BPP, BQP等）を把握する
- [ ] 近似アルゴリズムと近似困難性を理解する
- [ ] パラメータ化計算量（FPT）の基本を知る
- [ ] 空間計算量の概念と重要な定理を理解する

---

## 1. 計算複雑性理論の基礎

### 1.1 なぜ複雑性を考えるのか

```
計算可能性と計算複雑性の違い:

  計算可能性理論:
  - 問い: 「この問題は原理的に解けるか？」
  - 答え: 決定可能 / 決定不能
  - 時間やメモリの制約は考えない

  計算複雑性理論:
  - 問い: 「この問題は効率的に解けるか？」
  - 答え: 多項式時間 / 指数時間 / etc.
  - 計算資源（時間、空間）の制約を考える

  実務的な重要性:
  ┌────────────────────────────────────────────┐
  │ 問題の大きさ n = 100 の場合:               │
  │                                            │
  │ O(n)      = 100        → 瞬時              │
  │ O(n²)     = 10,000     → 瞬時              │
  │ O(n³)     = 1,000,000  → 約0.001秒         │
  │ O(n⁵)     = 10^10      → 約10秒            │
  │ O(2^n)    = 10^30      → 宇宙の寿命を超える │
  │ O(n!)     = 10^158     → 完全に不可能       │
  └────────────────────────────────────────────┘

  → 多項式時間と指数時間の差は実用上「解ける」と「解けない」の差
```

### 1.2 時間計算量の形式的定義

```
時間計算量の形式的定義:

  定義: チューリングマシンMの時間計算量 f(n):
  Mが長さnの任意の入力に対して、f(n)ステップ以内に停止する

  漸近的評価:
  - O記法（上界）: f(n) = O(g(n)) ⟺ ∃c,n₀: f(n) ≤ c·g(n) for n ≥ n₀
  - Ω記法（下界）: f(n) = Ω(g(n)) ⟺ ∃c,n₀: f(n) ≥ c·g(n) for n ≥ n₀
  - Θ記法（tight）: f(n) = Θ(g(n)) ⟺ f(n) = O(g(n)) かつ f(n) = Ω(g(n))

  時間複雑性クラス:
  TIME(f(n)) = { L | Lを O(f(n)) 時間で決定するTMが存在する }

  P = ∪_{k≥0} TIME(n^k)
    = TIME(n) ∪ TIME(n²) ∪ TIME(n³) ∪ ...

  EXPTIME = ∪_{k≥0} TIME(2^{n^k})

  注意: 多テープTMの時間計算量は1テープTMの二乗以内
  → 計算モデルの選択はクラスPに影響しない（多項式の範囲内で吸収）
```

---

## 2. 複雑性クラス

### 2.1 クラスP

```
P（Polynomial time）:
  多項式時間で「解ける」問題のクラス

  形式的定義:
  P = { L | あるk ≥ 0 に対して、L ∈ TIME(n^k) }

  Pに属する問題の例:

  ┌─────────────────────────────────────────────────┐
  │ 問題              │ アルゴリズム       │ 計算量     │
  ├───────────────────┼──────────────────┼──────────┤
  │ ソート             │ マージソート       │ O(n log n) │
  │ 最短経路           │ ダイクストラ法     │ O(V² + E)  │
  │ 最大フロー         │ Ford-Fulkerson    │ O(VE²)     │
  │ 素数判定           │ AKSアルゴリズム   │ O(n^6)     │
  │ 線形計画法         │ 楕円体法          │ 多項式      │
  │ 最大マッチング     │ Edmonds算法       │ O(V³)      │
  │ 2-SAT             │ 含意グラフ+SCC    │ O(n + m)   │
  │ 2-彩色判定         │ BFS/DFS          │ O(V + E)   │
  │ 連結性判定         │ BFS/DFS          │ O(V + E)   │
  │ 最小全域木         │ Kruskal / Prim   │ O(E log V) │
  │ 行列式の計算       │ ガウス消去法      │ O(n³)      │
  │ パターンマッチング │ KMP法            │ O(n + m)   │
  └─────────────────────────────────────────────────┘

  Pの意味:
  - 「効率的に解ける」の理論的定義
  - 実務上は O(n⁵) でも遅すぎることが多い
  - しかし理論的には多項式と指数の境界が本質的
```

```python
# Pに属する問題の実装例

# 1. 2-SAT（多項式時間で解ける充足可能性問題）
from collections import defaultdict

class TwoSAT:
    """
    2-SAT問題の解法
    時間計算量: O(n + m)（含意グラフ上のSCC分解）

    2-SATはPに属する（多項式時間で解ける）
    一方、3-SATはNP完全
    """

    def __init__(self, n):
        """n: 変数の数"""
        self.n = n
        self.graph = defaultdict(list)
        self.reverse_graph = defaultdict(list)

    def _neg(self, x):
        """変数xの否定を返す"""
        return x + self.n if x < self.n else x - self.n

    def add_clause(self, x, y):
        """
        節 (x ∨ y) を追加
        含意: ¬x → y, ¬y → x
        """
        neg_x = self._neg(x)
        neg_y = self._neg(y)
        self.graph[neg_x].append(y)
        self.graph[neg_y].append(x)
        self.reverse_graph[y].append(neg_x)
        self.reverse_graph[x].append(neg_y)

    def solve(self):
        """
        SCC分解による2-SAT解法

        返り値: 充足可能な場合は各変数の真偽値のリスト、
                充足不可能な場合はNone
        """
        # Kosarajuのアルゴリズムでトポロジカル順序を求める
        visited = set()
        order = []

        def dfs1(v):
            visited.add(v)
            for u in self.graph[v]:
                if u not in visited:
                    dfs1(u)
            order.append(v)

        for v in range(2 * self.n):
            if v not in visited:
                dfs1(v)

        # 逆グラフでSCC分解
        comp = [-1] * (2 * self.n)
        comp_id = 0

        def dfs2(v, c):
            comp[v] = c
            for u in self.reverse_graph[v]:
                if comp[u] == -1:
                    dfs2(u, c)

        for v in reversed(order):
            if comp[v] == -1:
                dfs2(v, comp_id)
                comp_id += 1

        # 充足可能性チェック
        for i in range(self.n):
            if comp[i] == comp[i + self.n]:
                return None  # x と ¬x が同じSCC → 充足不可能

        # 解の構築
        result = []
        for i in range(self.n):
            result.append(comp[i] > comp[i + self.n])

        return result


# 使用例
sat = TwoSAT(3)  # 3変数: x₀, x₁, x₂
# (x₀ ∨ x₁) ∧ (¬x₀ ∨ x₂) ∧ (¬x₁ ∨ ¬x₂)
sat.add_clause(0, 1)      # x₀ ∨ x₁
sat.add_clause(3, 2)      # ¬x₀ ∨ x₂  (3 = ¬0)
sat.add_clause(4, 5)      # ¬x₁ ∨ ¬x₂  (4 = ¬1, 5 = ¬2)

solution = sat.solve()
print(f"解: {solution}")  # [True, True, False] など


# 2. 最大二部マッチング（Hopcroft-Karp）
class BipartiteMatching:
    """
    二部グラフの最大マッチング
    Hopcroft-Karpアルゴリズム: O(E√V)
    → Pに属する
    """

    def __init__(self, n, m):
        """n: 左頂点数、m: 右頂点数"""
        self.n = n
        self.m = m
        self.graph = defaultdict(list)
        self.match_left = [-1] * n
        self.match_right = [-1] * m

    def add_edge(self, u, v):
        """左のuから右のvへの辺を追加"""
        self.graph[u].append(v)

    def bfs(self):
        """BFSで増加パスの長さを計算"""
        from collections import deque
        dist = [-1] * self.n
        queue = deque()

        for u in range(self.n):
            if self.match_left[u] == -1:
                dist[u] = 0
                queue.append(u)

        found = False
        while queue:
            u = queue.popleft()
            for v in self.graph[u]:
                w = self.match_right[v]
                if w == -1:
                    found = True
                elif dist[w] == -1:
                    dist[w] = dist[u] + 1
                    queue.append(w)

        self.dist = dist
        return found

    def dfs(self, u):
        """DFSで増加パスを見つけてマッチングを更新"""
        for v in self.graph[u]:
            w = self.match_right[v]
            if w == -1 or (self.dist[w] == self.dist[u] + 1 and self.dfs(w)):
                self.match_left[u] = v
                self.match_right[v] = u
                return True
        self.dist[u] = -1
        return False

    def solve(self):
        """最大マッチングを求める"""
        matching = 0
        while self.bfs():
            for u in range(self.n):
                if self.match_left[u] == -1:
                    if self.dfs(u):
                        matching += 1
        return matching
```

### 2.2 クラスNP

```
NP（Nondeterministic Polynomial time）:
  多項式時間で「解の正しさを検証できる」問題のクラス

  形式的定義（検証器ベース）:
  L ∈ NP ⟺ あるTM V（検証器）と多項式 p が存在して:
    - w ∈ L ⟺ あるc (|c| ≤ p(|w|)) が存在して V(w, c) = accept
    - Vは多項式時間で動作する
    - c は「証拠」「証明書」「witness」と呼ばれる

  形式的定義（非決定性TMベース）:
  NP = ∪_{k≥0} NTIME(n^k)
  NTIME(f(n)) = 非決定性TMが f(n) ステップ以内で受理する言語の集合

  NPの直感的理解:
  ┌────────────────────────────────────────────────────┐
  │ 「解を見つけるのは難しいかもしれないが、              │
  │  解が与えられればそれが正しいかは素早く確認できる」    │
  │                                                    │
  │  数独: 解くのは大変だが、完成した盤面の検証は簡単     │
  │  素因数分解: 因数を見つけるのは大変だが、             │
  │            掛け算の検証は簡単                        │
  │  パズル: 解くのは大変だが、正解かの確認は簡単         │
  └────────────────────────────────────────────────────┘

  NP に属する問題の例:

  問題             │ 証拠（witness）         │ 検証方法
  ─────────────────┼────────────────────────┼──────────────
  SAT              │ 変数の割り当て          │ 各節を評価
  ハミルトン閉路    │ 頂点の順列             │ 全辺が存在するか確認
  グラフ彩色        │ 各頂点の色             │ 隣接頂点が同色でないか
  ナップサック      │ 選んだ品物の集合       │ 重さと価値を合計
  クリーク          │ 頂点の部分集合         │ 全ペアが辺で結ばれるか
  巡回セールスマン  │ 都市の訪問順           │ 総距離を計算
  部分集合和        │ 元の部分集合           │ 和を計算
  整数計画問題      │ 変数への整数割り当て   │ 制約を確認
```

```python
# NP問題の検証器の実装例

class NPVerifier:
    """NP問題の検証器コレクション"""

    @staticmethod
    def verify_sat(formula, assignment):
        """
        SAT（充足可能性問題）の検証器
        formula: CNF形式の論理式 [[1, -2, 3], [-1, 3], ...]
                 正の数 = 変数、負の数 = 否定
        assignment: {変数番号: True/False}

        検証の計算量: O(n × m)（n=変数数、m=節数）
        → 多項式時間で検証可能 → NPに属する
        """
        for clause in formula:
            satisfied = False
            for literal in clause:
                var = abs(literal)
                value = assignment.get(var, False)
                if literal > 0 and value:
                    satisfied = True
                    break
                if literal < 0 and not value:
                    satisfied = True
                    break
            if not satisfied:
                return False
        return True

    @staticmethod
    def verify_hamiltonian_cycle(graph, cycle):
        """
        ハミルトン閉路の検証器
        graph: 隣接リスト {頂点: [隣接頂点, ...]}
        cycle: 頂点のリスト

        検証: O(V)
        """
        n = len(graph)

        # 全頂点を1回ずつ訪問しているか
        if len(cycle) != n:
            return False
        if set(cycle) != set(graph.keys()):
            return False

        # 各辺が存在するか
        for i in range(n):
            u = cycle[i]
            v = cycle[(i + 1) % n]
            if v not in graph[u]:
                return False

        return True

    @staticmethod
    def verify_graph_coloring(graph, coloring, k):
        """
        k-彩色の検証器
        graph: 隣接リスト
        coloring: {頂点: 色}
        k: 使用可能な色の数

        検証: O(V + E)
        """
        # 色の数がk以下か
        if len(set(coloring.values())) > k:
            return False

        # 全頂点に色が割り当てられているか
        if set(coloring.keys()) != set(graph.keys()):
            return False

        # 隣接する頂点が同じ色でないか
        for u in graph:
            for v in graph[u]:
                if coloring[u] == coloring[v]:
                    return False

        return True

    @staticmethod
    def verify_subset_sum(numbers, target, subset):
        """
        部分集合和問題の検証器
        numbers: 数の集合
        target: 目標の和
        subset: 選んだ要素のインデックス集合

        検証: O(n)
        """
        # subsetがnumbersの部分集合か
        if not all(0 <= i < len(numbers) for i in subset):
            return False

        # 和がtargetに等しいか
        return sum(numbers[i] for i in subset) == target

    @staticmethod
    def verify_clique(graph, clique, k):
        """
        k-クリークの検証器
        graph: 隣接リスト
        clique: 頂点の集合
        k: クリークの大きさ

        検証: O(k²)
        """
        if len(clique) != k:
            return False

        # 全ペアが辺で結ばれているか
        clique_list = list(clique)
        for i in range(len(clique_list)):
            for j in range(i + 1, len(clique_list)):
                u, v = clique_list[i], clique_list[j]
                if v not in graph.get(u, []):
                    return False
        return True


# 検証の実演
verifier = NPVerifier()

# SAT の検証
formula = [[1, -2, 3], [-1, 3], [2, -3]]  # (x₁ ∨ ¬x₂ ∨ x₃) ∧ ...
assignment = {1: True, 2: False, 3: True}
print(f"SAT検証: {verifier.verify_sat(formula, assignment)}")  # True

# 部分集合和の検証
numbers = [3, 7, 1, 8, 4]
target = 12
subset = {0, 2, 3}  # 3 + 1 + 8 = 12
print(f"部分集合和検証: {verifier.verify_subset_sum(numbers, target, subset)}")
```

### 2.3 NP完全とNP困難

```
NP完全（NP-Complete）:
  NPの中で「最も難しい」問題のクラス

  形式的定義:
  L が NP完全 ⟺
    1. L ∈ NP
    2. 任意の L' ∈ NP に対して L' ≤ₚ L
       （全てのNP問題がLに多項式時間帰着可能）

  1つのNP完全問題が多項式時間で解ければ:
  → 全てのNP問題が多項式時間で解ける → P = NP

  NP困難（NP-Hard）:
  L が NP困難 ⟺
    任意の L' ∈ NP に対して L' ≤ₚ L

  NP完全 = NP ∩ NP困難

  ┌──────────────────────────────────────────────────────┐
  │                                                      │
  │  P ≠ NP の場合のベン図:                               │
  │                                                      │
  │  ┌─────────────── NP困難 ──────────────────────┐     │
  │  │                                              │     │
  │  │     ┌──────────── NP ────────────────┐      │     │
  │  │     │                                │      │     │
  │  │     │  ┌── P ──┐  ┌─── NP完全 ──┐  │      │     │
  │  │     │  │ソート   │  │ SAT         │  │      │     │
  │  │     │  │最短経路 │  │ TSP(判定版) │  │      │     │
  │  │     │  │素数判定 │  │ グラフ彩色  │  │      │     │
  │  │     │  └────────┘  └─────────────┘  │      │     │
  │  │     │     NP中間（存在するなら）      │      │     │
  │  │     │     ・グラフ同型               │      │     │
  │  │     │     ・素因数分解               │      │     │
  │  │     └────────────────────────────────┘      │     │
  │  │  ┌─── NPに属さないNP困難 ─┐                 │     │
  │  │  │  停止問題               │                 │     │
  │  │  │  QBF（PSPACE完全）      │                 │     │
  │  │  └─────────────────────────┘                 │     │
  │  └──────────────────────────────────────────────┘     │
  │                                                      │
  └──────────────────────────────────────────────────────┘

  NP中間（NP-Intermediate）:
  P ≠ NP の場合、NP完全でもPにも属さない問題が存在する（Ladnerの定理）
  候補:
  - グラフ同型問題: NP完全かPか未解明
  - 素因数分解: NP完全とは考えられていない（量子ではP）
  - 離散対数: 素因数分解と同様の位置づけ
```

### 2.4 Cook-Levin定理

```
Cook-Levin定理:
  SAT（充足可能性問題）はNP完全である

  意義:
  - 最初のNP完全問題の証明
  - 以後、他の問題のNP完全性はSATからの帰着で証明

  証明の概略:

  1. SAT ∈ NP: 割り当てが与えられれば多項式時間で検証可能 ✓

  2. 任意のNP問題 L について L ≤ₚ SAT を示す:

  アイデア:
  - L ∈ NP なので、多項式時間の検証器 V が存在
  - V(w, c) の計算過程をブール式として符号化する
  - 各ステップのテープ内容、ヘッド位置、状態を変数で表現
  - 遷移関数の制約を節（clause）で表現

  構築する変数:
  - x_{i,j,s}: 時刻iでテープ位置jに記号sがある
  - h_{i,j}: 時刻iでヘッドが位置jにある
  - q_{i,s}: 時刻iで状態がsである

  構築する節:
  - 初期条件: 入力wがテープに書かれている
  - 遷移条件: 各ステップが遷移関数に従う
  - 受理条件: 最終的に受理状態に到達する

  結果の論理式のサイズ: O(t(n)² × |Σ|) ← 多項式

  → 任意のNP問題をSATに多項式時間で変換できる
  → SAT は NP完全 ∎
```

### 2.5 代表的なNP完全問題と帰着の連鎖

```
NP完全問題の帰着の連鎖:

  SAT (Cook-Levin定理)
   │
   ├──→ 3-SAT
   │     │
   │     ├──→ 独立集合 ──→ 頂点被覆 ──→ 集合被覆
   │     │
   │     ├──→ 3-彩色 ──→ k-彩色 (k ≥ 3)
   │     │
   │     ├──→ クリーク
   │     │
   │     ├──→ ハミルトン閉路 ──→ TSP（判定版）
   │     │
   │     └──→ 部分集合和 ──→ ナップサック ──→ 分割問題
   │
   └──→ 回路充足可能性 (Circuit-SAT)

  帰着のポイント:
  - A ≤ₚ B（AをBに帰着）: Bが解ければAも解ける
  - 新しい問題XのNP完全性を示すには:
    1. X ∈ NP を示す
    2. 既知のNP完全問題 Y について Y ≤ₚ X を示す
```

```python
# NP完全問題間の帰着の実装例

# 帰着1: 3-SAT → 独立集合
def reduce_3sat_to_independent_set(clauses, num_vars):
    """
    3-SATを独立集合問題に帰着する

    入力: 3-CNF式（各節が最大3リテラル）
    出力: グラフGと整数k

    (formula ∈ 3-SAT) ⟺ (G has independent set of size k)
    """
    # 各節の各リテラルに1つの頂点を作成
    vertices = []
    for i, clause in enumerate(clauses):
        for j, literal in enumerate(clause):
            vertices.append((i, j, literal))

    k = len(clauses)  # 独立集合のサイズ = 節の数

    # 辺の構築
    edges = []
    for idx1, v1 in enumerate(vertices):
        for idx2, v2 in enumerate(vertices):
            if idx1 >= idx2:
                continue
            # 同じ節内のリテラル間に辺を引く（1節から1つだけ選ぶ）
            if v1[0] == v2[0]:
                edges.append((idx1, idx2))
            # 矛盾するリテラル間に辺を引く（x と ¬x）
            elif v1[2] == -v2[2]:
                edges.append((idx1, idx2))

    return vertices, edges, k


# 帰着2: 独立集合 → 頂点被覆
def reduce_independent_set_to_vertex_cover(graph, n, k):
    """
    独立集合を頂点被覆に帰着する

    Gにサイズkの独立集合がある ⟺ Gにサイズ(n-k)の頂点被覆がある

    証明:
    SがGの独立集合 → V\Sが頂点被覆
    （SにはS内の2頂点を結ぶ辺がない → 全辺はV\Sの頂点を含む）
    """
    # グラフはそのまま、目標サイズを変換
    return graph, n - k


# 帰着3: 頂点被覆 → 集合被覆
def reduce_vertex_cover_to_set_cover(graph, n, k):
    """
    頂点被覆を集合被覆に帰着する

    全体集合U = 辺の集合
    各頂点vに対して、vに接続する辺の集合Sv
    k個以下のSvで全辺を被覆 ⟺ サイズk以下の頂点被覆が存在
    """
    # 全体集合 = 辺の集合
    universe = set()
    for u in graph:
        for v in graph[u]:
            edge = (min(u, v), max(u, v))
            universe.add(edge)

    # 各頂点の集合
    sets = {}
    for u in graph:
        sets[u] = set()
        for v in graph[u]:
            edge = (min(u, v), max(u, v))
            sets[u].add(edge)

    return universe, sets, k


# 帰着4: 部分集合和 → 分割問題
def reduce_subset_sum_to_partition(numbers, target):
    """
    部分集合和を分割問題に帰着する

    分割問題: 集合Sを2つの部分集合に分割して、
    両方の和が等しくなるか？

    帰着: S = numbers ∪ {|sum(numbers) - 2*target|}
    """
    total = sum(numbers)
    diff = abs(total - 2 * target)
    new_numbers = numbers + [diff]

    # new_numbersを和が等しい2つに分割可能
    # ⟺ numbersの部分集合の和がtargetに等しい
    return new_numbers
```

---

## 3. P ≠ NP 問題

### 3.1 問題の意味と帰結

```
P ≠ NP 問題:

  問い: P = NP か？

  クレイ数学研究所のミレニアム懸賞問題（100万ドル）

  もし P = NP が証明されたら:

  ┌─────────────────────────────────────────────┐
  │ 暗号学の崩壊                                 │
  │ - RSA, 楕円曲線暗号が破れる                  │
  │ - 全てのオンラインセキュリティが無効に        │
  │ - 新しい暗号体系の構築が必要                  │
  ├─────────────────────────────────────────────┤
  │ 最適化の革命                                 │
  │ - スケジューリング、経路最適化が完全に解ける  │
  │ - 創薬：タンパク質構造予測が完璧に            │
  │ - 物流、製造の完全最適化                      │
  ├─────────────────────────────────────────────┤
  │ AI/数学の飛躍                                │
  │ - 定理の自動証明が実現                       │
  │ - NP探索 = 最適な解の発見が容易に             │
  │ - 創造性の定義が揺らぐ                       │
  └─────────────────────────────────────────────┘

  もし P ≠ NP が証明されたら:

  ┌─────────────────────────────────────────────┐
  │ 安心の確認                                   │
  │ - 暗号は（理論的に）安全                      │
  │ - NP完全問題には本質的な困難性がある          │
  │ - 近似・ヒューリスティックの研究が重要に      │
  ├─────────────────────────────────────────────┤
  │ 新しい理論的ツール                           │
  │ - 証明手法が他の未解決問題に応用可能          │
  │ - 計算複雑性の理論が大きく前進               │
  └─────────────────────────────────────────────┘

  現在の状況:
  - ほとんどの研究者は P ≠ NP を予想（Gasarchの調査: 約83%）
  - しかし証明の目処は立っていない
  - P ≠ NP の証明には新しい数学的手法が必要
  - 既存の手法（対角化、相対化）では不十分
    → Baker-Gill-Solovay定理（1975）
  - 自然な証明（Natural Proofs）も不十分
    → Razborov-Rudich定理（1997）
```

### 3.2 P ≠ NP 証明の壁

```
P ≠ NP の証明が困難な理由:

  1. 相対化の壁（Relativization Barrier）
     Baker-Gill-Solovay (1975):
     - あるオラクルAに対して P^A = NP^A
     - 別のオラクルBに対して P^B ≠ NP^B
     → 対角化のような「相対化する」手法では
       P ≠ NP を証明も反証もできない

  2. 自然な証明の壁（Natural Proofs Barrier）
     Razborov-Rudich (1997):
     - 一方向性関数が存在するなら（暗号学的仮定）
     - 「自然な証明」では回路下界を示せない
     → 多くの従来の手法が使えない

  3. 代数化の壁（Algebrization Barrier）
     Aaronson-Wigderson (2009):
     - 対角化の拡張（代数的拡張を許す）でも不十分
     → さらに新しい手法が必要

  有望なアプローチ:
  - Geometric Complexity Theory (GCT): 代数幾何を使用
  - 回路下界の研究: 特定の回路モデルでの下界
  - 通信計算量: 関連する下界の証明
  - しかし、完全な証明にはまだ程遠い

  現在の部分的成果:
  - P ≠ EXPTIME（時間階層定理）
  - NP ⊄ SIZE(n^k) for any fixed k（Kannan, 1982）
  - ACC⁰ の回路下界（Williams, 2011）
```

---

## 4. その他の複雑性クラス

### 4.1 空間計算量

```
空間計算量:

  SPACE(f(n)) = { L | Lを O(f(n)) 空間で決定するTMが存在する }
  NSPACE(f(n)) = 非決定性TMでの空間計算量

  主要クラス:
  ┌────────────────────────────────────────────────────┐
  │ L (LOGSPACE) = SPACE(log n)                        │
  │   例: 到達可能性判定（有向グラフ、無向は未解明→解明）│
  │   例: パターンマッチング                            │
  │                                                    │
  │ NL = NSPACE(log n)                                 │
  │   例: 到達可能性判定（有向グラフ）                   │
  │   NL = co-NL（Immerman-Szelepcsényi定理）         │
  │                                                    │
  │ PSPACE = ∪_{k≥0} SPACE(n^k)                       │
  │   例: QBF（量化ブール式）                           │
  │   例: 一般化されたチェス、囲碁                      │
  │   PSPACE完全問題: QBF                               │
  │                                                    │
  │ EXPSPACE = ∪_{k≥0} SPACE(2^{n^k})                 │
  └────────────────────────────────────────────────────┘

  包含関係（分かっていること）:

  L ⊆ NL ⊆ P ⊆ NP ⊆ PSPACE ⊆ EXPTIME ⊆ EXPSPACE

  分かっていないこと:
  - L = NL ? （未解決）
  - P = NP ? （未解決）
  - NP = PSPACE ? （未解決）
  - ただし L ≠ PSPACE と P ≠ EXPTIME は証明済み

  Savitchの定理:
  NSPACE(f(n)) ⊆ SPACE(f(n)²)
  → 非決定性は空間を二乗にしか増やさない
  → PSPACE = NPSPACE
```

```python
# 空間効率的なアルゴリズムの例

# Savitchの定理に基づく到達可能性判定
def reachability_savitch(graph, start, end, n):
    """
    Savitchの定理に基づく到達可能性判定
    空間計算量: O(log²n)（再帰の深さ × 各レベルの使用量）

    アイデア: 「startからendに2^i歩以内で到達できるか？」を
    中間地点midを全て試して再帰的に判定
    """
    def can_reach(u, v, steps):
        """uからvにstepsステップ以内で到達可能か？"""
        if steps == 0:
            return u == v
        if steps == 1:
            return u == v or v in graph.get(u, [])

        half = steps // 2
        # 全ての中間地点を試す
        for mid in range(n):
            if can_reach(u, mid, half) and can_reach(mid, v, steps - half):
                return True
        return False

    return can_reach(start, end, n)


# PSPACE完全: QBF（量化ブール式）
def solve_qbf(formula):
    """
    量化ブール式の評価
    PSPACE完全問題

    例: ∀x ∃y (x ∨ y) ∧ (¬x ∨ ¬y)

    空間効率的に解ける（多項式空間）が
    時間は指数的になりうる
    """
    if not formula.quantifiers:
        # 量化子がなくなったら、ブール式を評価
        return evaluate_boolean(formula.expression, formula.assignment)

    quantifier, variable = formula.quantifiers[0]
    remaining = formula.without_first_quantifier()

    if quantifier == 'forall':
        # ∀x: x=True と x=False の両方で真
        remaining.assignment[variable] = True
        result_true = solve_qbf(remaining)
        remaining.assignment[variable] = False
        result_false = solve_qbf(remaining)
        return result_true and result_false

    elif quantifier == 'exists':
        # ∃x: x=True または x=False のどちらかで真
        remaining.assignment[variable] = True
        result_true = solve_qbf(remaining)
        if result_true:
            return True
        remaining.assignment[variable] = False
        return solve_qbf(remaining)
```

### 4.2 確率的複雑性クラス

```
確率的複雑性クラス:

  BPP（Bounded-Error Probabilistic Polynomial time）:
  - ランダム化TMが多項式時間で解ける
  - 正答率 ≥ 2/3（定数回繰り返せば指数的に改善可能）
  - P ⊆ BPP ⊆ PSPACE

  RP（Randomized Polynomial time）:
  - YES: 確率 ≥ 1/2 で accept
  - NO: 確率 1 で reject（片側誤りなし）
  - 例: Miller-Rabin素数判定（合成数を正しく判定）

  ZPP（Zero-Error Probabilistic Polynomial time）:
  - 誤りなし、期待実行時間が多項式
  - ZPP = RP ∩ co-RP

  PP（Probabilistic Polynomial time）:
  - 正答率 > 1/2（ギリギリでもOK）
  - PP は非常に強力（#P ⊆ P^{PP}）

  関係図:
  P ⊆ ZPP ⊆ RP ⊆ BPP ⊆ PP ⊆ PSPACE

  実務的な重要性:
  - 多くの実用的アルゴリズムはBPPに属する
  - ランダム性が「見かけの計算能力」を増やす
  - BPP = P と予想されている（derandomization仮説）
```

```python
# 確率的アルゴリズムの例

import random

# RP の例: Miller-Rabin素数判定
def miller_rabin(n, k=20):
    """
    Miller-Rabin素数判定
    RPに属する:
    - n が合成数 → 確率 ≥ 1 - 4^{-k} で「合成数」と答える
    - n が素数 → 確率 1 で「素数」と答える
    """
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False

    # n - 1 = 2^r × d (dは奇数)
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)

        if x == 1 or x == n - 1:
            continue

        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False  # 合成数

    return True  # おそらく素数


# BPP の例: ランダム化最小カット（Kargerのアルゴリズム）
def karger_min_cut(graph, n):
    """
    Kargerのランダム化最小カットアルゴリズム
    BPPに属する

    1回の実行で最小カットを見つける確率: ≥ 2/n²
    n²ln(n) 回繰り返せば高確率で正解
    """
    import copy

    # 辺の収縮
    vertices = list(range(n))
    edges = copy.deepcopy(graph)  # (u, v) のリスト

    # Union-Find
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        parent[px] = py

    remaining = n
    while remaining > 2:
        # ランダムに辺を選んで収縮
        while True:
            u, v = random.choice(edges)
            if find(u) != find(v):
                break

        union(u, v)
        remaining -= 1

    # 残った2つのスーパー頂点間の辺を数える
    cut_size = 0
    for u, v in edges:
        if find(u) != find(v):
            cut_size += 1

    return cut_size


# ZPP の例: ランダム化クイックソート
def randomized_quicksort(arr):
    """
    ランダム化クイックソート
    ZPPに属する: 常に正しい結果、期待実行時間 O(n log n)
    最悪の場合 O(n²) だが確率は非常に低い
    """
    if len(arr) <= 1:
        return arr

    pivot = random.choice(arr)
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return randomized_quicksort(left) + middle + randomized_quicksort(right)
```

### 4.3 量子複雑性クラス

```
量子複雑性クラス:

  BQP（Bounded-Error Quantum Polynomial time）:
  - 量子コンピュータで多項式時間、誤り確率 ≤ 1/3
  - P ⊆ BPP ⊆ BQP ⊆ PSPACE

  BQPで効率的に解ける問題:
  - 素因数分解（ショアのアルゴリズム）
  - 離散対数
  - ユニタリ行列のシミュレーション
  - 特定の代数的問題

  BQPでも効率的に解けないと予想される問題:
  - NP完全問題（SAT, TSP等）
  - ただし証明はされていない

  関係図:
  ┌──────────────────────────────────────────┐
  │                PSPACE                     │
  │  ┌────────────────────────────────────┐  │
  │  │              BQP                    │  │
  │  │  ┌──────────────────────────────┐  │  │
  │  │  │           BPP                 │  │  │
  │  │  │  ┌──────────────────────┐    │  │  │
  │  │  │  │         P             │    │  │  │
  │  │  │  └──────────────────────┘    │  │  │
  │  │  └──────────────────────────────┘  │  │
  │  └────────────────────────────────────┘  │
  │                                          │
  │    NP は BQP と異なる位置にあると予想     │
  │    NP ⊄ BQP かつ BQP ⊄ NP （予想）     │
  └──────────────────────────────────────────┘

  量子超越性（Quantum Supremacy）:
  - 2019年: Googleが53量子ビットで量子超越性を主張
  - 特定のタスクで古典コンピュータを超える実験的実証
  - ただし実用的な問題ではまだ限定的
```

### 4.4 計数問題クラス

```
計数問題クラス:

  #P (Sharp-P):
  - NPの判定版ではなく、解の「数」を数える
  - 例: #SAT = 充足する割り当ての数
  - 例: 完全マッチングの数（パーマネント計算）

  #P完全問題:
  - パーマネント計算（Valiant, 1979）
  - #SAT
  - グラフの彩色数（多項式の評価）

  重要な関係:
  - P ⊆ NP ⊆ P^{#P}
  - Toda の定理: PH ⊆ P^{#P}
    → #P は多項式階層全体を含む（非常に強力）

  近似計数:
  - 完全な計数は困難でも、近似計数は効率的にできることがある
  - FPRAS（完全多項式ランダム近似スキーム）
  - 例: DNF充足割り当て数の近似計数
```

---

## 5. 実務での対処法

### 5.1 NP完全問題への対処戦略

```
NP完全問題に直面した時の意思決定フロー:

  ┌──────────────────────────────┐
  │ 問題がNP完全/NP困難と判明    │
  └──────────┬───────────────────┘
             ↓
  ┌──────────────────────────────┐
  │ 入力サイズは小さいか？        │
  │ (n ≤ 20~25 程度)            │
  └────┬──────────┬──────────────┘
       │ YES      │ NO
       ↓          ↓
  ┌─────────┐  ┌──────────────────────────┐
  │ 厳密解  │  │ 特殊な構造はあるか？      │
  │ を求める │  └────┬──────────┬───────────┘
  └─────────┘       │ YES      │ NO
                     ↓          ↓
              ┌───────────┐  ┌──────────────────────┐
              │ 特殊ケース│  │ 近似保証は必要か？     │
              │ の効率的  │  └────┬──────────┬────────┘
              │ アルゴリズム│      │ YES      │ NO
              └───────────┘      ↓          ↓
                          ┌───────────┐  ┌──────────┐
                          │ 近似アルゴ │  │ ヒューリ  │
                          │ リズム     │  │ スティック │
                          └───────────┘  └──────────┘
```

### 5.2 厳密解（小規模の場合）

```python
# 1. ビット全探索（n ≤ 20）
def knapsack_brute_force(weights, values, capacity):
    """
    ナップサック問題のビット全探索
    O(2^n): n ≤ 20 程度まで実用的
    """
    n = len(weights)
    best_value = 0
    best_items = 0

    for mask in range(1 << n):
        total_weight = 0
        total_value = 0
        for i in range(n):
            if mask & (1 << i):
                total_weight += weights[i]
                total_value += values[i]

        if total_weight <= capacity and total_value > best_value:
            best_value = total_value
            best_items = mask

    return best_value, best_items


# 2. 動的計画法（指数時間だが高速化）
def tsp_dp(dist, n):
    """
    巡回セールスマン問題のDP解法（Held-Karp）
    O(n² × 2^n): ビット全探索の O(n! × n) より大幅に高速
    n ≤ 25 程度まで実用的
    """
    INF = float('inf')
    # dp[S][i] = 集合Sの都市を訪問し、最後にiにいる場合の最小コスト
    dp = [[INF] * n for _ in range(1 << n)]
    dp[1][0] = 0  # 都市0から開始

    for S in range(1 << n):
        for u in range(n):
            if dp[S][u] == INF:
                continue
            if not (S & (1 << u)):
                continue
            for v in range(n):
                if S & (1 << v):
                    continue
                new_S = S | (1 << v)
                dp[new_S][v] = min(dp[new_S][v], dp[S][u] + dist[u][v])

    # 全都市を訪問して0に戻る
    full = (1 << n) - 1
    result = min(dp[full][i] + dist[i][0] for i in range(1, n))

    return result


# 3. 分岐限定法（Branch and Bound）
def branch_and_bound_tsp(dist, n):
    """
    TSPの分岐限定法
    最悪 O(n!), 平均的にはDPより速いことが多い
    枝刈りの効果で大幅に探索を削減
    """
    import heapq

    INF = float('inf')
    best = INF

    def lower_bound(visited, current, cost):
        """下界の計算（簡易版: 最小辺の和）"""
        lb = cost
        for i in range(n):
            if i not in visited:
                min_edge = min(dist[i][j] for j in range(n) if j != i)
                lb += min_edge
        return lb

    # 優先度付きキュー: (下界, コスト, 現在地, 訪問済み)
    pq = [(lower_bound({0}, 0, 0), 0, 0, frozenset({0}))]

    while pq:
        lb, cost, current, visited = heapq.heappop(pq)

        if lb >= best:
            continue  # 枝刈り

        if len(visited) == n:
            total = cost + dist[current][0]
            best = min(best, total)
            continue

        for next_city in range(n):
            if next_city in visited:
                continue
            new_cost = cost + dist[current][next_city]
            new_visited = visited | {next_city}
            lb = lower_bound(new_visited, next_city, new_cost)
            if lb < best:
                heapq.heappush(pq, (lb, new_cost, next_city, new_visited))

    return best
```

### 5.3 近似アルゴリズム

```python
# 近似アルゴリズムの実装例

# 1. 頂点被覆の2-近似
def vertex_cover_2approx(graph):
    """
    頂点被覆の2-近似アルゴリズム

    近似比: 2（最適解の2倍以内を保証）
    時間計算量: O(V + E)

    アルゴリズム:
    1. 辺を1つ選ぶ
    2. その両端点を被覆に追加
    3. 追加した頂点に接続する辺を全て削除
    4. 辺がなくなるまで繰り返す
    """
    cover = set()
    edges = set()

    for u in graph:
        for v in graph[u]:
            if u < v:
                edges.add((u, v))

    remaining_edges = set(edges)

    while remaining_edges:
        # 辺を1つ選ぶ
        u, v = next(iter(remaining_edges))
        cover.add(u)
        cover.add(v)

        # 両端点に接続する辺を削除
        remaining_edges = {
            (a, b) for (a, b) in remaining_edges
            if a != u and a != v and b != u and b != v
        }

    return cover

# 証明: 2-近似
# - 選ばれた辺の集合をMとする（|M|本）
# - Mの辺は互いに端点を共有しない（マッチング）
# - cover = 2|M| 頂点
# - 最適解は各辺から少なくとも1つの端点を含む → OPT ≥ |M|
# - cover = 2|M| ≤ 2 × OPT ∎


# 2. 集合被覆の貪欲近似（O(log n)-近似）
def greedy_set_cover(universe, sets):
    """
    集合被覆の貪欲アルゴリズム

    近似比: H(n) = Σ_{i=1}^{n} 1/i ≈ ln(n)
    → これが最善（P ≠ NP なら (1-ε)ln(n) 未満は不可能）

    時間計算量: O(|universe| × |sets|)
    """
    uncovered = set(universe)
    selected = []
    remaining_sets = list(sets.items())

    while uncovered:
        # 最も多くの未被覆要素をカバーする集合を選ぶ
        best_set = max(
            remaining_sets,
            key=lambda s: len(s[1] & uncovered)
        )

        if len(best_set[1] & uncovered) == 0:
            break  # これ以上カバーできない

        selected.append(best_set[0])
        uncovered -= best_set[1]
        remaining_sets.remove(best_set)

    return selected


# 3. TSPのChristofides近似（3/2-近似）
def christofides_tsp(dist, n):
    """
    Christofides のアルゴリズム（概念的実装）

    三角不等式を満たすTSPに対して 3/2-近似を保証
    これは多項式時間で達成される最良の近似比

    手順:
    1. 最小全域木Tを求める
    2. Tの奇数次頂点の最小重み完全マッチングMを求める
    3. T ∪ M でオイラー閉路を求める
    4. ショートカットしてハミルトン閉路にする
    """
    # 1. 最小全域木
    mst = compute_mst(dist, n)

    # 2. 奇数次頂点の最小完全マッチング
    odd_vertices = [v for v in range(n) if degree(mst, v) % 2 == 1]
    matching = min_weight_perfect_matching(dist, odd_vertices)

    # 3. MST + マッチングでオイラーグラフ
    euler_graph = combine(mst, matching)
    euler_tour = find_euler_tour(euler_graph)

    # 4. ショートカット（重複頂点を飛ばす）
    visited = set()
    hamiltonian = []
    for v in euler_tour:
        if v not in visited:
            visited.add(v)
            hamiltonian.append(v)

    return hamiltonian

# 近似比の証明:
# MST ≤ OPT（最適TSPから1辺を除くとMSTの上界）
# マッチング ≤ OPT/2（奇数次頂点上の最適TSPの半分以下）
# よって: Christofides ≤ MST + Matching ≤ OPT + OPT/2 = 3/2 × OPT


# 4. ナップサック問題のFPTAS（完全多項式時間近似スキーム）
def knapsack_fptas(weights, values, capacity, epsilon):
    """
    ナップサック問題のFPTAS

    任意のε > 0 に対して、(1-ε)-近似を保証
    時間計算量: O(n² / ε)

    FPTASが存在 → 強いNP困難ではない
    """
    n = len(weights)
    v_max = max(values)

    # スケーリング
    K = epsilon * v_max / n
    scaled_values = [int(v / K) for v in values]
    V_total = sum(scaled_values)

    # スケールされた値でDP
    INF = float('inf')
    # dp[v] = 価値合計がちょうどvとなる最小重量
    dp = [INF] * (V_total + 1)
    dp[0] = 0

    for i in range(n):
        for v in range(V_total, scaled_values[i] - 1, -1):
            if dp[v - scaled_values[i]] + weights[i] < dp[v]:
                dp[v] = dp[v - scaled_values[i]] + weights[i]

    # 容量以内で最大の価値を見つける
    best_v = 0
    for v in range(V_total + 1):
        if dp[v] <= capacity:
            best_v = v

    return best_v * K  # 元のスケールに戻す
```

### 5.4 近似困難性

```
近似困難性（Inapproximability）:

  PCP定理（Probabilistically Checkable Proofs, 1992）:
  NP = PCP(log n, 1)
  → NP問題の証明は、定数個のビットをランダムにチェックするだけで
    高確率で正しさを検証できる

  PCP定理の帰結（近似困難性）:

  ┌────────────────────────────────────────────────────────┐
  │ 問題              │ 近似可能          │ 近似困難          │
  ├───────────────────┼──────────────────┼──────────────────┤
  │ 頂点被覆          │ 2-近似（達成）    │ 2-ε は NP困難     │
  │                   │                  │ (UGC仮定)         │
  │ 集合被覆          │ ln(n)-近似       │ (1-ε)ln(n) は     │
  │                   │ （達成）         │ NP困難            │
  │ MAX-3SAT          │ 7/8-近似（達成） │ 7/8+ε は NP困難   │
  │ MAX-CLIQUE        │ n^{1-ε}-近似     │ n^{1-ε} 未満は    │
  │                   │                  │ NP困難            │
  │ TSP（一般）       │ 近似不可能       │ 定数近似はNP困難   │
  │ TSP（三角不等式） │ 3/2-近似（達成） │ 220/219 は NP困難  │
  │ ナップサック       │ FPTAS あり       │ ─                 │
  └────────────────────────────────────────────────────────┘

  意味:
  - MAX-3SAT: 7/8を超える近似は（P ≠ NP なら）不可能
    → ランダムに割り当てても 7/8 は達成できる
    → これ以上は本質的に不可能
  - MAX-CLIQUE: n^{1-ε} 倍以内の近似すら不可能
    → 極めて近似困難
```

### 5.5 ヒューリスティック

```python
# ヒューリスティックの実装例

import random
import math

# 1. 焼きなまし法（Simulated Annealing）
def simulated_annealing_tsp(dist, n, initial_temp=10000,
                             cooling_rate=0.9995, min_temp=1e-8):
    """
    TSPの焼きなまし法

    保証: なし（ただし実用上は非常に良い解が得られる）
    実行時間: ユーザーが指定（反復回数 or 温度で制御）
    """
    # 初期解: ランダムな順列
    current = list(range(n))
    random.shuffle(current)

    def tour_cost(tour):
        return sum(dist[tour[i]][tour[(i+1) % n]] for i in range(n))

    current_cost = tour_cost(current)
    best = current[:]
    best_cost = current_cost
    temp = initial_temp

    while temp > min_temp:
        # 近傍: 2-opt（2辺を交換）
        i = random.randint(0, n - 2)
        j = random.randint(i + 1, n - 1)
        neighbor = current[:i] + current[i:j+1][::-1] + current[j+1:]
        neighbor_cost = tour_cost(neighbor)

        # 受理判定
        delta = neighbor_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current = neighbor
            current_cost = neighbor_cost

            if current_cost < best_cost:
                best = current[:]
                best_cost = current_cost

        temp *= cooling_rate

    return best, best_cost


# 2. 遺伝的アルゴリズム（Genetic Algorithm）
def genetic_algorithm_tsp(dist, n, pop_size=100, generations=1000,
                          mutation_rate=0.02):
    """
    TSPの遺伝的アルゴリズム

    構成要素:
    - 個体: 都市の順列（染色体）
    - 適応度: 経路長の逆数
    - 交叉: 順序交叉（OX）
    - 突然変異: 2-opt
    - 選択: トーナメント選択
    """
    def tour_cost(tour):
        return sum(dist[tour[i]][tour[(i+1) % n]] for i in range(n))

    def tournament_select(population, fitnesses, tournament_size=5):
        indices = random.sample(range(len(population)), tournament_size)
        best_idx = max(indices, key=lambda i: fitnesses[i])
        return population[best_idx]

    def order_crossover(parent1, parent2):
        """順序交叉（OX）"""
        start = random.randint(0, n - 2)
        end = random.randint(start + 1, n - 1)

        child = [-1] * n
        child[start:end+1] = parent1[start:end+1]

        fill_pos = (end + 1) % n
        parent2_pos = (end + 1) % n
        while -1 in child:
            if parent2[parent2_pos] not in child:
                child[fill_pos] = parent2[parent2_pos]
                fill_pos = (fill_pos + 1) % n
            parent2_pos = (parent2_pos + 1) % n

        return child

    def mutate(tour):
        """2-opt突然変異"""
        if random.random() < mutation_rate:
            i = random.randint(0, n - 2)
            j = random.randint(i + 1, n - 1)
            tour[i:j+1] = reversed(tour[i:j+1])
        return tour

    # 初期集団
    population = [random.sample(range(n), n) for _ in range(pop_size)]

    best_ever = None
    best_cost_ever = float('inf')

    for gen in range(generations):
        # 適応度計算
        costs = [tour_cost(ind) for ind in population]
        fitnesses = [1.0 / c for c in costs]

        # 最良個体の更新
        best_idx = min(range(pop_size), key=lambda i: costs[i])
        if costs[best_idx] < best_cost_ever:
            best_cost_ever = costs[best_idx]
            best_ever = population[best_idx][:]

        # 次世代の生成
        new_population = [best_ever[:]]  # エリート保存

        while len(new_population) < pop_size:
            p1 = tournament_select(population, fitnesses)
            p2 = tournament_select(population, fitnesses)
            child = order_crossover(p1, p2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

    return best_ever, best_cost_ever


# 3. タブー探索
def tabu_search_tsp(dist, n, max_iter=10000, tabu_size=20):
    """
    TSPのタブー探索

    直近のtabu_size回の移動を禁止（タブーリスト）
    → 局所最適解から脱出可能
    """
    current = list(range(n))
    random.shuffle(current)

    def tour_cost(tour):
        return sum(dist[tour[i]][tour[(i+1) % n]] for i in range(n))

    current_cost = tour_cost(current)
    best = current[:]
    best_cost = current_cost
    tabu_list = []

    for _ in range(max_iter):
        # 全ての2-opt近傍を列挙
        best_neighbor = None
        best_neighbor_cost = float('inf')
        best_move = None

        for i in range(n - 1):
            for j in range(i + 1, n):
                move = (i, j)
                neighbor = current[:i] + current[i:j+1][::-1] + current[j+1:]
                nc = tour_cost(neighbor)

                # タブーでない、または最良解を更新する場合
                if (move not in tabu_list or nc < best_cost):
                    if nc < best_neighbor_cost:
                        best_neighbor = neighbor
                        best_neighbor_cost = nc
                        best_move = move

        if best_neighbor is None:
            break

        current = best_neighbor
        current_cost = best_neighbor_cost

        # タブーリスト更新
        tabu_list.append(best_move)
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

        if current_cost < best_cost:
            best = current[:]
            best_cost = current_cost

    return best, best_cost
```

### 5.6 パラメータ化計算量（FPT）

```
パラメータ化計算量（Fixed-Parameter Tractability）:

  アイデア: 入力サイズ n に加えて、パラメータ k を考慮する

  FPT: O(f(k) × n^c) で解ける問題
  - f(k): kにのみ依存する（指数的でも可）
  - n^c: 入力サイズの多項式
  - kが小さければ実用的に解ける

  例: 頂点被覆（パラメータ k = 被覆のサイズ）
  - 力任せ: O(n^k) — kが大きいと非実用的
  - FPTアルゴリズム: O(2^k × n) — kが小さければ高速

  FPTの階層:
  ┌─────────────────────────────────────────────┐
  │ FPT ⊆ W[1] ⊆ W[2] ⊆ ... ⊆ XP             │
  │                                             │
  │ FPT: f(k) × n^c                            │
  │ XP: n^{f(k)}（パラメータが指数に入る）       │
  │                                             │
  │ FPT ≠ W[1] と予想                          │
  │ → パラメータ化版のP ≠ NP                    │
  └─────────────────────────────────────────────┘

  W[1]完全問題の例:
  - k-クリーク
  - 独立集合（パラメータ k）

  FPTの例:
  - 頂点被覆（パラメータ k）: O(1.2738^k + kn)
  - k-パス問題: O(2^k × n)（Color Coding）
  - 木幅が k のグラフ上のSAT: O(2^k × n)
```

```python
# FPTアルゴリズムの例

# 頂点被覆のFPTアルゴリズム（分岐限定法）
def vertex_cover_fpt(graph, k):
    """
    頂点被覆のFPTアルゴリズム
    時間計算量: O(2^k × (V + E))

    アイデア:
    辺(u,v)を1つ選ぶ → uを被覆に入れるか、vを入れるかの2択
    → 深さkの二分木を探索
    """
    def solve(edges, cover, remaining_k):
        # 辺がなくなったら成功
        if not edges:
            return cover

        # 予算がなくなったら失敗
        if remaining_k == 0:
            return None

        # 辺を1つ選ぶ
        u, v = next(iter(edges))

        # 選択1: uを被覆に入れる
        new_edges_u = {(a, b) for (a, b) in edges if a != u and b != u}
        result = solve(new_edges_u, cover | {u}, remaining_k - 1)
        if result is not None:
            return result

        # 選択2: vを被覆に入れる
        new_edges_v = {(a, b) for (a, b) in edges if a != v and b != v}
        result = solve(new_edges_v, cover | {v}, remaining_k - 1)
        if result is not None:
            return result

        return None  # サイズk以下の頂点被覆は存在しない

    # 辺の集合を構築
    edges = set()
    for u in graph:
        for v in graph[u]:
            if u < v:
                edges.add((u, v))

    return solve(edges, set(), k)


# Color Coding: k-パス問題のFPTアルゴリズム
def color_coding_k_path(graph, n, k):
    """
    k-パス問題: グラフに長さkのパスが存在するか？
    FPTアルゴリズム: O(2^k × E) （ランダム化）

    アイデア:
    1. 各頂点にランダムに k 色を割り当て
    2. 全ての色が使われるカラフルなパスをDPで見つける
    3. 成功確率 ≥ e^{-k} → O(e^k) 回繰り返せば高確率で見つかる
    """
    for trial in range(int(2.72 ** k) * 2):  # e^k × 2 回試行
        # ランダムな彩色
        colors = [random.randint(0, k - 1) for _ in range(n)]

        # DP: dp[v][S] = 頂点vで終わり、色集合Sを使うカラフルなパスが存在するか
        dp = [[False] * (1 << k) for _ in range(n)]

        for v in range(n):
            dp[v][1 << colors[v]] = True

        for S in range(1, 1 << k):
            for v in range(n):
                if not dp[v][S]:
                    continue
                for u in graph.get(v, []):
                    c = colors[u]
                    if not (S & (1 << c)):  # 新しい色
                        dp[u][S | (1 << c)] = True

        # 全色を使うパスが見つかったか？
        full_set = (1 << k) - 1
        for v in range(n):
            if dp[v][full_set]:
                return True

    return False
```

---

## 6. 重要な定理

### 6.1 時間階層定理と空間階層定理

```
時間階層定理（Time Hierarchy Theorem）:
  f(n) × log(f(n)) = o(g(n)) ならば
  DTIME(f(n)) ⊊ DTIME(g(n))

  帰結:
  - P ⊊ EXPTIME（P ≠ EXPTIME）
  - 時間が十分増えれば、より多くの問題が解ける

空間階層定理（Space Hierarchy Theorem）:
  f(n) = o(g(n)) ならば
  SPACE(f(n)) ⊊ SPACE(g(n))

  帰結:
  - L ⊊ PSPACE
  - 空間が増えれば、より多くの問題が解ける

注意:
  - P ≠ NP は時間階層定理からは導けない
  - 階層定理は「同じ計算モデル」内でのみ有効
  - 決定性 vs 非決定性の比較には使えない
```

### 6.2 Savitchの定理

```
Savitchの定理:
  NSPACE(f(n)) ⊆ SPACE(f(n)²)

  帰結:
  - NPSPACE = PSPACE
  - 非決定性は空間を二乗にしか増やさない
  - 対照的に、非決定性が時間に与える影響は未解明（P vs NP）

  証明のアイデア:
  NTM の受理計算を、中間地点を全て試すことで
  空間 O(f(n)²) の決定性TMでシミュレート

  実務的意味:
  - ゲームの最適戦略（PSPACE完全）は、
    非決定性を使わなくても多項式空間で計算可能
  - QBF（量化ブール式）も多項式空間で解ける
```

---

## 7. 実践演習

### 演習1: NP完全の帰着（基礎）

```
問題: 独立集合問題が頂点被覆問題に帰着できることを示せ。

定義:
- 独立集合: どの2頂点も辺で結ばれていない頂点の部分集合
- 頂点被覆: 全ての辺の少なくとも一端を含む頂点の部分集合

証明:
  Sが独立集合 ⟺ V \ S が頂点被覆

  (→) Sが独立集合とする。
  任意の辺 (u,v) について、u,v の少なくとも一方は S に属さない
  （両方属していたら独立でない）。
  よって少なくとも一方は V \ S に属する → V \ S は頂点被覆。

  (←) V \ S が頂点被覆とする。
  S の任意の2頂点 u,v について、(u,v) は辺でない
  （辺であれば u,v ∈ S なので V \ S に含まれず被覆にならない）。
  よって S は独立集合。

  帰着: サイズkの独立集合が存在 ⟺ サイズ(n-k)の頂点被覆が存在

  この帰着は O(1) 時間 → 多項式時間帰着 ∎
```

### 演習2: 近似アルゴリズム（応用）

```python
"""
演習: 貪欲法による集合被覆の近似アルゴリズムを実装し、近似比を実測せよ。
"""

import random

def generate_set_cover_instance(n, m, density=0.3):
    """集合被覆のランダムインスタンス生成"""
    universe = set(range(n))
    sets = {}
    for i in range(m):
        size = max(1, int(n * density * random.random()))
        sets[i] = set(random.sample(list(universe), min(size, n)))

    # 全要素が被覆可能であることを保証
    for elem in universe:
        random_set = random.choice(list(sets.keys()))
        sets[random_set].add(elem)

    return universe, sets


def optimal_set_cover(universe, sets):
    """厳密解（小規模用、指数時間）"""
    n = len(sets)
    keys = list(sets.keys())
    best = None

    for mask in range(1, 1 << n):
        covered = set()
        selected = []
        for i in range(n):
            if mask & (1 << i):
                covered |= sets[keys[i]]
                selected.append(keys[i])
        if covered >= universe:
            if best is None or len(selected) < len(best):
                best = selected

    return best


# 実験
for n in [10, 15, 20]:
    total_ratio = 0
    trials = 50
    for _ in range(trials):
        universe, sets = generate_set_cover_instance(n, n * 2)
        greedy = greedy_set_cover(universe, sets)
        optimal = optimal_set_cover(universe, sets)
        ratio = len(greedy) / len(optimal)
        total_ratio += ratio

    avg_ratio = total_ratio / trials
    print(f"n={n}: 平均近似比 = {avg_ratio:.3f} (理論上界: {math.log(n):.3f})")
```

### 演習3: SAT ソルバー（実装）

```python
"""
演習: DPLLアルゴリズム（SATソルバーの基礎）を実装せよ。
"""

def dpll(clauses, assignment=None):
    """
    DPLLアルゴリズム — SAT問題の標準的な解法

    最悪の場合 O(2^n) だが、実用上は非常に効率的
    現代のSATソルバー（MiniSat, Z3等）の基礎
    """
    if assignment is None:
        assignment = {}

    # 充足チェック
    clauses = simplify(clauses, assignment)

    # 空の節がある → 充足不可能
    if any(len(c) == 0 for c in clauses):
        return None

    # 全ての節が除去された → 充足可能
    if len(clauses) == 0:
        return assignment

    # ユニット伝播（Unit Propagation）
    unit_clauses = [c for c in clauses if len(c) == 1]
    while unit_clauses:
        literal = unit_clauses[0][0]
        var = abs(literal)
        value = literal > 0
        assignment[var] = value
        clauses = simplify(clauses, {var: value})

        if any(len(c) == 0 for c in clauses):
            return None
        if len(clauses) == 0:
            return assignment

        unit_clauses = [c for c in clauses if len(c) == 1]

    # 純リテラル除去（Pure Literal Elimination）
    all_literals = set()
    for clause in clauses:
        for lit in clause:
            all_literals.add(lit)

    for lit in list(all_literals):
        if -lit not in all_literals:
            var = abs(lit)
            assignment[var] = (lit > 0)
            clauses = simplify(clauses, {var: lit > 0})

    if len(clauses) == 0:
        return assignment

    # 分岐（Branching）
    var = abs(clauses[0][0])

    # True を試す
    result = dpll(clauses, {**assignment, var: True})
    if result is not None:
        return result

    # False を試す
    result = dpll(clauses, {**assignment, var: False})
    return result


def simplify(clauses, assignment):
    """割り当てに基づいて節を簡約化"""
    new_clauses = []
    for clause in clauses:
        new_clause = []
        satisfied = False
        for lit in clause:
            var = abs(lit)
            if var in assignment:
                if (lit > 0) == assignment[var]:
                    satisfied = True
                    break
                # リテラルがFalseの場合は節から除去
            else:
                new_clause.append(lit)
        if not satisfied:
            new_clauses.append(new_clause)
    return new_clauses
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| P | 多項式時間で解ける。ソート、最短経路、2-SAT等 |
| NP | 多項式時間で検証できる。P ⊆ NP |
| NP完全 | NPで最も難しい。1つ解ければ全NP問題が解ける |
| NP困難 | NP完全以上に難しい。NPに属さない場合もある |
| P ≠ NP? | CS最大の未解決問題。暗号の安全性に直結 |
| PSPACE | 多項式空間。QBFが完全問題。PSPACE = NPSPACE |
| BPP/BQP | 確率的/量子計算のクラス。BPP = Pが予想される |
| 近似アルゴリズム | 最適解の定数倍を保証。PCP定理で限界も判明 |
| FPT | パラメータkが小さい場合に効率的。f(k)×n^c |
| ヒューリスティック | 保証なしだが実用的。SA, GA, タブー探索等 |

---

## 次に読むべきガイド
→ [[03-information-theory.md]] — 情報理論

---

## 参考文献
1. Sipser, M. "Introduction to the Theory of Computation." Chapters 7-8.
2. Arora, S. & Barak, B. "Computational Complexity: A Modern Approach." Cambridge, 2009.
3. Cook, S. A. "The Complexity of Theorem-Proving Procedures." STOC, 1971.
4. Karp, R. M. "Reducibility Among Combinatorial Problems." 1972.
5. Garey, M. R. & Johnson, D. S. "Computers and Intractability." W. H. Freeman, 1979.
6. Vazirani, V. V. "Approximation Algorithms." Springer, 2001.
7. Downey, R. G. & Fellows, M. R. "Parameterized Complexity." Springer, 1999.
8. Arora, S., Lund, C., Motwani, R., Sudan, M., & Szegedy, M. "Proof Verification and the Hardness of Approximation Problems." JACM, 1998.
9. Williamson, D. P. & Shmoys, D. B. "The Design of Approximation Algorithms." Cambridge, 2011.
10. Christofides, N. "Worst-Case Analysis of a New Heuristic for the Travelling Salesman Problem." 1976.
