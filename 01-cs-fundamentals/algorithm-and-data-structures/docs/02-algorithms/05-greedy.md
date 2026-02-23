# 貪欲法（Greedy Algorithm）

> 各ステップで局所的に最適な選択を繰り返すことで、全体の最適解を効率的に求める設計手法を理解する

## この章で学ぶこと

1. **貪欲法の適用条件**（貪欲選択性質・最適部分構造）を見抜き、正当性を検証できる
2. **活動選択問題・ハフマン符号・最小全域木**を貪欲法で正しく解ける
3. **貪欲法と DP の使い分け**を判断でき、貪欲法が使えない場合を識別できる
4. **マトロイド理論**の基礎を理解し、貪欲法の正当性を体系的に判断できる

---

## 1. 貪欲法の原理

```
┌──────────────────────────────────────────────┐
│             貪欲法の2条件                      │
├──────────────────────────────────────────────┤
│                                               │
│  1. 貪欲選択性質 (Greedy Choice Property)      │
│     → 局所最適な選択が全体最適につながる        │
│                                               │
│  2. 最適部分構造 (Optimal Substructure)         │
│     → 部分問題の最適解から全体の最適解が得られる │
│                                               │
├──────────────────────────────────────────────┤
│                                               │
│  DP との違い:                                  │
│  DP  → 全ての選択肢を試して最適を選ぶ          │
│  貪欲 → 一度の選択で即座に決定（後戻りなし）    │
│                                               │
│  貪欲は DP より高速だが、適用範囲が狭い         │
└──────────────────────────────────────────────┘
```

### 貪欲法の設計手順

```
1. 問題を「選択の繰り返し」として定式化する
2. 各ステップの貪欲な選択基準を定める
3. 貪欲選択性質を証明（交換論法 or マトロイド）
4. 最適部分構造を確認する
5. 実装する

注意: Step 3 を省略すると、直感で間違える危険がある
```

### 貪欲法が適用できるかの判断フロー

```
問題を見たとき:

  最適化問題か?
    ├─ NO  → 貪欲法の対象外
    └─ YES → 局所最適 = 全体最適が成り立つか?
              ├─ YES → 貪欲法で解ける（証明は必要）
              │         ├─ 交換論法で証明可能? → 実装
              │         └─ マトロイド構造を持つ? → 実装
              └─ NO or 不明 → DP を検討
                    └─ 反例が見つかったら DP 確定
```

---

## 2. 活動選択問題（Activity Selection）

終了時間が最も早い活動を優先的に選び、できるだけ多くの活動をスケジュールする。

```
活動:  開始  終了
 a1:   1 --- 4
 a2:     3 ----- 5
 a3:  0 ---- 6
 a4:       5 --- 7
 a5:         3 ----- 9
 a6:            5 --------- 9
 a7:              6 --- 8
 a8:                  8 --- 11
 a9:                    8 ----- 12
 a10:                      2 ---------- 14
 a11:                              12 --- 16

タイムライン:
0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16
|--a1--|     |--a4--|  |--a7--|   |--a8--|   |--a11--|
               ← 終了時間順に貪欲選択 → 最大4活動
```

### 正当性の証明（交換論法）

```
定理: 終了時間が最も早い活動を選ぶ貪欲法は最適である。

証明（交換論法）:
  OPT を最適解、G を貪欲解とする。
  OPT の最初の活動が G の最初の活動 a₁ と異なるとする。

  OPT の最初の活動を a₁ に交換すると:
  - a₁ は全活動中で終了時間が最も早い
  - よって a₁ は OPT の最初の活動より早く終わるか同時に終わる
  - OPT の2番目以降の活動と矛盾しない
  - 交換後も有効な解で、活動数は同じ

  これを繰り返すと、OPT を G に変換できる。
  よって |G| = |OPT|、すなわち貪欲解は最適。 □
```

```python
def activity_selection(activities: list) -> list:
    """活動選択問題 - O(n log n)
    activities: [(start, end), ...]
    """
    # 終了時間でソート
    sorted_acts = sorted(activities, key=lambda x: x[1])
    selected = [sorted_acts[0]]
    last_end = sorted_acts[0][1]

    for start, end in sorted_acts[1:]:
        if start >= last_end:  # 前の活動と重ならない
            selected.append((start, end))
            last_end = end

    return selected

activities = [(1,4), (3,5), (0,6), (5,7), (3,9), (5,9),
              (6,8), (8,11), (8,12), (2,14), (12,16)]
result = activity_selection(activities)
print(f"選択された活動: {result}")
# [(1, 4), (5, 7), (8, 11), (12, 16)]
print(f"活動数: {len(result)}")  # 4
```

### 重み付き活動選択問題

活動に重み（利益）がある場合は、貪欲法では解けない。DP が必要。

```python
import bisect

def weighted_activity_selection(activities: list) -> int:
    """重み付き活動選択問題 - O(n log n)
    activities: [(start, end, weight), ...]
    貪欲では解けないため DP を使う
    """
    # 終了時間でソート
    activities.sort(key=lambda x: x[1])
    n = len(activities)

    # 各活動 i に対して、i と重ならない直前の活動を二分探索で求める
    ends = [a[1] for a in activities]

    def latest_non_conflict(i):
        target = activities[i][0]
        idx = bisect.bisect_right(ends, target) - 1
        return idx if idx < i else idx

    # DP
    dp = [0] * (n + 1)
    for i in range(1, n + 1):
        # 活動 i-1 を含まない
        dp[i] = dp[i - 1]
        # 活動 i-1 を含む
        j = latest_non_conflict(i - 1)
        dp[i] = max(dp[i], dp[j + 1] + activities[i - 1][2])

    return dp[n]

activities_w = [(1, 4, 5), (3, 5, 6), (0, 6, 8), (5, 7, 4), (6, 9, 2)]
print(weighted_activity_selection(activities_w))  # 13 (活動(0,6,8) + (6,9,2) ?)
```

---

## 3. ハフマン符号（Huffman Coding）

出現頻度の低い文字に長い符号、高い文字に短い符号を割り当て、全体のビット数を最小化する。

```
文字と頻度:
  a:45  b:13  c:12  d:16  e:9  f:5

ハフマン木の構築:
Step1: f(5) + e(9) = 14
Step2: c(12) + b(13) = 25
Step3: 14 + d(16) = 30
Step4: 25 + 30 = 55
Step5: a(45) + 55 = 100

         (100)
        /      \
     a(45)    (55)
             /     \
          (25)     (30)
         /    \    /    \
       c(12) b(13) (14) d(16)
                  /    \
                f(5)  e(9)

符号割当:
  a: 0       (1ビット)
  c: 100     (3ビット)
  b: 101     (3ビット)
  f: 1100    (4ビット)
  e: 1101    (4ビット)
  d: 111     (3ビット)
```

### ハフマン符号の最適性

```
定理: ハフマン符号は最適接頭辞符号である。

接頭辞符号: どの符号語も他の符号語の接頭辞でない
  → 曖昧さなくデコード可能

最適性の直感:
  - 出現頻度が最も低い2つの文字は、最適木で最も深い兄弟ノードに配置される
  - この2文字を統合しても、問題の構造は変わらない（最適部分構造）
  - 毎回頻度が最小の2つを統合する（貪欲選択）
```

```python
import heapq
from collections import Counter

class HuffmanNode:
    def __init__(self, char=None, freq=0, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(freq: dict) -> HuffmanNode:
    """ハフマン木を構築 - O(n log n)"""
    heap = [HuffmanNode(char=c, freq=f) for c, f in freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(freq=left.freq + right.freq,
                             left=left, right=right)
        heapq.heappush(heap, merged)

    return heap[0]

def build_codes(root: HuffmanNode, prefix="", codes=None) -> dict:
    """ハフマン符号を生成"""
    if codes is None:
        codes = {}

    if root.char is not None:
        codes[root.char] = prefix or "0"
        return codes

    if root.left:
        build_codes(root.left, prefix + "0", codes)
    if root.right:
        build_codes(root.right, prefix + "1", codes)

    return codes

def huffman_encode(text: str) -> tuple:
    """ハフマン符号化"""
    freq = Counter(text)
    tree = build_huffman_tree(freq)
    codes = build_codes(tree)
    encoded = ''.join(codes[c] for c in text)
    return encoded, codes, tree

def huffman_decode(encoded: str, tree: HuffmanNode) -> str:
    """ハフマン復号"""
    result = []
    node = tree
    for bit in encoded:
        if bit == '0':
            node = node.left
        else:
            node = node.right
        if node.char is not None:
            result.append(node.char)
            node = tree
    return ''.join(result)

# 使用例
text = "aaaaabbbccddddeefffff"
encoded, codes, tree = huffman_encode(text)
print("符号表:", codes)
print(f"元のサイズ: {len(text) * 8} ビット")
print(f"圧縮後: {len(encoded)} ビット")
print(f"圧縮率: {len(encoded) / (len(text) * 8):.1%}")

# 復号して検証
decoded = huffman_decode(encoded, tree)
print(f"復号結果一致: {decoded == text}")  # True
```

### 適応型ハフマン符号

```python
# 実務では「静的ハフマン」よりも「適応型ハフマン」が使われることが多い
#
# 静的ハフマン:
#   - 全文を2パスで処理（1パス目で頻度計算、2パス目で符号化）
#   - 符号表をデータと一緒に保存する必要がある
#
# 適応型ハフマン (Adaptive Huffman):
#   - 1パスで処理（文字を読みながら木を更新）
#   - 符号表の伝送が不要
#   - gzip、DEFLATE アルゴリズム等で使用
#
# 実用的な圧縮ライブラリ:
#   - zlib: DEFLATE（LZ77 + ハフマン）
#   - brotli: LZ77 + ハフマン + コンテキストモデリング
#   - zstd: LZ77 + FSE（有限状態エントロピー）
```

---

## 4. 最小全域木

### 4.1 Kruskal のアルゴリズム

辺を重みの昇順に調べ、サイクルを作らない辺を追加していく。

```
グラフ:
    A ---4--- B
    |       / |
    8     2   6
    |   /     |
    C ---3--- D
      \     /
       7   9
        \ /
         E

辺の重み順: (B,C,2) → (C,D,3) → (A,B,4) → (B,D,6) → (C,E,7) → (A,C,8) → (D,E,9)

Step1: B-C (2) 追加  ← サイクルなし
Step2: C-D (3) 追加  ← サイクルなし
Step3: A-B (4) 追加  ← サイクルなし
Step4: B-D (6) スキップ ← B-C-D でサイクル!
Step5: C-E (7) 追加  ← サイクルなし → V-1=4辺 → 完了

MST: B-C(2), C-D(3), A-B(4), C-E(7) = 合計 16
```

```python
class UnionFind:
    """Union-Find（Kruskal用）"""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n

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
        self.size[px] += self.size[py]
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

    def connected(self, x, y):
        return self.find(x) == self.find(y)

def kruskal(n: int, edges: list) -> tuple:
    """Kruskal法 - O(E log E)
    edges: [(weight, u, v), ...]
    返り値: (MST辺リスト, 合計重み)
    """
    edges.sort()  # 重みでソート
    uf = UnionFind(n)
    mst = []
    total = 0

    for w, u, v in edges:
        if uf.union(u, v):
            mst.append((u, v, w))
            total += w
            if len(mst) == n - 1:
                break

    return mst, total

# 頂点: 0=A, 1=B, 2=C, 3=D, 4=E
edges = [(4,0,1), (8,0,2), (2,1,2), (6,1,3), (3,2,3), (7,2,4), (9,3,4)]
mst, total = kruskal(5, edges)
print(f"MST辺: {mst}")     # [(1, 2, 2), (2, 3, 3), (0, 1, 4), (2, 4, 7)]
print(f"合計重み: {total}")  # 16
```

### 4.2 Prim のアルゴリズム

頂点ベースで MST を構築。密グラフでは Kruskal より効率的。

```python
import heapq

def prim(graph: dict, start: int = 0) -> tuple:
    """Prim法 - O((V + E) log V)
    graph: {u: [(v, weight), ...]}
    返り値: (MST辺リスト, 合計重み)
    """
    mst = []
    total = 0
    visited = {start}
    # (weight, from, to) のヒープ
    edges = [(w, start, v) for v, w in graph[start]]
    heapq.heapify(edges)

    while edges and len(mst) < len(graph) - 1:
        w, u, v = heapq.heappop(edges)
        if v in visited:
            continue
        visited.add(v)
        mst.append((u, v, w))
        total += w

        for next_v, next_w in graph[v]:
            if next_v not in visited:
                heapq.heappush(edges, (next_w, v, next_v))

    return mst, total

graph_prim = {
    0: [(1, 4), (2, 8)],
    1: [(0, 4), (2, 2), (3, 6)],
    2: [(0, 8), (1, 2), (3, 3), (4, 7)],
    3: [(1, 6), (2, 3), (4, 9)],
    4: [(2, 7), (3, 9)],
}
mst, total = prim(graph_prim)
print(f"MST辺: {mst}")     # Kruskal と同じ結果
print(f"合計重み: {total}")  # 16
```

### 4.3 Kruskal vs Prim

| 特性 | Kruskal | Prim |
|:---|:---|:---|
| ベース | 辺ベース | 頂点ベース |
| データ構造 | Union-Find | 優先度キュー |
| 計算量 | O(E log E) | O((V+E) log V) |
| 疎グラフ | 効率的 | やや非効率 |
| 密グラフ | 非効率 | 効率的 |
| 切断されたグラフ | 森を返す | 1つの木のみ |
| 実装の簡潔さ | やや複雑（UF必要） | 比較的簡潔 |

### 4.4 MST の正当性: カット性質

```
カット性質 (Cut Property):
  グラフのカット（頂点集合の2分割）を考える。
  カットをまたぐ辺のうち、重みが最小の辺は必ず何らかの MST に含まれる。

  この性質から:
  - Kruskal: 全体で最小の辺を選ぶ → ある2つの連結成分間のカットで最小
  - Prim: 木と非木の間のカットで最小の辺を選ぶ

  どちらもカット性質に基づいて正しい。
```

---

## 5. その他の貪欲法の例

### 5.1 分数ナップサック（Fractional Knapsack）

```python
def fractional_knapsack(items: list, capacity: float) -> float:
    """分数ナップサック - O(n log n)
    items: [(weight, value), ...]
    """
    # 単位重さあたりの価値でソート（降順）
    items_sorted = sorted(items, key=lambda x: x[1]/x[0], reverse=True)

    total_value = 0.0
    remaining = capacity

    for weight, value in items_sorted:
        if remaining >= weight:
            total_value += value
            remaining -= weight
        else:
            total_value += value * (remaining / weight)
            break

    return total_value

items = [(10, 60), (20, 100), (30, 120)]
print(fractional_knapsack(items, 50))  # 240.0
```

### 5.2 区間スケジューリング最大化

```python
def interval_scheduling(intervals: list) -> int:
    """重ならない区間の最大数 - O(n log n)"""
    intervals.sort(key=lambda x: x[1])  # 終了時間でソート
    count = 0
    last_end = float('-inf')

    for start, end in intervals:
        if start >= last_end:
            count += 1
            last_end = end

    return count

intervals = [(1,3), (2,5), (4,7), (1,8), (5,9), (8,10)]
print(interval_scheduling(intervals))  # 3: (1,3), (4,7), (8,10)
```

### 5.3 最小区間カバー

```python
def min_interval_cover(intervals: list, target_start: int, target_end: int) -> list:
    """[target_start, target_end] を最小数の区間でカバー - O(n log n)"""
    intervals.sort()
    result = []
    i = 0
    n = len(intervals)
    current = target_start

    while current < target_end and i < n:
        # current 以前に始まり、最も遠くまで伸びる区間を選ぶ
        best_end = current
        while i < n and intervals[i][0] <= current:
            best_end = max(best_end, intervals[i][1])
            i += 1

        if best_end == current:
            return []  # カバーできない

        result.append(best_end)
        current = best_end

    return result if current >= target_end else []

intervals = [(0, 3), (1, 5), (2, 7), (4, 9), (6, 10)]
print(min_interval_cover(intervals, 0, 10))  # [3, 7, 10]
```

### 5.4 ジョブスケジューリング（デッドライン付き）

```python
def job_scheduling_with_deadlines(jobs: list) -> tuple:
    """デッドライン付きジョブスケジューリング - O(n² log n)
    jobs: [(deadline, profit), ...]
    各ジョブは1単位時間で完了。デッドラインまでに完了すれば利益を得る。
    """
    # 利益の降順でソート
    jobs_sorted = sorted(enumerate(jobs), key=lambda x: x[1][1], reverse=True)
    max_deadline = max(d for d, _ in jobs)

    # スロット管理（1-indexed）
    slots = [False] * (max_deadline + 1)
    result = []
    total_profit = 0

    for idx, (deadline, profit) in jobs_sorted:
        # デッドライン以前で最も遅い空きスロットを探す
        for t in range(min(deadline, max_deadline), 0, -1):
            if not slots[t]:
                slots[t] = True
                result.append((idx, t, profit))
                total_profit += profit
                break

    return total_profit, result

jobs = [(2, 100), (1, 19), (2, 27), (1, 25), (3, 15)]
profit, schedule = job_scheduling_with_deadlines(jobs)
print(f"最大利益: {profit}")    # 142
print(f"スケジュール: {schedule}")
```

### 5.5 文字列圧縮（Run-Length Encoding）

```python
def run_length_encode(s: str) -> str:
    """ランレングス符号化 — 貪欲法の一種"""
    if not s:
        return ""

    result = []
    count = 1

    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            count += 1
        else:
            result.append(f"{s[i-1]}{count}")
            count = 1

    result.append(f"{s[-1]}{count}")
    return ''.join(result)

def run_length_decode(encoded: str) -> str:
    """ランレングス復号"""
    result = []
    i = 0
    while i < len(encoded):
        char = encoded[i]
        i += 1
        num = []
        while i < len(encoded) and encoded[i].isdigit():
            num.append(encoded[i])
            i += 1
        result.append(char * int(''.join(num)))
    return ''.join(result)

original = "AAABBBCCDDDDEEFFFFF"
encoded = run_length_encode(original)
print(f"符号化: {encoded}")      # A3B3C2D4E2F5
print(f"復号化: {run_length_decode(encoded)}")  # AAABBBCCDDDDEEFFFFF
```

### 5.6 Dijkstra法（貪欲法としての視点）

```python
# Dijkstra法は貪欲法の一例として理解できる:
#
# 貪欲選択: 「未確定頂点の中で距離が最小の頂点を確定する」
#
# 貪欲選択性質の証明:
#   - 未確定頂点 u の距離 d[u] が最小
#   - 別の経路 s → ... → w → ... → u が d[u] より短いと仮定
#   - しかし w は未確定なので d[w] >= d[u]
#   - 辺の重みは非負なので w 経由の経路は d[w] 以上 >= d[u]
#   - 矛盾 → d[u] は最短
#
# この証明が成り立つ条件: 全辺の重みが非負
# 負辺があると成り立たない → Bellman-Ford（DPベース）が必要
```

---

## 6. マトロイド理論と貪欲法

```
マトロイドの定義:
  集合 S と独立集合族 I が以下を満たすとき (S, I) はマトロイド:
  1. 空集合は独立 (∅ ∈ I)
  2. 遺伝性: A ∈ I かつ B ⊆ A ならば B ∈ I
  3. 交換性: A, B ∈ I かつ |A| < |B| ならば、
             ある b ∈ B\A が存在して A ∪ {b} ∈ I

定理（Rado-Edmonds）:
  重み付きマトロイドの最大重み独立集合は、
  貪欲法（重みの大きい要素から順に、独立性を保って追加）で求まる。

例:
  - グラフの辺集合 + 森（サイクルなし）の条件 → グラフ的マトロイド
  - → 最小全域木が貪欲法（Kruskal）で求まる理由
```

---

## 7. 貪欲法 vs DP 比較表

| 特性 | 貪欲法 | 動的計画法 |
|:---|:---|:---|
| 選択方法 | 局所最適を即座に決定 | 全選択肢を比較 |
| 後戻り | なし | なし（全探索済み） |
| 計算量 | 通常 O(n log n) | 通常 O(n^2) 以上 |
| 正当性の証明 | 必要（反例がないか確認） | 遷移式の正しさで証明 |
| 適用範囲 | 狭い（条件が厳しい） | 広い |
| 実装の簡潔さ | 簡潔 | やや複雑 |
| 空間計算量 | O(1)〜O(n) | O(n)〜O(n^2) |
| 最適性の保証 | 証明されていれば保証 | 常に保証 |

## 貪欲法で解ける問題・解けない問題

| 問題 | 貪欲で解けるか | 理由 |
|:---|:---|:---|
| 活動選択問題 | 解ける | 終了時間順の貪欲選択が最適 |
| 分数ナップサック | 解ける | 単価順の選択が最適 |
| 0/1 ナップサック | 解けない | 分割不可 → DP 必要 |
| ハフマン符号 | 解ける | 頻度最小の統合が最適 |
| 最小全域木 | 解ける | カット性質による正当性 |
| 最短経路（負辺なし） | 解ける | Dijkstra は貪欲 |
| コイン問題（一般） | 解けない | 特定額面でのみ貪欲が有効 |
| 重み付き活動選択 | 解けない | DP が必要 |
| 集合被覆問題 | 近似のみ | NP困難、貪欲は ln(n) 近似 |
| 最小点彩色 | 解けない | NP困難 |

---

## 8. 実務応用パターン

### 8.1 CDNのサーバー配置

```python
def greedy_facility_placement(cities: list, k: int) -> list:
    """k 個の施設を貪欲に配置（最大距離の最小化）
    これは近似アルゴリズム（最適解の2倍以内を保証）
    """
    n = len(cities)
    if k >= n:
        return list(range(n))

    # 最初の施設: 任意（ここでは都市0）
    facilities = [0]
    min_dist = [float('inf')] * n

    for _ in range(k - 1):
        # 各都市の最寄り施設までの距離を更新
        last = facilities[-1]
        for j in range(n):
            d = abs(cities[j][0] - cities[last][0]) + abs(cities[j][1] - cities[last][1])
            min_dist[j] = min(min_dist[j], d)

        # 最寄り施設から最も遠い都市を次の施設に
        farthest = max(range(n), key=lambda j: min_dist[j] if j not in facilities else -1)
        facilities.append(farthest)

    return facilities
```

### 8.2 タスクの締め切り最適化

```python
def minimize_lateness(tasks: list) -> tuple:
    """遅延の最大値を最小化するスケジューリング
    tasks: [(processing_time, deadline), ...]
    最適戦略: 締め切りが早い順に処理（EDF: Earliest Deadline First）
    """
    indexed = [(d, p, i) for i, (p, d) in enumerate(tasks)]
    indexed.sort()  # 締め切り順

    schedule = []
    current_time = 0
    max_lateness = 0

    for deadline, proc_time, idx in indexed:
        start = current_time
        finish = current_time + proc_time
        lateness = max(0, finish - deadline)
        max_lateness = max(max_lateness, lateness)
        schedule.append((idx, start, finish, lateness))
        current_time = finish

    return max_lateness, schedule

tasks = [(3, 6), (2, 8), (1, 9), (4, 9), (3, 14), (2, 15)]
max_late, sched = minimize_lateness(tasks)
print(f"最大遅延: {max_late}")
for idx, start, finish, late in sched:
    print(f"  タスク{idx}: {start}-{finish} (遅延: {late})")
```

### 8.3 ページキャッシュの置換戦略

```python
def optimal_page_replacement(pages: list, cache_size: int) -> int:
    """Belady の最適ページ置換アルゴリズム（将来のアクセスを知っている前提）
    page fault の回数を返す
    """
    cache = set()
    faults = 0

    for i, page in enumerate(pages):
        if page in cache:
            continue

        faults += 1

        if len(cache) < cache_size:
            cache.add(page)
        else:
            # キャッシュ内で、次に使われるのが最も遅いページを追い出す
            farthest = -1
            victim = None
            for cached_page in cache:
                try:
                    next_use = pages[i+1:].index(cached_page)
                except ValueError:
                    next_use = float('inf')  # 二度と使われない

                if next_use > farthest:
                    farthest = next_use
                    victim = cached_page

            cache.remove(victim)
            cache.add(page)

    return faults

pages = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1]
print(f"Page faults (OPT): {optimal_page_replacement(pages, 3)}")   # 9
print(f"Page faults (OPT): {optimal_page_replacement(pages, 4)}")   # 6
```

### 8.4 貪欲近似: 集合被覆問題

```python
def greedy_set_cover(universe: set, subsets: list) -> list:
    """集合被覆問題の貪欲近似 - O(|U| * |S|)
    NP困難問題だが、貪欲法で ln(|U|)+1 倍以内の近似解を得られる
    """
    uncovered = set(universe)
    selected = []

    while uncovered:
        # 未カバー要素を最も多くカバーする集合を選ぶ
        best = max(range(len(subsets)),
                   key=lambda i: len(subsets[i] & uncovered) if i not in selected else 0)

        if not (subsets[best] & uncovered):
            break  # これ以上カバーできない

        selected.append(best)
        uncovered -= subsets[best]

    return selected

universe = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
subsets = [
    {1, 2, 3, 8},      # S0
    {1, 2, 3, 4, 5},   # S1
    {4, 5, 7},          # S2
    {5, 6, 7},          # S3
    {6, 7, 8, 9, 10},  # S4
]
selected = greedy_set_cover(universe, subsets)
print(f"選択された集合: {selected}")  # [1, 4] or similar
```

---

## 9. アンチパターン

### アンチパターン1: 貪欲選択性質の未証明

```python
# BAD: 「直感的に正しそう」で貪欲法を適用
# コイン問題: 額面 = [1, 3, 4], 金額 = 6
# 貪欲（最大額面優先）: 4 + 1 + 1 = 3枚
# 最適解: 3 + 3 = 2枚  ← 貪欲は失敗!

def bad_coin_greedy(coins, amount):
    coins.sort(reverse=True)
    count = 0
    for coin in coins:
        count += amount // coin
        amount %= coin
    return count

print(bad_coin_greedy([1, 3, 4], 6))  # 3 (不正解、正解は2)

# GOOD: この問題にはDPを使う
def good_coin_dp(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for c in coins:
            if c <= i:
                dp[i] = min(dp[i], dp[i - c] + 1)
    return dp[amount]

print(good_coin_dp([1, 3, 4], 6))  # 2 (正解)
```

### アンチパターン2: 0/1 ナップサックに貪欲法

```python
# BAD: 0/1ナップサックに単価順の貪欲を適用
# アイテム: (重さ=10, 価値=60), (重さ=20, 価値=100), (重さ=30, 価値=120)
# 容量: 50
# 貪欲（単価順）: 60 + 100 = 160
# 最適解: 100 + 120 = 220  ← 重さ20+30=50で入る

# GOOD: 0/1ナップサックにはDPを使う
```

### アンチパターン3: ソートの基準を間違える

```python
# BAD: 活動選択で「開始時間」でソート → 最適でない
def bad_activity_selection(activities):
    activities.sort(key=lambda x: x[0])  # 開始時間でソート → 不正解
    ...

# GOOD: 「終了時間」でソート
def good_activity_selection(activities):
    activities.sort(key=lambda x: x[1])  # 終了時間でソート → 正解
    ...
```

### アンチパターン4: 貪欲法の「近似」を「最適」と誤解

```python
# BAD: 集合被覆問題に貪欲を使い、「最適解が得られた」と主張
# → 貪欲は近似解であり、最適とは限らない

# GOOD: 近似比を明記する
# 「貪欲法による近似解。最適解の ln(n)+1 倍以内を保証」
```

---

## 10. コイン問題における貪欲法の成立条件

```
コイン問題で貪欲法（大きい額面から使う）が最適になる条件:

1. 額面が canonical coin system を満たす
   例: [1, 5, 10, 25, 50, 100] (USドル) → 貪欲で最適
   例: [1, 5, 10, 50, 100, 500] (日本円) → 貪欲で最適

2. canonical でない例:
   [1, 3, 4]     → 金額6で貪欲が失敗
   [1, 5, 6, 9]  → 金額11で貪欲(9+1+1=3枚)が失敗、最適は(6+5=2枚)

判定方法:
   全ての額面の組について検証するか、
   Pearson (2005) のアルゴリズムで多項式時間で判定可能
```

---

## 11. FAQ

### Q1: 貪欲法の正当性はどう証明する？

**A:** 主な証明手法は2つ。(1) **交換論法**: 最適解が貪欲解と異なると仮定し、貪欲解の要素と交換しても最適性が維持される（または改善される）ことを示す。(2) **マトロイド理論**: 問題構造がマトロイドの公理を満たすなら、貪欲法が最適。実用的には反例を探す→見つからなければ交換論法で証明が一般的。

### Q2: 貪欲法とヒューリスティックの違いは？

**A:** 貪欲法は正当性が証明された場合、最適解を保証する。ヒューリスティックは近似解を素早く得る手法で、最適性は保証しない。貪欲選択性質が成り立たない問題に貪欲法を適用すると、それは（精度の低い）ヒューリスティックになる。

### Q3: Prim法とKruskal法の使い分けは？

**A:** 両方とも最小全域木を求める貪欲アルゴリズム。Kruskal は辺ベース（E log E）で疎グラフに強い。Prim は頂点ベース（優先度キューで V log V + E）で密グラフに強い。辺数が少なければ Kruskal、多ければ Prim が効率的。

### Q4: ダイクストラ法は貪欲法なのか？

**A:** はい。Dijkstra法は「未確定頂点のうち距離最小のものを確定する」という貪欲選択を繰り返す。非負辺の条件下でこの貪欲選択が最適であることが証明されている。負辺があると貪欲選択性質が崩れるため、Bellman-Ford（DPベース）が必要になる。

### Q5: 貪欲法を使うべきでないのはどんな場合？

**A:** (1) 反例が見つかる場合。(2) 問題が NP 困難で最適解が必要な場合（貪欲は近似のみ）。(3) 選択の影響が将来に及ぶ場合（例: 0/1ナップサック）。(4) 「最適」ではなく「全ての解」を列挙する必要がある場合。

### Q6: 貪欲法とビームサーチの関係は？

**A:** ビームサーチは貪欲法の拡張版。貪欲法が各ステップで1つの最良候補のみを保持するのに対し、ビームサーチは上位 k 個の候補を保持する。k=1 がまさに貪欲法。k を増やすと解の品質が向上するが、計算量も増える。自然言語処理のデコーディングなどで広く使われる。

---

## 12. まとめ

| 項目 | 要点 |
|:---|:---|
| 貪欲法の条件 | 貪欲選択性質 + 最適部分構造 |
| 活動選択問題 | 終了時間順に選択。区間スケジューリングの基本 |
| ハフマン符号 | 頻度最小のペアを統合。最適接頭辞符号 |
| Kruskal | 辺を重み順に追加。Union-Find でサイクル判定 |
| Prim | 頂点ベースで MST を構築。密グラフに有利 |
| 分数ナップサック | 単価順に選択。0/1 とは異なり貪欲で最適 |
| マトロイド | 貪欲法の正当性を保証する理論的枠組み |
| DP との使い分け | 貪欲で解けるなら貪欲（高速）、解けなければ DP |
| 正当性の証明 | 交換論法またはマトロイド理論で証明が必須 |

---

## 次に読むべきガイド

- [動的計画法](./04-dynamic-programming.md) -- 貪欲法で解けない問題への対処
- [分割統治法](./06-divide-conquer.md) -- もう一つの設計パラダイム
- [Union-Find](../03-advanced/00-union-find.md) -- Kruskal に不可欠なデータ構造

---

## 参考文献

1. Cormen, T. H. et al. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. -- 第15章
2. Huffman, D. A. (1952). "A Method for the Construction of Minimum-Redundancy Codes." *Proceedings of the IRE*.
3. Kruskal, J. B. (1956). "On the shortest spanning subtree of a graph." *Proceedings of the AMS*.
4. Prim, R. C. (1957). "Shortest connection networks and some generalizations." *Bell System Technical Journal*.
5. Kleinberg, J. & Tardos, E. (2005). *Algorithm Design*. Pearson. -- Chapter 4: Greedy Algorithms
6. Oxley, J. G. (2011). *Matroid Theory* (2nd ed.). Oxford University Press.
