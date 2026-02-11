# 貪欲法（Greedy Algorithm）

> 各ステップで局所的に最適な選択を繰り返すことで、全体の最適解を効率的に求める設計手法を理解する

## この章で学ぶこと

1. **貪欲法の適用条件**（貪欲選択性質・最適部分構造）を見抜き、正当性を検証できる
2. **活動選択問題・ハフマン符号・最小全域木**を貪欲法で正しく解ける
3. **貪欲法と DP の使い分け**を判断でき、貪欲法が使えない場合を識別できる

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

# 使用例
text = "aaaaabbbccddddeefffff"
encoded, codes, tree = huffman_encode(text)
print("符号表:", codes)
print(f"元のサイズ: {len(text) * 8} ビット")
print(f"圧縮後: {len(encoded)} ビット")
print(f"圧縮率: {len(encoded) / (len(text) * 8):.1%}")
```

---

## 4. Kruskal のアルゴリズム（最小全域木）

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

辺の重み順: (A,B,2)? → (C,D,3) → (A,B,4) → ...

Step1: C-D (3) 追加  ← サイクルなし
Step2: A-B (4) 追加  ← サイクルなし
Step3: B-C (2) 追加  ← サイクルなし
Step4: C-E (7) 追加  ← サイクルなし
Step5: B-D (6) スキップ ← B-C-D でサイクル!

MST:      B
         / |
        2  4
       /   |
      C-3-D  A
       \
        7
         \
          E
  合計重み: 2 + 3 + 4 + 7 = 16
```

```python
class UnionFind:
    """Union-Find（Kruskal用）"""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
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

---

## 5. その他の貪欲法の例

### 分数ナップサック（Fractional Knapsack）

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

### 区間スケジューリング最大化

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

---

## 6. 貪欲法 vs DP 比較表

| 特性 | 貪欲法 | 動的計画法 |
|:---|:---|:---|
| 選択方法 | 局所最適を即座に決定 | 全選択肢を比較 |
| 後戻り | なし | なし（全探索済み） |
| 計算量 | 通常 O(n log n) | 通常 O(n²) 以上 |
| 正当性の証明 | 必要（反例がないか確認） | 遷移式の正しさで証明 |
| 適用範囲 | 狭い（条件が厳しい） | 広い |
| 実装の簡潔さ | 簡潔 | やや複雑 |

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

---

## 7. アンチパターン

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

---

## 8. FAQ

### Q1: 貪欲法の正当性はどう証明する？

**A:** 主な証明手法は2つ。(1) **交換論法**: 最適解が貪欲解と異なると仮定し、貪欲解の要素と交換しても最適性が維持される（または改善される）ことを示す。(2) **マトロイド理論**: 問題構造がマトロイドの公理を満たすなら、貪欲法が最適。実用的には反例を探す→見つからなければ交換論法で証明が一般的。

### Q2: 貪欲法とヒューリスティックの違いは？

**A:** 貪欲法は正当性が証明された場合、最適解を保証する。ヒューリスティックは近似解を素早く得る手法で、最適性は保証しない。貪欲選択性質が成り立たない問題に貪欲法を適用すると、それは（精度の低い）ヒューリスティックになる。

### Q3: Prim法とKruskal法の使い分けは？

**A:** 両方とも最小全域木を求める貪欲アルゴリズム。Kruskal は辺ベース（E log E）で疎グラフに強い。Prim は頂点ベース（優先度キューで V log V + E）で密グラフに強い。辺数が少なければ Kruskal、多ければ Prim が効率的。

### Q4: ダイクストラ法は貪欲法なのか？

**A:** はい。Dijkstra法は「未確定頂点のうち距離最小のものを確定する」という貪欲選択を繰り返す。非負辺の条件下でこの貪欲選択が最適であることが証明されている。負辺があると貪欲選択性質が崩れるため、Bellman-Ford（DPベース）が必要になる。

---

## 9. まとめ

| 項目 | 要点 |
|:---|:---|
| 貪欲法の条件 | 貪欲選択性質 + 最適部分構造 |
| 活動選択問題 | 終了時間順に選択。区間スケジューリングの基本 |
| ハフマン符号 | 頻度最小のペアを統合。最適接頭辞符号 |
| Kruskal | 辺を重み順に追加。Union-Find でサイクル判定 |
| 分数ナップサック | 単価順に選択。0/1 とは異なり貪欲で最適 |
| DP との使い分け | 貪欲で解けるなら貪欲（高速）、解けなければ DP |

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
4. Kleinberg, J. & Tardos, E. (2005). *Algorithm Design*. Pearson. -- Chapter 4: Greedy Algorithms
