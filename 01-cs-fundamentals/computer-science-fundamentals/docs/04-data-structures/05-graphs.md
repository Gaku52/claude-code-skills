# グラフ（データ構造としての）

> グラフはノードとエッジの集合であり、木、リンクリスト、さらには配列までもがグラフの特殊ケースである。

## この章で学ぶこと

- [ ] グラフの表現方法（隣接リスト、隣接行列）を実装できる
- [ ] 重み付き/有向/無向グラフの違いを理解する
- [ ] 実務でのグラフ表現パターンを知る

## 前提知識

- グラフアルゴリズム → 参照: [[../03-algorithms/04-graph-algorithms.md]]

---

## 1. グラフの表現

### 1.1 隣接リスト

```python
# 無向グラフ
from collections import defaultdict

class Graph:
    def __init__(self):
        self.adj = defaultdict(list)

    def add_edge(self, u, v, weight=1):
        self.adj[u].append((v, weight))
        self.adj[v].append((u, weight))  # 無向グラフ

    def neighbors(self, u):
        return self.adj[u]

# 空間: O(V + E)
# 辺の追加: O(1)
# 辺の存在確認: O(degree)
# 全辺の列挙: O(V + E)
# → 疎なグラフ（辺が少ない）に最適
```

### 1.2 隣接行列

```python
class GraphMatrix:
    def __init__(self, n):
        self.n = n
        self.matrix = [[0] * n for _ in range(n)]

    def add_edge(self, u, v, weight=1):
        self.matrix[u][v] = weight
        self.matrix[v][u] = weight  # 無向グラフ

    def has_edge(self, u, v):
        return self.matrix[u][v] != 0

# 空間: O(V²)
# 辺の追加: O(1)
# 辺の存在確認: O(1)
# → 密なグラフ（辺が多い）に最適
# → フロイドワーシャルに適する
```

### 1.3 どちらを選ぶか

```
選択基準:

  ┌────────────────┬──────────────┬──────────────┐
  │ 条件           │ 隣接リスト   │ 隣接行列     │
  ├────────────────┼──────────────┼──────────────┤
  │ 疎なグラフ     │ ✅ 省メモリ  │ ❌ 無駄多い  │
  │ 密なグラフ     │ △          │ ✅ 効率的    │
  │ 辺の存在確認   │ O(degree)   │ O(1) ✅     │
  │ 全隣接ノード   │ O(degree) ✅│ O(V)        │
  │ メモリ         │ O(V+E) ✅   │ O(V²)       │
  └────────────────┴──────────────┴──────────────┘

  実務での目安:
  - ほとんどの場合: 隣接リスト（グラフは通常疎）
  - V < 1000 かつ密なグラフ: 隣接行列も検討
  - SNS(数億ノード): 隣接リスト一択
```

---

## 2. 実務でのグラフ

```
実務でのグラフ表現:

  1. RDB: テーブル間のリレーション → 暗黙のグラフ
     users, follows テーブル → ソーシャルグラフ

  2. Neo4j: グラフDB
     Cypher: MATCH (a)-[:FOLLOWS]->(b) WHERE a.name = 'Alice'

  3. GraphQL: APIのグラフ構造

  4. npm/pip: パッケージ依存関係グラフ（DAG）

  5. Kubernetes: サービス間通信（サービスメッシュ）
```

---

## 3. 実践演習

### 演習1: グラフ構築（基礎）
隣接リストと隣接行列の両方でグラフを実装し、BFS/DFSを実行せよ。

### 演習2: 二部グラフ判定（応用）
グラフが二部グラフかどうかをBFSで判定する関数を実装せよ。

### 演習3: 最小全域木（発展）
クラスカル法で最小全域木を求める関数を実装せよ（Union-Find使用）。

---

## まとめ

| 表現方法 | 空間 | 辺の確認 | 最適場面 |
|---------|------|---------|---------|
| 隣接リスト | O(V+E) | O(degree) | 疎グラフ、一般用途 |
| 隣接行列 | O(V²) | O(1) | 密グラフ、小規模 |
| エッジリスト | O(E) | O(E) | クラスカル法 |

---

## 次に読むべきガイド
→ [[06-advanced-structures.md]] — 高度なデータ構造

---

## 参考文献
1. Cormen, T. H. "Introduction to Algorithms." Chapter 22.
2. Sedgewick, R. "Algorithms." Chapter 4.1.
