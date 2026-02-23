# Union-Find（素集合データ構造）

> 要素のグループ化と所属判定を準定数時間で行うデータ構造を、経路圧縮・ランク付き合併・Kruskal応用を通じて理解する

## この章で学ぶこと

1. **Union-Find の2操作**（Find/Union）の原理と、経路圧縮・ランク付き合併による最適化を実装できる
2. **逆アッカーマン関数**α(n) による準定数時間の計算量を理解する
3. **Kruskal法・連結成分・等価クラス分け**など実用的な応用パターンを使いこなせる
4. **重み付きUnion-Find・永続Union-Find**などの発展的なバリエーションを理解する
5. **実務におけるクラスタリング・ネットワーク管理**への応用を実装できる

---

## 1. Union-Find の概念

Union-Find（素集合データ構造、Disjoint Set Union: DSU）は、要素の集合を互いに素な（重なりのない）部分集合に分割して管理するデータ構造である。主に以下の2つの操作を提供する。

- **Find(x)**: 要素 x が属する集合の代表元（根）を返す
- **Union(x, y)**: 要素 x と y が属する集合を統合する

```
Union-Find = 素集合の森（Disjoint Set Forest）

初期状態（各要素が独立）:
  {0} {1} {2} {3} {4} {5} {6} {7}

Union(0,1), Union(2,3), Union(4,5):
  {0,1} {2,3} {4,5} {6} {7}

Union(0,2), Union(4,6):
  {0,1,2,3} {4,5,6} {7}

Union(0,4):
  {0,1,2,3,4,5,6} {7}

Find(5) → 0（代表元）
Find(7) → 7（代表元）
Find(5) == Find(3) → True（同じ集合）
Find(5) == Find(7) → False（異なる集合）
```

### なぜ Union-Find が重要か

Union-Find は一見単純なデータ構造だが、以下のような場面で極めて強力である。

1. **動的な連結性判定**: グラフに辺が逐次追加される状況で「頂点 u と v は連結か？」をほぼ O(1) で回答
2. **最小全域木 (MST)**: Kruskal法のサイクル検出に不可欠
3. **等価クラス管理**: 同値関係の推移的閉包を効率的に管理
4. **画像処理**: ラベリング（連結成分の識別）
5. **ネットワーク設計**: 回線の冗長性判定、ルーティング

```
実世界の例:

1. ソーシャルネットワーク
   「AさんとBさんは（友達の友達を辿って）繋がっているか？」
   → Union-Find で友達関係を管理 → Find で O(α(n)) ≈ O(1) 判定

2. コンピュータネットワーク
   「マシンXとマシンYは通信可能か？」
   → ケーブルの接続をUnionで登録、到達可能性をFindで判定

3. 数独の領域管理
   「セルAとセルBは同じブロックに属するか？」

4. コンパイラの型推論
   「型変数αと型変数βは同じ型に解決されるか？」
```

---

## 2. 木構造による表現

Union-Find は内部的に森（木の集合）として表現される。各集合は1つの木に対応し、木の根がその集合の代表元となる。

```
ナイーブな実装（木が偏る）:

Union(0,1): 1→0      Union(0,1,2,3,4):
Union(0,2): 2→0          0
Union(0,3): 3→0         /|\ \
Union(0,4): 4→0        1 2 3 4    ← バランスが良い

最悪ケース（チェーン状）:
  Union(0,1), Union(1,2), Union(2,3), Union(3,4)
    0 ← 1 ← 2 ← 3 ← 4    ← Find(4) が O(n)!

経路圧縮後:
         0
       / | \ \
      1  2  3  4    ← Find(4) が O(1)!
```

### 配列による表現

Union-Find は `parent` 配列で表現する。`parent[i]` は要素 i の親を格納し、根の場合は `parent[i] = i` とする。

```
配列表現の例:

集合 {0,1,2,3} と {4,5,6} を管理:

parent: [0, 0, 0, 0, 4, 4, 4]
         ↑     ↑           ↑
         根    0の子       根

木構造:
    0       4
   /|\     / \
  1 2 3   5   6

Find(3) → parent[3] = 0 → parent[0] = 0 → 根は 0
Find(6) → parent[6] = 4 → parent[4] = 4 → 根は 4
```

---

## 3. 基本実装

```python
class UnionFind:
    """Union-Find（素集合データ構造）
    経路圧縮 + ランク付き合併で O(α(n)) ≈ O(1)
    """

    def __init__(self, n: int):
        self.parent = list(range(n))  # 各要素の親
        self.rank = [0] * n           # 木の高さの上界
        self.size = [1] * n           # 各集合のサイズ
        self.count = n                # 集合の数

    def find(self, x: int) -> int:
        """x の代表元（根）を返す - 経路圧縮付き"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 経路圧縮
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """x と y の集合を合併 - ランク付き合併
        返り値: 合併が行われたか（既に同じ集合ならFalse）
        """
        px, py = self.find(x), self.find(y)
        if px == py:
            return False

        # ランクが低い方を高い方の子にする
        if self.rank[px] < self.rank[py]:
            px, py = py, px

        self.parent[py] = px
        self.size[px] += self.size[py]

        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

        self.count -= 1
        return True

    def connected(self, x: int, y: int) -> bool:
        """x と y が同じ集合に属するか"""
        return self.find(x) == self.find(y)

    def get_size(self, x: int) -> int:
        """x が属する集合のサイズ"""
        return self.size[self.find(x)]

    def get_groups(self) -> dict:
        """全グループを辞書として返す {代表元: [要素リスト]}"""
        groups = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)
        return groups

# 使用例
uf = UnionFind(8)
uf.union(0, 1)
uf.union(2, 3)
uf.union(0, 2)
print(uf.connected(1, 3))  # True
print(uf.connected(0, 5))  # False
print(uf.get_size(0))      # 4
print(uf.count)             # 5 (集合数)
print(uf.get_groups())     # {0: [0, 1, 2, 3], 4: [4], 5: [5], 6: [6], 7: [7]}
```

### C++ 実装

```cpp
#include <vector>
#include <numeric>

class UnionFind {
    std::vector<int> parent, rank_, size_;
    int count_;

public:
    UnionFind(int n) : parent(n), rank_(n, 0), size_(n, 1), count_(n) {
        std::iota(parent.begin(), parent.end(), 0);
    }

    int find(int x) {
        if (parent[x] != x)
            parent[x] = find(parent[x]);  // 経路圧縮
        return parent[x];
    }

    bool unite(int x, int y) {
        int px = find(x), py = find(y);
        if (px == py) return false;

        if (rank_[px] < rank_[py]) std::swap(px, py);
        parent[py] = px;
        size_[px] += size_[py];
        if (rank_[px] == rank_[py]) rank_[px]++;
        count_--;
        return true;
    }

    bool connected(int x, int y) { return find(x) == find(y); }
    int getSize(int x) { return size_[find(x)]; }
    int getCount() const { return count_; }
};

// 使用例
// UnionFind uf(8);
// uf.unite(0, 1);
// uf.unite(2, 3);
// cout << uf.connected(1, 3) << endl;  // 0 (false)
// uf.unite(0, 2);
// cout << uf.connected(1, 3) << endl;  // 1 (true)
```

### TypeScript 実装

```typescript
class UnionFind {
    private parent: number[];
    private rank: number[];
    private size: number[];
    private _count: number;

    constructor(n: number) {
        this.parent = Array.from({ length: n }, (_, i) => i);
        this.rank = new Array(n).fill(0);
        this.size = new Array(n).fill(1);
        this._count = n;
    }

    find(x: number): number {
        if (this.parent[x] !== x) {
            this.parent[x] = this.find(this.parent[x]);
        }
        return this.parent[x];
    }

    union(x: number, y: number): boolean {
        let px = this.find(x);
        let py = this.find(y);
        if (px === py) return false;

        if (this.rank[px] < this.rank[py]) [px, py] = [py, px];
        this.parent[py] = px;
        this.size[px] += this.size[py];
        if (this.rank[px] === this.rank[py]) this.rank[px]++;
        this._count--;
        return true;
    }

    connected(x: number, y: number): boolean {
        return this.find(x) === this.find(y);
    }

    getSize(x: number): number {
        return this.size[this.find(x)];
    }

    get count(): number {
        return this._count;
    }
}
```

---

## 4. 経路圧縮の詳細

経路圧縮（Path Compression）は Find 操作時に、パス上のすべてのノードを直接根に接続する最適化技法である。これにより、次回以降の Find が高速化される。

```
Find(7) の経路圧縮:

Before:           After:
    0                 0
    |              / | \ \
    1             1  3  5  7
    |
    3
    |
    5
    |
    7

Find(7): 7→5→3→1→0  (根を発見)
圧縮:   7→0, 5→0, 3→0, 1→0  (全ノードを根の直下に)
```

```python
# 経路圧縮の3つの方法

# 方法1: 再帰（上記の実装）- 完全な経路圧縮
def find_recursive(self, x):
    if self.parent[x] != x:
        self.parent[x] = self.find(self.parent[x])
    return self.parent[x]

# 方法2: 反復（スタックオーバーフロー回避）- 完全な経路圧縮
def find_iterative(self, x):
    root = x
    while self.parent[root] != root:
        root = self.parent[root]
    # 経路上の全ノードを根に直接接続
    while self.parent[x] != root:
        next_x = self.parent[x]
        self.parent[x] = root
        x = next_x
    return root

# 方法3: パス分割（Path Splitting）- 実装が簡潔
def find_splitting(self, x):
    while self.parent[x] != x:
        self.parent[x] = self.parent[self.parent[x]]  # 祖父に接続
        x = self.parent[x]
    return x

# 方法4: パス半分化（Path Halving）- パス分割の変種
def find_halving(self, x):
    while self.parent[x] != x:
        self.parent[x] = self.parent[self.parent[x]]  # 祖父に接続
        x = self.parent[x]  # 2つ先に進む
    return x
```

### 経路圧縮の効果の可視化

```python
def visualize_compression():
    """経路圧縮の効果を可視化するデモ"""
    uf = UnionFind(10)

    # チェーン状の木を作成
    # 0 ← 1 ← 2 ← 3 ← 4 ← 5 ← 6 ← 7 ← 8 ← 9
    for i in range(9):
        uf.parent[i + 1] = i  # 強制的にチェーンを作る
    uf.rank[0] = 9
    uf.count = 1
    for i in range(10):
        uf.size[i] = 1
    uf.size[0] = 10

    print("Before compression:")
    print(f"  parent = {uf.parent}")
    # [0, 0, 1, 2, 3, 4, 5, 6, 7, 8]

    # Find(9) で経路圧縮が発動
    root = uf.find(9)
    print(f"\nAfter find(9):")
    print(f"  root = {root}")
    print(f"  parent = {uf.parent}")
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # すべてのノードが直接根に接続された

visualize_compression()
```

---

## 5. ランク付き合併とサイズ付き合併

### ランク付き合併（Union by Rank）

木の高さ（ランク）を基準に、低い方を高い方の子にする。木の高さが O(log n) に抑えられる。

```python
def union_by_rank(self, x, y):
    px, py = self.find(x), self.find(y)
    if px == py:
        return False

    # ランクが低い方を高い方の子にする
    if self.rank[px] < self.rank[py]:
        px, py = py, px  # px が高い方

    self.parent[py] = px
    if self.rank[px] == self.rank[py]:
        self.rank[px] += 1  # ランクが同じなら +1

    return True
```

### サイズ付き合併（Union by Size）

集合のサイズを基準に、小さい方を大きい方の子にする。実装がシンプルで実用上はランクと同等の性能。

```python
def union_by_size(self, x, y):
    px, py = self.find(x), self.find(y)
    if px == py:
        return False

    # サイズが小さい方を大きい方の子にする
    if self.size[px] < self.size[py]:
        px, py = py, px  # px が大きい方

    self.parent[py] = px
    self.size[px] += self.size[py]

    return True
```

### 合併戦略の比較

```
ランク付き合併 vs サイズ付き合併:

       ランク付き合併              サイズ付き合併
  木の高さを基準に合併        集合のサイズを基準に合併
  rank 配列が必要             size 配列が必要（他の用途にも有用）
  理論的に最適                実用上は同等の性能

  推奨: サイズ付き合併
  理由: size 配列は「集合の要素数は？」というクエリにも使える
```

---

## 6. Kruskal法への応用

Union-Find は Kruskal の最小全域木アルゴリズムで不可欠。サイクル判定を O(α(n)) で行うことで、全体として O(E log E) を実現する。

```python
def kruskal_mst(n: int, edges: list) -> tuple:
    """Kruskal法 - Union-Find使用 - O(E log E)
    edges: [(weight, u, v), ...]
    returns: (mst_edges, mst_weight)
    """
    edges.sort()  # 重みでソート
    uf = UnionFind(n)
    mst_edges = []
    mst_weight = 0

    for weight, u, v in edges:
        if not uf.connected(u, v):  # サイクル判定
            uf.union(u, v)
            mst_edges.append((u, v, weight))
            mst_weight += weight

            if len(mst_edges) == n - 1:
                break  # MST完成

    return mst_edges, mst_weight

# 使用例
# 頂点: 0-4, 辺: (重み, 始点, 終点)
edges = [
    (1, 0, 1), (4, 0, 2), (3, 1, 2),
    (2, 1, 3), (5, 2, 3), (7, 2, 4), (6, 3, 4)
]
mst, total = kruskal_mst(5, edges)
print(f"MST辺: {mst}")        # [(0, 1, 1), (1, 3, 2), (1, 2, 3), (3, 4, 6)]
print(f"合計重み: {total}")    # 12
```

### Kruskal法の動作を詳細にトレース

```
グラフ:
    0 ---1--- 1
    |  \      |  \
    4   3     2    2
    |     \   |      \
    2 ---5--- 3 ---6--- 4
              |
              7
              |
              4（辺2-4の重み）

辺をソート: (1,0,1) (2,1,3) (3,1,2) (4,0,2) (5,2,3) (6,3,4) (7,2,4)

Step 1: (1, 0, 1)  → 0-1 は非連結 → Union(0,1) ✓  MST: {(0,1,1)}
Step 2: (2, 1, 3)  → 1-3 は非連結 → Union(1,3) ✓  MST: {(0,1,1), (1,3,2)}
Step 3: (3, 1, 2)  → 1-2 は非連結 → Union(1,2) ✓  MST: {(0,1,1), (1,3,2), (1,2,3)}
Step 4: (4, 0, 2)  → 0-2 は連結   → スキップ ✗   （サイクルを形成）
Step 5: (5, 2, 3)  → 2-3 は連結   → スキップ ✗   （サイクルを形成）
Step 6: (6, 3, 4)  → 3-4 は非連結 → Union(3,4) ✓  MST: + (3,4,6)

結果: MST重み = 1 + 2 + 3 + 6 = 12
```

---

## 7. 応用パターン

### 連結成分の数（グラフ）

```python
def count_components(n: int, edges: list) -> int:
    """グラフの連結成分数"""
    uf = UnionFind(n)
    for u, v in edges:
        uf.union(u, v)
    return uf.count

print(count_components(5, [(0,1), (2,3)]))  # 3 ({0,1}, {2,3}, {4})
```

### 友達の友達（SNSグラフ）

```python
def friend_groups(n: int, friendships: list) -> list:
    """友達グループの一覧を返す"""
    uf = UnionFind(n)
    for a, b in friendships:
        uf.union(a, b)

    groups = {}
    for i in range(n):
        root = uf.find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    return list(groups.values())

friends = [(0,1), (1,2), (3,4)]
print(friend_groups(6, friends))
# [[0, 1, 2], [3, 4], [5]]
```

### 島の数（グリッド問題）

```python
def num_islands(grid: list) -> int:
    """2Dグリッドの島の数をUnion-Findで求める"""
    if not grid:
        return 0

    rows, cols = len(grid), len(grid[0])
    uf = UnionFind(rows * cols)
    water = 0

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '0':
                water += 1
                continue

            # 右と下の隣接セルとUnion
            for dr, dc in [(0, 1), (1, 0)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < rows and 0 <= nc < cols
                    and grid[nr][nc] == '1'):
                    uf.union(r * cols + c, nr * cols + nc)

    return uf.count - water

grid = [
    ['1','1','0','0','0'],
    ['1','1','0','0','0'],
    ['0','0','1','0','0'],
    ['0','0','0','1','1'],
]
print(num_islands(grid))  # 3
```

### 冗長な接続の検出

```python
def find_redundant_connection(edges: list) -> tuple:
    """木に1本余分な辺が追加されたグラフで、その余分な辺を見つける
    LeetCode 684: Redundant Connection
    """
    n = len(edges)
    uf = UnionFind(n + 1)

    for u, v in edges:
        if uf.connected(u, v):
            return (u, v)  # この辺がサイクルを作る＝冗長
        uf.union(u, v)

    return None

edges = [(1,2), (1,3), (2,3)]
print(find_redundant_connection(edges))  # (2, 3)

edges = [(1,2), (2,3), (3,4), (1,4), (1,5)]
print(find_redundant_connection(edges))  # (1, 4)
```

### 最大辺の最小化（ボトルネック最短路）

```python
def min_bottleneck_path(n: int, edges: list, s: int, t: int) -> int:
    """s から t へのパスにおいて、最大辺の重みを最小化する
    Kruskal法的なアプローチ: 辺を重みの小さい順に追加し、
    s-t が連結になった時点の辺の重みが答え
    """
    edges.sort(key=lambda e: e[2])  # 重みでソート
    uf = UnionFind(n)

    for u, v, w in edges:
        uf.union(u, v)
        if uf.connected(s, t):
            return w

    return -1  # s-t が到達不能

edges = [(0, 1, 3), (0, 2, 5), (1, 2, 1), (1, 3, 4), (2, 3, 2)]
print(min_bottleneck_path(4, edges, 0, 3))  # 3 (0→1→3, 最大辺=max(3,4)=4 ではなく 0→2→3 の max(5,2)=5 でもなく、0→1→2→3 の max(3,1,2)=3)
```

### 動的連結性とオフラインクエリ

```python
def process_connectivity_queries(n: int, operations: list) -> list:
    """オフラインで連結性クエリを処理する
    operations: [('union', u, v), ('query', u, v), ...]
    """
    uf = UnionFind(n)
    results = []

    for op in operations:
        if op[0] == 'union':
            _, u, v = op
            uf.union(u, v)
        elif op[0] == 'query':
            _, u, v = op
            results.append(uf.connected(u, v))
        elif op[0] == 'size':
            _, u = op[0], op[1]
            results.append(uf.get_size(op[1]))

    return results

ops = [
    ('union', 0, 1),
    ('union', 2, 3),
    ('query', 0, 3),   # False
    ('union', 1, 2),
    ('query', 0, 3),   # True
    ('size', 0, None),  # 4
]
# 結果: [False, True, 4]
```

### 等価クラス分け（文字列の等価判定）

```python
def equivalent_strings(pairs: list, s1: str, s2: str) -> bool:
    """文字の等価関係に基づいて2つの文字列が等価か判定
    LeetCode 839的なアプローチ
    pairs: [(a, b), ...] は 文字a と文字b が等価であることを意味
    """
    uf = UnionFind(26)  # a-z

    for a, b in pairs:
        uf.union(ord(a) - ord('a'), ord(b) - ord('a'))

    if len(s1) != len(s2):
        return False

    for c1, c2 in zip(s1, s2):
        if not uf.connected(ord(c1) - ord('a'), ord(c2) - ord('a')):
            return False

    return True

# 'a' と 'b' が等価, 'c' と 'd' が等価
pairs = [('a', 'b'), ('c', 'd')]
print(equivalent_strings(pairs, "abc", "bac"))  # True
print(equivalent_strings(pairs, "abc", "bae"))  # False
```

---

## 8. 最適化の効果比較表

| 最適化 | Find | Union | 備考 |
|:---|:---|:---|:---|
| ナイーブ（配列） | O(1) | O(n) | Quick-Find |
| ナイーブ（木） | O(n) | O(n) | 最悪ケース |
| 経路圧縮のみ | 償却 O(log n) | O(log n) | 大幅改善 |
| ランク付き合併のみ | O(log n) | O(log n) | 木の高さ制限 |
| 両方（最適） | 償却 O(α(n)) | 償却 O(α(n)) | 実質 O(1) |

### Quick-Find と Quick-Union

Union-Find の基本的な実装方式として Quick-Find と Quick-Union がある。

```python
class QuickFind:
    """Quick-Find: Find は O(1) だが Union が O(n)"""

    def __init__(self, n: int):
        self.id = list(range(n))  # 各要素の所属グループID
        self.n = n

    def find(self, x: int) -> int:
        return self.id[x]  # O(1)

    def union(self, x: int, y: int) -> None:
        px, py = self.id[x], self.id[y]
        if px == py:
            return
        # px に属する全要素を py に変更 → O(n)
        for i in range(self.n):
            if self.id[i] == px:
                self.id[i] = py

class QuickUnion:
    """Quick-Union: 単純な木構造（最適化なし）"""

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            x = self.parent[x]  # O(n) 最悪
        return x

    def union(self, x: int, y: int) -> None:
        self.parent[self.find(x)] = self.find(y)  # O(n) 最悪
```

## Union-Find vs 他の手法

| 手法 | 連結判定 | 合併 | 全成分列挙 | 用途 |
|:---|:---|:---|:---|:---|
| Union-Find | O(α(n)) | O(α(n)) | O(n) | 動的連結性 |
| BFS/DFS | O(V+E) | - | O(V+E) | 静的グラフ |
| 隣接行列 | O(1) | O(n) | O(n^2) | 密グラフ |

### パフォーマンスベンチマーク

```python
import time

def benchmark_union_find(n: int, ops: int):
    """Union-Find の性能を計測"""
    import random

    # Union-Find（最適化あり）
    uf_good = UnionFind(n)
    start = time.time()
    for _ in range(ops):
        x, y = random.randint(0, n-1), random.randint(0, n-1)
        uf_good.union(x, y)
    for _ in range(ops):
        x, y = random.randint(0, n-1), random.randint(0, n-1)
        uf_good.connected(x, y)
    good_time = time.time() - start

    print(f"n={n}, ops={ops}")
    print(f"  最適化あり: {good_time:.3f}秒")

# benchmark_union_find(1_000_000, 2_000_000)
# 結果例: n=1000000, ops=2000000 → 最適化あり: 1.2秒
```

---

## 9. 重み付き Union-Find

重み付き Union-Find は、要素間の相対的な重み（距離・差分）を管理するデータ構造である。「AとBの差は5」「BとCの差は3」→「AとCの差は8」のような推移的な関係を効率的に管理できる。

```python
class WeightedUnionFind:
    """重み付きUnion-Find
    要素間の相対的な重み（距離/差分）を管理
    """
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.weight = [0] * n  # 親との重みの差

    def find(self, x):
        if self.parent[x] != x:
            root = self.find(self.parent[x])
            self.weight[x] += self.weight[self.parent[x]]
            self.parent[x] = root
        return self.parent[x]

    def get_weight(self, x):
        """根からxまでの累積重み"""
        self.find(x)  # 経路圧縮
        return self.weight[x]

    def diff(self, x, y):
        """weight(y) - weight(x)"""
        if self.find(x) != self.find(y):
            return None  # 異なる集合
        return self.get_weight(y) - self.get_weight(x)

    def union(self, x, y, w):
        """weight(y) - weight(x) = w という関係を追加
        Returns: True if successfully merged, False if contradiction
        """
        px, py = self.find(x), self.find(y)
        if px == py:
            # 既に同じ集合 → 矛盾チェック
            return self.diff(x, y) == w

        w = w + self.weight[x] - self.weight[y]

        if self.rank[px] < self.rank[py]:
            self.parent[px] = py
            self.weight[px] = -w
        else:
            self.parent[py] = px
            self.weight[py] = w
            if self.rank[px] == self.rank[py]:
                self.rank[px] += 1

        return True

# 使用例: A - B = 3, B - C = 5 → A - C = 8
wuf = WeightedUnionFind(3)
wuf.union(0, 1, 3)  # weight[1] - weight[0] = 3
wuf.union(1, 2, 5)  # weight[2] - weight[1] = 5
print(wuf.diff(0, 2))  # 8 (weight[2] - weight[0] = 3+5)
```

### 重み付き Union-Find の応用例: 人の身長差

```python
def solve_height_differences(n: int, relations: list, queries: list) -> list:
    """身長差の問題
    relations: [(i, j, diff), ...] → 人jは人iより diff cm 高い
    queries: [(i, j), ...] → 人jは人iより何cm高いか？
    """
    wuf = WeightedUnionFind(n)

    for i, j, diff in relations:
        wuf.union(i, j, diff)

    results = []
    for i, j in queries:
        d = wuf.diff(i, j)
        if d is None:
            results.append("不明")
        else:
            results.append(f"{d}cm")

    return results

# 人0, 1, 2, 3, 4
# 人1は人0より10cm高い
# 人2は人1より5cm高い
# 人4は人3より8cm高い
relations = [(0, 1, 10), (1, 2, 5), (3, 4, 8)]
queries = [(0, 2), (0, 4), (3, 4)]
print(solve_height_differences(5, relations, queries))
# ['15cm', '不明', '8cm']
```

### 重み付き Union-Find の応用例: オンラインジャッジの相対評価

```python
def relative_scoring(n: int, comparisons: list) -> list:
    """相対評価の矛盾検出
    comparisons: [(i, j, diff), ...] → スコアj - スコアi = diff
    矛盾があるペアのインデックスを返す
    """
    wuf = WeightedUnionFind(n)
    contradictions = []

    for idx, (i, j, diff) in enumerate(comparisons):
        if not wuf.union(i, j, diff):
            # 矛盾を検出
            actual_diff = wuf.diff(i, j)
            contradictions.append({
                'index': idx,
                'claimed': diff,
                'actual': actual_diff,
                'pair': (i, j)
            })

    return contradictions

comparisons = [
    (0, 1, 3),   # スコア1 - スコア0 = 3
    (1, 2, 5),   # スコア2 - スコア1 = 5
    (0, 2, 7),   # スコア2 - スコア0 = 7 ← 3+5=8 と矛盾!
]
result = relative_scoring(3, comparisons)
print(result)
# [{'index': 2, 'claimed': 7, 'actual': 8, 'pair': (0, 2)}]
```

---

## 10. 永続 Union-Find

永続データ構造版の Union-Find は、過去の任意の状態にアクセスできる。タイムトラベルクエリに対応する。

```python
class PersistentUnionFind:
    """永続Union-Find（簡易版: ロールバック対応）
    操作の取り消し（ロールバック）が可能
    注意: 経路圧縮を使わない（ランク付き合併のみ）
    """

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.history = []  # (操作前の状態を記録)

    def find(self, x: int) -> int:
        """経路圧縮なしの Find（永続性のため）"""
        while self.parent[x] != x:
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        px, py = self.find(x), self.find(y)
        if px == py:
            self.history.append(None)  # 何もしなかった記録
            return False

        if self.rank[px] < self.rank[py]:
            px, py = py, px

        # 操作前の状態を保存
        self.history.append((py, self.parent[py], self.rank[px]))

        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

        return True

    def rollback(self):
        """直前の union 操作を取り消す"""
        if not self.history:
            return

        record = self.history.pop()
        if record is None:
            return  # 何もしなかった操作

        py, old_parent, old_rank_px = record
        px = self.parent[py]
        self.parent[py] = old_parent
        self.rank[px] = old_rank_px

    def save(self) -> int:
        """現在の状態のスナップショットID（履歴の長さ）"""
        return len(self.history)

    def restore(self, snapshot: int):
        """指定したスナップショットまでロールバック"""
        while len(self.history) > snapshot:
            self.rollback()

# 使用例
puf = PersistentUnionFind(5)
snap0 = puf.save()

puf.union(0, 1)
puf.union(2, 3)
snap1 = puf.save()

puf.union(0, 2)
print(puf.find(0) == puf.find(3))  # True

puf.restore(snap1)
print(puf.find(0) == puf.find(3))  # False（ロールバック後）

puf.restore(snap0)
print(puf.find(0) == puf.find(1))  # False（完全にロールバック）
```

---

## 11. Union-Find の実務応用

### ネットワーク障害検出

```python
def detect_network_partitions(n_servers: int, connections: list,
                               failures: list) -> dict:
    """ネットワーク障害時のパーティション検出
    connections: [(server_a, server_b), ...] 全接続リスト
    failures: [(server_a, server_b), ...] 障害が発生した接続
    """
    failure_set = set((min(a,b), max(a,b)) for a, b in failures)
    uf = UnionFind(n_servers)

    # 障害のない接続のみでUnion
    for a, b in connections:
        key = (min(a,b), max(a,b))
        if key not in failure_set:
            uf.union(a, b)

    partitions = uf.get_groups()
    result = {
        'num_partitions': len(partitions),
        'partitions': list(partitions.values()),
        'isolated_servers': [g[0] for g in partitions.values() if len(g) == 1],
        'largest_partition': max(len(g) for g in partitions.values()),
    }
    return result

connections = [(0,1), (1,2), (2,3), (3,4), (0,4), (2,5), (5,6)]
failures = [(2,3), (2,5)]
result = detect_network_partitions(7, connections, failures)
print(f"パーティション数: {result['num_partitions']}")
print(f"パーティション: {result['partitions']}")
print(f"孤立サーバ: {result['isolated_servers']}")
```

### 画像の連結成分ラベリング

```python
def label_connected_components(image: list) -> list:
    """二値画像の連結成分ラベリング
    image: 2D配列（1=前景, 0=背景）
    返り値: 各ピクセルのラベル（0=背景）
    """
    rows, cols = len(image), len(image[0])
    uf = UnionFind(rows * cols)

    # 4連結で隣接する前景ピクセルを Union
    for r in range(rows):
        for c in range(cols):
            if image[r][c] == 0:
                continue
            for dr, dc in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and image[nr][nc] == 1:
                    uf.union(r * cols + c, nr * cols + nc)

    # ラベルの割り当て
    label_map = {}
    label_counter = 1
    labels = [[0] * cols for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            if image[r][c] == 0:
                continue
            root = uf.find(r * cols + c)
            if root not in label_map:
                label_map[root] = label_counter
                label_counter += 1
            labels[r][c] = label_map[root]

    return labels

image = [
    [1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1],
    [0, 0, 0, 1, 0],
    [1, 1, 0, 0, 0],
]
labels = label_connected_components(image)
for row in labels:
    print(row)
# [1, 1, 0, 0, 2]
# [1, 0, 0, 2, 2]
# [0, 0, 0, 2, 0]
# [3, 3, 0, 0, 0]
```

### クラスタリングへの応用

```python
def single_linkage_clustering(points: list, k: int) -> list:
    """単連結クラスタリング
    points: [(x, y), ...]
    k: 目標クラスタ数
    返り値: 各点のクラスタラベル
    """
    import math
    n = len(points)

    # 全ペア間の距離を計算
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            dist = math.sqrt(dx * dx + dy * dy)
            edges.append((dist, i, j))

    edges.sort()  # 距離でソート

    uf = UnionFind(n)

    # クラスタ数が k になるまで統合
    for dist, i, j in edges:
        if uf.count <= k:
            break
        uf.union(i, j)

    # クラスタラベルの割り当て
    label_map = {}
    label_counter = 0
    labels = []
    for i in range(n):
        root = uf.find(i)
        if root not in label_map:
            label_map[root] = label_counter
            label_counter += 1
        labels.append(label_map[root])

    return labels

points = [(0,0), (1,1), (0,1), (10,10), (11,11), (10,11)]
labels = single_linkage_clustering(points, 2)
print(labels)  # [0, 0, 0, 1, 1, 1]
```

---

## 12. 高度なバリエーション

### 部分永続 Union-Find

```python
class PartiallyPersistentUnionFind:
    """部分永続 Union-Find
    過去の任意の時刻での connected クエリに O(log n) で回答
    """

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.time = [float('inf')] * n  # 根でなくなった時刻
        self.size_history = [[(0, 1)] for _ in range(n)]  # (時刻, サイズ)
        self.now = 0

    def find(self, x: int, t: int = None) -> int:
        """時刻 t での代表元"""
        if t is None:
            t = self.now
        while self.time[x] <= t:
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        self.now += 1
        x = self.find(x)
        y = self.find(y)
        if x == y:
            return False

        if self.rank[x] < self.rank[y]:
            x, y = y, x

        self.parent[y] = x
        self.time[y] = self.now

        new_size = self.size_history[x][-1][1] + self.size_history[y][-1][1]
        self.size_history[x].append((self.now, new_size))

        if self.rank[x] == self.rank[y]:
            self.rank[x] += 1

        return True

    def connected(self, x: int, y: int, t: int = None) -> bool:
        return self.find(x, t) == self.find(y, t)
```

### Union-Find with Undo（巻き戻し対応）

分割統治法と組み合わせて使われることが多い。オフラインの辺の追加・削除を扱える。

```python
class UnionFindWithUndo:
    """Undo 対応 Union-Find
    経路圧縮を使わず、ランク付き合併のみで O(log n)
    undo 操作でスタックに積んだ操作を巻き戻す
    """

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.stack = []  # 操作の記録スタック

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        px, py = self.find(x), self.find(y)
        if px == py:
            self.stack.append(None)
            return False

        if self.rank[px] < self.rank[py]:
            px, py = py, px

        self.stack.append((py, self.rank[px]))
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

        return True

    def undo(self):
        record = self.stack.pop()
        if record is None:
            return
        py, old_rank_px = record
        px = self.parent[py]
        self.parent[py] = py
        self.rank[px] = old_rank_px
```

---

## 13. 逆アッカーマン関数の理論

逆アッカーマン関数 α(n) は Union-Find の計算量解析で登場する関数である。

```
アッカーマン関数 A(m, n):
  A(0, n) = n + 1
  A(m, 0) = A(m-1, 1)
  A(m, n) = A(m-1, A(m, n-1))

急激に成長する:
  A(0, 0) = 1
  A(1, 1) = 3
  A(2, 2) = 7
  A(3, 3) = 61
  A(4, 4) = 2^(2^(2^...)) - 3  (65536回のべき乗タワー)

逆アッカーマン関数 α(n):
  α(n) = min{k : A(k, k) ≥ n}

  α(1) = 0
  α(4) = 2
  α(65536) = 3
  α(2^65536) = 4
  α(A(4,4)) = 5

  実用上の全ての入力 (n ≤ 10^80) に対して α(n) ≤ 4

→ O(α(n)) は実質的に O(1)
```

### Tarjan の証明の概要

1975年にTarjanが証明した定理:

> 経路圧縮とランク付き合併を併用した Union-Find に対して、m 回の操作（Find と Union の混合）の合計計算量は O(m * α(n)) である。

さらに1984年にTarjan と van Leeuwen が、これが漸近的に最適であることを証明した。つまり、Union-Find の問題に対して O(m * α(n)) より良いアルゴリズムは（ある計算モデルにおいて）存在しない。

---

## 14. アンチパターン

### アンチパターン1: 経路圧縮・ランク付き合併を省略

```python
# BAD: ナイーブなUnion-Find → Find が O(n) に退化
class BadUnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            x = self.parent[x]  # 経路圧縮なし!
        return x

    def union(self, x, y):
        self.parent[self.find(x)] = self.find(y)  # ランク考慮なし!

# GOOD: 両方の最適化を適用
class GoodUnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 経路圧縮
        return self.parent[x]
    # ... ランク付き合併も実装
```

### アンチパターン2: 毎回 O(n) で連結判定

```python
# BAD: 辺の追加のたびに BFS/DFS で連結判定 → O(E * (V+E))
for u, v in edges:
    graph[u].append(v)
    if bfs_connected(graph, 0, target):  # O(V+E) が毎回
        ...

# GOOD: Union-Find で O(E * α(n)) ≈ O(E)
uf = UnionFind(n)
for u, v in edges:
    uf.union(u, v)
    if uf.connected(0, target):  # O(α(n)) ≈ O(1)
        ...
```

### アンチパターン3: Union-Find で辺の削除をしようとする

```python
# BAD: Union-Find で辺の削除は効率的にできない
# Union-Find は「集合の統合」に特化しており、分割はサポートしない

# GOOD: 辺の削除が必要な場合のアプローチ
# 方法1: オフライン処理（逆順に辺を追加）
# 方法2: Link-Cut Tree を使う
# 方法3: 永続Union-Findのロールバック（操作順の逆順のみ）
```

### アンチパターン4: 再帰の深さを考慮しない

```python
# BAD: n が大きい場合に再帰的な Find でスタックオーバーフロー
import sys
# sys.setrecursionlimit(10**6) を設定しても危険

# GOOD: 反復版の Find を使う
def find_iterative(self, x):
    root = x
    while self.parent[root] != root:
        root = self.parent[root]
    while self.parent[x] != root:
        next_x = self.parent[x]
        self.parent[x] = root
        x = next_x
    return root
```

### アンチパターン5: 0-indexed と 1-indexed の混在

```python
# BAD: 問題の頂点番号が1-indexedなのにUnion-Findを0-indexedで作成
n = int(input())
uf = UnionFind(n)  # 0 ~ n-1
for _ in range(m):
    u, v = map(int, input().split())
    uf.union(u, v)  # u, v は 1 ~ n → IndexError!

# GOOD: サイズを n+1 にするか、入力時に -1 する
uf = UnionFind(n + 1)  # 0 ~ n（0は未使用）
for _ in range(m):
    u, v = map(int, input().split())
    uf.union(u, v)  # 1-indexed のまま使える
```

---

## 15. 競技プログラミングでの Union-Find テンプレート

```python
import sys
input = sys.stdin.readline

class UnionFind:
    """競技プログラミング用 Union-Find テンプレート"""
    __slots__ = ['parent', 'rank', 'size', 'count']

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n
        self.count = n

    def find(self, x: int) -> int:
        # 反復版（スタックオーバーフロー回避）
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, x: int, y: int) -> bool:
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        self.size[px] += self.size[py]
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        self.count -= 1
        return True

    def connected(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)

    def get_size(self, x: int) -> int:
        return self.size[self.find(x)]

# AtCoder ABC 典型問題の解法テンプレート
def solve():
    N, M = map(int, input().split())
    uf = UnionFind(N)

    for _ in range(M):
        a, b = map(int, input().split())
        a -= 1; b -= 1  # 0-indexed
        uf.union(a, b)

    # 連結成分数
    print(uf.count)

    # 最大の連結成分のサイズ
    max_size = max(uf.get_size(i) for i in range(N))
    print(max_size)
```

---

## 16. FAQ

### Q1: 逆アッカーマン関数 α(n) とは何か？

**A:** α(n) はアッカーマン関数の逆関数で、実用上の全ての入力（宇宙の原子数 ~10^80 まで）に対して 5 以下。つまり O(α(n)) は実質的に O(1)。経路圧縮とランク付き合併を併用した場合に得られる計算量。1975年にTarjanが証明した。

### Q2: Union-Find で要素の削除はできるか？

**A:** 標準的な Union-Find では要素の削除やグループの分割（Split）は効率的にできない。必要な場合は (1) 削除済みフラグで論理削除し新しい要素で代替、(2) Link-Cut Tree などの高度なデータ構造を使う、(3) 全体を再構築する、のいずれかを検討する。

### Q3: Union-Find はオンライン問題に強いか？

**A:** はい。辺が動的に追加される状況（オンラインクエリ）で連結性を管理するのに最適。BFS/DFS は辺が追加されるたびに再計算が必要だが、Union-Find は O(α(n)) で逐次的に処理できる。

### Q4: ランク付き合併とサイズ付き合併のどちらを使うべきか？

**A:** 理論的にはどちらも同じ計算量 O(α(n)) を達成する。実用上はサイズ付き合併が推奨される。理由は、サイズ情報は「この集合の要素数は？」という追加クエリに直接使えるため。ランクは木の高さの上界であり、直接的な意味を持たない。

### Q5: Union-Find の空間計算量は？

**A:** O(n)。parent 配列、rank（またはsize）配列が必要で、それぞれ n 要素。追加のデータ構造（重み付き、永続など）を使う場合はその分増加する。

### Q6: グラフの辺が削除される場合はどうするか？

**A:** Union-Find は辺の削除に対応していない。対処法は: (1) オフライン処理で辺の追加の逆順に処理する（逆から見れば辺の削除は辺の追加）、(2) Link-Cut Tree（Euler Tour Tree）を使う（オンラインで辺の追加・削除に対応、O(log n) per operation）、(3) 分割統治+Union-Find with Undo。

### Q7: Union-Find は並列処理に適しているか？

**A:** 標準的な Union-Find はスレッドセーフではない。並列版としては Wait-Free Union-Find（CAS操作ベース）が研究されている。実用上はロックを使うか、各スレッドで部分的な Union-Find を作り、最後にマージする方法がある。

---

## 17. まとめ

| 項目 | 要点 |
|:---|:---|
| 基本操作 | Find（代表元の取得）と Union（集合の合併） |
| 経路圧縮 | Find 時に全ノードを根に直接接続 → 木を平坦化 |
| ランク付き合併 | 低い木を高い木の子に → 木の高さを制限 |
| 計算量 | 両方の最適化で O(α(n)) ≈ 実質 O(1) |
| Kruskal応用 | サイクル判定に使用。O(E log E) で MST |
| 重み付き拡張 | 要素間の相対重みを管理 |
| 永続版 | ロールバック対応（経路圧縮なし） |
| 実務応用 | ネットワーク管理、画像処理、クラスタリング |

---

## 次に読むべきガイド

- [グラフ走査](../02-algorithms/02-graph-traversal.md) -- 連結成分の別アプローチ（BFS/DFS）
- [貪欲法](../02-algorithms/05-greedy.md) -- Kruskal法の詳細
- [セグメント木](./01-segment-tree.md) -- 別の高度データ構造

---

## 参考文献

1. Cormen, T. H. et al. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. -- 第19章
2. Tarjan, R. E. (1975). "Efficiency of a Good But Not Linear Set Union Algorithm." *JACM*.
3. Tarjan, R. E. & van Leeuwen, J. (1984). "Worst-case Analysis of Set Union Algorithms." *JACM*.
4. Sedgewick, R. & Wayne, K. (2011). *Algorithms* (4th ed.). Addison-Wesley. -- 1.5 Union-Find
5. Galil, Z. & Italiano, G. F. (1991). "Data Structures and Algorithms for Disjoint Set Union Problems." *ACM Computing Surveys*.
6. Alstrup, S. et al. (2014). "Union-Find with Constant Time Deletions." *ACM Transactions on Algorithms*.
