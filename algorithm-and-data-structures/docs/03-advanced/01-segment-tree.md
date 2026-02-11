# セグメント木（Segment Tree）

> 区間クエリと点更新を O(log n) で処理する木構造を、基本実装・遅延伝播・BITを通じて体系的に理解する

## この章で学ぶこと

1. **セグメント木の構造と基本操作**（構築・区間クエリ・点更新）を O(log n) で実装できる
2. **遅延伝播（Lazy Propagation）**で区間更新も O(log n) に拡張できる
3. **BIT（Binary Indexed Tree / Fenwick Tree）**との使い分けを理解し、適材適所で選択できる

---

## 1. セグメント木の概念

```
配列: [2, 1, 5, 3, 4, 2, 1, 6]

セグメント木（区間和）:
                  [24]              ← 全体の和
               /        \
          [11]             [13]     ← 前半/後半の和
         /    \           /    \
      [3]     [8]     [6]     [7]
     / \     / \     / \     / \
   [2] [1] [5] [3] [4] [2] [1] [6]  ← 葉 = 元の配列

区間 [1, 5) の和を求める:
  → [1] + [5,3] + [4] = 1 + 8 + 4 = 13
  → 3ノードの参照で回答（O(log n)）

ナイーブな配列:
  → 1 + 5 + 3 + 4 = 13
  → 4要素を走査（O(n)）
```

---

## 2. 基本実装（区間和）

```python
class SegmentTree:
    """セグメント木（区間和クエリ + 点更新）"""

    def __init__(self, data: list):
        self.n = len(data)
        self.tree = [0] * (4 * self.n)  # 十分なサイズ
        self._build(data, 1, 0, self.n - 1)

    def _build(self, data, node, start, end):
        """O(n) で構築"""
        if start == end:
            self.tree[node] = data[start]
            return
        mid = (start + end) // 2
        self._build(data, 2 * node, start, mid)
        self._build(data, 2 * node + 1, mid + 1, end)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def update(self, idx: int, val: int):
        """点更新 - O(log n)"""
        self._update(1, 0, self.n - 1, idx, val)

    def _update(self, node, start, end, idx, val):
        if start == end:
            self.tree[node] = val
            return
        mid = (start + end) // 2
        if idx <= mid:
            self._update(2 * node, start, mid, idx, val)
        else:
            self._update(2 * node + 1, mid + 1, end, idx, val)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def query(self, l: int, r: int) -> int:
        """区間 [l, r] の和を返す - O(log n)"""
        return self._query(1, 0, self.n - 1, l, r)

    def _query(self, node, start, end, l, r):
        if r < start or end < l:
            return 0  # 範囲外
        if l <= start and end <= r:
            return self.tree[node]  # 完全に含まれる
        mid = (start + end) // 2
        left_sum = self._query(2 * node, start, mid, l, r)
        right_sum = self._query(2 * node + 1, mid + 1, end, l, r)
        return left_sum + right_sum

# 使用例
data = [2, 1, 5, 3, 4, 2, 1, 6]
st = SegmentTree(data)
print(st.query(1, 4))  # 13 (1+5+3+4)
print(st.query(0, 7))  # 24 (全体の和)

st.update(2, 10)        # data[2] = 5 → 10
print(st.query(1, 4))  # 18 (1+10+3+4)
```

---

## 3. 区間最小値クエリ（RMQ）

```python
class SegmentTreeMin:
    """セグメント木（区間最小値クエリ）"""

    def __init__(self, data: list):
        self.n = len(data)
        self.tree = [float('inf')] * (4 * self.n)
        self._build(data, 1, 0, self.n - 1)

    def _build(self, data, node, start, end):
        if start == end:
            self.tree[node] = data[start]
            return
        mid = (start + end) // 2
        self._build(data, 2 * node, start, mid)
        self._build(data, 2 * node + 1, mid + 1, end)
        self.tree[node] = min(self.tree[2 * node], self.tree[2 * node + 1])

    def query(self, l: int, r: int) -> int:
        return self._query(1, 0, self.n - 1, l, r)

    def _query(self, node, start, end, l, r):
        if r < start or end < l:
            return float('inf')
        if l <= start and end <= r:
            return self.tree[node]
        mid = (start + end) // 2
        return min(
            self._query(2 * node, start, mid, l, r),
            self._query(2 * node + 1, mid + 1, end, l, r)
        )

    def update(self, idx: int, val: int):
        self._update(1, 0, self.n - 1, idx, val)

    def _update(self, node, start, end, idx, val):
        if start == end:
            self.tree[node] = val
            return
        mid = (start + end) // 2
        if idx <= mid:
            self._update(2 * node, start, mid, idx, val)
        else:
            self._update(2 * node + 1, mid + 1, end, idx, val)
        self.tree[node] = min(self.tree[2 * node], self.tree[2 * node + 1])

data = [5, 2, 8, 1, 9, 3, 7, 4]
st_min = SegmentTreeMin(data)
print(st_min.query(0, 3))  # 1
print(st_min.query(4, 7))  # 3
```

---

## 4. 遅延伝播（Lazy Propagation）

区間更新（例: 区間 [l,r] の全要素に v を加算）を O(log n) で処理する。

```
区間加算の遅延伝播:

加算前:        [24]
              /    \
          [11]      [13]
         /  \      /   \
       [3]  [8] [6]   [7]

区間 [2,5] に +3 を加算:
  → 影響ノードの lazy に +3 を記録
  → 実際の伝播は必要時まで遅延

          [24+12=36]
            /      \
      [11+6=17]   [13+6=19]
       /    \      /     \
     [3]  [8+6] [6+6]  [7]
           lazy   lazy
           =+3   =+3
```

```python
class LazySegmentTree:
    """遅延伝播付きセグメント木（区間加算 + 区間和クエリ）"""

    def __init__(self, data: list):
        self.n = len(data)
        self.tree = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        self._build(data, 1, 0, self.n - 1)

    def _build(self, data, node, start, end):
        if start == end:
            self.tree[node] = data[start]
            return
        mid = (start + end) // 2
        self._build(data, 2 * node, start, mid)
        self._build(data, 2 * node + 1, mid + 1, end)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def _push_down(self, node, start, end):
        """遅延値を子に伝播"""
        if self.lazy[node] != 0:
            mid = (start + end) // 2
            left_len = mid - start + 1
            right_len = end - mid

            self.tree[2 * node] += self.lazy[node] * left_len
            self.tree[2 * node + 1] += self.lazy[node] * right_len

            self.lazy[2 * node] += self.lazy[node]
            self.lazy[2 * node + 1] += self.lazy[node]

            self.lazy[node] = 0

    def range_update(self, l: int, r: int, val: int):
        """区間 [l, r] に val を加算 - O(log n)"""
        self._range_update(1, 0, self.n - 1, l, r, val)

    def _range_update(self, node, start, end, l, r, val):
        if r < start or end < l:
            return
        if l <= start and end <= r:
            self.tree[node] += val * (end - start + 1)
            self.lazy[node] += val
            return
        self._push_down(node, start, end)
        mid = (start + end) // 2
        self._range_update(2 * node, start, mid, l, r, val)
        self._range_update(2 * node + 1, mid + 1, end, l, r, val)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def query(self, l: int, r: int) -> int:
        """区間 [l, r] の和 - O(log n)"""
        return self._query(1, 0, self.n - 1, l, r)

    def _query(self, node, start, end, l, r):
        if r < start or end < l:
            return 0
        if l <= start and end <= r:
            return self.tree[node]
        self._push_down(node, start, end)
        mid = (start + end) // 2
        return (self._query(2 * node, start, mid, l, r) +
                self._query(2 * node + 1, mid + 1, end, l, r))

# 使用例
data = [1, 3, 5, 7, 9, 11]
lst = LazySegmentTree(data)
print(lst.query(1, 3))     # 15 (3+5+7)
lst.range_update(1, 4, 10)  # 区間[1,4]に+10
print(lst.query(1, 3))     # 45 (13+15+17)
```

---

## 5. BIT（Binary Indexed Tree / Fenwick Tree）

接頭辞和の計算と点更新を O(log n) で行う。セグメント木より実装が簡潔でメモリ効率が良い。

```
配列:  [3, 2, 5, 1, 4, 7, 2, 6]
index:  1  2  3  4  5  6  7  8

BIT の構造（1-indexed）:
tree[1] = a[1]           = 3
tree[2] = a[1]+a[2]      = 5
tree[3] = a[3]           = 5
tree[4] = a[1]+...+a[4]  = 11
tree[5] = a[5]           = 4
tree[6] = a[5]+a[6]      = 11
tree[7] = a[7]           = 2
tree[8] = a[1]+...+a[8]  = 30

tree[i] がカバーする範囲 = lowbit(i) = i & (-i) 個
  tree[4]: lowbit(4)=4 → a[1]~a[4]
  tree[6]: lowbit(6)=2 → a[5]~a[6]
  tree[7]: lowbit(7)=1 → a[7]
```

```python
class BIT:
    """Binary Indexed Tree (Fenwick Tree) - 1-indexed"""

    def __init__(self, n: int):
        self.n = n
        self.tree = [0] * (n + 1)

    @classmethod
    def from_array(cls, data: list):
        """配列から O(n) で構築"""
        bit = cls(len(data))
        for i, val in enumerate(data):
            bit.update(i + 1, val)  # 1-indexed
        return bit

    def update(self, i: int, delta: int):
        """a[i] に delta を加算 - O(log n)"""
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)  # 次のノード

    def prefix_sum(self, i: int) -> int:
        """a[1] + a[2] + ... + a[i] - O(log n)"""
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)  # 親ノード
        return s

    def range_sum(self, l: int, r: int) -> int:
        """a[l] + ... + a[r] - O(log n)"""
        return self.prefix_sum(r) - self.prefix_sum(l - 1)

# 使用例
data = [3, 2, 5, 1, 4, 7, 2, 6]
bit = BIT.from_array(data)
print(bit.range_sum(2, 5))  # 12 (2+5+1+4)
bit.update(3, 3)             # data[3] += 3 (5→8)
print(bit.range_sum(2, 5))  # 15 (2+8+1+4)
```

---

## 6. セグメント木 vs BIT 比較表

| 特性 | セグメント木 | BIT |
|:---|:---|:---|
| 空間計算量 | O(4n) | O(n) |
| 構築 | O(n) | O(n log n) |
| 点更新 | O(log n) | O(log n) |
| 区間クエリ | O(log n) | O(log n) |
| 区間更新 | O(log n)（遅延伝播） | O(log n)（range update BIT） |
| 対応クエリ | 和・最小・最大・GCD等 | 主に和（可逆演算） |
| 実装の複雑さ | やや複雑 | 簡潔 |
| 定数倍 | やや大きい | 小さい |

## 用途別選択ガイド

| 用途 | 推奨 | 理由 |
|:---|:---|:---|
| 区間和 + 点更新 | BIT | 実装簡潔、高速 |
| 区間最小/最大 | セグメント木 | BIT では非対応 |
| 区間加算 + 区間和 | 遅延伝播セグメント木 | 区間更新が必要 |
| 転倒数の計算 | BIT | 座標圧縮+点更新 |
| 2次元区間クエリ | 2D BIT or セグメント木 | 拡張が容易 |

---

## 7. 転倒数の計算（BIT応用）

```python
def count_inversions(arr: list) -> int:
    """転倒数をBITで O(n log n) で計算"""
    # 座標圧縮
    sorted_unique = sorted(set(arr))
    rank = {v: i + 1 for i, v in enumerate(sorted_unique)}

    bit = BIT(len(sorted_unique))
    inversions = 0

    for i in range(len(arr) - 1, -1, -1):
        # arr[i] より小さい要素で、arr[i] より右にあるものの数
        inversions += bit.prefix_sum(rank[arr[i]] - 1)
        bit.update(rank[arr[i]], 1)

    return inversions

data = [5, 3, 2, 4, 1]
print(count_inversions(data))  # 7
```

---

## 8. アンチパターン

### アンチパターン1: 配列サイズの見積もりミス

```python
# BAD: セグメント木のサイズが不足
class BadSegTree:
    def __init__(self, data):
        self.tree = [0] * (2 * len(data))  # 不足!
        # n が 2の冪でない場合、2n では足りない

# GOOD: 4n で十分なサイズを確保
class GoodSegTree:
    def __init__(self, data):
        self.tree = [0] * (4 * len(data))  # 安全
```

### アンチパターン2: 遅延伝播の push_down 忘れ

```python
# BAD: クエリ時に遅延値を伝播しない
def bad_query(self, node, start, end, l, r):
    if l <= start and end <= r:
        return self.tree[node]  # lazy が未反映!
    mid = (start + end) // 2
    # push_down なしで子にアクセス → 不正確な結果
    return (self._query(2*node, start, mid, l, r) +
            self._query(2*node+1, mid+1, end, l, r))

# GOOD: 子にアクセスする前に必ず push_down
def good_query(self, node, start, end, l, r):
    if l <= start and end <= r:
        return self.tree[node]
    self._push_down(node, start, end)  # 遅延値を伝播!
    mid = (start + end) // 2
    return (self._query(2*node, start, mid, l, r) +
            self._query(2*node+1, mid+1, end, l, r))
```

---

## 9. FAQ

### Q1: セグメント木のサイズはなぜ 4n？

**A:** セグメント木は完全二分木として構築され、n が 2 の冪でない場合、葉の数は次の 2 の冪に切り上げられる。ノードの索引付け（1-indexed, 子は 2*node と 2*node+1）を使うと、最大で 4n 程度のノードが必要になる。安全マージンを含めて 4n が定番。

### Q2: BIT で区間最小値は求められるか？

**A:** 標準の BIT では困難。BIT は「接頭辞に関する可逆な二項演算」に適しており、min/max は逆元が存在しないため、区間 [l, r] のクエリに分解できない。区間最小値にはセグメント木を使う。

### Q3: 2次元のセグメント木は？

**A:** 外側のセグメント木の各ノードが内側のセグメント木を持つ「2D セグメント木」が構築可能。計算量は O(log² n)、空間は O(n² log² n)。実用的には 2D BIT のほうが実装が簡潔。

---

## 10. まとめ

| 項目 | 要点 |
|:---|:---|
| セグメント木 | 区間クエリ + 点更新を O(log n) で処理 |
| 遅延伝播 | 区間更新を O(log n) に拡張。push_down が鍵 |
| BIT | 接頭辞和特化。セグメント木より簡潔で高速 |
| 適用場面 | 区間和・最小・最大・GCD の動的クエリ |
| サイズ | セグメント木は 4n、BIT は n+1 |
| 応用 | 転倒数、座標圧縮、2次元クエリ |

---

## 次に読むべきガイド

- [Union-Find](./00-union-find.md) -- 別の高度データ構造
- [文字列アルゴリズム](./02-string-algorithms.md) -- Trie等の木構造応用
- [競技プログラミング](../04-practice/01-competitive-programming.md) -- セグメント木の実践

---

## 参考文献

1. Cormen, T. H. et al. (2022). *Introduction to Algorithms* (4th ed.). MIT Press.
2. Fenwick, P. M. (1994). "A New Data Structure for Cumulative Frequency Tables." *Software: Practice and Experience*.
3. Halim, S. & Halim, F. (2013). *Competitive Programming 3*. -- Chapter 2: Data Structures
4. cp-algorithms. "Segment Tree." https://cp-algorithms.com/data_structures/segment_tree.html
