# セグメント木（Segment Tree）

> 区間クエリと点更新を O(log n) で処理する木構造を、基本実装・遅延伝播・BITを通じて体系的に理解する

## この章で学ぶこと

1. **セグメント木の構造と基本操作**（構築・区間クエリ・点更新）を O(log n) で実装できる
2. **遅延伝播（Lazy Propagation）**で区間更新も O(log n) に拡張できる
3. **BIT（Binary Indexed Tree / Fenwick Tree）**との使い分けを理解し、適材適所で選択できる
4. **抽象セグメント木**で任意のモノイド演算に対応できる
5. **永続セグメント木・マージソートツリー**などの発展的なバリエーションを理解する

---

## 1. セグメント木の概念

セグメント木は、配列に対する「区間クエリ」と「要素の更新」を効率的に処理するための完全二分木である。配列の各要素を葉に配置し、各内部ノードがその子ノードの区間に対する演算結果（和、最小値、最大値、GCDなど）を保持する。

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

### なぜセグメント木が重要か

```
セグメント木が必要な場面:

1. 動的な配列に対する区間クエリ
   → 値の更新が頻繁に行われ、その都度区間の集約値（和、最小値等）を求めたい

2. ナイーブ手法との比較:
   操作           | 配列     | 累積和   | セグメント木
   点更新         | O(1)     | O(n)     | O(log n)
   区間クエリ     | O(n)     | O(1)     | O(log n)

   → 更新とクエリの両方が多い場合にセグメント木が最適

3. 具体的なユースケース:
   - リアルタイムの株価の区間最小/最大クエリ
   - ゲームのスコアランキング（更新+順位クエリ）
   - データベースのrange queryの内部実装
   - 競技プログラミングの区間問題全般
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

### 非再帰版セグメント木（高速版）

再帰を使わない実装は定数倍が小さく、競技プログラミングでは実用上高速である。

```python
class SegmentTreeIterative:
    """非再帰セグメント木（区間和） - 定数倍が小さい"""

    def __init__(self, data: list):
        self.n = len(data)
        self.size = 1
        while self.size < self.n:
            self.size <<= 1
        self.tree = [0] * (2 * self.size)

        # 葉にデータを配置
        for i in range(self.n):
            self.tree[self.size + i] = data[i]

        # ボトムアップで構築
        for i in range(self.size - 1, 0, -1):
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]

    def update(self, idx: int, val: int):
        """点更新 - O(log n)"""
        idx += self.size
        self.tree[idx] = val
        idx >>= 1
        while idx >= 1:
            self.tree[idx] = self.tree[2 * idx] + self.tree[2 * idx + 1]
            idx >>= 1

    def query(self, l: int, r: int) -> int:
        """区間 [l, r) の和 - O(log n)"""
        result = 0
        l += self.size
        r += self.size
        while l < r:
            if l & 1:
                result += self.tree[l]
                l += 1
            if r & 1:
                r -= 1
                result += self.tree[r]
            l >>= 1
            r >>= 1
        return result

# 使用例（半開区間 [l, r) に注意）
data = [2, 1, 5, 3, 4, 2, 1, 6]
st = SegmentTreeIterative(data)
print(st.query(1, 5))  # 13 (1+5+3+4) ← [1, 5) = index 1,2,3,4
print(st.query(0, 8))  # 24 (全体)
```

### C++ 実装（非再帰版）

```cpp
#include <vector>
#include <functional>

template <typename T>
class SegmentTree {
    int n;
    std::vector<T> tree;
    T identity;
    std::function<T(T, T)> op;

public:
    SegmentTree(int n, T identity, std::function<T(T, T)> op)
        : n(n), tree(2 * n, identity), identity(identity), op(op) {}

    SegmentTree(const std::vector<T>& data, T identity, std::function<T(T, T)> op)
        : SegmentTree(data.size(), identity, op) {
        for (int i = 0; i < (int)data.size(); i++)
            tree[n + i] = data[i];
        for (int i = n - 1; i > 0; i--)
            tree[i] = op(tree[2 * i], tree[2 * i + 1]);
    }

    void update(int idx, T val) {
        tree[idx += n] = val;
        for (idx >>= 1; idx >= 1; idx >>= 1)
            tree[idx] = op(tree[2 * idx], tree[2 * idx + 1]);
    }

    T query(int l, int r) {  // [l, r)
        T left_result = identity, right_result = identity;
        for (l += n, r += n; l < r; l >>= 1, r >>= 1) {
            if (l & 1) left_result = op(left_result, tree[l++]);
            if (r & 1) right_result = op(tree[--r], right_result);
        }
        return op(left_result, right_result);
    }
};

// 使用例
// SegmentTree<int> st(data, 0, [](int a, int b){ return a + b; });  // 区間和
// SegmentTree<int> st(data, INT_MAX, [](int a, int b){ return min(a, b); });  // 区間最小
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

### 区間最小値とそのインデックス

```python
class SegmentTreeMinIndex:
    """区間最小値とそのインデックスを返すセグメント木"""

    def __init__(self, data: list):
        self.n = len(data)
        # tree[i] = (value, index) のペア
        self.tree = [(float('inf'), -1)] * (4 * self.n)
        self._build(data, 1, 0, self.n - 1)

    def _merge(self, a, b):
        """小さい方を返す（同値ならインデックスが小さい方）"""
        if a[0] < b[0]:
            return a
        elif a[0] > b[0]:
            return b
        else:
            return a if a[1] <= b[1] else b

    def _build(self, data, node, start, end):
        if start == end:
            self.tree[node] = (data[start], start)
            return
        mid = (start + end) // 2
        self._build(data, 2 * node, start, mid)
        self._build(data, 2 * node + 1, mid + 1, end)
        self.tree[node] = self._merge(self.tree[2 * node], self.tree[2 * node + 1])

    def query(self, l: int, r: int) -> tuple:
        """区間 [l, r] の最小値と位置を返す"""
        return self._query(1, 0, self.n - 1, l, r)

    def _query(self, node, start, end, l, r):
        if r < start or end < l:
            return (float('inf'), -1)
        if l <= start and end <= r:
            return self.tree[node]
        mid = (start + end) // 2
        left = self._query(2 * node, start, mid, l, r)
        right = self._query(2 * node + 1, mid + 1, end, l, r)
        return self._merge(left, right)

data = [5, 2, 8, 1, 9, 3, 7, 4]
st = SegmentTreeMinIndex(data)
val, idx = st.query(0, 7)
print(f"最小値: {val}, 位置: {idx}")  # 最小値: 1, 位置: 3
val, idx = st.query(4, 7)
print(f"最小値: {val}, 位置: {idx}")  # 最小値: 3, 位置: 5
```

---

## 4. 抽象セグメント木（モノイド）

任意の結合的二項演算に対応する汎用的なセグメント木。モノイド（結合律を満たす二項演算 + 単位元）であれば何でもセグメント木に載せられる。

```python
class AbstractSegmentTree:
    """抽象セグメント木 - 任意のモノイド演算に対応
    op: 二項演算 (結合律を満たす)
    e: 単位元 (op(a, e) = op(e, a) = a)
    """

    def __init__(self, data: list, op, e):
        self.n = len(data)
        self.op = op
        self.e = e
        self.size = 1
        while self.size < self.n:
            self.size <<= 1
        self.tree = [e] * (2 * self.size)

        # 葉にデータを配置
        for i in range(self.n):
            self.tree[self.size + i] = data[i]

        # ボトムアップで構築
        for i in range(self.size - 1, 0, -1):
            self.tree[i] = self.op(self.tree[2 * i], self.tree[2 * i + 1])

    def update(self, idx: int, val):
        """点更新"""
        idx += self.size
        self.tree[idx] = val
        idx >>= 1
        while idx >= 1:
            self.tree[idx] = self.op(self.tree[2 * idx], self.tree[2 * idx + 1])
            idx >>= 1

    def query(self, l: int, r: int):
        """区間 [l, r) のクエリ"""
        left_result = self.e
        right_result = self.e
        l += self.size
        r += self.size
        while l < r:
            if l & 1:
                left_result = self.op(left_result, self.tree[l])
                l += 1
            if r & 1:
                r -= 1
                right_result = self.op(self.tree[r], right_result)
            l >>= 1
            r >>= 1
        return self.op(left_result, right_result)

# --- さまざまなモノイドでの使用例 ---

data = [2, 1, 5, 3, 4, 2, 1, 6]

# 区間和 (和, 0)
st_sum = AbstractSegmentTree(data, lambda a, b: a + b, 0)
print(st_sum.query(1, 5))  # 13

# 区間最小値 (min, inf)
st_min = AbstractSegmentTree(data, min, float('inf'))
print(st_min.query(0, 8))  # 1

# 区間最大値 (max, -inf)
st_max = AbstractSegmentTree(data, max, float('-inf'))
print(st_max.query(0, 8))  # 6

# 区間GCD (gcd, 0)
from math import gcd
data_gcd = [12, 18, 24, 36]
st_gcd = AbstractSegmentTree(data_gcd, gcd, 0)
print(st_gcd.query(0, 4))  # 6

# 区間XOR (xor, 0)
st_xor = AbstractSegmentTree(data, lambda a, b: a ^ b, 0)
print(st_xor.query(0, 8))  # 2^1^5^3^4^2^1^6 = 0

# 区間積 (乗算, 1) + MOD
MOD = 10**9 + 7
st_prod = AbstractSegmentTree(data, lambda a, b: (a * b) % MOD, 1)
print(st_prod.query(0, 4))  # 2*1*5*3 = 30
```

---

## 5. 遅延伝播（Lazy Propagation）

区間更新（例: 区間 [l,r] の全要素に v を加算）を O(log n) で処理する。遅延伝播の核心は「必要になるまで更新を後回しにする」という考え方。

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

### 遅延伝播：区間代入 + 区間和クエリ

```python
class LazySegmentTreeAssign:
    """遅延伝播付きセグメント木（区間代入 + 区間和クエリ）
    区間 [l, r] の全要素を val に置き換える
    """

    def __init__(self, data: list):
        self.n = len(data)
        self.tree = [0] * (4 * self.n)
        self.lazy = [None] * (4 * self.n)  # None = 未伝播
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
        if self.lazy[node] is not None:
            mid = (start + end) // 2
            left_len = mid - start + 1
            right_len = end - mid

            self.tree[2 * node] = self.lazy[node] * left_len
            self.tree[2 * node + 1] = self.lazy[node] * right_len

            self.lazy[2 * node] = self.lazy[node]
            self.lazy[2 * node + 1] = self.lazy[node]

            self.lazy[node] = None

    def range_assign(self, l: int, r: int, val: int):
        """区間 [l, r] を val に代入 - O(log n)"""
        self._range_assign(1, 0, self.n - 1, l, r, val)

    def _range_assign(self, node, start, end, l, r, val):
        if r < start or end < l:
            return
        if l <= start and end <= r:
            self.tree[node] = val * (end - start + 1)
            self.lazy[node] = val
            return
        self._push_down(node, start, end)
        mid = (start + end) // 2
        self._range_assign(2 * node, start, mid, l, r, val)
        self._range_assign(2 * node + 1, mid + 1, end, l, r, val)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def query(self, l: int, r: int) -> int:
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
data = [1, 2, 3, 4, 5]
lst = LazySegmentTreeAssign(data)
print(lst.query(0, 4))     # 15 (1+2+3+4+5)
lst.range_assign(1, 3, 10)  # [1, 10, 10, 10, 5]
print(lst.query(0, 4))     # 36 (1+10+10+10+5)
```

### 遅延伝播：区間加算 + 区間最小値クエリ

```python
class LazySegmentTreeMinAdd:
    """遅延伝播付きセグメント木（区間加算 + 区間最小値クエリ）"""

    def __init__(self, data: list):
        self.n = len(data)
        self.tree = [float('inf')] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        self._build(data, 1, 0, self.n - 1)

    def _build(self, data, node, start, end):
        if start == end:
            self.tree[node] = data[start]
            return
        mid = (start + end) // 2
        self._build(data, 2 * node, start, mid)
        self._build(data, 2 * node + 1, mid + 1, end)
        self.tree[node] = min(self.tree[2 * node], self.tree[2 * node + 1])

    def _push_down(self, node):
        if self.lazy[node] != 0:
            for child in [2 * node, 2 * node + 1]:
                self.tree[child] += self.lazy[node]
                self.lazy[child] += self.lazy[node]
            self.lazy[node] = 0

    def range_add(self, l: int, r: int, val: int):
        self._range_add(1, 0, self.n - 1, l, r, val)

    def _range_add(self, node, start, end, l, r, val):
        if r < start or end < l:
            return
        if l <= start and end <= r:
            self.tree[node] += val
            self.lazy[node] += val
            return
        self._push_down(node)
        mid = (start + end) // 2
        self._range_add(2 * node, start, mid, l, r, val)
        self._range_add(2 * node + 1, mid + 1, end, l, r, val)
        self.tree[node] = min(self.tree[2 * node], self.tree[2 * node + 1])

    def query_min(self, l: int, r: int):
        return self._query_min(1, 0, self.n - 1, l, r)

    def _query_min(self, node, start, end, l, r):
        if r < start or end < l:
            return float('inf')
        if l <= start and end <= r:
            return self.tree[node]
        self._push_down(node)
        mid = (start + end) // 2
        return min(
            self._query_min(2 * node, start, mid, l, r),
            self._query_min(2 * node + 1, mid + 1, end, l, r)
        )

data = [5, 2, 8, 1, 9, 3]
lst = LazySegmentTreeMinAdd(data)
print(lst.query_min(0, 5))     # 1
lst.range_add(2, 4, -5)        # [5, 2, 3, -4, 4, 3]
print(lst.query_min(0, 5))     # -4
```

---

## 6. BIT（Binary Indexed Tree / Fenwick Tree）

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

ビット操作による親子関係:
  index   binary   lowbit  カバー範囲
    1     0001      1      [1,1]
    2     0010      2      [1,2]
    3     0011      1      [3,3]
    4     0100      4      [1,4]
    5     0101      1      [5,5]
    6     0110      2      [5,6]
    7     0111      1      [7,7]
    8     1000      8      [1,8]
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

    @classmethod
    def from_array_fast(cls, data: list):
        """配列から O(n) で構築（高速版）"""
        n = len(data)
        bit = cls(n)
        for i in range(1, n + 1):
            bit.tree[i] += data[i - 1]
            j = i + (i & (-i))
            if j <= n:
                bit.tree[j] += bit.tree[i]
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

    def lower_bound(self, target: int) -> int:
        """prefix_sum(i) >= target となる最小の i を返す
        （BIT上の二分探索）- O(log n)
        """
        pos = 0
        total = 0
        k = 1
        while k <= self.n:
            k <<= 1
        k >>= 1

        while k > 0:
            if pos + k <= self.n and total + self.tree[pos + k] < target:
                total += self.tree[pos + k]
                pos += k
            k >>= 1

        return pos + 1

# 使用例
data = [3, 2, 5, 1, 4, 7, 2, 6]
bit = BIT.from_array(data)
print(bit.range_sum(2, 5))  # 12 (2+5+1+4)
bit.update(3, 3)             # data[3] += 3 (5→8)
print(bit.range_sum(2, 5))  # 15 (2+8+1+4)
```

### 区間加算対応 BIT（Range Update BIT）

```python
class RangeUpdateBIT:
    """区間加算 + 点クエリ / 区間和クエリ に対応する BIT
    2本の BIT を使って区間加算を実現
    """

    def __init__(self, n: int):
        self.n = n
        self.bit1 = BIT(n)  # 差分用
        self.bit2 = BIT(n)  # 補正用

    def range_add(self, l: int, r: int, val: int):
        """区間 [l, r] に val を加算"""
        self.bit1.update(l, val)
        self.bit1.update(r + 1, -val)
        self.bit2.update(l, val * (l - 1))
        self.bit2.update(r + 1, -val * r)

    def prefix_sum(self, i: int) -> int:
        """a[1] + ... + a[i] の和"""
        return self.bit1.prefix_sum(i) * i - self.bit2.prefix_sum(i)

    def range_sum(self, l: int, r: int) -> int:
        """a[l] + ... + a[r] の和"""
        return self.prefix_sum(r) - self.prefix_sum(l - 1)

    def point_query(self, i: int) -> int:
        """a[i] の値"""
        return self.bit1.prefix_sum(i)

# 使用例
rbit = RangeUpdateBIT(8)
rbit.range_add(2, 5, 3)  # [0, 3, 3, 3, 3, 0, 0, 0]
print(rbit.point_query(3))   # 3
print(rbit.range_sum(1, 8))  # 12
```

### 2次元 BIT

```python
class BIT2D:
    """2次元 Binary Indexed Tree"""

    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.tree = [[0] * (cols + 1) for _ in range(rows + 1)]

    def update(self, r: int, c: int, delta: int):
        """(r, c) に delta を加算 - O(log R * log C)"""
        i = r
        while i <= self.rows:
            j = c
            while j <= self.cols:
                self.tree[i][j] += delta
                j += j & (-j)
            i += i & (-i)

    def prefix_sum(self, r: int, c: int) -> int:
        """(1,1) ~ (r,c) の矩形和 - O(log R * log C)"""
        s = 0
        i = r
        while i > 0:
            j = c
            while j > 0:
                s += self.tree[i][j]
                j -= j & (-j)
            i -= i & (-i)
        return s

    def range_sum(self, r1: int, c1: int, r2: int, c2: int) -> int:
        """(r1,c1) ~ (r2,c2) の矩形和"""
        return (self.prefix_sum(r2, c2)
                - self.prefix_sum(r1 - 1, c2)
                - self.prefix_sum(r2, c1 - 1)
                + self.prefix_sum(r1 - 1, c1 - 1))

# 使用例
bit2d = BIT2D(4, 4)
bit2d.update(1, 1, 3)
bit2d.update(2, 3, 5)
bit2d.update(3, 2, 7)
print(bit2d.range_sum(1, 1, 3, 3))  # 15 (3+5+7)
```

---

## 7. セグメント木 vs BIT 比較表

| 特性 | セグメント木 | BIT |
|:---|:---|:---|
| 空間計算量 | O(4n) | O(n) |
| 構築 | O(n) | O(n log n)（O(n)も可能） |
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
| k番目の要素 | BIT（二分探索付き） | lower_bound が効率的 |

---

## 8. 転倒数の計算（BIT応用）

```python
def count_inversions(arr: list) -> int:
    """転倒数をBITで O(n log n) で計算
    転倒数 = i < j かつ arr[i] > arr[j] となるペア (i, j) の数
    """
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
# (5,3),(5,2),(5,4),(5,1),(3,2),(3,1),(4,1) = 7
```

### k番目の要素の検索（BIT上の二分探索）

```python
def kth_smallest(bit: BIT, k: int) -> int:
    """BIT上で k 番目に小さい要素のインデックスを返す
    BIT[i] = 値 i が存在するか（0 or 1）として管理
    """
    return bit.lower_bound(k)

# 使用例: 動的な k 番目の要素
class DynamicKthElement:
    """要素の追加・削除を行いながら k 番目の要素を求める"""

    def __init__(self, max_val: int):
        self.bit = BIT(max_val)
        self.count = 0

    def add(self, val: int):
        self.bit.update(val, 1)
        self.count += 1

    def remove(self, val: int):
        self.bit.update(val, -1)
        self.count -= 1

    def kth(self, k: int) -> int:
        """k 番目に小さい要素を返す（1-indexed）"""
        return self.bit.lower_bound(k)

dke = DynamicKthElement(100)
dke.add(10)
dke.add(30)
dke.add(20)
dke.add(50)
print(dke.kth(1))  # 10（最小）
print(dke.kth(3))  # 30（3番目）
dke.remove(20)
print(dke.kth(2))  # 30（2番目）
```

---

## 9. セグメント木の応用問題

### セグメント木上の二分探索

```python
class SegmentTreeWithSearch(SegmentTree):
    """セグメント木上の二分探索
    「区間 [0, r] の和が target 以上になる最小の r」を O(log n) で求める
    """

    def find_first(self, target: int) -> int:
        """prefix_sum(r) >= target となる最小の r"""
        return self._find_first(1, 0, self.n - 1, target)

    def _find_first(self, node, start, end, target):
        if self.tree[node] < target:
            return -1  # この部分木では不足
        if start == end:
            return start  # 見つかった
        mid = (start + end) // 2
        # 左の子で十分なら左へ
        if self.tree[2 * node] >= target:
            return self._find_first(2 * node, start, mid, target)
        else:
            # 左の分を引いて右で探索
            return self._find_first(
                2 * node + 1, mid + 1, end,
                target - self.tree[2 * node]
            )
```

### 最長増加部分列（LIS）のセグメント木解法

```python
def lis_segtree(arr: list) -> int:
    """LIS の長さをセグメント木で O(n log n) で計算"""
    # 座標圧縮
    sorted_unique = sorted(set(arr))
    compress = {v: i for i, v in enumerate(sorted_unique)}
    m = len(sorted_unique)

    # 区間最大値のセグメント木
    st = AbstractSegmentTree([0] * m, max, 0)

    for val in arr:
        idx = compress[val]
        # val より小さい要素で終わる LIS の最大長
        if idx > 0:
            best = st.query(0, idx)  # [0, idx) の最大値
        else:
            best = 0
        # val で終わる LIS の長さを更新
        st.update(idx, best + 1)

    return st.query(0, m)

arr = [10, 9, 2, 5, 3, 7, 101, 18]
print(lis_segtree(arr))  # 4 (2, 3, 7, 101)
```

### 区間スケジューリング（セグメント木）

```python
def max_events_attended(events: list) -> int:
    """最大参加イベント数（各日1イベントのみ）
    events: [(start_day, end_day), ...]
    セグメント木で空いている日を効率的に検索
    """
    if not events:
        return 0

    # 終了日でソート（貪欲法）
    events.sort(key=lambda x: x[1])
    max_day = max(e[1] for e in events)

    # セグメント木（区間最小値）: 0=空き, 1=予約済み
    tree = [0] * (4 * (max_day + 1))
    n = max_day + 1

    def update(node, start, end, idx, val):
        if start == end:
            tree[node] = val
            return
        mid = (start + end) // 2
        if idx <= mid:
            update(2 * node, start, mid, idx, val)
        else:
            update(2 * node + 1, mid + 1, end, idx, val)
        tree[node] = min(tree[2 * node], tree[2 * node + 1])

    def find_empty(node, start, end, l, r):
        """[l, r] 内の最初の空き日を返す（-1 = 空きなし）"""
        if r < start or end < l or tree[node] >= 1:
            return -1
        if start == end:
            return start if tree[node] == 0 else -1
        mid = (start + end) // 2
        left = find_empty(2 * node, start, mid, l, r)
        if left != -1:
            return left
        return find_empty(2 * node + 1, mid + 1, end, l, r)

    count = 0
    for s, e in events:
        day = find_empty(1, 0, n - 1, s, e)
        if day != -1:
            update(1, 0, n - 1, day, 1)
            count += 1

    return count
```

---

## 10. 永続セグメント木

過去の任意のバージョンのセグメント木にアクセスできる。各更新で変更されたノードのみを新規作成する（共有構造）。

```python
class PersistentSegmentTree:
    """永続セグメント木（区間和）
    更新のたびに新しいバージョンが作られ、過去のバージョンも参照可能
    空間計算量: O(n + q * log n)（q = 更新回数）
    """

    class Node:
        __slots__ = ['left', 'right', 'val']

        def __init__(self, val=0, left=None, right=None):
            self.val = val
            self.left = left
            self.right = right

    def __init__(self, data: list):
        self.n = len(data)
        self.roots = []
        self.roots.append(self._build(data, 0, self.n - 1))

    def _build(self, data, start, end):
        if start == end:
            return self.Node(data[start])
        mid = (start + end) // 2
        left = self._build(data, start, mid)
        right = self._build(data, mid + 1, end)
        return self.Node(left.val + right.val, left, right)

    def update(self, version: int, idx: int, val: int) -> int:
        """version の木を基に idx を val に更新した新バージョンを作成
        Returns: 新バージョン番号
        """
        new_root = self._update(self.roots[version], 0, self.n - 1, idx, val)
        self.roots.append(new_root)
        return len(self.roots) - 1

    def _update(self, node, start, end, idx, val):
        if start == end:
            return self.Node(val)
        mid = (start + end) // 2
        if idx <= mid:
            new_left = self._update(node.left, start, mid, idx, val)
            return self.Node(new_left.val + node.right.val, new_left, node.right)
        else:
            new_right = self._update(node.right, mid + 1, end, idx, val)
            return self.Node(node.left.val + new_right.val, node.left, new_right)

    def query(self, version: int, l: int, r: int) -> int:
        """version の木で区間 [l, r] の和を求める"""
        return self._query(self.roots[version], 0, self.n - 1, l, r)

    def _query(self, node, start, end, l, r):
        if node is None or r < start or end < l:
            return 0
        if l <= start and end <= r:
            return node.val
        mid = (start + end) // 2
        return (self._query(node.left, start, mid, l, r) +
                self._query(node.right, mid + 1, end, l, r))

# 使用例
data = [1, 2, 3, 4, 5]
pst = PersistentSegmentTree(data)  # version 0

v1 = pst.update(0, 2, 10)  # version 1: [1, 2, 10, 4, 5]
v2 = pst.update(1, 4, 20)  # version 2: [1, 2, 10, 4, 20]

print(pst.query(0, 0, 4))  # 15 (version 0: 1+2+3+4+5)
print(pst.query(1, 0, 4))  # 22 (version 1: 1+2+10+4+5)
print(pst.query(2, 0, 4))  # 37 (version 2: 1+2+10+4+20)
```

---

## 11. Sparse Table（静的RMQ）

更新がない場合の区間最小値クエリに特化した構造。前処理 O(n log n)、クエリ O(1)。

```python
import math

class SparseTable:
    """Sparse Table - 静的 RMQ（Range Minimum Query）
    前処理: O(n log n), クエリ: O(1), 更新: 不可
    """

    def __init__(self, data: list):
        self.n = len(data)
        self.LOG = max(1, math.floor(math.log2(self.n))) + 1
        self.table = [[float('inf')] * self.n for _ in range(self.LOG)]

        # 初期化
        for i in range(self.n):
            self.table[0][i] = data[i]

        # DP で構築
        for j in range(1, self.LOG):
            for i in range(self.n - (1 << j) + 1):
                self.table[j][i] = min(
                    self.table[j-1][i],
                    self.table[j-1][i + (1 << (j-1))]
                )

    def query(self, l: int, r: int) -> int:
        """区間 [l, r] の最小値 - O(1)"""
        length = r - l + 1
        k = math.floor(math.log2(length))
        return min(self.table[k][l], self.table[k][r - (1 << k) + 1])

# 使用例
data = [5, 2, 8, 1, 9, 3, 7, 4]
sp = SparseTable(data)
print(sp.query(0, 3))  # 1
print(sp.query(4, 7))  # 3
print(sp.query(1, 6))  # 1

# RMQ の手法比較
# | 手法           | 前処理    | クエリ    | 更新     |
# |:-------------|:---------|:---------|:---------|
# | ナイーブ       | O(1)     | O(n)     | O(1)     |
# | セグメント木    | O(n)     | O(log n) | O(log n) |
# | Sparse Table  | O(n lg n)| O(1)     | 不可     |
# | 平方分割       | O(n)     | O(√n)    | O(1)     |
```

---

## 12. 平方分割（Sqrt Decomposition）

配列を √n ブロックに分割する手法。セグメント木より実装が簡単で、一部の問題では十分な性能を提供する。

```python
import math

class SqrtDecomposition:
    """平方分割 - 区間和クエリ + 点更新
    構築: O(n), クエリ: O(√n), 更新: O(1)
    """

    def __init__(self, data: list):
        self.n = len(data)
        self.block_size = max(1, int(math.sqrt(self.n)))
        self.num_blocks = (self.n + self.block_size - 1) // self.block_size
        self.data = data[:]
        self.blocks = [0] * self.num_blocks

        for i in range(self.n):
            self.blocks[i // self.block_size] += data[i]

    def update(self, idx: int, val: int):
        """点更新 - O(1)"""
        block = idx // self.block_size
        self.blocks[block] += val - self.data[idx]
        self.data[idx] = val

    def query(self, l: int, r: int) -> int:
        """区間 [l, r] の和 - O(√n)"""
        result = 0
        bl = l // self.block_size
        br = r // self.block_size

        if bl == br:
            # 同じブロック内
            for i in range(l, r + 1):
                result += self.data[i]
        else:
            # 左端の端数
            for i in range(l, (bl + 1) * self.block_size):
                result += self.data[i]
            # 中間の完全ブロック
            for b in range(bl + 1, br):
                result += self.blocks[b]
            # 右端の端数
            for i in range(br * self.block_size, r + 1):
                result += self.data[i]

        return result

# 使用例
data = [1, 3, 5, 7, 9, 11, 13, 15, 17]
sd = SqrtDecomposition(data)
print(sd.query(2, 6))  # 45 (5+7+9+11+13)
sd.update(4, 100)       # data[4] = 9 → 100
print(sd.query(2, 6))  # 136 (5+7+100+11+13)
```

---

## 13. アンチパターン

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

### アンチパターン3: BIT の 0-indexed 使用

```python
# BAD: BIT を 0-indexed で使う
# i & (-i) は i=0 のとき 0 → 無限ループ!
class BadBIT:
    def prefix_sum(self, i):
        s = 0
        while i > 0:  # i=0 の場合、i & (-i) = 0 で抜けるが...
            s += self.tree[i]
            i -= i & (-i)  # 0 - 0 = 0 → 問題なし
        return s

    def update(self, i, delta):
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)  # 0 + 0 = 0 → 無限ループ!

# GOOD: BIT は必ず 1-indexed で使う
class GoodBIT:
    def update(self, i, delta):
        # i は 1 以上であることを保証
        assert i >= 1
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)
```

### アンチパターン4: セグメント木の区間が閉区間/半開区間で混在

```python
# BAD: 閉区間 [l, r] と半開区間 [l, r) が混在
# セグメント木は閉区間、BIT は 1-indexed と一貫性を持たせる

# GOOD: 実装全体でどちらかに統一する
# 再帰版セグメント木: 閉区間 [l, r] が一般的
# 非再帰版セグメント木: 半開区間 [l, r) が一般的
# BIT: 1-indexed + 閉区間 [l, r]
```

---

## 14. FAQ

### Q1: セグメント木のサイズはなぜ 4n？

**A:** セグメント木は完全二分木として構築され、n が 2 の冪でない場合、葉の数は次の 2 の冪に切り上げられる。ノードの索引付け（1-indexed, 子は 2*node と 2*node+1）を使うと、最大で 4n 程度のノードが必要になる。安全マージンを含めて 4n が定番。非再帰版では 2 * (次の2の冪) で正確に確保できる。

### Q2: BIT で区間最小値は求められるか？

**A:** 標準の BIT では困難。BIT は「接頭辞に関する可逆な二項演算」に適しており、min/max は逆元が存在しないため、区間 [l, r] のクエリに分解できない。区間最小値にはセグメント木を使う。ただし、値の更新が増加方向のみ（値の減少がない）場合は BIT でも区間最小値が求められるという特殊ケースがある。

### Q3: 2次元のセグメント木は？

**A:** 外側のセグメント木の各ノードが内側のセグメント木を持つ「2D セグメント木」が構築可能。計算量は O(log^2 n)、空間は O(n^2 log^2 n)。実用的には 2D BIT のほうが実装が簡潔。さらに高度なものとして KD-Tree や R-Tree がある。

### Q4: 遅延伝播で複数の操作を組み合わせるには？

**A:** 例えば「区間代入」と「区間加算」を混在させる場合、遅延値を (add, assign) のペアで管理し、合成規則を正しく定義する必要がある。一般的には「assign があれば add をリセットし、add のみなら加算」のようなルールになる。このような複合遅延伝播は実装が難しく、バグの温床になりやすい。

### Q5: セグメント木のノード数が多くて MLE（メモリ制限超過）になる場合は？

**A:** (1) 動的セグメント木（ノードを必要時に生成）を使う。座標が大きいが実際に使われるインデックスが少ない場合に有効。(2) 座標圧縮で値の範囲を縮小する。(3) BIT が使える問題なら BIT に切り替える（メモリが約 1/4）。

---

## 15. まとめ

| 項目 | 要点 |
|:---|:---|
| セグメント木 | 区間クエリ + 点更新を O(log n) で処理 |
| 遅延伝播 | 区間更新を O(log n) に拡張。push_down が鍵 |
| BIT | 接頭辞和特化。セグメント木より簡潔で高速 |
| 抽象セグメント木 | モノイド演算を汎用化。任意の結合的演算に対応 |
| 永続セグメント木 | 過去のバージョンにアクセス可能 |
| Sparse Table | 静的RMQ に O(1) で回答。更新不可 |
| 平方分割 | 実装が簡単。O(√n) のクエリ |
| 適用場面 | 区間和・最小・最大・GCD の動的クエリ |

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
5. Bender, M. A. & Farach-Colton, M. (2000). "The LCA Problem Revisited." *LATIN*. -- Sparse Table
6. 秋葉拓哉ほか (2012). 『プログラミングコンテストチャレンジブック 第2版』. マイナビ出版.
