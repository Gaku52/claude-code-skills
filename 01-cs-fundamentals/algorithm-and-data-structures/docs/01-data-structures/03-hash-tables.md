# ハッシュテーブル — ハッシュ関数・衝突解決・ロードファクター

> 平均 O(1) のキー検索を実現するハッシュテーブルの内部構造、衝突解決戦略、性能チューニングを学ぶ。

---

## この章で学ぶこと

1. **ハッシュ関数** の設計原則と良い関数の条件
2. **衝突解決** — チェイン法とオープンアドレス法
3. **ロードファクター** とリハッシュの仕組み

---

## 1. ハッシュテーブルの基本構造

```
キー "apple" → ハッシュ関数 → インデックス 3

  バケット配列:
  [0] → null
  [1] → ("banana", 2)
  [2] → null
  [3] → ("apple", 5)     ← h("apple") = 3
  [4] → null
  [5] → ("cherry", 8)
  [6] → null
  [7] → ("date", 1)

  検索 "apple":
  1. h("apple") = 3
  2. bucket[3] を参照
  3. キーが一致 → 値 5 を返す
  → O(1) 平均
```

---

## 2. ハッシュ関数

### 2.1 良いハッシュ関数の条件

```python
# 条件:
# 1. 決定的: 同じ入力 → 同じ出力
# 2. 均一分布: 出力がバケット全体に均等に散らばる
# 3. 高速: 計算が O(キー長) 程度

# 文字列のハッシュ関数例（多項式ハッシュ）
def polynomial_hash(key, table_size, base=31):
    """多項式ハッシュ — O(len(key))"""
    h = 0
    for char in key:
        h = (h * base + ord(char)) % table_size
    return h

# Python の組み込み hash()
print(hash("hello"))    # 整数値を返す
print(hash(42))         # 整数はそのまま（概ね）
print(hash((1, 2, 3)))  # タプルはハッシュ可能
# hash([1, 2, 3])       # リストはハッシュ不可（mutable）
```

---

## 3. 衝突解決

### 3.1 チェイン法（Separate Chaining）

```
バケット配列 + 連結リスト:

  [0] → null
  [1] → ("banana",2) → ("fig",7) → null
  [2] → null
  [3] → ("apple",5) → ("grape",3) → null
  [4] → null

  h("banana") = h("fig") = 1  → チェインで格納
```

```python
class HashTableChaining:
    def __init__(self, size=16):
        self.size = size
        self.buckets = [[] for _ in range(size)]
        self.count = 0

    def _hash(self, key):
        return hash(key) % self.size

    def put(self, key, value):
        """O(1) 平均"""
        idx = self._hash(key)
        for i, (k, v) in enumerate(self.buckets[idx]):
            if k == key:
                self.buckets[idx][i] = (key, value)
                return
        self.buckets[idx].append((key, value))
        self.count += 1
        if self.count / self.size > 0.75:
            self._rehash()

    def get(self, key):
        """O(1) 平均"""
        idx = self._hash(key)
        for k, v in self.buckets[idx]:
            if k == key:
                return v
        raise KeyError(key)

    def _rehash(self):
        old = self.buckets
        self.size *= 2
        self.buckets = [[] for _ in range(self.size)]
        self.count = 0
        for bucket in old:
            for key, value in bucket:
                self.put(key, value)
```

### 3.2 オープンアドレス法（線形探索法）

```
h("apple") = 3, h("grape") = 3 → 衝突!

線形探索法: 次の空きスロットを探す
  [0] → null
  [1] → null
  [2] → null
  [3] → ("apple", 5)   ← h("apple") = 3
  [4] → ("grape", 3)   ← h("grape") = 3 → 衝突 → 3+1=4
  [5] → null
```

```python
class HashTableOpenAddr:
    DELETED = object()

    def __init__(self, size=16):
        self.size = size
        self.keys = [None] * size
        self.values = [None] * size
        self.count = 0

    def _hash(self, key):
        return hash(key) % self.size

    def put(self, key, value):
        idx = self._hash(key)
        while self.keys[idx] is not None and self.keys[idx] is not self.DELETED:
            if self.keys[idx] == key:
                self.values[idx] = value
                return
            idx = (idx + 1) % self.size
        self.keys[idx] = key
        self.values[idx] = value
        self.count += 1
        if self.count / self.size > 0.5:
            self._rehash()

    def get(self, key):
        idx = self._hash(key)
        while self.keys[idx] is not None:
            if self.keys[idx] == key:
                return self.values[idx]
            idx = (idx + 1) % self.size
        raise KeyError(key)

    def delete(self, key):
        idx = self._hash(key)
        while self.keys[idx] is not None:
            if self.keys[idx] == key:
                self.keys[idx] = self.DELETED
                self.values[idx] = None
                self.count -= 1
                return
            idx = (idx + 1) % self.size
        raise KeyError(key)

    def _rehash(self):
        old_keys, old_values = self.keys, self.values
        self.size *= 2
        self.keys = [None] * self.size
        self.values = [None] * self.size
        self.count = 0
        for k, v in zip(old_keys, old_values):
            if k is not None and k is not self.DELETED:
                self.put(k, v)
```

---

## 4. ロードファクター

```
ロードファクター α = 要素数 / バケット数

α の影響:
  ┌────────────────────────────────────┐
  │                                    │
  │ 探索時間                          │
  │ ▲                                 │
  │ │                        ╱        │
  │ │                     ╱           │
  │ │                  ╱   チェイン法  │
  │ │               ╱                 │
  │ │          ╱╱╱  オープンアドレス   │
  │ │    ╱╱╱                          │
  │ │╱╱                               │
  │ ┼──────────────────────► α        │
  │ 0   0.25  0.5  0.75  1.0         │
  │                                    │
  │ 推奨 α:                           │
  │   チェイン法: α < 0.75            │
  │   オープンアドレス法: α < 0.5      │
  └────────────────────────────────────┘
```

---

## 5. 比較表

### 表1: 衝突解決法の比較

| 特性 | チェイン法 | オープンアドレス法 |
|------|-----------|------------------|
| 構造 | 配列 + リスト | 配列のみ |
| メモリ | ポインタ分余分 | 密にパック |
| 最悪検索 | O(n) | O(n) |
| 削除 | 容易 | DELETED マーカー必要 |
| キャッシュ効率 | 低い | 高い |
| 推奨 α | < 0.75 | < 0.5 |
| 実装 | 簡単 | やや複雑 |

### 表2: 言語別ハッシュテーブル実装

| 言語 | 型名 | 衝突解決 | 初期容量 | 最大 α |
|------|------|---------|---------|--------|
| Python | dict | オープンアドレス | 8 | 2/3 |
| Java | HashMap | チェイン(+木) | 16 | 0.75 |
| C++ | unordered_map | チェイン | 実装依存 | 1.0 |
| Go | map | チェイン(バケット) | 実装依存 | 6.5 |
| Rust | HashMap | Robin Hood | 実装依存 | 7/8 |

---

## 6. アンチパターン

### アンチパターン1: mutable オブジェクトをキーにする

```python
# BAD: リストはハッシュ不可
d = {}
key = [1, 2, 3]
# d[key] = "value"  # TypeError: unhashable type: 'list'

# GOOD: タプルに変換
d[tuple(key)] = "value"

# BAD: カスタムクラスの __hash__ を変更可能フィールドで定義
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __hash__(self):
        return hash((self.x, self.y))
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

p = Point(1, 2)
d = {p: "origin"}
p.x = 10  # ハッシュ値が変わる → d[p] が見つからなくなる!
```

### アンチパターン2: 衝突が多いハッシュ関数

```python
# BAD: 全てのキーが同じバケットに → O(n) 探索
def terrible_hash(key, size):
    return 0  # 全て index 0

# BAD: 下位ビットだけ使う
def bad_hash(key, size):
    return key & 0xF  # 16通りしかない

# GOOD: Python の hash() + 素数サイズ
def good_hash(key, size):
    return hash(key) % size
```

---

## 7. FAQ

### Q1: Python の dict はなぜ挿入順序を保持するか？

**A:** Python 3.7 以降、dict はコンパクトな配列構造を採用し、挿入順序を保持する。内部的にはハッシュインデックス配列と、挿入順の密な配列の2層構造。

### Q2: ハッシュテーブルの最悪ケース O(n) はどう避けるか？

**A:** Java の HashMap は衝突が多いバケットを赤黒木に変換し O(log n) に改善する。また、ユニバーサルハッシュやカッコウハッシュで最悪ケースを確率的に回避できる。

### Q3: set と dict の内部構造は同じか？

**A:** Python では set と dict はほぼ同じハッシュテーブル構造。set は値を持たない分メモリ効率が良い。操作（in, add, remove）は同じ O(1) 平均。

---

## 8. まとめ

| 項目 | ポイント |
|------|---------|
| ハッシュ関数 | 均一分布・決定的・高速が条件 |
| チェイン法 | 実装が簡単。削除も容易 |
| オープンアドレス法 | キャッシュ効率が良い。Python dict が採用 |
| ロードファクター | 性能維持の鍵。閾値超過でリハッシュ |
| リハッシュ | テーブルサイズ倍増 + 全要素再挿入 |
| キーの条件 | immutable かつ __hash__ と __eq__ が整合的 |

---

## 次に読むべきガイド

- [木構造 — BST と平衡木](./04-trees.md)
- [時間空間トレードオフ — ブルームフィルタ](../00-complexity/02-space-time-tradeoff.md)

---

## 参考文献

1. Cormen, T.H. et al. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. — 第11章「Hash Tables」
2. Knuth, D.E. (1998). *The Art of Computer Programming, Volume 3*. Addison-Wesley. — ハッシュ法の理論
3. Python Developer's Guide. "Dictionaries." — https://docs.python.org/3/c-api/dict.html
