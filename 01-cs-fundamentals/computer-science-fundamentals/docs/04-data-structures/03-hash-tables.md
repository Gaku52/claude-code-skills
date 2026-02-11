# ハッシュテーブル

> ハッシュテーブルは「期待O(1)の探索」を実現する、実務で最も頻繁に使用されるデータ構造である。

## この章で学ぶこと

- [ ] ハッシュ関数と衝突解決の仕組みを理解する
- [ ] ハッシュテーブルの性能特性を説明できる
- [ ] 実務での使い分け（dict/set/Map等）を理解する

## 前提知識

- 配列 → 参照: [[00-arrays-and-strings.md]]
- 計算量解析 → 参照: [[../03-algorithms/01-complexity-analysis.md]]

---

## 1. ハッシュテーブルの仕組み

### 1.1 基本構造

```
ハッシュテーブル: キー → ハッシュ関数 → インデックス → 値

  キー "Alice" → hash("Alice") = 0x7A3B...
                → 0x7A3B % 8 = 3  (テーブルサイズ8)
                → table[3] = "Alice: 100"

  ┌─────┐
  │  0  │ → (空)
  │  1  │ → ("Bob", 85)
  │  2  │ → (空)
  │  3  │ → ("Alice", 100)
  │  4  │ → ("Charlie", 92) → ("Eve", 78)  ← 衝突(チェイニング)
  │  5  │ → (空)
  │  6  │ → ("Diana", 88)
  │  7  │ → (空)
  └─────┘
```

### 1.2 衝突解決

```
衝突解決の2大方式:

  1. チェイニング（Separate Chaining）:
     → 同じスロットにリンクリストで格納
     → 実装が単純、ロードファクターが1を超えてもOK
     → Java HashMap, Go map

  2. オープンアドレス法:
     → 衝突時に別のスロットを探す
     → メモリ効率が良い、キャッシュフレンドリー
     → Python dict, Rust HashMap

     探索方法:
     - 線形探索: h(k)+1, h(k)+2, ...（クラスタリング問題）
     - 二次探索: h(k)+1², h(k)+2², ...
     - ダブルハッシュ: h(k)+i×h2(k)

  ロードファクター = 要素数 / テーブルサイズ
  → チェイニング: 0.75で拡張（Java）
  → オープンアドレス: 2/3で拡張（Python）
```

### 1.3 計算量

```
ハッシュテーブルの計算量:

  ┌──────────┬──────────┬──────────┐
  │ 操作     │ 期待     │ 最悪     │
  ├──────────┼──────────┼──────────┤
  │ 挿入     │ O(1)     │ O(n)     │
  │ 検索     │ O(1)     │ O(n)     │
  │ 削除     │ O(1)     │ O(n)     │
  └──────────┴──────────┴──────────┘

  最悪ケース O(n) の条件:
  - 全てのキーが同じスロットに衝突
  - 意図的な攻撃（Hash DoS）

  対策:
  - ランダム化ハッシュ（SipHash: Python, Rust）
  - 赤黒木へのフォールバック（Java 8+ HashMap）
```

---

## 2. 実務での活用

### 2.1 各言語のハッシュテーブル

```python
# Python: dict（最も使用頻度が高いデータ構造）
d = {"name": "Alice", "age": 30}
d["email"] = "alice@example.com"  # O(1)
"name" in d  # O(1)

# Python: set（重複排除、集合演算）
s = {1, 2, 3}
s.add(4)          # O(1)
2 in s            # O(1)
s & {2, 3, 4}     # 積集合: {2, 3}
s | {4, 5}        # 和集合: {1, 2, 3, 4, 5}

# Counter（頻度カウント）
from collections import Counter
words = ["apple", "banana", "apple", "cherry", "banana", "apple"]
count = Counter(words)
# Counter({'apple': 3, 'banana': 2, 'cherry': 1})
count.most_common(2)  # [('apple', 3), ('banana', 2)]

# defaultdict（デフォルト値付き辞書）
from collections import defaultdict
graph = defaultdict(list)
graph["A"].append("B")  # KeyError なし
```

### 2.2 ハッシュテーブルの設計パターン

```python
# 1. メモ化（キャッシュ）
cache = {}
def expensive_computation(key):
    if key not in cache:
        cache[key] = compute(key)  # 初回のみ計算
    return cache[key]

# 2. グルーピング
from collections import defaultdict
def group_anagrams(words):
    groups = defaultdict(list)
    for word in words:
        key = tuple(sorted(word))
        groups[key].append(word)
    return list(groups.values())

# 3. Two Sum（ハッシュマップで O(n)）
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
```

---

## 3. 実践演習

### 演習1: 基本操作（基礎）
ハッシュテーブルをゼロから実装せよ（チェイニング方式、リサイズ付き）。

### 演習2: LRUキャッシュ（応用）
OrderedDict または dict + 双方向連結リストで LRUキャッシュを実装せよ。

### 演習3: 一貫性ハッシュ（発展）
分散システムで使われる一貫性ハッシュ（Consistent Hashing）を実装せよ。

---

## まとめ

| 概念 | ポイント |
|------|---------|
| ハッシュ関数 | キー→整数。均一分布が理想 |
| 衝突解決 | チェイニング or オープンアドレス |
| 計算量 | 期待O(1)、最悪O(n) |
| 実務 | dict/set/Map/Counter/defaultdict |
| 注意 | Hash DoS対策、順序は非保証（Python 3.7+は保証）|

---

## 次に読むべきガイド
→ [[04-trees.md]] — 木構造

---

## 参考文献
1. Cormen, T. H. "Introduction to Algorithms." Chapter 11: Hash Tables.
2. Sedgewick, R. "Algorithms." Chapter 3.4: Hash Tables.
