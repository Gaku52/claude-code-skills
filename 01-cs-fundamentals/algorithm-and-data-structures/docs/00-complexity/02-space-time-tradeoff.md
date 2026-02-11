# 時間空間トレードオフ — メモ化・テーブル・ブルームフィルタ

> メモリを追加で使用することで計算時間を削減する（または逆に空間を節約して時間を犠牲にする）手法を体系的に学ぶ。

---

## この章で学ぶこと

1. **メモ化** による再帰の高速化とそのコスト
2. **ルックアップテーブル** と事前計算のパターン
3. **ブルームフィルタ** など確率的データ構造による空間節約

---

## 1. トレードオフの基本概念

```
時間空間トレードオフの全体像:

  時間 ▲
       │  ● 素朴なアルゴリズム
       │     (時間大・空間小)
       │
       │        ● 適度なバランス
       │
       │              ● テーブル事前計算
       │                 (時間小・空間大)
       ┼──────────────────────────► 空間

  「空間を使えば時間が減り、空間を減らせば時間が増える」
```

---

## 2. メモ化（Memoization）

### 2.1 素朴なフィボナッチ vs メモ化

```python
# 素朴: 時間 O(2ⁿ), 空間 O(n)（スタック）
def fib_naive(n):
    if n <= 1:
        return n
    return fib_naive(n - 1) + fib_naive(n - 2)

# メモ化: 時間 O(n), 空間 O(n)
def fib_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib_memo(n - 1, memo) + fib_memo(n - 2, memo)
    return memo[n]
```

### 2.2 呼び出しの削減を図で理解

```
素朴なフィボナッチ fib(5) — 重複計算が多い:

          fib(5)
         /      \
      fib(4)    fib(3)
      /    \     /   \
   fib(3) fib(2) fib(2) fib(1)
   /  \    / \    / \
 f(2) f(1) f(1) f(0) f(1) f(0)
 / \
f(1) f(0)

→ 15 回の呼び出し

メモ化版 — 各値は1回だけ計算:

fib(5) → fib(4) → fib(3) → fib(2) → fib(1): 計算
                                     fib(0): 計算
                            fib(1): キャッシュ参照
                   fib(2): キャッシュ参照
          fib(3): キャッシュ参照

→ 5 回の計算 + キャッシュ参照
```

### 2.3 Python デコレータによるメモ化

```python
from functools import lru_cache

@lru_cache(maxsize=None)
def fib(n):
    """時間 O(n), 空間 O(n)"""
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)

# maxsize を指定すると LRU キャッシュで空間を制限できる
@lru_cache(maxsize=128)
def expensive_computation(x, y):
    # 重い計算
    return x ** y % 1000000007
```

---

## 3. ルックアップテーブル

### 3.1 事前計算テーブル

```python
# 例: popcount（ビットの1の数）
# 方法1: 毎回計算 — O(log n)
def popcount_naive(n):
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count

# 方法2: テーブル参照 — O(1) (8ビットテーブル)
TABLE = [0] * 256
for i in range(256):
    TABLE[i] = (i & 1) + TABLE[i >> 1]

def popcount_table(n):
    """32ビット整数の popcount — O(1) 時間, O(256) 空間"""
    return (TABLE[n & 0xFF] +
            TABLE[(n >> 8) & 0xFF] +
            TABLE[(n >> 16) & 0xFF] +
            TABLE[(n >> 24) & 0xFF])
```

### 3.2 三角関数テーブル

```python
import math

# 事前計算（1度刻み）
SIN_TABLE = [math.sin(math.radians(i)) for i in range(360)]
COS_TABLE = [math.cos(math.radians(i)) for i in range(360)]

def fast_sin(degrees):
    """近似 sin — O(1)"""
    return SIN_TABLE[int(degrees) % 360]
```

---

## 4. ブルームフィルタ

### 4.1 仕組み

```
ブルームフィルタ: 空間効率の良い所属判定

ビット配列 (m ビット): [0][0][0][0][0][0][0][0][0][0]

"apple" を追加 (ハッシュ関数 h1, h2, h3):
  h1("apple") = 2, h2("apple") = 5, h3("apple") = 8
  → [0][0][1][0][0][1][0][0][1][0]

"banana" を追加:
  h1("banana") = 1, h2("banana") = 5, h3("banana") = 9
  → [0][1][1][0][0][1][0][0][1][1]

"cherry" を検索:
  h1("cherry") = 2, h2("cherry") = 7, h3("cherry") = 9
  位置 2: ✓  位置 7: ✗  → 確実に「存在しない」

"date" を検索:
  h1("date") = 1, h2("date") = 5, h3("date") = 8
  位置 1: ✓  位置 5: ✓  位置 8: ✓  → 「たぶん存在する」（偽陽性の可能性）
```

### 4.2 Python 実装

```python
import hashlib

class BloomFilter:
    def __init__(self, size=1000, num_hashes=3):
        self.size = size
        self.num_hashes = num_hashes
        self.bit_array = [False] * size

    def _hashes(self, item):
        result = []
        for i in range(self.num_hashes):
            h = hashlib.md5(f"{item}{i}".encode()).hexdigest()
            result.append(int(h, 16) % self.size)
        return result

    def add(self, item):
        for pos in self._hashes(item):
            self.bit_array[pos] = True

    def might_contain(self, item):
        """True なら「たぶん存在」、False なら「確実に不在」"""
        return all(self.bit_array[pos] for pos in self._hashes(item))

# 使用例
bf = BloomFilter(size=10000, num_hashes=5)
for word in ["apple", "banana", "cherry"]:
    bf.add(word)

print(bf.might_contain("apple"))   # True（確実に追加済み）
print(bf.might_contain("grape"))   # False（確実に未追加）
```

---

## 5. 代表的なトレードオフパターン

### 5.1 ハッシュテーブル vs 線形探索

```python
# 空間 O(1), 時間 O(n) — 線形探索
def has_duplicate_space(arr):
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] == arr[j]:
                return True
    return False

# 空間 O(n), 時間 O(n) — ハッシュセット
def has_duplicate_time(arr):
    seen = set()
    for x in arr:
        if x in seen:
            return True
        seen.add(x)
    return False
```

---

## 6. 比較表

### 表1: トレードオフ手法の比較

| 手法 | 時間改善 | 空間コスト | 適用場面 |
|------|---------|-----------|---------|
| メモ化 | 指数→多項式 | O(部分問題数) | 重複部分問題 |
| ルックアップテーブル | O(f)→O(1) | O(テーブルサイズ) | 固定範囲の関数 |
| ハッシュセット | O(n²)→O(n) | O(n) | 存在判定・重複検出 |
| ブルームフィルタ | O(n)→O(k) | O(m) ≪ O(n) | 大規模存在判定 |
| ソート前処理 | O(n)→O(log n) per query | O(n log n) 時間 | 繰り返し検索 |

### 表2: ブルームフィルタと他のデータ構造

| 特性 | ハッシュセット | ブルームフィルタ | ソート済み配列 |
|------|-------------|---------------|-------------|
| 空間 | O(n) | O(m), m ≪ n | O(n) |
| 追加 | O(1) 平均 | O(k) | O(n) |
| 検索 | O(1) 平均 | O(k) | O(log n) |
| 偽陽性 | なし | あり | なし |
| 偽陰性 | なし | なし | なし |
| 削除 | O(1) | 不可（標準版） | O(n) |

---

## 7. アンチパターン

### アンチパターン1: 不要なメモ化

```python
# BAD: 重複部分問題がない再帰にメモ化
@lru_cache(maxsize=None)
def binary_search_recursive(arr_tuple, target, lo, hi):
    if lo > hi:
        return -1
    mid = (lo + hi) // 2
    if arr_tuple[mid] == target:
        return mid
    elif arr_tuple[mid] < target:
        return binary_search_recursive(arr_tuple, target, mid + 1, hi)
    else:
        return binary_search_recursive(arr_tuple, target, lo, mid - 1)

# 二分探索は各部分問題が1回しか呼ばれないためメモ化は無意味
# キャッシュの空間コスト O(log n) が無駄になる
```

### アンチパターン2: テーブルサイズを考慮しない

```python
# BAD: 範囲が広すぎるテーブル
# 2^32 = 約40億エントリ → メモリ不足
table = [0] * (2**32)  # 約 16GB のメモリが必要!

# GOOD: 分割して小さなテーブルで対応
table_8bit = [0] * 256  # 256 エントリで十分
```

---

## 8. FAQ

### Q1: メモ化とボトムアップDP（テーブル）はどちらが良いか？

**A:** メモ化（トップダウン）は必要な部分問題のみ計算するが再帰オーバーヘッドがある。ボトムアップは全部分問題を計算するがループで高速。部分問題の一部しか使わない場合はメモ化、全部使う場合はボトムアップが有利。

### Q2: ブルームフィルタの偽陽性率はどう制御する？

**A:** ビット配列サイズ m とハッシュ関数数 k を調整する。最適な k = (m/n) × ln2。n=100万、偽陽性率1%なら m ≈ 960万ビット（約1.2MB）で済む。

### Q3: 時間と空間どちらを優先すべきか？

**A:** 一般的には「時間優先」が多い。ただし、組み込みシステムやモバイルではメモリ制約が厳しいため空間優先。クラウド環境ではメモリコストが高いので、計算時間とインフラコストのバランスを考慮する。

---

## 9. まとめ

| 項目 | ポイント |
|------|---------|
| メモ化 | 重複部分問題があるとき有効。lru_cache で簡単に適用 |
| ルックアップテーブル | 小さい入力範囲の関数に有効。O(1) 参照 |
| ブルームフィルタ | 偽陽性を許容すれば空間を大幅削減 |
| ハッシュセット | 最も汎用的なトレードオフ手法 |
| 判断基準 | メモリ制約・クエリ頻度・許容誤差で選択 |

---

## 次に読むべきガイド

- [ハッシュテーブル — 衝突解決とロードファクター](../01-data-structures/03-hash-tables.md)
- [動的計画法 — メモ化とテーブルの詳細](../02-algorithms/04-dynamic-programming.md)

---

## 参考文献

1. Bloom, B.H. (1970). "Space/time trade-offs in hash coding with allowable errors." *Communications of the ACM*, 13(7), 422-426.
2. Cormen, T.H. et al. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. — 第14章「Dynamic Programming」
3. Mitzenmacher, M. & Upfal, E. (2017). *Probability and Computing* (2nd ed.). Cambridge University Press. — ブルームフィルタの確率分析
