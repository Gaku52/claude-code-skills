# 配列と文字列 — 動的配列・文字列アルゴリズム・二次元配列

> プログラミングの最も基本的なデータ構造である配列と文字列を深く理解し、動的配列の仕組み、文字列操作の計算量、二次元配列の走査パターンを学ぶ。

---

## この章で学ぶこと

1. **静的配列と動的配列** のメモリ構造と計算量の違い
2. **文字列アルゴリズム** の基本パターン（反転、回文、アナグラム）
3. **二次元配列** の走査・回転・スパイラル出力

---

## 1. 配列の基本

### 1.1 メモリレイアウト

```
静的配列 int arr[5]:
  アドレス:  0x100  0x104  0x108  0x10C  0x110
           ┌──────┬──────┬──────┬──────┬──────┐
           │  10  │  20  │  30  │  40  │  50  │
           └──────┴──────┴──────┴──────┴──────┘
  インデックス: [0]    [1]    [2]    [3]    [4]

  arr[i] のアドレス = base + i × sizeof(int)
  → O(1) のランダムアクセス
```

### 1.2 動的配列の成長戦略

```
Python list / Java ArrayList の内部動作:

容量 2: [A][B]          使用率 100%
  ↓ append(C) → リサイズ!
容量 4: [A][B][C][ ]    使用率 75%
  ↓ append(D)
容量 4: [A][B][C][D]    使用率 100%
  ↓ append(E) → リサイズ!
容量 8: [A][B][C][D][E][ ][ ][ ]  使用率 62.5%

成長率: 通常 1.5x〜2x
append の償却計算量: O(1)
```

---

## 2. コード例

### 例1: 二つの配列のマージ（ソート済み）

```python
def merge_sorted(a, b):
    """ソート済み配列のマージ — O(n+m)"""
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            result.append(a[i]); i += 1
        else:
            result.append(b[j]); j += 1
    result.extend(a[i:])
    result.extend(b[j:])
    return result
```

### 例2: 配列の回転（左回転）

```python
def rotate_left(arr, k):
    """配列を左に k 回転 — O(n) 時間, O(1) 空間"""
    n = len(arr)
    k = k % n
    def reverse(lo, hi):
        while lo < hi:
            arr[lo], arr[hi] = arr[hi], arr[lo]
            lo += 1; hi -= 1
    reverse(0, k - 1)
    reverse(k, n - 1)
    reverse(0, n - 1)
    return arr
```

### 例3: 文字列の回文判定

```python
def is_palindrome(s):
    """回文判定（英数字のみ）— O(n)"""
    s = ''.join(c.lower() for c in s if c.isalnum())
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1; right -= 1
    return True
```

### 例4: アナグラム判定

```python
from collections import Counter

def is_anagram(s, t):
    """アナグラム判定 — O(n)"""
    return Counter(s) == Counter(t)
```

### 例5: 二次元配列の90度回転

```python
def rotate_90(matrix):
    """N×N 行列を時計回りに90度回転 — O(n²)"""
    n = len(matrix)
    # 転置
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    # 各行を反転
    for row in matrix:
        row.reverse()
    return matrix
```

### 例6: スパイラル走査

```python
def spiral_order(matrix):
    """行列のスパイラル走査 — O(m×n)"""
    result = []
    if not matrix:
        return result
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1
    while top <= bottom and left <= right:
        for j in range(left, right + 1):
            result.append(matrix[top][j])
        top += 1
        for i in range(top, bottom + 1):
            result.append(matrix[i][right])
        right -= 1
        if top <= bottom:
            for j in range(right, left - 1, -1):
                result.append(matrix[bottom][j])
            bottom -= 1
        if left <= right:
            for i in range(bottom, top - 1, -1):
                result.append(matrix[i][left])
            left += 1
    return result
```

---

## 3. 二次元配列の走査パターン

```
行優先走査:           列優先走査:           対角線走査:
→ → → →             ↓   ↓   ↓   ↓       ╱ ╱ ╱ ╱
→ → → →             ↓   ↓   ↓   ↓     ╱ ╱ ╱ ╱
→ → → →             ↓   ↓   ↓   ↓   ╱ ╱ ╱ ╱
→ → → →             ↓   ↓   ↓   ↓ ╱ ╱ ╱ ╱

スパイラル走査:       ジグザグ走査:
→ → → ↓             → → → ↓
↑ → ↓ ↓                 ← ← ←
↑ ↑ ← ↓             → → → ↓
↑ ← ← ←                 ← ← ←
```

---

## 4. 二つのポインタ技法

```
Two Pointers パターン:

  ソート済み配列で和が target のペアを探す:

  arr = [1, 3, 5, 7, 9, 11]   target = 12

  ステップ1: left=0, right=5  → 1+11=12 → 発見!
             L                       R

  ステップ2 (もし合計が小さい場合): left を右へ
  ステップ3 (もし合計が大きい場合): right を左へ
```

```python
def two_sum_sorted(arr, target):
    """ソート済み配列で和が target のペア — O(n)"""
    left, right = 0, len(arr) - 1
    while left < right:
        s = arr[left] + arr[right]
        if s == target:
            return [left, right]
        elif s < target:
            left += 1
        else:
            right -= 1
    return []
```

---

## 5. 比較表

### 表1: 配列操作の計算量

| 操作 | 静的配列 | 動的配列 (Python list) | 連結リスト |
|------|---------|----------------------|-----------|
| アクセス [i] | O(1) | O(1) | O(n) |
| 先頭挿入 | O(n) | O(n) | O(1) |
| 末尾挿入 | - | O(1) 償却 | O(1)* |
| 中間挿入 | O(n) | O(n) | O(1)** |
| 先頭削除 | O(n) | O(n) | O(1) |
| 末尾削除 | O(1) | O(1) | O(n)* |
| 探索 | O(n) | O(n) | O(n) |

*末尾ポインタがある場合 **挿入位置が既知の場合

### 表2: 文字列操作の計算量（Python）

| 操作 | 計算量 | 備考 |
|------|--------|------|
| s[i] | O(1) | インデックスアクセス |
| s + t | O(n+m) | 新しい文字列を生成 |
| s.find(t) | O(n×m) | 最悪ケース |
| s.join(list) | O(合計長) | 連結の推奨方法 |
| s[:k] | O(k) | スライスはコピー |
| len(s) | O(1) | 長さは保存済み |

---

## 6. アンチパターン

### アンチパターン1: ループ内の文字列連結

```python
# BAD: O(n²) — 毎回新しい文字列を生成
result = ""
for s in strings:
    result += s  # O(len(result)) のコピーが毎回発生

# GOOD: O(n) — join を使う
result = "".join(strings)
```

### アンチパターン2: 配列の先頭への頻繁な挿入

```python
# BAD: O(n) × m 回 = O(nm)
arr = []
for x in data:
    arr.insert(0, x)  # 毎回全要素をシフト

# GOOD: collections.deque を使う — O(1) の先頭挿入
from collections import deque
d = deque()
for x in data:
    d.appendleft(x)
```

---

## 7. FAQ

### Q1: Python の list はどの程度メモリを余分に使うか？

**A:** Python の list は要素のポインタ配列で、通常 12.5%〜100% の余分な容量を持つ。成長率は約 1.125 倍で、Java の ArrayList (1.5 倍) や C++ の vector (2 倍) より控えめ。

### Q2: 文字列は immutable だと何が嬉しいのか？

**A:** (1) ハッシュ値をキャッシュできる（dict のキーに使える）、(2) スレッドセーフ、(3) インターンによるメモリ節約。ただし変更のたびに新しいオブジェクトが生成されるコストがある。

### Q3: numpy 配列と Python list の違いは？

**A:** numpy 配列は連続メモリに値を直接格納（C 配列に近い）。Python list はポインタ配列。numpy は数値計算で 10-100 倍高速だが、サイズ変更は苦手。

---

## 8. まとめ

| 項目 | ポイント |
|------|---------|
| 静的配列 | O(1) アクセス、固定サイズ |
| 動的配列 | append は償却 O(1)、成長率で空間効率が変わる |
| 文字列 | immutable の言語が多い。連結は join を使う |
| 二次元配列 | 走査パターン（行/列/対角線/スパイラル）を知る |
| Two Pointers | ソート済み配列で O(n) に改善する頻出テクニック |

---

## 次に読むべきガイド

- [連結リスト — 単方向/双方向とフロイドのアルゴリズム](./01-linked-lists.md)
- [ハッシュテーブル — ハッシュ関数と衝突解決](./03-hash-tables.md)

---

## 参考文献

1. Cormen, T.H. et al. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. — 配列とソートの基礎
2. Sedgewick, R. & Wayne, K. (2011). *Algorithms* (4th ed.). Addison-Wesley. — 配列の実装詳細
3. Python Documentation. "TimeComplexity." — https://wiki.python.org/moin/TimeComplexity
