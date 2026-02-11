# 計算量解析 — 再帰の計算量とマスター定理

> 再帰アルゴリズムの計算量を正確に求める手法を学ぶ。漸化式の立て方、マスター定理、再帰木法、置換法を体系的に解説する。

---

## この章で学ぶこと

1. **再帰の漸化式** の立て方と解法
2. **マスター定理** の3つのケースと適用条件
3. **再帰木法・置換法** による厳密な計算量導出

---

## 1. 再帰の計算量を求める手法

```
再帰の計算量解析手法:

┌──────────────────┐
│ 漸化式を立てる   │
└──────┬───────────┘
       ▼
┌──────────────────────────────────┐
│ 解法を選択                       │
│  ├─ マスター定理（定型パターン） │
│  ├─ 再帰木法（視覚的に理解）     │
│  └─ 置換法（帰納法で証明）       │
└──────────────────────────────────┘
```

---

## 2. 漸化式の立て方

### 例: マージソート

```python
def merge_sort(arr):
    """マージソート — T(n) = 2T(n/2) + O(n)"""
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])     # T(n/2)
    right = merge_sort(arr[mid:])    # T(n/2)
    return merge(left, right)        # O(n)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
```

漸化式: `T(n) = 2T(n/2) + cn`

---

## 3. マスター定理

### 3.1 一般形

```
T(n) = aT(n/b) + f(n)

  a : 再帰呼び出しの回数
  b : 問題サイズの縮小率
  f(n) : 分割・統合のコスト

  比較対象: n^(log_b(a))
```

### 3.2 三つのケース

```
ケース判定フローチャート:

  f(n) と n^(log_b(a)) を比較
         │
    ┌────┼────────────────┐
    ▼    ▼                ▼
 Case 1  Case 2         Case 3
 f(n)が   f(n)が         f(n)が
 小さい   同程度         大きい

Case 1: f(n) = O(n^(log_b(a) - ε))
  → T(n) = Θ(n^(log_b(a)))

Case 2: f(n) = Θ(n^(log_b(a)))
  → T(n) = Θ(n^(log_b(a)) · log n)

Case 3: f(n) = Ω(n^(log_b(a) + ε))
  かつ正則条件を満たす
  → T(n) = Θ(f(n))
```

### 3.3 マスター定理の適用例

```python
# 例1: マージソート T(n) = 2T(n/2) + n
# a=2, b=2, f(n)=n, n^(log_2(2)) = n^1 = n
# f(n) = Θ(n) = Θ(n^(log_b(a))) → Case 2
# T(n) = Θ(n log n)

# 例2: 二分探索 T(n) = T(n/2) + 1
# a=1, b=2, f(n)=1, n^(log_2(1)) = n^0 = 1
# f(n) = Θ(1) = Θ(n^(log_b(a))) → Case 2
# T(n) = Θ(log n)

# 例3: カラツバ乗算 T(n) = 3T(n/2) + n
# a=3, b=2, f(n)=n, n^(log_2(3)) ≈ n^1.585
# f(n) = O(n^(1.585-ε)) → Case 1
# T(n) = Θ(n^1.585)
```

---

## 4. 再帰木法

### 4.1 T(n) = 2T(n/2) + n の再帰木

```
                    n                     → コスト: n
                 /     \
             n/2         n/2              → コスト: n
            /   \       /   \
         n/4   n/4   n/4   n/4           → コスト: n
        / \   / \   / \   / \
       ... ... ... ... ... ... ...       → コスト: n
       |   |   |   |   |   |   |
       1   1   1   1   1   1   1         → コスト: n

  レベル数: log₂(n)
  各レベルのコスト: n
  合計: n × log₂(n) = O(n log n)
```

### 4.2 T(n) = 3T(n/4) + cn² の再帰木

```python
# a=3, b=4, f(n) = cn²
# n^(log_4(3)) ≈ n^0.793
# f(n) = cn² = Ω(n^(0.793+ε)) → Case 3
# T(n) = Θ(n²)

# 再帰木で確認:
# レベル k のノード数: 3^k
# レベル k の各ノードサイズ: n/4^k
# レベル k のコスト: 3^k × c(n/4^k)² = cn² × (3/16)^k
# 合計: cn² × Σ(3/16)^k = cn² × 16/13 = Θ(n²)
```

---

## 5. 置換法（帰納法）

```python
# T(n) = 2T(n/2) + n を T(n) = O(n log n) と予想

# 仮定: T(k) ≤ c·k·log(k) for all k < n

# 証明:
# T(n) = 2T(n/2) + n
#      ≤ 2·c·(n/2)·log(n/2) + n     (帰納法の仮定)
#      = c·n·(log(n) - 1) + n
#      = c·n·log(n) - c·n + n
#      = c·n·log(n) - (c-1)·n
#      ≤ c·n·log(n)                   (c ≥ 1 のとき)
#
# よって T(n) = O(n log n) ■
```

---

## 6. 一般的な再帰パターン

### 例1: 線形再帰

```python
def factorial(n):
    """T(n) = T(n-1) + O(1) → O(n)"""
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```

### 例2: 二分再帰

```python
def power(x, n):
    """T(n) = T(n/2) + O(1) → O(log n)"""
    if n == 0:
        return 1
    if n % 2 == 0:
        half = power(x, n // 2)
        return half * half
    else:
        return x * power(x, n - 1)
```

### 例3: 複数分岐再帰

```python
def fib(n):
    """T(n) = T(n-1) + T(n-2) + O(1) → O(φⁿ) ≈ O(1.618ⁿ)"""
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)
```

---

## 7. 比較表

### 表1: マスター定理の3ケース

| ケース | 条件 | 結果 | 直感 |
|--------|------|------|------|
| Case 1 | f(n) = O(n^(log_b(a)-ε)) | Θ(n^(log_b(a))) | 葉のコストが支配 |
| Case 2 | f(n) = Θ(n^(log_b(a))) | Θ(n^(log_b(a)) log n) | 各レベル均等 |
| Case 3 | f(n) = Ω(n^(log_b(a)+ε)) | Θ(f(n)) | ルートのコストが支配 |

### 表2: 代表的な漸化式と解

| 漸化式 | 解 | アルゴリズム例 |
|--------|-----|---------------|
| T(n) = T(n-1) + O(1) | O(n) | 線形走査・階乗 |
| T(n) = T(n/2) + O(1) | O(log n) | 二分探索 |
| T(n) = 2T(n/2) + O(n) | O(n log n) | マージソート |
| T(n) = 2T(n/2) + O(1) | O(n) | 二分木走査 |
| T(n) = T(n-1) + O(n) | O(n²) | 選択ソート |
| T(n) = 2T(n-1) + O(1) | O(2ⁿ) | ハノイの塔 |
| T(n) = T(n/2) + O(n) | O(n) | クイックセレクト(平均) |

---

## 8. アンチパターン

### アンチパターン1: マスター定理の適用条件を確認しない

```python
# BAD: T(n) = 2T(n/2) + n log n にマスター定理を直接適用
# f(n) = n log n, n^(log_2(2)) = n
# f(n) は n^(1+ε) より小さいが n^1 より大きい
# → Case 2 と Case 3 の間に落ちる
# → 基本のマスター定理は適用不可（拡張版が必要）
# 結果: T(n) = Θ(n log²n)
```

### アンチパターン2: 再帰の深さと呼び出し回数を混同する

```python
# BAD: 「二分再帰だから O(log n)」と誤解
def count_all(n):
    if n <= 0:
        return 0
    return 1 + count_all(n - 1) + count_all(n - 2)
# 深さは O(n) だが、呼び出し回数は O(2ⁿ)
# 深さ ≠ 計算量
```

---

## 9. FAQ

### Q1: マスター定理が使えないケースはどうする？

**A:** 再帰木法で視覚的に合計コストを計算するか、Akra-Bazzi 定理を使う。T(n) = T(n/3) + T(2n/3) + n のように分割が均等でない場合も再帰木法が有効。

### Q2: log の底は計算量に影響するか？

**A:** 漸近記法では影響しない。log_a(n) = log_b(n) / log_b(a) なので定数倍の違いしかない。ただし、底が 2 のとき O(log n) = O(log₂ n) と明示することもある。

### Q3: 再帰を反復に変換すると計算量は変わるか？

**A:** 時間計算量は通常変わらない。空間計算量は改善される場合がある（末尾再帰の最適化や、スタックを明示的に管理する場合）。ただし、末尾再帰最適化は言語による（Python は未対応）。

---

## 10. まとめ

| 項目 | ポイント |
|------|---------|
| 漸化式 | 再帰構造からコストの関係式を立てる |
| マスター定理 | T(n) = aT(n/b) + f(n) 形の定型解法 |
| 再帰木法 | 各レベルのコストを視覚化して合計する |
| 置換法 | 予想を立てて帰納法で証明 |
| 適用外のケース | Akra-Bazzi 定理、再帰木で対応 |
| log の底 | 漸近記法では無関係 |

---

## 次に読むべきガイド

- [時間空間トレードオフ — メモ化とブルームフィルタ](./02-space-time-tradeoff.md)
- [ソート — Quick/Merge/Heap の計算量比較](../02-algorithms/00-sorting.md)

---

## 参考文献

1. Cormen, T.H. et al. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. — 第4章「Divide-and-Conquer」
2. Akra, M. & Bazzi, L. (1998). "On the solution of linear recurrence equations." *Computational Optimization and Applications*, 10(2), 195-210.
3. Levitin, A. (2012). *Introduction to the Design and Analysis of Algorithms* (3rd ed.). Pearson. — 再帰の解析手法
