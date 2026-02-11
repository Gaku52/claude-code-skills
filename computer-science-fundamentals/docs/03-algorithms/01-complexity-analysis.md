# 計算量解析（Big-O記法）

> 「動く」コードと「速い」コードの違いは計算量にある。O(n²) と O(n log n) の差は、データが大きくなるほど致命的になる。

## この章で学ぶこと

- [ ] Big-O記法を使って時間計算量を表現できる
- [ ] 主要な計算量クラスを直感的に理解する
- [ ] コードを見て計算量を分析できる
- [ ] 空間計算量（メモリ使用量）も評価できる

## 前提知識

- アルゴリズムの基礎 → 参照: [[00-what-is-algorithm.md]]

---

## 1. Big-O記法

### 1.1 定義と直感

```
Big-O記法: 入力サイズ n に対する実行時間の「成長率」を表す

  厳密な定義:
    f(n) = O(g(n)) ⟺ ∃c > 0, ∃n₀ > 0 such that
    ∀n ≥ n₀: f(n) ≤ c × g(n)

  直感:
  「n が十分大きい時、f(n) は g(n) の定数倍以下で抑えられる」

  例:
    3n² + 5n + 100 = O(n²)
    → n が大きくなると n² が支配的
    → 5n も 100 も誤差の範囲

  ルール:
  1. 定数倍は無視: 5n → O(n)
  2. 低次項は無視: n² + n → O(n²)
  3. 底は無視:     log₂n = log₁₀n × 定数 → O(log n)
```

### 1.2 主要な計算量クラス

```
計算量クラスの比較（n = 入力サイズ）:

  ┌──────────────┬──────────────┬──────────────────────────┐
  │ 記法         │ 名前         │ n=100での演算数          │
  ├──────────────┼──────────────┼──────────────────────────┤
  │ O(1)         │ 定数時間     │ 1                        │
  │ O(log n)     │ 対数時間     │ 7                        │
  │ O(n)         │ 線形時間     │ 100                      │
  │ O(n log n)   │ 線形対数時間 │ 700                      │
  │ O(n²)        │ 二乗時間     │ 10,000                   │
  │ O(n³)        │ 三乗時間     │ 1,000,000                │
  │ O(2ⁿ)        │ 指数時間     │ 1.27 × 10³⁰            │
  │ O(n!)        │ 階乗時間     │ 9.33 × 10¹⁵⁷           │
  └──────────────┴──────────────┴──────────────────────────┘

  n=1,000,000（100万）での実行時間（1命令=1ns）:
  O(1):        0.001 μs     瞬時
  O(log n):    0.02 μs      瞬時
  O(n):        1 ms          瞬時
  O(n log n):  20 ms         一瞬
  O(n²):       16 分         コーヒー1杯
  O(n³):       31.7 年       人生が終わる
  O(2ⁿ):       宇宙の寿命を超える

  → O(n²) と O(n log n) の差は、n=100万で48,000倍！
```

### 1.3 成長率のグラフ（ASCII）

```
  演算数
  │
  │                                          ╱ O(2ⁿ)
  │                                        ╱
  │                                      ╱
  │                                    ╱
  │                               ╱──── O(n²)
  │                          ╱───
  │                    ╱────
  │              ╱────
  │         ╱───────────────── O(n log n)
  │    ╱──────────────────── O(n)
  │ ╱
  │──────────────────────── O(log n)
  │━━━━━━━━━━━━━━━━━━━━━━━ O(1)
  └───────────────────────────── n
```

---

## 2. 計算量の求め方

### 2.1 基本パターン

```python
# パターン1: O(1) — 定数時間
def get_first(arr):
    return arr[0]  # インデックスアクセスは O(1)

# パターン2: O(n) — 線形時間
def sum_array(arr):
    total = 0
    for x in arr:      # n回ループ
        total += x     # O(1) の操作
    return total
# → O(n)

# パターン3: O(n²) — 二重ループ
def has_duplicate(arr):
    n = len(arr)
    for i in range(n):       # n回
        for j in range(i+1, n):  # 最大 n-1 回
            if arr[i] == arr[j]:
                return True
    return False
# → O(n × n) = O(n²)

# パターン4: O(log n) — 半分に分割
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1     # 探索範囲が半分に
        else:
            right = mid - 1    # 探索範囲が半分に
    return -1
# → 毎回半分 → log₂(n) 回で終了 → O(log n)

# パターン5: O(n log n) — 分割統治
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])    # T(n/2)
    right = merge_sort(arr[mid:])   # T(n/2)
    return merge(left, right)       # O(n)
# T(n) = 2T(n/2) + O(n) → O(n log n)
```

### 2.2 再帰の計算量 — マスター定理

```
マスター定理:
  T(n) = a × T(n/b) + O(n^d) の形の再帰に対して:

  Case 1: d < log_b(a)  → T(n) = O(n^(log_b(a)))
  Case 2: d = log_b(a)  → T(n) = O(n^d × log n)
  Case 3: d > log_b(a)  → T(n) = O(n^d)

  適用例:
  ─────────────────────────────────────────────────
  マージソート: T(n) = 2T(n/2) + O(n)
    a=2, b=2, d=1 → log₂(2)=1=d → Case 2 → O(n log n) ✓

  二分探索: T(n) = T(n/2) + O(1)
    a=1, b=2, d=0 → log₂(1)=0=d → Case 2 → O(log n) ✓

  ストラッセンの行列乗算: T(n) = 7T(n/2) + O(n²)
    a=7, b=2, d=2 → log₂(7)≈2.81 > 2 → Case 1 → O(n^2.81) ✓

  線形探索（再帰版）: T(n) = T(n-1) + O(1)
    → マスター定理は適用不可（n/b ではなく n-1）
    → 直接展開: T(n) = T(n-1) + 1 = T(n-2) + 2 = ... = O(n)
```

### 2.3 償却計算量（Amortized Analysis）

```python
# 動的配列（Python list）の append は O(1)?

# 実際の動作:
# - 容量に余裕がある場合: O(1)
# - 容量が足りない場合: O(n)（新しい配列にコピー）

# 償却分析:
# n回の append での総コスト:
# 1, 1, 1, ..., 1, n, 1, 1, ..., 1, 2n, ...
#                  ↑ リサイズ          ↑ リサイズ
#
# 容量を2倍に拡張する場合:
# 総コスト = n + (1 + 2 + 4 + ... + n) = n + 2n = 3n
# 1回あたり = 3n / n = O(1)（償却）
#
# → 個々の操作はO(1)〜O(n)だが、
#   n回の操作全体で見るとO(n) → 1回あたりO(1)

# 他の償却O(1)の例:
# - ハッシュテーブルの挿入（リハッシュ時にO(n)）
# - Union-Find の操作（経路圧縮+ランク付き）
```

---

## 3. 空間計算量

### 3.1 メモリ使用量の分析

```python
# 空間計算量: アルゴリズムが使用する追加メモリ量

# O(1) 空間: 入力以外に固定量のメモリ
def find_max(arr):
    max_val = arr[0]  # 変数1つだけ
    for x in arr:
        if x > max_val:
            max_val = x
    return max_val
# 空間: O(1) — max_val のみ

# O(n) 空間: 入力に比例するメモリ
def reverse_array(arr):
    result = []        # 新しい配列
    for x in reversed(arr):
        result.append(x)
    return result
# 空間: O(n) — result 配列

# O(n) 空間 (再帰のスタック)
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
# 空間: O(n) — 再帰の深さが n（スタックフレーム n 個）

# O(log n) 空間
# マージソートの再帰の深さ: O(log n)
# ただし配列のコピーで O(n) 空間が必要

# 時間と空間のトレードオフ:
# 例: 重複チェック
# 方法1: O(n²) 時間, O(1) 空間 — 全ペア比較
# 方法2: O(n) 時間, O(n) 空間 — ハッシュセット使用
# → メモリを使って速度を買う
```

---

## 4. 実務での計算量

### 4.1 制約から計算量を逆算する

```
競技プログラミング / 実務での目安:

  1秒あたりの処理可能な演算数: 約 10^8 〜 10^9

  ┌──────────┬──────────────────┬──────────────────┐
  │ データ量  │ 許容計算量        │ 使えるアルゴリズム │
  ├──────────┼──────────────────┼──────────────────┤
  │ n ≤ 10   │ O(n!) ← OK      │ 全探索            │
  │ n ≤ 20   │ O(2ⁿ) ← OK     │ ビット全探索       │
  │ n ≤ 500  │ O(n³) ← OK     │ 3重ループ          │
  │ n ≤ 5000 │ O(n²) ← OK     │ 2重ループ          │
  │ n ≤ 10⁶  │ O(n log n) ← OK│ ソート, 二分探索   │
  │ n ≤ 10⁸  │ O(n) ← OK      │ 線形走査           │
  │ n ≤ 10¹⁸ │ O(log n) ← OK  │ 二分探索, 数学     │
  └──────────┴──────────────────┴──────────────────┘

  実務のWebアプリケーション:
  - APIレスポンス: 100ms以内 → O(n log n) まで（n=数万程度）
  - バッチ処理: 分〜時間 → O(n²) でも許容される場合あり
  - リアルタイム: 16ms以内(60fps) → O(n) 以下が望ましい
```

### 4.2 よくある最適化パターン

```python
# O(n²) → O(n) に改善する定番パターン

# パターン1: ハッシュマップで探索を O(1) に
# 問題: 配列から合計が target になるペアを見つける

# ❌ O(n²): 全ペア探索
def two_sum_brute(nums, target):
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]

# ✅ O(n): ハッシュマップ
def two_sum_hash(nums, target):
    seen = {}  # 値 → インデックス
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:  # O(1) ルックアップ
            return [seen[complement], i]
        seen[num] = i

# パターン2: ソートして二分探索
# 問題: ソート済み配列で target 以上の最小値を見つける

# ❌ O(n): 線形探索
def find_ceiling_linear(arr, target):
    for x in arr:
        if x >= target:
            return x

# ✅ O(log n): 二分探索
import bisect
def find_ceiling_binary(arr, target):
    idx = bisect.bisect_left(arr, target)
    return arr[idx] if idx < len(arr) else None

# パターン3: スライディングウィンドウ
# 問題: 長さ k の連続部分配列の最大合計

# ❌ O(nk): 毎回合計を計算
def max_sum_brute(arr, k):
    max_sum = 0
    for i in range(len(arr) - k + 1):
        max_sum = max(max_sum, sum(arr[i:i+k]))
    return max_sum

# ✅ O(n): ウィンドウをスライド
def max_sum_sliding(arr, k):
    window = sum(arr[:k])
    max_sum = window
    for i in range(k, len(arr)):
        window += arr[i] - arr[i-k]  # 追加と削除
        max_sum = max(max_sum, window)
    return max_sum
```

---

## 5. Big-O以外の漸近記法

### 5.1 Ω記法とΘ記法

```
3つの漸近記法:

  O（Big-O）:  上界 — 「最悪でもこの程度」
    f(n) = O(g(n)) → f(n) ≤ c × g(n)

  Ω（Big-Omega）: 下界 — 「少なくともこの程度」
    f(n) = Ω(g(n)) → f(n) ≥ c × g(n)

  Θ（Big-Theta）: 厳密な境界 — 「ちょうどこの程度」
    f(n) = Θ(g(n)) → f(n) = O(g(n)) かつ f(n) = Ω(g(n))

  例:
  比較ベースのソートの下界:  Ω(n log n)
  → どんなに工夫しても O(n log n) 未満にはできない

  マージソート: Θ(n log n) — 最悪・平均・最良が全て同じ
  クイックソート: O(n²) 最悪、Θ(n log n) 平均

  実務では:
  - Big-O（上界）が最も重要 → 最悪のケースを保証
  - 平均計算量も重要 → 実際のパフォーマンス
  - 最良計算量はあまり重要でない → 「運が良い場合」
```

---

## 6. 実践演習

### 演習1: 計算量の判定（基礎）
以下のコードの時間計算量と空間計算量を求めよ:
1. 配列の全ての2要素の組を出力するコード
2. 再帰的な二分探索
3. フィボナッチ数列の再帰的計算（メモ化なし vs あり）

### 演習2: 計算量の改善（応用）
O(n³) のアルゴリズム（3つの配列からの3Sum問題）を O(n²) に改善せよ。

### 演習3: 償却分析（発展）
スタック2つを使ったキューの実装で、各操作が償却 O(1) であることを証明せよ。

---

## FAQ

### Q1: Big-O は実行時間の正確な予測に使えますか？
**A**: いいえ。Big-O は「成長率」を表すだけで、定数倍の違いを無視する。O(n) でも定数が大きければ O(n log n) より遅い場合がある。実際のパフォーマンスはキャッシュ効率、分岐予測、メモリアクセスパターンなど多くの要因に依存する。

### Q2: O(1) は常に速いですか？
**A**: 必ずしも。O(1) は「入力サイズに依存しない」だけで、O(1) = 100万回の定数操作もあり得る。また、ハッシュテーブルの O(1) は「期待値」であり、最悪ケースは O(n)。実測が重要。

### Q3: 計算量を改善すべきか、定数倍を改善すべきか？
**A**: まず計算量の改善。O(n²)→O(n log n) は劇的。定数倍の改善（キャッシュ最適化等）は計算量が最適になった後で検討。ただし n が小さい場合は定数倍が支配的になることもある。

---

## まとめ

| 概念 | ポイント |
|------|---------|
| Big-O | 実行時間の成長率。定数倍と低次項を無視 |
| 主要クラス | O(1) < O(log n) < O(n) < O(n log n) < O(n²) < O(2ⁿ) |
| 求め方 | ループ数, 再帰の深さ, マスター定理 |
| 空間計算量 | 追加メモリ。時間との「トレードオフ」 |
| 実務 | n≤10⁶ → O(n log n)以下、APIは100ms以内 |

---

## 次に読むべきガイド
→ [[02-sorting-algorithms.md]] — ソートアルゴリズム

---

## 参考文献
1. Cormen, T. H. et al. "Introduction to Algorithms (CLRS)." Chapter 3: Growth of Functions.
2. Skiena, S. S. "The Algorithm Design Manual." Chapter 2: Algorithm Analysis.
3. Sedgewick, R. & Wayne, K. "Algorithms." 4th Edition, Chapter 1.4.
