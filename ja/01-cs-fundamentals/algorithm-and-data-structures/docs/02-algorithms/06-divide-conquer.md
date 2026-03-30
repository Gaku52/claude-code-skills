# 分割統治法（Divide and Conquer）

> 問題を小さな部分問題に分割し、再帰的に解いて統合する設計手法を、マージソート・大数乗算・最近接点対・Strassen 行列乗算を通じて理解する

## この章で学ぶこと

1. **分割統治法の3ステップ**（分割・統治・統合）を理解し、再帰的にアルゴリズムを設計できる
2. **マスター定理**で分割統治アルゴリズムの計算量を正確に解析できる
3. **Karatsuba 乗算・最近接点対・Strassen 行列乗算**など高度な分割統治アルゴリズムを実装できる
4. **分割統治と DP・貪欲法の違い**を理解し、問題に応じて適切な手法を選択できる
5. **再帰木の描画**と**漸化式の導出**を通じて、計算量を直感的に把握できる


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解
- [貪欲法（Greedy Algorithm）](./05-greedy.md) の内容を理解していること

---

## 1. 分割統治法の原理

### 1.1 3ステップのフレームワーク

分割統治法（Divide and Conquer）は、アルゴリズム設計における最も強力なパラダイムの一つである。その名の通り、問題を「分割」し、「統治」し、「統合」するという3つのステップで構成される。

```
┌──────────────────────────────────────────────────────────────┐
│              分割統治法の3ステップ                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Step 1. 分割 (Divide)                                       │
│     問題を同じ種類のより小さな部分問題に分割する                 │
│     - 通常は2つに分割（二分法）                                │
│     - 3つ以上に分割する場合もある（Karatsuba: 3分割）          │
│                                                              │
│  Step 2. 統治 (Conquer)                                      │
│     部分問題を再帰的に解く                                     │
│     - 部分問題が十分小さければ直接解く（基底条件 / base case）  │
│     - 基底条件の設計がアルゴリズム全体の正しさに直結する        │
│                                                              │
│  Step 3. 統合 (Combine)                                      │
│     部分問題の解を組み合わせて元の問題の解を構成する            │
│     - この統合ステップの効率が全体の計算量を左右する            │
│     - マージソートの merge、最近接点対のストリップ処理など      │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│  DP との決定的な違い:                                          │
│  DP       → 部分問題が重複する → 結果をキャッシュして再利用     │
│  分割統治  → 部分問題が独立   → 各部分問題をそのまま再帰で解く  │
│                                                              │
│  ただし例外もある:                                             │
│  - フィボナッチ数列の再帰木では部分問題が大量に重複             │
│    → 分割統治ではなく DP（メモ化再帰）で解くべき               │
│  - 行列連鎖乗算も部分問題が重複 → DP が適切                   │
└──────────────────────────────────────────────────────────────┘
```

### 1.2 再帰構造の可視化

分割統治のアルゴリズムは、再帰木（recursion tree）として可視化できる。再帰木の各ノードは1つの部分問題を表し、そのノードの子は分割された部分問題を表す。

```
分割統治の再帰木（マージソートの場合）:

深さ0:           [38, 27, 43, 3, 9, 82, 10]         ← 問題サイズ n
                 /                          \
深さ1:    [38, 27, 43]                [3, 9, 82, 10]  ← サイズ n/2 が2つ
           /       \                   /          \
深さ2:  [38]    [27, 43]          [3, 9]      [82, 10] ← サイズ n/4 が4つ
                /     \           /    \       /     \
深さ3:        [27]   [43]       [3]   [9]   [82]   [10] ← 基底条件

                    ↓ 統合（マージ）フェーズ ↓

深さ3:        [27]   [43]       [3]   [9]   [82]   [10]
                \     /           \    /       \     /
深さ2:       [27, 43]           [3, 9]       [10, 82]
           \       /                   \          /
深さ1:    [27, 38, 43]           [3, 9, 10, 82]
                 \                          /
深さ0:          [3, 9, 10, 27, 38, 43, 82]            ← 最終結果

再帰の深さ: O(log n)
各深さでの合計仕事量: O(n)（マージ処理）
全体の計算量: O(n) × O(log n) = O(n log n)
```

### 1.3 分割統治法の歴史的背景

分割統治法は古くからある発想である。その名称は政治学の「分割統治」（divide et impera）に由来する。アルゴリズム設計の文脈では、John von Neumann が 1945 年にマージソートを発明した時点まで遡ることができる。

アルゴリズム理論において分割統治法が体系的に研究されるようになったのは 1960 年代以降である。Karatsuba が 1962 年に大数乗算の高速アルゴリズムを発表し、Strassen が 1969 年に行列乗算の高速化を示したことで、分割統治法が計算量を根本的に改善できる強力な道具であることが認識された。

### 1.4 分割統治のテンプレート

あらゆる分割統治アルゴリズムに共通する骨格を Python のテンプレートとして示す。

```python
from typing import TypeVar, List, Callable, Any

T = TypeVar('T')

def divide_and_conquer(
    problem: T,
    base_case: Callable[[T], bool],
    solve_base: Callable[[T], Any],
    divide: Callable[[T], List[T]],
) -> Any:
    """分割統治法の汎用テンプレート

    Args:
        problem: 解くべき問題
        base_case: 基底条件の判定関数
        solve_base: 基底条件を直接解く関数
        divide: 問題を部分問題に分割する関数
        combine: 部分問題の解を統合する関数

    Returns:
        問題の解
    """
    # 基底条件: 問題が十分小さければ直接解く
    if base_case(problem):
        return solve_base(problem)

    # 分割: 問題を小さな部分問題に分割
    subproblems = divide(problem)

    # 統治: 各部分問題を再帰的に解く
    subsolutions = [
        divide_and_conquer(sub, base_case, solve_base, divide, combine)
        for sub in subproblems
    ]

    # 統合: 部分問題の解を組み合わせる
    return combine(subsolutions)


# --- テンプレートの使用例: 配列の最大値を求める ---
def find_max(arr: list) -> int:
    """分割統治で配列の最大値を求める"""
    return divide_and_conquer(
        problem=arr,
        base_case=lambda a: len(a) <= 1,
        solve_base=lambda a: a[0] if a else float('-inf'),
        divide=lambda a: [a[:len(a)//2], a[len(a)//2:]],
        combine=lambda results: max(results)
    )


# 動作確認
data = [3, 7, 2, 9, 1, 8, 5, 4, 6]
print(find_max(data))  # 9
```

---

## 2. マージソート -- 分割統治の教科書的な例

### 2.1 アルゴリズムの詳細

マージソート（Merge Sort）は、分割統治法を学ぶ上で最も重要なアルゴリズムである。John von Neumann が 1945 年に発明し、以来コンピュータサイエンスの教科書における分割統治法の代表例として位置づけられている。

**特長:**
- 安定ソート（同値要素の順序が保持される）
- 最悪計算量が O(n log n) で保証される（クイックソートは最悪 O(n^2)）
- 外部ソート（メモリに収まらないデータのソート）に適している
- 連結リストのソートに適している（追加メモリがほぼ不要）

```python
def merge_sort(arr: list) -> list:
    """マージソート - O(n log n)

    分割: 配列を半分に分割する
    統治: 各半分を再帰的にソートする
    統合: 2つのソート済み配列をマージする

    安定ソートであり、最悪でも O(n log n) を保証する。
    追加メモリ O(n) が必要。
    """
    # 基底条件: 要素が0個または1個なら既にソート済み
    if len(arr) <= 1:
        return arr

    # 分割: 中央で2つに分ける
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])     # 統治（左半分を再帰ソート）
    right = merge_sort(arr[mid:])    # 統治（右半分を再帰ソート）

    # 統合: 2つのソート済み配列をマージ
    return merge(left, right)


def merge(left: list, right: list) -> list:
    """2つのソート済み配列をマージする - O(n)

    両方の配列の先頭を比較し、小さい方を結果に追加する。
    どちらかが空になったら、残りをそのまま追加する。
    """
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:  # <= で安定性を保証
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    # 残りの要素を追加
    result.extend(left[i:])
    result.extend(right[j:])
    return result


# --- 動作確認 ---
data = [38, 27, 43, 3, 9, 82, 10]
sorted_data = merge_sort(data)
print(f"入力: {data}")
print(f"出力: {sorted_data}")
# 入力: [38, 27, 43, 3, 9, 82, 10]
# 出力: [3, 9, 10, 27, 38, 43, 82]

# 安定性の確認（同値要素の順序保持）
pairs = [(3, 'a'), (1, 'b'), (3, 'c'), (2, 'd'), (1, 'e')]
sorted_pairs = merge_sort(pairs)  # タプルの辞書順比較を利用
print(f"安定ソート: {sorted_pairs}")
# [(1, 'b'), (1, 'e'), (2, 'd'), (3, 'a'), (3, 'c')]
# (1,'b') が (1,'e') の前、(3,'a') が (3,'c') の前 → 安定
```

### 2.2 In-place マージソート

標準のマージソートは O(n) の追加メモリを必要とする。メモリ使用量を抑える工夫として、補助配列を1つだけ確保するバージョンを示す。

```python
def merge_sort_inplace(arr: list) -> None:
    """補助配列を1つだけ使うマージソート（in-place に近い版）

    元の配列を直接変更する。
    補助配列 aux を一度だけ確保し、毎回のマージで再利用する。
    """
    if len(arr) <= 1:
        return

    aux = [0] * len(arr)  # 補助配列を一度だけ確保

    def _sort(lo: int, hi: int) -> None:
        """arr[lo..hi] をソートする"""
        if lo >= hi:
            return

        mid = (lo + hi) // 2
        _sort(lo, mid)       # 左半分をソート
        _sort(mid + 1, hi)   # 右半分をソート

        # 最適化: 既にソート済みならマージをスキップ
        if arr[mid] <= arr[mid + 1]:
            return

        _merge(lo, mid, hi)

    def _merge(lo: int, mid: int, hi: int) -> None:
        """arr[lo..mid] と arr[mid+1..hi] をマージする"""
        # 補助配列にコピー
        for k in range(lo, hi + 1):
            aux[k] = arr[k]

        i, j = lo, mid + 1
        for k in range(lo, hi + 1):
            if i > mid:
                arr[k] = aux[j]; j += 1
            elif j > hi:
                arr[k] = aux[i]; i += 1
            elif aux[j] < aux[i]:
                arr[k] = aux[j]; j += 1
            else:
                arr[k] = aux[i]; i += 1

    _sort(0, len(arr) - 1)


# 動作確認
data = [38, 27, 43, 3, 9, 82, 10]
merge_sort_inplace(data)
print(data)  # [3, 9, 10, 27, 38, 43, 82]
```

### 2.3 ボトムアップ・マージソート

再帰を使わないマージソートも存在する。サイズ 1 のブロックから始めて、隣接するブロックをマージしていく反復的な手法である。

```python
def merge_sort_bottom_up(arr: list) -> list:
    """ボトムアップ・マージソート（非再帰版）

    サイズ 1 → 2 → 4 → 8 → ... と倍増しながらマージしていく。
    再帰のオーバーヘッドがなく、スタックオーバーフローの心配もない。
    """
    n = len(arr)
    if n <= 1:
        return arr[:]

    result = arr[:]
    width = 1  # 現在のブロックサイズ

    while width < n:
        for start in range(0, n, 2 * width):
            mid = min(start + width, n)
            end = min(start + 2 * width, n)

            left = result[start:mid]
            right = result[mid:end]
            merged = merge(left, right)  # 上で定義した merge 関数を使用
            result[start:start + len(merged)] = merged

        width *= 2

    return result


# 動作確認
data = [5, 3, 8, 1, 9, 2, 7, 4, 6]
print(merge_sort_bottom_up(data))  # [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

### 2.4 マージソートのトレース

マージソートの動作を段階的にトレースし、分割と統合の過程を確認する。

```
入力: [38, 27, 43, 3, 9, 82, 10]

=== 分割フェーズ ===
[38, 27, 43, 3, 9, 82, 10]
├── [38, 27, 43]
│   ├── [38]            ← 基底条件
│   └── [27, 43]
│       ├── [27]        ← 基底条件
│       └── [43]        ← 基底条件
└── [3, 9, 82, 10]
    ├── [3, 9]
    │   ├── [3]         ← 基底条件
    │   └── [9]         ← 基底条件
    └── [82, 10]
        ├── [82]        ← 基底条件
        └── [10]        ← 基底条件

=== 統合フェーズ（ボトムアップ） ===
merge([27], [43])       → [27, 43]
merge([38], [27,43])    → [27, 38, 43]
merge([3], [9])         → [3, 9]
merge([82], [10])       → [10, 82]
merge([3,9], [10,82])   → [3, 9, 10, 82]
merge([27,38,43], [3,9,10,82]) → [3, 9, 10, 27, 38, 43, 82]

結果: [3, 9, 10, 27, 38, 43, 82]
```

---

## 3. マスター定理 -- 計算量解析の決定版

### 3.1 定理の定義

分割統治アルゴリズムの計算量は、漸化式（recurrence relation）で表現される。マスター定理（Master Theorem）は、特定の形式の漸化式を直接的に解くための定理である。

```
┌──────────────────────────────────────────────────────────────────┐
│                    マスター定理 (Master Theorem)                   │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  漸化式: T(n) = a * T(n/b) + O(n^d)                             │
│                                                                  │
│  パラメータ:                                                      │
│    a : 部分問題の数（a >= 1）                                     │
│    b : 分割比率 / 各部分問題のサイズ = n/b（b > 1）              │
│    d : 統合コストの指数（d >= 0）                                 │
│                                                                  │
│  比較対象: c = log_b(a)  （再帰の「重さ」を表す）                │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ ケース1: d < log_b(a) の場合                                │ │
│  │   → T(n) = O(n^{log_b(a)})                                 │ │
│  │   → 再帰呼び出しが支配的（葉の仕事量が多い）               │ │
│  │                                                             │ │
│  │ ケース2: d = log_b(a) の場合                                │ │
│  │   → T(n) = O(n^d * log n)                                  │ │
│  │   → 再帰と統合がバランス（各レベルの仕事量が均等）          │ │
│  │                                                             │ │
│  │ ケース3: d > log_b(a) の場合                                │ │
│  │   → T(n) = O(n^d)                                          │ │
│  │   → 統合ステップが支配的（根の仕事量が多い）               │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  直感的な理解:                                                    │
│    - log_b(a) は再帰木の葉の数の増加率を表す                     │
│    - d は各レベルの仕事量の増加率を表す                           │
│    - この2つを比較して、どちらが支配的かを判断する               │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 再帰木による直感的理解

マスター定理の3つのケースを再帰木で理解する。

```
再帰木の構造 (T(n) = a * T(n/b) + O(n^d)):

深さ0:   n^d の仕事量              ← 1個の問題
          |
深さ1:   a 個の (n/b)^d の仕事量   ← a 個の部分問題
          |
深さ2:   a^2 個の (n/b^2)^d の仕事量
          |
  ...     ...
          |
深さk:   a^k 個の (n/b^k)^d の仕事量
          |
深さ log_b(n): a^{log_b(n)} = n^{log_b(a)} 個の O(1) の仕事量

各深さの合計仕事量:

  深さ k の合計 = a^k * (n/b^k)^d = n^d * (a / b^d)^k

  比率 r = a / b^d を考える:
    - r > 1 (つまり d < log_b(a)): 深くなるほど仕事量が増える → 葉が支配
    - r = 1 (つまり d = log_b(a)): 各深さで同じ仕事量 → 全レベルが均等
    - r < 1 (つまり d > log_b(a)): 浅いほど仕事量が多い → 根が支配
```

### 3.3 具体例での計算

```python
import math

def analyze_master_theorem(a: int, b: int, d: float, name: str = "") -> str:
    """マスター定理による計算量解析

    Args:
        a: 部分問題の数
        b: 分割比率
        d: 統合コストの指数
        name: アルゴリズム名（表示用）

    Returns:
        解析結果の文字列
    """
    log_b_a = math.log(a) / math.log(b)

    header = f"=== {name} ===" if name else "=== 解析結果 ==="
    lines = [
        header,
        f"  漸化式: T(n) = {a}T(n/{b}) + O(n^{d})",
        f"  a = {a}, b = {b}, d = {d}",
        f"  log_{b}({a}) = {log_b_a:.4f}",
    ]

    if abs(d - log_b_a) < 1e-9:
        lines.append(f"  ケース2: d = log_b(a) = {log_b_a:.4f}")
        lines.append(f"  => T(n) = O(n^{d} * log n)")
    elif d < log_b_a:
        lines.append(f"  ケース1: d = {d} < log_b(a) = {log_b_a:.4f}")
        lines.append(f"  => T(n) = O(n^{log_b_a:.4f})")
    else:
        lines.append(f"  ケース3: d = {d} > log_b(a) = {log_b_a:.4f}")
        lines.append(f"  => T(n) = O(n^{d})")

    return "\n".join(lines)


# 代表的なアルゴリズムの解析
print(analyze_master_theorem(2, 2, 1, "マージソート"))
# ケース2: T(n) = O(n log n)

print()
print(analyze_master_theorem(1, 2, 0, "二分探索"))
# ケース2: T(n) = O(log n)

print()
print(analyze_master_theorem(3, 2, 1, "Karatsuba 乗算"))
# ケース1: T(n) = O(n^1.585)

print()
print(analyze_master_theorem(7, 2, 2, "Strassen 行列乗算"))
# ケース1: T(n) = O(n^2.807)

print()
print(analyze_master_theorem(4, 2, 2, "特殊例"))
# ケース2: T(n) = O(n^2 log n)
```

### 3.4 マスター定理の適用例一覧

| アルゴリズム | 漸化式 | a | b | d | log_b(a) | ケース | 計算量 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| 二分探索 | T(n) = T(n/2) + O(1) | 1 | 2 | 0 | 0 | 2 | O(log n) |
| マージソート | T(n) = 2T(n/2) + O(n) | 2 | 2 | 1 | 1 | 2 | O(n log n) |
| Karatsuba | T(n) = 3T(n/2) + O(n) | 3 | 2 | 1 | 1.585 | 1 | O(n^1.585) |
| Strassen | T(n) = 7T(n/2) + O(n^2) | 7 | 2 | 2 | 2.807 | 1 | O(n^2.807) |
| 最近接点対 | T(n) = 2T(n/2) + O(n) | 2 | 2 | 1 | 1 | 2 | O(n log n) |
| 選択アルゴリズム | T(n) = T(n/2) + O(n) | 1 | 2 | 1 | 0 | 3 | O(n) |
| ナイーブ行列乗算 | T(n) = 8T(n/2) + O(n^2) | 8 | 2 | 2 | 3 | 1 | O(n^3) |

### 3.5 マスター定理が適用できないケース

マスター定理は万能ではない。以下のような漸化式には適用できない。

```
適用不可の漸化式:

1. 部分問題のサイズが不均等
   T(n) = T(n/3) + T(2n/3) + O(n)
   → 部分問題のサイズが n/3 と 2n/3 で異なる
   → Akra-Bazzi 定理または再帰木法で解析する

2. T(n) = aT(n/b) + f(n) の形でない
   T(n) = T(n-1) + T(n-2)
   → 減少幅が定数で、割り算ではない
   → 特性方程式法で解く

3. f(n) が多項式的でない
   T(n) = 2T(n/2) + n/log(n)
   → f(n) が n^d の形で表せない
   → 再帰木法や置換法で解析する
```

---

## 4. Karatsuba 乗算 -- 大数乗算の高速化

### 4.1 通常の乗算の問題点

2つの n 桁の数を掛け算するとき、小学校で習う筆算アルゴリズムは O(n^2) の計算量がかかる。暗号学や高精度演算など、数千桁以上の数を扱う場面ではこれがボトルネックになる。

1962 年、当時 23 歳の大学院生だった Anatolii Karatsuba は、乗算の計算量を O(n^{log_2 3}) ≈ O(n^{1.585}) に削減する方法を発見した。これは分割統治法の威力を示す画期的な成果だった。

### 4.2 Karatsuba のトリック

```
┌──────────────────────────────────────────────────────────────┐
│                 Karatsuba のアイデア                           │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  2つの n 桁の数 x, y を上位と下位に分割する:                  │
│    x = a * B^m + b    (B は基数、m = n/2)                    │
│    y = c * B^m + d                                           │
│                                                              │
│  ナイーブな計算:                                              │
│    x * y = ac * B^{2m} + (ad + bc) * B^m + bd                │
│    → 4回の乗算: ac, ad, bc, bd                               │
│                                                              │
│  Karatsuba のトリック:                                        │
│    p1 = a * c                                                │
│    p2 = b * d                                                │
│    p3 = (a + b) * (c + d) = ac + ad + bc + bd                │
│                                                              │
│    ad + bc = p3 - p1 - p2                                    │
│                                                              │
│    x * y = p1 * B^{2m} + (p3 - p1 - p2) * B^m + p2          │
│                                                              │
│  → 乗算が 4回 から 3回 に削減!                               │
│    (加減算は増えるが、乗算よりはるかに安い)                   │
│                                                              │
│  漸化式: T(n) = 3T(n/2) + O(n)                               │
│  マスター定理ケース1: T(n) = O(n^{log_2 3}) ≈ O(n^{1.585})  │
└──────────────────────────────────────────────────────────────┘
```

### 4.3 実装

```python
def karatsuba(x: int, y: int) -> int:
    """Karatsuba 乗算アルゴリズム - O(n^1.585)

    大きな整数の乗算を、ナイーブな O(n^2) から O(n^{1.585}) に改善する。
    再帰的に3回の乗算に分解する。
    """
    # 基底条件: 小さい数は直接掛ける
    if x < 10 or y < 10:
        return x * y

    # 符号の処理
    sign = 1
    if x < 0:
        x, sign = -x, -sign
    if y < 0:
        y, sign = -y, -sign

    # 桁数を揃える
    n = max(len(str(x)), len(str(y)))
    m = n // 2

    # 分割: x = a * 10^m + b, y = c * 10^m + d
    power = 10 ** m
    a, b = divmod(x, power)
    c, d = divmod(y, power)

    # 3回の再帰的乗算（4回ではなく3回!）
    p1 = karatsuba(a, c)           # a * c
    p2 = karatsuba(b, d)           # b * d
    p3 = karatsuba(a + b, c + d)   # (a+b) * (c+d)

    # 統合: x*y = p1 * 10^{2m} + (p3 - p1 - p2) * 10^m + p2
    result = p1 * (10 ** (2 * m)) + (p3 - p1 - p2) * power + p2

    return sign * result


# --- 検証 ---
# 小さい数
assert karatsuba(1234, 5678) == 1234 * 5678  # 7006652
print(f"1234 * 5678 = {karatsuba(1234, 5678)}")

# 大きい数
big_x = 3141592653589793238462643383279
big_y = 2718281828459045235360287471352
assert karatsuba(big_x, big_y) == big_x * big_y
print(f"大数の検証: OK")

# 負の数
assert karatsuba(-123, 456) == -123 * 456
assert karatsuba(-123, -456) == (-123) * (-456)
print(f"負の数の検証: OK")

# ゼロ
assert karatsuba(0, 12345) == 0
print(f"ゼロの検証: OK")

# 性能比較（概念的）
# n桁 × n桁 の計算:
#   ナイーブ: n^2 回の1桁乗算
#   Karatsuba: n^1.585 回の1桁乗算
# n = 1000 のとき:
#   ナイーブ: 1,000,000 回
#   Karatsuba: 約 38,000 回 → 約26倍高速
```

### 4.4 Karatsuba を超える乗算アルゴリズム

Karatsuba は大数乗算の高速化への扉を開いた。その後、さらに高速なアルゴリズムが発見されている。

| アルゴリズム | 計算量 | 発表年 | 備考 |
|:---|:---|:---:|:---|
| 筆算 | O(n^2) | - | 小学校で学ぶ方法 |
| Karatsuba | O(n^{1.585}) | 1962 | 3分割の分割統治 |
| Toom-Cook 3 | O(n^{1.465}) | 1963 | 5分割の分割統治 |
| Schonhage-Strassen | O(n log n log log n) | 1971 | FFT ベース |
| Harvey-van der Hoeven | O(n log n) | 2019 | 理論的下限に到達 |

---

## 5. 最近接点対問題（Closest Pair of Points）

### 5.1 問題の定義と応用

平面上に n 個の点が与えられたとき、ユークリッド距離が最小となる2点の組を求める問題である。

**応用分野:**
- 衝突検出（ゲーム、ロボティクス）
- クラスタリングの初期化
- 地理情報システム（最寄り施設の検索）
- 分子シミュレーション（最近接原子対の探索）

ナイーブな全探索は全てのペアを調べるため O(n^2) だが、分割統治法を用いると O(n log n) で解ける。

### 5.2 アルゴリズムの詳細

```
┌──────────────────────────────────────────────────────────────────┐
│              最近接点対の分割統治アルゴリズム                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  前処理: 全点を x 座標でソートする                                │
│                                                                  │
│  1. 分割: 中央の x 座標で左右に分ける                             │
│                                                                  │
│     左半分      |  右半分                                         │
│     *     *     |  *                                              │
│        *        |     *                                           │
│     *      *    |                                                 │
│          *      |  *                                              │
│     *           | *                                               │
│                 |                                                 │
│            中央線 x = mid_x                                       │
│                                                                  │
│  2. 統治: 左右それぞれで最近接点対を再帰的に求める                │
│     delta_L = 左半分の最近接距離                                   │
│     delta_R = 右半分の最近接距離                                   │
│     delta   = min(delta_L, delta_R)                               │
│                                                                  │
│  3. 統合: 境界をまたぐペアをチェック                               │
│     - 中央線から delta 以内の「ストリップ」内の点のみ調べる       │
│     - ストリップ内の点を y 座標でソートする                       │
│     - 各点について、y 座標の差が delta 未満の点のみ比較する       │
│     - 理論上、各点と比較する必要がある点は最大 7 個               │
│     - したがってストリップの処理は O(n) （ソート込みで O(n log n)）│
│                                                                  │
│   ←delta→|←delta→                                                │
│     *     |  *      ← ストリップ内の点だけ調べる                  │
│        *  |     *                                                 │
│     *     |*                                                      │
│     ← ストリップ →                                                │
│                                                                  │
│  漸化式: T(n) = 2T(n/2) + O(n log n) → O(n log^2 n)             │
│  最適化: y ソートを事前に行うと O(n log n)                       │
└──────────────────────────────────────────────────────────────────┘
```

### 5.3 実装

```python
import math
from typing import List, Tuple, Optional

Point = Tuple[float, float]

def closest_pair(points: List[Point]) -> Tuple[float, Point, Point]:
    """最近接点対を分割統治法で求める - O(n log^2 n)

    Args:
        points: 2次元平面上の点のリスト [(x, y), ...]

    Returns:
        (最小距離, 点1, 点2) のタプル
    """
    def dist(p1: Point, p2: Point) -> float:
        """2点間のユークリッド距離"""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def brute_force(pts: List[Point]) -> Tuple[float, Point, Point]:
        """3点以下の場合の全探索"""
        min_d = float('inf')
        best_p1, best_p2 = pts[0], pts[1]
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = dist(pts[i], pts[j])
                if d < min_d:
                    min_d = d
                    best_p1, best_p2 = pts[i], pts[j]
        return min_d, best_p1, best_p2

    def closest_in_strip(
        strip: List[Point], delta: float
    ) -> Tuple[float, Optional[Point], Optional[Point]]:
        """ストリップ内の最近接点対を求める

        y 座標でソート済みの strip 内で、距離が delta 未満のペアを探す。
        各点について比較する必要がある点は最大 7 個なので、O(n) で処理できる。
        """
        min_d = delta
        best_p1, best_p2 = None, None

        # y 座標でソート
        strip.sort(key=lambda p: p[1])

        for i in range(len(strip)):
            j = i + 1
            while j < len(strip) and (strip[j][1] - strip[i][1]) < min_d:
                d = dist(strip[i], strip[j])
                if d < min_d:
                    min_d = d
                    best_p1, best_p2 = strip[i], strip[j]
                j += 1

        return min_d, best_p1, best_p2

    def solve(pts_sorted_x: List[Point]) -> Tuple[float, Point, Point]:
        """分割統治の本体"""
        n = len(pts_sorted_x)

        # 基底条件: 3点以下なら全探索
        if n <= 3:
            return brute_force(pts_sorted_x)

        # 分割
        mid = n // 2
        mid_x = pts_sorted_x[mid][0]
        left_half = pts_sorted_x[:mid]
        right_half = pts_sorted_x[mid:]

        # 統治（再帰）
        dl, pl1, pl2 = solve(left_half)
        dr, pr1, pr2 = solve(right_half)

        # 左右の結果から小さい方を選ぶ
        if dl <= dr:
            delta, best_p1, best_p2 = dl, pl1, pl2
        else:
            delta, best_p1, best_p2 = dr, pr1, pr2

        # 統合: ストリップ内のチェック
        strip = [p for p in pts_sorted_x if abs(p[0] - mid_x) < delta]
        ds, ps1, ps2 = closest_in_strip(strip, delta)

        if ps1 is not None and ds < delta:
            return ds, ps1, ps2
        return delta, best_p1, best_p2

    # 前処理: x 座標でソート
    sorted_by_x = sorted(points, key=lambda p: p[0])
    return solve(sorted_by_x)


# --- 動作確認 ---
points = [
    (2, 3), (12, 30), (40, 50), (5, 1), (12, 10), (3, 4)
]
distance, p1, p2 = closest_pair(points)
print(f"最近接点対: {p1}, {p2}")
print(f"距離: {distance:.4f}")
# 最近接点対: (2, 3), (3, 4)
# 距離: 1.4142

# より多くの点でテスト
import random
random.seed(42)
many_points = [(random.uniform(0, 1000), random.uniform(0, 1000)) for _ in range(100)]
d, p1, p2 = closest_pair(many_points)
print(f"\n100点のテスト:")
print(f"最近接点対: ({p1[0]:.2f}, {p1[1]:.2f}), ({p2[0]:.2f}, {p2[1]:.2f})")
print(f"距離: {d:.4f}")
```

---

## 6. その他の重要な分割統治アルゴリズム

### 6.1 Strassen 行列乗算

n x n の行列同士の乗算は、ナイーブに計算すると O(n^3) かかる。1969 年に Volker Strassen は、分割統治法を用いて O(n^{2.807}) に改善する方法を発見した。

**基本的なアイデア:**
通常の 2x2 ブロック行列乗算では 8 回の乗算が必要だが、Strassen は巧妙な式変形により 7 回に削減した。

```
ナイーブな 2x2 ブロック行列乗算:

┌       ┐   ┌       ┐   ┌                   ┐
│ A   B │ × │ E   F │ = │ AE+BG    AF+BH    │
│ C   D │   │ G   H │   │ CE+DG    CF+DH    │
└       ┘   └       ┘   └                   ┘

→ 8回の行列乗算: AE, BG, AF, BH, CE, DG, CF, DH

Strassen の 7 回の乗算:

  M1 = (A + D)(E + H)
  M2 = (C + D) E
  M3 = A (F - H)
  M4 = D (G - E)
  M5 = (A + B) H
  M6 = (C - A)(E + F)
  M7 = (B - D)(G + H)

結果:
  ┌                          ┐
  │ M1+M4-M5+M7    M3+M5    │
  │ M2+M4        M1-M2+M3+M6│
  └                          ┘

漸化式: T(n) = 7T(n/2) + O(n^2)
マスター定理ケース1: O(n^{log_2 7}) ≈ O(n^{2.807})
```

```python
import numpy as np
from typing import Tuple

def strassen(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Strassen 行列乗算 - O(n^2.807)

    正方行列 A, B の積を Strassen アルゴリズムで計算する。
    サイズが 2 の冪でない場合はゼロパディングを行う。
    """
    n = A.shape[0]

    # 基底条件: 小さい行列はナイーブに計算
    if n <= 64:
        return A @ B

    # サイズが奇数の場合はパディング
    if n % 2 != 0:
        A = np.pad(A, ((0, 1), (0, 1)))
        B = np.pad(B, ((0, 1), (0, 1)))
        result = strassen(A, B)
        return result[:n, :n]

    mid = n // 2

    # 4つのブロックに分割
    A11, A12 = A[:mid, :mid], A[:mid, mid:]
    A21, A22 = A[mid:, :mid], A[mid:, mid:]
    B11, B12 = B[:mid, :mid], B[:mid, mid:]
    B21, B22 = B[mid:, :mid], B[mid:, mid:]

    # Strassen の 7 回の乗算
    M1 = strassen(A11 + A22, B11 + B22)
    M2 = strassen(A21 + A22, B11)
    M3 = strassen(A11, B12 - B22)
    M4 = strassen(A22, B21 - B11)
    M5 = strassen(A11 + A12, B22)
    M6 = strassen(A21 - A11, B11 + B12)
    M7 = strassen(A12 - A22, B21 + B22)

    # 結果の統合
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6

    # ブロックを結合
    C = np.zeros((n, n))
    C[:mid, :mid] = C11
    C[:mid, mid:] = C12
    C[mid:, :mid] = C21
    C[mid:, mid:] = C22

    return C


# --- 動作確認 ---
np.random.seed(42)
n = 128
A = np.random.randint(0, 10, (n, n)).astype(float)
B = np.random.randint(0, 10, (n, n)).astype(float)

C_naive = A @ B
C_strassen = strassen(A, B)

print(f"行列サイズ: {n}x{n}")
print(f"ナイーブとの差の最大値: {np.max(np.abs(C_naive - C_strassen)):.10f}")
# 行列サイズ: 128x128
# ナイーブとの差の最大値: 0.0000000000 (浮動小数点の範囲内)
```

### 6.2 べき乗の高速化（繰り返し二乗法）

```python
def fast_power(base: int, exp: int, mod: int = None) -> int:
    """繰り返し二乗法 - O(log n)

    分割統治の考え方:
      base^exp = (base^{exp/2})^2         (exp が偶数の場合)
      base^exp = base * (base^{(exp-1)/2})^2  (exp が奇数の場合)

    再帰版は理解しやすいが、スタックを消費する。
    """
    if exp == 0:
        return 1
    if exp == 1:
        return base % mod if mod else base

    if exp % 2 == 0:
        half = fast_power(base, exp // 2, mod)
        result = half * half
    else:
        half = fast_power(base, (exp - 1) // 2, mod)
        result = base * half * half

    return result % mod if mod else result


def fast_power_iterative(base: int, exp: int, mod: int = None) -> int:
    """繰り返し二乗法（反復版） - O(log n)

    実用的にはこちらを使う。スタックオーバーフローの心配がない。
    exp のビット表現を利用する。

    例: base^13 = base^(1101_2) = base^8 * base^4 * base^1
    """
    result = 1
    if mod:
        base %= mod

    while exp > 0:
        # exp の最下位ビットが 1 なら結果に掛ける
        if exp & 1:
            result *= base
            if mod:
                result %= mod
        # base を二乗する
        exp >>= 1
        base *= base
        if mod:
            base %= mod

    return result


# --- 動作確認 ---
print(fast_power(2, 30))              # 1073741824
print(fast_power(2, 30, 10**9 + 7))   # 1073741824
print(fast_power(3, 100, 10**9 + 7))  # 981453966

# 反復版との一致確認
for b in range(2, 10):
    for e in range(0, 50):
        assert fast_power(b, e, 997) == fast_power_iterative(b, e, 997)
print("全テスト合格")
```

### 6.3 最大部分配列和（分割統治版）

```python
def max_subarray_dc(arr: list) -> tuple:
    """最大部分配列和 - 分割統治版 O(n log n)

    分割: 配列を左半分と右半分に分ける
    統治: 各半分で最大部分配列和を求める
    統合: 中央をまたぐ最大部分配列和を求め、3つの中で最大を返す

    Kadane のアルゴリズム O(n) の方が速いが、
    分割統治の練習問題として重要。

    Returns:
        (最大和, 開始インデックス, 終了インデックス)
    """
    def solve(lo: int, hi: int) -> tuple:
        # 基底条件
        if lo == hi:
            return arr[lo], lo, hi

        mid = (lo + hi) // 2

        # 左半分の最大部分配列和
        left_max, ll, lr = solve(lo, mid)

        # 右半分の最大部分配列和
        right_max, rl, rr = solve(mid + 1, hi)

        # 中央をまたぐ最大部分配列和
        # 中央から左に伸ばす
        left_sum = float('-inf')
        total = 0
        cross_l = mid
        for i in range(mid, lo - 1, -1):
            total += arr[i]
            if total > left_sum:
                left_sum = total
                cross_l = i

        # 中央から右に伸ばす
        right_sum = float('-inf')
        total = 0
        cross_r = mid + 1
        for i in range(mid + 1, hi + 1):
            total += arr[i]
            if total > right_sum:
                right_sum = total
                cross_r = i

        cross_max = left_sum + right_sum

        # 3つの候補から最大を選ぶ
        if left_max >= right_max and left_max >= cross_max:
            return left_max, ll, lr
        elif right_max >= left_max and right_max >= cross_max:
            return right_max, rl, rr
        else:
            return cross_max, cross_l, cross_r

    if not arr:
        return 0, -1, -1

    return solve(0, len(arr) - 1)


# --- 動作確認 ---
data = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
max_sum, start, end = max_subarray_dc(data)
print(f"最大部分配列和: {max_sum}")
print(f"部分配列: {data[start:end+1]} (インデックス {start}..{end})")
# 最大部分配列和: 6
# 部分配列: [4, -1, 2, 1] (インデックス 3..6)
```

### 6.4 中央値の中央値（Median of Medians）

k 番目に小さい要素を最悪 O(n) で求めるアルゴリズム。クイックセレクトの最悪ケースを防ぐためにピボット選択に分割統治を用いる。

```python
def median_of_medians(arr: list, k: int) -> int:
    """中央値の中央値アルゴリズムによる k 番目に小さい要素の選択

    最悪計算量: O(n)
    漸化式: T(n) = T(n/5) + T(7n/10) + O(n)

    Args:
        arr: 数値のリスト
        k: 求める順位（0-indexed）

    Returns:
        k 番目に小さい要素
    """
    if len(arr) <= 5:
        return sorted(arr)[k]

    # Step 1: 5個ずつのグループに分割し、各グループの中央値を求める
    medians = []
    for i in range(0, len(arr), 5):
        group = sorted(arr[i:i + 5])
        medians.append(group[len(group) // 2])

    # Step 2: 中央値の中央値を再帰的に求める（ピボットとする）
    pivot = median_of_medians(medians, len(medians) // 2)

    # Step 3: ピボットで3分割
    low = [x for x in arr if x < pivot]
    equal = [x for x in arr if x == pivot]
    high = [x for x in arr if x > pivot]

    # Step 4: k がどの部分に属するか判定して再帰
    if k < len(low):
        return median_of_medians(low, k)
    elif k < len(low) + len(equal):
        return pivot
    else:
        return median_of_medians(high, k - len(low) - len(equal))


# --- 動作確認 ---
data = [3, 7, 2, 9, 1, 8, 5, 4, 6, 10]
for k in range(len(data)):
    result = median_of_medians(data[:], k)
    expected = sorted(data)[k]
    assert result == expected, f"k={k}: got {result}, expected {expected}"
    print(f"  {k}番目に小さい要素: {result}")

# 中央値の取得
median = median_of_medians(data[:], len(data) // 2)
print(f"\n中央値: {median}")  # 6
```

---

## 7. 分割統治の適用判断と設計パラダイムの比較

### 7.1 分割統治が有効な問題の特徴

```
分割統治の適用判断フローチャート:

問題を小さな部分問題に分割できるか?
│
├─ NO → 他の手法を検討（貪欲法、探索、数学的解法など）
│
└─ YES → 部分問題は独立しているか（重複がないか）?
          │
          ├─ NO → 動的計画法（DP）を検討
          │       例: フィボナッチ数列、LCS、行列連鎖乗算
          │
          └─ YES → 統合ステップは効率的か（O(n) 以下）?
                    │
                    ├─ NO → 分割が計算量を改善するか再検討
                    │       統合が O(n^2) なら分割の意味がない場合も
                    │
                    └─ YES → 分割統治が有効!
                              分割は均等か?
                              │
                              ├─ YES → マスター定理で解析可能
                              └─ NO  → 再帰木法で解析
```

### 7.2 設計パラダイムの包括的比較

| 特性 | 分割統治 | 動的計画法 (DP) | 貪欲法 | バックトラッキング |
|:---|:---|:---|:---|:---|
| 部分問題の関係 | 独立 | 重複あり | 独立 | 独立（探索木） |
| 解の構築 | 再帰 + 統合 | テーブル埋め | 逐次的な決定 | 試行 + 巻き戻し |
| 最適解の保証 | 問題による | 常に最適 | 貪欲選択性あれば最適 | 常に最適（全探索） |
| 典型的な計算量 | O(n log n) | O(n^2) ~ O(nW) | O(n log n) | O(2^n) ~ O(n!) |
| 空間計算量 | O(log n) ~ O(n) | O(n) ~ O(n^2) | O(1) ~ O(n) | O(n) |
| 代表的な問題 | マージソート | 最長共通部分列 | 活動選択問題 | N-Queen |
| 後戻り | しない | しない | しない | する |
| 部分問題のサイズ | n/b（割合で縮小） | 1ずつ縮小 | 1ずつ縮小 | 1ずつ縮小 |

### 7.3 分割統治と DP の境界 -- 具体例で理解する

同じ問題でも、部分問題の構造によって分割統治と DP を使い分ける。

```python
# --- 例: 行列のべき乗 ---
# DP 的アプローチ（メモ化）は不要 → 分割統治が最適

def matrix_power(M: list, n: int) -> list:
    """行列の n 乗を分割統治で計算 - O(k^3 log n)

    k は行列のサイズ。部分問題は独立なので分割統治が適切。
    """
    size = len(M)

    def identity(size: int) -> list:
        return [[1 if i == j else 0 for j in range(size)] for i in range(size)]

    def mat_mult(A: list, B: list) -> list:
        size = len(A)
        C = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                for k_idx in range(size):
                    C[i][j] += A[i][k_idx] * B[k_idx][j]
        return C

    if n == 0:
        return identity(size)
    if n == 1:
        return [row[:] for row in M]

    if n % 2 == 0:
        half = matrix_power(M, n // 2)
        return mat_mult(half, half)
    else:
        return mat_mult(M, matrix_power(M, n - 1))


# --- 応用: フィボナッチ数を O(log n) で計算 ---
def fibonacci_matrix(n: int) -> int:
    """行列べき乗法によるフィボナッチ数の計算 - O(log n)

    [[F(n+1), F(n)], [F(n), F(n-1)]] = [[1,1],[1,0]]^n
    """
    if n <= 1:
        return n

    M = [[1, 1], [1, 0]]
    result = matrix_power(M, n)
    return result[0][1]


# 動作確認
for i in range(15):
    print(f"F({i}) = {fibonacci_matrix(i)}", end="  ")
# F(0)=0 F(1)=1 F(2)=1 F(3)=2 F(4)=3 F(5)=5 F(6)=8 ...

print()
print(f"F(50) = {fibonacci_matrix(50)}")   # 12586269025
print(f"F(100) = {fibonacci_matrix(100)}") # 354224848179261915075
```

---

## 8. アンチパターンと陥りやすい罠

### アンチパターン 1: 不均等な分割による計算量の退化

分割統治で最も重要なのは「均等な分割」である。分割が偏ると再帰木の深さが O(n) に退化し、期待される計算量が得られない。

```python
# ============================================================
# BAD: 不均等な分割 → 最悪 O(n^2) に退化
# ============================================================
def bad_quicksort(arr: list) -> list:
    """最悪ケースのクイックソート

    ピボットに最小値を選ぶと、1 個 vs (n-1) 個の分割になる。
    再帰の深さが O(n) になり、全体が O(n^2) に退化する。

    漸化式: T(n) = T(n-1) + O(n) → O(n^2)
    """
    if len(arr) <= 1:
        return arr
    pivot = min(arr)  # 常に最小値をピボットに → 最悪の分割
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return bad_quicksort(left) + middle + bad_quicksort(right)


# ============================================================
# GOOD: 均等分割を保証するマージソート
# ============================================================
def good_merge_sort(arr: list) -> list:
    """マージソートは常に均等分割を保証する

    常に中央で分割するため、再帰の深さは O(log n) で安定。
    漸化式: T(n) = 2T(n/2) + O(n) → O(n log n) が常に保証される。
    """
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = good_merge_sort(arr[:mid])
    right = good_merge_sort(arr[mid:])
    return merge(left, right)  # 前述の merge 関数を使用


# ============================================================
# BETTER: ランダム化で平均的に均等な分割を実現
# ============================================================
import random

def randomized_quicksort(arr: list) -> list:
    """ランダム化クイックソート

    ピボットをランダムに選ぶことで、期待計算量 O(n log n) を達成。
    最悪ケースの確率は 1/n! で、事実上起きない。
    """
    if len(arr) <= 1:
        return arr
    pivot = random.choice(arr)  # ランダムにピボットを選択
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return randomized_quicksort(left) + middle + randomized_quicksort(right)


# 不均等分割の影響を確認
sorted_input = list(range(1000))
# bad_quicksort(sorted_input)  # RecursionError の危険あり!
print(f"ランダム化QS: {randomized_quicksort(sorted_input)[:10]}...")  # 正常動作
```

```
不均等分割の再帰木（最悪ケース）:

T(n)
├── T(0)     ← 空
└── T(n-1)
    ├── T(0)
    └── T(n-2)
        ├── T(0)
        └── T(n-3)
            └── ...
                └── T(1)

深さ: n
各レベルの仕事: O(n), O(n-1), O(n-2), ...
合計: O(n^2)  ← マージソートの O(n log n) に比べて大幅に劣化


均等分割の再帰木（理想ケース）:

              T(n)
         /          \
     T(n/2)        T(n/2)
     /    \         /    \
  T(n/4) T(n/4) T(n/4) T(n/4)
  ...     ...    ...     ...

深さ: log n
各レベルの仕事: O(n)（合計で）
合計: O(n log n)
```

### アンチパターン 2: 部分問題が重複しているのに分割統治を適用

```python
# ============================================================
# BAD: フィボナッチに素朴な分割統治 → O(2^n) の指数爆発
# ============================================================
call_count = 0

def fib_bad(n: int) -> int:
    """素朴な再帰フィボナッチ - O(2^n)

    fib(5) を計算するだけで fib(2) が 3 回、fib(3) が 2 回呼ばれる。
    部分問題が大量に重複しており、分割統治は不適切。
    """
    global call_count
    call_count += 1

    if n <= 1:
        return n
    return fib_bad(n - 1) + fib_bad(n - 2)


call_count = 0
result = fib_bad(20)
print(f"fib_bad(20) = {result}, 関数呼び出し回数: {call_count}")
# fib_bad(20) = 6765, 関数呼び出し回数: 21891


# ============================================================
# GOOD: メモ化再帰（DP）→ O(n)
# ============================================================
from functools import lru_cache

@lru_cache(maxsize=None)
def fib_dp(n: int) -> int:
    """メモ化再帰によるフィボナッチ - O(n)

    同じ部分問題を二度と再計算しない。
    """
    if n <= 1:
        return n
    return fib_dp(n - 1) + fib_dp(n - 2)

print(f"fib_dp(20) = {fib_dp(20)}")   # 6765
print(f"fib_dp(100) = {fib_dp(100)}") # 354224848179261915075


# ============================================================
# BEST: 行列べき乗法 → O(log n) ← 分割統治が適切に機能する例
# ============================================================
print(f"fibonacci_matrix(20) = {fibonacci_matrix(20)}")   # 6765
print(f"fibonacci_matrix(100) = {fibonacci_matrix(100)}") # 354224848179261915075
```

### アンチパターン 3: 基底条件の不備

```python
# ============================================================
# BAD: 基底条件が不十分 → 無限再帰
# ============================================================
def bad_binary_search(arr: list, target: int, lo: int, hi: int) -> int:
    """基底条件の不備がある二分探索"""
    mid = (lo + hi) // 2
    # BUG: lo > hi のチェックがない → 要素が存在しない場合に無限再帰
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return bad_binary_search(arr, target, mid + 1, hi)
    else:
        return bad_binary_search(arr, target, lo, mid - 1)


# ============================================================
# GOOD: 正しい基底条件
# ============================================================
def good_binary_search(arr: list, target: int, lo: int, hi: int) -> int:
    """正しい二分探索"""
    if lo > hi:  # 基底条件: 要素が見つからない
        return -1
    mid = (lo + hi) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return good_binary_search(arr, target, mid + 1, hi)
    else:
        return good_binary_search(arr, target, lo, mid - 1)


# 動作確認
arr = [1, 3, 5, 7, 9, 11, 13]
print(good_binary_search(arr, 7, 0, len(arr) - 1))   # 3
print(good_binary_search(arr, 6, 0, len(arr) - 1))   # -1（見つからない）
```

### アンチパターンの判断基準まとめ

| アンチパターン | 症状 | 原因 | 対策 |
|:---|:---|:---|:---|
| 不均等分割 | 計算量が O(n^2) に退化 | ピボット選択の失敗 | ランダム化 or 中央値の中央値 |
| 重複部分問題 | 指数的な計算量 | 部分問題の独立性を誤認 | DP（メモ化 or ボトムアップ）に切替 |
| 基底条件の不備 | 無限再帰 / スタックオーバーフロー | 終了条件の漏れ | 全ての終端ケースを列挙 |
| 統合コストの見落とし | 期待より遅い | 統合が O(n^2) | 統合の効率化 or 別手法を検討 |

---

## 9. 演習問題（3段階）

### 基礎レベル（Foundation）

**演習 1: 転倒数のカウント**

配列中の転倒数（inversion count）を分割統治で O(n log n) で求めよ。転倒とは i < j かつ arr[i] > arr[j] となるペア (i, j) である。

```python
def count_inversions(arr: list) -> tuple:
    """転倒数を分割統治で求める - O(n log n)

    マージソートの統合ステップで、右側の要素が先にマージされる回数を
    数えることで転倒数を求められる。

    Returns:
        (ソート済み配列, 転倒数)
    """
    if len(arr) <= 1:
        return arr[:], 0

    mid = len(arr) // 2
    left, left_inv = count_inversions(arr[:mid])
    right, right_inv = count_inversions(arr[mid:])

    merged = []
    inversions = left_inv + right_inv
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            # right[j] が left[i..] の全ての要素より小さい → 転倒
            merged.append(right[j])
            inversions += len(left) - i
            j += 1

    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged, inversions


# --- 動作確認 ---
test_cases = [
    ([1, 2, 3, 4, 5], 0),       # ソート済み → 転倒数 0
    ([5, 4, 3, 2, 1], 10),      # 逆順 → C(5,2) = 10
    ([2, 4, 1, 3, 5], 3),       # (2,1), (4,1), (4,3)
    ([1, 20, 6, 4, 5], 5),      # (20,6), (20,4), (20,5), (6,4), (6,5)
]

for arr, expected in test_cases:
    _, inv = count_inversions(arr)
    status = "OK" if inv == expected else "NG"
    print(f"  {arr} → 転倒数 {inv} (期待 {expected}) [{status}]")
```

**演習 2: べき乗和**

1^k + 2^k + ... + n^k を分割統治の考え方で効率的に計算せよ（ヒント: 直接計算は O(n log k) だが、分割統治的な思考で問題構造を理解する）。

```python
def power_sum(n: int, k: int, mod: int = 10**9 + 7) -> int:
    """1^k + 2^k + ... + n^k を計算する

    各項の計算に繰り返し二乗法 O(log k) を使用し、
    全体で O(n log k) となる。

    分割統治的な分解:
      S(n, k) = S(n/2, k) + sum(i^k for i in range(n/2+1, n+1))
      ただしこの場合は直接計算の方がシンプル。
    """
    total = 0
    for i in range(1, n + 1):
        total = (total + pow(i, k, mod)) % mod
    return total


# 動作確認
print(f"1^2 + 2^2 + ... + 10^2 = {power_sum(10, 2)}")  # 385
print(f"1^3 + 2^3 + ... + 10^3 = {power_sum(10, 3)}")  # 3025
# 検算: sum(i**2 for i in range(1,11)) = 385
# 検算: sum(i**3 for i in range(1,11)) = 3025
```

### 応用レベル（Intermediate）

**演習 3: 多数派要素（Majority Element）**

配列中に過半数を占める要素があればそれを返し、なければ None を返せ。分割統治で O(n log n) で解け。

```python
def majority_element(arr: list) -> int:
    """多数派要素を分割統治で求める - O(n log n)

    Boyer-Moore 投票アルゴリズムなら O(n) だが、
    分割統治の練習として実装する。

    アイデア:
    - 配列を左右に分割する
    - 左半分の多数派候補と右半分の多数派候補を再帰的に求める
    - 全体で各候補の出現回数をカウントして判定する
    """
    def solve(lo: int, hi: int):
        # 基底条件: 要素が1つ
        if lo == hi:
            return arr[lo]

        mid = (lo + hi) // 2
        left_maj = solve(lo, mid)
        right_maj = solve(mid + 1, hi)

        # 両方の多数派が同じなら確定
        if left_maj == right_maj:
            return left_maj

        # 異なる場合は全体でカウント
        left_count = sum(1 for i in range(lo, hi + 1) if arr[i] == left_maj)
        right_count = sum(1 for i in range(lo, hi + 1) if arr[i] == right_maj)

        threshold = (hi - lo + 1) // 2 + 1
        if left_count >= threshold:
            return left_maj
        if right_count >= threshold:
            return right_maj
        return None

    if not arr:
        return None

    result = solve(0, len(arr) - 1)

    # 最終確認
    if result is not None and arr.count(result) > len(arr) // 2:
        return result
    return None


# --- 動作確認 ---
print(majority_element([3, 3, 4, 2, 3, 3, 3]))  # 3
print(majority_element([1, 2, 3, 4, 5]))          # None
print(majority_element([2, 2, 1, 1, 1, 2, 2]))    # 2
```

**演習 4: k 番目に小さい要素を2つのソート済み配列から求める**

2つのソート済み配列 A, B が与えられたとき、統合した場合に k 番目に小さい要素を O(log(min(m, n))) で求めよ。

```python
def kth_of_two_sorted(A: list, B: list, k: int) -> int:
    """2つのソート済み配列の k 番目に小さい要素 - O(log(min(m,n)))

    分割統治のアイデア:
    - A の前半 i 個と B の前半 j 個（i + j = k）を考える
    - A[i-1] <= B[j] かつ B[j-1] <= A[i] なら、
      max(A[i-1], B[j-1]) が答え

    Args:
        A, B: ソート済み配列
        k: 求める順位（1-indexed）
    """
    # A が短い方になるようにする
    if len(A) > len(B):
        return kth_of_two_sorted(B, A, k)

    m, n = len(A), len(B)

    # i の範囲: A から最低 0 個、最大 min(m, k) 個取る
    lo = max(0, k - n)
    hi = min(m, k)

    while lo <= hi:
        i = (lo + hi) // 2  # A から i 個取る
        j = k - i            # B から j 個取る

        a_left = A[i - 1] if i > 0 else float('-inf')
        b_left = B[j - 1] if j > 0 else float('-inf')
        a_right = A[i] if i < m else float('inf')
        b_right = B[j] if j < n else float('inf')

        if a_left <= b_right and b_left <= a_right:
            return max(a_left, b_left)
        elif a_left > b_right:
            hi = i - 1
        else:
            lo = i + 1

    raise ValueError("入力が不正")


# --- 動作確認 ---
A = [1, 3, 5, 7, 9]
B = [2, 4, 6, 8, 10]
for k in range(1, 11):
    result = kth_of_two_sorted(A, B, k)
    print(f"  k={k}: {result}", end="")
print()
# k=1: 1, k=2: 2, ..., k=10: 10
```

### 発展レベル（Advanced）

**演習 5: 高速フーリエ変換（FFT）による多項式乗算**

2つの多項式の積を O(n log n) で計算する FFT は、分割統治の極致である。

```python
import cmath
from typing import List

def fft(a: List[complex], invert: bool = False) -> List[complex]:
    """高速フーリエ変換 (Cooley-Tukey FFT) - O(n log n)

    分割統治の適用:
    - 偶数番目の係数と奇数番目の係数に分割
    - それぞれを再帰的に FFT
    - バタフライ演算で統合

    Args:
        a: 多項式の係数（長さは 2 の冪）
        invert: True なら逆 FFT

    Returns:
        FFT 変換後の係数
    """
    n = len(a)
    if n == 1:
        return a[:]

    # 偶数番目と奇数番目に分割
    even = fft(a[0::2], invert)
    odd = fft(a[1::2], invert)

    # 回転因子
    angle = 2 * cmath.pi / n * (-1 if invert else 1)
    w = 1
    wn = cmath.exp(1j * angle)

    result = [0] * n
    for i in range(n // 2):
        result[i] = even[i] + w * odd[i]
        result[i + n // 2] = even[i] - w * odd[i]
        if invert:
            result[i] /= 2
            result[i + n // 2] /= 2
        w *= wn

    return result


def polynomial_multiply(a: List[int], b: List[int]) -> List[int]:
    """FFT による多項式乗算 - O(n log n)

    通常の多項式乗算は O(n^2) だが、FFT を使うと O(n log n) になる。

    手順:
    1. 係数表現 → 点値表現 (FFT)
    2. 点ごとに乗算 O(n)
    3. 点値表現 → 係数表現 (逆 FFT)

    Args:
        a, b: 多項式の係数リスト（a[i] は x^i の係数）

    Returns:
        積の多項式の係数リスト
    """
    result_len = len(a) + len(b) - 1

    # サイズを 2 の冪に拡張
    n = 1
    while n < result_len:
        n <<= 1

    fa = [complex(x) for x in a] + [0] * (n - len(a))
    fb = [complex(x) for x in b] + [0] * (n - len(b))

    # FFT で点値表現に変換
    fa = fft(fa)
    fb = fft(fb)

    # 点ごとに乗算
    fc = [fa[i] * fb[i] for i in range(n)]

    # 逆 FFT で係数表現に戻す
    fc = fft(fc, invert=True)

    # 実部を取り出して整数に丸める
    result = [round(c.real) for c in fc[:result_len]]
    return result


# --- 動作確認 ---
# (1 + 2x + 3x^2) * (4 + 5x) = 4 + 13x + 22x^2 + 15x^3
a = [1, 2, 3]
b = [4, 5]
product = polynomial_multiply(a, b)
print(f"({a}) * ({b}) = {product}")
# [4, 13, 22, 15]

# 検算: 123 * 45 = 5535
# (3 + 2*10 + 1*100) * (5 + 4*10) = 5535
a2 = [3, 2, 1]
b2 = [5, 4]
p2 = polynomial_multiply(a2, b2)
value = sum(c * (10 ** i) for i, c in enumerate(p2))
print(f"123 * 45 = {value}")  # 5535
```

---

## 10. FAQ（よくある質問）

### Q1: 分割統治と再帰の違いは何か?

**A:** 再帰は**実装技法**（関数が自分自身を呼び出すこと）であり、分割統治は**設計パラダイム**（問題を分割→統治→統合するという戦略）である。

分割統治は通常、再帰で実装される。しかし、再帰を使う全てのアルゴリズムが分割統治というわけではない。例えば DFS（深さ優先探索）は再帰で実装されるが、「問題を分割して統合する」という構造を持たないため分割統治とは呼ばない。

逆に、ボトムアップ・マージソートのように、分割統治の考え方を反復（ループ）で実装することも可能である。

| | 再帰 | 分割統治 |
|:---|:---|:---|
| 分類 | 実装技法 | 設計パラダイム |
| 定義 | 関数が自身を呼び出す | 分割→統治→統合の3ステップ |
| 例 | DFS, 階乗, ハノイの塔 | マージソート, Karatsuba, FFT |
| 関係 | 分割統治の実装に使われる | 再帰で実装されることが多い |

### Q2: マスター定理で解けない漸化式にはどう対処するか?

**A:** マスター定理が適用できない場合、以下の3つの方法がある。

1. **再帰木法 (Recursion Tree Method):** 再帰木を描き、各レベルの仕事量を合計する。直感的で汎用性が高い。

2. **置換法 (Substitution Method):** 解を推測し、数学的帰納法で証明する。厳密だが、正しい推測が必要。

3. **Akra-Bazzi 定理:** T(n) = sum(a_i * T(n/b_i)) + f(n) の形で、部分問題のサイズが不均等な場合にも適用可能。マスター定理の一般化。

```
例: T(n) = T(n/3) + T(2n/3) + O(n) の解析

再帰木法:
  深さ 0: n              仕事量 = n
  深さ 1: n/3 + 2n/3     仕事量 = n
  深さ 2: n/9 + 2n/9 + 2n/9 + 4n/9  仕事量 = n
  ...
  深さ k: 仕事量 = n（各深さで仕事量は n）

  最大深さ: log_{3/2}(n) ≈ 1.71 log n（最長パスは 2n/3 を辿る）
  最小深さ: log_3(n)

  → T(n) = O(n log n)
```

### Q3: Strassen 行列乗算は実用的か?

**A:** Strassen のアルゴリズムは理論的に O(n^{2.807}) であり、ナイーブな O(n^3) より漸近的に速い。しかし実用上は以下の理由から、使われる場面が限られる。

1. **定数係数が大きい:** Strassen は 18 回の加減算を必要とし、小さな行列ではナイーブな方法が速い。一般的に n > 数百以上で初めて有利になる。

2. **数値的安定性:** 加減算の回数が多いため、浮動小数点演算では丸め誤差が蓄積しやすい。科学技術計算ではこれが問題になる。

3. **キャッシュ効率:** 現代の CPU はキャッシュ階層を持つ。BLAS（Basic Linear Algebra Subprograms）などのライブラリは、キャッシュに最適化されたブロック行列乗算を実装しており、Strassen よりも実行速度が速いことが多い。

4. **GPU の台頭:** GPU は大規模な並列計算に特化しており、ナイーブな行列乗算でも極めて高速に実行できる。

結論: Strassen は理論的には重要だが、実用的には BLAS ライブラリや GPU を使う方が一般的である。ただし、巨大な行列（数千 x 数千以上）や整数行列の場合は有用なことがある。

### Q4: 分割統治は並列化に向いているか?

**A:** 分割統治は並列化に非常に適している。部分問題が独立であるため、各部分問題を異なるプロセッサやスレッドで同時に処理できる。

```python
from concurrent.futures import ThreadPoolExecutor

def parallel_merge_sort(arr: list, threshold: int = 10000) -> list:
    """並列マージソート

    配列が threshold 以上の場合、左右の再帰を別スレッドで実行する。
    小さな配列は通常の逐次処理にフォールバックする。

    注意: Python の GIL の制約により、CPU バウンドな処理では
    ThreadPoolExecutor よりも ProcessPoolExecutor の方が効果的。
    ここでは簡略化のため ThreadPoolExecutor を使用。
    """
    if len(arr) <= 1:
        return arr

    if len(arr) < threshold:
        return merge_sort(arr)  # 小さい配列は逐次処理

    mid = len(arr) // 2

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_left = executor.submit(parallel_merge_sort, arr[:mid], threshold)
        future_right = executor.submit(parallel_merge_sort, arr[mid:], threshold)
        left = future_left.result()
        right = future_right.result()

    return merge(left, right)


# 注意: 上記は概念実装。本番では multiprocessing や
# C 拡張（numpy.sort など）を使う方が実用的。
```

### Q5: 再帰の深さが深くなりすぎる場合はどうするか?

**A:** Python のデフォルトの再帰制限は 1000 回程度である。分割統治の再帰深さは O(log n) なので、n = 2^1000 でない限り問題にならない。ただし、不均等分割やバグにより O(n) の深さになるケースでは注意が必要。

対処法:
1. **再帰制限の引き上げ:** `sys.setrecursionlimit()` で一時的に制限を緩和する（ただし根本的な解決にはならない）。
2. **反復版への変換:** ボトムアップ・マージソートのように、再帰を反復に変換する。
3. **末尾再帰の最適化:** Python は末尾再帰最適化をサポートしていないが、手動でループに変換できる。
4. **分割の均等性を保証する:** 常に半分に分割すれば、深さは O(log n) に収まる。

### Q6: 分割統治で解ける典型的な競技プログラミングの問題は?

**A:** 以下が代表的な問題パターンである。

| パターン | 典型問題 | 計算量 |
|:---|:---|:---|
| マージソート応用 | 転倒数カウント | O(n log n) |
| 二分探索 | 値の探索、条件を満たす最小値 | O(log n) |
| 分割統治 + 座標幾何 | 最近接点対 | O(n log n) |
| セグメント木（区間クエリ） | 区間最大値、区間和 | O(n log n) 構築, O(log n) クエリ |
| FFT / NTT | 多項式乗算、畳み込み | O(n log n) |
| CDQ 分割統治 | 3次元偏順序カウント | O(n log^2 n) |

---

## 11. 計算量の包括的比較

### 分割統治アルゴリズムの計算量一覧

| アルゴリズム | 時間計算量 | 空間計算量 | 分割数 | 分割比率 | 統合コスト |
|:---|:---|:---|:---:|:---:|:---|
| 二分探索 | O(log n) | O(log n) 再帰 / O(1) 反復 | 1 | 1/2 | O(1) |
| マージソート | O(n log n) | O(n) | 2 | 1/2 | O(n) |
| クイックソート（平均） | O(n log n) | O(log n) | 2 | 可変 | O(n) |
| クイックソート（最悪） | O(n^2) | O(n) | 2 | 不均等 | O(n) |
| Karatsuba 乗算 | O(n^{1.585}) | O(n log n) | 3 | 1/2 | O(n) |
| Strassen 行列乗算 | O(n^{2.807}) | O(n^2) | 7 | 1/2 | O(n^2) |
| 最近接点対 | O(n log n) | O(n) | 2 | 1/2 | O(n) |
| FFT | O(n log n) | O(n) | 2 | 1/2 | O(n) |
| 中央値の中央値 | O(n) | O(n) | 1 | 7/10 | O(n) |
| 繰り返し二乗法 | O(log n) | O(log n) 再帰 / O(1) 反復 | 1 | 1/2 | O(1) |

---


## FAQ

### Q1: このトピックを学ぶ上で最も重要なポイントは何ですか？

実践的な経験を積むことが最も重要です。理論だけでなく、実際にコードを書いて動作を確認することで理解が深まります。

### Q2: 初心者がよく陥る間違いは何ですか？

基礎を飛ばして応用に進むことです。このガイドで説明している基本概念をしっかり理解してから、次のステップに進むことをお勧めします。

### Q3: 実務ではどのように活用されていますか？

このトピックの知識は、日常的な開発業務で頻繁に活用されます。特にコードレビューやアーキテクチャ設計の際に重要になります。

---

## 12. まとめ

### 分割統治法の核心

| 項目 | 要点 |
|:---|:---|
| 3ステップ | 分割→統治→統合の再帰的設計。基底条件の正しさが全体を左右する |
| マスター定理 | T(n) = aT(n/b) + O(n^d) → 3ケースの判定で計算量を即座に求められる |
| マージソート | 安定ソート、最悪 O(n log n) 保証。分割統治の最も基本的かつ実用的な例 |
| Karatsuba | 4回の乗算を3回に削減。O(n^2) → O(n^{1.585}) の劇的な改善 |
| Strassen | 8回の行列乗算を7回に削減。理論的意義は大きいが実用上は制約あり |
| 最近接点対 | ストリップ内の限定的チェックにより O(n^2) → O(n log n) を達成 |
| FFT | 多項式乗算を O(n^2) → O(n log n) に高速化。信号処理の基盤 |
| 適用判断 | 独立な部分問題 + 効率的な統合 + 均等な分割が成功の鍵 |

### 分割統治を使いこなすためのチェックリスト

1. **分割は均等か?** 不均等な分割は計算量を退化させる
2. **部分問題は独立か?** 重複があるなら DP に切り替える
3. **統合は効率的か?** 統合コストが全体の計算量を左右する
4. **基底条件は正しいか?** 全ての終端ケースをカバーしているか確認する
5. **マスター定理は適用可能か?** 漸化式を立てて計算量を解析する

---

## 次に読むべきガイド

- [ソートアルゴリズム](./00-sorting.md) -- マージソート・クイックソートの詳細な実装と比較
- [動的計画法](./04-dynamic-programming.md) -- 部分問題が重複する場合の設計手法
- [バックトラッキング](./07-backtracking.md) -- 別の再帰的問題解決パラダイム
- [探索アルゴリズム](./01-searching.md) -- 二分探索の詳細

---

## 参考文献

1. Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. -- 第4章「Divide-and-Conquer」は分割統治法の教科書的な解説であり、マスター定理の証明を含む。
2. Karatsuba, A. & Ofman, Y. (1963). "Multiplication of multidigit numbers on automata." *Soviet Physics Doklady*, 7(7), 595-596. -- 大数乗算の計算量が O(n^2) を下回れることを初めて示した歴史的論文。
3. Shamos, M. I. & Hoey, D. (1975). "Closest-point problems." *Proceedings of the 16th Annual Symposium on Foundations of Computer Science (FOCS)*. -- 最近接点対問題の O(n log n) アルゴリズムを提案した原論文。
4. Kleinberg, J. & Tardos, E. (2005). *Algorithm Design*. Pearson. -- 第5章「Divide and Conquer」は応用例が豊富で、漸化式の解き方を丁寧に解説している。
5. Strassen, V. (1969). "Gaussian elimination is not optimal." *Numerische Mathematik*, 13(4), 354-356. -- 行列乗算の計算量が O(n^3) を下回れることを示し、代数的計算量理論の端緒を開いた。
6. Cooley, J. W. & Tukey, J. W. (1965). "An algorithm for the machine calculation of complex Fourier series." *Mathematics of Computation*, 19(90), 297-301. -- 高速フーリエ変換（FFT）の原論文。分割統治法の最も影響力のある応用の一つ。
7. Sedgewick, R. & Wayne, K. (2011). *Algorithms* (4th ed.). Addison-Wesley. -- マージソートの実装バリエーション（ボトムアップ、最適化版）が詳しい。
