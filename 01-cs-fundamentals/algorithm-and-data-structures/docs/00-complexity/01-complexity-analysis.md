# 計算量解析 — 最悪/平均/最良ケース、償却解析、再帰の解析、マスター定理

> アルゴリズムの効率を正確に評価するための体系的手法を学ぶ。最悪・平均・最良ケースの違い、償却解析の考え方、再帰アルゴリズムの漸化式の立て方と解法（マスター定理、再帰木法、置換法）までを網羅する。

---

## この章で学ぶこと

1. **漸近記法**（O, Omega, Theta）の厳密な定義と使い分け
2. **最悪ケース・平均ケース・最良ケース** の区別と各解析手法
3. **償却解析**（集約法、配賦法、ポテンシャル法）の理論と実践
4. **再帰の漸化式** の立て方と3つの解法（マスター定理、再帰木法、置換法）
5. **マスター定理** の3つのケースと適用条件、適用外への対処
6. 計算量解析における **よくある落とし穴** と回避策

---

## 目次

1. [漸近記法の基礎](#1-漸近記法の基礎)
2. [最悪ケース・平均ケース・最良ケースの解析](#2-最悪ケース平均ケース最良ケースの解析)
3. [償却解析](#3-償却解析)
4. [再帰の計算量を求める手法](#4-再帰の計算量を求める手法)
5. [漸化式の立て方](#5-漸化式の立て方)
6. [マスター定理](#6-マスター定理)
7. [再帰木法](#7-再帰木法)
8. [置換法（帰納法）](#8-置換法帰納法)
9. [一般的な再帰パターン](#9-一般的な再帰パターン)
10. [比較表](#10-比較表)
11. [アンチパターン](#11-アンチパターン)
12. [エッジケース分析](#12-エッジケース分析)
13. [演習問題](#13-演習問題)
14. [FAQ](#14-faq)
15. [まとめ](#15-まとめ)
16. [参考文献](#16-参考文献)

---

## 1. 漸近記法の基礎

計算量解析では、入力サイズ n が十分大きくなったときのアルゴリズムの振る舞いを議論する。
そのために、定数係数や低次項を無視し、支配的な増加率だけを捉える **漸近記法** を用いる。

### 1.1 なぜ漸近記法を使うのか

具体的な実行時間（例: 3.2 秒）は、ハードウェアや実装言語、コンパイラの最適化など多数の要因で変動する。
漸近記法を用いる理由は以下の通りである:

- **ハードウェア非依存**: CPU 速度やメモリ帯域に依存しない評価が可能になる
- **スケーラビリティの予測**: 入力サイズが 10 倍になったとき、実行時間がどの程度増加するかを見積もれる
- **アルゴリズム間の比較**: 同じ問題を解く異なるアルゴリズムの本質的な効率差を議論できる

### 1.2 三つの漸近記法

```
漸近記法の関係図:

              O(g(n))        ── 上界（最悪でもこの程度）
             ╱      ╲
     実際の増加率 f(n)
             ╲      ╱
              Ω(g(n))        ── 下界（少なくともこの程度）

     上界と下界が一致するとき:
              Θ(g(n))        ── タイトな界（ちょうどこの程度）
```

**O 記法（上界）**: ある定数 c > 0 と n_0 > 0 が存在して、すべての n >= n_0 に対し f(n) <= c * g(n) が成り立つとき、f(n) = O(g(n)) と書く。

**Omega 記法（下界）**: ある定数 c > 0 と n_0 > 0 が存在して、すべての n >= n_0 に対し f(n) >= c * g(n) が成り立つとき、f(n) = Omega(g(n)) と書く。

**Theta 記法（タイトな界）**: f(n) = O(g(n)) かつ f(n) = Omega(g(n)) のとき、f(n) = Theta(g(n)) と書く。

### 1.3 漸近記法の計算規則

漸近記法を扱う上で、以下の規則が成り立つ。これらは証明によって裏付けられた性質であり、感覚的な「お約束」ではない。

| 規則 | 内容 | 理由 |
|------|------|------|
| 定数係数の無視 | O(5n) = O(n) | 定義中の c に 5 を吸収できるため |
| 低次項の無視 | O(n^2 + n) = O(n^2) | n が十分大きいと n^2 が n を支配するため |
| 対数の底の無視 | O(log_2 n) = O(log_10 n) | log_a(n) = log_b(n) / log_b(a) より定数倍の違いのみ |
| 加法則 | O(f) + O(g) = O(max(f, g)) | 支配項が全体のオーダーを決定するため |
| 乗法則 | O(f) * O(g) = O(f * g) | 各ステップが独立に繰り返される場合に適用 |

### 1.4 コード例: 漸近記法の直感を確認する

以下のコードは、各計算量の関数がどの程度の速さで増加するかを数値的に確認するものである。

```python
"""
漸近記法の増加率を数値的に比較するプログラム。
なぜこの比較が重要か: アルゴリズム選択時に、入力サイズの増加が
実行時間にどう影響するかを直感的に理解するためである。
"""

import math


def compare_growth_rates(sizes: list[int]) -> None:
    """異なる計算量クラスの増加率を表形式で表示する。

    Args:
        sizes: 比較する入力サイズのリスト
    """
    header = f"{'n':>10} | {'log n':>10} | {'n':>10} | {'n log n':>12} | {'n^2':>12} | {'2^n':>15}"
    separator = "-" * len(header)

    print(header)
    print(separator)

    for n in sizes:
        log_n = math.log2(n) if n > 0 else 0
        n_log_n = n * log_n
        n_squared = n * n
        # 2^n は n が大きいと天文学的数値になるため、上限を設ける
        two_to_n = 2 ** n if n <= 30 else float("inf")

        print(
            f"{n:>10} | "
            f"{log_n:>10.2f} | "
            f"{n:>10} | "
            f"{n_log_n:>12.1f} | "
            f"{n_squared:>12} | "
            f"{two_to_n:>15}"
        )


def main() -> None:
    sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    compare_growth_rates(sizes)
    print()

    # 計算量クラス間の「壁」を実感する
    print("=== n=20 における各関数の値 ===")
    n = 20
    print(f"  log n    = {math.log2(n):.2f}")
    print(f"  n        = {n}")
    print(f"  n log n  = {n * math.log2(n):.1f}")
    print(f"  n^2      = {n**2}")
    print(f"  n^3      = {n**3}")
    print(f"  2^n      = {2**n}")
    print(f"  n!       = {math.factorial(n)}")
    # 2^20 = 1,048,576 に対し 20! = 2,432,902,008,176,640,000
    # この差が「指数時間 vs 階乗時間」の壁を示している


if __name__ == "__main__":
    main()
```

想定される出力（先頭部分）:

```
         n |      log n |          n |      n log n |          n^2 |             2^n
------------------------------------------------------------------------------------
         1 |       0.00 |          1 |          0.0 |            1 |               2
         2 |       1.00 |          2 |          2.0 |            4 |               4
         4 |       2.00 |          4 |          8.0 |           16 |              16
         8 |       3.00 |          8 |         24.0 |           64 |             256
        16 |       4.00 |         16 |         64.0 |          256 |           65536
```

---

## 2. 最悪ケース・平均ケース・最良ケースの解析

同じアルゴリズムでも、入力データの内容によって実行時間は大きく変わる。
計算量解析では、この変動を3つの観点から捉える。

### 2.1 三つのケースの定義

```
入力空間と実行時間の関係図:

  実行時間
  ▲
  │    ×                              ── 最悪ケース W(n)
  │         ×        ×
  │    ×         ×        ×    ×
  │         ×        ×              × ── 平均ケース A(n)
  │    ×              ×        ×
  │                        ×
  │              ×                    ── 最良ケース B(n)
  │
  └─────────────────────────────────→ 入力パターン
           （サイズ n の全入力）

  W(n) = max { T(I) | |I| = n }    すべてのサイズ n の入力 I のうち最大の実行時間
  B(n) = min { T(I) | |I| = n }    すべてのサイズ n の入力 I のうち最小の実行時間
  A(n) = Σ T(I) × Pr(I)            各入力 I の実行時間の期待値
```

**なぜ最悪ケースを重視するのか**: システムの応答時間保証（SLA）やリアルタイムシステムでは、「ほとんどの場合速い」では不十分であり、「どんな入力に対しても一定時間内に応答する」ことが求められるためである。

**なぜ平均ケースも重要か**: 最悪ケースが滅多に起きない場合（例: クイックソートの O(n^2) は非常にまれ）、実用上の性能は平均ケースで評価した方が実態に即する。

### 2.2 コード例: 線形探索の三つのケース

```python
"""
線形探索における最悪・平均・最良ケースの解析。
同じアルゴリズムでも、ターゲットの位置により比較回数が大きく異なることを示す。
"""

import random
import statistics


def linear_search(arr: list[int], target: int) -> tuple[int, int]:
    """線形探索を行い、見つかったインデックスと比較回数を返す。

    Args:
        arr: 探索対象の配列
        target: 探索する値

    Returns:
        (見つかったインデックス, 比較回数) のタプル。
        見つからない場合はインデックスに -1 を返す。
    """
    comparisons = 0
    for i, val in enumerate(arr):
        comparisons += 1
        if val == target:
            return i, comparisons
    return -1, comparisons


def analyze_linear_search(n: int, trials: int = 10000) -> None:
    """線形探索の最悪・平均・最良ケースを実験的に分析する。

    なぜ実験的分析を行うか: 理論値との一致を確認することで、
    漸近解析の妥当性を検証するためである。

    Args:
        n: 配列のサイズ
        trials: 試行回数
    """
    arr = list(range(n))

    # 最良ケース: ターゲットが先頭にある場合
    _, best_comparisons = linear_search(arr, 0)

    # 最悪ケース: ターゲットが存在しない場合
    _, worst_comparisons = linear_search(arr, n + 1)

    # 平均ケース: ターゲットの位置がランダム
    comparison_counts = []
    for _ in range(trials):
        target = random.randint(0, n - 1)
        _, comps = linear_search(arr, target)
        comparison_counts.append(comps)

    avg_comparisons = statistics.mean(comparison_counts)

    print(f"=== 線形探索の解析 (n={n}) ===")
    print(f"  最良ケース: {best_comparisons} 回の比較   (理論値: 1)")
    print(f"  最悪ケース: {worst_comparisons} 回の比較   (理論値: {n})")
    print(f"  平均ケース: {avg_comparisons:.1f} 回の比較 (理論値: {(n + 1) / 2:.1f})")
    print()

    # 理論値との誤差を確認
    theoretical_avg = (n + 1) / 2
    error_pct = abs(avg_comparisons - theoretical_avg) / theoretical_avg * 100
    print(f"  平均ケースの理論値との誤差: {error_pct:.2f}%")
    # 試行回数が十分なら、この誤差は 1% 以下に収まることが想定される


def main() -> None:
    for n in [100, 1000, 10000]:
        analyze_linear_search(n)
        print()


if __name__ == "__main__":
    main()
```

### 2.3 コード例: クイックソートの三つのケース

クイックソートは、ピボットの選び方によって性能が劇的に変わるアルゴリズムの代表例である。

```python
"""
クイックソートにおける最悪・平均・最良ケースの具体例。
ピボット選択がアルゴリズムの性能にどのように影響するかを示す。
"""

import random
import time


def quicksort_count(arr: list[int]) -> tuple[list[int], int]:
    """クイックソート（先頭ピボット）を行い、比較回数も返す。

    なぜ先頭ピボットを使うか: 最悪ケースが発生しやすい実装として、
    ピボット選択の重要性を示すためである。

    Args:
        arr: ソート対象のリスト

    Returns:
        (ソート済みリスト, 比較回数) のタプル
    """
    if len(arr) <= 1:
        return arr[:], 0

    pivot = arr[0]  # 先頭要素をピボットにする（意図的に単純な選択）
    left = []
    right = []
    comparisons = 0

    for x in arr[1:]:
        comparisons += 1
        if x <= pivot:
            left.append(x)
        else:
            right.append(x)

    sorted_left, left_comps = quicksort_count(left)
    sorted_right, right_comps = quicksort_count(right)

    return sorted_left + [pivot] + sorted_right, comparisons + left_comps + right_comps


def demonstrate_quicksort_cases(n: int) -> None:
    """クイックソートの三つのケースを実験的に示す。

    Args:
        n: 配列のサイズ
    """
    print(f"=== クイックソートの解析 (n={n}) ===")

    # 最悪ケース: 既にソート済みの配列（先頭ピボットでは最悪）
    sorted_arr = list(range(n))
    _, worst_comps = quicksort_count(sorted_arr)
    print(f"  最悪ケース（ソート済み）: {worst_comps} 回の比較")
    print(f"    → 想定される計算量: O(n^2) = {n * (n - 1) // 2}")

    # 最良ケース: 毎回中央値がピボットになるケース
    # （実際に毎回中央値になる配列を構築するのは複雑なので、理論値を示す）
    import math
    theoretical_best = n * math.log2(max(n, 1))
    print(f"  最良ケースの理論値: O(n log n) ≈ {theoretical_best:.0f}")

    # 平均ケース: ランダムな配列
    total_comps = 0
    trials = 100
    for _ in range(trials):
        random_arr = list(range(n))
        random.shuffle(random_arr)
        _, comps = quicksort_count(random_arr)
        total_comps += comps
    avg_comps = total_comps / trials
    # 平均ケースの理論値: 2n ln n ≈ 1.39 n log_2 n
    theoretical_avg = 1.39 * n * math.log2(max(n, 1))
    print(f"  平均ケース（{trials}回平均）: {avg_comps:.0f} 回の比較")
    print(f"    → 理論値 1.39 n log n ≈ {theoretical_avg:.0f}")
    print()


def main() -> None:
    # n=500 程度なら最悪ケースでもスタックオーバーフローしない
    for n in [50, 100, 500]:
        demonstrate_quicksort_cases(n)


if __name__ == "__main__":
    main()
```

### 2.4 平均ケース解析の数学的導出

平均ケース解析は、入力の確率分布を仮定した上で、実行時間の期待値を計算する手法である。

**線形探索の平均ケース（ターゲットが配列内に存在する場合）**:

各位置 i (1 <= i <= n) にターゲットがある確率が等確率 1/n のとき:

```
A(n) = Σ_{i=1}^{n} i × (1/n) = (1/n) × n(n+1)/2 = (n+1)/2
```

よって A(n) = Theta((n+1)/2) = Theta(n) である。

**クイックソートの平均ケース**:

ピボットが k 番目に小さい要素になる確率が各 1/n のとき、比較回数の期待値 C(n) は:

```
C(n) = (n - 1) + (1/n) Σ_{k=0}^{n-1} [C(k) + C(n - 1 - k)]
     = (n - 1) + (2/n) Σ_{k=0}^{n-1} C(k)
```

この漸化式を解くと C(n) = 2n ln n + O(n) ≈ 1.39 n log_2 n となる。
この導出が意味するのは、ランダムな入力に対してクイックソートは「ほぼ最適な」 O(n log n) の性能を発揮するということである。

---

## 3. 償却解析

### 3.1 償却解析とは何か

償却解析（Amortized Analysis）は、一連の操作列全体のコストを評価する手法である。
個々の操作の最悪ケースだけを見ると悲観的すぎる場合に、操作列全体の「一操作あたりの平均コスト」を正確に求めるために使う。

**なぜ「単純な最悪ケースの合計」では不十分なのか**: 例えば動的配列の append 操作は、ほとんどが O(1) だが、たまに配列の拡張（コピー）で O(n) かかる。単純に「各操作 O(n)」として合計すると O(n^2) となるが、実際にはそこまで遅くない。償却解析はこの「たまにしか起きない高コスト操作」を正しく扱う。

```
償却解析の直感図:

  コスト
  ▲
  │
n │              *                              *
  │
  │
  │
  │        *                         *
  │
  │    *                    *
  │  *              *
1 │ * * * * * * * * * * * * * * * * * * * * * * * *
  └──────────────────────────────────────────────→ 操作番号
      1 2 3 4 5 6 7 8 9 ...

  大半の操作は O(1) だが、まれに O(n) のスパイクが発生する。
  償却解析は、全操作にわたるコストの合計を正確に見積もる。

  → n 回の操作の合計コスト = O(n)
  → 一操作あたりの償却コスト = O(1)
```

### 3.2 三つの償却解析手法

| 手法 | 考え方 | 適用場面 |
|------|--------|----------|
| 集約法（Aggregate） | 全操作の合計コストを直接計算し、操作数で割る | 計算が直接的に行える場合 |
| 配賦法（Accounting） | 安い操作に「貯金」を上乗せし、高い操作で使う | 操作ごとに均等なコストを割り当てたい場合 |
| ポテンシャル法（Potential） | データ構造の「ポテンシャル関数」を定義し、コストの増減を追跡 | 複雑なデータ構造の解析に適する |

### 3.3 コード例: 動的配列の償却解析

```python
"""
動的配列（倍増戦略）の償却解析。
三つの手法（集約法、配賦法、ポテンシャル法）を実装で示す。
"""


class DynamicArray:
    """倍増戦略を用いた動的配列の実装。

    なぜ倍増戦略を使うか:
    配列が満杯になったとき、容量を 2 倍にする。毎回 1 だけ増やす戦略では
    n 回の append に O(n^2) のコストがかかるが、倍増戦略なら O(n) で済む。
    この差は容量拡張の頻度に由来する。
    """

    def __init__(self) -> None:
        self._capacity = 1
        self._size = 0
        self._data = [None] * self._capacity
        self._total_cost = 0  # コスト追跡用
        self._operation_costs: list[int] = []  # 各操作のコスト記録

    def append(self, value: int) -> int:
        """要素を末尾に追加し、この操作のコスト（要素コピー回数 + 1）を返す。

        Args:
            value: 追加する値

        Returns:
            この操作にかかったコスト
        """
        cost = 1  # 要素の書き込みコスト

        if self._size == self._capacity:
            # 容量拡張: 全要素をコピーするコストが発生
            cost += self._size  # コピーコスト
            new_data = [None] * (self._capacity * 2)
            for i in range(self._size):
                new_data[i] = self._data[i]
            self._data = new_data
            self._capacity *= 2

        self._data[self._size] = value
        self._size += 1
        self._total_cost += cost
        self._operation_costs.append(cost)
        return cost

    @property
    def size(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def total_cost(self) -> int:
        return self._total_cost


def demonstrate_aggregate_method(n: int) -> None:
    """集約法による償却解析のデモンストレーション。

    集約法の考え方:
    n 回の append の合計コストを直接計算する。
    拡張が起きるのは i = 1, 2, 4, 8, ..., 2^k の時点で、
    拡張コストは 1 + 2 + 4 + ... + 2^k <= 2n。
    書き込みコストは n。
    よって合計コスト <= 3n → 一操作あたり O(1)。
    """
    arr = DynamicArray()
    print(f"=== 集約法による解析 (n={n}) ===")
    print(f"{'操作番号':>8} | {'この操作のコスト':>16} | {'累積コスト':>10} | {'容量':>6}")
    print("-" * 50)

    for i in range(1, n + 1):
        cost = arr.append(i)
        if cost > 1 or i <= 5 or i == n:  # 拡張時と先頭・末尾を表示
            print(f"{i:>8} | {cost:>16} | {arr.total_cost:>10} | {arr.capacity:>6}")

    amortized = arr.total_cost / n
    print(f"\n  合計コスト: {arr.total_cost}")
    print(f"  操作回数: {n}")
    print(f"  一操作あたりの償却コスト: {amortized:.2f}")
    print(f"  → 理論的な上界: 3.0 (3n/n)")
    print()


def demonstrate_accounting_method(n: int) -> None:
    """配賦法による償却解析のデモンストレーション。

    配賦法の考え方:
    各 append 操作に「償却コスト 3」を請求する。
      - 1: 要素の書き込み
      - 2: 将来の拡張に備えた「貯金」
    拡張時には、貯金されたコインで全コピーを賄う。
    """
    print(f"=== 配賦法による解析 (n={n}) ===")
    balance = 0  # 貯金残高
    amortized_cost_per_op = 3  # 各操作に請求する償却コスト

    capacity = 1
    size = 0
    all_balanced = True

    for i in range(1, n + 1):
        balance += amortized_cost_per_op  # 3 コインを請求
        balance -= 1  # 書き込みに 1 コイン使用

        if size == capacity:
            # 拡張: size 個のコピーにコインを使用
            balance -= size
            capacity *= 2
            if balance < 0:
                all_balanced = False

        size += 1

    print(f"  各操作への請求額: {amortized_cost_per_op}")
    print(f"  最終的な貯金残高: {balance}")
    print(f"  貯金が負にならなかったか: {'はい' if all_balanced else 'いいえ'}")
    print(f"  → 貯金が常に非負なら、償却コスト {amortized_cost_per_op} は正当")
    print()


def demonstrate_potential_method(n: int) -> None:
    """ポテンシャル法による償却解析のデモンストレーション。

    ポテンシャル法の考え方:
    ポテンシャル関数 Phi(D) = 2 * size - capacity と定義する。
    (ここで D はデータ構造の状態)

    償却コスト = 実コスト + Phi(D_after) - Phi(D_before)

    拡張なしの場合:
      実コスト = 1
      Phi 変化 = 2(size+1) - cap - (2*size - cap) = 2
      償却コスト = 1 + 2 = 3

    拡張ありの場合 (size == capacity のとき):
      実コスト = 1 + size (コピー)
      新容量 = 2 * capacity
      Phi_after = 2(size+1) - 2*capacity = 2*size + 2 - 2*size = 2
      Phi_before = 2*size - capacity = 2*size - size = size
      Phi 変化 = 2 - size
      償却コスト = (1 + size) + (2 - size) = 3
    """
    print(f"=== ポテンシャル法による解析 (n={n}) ===")

    capacity = 1
    size = 0

    for i in range(1, min(n, 20) + 1):
        phi_before = 2 * size - capacity

        actual_cost = 1
        expanded = False
        if size == capacity:
            actual_cost += size
            capacity *= 2
            expanded = True

        size += 1
        phi_after = 2 * size - capacity
        amortized = actual_cost + phi_after - phi_before

        if expanded or i <= 5:
            print(
                f"  操作 {i:>3}: "
                f"実コスト={actual_cost:>4}, "
                f"Phi変化={phi_after - phi_before:>4}, "
                f"償却コスト={amortized:>2}"
                f"{'  ← 拡張' if expanded else ''}"
            )

    print(f"  → すべての操作で償却コスト = 3（定数）")
    print()


def main() -> None:
    n = 64
    demonstrate_aggregate_method(n)
    demonstrate_accounting_method(n)
    demonstrate_potential_method(n)


if __name__ == "__main__":
    main()
```

### 3.4 償却解析の応用例: スタック with MultiPop

```python
"""
MultiPop 付きスタックの償却解析。
push は O(1)、multi_pop(k) は O(min(k, size)) だが、
n 回の操作列全体の償却コストは一操作あたり O(1) であることを示す。
"""


class StackWithMultiPop:
    """MultiPop 操作を持つスタック。

    なぜ償却解析が必要か:
    multi_pop(k) は最悪 O(n) だが、pop できるのは push された要素だけである。
    n 回の操作列中の pop 合計回数は push 回数を超えないため、
    全操作の合計コストは O(n) に収まる。
    """

    def __init__(self) -> None:
        self._stack: list[int] = []
        self._total_cost = 0

    def push(self, value: int) -> int:
        """要素をプッシュする。コスト: 1。"""
        self._stack.append(value)
        self._total_cost += 1
        return 1

    def multi_pop(self, k: int) -> tuple[list[int], int]:
        """最大 k 個の要素をポップする。コスト: min(k, size)。

        Args:
            k: ポップする最大個数

        Returns:
            (取り出した要素のリスト, コスト)
        """
        actual_pops = min(k, len(self._stack))
        popped = []
        for _ in range(actual_pops):
            popped.append(self._stack.pop())
        self._total_cost += actual_pops
        return popped, actual_pops

    @property
    def size(self) -> int:
        return len(self._stack)

    @property
    def total_cost(self) -> int:
        return self._total_cost


def demonstrate_multipop_amortized() -> None:
    """MultiPop スタックの償却解析デモ。"""
    stack = StackWithMultiPop()
    operations = []

    # 操作列: push を多数行い、時折 multi_pop で一気に取り出す
    import random
    random.seed(42)

    n = 100
    for i in range(n):
        if random.random() < 0.7 or stack.size == 0:
            cost = stack.push(i)
            operations.append(("push", cost))
        else:
            k = random.randint(1, stack.size)
            _, cost = stack.multi_pop(k)
            operations.append((f"multi_pop({k})", cost))

    print(f"=== MultiPop スタックの償却解析 ===")
    print(f"  操作回数: {n}")
    print(f"  合計コスト: {stack.total_cost}")
    print(f"  一操作あたりの償却コスト: {stack.total_cost / n:.2f}")
    print(f"  → 理論的な償却コスト: O(1)")
    print()

    # 最悪の個別操作のコストを確認
    max_cost = max(cost for _, cost in operations)
    print(f"  個別操作の最大コスト: {max_cost}")
    print(f"  → 個別の最悪ケースは O(n) だが、償却では O(1)")


if __name__ == "__main__":
    demonstrate_multipop_amortized()
```

### 3.5 よく使われるデータ構造の償却計算量

| データ構造 | 操作 | 最悪ケース | 償却コスト | 理由 |
|------------|------|------------|------------|------|
| 動的配列 | append | O(n) | O(1) | 倍増戦略により拡張頻度が指数的に減少 |
| 動的配列 | pop (末尾) | O(1) | O(1) | 縮小しない場合は常に O(1) |
| 二項ヒープ | insert | O(log n) | O(1) | 繰り上がりの合計コストが制限される |
| Splay 木 | 任意の操作 | O(n) | O(log n) | スプレー操作が木をバランスさせる |
| Union-Find | Union + Find | O(log n) | O(alpha(n)) ≈ O(1) | 経路圧縮とランクによる結合 |

---

## 4. 再帰の計算量を求める手法

再帰アルゴリズムの計算量解析は、まず漸化式を立て、次にそれを解くという二段階で行う。

```
再帰の計算量解析手法:

┌──────────────────────────────────┐
│ Step 1: 漸化式を立てる           │
│   再帰の構造からコスト関係式を   │
│   導出する                       │
└──────────┬───────────────────────┘
           ▼
┌──────────────────────────────────┐
│ Step 2: 漸化式を解く             │
│                                  │
│  ┌─ マスター定理                 │
│  │    T(n)=aT(n/b)+f(n) の形    │
│  │    → 3 ケースで即座に解決     │
│  │                               │
│  ├─ 再帰木法                     │
│  │    各レベルのコストを足し上げ  │
│  │    → 視覚的に理解しやすい     │
│  │                               │
│  └─ 置換法                       │
│       解を予想して帰納法で証明    │
│       → 最も厳密な証明           │
└──────────────────────────────────┘
```

### なぜ漸化式を経由するのか

再帰アルゴリズムの実行時間は、そのアルゴリズム自身の小さなインスタンスの実行時間で表現される。
これはそのまま漸化式の構造と一致する。漸化式を解くことで、入力サイズ n の閉じた式（非再帰的な式）が得られ、漸近的な増加率を議論できるようになる。

---

## 5. 漸化式の立て方

### 5.1 漸化式を立てる手順

1. **ベースケースを特定する**: 再帰が停止する条件と、その時のコスト
2. **再帰呼び出しの構造を特定する**: 何回呼ばれ、各呼び出しの問題サイズはいくつか
3. **再帰以外のコストを特定する**: 分割・統合・その他の処理にかかるコスト

### 5.2 例: マージソート

```python
"""
マージソートの漸化式導出。
各行のコストをコメントで注釈し、漸化式の構成要素を明示する。
"""


def merge_sort(arr: list[int]) -> list[int]:
    """マージソート — T(n) = 2T(n/2) + O(n)

    漸化式の導出:
    - ベースケース: len(arr) <= 1 のとき O(1)
    - 再帰呼び出し: 2 回、各サイズ n/2 → 2T(n/2)
    - 統合コスト: merge は各要素を 1 回ずつ見る → O(n)
    """
    if len(arr) <= 1:                  # O(1) — ベースケース
        return arr

    mid = len(arr) // 2                # O(1) — 分割位置の計算
    left = merge_sort(arr[:mid])       # T(n/2) — 左半分を再帰的にソート
    right = merge_sort(arr[mid:])      # T(n/2) — 右半分を再帰的にソート
    return merge(left, right)          # O(n) — 統合


def merge(left: list[int], right: list[int]) -> list[int]:
    """二つのソート済みリストを統合する。コスト: O(n)。

    なぜ O(n) か: 各要素は最大 1 回だけ比較・コピーされ、
    両リストの合計要素数は n であるため。
    """
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result


def verify_merge_sort() -> None:
    """マージソートの正当性と計算量を検証する。"""
    import random
    import time

    sizes = [1000, 2000, 4000, 8000, 16000]
    print("=== マージソートの計算量検証 ===")
    print(f"{'n':>8} | {'時間(ms)':>10} | {'比率':>8} | {'想定される比率':>14}")
    print("-" * 50)

    prev_time = None
    for n in sizes:
        arr = list(range(n))
        random.shuffle(arr)

        start = time.perf_counter()
        merge_sort(arr)
        elapsed = (time.perf_counter() - start) * 1000

        if prev_time and prev_time > 0:
            ratio = elapsed / prev_time
            # O(n log n) の場合、n を 2 倍にすると
            # 2n log(2n) / (n log n) = 2(1 + log2/logn) ≈ 2+ (n が大きいとき)
            print(f"{n:>8} | {elapsed:>10.2f} | {ratio:>8.2f} | ~2.0-2.3")
        else:
            print(f"{n:>8} | {elapsed:>10.2f} | {'---':>8} | ---")

        prev_time = elapsed


if __name__ == "__main__":
    verify_merge_sort()
```

漸化式: `T(n) = 2T(n/2) + cn` （c は定数）

### 5.3 例: ストラッセンの行列乗算

通常の行列乗算は 8 回の n/2 サイズの部分行列乗算を行うため T(n) = 8T(n/2) + O(n^2) → O(n^3)。
ストラッセンは乗算回数を 7 回に削減: T(n) = 7T(n/2) + O(n^2) → O(n^{log_2 7}) ≈ O(n^{2.807})。

```
通常の行列乗算 vs ストラッセン:

  通常:                     ストラッセン:
  a = 8, b = 2, f(n) = n²   a = 7, b = 2, f(n) = n²
  n^(log₂ 8) = n³           n^(log₂ 7) ≈ n^2.807
  Case 1 → Θ(n³)           Case 1 → Θ(n^2.807)

  → 乗算を 1 回減らすだけで、漸近的に高速化
```

---

## 6. マスター定理

### 6.1 一般形

マスター定理は、分割統治法の漸化式 T(n) = aT(n/b) + f(n) を、f(n) と n^{log_b(a)} の比較だけで解く定理である。

```
T(n) = aT(n/b) + f(n)

  a : 再帰呼び出しの回数（a >= 1）
  b : 問題サイズの縮小率（b > 1）
  f(n) : 分割・統合のコスト（非負）

  キーとなる値: n^(log_b(a))
    → これは「再帰木の葉の総数」に対応する
    → 再帰木は深さ log_b(n) で、各レベルで a 倍に分岐
    → 葉の数 = a^(log_b(n)) = n^(log_b(a))
```

### 6.2 三つのケース

```
ケース判定フローチャート:

  f(n) と n^(log_b(a)) を比較
         │
    ┌────┼────────────────┐
    ▼    ▼                ▼
 Case 1  Case 2         Case 3
 f(n)が   f(n)が         f(n)が
 小さい   同程度         大きい

Case 1: f(n) = O(n^(log_b(a) - epsilon))  （epsilon > 0）
  → 葉のコストが支配的
  → T(n) = Theta(n^(log_b(a)))
  → 直感: 再帰の「拡散」が速く、葉の数がコストを決める

Case 2: f(n) = Theta(n^(log_b(a)))
  → 各レベルのコストが均等
  → T(n) = Theta(n^(log_b(a)) * log n)
  → 直感: 各レベルで同じコスト × レベル数 = ×log n

Case 3: f(n) = Omega(n^(log_b(a) + epsilon))  （epsilon > 0）
  かつ正則条件: a*f(n/b) <= c*f(n) (c < 1, 十分大きな n)
  → ルートのコストが支配的
  → T(n) = Theta(f(n))
  → 直感: 各レベルで「統合コスト」が急速に減衰
```

**なぜ正則条件（Case 3）が必要なのか**: f(n) が n^{log_b(a)} より漸近的に大きくても、f の減少速度が不規則だと合計が発散する可能性がある。正則条件は「f のコストがレベルごとに一定割合で減少する」ことを保証し、幾何級数の収束を担保する。

### 6.3 マスター定理の適用例

```python
"""
マスター定理の適用例を体系的に示す。
各例で a, b, f(n) を特定し、どのケースに該当するかを判定する。
"""

import math


def master_theorem_analyze(
    a: int, b: int, f_desc: str, f_degree: float, algorithm: str
) -> None:
    """マスター定理によるケース判定を行う。

    Args:
        a: 再帰呼び出し回数
        b: 問題サイズの縮小率
        f_desc: f(n) の説明文字列
        f_degree: f(n) が Theta(n^f_degree) であるとき、その次数
        algorithm: アルゴリズム名
    """
    critical_exp = math.log(a) / math.log(b)  # log_b(a)
    print(f"--- {algorithm} ---")
    print(f"  漸化式: T(n) = {a}T(n/{b}) + {f_desc}")
    print(f"  a={a}, b={b}, n^(log_{b}({a})) = n^{critical_exp:.3f}")
    print(f"  f(n) = {f_desc} → 次数 {f_degree}")

    if f_degree < critical_exp:
        print(f"  f(n) の次数 {f_degree} < {critical_exp:.3f} → Case 1")
        print(f"  T(n) = Theta(n^{critical_exp:.3f})")
    elif abs(f_degree - critical_exp) < 0.001:
        print(f"  f(n) の次数 {f_degree} ≈ {critical_exp:.3f} → Case 2")
        print(f"  T(n) = Theta(n^{critical_exp:.3f} * log n)")
    else:
        print(f"  f(n) の次数 {f_degree} > {critical_exp:.3f} → Case 3")
        print(f"  T(n) = Theta({f_desc})")
    print()


def main() -> None:
    print("=== マスター定理の適用例集 ===\n")

    # 例1: マージソート
    master_theorem_analyze(2, 2, "n", 1.0, "マージソート")

    # 例2: 二分探索
    master_theorem_analyze(1, 2, "1", 0.0, "二分探索")

    # 例3: カラツバ乗算
    master_theorem_analyze(3, 2, "n", 1.0, "カラツバ乗算")

    # 例4: ストラッセン行列乗算
    master_theorem_analyze(7, 2, "n^2", 2.0, "ストラッセン行列乗算")

    # 例5: 二分木走査
    master_theorem_analyze(2, 2, "1", 0.0, "二分木走査")

    # 例6: T(n) = 4T(n/2) + n^2
    master_theorem_analyze(4, 2, "n^2", 2.0, "4T(n/2) + n^2")

    # 例7: T(n) = 4T(n/2) + n^3
    master_theorem_analyze(4, 2, "n^3", 3.0, "4T(n/2) + n^3")

    # 例8: クイックセレクト（平均）
    master_theorem_analyze(1, 2, "n", 1.0, "クイックセレクト(平均)")


if __name__ == "__main__":
    main()
```

### 6.4 マスター定理が適用できないケース

マスター定理には明確な適用限界がある。以下のケースでは適用できない:

**1. f(n) が多項式的に異ならない場合（ギャップケース）**

```
T(n) = 2T(n/2) + n log n

  a=2, b=2 → n^(log_2(2)) = n
  f(n) = n log n

  n log n は n^1 より大きいが、n^(1+epsilon) より小さい（任意の epsilon > 0 に対して）。
  → Case 2 と Case 3 の間に落ち、標準のマスター定理は適用不可。

  解: Akra-Bazzi 定理を適用するか、再帰木法で直接解く。
  結果: T(n) = Theta(n log^2 n)
```

**2. 問題サイズの分割が均等でない場合**

```
T(n) = T(n/3) + T(2n/3) + n

  再帰呼び出しのサイズが異なる → T(n) = aT(n/b) の形でない。
  → マスター定理は適用不可。

  解: 再帰木法で解く。
  最も深いパスは n → (2/3)n → (2/3)^2 n → ... → 1 で、
  深さは log_{3/2}(n)。各レベルのコストは O(n)。
  結果: T(n) = O(n log n)
```

**3. a < 1 または b <= 1 の場合**

マスター定理は a >= 1 かつ b > 1 を前提とする。これらの条件が満たされない場合は適用できない。

---

## 7. 再帰木法

再帰木法は、再帰の展開を木構造として可視化し、各レベルのコストを合計する手法である。
マスター定理の背後にある直感を理解するための最も強力なツールでもある。

### 7.1 T(n) = 2T(n/2) + n の再帰木

```
レベル 0:              n                           → コスト: n
                    /     \
レベル 1:        n/2       n/2                     → コスト: n/2 + n/2 = n
                / \        / \
レベル 2:    n/4  n/4   n/4  n/4                  → コスト: 4*(n/4) = n
              / \  / \  / \  / \
レベル 3: n/8 ...                                  → コスト: 8*(n/8) = n
             :
             :
レベル k:  2^k 個のノード、各サイズ n/2^k          → コスト: n

レベル log₂n: n 個の葉、各サイズ 1               → コスト: n

合計: n × (log₂n + 1) = Θ(n log n)

なぜ各レベルのコストが n になるか:
  レベル k にはノードが 2^k 個あり、各ノードのサイズは n/2^k。
  各ノードの「統合コスト」は n/2^k に比例する。
  2^k × (n/2^k) = n → レベルに依存しない定数。
```

### 7.2 T(n) = 3T(n/4) + cn^2 の再帰木

```
レベル 0:                cn²                           → コスト: cn²
                      /   |   \
レベル 1:      c(n/4)²  c(n/4)²  c(n/4)²             → コスト: 3c(n/4)² = (3/16)cn²
               / | \    / | \    / | \
レベル 2:    各 c(n/16)²                               → コスト: 9c(n/16)² = (3/16)²cn²
              :
レベル k:    3^k 個のノード、各サイズ n/4^k            → コスト: (3/16)^k × cn²

  合計: cn² × Σ_{k=0}^{∞} (3/16)^k
       = cn² × 1/(1 - 3/16)
       = cn² × 16/13
       = Θ(n²)

  なぜ幾何級数が収束するか: 公比 3/16 < 1 であるため。
  これは Case 3 に対応する: f(n) = n² が支配的で、
  深いレベルほどコストが急速に減衰する。
```

### 7.3 T(n) = T(n/3) + T(2n/3) + n の再帰木（不均等分割）

```
レベル 0:                    n                           → コスト: n
                          /     \
レベル 1:             n/3       2n/3                     → コスト: n/3 + 2n/3 = n
                     / \        / \
レベル 2:        n/9  2n/9  2n/9  4n/9                  → コスト: n
                 :                  :
                 :                  :

  最短パス: n → n/3 → n/9 → ... → 1   深さ log₃ n
  最長パス: n → 2n/3 → 4n/9 → ... → 1  深さ log_{3/2} n

  各レベルのコストは最大 n（葉に近いレベルではやや少ない）。
  レベル数は log_{3/2} n まで。

  合計: O(n × log_{3/2} n) = O(n log n)

  → 不均等分割でも、各レベルの合計コストが O(n) 程度なら
    全体は O(n log n) に収まる。
```

---

## 8. 置換法（帰納法）

置換法は、漸化式の解を「予想」し、数学的帰納法で正しさを証明する手法である。

### 8.1 手順

1. **解を予想する**（再帰木法などの結果、または経験から）
2. **帰納法の仮定を置く**: T(k) <= c * g(k) がすべての k < n に対して成り立つと仮定
3. **帰納ステップを証明する**: T(n) <= c * g(n) を導出
4. **ベースケースを確認する**: T(n_0) <= c * g(n_0) が成り立つ c を確認

### 8.2 例: T(n) = 2T(n/2) + n を O(n log n) と証明

```
予想: T(n) <= c * n * log(n) （c は適切な定数）

帰納法の仮定: すべての k < n に対し T(k) <= c * k * log(k)

帰納ステップ:
  T(n) = 2T(n/2) + n
       <= 2 * c * (n/2) * log(n/2) + n     （帰納法の仮定を適用）
       = c * n * (log(n) - log(2)) + n      （log(n/2) = log(n) - 1 を使用）
       = c * n * log(n) - c * n + n
       = c * n * log(n) - (c - 1) * n

  c >= 1 ならば -(c-1)*n <= 0 なので:
       <= c * n * log(n)

  よって T(n) <= c * n * log(n) が成立。 ■

ベースケース: T(1) = d（定数）とすると、
  c * 1 * log(1) = 0 なので T(1) <= c * 1 * log(1) は成り立たない。
  → n >= 2 をベースケースとし、T(2) <= c * 2 * log(2) = 2c。
  T(2) = 2T(1) + 2 = 2d + 2 なので、c >= d + 1 とすれば成立。
```

### 8.3 置換法の落とし穴

```python
"""
置換法でよくある間違いを示す。
"""

# 間違い1: 帰納法の仮定を「弱すぎる」形で置く
#
# 予想: T(n) = O(n) で T(n) = 2T(n/2) + n を証明しようとする
#
# T(n) = 2T(n/2) + n
#      <= 2 * c * (n/2) + n
#      = cn + n
#      = (c+1)n
#      ≠ cn   ← c が「育ってしまう」ため証明できない
#
# → これは正しい。T(n) = Θ(n log n) であって O(n) ではないから。

# 間違い2: 低次項を無視して「うまくいった」と誤認する
#
# 予想: T(n) <= cn で T(n) = T(n/2) + T(n/2) + 1 を証明
#
# T(n) <= c(n/2) + c(n/2) + 1 = cn + 1  ← 「ほぼ cn」だが cn 以下ではない！
#
# 正しい対処: T(n) <= cn - d とより強い仮定を置く
# T(n) <= c(n/2) - d + c(n/2) - d + 1 = cn - 2d + 1 <= cn - d (d >= 1)
```

---

## 9. 一般的な再帰パターン

### パターン1: 線形再帰 T(n) = T(n-1) + O(1) → O(n)

```python
def factorial(n: int) -> int:
    """階乗を再帰的に計算する。T(n) = T(n-1) + O(1) → O(n)

    なぜ O(n) か: 再帰の深さが n で、各レベルのコストが O(1) であるため。
    """
    if n <= 1:
        return 1
    return n * factorial(n - 1)


# 検証
assert factorial(0) == 1
assert factorial(1) == 1
assert factorial(5) == 120
assert factorial(10) == 3628800
```

### パターン2: 線形再帰（各レベルで O(n) の仕事） T(n) = T(n-1) + O(n) → O(n^2)

```python
def selection_sort(arr: list[int]) -> list[int]:
    """選択ソート。T(n) = T(n-1) + O(n) → O(n^2)

    なぜ O(n^2) か: 各ステップで残り要素から最小値を見つけるのに O(n)、
    それを n 回繰り返すため。漸化式を展開すると
    T(n) = n + (n-1) + (n-2) + ... + 1 = n(n+1)/2 = Θ(n^2)。
    """
    arr = arr[:]
    for i in range(len(arr)):
        min_idx = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr


# 検証
assert selection_sort([3, 1, 4, 1, 5]) == [1, 1, 3, 4, 5]
assert selection_sort([]) == []
assert selection_sort([1]) == [1]
```

### パターン3: 二分再帰（分割のみ） T(n) = T(n/2) + O(1) → O(log n)

```python
def binary_search(arr: list[int], target: int) -> int:
    """二分探索。T(n) = T(n/2) + O(1) → O(log n)

    なぜ O(log n) か: 各ステップで探索範囲が半分になり、
    範囲が 1 になるまで log₂n 回の比較で済むため。
    """
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1


# 検証
arr = [1, 3, 5, 7, 9, 11, 13]
assert binary_search(arr, 7) == 3
assert binary_search(arr, 1) == 0
assert binary_search(arr, 13) == 6
assert binary_search(arr, 4) == -1
```

### パターン4: 繰り返し二乗法 T(n) = T(n/2) + O(1) → O(log n)

```python
def power(x: float, n: int) -> float:
    """繰り返し二乗法。T(n) = T(n/2) + O(1) → O(log n)

    なぜ O(log n) か: 指数を 2 で割っていくため、再帰の深さは log₂n。
    各レベルでは乗算 1 回（O(1)）のみ。

    x^n = (x^(n/2))^2       (n が偶数)
    x^n = x * (x^(n-1/2))^2 (n が奇数)
    """
    if n == 0:
        return 1
    if n < 0:
        return 1 / power(x, -n)
    if n % 2 == 0:
        half = power(x, n // 2)
        return half * half
    else:
        return x * power(x, n - 1)


# 検証
assert power(2, 10) == 1024
assert power(3, 0) == 1
assert abs(power(2, -1) - 0.5) < 1e-10
```

### パターン5: 複数分岐再帰 T(n) = T(n-1) + T(n-2) + O(1) → O(phi^n)

```python
def fibonacci_naive(n: int) -> int:
    """素朴なフィボナッチ。T(n) = T(n-1) + T(n-2) + O(1) → O(phi^n)

    なぜ指数時間か: 再帰木の各ノードが 2 つの子を持ち、
    深さ n まで「ほぼ」完全二分木になるため。正確には黄金比
    phi = (1 + sqrt(5)) / 2 ≈ 1.618 のべき乗で増加する。
    """
    if n <= 1:
        return n
    return fibonacci_naive(n - 1) + fibonacci_naive(n - 2)


def fibonacci_dp(n: int) -> int:
    """動的計画法によるフィボナッチ。O(n) 時間、O(1) 空間。

    メモ化や DP テーブルで重複計算を排除することで、
    O(phi^n) → O(n) に改善される。
    """
    if n <= 1:
        return n
    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr
    return curr


# 検証
for i in range(10):
    assert fibonacci_naive(i) == fibonacci_dp(i)
```

### パターン6: 指数再帰 T(n) = 2T(n-1) + O(1) → O(2^n)

```python
def hanoi(n: int, source: str = "A", target: str = "C", auxiliary: str = "B") -> int:
    """ハノイの塔。T(n) = 2T(n-1) + O(1) → O(2^n)

    なぜ O(2^n) か: n 枚を移動するには、上の n-1 枚を 2 回移動し、
    さらに最大の円盤を 1 回移動する。
    T(n) = 2T(n-1) + 1 → T(n) = 2^n - 1。

    Returns:
        移動回数
    """
    if n == 0:
        return 0
    moves = 0
    moves += hanoi(n - 1, source, auxiliary, target)
    moves += 1  # 最大の円盤を移動
    moves += hanoi(n - 1, auxiliary, target, source)
    return moves


# 検証: T(n) = 2^n - 1
for n in range(1, 15):
    assert hanoi(n) == 2**n - 1, f"n={n}: {hanoi(n)} != {2**n - 1}"

print("ハノイの塔の移動回数:")
for n in [1, 5, 10, 15, 20]:
    print(f"  n={n:>2}: {2**n - 1:>8} 回")
```

---

## 10. 比較表

### 表1: マスター定理の3ケース詳細比較

| ケース | 条件 | 結果 | 直感的説明 | 再帰木での理解 |
|--------|------|------|------------|----------------|
| Case 1 | f(n) = O(n^{log_b(a) - epsilon}) | Theta(n^{log_b(a)}) | 葉のコストが支配 | 下層に向かってコスト増大、葉の合計が全体を決定 |
| Case 2 | f(n) = Theta(n^{log_b(a)}) | Theta(n^{log_b(a)} * log n) | 各レベル均等 | 全レベルが等しいコスト、レベル数分だけ積算 |
| Case 3 | f(n) = Omega(n^{log_b(a) + epsilon}) + 正則条件 | Theta(f(n)) | ルートが支配 | 上層に向かってコスト増大、ルートの寄与が支配的 |

### 表2: 代表的な漸化式と解の一覧

| 漸化式 | 解 | アルゴリズム例 | 解法の根拠 |
|--------|-----|---------------|-----------|
| T(n) = T(n-1) + O(1) | O(n) | 線形走査・階乗 | 合計 = 1+1+...+1 = n |
| T(n) = T(n-1) + O(n) | O(n^2) | 選択ソート・挿入ソート | 合計 = n+(n-1)+...+1 |
| T(n) = T(n/2) + O(1) | O(log n) | 二分探索 | マスター定理 Case 2 |
| T(n) = T(n/2) + O(n) | O(n) | クイックセレクト(平均) | マスター定理 Case 3 |
| T(n) = 2T(n/2) + O(1) | O(n) | 二分木走査 | マスター定理 Case 1 |
| T(n) = 2T(n/2) + O(n) | O(n log n) | マージソート | マスター定理 Case 2 |
| T(n) = 2T(n/2) + O(n^2) | O(n^2) | 非効率な分割統治 | マスター定理 Case 3 |
| T(n) = 3T(n/2) + O(n) | O(n^{1.585}) | カラツバ乗算 | マスター定理 Case 1 |
| T(n) = 7T(n/2) + O(n^2) | O(n^{2.807}) | ストラッセン行列乗算 | マスター定理 Case 1 |
| T(n) = 2T(n-1) + O(1) | O(2^n) | ハノイの塔 | 展開: 2^n - 1 |
| T(n) = T(n-1) + T(n-2) + O(1) | O(phi^n) | 素朴なフィボナッチ | 特性方程式の解 |

### 表3: 計算量クラスの増加率比較

| n | O(1) | O(log n) | O(n) | O(n log n) | O(n^2) | O(2^n) |
|---|------|----------|------|------------|--------|--------|
| 1 | 1 | 0 | 1 | 0 | 1 | 2 |
| 10 | 1 | 3.3 | 10 | 33 | 100 | 1,024 |
| 100 | 1 | 6.6 | 100 | 664 | 10,000 | 1.27 x 10^30 |
| 1,000 | 1 | 10.0 | 1,000 | 9,966 | 1,000,000 | -- |
| 10,000 | 1 | 13.3 | 10,000 | 132,877 | 100,000,000 | -- |
| 100,000 | 1 | 16.6 | 100,000 | 1,660,964 | 10,000,000,000 | -- |

「--」は値が天文学的に大きく実用上計算不可能であることを示す。

---

## 11. アンチパターン

### アンチパターン1: マスター定理の適用条件を確認しない

```python
"""
マスター定理の「ギャップケース」に陥る例。
"""

# BAD: T(n) = 2T(n/2) + n log n にマスター定理を直接適用しようとする
#
# a=2, b=2 → n^(log_2(2)) = n^1 = n
# f(n) = n log n
#
# Case 2 に該当するか？ → Case 2 は f(n) = Θ(n^(log_b(a))) を要求。
#   n log n ≠ Θ(n) なので Case 2 ではない。
#
# Case 3 に該当するか？ → Case 3 は f(n) = Ω(n^(1+ε)) を要求。
#   n log n = O(n^(1+ε)) for any ε > 0 なので Case 3 でもない。
#
# → 標準のマスター定理は適用不可（ギャップケース）
# → 拡張マスター定理を適用: f(n) = Θ(n log^k n) で k=1 のとき
#   T(n) = Θ(n log^(k+1) n) = Θ(n log^2 n)
#
# GOOD: 適用前に必ず 3 ケースのどれに該当するか確認し、
#        どれにも該当しない場合は再帰木法や Akra-Bazzi 定理を使う。
```

### アンチパターン2: 再帰の深さと呼び出し回数を混同する

```python
"""
再帰の「深さ」と「呼び出し総数」は別の概念である。
"""

# BAD: 「二分再帰だから O(log n)」と誤解する
def count_all(n: int) -> int:
    """T(n) = T(n-1) + T(n-2) + O(1)

    誤った推論: 「再帰が 2 つに分かれるから O(log n)」
    正しい分析: 深さは O(n)、呼び出し回数は O(φ^n)
    """
    if n <= 0:
        return 0
    return 1 + count_all(n - 1) + count_all(n - 2)


# GOOD: 深さと呼び出し回数を区別して分析する
#
#   再帰木の深さ: 最も深いパスの長さ → スタック使用量に影響
#   呼び出し回数: 全ノード数 → 時間計算量に影響
#
#   例: フィボナッチ再帰
#     深さ: O(n) — 左端のパスが n → n-1 → n-2 → ... → 0
#     呼び出し回数: O(φ^n) — ほぼ完全二分木に近い構造
```

### アンチパターン3: 平均ケースを「典型ケース」と同一視する

```python
"""
平均ケースは確率分布に基づく期待値であり、
「よくある入力」での性能とは必ずしも一致しない。
"""

# BAD: 「クイックソートの平均 O(n log n) だから、
#        ほぼすべての入力で O(n log n) になる」と誤解

# 正しい理解:
# 1. 平均ケースは「すべてのサイズ n の入力が等確率で出現する」
#    という仮定のもとでの期待値
# 2. 実際のアプリケーションでは入力に偏りがある場合がある
#    例: ほぼソート済みデータが頻繁に入力されるログ処理システム
# 3. 敵対的入力（adversarial input）が存在する環境では、
#    最悪ケースの保証が必要

# GOOD: 入力の分布を考慮してアルゴリズムを選択する
# - ランダムな入力が保証される → 平均ケースで評価可能
# - 敵対的入力の可能性あり → 最悪ケースの保証が必要
# - 特定のパターンが多い → そのパターンでの性能を個別に評価
```

### アンチパターン4: 再帰のオーバーヘッドを無視する

```python
"""
漸近的に同等でも、再帰のオーバーヘッドが定数係数に影響する。
"""

import time


def sum_recursive(n: int) -> int:
    """再帰版の合計。時間計算量: O(n)。"""
    if n <= 0:
        return 0
    return n + sum_recursive(n - 1)


def sum_iterative(n: int) -> int:
    """反復版の合計。時間計算量: O(n)。"""
    total = 0
    for i in range(1, n + 1):
        total += i
    return total


def compare_overhead() -> None:
    """再帰と反復のオーバーヘッド差を測定する。

    漸近記法では両方 O(n) だが、再帰版は関数呼び出しのオーバーヘッド
    （スタックフレームの生成・破棄）により定数係数が大きくなる。
    """
    import sys
    sys.setrecursionlimit(20000)

    n = 10000
    start = time.perf_counter()
    result_rec = sum_recursive(n)
    time_rec = time.perf_counter() - start

    start = time.perf_counter()
    result_iter = sum_iterative(n)
    time_iter = time.perf_counter() - start

    assert result_rec == result_iter

    print(f"n = {n}")
    print(f"  再帰版: {time_rec * 1000:.2f} ms")
    print(f"  反復版: {time_iter * 1000:.2f} ms")
    print(f"  比率: {time_rec / time_iter:.1f}x")
    print(f"  → 漸近的には同じ O(n) だが、定数係数が異なる")


if __name__ == "__main__":
    compare_overhead()
```

---

## 12. エッジケース分析

### エッジケース1: 入力サイズが非常に小さい場合の漸近解析の限界

漸近記法は n → ∞ での振る舞いを記述するため、小さな n では理論と実態が乖離することがある。

```python
"""
小さな入力サイズでは、漸近的に「劣る」アルゴリズムが
漸近的に「優れる」アルゴリズムより高速になるケースを示す。

なぜこの現象が起きるか:
O(n^2) アルゴリズムの定数係数が O(n log n) アルゴリズムより
十分小さい場合、n が小さいうちは定数係数の差が支配する。
"""

import time
import random


def insertion_sort(arr: list[int]) -> list[int]:
    """挿入ソート。最悪 O(n^2) だが定数係数が小さい。"""
    arr = arr[:]
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr


def merge_sort_full(arr: list[int]) -> list[int]:
    """マージソート。O(n log n) だが定数係数がやや大きい。"""
    if len(arr) <= 1:
        return arr[:]
    mid = len(arr) // 2
    left = merge_sort_full(arr[:mid])
    right = merge_sort_full(arr[mid:])
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


def find_crossover_point() -> None:
    """挿入ソートとマージソートの交差点を実験的に見つける。

    想定される結果: n が 20-50 程度で交差する。
    これが Tim Sort（Python の組み込みソート）が小さな部分配列に
    挿入ソートを使う理由である。
    """
    print("=== 小さな入力サイズでの比較 ===")
    print(f"{'n':>6} | {'挿入ソート(ms)':>14} | {'マージソート(ms)':>14} | {'速い方':>8}")
    print("-" * 55)

    trials = 1000
    for n in [5, 10, 15, 20, 30, 50, 100, 200, 500]:
        # 挿入ソートの計測
        total_ins = 0
        for _ in range(trials):
            arr = random.sample(range(n * 10), n)
            start = time.perf_counter()
            insertion_sort(arr)
            total_ins += time.perf_counter() - start

        # マージソートの計測
        total_merge = 0
        for _ in range(trials):
            arr = random.sample(range(n * 10), n)
            start = time.perf_counter()
            merge_sort_full(arr)
            total_merge += time.perf_counter() - start

        ins_ms = total_ins / trials * 1000
        merge_ms = total_merge / trials * 1000
        winner = "挿入" if ins_ms < merge_ms else "マージ"
        print(f"{n:>6} | {ins_ms:>14.4f} | {merge_ms:>14.4f} | {winner:>8}")

    print()
    print("  → n が小さいとき、O(n^2) の挿入ソートが O(n log n) のマージソートに勝つ。")
    print("    これは漸近記法が定数係数を無視することの実用的帰結である。")


if __name__ == "__main__":
    find_crossover_point()
```

**教訓**: 漸近記法はアルゴリズムの「スケーラビリティ」を評価するツールであり、小さな n での絶対性能を保証するものではない。実用的なソートアルゴリズム（Tim Sort、Introsort など）は、n が小さい部分配列に対して挿入ソートに切り替えることで、漸近的優位性と定数係数の優位性を両立させている。

### エッジケース2: 再帰の漸化式でベースケースのコストが無視できない場合

```python
"""
ベースケースのコストが O(1) でない場合、漸化式の解が変わることを示す。
"""


def matrix_multiply_recursive(
    A: list[list[float]], B: list[list[float]], n: int
) -> list[list[float]]:
    """再帰的行列乗算（単純な分割統治）。

    ベースケースが 1x1 行列のとき: T(1) = O(1)
    → T(n) = 8T(n/2) + O(n^2) → O(n^3)

    もしベースケースを k×k（k は定数）にして直接計算すると:
    T(k) = O(k^3) ≈ O(1)  （k が定数なら）
    漸近的な結果は変わらないが、定数係数が改善される。

    なぜこれが重要か: 実装では再帰のベースケースを適切に設定することで、
    関数呼び出しのオーバーヘッドを削減できる。ストラッセンのアルゴリズムでも、
    n が十分小さくなったら通常の O(n^3) 行列乗算に切り替える。
    """
    if n == 1:
        return [[A[0][0] * B[0][0]]]

    mid = n // 2

    # 部分行列の抽出（簡略化のためスライスを使用）
    A11 = [row[:mid] for row in A[:mid]]
    A12 = [row[mid:] for row in A[:mid]]
    A21 = [row[:mid] for row in A[mid:]]
    A22 = [row[mid:] for row in A[mid:]]

    B11 = [row[:mid] for row in B[:mid]]
    B12 = [row[mid:] for row in B[:mid]]
    B21 = [row[:mid] for row in B[mid:]]
    B22 = [row[mid:] for row in B[mid:]]

    # 8 回の再帰呼び出し
    C11 = matrix_add(
        matrix_multiply_recursive(A11, B11, mid),
        matrix_multiply_recursive(A12, B21, mid),
    )
    C12 = matrix_add(
        matrix_multiply_recursive(A11, B12, mid),
        matrix_multiply_recursive(A12, B22, mid),
    )
    C21 = matrix_add(
        matrix_multiply_recursive(A21, B11, mid),
        matrix_multiply_recursive(A22, B21, mid),
    )
    C22 = matrix_add(
        matrix_multiply_recursive(A21, B12, mid),
        matrix_multiply_recursive(A22, B22, mid),
    )

    # 結果の統合
    result = []
    for i in range(mid):
        result.append(C11[i] + C12[i])
    for i in range(mid):
        result.append(C21[i] + C22[i])
    return result


def matrix_add(
    A: list[list[float]], B: list[list[float]]
) -> list[list[float]]:
    """行列の加算。O(n^2)。"""
    n = len(A)
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]


def verify_recursive_matrix_multiply() -> None:
    """再帰的行列乗算の正当性を検証する。"""
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    C = matrix_multiply_recursive(A, B, 2)
    # 期待値: [[19, 22], [43, 50]]
    assert C == [[19, 22], [43, 50]], f"Got {C}"

    # 4x4 行列のテスト
    A4 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    B4 = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    C4 = matrix_multiply_recursive(A4, B4, 4)
    assert C4 == B4, "単位行列との積は元の行列に等しいはず"

    print("再帰的行列乗算: 全テスト通過")


if __name__ == "__main__":
    verify_recursive_matrix_multiply()
```

### エッジケース3: 償却解析で「負の貯金」が生じるシナリオ

```
償却コストの設定が不適切な場合、「貯金」が負になり得る。

例: 動的配列で倍増ではなく 1.5 倍に拡張する場合

  容量拡張の頻度が倍増戦略より多くなるため、
  1 操作あたりの償却コスト 3 では足りない場合がある。

  配賦法で正しく分析するには:
  - 拡張係数 α に基づいて、必要な「前払い額」を計算する
  - α = 2 の場合: 償却コスト 3 で十分
  - α = 1.5 の場合: 償却コストを増やす必要がある

  正しい計算:
  拡張係数 α で、各要素は最大 1/(α-1) 回コピーされる。
  α = 2 → 各要素は最大 1/(2-1) = 1 回コピー → 償却コスト ≈ 3
  α = 1.5 → 各要素は最大 1/(1.5-1) = 2 回コピー → 償却コスト ≈ 5

  教訓: 償却解析の結果はデータ構造の実装戦略に依存する。
        拡張係数を変えると、償却コストも変わる。
```

---

## 13. 演習問題

### 基礎レベル

**問題 B1: 漸化式の立式**

以下の各コードについて、漸化式を立て、計算量を求めよ。

```python
# (a) 最大値を再帰的に求める
def find_max(arr: list[int], n: int) -> int:
    if n == 1:
        return arr[0]
    return max(arr[n - 1], find_max(arr, n - 1))


# (b) 配列の合計を分割統治で求める
def divide_sum(arr: list[int], left: int, right: int) -> int:
    if left == right:
        return arr[left]
    mid = (left + right) // 2
    return divide_sum(arr, left, mid) + divide_sum(arr, mid + 1, right)


# (c) べき乗を再帰的に計算する（非効率版）
def power_naive(x: float, n: int) -> float:
    if n == 0:
        return 1
    return x * power_naive(x, n - 1)
```

**解答**:

```
(a) T(n) = T(n-1) + O(1)
    展開: T(n) = T(n-1) + c = T(n-2) + 2c = ... = T(1) + (n-1)c
    → T(n) = O(n)

(b) T(n) = 2T(n/2) + O(1)
    マスター定理: a=2, b=2, f(n)=O(1), n^(log_2 2) = n
    f(n) = O(n^(1-ε)) → Case 1 → T(n) = Θ(n)

(c) T(n) = T(n-1) + O(1)
    → T(n) = O(n)
    注: 繰り返し二乗法を使えば O(log n) に改善可能
```

---

**問題 B2: マスター定理の適用**

以下の各漸化式にマスター定理を適用し、計算量を求めよ。

```
(a) T(n) = 9T(n/3) + n
(b) T(n) = T(2n/3) + 1
(c) T(n) = 3T(n/4) + n log n
(d) T(n) = 2T(n/4) + sqrt(n)
```

**解答**:

```
(a) a=9, b=3, f(n)=n, n^(log_3 9) = n^2
    f(n) = n = O(n^(2-ε)) で ε=1 → Case 1
    T(n) = Θ(n^2)

(b) a=1, b=3/2, f(n)=1, n^(log_{3/2} 1) = n^0 = 1
    f(n) = 1 = Θ(n^0) → Case 2
    T(n) = Θ(log n)

(c) a=3, b=4, f(n)=n log n, n^(log_4 3) ≈ n^0.793
    f(n) = n log n = Ω(n^(0.793+ε)) → Case 3 の候補
    正則条件: 3 * (n/4) * log(n/4) <= c * n * log n
              (3/4) * n * (log n - log 4) <= c * n * log n
              → c = 3/4 < 1 で十分大きな n に対し成立
    T(n) = Θ(n log n)

(d) a=2, b=4, f(n)=√n, n^(log_4 2) = n^(1/2) = √n
    f(n) = √n = Θ(n^(1/2)) → Case 2
    T(n) = Θ(√n * log n)
```

---

**問題 B3: 最悪ケースと最良ケースの特定**

以下のアルゴリズムについて、最悪ケースと最良ケースの入力例を示し、それぞれの計算量を求めよ。

```python
def linear_search_first_even(arr: list[int]) -> int:
    """配列から最初の偶数を見つけて返す。見つからなければ -1。"""
    for i, val in enumerate(arr):
        if val % 2 == 0:
            return val
    return -1
```

**解答**:

```
最良ケース: arr = [2, 1, 3, 5, ...] （先頭が偶数）
  → 1 回の比較で終了。B(n) = O(1)

最悪ケース: arr = [1, 3, 5, 7, ...] （全て奇数）
  → n 回の比較が必要。W(n) = O(n)

平均ケース: 各要素が偶数である確率 p = 1/2 と仮定すると、
  最初の偶数が位置 k にある確率 = (1/2)^k * (1/2) = (1/2)^(k+1)
  期待比較回数 = Σ_{k=0}^{n-1} (k+1) * (1/2)^(k+1) ≈ 2 - (n+2)/2^n
  → A(n) = O(1) （期待値は定数に収束する）
```

---

### 応用レベル

**問題 A1: 再帰木法による解析**

T(n) = T(n/4) + T(3n/4) + cn の計算量を再帰木法で求めよ。

**ヒント**: 最短パスと最長パスの深さを求め、各レベルのコストを分析せよ。

**解答**:

```
再帰木の構造:
                        cn
                      /     \
                 cn/4       3cn/4           → 合計: cn
                /    \      /     \
           cn/16  3cn/16 3cn/16  9cn/16     → 合計: cn
              :                     :

  最短パス: n → n/4 → n/16 → ... → 1  深さ = log_4 n
  最長パス: n → 3n/4 → 9n/16 → ... → 1  深さ = log_{4/3} n

  各レベルのコスト合計:
  レベル 0: cn
  レベル 1: c(n/4) + c(3n/4) = cn
  レベル 2: cn（各ノードのサイズの合計が n に等しいため）

  レベル数の上界: log_{4/3} n

  → T(n) = Θ(n log n)

  この結果は直感的にも納得できる: 各レベルで合計コスト O(n) × log n レベル。
  不均等分割であっても、「合計サイズが各レベルで n を超えない」という
  性質が保たれるため、マージソートと同じ計算量になる。
```

---

**問題 A2: 償却解析の実践**

以下の「二進カウンタ」に対して、n 回の INCREMENT 操作の償却計算量を3つの手法で求めよ。

```python
class BinaryCounter:
    """k ビットの二進カウンタ。"""

    def __init__(self, k: int) -> None:
        self.bits = [0] * k
        self.k = k

    def increment(self) -> int:
        """カウンタを 1 増やす。ビット反転回数を返す。"""
        flips = 0
        i = 0
        while i < self.k and self.bits[i] == 1:
            self.bits[i] = 0  # 桁上がり
            flips += 1
            i += 1
        if i < self.k:
            self.bits[i] = 1
            flips += 1
        return flips
```

**解答**:

```
■ 集約法:
  n 回の INCREMENT で各ビットが何回反転するかを数える。
  - ビット 0: 毎回反転 → n 回
  - ビット 1: 2 回に 1 回反転 → n/2 回
  - ビット 2: 4 回に 1 回反転 → n/4 回
  - ビット i: 2^i 回に 1 回反転 → n/2^i 回

  合計反転回数 = Σ_{i=0}^{k-1} n/2^i < n × Σ_{i=0}^{∞} 1/2^i = 2n

  → n 回の操作で合計 O(n) → 一操作あたり O(1)

■ 配賦法:
  各 INCREMENT に償却コスト 2 を請求する。
  - ビットを 1 にセット: コスト 1 + 貯金 1
  - ビットを 0 にリセット: 貯金から 1 を消費

  すべてのリセットは過去のセットで「前払い」されているため、
  貯金は常に非負。→ 償却コスト O(1)

■ ポテンシャル法:
  Φ(D) = カウンタ中の 1 の個数

  INCREMENT で t 個のビットをリセット、1 個をセットする場合:
  実コスト = t + 1
  Φ変化 = (1 ビットセット - t ビットリセット) = 1 - t
  償却コスト = (t + 1) + (1 - t) = 2

  → 一操作あたりの償却コスト O(1)
```

---

**問題 A3: 置換法による証明**

T(n) = T(n/2) + T(n/4) + n が T(n) = O(n) であることを置換法で証明せよ。

**解答**:

```
予想: T(n) <= cn （c は適切な定数）

帰納法の仮定: すべての k < n に対し T(k) <= ck

帰納ステップ:
  T(n) = T(n/2) + T(n/4) + n
       <= c(n/2) + c(n/4) + n     （帰納法の仮定）
       = cn/2 + cn/4 + n
       = (3/4)cn + n
       = cn - cn/4 + n
       = cn - (c/4 - 1)n

  c/4 - 1 >= 0 すなわち c >= 4 のとき:
       <= cn

  よって c >= 4 とすれば T(n) <= cn = O(n)。 ■

  ベースケース: T(1) = d → c >= d で成立。
  c = max(4, d) とすれば全体で成立する。
```

---

### 発展レベル

**問題 D1: Akra-Bazzi 定理の適用**

Akra-Bazzi 定理を用いて T(n) = T(n/3) + T(2n/3) + n を解け。

**ヒント**: Akra-Bazzi 定理は T(n) = Σ a_i T(n/b_i) + g(n) の形で、
Σ a_i / b_i^p = 1 を満たす p を求め、T(n) = Θ(n^p (1 + ∫_1^n g(u)/u^{p+1} du)) とする。

**解答**:

```
T(n) = T(n/3) + T(2n/3) + n

Akra-Bazzi 条件: (1/3)^p + (2/3)^p = 1 を満たす p を求める。

  p = 1 を試す: 1/3 + 2/3 = 1 ✓

  g(n) = n なので:
  T(n) = Θ(n^1 × (1 + ∫_1^n u / u^2 du))
       = Θ(n × (1 + ∫_1^n 1/u du))
       = Θ(n × (1 + ln n))
       = Θ(n log n)

  → 再帰木法で得た結果と一致する。
```

---

**問題 D2: 拡張マスター定理**

T(n) = 4T(n/2) + n^2 log n の計算量を求めよ（標準のマスター定理は適用不可）。

**ヒント**: 拡張マスター定理: f(n) = Θ(n^{log_b a} × log^k n) のとき、T(n) = Θ(n^{log_b a} × log^{k+1} n)。

**解答**:

```
a = 4, b = 2, n^(log_2 4) = n^2
f(n) = n^2 log n = Θ(n^2 × log^1 n)

これは f(n) = Θ(n^(log_b a) × log^k n) で k = 1 のケース。

拡張マスター定理より:
T(n) = Θ(n^2 × log^(1+1) n) = Θ(n^2 log^2 n)

検証（再帰木法）:
  レベル l のコスト: 4^l × (n/2^l)^2 × log(n/2^l)
                    = n^2 × (log n - l)
  合計: Σ_{l=0}^{log n} n^2 × (log n - l)
       = n^2 × Σ_{j=0}^{log n} j
       = n^2 × (log n)(log n + 1)/2
       = Θ(n^2 log^2 n) ✓
```

---

**問題 D3: 償却計算量の下界証明**

動的配列において、拡張係数 alpha > 1 で容量を alpha 倍にする戦略を考える。
n 回の append 操作の合計コストが Omega(n) であることを証明し、
一操作あたりの償却コストが Theta(1/(alpha - 1)) に比例することを示せ。

**解答**:

```
n 回の append で、拡張は以下のタイミングで発生する:
  容量 1 → α → α² → ... → α^k （α^k >= n となる k まで）

  拡張回数: k = ⌈log_α n⌉

  各拡張時のコピーコスト: 拡張直前の要素数
  合計コピーコスト = 1 + α + α² + ... + α^{k-1}
                   = (α^k - 1) / (α - 1)
                   ≈ n / (α - 1)

  書き込みコスト: n

  合計コスト = n + n/(α-1) = n × (1 + 1/(α-1)) = n × α/(α-1)

  一操作あたりの償却コスト = α/(α-1)

  α = 2 → 2/1 = 2   （1 操作あたり 2）
  α = 1.5 → 1.5/0.5 = 3   （1 操作あたり 3）
  α → 1 → ∞   （拡張係数を 1 に近づけると発散）

  → 拡張係数 α を小さくするとメモリ効率は良くなるが、
    コピーの頻度が増えて時間コストが増大する。
    これが空間と時間のトレードオフの典型例である。
```

---

## 14. FAQ

### Q1: マスター定理が使えないケースはどうすればよいか？

**A:** 主に3つの代替手法がある:

1. **再帰木法**: 最も汎用的。T(n) = T(n/3) + T(2n/3) + n のように分割が不均等な場合にも対応できる。各レベルのコストを計算して合計する。

2. **Akra-Bazzi 定理**: T(n) = Σ a_i T(n/b_i) + g(n) の一般的な形に対応する、マスター定理の上位互換。ただし、g(n) が多項式的に有界である必要がある。

3. **置換法**: 解を予想して帰納法で証明する。再帰木法で得た予想を厳密化するのに使う。予想が正しいことの「最終確認」としての役割が大きい。

実用的な方針: まず再帰木法で解の見当をつけ、次にマスター定理（適用可能なら）または Akra-Bazzi 定理で確認し、必要に応じて置換法で厳密に証明する。

### Q2: log の底は計算量に影響するか？

**A:** 漸近記法（O, Θ, Ω）の中では影響しない。これは以下の数学的性質に基づく:

```
log_a(n) = log_b(n) / log_b(a)
```

log_b(a) は定数であるため、底の違いは定数倍の差にしかならない。漸近記法は定数倍を無視するので、O(log_2 n) = O(log_10 n) = O(ln n) = O(log n) である。

**ただし注意**: 漸近記法の「外」では底は重要である。例えば:
- 二分探索は正確には log_2 n 回の比較を行う
- 三分探索は log_3 n 回の比較を行う
- 実際の比較回数を議論するときは底を明示すべき

また、log の指数が異なる場合は影響する: O(log n) ≠ O(log^2 n)。

### Q3: 再帰を反復に変換すると計算量は変わるか？

**A:** 時間計算量は通常変わらない。ただし以下の点で差異が生じる:

| 観点 | 再帰版 | 反復版 |
|------|--------|--------|
| 時間計算量 | 同じ | 同じ |
| 空間計算量 | O(再帰の深さ) のスタック | 明示的スタックで制御可能 |
| 定数係数 | 関数呼び出しオーバーヘッドあり | オーバーヘッドなし |
| 末尾再帰最適化 | 言語依存（Python は非対応） | 不要 |
| スタックオーバーフロー | 深い再帰で発生しうる | 発生しない |

特に Python では再帰の深さが デフォルトで 1000 に制限されているため、深い再帰は反復に変換するか `sys.setrecursionlimit()` を明示的に設定する必要がある。

### Q4: 償却計算量は「平均計算量」と同じか？

**A:** 異なる。この混同は非常にありがちだが、明確に区別すべきである。

| 観点 | 償却計算量 | 平均計算量 |
|------|-----------|-----------|
| 対象 | 操作列全体のコスト | 入力の確率分布に基づく期待値 |
| 確率 | 一切使わない（決定論的） | 入力の確率分布を仮定 |
| 保証 | 最悪ケースでの保証 | 確率的な保証のみ |
| 例 | 動的配列: n 回の append で O(n) を保証 | クイックソート: ランダム入力で O(n log n) を期待 |

償却計算量は「n 回の操作の合計が確実に O(n) 以下」という保証であり、特定の操作が遅くても、操作列全体では帳尻が合うことを示す。これに対し、平均計算量は「典型的な入力に対する期待実行時間」であり、最悪ケースでは保証されない。

### Q5: なぜ計算量解析が重要なのか？ プロファイラで測定すれば十分ではないか？

**A:** 計算量解析とプロファイリングは相補的なツールであり、どちらか一方では不十分である。

計算量解析の利点:
- **実装前に性能を予測できる**: コードを書く前に、設計段階でボトルネックを特定できる
- **スケーラビリティの予測**: 「入力が 10 倍になったとき、どうなるか？」に答えられる
- **ハードウェア非依存**: 今日の高速なマシンでは気にならない差が、10 倍の入力では致命的になり得る

プロファイリングの利点:
- **定数係数やキャッシュ効率を反映**: 漸近記法では見えない実際の性能を測定できる
- **ボトルネックの特定**: コードのどの部分が実際に遅いかを正確に知れる

両者を組み合わせるのが理想的: まず計算量解析で O(n log n) vs O(n^2) 等の「大枠」を決め、次にプロファイリングで定数係数や実装の最適化を行う。

### Q6: 空間計算量は時間計算量とどう関係するか？

**A:** 一般に、空間計算量は時間計算量以下である。なぜなら、メモリの各セルに書き込むのに少なくとも O(1) 時間がかかるためである。

```
S(n) <= T(n) が常に成り立つ

ただし、逆は成り立たない:
  例: 二分探索 — T(n) = O(log n), S(n) = O(1)
  例: マージソート — T(n) = O(n log n), S(n) = O(n)

時間と空間のトレードオフの典型例:
  素朴なフィボナッチ: T(n) = O(φ^n), S(n) = O(n) （再帰スタック）
  メモ化フィボナッチ:  T(n) = O(n), S(n) = O(n) （キャッシュ）
  反復フィボナッチ:   T(n) = O(n), S(n) = O(1) （変数2つ）
```

---

## 15. まとめ

### 15.1 知識の全体像

```
計算量解析の全体マップ:

┌─────────────────────────────────────────────────┐
│                  計算量解析                       │
├─────────────┬──────────────┬─────────────────────┤
│  ケース分析  │  償却解析     │  再帰の解析          │
│             │              │                     │
│ ・最悪ケース │ ・集約法      │ ・漸化式の立式       │
│ ・平均ケース │ ・配賦法      │ ・マスター定理       │
│ ・最良ケース │ ・ポテンシャル │ ・再帰木法          │
│             │  法          │ ・置換法            │
│             │              │ ・Akra-Bazzi        │
└─────────────┴──────────────┴─────────────────────┘
         ↓                           ↓
  ┌──────────────────┐    ┌───────────────────────┐
  │ アルゴリズム選択   │    │ データ構造の評価       │
  │ の判断基準        │    │ の判断基準            │
  └──────────────────┘    └───────────────────────┘
```

### 15.2 核心ポイント

| 項目 | ポイント |
|------|---------|
| 漸近記法 | O（上界）、Omega（下界）、Theta（タイトな界）の3種を使い分ける |
| 最悪ケース | 応答時間保証が必要な場面で重視。SLA やリアルタイムシステムに不可欠 |
| 平均ケース | 入力の確率分布を仮定した期待値。実用上の性能評価に有用 |
| 償却解析 | 操作列全体のコストを評価。「たまに高い操作」を正しく扱う |
| 漸化式 | 再帰構造からコストの関係式を立てる。解析の出発点 |
| マスター定理 | T(n) = aT(n/b) + f(n) 形の定型解法。3ケースで分類 |
| 再帰木法 | 各レベルのコストを視覚化して合計する。最も直感的な手法 |
| 置換法 | 予想を立てて帰納法で証明する。厳密性を要求される場面で使用 |
| Akra-Bazzi | マスター定理の一般化。不均等分割にも対応 |
| 実用との接続 | 漸近解析は「大枠」、プロファイリングは「詳細」。両方必要 |

### 15.3 計算量解析を行う際のチェックリスト

1. **漸化式を正しく立てたか**: ベースケース、再帰呼び出し数、各呼び出しのサイズ、再帰外のコストを確認
2. **適用する定理の前提条件を満たしているか**: マスター定理の 3 ケース、正則条件
3. **ケースの区別をしているか**: 最悪・平均・最良のどれを議論しているかを明示
4. **償却解析が必要な場面を見逃していないか**: 個別操作の最悪ケースが「悲観的すぎ」ないか確認
5. **定数係数が実用上重要でないか**: 小さな n での性能が問題になる場面では定数係数も考慮

---

## 次に読むべきガイド

- [時間空間トレードオフ — メモ化とブルームフィルタ](./02-space-time-tradeoff.md)
- [ソート — Quick/Merge/Heap の計算量比較](../02-algorithms/00-sorting.md)

---

## 16. 参考文献

1. Cormen, T.H., Leiserson, C.E., Rivest, R.L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. — 第3章「Growth of Functions」、第4章「Divide-and-Conquer」、第16章「Amortized Analysis」。漸近記法の厳密な定義、マスター定理の証明、償却解析の3手法が詳述されている。

2. Akra, M. & Bazzi, L. (1998). "On the solution of linear recurrence equations." *Computational Optimization and Applications*, 10(2), 195-210. — マスター定理では扱えない不均等分割の漸化式を一般的に解く定理を提示した原論文。

3. Levitin, A. (2012). *Introduction to the Design and Analysis of Algorithms* (3rd ed.). Pearson. — 再帰の解析手法を豊富な例題とともに解説。初学者にとって最も取り組みやすい教科書のひとつ。

4. Sedgewick, R. & Flajolet, P. (2013). *An Introduction to the Analysis of Algorithms* (2nd ed.). Addison-Wesley. — 平均ケース解析の数学的手法を体系的に扱った上級教科書。クイックソートの平均ケース導出が特に詳しい。

5. Tarjan, R.E. (1985). "Amortized Computational Complexity." *SIAM Journal on Algebraic and Discrete Methods*, 6(2), 306-318. — 償却解析の概念を形式化した先駆的論文。ポテンシャル法の原型が示されている。

6. Knuth, D.E. (1997). *The Art of Computer Programming, Volume 1: Fundamental Algorithms* (3rd ed.). Addison-Wesley. — 計算量解析の数学的基礎を最も厳密に扱った古典。漸近記法の起源と精密な定義が記されている。

7. MIT OpenCourseWare. "6.006 Introduction to Algorithms" and "6.046J Design and Analysis of Algorithms." — MIT の計算量解析に関する講義資料。再帰の解析、マスター定理、償却解析の講義動画とノートが無償公開されている。
