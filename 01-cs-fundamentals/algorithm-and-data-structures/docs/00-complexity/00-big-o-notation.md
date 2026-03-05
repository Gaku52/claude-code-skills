# Big-O 記法と計算量の基礎

> アルゴリズムの効率を数学的に表現する O 記法・Omega 記法・Theta 記法、および空間計算量と償却計算量を体系的に学ぶ。帰納法による計算量の証明方法やよくある落とし穴まで網羅する。

---

## この章で学ぶこと

1. **O / Omega / Theta 記法** の数学的定義と使い分け
2. **小文字の漸近記法** (o 記法・ω 記法) の意味と応用
3. **帰納法による計算量の証明** 技法
4. **主要な計算量クラス** の成長速度と具体例
5. **空間計算量** の評価方法と再帰呼び出しのスタック消費
6. **償却計算量** の考え方と動的配列への応用
7. **アンチパターンとエッジケース** の把握
8. **3段階の演習問題** による定着

### 前提知識

- 基本的なプログラミング（Python の読み書きができる程度）
- 高校数学レベルの関数・不等式・極限の概念
- ループや再帰の基本的な理解

### 本ガイドを読み終えると

- 任意のアルゴリズムの計算量を O / Ω / Θ で正確に記述できる
- 帰着法・帰納法を用いて計算量を厳密に証明できる
- 面接やコードレビューで計算量に関する議論を自信を持って行える

---

## 1. なぜ計算量解析が必要なのか

アルゴリズムの「速さ」を議論するとき、実行時間を秒単位で測定するだけでは不十分である。なぜなら：

1. **ハードウェア依存性**: 同じアルゴリズムでも、実行するマシンによって時間は大きく異なる
2. **入力依存性**: 同じサイズの入力でも、データの並びによって処理時間が変わる
3. **スケーラビリティの不可視性**: n=100 では高速でも n=1,000,000 で破綻するアルゴリズムを見抜けない

計算量解析はこれらの問題を解決する。ハードウェアに依存しない数学的な枠組みで、入力サイズ n が大きくなったときのアルゴリズムの振る舞いを記述する。

```
具体例: n 個の要素からペアを全列挙する問題

方法A: 二重ループ     → 操作回数 ≈ n²/2
方法B: 工夫した方法   → 操作回数 ≈ n log n

n=100 のとき:   方法A ≈ 5,000    方法B ≈ 664     (差: 約7.5倍)
n=10,000のとき:  方法A ≈ 50,000,000  方法B ≈ 132,877  (差: 約376倍)
n=1,000,000のとき: 方法A ≈ 5×10¹¹   方法B ≈ 19,931,568 (差: 約25,000倍)

→ n が大きくなるほど差は劇的に拡大する
```

この「n が大きくなったときの振る舞い」を記述するのが **漸近記法（Asymptotic Notation）** である。

---

## 2. 漸近記法の数学的定義

### 2.1 Big-O 記法（上界: Upper Bound）

**定義:**

```
f(n) = O(g(n))
⟺ ∃ c > 0, ∃ n₀ > 0 such that ∀ n ≥ n₀: f(n) ≤ c · g(n)
```

日本語で述べると：「十分大きな n に対して、f(n) は c·g(n) 以下に収まる定数 c と閾値 n₀ が存在する」ことを意味する。

**なぜ「定数 c」が必要なのか:** 漸近記法は定数倍を無視して成長のオーダーだけに着目する。3n² も 100n² も同じ O(n²) として扱うために、定数 c で調整する余地を残している。

**なぜ「n₀」が必要なのか:** 小さい n では f(n) が c·g(n) を超えることがあっても、漸近的な（大規模な）振る舞いだけが重要だからである。n₀ 以降で常に上界が成り立てばよい。

```
  f(n)
  ▲
  │            ╱ c·g(n)
  │           ╱
  │      ×  ╱     ← n₀ より前では f(n) が c·g(n) を超える場合もある
  │     ╱×╱
  │    ╱╱  ← n₀ 以降は f(n) ≤ c·g(n) が常に成り立つ
  │  ╱╱
  │ ╱╱  f(n)
  │╱╱
  ┼──────────────────────► n
       n₀

  凡例: × は f(n) が c·g(n) を超えている点
        n₀ 以降は必ず f(n) ≤ c·g(n)
```

**証明の例: 3n² + 5n + 2 = O(n²)**

c = 4, n₀ = 6 とする。n ≥ 6 のとき：
- 3n² + 5n + 2 ≤ 3n² + 5n² + 2n² = 10n² (∵ n ≥ 1 なら n ≤ n², 1 ≤ n²)
- しかしもっと丁寧に: n ≥ 6 のとき 5n ≤ 5n²/6 · 6/n · n = 5n, ただし 5n ≤ n² (n ≥ 5)
- よって 3n² + 5n + 2 ≤ 3n² + n² + n² = 5n² ≤ 5 · n² (n ≥ 6 のとき 2 ≤ n²)

もっと簡潔に示す方法：n ≥ 1 のとき、5n ≤ 5n², 2 ≤ 2n² なので、
3n² + 5n + 2 ≤ 3n² + 5n² + 2n² = 10n²。よって c = 10, n₀ = 1 でも成立する。

**完全に動作する Python コードで定義を検証する:**

```python
#!/usr/bin/env python3
"""
Big-O 記法の定義を数値的に検証するプログラム。
f(n) = 3n² + 5n + 2 が O(n²) であることを確認する。
"""


def f(n: int) -> int:
    """解析対象の関数 f(n) = 3n² + 5n + 2"""
    return 3 * n * n + 5 * n + 2


def g(n: int) -> int:
    """上界を与える関数 g(n) = n²"""
    return n * n


def verify_big_o(c: float, n0: int, upper: int = 1000) -> bool:
    """
    f(n) ≤ c * g(n) が全ての n ≥ n₀ で成り立つかを検証する。

    なぜ数値的検証が有用か:
    - 証明の前に仮説を立てるための直感を得られる
    - 証明で選んだ c, n₀ が妥当かどうかを確認できる
    - 教育目的で定義を具体的に理解できる

    注意: これは数学的証明の代替にはならない。
    有限の範囲でしか検証できないためである。
    """
    for n in range(n0, upper + 1):
        if f(n) > c * g(n):
            print(f"  反例発見: n={n}, f(n)={f(n)}, c*g(n)={c * g(n)}")
            return False
    return True


def find_minimum_c(n0: int, precision: float = 0.01) -> float:
    """
    与えられた n₀ に対して、f(n) ≤ c * g(n) を満たす最小の c を探す。

    なぜ最小の c を探すのか:
    - Big-O の定義では「ある c が存在する」ことだけが必要だが、
      最小の c を知ると関数の実際の成長率に対する直感が深まる。
    """
    c = precision
    while c <= 1000:
        if verify_big_o(c, n0, upper=10000):
            return c
        c += precision
    return float('inf')


if __name__ == "__main__":
    print("=" * 60)
    print("Big-O 記法の定義検証: f(n) = 3n² + 5n + 2 = O(n²)")
    print("=" * 60)

    # c=10, n₀=1 での検証
    print("\n【検証1】c=10, n₀=1:")
    result = verify_big_o(c=10, n0=1)
    print(f"  結果: {'成立' if result else '不成立'}")

    # c=4, n₀=1 での検証
    print("\n【検証2】c=4, n₀=1:")
    result = verify_big_o(c=4, n0=1)
    print(f"  結果: {'成立' if result else '不成立'}")

    # c=3.5, n₀=1 での検証（ギリギリのケース）
    print("\n【検証3】c=3.5, n₀=1:")
    result = verify_big_o(c=3.5, n0=1)
    print(f"  結果: {'成立' if result else '不成立'}")

    # c=3.5, n₀=10 での検証
    print("\n【検証4】c=3.5, n₀=10:")
    result = verify_big_o(c=3.5, n0=10)
    print(f"  結果: {'成立' if result else '不成立'}")

    # 最小の c を探索
    print("\n【探索】n₀=1 に対する最小の c:")
    min_c = find_minimum_c(n0=1, precision=0.01)
    print(f"  最小 c ≈ {min_c}")

    # 比率 f(n)/g(n) の収束を観察
    print("\n【収束確認】f(n)/g(n) の推移:")
    print(f"  {'n':>8} | {'f(n)':>15} | {'g(n)':>15} | {'f(n)/g(n)':>10}")
    print(f"  {'-'*8}-+-{'-'*15}-+-{'-'*15}-+-{'-'*10}")
    for n in [1, 5, 10, 50, 100, 500, 1000, 10000]:
        fn = f(n)
        gn = g(n)
        ratio = fn / gn if gn > 0 else float('inf')
        print(f"  {n:>8} | {fn:>15,} | {gn:>15,} | {ratio:>10.4f}")

    print("\n→ n が大きくなると f(n)/g(n) は 3.0 に収束する。")
    print("  これは f(n) の主要項が 3n² であることと一致する。")
```

### 2.2 Big-Omega 記法（下界: Lower Bound）

**定義:**

```
f(n) = Ω(g(n))
⟺ ∃ c > 0, ∃ n₀ > 0 such that ∀ n ≥ n₀: f(n) ≥ c · g(n)
```

Big-O が「最悪でもこの程度に収まる」ことを示すのに対し、Big-Omega は「少なくともこの程度はかかる」ことを示す。

**なぜ下界が重要なのか:** あるアルゴリズムが Ω(n log n) であると証明できれば、どんなに工夫しても n log n より速くはならないことが分かる。比較ベースのソートが Ω(n log n) であることは、ソートアルゴリズムの設計において根本的な制約を与える。

**証明の例: 3n² + 5n + 2 = Ω(n²)**

c = 3, n₀ = 1 とする。n ≥ 1 のとき：
- 3n² + 5n + 2 ≥ 3n² (∵ 5n + 2 ≥ 0)
- 3n² = 3 · n²

よって c = 3, n₀ = 1 で定義が満たされる。

### 2.3 Big-Theta 記法（タイトな境界: Tight Bound）

**定義:**

```
f(n) = Θ(g(n))
⟺ f(n) = O(g(n)) かつ f(n) = Ω(g(n))

同値な定義:
⟺ ∃ c₁ > 0, c₂ > 0, n₀ > 0 such that
   ∀ n ≥ n₀: c₁ · g(n) ≤ f(n) ≤ c₂ · g(n)
```

上界と下界が同じオーダーのとき、最も精密な特性付けとなる。

```
  f(n)
  ▲
  │         ╱ c₂·g(n)    ← 上界
  │        ╱
  │      ╱╱ f(n)         ← f(n) は2本の線の間に挟まれる
  │    ╱╱╱
  │  ╱╱╱   c₁·g(n)      ← 下界
  │╱╱╱
  ┼──────────────────────► n
       n₀

  n₀ 以降、f(n) は常に c₁·g(n) と c₂·g(n) の間に存在する。
  これが「タイトな境界」の意味である。
```

**証明の例: 3n² + 5n + 2 = Θ(n²)**

上の 2.1 と 2.2 から：
- O(n²): c₂ = 10, n₀ = 1 で成立
- Ω(n²): c₁ = 3, n₀ = 1 で成立

よって c₁ = 3, c₂ = 10, n₀ = 1 で Θ(n²) が成立する。

### 2.4 小文字の漸近記法（Little-o と Little-omega）

**Little-o 記法（狭義上界）:**

```
f(n) = o(g(n))
⟺ ∀ c > 0, ∃ n₀ > 0 such that ∀ n ≥ n₀: f(n) < c · g(n)

同値条件: lim(n→∞) f(n)/g(n) = 0
```

Big-O との違い: Big-O は「ある c が存在する」（∃c）のに対し、Little-o は「全ての c に対して」（∀c）成り立つ。つまり f(n) は g(n) に比べて漸近的に無視できるほど小さい。

例: n = o(n²) は成り立つが、n² = o(n²) は成り立たない。

**Little-omega 記法（狭義下界）:**

```
f(n) = ω(g(n))
⟺ ∀ c > 0, ∃ n₀ > 0 such that ∀ n ≥ n₀: f(n) > c · g(n)

同値条件: lim(n→∞) f(n)/g(n) = ∞
```

例: n² = ω(n) は成り立つが、n² = ω(n²) は成り立たない。

**漸近記法の関係を数学の不等式で直感的に理解する:**

```
漸近記法と不等式のアナロジー:

  f(n) = O(g(n))    ←→  a ≤ b      (以下)
  f(n) = Ω(g(n))    ←→  a ≥ b      (以上)
  f(n) = Θ(g(n))    ←→  a = b      (等しい、オーダーの意味で)
  f(n) = o(g(n))    ←→  a < b      (未満、真に小さい)
  f(n) = ω(g(n))    ←→  a > b      (超、真に大きい)

  ※ ただしこのアナロジーは直感的理解のためであり、
    厳密には「定数倍の範囲で」という意味である。
```

### 2.5 漸近記法の重要な性質

**推移律（Transitivity）:**
- f(n) = O(g(n)) かつ g(n) = O(h(n)) ⟹ f(n) = O(h(n))
- Ω, Θ, o, ω でも同様に成り立つ

**反射律（Reflexivity）:**
- f(n) = O(f(n))
- f(n) = Ω(f(n))
- f(n) = Θ(f(n))
- ただし f(n) = o(f(n)) は成り立たない（自分自身より真に小さくはない）

**対称律（Symmetry）:**
- f(n) = Θ(g(n)) ⟺ g(n) = Θ(f(n))

**転置対称律（Transpose Symmetry）:**
- f(n) = O(g(n)) ⟺ g(n) = Ω(f(n))
- f(n) = o(g(n)) ⟺ g(n) = ω(f(n))

**和の規則:**
- f(n) = O(h(n)) かつ g(n) = O(h(n)) ⟹ f(n) + g(n) = O(h(n))
- より一般に: O(f(n)) + O(g(n)) = O(max(f(n), g(n)))

**積の規則:**
- O(f(n)) · O(g(n)) = O(f(n) · g(n))

---

## 3. 帰納法による計算量の証明

アルゴリズムの計算量を厳密に証明するには、数学的帰納法が強力な道具となる。特に再帰的アルゴリズムにおいて威力を発揮する。

### 3.1 帰納法の基本構造

```
計算量証明における帰納法の手順:

Step 1: 命題 P(n) を明確に述べる
        例: 「T(n) ≤ c · n log n が全ての n ≥ n₀ で成り立つ」

Step 2: 基底段階（Base Case）
        小さな n で P(n) が成り立つことを直接確認する

Step 3: 帰納段階（Inductive Step）
        P(k) が k < n の全てで成り立つと仮定し（帰納仮定）、
        P(n) が成り立つことを示す

Step 4: 結論
        帰納法により、全ての n ≥ n₀ で P(n) が成り立つ
```

### 3.2 例: マージソートの計算量 O(n log n) の証明

マージソートの漸化式：
```
T(n) = 2T(n/2) + cn    (n > 1)
T(1) = c               (定数)
```

**なぜこの漸化式になるのか:**
- 配列を半分に分割して2つの部分配列をそれぞれソート → 2T(n/2)
- 2つのソート済み配列をマージする処理 → cn（全要素を1回ずつ見る）

**証明: T(n) ≤ cn log₂ n を示す（n は2のべき乗とする）**

**基底段階:** n = 2 のとき
- T(2) = 2T(1) + 2c = 2c + 2c = 4c
- cn log₂ n = 2c · 1 = 2c
- T(2) = 4c > 2c なので、この形では成り立たない

修正: T(n) ≤ cn log₂ n + cn を示す。

**基底段階:** n = 1 のとき
- T(1) = c
- cn log₂ n + cn = 0 + c = c ✓

**帰納段階:** n/2 以下の全てのサイズで成り立つと仮定する。

```
T(n) = 2T(n/2) + cn
     ≤ 2[c(n/2)log₂(n/2) + c(n/2)] + cn    (帰納仮定より)
     = cn[log₂(n/2)] + cn + cn
     = cn[log₂ n - 1] + 2cn
     = cn·log₂ n - cn + 2cn
     = cn·log₂ n + cn ✓
```

よって T(n) = O(n log n) が証明された。

**完全に動作する Python コードで検証する:**

```python
#!/usr/bin/env python3
"""
マージソートの計算量 O(n log n) を帰納法の結果と照合して検証する。
実際のマージソートの操作回数をカウントし、理論値と比較する。
"""

import math


class MergeSortCounter:
    """
    マージソートの比較回数を正確にカウントするクラス。

    なぜクラスを使うのか:
    - 再帰関数内でカウンタを更新するため、ミュータブルな状態が必要
    - グローバル変数を避け、カプセル化するため
    """

    def __init__(self):
        self.comparisons = 0  # 比較回数
        self.assignments = 0  # 代入回数

    def reset(self):
        """カウンタをリセットする"""
        self.comparisons = 0
        self.assignments = 0

    def merge_sort(self, arr: list) -> list:
        """
        マージソートを実行し、比較・代入回数をカウントする。

        返り値: ソート済み配列
        """
        if len(arr) <= 1:
            return arr

        mid = len(arr) // 2
        left = self.merge_sort(arr[:mid])
        right = self.merge_sort(arr[mid:])
        return self._merge(left, right)

    def _merge(self, left: list, right: list) -> list:
        """2つのソート済み配列をマージする"""
        result = []
        i = j = 0

        while i < len(left) and j < len(right):
            self.comparisons += 1
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
            self.assignments += 1

        while i < len(left):
            result.append(left[i])
            i += 1
            self.assignments += 1

        while j < len(right):
            result.append(right[j])
            j += 1
            self.assignments += 1

        return result


def theoretical_upper_bound(n: int) -> float:
    """
    理論上の上界 c * n * log₂(n) を計算する。

    なぜ log₂ を使うのか:
    マージソートは毎回半分に分割するため、再帰の深さは log₂(n)。
    各レベルで O(n) の作業を行うため、合計は n * log₂(n) となる。
    """
    if n <= 1:
        return 0
    return n * math.log2(n)


if __name__ == "__main__":
    import random

    counter = MergeSortCounter()

    print("=" * 70)
    print("マージソートの計算量検証: 比較回数 vs 理論値 n·log₂(n)")
    print("=" * 70)
    print(f"{'n':>8} | {'比較回数':>10} | {'n·log₂n':>12} | {'比率':>8} | {'判定':>6}")
    print(f"{'-'*8}-+-{'-'*10}-+-{'-'*12}-+-{'-'*8}-+-{'-'*6}")

    for exp in range(1, 16):
        n = 2 ** exp
        arr = list(range(n))
        random.seed(42)  # 再現性のため固定シード
        random.shuffle(arr)

        counter.reset()
        sorted_arr = counter.merge_sort(arr[:])

        # ソート結果の正当性確認
        assert sorted_arr == sorted(arr), "ソート結果が不正"

        theory = theoretical_upper_bound(n)
        ratio = counter.comparisons / theory if theory > 0 else 0
        within = "OK" if ratio <= 1.5 else "WARN"

        print(f"{n:>8,} | {counter.comparisons:>10,} | {theory:>12,.1f} | {ratio:>8.4f} | {within:>6}")

    print()
    print("→ 比率が定数に収束することから、比較回数 = Θ(n log n) が確認できる。")
    print("  比率が 1.0 以下にならないのは、マージ処理の定数項があるため。")
```

### 3.3 帰納法の注意点: 基底段階の重要性

帰納法で計算量を証明する際、よくある誤りは基底段階の検証を怠ることである。

```
【誤った証明の例】

主張: T(n) = 2T(⌊n/2⌋) + n に対して T(n) = O(n)

帰納段階:
  T(n) = 2T(⌊n/2⌋) + n
       ≤ 2 · c · ⌊n/2⌋ + n     (帰納仮定)
       ≤ 2 · c · (n/2) + n
       = cn + n
       = (c+1)n                  ← これは cn 以下にならない！

この「証明」は帰納段階で破綻している。
T(n) ≤ cn を示そうとしても cn + n ≤ cn にはならない。
つまり T(n) = O(n) は成り立たない（実際には T(n) = Θ(n log n)）。
```

### 3.4 置換法（Substitution Method）の手順

再帰的な計算量を証明する一般的な手法が「置換法」である。

```
置換法の3ステップ:

1. 推測（Guess）: 答えの形を推測する
   例: T(n) = O(n log n) と推測

2. 代入（Substitute）: 推測を漸化式に代入し、帰納法で証明する
   帰納仮定 T(k) ≤ ck log k (k < n) を使い、T(n) ≤ cn log n を示す

3. 定数の決定: 基底段階を含めて全体を満たす定数 c を決定する
```

**推測のコツ:**
- 再帰木を描いて各レベルのコストを概算する
- 類似の漸化式の既知の解を参考にする
- マスター定理（次章で詳述）で当たりを付ける

### 3.5 例: 再帰的な二分探索の計算量証明

```python
#!/usr/bin/env python3
"""
二分探索の計算量 O(log n) を帰納法で証明し、数値的に検証する。

漸化式: T(n) = T(n/2) + c  (n > 1)
        T(1) = c

主張: T(n) ≤ c · (log₂ n + 1) = O(log n)
"""

import math
import random
import time


def binary_search_recursive(arr: list, target: int,
                            lo: int, hi: int,
                            depth: int = 0) -> tuple:
    """
    再帰的二分探索。探索結果と再帰の深さを返す。

    なぜ深さを返すのか:
    - 計算量 O(log n) の理論値と実際の再帰深度を比較するため
    - 帰納法の証明結果を数値的に確認できる

    返り値: (見つかったインデックス or -1, 再帰深度)
    """
    if lo > hi:
        return (-1, depth)

    mid = (lo + hi) // 2

    if arr[mid] == target:
        return (mid, depth + 1)
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, hi, depth + 1)
    else:
        return binary_search_recursive(arr, target, lo, mid - 1, depth + 1)


if __name__ == "__main__":
    print("=" * 60)
    print("二分探索の計算量検証: 再帰深度 vs log₂(n)")
    print("=" * 60)
    print(f"{'n':>10} | {'最大深度':>8} | {'log₂(n)+1':>10} | {'判定':>6}")
    print(f"{'-'*10}-+-{'-'*8}-+-{'-'*10}-+-{'-'*6}")

    for exp in range(1, 21):
        n = 2 ** exp
        arr = list(range(n))

        # 全要素を探索して最大深度を記録
        max_depth = 0
        for target in [0, n // 4, n // 2, 3 * n // 4, n - 1, n]:
            _, depth = binary_search_recursive(arr, target, 0, n - 1)
            max_depth = max(max_depth, depth)

        theory = math.log2(n) + 1
        ok = "OK" if max_depth <= theory + 1 else "WARN"

        print(f"{n:>10,} | {max_depth:>8} | {theory:>10.1f} | {ok:>6}")

    print()
    print("→ 最大深度は常に log₂(n) + 1 以下に収まっている。")
    print("  これは T(n) = O(log n) と一致する。")
```

### 3.6 帰納法による下界の証明

上界だけでなく、下界の証明にも帰納法は使える。

**例: 比較ベースソートの下界 Ω(n log n) の証明概略**

比較ベースのソートアルゴリズムは、n 個の要素の順列を決定するために少なくとも log₂(n!) 回の比較が必要である。

```
なぜ log₂(n!) なのか:

n 個の要素には n! 通りの順列がある。
各比較は2分岐（≤ か >）なので、k 回の比較で区別できる順列は最大 2^k 通り。
全順列を区別するには:
  2^k ≥ n!
  k ≥ log₂(n!)

スターリングの近似 n! ≈ (n/e)^n より:
  log₂(n!) ≈ n log₂(n/e) = n log₂ n - n log₂ e ≈ n log₂ n - 1.443n

よって k = Ω(n log n)

  決定木の概念図:

                    a₁ ≤ a₂?
                   /         \
              a₂ ≤ a₃?     a₁ ≤ a₃?
              /     \       /     \
          [1,2,3]  ...   ...    ...

  葉ノードの数 ≥ n! であるため、
  木の高さ ≥ log₂(n!) = Ω(n log n)
```

---

## 4. 主要な計算量クラスの詳細

### 4.1 O(1) — 定数時間

入力サイズに関わらず一定の操作回数で完了する。

**典型的な操作:**
- 配列の添字アクセス: `arr[i]`
- ハッシュテーブルの平均的な参照/挿入
- スタックの push/pop
- 変数への代入

```python
#!/usr/bin/env python3
"""
O(1) 操作の例と、O(1) に見えて実は O(1) でない操作の比較。

なぜこの区別が重要なのか:
- 見た目が1行でも計算量が O(1) とは限らない
- 内部で何が行われているかを理解する必要がある
"""


def constant_time_access(arr: list, index: int) -> int:
    """
    配列の添字アクセス — O(1)

    なぜ O(1) なのか:
    配列はメモリ上の連続領域に格納されるため、
    先頭アドレス + index × 要素サイズ で直接アドレスを計算できる。
    この計算は入力サイズに依存しない。
    """
    return arr[index]


def hash_table_lookup(d: dict, key: str) -> object:
    """
    ハッシュテーブルの参照 — 平均 O(1), 最悪 O(n)

    なぜ平均 O(1) なのか:
    ハッシュ関数がキーを均等に分散させるため、
    衝突が少なければ1回のハッシュ計算とメモリアクセスで済む。

    なぜ最悪 O(n) なのか:
    全てのキーが同じハッシュ値に衝突した場合、
    連結リストの線形探索が必要になる。
    """
    return d.get(key)


def looks_constant_but_not(s: str) -> str:
    """
    O(1) に見えるが実は O(n) — 文字列の結合

    なぜ O(n) なのか:
    Python の文字列はイミュータブルなので、
    s + "x" は長さ len(s)+1 の新しい文字列を生成する。
    この生成に O(len(s)) のコピーが必要。
    """
    return s + "x"


if __name__ == "__main__":
    import time

    print("=" * 60)
    print("O(1) 操作 vs 隠れた O(n) 操作の比較")
    print("=" * 60)

    # 配列アクセスは O(1): サイズに関わらず一定時間
    print("\n【配列の添字アクセス（O(1)）】")
    for size in [100, 10_000, 1_000_000, 100_000_000]:
        arr = list(range(size))
        start = time.perf_counter_ns()
        for _ in range(10000):
            _ = arr[size // 2]
        elapsed = (time.perf_counter_ns() - start) / 10000
        print(f"  サイズ {size:>12,}: 1回あたり {elapsed:>8.1f} ns")

    # 文字列結合は O(n): サイズに比例して遅くなる
    print("\n【文字列結合（O(n)に見えないが O(n)）】")
    for size in [100, 1_000, 10_000, 100_000]:
        s = "a" * size
        start = time.perf_counter_ns()
        for _ in range(100):
            _ = s + "x"
        elapsed = (time.perf_counter_ns() - start) / 100
        print(f"  長さ {size:>12,}: 1回あたり {elapsed:>10.1f} ns")
```

### 4.2 O(log n) — 対数時間

問題のサイズが各ステップで半分（または定数分の1）に減少する。

**典型的な操作:**
- 二分探索
- 平衡二分探索木の参照/挿入/削除
- ユークリッドの互除法

**なぜ log n なのか:**
n を何回半分にすると 1 になるか？ → n / 2^k = 1 → k = log₂ n

```
n = 1024 の二分探索:

ステップ 0: 探索範囲 1024 要素  [0 .......... 1023]
ステップ 1: 探索範囲  512 要素  [0 ..... 511]
ステップ 2: 探索範囲  256 要素  [0 .. 255]
ステップ 3: 探索範囲  128 要素  [0 . 127]
ステップ 4: 探索範囲   64 要素
ステップ 5: 探索範囲   32 要素
ステップ 6: 探索範囲   16 要素
ステップ 7: 探索範囲    8 要素
ステップ 8: 探索範囲    4 要素
ステップ 9: 探索範囲    2 要素
ステップ10: 探索範囲    1 要素  → 発見 or 不在

log₂(1024) = 10 ステップ。100万要素でも約20ステップ。
```

### 4.3 O(n) — 線形時間

入力の全要素を定数回ずつ処理する。

**典型的な操作:**
- 配列の線形探索
- 配列の最大値/最小値の計算
- カウンティングソート
- 連結リストの走査

### 4.4 O(n log n) — 線形対数時間

効率的なソートアルゴリズムの典型的な計算量。

**典型的なアルゴリズム:**
- マージソート
- ヒープソート
- クイックソート（平均）
- 高速フーリエ変換 (FFT)

**なぜ多くの問題で n log n が現れるのか:**
- 分割統治法で問題を半分に分け（log n レベル）、各レベルで n の作業をする
- 比較ベースソートの理論的下界が Ω(n log n) である

### 4.5 O(n²) — 二乗時間

全てのペアを調べる必要がある場合に現れる。

**典型的なアルゴリズム:**
- バブルソート、選択ソート、挿入ソート
- 素朴な行列乗算の部分ステップ
- 素朴な最近点対問題

### 4.6 O(2ⁿ) と O(n!) — 指数時間と階乗時間

組合せ爆発が起こる問題に現れる。

**典型的な問題:**
- O(2ⁿ): 部分集合の全列挙、素朴な動的計画法
- O(n!): 全順列の列挙、巡回セールスマン問題（素朴解法）

### 4.7 計算量クラスの成長速度比較表

| 計算量 | 名称 | n=10 | n=20 | n=50 | n=100 | n=1000 | 具体例 |
|--------|------|------|------|------|-------|--------|--------|
| O(1) | 定数 | 1 | 1 | 1 | 1 | 1 | 配列添字アクセス |
| O(log n) | 対数 | 3.3 | 4.3 | 5.6 | 6.6 | 10.0 | 二分探索 |
| O(√n) | 平方根 | 3.2 | 4.5 | 7.1 | 10.0 | 31.6 | 素数判定の試し割り |
| O(n) | 線形 | 10 | 20 | 50 | 100 | 1,000 | 線形探索 |
| O(n log n) | 線形対数 | 33 | 86 | 282 | 664 | 9,966 | マージソート |
| O(n²) | 二乗 | 100 | 400 | 2,500 | 10,000 | 10⁶ | バブルソート |
| O(n³) | 三乗 | 1,000 | 8,000 | 125,000 | 10⁶ | 10⁹ | 素朴な行列乗算 |
| O(2ⁿ) | 指数 | 1,024 | 10⁶ | 10¹⁵ | 10³⁰ | 10³⁰¹ | 部分集合列挙 |
| O(n!) | 階乗 | 3.6×10⁶ | 2.4×10¹⁸ | 3×10⁶⁴ | 9×10¹⁵⁷ | - | 全順列列挙 |

**想定される処理時間の目安（1秒あたり10⁸回の操作を想定）:**

| 計算量 | n = 10⁶ で必要な時間 |
|--------|---------------------|
| O(n) | 約 0.01 秒 |
| O(n log n) | 約 0.2 秒 |
| O(n²) | 約 2.8 時間 |
| O(n³) | 約 31.7 年 |
| O(2ⁿ) | 宇宙の年齢を遥かに超える |

---

## 5. 空間計算量

### 5.1 空間計算量とは何か

空間計算量は、アルゴリズムが必要とするメモリ量を入力サイズ n の関数として表現したものである。

**なぜ空間計算量を考える必要があるのか:**
- メモリは有限のリソースであり、時間と同様に制約となる
- 組み込みシステムやモバイルデバイスではメモリが特に制限される
- キャッシュ効率を考えると、少ないメモリで済むアルゴリズムの方が実際には高速な場合がある
- 分散システムではノード間のデータ転送量にも影響する

### 5.2 全体空間 vs 補助空間

| 方式 | 定義 | 例 |
|------|------|-----|
| **全体空間計算量** | 入力データ自体 + アルゴリズムが追加で使うメモリ | ソートの入力配列 + 作業用配列 |
| **補助空間計算量** | アルゴリズムが追加で使うメモリのみ | 作業用配列のみ（入力は含めない） |

一般的に「空間計算量」と言った場合は**補助空間計算量**を指すことが多い。ただし、文脈によって異なるため、どちらを指しているか明示するのが望ましい。

```
例: マージソートの空間計算量

入力配列: [3, 1, 4, 1, 5, 9, 2, 6]   ← サイズ n（全体空間に含む）

マージ時の作業配列: [_, _, _, _, _, _, _, _]  ← サイズ n（補助空間）

全体空間: O(n) + O(n) = O(n)
補助空間: O(n)

比較: ヒープソートは補助空間 O(1) で動作する（インプレースソート）
```

### 5.3 再帰のスタック消費

再帰呼び出しは関数呼び出しのたびにスタックフレームを消費する。この消費量は見落とされやすいが、空間計算量の重要な要素である。

```
fib(5) の呼び出しスタック（最も深い時点）:

┌─────────────────────────┐
│ fib(1)  ← 現在実行中     │ スタックフレーム5
├─────────────────────────┤
│ fib(2)  ← fib(1)を呼出   │ スタックフレーム4
├─────────────────────────┤
│ fib(3)  ← fib(2)を呼出   │ スタックフレーム3
├─────────────────────────┤
│ fib(4)  ← fib(3)を呼出   │ スタックフレーム2
├─────────────────────────┤
│ fib(5)  ← 最初の呼び出し  │ スタックフレーム1
└─────────────────────────┘

最大スタック深度 = n = 5 → 空間計算量 O(n)

注意: fib の素朴再帰は時間 O(2ⁿ) だが空間は O(n)。
なぜなら、同時にスタック上に存在するフレームは最大 n 個だから。
左の子の計算が完了してスタックが巻き戻った後に、右の子の計算が始まる。
```

**完全に動作する Python コードでスタック深度を可視化する:**

```python
#!/usr/bin/env python3
"""
再帰呼び出しのスタック深度を計測・可視化するプログラム。
様々な再帰アルゴリズムのスタック消費を比較する。
"""

import sys


class StackDepthTracker:
    """
    再帰呼び出しのスタック深度を追跡するクラス。

    なぜ追跡が必要か:
    - Python のデフォルトの再帰制限は 1000
    - 再帰深度を意識せずに書くと RecursionError が発生する
    - 空間計算量の分析に再帰深度の情報が不可欠
    """

    def __init__(self):
        self.max_depth = 0
        self.current_depth = 0

    def enter(self):
        """再帰呼び出し開始時に呼ぶ"""
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)

    def leave(self):
        """再帰呼び出し終了時に呼ぶ"""
        self.current_depth -= 1

    def reset(self):
        """カウンタをリセット"""
        self.max_depth = 0
        self.current_depth = 0


def fib_naive(n: int, tracker: StackDepthTracker) -> int:
    """
    フィボナッチ数の素朴再帰。
    時間: O(2ⁿ), 空間: O(n)（スタック深度）
    """
    tracker.enter()
    try:
        if n <= 1:
            return n
        result = fib_naive(n - 1, tracker) + fib_naive(n - 2, tracker)
        return result
    finally:
        tracker.leave()


def factorial_recursive(n: int, tracker: StackDepthTracker) -> int:
    """
    階乗の再帰計算。
    時間: O(n), 空間: O(n)（スタック深度）
    """
    tracker.enter()
    try:
        if n <= 1:
            return 1
        return n * factorial_recursive(n - 1, tracker)
    finally:
        tracker.leave()


def binary_search_rec(arr: list, target: int,
                      lo: int, hi: int,
                      tracker: StackDepthTracker) -> int:
    """
    二分探索の再帰版。
    時間: O(log n), 空間: O(log n)（スタック深度）
    """
    tracker.enter()
    try:
        if lo > hi:
            return -1
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            return binary_search_rec(arr, target, mid + 1, hi, tracker)
        else:
            return binary_search_rec(arr, target, lo, mid - 1, tracker)
    finally:
        tracker.leave()


if __name__ == "__main__":
    tracker = StackDepthTracker()

    print("=" * 60)
    print("再帰アルゴリズムのスタック深度比較")
    print("=" * 60)

    # フィボナッチ（素朴再帰）— 空間 O(n)
    print("\n【フィボナッチ（素朴再帰）】空間 O(n)")
    for n in [5, 10, 15, 20]:
        tracker.reset()
        fib_naive(n, tracker)
        print(f"  fib({n:>2}): 最大スタック深度 = {tracker.max_depth}")

    # 階乗（再帰）— 空間 O(n)
    print("\n【階乗（再帰）】空間 O(n)")
    for n in [5, 10, 50, 100]:
        tracker.reset()
        factorial_recursive(n, tracker)
        print(f"  {n:>3}!: 最大スタック深度 = {tracker.max_depth}")

    # 二分探索（再帰）— 空間 O(log n)
    print("\n【二分探索（再帰）】空間 O(log n)")
    for exp in [4, 8, 12, 16, 20]:
        n = 2 ** exp
        arr = list(range(n))
        tracker.reset()
        binary_search_rec(arr, n - 1, 0, n - 1, tracker)
        print(f"  n={n:>8,}: 最大スタック深度 = {tracker.max_depth:>3}"
              f"  (log₂ n = {exp})")

    print()
    print("→ 二分探索の再帰版は空間 O(log n)。")
    print("  反復版（while ループ）に書き換えると空間 O(1) になる。")
    print("  空間効率が重要な場合は反復版を選ぶべきである。")
```

### 5.4 末尾再帰と空間最適化

```
末尾再帰（Tail Recursion）:
再帰呼び出しが関数の最後の操作である場合、
コンパイラ/インタプリタがスタックフレームを再利用できる可能性がある。

通常の再帰（階乗）:
  def factorial(n):
      if n <= 1: return 1
      return n * factorial(n-1)  ← 乗算が再帰の後にある → 末尾再帰でない

末尾再帰版:
  def factorial_tail(n, acc=1):
      if n <= 1: return acc
      return factorial_tail(n-1, n*acc)  ← 再帰呼び出しが最後 → 末尾再帰

注意: Python は末尾再帰最適化を行わない（設計上の判断）。
Scheme, Haskell, Scala などの言語では末尾再帰最適化が保証されている。
Python で深い再帰が必要な場合は、反復（ループ）に書き換えるのが一般的。
```

### 5.5 時間と空間のトレードオフ

多くの場合、時間計算量と空間計算量はトレードオフの関係にある。

| アプローチ | 時間 | 空間 | 例 |
|-----------|------|------|-----|
| 計算を繰り返す | 長い | 小さい | 毎回フィボナッチを再計算 |
| 結果をキャッシュ | 短い | 大きい | メモ化でフィボナッチの結果を保存 |
| ルックアップテーブル | 最短 | 最大 | 事前に全結果を計算して保存 |

```python
#!/usr/bin/env python3
"""
フィボナッチ数の計算: 時間と空間のトレードオフを実証する。

3つの実装を比較し、トレードオフを数値的に確認する。
"""

import time
from functools import lru_cache


def fib_naive(n: int) -> int:
    """
    素朴再帰: 時間 O(2ⁿ), 空間 O(n)

    なぜ遅いのか:
    同じ部分問題を何度も計算する。fib(30) を求めるのに
    fib(1) は 832,040 回も呼ばれる。
    """
    if n <= 1:
        return n
    return fib_naive(n - 1) + fib_naive(n - 2)


def fib_memoized(n: int, memo: dict = None) -> int:
    """
    メモ化再帰: 時間 O(n), 空間 O(n)

    なぜ速いのか:
    各部分問題を一度だけ計算し、結果を辞書に保存する。
    2回目以降は辞書参照（O(1)）で済む。
    """
    if memo is None:
        memo = {}
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib_memoized(n - 1, memo) + fib_memoized(n - 2, memo)
    return memo[n]


def fib_iterative(n: int) -> int:
    """
    反復（ボトムアップ）: 時間 O(n), 空間 O(1)

    なぜ空間 O(1) なのか:
    直前の2つの値だけを保持すれば十分。
    配列全体を保存する必要がない。
    """
    if n <= 1:
        return n
    prev2, prev1 = 0, 1
    for _ in range(2, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current
    return prev1


if __name__ == "__main__":
    print("=" * 65)
    print("フィボナッチ数の計算: 時間と空間のトレードオフ")
    print("=" * 65)

    # 正当性確認
    for n in range(20):
        assert fib_iterative(n) == fib_memoized(n)

    # 性能比較（小さい n で素朴版も含む）
    print(f"\n{'方法':>14} | {'n':>6} | {'結果':>15} | {'所要時間':>12}")
    print(f"{'-'*14}-+-{'-'*6}-+-{'-'*15}-+-{'-'*12}")

    for n in [10, 20, 30, 35]:
        # 素朴再帰
        start = time.perf_counter()
        result = fib_naive(n)
        elapsed_naive = time.perf_counter() - start

        # メモ化再帰
        start = time.perf_counter()
        result_memo = fib_memoized(n)
        elapsed_memo = time.perf_counter() - start

        # 反復
        start = time.perf_counter()
        result_iter = fib_iterative(n)
        elapsed_iter = time.perf_counter() - start

        print(f"{'素朴再帰':>14} | {n:>6} | {result:>15,} | {elapsed_naive:>10.6f} s")
        print(f"{'メモ化':>14} | {n:>6} | {result_memo:>15,} | {elapsed_memo:>10.6f} s")
        print(f"{'反復':>14} | {n:>6} | {result_iter:>15,} | {elapsed_iter:>10.6f} s")
        print(f"{'-'*14}-+-{'-'*6}-+-{'-'*15}-+-{'-'*12}")

    # 大きい n（素朴版は遅すぎるので除外）
    print(f"\n大きい n での比較（素朴再帰は省略）:")
    print(f"{'方法':>14} | {'n':>6} | {'所要時間':>12}")
    print(f"{'-'*14}-+-{'-'*6}-+-{'-'*12}")

    for n in [100, 500, 1000, 5000]:
        start = time.perf_counter()
        fib_memoized(n, {})
        elapsed_memo = time.perf_counter() - start

        start = time.perf_counter()
        fib_iterative(n)
        elapsed_iter = time.perf_counter() - start

        print(f"{'メモ化':>14} | {n:>6} | {elapsed_memo:>10.6f} s")
        print(f"{'反復':>14} | {n:>6} | {elapsed_iter:>10.6f} s")
        print(f"{'-'*14}-+-{'-'*6}-+-{'-'*12}")

    print()
    print("→ メモ化と反復は同じ時間計算量 O(n) だが、")
    print("  反復版は空間 O(1) で済むため、大きい n では有利。")
```

---

## 6. 償却計算量（Amortized Analysis）

### 6.1 償却計算量とは何か

償却計算量は、一連の操作の**合計コスト**を操作回数で割ったものである。個々の操作は高コストになり得るが、一連の操作全体で見ると平均的に低コストであることを示す。

**なぜ「平均計算量」とは違うのか:**
- **平均計算量**: ランダムな入力に対する期待値。確率分布を仮定する
- **償却計算量**: 最悪の操作列に対する1操作あたりのコスト。確率は関係ない

償却計算量は「保証」を与える。どのような操作列であっても、n 回の操作の合計コストが O(n·f(n)) であれば、1操作あたりの償却コストは O(f(n)) である。

### 6.2 動的配列の追加操作（Aggregate Method）

```
動的配列（Python の list.append）の仕組み:

初期容量 = 1 とし、満杯になったら容量を2倍に拡張する。

操作  容量  サイズ  コスト  説明
─────────────────────────────────────────────
 1     1     1      1     通常の追加
 2     2     2      1+1   拡張(1要素コピー) + 追加
 3     4     3      2+1   拡張(2要素コピー) + 追加
 4     4     4      1     通常の追加
 5     8     5      4+1   拡張(4要素コピー) + 追加
 6     8     6      1     通常の追加
 7     8     7      1     通常の追加
 8     8     8      1     通常の追加
 9    16     9      8+1   拡張(8要素コピー) + 追加
─────────────────────────────────────────────

n 回の操作の合計コスト:
  通常の追加:     n 回 × 1 = n
  拡張時のコピー: 1 + 2 + 4 + 8 + ... + 2^⌊log₂n⌋ ≤ 2n

  合計 ≤ n + 2n = 3n = O(n)

償却コスト = O(n) / n = O(1)
```

### 6.3 三つの分析手法

#### 手法1: 集約法（Aggregate Method）

n 回の操作の合計コストを計算し、n で割る。上の動的配列の例がこの手法に該当する。

#### 手法2: 会計法（Accounting Method）

各操作に「信用（credit）」を割り当てる。安い操作で余分に貯金し、高い操作で使う。

```
動的配列の会計法:

各 append 操作に 3 の信用を割り当てる:
  - 1: 自分自身の追加コスト
  - 1: 将来の拡張時に自分がコピーされるコスト
  - 1: 拡張時に既存要素の1つがコピーされるコスト

操作  信用支払  実コスト  信用残高  説明
──────────────────────────────────────────
 1      3        1         2       通常追加、2を貯金
 2      3        2         3       拡張(コスト2)、貯金使用
 3      3        3         3       拡張(コスト3)、貯金使用
 4      3        1         5       通常追加、2を貯金
 5      3        5         3       拡張(コスト5)、貯金使用
 ...
──────────────────────────────────────────

信用残高は常に非負 → 各操作の償却コスト ≤ 3 = O(1)
```

#### 手法3: ポテンシャル法（Potential Method）

データ構造の「ポテンシャルエネルギー」を定義し、各操作による変化を追跡する。

```
動的配列のポテンシャル法:

ポテンシャル関数: Φ(D) = 2 × サイズ - 容量

操作 i の償却コスト:
  ĉᵢ = cᵢ + Φ(Dᵢ) - Φ(Dᵢ₋₁)

拡張なしの場合 (cᵢ = 1):
  ĉᵢ = 1 + (2(s+1) - cap) - (2s - cap)
     = 1 + 2 = 3

拡張ありの場合 (cᵢ = s + 1, 旧容量 = s, 新容量 = 2s):
  ĉᵢ = (s+1) + (2(s+1) - 2s) - (2s - s)
     = (s+1) + 2 - s = 3

どちらの場合も償却コスト = 3 = O(1) ✓
```

### 6.4 Python の動的配列を検証する

```python
#!/usr/bin/env python3
"""
Python の list の内部容量変化を観察し、償却 O(1) を検証する。

sys.getsizeof() を使って list オブジェクトのメモリサイズを追跡する。
"""

import sys


def observe_list_growth(max_elements: int = 100) -> list:
    """
    list に要素を1つずつ追加し、メモリサイズの変化を記録する。

    なぜ sys.getsizeof を使うのか:
    - list オブジェクト自体のサイズ（ポインタ配列の容量）を取得できる
    - 要素の中身のサイズは含まないが、容量変化の観察には十分
    """
    records = []
    lst = []
    prev_size = sys.getsizeof(lst)

    for i in range(max_elements):
        lst.append(i)
        curr_size = sys.getsizeof(lst)
        if curr_size != prev_size:
            records.append({
                'index': i,
                'elements': i + 1,
                'size_bytes': curr_size,
                'growth': curr_size - prev_size,
                'resized': True
            })
        else:
            records.append({
                'index': i,
                'elements': i + 1,
                'size_bytes': curr_size,
                'growth': 0,
                'resized': False
            })
        prev_size = curr_size

    return records


def calculate_amortized_cost(records: list) -> None:
    """
    集約法による償却コストを計算する。

    各リサイズを「高コスト操作」としてカウントし、
    全操作の平均コストを求める。
    """
    total_cost = 0
    resize_count = 0

    for r in records:
        if r['resized']:
            # リサイズ時のコスト = 既存要素のコピー + 追加
            total_cost += r['elements']
            resize_count += 1
        else:
            # 通常の追加
            total_cost += 1

    n = len(records)
    amortized = total_cost / n if n > 0 else 0

    print(f"\n  全 {n} 回の操作:")
    print(f"    合計コスト: {total_cost}")
    print(f"    リサイズ回数: {resize_count}")
    print(f"    償却コスト: {amortized:.2f} = O(1)")


if __name__ == "__main__":
    print("=" * 65)
    print("Python list の動的配列としての振る舞い")
    print("=" * 65)

    records = observe_list_growth(100)

    print("\nリサイズが発生したタイミング:")
    print(f"  {'要素数':>6} | {'サイズ(bytes)':>12} | {'増加量(bytes)':>13}")
    print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*13}")

    for r in records:
        if r['resized']:
            print(f"  {r['elements']:>6} | {r['size_bytes']:>12} | +{r['growth']:>12}")

    calculate_amortized_cost(records)

    # 大規模での検証
    print("\n" + "=" * 65)
    print("大規模データでの償却コスト検証")
    print("=" * 65)

    for n in [1_000, 10_000, 100_000, 1_000_000]:
        lst = []
        resize_count = 0
        prev_size = sys.getsizeof(lst)

        for i in range(n):
            lst.append(i)
            curr_size = sys.getsizeof(lst)
            if curr_size != prev_size:
                resize_count += 1
                prev_size = curr_size

        print(f"  n={n:>10,}: リサイズ回数 = {resize_count:>4}"
              f"  (log₂ n ≈ {n.bit_length():>3})")

    print()
    print("→ リサイズ回数は O(log n) であり、全操作の合計は O(n)。")
    print("  したがって1回あたりの償却コストは O(1)。")
```

### 6.5 償却計算量が使われる他のデータ構造

| データ構造 | 操作 | 最悪ケース | 償却 |
|-----------|------|-----------|------|
| 動的配列 | append | O(n) | O(1) |
| スプレー木 | 検索/挿入/削除 | O(n) | O(log n) |
| フィボナッチヒープ | decrease-key | O(n) | O(1) |
| Union-Find | union/find | O(log n) | O(α(n)) ≈ O(1) |
| ハッシュテーブル | 挿入（リハッシュあり） | O(n) | O(1) |

---

## 7. よくある計算量パターンの認識

### 7.1 パターン認識のフレームワーク

コードを見て計算量を判断する際の指針を体系化する。

```
計算量パターン認識チャート:

[ループがない]──────────────────────── O(1)
     │
[単一ループ n 回]───────────────────── O(n)
     │
[ループ内でサイズが半減]────────────── O(log n)
     │
[二重ループ（独立）]───────────────── O(n²)
     │
[ループ + 内部でソート]────────────── O(n² log n) or O(n log n)
     │
[分割統治 + 線形マージ]────────────── O(n log n)
     │
[全部分集合を列挙]─────────────────── O(2ⁿ)
     │
[全順列を列挙]─────────────────────── O(n!)
```

### 7.2 よくある落とし穴: 隠れた計算量

```python
#!/usr/bin/env python3
"""
コードの見た目と実際の計算量が異なる典型的なケースを示す。

なぜこれが重要か:
- コードレビューで計算量の問題を見抜くには、
  各操作の内部コストを理解する必要がある
- 「1行 = O(1)」という思い込みは危険
"""


def hidden_quadratic_string(n: int) -> str:
    """
    【落とし穴1】文字列の繰り返し結合 — 見た目 O(n)、実質 O(n²)

    なぜ O(n²) なのか:
    Python の文字列はイミュータブル。各 += で新しい文字列が生成される。
    i 回目の結合で長さ i の文字列をコピーするため、
    合計 = 1 + 2 + ... + n = n(n+1)/2 = O(n²)

    正しい方法: リストに追加して最後に join する → O(n)
    """
    result = ""
    for i in range(n):
        result += "a"  # 毎回 O(i) のコピー
    return result


def correct_string_building(n: int) -> str:
    """O(n) での文字列構築"""
    parts = []
    for i in range(n):
        parts.append("a")  # O(1) 償却
    return "".join(parts)  # O(n) で結合


def hidden_quadratic_list(arr: list) -> list:
    """
    【落とし穴2】リスト先頭への挿入 — 見た目 O(n)、実質 O(n²)

    なぜ O(n²) なのか:
    list.insert(0, x) はリスト全体を1つずつ後ろにシフトする必要がある。
    i 回目の挿入で i 個の要素をシフトするため、合計 O(n²)。

    正しい方法: collections.deque を使う → appendleft が O(1)
    """
    result = []
    for x in arr:
        result.insert(0, x)  # 毎回 O(len(result)) のシフト
    return result


def hidden_quadratic_membership(arr: list, targets: list) -> list:
    """
    【落とし穴3】リストでの in 演算 — 見た目 O(n)、実質 O(n²)

    なぜ O(n²) なのか:
    list の `in` 演算は O(n) の線形探索。
    n 個のターゲットそれぞれに対して O(n) なので合計 O(n²)。

    正しい方法: set に変換してから判定 → O(n) で済む
    """
    found = []
    for target in targets:
        if target in arr:  # O(n) の線形探索
            found.append(target)
    return found


def correct_membership(arr: list, targets: list) -> list:
    """set を使った O(n) の所属判定"""
    arr_set = set(arr)  # O(n) で set を構築
    found = []
    for target in targets:
        if target in arr_set:  # O(1) のハッシュ参照
            found.append(target)
    return found


if __name__ == "__main__":
    import time

    print("=" * 65)
    print("隠れた O(n²) パターンの検出と修正")
    print("=" * 65)

    # パターン1: 文字列結合
    print("\n【パターン1: 文字列結合】")
    for n in [1000, 5000, 10000, 50000]:
        start = time.perf_counter()
        hidden_quadratic_string(n)
        t_bad = time.perf_counter() - start

        start = time.perf_counter()
        correct_string_building(n)
        t_good = time.perf_counter() - start

        speedup = t_bad / t_good if t_good > 0 else float('inf')
        print(f"  n={n:>6}: += {t_bad:.4f}s | join {t_good:.6f}s"
              f" | 高速化: {speedup:.0f}x")

    # パターン3: 所属判定
    print("\n【パターン3: 所属判定】")
    for n in [1000, 5000, 10000, 50000]:
        arr = list(range(n))
        targets = list(range(0, n, 2))  # 偶数のみ

        start = time.perf_counter()
        hidden_quadratic_membership(arr, targets)
        t_bad = time.perf_counter() - start

        start = time.perf_counter()
        correct_membership(arr, targets)
        t_good = time.perf_counter() - start

        speedup = t_bad / t_good if t_good > 0 else float('inf')
        print(f"  n={n:>6}: list {t_bad:.4f}s | set {t_good:.6f}s"
              f" | 高速化: {speedup:.0f}x")

    print()
    print("→ n が大きくなるほど高速化の倍率が増加する。")
    print("  これは O(n²) vs O(n) の差が顕在化しているため。")
```

### 7.3 ループ変数の変化パターンと計算量

| ループパターン | コード例 | 計算量 | 理由 |
|--------------|---------|--------|------|
| 線形増加 | `for i in range(n)` | O(n) | i が 1 ずつ増加 |
| 定数ステップ | `for i in range(0, n, k)` | O(n/k) = O(n) | k は定数 |
| 倍々増加 | `while i < n: i *= 2` | O(log n) | i が 2 倍ずつ増加 |
| 平方根減少 | `while n > 1: n = sqrt(n)` | O(log log n) | 二重対数 |
| 独立二重ループ | `for i in range(n): for j in range(n)` | O(n²) | n × n |
| 依存二重ループ | `for i in range(n): for j in range(i)` | O(n²) | Σi = n(n-1)/2 |
| 三重ループ | `for i,j,k in range(n)³` | O(n³) | n × n × n |
| 外ループ×内ログ | `for i: while j<n: j*=2` | O(n log n) | n × log n |

---

## 8. アンチパターン

### アンチパターン1: 定数係数を漸近記法に含める

```python
# BAD: "ループが2回あるから O(2n) だ" → 誤り
def double_pass(arr: list) -> int:
    """
    2回の線形走査を行う関数。

    誤った分析: O(2n)
    正しい分析: O(n)

    なぜ定数係数を無視するのか:
    漸近記法は入力サイズが大きくなったときの「成長率」を表現する。
    2n も 100n も n に対して線形に成長するため、同じ O(n) である。
    定数係数はハードウェアや言語実装に依存する部分であり、
    アルゴリズムの本質的な効率とは別の問題である。
    """
    total = 0
    for x in arr:     # 第1パス: O(n)
        total += x
    for x in arr:     # 第2パス: O(n)
        total += x * 2
    return total
    # O(n) + O(n) = O(n) ← 定数倍は吸収される


# 同様の誤り:
# × O(n/2) → ○ O(n)   （半分のループでも線形は線形）
# × O(3n²) → ○ O(n²)  （定数倍は無視）
# × O(n² + n) → ○ O(n²) （低次項も無視）
```

### アンチパターン2: 最良ケースだけで計算量を述べる

```python
# BAD: "クイックソートは O(n log n)" → 不完全
def quicksort(arr: list) -> list:
    """
    クイックソートの計算量を正確に述べると:

    - 最良ケース: O(n log n)  ← ピボットが毎回中央値
    - 平均ケース: O(n log n)  ← ランダムな入力
    - 最悪ケース: O(n²)       ← ソート済みの入力 + 先頭ピボット

    なぜケースの区別が重要か:
    「O(n log n)」とだけ述べると、最悪ケースの O(n²) が隠蔽される。
    セキュリティの文脈では、攻撃者が意図的に最悪ケースを引き起こす
    ことが可能な場合がある（例: ハッシュテーブルの衝突攻撃）。

    正しい記述例:
    「クイックソートの平均計算量は O(n log n)、最悪計算量は O(n²)。
     ランダム化ピボット選択により最悪ケースの確率を無視できるほど
     小さくできる。」
    """
    if len(arr) <= 1:
        return arr
    pivot = arr[0]  # 先頭をピボットに（最悪ケースの原因）
    left = [x for x in arr[1:] if x <= pivot]
    right = [x for x in arr[1:] if x > pivot]
    return quicksort(left) + [pivot] + quicksort(right)


# GOOD: ランダム化ピボットで最悪ケースを回避
import random

def quicksort_randomized(arr: list) -> list:
    """
    ランダム化クイックソート。
    平均・期待計算量: O(n log n)
    最悪計算量: O(n²)（ただし確率は非常に低い）
    """
    if len(arr) <= 1:
        return arr
    pivot = random.choice(arr)  # ランダムにピボットを選択
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort_randomized(left) + middle + quicksort_randomized(right)
```

### アンチパターン3: 空間計算量を完全に無視する

```python
# BAD: 時間計算量だけ考えて空間計算量を無視する
def get_all_pairs(arr: list) -> list:
    """
    全ペアを生成する。
    時間: O(n²)
    空間: O(n²) ← これを無視すると危険

    なぜ危険か:
    n = 100,000 の場合、ペア数は 10^10。
    各ペアが8バイトとすると約 80GB のメモリが必要。
    これは大半のマシンのメモリを超える。

    対策:
    1. ジェネレータを使い、全ペアを同時にメモリに保持しない
    2. 必要なペアだけを生成するアルゴリズムに変更する
    """
    return [(a, b) for a in arr for b in arr]  # O(n²) の空間


# GOOD: ジェネレータで空間 O(1) に
def get_all_pairs_generator(arr: list):
    """ペアを1つずつ生成 — 時間 O(n²), 空間 O(1)"""
    for a in arr:
        for b in arr:
            yield (a, b)
```

### アンチパターン4: Big-O の等号を数学的等号と混同する

```
× 誤り: "O(n) = O(n²) だから n = n² だ"

Big-O における等号は数学的な等号ではない。
正確には「∈（属する）」の意味で使われている。

f(n) = O(g(n)) は「f(n) ∈ O(g(n))」と読むべきである。
つまり f(n) が集合 O(g(n)) に属するということ。

O(n) ⊂ O(n²) は正しい。n が n² 以下のオーダーだから。
だが「O(n) = O(n²)」と書くのは、右から左が成り立たないため不正確。

正しくは:
  O(n) ⊂ O(n²)    n ∈ O(n²) は正しい
  O(n²) ⊄ O(n)    n² ∈ O(n) は偽
```

---

## 9. エッジケース分析

### エッジケース1: 入力サイズ 0 と 1

```python
#!/usr/bin/env python3
"""
エッジケースでの計算量の振る舞いを検証する。

なぜエッジケースが重要か:
- 漸近記法は「十分大きな n」に対する記述だが、
  実際のプログラムでは n=0 や n=1 も処理する必要がある
- エッジケースでの定数時間の処理を忘れると、
  ゼロ除算やインデックスエラーの原因になる
"""


def binary_search(arr: list, target: int) -> int:
    """
    二分探索: O(log n) だが、n=0 や n=1 の場合はどうなるか？

    n = 0: ループに入らず即座に -1 を返す → O(1)
    n = 1: 1回の比較で完了 → O(1)

    漸近記法的には O(log n) に含まれる（O(1) ⊂ O(log n)）が、
    実装上はこれらのケースを明示的に処理するのが安全。
    """
    if not arr:  # n = 0 の処理
        return -1
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def merge_sort(arr: list) -> list:
    """
    マージソート: O(n log n) だが、n ≤ 1 では再帰しない。

    多くの実装では n が小さい場合に挿入ソートに切り替える。
    なぜなら、小さい n では挿入ソートの方が定数項が小さく高速だから。
    一般的な閾値は n = 16〜64 程度。
    """
    if len(arr) <= 1:
        return arr[:]
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    # マージ処理
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


if __name__ == "__main__":
    # エッジケースのテスト
    print("=" * 50)
    print("エッジケースでの動作検証")
    print("=" * 50)

    # n = 0
    assert binary_search([], 5) == -1
    assert merge_sort([]) == []
    print("n=0: 全テスト合格")

    # n = 1
    assert binary_search([5], 5) == 0
    assert binary_search([5], 3) == -1
    assert merge_sort([42]) == [42]
    print("n=1: 全テスト合格")

    # n = 2
    assert binary_search([1, 3], 3) == 1
    assert merge_sort([3, 1]) == [1, 3]
    print("n=2: 全テスト合格")

    print("\n→ エッジケースでも正しく動作することを確認。")
```

### エッジケース2: 整数オーバーフローと計算量

```
問題: (lo + hi) // 2 でのオーバーフロー

lo = 2,000,000,000  hi = 2,000,000,000 の場合:
  lo + hi = 4,000,000,000  ← 32ビット整数の上限(2^31 - 1 ≈ 2.1×10⁹)を超える

Python では整数に上限がないため問題にならないが、
C, Java, C++ では深刻なバグになる。

修正方法:
  mid = lo + (hi - lo) // 2

なぜこれが安全か:
  hi - lo は常に非負かつ hi 以下。
  lo + (非負の値) は lo + hi 以下。
  したがってオーバーフローしない。

計算量への影響:
  計算量自体は O(log n) で変わらないが、
  オーバーフローにより無限ループに陥る可能性がある。
  「正しい計算量の分析」は「正しい実装」を前提としている。
```

### エッジケース3: ハッシュテーブルの最悪ケース

```
通常: ハッシュテーブルの参照は O(1) と言われる

しかし最悪ケースでは:
  - 全てのキーが同じバケットに衝突 → O(n) の線形探索
  - リハッシュが発生 → O(n) のコピー
  - ハッシュ関数の計算自体に O(k) かかる場合がある
    （k はキーの長さ、文字列の場合）

想定される影響:
  - n 個のキーを挿入: 通常 O(n)、最悪 O(n²)
  - 攻撃者が意図的に衝突を起こすキーを送信 → DoS 攻撃

対策:
  1. ユニバーサルハッシュ関数でランダム性を導入
  2. Python 3.3+ ではハッシュのランダム化がデフォルト有効
  3. 衝突数が閾値を超えたらバランス木に切り替え（Java 8+ の HashMap）
```

---

## 10. 演習問題

### 10.1 基礎レベル

**問題 B1: 計算量の判定**

以下の各関数の時間計算量を O 記法で答えよ。

```python
def func_a(n: int) -> int:
    total = 0
    for i in range(n):
        for j in range(10):
            total += i * j
    return total


def func_b(n: int) -> int:
    total = 0
    i = 1
    while i < n:
        total += i
        i *= 2
    return total


def func_c(arr: list) -> list:
    result = []
    for x in arr:
        if x not in result:
            result.append(x)
    return result
```

<details>
<summary>解答 B1</summary>

- `func_a`: O(n)。内側のループは定数回（10回）なので O(10n) = O(n)。
- `func_b`: O(log n)。i が 1, 2, 4, 8, ... と倍増するため、ループ回数は log₂ n 回。
- `func_c`: O(n²)。`x not in result` は最悪 O(n) の線形探索であり、それが n 回繰り返されるため O(n²)。重複除去には set を使うと O(n) になる。

</details>

**問題 B2: O 記法の定義の適用**

f(n) = 5n³ + 3n² + 7 について、以下を証明せよ：
1. f(n) = O(n³)
2. f(n) = Ω(n³)
3. f(n) = Θ(n³)

<details>
<summary>解答 B2</summary>

1. **O(n³)の証明:** n ≥ 1 のとき、5n³ + 3n² + 7 ≤ 5n³ + 3n³ + 7n³ = 15n³。よって c=15, n₀=1 で f(n) ≤ 15n³ が成立。

2. **Ω(n³)の証明:** 全ての n ≥ 1 に対して、5n³ + 3n² + 7 ≥ 5n³。よって c=5, n₀=1 で f(n) ≥ 5n³ が成立。

3. **Θ(n³)の証明:** 上の 1. と 2. から、c₁=5, c₂=15, n₀=1 で 5n³ ≤ f(n) ≤ 15n³ が成立。よって f(n) = Θ(n³)。

</details>

**問題 B3: 計算量の比較**

以下の関数の組について、f(n) = O(g(n))、f(n) = Ω(g(n))、f(n) = Θ(g(n)) のいずれが成り立つか答えよ。

1. f(n) = n², g(n) = n³
2. f(n) = 2ⁿ, g(n) = 3ⁿ
3. f(n) = log₂ n, g(n) = log₁₀ n
4. f(n) = n log n, g(n) = n√n

<details>
<summary>解答 B3</summary>

1. f(n) = O(g(n))。n² ≤ n³ (n ≥ 1)。ただし Θ ではない（n² = o(n³)）。
2. f(n) = O(g(n))。2ⁿ ≤ 3ⁿ。また 2ⁿ = o(3ⁿ) なので Θ ではない。
3. f(n) = Θ(g(n))。log₂ n = log₁₀ n / log₁₀ 2 = (1/log₁₀ 2) · log₁₀ n。底の変換は定数倍なので同じオーダー。
4. f(n) = O(g(n))。n log n = o(n^1.5) なので Θ ではない（log n = o(√n) であるため）。

</details>

### 10.2 応用レベル

**問題 A1: 再帰の計算量**

以下の再帰関数の時間計算量と空間計算量を求めよ。

```python
def mystery(n: int) -> int:
    if n <= 0:
        return 0
    return mystery(n - 1) + mystery(n - 1)
```

<details>
<summary>解答 A1</summary>

**時間計算量: O(2ⁿ)**

漸化式: T(n) = 2T(n-1) + O(1)
展開: T(n) = 2T(n-1) = 2·2T(n-2) = 4T(n-2) = ... = 2ⁿ·T(0) = O(2ⁿ)

**空間計算量: O(n)**

最大スタック深度は n。左の `mystery(n-1)` が完了してスタックが巻き戻った後に、右の `mystery(n-1)` が実行される。同時にスタック上に存在するフレームは最大 n+1 個。

注意: この関数は `2 * mystery(n-1)` と等価だが、再帰を2回呼ぶため計算量が指数的になる。

</details>

**問題 A2: 償却計算量の分析**

以下の `MultiStack` クラスの `push` と `multipop(k)` 操作の償却計算量を求めよ。

```python
class MultiStack:
    def __init__(self):
        self.stack = []

    def push(self, x):
        """要素を1つ追加 — コスト 1"""
        self.stack.append(x)

    def multipop(self, k):
        """最大 k 個の要素をポップ — コスト min(k, len(stack))"""
        count = min(k, len(self.stack))
        for _ in range(count):
            self.stack.pop()
        return count
```

<details>
<summary>解答 A2</summary>

**会計法による分析:**

各 push 操作に 2 の信用を割り当てる:
- 1: push 自体のコスト
- 1: 将来の pop のための貯金

multipop(k) は最大 k 回 pop するが、各 pop のコストは push 時に前払いされている。

n 回の操作（push と multipop の混合）の合計コスト:
- 各要素は最大1回 push され、最大1回 pop される
- push の合計回数を m とすると、pop の合計回数も最大 m
- 合計コスト ≤ 2m ≤ 2n

**結論:**
- push の償却コスト: O(1)
- multipop(k) の償却コスト: O(1)（k によらない！）

直感的説明: multipop で k 個ポップするためには、事前に k 回の push が必要。コストは push に「前借り」されている。

</details>

**問題 A3: 隠れた計算量の発見**

以下のコードの計算量を求めよ。見た目と実際の計算量が異なる点に注意せよ。

```python
def process(data: list) -> list:
    result = []
    for item in data:                  # n 回
        result = result + [item]       # ← この行に注目
    return result
```

<details>
<summary>解答 A3</summary>

**時間計算量: O(n²)**

`result = result + [item]` は `result.append(item)` とは異なる。
- `result + [item]` は新しいリストを生成し、既存の result の全要素をコピーする
- i 回目のループで長さ i のリストをコピーするため、コスト O(i)
- 合計: 1 + 2 + ... + n = n(n+1)/2 = O(n²)

`result.append(item)` を使えば償却 O(1) × n 回 = O(n) になる。

**教訓:** Python の `+` によるリスト結合は新しいリストを生成する。ループ内でのリスト結合は `append` を使うべきである。

</details>

### 10.3 発展レベル

**問題 E1: 置換法による証明**

漸化式 T(n) = 4T(n/2) + n について、T(n) = O(n²) を置換法で証明せよ。

<details>
<summary>解答 E1</summary>

**主張:** T(n) ≤ cn² (ある定数 c > 0 に対して)

**帰納仮定:** T(k) ≤ ck² (全ての k < n に対して)

**帰納段階:**
```
T(n) = 4T(n/2) + n
     ≤ 4c(n/2)² + n          (帰納仮定)
     = 4c · n²/4 + n
     = cn² + n
```

ここで cn² + n ≤ cn² を示す必要があるが、n > 0 のためこれは不可能。

**修正:** T(n) ≤ cn² - dn を推測する（低次項を引く技法）。

```
T(n) = 4T(n/2) + n
     ≤ 4[c(n/2)² - d(n/2)] + n
     = 4c·n²/4 - 4d·n/2 + n
     = cn² - 2dn + n
     = cn² - dn - (dn - n)
     = cn² - dn - n(d - 1)
     ≤ cn² - dn                (d ≥ 1 のとき)
```

よって d ≥ 1 のとき帰納段階が成立する。基底段階も適切な c を選べば成立。

**結論:** T(n) = O(n²)

（マスター定理でも確認: a=4, b=2, f(n)=n。n^{log_b a} = n^{log₂ 4} = n²。f(n) = n = O(n^{2-ε}) (ε=1) なので Case 1: T(n) = Θ(n²)）

</details>

**問題 E2: 下界の証明**

比較ベースの探索アルゴリズムが、ソート済み配列上で最悪 Ω(log n) の比較を必要とすることを証明せよ。

<details>
<summary>解答 E2</summary>

**証明（決定木による議論）:**

比較ベースの探索アルゴリズムは、各ステップで1つの比較（≤ か >）を行い、その結果に基づいて分岐する。これは二分決定木としてモデル化できる。

1. 探索対象は n 個の要素のうちの1つ、または「存在しない」の n+1 通りの結果がある。
2. 二分決定木の高さ h の木には最大 2^h 個の葉がある。
3. 全ての結果を区別するには: 2^h ≥ n + 1
4. よって h ≥ log₂(n + 1) = Ω(log n)

**結論:** ソート済み配列上の探索は最悪 Ω(log n) 回の比較が必要。二分探索は O(log n) で動作するため、最適なアルゴリズムである。

</details>

**問題 E3: 実践的な計算量改善**

以下の O(n³) のコードを O(n²) に改善せよ。

```python
def count_triples(arr: list, target: int) -> int:
    """arr の中から合計が target になる3つ組の個数を数える"""
    n = len(arr)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if arr[i] + arr[j] + arr[k] == target:
                    count += 1
    return count
```

<details>
<summary>解答 E3</summary>

```python
def count_triples_optimized(arr: list, target: int) -> int:
    """
    ソート + Two Pointer で O(n²) に改善。

    なぜ O(n²) になるのか:
    - 外側のループ: O(n)
    - 内側の Two Pointer: O(n)（左右から挟み込むため線形）
    - 合計: O(n²)
    - ソート: O(n log n) < O(n²) なので全体は O(n²)
    """
    arr_sorted = sorted(arr)  # O(n log n)
    n = len(arr_sorted)
    count = 0

    for i in range(n - 2):                    # O(n)
        left = i + 1
        right = n - 1
        while left < right:                   # O(n) 償却
            s = arr_sorted[i] + arr_sorted[left] + arr_sorted[right]
            if s == target:
                count += 1
                left += 1
                right -= 1
            elif s < target:
                left += 1
            else:
                right -= 1

    return count
```

注意: この解答は重複要素がある場合の処理が簡略化されている。完全な実装では、同じ値の要素をスキップするロジックが必要になる場合がある。

</details>

---

## 11. 比較表まとめ

### 表1: 漸近記法の完全比較

| 記法 | 意味 | 数学的定義 | 直感的説明 | 数の不等式との対応 |
|------|------|-----------|-----------|-------------------|
| f = O(g) | 上界 | ∃c>0, n₀: f(n) ≤ cg(n) | 最悪でも g 程度 | a ≤ b |
| f = Ω(g) | 下界 | ∃c>0, n₀: f(n) ≥ cg(n) | 少なくとも g 程度 | a ≥ b |
| f = Θ(g) | タイト | c₁g(n) ≤ f(n) ≤ c₂g(n) | ちょうど g 程度 | a = b |
| f = o(g) | 狭義上界 | ∀c>0, ∃n₀: f(n) < cg(n) | g より真に小さい | a < b |
| f = ω(g) | 狭義下界 | ∀c>0, ∃n₀: f(n) > cg(n) | g より真に大きい | a > b |

### 表2: データ構造の操作別計算量

| データ構造 | 参照 | 挿入 | 削除 | 探索 | 空間 |
|-----------|------|------|------|------|------|
| 配列 | O(1) | O(n) | O(n) | O(n) | O(n) |
| 動的配列 | O(1) | O(1)* | O(n) | O(n) | O(n) |
| 連結リスト | O(n) | O(1)** | O(1)** | O(n) | O(n) |
| ハッシュテーブル | - | O(1)* | O(1)* | O(1)* | O(n) |
| 二分探索木(平衡) | - | O(log n) | O(log n) | O(log n) | O(n) |
| ヒープ | O(1)*** | O(log n) | O(log n) | O(n) | O(n) |

\* 償却または平均
\*\* 挿入/削除する位置が既知の場合
\*\*\* 最小値(または最大値)のみ O(1)

---

## 12. FAQ

### Q1: O(n) と Θ(n) の違いは何か？

**A:** O(n) は「n 以下のオーダー」であり上界のみを規定する。Θ(n) は「ちょうど n のオーダー」であり上界かつ下界を規定する。

具体例: f(n) = 5n + 3 について
- f(n) = O(n) ✓（上界として n のオーダー）
- f(n) = O(n²) ✓（n² も上界だが「緩い」上界）
- f(n) = Θ(n) ✓（タイトな上下界）
- f(n) = Θ(n²) ✗（下界が n なので n² のオーダーではない）

**実用上の指針:** 可能であれば Θ を使う方が情報量が多い。しかし、最悪ケースの上界だけが分かっている場合は O を使う。

### Q2: ループが入れ子でなくても O(n²) になることはあるか？

**A:** ある。代表的な例を3つ示す：

```python
# 例1: 文字列の繰り返し連結
s = ""
for i in range(n):
    s += str(i)  # 毎回新しい文字列を生成 → O(1+2+...+n) = O(n²)

# 例2: リスト先頭への挿入
result = []
for x in data:
    result.insert(0, x)  # 毎回全要素をシフト → O(n²)

# 例3: 再帰展開
# T(n) = T(n-1) + n → T(n) = n + (n-1) + ... + 1 = O(n²)
```

### Q3: 償却計算量と平均計算量はどう違うのか？

**A:**

| 観点 | 平均計算量 | 償却計算量 |
|------|-----------|-----------|
| 対象 | 単一の操作 | 一連の操作列 |
| 仮定 | 入力の確率分布を仮定 | 確率の仮定なし |
| 保証 | 期待値として成立 | 最悪の操作列でも成立 |
| 例 | クイックソートの平均 O(n log n) | 動的配列の append 償却 O(1) |

動的配列の append は、どのような順序で操作しても、n 回の操作の合計が O(n) であることが**保証**されている。一方、クイックソートの O(n log n) は「ランダムな入力に対する期待値」であり、特定の入力に対しては O(n²) になる可能性がある。

### Q4: Big-O は最悪ケースを表すのか？

**A:** いいえ。Big-O は数学的な上界の記法であり、最良・平均・最悪のどのケースにも適用できる。

```
誤解: 「Big-O = 最悪ケース」
正解: Big-O は任意の関数の上界を表す記法

正しい用法の例:
  「クイックソートの最悪計算量は O(n²)」
  「クイックソートの平均計算量は O(n log n)」
  「クイックソートの最良計算量は O(n log n)」

全て Big-O を使っているが、それぞれ異なるケースを述べている。
ケースと記法は直交した概念である。
```

### Q5: 同じ O(n log n) のアルゴリズムでもパフォーマンスが異なるのはなぜか？

**A:** Big-O 記法は定数係数と低次項を隠蔽するためである。

```
例: マージソートとクイックソート

マージソート: T(n) ≈ 1.44 · n · log₂ n（比較回数の期待値）
クイックソート: T(n) ≈ 1.39 · n · log₂ n（比較回数の期待値）

どちらも O(n log n) だが:
1. 定数係数が異なる（クイックソートの方が若干少ない）
2. キャッシュ効率が異なる（クイックソートはメモリの局所性が高い）
3. 追加メモリが異なる（マージソートは O(n)、クイックソートは O(log n)）

→ Big-O が同じでも、実際のパフォーマンスは大きく異なり得る。
  Big-O は「オーダーの比較」のためのツールであり、
  同じオーダー内の比較には不向き。
```

### Q6: log の底は計算量に影響するか？

**A:** 漸近記法においては影響しない。なぜなら、底の変換は定数倍だからである。

```
log_a(n) = log_b(n) / log_b(a)

log_b(a) は定数なので:
  O(log_a n) = O(log_b n / log_b a) = O(log_b n)

具体例:
  log₂(1024) = 10
  log₁₀(1024) ≈ 3.01
  log₃(1024) ≈ 6.29

いずれも定数倍の関係にある → 同じ O(log n)

ただし実際のアルゴリズムでは:
  - 二分探索 → 各ステップで半分に → 操作回数 = log₂ n
  - 三分探索 → 各ステップで1/3に → 操作回数 = log₃ n ≈ 0.63 · log₂ n
  漸近記法では同じだが、定数係数は異なる。
```

---

## 13. まとめ

| 項目 | ポイント |
|------|---------|
| O 記法 | 上界を示す。「最悪でもこの程度」。定数 c と閾値 n₀ が存在する |
| Ω 記法 | 下界を示す。「少なくともこの程度かかる」。下限の証明に使用 |
| Θ 記法 | タイトな境界。上界 = 下界のときの正確なオーダー |
| o / ω 記法 | 狭義の上界/下界。「真に小さい/大きい」ことを示す |
| 帰納法 | 再帰的な計算量の厳密な証明に不可欠。基底段階を忘れない |
| 空間計算量 | 補助メモリ + 再帰スタックを考慮。時間とトレードオフ |
| 償却計算量 | 連続操作の合計コスト / 操作回数。確率に依存しない保証 |
| 定数係数 | 漸近記法では無視。同オーダー内の比較には不向き |
| ケースの明示 | 最良/平均/最悪を区別して記述すべき |
| 隠れた計算量 | 文字列結合、リスト結合、`in` 演算に注意 |

---

## 次に読むべきガイド

- [計算量解析 -- 再帰の計算量とマスター定理](./01-complexity-analysis.md)
- [時間空間トレードオフ -- メモ化とブルームフィルタ](./02-space-time-tradeoff.md)

---

## 参考文献

1. Cormen, T.H., Leiserson, C.E., Rivest, R.L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. -- 第3章「Characterizing Running Times」で O/Ω/Θ の厳密な定義と性質を解説。第4章「Divide-and-Conquer」でマスター定理と置換法を詳述。第16章「Amortized Analysis」で集約法・会計法・ポテンシャル法を体系的に解説。
2. Knuth, D.E. (1997). *The Art of Computer Programming, Volume 1: Fundamental Algorithms* (3rd ed.). Addison-Wesley. -- 漸近記法の原典。Knuth は Big-O 記法を計算機科学に導入した人物であり、Ω 記法と Θ 記法の必要性も提唱した。
3. Skiena, S.S. (2020). *The Algorithm Design Manual* (3rd ed.). Springer. -- 第2章「Algorithm Analysis」で計算量の実践的な分析手法を解説。特にコードから計算量を読み取るパターン認識が有用。
4. Sedgewick, R. & Wayne, K. (2011). *Algorithms* (4th ed.). Addison-Wesley. -- 計算量の実践的解説に優れ、Java のコード例と豊富な図解を提供。
5. Sipser, M. (2012). *Introduction to the Theory of Computation* (3rd ed.). Cengage Learning. -- 計算量クラス（P, NP 等）の理論的基盤を提供。計算量理論のより深い理解を目指す場合に参照。
6. MIT OpenCourseWare. *6.006 Introduction to Algorithms*. Massachusetts Institute of Technology. https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/ -- MIT の計算量解析の講義資料。無料で利用可能。
7. Roughgarden, T. (2017). *Algorithms Illuminated* (Part 1). Soundlikeyourself Publishing. -- 漸近記法とアルゴリズム設計の入門に最適。直感的な説明と厳密な議論のバランスが良い。

