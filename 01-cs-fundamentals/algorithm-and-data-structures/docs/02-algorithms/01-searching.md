# 探索アルゴリズム

> データ集合から目的の要素を効率的に見つけ出す手法を、線形探索・二分探索・補間探索・指数探索・三分探索の多角的な視点で理解する

## この章で学ぶこと

1. **線形探索・二分探索・補間探索**の原理と計算量を比較し、適切な場面で使い分けられる
2. **二分探索の変形パターン**（lower_bound, upper_bound, 条件探索, 浮動小数点探索）を正確に実装できる
3. **指数探索・三分探索**など発展的アルゴリズムの動作原理と適用条件を理解する
4. **探索とデータ構造の関係**を理解し、前処理としてのソートやインデックスの意義を把握する
5. **各言語の標準ライブラリ**を活用し、バグのない探索処理を素早く実装できる

---

## 1. 探索アルゴリズムの全体像

### 1.1 探索とは何か

探索（Search）とは、データの集合の中から特定の条件を満たす要素を見つけ出す操作である。プログラミングにおいて最も基本的かつ頻出する操作の一つであり、データベースのクエリ処理、ファイルシステムの検索、Web 検索エンジンのインデックス参照など、あらゆるソフトウェアの根幹を支えている。

探索アルゴリズムの選択は、以下の要因によって決定される。

- **データの構造**: 配列、リンクリスト、木、グラフなど
- **データの状態**: ソート済みか未ソートか、分布の特性
- **データの規模**: 要素数が数十か数百万か
- **探索の頻度**: 一度きりの探索か、繰り返し探索するか
- **メモリ制約**: 追加のデータ構造を構築できるか

### 1.2 探索アルゴリズムの分類体系

```
                        探索アルゴリズム
                             │
            ┌────────────────┼────────────────┐
            │                │                │
       配列上の探索      木構造の探索     ハッシュ探索
            │                │                │
     ┌──────┼──────┐    ┌───┴───┐         ┌──┴──┐
     │      │      │    │       │         │     │
   線形   二分   補間  BST   B-Tree   チェイン  開番地
   探索   探索   探索  探索    探索      法      法
     │
   ┌─┴──┐
   │    │
  単純  番兵法
```

本ガイドでは「配列上の探索」に焦点を当て、線形探索から三分探索までを体系的に解説する。木構造の探索は別ガイド「木とヒープ」、ハッシュ探索は「ハッシュテーブル」で扱う。

### 1.3 計算量の直感的理解

各探索アルゴリズムが要素数 n の配列を処理する場合の比較回数の目安を示す。

| 要素数 n | 線形探索 O(n) | 二分探索 O(log n) | 補間探索 O(log log n) |
|:---------|:-------------|:-----------------|:---------------------|
| 10 | 10 | 4 | 2 |
| 100 | 100 | 7 | 3 |
| 1,000 | 1,000 | 10 | 4 |
| 10,000 | 10,000 | 14 | 4 |
| 100,000 | 100,000 | 17 | 5 |
| 1,000,000 | 1,000,000 | 20 | 5 |
| 10,000,000 | 10,000,000 | 24 | 5 |

この表から分かるように、二分探索は要素数が 1,000 万であっても 24 回の比較で探索が完了する。さらに補間探索はデータが均一分布であれば 5 回程度で済む。アルゴリズムの選択がいかに性能に影響するかが一目瞭然である。

### 1.4 探索アルゴリズム選択のフローチャート

```
データはソート済みか？
├─ NO ─── 探索は 1 回だけか？
│          ├─ YES → 線形探索 O(n)
│          └─ NO ── ソートしてから二分探索 O(n log n + k log n)
│                   ※ k 回探索するなら k > n/log n でソートが有利
│
└─ YES ── データは均一分布か？
           ├─ YES → 補間探索 O(log log n)
           └─ NO ── データサイズは既知か？
                     ├─ YES → 二分探索 O(log n)
                     └─ NO ── 指数探索 O(log n)
```

---

## 2. 線形探索（Linear Search）

### 2.1 概要と原理

線形探索は、先頭から順番に 1 つずつ要素を比較していく最も単純な探索アルゴリズムである。前処理が不要で、ソートされていないデータにも適用できるため、小規模データや単発の探索において最も実用的な手法である。

**動作の可視化:**

```
探索: key = 7
配列: [3, 8, 1, 7, 5, 2]

Step 1: i=0  [3, 8, 1, 7, 5, 2]
              ^
              3 != 7 → 次へ

Step 2: i=1  [3, 8, 1, 7, 5, 2]
                 ^
                 8 != 7 → 次へ

Step 3: i=2  [3, 8, 1, 7, 5, 2]
                    ^
                    1 != 7 → 次へ

Step 4: i=3  [3, 8, 1, 7, 5, 2]
                       ^
                       7 == 7 → 発見! インデックス 3 を返す
```

### 2.2 計算量分析

| ケース | 比較回数 | 説明 |
|:-------|:---------|:-----|
| 最良 | O(1) | 先頭要素がターゲット |
| 平均 | O(n) | ターゲットが中間付近にある場合、n/2 回 |
| 最悪 | O(n) | ターゲットが末尾にあるか存在しない |
| 空間 | O(1) | 追加メモリ不要 |

平均比較回数の導出: 要素がランダムに配置されており、各位置に等確率で存在すると仮定すると、平均比較回数は (1 + 2 + ... + n) / n = (n + 1) / 2 となる。

### 2.3 基本実装

```python
def linear_search(arr: list, target) -> int:
    """線形探索 - O(n)

    配列 arr からターゲットを探し、見つかればインデックスを返す。
    見つからなければ -1 を返す。

    Args:
        arr: 探索対象の配列
        target: 探索する値
    Returns:
        見つかった場合はインデックス、見つからなければ -1
    """
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1


# --- 動作確認 ---
data = [15, 23, 8, 42, 16, 4]
print(linear_search(data, 42))   # 3
print(linear_search(data, 99))   # -1
print(linear_search(data, 15))   # 0  (先頭要素)
print(linear_search(data, 4))    # 5  (末尾要素)
print(linear_search([], 1))      # -1 (空配列)
```

### 2.4 番兵法（Sentinel Search）

番兵法は、配列の末尾にターゲットと同じ値（番兵: sentinel）を配置することで、ループ内の境界チェックを省略する最適化手法である。通常の線形探索では各反復で「境界チェック」と「値の比較」の 2 回の条件判定が必要だが、番兵法ではこれを 1 回に削減できる。

```python
def sentinel_search(arr: list, target) -> int:
    """番兵法による線形探索 - ループ内の条件判定を半減

    注意: 元の配列を一時的に変更するため、マルチスレッド環境では安全でない。

    Args:
        arr: 探索対象の配列（要素数 1 以上）
        target: 探索する値
    Returns:
        見つかった場合はインデックス、見つからなければ -1
    """
    if len(arr) == 0:
        return -1

    n = len(arr)
    last = arr[n - 1]
    arr[n - 1] = target  # 番兵を設置

    i = 0
    while arr[i] != target:
        i += 1

    arr[n - 1] = last  # 元の値を復元

    if i < n - 1 or last == target:
        return i
    return -1


# --- 動作確認 ---
data = [15, 23, 8, 42, 16, 4]
print(sentinel_search(data, 42))   # 3
print(sentinel_search(data, 99))   # -1
print(sentinel_search(data, 4))    # 5  (末尾要素 = 番兵位置)

# 番兵法の効果: 100万要素での比較
import time

large_data = list(range(1_000_000))
target = 999_999  # 最悪ケース（末尾に存在）

start = time.perf_counter()
for _ in range(10):
    linear_search(large_data, target)
t1 = time.perf_counter() - start

start = time.perf_counter()
for _ in range(10):
    sentinel_search(large_data, target)
t2 = time.perf_counter() - start

print(f"通常の線形探索: {t1:.4f}秒")
print(f"番兵法:         {t2:.4f}秒")
# Python では解釈コストが大きいため差は小さいが、C/C++ では顕著な差が出る
```

### 2.5 全探索（Find All）

特定の値の全出現位置を取得する場合の実装を示す。

```python
def find_all(arr: list, target) -> list[int]:
    """ターゲットの全出現インデックスをリストで返す

    Args:
        arr: 探索対象の配列
        target: 探索する値
    Returns:
        ターゲットが出現する全インデックスのリスト
    """
    return [i for i, val in enumerate(arr) if val == target]


# --- 動作確認 ---
data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
print(find_all(data, 5))    # [4, 8, 10]
print(find_all(data, 1))    # [1, 3]
print(find_all(data, 7))    # []
```

### 2.6 条件付き線形探索

値の一致だけでなく、任意の条件関数を使った探索も線形探索の重要な応用である。

```python
from typing import Callable, TypeVar

T = TypeVar('T')

def find_first(arr: list[T], predicate: Callable[[T], bool]) -> int:
    """条件を満たす最初の要素のインデックスを返す

    Args:
        arr: 探索対象の配列
        predicate: 条件関数（要素を受け取り bool を返す）
    Returns:
        条件を満たす最初の要素のインデックス、なければ -1
    """
    for i, val in enumerate(arr):
        if predicate(val):
            return i
    return -1


def find_min_by(arr: list[T], key: Callable[[T], float]) -> int:
    """キー関数の値が最小の要素のインデックスを返す

    Args:
        arr: 探索対象の配列（要素数 1 以上）
        key: 比較キーを返す関数
    Returns:
        キー値が最小の要素のインデックス
    """
    if not arr:
        return -1
    min_idx = 0
    min_val = key(arr[0])
    for i in range(1, len(arr)):
        v = key(arr[i])
        if v < min_val:
            min_val = v
            min_idx = i
    return min_idx


# --- 動作確認 ---
students = [
    {"name": "Alice", "score": 85},
    {"name": "Bob", "score": 92},
    {"name": "Charlie", "score": 78},
    {"name": "Diana", "score": 96},
]

# 90点以上の最初の学生
idx = find_first(students, lambda s: s["score"] >= 90)
print(f"90点以上の最初の学生: {students[idx]['name']}")  # Bob

# スコアが最低の学生
idx = find_min_by(students, lambda s: s["score"])
print(f"最低点の学生: {students[idx]['name']}")  # Charlie
```

### 2.7 線形探索の使いどころ

**線形探索が最適な場面:**
- 要素数が 50 以下程度の小規模データ（ソートのオーバーヘッドが無駄）
- 探索が 1 回限りで、前処理の時間が取れない場合
- リンクリストなどランダムアクセスが O(n) のデータ構造
- データが頻繁に追加・削除される場合（ソート維持のコストが高い）
- 条件が複雑で、単純な大小比較では判定できない場合

**線形探索を避けるべき場面:**
- 同じデータに対して繰り返し探索する場合 → ソートして二分探索
- 要素数が数千以上の大規模データ → 二分探索またはハッシュテーブル
- リアルタイム性が要求される処理 → O(1) のハッシュ探索

---

## 3. 二分探索（Binary Search）

### 3.1 概要と原理

二分探索は、**ソート済み配列**を対象に、探索範囲を半分ずつ絞り込んでいくアルゴリズムである。各ステップで中央要素とターゲットを比較し、ターゲットが中央より大きければ右半分、小さければ左半分に探索範囲を狭める。この操作により、n 要素の配列を最大 ceil(log2(n)) 回の比較で探索できる。

二分探索は、コンピュータサイエンスにおいて最も重要なアルゴリズムの一つである。Jon Bentley は著書 *Programming Pearls* の中で、「二分探索のアイデアは驚くほど単純だが、正しく実装できるプログラマーは全体の 10% に過ぎない」と述べている。特に境界条件（off-by-one エラー）とループ不変量の管理が正確な実装の鍵となる。

### 3.2 動作の可視化

```
探索: key = 23
配列: [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]
       0  1  2   3   4   5   6   7   8   9

=== ステップ 1 ===
low=0, high=9, mid=(0+9)//2=4
[2, 5, 8, 12, 16, 23, 38, 56, 72, 91]
  L               M                H
arr[4]=16 < 23 → ターゲットは右半分にある → low = mid + 1 = 5

=== ステップ 2 ===
low=5, high=9, mid=(5+9)//2=7
[2, 5, 8, 12, 16, 23, 38, 56, 72, 91]
                   L          M      H
arr[7]=56 > 23 → ターゲットは左半分にある → high = mid - 1 = 6

=== ステップ 3 ===
low=5, high=6, mid=(5+6)//2=5
[2, 5, 8, 12, 16, 23, 38, 56, 72, 91]
                   LM    H
arr[5]=23 == 23 → 発見! インデックス 5 を返す

結果: 3 回の比較で探索完了（線形探索なら 6 回必要）
```

### 3.3 ループ不変量による正当性の証明

二分探索の正しさを理解するには、**ループ不変量（loop invariant）**の概念が重要である。

```
ループ不変量: ターゲットが配列中に存在するならば、arr[low..high] の範囲内にある。

初期化: low=0, high=n-1 で配列全体が探索範囲 → 不変量成立
維持:   arr[mid] < target → low=mid+1 としても不変量は維持される
        （target は arr[mid+1..high] にあるはず）
        arr[mid] > target → high=mid-1 としても同様
終了:   low > high → 探索範囲が空 → ターゲットは存在しない
        arr[mid] == target → 発見
```

この不変量により、while ループの各反復で探索範囲が確実に縮小し、必ず有限回で終了することが保証される。

### 3.4 基本実装（反復版）

```python
def binary_search(arr: list, target) -> int:
    """二分探索（反復版）- O(log n)

    ソート済み配列 arr からターゲットを探す。

    ループ不変量:
        target が arr に存在するなら arr[low..high] の範囲内にある。

    Args:
        arr: ソート済み配列
        target: 探索する値
    Returns:
        見つかった場合はインデックス、見つからなければ -1
    """
    low, high = 0, len(arr) - 1

    while low <= high:
        mid = low + (high - low) // 2  # オーバーフロー防止
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1

    return -1


# --- 動作確認 ---
data = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]
print(binary_search(data, 23))    # 5
print(binary_search(data, 2))     # 0  (先頭要素)
print(binary_search(data, 91))    # 9  (末尾要素)
print(binary_search(data, 99))    # -1 (存在しない: 範囲外・上)
print(binary_search(data, 1))     # -1 (存在しない: 範囲外・下)
print(binary_search(data, 10))    # -1 (存在しない: 範囲内)
print(binary_search([], 5))       # -1 (空配列)
print(binary_search([42], 42))    # 0  (要素1つ: 一致)
print(binary_search([42], 99))    # -1 (要素1つ: 不一致)
```

### 3.5 再帰版の実装

```python
def binary_search_recursive(arr: list, target, low: int = 0, high: int = None) -> int:
    """二分探索（再帰版）- O(log n) 時間, O(log n) 空間（コールスタック）

    再帰版は理解しやすいが、スタックオーバーフローのリスクがあるため
    本番コードでは反復版を推奨する。

    Args:
        arr: ソート済み配列
        target: 探索する値
        low: 探索範囲の下限（デフォルト 0）
        high: 探索範囲の上限（デフォルト len(arr)-1）
    Returns:
        見つかった場合はインデックス、見つからなければ -1
    """
    if high is None:
        high = len(arr) - 1
    if low > high:
        return -1

    mid = low + (high - low) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, high)
    else:
        return binary_search_recursive(arr, target, low, mid - 1)


# --- 動作確認 ---
data = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]
print(binary_search_recursive(data, 23))  # 5
print(binary_search_recursive(data, 99))  # -1
```

### 3.6 lower_bound と upper_bound

二分探索の最も重要な変形が **lower_bound** と **upper_bound** である。これらは「ソート済み配列への挿入位置」を求めるアルゴリズムであり、値の一致判定だけでなく範囲クエリや出現回数の算出に不可欠である。

```
配列: [1, 2, 2, 2, 3, 4, 5]
       0  1  2  3  4  5  6

lower_bound(2) = 1  ← 2 以上の最小インデックス（最初の 2）
upper_bound(2) = 4  ← 2 より大きい最小インデックス（最後の 2 の次）

出現回数 = upper_bound - lower_bound = 4 - 1 = 3

                lower_bound(2)         upper_bound(2)
                    |                       |
         [1,  2,  2,  2,  3,  4,  5]
          0   1   2   3   4   5   6
              ^^^^^^^^^^
              2 が存在する範囲
```

```python
def lower_bound(arr: list, target) -> int:
    """target 以上の最小インデックスを返す（C++ の std::lower_bound 相当）

    全要素が target 未満の場合は len(arr) を返す。

    ループ不変量:
        arr[0..low-1] の全要素 < target
        arr[high..n-1] の全要素 >= target

    Args:
        arr: ソート済み配列
        target: 探索する値
    Returns:
        target 以上の最初の要素のインデックス（0 ~ len(arr)）
    """
    low, high = 0, len(arr)
    while low < high:
        mid = low + (high - low) // 2
        if arr[mid] < target:
            low = mid + 1
        else:
            high = mid
    return low


def upper_bound(arr: list, target) -> int:
    """target より大きい最小インデックスを返す（C++ の std::upper_bound 相当）

    全要素が target 以下の場合は len(arr) を返す。

    ループ不変量:
        arr[0..low-1] の全要素 <= target
        arr[high..n-1] の全要素 > target

    Args:
        arr: ソート済み配列
        target: 探索する値
    Returns:
        target より大きい最初の要素のインデックス（0 ~ len(arr)）
    """
    low, high = 0, len(arr)
    while low < high:
        mid = low + (high - low) // 2
        if arr[mid] <= target:
            low = mid + 1
        else:
            high = mid
    return low


def count_occurrences(arr: list, target) -> int:
    """ソート済み配列中のターゲットの出現回数を O(log n) で求める"""
    return upper_bound(arr, target) - lower_bound(arr, target)


def find_range(arr: list, target) -> tuple[int, int]:
    """ソート済み配列中のターゲットの出現範囲 [first, last] を返す

    存在しない場合は (-1, -1) を返す。
    """
    lb = lower_bound(arr, target)
    if lb == len(arr) or arr[lb] != target:
        return (-1, -1)
    ub = upper_bound(arr, target)
    return (lb, ub - 1)


# --- 動作確認 ---
data = [1, 2, 2, 2, 3, 4, 5]
print(lower_bound(data, 2))         # 1
print(upper_bound(data, 2))         # 4
print(count_occurrences(data, 2))   # 3
print(find_range(data, 2))          # (1, 3)
print(find_range(data, 6))          # (-1, -1) 存在しない

# 挿入位置としての使い方
print(lower_bound(data, 2.5))       # 4 (2.5を挿入するなら位置4)
print(lower_bound(data, 0))         # 0 (先頭に挿入)
print(lower_bound(data, 10))        # 7 (末尾に挿入)
```

### 3.7 条件二分探索（Predicate Binary Search）

「ある条件を満たす最小（最大）の値を求める」問題は、条件関数が単調（False, False, ..., True, True, ...）であれば二分探索で解ける。これは競技プログラミングや最適化問題で頻出するパターンである。

```python
def binary_search_condition(low: int, high: int, condition) -> int:
    """条件を満たす最小値を二分探索で求める

    前提条件（単調性）:
        condition(x) が False, False, ..., True, True, ...
        という形の単調増加関数であること。

    Args:
        low: 探索範囲の下限
        high: 探索範囲の上限
        condition: 判定関数（int -> bool）
    Returns:
        condition(x) が True となる最小の x
    """
    while low < high:
        mid = low + (high - low) // 2
        if condition(mid):
            high = mid
        else:
            low = mid + 1
    return low


# --- 例1: x^2 >= 100 となる最小の正整数 ---
result = binary_search_condition(1, 100, lambda x: x * x >= 100)
print(f"x^2 >= 100 の最小 x: {result}")  # 10

# --- 例2: 配列の要素の合計が S 以下になる最大の部分配列長 ---
def max_subarray_length(arr: list[int], max_sum: int) -> int:
    """連続部分配列の合計が max_sum 以下になる最大長を求める"""
    prefix = [0] * (len(arr) + 1)
    for i in range(len(arr)):
        prefix[i + 1] = prefix[i] + arr[i]

    def can_fit(length: int) -> bool:
        """長さ length の連続部分配列で合計が max_sum 以下のものが存在するか"""
        for i in range(len(arr) - length + 1):
            if prefix[i + length] - prefix[i] <= max_sum:
                return True
        return False

    # can_fit は「短い方が True になりやすい」ので、反転して探索
    # 「長さ L で can_fit が False になる最小の L」を求め、L-1 が答え
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if can_fit(mid + 1):  # mid+1 の長さでも可能か？
            lo = mid + 1
        else:
            hi = mid
    return lo


arr = [1, 2, 3, 4, 5]
print(f"合計 6 以下の最大部分配列長: {max_subarray_length(arr, 6)}")   # 3 ([1,2,3])
print(f"合計 10 以下の最大部分配列長: {max_subarray_length(arr, 10)}")  # 4 ([1,2,3,4])
```

### 3.8 浮動小数点数の二分探索

整数ではなく実数値上で二分探索を行う場合、`low <= high` の代わりに精度条件または固定反復回数を使う。

```python
import math

def binary_search_float(f, target: float, low: float, high: float,
                        iterations: int = 100) -> float:
    """浮動小数点数の二分探索

    f(x) が単調増加で f(low) <= target <= f(high) の条件下で
    f(x) = target となる x を近似する。

    Args:
        f: 単調増加関数
        target: 目標値
        low: 探索範囲の下限
        high: 探索範囲の上限
        iterations: 反復回数（100回で約 10^-30 の精度）
    Returns:
        f(x) ≈ target となる x の近似値
    """
    for _ in range(iterations):
        mid = (low + high) / 2
        if f(mid) < target:
            low = mid
        else:
            high = mid
    return (low + high) / 2


# --- 例1: 平方根の計算（sqrt(2)）---
sqrt2 = binary_search_float(lambda x: x * x, 2.0, 0.0, 2.0)
print(f"sqrt(2) = {sqrt2:.15f}")          # 1.414213562373095
print(f"math.sqrt(2) = {math.sqrt(2):.15f}")  # 1.414213562373095

# --- 例2: 方程式 x^3 + x = 10 の解 ---
solution = binary_search_float(lambda x: x**3 + x, 10.0, 0.0, 10.0)
print(f"x^3 + x = 10 の解: x = {solution:.10f}")  # 2.0462606289

# 検算
print(f"検算: {solution**3 + solution:.10f}")  # 10.0000000000
```

---

## 4. 補間探索（Interpolation Search）

### 4.1 概要と原理

補間探索は、データが**均一分布**している場合に二分探索を上回る性能を発揮するアルゴリズムである。電話帳で「田中」を探すときに中央ではなく後半の「た行」付近を開くのと同じ発想で、ターゲットの値に基づいて探索位置を「推定」する。

二分探索が常に中央を選ぶのに対し、補間探索は以下の式で探索位置を計算する。

```
                (target - arr[low]) * (high - low)
pos = low + ──────────────────────────────────────────
                    arr[high] - arr[low]
```

### 4.2 動作の可視化

```
配列: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
       0   1   2   3   4   5   6   7   8    9
target = 70

=== 二分探索の場合 ===
Step 1: mid = (0+9)//2 = 4 → arr[4] = 50 < 70 → 右へ
Step 2: mid = (5+9)//2 = 7 → arr[7] = 80 > 70 → 左へ
Step 3: mid = (5+6)//2 = 5 → arr[5] = 60 < 70 → 右へ
Step 4: mid = (6+6)//2 = 6 → arr[6] = 70 == 70 → 発見!
→ 4 ステップ

=== 補間探索の場合 ===
Step 1: pos = 0 + (70-10)*(9-0)/(100-10) = 0 + 60*9/90 = 6
        → arr[6] = 70 == 70 → 発見!
→ 1 ステップ（一発で的中!）
```

### 4.3 計算量分析

| ケース | 計算量 | 条件 |
|:-------|:-------|:-----|
| 最良 | O(1) | ターゲットが推定位置に的中 |
| 平均（均一分布） | O(log log n) | データが均一に分布 |
| 最悪（偏った分布） | O(n) | データが指数的に偏っている場合 |
| 空間 | O(1) | 追加メモリ不要 |

**O(log log n) の直感:** n = 10^9（10 億）のとき、log n ≈ 30 だが log log n ≈ 5 である。均一分布データでは驚異的な速度を発揮する。

### 4.4 実装

```python
def interpolation_search(arr: list[int | float], target) -> int:
    """補間探索 - 均一分布で O(log log n)

    前提条件:
        - 配列がソート済みであること
        - 数値データであること（補間計算に必要）
        - 最高性能を発揮するにはデータが均一分布であること

    Args:
        arr: ソート済みの数値配列
        target: 探索する値
    Returns:
        見つかった場合はインデックス、見つからなければ -1
    """
    low, high = 0, len(arr) - 1

    while (low <= high and
           arr[low] <= target <= arr[high]):

        if low == high:
            return low if arr[low] == target else -1

        # 補間による位置推定
        pos = low + int(
            (target - arr[low]) * (high - low) /
            (arr[high] - arr[low])
        )

        if arr[pos] == target:
            return pos
        elif arr[pos] < target:
            low = pos + 1
        else:
            high = pos - 1

    return -1


# --- 動作確認 ---
# 均一分布データ
uniform_data = list(range(10, 1001, 10))  # [10, 20, 30, ..., 1000]
print(interpolation_search(uniform_data, 700))   # 69
print(interpolation_search(uniform_data, 10))    # 0
print(interpolation_search(uniform_data, 1000))  # 99
print(interpolation_search(uniform_data, 15))    # -1 (存在しない)

# 非均一分布データでは性能が劣化する
import random
skewed_data = sorted([2**i for i in range(20)])  # 指数分布
print(f"指数分布データ: {skewed_data[:5]}...{skewed_data[-3:]}")
# [1, 2, 4, 8, 16, ...262144, 524288]
# → このようなデータでは二分探索の方が安全
```

### 4.5 補間探索の改良: 補間二分探索

純粋な補間探索は偏った分布で O(n) に退化するリスクがある。これを防ぐため、補間と二分の折衷案を使う実装が存在する。

```python
def interpolation_binary_search(arr: list[int | float], target) -> int:
    """補間探索と二分探索のハイブリッド

    補間で推定した位置が探索範囲の外側に出た場合は
    二分探索にフォールバックする。

    Args:
        arr: ソート済みの数値配列
        target: 探索する値
    Returns:
        見つかった場合はインデックス、見つからなければ -1
    """
    low, high = 0, len(arr) - 1

    while low <= high and arr[low] <= target <= arr[high]:
        if low == high:
            return low if arr[low] == target else -1

        # 補間による位置推定
        pos = low + int(
            (target - arr[low]) * (high - low) /
            (arr[high] - arr[low])
        )

        # 安全性チェック: 探索範囲の中央 1/4 ~ 3/4 に収まらなければ
        # 二分探索にフォールバック
        quarter = (high - low) // 4
        if pos < low + quarter or pos > high - quarter:
            pos = low + (high - low) // 2

        if arr[pos] == target:
            return pos
        elif arr[pos] < target:
            low = pos + 1
        else:
            high = pos - 1

    return -1


# --- 動作確認 ---
# 偏った分布でも安全に動作
skewed = sorted([2**i for i in range(20)])
for val in [1, 16, 256, 524288]:
    result = interpolation_binary_search(skewed, val)
    print(f"探索({val}) → インデックス {result}")
```

### 4.6 補間探索を使うべき場面

**適している場面:**
- データが数値型で均一分布している場合（連番 ID、タイムスタンプ等）
- 配列が巨大（数百万要素以上）で探索回数が多い場合
- メモリアクセスのコストが高く、比較回数を最小化したい場合

**避けるべき場面:**
- データの分布が不明または偏っている場合
- 文字列など、補間計算ができないデータ型
- 配列が小規模（1000 以下）で、二分探索との差が無視できる場合

---

## 5. 指数探索（Exponential Search）

### 5.1 概要と原理

指数探索は、**サイズが未知の、または無限に大きいソート済み配列**でターゲットが存在する範囲を素早く特定し、その範囲内で二分探索を適用するアルゴリズムである。

名前の由来は、探索範囲を 1, 2, 4, 8, 16, ... と指数的に拡大しながらターゲットを含む範囲を特定する点にある。ターゲットがインデックス k にある場合、O(log k) で範囲を特定できるため、ターゲットが配列の先頭付近にあるケースで特に高速に動作する。

### 5.2 動作の可視化

```
配列: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
       0  1  2  3   4   5   6   7   8   9  10  11  12  13  14
target = 19

Phase 1: 指数的に範囲を拡大
  bound=1:  arr[1]=3  < 19 → 拡大
  bound=2:  arr[2]=5  < 19 → 拡大
  bound=4:  arr[4]=11 < 19 → 拡大
  bound=8:  arr[8]=23 >= 19 → 停止!
  → ターゲットは arr[4..8] の範囲にある

Phase 2: 範囲 [4, 8] で二分探索
  low=4, high=8, mid=6 → arr[6]=17 < 19 → low=7
  low=7, high=8, mid=7 → arr[7]=19 == 19 → 発見!

結果: ターゲットが位置 7 にあり、log(7) ≈ 3 ステップで範囲特定
```

### 5.3 実装

```python
def exponential_search(arr: list, target) -> int:
    """指数探索 - O(log k) で範囲特定 + O(log k) で二分探索

    ターゲットがインデックス k にある場合、O(log k) で探索完了。
    配列全体のサイズ n ではなく、ターゲットの位置 k に依存する。

    Args:
        arr: ソート済み配列
        target: 探索する値
    Returns:
        見つかった場合はインデックス、見つからなければ -1
    """
    if len(arr) == 0:
        return -1

    # 先頭要素をチェック
    if arr[0] == target:
        return 0

    # 指数的に範囲を拡大して上界を見つける
    bound = 1
    while bound < len(arr) and arr[bound] < target:
        bound *= 2

    # 特定した範囲 [bound//2, min(bound, n-1)] で二分探索
    low = bound // 2
    high = min(bound, len(arr) - 1)

    while low <= high:
        mid = low + (high - low) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1

    return -1


# --- 動作確認 ---
primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
print(exponential_search(primes, 19))    # 7
print(exponential_search(primes, 2))     # 0  (先頭)
print(exponential_search(primes, 47))    # 14 (末尾)
print(exponential_search(primes, 20))    # -1 (存在しない)
print(exponential_search([], 5))         # -1 (空配列)

# 大規模配列で先頭付近のターゲット → 非常に高速
large = list(range(10_000_000))
print(exponential_search(large, 42))     # 42 (log(42)≈6 ステップで範囲特定)
```

### 5.4 指数探索の活用場面

**最適な場面:**
- ターゲットが配列の先頭付近にあることが多い場合
- 配列のサイズが非常に大きく、末尾まで走査したくない場合
- 無限リスト（ジェネレータ等）で上界が不明な場合
- 二分探索のバリアントとして、範囲を事前に絞り込みたい場合

---

## 6. 三分探索（Ternary Search）

### 6.1 概要と原理

三分探索は、**単峰関数**（unimodal function: 一つの極値を持つ関数）の極値を求めるアルゴリズムである。二分探索が「ソート済み配列での値の探索」に使われるのに対し、三分探索は「連続関数の最大値または最小値の位置」を求めるのに使われる。

探索範囲 [low, high] を 3 等分する 2 つの点 m1, m2 を設定し、f(m1) と f(m2) を比較することで極値が含まれる範囲を 2/3 に絞り込む。

### 6.2 動作の可視化

```
f(x) = -(x-5)^2 + 25 の最大値を [0, 10] で探索

      f(x)
  25  |         *
      |       *   *
  20  |     *       *
      |   *           *
  15  | *               *
      |*                 *
   0  +---+---+---+---+---+---
      0   2   4   5   6   8  10
                  ^
                最大値 x=5

=== Step 1: low=0, high=10 ===
  m1 = 0 + (10-0)/3 = 3.33   → f(3.33) = 22.22
  m2 = 10 - (10-0)/3 = 6.67  → f(6.67) = 22.22
  f(m1) ≈ f(m2) → 微妙だが f(m1) <= f(m2) なので low = m1

=== Step 2: low=3.33, high=10 ===
  m1 = 3.33 + (10-3.33)/3 = 5.56  → f(5.56) = 24.69
  m2 = 10 - (10-3.33)/3 = 7.78    → f(7.78) = 17.28
  f(m1) > f(m2) → high = m2

... 反復を続けると x=5 に収束
```

### 6.3 実装

```python
def ternary_search_max(f, low: float, high: float,
                       iterations: int = 200) -> float:
    """三分探索で単峰関数の最大値の位置を求める

    前提条件:
        f は [low, high] の範囲で単峰（unimodal）であること。
        すなわち、ある点 x* で f が最大となり、x < x* では単調増加、
        x > x* では単調減少であること。

    Args:
        f: 単峰関数
        low: 探索範囲の下限
        high: 探索範囲の上限
        iterations: 反復回数
    Returns:
        最大値をとる x の近似値
    """
    for _ in range(iterations):
        m1 = low + (high - low) / 3
        m2 = high - (high - low) / 3
        if f(m1) < f(m2):
            low = m1
        else:
            high = m2
    return (low + high) / 2


def ternary_search_min(f, low: float, high: float,
                       iterations: int = 200) -> float:
    """三分探索で凸関数の最小値の位置を求める

    前提条件:
        f は [low, high] の範囲で下に凸（convex）であること。

    Args:
        f: 凸関数
        low: 探索範囲の下限
        high: 探索範囲の上限
        iterations: 反復回数
    Returns:
        最小値をとる x の近似値
    """
    for _ in range(iterations):
        m1 = low + (high - low) / 3
        m2 = high - (high - low) / 3
        if f(m1) > f(m2):
            low = m1
        else:
            high = m2
    return (low + high) / 2


# --- 例1: 二次関数の最大値 ---
# f(x) = -(x-5)^2 + 25 → 最大値は x=5 で f(5)=25
f1 = lambda x: -(x - 5)**2 + 25
x_max = ternary_search_max(f1, 0, 10)
print(f"最大値の位置: x = {x_max:.6f}, f(x) = {f1(x_max):.6f}")
# x = 5.000000, f(x) = 25.000000

# --- 例2: 二次関数の最小値 ---
# f(x) = (x-3)^2 + 1 → 最小値は x=3 で f(3)=1
f2 = lambda x: (x - 3)**2 + 1
x_min = ternary_search_min(f2, -10, 10)
print(f"最小値の位置: x = {x_min:.6f}, f(x) = {f2(x_min):.6f}")
# x = 3.000000, f(x) = 1.000000

# --- 例3: 実用問題 - 最適な価格設定 ---
# 利益 = 価格 * 需要(価格) - 固定費
# 需要(p) = 1000 - 10p (価格が高いと需要が減る)
# 利益(p) = p * (1000 - 10p) - 500
profit = lambda p: p * (1000 - 10 * p) - 500
optimal_price = ternary_search_max(profit, 0, 100)
print(f"最適価格: {optimal_price:.2f}円")
print(f"最大利益: {profit(optimal_price):.2f}円")
# 最適価格: 50.00円, 最大利益: 24500.00円
```

### 6.4 整数版三分探索

競技プログラミングでは整数範囲で三分探索を行う場合がある。

```python
def ternary_search_int_min(f, low: int, high: int) -> int:
    """整数範囲で凸関数の最小値の位置を求める

    Args:
        f: 凸関数（整数 -> 数値）
        low: 探索範囲の下限
        high: 探索範囲の上限
    Returns:
        最小値をとる整数 x
    """
    while high - low > 2:
        m1 = low + (high - low) // 3
        m2 = high - (high - low) // 3
        if f(m1) > f(m2):
            low = m1
        else:
            high = m2

    # 残り数個の候補を全探索
    best = low
    for x in range(low, high + 1):
        if f(x) < f(best):
            best = x
    return best


# --- 例: |x - 7| + |x - 3| の最小値（区間 [0, 10]）---
f = lambda x: abs(x - 7) + abs(x - 3)
result = ternary_search_int_min(f, 0, 10)
print(f"最小値の位置: x = {result}, f(x) = {f(result)}")
# 3 <= x <= 7 の任意の整数で f(x) = 4
```

### 6.5 三分探索 vs 二分探索（微分版）

関数の微分が計算可能な場合、三分探索の代わりに微分値に対する二分探索を使うことで、より高速に収束させることができる。

```python
def golden_section_search(f, low: float, high: float,
                          tol: float = 1e-12) -> float:
    """黄金分割探索 - 三分探索の改良版

    黄金比 φ = (1+sqrt(5))/2 を利用して探索点を配置する。
    三分探索では各反復で 2 回の関数評価が必要だが、
    黄金分割探索では 1 回で済む（前回の計算を再利用）。

    Args:
        f: 単峰関数（最小値を探索）
        low: 探索範囲の下限
        high: 探索範囲の上限
        tol: 収束判定の許容誤差
    Returns:
        最小値をとる x の近似値
    """
    phi = (1 + 5**0.5) / 2  # 黄金比 ≈ 1.618
    resphi = 2 - phi          # ≈ 0.382

    x1 = low + resphi * (high - low)
    x2 = high - resphi * (high - low)
    f1 = f(x1)
    f2 = f(x2)

    while abs(high - low) > tol:
        if f1 < f2:
            high = x2
            x2, f2 = x1, f1
            x1 = low + resphi * (high - low)
            f1 = f(x1)
        else:
            low = x1
            x1, f1 = x2, f2
            x2 = high - resphi * (high - low)
            f2 = f(x2)

    return (low + high) / 2


# --- 動作確認 ---
f = lambda x: (x - 3.7)**2 + 2.1
result = golden_section_search(f, 0, 10)
print(f"最小値の位置: x = {result:.10f}")  # 3.7000000000
print(f"f(x) = {f(result):.10f}")          # 2.1000000000
```

---

## 7. 探索アルゴリズムの比較と選択指針

### 7.1 総合比較表

| アルゴリズム | 最良 | 平均 | 最悪 | 前提条件 | 空間 | 用途 |
|:---|:---|:---|:---|:---|:---|:---|
| 線形探索 | O(1) | O(n) | O(n) | なし | O(1) | 小規模/未ソート |
| 番兵付き線形探索 | O(1) | O(n) | O(n) | なし | O(1) | 線形探索の定数倍改善 |
| 二分探索 | O(1) | O(log n) | O(log n) | ソート済み | O(1) | 最も汎用的 |
| 補間探索 | O(1) | O(log log n) | O(n) | ソート済み+均一分布 | O(1) | 均一分布の数値データ |
| 指数探索 | O(1) | O(log k) | O(log n) | ソート済み | O(1) | サイズ未知/先頭付近 |
| 三分探索 | - | O(log n) | O(log n) | 単峰関数 | O(1) | 極値探索 |
| 黄金分割探索 | - | O(log n) | O(log n) | 単峰関数 | O(1) | 極値探索（改良版） |
| ハッシュ探索 | O(1) | O(1) | O(n) | ハッシュテーブル | O(n) | 高頻度探索 |

### 7.2 実行性能の比較（Python ベンチマーク）

以下は、各アルゴリズムの相対的な性能差を把握するためのベンチマークコードである。

```python
import time
import bisect
import random

def benchmark_search_algorithms():
    """各探索アルゴリズムの性能比較"""
    sizes = [1_000, 10_000, 100_000, 1_000_000]

    for n in sizes:
        data = list(range(n))
        target = n - 1  # 最悪ケース（末尾）

        # 線形探索
        start = time.perf_counter()
        for _ in range(100):
            linear_search(data, target)
        t_linear = (time.perf_counter() - start) / 100

        # 二分探索
        start = time.perf_counter()
        for _ in range(100_000):
            binary_search(data, target)
        t_binary = (time.perf_counter() - start) / 100_000

        # bisect（C実装）
        start = time.perf_counter()
        for _ in range(100_000):
            idx = bisect.bisect_left(data, target)
        t_bisect = (time.perf_counter() - start) / 100_000

        # 補間探索
        start = time.perf_counter()
        for _ in range(100_000):
            interpolation_search(data, target)
        t_interp = (time.perf_counter() - start) / 100_000

        print(f"n={n:>10,}: "
              f"線形={t_linear:.6f}s "
              f"二分={t_binary:.6f}s "
              f"bisect={t_bisect:.6f}s "
              f"補間={t_interp:.6f}s")


# benchmark_search_algorithms()  # コメントを外して実行
```

### 7.3 探索アルゴリズムと前処理のトレードオフ

探索を繰り返す場合、「ソート + 二分探索」と「毎回線形探索」のどちらが効率的かは、探索回数 k とデータサイズ n の関係で決まる。

```
                     コスト比較
                     ─────────────────────────
    毎回線形探索:     k * O(n)
    ソート+二分探索:  O(n log n) + k * O(log n)
    ハッシュ構築+探索: O(n) + k * O(1)

    損益分岐点（ソート vs 線形）:
        k * n > n log n + k * log n
        k > (n log n) / (n - log n)
        k ≈ log n  (n が十分大きい場合)

    つまり n=1,000,000 の場合、探索が 20 回以上ならソートする価値がある。
```

| 探索回数 k | n=1,000 | n=100,000 | n=10,000,000 |
|:-----------|:--------|:----------|:-------------|
| 1 回 | 線形探索 | 線形探索 | 線形探索 |
| 10 回 | 線形探索 | ソート+二分 | ソート+二分 |
| 100 回 | ソート+二分 | ソート+二分 | ソート+二分 |
| 10,000 回 | ハッシュ | ハッシュ | ハッシュ |

---

## 8. 標準ライブラリの活用

### 8.1 Python: bisect モジュール

Python の `bisect` モジュールは C 言語で実装された高速な二分探索ライブラリである。自前実装よりも大幅に高速であり、本番コードでは必ずこちらを使うべきである。

```python
import bisect

data = [1, 3, 5, 7, 9, 11, 13]

# --- 基本操作 ---
# bisect_left: target 以上の最小インデックス（lower_bound 相当）
print(bisect.bisect_left(data, 7))    # 3
print(bisect.bisect_left(data, 6))    # 3 (6が入るべき位置)
print(bisect.bisect_left(data, 0))    # 0 (全要素より小さい)
print(bisect.bisect_left(data, 20))   # 7 (全要素より大きい)

# bisect_right: target より大きい最小インデックス（upper_bound 相当）
print(bisect.bisect_right(data, 7))   # 4
print(bisect.bisect_right(data, 6))   # 3

# insort: ソート順を維持しながら挿入 - O(n)（挿入自体は O(log n) で位置特定）
bisect.insort(data, 6)
print(data)  # [1, 3, 5, 6, 7, 9, 11, 13]

# --- 実用パターン ---
# パターン1: 要素の存在確認
def contains(sorted_arr: list, target) -> bool:
    """ソート済み配列にターゲットが存在するか O(log n) で判定"""
    idx = bisect.bisect_left(sorted_arr, target)
    return idx < len(sorted_arr) and sorted_arr[idx] == target

data2 = [2, 4, 6, 8, 10]
print(contains(data2, 6))    # True
print(contains(data2, 5))    # False

# パターン2: 範囲カウント（a 以上 b 以下の要素数）
def count_in_range(sorted_arr: list, a, b) -> int:
    """ソート済み配列で [a, b] の範囲にある要素数を O(log n) で求める"""
    return bisect.bisect_right(sorted_arr, b) - bisect.bisect_left(sorted_arr, a)

data3 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(count_in_range(data3, 3, 7))    # 5 (3,4,5,6,7)
print(count_in_range(data3, 5, 5))    # 1 (5のみ)
print(count_in_range(data3, 11, 20))  # 0

# パターン3: 最近傍探索（最も近い値を見つける）
def find_nearest(sorted_arr: list, target) -> int:
    """ソート済み配列でターゲットに最も近い値のインデックスを返す"""
    if not sorted_arr:
        return -1
    idx = bisect.bisect_left(sorted_arr, target)
    if idx == 0:
        return 0
    if idx == len(sorted_arr):
        return len(sorted_arr) - 1
    # 左右の候補を比較
    if target - sorted_arr[idx - 1] <= sorted_arr[idx] - target:
        return idx - 1
    return idx

temps = [10, 15, 20, 25, 30, 35, 40]
idx = find_nearest(temps, 22)
print(f"22 に最も近い値: {temps[idx]}")  # 20

# パターン4: 成績のグレード判定
def grade(score: int) -> str:
    """点数からグレードを判定（bisect の典型的な使い方）"""
    breakpoints = [60, 70, 80, 90]
    grades = ['F', 'D', 'C', 'B', 'A']
    idx = bisect.bisect(breakpoints, score)
    return grades[idx]

for s in [55, 60, 75, 85, 95]:
    print(f"Score {s}: Grade {grade(s)}")
# Score 55: Grade F
# Score 60: Grade D
# Score 75: Grade C
# Score 85: Grade B
# Score 95: Grade A
```

### 8.2 Python: SortedContainers ライブラリ

標準ライブラリではないが、`sortedcontainers` は自動でソート順を維持するコンテナを提供し、探索・挿入・削除を効率的に行える。

```python
# pip install sortedcontainers
from sortedcontainers import SortedList

sl = SortedList([5, 1, 3, 7, 2])
print(sl)  # SortedList([1, 2, 3, 5, 7])

# 追加: O(log n)
sl.add(4)
print(sl)  # SortedList([1, 2, 3, 4, 5, 7])

# 探索: O(log n)
print(sl.index(4))           # 3
print(sl.bisect_left(4))     # 3
print(sl.bisect_right(4))    # 4

# 範囲取得: O(log n + k)
print(list(sl.irange(2, 5)))  # [2, 3, 4, 5]

# 削除: O(log n)
sl.remove(3)
print(sl)  # SortedList([1, 2, 4, 5, 7])
```

### 8.3 C++ 標準ライブラリ

```cpp
#include <algorithm>
#include <vector>
#include <iostream>
using namespace std;

int main() {
    vector<int> v = {1, 3, 5, 7, 9, 11, 13};

    // binary_search: 存在判定
    cout << boolalpha;
    cout << binary_search(v.begin(), v.end(), 7) << endl;  // true
    cout << binary_search(v.begin(), v.end(), 6) << endl;  // false

    // lower_bound: target 以上の最小位置
    auto it = lower_bound(v.begin(), v.end(), 7);
    cout << "lower_bound(7): index=" << (it - v.begin()) << endl;  // 3

    // upper_bound: target より大きい最小位置
    it = upper_bound(v.begin(), v.end(), 7);
    cout << "upper_bound(7): index=" << (it - v.begin()) << endl;  // 4

    // equal_range: lower_bound と upper_bound を同時に取得
    auto [lo, hi] = equal_range(v.begin(), v.end(), 7);
    cout << "range: [" << (lo - v.begin()) << ", "
         << (hi - v.begin()) << ")" << endl;  // [3, 4)

    return 0;
}
```

### 8.4 Java 標準ライブラリ

```java
import java.util.Arrays;
import java.util.Collections;
import java.util.ArrayList;
import java.util.List;

public class SearchExample {
    public static void main(String[] args) {
        // 配列版
        int[] arr = {2, 5, 8, 12, 16, 23, 38, 56};

        // Arrays.binarySearch: 見つかればインデックス、見つからなければ -(挿入位置)-1
        System.out.println(Arrays.binarySearch(arr, 23));   // 5
        System.out.println(Arrays.binarySearch(arr, 10));   // -4 (挿入位置3 → -(3)-1)

        // 挿入位置の取得
        int idx = Arrays.binarySearch(arr, 10);
        int insertionPoint = idx >= 0 ? idx : -(idx + 1);
        System.out.println("10の挿入位置: " + insertionPoint);  // 3

        // List版
        List<Integer> list = new ArrayList<>(List.of(2, 5, 8, 12, 16, 23, 38, 56));
        System.out.println(Collections.binarySearch(list, 23));  // 5
    }
}
```

### 8.5 Go 標準ライブラリ

```go
package main

import (
    "fmt"
    "sort"
)

func main() {
    data := []int{2, 5, 8, 12, 16, 23, 38, 56}

    // sort.SearchInts: ターゲット以上の最小インデックス（lower_bound 相当）
    idx := sort.SearchInts(data, 23)
    fmt.Printf("SearchInts(23): %d\n", idx) // 5

    idx = sort.SearchInts(data, 10)
    fmt.Printf("SearchInts(10): %d\n", idx) // 3 (挿入位置)

    // sort.Search: 条件関数ベース（汎用版）
    // f(i) が true となる最小の i を返す
    idx = sort.Search(len(data), func(i int) bool {
        return data[i] >= 20
    })
    fmt.Printf("20以上の最小インデックス: %d (値=%d)\n", idx, data[idx])
    // 5 (値=23)
}
```

### 8.6 言語別標準ライブラリ比較表

| 言語 | 関数名 | lower_bound | upper_bound | 返り値 |
|:---|:---|:---|:---|:---|
| Python | `bisect.bisect_left()` | 直接対応 | `bisect_right()` | インデックス |
| C++ | `std::lower_bound()` | 直接対応 | `std::upper_bound()` | イテレータ |
| Java | `Arrays.binarySearch()` | 変換が必要 | 変換が必要 | インデックス or 負値 |
| Go | `sort.Search()` | 条件関数で実現 | 条件関数で実現 | インデックス |
| Rust | `slice::binary_search()` | `partition_point()` | `partition_point()` | Result型 |
| JavaScript | なし（自前実装） | - | - | - |
| C# | `Array.BinarySearch()` | 変換が必要 | 変換が必要 | インデックス or 負値 |

