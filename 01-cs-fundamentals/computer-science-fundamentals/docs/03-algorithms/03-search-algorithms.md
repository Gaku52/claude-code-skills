# 探索アルゴリズム — 理論・実装・応用の総合ガイド

> 「データを見つける」ことはコンピューティングの最も基本的な操作であり、探索の効率がシステム全体の性能を左右する。
> 適切な探索アルゴリズムの選択は、応答時間を数桁改善し、ユーザ体験とインフラコストの双方に直結する。

---

## この章で学ぶこと

- [ ] 線形探索の本質と、それが最適解となる条件を正確に理解する
- [ ] 二分探索の基本形・境界探索・応用パターンを自在に実装できる
- [ ] ハッシュベース探索の内部構造と衝突解決を理解し、適材適所で使える
- [ ] グラフ探索（BFS / DFS）の原理と典型的な応用問題を解ける
- [ ] A* 探索の理論的背景を理解し、経路探索問題へ適用できる
- [ ] 各探索アルゴリズムの性能特性を比較し、場面に応じた最適な選択ができる

## 前提知識

- 計算量解析 → 参照: [[01-complexity-analysis.md]]
- 基本的なデータ構造（配列、連結リスト、スタック、キュー） → 参照: 02-data-structures 章
- Python の基本文法（リスト、辞書、クラス定義）

---

## 第1部: 順次探索

---

## 1. 線形探索（Linear Search）

### 1.1 アルゴリズムの本質

線形探索は、データ構造の先頭から末尾まで要素を一つずつ確認していく、最も直感的な探索手法である。前提条件を一切必要とせず、あらゆるデータ構造に適用可能という汎用性を持つ。

計算量は以下の通りである。

| ケース | 時間計算量 | 説明 |
|--------|-----------|------|
| 最良 | O(1) | 先頭要素が対象 |
| 平均 | O(n) | 平均 n/2 回の比較 |
| 最悪 | O(n) | 末尾要素または不在 |
| 空間 | O(1) | 追加メモリ不要 |

### 1.2 基本実装

```python
def linear_search(arr: list, target) -> int:
    """
    線形探索: 先頭から順に要素を走査する。

    Args:
        arr: 探索対象のリスト（ソート不要）
        target: 探索する値

    Returns:
        見つかった場合はインデックス、見つからなければ -1

    計算量: O(n) 時間, O(1) 空間
    """
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1


# --- 使用例 ---
data = [4, 2, 7, 1, 9, 3, 8, 5]

print(linear_search(data, 9))   # => 4  (インデックス4に存在)
print(linear_search(data, 6))   # => -1 (存在しない)
```

### 1.3 バリエーション

#### 1.3.1 番兵法（Sentinel Linear Search）

ループ内で毎回行われる境界チェック `i < len(arr)` を排除し、比較回数を半減させるテクニックである。

```python
def sentinel_linear_search(arr: list, target) -> int:
    """
    番兵法による線形探索。
    配列末尾に target を番兵として追加し、境界チェックを不要にする。

    注意: 元配列を一時的に変更するため、スレッドセーフではない。

    計算量: O(n) 時間（定数倍の改善）, O(1) 空間
    """
    n = len(arr)
    if n == 0:
        return -1

    # 末尾を退避して番兵を設置
    last = arr[n - 1]
    arr[n - 1] = target

    i = 0
    while arr[i] != target:
        i += 1

    # 末尾を復元
    arr[n - 1] = last

    # 最後に見つかった位置が元の末尾かどうかを判別
    if i < n - 1 or arr[n - 1] == target:
        return i
    return -1


# --- 動作の可視化 ---
# arr = [4, 2, 7, 1, 9], target = 7
#
# 番兵設置後: [4, 2, 7, 1, 7]  ← 末尾に7を配置
#
# i=0: arr[0]=4 != 7 → 次へ
# i=1: arr[1]=2 != 7 → 次へ
# i=2: arr[2]=7 == 7 → ループ終了！
#
# i=2 < n-1=4 なので、元の配列で見つかったと判定
# 復元: [4, 2, 7, 1, 9]
# return 2
```

#### 1.3.2 全件検索（Find All）

```python
def find_all(arr: list, target) -> list[int]:
    """
    target と一致する全てのインデックスを返す。

    Returns:
        一致するインデックスのリスト（空リストは不在を意味する）

    計算量: O(n) 時間, O(k) 空間（k = 一致数）
    """
    return [i for i, val in enumerate(arr) if val == target]


# --- 使用例 ---
data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
print(find_all(data, 5))   # => [4, 8, 10]
print(find_all(data, 7))   # => []
```

#### 1.3.3 条件付き探索

```python
def find_first_match(arr: list, predicate) -> int:
    """
    与えられた条件を最初に満たす要素のインデックスを返す。

    Args:
        predicate: 要素を受け取りboolを返す関数

    計算量: O(n) 時間, O(1) 空間
    """
    for i, val in enumerate(arr):
        if predicate(val):
            return i
    return -1


# --- 使用例 ---
students = [
    {"name": "Alice", "score": 72},
    {"name": "Bob", "score": 95},
    {"name": "Charlie", "score": 88},
]
# スコア90以上の最初の学生
idx = find_first_match(students, lambda s: s["score"] >= 90)
print(students[idx]["name"])  # => "Bob"
```

### 1.4 線形探索が最適となる場面

線形探索は「単純だから劣る」というわけではない。以下の場面では最適解となる。

1. **要素数が少ない場合（n < 50 程度）**: 二分探索のオーバーヘッド（ソートの維持、関数呼び出し、分岐予測ミス）が相対的に大きくなり、線形探索の方が高速になることがある
2. **一度きりの検索**: ソートに O(n log n) かかるため、1回の検索なら線形探索の O(n) の方が総コストが低い
3. **データが頻繁に変更される場合**: 挿入・削除のたびにソート状態を維持するコストが探索コストを上回る
4. **連結リストなど非ランダムアクセスなデータ構造**: 二分探索は O(1) のランダムアクセスを前提とするため、連結リストでは使えない

### 1.5 ASCII 図解: 線形探索の動作

```
線形探索: arr = [4, 2, 7, 1, 9, 3, 8, 5], target = 9

Step 1: [4] 2  7  1  9  3  8  5     4 != 9 → 次へ
         ^
Step 2:  4 [2] 7  1  9  3  8  5     2 != 9 → 次へ
            ^
Step 3:  4  2 [7] 1  9  3  8  5     7 != 9 → 次へ
               ^
Step 4:  4  2  7 [1] 9  3  8  5     1 != 9 → 次へ
                  ^
Step 5:  4  2  7  1 [9] 3  8  5     9 == 9 → 発見！ return 4
                     ^

比較回数: 5回（最悪は n=8 回）
```

---

## 第2部: 分割統治型探索

---

## 2. 二分探索（Binary Search）

### 2.1 アルゴリズムの本質

二分探索は、**ソート済み**のデータに対して探索範囲を毎回半分に絞り込むことで、対数時間での探索を実現するアルゴリズムである。データ量が 2 倍になっても比較回数は 1 回増えるだけという驚異的な効率を持つ。

| データ量 n | 二分探索の比較回数 | 線形探索の最悪比較回数 |
|-----------|------------------|---------------------|
| 100 | 7 | 100 |
| 10,000 | 14 | 10,000 |
| 1,000,000 | 20 | 1,000,000 |
| 1,000,000,000 | 30 | 1,000,000,000 |

この表が示す通り、10 億件のデータでもわずか 30 回の比較で探索が完了する。

### 2.2 基本実装（反復版）

```python
def binary_search(arr: list, target) -> int:
    """
    二分探索: ソート済み配列で target を探す。

    Args:
        arr: ソート済みリスト（昇順）
        target: 探索する値

    Returns:
        見つかった場合はインデックス、見つからなければ -1

    計算量: O(log n) 時間, O(1) 空間
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = left + (right - left) // 2  # オーバーフロー防止の定石

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
```

### 2.3 ASCII 図解: 二分探索の動作

```
二分探索: arr = [1, 3, 5, 7, 9, 11, 13, 15, 17], target = 11

Step 1: left=0, right=8
        [1  3  5  7 [9] 11  13  15  17]
                     ^mid=4
        arr[4]=9 < 11 → left = 5

Step 2: left=5, right=8
         1  3  5  7  9 [11  13 [15] 17]
                                ^mid=6...
        wait: mid = 5 + (8-5)//2 = 6
        arr[6]=13 > 11 → right = 5

Step 3: left=5, right=5
         1  3  5  7  9 [11] 13  15  17
                         ^mid=5
        arr[5]=11 == 11 → 発見！ return 5

比較回数: 3回（log2(9) ≈ 3.17）

探索範囲の縮小:
  Step 1: |■■■■■■■■■| 9要素
  Step 2: |    ■■■■ | 4要素
  Step 3: |    ■    | 1要素
```

### 2.4 再帰版の実装

```python
def binary_search_recursive(arr: list, target, left: int = 0,
                             right: int = None) -> int:
    """
    二分探索の再帰版。

    計算量: O(log n) 時間, O(log n) 空間（コールスタック）
    """
    if right is None:
        right = len(arr) - 1

    if left > right:
        return -1

    mid = left + (right - left) // 2

    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)
```

> **反復版 vs 再帰版**: 反復版は O(1) の空間計算量で済むため、実務では反復版が推奨される。再帰版はスタックオーバーフローのリスクがあり、Python のデフォルトの再帰制限（1000）に達する可能性がある。ただし再帰版はコードの可読性に優れ、アルゴリズムの本質が伝わりやすい。

### 2.5 よくあるバグと対策（Top 5）

二分探索は「正しく実装できるプログラマは全体の 10% に満たない」と言われるほど、バグを埋め込みやすいアルゴリズムである（Jon Bentley, "Programming Pearls"）。

```python
# ================================================================
# バグ1: 中間値計算のオーバーフロー
# ================================================================
# C/C++/Java などの固定長整数で問題になる。Python は多倍長整数
# なので実害はないが、他言語への移植を考えて安全な形を使う。

mid = (left + right) // 2           # 非推奨: left + right がオーバーフローする可能性
mid = left + (right - left) // 2    # 推奨: 安全な計算

# ================================================================
# バグ2: 無限ループ
# ================================================================
# left = 3, right = 4, mid = 3 のとき:
#   arr[mid] < target → left = mid   ← 進まない！無限ループ
#   正しくは: left = mid + 1

# ================================================================
# バグ3: off-by-one エラー（ループ条件）
# ================================================================
# while left < right   → 探索範囲に left == right の場合を含まない
# while left <= right  → 探索範囲に left == right の場合を含む
# どちらを使うかで、更新式とセットで正しさが変わる

# ================================================================
# バグ4: 右端の初期値
# ================================================================
# right = len(arr)     → 範囲外アクセスの可能性
# right = len(arr) - 1 → 完全一致検索ではこちら
# right = len(arr)     → lower_bound / upper_bound ではこちら（半開区間）

# ================================================================
# バグ5: 等号の処理ミス
# ================================================================
# arr[mid] < target  vs  arr[mid] <= target
# lower_bound と upper_bound の違いはここだけ！
# 間違えると、重複要素がある場合に結果が異なる
```

### 2.6 二分探索の 2 大テンプレート

実務で二分探索を正確に実装するためには、「完全一致テンプレート」と「境界探索テンプレート」の 2 つを習得するのが効果的である。

```python
# ================================================================
# テンプレートA: 完全一致探索
# ================================================================
# 用途: target と等しい要素のインデックスを返す
# ループ条件: while left <= right
# 更新: left = mid + 1, right = mid - 1
# 初期値: left = 0, right = len(arr) - 1
# 終了: target が見つかれば mid を返す。見つからなければ -1

def search_exact(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# ================================================================
# テンプレートB: 境界探索（lower_bound / upper_bound）
# ================================================================
# 用途: 条件を満たす最小（または最大）のインデックスを返す
# ループ条件: while left < right
# 更新: left = mid + 1 または right = mid（方向による）
# 初期値: left = 0, right = len(arr)（半開区間 [left, right) ）
# 終了: left == right となった位置が答え
```

### 2.7 境界探索の実装

```python
import bisect


def lower_bound(arr: list, target) -> int:
    """
    target 以上の最小のインデックスを返す（leftmost insertion point）。

    全要素が target 未満の場合は len(arr) を返す。
    Python 標準ライブラリでは bisect.bisect_left に相当する。

    例: arr = [1, 3, 3, 3, 5, 7], target = 3
        → return 1（最初の 3 のインデックス）

    計算量: O(log n) 時間, O(1) 空間
    """
    left, right = 0, len(arr)
    while left < right:
        mid = left + (right - left) // 2
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left


def upper_bound(arr: list, target) -> int:
    """
    target より大きい最小のインデックスを返す（rightmost insertion point）。

    全要素が target 以下の場合は len(arr) を返す。
    Python 標準ライブラリでは bisect.bisect_right に相当する。

    例: arr = [1, 3, 3, 3, 5, 7], target = 3
        → return 4（最後の 3 の次のインデックス）

    計算量: O(log n) 時間, O(1) 空間
    """
    left, right = 0, len(arr)
    while left < right:
        mid = left + (right - left) // 2
        if arr[mid] <= target:    # ← lower_bound との唯一の違い
            left = mid + 1
        else:
            right = mid
    return left


# --- lower_bound と upper_bound の関係 ---
# arr = [1, 3, 3, 3, 5, 7]
#
#  idx:  0  1  2  3  4  5
#  val:  1  3  3  3  5  7
#           ^        ^
#           |        |
#     lower_bound=1  upper_bound=4
#
# target=3 の個数 = upper_bound - lower_bound = 4 - 1 = 3

def count_occurrences(arr: list, target) -> int:
    """ソート済み配列で target の出現回数を O(log n) で数える。"""
    return upper_bound(arr, target) - lower_bound(arr, target)


# --- bisect モジュールの活用 ---
arr = [1, 3, 3, 3, 5, 7]

# lower_bound
print(bisect.bisect_left(arr, 3))   # => 1

# upper_bound
print(bisect.bisect_right(arr, 3))  # => 4

# target の出現回数
print(bisect.bisect_right(arr, 3) - bisect.bisect_left(arr, 3))  # => 3

# ソート順を維持した挿入
bisect.insort(arr, 4)
print(arr)  # => [1, 3, 3, 3, 4, 5, 7]
```

### 2.8 二分探索の応用パターン

#### 2.8.1 答えで二分探索（Binary Search on Answer）

「条件を満たす最小（または最大）の値は何か？」という問題に対して、答えの候補空間を二分探索する強力なテクニックである。

適用条件は次の通りである。
- 答えが連続的（または離散的だが順序付き）な空間に存在する
- 「答えが x 以上のとき条件を満たすか？」の判定が単調（あるしきい値を境に Yes/No が切り替わる）

```python
def max_rope_length(ropes: list[int], k: int) -> int:
    """
    n 本のロープを切って k 本以上の同じ長さのロープを作る。
    最大で何 cm の長さにできるか？

    例: ropes = [802, 743, 457, 539], k = 11
    → answer = 200（200cm のロープが合計 11 本取れる）

    アプローチ:
    - 答え（ロープの長さ）の範囲は [1, max(ropes)]
    - ある長さ L で切ったとき k 本以上取れるかを判定
    - 判定関数は単調減少（L が大きいほど取れる本数は減る）
    - → 条件を満たす最大の L を二分探索

    計算量: O(n * log(max(ropes))) 時間, O(1) 空間
    """
    def can_cut(length: int) -> bool:
        """長さ length で切ったとき k 本以上取れるか？"""
        return sum(r // length for r in ropes) >= k

    left, right = 1, max(ropes)
    result = 0

    while left <= right:
        mid = left + (right - left) // 2
        if can_cut(mid):
            result = mid        # 条件を満たすので、もっと大きくできるか試す
            left = mid + 1
        else:
            right = mid - 1     # 条件を満たさないので、小さくする

    return result


# --- 動作例 ---
ropes = [802, 743, 457, 539]
print(max_rope_length(ropes, 11))  # => 200

# 判定の様子:
# L=401: 802//401 + 743//401 + 457//401 + 539//401 = 2+1+1+1 = 5 < 11 → NG
# L=200: 802//200 + 743//200 + 457//200 + 539//200 = 4+3+2+2 = 11 >= 11 → OK
# L=201: 802//201 + 743//201 + 457//201 + 539//201 = 3+3+2+2 = 10 < 11 → NG
# → answer = 200
```

#### 2.8.2 回転ソート済み配列での二分探索

```python
def search_rotated(arr: list, target) -> int:
    """
    回転ソート済み配列で target を探す。

    回転ソート済み配列とは、ソート済み配列をある位置で分割し、
    前後を入れ替えたもの。
    例: [0,1,2,4,5,6,7] → [4,5,6,7,0,1,2]（インデックス3で回転）

    ポイント: mid で分割すると、少なくとも片方はソート済み。
    ソート済みの方で target が範囲内にあるかを判定する。

    計算量: O(log n) 時間, O(1) 空間
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if arr[mid] == target:
            return mid

        # 左半分がソート済みかを判定
        if arr[left] <= arr[mid]:
            # target が左半分の範囲内にあるか
            if arr[left] <= target < arr[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            # 右半分がソート済み
            # target が右半分の範囲内にあるか
            if arr[mid] < target <= arr[right]:
                left = mid + 1
            else:
                right = mid - 1

    return -1


# --- 動作例 ---
# arr = [4, 5, 6, 7, 0, 1, 2], target = 0
#
# Step 1: left=0, right=6, mid=3
#   arr[0]=4 <= arr[3]=7 → 左半分 [4,5,6,7] はソート済み
#   4 <= 0 < 7 ? No → target は右半分にある → left = 4
#
# Step 2: left=4, right=6, mid=5
#   arr[4]=0 <= arr[5]=1 → 左半分 [0,1] はソート済み
#   0 <= 0 < 1 ? Yes → right = 4
#
# Step 3: left=4, right=4, mid=4
#   arr[4]=0 == 0 → 発見！ return 4
```

#### 2.8.3 ピーク要素の探索

```python
def find_peak_element(arr: list) -> int:
    """
    配列内のピーク要素（両隣より大きい要素）のインデックスを返す。

    arr[-1] = arr[n] = -∞ と仮定する。
    ピーク要素が複数ある場合、どれか1つを返せばよい。

    ポイント: arr[mid] < arr[mid+1] なら右側にピークが存在する。
    （上り坂の先には必ず山頂がある）

    計算量: O(log n) 時間, O(1) 空間
    """
    left, right = 0, len(arr) - 1

    while left < right:
        mid = left + (right - left) // 2
        if arr[mid] < arr[mid + 1]:
            left = mid + 1     # 右側にピークがある
        else:
            right = mid        # 左側（mid含む）にピークがある

    return left


# --- 動作例 ---
print(find_peak_element([1, 3, 5, 4, 2]))  # => 2（arr[2]=5がピーク）
print(find_peak_element([1, 2, 3, 4, 5]))  # => 4（単調増加なら末尾）
```

#### 2.8.4 平方根の二分探索

```python
def integer_sqrt(n: int) -> int:
    """
    非負整数 n の整数平方根を求める（切り捨て）。

    x * x <= n を満たす最大の x を二分探索で求める。

    計算量: O(log n) 時間, O(1) 空間
    """
    if n < 2:
        return n

    left, right = 1, n // 2
    result = 1

    while left <= right:
        mid = left + (right - left) // 2
        if mid * mid == n:
            return mid
        elif mid * mid < n:
            result = mid
            left = mid + 1
        else:
            right = mid - 1

    return result


# --- 検証 ---
print(integer_sqrt(16))   # => 4
print(integer_sqrt(27))   # => 5（5*5=25 <= 27 < 36=6*6）
print(integer_sqrt(100))  # => 10
```

### 2.9 二分探索の判定フローチャート

```
二分探索を使うべきか？ — 判定フローチャート

問題を確認
    │
    ├── データはソート済み or ソート可能か？
    │       │
    │       ├── Yes ─→ 完全一致検索？
    │       │            │
    │       │            ├── Yes ─→ テンプレートA（完全一致）を使用
    │       │            │
    │       │            └── No ──→ 境界値や範囲の探索？
    │       │                          │
    │       │                          ├── Yes ─→ テンプレートB（lower/upper_bound）を使用
    │       │                          │
    │       │                          └── No ──→ 他のアプローチを検討
    │       │
    │       └── No ──→ 答えの候補空間は単調か？
    │                    │
    │                    ├── Yes ─→ 「答えで二分探索」パターンを使用
    │                    │
    │                    └── No ──→ 二分探索は不適。他の手法を検討
    │
    └── 特殊構造か？（回転配列、山型配列など）
             │
             ├── Yes ─→ 構造に応じた変形二分探索を使用
             │
             └── No ──→ 線形探索 or グラフ探索を検討
```

---

## 第3部: ハッシュベース探索

---

## 3. ハッシュベース探索

### 3.1 ハッシュテーブルの原理

ハッシュテーブルは、**ハッシュ関数**を用いてキーを配列のインデックスに変換することで、平均 O(1) の探索・挿入・削除を実現するデータ構造である。

#### ハッシュ関数の要件

| 要件 | 説明 | 違反した場合 |
|------|------|------------|
| 決定的 | 同じ入力に対して常に同じ出力 | 探索結果が不安定になる |
| 均一分布 | 出力値が偏らず均等に分布 | 特定のスロットに集中し性能劣化 |
| 高速計算 | ハッシュ値の計算コストが低い | 探索の O(1) の意味がなくなる |

### 3.2 ASCII 図解: ハッシュテーブルの構造

```
ハッシュテーブル（チェイニング方式）

  キー "apple"  →  hash("apple") % 8 = 3
  キー "banana" →  hash("banana") % 8 = 6
  キー "cherry" →  hash("cherry") % 8 = 3  ← 衝突！
  キー "date"   →  hash("date") % 8 = 1

  バケット配列（サイズ 8）:
  ┌─────┬───────────────────────────────────┐
  │  0  │ (empty)                           │
  ├─────┼───────────────────────────────────┤
  │  1  │ → ["date", 値] → None            │
  ├─────┼───────────────────────────────────┤
  │  2  │ (empty)                           │
  ├─────┼───────────────────────────────────┤
  │  3  │ → ["apple", 値] → ["cherry", 値] → None  ← チェイン
  ├─────┼───────────────────────────────────┤
  │  4  │ (empty)                           │
  ├─────┼───────────────────────────────────┤
  │  5  │ (empty)                           │
  ├─────┼───────────────────────────────────┤
  │  6  │ → ["banana", 値] → None          │
  ├─────┼───────────────────────────────────┤
  │  7  │ (empty)                           │
  └─────┴───────────────────────────────────┘

  探索 "cherry":
  1. hash("cherry") % 8 = 3
  2. バケット 3 のチェインを走査
  3. "apple" != "cherry" → 次へ
  4. "cherry" == "cherry" → 発見！
```

### 3.3 衝突解決法の比較

ハッシュ関数がどれだけ優秀でも、鳩の巣原理により衝突は避けられない。衝突解決には大きく 2 つの手法がある。

```
衝突解決法の比較

  ┌──────────────────┬─────────────────────┬─────────────────────┐
  │ 特性             │ チェイニング         │ オープンアドレス法   │
  ├──────────────────┼─────────────────────┼─────────────────────┤
  │ 衝突時の処理     │ リンクリストに追加   │ 別のスロットを探す   │
  │ メモリ効率       │ ポインタ分のオーバー │ テーブル内で完結     │
  │                  │ ヘッドがある         │                     │
  │ キャッシュ効率   │ 低い（ポインタ追跡） │ 高い（連続メモリ）   │
  │ ロードファクター │ 1.0 超も可能         │ 1.0 未満が必須       │
  │ 削除の容易さ     │ 容易                 │ 複雑（墓標が必要）   │
  │ 最悪ケース       │ O(n)（全衝突時）     │ O(n)（全衝突時）     │
  │ 採用例           │ Java HashMap         │ Python dict          │
  └──────────────────┴─────────────────────┴─────────────────────┘

  オープンアドレス法の探索方式:
  - 線形探索法: h(k)+1, h(k)+2, h(k)+3, ...
    → クラスタリングが発生しやすい
  - 二次探索法: h(k)+1^2, h(k)+2^2, h(k)+3^2, ...
    → 一次クラスタリングを軽減
  - ダブルハッシング: h(k)+i*h2(k) で別のハッシュ関数を利用
    → 最も均一だが計算コストが高い
```

### 3.4 ハッシュテーブルの実装

```python
class HashTable:
    """
    チェイニング方式のハッシュテーブル実装。

    特徴:
    - 動的リサイズ（ロードファクター 0.75 超で 2 倍に拡張）
    - 挿入・探索・削除をサポート
    """

    def __init__(self, initial_capacity: int = 16):
        self._capacity = initial_capacity
        self._size = 0
        self._buckets: list[list] = [[] for _ in range(self._capacity)]
        self._load_factor_threshold = 0.75

    def _hash(self, key) -> int:
        """キーのハッシュ値をバケットインデックスに変換する。"""
        return hash(key) % self._capacity

    def put(self, key, value) -> None:
        """キーと値のペアを挿入する。既存キーは値を更新する。"""
        if self._size / self._capacity > self._load_factor_threshold:
            self._resize()

        idx = self._hash(key)
        bucket = self._buckets[idx]

        # 既存キーの更新チェック
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return

        # 新規挿入
        bucket.append((key, value))
        self._size += 1

    def get(self, key, default=None):
        """キーに対応する値を返す。存在しなければ default を返す。"""
        idx = self._hash(key)
        for k, v in self._buckets[idx]:
            if k == key:
                return v
        return default

    def remove(self, key) -> bool:
        """キーを削除する。削除できたら True を返す。"""
        idx = self._hash(key)
        bucket = self._buckets[idx]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket.pop(i)
                self._size -= 1
                return True
        return False

    def __contains__(self, key) -> bool:
        """'key in table' をサポートする。"""
        idx = self._hash(key)
        return any(k == key for k, _ in self._buckets[idx])

    def __len__(self) -> int:
        return self._size

    def _resize(self) -> None:
        """バケット数を 2 倍に拡張し、全要素を再配置する。"""
        old_buckets = self._buckets
        self._capacity *= 2
        self._buckets = [[] for _ in range(self._capacity)]
        self._size = 0

        for bucket in old_buckets:
            for key, value in bucket:
                self.put(key, value)


# --- 使用例 ---
ht = HashTable()
ht.put("name", "Alice")
ht.put("age", 30)
ht.put("city", "Tokyo")

print(ht.get("name"))     # => "Alice"
print(ht.get("age"))      # => 30
print("city" in ht)        # => True
print(len(ht))             # => 3

ht.remove("age")
print(ht.get("age"))       # => None
print(len(ht))             # => 2
```

### 3.5 Python における dict / set の内部実装

Python の `dict` は、オープンアドレス法を採用した高度に最適化されたハッシュテーブルである。

| 特性 | Python dict の実装 |
|------|-------------------|
| 衝突解決 | オープンアドレス法（ランダム探索に近い） |
| ロードファクター | 2/3（約 66.7%）でリハッシュ |
| ハッシュ関数 | SipHash（Hash DoS 対策） |
| メモリレイアウト | Compact dict（Python 3.6+、挿入順序を保持） |
| 初期サイズ | 8 スロット |
| 拡張係数 | 3 倍（使用中のスロット数 > 2/3 * 容量で拡張） |

```python
# Python dict/set の計算量

# dict
d = {}
d[key] = value       # 挿入:     平均 O(1), 最悪 O(n)
value = d[key]        # 探索:     平均 O(1), 最悪 O(n)
del d[key]            # 削除:     平均 O(1), 最悪 O(n)
key in d              # 存在確認: 平均 O(1), 最悪 O(n)

# set
s = set()
s.add(elem)           # 追加:     平均 O(1)
elem in s             # 存在確認: 平均 O(1)
s.remove(elem)        # 削除:     平均 O(1)
s1 & s2               # 積集合:   O(min(len(s1), len(s2)))
s1 | s2               # 和集合:   O(len(s1) + len(s2))

# frozenset はハッシュ可能 → dict のキーや set の要素にできる
fs = frozenset([1, 2, 3])
d = {fs: "value"}     # OK
```

### 3.6 二分探索 vs ハッシュテーブル — 使い分け基準

```
探索手法の選択ガイド

  ┌────────────────────────┬──────────────┬──────────────────┐
  │ 比較項目               │ 二分探索      │ ハッシュテーブル  │
  ├────────────────────────┼──────────────┼──────────────────┤
  │ 平均探索時間           │ O(log n)     │ O(1)             │
  │ 最悪探索時間           │ O(log n)     │ O(n)             │
  │ 前提条件               │ ソート済み   │ ハッシュ関数      │
  │ 追加メモリ             │ O(1)         │ O(n)             │
  │ 範囲検索               │ 効率的       │ 全探索が必要     │
  │ 順序付き列挙           │ 容易         │ 不可能           │
  │ キャッシュ効率         │ 良好         │ 劣る             │
  │ 最悪ケースの予測可能性 │ 高い         │ 低い             │
  │ 実装の複雑さ           │ 中程度       │ 低い             │
  │ 動的なデータ更新       │ ソート維持が │ O(1) で更新可能  │
  │                        │ 必要（高コスト）│                │
  └────────────────────────┴──────────────┴──────────────────┘

  選択指針:
  ✓ 完全一致のみ & 大量データ → ハッシュテーブル
  ✓ 範囲検索が必要 → 二分探索 or B-Tree
  ✓ メモリ制約が厳しい → 二分探索（配列上で実施、追加メモリ O(1)）
  ✓ 最悪ケース保証が必要 → 二分探索
  ✓ データが頻繁に変更される → ハッシュテーブル
  ✓ 順序付きの列挙が必要 → ソート済み配列 + 二分探索
```

---

## 第4部: グラフ探索

---

## 4. 幅優先探索（BFS: Breadth-First Search）

### 4.1 アルゴリズムの本質

BFS は、始点から近い頂点から順に探索する手法であり、**キュー**を用いて実装される。重み無しグラフにおける最短経路問題を解く基本的なアルゴリズムである。

主な特性は以下の通りである。

| 特性 | 値 |
|------|-----|
| 時間計算量 | O(V + E)（V: 頂点数, E: 辺数） |
| 空間計算量 | O(V)（キュー + 訪問済み集合） |
| 最短経路 | 重み無しグラフで保証 |
| 完全性 | 有限グラフでは解が存在すれば必ず見つかる |

### 4.2 基本実装

```python
from collections import deque


def bfs(graph: dict[str, list[str]], start: str) -> list[str]:
    """
    幅優先探索: 始点から近い順に全頂点を訪問する。

    Args:
        graph: 隣接リスト表現のグラフ
        start: 始点のノード

    Returns:
        訪問順のノードリスト

    計算量: O(V + E) 時間, O(V) 空間
    """
    visited = set([start])
    queue = deque([start])
    order = []

    while queue:
        node = queue.popleft()       # キューの先頭を取り出す
        order.append(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return order


# --- 使用例 ---
graph = {
    "A": ["B", "C"],
    "B": ["A", "D", "E"],
    "C": ["A", "F"],
    "D": ["B"],
    "E": ["B", "F"],
    "F": ["C", "E"],
}

print(bfs(graph, "A"))
# => ['A', 'B', 'C', 'D', 'E', 'F']
```

### 4.3 ASCII 図解: BFS の動作

```
グラフ構造:
        A
       / \
      B   C
     / \   \
    D   E - F

BFS の訪問順（始点: A）:

  レベル 0:  [A]
             ↓ A の隣接: B, C
  レベル 1:  [B, C]
             ↓ B の隣接: D, E  /  C の隣接: F
  レベル 2:  [D, E, F]

  キューの変化:
  初期:      Queue=[A],     Visited={A}
  Step 1:    Queue=[B,C],   Visited={A,B,C}     ← A を取り出し
  Step 2:    Queue=[C,D,E], Visited={A,B,C,D,E} ← B を取り出し
  Step 3:    Queue=[D,E,F], Visited={A,B,C,D,E,F} ← C を取り出し
  Step 4:    Queue=[E,F],   Visited={A,B,C,D,E,F} ← D を取り出し
  Step 5:    Queue=[F],     Visited={A,B,C,D,E,F} ← E を取り出し
  Step 6:    Queue=[],      Visited={A,B,C,D,E,F} ← F を取り出し

  訪問順: A → B → C → D → E → F
```

### 4.4 BFS による最短経路の復元

```python
from collections import deque


def bfs_shortest_path(graph: dict, start: str,
                      goal: str) -> list[str] | None:
    """
    重み無しグラフにおける最短経路を BFS で求める。

    Returns:
        最短経路のノードリスト。到達不能なら None。

    計算量: O(V + E) 時間, O(V) 空間
    """
    if start == goal:
        return [start]

    visited = set([start])
    queue = deque([(start, [start])])   # (現在のノード, 経路)

    while queue:
        node, path = queue.popleft()

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                new_path = path + [neighbor]

                if neighbor == goal:
                    return new_path

                visited.add(neighbor)
                queue.append((neighbor, new_path))

    return None  # 到達不能


# --- 使用例 ---
graph = {
    "A": ["B", "C"],
    "B": ["A", "D", "E"],
    "C": ["A", "F"],
    "D": ["B"],
    "E": ["B", "F"],
    "F": ["C", "E"],
}

print(bfs_shortest_path(graph, "A", "F"))  # => ['A', 'C', 'F']
print(bfs_shortest_path(graph, "D", "F"))  # => ['D', 'B', 'E', 'F']
```

### 4.5 BFS の典型的な応用

| 応用 | 説明 |
|------|------|
| 最短経路（重み無し） | 辺の重みが全て同じグラフでの最短経路 |
| レベル順走査 | 木構造のレベル（深さ）ごとの走査 |
| 連結成分の検出 | グラフ内の連結成分を列挙 |
| 二部グラフ判定 | 2 色で塗り分けられるかを判定 |
| 迷路の最短解 | グリッド上の最短経路問題 |
| ソーシャルグラフ | 「友達の友達」の距離計算 |

```python
from collections import deque


def shortest_path_in_maze(maze: list[list[int]],
                          start: tuple[int, int],
                          goal: tuple[int, int]) -> int:
    """
    2Dグリッド迷路の最短経路長を BFS で求める。

    maze[r][c] = 0: 通行可能, 1: 壁
    上下左右の 4 方向に移動可能。

    Returns:
        最短ステップ数。到達不能なら -1。
    """
    rows, cols = len(maze), len(maze[0])
    if maze[start[0]][start[1]] == 1 or maze[goal[0]][goal[1]] == 1:
        return -1

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    visited = set([start])
    queue = deque([(start[0], start[1], 0)])  # (行, 列, ステップ数)

    while queue:
        r, c, steps = queue.popleft()

        if (r, c) == goal:
            return steps

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols
                    and maze[nr][nc] == 0
                    and (nr, nc) not in visited):
                visited.add((nr, nc))
                queue.append((nr, nc, steps + 1))

    return -1


# --- 使用例 ---
maze = [
    [0, 0, 0, 0, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
]
print(shortest_path_in_maze(maze, (0, 0), (4, 4)))  # => 8
```

---

## 5. 深さ優先探索（DFS: Depth-First Search）

### 5.1 アルゴリズムの本質

DFS は、一つの方向をできるだけ深く探索し、行き止まりに達したらバックトラックして別の方向を探索する手法である。**スタック**（または再帰呼び出し）を用いて実装される。

| 特性 | 値 |
|------|-----|
| 時間計算量 | O(V + E) |
| 空間計算量 | O(V)（再帰: コールスタック、反復: 明示的スタック） |
| 最短経路 | **保証しない** |
| 完全性 | 有限グラフでは保証（無限グラフでは保証しない） |

### 5.2 再帰版と反復版の実装

```python
def dfs_recursive(graph: dict[str, list[str]], start: str,
                  visited: set = None) -> list[str]:
    """
    深さ優先探索（再帰版）。

    計算量: O(V + E) 時間, O(V) 空間（コールスタック）
    """
    if visited is None:
        visited = set()

    visited.add(start)
    order = [start]

    for neighbor in graph[start]:
        if neighbor not in visited:
            order.extend(dfs_recursive(graph, neighbor, visited))

    return order


def dfs_iterative(graph: dict[str, list[str]], start: str) -> list[str]:
    """
    深さ優先探索（反復版）。
    明示的なスタックを使い、再帰によるスタックオーバーフローを回避する。

    計算量: O(V + E) 時間, O(V) 空間
    """
    visited = set()
    stack = [start]
    order = []

    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)

        # 隣接ノードを逆順にスタックに追加
        # （元の順序で探索するため）
        for neighbor in reversed(graph[node]):
            if neighbor not in visited:
                stack.append(neighbor)

    return order


# --- 使用例 ---
graph = {
    "A": ["B", "C"],
    "B": ["A", "D", "E"],
    "C": ["A", "F"],
    "D": ["B"],
    "E": ["B", "F"],
    "F": ["C", "E"],
}

print(dfs_recursive(graph, "A"))   # => ['A', 'B', 'D', 'E', 'F', 'C']
print(dfs_iterative(graph, "A"))   # => ['A', 'B', 'D', 'E', 'F', 'C']
```

### 5.3 ASCII 図解: DFS の動作

```
グラフ構造:
        A
       / \
      B   C
     / \   \
    D   E - F

DFS の訪問順（始点: A、再帰版）:

  Stack の変化（再帰コールスタック）:

  ① visit(A)  →  Stack: [A]
  ② visit(B)  →  Stack: [A, B]       ← A → B
  ③ visit(D)  →  Stack: [A, B, D]    ← B → D
     D は末端 → バックトラック
  ④ visit(E)  →  Stack: [A, B, E]    ← B → E
  ⑤ visit(F)  →  Stack: [A, B, E, F] ← E → F
     F の隣接 C は未訪問だが...
  ⑥ visit(C)  →  Stack: [A, B, E, F, C] ← F → C
     C の全隣接（A, F）は訪問済み → バックトラック

  訪問順: A → B → D → E → F → C

  BFS との比較:
  BFS: A → B → C → D → E → F （横に広がる）
  DFS: A → B → D → E → F → C （縦に深く潜る）
```

### 5.4 DFS の典型的な応用

| 応用 | 説明 |
|------|------|
| トポロジカルソート | DAG（有向非巡回グラフ）のノードを依存順に並べる |
| サイクル検出 | グラフ内の閉路の有無を判定 |
| 連結成分の検出 | 無向グラフの連結成分を列挙 |
| 強連結成分（SCC） | 有向グラフの強連結成分を分解（Tarjan / Kosaraju） |
| バックトラッキング | N-Queens、数独、パズルなどの組合せ探索 |
| 経路の全列挙 | 始点から終点への全経路を列挙 |

#### 5.4.1 トポロジカルソート

```python
def topological_sort(graph: dict[str, list[str]]) -> list[str]:
    """
    有向非巡回グラフ（DAG）のトポロジカルソートを DFS で求める。

    全てのエッジ (u, v) について、結果リスト内で u が v より前に来る。

    用途:
    - ビルドシステムのタスク依存解決
    - パッケージマネージャの依存関係解決
    - コンパイラの命令スケジューリング

    計算量: O(V + E) 時間, O(V) 空間
    """
    visited = set()
    result = []

    def dfs(node):
        visited.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor)
        result.append(node)  # 帰りがけ順で追加

    for node in graph:
        if node not in visited:
            dfs(node)

    return result[::-1]  # 逆順がトポロジカル順序


# --- 使用例: コンパイル依存関係 ---
# A → B, A → C, B → D, C → D の順に依存
dependencies = {
    "main.c":    ["utils.h", "math.h"],
    "utils.h":   ["types.h"],
    "math.h":    ["types.h"],
    "types.h":   [],
}

order = topological_sort(dependencies)
print(order)
# => ['types.h', 'utils.h', 'math.h', 'main.c']
# types.h を最初にコンパイルすべき
```

#### 5.4.2 サイクル検出

```python
def has_cycle(graph: dict[str, list[str]]) -> bool:
    """
    有向グラフにサイクルが存在するかを DFS で判定する。

    3 色マーキング:
    - WHITE (未訪問): まだ訪問していない
    - GRAY  (探索中): 現在の DFS パス上にある
    - BLACK (完了):   全ての子孫を探索済み

    GRAY のノードに再訪問したらサイクルが存在する。

    計算量: O(V + E) 時間, O(V) 空間
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in graph}

    def dfs(node) -> bool:
        color[node] = GRAY
        for neighbor in graph.get(node, []):
            if color.get(neighbor, WHITE) == GRAY:
                return True    # サイクル検出！
            if color.get(neighbor, WHITE) == WHITE:
                if dfs(neighbor):
                    return True
        color[node] = BLACK
        return False

    for node in graph:
        if color[node] == WHITE:
            if dfs(node):
                return True
    return False


# --- 使用例 ---
# サイクルあり: A → B → C → A
graph_with_cycle = {"A": ["B"], "B": ["C"], "C": ["A"]}
print(has_cycle(graph_with_cycle))  # => True

# サイクルなし（DAG）
dag = {"A": ["B", "C"], "B": ["D"], "C": ["D"], "D": []}
print(has_cycle(dag))  # => False
```

### 5.5 BFS vs DFS — 比較表

```
BFS と DFS の比較

  ┌──────────────────────┬────────────────────┬────────────────────┐
  │ 特性                 │ BFS                │ DFS                │
  ├──────────────────────┼────────────────────┼────────────────────┤
  │ データ構造           │ キュー（FIFO）      │ スタック（LIFO）    │
  │ 探索順序             │ レベル順（横に広く）│ 深さ優先（縦に深く）│
  │ 最短経路（重み無し） │ 保証する           │ 保証しない         │
  │ 空間計算量           │ O(b^d)             │ O(b*d)             │
  │                      │ b=分岐数, d=深さ   │                    │
  │ メモリ使用量         │ 多い（全レベル保持）│ 少ない             │
  │ 完全性               │ 有限グラフで保証   │ 有限グラフで保証   │
  │ 最適性               │ 均一コストで最適   │ 非最適             │
  ├──────────────────────┼────────────────────┼────────────────────┤
  │ 向いている問題       │ 最短経路           │ トポロジカルソート │
  │                      │ レベル順走査       │ サイクル検出       │
  │                      │ 二部グラフ判定     │ バックトラッキング │
  │                      │ 迷路の最短解       │ 強連結成分         │
  │                      │ ネットワーク距離   │ パズル解法         │
  └──────────────────────┴────────────────────┴────────────────────┘

  選択指針:
  ✓ 最短経路が必要 → BFS
  ✓ メモリを節約したい → DFS
  ✓ 全解の列挙 → DFS（バックトラッキング）
  ✓ 依存関係の解決 → DFS（トポロジカルソート）
  ✓ 到達可能性のみ → どちらでも可（DFS の方が実装が簡潔）
```

---

## 第5部: ヒューリスティック探索

---

## 6. A* 探索（A-star Search）

### 6.1 アルゴリズムの本質

A* は、ダイクストラ法と貪欲最良優先探索を組み合わせた、**最適な経路**を効率的に見つけるアルゴリズムである。評価関数 f(n) = g(n) + h(n) を用いて、最も有望なノードを優先的に探索する。

| 記号 | 意味 |
|------|------|
| g(n) | 始点から現在のノード n までの実コスト |
| h(n) | ノード n からゴールまでの**推定コスト**（ヒューリスティック関数） |
| f(n) | g(n) + h(n) = 総推定コスト |

#### ヒューリスティック関数の条件

A* が最適解を保証するには、h(n) が**許容的（admissible）**である必要がある。すなわち、h(n) は真のコストを超えてはならない（h(n) <= 実際のコスト）。

代表的なヒューリスティック関数は次の通りである。

| 関数名 | 定義 | 移動方向 | 用途 |
|--------|------|---------|------|
| マンハッタン距離 | \|x1-x2\| + \|y1-y2\| | 4方向（上下左右） | グリッドマップ |
| ユークリッド距離 | sqrt((x1-x2)^2 + (y1-y2)^2) | 任意方向 | 自由移動マップ |
| チェビシェフ距離 | max(\|x1-x2\|, \|y1-y2\|) | 8方向（斜め含む） | チェス盤 |
| ゼロ関数 | 0 | - | ダイクストラ法に退化 |

### 6.2 実装

```python
import heapq
from typing import Callable


def a_star(graph: dict, start, goal,
           h: Callable, get_neighbors: Callable) -> tuple[list, float]:
    """
    A* 探索アルゴリズム。

    Args:
        graph: グラフデータ（get_neighbors の実装に依存）
        start: 始点ノード
        goal: 目標ノード
        h: ヒューリスティック関数 h(node, goal) -> float
        get_neighbors: 隣接ノードとコストを返す関数
                       get_neighbors(graph, node) -> [(neighbor, cost), ...]

    Returns:
        (経路リスト, 総コスト) のタプル。到達不能なら ([], float('inf'))

    計算量: 最悪 O(b^d) 時間・空間（b=分岐数, d=深さ）
            良いヒューリスティックで大幅に削減される
    """
    # 優先度キュー: (f値, ノード)
    open_set = [(h(start, goal), 0, start)]  # (f, g, node)
    came_from = {}
    g_score = {start: 0}
    closed_set = set()

    while open_set:
        f, g, current = heapq.heappop(open_set)

        if current == goal:
            # 経路を復元
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1], g

        if current in closed_set:
            continue
        closed_set.add(current)

        for neighbor, cost in get_neighbors(graph, current):
            if neighbor in closed_set:
                continue

            tentative_g = g + cost

            if tentative_g < g_score.get(neighbor, float('inf')):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                f_score = tentative_g + h(neighbor, goal)
                heapq.heappush(open_set, (f_score, tentative_g, neighbor))

    return [], float('inf')  # 到達不能


# --- グリッドマップでの使用例 ---

def manhattan_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    """マンハッタン距離（4方向移動のヒューリスティック）。"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def grid_neighbors(grid: list[list[int]],
                   node: tuple[int, int]) -> list[tuple]:
    """グリッド上の隣接セル（上下左右、壁以外）を返す。"""
    rows, cols = len(grid), len(grid[0])
    r, c = node
    neighbors = []
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
            neighbors.append(((nr, nc), 1))  # コスト 1
    return neighbors


# グリッドマップ（0: 通行可, 1: 壁）
grid = [
    [0, 0, 0, 0, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
]

path, cost = a_star(
    grid, (0, 0), (4, 4),
    h=manhattan_distance,
    get_neighbors=grid_neighbors
)
print(f"経路: {path}")  # => [(0,0), (0,1), (0,2), (1,2), (2,2), ...]
print(f"コスト: {cost}")  # => 8
```

### 6.3 ASCII 図解: A* の探索過程

```
A* 探索: グリッドマップ（S=始点, G=終点, #=壁, .=通行可）

  初期マップ:
    S . . . .
    # # . # .
    . . . . .
    . # # # .
    . . . . G

  探索過程（数字は f 値 = g + h）:

  Step 1: S を展開
    [S] 1  2  3  4       f(S) = 0 + 8 = 8
     #  #  .  #  .
     .  .  .  .  .
     .  #  #  #  .
     .  .  .  .  G

  Step 5 付近:
     S  1  2  .  .       ← 上方を探索
     #  #  3  #  .
     .  .  4  .  .       ← 右下方向に進む
     .  #  #  #  .
     .  .  .  .  G

  最終結果:
     *  *  *  .  .       * = 最短経路
     #  #  *  #  .
     .  .  *  *  *
     .  #  #  #  *
     .  .  .  .  *

  経路長: 8 ステップ

  A* vs BFS の比較（このマップ）:
  - BFS: 全方向に均等に探索 → 展開ノード数が多い
  - A* : ゴール方向を優先 → 展開ノード数が少ない（効率的）
```

### 6.4 A* の特殊ケースとの関係

```
A* とその特殊ケース

  h(n) = 0 の場合:
    f(n) = g(n) + 0 = g(n)
    → ダイクストラ法に退化（全方向を均等に探索）

  g(n) = 0 の場合:
    f(n) = 0 + h(n) = h(n)
    → 貪欲最良優先探索（最適性を保証しない）

  h(n) = 真のコストの場合:
    → 最適経路上のノードのみを展開（理想的だが計算不能な場合が多い）

  関係図:
  ┌─────────────────────────────────────────┐
  │                A*                        │
  │    f(n) = g(n) + h(n)                    │
  │                                          │
  │  ┌──────────────┐  ┌──────────────────┐  │
  │  │ h(n) = 0     │  │ g(n) = 0         │  │
  │  │ → ダイクストラ│  │ → 貪欲最良優先   │  │
  │  │  （最適・遅い）│  │  （非最適・速い）│  │
  │  └──────────────┘  └──────────────────┘  │
  │                                          │
  │  h(n) が許容的 → 最適解を保証            │
  │  h(n) が整合的 → 効率的な探索を保証      │
  └─────────────────────────────────────────┘
```

### 6.5 ヒューリスティック関数の選択と影響

```python
import math


# --- ヒューリスティック関数の例 ---

def manhattan(a: tuple, b: tuple) -> float:
    """マンハッタン距離: 4方向移動に最適。"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a: tuple, b: tuple) -> float:
    """ユークリッド距離: 自由移動に最適。"""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def chebyshev(a: tuple, b: tuple) -> float:
    """チェビシェフ距離: 8方向移動に最適。"""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def zero_heuristic(a: tuple, b: tuple) -> float:
    """ゼロヒューリスティック: ダイクストラ法と同等。"""
    return 0


# --- ヒューリスティックの強さと探索効率の関係 ---
#
# h(n) が大きいほど:
# ✓ 探索ノード数が減る（効率的）
# ✗ 最適性を失うリスクが上がる（h(n) > 真のコストの場合）
#
# h(n) が小さいほど:
# ✓ 最適解の保証が確実
# ✗ 探索ノード数が増える（非効率）
#
# 理想: h(n) = 真のコスト（計算できれば探索不要）
# 現実: 許容的な範囲で h(n) を最大化する
```

---

## 第6部: 実務での探索

---

## 7. データベースにおける探索

### 7.1 インデックスの種類と特性

```
データベースインデックスの比較

  ┌───────────────┬───────────────┬─────────────┬──────────────┐
  │ インデックス   │ 探索計算量     │ 範囲検索     │ 用途         │
  ├───────────────┼───────────────┼─────────────┼──────────────┤
  │ B-Tree        │ O(log n)      │ 効率的      │ 汎用         │
  │ Hash          │ O(1) 平均     │ 不可        │ 完全一致     │
  │ GiST          │ O(log n)      │ 効率的      │ 幾何/全文    │
  │ GIN           │ O(1)〜O(k)   │ 一部可能    │ 全文/配列    │
  │ BRIN          │ O(1)          │ 効率的      │ 大規模連番   │
  └───────────────┴───────────────┴─────────────┴──────────────┘
```

### 7.2 B-Tree インデックスの構造

```
B-Tree インデックス（簡略化）

  目標: email = 'test@example.com' を検索

                    ┌─────────────────────┐
                    │   [M]               │    ← ルートノード
                    │ < M    |    >= M    │
                    └──┬──────────┬───────┘
                       │          │
            ┌──────────┘          └──────────┐
            ▼                                 ▼
  ┌─────────────────┐              ┌─────────────────┐
  │ [D]  [H]        │              │ [R]  [V]        │
  │ <D | D-H | >H   │              │ <R | R-V | >V   │
  └──┬───┬────┬─────┘              └──┬───┬────┬─────┘
     │   │    │                       │   │    │
     ▼   ▼    ▼                       ▼   ▼    ▼
   [葉] [葉] [葉]                  [葉] [葉] [葉]
   a-c  d-g   h-l                  m-q  r-u   v-z
                                        ↑
                                   test@ はここ！

  特徴:
  - 各ノードは 1 ディスクページ（4KB〜16KB）に収まる
  - 扇出数（fanout）: 通常 100〜500
  - 100万件でも高さ 3〜4（= ディスクI/O 3〜4回）
  - 葉ノードはリンクリストで連結 → 範囲検索が効率的

  インデックスなし vs あり:
    SELECT * FROM users WHERE email = 'test@example.com';
    - インデックスなし: フルテーブルスキャン O(n) → 100万行で数秒
    - B-Tree: O(log n) → 100万行で数ミリ秒
```

### 7.3 全文検索と転置インデックス

```
転置インデックス（Inverted Index）

  文書データ:
    Doc1: "The quick brown fox jumps"
    Doc2: "The lazy brown dog sleeps"
    Doc3: "Quick fox runs fast"

  転置インデックス:
    ┌───────────┬──────────────────┐
    │ トークン   │ ポスティングリスト │
    ├───────────┼──────────────────┤
    │ brown     │ [Doc1:3, Doc2:3] │
    │ dog       │ [Doc2:4]         │
    │ fast      │ [Doc3:4]         │
    │ fox       │ [Doc1:4, Doc3:2] │
    │ jumps     │ [Doc1:5]         │
    │ lazy      │ [Doc2:2]         │
    │ quick     │ [Doc1:2, Doc3:1] │
    │ runs      │ [Doc3:3]         │
    │ sleeps    │ [Doc2:5]         │
    │ the       │ [Doc1:1, Doc2:1] │
    └───────────┴──────────────────┘

  検索 "quick AND fox":
    quick → {Doc1, Doc3}
    fox   → {Doc1, Doc3}
    AND   → {Doc1, Doc3}  ← 積集合

  検索 "brown OR dog":
    brown → {Doc1, Doc2}
    dog   → {Doc2}
    OR    → {Doc1, Doc2}  ← 和集合

  代表的な技術:
  - Elasticsearch / OpenSearch: Lucene ベースの分散全文検索エンジン
  - PostgreSQL: tsvector + GIN インデックス
  - SQLite FTS5: 軽量な全文検索拡張
  - Apache Solr: Lucene ベースのエンタープライズ検索
```

---

## 第7部: 総合比較と選択指針

---

## 8. 探索アルゴリズム総合比較表

```
全探索アルゴリズムの比較

┌──────────────────┬───────────┬───────────┬──────────┬──────────────────┐
│ アルゴリズム     │ 平均時間   │ 最悪時間   │ 空間     │ 前提条件         │
├──────────────────┼───────────┼───────────┼──────────┼──────────────────┤
│ 線形探索         │ O(n)      │ O(n)      │ O(1)     │ なし             │
│ 二分探索         │ O(log n)  │ O(log n)  │ O(1)     │ ソート済み       │
│ ハッシュ探索     │ O(1)      │ O(n)      │ O(n)     │ ハッシュ関数     │
│ BFS              │ O(V+E)    │ O(V+E)    │ O(V)     │ グラフ構造       │
│ DFS              │ O(V+E)    │ O(V+E)    │ O(V)     │ グラフ構造       │
│ A*               │ O(b^d)*   │ O(b^d)    │ O(b^d)   │ ヒューリスティック│
│ B-Tree探索       │ O(log n)  │ O(log n)  │ O(n)     │ 構築済みツリー   │
│ 転置インデックス │ O(1)〜O(k)│ O(n)      │ O(n+m)   │ 構築済みインデックス│
└──────────────────┴───────────┴───────────┴──────────┴──────────────────┘

  * A* の平均計算量は h(n) の品質に依存する。良い h(n) で大幅に改善される。
  b = 分岐数, d = 解の深さ, V = 頂点数, E = 辺数, k = 結果数, m = 総トークン数
```

### 8.1 問題別の推奨アルゴリズム

| 問題の種類 | 推奨アルゴリズム | 理由 |
|-----------|----------------|------|
| 小規模配列の探索 | 線形探索 | ソート不要、オーバーヘッドが最小 |
| ソート済み大規模配列 | 二分探索 | O(log n) で追加メモリ不要 |
| キーバリュー検索 | ハッシュテーブル | O(1) の平均探索時間 |
| 最短経路（重み無し） | BFS | 最短性を保証 |
| 最短経路（重み付き） | A* / ダイクストラ | ヒューリスティックで効率化 |
| 全解列挙 | DFS + バックトラッキング | メモリ効率が良い |
| 依存関係の解決 | DFS（トポロジカルソート） | DAG の順序付け |
| データベース検索 | B-Tree インデックス | ディスクI/O 最適化 |
| テキスト検索 | 転置インデックス | 全文検索に特化 |

---

## 第8部: アンチパターンと注意点

---

## 9. アンチパターン

### 9.1 アンチパターン1: 「とりあえず線形探索」症候群

**症状**: データ量やアクセスパターンを分析せず、常に線形探索を使う。

```python
# --- NG パターン ---
# 10万件のユーザリストから毎回線形探索
def find_user_by_email_bad(users: list[dict], email: str) -> dict | None:
    """
    問題: リクエストのたびに O(n) の探索が走る。
    1秒あたり1000リクエスト × 10万件 = 1億回の比較/秒
    """
    for user in users:
        if user["email"] == email:
            return user
    return None


# --- OK パターン ---
# 起動時にインデックスを構築し、O(1) で探索
def build_user_index(users: list[dict]) -> dict[str, dict]:
    """O(n) で一度だけインデックスを構築する。"""
    return {user["email"]: user for user in users}

user_index = build_user_index(users)

def find_user_by_email_good(email: str) -> dict | None:
    """O(1) で探索。"""
    return user_index.get(email)

# 判断基準:
# - 検索が 1 回だけ → 線形探索で十分
# - 検索が繰り返される → インデックス構築 + O(1) 探索
# - データが頻繁に変更 → 更新コストとのバランスを考える
```

### 9.2 アンチパターン2: 不適切なハッシュ関数の選択

**症状**: ハッシュ関数の品質を検証せず、衝突率が高い関数を使用する。

```python
# --- NG パターン ---
class BadHashTable:
    """悪いハッシュ関数の例: 全ての文字列が同じバケットに入る可能性。"""

    def _hash(self, key: str) -> int:
        # 文字列長でハッシュ → 同じ長さの文字列は全て衝突
        return len(key) % self._capacity


# --- NG パターン2 ---
class AnotherBadHashTable:
    """別の悪い例: 先頭文字のみでハッシュ。"""

    def _hash(self, key: str) -> int:
        # 先頭文字の ASCII 値 → "apple", "avocado", "apricot" が全て衝突
        return ord(key[0]) % self._capacity


# --- OK パターン ---
class GoodHashTable:
    """Python 組み込みの hash() を使う（SipHash ベース）。"""

    def _hash(self, key) -> int:
        return hash(key) % self._capacity

# 教訓:
# - 自前のハッシュ関数を作らない（言語標準のものを使う）
# - ハッシュ関数はキーの全情報を使うべき
# - セキュリティが重要な場面では暗号学的ハッシュ関数を使う
```

### 9.3 アンチパターン3: BFS で重み付きグラフの最短経路を求める

**症状**: 辺に重みがあるグラフに対して BFS を使い、正しくない最短経路を得る。

```python
# --- NG パターン ---
# BFS は重み無しグラフでのみ最短経路を保証する。
# 辺に異なる重みがある場合、BFS は最適解を返さない。
#
# 例: A→B(コスト1), A→C(コスト5), B→C(コスト1)
# BFS: A→C (コスト5) を最短と判定する可能性がある
# 正解: A→B→C (コスト2) が最短

# --- OK パターン ---
# 重み付きグラフではダイクストラ法または A* を使用する。
import heapq

def dijkstra(graph: dict, start: str) -> dict[str, float]:
    """ダイクストラ法: 重み付きグラフの最短経路。"""
    dist = {start: 0}
    pq = [(0, start)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float('inf')):
            continue
        for v, w in graph.get(u, []):
            nd = d + w
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))

    return dist
```

### 9.4 アンチパターン4: 二分探索の前提条件を確認しない

**症状**: ソートされていない配列に二分探索を適用し、誤った結果を得る。

```python
# --- NG パターン ---
data = [3, 1, 4, 1, 5, 9, 2, 6]  # ソートされていない！
result = binary_search(data, 5)    # 不正な結果を返す可能性大

# --- OK パターン ---
# 方法1: 事前にソートする
data.sort()  # O(n log n)
result = binary_search(data, 5)    # 正しい結果

# 方法2: ソート済みを仮定できない場合は線形探索を使う
result = linear_search(data, 5)

# 方法3: 頻繁に検索するなら、ソート済みリストを維持する
import bisect
sorted_data = sorted(data)  # 初回のみ O(n log n)
bisect.insort(sorted_data, new_element)  # 挿入時に O(n)（シフトが発生）
# 挿入が多い場合は SortedList (sortedcontainers) を検討
```

---

## 第9部: 演習問題

---

## 10. 演習問題（3 段階）

### 10.1 基礎レベル

#### 演習 B1: 線形探索のバリエーション

以下の関数を実装せよ。

```
1. find_min(arr): リストの最小値のインデックスを返す（組み込み関数を使わずに）
2. find_last(arr, target): target が最後に出現するインデックスを返す
3. count_if(arr, predicate): 条件を満たす要素の個数を返す
```

**テストケース**:
```python
assert find_min([5, 3, 8, 1, 9, 2]) == 3
assert find_last([1, 3, 5, 3, 7, 3], 3) == 5
assert count_if([1, 2, 3, 4, 5, 6], lambda x: x % 2 == 0) == 3
```

#### 演習 B2: 二分探索の基本

以下の関数を実装せよ。

```
1. binary_search(arr, target): 完全一致探索
2. lower_bound(arr, target): target 以上の最小インデックス
3. upper_bound(arr, target): target より大きい最小インデックス
4. count_in_range(arr, lo, hi): lo <= x <= hi を満たす要素数
   （ヒント: upper_bound と lower_bound の組み合わせ）
```

**テストケース**:
```python
arr = [1, 2, 2, 3, 3, 3, 4, 5, 5]
assert binary_search(arr, 3) in [3, 4, 5]  # いずれかの 3 のインデックス
assert lower_bound(arr, 3) == 3
assert upper_bound(arr, 3) == 6
assert count_in_range(arr, 2, 4) == 6  # [2,2,3,3,3,4] の6個
```

### 10.2 応用レベル

#### 演習 A1: 答えで二分探索

**問題**: N 人の作業者に M 個のタスクを連番で割り当てる。各タスクの所要時間が与えられる。全員に連続する番号のタスクを割り当てたとき、最も負荷の高い作業者の所要時間合計を最小化せよ。

```
入力: tasks = [7, 2, 5, 10, 8], workers = 2
出力: 18

説明: [7, 2, 5] と [10, 8] に分割 → 最大は max(14, 18) = 18
      [7, 2, 5, 10] と [8] → max(24, 8) = 24 → 非最適
```

**ヒント**: 「最大の所要時間合計が X 以下になるように分割できるか？」を判定関数とし、X を二分探索する。

#### 演習 A2: BFS による最短変換

**問題**: 与えられた単語リストを使い、始点の単語から終点の単語へ、1 文字ずつ変換する最短ステップ数を求めよ（Word Ladder）。

```
入力: begin = "hit", end = "cog",
      words = ["hot", "dot", "dog", "lot", "log", "cog"]
出力: 5  ("hit" → "hot" → "dot" → "dog" → "cog")
```

### 10.3 発展レベル

#### 演習 C1: A* による 15 パズル

**問題**: 4x4 の 15 パズル（スライドパズル）を A* で解くプログラムを実装せよ。

```
初期状態:        目標状態:
 1  2  3  4      1  2  3  4
 5  6  _  8      5  6  7  8
 9 10  7 11      9 10 11 12
13 14 15 12     13 14 15  _
```

**要件**:
1. ヒューリスティック関数としてマンハッタン距離の合計を使用する
2. 解の手順（移動方向の列）を出力する
3. 展開したノード数も出力し、ヒューリスティックの効果を確認する

#### 演習 C2: ハッシュテーブルの完全実装

**問題**: オープンアドレス法（ダブルハッシング）によるハッシュテーブルを実装せよ。

**要件**:
1. 挿入、探索、削除、リサイズをサポート
2. 削除時に墓標（tombstone）を使用
3. ロードファクター 0.5 超で 2 倍に拡張
4. イテレータ（`__iter__`）をサポート
5. 衝突回数を記録するデバッグモードを実装

---

## 第10部: FAQ・参考文献

---

## 11. FAQ（よくある質問）

### Q1: 二分探索はソート済み配列以外にも使えるか？

**A**: 使える。「答えで二分探索（Binary Search on Answer）」は非常に強力なパターンである。ある条件が「しきい値を境に成立/不成立が切り替わる単調性」を持つならば、そのしきい値を二分探索で効率的に求められる。これは最適化問題を判定問題に帰着させるテクニックであり、競技プログラミングでも実務でも頻出する。具体的には「N 日以内に荷物を全て配送できる最小のトラック台数」「全員が満足する最小のピザサイズ」といった問題に適用できる。

### Q2: ハッシュテーブルの最悪 O(n) は実務上問題になるか？

**A**: 通常は問題にならない。適切なハッシュ関数（Python の SipHash など）とロードファクター管理（2/3 以下で自動リハッシュ）により、衝突は統計的に最小化される。ただし、**Hash DoS 攻撃**には注意が必要である。攻撃者が意図的に同じハッシュ値を持つ入力を大量に送ることで、O(n) の探索を強制しサービスを停止させる手法が知られている。Python は 3.3 以降、ハッシュのランダム化（PYTHONHASHSEED）と SipHash の採用でこの攻撃を緩和している。

### Q3: BFS と DFS はどちらが良いか？

**A**: 問題の性質によって使い分ける。最短経路が必要なら BFS、メモリを節約したいなら DFS、全解の列挙ならDFS（バックトラッキング）、依存関係の解決なら DFS（トポロジカルソート）が適している。到達可能性のみを判定する場合はどちらでも構わないが、DFS の方が実装が簡潔になることが多い。なお、BFS は解がルートに近い場合に有利であり、DFS は解が深い位置にある場合や、枝刈りが効く場合に有利である。

### Q4: A* のヒューリスティック関数はどう設計すればよいか？

**A**: 以下の 3 つの原則に従う。(1) **許容的であること**: h(n) は真のコスト以下でなければならない。これが満たされないと最適解が保証されない。(2) **整合的（consistent）であること**: 任意のノード n とその隣接ノード m に対して h(n) <= cost(n, m) + h(m) が成り立つこと。これが満たされると、A* は各ノードを高々 1 回しか展開しない。(3) **できるだけ大きい値を返すこと**: 許容性の範囲内で h(n) が大きいほど、探索ノード数が少なくなり効率的になる。グリッドマップの 4 方向移動にはマンハッタン距離、8 方向移動にはチェビシェフ距離、自由移動にはユークリッド距離が一般的に使われる。

### Q5: データベースのインデックスはいくつ作るべきか？

**A**: クエリパターンに応じて必要最小限にする。インデックスは読み取りを高速化するが、書き込み（INSERT / UPDATE / DELETE）を遅くする。各インデックスは追加のストレージも消費する。一般的な指針として、(1) WHERE 句で頻繁に使われるカラムにインデックスを作成する、(2) JOIN のキーカラムにインデックスを作成する、(3) 選択性（cardinality）が低いカラム（例: boolean）にはインデックスの効果が薄い、(4) 複合インデックスは先頭カラムの選択性が重要である、(5) EXPLAIN ANALYZE で実際のクエリプランを確認して判断する。

### Q6: 探索アルゴリズムの学習順序としてどの順番が効果的か？

**A**: 以下の順序を推奨する。(1) 線形探索（基本の確認） → (2) 二分探索（分割統治の基礎） → (3) ハッシュ探索（ハッシュテーブルの理解） → (4) BFS/DFS（グラフ探索の基礎） → (5) A*（ヒューリスティック探索）。各段階で演習問題を解いてから次に進むことで、理解が深まる。特に二分探索は「答えで二分探索」パターンまで習得すると、問題解決の幅が大きく広がる。

---

## 12. 参考文献

### 12.1 書籍

1. **Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C.** (2022). *Introduction to Algorithms* (4th ed.). MIT Press. — 通称 CLRS。探索アルゴリズムの理論的基盤を包括的にカバー。Chapter 11（Hash Tables）、Chapter 12（Binary Search Trees）、Chapter 22（BFS/DFS）が特に関連する。

2. **Knuth, D. E.** (1998). *The Art of Computer Programming, Vol. 3: Sorting and Searching* (2nd ed.). Addison-Wesley. — 探索アルゴリズムの数学的解析の決定版。ハッシュ法の詳細な分析が秀逸。

3. **Sedgewick, R., & Wayne, K.** (2011). *Algorithms* (4th ed.). Addison-Wesley. — 実装に重点を置いた教科書。Java によるコード例が豊富で、可視化ツールも公開されている。

4. **Bentley, J.** (2000). *Programming Pearls* (2nd ed.). Addison-Wesley. — 二分探索の正しい実装がいかに難しいかを示す有名な逸話を含む。Column 4 が特に関連する。

5. **Skiena, S. S.** (2020). *The Algorithm Design Manual* (3rd ed.). Springer. — 実務での探索アルゴリズムの応用に焦点を当てた実践的な教科書。

### 12.2 論文

6. **Comer, D.** (1979). "The Ubiquitous B-Tree." *ACM Computing Surveys*, 11(2), 121-137. — B-Tree の概念と応用を網羅した古典的なサーベイ論文。

7. **Hart, P. E., Nilsson, N. J., & Raphael, B.** (1968). "A Formal Basis for the Heuristic Determination of Minimum Cost Paths." *IEEE Transactions on Systems Science and Cybernetics*, SSC-4(2), 100-107. — A* アルゴリズムの原論文。

8. **Tarjan, R. E.** (1972). "Depth-First Search and Linear Graph Algorithms." *SIAM Journal on Computing*, 1(2), 146-160. — DFS の理論的基盤を確立した重要論文。

### 12.3 オンラインリソース

9. **VisuAlgo** (https://visualgo.net/) — 各種探索アルゴリズムのインタラクティブな可視化ツール。動作の理解に有用。

10. **Python bisect モジュール公式ドキュメント** — https://docs.python.org/3/library/bisect.html — Python 標準ライブラリの二分探索実装。

---

## 13. まとめ

### 13.1 探索アルゴリズム早見表

| 探索方法 | 計算量 | 前提 | 最適な用途 |
|---------|--------|------|-----------|
| 線形探索 | O(n) | なし | 小規模/未ソート/一度きりの検索 |
| 二分探索 | O(log n) | ソート済み | 大規模/範囲検索/境界探索 |
| ハッシュ探索 | O(1)期待 | ハッシュ関数 | 完全一致/高頻度検索 |
| BFS | O(V+E) | グラフ構造 | 最短経路（重み無し）/レベル順走査 |
| DFS | O(V+E) | グラフ構造 | トポロジカルソート/サイクル検出/全解列挙 |
| A* | O(b^d)* | ヒューリスティック | 最短経路（重み付き）/経路探索 |
| B-Tree | O(log n) | 構築済み | DB/ファイルシステム |
| 転置インデックス | O(1)〜O(k) | 構築済み | 全文検索 |

### 13.2 キーポイント

1. **探索はデータ構造と不可分**: 適切なデータ構造を選ぶことが、効率的な探索の前提条件となる
2. **前提条件を確認する**: 二分探索にはソート、ハッシュ探索にはハッシュ関数、A* にはヒューリスティックが必要
3. **問題の性質に合わせて選択**: 完全一致かか範囲検索か、最短性が必要か、データは静的か動的か
4. **計算量は平均と最悪を区別**: ハッシュの O(1) は「期待値」であり、最悪は O(n) になり得る
5. **実務ではライブラリを活用**: Python の bisect、dict/set、collections.deque などを適切に使う

---

## 次に読むべきガイド

→ [[04-graph-algorithms.md]] — グラフアルゴリズム（ダイクストラ法、最小全域木など）

---
