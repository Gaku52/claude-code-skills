# ソートアルゴリズム

> データを特定の順序に並べ替える基本アルゴリズム群を、計算量・安定性・実装の観点から体系的に理解する

## この章で学ぶこと

1. **7種のソートアルゴリズム**の原理・実装・計算量を比較できるようになる
2. **安定性・in-place性・適応性**の違いから、場面に応じた最適なソートを選択できる
3. **分割統治・ヒープ・計数**といった異なるパラダイムがソートにどう適用されるかを理解する
4. **TimSort・Introsort** など現代の実用ソートの設計思想を把握する
5. **外部ソート・並列ソート** など発展的なトピックの基礎を理解する

---

## 目次

1. [ソートの全体像](#1-ソートの全体像)
2. [バブルソート](#2-バブルソートbubble-sort)
3. [選択ソート](#3-選択ソートselection-sort)
4. [挿入ソート](#4-挿入ソートinsertion-sort)
5. [マージソート](#5-マージソートmerge-sort)
6. [クイックソート](#6-クイックソートquick-sort)
7. [ヒープソート](#7-ヒープソートheap-sort)
8. [非比較ベースソート](#8-非比較ベースソート)
9. [高度なソートアルゴリズム](#9-高度なソートアルゴリズム)
10. [計算量比較表と用途別選択ガイド](#10-計算量比較表と用途別選択ガイド)
11. [アンチパターン](#11-アンチパターン)
12. [演習問題](#12-演習問題)
13. [FAQ](#13-faq)
14. [まとめ](#14-まとめ)
15. [参考文献](#15-参考文献)

---

## 1. ソートの全体像

### 1.1 ソートとは何か

ソート（整列）とは、データの集合を特定の順序関係に従って並べ替える操作である。コンピュータサイエンスにおいて最も基本的かつ重要な問題の一つであり、探索の前処理、データの可視化、重複除去、統計処理など、あらゆる場面で利用される。

Donald Knuth は *The Art of Computer Programming* において「コンピュータの計算時間のうち、ソートに費やされる時間は全体の25%以上にのぼる」と推定した。現代のシステムでもデータベースのインデックス構築、ファイルシステムのディレクトリ表示、検索エンジンのランキングなど、ソートは至る所に存在する。

### 1.2 ソートの分類体系

```
┌──────────────────────────────────────────────────────────────────────┐
│                     ソートアルゴリズムの分類体系                        │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              比較ベースソート (Comparison-based)              │    │
│  │          理論的下限: Omega(n log n)                           │    │
│  ├───────────────────────┬─────────────────────────────────────┤    │
│  │  単純: O(n^2)         │  効率的: O(n log n)                 │    │
│  │  ・バブルソート        │  ・マージソート                      │    │
│  │  ・選択ソート          │  ・クイックソート                    │    │
│  │  ・挿入ソート          │  ・ヒープソート                      │    │
│  │  ・シェルソート        │  ・TimSort (ハイブリッド)            │    │
│  │                       │  ・Introsort (ハイブリッド)          │    │
│  └───────────────────────┴─────────────────────────────────────┘    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │            非比較ベースソート (Non-comparison-based)          │    │
│  │          条件付きで O(n) 達成可能                             │    │
│  ├─────────────────────────────────────────────────────────────┤    │
│  │  ・計数ソート (Counting Sort)    -- 整数・範囲小             │    │
│  │  ・基数ソート (Radix Sort)       -- 固定長キー               │    │
│  │  ・バケットソート (Bucket Sort)   -- 一様分布                 │    │
│  └─────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
```

### 1.3 ソートの重要な性質

ソートアルゴリズムを評価する際には、時間計算量だけでなく以下の性質も考慮する必要がある。

**安定性 (Stability)**

安定なソートとは、等しいキーを持つ要素の相対的な順序がソート前後で保たれることを保証するソートである。

```
安定ソートの例:
入力:  [(3,"Alice"), (1,"Bob"), (3,"Charlie"), (2,"Dave")]
        キーでソート
出力:  [(1,"Bob"), (2,"Dave"), (3,"Alice"), (3,"Charlie")]
        ↑ Alice と Charlie の相対順序が保持されている

不安定ソートの例:
出力:  [(1,"Bob"), (2,"Dave"), (3,"Charlie"), (3,"Alice")]
        ↑ Alice と Charlie の相対順序が逆転する可能性がある
```

安定性が重要になるのは、複数のキーでソートする場合である。たとえば、まず名前でソートし、次に成績でソートするとき、安定ソートであれば同一成績の学生は名前順が維持される。

**in-place性**

追加のメモリ使用量が O(1) または O(log n) であるソートを in-place ソートと呼ぶ。組み込みシステムやメモリ制約の厳しい環境では in-place 性が重要になる。

**適応性 (Adaptivity)**

入力がほぼ整列済みの場合に、計算量が改善されるソートを適応的ソートと呼ぶ。挿入ソートは典型例で、ほぼ整列済みなら O(n) で完了する。

### 1.4 比較ベースソートの理論的下限

比較ベースのソートアルゴリズムには、**O(n log n)** という理論的下限が存在する。これは決定木モデルによって証明される。

```
n=3 の決定木 (要素 a, b, c):

                    a < b ?
                   /       \
                yes         no
               /               \
          b < c ?             a < c ?
         /     \             /     \
       yes     no          yes     no
       /         \         /         \
   [a,b,c]    a < c ?  [b,a,c]   b < c ?
              /     \            /     \
            yes     no         yes     no
            /         \        /         \
        [a,c,b]    [c,a,b] [b,c,a]   [c,b,a]

葉の数 = n! = 6 (3つの要素の全順列)
木の高さ >= log2(n!) = Omega(n log n)   (スターリングの近似より)
```

n 個の要素の全順列は n! 通りあり、決定木の各葉はそのうち1つに対応する。木の高さ（＝最悪の比較回数）は log2(n!) 以上であり、スターリングの近似 n! ≈ (n/e)^n から log2(n!) = Omega(n log n) が導かれる。

---

## 2. バブルソート（Bubble Sort）

### 2.1 アルゴリズムの原理

隣接する要素を比較・交換し、最大値を末尾に「泡」のように浮かせる操作を繰り返す。名前の由来は、大きな値が配列の末尾に向かって「泡立つ」ように移動する様子から来ている。

### 2.2 動作の可視化

```
初期配列: [5, 3, 8, 1, 2]

パス1: 未ソート部分 [5, 3, 8, 1, 2]
  比較 5>3 → 交換  [3, 5, 8, 1, 2]
  比較 5<8 → 維持  [3, 5, 8, 1, 2]
  比較 8>1 → 交換  [3, 5, 1, 8, 2]
  比較 8>2 → 交換  [3, 5, 1, 2, 8]  ← 8 が確定位置へ
                                  ~~~~~~~~

パス2: 未ソート部分 [3, 5, 1, 2] | 確定 [8]
  比較 3<5 → 維持  [3, 5, 1, 2, 8]
  比較 5>1 → 交換  [3, 1, 5, 2, 8]
  比較 5>2 → 交換  [3, 1, 2, 5, 8]  ← 5 が確定位置へ
                             ~~~~~

パス3: 未ソート部分 [3, 1, 2] | 確定 [5, 8]
  比較 3>1 → 交換  [1, 3, 2, 5, 8]
  比較 3>2 → 交換  [1, 2, 3, 5, 8]  ← 3 が確定位置へ
                          ~~~~~~~~

パス4: 未ソート部分 [1, 2] | 確定 [3, 5, 8]
  比較 1<2 → 維持  [1, 2, 3, 5, 8]  ← 交換なし → 完了!

結果: [1, 2, 3, 5, 8]
```

### 2.3 実装（Python）

```python
def bubble_sort(arr: list) -> list:
    """バブルソート - 安定・in-place

    隣接要素を比較・交換し、最大値を末尾に移動させることを繰り返す。
    最適化: 1パスで交換がなければソート済みと判定して早期終了する。

    Args:
        arr: ソート対象のリスト（破壊的に変更される）

    Returns:
        ソート済みのリスト（入力と同一オブジェクト）

    計算量:
        最良: O(n)   -- 入力がソート済みの場合
        平均: O(n^2)
        最悪: O(n^2)
        空間: O(1)
    """
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(n - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr


# --- 動作確認 ---
if __name__ == "__main__":
    data = [64, 34, 25, 12, 22, 11, 90]
    print(f"入力: {data}")
    result = bubble_sort(data)
    print(f"出力: {result}")
    # 入力: [64, 34, 25, 12, 22, 11, 90]
    # 出力: [11, 12, 22, 25, 34, 64, 90]

    # ソート済みデータ → O(n) で完了
    sorted_data = [1, 2, 3, 4, 5]
    bubble_sort(sorted_data)
    print(f"ソート済み: {sorted_data}")  # [1, 2, 3, 4, 5]
```

### 2.4 計算量分析

| ケース | 比較回数 | 交換回数 | 説明 |
|:---|:---|:---|:---|
| 最良 | O(n) | 0 | 入力がソート済み（swapped フラグで早期終了） |
| 平均 | O(n^2) | O(n^2) | ランダムな入力 |
| 最悪 | O(n^2) | O(n^2) | 逆順にソートされた入力 |

バブルソートの比較回数は最悪で n(n-1)/2 回、交換回数は転倒数（inversion count）に等しい。転倒数とは、i < j かつ arr[i] > arr[j] となるペアの数である。

### 2.5 バリエーション: カクテルシェーカーソート

バブルソートは一方向にしかスキャンしないが、カクテルシェーカーソートは前後交互にスキャンする。「うさぎ問題」（小さい値が配列の末尾にある場合の移動の遅さ）を軽減する。

```python
def cocktail_shaker_sort(arr: list) -> list:
    """カクテルシェーカーソート（双方向バブルソート）

    バブルソートの改良版。前方向と後方向を交互にスキャンする。
    末尾付近の小さい値の移動が高速化される。

    計算量は最悪 O(n^2) で変わらないが、定数係数が改善される場合がある。
    """
    n = len(arr)
    start = 0
    end = n - 1
    swapped = True

    while swapped:
        swapped = False

        # 前方スキャン: 最大値を末尾へ
        for i in range(start, end):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swapped = True
        end -= 1

        if not swapped:
            break

        swapped = False

        # 後方スキャン: 最小値を先頭へ
        for i in range(end, start, -1):
            if arr[i] < arr[i - 1]:
                arr[i], arr[i - 1] = arr[i - 1], arr[i]
                swapped = True
        start += 1

    return arr


# --- 動作確認 ---
if __name__ == "__main__":
    data = [5, 1, 4, 2, 8, 0, 2]
    print(cocktail_shaker_sort(data))  # [0, 1, 2, 2, 4, 5, 8]
```

---

## 3. 選択ソート（Selection Sort）

### 3.1 アルゴリズムの原理

未ソート部分から最小値を見つけ、未ソート部分の先頭と交換する。この操作を繰り返すことで、先頭から順にソート済みの部分が拡大していく。

### 3.2 動作の可視化

```
初期配列: [29, 10, 14, 37, 13]

ステップ1: 未ソート [29, 10, 14, 37, 13]
  最小値 = 10 (インデックス 1)
  29 と 10 を交換
  結果: [10 | 29, 14, 37, 13]
         ~~

ステップ2: 未ソート [29, 14, 37, 13]
  最小値 = 13 (インデックス 4)
  29 と 13 を交換
  結果: [10, 13 | 14, 37, 29]
             ~~

ステップ3: 未ソート [14, 37, 29]
  最小値 = 14 (インデックス 2) -- 自分自身
  交換不要
  結果: [10, 13, 14 | 37, 29]
                 ~~

ステップ4: 未ソート [37, 29]
  最小値 = 29 (インデックス 4)
  37 と 29 を交換
  結果: [10, 13, 14, 29 | 37]
                     ~~

完了: [10, 13, 14, 29, 37]
```

### 3.3 実装

```python
def selection_sort(arr: list) -> list:
    """選択ソート - 不安定・in-place

    未ソート部分の最小値を見つけ、先頭に配置することを繰り返す。
    交換回数は常に O(n) であり、書き込みコストが高い場合に有利。

    Args:
        arr: ソート対象のリスト

    Returns:
        ソート済みのリスト

    計算量:
        最良/平均/最悪: O(n^2) -- 入力に依らず常に同じ
        空間: O(1)
    """
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        if min_idx != i:
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr


# --- 動作確認 ---
if __name__ == "__main__":
    data = [29, 10, 14, 37, 13]
    print(selection_sort(data))  # [10, 13, 14, 29, 37]
```

### 3.4 選択ソートが不安定な理由

選択ソートが不安定であることを具体例で確認する。

```
入力: [3a, 2, 3b, 1]    (3a と 3b は同じキー値 3)

ステップ1: 最小値 = 1、3a と 1 を交換
  [1, 2, 3b, 3a]    ← 3a と 3b の相対順序が逆転!

結果: [1, 2, 3b, 3a]    ← 不安定
```

交換操作で離れた位置の要素を入れ替えるため、同一キーの要素の相対順序が崩れる。

### 3.5 選択ソートの特徴と用途

選択ソートは計算量の面では優れていないが、以下の特徴から特定の状況で有用である。

- **交換回数が O(n)**: 比較回数は O(n^2) だが、交換は各ステップで最大1回しか発生しない。書き込みコストが高い媒体（フラッシュメモリなど）では有利になる場合がある。
- **実装の単純さ**: コード量が少なく、バグが入りにくい。
- **予測可能な性能**: 入力データの内容に関わらず、常に同じ計算量。

---

## 4. 挿入ソート（Insertion Sort）

### 4.1 アルゴリズムの原理

トランプの手札を整列するように、未ソート部分の先頭要素を取り出し、ソート済み部分の適切な位置に挿入する。人間が自然に行うソート手法に最も近い。

### 4.2 動作の可視化

```
初期: [5, 2, 4, 6, 1, 3]
       ソート済み|未ソート

i=1: key=2
  ソート済み [5] に 2 を挿入
  5 > 2 → 5 を右シフト
  位置 0 に 2 を挿入
  結果: [2, 5 | 4, 6, 1, 3]
         ~~~~

i=2: key=4
  ソート済み [2, 5] に 4 を挿入
  5 > 4 → 5 を右シフト
  2 < 4 → 停止
  位置 1 に 4 を挿入
  結果: [2, 4, 5 | 6, 1, 3]
         ~~~~~~~

i=3: key=6
  ソート済み [2, 4, 5] に 6 を挿入
  5 < 6 → 停止（移動不要）
  結果: [2, 4, 5, 6 | 1, 3]
         ~~~~~~~~~~

i=4: key=1
  ソート済み [2, 4, 5, 6] に 1 を挿入
  6 > 1 → 右シフト
  5 > 1 → 右シフト
  4 > 1 → 右シフト
  2 > 1 → 右シフト
  位置 0 に 1 を挿入
  結果: [1, 2, 4, 5, 6 | 3]
         ~~~~~~~~~~~~~

i=5: key=3
  ソート済み [1, 2, 4, 5, 6] に 3 を挿入
  6 > 3 → 右シフト
  5 > 3 → 右シフト
  4 > 3 → 右シフト
  2 < 3 → 停止
  位置 2 に 3 を挿入
  結果: [1, 2, 3, 4, 5, 6]
         ~~~~~~~~~~~~~~~~

完了!
```

### 4.3 実装

```python
def insertion_sort(arr: list) -> list:
    """挿入ソート - 安定・in-place・適応的

    ソート済み部分に未ソート要素を一つずつ挿入する。
    ほぼ整列済みのデータに対して非常に高速（O(n) に近い）。

    Args:
        arr: ソート対象のリスト

    Returns:
        ソート済みのリスト

    計算量:
        最良: O(n)   -- 入力がソート済みの場合
        平均: O(n^2)
        最悪: O(n^2) -- 入力が逆順の場合
        空間: O(1)
    """
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr


# --- 動作確認 ---
if __name__ == "__main__":
    # ランダムデータ
    data = [5, 2, 4, 6, 1, 3]
    print(insertion_sort(data))  # [1, 2, 3, 4, 5, 6]

    # ほぼ整列済みデータ → O(n) に近い
    nearly_sorted = [1, 3, 2, 4, 6, 5]
    print(insertion_sort(nearly_sorted))  # [1, 2, 3, 4, 5, 6]
```

### 4.4 二分挿入ソート

挿入位置の探索を二分探索で行うことで、比較回数を O(n log n) に削減できる。ただし、要素のシフト操作は依然 O(n^2) のため、全体の計算量は O(n^2) のままである。

```python
import bisect

def binary_insertion_sort(arr: list) -> list:
    """二分挿入ソート - 安定・in-place

    挿入位置を二分探索で決定する改良版。
    比較回数は O(n log n) に削減されるが、
    シフト操作は O(n^2) のまま。

    連結リストと組み合わせればシフトが O(1) になるが、
    二分探索には O(n) のアクセスが必要なため相殺される。
    """
    for i in range(1, len(arr)):
        key = arr[i]
        # bisect_left で挿入位置を二分探索
        pos = bisect.bisect_left(arr, key, 0, i)
        # pos から i-1 までの要素を右にシフト
        for j in range(i, pos, -1):
            arr[j] = arr[j - 1]
        arr[pos] = key
    return arr


# --- 動作確認 ---
if __name__ == "__main__":
    data = [37, 23, 0, 17, 12, 72, 31]
    print(binary_insertion_sort(data))  # [0, 12, 17, 23, 31, 37, 72]
```

### 4.5 挿入ソートの重要性

挿入ソートは単純な O(n^2) アルゴリズムの中で最も実用的であり、以下の理由から現代のソートアルゴリズムの部品として広く使われている。

1. **小規模データに最適**: n が小さい場合、O(n log n) アルゴリズムのオーバーヘッド（再帰呼び出し、関数呼び出しコスト）が相対的に大きくなる。多くのライブラリ実装では、n < 16～64 程度で挿入ソートに切り替える。
2. **適応的**: ほぼ整列済みデータに対して O(n) で動作する。
3. **安定**: 等しいキーの相対順序を保持する。
4. **オンライン**: データが逐次的に到着する場合にも対応できる。
5. **キャッシュ効率が高い**: 隣接メモリへの連続アクセスが中心。

---

## 5. マージソート（Merge Sort）

### 5.1 アルゴリズムの原理

分割統治法の代表例。配列を再帰的に半分に分割し、各部分をソートした後、ソート済みの部分配列をマージ（統合）する。John von Neumann が 1945 年に考案した。

**3つのステップ:**
1. **分割 (Divide)**: 配列を半分に分割する
2. **征服 (Conquer)**: 各半分を再帰的にソートする
3. **統合 (Combine)**: 2つのソート済み配列をマージする

### 5.2 動作の可視化

```
分割フェーズ (トップダウン):
                [38, 27, 43, 3, 9, 82, 10]
                     /                \
            [38, 27, 43]         [3, 9, 82, 10]
              /      \             /         \
          [38]    [27, 43]     [3, 9]     [82, 10]
                   /   \        / \         /   \
                [27]  [43]   [3]  [9]    [82]  [10]

マージフェーズ (ボトムアップ):
                [27]  [43]   [3]  [9]    [82]  [10]
                   \   /        \ /         \   /
                [27, 43]     [3, 9]      [10, 82]
              \      /             \         /
          [27, 38, 43]       [3, 9, 10, 82]
                     \                /
              [3, 9, 10, 27, 38, 43, 82]

マージの詳細 ([27, 38, 43] と [3, 9, 10, 82]):
  L = [27, 38, 43]    R = [3, 9, 10, 82]
       ^                    ^
  比較: 27 > 3  → 3 を出力    結果: [3]
  比較: 27 > 9  → 9 を出力    結果: [3, 9]
  比較: 27 > 10 → 10 を出力   結果: [3, 9, 10]
  比較: 27 < 82 → 27 を出力   結果: [3, 9, 10, 27]
  比較: 38 < 82 → 38 を出力   結果: [3, 9, 10, 27, 38]
  比較: 43 < 82 → 43 を出力   結果: [3, 9, 10, 27, 38, 43]
  L が空 → R の残りを出力     結果: [3, 9, 10, 27, 38, 43, 82]
```

### 5.3 実装

```python
def merge_sort(arr: list) -> list:
    """マージソート - 安定・O(n) 追加メモリ

    分割統治法により配列を再帰的に分割・マージする。
    安定性と O(n log n) の保証を兼ね備える。

    Args:
        arr: ソート対象のリスト

    Returns:
        新しいソート済みリスト

    計算量:
        最良/平均/最悪: O(n log n)
        空間: O(n) -- マージ時の一時配列
    """
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return _merge(left, right)


def _merge(left: list, right: list) -> list:
    """2つのソート済み配列をマージする

    両方の配列の先頭から比較し、小さい方を結果に追加する。
    <= で比較することで安定性を保証する。
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


# --- 動作確認 ---
if __name__ == "__main__":
    data = [38, 27, 43, 3, 9, 82, 10]
    print(merge_sort(data))  # [3, 9, 10, 27, 38, 43, 82]
```

### 5.4 ボトムアップ・マージソート

再帰を使わない反復版の実装。スタックオーバーフローのリスクがなく、定数的なオーバーヘッドも小さい。

```python
def merge_sort_bottom_up(arr: list) -> list:
    """ボトムアップ・マージソート（反復版）

    再帰を使わず、サイズ 1 の部分配列から始めて
    順次マージサイズを倍にしていく。
    再帰版と同じ O(n log n) だが、関数呼び出しの
    オーバーヘッドがない。
    """
    n = len(arr)
    if n <= 1:
        return arr[:]

    result = arr[:]
    width = 1

    while width < n:
        for i in range(0, n, 2 * width):
            left = result[i:i + width]
            right = result[i + width:i + 2 * width]
            merged = _merge(left, right)
            result[i:i + len(merged)] = merged
        width *= 2

    return result


# --- 動作確認 ---
if __name__ == "__main__":
    data = [5, 2, 4, 7, 1, 3, 2, 6]
    print(merge_sort_bottom_up(data))  # [1, 2, 2, 3, 4, 5, 6, 7]
```

### 5.5 マージソートの特徴

- **安定性保証**: `<=` で比較することで、等しいキーの相対順序を保持
- **O(n log n) 保証**: 入力データの内容に関わらず常に O(n log n)
- **外部ソートとの親和性**: ディスク上の大規模データを部分的に読み込んでマージする外部ソートの基盤
- **並列化との親和性**: 分割された部分配列を独立にソートできるため、並列処理に向いている
- **欠点**: O(n) の追加メモリが必要

---

## 6. クイックソート（Quick Sort）

### 6.1 アルゴリズムの原理

Tony Hoare が 1959 年に考案。ピボット（基準値）を選び、配列を「ピボット以下」と「ピボット以上」の2つに分割し、それぞれを再帰的にソートする。分割統治法の一種だが、マージソートとは異なり「統合」のステップが不要で、分割（パーティション）に計算量が集中する。

### 6.2 パーティションの可視化

```
Lomuto パーティション (ピボット = 末尾要素):

配列: [10, 80, 30, 90, 40, 50, 70]   ピボット = 70
       i
       j

j=0: arr[0]=10 <= 70  → i++, swap(arr[0],arr[0])
  [10, 80, 30, 90, 40, 50, 70]
        i

j=1: arr[1]=80 > 70   → 何もしない
  [10, 80, 30, 90, 40, 50, 70]
        i

j=2: arr[2]=30 <= 70  → i++, swap(arr[1],arr[2])
  [10, 30, 80, 90, 40, 50, 70]
            i

j=3: arr[3]=90 > 70   → 何もしない

j=4: arr[4]=40 <= 70  → i++, swap(arr[2],arr[4])
  [10, 30, 40, 90, 80, 50, 70]
                i

j=5: arr[5]=50 <= 70  → i++, swap(arr[3],arr[5])
  [10, 30, 40, 50, 80, 90, 70]
                    i

最後: swap(arr[i+1], arr[high]) = swap(arr[4], arr[6])
  [10, 30, 40, 50, 70, 90, 80]
                    ^^
                  ピボット位置確定

左側 [10, 30, 40, 50] は全て <= 70
右側 [90, 80] は全て > 70
```

### 6.3 実装

```python
def quick_sort(arr: list, low: int = 0, high: int = None) -> list:
    """クイックソート - 不安定・in-place

    ピボットによる分割と再帰的ソートを行う。
    平均的に最も高速な汎用ソートアルゴリズム。

    Args:
        arr: ソート対象のリスト
        low: ソート範囲の開始インデックス
        high: ソート範囲の終了インデックス

    Returns:
        ソート済みのリスト

    計算量:
        最良: O(n log n)
        平均: O(n log n)
        最悪: O(n^2) -- ソート済み配列 + 末尾ピボット
        空間: O(log n) -- 再帰スタック（平均）
    """
    if high is None:
        high = len(arr) - 1
    if low < high:
        pivot_idx = _partition(arr, low, high)
        quick_sort(arr, low, pivot_idx - 1)
        quick_sort(arr, pivot_idx + 1, high)
    return arr


def _partition(arr: list, low: int, high: int) -> int:
    """Lomuto のパーティション"""
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


# --- 動作確認 ---
if __name__ == "__main__":
    data = [10, 7, 8, 9, 1, 5]
    print(quick_sort(data[:]))  # [1, 5, 7, 8, 9, 10]
```

### 6.4 ピボット選択戦略

ピボットの選択はクイックソートの性能に決定的な影響を与える。

```python
import random


def quick_sort_randomized(arr: list, low: int = 0, high: int = None) -> list:
    """ランダム化クイックソート

    ピボットをランダムに選択することで、
    特定の入力パターンによる最悪ケースを確率的に回避する。
    期待計算量は O(n log n)。
    """
    if high is None:
        high = len(arr) - 1
    if low < high:
        rand_idx = random.randint(low, high)
        arr[rand_idx], arr[high] = arr[high], arr[rand_idx]
        pivot_idx = _partition(arr, low, high)
        quick_sort_randomized(arr, low, pivot_idx - 1)
        quick_sort_randomized(arr, pivot_idx + 1, high)
    return arr


def partition_median_of_three(arr: list, low: int, high: int) -> int:
    """三値の中央値によるパーティション

    先頭・中央・末尾の3要素の中央値をピボットに選ぶ。
    ソート済み・逆順データでの最悪ケースを回避する。
    """
    mid = (low + high) // 2
    if arr[low] > arr[mid]:
        arr[low], arr[mid] = arr[mid], arr[low]
    if arr[low] > arr[high]:
        arr[low], arr[high] = arr[high], arr[low]
    if arr[mid] > arr[high]:
        arr[mid], arr[high] = arr[high], arr[mid]
    # 中央値(mid)をピボットとして high-1 に退避
    arr[mid], arr[high - 1] = arr[high - 1], arr[mid]
    pivot = arr[high - 1]

    i = low
    j = high - 1
    while True:
        i += 1
        while arr[i] < pivot:
            i += 1
        j -= 1
        while arr[j] > pivot:
            j -= 1
        if i >= j:
            break
        arr[i], arr[j] = arr[j], arr[i]
    arr[i], arr[high - 1] = arr[high - 1], arr[i]
    return i


# --- 動作確認 ---
if __name__ == "__main__":
    data = [10, 7, 8, 9, 1, 5]
    print(quick_sort_randomized(data[:]))  # [1, 5, 7, 8, 9, 10]
```

### 6.5 三方分割（Dutch National Flag）

重複要素が多い場合、通常のクイックソートは効率が悪い。Dijkstra の「オランダ国旗問題」を応用した三方分割により、ピボットと等しい要素をまとめて処理できる。

```python
def quick_sort_three_way(arr: list, low: int = 0, high: int = None) -> list:
    """三方分割クイックソート

    配列を「ピボット未満」「ピボットと等しい」「ピボット超過」
    の3つに分割する。重複要素が多い場合に特に有効。

    計算量:
        重複なし: O(n log n)
        重複多数: O(n) に近づく（等しい要素をスキップ）
    """
    if high is None:
        high = len(arr) - 1
    if low >= high:
        return arr

    pivot = arr[low]
    lt = low      # arr[low..lt-1]   < pivot
    i = low + 1   # arr[lt..i-1]    == pivot
    gt = high     # arr[gt+1..high]  > pivot

    while i <= gt:
        if arr[i] < pivot:
            arr[lt], arr[i] = arr[i], arr[lt]
            lt += 1
            i += 1
        elif arr[i] > pivot:
            arr[i], arr[gt] = arr[gt], arr[i]
            gt -= 1
        else:
            i += 1

    quick_sort_three_way(arr, low, lt - 1)
    quick_sort_three_way(arr, gt + 1, high)
    return arr


# --- 動作確認 ---
if __name__ == "__main__":
    data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    print(quick_sort_three_way(data))  # [1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9]
```

### 6.6 クイックソートの最悪ケース分析

| 入力パターン | 固定ピボット(末尾) | ランダムピボット | 三値中央値 |
|:---|:---|:---|:---|
| ソート済み | O(n^2) | O(n log n) 期待 | O(n log n) |
| 逆順 | O(n^2) | O(n log n) 期待 | O(n log n) |
| 全て同じ値 | O(n^2) | O(n^2) | O(n^2) |
| ランダム | O(n log n) | O(n log n) | O(n log n) |

全て同じ値の場合は三方分割を使えば O(n) になる。

---

## 7. ヒープソート（Heap Sort）

### 7.1 アルゴリズムの原理

最大ヒープ（親が子以上の完全二分木）を構築し、ルート（最大値）を末尾に移動してヒープサイズを縮小する操作を繰り返す。J. W. J. Williams が 1964 年に考案した。

### 7.2 ヒープの構造と配列表現

```
最大ヒープの配列表現:

インデックス:   0   1   2   3   4   5   6
配列:        [90, 70, 80, 30, 50, 40, 20]

ツリー表現:
                 90 (i=0)
               /    \
           70 (i=1)   80 (i=2)
           / \         / \
      30(i=3) 50(i=4) 40(i=5) 20(i=6)

親子の関係（0-indexed）:
  親:     (i - 1) // 2
  左の子: 2 * i + 1
  右の子: 2 * i + 2

ヒープソートの手順:
  1. 配列全体を最大ヒープに構築する（ボトムアップ）
  2. ルート（最大値）を末尾と交換する
  3. ヒープサイズを 1 縮小し、ルートを heapify する
  4. ヒープサイズが 1 になるまで 2-3 を繰り返す

ソート過程:
  [90, 70, 80, 30, 50, 40, 20]  ← 最大ヒープ
   90 と 20 を交換 → heapify
  [80, 70, 40, 30, 50, 20 | 90]
   80 と 20 を交換 → heapify
  [70, 50, 40, 30, 20 | 80, 90]
   ...繰り返し...
  [20, 30, 40, 50, 70, 80, 90]  ← ソート完了
```

### 7.3 実装

```python
def heap_sort(arr: list) -> list:
    """ヒープソート - 不安定・in-place・O(n log n) 保証

    最大ヒープを構築し、ルートを末尾に移動させることを繰り返す。
    最悪でも O(n log n) が保証され、O(1) 追加メモリで動作する。

    Args:
        arr: ソート対象のリスト

    Returns:
        ソート済みのリスト

    計算量:
        最良/平均/最悪: O(n log n)
        空間: O(1)
    """
    n = len(arr)

    # 最大ヒープ構築（ボトムアップ）
    for i in range(n // 2 - 1, -1, -1):
        _heapify(arr, n, i)

    # ルートを末尾に移動してヒープサイズを縮小
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        _heapify(arr, i, 0)

    return arr


def _heapify(arr: list, n: int, i: int) -> None:
    """ノード i を根とする部分木を最大ヒープ化する"""
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[left] > arr[largest]:
        largest = left
    if right < n and arr[right] > arr[largest]:
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        _heapify(arr, n, largest)


# --- 動作確認 ---
if __name__ == "__main__":
    data = [12, 11, 13, 5, 6, 7]
    print(heap_sort(data))  # [5, 6, 7, 11, 12, 13]

    data2 = [4, 10, 3, 5, 1]
    print(heap_sort(data2))  # [1, 3, 4, 5, 10]
```

### 7.4 ヒープ構築の計算量

ヒープ構築は直感的には O(n log n) に思えるが、実際には **O(n)** である。これはボトムアップ構築において、葉に近いノードほど heapify のコストが小さいためである。

```
高さ h のノード数と heapify コスト:

高さ 0 (葉):     n/2 個  x O(0) = 0
高さ 1:          n/4 個  x O(1) = n/4
高さ 2:          n/8 個  x O(2) = n/4
高さ 3:          n/16 個 x O(3) = 3n/16
  ...
高さ log n (根): 1 個    x O(log n) = log n

合計 = Sum_{h=0}^{log n} ceil(n / 2^{h+1}) * h
     = n * Sum_{h=0}^{inf} h / 2^{h+1}
     = n * 1
     = O(n)
```

### 7.5 ヒープソートの特徴

| 特性 | 詳細 |
|:---|:---|
| 最悪計算量保証 | 常に O(n log n) -- クイックソートの O(n^2) 退化がない |
| メモリ効率 | O(1) 追加メモリ -- マージソートの O(n) が不要 |
| 不安定 | 等しいキーの相対順序は保証されない |
| キャッシュ効率 | 悪い -- ヒープの親子アクセスがメモリ上で離れている |
| 実用性能 | 定数係数が大きく、同じ O(n log n) でもクイックソートより遅いことが多い |

---

## 8. 非比較ベースソート

### 8.1 概要

比較ベースソートには O(n log n) の理論的下限が存在するが、非比較ベースソートはこの制約を受けない。データの性質（整数、固定長、一様分布など）を利用して、条件付きで O(n) のソートを実現する。

### 8.2 計数ソート（Counting Sort）

要素の出現回数をカウントし、累積和で最終位置を決定する。

```
入力:  [4, 2, 2, 8, 3, 3, 1]
範囲:  1..8 (min=1, max=8, range=8)

Step 1: カウント配列を作成
  値:      1  2  3  4  5  6  7  8
  count: [ 1, 2, 2, 1, 0, 0, 0, 1 ]

Step 2: 累積和を計算
  値:      1  2  3  4  5  6  7  8
  count: [ 1, 3, 5, 6, 6, 6, 6, 7 ]
  意味: 値 k 以下の要素は count[k] 個

Step 3: 出力配列を構築（末尾から走査して安定性を保証）
  i=6: arr[6]=1 → output[0] = 1
  i=5: arr[5]=3 → output[4] = 3
  i=4: arr[4]=3 → output[3] = 3
  i=3: arr[3]=8 → output[6] = 8
  i=2: arr[2]=2 → output[2] = 2
  i=1: arr[1]=2 → output[1] = 2
  i=0: arr[0]=4 → output[5] = 4

出力: [1, 2, 2, 3, 3, 4, 8]
```

```python
def counting_sort(arr: list) -> list:
    """計数ソート - 安定・O(n + k)

    要素の出現回数をカウントし、累積和で位置を決定する。
    整数データで値の範囲 k が小さい場合に最適。

    Args:
        arr: 整数のリスト

    Returns:
        新しいソート済みリスト

    計算量:
        時間: O(n + k) -- k は値の範囲
        空間: O(n + k)
    """
    if not arr:
        return arr

    max_val = max(arr)
    min_val = min(arr)
    range_val = max_val - min_val + 1

    count = [0] * range_val
    output = [0] * len(arr)

    for num in arr:
        count[num - min_val] += 1

    for i in range(1, range_val):
        count[i] += count[i - 1]

    for i in range(len(arr) - 1, -1, -1):
        output[count[arr[i] - min_val] - 1] = arr[i]
        count[arr[i] - min_val] -= 1

    return output


# --- 動作確認 ---
if __name__ == "__main__":
    data = [4, 2, 2, 8, 3, 3, 1]
    print(counting_sort(data))  # [1, 2, 2, 3, 3, 4, 8]
```

### 8.3 基数ソート（Radix Sort）

各桁ごとに安定ソート（通常は計数ソート）を適用することで、複数桁の整数をソートする。最下位桁（LSD: Least Significant Digit）から処理する方法が一般的。

```
LSD 基数ソートの動作 (基数=10):

入力: [170, 45, 75, 90, 802, 24, 2, 66]

1の位でソート:
  17[0], 9[0]  → 0 のバケット
  80[2], [2]   → 2 のバケット
  2[4]         → 4 のバケット
  4[5], 7[5]   → 5 のバケット
  6[6]         → 6 のバケット
  結果: [170, 90, 802, 2, 24, 45, 75, 66]

10の位でソート:
  8[0]2, [0]2  → 0 のバケット
  [2]4         → 2 のバケット
  [4]5         → 4 のバケット
  [6]6         → 6 のバケット
  1[7]0, [7]5  → 7 のバケット
  [9]0         → 9 のバケット
  結果: [802, 2, 24, 45, 66, 170, 75, 90]

100の位でソート:
  [0]02, [0]24, [0]45, [0]66, [0]75, [0]90  → 0
  [1]70                                      → 1
  [8]02                                      → 8
  結果: [2, 24, 45, 66, 75, 90, 170, 802]
```

```python
def radix_sort(arr: list) -> list:
    """基数ソート (LSD) - 安定・O(d * (n + k))

    最下位桁から順に計数ソートを適用する。
    d は最大桁数、k は基数（通常 10）。

    Args:
        arr: 非負整数のリスト

    Returns:
        新しいソート済みリスト

    計算量:
        時間: O(d * (n + k))
        空間: O(n + k)
    """
    if not arr:
        return arr

    result = arr[:]
    max_val = max(result)

    exp = 1
    while max_val // exp > 0:
        result = _counting_sort_by_digit(result, exp)
        exp *= 10

    return result


def _counting_sort_by_digit(arr: list, exp: int) -> list:
    """特定の桁に基づく計数ソート"""
    n = len(arr)
    output = [0] * n
    count = [0] * 10

    for num in arr:
        digit = (num // exp) % 10
        count[digit] += 1

    for i in range(1, 10):
        count[i] += count[i - 1]

    for i in range(n - 1, -1, -1):
        digit = (arr[i] // exp) % 10
        output[count[digit] - 1] = arr[i]
        count[digit] -= 1

    return output


# --- 動作確認 ---
if __name__ == "__main__":
    data = [170, 45, 75, 90, 802, 24, 2, 66]
    print(radix_sort(data))  # [2, 24, 45, 66, 75, 90, 170, 802]
```

### 8.4 バケットソート（Bucket Sort）

入力が一様分布に従う場合に有効。値の範囲を等間隔のバケットに分割し、各バケット内を個別にソートする。

```python
def bucket_sort(arr: list, bucket_count: int = 10) -> list:
    """バケットソート - 安定（内部ソートが安定なら）

    値の範囲を等間隔のバケットに分割し、
    各バケット内を挿入ソートでソートする。
    一様分布の浮動小数点データに最適。

    Args:
        arr: 数値のリスト
        bucket_count: バケットの数

    Returns:
        新しいソート済みリスト

    計算量:
        平均: O(n + n^2/k + k) -- k=n なら O(n)
        最悪: O(n^2) -- 全要素が同一バケットに集中
        空間: O(n + k)
    """
    if not arr:
        return arr

    min_val = min(arr)
    max_val = max(arr)

    if min_val == max_val:
        return arr[:]

    range_val = max_val - min_val

    buckets: list[list] = [[] for _ in range(bucket_count)]

    for num in arr:
        idx = int((num - min_val) / range_val * (bucket_count - 1))
        buckets[idx].append(num)

    result = []
    for bucket in buckets:
        insertion_sort(bucket)
        result.extend(bucket)

    return result


# --- 動作確認 ---
if __name__ == "__main__":
    data = [0.897, 0.565, 0.656, 0.1234, 0.665, 0.3434]
    print(bucket_sort(data, 5))
    # [0.1234, 0.3434, 0.565, 0.656, 0.665, 0.897]
```

### 8.5 非比較ベースソートの比較表

| アルゴリズム | 計算量 | 空間 | 安定 | 適用条件 |
|:---|:---|:---|:---|:---|
| 計数ソート | O(n + k) | O(n + k) | Yes | 整数、値の範囲 k が小さい |
| 基数ソート | O(d(n + k)) | O(n + k) | Yes | 固定長キー、d 桁 |
| バケットソート | O(n) 平均 | O(n + k) | Yes* | 一様分布のデータ |

*内部ソートに安定ソートを使用した場合

---

## 9. 高度なソートアルゴリズム

### 9.1 シェルソート（Shell Sort）

Donald Shell が 1959 年に考案。挿入ソートの改良版で、離れた要素同士を比較・交換することで、要素の大きな移動を効率的に行う。ギャップ（間隔）を徐々に小さくしながら挿入ソートを繰り返す。

```
シェルソートの動作（ギャップ列: 4, 2, 1）:

初期: [35, 33, 42, 10, 14, 19, 27, 44, 26, 31]

ギャップ 4:
  グループ1: [35, 14, 26]  → [14, 26, 35]
  グループ2: [33, 19, 31]  → [19, 31, 33]
  グループ3: [42, 27]      → [27, 42]
  グループ4: [10, 44]      → [10, 44]
  結果: [14, 19, 27, 10, 26, 31, 42, 44, 35, 33]

ギャップ 2:
  グループ1: [14, 27, 26, 42, 35]  → [14, 26, 27, 35, 42]
  グループ2: [19, 10, 31, 44, 33]  → [10, 19, 31, 33, 44]
  結果: [14, 10, 26, 19, 27, 31, 35, 33, 42, 44]

ギャップ 1 (通常の挿入ソート):
  結果: [10, 14, 19, 26, 27, 31, 33, 35, 42, 44]
```

```python
def shell_sort(arr: list) -> list:
    """シェルソート - 不安定・in-place

    ギャップを縮小しながら挿入ソートを繰り返す。
    ギャップ列の選択により計算量が変わる。

    Knuth のギャップ列: 1, 4, 13, 40, 121, ...  (3h + 1)
    このギャップ列での計算量は O(n^{3/2})。

    Args:
        arr: ソート対象のリスト

    Returns:
        ソート済みのリスト

    計算量:
        Knuth列: O(n^{3/2})
        Sedgewick列: O(n^{4/3})
        空間: O(1)
    """
    n = len(arr)

    # Knuth のギャップ列を計算
    gap = 1
    while gap < n // 3:
        gap = gap * 3 + 1  # 1, 4, 13, 40, 121, ...

    while gap >= 1:
        # ギャップ付き挿入ソート
        for i in range(gap, n):
            key = arr[i]
            j = i
            while j >= gap and arr[j - gap] > key:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = key
        gap //= 3

    return arr


# --- 動作確認 ---
if __name__ == "__main__":
    data = [35, 33, 42, 10, 14, 19, 27, 44, 26, 31]
    print(shell_sort(data))  # [10, 14, 19, 26, 27, 31, 33, 35, 42, 44]
```

### 9.2 TimSort

Python の `list.sort()` と `sorted()`、Java の `Arrays.sort()` で使用されているハイブリッドソートアルゴリズム。Tim Peters が 2002 年に設計した。マージソートと挿入ソートを組み合わせ、実データのパターンを活用して高速化する。

```
TimSort の戦略:

1. 配列を "run" (既にソートされた部分列) に分割する
   入力: [1, 3, 5, 2, 4, 6, 8, 7, 9]
         |--run1--|  |---run2---|  |-run3-|
         [1, 3, 5]  [2, 4, 6, 8]  [7, 9]

2. 短い run は挿入ソートで minrun 以上の長さに拡張する
   (minrun は通常 32-64、n の上位6ビットから算出)

3. run をスタックに積み、マージルールに従ってマージする
   マージルール (不変条件):
     A > B + C  (3つ目と2つ目の合計より大きい)
     B > C      (2つ目が3つ目より大きい)
   この条件が崩れたら、隣接 run をマージする

4. galloping mode: マージ中に一方の配列から連続して
   要素が選ばれる場合、二分探索で一括処理する
```

**TimSort の設計思想:**

- **実データは完全にランダムではない**: 多くの実データには部分的な秩序（run）が存在する。TimSort はこれを検出して活用する。
- **小規模部分配列には挿入ソートが最適**: run が短い場合は挿入ソートで処理する。キャッシュ効率が良く、オーバーヘッドも小さい。
- **安定性の保証**: マージベースのため安定性を保持する。
- **最悪でも O(n log n)**: マージソートの最悪計算量保証を維持する。

### 9.3 Introsort（内省的ソート）

David Musser が 1997 年に考案。クイックソートを基本とし、再帰の深さが 2 * log(n) を超えたらヒープソートに切り替え、小規模部分配列には挿入ソートを使用するハイブリッドアルゴリズム。C++ の `std::sort` で使用されている。

```
Introsort の動作:

1. 最初はクイックソートで分割
2. 再帰深度を監視
   - 深度 < 2*log(n) → クイックソートを継続
   - 深度 >= 2*log(n) → ヒープソートに切り替え（最悪ケース回避）
3. 部分配列のサイズが閾値以下 → 挿入ソートに切り替え

結果:
  - 平均: クイックソートの高速性 O(n log n)
  - 最悪: ヒープソートの保証 O(n log n)
  - 小規模: 挿入ソートの低オーバーヘッド
```

```python
import math


def introsort(arr: list) -> list:
    """Introsort（内省的ソート）

    クイックソート + ヒープソート + 挿入ソートのハイブリッド。
    C++ の std::sort の基盤アルゴリズム。

    計算量:
        最良/平均/最悪: O(n log n)
        空間: O(log n)
    """
    max_depth = 2 * math.floor(math.log2(max(len(arr), 1)))
    _introsort_impl(arr, 0, len(arr) - 1, max_depth)
    return arr


def _introsort_impl(arr: list, low: int, high: int, depth_limit: int) -> None:
    """Introsort の内部実装"""
    INSERTION_THRESHOLD = 16

    while high - low + 1 > INSERTION_THRESHOLD:
        if depth_limit == 0:
            # 深度超過 → ヒープソートに切り替え
            sub = arr[low:high + 1]
            heap_sort(sub)
            arr[low:high + 1] = sub
            return

        depth_limit -= 1

        # 三値中央値でピボット選択
        mid = (low + high) // 2
        if arr[mid] < arr[low]:
            arr[low], arr[mid] = arr[mid], arr[low]
        if arr[high] < arr[low]:
            arr[low], arr[high] = arr[high], arr[low]
        if arr[high] < arr[mid]:
            arr[mid], arr[high] = arr[high], arr[mid]

        pivot = arr[mid]
        arr[mid], arr[high - 1] = arr[high - 1], arr[mid]

        # パーティション
        i = low
        j = high - 1
        while True:
            i += 1
            while arr[i] < pivot:
                i += 1
            j -= 1
            while arr[j] > pivot:
                j -= 1
            if i >= j:
                break
            arr[i], arr[j] = arr[j], arr[i]

        arr[i], arr[high - 1] = arr[high - 1], arr[i]

        # 短い方を再帰、長い方をループ（末尾再帰最適化）
        if i - low < high - i:
            _introsort_impl(arr, low, i - 1, depth_limit)
            low = i + 1
        else:
            _introsort_impl(arr, i + 1, high, depth_limit)
            high = i - 1

    # 閾値以下は挿入ソート
    for i in range(low + 1, high + 1):
        key = arr[i]
        j = i - 1
        while j >= low and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


# --- 動作確認 ---
if __name__ == "__main__":
    data = [5, 3, 8, 4, 2, 7, 1, 10, 6, 9]
    print(introsort(data))  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

### 9.4 外部ソート（External Sort）

メインメモリに収まりきらない大規模データをソートする手法。ディスクとメモリ間のI/Oを最小化することが鍵。

```
外部マージソートの手順:

1. 分割フェーズ:
   巨大ファイル (例: 100GB)
   → メモリに収まるチャンク (例: 1GB) に分割
   → 各チャンクをメモリ内でソートしてディスクに書き出す
   → 100個のソート済みチャンクファイルが生成される

2. マージフェーズ:
   ┌──────────┐  ┌──────────┐       ┌──────────┐
   │ Chunk 1  │  │ Chunk 2  │  ...  │ Chunk 100│
   └────┬─────┘  └────┬─────┘       └────┬─────┘
        │             │                   │
        v             v                   v
   ┌──────────────────────────────────────────┐
   │     k-way マージ (最小ヒープ使用)         │
   │     各チャンクの先頭要素をヒープに格納     │
   │     最小値を取り出し → 出力ファイルへ      │
   │     取り出したチャンクから次の要素を補充    │
   └──────────────────────┬───────────────────┘
                          │
                          v
                  ┌──────────────┐
                  │ ソート済み    │
                  │ 出力ファイル  │
                  └──────────────┘
```

---

## 10. 計算量比較表と用途別選択ガイド

### 10.1 全アルゴリズム計算量比較表

| アルゴリズム | 最良 | 平均 | 最悪 | 空間 | 安定 | in-place | 適応的 |
|:---|:---|:---|:---|:---|:---|:---|:---|
| バブルソート | O(n) | O(n^2) | O(n^2) | O(1) | Yes | Yes | Yes |
| 選択ソート | O(n^2) | O(n^2) | O(n^2) | O(1) | No | Yes | No |
| 挿入ソート | O(n) | O(n^2) | O(n^2) | O(1) | Yes | Yes | Yes |
| シェルソート | O(n log n) | O(n^{3/2}) | O(n^{3/2}) | O(1) | No | Yes | Yes |
| マージソート | O(n log n) | O(n log n) | O(n log n) | O(n) | Yes | No | No |
| クイックソート | O(n log n) | O(n log n) | O(n^2) | O(log n) | No | Yes | No |
| ヒープソート | O(n log n) | O(n log n) | O(n log n) | O(1) | No | Yes | No |
| TimSort | O(n) | O(n log n) | O(n log n) | O(n) | Yes | No | Yes |
| Introsort | O(n log n) | O(n log n) | O(n log n) | O(log n) | No | Yes | No |
| 計数ソート | O(n+k) | O(n+k) | O(n+k) | O(n+k) | Yes | No | -- |
| 基数ソート | O(d(n+k)) | O(d(n+k)) | O(d(n+k)) | O(n+k) | Yes | No | -- |
| バケットソート | O(n+k) | O(n+k) | O(n^2) | O(n+k) | Yes* | No | -- |

### 10.2 用途別選択ガイド

| 状況 | 推奨アルゴリズム | 理由 |
|:---|:---|:---|
| 小規模データ（n < 50） | 挿入ソート | オーバーヘッドが最小 |
| ほぼ整列済み | 挿入ソート / TimSort | 適応的で O(n) に近い |
| 汎用（ライブラリ） | TimSort | Python/Java 標準、安定 + 高速 |
| メモリ制約あり | ヒープソート | O(1) 追加メモリ、O(n log n) 保証 |
| 整数・範囲が小さい | 計数ソート | O(n+k) で最速 |
| 平均性能重視 | クイックソート / Introsort | 定数係数が小さい |
| 安定性必須 | マージソート / TimSort | O(n log n) かつ安定 |
| 外部ソート（大容量） | 外部マージソート | 逐次アクセスに強い |
| C++ 標準ライブラリ | Introsort | std::sort の基盤 |
| 重複が多いデータ | 三方分割クイックソート | 重複要素を効率的にスキップ |
| 固定長の文字列/数値 | 基数ソート | O(d*n) で比較不要 |

---

## 11. アンチパターン

### アンチパターン1: ピボット選択の固定化

```python
# BAD: 常に末尾をピボットにする
# ソート済み配列で O(n^2) に退化する
def bad_quicksort(arr, low, high):
    pivot = arr[high]  # ソート済みで最悪ケース
    # ...パーティション処理...


# GOOD: 三値の中央値 or ランダム選択
import random

def good_quicksort(arr, low, high):
    if low < high:
        # ランダムピボット選択
        rand_idx = random.randint(low, high)
        arr[rand_idx], arr[high] = arr[high], arr[rand_idx]
        # ...パーティション処理...
```

**問題点**: ソート済み・逆順・ほぼソート済みのデータで O(n^2) に退化する。本番システムでは入力データのパターンを事前に知ることができないため、固定ピボットは危険。

### アンチパターン2: 計数ソートの値域無視

```python
# BAD: 値域が巨大な場合に計数ソートを使う
data = [1, 1000000000, 2]  # count 配列が 10^9 サイズ!
counting_sort(data)         # メモリ不足でクラッシュ

# GOOD: 値域を確認してからアルゴリズムを選択
def smart_sort(data: list) -> list:
    if not data:
        return data
    range_val = max(data) - min(data) + 1
    if range_val <= len(data) * 10:
        return counting_sort(data)  # 値域が小さい → 計数ソート
    else:
        return sorted(data)  # 値域が大きい → 比較ソート
```

**問題点**: 計数ソートの空間計算量は O(n + k) であり、k（値の範囲）が巨大な場合はメモリを大量に消費する。値域の確認なしに使用するのは危険。

### アンチパターン3: 安定性を無視した選択

```python
# BAD: レコードのソートに不安定ソートを使い順序が崩れる
students = [(3, "Alice"), (1, "Bob"), (3, "Charlie")]
# ヒープソートを使うと同一キーの Alice, Charlie の順序が保証されない

# GOOD: 安定ソートを使う（Python の sort は TimSort で安定）
students.sort(key=lambda x: x[0])
# [(1, "Bob"), (3, "Alice"), (3, "Charlie")]  ← 元の順序が保持される
```

**問題点**: 多段ソート（まず名前順、次に成績順など）で不安定ソートを使うと、先にソートした順序が崩れる。データベースの ORDER BY 相当の処理では安定性が不可欠。

### アンチパターン4: 再帰深度の無制限

```python
# BAD: 再帰深度を制限しないクイックソート
# 最悪ケースで再帰深度が O(n) になり、スタックオーバーフローが発生
def bad_quicksort(arr, low, high):
    if low < high:
        p = partition(arr, low, high)
        bad_quicksort(arr, low, p - 1)   # 左側を再帰
        bad_quicksort(arr, p + 1, high)  # 右側を再帰
        # n=10000 のソート済みデータで RecursionError!

# GOOD: Introsort パターンで深度制限 + 末尾再帰最適化
import math

def good_quicksort(arr, low, high, depth=None):
    if depth is None:
        depth = 2 * math.floor(math.log2(max(len(arr), 1)))
    while low < high:
        if depth == 0:
            # ヒープソートにフォールバック
            sub = arr[low:high + 1]
            heap_sort(sub)
            arr[low:high + 1] = sub
            return
        depth -= 1
        p = _partition(arr, low, high)
        # 短い方を再帰、長い方をループ（末尾再帰最適化）
        if p - low < high - p:
            good_quicksort(arr, low, p - 1, depth)
            low = p + 1
        else:
            good_quicksort(arr, p + 1, high, depth)
            high = p - 1
```

---

## 12. 演習問題

### 基礎レベル

**問題 B1: ソートアルゴリズムの手動トレース**

以下の配列 `[6, 3, 8, 2, 7, 4]` に対して、挿入ソートの各ステップを手動でトレースせよ。各ステップでの配列の状態を書き出すこと。

<details>
<summary>解答例</summary>

```
初期: [6, 3, 8, 2, 7, 4]

i=1: key=3, 6>3 → シフト
  [3, 6, 8, 2, 7, 4]

i=2: key=8, 6<8 → 移動なし
  [3, 6, 8, 2, 7, 4]

i=3: key=2, 8>2 → シフト, 6>2 → シフト, 3>2 → シフト
  [2, 3, 6, 8, 7, 4]

i=4: key=7, 8>7 → シフト, 6<7 → 停止
  [2, 3, 6, 7, 8, 4]

i=5: key=4, 8>4 → シフト, 7>4 → シフト, 6>4 → シフト, 3<4 → 停止
  [2, 3, 4, 6, 7, 8]
```

</details>

**問題 B2: 安定性の判定**

以下のソートアルゴリズムのうち、安定なものを全て選べ。

(a) バブルソート  (b) 選択ソート  (c) 挿入ソート  (d) マージソート  (e) クイックソート  (f) ヒープソート  (g) 計数ソート

<details>
<summary>解答</summary>

安定なソート: **(a) バブルソート、(c) 挿入ソート、(d) マージソート、(g) 計数ソート**

不安定なソート: (b) 選択ソート、(e) クイックソート、(f) ヒープソート

選択ソートは離れた位置の要素を交換するため不安定。クイックソートはパーティション操作で相対順序が崩れる。ヒープソートはヒープの再構築で相対順序が崩れる。

</details>

**問題 B3: 計算量の比較**

n = 1,000,000 のランダムデータをソートする場合、バブルソート O(n^2) とマージソート O(n log n) のおおよその計算ステップ数をそれぞれ見積もれ。

<details>
<summary>解答</summary>

```
バブルソート: n^2 = (10^6)^2 = 10^12 ステップ
マージソート: n * log2(n) = 10^6 * 20 = 2 * 10^7 ステップ

差: 10^12 / (2 * 10^7) = 50,000 倍

仮に 1 ステップ = 1ナノ秒 とすると:
  バブルソート: 10^12 ns = 1000 秒 (約 16.7 分)
  マージソート: 2 * 10^7 ns = 0.02 秒

この差がアルゴリズムの計算量クラスの重要性を示している。
```

</details>

### 応用レベル

**問題 A1: k 番目に小さい要素の探索**

ソートを使わずに、配列の中から k 番目に小さい要素を平均 O(n) で見つけるアルゴリズムを実装せよ（Quickselect アルゴリズム）。

<details>
<summary>解答例</summary>

```python
import random


def quickselect(arr: list, k: int) -> int:
    """Quickselect - 平均 O(n) で k 番目に小さい要素を見つける

    クイックソートのパーティションを利用し、
    必要な側だけを再帰的に処理することで計算量を削減する。

    Args:
        arr: 数値のリスト
        k: 1-indexed で何番目に小さい要素を求めるか

    Returns:
        k 番目に小さい要素
    """
    if k < 1 or k > len(arr):
        raise ValueError(f"k={k} is out of range for array of size {len(arr)}")

    arr = arr[:]  # 元の配列を変更しないためにコピー
    return _quickselect(arr, 0, len(arr) - 1, k - 1)


def _quickselect(arr: list, low: int, high: int, k: int) -> int:
    if low == high:
        return arr[low]

    # ランダムピボット
    pivot_idx = random.randint(low, high)
    arr[pivot_idx], arr[high] = arr[high], arr[pivot_idx]

    # パーティション
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    pivot_pos = i + 1

    if k == pivot_pos:
        return arr[pivot_pos]
    elif k < pivot_pos:
        return _quickselect(arr, low, pivot_pos - 1, k)
    else:
        return _quickselect(arr, pivot_pos + 1, high, k)


# --- 動作確認 ---
if __name__ == "__main__":
    data = [7, 10, 4, 3, 20, 15]
    print(quickselect(data, 3))  # 7 (3番目に小さい要素)
    print(quickselect(data, 1))  # 3 (最小値)
    print(quickselect(data, 6))  # 20 (最大値)
```

</details>

**問題 A2: マージソートを用いた転倒数の計算**

配列の転倒数（inversion count: i < j かつ arr[i] > arr[j] となるペアの数）をマージソートを改造して O(n log n) で計算する関数を実装せよ。

<details>
<summary>解答例</summary>

```python
def count_inversions(arr: list) -> tuple[list, int]:
    """マージソートを利用して転倒数を O(n log n) で計算する

    マージ時に右側配列の要素が先に出力される回数を数える。
    右側要素が選ばれたとき、左側の残り要素数が転倒数への寄与。

    Args:
        arr: 数値のリスト

    Returns:
        (ソート済みリスト, 転倒数) のタプル
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
            merged.append(right[j])
            # 左側の残り要素数が全て転倒ペアを形成
            inversions += len(left) - i
            j += 1

    merged.extend(left[i:])
    merged.extend(right[j:])

    return merged, inversions


# --- 動作確認 ---
if __name__ == "__main__":
    data = [2, 4, 1, 3, 5]
    sorted_arr, inv_count = count_inversions(data)
    print(f"ソート済み: {sorted_arr}")  # [1, 2, 3, 4, 5]
    print(f"転倒数: {inv_count}")       # 3
    # 転倒ペア: (2,1), (4,1), (4,3)

    data2 = [5, 4, 3, 2, 1]
    _, inv_count2 = count_inversions(data2)
    print(f"逆順の転倒数: {inv_count2}")  # 10 = 5*4/2
```

</details>

### 発展レベル

**問題 C1: 比較ベースソートの下限証明**

決定木モデルを用いて、比較ベースソートの最悪計算量が Omega(n log n) であることを証明せよ。

<details>
<summary>解答の要点</summary>

**証明の概要:**

1. n 個の要素のソートは、n! 通りの順列のうちどれが入力かを特定する問題と等価である。

2. 比較ベースのアルゴリズムは二分決定木としてモデル化できる。各内部ノードは「a_i < a_j ?」という比較を表し、YES/NO の2つの子を持つ。

3. 決定木の各葉は、ある特定の順列に対応する出力を表す。正しくソートするには、全ての n! 通りの順列に対して異なる葉が必要。

4. したがって、決定木の葉の数 L >= n! である。

5. 高さ h の二分木の葉の数は最大 2^h であるから:
   2^h >= L >= n!
   h >= log2(n!)

6. スターリングの近似 n! >= (n/e)^n より:
   h >= log2((n/e)^n) = n * log2(n/e) = n * (log2(n) - log2(e))
   h = Omega(n log n)

7. 決定木の高さは最悪ケースの比較回数に対応するため、比較ベースソートの最悪計算量は Omega(n log n)。

この下限は情報理論的な議論であり、特定のアルゴリズムに依存しない普遍的な結果である。

</details>

---

## 13. FAQ

### Q1: Python の `sort()` と `sorted()` は何のアルゴリズム？

**A:** TimSort（マージソート + 挿入ソートのハイブリッド）を使用している。安定で、最良 O(n)、平均・最悪 O(n log n)。実データのパターン（run: 既にソートされた部分列）を検出して効率化する。`sort()` は in-place で元のリストを変更し、`sorted()` は新しいリストを返す。Java の `Arrays.sort()` もオブジェクト配列に TimSort を使用する（プリミティブ型には Dual-Pivot Quicksort を使用）。

### Q2: O(n log n) より速いソートは存在するか？

**A:** 比較ベースのソートでは O(n log n) が理論的下限である（決定木の高さによる証明）。ただし、非比較ベースのソート（計数ソート、基数ソート、バケットソート）はこの制約を受けず、データの性質に応じて O(n) を達成できる。ただし、これらは入力データに対する前提条件（整数であること、値の範囲が限定的であること、等）が必要である。

### Q3: 実務で自前ソートを実装すべきか？

**A:** 通常は NO。言語標準のソート（Python の TimSort、C++ の Introsort、Java の Dual-Pivot Quicksort 等）は高度に最適化されており、自前実装より高速かつ堅牢。自前実装が正当化されるのは以下の限定的な場面のみ:
- 外部ソート（ディスク上の大規模データ）
- 特殊なキー関数やカスタム比較が必要な場合
- 組み込みシステムでメモリ制約が極めて厳しい場合
- 教育・研究目的

### Q4: クイックソートとマージソートの使い分けは？

**A:** クイックソートは平均的に高速（キャッシュ効率が良く、定数係数が小さい）だが最悪 O(n^2) のリスクがある。マージソートは O(n log n) が保証され安定だが、O(n) の追加メモリが必要。実用的な指針:
- **安定性が必要** → マージソート / TimSort
- **メモリ制約がある** → クイックソート（Introsort で最悪ケースを回避）
- **外部ソート** → マージソート（逐次アクセスパターンがディスクI/Oに適合）
- **汎用的な選択** → 言語標準のソート関数を使用する

### Q5: なぜ C++ は不安定ソート（std::sort）をデフォルトにしているのか？

**A:** `std::sort` は Introsort をベースとしており、in-place（O(log n) の追加メモリ）で動作する。安定性を保証するには O(n) の追加メモリが必要なマージベースのアルゴリズムが必要で、性能とメモリ効率のトレードオフから不安定ソートがデフォルトに選ばれた。安定ソートが必要な場合は `std::stable_sort`（マージソートベース）を使用する。

### Q6: ソートの並列化はどのように行うか？

**A:** 主なアプローチは以下の通り:
- **並列マージソート**: 分割された部分配列を独立したスレッドでソートし、マージする。分割フェーズは自然に並列化でき、マージフェーズも並列マージアルゴリズムが存在する。
- **並列クイックソート**: パーティション後の左右を別スレッドで処理する。ただしパーティション自体の並列化は難しい。
- **ソーティングネットワーク**: Bitonic Sort や Odd-Even Merge Sort など、比較の順序が入力に依存しないアルゴリズム。GPU での並列ソートに適している。
- **サンプルソート**: データからサンプルを取り出してピボットの組を決定し、データを均等に分割して各プロセッサに配分する。分散環境向け。

---

## 14. まとめ

### 14.1 要点の整理

| 項目 | 要点 |
|:---|:---|
| O(n^2) ソート | バブル・選択・挿入は教育的、小規模データや部分ソートに有用 |
| O(n log n) ソート | マージ・クイック・ヒープが実用の中心 |
| 非比較ソート | 計数・基数・バケットは条件付きで O(n) |
| ハイブリッドソート | TimSort（マージ+挿入）、Introsort（クイック+ヒープ+挿入）が現代の標準 |
| 安定性 | レコードの相対順序を保持するかどうか。マージ・挿入・計数・TimSort は安定 |
| 実務選択 | 言語標準ソート（TimSort / Introsort）が第一選択、特殊要件時のみ自前実装 |
| ピボット戦略 | ランダム化 or 三値中央値で最悪ケースを回避 |
| 理論的下限 | 比較ベースソートは O(n log n) が下限（決定木モデルによる証明） |

### 14.2 アルゴリズム選択のフローチャート

```
ソートアルゴリズム選択フロー:

                        データの性質は?
                       /              \
                整数 (範囲小)        一般的
                     |                  |
               計数/基数ソート     データサイズは?
                                  /     |      \
                              小(n<50)  中      大(外部)
                                |       |        |
                            挿入ソート  |    外部マージソート
                                       |
                                安定性は必要?
                               /            \
                             Yes             No
                              |               |
                         TimSort /      メモリ制約は?
                       マージソート     /          \
                                    厳しい       余裕あり
                                      |            |
                                 ヒープソート   Introsort /
                                              クイックソート
```

### 14.3 学習のロードマップ

```
Level 1 (入門):
  挿入ソート → バブルソート → 選択ソート
  ↓
Level 2 (中級):
  マージソート → クイックソート → ヒープソート
  ↓
Level 3 (上級):
  計数/基数/バケットソート → シェルソート
  ↓
Level 4 (発展):
  TimSort → Introsort → 外部ソート → 並列ソート
  ↓
Level 5 (理論):
  計算量の下限証明 → 情報理論的最適性 → 適応的ソートの理論
```

---

## 次に読むべきガイド

- [探索アルゴリズム](./01-searching.md) -- ソート済みデータに対する効率的な探索
- [分割統治法](./06-divide-conquer.md) -- マージソート・クイックソートの設計パラダイム
- [動的計画法](./04-dynamic-programming.md) -- 最適部分構造を活用する別のパラダイム

---

## 15. 参考文献

1. Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. -- 第2章（挿入ソート）、第6章（ヒープソート）、第7章（クイックソート）、第8章（線形時間ソート）
2. Sedgewick, R. & Wayne, K. (2011). *Algorithms* (4th ed.). Addison-Wesley. -- Part 2: Sorting（マージソート、クイックソート、優先度キュー、応用）
3. Knuth, D. E. (1998). *The Art of Computer Programming, Volume 3: Sorting and Searching* (2nd ed.). Addison-Wesley. -- ソートアルゴリズムの網羅的な分析と歴史
4. Python Documentation. "Sorting HOW TO." https://docs.python.org/3/howto/sorting.html -- Python のソート機能の公式ガイド
5. McIlroy, P. (1993). "Optimistic Sorting and Information Theoretic Complexity." *Proceedings of the Fourth Annual ACM-SIAM Symposium on Discrete Algorithms (SODA)*. -- TimSort の理論的基盤
6. Musser, D. R. (1997). "Introspective Sorting and Selection Algorithms." *Software: Practice and Experience*, 27(8), 983-993. -- Introsort の原論文
7. Peters, T. (2002). "[Python-Dev] Sorting." https://mail.python.org/pipermail/python-dev/2002-July/026837.html -- TimSort の設計文書

---
