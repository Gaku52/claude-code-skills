# 配列と文字列 — 動的配列・文字列アルゴリズム・二次元配列の完全ガイド

> プログラミングの最も基本的なデータ構造である配列と文字列を深く理解し、動的配列の仕組み、文字列操作の計算量、二次元配列の走査パターン、Two Pointers・Sliding Window 等の頻出技法までを体系的に学ぶ。

---


## この章で学ぶこと

- [ ] 基本概念と用語の理解
- [ ] 実装パターンとベストプラクティスの習得
- [ ] 実務での適用方法の把握
- [ ] トラブルシューティングの基本

---

## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解

---

## 目次

1. [配列の基本 — メモリモデルとアドレス計算](#1-配列の基本--メモリモデルとアドレス計算)
2. [動的配列の内部実装 — 成長戦略と償却解析](#2-動的配列の内部実装--成長戦略と償却解析)
3. [多次元配列 — 行優先と列優先のメモリ配置](#3-多次元配列--行優先と列優先のメモリ配置)
4. [文字列の基本 — 不変性・エンコーディング・内部表現](#4-文字列の基本--不変性エンコーディング内部表現)
5. [配列の基本アルゴリズム — 回転・マージ・パーティション](#5-配列の基本アルゴリズム--回転マージパーティション)
6. [文字列アルゴリズム — 回文・アナグラム・パターン検索](#6-文字列アルゴリズム--回文アナグラムパターン検索)
7. [Two Pointers と Sliding Window](#7-two-pointers-と-sliding-window)
8. [二次元配列の走査パターン — スパイラル・対角線・回転](#8-二次元配列の走査パターン--スパイラル対角線回転)
9. [比較表と計算量チートシート](#9-比較表と計算量チートシート)
10. [アンチパターンと正しい書き方](#10-アンチパターンと正しい書き方)
11. [演習問題 — 基礎・応用・発展の3段階](#11-演習問題--基礎応用発展の3段階)
12. [FAQ — よくある質問と回答](#12-faq--よくある質問と回答)
13. [まとめと次のステップ](#13-まとめと次のステップ)
14. [参考文献](#14-参考文献)

---

## 1. 配列の基本 — メモリモデルとアドレス計算

### 1.1 配列とは何か

配列 (Array) は、同一型の要素を連続したメモリ領域に格納するデータ構造である。
コンピュータサイエンスにおいて最も基本的であり、かつ最も重要なデータ構造の1つである。

配列が重要である理由:

- **O(1) のランダムアクセス**: インデックスから直接アドレスを計算できる
- **キャッシュ効率**: 連続メモリ配置により CPU キャッシュのヒット率が高い
- **他のデータ構造の基盤**: ハッシュテーブル、ヒープ、動的配列など多くの構造が配列上に構築される
- **言語レベルの最適化**: ほぼすべてのプログラミング言語がネイティブサポートを提供

### 1.2 メモリレイアウトとアドレス計算

配列の最大の特徴は、要素が連続したメモリアドレスに配置されることである。
これにより、任意のインデックスへの O(1) アクセスが可能になる。

```
静的配列 int arr[5] のメモリレイアウト (各要素 4 バイト):

  ベースアドレス: 0x1000

  アドレス:  0x1000  0x1004  0x1008  0x100C  0x1010
            ┌───────┬───────┬───────┬───────┬───────┐
            │  10   │  20   │  30   │  40   │  50   │
            └───────┴───────┴───────┴───────┴───────┘
  index:      [0]     [1]     [2]     [3]     [4]

  アドレス計算式:
    arr[i] のアドレス = base_address + i * sizeof(element)
    arr[3] のアドレス = 0x1000 + 3 * 4 = 0x100C

  この計算はハードウェアレベルで 1 命令で完了する → O(1)
```

### 1.3 静的配列 vs 動的配列

静的配列はコンパイル時にサイズが決定し、スタック上に配置されることが多い。
動的配列は実行時にサイズを変更でき、ヒープ上に配置される。

```
静的配列 (C言語):
  int arr[5];                   // スタック上に 20 バイト確保
  ┌─────────────────────────┐
  │  スタック領域 (固定)      │
  │  arr: [_][_][_][_][_]    │   サイズ変更不可
  └─────────────────────────┘

動的配列 (C言語):
  int *arr = malloc(5 * sizeof(int));  // ヒープ上に確保
  ┌─────────────────────────┐
  │  スタック領域             │
  │  arr: ポインタ ──────────┼──┐
  └─────────────────────────┘  │
                                ↓
  ┌─────────────────────────────────┐
  │  ヒープ領域                      │
  │  [_][_][_][_][_][...余裕...]    │   realloc で拡張可能
  └─────────────────────────────────┘
```

### 1.4 各言語における配列の実現

言語によって配列の実現方法は大きく異なる:

| 言語 | 静的配列 | 動的配列 | 特徴 |
|------|---------|---------|------|
| C | `int arr[N]` | `malloc` + `realloc` | 境界チェックなし、最高速 |
| C++ | `std::array<T,N>` | `std::vector<T>` | RAII による自動メモリ管理 |
| Java | `int[]` | `ArrayList<Integer>` | 境界チェックあり、プリミティブ型は直接格納 |
| Python | なし | `list` | 全要素がオブジェクトポインタ |
| Go | `[N]T` | スライス `[]T` | 値型配列 + 参照型スライス |
| Rust | `[T; N]` | `Vec<T>` | 所有権システムによる安全性保証 |

### 1.5 コード例: 配列の基本操作と計算量の確認

```python
"""
配列の基本操作と各操作の計算量を確認するプログラム
"""
import time
import sys


def demonstrate_array_basics():
    """配列（Python list）の基本操作デモ"""

    # === 生成 ===
    arr = [10, 20, 30, 40, 50]
    print(f"配列: {arr}")
    print(f"長さ: {len(arr)}")
    print(f"メモリサイズ: {sys.getsizeof(arr)} bytes")
    print()

    # === O(1) ランダムアクセス ===
    print(f"arr[0] = {arr[0]}")   # 先頭要素
    print(f"arr[2] = {arr[2]}")   # 中間要素
    print(f"arr[-1] = {arr[-1]}") # 末尾要素
    print()

    # === O(1) 償却の末尾追加 ===
    arr.append(60)
    print(f"append(60) 後: {arr}")

    # === O(n) の先頭挿入 ===
    arr.insert(0, 5)
    print(f"insert(0, 5) 後: {arr}")
    # 全要素を1つ右にシフトする必要があるため O(n)

    # === O(n) の中間削除 ===
    arr.pop(3)
    print(f"pop(3) 後: {arr}")
    # 削除位置以降の要素を左にシフトするため O(n)

    # === O(1) の末尾削除 ===
    arr.pop()
    print(f"pop() 後: {arr}")
    print()

    # === スライス操作 ===
    print(f"arr[1:4] = {arr[1:4]}")     # O(k) k=スライス長
    print(f"arr[::2] = {arr[::2]}")     # O(n/2)
    print(f"arr[::-1] = {arr[::-1]}")   # O(n) 反転コピー


def measure_append_performance():
    """動的配列の append 性能を計測する"""
    sizes = [10_000, 100_000, 1_000_000]

    for n in sizes:
        start = time.perf_counter()
        arr = []
        for i in range(n):
            arr.append(i)
        elapsed = time.perf_counter() - start
        print(f"n={n:>10,}: append {n}回 = {elapsed:.4f}秒")
        # 結果は n にほぼ比例する（append 1回あたり O(1) 償却）


def measure_insert_front_performance():
    """先頭挿入の性能を計測する（O(n) × n = O(n^2)）"""
    sizes = [10_000, 50_000, 100_000]

    for n in sizes:
        start = time.perf_counter()
        arr = []
        for i in range(n):
            arr.insert(0, i)
        elapsed = time.perf_counter() - start
        print(f"n={n:>10,}: insert(0) {n}回 = {elapsed:.4f}秒")
        # n が 5 倍になると時間は約 25 倍（O(n^2) の特徴）


if __name__ == "__main__":
    print("=== 配列の基本操作 ===")
    demonstrate_array_basics()
    print()
    print("=== append 性能計測 ===")
    measure_append_performance()
    print()
    print("=== insert(0) 性能計測 ===")
    measure_insert_front_performance()
```

---

## 2. 動的配列の内部実装 — 成長戦略と償却解析

### 2.1 動的配列の仕組み

動的配列は、固定サイズの内部配列を持ち、容量が足りなくなったときに
より大きな配列を確保してデータをコピーする。この「リサイズ」操作は
O(n) かかるが、頻度が低いため、append 全体としては償却 O(1) となる。

```
動的配列の成長過程 (成長率 2x の場合):

Step 1: 初期状態 (容量=1)
  [A]            使用: 1/1 (100%)

Step 2: append(B) → 容量不足! リサイズ (容量 1→2)
  新配列を確保 → [A] をコピー → [B] を追加
  [A][B]          使用: 2/2 (100%)

Step 3: append(C) → 容量不足! リサイズ (容量 2→4)
  新配列を確保 → [A][B] をコピー → [C] を追加
  [A][B][C][ ]    使用: 3/4 (75%)

Step 4: append(D) → 空きあり、そのまま追加
  [A][B][C][D]    使用: 4/4 (100%)

Step 5: append(E) → 容量不足! リサイズ (容量 4→8)
  新配列を確保 → [A][B][C][D] をコピー → [E] を追加
  [A][B][C][D][E][ ][ ][ ]    使用: 5/8 (62.5%)

Step 6-8: append(F), append(G), append(H) → 空きあり
  [A][B][C][D][E][F][G][H]    使用: 8/8 (100%)

Step 9: append(I) → 容量不足! リサイズ (容量 8→16)
  [A][B][C][D][E][F][G][H][I][ ][ ][ ][ ][ ][ ][ ]  使用: 9/16 (56.25%)
```

### 2.2 成長率の選択と各言語の実装

成長率は空間効率と時間効率のトレードオフである:

| 実装 | 成長率 | 特徴 |
|------|--------|------|
| Python `list` | 約 1.125x | メモリ効率重視、細かいステップで成長 |
| Java `ArrayList` | 1.5x | バランス型 |
| C++ `std::vector` (GCC) | 2x | 速度重視、メモリ使用量は最大 2 倍 |
| C++ `std::vector` (MSVC) | 1.5x | バランス型 |
| Go スライス | 2x (小), 1.25x (大) | サイズに応じて成長率を変化 |
| Rust `Vec` | 2x | 速度重視 |

### 2.3 償却解析 — なぜ append は O(1) と言えるのか

償却解析 (Amortized Analysis) は、個々の操作ではなく一連の操作全体の
平均コストを分析する手法である。

**集計法 (Aggregate Method)**:

n 回の append を考える (成長率 2x):
- リサイズが発生するのは、サイズが 1, 2, 4, 8, ..., 2^k (2^k <= n) のとき
- コピーコストの合計: 1 + 2 + 4 + ... + 2^k = 2^(k+1) - 1 < 2n
- n 回の append 全体のコスト: n (追加) + 2n (コピー) = 3n
- 1 回あたりの償却コスト: 3n / n = O(1)

**バンカー法 (Accounting Method)**:

各 append に「3 コイン」を課金する:
- 1 コイン: 要素の追加に使用
- 2 コイン: 貯金（将来のリサイズのために貯める）

リサイズ時、容量 m から 2m に拡張する場合:
- m 個の要素をコピーする必要がある
- 前回のリサイズ以降に追加された m 個の要素が、各 2 コインずつ貯金している
- 合計 2m コインあるが、コピーには m コインで足りる

したがって、各操作の償却コストは O(1) である。

### 2.4 コード例: 動的配列の自作実装

```python
"""
動的配列 (DynamicArray) の自作実装
Python の list の内部動作を理解するための教育用コード
"""
import ctypes


class DynamicArray:
    """成長率 2x の動的配列"""

    def __init__(self):
        self._size = 0          # 実際の要素数
        self._capacity = 1      # 内部配列の容量
        self._array = self._make_array(self._capacity)
        self._resize_count = 0  # リサイズ回数の追跡

    def _make_array(self, capacity):
        """指定容量の内部配列を確保する"""
        return (capacity * ctypes.py_object)()

    def __len__(self):
        """要素数を返す — O(1)"""
        return self._size

    def __getitem__(self, index):
        """インデックスアクセス — O(1)"""
        if not 0 <= index < self._size:
            raise IndexError(f"index {index} out of range [0, {self._size})")
        return self._array[index]

    def __setitem__(self, index, value):
        """インデックスによる代入 — O(1)"""
        if not 0 <= index < self._size:
            raise IndexError(f"index {index} out of range [0, {self._size})")
        self._array[index] = value

    def append(self, value):
        """末尾に要素を追加 — O(1) 償却"""
        if self._size == self._capacity:
            self._resize(2 * self._capacity)
        self._array[self._size] = value
        self._size += 1

    def insert(self, index, value):
        """指定位置に要素を挿入 — O(n)"""
        if not 0 <= index <= self._size:
            raise IndexError(f"index {index} out of range [0, {self._size}]")
        if self._size == self._capacity:
            self._resize(2 * self._capacity)
        # index 以降の要素を右にシフト
        for i in range(self._size, index, -1):
            self._array[i] = self._array[i - 1]
        self._array[index] = value
        self._size += 1

    def pop(self, index=None):
        """要素を削除して返す — O(1) 末尾, O(n) それ以外"""
        if self._size == 0:
            raise IndexError("pop from empty array")
        if index is None:
            index = self._size - 1
        if not 0 <= index < self._size:
            raise IndexError(f"index {index} out of range [0, {self._size})")

        value = self._array[index]
        # index 以降の要素を左にシフト
        for i in range(index, self._size - 1):
            self._array[i] = self._array[i + 1]
        self._size -= 1

        # 使用率が 25% 以下になったら縮小
        if self._size > 0 and self._size <= self._capacity // 4:
            self._resize(self._capacity // 2)

        return value

    def _resize(self, new_capacity):
        """内部配列のリサイズ — O(n)"""
        new_array = self._make_array(new_capacity)
        for i in range(self._size):
            new_array[i] = self._array[i]
        self._array = new_array
        self._capacity = new_capacity
        self._resize_count += 1

    def __repr__(self):
        items = [str(self._array[i]) for i in range(self._size)]
        return (f"DynamicArray([{', '.join(items)}], "
                f"size={self._size}, capacity={self._capacity}, "
                f"resizes={self._resize_count})")


def demo_dynamic_array():
    """DynamicArray の動作デモ"""
    da = DynamicArray()
    print(f"初期状態: {da}")

    # 要素を追加しながらリサイズの過程を観察
    for i in range(1, 17):
        da.append(i * 10)
        print(f"append({i*10:>3}): size={len(da):>2}, "
              f"capacity={da._capacity:>2}, resizes={da._resize_count}")

    print(f"\n最終状態: {da}")
    print(f"da[5] = {da[5]}")

    # 要素の削除
    removed = da.pop()
    print(f"\npop() = {removed}")
    removed = da.pop(0)
    print(f"pop(0) = {removed}")
    print(f"削除後: {da}")


if __name__ == "__main__":
    demo_dynamic_array()
```

### 2.5 動的配列の縮小戦略

動的配列は拡張だけでなく、要素数が大幅に減少した場合に縮小も行う。
ただし、縮小の閾値は拡張の逆にしてはならない。

```
危険な設計（スラッシング問題）:
  成長率 2x、縮小閾値 50% の場合

  容量 8, 要素 4: [A][B][C][D][ ][ ][ ][ ]   使用率 50%
  append(E) → 使用率 62.5%、リサイズ不要
  pop() → 使用率 50%未満 → 縮小して容量 4 に
  append(E) → 容量不足 → 拡張して容量 8 に
  pop() → 縮小して容量 4 に
  ...永遠にリサイズが繰り返される!

  安全な設計:
  成長率 2x、縮小閾値 25% に設定
  → 拡張と縮小の閾値に十分なバッファを持たせる
```

---

## 3. 多次元配列 — 行優先と列優先のメモリ配置

### 3.1 メモリ上の配置方式

二次元配列は論理的には行と列の格子だが、メモリ上は一次元に展開される。
展開方式は言語により異なり、パフォーマンスに大きな影響を与える。

```
論理的な二次元配列 (3x4):
          列0  列1  列2  列3
  行0  [  1    2    3    4  ]
  行1  [  5    6    7    8  ]
  行2  [  9   10   11   12  ]

行優先 (Row-Major) — C, C++, Python, Java:
  メモリ: [1][2][3][4][5][6][7][8][9][10][11][12]
           ←── 行0 ──→←── 行1 ──→←── 行2 ───→

  アドレス計算: arr[i][j] = base + (i * cols + j) * sizeof(element)
  arr[1][2] = base + (1 * 4 + 2) * 4 = base + 24

列優先 (Column-Major) — Fortran, MATLAB, Julia, R:
  メモリ: [1][5][9][2][6][10][3][7][11][4][8][12]
           ←─ 列0 ─→←─ 列1 ──→←─ 列2 ──→←─ 列3 ─→

  アドレス計算: arr[i][j] = base + (j * rows + i) * sizeof(element)
```

### 3.2 キャッシュ効率と走査順序

CPU キャッシュは「空間的局所性」を利用する。つまり、あるアドレスにアクセスすると
近くのアドレスもキャッシュラインに読み込まれる。このため、メモリの配置方式に
合わせた走査順序を選択することが重要である。

```
行優先格納の場合のキャッシュ効率:

  高速（行方向に走査 — キャッシュヒット率が高い）:
    for i in range(rows):
        for j in range(cols):     ← 連続メモリを順番にアクセス
            process(arr[i][j])

  低速（列方向に走査 — キャッシュミスが多発）:
    for j in range(cols):
        for i in range(rows):     ← メモリ上で飛び飛びにアクセス
            process(arr[i][j])

  想定される性能差: 大きな配列 (例: 10000x10000) では 2~10 倍の差が出る
```

### 3.3 コード例: 行優先 vs 列優先の走査性能比較

```python
"""
行優先と列優先の走査速度を比較するプログラム
キャッシュ効率の違いを体感する
"""
import time


def create_matrix(rows, cols):
    """rows x cols の行列を生成する"""
    return [[i * cols + j for j in range(cols)] for i in range(rows)]


def row_major_sum(matrix, rows, cols):
    """行優先走査で合計を計算 — キャッシュフレンドリー"""
    total = 0
    for i in range(rows):
        for j in range(cols):
            total += matrix[i][j]
    return total


def col_major_sum(matrix, rows, cols):
    """列優先走査で合計を計算 — キャッシュアンフレンドリー"""
    total = 0
    for j in range(cols):
        for i in range(rows):
            total += matrix[i][j]
    return total


def benchmark():
    """走査順序による性能差を計測する"""
    sizes = [(1000, 1000), (3000, 3000), (5000, 5000)]

    for rows, cols in sizes:
        matrix = create_matrix(rows, cols)

        # 行優先走査
        start = time.perf_counter()
        s1 = row_major_sum(matrix, rows, cols)
        t_row = time.perf_counter() - start

        # 列優先走査
        start = time.perf_counter()
        s2 = col_major_sum(matrix, rows, cols)
        t_col = time.perf_counter() - start

        assert s1 == s2
        ratio = t_col / t_row if t_row > 0 else float('inf')
        print(f"{rows}x{cols}: 行優先={t_row:.4f}s, "
              f"列優先={t_col:.4f}s, 比率={ratio:.2f}x")


if __name__ == "__main__":
    benchmark()
```

### 3.4 多次元配列の表現方式

Python では多次元配列を表現する方法が複数あり、用途によって使い分ける:

| 方式 | コード例 | メモリ配置 | 適用場面 |
|------|---------|-----------|---------|
| リストのリスト | `[[0]*n for _ in range(m)]` | 非連続 | 小規模、汎用 |
| numpy ndarray | `np.zeros((m, n))` | 連続（行優先） | 数値計算 |
| array モジュール | `array.array('i', [...])` | 連続（一次元のみ） | メモリ効率重視 |
| ctypes 配列 | `(c_int * n * m)()` | 連続 | C ライブラリ連携 |

---

## 4. 文字列の基本 — 不変性・エンコーディング・内部表現

### 4.1 文字列のメモリ表現

文字列は「文字の配列」として格納されるが、文字をどのようにバイト列にエンコードするかは
言語やプラットフォームにより異なる。

```
ASCII 文字列 "Hello" のメモリレイアウト:

  C 言語 (NULL 終端):
  ┌─────┬─────┬─────┬─────┬─────┬─────┐
  │ 'H' │ 'e' │ 'l' │ 'l' │ 'o' │ '\0'│
  │ 0x48│ 0x65│ 0x6C│ 0x6C│ 0x6F│ 0x00│
  └─────┴─────┴─────┴─────┴─────┴─────┘
  6 バイト (NULL 終端含む)

  Python 3 (コンパクト ASCII):
  ┌──────────────────────────────┐
  │ PyObject ヘッダ (16 bytes)    │
  │ hash: キャッシュ済み (8 bytes) │
  │ length: 5 (8 bytes)          │
  │ ┌─────┬─────┬─────┬─────┬─────┐
  │ │ 'H' │ 'e' │ 'l' │ 'l' │ 'o' │
  │ └─────┴─────┴─────┴─────┴─────┘
  └──────────────────────────────┘
  約 54 バイト (オブジェクトヘッダ含む)

  Java (UTF-16、Java 9+ はコンパクト):
  ┌──────────────────────────────┐
  │ Object ヘッダ (12 bytes)      │
  │ hash: 0 (未計算) (4 bytes)    │
  │ value: byte[] 参照 (8 bytes)  │
  │ coder: LATIN1 or UTF16       │
  └──────────────────────────────┘
```

### 4.2 文字列の不変性 (Immutability)

多くの言語で文字列は不変 (immutable) である。変更操作は常に新しい文字列オブジェクトを生成する。

```
Python での文字列「変更」の実際の動作:

  s = "Hello"
  id(s) → 0x7f8001000  ← アドレスA

  s = s + " World"
  id(s) → 0x7f8002000  ← アドレスB（新しいオブジェクト!）

  メモリ上の変化:
  操作前:
    s ──→ [H][e][l][l][o]                   (アドレスA)

  操作後:
    s ──→ [H][e][l][l][o][ ][W][o][r][l][d] (アドレスB、新規生成)
           [H][e][l][l][o]                   (アドレスA、GC 対象)

  不変性の利点:
  1. ハッシュ値をキャッシュ → dict のキーとして O(1) で使用可能
  2. スレッドセーフ → ロック不要で共有可能
  3. 文字列インターン → 同一内容を共有してメモリ節約
```

不変性を持つ言語と持たない言語:

| 不変 (Immutable) | 可変 (Mutable) |
|-----------------|----------------|
| Python, Java, C#, Go, JavaScript | C (char[]), C++ (std::string), Rust (String) |

### 4.3 エンコーディングの基礎

```
Unicode エンコーディングの比較:

  文字 "A" (U+0041):
    UTF-8:  [0x41]                    — 1 バイト
    UTF-16: [0x00][0x41]              — 2 バイト
    UTF-32: [0x00][0x00][0x00][0x41]  — 4 バイト

  文字 "あ" (U+3042):
    UTF-8:  [0xE3][0x81][0x82]        — 3 バイト
    UTF-16: [0x30][0x42]              — 2 バイト
    UTF-32: [0x00][0x00][0x30][0x42]  — 4 バイト

  絵文字 "😀" (U+1F600):
    UTF-8:  [0xF0][0x9F][0x98][0x80]  — 4 バイト
    UTF-16: [0xD83D][0xDE00]          — 4 バイト (サロゲートペア)
    UTF-32: [0x00][0x01][0xF6][0x00]  — 4 バイト
```

| エンコーディング | バイト数 | 特徴 | 主な使用場面 |
|----------------|---------|------|-------------|
| UTF-8 | 1-4 バイト/文字 | ASCII 互換、可変長 | Web, Linux, ファイル |
| UTF-16 | 2-4 バイト/文字 | BMP は 2 バイト | Windows, Java, JavaScript |
| UTF-32 | 4 バイト/文字 | 固定長、O(1) アクセス | 内部処理 |

### 4.4 Python 3 の文字列内部表現 (PEP 393)

Python 3.3 以降は PEP 393 (Flexible String Representation) により、
文字列の内容に応じて最適なエンコーディングを自動選択する:

| 文字の範囲 | 内部エンコーディング | 1文字のサイズ |
|-----------|-------------------|-------------|
| U+0000 - U+00FF (Latin-1) | Latin-1 | 1 バイト |
| U+0000 - U+FFFF (BMP) | UCS-2 | 2 バイト |
| U+0000 - U+10FFFF (全範囲) | UCS-4 | 4 バイト |

### 4.5 コード例: 文字列操作の計算量を意識した実装

```python
"""
文字列操作の計算量を意識した実装パターン集
不変文字列の言語での正しい文字列構築方法を学ぶ
"""
import sys
import time


def string_building_comparison():
    """文字列構築: 連結 vs join の性能比較"""
    n = 50000
    words = [f"word{i}" for i in range(n)]

    # 方法1: += による連結 — O(n^2)
    start = time.perf_counter()
    result1 = ""
    for w in words:
        result1 += w
    t1 = time.perf_counter() - start

    # 方法2: join — O(n)
    start = time.perf_counter()
    result2 = "".join(words)
    t2 = time.perf_counter() - start

    # 方法3: io.StringIO — O(n)
    import io
    start = time.perf_counter()
    buf = io.StringIO()
    for w in words:
        buf.write(w)
    result3 = buf.getvalue()
    t3 = time.perf_counter() - start

    assert result1 == result2 == result3
    print(f"+=  連結:   {t1:.4f}秒")
    print(f"join:       {t2:.4f}秒")
    print(f"StringIO:   {t3:.4f}秒")
    print(f"join は += の約 {t1/t2:.1f} 倍高速")


def unicode_details():
    """Python の文字列内部表現を観察する"""
    strings = [
        "Hello",           # ASCII のみ → Latin-1 (1バイト/文字)
        "Bonjour cafe\u0301",  # Latin-1 範囲 → Latin-1
        "こんにちは",       # BMP → UCS-2 (2バイト/文字)
        "Hello 😀",        # 絵文字含む → UCS-4 (4バイト/文字)
    ]

    for s in strings:
        size = sys.getsizeof(s)
        # ヘッダを除いたデータ部分のサイズを推定
        overhead = sys.getsizeof("") # 空文字列のサイズ
        data_size = size - overhead
        per_char = data_size / len(s) if len(s) > 0 else 0
        print(f"'{s}': len={len(s)}, size={size}B, "
              f"data={data_size}B, ~{per_char:.1f}B/char")


if __name__ == "__main__":
    print("=== 文字列構築の性能比較 ===")
    string_building_comparison()
    print()
    print("=== Unicode 内部表現 ===")
    unicode_details()
```

### 4.6 文字列操作の計算量まとめ

| 操作 | Python | Java | C++ (std::string) | 備考 |
|------|--------|------|--------------------|------|
| インデックスアクセス `s[i]` | O(1) | O(1) | O(1) | 固定幅エンコーディング前提 |
| 連結 `s + t` | O(n+m) | O(n+m) | O(n+m) | 新しい文字列を生成 |
| 部分文字列 `s[i:j]` | O(j-i) | O(j-i) | O(j-i) | コピーが発生 |
| 検索 `s.find(t)` | O(n*m) 最悪 | O(n*m) 最悪 | O(n*m) 最悪 | 最悪ケース |
| 比較 `s == t` | O(min(n,m)) | O(min(n,m)) | O(min(n,m)) | 先頭から比較 |
| ハッシュ `hash(s)` | O(n) 初回 | O(n) 初回 | O(n) | 初回計算後キャッシュ |
| 長さ `len(s)` | O(1) | O(1) | O(1) | 事前に保存 |
| 置換 `s.replace(a, b)` | O(n*m) | O(n*m) | O(n*m) | 全出現箇所を置換 |

---

## 5. 配列の基本アルゴリズム — 回転・マージ・パーティション

### 5.1 ソート済み配列のマージ

2つのソート済み配列を1つのソート済み配列にマージする操作は、
マージソートの中核であり、配列アルゴリズムの基本中の基本である。

```
マージの過程の可視化:

  a = [1, 3, 5, 7]    b = [2, 4, 6, 8]
       ^                    ^
       i=0                  j=0

  比較: a[0]=1 < b[0]=2 → result に 1 を追加, i++
  比較: a[1]=3 > b[0]=2 → result に 2 を追加, j++
  比較: a[1]=3 < b[1]=4 → result に 3 を追加, i++
  比較: a[2]=5 > b[1]=4 → result に 4 を追加, j++
  ...

  result = [1, 2, 3, 4, 5, 6, 7, 8]
```

```python
"""
ソート済み配列のマージ — 完全動作版
"""


def merge_sorted(a: list, b: list) -> list:
    """
    2つのソート済み配列をマージして1つのソート済み配列を返す

    時間計算量: O(n + m)  n = len(a), m = len(b)
    空間計算量: O(n + m)  結果配列のぶん
    """
    result = []
    i = j = 0

    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1

    # 残りの要素を追加
    result.extend(a[i:])
    result.extend(b[j:])
    return result


def merge_sorted_inplace(arr: list, mid: int) -> None:
    """
    arr[0:mid] と arr[mid:] がそれぞれソート済みの場合、
    arr 全体をソート済みにする（追加メモリ O(n)）
    """
    left = arr[:mid]
    right = arr[mid:]
    i = j = k = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            arr[k] = left[i]
            i += 1
        else:
            arr[k] = right[j]
            j += 1
        k += 1

    while i < len(left):
        arr[k] = left[i]
        i += 1
        k += 1

    while j < len(right):
        arr[k] = right[j]
        j += 1
        k += 1


# テスト
if __name__ == "__main__":
    # 基本テスト
    assert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]
    assert merge_sorted([], [1, 2, 3]) == [1, 2, 3]
    assert merge_sorted([1], []) == [1]
    assert merge_sorted([], []) == []
    assert merge_sorted([1, 1, 1], [1, 1]) == [1, 1, 1, 1, 1]

    # in-place テスト
    arr = [1, 3, 5, 2, 4, 6]
    merge_sorted_inplace(arr, 3)
    assert arr == [1, 2, 3, 4, 5, 6]
    print("全テスト通過")
```

### 5.2 配列の回転

配列の回転は、要素を指定した数だけ左または右に循環シフトする操作である。
3回の反転 (reversal) で O(n) 時間・O(1) 空間で実現できる。

```
左回転 k=2 の反転アルゴリズム:

  元の配列:  [1, 2, 3, 4, 5, 6, 7]

  Step 1: 先頭 k 個を反転
           [2, 1, | 3, 4, 5, 6, 7]

  Step 2: 残り n-k 個を反転
           [2, 1, | 7, 6, 5, 4, 3]

  Step 3: 全体を反転
           [3, 4, 5, 6, 7, 1, 2]   ← 左に2回転した結果!

  なぜこれで正しいのか:
    元: [a1, a2, ..., ak, b1, b2, ..., bn-k]
    目的: [b1, b2, ..., bn-k, a1, a2, ..., ak]

    Step 1: [ak, ..., a2, a1, b1, b2, ..., bn-k]   = rev(A) + B
    Step 2: [ak, ..., a2, a1, bn-k, ..., b2, b1]   = rev(A) + rev(B)
    Step 3: [b1, b2, ..., bn-k, a1, a2, ..., ak]   = rev(rev(A) + rev(B))
           = B + A                                   ← 目的達成!
```

```python
"""
配列の回転 — 3つの方法の比較
"""


def rotate_left_reversal(arr: list, k: int) -> list:
    """
    反転アルゴリズムによる左回転
    時間: O(n), 空間: O(1) — in-place
    """
    if not arr:
        return arr
    n = len(arr)
    k = k % n
    if k == 0:
        return arr

    def reverse(lo: int, hi: int) -> None:
        while lo < hi:
            arr[lo], arr[hi] = arr[hi], arr[lo]
            lo += 1
            hi -= 1

    reverse(0, k - 1)      # 先頭 k 要素を反転
    reverse(k, n - 1)      # 残り n-k 要素を反転
    reverse(0, n - 1)      # 全体を反転
    return arr


def rotate_left_slice(arr: list, k: int) -> list:
    """
    スライスによる左回転
    時間: O(n), 空間: O(n)
    """
    if not arr:
        return arr
    k = k % len(arr)
    return arr[k:] + arr[:k]


def rotate_right(arr: list, k: int) -> list:
    """
    右回転 — 左回転 (n - k) と同等
    時間: O(n), 空間: O(1)
    """
    if not arr:
        return arr
    n = len(arr)
    k = k % n
    return rotate_left_reversal(arr, n - k)


# テスト
if __name__ == "__main__":
    # 左回転テスト
    assert rotate_left_slice([1, 2, 3, 4, 5, 6, 7], 2) == [3, 4, 5, 6, 7, 1, 2]

    arr = [1, 2, 3, 4, 5, 6, 7]
    rotate_left_reversal(arr, 2)
    assert arr == [3, 4, 5, 6, 7, 1, 2]

    # 右回転テスト
    arr2 = [1, 2, 3, 4, 5, 6, 7]
    rotate_right(arr2, 2)
    assert arr2 == [6, 7, 1, 2, 3, 4, 5]

    # エッジケース
    assert rotate_left_slice([], 3) == []
    assert rotate_left_slice([1], 5) == [1]
    assert rotate_left_slice([1, 2], 4) == [1, 2]  # k % n = 0
    print("全テスト通過")
```

### 5.3 Dutch National Flag (三方分割)

配列を3つの領域に分割するアルゴリズム。クイックソートの3-way partition の基礎となる。

```python
"""
Dutch National Flag 問題 — 3つの値を持つ配列のソート
LeetCode 75: Sort Colors と同等
"""


def dutch_national_flag(arr: list, pivot: int = 1) -> list:
    """
    配列を [< pivot | == pivot | > pivot] の3領域に分割する

    時間計算量: O(n) — 1パスで完了
    空間計算量: O(1) — in-place

    3つのポインタ:
      lo:  次に < pivot の要素を置く位置
      mid: 現在調査中の位置
      hi:  次に > pivot の要素を置く位置
    """
    lo = mid = 0
    hi = len(arr) - 1

    while mid <= hi:
        if arr[mid] < pivot:
            arr[lo], arr[mid] = arr[mid], arr[lo]
            lo += 1
            mid += 1
        elif arr[mid] > pivot:
            arr[mid], arr[hi] = arr[hi], arr[mid]
            hi -= 1
            # mid は進めない（交換で来た要素をまだ調べていない）
        else:
            mid += 1

    return arr


# テスト
if __name__ == "__main__":
    assert dutch_national_flag([2, 0, 1, 2, 0, 1, 0]) == [0, 0, 0, 1, 1, 2, 2]
    assert dutch_national_flag([1, 1, 1]) == [1, 1, 1]
    assert dutch_national_flag([2, 1, 0]) == [0, 1, 2]
    assert dutch_national_flag([]) == []
    assert dutch_national_flag([0]) == [0]
    print("全テスト通過")
```

---

## 6. 文字列アルゴリズム — 回文・アナグラム・パターン検索

### 6.1 回文判定

回文 (Palindrome) は前から読んでも後ろから読んでも同じ文字列である。
Two Pointers を使って O(n) 時間、O(1) 空間で判定できる。

```python
"""
回文判定 — 3つのバリエーション
"""


def is_palindrome(s: str) -> bool:
    """
    英数字のみを対象とした回文判定
    時間: O(n), 空間: O(n) — フィルタリングした文字列を生成
    """
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]


def is_palindrome_two_pointers(s: str) -> bool:
    """
    Two Pointers による回文判定（空間 O(1)）
    英数字のみを対象、大文字小文字を区別しない
    """
    left, right = 0, len(s) - 1

    while left < right:
        # 英数字でない文字をスキップ
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1

        if s[left].lower() != s[right].lower():
            return False
        left += 1
        right -= 1

    return True


def is_almost_palindrome(s: str) -> bool:
    """
    最大1文字を削除して回文にできるか判定
    LeetCode 680: Valid Palindrome II
    """
    def check(lo: int, hi: int) -> bool:
        while lo < hi:
            if s[lo] != s[hi]:
                return False
            lo += 1
            hi -= 1
        return True

    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            # 左を飛ばすか右を飛ばすかの2択
            return check(left + 1, right) or check(left, right - 1)
        left += 1
        right -= 1
    return True


# テスト
if __name__ == "__main__":
    # 基本テスト
    assert is_palindrome("A man, a plan, a canal: Panama") is True
    assert is_palindrome("race a car") is False
    assert is_palindrome("") is True

    # Two Pointers 版
    assert is_palindrome_two_pointers("A man, a plan, a canal: Panama") is True
    assert is_palindrome_two_pointers("race a car") is False

    # Almost Palindrome
    assert is_almost_palindrome("abca") is True   # 'c' を削除 → "aba"
    assert is_almost_palindrome("abc") is False
    assert is_almost_palindrome("aba") is True     # 削除不要
    assert is_almost_palindrome("a") is True
    print("全テスト通過")
```

### 6.2 アナグラム判定と検索

アナグラム (Anagram) は、文字の並び替えで作れる文字列のペアである。
判定には文字頻度カウントを使う。

```python
"""
アナグラム判定と部分文字列アナグラム検索
"""
from collections import Counter


def is_anagram(s: str, t: str) -> bool:
    """
    2つの文字列がアナグラムかどうかを判定する
    時間: O(n), 空間: O(k) k=文字種数
    """
    if len(s) != len(t):
        return False
    return Counter(s) == Counter(t)


def is_anagram_sort(s: str, t: str) -> bool:
    """
    ソートによるアナグラム判定（別解）
    時間: O(n log n), 空間: O(n)
    """
    return sorted(s) == sorted(t)


def find_anagrams(s: str, p: str) -> list:
    """
    文字列 s の中で p のアナグラムとなる部分文字列の開始位置を全て返す
    LeetCode 438: Find All Anagrams in a String

    Sliding Window + 文字頻度カウントで O(n) を実現

    時間: O(n), 空間: O(k) k=文字種数
    """
    if len(p) > len(s):
        return []

    result = []
    p_count = Counter(p)
    window = Counter(s[:len(p)])

    if window == p_count:
        result.append(0)

    for i in range(len(p), len(s)):
        # ウィンドウの右端を追加
        window[s[i]] += 1
        # ウィンドウの左端を削除
        left_char = s[i - len(p)]
        window[left_char] -= 1
        if window[left_char] == 0:
            del window[left_char]

        if window == p_count:
            result.append(i - len(p) + 1)

    return result


# テスト
if __name__ == "__main__":
    # アナグラム判定
    assert is_anagram("listen", "silent") is True
    assert is_anagram("hello", "world") is False
    assert is_anagram("", "") is True
    assert is_anagram("a", "ab") is False

    # アナグラム検索
    assert find_anagrams("cbaebabacd", "abc") == [0, 6]
    assert find_anagrams("abab", "ab") == [0, 1, 2]
    assert find_anagrams("a", "ab") == []
    print("全テスト通過")
```

### 6.3 最長共通接頭辞

```python
"""
文字列配列の最長共通接頭辞 (Longest Common Prefix)
LeetCode 14
"""


def longest_common_prefix(strs: list) -> str:
    """
    文字列リストの最長共通接頭辞を返す

    方法: 縦スキャン — 各位置の文字を全文字列で比較
    時間: O(S) S=全文字列の文字数合計
    空間: O(1) (結果を除く)
    """
    if not strs:
        return ""

    for i in range(len(strs[0])):
        char = strs[0][i]
        for j in range(1, len(strs)):
            if i >= len(strs[j]) or strs[j][i] != char:
                return strs[0][:i]

    return strs[0]


def longest_common_prefix_binary_search(strs: list) -> str:
    """
    二分探索による最長共通接頭辞

    時間: O(S * log m) S=最短文字列長, m=最短文字列長
    """
    if not strs:
        return ""

    min_len = min(len(s) for s in strs)

    def is_common_prefix(length: int) -> bool:
        prefix = strs[0][:length]
        return all(s[:length] == prefix for s in strs)

    lo, hi = 0, min_len
    while lo <= hi:
        mid = (lo + hi) // 2
        if is_common_prefix(mid):
            lo = mid + 1
        else:
            hi = mid - 1

    return strs[0][:(lo + hi) // 2 + (1 if lo > hi else 0)]


# テスト
if __name__ == "__main__":
    assert longest_common_prefix(["flower", "flow", "flight"]) == "fl"
    assert longest_common_prefix(["dog", "racecar", "car"]) == ""
    assert longest_common_prefix(["alone"]) == "alone"
    assert longest_common_prefix([]) == ""
    assert longest_common_prefix(["", "b"]) == ""
    assert longest_common_prefix(["ab", "ab", "ab"]) == "ab"
    print("全テスト通過")
```

---

## 7. Two Pointers と Sliding Window

### 7.1 Two Pointers の概要

Two Pointers は、配列や文字列の問題で2つのポインタを使って
O(n^2) のブルートフォースを O(n) に改善する技法である。

```
Two Pointers の3つのパターン:

パターン1: 対向ポインタ (Opposite Direction)
  用途: ソート済み配列でペア検索、回文判定
  ┌─────────────────────────────┐
  │  [1] [3] [5] [7] [9] [11]  │
  │   L→              ←R       │
  └─────────────────────────────┘
  L は右へ、R は左へ移動

パターン2: 同方向ポインタ (Same Direction)
  用途: 重複削除、条件を満たす部分列
  ┌─────────────────────────────┐
  │  [1] [1] [2] [2] [3] [4]   │
  │   S→                        │
  │       F→                    │
  └─────────────────────────────┘
  Slow と Fast が同じ方向に進む

パターン3: 異速ポインタ (Fast and Slow)
  用途: サイクル検出、中間要素の発見
  ┌─────────────────────────────┐
  │  [a]→[b]→[c]→[d]→[e]→...  │
  │   S→                        │
  │       F──→                  │
  └─────────────────────────────┘
  Slow は1歩、Fast は2歩ずつ
```

### 7.2 コード例: Two Pointers の代表的な問題

```python
"""
Two Pointers パターン集 — 代表的な5問題
"""


def two_sum_sorted(arr: list, target: int) -> list:
    """
    ソート済み配列で和が target のペアのインデックスを返す
    LeetCode 167: Two Sum II

    パターン: 対向ポインタ
    時間: O(n), 空間: O(1)
    """
    left, right = 0, len(arr) - 1

    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1

    return []  # ペアが見つからない


def remove_duplicates(arr: list) -> int:
    """
    ソート済み配列から重複を除去し、ユニーク要素数を返す
    LeetCode 26: Remove Duplicates from Sorted Array

    パターン: 同方向ポインタ (Slow/Fast)
    時間: O(n), 空間: O(1)
    """
    if not arr:
        return 0

    slow = 0
    for fast in range(1, len(arr)):
        if arr[fast] != arr[slow]:
            slow += 1
            arr[slow] = arr[fast]

    return slow + 1  # ユニーク要素数


def three_sum(nums: list) -> list:
    """
    配列から和が 0 になる3つ組を全て返す（重複なし）
    LeetCode 15: 3Sum

    ソート + 固定1つ + Two Pointers
    時間: O(n^2), 空間: O(1) (結果を除く)
    """
    nums.sort()
    result = []

    for i in range(len(nums) - 2):
        # 重複スキップ
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        left, right = i + 1, len(nums) - 1

        while left < right:
            total = nums[i] + nums[left] + nums[right]

            if total == 0:
                result.append([nums[i], nums[left], nums[right]])
                # 重複スキップ
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1

    return result


def container_with_most_water(height: list) -> int:
    """
    最大の水量を持つ容器を求める
    LeetCode 11: Container With Most Water

    パターン: 対向ポインタ
    時間: O(n), 空間: O(1)
    """
    left, right = 0, len(height) - 1
    max_area = 0

    while left < right:
        width = right - left
        h = min(height[left], height[right])
        max_area = max(max_area, width * h)

        if height[left] < height[right]:
            left += 1
        else:
            right -= 1

    return max_area


def move_zeroes(nums: list) -> None:
    """
    0 を末尾に移動し、非ゼロ要素の順序を保持
    LeetCode 283: Move Zeroes

    パターン: 同方向ポインタ
    時間: O(n), 空間: O(1)
    """
    slow = 0
    for fast in range(len(nums)):
        if nums[fast] != 0:
            nums[slow], nums[fast] = nums[fast], nums[slow]
            slow += 1


# テスト
if __name__ == "__main__":
    # Two Sum Sorted
    assert two_sum_sorted([1, 3, 5, 7, 9, 11], 12) == [0, 5]  # 1+11=12
    assert two_sum_sorted([2, 7, 11, 15], 9) == [0, 1]

    # Remove Duplicates
    arr = [1, 1, 2, 2, 3, 4, 4]
    k = remove_duplicates(arr)
    assert arr[:k] == [1, 2, 3, 4]

    # 3Sum
    assert three_sum([-1, 0, 1, 2, -1, -4]) == [[-1, -1, 2], [-1, 0, 1]]

    # Container With Most Water
    assert container_with_most_water([1, 8, 6, 2, 5, 4, 8, 3, 7]) == 49

    # Move Zeroes
    nums = [0, 1, 0, 3, 12]
    move_zeroes(nums)
    assert nums == [1, 3, 12, 0, 0]

    print("全テスト通過")
```

### 7.3 Sliding Window の概要

Sliding Window は、連続する部分配列（部分文字列）に関する問題を効率的に解く技法である。
ウィンドウの左端と右端を適切に動かしながら、ウィンドウ内の状態を更新する。

```
Sliding Window の動作イメージ (固定サイズ k=3):

  配列: [2, 1, 5, 1, 3, 2]

  Window 1: [2, 1, 5]          sum = 8
  Window 2:    [1, 5, 1]       sum = 8 - 2 + 1 = 7
  Window 3:       [5, 1, 3]    sum = 7 - 1 + 3 = 9  ← 最大
  Window 4:          [1, 3, 2] sum = 9 - 5 + 2 = 6

  右端を追加 (+) して左端を削除 (-) → O(1) で更新
  全体: O(n)

Sliding Window の動作イメージ (可変サイズ):

  目的: 和が target 以上の最短部分配列
  配列: [2, 3, 1, 2, 4, 3], target = 7

  [2, 3, 1, 2]              sum=8 >= 7 → 長さ4, 左を縮める
     [3, 1, 2]              sum=6 <  7 → 右を伸ばす
     [3, 1, 2, 4]           sum=10>= 7 → 長さ4, 左を縮める
        [1, 2, 4]           sum=7 >= 7 → 長さ3, 左を縮める
           [2, 4]           sum=6 <  7 → 右を伸ばす
           [2, 4, 3]        sum=9 >= 7 → 長さ3, 左を縮める
              [4, 3]        sum=7 >= 7 → 長さ2 ← 最短!
```

### 7.4 コード例: Sliding Window の代表的な問題

```python
"""
Sliding Window パターン集
"""


def max_sum_subarray(arr: list, k: int) -> int:
    """
    サイズ k の連続部分配列の最大和を返す
    固定サイズ Sliding Window

    時間: O(n), 空間: O(1)
    """
    if len(arr) < k:
        return 0

    # 最初のウィンドウの和
    window_sum = sum(arr[:k])
    max_sum = window_sum

    # ウィンドウをスライド
    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i - k]
        max_sum = max(max_sum, window_sum)

    return max_sum


def min_subarray_len(target: int, nums: list) -> int:
    """
    和が target 以上の最短部分配列の長さを返す
    LeetCode 209: Minimum Size Subarray Sum

    可変サイズ Sliding Window
    時間: O(n), 空間: O(1)
    """
    min_len = float('inf')
    window_sum = 0
    left = 0

    for right in range(len(nums)):
        window_sum += nums[right]

        while window_sum >= target:
            min_len = min(min_len, right - left + 1)
            window_sum -= nums[left]
            left += 1

    return min_len if min_len != float('inf') else 0


def longest_substring_without_repeating(s: str) -> int:
    """
    繰り返し文字のない最長部分文字列の長さを返す
    LeetCode 3: Longest Substring Without Repeating Characters

    可変サイズ Sliding Window + HashSet
    時間: O(n), 空間: O(k) k=文字種数
    """
    char_set = set()
    max_len = 0
    left = 0

    for right in range(len(s)):
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        char_set.add(s[right])
        max_len = max(max_len, right - left + 1)

    return max_len


# テスト
if __name__ == "__main__":
    # 固定サイズ
    assert max_sum_subarray([2, 1, 5, 1, 3, 2], 3) == 9
    assert max_sum_subarray([1, 2], 3) == 0

    # 最短部分配列
    assert min_subarray_len(7, [2, 3, 1, 2, 4, 3]) == 2
    assert min_subarray_len(100, [1, 2, 3]) == 0
    assert min_subarray_len(4, [1, 4, 4]) == 1

    # 最長部分文字列
    assert longest_substring_without_repeating("abcabcbb") == 3
    assert longest_substring_without_repeating("bbbbb") == 1
    assert longest_substring_without_repeating("pwwkew") == 3
    assert longest_substring_without_repeating("") == 0

    print("全テスト通過")
```

---

## 8. 二次元配列の走査パターン — スパイラル・対角線・回転

### 8.1 主要な走査パターン一覧

```
4x4 行列での各走査パターン:

  行列:
  ┌────┬────┬────┬────┐
  │  1 │  2 │  3 │  4 │
  ├────┼────┼────┼────┤
  │  5 │  6 │  7 │  8 │
  ├────┼────┼────┼────┤
  │  9 │ 10 │ 11 │ 12 │
  ├────┼────┼────┼────┤
  │ 13 │ 14 │ 15 │ 16 │
  └────┴────┴────┴────┘

  (a) 行優先走査:  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
  (b) 列優先走査:  1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16
  (c) スパイラル:  1, 2, 3, 4, 8, 12, 16, 15, 14, 13, 9, 5, 6, 7, 11, 10
  (d) 対角線走査:  1 | 2, 5 | 3, 6, 9 | 4, 7, 10, 13 | 8, 11, 14 | 12, 15 | 16
  (e) ジグザグ:    1 | 5, 2 | 3, 6, 9 | 13, 10, 7, 4 | 8, 11, 14 | 15, 12 | 16

走査方向の可視化:

  スパイラル走査:          ジグザグ (対角線) 走査:
  → → → ↓                ↗   ↗   ↗
  ↑ → ↓ ↓              ↙   ↙   ↙
  ↑ ↑ ← ↓                ↗   ↗   ↗
  ↑ ← ← ←              ↙   ↙   ↙
```

### 8.2 コード例: スパイラル走査

```python
"""
二次元配列のスパイラル走査 — 完全動作版
"""


def spiral_order(matrix: list) -> list:
    """
    m x n 行列のスパイラル (渦巻き) 走査
    LeetCode 54: Spiral Matrix

    4つの境界 (top, bottom, left, right) を管理して
    外側から内側へ渦巻き状に走査する

    時間: O(m * n), 空間: O(1) (結果を除く)
    """
    if not matrix or not matrix[0]:
        return []

    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1

    while top <= bottom and left <= right:
        # 上辺: 左から右へ
        for j in range(left, right + 1):
            result.append(matrix[top][j])
        top += 1

        # 右辺: 上から下へ
        for i in range(top, bottom + 1):
            result.append(matrix[i][right])
        right -= 1

        # 下辺: 右から左へ（行が残っている場合のみ）
        if top <= bottom:
            for j in range(right, left - 1, -1):
                result.append(matrix[bottom][j])
            bottom -= 1

        # 左辺: 下から上へ（列が残っている場合のみ）
        if left <= right:
            for i in range(bottom, top - 1, -1):
                result.append(matrix[i][left])
            left += 1

    return result


def generate_spiral_matrix(n: int) -> list:
    """
    n x n のスパイラル行列を生成する
    LeetCode 59: Spiral Matrix II

    1 から n^2 までの数をスパイラル順に配置
    """
    matrix = [[0] * n for _ in range(n)]
    top, bottom = 0, n - 1
    left, right = 0, n - 1
    num = 1

    while top <= bottom and left <= right:
        for j in range(left, right + 1):
            matrix[top][j] = num
            num += 1
        top += 1

        for i in range(top, bottom + 1):
            matrix[i][right] = num
            num += 1
        right -= 1

        if top <= bottom:
            for j in range(right, left - 1, -1):
                matrix[bottom][j] = num
                num += 1
            bottom -= 1

        if left <= right:
            for i in range(bottom, top - 1, -1):
                matrix[i][left] = num
                num += 1
            left += 1

    return matrix


# テスト
if __name__ == "__main__":
    # スパイラル走査
    m1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert spiral_order(m1) == [1, 2, 3, 6, 9, 8, 7, 4, 5]

    m2 = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    assert spiral_order(m2) == [1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]

    assert spiral_order([]) == []

    # スパイラル行列生成
    assert generate_spiral_matrix(3) == [
        [1, 2, 3],
        [8, 9, 4],
        [7, 6, 5]
    ]
    print("全テスト通過")
```

### 8.3 コード例: 行列の90度回転

```python
"""
N x N 行列の回転 — 転置 + 反転による O(1) 空間の解法
LeetCode 48: Rotate Image
"""


def rotate_90_clockwise(matrix: list) -> list:
    """
    N x N 行列を時計回りに90度回転（in-place）

    方法: 転置 → 各行を反転
    時間: O(n^2), 空間: O(1)
    """
    n = len(matrix)

    # Step 1: 転置（行と列を入れ替え）
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

    # Step 2: 各行を反転
    for row in matrix:
        row.reverse()

    return matrix


def rotate_90_counterclockwise(matrix: list) -> list:
    """
    N x N 行列を反時計回りに90度回転（in-place）

    方法: 転置 → 各列を反転（= 上下反転）
    時間: O(n^2), 空間: O(1)
    """
    n = len(matrix)

    # Step 1: 転置
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

    # Step 2: 上下反転
    matrix.reverse()

    return matrix


def rotate_180(matrix: list) -> list:
    """
    N x N 行列を180度回転（in-place）

    方法: 上下反転 → 各行を反転
    時間: O(n^2), 空間: O(1)
    """
    matrix.reverse()
    for row in matrix:
        row.reverse()
    return matrix


# テスト
if __name__ == "__main__":
    # 時計回り 90度
    m1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    rotate_90_clockwise(m1)
    assert m1 == [[7, 4, 1], [8, 5, 2], [9, 6, 3]]

    # 反時計回り 90度
    m2 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    rotate_90_counterclockwise(m2)
    assert m2 == [[3, 6, 9], [2, 5, 8], [1, 4, 7]]

    # 180度
    m3 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    rotate_180(m3)
    assert m3 == [[9, 8, 7], [6, 5, 4], [3, 2, 1]]

    print("全テスト通過")
```

### 8.4 コード例: 対角線走査

```python
"""
行列の対角線走査
LeetCode 498: Diagonal Traverse
"""


def diagonal_traverse(matrix: list) -> list:
    """
    m x n 行列をジグザグ対角線順に走査する

    時間: O(m * n), 空間: O(1) (結果を除く)
    """
    if not matrix or not matrix[0]:
        return []

    m, n = len(matrix), len(matrix[0])
    result = []
    row, col = 0, 0
    going_up = True

    for _ in range(m * n):
        result.append(matrix[row][col])

        if going_up:
            if col == n - 1:          # 右端に到達
                row += 1
                going_up = False
            elif row == 0:            # 上端に到達
                col += 1
                going_up = False
            else:                     # 右上に移動
                row -= 1
                col += 1
        else:
            if row == m - 1:          # 下端に到達
                col += 1
                going_up = True
            elif col == 0:            # 左端に到達
                row += 1
                going_up = True
            else:                     # 左下に移動
                row += 1
                col -= 1

    return result


# テスト
if __name__ == "__main__":
    m1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert diagonal_traverse(m1) == [1, 2, 4, 7, 5, 3, 6, 8, 9]

    m2 = [[1, 2], [3, 4]]
    assert diagonal_traverse(m2) == [1, 2, 3, 4]

    assert diagonal_traverse([]) == []

    print("全テスト通過")
```

---

## 9. 比較表と計算量チートシート

### 表1: 配列操作の計算量比較

| 操作 | 静的配列 (C) | 動的配列 (Python list) | 連結リスト | deque |
|------|-------------|----------------------|-----------|-------|
| アクセス `[i]` | O(1) | O(1) | O(n) | O(n) |
| 先頭挿入 | O(n) | O(n) | O(1) | O(1) |
| 末尾挿入 | 不可 | O(1) 償却 | O(1)* | O(1) |
| 中間挿入 | O(n) | O(n) | O(1)** | O(n) |
| 先頭削除 | O(n) | O(n) | O(1) | O(1) |
| 末尾削除 | O(1) | O(1) | O(n)* | O(1) |
| 中間削除 | O(n) | O(n) | O(1)** | O(n) |
| 探索 (未ソート) | O(n) | O(n) | O(n) | O(n) |
| 探索 (ソート済) | O(log n) | O(log n) | O(n) | O(n) |
| メモリ効率 | 最良 | 良 (余剰容量あり) | 悪 (ポインタ分) | 良 |
| キャッシュ効率 | 最良 | 良 | 悪 | 中 |

`*` 末尾ポインタがある場合
`**` 挿入/削除位置が既知の場合（位置の探索は O(n)）

### 表2: 文字列操作の計算量（言語間比較）

| 操作 | Python (str) | Java (String) | C++ (std::string) | Go (string) |
|------|-------------|---------------|-------------------|-------------|
| インデックス `s[i]` | O(1) | O(1) | O(1) | O(1)*** |
| 連結 `s + t` | O(n+m) | O(n+m) | O(n+m) | O(n+m) |
| 部分文字列 | O(k) コピー | O(k) コピー | O(k) コピー | O(k) コピー |
| 検索 `find` | O(nm) 最悪 | O(nm) 最悪 | O(nm) 最悪 | O(nm) 最悪 |
| 長さ取得 | O(1) | O(1) | O(1) | O(1) |
| 比較 | O(n) | O(n) | O(n) | O(n) |
| ハッシュ | O(n) 初回のみ | O(n) 初回のみ | O(n) | O(n) |
| 不変性 | 不変 | 不変 | 可変 | 不変 |
| 可変版 | `list` + `join` | `StringBuilder` | `std::string` 自体 | `[]byte` |

`***` Go の string はバイト列のため、マルチバイト文字は `[]rune` への変換が必要

### 表3: 配列の実装比較（メモリ特性）

| 実装 | 要素格納 | オーバーヘッド | キャッシュライン効率 |
|------|---------|--------------|-------------------|
| C `int[]` | 値を直接格納 | なし | 最高 |
| C++ `vector<int>` | 値を直接格納 | 24B (ptr+size+cap) | 最高 |
| Java `int[]` | 値を直接格納 | 16B (ヘッダ) | 高い |
| Java `ArrayList<Integer>` | ボクシング + ポインタ | 大きい | 低い |
| Python `list` | ポインタの配列 | 56B + 8B/要素 | 低い |
| NumPy `ndarray` | 値を直接格納 | 固定ヘッダ | 最高 |

### 表4: アルゴリズムパターンの使い分け

| パターン | 時間計算量 | 適用条件 | 代表問題 |
|---------|-----------|---------|---------|
| ブルートフォース | O(n^2) | 制約が小さい (n <= 1000) | 全ペア検査 |
| ソート + Two Pointers | O(n log n) | ソート可能、ペア検索 | Two Sum, 3Sum |
| ハッシュマップ | O(n) | 存在チェック、頻度カウント | アナグラム判定 |
| 固定 Sliding Window | O(n) | 固定サイズの部分配列 | 最大和部分配列 |
| 可変 Sliding Window | O(n) | 条件を満たす最短/最長部分配列 | 最短部分配列 |
| 二分探索 | O(log n) | ソート済み配列 | 値の検索 |
| Dutch National Flag | O(n) | 3値への分割 | Sort Colors |
| Prefix Sum | O(n) 前処理 + O(1) クエリ | 範囲和のクエリ | Subarray Sum |

---

## 10. アンチパターンと正しい書き方

### アンチパターン1: ループ内の文字列連結

不変文字列の言語でループ内の `+=` を使うと O(n^2) になる。
これは面接・コードレビューで最もよく指摘されるアンチパターンの1つである。

```python
"""
アンチパターン1: ループ内の文字列連結
"""


def build_string_bad(words: list) -> str:
    """BAD: O(n^2) — 毎回新しい文字列を生成してコピー"""
    result = ""
    for w in words:
        result += w  # len(result) のコピーが毎回発生
    return result
    # n 個の長さ L の文字列の場合:
    # コピー量 = L + 2L + 3L + ... + nL = L * n(n+1)/2 = O(n^2 * L)


def build_string_good(words: list) -> str:
    """GOOD: O(n) — join は内部で合計長を計算してから1回だけ確保"""
    return "".join(words)
    # 1. 合計長を計算: O(n)
    # 2. 結果バッファを1回確保: O(合計長)
    # 3. 各文字列をコピー: O(合計長)
    # 合計: O(n + 合計長) = O(n)


def build_string_also_good(words: list) -> str:
    """ALSO GOOD: リストに蓄積してから join"""
    parts = []
    for w in words:
        # 必要なら加工してから追加
        parts.append(w.upper())
    return " ".join(parts)
```

### アンチパターン2: 配列の先頭への頻繁な挿入/削除

Python の `list` で先頭への `insert(0, x)` や `pop(0)` を繰り返すと、
毎回全要素をシフトするため O(n) かかり、全体として O(n^2) になる。

```python
"""
アンチパターン2: 配列の先頭への頻繁な操作
"""
from collections import deque


def queue_bad(operations: list) -> list:
    """BAD: list をキューとして使う — dequeue が O(n)"""
    queue = []
    results = []
    for op, val in operations:
        if op == "enqueue":
            queue.append(val)       # O(1) — これは問題ない
        elif op == "dequeue":
            results.append(queue.pop(0))  # O(n) — 毎回全要素をシフト!
    return results


def queue_good(operations: list) -> list:
    """GOOD: deque を使う — 両端 O(1)"""
    queue = deque()
    results = []
    for op, val in operations:
        if op == "enqueue":
            queue.append(val)       # O(1)
        elif op == "dequeue":
            results.append(queue.popleft())  # O(1)
    return results
```

### アンチパターン3: 不要な二次元配列のコピーの罠

```python
"""
アンチパターン3: 二次元配列の浅いコピーの罠
"""


def create_matrix_bad(rows: int, cols: int) -> list:
    """BAD: 全行が同じリストオブジェクトを参照する"""
    return [[0] * cols] * rows
    # matrix[0] is matrix[1] is ... → True
    # matrix[0][0] = 1 とすると全行の [0] が 1 になる!


def create_matrix_good(rows: int, cols: int) -> list:
    """GOOD: 各行が独立したリストオブジェクト"""
    return [[0] * cols for _ in range(rows)]
    # matrix[0] is matrix[1] → False
    # 各行は独立して変更可能


# 実演
if __name__ == "__main__":
    bad = create_matrix_bad(3, 3)
    bad[0][0] = 99
    print(f"BAD:  {bad}")    # [[99, 0, 0], [99, 0, 0], [99, 0, 0]] ← 全行に影響!

    good = create_matrix_good(3, 3)
    good[0][0] = 99
    print(f"GOOD: {good}")   # [[99, 0, 0], [0, 0, 0], [0, 0, 0]] ← 意図通り
```

### アンチパターン4: in 演算子の誤用

```python
"""
アンチパターン4: list での in 演算子は O(n)
"""


def contains_bad(data: list, targets: list) -> list:
    """BAD: list で in を使う — O(n) × m = O(nm)"""
    return [t for t in targets if t in data]  # data が list だと O(n)


def contains_good(data: list, targets: list) -> list:
    """GOOD: set に変換してから in を使う — O(1) × m = O(n+m)"""
    data_set = set(data)  # O(n) で変換
    return [t for t in targets if t in data_set]  # O(1) のルックアップ
```

---

## 11. 演習問題 — 基礎・応用・発展の3段階

### 基礎レベル（配列と文字列の基本操作）

**問題 B1: 配列の最大値と最小値を1パスで求める**

配列から最大値と最小値を同時に見つける関数を実装せよ。
ただしソートを使わず、1回の走査 O(n) で完了すること。

```python
def find_min_max(arr: list) -> tuple:
    """
    配列の最小値と最大値を1パスで求める
    戻り値: (min_val, max_val)
    制約: arr は空でない
    """
    # ここに実装を書く
    pass


# テストケース
assert find_min_max([3, 1, 4, 1, 5, 9, 2, 6]) == (1, 9)
assert find_min_max([42]) == (42, 42)
assert find_min_max([-5, -1, -10, -3]) == (-10, -1)
```

<details>
<summary>解答例（クリックで展開）</summary>

```python
def find_min_max(arr: list) -> tuple:
    min_val = max_val = arr[0]
    for x in arr[1:]:
        if x < min_val:
            min_val = x
        elif x > max_val:
            max_val = x
    return (min_val, max_val)
```

</details>

**問題 B2: 文字列内の文字出現頻度を集計する**

文字列内の各文字の出現回数を辞書として返す関数を実装せよ。
大文字・小文字は区別しない。空白と記号は除外すること。

```python
def char_frequency(s: str) -> dict:
    """
    文字列中の英数字の出現頻度を返す（小文字に統一）
    """
    # ここに実装を書く
    pass


# テストケース
assert char_frequency("Hello, World!") == {
    'h': 1, 'e': 1, 'l': 3, 'o': 2, 'w': 1, 'r': 1, 'd': 1
}
assert char_frequency("") == {}
```

<details>
<summary>解答例（クリックで展開）</summary>

```python
def char_frequency(s: str) -> dict:
    freq = {}
    for c in s.lower():
        if c.isalnum():
            freq[c] = freq.get(c, 0) + 1
    return freq
```

</details>

**問題 B3: 2つの文字列が回転関係にあるか判定する**

文字列 s を何回か左回転すると t になるかを判定せよ。
ヒント: s+s の中に t が含まれるか？

```python
def is_rotation(s: str, t: str) -> bool:
    """
    t が s の回転で得られるか判定する
    例: "waterbottle" は "erbottlewat" の回転 → True
    """
    # ここに実装を書く
    pass


# テストケース
assert is_rotation("waterbottle", "erbottlewat") is True
assert is_rotation("abc", "cab") is True
assert is_rotation("abc", "acb") is False
assert is_rotation("", "") is True
```

<details>
<summary>解答例（クリックで展開）</summary>

```python
def is_rotation(s: str, t: str) -> bool:
    if len(s) != len(t):
        return False
    if len(s) == 0:
        return True
    return t in (s + s)
```

</details>

---

### 応用レベル（アルゴリズムの組み合わせ）

**問題 A1: 積の配列（自分以外の全要素の積）**

配列 nums が与えられたとき、`result[i]` が `nums[i]` 以外の全要素の積になるような
配列を返せ。除算を使わずに O(n) で解くこと。

```python
def product_except_self(nums: list) -> list:
    """
    LeetCode 238: Product of Array Except Self
    除算なし、O(n) 時間、O(1) 空間（結果配列を除く）
    """
    # ここに実装を書く
    pass


# テストケース
assert product_except_self([1, 2, 3, 4]) == [24, 12, 8, 6]
assert product_except_self([-1, 1, 0, -3, 3]) == [0, 0, 9, 0, 0]
```

<details>
<summary>解答例（クリックで展開）</summary>

```python
def product_except_self(nums: list) -> list:
    n = len(nums)
    result = [1] * n

    # 左からの累積積
    left_product = 1
    for i in range(n):
        result[i] = left_product
        left_product *= nums[i]

    # 右からの累積積を掛ける
    right_product = 1
    for i in range(n - 1, -1, -1):
        result[i] *= right_product
        right_product *= nums[i]

    return result
```

</details>

**問題 A2: 最大和の連続部分配列（Kadane's Algorithm）**

配列の連続部分配列の中で和が最大のものを求めよ。

```python
def max_subarray_sum(nums: list) -> int:
    """
    LeetCode 53: Maximum Subarray
    Kadane's Algorithm — O(n) 時間、O(1) 空間
    """
    # ここに実装を書く
    pass


# テストケース
assert max_subarray_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6  # [4,-1,2,1]
assert max_subarray_sum([1]) == 1
assert max_subarray_sum([-1, -2, -3]) == -1
```

<details>
<summary>解答例（クリックで展開）</summary>

```python
def max_subarray_sum(nums: list) -> int:
    max_sum = current_sum = nums[0]
    for x in nums[1:]:
        current_sum = max(x, current_sum + x)
        max_sum = max(max_sum, current_sum)
    return max_sum
```

</details>

**問題 A3: 最長回文部分文字列**

文字列 s の中で最長の回文部分文字列を返せ。

```python
def longest_palindrome_substring(s: str) -> str:
    """
    LeetCode 5: Longest Palindromic Substring
    中心拡張法 — O(n^2) 時間、O(1) 空間
    """
    # ここに実装を書く
    pass


# テストケース
assert longest_palindrome_substring("babad") in ("bab", "aba")
assert longest_palindrome_substring("cbbd") == "bb"
assert longest_palindrome_substring("a") == "a"
```

<details>
<summary>解答例（クリックで展開）</summary>

```python
def longest_palindrome_substring(s: str) -> str:
    if len(s) < 2:
        return s

    start, max_len = 0, 1

    def expand(left: int, right: int) -> None:
        nonlocal start, max_len
        while left >= 0 and right < len(s) and s[left] == s[right]:
            if right - left + 1 > max_len:
                start = left
                max_len = right - left + 1
            left -= 1
            right += 1

    for i in range(len(s)):
        expand(i, i)      # 奇数長の回文
        expand(i, i + 1)  # 偶数長の回文

    return s[start:start + max_len]
```

</details>

---

### 発展レベル（高度なアルゴリズム設計）

**問題 E1: 雨水トラップ問題**

高さの配列が与えられたとき、雨が降った後にトラップできる水の総量を求めよ。

```python
def trap_rainwater(height: list) -> int:
    """
    LeetCode 42: Trapping Rain Water
    Two Pointers — O(n) 時間、O(1) 空間
    """
    # ここに実装を書く
    pass


# テストケース
assert trap_rainwater([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]) == 6
assert trap_rainwater([4, 2, 0, 3, 2, 5]) == 9
assert trap_rainwater([]) == 0
```

<details>
<summary>解答例（クリックで展開）</summary>

```python
def trap_rainwater(height: list) -> int:
    if not height:
        return 0

    left, right = 0, len(height) - 1
    left_max = right_max = 0
    water = 0

    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                water += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                water += right_max - height[right]
            right -= 1

    return water
```

</details>

**問題 E2: 最小ウィンドウ部分文字列**

文字列 s と t が与えられたとき、t の全文字を含む s の最小部分文字列を返せ。

```python
def min_window(s: str, t: str) -> str:
    """
    LeetCode 76: Minimum Window Substring
    Sliding Window — O(n + m) 時間
    """
    # ここに実装を書く
    pass


# テストケース
assert min_window("ADOBECODEBANC", "ABC") == "BANC"
assert min_window("a", "a") == "a"
assert min_window("a", "aa") == ""
```

<details>
<summary>解答例（クリックで展開）</summary>

```python
from collections import Counter

def min_window(s: str, t: str) -> str:
    if not s or not t or len(s) < len(t):
        return ""

    t_count = Counter(t)
    required = len(t_count)       # t に含まれるユニーク文字数
    formed = 0                     # 条件を満たした文字種数
    window_counts = {}

    ans = (float('inf'), 0, 0)     # (長さ, 左, 右)
    left = 0

    for right in range(len(s)):
        char = s[right]
        window_counts[char] = window_counts.get(char, 0) + 1

        if char in t_count and window_counts[char] == t_count[char]:
            formed += 1

        while left <= right and formed == required:
            if right - left + 1 < ans[0]:
                ans = (right - left + 1, left, right)

            left_char = s[left]
            window_counts[left_char] -= 1
            if left_char in t_count and window_counts[left_char] < t_count[left_char]:
                formed -= 1
            left += 1

    return "" if ans[0] == float('inf') else s[ans[1]:ans[2] + 1]
```

</details>

**問題 E3: 文字列の全順列（Permutation）**

文字列の全ての順列を重複なく列挙せよ。

```python
def string_permutations(s: str) -> list:
    """
    文字列の全順列を辞書順でソートして返す
    重複文字がある場合も重複なく列挙する
    """
    # ここに実装を書く
    pass


# テストケース
assert string_permutations("abc") == ["abc", "acb", "bac", "bca", "cab", "cba"]
assert string_permutations("aab") == ["aab", "aba", "baa"]
assert string_permutations("a") == ["a"]
```

<details>
<summary>解答例（クリックで展開）</summary>

```python
def string_permutations(s: str) -> list:
    result = []
    chars = sorted(s)

    def backtrack(path: list, remaining: list) -> None:
        if not remaining:
            result.append("".join(path))
            return
        for i in range(len(remaining)):
            # 重複スキップ
            if i > 0 and remaining[i] == remaining[i - 1]:
                continue
            backtrack(
                path + [remaining[i]],
                remaining[:i] + remaining[i + 1:]
            )

    backtrack([], chars)
    return result
```

</details>

---

## 12. FAQ — よくある質問と回答

### Q1: Python の list はどの程度メモリを余分に使うのか？

**A:** Python の `list` は要素のポインタ配列を内部に持つ。各要素は 8 バイトの
ポインタとしてカウントされ、さらにリスト自体のヘッダが約 56 バイトある。
成長率は約 1.125 倍（正確には `new_size = old_size + (old_size >> 3) + 6`）で、
Java の `ArrayList` (1.5 倍) や C++ の `vector` (2 倍) より控えめである。

容量 n の list のメモリ使用量は概算で `56 + 8n` バイト（ポインタの先の
オブジェクト自体のサイズは含まない）。整数のリスト `[0, 1, ..., 99]` の場合、
list 本体 = 56 + 800 = 856 バイト、int オブジェクト（各 28 バイト）=
2800 バイト、合計約 3.6 KB になる。同じデータを `numpy.array` で格納すると
100 * 8 + ヘッダ = 約 900 バイトとなり、約 4 倍のメモリ効率になる。

### Q2: 文字列が immutable であることのメリットとデメリットは？

**A:**

**メリット:**
1. **ハッシュ値のキャッシュ**: 文字列を辞書のキーとして使う際、ハッシュ値を
   一度計算すれば再利用できる。Python の `dict` が文字列キーで高速なのはこのため。
2. **スレッドセーフ**: 複数スレッドから同時に読み取っても安全。ロック不要。
3. **文字列インターン**: 同一内容の文字列を自動的に共有してメモリを節約。
   Python では短い文字列や識別子に使われる文字列が自動インターンされる。
4. **関数の副作用の排除**: 文字列を関数に渡しても、元の文字列が変更されない保証。

**デメリット:**
1. **変更のたびに新しいオブジェクトが生成される**: 特にループ内の連結で顕著。
2. **メモリの断片化**: 大量の短い文字列の生成・破棄で GC の負荷が増える。

### Q3: numpy 配列と Python list の選択基準は？

**A:** 以下の基準で使い分ける:

| 基準 | Python list | numpy ndarray |
|------|------------|---------------|
| 要素の型 | 混在可能 | 単一型のみ |
| サイズ変更 | append/pop が高速 | リサイズは苦手 |
| 数値演算 | 遅い (ループ必要) | 高速 (ベクトル化) |
| メモリ効率 | 低い (ポインタ配列) | 高い (値を直接格納) |
| ブロードキャスト | なし | 強力なサポート |
| スライス | コピーを返す | ビューを返す (メモリ共有) |

**結論**: 数値の同型配列で算術演算が主なら numpy。異なる型が混在する、頻繁に
サイズが変わる、または非数値データなら list を使う。

### Q4: 配列のソートアルゴリズムの選択に配列のサイズは影響するか？

**A:** 大きく影響する。Python の `sorted()` / `list.sort()` は Tim Sort を使い、
n が小さい場合（通常 64 要素以下）は挿入ソートに切り替える。挿入ソートは
O(n^2) だが、キャッシュ効率が良く、オーバーヘッドが小さいため、小規模データでは
O(n log n) のアルゴリズムより高速になる。C++ の `std::sort` も同様に、
小規模では挿入ソートにフォールバックする。

### Q5: Two Pointers と Sliding Window の使い分けは？

**A:**

- **Two Pointers**: 主にソート済み配列で使用。2つのポインタが条件に応じて
  独立に動く。典型例: ソート済み配列でのペア検索、回文判定。
- **Sliding Window**: 連続した部分配列/部分文字列を対象とする。ウィンドウの
  左右端が同じ方向に動く。典型例: 最大和部分配列、条件を満たす最長/最短部分文字列。

判断基準:
1. 問題が「連続した部分配列/部分文字列」に関するものなら → Sliding Window
2. 問題が「2つの要素の組み合わせ」に関するものなら → Two Pointers
3. 配列がソート済みでペアを探すなら → 対向 Two Pointers

### Q6: 行列の走査問題で境界条件を間違えないコツは？

**A:** 3つのテクニックがある:

1. **境界変数を4つ明示的に管理する**: `top`, `bottom`, `left`, `right` を使い、
   各辺の走査後に対応する境界を更新する。
2. **走査の前に残り確認**: 下辺を走査する前に `top <= bottom` を、
   左辺を走査する前に `left <= right` を確認する。これを忘れると
   要素の二重カウントが発生する。
3. **小さい例でトレース**: 1x1, 1xN, Nx1, 2x2 の行列で手動トレースして
   境界条件が正しいか確認する。

---


## FAQ

### Q1: このトピックを学ぶ上で最も重要なポイントは何ですか？

実践的な経験を積むことが最も重要です。理論だけでなく、実際にコードを書いて動作を確認することで理解が深まります。

### Q2: 初心者がよく陥る間違いは何ですか？

基礎を飛ばして応用に進むことです。このガイドで説明している基本概念をしっかり理解してから、次のステップに進むことをお勧めします。

### Q3: 実務ではどのように活用されていますか？

このトピックの知識は、日常的な開発業務で頻繁に活用されます。特にコードレビューやアーキテクチャ設計の際に重要になります。

---

## 13. まとめと次のステップ

### この章で学んだこと

| 項目 | ポイント |
|------|---------|
| 静的配列 | 連続メモリに格納、O(1) ランダムアクセス、固定サイズ |
| 動的配列 | append は償却 O(1)、成長率で空間効率が変わる、縮小戦略も重要 |
| 多次元配列 | 行優先/列優先でキャッシュ効率が異なる、走査順序を意識する |
| 文字列 | immutable の言語が多い、連結は join を使う、エンコーディングに注意 |
| Two Pointers | 対向・同方向・異速の3パターン、O(n^2) を O(n) に改善 |
| Sliding Window | 固定サイズと可変サイズの2種類、連続部分配列の問題に適用 |
| 二次元配列走査 | スパイラル・対角線・ジグザグ等、境界管理が鍵 |

### 頻出 LeetCode 問題リスト

| 難易度 | 番号 | 問題名 | パターン |
|--------|------|--------|---------|
| Easy | 1 | Two Sum | Hash Map |
| Easy | 26 | Remove Duplicates | Two Pointers |
| Easy | 14 | Longest Common Prefix | 縦スキャン |
| Easy | 283 | Move Zeroes | Two Pointers |
| Medium | 3 | Longest Substring Without Repeating | Sliding Window |
| Medium | 15 | 3Sum | Sort + Two Pointers |
| Medium | 48 | Rotate Image | 転置 + 反転 |
| Medium | 54 | Spiral Matrix | 境界管理 |
| Medium | 238 | Product of Array Except Self | Prefix/Suffix |
| Medium | 438 | Find All Anagrams | Sliding Window |
| Hard | 42 | Trapping Rain Water | Two Pointers |
| Hard | 76 | Minimum Window Substring | Sliding Window |

### 次に読むべきガイド

- [連結リスト — 単方向/双方向とフロイドのアルゴリズム](./01-linked-lists.md)
- [スタックとキュー — LIFO/FIFO と単調スタック](./02-stacks-queues.md)
- [ハッシュテーブル — ハッシュ関数と衝突解決](./03-hash-tables.md)

---

## 14. 参考文献

1. Cormen, T.H., Leiserson, C.E., Rivest, R.L. & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. — 第2章「Getting Started」で配列のマージとソートの基礎、第17章「Amortized Analysis」で動的配列の償却解析を詳細に解説。

2. Sedgewick, R. & Wayne, K. (2011). *Algorithms* (4th ed.). Addison-Wesley. — Java を使った配列とソートの実装詳細。Web サイト (algs4.cs.princeton.edu) に豊富なビジュアライゼーションがある。

3. Skiena, S.S. (2020). *The Algorithm Design Manual* (3rd ed.). Springer. — 実践的な配列・文字列アルゴリズムの設計パターンと、面接対策に役立つ「War Stories」を収録。

4. Python Software Foundation. "Time Complexity." Python Wiki. https://wiki.python.org/moin/TimeComplexity — Python の組み込みデータ構造の操作別計算量の公式リファレンス。

5. Knuth, D.E. (1997). *The Art of Computer Programming, Volume 1: Fundamental Algorithms* (3rd ed.). Addison-Wesley. — 配列と線形リストの数学的基礎。

6. CPython Source Code. `Objects/listobject.c`. https://github.com/python/cpython — Python list の動的配列実装の実際のソースコード。成長率の計算式 `new_allocated = (size_t)newsize + (newsize >> 3) + (newsize < 9 ? 3 : 6)` を確認できる。

7. Bentley, J. (1986). *Programming Pearls*. Addison-Wesley. — 配列の回転アルゴリズム（反転による手法）の元となった名著。第2章「Aha! Algorithms」で紹介。

---

> **免責事項**: 本ガイドのコード例はすべて教育目的で作成されている。プロダクションコードでは、各言語の標準ライブラリが提供する最適化された実装を使用することを推奨する。

---

## 次に読むべきガイド

- [連結リスト — 単方向・双方向・循環・フロイドのアルゴリズム](./01-linked-lists.md) - 次のトピックへ進む

---

## 参考文献

- [MDN Web Docs](https://developer.mozilla.org/) - Web技術のリファレンス
- [Wikipedia](https://ja.wikipedia.org/) - 技術概念の概要
