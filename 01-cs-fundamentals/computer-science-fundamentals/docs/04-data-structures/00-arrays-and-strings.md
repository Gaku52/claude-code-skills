# 配列と文字列

> 配列はメモリ上の連続した領域であり、そのシンプルさゆえにCPUキャッシュと最も相性が良いデータ構造である。
> すべてのプログラマが最初に学び、最後まで使い続ける基本中の基本である。

## この章で学ぶこと

- [ ] 配列のメモリレイアウトとCPUキャッシュの関係を理解する
- [ ] 静的配列と動的配列の内部実装を説明できる
- [ ] 文字列のエンコーディング（UTF-8/UTF-16/UTF-32）の違いと実務的影響を理解する
- [ ] 配列に対する典型テクニック（二分探索、尺取り法、スライディングウィンドウ）を実装できる
- [ ] 文字列探索アルゴリズム（KMP、Rabin-Karp、Boyer-Moore）の原理を説明できる
- [ ] アンチパターンを認識し、パフォーマンスの落とし穴を回避できる

## 前提知識

- メモリの基礎 → 参照: [[../01-hardware-basics/01-memory-hierarchy.md]]
- 計算量解析 → 参照: [[../03-algorithms/01-complexity-analysis.md]]

---

## 1. なぜ配列と文字列が重要か

### 1.1 コンピュータの最も原始的なデータ構造

配列（array）は、同じ型の要素をメモリ上の連続した領域に格納するデータ構造である。
この「連続性」こそが、配列を他のあらゆるデータ構造と区別する最も重要な特性であり、
現代のCPUアーキテクチャにおいてパフォーマンス上の決定的な優位性をもたらす。

なぜ配列がこれほど重要なのかを理解するには、ハードウェアの視点から考える必要がある。

### 1.2 メモリの連続性とキャッシュ局所性

現代のCPUは、メインメモリ（DRAM）へのアクセスに比べてキャッシュへのアクセスが
桁違いに高速である。典型的なアクセス時間は以下のとおりである。

```
CPU キャッシュ階層とアクセス時間:

  ┌─────────────────────────────────────────────────────────┐
  │                      CPU コア                            │
  │  ┌─────────────────────────────────────────────────┐    │
  │  │  レジスタ          : ~0.3 ns    (1 サイクル)      │    │
  │  └─────────────────────────────────────────────────┘    │
  │  ┌─────────────────────────────────────────────────┐    │
  │  │  L1 キャッシュ (64KB) : ~1 ns   (3-4 サイクル)    │    │
  │  └─────────────────────────────────────────────────┘    │
  │  ┌─────────────────────────────────────────────────┐    │
  │  │  L2 キャッシュ (256KB): ~4 ns   (12 サイクル)     │    │
  │  └─────────────────────────────────────────────────┘    │
  └─────────────────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────────────────┐
  │  L3 キャッシュ (8MB)     : ~12 ns  (40 サイクル)       │
  └─────────────────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────────────────┐
  │  メインメモリ (DRAM)     : ~100 ns (300+ サイクル)     │
  └─────────────────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────────────────┐
  │  SSD                     : ~100,000 ns                  │
  └─────────────────────────────────────────────────────────┘

  → L1 キャッシュは DRAM の約100倍高速
  → 配列の連続アクセスは、キャッシュライン (64B) 単位でプリフェッチされる
  → 連結リストはポインタを辿るためキャッシュミスが頻発する
```

CPUがメモリからデータを読み込むとき、1バイトだけではなく「キャッシュライン」と呼ばれる
64バイトの塊をまとめて取得する。配列の要素は連続しているため、1つの要素にアクセスすると
近隣の要素も同時にキャッシュに載る。これが「空間的局所性（spatial locality）」である。

さらに、配列を順番に走査するパターンはCPUのハードウェアプリフェッチャが検出し、
次に必要になるデータを事前にキャッシュに読み込む。これにより、実質的なメモリアクセス
待ち時間がほぼゼロになる場面すらある。

### 1.3 配列 vs 連結リスト: キャッシュの影響

理論的な計算量だけでは実際のパフォーマンスは予測できない。
以下のCプログラムで配列と連結リストの走査速度を比較してみよう。

```c
/* cache_benchmark.c - 配列 vs 連結リストの走査速度比較 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 10000000

typedef struct Node {
    int value;
    struct Node *next;
} Node;

int main(void) {
    /* --- 配列の走査 --- */
    int *arr = malloc(sizeof(int) * N);
    for (int i = 0; i < N; i++) arr[i] = i;

    clock_t start = clock();
    long long sum_arr = 0;
    for (int i = 0; i < N; i++) {
        sum_arr += arr[i];
    }
    clock_t end = clock();
    double time_arr = (double)(end - start) / CLOCKS_PER_SEC;

    /* --- 連結リストの走査 --- */
    Node *head = NULL;
    for (int i = N - 1; i >= 0; i--) {
        Node *node = malloc(sizeof(Node));
        node->value = i;
        node->next = head;
        head = node;
    }

    start = clock();
    long long sum_list = 0;
    Node *cur = head;
    while (cur) {
        sum_list += cur->value;
        cur = cur->next;
    }
    end = clock();
    double time_list = (double)(end - start) / CLOCKS_PER_SEC;

    printf("配列走査     : %.4f 秒 (sum = %lld)\n", time_arr, sum_arr);
    printf("連結リスト走査: %.4f 秒 (sum = %lld)\n", time_list, sum_list);
    printf("比率         : %.1fx\n", time_list / time_arr);

    /* メモリ解放 */
    free(arr);
    while (head) {
        Node *tmp = head;
        head = head->next;
        free(tmp);
    }
    return 0;
}
```

典型的な実行結果:

```
配列走査     : 0.0123 秒 (sum = 49999995000000)
連結リスト走査: 0.0891 秒 (sum = 49999995000000)
比率         : 7.2x
```

理論的にはどちらも O(n) の走査であるにもかかわらず、配列は連結リストの約5〜10倍高速である。
この差はキャッシュミスの頻度によるものであり、データサイズが大きくなるほど顕著になる。

---

## 2. 静的配列

### 2.1 メモリモデル

静的配列はコンパイル時にサイズが決定し、スタックまたはデータセグメントに配置される。

```
静的配列のメモリレイアウト:

  宣言: int arr[5] = {10, 20, 30, 40, 50};
  型: int (4バイト)

  ベースアドレス: 0x7ffc00001000

  アドレス         値       インデックス
  ─────────────────────────────────────
  0x7ffc00001000   10       arr[0]
  0x7ffc00001004   20       arr[1]
  0x7ffc00001008   30       arr[2]
  0x7ffc0000100C   40       arr[3]
  0x7ffc00001010   50       arr[4]
  ─────────────────────────────────────

  アドレス計算式:
    arr[i] のアドレス = base_address + i * sizeof(int)
                      = 0x7ffc00001000 + i * 4

  例: arr[3] = 0x7ffc00001000 + 3 * 4 = 0x7ffc0000100C

  → 加算と乗算各1回でアドレスが確定 → O(1) ランダムアクセス
```

この単純なアドレス計算がO(1)ランダムアクセスを保証する。
連結リストでは i 番目の要素に到達するためにノードを i 回辿る必要がある（O(n)）のに対し、
配列はインデックスから直接メモリアドレスを算出できる。

### 2.2 多次元配列: 行優先 vs 列優先

2次元配列を1次元のメモリ空間にどう配置するかには2つの方式がある。

```
2次元配列 matrix[3][4] のメモリレイアウト:

  論理的な表現:
      列0  列1  列2  列3
  行0 [  1,   2,   3,   4 ]
  行1 [  5,   6,   7,   8 ]
  行2 [  9,  10,  11,  12 ]

  ■ 行優先 (Row-major order) — C, C++, Python (numpy default), Rust
  メモリ: [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 ]
           ←─ 行0 ──→ ←─ 行1 ──→ ←──  行2  ──→

  matrix[i][j] のアドレス = base + (i * num_cols + j) * sizeof(element)
  例: matrix[1][2] = base + (1 * 4 + 2) * 4 = base + 24

  ■ 列優先 (Column-major order) — Fortran, MATLAB, Julia, R
  メモリ: [ 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12 ]
           ←─列0─→ ←─列1──→ ←──列2──→ ←──列3──→

  matrix[i][j] のアドレス = base + (j * num_rows + i) * sizeof(element)
  例: matrix[1][2] = base + (2 * 3 + 1) * 4 = base + 28
```

行優先配列を行方向に走査するとキャッシュラインに沿ってアクセスするため高速であるが、
列方向に走査するとキャッシュラインを跨ぐためキャッシュミスが増える。

以下のCプログラムでその差を確認できる。

```c
/* row_vs_col.c - 行優先走査 vs 列優先走査の速度比較 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ROWS 4096
#define COLS 4096

int main(void) {
    int (*matrix)[COLS] = malloc(sizeof(int[ROWS][COLS]));

    /* 初期化 */
    for (int i = 0; i < ROWS; i++)
        for (int j = 0; j < COLS; j++)
            matrix[i][j] = i + j;

    /* 行優先走査 (キャッシュフレンドリー) */
    clock_t start = clock();
    long long sum1 = 0;
    for (int i = 0; i < ROWS; i++)
        for (int j = 0; j < COLS; j++)
            sum1 += matrix[i][j];
    clock_t end = clock();
    double time_row = (double)(end - start) / CLOCKS_PER_SEC;

    /* 列優先走査 (キャッシュ非フレンドリー) */
    start = clock();
    long long sum2 = 0;
    for (int j = 0; j < COLS; j++)
        for (int i = 0; i < ROWS; i++)
            sum2 += matrix[i][j];
    end = clock();
    double time_col = (double)(end - start) / CLOCKS_PER_SEC;

    printf("行優先走査: %.4f 秒 (sum = %lld)\n", time_row, sum1);
    printf("列優先走査: %.4f 秒 (sum = %lld)\n", time_col, sum2);
    printf("比率      : %.1fx\n", time_col / time_row);

    free(matrix);
    return 0;
}
```

典型的な結果: 列優先走査は行優先走査の3〜8倍遅い。
これは計算量がどちらも O(n^2) であるにもかかわらず、キャッシュの効果だけで
これだけの差が生まれることを示している。

### 2.3 多次元配列の実務的注意

数値計算やデータサイエンスの分野では、行列演算のパフォーマンスが特に重要である。
NumPyはデフォルトで行優先（C order）だが、Fortranとのインターフェースでは
列優先（F order）を指定する必要がある場合がある。

```python
import numpy as np

# C order (行優先, デフォルト)
a = np.array([[1, 2, 3], [4, 5, 6]], order='C')
print(a.strides)  # (24, 8) — 行を1つ進むには24バイト、列は8バイト

# Fortran order (列優先)
b = np.array([[1, 2, 3], [4, 5, 6]], order='F')
print(b.strides)  # (8, 16) — 行を1つ進むには8バイト、列は16バイト

# パフォーマンスの違い
import timeit

n = 2000
c_arr = np.random.rand(n, n)               # C order
f_arr = np.asfortranarray(np.random.rand(n, n))  # Fortran order

# 行方向の合計 (C order が有利)
t1 = timeit.timeit(lambda: c_arr.sum(axis=1), number=100)
t2 = timeit.timeit(lambda: f_arr.sum(axis=1), number=100)
print(f"行合計: C order={t1:.4f}s, F order={t2:.4f}s")

# 列方向の合計 (Fortran order が有利)
t3 = timeit.timeit(lambda: c_arr.sum(axis=0), number=100)
t4 = timeit.timeit(lambda: f_arr.sum(axis=0), number=100)
print(f"列合計: C order={t3:.4f}s, F order={t4:.4f}s")
```

---

## 3. 動的配列

### 3.1 動的配列の基本概念

動的配列（dynamic array）は、要素数の上限を事前に決めることなく、
必要に応じて自動的にサイズを拡張する配列である。
Python の `list`、Java の `ArrayList`、C++ の `std::vector`、Rust の `Vec<T>` がこれに該当する。

内部的には固定サイズの配列を保持し、容量が不足したときに
より大きな配列を確保して全要素をコピーする、という戦略を取る。

```
動的配列のリサイズ動作:

  初期状態 (capacity=4, size=4):
  ┌───┬───┬───┬───┐
  │ A │ B │ C │ D │
  └───┴───┴───┴───┘

  要素 'E' を追加 → 容量不足！

  Step 1: 新しい配列を確保 (capacity=8)
  ┌───┬───┬───┬───┬───┬───┬───┬───┐
  │   │   │   │   │   │   │   │   │
  └───┴───┴───┴───┴───┴───┴───┴───┘

  Step 2: 既存要素をコピー
  ┌───┬───┬───┬───┬───┬───┬───┬───┐
  │ A │ B │ C │ D │   │   │   │   │
  └───┴───┴───┴───┴───┴───┴───┴───┘

  Step 3: 新しい要素を追加
  ┌───┬───┬───┬───┬───┬───┬───┬───┐
  │ A │ B │ C │ D │ E │   │   │   │
  └───┴───┴───┴───┴───┴───┴───┴───┘
  (size=5, capacity=8)

  Step 4: 古い配列を解放

  → リサイズのコスト: O(n) (全要素コピー)
  → ただし、リサイズの頻度を制御することで
     append の「償却計算量」を O(1) にできる
```

### 3.2 償却計算量の解析

動的配列の `append` 操作において、容量が不足したときに倍の容量を確保する
「倍増戦略（doubling strategy）」を採用するとする。

n 回の append 操作を行った場合の総コストを計算する。

```
倍増戦略の償却解析:

  n 回の append にかかる総コスト:

  通常の append:    コスト 1 × n 回    = n
  リサイズ発生時の
  コピーコスト:     1 + 2 + 4 + 8 + ... + n/2 + n
                    = 2n - 1 (等比級数の和)

  総コスト = n + (2n - 1) = 3n - 1

  1回あたりの償却コスト = (3n - 1) / n ≈ 3 = O(1)

  ■ 別の証明: ポテンシャル法
  ポテンシャル関数: Φ(i) = 2 * size - capacity
  ※ Φ はリサイズ直後に 0、リサイズ直前に size と等しくなる

  通常の append:
    実コスト = 1
    ΔΦ = 2  (size が 1 増える)
    償却コスト = 1 + 2 = 3

  リサイズを伴う append:
    実コスト = size + 1  (コピー + 追加)
    ΔΦ = 2 - size  (capacity が倍になり、size が 1 増える)
    償却コスト = (size + 1) + (2 - size) = 3

  → いずれの場合も償却コストは 3 = O(1)
```

### 3.3 成長因子の選択

倍増（成長因子 2.0）は理論的にはシンプルだが、実際の実装では異なる成長因子が使われることがある。

```
各言語/ライブラリの成長因子:

  ┌────────────────────┬─────────┬─────────────────────────────┐
  │ 実装               │ 成長因子 │ 理由                         │
  ├────────────────────┼─────────┼─────────────────────────────┤
  │ CPython list       │ ~1.125  │ 小さいサイズで節約、大きい   │
  │                    │         │ サイズで徐々に増加           │
  │ Java ArrayList     │ 1.5     │ メモリ効率とコピーコストの   │
  │                    │         │ バランス                     │
  │ C++ std::vector    │ 2.0*    │ 実装依存。GCC=2, MSVC=1.5   │
  │ Rust Vec           │ 2.0     │ シンプルさ優先               │
  │ Go slice           │ ~1.25-2 │ サイズに応じて変動           │
  │ C# List<T>         │ 2.0     │                             │
  └────────────────────┴─────────┴─────────────────────────────┘

  成長因子のトレードオフ:
  - 大きい成長因子 (例: 2.0):
    → リサイズ回数が少ない → append は速い
    → メモリの無駄が最大50%
    → 古い領域を再利用できない問題 (後述)

  - 小さい成長因子 (例: 1.5):
    → リサイズ回数が多い → append はやや遅い
    → メモリの無駄が最大33%
    → 古い領域を再利用しやすい
```

成長因子が2.0の場合、解放された古い領域のサイズの合計は常に新しい配列のサイズより小さい。
つまり、古い領域を再利用して新しい配列を配置することが原理的にできない。
成長因子を1.5にすると、数回のリサイズ後に古い領域の合計が新しい配列を収められるサイズになり、
メモリアロケータがこれを再利用できる可能性が生まれる。
これが Java や MSVC が成長因子1.5を選択した理由の1つである。

### 3.4 CPython list の内部実装

CPython の list は、PyObject へのポインタの配列である。

```python
# CPython list の内部構造を観察する
import sys

lst = []
prev_size = sys.getsizeof(lst)
print(f"{'len':>4s}  {'sizeof':>8s}  {'capacity推定':>12s}  {'リサイズ':>8s}")
print("-" * 50)

for i in range(65):
    lst.append(i)
    current_size = sys.getsizeof(lst)
    # ポインタサイズ (64bit) = 8バイト
    # オーバーヘッド = 空listのサイズ
    overhead = sys.getsizeof([])
    capacity = (current_size - overhead) // 8
    resized = "◆" if current_size != prev_size else ""
    if resized or i < 10 or i % 10 == 0:
        print(f"{len(lst):4d}  {current_size:8d}  {capacity:12d}  {resized:>8s}")
    prev_size = current_size

# CPython の成長パターン (cpython/Objects/listobject.c より):
# new_allocated = new_size + (new_size >> 3) + (6 if new_size < 9 else 3)
# つまり: new_allocated = new_size + new_size/8 + 定数
# 成長率は約 12.5% ずつ増加 (= 成長因子 ~1.125)
```

CPython の list は「ポインタの配列」であるため、要素自体はメモリ上の別の場所にある。
これは真の連続配列ではないため、数値計算では NumPy の ndarray を使うべきである。

```
CPython list vs NumPy ndarray のメモリレイアウト:

  ■ CPython list (ポインタの配列):
  list オブジェクト
  ┌──────────────────┐
  │ ob_refcnt        │
  │ ob_type          │
  │ ob_size (長さ)    │
  │ ob_item ─────────┼──→ ┌─────┬─────┬─────┬─────┐ ポインタ配列
  │ allocated        │    │ ptr │ ptr │ ptr │ ptr │
  └──────────────────┘    └──┬──┴──┬──┴──┬──┴──┬──┘
                             │     │     │     │
                             ↓     ↓     ↓     ↓
                          int(1) int(2) int(3) int(4)  ← 各 PyObject
                          (28B)  (28B)  (28B)  (28B)     ヒープ上に散在

  → 1要素あたり: ポインタ 8B + PyObject 28B = 36B (int の場合)
  → キャッシュ効率: 悪い (ポインタ間接参照でキャッシュミス)

  ■ NumPy ndarray (連続配列):
  ndarray オブジェクト
  ┌──────────────────┐
  │ (ヘッダ)          │
  │ data ────────────┼──→ ┌─────────┬─────────┬─────────┬─────────┐
  │ shape            │    │ 1 (8B)  │ 2 (8B)  │ 3 (8B)  │ 4 (8B)  │
  │ strides          │    └─────────┴─────────┴─────────┴─────────┘
  │ dtype            │    ← 連続したメモリ領域、値が直接格納 →
  └──────────────────┘

  → 1要素あたり: 8B (int64の場合)
  → キャッシュ効率: 極めて高い
```

---

## 4. 文字列

### 4.1 文字列の本質

文字列は「文字の配列」である。ただし、「文字」の定義はエンコーディングによって異なるため、
文字列処理は見た目よりもはるかに複雑である。

### 4.2 エンコーディング: UTF-8, UTF-16, UTF-32

Unicode 文字を表現する3つの主要なエンコーディングを比較する。

```
Unicode エンコーディング比較:

  文字 "A" (U+0041):
    UTF-8:  [0x41]                          — 1バイト
    UTF-16: [0x0041]                        — 2バイト
    UTF-32: [0x00000041]                    — 4バイト

  文字 "あ" (U+3042):
    UTF-8:  [0xE3, 0x81, 0x82]             — 3バイト
    UTF-16: [0x3042]                        — 2バイト
    UTF-32: [0x00003042]                    — 4バイト

  文字 "𠀋" (U+2000B, CJK拡張B):
    UTF-8:  [0xF0, 0xA0, 0x80, 0x8B]       — 4バイト
    UTF-16: [0xD840, 0xDC0B]               — 4バイト (サロゲートペア)
    UTF-32: [0x0002000B]                    — 4バイト

  絵文字 "👨‍👩‍👧‍👦" (家族):
    → 7つのコードポイントから構成される
    U+1F468 U+200D U+1F469 U+200D U+1F467 U+200D U+1F466
    UTF-8:  25 バイト
    UTF-16: 18 バイト (サロゲートペアを含む)
    UTF-32: 28 バイト
```

各エンコーディングの特性:

```
┌─────────────┬────────────┬────────────┬────────────────────────┐
│ 特性         │ UTF-8      │ UTF-16     │ UTF-32                 │
├─────────────┼────────────┼────────────┼────────────────────────┤
│ コード単位   │ 1バイト     │ 2バイト    │ 4バイト                │
│ 可変長       │ 1-4バイト   │ 2 or 4B   │ 固定4バイト            │
│ ASCII互換    │ ○          │ ×          │ ×                     │
│ ランダム     │ ×          │ ×※        │ ○                     │
│ アクセス     │ (可変長)    │ (サロゲート)│ (固定長)               │
│ ASCII文書の  │ 最小        │ 2倍       │ 4倍                    │
│ サイズ       │            │            │                       │
│ 日本語文書の │ 1.5倍      │ 最小       │ 2倍                    │
│ サイズ       │            │            │                       │
│ 主な用途     │ ファイル    │ Windows    │ 内部処理               │
│             │ ネットワーク │ Java, JS   │ (使用は稀)             │
│ null安全     │ ○          │ ×※※      │ ×※※                  │
└─────────────┴────────────┴────────────┴────────────────────────┘

※ サロゲートペアがあるため厳密にはランダムアクセス不可
※※ 文字列中に 0x00 が出現する可能性がある
```

### 4.3 UTF-8 の設計の巧みさ

UTF-8 は Ken Thompson と Rob Pike が1992年に設計したエンコーディングであり、
以下の優れた特性を持つ。

```
UTF-8 のバイトパターン:

  範囲              バイト数  ビットパターン
  ──────────────────────────────────────────────
  U+0000..U+007F    1        0xxxxxxx
  U+0080..U+07FF    2        110xxxxx 10xxxxxx
  U+0800..U+FFFF    3        1110xxxx 10xxxxxx 10xxxxxx
  U+10000..U+10FFFF 4        11110xxx 10xxxxxx 10xxxxxx 10xxxxxx

  設計の巧みさ:
  1. ASCII (U+0000-U+007F) と完全互換
     → 既存のASCIIテキストは変更なしでUTF-8として有効
  2. 先頭バイトだけでバイト数が分かる
     → 0xxxxxxx: 1バイト
     → 110xxxxx: 2バイト
     → 1110xxxx: 3バイト
     → 11110xxx: 4バイト
  3. 後続バイトは必ず 10xxxxxx で始まる
     → バイト列のどの位置からでも文字の境界を特定可能
     → 自己同期 (self-synchronizing) 性
  4. バイト列のソート順 = コードポイントのソート順
     → memcmp でUnicode順の比較が可能
  5. NULL バイト (0x00) は U+0000 のみ
     → C言語の文字列関数がそのまま使える
```

### 4.4 文字列の不変性（Immutability）

多くの言語で文字列はイミュータブル（変更不能）として設計されている。

イミュータブル文字列の利点:
- スレッドセーフ: 複数スレッドから安全に読み取れる
- ハッシュのキャッシュ: ハッシュ値を一度計算すれば再利用できる
- セキュリティ: 文字列の内容が変わらないことを保証できる
- 文字列インターニング: 同じ内容の文字列を共有できる

イミュータブル文字列の注意点:
- 文字列を「変更」すると新しいオブジェクトが生成される → O(n)
- ループ内での連結は O(n^2) になりやすい

```python
# アンチパターン: ループ内での文字列連結 — O(n^2)
def build_string_bad(n):
    result = ""
    for i in range(n):
        result += str(i) + ","  # 毎回新しい文字列を生成
    return result

# 正しいパターン: join を使う — O(n)
def build_string_good(n):
    parts = [str(i) for i in range(n)]
    return ",".join(parts)

# ベンチマーク
import timeit
n = 50000
t1 = timeit.timeit(lambda: build_string_bad(n), number=5)
t2 = timeit.timeit(lambda: build_string_good(n), number=5)
print(f"連結 (+= ):  {t1:.3f}s")
print(f"join:        {t2:.3f}s")
print(f"比率:        {t1/t2:.1f}x")
# 典型的な結果: join は += の10倍以上高速 (n=50000)
```

### 4.5 文字列インターニング

文字列インターニングは、同じ内容の文字列が複数存在する場合に
1つのインスタンスを共有する最適化手法である。

```python
# CPython の文字列インターニング
a = "hello"
b = "hello"
print(a is b)       # True — 同じオブジェクトを共有

# コンパイル時にインターン化される条件 (CPython):
# - 識別子として有効な文字列 (英数字とアンダースコアのみ)
# - 長さが一定以下

c = "hello world"
d = "hello world"
print(c is d)       # True (CPython 3.x ではコンパイル時定数はインターン化)

# 実行時に動的生成された文字列は通常インターン化されない
e = "hello" + " " + "world"
f = "hello" + " " + "world"
# CPython の最適化レベルによって結果が異なる

# 明示的なインターニング
import sys
g = sys.intern("hello world 123")
h = sys.intern("hello world 123")
print(g is h)       # True — 常に同じオブジェクト

# Java の場合:
# String s1 = "hello";    // リテラルプールから取得
# String s2 = "hello";    // 同じ参照
# s1 == s2                // true (参照比較)
# new String("hello").intern() == s1  // true
```

---

## 5. 文字列アルゴリズム

文字列探索は、テキスト T の中からパターン P を見つける問題である。
素朴な方法は O(nm) だが、高度なアルゴリズムを使えば O(n+m) や O(n/m) も可能になる。

### 5.1 素朴な文字列探索

```python
def naive_search(text: str, pattern: str) -> list[int]:
    """素朴な文字列探索 — O(nm)"""
    n, m = len(text), len(pattern)
    positions = []
    for i in range(n - m + 1):
        if text[i:i+m] == pattern:
            positions.append(i)
    return positions

# テスト
text = "AABAACAADAABAABA"
pattern = "AABA"
print(naive_search(text, pattern))  # [0, 9, 12]
```

最悪ケース: `text = "AAAAAAAAB"`, `pattern = "AAAAB"` のように、
ほぼ一致するがパターン末尾で不一致になるケースで O(nm) になる。

### 5.2 KMP (Knuth-Morris-Pratt) アルゴリズム

KMP アルゴリズムは、パターンの接頭辞と接尾辞の一致情報を事前に計算し、
不一致が起きたときにパターンを効率的にずらすことで O(n+m) を実現する。

```python
def kmp_search(text: str, pattern: str) -> list[int]:
    """KMP アルゴリズム — O(n + m)"""
    n, m = len(text), len(pattern)
    if m == 0:
        return []

    # 失敗関数 (failure function) の構築 — O(m)
    # fail[i] = pattern[0:i+1] の真の接頭辞と接尾辞の最長一致長
    fail = [0] * m
    j = 0
    for i in range(1, m):
        while j > 0 and pattern[i] != pattern[j]:
            j = fail[j - 1]
        if pattern[i] == pattern[j]:
            j += 1
        fail[i] = j

    # 探索 — O(n)
    positions = []
    j = 0
    for i in range(n):
        while j > 0 and text[i] != pattern[j]:
            j = fail[j - 1]
        if text[i] == pattern[j]:
            j += 1
        if j == m:
            positions.append(i - m + 1)
            j = fail[j - 1]

    return positions

# テスト
text = "AABAACAADAABAABA"
pattern = "AABA"
print(kmp_search(text, pattern))  # [0, 9, 12]

# 失敗関数の動作例:
# pattern = "AABA"
# fail = [0, 1, 0, 1]
#
# A  A  B  A
# 0  1  0  1
# ↑  ↑  ↑  ↑
# 無  A  無  A   ← 接頭辞=接尾辞の最長一致
```

### 5.3 Rabin-Karp アルゴリズム

ローリングハッシュを用いて、ハッシュ値の一致でパターンを検出する。
平均 O(n+m)、最悪 O(nm) だが、複数パターンの同時検索に強い。

```python
def rabin_karp_search(text: str, pattern: str, base: int = 256,
                      mod: int = 101) -> list[int]:
    """Rabin-Karp アルゴリズム — 平均 O(n + m)"""
    n, m = len(text), len(pattern)
    if m > n:
        return []

    positions = []

    # パターンのハッシュ値を計算
    p_hash = 0
    t_hash = 0
    h = pow(base, m - 1, mod)  # base^(m-1) mod mod

    for i in range(m):
        p_hash = (p_hash * base + ord(pattern[i])) % mod
        t_hash = (t_hash * base + ord(text[i])) % mod

    # スライドしながら比較
    for i in range(n - m + 1):
        if p_hash == t_hash:
            # ハッシュ一致 → 実際に比較 (偽陽性を排除)
            if text[i:i+m] == pattern:
                positions.append(i)

        # ローリングハッシュの更新
        if i < n - m:
            t_hash = (t_hash - ord(text[i]) * h) * base + ord(text[i + m])
            t_hash %= mod

    return positions

# テスト
text = "AABAACAADAABAABA"
pattern = "AABA"
print(rabin_karp_search(text, pattern))  # [0, 9, 12]
```

### 5.4 Boyer-Moore アルゴリズム（簡略版）

Boyer-Moore はパターンを末尾から比較し、不一致文字の情報を使って
大きくスキップすることで、実用的には O(n/m) に近い性能を発揮する。
ここでは Bad Character ルールのみ実装した簡略版を示す。

```python
def boyer_moore_search(text: str, pattern: str) -> list[int]:
    """Boyer-Moore (Bad Character ルールのみ) — 実用的に O(n/m)"""
    n, m = len(text), len(pattern)
    if m == 0 or m > n:
        return []

    # Bad Character テーブル: 各文字がパターン中で最後に出現する位置
    bad_char = {}
    for i in range(m):
        bad_char[pattern[i]] = i

    positions = []
    s = 0  # テキスト上のシフト量
    while s <= n - m:
        j = m - 1  # パターン末尾から比較

        while j >= 0 and pattern[j] == text[s + j]:
            j -= 1

        if j < 0:
            # 完全一致
            positions.append(s)
            s += 1  # Good Suffix ルールがあればもっとスキップ可能
        else:
            # 不一致
            char = text[s + j]
            skip = j - bad_char.get(char, -1)
            s += max(1, skip)

    return positions

# テスト
text = "AABAACAADAABAABA"
pattern = "AABA"
print(boyer_moore_search(text, pattern))  # [0, 9, 12]
```

### 5.5 文字列探索アルゴリズムの比較

```
┌──────────────┬────────────┬────────────┬──────────────────────────┐
│ アルゴリズム   │ 最良       │ 最悪       │ 特徴                      │
├──────────────┼────────────┼────────────┼──────────────────────────┤
│ 素朴          │ O(n)       │ O(nm)      │ 短いパターンなら十分      │
│ KMP          │ O(n+m)     │ O(n+m)     │ 最悪保証あり。ストリーム   │
│              │            │            │ 処理向き                  │
│ Rabin-Karp   │ O(n+m)     │ O(nm)      │ 複数パターン同時検索。    │
│              │            │            │ 盗用検知                  │
│ Boyer-Moore  │ O(n/m)     │ O(nm)*     │ 実用最速。長いパターンで  │
│              │            │            │ 特に効果的                │
│ Aho-Corasick │ O(n+m+z)   │ O(n+m+z)   │ 複数パターン。辞書検索    │
│              │            │            │ z=出現回数                │
└──────────────┴────────────┴────────────┴──────────────────────────┘

* Boyer-Moore の最悪はGalilの改良で O(n+m) にできる
```

---

## 6. 配列の典型テクニック

### 6.1 二分探索

ソート済み配列に対する最も重要なアルゴリズム。O(log n) で要素を検索できる。

```python
def binary_search(arr: list[int], target: int) -> int:
    """二分探索 — O(log n)
    見つかった場合はインデックスを、見つからない場合は -1 を返す
    """
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2  # オーバーフロー防止
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

# lower_bound: target 以上の最小のインデックス
def lower_bound(arr: list[int], target: int) -> int:
    """C++ std::lower_bound 相当 — O(log n)"""
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo

# upper_bound: target より大きい最小のインデックス
def upper_bound(arr: list[int], target: int) -> int:
    """C++ std::upper_bound 相当 — O(log n)"""
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if arr[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    return lo

# テスト
arr = [1, 3, 3, 3, 5, 7, 9]
print(binary_search(arr, 3))    # 2 (最初に見つかったもの)
print(lower_bound(arr, 3))      # 1 (3以上の最小インデックス)
print(upper_bound(arr, 3))      # 4 (3より大きい最小インデックス)
print(lower_bound(arr, 4))      # 4 (4以上の最小 → arr[4]=5)
```

**二分探索の off-by-one エラーを防ぐポイント:**

1. `lo <= hi` vs `lo < hi`: 検索範囲が [lo, hi] か [lo, hi) かで異なる
2. `mid = lo + (hi - lo) // 2`: `(lo + hi) // 2` はオーバーフローの危険あり
3. lower_bound / upper_bound のどちらを使うべきか明確にする

### 6.2 尺取り法 (Two Pointers)

ソート済み配列の2つのポインタを使って条件を満たす組を探すテクニック。

```python
def two_sum_sorted(arr: list[int], target: int) -> tuple[int, int] | None:
    """ソート済み配列で合計が target になるペアを探す — O(n)"""
    left, right = 0, len(arr) - 1
    while left < right:
        s = arr[left] + arr[right]
        if s == target:
            return (left, right)
        elif s < target:
            left += 1
        else:
            right -= 1
    return None

# テスト
arr = [1, 2, 4, 6, 8, 10]
print(two_sum_sorted(arr, 10))  # (1, 4) → arr[1]+arr[4] = 2+8 = 10

# 応用: 3Sum (合計が 0 になる3つ組を列挙)
def three_sum(arr: list[int]) -> list[tuple[int, int, int]]:
    """3つの要素の合計が0になる組み合わせを列挙 — O(n^2)"""
    arr.sort()
    n = len(arr)
    result = []
    for i in range(n - 2):
        if i > 0 and arr[i] == arr[i - 1]:
            continue  # 重複スキップ
        left, right = i + 1, n - 1
        while left < right:
            s = arr[i] + arr[left] + arr[right]
            if s == 0:
                result.append((arr[i], arr[left], arr[right]))
                while left < right and arr[left] == arr[left + 1]:
                    left += 1  # 重複スキップ
                while left < right and arr[right] == arr[right - 1]:
                    right -= 1  # 重複スキップ
                left += 1
                right -= 1
            elif s < 0:
                left += 1
            else:
                right -= 1
    return result

print(three_sum([-1, 0, 1, 2, -1, -4]))
# [(-1, -1, 2), (-1, 0, 1)]
```

### 6.3 スライディングウィンドウ

固定長または可変長のウィンドウを配列上でスライドさせるテクニック。

```python
# 固定長ウィンドウ: 長さ k の連続部分配列の最大合計
def max_sum_subarray(arr: list[int], k: int) -> int:
    """固定長スライディングウィンドウ — O(n)"""
    n = len(arr)
    if n < k:
        return 0
    window_sum = sum(arr[:k])
    best = window_sum
    for i in range(k, n):
        window_sum += arr[i] - arr[i - k]
        best = max(best, window_sum)
    return best

print(max_sum_subarray([1, 4, 2, 10, 2, 3, 1, 0, 20], 4))  # 24

# 可変長ウィンドウ: 合計が target 以上の最短部分配列
def min_subarray_len(arr: list[int], target: int) -> int:
    """可変長スライディングウィンドウ — O(n)"""
    n = len(arr)
    left = 0
    current_sum = 0
    min_len = float('inf')
    for right in range(n):
        current_sum += arr[right]
        while current_sum >= target:
            min_len = min(min_len, right - left + 1)
            current_sum -= arr[left]
            left += 1
    return min_len if min_len != float('inf') else 0

print(min_subarray_len([2, 3, 1, 2, 4, 3], 7))  # 2 (部分配列 [4, 3])

# 可変長ウィンドウ: 重複なしの最長部分文字列
def longest_unique_substring(s: str) -> int:
    """重複なしの最長部分文字列の長さ — O(n)"""
    seen = {}
    left = 0
    max_len = 0
    for right, char in enumerate(s):
        if char in seen and seen[char] >= left:
            left = seen[char] + 1
        seen[char] = right
        max_len = max(max_len, right - left + 1)
    return max_len

print(longest_unique_substring("abcabcbb"))  # 3 ("abc")
print(longest_unique_substring("pwwkew"))    # 3 ("wke")
```

### 6.4 プレフィックス和（累積和）

区間の合計クエリを O(1) で処理するための前処理テクニック。

```python
class PrefixSum:
    """プレフィックス和 — 構築 O(n), クエリ O(1)"""

    def __init__(self, arr: list[int]):
        n = len(arr)
        self.prefix = [0] * (n + 1)
        for i in range(n):
            self.prefix[i + 1] = self.prefix[i] + arr[i]

    def range_sum(self, left: int, right: int) -> int:
        """arr[left:right+1] の合計を O(1) で返す"""
        return self.prefix[right + 1] - self.prefix[left]

# テスト
arr = [3, 1, 4, 1, 5, 9, 2, 6]
ps = PrefixSum(arr)
print(ps.range_sum(2, 5))  # arr[2]+arr[3]+arr[4]+arr[5] = 4+1+5+9 = 19
print(ps.range_sum(0, 7))  # 全体の合計 = 31

# 2次元プレフィックス和
class PrefixSum2D:
    """2次元プレフィックス和 — 構築 O(nm), クエリ O(1)"""

    def __init__(self, matrix: list[list[int]]):
        if not matrix:
            self.prefix = [[]]
            return
        rows, cols = len(matrix), len(matrix[0])
        self.prefix = [[0] * (cols + 1) for _ in range(rows + 1)]
        for i in range(rows):
            for j in range(cols):
                self.prefix[i+1][j+1] = (matrix[i][j]
                    + self.prefix[i][j+1]
                    + self.prefix[i+1][j]
                    - self.prefix[i][j])

    def range_sum(self, r1: int, c1: int, r2: int, c2: int) -> int:
        """matrix[r1:r2+1][c1:c2+1] の合計を O(1) で返す"""
        return (self.prefix[r2+1][c2+1]
                - self.prefix[r1][c2+1]
                - self.prefix[r2+1][c1]
                + self.prefix[r1][c1])

# テスト
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
ps2d = PrefixSum2D(matrix)
print(ps2d.range_sum(0, 0, 1, 1))  # 1+2+4+5 = 12
print(ps2d.range_sum(1, 1, 2, 2))  # 5+6+8+9 = 28
```

---

## 7. 各言語の実装

### 7.1 CPython list

CPython の list は `Objects/listobject.c` に実装されている。

```
CPython list の主要データ構造:

  typedef struct {
      PyObject_VAR_HEAD          // ob_refcnt, ob_type, ob_size
      PyObject **ob_item;        // ポインタの配列へのポインタ
      Py_ssize_t allocated;      // 確保済みスロット数
  } PyListObject;

  ob_size    = 現在の要素数 (len)
  allocated  = 確保済みのスロット数 (capacity)

  成長ポリシー (listobject.c, list_resize):
  new_allocated = new_size + (new_size >> 3) + (new_size < 9 ? 3 : 6)

  具体的な allocated の遷移:
  len=0 → allocated=0
  len=1 → allocated=4     (0 + 0 + 3 + 1 = 4)
  len=5 → allocated=8     (5 + 0 + 3 = 8)
  len=9 → allocated=16    (9 + 1 + 6 = 16)
  len=17 → allocated=25   (17 + 2 + 6 = 25)
  len=26 → allocated=35   (26 + 3 + 6 = 35)
  ...

  特徴:
  - 小さいリストでは余分なメモリを節約
  - 大きいリストでは約12.5%ずつ成長
  - 倍増戦略(×2)より省メモリだが、リサイズ頻度はやや高い
```

### 7.2 Java ArrayList

```
Java ArrayList<E> の内部構造:

  public class ArrayList<E> {
      private Object[] elementData;   // 要素の配列
      private int size;               // 現在の要素数

      // デフォルト初期容量
      private static final int DEFAULT_CAPACITY = 10;
  }

  成長ポリシー:
  private void grow(int minCapacity) {
      int oldCapacity = elementData.length;
      int newCapacity = oldCapacity + (oldCapacity >> 1);  // 1.5倍
      // ...
  }

  allocated の遷移:
  10 → 15 → 22 → 33 → 49 → 73 → 109 → ...

  特徴:
  - デフォルト初期容量 10
  - 成長因子 1.5 (50%増加)
  - trimToSize() で余分な容量を解放可能
  - ensureCapacity() で事前に容量を確保可能
  - ジェネリクスにより型安全
  - ただし、プリミティブ型は直接格納できない（ボクシングが必要）
    → int[] が必要な場合は配列を直接使う
```

### 7.3 Rust Vec\<T>

```
Rust Vec<T> の内部構造:

  pub struct Vec<T> {
      ptr: NonNull<T>,    // ヒープ上の配列へのポインタ
      len: usize,         // 現在の要素数
      cap: usize,         // 確保済みの要素数
  }

  // Vec のメモリレイアウト (スタック上: 24バイト)
  // ┌──────┬──────┬──────┐
  // │ ptr  │ len  │ cap  │  ← スタック上 (各8バイト)
  // └──┬───┴──────┴──────┘
  //    │
  //    ↓ ヒープ上
  //    ┌─────┬─────┬─────┬─────┬─────┬─────┐
  //    │ T   │ T   │ T   │ T   │     │     │
  //    └─────┴─────┴─────┴─────┴─────┴─────┘
  //    ← len=4 →←── 未使用 ──→
  //    ←──────── cap=6 ────────→

  成長ポリシー:
  - 成長因子: 2.0
  - ゼロコスト抽象化: Vec は &[T] (スライス) に暗黙変換
  - Drop トレイト: スコープを抜けると自動解放

  特徴:
  - 所有権システムにより、ダングリングポインタが発生しない
  - unsafe なしでバッファオーバーフローが発生しない
  - イテレータが最適化により for ループと同等の速度
  - Vec<u8> は文字列のバイト列として利用可能
```

### 7.4 Go slice

```
Go slice の内部構造:

  type slice struct {
      array unsafe.Pointer  // 基底配列へのポインタ
      len   int             // 現在の要素数
      cap   int             // 容量
  }

  // スライスの3要素構造:
  // ┌────────┬──────┬──────┐
  // │ array  │ len  │ cap  │
  // └───┬────┴──────┴──────┘
  //     │
  //     ↓
  //     ┌───┬───┬───┬───┬───┬───┐
  //     │ 1 │ 2 │ 3 │   │   │   │
  //     └───┴───┴───┴───┴───┴───┘
  //     ← len=3 →
  //     ←──── cap=6 ────→

  成長ポリシー (Go 1.18+):
  - cap < 256:      ×2 (倍増)
  - cap >= 256:     ×(1.25 + 192/cap)  (徐々に1.25倍に近づく)

  特徴:
  - スライスは参照型 (コピーしても基底配列を共有)
  - append は新しいスライスを返す (cap超過時は新配列割当)
  - re-slicing: s[1:3] はコピーを作らない
  - make([]int, len, cap) で長さと容量を別々に指定
```

---

## 8. メモリアラインメントとSIMD

### 8.1 メモリアラインメント

CPUは特定のバイト境界（alignment boundary）に揃ったアドレスからの
メモリアクセスを効率的に処理する。

```
メモリアラインメントの例:

  struct Example {
      char   a;    // 1 バイト
      int    b;    // 4 バイト
      char   c;    // 1 バイト
      double d;    // 8 バイト
  };

  ■ パディングなし (理論的):
  オフセット: 0  1  2  3  4  5  6  7  8  9  10 11 12 13
             [a][b  b  b  b][c][d  d  d  d  d  d  d  d]
             合計: 14 バイト

  ■ パディングあり (実際):
  オフセット: 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
             [a][..padding..][b  b  b  b][c][..padding........]
  オフセット: 16 17 18 19 20 21 22 23
             [d  d  d  d  d  d  d  d]
             合計: 24 バイト (14 + 10バイトのパディング)

  ■ フィールド順序を最適化:
  struct ExampleOptimized {
      double d;    // 8 バイト (オフセット 0)
      int    b;    // 4 バイト (オフセット 8)
      char   a;    // 1 バイト (オフセット 12)
      char   c;    // 1 バイト (オフセット 13)
      // 2バイトのパディング
  };
  合計: 16 バイト (14 + 2バイトのパディング)

  → フィールドの順序を変えるだけで 24B → 16B に削減
  → 大量の構造体を配列で持つ場合、33%のメモリ節約
```

### 8.2 SIMD (Single Instruction, Multiple Data)

SIMD は1つの命令で複数のデータを同時に処理する CPU の機能である。
配列の連続性がSIMD活用の前提条件となる。

```
SIMD の概念:

  通常の加算 (スカラー):
  a[0] + b[0] → c[0]     1回目
  a[1] + b[1] → c[1]     2回目
  a[2] + b[2] → c[2]     3回目
  a[3] + b[3] → c[3]     4回目
  → 4回の命令が必要

  SIMD 加算 (ベクトル):
  ┌─────────┬─────────┬─────────┬─────────┐
  │ a[0]    │ a[1]    │ a[2]    │ a[3]    │  128-bit レジスタ
  └────┬────┴────┬────┴────┬────┴────┬────┘
       +         +         +         +       ← 1命令
  ┌────┴────┬────┴────┬────┴────┬────┴────┐
  │ b[0]    │ b[1]    │ b[2]    │ b[3]    │  128-bit レジスタ
  └────┬────┴────┬────┴────┬────┴────┬────┘
       =         =         =         =
  ┌────┴────┬────┴────┬────┴────┬────┴────┐
  │ c[0]    │ c[1]    │ c[2]    │ c[3]    │  結果
  └─────────┴─────────┴─────────┴─────────┘
  → 1回の命令で完了 (理論上4倍速)

  主な SIMD 命令セット:
  ┌─────────────┬──────────┬──────────────────────┐
  │ 名前         │ ビット幅  │ 対応プロセッサ        │
  ├─────────────┼──────────┼──────────────────────┤
  │ SSE          │ 128-bit  │ x86 (2000年〜)       │
  │ AVX2         │ 256-bit  │ x86 (2013年〜)       │
  │ AVX-512      │ 512-bit  │ x86 (2017年〜)       │
  │ NEON         │ 128-bit  │ ARM                  │
  │ SVE/SVE2     │ 可変     │ ARM (2020年〜)       │
  └─────────────┴──────────┴──────────────────────┘

  float (32bit) の場合の同時処理数:
  SSE:     128 / 32 =  4 要素
  AVX2:    256 / 32 =  8 要素
  AVX-512: 512 / 32 = 16 要素
```

NumPy や BLAS ライブラリは内部的にSIMDを活用しており、
連続した配列に対する数値演算が極めて高速になる。

```python
import numpy as np
import timeit

n = 10_000_000

# Python list での合計
py_list = list(range(n))
t1 = timeit.timeit(lambda: sum(py_list), number=10)

# NumPy array での合計 (SIMD活用)
np_arr = np.arange(n, dtype=np.int64)
t2 = timeit.timeit(lambda: np_arr.sum(), number=10)

print(f"Python sum : {t1:.4f}s")
print(f"NumPy sum  : {t2:.4f}s")
print(f"速度比     : {t1/t2:.0f}x")
# 典型的な結果: NumPy は Python の 50-100 倍高速
```

---

## 9. トレードオフと比較分析

### 9.1 配列 vs 連結リスト

```
┌──────────────────┬──────────────┬──────────────┬──────────────────┐
│ 操作              │ 配列 (動的)   │ 連結リスト    │ 勝者              │
├──────────────────┼──────────────┼──────────────┼──────────────────┤
│ インデックス      │ O(1)         │ O(n)         │ 配列 ◎           │
│ アクセス          │              │              │                  │
├──────────────────┼──────────────┼──────────────┼──────────────────┤
│ 先頭に挿入       │ O(n)         │ O(1)         │ 連結リスト ◎     │
├──────────────────┼──────────────┼──────────────┼──────────────────┤
│ 末尾に挿入       │ O(1) 償却    │ O(1)*        │ 同等              │
│                  │              │ *末尾参照保持 │                  │
├──────────────────┼──────────────┼──────────────┼──────────────────┤
│ 中間に挿入       │ O(n)         │ O(1)**       │ 連結リスト ◎     │
│                  │              │ **位置既知時  │                  │
├──────────────────┼──────────────┼──────────────┼──────────────────┤
│ 検索             │ O(n)         │ O(n)         │ 配列 ○           │
│ (未ソート)       │ キャッシュ効率 │ キャッシュ非効率│ (定数倍が良い)    │
├──────────────────┼──────────────┼──────────────┼──────────────────┤
│ 検索             │ O(log n)     │ O(n)         │ 配列 ◎           │
│ (ソート済)       │ 二分探索可能  │              │                  │
├──────────────────┼──────────────┼──────────────┼──────────────────┤
│ メモリ使用量     │ 要素のみ     │ 要素+ポインタ │ 配列 ○           │
│                  │ (+余剰容量)  │ (8-16B/ノード)│                  │
├──────────────────┼──────────────┼──────────────┼──────────────────┤
│ キャッシュ効率   │ 極めて高い   │ 低い         │ 配列 ◎           │
├──────────────────┼──────────────┼──────────────┼──────────────────┤
│ メモリ断片化     │ なし         │ あり         │ 配列 ○           │
└──────────────────┴──────────────┴──────────────┴──────────────────┘

結論: ほとんどのユースケースで配列 (動的配列) が優位。
連結リストが有利なのは:
1. 先頭への頻繁な挿入/削除 (デキュー、スタック)
2. 要素の移動コストが大きい場合 (巨大な構造体)
3. メモリの断片化が許容でき、リアルタイム性が求められる場合
   (配列のリサイズは一度に O(n) のコピーが発生する)
```

### 9.2 文字列エンコーディング比較（実用的視点）

```
┌─────────────┬──────────────┬──────────────┬──────────────────┐
│ 基準         │ UTF-8        │ UTF-16       │ UTF-32           │
├─────────────┼──────────────┼──────────────┼──────────────────┤
│ ASCII文書    │ 1B/文字 ◎   │ 2B/文字      │ 4B/文字          │
│ サイズ       │              │              │                  │
├─────────────┼──────────────┼──────────────┼──────────────────┤
│ 日本語文書   │ 3B/文字      │ 2B/文字 ◎   │ 4B/文字          │
│ サイズ       │              │              │                  │
├─────────────┼──────────────┼──────────────┼──────────────────┤
│ 絵文字       │ 4B/文字      │ 4B/文字      │ 4B/文字          │
│ サイズ       │              │ (サロゲート)  │                  │
├─────────────┼──────────────┼──────────────┼──────────────────┤
│ ランダム     │ × (可変長)   │ △           │ ○ (固定長)       │
│ アクセス     │              │ (ほぼ固定)   │                  │
├─────────────┼──────────────┼──────────────┼──────────────────┤
│ ASCII互換    │ ○            │ ×           │ ×                │
├─────────────┼──────────────┼──────────────┼──────────────────┤
│ バイト順序   │ 不要         │ 必要 (BOM)   │ 必要 (BOM)       │
│ マーク(BOM)  │              │              │                  │
├─────────────┼──────────────┼──────────────┼──────────────────┤
│ ネットワーク │ 標準 ◎      │ 使用少       │ 使用極少         │
├─────────────┼──────────────┼──────────────┼──────────────────┤
│ ファイル     │ 標準 ◎      │ Windows系    │ ほぼ使用なし     │
│ 保存         │              │              │                  │
├─────────────┼──────────────┼──────────────┼──────────────────┤
│ 主な採用先   │ Linux, Web   │ Windows      │ 内部処理のみ     │
│              │ macOS, Go    │ Java, JS     │                  │
│              │ Rust, Python │ C# (.NET)    │                  │
└─────────────┴──────────────┴──────────────┴──────────────────┘

実用上の推奨:
- 外部データ交換: UTF-8 一択
- Windows API: UTF-16 (WideChar)
- 内部処理: 言語のデフォルトに従う
- パフォーマンスクリティカル: UTF-8 + バイト単位処理
```

### 9.3 各言語の動的配列の比較

```
┌──────────────┬─────────────┬───────────┬──────────┬────────────┐
│ 特性          │ Python list │ Java      │ C++      │ Rust Vec   │
│              │             │ ArrayList │ vector   │            │
├──────────────┼─────────────┼───────────┼──────────┼────────────┤
│ 要素型       │ 任意        │ Object    │ 任意の型  │ 任意の型   │
│              │ (PyObject*) │ (参照型)  │ (テンプレ)│ (ジェネリク)│
├──────────────┼─────────────┼───────────┼──────────┼────────────┤
│ 連続性       │ ポインタのみ│ 参照のみ  │ 値が連続  │ 値が連続    │
├──────────────┼─────────────┼───────────┼──────────┼────────────┤
│ 成長因子     │ ~1.125      │ 1.5       │ 2 (GCC)  │ 2          │
├──────────────┼─────────────┼───────────┼──────────┼────────────┤
│ 初期容量     │ 0           │ 10        │ 0        │ 0          │
├──────────────┼─────────────┼───────────┼──────────┼────────────┤
│ 境界チェック  │ あり        │ あり      │ なし*    │ あり       │
│              │             │           │ *at()あり│            │
├──────────────┼─────────────┼───────────┼──────────┼────────────┤
│ null安全     │ N/A         │ 要素にnull│ N/A      │ Option<T>  │
│              │ (None可)    │ 可能      │          │ で安全     │
├──────────────┼─────────────┼───────────┼──────────┼────────────┤
│ スレッド安全 │ GIL         │ ×※       │ ×        │ 所有権で   │
│              │             │           │          │ コンパイル │
│              │             │           │          │ 時保証     │
├──────────────┼─────────────┼───────────┼──────────┼────────────┤
│ メモリ管理   │ GC (参照    │ GC        │ 手動/    │ 所有権     │
│              │  カウント)  │           │ RAII     │ (Drop)     │
└──────────────┴─────────────┴───────────┴──────────┴────────────┘

※ Java: Collections.synchronizedList() または CopyOnWriteArrayList を使用
```

---

## 10. アンチパターン

### アンチパターン1: ループ内での文字列連結

**問題:** イミュータブルな文字列をループ内で `+=` で連結すると、
毎回新しい文字列が生成される。n回の連結で O(n^2) のコストがかかる。

```python
# NG: O(n^2) — 毎回文字列全体をコピー
def build_csv_bad(rows: list[list[str]]) -> str:
    result = ""
    for row in rows:
        result += ",".join(row) + "\n"  # 毎回新しい文字列を生成
    return result

# OK: O(n) — リストに貯めて最後に結合
def build_csv_good(rows: list[list[str]]) -> str:
    lines = []
    for row in rows:
        lines.append(",".join(row))
    return "\n".join(lines)

# さらに良い: ジェネレータ式
def build_csv_best(rows: list[list[str]]) -> str:
    return "\n".join(",".join(row) for row in rows)
```

**影響の規模:**
- n=1,000: NG版は約2倍遅い
- n=10,000: NG版は約20倍遅い
- n=100,000: NG版は約200倍遅い

Java でも同様:

```java
// NG: O(n^2)
String result = "";
for (String s : list) {
    result += s;  // 毎回新しい String オブジェクト生成
}

// OK: O(n)
StringBuilder sb = new StringBuilder();
for (String s : list) {
    sb.append(s);
}
String result = sb.toString();
```

### アンチパターン2: 配列の先頭への頻繁な挿入/削除

**問題:** 配列の先頭に要素を挿入/削除すると、全要素をシフトする必要がある。
これが繰り返されると O(n^2) になる。

```python
# NG: O(n^2) — 毎回全要素をシフト
def build_reversed_bad(items: list) -> list:
    result = []
    for item in items:
        result.insert(0, item)  # O(n) × n回 = O(n^2)
    return result

# OK: O(n) — 末尾に追加して反転
def build_reversed_good(items: list) -> list:
    result = []
    for item in items:
        result.append(item)  # O(1) 償却
    result.reverse()  # O(n) — 1回だけ
    return result

# さらに良い: collections.deque を使う
from collections import deque
def build_reversed_deque(items: list) -> deque:
    result = deque()
    for item in items:
        result.appendleft(item)  # O(1)
    return result

# ベンチマーク
import timeit
n = 50000
items = list(range(n))
t1 = timeit.timeit(lambda: build_reversed_bad(items), number=5)
t2 = timeit.timeit(lambda: build_reversed_good(items), number=5)
t3 = timeit.timeit(lambda: build_reversed_deque(items), number=5)
print(f"insert(0,x): {t1:.3f}s")
print(f"append+rev : {t2:.3f}s")
print(f"deque      : {t3:.3f}s")
```

### アンチパターン3: 不要なコピーの生成

```python
# NG: スライスは新しいリストを生成する（意図しないコピー）
def process_subarray_bad(arr: list, start: int, end: int) -> int:
    sub = arr[start:end]  # O(end-start) のコピーが発生
    return sum(sub)

# OK: インデックスで範囲を指定して処理
def process_subarray_good(arr: list, start: int, end: int) -> int:
    total = 0
    for i in range(start, end):
        total += arr[i]  # コピーなし
    return total

# NumPy の場合: ビュー vs コピー
import numpy as np
arr = np.array([1, 2, 3, 4, 5])

view = arr[1:4]      # ビュー（コピーなし、元配列と共有）
view[0] = 99         # arr も変更される！
print(arr)            # [1, 99, 3, 4, 5]

copy = arr[1:4].copy()  # 明示的なコピー
copy[0] = 0           # arr は変更されない
print(arr)            # [1, 99, 3, 4, 5]
```

---

## 11. エッジケース分析

### エッジケース1: 空配列と単一要素配列

配列操作で最も見落としやすいのが、空配列と単一要素配列の処理である。

```python
def find_max_bad(arr: list[int]) -> int:
    """NG: 空配列で例外"""
    max_val = arr[0]  # IndexError if arr is empty!
    for x in arr[1:]:
        max_val = max(max_val, x)
    return max_val

def find_max_good(arr: list[int]) -> int | None:
    """OK: 空配列を適切に処理"""
    if not arr:
        return None
    max_val = arr[0]
    for x in arr[1:]:
        max_val = max(max_val, x)
    return max_val

# テスト
print(find_max_good([]))         # None
print(find_max_good([42]))       # 42
print(find_max_good([3, 1, 4]))  # 4

# 二分探索でのエッジケース
def binary_search_safe(arr: list[int], target: int) -> int:
    """空配列でも安全な二分探索"""
    if not arr:
        return -1
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

# 回文判定のエッジケース
def is_palindrome(s: str) -> bool:
    """空文字列と1文字は回文"""
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True  # 空文字列と1文字は True

assert is_palindrome("") == True
assert is_palindrome("a") == True
assert is_palindrome("ab") == False
assert is_palindrome("aba") == True
```

### エッジケース2: 整数オーバーフロー

多くの言語で、配列のインデックス計算や要素の合計で整数オーバーフローが発生しうる。

```c
/* C言語での二分探索のオーバーフロー問題 */

/* NG: lo + hi がオーバーフローする可能性 */
int mid_bad = (lo + hi) / 2;
/* lo = 2,000,000,000, hi = 2,000,000,000 の場合
   lo + hi = 4,000,000,000 > INT_MAX (2,147,483,647)
   → 負の値になり、不正なインデックスに */

/* OK: オーバーフローしない計算 */
int mid_good = lo + (hi - lo) / 2;
/* hi - lo は常に非負で、lo + (hi-lo)/2 は lo 以上 hi 以下 */

/* Java でも同じ問題が存在する:
   java.util.Arrays.binarySearch() は
   JDK 6 で (lo + hi) >>> 1 に修正された
   (符号なし右シフトでオーバーフローを回避) */
```

```python
# Python は任意精度整数のためオーバーフローは発生しないが、
# 配列の合計がメモリを圧迫する場合がある

# 巨大な配列の合計
import sys
huge_list = [10**100] * 1000000
total = sum(huge_list)
# Python の int は任意精度なのでオーバーフローしないが、
# 各要素がPyObjectとして巨大なメモリを消費する

# NumPy の場合はオーバーフローする
import numpy as np
arr = np.array([2**62, 2**62], dtype=np.int64)
print(arr.sum())  # オーバーフロー! → 負の値になる
# 対策: dtype=np.int128 は存在しないため、Python の int に変換
print(sum(int(x) for x in arr))  # 正しい結果
```

### エッジケース3: Unicode文字列の「文字数」

```python
# 「文字数」は何を意味するかに依存する

text = "👨‍👩‍👧‍👦"  # 家族の絵文字

# バイト数 (UTF-8)
print(len(text.encode('utf-8')))    # 25

# コードポイント数
print(len(text))                     # 7 (Python は Unicode コードポイント単位)

# 書記素クラスタ (grapheme cluster) 数 — 人間が認識する「文字」の数
# 標準ライブラリだけでは取得困難。
# pip install grapheme
# import grapheme
# print(grapheme.length(text))       # 1

# 日本語の例
text_ja = "が"  # U+304C (が) — 1コードポイント
text_ja2 = "が"  # U+304B (か) + U+3099 (濁点) — 2コードポイント
# 見た目は同じだが len() が異なる場合がある

# 正規化
import unicodedata
nfc = unicodedata.normalize('NFC', text_ja2)   # 合成形: 1コードポイント
nfd = unicodedata.normalize('NFD', text_ja)    # 分解形: 2コードポイント
print(len(nfc), len(nfd))  # 1, 2

# 安全な文字列比較: 正規化してから比較
def safe_compare(s1: str, s2: str) -> bool:
    return unicodedata.normalize('NFC', s1) == unicodedata.normalize('NFC', s2)
```

---

## 12. 演習問題

### 演習1: 基礎レベル

**問題1-1: 配列の回転**

配列を k 個分だけ右に回転する関数を実装せよ。
例: `[1,2,3,4,5,6,7]`, k=3 → `[5,6,7,1,2,3,4]`

ヒント: 3回の反転で実現できる。

```python
def rotate_array(arr: list, k: int) -> None:
    """配列をインプレースで k 個分右に回転 — O(n) 時間, O(1) 空間"""
    n = len(arr)
    if n == 0:
        return
    k = k % n  # k が n より大きい場合に対応

    def reverse(start: int, end: int) -> None:
        while start < end:
            arr[start], arr[end] = arr[end], arr[start]
            start += 1
            end -= 1

    # Step 1: 全体を反転   [1,2,3,4,5,6,7] → [7,6,5,4,3,2,1]
    reverse(0, n - 1)
    # Step 2: 前半を反転   [7,6,5,4,3,2,1] → [5,6,7,4,3,2,1]
    reverse(0, k - 1)
    # Step 3: 後半を反転   [5,6,7,4,3,2,1] → [5,6,7,1,2,3,4]
    reverse(k, n - 1)

# テスト
arr = [1, 2, 3, 4, 5, 6, 7]
rotate_array(arr, 3)
print(arr)  # [5, 6, 7, 1, 2, 3, 4]
```

**問題1-2: 回文判定**

英数字以外を無視し、大文字小文字を区別せずに回文かどうかを判定する関数を実装せよ。

```python
def is_palindrome_alnum(s: str) -> bool:
    """英数字のみで回文判定 — O(n) 時間, O(1) 空間"""
    left, right = 0, len(s) - 1
    while left < right:
        # 英数字以外をスキップ
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        if s[left].lower() != s[right].lower():
            return False
        left += 1
        right -= 1
    return True

# テスト
print(is_palindrome_alnum("A man, a plan, a canal: Panama"))  # True
print(is_palindrome_alnum("race a car"))                       # False
print(is_palindrome_alnum(""))                                 # True
```

### 演習2: 応用レベル

**問題2-1: 最長回文部分文字列**

与えられた文字列の中で最長の回文部分文字列を求めよ。

```python
def longest_palindrome_substring(s: str) -> str:
    """最長回文部分文字列 — O(n^2) 時間, O(1) 空間
    (Manacher のアルゴリズムで O(n) も可能)
    """
    if not s:
        return ""

    start, max_len = 0, 1

    def expand_around_center(left: int, right: int) -> tuple[int, int]:
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return left + 1, right - left - 1

    for i in range(len(s)):
        # 奇数長の回文
        l1, len1 = expand_around_center(i, i)
        if len1 > max_len:
            start, max_len = l1, len1

        # 偶数長の回文
        if i + 1 < len(s):
            l2, len2 = expand_around_center(i, i + 1)
            if len2 > max_len:
                start, max_len = l2, len2

    return s[start:start + max_len]

# テスト
print(longest_palindrome_substring("babad"))     # "bab" or "aba"
print(longest_palindrome_substring("cbbd"))      # "bb"
print(longest_palindrome_substring("racecar"))   # "racecar"
```

**問題2-2: 行列の90度回転**

n×n の行列を時計回りに90度回転する関数を実装せよ（インプレース）。

```python
def rotate_matrix(matrix: list[list[int]]) -> None:
    """n×n 行列を時計回りに90度回転 — O(n^2) 時間, O(1) 空間"""
    n = len(matrix)

    # Step 1: 転置 (行と列を入れ替え)
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

    # Step 2: 各行を反転
    for i in range(n):
        matrix[i].reverse()

# テスト
matrix = [
    [1,  2,  3,  4],
    [5,  6,  7,  8],
    [9,  10, 11, 12],
    [13, 14, 15, 16]
]
rotate_matrix(matrix)
for row in matrix:
    print(row)
# [13, 9,  5, 1]
# [14, 10, 6, 2]
# [15, 11, 7, 3]
# [16, 12, 8, 4]
```

### 演習3: 発展レベル

**問題3-1: スパイラル順で行列を出力**

m×n の行列の要素をスパイラル（渦巻き）順で出力する関数を実装せよ。

```python
def spiral_order(matrix: list[list[int]]) -> list[int]:
    """スパイラル順で行列要素を返す — O(mn) 時間"""
    if not matrix or not matrix[0]:
        return []

    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1

    while top <= bottom and left <= right:
        # 上の行: 左→右
        for j in range(left, right + 1):
            result.append(matrix[top][j])
        top += 1

        # 右の列: 上→下
        for i in range(top, bottom + 1):
            result.append(matrix[i][right])
        right -= 1

        # 下の行: 右→左
        if top <= bottom:
            for j in range(right, left - 1, -1):
                result.append(matrix[bottom][j])
            bottom -= 1

        # 左の列: 下→上
        if left <= right:
            for i in range(bottom, top - 1, -1):
                result.append(matrix[i][left])
            left += 1

    return result

# テスト
matrix = [
    [1,  2,  3,  4],
    [5,  6,  7,  8],
    [9,  10, 11, 12]
]
print(spiral_order(matrix))
# [1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]
```

**問題3-2: 最小ウィンドウ部分文字列**

文字列 s と t が与えられたとき、s の中で t の全文字を含む最小の部分文字列を求めよ。

```python
from collections import Counter

def min_window(s: str, t: str) -> str:
    """最小ウィンドウ部分文字列 — O(|s| + |t|)"""
    if not s or not t:
        return ""

    need = Counter(t)
    missing = len(t)  # まだ満たされていない文字数
    best_start, best_len = 0, float('inf')
    left = 0

    for right, char in enumerate(s):
        if need[char] > 0:
            missing -= 1
        need[char] -= 1

        # すべての文字が含まれたらウィンドウを縮小
        while missing == 0:
            window_len = right - left + 1
            if window_len < best_len:
                best_start, best_len = left, window_len

            need[s[left]] += 1
            if need[s[left]] > 0:
                missing += 1
            left += 1

    return "" if best_len == float('inf') else s[best_start:best_start + best_len]

# テスト
print(min_window("ADOBECODEBANC", "ABC"))  # "BANC"
print(min_window("a", "a"))                 # "a"
print(min_window("a", "aa"))                # "" (不可能)
```

---

## 13. FAQ

### Q1: Python の list と tuple はどう使い分けるべきか？

**A:** 意味的に異なるものとして使い分けるべきである。

- **list**: 同種の要素の可変長コレクション。例: ユーザー一覧、スコアの履歴
- **tuple**: 異種の要素の固定長レコード。例: (名前, 年齢)、(x座標, y座標)

パフォーマンス面では:
- tuple はイミュータブルなためハッシュ可能であり、辞書のキーや集合の要素に使える
- tuple の生成は list より若干高速（メモリアロケーションが単純）
- CPython は小さい tuple をキャッシュして再利用する（free list）

```python
import sys
print(sys.getsizeof([1, 2, 3]))   # 120 (余剰容量を含む)
print(sys.getsizeof((1, 2, 3)))   # 64  (必要最小限)
```

### Q2: 配列を使うべきか、ハッシュマップ（辞書）を使うべきか？

**A:** 以下の判断基準で決める。

- **配列を使う場合**: キーが0からの連続整数で、密な（ほとんどの位置に値がある）場合
  - 例: 26文字のカウント → `counts = [0] * 26`
  - 例: dp テーブル → `dp = [0] * (n + 1)`
  - メリット: O(1) アクセス、キャッシュ効率が高い、メモリオーバーヘッドが最小

- **ハッシュマップを使う場合**: キーが疎（sparse）、非整数、または範囲が広い場合
  - 例: 単語の出現頻度 → `Counter(words)`
  - 例: 座標 → `grid = {(x, y): value}`
  - メリット: 任意のキー、動的な追加/削除

```python
# 配列が適切: 英小文字の頻度カウント
def count_letters_array(s: str) -> list[int]:
    counts = [0] * 26
    for c in s:
        counts[ord(c) - ord('a')] += 1
    return counts

# 辞書が適切: 任意の文字の頻度カウント
from collections import Counter
def count_chars_dict(s: str) -> dict[str, int]:
    return Counter(s)
```

### Q3: 文字列の比較で `==` と `is` は何が違うのか？

**A:** `==` は値の等価性、`is` はオブジェクトの同一性を検査する。
文字列の比較には常に `==` を使うべきである。

```python
a = "hello"
b = "hello"
c = "hel" + "lo"

print(a == b)    # True  — 値が同じ
print(a is b)    # True  — CPython がインターン化（実装依存!）

# 動的に生成された文字列
d = "".join(["h", "e", "l", "l", "o"])
print(a == d)    # True  — 値が同じ
print(a is d)    # False — 異なるオブジェクト（インターン化されない場合）

# 重要: 'is' での文字列比較は実装依存のため、絶対に使ってはならない
# PyPy, Jython, IronPython では CPython と異なる結果になる
```

### Q4: 配列のソートはどのアルゴリズムが使われているか？

**A:** 主要な言語のソート実装:

- **Python (Timsort)**: マージソート + 挿入ソートのハイブリッド。O(n log n) 最悪、O(n) 最良（既にソートされている場合）。安定ソート。
- **Java (Arrays.sort)**: プリミティブ型はDual Pivot Quicksort、オブジェクト型はTimsort。
- **C++ (std::sort)**: Introsort（クイックソート + ヒープソート + 挿入ソート）。不安定ソート。安定ソートが必要なら `std::stable_sort`。
- **Rust (sort)**: Timsortベース。安定ソート。`sort_unstable` は不安定だがやや高速。

### Q5: なぜ多くの言語で文字列はイミュータブルなのか？

**A:** 以下の理由から、イミュータブル設計が合理的と判断されている。

1. **スレッドセーフ**: ロックなしで複数スレッドから安全に読み取れる
2. **ハッシュキャッシュ**: ハッシュ値を一度計算すれば変更されないため再利用可能。辞書のキーとして使える
3. **セキュリティ**: ファイルパスやURLなど、セキュリティに関わる文字列が不意に変更されることを防ぐ
4. **最適化**: コンパイラが文字列の不変性を利用した最適化を行える（インターニング、定数伝播等）
5. **APIの単純化**: 文字列を渡した後に変更される心配がないため、防御的コピーが不要

ただし、頻繁な変更が必要な場合のために各言語はミュータブルな文字列ビルダーを提供している:
- Python: `io.StringIO`, `list` + `join`
- Java: `StringBuilder`, `StringBuffer`
- C#: `StringBuilder`
- Rust: `String` (Rust の String はミュータブル)

---

## 14. まとめ

| 概念 | ポイント |
|------|---------|
| 配列の本質 | 連続メモリ上の同型要素の列。O(1)ランダムアクセス |
| キャッシュ効率 | 配列はキャッシュラインに沿うため、理論値以上の速度を発揮 |
| 静的配列 | コンパイル時にサイズ確定。スタック配置可能 |
| 動的配列 | 倍増戦略で償却O(1)のappend。成長因子は1.125〜2.0 |
| 多次元配列 | 行優先 vs 列優先。走査順序がキャッシュ効率に直結 |
| 文字列 | 多くの言語でイミュータブル。UTF-8が事実上の標準 |
| エンコーディング | UTF-8: ASCII互換・可変長、UTF-16: Windows/Java、UTF-32: 固定長 |
| 文字列探索 | KMP: 最悪O(n+m)保証、Boyer-Moore: 実用最速O(n/m) |
| 典型テクニック | 二分探索、尺取り法、スライディングウィンドウ、プレフィックス和 |
| SIMD | 連続配列であることがSIMD活用の前提。NumPy等が内部的に利用 |
| アンチパターン | ループ内文字列連結(O(n^2))、先頭挿入(O(n^2))、不要コピー |
| エッジケース | 空配列、整数オーバーフロー、Unicode正規化 |

**最も重要な教訓:** 配列は最も単純なデータ構造だが、その単純さこそが
現代のハードウェアとの相性の良さを生み、多くの場面で最適な選択肢となる。
データ構造を選択する際は、O記法だけでなくキャッシュの影響も考慮すべきである。

---

## 次に読むべきガイド

→ [[01-linked-lists.md]] — 連結リスト: 配列との対比で理解を深める
→ [[02-stacks-and-queues.md]] — スタックとキュー: 配列をベースにした抽象データ型
→ [[03-hash-tables.md]] — ハッシュテーブル: 配列+ハッシュ関数で高速検索

---

## 参考文献

1. Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 2: Getting Started, Chapter 32: String Matching.
   — アルゴリズムの教科書として最も信頼性の高い参考書。配列の基礎から文字列探索アルゴリズム（KMP, Rabin-Karp, Boyer-Moore）まで網羅的に解説。

2. Sedgewick, R., & Wayne, K. (2011). *Algorithms* (4th ed.). Addison-Wesley. Chapter 1.3: Bags, Queues, and Stacks; Chapter 5: Strings.
   — 実装重視のアルゴリズム教科書。Java による具体的な実装例と豊富な図解が特徴。動的配列の償却解析と文字列処理アルゴリズムの実践的な解説。

3. Bryant, R. E., & O'Hallaron, D. R. (2015). *Computer Systems: A Programmer's Perspective* (3rd ed.). Pearson. Chapter 6: The Memory Hierarchy.
   — キャッシュの仕組みと配列アクセスパターンの関係を深く理解するための必読書。空間的局所性・時間的局所性の概念とキャッシュラインの動作を詳細に解説。

4. Pike, R., & Thompson, K. (2003). "Hello World" or Καλημέρα κόσμε or こんにちは 世界. *Proceedings of the USENIX Annual Technical Conference*.
   — UTF-8の設計者によるUnicodeエンコーディングの解説。UTF-8がなぜ優れた設計なのかを理解するための原典。

5. Knuth, D. E. (1997). *The Art of Computer Programming, Volume 1: Fundamental Algorithms* (3rd ed.). Addison-Wesley.
   — 配列と基本データ構造の数学的基盤。情報密度分析やアドレス計算の理論的背景を深く掘り下げる。

6. CPython Source Code: `Objects/listobject.c`, `Objects/unicodeobject.c`.
   — CPython における list と文字列の実装の一次資料。成長ポリシーやインターニングの具体的なコードを確認できる。URL: https://github.com/python/cpython
