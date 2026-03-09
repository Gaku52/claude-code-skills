# メモリ割り当て戦略 (Memory Allocation Strategies)

> **メモリアロケータの設計は、システム全体のスループット・レイテンシ・安定性を左右する最も重要な基盤技術の一つである。**
> 本章では、動的メモリ割り当ての原理からプロダクションレベルのアロケータ内部構造、ガベージコレクション、メモリリーク解析まで、包括的に解説する。

---

## この章で学ぶこと

- [ ] スタック / ヒープの構造的差異とメモリレイアウトを正確に説明できる
- [ ] First Fit / Best Fit / Worst Fit / Buddy System 等の割り当てアルゴリズムを比較できる
- [ ] ptmalloc2, jemalloc, tcmalloc, mimalloc の内部アーキテクチャを理解する
- [ ] ガベージコレクションの主要アルゴリズム（マーク&スイープ、世代別、参照カウント）を実装レベルで把握する
- [ ] メモリリーク・断片化の検出と対策を実践できる
- [ ] Rust の所有権モデルがなぜメモリ安全性を保証するかコンパイラ視点で理解する
- [ ] カーネルレベルのメモリ管理（sbrk, mmap, SLAB アロケータ）と連携を説明できる
- [ ] リアルタイムシステムや組み込み環境でのメモリ割り当て制約を理解する


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解
- [ページング ── 仮想メモリ・ページテーブル・TLB・ページ置換・スワッピング](./01-paging.md) の内容を理解していること

---

## 目次

1. [プロセスのメモリレイアウト](#1-プロセスのメモリレイアウト)
2. [動的メモリ割り当ての基礎](#2-動的メモリ割り当ての基礎)
3. [割り当てアルゴリズム詳解](#3-割り当てアルゴリズム詳解)
4. [主要なメモリアロケータ](#4-主要なメモリアロケータ)
5. [カーネルレベルのメモリ管理](#5-カーネルレベルのメモリ管理)
6. [ガベージコレクション深掘り](#6-ガベージコレクション深掘り)
7. [メモリ断片化と最適化](#7-メモリ断片化と最適化)
8. [メモリリークとデバッグ](#8-メモリリークとデバッグ)
9. [言語ランタイムとメモリモデル](#9-言語ランタイムとメモリモデル)
10. [アンチパターンと設計原則](#10-アンチパターンと設計原則)
11. [実践演習（3段階）](#11-実践演習3段階)
12. [FAQ](#12-faq)
13. [まとめ](#13-まとめ)
14. [参考文献](#14-参考文献)

---

## 1. プロセスのメモリレイアウト

### 1.1 仮想アドレス空間の全体像

すべてのプロセスは、OSから独立した仮想アドレス空間を与えられる。典型的な Linux x86-64 プロセスのメモリレイアウトは以下のとおりである。

```
  仮想アドレス空間（Linux x86-64, 48ビット）

  高位アドレス (0x7FFF_FFFF_FFFF)
  ┌─────────────────────────────────────────┐
  │           カーネル空間                    │  ← ユーザプロセスからアクセス不可
  │       (上位半分, 約128TB)                │
  ├─────────────────────────────────────────┤  0x7FFF_FFFF_FFFF
  │                                         │
  │           スタック (Stack)               │  ← 高位→低位方向に成長
  │           [ローカル変数, 引数,           │     デフォルト上限: 8MB (ulimit -s)
  │            リターンアドレス]              │
  │              ↓ 成長方向                  │
  ├─────────────────────────────────────────┤
  │                                         │
  │        メモリマップ領域                   │  ← mmap(), 共有ライブラリ,
  │        (Memory-Mapped Region)            │     大きなmalloc (>128KB)
  │                                         │
  ├─────────────────────────────────────────┤
  │              ↑ 成長方向                  │
  │           ヒープ (Heap)                  │  ← malloc/free で管理
  │           [動的に確保されるデータ]        │     brk/sbrk で拡張
  ├─────────────────────────────────────────┤  ← Program Break
  │           BSS セグメント                 │  ← 未初期化グローバル/静的変数
  │           (ゼロ初期化)                   │
  ├─────────────────────────────────────────┤
  │           データセグメント               │  ← 初期化済みグローバル/静的変数
  │           (初期化済みデータ)              │
  ├─────────────────────────────────────────┤
  │           テキストセグメント             │  ← 実行可能コード (読み取り専用)
  │           (プログラムコード)              │     文字列リテラルもここ
  ├─────────────────────────────────────────┤  0x0000_0040_0000 (典型的な開始点)
  │           NULL ガードページ              │  ← NULLポインタ参照を検出
  └─────────────────────────────────────────┘
  低位アドレス (0x0000_0000_0000)
```

### 1.2 各セグメントの詳細特性

| セグメント | 内容 | 権限 | サイズ | ライフタイム |
|:---:|:---|:---:|:---:|:---:|
| テキスト | 機械語命令, 文字列リテラル | R-X | 固定 | プロセス全体 |
| データ | 初期化済みグローバル/静的変数 | RW- | 固定 | プロセス全体 |
| BSS | 未初期化グローバル/静的変数 | RW- | 固定 | プロセス全体 |
| ヒープ | malloc/new で動的確保 | RW- | 可変 | 明示的解放まで |
| mmap | 共有ライブラリ, 大きな割り当て | 可変 | 可変 | munmap まで |
| スタック | ローカル変数, 引数, 戻りアドレス | RW- | 制限付き | 関数スコープ |

### 1.3 スタックとヒープの構造的差異

スタックとヒープはともに実行時にデータを格納するが、その管理方式は根本的に異なる。

```
  ┌─────────── スタック ─────────────┐    ┌─────────── ヒープ ──────────────┐
  │                                  │    │                                 │
  │  ┌──────────────────────┐       │    │  フリーリスト管理:               │
  │  │ main() のフレーム     │ ←SP  │    │                                 │
  │  │  local_a = 10        │       │    │  ┌────┐  ┌──────┐  ┌────┐     │
  │  │  local_b = 20        │       │    │  │使用│→│ 空き │→│使用│→... │
  │  ├──────────────────────┤       │    │  │ 8B │  │ 64B  │  │32B │     │
  │  │ foo() のフレーム      │       │    │  └────┘  └──────┘  └────┘     │
  │  │  buf[256]            │       │    │                                 │
  │  │  saved_rbp           │       │    │  malloc(24) の要求:              │
  │  │  return_addr         │       │    │  → 64B の空きブロックを分割     │
  │  ├──────────────────────┤       │    │  → 24B+8B(ヘッダ)=32Bを割当   │
  │  │ bar() のフレーム      │       │    │  → 残り 32B が新しい空きに     │
  │  │  tmp = 99            │       │    │                                 │
  │  └──────────────────────┘       │    │  free() の操作:                 │
  │         ↓ 成長方向              │    │  → ブロックを空きリストに返却   │
  │                                  │    │  → 隣接する空きと結合(coalesce)│
  │  操作: push/pop (O(1))          │    │                                 │
  │  管理: ハードウェア(SP レジスタ)  │    │  操作: 探索+分割/結合           │
  │  断片化: 発生しない              │    │  管理: ソフトウェア(アロケータ)  │
  │  速度: 極めて高速               │    │  断片化: 発生する               │
  └──────────────────────────────────┘    └─────────────────────────────────┘
```

**性能比較表: スタック vs ヒープ**

| 特性 | スタック | ヒープ |
|:---|:---:|:---:|
| 割り当て速度 | ~1 ns（SP移動のみ） | ~50-200 ns（探索+管理） |
| 解放速度 | ~1 ns（SP復帰のみ） | ~50-100 ns（リスト操作） |
| 最大サイズ | 1-8 MB（OS設定依存） | 物理メモリ+スワップまで |
| 断片化 | 発生しない | 外部/内部断片化が発生 |
| スレッド安全性 | 各スレッド独立 | 同期が必要（ロック等） |
| キャッシュ効率 | 極めて高い（局所性良好） | 低い（散在しやすい） |
| ライフタイム | 関数スコープに自動連動 | プログラマが明示管理 |
| オーバーフロー | スタックオーバーフロー | OOM（Out of Memory） |

### 1.4 コード例: メモリ配置の確認

```c
/* コード例1: プロセスメモリレイアウトの確認 (Linux) */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

/* グローバル変数 → データセグメント or BSS */
int initialized_global = 42;       /* データセグメント（初期化済み） */
int uninitialized_global;          /* BSS セグメント（ゼロ初期化） */
const char *string_literal = "Hello, Memory!";  /* テキストセグメント */

void demonstrate_layout(void) {
    /* ローカル変数 → スタック */
    int stack_var = 100;
    char stack_array[64];

    /* 動的割り当て → ヒープ */
    int *heap_var = (int *)malloc(sizeof(int) * 256);
    if (!heap_var) {
        perror("malloc failed");
        return;
    }

    /* 静的ローカル変数 → データセグメント */
    static int static_local = 55;

    printf("=== メモリレイアウト確認 ===\n");
    printf("テキスト (関数アドレス):       %p\n", (void *)demonstrate_layout);
    printf("テキスト (文字列リテラル):     %p\n", (void *)string_literal);
    printf("データ  (初期化済みグローバル): %p\n", (void *)&initialized_global);
    printf("データ  (静的ローカル):         %p\n", (void *)&static_local);
    printf("BSS    (未初期化グローバル):   %p\n", (void *)&uninitialized_global);
    printf("ヒープ  (malloc):              %p\n", (void *)heap_var);
    printf("スタック (ローカル変数):        %p\n", (void *)&stack_var);
    printf("スタック (配列):               %p\n", (void *)stack_array);
    printf("Program Break (brk):           %p\n", sbrk(0));

    free(heap_var);
}

int main(void) {
    demonstrate_layout();

    /* /proc/self/maps で詳細なメモリマップを確認 */
    printf("\n=== /proc/self/maps (抜粋) ===\n");
    char cmd[64];
    snprintf(cmd, sizeof(cmd), "cat /proc/%d/maps | head -20", getpid());
    system(cmd);

    return 0;
}
```

**実行結果例（アドレスは環境により異なる）:**

```
=== メモリレイアウト確認 ===
テキスト (関数アドレス):       0x55a3b2c01169
テキスト (文字列リテラル):     0x55a3b2c02008
データ  (初期化済みグローバル): 0x55a3b2c04010
データ  (静的ローカル):         0x55a3b2c04018
BSS    (未初期化グローバル):   0x55a3b2c04014
ヒープ  (malloc):              0x55a3b3a092a0
スタック (ローカル変数):        0x7ffd2e4b3c5c
スタック (配列):               0x7ffd2e4b3c10
Program Break (brk):           0x55a3b3a2a000
```

アドレスの大小関係から、テキスト < データ < BSS < ヒープ << スタック の配置順序が確認できる。ASLR (Address Space Layout Randomization) が有効な場合、毎回アドレスは変化するが、相対的な位置関係は保たれる。

---

## 2. 動的メモリ割り当ての基礎

### 2.1 システムコールインタフェース

ユーザ空間のメモリアロケータは、最終的にカーネルのシステムコールを通じて物理メモリを確保する。Linux における主要なインタフェースは `brk`/`sbrk` と `mmap` の2種類である。

**brk / sbrk:**

```c
/* brk(): Program Break を指定アドレスに設定 */
int brk(void *addr);

/* sbrk(): Program Break を increment バイト移動し、旧アドレスを返す */
void *sbrk(intptr_t increment);
```

`brk`/`sbrk` はヒープ領域の末尾（Program Break）を移動することで連続した仮想アドレス空間を拡張する。glibc の `malloc` は小〜中サイズの割り当てに `sbrk` を使用する。

**mmap:**

```c
/* mmap(): 仮想アドレス空間に新しいマッピングを作成 */
void *mmap(void *addr, size_t length, int prot, int flags,
           int fd, off_t offset);

/* munmap(): マッピングを解放 */
int munmap(void *addr, size_t length);
```

`mmap` は独立した仮想メモリ領域を確保する。glibc の `malloc` ではデフォルトで 128KB (MMAP_THRESHOLD) を超える割り当てに `mmap(MAP_ANONYMOUS)` を使用する。`mmap` の利点は、`munmap` で即座にカーネルへメモリを返却できることである（`sbrk` はヒープ末尾からしか縮小できない）。

### 2.2 malloc/free の内部動作

```c
/* コード例2: malloc/free のヘッダ構造（簡略版） */

/*
 * malloc が返すポインタの直前に、管理用ヘッダが置かれる。
 * このヘッダには、ブロックサイズや使用/空きフラグが格納される。
 *
 *    malloc(24) が返すメモリブロック:
 *
 *    ┌──────────────────┬───────────────────────────┐
 *    │  ヘッダ (8-16B)  │    ユーザデータ (24B)      │
 *    │  size | flags    │    ← malloc() の戻り値    │
 *    └──────────────────┴───────────────────────────┘
 *    ↑                  ↑
 *    実際の開始位置       ユーザに返されるポインタ
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

/* 簡易アロケータのブロックヘッダ */
typedef struct block_header {
    size_t size;              /* ブロックサイズ（ヘッダ含む） */
    int is_free;              /* 0: 使用中, 1: 空き */
    struct block_header *next; /* 次のブロックへのポインタ */
} block_header_t;

#define HEADER_SIZE sizeof(block_header_t)
#define ALIGN(size) (((size) + 7) & ~7)  /* 8バイトアラインメント */

static block_header_t *free_list_head = NULL;

/* 簡易 malloc 実装 (First Fit) */
void *simple_malloc(size_t size) {
    if (size == 0) return NULL;

    size_t aligned_size = ALIGN(size);
    size_t total_size = HEADER_SIZE + aligned_size;

    /* フリーリストから適切なブロックを探索 (First Fit) */
    block_header_t *current = free_list_head;
    block_header_t *prev = NULL;

    while (current != NULL) {
        if (current->is_free && current->size >= total_size) {
            /* 十分な大きさの空きブロックを発見 */

            /* 分割可能なら分割する */
            if (current->size >= total_size + HEADER_SIZE + 8) {
                block_header_t *new_block =
                    (block_header_t *)((char *)current + total_size);
                new_block->size = current->size - total_size;
                new_block->is_free = 1;
                new_block->next = current->next;

                current->size = total_size;
                current->next = new_block;
            }

            current->is_free = 0;
            return (void *)((char *)current + HEADER_SIZE);
        }
        prev = current;
        current = current->next;
    }

    /* フリーリストに適切なブロックがない → sbrk で拡張 */
    block_header_t *new_block = (block_header_t *)sbrk(total_size);
    if (new_block == (void *)-1) {
        return NULL;  /* メモリ不足 */
    }

    new_block->size = total_size;
    new_block->is_free = 0;
    new_block->next = NULL;

    if (prev != NULL) {
        prev->next = new_block;
    } else {
        free_list_head = new_block;
    }

    return (void *)((char *)new_block + HEADER_SIZE);
}

/* 簡易 free 実装 */
void simple_free(void *ptr) {
    if (ptr == NULL) return;

    block_header_t *header =
        (block_header_t *)((char *)ptr - HEADER_SIZE);
    header->is_free = 1;

    /* 隣接する空きブロックの結合 (Coalescing) */
    block_header_t *current = free_list_head;
    while (current != NULL) {
        if (current->is_free && current->next != NULL
            && current->next->is_free) {
            /* 隣接する2つの空きブロックを結合 */
            current->size += current->next->size;
            current->next = current->next->next;
            continue;  /* さらに結合できる可能性 */
        }
        current = current->next;
    }
}
```

### 2.3 アラインメント要件

現代のプロセッサは、データが特定のバイト境界に配置されていることを要求する。不適切なアラインメントはパフォーマンス低下やハードウェア例外を引き起こす。

| データ型 | 一般的なアラインメント | 理由 |
|:---|:---:|:---|
| char | 1 バイト | バイト単位でアクセス |
| short | 2 バイト | 16ビットバス幅 |
| int, float | 4 バイト | 32ビットレジスタ |
| long, double, ポインタ | 8 バイト | 64ビットレジスタ |
| SSE/AVX ベクトル | 16 / 32 バイト | SIMD 命令要件 |
| キャッシュライン | 64 バイト | False Sharing 防止 |

`malloc` が返すポインタは、プラットフォームの最大アラインメント要件（通常 16 バイト）を満たすように調整される。C11 の `aligned_alloc` や POSIX の `posix_memalign` を使えば、任意のアラインメント（2のべき乗）を指定可能である。

```c
/* アラインメントを指定した割り当て */
#include <stdlib.h>
#include <stdio.h>

int main(void) {
    /* 64バイトアラインメント（キャッシュライン境界） */
    void *ptr = NULL;
    int ret = posix_memalign(&ptr, 64, 1024);
    if (ret != 0) {
        fprintf(stderr, "posix_memalign failed\n");
        return 1;
    }
    printf("64-byte aligned pointer: %p (%%64 == %lu)\n",
           ptr, (unsigned long)ptr % 64);
    free(ptr);

    /* C11 aligned_alloc (サイズはアラインメントの倍数であること) */
    void *ptr2 = aligned_alloc(32, 1024);
    printf("32-byte aligned pointer: %p (%%32 == %lu)\n",
           ptr2, (unsigned long)ptr2 % 32);
    free(ptr2);

    return 0;
}
```

---

## 3. 割り当てアルゴリズム詳解

### 3.1 フリーリストベースのアルゴリズム

フリーリストは、未使用のメモリブロックを連結リストで管理するデータ構造である。新しい割り当て要求が来たとき、フリーリストからどのブロックを選ぶかが各アルゴリズムの差異となる。

```
  フリーリストの状態（各数字はブロックサイズ）:

  HEAD → [32B 空き] → [64B 使用] → [128B 空き] → [16B 空き] → [256B 空き] → NULL

  malloc(50) を要求した場合:

  ■ First Fit:  [128B 空き] を選択 ← 先頭から探索し最初に適合するブロック
  ■ Best Fit:   [64B ...] をスキップ → [128B 空き] ではなく... 全探索
                 適合する最小 = [128B 空き] を選択
  ■ Worst Fit:  [256B 空き] を選択 ← 最大の空きブロックを選択
  ■ Next Fit:   前回の探索終了位置から開始

  First Fit の動作:
  ┌────┐    ┌────┐    ┌──────────┐    ┌────┐    ┌────────┐
  │32B │ →  │64B │ →  │  128B    │ →  │16B │ →  │ 256B   │
  │空き│    │使用│    │  空き    │    │空き│    │ 空き   │
  └────┘    └────┘    └──────────┘    └────┘    └────────┘
                          ↑
                      ここを分割!
                      50B→使用 | 78B→空き
```

### 3.2 アルゴリズム比較表

| アルゴリズム | 探索時間 | 外部断片化 | 内部断片化 | 実装複雑度 | 特徴 |
|:---:|:---:|:---:|:---:|:---:|:---|
| First Fit | O(n) 平均高速 | 中 | 低 | 低 | リスト先頭に小ブロックが蓄積 |
| Next Fit | O(n) 分散 | 高 | 低 | 低 | 探索が分散するが断片化悪化 |
| Best Fit | O(n) 全探索 | 低 | 高 (小残片) | 低 | 微小な空きブロックが大量発生 |
| Worst Fit | O(n) 全探索 | 高 | 低 | 低 | 大ブロックがすぐ枯渇 |
| Buddy System | O(log n) | 中 | 高 (2のべき乗) | 中 | 結合が高速、Linux カーネル採用 |
| Segregated Fit | O(1) 同サイズ | 低 | 中 | 高 | サイズクラスごとにリスト管理 |
| TLSF | O(1) 保証 | 低 | 中 | 高 | リアルタイムシステム向け |

### 3.3 Buddy System（バディシステム）

Buddy System は Linux カーネルの物理ページアロケータで採用されている。メモリを2のべき乗サイズのブロックに分割して管理する。

```
  Buddy System: 128B のメモリプールから 20B を割り当てる例

  初期状態:
  ┌───────────────────────────────────────────────────┐
  │                   128B (order 7)                   │
  └───────────────────────────────────────────────────┘

  ステップ1: 128B → 64B + 64B に分割
  ┌─────────────────────────┬─────────────────────────┐
  │       64B (order 6)     │       64B (order 6)     │
  └─────────────────────────┴─────────────────────────┘

  ステップ2: 左の64B → 32B + 32B に分割
  ┌────────────┬────────────┬─────────────────────────┐
  │ 32B (o5)   │ 32B (o5)   │       64B (order 6)     │
  └────────────┴────────────┴─────────────────────────┘

  ステップ3: 左の32Bを割り当て (20Bの要求に対し32Bを割り当て)
  ┌────────────┬────────────┬─────────────────────────┐
  │■ 32B 使用 ■│ 32B 空き   │       64B 空き           │
  └────────────┴────────────┴─────────────────────────┘
  ← 20B に対して 32B → 内部断片化 12B (37.5%)

  解放時: バディ（隣接する同サイズブロック）が空きなら結合
  32B + 32B → 64B → 64B + 64B → 128B （完全に元に戻る）
```

**Buddy System の長所と短所:**

- 長所: 結合が O(1) で高速（バディのアドレスはビット演算で算出可能）
- 長所: 外部断片化が限定的（同サイズのバディ単位で管理）
- 短所: 内部断片化が大きい（要求サイズを2のべき乗に切り上げ）
- 短所: 33B の割り当てに 64B 必要（内部断片化 ~48%）

### 3.4 Segregated Free Lists（分離フリーリスト）

現代の高性能アロケータの多くが採用する手法。サイズクラスごとに個別のフリーリストを持つ。

```
  Segregated Free Lists の構造:

  サイズクラス    フリーリスト
  ┌──────────┐
  │   8B     │ → [空き] → [空き] → [空き] → NULL
  ├──────────┤
  │  16B     │ → [空き] → [空き] → NULL
  ├──────────┤
  │  32B     │ → NULL  (空きなし → 上位クラスから分割)
  ├──────────┤
  │  64B     │ → [空き] → [空き] → [空き] → [空き] → NULL
  ├──────────┤
  │  128B    │ → [空き] → NULL
  ├──────────┤
  │  256B    │ → [空き] → NULL
  ├──────────┤
  │  512B    │ → NULL
  ├──────────┤
  │  ...     │
  ├──────────┤
  │  large   │ → ツリーまたはソートリストで管理
  └──────────┘

  malloc(20) の場合:
  → サイズクラス 32B のリストから取得 (O(1))
  → リストが空なら 64B クラスから1つ取って分割
```

---

## 4. 主要なメモリアロケータ

### 4.1 ptmalloc2（glibc 標準アロケータ）

ptmalloc2 は Doug Lea の dlmalloc をマルチスレッド対応に拡張したもので、ほぼ全ての Linux ディストリビューションのデフォルトアロケータである。

**内部アーキテクチャ:**

```
  ptmalloc2 の構造:

  ┌─────────────────────────────────────────────────────┐
  │                   ptmalloc2                          │
  │                                                      │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
  │  │ Arena 0  │  │ Arena 1  │  │ Arena N  │  ...     │
  │  │ (main)   │  │ (thread) │  │ (thread) │          │
  │  │          │  │          │  │          │          │
  │  │ ┌──────┐ │  │ ┌──────┐ │  │ ┌──────┐ │          │
  │  │ │fast  │ │  │ │fast  │ │  │ │fast  │ │          │
  │  │ │bins  │ │  │ │bins  │ │  │ │bins  │ │          │
  │  │ │(LIFO)│ │  │ │(LIFO)│ │  │ │(LIFO)│ │          │
  │  │ ├──────┤ │  │ ├──────┤ │  │ ├──────┤ │          │
  │  │ │small │ │  │ │small │ │  │ │small │ │          │
  │  │ │bins  │ │  │ │bins  │ │  │ │bins  │ │          │
  │  │ │(FIFO)│ │  │ │(FIFO)│ │  │ │(FIFO)│ │          │
  │  │ ├──────┤ │  │ ├──────┤ │  │ ├──────┤ │          │
  │  │ │large │ │  │ │large │ │  │ │large │ │          │
  │  │ │bins  │ │  │ │bins  │ │  │ │bins  │ │          │
  │  │ │(sort)│ │  │ │(sort)│ │  │ │(sort)│ │          │
  │  │ ├──────┤ │  │ ├──────┤ │  │ ├──────┤ │          │
  │  │ │unsort│ │  │ │unsort│ │  │ │unsort│ │          │
  │  │ │bin   │ │  │ │bin   │ │  │ │bin   │ │          │
  │  │ └──────┘ │  │ └──────┘ │  │ └──────┘ │          │
  │  └──────────┘  └──────────┘  └──────────┘          │
  │                                                      │
  │  サイズ閾値:                                          │
  │    fastbin:  ≤ 160B (64bit)  → LIFO, ロック不要     │
  │    smallbin: ≤ 512B          → FIFO, 正確なサイズ   │
  │    largebin: > 512B          → ソートツリー          │
  │    mmap:     > 128KB         → mmap() で直接確保    │
  └─────────────────────────────────────────────────────┘
```

**ptmalloc2 の割り当てフロー:**

1. サイズが fastbin 範囲内 → fastbin から取得（ロック不要、最速）
2. サイズが smallbin 範囲内 → smallbin から取得
3. unsorted bin を走査 → 適切なブロックがあれば取得、なければ分類
4. largebin から Best Fit で取得
5. top chunk から分割
6. sbrk() でヒープ拡張、または mmap() で直接確保

### 4.2 jemalloc（FreeBSD / Redis / Firefox）

Jason Evans が FreeBSD 向けに設計したアロケータ。断片化の低さとスレッドスケーラビリティに優れる。

**主要な設計特徴:**

- **スレッドキャッシュ (tcache):** 各スレッドがローカルキャッシュを持ち、小さい割り当てはロック不要
- **アリーナ (arena):** 通常 CPU コア数 × 4 のアリーナを作成し、スレッドを分散配置
- **サイズクラス:** Small (≤14336B), Large (≤4MB), Huge (>4MB) の3段階
- **エクステント (extent):** メモリ管理の基本単位。ページ単位で管理
- **統計機能:** `malloc_stats_print()` で詳細なメモリ使用統計を取得可能

```c
/* コード例3: jemalloc の統計情報取得 */
#include <jemalloc/jemalloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    /* 様々なサイズのメモリを確保 */
    void *ptrs[1000];
    for (int i = 0; i < 1000; i++) {
        ptrs[i] = malloc(rand() % 4096 + 1);
    }

    /* 半分を解放（断片化を発生させる） */
    for (int i = 0; i < 1000; i += 2) {
        free(ptrs[i]);
        ptrs[i] = NULL;
    }

    /* jemalloc の統計情報を出力 */
    malloc_stats_print(NULL, NULL, NULL);

    /* 特定の統計値を取得 */
    size_t allocated, active, resident;
    size_t sz = sizeof(size_t);

    mallctl("stats.allocated", &allocated, &sz, NULL, 0);
    mallctl("stats.active", &active, &sz, NULL, 0);
    mallctl("stats.resident", &resident, &sz, NULL, 0);

    printf("\n=== メモリ使用状況 ===\n");
    printf("割り当て済み (allocated): %zu bytes\n", allocated);
    printf("アクティブ   (active):    %zu bytes\n", active);
    printf("常駐         (resident):  %zu bytes\n", resident);
    printf("断片化率: %.2f%%\n",
           (1.0 - (double)allocated / active) * 100);

    /* 残りを解放 */
    for (int i = 1; i < 1000; i += 2) {
        free(ptrs[i]);
    }

    return 0;
}
/* コンパイル: gcc -o test test.c -ljemalloc */
```

### 4.3 tcmalloc（Google）

Google が開発した Thread-Caching Malloc。小オブジェクトの高速割り当てに特化している。

**アーキテクチャ:**

```
  tcmalloc のアーキテクチャ:

  ┌──────────────────────────────────────────────────┐
  │                   tcmalloc                        │
  │                                                    │
  │  ┌─────────────┐ ┌─────────────┐ ┌────────────┐ │
  │  │ Thread Cache │ │ Thread Cache │ │Thread Cache│ │
  │  │  (per-thread)│ │  (per-thread)│ │(per-thread)│ │
  │  │  ロック不要   │ │  ロック不要   │ │ロック不要  │ │
  │  └──────┬──────┘ └──────┬──────┘ └─────┬──────┘ │
  │         │               │              │         │
  │         ▼               ▼              ▼         │
  │  ┌──────────────────────────────────────────┐    │
  │  │         Central Free List                │    │
  │  │     (サイズクラスごとにリスト管理)         │    │
  │  │     スピンロックで保護                    │    │
  │  └────────────────────┬─────────────────────┘    │
  │                       │                          │
  │                       ▼                          │
  │  ┌──────────────────────────────────────────┐    │
  │  │         Page Heap                        │    │
  │  │     (ページ単位の大きなブロック管理)       │    │
  │  │     span: 連続ページの集まり              │    │
  │  └──────────────────────────────────────────┘    │
  │                                                    │
  │  小オブジェクト (≤256KB):                          │
  │    Thread Cache → Central List → Page Heap        │
  │  大オブジェクト (>256KB):                          │
  │    Page Heap から直接割り当て                       │
  └──────────────────────────────────────────────────┘
```

### 4.4 mimalloc（Microsoft Research）

Microsoft Research が 2019 年に公開した最新のアロケータ。ベンチマークで他のアロケータを上回る性能を示している。

**設計上の特徴:**

- **セグメント方式:** 大きなメモリブロック（セグメント）を取得し、内部でページに分割
- **フリーリストのシャーディング:** ローカルフリーリストとスレッドフリーリストを分離
- **ページ内管理:** 同一サイズクラスのオブジェクトを同一ページに配置
- **遅延フリー:** 他スレッドからの `free` はスレッドフリーリストに追加され、所有スレッドが後で処理

### 4.5 アロケータ総合比較表

| 特性 | ptmalloc2 | jemalloc | tcmalloc | mimalloc |
|:---|:---:|:---:|:---:|:---:|
| 開発元 | glibc | FreeBSD/Meta | Google | Microsoft |
| スレッドキャッシュ | アリーナ単位 | tcache | Thread Cache | ローカルリスト |
| 小オブジェクト速度 | 中 | 高 | 極めて高 | 極めて高 |
| 大オブジェクト速度 | 中 | 高 | 中 | 高 |
| メモリ使用効率 | 中 | 高 | 中 | 高 |
| 断片化耐性 | 低-中 | 高 | 中 | 極めて高 |
| 統計/デバッグ | 限定的 | 非常に充実 | 充実 | 充実 |
| メモリ返却 | 遅い | 良好 | 良好 | 良好 |
| 主な採用例 | Linux 標準 | Redis, Firefox | Chrome, gRPC | .NET, 研究 |
| ライセンス | LGPL | BSD-2 | Apache 2.0 | MIT |
| リアルタイム適性 | 低 | 中 | 中 | 中-高 |

**アロケータの選択指針:**

- **デフォルト（特に理由がない場合）:** ptmalloc2（OS標準をそのまま使用）
- **Redis のように大量の小オブジェクト:** jemalloc（断片化が少ない）
- **マルチスレッドで小オブジェクト中心:** tcmalloc（スレッドキャッシュが効く）
- **最新のベンチマーク性能を求める場合:** mimalloc（全体的に高性能）
- **メモリ使用量の可視化が必要:** jemalloc（統計機能が最も充実）

---

## 5. カーネルレベルのメモリ管理

### 5.1 物理ページアロケータ（Buddy Allocator）

Linux カーネルは物理メモリをページ（通常 4KB）単位で管理する。Buddy Allocator は 2^0 〜 2^10 ページ（4KB〜4MB）の連続した物理ページブロックを管理する。

```
  Linux Buddy Allocator の構造 (/proc/buddyinfo):

  Order:    0     1     2     3     4     5     6     7     8     9    10
  Size:    4KB   8KB  16KB  32KB  64KB 128KB 256KB 512KB  1MB   2MB   4MB
          ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
  Zone    │     │     │     │     │     │     │     │     │     │     │     │
  DMA     │  3  │  1  │  0  │  0  │  2  │  1  │  0  │  1  │  1  │  1  │  3  │
          ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
  DMA32   │ 12  │  8  │  6  │  4  │  3  │  2  │  1  │  1  │  0  │  0  │  1  │
          ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
  Normal  │1024 │ 512 │ 256 │ 128 │  64 │  32 │  16 │   8 │   4 │   2 │   1 │
          └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘

  割り当て要求: 32KB (order 3) が必要
  1. order 3 のフリーリストを確認
  2. 空きがなければ order 4 (64KB) を取得して分割
  3. 64KB → 32KB + 32KB (片方が割り当て、片方がorder 3のフリーリストへ)

  解放時:
  1. バディ（同じ親から分割されたペア）が空きか確認
  2. 空きなら結合して上位orderへ → 再帰的に結合を試行
```

### 5.2 SLAB アロケータ

Buddy Allocator はページ単位（最小 4KB）の管理であり、カーネル内部で頻繁に使用される小さいオブジェクト（数十〜数百バイトの構造体）の管理には無駄が多い。SLAB アロケータはこの問題を解決するために Bonwick (1994) が設計した。

```
  SLAB アロケータの3層構造:

  ┌────────────────────────────────────────────────────────┐
  │                    Cache                                │
  │  (オブジェクト型ごとに1つ: task_struct, inode, dentry等) │
  │                                                         │
  │  ┌─────────────────────────────────────────────┐       │
  │  │  slabs_full     (全オブジェクト使用中)        │       │
  │  │  ┌──────┐ ┌──────┐ ┌──────┐                │       │
  │  │  │ slab │→│ slab │→│ slab │→ ...           │       │
  │  │  └──────┘ └──────┘ └──────┘                │       │
  │  ├─────────────────────────────────────────────┤       │
  │  │  slabs_partial  (一部オブジェクト使用中)     │       │
  │  │  ┌──────┐ ┌──────┐                         │       │
  │  │  │ slab │→│ slab │→ ...  ← 割り当てはここ │       │
  │  │  └──────┘ └──────┘                         │       │
  │  ├─────────────────────────────────────────────┤       │
  │  │  slabs_empty    (全オブジェクト空き)         │       │
  │  │  ┌──────┐                                   │       │
  │  │  │ slab │→ ...  ← メモリ圧迫時に回収      │       │
  │  │  └──────┘                                   │       │
  │  └─────────────────────────────────────────────┘       │
  │                                                         │
  │  1つのSLABの内部構造:                                   │
  │  ┌────┬────┬────┬────┬────┬────┬────┬────┐             │
  │  │obj │obj │obj │obj │ ...│obj │obj │管理│             │
  │  │ 0  │ 1  │ 2  │ 3  │    │n-1 │ n  │情報│             │
  │  └────┴────┴────┴────┴────┴────┴────┴────┘             │
  │  ← 1ページ (4KB) or 複数ページ →                        │
  │  各objは同一サイズ（例: task_struct = 約6KB）            │
  └────────────────────────────────────────────────────────┘
```

Linux カーネルでは SLAB の後継として **SLUB**（Unqueued SLAB、2.6.22以降のデフォルト）と **SLOB**（組み込み向けの簡易版）が存在する。

| 実装 | 特徴 | 用途 |
|:---:|:---|:---|
| SLAB | 元祖。オブジェクト着色、per-CPU キャッシュ | 従来のサーバ |
| SLUB | シンプルな設計、メタデータをオブジェクト内に格納 | 現在のデフォルト |
| SLOB | 最小限のメモリオーバーヘッド | 組み込み（数MB RAM） |

### 5.3 /proc によるカーネルメモリの観察

```bash
# コード例4: カーネルメモリ情報の確認コマンド集

# --- 物理メモリ全体の状態 ---
cat /proc/meminfo
# MemTotal:       16384000 kB    ← 物理メモリ総量
# MemFree:         2048000 kB    ← 完全に空きのメモリ
# MemAvailable:    8192000 kB    ← 利用可能（キャッシュ含む）
# Buffers:          512000 kB    ← ブロックデバイスバッファ
# Cached:          4096000 kB    ← ページキャッシュ
# Slab:             256000 kB    ← SLABアロケータ合計

# --- Buddy Allocator の状態 ---
cat /proc/buddyinfo
# Node 0, zone   Normal  1024  512  256  128  64  32  16  8  4  2  1

# --- SLAB の詳細統計 ---
cat /proc/slabinfo | head -20
# name            <active_objs> <num_objs> <objsize> ...
# task_struct          512        520       6016     ...
# inode_cache         2048       2060        592     ...
# dentry              4096       4100        192     ...

# --- プロセスごとのメモリマップ ---
cat /proc/self/maps | head -10
# 55a3b2c00000-55a3b2c02000 r--p  ... /path/to/binary   (テキスト)
# 55a3b2c02000-55a3b2c04000 r-xp  ... /path/to/binary   (コード)
# 7ffd2e490000-7ffd2e4b2000 rw-p  ... [stack]           (スタック)

# --- メモリ使用量トップ10プロセス ---
ps aux --sort=-%mem | head -11

# --- vmstat でメモリ活動をモニタ ---
vmstat 1 5
# procs ---memory--- ---swap-- -----io---- -system-- ------cpu-----
#  r  b   swpd   free   buff  cache   si   so    bi    bo   in   cs ...
```

### 5.4 Demand Paging とオーバーコミット

Linux はデフォルトでメモリのオーバーコミットを許可する。`malloc` が成功しても、実際の物理ページは書き込み時に初めて割り当てられる（Demand Paging）。物理メモリが枯渇すると OOM Killer がプロセスを強制終了する。

```
  Demand Paging の流れ:

  malloc(4096)
       │
       ▼
  仮想ページが確保される（物理ページはまだ割り当てなし）
  ページテーブルエントリ: Present=0
       │
       ▼
  *ptr = 42;  ← 最初の書き込み
       │
       ▼
  ページフォルト発生 (Present=0)
       │
       ▼
  カーネルが物理ページを割り当て
  ページテーブルを更新: Present=1, PFN=物理フレーム番号
       │
       ▼
  書き込みが完了

  vm.overcommit_memory の設定:
  0 (デフォルト): ヒューリスティックなオーバーコミット
  1: 常にオーバーコミットを許可（malloc は常に成功）
  2: オーバーコミット禁止（swap + RAM × ratio が上限）
```

---

## 6. ガベージコレクション深掘り

### 6.1 GC の基本分類

ガベージコレクション（GC）は、プログラムが使用しなくなったメモリを自動的に回収する仕組みである。手動のメモリ管理（malloc/free）では避けられないメモリリーク・ダブルフリー・Use-After-Free などのバグを根本的に防ぐ。

**GC の基本分類表:**

| 分類軸 | 選択肢 | 説明 |
|:---|:---|:---|
| 回収タイミング | Stop-the-World / Concurrent / Incremental | アプリ停止の有無・程度 |
| 到達性判定 | トレーシング / 参照カウント | ゴミの判定方法 |
| 移動の有無 | Compacting (移動あり) / Non-compacting | 断片化への対応 |
| 世代管理 | 世代別 / 非世代別 | オブジェクト寿命の活用 |
| 対象範囲 | 全体GC / 部分GC (Minor/Major) | 回収範囲の粒度 |

### 6.2 マーク & スイープ（Mark and Sweep）

最も基本的なトレーシング GC アルゴリズム。

```
  マーク & スイープ の動作:

  【マークフェーズ】
  ルートセット（スタック, グローバル変数, レジスタ）から
  到達可能な全オブジェクトにマークを付ける

  ルート
   │
   ▼
  [A]──→[B]──→[C]      [G]──→[H]
   │      │                     ↑
   ▼      ▼                     │
  [D]    [E]──→[F]      [I]──→[J]
                          ↑
  ルートから到達可能:       到達不能（ゴミ）:
  A, B, C, D, E, F        G, H, I, J

  マーク後:
  [A✓]──→[B✓]──→[C✓]   [G ]──→[H ]
   │       │                     ↑
   ▼       ▼                     │
  [D✓]   [E✓]──→[F✓]   [I ]──→[J ]

  【スイープフェーズ】
  ヒープ全体を走査し、マークされていないオブジェクトを回収

  ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
  │ A✓ │ G  │ B✓ │ H  │ D✓ │ I  │ E✓ │ J  │ C✓ │ F✓ │
  └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘
                     ↓ スイープ
  ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
  │ A  │空き│ B  │空き│ D  │空き│ E  │空き│ C  │ F  │
  └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘
  ※ マークをクリア、未マークを空きリストに追加

  問題点:
  - Stop-the-World: マーク中はアプリケーション停止
  - 断片化: 回収後のメモリが散在（→ Compaction で対策）
  - 全ヒープ走査: ヒープが大きいとスイープに時間がかかる
```

### 6.3 世代別 GC（Generational GC）

**弱い世代仮説（Weak Generational Hypothesis）:** ほとんどのオブジェクトは若くして死ぬ。この仮説に基づき、オブジェクトを世代（Generation）で分けて管理する。

```
  世代別 GC の構造（Java HotSpot JVM の例）:

  ┌────────────────────────────────────────────────────────────┐
  │                     Java ヒープ                             │
  │                                                             │
  │  ┌──────────────────────────────┐  ┌───────────────────┐  │
  │  │       Young Generation       │  │  Old Generation   │  │
  │  │         (若い世代)            │  │   (古い世代)       │  │
  │  │  ┌───────┬───────┬────────┐ │  │                   │  │
  │  │  │ Eden  │  S0   │   S1   │ │  │  テニュア世代     │  │
  │  │  │       │(From) │  (To)  │ │  │  (長寿オブジェクト)│  │
  │  │  │ 新規  │Survivor│Survivor│ │  │                   │  │
  │  │  │ 割当  │       │        │ │  │                   │  │
  │  │  └───────┴───────┴────────┘ │  │                   │  │
  │  │   ← Minor GC (高頻度, 高速) │  │  ← Major GC      │  │
  │  └──────────────────────────────┘  │    (低頻度, 遅い) │  │
  │                                     └───────────────────┘  │
  │                                                             │
  │  オブジェクトのライフサイクル:                                │
  │  1. Eden で生まれる                                         │
  │  2. Minor GC で生き残り → Survivor (S0/S1 間を往復)        │
  │  3. 一定回数の Minor GC を生存 → Old Generation に昇格      │
  │  4. Old Generation が満杯 → Major GC (Full GC)             │
  └────────────────────────────────────────────────────────────┘

  ライトバリア (Write Barrier):
  Old → Young への参照が書き込まれたとき記録
  → Minor GC 時に Old 全体をスキャンせずに済む
  → Card Table や Remembered Set で管理
```

**Java GC の進化:**

| GC 実装 | 世代 | 停止時間 | スループット | 特徴 |
|:---|:---:|:---:|:---:|:---|
| Serial GC | あり | 長い | 低 | シングルスレッド、小規模向け |
| Parallel GC | あり | 中 | 高 | マルチスレッドで並列回収 |
| CMS | あり | 短い | 中 | Concurrent Mark Sweep、JDK 14で廃止 |
| G1 GC | リージョン | 短い | 高 | リージョン単位、JDK 9以降デフォルト |
| ZGC | リージョン | 極短(<1ms) | 高 | カラーポインタ、TB級ヒープ対応 |
| Shenandoah | リージョン | 極短 | 高 | Red Hat 開発、並行コンパクション |

### 6.4 参照カウント（Reference Counting）

各オブジェクトが「自分を参照しているポインタの数」を保持する方式。参照数が 0 になったら即座に回収できる。

```python
# コード例5: Python における参照カウントと循環参照

import sys
import gc

# --- 参照カウントの確認 ---
a = [1, 2, 3]
print(f"参照カウント: {sys.getrefcount(a)}")  # 2 (a + getrefcount引数)

b = a  # 参照を追加
print(f"参照カウント: {sys.getrefcount(a)}")  # 3

del b  # 参照を削除
print(f"参照カウント: {sys.getrefcount(a)}")  # 2

# --- 循環参照の問題 ---
class Node:
    def __init__(self, name):
        self.name = name
        self.ref = None
    def __del__(self):
        print(f"  Node({self.name}) が回収されました")

# 循環参照を作成
node_a = Node("A")
node_b = Node("B")
node_a.ref = node_b  # A → B
node_b.ref = node_a  # B → A  ← 循環参照!

# 外部参照を削除しても参照カウントは 0 にならない
del node_a
del node_b
# → 参照カウントだけでは回収されない!

# Python の GC (世代別トレーシング) が循環参照を検出して回収
print("gc.collect() 呼び出し前:")
collected = gc.collect()
print(f"gc.collect() で回収されたオブジェクト: {collected}")

# --- gc の世代別統計 ---
print(f"\nGC 世代別閾値: {gc.get_threshold()}")  # (700, 10, 10)
print(f"GC 世代別カウント: {gc.get_count()}")
# 第0世代: 700 割り当て後に GC
# 第1世代: 第0世代 GC 10回ごとに GC
# 第2世代: 第1世代 GC 10回ごとに GC

# --- 弱参照による循環参照の回避 ---
import weakref

class SafeNode:
    def __init__(self, name):
        self.name = name
        self._ref = None

    @property
    def ref(self):
        if self._ref is not None:
            return self._ref()
        return None

    @ref.setter
    def ref(self, node):
        self._ref = weakref.ref(node) if node else None

safe_a = SafeNode("A")
safe_b = SafeNode("B")
safe_a.ref = safe_b
safe_b.ref = safe_a  # 弱参照なので循環参照にならない

del safe_a  # safe_b._ref() は None を返すようになる
```

### 6.5 Go の GC（Tri-color Mark and Sweep）

Go は非世代別の並行トレーシング GC を採用している。三色マーキングアルゴリズムにより、アプリケーションの停止時間を最小化する。

```
  三色マーキング (Tri-color Marking):

  白: 未訪問（GC完了後に回収対象）
  灰: 訪問済みだが子を未探索
  黒: 訪問済みで子もすべて探索済み

  初期状態: 全オブジェクトが白
  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐
  │A○│→│B○│→│C○│  │D○│  │E○│→│F○│
  └──┘  └──┘  └──┘  └──┘  └──┘  └──┘
  ルート: A, E

  ステップ1: ルートを灰色に
  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐
  │A◐│→│B○│→│C○│  │D○│  │E◐│→│F○│
  └──┘  └──┘  └──┘  └──┘  └──┘  └──┘

  ステップ2: 灰色オブジェクトの子を灰色に、自身を黒に
  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐
  │A●│→│B◐│→│C○│  │D○│  │E●│→│F◐│
  └──┘  └──┘  └──┘  └──┘  └──┘  └──┘

  ステップ3: 繰り返し
  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐
  │A●│→│B●│→│C◐│  │D○│  │E●│→│F●│
  └──┘  └──┘  └──┘  └──┘  └──┘  └──┘

  最終: 灰色がなくなったら完了
  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐
  │A●│→│B●│→│C●│  │D○│  │E●│→│F●│
  └──┘  └──┘  └──┘  └──┘  └──┘  └──┘
  白(D)を回収!  ○=白(回収) ◐=灰 ●=黒(生存)

  Go GC の特徴:
  - Stop-the-World は数十μs〜数百μs
  - ライトバリアで並行マーキング中の整合性を保持
  - GOGC 環境変数でGC頻度を調整（デフォルト100 = ヒープ2倍で発火）
```

### 6.6 Rust の所有権モデル

Rust は GC を使わず、コンパイル時の所有権（Ownership）と借用（Borrowing）ルールによってメモリ安全性を保証する。

```
  Rust 所有権の3つのルール:

  1. 各値には唯一の所有者（Owner）が存在する
  2. 所有者がスコープを抜けると値は自動的にドロップされる
  3. 所有権は移動（Move）するか、借用（Borrow）される

  所有権の移動 (Move):
  ┌──────────┐          ┌──────────┐
  │ let s1 = │          │ let s2 = │
  │ String:: │          │ s1;      │
  │ from("hi")          │          │
  └────┬─────┘          └────┬─────┘
       │                     │
       │  move               │
       ▼                     ▼
  ┌──────────┐          ┌──────────┐
  │ s1 (無効) │          │ s2 (有効) │
  │ ptr: ──  │          │ ptr: ──┐ │
  │ len: 2   │          │ len: 2 │ │
  │ cap: 2   │          │ cap: 2 │ │
  └──────────┘          └────────┼─┘
                                 │
                                 ▼
                          ┌──────────┐
                          │ ヒープ    │
                          │ "hi"     │
                          └──────────┘

  借用 (Borrow):
  - 不変借用 (&T): 複数同時OK、変更不可
  - 可変借用 (&mut T): 1つだけ、変更可能
  - 不変借用と可変借用は同時に存在できない
  → コンパイル時にデータ競合を防止
```

---

## 7. メモリ断片化と最適化

### 7.1 外部断片化 vs 内部断片化

```
  【外部断片化 (External Fragmentation)】
  空きメモリの合計は十分だが、連続した領域が不足

  malloc(120) を要求:
  ┌────┬────────┬────┬──────┬────┬──────────┐
  │使用│空き 64B│使用│空き  │使用│空き 80B  │
  │    │        │    │48B  │    │          │
  └────┴────────┴────┴──────┴────┴──────────┘
  空き合計: 64 + 48 + 80 = 192B >= 120B だが
  連続した 120B の空きがない → 割り当て失敗!

  【内部断片化 (Internal Fragmentation)】
  割り当てたブロック内部の無駄な余り

  malloc(20) を要求 → アロケータは 32B のブロックを割り当て
  ┌────────────────────────────────┐
  │ ユーザデータ 20B │ 余り 12B    │  ← 12B が無駄（内部断片化）
  └────────────────────────────────┘

  Buddy System の場合:
  malloc(33) → 64B を割り当て → 31B の無駄 (48% の内部断片化)
```

### 7.2 コンパクション（Compaction）

外部断片化を解消するために、使用中のオブジェクトを移動して連続した空き領域を作る技術。GC を持つ言語ランタイム（Java, .NET, Go）で広く使用される。

**コンパクションの種類:**

| 方式 | 説明 | 長所 | 短所 |
|:---|:---|:---|:---|
| 任意移動 | オブジェクトを任意の位置に移動 | 最適配置可能 | 参照の更新コスト大 |
| スライド | オブジェクトを一方向にスライド | 相対順序を保持 | 2パス必要 |
| コピー | 生存オブジェクトを別領域にコピー | 高速（1パス） | 2倍のメモリが必要 |

### 7.3 メモリプールパターン

高頻度の割り当て/解放が発生する場面では、汎用アロケータではなく専用のメモリプールを使うことで性能を大幅に向上できる。

```c
/* コード例6: 固定サイズメモリプール実装 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef struct memory_pool {
    size_t block_size;      /* 各ブロックのサイズ */
    size_t pool_size;       /* プール内のブロック数 */
    uint8_t *memory;        /* メモリ領域 */
    uint8_t *free_list;     /* フリーリストの先頭 */
    size_t allocated_count; /* 割り当て済みブロック数 */
} memory_pool_t;

/* プール初期化 */
memory_pool_t *pool_create(size_t block_size, size_t pool_size) {
    /* ブロックサイズはポインタサイズ以上が必要 */
    if (block_size < sizeof(void *)) {
        block_size = sizeof(void *);
    }

    memory_pool_t *pool = (memory_pool_t *)malloc(sizeof(memory_pool_t));
    pool->block_size = block_size;
    pool->pool_size = pool_size;
    pool->allocated_count = 0;

    /* 連続したメモリ領域を確保 */
    pool->memory = (uint8_t *)malloc(block_size * pool_size);

    /* フリーリストを構築（各ブロックの先頭に次のブロックへのポインタ） */
    pool->free_list = pool->memory;
    for (size_t i = 0; i < pool_size - 1; i++) {
        uint8_t *current = pool->memory + i * block_size;
        uint8_t *next = pool->memory + (i + 1) * block_size;
        *(uint8_t **)current = next;
    }
    /* 最後のブロックの次は NULL */
    *(uint8_t **)(pool->memory + (pool_size - 1) * block_size) = NULL;

    return pool;
}

/* ブロック取得 O(1) */
void *pool_alloc(memory_pool_t *pool) {
    if (pool->free_list == NULL) {
        return NULL;  /* プール枯渇 */
    }
    void *block = pool->free_list;
    pool->free_list = *(uint8_t **)pool->free_list;
    pool->allocated_count++;
    return block;
}

/* ブロック返却 O(1) */
void pool_free(memory_pool_t *pool, void *block) {
    *(uint8_t **)block = pool->free_list;
    pool->free_list = (uint8_t *)block;
    pool->allocated_count--;
}

/* プール破棄 */
void pool_destroy(memory_pool_t *pool) {
    free(pool->memory);
    free(pool);
}

/* 使用例: ゲームエンジンのパーティクルシステム */
typedef struct particle {
    float x, y, z;
    float vx, vy, vz;
    float lifetime;
    int active;
} particle_t;

int main(void) {
    /* 10000 個のパーティクル用プールを作成 */
    memory_pool_t *particle_pool =
        pool_create(sizeof(particle_t), 10000);

    printf("プール作成: ブロックサイズ=%zu, 容量=%zu\n",
           particle_pool->block_size, particle_pool->pool_size);

    /* パーティクルを高速に割り当て */
    particle_t *particles[100];
    for (int i = 0; i < 100; i++) {
        particles[i] = (particle_t *)pool_alloc(particle_pool);
        particles[i]->x = (float)i;
        particles[i]->active = 1;
    }

    printf("割り当て済み: %zu\n", particle_pool->allocated_count);

    /* パーティクルを返却 */
    for (int i = 0; i < 100; i++) {
        pool_free(particle_pool, particles[i]);
    }

    printf("返却後: %zu\n", particle_pool->allocated_count);

    pool_destroy(particle_pool);
    return 0;
}
```

### 7.4 Huge Pages（大ページ）

通常の 4KB ページでは、大量のメモリを使用するアプリケーション（データベース、JVM）で TLB ミスが頻発する。Huge Pages（2MB / 1GB）を使うことで TLB エントリ数を削減し、アドレス変換のオーバーヘッドを軽減できる。

```bash
# Huge Pages の確認と設定
cat /proc/meminfo | grep Huge
# HugePages_Total:     128
# HugePages_Free:       64
# HugePages_Rsvd:       32
# Hugepagesize:       2048 kB

# Transparent Huge Pages (THP) の状態確認
cat /sys/kernel/mm/transparent_hugepage/enabled
# [always] madvise never

# データベース（Redis, PostgreSQL）では THP を無効にすることが多い
# → レイテンシのばらつき（THP のコンパクション処理）を避けるため
```

| 項目 | 通常ページ (4KB) | Huge Page (2MB) | Giant Page (1GB) |
|:---|:---:|:---:|:---:|
| ページサイズ | 4 KB | 2 MB | 1 GB |
| TLB エントリ 1つで | 4 KB をカバー | 2 MB をカバー | 1 GB をカバー |
| 1GB をマップ | 262,144 エントリ | 512 エントリ | 1 エントリ |
| 用途 | 汎用 | DB, JVM | HPC, 大規模DB |

---

## 8. メモリリークとデバッグ

### 8.1 メモリリークの分類

メモリリークは「割り当てたメモリへの参照を失い、解放できなくなる」現象である。長時間実行されるサーバプロセスでは致命的な問題となる。

| リークの種類 | 原因 | 検出難易度 | 言語 |
|:---|:---|:---:|:---|
| 直接リーク | free/delete の呼び忘れ | 低 | C, C++ |
| 間接リーク | リーク済みポインタ経由の到達不能メモリ | 中 | C, C++ |
| 循環参照 | 相互参照でカウントが0にならない | 中 | Python, Swift, JS |
| イベントリスナリーク | removeEventListener の未呼び出し | 高 | JavaScript |
| クロージャリーク | クロージャが大きなスコープをキャプチャ | 高 | JS, Python, Go |
| キャッシュリーク | 無制限にデータを蓄積するキャッシュ | 高 | 全言語 |
| スレッドローカルリーク | スレッドプール内で蓄積 | 高 | Java, Go |

### 8.2 Valgrind / AddressSanitizer による検出

```c
/* リーク検出対象のサンプルコード (leak_example.c) */
#include <stdlib.h>
#include <string.h>

void direct_leak(void) {
    /* 直接リーク: malloc したが free しない */
    int *data = (int *)malloc(sizeof(int) * 100);
    data[0] = 42;
    /* free(data) を忘れている! */
}

void indirect_leak(void) {
    /* 間接リーク: リンクリストのヘッドを失う */
    struct node {
        int value;
        struct node *next;
    };

    struct node *head = (struct node *)malloc(sizeof(struct node));
    head->value = 1;
    head->next = (struct node *)malloc(sizeof(struct node));
    head->next->value = 2;
    head->next->next = NULL;

    /* head のみ free → head->next がリーク */
    /* 正しくは: リスト全体をトラバースして free */
    free(head);  /* head->next は間接リーク */
}

int main(void) {
    direct_leak();
    indirect_leak();
    return 0;
}
```

```bash
# --- Valgrind でメモリリーク検出 ---
gcc -g -O0 leak_example.c -o leak_example
valgrind --leak-check=full --show-leak-kinds=all ./leak_example

# Valgrind 出力例:
# ==12345== HEAP SUMMARY:
# ==12345==   in use at exit: 416 bytes in 2 blocks
# ==12345==   total heap usage: 3 allocs, 1 frees, 432 bytes allocated
# ==12345==
# ==12345== 400 bytes in 1 blocks are definitely lost
# ==12345==    at 0x4C2FB0F: malloc (in /usr/lib/valgrind/...)
# ==12345==    by 0x401156: direct_leak (leak_example.c:6)
# ==12345==    by 0x4011A3: main (leak_example.c:32)
# ==12345==
# ==12345== 16 bytes in 1 blocks are indirectly lost
# ==12345==    at 0x4C2FB0F: malloc (in /usr/lib/valgrind/...)
# ==12345==    by 0x401178: indirect_leak (leak_example.c:17)

# --- AddressSanitizer (ASan) でメモリエラー検出 ---
gcc -fsanitize=address -fno-omit-frame-pointer -g leak_example.c -o leak_asan
./leak_asan

# --- LeakSanitizer を単独で使用 ---
gcc -fsanitize=leak -g leak_example.c -o leak_lsan
./leak_lsan
```

### 8.3 JavaScript のメモリリークパターン

```javascript
// コード例7: JavaScript における典型的なメモリリークパターン

// --- パターン1: イベントリスナーの未解除 ---
class LeakyComponent {
    constructor() {
        this.data = new Array(1000000).fill('x'); // 大きなデータ
        // イベントリスナーを登録
        this.handler = () => this.handleResize();
        window.addEventListener('resize', this.handler);
    }

    handleResize() {
        console.log('resize', this.data.length);
    }

    // destroy を呼び忘れるとリーク
    destroy() {
        window.removeEventListener('resize', this.handler);
    }
}

// --- パターン2: クロージャによるキャプチャ ---
function createLeak() {
    const hugeArray = new Array(1000000).fill('leak');

    // この関数が生きている限り hugeArray も保持される
    return function() {
        // hugeArray を直接使わなくても、
        // 同じスコープの変数がキャプチャされることがある
        console.log('closure alive');
    };
}

// --- パターン3: 無制限キャッシュ ---
const cache = new Map();
function addToCache(key, value) {
    // キャッシュが無限に成長する
    cache.set(key, value);
}

// 修正: WeakMap または LRU キャッシュを使用
const weakCache = new WeakMap(); // キーがGCされるとエントリも消える

// 修正: LRU キャッシュ（最大サイズ制限付き）
class LRUCache {
    constructor(maxSize) {
        this.maxSize = maxSize;
        this.cache = new Map();
    }

    get(key) {
        if (this.cache.has(key)) {
            const value = this.cache.get(key);
            this.cache.delete(key);
            this.cache.set(key, value); // 末尾に移動
            return value;
        }
        return undefined;
    }

    set(key, value) {
        if (this.cache.has(key)) {
            this.cache.delete(key);
        } else if (this.cache.size >= this.maxSize) {
            // 最も古いエントリを削除
            const oldestKey = this.cache.keys().next().value;
            this.cache.delete(oldestKey);
        }
        this.cache.set(key, value);
    }
}

// --- Chrome DevTools でのデバッグ手順 ---
// 1. Memory タブ → "Take heap snapshot"
// 2. 操作を実行
// 3. 再度スナップショット取得
// 4. "Comparison" ビューで差分を確認
// 5. Retainers パネルで何がオブジェクトを保持しているか特定
```

### 8.4 Go の pprof によるメモリプロファイリング

```go
// コード例8: Go のメモリプロファイリング
package main

import (
    "fmt"
    "net/http"
    _ "net/http/pprof" // pprof エンドポイントを有効化
    "runtime"
    "time"
)

// 意図的にメモリを蓄積するキャッシュ
var leakyCache = make(map[string][]byte)

func simulateWork() {
    for i := 0; ; i++ {
        // キャッシュに無制限追加（リークの原因）
        key := fmt.Sprintf("key-%d", i)
        leakyCache[key] = make([]byte, 1024) // 1KB ずつ蓄積

        if i%1000 == 0 {
            var m runtime.MemStats
            runtime.ReadMemStats(&m)
            fmt.Printf("Alloc=%v MiB, Sys=%v MiB, NumGC=%v\n",
                m.Alloc/1024/1024,
                m.Sys/1024/1024,
                m.NumGC,
            )
        }
        time.Sleep(time.Millisecond)
    }
}

func main() {
    // pprof HTTP サーバを起動
    go func() {
        fmt.Println("pprof: http://localhost:6060/debug/pprof/")
        http.ListenAndServe(":6060", nil)
    }()

    simulateWork()
}

// プロファイリングコマンド:
// go tool pprof http://localhost:6060/debug/pprof/heap
// (pprof) top 10        ← メモリ使用量トップ10
// (pprof) web           ← グラフをブラウザで表示
// (pprof) list main     ← ソースコード注釈付き表示
```

---

## 9. 言語ランタイムとメモリモデル

### 9.1 各言語のメモリ管理方式比較

プログラミング言語ごとにメモリ管理のアプローチは大きく異なる。以下に主要言語の比較を示す。

| 言語 | 管理方式 | ヒープ割り当て | 解放タイミング | スタック割り当て | 安全性保証 |
|:---|:---|:---|:---|:---|:---|
| C | 手動 (malloc/free) | 明示的 | 明示的 (free) | 自動 | なし |
| C++ | 手動 + RAII | new/make_unique | デストラクタ | 自動 | RAII で部分的 |
| Rust | 所有権 + 借用 | Box, Vec 等 | スコープ終了時 | 自動 | コンパイル時保証 |
| Java | 世代別 GC | new | GC が回収 | JIT が最適化 | ランタイム保証 |
| Go | 並行 GC | make, new | GC が回収 | エスケープ解析 | ランタイム保証 |
| Python | 参照カウント + GC | 暗黙的 | 参照カウント=0 or GC | なし（全てヒープ） | ランタイム保証 |
| JavaScript | 世代別 GC (V8) | 暗黙的 | GC が回収 | JIT が最適化 | ランタイム保証 |
| Swift | ARC | 暗黙的 | 参照カウント=0 | 値型はスタック | コンパイル時 ARC |
| Zig | 手動 + アロケータ抽象 | アロケータ経由 | 明示的 (defer) | 自動 | コンパイル時チェック |

### 9.2 C++ の RAII とスマートポインタ

C++ は RAII（Resource Acquisition Is Initialization）パターンにより、手動メモリ管理の安全性を向上させる。

```cpp
// コード例9: C++ スマートポインタによるメモリ管理
#include <memory>
#include <vector>
#include <iostream>

class Resource {
public:
    Resource(int id) : id_(id) {
        std::cout << "Resource " << id_ << " acquired\n";
    }
    ~Resource() {
        std::cout << "Resource " << id_ << " released\n";
    }
    void use() { std::cout << "Using Resource " << id_ << "\n"; }
private:
    int id_;
};

void demonstrate_smart_pointers() {
    // --- unique_ptr: 排他的所有権（コピー不可、ムーブのみ） ---
    {
        auto r1 = std::make_unique<Resource>(1);
        r1->use();
        // スコープ終了時に自動解放
        // auto r2 = r1;  // コンパイルエラー! コピー不可
        auto r2 = std::move(r1);  // ムーブは可能
        // r1 は nullptr になる
    } // r2 のデストラクタが Resource を解放

    // --- shared_ptr: 共有所有権（参照カウント） ---
    {
        auto r3 = std::make_shared<Resource>(3);
        std::cout << "ref count: " << r3.use_count() << "\n"; // 1

        {
            auto r4 = r3;  // 共有
            std::cout << "ref count: " << r3.use_count() << "\n"; // 2
        } // r4 スコープ終了: カウント 2→1

        std::cout << "ref count: " << r3.use_count() << "\n"; // 1
    } // r3 スコープ終了: カウント 1→0 → 解放

    // --- weak_ptr: 循環参照の防止 ---
    {
        auto parent = std::make_shared<Resource>(5);
        std::weak_ptr<Resource> weak_ref = parent;

        if (auto locked = weak_ref.lock()) {
            locked->use();  // まだ有効
        }

        parent.reset();  // 解放

        if (auto locked = weak_ref.lock()) {
            locked->use();  // ここには到達しない
        } else {
            std::cout << "Resource already released\n";
        }
    }

    // --- unique_ptr + カスタムデリータ ---
    {
        auto file_deleter =  {
            if (f) {
                std::cout << "Closing file\n";
                fclose(f);
            }
        };
        std::unique_ptr<FILE, decltype(file_deleter)>
            file(fopen("/tmp/test.txt", "w"), file_deleter);
        if (file) {
            fprintf(file.get(), "Hello RAII\n");
        }
    } // ファイルは自動でクローズされる
}

int main() {
    demonstrate_smart_pointers();
    return 0;
}
```

### 9.3 Go のエスケープ解析

Go コンパイラはエスケープ解析（Escape Analysis）により、変数をスタックに配置するかヒープに配置するかを自動的に決定する。

```go
// Go エスケープ解析の例
package main

import "fmt"

// スタックに配置される（関数外に参照が漏れない）
func stackAlloc() int {
    x := 42
    return x  // 値コピー → x はスタックに留まる
}

// ヒープに配置される（ポインタが関数外に漏れる = エスケープ）
func heapAlloc() *int {
    x := 42
    return &x  // ポインタが外に漏れる → x はヒープに配置
}

// インタフェース経由でエスケープ
func interfaceEscape() {
    x := 42
    fmt.Println(x)  // Println(interface{}) → x がエスケープ
}

// エスケープ解析の確認コマンド:
// go build -gcflags="-m -m" main.go
// ./main.go:13:2: x escapes to heap:
// ./main.go:13:2:   flow: ~r0 = &x:
// ./main.go:13:2:     from &x (address-of) at ./main.go:14:9
```

### 9.4 リアルタイムシステムにおけるメモリ割り当て

リアルタイムシステムでは、メモリ割り当ての最悪実行時間（WCET: Worst-Case Execution Time）が保証されなければならない。

| 要件 | 汎用システム | リアルタイムシステム |
|:---|:---|:---|
| 割り当て時間 | 平均的に高速であればよい | 最悪時間が保証されること |
| GC ポーズ | 許容される | 許容されない |
| メモリ断片化 | 徐々に増加しても再起動で対応 | 長時間運用で断片化ゼロが必要 |
| 使用アロケータ | malloc, jemalloc 等 | メモリプール, TLSF |
| 障害時の挙動 | OOM Killer 等 | フェイルセーフ動作が必須 |

**TLSF（Two-Level Segregated Fit）:**

リアルタイムシステム向けに設計されたアロケータで、割り当て・解放ともに O(1) が保証される。ビットマップとセグメントリストの2レベル索引により、フリーリスト探索を定数時間で実行する。航空宇宙、自動車、医療機器などのミッションクリティカルシステムで採用されている。

---

## 10. アンチパターンと設計原則

### 10.1 アンチパターン1: 「malloc して忘れる」パターン

**問題:** 関数内で動的にメモリを確保し、エラーパス等で解放を忘れる。

```c
/* アンチパターン: 複数リソースの獲得と不完全な解放 */
int process_data_BAD(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) return -1;

    char *buffer = (char *)malloc(4096);
    if (!buffer) {
        /* file のクローズを忘れている! */
        return -1;
    }

    int *results = (int *)malloc(sizeof(int) * 1000);
    if (!results) {
        /* buffer の free を忘れている! */
        /* file のクローズも忘れている! */
        return -1;
    }

    /* ... 処理 ... */

    free(results);
    free(buffer);
    fclose(file);
    return 0;
}

/* 修正: goto による統一的なクリーンアップ（Linux カーネルスタイル） */
int process_data_GOOD(const char *filename) {
    int ret = -1;
    FILE *file = NULL;
    char *buffer = NULL;
    int *results = NULL;

    file = fopen(filename, "r");
    if (!file) goto cleanup;

    buffer = (char *)malloc(4096);
    if (!buffer) goto cleanup;

    results = (int *)malloc(sizeof(int) * 1000);
    if (!results) goto cleanup;

    /* ... 処理 ... */
    ret = 0;

cleanup:
    free(results);   /* free(NULL) は安全 */
    free(buffer);    /* free(NULL) は安全 */
    if (file) fclose(file);
    return ret;
}
```

**教訓:**
- C 言語では `goto cleanup` パターンが推奨される（Linux カーネルのコーディング規約でも採用）
- C++ では RAII（スマートポインタ、ファイルストリーム）を使えば自動的に解決
- `free(NULL)` は C 標準で安全と定義されているため、NULL チェック不要

### 10.2 アンチパターン2: 「巨大な一時バッファを毎回 malloc」パターン

**問題:** ループ内で毎回大きなバッファを malloc/free し、不要なオーバーヘッドを発生させる。

```c
/* アンチパターン: ループ内での繰り返し割り当て */
void process_items_BAD(int *items, int count) {
    for (int i = 0; i < count; i++) {
        /* 毎回 4KB の一時バッファを確保・解放 */
        char *tmp = (char *)malloc(4096);
        snprintf(tmp, 4096, "Processing item %d", items[i]);
        /* ... tmp を使った処理 ... */
        free(tmp);
        /* → count が 100万回なら、malloc/free が 100万回呼ばれる */
    }
}

/* 修正1: ループ外で一度だけ確保 */
void process_items_GOOD1(int *items, int count) {
    char *tmp = (char *)malloc(4096);
    if (!tmp) return;

    for (int i = 0; i < count; i++) {
        snprintf(tmp, 4096, "Processing item %d", items[i]);
        /* ... tmp を使った処理 ... */
    }
    free(tmp);
}

/* 修正2: サイズが固定なら VLA またはスタックを使用 */
void process_items_GOOD2(int *items, int count) {
    char tmp[4096];  /* スタック上に確保（固定サイズ） */

    for (int i = 0; i < count; i++) {
        snprintf(tmp, sizeof(tmp), "Processing item %d", items[i]);
        /* ... tmp を使った処理 ... */
    }
    /* 自動的に解放される */
}
```

**教訓:**
- ループ内の malloc/free は大きなオーバーヘッド（システムコール、ロック競合、断片化）
- 可能ならバッファをループ外で一度だけ確保する
- 固定サイズの小バッファ（~数 KB）はスタックに置く方が高速
- 大きなバッファが必要な場合はメモリプールを検討する

### 10.3 設計原則まとめ

| 原則 | 説明 | 適用場面 |
|:---|:---|:---|
| RAII | リソース獲得をオブジェクト初期化に結びつける | C++, Rust |
| 所有権の明確化 | メモリの所有者を1つに限定 | 全言語 |
| バッファ再利用 | 頻繁な割り当てを避けてバッファを使い回す | ホットパス |
| メモリプール | 同一サイズの割り当てを専用プールで管理 | ゲーム, サーバ |
| アリーナアロケータ | フェーズ単位でまとめて確保・一括解放 | コンパイラ, パーサ |
| コピーオンライト | 実際に変更が発生するまでメモリ共有 | fork, 文字列 |
| ゼロコピー | データのコピーを避けて参照を渡す | ネットワーク, I/O |

---

## 11. 実践演習（3段階）

### 演習1: [基礎] メモリレイアウトの確認

**課題:** 以下の C プログラムの各変数がどのメモリセグメントに配置されるか答えよ。

```c
#include <stdlib.h>
#include <string.h>

int global_initialized = 42;        /* → ? */
int global_uninitialized;           /* → ? */
const char *literal = "hello";      /* → ? (ポインタ自体は?) */

void function_example(int param) {  /* param → ? */
    int local = 10;                 /* → ? */
    static int persistent = 5;     /* → ? */
    int *dynamic = malloc(100);    /* dynamic → ?, *dynamic → ? */
    char buf[256];                 /* → ? */

    free(dynamic);
}
```

<details>
<summary>解答を表示</summary>

| 変数 | セグメント | 理由 |
|:---|:---|:---|
| `global_initialized` | データ | 初期化済みグローバル変数 |
| `global_uninitialized` | BSS | 未初期化グローバル変数（ゼロ初期化） |
| `literal` (ポインタ自体) | データ | 初期化済みグローバルポインタ |
| `"hello"` (文字列本体) | テキスト(rodata) | 文字列リテラル（読み取り専用） |
| `param` | スタック | 関数引数 |
| `local` | スタック | ローカル変数 |
| `persistent` | データ | static 変数（初期化済み） |
| `dynamic` (ポインタ自体) | スタック | ローカル変数 |
| `*dynamic` (指す先) | ヒープ | malloc で動的確保 |
| `buf[256]` | スタック | ローカル配列 |

</details>

### 演習2: [応用] メモリリークの検出と修正

**課題:** 以下のコードにはメモリリークが3箇所ある。全て特定し修正せよ。

```c
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    char *name;
    int *scores;
    int score_count;
} Student;

Student *create_student(const char *name, int count) {
    Student *s = (Student *)malloc(sizeof(Student));
    s->name = strdup(name);
    s->scores = (int *)malloc(sizeof(int) * count);
    s->score_count = count;
    return s;
}

void process_students(void) {
    Student *alice = create_student("Alice", 5);
    Student *bob = create_student("Bob", 3);

    /* Alice のスコアを設定 */
    for (int i = 0; i < alice->score_count; i++) {
        alice->scores[i] = 80 + i;
    }

    /* エラーが発生した場合の早期リターン */
    if (alice->scores[0] < 50) {
        free(alice);  /* リーク! name と scores を free していない */
        return;       /* bob もリーク! */
    }

    /* Bob の名前を変更 */
    bob->name = strdup("Robert");  /* リーク! 元の "Bob" が漏れる */

    free(alice->scores);
    free(alice->name);
    free(alice);
    free(bob->scores);
    free(bob->name);
    free(bob);
}
```

<details>
<summary>解答を表示</summary>

**リーク箇所:**

1. **早期リターン時に `alice->name` と `alice->scores` を free していない**
2. **早期リターン時に `bob` 全体（name, scores, 本体）を free していない**
3. **`bob->name = strdup("Robert")` で元の `"Bob"` の領域が漏れる**

**修正版:**

```c
void destroy_student(Student *s) {
    if (s) {
        free(s->scores);
        free(s->name);
        free(s);
    }
}

void process_students_fixed(void) {
    Student *alice = create_student("Alice", 5);
    Student *bob = create_student("Bob", 3);

    for (int i = 0; i < alice->score_count; i++) {
        alice->scores[i] = 80 + i;
    }

    if (alice->scores[0] < 50) {
        destroy_student(alice);  /* 修正1: 完全に解放 */
        destroy_student(bob);    /* 修正2: bob も解放 */
        return;
    }

    /* 修正3: 元の name を先に free してから差し替え */
    free(bob->name);
    bob->name = strdup("Robert");

    destroy_student(alice);
    destroy_student(bob);
}
```

</details>

### 演習3: [発展] カスタムアロケータの設計

**課題:** 以下の要件を満たすアリーナアロケータを設計・実装せよ。

**要件:**
1. 初期化時に大きなメモリブロック（アリーナ）を一括確保する
2. `arena_alloc(size)` でアリーナからメモリを「バンプ」割り当てする（ポインタを進めるだけ）
3. 個別の `free` は不要。`arena_reset()` で全体を一括リセットする
4. `arena_destroy()` でアリーナ全体を解放する
5. アラインメント（8バイト境界）を保証する

**ヒント:** このパターンはコンパイラの AST 構築やリクエスト単位のサーバ処理で広く使用される。

<details>
<summary>解答例を表示</summary>

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define ARENA_ALIGN 8
#define ALIGN_UP(x, a) (((x) + (a) - 1) & ~((a) - 1))

typedef struct arena {
    uint8_t *base;      /* アリーナの先頭 */
    size_t size;        /* アリーナの総サイズ */
    size_t offset;      /* 次の割り当て位置 */
} arena_t;

/* アリーナ初期化 */
arena_t *arena_create(size_t size) {
    arena_t *arena = (arena_t *)malloc(sizeof(arena_t));
    if (!arena) return NULL;

    arena->base = (uint8_t *)malloc(size);
    if (!arena->base) {
        free(arena);
        return NULL;
    }
    arena->size = size;
    arena->offset = 0;
    return arena;
}

/* バンプ割り当て O(1) */
void *arena_alloc(arena_t *arena, size_t size) {
    size_t aligned_offset = ALIGN_UP(arena->offset, ARENA_ALIGN);
    if (aligned_offset + size > arena->size) {
        return NULL;  /* アリーナ枯渇 */
    }
    void *ptr = arena->base + aligned_offset;
    arena->offset = aligned_offset + size;
    return ptr;
}

/* 全体リセット O(1) — 個別 free は不要 */
void arena_reset(arena_t *arena) {
    arena->offset = 0;
}

/* アリーナ破棄 */
void arena_destroy(arena_t *arena) {
    if (arena) {
        free(arena->base);
        free(arena);
    }
}

/* 使用例: コンパイラの AST 構築 */
typedef struct ast_node {
    int type;
    struct ast_node *left;
    struct ast_node *right;
    int value;
} ast_node_t;

int main(void) {
    arena_t *arena = arena_create(1024 * 1024); /* 1MB アリーナ */

    /* AST ノードをアリーナから高速割り当て */
    ast_node_t *root = (ast_node_t *)arena_alloc(arena, sizeof(ast_node_t));
    root->type = 1;
    root->value = 42;
    root->left = (ast_node_t *)arena_alloc(arena, sizeof(ast_node_t));
    root->right = (ast_node_t *)arena_alloc(arena, sizeof(ast_node_t));

    printf("使用量: %zu / %zu bytes (%.1f%%)\n",
           arena->offset, arena->size,
           (double)arena->offset / arena->size * 100);

    /* リクエスト処理完了 → 一括リセット（個別 free 不要!） */
    arena_reset(arena);
    printf("リセット後: %zu / %zu bytes\n", arena->offset, arena->size);

    arena_destroy(arena);
    return 0;
}
```

**アリーナアロケータの利点:**
- 割り当て: ポインタの加算のみ → O(1)、ロック不要
- 解放: 一括リセットのみ → O(1)、個別 free 不要
- 断片化: 発生しない（連続領域をシーケンシャルに使用）
- キャッシュ効率: 極めて高い（連続メモリアクセス）

</details>

---

## 12. FAQ

### Q1: malloc はスレッドセーフか?

glibc の ptmalloc2 はスレッドセーフである。内部でミューテックスを使用してアリーナ（メモリプール）を保護し、スレッドごとに異なるアリーナを割り当てることでロック競合を減らしている。ただし、高スレッド数環境ではアリーナの競合がボトルネックになることがある。jemalloc や tcmalloc はスレッドローカルキャッシュにより、大部分の割り当て操作をロック不要で実行でき、高い並行性を実現する。性能が問題になる場合は `LD_PRELOAD` でアロケータを差し替えることが可能である。

```bash
# jemalloc に差し替えて実行
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so ./my_program

# tcmalloc に差し替えて実行
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so ./my_program
```

### Q2: Rust の所有権でメモリリークは完全に防げるか?

いいえ。Rust でもメモリリークは発生し得る。`Rc<T>` の循環参照、`std::mem::forget` による意図的なドロップ抑制、`Box::leak` による明示的なリーク、終了しないスレッドが保持するメモリなど、リークが起こるパターンは複数ある。ただし、Rust が保証するのは **メモリ安全性**（Use-After-Free、ダブルフリー、データ競合の防止）であり、メモリリークは安全性の問題ではないと位置づけられている。循環参照は `Weak<T>` を使って対策する。

### Q3: GC のある言語で明示的にメモリを解放する方法はあるか?

多くの GC 言語では、GC の実行をヒントとして要求することはできるが、即座の解放を保証することはできない。Java の `System.gc()` は「GC を実行してほしい」というヒントに過ぎず、JVM は無視する場合がある。Go の `runtime.GC()` は同期的に GC を実行するが、日常的に呼ぶべきではない。Python の `gc.collect()` は循環参照を回収するが、参照カウントが 0 のオブジェクトは `del` 時点で即座に回収される。実践的には、大きなオブジェクトへの参照を `null` / `None` / `nil` にセットし、GC が回収可能な状態にすることが推奨される。

### Q4: mmap と brk/sbrk の使い分けの基準は何か?

glibc の malloc はデフォルトで `MMAP_THRESHOLD`（128KB）を閾値として使い分ける。128KB 以下の割り当ては `sbrk` でヒープを拡張し、128KB を超える割り当ては `mmap(MAP_ANONYMOUS)` で独立した仮想メモリ領域を確保する。`mmap` の利点は、`munmap` で即座にカーネルへメモリを返却できることである。一方 `sbrk` で拡張されたヒープは、末尾の空きブロックからしか縮小できないため、途中のブロックが使用中だとメモリがカーネルに返却されない。この閾値は `mallopt(M_MMAP_THRESHOLD, size)` で変更可能である。

### Q5: OOM Killer はどのプロセスを終了させるか?

Linux の OOM Killer は `/proc/[pid]/oom_score` に基づいてプロセスを選択する。スコアが高いほど終了されやすい。スコアは主にプロセスのメモリ使用量に基づくが、`oom_score_adj`（-1000〜+1000）で調整可能である。重要なプロセス（データベース等）は `oom_score_adj = -1000` に設定して OOM Killer の対象外にすることができる。ただし、過度に多くのプロセスを保護すると、OOM 発生時に適切なプロセスを終了できず、システム全体がハングする可能性がある。

```bash
# 現在のプロセスの OOM スコアを確認
cat /proc/self/oom_score

# 重要なプロセスを OOM Killer から保護
echo -1000 > /proc/$(pidof my_database)/oom_score_adj
```

---


## FAQ

### Q1: このトピックを学ぶ上で最も重要なポイントは何ですか？

実践的な経験を積むことが最も重要です。理論だけでなく、実際にコードを書いて動作を確認することで理解が深まります。

### Q2: 初心者がよく陥る間違いは何ですか？

基礎を飛ばして応用に進むことです。このガイドで説明している基本概念をしっかり理解してから、次のステップに進むことをお勧めします。

### Q3: 実務ではどのように活用されていますか？

このトピックの知識は、日常的な開発業務で頻繁に活用されます。特にコードレビューやアーキテクチャ設計の際に重要になります。

---

## 13. まとめ

### 全体概要

本章では、メモリ割り当て戦略をプロセスのメモリレイアウトから始まり、アルゴリズム、プロダクションレベルのアロケータ、カーネルの物理メモリ管理、ガベージコレクション、断片化対策、デバッグ手法、言語ランタイムごとの特徴まで包括的に解説した。

### 重要概念の総整理

| 概念 | 要点 |
|:---|:---|
| メモリレイアウト | テキスト, データ, BSS, ヒープ, mmap, スタックの6領域 |
| スタック vs ヒープ | スタック: O(1) 割り当て, 自動解放 / ヒープ: 柔軟だが管理コスト大 |
| 割り当てアルゴリズム | First Fit(高速), Best Fit(断片化少), Buddy(カーネル), Segregated(現代的) |
| ptmalloc2 | Linux 標準。アリーナ + fastbin/smallbin/largebin の階層構造 |
| jemalloc | 低断片化 + 充実した統計機能。Redis, Firefox が採用 |
| tcmalloc | スレッドキャッシュで小オブジェクト割り当てが極めて高速。Chrome が採用 |
| mimalloc | セグメントベース + フリーリストシャーディング。全体的に高性能 |
| Buddy Allocator | 2のべき乗ブロック管理。Linux カーネルの物理ページ割り当て |
| SLAB/SLUB | カーネル内小オブジェクト管理。オブジェクト型ごとにキャッシュ |
| マーク & スイープ | ルートからの到達性でゴミ判定。基本だが STW が発生 |
| 世代別 GC | 弱い世代仮説に基づく。Minor GC(高頻度) + Major GC(低頻度) |
| 参照カウント | 即時回収可能だが循環参照に弱い。Python, Swift(ARC) が採用 |
| 三色マーキング | 並行 GC の基盤。白/灰/黒でマーキング。Go が採用 |
| Rust 所有権 | GC なし。コンパイル時にメモリ安全性を保証。ゼロコスト |
| 外部断片化 | 空き合計は十分だが連続領域が不足。コンパクションで対策 |
| 内部断片化 | 割り当てブロック内の無駄。サイズクラスの細分化で軽減 |
| メモリプール | 固定サイズ割り当てを O(1) で実行。ゲーム、サーバ向け |
| アリーナアロケータ | バンプ割り当て + 一括解放。コンパイラ、リクエスト処理向け |
| RAII | リソースの獲得と解放をオブジェクトのライフタイムに結びつける |
| Demand Paging | 実際のアクセス時まで物理ページ割り当てを遅延 |

### 次のステップ

メモリ割り当ての知識をさらに深めるために、以下のトピックに進むことを推奨する:

- **仮想メモリとページング:** ページテーブル、TLB、ページ置換アルゴリズム
- **メモリマップドI/O:** mmap によるファイルアクセスの高速化
- **NUMA アーキテクチャ:** マルチソケットシステムでのメモリ配置最適化
- **永続メモリ (PMEM):** Intel Optane 等の不揮発メモリプログラミング

---

## 次に読むべきガイド


---

## 14. 参考文献

1. Silberschatz, A., Galvin, P. B., & Gagne, G. *Operating System Concepts*, 10th Edition, Chapter 9: Main Memory. Wiley, 2018. --- メモリ管理の教科書的な解説。連続割り当て、ページング、セグメンテーションを体系的にカバー。

2. Tanenbaum, A. S., & Bos, H. *Modern Operating Systems*, 4th Edition, Chapter 3: Memory Management. Pearson, 2014. --- Buddy System、SLAB アロケータ、仮想メモリの詳細な解説。

3. Evans, J. "A Scalable Concurrent malloc(3) Implementation for FreeBSD." BSDCan, 2006. --- jemalloc の設計思想と実装の詳細を解説した原論文。

4. Ghemawat, S. & Menage, P. "TCMalloc: Thread-Caching Malloc." Google Performance Tools Documentation. --- tcmalloc のアーキテクチャ、スレッドキャッシュとページヒープの設計。

5. Leijen, D., Zorn, B., & de Moura, L. "Mimalloc: Free List Sharding in Action." Microsoft Research, 2019. --- mimalloc の設計原理、フリーリストシャーディングの効果を示したベンチマーク。

6. Bonwick, J. "The Slab Allocator: An Object-Caching Kernel Memory Allocator." USENIX Summer 1994 Technical Conference. --- SLAB アロケータの原論文。オブジェクトキャッシングの概念を提唱。

7. Jones, R., Hosking, A., & Moss, E. *The Garbage Collection Handbook: The Art of Automatic Memory Management*. CRC Press, 2011. --- GC アルゴリズムの包括的な参考書。マーク&スイープ、世代別、並行GC を網羅。

8. Klabnik, S. & Nichols, C. *The Rust Programming Language*. No Starch Press, 2019. Chapter 4: Understanding Ownership. --- Rust の所有権、借用、ライフタイムの公式解説。

9. Love, R. *Linux Kernel Development*, 3rd Edition, Chapter 12: Memory Management. Addison-Wesley, 2010. --- Linux カーネルの Buddy Allocator、SLAB、vmalloc の解説。

10. Masmano, M., Ripoll, I., Crespo, A., & Real, J. "TLSF: A New Dynamic Memory Allocator for Real-Time Systems." 16th Euromicro Conference on Real-Time Systems (ECRTS), 2004. --- TLSF アロケータの原論文。O(1) 保証の割り当てアルゴリズム。

---

## 参考文献

- [MDN Web Docs](https://developer.mozilla.org/) - Web技術のリファレンス
- [Wikipedia](https://ja.wikipedia.org/) - 技術概念の概要
