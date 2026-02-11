# メモリ割り当て戦略

> メモリアロケータの選択は、アプリケーションのパフォーマンスに直接影響する。

## この章で学ぶこと

- [ ] 動的メモリ割り当ての仕組みを理解する
- [ ] 主要なアロケータの違いを知る
- [ ] メモリリークとその対策を説明できる

---

## 1. 動的メモリ割り当て

```
スタック vs ヒープ:

  スタック:
  - LIFO（後入れ先出し）
  - 超高速（スタックポインタの移動のみ）
  - サイズ固定（通常1〜8MB）
  - 関数終了時に自動解放
  - ローカル変数、関数引数

  ヒープ:
  - 任意順序の割り当て/解放
  - スタックより遅い（フリーリスト管理）
  - サイズ制限が緩い（GBオーダー）
  - 手動管理（またはGC）
  - 動的サイズのデータ

割り当て戦略:
  First Fit:  最初に見つかった十分な空き領域を使用
  Best Fit:   最も小さい十分な空き領域を使用
  Worst Fit:  最も大きい空き領域を使用
  Next Fit:   前回の位置から探索

  ┌─────────────────────────────────────┐
  │ First Fit: 高速だが断片化しやすい   │
  │ Best Fit:  断片化少ないが探索コスト │
  │ → 実際のアロケータはもっと複雑な   │
  │   ハイブリッド戦略を使用            │
  └─────────────────────────────────────┘
```

---

## 2. 主要なメモリアロケータ

```
1. ptmalloc2（glibc標準）:
   → スレッドごとにアリーナ（メモリプール）
   → 小サイズ: fastbin（LIFO、高速）
   → 中サイズ: smallbin/largebin
   → 大サイズ: mmap()で直接確保

2. jemalloc（FreeBSD, Firefox, Redis）:
   → スレッドローカルキャッシュ
   → サイズクラスに分類して管理
   → 断片化が少ない
   → メモリ使用量の統計が充実

3. tcmalloc（Google）:
   → Thread-Caching Malloc
   → スレッドローカルキャッシュ + 中央キャッシュ
   → 小オブジェクトの割り当てが超高速
   → Chrome, gRPCが使用

4. mimalloc（Microsoft）:
   → 最新のハイパフォーマンスアロケータ
   → セグメントベースの管理
   → 非常に低い断片化

比較:
┌──────────────┬──────────┬──────────┬──────────┐
│ アロケータ    │ 速度     │ 断片化   │ 採用例   │
├──────────────┼──────────┼──────────┼──────────┤
│ ptmalloc2    │ ○       │ △       │ Linux標準│
│ jemalloc     │ ◎       │ ◎       │ Redis    │
│ tcmalloc     │ ◎       │ ○       │ Chrome   │
│ mimalloc     │ ◎       │ ◎       │ 最新     │
└──────────────┴──────────┴──────────┴──────────┘
```

---

## 3. ガベージコレクション

```
手動管理 vs GC:

  手動管理（C/C++/Rust）:
  → malloc/free, new/delete
  → 最高性能だがバグの温床
  → メモリリーク、ダブルフリー、Use-After-Free

  GC（Java, Go, Python, JS）:
  → ランタイムが自動でメモリ回収
  → 安全だがGCポーズがある

GCアルゴリズム:

  マーク&スイープ:
  1. ルートから到達可能なオブジェクトをマーク
  2. マークされていないオブジェクトを回収
  → シンプルだがStop-the-Worldが発生

  世代別GC:
  オブジェクトの大半は短命（弱い世代仮説）
  → Young Gen: 頻繁にGC（Minor GC）
  → Old Gen: まれにGC（Major GC）
  → Java, .NET, V8(JS) が採用

  参照カウント:
  各オブジェクトの参照数を追跡
  → 参照数が0になったら即座に回収
  → 循環参照が回収できない（→弱参照で対策）
  → Python, Swift(ARC), Objective-C(ARC) が採用

Rustのアプローチ:
  所有権（Ownership）+ 借用（Borrowing）で
  コンパイル時にメモリ安全性を保証
  → GCなし、手動free不要
  → ゼロコスト抽象化
```

---

## 4. メモリリークとデバッグ

```
メモリリーク:
  割り当てたメモリが解放されず蓄積
  → 長時間実行でメモリ枯渇 → OOM

  一般的な原因:
  1. free/deleteの呼び忘れ（C/C++）
  2. 循環参照（Python, JS の参照カウント系）
  3. イベントリスナーの解除忘れ
  4. キャッシュの無制限増加
  5. クロージャによるキャプチャ

デバッグツール:
  C/C++:
  - Valgrind (memcheck): メモリリーク検出
  - AddressSanitizer (ASan): メモリエラー検出
  - LeakSanitizer (LSan): リーク特定

  Java:
  - JVisualVM: ヒープダンプ分析
  - Eclipse MAT: メモリ分析ツール

  Go:
  - pprof: メモリプロファイリング

  JavaScript:
  - Chrome DevTools: ヒープスナップショット
```

---

## 実践演習

### 演習1: [基礎] — メモリレイアウトの確認

```c
// 以下のCプログラムの各変数がどこに配置されるか答えよ
#include <stdlib.h>

int global_var = 42;           // → ?
int uninitialized_var;         // → ?
const char *str = "hello";     // → ?

int main() {
    int local_var = 10;        // → ?
    static int static_var = 5; // → ?
    int *heap_var = malloc(4); // → ?

    free(heap_var);
    return 0;
}
// 選択肢: テキスト, データ, BSS, スタック, ヒープ
```

### 演習2: [応用] — メモリリークの検出

```bash
# Valgrindでメモリリーク検出（Linux）
valgrind --leak-check=full ./my_program

# ASanでメモリエラー検出
gcc -fsanitize=address -g program.c -o program
./program
```

---

## FAQ

### Q1: mallocはスレッドセーフか？

glibc の ptmalloc2 はスレッドセーフ。内部でロックを使用するが、アリーナ（メモリプール）をスレッドごとに分けることで競合を減らしている。jemalloc/tcmalloc はスレッドローカルキャッシュでさらに高速。

### Q2: Rustの所有権でメモリリークは完全に防げるか？

いいえ。Rustでも `Rc` の循環参照や `mem::forget` で意図的にメモリリークを発生させることは可能。ただし、**安全でない（unsafe）コード以外**ではUse-After-FreeやダブルフリーはコンパイラがREJECTする。

---

## まとめ

| 概念 | ポイント |
|------|---------|
| スタック vs ヒープ | スタック(高速,自動) vs ヒープ(柔軟,手動) |
| アロケータ | ptmalloc(標準), jemalloc(Redis), tcmalloc(Google) |
| GC | マーク&スイープ, 世代別, 参照カウント |
| Rust | 所有権システムでGCなしメモリ安全 |

---

## 次に読むべきガイド
→ [[../03-file-systems/00-fs-basics.md]] — ファイルシステムの基礎

---

## 参考文献
1. Silberschatz, A. et al. "Operating System Concepts." 10th Ed, Ch.9, 2018.
2. Evans, J. "A Scalable Concurrent malloc Implementation for FreeBSD." BSDCan, 2006.
