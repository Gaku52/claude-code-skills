# 参照カウント vs トレーシングGC

> メモリ回収の2大戦略。それぞれの仕組み・トレードオフ・適用場面を理解し、言語の特性を深く把握する。

---

## この章で学ぶこと

- [ ] 参照カウントとトレーシングGCの動作原理を説明できる
- [ ] 各方式の長所・短所をトレードオフの観点から判断できる
- [ ] 循環参照問題の本質と解決策を理解する
- [ ] ハイブリッド方式が主流である理由を論理的に説明できる
- [ ] 用途・制約に応じて適切なメモリ管理戦略を選択できる
- [ ] 各言語の GC 実装を比較し設計思想の違いを把握する

---

## 前提知識

本ガイドを十分に活用するために、以下の知識があることが望ましい。

| 領域 | 必要度 | 内容 |
|------|--------|------|
| メモリレイアウト基礎 | 必須 | スタックとヒープの違い、ポインタの概念 |
| データ構造 | 必須 | リンクリスト、グラフの基礎 |
| プログラミング基礎 | 必須 | Python, C, Java のいずれかの経験 |
| OS の基礎 | 推奨 | 仮想メモリ、ページテーブルの概念 |
| マルチスレッド | 推奨 | アトミック操作、ロックの概念 |

---

## 第1章: メモリ管理の全体像 ── なぜ自動回収が必要か

### 1.1 手動管理の時代

C 言語に代表される手動メモリ管理では、`malloc` で確保したメモリを `free` で明示的に解放する。この方式はプログラマに完全な制御を与えるが、2つの致命的なバグを生みやすい。

```c
/* ダングリングポインタ: 解放後のメモリにアクセス */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char* create_greeting(const char* name) {
    char* buf = (char*)malloc(64);
    if (!buf) return NULL;
    snprintf(buf, 64, "Hello, %s!", name);
    return buf;
}

void dangerous_example(void) {
    char* msg = create_greeting("Alice");
    printf("%s\n", msg);   /* 正常 */
    free(msg);              /* 解放 */

    /* ダングリングポインタ: 未定義動作 */
    /* printf("%s\n", msg); */

    /* ダブルフリー: 致命的バグ */
    /* free(msg); */
}

/* メモリリーク: 解放し忘れ */
void leaky_function(void) {
    for (int i = 0; i < 1000000; i++) {
        char* buf = (char*)malloc(1024);
        /* buf を使う処理 ... */
        /* free(buf) を忘れている → 1GB のリーク */
    }
}
```

NASA のソフトウェア障害データベース（Robson, 2019）によれば、宇宙関連ソフトウェアのバグの約 15% がメモリ管理に起因する。手動管理のリスクは「正しく書ける天才プログラマ」の存在を前提としており、現実のチーム開発では自動化が不可欠となる。

### 1.2 自動メモリ管理の2大戦略

自動メモリ管理には、大きく分けて2つのアプローチがある。

```
┌──────────────────────────────────────────────────────────┐
│              自動メモリ管理の分類木                        │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  自動メモリ管理                                           │
│  ├── 参照カウント (Reference Counting)                    │
│  │   ├── 素朴な参照カウント                               │
│  │   ├── 遅延参照カウント (Deferred RC)                   │
│  │   ├── 重み付き参照カウント (Weighted RC)                │
│  │   └── ARC (Automatic Reference Counting)              │
│  │                                                       │
│  ├── トレーシングGC (Tracing GC)                          │
│  │   ├── Mark-Sweep                                      │
│  │   ├── Mark-Compact                                    │
│  │   ├── Copying GC (Semi-space)                         │
│  │   ├── 世代別GC (Generational GC)                      │
│  │   ├── 並行GC (Concurrent GC)                          │
│  │   └── リージョンベースGC                               │
│  │                                                       │
│  ├── ハイブリッド方式                                     │
│  │   ├── RC + トレーシング (Python)                       │
│  │   └── トレーシング + IDisposable (C#)                  │
│  │                                                       │
│  └── 所有権ベース (Rust)                                  │
│      ├── 所有権 + 借用                                    │
│      ├── Rc<T> (単一スレッド RC)                          │
│      └── Arc<T> (マルチスレッド RC)                       │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 1.3 歴史的経緯

参照カウントとトレーシングGCは、ほぼ同時期に独立して考案された。

| 年代 | 出来事 | 方式 |
|------|--------|------|
| 1960 | George Collins が参照カウントを提案 | RC |
| 1960 | John McCarthy が Lisp で Mark-Sweep を実装 | Tracing |
| 1963 | Marvin Minsky が循環参照問題を指摘 | RC の限界 |
| 1969 | Fenichel & Yochelson が Copying GC を提案 | Tracing |
| 1984 | Lieberman & Hewitt が世代別GCを提案 | Tracing |
| 1996 | Java 1.0 が Mark-Sweep GC を搭載 | Tracing |
| 2003 | Python 2.0 で世代別 GC を追加 | Hybrid |
| 2011 | Apple が ARC を導入（Objective-C / Swift） | RC |
| 2015 | Rust 1.0 が所有権システムを標準化 | Ownership |
| 2017 | Java の ZGC が 10ms 以下の停止時間を達成 | Tracing |
| 2020 | Go 1.15 で GC レイテンシが 500μs 以下に | Tracing |

この歴史が示すように、両方式は60年以上にわたって並行して進化してきた。どちらか一方が「正解」なのではなく、対象領域・制約によって最適な選択が変わる。

---

## 第2章: 参照カウント（Reference Counting）の詳細

### 2.1 基本原理

参照カウントの核心は単純である。「各オブジェクトが、自分を指すポインタの数を記録する」。カウントが 0 になった瞬間、そのオブジェクトは不要と判断して即座に回収する。

```
参照カウントの基本動作フロー:

  操作                    オブジェクト        カウント値
  ─────────────────────────────────────────────────────
  a = Object()            [Obj_X]             1
  b = a                   [Obj_X]             2
  c = a                   [Obj_X]             3
  del c                   [Obj_X]             2
  b = None                [Obj_X]             1
  del a                   [Obj_X]             0  → 即座に解放!

  タイムライン:
  t0  t1  t2  t3  t4  t5
   │   │   │   │   │   │
   1   2   3   2   1   0 ─→ free()
   ▲   ▲   ▲   ▼   ▼   ▼
  new ref ref unref unref unref
```

### 2.2 参照カウントの内部構造

典型的な参照カウント付きオブジェクトのメモリレイアウトを示す。

```
┌─────────────────────────────────────────┐
│   参照カウント付きオブジェクトのレイアウト   │
├─────────────────────────────────────────┤
│                                         │
│  アドレス      内容          サイズ       │
│  ──────────────────────────────────────  │
│  +0x00    [ refcount    ]   8 bytes     │
│  +0x08    [ type pointer]   8 bytes     │
│  +0x10    [ hash cache  ]   8 bytes     │
│  +0x18    [ payload ... ]   可変        │
│                                         │
│  ※ CPython (64bit) の PyObject 構造:    │
│  typedef struct {                       │
│      Py_ssize_t ob_refcnt;  // +0x00    │
│      PyTypeObject *ob_type; // +0x08    │
│      // payload follows                 │
│  } PyObject;                            │
│                                         │
└─────────────────────────────────────────┘
```

### 2.3 Python での参照カウント観察

Python は主要な GC メカニズムとして参照カウントを採用している。`sys.getrefcount()` で参照数を直接確認できる。

```python
"""
参照カウントの詳細な観察 (Python 3.12+)
"""
import sys
import gc

# --- 基本的な参照カウントの増減 ---
class TrackedObject:
    """参照カウントを追跡するクラス"""
    _count = 0

    def __init__(self, name: str):
        TrackedObject._count += 1
        self.name = name
        self.id_num = TrackedObject._count
        print(f"  [CREATE] {self.name} (id={self.id_num})")

    def __del__(self):
        print(f"  [DELETE] {self.name} (id={self.id_num})")

def demonstrate_refcount():
    """参照カウントの増減を段階的に確認"""
    print("=== 参照カウント基本動作 ===\n")

    # Step 1: オブジェクト生成
    obj = TrackedObject("Alpha")
    base_count = sys.getrefcount(obj) - 1  # getrefcount自身の参照を除く
    print(f"  参照数: {base_count} (変数 obj のみ)\n")

    # Step 2: 別名を追加
    alias1 = obj
    print(f"  参照数: {sys.getrefcount(obj) - 1} (obj + alias1)\n")

    # Step 3: リストに格納
    container = [obj, obj, obj]
    print(f"  参照数: {sys.getrefcount(obj) - 1} (obj + alias1 + list*3)\n")

    # Step 4: リストから削除
    del container
    print(f"  参照数: {sys.getrefcount(obj) - 1} (obj + alias1)\n")

    # Step 5: 全参照を削除
    del alias1
    print(f"  参照数: {sys.getrefcount(obj) - 1} (obj のみ)\n")

    del obj
    print("  ↑ refcount=0 → __del__ が即座に呼ばれる\n")

def demonstrate_weak_reference():
    """弱参照は参照カウントを増加させないことを確認"""
    import weakref

    print("=== 弱参照と参照カウント ===\n")

    obj = TrackedObject("Beta")
    print(f"  強参照の数: {sys.getrefcount(obj) - 1}")

    # 弱参照を作成
    weak = weakref.ref(obj)
    print(f"  弱参照追加後: {sys.getrefcount(obj) - 1}  ← 変化なし!")
    print(f"  弱参照経由のアクセス: {weak().name}")

    # 強参照を削除
    del obj
    print(f"  強参照削除後: weak() = {weak()}  ← None になる\n")

if __name__ == "__main__":
    demonstrate_refcount()
    demonstrate_weak_reference()
```

出力例:
```
=== 参照カウント基本動作 ===

  [CREATE] Alpha (id=1)
  参照数: 1 (変数 obj のみ)

  参照数: 2 (obj + alias1)

  参照数: 5 (obj + alias1 + list*3)

  参照数: 2 (obj + alias1)

  参照数: 1 (obj のみ)

  [DELETE] Alpha (id=1)
  ↑ refcount=0 → __del__ が即座に呼ばれる

=== 弱参照と参照カウント ===

  [CREATE] Beta (id=2)
  強参照の数: 1
  弱参照追加後: 1  ← 変化なし!
  弱参照経由のアクセス: Beta
  [DELETE] Beta (id=2)
  強参照削除後: weak() = None  ← None になる
```

### 2.4 Swift の ARC（Automatic Reference Counting）

Swift は、Apple が設計した最も洗練された参照カウント実装の一つである。コンパイラが retain/release 呼び出しを自動挿入するため、プログラマは手動でカウントを管理する必要がない。

```swift
// Swift ARC の動作を詳細に観察する例
import Foundation

// === 基本的な ARC の動作 ===
class Document {
    let title: String
    var author: Author?

    init(title: String) {
        self.title = title
        print("  Document '\(title)' を生成 (refcount=1)")
    }

    deinit {
        print("  Document '\(title)' を解放")
    }
}

class Author {
    let name: String
    // weak を使って循環参照を防止
    weak var primaryDocument: Document?

    init(name: String) {
        self.name = name
        print("  Author '\(name)' を生成")
    }

    deinit {
        print("  Author '\(name)' を解放")
    }
}

func arcDemo() {
    print("=== ARC 基本動作 ===")

    // refcount: doc=1
    var doc: Document? = Document(title: "GC Handbook")

    // refcount: author=1
    var author: Author? = Author(name: "Jones")

    // doc の author プロパティに代入 → author の refcount=2
    doc?.author = author

    // author の primaryDocument に代入 → weak なので doc の refcount は増えない
    author?.primaryDocument = doc

    print("\n  --- 参照を解放 ---")

    // author の refcount: 2 → 1 (doc.author がまだ保持)
    author = nil
    print("  author=nil 後: Author はまだ生存 (doc.author が保持)")

    // doc の refcount: 1 → 0 → 解放
    // doc 解放時に doc.author も解放 → author の refcount: 1 → 0 → 解放
    doc = nil
    print("  doc=nil 後: 連鎖的に両方解放された")
}

// === unowned の使用例 ===
class CreditCard {
    let number: String
    // unowned: nil にならないことが保証されている場合に使用
    unowned let owner: Customer

    init(number: String, owner: Customer) {
        self.number = number
        self.owner = owner
        print("  CreditCard \(number) を生成")
    }

    deinit {
        print("  CreditCard \(number) を解放")
    }
}

class Customer {
    let name: String
    var card: CreditCard?

    init(name: String) {
        self.name = name
        print("  Customer '\(name)' を生成")
    }

    deinit {
        print("  Customer '\(name)' を解放")
    }
}

func unownedDemo() {
    print("\n=== unowned の動作 ===")

    var customer: Customer? = Customer(name: "Alice")
    customer!.card = CreditCard(number: "1234-5678", owner: customer!)

    // customer を解放 → card も連鎖的に解放
    customer = nil
    // CreditCard.owner は unowned なので Customer の refcount を増やさない
    // Customer 解放 → Customer.card 解放 → CreditCard 解放
}

arcDemo()
unownedDemo()
```

### 2.5 参照カウントの最適化手法

素朴な参照カウントには性能上の課題がある。現代の実装では以下の最適化が施されている。

```
┌───────────────────────────────────────────────────────────┐
│          参照カウント最適化テクニック一覧                     │
├───────────────────────────────────────────────────────────┤
│                                                           │
│  1. 遅延参照カウント (Deferred Reference Counting)         │
│     ────────────────────────────────────────────          │
│     ローカル変数からの参照ではカウントを増減しない。          │
│     ヒープからの参照のみカウントする。                       │
│     → 参照操作の 80-90% を削減可能。                       │
│                                                           │
│     通常: a = obj  → obj.refcount++                       │
│     遅延: a = obj  → (何もしない / ローカル変数は除外)      │
│                                                           │
│  2. 重み付き参照カウント (Weighted Reference Counting)      │
│     ────────────────────────────────────────────          │
│     参照コピー時にカウントを分割（加算を回避）。             │
│     分散システムで特に有効。                                │
│                                                           │
│     初期: obj.weight = 1024                               │
│     コピー: original.weight /= 2, copy.weight = 512       │
│     → アトミックな加算操作が不要になる。                    │
│                                                           │
│  3. バッファリング (Coalesced Reference Counting)          │
│     ────────────────────────────────────────────          │
│     短命な参照の増減をバッファリングし、                     │
│     最終的な差分だけを反映する。                            │
│                                                           │
│     a = obj; b = a; c = a; del b; del c;                  │
│     → +1, +1, +1, -1, -1 の代わりに最終結果 +1 のみ適用    │
│                                                           │
│  4. Swift のサイドテーブル最適化                            │
│     ────────────────────────────────────────────          │
│     weak 参照がない場合はインライン refcount (高速)。        │
│     weak 参照が生じた時点でサイドテーブルを遅延生成。        │
│                                                           │
│     [Object Header]                                       │
│      ├── strong refcount: inline (8 bytes)                │
│      ├── unowned refcount: inline (8 bytes)               │
│      └── weak refcount: → Side Table (遅延割り当て)        │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

### 2.6 マルチスレッド環境での参照カウント

マルチスレッド環境では、参照カウントの増減がデータ競合を引き起こす可能性がある。そのためアトミック操作が必要となるが、これは大きなコストを伴う。

```rust
// Rust: Rc<T> vs Arc<T> ─ 単一スレッド vs マルチスレッド参照カウント
use std::rc::Rc;
use std::sync::Arc;
use std::thread;

fn single_thread_rc() {
    // Rc<T>: 単一スレッド専用（アトミック操作なし → 高速）
    let a = Rc::new(vec![1, 2, 3]);
    println!("参照カウント: {}", Rc::strong_count(&a));  // 1

    let b = Rc::clone(&a);  // 非アトミックなインクリメント
    println!("参照カウント: {}", Rc::strong_count(&a));  // 2

    drop(b);
    println!("参照カウント: {}", Rc::strong_count(&a));  // 1
}

fn multi_thread_arc() {
    // Arc<T>: マルチスレッド対応（アトミック操作 → やや遅い）
    let data = Arc::new(vec![1, 2, 3, 4, 5]);
    let mut handles = vec![];

    for i in 0..4 {
        let data_clone = Arc::clone(&data);  // アトミックなインクリメント
        handles.push(thread::spawn(move || {
            println!("Thread {}: sum = {}", i, data_clone.iter().sum::<i32>());
            // スコープ終了時にアトミックなデクリメント
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }
    // 全スレッド終了後、data が唯一の参照 → refcount=1
    println!("最終参照カウント: {}", Arc::strong_count(&data));
}

fn main() {
    println!("=== 単一スレッド (Rc) ===");
    single_thread_rc();

    println!("\n=== マルチスレッド (Arc) ===");
    multi_thread_arc();
}
```

アトミック操作のコストは無視できない。x86-64 アーキテクチャでの比較:

| 操作 | サイクル数（概算） | 備考 |
|------|-------------------|------|
| 通常のインクリメント (`i++`) | 1 | レジスタ操作 |
| `lock inc` (アトミック) | 10-20 | キャッシュライン排他制御 |
| `lock cmpxchg` (CAS) | 15-30 | Compare-And-Swap |
| キャッシュミス + アトミック | 100-300 | 別コアのキャッシュから転送 |

この性能差が、Rust が `Rc` と `Arc` を分離している理由であり、Swift が iOS（主に単一メインスレッドで UI 操作）で ARC を採用できる理由でもある。

---

## 第3章: トレーシングGC（Tracing GC）の詳細

### 3.1 基本原理 ── 到達可能性解析

トレーシング GC の核心は「ルートから到達可能かどうか」でオブジェクトの生死を判定することにある。参照カウントが「何人が自分を参照しているか」を個々のオブジェクトが管理するのに対し、トレーシング GC は全オブジェクトのグラフを俯瞰的に走査する。

```
トレーシングGCの到達可能性解析:

  GC Roots（スタック変数、グローバル変数、レジスタ）
    │
    ├──→ [A] ──→ [B] ──→ [C]
    │     │
    │     └──→ [D] ──→ [E]
    │
    └──→ [F]

    (孤立)  [G] ──→ [H]       ← ルートから到達不能
            [I] ←─→ [J]       ← 循環参照でも到達不能

  マークフェーズ後:
    到達可能: {A, B, C, D, E, F}  → 生存
    到達不能: {G, H, I, J}        → 回収対象

  ★ 循環参照 I↔J もルートから辿れないので正しく回収される
```

### 3.2 Mark-Sweep アルゴリズム

最も基本的なトレーシング GC アルゴリズムであり、John McCarthy が 1960 年に Lisp で実装した方式。

```
Mark-Sweep アルゴリズムの2フェーズ:

Phase 1: Mark（マーク）
─────────────────────────────────────────
  ルートから DFS/BFS で全到達可能オブジェクトにマークを付ける

  mark(root):
    if root.marked: return
    root.marked = true
    for ref in root.references:
        mark(ref)

  ヒープ: [A*] [B*] [C ] [D*] [E ] [F*] [G ] [H ]
          (* = マーク済み)

Phase 2: Sweep（スイープ）
─────────────────────────────────────────
  ヒープを線形スキャンし、マークされていないオブジェクトを解放

  sweep():
    for obj in heap:
        if obj.marked:
            obj.marked = false   // 次回GCのためにリセット
        else:
            free(obj)            // 回収

  ヒープ: [A ] [B ] [   ] [D ] [   ] [F ] [   ] [   ]
                      ↑          ↑          ↑     ↑
                    回収済み    回収済み    回収済み  回収済み

問題点: メモリの断片化が発生する（空き領域が散在）
```

### 3.3 Mark-Compact アルゴリズム

Mark-Sweep の断片化問題を解決するために、生存オブジェクトをメモリの一方に寄せる（コンパクション）方式。

```
Mark-Compact の動作:

Before GC:
  [A] [_] [B] [_] [_] [C] [_] [D] [_] [_]
   ↑       ↑             ↑       ↑
  live    live          live    live

  _ = ゴミ（回収対象）

After Mark-Compact:
  [A] [B] [C] [D] [            空き領域            ]

  利点: 連続した大きな空き領域が確保できる
       → メモリアロケーションが高速（バンプポインタ）
  欠点: オブジェクトの移動コストが大きい
       → 全参照の書き換えが必要
```

### 3.4 Copying GC（Semi-space）

ヒープを2つの半空間に分割し、生存オブジェクトを一方から他方にコピーする方式。Cheney (1970) のアルゴリズムが有名。

```
Copying GC (Semi-space):

  ヒープを From-space と To-space に二分割

  GC前:
  From-space: [A] [_] [B] [_] [C] [_] [_] [D]
  To-space:   [                                ]

  GC実行（生存オブジェクトを To-space にコピー）:
  From-space: [A] [_] [B] [_] [C] [_] [_] [D]   ← 全体を破棄
  To-space:   [A'] [B'] [C'] [D'] [            ]  ← コンパクト済み

  GC後（From/To を入れ替え）:
  From-space: [A'] [B'] [C'] [D'] [            ]  ← 新しい From
  To-space:   [                                ]   ← 新しい To

  利点:
    - コンパクション済み → アロケーションが O(1)（バンプポインタ）
    - 生存オブジェクトのみ処理 → ゴミが多いほど高速
  欠点:
    - ヒープ使用効率が 50%（常に半分が未使用）
    - 長寿命オブジェクトも毎回コピーされる
```

### 3.5 世代別GC（Generational GC）

「世代仮説」に基づくアルゴリズム。これは「ほとんどのオブジェクトは若くして死ぬ」という経験的観測に基づく。

```
世代別GCの構造:

  世代仮説 (Generational Hypothesis):
  ──────────────────────────────────
  "Most objects die young."

  オブジェクトの生存曲線:

  生存率
  100%│*
     │ *
     │  *
     │   *
     │    **
     │      ***
     │         ******
     │               ****************
   0%│─────────────────────────────── 年齢
     新生         中年         老年

  世代別ヒープレイアウト (Java HotSpot の例):
  ┌─────────────────────────────────────────────┐
  │  Young Generation (新世代)                    │
  │  ┌────────┬────────┬────────┐               │
  │  │ Eden   │ S0     │ S1     │               │
  │  │(新規)  │(From)  │(To)    │               │
  │  └────────┴────────┴────────┘               │
  │  ← Minor GC: 頻繁だが高速 (数ms)            │
  ├─────────────────────────────────────────────┤
  │  Old Generation (旧世代)                      │
  │  ┌─────────────────────────────────────┐    │
  │  │  長寿命オブジェクト                    │    │
  │  │  (Minor GC を N 回生き延びた)          │    │
  │  └─────────────────────────────────────┘    │
  │  ← Major GC: まれだが低速 (数十ms〜数百ms)  │
  └─────────────────────────────────────────────┘

  昇格 (Promotion):
    Eden → S0/S1 → ... → Old Generation
    (Minor GC を生き延びるたびに年齢 +1、閾値超えで昇格)
```

### 3.6 Java での GC 動作観察

```java
/**
 * Java GC の動作を観察するプログラム
 * 実行: java -verbose:gc -Xms64m -Xmx256m GCObservation
 */
import java.lang.ref.WeakReference;
import java.lang.ref.PhantomReference;
import java.lang.ref.ReferenceQueue;
import java.util.ArrayList;
import java.util.List;

public class GCObservation {

    // ファイナライザの実行タイミングを観察
    static class TrackedObject {
        private final String name;
        private final byte[] payload;  // メモリ消費用

        TrackedObject(String name, int sizeKB) {
            this.name = name;
            this.payload = new byte[sizeKB * 1024];
        }

        @Override
        protected void finalize() throws Throwable {
            System.out.printf("  [FINALIZE] %s (thread=%s)%n",
                name, Thread.currentThread().getName());
            super.finalize();
        }
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=== トレーシングGC観察 (Java) ===\n");

        // --- 弱参照の動作 ---
        System.out.println("1. 弱参照の動作:");
        TrackedObject strong = new TrackedObject("Strong", 1);
        WeakReference<TrackedObject> weak =
            new WeakReference<>(new TrackedObject("WeakOnly", 1));

        System.out.printf("  GC前: weak.get() = %s%n",
            weak.get() != null ? "存在" : "null");

        System.gc();  // GC を要求（保証はされない）
        Thread.sleep(100);

        System.out.printf("  GC後: weak.get() = %s%n",
            weak.get() != null ? "存在" : "null");

        // --- メモリ圧迫による GC 発生 ---
        System.out.println("\n2. メモリ圧迫によるGC:");
        List<byte[]> pressure = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            pressure.add(new byte[1024 * 1024]);  // 1MB ずつ確保
            if (i % 20 == 0) {
                System.out.printf("  確保: %dMB%n", i + 1);
            }
        }
        pressure.clear();  // 参照を切る
        System.gc();

        // --- 循環参照の回収確認 ---
        System.out.println("\n3. 循環参照の回収:");
        TrackedObject nodeA = new TrackedObject("NodeA", 1);
        TrackedObject nodeB = new TrackedObject("NodeB", 1);
        // Java ではフィールドで循環を作る（簡略化のためリフレクション省略）
        // 重要: ルートからの参照を切れば循環していても回収される
        nodeA = null;
        nodeB = null;
        System.gc();
        Thread.sleep(200);

        System.out.println("\n=== 完了 ===");
    }
}
```

### 3.7 主要な GC 実装の比較

| GC 実装 | 言語/VM | アルゴリズム | 最大停止時間 | 特徴 |
|---------|---------|-------------|-------------|------|
| G1 GC | Java (default) | リージョン + 世代別 | ~200ms | バランス型、大ヒープ向け |
| ZGC | Java 15+ | カラーポインタ + 並行 | <1ms (目標) | 超低レイテンシ |
| Shenandoah | Java (RedHat) | ブルックスポインタ + 並行 | <10ms | ZGC の対抗馬 |
| Go GC | Go | 並行 Mark-Sweep | <500us | シンプルさ重視、世代別なし |
| V8 GC | JavaScript | 世代別 + 並行 + インクリメンタル | 数ms | Orinoco プロジェクト |
| .NET GC | C# | 世代別 + コンパクション | ~10ms (Server) | Workstation/Server モード |
| Boehm GC | C/C++ | 保守的 Mark-Sweep | 可変 | 型情報なしで動作 |

---

## 第4章: 循環参照 ── 参照カウント最大の敵

### 4.1 循環参照はなぜ発生するか

循環参照は、データモデリングにおいて自然に発生する。親子関係、双方向リンクリスト、オブザーバーパターンなど、多くの設計パターンが本質的に循環構造を含む。

```
循環参照が自然発生するデータ構造:

1. 親子関係（DOM ツリー）
   Parent ──→ Children[]
      ↑             │
      └─────────────┘ (child.parentNode)

2. 双方向リンクリスト
   [A] ←→ [B] ←→ [C] ←→ [D]

3. オブザーバーパターン
   Subject ──→ Observer[]
      ↑              │
      └──────────────┘ (observer.subject)

4. グラフ構造（SNS の友人関係）
   [User1] ←→ [User2]
      ↑  ↘        ↗  ↑
      │   [User3]    │
      └──────────────┘

5. キャッシュ + コールバック
   Cache ──→ Entry ──→ Callback ──→ Cache
```

### 4.2 Python における循環参照の観察と対策

```python
"""
循環参照の生成、検出、回収を詳細に観察する
"""
import gc
import sys
import weakref

class Node:
    """循環参照を作りやすいノードクラス"""
    _instances = 0

    def __init__(self, name: str):
        Node._instances += 1
        self.name = name
        self.neighbors: list = []
        print(f"  [+] Node '{name}' 生成 (総数: {Node._instances})")

    def __del__(self):
        Node._instances -= 1
        print(f"  [-] Node '{name}' 解放 (総数: {Node._instances})")

    def __repr__(self):
        return f"Node({self.name})"

def demo_circular_reference():
    """循環参照の生成と GC による回収"""
    print("=== 循環参照デモ ===\n")

    # GC を一時的に無効化して参照カウントのみの動作を観察
    gc.disable()
    print("1. GC無効状態で循環参照を作成:")

    a = Node("A")
    b = Node("B")
    a.neighbors.append(b)   # A → B
    b.neighbors.append(a)   # B → A  (循環!)
    print(f"   A の参照数: {sys.getrefcount(a) - 1}")
    print(f"   B の参照数: {sys.getrefcount(b) - 1}")

    # ローカル変数の参照を削除
    del a, b
    print("\n   del a, b 後: __del__ が呼ばれない! (循環参照リーク)")
    print(f"   未回収の Node 数: {Node._instances}")

    # GC を有効化して循環参照を回収
    print("\n2. GC を有効化して回収:")
    gc.enable()
    collected = gc.collect()
    print(f"   回収されたオブジェクト数: {collected}")
    print(f"   残りの Node 数: {Node._instances}")

def demo_weak_reference_solution():
    """弱参照による循環参照回避"""
    print("\n=== 弱参照による解決 ===\n")

    class SafeNode:
        def __init__(self, name: str):
            self.name = name
            self._neighbors_strong: list = []
            self._neighbors_weak: list = []

        def add_strong_ref(self, other):
            """強参照: 親 → 子"""
            self._neighbors_strong.append(other)

        def add_weak_ref(self, other):
            """弱参照: 子 → 親"""
            self._neighbors_weak.append(weakref.ref(other))

        def __del__(self):
            print(f"  [-] SafeNode '{self.name}' 解放")

    gc.disable()  # GC 無効でも回収されることを確認

    parent = SafeNode("Parent")
    child = SafeNode("Child")

    parent.add_strong_ref(child)   # 親 → 子: 強参照
    child.add_weak_ref(parent)     # 子 → 親: 弱参照

    del parent, child
    print("  del 後: 参照カウントだけで正しく回収された!")

    gc.enable()

if __name__ == "__main__":
    demo_circular_reference()
    demo_weak_reference_solution()
```

### 4.3 各言語の循環参照対策

| 言語 | 主方式 | 循環参照対策 | プログラマの責務 |
|------|--------|-------------|----------------|
| Python | RC + GC | 世代別GC が循環を検出 | `gc.collect()` を理解する |
| Swift | ARC | `weak` / `unowned` キーワード | 明示的に弱参照を指定 |
| Rust | 所有権 | `Weak<T>` 型 | 循環が必要な場合に `Weak` を使用 |
| Java | Tracing GC | GC が自動処理 | 特になし（GC に任せる） |
| JavaScript | Tracing GC | GC が自動処理 | 特になし |
| Objective-C | ARC | `__weak` / `__unsafe_unretained` | 明示的に弱参照を指定 |
| PHP | RC + GC | 循環検出 GC を搭載 | 特になし（PHP 5.3+） |
| Perl | RC | 手動 + `Scalar::Util::weaken` | 明示的に弱参照化 |

### 4.4 循環参照検出アルゴリズム

Python の世代別 GC が使う循環参照検出アルゴリズム（試行的削除）の概要を示す。

```
Python の循環参照検出 (Trial Deletion):

Step 1: 対象コンテナオブジェクトの仮の参照カウントを計算
        gc_refs = ob_refcnt  (実際の参照カウントをコピー)

Step 2: 内部参照を仮削除
        各オブジェクトの参照先について gc_refs を -1

        例: A(gc_refs=2) ──→ B(gc_refs=2)
                ↑                │
                └────────────────┘

        内部参照を引く:
        A: gc_refs = 2 - 1(Bからの参照) = 1
        B: gc_refs = 2 - 1(Aからの参照) = 1

        もし外部参照がない場合:
        A: gc_refs = 1 - 1 = 0
        B: gc_refs = 1 - 1 = 0

Step 3: gc_refs == 0 のオブジェクトは
        外部からの参照がない → 循環参照のみ → 回収可能

Step 4: gc_refs > 0 のオブジェクトから到達可能なものは
        外部から参照されている → 生存

  この方式により、参照カウントの弱点である循環参照を
  トレーシングなしで検出できる（ただしコンテナ型のみ）。
```

---

## 第5章: 性能比較 ── 定量的分析

### 5.1 スループット vs レイテンシのトレードオフ

メモリ管理方式の性能は、大きく2つの軸で評価される。

```
スループット vs レイテンシのトレードオフ図:

  スループット（単位時間あたりの処理量）
  高い │
       │     * トレーシングGC (バッチ処理で効率的)
       │
       │           * ハイブリッド
       │
       │  * 参照カウント           * 所有権 (Rust)
       │   (毎操作のオーバーヘッド)    (コンパイル時解決)
  低い │
       └──────────────────────────────────
       短い                          長い
             最大停止時間（レイテンシ）

  ※ 所有権ベースは実行時コストが最小だが
     コンパイル時間と開発者の学習コストが高い
```

### 5.2 メモリ使用量の比較

各方式のメモリオーバーヘッドを定量的に比較する。

```
オブジェクトあたりのメモリオーバーヘッド:

┌───────────────────┬────────────┬──────────────────────────────┐
│ 方式              │ オーバーヘッド│ 内訳                          │
├───────────────────┼────────────┼──────────────────────────────┤
│ 参照カウント       │ 8-16 bytes │ refcount (8B)                │
│ (CPython)         │            │ + type ptr (8B)              │
├───────────────────┼────────────┼──────────────────────────────┤
│ Swift ARC         │ 16 bytes   │ strong RC (8B)               │
│                   │            │ + unowned RC (8B)            │
│                   │            │ + side table (遅延, +16B)    │
├───────────────────┼────────────┼──────────────────────────────┤
│ Mark-Sweep        │ 1 bit      │ マークビットのみ              │
│ (最小)            │            │ (ビットマップ管理の場合 0)     │
├───────────────────┼────────────┼──────────────────────────────┤
│ Copying GC        │ ヒープ×2   │ From/To 空間で常に半分未使用   │
│                   │            │ オブジェクト自体の追加は 0     │
├───────────────────┼────────────┼──────────────────────────────┤
│ 世代別GC          │ 可変       │ Remembered Set + カードテーブル│
│ (Java G1)        │            │ ヒープの約 1-5% 程度          │
├───────────────────┼────────────┼──────────────────────────────┤
│ 所有権 (Rust)     │ 0 bytes    │ 実行時オーバーヘッドなし       │
│                   │            │ (Rc使用時は 8-16B)           │
└───────────────────┴────────────┴──────────────────────────────┘
```

### 5.3 ユースケース別の性能特性

```
┌──────────────────────────┬─────────┬─────────┬─────────┐
│ ユースケース              │ RC      │ Tracing │ 所有権   │
├──────────────────────────┼─────────┼─────────┼─────────┤
│ 小オブジェクト大量生成     │ ×遅い   │ ◎高速   │ ◎高速   │
│ (一時変数、文字列連結)     │ RC操作  │ まとめて │ スタック │
├──────────────────────────┼─────────┼─────────┼─────────┤
│ 長寿命オブジェクト中心     │ ◎安定   │ △昇格   │ ○良好   │
│ (キャッシュ、設定)        │ カウント │ コスト大 │         │
├──────────────────────────┼─────────┼─────────┼─────────┤
│ リアルタイム処理           │ ○予測可 │ ×STW   │ ◎最良   │
│ (ゲーム、音声)            │ 能      │ 発生    │         │
├──────────────────────────┼─────────┼─────────┼─────────┤
│ 大量並行処理              │ ×ボトル │ ◎効率的 │ ○良好   │
│ (Web サーバー)            │ ネック  │         │         │
├──────────────────────────┼─────────┼─────────┼─────────┤
│ グラフ構造操作             │ ×循環  │ ◎自然   │ △複雑   │
│ (SNS、推薦エンジン)       │ 問題   │         │ Rc必要  │
├──────────────────────────┼─────────┼─────────┼─────────┤
│ 組み込み / IoT            │ △予測  │ ×不向き │ ◎最適   │
│ (メモリ 64KB 未満)        │ 可能   │         │         │
└──────────────────────────┴─────────┴─────────┴─────────┘

凡例: ◎=最適  ○=良好  △=注意が必要  ×=不向き
```

### 5.4 GC 停止時間の推移（Java の進化）

Java の GC は長年にわたりレイテンシ改善を追求してきた。

| GC 実装 | 登場時期 | 最大停止時間（目安） | ヒープサイズ対応 |
|---------|---------|--------------------|----|
| Serial GC | Java 1.0 | 数秒 | ~数百MB |
| Parallel GC | Java 1.4 | 数百ms | ~数GB |
| CMS | Java 1.5 | ~100ms | ~数GB |
| G1 GC | Java 7 | ~200ms (目標指定可) | ~数十GB |
| Shenandoah | Java 12 | <10ms | ~数TB |
| ZGC | Java 15 | <1ms (目標) | ~16TB |

この進化は、「トレーシング GC はレイテンシが悪い」という批判に対する直接的な回答であり、並行処理と低レイテンシの両立が技術的に可能であることを示している。

---

## 第6章: ハイブリッド方式 ── 現実の選択

### 6.1 Python のハイブリッド戦略

Python は参照カウントとトレーシング GC を組み合わせた最も有名なハイブリッド実装である。

```
Python のメモリ管理アーキテクチャ:

  ┌─────────────────────────────────────────┐
  │        Python メモリ管理スタック          │
  ├─────────────────────────────────────────┤
  │                                         │
  │  Layer 3: オブジェクト固有アロケータ      │
  │  ┌─────────────────────────────────┐    │
  │  │ int, float, list, dict, ...     │    │
  │  │ (フリーリスト + オブジェクトプール) │    │
  │  └─────────────────────────────────┘    │
  │                                         │
  │  Layer 2: Python オブジェクトアロケータ   │
  │  ┌─────────────────────────────────┐    │
  │  │ pymalloc (512 bytes 以下)       │    │
  │  │ アリーナ → プール → ブロック      │    │
  │  └─────────────────────────────────┘    │
  │                                         │
  │  Layer 1: Python メモリアロケータ        │
  │  ┌─────────────────────────────────┐    │
  │  │ malloc / free ラッパー           │    │
  │  └─────────────────────────────────┘    │
  │                                         │
  │  Layer 0: OS メモリ管理                  │
  │  ┌─────────────────────────────────┐    │
  │  │ brk, mmap, VirtualAlloc, ...    │    │
  │  └─────────────────────────────────┘    │
  │                                         │
  ├─────────────────────────────────────────┤
  │  GC サブシステム（循環参照専用）          │
  │  ┌─────────────────────────────────┐    │
  │  │ 世代0: 新オブジェクト(閾値700)   │    │
  │  │ 世代1: 中間(閾値10)             │    │
  │  │ 世代2: 長寿命(閾値10)           │    │
  │  │                                 │    │
  │  │ 回収対象: コンテナ型のみ          │    │
  │  │ (list, dict, set, class, ...)   │    │
  │  │ 非対象: int, float, str, ...    │    │
  │  └─────────────────────────────────┘    │
  └─────────────────────────────────────────┘

  動作フロー:
  1. オブジェクト生成 → 参照カウント = 1
  2. 参照コピー → カウント++
  3. 参照消滅 → カウント--
  4. カウント == 0 → 即座に __del__ + 解放 (大多数はここで回収)
  5. 循環参照で残ったもの → 世代別GC が定期的に回収
```

### 6.2 Python GC の制御と監視

```python
"""
Python GC の制御と監視 ── 実践的な使い方
"""
import gc
import sys

def inspect_gc_configuration():
    """GC の設定を確認"""
    print("=== Python GC 設定 ===\n")

    # 世代別 GC の閾値
    thresholds = gc.get_threshold()
    print(f"  世代別GC閾値: gen0={thresholds[0]}, "
          f"gen1={thresholds[1]}, gen2={thresholds[2]}")
    print(f"  意味: 世代0は {thresholds[0]} 回のアロケーション後にGC実行")
    print(f"        世代1は 世代0が {thresholds[1]} 回実行後にGC実行")
    print(f"        世代2は 世代1が {thresholds[2]} 回実行後にGC実行\n")

    # 現在の統計
    stats = gc.get_stats()
    for i, stat in enumerate(stats):
        print(f"  世代{i}: collections={stat['collections']}, "
              f"collected={stat['collected']}, "
              f"uncollectable={stat['uncollectable']}")

def demonstrate_gc_tuning():
    """性能チューニングの例"""
    print("\n=== GC チューニング ===\n")

    # 大量オブジェクト生成時は GC を一時停止
    print("  1. バッチ処理中の GC 一時停止:")
    gc.disable()
    objects = []
    for i in range(100000):
        objects.append({"index": i, "data": f"item_{i}"})
    gc.enable()
    gc.collect()
    print(f"     {len(objects)} オブジェクト生成完了\n")

    # 閾値調整
    print("  2. GC 閾値のカスタマイズ:")
    original = gc.get_threshold()
    print(f"     デフォルト: {original}")

    # レイテンシ重視: 頻繁に少量回収
    gc.set_threshold(100, 5, 5)
    print(f"     レイテンシ重視: {gc.get_threshold()}")

    # スループット重視: まれに大量回収
    gc.set_threshold(50000, 20, 20)
    print(f"     スループット重視: {gc.get_threshold()}")

    # デフォルトに戻す
    gc.set_threshold(*original)
    print(f"     復元: {gc.get_threshold()}")

def demonstrate_gc_callbacks():
    """GC コールバックによる監視"""
    print("\n=== GC コールバック監視 ===\n")

    def gc_callback(phase, info):
        if phase == "start":
            print(f"  [GC START] 世代{info['generation']}")
        elif phase == "stop":
            print(f"  [GC STOP]  回収={info['collected']}, "
                  f"回収不能={info['uncollectable']}")

    gc.callbacks.append(gc_callback)

    # 循環参照を作って GC を発動させる
    for _ in range(5):
        a, b = [], []
        a.append(b)
        b.append(a)
        del a, b

    gc.collect()

    gc.callbacks.remove(gc_callback)

if __name__ == "__main__":
    inspect_gc_configuration()
    demonstrate_gc_tuning()
    demonstrate_gc_callbacks()
```

### 6.3 .NET (C#) のハイブリッド戦略

C# はトレーシング GC を主軸としつつ、`IDisposable` パターンで決定的な解放を提供する。

```csharp
// C# のメモリ管理: トレーシングGC + IDisposable

using System;
using System.Buffers;
using System.IO;

// === IDisposable パターン ===
// GC による非決定的回収 + using による決定的解放の組み合わせ

public class ManagedResource : IDisposable
{
    private FileStream? _stream;
    private byte[]? _buffer;
    private bool _disposed = false;

    public ManagedResource(string path)
    {
        _stream = new FileStream(path, FileMode.OpenOrCreate);
        _buffer = ArrayPool<byte>.Shared.Rent(4096);
        Console.WriteLine("  リソース確保: ファイル + バッファ");
    }

    // 決定的解放: using ブロック終了時に呼ばれる
    public void Dispose()
    {
        Dispose(disposing: true);
        GC.SuppressFinalize(this);  // ファイナライザ不要を通知
    }

    protected virtual void Dispose(bool disposing)
    {
        if (_disposed) return;

        if (disposing)
        {
            // マネージドリソースの解放
            _stream?.Dispose();
            if (_buffer != null)
            {
                ArrayPool<byte>.Shared.Return(_buffer);
                _buffer = null;
            }
            Console.WriteLine("  Dispose: マネージドリソース解放");
        }

        // アンマネージドリソースの解放（もしあれば）
        _disposed = true;
    }

    // 安全ネット: Dispose 忘れ時に GC が呼ぶ
    ~ManagedResource()
    {
        Console.WriteLine("  Finalizer: Dispose 忘れを検出!");
        Dispose(disposing: false);
    }
}

// === 使用例 ===
class Program
{
    static void Main()
    {
        Console.WriteLine("=== 決定的解放 (using) ===");
        // using ブロック: スコープ終了時に自動で Dispose()
        using (var resource = new ManagedResource("test.dat"))
        {
            // resource を使用...
        }  // ← ここで Dispose() が呼ばれる (GC を待たない)

        Console.WriteLine("\n=== C# 8.0 using 宣言 ===");
        // using 宣言: 変数のスコープ終了時に Dispose()
        using var resource2 = new ManagedResource("test2.dat");
        // resource2 を使用...
        // メソッド終了時に Dispose()

        Console.WriteLine("\n=== Span<T> によるゼロアロケーション ===");
        // GC を完全に回避するアプローチ
        Span<int> stackArray = stackalloc int[100];
        for (int i = 0; i < 100; i++)
            stackArray[i] = i * i;
        Console.WriteLine($"  スタック上の配列: [{stackArray[0]}, {stackArray[1]}, ...]");
    }
}
```

### 6.4 Rust の所有権 + 必要時RC

Rust は所有権システムをデフォルトとし、循環構造が必要な場合にのみ `Rc<T>` / `Arc<T>` と `Weak<T>` を使う。

```rust
// Rust: 所有権 + 参照カウントのハイブリッド
use std::cell::RefCell;
use std::rc::{Rc, Weak};

// === 木構造: 所有権で十分 ===
#[derive(Debug)]
struct TreeNode {
    value: i32,
    children: Vec<TreeNode>,  // 親が子を所有
}

impl TreeNode {
    fn new(value: i32) -> Self {
        TreeNode { value, children: vec![] }
    }

    fn add_child(&mut self, child: TreeNode) {
        self.children.push(child);
    }
}

// === グラフ構造: Rc + Weak が必要 ===
#[derive(Debug)]
struct GraphNode {
    name: String,
    // 子への強参照
    children: RefCell<Vec<Rc<GraphNode>>>,
    // 親への弱参照（循環回避）
    parent: RefCell<Weak<GraphNode>>,
}

impl GraphNode {
    fn new(name: &str) -> Rc<Self> {
        Rc::new(GraphNode {
            name: name.to_string(),
            children: RefCell::new(vec![]),
            parent: RefCell::new(Weak::new()),
        })
    }

    fn add_child(parent: &Rc<GraphNode>, child: &Rc<GraphNode>) {
        // 親 → 子: 強参照
        parent.children.borrow_mut().push(Rc::clone(child));
        // 子 → 親: 弱参照
        *child.parent.borrow_mut() = Rc::downgrade(parent);
    }
}

impl Drop for GraphNode {
    fn drop(&mut self) {
        println!("  GraphNode '{}' を解放", self.name);
    }
}

fn main() {
    println!("=== 木構造 (所有権のみ) ===");
    {
        let mut root = TreeNode::new(1);
        let mut child1 = TreeNode::new(2);
        child1.add_child(TreeNode::new(4));
        child1.add_child(TreeNode::new(5));
        root.add_child(child1);
        root.add_child(TreeNode::new(3));
        println!("  {:?}", root);
    }  // root のスコープ終了 → 全ノードが自動解放（GC不要）

    println!("\n=== グラフ構造 (Rc + Weak) ===");
    {
        let parent = GraphNode::new("Parent");
        let child1 = GraphNode::new("Child1");
        let child2 = GraphNode::new("Child2");

        GraphNode::add_child(&parent, &child1);
        GraphNode::add_child(&parent, &child2);

        println!("  parent strong_count: {}", Rc::strong_count(&parent));
        println!("  child1 strong_count: {}", Rc::strong_count(&child1));

        // 子から親へのアクセス（弱参照経由）
        if let Some(p) = child1.parent.borrow().upgrade() {
            println!("  child1 の親: {}", p.name);
        }
    }  // スコープ終了 → Rc のカウントが 0 になり全て解放
    println!("  全ノード解放完了（循環参照なし）");
}
```

---

## 第7章: アンチパターンと設計上の落とし穴

### 7.1 アンチパターン1: Python での __del__ 依存

`__del__` メソッド（ファイナライザ）に重要なクリーンアップロジックを置くのは危険である。

```python
"""
アンチパターン: __del__ にリソース解放を依存する

問題点:
  1. 循環参照があると __del__ が呼ばれないことがある
  2. __del__ の実行順序は保証されない
  3. __del__ 内で例外が発生すると無視される
  4. インタプリタ終了時の __del__ の動作は未定義
"""

# NG: __del__ に依存したリソース管理
class BadDatabaseConnection:
    def __init__(self, dsn: str):
        self.conn = connect_to_database(dsn)  # 仮の関数
        self.is_open = True

    def __del__(self):
        # 危険: 循環参照があると呼ばれない可能性がある
        # 危険: インタプリタ終了時に connect_to_database が
        #       既に None になっている可能性がある
        if self.is_open:
            self.conn.close()
            self.is_open = False

# OK: コンテキストマネージャを使う
class GoodDatabaseConnection:
    def __init__(self, dsn: str):
        self.conn = connect_to_database(dsn)
        self.is_open = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False  # 例外は再送出

    def close(self):
        if self.is_open:
            self.conn.close()
            self.is_open = False

    def __del__(self):
        # 安全ネットとしてのみ使用
        if self.is_open:
            import warnings
            warnings.warn(
                f"GoodDatabaseConnection was not properly closed. "
                f"Use 'with' statement or call close() explicitly.",
                ResourceWarning,
                stacklevel=1
            )
            self.close()

# 正しい使い方:
# with GoodDatabaseConnection("postgresql://...") as db:
#     db.execute("SELECT ...")
# ← ここで確実に close() が呼ばれる
```

### 7.2 アンチパターン2: Swift のクロージャによる意図しない強参照

Swift で最も頻繁に遭遇するメモリリークパターンは、クロージャがオブジェクトを強参照でキャプチャすることによる循環参照である。

```swift
// アンチパターン: クロージャによる循環参照

import Foundation

// NG: クロージャが self を強参照でキャプチャ
class LeakyViewController {
    var name: String
    var onComplete: (() -> Void)?

    init(name: String) {
        self.name = name
        print("  [\(name)] init")
    }

    func setupCallback() {
        // 危険: self を暗黙的に強参照キャプチャ
        onComplete = {
            print("Completed: \(self.name)")  // self への強参照!
        }
        // 循環: self → onComplete → closure → self
    }

    deinit {
        print("  [\(name)] deinit")  // 呼ばれない!
    }
}

// OK: キャプチャリストで [weak self] を指定
class SafeViewController {
    var name: String
    var onComplete: (() -> Void)?

    init(name: String) {
        self.name = name
        print("  [\(name)] init")
    }

    func setupCallback() {
        // 安全: [weak self] でキャプチャ
        onComplete = { [weak self] in
            guard let self = self else {
                print("  既に解放済み")
                return
            }
            print("  Completed: \(self.name)")
        }
    }

    deinit {
        print("  [\(name)] deinit")  // 正しく呼ばれる
    }
}

// テスト
func testLeak() {
    print("=== リークするケース ===")
    var leaky: LeakyViewController? = LeakyViewController(name: "Leaky")
    leaky?.setupCallback()
    leaky = nil  // deinit が呼ばれない → リーク!
    print("  (deinit 未呼出 = メモリリーク)\n")
}

func testSafe() {
    print("=== 安全なケース ===")
    var safe: SafeViewController? = SafeViewController(name: "Safe")
    safe?.setupCallback()
    safe = nil  // deinit が正しく呼ばれる
}

testLeak()
testSafe()
```

### 7.3 アンチパターン3: Java での不要な強参照保持

```java
/**
 * アンチパターン: コレクションに不要なオブジェクトを保持し続ける
 *
 * トレーシングGCでも、ルートから到達可能な限り回収されない。
 * GC は「不要なオブジェクト」ではなく「到達不能なオブジェクト」を回収する。
 */
import java.util.*;
import java.lang.ref.WeakReference;

public class MemoryLeakPatterns {

    // NG: 無限に成長するキャッシュ
    static class BadCache {
        private final Map<String, byte[]> cache = new HashMap<>();

        void put(String key, byte[] value) {
            cache.put(key, value);  // 際限なく追加される
            // GC は cache から到達可能なので回収しない
        }
    }

    // OK: WeakHashMap による自動回収キャッシュ
    static class GoodCache {
        // キーが GC されると対応するエントリも自動削除
        private final WeakHashMap<String, byte[]> cache = new WeakHashMap<>();

        void put(String key, byte[] value) {
            cache.put(key, value);
        }
    }

    // OK: サイズ制限付き LRU キャッシュ
    static class BoundedCache<K, V> extends LinkedHashMap<K, V> {
        private final int maxSize;

        BoundedCache(int maxSize) {
            super(maxSize, 0.75f, true);  // accessOrder=true
            this.maxSize = maxSize;
        }

        @Override
        protected boolean removeEldestEntry(Map.Entry<K, V> eldest) {
            return size() > maxSize;  // 上限超過で最古エントリを削除
        }
    }
}
```

---

## 第8章: 現代の GC 技術 ── 最前線

### 8.1 ZGC: サブミリ秒停止の実現

Java の ZGC は、ヒープサイズに関係なく停止時間を 1ms 以下に抑えることを目標とする革新的な GC 実装である。

```
ZGC の核心技術: カラーポインタ (Colored Pointers)

  通常の64ビットポインタ:
  ┌────────────────────────────────────────────┐
  │ 63                                       0 │
  │ [         仮想アドレス (48bit)             ] │
  └────────────────────────────────────────────┘

  ZGC のカラーポインタ:
  ┌────────────────────────────────────────────┐
  │ 63    46  45  44  43  42  41           0   │
  │ [未使用] [F] [R] [M1][M0][ アドレス(42bit)]│
  └────────────────────────────────────────────┘
         │   │   │   │    │
         │   │   │   └────┴── マークビット (GC サイクル用)
         │   │   └── Remapped ビット
         │   └── Finalizable ビット
         └── 未使用

  ロードバリア:
    オブジェクト参照の読み取り時にポインタのメタデータを検査。
    もしオブジェクトが移動済みなら、新しいアドレスに転送。

    // 疑似コード
    Object* load_reference(Object** addr) {
        Object* ptr = *addr;
        if (is_bad_color(ptr)) {
            ptr = relocate_or_remap(ptr);
            *addr = ptr;  // ポインタを更新
        }
        return ptr;
    }

  結果:
    - Stop-The-World は GC ルートのスキャンのみ（数百μs）
    - マーク、リロケーション、参照更新は全て並行実行
    - ヒープサイズが TB 級でも停止時間は変わらない
```

### 8.2 Go の GC: シンプルさの哲学

Go の GC は意図的に世代別を採用せず、並行 Mark-Sweep のシンプルな設計を貫いている。

```
Go GC のアーキテクチャ:

  設計哲学: "Do less, but do it concurrently"

  ┌──────────────────────────────────────────┐
  │           Go GC のフェーズ                │
  ├──────────────────────────────────────────┤
  │                                          │
  │  1. Mark Setup (STW)     ~10-30μs       │
  │     ├── ライトバリアを有効化              │
  │     └── GC ワーカーgoroutineを起動        │
  │                                          │
  │  2. Concurrent Mark       並行実行        │
  │     ├── GC ワーカーが到達可能性を解析     │
  │     ├── アプリケーションは通常通り実行     │
  │     └── ライトバリアで新しい参照を追跡     │
  │                                          │
  │  3. Mark Termination (STW) ~10-30μs     │
  │     ├── ライトバリアを無効化              │
  │     └── マークの完了を確認                │
  │                                          │
  │  4. Concurrent Sweep      並行実行        │
  │     ├── マークされていないオブジェクトを回収│
  │     └── 次のアロケーション時に遅延実行     │
  │                                          │
  └──────────────────────────────────────────┘

  GOGC パラメータ:
  ──────────────────
  GOGC=100 (デフォルト): ライブデータの 100% 分の新規割り当て後にGC
  GOGC=200: より多くのメモリを使うがGC頻度は半減
  GOGC=50:  メモリ使用量を抑えるがGC頻度は倍増
  GOGC=off: GC を完全に無効化

  Go 1.19+ の GOMEMLIMIT:
  ──────────────────────
  メモリ使用量のソフトリミットを設定可能。
  リミット接近時に積極的にGCを実行。
  OOM 回避と性能のバランスを取る。
```

### 8.3 V8 (JavaScript) の Orinoco プロジェクト

Chrome の JavaScript エンジン V8 は、Orinoco プロジェクトで GC の大幅な並行化・並列化を推進した。

```
V8 Orinoco GC アーキテクチャ:

  ┌─────────────────────────────────────────┐
  │  Young Generation (Scavenger)            │
  │  ├── Semi-space copying GC               │
  │  ├── 並列実行 (複数ワーカースレッド)       │
  │  └── 停止時間: 1-2ms                     │
  ├─────────────────────────────────────────┤
  │  Old Generation                          │
  │  ├── 並行マーキング                       │
  │  │   └── メインスレッドと並行に実行        │
  │  ├── インクリメンタルマーキング             │
  │  │   └── 作業を小さなステップに分割        │
  │  ├── 遅延スイーピング                     │
  │  │   └── アロケーション時にオンデマンド     │
  │  └── 並行コンパクション                    │
  │      └── バックグラウンドスレッドで実行     │
  └─────────────────────────────────────────┘

  最適化技術:
  ─────────
  1. Write Barrier の最適化
     - ストアバッファリングで世代間参照を高速追跡
  2. Idle-time GC
     - ブラウザのアイドル時間を活用してGCを実行
     - requestIdleCallback と連携
  3. Concurrent allocation
     - バックグラウンドスレッドでメモリ確保を先行実行
```

### 8.4 次世代技術のトレンド

| トレンド | 概要 | 代表例 |
|---------|------|-------|
| リージョンベースメモリ管理 | オブジェクトをリージョン単位でまとめて回収 | Austral, MLKit |
| 所有権型 + エスケープ解析 | コンパイル時にスタック割り当て可能なオブジェクトを特定 | Java (JIT), Go |
| ハードウェア支援 GC | メモリコントローラや専用命令で GC を加速 | ARM MTE, Intel MPX |
| 不変データ構造の活用 | 不変データは世代昇格が不要で GC 負荷を軽減 | Clojure, Haskell |
| アリーナアロケーション | GC を回避し、リクエスト単位でまとめて解放 | Go arena (実験), Zig |

---

## 第9章: 用途別の選択指針 ── 設計判断フレームワーク

### 9.1 意思決定フロー

メモリ管理方式を選択する際の判断フローを示す。

```
メモリ管理方式の選択フローチャート:

  [スタート]
     │
     ▼
  リアルタイム制約があるか?
     │
     ├── はい → 停止時間の許容範囲は?
     │          │
     │          ├── 0ms (ハードリアルタイム)
     │          │   → 所有権ベース (Rust) or 手動管理 (C)
     │          │
     │          ├── <1ms (ソフトリアルタイム)
     │          │   → ARC (Swift) or ZGC (Java)
     │          │
     │          └── <10ms (準リアルタイム)
     │              → Go GC or Shenandoah (Java)
     │
     └── いいえ → スループットが重要か?
                  │
                  ├── はい → トレーシングGC (Java G1/Parallel)
                  │
                  └── いいえ → 開発速度が重要か?
                               │
                               ├── はい → ハイブリッド (Python, Ruby)
                               │
                               └── いいえ → メモリ効率が重要か?
                                            │
                                            ├── はい → 所有権 (Rust)
                                            │
                                            └── いいえ → 好みで選択
```

### 9.2 ドメイン別の推奨方式

```
┌───────────────────────────┬──────────────┬──────────────────────┐
│ ドメイン                   │ 推奨方式      │ 代表的な言語/技術     │
├───────────────────────────┼──────────────┼──────────────────────┤
│ モバイルアプリ (iOS)       │ ARC          │ Swift                │
│ モバイルアプリ (Android)   │ トレーシング  │ Kotlin/Java (ART)    │
│ Web フロントエンド         │ トレーシング  │ JavaScript (V8)      │
│ Web バックエンド           │ トレーシング  │ Java, Go, C#         │
│ データサイエンス           │ ハイブリッド  │ Python               │
│ ゲームエンジン             │ ARC/手動     │ Swift, C++, Rust     │
│ OS / カーネル             │ 手動/所有権  │ C, Rust              │
│ 組み込みシステム           │ 所有権/手動  │ Rust, C              │
│ 分散システム              │ トレーシング  │ Go, Java, Erlang     │
│ 機械学習推論              │ ハイブリッド  │ Python + C拡張       │
│ ブロックチェーン           │ トレーシング  │ Go, Rust, Solidity   │
│ CLI ツール                │ 所有権       │ Rust, Go             │
│ デスクトップアプリ         │ トレーシング  │ C#(WPF), Java(Swing) │
│ HPC (高性能計算)          │ 手動/所有権  │ C, C++, Rust, Fortran│
│ マイクロサービス           │ トレーシング  │ Go, Java, C#         │
└───────────────────────────┴──────────────┴──────────────────────┘
```

### 9.3 移行パスの考慮

既存システムのメモリ管理方式を変更する場合の指針を示す。

| 移行元 | 移行先 | 難易度 | 主な課題 |
|--------|--------|--------|---------|
| C (手動) | Rust (所有権) | 高 | 借用規則の学習、unsafe の適切な使用 |
| Python (RC+GC) | Go (Tracing) | 中 | GC チューニング、ポインタ意識 |
| Java (Tracing) | Go (Tracing) | 低 | GC パラメータ体系の違い |
| Obj-C (MRC) | Swift (ARC) | 低 | 自動化による安心感、weak/unowned の正しい使用 |
| C++ (手動+RAII) | Rust (所有権) | 中 | ライフタイム注釈、スマートポインタの対応関係 |
| Ruby (Tracing) | Java (Tracing) | 低 | 型システムの違いが主な課題 |

---

## 第10章: 演習問題 ── 3段階の実践

### 10.1 初級演習: 参照カウントの追跡

**課題**: 以下の Python コードにおいて、各行の実行後のオブジェクト X の参照カウントを手動で追跡せよ。

```python
"""
演習1: 参照カウントの手動追跡

各行の実行後に、X オブジェクトの参照カウントを答えよ。
sys.getrefcount() の「引数分の +1」は無視して純粋なカウントで回答すること。
"""
import sys

class X:
    pass

# Line 1
a = X()          # Q1: X の参照カウントは?

# Line 2
b = a            # Q2: X の参照カウントは?

# Line 3
c = [a, b, a]    # Q3: X の参照カウントは?

# Line 4
d = {"key": a}   # Q4: X の参照カウントは?

# Line 5
del c            # Q5: X の参照カウントは?

# Line 6
b = None         # Q6: X の参照カウントは?

# Line 7
d.clear()        # Q7: X の参照カウントは?

# Line 8
del a            # Q8: X の参照カウントは? X は解放されるか?
```

**解答**:
```
Q1: 1  (a のみ)
Q2: 2  (a, b)
Q3: 5  (a, b, c[0], c[1]=aの別名だがリスト要素として+1, c[2])
         → 正確には a, b, c[0], c[2] = 4個の参照 + b の参照 = 5
         ※ c = [a, b, a] では a が2回、b(=a)が1回リストに入る
         → a=1, b=1, list[0]=a=1, list[1]=b=aなので+1, list[2]=a=1 → 合計5
Q4: 6  (a, b, c[0], c[1], c[2], d["key"])
Q5: 3  (a, b, d["key"])  ← リスト c の解放で3つ減少
Q6: 2  (a, d["key"])
Q7: 1  (a のみ)
Q8: 0  → X は即座に解放される
```

### 10.2 中級演習: 循環参照の検出と解決

**課題**: 以下のコードにはメモリリークがある。原因を特定し、2つの異なる方法で修正せよ。

```python
"""
演習2: 循環参照のメモリリークを修正する

以下の EventSystem クラスにはメモリリークがある。
原因を特定し、(A) 弱参照、(B) 明示的切断 の2つの方法で修正せよ。
"""
import gc

class EventEmitter:
    def __init__(self, name: str):
        self.name = name
        self.listeners = []

    def on(self, callback):
        self.listeners.append(callback)

    def emit(self, data):
        for listener in self.listeners:
            listener(data)

    def __del__(self):
        print(f"  EventEmitter '{self.name}' 解放")

class Widget:
    def __init__(self, widget_id: str):
        self.widget_id = widget_id
        self.emitter = EventEmitter(f"emitter-{widget_id}")
        # 問題: self.handle_event は self への強参照を含むバウンドメソッド
        self.emitter.on(self.handle_event)

    def handle_event(self, data):
        print(f"  Widget {self.widget_id} received: {data}")

    def __del__(self):
        print(f"  Widget '{self.widget_id}' 解放")

# テスト
gc.disable()
w = Widget("btn-1")
w.emitter.emit("click")
del w
print("  del w 後: Widget も EventEmitter も解放されない (リーク!)")
gc.enable()
gc.collect()
print("  gc.collect() 後: GC が循環参照を回収")
```

**解答A: 弱参照を使った修正**:
```python
import weakref

class WidgetFixA:
    def __init__(self, widget_id: str):
        self.widget_id = widget_id
        self.emitter = EventEmitter(f"emitter-{widget_id}")
        # WeakMethod で弱参照のバウンドメソッドを登録
        weak_self = weakref.ref(self)
        def weak_handler(data, _ref=weak_self):
            obj = _ref()
            if obj is not None:
                obj.handle_event(data)
        self.emitter.on(weak_handler)

    def handle_event(self, data):
        print(f"  Widget {self.widget_id} received: {data}")

    def __del__(self):
        print(f"  WidgetFixA '{self.widget_id}' 解放")
```

**解答B: 明示的切断を使った修正**:
```python
class WidgetFixB:
    def __init__(self, widget_id: str):
        self.widget_id = widget_id
        self.emitter = EventEmitter(f"emitter-{widget_id}")
        self.emitter.on(self.handle_event)

    def handle_event(self, data):
        print(f"  Widget {self.widget_id} received: {data}")

    def destroy(self):
        """明示的にリスナーを切断してから参照を解除"""
        self.emitter.listeners.clear()
        self.emitter = None

    def __del__(self):
        print(f"  WidgetFixB '{self.widget_id}' 解放")

# 使い方:
# w = WidgetFixB("btn-1")
# w.destroy()  # 明示的切断
# del w        # 参照カウントで即座に回収
```

### 10.3 上級演習: 簡易参照カウント GC の実装

**課題**: Python で簡易的な参照カウント GC シミュレータを実装せよ。循環参照の検出機能も含めること。

```python
"""
演習3: 簡易参照カウント GC シミュレータの実装

以下のスケルトンを完成させよ。
要件:
  1. オブジェクトの生成と参照カウントの管理
  2. 参照の追加と削除
  3. 参照カウント == 0 での自動解放
  4. 循環参照の検出（ボーナス）
"""
from __future__ import annotations
from typing import Optional

class ManagedObject:
    """GC で管理されるオブジェクト"""
    _next_id = 0

    def __init__(self, name: str, gc: 'SimpleGC'):
        ManagedObject._next_id += 1
        self.id = ManagedObject._next_id
        self.name = name
        self.refcount = 0
        self.references: list[ManagedObject] = []
        self._gc = gc
        self._alive = True

    def add_reference(self, target: 'ManagedObject'):
        """target への参照を追加"""
        self.references.append(target)
        target.refcount += 1
        print(f"  {self.name} → {target.name}  "
              f"({target.name}.refcount = {target.refcount})")

    def remove_reference(self, target: 'ManagedObject'):
        """target への参照を削除"""
        if target in self.references:
            self.references.remove(target)
            target.refcount -= 1
            print(f"  {self.name} -/→ {target.name}  "
                  f"({target.name}.refcount = {target.refcount})")
            if target.refcount == 0:
                self._gc.free(target)

    def __repr__(self):
        refs = [r.name for r in self.references]
        return (f"Obj({self.name}, rc={self.refcount}, "
                f"refs={refs}, alive={self._alive})")


class SimpleGC:
    """簡易参照カウント GC"""

    def __init__(self):
        self.heap: list[ManagedObject] = []
        self.roots: dict[str, ManagedObject] = {}

    def allocate(self, name: str) -> ManagedObject:
        """新しいオブジェクトを割り当て"""
        obj = ManagedObject(name, self)
        self.heap.append(obj)
        print(f"  [ALLOC] {name} (heap size: {len(self.heap)})")
        return obj

    def add_root(self, var_name: str, obj: ManagedObject):
        """ルート変数を追加（参照カウント +1）"""
        if var_name in self.roots:
            old = self.roots[var_name]
            old.refcount -= 1
            if old.refcount == 0:
                self.free(old)
        self.roots[var_name] = obj
        obj.refcount += 1
        print(f"  [ROOT] {var_name} = {obj.name}  "
              f"({obj.name}.refcount = {obj.refcount})")

    def remove_root(self, var_name: str):
        """ルート変数を削除（参照カウント -1）"""
        if var_name in self.roots:
            obj = self.roots.pop(var_name)
            obj.refcount -= 1
            print(f"  [DEL]  {var_name}  "
                  f"({obj.name}.refcount = {obj.refcount})")
            if obj.refcount == 0:
                self.free(obj)

    def free(self, obj: ManagedObject):
        """オブジェクトを解放（参照先のカウントも減算）"""
        if not obj._alive:
            return
        obj._alive = False
        print(f"  [FREE] {obj.name}")

        # このオブジェクトが参照している先のカウントを減算
        for ref in obj.references[:]:
            ref.refcount -= 1
            print(f"    cascade: {ref.name}.refcount = {ref.refcount}")
            if ref.refcount == 0:
                self.free(ref)

        obj.references.clear()
        self.heap.remove(obj)

    def detect_cycles(self) -> list[set[str]]:
        """循環参照を検出（到達不能なオブジェクト群を返す）"""
        # ルートから到達可能なオブジェクトをマーク
        reachable = set()

        def mark(obj: ManagedObject):
            if obj.id in reachable:
                return
            reachable.add(obj.id)
            for ref in obj.references:
                if ref._alive:
                    mark(ref)

        for obj in self.roots.values():
            if obj._alive:
                mark(obj)

        # 到達不能なオブジェクトを検出
        unreachable = [obj for obj in self.heap
                       if obj.id not in reachable and obj._alive]

        if unreachable:
            names = {obj.name for obj in unreachable}
            print(f"  [CYCLE] 到達不能オブジェクト検出: {names}")
            return [names]
        return []

    def collect_cycles(self) -> int:
        """循環参照を強制回収"""
        cycles = self.detect_cycles()
        count = 0
        for cycle in cycles:
            for obj in self.heap[:]:
                if obj.name in cycle and obj._alive:
                    self.free(obj)
                    count += 1
        return count

    def status(self):
        """現在の状態を表示"""
        print(f"\n  --- GC Status ---")
        print(f"  Roots: {list(self.roots.keys())}")
        print(f"  Heap ({len(self.heap)} objects):")
        for obj in self.heap:
            print(f"    {obj}")
        print()


# === テスト実行 ===
def test_simple_gc():
    print("=== 簡易GCシミュレータ テスト ===\n")

    gc = SimpleGC()

    # 通常の参照カウント動作
    print("--- 1. 基本的な割り当てと解放 ---")
    a = gc.allocate("A")
    b = gc.allocate("B")
    gc.add_root("x", a)
    gc.add_root("y", b)
    a.add_reference(b)
    gc.status()

    gc.remove_root("x")  # A のカウントが 0 → A 解放 → B のカウント -1
    gc.status()

    gc.remove_root("y")  # B のカウントが 0 → B 解放
    gc.status()

    # 循環参照テスト
    print("--- 2. 循環参照の検出 ---")
    c = gc.allocate("C")
    d = gc.allocate("D")
    gc.add_root("z", c)
    c.add_reference(d)
    d.add_reference(c)  # 循環: C ↔ D
    gc.status()

    gc.remove_root("z")  # C.refcount=1 (D→C), D.refcount=1 (C→D) → リーク!
    gc.status()

    print("--- 3. 循環参照の回収 ---")
    collected = gc.collect_cycles()
    print(f"  回収数: {collected}")
    gc.status()

if __name__ == "__main__":
    test_simple_gc()
```

---

## 第11章: 総合比較表

### 11.1 方式別の包括的比較

```
┌──────────────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│ 評価項目              │ 参照カウント  │ トレーシングGC│ ハイブリッド  │ 所有権ベース  │
├──────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ 回収タイミング         │ 即座(決定的)  │ 不定(非決定的)│ 即座+定期     │ スコープ終了  │
│ 循環参照処理          │ 不可         │ 可能         │ 可能         │ Weak で回避  │
│ 最大停止時間          │ 0ms          │ 数ms〜数百ms │ 短い         │ 0ms          │
│ スループット          │ やや低い      │ 高い         │ 中〜高       │ 最高         │
│ メモリ効率            │ 中           │ 低〜中       │ 中           │ 最高         │
│ オブジェクト毎コスト   │ 8-16 bytes   │ 0-1 bit      │ 8-16 bytes   │ 0 bytes      │
│ マルチスレッド性能     │ 低(atomic)   │ 高           │ 中           │ 高           │
│ 実装複雑度            │ 低           │ 高           │ 高           │ 中(コンパイラ)│
│ 学習コスト(開発者)     │ 低           │ 低           │ 低           │ 高           │
│ デストラクタの確実性   │ 高           │ 低           │ 中           │ 高           │
│ デバッグ容易性        │ 高           │ 低           │ 中           │ 中           │
│ 代表言語             │ Swift,Python │ Java,Go,JS   │ Python,C#    │ Rust         │
│ 適用領域             │ モバイル     │ サーバー     │ スクリプト   │ システム     │
│                      │ デスクトップ  │ Web          │ データ分析   │ 組み込み     │
└──────────────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
```

### 11.2 言語別のメモリ管理実装詳細

| 言語 | 主方式 | 補助方式 | GC アルゴリズム | 停止時間目安 | 特記事項 |
|------|--------|---------|---------------|------------|---------|
| Python | RC | 世代別GC | 試行的削除 | ~10ms | GIL による簡素化 |
| Swift | ARC | なし | なし | 0ms | コンパイラ挿入 retain/release |
| Java | Tracing | - | G1/ZGC/Shenandoah | <1ms (ZGC) | 最も成熟した GC エコシステム |
| Go | Tracing | - | 並行 Mark-Sweep | <500us | 世代別なし（意図的） |
| JavaScript (V8) | Tracing | - | 世代別+並行+インクリメンタル | ~1-2ms | Orinoco プロジェクト |
| C# (.NET) | Tracing | IDisposable | 世代別+コンパクション | ~10ms | Server/Workstation モード |
| Ruby | Tracing | - | 世代別 Mark-Sweep | ~10ms | Ruby 3.x で改善 |
| Rust | 所有権 | Rc/Arc | なし | 0ms | コンパイル時検証 |
| Erlang/Elixir | Tracing | - | プロセス別 GC | <1ms/proc | プロセス単位で独立した GC |
| OCaml | Tracing | - | 世代別+インクリメンタル | 数ms | 関数型言語向けに最適化 |
| Haskell | Tracing | - | 世代別 Copying GC | 可変 | 遅延評価との相互作用 |
| PHP | RC | 循環検出 | Mark-Sweep (循環のみ) | ~1ms | リクエスト終了で全解放 |
| Perl | RC | なし | なし | 0ms | 循環参照は手動管理 |
| Lua | Tracing | - | インクリメンタル Mark-Sweep | 数ms | 軽量 GC |

---

## 第12章: FAQ（よくある質問）

### Q1: 「参照カウントは遅い」という通説は正しいか?

**回答**: 部分的に正しいが、単純化しすぎている。

素朴な参照カウントは確かに毎回の参照操作にオーバーヘッドが生じる。特にマルチスレッド環境ではアトミック操作のコストが大きい。しかし、現代の最適化された参照カウント（Swift の ARC、遅延参照カウントなど）は、多くのユースケースでトレーシング GC と同等以上の性能を発揮する。

ポイントは「何を測定するか」である。スループット（単位時間あたりの処理量）ではトレーシング GC が有利な傾向にあるが、レイテンシ（最大停止時間）では参照カウントが有利である。アプリケーションの要件に応じて「遅い」の意味が変わる。

### Q2: Go はなぜ世代別GCを採用しないのか?

**回答**: Go チームは意図的に世代別 GC を採用していない。理由は以下の通り:

1. **値型の活用**: Go はスライス、マップのキー、構造体などを値型として扱うことが多く、ヒープアロケーションが他の言語より少ない。世代仮説（「ほとんどのオブジェクトは若くして死ぬ」）の前提が弱くなる。

2. **ライトバリアのコスト**: 世代別 GC にはライトバリア（世代間参照の追跡）が必要だが、これは全ポインタ書き込みにオーバーヘッドを加える。Go はこのコストを避けたい。

3. **シンプルさ**: Go の設計哲学は「シンプルさ」を重視する。世代別 GC はチューニングパラメータが増え、複雑さが増す。

4. **十分な性能**: 並行 Mark-Sweep だけで 500μs 以下の停止時間を達成しており、多くのユースケースで十分である。

ただし、Go 1.19 で導入された `GOMEMLIMIT` は、将来的な世代別 GC 導入への布石とも解釈できる。

### Q3: Rust では GC が完全に不要なのか?

**回答**: 厳密には「不要」ではなく「デフォルトでは不要」が正しい。

Rust の所有権システムはコンパイル時にメモリの生存期間を決定するため、実行時の GC は不要である。しかし、以下の場合にはランタイムのメモリ管理が必要となる:

- **共有所有権が必要な場合**: `Rc<T>`（単一スレッド）や `Arc<T>`（マルチスレッド）を使用。これは参照カウントである。
- **循環構造が必要な場合**: `Weak<T>` と組み合わせるか、アリーナアロケータを使用。
- **FFI でCライブラリと連携する場合**: C側のメモリ管理に合わせる必要がある。

また、`rust-gc` クレートなどの外部ライブラリを使えば、Rust でもトレーシング GC を利用可能。ゲームエンジンや言語処理系の実装では有用な場合がある。

### Q4: Python の GC を無効化するとどうなるか?

**回答**: `gc.disable()` で世代別 GC を無効化できるが、参照カウントは引き続き動作する。

- 循環参照を含まないオブジェクトは通常通り即座に回収される
- 循環参照を含むオブジェクトはメモリリークとなる
- Instagram は実際に本番環境で GC を無効化し、メモリ使用量を 10% 削減した（2017年の発表）。ただしこれは、循環参照が発生しないようにコードを厳密に管理した上での判断である

### Q5: ファイナライザ（デストラクタ）に依存してよいのはどの言語か?

**回答**: 言語のメモリ管理方式によって信頼性が異なる。

| 言語 | ファイナライザ | 信頼性 | 推奨される代替手段 |
|------|--------------|--------|------------------|
| Rust | `Drop` trait | 確実（スコープ終了時） | 不要（Drop で十分） |
| Swift | `deinit` | ほぼ確実（ARC） | weak/unowned の正しい使用 |
| C++ | デストラクタ | 確実（RAII） | 不要（RAII で十分） |
| Python | `__del__` | 不確実 | `with` 文 / `contextlib` |
| Java | `finalize` (deprecated) | 非推奨 | `try-with-resources` / `Cleaner` |
| C# | `~Finalizer` | 遅延実行 | `IDisposable` + `using` |
| Go | `runtime.SetFinalizer` | 不確実 | `defer` / 明示的 `Close()` |

---

## 第13章: まとめ

### 13.1 核心的な学び

1. **参照カウントとトレーシング GC はトレードオフの関係** にある。前者は即時性と予測可能性に優れ、後者はスループットと循環参照処理に優れる。

2. **現代の言語はほぼ全てがハイブリッド** である。純粋な参照カウントのみ、純粋なトレーシング GC のみの言語は少数派。多くの言語が複数の手法を組み合わせている。

3. **所有権ベースのメモリ管理（Rust）は第三の道** を提示した。コンパイル時にメモリの生存期間を決定することで、実行時コストをゼロにする。ただし学習コストが高い。

4. **GC 技術は急速に進化** している。Java の ZGC は 1ms 以下の停止時間を実現し、「トレーシング GC = 長い停止」という通説を覆しつつある。

5. **最適な選択はドメインに依存** する。リアルタイム性が重要なら参照カウントか所有権、スループットが重要ならトレーシング GC、開発速度が重要ならハイブリッドが適する。

### 13.2 方式別サマリー

| 方式 | 回収タイミング | 循環参照 | 停止時間 | 代表言語 |
|------|-------------|---------|---------|---------|
| 参照カウント | 即座 | 不可 | なし | Swift, Python |
| トレーシング | バッチ | 可能 | あり | Java, Go, JS |
| ハイブリッド | 混合 | 可能 | 最小 | Python, C# |
| 所有権 | スコープ終了 | N/A | なし | Rust |

---


## FAQ

### Q1: このトピックを学ぶ上で最も重要なポイントは何ですか？

実践的な経験を積むことが最も重要です。理論だけでなく、実際にコードを書いて動作を確認することで理解が深まります。

### Q2: 初心者がよく陥る間違いは何ですか？

基礎を飛ばして応用に進むことです。このガイドで説明している基本概念をしっかり理解してから、次のステップに進むことをお勧めします。

### Q3: 実務ではどのように活用されていますか？

このトピックの知識は、日常的な開発業務で頻繁に活用されます。特にコードレビューやアーキテクチャ設計の際に重要になります。

---

## まとめ

このガイドでは以下の重要なポイントを学びました:

- 基本概念と原則の理解
- 実践的な実装パターン
- ベストプラクティスと注意点
- 実務での活用方法

---

## 次に読むべきガイド


---

## 用語集

| 用語 | 英語 | 説明 |
|------|------|------|
| 参照カウント | Reference Counting (RC) | オブジェクトへの参照数を記録し、0 で回収する方式 |
| トレーシングGC | Tracing GC | ルートから到達可能なオブジェクトを辿る方式 |
| ARC | Automatic Reference Counting | コンパイラが retain/release を自動挿入する参照カウント |
| STW | Stop-The-World | GC 実行中にアプリケーションが停止する現象 |
| 世代仮説 | Generational Hypothesis | 大半のオブジェクトは短命であるという経験的観測 |
| ライトバリア | Write Barrier | ポインタ書き込み時に GC に通知する機構 |
| ロードバリア | Load Barrier | ポインタ読み取り時に GC の状態を検査する機構 |
| コンパクション | Compaction | 生存オブジェクトをメモリの一方に寄せ断片化を解消する処理 |
| 弱参照 | Weak Reference | 参照カウントを増加させない参照 |
| ルート集合 | Root Set | GC の探索起点となる変数群（スタック、グローバル、レジスタ） |
| カラーポインタ | Colored Pointer | ポインタのビットに GC メタデータを埋め込む技法（ZGC） |
| フリーリスト | Free List | 解放済みメモリブロックの連結リスト |
| バンプポインタ | Bump Pointer | ポインタを進めるだけで高速にメモリ確保する方式 |
| アリーナ | Arena | 一括確保・一括解放するメモリ領域 |
| RAII | Resource Acquisition Is Initialization | リソースの寿命をオブジェクトのスコープに紐づけるイディオム |

---

## 参考文献

1. Jones, R., Hosking, A. & Moss, E. *The Garbage Collection Handbook: The Art of Automatic Memory Management.* 2nd Edition, CRC Press, 2023. -- GC の包括的な教科書。Mark-Sweep、Copying GC、世代別 GC、並行 GC の全てを詳細に解説。

2. Bacon, D. F., Cheng, P. & Rajan, V. T. "A Unified Theory of Garbage Collection." *ACM SIGPLAN Notices*, Vol. 39, No. 10, pp. 50-68, 2004. (OOPSLA 2004) -- 参照カウントとトレーシング GC が数学的に双対であることを証明した画期的論文。

3. Apple Inc. *Automatic Reference Counting (ARC) -- Swift Documentation.* 2024. https://docs.swift.org/swift-book/documentation/the-swift-programming-language/automaticreferencecounting/ -- Swift ARC の公式ドキュメント。weak, unowned, クロージャキャプチャリストの使い方。

4. Klabnik, S. & Nichols, C. *The Rust Programming Language.* 2nd Edition, No Starch Press, 2023. -- 所有権システム、借用、ライフタイム、`Rc<T>` と `Arc<T>` の解説。

5. Oracle. *Java Garbage Collection Tuning Guide -- Java SE 21.* 2024. https://docs.oracle.com/en/java/javase/21/gctuning/ -- Java の各 GC（Serial, Parallel, G1, ZGC, Shenandoah）のチューニングガイド。

6. Prossimo, R. et al. "Trash Talk: A Deep Dive into V8's Garbage Collection." *V8 Blog*, 2019. https://v8.dev/blog/trash-talk -- V8 の Orinoco GC プロジェクトの詳細な解説。

7. Go Team. *A Guide to the Go Garbage Collector.* 2022. https://tip.golang.org/doc/gc-guide -- Go GC の設計思想、GOGC パラメータ、GOMEMLIMIT の解説。

8. Instagram Engineering. "Dismissing Python Garbage Collection at Instagram." *Instagram Engineering Blog*, 2017. -- Python の世代別 GC を無効化して性能改善した事例。

9. Tene, G., Iyengar, B. & Wolf, M. "C4: The Continuously Concurrent Compacting Collector." *ACM ISMM*, 2011. -- Azul の C4 GC（ZGC の前身的アイデア）の論文。

10. Collins, G. E. "A Method for Overlapping and Erasure of Lists." *Communications of the ACM*, Vol. 3, No. 12, pp. 655-657, 1960. -- 参照カウントの原論文。

---

*本ガイドは MIT レベルの CS 教育を目標とした包括的な資料である。基礎概念から最新の GC 技術まで体系的にカバーし、各言語の実装を横断的に比較することで、メモリ管理に対する深い理解を促すことを意図している。*
