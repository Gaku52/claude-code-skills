# スタックとヒープ

> プログラムのメモリは「スタック」（高速・自動管理）と「ヒープ」（柔軟・手動/GC管理）に分かれる。この区別を理解することが、パフォーマンスとメモリ安全性の基盤となる。

## この章で学ぶこと

- [ ] スタックとヒープの違いと特性を理解する
- [ ] 各言語でのメモリ配置の違いを把握する
- [ ] メモリレイアウトを意識した効率的なコードが書ける
- [ ] アライメントとパディングの影響を理解する
- [ ] キャッシュフレンドリーなデータ設計ができる
- [ ] メモリ関連のバグ（リーク、バッファオーバーフロー等）を防ぐ知識を持つ


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解

---

## 1. メモリレイアウト

```
プロセスのメモリ空間:

  高アドレス ┌─────────────────────┐
             │    カーネル空間      │  ← ユーザプロセスからアクセス不可
             ├─────────────────────┤
             │    スタック ↓        │  ← 自動管理、高速、サイズ制限
             │    (Stack)          │     関数のローカル変数、引数
             │                     │
             │    ↓ 成長方向       │
             │                     │
             │    (未使用領域)      │  ← スタックとヒープの間の空き
             │                     │
             │    ↑ 成長方向       │
             │                     │
             │    ヒープ ↑         │  ← 動的管理、柔軟、大きなデータ
             │    (Heap)           │     new/malloc で確保
             ├─────────────────────┤
             │    BSS              │  ← 未初期化グローバル変数（0で初期化）
             ├─────────────────────┤
             │    Data             │  ← 初期化済みグローバル変数・静的変数
             ├─────────────────────┤
             │    Text (Code)      │  ← プログラムの命令コード（読み取り専用）
  低アドレス └─────────────────────┘
```

### 各セグメントの詳細

```
Text (Code) セグメント:
  - 実行可能な機械語命令が格納される
  - 読み取り専用（書き込み不可）
  - 複数プロセスで共有可能（共有ライブラリ）
  - サイズはコンパイル時に決定

Data セグメント:
  - 初期化済みのグローバル変数・静的変数
  - 例: static int count = 42;
  - プログラム起動時にファイルからロード

BSS (Block Started by Symbol) セグメント:
  - 未初期化のグローバル変数・静的変数
  - ゼロで初期化される
  - 例: static int buffer[1024];
  - ファイル上はサイズ情報のみ（実際の0は格納しない）

ヒープ:
  - malloc/new 等で動的に確保
  - プログラマ（またはGC）が管理
  - 低アドレスから高アドレスに向かって成長

スタック:
  - 関数呼び出し時に自動確保
  - LIFO で管理
  - 高アドレスから低アドレスに向かって成長
```

### 仮想メモリと物理メモリ

```
仮想メモリの役割:
  - 各プロセスに独立したアドレス空間を提供
  - 物理メモリより大きなアドレス空間を使用可能
  - メモリ保護（他プロセスのメモリにアクセス不可）

ページテーブルによるマッピング:
  仮想アドレス ──→ ページテーブル ──→ 物理アドレス

  ┌──────────────────┐      ┌──────────────────┐
  │ 仮想ページ 0      │──→   │ 物理フレーム 5    │
  │ 仮想ページ 1      │──→   │ 物理フレーム 2    │
  │ 仮想ページ 2      │──→   │ ディスク（スワップ）│
  │ 仮想ページ 3      │──→   │ 物理フレーム 8    │
  └──────────────────┘      └──────────────────┘

  - ページサイズ: 通常 4KB（x86/x86_64）
  - TLB（Translation Lookaside Buffer）でキャッシュ
  - ページフォルト: 物理メモリにないページへのアクセス → OS がロード

メモリマップドファイル:
  - ファイルの内容を仮想メモリにマッピング
  - mmap (POSIX) / CreateFileMapping (Windows)
  - 大きなファイルの効率的な読み書き
  - 複数プロセス間でのメモリ共有
```

---

## 2. スタック（Stack）

```
特徴:
  - LIFO（Last In, First Out）
  - 自動的に確保・解放（関数の開始/終了時）
  - 非常に高速（ポインタの加減算のみ）
  - サイズ制限あり（通常 1〜8 MB）
  - 連続したメモリ領域
  - スレッドごとに独立

格納されるもの:
  - 関数の引数
  - ローカル変数（固定サイズ）
  - 戻りアドレス
  - フレームポインタ
  - レジスタの退避値
  - アライメントパディング
```

### スタックフレームの詳細

```c
// C: スタック上の変数
void example() {
    int x = 42;           // スタック上に 4 バイト確保
    double y = 3.14;      // スタック上に 8 バイト確保
    char buf[256];        // スタック上に 256 バイト確保
}  // 関数終了時に全て自動解放

// 関数呼び出しのスタックフレーム:
// ┌──────────────────┐ ← SP（スタックポインタ）
// │ buf[256]         │
// │ y (3.14)         │
// │ x (42)           │
// │ 戻りアドレス      │
// │ 前のフレームポインタ│
// ├──────────────────┤ ← 呼び出し元のフレーム

// 詳細なスタックフレーム（x86_64 System V ABI）
// ┌──────────────────┐ ← RSP（スタックポインタ）
// │ ローカル変数      │
// │ 一時変数          │
// │ 退避レジスタ      │
// │ パディング        │  ← 16バイトアライメント
// ├──────────────────┤ ← RBP（ベースポインタ）
// │ 前の RBP         │
// │ 戻りアドレス      │
// │ 第7引数以降      │  ← レジスタに収まらない引数
// ├──────────────────┤
// │ 呼び出し元の      │
// │ スタックフレーム   │
// └──────────────────┘
```

### 呼び出し規約

```
x86_64 System V ABI（Linux/macOS）:
  引数: RDI, RSI, RDX, RCX, R8, R9（6個までレジスタ）
  浮動小数点: XMM0-XMM7（8個までレジスタ）
  戻り値: RAX（整数）、XMM0（浮動小数点）
  7個目以降の引数: スタック

x86_64 Windows ABI:
  引数: RCX, RDX, R8, R9（4個までレジスタ）
  浮動小数点: XMM0-XMM3
  5個目以降の引数: スタック
  シャドウスペース: 32バイト（呼び出し先がレジスタ引数を退避するため）
```

```c
// 呼び出し規約の影響を理解する例
struct SmallStruct { int x, y; };        // 16バイト → レジスタで渡せる
struct LargeStruct { int data[100]; };   // 400バイト → ポインタで渡される

// 小さい構造体は値渡しでもオーバーヘッドが小さい
void process_small(struct SmallStruct s) {
    // s はレジスタまたは少量のスタックで渡される
}

// 大きい構造体はポインタで渡すべき
void process_large(const struct LargeStruct *s) {
    // ポインタ（8バイト）だけがレジスタで渡される
}
```

### 各言語でのスタック使用

```rust
// Rust: スタック上のデータ
fn example() {
    let x: i32 = 42;          // スタック（4バイト）
    let point = (3.0_f64, 4.0_f64); // スタック（16バイト）
    let arr = [1, 2, 3, 4, 5]; // スタック（20バイト）

    // 構造体もスタック上
    struct Point { x: f64, y: f64 }
    let p = Point { x: 1.0, y: 2.0 }; // スタック（16バイト）

    // enum もスタック上
    let opt: Option<i32> = Some(42); // スタック（8バイト: i32 + 判別子）
}  // 全て自動解放

// Rust ではスタックに置かれる型のサイズがコンパイル時に分かる
// Sized トレイト: コンパイル時にサイズが決定する型
fn takes_sized<T: Sized>(value: T) {
    // T のサイズはコンパイル時に決定
}

// ?Sized: サイズが不定の型も受け入れる
fn takes_unsized<T: ?Sized>(value: &T) {
    // T は [u8] や str のようなサイズ不定の型でもOK
}
```

```go
// Go: スタックの動的成長
// Go のスタックは初期サイズが小さく（数KB）、必要に応じて自動成長する
// これにより大量の goroutine を効率的に扱える

func example() {
    x := 42              // スタック（エスケープ解析の結果による）
    arr := [5]int{1,2,3,4,5} // スタック（固定サイズ配列）
    // スライス自体のヘッダ（ポインタ、長さ、容量）はスタック
    // スライスの背後のデータはヒープ
}

// Go のスタックの特徴:
// - 初期サイズ: 2KB〜8KB（バージョンにより異なる）
// - 必要に応じて2倍ずつ成長
// - 成長時はコピー方式（contiguous stack）
// - goroutine ごとに独立したスタック
// - スタックポインタの書き換えが必要（コピー時）
```

```java
// Java: スタックに置かれるもの
void example() {
    int x = 42;           // スタック（プリミティブ型）
    double y = 3.14;      // スタック（プリミティブ型）
    boolean flag = true;  // スタック（プリミティブ型）

    // 参照変数はスタック、オブジェクト本体はヒープ
    String s = "hello";   // 参照（8バイト）はスタック
                          // "hello" オブジェクトはヒープ

    int[] arr = new int[10]; // 参照はスタック、配列本体はヒープ
    Object obj = new Object(); // 参照はスタック、オブジェクトはヒープ
}

// JIT コンパイラのスカラー置換（Scalar Replacement）
// JIT が脱出解析（Escape Analysis）を行い、
// ヒープ確保を省略してスタックに置くことがある
void optimized() {
    Point p = new Point(1, 2);  // 脱出しない → スカラー置換される可能性
    int sum = p.x + p.y;        // p.x と p.y が直接スタックの変数になる
}
```

```c
// C++: スタックとスマートポインタ
void example() {
    // スタック上のオブジェクト（RAII）
    std::string s = "hello";     // スタック上の String オブジェクト
                                  // 内部バッファはヒープ

    std::vector<int> v = {1, 2, 3}; // スタック上の Vector オブジェクト
                                     // 要素データはヒープ

    // SSO (Small String Optimization)
    // 短い文字列（通常15〜22文字以下）はヒープを使わずスタック内に格納
    std::string short_s = "hi"; // ヒープ確保なし（SSO）

    // SBO (Small Buffer Optimization) は std::function 等でも使われる
    std::function<int(int)> f =  { return x * 2; };
    // 小さなラムダはヒープ確保なし
}
```

---

## 3. ヒープ（Heap）

```
特徴:
  - 動的にサイズを決定できる
  - プログラマ（またはGC）が管理
  - スタックより低速（アロケータの管理コスト）
  - サイズ制限は物理メモリ（+スワップ）まで
  - 断片化の問題がある
  - 全スレッドで共有

格納されるもの:
  - 動的サイズのデータ（文字列、配列、コレクション）
  - オブジェクト（多くの言語で）
  - 関数の寿命を超えて生存するデータ
  - クロージャのキャプチャ変数（多くの場合）
```

### メモリアロケータの仕組み

```
malloc/free の内部動作（概念的）:

  1. フリーリスト方式:
     ┌─────┐   ┌─────┐   ┌─────┐
     │Free │──→│Free │──→│Free │──→ NULL
     │64B  │   │128B │   │32B  │
     └─────┘   └─────┘   └─────┘

     malloc(50) → 64B のブロックを返す
     free(ptr)  → フリーリストに返却

  2. バディシステム:
     メモリを2のべき乗サイズのブロックに分割
     256B → [128B] + [128B]
            [128B] → [64B] + [64B]
     要求に最も近い2のべき乗サイズのブロックを割り当て

  3. スラブアロケータ（Linux カーネル）:
     同じサイズのオブジェクトをまとめて管理
     キャッシュ効率が良い
     内部断片化を最小限に抑える

  4. jemalloc / tcmalloc:
     スレッドローカルキャッシュで高速化
     サイズクラスごとにプール管理
     断片化を最小限に抑える設計
```

### メモリ断片化

```
外部断片化:
  ┌────┬─────┬────┬─────┬────┐
  │使用│ Free │使用│ Free │使用│
  │64B │ 32B  │64B │ 48B  │64B │
  └────┴─────┴────┴─────┴────┘
  合計 80B の空きがあるが、64B のブロックを確保できない
  → 空きが散在して大きなブロックを確保不可

内部断片化:
  要求: 50B → 割り当て: 64B（アライメントのため）
  14B が無駄になる

対策:
  - コンパクション: 生存オブジェクトを移動して空きをまとめる（GC言語）
  - メモリプール: 同じサイズのオブジェクトを専用領域で管理
  - Arena アロケータ: まとめて確保・まとめて解放
  - バンプアロケータ: ポインタを進めるだけで確保（解放は一括）
```

```c
// C: ヒープの手動管理
void example() {
    // malloc でヒープに確保
    int *arr = (int *)malloc(100 * sizeof(int));
    if (arr == NULL) {
        // メモリ確保失敗
        return;
    }

    arr[0] = 42;

    // realloc でサイズ変更
    int *new_arr = (int *)realloc(arr, 200 * sizeof(int));
    if (new_arr == NULL) {
        free(arr);  // realloc 失敗時は元のメモリを解放
        return;
    }
    arr = new_arr;

    // calloc: ゼロ初期化付き確保
    int *zeroed = (int *)calloc(100, sizeof(int));

    // 必ず free で解放（忘れるとメモリリーク）
    free(arr);
    arr = NULL;  // ダングリングポインタ防止
    free(zeroed);
    zeroed = NULL;
}

// メモリリークの典型的パターン
void leak_example() {
    char *buf = (char *)malloc(256);
    // ... 処理 ...
    if (error_condition) {
        return;  // free を忘れている → メモリリーク
    }
    free(buf);
}

// ダングリングポインタ
void dangling_example() {
    int *p = (int *)malloc(sizeof(int));
    *p = 42;
    free(p);
    // *p = 100;  // 未定義動作! 解放済みメモリへのアクセス
}

// バッファオーバーフロー
void overflow_example() {
    char buf[10];
    strcpy(buf, "This string is way too long"); // バッファオーバーフロー
    // スタック上の他のデータ（戻りアドレス等）を上書きする可能性
    // セキュリティ脆弱性の原因
}
```

```rust
// Rust: Box でヒープに確保
fn example() {
    let x = Box::new(42);     // ヒープに i32 を確保
    let s = String::from("hello"); // ヒープに文字列を確保
    let v = vec![1, 2, 3];    // ヒープに配列を確保

    // スコープを抜けると自動的に解放（所有権システム）

    // String のメモリレイアウト
    // スタック: [ポインタ | 長さ | 容量] = 24バイト
    //            ↓
    // ヒープ:  [h|e|l|l|o|_|_|_]  (容量分確保)

    // Vec のメモリレイアウト
    // スタック: [ポインタ | 長さ | 容量] = 24バイト
    //            ↓
    // ヒープ:  [1|2|3|_|_|_|_|_]  (容量分確保)
}

// カスタムアロケータ（Rust 1.28+ で安定化）
use std::alloc::{GlobalAlloc, System, Layout};

struct CountingAllocator;

static ALLOCATED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATED.fetch_add(layout.size(), std::sync::atomic::Ordering::SeqCst);
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        ALLOCATED.fetch_sub(layout.size(), std::sync::atomic::Ordering::SeqCst);
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static A: CountingAllocator = CountingAllocator;
```

```python
# Python: ほぼ全てがヒープ
x = 42           # ヒープ上の int オブジェクト
s = "hello"      # ヒープ上の str オブジェクト
lst = [1, 2, 3]  # ヒープ上の list オブジェクト

# GC が自動的に解放
# プログラマはメモリ管理を意識しなくてよい

# Python オブジェクトのメモリレイアウト（CPython）
import sys
sys.getsizeof(42)       # → 28 バイト（int オブジェクト）
sys.getsizeof("hello")  # → 54 バイト（str オブジェクト）
sys.getsizeof([1,2,3])  # → 120 バイト（list オブジェクト + 参照×3）

# 各オブジェクトのヘッダ:
#   参照カウント: 8 バイト
#   型ポインタ:   8 バイト
#   → 最低 16 バイトのオーバーヘッド

# メモリプロファイリング
import tracemalloc
tracemalloc.start()

# ... 処理 ...

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
for stat in top_stats[:10]:
    print(stat)
```

```javascript
// JavaScript: 全てがヒープ（V8 エンジンの場合）
// ただし JIT の最適化でスタックに置かれることもある

// V8 のオブジェクトレイアウト
// Hidden Class（Map）ベースの高速プロパティアクセス
const obj = { x: 1, y: 2 };
// Hidden Class: { x: offset 0, y: offset 8 }
// インラインプロパティとして高速アクセス

// SMI (Small Integer): 31ビット整数はポインタにインライン化
// → ヒープ確保なし
const x = 42;  // SMI として表現（ヒープ確保なし）

// HeapNumber: SMI に収まらない数値はヒープ確保
const y = 1.5;  // HeapNumber としてヒープに確保

// ArrayBuffer: TypedArray のバッキングストア
const buffer = new ArrayBuffer(1024);     // 1KB のヒープ確保
const view = new Float64Array(buffer);    // ビューはヒープ確保が小さい
```

---

## 4. 言語ごとのメモリ配置

```
┌────────────────┬────────────────────┬──────────────────┐
│ 言語            │ スタック            │ ヒープ             │
├────────────────┼────────────────────┼──────────────────┤
│ C / C++        │ プリミティブ型      │ malloc/new        │
│                │ 固定サイズ配列      │ 動的配列          │
│                │ 構造体              │ ポインタ経由      │
│                │ SSO 対象の短い文字列│ 長い文字列        │
├────────────────┼────────────────────┼──────────────────┤
│ Rust           │ プリミティブ型      │ Box<T>           │
│                │ 固定サイズ型        │ String, Vec<T>   │
│                │ 参照（&T）         │ Rc<T>, Arc<T>    │
│                │ 配列 [T; N]        │ HashMap, BTreeMap│
├────────────────┼────────────────────┼──────────────────┤
│ Go             │ エスケープ解析で決定│ エスケープ解析で決定│
│                │ （コンパイラが最適化）│                  │
│                │ 小さい構造体        │ 大きい構造体      │
│                │ ローカル変数        │ ポインタで返す値  │
├────────────────┼────────────────────┼──────────────────┤
│ Java           │ プリミティブ型      │ 全オブジェクト    │
│                │ 参照変数自体        │ 配列、String      │
│                │                    │ (JIT脱出解析あり) │
├────────────────┼────────────────────┼──────────────────┤
│ C#             │ 値型(struct)       │ 参照型(class)    │
│                │ プリミティブ型      │ 配列、String      │
│                │ stackalloc         │ new              │
├────────────────┼────────────────────┼──────────────────┤
│ Python/Ruby    │ （ほぼ使わない）    │ 全オブジェクト    │
│ JavaScript     │ (JIT最適化除く)    │                  │
├────────────────┼────────────────────┼──────────────────┤
│ Swift          │ 値型(struct, enum) │ 参照型(class)    │
│                │ プロトコル値型      │ ARC管理の         │
│                │                    │ クラスインスタンス │
└────────────────┴────────────────────┴──────────────────┘
```

### Go のエスケープ解析

```go
// Go: コンパイラが自動的にスタック/ヒープを決定
func example() *int {
    x := 42       // x は関数外に返されるためヒープに配置
    return &x     // エスケープ解析: x がエスケープする
}

func local() {
    x := 42       // x は関数内でのみ使用 → スタックに配置
    fmt.Println(x)
}

// go build -gcflags="-m" で確認可能
// ./main.go:3:2: moved to heap: x

// エスケープの典型的なパターン
func escapeExamples() {
    // 1. ポインタを返す → エスケープ
    createUser := func() *User {
        u := User{Name: "Gaku"}  // ヒープに配置
        return &u
    }

    // 2. インターフェースに代入 → エスケープ（の可能性）
    var w io.Writer
    buf := new(bytes.Buffer)  // ヒープに配置
    w = buf

    // 3. クロージャにキャプチャ → エスケープ
    x := 42
    fn := func() int {
        return x  // x がクロージャにキャプチャされる → ヒープ
    }

    // 4. スライスの append で容量不足 → ヒープに再確保
    s := make([]int, 0, 4)
    for i := 0; i < 10; i++ {
        s = append(s, i)  // 容量超過時にヒープに再確保
    }

    // 5. go ルーチンに渡す → エスケープ
    ch := make(chan *Data)
    go func() {
        d := &Data{}  // ヒープに配置
        ch <- d
    }()
}

// エスケープ解析の最適化テクニック
// 1. ポインタの代わりに値を返す（小さい構造体の場合）
func createPoint() Point {  // スタックにコピーで返される
    return Point{X: 1, Y: 2}
}

// 2. バッファを引数で受け取る
func readInto(buf []byte) int {  // buf の管理は呼び出し側
    // ...
    return n
}

// 3. sync.Pool でヒープ確保を再利用
var bufPool = sync.Pool{
    New: func() interface{} {
        return make([]byte, 4096)
    },
}

func processRequest() {
    buf := bufPool.Get().([]byte)
    defer bufPool.Put(buf)
    // buf を使用（ヒープ確保の再利用）
}
```

### C# の値型と参照型

```csharp
// C#: struct（値型）vs class（参照型）
struct PointStruct {  // スタック上に配置
    public double X;
    public double Y;
}

class PointClass {    // ヒープ上に配置
    public double X;
    public double Y;
}

void Example() {
    PointStruct ps = new PointStruct { X = 1, Y = 2 };  // スタック（16バイト）
    PointClass pc = new PointClass { X = 1, Y = 2 };    // ヒープ（16バイト + ヘッダ）
                                                          // 参照（8バイト）はスタック

    // 配列
    PointStruct[] arr1 = new PointStruct[100];  // 連続メモリ（100 × 16バイト）
    PointClass[] arr2 = new PointClass[100];    // 参照の配列 + 個別オブジェクト

    // Span<T>: ヒープ確保なしのスライス
    Span<int> span = stackalloc int[10];  // スタック上に確保
    span[0] = 42;

    // ReadOnlySpan<T>: 文字列のゼロコピースライシング
    ReadOnlySpan<char> slice = "Hello, World!".AsSpan(0, 5);
}

// ref struct: スタック専用の構造体（ヒープに置けない）
ref struct StackOnly {
    public Span<int> Data;
    // Span を含むためヒープに置けない
}
```

---

## 5. アライメントとパディング

```
アライメント: データの配置アドレスの制約
  - int (4バイト) は 4 の倍数のアドレスに配置
  - double (8バイト) は 8 の倍数のアドレスに配置
  - char (1バイト) はどこでもOK

  理由: CPU がアライメントされたアドレスから効率的に読み取れるため
        未アライメントアクセスは遅い or エラーになる

パディング: アライメントを満たすための空きバイト
```

```c
// C: 構造体のパディング
struct Bad {
    char a;     // 1バイト
                // 7バイトのパディング（次の double のアライメントのため）
    double b;   // 8バイト
    char c;     // 1バイト
                // 7バイトのパディング（構造体のアライメントのため）
};
// sizeof(struct Bad) = 24 バイト（実データ 10 バイトなのに！）

struct Good {
    double b;   // 8バイト
    char a;     // 1バイト
    char c;     // 1バイト
                // 6バイトのパディング
};
// sizeof(struct Good) = 16 バイト（8バイト節約！）

// パディング最適化のルール:
// 大きいフィールドを先に、小さいフィールドを後に配置する
```

```rust
// Rust: コンパイラが自動的に並べ替える（デフォルト）
struct AutoReorder {
    a: u8,     // Rust コンパイラがフィールドの順序を最適化する
    b: f64,
    c: u8,
}
// repr(Rust) がデフォルト → コンパイラが最適な順序に並べ替え可能

// C 互換レイアウトを強制する場合
#[repr(C)]
struct CLayout {
    a: u8,     // フィールドの順序が保持される
    b: f64,
    c: u8,
}

// サイズの確認
use std::mem;
println!("AutoReorder: {}", mem::size_of::<AutoReorder>()); // 16 (最適化)
println!("CLayout: {}", mem::size_of::<CLayout>());         // 24 (C互換)

// パック構造体（パディングなし、ただしアクセスが遅い可能性）
#[repr(packed)]
struct Packed {
    a: u8,
    b: f64,
    c: u8,
}
// size_of::<Packed>() = 10 バイト（パディングなし）
```

---

## 6. キャッシュフレンドリーなデータ設計

```
CPU キャッシュの階層:
  L1 キャッシュ: 32-64 KB、レイテンシ ~1ns（4サイクル）
  L2 キャッシュ: 256 KB - 1 MB、レイテンシ ~5ns
  L3 キャッシュ: 数MB - 数十MB、レイテンシ ~20ns
  メインメモリ: 数GB - 数TB、レイテンシ ~100ns

キャッシュライン: 通常 64 バイト
  メモリアクセスは 64 バイト単位で行われる
  → 近くのデータが自動的にキャッシュに載る

キャッシュフレンドリー = 連続したメモリを順番にアクセスすること
```

```rust
// 配列（連続メモリ）vs リンクリスト（散在メモリ）
// 配列は圧倒的にキャッシュフレンドリー

// AoS (Array of Structs) vs SoA (Struct of Arrays)

// AoS: 構造体の配列（一般的だがキャッシュ効率が悪い場合がある）
struct Particle {
    x: f32,
    y: f32,
    z: f32,
    vx: f32,
    vy: f32,
    vz: f32,
    mass: f32,
    charge: f32,
}
let particles: Vec<Particle> = vec![/* ... */];

// x 座標だけ処理する場合:
// [x,y,z,vx,vy,vz,m,c | x,y,z,vx,vy,vz,m,c | ...]
//  ↑                      ↑
// キャッシュラインに不要なデータも含まれる

// SoA: 配列の構造体（特定フィールドのバッチ処理に最適）
struct Particles {
    x: Vec<f32>,
    y: Vec<f32>,
    z: Vec<f32>,
    vx: Vec<f32>,
    vy: Vec<f32>,
    vz: Vec<f32>,
    mass: Vec<f32>,
    charge: Vec<f32>,
}

// x 座標だけ処理する場合:
// [x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x...]
// キャッシュラインが全て有効データ → 高速

// ベンチマーク例（概念的）
fn update_positions_aos(particles: &mut [Particle], dt: f32) {
    for p in particles.iter_mut() {
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        p.z += p.vz * dt;
    }
}

fn update_positions_soa(particles: &mut Particles, dt: f32) {
    for i in 0..particles.x.len() {
        particles.x[i] += particles.vx[i] * dt;
        particles.y[i] += particles.vy[i] * dt;
        particles.z[i] += particles.vz[i] * dt;
    }
}
// SoA 版は SIMD 最適化がかかりやすい
```

```go
// Go: スライスと map のメモリ特性
func cacheExample() {
    // スライス: 連続メモリ → キャッシュフレンドリー
    numbers := make([]int, 1000000)
    sum := 0
    for _, n := range numbers {
        sum += n  // シーケンシャルアクセス → 高速
    }

    // map: ハッシュテーブル → キャッシュ非フレンドリー
    m := make(map[int]int, 1000000)
    sum = 0
    for _, v := range m {
        sum += v  // ランダムアクセス → キャッシュミスが多い
    }

    // 構造体のポインタスライス vs 値スライス
    type Point struct {
        X, Y, Z float64
    }

    // 値スライス: 連続メモリ（推奨）
    points := make([]Point, 1000)

    // ポインタスライス: ポインタは連続だがデータが散在
    ptrs := make([]*Point, 1000)
    for i := range ptrs {
        ptrs[i] = &Point{} // 各 Point は別のヒープ位置
    }
}
```

---

## 7. スタックオーバーフロー

```
スタックのサイズは制限されている（通常1〜8MB）

原因:
  1. 深すぎる再帰
  2. スタック上の巨大な配列
  3. 相互再帰
  4. 無限再帰（バグ）

対策:
  - 再帰の代わりにループを使う
  - 末尾再帰最適化（TCO）がある言語を使う
  - 大きなデータはヒープに置く
  - スタックサイズを増やす（ulimit -s / CreateThread）
```

```rust
// スタックオーバーフローの例
fn infinite_recursion(n: i32) -> i32 {
    infinite_recursion(n + 1)  // 終了条件なし → スタックオーバーフロー
}

// 巨大なスタック確保
fn big_stack() {
    let arr = [0u8; 10_000_000];  // 10MB → スタックオーバーフロー
    // 解決: let arr = vec![0u8; 10_000_000]; // ヒープに配置
}

// 末尾再帰 vs 通常の再帰
fn factorial_recursive(n: u64) -> u64 {
    if n <= 1 { 1 }
    else { n * factorial_recursive(n - 1) }  // スタックフレームが積み上がる
}

fn factorial_tail(n: u64, acc: u64) -> u64 {
    if n <= 1 { acc }
    else { factorial_tail(n - 1, n * acc) }  // 末尾位置の再帰呼び出し
}
// ※ Rust は TCO を保証しない。ループに書き換えるのが安全

fn factorial_loop(n: u64) -> u64 {
    let mut result = 1u64;
    for i in 2..=n {
        result *= i;
    }
    result
}
```

```haskell
-- Haskell: 末尾再帰と遅延評価の注意点
-- 厳密評価版（効率的）
factorial :: Integer -> Integer
factorial n = go n 1
  where
    go 0 acc = acc
    go n acc = go (n - 1) (acc * n)  -- $! で厳密評価を強制すべき

-- foldl vs foldl'
-- foldl はサンクを積み上げる（メモリ消費）
-- foldl' は厳密に評価する（推奨）
sum' :: [Int] -> Int
sum' = foldl' (+) 0  -- Data.List.foldl'
```

### 各言語のスタックサイズ

```
┌──────────────┬─────────────────────┬─────────────────────┐
│ 言語          │ デフォルトスタック    │ 変更方法              │
├──────────────┼─────────────────────┼─────────────────────┤
│ C/C++ (Linux)│ 8 MB                │ ulimit -s / pthread │
│ C/C++ (macOS)│ 8 MB (main)        │ ulimit -s            │
│ Rust         │ 8 MB (main)        │ std::thread::Builder │
│              │ 2 MB (spawn)       │ .stack_size(bytes)   │
│ Java         │ 512 KB - 1 MB      │ -Xss オプション       │
│ Go           │ 2-8 KB (初期)      │ 自動成長（制限なし）   │
│ Python       │ 1000フレーム(制限)  │ sys.setrecursionlimit│
│ JavaScript   │ エンジン依存        │ --stack-size (Node)  │
│ Swift        │ 8 MB (main)        │ Thread API           │
│ C#           │ 1 MB               │ Thread コンストラクタ  │
└──────────────┴─────────────────────┴─────────────────────┘
```

---

## 8. メモリデバッグツール

```
Valgrind (C/C++):
  - メモリリーク検出
  - 未初期化メモリの使用検出
  - バッファオーバーフロー検出
  - 使い方: valgrind --leak-check=full ./program

AddressSanitizer (ASan):
  - コンパイル時計装（GCC/Clang）
  - バッファオーバーフロー、Use-After-Free 検出
  - 使い方: gcc -fsanitize=address program.c

Miri (Rust):
  - 未定義動作の検出
  - メモリ安全性の動的検証
  - 使い方: cargo +nightly miri run

pprof (Go):
  - メモリプロファイリング
  - ヒープ使用量の可視化
  - 使い方: import _ "net/http/pprof"

heaptrack (C/C++):
  - ヒープメモリの使用履歴を記録
  - 確保元のスタックトレースを可視化

Chrome DevTools (JavaScript):
  - ヒープスナップショット
  - アロケーションタイムライン
  - メモリリーク検出

tracemalloc (Python):
  - メモリ確保のトレース
  - 確保元のコード位置を特定
```

```bash
# Valgrind の使用例
valgrind --leak-check=full --show-leak-kinds=all ./my_program

# AddressSanitizer
gcc -fsanitize=address -g -o my_program my_program.c
./my_program

# Rust Miri
cargo +nightly miri run

# Go pprof
go tool pprof http://localhost:6060/debug/pprof/heap
```

---

## 9. Arena アロケータとバンプアロケータ

```rust
// Arena アロケータ: まとめて確保・まとめて解放
// パーサーやコンパイラで頻繁に使われる

// 概念的な実装
struct Arena {
    chunks: Vec<Vec<u8>>,
    current: Vec<u8>,
    offset: usize,
}

impl Arena {
    fn new(chunk_size: usize) -> Self {
        Arena {
            chunks: Vec::new(),
            current: vec![0u8; chunk_size],
            offset: 0,
        }
    }

    fn alloc<T>(&mut self, value: T) -> &mut T {
        let size = std::mem::size_of::<T>();
        let align = std::mem::align_of::<T>();

        // アライメント調整
        let offset = (self.offset + align - 1) & !(align - 1);

        if offset + size > self.current.len() {
            // 新しいチャンクを確保
            let old = std::mem::replace(
                &mut self.current,
                vec![0u8; self.current.len()],
            );
            self.chunks.push(old);
            self.offset = 0;
        }

        let ptr = &mut self.current[self.offset] as *mut u8 as *mut T;
        unsafe {
            ptr.write(value);
            self.offset += size;
            &mut *ptr
        }
    }
}
// Arena が drop されると全メモリが一括解放
// → 個別の free/drop が不要で高速

// 実用例: bumpalo クレート
// use bumpalo::Bump;
// let bump = Bump::new();
// let x = bump.alloc(42);
// let s = bump.alloc_str("hello");
// // bump がスコープを抜けると全て解放
```

```go
// Go: sync.Pool によるオブジェクト再利用
var nodePool = sync.Pool{
    New: func() interface{} {
        return new(ASTNode)
    },
}

func parseExpression(tokens []Token) *ASTNode {
    node := nodePool.Get().(*ASTNode)
    // node を使用
    return node
}

func freeNode(node *ASTNode) {
    *node = ASTNode{} // ゼロ値にリセット
    nodePool.Put(node)
}
```


---

## トラブルシューティング

### よくあるエラーと解決策

| エラー | 原因 | 解決策 |
|--------|------|--------|
| 初期化エラー | 設定ファイルの不備 | 設定ファイルのパスと形式を確認 |
| タイムアウト | ネットワーク遅延/リソース不足 | タイムアウト値の調整、リトライ処理の追加 |
| メモリ不足 | データ量の増大 | バッチ処理の導入、ページネーションの実装 |
| 権限エラー | アクセス権限の不足 | 実行ユーザーの権限確認、設定の見直し |
| データ不整合 | 並行処理の競合 | ロック機構の導入、トランザクション管理 |

### デバッグの手順

1. **エラーメッセージの確認**: スタックトレースを読み、発生箇所を特定する
2. **再現手順の確立**: 最小限のコードでエラーを再現する
3. **仮説の立案**: 考えられる原因をリストアップする
4. **段階的な検証**: ログ出力やデバッガを使って仮説を検証する
5. **修正と回帰テスト**: 修正後、関連する箇所のテストも実行する

```python
# デバッグ用ユーティリティ
import logging
import traceback
from functools import wraps

# ロガーの設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def debug_decorator(func):
    """関数の入出力をログ出力するデコレータ"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"呼び出し: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"戻り値: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"例外発生: {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

@debug_decorator
def process_data(items):
    """データ処理（デバッグ対象）"""
    if not items:
        raise ValueError("空のデータ")
    return [item * 2 for item in items]
```

### パフォーマンス問題の診断

パフォーマンス問題が発生した場合の診断手順:

1. **ボトルネックの特定**: プロファイリングツールで計測
2. **メモリ使用量の確認**: メモリリークの有無をチェック
3. **I/O待ちの確認**: ディスクやネットワークI/Oの状況を確認
4. **同時接続数の確認**: コネクションプールの状態を確認

| 問題の種類 | 診断ツール | 対策 |
|-----------|-----------|------|
| CPU負荷 | cProfile, py-spy | アルゴリズム改善、並列化 |
| メモリリーク | tracemalloc, objgraph | 参照の適切な解放 |
| I/Oボトルネック | strace, iostat | 非同期I/O、キャッシュ |
| DB遅延 | EXPLAIN, slow query log | インデックス、クエリ最適化 |

---

## 設計判断ガイド

### 選択基準マトリクス

技術選択を行う際の判断基準を以下にまとめます。

| 判断基準 | 重視する場合 | 妥協できる場合 |
|---------|------------|-------------|
| パフォーマンス | リアルタイム処理、大規模データ | 管理画面、バッチ処理 |
| 保守性 | 長期運用、チーム開発 | プロトタイプ、短期プロジェクト |
| スケーラビリティ | 成長が見込まれるサービス | 社内ツール、固定ユーザー |
| セキュリティ | 個人情報、金融データ | 公開データ、社内利用 |
| 開発速度 | MVP、市場投入スピード | 品質重視、ミッションクリティカル |

### アーキテクチャパターンの選択

```
┌─────────────────────────────────────────────────┐
│              アーキテクチャ選択フロー              │
├─────────────────────────────────────────────────┤
│                                                 │
│  ① チーム規模は？                                │
│    ├─ 小規模（1-5人）→ モノリス                   │
│    └─ 大規模（10人+）→ ②へ                       │
│                                                 │
│  ② デプロイ頻度は？                               │
│    ├─ 週1回以下 → モノリス + モジュール分割         │
│    └─ 毎日/複数回 → ③へ                          │
│                                                 │
│  ③ チーム間の独立性は？                            │
│    ├─ 高い → マイクロサービス                      │
│    └─ 中程度 → モジュラーモノリス                   │
│                                                 │
└─────────────────────────────────────────────────┘
```

### トレードオフの分析

技術的な判断には必ずトレードオフが伴います。以下の観点で分析を行いましょう:

**1. 短期 vs 長期のコスト**
- 短期的に速い方法が長期的には技術的負債になることがある
- 逆に、過剰な設計は短期的なコストが高く、プロジェクトの遅延を招く

**2. 一貫性 vs 柔軟性**
- 統一された技術スタックは学習コストが低い
- 多様な技術の採用は適材適所が可能だが、運用コストが増加

**3. 抽象化のレベル**
- 高い抽象化は再利用性が高いが、デバッグが困難になる場合がある
- 低い抽象化は直感的だが、コードの重複が発生しやすい

```python
# 設計判断の記録テンプレート
class ArchitectureDecisionRecord:
    """ADR (Architecture Decision Record) の作成"""

    def __init__(self, title: str):
        self.title = title
        self.context = ""
        self.decision = ""
        self.consequences = []
        self.alternatives = []

    def set_context(self, context: str):
        """背景と課題の記述"""
        self.context = context
        return self

    def set_decision(self, decision: str):
        """決定内容の記述"""
        self.decision = decision
        return self

    def add_consequence(self, consequence: str, positive: bool = True):
        """結果の追加"""
        self.consequences.append({
            'description': consequence,
            'type': 'positive' if positive else 'negative'
        })
        return self

    def add_alternative(self, name: str, reason_rejected: str):
        """却下した代替案の追加"""
        self.alternatives.append({
            'name': name,
            'reason_rejected': reason_rejected
        })
        return self

    def to_markdown(self) -> str:
        """Markdown形式で出力"""
        md = f"# ADR: {self.title}\n\n"
        md += f"## 背景\n{self.context}\n\n"
        md += f"## 決定\n{self.decision}\n\n"
        md += "## 結果\n"
        for c in self.consequences:
            icon = "✅" if c['type'] == 'positive' else "⚠️"
            md += f"- {icon} {c['description']}\n"
        md += "\n## 却下した代替案\n"
        for a in self.alternatives:
            md += f"- **{a['name']}**: {a['reason_rejected']}\n"
        return md
```

---

## チーム開発での活用

### コードレビューのチェックリスト

このトピックに関連するコードレビューで確認すべきポイント:

- [ ] 命名規則が一貫しているか
- [ ] エラーハンドリングが適切か
- [ ] テストカバレッジは十分か
- [ ] パフォーマンスへの影響はないか
- [ ] セキュリティ上の問題はないか
- [ ] ドキュメントは更新されているか

### ナレッジ共有のベストプラクティス

| 方法 | 頻度 | 対象 | 効果 |
|------|------|------|------|
| ペアプログラミング | 随時 | 複雑なタスク | 即時のフィードバック |
| テックトーク | 週1回 | チーム全体 | 知識の水平展開 |
| ADR (設計記録) | 都度 | 将来のメンバー | 意思決定の透明性 |
| 振り返り | 2週間ごと | チーム全体 | 継続的改善 |
| モブプログラミング | 月1回 | 重要な設計 | 合意形成 |

### 技術的負債の管理

```
優先度マトリクス:

        影響度 高
          │
    ┌─────┼─────┐
    │ 計画 │ 即座 │
    │ 的に │ に   │
    │ 対応 │ 対応 │
    ├─────┼─────┤
    │ 記録 │ 次の │
    │ のみ │ Sprint│
    │     │ で   │
    └─────┼─────┘
          │
        影響度 低
    発生頻度 低  発生頻度 高
```

---

## セキュリティの考慮事項

### 一般的な脆弱性と対策

| 脆弱性 | リスクレベル | 対策 | 検出方法 |
|--------|------------|------|---------|
| インジェクション攻撃 | 高 | 入力値のバリデーション・パラメータ化クエリ | SAST/DAST |
| 認証の不備 | 高 | 多要素認証・セッション管理の強化 | ペネトレーションテスト |
| 機密データの露出 | 高 | 暗号化・アクセス制御 | セキュリティ監査 |
| 設定の不備 | 中 | セキュリティヘッダー・最小権限の原則 | 構成スキャン |
| ログの不足 | 中 | 構造化ログ・監査証跡 | ログ分析 |

### セキュアコーディングのベストプラクティス

```python
# セキュアコーディング例
import hashlib
import secrets
import hmac
from typing import Optional

class SecurityUtils:
    """セキュリティユーティリティ"""

    @staticmethod
    def generate_token(length: int = 32) -> str:
        """暗号学的に安全なトークン生成"""
        return secrets.token_urlsafe(length)

    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> tuple:
        """パスワードのハッシュ化"""
        if salt is None:
            salt = secrets.token_hex(16)
        hashed = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            iterations=100000
        )
        return hashed.hex(), salt

    @staticmethod
    def verify_password(password: str, hashed: str, salt: str) -> bool:
        """パスワードの検証"""
        new_hash, _ = SecurityUtils.hash_password(password, salt)
        return hmac.compare_digest(new_hash, hashed)

    @staticmethod
    def sanitize_input(value: str) -> str:
        """入力値のサニタイズ"""
        dangerous_chars = ['<', '>', '"', "'", '&', '\\']
        result = value
        for char in dangerous_chars:
            result = result.replace(char, '')
        return result.strip()

# 使用例
token = SecurityUtils.generate_token()
hashed, salt = SecurityUtils.hash_password("my_password")
is_valid = SecurityUtils.verify_password("my_password", hashed, salt)
```

### セキュリティチェックリスト

- [ ] 全ての入力値がバリデーションされている
- [ ] 機密情報がログに出力されていない
- [ ] HTTPS が強制されている
- [ ] CORS ポリシーが適切に設定されている
- [ ] 依存パッケージの脆弱性スキャンが実施されている
- [ ] エラーメッセージに内部情報が含まれていない

---

## マイグレーションガイド

### バージョンアップ時の注意点

| バージョン | 主な変更点 | 移行作業 | 影響範囲 |
|-----------|-----------|---------|---------|
| v1.x → v2.x | API設計の刷新 | エンドポイント変更 | 全クライアント |
| v2.x → v3.x | 認証方式の変更 | トークン形式更新 | 認証関連 |
| v3.x → v4.x | データモデル変更 | マイグレーションスクリプト実行 | DB関連 |

### 段階的移行の手順

```python
# マイグレーションスクリプトのテンプレート
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Callable

logger = logging.getLogger(__name__)

class MigrationRunner:
    """段階的マイグレーション実行エンジン"""

    def __init__(self, migration_dir: str):
        self.migration_dir = Path(migration_dir)
        self.migrations: List[Dict] = []
        self.completed: List[str] = []

    def register(self, version: str, description: str,
                 up: Callable, down: Callable):
        """マイグレーションの登録"""
        self.migrations.append({
            'version': version,
            'description': description,
            'up': up,
            'down': down,
            'registered_at': datetime.now().isoformat()
        })

    def run_up(self, target_version: str = None):
        """マイグレーションの実行（アップグレード）"""
        for migration in self.migrations:
            if migration['version'] in self.completed:
                continue
            logger.info(f"実行中: {migration['version']} - "
                       f"{migration['description']}")
            try:
                migration['up']()
                self.completed.append(migration['version'])
                logger.info(f"完了: {migration['version']}")
            except Exception as e:
                logger.error(f"失敗: {migration['version']}: {e}")
                raise
            if target_version and migration['version'] == target_version:
                break

    def run_down(self, target_version: str):
        """マイグレーションのロールバック"""
        for migration in reversed(self.migrations):
            if migration['version'] not in self.completed:
                continue
            if migration['version'] == target_version:
                break
            logger.info(f"ロールバック: {migration['version']}")
            migration['down']()
            self.completed.remove(migration['version'])

    def status(self) -> Dict:
        """マイグレーション状態の確認"""
        return {
            'total': len(self.migrations),
            'completed': len(self.completed),
            'pending': len(self.migrations) - len(self.completed),
            'versions': {
                m['version']: 'completed'
                if m['version'] in self.completed else 'pending'
                for m in self.migrations
            }
        }
```

### ロールバック計画

移行作業には必ずロールバック計画を準備してください:

1. **データのバックアップ**: 移行前に完全バックアップを取得
2. **テスト環境での検証**: 本番と同等の環境で事前検証
3. **段階的なロールアウト**: カナリアリリースで段階的に展開
4. **監視の強化**: 移行中はメトリクスの監視間隔を短縮
5. **判断基準の明確化**: ロールバックを判断する基準を事前に定義

---

## 用語集

| 用語 | 英語表記 | 説明 |
|------|---------|------|
| 抽象化 | Abstraction | 複雑な実装の詳細を隠し、本質的なインターフェースのみを公開すること |
| カプセル化 | Encapsulation | データと操作を一つの単位にまとめ、外部からのアクセスを制御すること |
| 凝集度 | Cohesion | モジュール内の要素がどの程度関連しているかの指標 |
| 結合度 | Coupling | モジュール間の依存関係の度合い |
| リファクタリング | Refactoring | 外部の振る舞いを変えずにコードの内部構造を改善すること |
| テスト駆動開発 | TDD (Test-Driven Development) | テストを先に書いてから実装するアプローチ |
| 継続的インテグレーション | CI (Continuous Integration) | コードの変更を頻繁に統合し、自動テストで検証するプラクティス |
| 継続的デリバリー | CD (Continuous Delivery) | いつでもリリース可能な状態を維持するプラクティス |
| 技術的負債 | Technical Debt | 短期的な解決策を選んだことで将来的に発生する追加作業 |
| ドメイン駆動設計 | DDD (Domain-Driven Design) | ビジネスドメインの知識に基づいてソフトウェアを設計するアプローチ |
| マイクロサービス | Microservices | アプリケーションを小さな独立したサービスの集合として構築するアーキテクチャ |
| サーキットブレーカー | Circuit Breaker | 障害の連鎖を防ぐための設計パターン |
| イベント駆動 | Event-Driven | イベントの発生と処理に基づくアーキテクチャパターン |
| 冪等性 | Idempotency | 同じ操作を複数回実行しても結果が変わらない性質 |
| オブザーバビリティ | Observability | システムの内部状態を外部から観測可能にする能力 |
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

| 特性 | スタック | ヒープ |
|------|---------|--------|
| 速度 | 非常に高速（ポインタ加減算） | 低速（アロケーション） |
| 管理 | 自動（LIFO） | 手動/GC |
| サイズ | 制限あり（1-8MB） | 大きい（物理メモリまで） |
| 寿命 | 関数スコープ | 任意（参照がある限り） |
| 用途 | ローカル変数・引数 | 動的データ・オブジェクト |
| 断片化 | なし | あり |
| スレッド安全 | スレッド独立 | 共有（同期が必要） |
| キャッシュ | フレンドリー | ミスが起きやすい |

| デバッグツール | 対象言語 | 検出できる問題 |
|-------------|---------|-------------|
| Valgrind | C/C++ | リーク、未初期化、オーバーフロー |
| ASan | C/C++/Rust | バッファオーバーフロー、UAF |
| Miri | Rust | 未定義動作、安全性違反 |
| pprof | Go | メモリプロファイル |
| DevTools | JavaScript | ヒープスナップショット |

---

## 次に読むべきガイド

---

## 10. 実務でのメモリ最適化パターン

### 文字列のインターニング

```
文字列インターニング: 同じ内容の文字列を1つだけメモリに保持する

  Python: 短い文字列は自動的にインターン
    a = "hello"
    b = "hello"
    a is b  # True（同じオブジェクト）

  Java: String Pool
    String s1 = "hello";  // String Pool に配置
    String s2 = "hello";  // 同じオブジェクトを参照
    s1 == s2  // true

    String s3 = new String("hello");  // 新しいオブジェクト
    s1 == s3  // false
    s1.equals(s3)  // true

  用途: シンボルテーブル、設定キー、タグ名など
  → 同じ文字列が大量に出現する場合にメモリ節約
```

### メモリマッピングとゼロコピー

```rust
// Rust: メモリマップドファイル（memmap2 クレート）
use memmap2::Mmap;
use std::fs::File;

fn read_large_file(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };

    // mmap は &[u8] として扱える
    // ファイル全体をメモリにコピーせずにアクセス
    let first_100 = &mmap[..100];
    println!("First 100 bytes: {:?}", first_100);

    // OS がページ単位でオンデマンドにロード
    // 大きなファイルでも効率的
    Ok(())
}
```

```go
// Go: bytes.Buffer の効率的な使い方
func efficientStringBuilding() string {
    var buf bytes.Buffer
    buf.Grow(1024) // 事前にサイズを予約（再確保を減らす）

    for i := 0; i < 100; i++ {
        fmt.Fprintf(&buf, "Line %d\n", i)
    }
    return buf.String()
}

// strings.Builder（Go 1.10+）
func efficientStringBuilder() string {
    var sb strings.Builder
    sb.Grow(1024)

    for i := 0; i < 100; i++ {
        sb.WriteString("Item ")
        sb.WriteString(strconv.Itoa(i))
        sb.WriteByte('\n')
    }
    return sb.String()
}
```

### オブジェクトプーリング

```java
// Java: オブジェクトプール
public class ObjectPool<T> {
    private final Queue<T> pool;
    private final Supplier<T> factory;
    private final Consumer<T> reset;
    private final int maxSize;

    public ObjectPool(Supplier<T> factory, Consumer<T> reset, int maxSize) {
        this.pool = new ConcurrentLinkedQueue<>();
        this.factory = factory;
        this.reset = reset;
        this.maxSize = maxSize;
    }

    public T acquire() {
        T obj = pool.poll();
        return obj != null ? obj : factory.get();
    }

    public void release(T obj) {
        if (pool.size() < maxSize) {
            reset.accept(obj);
            pool.offer(obj);
        }
        // maxSize を超えた場合は GC に任せる
    }
}

// 使用例: StringBuilder のプール
ObjectPool<StringBuilder> sbPool = new ObjectPool<>(
    StringBuilder::new,
    sb -> sb.setLength(0),
    100
);

StringBuilder sb = sbPool.acquire();
try {
    sb.append("Hello");
    sb.append(" World");
    String result = sb.toString();
} finally {
    sbPool.release(sb);
}
```

---

## 実践演習

### 演習1: [基礎] -- メモリレイアウトの観察
C または Rust で構造体のサイズとアライメントを出力するプログラムを書き、パディングの影響を確認する。フィールドの順序を変えてサイズがどう変わるか観察する。

### 演習2: [応用] -- バンプアロケータの実装
Rust で簡易的なバンプアロケータを実装する。alloc, reset メソッドを持ち、固定サイズのバッファから順番にメモリを切り出す。

### 演習3: [応用] -- エスケープ解析の体験
Go プログラムを `go build -gcflags="-m"` でコンパイルし、どの変数がヒープにエスケープするか観察する。エスケープを減らすリファクタリングを行う。

### 演習4: [発展] -- キャッシュ性能の計測
AoS と SoA の両方のレイアウトで粒子シミュレーションを実装し、ベンチマークで性能差を計測する。

---

## 参考文献
1. Bryant, R. & O'Hallaron, D. "Computer Systems: A Programmer's Perspective." 3rd Ed, Ch.9, 2015.
2. Klabnik, S. & Nichols, C. "The Rust Programming Language." Ch.4, 2023.
3. Drepper, U. "What Every Programmer Should Know About Memory." 2007.
4. Intel. "Intel 64 and IA-32 Architectures Optimization Reference Manual." 2023.
5. Fog, A. "Optimizing Software in C++." 2023.
6. Boehm, H. "Bounding Space Usage of Conservative Garbage Collectors." POPL, 2002.
7. Emery Berger et al. "Reconsidering Custom Memory Allocation." OOPSLA, 2002.
8. Go Team. "Go Memory Model." go.dev/ref/mem.
9. Oracle. "JVM Specification: Run-Time Data Areas." Ch.2.5.
10. Apple. "Swift Memory Layout." developer.apple.com.


---

## 補足: さらなる学習のために

### このトピックの発展的な側面

本ガイドで扱った内容は基礎的な部分をカバーしていますが、さらに深く学ぶための方向性をいくつか紹介します。

#### 理論的な深掘り

このトピックの背景には、長年にわたる研究と実践の蓄積があります。基本的な概念を理解した上で、以下の方向性で学習を深めることをお勧めします:

1. **歴史的な経緯の理解**: 現在のベストプラクティスがなぜそうなったのかを理解することで、より深い洞察が得られます
2. **関連分野との接点**: 隣接する分野の知識を取り入れることで、視野が広がり、より創造的なアプローチが可能になります
3. **最新のトレンドの把握**: 技術や手法は常に進化しています。定期的に最新の動向をチェックしましょう

#### 実践的なスキル向上

理論的な知識を実践に結びつけるために:

- **定期的な練習**: 週に数回、意識的に実践する時間を確保する
- **フィードバックループ**: 自分の成果を客観的に評価し、改善点を見つける
- **記録と振り返り**: 学習の過程を記録し、定期的に振り返る
- **コミュニティへの参加**: 同じ分野に興味を持つ人々と交流し、知見を共有する
- **メンターの活用**: 経験者からのアドバイスは、独学では得られない視点を提供してくれます


### 継続的な成長のために

学習は一度で完了するものではなく、継続的なプロセスです。以下のサイクルを意識して、着実にスキルを向上させていきましょう:

1. **学ぶ（Learn）**: 新しい概念や技術を理解する
2. **試す（Try）**: 実際に手を動かして実践する
3. **振り返る（Reflect）**: 成果と課題を分析する
4. **共有する（Share）**: 学んだことを他者と共有する
5. **改善する（Improve）**: フィードバックを基に改善する

このサイクルを繰り返すことで、単なる知識の蓄積ではなく、実践的なスキルとして定着させることができます。また、共有のステップを含めることで、コミュニティへの貢献にもつながります。

### 学習記録の重要性

学習の効果を最大化するために、以下の記録をつけることをお勧めします:

- **日付と学習内容**: 何をいつ学んだかを記録
- **理解度の自己評価**: 1-5段階で理解度を評価
- **疑問点**: わからなかったことや深掘りしたい点
- **実践メモ**: 実際に試してみた結果と気づき
- **関連リソース**: 参考になった資料やリンク

これらの記録は、後から振り返る際に非常に有用です。特に、疑問点を記録しておくことで、後の学習で自然と解決されることが多くあります。

また、学習記録を公開することで（ブログ、SNS等）、同じ分野を学ぶ仲間とつながるきっかけにもなります。アウトプットすることで理解が深まり、フィードバックを得られるという好循環が生まれます。