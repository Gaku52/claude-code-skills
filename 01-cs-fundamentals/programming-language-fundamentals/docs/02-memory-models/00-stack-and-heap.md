# スタックとヒープ

> プログラムのメモリは「スタック」（高速・自動管理）と「ヒープ」（柔軟・手動/GC管理）に分かれる。この区別を理解することが、パフォーマンスとメモリ安全性の基盤となる。

## この章で学ぶこと

- [ ] スタックとヒープの違いと特性を理解する
- [ ] 各言語でのメモリ配置の違いを把握する
- [ ] メモリレイアウトを意識した効率的なコードが書ける
- [ ] アライメントとパディングの影響を理解する
- [ ] キャッシュフレンドリーなデータ設計ができる
- [ ] メモリ関連のバグ（リーク、バッファオーバーフロー等）を防ぐ知識を持つ

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
    std::function<int(int)> f = [](int x) { return x * 2; };
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
→ [[01-garbage-collection.md]] -- ガベージコレクション

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
