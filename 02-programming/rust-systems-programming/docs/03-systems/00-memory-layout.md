# メモリレイアウト — スタック/ヒープ、repr

> Rust のデータ型がメモリ上でどう配置されるかを理解し、repr 属性やアライメント制御で低レベル最適化を行う技術を習得する

## この章で学ぶこと

1. **スタックとヒープ** — 各領域の特性、割当コスト、所有権との関係
2. **型のメモリレイアウト** — サイズ、アライメント、パディング、repr 属性
3. **スマートポインタの内部構造** — Box, Vec, String, Arc のメモリ配置
4. **高度なメモリ制御** — アロケータAPI、メモリマップドI/O、キャッシュ最適化
5. **実践的なメモリプロファイリング** — ツールと手法

---

## 1. プロセスメモリマップ

```
┌──────────────────── プロセスメモリ空間 ────────────────┐
│  高位アドレス                                          │
│  ┌─────────────────────────────────────────┐          │
│  │           Stack (スタック)                │ ↓ 成長   │
│  │  - ローカル変数、関数引数                  │          │
│  │  - 固定サイズ (通常 8MB)                  │          │
│  │  - LIFO (超高速割当/解放)                 │          │
│  └─────────────────────────────────────────┘          │
│                        ↕ (未使用空間)                   │
│  ┌─────────────────────────────────────────┐          │
│  │           Heap (ヒープ)                   │ ↑ 成長   │
│  │  - 動的割当 (Box, Vec, String)           │          │
│  │  - OS アロケータ経由                      │          │
│  │  - 任意のサイズ・ライフタイム              │          │
│  └─────────────────────────────────────────┘          │
│  ┌─────────────────────────────────────────┐          │
│  │  BSS (未初期化静的変数)                    │          │
│  ├─────────────────────────────────────────┤          │
│  │  Data (初期化済み静的変数)                  │          │
│  ├─────────────────────────────────────────┤          │
│  │  Text (実行コード) [読み取り専用]           │          │
│  └─────────────────────────────────────────┘          │
│  低位アドレス                                          │
└────────────────────────────────────────────────────────┘
```

### 各セグメントの詳細

| セグメント | 内容 | 権限 | サイズ |
|---|---|---|---|
| Text | 機械語命令 | 読み取り+実行 | 固定 |
| Data | `static mut X: i32 = 42;` 等 | 読み書き | 固定 |
| BSS | `static mut Y: i32 = 0;` 等 | 読み書き | 固定 |
| Heap | `Box::new()`, `Vec::new()` | 読み書き | 可変(上方成長) |
| Stack | ローカル変数、戻りアドレス | 読み書き | 可変(下方成長) |

### コード例: メモリアドレスの観察

```rust
use std::mem;

static GLOBAL: i32 = 100;
static mut GLOBAL_MUT: i32 = 200;

fn main() {
    // スタック上の変数
    let stack_var: i32 = 42;
    let stack_arr: [u8; 16] = [0; 16];

    // ヒープ上の変数
    let heap_var = Box::new(42i32);
    let heap_vec: Vec<u8> = vec![0; 16];

    println!("=== メモリアドレス観察 ===");
    println!("Text セグメント:");
    println!("  main 関数:    {:p}", main as *const ());

    println!("Data/BSS セグメント:");
    println!("  GLOBAL:       {:p}", &GLOBAL);
    unsafe {
        println!("  GLOBAL_MUT:   {:p}", &GLOBAL_MUT);
    }

    println!("スタック:");
    println!("  stack_var:    {:p}", &stack_var);
    println!("  stack_arr:    {:p}", &stack_arr);

    println!("ヒープ:");
    println!("  heap_var:     {:p}", &*heap_var);
    println!("  heap_vec[0]:  {:p}", heap_vec.as_ptr());

    // スタックのポインタ値 > ヒープのポインタ値 (一般的)
    let stack_addr = &stack_var as *const i32 as usize;
    let heap_addr = &*heap_var as *const i32 as usize;
    println!("\nスタック (0x{:x}) > ヒープ (0x{:x}): {}",
        stack_addr, heap_addr, stack_addr > heap_addr);
}
```

---

## 2. スタックとヒープの比較

### コード例1: スタック vs ヒープ割当

```rust
use std::mem;

fn main() {
    // スタック割当: コンパイル時にサイズ確定
    let x: i32 = 42;              // 4 bytes on stack
    let arr: [u8; 1024] = [0; 1024]; // 1024 bytes on stack
    let point: (f64, f64) = (1.0, 2.0); // 16 bytes on stack

    // ヒープ割当: 実行時にサイズ決定可能
    let boxed: Box<i32> = Box::new(42);    // ポインタ(8b) on stack, 4b on heap
    let vec: Vec<u8> = vec![0; 1024];      // 24b on stack, 1024b on heap
    let string: String = "hello".to_string(); // 24b on stack, 5b on heap

    println!("--- スタック上のサイズ ---");
    println!("i32:        {} bytes", mem::size_of::<i32>());
    println!("[u8; 1024]: {} bytes", mem::size_of::<[u8; 1024]>());
    println!("(f64, f64): {} bytes", mem::size_of::<(f64, f64)>());
    println!("Box<i32>:   {} bytes", mem::size_of::<Box<i32>>());
    println!("Vec<u8>:    {} bytes", mem::size_of::<Vec<u8>>());
    println!("String:     {} bytes", mem::size_of::<String>());
    // Box<i32>:   8 bytes  (ポインタのみ)
    // Vec<u8>:    24 bytes (ptr + len + capacity)
    // String:     24 bytes (ptr + len + capacity)
}
```

### スマートポインタのメモリ配置

```
┌──────── Stack ────────┐     ┌──────── Heap ────────┐
│                        │     │                      │
│  Box<i32>              │     │                      │
│  ┌──────────┐          │     │  ┌────┐             │
│  │ ptr ─────┼──────────┼─────┼→ │ 42 │             │
│  └──────────┘          │     │  └────┘             │
│  8 bytes               │     │  4 bytes             │
│                        │     │                      │
│  Vec<u8> "abc"         │     │                      │
│  ┌──────────┐          │     │  ┌───┬───┬───┬───┐  │
│  │ ptr ─────┼──────────┼─────┼→ │ a │ b │ c │   │  │
│  ├──────────┤          │     │  └───┴───┴───┴───┘  │
│  │ len: 3   │          │     │  capacity: 4         │
│  ├──────────┤          │     │                      │
│  │ cap: 4   │          │     │                      │
│  └──────────┘          │     │                      │
│  24 bytes              │     │                      │
│                        │     │                      │
│  Arc<Data>             │     │  ┌──────────────┐   │
│  ┌──────────┐          │     │  │ strong: 2    │   │
│  │ ptr ─────┼──────────┼─────┼→ │ weak: 1      │   │
│  └──────────┘          │     │  │ data: Data   │   │
│  8 bytes               │     │  └──────────────┘   │
└────────────────────────┘     └──────────────────────┘
```

### コード例: スタックオーバーフローの検出

```rust
/// スタックサイズの限界を確認する例
fn recursive_stack_usage(depth: usize) {
    // 各呼び出しで約 1KB のスタックを消費
    let _buffer = [0u8; 1024];
    if depth > 0 {
        recursive_stack_usage(depth - 1);
    }
}

fn main() {
    // デフォルトスタックサイズ (8MB) では約 8000 回の再帰が限界
    // スタックオーバーフロー → プロセスが SIGSEGV で終了

    // 安全策: スレッドビルダーでスタックサイズを指定
    let builder = std::thread::Builder::new()
        .name("large-stack".into())
        .stack_size(32 * 1024 * 1024); // 32MB

    let handle = builder.spawn(|| {
        recursive_stack_usage(30000); // 32MB なら余裕
        println!("再帰完了");
    }).unwrap();

    handle.join().unwrap();

    // stacker クレートで動的スタック拡張も可能
    // stacker::maybe_grow(32 * 1024, 1024 * 1024, || { ... });
}
```

### コード例: Vec の成長戦略とメモリ再割当

```rust
fn main() {
    let mut v: Vec<i32> = Vec::new();

    println!("=== Vec の成長戦略 ===");
    println!("{:>5} {:>10} {:>10} {:>18}", "len", "capacity", "size(B)", "ptr");

    let mut prev_ptr = v.as_ptr();
    for i in 0..33 {
        v.push(i);
        let ptr = v.as_ptr();
        let reallocated = if ptr != prev_ptr { " ← 再割当!" } else { "" };
        if ptr != prev_ptr || i == 0 {
            println!("{:>5} {:>10} {:>10} {:>18p}{}",
                v.len(),
                v.capacity(),
                v.capacity() * std::mem::size_of::<i32>(),
                ptr,
                reallocated
            );
        }
        prev_ptr = ptr;
    }
    // 出力例:
    //   len   capacity   size(B)                ptr
    //     1          4        16   0x600000000010 ← 再割当!
    //     5          8        32   0x600000000030 ← 再割当!
    //     9         16        64   0x600000000050 ← 再割当!
    //    17         32       128   0x600000000090 ← 再割当!
    //    33         64       256   0x600000000110 ← 再割当!

    // with_capacity で事前確保すれば再割当を避けられる
    let v2: Vec<i32> = Vec::with_capacity(100);
    println!("\nwith_capacity(100): len={}, capacity={}", v2.len(), v2.capacity());

    // shrink_to_fit で余分な容量を解放
    let mut v3 = vec![1, 2, 3, 4, 5];
    v3.reserve(1000);
    println!("reserve後: capacity={}", v3.capacity());
    v3.shrink_to_fit();
    println!("shrink後:  capacity={}", v3.capacity());
}
```

### コード例: ヒープ割当のベンチマーク

```rust
use std::time::Instant;

fn benchmark_stack_vs_heap() {
    const ITERATIONS: usize = 1_000_000;

    // スタック割当のベンチマーク
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _data = [0u8; 256]; // スタック上に256バイト
        std::hint::black_box(&_data);
    }
    let stack_time = start.elapsed();

    // ヒープ割当のベンチマーク
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _data = Box::new([0u8; 256]); // ヒープ上に256バイト
        std::hint::black_box(&_data);
    }
    let heap_time = start.elapsed();

    println!("スタック割当: {:?} ({} 回)", stack_time, ITERATIONS);
    println!("ヒープ割当:   {:?} ({} 回)", heap_time, ITERATIONS);
    println!("ヒープ/スタック比: {:.1}x",
        heap_time.as_nanos() as f64 / stack_time.as_nanos() as f64);
}

fn main() {
    benchmark_stack_vs_heap();
    // 典型的な結果:
    // スタック割当: 1.2ms (1000000 回)
    // ヒープ割当:   25ms (1000000 回)
    // ヒープ/スタック比: 20.8x
}
```

---

## 3. 型のメモリレイアウト

### コード例2: サイズとアライメントの確認

```rust
use std::mem;

#[repr(C)]
struct CLayout {
    a: u8,    // 1 byte + 3 padding
    b: u32,   // 4 bytes
    c: u8,    // 1 byte + 3 padding
}
// サイズ: 12 bytes (C互換レイアウト)

struct RustLayout {
    a: u8,
    b: u32,
    c: u8,
}
// Rustコンパイラが最適化 → フィールド並び替え可能
// サイズ: 8 bytes (b, a, c の順に並べて最適化)

fn main() {
    println!("CLayout:    size={}, align={}",
        mem::size_of::<CLayout>(), mem::align_of::<CLayout>());
    println!("RustLayout: size={}, align={}",
        mem::size_of::<RustLayout>(), mem::align_of::<RustLayout>());

    // 列挙体のサイズ
    println!("Option<u8>:       {}", mem::size_of::<Option<u8>>());       // 2
    println!("Option<Box<u8>>:  {}", mem::size_of::<Option<Box<u8>>>()); // 8 (ニッチ最適化!)
    println!("Option<&u8>:      {}", mem::size_of::<Option<&u8>>());     // 8 (null最適化!)
}
```

### コード例3: repr 属性の種類

```rust
// repr(C) — C言語と同じレイアウト (FFI用)
#[repr(C)]
struct FFIPoint {
    x: f64,
    y: f64,
}

// repr(transparent) — 内部の単一フィールドと同じレイアウト
#[repr(transparent)]
struct Meters(f64);
// Meters と f64 は ABI互換 → FFI で安全にキャスト可能

// repr(packed) — パディングなし (アクセス遅い可能性)
#[repr(C, packed)]
struct PackedHeader {
    magic: u8,
    version: u32,  // アライメントなし → アクセスが遅い場合あり
    length: u16,
}
// サイズ: 7 bytes (パディングなし)

// repr(align(N)) — 最小アライメントを指定
#[repr(align(64))]
struct CacheLine {
    data: [u8; 64],
}
// キャッシュラインに整列 → false sharing 防止

// repr(u8/u16/u32...) — 列挙体の判別子サイズを指定
#[repr(u8)]
enum PacketType {
    Ping = 0,
    Pong = 1,
    Data = 2,
}
```

### コード例: パディングの可視化

```rust
use std::mem;

/// フィールドのオフセットを取得するマクロ
macro_rules! offset_of {
    ($type:ty, $field:ident) => {{
        let dummy = core::mem::MaybeUninit::<$type>::uninit();
        let base = dummy.as_ptr() as usize;
        let field = unsafe { &(*dummy.as_ptr()).$field as *const _ as usize };
        field - base
    }};
}

#[repr(C)]
struct Example1 {
    a: u8,    // offset 0, size 1
    // padding: 3 bytes
    b: u32,   // offset 4, size 4
    c: u8,    // offset 8, size 1
    // padding: 1 byte
    d: u16,   // offset 10, size 2
}

#[repr(C)]
struct Example2 {
    b: u32,   // offset 0, size 4 — 最大アライメント型を先頭に
    d: u16,   // offset 4, size 2
    a: u8,    // offset 6, size 1
    c: u8,    // offset 7, size 1
}

fn main() {
    println!("=== パディング解析 ===");
    println!("Example1 (非効率な配置):");
    println!("  size={}, align={}", mem::size_of::<Example1>(), mem::align_of::<Example1>());
    println!("  a: offset={}", offset_of!(Example1, a));
    println!("  b: offset={}", offset_of!(Example1, b));
    println!("  c: offset={}", offset_of!(Example1, c));
    println!("  d: offset={}", offset_of!(Example1, d));
    // size=12, padding=4 bytes

    println!("\nExample2 (効率的な配置):");
    println!("  size={}, align={}", mem::size_of::<Example2>(), mem::align_of::<Example2>());
    println!("  b: offset={}", offset_of!(Example2, b));
    println!("  d: offset={}", offset_of!(Example2, d));
    println!("  a: offset={}", offset_of!(Example2, a));
    println!("  c: offset={}", offset_of!(Example2, c));
    // size=8, padding=0 bytes
}
```

### コード例: 各種プリミティブ型のサイズとアライメント一覧

```rust
use std::mem;

fn print_layout<T>(name: &str) {
    println!("  {:<24} size={:>2}, align={:>2}",
        name,
        mem::size_of::<T>(),
        mem::align_of::<T>());
}

fn main() {
    println!("=== プリミティブ型 ===");
    print_layout::<bool>("bool");
    print_layout::<u8>("u8");
    print_layout::<u16>("u16");
    print_layout::<u32>("u32");
    print_layout::<u64>("u64");
    print_layout::<u128>("u128");
    print_layout::<usize>("usize");
    print_layout::<f32>("f32");
    print_layout::<f64>("f64");
    print_layout::<char>("char");

    println!("\n=== ポインタ型 ===");
    print_layout::<*const u8>("*const u8");
    print_layout::<*const [u8]>("*const [u8] (fat ptr)");
    print_layout::<*const dyn std::fmt::Debug>("*const dyn Debug (fat ptr)");
    print_layout::<&u8>("&u8");
    print_layout::<&[u8]>("&[u8] (slice ref)");
    print_layout::<&dyn std::fmt::Debug>("&dyn Debug (trait obj)");

    println!("\n=== コレクション型 ===");
    print_layout::<Vec<u8>>("Vec<u8>");
    print_layout::<String>("String");
    print_layout::<Box<u8>>("Box<u8>");
    print_layout::<std::collections::HashMap<u64, u64>>("HashMap<u64, u64>");
    print_layout::<std::collections::BTreeMap<u64, u64>>("BTreeMap<u64, u64>");
    print_layout::<std::collections::VecDeque<u8>>("VecDeque<u8>");

    println!("\n=== スマートポインタ ===");
    print_layout::<std::sync::Arc<u8>>("Arc<u8>");
    print_layout::<std::rc::Rc<u8>>("Rc<u8>");
    print_layout::<std::sync::Mutex<u8>>("Mutex<u8>");
    print_layout::<std::sync::RwLock<u8>>("RwLock<u8>");

    println!("\n=== ゼロサイズ型 ===");
    print_layout::<()>("()");
    print_layout::<std::marker::PhantomData<u8>>("PhantomData<u8>");
    print_layout::<[u8; 0]>("[u8; 0]");
}
```

---

## 4. 列挙体のメモリ最適化

```
┌──────────── enum のメモリレイアウト ────────────┐
│                                                  │
│  enum Shape {                                    │
│      Circle(f64),        // 半径                 │
│      Rect(f64, f64),     // 幅, 高さ             │
│      Point,                                      │
│  }                                               │
│                                                  │
│  メモリ: [tag: 8 bytes][data: 16 bytes] = 24b   │
│                                                  │
│  Circle: [0][radius: f64][padding: 8]           │
│  Rect:   [1][width: f64][height: f64]           │
│  Point:  [2][unused: 16]                        │
│                                                  │
│  ── ニッチ最適化 ──                               │
│                                                  │
│  Option<Box<T>>:                                 │
│    Some(ptr) → [ptr値]     (8 bytes)            │
│    None      → [0x0]       (8 bytes)            │
│  ※ Box は非NULL保証 → 0 を None に使える        │
│  ※ tag 不要! Box と同じサイズ!                   │
└──────────────────────────────────────────────────┘
```

### コード例4: ニッチ最適化の確認

```rust
use std::mem::size_of;
use std::num::NonZeroU64;

fn main() {
    // ニッチ最適化なし
    println!("Option<u64>:       {} bytes", size_of::<Option<u64>>());       // 16

    // ニッチ最適化あり (NonZero の 0 を None に使う)
    println!("Option<NonZeroU64>: {} bytes", size_of::<Option<NonZeroU64>>()); // 8

    // ポインタ型はニッチ最適化される
    println!("Option<Box<i32>>:   {} bytes", size_of::<Option<Box<i32>>>()); // 8
    println!("Option<&i32>:       {} bytes", size_of::<Option<&i32>>());     // 8
    println!("Option<String>:     {} bytes", size_of::<Option<String>>());   // 24 (Stringと同じ!)

    // Result も最適化
    println!("Result<Box<i32>, Box<str>>: {} bytes",
        size_of::<Result<Box<i32>, Box<str>>>());  // 16
}
```

### コード例: enum サイズの詳細分析

```rust
use std::mem;

// 各バリアントのデータサイズが異なる enum
enum Message {
    Quit,                       // 0 bytes のデータ
    Move { x: i32, y: i32 },   // 8 bytes のデータ
    Write(String),              // 24 bytes のデータ
    Color(u8, u8, u8),          // 3 bytes のデータ
}

// ネストされた enum
enum Outer {
    A(Inner),
    B(u8),
}

enum Inner {
    X(u64),
    Y(u32),
}

fn main() {
    println!("=== enum サイズ分析 ===");
    println!("Message:  size={}, align={}",
        mem::size_of::<Message>(), mem::align_of::<Message>());
    // サイズは最大バリアント (Write: 24 bytes) + tag + padding

    println!("Outer:    size={}", mem::size_of::<Outer>());
    println!("Inner:    size={}", mem::size_of::<Inner>());

    // ネストされた Option のニッチ最適化
    println!("\n=== 多重 Option のニッチ最適化 ===");
    println!("Option<bool>:                       {} bytes", mem::size_of::<Option<bool>>());
    println!("Option<Option<bool>>:               {} bytes", mem::size_of::<Option<Option<bool>>>());
    // bool は 0 か 1 なので、2 を None に使える → 1 byte!

    println!("Option<Option<Option<bool>>>:        {} bytes",
        mem::size_of::<Option<Option<Option<bool>>>>());
    // 0=false, 1=true, 2=Some(None), 3=None → まだ 1 byte!

    // 参照のネスト
    println!("\nOption<&u8>:                        {} bytes", mem::size_of::<Option<&u8>>());
    println!("Option<Option<&u8>>:                {} bytes", mem::size_of::<Option<Option<&u8>>>());
    // Option<&u8> は null=None, 非null=Some → 8 bytes
    // Option<Option<&u8>> はニッチが足りない → 16 bytes
}
```

### コード例: enum のサイズを最小化するテクニック

```rust
use std::mem;

// 改善前: 最大バリアントに合わせて巨大になる
enum LargeEnum {
    Small(u32),
    Medium([u8; 64]),
    Large([u8; 1024]),  // ← この 1024 bytes が全体サイズを決定
}

// 改善方法1: 大きなバリアントを Box で包む
enum OptimizedEnum1 {
    Small(u32),
    Medium([u8; 64]),
    Large(Box<[u8; 1024]>),  // 8 bytes (ポインタ)
}

// 改善方法2: 共通データを外に出す
struct MessageBase {
    id: u64,
    timestamp: u64,
}

enum MessagePayload {
    Text(String),
    Binary(Vec<u8>),
    Ping,
}

struct OptimizedMessage {
    base: MessageBase,
    payload: MessagePayload,
}

// 改善方法3: インデックスで間接参照
struct Arena {
    texts: Vec<String>,
    binaries: Vec<Vec<u8>>,
}

enum ArenaMessage {
    Text(usize),     // texts へのインデックス
    Binary(usize),   // binaries へのインデックス
    Ping,
}

fn main() {
    println!("LargeEnum:       {} bytes", mem::size_of::<LargeEnum>());
    println!("OptimizedEnum1:  {} bytes", mem::size_of::<OptimizedEnum1>());
    println!("OptimizedMessage: {} bytes", mem::size_of::<OptimizedMessage>());
    println!("ArenaMessage:    {} bytes", mem::size_of::<ArenaMessage>());
}
```

---

## 5. ゼロサイズ型 (ZST) とファントムデータ

### コード例5: ZST の活用

```rust
use std::marker::PhantomData;
use std::mem;

// ゼロサイズ型: メモリを消費しない
struct Meters;
struct Seconds;

// 型レベルで単位を区別 (メモリコストなし)
struct Quantity<Unit> {
    value: f64,
    _unit: PhantomData<Unit>,
}

impl<U> Quantity<U> {
    fn new(value: f64) -> Self {
        Quantity { value, _unit: PhantomData }
    }
}

fn main() {
    let distance = Quantity::<Meters>::new(100.0);
    let time = Quantity::<Seconds>::new(9.58);

    // Quantity<Meters> と Quantity<Seconds> は異なる型
    // → コンパイル時に単位の混同を防止

    println!("Quantity<Meters> サイズ: {}", mem::size_of::<Quantity<Meters>>());
    // 8 bytes (f64 のみ。PhantomData は 0 bytes)

    // Vec<()> のサイズ
    let units: Vec<()> = vec![(); 1000];
    println!("Vec<()> 要素は {} bytes", mem::size_of::<()>()); // 0
    // メモリ割当されない (len だけ追跡)
}
```

### コード例: 型状態パターンでの ZST 活用

```rust
use std::marker::PhantomData;

// 型状態を表す ZST
struct Idle;
struct Connected;
struct Authenticated;

// 接続の状態を型パラメータで管理
struct Connection<State> {
    host: String,
    port: u16,
    _state: PhantomData<State>,
}

impl Connection<Idle> {
    fn new(host: &str, port: u16) -> Self {
        Connection {
            host: host.to_string(),
            port,
            _state: PhantomData,
        }
    }

    fn connect(self) -> Result<Connection<Connected>, String> {
        println!("接続中: {}:{}", self.host, self.port);
        Ok(Connection {
            host: self.host,
            port: self.port,
            _state: PhantomData,
        })
    }
}

impl Connection<Connected> {
    fn authenticate(self, _token: &str) -> Result<Connection<Authenticated>, String> {
        println!("認証中...");
        Ok(Connection {
            host: self.host,
            port: self.port,
            _state: PhantomData,
        })
    }

    fn disconnect(self) -> Connection<Idle> {
        println!("切断");
        Connection {
            host: self.host,
            port: self.port,
            _state: PhantomData,
        }
    }
}

impl Connection<Authenticated> {
    fn query(&self, sql: &str) -> Result<String, String> {
        println!("クエリ実行: {}", sql);
        Ok("結果".to_string())
    }

    fn disconnect(self) -> Connection<Idle> {
        println!("切断");
        Connection {
            host: self.host,
            port: self.port,
            _state: PhantomData,
        }
    }
}

fn main() {
    use std::mem;

    // すべての状態で同じサイズ (ZST のおかげ)
    println!("Connection<Idle>:          {} bytes", mem::size_of::<Connection<Idle>>());
    println!("Connection<Connected>:     {} bytes", mem::size_of::<Connection<Connected>>());
    println!("Connection<Authenticated>: {} bytes", mem::size_of::<Connection<Authenticated>>());

    // コンパイル時に不正な状態遷移を防止
    let conn = Connection::<Idle>::new("localhost", 5432);
    let conn = conn.connect().unwrap();
    // conn.query("SELECT 1"); // コンパイルエラー! Connected 状態では query できない
    let conn = conn.authenticate("secret").unwrap();
    let _result = conn.query("SELECT 1").unwrap(); // OK
}
```

### コード例: PhantomData による所有権・ライフタイムの表現

```rust
use std::marker::PhantomData;
use std::mem;

/// PhantomData で所有権の意味を表現する例
struct Owned<T> {
    ptr: *mut T,
    _owns: PhantomData<T>, // T を「所有」する → drop 時に T を解放する責任
}

struct Borrowed<'a, T> {
    ptr: *const T,
    _borrows: PhantomData<&'a T>, // T を「借用」する → ライフタイム 'a が有効
}

impl<T> Owned<T> {
    fn new(value: T) -> Self {
        let ptr = Box::into_raw(Box::new(value));
        Owned {
            ptr,
            _owns: PhantomData,
        }
    }

    fn as_ref(&self) -> &T {
        unsafe { &*self.ptr }
    }
}

impl<T> Drop for Owned<T> {
    fn drop(&mut self) {
        unsafe {
            drop(Box::from_raw(self.ptr));
        }
    }
}

/// PhantomData で共変性・反変性を制御
struct Covariant<'a, T: 'a> {
    _phantom: PhantomData<&'a T>, // 共変: T が 'a より長生きすればOK
}

struct Invariant<'a, T: 'a> {
    _phantom: PhantomData<&'a mut T>, // 不変: 完全一致が必要
}

fn main() {
    println!("Owned<u64>:    {} bytes", mem::size_of::<Owned<u64>>());    // 8 (ptr のみ)
    println!("Borrowed<u64>: {} bytes", mem::size_of::<Borrowed<u64>>()); // 8 (ptr のみ)

    let owned = Owned::new(42u64);
    println!("値: {}", owned.as_ref());
}
```

---

## 6. 高度なメモリ制御

### コード例: カスタムアロケータ

```rust
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

/// メモリ使用量を追跡するアロケータ
struct TrackingAllocator {
    inner: System,
    allocated: AtomicUsize,
    deallocated: AtomicUsize,
    allocation_count: AtomicUsize,
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = self.inner.alloc(layout);
        if !ptr.is_null() {
            self.allocated.fetch_add(layout.size(), Ordering::Relaxed);
            self.allocation_count.fetch_add(1, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.inner.dealloc(ptr, layout);
        self.deallocated.fetch_add(layout.size(), Ordering::Relaxed);
    }
}

impl TrackingAllocator {
    const fn new() -> Self {
        TrackingAllocator {
            inner: System,
            allocated: AtomicUsize::new(0),
            deallocated: AtomicUsize::new(0),
            allocation_count: AtomicUsize::new(0),
        }
    }

    fn report(&self) {
        let alloc = self.allocated.load(Ordering::Relaxed);
        let dealloc = self.deallocated.load(Ordering::Relaxed);
        let count = self.allocation_count.load(Ordering::Relaxed);
        println!("=== メモリレポート ===");
        println!("  総割当量:   {} bytes", alloc);
        println!("  総解放量:   {} bytes", dealloc);
        println!("  現在使用量: {} bytes", alloc - dealloc);
        println!("  割当回数:   {} 回", count);
    }
}

#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator::new();

fn main() {
    ALLOCATOR.report();

    {
        let _v: Vec<u8> = vec![0; 1024];
        let _s = String::from("Hello, allocator!");
        let _b = Box::new([0u8; 256]);
        ALLOCATOR.report();
    }

    // スコープを抜けると解放される
    ALLOCATOR.report();
}
```

### コード例: MaybeUninit による未初期化メモリの安全な利用

```rust
use std::mem::MaybeUninit;

/// MaybeUninit を使って配列を効率的に初期化する
fn create_fibonacci_array() -> [u64; 50] {
    let mut arr: [MaybeUninit<u64>; 50] = unsafe {
        MaybeUninit::uninit().assume_init()
    };

    arr[0] = MaybeUninit::new(0);
    arr[1] = MaybeUninit::new(1);

    for i in 2..50 {
        let a = unsafe { arr[i - 1].assume_init() };
        let b = unsafe { arr[i - 2].assume_init() };
        arr[i] = MaybeUninit::new(a + b);
    }

    // 全要素が初期化済みなので安全に変換
    unsafe {
        // MaybeUninit<T> と T はメモリレイアウトが同じ
        std::mem::transmute(arr)
    }
}

/// MaybeUninit を使った遅延初期化パターン
struct LazyBuffer {
    data: MaybeUninit<[u8; 4096]>,
    initialized: bool,
}

impl LazyBuffer {
    fn new() -> Self {
        LazyBuffer {
            data: MaybeUninit::uninit(),
            initialized: false,
        }
    }

    fn get_or_init(&mut self) -> &[u8; 4096] {
        if !self.initialized {
            // 初回アクセス時のみ初期化
            self.data = MaybeUninit::new([0u8; 4096]);
            self.initialized = true;
        }
        unsafe { self.data.assume_init_ref() }
    }
}

fn main() {
    let fib = create_fibonacci_array();
    println!("fib[49] = {}", fib[49]); // 7778742049

    let mut buf = LazyBuffer::new();
    let data = buf.get_or_init();
    println!("Buffer size: {}", data.len());
}
```

### コード例: アライメント制御とキャッシュ最適化

```rust
use std::mem;

/// false sharing を防ぐためのキャッシュライン整列
#[repr(align(64))]
struct CacheAligned<T> {
    value: T,
}

/// 複数スレッドで独立にアクセスするカウンタ
struct PerThreadCounters {
    // 各カウンタが別のキャッシュラインに配置される
    counters: [CacheAligned<std::sync::atomic::AtomicU64>; 8],
}

impl PerThreadCounters {
    fn new() -> Self {
        PerThreadCounters {
            counters: std::array::from_fn(|_| CacheAligned {
                value: std::sync::atomic::AtomicU64::new(0),
            }),
        }
    }

    fn increment(&self, thread_id: usize) {
        self.counters[thread_id % 8]
            .value
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    fn total(&self) -> u64 {
        self.counters
            .iter()
            .map(|c| c.value.load(std::sync::atomic::Ordering::Relaxed))
            .sum()
    }
}

/// SIMD 用のアライメント
#[repr(align(32))]
struct SimdAligned {
    data: [f32; 8], // AVX2 用に 32 バイト境界に整列
}

fn main() {
    println!("CacheAligned<u64>: size={}, align={}",
        mem::size_of::<CacheAligned<u64>>(),
        mem::align_of::<CacheAligned<u64>>());
    // size=64, align=64

    println!("SimdAligned: size={}, align={}",
        mem::size_of::<SimdAligned>(),
        mem::align_of::<SimdAligned>());
    // size=32, align=32

    let counters = PerThreadCounters::new();
    // 各カウンタは 64 バイト間隔 → false sharing なし
    println!("PerThreadCounters total size: {} bytes",
        mem::size_of::<PerThreadCounters>());
}
```

---

## 7. 実践的なメモリプロファイリング

### コード例: std::alloc を使ったメモリ使用量の測定

```rust
use std::alloc::{Layout, alloc, dealloc};

/// 手動メモリ割当のデモンストレーション
fn manual_allocation_demo() {
    // Layout: サイズとアライメントを指定
    let layout = Layout::new::<[u8; 256]>();
    println!("Layout: size={}, align={}", layout.size(), layout.align());

    unsafe {
        // メモリ割当
        let ptr = alloc(layout);
        if ptr.is_null() {
            panic!("メモリ割当失敗");
        }

        // 初期化
        std::ptr::write_bytes(ptr, 0, 256);

        // 使用
        *ptr = 42;
        println!("割当された値: {}", *ptr);

        // 解放
        dealloc(ptr, layout);
    }

    // Layout::from_size_align で動的にレイアウトを作成
    let dynamic_layout = Layout::from_size_align(1024, 16).unwrap();
    println!("Dynamic layout: size={}, align={}",
        dynamic_layout.size(), dynamic_layout.align());
}

/// コレクションのメモリ使用量を推定する
fn estimate_collection_memory() {
    use std::mem;
    use std::collections::HashMap;

    // Vec のメモリ使用量
    let v: Vec<u64> = (0..1000).collect();
    let stack_size = mem::size_of::<Vec<u64>>();
    let heap_size = v.capacity() * mem::size_of::<u64>();
    println!("Vec<u64> (1000要素):");
    println!("  スタック: {} bytes", stack_size);
    println!("  ヒープ:   {} bytes", heap_size);
    println!("  合計:     {} bytes", stack_size + heap_size);

    // HashMap のメモリ使用量 (概算)
    let mut map: HashMap<u64, String> = HashMap::new();
    for i in 0..100 {
        map.insert(i, format!("value_{}", i));
    }
    let map_stack = mem::size_of::<HashMap<u64, String>>();
    // HashMap の内部は複雑だが、概算値を計算
    let avg_value_size = 16; // String のヒープデータ平均サイズ
    let entry_overhead = mem::size_of::<u64>() + mem::size_of::<String>() + 8; // key + value + metadata
    let estimated_heap = map.capacity() * entry_overhead + 100 * avg_value_size;
    println!("\nHashMap<u64, String> (100要素):");
    println!("  スタック:     {} bytes", map_stack);
    println!("  ヒープ (概算): {} bytes", estimated_heap);
}

fn main() {
    manual_allocation_demo();
    println!();
    estimate_collection_memory();
}
```

### コード例: メモリリークの検出パターン

```rust
use std::sync::Arc;
use std::cell::RefCell;

/// 循環参照によるメモリリークの例
fn demonstrate_circular_reference() {
    struct Node {
        value: i32,
        next: RefCell<Option<Arc<Node>>>,
    }

    let a = Arc::new(Node {
        value: 1,
        next: RefCell::new(None),
    });

    let b = Arc::new(Node {
        value: 2,
        next: RefCell::new(Some(Arc::clone(&a))),
    });

    // 循環参照を作成
    *a.next.borrow_mut() = Some(Arc::clone(&b));

    println!("a strong count: {}", Arc::strong_count(&a)); // 2
    println!("b strong count: {}", Arc::strong_count(&b)); // 2
    // → スコープを抜けても strong count が 0 にならず、メモリリーク!
}

/// Weak 参照で循環参照を防ぐ
fn demonstrate_weak_reference() {
    use std::sync::Weak;

    struct SafeNode {
        value: i32,
        next: RefCell<Option<Arc<SafeNode>>>,
        prev: RefCell<Option<Weak<SafeNode>>>, // 弱い参照 → 循環しない
    }

    let a = Arc::new(SafeNode {
        value: 1,
        next: RefCell::new(None),
        prev: RefCell::new(None),
    });

    let b = Arc::new(SafeNode {
        value: 2,
        next: RefCell::new(None),
        prev: RefCell::new(Some(Arc::downgrade(&a))), // Weak 参照
    });

    *a.next.borrow_mut() = Some(Arc::clone(&b));

    println!("a strong count: {}", Arc::strong_count(&a)); // 1
    println!("b strong count: {}", Arc::strong_count(&b)); // 2
    println!("a weak count:   {}", Arc::weak_count(&a));   // 1
    // → 正常に解放される
}

fn main() {
    println!("=== 循環参照 (メモリリーク) ===");
    demonstrate_circular_reference();
    println!("\n=== Weak 参照 (リーク防止) ===");
    demonstrate_weak_reference();
}
```

---

## 8. 比較表

### スタック vs ヒープ

| 特性 | スタック | ヒープ |
|---|---|---|
| 割当速度 | 超高速 (SP移動のみ) | 遅い (アロケータ呼び出し) |
| 解放速度 | 超高速 (自動) | 遅い (free/dealloc) |
| サイズ制約 | 固定・小サイズ (通常8MB上限) | 動的・大サイズ可 |
| ライフタイム | スコープに紐付き | 任意 (所有権で管理) |
| キャッシュ効率 | 非常に高い (局所性) | 低い (分散配置) |
| フラグメンテーション | なし | あり |
| スレッド | 各スレッドに独立 | 全スレッドで共有 |
| オーバーフロー | SIGSEGV で終了 | OOM エラー |

### repr 属性の比較

| repr | レイアウト | 用途 | 注意点 |
|---|---|---|---|
| (デフォルト) | コンパイラ最適化 | 通常のRustコード | フィールド順不定 |
| `repr(C)` | C言語互換 | FFI、外部ライブラリ | パディングあり |
| `repr(transparent)` | 内部型と同一 | ニュータイプパターン | 単一フィールド限定 |
| `repr(packed)` | パディングなし | バイナリプロトコル | アクセス性能低下 |
| `repr(align(N))` | 最小アライメントN | キャッシュライン整列 | メモリ消費増加 |

### スマートポインタのメモリコスト

| 型 | スタック上サイズ | ヒープオーバーヘッド | 用途 |
|---|---|---|---|
| `Box<T>` | 8 bytes (ptr) | 0 | 単一値の所有権 |
| `Rc<T>` | 8 bytes (ptr) | 16 bytes (strong+weak) | シングルスレッド共有 |
| `Arc<T>` | 8 bytes (ptr) | 16 bytes (atomic strong+weak) | マルチスレッド共有 |
| `Vec<T>` | 24 bytes (ptr+len+cap) | 0 | 動的配列 |
| `String` | 24 bytes (ptr+len+cap) | 0 | UTF-8 文字列 |
| `Cow<'a, T>` | 24 bytes (enum) | 条件付き | 遅延コピー |

---

## 9. アンチパターン

### アンチパターン1: 不必要なヒープ割当

```rust
// NG: 小さな固定長データを Box で包む
fn bad() -> Box<(f64, f64)> {
    Box::new((1.0, 2.0)) // 16 bytes をヒープに割当 → 無駄
}

// OK: そのまま返す (コピー/ムーブで十分)
fn good() -> (f64, f64) {
    (1.0, 2.0) // スタック上でコピー
}

// Box が必要な場面:
// - 再帰型 (コンパイル時にサイズ不定)
// - trait object (dyn Trait)
// - 大きな構造体のムーブコスト回避
```

### アンチパターン2: repr(packed) の乱用

```rust
// NG: パフォーマンス重視のコードで packed を使う
#[repr(packed)]
struct BadPerf {
    flag: u8,
    value: u64, // 非アラインアクセス → CPU性能低下
}

// OK: アライメントを活かしてパディングを最小化
struct GoodPerf {
    value: u64,  // 8 byte align → 先頭に配置
    flag: u8,    // 末尾 → パディング最小
}
// packed は バイナリフォーマット解析など、
// レイアウトの正確性が性能より重要な場面でのみ使用
```

### アンチパターン3: 過剰な Clone

```rust
use std::sync::Arc;

// NG: 大きなデータを毎回 clone
fn bad_process(data: &Vec<String>) {
    let cloned = data.clone(); // 全文字列をコピー → メモリ倍増
    process_data(cloned);
}

// OK: 参照で渡す
fn good_process(data: &[String]) {
    process_data_ref(data); // コピーなし
}

// OK: 共有所有権が必要なら Arc を使う
fn good_shared_process(data: Arc<Vec<String>>) {
    let shared = Arc::clone(&data); // ポインタのコピーのみ (8 bytes)
    std::thread::spawn(move || {
        process_data_ref(&shared);
    });
}

fn process_data(_data: Vec<String>) {}
fn process_data_ref(_data: &[String]) {}
```

### アンチパターン4: String の非効率な構築

```rust
fn main() {
    let items = vec!["a", "b", "c", "d", "e"];

    // NG: format! の連鎖 → 毎回新しい String を割当
    let mut result = String::new();
    for item in &items {
        result = format!("{}{}, ", result, item);
        // 毎回 result 全体 + item をコピーして新しい String を作成
        // O(n^2) のメモリ割当!
    }

    // OK: push_str / write! を使う
    let mut result = String::with_capacity(items.len() * 4); // 事前確保
    for (i, item) in items.iter().enumerate() {
        if i > 0 { result.push_str(", "); }
        result.push_str(item);
    }
    // O(n) のメモリ割当

    // OK: join を使う (最もシンプル)
    let result = items.join(", ");
    println!("{}", result);
}
```

---

## 10. 実践パターン集

### パターン1: Small String Optimization (SSO)

```rust
use std::mem;

/// スタック上に小さな文字列を格納する型
/// 23バイト以下はスタック上、それ以上はヒープに割当
enum SmallString {
    Inline {
        data: [u8; 23],
        len: u8,
    },
    Heap(String),
}

impl SmallString {
    fn new(s: &str) -> Self {
        if s.len() <= 23 {
            let mut data = [0u8; 23];
            data[..s.len()].copy_from_slice(s.as_bytes());
            SmallString::Inline {
                data,
                len: s.len() as u8,
            }
        } else {
            SmallString::Heap(s.to_string())
        }
    }

    fn as_str(&self) -> &str {
        match self {
            SmallString::Inline { data, len } => {
                std::str::from_utf8(&data[..*len as usize]).unwrap()
            }
            SmallString::Heap(s) => s.as_str(),
        }
    }

    fn is_inline(&self) -> bool {
        matches!(self, SmallString::Inline { .. })
    }
}

fn main() {
    println!("SmallString size: {} bytes", mem::size_of::<SmallString>());

    let short = SmallString::new("Hello");
    let long = SmallString::new("This is a very long string that exceeds the inline buffer");

    println!("'{}' inline={}", short.as_str(), short.is_inline()); // true
    println!("'{}' inline={}", long.as_str(), long.is_inline());   // false
}
```

### パターン2: Arena Allocator パターン

```rust
/// 簡易的な Arena アロケータ
/// 大量の小さなオブジェクトを高速に割当し、一括で解放する
struct Arena {
    chunks: Vec<Vec<u8>>,
    current: Vec<u8>,
    chunk_size: usize,
}

impl Arena {
    fn new(chunk_size: usize) -> Self {
        Arena {
            chunks: Vec::new(),
            current: Vec::with_capacity(chunk_size),
            chunk_size,
        }
    }

    fn alloc(&mut self, size: usize) -> &mut [u8] {
        // アライメントを 8 バイトに揃える
        let aligned_size = (size + 7) & !7;

        if self.current.len() + aligned_size > self.current.capacity() {
            // 現在のチャンクが足りない → 新しいチャンクを作成
            let old = std::mem::replace(
                &mut self.current,
                Vec::with_capacity(self.chunk_size.max(aligned_size)),
            );
            if !old.is_empty() {
                self.chunks.push(old);
            }
        }

        let start = self.current.len();
        self.current.resize(start + aligned_size, 0);
        &mut self.current[start..start + size]
    }

    fn bytes_allocated(&self) -> usize {
        self.chunks.iter().map(|c| c.len()).sum::<usize>() + self.current.len()
    }

    fn reset(&mut self) {
        self.chunks.clear();
        self.current.clear();
    }
}

fn main() {
    let mut arena = Arena::new(4096);

    // 1000 個の小さなオブジェクトを高速に割当
    for i in 0..1000 {
        let buf = arena.alloc(32);
        buf[0] = (i % 256) as u8;
    }

    println!("Arena 割当量: {} bytes", arena.bytes_allocated());

    // 一括解放
    arena.reset();
    println!("リセット後: {} bytes", arena.bytes_allocated());
}
```

### パターン3: メモリ効率的なデータ構造 (SoA vs AoS)

```rust
use std::time::Instant;

// AoS (Array of Structures) — 構造体の配列
struct ParticleAoS {
    x: f32,
    y: f32,
    z: f32,
    vx: f32,
    vy: f32,
    vz: f32,
    mass: f32,
    _padding: f32, // 32 bytes total, キャッシュライン半分
}

// SoA (Structure of Arrays) — 配列の構造体
struct ParticlesSoA {
    x: Vec<f32>,
    y: Vec<f32>,
    z: Vec<f32>,
    vx: Vec<f32>,
    vy: Vec<f32>,
    vz: Vec<f32>,
    mass: Vec<f32>,
}

impl ParticlesSoA {
    fn new(n: usize) -> Self {
        ParticlesSoA {
            x: vec![0.0; n],
            y: vec![0.0; n],
            z: vec![0.0; n],
            vx: vec![1.0; n],
            vy: vec![1.0; n],
            vz: vec![1.0; n],
            mass: vec![1.0; n],
        }
    }

    /// 位置を更新 (x, y, z と vx, vy, vz のみアクセス)
    fn update_positions(&mut self, dt: f32) {
        for i in 0..self.x.len() {
            self.x[i] += self.vx[i] * dt;
            self.y[i] += self.vy[i] * dt;
            self.z[i] += self.vz[i] * dt;
        }
    }
}

fn main() {
    let n = 1_000_000;

    // SoA: 位置更新時に mass をロードしない → キャッシュ効率が良い
    let mut particles = ParticlesSoA::new(n);
    let start = Instant::now();
    for _ in 0..100 {
        particles.update_positions(0.016);
    }
    let soa_time = start.elapsed();
    println!("SoA 位置更新 (100回): {:?}", soa_time);

    // AoS: 位置更新でも mass + padding が キャッシュにロードされる → 無駄
    let mut aos: Vec<ParticleAoS> = (0..n)
        .map(|_| ParticleAoS {
            x: 0.0, y: 0.0, z: 0.0,
            vx: 1.0, vy: 1.0, vz: 1.0,
            mass: 1.0, _padding: 0.0,
        })
        .collect();
    let start = Instant::now();
    for _ in 0..100 {
        for p in aos.iter_mut() {
            p.x += p.vx * 0.016;
            p.y += p.vy * 0.016;
            p.z += p.vz * 0.016;
        }
    }
    let aos_time = start.elapsed();
    println!("AoS 位置更新 (100回): {:?}", aos_time);
    println!("SoA/AoS 比: {:.2}x",
        aos_time.as_nanos() as f64 / soa_time.as_nanos() as f64);
}
```

---

## FAQ

### Q1: なぜ `Vec<T>` はスタック上で24バイトなの?

**A:** `ptr`(8バイト) + `len`(8バイト) + `capacity`(8バイト) = 24バイトです。ヒープ上の実データへのポインタ、現在の要素数、確保済み容量を保持しています。`String` も内部的に `Vec<u8>` なので同じ構造です。

### Q2: `Box<dyn Trait>` はなぜ16バイト?

**A:** データへのポインタ(8バイト) + vtable へのポインタ(8バイト) のファットポインタです。vtable には型のサイズ、drop 関数、各メソッドの関数ポインタが格納されています。

### Q3: enum のサイズを小さくするには?

**A:** (1) 大きなバリアントを `Box` で包む (2) `NonZero*` 型でニッチ最適化を活用 (3) フィールド型を小さくする。

```rust
// 改善前: 最大バリアントに合わせて 104 bytes
enum Message {
    Quit,
    Echo(String),       // 24 bytes
    Data([u8; 100]),    // 100 bytes ← これが支配的
}

// 改善後: Box で包んで 32 bytes
enum MessageOpt {
    Quit,
    Echo(String),
    Data(Box<[u8; 100]>), // 8 bytes (ポインタのみ)
}
```

### Q4: `Cow<str>` はいつ使うべき?

**A:** 入力が借用のままでOKな場合が多いが、場合によってはクローンが必要な場面で `Cow` を使います。例えばパース処理で、ほとんどの入力はそのまま返せるが、エスケープ処理が必要な場合だけコピーする、というパターンです。

```rust
use std::borrow::Cow;

fn escape_html(input: &str) -> Cow<str> {
    if input.contains(['<', '>', '&', '"', '\'']) {
        // エスケープが必要 → 新しい String を作成
        Cow::Owned(
            input
                .replace('&', "&amp;")
                .replace('<', "&lt;")
                .replace('>', "&gt;")
                .replace('"', "&quot;")
                .replace('\'', "&#39;")
        )
    } else {
        // エスケープ不要 → 元の文字列を借用
        Cow::Borrowed(input)
    }
}

fn main() {
    let safe = escape_html("Hello, World!");      // Borrowed (コピーなし)
    let escaped = escape_html("<script>alert(1)</script>"); // Owned (コピーあり)
    println!("{}", safe);
    println!("{}", escaped);
}
```

### Q5: `Pin<T>` とメモリレイアウトの関係は?

**A:** `Pin<T>` はメモリ上の位置を固定することを保証するラッパーです。自己参照構造体（内部に自身のフィールドへのポインタを持つ型）が安全に動作するために必要です。async/await の Future が内部的に自己参照構造体を生成するため、Pin が不可欠です。

```rust
use std::pin::Pin;
use std::marker::PhantomPinned;

/// 自己参照構造体の例
struct SelfReferential {
    data: String,
    ptr_to_data: *const String, // data フィールドへのポインタ
    _pin: PhantomPinned,        // !Unpin を実装 → move を禁止
}

impl SelfReferential {
    fn new(data: String) -> Pin<Box<Self>> {
        let mut boxed = Box::new(SelfReferential {
            data,
            ptr_to_data: std::ptr::null(),
            _pin: PhantomPinned,
        });
        let ptr = &boxed.data as *const String;
        boxed.ptr_to_data = ptr;
        // Pin で包んで移動を禁止
        unsafe { Pin::new_unchecked(boxed) }
    }

    fn get_data(self: Pin<&Self>) -> &str {
        &self.data
    }
}

fn main() {
    let sr = SelfReferential::new("pinned data".to_string());
    println!("data: {}", sr.as_ref().get_data());
}
```

### Q6: メモリアロケータを切り替えるには?

**A:** `#[global_allocator]` 属性で指定します。高性能アロケータとして jemalloc や mimalloc が人気です。

```rust
// jemalloc の例
// Cargo.toml: tikv-jemallocator = "0.6"
#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

// mimalloc の例
// Cargo.toml: mimalloc = "0.1"
// use mimalloc::MiMalloc;
// #[global_allocator]
// static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    // 以降すべてのヒープ割当が指定アロケータを使用
    let v: Vec<u64> = (0..1_000_000).collect();
    println!("要素数: {}", v.len());
}
```

---

## まとめ

| 項目 | 要点 |
|---|---|
| スタック | 高速・固定サイズ・スコープ管理。ローカル変数に最適 |
| ヒープ | 動的サイズ・長寿命。Box/Vec/String が管理 |
| repr(C) | FFI でC言語と互換レイアウトが必要な時 |
| repr(transparent) | ニュータイプとABI互換を保証 |
| アライメント | CPU のメモリアクセス効率に直結 |
| ニッチ最適化 | Option<Box<T>> が Box<T> と同サイズ |
| ZST | PhantomData で型情報を付加(メモリコストゼロ) |
| MaybeUninit | 未初期化メモリの安全な操作 |
| Arena | 大量小オブジェクトの高速割当・一括解放 |
| SoA | キャッシュ効率を重視したデータ配置 |
| カスタムアロケータ | jemalloc/mimalloc で性能向上 |

## 次に読むべきガイド

- [並行性](./01-concurrency.md) — メモリモデルとスレッド安全性
- [FFI](./02-ffi-interop.md) — repr(C)を使った外部言語連携
- [組み込み/WASM](./03-embedded-wasm.md) — メモリ制約環境での最適化

## 参考文献

1. **The Rustonomicon — Data Representation**: https://doc.rust-lang.org/nomicon/data.html
2. **Type Layout (Rust Reference)**: https://doc.rust-lang.org/reference/type-layout.html
3. **Visualizing Rust Memory Layout**: https://www.youtube.com/watch?v=rDoqT-a6UFg
4. **std::alloc Module**: https://doc.rust-lang.org/std/alloc/index.html
5. **Pin and Unpin Explained**: https://doc.rust-lang.org/std/pin/index.html
