# メモリレイアウト — スタック/ヒープ、repr

> Rust のデータ型がメモリ上でどう配置されるかを理解し、repr 属性やアライメント制御で低レベル最適化を行う技術を習得する

## この章で学ぶこと

1. **スタックとヒープ** — 各領域の特性、割当コスト、所有権との関係
2. **型のメモリレイアウト** — サイズ、アライメント、パディング、repr 属性
3. **スマートポインタの内部構造** — Box, Vec, String, Arc のメモリ配置

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

---

## 6. 比較表

### スタック vs ヒープ

| 特性 | スタック | ヒープ |
|---|---|---|
| 割当速度 | 超高速 (SP移動のみ) | 遅い (アロケータ呼び出し) |
| 解放速度 | 超高速 (自動) | 遅い (free/dealloc) |
| サイズ制約 | 固定・小サイズ (通常8MB上限) | 動的・大サイズ可 |
| ライフタイム | スコープに紐付き | 任意 (所有権で管理) |
| キャッシュ効率 | 非常に高い (局所性) | 低い (分散配置) |
| フラグメンテーション | なし | あり |

### repr 属性の比較

| repr | レイアウト | 用途 | 注意点 |
|---|---|---|---|
| (デフォルト) | コンパイラ最適化 | 通常のRustコード | フィールド順不定 |
| `repr(C)` | C言語互換 | FFI、外部ライブラリ | パディングあり |
| `repr(transparent)` | 内部型と同一 | ニュータイプパターン | 単一フィールド限定 |
| `repr(packed)` | パディングなし | バイナリプロトコル | アクセス性能低下 |
| `repr(align(N))` | 最小アライメントN | キャッシュライン整列 | メモリ消費増加 |

---

## 7. アンチパターン

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

## 次に読むべきガイド

- [並行性](./01-concurrency.md) — メモリモデルとスレッド安全性
- [FFI](./02-ffi-interop.md) — repr(C)を使った外部言語連携
- [組み込み/WASM](./03-embedded-wasm.md) — メモリ制約環境での最適化

## 参考文献

1. **The Rustonomicon — Data Representation**: https://doc.rust-lang.org/nomicon/data.html
2. **Type Layout (Rust Reference)**: https://doc.rust-lang.org/reference/type-layout.html
3. **Visualizing Rust Memory Layout**: https://www.youtube.com/watch?v=rDoqT-a6UFg
