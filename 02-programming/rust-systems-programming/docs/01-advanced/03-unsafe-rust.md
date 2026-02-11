# unsafe Rust -- 安全性の境界を越える低レベルプログラミング

> unsafe Rustは借用チェッカーの制約を超えた操作(生ポインタ操作、FFI、ハードウェアアクセス等)を可能にし、安全な抽象化の中に封じ込めることで高レベルAPIの安全性を維持する。

---

## この章で学ぶこと

1. **unsafe の5つの超能力** -- unsafe ブロック内で許可される5つの操作を理解する
2. **生ポインタとFFI** -- *const T / *mut T の操作とC言語連携の方法を習得する
3. **安全な抽象化** -- unsafe を内部に封じ込めて安全なAPIを公開するパターンを学ぶ

---

## 1. unsafe の5つの超能力

```
┌──────────────────────────────────────────────────────┐
│        unsafe で解禁される5つの操作                    │
├──────────────────────────────────────────────────────┤
│ 1. 生ポインタのデリファレンス (*const T / *mut T)     │
│ 2. unsafe 関数・メソッドの呼び出し                    │
│ 3. 可変 static 変数へのアクセス・変更                 │
│ 4. unsafe トレイトの実装                              │
│ 5. union のフィールドアクセス                         │
├──────────────────────────────────────────────────────┤
│  重要: unsafe は借用チェックを無効にしない！          │
│  所有権・型チェックは unsafe 内でも有効               │
└──────────────────────────────────────────────────────┘
```

### 例1: 生ポインタの基本

```rust
fn main() {
    let mut num = 42;

    // 生ポインタの作成は安全(デリファレンスが unsafe)
    let r1 = &num as *const i32;
    let r2 = &mut num as *mut i32;

    unsafe {
        println!("r1 = {}", *r1);
        *r2 = 100;
        println!("r2 = {}", *r2);
    }

    // 任意のアドレスを指す生ポインタ(非常に危険)
    let address = 0x012345usize;
    let _r = address as *const i32;
    // unsafe { println!("{}", *_r); } // 未定義動作！
}
```

---

## 2. unsafe 関数

### 例2: unsafe 関数の定義と呼び出し

```rust
/// 2つのスライスの内容をインプレースで入れ替える。
/// # Safety
/// - `a` と `b` は同じ長さでなければならない
/// - `a` と `b` は重複してはならない
unsafe fn swap_buffers(a: &mut [u8], b: &mut [u8]) {
    debug_assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        std::ptr::swap(
            a.as_mut_ptr().add(i),
            b.as_mut_ptr().add(i),
        );
    }
}

fn main() {
    let mut a = vec![1, 2, 3];
    let mut b = vec![4, 5, 6];

    // 安全性の前提条件を確認してから呼び出す
    assert_eq!(a.len(), b.len());
    unsafe {
        swap_buffers(&mut a, &mut b);
    }
    println!("a={:?}, b={:?}", a, b); // a=[4,5,6], b=[1,2,3]
}
```

### 例3: split_at_mut の実装(標準ライブラリの内部)

```rust
fn split_at_mut(values: &mut [i32], mid: usize) -> (&mut [i32], &mut [i32]) {
    let len = values.len();
    let ptr = values.as_mut_ptr();

    assert!(mid <= len);

    unsafe {
        (
            std::slice::from_raw_parts_mut(ptr, mid),
            std::slice::from_raw_parts_mut(ptr.add(mid), len - mid),
        )
    }
}

fn main() {
    let mut v = vec![1, 2, 3, 4, 5, 6];
    let (left, right) = split_at_mut(&mut v, 3);
    left[0] = 10;
    right[0] = 40;
    println!("{:?}", v); // [10, 2, 3, 40, 5, 6]
}
```

---

## 3. FFI (Foreign Function Interface)

### 例4: C関数の呼び出し

```rust
extern "C" {
    fn abs(input: i32) -> i32;
    fn strlen(s: *const std::os::raw::c_char) -> usize;
}

fn main() {
    unsafe {
        println!("abs(-5) = {}", abs(-5));

        let s = std::ffi::CString::new("hello").unwrap();
        println!("strlen = {}", strlen(s.as_ptr()));
    }
}
```

### 例5: RustからCに関数を公開

```rust
/// C から呼び出し可能な関数
#[no_mangle]
pub extern "C" fn rust_add(a: i32, b: i32) -> i32 {
    a + b
}

/// C のコールバックを受け取る
#[no_mangle]
pub extern "C" fn process_with_callback(
    data: *const i32,
    len: usize,
    callback: extern "C" fn(i32) -> i32,
) -> i32 {
    let mut sum = 0;
    unsafe {
        for i in 0..len {
            sum += callback(*data.add(i));
        }
    }
    sum
}
```

### FFI の安全性境界

```
  ┌────────────────────────────────┐
  │  Rust (安全な世界)             │
  │                                │
  │  pub fn safe_api(input: &str)  │
  │     │                          │
  │     ▼                          │
  │  ┌──────────────────────────┐  │
  │  │ unsafe {                 │  │
  │  │   // 入力の検証          │  │
  │  │   // CString への変換    │  │
  │  │   // C関数の呼び出し     │  │
  │  │   // 結果の検証          │  │
  │  │   // Rust型への変換      │  │
  │  │ }                        │  │
  │  └──────────────────────────┘  │
  │     │                          │
  │     ▼                          │
  │  Result<T, E> を返す           │
  └────────────────────────────────┘
           │
           ▼
  ┌────────────────────────────────┐
  │  C ライブラリ (unsafe な世界)  │
  └────────────────────────────────┘
```

---

## 4. unsafe トレイト

### 例6: unsafe トレイトの実装

```rust
// Send / Sync は unsafe トレイト
// コンパイラが自動実装するが、手動実装も可能

struct MyWrapper(*mut i32);

// Safety: MyWrapper は内部のポインタを適切に管理し、
// スレッド間で安全に送信できることを保証する
unsafe impl Send for MyWrapper {}
unsafe impl Sync for MyWrapper {}

// カスタム unsafe トレイト
/// # Safety
/// 実装者は `validate()` が true を返す場合のみ
/// `process()` を呼び出すことを保証しなければならない
unsafe trait Validated {
    fn validate(&self) -> bool;
    fn process(&self);
}
```

---

## 5. 可変 static 変数と union

### 例7: 可変 static と union

```rust
// 可変 static (グローバル変数)
static mut COUNTER: u32 = 0;

fn increment_counter() {
    unsafe {
        COUNTER += 1;
    }
}

fn get_counter() -> u32 {
    unsafe { COUNTER }
}

// union: 同じメモリ領域を異なる型として解釈
#[repr(C)]
union IntOrFloat {
    i: i32,
    f: f32,
}

fn main() {
    increment_counter();
    increment_counter();
    println!("カウンタ: {}", get_counter());

    let u = IntOrFloat { f: 1.0 };
    unsafe {
        println!("float として: {}", u.f);
        println!("int として: {:#010x}", u.i); // IEEE 754 表現
    }
}
```

---

## 6. 安全な抽象化パターン

```
┌────────────────────────────────────────────────────────┐
│             安全な抽象化の原則                           │
├────────────────────────────────────────────────────────┤
│                                                        │
│ 1. unsafe を最小限のブロックに限定する                  │
│ 2. 安全なAPIを公開し、unsafe を内部に隠蔽する          │
│ 3. # Safety ドキュメントで前提条件を明記する            │
│ 4. debug_assert! で不変条件を検証する                   │
│ 5. Miri でundefined behaviorを検出する                  │
│                                                        │
│  ┌──────────────────────────────────┐                  │
│  │  pub fn safe_function(...)       │  ← ユーザーが見る│
│  │    → 入力検証                    │                  │
│  │    → unsafe { ... }             │  ← 内部に隠蔽   │
│  │    → 結果検証                    │                  │
│  │    → 安全な型で返す              │                  │
│  └──────────────────────────────────┘                  │
└────────────────────────────────────────────────────────┘
```

---

## 7. 比較表

### 7.1 safe vs unsafe のスコープ

| 操作 | safe Rust | unsafe Rust |
|------|-----------|-------------|
| 参照のデリファレンス | 常に安全 | 生ポインタで可能 |
| 配列アクセス | 境界チェック付き | get_unchecked で省略可 |
| 型変換 | From/Into | transmute で任意変換 |
| FFI | 不可 | extern ブロックで可能 |
| グローバル可変状態 | 不可 | static mut で可能 |
| ライフタイム | コンパイラが検証 | プログラマが保証 |

### 7.2 unsafe の代替手段

| したいこと | unsafe を使う前に検討 | unsafe が必要な場合 |
|-----------|---------------------|-------------------|
| 複数の可変参照 | RefCell, Mutex | split_at_mut 的な分割 |
| グローバル状態 | OnceCell, AtomicU32 | static mut (非推奨) |
| 型の再解釈 | From/TryFrom | transmute |
| C連携 | 安全なラッパークレート | 直接 extern "C" |
| パフォーマンス | アルゴリズム改善 | get_unchecked 等 |

---

## 8. アンチパターン

### アンチパターン1: 広すぎる unsafe ブロック

```rust
// BAD: unsafe ブロックが大きすぎる
unsafe {
    let data = fetch_data();           // 安全な操作
    let processed = process(data);      // 安全な操作
    let ptr = processed.as_ptr();       // 安全な操作
    let value = *ptr;                   // ← これだけが unsafe
    validate(value);                    // 安全な操作
}

// GOOD: unsafe を最小限に
let data = fetch_data();
let processed = process(data);
let ptr = processed.as_ptr();
let value = unsafe { *ptr };  // unsafe は本当に必要な部分のみ
validate(value);
```

### アンチパターン2: Safety ドキュメントの欠如

```rust
// BAD: なぜ unsafe なのか説明がない
pub unsafe fn do_thing(ptr: *const u8, len: usize) {
    // ...
}

// GOOD: 前提条件を明記
/// バッファからデータを読み取る。
///
/// # Safety
///
/// - `ptr` は有効な読み取り可能なメモリ領域を指していなければならない
/// - `ptr` から `len` バイト分のメモリが有効でなければならない
/// - メモリはこの関数の実行中、他のスレッドから変更されてはならない
pub unsafe fn do_thing_good(ptr: *const u8, len: usize) {
    // ...
}
```

---

## 9. FAQ

### Q1: unsafe を使うとRustの安全性が失われますか？

**A:** 部分的にです。unsafe ブロック内ではメモリ安全性の責任がプログラマに移りますが、unsafe の外側ではコンパイラの保証は維持されます。重要なのは unsafe を最小限にして安全な抽象化で包むことです。

### Q2: Miri とは何ですか？

**A:** Miri は Rust の中間表現(MIR)インタプリタで、未定義動作を検出するツールです:
```bash
rustup +nightly component add miri
cargo +nightly miri test
```
メモリリーク、データ競合、不正なポインタ操作などを実行時に検出します。

### Q3: transmute はいつ使いますか？

**A:** ほとんどの場合、使うべきではありません。`transmute` はビットパターンをそのまま別の型として解釈するため、非常に危険です。代わりに `as` キャスト、`From/TryFrom`、`bytemuck` クレートの使用を検討してください。

---

## 10. まとめ

| 概念 | 要点 |
|------|------|
| unsafe ブロック | 5つの超能力を解禁。最小限に抑える |
| 生ポインタ | *const T / *mut T。デリファレンスが unsafe |
| FFI | extern "C" でC言語関数を呼び出し/公開 |
| unsafe トレイト | Send/Sync 等。実装者が安全性を保証 |
| static mut | グローバル可変状態。データ競合の危険 |
| 安全な抽象化 | unsafe を内部に封じ、安全なAPIを公開 |
| Miri | 未定義動作を検出するツール |

---

## 次に読むべきガイド

- [04-macros.md](04-macros.md) -- マクロで安全な抽象化を生成
- [../03-systems/02-ffi-interop.md](../03-systems/02-ffi-interop.md) -- FFI の実践(bindgen, PyO3)
- [../03-systems/00-memory-layout.md](../03-systems/00-memory-layout.md) -- メモリレイアウトの詳細

---

## 参考文献

1. **The Rust Programming Language - Ch.19.1 Unsafe Rust** -- https://doc.rust-lang.org/book/ch19-01-unsafe-rust.html
2. **The Rustonomicon** -- https://doc.rust-lang.org/nomicon/
3. **Miri - An Interpreter for Rust's MIR** -- https://github.com/rust-lang/miri
4. **Rust Unsafe Code Guidelines** -- https://rust-lang.github.io/unsafe-code-guidelines/
