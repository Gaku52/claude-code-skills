# unsafe Rust -- 安全性の境界を越える低レベルプログラミング

> unsafe Rustは借用チェッカーの制約を超えた操作(生ポインタ操作、FFI、ハードウェアアクセス等)を可能にし、安全な抽象化の中に封じ込めることで高レベルAPIの安全性を維持する。

---

## この章で学ぶこと

1. **unsafe の5つの超能力** -- unsafe ブロック内で許可される5つの操作を理解する
2. **生ポインタとFFI** -- *const T / *mut T の操作とC言語連携の方法を習得する
3. **安全な抽象化** -- unsafe を内部に封じ込めて安全なAPIを公開するパターンを学ぶ
4. **unsafe トレイト** -- Send / Sync の手動実装と独自 unsafe トレイトの設計を理解する
5. **未定義動作と Miri** -- UB の種類と検出ツールの使い方を習得する

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
│  unsafe は「コンパイラが検証できない安全性の責任を    │
│  プログラマが引き受ける」という意味                    │
└──────────────────────────────────────────────────────┘
```

### 1.1 unsafe が「しないこと」

unsafe ブロックは以下のチェックを無効にしない:
- **所有権ルール**: ムーブ後の使用は引き続きエラー
- **型チェック**: 型の不一致は引き続きエラー
- **借用ルール**: `&T` と `&mut T` の同時使用は引き続きエラー(ただし生ポインタを使えば回避可能)
- **ライフタイムチェック**: 参照のライフタイムは引き続き検証される

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

    // null ポインタ
    let null_ptr: *const i32 = std::ptr::null();
    println!("null ポインタ: {:?}", null_ptr);
    assert!(null_ptr.is_null());

    // 生ポインタの算術
    let arr = [10, 20, 30, 40, 50];
    let ptr = arr.as_ptr();
    unsafe {
        for i in 0..arr.len() {
            println!("arr[{}] = {}", i, *ptr.add(i));
        }
    }
}
```

### 例2: 生ポインタと参照の変換

```rust
fn main() {
    let mut value = 42;

    // 参照 → 生ポインタ (安全)
    let raw_const: *const i32 = &value;
    let raw_mut: *mut i32 = &mut value;

    // 生ポインタ → 参照 (unsafe)
    unsafe {
        let ref_const: &i32 = &*raw_const;
        let ref_mut: &mut i32 = &mut *raw_mut;
        println!("const ref: {}", ref_const);
        *ref_mut = 100;
        println!("mut ref: {}", ref_mut);
    }

    // NonNull: null でないことが保証される生ポインタ
    let non_null = std::ptr::NonNull::new(&mut value as *mut i32).unwrap();
    unsafe {
        println!("NonNull: {}", *non_null.as_ptr());
    }

    // スライスの生ポインタ
    let slice = &[1, 2, 3, 4, 5];
    let ptr = slice.as_ptr();
    let len = slice.len();
    unsafe {
        // 生ポインタとスライスの再構築
        let reconstructed = std::slice::from_raw_parts(ptr, len);
        println!("再構築: {:?}", reconstructed);
    }
}
```

---

## 2. unsafe 関数

### 例3: unsafe 関数の定義と呼び出し

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

### 例4: split_at_mut の実装(標準ライブラリの内部)

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
    println!("{:?}", left);  // [10, 2, 3]
    println!("{:?}", right); // [40, 5, 6]
}
```

### 例5: unsafe を使ったゼロコスト型変換

```rust
// repr(transparent) で内部表現が同一であることを保証
#[repr(transparent)]
struct Meters(f64);

#[repr(transparent)]
struct Kilometers(f64);

impl Meters {
    fn new(value: f64) -> Self {
        Meters(value)
    }

    fn to_kilometers(&self) -> f64 {
        self.0 / 1000.0
    }
}

// スライスの型変換 (ゼロコスト)
fn meters_slice_to_f64(meters: &[Meters]) -> &[f64] {
    // Safety: Meters は repr(transparent) で f64 と同じレイアウト
    unsafe {
        std::slice::from_raw_parts(
            meters.as_ptr() as *const f64,
            meters.len(),
        )
    }
}

fn main() {
    let measurements = vec![Meters::new(1000.0), Meters::new(2500.0), Meters::new(500.0)];
    let raw_values = meters_slice_to_f64(&measurements);
    println!("生の値: {:?}", raw_values); // [1000.0, 2500.0, 500.0]

    // std::mem::transmute の安全な代替
    let x: u32 = 0x41424344;
    let bytes: [u8; 4] = x.to_ne_bytes();
    println!("バイト: {:?}", bytes);
}
```

---

## 3. FFI (Foreign Function Interface)

### 例6: C関数の呼び出し

```rust
extern "C" {
    fn abs(input: i32) -> i32;
    fn strlen(s: *const std::os::raw::c_char) -> usize;
    fn memcpy(
        dest: *mut std::os::raw::c_void,
        src: *const std::os::raw::c_void,
        n: usize,
    ) -> *mut std::os::raw::c_void;
}

fn main() {
    unsafe {
        // abs の呼び出し
        println!("abs(-5) = {}", abs(-5));
        println!("abs(10) = {}", abs(10));

        // strlen の呼び出し
        let s = std::ffi::CString::new("hello").unwrap();
        println!("strlen = {}", strlen(s.as_ptr()));

        // memcpy の呼び出し
        let src = [1u8, 2, 3, 4, 5];
        let mut dest = [0u8; 5];
        memcpy(
            dest.as_mut_ptr() as *mut std::os::raw::c_void,
            src.as_ptr() as *const std::os::raw::c_void,
            src.len(),
        );
        println!("memcpy結果: {:?}", dest); // [1, 2, 3, 4, 5]
    }
}
```

### 例7: RustからCに関数を公開

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

/// 文字列を受け取って処理する
#[no_mangle]
pub extern "C" fn rust_process_string(
    input: *const std::os::raw::c_char,
) -> *mut std::os::raw::c_char {
    unsafe {
        if input.is_null() {
            return std::ptr::null_mut();
        }
        let c_str = std::ffi::CStr::from_ptr(input);
        let rust_str = c_str.to_str().unwrap_or("invalid utf8");
        let processed = format!("Processed: {}", rust_str.to_uppercase());
        let c_string = std::ffi::CString::new(processed).unwrap();
        c_string.into_raw() // 呼び出し側が rust_free_string で解放する必要がある
    }
}

/// Rust で確保した文字列を解放する
#[no_mangle]
pub extern "C" fn rust_free_string(s: *mut std::os::raw::c_char) {
    if !s.is_null() {
        unsafe {
            let _ = std::ffi::CString::from_raw(s);
            // CString が drop される → メモリ解放
        }
    }
}

fn main() {
    println!("rust_add(3, 4) = {}", rust_add(3, 4));
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

### 例8: FFI 安全ラッパーの実装

```rust
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

// C ライブラリの関数を宣言
extern "C" {
    fn setenv(name: *const c_char, value: *const c_char, overwrite: i32) -> i32;
    fn getenv(name: *const c_char) -> *const c_char;
}

// 安全なラッパー
mod safe_env {
    use super::*;

    #[derive(Debug)]
    pub enum EnvError {
        InvalidName(String),
        InvalidValue(String),
        SetFailed,
    }

    impl std::fmt::Display for EnvError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                EnvError::InvalidName(s) => write!(f, "無効な環境変数名: {}", s),
                EnvError::InvalidValue(s) => write!(f, "無効な値: {}", s),
                EnvError::SetFailed => write!(f, "環境変数の設定に失敗"),
            }
        }
    }

    /// 環境変数を安全に設定する
    pub fn set_env(name: &str, value: &str) -> Result<(), EnvError> {
        // 入力の検証
        if name.contains('\0') {
            return Err(EnvError::InvalidName(name.to_string()));
        }
        if value.contains('\0') {
            return Err(EnvError::InvalidValue(value.to_string()));
        }

        let c_name = CString::new(name)
            .map_err(|_| EnvError::InvalidName(name.to_string()))?;
        let c_value = CString::new(value)
            .map_err(|_| EnvError::InvalidValue(value.to_string()))?;

        // unsafe ブロックを最小限に
        let result = unsafe { setenv(c_name.as_ptr(), c_value.as_ptr(), 1) };

        if result == 0 {
            Ok(())
        } else {
            Err(EnvError::SetFailed)
        }
    }

    /// 環境変数を安全に取得する
    pub fn get_env(name: &str) -> Option<String> {
        let c_name = CString::new(name).ok()?;

        unsafe {
            let ptr = getenv(c_name.as_ptr());
            if ptr.is_null() {
                None
            } else {
                CStr::from_ptr(ptr)
                    .to_str()
                    .ok()
                    .map(|s| s.to_string())
            }
        }
    }
}

fn main() {
    // 安全なAPIを通して使用
    match safe_env::set_env("MY_VAR", "hello") {
        Ok(()) => println!("環境変数を設定しました"),
        Err(e) => println!("エラー: {}", e),
    }

    if let Some(value) = safe_env::get_env("MY_VAR") {
        println!("MY_VAR = {}", value);
    }
}
```

---

## 4. unsafe トレイト

### 例9: unsafe トレイトの実装

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

struct SafeData {
    value: i32,
}

unsafe impl Validated for SafeData {
    fn validate(&self) -> bool {
        self.value > 0
    }

    fn process(&self) {
        // validate() が true の場合のみ呼ばれることを前提とする
        println!("処理中: {}", self.value);
    }
}

fn execute_validated<T: Validated>(item: &T) {
    if item.validate() {
        item.process();
    } else {
        println!("バリデーション失敗");
    }
}

fn main() {
    let data = SafeData { value: 42 };
    execute_validated(&data);

    let invalid = SafeData { value: -1 };
    execute_validated(&invalid);
}
```

### 例10: Send / Sync の理解と手動実装

```rust
use std::cell::UnsafeCell;

// UnsafeCell を使ったスレッド安全な型の実装
struct ThreadSafeCounter {
    count: UnsafeCell<u64>,
}

// Safety: 内部のアクセスは atomic 操作で保護する
// (この例では簡略化のため Mutex 的な保護は省略)
unsafe impl Send for ThreadSafeCounter {}
unsafe impl Sync for ThreadSafeCounter {}

impl ThreadSafeCounter {
    fn new() -> Self {
        ThreadSafeCounter {
            count: UnsafeCell::new(0),
        }
    }

    // 注意: この実装はデモ目的。実際にはアトミック操作が必要
    fn get(&self) -> u64 {
        unsafe { *self.count.get() }
    }
}

// Send / Sync が自動実装されない型の例
struct NotSend {
    data: *mut i32,  // 生ポインタは Send/Sync を実装しない
}

// Rc は Send を実装しない(参照カウントがスレッド安全でないため)
// use std::rc::Rc;
// fn send_rc<T: Send>(t: T) {} // Rc<T> を渡すとコンパイルエラー

fn main() {
    let counter = ThreadSafeCounter::new();
    println!("カウント: {}", counter.get());
}
```

---

## 5. 可変 static 変数と union

### 例11: 可変 static と union

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

### 例12: static mut の安全な代替手段

```rust
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

// 方法1: AtomicU64 (アトミック操作)
static ATOMIC_COUNTER: AtomicU64 = AtomicU64::new(0);

fn increment_atomic() {
    ATOMIC_COUNTER.fetch_add(1, Ordering::Relaxed);
}

fn get_atomic() -> u64 {
    ATOMIC_COUNTER.load(Ordering::Relaxed)
}

// 方法2: OnceLock (初期化が一度だけの値)
static CONFIG: OnceLock<String> = OnceLock::new();

fn get_config() -> &'static str {
    CONFIG.get_or_init(|| {
        // 初期化処理(一度だけ実行される)
        String::from("production")
    })
}

// 方法3: Mutex (一般的な可変グローバル状態)
use std::sync::Mutex;

static GLOBAL_STATE: Mutex<Vec<String>> = Mutex::new(Vec::new());

fn add_to_state(item: String) {
    GLOBAL_STATE.lock().unwrap().push(item);
}

fn get_state() -> Vec<String> {
    GLOBAL_STATE.lock().unwrap().clone()
}

fn main() {
    // AtomicU64
    increment_atomic();
    increment_atomic();
    increment_atomic();
    println!("アトミックカウンタ: {}", get_atomic());

    // OnceLock
    println!("設定: {}", get_config());
    println!("設定(2回目): {}", get_config()); // 同じ値

    // Mutex
    add_to_state("hello".to_string());
    add_to_state("world".to_string());
    println!("グローバル状態: {:?}", get_state());
}
```

### 例13: union の実用例

```rust
// ネットワークプロトコルでの union の使用
#[repr(C)]
union IpAddress {
    v4: [u8; 4],
    v6: [u8; 16],
}

struct NetworkPacket {
    is_v6: bool,
    addr: IpAddress,
}

impl NetworkPacket {
    fn new_v4(addr: [u8; 4]) -> Self {
        NetworkPacket {
            is_v6: false,
            addr: IpAddress { v4: addr },
        }
    }

    fn new_v6(addr: [u8; 16]) -> Self {
        NetworkPacket {
            is_v6: true,
            addr: IpAddress { v6: addr },
        }
    }

    fn display_addr(&self) {
        unsafe {
            if self.is_v6 {
                println!("IPv6: {:?}", self.addr.v6);
            } else {
                println!("IPv4: {}.{}.{}.{}",
                    self.addr.v4[0], self.addr.v4[1],
                    self.addr.v4[2], self.addr.v4[3]);
            }
        }
    }
}

// MaybeUninit: 初期化されていないメモリの安全な扱い
fn demo_maybe_uninit() {
    use std::mem::MaybeUninit;

    // 初期化されていない配列を効率的に構築
    let mut arr: [MaybeUninit<String>; 5] = unsafe {
        MaybeUninit::uninit().assume_init()
    };

    for (i, elem) in arr.iter_mut().enumerate() {
        elem.write(format!("item_{}", i));
    }

    // 安全に初期化済み配列に変換
    let arr: [String; 5] = unsafe {
        // transmute で MaybeUninit<String> → String に変換
        std::mem::transmute::<[MaybeUninit<String>; 5], [String; 5]>(arr)
    };

    for item in &arr {
        println!("  {}", item);
    }
}

fn main() {
    let pkt_v4 = NetworkPacket::new_v4([192, 168, 1, 1]);
    let pkt_v6 = NetworkPacket::new_v6([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]);

    pkt_v4.display_addr();
    pkt_v6.display_addr();

    println!("--- MaybeUninit デモ ---");
    demo_maybe_uninit();
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

### 例14: 安全な抽象化の完全な例

```rust
/// 固定サイズのリングバッファ
///
/// 内部で unsafe を使用しているが、公開APIは完全に安全。
pub struct RingBuffer<T> {
    buffer: Box<[std::mem::MaybeUninit<T>]>,
    head: usize,
    tail: usize,
    len: usize,
    capacity: usize,
}

impl<T> RingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "容量は1以上");
        let buffer = (0..capacity)
            .map(|_| std::mem::MaybeUninit::uninit())
            .collect::<Vec<_>>()
            .into_boxed_slice();
        RingBuffer {
            buffer,
            head: 0,
            tail: 0,
            len: 0,
            capacity,
        }
    }

    pub fn push(&mut self, value: T) -> Option<T> {
        let old = if self.len == self.capacity {
            // バッファが満杯: 最古の要素を取り出す
            let old = unsafe { self.buffer[self.head].assume_init_read() };
            self.head = (self.head + 1) % self.capacity;
            self.len -= 1;
            Some(old)
        } else {
            None
        };

        self.buffer[self.tail].write(value);
        self.tail = (self.tail + 1) % self.capacity;
        self.len += 1;
        old
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            let value = unsafe { self.buffer[self.head].assume_init_read() };
            self.head = (self.head + 1) % self.capacity;
            self.len -= 1;
            Some(value)
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl<T> Drop for RingBuffer<T> {
    fn drop(&mut self) {
        // 初期化済みの要素を正しく drop する
        while self.pop().is_some() {}
    }
}

fn main() {
    let mut buf = RingBuffer::new(3);
    buf.push(1);
    buf.push(2);
    buf.push(3);
    println!("バッファサイズ: {}", buf.len()); // 3

    let evicted = buf.push(4); // 1 が追い出される
    println!("追い出された値: {:?}", evicted); // Some(1)

    while let Some(val) = buf.pop() {
        println!("取り出し: {}", val); // 2, 3, 4
    }
}
```

### 例15: unsafe を使った高性能な文字列処理

```rust
/// UTF-8 バリデーションをスキップして String を構築する。
/// 呼び出し側がUTF-8の妥当性を保証する場合にのみ使用。
///
/// # Safety
/// `bytes` は有効なUTF-8でなければならない。
unsafe fn string_from_utf8_unchecked(bytes: Vec<u8>) -> String {
    String::from_utf8_unchecked(bytes)
}

/// ASCII文字列を大文字に変換する（インプレース）
/// 標準の to_uppercase() より高速（アロケーションなし）
fn ascii_uppercase_inplace(s: &mut String) {
    // Safety: ASCII の大文字変換はUTF-8の妥当性を保持する
    // (ASCII文字は1バイトで、大文字変換後もASCIIの範囲内)
    unsafe {
        let bytes = s.as_bytes_mut();
        for byte in bytes.iter_mut() {
            if *byte >= b'a' && *byte <= b'z' {
                *byte -= 32; // 'a' - 'A' = 32
            }
        }
    }
}

/// バイト列から部分文字列を安全に抽出する
pub fn safe_substring(s: &str, start: usize, end: usize) -> Option<&str> {
    if start > end || end > s.len() {
        return None;
    }

    // UTF-8 の境界チェック
    if !s.is_char_boundary(start) || !s.is_char_boundary(end) {
        return None;
    }

    // Safety: 境界チェック済み
    Some(unsafe { s.get_unchecked(start..end) })
}

fn main() {
    // ASCII 大文字変換
    let mut s = String::from("hello, world!");
    ascii_uppercase_inplace(&mut s);
    println!("大文字: {}", s); // HELLO, WORLD!

    // 安全な部分文字列抽出
    let text = "こんにちは、世界！";
    match safe_substring(text, 0, 15) {
        Some(sub) => println!("部分文字列: {}", sub),
        None => println!("無効な範囲"),
    }

    // UTF-8 バイト境界エラーの検出
    match safe_substring(text, 0, 1) {
        Some(sub) => println!("部分文字列: {}", sub),
        None => println!("UTF-8 境界エラー"), // 日本語は3バイト/文字
    }

    match safe_substring(text, 0, 3) {
        Some(sub) => println!("部分文字列: {}", sub), // "こ"
        None => println!("UTF-8 境界エラー"),
    }
}
```

---

## 7. 未定義動作 (Undefined Behavior)

### 7.1 Rust における未定義動作の種類

```
┌────────────────────────────────────────────────────────┐
│          未定義動作 (UB) の主な種類                       │
├────────────────────────────────────────────────────────┤
│                                                        │
│ 1. ダングリングポインタのデリファレンス                  │
│ 2. 不正なアラインメントのポインタデリファレンス          │
│ 3. データ競合 (data race)                               │
│ 4. 無効な値の生成 (例: bool に 2 を入れる)              │
│ 5. null 参照の生成                                      │
│ 6. 初期化されていないメモリの読み取り                    │
│ 7. aliasing ルール違反 (&T と &mut T の同時存在)        │
│ 8. 不正な enum 判別子                                   │
│ 9. unwinding を extern "C" 関数の境界を越えて伝播      │
│ 10. UTF-8 でない &str の生成                            │
│                                                        │
│ UB が起きると:                                          │
│ - コンパイラの最適化が誤った結果を生む                   │
│ - クラッシュ、データ破壊、セキュリティ脆弱性            │
│ - 「今は動いているが将来壊れる」コード                   │
└────────────────────────────────────────────────────────┘
```

### 例16: 典型的な UB のパターン

```rust
fn demonstrate_ub_patterns() {
    // UB パターン1: ダングリングポインタ
    // let ptr: *const i32;
    // {
    //     let x = 42;
    //     ptr = &x;
    // }
    // unsafe { println!("{}", *ptr); }  // UB!

    // UB パターン2: &T と &mut T の同時存在
    // let mut x = 42;
    // let r1 = &x as *const i32;
    // let r2 = &mut x as *mut i32;
    // unsafe {
    //     *r2 = 100;
    //     println!("{}", *r1);  // UB! aliasing violation
    // }

    // UB パターン3: 不正な値
    // let b: bool = unsafe { std::mem::transmute(2u8) };  // UB! bool は 0 or 1 のみ

    // UB パターン4: アラインメント違反
    // let bytes = [0u8; 8];
    // let ptr = bytes.as_ptr().add(1) as *const u32;
    // unsafe { println!("{}", *ptr); }  // UB! アラインメント違反

    // 安全な代替手段
    let bytes = [0u8; 8];
    let value = u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    println!("安全な読み取り: {}", value);
}

fn main() {
    demonstrate_ub_patterns();
}
```

### 7.2 Miri による UB 検出

```bash
# Miri のインストール
rustup +nightly component add miri

# テストを Miri で実行
cargo +nightly miri test

# 特定のテストだけ実行
cargo +nightly miri test test_name

# バイナリを Miri で実行
cargo +nightly miri run
```

```rust
// Miri で検出できる問題の例
#[cfg(test)]
mod tests {
    #[test]
    fn test_valid_code() {
        let mut v = vec![1, 2, 3];
        let ptr = v.as_ptr();
        v.push(4);
        // Miri: ptr はもはや有効でない可能性がある
        // (push で再アロケーションが起きた場合)
    }

    #[test]
    fn test_safe_access() {
        let v = vec![1, 2, 3];
        let ptr = v.as_ptr();
        unsafe {
            // push していないので ptr はまだ有効
            assert_eq!(*ptr, 1);
            assert_eq!(*ptr.add(1), 2);
            assert_eq!(*ptr.add(2), 3);
        }
    }
}
```

---

## 8. 比較表

### 8.1 safe vs unsafe のスコープ

| 操作 | safe Rust | unsafe Rust |
|------|-----------|-------------|
| 参照のデリファレンス | 常に安全 | 生ポインタで可能 |
| 配列アクセス | 境界チェック付き | get_unchecked で省略可 |
| 型変換 | From/Into | transmute で任意変換 |
| FFI | 不可 | extern ブロックで可能 |
| グローバル可変状態 | 不可 | static mut で可能 |
| ライフタイム | コンパイラが検証 | プログラマが保証 |
| union アクセス | 不可 | unsafe ブロック内で可能 |

### 8.2 unsafe の代替手段

| したいこと | unsafe を使う前に検討 | unsafe が必要な場合 |
|-----------|---------------------|-------------------|
| 複数の可変参照 | RefCell, Mutex | split_at_mut 的な分割 |
| グローバル状態 | OnceLock, AtomicU32 | static mut (非推奨) |
| 型の再解釈 | From/TryFrom | transmute |
| C連携 | 安全なラッパークレート | 直接 extern "C" |
| パフォーマンス | アルゴリズム改善 | get_unchecked 等 |
| 自己参照型 | Pin, ouroboros クレート | 生ポインタ |
| 初期化遅延 | OnceLock, LazyLock | MaybeUninit |

### 8.3 repr 属性の比較

| 属性 | 意味 | 用途 |
|------|------|------|
| `#[repr(Rust)]` | デフォルト。コンパイラが自由にレイアウト | 通常の構造体 |
| `#[repr(C)]` | C言語互換のレイアウト | FFI、union |
| `#[repr(transparent)]` | 内部型と同じレイアウト | ニュータイプパターン |
| `#[repr(packed)]` | パディングなし | 省メモリ(アラインメント注意) |
| `#[repr(align(N))]` | N バイトアラインメント | SIMD、キャッシュライン |

---

## 9. アンチパターン

### アンチパターン1: 広すぎる unsafe ブロック

```rust
// BAD: unsafe ブロックが大きすぎる
fn bad_example(data: &[u8]) -> u8 {
    unsafe {
        let processed = data.iter().sum::<u8>();  // 安全な操作
        let ptr = data.as_ptr();                   // 安全な操作
        let value = *ptr;                          // ← これだけが unsafe
        processed.wrapping_add(value)              // 安全な操作
    }
}

// GOOD: unsafe を最小限に
fn good_example(data: &[u8]) -> u8 {
    let processed: u8 = data.iter().sum();
    let ptr = data.as_ptr();
    let value = unsafe { *ptr };  // unsafe は本当に必要な部分のみ
    processed.wrapping_add(value)
}

fn main() {
    let data = vec![1u8, 2, 3, 4, 5];
    println!("bad: {}", bad_example(&data));
    println!("good: {}", good_example(&data));
}
```

### アンチパターン2: Safety ドキュメントの欠如

```rust
// BAD: なぜ unsafe なのか説明がない
pub unsafe fn do_thing_bad(ptr: *const u8, len: usize) -> Vec<u8> {
    std::slice::from_raw_parts(ptr, len).to_vec()
}

// GOOD: 前提条件を明記
/// バッファからデータを読み取る。
///
/// # Safety
///
/// - `ptr` は有効な読み取り可能なメモリ領域を指していなければならない
/// - `ptr` から `len` バイト分のメモリが有効でなければならない
/// - メモリはこの関数の実行中、他のスレッドから変更されてはならない
/// - `ptr` のアラインメントは u8 に対して正しくなければならない
pub unsafe fn do_thing_good(ptr: *const u8, len: usize) -> Vec<u8> {
    debug_assert!(!ptr.is_null(), "null ポインタが渡されました");
    std::slice::from_raw_parts(ptr, len).to_vec()
}

fn main() {
    let data = vec![1u8, 2, 3, 4, 5];
    let result = unsafe { do_thing_good(data.as_ptr(), data.len()) };
    println!("{:?}", result);
}
```

### アンチパターン3: transmute の乱用

```rust
// BAD: transmute で型安全性を破壊
fn bad_transmute() {
    // let x: f32 = unsafe { std::mem::transmute(0x42280000u32) };
    // 危険: エンディアン依存、将来の repr 変更で壊れる
}

// GOOD: 安全な型変換メソッドを使う
fn good_conversion() {
    // f32 のビットパターンを取得
    let x: f32 = 42.0;
    let bits = x.to_bits();
    println!("f32 {} のビット: {:#010x}", x, bits);

    // ビットパターンから f32 を復元
    let restored = f32::from_bits(bits);
    println!("復元: {}", restored);

    // バイト列との変換
    let bytes = x.to_ne_bytes();
    let from_bytes = f32::from_ne_bytes(bytes);
    println!("バイト経由: {}", from_bytes);
}

// BAD: enum の transmute
// fn bad_enum() {
//     let x: Option<i32> = unsafe { std::mem::transmute(42i32) };
//     // UB! Option<i32> の内部表現は保証されていない
// }

fn main() {
    good_conversion();
}
```

### アンチパターン4: unsafe で借用チェックを回避

```rust
// BAD: unsafe で借用ルールを回避しようとする
fn bad_alias() {
    let mut data = vec![1, 2, 3];
    // let ptr = data.as_mut_ptr();
    // let r1 = &data; // 不変参照
    // unsafe { *ptr = 42; } // 可変アクセス → UB!
    // println!("{:?}", r1); // r1 はまだ使われている
}

// GOOD: 安全なAPIを使う
fn good_design() {
    let mut data = vec![1, 2, 3];

    // Cell/RefCell で内部可変性
    use std::cell::RefCell;
    let data = RefCell::new(vec![1, 2, 3]);
    {
        let r1 = data.borrow();
        println!("{:?}", r1);
    } // r1 をドロップ
    data.borrow_mut()[0] = 42;
    println!("{:?}", data.borrow());
}

fn main() {
    bad_alias();
    good_design();
}
```

---

## 10. 実践: unsafe を使った高性能データ構造

### 例17: 侵入型リンクリスト

```rust
use std::ptr;

struct Node<T> {
    value: T,
    next: *mut Node<T>,
    prev: *mut Node<T>,
}

pub struct LinkedList<T> {
    head: *mut Node<T>,
    tail: *mut Node<T>,
    len: usize,
}

impl<T> LinkedList<T> {
    pub fn new() -> Self {
        LinkedList {
            head: ptr::null_mut(),
            tail: ptr::null_mut(),
            len: 0,
        }
    }

    pub fn push_back(&mut self, value: T) {
        let node = Box::into_raw(Box::new(Node {
            value,
            next: ptr::null_mut(),
            prev: self.tail,
        }));

        if self.tail.is_null() {
            self.head = node;
        } else {
            unsafe {
                (*self.tail).next = node;
            }
        }
        self.tail = node;
        self.len += 1;
    }

    pub fn push_front(&mut self, value: T) {
        let node = Box::into_raw(Box::new(Node {
            value,
            next: self.head,
            prev: ptr::null_mut(),
        }));

        if self.head.is_null() {
            self.tail = node;
        } else {
            unsafe {
                (*self.head).prev = node;
            }
        }
        self.head = node;
        self.len += 1;
    }

    pub fn pop_front(&mut self) -> Option<T> {
        if self.head.is_null() {
            return None;
        }

        unsafe {
            let node = Box::from_raw(self.head);
            self.head = node.next;
            if self.head.is_null() {
                self.tail = ptr::null_mut();
            } else {
                (*self.head).prev = ptr::null_mut();
            }
            self.len -= 1;
            Some(node.value)
        }
    }

    pub fn pop_back(&mut self) -> Option<T> {
        if self.tail.is_null() {
            return None;
        }

        unsafe {
            let node = Box::from_raw(self.tail);
            self.tail = node.prev;
            if self.tail.is_null() {
                self.head = ptr::null_mut();
            } else {
                (*self.tail).next = ptr::null_mut();
            }
            self.len -= 1;
            Some(node.value)
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T> Drop for LinkedList<T> {
    fn drop(&mut self) {
        while self.pop_front().is_some() {}
    }
}

fn main() {
    let mut list = LinkedList::new();
    list.push_back(1);
    list.push_back(2);
    list.push_back(3);
    list.push_front(0);

    println!("リストサイズ: {}", list.len());

    while let Some(val) = list.pop_front() {
        println!("  {}", val);
    }
}
```

---

## 11. FAQ

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

**A:** ほとんどの場合、使うべきではありません。`transmute` はビットパターンをそのまま別の型として解釈するため、非常に危険です。代わりに `as` キャスト、`From/TryFrom`、`to_bits/from_bits`、`to_ne_bytes/from_ne_bytes`、`bytemuck` クレートの使用を検討してください。

### Q4: `#[repr(C)]` はいつ必要ですか？

**A:** 主に以下の場面で必要です:
- FFI でC言語とデータをやり取りする場合
- メモリレイアウトを明示的に制御したい場合
- union を使用する場合
- ファイルフォーマットやネットワークプロトコルの構造体定義

### Q5: unsafe fn vs unsafe ブロック、どちらを使うべきですか？

**A:** 以下の基準で選択します:
- **unsafe fn**: 呼び出し側が安全性の前提条件を満たす責任がある場合。前提条件を `# Safety` ドキュメントに記載
- **unsafe ブロック**: 関数自体は安全だが、内部で unsafe 操作が必要な場合。入力検証を行い、安全なAPIとして公開

```rust
// unsafe fn: 呼び出し側に責任がある
/// # Safety
/// ptr は有効で、len バイトの読み取りが可能であること
pub unsafe fn read_buffer(ptr: *const u8, len: usize) -> Vec<u8> {
    std::slice::from_raw_parts(ptr, len).to_vec()
}

// 安全な関数 + 内部 unsafe: 関数が安全性を保証
pub fn safe_split(s: &mut [i32], mid: usize) -> (&mut [i32], &mut [i32]) {
    assert!(mid <= s.len());
    let ptr = s.as_mut_ptr();
    unsafe {
        (
            std::slice::from_raw_parts_mut(ptr, mid),
            std::slice::from_raw_parts_mut(ptr.add(mid), s.len() - mid),
        )
    }
}
```

---

## 12. まとめ

| 概念 | 要点 |
|------|------|
| unsafe ブロック | 5つの超能力を解禁。最小限に抑える |
| 生ポインタ | *const T / *mut T。デリファレンスが unsafe |
| FFI | extern "C" でC言語関数を呼び出し/公開 |
| unsafe トレイト | Send/Sync 等。実装者が安全性を保証 |
| static mut | グローバル可変状態。AtomicやMutexで代替推奨 |
| union | 同一メモリを複数の型で解釈。判別子で管理 |
| 安全な抽象化 | unsafe を内部に封じ、安全なAPIを公開 |
| Miri | 未定義動作を検出するツール |
| repr(C) | C言語互換のメモリレイアウト |
| MaybeUninit | 未初期化メモリの安全な扱い |
| # Safety | unsafe 関数の前提条件を文書化 |

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
5. **Rust Reference - Unsafety** -- https://doc.rust-lang.org/reference/unsafety.html
6. **std::mem::MaybeUninit** -- https://doc.rust-lang.org/std/mem/union.MaybeUninit.html
