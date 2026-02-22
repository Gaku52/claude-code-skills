# FFI（Foreign Function Interface）

> Rust と他言語の相互運用を理解し、bindgen・PyO3・napi-rs を使ったクロス言語連携を実践的に習得する

## この章で学ぶこと

1. **FFI の基本概念** — C ABI、unsafe の扱い、メモリ管理の責任分界
2. **C/C++ 連携** — bindgen によるバインディング自動生成と安全なラッパー設計
3. **高レベル連携** — PyO3 による Python 拡張、napi-rs による Node.js ネイティブモジュール
4. **cxx クレート** — C++ との型安全な双方向連携
5. **UniFFI** — モバイル (Kotlin/Swift) 向けのマルチ言語バインディング

---

## 1. FFI の基本

```
FFI の仕組み
=============

Rust                 C ABI 境界              他言語
+-------------+     +-----------+     +-------------+
| safe Rust   |     | extern "C"|     | Python      |
| code        | --> | #[no_mangle] | <-- | ctypes /    |
|             |     | unsafe {}  |     | PyO3        |
+-------------+     +-----------+     +-------------+

  - C ABI は言語間の共通インターフェース
  - Rust 側は extern "C" で C 互換関数を公開
  - メモリ管理の責任を明確にすることが最重要
```

### FFI で使われる主要な型マッピング

| Rust 型 | C 型 | サイズ | 備考 |
|---|---|---|---|
| `i8` / `u8` | `int8_t` / `uint8_t` | 1 byte | |
| `i16` / `u16` | `int16_t` / `uint16_t` | 2 bytes | |
| `i32` / `u32` | `int32_t` / `uint32_t` | 4 bytes | |
| `i64` / `u64` | `int64_t` / `uint64_t` | 8 bytes | |
| `f32` | `float` | 4 bytes | |
| `f64` | `double` | 8 bytes | |
| `bool` | `bool` / `_Bool` | 1 byte | C99 以降 |
| `*const T` | `const T*` | ptr size | |
| `*mut T` | `T*` | ptr size | |
| `*const c_char` | `const char*` | ptr size | NUL 終端文字列 |
| `()` | `void` | 0 | 戻り値のみ |
| `Option<&T>` | `const T*` (nullable) | ptr size | ニッチ最適化 |
| `Option<extern "C" fn()>` | 関数ポインタ (nullable) | ptr size | ニッチ最適化 |

### コード例 1: Rust から C 関数を呼び出す

```rust
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};

extern "C" {
    fn strlen(s: *const c_char) -> usize;
    fn getenv(name: *const c_char) -> *const c_char;
}

fn safe_strlen(s: &str) -> usize {
    let c_str = CString::new(s).expect("CString::new failed");
    unsafe { strlen(c_str.as_ptr()) }
}

fn safe_getenv(name: &str) -> Option<String> {
    let c_name = CString::new(name).expect("CString::new failed");
    unsafe {
        let ptr = getenv(c_name.as_ptr());
        if ptr.is_null() {
            None
        } else {
            Some(CStr::from_ptr(ptr).to_string_lossy().into_owned())
        }
    }
}

fn main() {
    println!("Length: {}", safe_strlen("Hello, FFI!"));
    if let Some(home) = safe_getenv("HOME") {
        println!("HOME: {}", home);
    }
}
```

### コード例 2: Rust の関数を C に公開

```rust
// lib.rs -- cdylib としてビルド
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

/// C から呼び出し可能な文字列反転関数
#[no_mangle]
pub extern "C" fn rust_string_reverse(input: *const c_char) -> *mut c_char {
    if input.is_null() {
        return ptr::null_mut();
    }
    let c_str = unsafe { CStr::from_ptr(input) };
    let rust_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };
    let reversed: String = rust_str.chars().rev().collect();
    match CString::new(reversed) {
        Ok(c_string) => c_string.into_raw(),  // 所有権を呼び出し側に渡す
        Err(_) => ptr::null_mut(),
    }
}

/// メモリ解放用関数（呼び出し側が必ず呼ぶ）
#[no_mangle]
pub extern "C" fn rust_string_free(s: *mut c_char) {
    if !s.is_null() {
        unsafe { drop(CString::from_raw(s)); }
    }
}
```

```c
/* C 側: main.c */
#include <stdio.h>
extern char* rust_string_reverse(const char* input);
extern void rust_string_free(char* s);

int main() {
    char* result = rust_string_reverse("Hello, Rust!");
    if (result) {
        printf("Reversed: %s\n", result);  // "!tsuR ,olleH"
        rust_string_free(result);  // 必ず解放
    }
    return 0;
}
```

### コード例: CString と CStr の違いと使い分け

```rust
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

fn demonstrate_cstring_cstr() {
    // CString: Rust が所有する NUL 終端文字列 (ヒープ割当)
    // - Rust → C に文字列を渡すときに使う
    // - 内部にNULバイトがあると panic
    let owned = CString::new("Hello, FFI!").unwrap();
    println!("CString: {:?}", owned);
    println!("  as_ptr: {:p}", owned.as_ptr());
    println!("  as_bytes_with_nul: {:?}", owned.as_bytes_with_nul());

    // CStr: NUL 終端文字列への借用 (ヒープ割当なし)
    // - C → Rust に文字列を受け取るときに使う
    let borrowed: &CStr = owned.as_c_str();
    println!("CStr: {:?}", borrowed);

    // C から受け取ったポインタを CStr に変換
    let ptr: *const c_char = owned.as_ptr();
    let from_ptr: &CStr = unsafe { CStr::from_ptr(ptr) };
    println!("from_ptr: {:?}", from_ptr.to_str().unwrap());

    // to_str() vs to_string_lossy()
    // to_str():          Result<&str> — 不正な UTF-8 でエラー
    // to_string_lossy(): Cow<str>    — 不正な UTF-8 を U+FFFD に置換
    match from_ptr.to_str() {
        Ok(s) => println!("Valid UTF-8: {}", s),
        Err(e) => println!("Invalid UTF-8: {}", e),
    }

    // NUL バイトを含む文字列は CString::new() でエラー
    match CString::new("hello\0world") {
        Ok(_) => unreachable!(),
        Err(e) => println!("NulError: {} (位置: {})", e, e.nul_position()),
    }
}

fn main() {
    demonstrate_cstring_cstr();
}
```

### コード例: 構造体の受け渡し

```rust
use std::os::raw::c_char;
use std::ffi::CStr;

/// C と共有する構造体 — repr(C) が必須
#[repr(C)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

#[repr(C)]
pub struct Rect {
    pub origin: Point,
    pub width: f64,
    pub height: f64,
}

#[repr(C)]
pub struct ErrorInfo {
    pub code: i32,
    pub message: [c_char; 256],
}

/// 構造体を値渡し
#[no_mangle]
pub extern "C" fn rect_area(rect: Rect) -> f64 {
    rect.width * rect.height
}

/// 構造体をポインタ渡し (大きな構造体に適切)
#[no_mangle]
pub extern "C" fn rect_area_ptr(rect: *const Rect) -> f64 {
    if rect.is_null() {
        return 0.0;
    }
    let rect = unsafe { &*rect };
    rect.width * rect.height
}

/// 構造体を出力パラメータとして返す
#[no_mangle]
pub extern "C" fn create_rect(x: f64, y: f64, w: f64, h: f64, out: *mut Rect) -> i32 {
    if out.is_null() {
        return -1;
    }
    unsafe {
        (*out).origin = Point { x, y };
        (*out).width = w;
        (*out).height = h;
    }
    0 // 成功
}

/// エラー情報を構造体で返す
#[no_mangle]
pub extern "C" fn get_error_info(out: *mut ErrorInfo) -> i32 {
    if out.is_null() {
        return -1;
    }

    let msg = "File not found";
    let msg_bytes = msg.as_bytes();

    unsafe {
        (*out).code = 404;
        let dest = &mut (*out).message;
        let len = msg_bytes.len().min(dest.len() - 1);
        std::ptr::copy_nonoverlapping(
            msg_bytes.as_ptr() as *const c_char,
            dest.as_mut_ptr(),
            len,
        );
        dest[len] = 0; // NUL 終端
    }
    0
}

fn main() {
    let rect = Rect {
        origin: Point { x: 0.0, y: 0.0 },
        width: 10.0,
        height: 5.0,
    };
    println!("面積: {}", rect_area(rect));
}
```

### コード例: コールバック関数の受け渡し

```rust
use std::os::raw::c_void;

/// C 側から渡されるコールバック型
type ProgressCallback = extern "C" fn(current: u64, total: u64, user_data: *mut c_void);

/// コールバックを受け取る関数
#[no_mangle]
pub extern "C" fn process_with_callback(
    data_ptr: *const u8,
    data_len: usize,
    callback: Option<ProgressCallback>,
    user_data: *mut c_void,
) -> i32 {
    if data_ptr.is_null() || data_len == 0 {
        return -1;
    }

    let data = unsafe { std::slice::from_raw_parts(data_ptr, data_len) };
    let total = data.len() as u64;

    for (i, _chunk) in data.chunks(1024).enumerate() {
        let current = ((i + 1) * 1024).min(data.len()) as u64;

        // コールバックが設定されていれば呼び出す
        if let Some(cb) = callback {
            cb(current, total, user_data);
        }
    }

    0 // 成功
}

/// Rust 側のコールバック実装例
extern "C" fn my_progress(current: u64, total: u64, _user_data: *mut c_void) {
    let percent = (current as f64 / total as f64 * 100.0) as u32;
    println!("進捗: {}% ({}/{})", percent, current, total);
}

fn main() {
    let data = vec![0u8; 10000];
    process_with_callback(
        data.as_ptr(),
        data.len(),
        Some(my_progress),
        std::ptr::null_mut(),
    );
}
```

---

## 2. bindgen による自動バインディング

```
bindgen のワークフロー
========================

C/C++ ヘッダー (.h)
       |
       v
  [bindgen]  <-- build.rs で自動実行
       |
       v
Rust バインディング (.rs)
       |
       v
安全なラッパー (safe API)
       |
       v
アプリケーションコード
```

### コード例 3: bindgen を使った C ライブラリのラッピング

```rust
// build.rs
fn main() {
    cc::Build::new()
        .file("vendor/mylib.c")
        .compile("mylib");

    let bindings = bindgen::Builder::default()
        .header("vendor/mylib.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .allowlist_function("mylib_.*")
        .allowlist_type("MyLib.*")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings.write_to_file(out_path.join("bindings.rs")).unwrap();

    println!("cargo:rustc-link-lib=mylib");
}

// src/lib.rs
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

// 安全なラッパー
pub struct MyLibHandle {
    raw: *mut MyLibContext,
}

impl MyLibHandle {
    pub fn new() -> Result<Self, String> {
        let raw = unsafe { mylib_create() };
        if raw.is_null() {
            Err("Failed to create context".to_string())
        } else {
            Ok(MyLibHandle { raw })
        }
    }

    pub fn process(&self, data: &[u8]) -> Result<Vec<u8>, String> {
        let mut out_len: usize = 0;
        let result = unsafe {
            mylib_process(self.raw, data.as_ptr(), data.len(), &mut out_len)
        };
        if result.is_null() {
            Err("Processing failed".to_string())
        } else {
            let output = unsafe {
                std::slice::from_raw_parts(result, out_len).to_vec()
            };
            unsafe { mylib_free(result) };
            Ok(output)
        }
    }
}

impl Drop for MyLibHandle {
    fn drop(&mut self) {
        unsafe { mylib_destroy(self.raw) };
    }
}

unsafe impl Send for MyLibHandle {}
```

### コード例: bindgen の高度な設定

```rust
// build.rs — より詳細な bindgen 設定
fn main() {
    // システムライブラリのリンク
    pkg_config::Config::new()
        .probe("libssl")
        .expect("OpenSSL not found");

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        // 特定の関数・型のみ生成
        .allowlist_function("SSL_.*")
        .allowlist_type("SSL_CTX")
        // ブロックリスト
        .blocklist_function("SSL_internal_.*")
        // 型のマッピングカスタマイズ
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: true,
        })
        // レイアウトテスト生成
        .layout_tests(true)
        // derive の自動付与
        .derive_debug(true)
        .derive_default(true)
        .derive_eq(true)
        .derive_hash(true)
        // ドキュメントコメントの生成
        .generate_comments(true)
        // include パスの追加
        .clang_arg("-I/usr/include")
        .clang_arg("-I/usr/local/include")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings.write_to_file(out_path.join("bindings.rs")).unwrap();
}
```

### コード例: 安全なラッパーの設計パターン

```rust
use std::ffi::CString;
use std::os::raw::c_char;
use std::ptr;

// FFI 関数の宣言 (bindgen が生成するもの)
extern "C" {
    fn db_open(path: *const c_char) -> *mut DbHandle;
    fn db_close(handle: *mut DbHandle);
    fn db_query(handle: *mut DbHandle, sql: *const c_char, out: *mut *mut ResultSet) -> i32;
    fn db_result_next(rs: *mut ResultSet) -> i32;
    fn db_result_get_str(rs: *mut ResultSet, col: i32) -> *const c_char;
    fn db_result_free(rs: *mut ResultSet);
}

#[repr(C)]
pub struct DbHandle {
    _private: [u8; 0], // 不透明型
}

#[repr(C)]
pub struct ResultSet {
    _private: [u8; 0],
}

/// 安全なデータベースラッパー
pub struct Database {
    handle: *mut DbHandle,
}

impl Database {
    pub fn open(path: &str) -> Result<Self, String> {
        let c_path = CString::new(path).map_err(|e| e.to_string())?;
        let handle = unsafe { db_open(c_path.as_ptr()) };
        if handle.is_null() {
            Err(format!("Failed to open database: {}", path))
        } else {
            Ok(Database { handle })
        }
    }

    pub fn query(&self, sql: &str) -> Result<QueryResult, String> {
        let c_sql = CString::new(sql).map_err(|e| e.to_string())?;
        let mut result_ptr: *mut ResultSet = ptr::null_mut();
        let rc = unsafe { db_query(self.handle, c_sql.as_ptr(), &mut result_ptr) };
        if rc != 0 {
            return Err(format!("Query failed with code: {}", rc));
        }
        Ok(QueryResult { handle: result_ptr })
    }
}

impl Drop for Database {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { db_close(self.handle) };
        }
    }
}

// Send は手動で保証 (スレッド安全なC API の場合のみ)
unsafe impl Send for Database {}

/// クエリ結果のイテレータ
pub struct QueryResult {
    handle: *mut ResultSet,
}

impl QueryResult {
    pub fn next_row(&mut self) -> bool {
        unsafe { db_result_next(self.handle) != 0 }
    }

    pub fn get_string(&self, column: i32) -> Option<String> {
        let ptr = unsafe { db_result_get_str(self.handle, column) };
        if ptr.is_null() {
            None
        } else {
            let cstr = unsafe { std::ffi::CStr::from_ptr(ptr) };
            Some(cstr.to_string_lossy().into_owned())
        }
    }
}

impl Drop for QueryResult {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { db_result_free(self.handle) };
        }
    }
}

// 使用例
fn main() {
    // let db = Database::open("/tmp/test.db").unwrap();
    // let mut result = db.query("SELECT name FROM users").unwrap();
    // while result.next_row() {
    //     if let Some(name) = result.get_string(0) {
    //         println!("Name: {}", name);
    //     }
    // }
}
```

---

## 3. PyO3 -- Python 拡張

### コード例 4: PyO3 による Python モジュール

```rust
// Cargo.toml: pyo3 = { version = "0.22", features = ["extension-module"] }
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::collections::HashMap;

#[pyfunction]
fn fibonacci(n: u64) -> PyResult<u64> {
    if n > 93 {
        return Err(PyValueError::new_err("n must be <= 93"));
    }
    let (mut a, mut b) = (0u64, 1u64);
    for _ in 0..n {
        let temp = b;
        b = a.checked_add(b).ok_or_else(|| PyValueError::new_err("Overflow"))?;
        a = temp;
    }
    Ok(a)
}

#[pyfunction]
fn word_count(text: &str) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for word in text.split_whitespace() {
        *counts.entry(word.to_lowercase()).or_insert(0) += 1;
    }
    counts
}

#[pyclass]
struct DataFrame {
    columns: Vec<String>,
    data: Vec<Vec<f64>>,
}

#[pymethods]
impl DataFrame {
    #[new]
    fn new(columns: Vec<String>) -> Self {
        let col_count = columns.len();
        DataFrame { columns, data: vec![Vec::new(); col_count] }
    }

    fn add_row(&mut self, values: Vec<f64>) -> PyResult<()> {
        if values.len() != self.columns.len() {
            return Err(PyValueError::new_err("Column count mismatch"));
        }
        for (col, val) in self.data.iter_mut().zip(values.iter()) {
            col.push(*val);
        }
        Ok(())
    }

    fn mean(&self, column: &str) -> PyResult<f64> {
        let idx = self.columns.iter().position(|c| c == column)
            .ok_or_else(|| PyValueError::new_err("Column not found"))?;
        let col = &self.data[idx];
        if col.is_empty() { return Err(PyValueError::new_err("Empty")); }
        Ok(col.iter().sum::<f64>() / col.len() as f64)
    }
}

#[pymodule]
fn my_rust_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fibonacci, m)?)?;
    m.add_function(wrap_pyfunction!(word_count, m)?)?;
    m.add_class::<DataFrame>()?;
    Ok(())
}
```

```python
# Python 側
import my_rust_module

print(my_rust_module.fibonacci(50))   # 12586269025
print(my_rust_module.word_count("hello world hello rust"))

df = my_rust_module.DataFrame(["x", "y"])
df.add_row([1.0, 2.0])
df.add_row([3.0, 4.0])
print(df.mean("y"))  # 3.0
```

### コード例: PyO3 の高度な機能

```rust
use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyIOError};
use pyo3::types::{PyDict, PyList, PyBytes};
use std::io::Read;

/// Python のプロトコルを実装 (__len__, __getitem__, __iter__ 等)
#[pyclass]
struct Matrix {
    data: Vec<Vec<f64>>,
    rows: usize,
    cols: usize,
}

#[pymethods]
impl Matrix {
    #[new]
    fn new(rows: usize, cols: usize) -> Self {
        Matrix {
            data: vec![vec![0.0; cols]; rows],
            rows,
            cols,
        }
    }

    /// Python の len() に対応
    fn __len__(&self) -> usize {
        self.rows * self.cols
    }

    /// Python の str() / print() に対応
    fn __repr__(&self) -> String {
        format!("Matrix({}x{})", self.rows, self.cols)
    }

    /// Python の matrix[i, j] に対応
    fn __getitem__(&self, idx: (usize, usize)) -> PyResult<f64> {
        let (i, j) = idx;
        if i >= self.rows || j >= self.cols {
            return Err(PyValueError::new_err("Index out of bounds"));
        }
        Ok(self.data[i][j])
    }

    /// Python の matrix[i, j] = value に対応
    fn __setitem__(&mut self, idx: (usize, usize), value: f64) -> PyResult<()> {
        let (i, j) = idx;
        if i >= self.rows || j >= self.cols {
            return Err(PyValueError::new_err("Index out of bounds"));
        }
        self.data[i][j] = value;
        Ok(())
    }

    /// 行列の転置
    fn transpose(&self) -> Matrix {
        let mut result = Matrix::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[j][i] = self.data[i][j];
            }
        }
        result
    }

    /// NumPy 配列との相互変換 (numpy feature が必要)
    fn to_list(&self) -> Vec<Vec<f64>> {
        self.data.clone()
    }

    /// バイト列に変換 (シリアライズ)
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.rows.to_le_bytes());
        bytes.extend_from_slice(&self.cols.to_le_bytes());
        for row in &self.data {
            for &val in row {
                bytes.extend_from_slice(&val.to_le_bytes());
            }
        }
        PyBytes::new(py, &bytes)
    }

    /// クラスメソッド
    #[classmethod]
    fn identity(_cls: &Bound<'_, pyo3::types::PyType>, size: usize) -> Self {
        let mut m = Matrix::new(size, size);
        for i in 0..size {
            m.data[i][i] = 1.0;
        }
        m
    }

    /// 静的メソッド
    #[staticmethod]
    fn zeros(rows: usize, cols: usize) -> Self {
        Matrix::new(rows, cols)
    }
}

/// async 関数の PyO3 での扱い
#[pyfunction]
fn read_file_sync(path: &str) -> PyResult<String> {
    std::fs::read_to_string(path)
        .map_err(|e| PyIOError::new_err(e.to_string()))
}

/// Python の dict を受け取って処理
#[pyfunction]
fn process_config(config: &Bound<'_, PyDict>) -> PyResult<String> {
    let host = config
        .get_item("host")?
        .map(|v| v.extract::<String>())
        .transpose()?
        .unwrap_or_else(|| "localhost".to_string());

    let port = config
        .get_item("port")?
        .map(|v| v.extract::<u16>())
        .transpose()?
        .unwrap_or(8080);

    Ok(format!("{}:{}", host, port))
}

#[pymodule]
fn advanced_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Matrix>()?;
    m.add_function(wrap_pyfunction!(read_file_sync, m)?)?;
    m.add_function(wrap_pyfunction!(process_config, m)?)?;
    Ok(())
}
```

---

## 4. napi-rs -- Node.js ネイティブモジュール

### コード例 5: napi-rs による Node.js アドオン

```rust
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::collections::HashMap;

#[napi]
pub fn hash_data(input: String, salt: String) -> String {
    use std::hash::{DefaultHasher, Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    format!("{}{}", input, salt).hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

#[napi]
pub async fn parse_large_json(path: String) -> Result<u32> {
    let content = tokio::fs::read_to_string(&path).await
        .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))?;
    let value: serde_json::Value = serde_json::from_str(&content)
        .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))?;

    fn count_keys(v: &serde_json::Value) -> u32 {
        match v {
            serde_json::Value::Object(m) =>
                m.len() as u32 + m.values().map(count_keys).sum::<u32>(),
            serde_json::Value::Array(a) => a.iter().map(count_keys).sum(),
            _ => 0,
        }
    }
    Ok(count_keys(&value))
}

#[napi]
pub struct TextProcessor {
    buffer: Vec<String>,
}

#[napi]
impl TextProcessor {
    #[napi(constructor)]
    pub fn new() -> Self {
        TextProcessor { buffer: Vec::new() }
    }

    #[napi]
    pub fn add_line(&mut self, line: String) {
        self.buffer.push(line);
    }

    #[napi]
    pub fn word_frequencies(&self) -> HashMap<String, u32> {
        let mut freq = HashMap::new();
        for line in &self.buffer {
            for word in line.split_whitespace() {
                *freq.entry(word.to_lowercase()).or_insert(0) += 1;
            }
        }
        freq
    }
}
```

```javascript
const { hashData, parseLargeJson, TextProcessor } = require('./index');

console.log(hashData('password', 'salt'));

const processor = new TextProcessor();
processor.addLine("hello world");
processor.addLine("hello rust");
console.log(processor.wordFrequencies());
```

### コード例: napi-rs の高度な機能

```rust
use napi::bindgen_prelude::*;
use napi_derive::napi;

/// TypeScript の enum に対応
#[napi]
pub enum LogLevel {
    Debug,
    Info,
    Warn,
    Error,
}

/// TypeScript のインターフェースに対応する構造体
#[napi(object)]
pub struct ServerConfig {
    pub host: String,
    pub port: u32,
    pub max_connections: Option<u32>,
    pub tls_enabled: bool,
}

/// ストリーミング処理 (Node.js の ReadableStream 風)
#[napi]
pub struct LineReader {
    lines: Vec<String>,
    position: usize,
}

#[napi]
impl LineReader {
    #[napi(constructor)]
    pub fn new(content: String) -> Self {
        let lines: Vec<String> = content.lines().map(|l| l.to_string()).collect();
        LineReader { lines, position: 0 }
    }

    #[napi]
    pub fn next_line(&mut self) -> Option<String> {
        if self.position < self.lines.len() {
            let line = self.lines[self.position].clone();
            self.position += 1;
            Some(line)
        } else {
            None
        }
    }

    #[napi]
    pub fn remaining(&self) -> u32 {
        (self.lines.len() - self.position) as u32
    }

    #[napi]
    pub fn reset(&mut self) {
        self.position = 0;
    }
}

/// コールバック関数を受け取る
#[napi]
pub fn process_items(
    items: Vec<String>,
    #[napi(ts_arg_type = "(item: string, index: number) => string")]
    callback: Function<(String, u32), String>,
) -> Result<Vec<String>> {
    let mut results = Vec::new();
    for (i, item) in items.iter().enumerate() {
        let result = callback.call((item.clone(), i as u32))?;
        results.push(result);
    }
    Ok(results)
}

/// Buffer (バイナリデータ) の受け渡し
#[napi]
pub fn compress_data(input: Buffer) -> Result<Buffer> {
    // 簡易的な RLE 圧縮
    let data = input.as_ref();
    let mut compressed = Vec::new();
    let mut i = 0;

    while i < data.len() {
        let byte = data[i];
        let mut count = 1u8;
        while i + count as usize < data.len()
            && data[i + count as usize] == byte
            && count < 255
        {
            count += 1;
        }
        compressed.push(count);
        compressed.push(byte);
        i += count as usize;
    }

    Ok(compressed.into())
}
```

---

## 5. cxx クレート — C++ との型安全な連携

### コード例: cxx による C++ バインディング

```rust
// src/main.rs
#[cxx::bridge]
mod ffi {
    // C++ に公開する Rust 型
    struct RustConfig {
        name: String,
        value: i64,
    }

    // C++ 側で定義された関数
    unsafe extern "C++" {
        include!("my_project/cpp_lib.h");

        type CppProcessor;

        fn new_processor(config: &str) -> UniquePtr<CppProcessor>;
        fn process(self: &CppProcessor, input: &[u8]) -> Vec<u8>;
        fn get_stats(self: &CppProcessor) -> String;
    }

    // Rust 側で定義し、C++ から呼べる関数
    extern "Rust" {
        fn rust_log(message: &str);
        fn rust_compute(values: &[f64]) -> f64;
    }
}

fn rust_log(message: &str) {
    eprintln!("[Rust] {}", message);
}

fn rust_compute(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn main() {
    let processor = ffi::new_processor("default");
    let input = b"Hello from Rust";
    let output = processor.process(input);
    println!("処理結果: {} bytes", output.len());
    println!("統計: {}", processor.get_stats());
}
```

```cpp
// cpp_lib.h
#pragma once
#include "rust/cxx.h"
#include <memory>
#include <string>

class CppProcessor {
public:
    CppProcessor(rust::Str config);
    rust::Vec<uint8_t> process(rust::Slice<const uint8_t> input) const;
    rust::String get_stats() const;

private:
    std::string config_;
    mutable size_t processed_count_ = 0;
};

std::unique_ptr<CppProcessor> new_processor(rust::Str config);
```

---

## FFI 方式比較表

| 方式 | 対象言語 | 安全性 | 性能 | 開発体験 | ユースケース |
|---|---|---|---|---|---|
| **raw FFI** | C/C++ | 低（unsafe 必須） | 最高 | 低 | 既存 C ライブラリの利用 |
| **bindgen** | C/C++ | 中（自動生成） | 最高 | 中 | 大規模 C ヘッダーのバインド |
| **cxx** | C++ | 高（型安全） | 高 | 高 | C++ との双方向連携 |
| **PyO3** | Python | 高 | 高 | 高 | Python 拡張モジュール |
| **napi-rs** | Node.js | 高 | 高 | 高 | Node.js ネイティブアドオン |
| **uniffi** | 多言語 | 高 | 中 | 中 | モバイル（Kotlin/Swift） |

### パフォーマンス改善の目安

| 操作 | Pure Python | PyO3 (Rust) | 速度向上 |
|---|---|---|---|
| フィボナッチ(40) | 25s | 0.5s | 50x |
| 文字列処理(1M行) | 3.2s | 0.15s | 21x |
| JSON パース(100MB) | 8.5s | 1.2s | 7x |
| 画像リサイズ(4K) | 2.1s | 0.3s | 7x |

### Cargo.toml の crate-type 設定

| crate-type | 出力 | 用途 |
|---|---|---|
| `lib` | `.rlib` | Rust ライブラリ (デフォルト) |
| `cdylib` | `.so` / `.dylib` / `.dll` | C 互換動的ライブラリ (FFI 用) |
| `staticlib` | `.a` / `.lib` | C 互換静的ライブラリ |
| `dylib` | `.so` / `.dylib` / `.dll` | Rust 動的ライブラリ |

```toml
# Cargo.toml の設定例
[lib]
name = "my_ffi_lib"
crate-type = ["cdylib", "lib"]  # FFI + Rust 両方で使用

[dependencies]
# PyO3 の場合
pyo3 = { version = "0.22", features = ["extension-module"] }

# napi-rs の場合
napi = { version = "2", features = ["async"] }
napi-derive = "2"
```

---

## アンチパターン

### 1. メモリ解放の責任不明確

```rust
// [NG] 解放関数を提供しない
#[no_mangle]
pub extern "C" fn create_data() -> *mut Data {
    Box::into_raw(Box::new(Data::new()))
    // 呼び出し側はどう解放する?
}

// [OK] 必ず対になる解放関数を提供
#[no_mangle]
pub extern "C" fn create_data() -> *mut Data {
    Box::into_raw(Box::new(Data::new()))
}

#[no_mangle]
pub extern "C" fn free_data(ptr: *mut Data) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)); }
    }
}
```

### 2. FFI 境界でのパニック

**問題**: Rust のパニックが FFI 境界を越えると未定義動作になる。

```rust
// [NG] パニックの可能性
#[no_mangle]
pub extern "C" fn process(data: *const u8, len: usize) -> i32 {
    let slice = unsafe { std::slice::from_raw_parts(data, len) };
    slice[100] as i32  // 範囲外でパニック!
}

// [OK] catch_unwind でパニックを捕捉
#[no_mangle]
pub extern "C" fn process(data: *const u8, len: usize) -> i32 {
    std::panic::catch_unwind(|| {
        if data.is_null() || len == 0 { return -1; }
        let slice = unsafe { std::slice::from_raw_parts(data, len) };
        slice.get(100).copied().map_or(-1, |v| v as i32)
    }).unwrap_or(-1)
}
```

### 3. 文字列エンコーディングの不一致

```rust
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

// [NG] UTF-8 前提で CStr を変換
#[no_mangle]
pub extern "C" fn bad_process_string(s: *const c_char) -> i32 {
    let cstr = unsafe { CStr::from_ptr(s) };
    let rust_str = cstr.to_str().unwrap(); // パニックの可能性!
    rust_str.len() as i32
}

// [OK] 不正な UTF-8 を安全に処理
#[no_mangle]
pub extern "C" fn good_process_string(s: *const c_char) -> i32 {
    if s.is_null() {
        return -1;
    }
    let cstr = unsafe { CStr::from_ptr(s) };
    match cstr.to_str() {
        Ok(valid_str) => valid_str.len() as i32,
        Err(_) => {
            // lossy 変換でフォールバック
            let lossy = cstr.to_string_lossy();
            lossy.len() as i32
        }
    }
}
```

### 4. スレッド安全性の無視

```rust
// [NG] スレッド安全でない C ライブラリを Send/Sync マーク
struct UnsafeWrapper {
    handle: *mut CLibHandle,
}

unsafe impl Send for UnsafeWrapper {} // 危険!
unsafe impl Sync for UnsafeWrapper {} // 危険!

// [OK] Mutex でラップしてスレッド安全性を確保
struct SafeWrapper {
    handle: std::sync::Mutex<*mut CLibHandle>,
}

unsafe impl Send for SafeWrapper {} // Mutex でガード済み

impl SafeWrapper {
    fn call(&self, arg: i32) -> i32 {
        let handle = self.handle.lock().unwrap();
        unsafe { c_lib_call(*handle, arg) }
    }
}

#[repr(C)]
struct CLibHandle { _private: [u8; 0] }
extern "C" { fn c_lib_call(h: *mut CLibHandle, arg: i32) -> i32; }
```

---

## FAQ

### Q1: PyO3 と ctypes/cffi のどちらを使うべきですか？

**A**: PyO3 を推奨します。ctypes/cffi は C ABI を直接呼ぶため型安全性がなく手動マーシャリングが必要です。PyO3 は Rust の型システムを活かして Python 型との自動変換を行い、エラーハンドリングも自然です。既存 C ライブラリを Python から呼ぶだけなら ctypes が最小労力です。

### Q2: FFI のデバッグ手法は？

**A**: 以下を組み合わせます:
1. **Valgrind / AddressSanitizer**: メモリエラーの検出
2. **RUST_BACKTRACE=1**: Rust 側のスタックトレース
3. **ログ出力**: FFI 境界の入出力を記録
4. **テスト**: Rust 側で安全なラッパーを十分にテストしてから FFI に公開

### Q3: napi-rs と N-API (node-addon-api) の違いは？

**A**: N-API は C/C++ で Node.js アドオンを書く公式 API、napi-rs はそれを Rust から使うラッパーです。Rust のメモリ安全性の恩恵を受けられ、パフォーマンスはほぼ同等ですがメモリバグのリスクが大幅に低下します。

### Q4: cxx と bindgen のどちらを使うべきですか？

**A**: C ライブラリには bindgen、C++ ライブラリには cxx が推奨です。cxx は Rust と C++ の型を直接マッピングでき、String, Vec, UniquePtr 等の安全な型変換を提供します。bindgen は C ヘッダーから自動生成しますが、C++ テンプレートやオーバーロードの処理は苦手です。

### Q5: FFI で大きなデータを効率的に渡すには？

**A**: コピーを避けることが重要です。ポインタとサイズのペアでスライスとして渡すか、共有メモリを使います。

```rust
/// [OK] ゼロコピーでバイト列を渡す
#[no_mangle]
pub extern "C" fn process_buffer(data: *const u8, len: usize) -> i32 {
    if data.is_null() || len == 0 {
        return -1;
    }
    // データをコピーせずにスライスとして参照
    let slice = unsafe { std::slice::from_raw_parts(data, len) };
    // ... 処理 ...
    0
}

/// [NG] データをコピーして処理 (大きなデータで非効率)
#[no_mangle]
pub extern "C" fn bad_process_buffer(data: *const u8, len: usize) -> i32 {
    let slice = unsafe { std::slice::from_raw_parts(data, len) };
    let copied: Vec<u8> = slice.to_vec(); // 不要なコピー!
    // ... 処理 ...
    0
}
```

### Q6: UniFFI とは何ですか？

**A**: Mozilla が開発した多言語バインディングジェネレータです。UDL (Universal Definition Language) ファイルからKotlin, Swift, Python, Ruby のバインディングを自動生成します。モバイルアプリでRustのコアロジックを共有する場合に最適です。

```rust
// UDL ファイル (my_lib.udl):
// namespace my_lib {
//     string hello(string name);
//     u64 add(u64 a, u64 b);
// };
//
// interface Calculator {
//     constructor();
//     void push(f64 value);
//     f64 result();
// };

// Rust 実装
pub fn hello(name: String) -> String {
    format!("Hello, {}!", name)
}

pub fn add(a: u64, b: u64) -> u64 {
    a + b
}

pub struct Calculator {
    stack: Vec<f64>,
}

impl Calculator {
    pub fn new() -> Self {
        Calculator { stack: Vec::new() }
    }

    pub fn push(&mut self, value: f64) {
        self.stack.push(value);
    }

    pub fn result(&self) -> f64 {
        self.stack.last().copied().unwrap_or(0.0)
    }
}
```

---

## FFI テストのベストプラクティス

### コード例: FFI ラッパーのテスト

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn test_string_reverse() {
        let input = CString::new("Hello").unwrap();
        let result = rust_string_reverse(input.as_ptr());
        assert!(!result.is_null());

        let result_str = unsafe { CStr::from_ptr(result) };
        assert_eq!(result_str.to_str().unwrap(), "olleH");

        // メモリリーク防止: 必ず解放
        rust_string_free(result);
    }

    #[test]
    fn test_null_input() {
        let result = rust_string_reverse(std::ptr::null());
        assert!(result.is_null());
    }

    #[test]
    fn test_empty_string() {
        let input = CString::new("").unwrap();
        let result = rust_string_reverse(input.as_ptr());
        assert!(!result.is_null());

        let result_str = unsafe { CStr::from_ptr(result) };
        assert_eq!(result_str.to_str().unwrap(), "");
        rust_string_free(result);
    }

    #[test]
    fn test_struct_passing() {
        let rect = Rect {
            origin: Point { x: 0.0, y: 0.0 },
            width: 10.0,
            height: 5.0,
        };
        assert!((rect_area(rect) - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_callback() {
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::os::raw::c_void;

        static CALL_COUNT: AtomicU64 = AtomicU64::new(0);

        extern "C" fn test_callback(_current: u64, _total: u64, _user_data: *mut c_void) {
            CALL_COUNT.fetch_add(1, Ordering::Relaxed);
        }

        let data = vec![0u8; 5000];
        CALL_COUNT.store(0, Ordering::Relaxed);

        process_with_callback(
            data.as_ptr(),
            data.len(),
            Some(test_callback),
            std::ptr::null_mut(),
        );

        assert!(CALL_COUNT.load(Ordering::Relaxed) > 0);
    }
}
```

### FFI プロジェクトの CI 設定

```yaml
# .github/workflows/ffi-test.yml
name: FFI Tests
on: [push, pull_request]

jobs:
  test:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - name: Install system deps (Linux)
        if: runner.os == 'Linux'
        run: sudo apt-get install -y libclang-dev
      - name: Run tests
        run: cargo test --verbose
      - name: Run Miri (メモリ安全性チェック)
        if: runner.os == 'Linux'
        run: |
          rustup component add miri
          cargo +nightly miri test -- --skip ffi_integration
      - name: Valgrind check (Linux)
        if: runner.os == 'Linux'
        run: |
          sudo apt-get install -y valgrind
          cargo build --release
          valgrind --leak-check=full ./target/release/ffi_test_binary
```

---

## まとめ

| 項目 | 要点 |
|---|---|
| FFI の基本 | `extern "C"` + `#[no_mangle]` で C ABI 互換関数を公開 |
| メモリ管理 | 所有権の責任を明確に。生成と解放は必ずペアで提供 |
| CString / CStr | Rust → C は CString、C → Rust は CStr |
| repr(C) | 構造体を C と共有する場合に必須 |
| bindgen | C ヘッダーから Rust バインディングを自動生成 |
| cxx | C++ との型安全な双方向連携 |
| PyO3 | Python 拡張を Rust で記述。10-50倍の高速化が期待できる |
| napi-rs | Node.js ネイティブアドオンを Rust で記述。async 対応 |
| UniFFI | Kotlin/Swift/Python 向けマルチ言語バインディング |
| 安全性 | FFI 境界でのパニック防止、null チェック、エラーハンドリング |
| catch_unwind | FFI 境界でパニックを捕捉して未定義動作を防止 |

## 次に読むべきガイド

- [ベストプラクティス](../04-ecosystem/04-best-practices.md) — Rust 全般の設計指針と品質管理
- [非同期プログラミング](../02-advanced/01-async.md) — async FFI の設計パターン

## 参考文献

1. **The Rust Reference**: [FFI](https://doc.rust-lang.org/reference/items/external-blocks.html) — FFI の公式仕様
2. **PyO3 公式ガイド**: [PyO3 User Guide](https://pyo3.rs/) — PyO3 の包括的なドキュメント
3. **napi-rs 公式**: [NAPI-RS Documentation](https://napi.rs/) — napi-rs の使い方とサンプル集
4. **cxx ガイド**: [CXX Documentation](https://cxx.rs/) — cxx の型マッピングと使用方法
5. **UniFFI**: [UniFFI User Guide](https://mozilla.github.io/uniffi-rs/) — 多言語バインディングの生成
