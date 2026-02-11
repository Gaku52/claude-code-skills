# FFI（Foreign Function Interface）

> Rust と他言語の相互運用を理解し、bindgen・PyO3・napi-rs を使ったクロス言語連携を実践的に習得する

## この章で学ぶこと

1. **FFI の基本概念** — C ABI、unsafe の扱い、メモリ管理の責任分界
2. **C/C++ 連携** — bindgen によるバインディング自動生成と安全なラッパー設計
3. **高レベル連携** — PyO3 による Python 拡張、napi-rs による Node.js ネイティブモジュール

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

---

## まとめ

| 項目 | 要点 |
|---|---|
| FFI の基本 | `extern "C"` + `#[no_mangle]` で C ABI 互換関数を公開 |
| メモリ管理 | 所有権の責任を明確に。生成と解放は必ずペアで提供 |
| bindgen | C ヘッダーから Rust バインディングを自動生成 |
| PyO3 | Python 拡張を Rust で記述。10-50倍の高速化が期待できる |
| napi-rs | Node.js ネイティブアドオンを Rust で記述。async 対応 |
| 安全性 | FFI 境界でのパニック防止、null チェック、エラーハンドリング |

## 次に読むべきガイド

- [ベストプラクティス](../04-ecosystem/04-best-practices.md) — Rust 全般の設計指針と品質管理
- [非同期プログラミング](../02-advanced/01-async.md) — async FFI の設計パターン

## 参考文献

1. **The Rust Reference**: [FFI](https://doc.rust-lang.org/reference/items/external-blocks.html) — FFI の公式仕様
2. **PyO3 公式ガイド**: [PyO3 User Guide](https://pyo3.rs/) — PyO3 の包括的なドキュメント
3. **napi-rs 公式**: [NAPI-RS Documentation](https://napi.rs/) — napi-rs の使い方とサンプル集
