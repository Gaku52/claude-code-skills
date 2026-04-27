# FFI (Foreign Function Interface)

> Understand interoperability between Rust and other languages, and master cross-language integration in practice using bindgen, PyO3, and napi-rs

## What you will learn in this chapter

1. **FFI fundamentals** — C ABI, handling unsafe, and the boundary of memory management responsibility
2. **C/C++ integration** — Automatic binding generation with bindgen and safe wrapper design
3. **High-level integration** — Python extensions with PyO3, Node.js native modules with napi-rs
4. **The cxx crate** — Type-safe bidirectional integration with C++
5. **UniFFI** — Multi-language bindings for mobile (Kotlin/Swift)


## Prerequisites

Before reading this guide, the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Familiarity with the contents of [Concurrency — Threads, Mutex/RwLock, rayon](./01-concurrency.md)

---

## 1. FFI Fundamentals

```
How FFI Works
=============

Rust                 C ABI Boundary          Other Languages
+-------------+     +-----------+     +-------------+
| safe Rust   |     | extern "C"|     | Python      |
| code        | --> | #[no_mangle] | <-- | ctypes /    |
|             |     | unsafe {}  |     | PyO3        |
+-------------+     +-----------+     +-------------+

  - C ABI is the common interface between languages
  - The Rust side exposes C-compatible functions via extern "C"
  - Clarifying the responsibility for memory management is paramount
```

### Major type mappings used in FFI

| Rust type | C type | Size | Notes |
|---|---|---|---|
| `i8` / `u8` | `int8_t` / `uint8_t` | 1 byte | |
| `i16` / `u16` | `int16_t` / `uint16_t` | 2 bytes | |
| `i32` / `u32` | `int32_t` / `uint32_t` | 4 bytes | |
| `i64` / `u64` | `int64_t` / `uint64_t` | 8 bytes | |
| `f32` | `float` | 4 bytes | |
| `f64` | `double` | 8 bytes | |
| `bool` | `bool` / `_Bool` | 1 byte | C99 and later |
| `*const T` | `const T*` | ptr size | |
| `*mut T` | `T*` | ptr size | |
| `*const c_char` | `const char*` | ptr size | NUL-terminated string |
| `()` | `void` | 0 | Return value only |
| `Option<&T>` | `const T*` (nullable) | ptr size | Niche optimization |
| `Option<extern "C" fn()>` | function pointer (nullable) | ptr size | Niche optimization |

### Code example 1: Calling a C function from Rust

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

### Code example 2: Exposing Rust functions to C

```rust
// lib.rs -- build as cdylib
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

/// String reversal function callable from C
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
        Ok(c_string) => c_string.into_raw(),  // Transfer ownership to the caller
        Err(_) => ptr::null_mut(),
    }
}

/// Memory deallocation function (the caller must always invoke this)
#[no_mangle]
pub extern "C" fn rust_string_free(s: *mut c_char) {
    if !s.is_null() {
        unsafe { drop(CString::from_raw(s)); }
    }
}
```

```c
/* C side: main.c */
#include <stdio.h>
extern char* rust_string_reverse(const char* input);
extern void rust_string_free(char* s);

int main() {
    char* result = rust_string_reverse("Hello, Rust!");
    if (result) {
        printf("Reversed: %s\n", result);  // "!tsuR ,olleH"
        rust_string_free(result);  // Always free
    }
    return 0;
}
```

### Code example: Differences between CString and CStr and how to use them

```rust
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

fn demonstrate_cstring_cstr() {
    // CString: NUL-terminated string owned by Rust (heap allocated)
    // - Used when passing strings from Rust to C
    // - Panics if it contains an interior NUL byte
    let owned = CString::new("Hello, FFI!").unwrap();
    println!("CString: {:?}", owned);
    println!("  as_ptr: {:p}", owned.as_ptr());
    println!("  as_bytes_with_nul: {:?}", owned.as_bytes_with_nul());

    // CStr: A borrow of a NUL-terminated string (no heap allocation)
    // - Used when receiving strings from C into Rust
    let borrowed: &CStr = owned.as_c_str();
    println!("CStr: {:?}", borrowed);

    // Convert a pointer received from C into a CStr
    let ptr: *const c_char = owned.as_ptr();
    let from_ptr: &CStr = unsafe { CStr::from_ptr(ptr) };
    println!("from_ptr: {:?}", from_ptr.to_str().unwrap());

    // to_str() vs to_string_lossy()
    // to_str():          Result<&str> — errors on invalid UTF-8
    // to_string_lossy(): Cow<str>    — replaces invalid UTF-8 with U+FFFD
    match from_ptr.to_str() {
        Ok(s) => println!("Valid UTF-8: {}", s),
        Err(e) => println!("Invalid UTF-8: {}", e),
    }

    // A string containing a NUL byte causes CString::new() to error
    match CString::new("hello\0world") {
        Ok(_) => unreachable!(),
        Err(e) => println!("NulError: {} (position: {})", e, e.nul_position()),
    }
}

fn main() {
    demonstrate_cstring_cstr();
}
```

### Code example: Passing structs

```rust
use std::os::raw::c_char;
use std::ffi::CStr;

/// Struct shared with C — repr(C) is required
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

/// Pass struct by value
#[no_mangle]
pub extern "C" fn rect_area(rect: Rect) -> f64 {
    rect.width * rect.height
}

/// Pass struct by pointer (suitable for large structs)
#[no_mangle]
pub extern "C" fn rect_area_ptr(rect: *const Rect) -> f64 {
    if rect.is_null() {
        return 0.0;
    }
    let rect = unsafe { &*rect };
    rect.width * rect.height
}

/// Return struct via an out parameter
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
    0 // Success
}

/// Return error information via a struct
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
        dest[len] = 0; // NUL terminator
    }
    0
}

fn main() {
    let rect = Rect {
        origin: Point { x: 0.0, y: 0.0 },
        width: 10.0,
        height: 5.0,
    };
    println!("Area: {}", rect_area(rect));
}
```

### Code example: Passing callback functions

```rust
use std::os::raw::c_void;

/// Callback type passed in from the C side
type ProgressCallback = extern "C" fn(current: u64, total: u64, user_data: *mut c_void);

/// Function that accepts a callback
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

        // Call the callback if it has been set
        if let Some(cb) = callback {
            cb(current, total, user_data);
        }
    }

    0 // Success
}

/// Example callback implementation on the Rust side
extern "C" fn my_progress(current: u64, total: u64, _user_data: *mut c_void) {
    let percent = (current as f64 / total as f64 * 100.0) as u32;
    println!("Progress: {}% ({}/{})", percent, current, total);
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

## 2. Automatic bindings with bindgen

```
bindgen Workflow
========================

C/C++ header (.h)
       |
       v
  [bindgen]  <-- run automatically in build.rs
       |
       v
Rust bindings (.rs)
       |
       v
Safe wrapper (safe API)
       |
       v
Application code
```

### Code example 3: Wrapping a C library with bindgen

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

// Safe wrapper
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

### Code example: Advanced bindgen configuration

```rust
// build.rs — more detailed bindgen configuration
fn main() {
    // Link a system library
    pkg_config::Config::new()
        .probe("libssl")
        .expect("OpenSSL not found");

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        // Generate only specific functions/types
        .allowlist_function("SSL_.*")
        .allowlist_type("SSL_CTX")
        // Block list
        .blocklist_function("SSL_internal_.*")
        // Customize type mappings
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: true,
        })
        // Generate layout tests
        .layout_tests(true)
        // Auto-derive
        .derive_debug(true)
        .derive_default(true)
        .derive_eq(true)
        .derive_hash(true)
        // Generate doc comments
        .generate_comments(true)
        // Add include paths
        .clang_arg("-I/usr/include")
        .clang_arg("-I/usr/local/include")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings.write_to_file(out_path.join("bindings.rs")).unwrap();
}
```

### Code example: Safe wrapper design pattern

```rust
use std::ffi::CString;
use std::os::raw::c_char;
use std::ptr;

// FFI function declarations (what bindgen generates)
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
    _private: [u8; 0], // Opaque type
}

#[repr(C)]
pub struct ResultSet {
    _private: [u8; 0],
}

/// Safe database wrapper
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

// Send is asserted manually (only when the C API is thread-safe)
unsafe impl Send for Database {}

/// Iterator over query results
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

// Usage example
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

## 3. PyO3 -- Python extensions

### Code example 4: A Python module with PyO3

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
# Python side
import my_rust_module

print(my_rust_module.fibonacci(50))   # 12586269025
print(my_rust_module.word_count("hello world hello rust"))

df = my_rust_module.DataFrame(["x", "y"])
df.add_row([1.0, 2.0])
df.add_row([3.0, 4.0])
print(df.mean("y"))  # 3.0
```

### Code example: Advanced features of PyO3

```rust
use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyIOError};
use pyo3::types::{PyDict, PyList, PyBytes};
use std::io::Read;

/// Implement Python protocols (__len__, __getitem__, __iter__, etc.)
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

    /// Corresponds to Python's len()
    fn __len__(&self) -> usize {
        self.rows * self.cols
    }

    /// Corresponds to Python's str() / print()
    fn __repr__(&self) -> String {
        format!("Matrix({}x{})", self.rows, self.cols)
    }

    /// Corresponds to Python's matrix[i, j]
    fn __getitem__(&self, idx: (usize, usize)) -> PyResult<f64> {
        let (i, j) = idx;
        if i >= self.rows || j >= self.cols {
            return Err(PyValueError::new_err("Index out of bounds"));
        }
        Ok(self.data[i][j])
    }

    /// Corresponds to Python's matrix[i, j] = value
    fn __setitem__(&mut self, idx: (usize, usize), value: f64) -> PyResult<()> {
        let (i, j) = idx;
        if i >= self.rows || j >= self.cols {
            return Err(PyValueError::new_err("Index out of bounds"));
        }
        self.data[i][j] = value;
        Ok(())
    }

    /// Matrix transpose
    fn transpose(&self) -> Matrix {
        let mut result = Matrix::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[j][i] = self.data[i][j];
            }
        }
        result
    }

    /// Mutual conversion with NumPy arrays (requires the numpy feature)
    fn to_list(&self) -> Vec<Vec<f64>> {
        self.data.clone()
    }

    /// Convert to a byte sequence (serialization)
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

    /// Class method
    #[classmethod]
    fn identity(_cls: &Bound<'_, pyo3::types::PyType>, size: usize) -> Self {
        let mut m = Matrix::new(size, size);
        for i in 0..size {
            m.data[i][i] = 1.0;
        }
        m
    }

    /// Static method
    #[staticmethod]
    fn zeros(rows: usize, cols: usize) -> Self {
        Matrix::new(rows, cols)
    }
}

/// Handling async functions in PyO3
#[pyfunction]
fn read_file_sync(path: &str) -> PyResult<String> {
    std::fs::read_to_string(path)
        .map_err(|e| PyIOError::new_err(e.to_string()))
}

/// Receive and process a Python dict
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

## 4. napi-rs -- Node.js native modules

### Code example 5: A Node.js addon with napi-rs

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

### Code example: Advanced features of napi-rs

```rust
use napi::bindgen_prelude::*;
use napi_derive::napi;

/// Corresponds to a TypeScript enum
#[napi]
pub enum LogLevel {
    Debug,
    Info,
    Warn,
    Error,
}

/// Struct corresponding to a TypeScript interface
#[napi(object)]
pub struct ServerConfig {
    pub host: String,
    pub port: u32,
    pub max_connections: Option<u32>,
    pub tls_enabled: bool,
}

/// Streaming processing (similar to Node.js's ReadableStream)
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

/// Receive a callback function
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

/// Passing Buffer (binary data)
#[napi]
pub fn compress_data(input: Buffer) -> Result<Buffer> {
    // Simple RLE compression
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

## 5. The cxx crate — Type-safe integration with C++

### Code example: C++ bindings with cxx

```rust
// src/main.rs
#[cxx::bridge]
mod ffi {
    // Rust types exposed to C++
    struct RustConfig {
        name: String,
        value: i64,
    }

    // Functions defined on the C++ side
    unsafe extern "C++" {
        include!("my_project/cpp_lib.h");

        type CppProcessor;

        fn new_processor(config: &str) -> UniquePtr<CppProcessor>;
        fn process(self: &CppProcessor, input: &[u8]) -> Vec<u8>;
        fn get_stats(self: &CppProcessor) -> String;
    }

    // Functions defined on the Rust side, callable from C++
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
    println!("Result: {} bytes", output.len());
    println!("Stats: {}", processor.get_stats());
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

## FFI approach comparison table

| Approach | Target language | Safety | Performance | DX | Use cases |
|---|---|---|---|---|---|
| **raw FFI** | C/C++ | Low (unsafe required) | Highest | Low | Using existing C libraries |
| **bindgen** | C/C++ | Medium (auto-generated) | Highest | Medium | Binding large C headers |
| **cxx** | C++ | High (type safe) | High | High | Bidirectional C++ integration |
| **PyO3** | Python | High | High | High | Python extension modules |
| **napi-rs** | Node.js | High | High | High | Node.js native add-ons |
| **uniffi** | Multi-language | High | Medium | Medium | Mobile (Kotlin/Swift) |

### Performance improvement guidelines

| Operation | Pure Python | PyO3 (Rust) | Speedup |
|---|---|---|---|
| Fibonacci(40) | 25s | 0.5s | 50x |
| String processing (1M lines) | 3.2s | 0.15s | 21x |
| JSON parsing (100MB) | 8.5s | 1.2s | 7x |
| Image resize (4K) | 2.1s | 0.3s | 7x |

### crate-type setting in Cargo.toml

| crate-type | Output | Use |
|---|---|---|
| `lib` | `.rlib` | Rust library (default) |
| `cdylib` | `.so` / `.dylib` / `.dll` | C-compatible dynamic library (for FFI) |
| `staticlib` | `.a` / `.lib` | C-compatible static library |
| `dylib` | `.so` / `.dylib` / `.dll` | Rust dynamic library |

```toml
# Example Cargo.toml configuration
[lib]
name = "my_ffi_lib"
crate-type = ["cdylib", "lib"]  # Use for both FFI and Rust

[dependencies]
# For PyO3
pyo3 = { version = "0.22", features = ["extension-module"] }

# For napi-rs
napi = { version = "2", features = ["async"] }
napi-derive = "2"
```

---

## Anti-patterns

### 1. Unclear memory deallocation responsibility

```rust
// [NG] Not providing a free function
#[no_mangle]
pub extern "C" fn create_data() -> *mut Data {
    Box::into_raw(Box::new(Data::new()))
    // How is the caller supposed to free this?
}

// [OK] Always provide a paired deallocation function
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

### 2. Panicking across the FFI boundary

**Problem**: A Rust panic crossing the FFI boundary leads to undefined behavior.

```rust
// [NG] Possibility of panic
#[no_mangle]
pub extern "C" fn process(data: *const u8, len: usize) -> i32 {
    let slice = unsafe { std::slice::from_raw_parts(data, len) };
    slice[100] as i32  // Out-of-range access panics!
}

// [OK] Catch panics with catch_unwind
#[no_mangle]
pub extern "C" fn process(data: *const u8, len: usize) -> i32 {
    std::panic::catch_unwind(|| {
        if data.is_null() || len == 0 { return -1; }
        let slice = unsafe { std::slice::from_raw_parts(data, len) };
        slice.get(100).copied().map_or(-1, |v| v as i32)
    }).unwrap_or(-1)
}
```

### 3. String encoding mismatch

```rust
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

// [NG] Converting CStr assuming UTF-8
#[no_mangle]
pub extern "C" fn bad_process_string(s: *const c_char) -> i32 {
    let cstr = unsafe { CStr::from_ptr(s) };
    let rust_str = cstr.to_str().unwrap(); // Possible panic!
    rust_str.len() as i32
}

// [OK] Safely handle invalid UTF-8
#[no_mangle]
pub extern "C" fn good_process_string(s: *const c_char) -> i32 {
    if s.is_null() {
        return -1;
    }
    let cstr = unsafe { CStr::from_ptr(s) };
    match cstr.to_str() {
        Ok(valid_str) => valid_str.len() as i32,
        Err(_) => {
            // Fall back to lossy conversion
            let lossy = cstr.to_string_lossy();
            lossy.len() as i32
        }
    }
}
```

### 4. Ignoring thread safety

```rust
// [NG] Marking a thread-unsafe C library as Send/Sync
struct UnsafeWrapper {
    handle: *mut CLibHandle,
}

unsafe impl Send for UnsafeWrapper {} // Dangerous!
unsafe impl Sync for UnsafeWrapper {} // Dangerous!

// [OK] Wrap in a Mutex to ensure thread safety
struct SafeWrapper {
    handle: std::sync::Mutex<*mut CLibHandle>,
}

unsafe impl Send for SafeWrapper {} // Guarded by Mutex

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

## Hands-on exercises

### Exercise 1: Basic implementation

Implement code that satisfies the following requirements.

**Requirements:**
- Validate input data
- Implement appropriate error handling
- Also write test code

```python
# Exercise 1: Template for basic implementation
class Exercise1:
    """Exercise on a basic implementation pattern"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """Validate input value"""
        if value is None:
            raise ValueError("Input value is None")
        return True

    def process(self, value):
        """Main data processing logic"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """Retrieve processing results"""
        return {
            'count': len(self.data),
            'data': self.data
        }

# Tests
def test_exercise1():
    ex = Exercise1()
    assert ex.process(1) == [1]
    assert ex.process(2) == [1, 2]
    assert ex.get_results()['count'] == 2

    try:
        ex.process(None)
        assert False, "An exception should be raised"
    except ValueError:
        pass

    print("All tests passed!")

test_exercise1()
```

### Exercise 2: Advanced patterns

Extend the basic implementation by adding the following features.

```python
# Exercise 2: Advanced patterns
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """Exercise on advanced patterns"""

    def __init__(self, max_size: int = 100):
        self._items: List[Dict] = []
        self._max_size = max_size
        self._created_at = datetime.now()

    def add(self, key: str, value: any) -> bool:
        """Add an item (with size limit)"""
        if len(self._items) >= self._max_size:
            return False
        self._items.append({
            'key': key,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        return True

    def find(self, key: str) -> Optional[Dict]:
        """Look up by key"""
        for item in reversed(self._items):
            if item['key'] == key:
                return item
        return None

    def remove(self, key: str) -> bool:
        """Remove by key"""
        for i, item in enumerate(self._items):
            if item['key'] == key:
                self._items.pop(i)
                return True
        return False

    def stats(self) -> Dict:
        """Statistics"""
        return {
            'total_items': len(self._items),
            'max_size': self._max_size,
            'usage_percent': len(self._items) / self._max_size * 100,
            'uptime': str(datetime.now() - self._created_at)
        }

# Tests
def test_advanced():
    ex = AdvancedExercise(max_size=3)
    assert ex.add("a", 1) == True
    assert ex.add("b", 2) == True
    assert ex.add("c", 3) == True
    assert ex.add("d", 4) == False  # Size limit
    assert ex.find("b")['value'] == 2
    assert ex.remove("b") == True
    assert ex.find("b") is None
    stats = ex.stats()
    assert stats['total_items'] == 2
    print("All advanced tests passed!")

test_advanced()
```

### Exercise 3: Performance optimization

Improve the performance of the following code.

```python
# Exercise 3: Performance optimization
import time
from functools import lru_cache

# Before optimization (O(n^2))
def slow_search(data: list, target: int) -> int:
    """Inefficient search"""
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] + data[j] == target:
                return (i, j)
    return (-1, -1)

# After optimization (O(n))
def fast_search(data: list, target: int) -> tuple:
    """Efficient search using a hash map"""
    seen = {}
    for i, num in enumerate(data):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return (-1, -1)

# Benchmark
def benchmark():
    import random
    data = list(range(5000))
    random.shuffle(data)
    target = data[100] + data[4000]

    start = time.time()
    result1 = slow_search(data, target)
    slow_time = time.time() - start

    start = time.time()
    result2 = fast_search(data, target)
    fast_time = time.time() - start

    print(f"Inefficient version: {slow_time:.4f}s")
    print(f"Efficient version:   {fast_time:.6f}s")
    print(f"Speedup:             {slow_time/fast_time:.0f}x")

benchmark()
```

**Key points:**
- Be aware of algorithmic complexity
- Choose appropriate data structures
- Measure the effect with benchmarks

---

## Troubleshooting

### Common errors and solutions

| Error | Cause | Solution |
|--------|------|--------|
| Initialization error | Misconfigured config file | Verify the path and format of the configuration file |
| Timeout | Network latency / lack of resources | Adjust the timeout value, add retry logic |
| Out of memory | Increased data volume | Introduce batch processing, implement pagination |
| Permission error | Insufficient access rights | Verify the executing user's permissions, review settings |
| Data inconsistency | Race conditions in concurrent processing | Introduce locking mechanisms, manage transactions |

### Debugging procedure

1. **Check the error message**: Read the stack trace and identify where the error occurred
2. **Establish reproduction steps**: Reproduce the error with minimal code
3. **Form hypotheses**: List the possible causes
4. **Verify step by step**: Use logs and debuggers to test the hypotheses
5. **Fix and regression test**: After fixing, also run tests on related areas

```python
# Debugging utilities
import logging
import traceback
from functools import wraps

# Logger configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def debug_decorator(func):
    """Decorator that logs function inputs and outputs"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Call: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Return: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"Exception: {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

@debug_decorator
def process_data(items):
    """Data processing (debug target)"""
    if not items:
        raise ValueError("Empty data")
    return [item * 2 for item in items]
```

### Diagnosing performance issues

Procedure for diagnosing performance issues:

1. **Identify the bottleneck**: Measure with profiling tools
2. **Check memory usage**: Look for memory leaks
3. **Check I/O wait**: Inspect disk and network I/O
4. **Check concurrent connections**: Inspect the connection pool state

| Issue type | Diagnostic tool | Countermeasure |
|-----------|-----------|------|
| CPU load | cProfile, py-spy | Algorithmic improvement, parallelization |
| Memory leak | tracemalloc, objgraph | Properly release references |
| I/O bottleneck | strace, iostat | Async I/O, caching |
| DB latency | EXPLAIN, slow query log | Indexes, query optimization |
---

## FAQ

### Q1: Should I use PyO3 or ctypes/cffi?

**A**: PyO3 is recommended. ctypes/cffi call the C ABI directly, so they lack type safety and require manual marshalling. PyO3 leverages Rust's type system to automatically convert to and from Python types, and error handling is natural. If you only want to call an existing C library from Python, ctypes is the lowest-effort option.

### Q2: How do you debug FFI?

**A**: Combine the following:
1. **Valgrind / AddressSanitizer**: Detect memory errors
2. **RUST_BACKTRACE=1**: Stack traces on the Rust side
3. **Logging**: Record inputs and outputs at the FFI boundary
4. **Tests**: Thoroughly test the safe wrapper on the Rust side before exposing it via FFI

### Q3: What is the difference between napi-rs and N-API (node-addon-api)?

**A**: N-API is the official API for writing Node.js add-ons in C/C++; napi-rs is a wrapper that lets you use it from Rust. You gain the benefits of Rust's memory safety, performance is roughly equivalent, and the risk of memory bugs drops significantly.

### Q4: Should I use cxx or bindgen?

**A**: bindgen is recommended for C libraries; cxx is recommended for C++ libraries. cxx can directly map Rust and C++ types and provides safe type conversions for String, Vec, UniquePtr, etc. bindgen auto-generates from C headers but is not good at handling C++ templates and overloads.

### Q5: How can I efficiently pass large data across FFI?

**A**: Avoiding copies is essential. Pass it as a slice using a pointer-and-length pair, or use shared memory.

```rust
/// [OK] Pass a byte sequence with zero copy
#[no_mangle]
pub extern "C" fn process_buffer(data: *const u8, len: usize) -> i32 {
    if data.is_null() || len == 0 {
        return -1;
    }
    // Reference as a slice without copying the data
    let slice = unsafe { std::slice::from_raw_parts(data, len) };
    // ... processing ...
    0
}

/// [NG] Copy the data and process it (inefficient for large data)
#[no_mangle]
pub extern "C" fn bad_process_buffer(data: *const u8, len: usize) -> i32 {
    let slice = unsafe { std::slice::from_raw_parts(data, len) };
    let copied: Vec<u8> = slice.to_vec(); // Unnecessary copy!
    // ... processing ...
    0
}
```

### Q6: What is UniFFI?

**A**: It is a multi-language binding generator developed by Mozilla. It auto-generates bindings for Kotlin, Swift, Python, and Ruby from a UDL (Universal Definition Language) file. It is ideal for sharing Rust core logic across mobile apps.

```rust
// UDL file (my_lib.udl):
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

// Rust implementation
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

## Best practices for FFI testing

### Code example: Testing the FFI wrapper

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

        // Prevent memory leaks: always free
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

### CI configuration for an FFI project

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
      - name: Run Miri (memory safety check)
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

## Summary

| Item | Key point |
|---|---|
| FFI fundamentals | Expose C ABI-compatible functions with `extern "C"` + `#[no_mangle]` |
| Memory management | Make ownership responsibility clear. Always provide creation and deallocation as a pair |
| CString / CStr | Use CString for Rust → C, CStr for C → Rust |
| repr(C) | Required when sharing structs with C |
| bindgen | Auto-generate Rust bindings from C headers |
| cxx | Type-safe bidirectional integration with C++ |
| PyO3 | Write Python extensions in Rust. Expect 10-50x speedups |
| napi-rs | Write Node.js native add-ons in Rust. Async support |
| UniFFI | Multi-language bindings for Kotlin/Swift/Python |
| Safety | Prevent panics across the FFI boundary, null checks, error handling |
| catch_unwind | Catch panics at the FFI boundary to prevent undefined behavior |

## What to read next

- [Best practices](../04-ecosystem/04-best-practices.md) — General Rust design principles and quality management
- Asynchronous programming — Design patterns for async FFI

## References

1. **The Rust Reference**: [FFI](https://doc.rust-lang.org/reference/items/external-blocks.html) — Official FFI specification
2. **PyO3 Official Guide**: [PyO3 User Guide](https://pyo3.rs/) — Comprehensive PyO3 documentation
3. **napi-rs Official**: [NAPI-RS Documentation](https://napi.rs/) — How to use napi-rs with samples
4. **cxx Guide**: [CXX Documentation](https://cxx.rs/) — Type mapping and usage of cxx
5. **UniFFI**: [UniFFI User Guide](https://mozilla.github.io/uniffi-rs/) — Generating multi-language bindings
