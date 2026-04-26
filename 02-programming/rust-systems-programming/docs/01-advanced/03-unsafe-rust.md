# unsafe Rust -- Low-Level Programming Beyond the Boundary of Safety

> unsafe Rust enables operations that go beyond the constraints of the borrow checker (raw pointer manipulation, FFI, hardware access, etc.), and by encapsulating them inside safe abstractions, it preserves the safety of high-level APIs.

---

## What You Will Learn in This Chapter

1. **The 5 Superpowers of unsafe** -- Understand the five operations permitted inside an unsafe block
2. **Raw Pointers and FFI** -- Master the manipulation of *const T / *mut T and how to interface with C
3. **Safe Abstractions** -- Learn the patterns for confining unsafe internally and exposing safe APIs
4. **unsafe Traits** -- Understand manual implementation of Send / Sync and the design of custom unsafe traits
5. **Undefined Behavior and Miri** -- Master the kinds of UB and how to use detection tools


## Prerequisites

Reading the following before this guide will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding the contents of [Closures and Fn Traits -- Function Objects That Capture Their Environment](./02-closures-fn-traits.md)

---

## 1. The 5 Superpowers of unsafe

```
┌──────────────────────────────────────────────────────┐
│        The 5 Operations Unlocked by unsafe           │
├──────────────────────────────────────────────────────┤
│ 1. Dereferencing raw pointers (*const T / *mut T)    │
│ 2. Calling unsafe functions and methods              │
│ 3. Accessing/modifying mutable static variables      │
│ 4. Implementing unsafe traits                        │
│ 5. Accessing fields of a union                       │
├──────────────────────────────────────────────────────┤
│  Important: unsafe does NOT disable borrow checking! │
│  Ownership and type checks are still in force inside │
│  unsafe. unsafe means "the programmer takes          │
│  responsibility for safety that the compiler cannot  │
│  verify."                                            │
└──────────────────────────────────────────────────────┘
```

### 1.1 What unsafe Does NOT Do

An unsafe block does not disable the following checks:
- **Ownership rules**: Use after move is still an error
- **Type checking**: Type mismatches are still errors
- **Borrow rules**: Simultaneous use of `&T` and `&mut T` is still an error (though it can be circumvented using raw pointers)
- **Lifetime checking**: Reference lifetimes are still verified

### Example 1: Raw Pointer Basics

```rust
fn main() {
    let mut num = 42;

    // Creating a raw pointer is safe (dereferencing is unsafe)
    let r1 = &num as *const i32;
    let r2 = &mut num as *mut i32;

    unsafe {
        println!("r1 = {}", *r1);
        *r2 = 100;
        println!("r2 = {}", *r2);
    }

    // A raw pointer to an arbitrary address (extremely dangerous)
    let address = 0x012345usize;
    let _r = address as *const i32;
    // unsafe { println!("{}", *_r); } // Undefined behavior!

    // Null pointer
    let null_ptr: *const i32 = std::ptr::null();
    println!("null pointer: {:?}", null_ptr);
    assert!(null_ptr.is_null());

    // Raw pointer arithmetic
    let arr = [10, 20, 30, 40, 50];
    let ptr = arr.as_ptr();
    unsafe {
        for i in 0..arr.len() {
            println!("arr[{}] = {}", i, *ptr.add(i));
        }
    }
}
```

### Example 2: Conversion Between Raw Pointers and References

```rust
fn main() {
    let mut value = 42;

    // Reference -> raw pointer (safe)
    let raw_const: *const i32 = &value;
    let raw_mut: *mut i32 = &mut value;

    // Raw pointer -> reference (unsafe)
    unsafe {
        let ref_const: &i32 = &*raw_const;
        let ref_mut: &mut i32 = &mut *raw_mut;
        println!("const ref: {}", ref_const);
        *ref_mut = 100;
        println!("mut ref: {}", ref_mut);
    }

    // NonNull: a raw pointer guaranteed to be non-null
    let non_null = std::ptr::NonNull::new(&mut value as *mut i32).unwrap();
    unsafe {
        println!("NonNull: {}", *non_null.as_ptr());
    }

    // Raw pointer of a slice
    let slice = &[1, 2, 3, 4, 5];
    let ptr = slice.as_ptr();
    let len = slice.len();
    unsafe {
        // Reconstructing a slice from a raw pointer
        let reconstructed = std::slice::from_raw_parts(ptr, len);
        println!("reconstructed: {:?}", reconstructed);
    }
}
```

---

## 2. unsafe Functions

### Example 3: Defining and Calling unsafe Functions

```rust
/// Swaps the contents of two slices in place.
/// # Safety
/// - `a` and `b` must have the same length
/// - `a` and `b` must not overlap
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

    // Verify the safety preconditions before calling
    assert_eq!(a.len(), b.len());
    unsafe {
        swap_buffers(&mut a, &mut b);
    }
    println!("a={:?}, b={:?}", a, b); // a=[4,5,6], b=[1,2,3]
}
```

### Example 4: Implementing split_at_mut (Standard Library Internals)

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

### Example 5: Zero-Cost Type Conversions Using unsafe

```rust
// repr(transparent) guarantees identical internal representation
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

// Slice type conversion (zero-cost)
fn meters_slice_to_f64(meters: &[Meters]) -> &[f64] {
    // Safety: Meters is repr(transparent) and has the same layout as f64
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
    println!("raw values: {:?}", raw_values); // [1000.0, 2500.0, 500.0]

    // A safe alternative to std::mem::transmute
    let x: u32 = 0x41424344;
    let bytes: [u8; 4] = x.to_ne_bytes();
    println!("bytes: {:?}", bytes);
}
```

---

## 3. FFI (Foreign Function Interface)

### Example 6: Calling C Functions

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
        // Calling abs
        println!("abs(-5) = {}", abs(-5));
        println!("abs(10) = {}", abs(10));

        // Calling strlen
        let s = std::ffi::CString::new("hello").unwrap();
        println!("strlen = {}", strlen(s.as_ptr()));

        // Calling memcpy
        let src = [1u8, 2, 3, 4, 5];
        let mut dest = [0u8; 5];
        memcpy(
            dest.as_mut_ptr() as *mut std::os::raw::c_void,
            src.as_ptr() as *const std::os::raw::c_void,
            src.len(),
        );
        println!("memcpy result: {:?}", dest); // [1, 2, 3, 4, 5]
    }
}
```

### Example 7: Exposing Functions from Rust to C

```rust
/// A function callable from C
#[no_mangle]
pub extern "C" fn rust_add(a: i32, b: i32) -> i32 {
    a + b
}

/// Receives a callback from C
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

/// Receives and processes a string
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
        c_string.into_raw() // The caller must free it via rust_free_string
    }
}

/// Frees a string allocated by Rust
#[no_mangle]
pub extern "C" fn rust_free_string(s: *mut std::os::raw::c_char) {
    if !s.is_null() {
        unsafe {
            let _ = std::ffi::CString::from_raw(s);
            // CString is dropped -> memory is freed
        }
    }
}

fn main() {
    println!("rust_add(3, 4) = {}", rust_add(3, 4));
}
```

### The Safety Boundary of FFI

```
  ┌────────────────────────────────┐
  │  Rust (the safe world)         │
  │                                │
  │  pub fn safe_api(input: &str)  │
  │     │                          │
  │     ▼                          │
  │  ┌──────────────────────────┐  │
  │  │ unsafe {                 │  │
  │  │   // Validate input      │  │
  │  │   // Convert to CString  │  │
  │  │   // Call C function     │  │
  │  │   // Validate result     │  │
  │  │   // Convert to Rust type│  │
  │  │ }                        │  │
  │  └──────────────────────────┘  │
  │     │                          │
  │     ▼                          │
  │  Returns Result<T, E>          │
  └────────────────────────────────┘
           │
           ▼
  ┌────────────────────────────────┐
  │  C library (the unsafe world)  │
  └────────────────────────────────┘
```

### Example 8: Implementing a Safe FFI Wrapper

```rust
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

// Declare functions from a C library
extern "C" {
    fn setenv(name: *const c_char, value: *const c_char, overwrite: i32) -> i32;
    fn getenv(name: *const c_char) -> *const c_char;
}

// Safe wrapper
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
                EnvError::InvalidName(s) => write!(f, "invalid environment variable name: {}", s),
                EnvError::InvalidValue(s) => write!(f, "invalid value: {}", s),
                EnvError::SetFailed => write!(f, "failed to set environment variable"),
            }
        }
    }

    /// Safely sets an environment variable
    pub fn set_env(name: &str, value: &str) -> Result<(), EnvError> {
        // Validate input
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

        // Keep the unsafe block minimal
        let result = unsafe { setenv(c_name.as_ptr(), c_value.as_ptr(), 1) };

        if result == 0 {
            Ok(())
        } else {
            Err(EnvError::SetFailed)
        }
    }

    /// Safely retrieves an environment variable
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
    // Use through the safe API
    match safe_env::set_env("MY_VAR", "hello") {
        Ok(()) => println!("environment variable set"),
        Err(e) => println!("error: {}", e),
    }

    if let Some(value) = safe_env::get_env("MY_VAR") {
        println!("MY_VAR = {}", value);
    }
}
```

---

## 4. unsafe Traits

### Example 9: Implementing unsafe Traits

```rust
// Send / Sync are unsafe traits
// The compiler implements them automatically, but manual implementation is also possible

struct MyWrapper(*mut i32);

// Safety: MyWrapper manages its internal pointer appropriately and
// guarantees safe transmission across threads
unsafe impl Send for MyWrapper {}
unsafe impl Sync for MyWrapper {}

// Custom unsafe trait
/// # Safety
/// The implementor must guarantee that `process()` is only called when
/// `validate()` returns true
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
        // Assumes it is only called when validate() returns true
        println!("processing: {}", self.value);
    }
}

fn execute_validated<T: Validated>(item: &T) {
    if item.validate() {
        item.process();
    } else {
        println!("validation failed");
    }
}

fn main() {
    let data = SafeData { value: 42 };
    execute_validated(&data);

    let invalid = SafeData { value: -1 };
    execute_validated(&invalid);
}
```

### Example 10: Understanding and Manually Implementing Send / Sync

```rust
use std::cell::UnsafeCell;

// Implementing a thread-safe type using UnsafeCell
struct ThreadSafeCounter {
    count: UnsafeCell<u64>,
}

// Safety: Internal access is protected by atomic operations
// (in this example, Mutex-like protection is omitted for simplicity)
unsafe impl Send for ThreadSafeCounter {}
unsafe impl Sync for ThreadSafeCounter {}

impl ThreadSafeCounter {
    fn new() -> Self {
        ThreadSafeCounter {
            count: UnsafeCell::new(0),
        }
    }

    // Note: this implementation is for demonstration purposes; in practice atomic operations are required
    fn get(&self) -> u64 {
        unsafe { *self.count.get() }
    }
}

// Examples of types where Send / Sync are not auto-implemented
struct NotSend {
    data: *mut i32,  // Raw pointers do not implement Send/Sync
}

// Rc does not implement Send (because reference counts are not thread-safe)
// use std::rc::Rc;
// fn send_rc<T: Send>(t: T) {} // Passing Rc<T> would cause a compile error

fn main() {
    let counter = ThreadSafeCounter::new();
    println!("count: {}", counter.get());
}
```

---

## 5. Mutable static Variables and union

### Example 11: Mutable static and union

```rust
// Mutable static (global variable)
static mut COUNTER: u32 = 0;

fn increment_counter() {
    unsafe {
        COUNTER += 1;
    }
}

fn get_counter() -> u32 {
    unsafe { COUNTER }
}

// union: interprets the same memory region as different types
#[repr(C)]
union IntOrFloat {
    i: i32,
    f: f32,
}

fn main() {
    increment_counter();
    increment_counter();
    println!("counter: {}", get_counter());

    let u = IntOrFloat { f: 1.0 };
    unsafe {
        println!("as float: {}", u.f);
        println!("as int:   {:#010x}", u.i); // IEEE 754 representation
    }
}
```

### Example 12: Safe Alternatives to static mut

```rust
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

// Approach 1: AtomicU64 (atomic operations)
static ATOMIC_COUNTER: AtomicU64 = AtomicU64::new(0);

fn increment_atomic() {
    ATOMIC_COUNTER.fetch_add(1, Ordering::Relaxed);
}

fn get_atomic() -> u64 {
    ATOMIC_COUNTER.load(Ordering::Relaxed)
}

// Approach 2: OnceLock (a value initialized only once)
static CONFIG: OnceLock<String> = OnceLock::new();

fn get_config() -> &'static str {
    CONFIG.get_or_init(|| {
        // Initialization logic (executed only once)
        String::from("production")
    })
}

// Approach 3: Mutex (general-purpose mutable global state)
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
    println!("atomic counter: {}", get_atomic());

    // OnceLock
    println!("config: {}", get_config());
    println!("config (2nd): {}", get_config()); // same value

    // Mutex
    add_to_state("hello".to_string());
    add_to_state("world".to_string());
    println!("global state: {:?}", get_state());
}
```

### Example 13: Practical Use of union

```rust
// Using a union for a network protocol
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

// MaybeUninit: safe handling of uninitialized memory
fn demo_maybe_uninit() {
    use std::mem::MaybeUninit;

    // Efficiently construct an uninitialized array
    let mut arr: [MaybeUninit<String>; 5] = unsafe {
        MaybeUninit::uninit().assume_init()
    };

    for (i, elem) in arr.iter_mut().enumerate() {
        elem.write(format!("item_{}", i));
    }

    // Safely convert into a fully initialized array
    let arr: [String; 5] = unsafe {
        // transmute MaybeUninit<String> -> String
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

    println!("--- MaybeUninit demo ---");
    demo_maybe_uninit();
}
```

---

## 6. Safe Abstraction Patterns

```
┌────────────────────────────────────────────────────────┐
│             Principles of Safe Abstraction              │
├────────────────────────────────────────────────────────┤
│                                                        │
│ 1. Limit unsafe to the smallest possible block         │
│ 2. Expose a safe API; hide unsafe internally           │
│ 3. State preconditions in a # Safety doc comment       │
│ 4. Verify invariants with debug_assert!                │
│ 5. Detect undefined behavior with Miri                 │
│                                                        │
│  ┌──────────────────────────────────┐                  │
│  │  pub fn safe_function(...)       │  <- what users see│
│  │    -> Validate input             │                  │
│  │    -> unsafe { ... }            │  <- hidden inside│
│  │    -> Validate result            │                  │
│  │    -> Return as a safe type      │                  │
│  └──────────────────────────────────┘                  │
└────────────────────────────────────────────────────────┘
```

### Example 14: A Complete Example of Safe Abstraction

```rust
/// A fixed-size ring buffer
///
/// It uses unsafe internally, but the public API is fully safe.
pub struct RingBuffer<T> {
    buffer: Box<[std::mem::MaybeUninit<T>]>,
    head: usize,
    tail: usize,
    len: usize,
    capacity: usize,
}

impl<T> RingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "capacity must be at least 1");
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
            // Buffer is full: evict the oldest element
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
        // Properly drop the initialized elements
        while self.pop().is_some() {}
    }
}

fn main() {
    let mut buf = RingBuffer::new(3);
    buf.push(1);
    buf.push(2);
    buf.push(3);
    println!("buffer size: {}", buf.len()); // 3

    let evicted = buf.push(4); // 1 is evicted
    println!("evicted value: {:?}", evicted); // Some(1)

    while let Some(val) = buf.pop() {
        println!("popped: {}", val); // 2, 3, 4
    }
}
```

### Example 15: High-Performance String Processing Using unsafe

```rust
/// Constructs a String while skipping UTF-8 validation.
/// Use only when the caller guarantees UTF-8 validity.
///
/// # Safety
/// `bytes` must be valid UTF-8.
unsafe fn string_from_utf8_unchecked(bytes: Vec<u8>) -> String {
    String::from_utf8_unchecked(bytes)
}

/// Converts an ASCII string to uppercase (in place).
/// Faster than the standard to_uppercase() (no allocation).
fn ascii_uppercase_inplace(s: &mut String) {
    // Safety: ASCII uppercase conversion preserves UTF-8 validity
    // (ASCII characters are 1 byte, and remain in the ASCII range after uppercasing)
    unsafe {
        let bytes = s.as_bytes_mut();
        for byte in bytes.iter_mut() {
            if *byte >= b'a' && *byte <= b'z' {
                *byte -= 32; // 'a' - 'A' = 32
            }
        }
    }
}

/// Safely extracts a substring from a byte sequence
pub fn safe_substring(s: &str, start: usize, end: usize) -> Option<&str> {
    if start > end || end > s.len() {
        return None;
    }

    // Check UTF-8 character boundaries
    if !s.is_char_boundary(start) || !s.is_char_boundary(end) {
        return None;
    }

    // Safety: boundaries have been checked
    Some(unsafe { s.get_unchecked(start..end) })
}

fn main() {
    // ASCII uppercase conversion
    let mut s = String::from("hello, world!");
    ascii_uppercase_inplace(&mut s);
    println!("uppercase: {}", s); // HELLO, WORLD!

    // Safe substring extraction
    let text = "こんにちは、世界！";
    match safe_substring(text, 0, 15) {
        Some(sub) => println!("substring: {}", sub),
        None => println!("invalid range"),
    }

    // Detection of UTF-8 byte boundary errors
    match safe_substring(text, 0, 1) {
        Some(sub) => println!("substring: {}", sub),
        None => println!("UTF-8 boundary error"), // Japanese characters are 3 bytes each
    }

    match safe_substring(text, 0, 3) {
        Some(sub) => println!("substring: {}", sub), // "こ"
        None => println!("UTF-8 boundary error"),
    }
}
```

---

## 7. Undefined Behavior

### 7.1 Kinds of Undefined Behavior in Rust

```
┌────────────────────────────────────────────────────────┐
│          Major Kinds of Undefined Behavior (UB)         │
├────────────────────────────────────────────────────────┤
│                                                        │
│ 1. Dereferencing a dangling pointer                    │
│ 2. Dereferencing a misaligned pointer                  │
│ 3. Data race                                           │
│ 4. Producing an invalid value (e.g., 2 in a bool)      │
│ 5. Producing a null reference                          │
│ 6. Reading uninitialized memory                        │
│ 7. Aliasing rule violations (&T and &mut T coexist)    │
│ 8. Invalid enum discriminant                           │
│ 9. Unwinding propagating across an extern "C" boundary │
│ 10. Producing a non-UTF-8 &str                         │
│                                                        │
│ When UB occurs:                                        │
│ - Compiler optimizations may produce wrong results     │
│ - Crashes, data corruption, security vulnerabilities   │
│ - "Works today but breaks tomorrow" code               │
└────────────────────────────────────────────────────────┘
```

### Example 16: Typical UB Patterns

```rust
fn demonstrate_ub_patterns() {
    // UB pattern 1: dangling pointer
    // let ptr: *const i32;
    // {
    //     let x = 42;
    //     ptr = &x;
    // }
    // unsafe { println!("{}", *ptr); }  // UB!

    // UB pattern 2: simultaneous existence of &T and &mut T
    // let mut x = 42;
    // let r1 = &x as *const i32;
    // let r2 = &mut x as *mut i32;
    // unsafe {
    //     *r2 = 100;
    //     println!("{}", *r1);  // UB! aliasing violation
    // }

    // UB pattern 3: invalid value
    // let b: bool = unsafe { std::mem::transmute(2u8) };  // UB! bool may only be 0 or 1

    // UB pattern 4: alignment violation
    // let bytes = [0u8; 8];
    // let ptr = bytes.as_ptr().add(1) as *const u32;
    // unsafe { println!("{}", *ptr); }  // UB! alignment violation

    // Safe alternative
    let bytes = [0u8; 8];
    let value = u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    println!("safe read: {}", value);
}

fn main() {
    demonstrate_ub_patterns();
}
```

### 7.2 UB Detection with Miri

```bash
# Install Miri
rustup +nightly component add miri

# Run tests under Miri
cargo +nightly miri test

# Run only a specific test
cargo +nightly miri test test_name

# Run a binary under Miri
cargo +nightly miri run
```

```rust
// Example of issues Miri can detect
#[cfg(test)]
mod tests {
    #[test]
    fn test_valid_code() {
        let mut v = vec![1, 2, 3];
        let ptr = v.as_ptr();
        v.push(4);
        // Miri: ptr may no longer be valid
        // (if push triggered a reallocation)
    }

    #[test]
    fn test_safe_access() {
        let v = vec![1, 2, 3];
        let ptr = v.as_ptr();
        unsafe {
            // No push has happened, so ptr is still valid
            assert_eq!(*ptr, 1);
            assert_eq!(*ptr.add(1), 2);
            assert_eq!(*ptr.add(2), 3);
        }
    }
}
```

---

## 8. Comparison Tables

### 8.1 Scope of safe vs unsafe

| Operation | safe Rust | unsafe Rust |
|-----------|-----------|-------------|
| Reference dereference | Always safe | Possible via raw pointers |
| Array access | With bounds checking | Can be skipped via get_unchecked |
| Type conversion | From/Into | Arbitrary conversion via transmute |
| FFI | Not allowed | Allowed within an extern block |
| Global mutable state | Not allowed | Allowed via static mut |
| Lifetimes | Verified by the compiler | Guaranteed by the programmer |
| union access | Not allowed | Allowed inside an unsafe block |

### 8.2 Alternatives to unsafe

| What you want to do | Consider before unsafe | When unsafe is necessary |
|---------------------|------------------------|--------------------------|
| Multiple mutable references | RefCell, Mutex | split_at_mut-like splitting |
| Global state | OnceLock, AtomicU32 | static mut (not recommended) |
| Reinterpreting types | From/TryFrom | transmute |
| C interop | Safe wrapper crates | Direct extern "C" |
| Performance | Algorithmic improvements | get_unchecked, etc. |
| Self-referential types | Pin, ouroboros crate | Raw pointers |
| Lazy initialization | OnceLock, LazyLock | MaybeUninit |

### 8.3 Comparison of repr Attributes

| Attribute | Meaning | Use case |
|-----------|---------|----------|
| `#[repr(Rust)]` | Default. The compiler is free to choose the layout | Ordinary structs |
| `#[repr(C)]` | C-compatible layout | FFI, unions |
| `#[repr(transparent)]` | Same layout as the inner type | Newtype pattern |
| `#[repr(packed)]` | No padding | Memory savings (watch out for alignment) |
| `#[repr(align(N))]` | N-byte alignment | SIMD, cache lines |

---

## 9. Anti-Patterns

### Anti-Pattern 1: Overly Broad unsafe Block

```rust
// BAD: the unsafe block is too large
fn bad_example(data: &[u8]) -> u8 {
    unsafe {
        let processed = data.iter().sum::<u8>();  // safe operation
        let ptr = data.as_ptr();                   // safe operation
        let value = *ptr;                          // <- only this is unsafe
        processed.wrapping_add(value)              // safe operation
    }
}

// GOOD: keep unsafe minimal
fn good_example(data: &[u8]) -> u8 {
    let processed: u8 = data.iter().sum();
    let ptr = data.as_ptr();
    let value = unsafe { *ptr };  // unsafe only where it is truly necessary
    processed.wrapping_add(value)
}

fn main() {
    let data = vec![1u8, 2, 3, 4, 5];
    println!("bad: {}", bad_example(&data));
    println!("good: {}", good_example(&data));
}
```

### Anti-Pattern 2: Missing Safety Documentation

```rust
// BAD: no explanation of why it is unsafe
pub unsafe fn do_thing_bad(ptr: *const u8, len: usize) -> Vec<u8> {
    std::slice::from_raw_parts(ptr, len).to_vec()
}

// GOOD: state the preconditions explicitly
/// Reads data from a buffer.
///
/// # Safety
///
/// - `ptr` must point to a valid, readable memory region
/// - `len` bytes of memory starting at `ptr` must be valid
/// - The memory must not be modified by another thread during execution of this function
/// - `ptr` must be properly aligned for u8
pub unsafe fn do_thing_good(ptr: *const u8, len: usize) -> Vec<u8> {
    debug_assert!(!ptr.is_null(), "a null pointer was passed");
    std::slice::from_raw_parts(ptr, len).to_vec()
}

fn main() {
    let data = vec![1u8, 2, 3, 4, 5];
    let result = unsafe { do_thing_good(data.as_ptr(), data.len()) };
    println!("{:?}", result);
}
```

### Anti-Pattern 3: Abuse of transmute

```rust
// BAD: destroying type safety with transmute
fn bad_transmute() {
    // let x: f32 = unsafe { std::mem::transmute(0x42280000u32) };
    // Dangerous: endianness-dependent, will break with future repr changes
}

// GOOD: use safe type-conversion methods
fn good_conversion() {
    // Get the bit pattern of f32
    let x: f32 = 42.0;
    let bits = x.to_bits();
    println!("bits of f32 {}: {:#010x}", x, bits);

    // Reconstruct f32 from a bit pattern
    let restored = f32::from_bits(bits);
    println!("restored: {}", restored);

    // Conversion via byte sequence
    let bytes = x.to_ne_bytes();
    let from_bytes = f32::from_ne_bytes(bytes);
    println!("via bytes: {}", from_bytes);
}

// BAD: transmute on an enum
// fn bad_enum() {
//     let x: Option<i32> = unsafe { std::mem::transmute(42i32) };
//     // UB! The internal representation of Option<i32> is not guaranteed
// }

fn main() {
    good_conversion();
}
```

### Anti-Pattern 4: Bypassing Borrow Checking with unsafe

```rust
// BAD: trying to bypass the borrow rules with unsafe
fn bad_alias() {
    let mut data = vec![1, 2, 3];
    // let ptr = data.as_mut_ptr();
    // let r1 = &data; // immutable reference
    // unsafe { *ptr = 42; } // mutable access -> UB!
    // println!("{:?}", r1); // r1 is still in use
}

// GOOD: use a safe API
fn good_design() {
    let mut data = vec![1, 2, 3];

    // Interior mutability with Cell/RefCell
    use std::cell::RefCell;
    let data = RefCell::new(vec![1, 2, 3]);
    {
        let r1 = data.borrow();
        println!("{:?}", r1);
    } // drop r1
    data.borrow_mut()[0] = 42;
    println!("{:?}", data.borrow());
}

fn main() {
    bad_alias();
    good_design();
}
```

---

## 10. In Practice: High-Performance Data Structures Using unsafe

### Example 17: An Intrusive Linked List

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

    println!("list size: {}", list.len());

    while let Some(val) = list.pop_front() {
        println!("  {}", val);
    }
}
```


---

## Hands-On Exercises

### Exercise 1: Basic Implementation

Implement code that meets the following requirements.

**Requirements:**
- Validate input data
- Handle errors appropriately
- Also write test code

```python
# Exercise 1: Template for basic implementation
class Exercise1:
    """Exercise on basic implementation patterns"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """Validate the input value"""
        if value is None:
            raise ValueError("input value is None")
        return True

    def process(self, value):
        """Main data processing logic"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """Retrieve the processing results"""
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
        assert False, "an exception should have been raised"
    except ValueError:
        pass

    print("All tests passed!")

test_exercise1()
```

### Exercise 2: Advanced Patterns

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
        """Search by key"""
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
    assert ex.add("d", 4) == False  # size limit
    assert ex.find("b")['value'] == 2
    assert ex.remove("b") == True
    assert ex.find("b") is None
    stats = ex.stats()
    assert stats['total_items'] == 2
    print("All advanced tests passed!")

test_advanced()
```

### Exercise 3: Performance Optimization

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
    print(f"Speedup: {slow_time/fast_time:.0f}x")

benchmark()
```

**Key points:**
- Be conscious of the algorithmic complexity
- Choose appropriate data structures
- Measure the effect with benchmarks

---

## Troubleshooting

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| Initialization error | Issues with the configuration file | Check the path and format of the configuration file |
| Timeout | Network latency / resource shortage | Adjust timeout values, add retry logic |
| Out of memory | Growth in data volume | Introduce batch processing, implement pagination |
| Permission error | Lack of access rights | Check the executing user's permissions, review settings |
| Data inconsistency | Concurrency conflicts | Introduce locking mechanisms, manage transactions |

### Debugging Procedure

1. **Check the error message**: Read the stack trace and identify the location of the failure
2. **Establish reproduction steps**: Reproduce the error with the minimum amount of code
3. **Form hypotheses**: List possible causes
4. **Verify step by step**: Use logging output and a debugger to verify your hypotheses
5. **Fix and run regression tests**: After fixing, also run tests for related areas

```python
# Utility for debugging
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
    """Decorator that logs the inputs and outputs of a function"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"call: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"return: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"exception: {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

@debug_decorator
def process_data(items):
    """Data processing (debug target)"""
    if not items:
        raise ValueError("empty data")
    return [item * 2 for item in items]
```

### Diagnosing Performance Issues

The diagnosis procedure when performance issues occur:

1. **Identify the bottleneck**: Measure with profiling tools
2. **Check memory usage**: Check whether there are any memory leaks
3. **Check I/O waits**: Check disk and network I/O conditions
4. **Check the number of concurrent connections**: Check the state of the connection pool

| Issue type | Diagnostic tool | Countermeasure |
|------------|-----------------|----------------|
| CPU load | cProfile, py-spy | Algorithmic improvements, parallelization |
| Memory leaks | tracemalloc, objgraph | Properly release references |
| I/O bottleneck | strace, iostat | Asynchronous I/O, caching |
| DB latency | EXPLAIN, slow query log | Indexes, query optimization |

---

## Design Decision Guide

### Selection Criteria Matrix

The following summarizes the decision criteria when making technology choices.

| Criterion | When to prioritize | When you can compromise |
|-----------|---------------------|--------------------------|
| Performance | Real-time processing, large-scale data | Admin screens, batch processing |
| Maintainability | Long-term operation, team development | Prototypes, short-term projects |
| Scalability | Services with expected growth | Internal tools, fixed user base |
| Security | Personal information, financial data | Public data, internal use |
| Development speed | MVP, time to market | Quality-focused, mission-critical |

### Choosing an Architectural Pattern

```
┌─────────────────────────────────────────────────┐
│           Architecture Selection Flow             │
├─────────────────────────────────────────────────┤
│                                                 │
│  (1) What is the team size?                      │
│    |- Small (1-5 people) -> Monolith              │
│    |- Large (10+ people) -> Go to (2)             │
│                                                 │
│  (2) What is the deployment frequency?            │
│    |- Once a week or less -> Monolith + modular   │
│    |- Daily/multiple times -> Go to (3)           │
│                                                 │
│  (3) How independent are the teams?               │
│    |- High -> Microservices                       │
│    |- Medium -> Modular monolith                  │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Analyzing Trade-offs

Technical decisions always involve trade-offs. Analyze them from the following perspectives:

**1. Short-term vs long-term cost**
- A method that is fast in the short term can become technical debt in the long term
- Conversely, over-engineering has a high short-term cost and may delay the project

**2. Consistency vs flexibility**
- A unified technology stack has a low learning cost
- Adopting diverse technologies enables right-tool-for-the-job choices, but increases operational costs

**3. Level of abstraction**
- Higher abstraction has higher reusability but can make debugging difficult
- Lower abstraction is intuitive but tends to produce code duplication

```python
# Template for recording design decisions
class ArchitectureDecisionRecord:
    """Creating an ADR (Architecture Decision Record)"""

    def __init__(self, title: str):
        self.title = title
        self.context = ""
        self.decision = ""
        self.consequences = []
        self.alternatives = []

    def set_context(self, context: str):
        """Describe the background and the problem"""
        self.context = context
        return self

    def set_decision(self, decision: str):
        """Describe the decision content"""
        self.decision = decision
        return self

    def add_consequence(self, consequence: str, positive: bool = True):
        """Add a consequence"""
        self.consequences.append({
            'description': consequence,
            'type': 'positive' if positive else 'negative'
        })
        return self

    def add_alternative(self, name: str, reason_rejected: str):
        """Add a rejected alternative"""
        self.alternatives.append({
            'name': name,
            'reason_rejected': reason_rejected
        })
        return self

    def to_markdown(self) -> str:
        """Output in Markdown format"""
        md = f"# ADR: {self.title}\n\n"
        md += f"## Context\n{self.context}\n\n"
        md += f"## Decision\n{self.decision}\n\n"
        md += "## Consequences\n"
        for c in self.consequences:
            icon = "[+]" if c['type'] == 'positive' else "[!]"
            md += f"- {icon} {c['description']}\n"
        md += "\n## Rejected Alternatives\n"
        for a in self.alternatives:
            md += f"- **{a['name']}**: {a['reason_rejected']}\n"
        return md
```
---

## 11. FAQ

### Q1: Does using unsafe lose Rust's safety?

**A:** Partially. Inside an unsafe block, the responsibility for memory safety shifts to the programmer, but outside unsafe the compiler's guarantees are still maintained. The important thing is to keep unsafe minimal and wrap it in safe abstractions.

### Q2: What is Miri?

**A:** Miri is an interpreter for Rust's intermediate representation (MIR), and it is a tool that detects undefined behavior:
```bash
rustup +nightly component add miri
cargo +nightly miri test
```
It detects memory leaks, data races, and invalid pointer operations at runtime.

### Q3: When do you use transmute?

**A:** In most cases, you should not. `transmute` reinterprets a bit pattern as a different type, which is extremely dangerous. Instead, consider `as` casts, `From/TryFrom`, `to_bits/from_bits`, `to_ne_bytes/from_ne_bytes`, or the `bytemuck` crate.

### Q4: When is `#[repr(C)]` necessary?

**A:** It is mainly needed in the following cases:
- When exchanging data with C in FFI
- When you want to control the memory layout explicitly
- When using a union
- For struct definitions in file formats or network protocols

### Q5: unsafe fn vs an unsafe block, which should I use?

**A:** Use the following criteria to choose:
- **unsafe fn**: When the caller is responsible for satisfying the safety preconditions. Document the preconditions in a `# Safety` doc comment
- **unsafe block**: When the function itself is safe but unsafe operations are required internally. Validate inputs and expose it as a safe API

```rust
// unsafe fn: the caller bears responsibility
/// # Safety
/// `ptr` must be valid and `len` bytes must be readable
pub unsafe fn read_buffer(ptr: *const u8, len: usize) -> Vec<u8> {
    std::slice::from_raw_parts(ptr, len).to_vec()
}

// Safe function + internal unsafe: the function guarantees safety
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


## FAQ

### Q1: What is the most important point in learning this topic?

Gaining practical experience is most important. Beyond theory, your understanding deepens by actually writing code and verifying its behavior.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals and moving on to advanced topics. We recommend that you firmly understand the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this used in the workplace?

Knowledge of this topic is frequently used in everyday development work. It is particularly important during code reviews and when designing architecture.

---

## 12. Summary

| Concept | Key points |
|---------|-----------|
| unsafe block | Unlocks the 5 superpowers. Keep it minimal |
| Raw pointer | *const T / *mut T. Dereferencing is unsafe |
| FFI | Call/expose C functions via extern "C" |
| unsafe trait | Send/Sync, etc. The implementor guarantees safety |
| static mut | Global mutable state. Prefer Atomic or Mutex alternatives |
| union | Same memory interpreted as multiple types. Manage with a discriminant |
| Safe abstraction | Confine unsafe internally and expose a safe API |
| Miri | Tool for detecting undefined behavior |
| repr(C) | C-compatible memory layout |
| MaybeUninit | Safe handling of uninitialized memory |
| # Safety | Document the preconditions of an unsafe function |

---

## Recommended Next Reading

- [04-macros.md](04-macros.md) -- Generate safe abstractions with macros
- [../03-systems/02-ffi-interop.md](../03-systems/02-ffi-interop.md) -- FFI in practice (bindgen, PyO3)
- [../03-systems/00-memory-layout.md](../03-systems/00-memory-layout.md) -- Memory layout in detail

---

## References

1. **The Rust Programming Language - Ch.19.1 Unsafe Rust** -- https://doc.rust-lang.org/book/ch19-01-unsafe-rust.html
2. **The Rustonomicon** -- https://doc.rust-lang.org/nomicon/
3. **Miri - An Interpreter for Rust's MIR** -- https://github.com/rust-lang/miri
4. **Rust Unsafe Code Guidelines** -- https://rust-lang.github.io/unsafe-code-guidelines/
5. **Rust Reference - Unsafety** -- https://doc.rust-lang.org/reference/unsafety.html
6. **std::mem::MaybeUninit** -- https://doc.rust-lang.org/std/mem/union.MaybeUninit.html
