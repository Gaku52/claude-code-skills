# Memory Layout — Stack/Heap, repr

> Understand how Rust data types are laid out in memory, and master low-level optimization techniques using the repr attribute and alignment control.

## What You Will Learn in This Chapter

1. **Stack and Heap** — Characteristics of each region, allocation cost, and relationship with ownership
2. **Memory Layout of Types** — Size, alignment, padding, and the repr attribute
3. **Internal Structure of Smart Pointers** — Memory layout of Box, Vec, String, and Arc
4. **Advanced Memory Control** — Allocator API, memory-mapped I/O, and cache optimization
5. **Practical Memory Profiling** — Tools and techniques


## Prerequisites

Before reading this guide, the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts

---

## 1. Process Memory Map

```
┌──────────────────── Process Memory Space ─────────────┐
│  High Address                                          │
│  ┌─────────────────────────────────────────┐          │
│  │           Stack                           │ ↓ grows  │
│  │  - Local variables, function arguments    │          │
│  │  - Fixed size (typically 8MB)             │          │
│  │  - LIFO (ultra-fast alloc/dealloc)        │          │
│  └─────────────────────────────────────────┘          │
│                        ↕ (unused space)                 │
│  ┌─────────────────────────────────────────┐          │
│  │           Heap                            │ ↑ grows  │
│  │  - Dynamic allocation (Box, Vec, String) │          │
│  │  - Via OS allocator                       │          │
│  │  - Arbitrary size and lifetime            │          │
│  └─────────────────────────────────────────┘          │
│  ┌─────────────────────────────────────────┐          │
│  │  BSS (uninitialized static variables)     │          │
│  ├─────────────────────────────────────────┤          │
│  │  Data (initialized static variables)      │          │
│  ├─────────────────────────────────────────┤          │
│  │  Text (executable code) [read-only]       │          │
│  └─────────────────────────────────────────┘          │
│  Low Address                                           │
└────────────────────────────────────────────────────────┘
```

### Details of Each Segment

| Segment | Contents | Permissions | Size |
|---|---|---|---|
| Text | Machine instructions | Read + Execute | Fixed |
| Data | `static mut X: i32 = 42;` etc. | Read/Write | Fixed |
| BSS | `static mut Y: i32 = 0;` etc. | Read/Write | Fixed |
| Heap | `Box::new()`, `Vec::new()` | Read/Write | Variable (grows upward) |
| Stack | Local variables, return addresses | Read/Write | Variable (grows downward) |

### Code Example: Observing Memory Addresses

```rust
use std::mem;

static GLOBAL: i32 = 100;
static mut GLOBAL_MUT: i32 = 200;

fn main() {
    // Variables on the stack
    let stack_var: i32 = 42;
    let stack_arr: [u8; 16] = [0; 16];

    // Variables on the heap
    let heap_var = Box::new(42i32);
    let heap_vec: Vec<u8> = vec![0; 16];

    println!("=== Memory Address Observation ===");
    println!("Text segment:");
    println!("  main function: {:p}", main as *const ());

    println!("Data/BSS segment:");
    println!("  GLOBAL:       {:p}", &GLOBAL);
    unsafe {
        println!("  GLOBAL_MUT:   {:p}", &GLOBAL_MUT);
    }

    println!("Stack:");
    println!("  stack_var:    {:p}", &stack_var);
    println!("  stack_arr:    {:p}", &stack_arr);

    println!("Heap:");
    println!("  heap_var:     {:p}", &*heap_var);
    println!("  heap_vec[0]:  {:p}", heap_vec.as_ptr());

    // Stack pointer value > heap pointer value (typical)
    let stack_addr = &stack_var as *const i32 as usize;
    let heap_addr = &*heap_var as *const i32 as usize;
    println!("\nStack (0x{:x}) > Heap (0x{:x}): {}",
        stack_addr, heap_addr, stack_addr > heap_addr);
}
```

---

## 2. Stack vs Heap Comparison

### Code Example 1: Stack vs Heap Allocation

```rust
use std::mem;

fn main() {
    // Stack allocation: size determined at compile time
    let x: i32 = 42;              // 4 bytes on stack
    let arr: [u8; 1024] = [0; 1024]; // 1024 bytes on stack
    let point: (f64, f64) = (1.0, 2.0); // 16 bytes on stack

    // Heap allocation: size can be determined at runtime
    let boxed: Box<i32> = Box::new(42);    // pointer (8b) on stack, 4b on heap
    let vec: Vec<u8> = vec![0; 1024];      // 24b on stack, 1024b on heap
    let string: String = "hello".to_string(); // 24b on stack, 5b on heap

    println!("--- Sizes on the stack ---");
    println!("i32:        {} bytes", mem::size_of::<i32>());
    println!("[u8; 1024]: {} bytes", mem::size_of::<[u8; 1024]>());
    println!("(f64, f64): {} bytes", mem::size_of::<(f64, f64)>());
    println!("Box<i32>:   {} bytes", mem::size_of::<Box<i32>>());
    println!("Vec<u8>:    {} bytes", mem::size_of::<Vec<u8>>());
    println!("String:     {} bytes", mem::size_of::<String>());
    // Box<i32>:   8 bytes  (pointer only)
    // Vec<u8>:    24 bytes (ptr + len + capacity)
    // String:     24 bytes (ptr + len + capacity)
}
```

### Memory Layout of Smart Pointers

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

### Code Example: Detecting Stack Overflow

```rust
/// Example to verify the limit of stack size
fn recursive_stack_usage(depth: usize) {
    // Each call consumes about 1KB of stack
    let _buffer = [0u8; 1024];
    if depth > 0 {
        recursive_stack_usage(depth - 1);
    }
}

fn main() {
    // With the default stack size (8MB), about 8000 recursions is the limit
    // Stack overflow → process terminates with SIGSEGV

    // Safety measure: specify stack size with a thread builder
    let builder = std::thread::Builder::new()
        .name("large-stack".into())
        .stack_size(32 * 1024 * 1024); // 32MB

    let handle = builder.spawn(|| {
        recursive_stack_usage(30000); // plenty of room with 32MB
        println!("Recursion complete");
    }).unwrap();

    handle.join().unwrap();

    // Dynamic stack expansion is also possible with the stacker crate
    // stacker::maybe_grow(32 * 1024, 1024 * 1024, || { ... });
}
```

### Code Example: Vec Growth Strategy and Memory Reallocation

```rust
fn main() {
    let mut v: Vec<i32> = Vec::new();

    println!("=== Vec Growth Strategy ===");
    println!("{:>5} {:>10} {:>10} {:>18}", "len", "capacity", "size(B)", "ptr");

    let mut prev_ptr = v.as_ptr();
    for i in 0..33 {
        v.push(i);
        let ptr = v.as_ptr();
        let reallocated = if ptr != prev_ptr { " ← realloc!" } else { "" };
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
    // Example output:
    //   len   capacity   size(B)                ptr
    //     1          4        16   0x600000000010 ← realloc!
    //     5          8        32   0x600000000030 ← realloc!
    //     9         16        64   0x600000000050 ← realloc!
    //    17         32       128   0x600000000090 ← realloc!
    //    33         64       256   0x600000000110 ← realloc!

    // Pre-reserving with with_capacity avoids reallocation
    let v2: Vec<i32> = Vec::with_capacity(100);
    println!("\nwith_capacity(100): len={}, capacity={}", v2.len(), v2.capacity());

    // Free excess capacity with shrink_to_fit
    let mut v3 = vec![1, 2, 3, 4, 5];
    v3.reserve(1000);
    println!("After reserve: capacity={}", v3.capacity());
    v3.shrink_to_fit();
    println!("After shrink: capacity={}", v3.capacity());
}
```

### Code Example: Heap Allocation Benchmark

```rust
use std::time::Instant;

fn benchmark_stack_vs_heap() {
    const ITERATIONS: usize = 1_000_000;

    // Benchmark for stack allocation
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _data = [0u8; 256]; // 256 bytes on the stack
        std::hint::black_box(&_data);
    }
    let stack_time = start.elapsed();

    // Benchmark for heap allocation
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _data = Box::new([0u8; 256]); // 256 bytes on the heap
        std::hint::black_box(&_data);
    }
    let heap_time = start.elapsed();

    println!("Stack allocation: {:?} ({} times)", stack_time, ITERATIONS);
    println!("Heap allocation:  {:?} ({} times)", heap_time, ITERATIONS);
    println!("Heap/Stack ratio: {:.1}x",
        heap_time.as_nanos() as f64 / stack_time.as_nanos() as f64);
}

fn main() {
    benchmark_stack_vs_heap();
    // Typical results:
    // Stack allocation: 1.2ms (1000000 times)
    // Heap allocation:  25ms (1000000 times)
    // Heap/Stack ratio: 20.8x
}
```

---

## 3. Memory Layout of Types

### Code Example 2: Checking Size and Alignment

```rust
use std::mem;

#[repr(C)]
struct CLayout {
    a: u8,    // 1 byte + 3 padding
    b: u32,   // 4 bytes
    c: u8,    // 1 byte + 3 padding
}
// Size: 12 bytes (C-compatible layout)

struct RustLayout {
    a: u8,
    b: u32,
    c: u8,
}
// The Rust compiler optimizes → fields can be reordered
// Size: 8 bytes (optimized by ordering b, a, c)

fn main() {
    println!("CLayout:    size={}, align={}",
        mem::size_of::<CLayout>(), mem::align_of::<CLayout>());
    println!("RustLayout: size={}, align={}",
        mem::size_of::<RustLayout>(), mem::align_of::<RustLayout>());

    // Sizes of enums
    println!("Option<u8>:       {}", mem::size_of::<Option<u8>>());       // 2
    println!("Option<Box<u8>>:  {}", mem::size_of::<Option<Box<u8>>>()); // 8 (niche optimization!)
    println!("Option<&u8>:      {}", mem::size_of::<Option<&u8>>());     // 8 (null optimization!)
}
```

### Code Example 3: Types of repr Attributes

```rust
// repr(C) — same layout as C language (for FFI)
#[repr(C)]
struct FFIPoint {
    x: f64,
    y: f64,
}

// repr(transparent) — same layout as the inner single field
#[repr(transparent)]
struct Meters(f64);
// Meters and f64 are ABI-compatible → safe to cast in FFI

// repr(packed) — no padding (access may be slow)
#[repr(C, packed)]
struct PackedHeader {
    magic: u8,
    version: u32,  // no alignment → access may be slow
    length: u16,
}
// Size: 7 bytes (no padding)

// repr(align(N)) — specify minimum alignment
#[repr(align(64))]
struct CacheLine {
    data: [u8; 64],
}
// Aligned to cache line → prevents false sharing

// repr(u8/u16/u32...) — specify enum discriminant size
#[repr(u8)]
enum PacketType {
    Ping = 0,
    Pong = 1,
    Data = 2,
}
```

### Code Example: Visualizing Padding

```rust
use std::mem;

/// Macro to get the offset of a field
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
    b: u32,   // offset 0, size 4 — place largest-aligned type first
    d: u16,   // offset 4, size 2
    a: u8,    // offset 6, size 1
    c: u8,    // offset 7, size 1
}

fn main() {
    println!("=== Padding Analysis ===");
    println!("Example1 (inefficient layout):");
    println!("  size={}, align={}", mem::size_of::<Example1>(), mem::align_of::<Example1>());
    println!("  a: offset={}", offset_of!(Example1, a));
    println!("  b: offset={}", offset_of!(Example1, b));
    println!("  c: offset={}", offset_of!(Example1, c));
    println!("  d: offset={}", offset_of!(Example1, d));
    // size=12, padding=4 bytes

    println!("\nExample2 (efficient layout):");
    println!("  size={}, align={}", mem::size_of::<Example2>(), mem::align_of::<Example2>());
    println!("  b: offset={}", offset_of!(Example2, b));
    println!("  d: offset={}", offset_of!(Example2, d));
    println!("  a: offset={}", offset_of!(Example2, a));
    println!("  c: offset={}", offset_of!(Example2, c));
    // size=8, padding=0 bytes
}
```

### Code Example: Size and Alignment Table for Various Primitive Types

```rust
use std::mem;

fn print_layout<T>(name: &str) {
    println!("  {:<24} size={:>2}, align={:>2}",
        name,
        mem::size_of::<T>(),
        mem::align_of::<T>());
}

fn main() {
    println!("=== Primitive Types ===");
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

    println!("\n=== Pointer Types ===");
    print_layout::<*const u8>("*const u8");
    print_layout::<*const [u8]>("*const [u8] (fat ptr)");
    print_layout::<*const dyn std::fmt::Debug>("*const dyn Debug (fat ptr)");
    print_layout::<&u8>("&u8");
    print_layout::<&[u8]>("&[u8] (slice ref)");
    print_layout::<&dyn std::fmt::Debug>("&dyn Debug (trait obj)");

    println!("\n=== Collection Types ===");
    print_layout::<Vec<u8>>("Vec<u8>");
    print_layout::<String>("String");
    print_layout::<Box<u8>>("Box<u8>");
    print_layout::<std::collections::HashMap<u64, u64>>("HashMap<u64, u64>");
    print_layout::<std::collections::BTreeMap<u64, u64>>("BTreeMap<u64, u64>");
    print_layout::<std::collections::VecDeque<u8>>("VecDeque<u8>");

    println!("\n=== Smart Pointers ===");
    print_layout::<std::sync::Arc<u8>>("Arc<u8>");
    print_layout::<std::rc::Rc<u8>>("Rc<u8>");
    print_layout::<std::sync::Mutex<u8>>("Mutex<u8>");
    print_layout::<std::sync::RwLock<u8>>("RwLock<u8>");

    println!("\n=== Zero-Sized Types ===");
    print_layout::<()>("()");
    print_layout::<std::marker::PhantomData<u8>>("PhantomData<u8>");
    print_layout::<[u8; 0]>("[u8; 0]");
}
```

---

## 4. Memory Optimization of Enums

```
┌──────────── enum Memory Layout ─────────────────┐
│                                                  │
│  enum Shape {                                    │
│      Circle(f64),        // radius               │
│      Rect(f64, f64),     // width, height        │
│      Point,                                      │
│  }                                               │
│                                                  │
│  Memory: [tag: 8 bytes][data: 16 bytes] = 24b   │
│                                                  │
│  Circle: [0][radius: f64][padding: 8]           │
│  Rect:   [1][width: f64][height: f64]           │
│  Point:  [2][unused: 16]                        │
│                                                  │
│  ── Niche Optimization ──                        │
│                                                  │
│  Option<Box<T>>:                                 │
│    Some(ptr) → [ptr value]    (8 bytes)         │
│    None      → [0x0]          (8 bytes)         │
│  * Box guarantees non-NULL → 0 can be used as None │
│  * No tag needed! Same size as Box!              │
└──────────────────────────────────────────────────┘
```

### Code Example 4: Verifying Niche Optimization

```rust
use std::mem::size_of;
use std::num::NonZeroU64;

fn main() {
    // Without niche optimization
    println!("Option<u64>:       {} bytes", size_of::<Option<u64>>());       // 16

    // With niche optimization (using 0 of NonZero as None)
    println!("Option<NonZeroU64>: {} bytes", size_of::<Option<NonZeroU64>>()); // 8

    // Pointer types are niche-optimized
    println!("Option<Box<i32>>:   {} bytes", size_of::<Option<Box<i32>>>()); // 8
    println!("Option<&i32>:       {} bytes", size_of::<Option<&i32>>());     // 8
    println!("Option<String>:     {} bytes", size_of::<Option<String>>());   // 24 (same as String!)

    // Result is also optimized
    println!("Result<Box<i32>, Box<str>>: {} bytes",
        size_of::<Result<Box<i32>, Box<str>>>());  // 16
}
```

### Code Example: Detailed Analysis of enum Sizes

```rust
use std::mem;

// An enum where each variant has different data sizes
enum Message {
    Quit,                       // 0 bytes of data
    Move { x: i32, y: i32 },   // 8 bytes of data
    Write(String),              // 24 bytes of data
    Color(u8, u8, u8),          // 3 bytes of data
}

// Nested enum
enum Outer {
    A(Inner),
    B(u8),
}

enum Inner {
    X(u64),
    Y(u32),
}

fn main() {
    println!("=== enum Size Analysis ===");
    println!("Message:  size={}, align={}",
        mem::size_of::<Message>(), mem::align_of::<Message>());
    // Size = largest variant (Write: 24 bytes) + tag + padding

    println!("Outer:    size={}", mem::size_of::<Outer>());
    println!("Inner:    size={}", mem::size_of::<Inner>());

    // Niche optimization for nested Option
    println!("\n=== Niche Optimization for Nested Options ===");
    println!("Option<bool>:                       {} bytes", mem::size_of::<Option<bool>>());
    println!("Option<Option<bool>>:               {} bytes", mem::size_of::<Option<Option<bool>>>());
    // bool is 0 or 1, so 2 can be used as None → 1 byte!

    println!("Option<Option<Option<bool>>>:        {} bytes",
        mem::size_of::<Option<Option<Option<bool>>>>());
    // 0=false, 1=true, 2=Some(None), 3=None → still 1 byte!

    // Nested references
    println!("\nOption<&u8>:                        {} bytes", mem::size_of::<Option<&u8>>());
    println!("Option<Option<&u8>>:                {} bytes", mem::size_of::<Option<Option<&u8>>>());
    // Option<&u8> uses null=None, non-null=Some → 8 bytes
    // Option<Option<&u8>> doesn't have enough niches → 16 bytes
}
```

### Code Example: Techniques to Minimize enum Size

```rust
use std::mem;

// Before improvement: becomes huge to fit the largest variant
enum LargeEnum {
    Small(u32),
    Medium([u8; 64]),
    Large([u8; 1024]),  // ← these 1024 bytes determine the total size
}

// Improvement 1: wrap the large variant in a Box
enum OptimizedEnum1 {
    Small(u32),
    Medium([u8; 64]),
    Large(Box<[u8; 1024]>),  // 8 bytes (pointer)
}

// Improvement 2: extract common data outside
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

// Improvement 3: indirect reference via index
struct Arena {
    texts: Vec<String>,
    binaries: Vec<Vec<u8>>,
}

enum ArenaMessage {
    Text(usize),     // index into texts
    Binary(usize),   // index into binaries
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

## 5. Zero-Sized Types (ZST) and PhantomData

### Code Example 5: Using ZSTs

```rust
use std::marker::PhantomData;
use std::mem;

// Zero-sized types: do not consume memory
struct Meters;
struct Seconds;

// Distinguishing units at the type level (no memory cost)
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

    // Quantity<Meters> and Quantity<Seconds> are different types
    // → unit confusion is prevented at compile time

    println!("Quantity<Meters> size: {}", mem::size_of::<Quantity<Meters>>());
    // 8 bytes (only f64; PhantomData is 0 bytes)

    // Size of Vec<()>
    let units: Vec<()> = vec![(); 1000];
    println!("Vec<()> element is {} bytes", mem::size_of::<()>()); // 0
    // No memory is allocated (only len is tracked)
}
```

### Code Example: Using ZSTs in the Type-State Pattern

```rust
use std::marker::PhantomData;

// ZSTs representing type states
struct Idle;
struct Connected;
struct Authenticated;

// Manage connection state via type parameter
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
        println!("Connecting: {}:{}", self.host, self.port);
        Ok(Connection {
            host: self.host,
            port: self.port,
            _state: PhantomData,
        })
    }
}

impl Connection<Connected> {
    fn authenticate(self, _token: &str) -> Result<Connection<Authenticated>, String> {
        println!("Authenticating...");
        Ok(Connection {
            host: self.host,
            port: self.port,
            _state: PhantomData,
        })
    }

    fn disconnect(self) -> Connection<Idle> {
        println!("Disconnect");
        Connection {
            host: self.host,
            port: self.port,
            _state: PhantomData,
        }
    }
}

impl Connection<Authenticated> {
    fn query(&self, sql: &str) -> Result<String, String> {
        println!("Executing query: {}", sql);
        Ok("result".to_string())
    }

    fn disconnect(self) -> Connection<Idle> {
        println!("Disconnect");
        Connection {
            host: self.host,
            port: self.port,
            _state: PhantomData,
        }
    }
}

fn main() {
    use std::mem;

    // Same size in all states (thanks to ZSTs)
    println!("Connection<Idle>:          {} bytes", mem::size_of::<Connection<Idle>>());
    println!("Connection<Connected>:     {} bytes", mem::size_of::<Connection<Connected>>());
    println!("Connection<Authenticated>: {} bytes", mem::size_of::<Connection<Authenticated>>());

    // Prevent invalid state transitions at compile time
    let conn = Connection::<Idle>::new("localhost", 5432);
    let conn = conn.connect().unwrap();
    // conn.query("SELECT 1"); // compile error! cannot query in Connected state
    let conn = conn.authenticate("secret").unwrap();
    let _result = conn.query("SELECT 1").unwrap(); // OK
}
```

### Code Example: Expressing Ownership and Lifetimes via PhantomData

```rust
use std::marker::PhantomData;
use std::mem;

/// Example expressing ownership semantics with PhantomData
struct Owned<T> {
    ptr: *mut T,
    _owns: PhantomData<T>, // "owns" T → responsible for freeing T on drop
}

struct Borrowed<'a, T> {
    ptr: *const T,
    _borrows: PhantomData<&'a T>, // "borrows" T → lifetime 'a is valid
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

/// Controlling covariance and contravariance with PhantomData
struct Covariant<'a, T: 'a> {
    _phantom: PhantomData<&'a T>, // covariant: OK if T outlives 'a
}

struct Invariant<'a, T: 'a> {
    _phantom: PhantomData<&'a mut T>, // invariant: requires exact match
}

fn main() {
    println!("Owned<u64>:    {} bytes", mem::size_of::<Owned<u64>>());    // 8 (ptr only)
    println!("Borrowed<u64>: {} bytes", mem::size_of::<Borrowed<u64>>()); // 8 (ptr only)

    let owned = Owned::new(42u64);
    println!("value: {}", owned.as_ref());
}
```

---

## 6. Advanced Memory Control

### Code Example: Custom Allocator

```rust
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Allocator that tracks memory usage
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
        println!("=== Memory Report ===");
        println!("  Total allocated:   {} bytes", alloc);
        println!("  Total deallocated: {} bytes", dealloc);
        println!("  Currently in use:  {} bytes", alloc - dealloc);
        println!("  Allocation count:  {} times", count);
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

    // Memory is freed once the scope exits
    ALLOCATOR.report();
}
```

### Code Example: Safe Use of Uninitialized Memory with MaybeUninit

```rust
use std::mem::MaybeUninit;

/// Use MaybeUninit to efficiently initialize an array
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

    // All elements are initialized, so safe to convert
    unsafe {
        // MaybeUninit<T> and T have the same memory layout
        std::mem::transmute(arr)
    }
}

/// Lazy initialization pattern using MaybeUninit
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
            // Initialize only on first access
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

### Code Example: Alignment Control and Cache Optimization

```rust
use std::mem;

/// Cache-line alignment to prevent false sharing
#[repr(align(64))]
struct CacheAligned<T> {
    value: T,
}

/// Counters accessed independently by multiple threads
struct PerThreadCounters {
    // Each counter is placed on a separate cache line
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

/// Alignment for SIMD
#[repr(align(32))]
struct SimdAligned {
    data: [f32; 8], // aligned to 32-byte boundary for AVX2
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
    // Each counter is 64 bytes apart → no false sharing
    println!("PerThreadCounters total size: {} bytes",
        mem::size_of::<PerThreadCounters>());
}
```

---

## 7. Practical Memory Profiling

### Code Example: Measuring Memory Usage Using std::alloc

```rust
use std::alloc::{Layout, alloc, dealloc};

/// Demonstration of manual memory allocation
fn manual_allocation_demo() {
    // Layout: specifies size and alignment
    let layout = Layout::new::<[u8; 256]>();
    println!("Layout: size={}, align={}", layout.size(), layout.align());

    unsafe {
        // Allocate memory
        let ptr = alloc(layout);
        if ptr.is_null() {
            panic!("Memory allocation failed");
        }

        // Initialize
        std::ptr::write_bytes(ptr, 0, 256);

        // Use
        *ptr = 42;
        println!("Allocated value: {}", *ptr);

        // Deallocate
        dealloc(ptr, layout);
    }

    // Use Layout::from_size_align to create a layout dynamically
    let dynamic_layout = Layout::from_size_align(1024, 16).unwrap();
    println!("Dynamic layout: size={}, align={}",
        dynamic_layout.size(), dynamic_layout.align());
}

/// Estimate memory usage of collections
fn estimate_collection_memory() {
    use std::mem;
    use std::collections::HashMap;

    // Memory usage of Vec
    let v: Vec<u64> = (0..1000).collect();
    let stack_size = mem::size_of::<Vec<u64>>();
    let heap_size = v.capacity() * mem::size_of::<u64>();
    println!("Vec<u64> (1000 elements):");
    println!("  Stack: {} bytes", stack_size);
    println!("  Heap:  {} bytes", heap_size);
    println!("  Total: {} bytes", stack_size + heap_size);

    // Memory usage of HashMap (estimate)
    let mut map: HashMap<u64, String> = HashMap::new();
    for i in 0..100 {
        map.insert(i, format!("value_{}", i));
    }
    let map_stack = mem::size_of::<HashMap<u64, String>>();
    // HashMap internals are complex, but compute an estimate
    let avg_value_size = 16; // average heap data size of String
    let entry_overhead = mem::size_of::<u64>() + mem::size_of::<String>() + 8; // key + value + metadata
    let estimated_heap = map.capacity() * entry_overhead + 100 * avg_value_size;
    println!("\nHashMap<u64, String> (100 elements):");
    println!("  Stack:           {} bytes", map_stack);
    println!("  Heap (estimate): {} bytes", estimated_heap);
}

fn main() {
    manual_allocation_demo();
    println!();
    estimate_collection_memory();
}
```

### Code Example: Patterns for Detecting Memory Leaks

```rust
use std::sync::Arc;
use std::cell::RefCell;

/// Example of a memory leak from a circular reference
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

    // Create a circular reference
    *a.next.borrow_mut() = Some(Arc::clone(&b));

    println!("a strong count: {}", Arc::strong_count(&a)); // 2
    println!("b strong count: {}", Arc::strong_count(&b)); // 2
    // → strong count never reaches 0 even after the scope ends, leaking memory!
}

/// Prevent circular references with Weak
fn demonstrate_weak_reference() {
    use std::sync::Weak;

    struct SafeNode {
        value: i32,
        next: RefCell<Option<Arc<SafeNode>>>,
        prev: RefCell<Option<Weak<SafeNode>>>, // weak reference → no cycle
    }

    let a = Arc::new(SafeNode {
        value: 1,
        next: RefCell::new(None),
        prev: RefCell::new(None),
    });

    let b = Arc::new(SafeNode {
        value: 2,
        next: RefCell::new(None),
        prev: RefCell::new(Some(Arc::downgrade(&a))), // Weak reference
    });

    *a.next.borrow_mut() = Some(Arc::clone(&b));

    println!("a strong count: {}", Arc::strong_count(&a)); // 1
    println!("b strong count: {}", Arc::strong_count(&b)); // 2
    println!("a weak count:   {}", Arc::weak_count(&a));   // 1
    // → freed normally
}

fn main() {
    println!("=== Circular Reference (memory leak) ===");
    demonstrate_circular_reference();
    println!("\n=== Weak Reference (leak prevention) ===");
    demonstrate_weak_reference();
}
```

---

## 8. Comparison Tables

### Stack vs Heap

| Characteristic | Stack | Heap |
|---|---|---|
| Allocation speed | Ultra-fast (just SP move) | Slow (allocator call) |
| Deallocation speed | Ultra-fast (automatic) | Slow (free/dealloc) |
| Size constraint | Fixed and small (typically 8MB max) | Dynamic, can be large |
| Lifetime | Tied to scope | Arbitrary (managed by ownership) |
| Cache efficiency | Very high (locality) | Low (scattered) |
| Fragmentation | None | Yes |
| Threads | Independent per thread | Shared across all threads |
| Overflow | Terminates with SIGSEGV | OOM error |

### Comparison of repr Attributes

| repr | Layout | Use Case | Caveats |
|---|---|---|---|
| (default) | Compiler-optimized | Normal Rust code | Field order undefined |
| `repr(C)` | C-compatible | FFI, external libraries | Has padding |
| `repr(transparent)` | Same as inner type | Newtype pattern | Single field only |
| `repr(packed)` | No padding | Binary protocols | Reduced access performance |
| `repr(align(N))` | Minimum alignment N | Cache-line alignment | Increased memory consumption |

### Memory Cost of Smart Pointers

| Type | Stack Size | Heap Overhead | Use Case |
|---|---|---|---|
| `Box<T>` | 8 bytes (ptr) | 0 | Single value ownership |
| `Rc<T>` | 8 bytes (ptr) | 16 bytes (strong+weak) | Single-threaded sharing |
| `Arc<T>` | 8 bytes (ptr) | 16 bytes (atomic strong+weak) | Multi-threaded sharing |
| `Vec<T>` | 24 bytes (ptr+len+cap) | 0 | Dynamic array |
| `String` | 24 bytes (ptr+len+cap) | 0 | UTF-8 string |
| `Cow<'a, T>` | 24 bytes (enum) | Conditional | Lazy copy |

---

## 9. Anti-Patterns

### Anti-Pattern 1: Unnecessary Heap Allocation

```rust
// NG: wrapping small fixed-size data in Box
fn bad() -> Box<(f64, f64)> {
    Box::new((1.0, 2.0)) // allocates 16 bytes on the heap → wasteful
}

// OK: just return as-is (copy/move is sufficient)
fn good() -> (f64, f64) {
    (1.0, 2.0) // copy on the stack
}

// When Box is needed:
// - Recursive types (size unknown at compile time)
// - Trait objects (dyn Trait)
// - Avoiding move cost of large structs
```

### Anti-Pattern 2: Overuse of repr(packed)

```rust
// NG: using packed in performance-sensitive code
#[repr(packed)]
struct BadPerf {
    flag: u8,
    value: u64, // unaligned access → CPU performance penalty
}

// OK: leverage alignment to minimize padding
struct GoodPerf {
    value: u64,  // 8-byte aligned → place at the front
    flag: u8,    // at the end → minimum padding
}
// Use packed only when layout precision matters more than performance,
// such as parsing binary formats.
```

### Anti-Pattern 3: Excessive Clone

```rust
use std::sync::Arc;

// NG: cloning large data every time
fn bad_process(data: &Vec<String>) {
    let cloned = data.clone(); // copies all strings → memory doubles
    process_data(cloned);
}

// OK: pass by reference
fn good_process(data: &[String]) {
    process_data_ref(data); // no copy
}

// OK: use Arc for shared ownership when needed
fn good_shared_process(data: Arc<Vec<String>>) {
    let shared = Arc::clone(&data); // only pointer copy (8 bytes)
    std::thread::spawn(move || {
        process_data_ref(&shared);
    });
}

fn process_data(_data: Vec<String>) {}
fn process_data_ref(_data: &[String]) {}
```

### Anti-Pattern 4: Inefficient String Construction

```rust
fn main() {
    let items = vec!["a", "b", "c", "d", "e"];

    // NG: chaining format! → allocates a new String every time
    let mut result = String::new();
    for item in &items {
        result = format!("{}{}, ", result, item);
        // copies the entire result + item to create a new String each time
        // O(n^2) memory allocations!
    }

    // OK: use push_str / write!
    let mut result = String::with_capacity(items.len() * 4); // pre-reserve
    for (i, item) in items.iter().enumerate() {
        if i > 0 { result.push_str(", "); }
        result.push_str(item);
    }
    // O(n) memory allocations

    // OK: use join (simplest)
    let result = items.join(", ");
    println!("{}", result);
}
```

---

## 10. Practical Patterns

### Pattern 1: Small String Optimization (SSO)

```rust
use std::mem;

/// A type that stores small strings on the stack
/// 23 bytes or less are on the stack; larger are allocated on the heap
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

### Pattern 2: Arena Allocator Pattern

```rust
/// Simple arena allocator
/// Quickly allocates many small objects and frees them all at once
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
        // Align to 8 bytes
        let aligned_size = (size + 7) & !7;

        if self.current.len() + aligned_size > self.current.capacity() {
            // Current chunk is insufficient → create a new chunk
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

    // Quickly allocate 1000 small objects
    for i in 0..1000 {
        let buf = arena.alloc(32);
        buf[0] = (i % 256) as u8;
    }

    println!("Arena allocation: {} bytes", arena.bytes_allocated());

    // Bulk free
    arena.reset();
    println!("After reset: {} bytes", arena.bytes_allocated());
}
```

### Pattern 3: Memory-Efficient Data Structures (SoA vs AoS)

```rust
use std::time::Instant;

// AoS (Array of Structures)
struct ParticleAoS {
    x: f32,
    y: f32,
    z: f32,
    vx: f32,
    vy: f32,
    vz: f32,
    mass: f32,
    _padding: f32, // 32 bytes total, half a cache line
}

// SoA (Structure of Arrays)
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

    /// Update positions (only access x, y, z and vx, vy, vz)
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

    // SoA: mass is not loaded during position update → good cache efficiency
    let mut particles = ParticlesSoA::new(n);
    let start = Instant::now();
    for _ in 0..100 {
        particles.update_positions(0.016);
    }
    let soa_time = start.elapsed();
    println!("SoA position update (100 iters): {:?}", soa_time);

    // AoS: mass + padding are loaded into the cache even during position update → wasteful
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
    println!("AoS position update (100 iters): {:?}", aos_time);
    println!("SoA/AoS ratio: {:.2}x",
        aos_time.as_nanos() as f64 / soa_time.as_nanos() as f64);
}
```

---

## FAQ

### Q1: Why is `Vec<T>` 24 bytes on the stack?

**A:** `ptr` (8 bytes) + `len` (8 bytes) + `capacity` (8 bytes) = 24 bytes. It holds a pointer to the actual data on the heap, the current element count, and the reserved capacity. `String` is internally a `Vec<u8>`, so it has the same structure.

### Q2: Why is `Box<dyn Trait>` 16 bytes?

**A:** It is a fat pointer: a pointer to the data (8 bytes) + a pointer to the vtable (8 bytes). The vtable stores the type's size, drop function, and function pointers for each method.

### Q3: How do I shrink the size of an enum?

**A:** (1) Wrap large variants in `Box`. (2) Leverage niche optimization with `NonZero*` types. (3) Use smaller field types.

```rust
// Before improvement: 104 bytes to fit the largest variant
enum Message {
    Quit,
    Echo(String),       // 24 bytes
    Data([u8; 100]),    // 100 bytes ← this dominates
}

// After improvement: 32 bytes by wrapping in Box
enum MessageOpt {
    Quit,
    Echo(String),
    Data(Box<[u8; 100]>), // 8 bytes (pointer only)
}
```

### Q4: When should I use `Cow<str>`?

**A:** Use `Cow` when input often suffices as a borrow but sometimes a clone is required. For example, in parsing where most input can be returned as-is, but only inputs that need escape processing require copying.

```rust
use std::borrow::Cow;

fn escape_html(input: &str) -> Cow<str> {
    if input.contains(['<', '>', '&', '"', '\'']) {
        // Escape needed → create new String
        Cow::Owned(
            input
                .replace('&', "&amp;")
                .replace('<', "&lt;")
                .replace('>', "&gt;")
                .replace('"', "&quot;")
                .replace('\'', "&#39;")
        )
    } else {
        // No escape needed → borrow the original string
        Cow::Borrowed(input)
    }
}

fn main() {
    let safe = escape_html("Hello, World!");      // Borrowed (no copy)
    let escaped = escape_html("<script>alert(1)</script>"); // Owned (copied)
    println!("{}", safe);
    println!("{}", escaped);
}
```

### Q5: What is the relationship between `Pin<T>` and memory layout?

**A:** `Pin<T>` is a wrapper that guarantees the position in memory is fixed. It is necessary for self-referential structs (types that internally hold pointers to their own fields) to operate safely. Pin is essential for async/await because Futures internally generate self-referential structs.

```rust
use std::pin::Pin;
use std::marker::PhantomPinned;

/// Example of a self-referential struct
struct SelfReferential {
    data: String,
    ptr_to_data: *const String, // pointer to the data field
    _pin: PhantomPinned,        // implements !Unpin → forbids move
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
        // Wrap in Pin to forbid moves
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

### Q6: How do I switch the memory allocator?

**A:** Specify it with the `#[global_allocator]` attribute. Popular high-performance allocators include jemalloc and mimalloc.

```rust
// Example with jemalloc
// Cargo.toml: tikv-jemallocator = "0.6"
#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

// Example with mimalloc
// Cargo.toml: mimalloc = "0.1"
// use mimalloc::MiMalloc;
// #[global_allocator]
// static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    // From here on, all heap allocations use the specified allocator
    let v: Vec<u64> = (0..1_000_000).collect();
    println!("element count: {}", v.len());
}
```

---

## Summary

| Item | Key Point |
|---|---|
| Stack | Fast, fixed size, scope-managed. Ideal for local variables |
| Heap | Dynamic size, long-lived. Managed by Box/Vec/String |
| repr(C) | When you need a C-compatible layout for FFI |
| repr(transparent) | Guarantees newtype and ABI compatibility |
| Alignment | Directly affects CPU memory access efficiency |
| Niche Optimization | Option<Box<T>> is the same size as Box<T> |
| ZST | Add type information with PhantomData (zero memory cost) |
| MaybeUninit | Safe operations on uninitialized memory |
| Arena | Fast allocation and bulk free of many small objects |
| SoA | Data layout focused on cache efficiency |
| Custom Allocators | Performance gains with jemalloc/mimalloc |

## Recommended Next Reading

- [Concurrency](./01-concurrency.md) — memory model and thread safety
- [FFI](./02-ffi-interop.md) — interfacing with foreign languages using repr(C)
- [Embedded/WASM](./03-embedded-wasm.md) — optimization in memory-constrained environments

## References

1. **The Rustonomicon — Data Representation**: https://doc.rust-lang.org/nomicon/data.html
2. **Type Layout (Rust Reference)**: https://doc.rust-lang.org/reference/type-layout.html
3. **Visualizing Rust Memory Layout**: https://www.youtube.com/watch?v=rDoqT-a6UFg
4. **std::alloc Module**: https://doc.rust-lang.org/std/alloc/index.html
5. **Pin and Unpin Explained**: https://doc.rust-lang.org/std/pin/index.html
