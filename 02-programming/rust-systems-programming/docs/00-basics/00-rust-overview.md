# Rust Overview -- A Systems Programming Language That Combines Safety, Performance, and Ownership

> Rust is the only systems programming language that simultaneously achieves "safety," "speed," and "concurrency," guaranteeing memory safety without a garbage collector.

---

## What You Will Learn in This Chapter

1. **Rust's design philosophy** -- Understand the three pillars of zero-cost abstractions, ownership, and type safety
2. **Differences from other languages** -- Grasp Rust's positioning through comparison with C/C++/Go/Python
3. **The big picture of the ecosystem** -- Get a handle on Cargo, crates.io, and the structure of the toolchain
4. **The basic syntax of the language** -- Learn the basics of variables, functions, control flow, and pattern matching
5. **A practical development workflow** -- Experience the entire flow from project creation to testing and documentation generation

## Prerequisite Knowledge

Your understanding will be deeper if you have the following knowledge before reading this guide:

- Basic programming knowledge
- Understanding of related fundamental concepts

---

## 1. The History and Design Philosophy of Rust

### 1.1 The Origins

```
+-----------------------------------------------------------+
|  2006  Graydon Hoare started it as a personal project      |
|  2009  Mozilla officially sponsored it                     |
|  2010  First public announcement                           |
|  2012  Rust 0.1 released (first official pre-release)      |
|  2015  Rust 1.0 stable release                             |
|  2018  Rust 2018 Edition (NLL, async preparation)          |
|  2020  Community continued after Mozilla layoffs           |
|  2021  Rust 2021 Edition / Rust Foundation established     |
|  2022  Linux kernel officially supports Rust               |
|  2024  Rust 2024 Edition                                   |
+-----------------------------------------------------------+
```

Rust was originally a language whose development was started in 2006 as a personal project by Graydon Hoare, an engineer at Mozilla. While working on browser engine development with C++, Hoare felt the need for a language that combines memory safety and performance, and so he began designing Rust.

In 2009, Mozilla became the official sponsor and adopted it as the development language for the Servo browser engine. The Servo project became a place that proved Rust's practicality, and a great deal of insight regarding concurrent processing and memory safety was fed back.

On May 15, 2015, Rust 1.0 was released as a stable version, and the guarantee of backward compatibility began. Since then, continuous improvements have been made on a six-week release cycle.

In 2021, the Rust Foundation was established, with AWS, Google, Huawei, Microsoft, and Mozilla as the five founding members. This secured the long-term sustainability of the language.

### 1.2 Design Principles

```
+---------------------+---------------------+---------------------+
|  Safety             |  Speed              |  Concurrency        |
+---------------------+---------------------+---------------------+
| - Ownership system  | - Zero-cost abstr.  | - Send / Sync       |
| - Borrow checker    | - LLVM backend      | - Data race prev.   |
| - Lifetimes         | - Inline expansion  | - fearless concur.  |
| - Option eliminates | - Monomorphization  | - async/await       |
|   null              | - Stack-first       | - Channel comm.     |
| - Result for errors | - SIMD support      | - Arc/Mutex         |
| - Explicit unsafe   |                     |                     |
|   boundaries        |                     |                     |
+---------------------+---------------------+---------------------+
```

#### Zero-Cost Abstractions

One of Rust's most important design principles is zero-cost abstractions. Rust thoroughly applies the principle that Bjarne Stroustrup advocated in C++:

> "What you don't use, you don't pay for. And further, what you do use, you couldn't hand-code any better."

In Rust, even when you use high-level abstractions such as iterators, generics, and traits, the compiler-optimized result delivers performance equivalent to hand-written low-level code.

```rust
// High-level iterator chain
let sum: i32 = (0..1000)
    .filter(|x| x % 2 == 0)
    .map(|x| x * x)
    .sum();

// The compiler optimizes this into machine code equivalent to a hand-written loop.
// Through LLVM optimization passes, intermediate iterator objects are completely eliminated.
```

#### Type Safety

Rust's type system detects, at compile time, many bugs that in C/C++ can only be discovered at runtime. By using `Option<T>` instead of null pointers and `Result<T, E>` instead of exceptions, error-handling paths are enforced at the type level.

```rust
// Use Option instead of null
fn find_user(id: u64) -> Option<User> {
    // Returning None safely expresses "not found"
    if id == 0 { return None; }
    Some(User { id, name: "example".to_string() })
}

// The caller must always handle the None case
match find_user(42) {
    Some(user) => println!("Found: {}", user.name),
    None => println!("User not found"),
}
```

#### The unsafe Boundary

While Rust guarantees complete safety, it provides an "escape hatch" for performing low-level operations through the `unsafe` keyword. This makes possible OS system calls, FFI (Foreign Function Interface), and performance-specialized optimizations.

```rust
// In an unsafe block, the following operations are permitted:
// 1. Dereferencing raw pointers
// 2. Calling unsafe functions and methods
// 3. Accessing mutable static variables
// 4. Implementing unsafe traits
// 5. Accessing fields of a union

fn raw_pointer_example() {
    let mut num = 5;
    let r1 = &num as *const i32;     // Raw pointer (immutable)
    let r2 = &mut num as *mut i32;   // Raw pointer (mutable)

    unsafe {
        println!("r1 = {}", *r1);
        *r2 = 10;
        println!("r2 = {}", *r2);
    }
}
```

The important point is that `unsafe` only disables "some" of the compiler's safety checks; the rules of ownership and lifetimes themselves still apply.

### 1.3 Problems Rust Solves

Rust was designed primarily to solve the following problems:

```
┌──────────────────────────────────────────────────────────────┐
│ Problem in C/C++          │ Rust's solution                   │
├───────────────────────────┼──────────────────────────────────┤
│ Dangling pointers         │ Prevented by ownership +          │
│                           │ lifetimes                        │
│ Double free               │ Prevented by move semantics       │
│ Buffer overflow           │ Prevented by bounds checking +    │
│                           │ slices                           │
│ Data races                │ Prevented by borrow rules +       │
│                           │ Send/Sync                        │
│ Null pointer dereference  │ Prevented at the type level by    │
│                           │ Option<T>                        │
│ Memory leaks (general)    │ Automatically managed by RAII +   │
│                           │ Drop                             │
│ Use of uninitialized      │ Compiler enforces initialization  │
│ variables                 │                                  │
│ Integer overflow          │ Panics in debug builds            │
└───────────────────────────┴──────────────────────────────────┘
```

---

## 2. Code Examples

### Example 1: Hello, World!

```rust
fn main() {
    println!("Hello, World!");
}
```

`println!` is a macro (the trailing `!` is the marker), and it validates the format string at compile time. Format string mismatches like those in C's `printf` become compilation errors.

### Example 2: Variables and Immutability

```rust
fn main() {
    let x = 5;          // Immutable by default
    // x = 6;           // Compilation error!
    let mut y = 10;     // Declared mutable with mut
    y += 1;
    println!("x={}, y={}", x, y);

    // Shadowing: bind a new variable with the same name
    let x = x + 1;      // New x (the type can also be changed)
    let x = x * 2;
    println!("x after shadowing = {}", x); // 12

    // With shadowing you can change the type
    let spaces = "   ";           // &str type
    let spaces = spaces.len();    // Changed to usize type
    println!("Number of spaces: {}", spaces);
}
```

In Rust, variables are immutable by default. This is a design influenced by functional programming; making immutability the default improves code safety and predictability. When a variable needs to be mutable, the `mut` keyword must be added explicitly.

### Example 3: Basics of Ownership

```rust
fn main() {
    let s1 = String::from("hello");
    let s2 = s1;        // Ownership of s1 is moved to s2
    // println!("{}", s1); // Compilation error: s1 is invalid
    println!("{}", s2);
}
```

### Example 4: Functions and Return Values

```rust
fn add(a: i32, b: i32) -> i32 {
    a + b   // No semicolon = returned as an expression
}

fn main() {
    let result = add(3, 4);
    println!("3 + 4 = {}", result);
}
```

In Rust, when the last expression of a function is written without a semicolon, it is treated as the return value. By convention, the `return` keyword is used only for early returns.

### Example 5: Pattern Matching

```rust
enum Direction {
    North,
    South,
    East,
    West,
}

fn describe(dir: Direction) -> &'static str {
    match dir {
        Direction::North => "North",
        Direction::South => "South",
        Direction::East  => "East",
        Direction::West  => "West",
    }
}

fn main() {
    println!("{}", describe(Direction::North));
}
```

`match` expressions are required to be exhaustive. If not all patterns are covered, a compilation error occurs, which prevents pattern omissions.

### Example 6: Structs and Methods

```rust
struct Point {
    x: f64,
    y: f64,
}

impl Point {
    // Associated function (constructor convention)
    fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    // Method (takes &self)
    fn distance(&self, other: &Point) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }

    // Distance from the origin
    fn magnitude(&self) -> f64 {
        (self.x.powi(2) + self.y.powi(2)).sqrt()
    }

    // Mutable method (takes &mut self)
    fn translate(&mut self, dx: f64, dy: f64) {
        self.x += dx;
        self.y += dy;
    }
}

fn main() {
    let a = Point::new(0.0, 0.0);
    let b = Point::new(3.0, 4.0);
    println!("Distance: {}", a.distance(&b)); // 5.0
    println!("Magnitude of b: {}", b.magnitude()); // 5.0

    let mut c = Point::new(1.0, 1.0);
    c.translate(2.0, 3.0);
    println!("After translation: ({}, {})", c.x, c.y); // (3.0, 4.0)
}
```

### Example 7: Control Flow

```rust
fn main() {
    // if expression (in Rust, if returns a value as an expression)
    let number = 7;
    let kind = if number % 2 == 0 { "even" } else { "odd" };
    println!("{} is {}", number, kind);

    // loop (infinite loop, can return a value with break)
    let mut counter = 0;
    let result = loop {
        counter += 1;
        if counter == 10 {
            break counter * 2; // 20 is bound to result
        }
    };
    println!("Result of loop: {}", result);

    // while loop
    let mut n = 5;
    while n > 0 {
        println!("{}!", n);
        n -= 1;
    }
    println!("Liftoff!");

    // for loop (Range)
    for i in 1..=5 {
        print!("{} ", i); // 1 2 3 4 5
    }
    println!();

    // for loop (collection)
    let fruits = vec!["apple", "orange", "banana"];
    for (i, fruit) in fruits.iter().enumerate() {
        println!("{}: {}", i, fruit);
    }

    // while let (combined with pattern matching)
    let mut stack = vec![1, 2, 3];
    while let Some(top) = stack.pop() {
        println!("Pop: {}", top);
    }
}
```

### Example 8: Basics of Closures

```rust
fn main() {
    // Closure (anonymous function)
    let add = |a: i32, b: i32| -> i32 { a + b };
    println!("3 + 4 = {}", add(3, 4));

    // More concise with type inference
    let multiply = |a, b| a * b;
    println!("3 * 4 = {}", multiply(3, 4));

    // Capturing the environment
    let offset = 10;
    let add_offset = |x| x + offset;
    println!("5 + 10 = {}", add_offset(5));

    // Combined with iterators
    let numbers = vec![1, 2, 3, 4, 5];
    let sum: i32 = numbers.iter()
        .filter(|&&x| x > 2)
        .map(|&x| x * x)
        .sum();
    println!("Sum of squares of numbers greater than 2: {}", sum); // 9 + 16 + 25 = 50
}
```

### Example 9: Basics of Traits

```rust
trait Printable {
    fn format_string(&self) -> String;

    // Default implementation
    fn print(&self) {
        println!("{}", self.format_string());
    }
}

struct Article {
    title: String,
    content: String,
}

impl Printable for Article {
    fn format_string(&self) -> String {
        format!("[Article] {}\n{}", self.title, self.content)
    }
}

struct Tweet {
    user: String,
    message: String,
}

impl Printable for Tweet {
    fn format_string(&self) -> String {
        format!("@{}: {}", self.user, self.message)
    }
}

// Function using trait bounds
fn display_item(item: &impl Printable) {
    item.print();
}

fn main() {
    let article = Article {
        title: "Introduction to Rust".to_string(),
        content: "Rust is a wonderful language.".to_string(),
    };
    let tweet = Tweet {
        user: "rustlang".to_string(),
        message: "Rust 1.0 released!".to_string(),
    };

    display_item(&article);
    display_item(&tweet);
}
```

### Example 10: Basics of Error Handling

```rust
use std::fs;
use std::io;

fn read_username_from_file() -> Result<String, io::Error> {
    let content = fs::read_to_string("username.txt")?;
    Ok(content.trim().to_string())
}

fn main() {
    match read_username_from_file() {
        Ok(name) => println!("Username: {}", name),
        Err(e) => eprintln!("Error: {}", e),
    }

    // Example using Option
    let numbers = vec![10, 20, 30];
    let first = numbers.first();
    match first {
        Some(n) => println!("First element: {}", n),
        None => println!("Empty vector"),
    }

    // Concise pattern matching with if let
    if let Some(n) = numbers.get(1) {
        println!("Second element: {}", n);
    }
}
```

---

## 3. Diagrams

### 3.1 Build Pipeline

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  .rs     │───>│  rustc   │───>│   LLVM   │───>│  Binary  │
│  source  │    │ frontend │    │ IR/optim.│    │  exec.   │
│          │    │          │    │          │    │  file    │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
                     │                │
                     ▼                ▼
              ┌──────────────┐  ┌──────────────┐
              │ Borrow check │  │ Optim. pass  │
              │ Type check   │  │ Inlining     │
              │ Lifetimes    │  │ Dead-code    │
              │ Pattern check│  │ removal,SIMD │
              └──────────────┘  └──────────────┘
```

Details of the build pipeline:

1. **Lexical analysis & parsing**: Convert the `.rs` file into an AST (Abstract Syntax Tree)
2. **Name resolution**: Resolve variable names, type names, and module names
3. **Type checking**: Verify type consistency (including type inference)
4. **Borrow checking**: Verify the rules of ownership, borrowing, and lifetimes
5. **MIR generation**: Generate the intermediate representation (Mid-level IR)
6. **MIR optimization**: Constant propagation, dead-code elimination, etc.
7. **LLVM IR generation**: Convert to the intermediate representation that LLVM understands
8. **LLVM optimization**: Inline expansion, loop optimization, SIMD conversion, etc.
9. **Code generation**: Generate machine code for the target architecture
10. **Linking**: Link object files to generate the executable binary

### 3.2 Memory Model (Stack vs. Heap)

```
       Stack                          Heap
  ┌─────────────────┐         ┌───────────────┐
  │ ptr ──────────────────────>│ h e l l o     │
  │ len = 5         │         └───────────────┘
  │ capacity = 5    │
  ├─────────────────┤         ┌───────────────────┐
  │ ptr ──────────────────────>│ [1, 2, 3, 4, 5]  │
  │ len = 5         │         └───────────────────┘
  │ capacity = 8    │
  ├─────────────────┤
  │ x: i32 = 42     │
  ├─────────────────┤
  │ y: bool = true  │
  ├─────────────────┤
  │ z: f64 = 3.14   │
  └─────────────────┘
   Fixed size, fast        Variable size, dynamic
   LIFO (last-in,          Managed by the OS
   first-out)              (malloc/free)
   Allocated and freed     Rust manages it
   automatically           automatically via RAII
```

How to use the stack and the heap:

- **Stack**: Types whose size is known at compile time (`i32`, `f64`, `bool`, fixed-length arrays, tuples)
- **Heap**: Types whose size changes at runtime (`String`, `Vec<T>`, `HashMap<K,V>`, `Box<T>`)

Rust's `String` type holds three fields on the stack (pointer, length, capacity), and the actual string data is stored on the heap. When the owner goes out of scope, the heap memory is automatically released by the `Drop` trait.

### 3.3 Toolchain Composition

```
┌─────────────────────────────────────────────────────────┐
│                      rustup                             │
│  (Toolchain management: stable/beta/nightly)            │
├──────────┬──────────┬───────────────────────────────────┤
│  rustc   │  cargo   │  Other tools                      │
│ Compiler │ Build    │  rustfmt   -- Code formatter      │
│          │ Package  │  clippy    -- Static-analysis     │
│          │ Test     │              linter               │
│          │ Docs     │  rust-analyzer -- LSP server      │
│          │ Bench    │  miri    -- Undefined-behavior    │
│          │          │            detector               │
│          │          │  cargo-audit -- Vulnerability     │
│          │          │                  scan             │
│          │          │  cargo-expand -- Macro-expansion  │
│          │          │                   inspection      │
└──────────┴──────────┴───────────────────────────────────┘
                │
                ▼
         ┌────────────┐
         │ crates.io  │
         │ Registry   │  Over 150,000 crates published
         └────────────┘
```

### 3.4 Rust's Module System

```
my_project/
├── Cargo.toml          # Project settings & dependencies
├── Cargo.lock          # Dependency lock file
├── src/
│   ├── main.rs         # Entry point of the binary crate
│   ├── lib.rs          # Root of the library crate
│   ├── config.rs       # Module file
│   └── utils/          # Submodule directory
│       ├── mod.rs      # Root of the utils module
│       ├── parser.rs   # utils::parser submodule
│       └── formatter.rs# utils::formatter submodule
├── tests/
│   └── integration_test.rs  # Integration tests
├── benches/
│   └── benchmark.rs    # Benchmarks
└── examples/
    └── demo.rs         # Sample code
```

```rust
// src/lib.rs
pub mod config;       // Loads config.rs
pub mod utils;        // Loads utils/mod.rs

// src/utils/mod.rs
pub mod parser;       // Loads utils/parser.rs
pub mod formatter;    // Loads utils/formatter.rs

// src/main.rs
use my_project::config::Config;
use my_project::utils::parser::parse;

fn main() {
    let config = Config::load("settings.toml").unwrap();
    let data = parse(&config.input_file).unwrap();
    println!("Parsing complete: {} items", data.len());
}
```

### 3.5 Overview Diagram of Rust's Ownership Model

```
┌──────────────────────────────────────────────────────────┐
│                  Ownership System                         │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────┐  Move    ┌────────┐                         │
│  │ Var. A │ ──────>  │ Var. B │  A is invalidated       │
│  └────────┘          └────────┘                         │
│                                                          │
│  ┌────────┐  Borrow  ┌────────┐                         │
│  │ Var. A │ <──────  │ Ref &A │  A remains valid        │
│  └────────┘          └────────┘                         │
│                                                          │
│  ┌────────┐  Clone   ┌────────┐                         │
│  │ Var. A │ ──────── │ Var. B │  An independent copy    │
│  └────────┘          └────────┘                         │
│                                                          │
│  Rules:                                                  │
│  1. Each value has a single owner                        │
│  2. When the owner goes out of scope, Drop is called     │
│  3. Multiple immutable references can coexist            │
│  4. Only one mutable reference at a time                 │
│  5. References must always be valid                      │
└──────────────────────────────────────────────────────────┘
```

---

## 4. Comparison Tables

### 4.1 Rust vs. Other Systems Languages

| Property | Rust | C | C++ | Go | Python |
|------|------|---|-----|----|--------|
| Memory safety | Compile-time guaranteed | Manual management | Manual + RAII | GC | GC |
| Data race prevention | Compile-time guaranteed | None | None | Runtime detection | GIL |
| Zero-cost abstractions | Yes | N/A | Yes | No (GC) | No |
| Package manager | Cargo (standard) | None (CMake, etc.) | None (vcpkg, etc.) | go mod | pip |
| Learning curve | Steep | Moderate | Steep | Gentle | Gentle |
| Compilation speed | Slow-ish | Fast | Slow | Very fast | N/A |
| null | Option type | NULL pointer | nullptr | nil | None |
| Error handling | Result type | Return value/errno | Exceptions | error return value | Exceptions |
| Concurrent processing | async/await, threads | pthread | std::thread | goroutine | asyncio |
| Binary size | Small | Smallest | Largish | Slightly large | N/A |
| Cross-compilation | Easy | Complex | Complex | Very easy | N/A |
| Wasm support | Excellent | Emscripten | Emscripten | TinyGo | Pyodide |

### 4.2 Comparison of Rust Editions

| Feature | 2015 | 2018 | 2021 | 2024 |
|------|------|------|------|------|
| NLL (Non-Lexical Lifetimes) | - | Introduced | Stable | Stable |
| async/await | - | Introduced | Stable | Improved |
| Module paths | Old style | New style | New style | New style |
| Closure capture | Whole | Whole | Partial | Partial |
| dyn Trait | Implicit | Explicit required | Explicit required | Explicit required |
| try blocks | - | - | - | Toward stabilization |
| let-else | - | - | Introduced | Stable |
| impl Trait in type aliases | - | - | - | Toward stabilization |

### 4.3 Application Areas of Rust

| Area | Representative projects | Strengths of Rust |
|------|---------------------|-----------|
| Web backends | Actix Web, Axum, Rocket | High throughput, low memory |
| CLI tools | ripgrep, bat, fd, exa | Fast startup, single binary |
| WebAssembly | wasm-bindgen, Yew, Leptos | Small binaries, fast execution |
| Embedded | embedded-hal, RTIC | Zero cost, no_std support |
| OS/kernel | Redox OS, Linux drivers | Memory safety, low-level control |
| Games | Bevy, Amethyst | High performance, ECS |
| Blockchain | Solana, Polkadot, Near | Safety, performance |
| Databases | TiKV, SurrealDB | Concurrent processing, reliability |
| Networking | Tokio, Hyper, Cloudflare | Asynchronous I/O, low latency |

---

## 5. How to Use Cargo

### 5.1 Basic Commands

```bash
# Project creation
cargo new my_project          # Binary project
cargo new my_lib --lib        # Library project
cargo init                    # Initialize an existing directory

# Build and run
cargo build                   # Debug build
cargo build --release         # Release build (with optimization)
cargo run                     # Build and run
cargo run --release           # Run in release mode
cargo run -- arg1 arg2        # Run with arguments

# Tests
cargo test                    # Run all tests
cargo test test_name          # Run a specific test
cargo test -- --nocapture     # Show println! output
cargo test --doc              # Doc tests only

# Quality tools
cargo clippy                  # Static analysis
cargo fmt                     # Code formatting
cargo doc --open              # Generate and view documentation
cargo audit                   # Security audit (requires installation)

# Dependencies
cargo add serde               # Add a dependency (cargo-edit)
cargo update                  # Update dependencies
cargo tree                    # Display dependency tree

# Others
cargo bench                   # Run benchmarks
cargo clean                   # Remove build artifacts
cargo check                   # Compile check (no binary generation, fast)
```

### 5.2 Structure of Cargo.toml

```toml
[package]
name = "my_project"
version = "0.1.0"
edition = "2021"
authors = ["Your Name <you@example.com>"]
description = "A useful project"
license = "MIT"
repository = "https://github.com/user/my_project"

[dependencies]
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1", features = ["full"] }
anyhow = "1.0"
clap = { version = "4", features = ["derive"] }

[dev-dependencies]
criterion = "0.5"
mockall = "0.12"

[profile.release]
opt-level = 3      # Maximum optimization
lto = true          # Link-time optimization
codegen-units = 1   # Number of code-generation units (slower but more optimized)
strip = true        # Strip debug symbols

[profile.dev]
opt-level = 0       # No optimization (fast build)
debug = true        # With debug information
```

---

## 6. Anti-Patterns

### Anti-Pattern 1: Solving everything with `clone()`

```rust
// BAD: Avoiding all ownership errors with clone
fn process(data: Vec<String>) {
    let copy = data.clone(); // Unnecessary heap allocation
    for item in data.clone() {
        println!("{}", item);
    }
}

// GOOD: Make use of references
fn process_good(data: &[String]) {
    for item in data {
        println!("{}", item);
    }
}
```

### Anti-Pattern 2: Overuse of `unwrap()`

```rust
// BAD: unwrap is dangerous in production code
fn read_config() -> String {
    std::fs::read_to_string("config.toml").unwrap() // Panic!
}

// GOOD: Properly propagate Result
fn read_config_good() -> Result<String, std::io::Error> {
    std::fs::read_to_string("config.toml")
}
```

### Anti-Pattern 3: Unnecessarily owning a `String`

```rust
// BAD: Requiring a String as the argument
fn greet(name: String) {
    println!("Hello, {}!", name);
}

// GOOD: Accepting &str allows passing both String and &str
fn greet_good(name: &str) {
    println!("Hello, {}!", name);
}

fn main() {
    let owned = String::from("Taro");
    let borrowed = "Hanako";

    greet_good(&owned);    // String -> &str (automatic Deref)
    greet_good(borrowed);  // &str passed as is
}
```

### Anti-Pattern 4: Excessively deep nesting

```rust
// BAD: Deeply nested match
fn process_data(input: Option<Result<String, io::Error>>) {
    match input {
        Some(result) => {
            match result {
                Ok(data) => {
                    match data.parse::<i32>() {
                        Ok(num) => println!("{}", num),
                        Err(e) => eprintln!("Parse error: {}", e),
                    }
                }
                Err(e) => eprintln!("IO error: {}", e),
            }
        }
        None => eprintln!("No input"),
    }
}

// GOOD: Flatten with early returns and combinators
fn process_data_good(input: Option<Result<String, io::Error>>) -> Result<(), Box<dyn std::error::Error>> {
    let data = input.ok_or("No input")?;
    let text = data?;
    let num: i32 = text.parse()?;
    println!("{}", num);
    Ok(())
}
```

### Anti-Pattern 5: C-style for loops

```rust
// BAD: Index-based loop
fn sum_vec(v: &[i32]) -> i32 {
    let mut sum = 0;
    for i in 0..v.len() {
        sum += v[i]; // Bounds check runs every time
    }
    sum
}

// GOOD: Use an iterator
fn sum_vec_good(v: &[i32]) -> i32 {
    v.iter().sum()
}
```

---

## 7. Testing in Rust

### 7.1 Unit Tests

```rust
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

pub fn divide(a: f64, b: f64) -> Result<f64, String> {
    if b == 0.0 {
        Err("Division by zero error".to_string())
    } else {
        Ok(a / b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        assert_eq!(add(2, 3), 5);
        assert_eq!(add(-1, 1), 0);
        assert_eq!(add(0, 0), 0);
    }

    #[test]
    fn test_divide_success() {
        let result = divide(10.0, 2.0);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 5.0);
    }

    #[test]
    fn test_divide_by_zero() {
        let result = divide(10.0, 0.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Division by zero error");
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_out_of_bounds() {
        let v = vec![1, 2, 3];
        let _ = v[10]; // Panic!
    }

    #[test]
    fn test_with_result() -> Result<(), String> {
        let result = divide(10.0, 2.0)?;
        assert_eq!(result, 5.0);
        Ok(())
    }
}
```

### 7.2 Documentation Tests

```rust
/// Adds two numbers.
///
/// # Arguments
///
/// * `a` - The first number
/// * `b` - The second number
///
/// # Returns
///
/// The sum of the two numbers
///
/// # Examples
///
/// ```
/// use my_crate::add;
/// assert_eq!(add(2, 3), 5);
/// assert_eq!(add(-1, 1), 0);
/// ```
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}
```

With `cargo test --doc`, the code examples inside the documentation are run as tests. This guarantees that the code examples in the documentation are always correct.


---

## Hands-on Exercises

### Exercise 1: Basic Implementation

Implement code that satisfies the following requirements.

**Requirements:**
- Validate the input data
- Implement appropriate error handling
- Also create test code

```python
# Exercise 1: Template for basic implementation
class Exercise1:
    """Practice with basic implementation patterns"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """Validate the input value"""
        if value is None:
            raise ValueError("Input value is None")
        return True

    def process(self, value):
        """Main logic for data processing"""
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
        assert False, "An exception should have occurred"
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
    """Practice with advanced patterns"""

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
    assert ex.add("d", 4) == False  # Size limit
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

    print(f"Inefficient version: {slow_time:.4f} sec")
    print(f"Efficient version:   {fast_time:.6f} sec")
    print(f"Speedup factor: {slow_time/fast_time:.0f}x")

benchmark()
```

**Key Points:**
- Be conscious of algorithmic complexity
- Choose appropriate data structures
- Measure the effect with benchmarks

---

## Troubleshooting

### Common Errors and Solutions

| Error | Cause | Solution |
|--------|------|--------|
| Initialization error | Defective configuration file | Check the path and format of the configuration file |
| Timeout | Network latency / insufficient resources | Adjust timeout values, add retry logic |
| Out of memory | Increase in data volume | Introduce batch processing, implement pagination |
| Permission error | Insufficient access rights | Check the executing user's permissions, review settings |
| Data inconsistency | Concurrency conflicts | Introduce locking mechanisms, manage transactions |

### Debugging Procedure

1. **Check the error message**: Read the stack trace and identify the location of occurrence
2. **Establish a reproduction procedure**: Reproduce the error with minimal code
3. **Formulate a hypothesis**: List the conceivable causes
4. **Step-by-step verification**: Use logging output and debuggers to verify the hypothesis
5. **Fix and regression testing**: After fixing, also run tests for related parts

```python
# Debugging utilities
import logging
import traceback
from functools import wraps

# Configure the logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def debug_decorator(func):
    """Decorator that logs function input and output"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Call: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Return value: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"Exception occurred: {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

@debug_decorator
def process_data(items):
    """Data processing (target of debugging)"""
    if not items:
        raise ValueError("Empty data")
    return [item * 2 for item in items]
```

### Diagnosing Performance Problems

Diagnostic procedure when a performance problem occurs:

1. **Identify the bottleneck**: Measure with profiling tools
2. **Check memory usage**: Check whether there is a memory leak
3. **Check I/O wait**: Check the status of disk and network I/O
4. **Check the number of concurrent connections**: Check the state of the connection pool

| Type of problem | Diagnostic tool | Countermeasure |
|-----------|-----------|------|
| CPU load | cProfile, py-spy | Algorithm improvement, parallelization |
| Memory leak | tracemalloc, objgraph | Properly release references |
| I/O bottleneck | strace, iostat | Asynchronous I/O, caching |
| DB latency | EXPLAIN, slow query log | Indexes, query optimization |

---

## Design Decision Guide

### Selection Criteria Matrix

The criteria for making technical choices are summarized below.

| Decision criterion | When emphasized | When can be compromised |
|---------|------------|-------------|
| Performance | Real-time processing, large-scale data | Admin screens, batch processing |
| Maintainability | Long-term operation, team development | Prototypes, short-term projects |
| Scalability | Services expected to grow | Internal tools, fixed user base |
| Security | Personal information, financial data | Public data, internal use |
| Development speed | MVP, time-to-market | Quality-focused, mission-critical |

### Choosing an Architecture Pattern

```
┌─────────────────────────────────────────────────┐
│            Architecture Selection Flow           │
├─────────────────────────────────────────────────┤
│                                                 │
│  (1) What is the team size?                      │
│    ├─ Small (1-5 people) -> Monolith              │
│    └─ Large (10+ people) -> go to (2)             │
│                                                 │
│  (2) What is the deployment frequency?            │
│    ├─ Once a week or less -> Monolith +          │
│    │                          module split        │
│    └─ Daily / multiple times -> go to (3)        │
│                                                 │
│  (3) How independent are the teams?               │
│    ├─ High -> Microservices                       │
│    └─ Moderate -> Modular monolith                │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Trade-off Analysis

Technical decisions always involve trade-offs. Conduct your analysis from the following perspectives:

**1. Short-term vs. long-term cost**
- A method that is fast in the short term can become technical debt in the long term
- Conversely, over-engineering has a high short-term cost and can cause project delays

**2. Consistency vs. flexibility**
- A unified technology stack has a low learning cost
- Adopting diverse technologies allows for the right tool for the right job, but operating costs increase

**3. Level of abstraction**
- High abstraction has high reusability but can make debugging difficult
- Low abstraction is intuitive but tends to cause code duplication

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
        """Describe the background and the issue"""
        self.context = context
        return self

    def set_decision(self, decision: str):
        """Describe the decided content"""
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
        md += f"## Background\n{self.context}\n\n"
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

## Application Scenarios in Practice

### Scenario 1: MVP Development at a Startup

**Situation:** Need to release a product quickly with limited resources

**Approach:**
- Choose a simple architecture
- Focus on the bare-minimum features
- Automated tests only on critical paths
- Introduce monitoring early

**Lessons learned:**
- Don't seek perfection too much (YAGNI principle)
- Get user feedback early
- Manage technical debt consciously

### Scenario 2: Modernization of a Legacy System

**Situation:** Gradually renovate a system that has been in operation for over 10 years

**Approach:**
- Migrate gradually with the Strangler Fig pattern
- If there are no existing tests, first create Characterization Tests
- Coexist old and new systems with an API gateway
- Carry out data migration in stages

| Phase | Work content | Estimated duration | Risk |
|---------|---------|---------|--------|
| 1. Investigation | Current-state analysis, mapping dependencies | 2-4 weeks | Low |
| 2. Foundation | Build CI/CD, set up test environment | 4-6 weeks | Low |
| 3. Migration start | Migrate peripheral features sequentially | 3-6 months | Medium |
| 4. Core migration | Migrate core features | 6-12 months | High |
| 5. Completion | Decommission the old system | 2-4 weeks | Medium |

### Scenario 3: Development with a Large Team

**Situation:** 50 or more engineers develop the same product

**Approach:**
- Clarify boundaries with Domain-Driven Design
- Set ownership per team
- Manage shared libraries via the Inner Source approach
- Design API-first to minimize inter-team dependencies

```python
# Defining API contracts between teams
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class APIContract:
    """API contract between teams"""
    endpoint: str
    method: str
    owner_team: str
    consumers: List[str]
    sla_ms: int  # Response-time SLA
    priority: Priority

    def validate_sla(self, actual_ms: int) -> bool:
        """Check SLA compliance"""
        return actual_ms <= self.sla_ms

    def to_openapi(self) -> dict:
        """Output in OpenAPI format"""
        return {
            'path': self.endpoint,
            'method': self.method,
            'x-owner': self.owner_team,
            'x-consumers': self.consumers,
            'x-sla-ms': self.sla_ms
        }

# Usage example
contracts = [
    APIContract(
        endpoint="/api/v1/users",
        method="GET",
        owner_team="user-team",
        consumers=["order-team", "notification-team"],
        sla_ms=200,
        priority=Priority.HIGH
    ),
    APIContract(
        endpoint="/api/v1/orders",
        method="POST",
        owner_team="order-team",
        consumers=["payment-team", "inventory-team"],
        sla_ms=500,
        priority=Priority.CRITICAL
    )
]
```

### Scenario 4: A Performance-Critical System

**Situation:** A system that requires millisecond-level responses

**Optimization points:**
1. Caching strategy (L1: in-memory, L2: Redis, L3: CDN)
2. Use of asynchronous processing
3. Connection pooling
4. Query optimization and index design

| Optimization technique | Effect | Implementation cost | Application context |
|-----------|------|-----------|---------|
| In-memory cache | High | Low | Frequently accessed data |
| CDN | High | Low | Static content |
| Asynchronous processing | Medium | Medium | Operations with much I/O wait |
| DB optimization | High | High | When queries are slow |
| Code optimization | Low to medium | High | When CPU-bound |
---

## 8. FAQ

### Q1: Is the learning curve of Rust really steep?

**A:** Yes, especially the concepts of ownership and lifetimes are unique to Rust and not found in other languages. However, once mastered, the compiler prevents many bugs in advance, which significantly reduces debugging time. Many developers report that "productivity starts to rise after 3-6 months."

Tips for learning:
- First, read through The Rust Programming Language (the official Book)
- Practice with the Rustlings exercises
- Carefully read the compiler's error messages (Rust's error messages are very kind)
- It's fine to use `clone()` a lot at first. Write working code, then optimize

### Q2: For what kinds of projects is Rust suited?

**A:** Rust is particularly well-suited in cases such as the following:
- Performance is paramount (game engines, browser engines)
- Memory safety is essential (OS, embedded, security tools)
- WebAssembly (high affinity with Wasm)
- CLI tools (single-binary distribution is easy)
- High-throughput network services (proxies, load balancers)

Conversely, in the following cases, Go may be more suitable:
- Quick prototyping is needed
- You want to minimize the team's learning cost
- Mass-producing microservices

### Q3: Why is Rust safe even though it has no garbage collector?

**A:** Rust manages memory with the "ownership system." Each value has a single "owner," and when the owner goes out of scope, memory is automatically released (via the Drop trait). Because the compiler statically verifies the borrow rules, dangling pointers and double frees do not occur.

### Q4: Why is Rust's compilation slow?

**A:** The main reasons Rust's compilation is slow are:
1. **Borrow checking**: Verifying ownership and lifetimes is computationally expensive
2. **Monomorphization**: When generics are used, code is generated for each concrete type
3. **LLVM optimization**: High-quality optimization takes time
4. **Link-time optimization (LTO)**: When enabled in release builds, it becomes especially slow

Improvement measures include using `cargo check` (which skips binary generation), making use of incremental compilation, and introducing `sccache`.

### Q5: In a single phrase, what is the difference between Rust and C++?

**A:** "C++ lets you write both correct and dangerous code. Rust has the compiler reject the dangerous code."

C++ gives the programmer full freedom, but as a result memory-safety bugs can occur. Because Rust's compiler guarantees safety, it can prevent at compile time about 70% of the memory-safety bugs that have long been a problem in C++ (according to a Microsoft Research study).

### Q6: Can you do Web development with Rust?

**A:** Yes. For backends, there are mature frameworks such as Actix Web, Axum, and Rocket. For frontends, there are frameworks that leverage WebAssembly such as Yew, Leptos, and Dioxus. Furthermore, by using Tauri you can also develop desktop applications.

```rust
// A simple web server using Axum
use axum::{routing::get, Router};

async fn hello() -> &'static str {
    "Hello, World!"
}

#[tokio::main]
async fn main() {
    let app = Router::new().route("/", get(hello));
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000")
        .await
        .unwrap();
    axum::serve(listener, app).await.unwrap();
}
```

---

## 9. Practical Development Tips

### 9.1 Getting Along with the Compiler

Rust's compiler (rustc) is very strict, but its error messages are exceedingly helpful. If you carefully read the lines containing "help:" and "note:" in error messages, in most cases solutions are presented.

```
error[E0382]: borrow of moved value: `s1`
 --> src/main.rs:4:20
  |
2 |     let s1 = String::from("hello");
  |         -- move occurs because `s1` has type `String`
3 |     let s2 = s1;
  |              -- value moved here
4 |     println!("{}", s1);
  |                    ^^ value borrowed here after move
  |
  = note: this error originates in the macro `$crate::format_args_nl`
help: consider cloning the value if the performance cost is acceptable
  |
3 |     let s2 = s1.clone();
  |                ++++++++
```

### 9.2 Recommended Development Tools

| Tool | Use | Installation |
|--------|------|-------------|
| rust-analyzer | IDE support (LSP) | VS Code extension, etc. |
| clippy | Linter | `rustup component add clippy` |
| rustfmt | Formatter | `rustup component add rustfmt` |
| cargo-watch | Auto-build on file change | `cargo install cargo-watch` |
| cargo-expand | Inspect macro expansion | `cargo install cargo-expand` |
| cargo-audit | Security vulnerability scanning | `cargo install cargo-audit` |
| cargo-flamegraph | Performance profiling | `cargo install flamegraph` |
| bacon | Lightweight build watcher | `cargo install bacon` |

### 9.3 Frequently Used Design Patterns

```rust
// Builder pattern
struct ServerConfig {
    host: String,
    port: u16,
    max_connections: u32,
    timeout_ms: u64,
}

struct ServerConfigBuilder {
    host: String,
    port: u16,
    max_connections: u32,
    timeout_ms: u64,
}

impl ServerConfigBuilder {
    fn new() -> Self {
        Self {
            host: "localhost".to_string(),
            port: 8080,
            max_connections: 100,
            timeout_ms: 30000,
        }
    }

    fn host(mut self, host: &str) -> Self {
        self.host = host.to_string();
        self
    }

    fn port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    fn max_connections(mut self, max: u32) -> Self {
        self.max_connections = max;
        self
    }

    fn timeout_ms(mut self, timeout: u64) -> Self {
        self.timeout_ms = timeout;
        self
    }

    fn build(self) -> ServerConfig {
        ServerConfig {
            host: self.host,
            port: self.port,
            max_connections: self.max_connections,
            timeout_ms: self.timeout_ms,
        }
    }
}

fn main() {
    let config = ServerConfigBuilder::new()
        .host("0.0.0.0")
        .port(3000)
        .max_connections(1000)
        .build();

    println!("Server: {}:{}", config.host, config.port);
}
```

```rust
// Newtype pattern
struct Email(String);
struct UserId(u64);

impl Email {
    fn new(email: &str) -> Result<Self, String> {
        if email.contains('@') {
            Ok(Email(email.to_string()))
        } else {
            Err("Invalid email address".to_string())
        }
    }

    fn as_str(&self) -> &str {
        &self.0
    }
}

fn send_email(to: &Email, subject: &str) {
    println!("To: {}, Subject: {}", to.as_str(), subject);
}

fn main() {
    let email = Email::new("user@example.com").unwrap();
    send_email(&email, "Test");
    // send_email(&"invalid", "Test"); // Compilation error! Wrong type
}
```

---


## FAQ

### Q1: What is the most important point in learning this topic?

Gaining hands-on experience is the most important thing. Beyond just theory, your understanding deepens by actually writing code and verifying its behavior.

### Q2: What mistakes do beginners commonly fall into?

Skipping the basics and moving on to advanced topics. We recommend that you first thoroughly understand the basic concepts explained in this guide, and then move on to the next step.

### Q3: How is it used in practice?

The knowledge of this topic is frequently used in everyday development work. It is especially important during code reviews and architecture design.

---

## 10. Summary

| Item | Key point |
|------|------|
| Design philosophy | Achieves safety, speed, and concurrency simultaneously |
| Memory management | Ownership system (no GC) |
| Type system | Strong static typing + generics + traits |
| Toolchain | rustup / cargo / clippy / rustfmt |
| Ecosystem | Over 150,000 crates on crates.io |
| Use cases | Systems, web, CLI, Wasm, embedded |
| Editions | Evolves gradually while preserving backward compatibility |
| Testing | Unit tests, integration tests, and doc tests are standard equipment |
| Error handling | Type-safe error handling via Result/Option |
| Concurrent processing | Send/Sync traits prevent data races at compile time |

---

## Recommended Next Guides

- [01-ownership-borrowing.md](01-ownership-borrowing.md) -- Understand ownership and borrowing deeply
- [02-types-and-traits.md](02-types-and-traits.md) -- Learn abstraction with types and traits
- [03-error-handling.md](03-error-handling.md) -- Error handling using Result/Option
- [04-collections-iterators.md](04-collections-iterators.md) -- Collections and iterators
- [../04-ecosystem/00-cargo-workspace.md](../04-ecosystem/00-cargo-workspace.md) -- Master how to use Cargo

---

## References

1. **The Rust Programming Language (the official Book)** -- https://doc.rust-lang.org/book/
2. **Rust by Example** -- https://doc.rust-lang.org/rust-by-example/
3. **Rust Reference** -- https://doc.rust-lang.org/reference/
4. **Rustlings (exercises)** -- https://github.com/rust-lang/rustlings
5. **Rust Foundation** -- https://foundation.rust-lang.org/
6. **Rust API Guidelines** -- https://rust-lang.github.io/api-guidelines/
7. **The Rustonomicon (unsafe Rust)** -- https://doc.rust-lang.org/nomicon/
8. **Rust Playground** -- https://play.rust-lang.org/
9. **This Week in Rust** -- https://this-week-in-rust.org/
10. **Rust Design Patterns** -- https://rust-unofficial.github.io/patterns/
