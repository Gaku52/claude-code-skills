# Systems Language Comparison (C, C++, Rust, Go, Zig)

> Systems languages provide "close-to-hardware control" and "high performance." They form the foundation of OSes, drivers, game engines, and infrastructure tools.

## Learning Objectives

- [ ] Understand the characteristics and application domains of major systems languages
- [ ] Understand the differences in memory management strategies
- [ ] Be able to judge the trade-offs between safety and performance
- [ ] Understand each language's build system and toolchain
- [ ] Be able to select the appropriate language based on project requirements
- [ ] Be able to compare error handling strategies across languages


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding of the content in [Scripting Language Comparison (Python, Ruby, JavaScript, PHP, Perl)](./00-scripting-languages.md)

---

## 1. Comparison Table

```
┌──────────────┬────────┬────────┬────────┬────────┬────────┐
│              │ C      │ C++    │ Rust   │ Go     │ Zig    │
├──────────────┼────────┼────────┼────────┼────────┼────────┤
│ Year Created │ 1972   │ 1985   │ 2015   │ 2012   │ 2016   │
├──────────────┼────────┼────────┼────────┼────────┼────────┤
│ Designer     │ D.Ritchie│Stroustrup│ Hoare+│ Pike+  │ A.Kelley│
│              │        │        │ Mozilla│ Google │        │
├──────────────┼────────┼────────┼────────┼────────┼────────┤
│ Memory       │ Manual │ Manual │ Owner- │ GC     │ Manual │
│ Management   │ malloc │ RAII   │ ship   │ Concur.│ alloc  │
│              │ free   │ smart  │ Borrow │ GC     │ comptime│
│              │        │ ptr    │ checker│ Low lat│        │
├──────────────┼────────┼────────┼────────┼────────┼────────┤
│ Safety       │ Low    │ Medium │ High   │ High   │ Medium │
│              │ Many UB│ Has UB │ No UB  │ Memory │ Has UB │
│              │        │        │ (safe) │ safe   │        │
├──────────────┼────────┼────────┼────────┼────────┼────────┤
│ Performance  │ Fastest│ Fastest│ Fastest│ Fast   │ Fastest│
│              │        │        │        │ GC     │        │
│              │        │        │        │ pause  │        │
├──────────────┼────────┼────────┼────────┼────────┼────────┤
│ Compile Speed│ Fast   │ Slow   │ Slow   │ Very   │ Fast   │
│              │        │ Headers│ Borrow │ Fast   │ Increm.│
│              │        │        │ check  │        │        │
├──────────────┼────────┼────────┼────────┼────────┼────────┤
│ Learning     │ Medium │ High   │ High   │ Low    │ Medium │
│ Curve        │        │ Massive│ Borrow │ 25 KW  │        │
│              │        │ spec   │        │        │        │
├──────────────┼────────┼────────┼────────┼────────┼────────┤
│ Abstraction  │ Minimal│ Rich   │ Rich   │ Minimal│ Minimal│
│ Level        │ Funcs  │ Templat│ Traits │ Inter- │ comptime│
│              │        │ OOP    │ Generic│ faces  │        │
├──────────────┼────────┼────────┼────────┼────────┼────────┤
│ Error        │ Return │ Except.│ Result │ error  │ error  │
│ Handling     │ errno  │ RAII   │ Option │ multi  │ union  │
├──────────────┼────────┼────────┼────────┼────────┼────────┤
│ Concurrency  │ pthread│ thread │ Send/  │ gorout.│ async  │
│              │ fork   │ async  │ Sync   │ channel│ evented│
├──────────────┼────────┼────────┼────────┼────────┼────────┤
│ Primary      │ OS     │ Games  │ Infra  │ Cloud  │ Embedded│
│ Use Cases    │ Embed. │ Browser│ CLI    │ Tools  │ Systems│
│              │ Kernel │ DB     │ Wasm   │ Micro- │ Games  │
│              │        │        │        │ service│        │
├──────────────┼────────┼────────┼────────┼────────┼────────┤
│ Standard     │ Minimal│ Large  │ Medium │ Rich   │ Minimal│
│ Library      │ libc   │ STL    │ std    │ net etc│ std    │
├──────────────┼────────┼────────┼────────┼────────┼────────┤
│ Build System │ Make   │ CMake  │ Cargo  │ go     │ zig    │
│              │ Meson  │ Bazel  │        │ build  │ build  │
└──────────────┴────────┴────────┴────────┴────────┴────────┘
```

---

## 2. Detailed Comparison of Memory Management Models

### 2.1 C — Manual Memory Management

```c
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// C: Manual memory management — managed with malloc/free pairs
typedef struct {
    char* name;
    int age;
    char** tags;
    int tag_count;
} User;

User* user_create(const char* name, int age) {
    User* user = (User*)malloc(sizeof(User));
    if (!user) return NULL;  // Memory allocation failure

    user->name = strdup(name);  // Allocate a copy of the string
    if (!user->name) {
        free(user);
        return NULL;
    }

    user->age = age;
    user->tags = NULL;
    user->tag_count = 0;
    return user;
}

int user_add_tag(User* user, const char* tag) {
    // Expand the array with realloc
    char** new_tags = (char**)realloc(
        user->tags,
        sizeof(char*) * (user->tag_count + 1)
    );
    if (!new_tags) return -1;  // Memory allocation failure

    user->tags = new_tags;
    user->tags[user->tag_count] = strdup(tag);
    if (!user->tags[user->tag_count]) return -1;

    user->tag_count++;
    return 0;
}

void user_destroy(User* user) {
    if (!user) return;

    free(user->name);
    for (int i = 0; i < user->tag_count; i++) {
        free(user->tags[i]);
    }
    free(user->tags);
    free(user);
}

// Usage example
void example(void) {
    User* alice = user_create("Alice", 30);
    if (!alice) {
        fprintf(stderr, "Memory allocation failed\n");
        return;
    }

    user_add_tag(alice, "admin");
    user_add_tag(alice, "developer");

    printf("Name: %s, Age: %d\n", alice->name, alice->age);

    user_destroy(alice);  // Must free (forgetting causes memory leak)
    // alice = NULL;      // Prevent dangling pointer (recommended)
}

// Typical C bug: Use After Free
void dangerous_example(void) {
    char* name = strdup("Alice");
    free(name);
    // printf("%s\n", name);  // Undefined behavior! Accessing freed memory
}

// Typical C bug: Buffer Overflow
void buffer_overflow(void) {
    char buf[10];
    // strcpy(buf, "This is a very long string");  // Dangerous!
    strncpy(buf, "This is a very long string", sizeof(buf) - 1);  // Safe
    buf[sizeof(buf) - 1] = '\0';
}

// Typical C bug: Double Free
void double_free_example(void) {
    char* ptr = malloc(100);
    free(ptr);
    // free(ptr);  // Undefined behavior! Double free
}
```

### 2.2 C++ — RAII and Smart Pointers

```cpp
#include <string>
#include <memory>
#include <vector>
#include <iostream>
#include <optional>

// C++: RAII (Resource Acquisition Is Initialization)
// Acquire in constructor, release in destructor

class User {
public:
    User(std::string name, int age)
        : name_(std::move(name)), age_(age) {}

    // Destructor — automatically called when going out of scope
    ~User() {
        std::cout << "User " << name_ << " destroyed" << std::endl;
    }

    void add_tag(std::string tag) {
        tags_.push_back(std::move(tag));
    }

    const std::string& name() const { return name_; }
    int age() const { return age_; }
    const std::vector<std::string>& tags() const { return tags_; }

private:
    std::string name_;      // std::string manages internal memory
    int age_;
    std::vector<std::string> tags_;  // vector manages array memory
};

// Smart pointer usage patterns
void smart_pointer_example() {
    // unique_ptr: Exclusive ownership (most common)
    auto alice = std::make_unique<User>("Alice", 30);
    alice->add_tag("admin");

    // Ownership transfer (move)
    auto owner = std::move(alice);
    // alice can no longer be used (nullptr)

    // shared_ptr: Shared ownership (reference counting)
    auto bob = std::make_shared<User>("Bob", 25);
    {
        auto bob_ref = bob;  // Reference count +1
        std::cout << "ref count: " << bob.use_count() << std::endl;  // 2
    }
    // bob_ref goes out of scope, reference count -1
    std::cout << "ref count: " << bob.use_count() << std::endl;  // 1

    // weak_ptr: Prevents circular references
    std::weak_ptr<User> weak = bob;
    if (auto locked = weak.lock()) {
        std::cout << "User still alive: " << locked->name() << std::endl;
    }
}

// Move semantics (C++11 and later)
class LargeBuffer {
    std::vector<uint8_t> data_;

public:
    explicit LargeBuffer(size_t size) : data_(size, 0) {}

    // Move constructor — transfers ownership of data (no copy)
    LargeBuffer(LargeBuffer&& other) noexcept
        : data_(std::move(other.data_)) {}

    // Move assignment operator
    LargeBuffer& operator=(LargeBuffer&& other) noexcept {
        data_ = std::move(other.data_);
        return *this;
    }

    // Disable copy (prevent unintended copying of large data)
    LargeBuffer(const LargeBuffer&) = delete;
    LargeBuffer& operator=(const LargeBuffer&) = delete;

    size_t size() const { return data_.size(); }
};

// Type-safe null handling with std::optional
std::optional<User> find_user(const std::string& name) {
    if (name == "Alice") {
        return User("Alice", 30);
    }
    return std::nullopt;
}

void optional_example() {
    auto user = find_user("Alice");
    if (user.has_value()) {
        std::cout << user->name() << std::endl;
    }

    // Default value with value_or
    auto name = find_user("Bob")
        .transform( { return u.name(); })
        .value_or("Unknown");
}

// Concepts (C++20) — Type constraints for templates
template<typename T>
concept Printable = requires(T t) {
    { std::cout << t } -> std::same_as<std::ostream&>;
};

template<Printable T>
void print(const T& value) {
    std::cout << value << std::endl;
}

// Ranges (C++20) — Functional-style data processing
#include <ranges>
#include <algorithm>

void ranges_example() {
    std::vector<int> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    auto result = numbers
        | std::views::filter( { return n % 2 == 0; })
        | std::views::transform( { return n * n; })
        | std::views::take(3);

    for (int n : result) {
        std::cout << n << " ";  // 4 16 36
    }
}
```

### 2.3 Rust — Ownership and the Borrow Checker

```rust
use std::collections::HashMap;

// Rust: Ownership system — guarantees memory safety at compile time
// Three rules:
// 1. Each value has one owner
// 2. When the owner goes out of scope, the value is dropped
// 3. &T (immutable borrow) can have multiple, &mut T (mutable borrow) only one

struct User {
    name: String,
    age: u32,
    tags: Vec<String>,
}

impl User {
    fn new(name: impl Into<String>, age: u32) -> Self {
        User {
            name: name.into(),
            age,
            tags: Vec::new(),
        }
    }

    fn add_tag(&mut self, tag: impl Into<String>) {
        self.tags.push(tag.into());
    }

    // &self: Immutable borrow (read-only)
    fn display(&self) -> String {
        format!("{} (age: {}, tags: {:?})", self.name, self.age, self.tags)
    }

    // self: Consumes ownership (cannot be used after calling)
    fn into_name(self) -> String {
        self.name  // Ownership is moved
    }
}

fn ownership_example() {
    let mut alice = User::new("Alice", 30);
    alice.add_tag("admin");
    alice.add_tag("developer");

    // Immutable borrows (multiple allowed simultaneously)
    let display1 = alice.display();
    let display2 = alice.display();
    println!("{}", display1);
    println!("{}", display2);

    // Ownership transfer
    let name = alice.into_name();
    println!("Name: {}", name);
    // println!("{}", alice.display());  // Compile error! alice can no longer be used
}

// Lifetimes — communicate reference validity to the compiler
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}

struct Config<'a> {
    name: &'a str,
    values: Vec<&'a str>,
}

impl<'a> Config<'a> {
    fn new(name: &'a str) -> Self {
        Config { name, values: Vec::new() }
    }

    fn add_value(&mut self, value: &'a str) {
        self.values.push(value);
    }
}

// Error handling with Result/Option
use std::fs;
use std::io;

fn read_config(path: &str) -> Result<HashMap<String, String>, io::Error> {
    let content = fs::read_to_string(path)?;  // ? operator propagates errors
    let mut config = HashMap::new();

    for line in content.lines() {
        if let Some((key, value)) = line.split_once('=') {
            config.insert(key.trim().to_string(), value.trim().to_string());
        }
    }

    Ok(config)
}

// Pattern matching and enums (algebraic data types)
#[derive(Debug)]
enum Shape {
    Circle { radius: f64 },
    Rectangle { width: f64, height: f64 },
    Triangle { base: f64, height: f64 },
}

impl Shape {
    fn area(&self) -> f64 {
        match self {
            Shape::Circle { radius } => std::f64::consts::PI * radius * radius,
            Shape::Rectangle { width, height } => width * height,
            Shape::Triangle { base, height } => base * height / 2.0,
        }
    }
}

// Traits — the foundation of interfaces and generics
trait Summary {
    fn summarize(&self) -> String;

    // Default implementation
    fn summarize_short(&self) -> String {
        format!("{}...", &self.summarize()[..20])
    }
}

impl Summary for User {
    fn summarize(&self) -> String {
        format!("{} (age {})", self.name, self.age)
    }
}

// Generics with trait bounds
fn print_summary(item: &impl Summary) {
    println!("{}", item.summarize());
}

// Complex constraints with where clauses
fn process_items<T>(items: &[T]) -> Vec<String>
where
    T: Summary + std::fmt::Debug,
{
    items.iter()
        .map(|item| item.summarize())
        .collect()
}

// Concurrency — Compile-time safety guaranteed by Send/Sync traits
use std::sync::{Arc, Mutex};
use std::thread;

fn concurrent_counter() {
    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];

    for _ in 0..10 {
        let counter = Arc::clone(&counter);
        let handle = thread::spawn(move || {
            let mut num = counter.lock().unwrap();
            *num += 1;
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Counter: {}", *counter.lock().unwrap());  // 10
}

// Message passing with channels
use std::sync::mpsc;

fn channel_example() {
    let (tx, rx) = mpsc::channel();

    // Sender side
    for i in 0..5 {
        let tx = tx.clone();
        thread::spawn(move || {
            tx.send(format!("Message {}", i)).unwrap();
        });
    }
    drop(tx);  // Close the original sender

    // Receiver side
    for received in rx {
        println!("Got: {}", received);
    }
}

// async/await (asynchronous processing)
use tokio;

#[tokio::main]
async fn main() {
    let urls = vec![
        "https://example.com",
        "https://example.org",
    ];

    let mut handles = vec![];
    for url in urls {
        handles.push(tokio::spawn(async move {
            let resp = reqwest::get(url).await.unwrap();
            (url.to_string(), resp.status().as_u16())
        }));
    }

    for handle in handles {
        let (url, status) = handle.await.unwrap();
        println!("{}: {}", url, status);
    }
}
```

### 2.4 Go — Garbage Collector + Lightweight Concurrency

```go
package main

import (
    "context"
    "fmt"
    "log"
    "sync"
    "time"
)

// Go: GC-based memory management + lightweight concurrency via goroutines
// Design philosophy: Simplicity, fast compilation, concurrency

// Structs (Go has no classes)
type User struct {
    Name string
    Age  int
    Tags []string
}

// Methods (functions with receivers)
func (u *User) AddTag(tag string) {
    u.Tags = append(u.Tags, tag)
}

func (u User) Display() string {
    return fmt.Sprintf("%s (age: %d, tags: %v)", u.Name, u.Age, u.Tags)
}

// Interfaces — implicit implementation (no implements declaration needed)
type Summarizer interface {
    Summarize() string
}

func (u User) Summarize() string {
    return fmt.Sprintf("%s (age %d)", u.Name, u.Age)
}

// User implicitly implements Summarizer
func PrintSummary(s Summarizer) {
    fmt.Println(s.Summarize())
}

// Error handling — explicit error value return
type AppError struct {
    Code    int
    Message string
}

func (e *AppError) Error() string {
    return fmt.Sprintf("[%d] %s", e.Code, e.Message)
}

func FindUser(name string) (*User, error) {
    if name == "" {
        return nil, &AppError{Code: 400, Message: "name is required"}
    }
    if name == "Alice" {
        return &User{Name: "Alice", Age: 30}, nil
    }
    return nil, &AppError{Code: 404, Message: "user not found"}
}

// Error identification with errors.Is / errors.As (Go 1.13+)
func example() {
    user, err := FindUser("Bob")
    if err != nil {
        var appErr *AppError
        if errors.As(err, &appErr) {
            log.Printf("App error: code=%d, msg=%s", appErr.Code, appErr.Message)
        } else {
            log.Printf("Unknown error: %v", err)
        }
        return
    }
    fmt.Println(user.Display())
}

// Goroutine + Channel — The core of Go's concurrency
func goroutineExample() {
    ch := make(chan string, 10)  // Buffered channel

    urls := []string{
        "https://example.com",
        "https://example.org",
        "https://example.net",
    }

    for _, url := range urls {
        go func(u string) {
            // Execute HTTP requests concurrently
            result := fmt.Sprintf("Fetched: %s", u)
            ch <- result
        }(url)
    }

    for range urls {
        fmt.Println(<-ch)
    }
}

// select statement — waiting on multiple channels
func selectExample() {
    ch1 := make(chan string)
    ch2 := make(chan string)

    go func() {
        time.Sleep(100 * time.Millisecond)
        ch1 <- "from ch1"
    }()

    go func() {
        time.Sleep(200 * time.Millisecond)
        ch2 <- "from ch2"
    }()

    for i := 0; i < 2; i++ {
        select {
        case msg := <-ch1:
            fmt.Println(msg)
        case msg := <-ch2:
            fmt.Println(msg)
        case <-time.After(1 * time.Second):
            fmt.Println("timeout")
        }
    }
}

// Cancellation with Context
func contextExample(ctx context.Context) error {
    ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
    defer cancel()

    ch := make(chan string, 1)
    go func() {
        // Long-running processing
        time.Sleep(3 * time.Second)
        ch <- "done"
    }()

    select {
    case result := <-ch:
        fmt.Println(result)
        return nil
    case <-ctx.Done():
        return ctx.Err()  // context.DeadlineExceeded or context.Canceled
    }
}

// WaitGroup to wait for multiple goroutine completions
func waitGroupExample() {
    var wg sync.WaitGroup
    results := make([]string, 5)

    for i := 0; i < 5; i++ {
        wg.Add(1)
        go func(idx int) {
            defer wg.Done()
            results[idx] = fmt.Sprintf("result-%d", idx)
        }(i)
    }

    wg.Wait()
    fmt.Println(results)
}

// Generics (Go 1.18+)
func MapT any, U any U) []U {
    result := make([]U, len(slice))
    for i, v := range slice {
        result[i] = f(v)
    }
    return result
}

func FilterT any bool) []T {
    var result []T
    for _, v := range slice {
        if predicate(v) {
            result = append(result, v)
        }
    }
    return result
}

// Type constraints
type Number interface {
    ~int | ~int64 | ~float64
}

func SumT Number T {
    var total T
    for _, n := range numbers {
        total += n
    }
    return total
}

// Usage example
func genericsExample() {
    names := []string{"Alice", "Bob", "Carol"}
    upper := Map(names, strings.ToUpper)
    // → ["ALICE", "BOB", "CAROL"]

    numbers := []int{1, 2, 3, 4, 5, 6}
    evens := Filter(numbers, func(n int) bool { return n%2 == 0 })
    // → [2, 4, 6]

    total := Sum([]int{1, 2, 3, 4, 5})
    // → 15
}
```

### 2.5 Zig — Compile-Time Computation and Explicitness

```zig
const std = @import("std");
const Allocator = std.mem.Allocator;

// Zig: Manual memory management + comptime (compile-time computation)
// Design philosophy: No hidden control flow, no hidden memory allocations

// Explicitly passing allocators (no hidden allocations)
const User = struct {
    name: []const u8,
    age: u32,
    tags: std.ArrayList([]const u8),
    allocator: Allocator,

    pub fn init(allocator: Allocator, name: []const u8, age: u32) User {
        return User{
            .name = name,
            .age = age,
            .tags = std.ArrayList([]const u8).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *User) void {
        self.tags.deinit();
    }

    pub fn addTag(self: *User, tag: []const u8) !void {
        try self.tags.append(tag);
    }

    pub fn display(self: User) void {
        std.debug.print("User: {s} (age: {})\n", .{ self.name, self.age });
    }
};

// Error handling — error union types
const FileError = error{
    FileNotFound,
    PermissionDenied,
    OutOfMemory,
};

fn readConfig(path: []const u8) FileError![]const u8 {
    const file = std.fs.cwd().openFile(path, .{}) catch |err| switch (err) {
        error.FileNotFound => return FileError.FileNotFound,
        error.AccessDenied => return FileError.PermissionDenied,
        else => return FileError.FileNotFound,
    };
    defer file.close();

    // File reading
    return file.readToEndAlloc(std.heap.page_allocator, 1024 * 1024) catch {
        return FileError.OutOfMemory;
    };
}

// comptime — Compile-time computation (Zig's most distinctive feature)
fn fibonacci(comptime n: u32) u32 {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

// Computed at compile time, so zero runtime cost
const fib_10 = fibonacci(10);  // Computed as 55 at compile time

// Type generation with comptime
fn Matrix(comptime T: type, comptime rows: usize, comptime cols: usize) type {
    return struct {
        data: [rows][cols]T,

        const Self = @This();

        pub fn init() Self {
            return Self{
                .data = [_][cols]T{[_]T{0} ** cols} ** rows,
            };
        }

        pub fn get(self: Self, row: usize, col: usize) T {
            return self.data[row][col];
        }

        pub fn set(self: *Self, row: usize, col: usize, value: T) void {
            self.data[row][col] = value;
        }
    };
}

// Usage example
const Mat3x3 = Matrix(f64, 3, 3);

pub fn main() void {
    var mat = Mat3x3.init();
    mat.set(0, 0, 1.0);
    mat.set(1, 1, 1.0);
    mat.set(2, 2, 1.0);
    // 3x3 identity matrix
}

// defer / errdefer — Resource management
fn processFile(path: []const u8) !void {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();  // Always executed when function exits

    const buffer = try std.heap.page_allocator.alloc(u8, 4096);
    errdefer std.heap.page_allocator.free(buffer);  // Executed only on error

    // File processing...
}

// Tests (built into the language)
test "fibonacci" {
    try std.testing.expectEqual(fibonacci(0), 0);
    try std.testing.expectEqual(fibonacci(1), 1);
    try std.testing.expectEqual(fibonacci(10), 55);
}

test "user creation" {
    var user = User.init(std.testing.allocator, "Alice", 30);
    defer user.deinit();

    try user.addTag("admin");
    try std.testing.expectEqual(user.tags.items.len, 1);
}
```

---

## 3. Comparison of Error Handling Strategies

```
┌──────────┬────────────────────┬──────────────────────────────────┐
│ Language │ Primary Mechanism  │ Characteristics                  │
├──────────┼────────────────────┼──────────────────────────────────┤
│ C        │ Return values +    │ Easy to miss error checks        │
│          │ errno              │ Limited information              │
├──────────┼────────────────────┼──────────────────────────────────┤
│ C++      │ Exceptions + RAII  │ Auto cleanup via stack unwinding │
│          │ std::expected      │ noexcept to declare exception-   │
│          │ (C++23)            │ free functions                   │
├──────────┼────────────────────┼──────────────────────────────────┤
│ Rust     │ Result<T,E> +      │ Concise error propagation with ? │
│          │ Option<T>          │ panic! only for unrecoverable    │
│          │                    │ errors                           │
├──────────┼────────────────────┼──────────────────────────────────┤
│ Go       │ error interface    │ Return (value, error) tuples     │
│          │ errors.Is/As       │ Explicit but tends to be verbose │
├──────────┼────────────────────┼──────────────────────────────────┤
│ Zig      │ error union        │ Concise with try / catch         │
│          │ errdefer           │ Easy error-time cleanup          │
└──────────┴────────────────────┴──────────────────────────────────┘
```

### Comparing the Same Error Handling Pattern

```c
// C: Communicating errors via return values
#include <stdio.h>
#include <errno.h>

int read_int_from_file(const char* path, int* result) {
    FILE* f = fopen(path, "r");
    if (!f) {
        return -1;  // Error (error code in errno)
    }

    if (fscanf(f, "%d", result) != 1) {
        fclose(f);
        return -2;  // Parse error
    }

    fclose(f);
    return 0;  // Success
}

// Caller side
void caller(void) {
    int value;
    int ret = read_int_from_file("config.txt", &value);
    if (ret == -1) {
        fprintf(stderr, "Cannot open file: %s\n", strerror(errno));
    } else if (ret == -2) {
        fprintf(stderr, "Parse error\n");
    } else {
        printf("Value: %d\n", value);
    }
}
```

```cpp
// C++: Exceptions + std::expected (C++23)
#include <expected>
#include <fstream>
#include <string>

enum class ReadError {
    FileNotFound,
    ParseError,
};

// C++23: std::expected
std::expected<int, ReadError> read_int_from_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return std::unexpected(ReadError::FileNotFound);
    }

    int value;
    if (!(file >> value)) {
        return std::unexpected(ReadError::ParseError);
    }

    return value;
}

// Caller side
void caller() {
    auto result = read_int_from_file("config.txt");
    if (result.has_value()) {
        std::cout << "Value: " << result.value() << std::endl;
    } else {
        switch (result.error()) {
            case ReadError::FileNotFound:
                std::cerr << "File not found" << std::endl;
                break;
            case ReadError::ParseError:
                std::cerr << "Parse error" << std::endl;
                break;
        }
    }

    // Chaining with transform
    auto doubled = read_int_from_file("config.txt")
        .transform( { return v * 2; });
}
```

```rust
// Rust: Result + ? operator
use std::fs;
use std::num::ParseIntError;
use thiserror::Error;

#[derive(Error, Debug)]
enum ReadError {
    #[error("Cannot open file: {0}")]
    FileNotFound(#[from] std::io::Error),
    #[error("Parse error: {0}")]
    ParseError(#[from] ParseIntError),
}

fn read_int_from_file(path: &str) -> Result<i32, ReadError> {
    let content = fs::read_to_string(path)?;  // io::Error → ReadError
    let value: i32 = content.trim().parse()?;  // ParseIntError → ReadError
    Ok(value)
}

// Caller side
fn caller() {
    match read_int_from_file("config.txt") {
        Ok(value) => println!("Value: {}", value),
        Err(ReadError::FileNotFound(e)) => eprintln!("File error: {}", e),
        Err(ReadError::ParseError(e)) => eprintln!("Parse error: {}", e),
    }

    // Chaining with map / and_then
    let doubled = read_int_from_file("config.txt")
        .map(|v| v * 2);
}
```

```go
// Go: error interface
package main

import (
    "errors"
    "fmt"
    "os"
    "strconv"
    "strings"
)

type ReadError struct {
    Kind    string
    Message string
    Err     error
}

func (e *ReadError) Error() string {
    return fmt.Sprintf("%s: %s", e.Kind, e.Message)
}

func (e *ReadError) Unwrap() error {
    return e.Err
}

func readIntFromFile(path string) (int, error) {
    data, err := os.ReadFile(path)
    if err != nil {
        return 0, &ReadError{
            Kind:    "file_not_found",
            Message: fmt.Sprintf("Cannot open file: %s", path),
            Err:     err,
        }
    }

    value, err := strconv.Atoi(strings.TrimSpace(string(data)))
    if err != nil {
        return 0, &ReadError{
            Kind:    "parse_error",
            Message: "Parse error",
            Err:     err,
        }
    }

    return value, nil
}

// Caller side
func caller() {
    value, err := readIntFromFile("config.txt")
    if err != nil {
        var readErr *ReadError
        if errors.As(err, &readErr) {
            fmt.Printf("Error type: %s, Message: %s\n", readErr.Kind, readErr.Message)
        } else {
            fmt.Printf("Unknown error: %v\n", err)
        }
        return
    }
    fmt.Printf("Value: %d\n", value)
}
```

---

## 4. Build Systems and Toolchains

```
┌──────────┬──────────────┬─────────────────────────────────────┐
│ Language │ Build Tools  │ Characteristics                     │
├──────────┼──────────────┼─────────────────────────────────────┤
│ C        │ Make, CMake  │ Historically most widely used        │
│          │ Meson, Ninja │ Build script writing is complex      │
│          │              │ High platform dependency             │
├──────────┼──────────────┼─────────────────────────────────────┤
│ C++      │ CMake        │ De facto standard but config is hard │
│          │ Bazel        │ For large projects (by Google)       │
│          │ Conan,vcpkg  │ Package managers                     │
├──────────┼──────────────┼─────────────────────────────────────┤
│ Rust     │ Cargo        │ Integrated build+pkg+test+bench      │
│          │              │ TOML config, cargo.lock for          │
│          │              │ reproducibility                      │
│          │              │ Best-in-class toolchain experience   │
├──────────┼──────────────┼─────────────────────────────────────┤
│ Go       │ go build     │ Module management with go mod        │
│          │              │ No external tools needed, everything │
│          │              │ with the go command                  │
│          │              │ Cross-compilation is very easy       │
├──────────┼──────────────┼─────────────────────────────────────┤
│ Zig      │ zig build    │ Declarative builds with build.zig   │
│          │              │ Can also be used as a C/C++ cross-   │
│          │              │ compiler                             │
│          │              │ Bundles libc                         │
└──────────┴──────────────┴─────────────────────────────────────┘
```

```toml
# Rust: Cargo.toml — Exemplary dependency management
[package]
name = "my-cli-tool"
version = "0.1.0"
edition = "2021"
rust-version = "1.75"

[dependencies]
clap = { version = "4", features = ["derive"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tokio = { version = "1", features = ["full"] }
anyhow = "1"
tracing = "0.1"

[dev-dependencies]
assert_cmd = "2"
predicates = "3"
tempfile = "3"

[profile.release]
lto = true        # Link-time optimization
strip = true      # Strip debug info
codegen-units = 1 # Maximum optimization
```

```go
// Go: go.mod — Simple module management
// go.mod
module github.com/user/myapp

go 1.22

require (
    github.com/gin-gonic/gin v1.9.1
    github.com/jackc/pgx/v5 v5.5.3
    go.uber.org/zap v1.27.0
)

// Cross-compilation (single command)
// GOOS=linux GOARCH=amd64 go build -o myapp-linux
// GOOS=darwin GOARCH=arm64 go build -o myapp-mac
// GOOS=windows GOARCH=amd64 go build -o myapp.exe
```

---

## 5. Application Domain Details

### 5.1 Best Language by Domain Mapping

```
OS Kernels:
  C:    Linux kernel (over 30 million lines)
  Rust: Linux kernel new modules (officially supported since 6.1)
  C++:  Parts of the Windows kernel

Browser Engines:
  C++:  Chromium (Blink), WebKit
  Rust: Firefox (Servo components → Stylo CSS engine)

Databases:
  C:    SQLite, PostgreSQL
  C++:  MySQL, MongoDB, RocksDB, ClickHouse
  Rust: SurrealDB, TiKV, Neon (PostgreSQL-compatible)
  Go:   CockroachDB, TiDB, InfluxDB

Game Engines:
  C++:  Unreal Engine, Unity (C# + C++ internals)
  Rust: Bevy (emerging but growing)
  Zig:  Some indie game engines

Cloud Infrastructure:
  Go:   Docker, Kubernetes, Terraform, Prometheus, Grafana, etcd
  Rust: Firecracker (AWS Lambda foundation), Bottlerocket, Linkerd2-proxy

CLI Tools:
  Rust: ripgrep, bat, fd, exa/eza, starship, zoxide, delta
  Go:   gh (GitHub CLI), lazygit, fzf, Hugo, k9s

Cryptography & Security:
  Rust: Parts of BoringSSL, rustls
  C:    OpenSSL, libsodium
  Go:   crypto/tls (standard library)

Embedded / IoT:
  C:    Overwhelming market share (FreeRTOS, parts of Zephyr)
  Rust: Embassy (async embedded), RTIC
  Zig:  Embedded Linux, microcontrollers

WebAssembly:
  Rust: Yew, Leptos, wasm-bindgen
  C/C++: Emscripten
  Go:   TinyGo (optimized version)
  Zig:  Native Wasm output
```

### 5.2 Performance Benchmarks

```
Benchmark: HTTP Server (requests/sec, higher is better)
  C (direct epoll):  500,000+
  Rust (actix):      400,000+
  Go (net/http):     200,000+
  C++ (drogon):      350,000+
  Zig (zap):         450,000+

Benchmark: JSON Parsing (1GB file)
  C (simdjson):       2.5 GB/s
  Rust (simd-json):   2.3 GB/s
  C++ (simdjson):     2.5 GB/s
  Go (encoding):      0.3 GB/s
  Go (sonic):         1.5 GB/s

Benchmark: Compile Time (medium-sized project)
  Go:    2-5 seconds
  C:     5-15 seconds
  Zig:   5-15 seconds
  Rust:  30-120 seconds (incremental: 10-30 seconds)
  C++:   60-300 seconds

Benchmark: Binary Size (Hello World)
  C:     16 KB (static: 800 KB)
  Go:    1.8 MB (static by default)
  Rust:  300 KB (stripped)
  Zig:   5 KB (stripped)
  C++:   20 KB (dynamic)

* Actual performance depends heavily on the workload
* Treat benchmarks as reference values
```

---

## 6. The Memory Safety Discussion

### 6.1 U.S. Government Advisory (2024)

```
In February 2024, the U.S. White House recommended "transition to memory-safe languages":
- Memory-related vulnerabilities in C/C++ are a primary cause of cyberattacks
- ~70% of CVEs are caused by memory safety issues
- Recommended memory-safe languages: Rust, Go, Java, C#, etc.

Impact:
- Accelerated Rust adoption in the Linux kernel
- Increasing Rust ratio in new Android code
- DARPA's TRACTOR program (automated C → Rust conversion research)
- NSA cybersecurity guidance recommending Rust
```

### 6.2 Memory Safety Mechanisms by Language

```
C:
  ✗ Buffer overflow
  ✗ Use After Free
  ✗ Double free
  ✗ Null pointer dereference
  △ Dynamic detection with AddressSanitizer, Valgrind

C++:
  △ Partially solved with smart pointers
  △ Resource leak prevention with RAII
  ✗ Unsafe operations with raw pointers still possible
  △ Static analysis tools (Clang-Tidy, PVS-Studio)

Rust:
  ✓ Compile-time guarantees via ownership + borrow checker
  ✓ No null (expressed with Option<T>)
  ✓ No data races (Send/Sync traits)
  △ No guarantees inside unsafe blocks (keep to a minimum)

Go:
  ✓ No memory leaks/dangling pointers thanks to GC
  ✓ Bounds checking (at runtime)
  △ Race detector (dynamic detection)
  ✗ Panics from null pointers (nil) are possible

Zig:
  △ Manual management but allocator explicitness aids tracking
  △ Leak detection with test allocator
  △ Undefined behavior exists but is minimized
  ✓ Safety checks controllable via build mode
```

---

## 7. Detailed Concurrency Model Comparison

```
┌──────────────┬──────────────────────────────────────────────┐
│ C            │ POSIX threads (pthread)                      │
│              │ - Low-level, direct OS thread manipulation   │
│              │ - mutex, condition variable, semaphore        │
│              │ - Error-prone (deadlocks, races)             │
├──────────────┼──────────────────────────────────────────────┤
│ C++          │ std::thread + std::async (C++11)             │
│              │ std::jthread (C++20, auto-join)              │
│              │ - std::mutex, std::shared_mutex              │
│              │ - Lock-free programming with std::atomic     │
│              │ - coroutines (C++20)                         │
├──────────────┼──────────────────────────────────────────────┤
│ Rust         │ std::thread + crossbeam                      │
│              │ - Compile-time safety via Send/Sync traits   │
│              │ - Shared state with Arc<Mutex<T>>            │
│              │ - mpsc channels, crossbeam channels          │
│              │ - async/await (tokio, async-std)              │
│              │ - Rayon (data parallelism)                   │
├──────────────┼──────────────────────────────────────────────┤
│ Go           │ goroutine + channel                          │
│              │ - Lightweight (initial 2KB stack, dynamic)   │
│              │ - Can run millions of goroutines             │
│              │ - select statement for multi-channel waiting │
│              │ - "Don't communicate by sharing memory;       │
│              │    share memory by communicating"             │
├──────────────┼──────────────────────────────────────────────┤
│ Zig          │ async/await (built into the language)        │
│              │ - Event-driven I/O                           │
│              │ - OS threads with std.Thread                 │
│              │ - Control via allocators                     │
└──────────────┴──────────────────────────────────────────────┘
```

---

## 8. Practical Project Structure Examples

### 8.1 Rust CLI Project

```
my-cli/
├── Cargo.toml
├── Cargo.lock
├── src/
│   ├── main.rs           # Entry point
│   ├── lib.rs            # Library root
│   ├── cli.rs            # CLI definition with clap
│   ├── config.rs         # Configuration management
│   ├── commands/
│   │   ├── mod.rs
│   │   ├── init.rs       # init subcommand
│   │   └── run.rs        # run subcommand
│   ├── core/
│   │   ├── mod.rs
│   │   ├── engine.rs     # Core logic
│   │   └── types.rs      # Type definitions
│   └── utils/
│       ├── mod.rs
│       └── fs.rs         # File system utilities
├── tests/
│   ├── integration_test.rs
│   └── fixtures/
├── benches/
│   └── benchmark.rs      # Benchmarking with criterion
└── .github/
    └── workflows/
        └── ci.yml
```

### 8.2 Go Web API Project

```
myapp/
├── go.mod
├── go.sum
├── cmd/
│   └── server/
│       └── main.go        # Entry point
├── internal/              # Cannot be imported externally
│   ├── handler/
│   │   ├── user.go        # User handler
│   │   └── middleware.go  # Middleware
│   ├── service/
│   │   └── user.go        # Business logic
│   ├── repository/
│   │   └── user.go        # Data access
│   ├── model/
│   │   └── user.go        # Domain model
│   └── config/
│       └── config.go      # Configuration
├── pkg/                   # Can be imported externally
│   └── response/
│       └── json.go
├── migrations/
│   └── 001_create_users.sql
├── Dockerfile
├── Makefile
└── .github/
    └── workflows/
        └── ci.yml
```

---

## 9. Detailed Selection Guideline Flowchart

```
Q1: Can you tolerate GC pauses?
├── Yes → Q2
└── No → Q3

Q2: Are simplicity and fast compilation important?
├── Yes → Go
│   Use cases: Microservices, CLI, DevOps tools
│   Advantage: Easy to learn, easy to standardize across teams
└── No → Q4

Q3: Is memory safety required?
├── Yes → Rust
│   Use cases: Infrastructure, security, Wasm, CLI
│   Advantage: Compile-time safety guarantees, zero-cost abstractions
└── No → Q5

Q4: Will you make heavy use of functional programming or generics?
├── Yes → Rust
│   Use cases: Libraries, frameworks, language tools
└── No → Go
    Use cases: CRUD APIs, network services

Q5: Do you need integration with existing C/C++ codebases?
├── C++ codebase → C++
│   Use cases: Games, browsers, extending existing systems
├── C codebase → C or Zig
│   Zig can directly import C headers
└── New project → Rust or Zig
    Zig: C replacement, embedded-focused

Q6: Is it game development?
├── Yes → C++ (Unreal) or Rust (Bevy)
└── No → Follow the flowchart above
```

### Common Misconceptions and Corrections

```
Misconception: "Go is slow"
Reality: Has GC but is fast enough for HTTP servers. No need to choose C++
         for many use cases.

Misconception: "Rust is too hard to be practical"
Reality: The learning curve is steep, but productivity is high once proficient.
         The 2-4 week period of getting used to the borrow checker is the hurdle.

Misconception: "C is old and should not be used"
Reality: Still the optimal choice for embedded systems, kernels, and specialized
         systems. C has the best ABI stability.

Misconception: "C++ is too complex"
Reality: Modern C++ (C++17/20/23) has improved significantly.
         However, you don't need to use all features. Limit features used per project.

Misconception: "Zig is still experimental"
Reality: Production usage is growing. Uber's bazel-zig-cc,
         Bun (JavaScript runtime) is written in Zig.
```

---

## 10. Learning Resources and Roadmap

```
C:
  Beginner: K&R "The C Programming Language"
  Practical: "Expert C Programming"
  Duration: Basic syntax 2 weeks, mastering pointers 1-2 months

C++:
  Beginner: "A Tour of C++" (Stroustrup)
  Practical: "Effective Modern C++" (Meyers)
  Duration: Basic syntax 1 month, mastering Modern C++ 3-6 months

Rust:
  Beginner: "The Rust Programming Language" (official Book)
  Practical: "Rust in Action", "Zero To Production"
  Duration: Basic syntax 2-3 weeks, overcoming borrow checker 1-2 months

Go:
  Beginner: "The Go Programming Language" (Donovan & Kernighan)
  Practical: "Let's Go" (Alex Edwards)
  Duration: Basic syntax 1-2 weeks, production level 1-2 months

Zig:
  Beginner: ziglearn.org, "Zig Guide"
  Practical: zig.guide, std library source code
  Duration: Basic syntax 2-3 weeks, mastering comptime 1-2 months
```


---

## Hands-On Exercises

### Exercise 1: Basic Implementation

Implement code that satisfies the following requirements.

**Requirements:**
- Validate input data
- Implement appropriate error handling
- Write test code as well

```python
# Exercise 1: Basic Implementation Template
class Exercise1:
    """Exercise for basic implementation patterns"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """Validate input value"""
        if value is None:
            raise ValueError("Input value is None")
        return True

    def process(self, value):
        """Main processing logic"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """Get processing results"""
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
        assert False, "Should have raised an exception"
    except ValueError:
        pass

    print("All tests passed!")

test_exercise1()
```

### Exercise 2: Advanced Patterns

Extend the basic implementation by adding the following features.

```python
# Exercise 2: Advanced Patterns
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """Exercise for advanced patterns"""

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
        """Delete by key"""
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
# Exercise 3: Performance Optimization
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
    print(f"Speedup factor: {slow_time/fast_time:.0f}x")

benchmark()
```

**Key Points:**
- Be mindful of algorithmic complexity
- Choose appropriate data structures
- Measure effectiveness with benchmarks
---


## FAQ

### Q1: What is the most important point for learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory, but by actually writing code and verifying its behavior.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving to the next step.

### Q3: How is this applied in professional practice?

The knowledge in this topic is frequently used in day-to-day development work. It becomes particularly important during code reviews and architecture design.

---

## Summary

| Language | Philosophy | Best Use Case | Status in 2025 |
|------|------|-----------------|-------------|
| C | Minimal abstraction | OS, Kernel, Embedded | Unshakeable position. The lingua franca of ABIs |
| C++ | Zero-cost abstraction | Games, Browsers, DBs | Major improvements with C++23. Still massive |
| Rust | Safety + Performance | Infrastructure, CLI, Wasm | Rapid growth. Legitimacy established via Linux kernel adoption |
| Go | Simplicity + Concurrency | Cloud, Microservices | De facto standard language for cloud infrastructure |
| Zig | Modern C replacement | Embedded, Systems | Gained visibility through Bun. A successor candidate to C |

---

## Recommended Next Guides

---

## References
1. Blandy, J., Orendorff, J. & Tindall, L. "Programming Rust." 2nd Ed, O'Reilly, 2021.
2. Donovan, A. & Kernighan, B. "The Go Programming Language." Addison-Wesley, 2015.
3. Stroustrup, B. "A Tour of C++." 3rd Ed, Addison-Wesley, 2022.
4. Kernighan, B. & Ritchie, D. "The C Programming Language." 2nd Ed, Prentice Hall, 1988.
5. Klabnik, S. & Nichols, C. "The Rust Programming Language." No Starch Press, 2023.
6. "The White House: Back to the Building Blocks." Technical Report, 2024.
7. "Rust for Linux." rust-for-linux.com.
8. "State of Developer Ecosystem 2024." JetBrains.
9. Kelley, A. "The Zig Programming Language." ziglang.org.
10. "Benchmarks Game." benchmarksgame-team.pages.debian.net.
