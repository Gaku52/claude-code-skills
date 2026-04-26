# Types and Traits -- The Foundation of Rust's Type System and Polymorphism

> Rust's type system is built on algebraic data types via struct/enum and ad-hoc polymorphism via traits, achieving zero-cost abstractions when combined with generics.

---

## What You Will Learn in This Chapter

1. **struct and enum** -- Understand the definition and use of algebraic data types (product types and sum types)
2. **Traits** -- Master the definition, implementation, and default methods of interfaces
3. **Generics and Trait Bounds** -- Learn how to write generic code with type parameters and constraints
4. **Dynamic Dispatch and Static Dispatch** -- Understand when to use trait objects versus monomorphization
5. **Advanced Trait Patterns** -- Learn associated types, supertraits, and blanket implementations


## Prerequisites

The following knowledge will deepen your understanding before reading this guide:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding of [Ownership and Borrowing -- Rust's Most Innovative Memory Management Paradigm](./01-ownership-borrowing.md)

---

## 1. Basic Types

### 1.1 List of Primitive Types

```
┌─────────────────────────────────────────────────────┐
│                 Rust Primitive Types                │
├──────────┬──────────────────────────────────────────┤
│ Integer  │ i8 i16 i32 i64 i128 isize                │
│          │ u8 u16 u32 u64 u128 usize                │
├──────────┼──────────────────────────────────────────┤
│ Float    │ f32, f64                                 │
├──────────┼──────────────────────────────────────────┤
│ Boolean  │ bool                                     │
├──────────┼──────────────────────────────────────────┤
│ Char     │ char (4-byte Unicode scalar value)       │
├──────────┼──────────────────────────────────────────┤
│ Tuple    │ (T1, T2, ...) -- combination of types    │
├──────────┼──────────────────────────────────────────┤
│ Array    │ [T; N] -- fixed length, on stack         │
├──────────┼──────────────────────────────────────────┤
│ Slice    │ [T] -- dynamically sized type (DST),     │
│          │        always used through references    │
├──────────┼──────────────────────────────────────────┤
│ Reference│ &T, &mut T                               │
├──────────┼──────────────────────────────────────────┤
│ String   │ str -- string slice (DST)                │
├──────────┼──────────────────────────────────────────┤
│ unit     │ () -- no value (equivalent to C's void)  │
├──────────┼──────────────────────────────────────────┤
│ never    │ ! -- return type of non-returning fns    │
└──────────┴──────────────────────────────────────────┘
```

### 1.2 Numeric Type Details

```rust
fn main() {
    // Integer literal notations
    let decimal = 98_222;           // decimal (separable with _)
    let hex = 0xff;                 // hexadecimal
    let octal = 0o77;              // octal
    let binary = 0b1111_0000;       // binary
    let byte = b'A';               // byte literal (u8)

    // Type suffix
    let x = 42u32;                  // u32 type
    let y = 3.14f64;               // f64 type

    // Type conversion (as cast)
    let a: i32 = 42;
    let b: f64 = a as f64;         // widening conversion
    let c: u8 = 300u16 as u8;      // narrowing conversion (truncated: 44)

    // Safe type conversion
    let d: u16 = 300;
    let e: u8 = u8::try_from(d).unwrap_or(u8::MAX);  // returns Result

    println!("decimal={}, hex={}, binary={}", decimal, hex, binary);
    println!("b={}, c={}, e={}", b, c, e);
}
```

### 1.3 String Type Hierarchy

```
┌──────────────────────────────────────────────────────────┐
│                 Rust String Type Hierarchy               │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Owned types         Reference types (slices)            │
│  ┌──────────┐       ┌──────────┐                        │
│  │  String   │ ────> │  &str    │  Deref coercion       │
│  │ (heap)   │       │ (ref)    │                        │
│  └──────────┘       └──────────┘                        │
│                                                          │
│  ┌──────────┐       ┌──────────┐                        │
│  │ OsString │ ────> │  &OsStr  │  OS-specific strings    │
│  └──────────┘       └──────────┘                        │
│                                                          │
│  ┌──────────┐       ┌──────────┐                        │
│  │  CString │ ────> │  &CStr   │  C-compatible (NUL-term)│
│  └──────────┘       └──────────┘                        │
│                                                          │
│  ┌──────────┐       ┌──────────┐                        │
│  │ PathBuf  │ ────> │  &Path   │  File paths             │
│  └──────────┘       └──────────┘                        │
└──────────────────────────────────────────────────────────┘
```

```rust
fn main() {
    // String (owned) and &str (borrowed)
    let owned: String = String::from("hello");
    let borrowed: &str = "hello";  // string literals are &'static str

    // String -> &str
    let slice: &str = &owned;
    let slice2: &str = owned.as_str();

    // &str -> String
    let owned2: String = borrowed.to_string();
    let owned3: String = String::from(borrowed);

    // String concatenation
    let s1 = String::from("hello");
    let s2 = String::from(" world");
    let s3 = s1 + &s2;  // s1 is moved, s2 is borrowed
    // println!("{}", s1);  // error: s1 has been moved
    println!("{}", s3);

    // format! macro (does not move any variables)
    let s4 = String::from("hello");
    let s5 = format!("{} {}", s4, "world");
    println!("{}, {}", s4, s5); // both still valid
}
```

---

## 2. struct (Structures)

### Example 1: Named-field Structures

```rust
struct User {
    name: String,
    email: String,
    age: u32,
    active: bool,
}

impl User {
    // Associated function (constructor)
    fn new(name: &str, email: &str, age: u32) -> Self {
        Self {
            name: name.to_string(),
            email: email.to_string(),
            age,
            active: true,
        }
    }

    // Builder-style method using field update syntax
    fn with_active(self, active: bool) -> Self {
        Self { active, ..self }
    }

    fn display_name(&self) -> &str {
        &self.name
    }

    fn is_adult(&self) -> bool {
        self.age >= 18
    }
}

fn main() {
    let user = User::new("Tanaka", "tanaka@example.com", 30);
    println!("{} ({} years old)", user.name, user.age);

    // Struct update syntax
    let user2 = User {
        name: String::from("Suzuki"),
        email: String::from("suzuki@example.com"),
        ..user  // remaining fields are taken from user (note: move occurs)
    };
    println!("{} ({} years old)", user2.name, user2.age); // age=30, active=true

    // Method chaining
    let user3 = User::new("Sato", "sato@example.com", 15)
        .with_active(false);
    println!("Is {} an adult? {}", user3.display_name(), user3.is_adult());
}
```

### Example 2: Tuple Structures and Unit Structures

```rust
// Tuple struct: no field names (useful for the newtype pattern)
struct Color(u8, u8, u8);
struct Meters(f64);
struct Celsius(f64);
struct Fahrenheit(f64);

// Ensure type safety with the newtype pattern
impl Celsius {
    fn to_fahrenheit(&self) -> Fahrenheit {
        Fahrenheit(self.0 * 9.0 / 5.0 + 32.0)
    }
}

impl Fahrenheit {
    fn to_celsius(&self) -> Celsius {
        Celsius((self.0 - 32.0) * 5.0 / 9.0)
    }
}

// Unit struct: no fields (used as a marker type)
struct Marker;
struct Production;
struct Development;

fn main() {
    let red = Color(255, 0, 0);
    let distance = Meters(42.0);
    let temp_c = Celsius(100.0);
    let temp_f = temp_c.to_fahrenheit();

    println!("R={}", red.0);
    println!("distance={}m", distance.0);
    println!("{}°C = {}°F", temp_c.0, temp_f.0);

    // The type system prevents mistakes that would confuse Meters and f64
    // let wrong: Meters = Celsius(30.0);  // compile error!
}
```

### 2.1 Memory Layout of Structures

```
┌──────────────────────────────────────────────────────┐
│ Memory layout of struct User                         │
├──────────────────────────────────────────────────────┤
│                                                      │
│  Stack                          Heap                 │
│  ┌──────────────────────┐                            │
│  │ name: String         │                            │
│  │   ptr ─────────────────────> "Tanaka"             │
│  │   len: 6             │                            │
│  │   cap: 6             │                            │
│  ├──────────────────────┤                            │
│  │ email: String        │                            │
│  │   ptr ─────────────────────> "tanaka@example.com" │
│  │   len: 18            │                            │
│  │   cap: 18            │                            │
│  ├──────────────────────┤                            │
│  │ age: u32 = 30        │   (4 bytes)                │
│  ├──────────────────────┤                            │
│  │ active: bool = true  │   (1 byte + padding)       │
│  └──────────────────────┘                            │
│                                                      │
│  The compiler optimizes field ordering to            │
│  minimize padding (unless repr(C) is specified)      │
└──────────────────────────────────────────────────────┘
```

---

## 3. enum (Enumerations)

### Example 3: enum as an Algebraic Data Type

```rust
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
            Shape::Triangle { base, height } => 0.5 * base * height,
        }
    }

    fn perimeter(&self) -> f64 {
        match self {
            Shape::Circle { radius } => 2.0 * std::f64::consts::PI * radius,
            Shape::Rectangle { width, height } => 2.0 * (width + height),
            Shape::Triangle { base, height } => {
                // Assuming an isosceles triangle
                let side = ((*base / 2.0).powi(2) + height.powi(2)).sqrt();
                base + 2.0 * side
            }
        }
    }

    fn describe(&self) -> String {
        match self {
            Shape::Circle { radius } => format!("Circle with radius {}", radius),
            Shape::Rectangle { width, height } => format!("Rectangle {}x{}", width, height),
            Shape::Triangle { base, height } => format!("Triangle with base {}, height {}", base, height),
        }
    }
}

fn main() {
    let shapes = vec![
        Shape::Circle { radius: 5.0 },
        Shape::Rectangle { width: 4.0, height: 6.0 },
        Shape::Triangle { base: 3.0, height: 8.0 },
    ];
    for s in &shapes {
        println!("{}: area={:.2}, perimeter={:.2}", s.describe(), s.area(), s.perimeter());
    }
}
```

### 3.1 Option and Result

```
┌────────────────────────────┬────────────────────────────────┐
│   Option<T>                │   Result<T, E>                 │
├────────────────────────────┼────────────────────────────────┤
│ enum Option<T> {           │ enum Result<T, E> {            │
│     Some(T),               │     Ok(T),                     │
│     None,                  │     Err(E),                    │
│ }                          │ }                              │
├────────────────────────────┼────────────────────────────────┤
│ Possibility that a value    │ Possibility that an operation  │
│ may not be present          │ may fail                       │
│ Safe alternative to null    │ Safe alternative to exceptions │
└────────────────────────────┴────────────────────────────────┘
```

### 3.2 Advanced enum Patterns

```rust
// Express state by attaching data to enum variants
#[derive(Debug)]
enum HttpResponse {
    Ok { body: String, content_type: String },
    NotFound { path: String },
    Redirect { url: String, permanent: bool },
    ServerError { message: String, code: u16 },
}

impl HttpResponse {
    fn status_code(&self) -> u16 {
        match self {
            HttpResponse::Ok { .. } => 200,
            HttpResponse::NotFound { .. } => 404,
            HttpResponse::Redirect { permanent: true, .. } => 301,
            HttpResponse::Redirect { permanent: false, .. } => 302,
            HttpResponse::ServerError { code, .. } => *code,
        }
    }

    fn is_success(&self) -> bool {
        matches!(self, HttpResponse::Ok { .. })
    }
}

// C-compatible enum
#[repr(u8)]
enum Color {
    Red = 1,
    Green = 2,
    Blue = 3,
}

// Size optimization for enums (Null Pointer Optimization)
fn size_demo() {
    use std::mem::size_of;

    // Option<Box<T>> is the same size as Box<T>!
    // None is internally represented as a null pointer
    assert_eq!(size_of::<Box<i32>>(), size_of::<Option<Box<i32>>>());
    assert_eq!(size_of::<&i32>(), size_of::<Option<&i32>>());

    println!("Box<i32>: {} bytes", size_of::<Box<i32>>());
    println!("Option<Box<i32>>: {} bytes", size_of::<Option<Box<i32>>>());
}

fn main() {
    let response = HttpResponse::Ok {
        body: "<h1>Hello</h1>".to_string(),
        content_type: "text/html".to_string(),
    };
    println!("Status: {}", response.status_code());
    println!("Success?: {}", response.is_success());

    size_demo();
}
```

### 3.3 Memory Layout of enums

```
┌──────────────────────────────────────────────────────┐
│ Memory layout of enum Shape                          │
├──────────────────────────────────────────────────────┤
│                                                      │
│  All variants occupy memory of the same size         │
│  (size of the largest variant + size of the tag)     │
│                                                      │
│  Circle:                                             │
│  ┌─────────┬──────────────┬──────────────┐          │
│  │ tag = 0 │ radius: f64  │  (unused)    │          │
│  └─────────┴──────────────┴──────────────┘          │
│                                                      │
│  Rectangle:                                          │
│  ┌─────────┬──────────────┬──────────────┐          │
│  │ tag = 1 │ width: f64   │ height: f64  │          │
│  └─────────┴──────────────┴──────────────┘          │
│                                                      │
│  Triangle:                                           │
│  ┌─────────┬──────────────┬──────────────┐          │
│  │ tag = 2 │ base: f64    │ height: f64  │          │
│  └─────────┴──────────────┴──────────────┘          │
│                                                      │
│  Size = tag(1-8 bytes) + max(variant) + padding      │
└──────────────────────────────────────────────────────┘
```

---

## 4. impl Blocks

### Example 4: Methods and Associated Functions

```rust
struct Rectangle {
    width: f64,
    height: f64,
}

impl Rectangle {
    // Associated function (constructor) -- does not take Self as a parameter
    fn new(width: f64, height: f64) -> Self {
        Self { width, height }
    }

    fn square(size: f64) -> Self {
        Self { width: size, height: size }
    }

    // Method -- takes &self as a parameter
    fn area(&self) -> f64 {
        self.width * self.height
    }

    fn perimeter(&self) -> f64 {
        2.0 * (self.width + self.height)
    }

    fn is_square(&self) -> bool {
        (self.width - self.height).abs() < f64::EPSILON
    }

    fn can_hold(&self, other: &Rectangle) -> bool {
        self.width >= other.width && self.height >= other.height
    }

    // Mutable method
    fn scale(&mut self, factor: f64) {
        self.width *= factor;
        self.height *= factor;
    }

    // Method that consumes self (used in Builder pattern, etc.)
    fn into_square(self) -> Rectangle {
        let side = self.width.max(self.height);
        Rectangle { width: side, height: side }
    }
}

// You can have multiple impl blocks (useful for separating trait implementations)
impl std::fmt::Display for Rectangle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Rectangle({}x{})", self.width, self.height)
    }
}

fn main() {
    let mut rect = Rectangle::new(10.0, 5.0);
    println!("{}", rect);                    // Rectangle(10x5)
    println!("Area: {}", rect.area());       // 50.0
    println!("Perimeter: {}", rect.perimeter());  // 30.0
    println!("Square? {}", rect.is_square()); // false

    let small = Rectangle::new(3.0, 2.0);
    println!("Can hold? {}", rect.can_hold(&small)); // true

    rect.scale(2.0);
    println!("After scaling: {}", rect);            // Rectangle(20x10)

    let sq = Rectangle::square(5.0);
    println!("Is {} a square? {}", sq, sq.is_square()); // true
}
```

---

## 5. Traits

### 5.1 Trait Definition and Implementation

### Example 5: Defining and Implementing a Trait

```rust
trait Summary {
    // Required method
    fn summarize_author(&self) -> String;

    // Default implementation
    fn summarize(&self) -> String {
        format!("(Read more from {}...)", self.summarize_author())
    }

    // A default implementation can call other methods
    fn preview(&self) -> String {
        let summary = self.summarize();
        if summary.len() > 50 {
            format!("{}...", &summary[..50])
        } else {
            summary
        }
    }
}

struct Article {
    title: String,
    author: String,
    content: String,
}

impl Summary for Article {
    fn summarize_author(&self) -> String {
        self.author.clone()
    }

    fn summarize(&self) -> String {
        format!("{} -- {} (by {})", self.title, &self.content[..20], self.author)
    }
}

struct Tweet {
    username: String,
    text: String,
}

impl Summary for Tweet {
    fn summarize_author(&self) -> String {
        format!("@{}", self.username)
    }
    // summarize() uses the default implementation
}
```

### Example 6: Generics with Trait Bounds

```rust
use std::fmt::{Display, Debug};

// Method 1: Trait bound syntax
fn notify<T: Summary + Display>(item: &T) {
    println!("Breaking news: {}", item.summarize());
}

// Method 2: where clause (recommended for complex cases)
fn complex_function<T, U>(t: &T, u: &U) -> String
where
    T: Summary + Clone,
    U: Display + Debug,
{
    format!("{}: {:?}", t.summarize(), u)
}

// Method 3: impl Trait syntax (for parameters)
fn notify_simple(item: &impl Summary) {
    println!("Breaking news: {}", item.summarize());
}

// impl Trait can also be used for return values (but only a single concrete type)
fn create_summarizable() -> impl Summary {
    Tweet {
        username: String::from("rustlang"),
        text: String::from("Rust is great!"),
    }
}

// Combining multiple trait bounds
fn process_and_display<T>(item: T)
where
    T: Summary + Display + Clone + Debug,
{
    let cloned = item.clone();
    println!("Summary: {}", item.summarize());
    println!("Display: {}", item);
    println!("Debug: {:?}", cloned);
}
```

### 5.2 Commonly Used Standard Traits

```
┌───────────────┬───────────────────────────────────────────┐
│ Trait         │ Purpose                                   │
├───────────────┼───────────────────────────────────────────┤
│ Display       │ {} formatted display                       │
│ Debug         │ {:?} debug display                         │
│ Clone         │ Explicit deep copy (.clone())              │
│ Copy          │ Implicit bit copy                          │
│ PartialEq/Eq  │ == / != comparison                         │
│ PartialOrd/Ord│ < > <= >= comparison / sorting             │
│ Hash          │ Hash value computation (required for       │
│               │ HashMap keys)                              │
│ Default       │ Default value generation                   │
│ From/Into     │ Type conversion                            │
│ TryFrom/TryInto│ Fallible type conversion                  │
│ Iterator      │ Iterator protocol                          │
│ IntoIterator  │ Make usable with for loops                 │
│ Drop          │ Destructor (handling on scope exit)        │
│ Deref/DerefMut│ Automatic dereferencing / smart pointers   │
│ AsRef/AsMut   │ Conversion to references                   │
│ Borrow        │ Conversion as a borrow (consistency        │
│               │ guarantee for Hash/Eq)                     │
│ Send/Sync     │ Thread safety markers                      │
│ Sized         │ Size known at compile time                 │
│ Fn/FnMut/FnOnce│ Closure / function call                   │
└───────────────┴───────────────────────────────────────────┘
```

### Example 7: Automatic Implementation with the derive Macro

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
struct Config {
    host: String,
    port: u16,
    debug: bool,
}

fn main() {
    let config = Config {
        host: "localhost".to_string(),
        port: 8080,
        debug: true,
    };
    let config2 = config.clone();
    println!("{:?}", config);
    println!("Same? {}", config == config2);

    let default_config = Config::default();
    println!("Default: {:?}", default_config);
    // Config { host: "", port: 0, debug: false }
}
```

### Example 8: Implementing Display and From/Into

```rust
use std::fmt;

#[derive(Debug)]
struct Temperature {
    celsius: f64,
}

impl Temperature {
    fn new(celsius: f64) -> Self {
        Temperature { celsius }
    }

    fn fahrenheit(&self) -> f64 {
        self.celsius * 9.0 / 5.0 + 32.0
    }
}

// Manual implementation of the Display trait
impl fmt::Display for Temperature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.1}°C ({:.1}°F)", self.celsius, self.fahrenheit())
    }
}

// Implementing the From trait (type conversion)
impl From<f64> for Temperature {
    fn from(celsius: f64) -> Self {
        Temperature { celsius }
    }
}

impl From<i32> for Temperature {
    fn from(celsius: i32) -> Self {
        Temperature { celsius: celsius as f64 }
    }
}

// Implementing From<T> for U automatically enables Into<U> for T
fn display_temp(temp: impl Into<Temperature>) {
    let t: Temperature = temp.into();
    println!("{}", t);
}

fn main() {
    let temp = Temperature::new(100.0);
    println!("{}", temp);       // Display: "100.0°C (212.0°F)"
    println!("{:?}", temp);     // Debug: "Temperature { celsius: 100.0 }"

    // Conversion via From/Into
    let t1: Temperature = 36.5f64.into();
    let t2: Temperature = Temperature::from(100);
    display_temp(25.0f64);
    display_temp(0);
}
```

### Example 9: Implementing PartialEq and Ord

```rust
#[derive(Debug, Clone)]
struct Student {
    name: String,
    score: u32,
}

// PartialEq: treat students with the same name as equal
impl PartialEq for Student {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}
impl Eq for Student {}

// Ord: sort by score (descending)
impl PartialOrd for Student {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Student {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.score.cmp(&self.score) // descending order
    }
}

fn main() {
    let mut students = vec![
        Student { name: "Tanaka".to_string(), score: 85 },
        Student { name: "Suzuki".to_string(), score: 92 },
        Student { name: "Sato".to_string(), score: 78 },
    ];

    students.sort(); // sort by Ord (score descending)
    for s in &students {
        println!("{}: {} points", s.name, s.score);
    }
    // Suzuki: 92 points, Tanaka: 85 points, Sato: 78 points
}
```

---

## 6. Generics

### Example 10: Generic Structures and Functions

```rust
use std::fmt::Display;

// Generic struct
struct Pair<T> {
    first: T,
    second: T,
}

impl<T: PartialOrd + Display> Pair<T> {
    fn new(first: T, second: T) -> Self {
        Self { first, second }
    }

    fn larger(&self) -> &T {
        if self.first >= self.second {
            &self.first
        } else {
            &self.second
        }
    }
}

// Pair of different types
struct MixedPair<T, U> {
    first: T,
    second: U,
}

impl<T: Display, U: Display> MixedPair<T, U> {
    fn display(&self) {
        println!("({}, {})", self.first, self.second);
    }
}

// Provide additional methods only for specific types
impl Pair<f64> {
    fn average(&self) -> f64 {
        (self.first + self.second) / 2.0
    }
}

// Generic function
fn find_max<T: PartialOrd>(list: &[T]) -> Option<&T> {
    if list.is_empty() {
        return None;
    }
    let mut max = &list[0];
    for item in &list[1..] {
        if item > max {
            max = item;
        }
    }
    Some(max)
}

fn main() {
    let pair = Pair::new(10, 20);
    println!("Larger: {}", pair.larger()); // 20

    let float_pair = Pair::new(3.14, 2.71);
    println!("Average: {}", float_pair.average()); // 2.925

    let mixed = MixedPair { first: "hello", second: 42 };
    mixed.display(); // (hello, 42)

    let numbers = vec![34, 50, 25, 100, 65];
    println!("Maximum: {}", find_max(&numbers).unwrap());
}
```

### 6.1 Monomorphization

```
Compile-time processing of generics:

Source code:
  fn max<T: PartialOrd>(a: T, b: T) -> T { ... }

  max(1i32, 2i32);
  max(3.14f64, 2.71f64);
  max("hello", "world");

After compilation (monomorphization):
  fn max_i32(a: i32, b: i32) -> i32 { ... }
  fn max_f64(a: f64, b: f64) -> f64 { ... }
  fn max_str(a: &str, b: &str) -> &str { ... }

  max_i32(1, 2);
  max_f64(3.14, 2.71);
  max_str("hello", "world");

-> No runtime overhead (zero-cost abstraction)
-> However, binary size may increase
```

---

## 7. Dynamic Dispatch and Trait Objects

### Example 11: dyn Trait (Dynamic Dispatch)

```rust
trait Animal {
    fn name(&self) -> &str;
    fn sound(&self) -> &str;
    fn info(&self) -> String {
        format!("{} says \"{}\"", self.name(), self.sound())
    }
}

struct Dog { name: String }
struct Cat { name: String }
struct Bird { name: String }

impl Animal for Dog {
    fn name(&self) -> &str { &self.name }
    fn sound(&self) -> &str { "Woof" }
}

impl Animal for Cat {
    fn name(&self) -> &str { &self.name }
    fn sound(&self) -> &str { "Meow" }
}

impl Animal for Bird {
    fn name(&self) -> &str { &self.name }
    fn sound(&self) -> &str { "Tweet" }
}

fn main() {
    // Putting different types into the same collection -> dyn Trait is needed
    let animals: Vec<Box<dyn Animal>> = vec![
        Box::new(Dog { name: "Pochi".to_string() }),
        Box::new(Cat { name: "Tama".to_string() }),
        Box::new(Bird { name: "Pi-ta".to_string() }),
    ];

    for animal in &animals {
        println!("{}", animal.info());
    }

    // Can also be used as function arguments
    fn describe_animal(animal: &dyn Animal) {
        println!("Animal: {} - {}", animal.name(), animal.sound());
    }

    describe_animal(&Dog { name: "Shiro".to_string() });
    describe_animal(&Cat { name: "Kuro".to_string() });
}
```

### 7.1 How vtable (Virtual Function Table) Works

```
  Memory layout of a trait object &dyn Animal:

  Fat pointer (2 words)
  ┌──────────────┐
  │ data ptr ────────────> Actual data (Dog, Cat, etc.)
  │ vtable ptr ──────────> vtable
  └──────────────┘

  vtable of Dog:
  ┌──────────────────────┐
  │ drop()               │  → Dog::drop
  │ size                 │  → sizeof(Dog)
  │ align                │  → alignof(Dog)
  │ name()               │  → Dog::name
  │ sound()              │  → Dog::sound
  │ info()               │  → Animal::info (default impl)
  └──────────────────────┘

  vtable of Cat:
  ┌──────────────────────┐
  │ drop()               │  → Cat::drop
  │ size                 │  → sizeof(Cat)
  │ align                │  → alignof(Cat)
  │ name()               │  → Cat::name
  │ sound()              │  → Cat::sound
  │ info()               │  → Animal::info (default impl)
  └──────────────────────┘
```

### 7.2 Object Safety

```rust
// Object-safe trait (can be used as dyn Trait)
trait Drawable {
    fn draw(&self);
    fn bounding_box(&self) -> (f64, f64, f64, f64);
}

// Non-object-safe trait (cannot be used as dyn Trait)
trait NotObjectSafe {
    fn create() -> Self;           // associated function returning Self
    fn compare(&self, other: &Self);  // takes Self as a parameter
    fn generic_method<T>(&self, t: T);  // generic method
}

// Conditions for object safety:
// 1. Does not require Self: Sized
// 2. Methods do not use Self in their return type (except with where Self: Sized guard)
// 3. No generic type parameters
// 4. No associated constants

// Technique to make a trait partially object-safe
trait Clonable: Clone {
    fn clone_box(&self) -> Box<dyn Clonable>;
}

impl<T: Clone + Clonable + 'static> Clonable for T {
    fn clone_box(&self) -> Box<dyn Clonable> {
        Box::new(self.clone())
    }
}
```

---

## 8. Advanced Trait Patterns

### 8.1 Associated Types

```rust
// Iterator definition using associated types
trait MyIterator {
    type Item;  // associated type

    fn next(&mut self) -> Option<Self::Item>;
}

struct Counter {
    count: u32,
    max: u32,
}

impl MyIterator for Counter {
    type Item = u32;  // concretize the associated type

    fn next(&mut self) -> Option<Self::Item> {
        if self.count < self.max {
            self.count += 1;
            Some(self.count)
        } else {
            None
        }
    }
}

// Comparison: associated types vs generics
// Associated types: only one implementation per type
// Generics: multiple implementations per type are possible

// Generics version (multiple conversion targets can be defined)
trait ConvertTo<T> {
    fn convert(&self) -> T;
}

struct Celsius(f64);

impl ConvertTo<f64> for Celsius {
    fn convert(&self) -> f64 { self.0 }
}

impl ConvertTo<String> for Celsius {
    fn convert(&self) -> String { format!("{}°C", self.0) }
}
```

### 8.2 Supertraits

```rust
use std::fmt;

// A trait that requires Display (supertrait)
trait Printable: fmt::Display + fmt::Debug {
    fn print(&self) {
        println!("Display: {}", self);
    }

    fn debug_print(&self) {
        println!("Debug: {:?}", self);
    }

    fn pretty_print(&self) {
        println!("===== {} =====", self);
    }
}

#[derive(Debug)]
struct Report {
    title: String,
    content: String,
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.title, self.content)
    }
}

// Since Display + Debug are implemented, Printable can be implemented
impl Printable for Report {}

fn main() {
    let report = Report {
        title: "Monthly Report".to_string(),
        content: "Sales increased by 10% from last month".to_string(),
    };
    report.print();
    report.debug_print();
    report.pretty_print();
}
```

### 8.3 Blanket Implementations

```rust
// Blanket implementation: implement for all types that satisfy a condition at once
trait Greet {
    fn greet(&self) -> String;
}

// Implement Greet for all types that implement Display
impl<T: std::fmt::Display> Greet for T {
    fn greet(&self) -> String {
        format!("Hello, {}!", self)
    }
}

fn main() {
    println!("{}", "Taro".greet());     // Hello, Taro!
    println!("{}", 42.greet());          // Hello, 42!
    println!("{}", 3.14f64.greet());     // Hello, 3.14!
}
```

### 8.4 Operator Overloading

```rust
use std::ops::{Add, Mul, Neg};

#[derive(Debug, Clone, Copy, PartialEq)]
struct Vector2D {
    x: f64,
    y: f64,
}

impl Vector2D {
    fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    fn magnitude(&self) -> f64 {
        (self.x.powi(2) + self.y.powi(2)).sqrt()
    }

    fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y
    }
}

// + operator
impl Add for Vector2D {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

// * operator (scalar multiplication)
impl Mul<f64> for Vector2D {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self::Output {
        Self {
            x: self.x * scalar,
            y: self.y * scalar,
        }
    }
}

// - operator (sign inversion)
impl Neg for Vector2D {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            x: -self.x,
            y: -self.y,
        }
    }
}

// Display
impl std::fmt::Display for Vector2D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

fn main() {
    let v1 = Vector2D::new(3.0, 4.0);
    let v2 = Vector2D::new(1.0, 2.0);

    println!("v1 + v2 = {}", v1 + v2);       // (4, 6)
    println!("v1 * 2 = {}", v1 * 2.0);        // (6, 8)
    println!("-v1 = {}", -v1);                 // (-3, -4)
    println!("|v1| = {}", v1.magnitude());     // 5
    println!("v1 . v2 = {}", v1.dot(&v2));     // 11
}
```

---

## 9. Comparison Tables

### 9.1 struct vs enum

| Property | struct | enum |
|------|--------|------|
| Algebraic type | Product type | Sum type |
| Fields | Holds all fields simultaneously | Only one of the variants |
| Pattern matching | Destructuring assignment | Exhaustive branching with match |
| Memory | Sum of all fields | Largest variant + tag |
| Use case | Grouping of data | Representing states/choices |
| Type safety | Guaranteed by field types | Guaranteed by exhaustive variants |

### 9.2 Static Dispatch vs Dynamic Dispatch

| Property | Static (impl Trait / generics) | Dynamic (dyn Trait) |
|------|-------------------------------|------------------|
| Mechanism | Monomorphization | vtable (virtual function table) |
| Execution speed | Fast (inlining possible) | Some overhead |
| Binary size | Tends to grow large | Small |
| Type erasure | None (concrete types determined at compile time) | Yes |
| Usage | `fn f(x: impl Trait)` | `fn f(x: &dyn Trait)` |
| Object safety | Not required | Required |
| Collections | Single type only | Different types can be mixed |
| Compile time | Code generated per type (can be slow) | Code sharing (fast) |

### 9.3 Trait-Related Comparison

| Pattern | Notation | Purpose |
|----------|------|------|
| Trait bound | `T: Clone + Debug` | Constraints on generic functions |
| where clause | `where T: Clone` | Make complex bounds more readable |
| impl Trait (parameter) | `item: &impl Summary` | Concise trait bound |
| impl Trait (return) | `-> impl Summary` | Hide concrete types |
| dyn Trait | `&dyn Summary` | Dynamic dispatch |
| Box<dyn Trait> | `Box<dyn Summary>` | Trait object on the heap |
| Associated type | `type Item = u32;` | One associated type per type |
| derive | `#[derive(Debug)]` | Automatic implementation of standard traits |

---

## 10. Anti-patterns

### Anti-pattern 1: Trying to put &str in a String field

```rust
// BAD: Requires lifetimes and becomes complicated
// struct User<'a> {
//     name: &'a str,  // lifetime annotations propagate, causing complexity
// }

// GOOD: Use owned types (recommended in most cases)
struct User {
    name: String,  // The struct owns its own data
}
```

### Anti-pattern 2: Unnecessary Use of Trait Objects

```rust
// BAD: Dynamic dispatch when generics would suffice
fn process(items: &[Box<dyn Summary>]) {
    for item in items {
        println!("{}", item.summarize());
    }
}

// GOOD: Static dispatch when types are uniform
fn process_good<T: Summary>(items: &[T]) {
    for item in items {
        println!("{}", item.summarize());
    }
}
// Note: dyn Trait is the right answer when mixing different types
```

### Anti-pattern 3: Unnecessarily Complex Trait Bounds

```rust
// BAD: Adding trait bounds that are not used
fn print_item<T: Display + Debug + Clone + Send + Sync>(item: &T) {
    println!("{}", item);  // Only Display is used
}

// GOOD: Minimal necessary trait bounds
fn print_item_good<T: Display>(item: &T) {
    println!("{}", item);
}
```

### Anti-pattern 4: Excessive Use of enum

```rust
// BAD: Every match expression must be modified each time a state is added
enum Shape {
    Circle(f64),
    Rectangle(f64, f64),
    Triangle(f64, f64),
    // Pentagon, Hexagon, ... continues to grow
}

fn area(shape: &Shape) -> f64 {
    match shape {
        Shape::Circle(r) => std::f64::consts::PI * r * r,
        Shape::Rectangle(w, h) => w * h,
        Shape::Triangle(b, h) => 0.5 * b * h,
        // Must be modified each time a new variant is added
    }
}

// GOOD: Using traits gives a design that is open to extension
trait ShapeTrait {
    fn area(&self) -> f64;
}

// New shapes can be added simply with new structs + impls
struct Pentagon { side: f64 }
impl ShapeTrait for Pentagon {
    fn area(&self) -> f64 {
        // Area of a regular pentagon
        0.25 * (5.0f64).sqrt() * (5.0 + 2.0 * (5.0f64).sqrt()) * self.side.powi(2)
    }
}
```


---

## Practical Exercises

### Exercise 1: Basic Implementation

Implement code that satisfies the following requirements.

**Requirements:**
- Validate input data
- Implement appropriate error handling
- Also write test code

```python
# Exercise 1: Template for basic implementation
class Exercise1:
    """Exercise for basic implementation patterns"""

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
        """Get processing results"""
        return {
            'count': len(self.data),
            'data': self.data
        }

# Test
def test_exercise1():
    ex = Exercise1()
    assert ex.process(1) == [1]
    assert ex.process(2) == [1, 2]
    assert ex.get_results()['count'] == 2

    try:
        ex.process(None)
        assert False, "An exception should have been raised"
    except ValueError:
        pass

    print("All tests passed!")

test_exercise1()
```

### Exercise 2: Advanced Patterns

Extend the basic implementation to add the following features.

```python
# Exercise 2: Advanced patterns
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

# Test
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

    print(f"Inefficient version: {slow_time:.4f} sec")
    print(f"Efficient version:   {fast_time:.6f} sec")
    print(f"Speedup ratio: {slow_time/fast_time:.0f}x")

benchmark()
```

**Key points:**
- Be aware of algorithmic complexity
- Choose appropriate data structures
- Measure the effect with benchmarks

---

## Troubleshooting

### Common Errors and Solutions

| Error | Cause | Solution |
|--------|------|--------|
| Initialization error | Defective configuration file | Check the path and format of the configuration file |
| Timeout | Network latency / resource shortage | Adjust timeout values, add retry logic |
| Out of memory | Increase in data volume | Introduce batch processing, implement pagination |
| Permission error | Insufficient access permissions | Check execution user permissions, review settings |
| Data inconsistency | Concurrent processing conflicts | Introduce locking mechanism, transaction management |

### Debugging Procedure

1. **Check the error message**: Read the stack trace and identify the location of the error
2. **Establish reproduction steps**: Reproduce the error with minimal code
3. **Formulate hypotheses**: List possible causes
4. **Stepwise verification**: Verify hypotheses using log output or a debugger
5. **Fix and regression testing**: After fixing, also run tests for related areas

```python
# Debugging utility
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
            logger.debug(f"Return value: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"Exception raised: {func.__name__}: {e}")
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

### Diagnosing Performance Problems

Diagnostic procedure when performance problems occur:

1. **Identify bottlenecks**: Measure with profiling tools
2. **Check memory usage**: Check for memory leaks
3. **Check I/O wait**: Check the status of disk and network I/O
4. **Check the number of concurrent connections**: Check the state of connection pools

| Type of problem | Diagnostic tool | Countermeasure |
|-----------|-----------|------|
| CPU load | cProfile, py-spy | Algorithm improvement, parallelization |
| Memory leak | tracemalloc, objgraph | Proper release of references |
| I/O bottleneck | strace, iostat | Asynchronous I/O, caching |
| DB latency | EXPLAIN, slow query log | Indexing, query optimization |
---

## 11. FAQ

### Q1: What is the criterion for choosing between String and &str for struct fields?

**A:** As a rule, use owned types (String) for structs. The struct can manage data independently and avoid the propagation of lifetimes. Only consider `&str` + lifetimes when performance is important and the struct is short-lived (such as intermediate results in a parser).

### Q2: How do you choose between `impl Trait` and `dyn Trait`?

**A:**
- **`impl Trait`**: When the concrete type is determined at compile time. Fast and type-safe.
- **`dyn Trait`**: When you need to handle different types at runtime. Such as when you want to put different types into a collection like `Vec<Box<dyn Trait>>`.

### Q3: When implementing a trait, what is the orphan rule?

**A:** You cannot implement a trait defined in another crate for a type defined in another crate. At least either the type or the trait must be defined in your own crate. This prevents implementation conflicts.

```rust
// OK: Implement your own trait for an external type
impl MyTrait for Vec<i32> { ... }

// OK: Implement an external trait for your own type
impl Display for MyStruct { ... }

// NG: Implement an external trait for an external type
// impl Display for Vec<i32> { ... }  // compile error
```

### Q4: What is the difference between associated types and generics?

**A:**
- **Associated types**: Only one implementation per type is possible. The Item of Iterator is a representative example.
- **Generics**: Multiple implementations per type are possible. From<T> is a representative example.

```rust
// Associated type: Iterator for Vec<i32> is Item=&i32 only
impl Iterator for MyIter {
    type Item = u32;  // fixed
    fn next(&mut self) -> Option<u32> { ... }
}

// Generics: multiple From implementations on the same type are possible
impl From<String> for MyType { ... }
impl From<i32> for MyType { ... }
```

### Q5: What is the list of derivable traits?

**A:** In the standard library, the following traits can be derived:
- `Debug`, `Clone`, `Copy`
- `PartialEq`, `Eq`
- `PartialOrd`, `Ord`
- `Hash`, `Default`

In external crates, traits such as `serde::Serialize`, `serde::Deserialize`, and `thiserror::Error` can also be derived.

---


## FAQ

### Q1: What is the most important point when learning this topic?

Gaining practical experience is the most important thing. Beyond theory, writing code yourself and confirming its behavior deepens understanding.

### Q2: What mistakes do beginners often make?

Skipping the basics and moving on to applications. We recommend thoroughly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this used in practice?

Knowledge of this topic is frequently utilized in everyday development work. It is especially important during code reviews and architectural design.

---

## 12. Summary

| Concept | Key points |
|------|------|
| struct | Three kinds: named, tuple, and unit. Product type |
| enum | Sum type with variants. Branched via pattern matching |
| impl | Block for defining methods and associated functions |
| trait | Interface definition. Default implementations are also possible |
| Generics | Write generic code with type parameters |
| Trait bound | Add constraints to generics (`T: Clone + Debug`) |
| derive | Automatic implementation of standard traits |
| Static/Dynamic dispatch | Monomorphization vs vtable. Choose according to use case |
| Associated type | Type parameter defined within a trait |
| Operator overloading | Implement traits from std::ops |
| Blanket impl | Bulk implementation for all types satisfying a condition |

---

## Recommended Next Reading

- [03-error-handling.md](03-error-handling.md) -- Error handling using Result/Option
- [04-collections-iterators.md](04-collections-iterators.md) -- Collections and iterators
- [../01-advanced/00-lifetimes.md](../01-advanced/00-lifetimes.md) -- Detailed explanation of lifetimes

---

## References

1. **The Rust Programming Language - Ch.5 Structs, Ch.6 Enums, Ch.10 Generics/Traits** -- https://doc.rust-lang.org/book/
2. **Rust by Example - Custom Types** -- https://doc.rust-lang.org/rust-by-example/custom_types.html
3. **Rust API Guidelines - Type Safety** -- https://rust-lang.github.io/api-guidelines/type-safety.html
4. **The Rustonomicon - Trait Objects** -- https://doc.rust-lang.org/nomicon/exotic-sizes.html
5. **Rust Design Patterns** -- https://rust-unofficial.github.io/patterns/
