# Ownership and Borrowing -- Rust's Most Revolutionary Memory Management Paradigm

> Ownership and Borrowing constitute Rust's unique memory management model, guaranteeing memory safety and freedom from data races at compile time without a garbage collector.

---

## What You Will Learn in This Chapter

1. **The three rules of ownership** -- Understand how each value has a single owner and is released when it goes out of scope
2. **Move and Copy** -- Master the difference between moving and duplicating values, and how to choose between the Copy/Clone traits
3. **Borrowing and lifetime basics** -- Learn the rules for immutable and mutable references, and an introduction to lifetimes
4. **Practical patterns** -- Acquire function and struct design patterns that leverage ownership


## Prerequisites

Reading the following before this guide will deepen your understanding:

- Basic programming knowledge
- Understanding of related fundamental concepts
- The content of [Rust Overview -- A Systems Programming Language Fusing Safety, Performance, and Ownership](./00-rust-overview.md)

---

## 1. Basic Rules of Ownership

### 1.1 The Three Rules

```
┌──────────────────────────────────────────────────┐
│         The Three Rules of Ownership              │
├──────────────────────────────────────────────────┤
│ 1. Each value has a variable called its "owner"  │
│ 2. There can only be one owner at a time         │
│ 3. When the owner goes out of scope, the value   │
│    is dropped                                    │
└──────────────────────────────────────────────────┘
```

These three rules form the foundation of Rust's memory management. In C/C++, the programmer manages memory manually; in Java/Python, a garbage collector manages it automatically. Rust takes a third path, achieving zero-runtime-cost memory management by tracking ownership at compile time.

### Example 1: Ownership and Scope

```rust
fn main() {
    {
        let s = String::from("hello"); // s comes into scope
        println!("{}", s);             // s is valid
    }                                  // s goes out of scope -> drop() is called
    // println!("{}", s);              // compile error: s does not exist
}
```

When the variable `s` goes out of the curly-brace scope, Rust automatically calls the `drop` function to free the memory. This is similar to C++'s RAII (Resource Acquisition Is Initialization) pattern, but Rust's ownership concept structurally eliminates double-free and dangling pointers.

### Example 2: Move Semantics

```rust
fn main() {
    let s1 = String::from("hello");
    let s2 = s1;                    // move: s1 -> s2
    // println!("{}", s1);          // error: s1 has been invalidated
    println!("{}", s2);             // OK
}
```

Because the `String` type holds its data on the heap, a "move" occurs upon assignment. A move is the transfer of ownership, and the original variable is invalidated. This prevents the situation where two variables own the same heap region (the cause of double-free).

### 1.2 Diagram of a Move

```
  Before move:                  After move:
  s1                            s1 (invalid)
  ┌──────────┐                  ┌──────────┐
  │ ptr ─────────┐              │ (invalid)│
  │ len: 5   │   │              └──────────┘
  │ cap: 5   │   │
  └──────────┘   │              s2
                 │              ┌──────────┐
                 │              │ ptr ─────────┐
                 │              │ len: 5   │   │
                 │              │ cap: 5   │   │
                 ▼              └──────────┘   │
  ┌──────────────┐                             │
  │ h e l l o    │<────────────────────────────┘
  └──────────────┘
  Only one copy of the data exists on the heap (it is not copied)
```

### 1.3 Situations Where Moves Occur

Moves occur implicitly in many situations. Understanding which operations cause a move is crucial in Rust programming.

```rust
fn main() {
    let s = String::from("hello");

    // (1) Move on variable binding
    let s2 = s;
    // s is invalid

    // (2) Move when passing as a function argument
    let s3 = String::from("world");
    takes_string(s3);
    // s3 is invalid

    // (3) Move on function return value
    let s4 = gives_string();
    // s4 receives ownership

    // (4) Move when inserting into a collection
    let s5 = String::from("item");
    let mut v = Vec::new();
    v.push(s5);
    // s5 is invalid (Vec owns it)

    // (5) Move via pattern matching
    let opt = Some(String::from("data"));
    if let Some(inner) = opt {
        println!("{}", inner);
    }
    // opt is invalid (moved into inner)

    // (6) Move when constructing a struct
    let name = String::from("Taro");
    let user = User { name };   // name is invalid
    println!("{}", user.name);   // OK: accessed as user.name
}

fn takes_string(s: String) {
    println!("received: {}", s);
    // s is dropped at the end of this function
}

fn gives_string() -> String {
    let s = String::from("a new string");
    s // returns ownership to the caller
}

struct User {
    name: String,
}
```

### 1.4 The Drop Trait and RAII

In Rust, when an owner goes out of scope, the `drop` method of the `Drop` trait is automatically called. This can be leveraged to automatically release resources such as file handles, network connections, and locks.

```rust
struct DatabaseConnection {
    url: String,
    connected: bool,
}

impl DatabaseConnection {
    fn new(url: &str) -> Self {
        println!("opening connection: {}", url);
        DatabaseConnection {
            url: url.to_string(),
            connected: true,
        }
    }

    fn query(&self, sql: &str) -> Vec<String> {
        println!("executing query: {}", sql);
        vec!["result1".to_string(), "result2".to_string()]
    }
}

impl Drop for DatabaseConnection {
    fn drop(&mut self) {
        println!("closing connection: {}", self.url);
        self.connected = false;
    }
}

fn main() {
    {
        let conn = DatabaseConnection::new("postgres://localhost/mydb");
        let results = conn.query("SELECT * FROM users");
        println!("results: {:?}", results);
    } // conn goes out of scope -> drop() is automatically called -> connection closed

    println!("the connection was automatically closed");
}
```

### 1.5 Stack Data vs. Heap Data

```
┌────────────────────────────────────────────────────────────┐
│         Memory Layout and the Move Relationship             │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Stack only (Copy types)       Uses heap (non-Copy types)  │
│  ┌─────┐    copy   ┌─────┐    ┌─────┐  move   ┌─────┐    │
│  │  42 │ ───────>  │  42 │    │ ptr │ ─────>  │ ptr │    │
│  └─────┘           └─────┘    │ len │         │ len │    │
│  i32: both valid              │ cap │         │ cap │    │
│                               └──┬──┘         └──┬──┘    │
│                                  │ (invalid)     │        │
│                                  └──────┐    ┌───┘        │
│                                         ▼    ▼            │
│                                    ┌──────────┐           │
│                                    │heap data │           │
│                                    └──────────┘           │
│                                    Only one pointer valid │
└────────────────────────────────────────────────────────────┘
```

---

## 2. Copy and Clone

### Example 3: The Copy Trait (Stack Values)

```rust
fn main() {
    let x: i32 = 42;
    let y = x;          // copy (i32 implements the Copy trait)
    println!("x={}, y={}", x, y); // both are valid!

    // A tuple is also Copy if all its elements are Copy
    let point = (3, 4);
    let point2 = point;
    println!("point={:?}, point2={:?}", point, point2); // both valid

    // An array is also Copy if its elements are Copy
    let arr = [1, 2, 3, 4, 5];
    let arr2 = arr;
    println!("arr={:?}, arr2={:?}", arr, arr2); // both valid

    // References are also Copy
    let s = String::from("hello");
    let r1 = &s;
    let r2 = r1;  // copies the reference (the String itself is not copied)
    println!("r1={}, r2={}", r1, r2); // both valid
}
```

### Example 4: The Clone Trait (Explicit Deep Copy)

```rust
fn main() {
    let s1 = String::from("hello");
    let s2 = s1.clone();           // explicitly copies the heap data as well
    println!("s1={}, s2={}", s1, s2); // both valid

    // Cloning a Vec
    let v1 = vec![1, 2, 3, 4, 5];
    let v2 = v1.clone();
    println!("v1={:?}, v2={:?}", v1, v2);

    // Cloning a nested data structure
    let nested = vec![
        vec![1, 2, 3],
        vec![4, 5, 6],
    ];
    let nested_clone = nested.clone(); // all data is deep-copied
    println!("nested={:?}", nested);
    println!("nested_clone={:?}", nested_clone);
}
```

### Diagram of Clone

```
  Before clone():
  s1                     heap
  ┌──────────┐          ┌───────────┐
  │ ptr ────────────────>│ h e l l o │
  │ len: 5   │          └───────────┘
  │ cap: 5   │
  └──────────┘

  After clone():
  s1                     heap
  ┌──────────┐          ┌───────────┐
  │ ptr ────────────────>│ h e l l o │  <- original data
  │ len: 5   │          └───────────┘
  │ cap: 5   │
  └──────────┘
                         ┌───────────┐
  s2                     │ h e l l o │  <- new copy
  ┌──────────┐          └───────────┘
  │ ptr ────────────────>│
  │ len: 5   │
  │ cap: 5   │
  └──────────┘
  Two independent heap regions exist
```

### Types That Implement Copy and Types That Do Not

```
┌─────────────────────────────┬────────────────────────────┐
│   Types that are Copy       │   Types that are not Copy  │
├─────────────────────────────┼────────────────────────────┤
│ i8, i16, i32, i64, i128    │ String                     │
│ u8, u16, u32, u64, u128    │ Vec<T>                     │
│ f32, f64                    │ Box<T>                     │
│ bool                        │ HashMap<K, V>              │
│ char                        │ HashSet<T>                 │
│ isize, usize               │ File, TcpStream            │
│ (i32, bool) -- all Copy    │ Rc<T>, Arc<T>              │
│ [i32; 5] -- fixed array    │ MutexGuard<T>              │
│ &T -- immutable reference   │ &mut T -- mutable ref      │
│ fn pointer                  │ closures (depends on capture)│
│ *const T, *mut T -- raw ptr│ dyn Trait                  │
└─────────────────────────────┴────────────────────────────┘
```

### 2.1 Custom Implementation of the Copy Trait

```rust
// To derive Copy, every field must also be Copy
#[derive(Debug, Clone, Copy)]
struct Point {
    x: f64,
    y: f64,
}

// A struct that cannot be Copy
// #[derive(Clone, Copy)]  // compile error! String is not Copy
#[derive(Debug, Clone)]
struct NamedPoint {
    name: String,
    x: f64,
    y: f64,
}

fn main() {
    let p1 = Point { x: 1.0, y: 2.0 };
    let p2 = p1;           // Copy
    println!("p1={:?}", p1); // OK: p1 is still valid

    let np1 = NamedPoint {
        name: "origin".to_string(),
        x: 0.0,
        y: 0.0,
    };
    let np2 = np1.clone();  // Clone is required
    // let np3 = np1;       // move! np1 becomes invalid
    println!("np2={:?}", np2);
}
```

### 2.2 Relationship Between Copy and Clone

```
┌─────────────────────────────────────────────────────┐
│           Relationship Between Copy and Clone        │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Copy is a sub-trait of Clone                        │
│  (Implementing Copy also requires Clone)             │
│                                                      │
│  pub trait Copy: Clone { }                           │
│                                                      │
│  Meaning of Copy:                                    │
│  - A type safe for bitwise copying                   │
│  - Implicitly copied (on assignment, function args)  │
│  - Only types that have no heap allocation           │
│                                                      │
│  Meaning of Clone:                                   │
│  - A type that provides explicit deep copying        │
│  - Requires a .clone() call                          │
│  - Implementable for any type (heap allocation OK)   │
│                                                      │
│  ┌────────────────┐                                  │
│  │    Clone       │                                  │
│  │  ┌──────────┐  │                                  │
│  │  │   Copy   │  │                                  │
│  │  │ i32,bool │  │                                  │
│  │  │ f64,char │  │                                  │
│  │  └──────────┘  │                                  │
│  │  String, Vec   │                                  │
│  │  HashMap       │                                  │
│  └────────────────┘                                  │
└─────────────────────────────────────────────────────┘
```

---

## 3. Borrowing (References)

### 3.1 Borrowing Rules

```
┌──────────────────────────────────────────────────┐
│              Borrowing Rules                      │
├──────────────────────────────────────────────────┤
│ 1. You can have multiple immutable refs (&T)      │
│ 2. You can have only one mutable ref (&mut T)     │
│ 3. Immutable and mutable refs cannot coexist      │
│ 4. References must always be valid                │
└──────────────────────────────────────────────────┘
```

These rules are the core mechanism by which Rust prevents data races at compile time. A data race occurs when the following three conditions are met simultaneously:

1. Two or more pointers access the same data simultaneously
2. At least one pointer writes to the data
3. There is no mechanism to synchronize access to the data

By eliminating the combination of conditions 1 and 2 at compile time, Rust's borrowing rules make data races structurally impossible.

### Example 5: Immutable References (Shared References)

```rust
fn calculate_length(s: &String) -> usize {
    s.len()
    // s is dropped here, but since it does not own the data, the data is not freed
}

fn main() {
    let s = String::from("hello");
    let len = calculate_length(&s);  // borrow (passing a reference)
    println!("the length of '{}' is {}", s, len); // s is still valid
}
```

You create a reference using the `&` symbol. Because a reference does not own the value, the original value is not freed when the reference goes out of scope.

### Example 6: Mutable References

```rust
fn append_world(s: &mut String) {
    s.push_str(", world!");
}

fn main() {
    let mut s = String::from("hello");
    append_world(&mut s);
    println!("{}", s); // "hello, world!"
}
```

Using a mutable reference `&mut`, you can modify the value at the borrow site. However, only one mutable reference can exist at a time.

### Example 7: Borrowing Rule Violations and NLL

```rust
fn main() {
    let mut s = String::from("hello");

    let r1 = &s;        // OK: immutable reference 1
    let r2 = &s;        // OK: immutable reference 2
    println!("{}, {}", r1, r2);
    // r1 and r2 are not used after this point (NLL)

    let r3 = &mut s;    // OK: r1, r2's lifetimes have already ended
    println!("{}", r3);
}
```

NLL (Non-Lexical Lifetimes) is a feature introduced in Rust 2018 Edition that ends a reference's lifetime at "the point of last use" rather than at the lexical scope (the curly-brace boundary). With this, the code above compiles correctly.

### Example 8: Borrowing Rule Violation (Compile Error)

```rust
fn main() {
    let mut s = String::from("hello");

    let r1 = &s;          // immutable reference
    let r2 = &mut s;      // error! cannot create a mutable ref while an immutable ref is alive
    println!("{}", r1);    // r1 is still being used
}
```

```
error[E0502]: cannot borrow `s` as mutable because it is also borrowed as immutable
 --> src/main.rs:4:14
  |
3 |     let r1 = &s;
  |              -- immutable borrow occurs here
4 |     let r2 = &mut s;
  |              ^^^^^^ mutable borrow occurs here
5 |     println!("{}", r1);
  |                    -- immutable borrow later used here
```

### 3.2 Reference Lifecycle Diagram

```
    Time axis ->
    ├───────────┤
    │ r1 = &s   │   (immutable ref: alive)
    ├───────────┤
    │ r2 = &s   │   (immutable ref: alive)
    ├───────────┤
    │ println!  │   (last use of r1, r2 = ended by NLL)
    │           │
    │ r3 = &mut │   (mutable ref: alive from here -> OK)
    ├───────────┤
    │ println!  │   (last use of r3)
    └───────────┘

    NLL (Non-Lexical Lifetimes):
    A reference's lifetime ends at "the point of last use"
```

### 3.3 Why the Exclusivity of Mutable References Matters

```rust
// If two simultaneous mutable references were allowed...
// (the following is a hypothetical dangerous example: actual Rust gives a compile error)
fn hypothetical_danger() {
    let mut data = vec![1, 2, 3];

    // Suppose two mutable references could exist at once:
    // let r1 = &mut data;  // mutable ref 1
    // let r2 = &mut data;  // mutable ref 2 (actually an error)

    // r1.push(4);          // Vec may trigger reallocation
    // println!("{}", r2[0]); // r2 would point to invalid memory!
    //                        // -> use-after-free vulnerability
}

// Rust prevents this at compile time
fn safe_version() {
    let mut data = vec![1, 2, 3];

    // Only one mutable reference
    let r1 = &mut data;
    r1.push(4);
    // r1's lifetime ends

    // Acquire a new mutable reference
    let r2 = &mut data;
    println!("{}", r2[0]); // safe
}
```

### 3.4 Reborrowing

```rust
fn main() {
    let mut s = String::from("hello");
    let r = &mut s;

    // Reborrow: create an immutable reference from a mutable reference
    let r2 = &*r;  // reborrow (also happens implicitly)
    println!("{}", r2);

    // Reborrow: create a temporary mutable reference from a mutable reference
    modify(r);  // &mut String is reborrowed as &mut String
    println!("{}", r);
}

fn modify(s: &mut String) {
    s.push_str(", world!");
}
```

Reborrowing is the mechanism by which a temporary new reference is created from an existing reference. When you pass a `&mut` argument to a function, the original mutable reference is temporarily "frozen" and becomes usable again once the function returns.

---

## 4. Functions and Ownership

### Example 9: Moving and Returning Ownership

```rust
fn takes_ownership(s: String) -> String {
    println!("received: {}", s);
    s  // returns ownership
}

fn main() {
    let s1 = String::from("hello");
    let s2 = takes_ownership(s1); // s1 -> function -> s2
    // println!("{}", s1);        // error: s1 is invalid
    println!("{}", s2);           // OK
}
```

### Example 10: Best Practices for Function Design Using References

```rust
// Pattern 1: read only -> immutable reference &T
fn print_info(s: &str) {
    println!("string: {}, length: {}", s, s.len());
}

// Pattern 2: needs to mutate -> mutable reference &mut T
fn make_uppercase(s: &mut String) {
    *s = s.to_uppercase();
}

// Pattern 3: needs ownership -> T (pass by value)
fn into_bytes(s: String) -> Vec<u8> {
    s.into_bytes()  // consumes the String and returns Vec<u8>
}

// Pattern 4: take ownership conditionally -> Cow (Clone on Write)
use std::borrow::Cow;

fn ensure_uppercase(s: &str) -> Cow<'_, str> {
    if s.chars().all(|c| c.is_uppercase()) {
        Cow::Borrowed(s)        // no change needed: return the borrow as-is
    } else {
        Cow::Owned(s.to_uppercase()) // change needed: return a new String
    }
}

fn main() {
    let s = String::from("hello");

    // Pattern 1: immutable reference
    print_info(&s);
    println!("s is still usable: {}", s);

    // Pattern 2: mutable reference
    let mut s2 = String::from("hello");
    make_uppercase(&mut s2);
    println!("uppercase: {}", s2);

    // Pattern 3: move ownership
    let s3 = String::from("hello");
    let bytes = into_bytes(s3);
    // s3 is no longer usable
    println!("bytes: {:?}", bytes);

    // Pattern 4: Cow
    let result = ensure_uppercase("HELLO");
    println!("Cow: {}", result);
}
```

### Example 11: Efficient Borrowing via Slices

```rust
fn first_word(s: &str) -> &str {
    let bytes = s.as_bytes();
    for (i, &byte) in bytes.iter().enumerate() {
        if byte == b' ' {
            return &s[..i];
        }
    }
    s
}

fn longest_word(s: &str) -> &str {
    s.split_whitespace()
        .max_by_key(|word| word.len())
        .unwrap_or("")
}

fn main() {
    let sentence = String::from("hello world foo bar");
    let word = first_word(&sentence);
    println!("first word: {}", word); // "hello"

    let longest = longest_word(&sentence);
    println!("longest word: {}", longest); // "hello" or "world"

    // String literals can also be passed as slices
    let word2 = first_word("good morning");
    println!("first word: {}", word2); // "good"
}
```

### Example 12: Borrowing and Ownership in Structs

```rust
// A struct using owned types (most common)
struct OwnedUser {
    name: String,
    email: String,
}

// A struct using borrowed types (lifetime annotations required)
struct BorrowedUser<'a> {
    name: &'a str,
    email: &'a str,
}

// Examples of how to choose
fn create_owned_user(name: &str, email: &str) -> OwnedUser {
    OwnedUser {
        name: name.to_string(),
        email: email.to_string(),
    }
}

fn create_borrowed_user<'a>(name: &'a str, email: &'a str) -> BorrowedUser<'a> {
    BorrowedUser { name, email }
}

fn main() {
    // Owned: the struct owns its data, so there are no lifetime constraints
    let owned = create_owned_user("Tanaka", "tanaka@example.com");
    println!("{}: {}", owned.name, owned.email);

    // Borrowed: cannot outlive the original data
    let name = String::from("Suzuki");
    let email = String::from("suzuki@example.com");
    let borrowed = create_borrowed_user(&name, &email);
    println!("{}: {}", borrowed.name, borrowed.email);
    // name and email are valid here -> borrowed is valid as well
}
```

---

## 5. Lifetime Basics

### 5.1 Basics of Lifetime Annotations

A lifetime annotation is a mechanism that tells the compiler how long a reference is valid. The annotation itself does not change a reference's lifespan; it merely describes the relationships between multiple references to the compiler.

```rust
// No lifetime annotation (compile error)
// fn longest(x: &str, y: &str) -> &str {
//     if x.len() > y.len() { x } else { y }
// }

// With lifetime annotation
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}

fn main() {
    let string1 = String::from("a long string");
    let result;
    {
        let string2 = String::from("short");
        result = longest(string1.as_str(), string2.as_str());
        println!("longer: {}", result); // OK: string2 is still valid
    }
    // println!("{}", result); // error: string2's lifetime has ended
}
```

### 5.2 Lifetime Elision Rules

The Rust compiler has three lifetime elision rules; in many cases, you do not need to write explicit lifetime annotations.

```rust
// Rule 1: each input reference is assigned its own unique lifetime
fn first(s: &str) -> &str { &s[..1] }
// expanded: fn first<'a>(s: &'a str) -> &'a str

// Rule 2: if there is exactly one input reference, its lifetime applies to all outputs
fn first_word(s: &str) -> &str {
    s.split_whitespace().next().unwrap_or("")
}
// expanded: fn first_word<'a>(s: &'a str) -> &'a str

// Rule 3: in methods, the lifetime of &self applies to outputs
struct Parser {
    input: String,
}

impl Parser {
    fn first_token(&self) -> &str {
        self.input.split_whitespace().next().unwrap_or("")
    }
    // expanded: fn first_token<'a>(&'a self) -> &'a str
}

// When the rules cannot apply, explicit lifetime annotations are required
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}
```

### 5.3 The 'static Lifetime

```rust
// 'static lifetime: valid for the entire duration of the program's execution
fn get_greeting() -> &'static str {
    "Hello!"  // string literals are 'static
}

// Constants are also 'static
static GLOBAL_CONFIG: &str = "default config";

fn main() {
    let greeting = get_greeting();
    println!("{}", greeting);
    println!("{}", GLOBAL_CONFIG);

    // T: 'static can also mean "an owned type"
    // String satisfies 'static (because it contains no references)
    fn takes_owned<T: 'static>(value: T) {
        // T is a type that contains no references
        std::mem::drop(value);
    }

    takes_owned(String::from("hello")); // OK
    takes_owned(42i32);                  // OK
    // takes_owned(&String::from("hello")); // error: a temporary reference is not 'static
}
```

---

## 6. Advanced Ownership Patterns

### 6.1 Interior Mutability

```rust
use std::cell::{Cell, RefCell};

// Cell<T>: interior mutability for Copy types
struct Counter {
    count: Cell<u32>,
}

impl Counter {
    fn new() -> Self {
        Counter { count: Cell::new(0) }
    }

    fn increment(&self) {
        // Internal state can be changed even though &self (immutable reference)
        self.count.set(self.count.get() + 1);
    }

    fn get(&self) -> u32 {
        self.count.get()
    }
}

// RefCell<T>: interior mutability for arbitrary types (runtime-checked)
struct CachedValue {
    value: String,
    cache: RefCell<Option<String>>,
}

impl CachedValue {
    fn new(value: String) -> Self {
        CachedValue {
            value,
            cache: RefCell::new(None),
        }
    }

    fn get_computed(&self) -> String {
        // cache can be modified even with &self
        let mut cache = self.cache.borrow_mut();
        if cache.is_none() {
            println!("computing cache...");
            *cache = Some(format!("computed_{}", self.value));
        }
        cache.clone().unwrap()
    }
}

fn main() {
    let counter = Counter::new();
    counter.increment();
    counter.increment();
    counter.increment();
    println!("count: {}", counter.get()); // 3

    let cached = CachedValue::new("hello".to_string());
    println!("{}", cached.get_computed()); // "computing cache..." -> "computed_hello"
    println!("{}", cached.get_computed()); // cache hit -> "computed_hello"
}
```

### 6.2 Smart Pointers and Ownership

```rust
use std::rc::Rc;

// Box<T>: single ownership on the heap
fn box_example() {
    let b = Box::new(5);
    println!("Box: {}", b);

    // Recursive data structure
    enum List {
        Cons(i32, Box<List>),
        Nil,
    }

    let list = List::Cons(1,
        Box::new(List::Cons(2,
            Box::new(List::Cons(3,
                Box::new(List::Nil))))));
}

// Rc<T>: shared ownership via reference counting (single-threaded)
fn rc_example() {
    let a = Rc::new(String::from("shared data"));
    println!("reference count: {}", Rc::strong_count(&a)); // 1

    let b = Rc::clone(&a);  // increments the reference count (does not copy data)
    println!("reference count: {}", Rc::strong_count(&a)); // 2

    {
        let c = Rc::clone(&a);
        println!("reference count: {}", Rc::strong_count(&a)); // 3
    }
    // c is dropped
    println!("reference count: {}", Rc::strong_count(&a)); // 2

    println!("a={}, b={}", a, b);
}

fn main() {
    box_example();
    rc_example();
}
```

### 6.3 Ownership and Pattern Matching

```rust
enum Message {
    Text(String),
    Number(i32),
    Pair(String, String),
}

fn process_message(msg: Message) {
    match msg {
        // Ownership of msg is moved
        Message::Text(text) => {
            println!("text: {}", text);
            // owns text
        }
        Message::Number(n) => {
            println!("number: {}", n);
        }
        Message::Pair(a, b) => {
            println!("pair: {} / {}", a, b);
        }
    }
    // msg is fully moved and cannot be used
}

fn process_message_ref(msg: &Message) {
    match msg {
        // Pattern matching on a reference only borrows
        Message::Text(text) => {
            println!("text: {}", text);
            // text is &String
        }
        Message::Number(n) => {
            println!("number: {}", n);
        }
        Message::Pair(a, b) => {
            println!("pair: {} / {}", a, b);
        }
    }
    // msg is still usable
}

fn main() {
    let msg = Message::Text("hello".to_string());
    process_message_ref(&msg);
    process_message_ref(&msg);  // OK: a reference, so reusable

    let msg2 = Message::Pair("left".to_string(), "right".to_string());
    process_message(msg2);
    // process_message(msg2);  // error: already moved
}
```

---

## 7. Comparison Tables

### 7.1 Move vs. Copy vs. Clone

| Operation | Heap copy | Original value | Auto/explicit | Cost | Use case |
|------|------------|--------|-----------|--------|------|
| Move | None | Invalidated | Auto | O(1) | Transfer of ownership |
| Copy | N/A (stack only) | Valid | Auto | O(1) | Duplication of small values |
| Clone | Yes | Valid | Explicit (.clone()) | O(n) | When deep copy is required |
| Reference (&T) | None | Valid | Explicit (&) | O(1) | Read-only access |

### 7.2 Immutable References vs. Mutable References

| Property | `&T` (immutable ref) | `&mut T` (mutable ref) |
|------|-----------------|---------------------|
| Number that can coexist | Multiple | Only one |
| Data modification | Not allowed | Allowed |
| Alias | Shared reference | Exclusive reference |
| Send/Sync | Safe if T: Sync | Safe if T: Send |
| Coexistence with other references | Cannot coexist with &mut T | Cannot coexist with &T |
| Compiler optimization | Aliasing optimizations possible | Strong optimizations due to exclusivity |

### 7.3 Choosing Between Owned and Borrowed Types

| Situation | Recommended | Reason |
|------|------|------|
| Struct fields | String (owned) | Avoids the complexity of lifetimes |
| Function argument (read) | &str (borrowed) | Highly flexible |
| Function argument (modify) | &mut String | Caller can still use it |
| Function argument (consume) | String (owned) | When the value is consumed |
| Return value (a new value) | String (owned) | Cannot return refs to local variables |
| Return value (part of an argument) | &str + lifetime | Efficient |
| Short-lived temporary structs | &str + lifetime | Performance-focused |

---

## 8. Anti-Patterns

### Anti-Pattern 1: Cloning More Than Necessary

```rust
// BAD: cloning when a reference would suffice
fn print_length(s: String) {
    println!("length: {}", s.len());
}
fn bad_example() {
    let s = String::from("hello");
    print_length(s.clone()); // unnecessary clone
    print_length(s.clone()); // another unnecessary clone
    println!("{}", s);
}

// GOOD: use a reference
fn print_length_good(s: &str) {
    println!("length: {}", s.len());
}
fn good_example() {
    let s = String::from("hello");
    print_length_good(&s);
    print_length_good(&s);
    println!("{}", s);
}
```

### Anti-Pattern 2: Attempting a Dangling Reference

```rust
// BAD: trying to return a reference to a local variable
// fn dangle() -> &String {
//     let s = String::from("hello");
//     &s  // s is dropped at the end of this function -> dangling reference!
// }

// GOOD: return ownership instead
fn no_dangle() -> String {
    let s = String::from("hello");
    s  // moves and returns ownership
}
```

### Anti-Pattern 3: Unnecessary Mutable References

```rust
// BAD: using &mut even though no modification is performed
fn just_read(data: &mut Vec<i32>) -> i32 {
    data.iter().sum()
}

// GOOD: an immutable reference suffices for read-only access
fn just_read_good(data: &[i32]) -> i32 {
    data.iter().sum()
}
```

### Anti-Pattern 4: Overusing References in Struct Fields

```rust
// BAD: lifetimes propagate unnecessarily
struct Config<'a> {
    host: &'a str,
    port: u16,
    database: &'a str,
}

// A function returning such a struct becomes very complex
// fn load_config<'a>() -> Config<'a> { ... }  // <- lifetime management is difficult

// GOOD: use owned types
struct ConfigGood {
    host: String,
    port: u16,
    database: String,
}

fn load_config() -> ConfigGood {
    ConfigGood {
        host: "localhost".to_string(),
        port: 5432,
        database: "myapp".to_string(),
    }
}
```

### Anti-Pattern 5: Modifying a Collection While Iterating

```rust
fn main() {
    let mut v = vec![1, 2, 3, 4, 5];

    // BAD: trying to modify the collection during iteration
    // for item in &v {
    //     if *item > 3 {
    //         v.push(*item * 2);  // error! cannot mutably borrow while immutably borrowed
    //     }
    // }

    // GOOD: collect results into another collection, then append
    let additions: Vec<i32> = v.iter()
        .filter(|&&x| x > 3)
        .map(|&x| x * 2)
        .collect();
    v.extend(additions);
    println!("{:?}", v); // [1, 2, 3, 4, 5, 8, 10]

    // GOOD: use retain to drop elements that don't meet the condition
    let mut v2 = vec![1, 2, 3, 4, 5];
    v2.retain(|&x| x % 2 == 0);
    println!("{:?}", v2); // [2, 4]
}
```

---

## 9. Practical Examples: Designs That Leverage Ownership

### 9.1 The State Machine Pattern

```rust
// Use ownership to express state transitions at the type level
struct Idle;
struct Running {
    start_time: std::time::Instant,
}
struct Finished {
    duration: std::time::Duration,
}

struct Task<State> {
    name: String,
    state: State,
}

impl Task<Idle> {
    fn new(name: &str) -> Self {
        Task {
            name: name.to_string(),
            state: Idle,
        }
    }

    // Consumes self and returns a Task in a new state
    fn start(self) -> Task<Running> {
        println!("starting task '{}'", self.name);
        Task {
            name: self.name,
            state: Running {
                start_time: std::time::Instant::now(),
            },
        }
    }
}

impl Task<Running> {
    fn finish(self) -> Task<Finished> {
        let duration = self.state.start_time.elapsed();
        println!("task '{}' finished ({:?})", self.name, duration);
        Task {
            name: self.name,
            state: Finished { duration },
        }
    }
}

impl Task<Finished> {
    fn report(&self) {
        println!("report: '{}' completed in {:?}", self.name, self.state.duration);
    }
}

fn main() {
    let task = Task::new("data processing");
    // task.finish();  // compile error! cannot go directly from Idle to Finished
    let running = task.start();
    // task.start();   // compile error! task has already been moved
    let finished = running.finish();
    finished.report();
}
```

### 9.2 Resource Management via Ownership

```rust
use std::fs::File;
use std::io::{self, Write, BufWriter};

// A struct that owns the file
struct Logger {
    writer: BufWriter<File>,
    count: u64,
}

impl Logger {
    fn new(path: &str) -> io::Result<Self> {
        let file = File::create(path)?;
        Ok(Logger {
            writer: BufWriter::new(file),
            count: 0,
        })
    }

    fn log(&mut self, message: &str) -> io::Result<()> {
        self.count += 1;
        writeln!(self.writer, "[{}] {}", self.count, message)?;
        Ok(())
    }

    // Consume ownership to flush the file reliably
    fn close(mut self) -> io::Result<()> {
        self.writer.flush()?;
        println!("closed log file ({} entries)", self.count);
        Ok(())
        // self is dropped, and the File is automatically closed
    }
}

fn main() -> io::Result<()> {
    let mut logger = Logger::new("/tmp/app.log")?;
    logger.log("application started")?;
    logger.log("processing")?;
    logger.log("application terminating")?;
    logger.close()?;
    // logger cannot be used (consumed by close)
    // logger.log("one more");  // compile error!
    Ok(())
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
# Exercise 1: template for basic implementation
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
        """Main logic for data processing"""
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
        assert False, "an exception should have been raised"
    except ValueError:
        pass

    print("all tests passed!")

test_exercise1()
```

### Exercise 2: Applied Patterns

Extend the basic implementation with the following features.

```python
# Exercise 2: applied patterns
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """Exercise on applied patterns"""

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
    print("all advanced tests passed!")

test_advanced()
```

### Exercise 3: Performance Optimization

Improve the performance of the following code.

```python
# Exercise 3: performance optimization
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

    print(f"inefficient: {slow_time:.4f}s")
    print(f"efficient:   {fast_time:.6f}s")
    print(f"speedup:     {slow_time/fast_time:.0f}x")

benchmark()
```

**Points:**
- Be conscious of algorithmic complexity
- Choose appropriate data structures
- Measure improvements with benchmarks

---

## Troubleshooting

### Common Errors and Solutions

| Error | Cause | Solution |
|--------|------|--------|
| Initialization error | Misconfigured config file | Verify config file path and format |
| Timeout | Network latency / lack of resources | Adjust timeout values; add retry logic |
| Out of memory | Increasing data volume | Introduce batch processing; implement pagination |
| Permission error | Insufficient access rights | Check the running user's permissions; review settings |
| Data inconsistency | Concurrency conflicts | Introduce locking mechanisms; use transaction management |

### Debugging Steps

1. **Check the error message**: read the stack trace and identify the location of the issue
2. **Establish reproduction steps**: reproduce the error with minimal code
3. **Form a hypothesis**: list possible causes
4. **Verify incrementally**: validate the hypothesis with logging or a debugger
5. **Fix and regression-test**: after fixing, also run tests for related areas

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
    """Data processing (subject to debugging)"""
    if not items:
        raise ValueError("empty data")
    return [item * 2 for item in items]
```

### Diagnosing Performance Problems

Diagnostic procedure when performance issues occur:

1. **Identify the bottleneck**: measure with profiling tools
2. **Check memory usage**: check for memory leaks
3. **Check I/O wait**: examine disk and network I/O activity
4. **Check concurrent connections**: examine the connection pool state

| Type of issue | Diagnostic tool | Countermeasure |
|-----------|-----------|------|
| CPU load | cProfile, py-spy | Algorithmic improvements; parallelization |
| Memory leak | tracemalloc, objgraph | Properly release references |
| I/O bottleneck | strace, iostat | Asynchronous I/O; caching |
| DB latency | EXPLAIN, slow query log | Indexes; query optimization |

---

## Design Decision Guide

### Selection Criteria Matrix

The following summarizes criteria for technology selection.

| Criterion | When prioritized | When can be compromised |
|---------|------------|-------------|
| Performance | Real-time processing; large-scale data | Admin panels; batch processing |
| Maintainability | Long-term operation; team development | Prototypes; short-term projects |
| Scalability | Services expected to grow | Internal tools; fixed user base |
| Security | Personal data; financial data | Public data; internal use |
| Development speed | MVP; time-to-market | Quality-focused; mission-critical |

### Architecture Pattern Selection

```
┌─────────────────────────────────────────────────┐
│           Architecture Selection Flow           │
├─────────────────────────────────────────────────┤
│                                                 │
│  (1) Team size?                                 │
│    ├─ Small (1-5 people) -> monolith            │
│    └─ Large (10+ people) -> go to (2)           │
│                                                 │
│  (2) Deployment frequency?                      │
│    ├─ Once a week or less -> monolith + modules │
│    └─ Daily / multiple times -> go to (3)       │
│                                                 │
│  (3) Independence between teams?                │
│    ├─ High -> microservices                     │
│    └─ Moderate -> modular monolith              │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Trade-off Analysis

Technical decisions always involve trade-offs. Analyze them from the following perspectives:

**1. Short-term vs. long-term cost**
- A method that is fast in the short term may become technical debt in the long term
- Conversely, over-engineering has high short-term cost and can delay the project

**2. Consistency vs. flexibility**
- A unified technology stack has a low learning cost
- Adopting diverse technologies allows the right tool for the job, but increases operational cost

**3. Level of abstraction**
- High abstraction yields high reusability but can make debugging difficult
- Low abstraction is intuitive but tends to result in code duplication

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
        """Describe the decision"""
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
        """Output as Markdown"""
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

## 10. FAQ

### Q1: When does a move occur?

**A:** A move occurs in the following cases:
- `let y = x;` (for types that do not implement Copy)
- Passing a value to a function: `func(x)`
- Returning a value from a function: `return x`
- Inserting a value into a collection: `vec.push(x)`
- Extracting a value via pattern matching: `if let Some(v) = opt`
- Initializing a struct field: `Struct { field: x }`

Types that implement the Copy trait (such as i32, bool, f64) are copied rather than moved.

### Q2: What is the difference between `&str` and `&String`?

**A:** `&str` is a string slice -- a "fat pointer" that holds a reference to string data plus length information. `&String` is a reference to a `String`. By convention, function arguments use `&str`. Because `&String` is automatically dereferenced to `&str` (Deref coercion), `&str` is more general-purpose.

```rust
fn accepts_str(s: &str) {
    println!("{}", s);
}

fn main() {
    let owned = String::from("hello");
    let literal = "world";

    accepts_str(&owned);    // &String -> &str (Deref coercion)
    accepts_str(literal);   // &str as-is
    accepts_str(&owned[1..]); // a slice can also be passed
}
```

### Q3: Why can there only be one mutable reference at a time?

**A:** To prevent data races. A data race occurs when these three conditions are met:
1. Two or more pointers access the same data
2. At least one of them performs a write
3. There is no synchronization of access

By limiting mutable references to one, the combination of conditions 1 and 2 can be eliminated at compile time.

### Q4: How should I choose between Clone and Copy?

**A:**
- **Copy**: small values on the stack (i32, f64, bool, etc.); duplicated implicitly
- **Clone**: types containing heap data (String, Vec, etc.); requires an explicit `.clone()` call

To implement Copy on your own type, every field must also be Copy. Copy implies "cheap copying"; Clone implies "arbitrary cost."

### Q5: When do I need to write lifetimes explicitly?

**A:** When the compiler's elision rules cannot infer them. Mainly:
- In functions with multiple input references whose return value contains a reference
- In structs containing reference fields
- In trait implementations where lifetime relationships among references are complex

```rust
// Inferable via elision rules -> no annotation needed
fn first(s: &str) -> &str { &s[..1] }

// Multiple input references -> annotation required
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}

// Struct with a reference field -> annotation required
struct Excerpt<'a> {
    text: &'a str,
}
```

### Q6: When should I use RefCell<T>?

**A:** When you want borrowing rules to be checked at runtime instead of at compile time. Typical use cases:
- When you want to modify internal state through an immutable reference (the interior mutability pattern)
- When modifying the internal state of a trait object
- When the compiler cannot prove safety but the programmer is sure that it is safe

That said, since RefCell can panic at runtime, you should use ordinary borrowing whenever possible.

---


## FAQ

### Q1: What is the most important point to keep in mind when learning this topic?

Gaining hands-on experience is the most important. Beyond theory, your understanding deepens when you actually write code and verify behavior.

### Q2: What mistakes do beginners commonly make?

Skipping the fundamentals and rushing to advanced topics. We recommend firmly grasping the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this used in real-world practice?

Knowledge of this topic is frequently applied in everyday development work. It becomes especially important during code reviews and architecture design.

---

## 11. Summary

| Concept | Key point |
|------|------|
| Ownership | Each value has a single owner; auto drop at end of scope |
| Move | Ownership is transferred on assignment / function call; original variable is invalidated |
| Copy | Small values on the stack are copied implicitly (i32, bool, etc.) |
| Clone | Explicit deep copy of heap data |
| Immutable reference (&T) | Multiple allowed simultaneously; data cannot be modified |
| Mutable reference (&mut T) | Only one at a time; data can be modified |
| NLL | A reference's lifetime ends at its last point of use |
| Slice | A reference to part of data; does not own |
| Lifetime | Annotation that tells the compiler how long a reference is valid |
| Drop | Destructor automatically called at end of scope |
| Interior mutability | Cell/RefCell enable modification through an immutable reference |
| Smart pointers | Box/Rc/Arc extend ownership patterns |

---

## Recommended Next Reading

- [02-types-and-traits.md](02-types-and-traits.md) -- Learn abstraction with types and traits
- [../01-advanced/00-lifetimes.md](../01-advanced/00-lifetimes.md) -- Understand lifetimes in detail
- [../01-advanced/01-smart-pointers.md](../01-advanced/01-smart-pointers.md) -- Extend ownership with Box/Rc/Arc

---

## References

1. **The Rust Programming Language - Ch.4 Understanding Ownership** -- https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html
2. **Rust by Example - Ownership** -- https://doc.rust-lang.org/rust-by-example/scope/move.html
3. **The Rustonomicon - Ownership** -- https://doc.rust-lang.org/nomicon/ownership.html
4. **Non-Lexical Lifetimes (NLL) RFC** -- https://rust-lang.github.io/rfcs/2094-nll.html
5. **Rust API Guidelines - Ownership** -- https://rust-lang.github.io/api-guidelines/ownership.html
6. **Learning Rust With Entirely Too Many Linked Lists** -- https://rust-unofficial.github.io/too-many-lists/
