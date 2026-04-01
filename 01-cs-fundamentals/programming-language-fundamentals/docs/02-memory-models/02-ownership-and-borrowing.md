# Ownership and Borrowing --- A Complete Guide to Rust's Memory Management Model

> **Learning Objective**: Understand the three pillars of Rust's ownership system --- Ownership, Borrowing, and Lifetimes --- and become capable of designing safe and performant programs without a GC. Systematically learn the principles of this revolutionary mechanism that guarantees memory safety at compile time, through abundant code examples and diagrams.

---

## Table of Contents

1. [Introduction --- Why Ownership Is Needed](#1-introduction----why-ownership-is-needed)
2. [The Three Rules of Ownership](#2-the-three-rules-of-ownership)
3. [Move Semantics and Copy Semantics](#3-move-semantics-and-copy-semantics)
4. [Borrowing --- The Discipline of References](#4-borrowing---the-discipline-of-references)
5. [Lifetimes](#5-lifetimes)
6. [Smart Pointers and Extensions of Ownership](#6-smart-pointers-and-extensions-of-ownership)
7. [Ownership Patterns --- Applications to Design](#7-ownership-patterns----applications-to-design)
8. [Comparison with Other Languages --- Memory Safety Approaches](#8-comparison-with-other-languages----memory-safety-approaches)
9. [Anti-Patterns and Pitfalls](#9-anti-patterns-and-pitfalls)
10. [Practical Exercises (3 Levels)](#10-practical-exercises-3-levels)
11. [FAQ --- Frequently Asked Questions](#11-faq----frequently-asked-questions)
12. [Summary](#12-summary)
13. [References](#13-references)

---

## What You Will Learn in This Chapter

- [ ] Understand the three rules of ownership and explain the difference between move and copy
- [ ] Understand the borrowing rules for immutable and mutable references and explain why data races cannot occur
- [ ] Understand the concept of lifetimes and explain how dangling references are prevented
- [ ] Make informed decisions about when to use smart pointers (Box, Rc, Arc, RefCell)
- [ ] Design safe APIs using ownership patterns
- [ ] Compare and contrast memory management approaches in other languages such as C++/Swift/Go/Java


## Prerequisites

Having the following knowledge will deepen your understanding before reading this guide:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding of the content in [Garbage Collection (GC) Complete Guide](./01-garbage-collection.md)

---

## 1. Introduction --- Why Ownership Is Needed

### 1.1 Historical Challenges of Memory Management

The history of programming languages has been a battle with the question "How do we manage memory safely?" Since the era of C, programmers have struggled with the following bugs:

```
Classification of major memory safety bugs:

+---------------------------+------------------------------------------+
| Bug Type                  | Description                              |
+---------------------------+------------------------------------------+
| Dangling pointer          | Reference to freed memory                |
| Double free               | Freeing the same memory twice            |
| Memory leak               | Forgetting to free unneeded memory       |
| Buffer overflow           | Accessing beyond the allocated region     |
| Data race                 | Multiple threads reading/writing          |
|                           | simultaneously                           |
| Use-After-Free            | Using memory after it has been freed     |
+---------------------------+------------------------------------------+
```

According to a survey by Microsoft's security team, approximately 70% of their product security vulnerabilities are attributed to memory safety issues. Google's Chrome team has reported similar figures.

### 1.2 Traditional Approaches and Their Limitations

```
Three approaches to memory management:

  [Manual Management]    [GC]               [Ownership]
   C / C++            Java / Go / Python      Rust
      |                    |                   |
      v                    v                   v
  malloc/free      Runtime auto-reclaims   Compile-time verification
      |                    |                   |
  +--------+         +----------+        +----------+
  | Fast   |         | Safe     |        | Fast     |
  | Unsafe |         | GC pause |        | Safe     |
  +--------+         +----------+        +----------+
```

| Approach | Representative Languages | Safety | Performance | Predictability |
|----------|-------------------------|--------|------------|---------------|
| Manual management | C, C++ | Low (programmer-dependent) | High | High |
| GC (tracing) | Java, Go, C# | High | Medium (GC pauses) | Low |
| Reference counting | Swift, Python, Obj-C | Medium (circular reference issues) | Medium | Medium |
| **Ownership system** | **Rust** | **High** | **High** | **High** |

Rust introduced a new paradigm called Ownership to overcome the weaknesses of these traditional approaches. Because safety is verified at compile time, it achieves safety equal to or greater than GC with zero runtime overhead.

### 1.3 The Core Insight of the Ownership System

Rust's ownership system is based on the following insight:

> **"If every value has exactly one owner, and the value is destroyed when the owner goes out of scope, then neither memory leaks nor double frees can occur."**

From this simple principle, the sophisticated mechanisms of move semantics, borrowing, and lifetimes are derived.

```
Overview of the Ownership System:

  +-----------------------------------------------------+
  |                  Ownership System                     |
  |                                                       |
  |  +-------------+  +--------------+  +------------+  |
  |  | Ownership   |  |  Borrowing   |  |  Lifetime  |  |
  |  | Rules       |  |  Rules       |  |            |  |
  |  |             |  |              |  |            |  |
  |  | - Single    |  | - &T: many   |  | - Validity |  |
  |  |   owner     |  |   allowed    |  |   period   |  |
  |  | - Drop at   |  | - &mut T:    |  |   of refs  |  |
  |  |   scope end |  |   only one   |  | - 'a annot |  |
  |  | - Move      |  | - Mutual     |  | - Elision  |  |
  |  |             |  |   exclusion  |  |   rules    |  |
  |  +------+------+  +------+-------+  +-----+------+  |
  |         |               |                |          |
  |         +---------------+----------------+          |
  |                         |                            |
  |                   Borrow Checker                      |
  |                  (Borrow Checker)                     |
  |                   All checks at compile time          |
  +-----------------------------------------------------+
```

---

## 2. The Three Rules of Ownership

### 2.1 Rule Definitions

Rust's ownership is defined by the following three rules:

```
+--------------------------------------------------------------+
|                  The Three Rules of Ownership                   |
|                                                                |
|  Rule 1: Each value has exactly one "owner" (variable)        |
|                                                                |
|  Rule 2: When the owner goes out of scope, the value is       |
|          automatically destroyed (drop is called)              |
|                                                                |
|  Rule 3: Ownership can be "transferred" (moved), but          |
|          copying is not performed by default                    |
+--------------------------------------------------------------+
```

### 2.2 Rule 1 --- Single Owner

Every value is owned by exactly one variable at any given moment. This variable is called the "Owner."

```rust
// ===== Code Example 1: Basics of Ownership =====

fn main() {
    // s is the owner of the String value "hello"
    let s = String::from("hello");

    // Memory layout of String:
    //
    //  Stack (s)            Heap
    //  +---------+         +---+---+---+---+---+
    //  | ptr   --|-------->| h | e | l | l | o |
    //  | len: 5  |         +---+---+---+---+---+
    //  | cap: 5  |
    //  +---------+
    //
    // s holds metadata on the stack (pointer, length, capacity),
    // and the actual string data is stored on the heap.

    println!("{}", s); // Access the value through s
}
// <- s goes out of scope -> drop(s) is called, heap memory is freed
```

### 2.3 Rule 2 --- Scope and Automatic Destruction

When the owner goes out of scope, Rust automatically calls the `drop` function to destroy the value. This mechanism follows the same principle as C++'s RAII (Resource Acquisition Is Initialization), but in Rust, the compiler strictly enforces it.

```rust
// ===== Code Example 2: Behavior of Scope and drop =====

struct DatabaseConnection {
    url: String,
}

impl DatabaseConnection {
    fn new(url: &str) -> Self {
        println!("[OPEN] Establishing DB connection: {}", url);
        DatabaseConnection { url: url.to_string() }
    }
}

impl Drop for DatabaseConnection {
    fn drop(&mut self) {
        println!("[CLOSE] Closing DB connection: {}", self.url);
    }
}

fn process_data() {
    let conn = DatabaseConnection::new("postgres://localhost/mydb");
    // Processing with conn...
    println!("[QUERY] Fetching data...");

    {
        let temp_conn = DatabaseConnection::new("postgres://localhost/tempdb");
        println!("[QUERY] Fetching temporary data...");
    } // <- temp_conn goes out of scope -> drop called
      //   Output: [CLOSE] Closing DB connection: postgres://localhost/tempdb

    println!("[QUERY] Additional processing...");
} // <- conn goes out of scope -> drop called
  //   Output: [CLOSE] Closing DB connection: postgres://localhost/mydb

// Execution result:
// [OPEN]  Establishing DB connection: postgres://localhost/mydb
// [QUERY] Fetching data...
// [OPEN]  Establishing DB connection: postgres://localhost/tempdb
// [QUERY] Fetching temporary data...
// [CLOSE] Closing DB connection: postgres://localhost/tempdb
// [QUERY] Additional processing...
// [CLOSE] Closing DB connection: postgres://localhost/mydb
```

As this example demonstrates, the ownership and scope mechanism structurally prevents resource leaks. This applies to any resource: file handles, network connections, locks, and more.

### 2.4 Rule 3 --- Move (Transfer of Ownership)

Ownership of a value can be transferred to another variable. After the transfer, the original variable becomes invalid.

```rust
let s1 = String::from("hello");
let s2 = s1;  // Ownership moves from s1 to s2 (move)
// println!("{}", s1);  // Compile error! s1 can no longer be used
println!("{}", s2);     // OK: s2 is the new owner
```

The following diagram illustrates the memory changes when a move occurs:

```
Before move:
  s1                     Heap
  +---------+           +---+---+---+---+---+
  | ptr   --|---------->| h | e | l | l | o |
  | len: 5  |           +---+---+---+---+---+
  | cap: 5  |
  +---------+

After move (let s2 = s1):
  s1 (invalid)           Heap
  +---------+           +---+---+---+---+---+
  | (invalid)|     .---->| h | e | l | l | o |
  +---------+     |     +---+---+---+---+---+
                  |
  s2              |
  +---------+     |
  | ptr   --|-----'
  | len: 5  |
  | cap: 5  |
  +---------+

  Key points:
  - Heap data is NOT copied (only the pointer is moved)
  - s1 is invalidated and can no longer be accessed
  - Heap memory is freed only when s2 goes out of scope
  - Double free is structurally impossible
```

---

## 3. Move Semantics and Copy Semantics

### 3.1 Situations Where Move Occurs

Moves occur not only with assignments, but in various situations:

```rust
// ===== Code Example 3: Various situations where moves occur =====

fn main() {
    // (1) Assignment to a variable
    let s1 = String::from("hello");
    let s2 = s1;  // Move

    // (2) Passing as a function argument
    let s3 = String::from("world");
    take_ownership(s3);  // Ownership of s3 moves to the function
    // println!("{}", s3);  // Compile error!

    // (3) Function return value
    let s4 = give_ownership();  // Receive ownership from the function
    println!("{}", s4);  // OK

    // (4) Push to a vector
    let s5 = String::from("item");
    let mut vec = Vec::new();
    vec.push(s5);  // Ownership of s5 moves to the vector
    // println!("{}", s5);  // Compile error!

    // (5) Pattern matching
    let opt = Some(String::from("data"));
    match opt {
        Some(s) => println!("Got: {}", s),  // Ownership moves to s
        None => println!("None"),
    }
    // println!("{:?}", opt);  // Compile error!
}

fn take_ownership(s: String) {
    println!("Took ownership of: {}", s);
} // <- s goes out of scope -> memory freed

fn give_ownership() -> String {
    String::from("gifted")  // Transfer ownership to the caller
}
```

### 3.2 The Copy Trait and Clone Trait

Some types are copied rather than moved. This is limited to types that implement the `Copy` trait.

```rust
// Types that implement the Copy trait (types that are copied)
let x: i32 = 42;
let y = x;     // Copy (x is still valid)
println!("{}", x);  // OK

let a: f64 = 3.14;
let b = a;     // Copy
println!("{}", a);  // OK

let c: bool = true;
let d = c;     // Copy

let e: char = 'A';
let f = e;     // Copy

let g: (i32, f64) = (1, 2.0);
let h = g;     // Copy because all tuple elements are Copy

// Types that do NOT implement the Copy trait (types that are moved)
let s1 = String::from("hello");
let s2 = s1;   // Move (s1 is invalid)

let v1 = vec![1, 2, 3];
let v2 = v1;   // Move (v1 is invalid)

// Explicit clone (deep copy)
let s3 = String::from("world");
let s4 = s3.clone();  // Full copy including heap data
println!("{} {}", s3, s4);  // Both valid
```

### 3.3 Difference Between Copy and Clone

| Property | Copy | Clone |
|----------|------|-------|
| Behavior | Bitwise shallow copy | Arbitrary custom logic (usually deep copy) |
| Implicitness | Implicit (auto-copied on assignment) | Explicit (requires `.clone()` call) |
| Performance | Always lightweight (stack copy) | Depends on type (may involve heap allocation) |
| Requirements | All fields of the type must be Copy | Can be implemented even with Drop |
| Examples | i32, f64, bool, char, &T | String, Vec, HashMap |
| Coexistence with Drop | Impossible (Copy and Drop are mutually exclusive) | Possible |

```
Flowchart for deciding between Copy and Clone:

  When assigning type T
       |
       v
  Does T implement Copy?
       |
  +----+----+
  Yes       No
  |         |
  v         v
 Implicit  Move (ownership transfer)
 copy      (original variable is invalid)
 (T is          |
  still         v
  valid)   Calling clone() explicitly
           allows copying
```

### 3.4 Implementing Copy/Clone for Custom Types

```rust
// Derive Copy + Clone (only when all fields are Copy)
#[derive(Debug, Copy, Clone)]
struct Point {
    x: f64,
    y: f64,
}

let p1 = Point { x: 1.0, y: 2.0 };
let p2 = p1;  // Copy (p1 is valid)
println!("{:?} {:?}", p1, p2);

// Derive Clone only (when containing heap data)
#[derive(Debug, Clone)]
struct Person {
    name: String,    // String is not Copy
    age: u32,
}

let alice = Person { name: String::from("Alice"), age: 30 };
let alice2 = alice.clone();  // Explicit clone
// let alice3 = alice;       // Move (will be moved if not cloned)
println!("{:?}", alice2);

// Custom Clone implementation
#[derive(Debug)]
struct Config {
    name: String,
    values: Vec<i32>,
    read_count: std::cell::Cell<u32>,
}

impl Clone for Config {
    fn clone(&self) -> Self {
        Config {
            name: self.name.clone(),
            values: self.values.clone(),
            read_count: std::cell::Cell::new(0),  // Reset count on clone
        }
    }
}
```

---

## 4. Borrowing --- The Discipline of References

### 4.1 Basic Concept of Borrowing

The mechanism for using a value without transferring ownership is "Borrowing." Borrowing is done through references (`&T` or `&mut T`).

```
Conceptual diagram of borrowing:

  Owner s              Borrower r
  +---------+          +---------+
  | ptr   --|------.   |         |
  | len: 5  |      |   | ptr   --|---.
  | cap: 5  |      |   +---------+   |
  +---------+      |                  |
                   v                  |
                +---+---+---+---+---+ |
                | h | e | l | l | o |<'
                +---+---+---+---+---+

  r = &s  ->  r is "borrowing" the data owned by s
  Key points:
  - r does not own the data
  - When r goes out of scope, the data is not freed
  - If s goes out of scope before r, r becomes dangling
    -> The compiler prevents this (lifetime check)
```

### 4.2 The Three Rules of Borrowing

```
+--------------------------------------------------------------+
|                  The Three Rules of Borrowing                   |
|                                                                |
|  Rule 1: Any number of immutable references (&T) can          |
|          exist simultaneously                                   |
|          -> "Multiple readers" is fine                          |
|                                                                |
|  Rule 2: Only one mutable reference (&mut T) can exist        |
|          at a time                                              |
|          -> "Only one writer"                                   |
|                                                                |
|  Rule 3: Immutable and mutable references cannot exist         |
|          simultaneously                                         |
|          -> "You can't write while someone is reading"          |
+--------------------------------------------------------------+
```

### 4.3 Immutable Borrowing (&T)

```rust
// ===== Code Example 4: Details of immutable borrowing =====

fn main() {
    let s = String::from("hello, world");

    // Multiple immutable references can be created simultaneously
    let r1 = &s;
    let r2 = &s;
    let r3 = &s;

    println!("{}, {}, {}", r1, r2, r3); // All valid

    // Pass immutable references to functions
    let length = calculate_length(&s);
    let first = first_word(&s);

    println!("Length of '{}': {}, first word: '{}'", s, length, first);
    // s is still valid (ownership has not been transferred)
}

// Function that receives an immutable reference
fn calculate_length(s: &String) -> usize {
    s.len()
    // s goes out of scope here, but since it's a reference, nothing happens
    // The original data is not freed
}

fn first_word(s: &str) -> &str {
    let bytes = s.as_bytes();
    for (i, &byte) in bytes.iter().enumerate() {
        if byte == b' ' {
            return &s[..i];
        }
    }
    s
}
```

### 4.4 Mutable Borrowing (&mut T)

```rust
fn main() {
    let mut s = String::from("hello");

    // Create a mutable reference
    let r = &mut s;
    r.push_str(", world!");
    println!("{}", r);  // "hello, world!"

    // Only one mutable reference at a time
    let mut data = vec![1, 2, 3];

    let r1 = &mut data;
    // let r2 = &mut data;  // Compile error! Second mutable reference
    r1.push(4);
    println!("{:?}", r1);

    // After r1's use ends, a new mutable reference can be created
    let r2 = &mut data;
    r2.push(5);
    println!("{:?}", r2);
}

// Function that receives a mutable reference
fn append_greeting(s: &mut String) {
    s.push_str(", world!");
}
```

### 4.5 Examples of Borrowing Rule Violations and Solutions

```rust
fn main() {
    // --- Violation 1: Taking a mutable reference while immutable references exist ---
    let mut s = String::from("hello");
    let r1 = &s;         // Immutable reference
    let r2 = &s;         // Immutable reference (OK)
    // let r3 = &mut s;  // Compile error!
    //   Cannot take a mutable reference while r1, r2 are alive
    println!("{} {}", r1, r2);
    // Here, the last use of r1, r2 ends (NLL: Non-Lexical Lifetimes)

    let r3 = &mut s;  // OK: r1, r2 are no longer used
    r3.push_str("!");
    println!("{}", r3);

    // --- Violation 2: Two mutable references simultaneously ---
    let mut v = vec![1, 2, 3, 4, 5];
    // split_at_mut is a safe way to get two mutable slices
    let (left, right) = v.split_at_mut(3);
    left[0] = 10;
    right[0] = 40;
    println!("{:?} {:?}", left, right); // [10, 2, 3] [40, 5]
}
```

### 4.6 NLL (Non-Lexical Lifetimes)

Since Rust 2018, the scope of borrows ends at the **point of last use**, not at the lexical (curly brace) boundary. This is called NLL (Non-Lexical Lifetimes).

```rust
fn main() {
    let mut s = String::from("hello");

    // In Rust 2015, r1 was valid until the end of the block
    // In Rust 2018+ (NLL), r1 ends at the line where it is last used

    let r1 = &s;
    println!("{}", r1);  // Last use of r1 -> r1's borrow ends here

    let r2 = &mut s;     // OK: r1 is no longer used
    r2.push_str(", world");
    println!("{}", r2);
}
```

```
Visualization of NLL behavior:

  Line#    Code                     r1's active range   r2's active range
  ------   ----                     ----------------    ----------------
  1        let r1 = &s;            |-- Start
  2        println!("{}", r1);      |-- End (last use)
  3        let r2 = &mut s;                              |-- Start
  4        r2.push_str(", world");                       |
  5        println!("{}", r2);                           |-- End

  Since the active ranges of r1 and r2 do not overlap, compilation succeeds
```

### 4.7 Why These Borrowing Rules Are Needed --- Preventing Data Races

```
Three conditions for a data race to occur (all must hold simultaneously):

  Condition 1: Two or more pointers access the same data
  Condition 2: At least one is performing a write
  Condition 3: There is no synchronization of access

Rust's borrowing rules structurally eliminate this:

  Pattern A: Multiple immutable references (&T, &T, &T)
    -> Condition 2 is not met (all are read-only) -> Safe

  Pattern B: One mutable reference (&mut T)
    -> Condition 1 is not met (only one access) -> Safe

  Pattern C: Immutable reference + mutable reference -> Compile error!
    -> Conditions 1, 2, 3 can all hold -> Compiler rejects

  Conclusion: Data races are impossible at compile time in Rust
```

This mechanism is particularly powerful in concurrent programming. Data races that can only be detected at runtime in other languages are all detected at compile time in Rust.

### 4.8 Borrowing and Iteration

Special care is needed with collection borrowing during iteration, particularly when modifying a collection while iterating over it.

```rust
fn main() {
    let mut scores = vec![100, 85, 92, 78, 95];

    // NG: Trying to modify the vector during iteration
    // for &score in &scores {
    //     if score < 80 {
    //         scores.push(0);  // Compile error!
    //         // &scores (immutable borrow) and scores.push (mutable borrow) conflict
    //     }
    // }

    // OK: Collect conditions first, modify later
    let low_scores: Vec<usize> = scores
        .iter()
        .enumerate()
        .filter(|(_, &s)| s < 80)
        .map(|(i, _)| i)
        .collect();

    for &idx in &low_scores {
        scores[idx] = 0;  // No immutable borrow exists anymore
    }
    println!("{:?}", scores); // [100, 85, 92, 0, 95]

    // OK: Using retain (internally handled safely)
    let mut names = vec!["Alice", "Bob", "Charlie", "Dave"];
    names.retain(|name| name.len() > 3);
    println!("{:?}", names); // ["Alice", "Charlie", "Dave"]
}
```

---

## 5. Lifetimes

### 5.1 What Is a Lifetime?

A lifetime is a concept that represents the period during which a reference is valid. Every reference has a lifetime, and the compiler tracks lifetimes to prevent dangling references (references to freed memory).

```
Conceptual diagram of lifetimes:

  fn main() {
      let r;                  // -----+-- 'a (r's lifetime)
      {                       //      |
          let x = 5;          // -+-- 'b (x's lifetime)
          r = &x;             //  |   |   r references x
      }                       // -+   |   x goes out of scope -> freed
      // println!("{}", r);   //      |   Dangling! Compile error
  }                           // -----+

  'b is shorter than 'a -> Assigning x's reference to r is unsafe
  The compiler detects this and produces an error
```

### 5.2 Prevention of Dangling References

```rust
// ===== Code Example 5: Dangling references and their prevention =====

// NG: Trying to return a dangling reference
// fn dangling() -> &String {
//     let s = String::from("hello");
//     &s  // Compile error: s is destroyed in this function but trying to return a reference
// }
// Error message:
//   this function's return type contains a borrowed value,
//   but there is no value for it to be borrowed from

// Solution 1: Return ownership (move)
fn not_dangling_v1() -> String {
    let s = String::from("hello");
    s  // Move ownership back
}

// Solution 2: 'static lifetime (valid for the entire program)
fn not_dangling_v2() -> &'static str {
    "hello"  // String literals have 'static lifetime
}

// Solution 3: Return a reference from an argument (explicit lifetime)
fn not_dangling_v3<'a>(s: &'a str) -> &'a str {
    &s[..3]  // Return a reference with the same lifetime as the argument
}
```

### 5.3 Lifetime Annotations

When a function receives references and returns a reference, the compiler needs to know the relationship between the return value's lifetime and the arguments' lifetimes. This is made explicit with lifetime annotations.

```rust
// Lifetime annotation syntax: 'a, 'b, 'c ... (conventionally lowercase letters)

// Function that returns the longer of two string slices
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}

// Meaning of 'a:
// "The returned reference is valid for a period equal to or shorter than
//  the shorter of the lifetimes of x and y"

fn main() {
    let string1 = String::from("long string");
    let result;
    {
        let string2 = String::from("xyz");
        result = longest(string1.as_str(), string2.as_str());
        println!("Longest: {}", result);  // OK: string2 is still valid
    }
    // println!("{}", result);  // Compile error!
    // string2 has been freed, so result is invalid
}

// Arguments with different lifetimes
fn first_or_default<'a, 'b>(first: &'a str, _default: &'b str) -> &'a str {
    first  // Return value has the same lifetime as first
}
```

### 5.4 Lifetimes in Structs

When a struct holds a reference, lifetime annotations are required.

```rust
// Lifetime annotations are mandatory for structs holding references
#[derive(Debug)]
struct Excerpt<'a> {
    part: &'a str,  // Reference valid for the duration of 'a
}

impl<'a> Excerpt<'a> {
    fn level(&self) -> i32 {
        3  // No lifetime annotation needed since no reference is returned
    }

    fn announce_and_return(&self, announcement: &str) -> &str {
        println!("Attention: {}", announcement);
        self.part  // Returns a reference with the same lifetime as self.part
    }
}

fn main() {
    let novel = String::from("Call me Ishmael. Some years ago...");
    let first_sentence;
    {
        let sentences: Vec<&str> = novel.split('.').collect();
        first_sentence = Excerpt { part: sentences[0] };
    }
    // first_sentence is still valid:
    // The novel that part references is still in scope
    println!("{:?}", first_sentence);
}
```

### 5.5 Lifetime Elision Rules

In many cases, the compiler automatically infers lifetimes. These are called lifetime elision rules.

```
Lifetime elision rules (3 rules):

  Rule 1 (input lifetimes):
    Assign a separate lifetime to each reference parameter
    fn foo(x: &str, y: &str) -> fn foo<'a, 'b>(x: &'a str, y: &'b str)

  Rule 2 (output lifetime - single input):
    If there is exactly one input lifetime, assign it to the output
    fn foo(x: &str) -> &str -> fn foo<'a>(x: &'a str) -> &'a str

  Rule 3 (output lifetime - method):
    If there is &self or &mut self in a method,
    assign self's lifetime to the output
    fn foo(&self, x: &str) -> &str -> self's lifetime

  If the output lifetime cannot be determined after applying all 3 rules:
  -> Compile error -> The programmer must annotate explicitly
```

```rust
// Examples where elision rules apply

// Code as written                    Compiler-inferred full form
fn first_word(s: &str) -> &str  // fn first_word<'a>(s: &'a str) -> &'a str
{
    &s[..s.find(' ').unwrap_or(s.len())]
}

// Example that cannot be elided (two input references)
// fn longest(x: &str, y: &str) -> &str  // Compile error!
// -> Ambiguous which lifetime to return
// -> Explicit annotation needed:
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}
```

### 5.6 The 'static Lifetime

`'static` is a lifetime that is valid for the entire duration of program execution.

```rust
// String literals are 'static
let s: &'static str = "I live forever";

// 'static values are valid until program termination
// They are embedded directly in the binary, so no deallocation is needed

// Caution: Using 'static carelessly is an anti-pattern
// Only use when the value truly needs to be valid for the entire program
// Even if the error message suggests 'static, in most cases
// it's a sign that the design should be reconsidered
```

---

## 6. Smart Pointers and Extensions of Ownership

### 6.1 Overview of Smart Pointers

There are cases where ownership's "only one owner per value" rule is insufficient. Smart pointers are types designed to handle these cases.

```
Classification of smart pointers:

  +---------------------------------------------------------+
  |                Smart Pointer Taxonomy                     |
  |                                                           |
  |  +-----------+  +-----------+  +-------------------+    |
  |  |  Box<T>   |  |   Rc<T>   |  |     Arc<T>        |    |
  |  |           |  |           |  |                   |    |
  |  | Heap      |  | Reference |  | Atomic            |    |
  |  | alloc     |  | counting  |  | reference count   |    |
  |  | Single    |  | Shared    |  | Thread-safe       |    |
  |  | owner     |  | ownership |  | shared ownership  |    |
  |  |           |  | Single    |  |                   |    |
  |  |           |  | thread    |  |                   |    |
  |  +-----------+  +-----------+  +-------------------+    |
  |                                                           |
  |  +-------------------+  +-------------------------+     |
  |  |   RefCell<T>      |  |   Cow<'a, T>            |     |
  |  |                   |  |                         |     |
  |  | Interior          |  | Clone-on-Write          |     |
  |  | mutability        |  | Lazy clone              |     |
  |  | Runtime borrow    |  |                         |     |
  |  | checking          |  |                         |     |
  |  +-------------------+  +-------------------------+     |
  +---------------------------------------------------------+
```

### 6.2 Box -- Heap Allocation and Recursive Types

```rust
// Box<T>: Places data on the heap, keeping only a pointer on the stack

// Use case 1: Types with unknown size at compile time
// Recursive data types (compile error without Box)
#[derive(Debug)]
enum List {
    Cons(i32, Box<List>),  // Indirection via Box gives a known size
    Nil,
}

use List::{Cons, Nil};

fn main() {
    let list = Cons(1, Box::new(Cons(2, Box::new(Cons(3, Box::new(Nil))))));
    println!("{:?}", list);
    // Cons(1, Cons(2, Cons(3, Nil)))
}

// Use case 2: Reducing the cost of moving large data
struct LargeData {
    buffer: [u8; 1_000_000],  // 1MB of data
}

fn process(data: Box<LargeData>) {
    // Moving a Box only copies the pointer (8 bytes)
    // Directly moving LargeData would copy 1MB
    println!("Processing {} bytes", data.buffer.len());
}

// Use case 3: Trait objects (dynamic dispatch)
trait Animal {
    fn speak(&self) -> &str;
}

struct Dog;
struct Cat;

impl Animal for Dog {
    fn speak(&self) -> &str { "Woof!" }
}

impl Animal for Cat {
    fn speak(&self) -> &str { "Meow!" }
}

fn get_animal(is_dog: bool) -> Box<dyn Animal> {
    if is_dog {
        Box::new(Dog)
    } else {
        Box::new(Cat)
    }
}
```

### 6.3 Rc -- Reference Counting (Single Thread)

```rust
use std::rc::Rc;

// Rc<T>: Reference counting for multiple owners
// Can only be used in a single thread

fn main() {
    // Graph structure: Both A and B reference node C
    //
    //     A ---+
    //          |
    //          v
    //          C
    //          ^
    //          |
    //     B ---+

    let c = Rc::new(String::from("shared data"));
    println!("Reference count (initial): {}", Rc::strong_count(&c));  // 1

    let a = Rc::clone(&c);  // Count +1 (NOT a clone of the data!)
    println!("Reference count (after creating a): {}", Rc::strong_count(&c));  // 2

    {
        let b = Rc::clone(&c);  // Count +1
        println!("Reference count (after creating b): {}", Rc::strong_count(&c));  // 3
    }  // b goes out of scope -> Count -1

    println!("Reference count (after dropping b): {}", Rc::strong_count(&c));  // 2

    // Data is freed when all Rc instances are dropped
}
// a, c are dropped -> Count 0 -> Data freed

// Weak<T>: Weak reference to prevent circular references
use std::rc::Weak;

#[derive(Debug)]
struct Node {
    value: i32,
    parent: RefCell<Weak<Node>>,      // Weak reference (not counted)
    children: RefCell<Vec<Rc<Node>>>,  // Strong reference
}

// Parent -> Child: Rc (strong reference)
// Child -> Parent: Weak (weak reference)
// -> No circular reference, so no memory leak
```

### 6.4 Arc -- Atomic Reference Counting (Multi-threaded)

```rust
use std::sync::Arc;
use std::thread;

// Arc<T>: Thread-safe version of Rc
// Manages reference count with atomic operations, so there is slight overhead

fn main() {
    let data = Arc::new(vec![1, 2, 3, 4, 5]);
    let mut handles = vec![];

    for i in 0..3 {
        let data_clone = Arc::clone(&data);
        let handle = thread::spawn(move || {
            let sum: i32 = data_clone.iter().sum();
            println!("Thread {}: sum = {}", i, sum);
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Final ref count: {}", Arc::strong_count(&data));  // 1
}

// Arc + Mutex: Sharing mutable data across threads
use std::sync::Mutex;

fn concurrent_counter() {
    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];

    for _ in 0..10 {
        let counter_clone = Arc::clone(&counter);
        let handle = thread::spawn(move || {
            let mut num = counter_clone.lock().unwrap();
            *num += 1;
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Counter: {}", *counter.lock().unwrap());  // 10
}
```

### 6.5 RefCell -- Interior Mutability

```rust
use std::cell::RefCell;

// RefCell<T>: Checks borrowing rules at runtime instead of compile time
// Used when "you want to mutate internals from an immutable reference"

fn main() {
    let data = RefCell::new(vec![1, 2, 3]);

    // Obtain a mutable borrow from immutable data
    data.borrow_mut().push(4);
    println!("{:?}", data.borrow());  // [1, 2, 3, 4]

    // Runtime borrowing rule violations cause a panic
    // let r1 = data.borrow();      // Immutable borrow
    // let r2 = data.borrow_mut();  // Panic! Mutable borrow while immutably borrowed
}

// Common usage pattern: Rc<RefCell<T>>
// Multiple owners + interior mutability
use std::rc::Rc;

#[derive(Debug)]
struct SharedState {
    value: Rc<RefCell<i32>>,
}

impl SharedState {
    fn new(val: i32) -> Self {
        SharedState { value: Rc::new(RefCell::new(val)) }
    }

    fn increment(&self) {
        *self.value.borrow_mut() += 1;
    }

    fn get(&self) -> i32 {
        *self.value.borrow()
    }
}
```

### 6.6 Smart Pointer Selection Guide

| Requirement | Recommended Type | Notes |
|-------------|-----------------|-------|
| Heap allocation, single owner | `Box<T>` | Simplest |
| Multiple owners (single thread) | `Rc<T>` | Reference counting |
| Multiple owners (multi-threaded) | `Arc<T>` | Atomic reference counting |
| Mutate internals from immutable ref | `RefCell<T>` | Runtime checking |
| Shared + mutable (single thread) | `Rc<RefCell<T>>` | Common combination |
| Shared + mutable (multi-threaded) | `Arc<Mutex<T>>` | Mutual exclusion via lock |
| Circular reference parent -> child | `Rc<T>` / `Arc<T>` | Strong reference |
| Circular reference child -> parent | `Weak<T>` | Weak reference |
| Lazy copy | `Cow<'a, T>` | Clone only when needed |

```
Smart pointer selection flowchart:

  What is the data ownership pattern?
       |
  +----+---------------+
  Single  Shared          Shared + mutation
  owner   (read-only)     needed
  |       |               |
  v       v               v
Box<T>  Cross thread?    Cross thread?
        |               |
     +--+--+         +--+--+
     No    Yes       No    Yes
     |      |        |      |
     v      v        v      v
   Rc<T>  Arc<T>  Rc<       Arc<
                  RefCell   Mutex
                  <T>>      <T>>
```

---

## 7. Ownership Patterns --- Applications to Design

### 7.1 Builder Pattern and Ownership

The Builder pattern leverages ownership transfer to achieve object construction through method chaining.

```rust
// ===== Code Example 6: Builder Pattern =====

#[derive(Debug)]
struct HttpRequest {
    method: String,
    url: String,
    headers: Vec<(String, String)>,
    body: Option<String>,
}

struct HttpRequestBuilder {
    method: String,
    url: String,
    headers: Vec<(String, String)>,
    body: Option<String>,
}

impl HttpRequestBuilder {
    fn new(url: &str) -> Self {
        HttpRequestBuilder {
            method: "GET".to_string(),
            url: url.to_string(),
            headers: Vec::new(),
            body: None,
        }
    }

    // Consumes self and returns it -> enables method chaining
    fn method(mut self, method: &str) -> Self {
        self.method = method.to_string();
        self
    }

    fn header(mut self, key: &str, value: &str) -> Self {
        self.headers.push((key.to_string(), value.to_string()));
        self
    }

    fn body(mut self, body: &str) -> Self {
        self.body = Some(body.to_string());
        self
    }

    fn build(self) -> HttpRequest {
        HttpRequest {
            method: self.method,
            url: self.url,
            headers: self.headers,
            body: self.body,
        }
    }
}

fn main() {
    let request = HttpRequestBuilder::new("https://api.example.com/data")
        .method("POST")
        .header("Content-Type", "application/json")
        .header("Authorization", "Bearer token123")
        .body(r#"{"key": "value"}"#)
        .build();

    println!("{:#?}", request);
}
```

### 7.2 State Machine via Ownership

A pattern that uses the type system and ownership to prevent invalid state transitions at compile time.

```rust
// ===== Code Example 7: Type-Driven State Machine =====

// Represent each state as a type
struct Draft;
struct PendingReview;
struct Published;

struct BlogPost<State> {
    title: String,
    content: String,
    state: std::marker::PhantomData<State>,
}

// Methods available only in the Draft state
impl BlogPost<Draft> {
    fn new(title: &str) -> Self {
        BlogPost {
            title: title.to_string(),
            content: String::new(),
            state: std::marker::PhantomData,
        }
    }

    fn add_content(&mut self, text: &str) {
        self.content.push_str(text);
    }

    // Transition from Draft -> PendingReview (consumes the original and returns a new type)
    fn request_review(self) -> BlogPost<PendingReview> {
        BlogPost {
            title: self.title,
            content: self.content,
            state: std::marker::PhantomData,
        }
    }
}

// Methods available only in the PendingReview state
impl BlogPost<PendingReview> {
    fn approve(self) -> BlogPost<Published> {
        BlogPost {
            title: self.title,
            content: self.content,
            state: std::marker::PhantomData,
        }
    }

    fn reject(self) -> BlogPost<Draft> {
        BlogPost {
            title: self.title,
            content: self.content,
            state: std::marker::PhantomData,
        }
    }
}

// Only Published state exposes content
impl BlogPost<Published> {
    fn content(&self) -> &str {
        &self.content
    }
}

fn main() {
    let mut post = BlogPost::<Draft>::new("Ownership Guide");
    post.add_content("Rust's ownership system is...");

    // post.content();  // Compile error! content() doesn't exist in Draft state
    let post = post.request_review();
    // post.add_content("more");  // Compile error! Cannot modify in PendingReview
    let post = post.approve();
    println!("{}", post.content());  // OK: Only accessible in Published state
}
```

### 7.3 Efficient API Design Through Borrowing

```rust
// Accepting &str allows both String and &str to be passed
fn greet(name: &str) {
    println!("Hello, {}!", name);
}

fn main() {
    let owned = String::from("Alice");
    let borrowed = "Bob";

    greet(&owned);    // Automatic conversion from String -> &str (Deref)
    greet(borrowed);  // &str is passed directly

    // Using the AsRef trait for even more flexibility
    fn process_path<P: AsRef<std::path::Path>>(path: P) {
        let path = path.as_ref();
        println!("Processing: {:?}", path);
    }

    process_path("/home/user/file.txt");         // &str
    process_path(String::from("/tmp/data.csv")); // String
    process_path(std::path::PathBuf::from("/var/log"));  // PathBuf
}
```

### 7.4 Cow --- Clone on Write Pattern

```rust
use std::borrow::Cow;

// Cow<'a, T>: Lazy copy that only clones when needed
// - Read-only -> remains a reference (zero cost)
// - Modification needed -> cloned at that point

fn normalize_name(name: &str) -> Cow<'_, str> {
    if name.contains(char::is_uppercase) {
        // Contains uppercase -> transformation needed -> create a new String
        Cow::Owned(name.to_lowercase())
    } else {
        // No transformation needed -> return the original reference (zero cost)
        Cow::Borrowed(name)
    }
}

fn main() {
    let name1 = normalize_name("alice");    // Borrowed: no clone
    let name2 = normalize_name("BOB");      // Owned: cost of to_lowercase()
    let name3 = normalize_name("charlie");  // Borrowed: no clone

    println!("{}, {}, {}", name1, name2, name3);
}
```

---

## 8. Comparison with Other Languages --- Memory Safety Approaches

### 8.1 Overview of Memory Management Approaches by Language

| Language | Approach | GC Pause | Memory Safety | Data Race Prevention | Performance Overhead |
|----------|----------|----------|--------------|---------------------|---------------------|
| **Rust** | Ownership + Borrowing | None | Compile-time guarantee | Compile-time guarantee | Zero |
| C | Manual (malloc/free) | None | None | None | Zero |
| C++ | RAII + Smart pointers | None | Partial (discipline-dependent) | None | Near zero |
| Swift | ARC | None | High | Partial (Actor) | Low |
| Go | Tracing GC | Yes (short) | High | Runtime detection (race detector) | Moderate |
| Java | Generational GC | Yes | High | Runtime detection | Moderate to high |
| Python | Reference counting + GC | Yes | High (GIL) | Mitigated by GIL | High |
| Kotlin | JVM GC | Yes | High | Runtime detection | Moderate to high |

### 8.2 Comparison with C++ --- RAII and Smart Pointers

```cpp
// C++: RAII + unique_ptr (equivalent to Rust's Box)
#include <memory>
#include <string>
#include <iostream>

class Resource {
    std::string name_;
public:
    Resource(const std::string& name) : name_(name) {
        std::cout << "Acquired: " << name_ << std::endl;
    }
    ~Resource() {
        std::cout << "Released: " << name_ << std::endl;
    }
};

void cpp_example() {
    // unique_ptr: Single ownership (similar to Rust's Box)
    auto r1 = std::make_unique<Resource>("DB Connection");
    auto r2 = std::move(r1);  // Move (r1 becomes nullptr)
    // r1->...  // Undefined behavior! (In Rust, this would be a compile error)

    // shared_ptr: Shared ownership (similar to Rust's Rc/Arc)
    auto r3 = std::make_shared<Resource>("Cache");
    auto r4 = r3;  // Reference count +1
}
// r2, r3, r4 go out of scope -> automatic deallocation
```

```
Safety comparison: C++ vs Rust

  Problem                   C++                          Rust
  ---------------------------------------------------------------
  Dangling pointer          unique_ptr becomes nullptr   Compile error
                            -> Possible undefined         -> Zero cost
                              behavior at runtime
  Data race                 Detection tools (TSan)       Compile error
                            -> Runtime only               -> Compile time
  Double free               Prevented by smart ptrs      Structurally impossible
                            -> Still possible with         -> Move semantics
                              raw pointers
  Use-After-Free            Undefined behavior            Compile error
                            -> May remain as a bug        -> Completely prevented
```

### 8.3 Comparison with Swift --- ARC

```swift
// Swift: ARC (Automatic Reference Counting)
class User {
    var name: String
    var friend: User?  // Strong reference -> risk of circular reference

    init(name: String) {
        self.name = name
        print("User \(name) created")
    }

    deinit {
        print("User \(name) deallocated")
    }
}

// Example of circular reference
var alice: User? = User(name: "Alice")
var bob: User? = User(name: "Bob")
alice?.friend = bob    // Alice -> Bob (strong reference)
bob?.friend = alice    // Bob -> Alice (strong reference) -> Circular reference!
alice = nil
bob = nil
// Even setting both to nil doesn't call deinit -> Memory leak!

// Solution: Use weak or unowned
class SafeUser {
    var name: String
    weak var friend: SafeUser?  // Weak reference

    init(name: String) { self.name = name }
    deinit { print("SafeUser \(name) deallocated") }
}
```

```
Ownership model comparison: Rust vs Swift

  Property                  Rust                        Swift
  ---------------------------------------------------------------
  Ownership model           Static ownership + borrowing ARC (reference counting)
  Safety verification       Compile time                 Runtime
  Runtime overhead          Zero                         Cost of count operations
  Circular references       Prevented with Weak<T>       Prevented with weak/unowned
                            Compiler enforces structure   Depends on programmer judgment
  Value vs Reference type   Everything is a value type   struct = value type
                            (moved)                      class = reference type
  Concurrency safety        Send/Sync traits             Actor model
                            Compile-time guarantee        Runtime verification
```

### 8.4 Comparison with Go/Java --- GC-Based Approaches

```go
// Go: Garbage Collection
package main

import "fmt"

func main() {
    // Go has no concept of ownership
    // GC automatically reclaims unneeded memory
    s1 := "hello"
    s2 := s1  // Copy (strings are immutable)
    fmt.Println(s1, s2)

    // Slices are reference types (implicit sharing)
    a := []int{1, 2, 3}
    b := a  // Shallow copy (points to the same array)
    b[0] = 100
    fmt.Println(a)  // [100 2 3]  <- a is also changed!
    // In Rust, such implicit sharing is prevented by moves
}
```

```
Performance characteristics: GC vs Ownership

  +-------------------------------------------------------+
  |    Latency comparison (conceptual diagram)              |
  |                                                       |
  |  Rust (ownership):                                    |
  |  --------------------------------- Constant latency   |
  |                                                       |
  |  Go (GC):                                             |
  |  ---------+------------------+--------- Sporadic GC   |
  |           |                  |          pauses         |
  |           GC pause           GC pause                  |
  |           (~1ms)             (~1ms)                    |
  |                                                       |
  |  Java (GC, pre-ZGC):                                  |
  |  ----------------+------------------- Long GC pause   |
  |                  |                                    |
  |                  GC pause                              |
  |                  (~10-100ms)                           |
  |                                                       |
  |  Application domains:                                 |
  |  - Rust: Real-time systems, OS, game engines          |
  |  - Go: Web servers, microservices                     |
  |  - Java: Enterprise, large-scale web apps             |
  +-------------------------------------------------------+
```

---

## 9. Anti-Patterns and Pitfalls

### 9.1 Anti-Pattern 1: Overuse of clone()

When encountering ownership or borrowing errors, inserting `.clone()` to make the code compile is the most common anti-pattern. clone() works, but it causes unnecessary heap allocations and copies, potentially degrading performance significantly.

```rust
// ===== Anti-Pattern: Overuse of clone() =====

// NG: Cloning when borrowing would suffice
fn bad_process_items(items: &Vec<String>) {
    for item in items {
        let owned = item.clone();  // NG: unnecessary clone
        println!("{}", owned);
        // owned is dropped on this line -> the clone was pointless
    }
}

// OK: Use references as-is
fn good_process_items(items: &[String]) {
    for item in items {
        println!("{}", item);  // Using &String directly is sufficient
    }
}

// NG: Cloning for HashMap key lookup
fn bad_lookup(map: &std::collections::HashMap<String, i32>, key: &str) -> Option<i32> {
    let owned_key = key.to_string();  // NG: unnecessary allocation
    map.get(&owned_key).copied()
}

// OK: Look up directly with &str
fn good_lookup(map: &std::collections::HashMap<String, i32>, key: &str) -> Option<i32> {
    map.get(key).copied()  // HashMap<String, _> can be searched with &str
}

// Cases where clone is justified:
// 1. When data needs to be sent to another thread
// 2. When data needs to be owned as a struct field
// 3. When data needs to be modified independently from the original
fn justified_clone(data: &[String]) -> Vec<String> {
    // Returning filter results as a new vector -> clone is necessary
    data.iter()
        .filter(|s| s.starts_with("important"))
        .cloned()
        .collect()
}
```

```
Checklist for detecting clone() overuse:

  Q1: Can the code compile without this clone?
      -> Consider whether a reference (&T) can be used
      -> Consider whether lifetime annotations can solve it

  Q2: Is the cloned data being modified?
      -> If not modified, a reference is sufficient

  Q3: Does the cloned data need to outlive the original?
      -> If it's consumed in the same scope, clone is unnecessary

  Q4: Is this a frequently called hot path?
      -> Clone inside loops has a significant performance impact
      -> Consider Cow<T> for lazy copying
```

### 9.2 Anti-Pattern 2: Excessive Rc<RefCell<T>> / Arc<Mutex<T>>

Rc<RefCell<T>> and Arc<Mutex<T>> are convenient, but overusing them causes Rust to lose its static safety benefits, effectively resulting in code that resembles a "GC-based language."

```rust
// ===== Anti-Pattern: Excessive use of Rc<RefCell<T>> =====

use std::rc::Rc;
use std::cell::RefCell;

// NG: Wrapping everything in Rc<RefCell<T>> --- "GC style"
struct BadGameState {
    player_hp: Rc<RefCell<i32>>,
    player_mp: Rc<RefCell<i32>>,
    enemies: Rc<RefCell<Vec<Rc<RefCell<Enemy>>>>>,
    items: Rc<RefCell<Vec<Rc<RefCell<Item>>>>>,
}
// Problems:
// 1. Risk of runtime panics (borrowing rule violations)
// 2. Overhead of reference counting
// 3. Severely degraded code readability
// 4. Loss of compile-time safety guarantees

// OK: Design with clear ownership
struct GoodGameState {
    player: Player,
    enemies: Vec<Enemy>,
    items: Vec<Item>,
}

struct Player {
    hp: i32,
    mp: i32,
}

struct Enemy {
    name: String,
    hp: i32,
}

struct Item {
    name: String,
    effect: i32,
}

impl GoodGameState {
    // Safe access through clear borrowing
    fn apply_damage(&mut self, enemy_idx: usize, damage: i32) {
        if let Some(enemy) = self.enemies.get_mut(enemy_idx) {
            enemy.hp -= damage;
        }
    }

    fn heal_player(&mut self, amount: i32) {
        self.player.hp += amount;
    }
}

// Cases where Rc<RefCell<T>> is justified:
// - Graph structures (mutual references between nodes)
// - Observer pattern (multiple observers watching the same data)
// - GUI frameworks (references between widgets)
```

### 9.3 Pitfall: Lifetime Traps

```rust
// Pitfall 1: Complexity of holding references in structs
// Structs with references propagate lifetimes and constrain callers

struct Config<'a> {
    name: &'a str,
    values: &'a [i32],
}

// This function is constrained by config's lifetime
fn process_config<'a>(config: &Config<'a>) -> &'a str {
    config.name
}

// Limit references in structs to short-lived objects.
// For long-lived structs, owned types (String, Vec<i32>) are easier to work with.

// OK: Design with owned types (simpler and easier to handle)
struct OwnedConfig {
    name: String,
    values: Vec<i32>,
}

// Pitfall 2: Closure and borrow conflicts
fn closure_trap() {
    let mut data = vec![1, 2, 3];

    // NG: Closure immutably borrows data + direct mutable borrow
    // let print_data = || println!("{:?}", data);
    // data.push(4);  // Compile error!
    // print_data();

    // OK: Group necessary operations together
    data.push(4);
    let print_data = || println!("{:?}", data);
    print_data();  // [1, 2, 3, 4]
}

// Pitfall 3: String slice lifetimes
fn string_lifetime_trap() {
    let result;
    {
        let s = String::from("hello");
        // result = &s[..];  // Compile error: cannot outlive s
        result = s;  // OK: move ownership
    }
    println!("{}", result);
}
```

### 9.4 Common Compile Errors and Their Solutions

```
+----------------------------------------------------------------------+
|              Common Borrow Checker Errors and Solutions                 |
+----------------------------------------------------------------------+
|                                                                        |
|  E0382: use of moved value                                            |
|  -> Cause: Using a variable after it has been moved                   |
|  -> Fix: clone(), use references, or separate scopes                  |
|                                                                        |
|  E0502: cannot borrow X as mutable because it is also                 |
|         borrowed as immutable                                          |
|  -> Cause: Mutable borrow while immutably borrowed                    |
|  -> Fix: Finish using immutable references first (leverage NLL)       |
|                                                                        |
|  E0499: cannot borrow X as mutable more than once at a time           |
|  -> Cause: Two simultaneous mutable references                        |
|  -> Fix: split_at_mut, temporary variables, or index-based separation |
|                                                                        |
|  E0106: missing lifetime specifier                                     |
|  -> Cause: Cannot infer lifetime of the return reference              |
|  -> Fix: Add lifetime annotation 'a                                   |
|                                                                        |
|  E0597: X does not live long enough                                    |
|  -> Cause: Referenced data goes out of scope and is freed             |
|  -> Fix: Return ownership, or widen the data's scope                  |
+----------------------------------------------------------------------+
```

---

## 10. Practical Exercises (3 Levels)

### Exercise 1: Basics --- Experiencing Move and Borrowing

**Objective**: Write a program that correctly uses ownership move, immutable borrowing, and mutable borrowing.

```rust
// ===== Exercise 1: Template =====
// Fix the compile errors in the following code.
// Rule: .clone() is forbidden. Solve using references (borrowing).

fn main() {
    let mut names = vec![
        String::from("Alice"),
        String::from("Bob"),
        String::from("Charlie"),
    ];

    // Task 1: Display all names (names will be used later)
    print_names(names);  // <-- Fix here

    // Task 2: Add a new name
    add_name(names, "Dave");  // <-- Fix here

    // Task 3: Find the longest name
    let longest = find_longest(names);  // <-- Fix here
    println!("Longest name: {}", longest);

    // Task 4: Convert all names to uppercase
    uppercase_all(names);  // <-- Fix here

    // Final check
    print_names(names);  // <-- Fix here
}

fn print_names(names: Vec<String>) {
    // <-- Fix the signature
    for name in names {
        println!("- {}", name);
    }
}

fn add_name(names: Vec<String>, name: &str) {
    // <-- Fix the signature
    names.push(String::from(name));
}

fn find_longest(names: Vec<String>) -> String {
    // <-- Fix the signature
    names.iter().max_by_key(|n| n.len()).unwrap()
}

fn uppercase_all(names: Vec<String>) {
    // <-- Fix the signature
    for name in names {
        *name = name.to_uppercase();
    }
}
```

**Model Answer**:

```rust
fn main() {
    let mut names = vec![
        String::from("Alice"),
        String::from("Bob"),
        String::from("Charlie"),
    ];

    print_names(&names);                  // Immutable borrow
    add_name(&mut names, "Dave");         // Mutable borrow
    let longest = find_longest(&names);   // Immutable borrow
    println!("Longest name: {}", longest);
    uppercase_all(&mut names);            // Mutable borrow
    print_names(&names);                  // Immutable borrow
}

fn print_names(names: &[String]) {        // Slice reference
    for name in names {
        println!("- {}", name);
    }
}

fn add_name(names: &mut Vec<String>, name: &str) {
    names.push(String::from(name));
}

fn find_longest<'a>(names: &'a [String]) -> &'a str {
    names.iter().map(|n| n.as_str()).max_by_key(|n| n.len()).unwrap()
}

fn uppercase_all(names: &mut Vec<String>) {
    for name in names.iter_mut() {
        *name = name.to_uppercase();
    }
}
```

### Exercise 2: Intermediate --- Implementing a Safe Linked List

**Objective**: Implement a singly linked list using Box and verify that memory is automatically managed through ownership.

```rust
// ===== Exercise 2: Linked List Implementation =====

#[derive(Debug)]
enum List<T> {
    Cons(T, Box<List<T>>),
    Nil,
}

impl<T: std::fmt::Display> List<T> {
    /// Create an empty list
    fn new() -> Self {
        List::Nil
    }

    /// Add an element to the front of the list
    fn prepend(self, value: T) -> Self {
        List::Cons(value, Box::new(self))
    }

    /// Return the length of the list
    fn len(&self) -> usize {
        match self {
            List::Nil => 0,
            List::Cons(_, tail) => 1 + tail.len(),
        }
    }

    /// Convert the list to a string representation
    fn to_string_repr(&self) -> String {
        match self {
            List::Nil => String::from("Nil"),
            List::Cons(head, tail) => {
                format!("{} -> {}", head, tail.to_string_repr())
            }
        }
    }

    /// Return an iterator
    fn iter(&self) -> ListIter<'_, T> {
        ListIter { current: self }
    }
}

struct ListIter<'a, T> {
    current: &'a List<T>,
}

impl<'a, T> Iterator for ListIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.current {
            List::Nil => None,
            List::Cons(value, tail) => {
                self.current = tail;
                Some(value)
            }
        }
    }
}

fn main() {
    // Build the list (prepend consumes self via move)
    let list = List::new()
        .prepend(3)
        .prepend(2)
        .prepend(1);

    println!("List: {}", list.to_string_repr());
    // Output: List: 1 -> 2 -> 3 -> Nil

    println!("Length: {}", list.len());  // 3

    // Traverse with an iterator (immutable borrow)
    let sum: i32 = list.iter().sum();
    println!("Sum: {}", sum);  // 6

    // When list goes out of scope, all nodes are automatically freed
    // Box's drop is called recursively
}
// <- list is dropped:
//   Cons(1, Box) -> Box is dropped -> Cons(2, Box) -> Box is dropped -> Cons(3, Box) -> Nil
```

**Extension challenges**: Add the following features:

1. `map` method: Return a new list with a function applied to each element
2. `filter` method: Return a new list with only elements that satisfy a condition
3. `reverse` method: Reverse the list (consume ownership and return a new list)

### Exercise 3: Advanced --- Ownership Puzzles

**Objective**: Understand code that the borrow checker rejects and fix it correctly.

```rust
// ===== Puzzle 1: Vector element reference and modification =====
// The following code produces a compile error. Explain why and fix it.

fn puzzle_1() {
    let mut v = vec![1, 2, 3, 4, 5];
    let first = &v[0];      // Immutable borrow
    v.push(6);              // Mutable borrow (vector may reallocate)
    println!("{}", first);  // first could be dangling!
}
// Hint: If push causes the vector to reallocate,
// the memory first pointed to becomes invalid.

// Fixed version:
fn puzzle_1_fixed() {
    let mut v = vec![1, 2, 3, 4, 5];
    let first = v[0];  // Copy the value (i32 is Copy)
    v.push(6);
    println!("{}", first);  // OK: using the copied value
}


// ===== Puzzle 2: Partial borrowing of a struct =====
// The following code produces a compile error. Explain why and fix it.

struct User {
    name: String,
    email: String,
    age: u32,
}

fn puzzle_2() {
    let mut user = User {
        name: String::from("Alice"),
        email: String::from("alice@example.com"),
        age: 30,
    };

    let name_ref = &user.name;  // Immutable borrow of name
    user.age += 1;              // Mutable borrow of age
    println!("{}", name_ref);
    // Note: In Rust 2021, this code compiles!
    // Field-level disjoint borrows are allowed.
    // However, it may not compile when accessing through methods.
}

fn puzzle_2_method() {
    let mut user = User {
        name: String::from("Alice"),
        email: String::from("alice@example.com"),
        age: 30,
    };

    let name_ref = &user.name;
    // user.celebrate_birthday();  // Requires &mut self -> borrows entire user
    // println!("{}", name_ref);   // Compile error!
    println!("{}", name_ref);

    // Fix: Modify the field directly instead of using a method
    user.age += 1;
}


// ===== Puzzle 3: Closures and ownership =====
fn puzzle_3() {
    let mut numbers = vec![5, 2, 8, 1, 9, 3];

    // Closure for sorting
    let sort_desc = |v: &mut Vec<i32>| {
        v.sort_by(|a, b| b.cmp(a));
    };

    sort_desc(&mut numbers);
    println!("{:?}", numbers);  // [9, 8, 5, 3, 2, 1]

    // move closure: Transfer ownership to the closure
    let numbers2 = vec![10, 20, 30];
    let sum_fn = move || -> i32 {
        numbers2.iter().sum()
    };
    // println!("{:?}", numbers2);  // Compile error! Ownership moved to the closure
    println!("Sum: {}", sum_fn());  // 60
}


// ===== Puzzle 4: Lifetime inference =====
// Add lifetime annotations to the following function signatures.

// Q: fn first_or_second(a: &str, b: &str, use_first: bool) -> &str
// A:
fn first_or_second<'a>(a: &'a str, b: &'a str, use_first: bool) -> &'a str {
    if use_first { a } else { b }
}

// Q: fn get_or_insert(map: &mut HashMap<String, String>, key: &str) -> &str
// Hint: This cannot be solved with references alone. The return type needs to change.
use std::collections::HashMap;
fn get_or_insert(map: &mut HashMap<String, String>, key: &str) -> String {
    map.entry(key.to_string())
        .or_insert_with(|| format!("default_{}", key))
        .clone()
}


// ===== Puzzle 5: Designing mutual references =====
// Design a parent-child data structure with ownership in mind.

use std::rc::{Rc, Weak};
use std::cell::RefCell;

#[derive(Debug)]
struct Parent {
    name: String,
    children: RefCell<Vec<Rc<Child>>>,
}

#[derive(Debug)]
struct Child {
    name: String,
    parent: Weak<Parent>,  // Weak reference to prevent cycles
}

fn puzzle_5() {
    let parent = Rc::new(Parent {
        name: String::from("Parent"),
        children: RefCell::new(vec![]),
    });

    let child1 = Rc::new(Child {
        name: String::from("Child1"),
        parent: Rc::downgrade(&parent),
    });

    let child2 = Rc::new(Child {
        name: String::from("Child2"),
        parent: Rc::downgrade(&parent),
    });

    parent.children.borrow_mut().push(Rc::clone(&child1));
    parent.children.borrow_mut().push(Rc::clone(&child2));

    // Access parent from child (upgrade Weak to Rc)
    if let Some(p) = child1.parent.upgrade() {
        println!("{}'s parent: {}", child1.name, p.name);
    }

    println!("Parent's number of children: {}", parent.children.borrow().len());
}
```

---

## 11. FAQ --- Frequently Asked Questions

### Q1: Why is having "ownership" safer than C++ RAII?

**A**: C++ RAII guarantees "automatic resource deallocation," but **does not guarantee prevention of dangling references**. In C++, you can extract a raw pointer from a `unique_ptr` and access it after deallocation, which results in undefined behavior. Rust's borrow checker completely tracks the validity period of references at compile time, making dangling references impossible at the compile level. In other words, while RAII guarantees "when to free," Rust's ownership also guarantees "safety of access."

### Q2: Coming from a GC language, what tips help with getting used to ownership?

**A**: The following step-by-step approach is effective:

1. **It's fine to use clone() liberally at first**: Write working code first, then remove clone() calls later
2. **Decide "who owns the data" first**: When designing functions and data structures, clarify the owner
3. **Think of references as "temporary peeks"**: Borrowing is just "taking a quick look" at data, not owning it
4. **Read compiler error messages carefully**: Rust's error messages are extremely detailed and even suggest fixes
5. **Start by distinguishing between String and &str**: This is the best entry point for understanding the relationship between owned and borrowed types

### Q3: Are there data structures that cannot be expressed with the ownership system?

**A**: Doubly linked lists, graph structures, and data structures with circular references cannot be directly expressed with ownership's "single owner" rule alone. The following approaches are available:

- **Rc<RefCell<T>>**: Shared ownership + interior mutability in a single thread
- **Arena allocator**: Store all nodes in a single vector and reference by index
- **unsafe**: Programmer guarantees safety (last resort)
- **External crates**: `petgraph` (graphs), `slotmap` (index-based references)

The Arena pattern (using vector indices as "pointers") in particular is widely recommended as safe and efficient.

```rust
// Example of the Arena pattern
struct Arena<T> {
    nodes: Vec<T>,
}

type NodeId = usize;

struct GraphNode {
    value: String,
    edges: Vec<NodeId>,  // Reference other nodes by index
}

impl Arena<GraphNode> {
    fn add_node(&mut self, value: String) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(GraphNode { value, edges: vec![] });
        id
    }

    fn add_edge(&mut self, from: NodeId, to: NodeId) {
        self.nodes[from].edges.push(to);
    }
}
```

### Q4: What is the relationship between async/await and ownership?

**A**: In asynchronous functions (async fn), values held across `.await` points are stored inside the Future. Therefore, holding references across `.await` points easily leads to lifetime issues. The general advice is to use owned types (String, Vec, etc.) rather than references in async code, and share with Arc when necessary.

### Q5: What happens to ownership rules when using unsafe?

**A**: Even within `unsafe` blocks, ownership rules are **logically still in effect**, but some compiler checks are disabled. When using unsafe, the programmer must manually guarantee the following invariants:

1. References must not be dangling
2. Borrowing rules (shared XOR mutable) must be upheld
3. Memory must be properly initialized
4. Data races must not occur

unsafe is needed for standard library internals and FFI (Foreign Function Interface), but there is almost no need to use it in application code.

---


## FAQ

### Q1: What is the most important point in learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory, but by actually writing code and verifying behavior.

### Q2: What is a common mistake beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this used in practice?

Knowledge of this topic is frequently applied in daily development work. It becomes particularly important during code reviews and architecture design.

---

## 12. Summary

### 12.1 Complete Concept Map

| Concept | Description | Keywords |
|---------|-------------|----------|
| Ownership | Only one owner per value | `let`, move, `drop` |
| Move | Transfer of ownership (original variable becomes invalid) | Assignment, function args, return values |
| Copy | Bitwise shallow copy | `Copy` trait, primitive types |
| Clone | Deep copy (explicit) | `.clone()`, `Clone` trait |
| Immutable borrowing (&T) | Read-only reference (multiple allowed) | Shared reference, immutable reference |
| Mutable borrowing (&mut T) | Writable reference (only one) | Exclusive reference, mutable reference |
| Lifetime | Validity period of a reference | `'a`, `'static`, elision rules |
| NLL | Non-Lexical Lifetimes | Scope ends at last use point |
| Box\<T\> | Heap allocation, single owner | Recursive types, trait objects |
| Rc\<T\> | Reference counting (single thread) | Shared ownership, `Weak<T>` |
| Arc\<T\> | Atomic reference counting | Multi-threaded, `Mutex<T>` |
| RefCell\<T\> | Interior mutability (runtime check) | `borrow()`, `borrow_mut()` |
| Cow\<T\> | Clone on Write | Lazy copy, optimization |

### 12.2 Five Key Principles of the Ownership System

```
+--------------------------------------------------------------+
|            Five Key Principles of the Ownership System          |
|                                                                |
|  1. Every value has exactly one owner                          |
|     -> Automatically freed when the owner disappears           |
|                                                                |
|  2. Shared and mutable are mutually exclusive                  |
|     -> Either multiple &T or one &mut T                        |
|                                                                |
|  3. A reference cannot outlive its owner                       |
|     -> Guaranteed at compile time via lifetimes                |
|                                                                |
|  4. Move transfers ownership; clone duplicates the value       |
|     -> Choose appropriately as needed                          |
|                                                                |
|  5. Smart pointers extend the ownership rules                  |
|     -> Flexibly handle cases with Box, Rc, Arc, RefCell        |
+--------------------------------------------------------------+
```

### 12.3 Next Steps in Learning

1. **Practice**: Experience ownership in small projects (CLI tools, Web APIs)
2. **Concurrent programming**: Learn the relationship between Send/Sync traits and ownership
3. **Unsafe Rust**: Understand the mechanisms behind safe abstractions
4. **Macros**: Techniques to reduce ownership-related boilerplate

---

## Next Guides to Read


---

## 13. References

### Books

1. Klabnik, S. & Nichols, C. *The Rust Programming Language*, 2nd Edition. No Starch Press, 2023. Chapter 4 "Understanding Ownership", Chapter 10 "Generic Types, Traits, and Lifetimes", Chapter 15 "Smart Pointers".
2. Blandy, J., Orendorff, J. & Tindall, L. *Programming Rust: Fast, Safe Systems Development*, 2nd Edition. O'Reilly Media, 2021. Part II "Ownership and References".
3. Gjengset, J. *Rust for Rustaceans: Idiomatic Programming for Experienced Developers*. No Starch Press, 2021. Chapter 1 "Foundations" (Ownership, Borrowing, Lifetimes).

### Official Documentation and Papers

4. The Rust Reference. "Ownership." https://doc.rust-lang.org/reference/
5. The Rustonomicon. "Ownership and Lifetimes." https://doc.rust-lang.org/nomicon/
6. Matsakis, N. "Non-Lexical Lifetimes (NLL)." Rust RFC 2094, 2017. https://rust-lang.github.io/rfcs/2094-nll.html
7. Jung, R., et al. "RustBelt: Securing the Foundations of the Rust Programming Language." *Proceedings of the ACM on Programming Languages (POPL)*, 2018.

### Web Resources

8. Rust By Example. "Ownership and Moves." https://doc.rust-lang.org/rust-by-example/scope/move.html
9. Brown, W. "Too Many Linked Lists." https://rust-unofficial.github.io/too-many-lists/ --- A comprehensive guide to linked list implementations in Rust.
10. Microsoft Security Response Center. "A proactive approach to more secure code." 2019. https://msrc.microsoft.com/ --- Survey report that memory safety vulnerabilities account for approximately 70% of all vulnerabilities.

---

## References

- [MDN Web Docs](https://developer.mozilla.org/) - Web technology reference
- [Wikipedia](https://en.wikipedia.org/) - Overview of technical concepts
