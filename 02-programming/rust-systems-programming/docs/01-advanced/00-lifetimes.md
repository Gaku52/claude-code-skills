# Lifetimes In Depth -- The Mechanism for Proving Reference Validity at Compile Time

> Lifetimes are the mechanism by which the Rust compiler tracks the validity period of references, eliminating dangling references and use-after-free at compile time.

---

## What You Will Learn in This Chapter

1. **Lifetime annotations 'a** -- Understand the meaning and syntax of lifetime parameters in function signatures
2. **Lifetime elision rules** -- Master the three rules by which the compiler implicitly infers annotations
3. **Advanced lifetimes** -- Learn about lifetimes on structs, HRTB, and 'static
4. **NLL (Non-Lexical Lifetimes)** -- Understand the improved lifetime analysis introduced in Rust 2018
5. **Practical patterns** -- Master how to handle complex lifetime scenarios encountered in practice


## Prerequisite Knowledge

Your understanding will be deeper if you have the following knowledge before reading this guide:

- Basic programming knowledge
- Understanding of related fundamental concepts

---

## 1. Fundamental Concepts of Lifetimes

### 1.1 Why Lifetimes Are Necessary

Rust is a language without a garbage collector (GC), and it guarantees memory safety at compile time. Lifetimes are one of the core mechanisms that achieve this, preventing the following problems:

- **Dangling reference**: A reference to memory that has already been freed
- **Use-after-free**: Memory access after deallocation
- **Data race**: The foundation for exclusive control through reference validity management

In C/C++, these issues cause runtime crashes or undefined behavior, but in Rust they are detected and eliminated at compile time.

```rust
// A typical dangling reference that occurs in C
// int* create_int() {
//     int x = 42;
//     return &x;  // The stack frame disappears -> dangling!
// }

// In Rust, equivalent code becomes a compile error
// fn create_ref() -> &i32 {
//     let x = 42;
//     &x  // Compile error: `x` does not live long enough
// }

// Correct approach: return ownership
fn create_value() -> i32 {
    42
}

// Or allocate on the heap and return ownership
fn create_string() -> String {
    String::from("hello")
}
```

### 1.2 Preventing Dangling References

```rust
// This code becomes a compile error
// fn dangle() -> &String {
//     let s = String::from("hello");
//     &s  // s is dropped at the end of this function -> dangling reference!
// }

// Correct approach: return ownership
fn no_dangle() -> String {
    String::from("hello")
}

fn main() {
    let s = no_dangle();
    println!("{}", s); // OK: s holds ownership
}
```

### 1.3 Visualizing Lifetimes

```
fn main() {
    let r;                // ---------+-- 'a
                          //          |
    {                     //          |
        let x = 5;       // -+-- 'b  |
        r = &x;          //  |       |
    }                     // -+       |  <- x is dropped
                          //          |
    // println!("{}", r); //          |  <- r is an invalid reference -> error
}                         // ---------+

'b is shorter than 'a -> r = &x is invalid
```

### 1.4 How the Borrow Checker Works

The borrow checker validates lifetimes through the following steps:

1. **Lifetime assignment**: Assigns a lifetime region to each reference
2. **Constraint collection**: Collects constraints from function signatures and variable usage sites
3. **Constraint resolution**: Verifies whether an assignment of lifetimes that simultaneously satisfies all constraints exists
4. **Error reporting**: When constraints cannot be satisfied, generates a concrete error message

```rust
fn example() {
    let x = String::from("hello");  // x's lifetime begins
    let r = &x;                      // r is a reference to x. The constraint is 'r <= 'x
    println!("{}", r);               // r's last point of use
    // r's lifetime ends (NLL)
    drop(x);                         // x's lifetime ends -> OK
}

fn failing_example() {
    let r;
    {
        let x = String::from("hello");
        r = &x;                      // 'r continues to the outer scope
    }                                // x is dropped -> 'x ends
    // println!("{}", r);            // 'r > 'x -> constraint violation -> error
}
```

---

## 2. Lifetime Annotations

### 2.1 Lifetime Annotation Syntax

Lifetime annotations are written as a lowercase alphabetic name following an apostrophe `'`. By convention, short names like `'a`, `'b`, `'c` are used.

```rust
// Basic syntax
&'a T        // An immutable reference with lifetime 'a
&'a mut T    // A mutable reference with lifetime 'a

// Generic lifetime parameters
fn function<'a>(x: &'a str) -> &'a str { x }

// Multiple lifetime parameters
fn function2<'a, 'b>(x: &'a str, y: &'b str) -> &'a str { x }

// With lifetime bounds
fn function3<'a, 'b: 'a>(x: &'a str, y: &'b str) -> &'a str {
    if x.len() > 0 { x } else { y }
}
```

### Example 1: Basic Lifetime Annotations

```rust
// Returns the longer of two string slices
// The lifetime of the return value is constrained to the shorter of the argument lifetimes
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() {
        x
    } else {
        y
    }
}

fn main() {
    let string1 = String::from("long string");
    let result;
    {
        let string2 = String::from("xyz");
        result = longest(string1.as_str(), string2.as_str());
        println!("longer one: {}", result); // OK: string2 is still valid
    }
    // println!("{}", result); // Error: string2 has been dropped
}
```

### Example 2: Arguments with Different Lifetimes

```rust
// When x and y can have different lifetimes
fn first<'a, 'b>(x: &'a str, _y: &'b str) -> &'a str {
    x // The return value depends only on x's lifetime
}

fn main() {
    let s1 = String::from("hello");
    let result;
    {
        let s2 = String::from("world");
        result = first(&s1, &s2);
    }
    println!("{}", result); // OK: result has s1's lifetime
}
```

### Example 3: When the Return Value Is a New Value

```rust
// When the return value is a new value rather than a reference to an argument, no lifetime annotation is needed
fn combine(x: &str, y: &str) -> String {
    format!("{}{}", x, y) // Returns a new String -> no lifetime needed
}

// The following is a compile error
// fn bad_return<'a>(x: &'a str) -> &'a str {
//     let s = String::from("created inside");
//     &s  // Cannot return a reference to a local variable
// }

fn main() {
    let result = combine("hello", " world");
    println!("{}", result);
}
```

### Example 4: Multiple Return Value Candidates

```rust
// When different arguments may be returned depending on a condition, lifetimes must be unified
fn select<'a>(condition: bool, x: &'a str, y: &'a str) -> &'a str {
    if condition { x } else { y }
}

// More precise lifetime specification
fn select_first<'a, 'b>(condition: bool, x: &'a str, _y: &'b str) -> &'a str {
    if condition {
        x
    } else {
        // y cannot be returned: 'b != 'a
        // Return a default value instead
        "default"  // &'static str can be coerced to any lifetime
    }
}

fn main() {
    let s1 = String::from("first");
    let result;
    {
        let s2 = String::from("second");
        result = select(true, &s1, &s2);
        println!("{}", result);
    }

    let s3 = String::from("third");
    let result2;
    {
        let s4 = String::from("fourth");
        result2 = select_first(false, &s3, &s4);
    }
    println!("{}", result2); // OK: "default" is 'static
}
```

---

## 3. Lifetime Elision Rules

```
+----------------------------------------------------------+
|            Lifetime Elision Rules                        |
+----------------------------------------------------------+
|                                                          |
| Rule 1 (input): Assign a separate lifetime to each       |
|   reference parameter                                    |
|   fn f(x: &str, y: &str)                                 |
|   -> fn f<'a, 'b>(x: &'a str, y: &'b str)                |
|                                                          |
| Rule 2 (output): If there is one input lifetime, it is   |
|   applied to the output                                  |
|   fn f(x: &str) -> &str                                  |
|   -> fn f<'a>(x: &'a str) -> &'a str                     |
|                                                          |
| Rule 3 (method): The lifetime of &self is applied to     |
|   the output                                             |
|   fn f(&self, x: &str) -> &str                           |
|   -> fn f<'a, 'b>(&'a self, x: &'b str) -> &'a str       |
|                                                          |
| If the three rules cannot determine the lifetime ->      |
|   explicit annotation is required                        |
+----------------------------------------------------------+
```

### Example 5: Examples of Applying the Elision Rules

```rust
// === Elidable using Rules 1 + 2 ===

// Elided form
fn first_word(s: &str) -> &str {
    s.split_whitespace().next().unwrap_or("")
}

// When expanded, it looks like this
fn first_word_explicit<'a>(s: &'a str) -> &'a str {
    s.split_whitespace().next().unwrap_or("")
}

// === Only Rule 1 applies -> output lifetime cannot be determined -> explicit annotation needed ===

// A non-elidable case -> explicit annotation is required
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {  // Two inputs
    if x.len() > y.len() { x } else { y }
}

// === Application of Rule 3 ===

struct MyString {
    data: String,
}

impl MyString {
    // Rule 3: the lifetime of &self is applied to the return value
    // Elided form
    fn as_str(&self) -> &str {
        &self.data
    }

    // When expanded, it looks like this
    fn as_str_explicit<'a>(&'a self) -> &'a str {
        &self.data
    }

    // Rule 3: lifetimes of other arguments are ignored
    fn with_prefix(&self, prefix: &str) -> &str {
        // The lifetime of &self is applied to the return value,
        // not the lifetime of prefix
        &self.data
    }
}
```

### Example 6: Complex Cases Where the Elision Rules Do Not Apply

```rust
// Case 1: Two input references, ambiguous which the return depends on
// fn ambiguous(x: &str, y: &str) -> &str { ... }  // Compile error
fn not_ambiguous<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}

// Case 2: Lifetime of trait objects
// The lifetime of Box<dyn Trait> defaults to 'static
fn create_trait_obj() -> Box<dyn std::fmt::Display> {
    Box::new(42)  // i32 is 'static
}

// A non-'static trait object
fn create_trait_obj_with_ref<'a>(s: &'a str) -> Box<dyn std::fmt::Display + 'a> {
    Box::new(s)
}

// Case 3: Lifetime of impl Trait
fn create_iter<'a>(s: &'a str) -> impl Iterator<Item = char> + 'a {
    s.chars()
}

fn main() {
    let s = String::from("hello world");
    let result = not_ambiguous(&s, "default");
    println!("{}", result);

    let obj = create_trait_obj();
    println!("{}", obj);

    let chars: Vec<char> = create_iter(&s).collect();
    println!("{:?}", chars);
}
```

---

## 4. Lifetimes on Structs

### Example 7: A Struct Holding a Reference

```rust
#[derive(Debug)]
struct Excerpt<'a> {
    part: &'a str,
}

impl<'a> Excerpt<'a> {
    fn level(&self) -> i32 {
        3 // Elision rule 3: the lifetime of &self is applied
    }

    fn announce_and_return(&self, announcement: &str) -> &str {
        println!("Announcement: {}", announcement);
        self.part // Elision rule 3: &self -> return value's lifetime
    }
}

fn main() {
    let novel = String::from("Once upon a time. In a certain place...");
    let first_sentence;
    {
        let excerpt = Excerpt {
            part: novel.split('.').next().unwrap(),
        };
        first_sentence = excerpt.announce_and_return("Attention!");
        println!("{:?}", excerpt);
    }
    // first_sentence is a slice of novel, so it's OK as long as novel is valid
    println!("{}", first_sentence);
}
```

### Example 8: A Struct with Multiple Lifetimes

```rust
#[derive(Debug)]
struct Pair<'a, 'b> {
    first: &'a str,
    second: &'b str,
}

impl<'a, 'b> Pair<'a, 'b> {
    fn new(first: &'a str, second: &'b str) -> Self {
        Pair { first, second }
    }

    fn first(&self) -> &'a str {
        self.first
    }

    fn second(&self) -> &'b str {
        self.second
    }

    // A return value that depends on both lifetimes
    fn longer(&self) -> &str
    where
        'a: 'b,  // Constraint: 'a outlives 'b
    {
        if self.first.len() > self.second.len() {
            self.first
        } else {
            self.second
        }
    }
}

fn main() {
    let s1 = String::from("hello");
    let result;
    {
        let s2 = String::from("world!!!");
        let pair = Pair::new(&s1, &s2);
        println!("first: {}, second: {}", pair.first(), pair.second());
        // pair.longer() can only be used within the constraint of 'b
        let longer = pair.longer();
        println!("longer: {}", longer);
    }
    // result = pair.longer(); // Error because pair has been dropped
}
```

### Example 9: Combining Struct Lifetimes and Generics

```rust
use std::fmt::Display;

#[derive(Debug)]
struct Annotated<'a, T> {
    label: &'a str,
    value: T,
}

impl<'a, T: Display> Annotated<'a, T> {
    fn new(label: &'a str, value: T) -> Self {
        Annotated { label, value }
    }

    fn display(&self) {
        println!("{}: {}", self.label, self.value);
    }

    fn label(&self) -> &'a str {
        self.label
    }
}

impl<'a, T: Display> std::fmt::Display for Annotated<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}", self.label, self.value)
    }
}

fn main() {
    let label = String::from("Temperature");
    let annotated = Annotated::new(&label, 36.5_f64);
    annotated.display();
    println!("{}", annotated);
    println!("Label: {}", annotated.label());
}
```

### Example 10: The Self-Referential Struct Problem and Workarounds

```rust
// Self-referential structs cannot be created directly
// struct SelfRef {
//     data: String,
//     reference: &str,  // We want to refer to data, but cannot specify a lifetime
// }

// Workaround 1: Indirect reference via index
#[derive(Debug)]
struct TextWithHighlight {
    text: String,
    highlight_start: usize,
    highlight_end: usize,
}

impl TextWithHighlight {
    fn new(text: String, start: usize, end: usize) -> Self {
        assert!(end <= text.len());
        assert!(start <= end);
        TextWithHighlight {
            text,
            highlight_start: start,
            highlight_end: end,
        }
    }

    fn highlighted(&self) -> &str {
        &self.text[self.highlight_start..self.highlight_end]
    }

    fn full_text(&self) -> &str {
        &self.text
    }
}

// Workaround 2: Separated structs
#[derive(Debug)]
struct TextOwner {
    text: String,
}

#[derive(Debug)]
struct TextRef<'a> {
    owner: &'a TextOwner,
    start: usize,
    end: usize,
}

impl<'a> TextRef<'a> {
    fn new(owner: &'a TextOwner, start: usize, end: usize) -> Self {
        assert!(end <= owner.text.len());
        TextRef { owner, start, end }
    }

    fn get(&self) -> &str {
        &self.owner.text[self.start..self.end]
    }
}

fn main() {
    // Workaround 1
    let tw = TextWithHighlight::new("Hello, Rust World!".to_string(), 7, 11);
    println!("Full: {}", tw.full_text());
    println!("Highlighted: {}", tw.highlighted());

    // Workaround 2
    let owner = TextOwner {
        text: "Hello, Rust World!".to_string(),
    };
    let text_ref = TextRef::new(&owner, 7, 11);
    println!("Reference: {}", text_ref.get());
}
```

---

## 5. The 'static Lifetime

### 5.1 The Two Meanings of 'static

`'static` has two distinct meanings that are easily confused:

1. **`&'static T`**: A reference valid for the entire duration of the program
2. **`T: 'static`**: The type T satisfies the 'static lifetime bound (all owned types satisfy this)

```
+------------------------------------------------------+
|  The two meanings of 'static                         |
+------------------------------------------------------+
|                                                      |
|  &'static T = a reference valid for the entire       |
|    program duration                                  |
|    Example: string literals &'static str             |
|    Example: references to static variables           |
|    Example: references created with Box::leak()      |
|                                                      |
|  T: 'static = T contains no references, or only      |
|    'static references                                |
|    Example: String, Vec<i32>, i32 (all owned types   |
|    satisfy this)                                     |
|    Example: &'static str (if a reference, it must    |
|    be 'static)                                       |
|                                                      |
|  Important: T: 'static does not mean "lives forever" |
|    but rather "is *able* to live forever"            |
+------------------------------------------------------+
```

### Example 11: Correct Use of 'static

```rust
// String literals are 'static
let s: &'static str = "This string is embedded in the binary";

// 'static bound: the type contains no references, or only 'static references
fn spawn_task<T: Send + 'static>(value: T) {
    std::thread::spawn(move || {
        println!("Processing in thread");
        drop(value);
    });
}

// 'static does not mean "lives forever" but "is able to live forever"
// All owned types (String, Vec<T>) satisfy the 'static bound
fn accepts_static<T: 'static>(val: T) {
    // If T contains references, only 'static references are allowed
    // If T is an owned type, it always satisfies the bound
}

fn main() {
    let owned = String::from("hello");
    accepts_static(owned); // OK: String is an owned type

    let s: &'static str = "hello";
    accepts_static(s); // OK: 'static reference

    // let local = String::from("hello");
    // accepts_static(&local); // Error: &local is not 'static
}
```

### Example 12: Box::leak and Creating 'static References

```rust
fn create_static_str(s: String) -> &'static str {
    // Box::leak intentionally leaks the heap memory to obtain a 'static reference
    // Note: the memory is never freed, so use this only in limited situations
    Box::leak(s.into_boxed_str())
}

// A pattern also used internally by lazy_static! and OnceCell
use std::sync::OnceLock;

static CONFIG: OnceLock<String> = OnceLock::new();

fn get_config() -> &'static str {
    CONFIG.get_or_init(|| {
        // In practice, read from a file or environment variable
        String::from("production")
    })
}

fn main() {
    let dynamic_string = String::from("Dynamically created string");
    let static_ref = create_static_str(dynamic_string);
    println!("{}", static_ref);

    let config = get_config();
    println!("Config: {}", config);
}
```

### Example 13: Misuse of 'static and How to Fix It

```rust
// Misuse 1: unnecessary 'static constraint
// BAD: requires 'static more than necessary
fn process_bad(data: &'static str) {
    println!("{}", data);
}

// GOOD: accepts any lifetime
fn process_good(data: &str) {
    println!("{}", data);
}

// Misuse 2: default 'static for trait objects
// BAD: unintentionally requires 'static
fn take_display_bad(item: Box<dyn std::fmt::Display>) {
    // Box<dyn Display> is the same as Box<dyn Display + 'static>
    println!("{}", item);
}

// GOOD: explicitly specify the lifetime
fn take_display_good<'a>(item: Box<dyn std::fmt::Display + 'a>) {
    println!("{}", item);
}

fn main() {
    // process_bad cannot accept a reference to a String
    // let s = String::from("hello");
    // process_bad(&s); // Error

    process_bad("Literals are OK");
    process_good("Literals are also OK");
    let s = String::from("Variables are also OK");
    process_good(&s);
}
```

---

## 6. Higher-Rank Trait Bounds (HRTB)

```
+------------------------------------------------------+
|  HRTB (Higher-Rank Trait Bounds)                     |
|                                                      |
|  for<'a> means "for any lifetime 'a"                 |
|                                                      |
|  fn apply<F>(f: F)                                   |
|  where                                               |
|      F: for<'a> Fn(&'a str) -> &'a str               |
|                                                      |
|  -> F must be "a function that works no matter       |
|     what lifetime of reference is passed to it"      |
|                                                      |
|  Typical situations where HRTB is needed:            |
|  - Closure arguments that take a reference and       |
|    return a reference                                |
|  - Trait objects with methods having generic         |
|    lifetimes                                         |
|  - Callback-style APIs like Iterator::for_each       |
+------------------------------------------------------+
```

### Example 14: Practical Use of HRTB

```rust
fn apply_to_both<F>(f: F, a: &str, b: &str)
where
    F: for<'a> Fn(&'a str) -> &'a str,
{
    println!("{}", f(a));
    println!("{}", f(b));
}

fn identity(s: &str) -> &str {
    s
}

fn first_word(s: &str) -> &str {
    s.split_whitespace().next().unwrap_or(s)
}

fn main() {
    let s1 = String::from("hello world");
    let s2 = String::from("rust programming");
    apply_to_both(identity, &s1, &s2);
    apply_to_both(first_word, &s1, &s2);
}
```

### Example 15: Combining HRTB and Closures

```rust
// A parser-combinator-like design using HRTB
trait Parser {
    fn parse<'input>(&self, input: &'input str) -> Option<(&'input str, &'input str)>;
}

struct Literal {
    expected: String,
}

impl Parser for Literal {
    fn parse<'input>(&self, input: &'input str) -> Option<(&'input str, &'input str)> {
        if input.starts_with(&self.expected) {
            Some((&input[..self.expected.len()], &input[self.expected.len()..]))
        } else {
            None
        }
    }
}

struct Sequence {
    parsers: Vec<Box<dyn Parser>>,
}

impl Parser for Sequence {
    fn parse<'input>(&self, input: &'input str) -> Option<(&'input str, &'input str)> {
        let mut remaining = input;
        let mut matched_end = 0;

        for parser in &self.parsers {
            match parser.parse(remaining) {
                Some((_, rest)) => {
                    matched_end = input.len() - rest.len();
                    remaining = rest;
                }
                None => return None,
            }
        }

        Some((&input[..matched_end], remaining))
    }
}

fn apply_parser<P>(parser: &P, inputs: &[&str])
where
    P: for<'a> Fn(&'a str) -> Option<(&'a str, &'a str)>,
{
    for input in inputs {
        match parser(input) {
            Some((matched, rest)) => {
                println!("Match: '{}', remaining: '{}'", matched, rest);
            }
            None => {
                println!("No match: '{}'", input);
            }
        }
    }
}

fn main() {
    let literal = Literal {
        expected: "hello".to_string(),
    };

    let inputs = ["hello world", "hello", "goodbye", "hello!"];
    for input in &inputs {
        match literal.parse(input) {
            Some((matched, rest)) => println!("'{}' -> matched='{}', rest='{}'", input, matched, rest),
            None => println!("'{}' -> no match", input),
        }
    }

    // Closure version
    let prefix_parser = |input: &str| -> Option<(&str, &str)> {
        if input.starts_with("rust") {
            Some((&input[..4], &input[4..]))
        } else {
            None
        }
    };

    let test_inputs = ["rust is great", "rust", "python"];
    apply_parser(&prefix_parser, &test_inputs);
}
```

---

## 7. Lifetime Subtyping

```
Lifetime containment relationship:
  'a: 'b means "'a outlives 'b"

  +---------------------------------+
  | 'static                         |
  |  +---------------------------+  |
  |  | 'a                        |  |
  |  |  +---------------------+  |  |
  |  |  | 'b                  |  |  |
  |  |  +---------------------+  |  |
  |  +---------------------------+  |
  +---------------------------------+

  'static: 'a: 'b
  'static is longer than every lifetime
  A reference with a longer lifetime can be used
  where a shorter lifetime is expected
  (covariance)
```

### Example 16: Lifetime Bounds

```rust
// 'a: 'b means "'a is at least as long as 'b"
fn select<'a, 'b: 'a>(first: &'a str, second: &'b str) -> &'a str {
    if first.len() > second.len() {
        first
    } else {
        second // Since 'b: 'a, a 'b reference can be returned as 'a
    }
}

fn main() {
    let s1 = String::from("hello");
    let result;
    {
        let s2 = String::from("world!!");
        result = select(&s1, &s2);
        println!("{}", result);
    }
    // result is constrained to s1's lifetime ('a)
    // s2's lifetime ('b) is at least 'a, so it's OK
}
```

### Example 17: Covariance and Contravariance

```rust
// Covariance of lifetimes
// &'long T can be used as &'short T (subtype)
fn demonstrate_covariance() {
    let long_lived = String::from("long");

    // Use 'long as 'short
    fn take_short<'short>(s: &'short str) -> &'short str {
        s
    }

    // 'static is a subtype of every lifetime
    let static_str: &'static str = "static";
    let result = take_short(static_str); // 'static -> 'short is OK
    println!("{}", result);

    let result2 = take_short(&long_lived); // Ordinary lifetimes are also OK
    println!("{}", result2);
}

// Practical example of lifetime bounds
struct Container<'a> {
    data: Vec<&'a str>,
}

impl<'a> Container<'a> {
    fn new() -> Self {
        Container { data: Vec::new() }
    }

    // 'b: 'a -> 'b outlives 'a
    // That is, a reference with a longer lifetime than 'a can be added
    fn add<'b: 'a>(&mut self, item: &'b str) {
        self.data.push(item);
    }

    fn get_all(&self) -> &[&'a str] {
        &self.data
    }
}

fn main() {
    demonstrate_covariance();

    let s1 = String::from("hello");
    let s2 = String::from("world");

    let mut container = Container::new();
    container.add(&s1);
    container.add(&s2);
    container.add("static string"); // &'static str can also be added

    for item in container.get_all() {
        println!("{}", item);
    }
}
```

---

## 8. NLL (Non-Lexical Lifetimes)

### 8.1 Overview of NLL

NLL (Non-Lexical Lifetimes), introduced in the Rust 2018 Edition, is a mechanism that determines the end point of a lifetime based on "the last point of use" rather than on the lexical scope (block boundary).

```
+------------------------------------------------------+
|  Before NLL (Rust 2015)                              |
|                                                      |
|  let mut data = vec![1, 2, 3];                       |
|  let r = &data[0];         // 'r continues to scope  |
|                            // boundary               |
|  println!("{}", r);                                  |
|  // r is no longer used, but...                      |
|  data.push(4);             // Error! 'r still active |
|                                                      |
+------------------------------------------------------+
|  After NLL (Rust 2018+)                              |
|                                                      |
|  let mut data = vec![1, 2, 3];                       |
|  let r = &data[0];         // 'r begins              |
|  println!("{}", r);        // last use of 'r -> end  |
|  data.push(4);             // OK! 'r has ended       |
+------------------------------------------------------+
```

### Example 18: Improvements via NLL

```rust
fn main() {
    // Code that would not compile without NLL

    // Case 1: borrowing in a conditional branch
    let mut data = vec![1, 2, 3, 4, 5];
    let first = &data[0];
    println!("first: {}", first);
    // NLL: first's last use is here -> lifetime ends
    data.push(6); // OK
    println!("data: {:?}", data);

    // Case 2: HashMap entry pattern
    use std::collections::HashMap;
    let mut map = HashMap::new();
    map.insert("key", vec![1]);

    // Without NLL, get and insert could not be used together
    match map.get("key") {
        Some(v) => println!("found: {:?}", v),
        None => {
            map.insert("key", vec![2]);
        }
    }

    // Case 3: conditional mutable borrow
    let mut v = vec![1, 2, 3];
    let r = &v;
    println!("immutable borrow: {:?}", r);
    // r is no longer used -> NLL ends the lifetime
    v.push(4);
    println!("after modification: {:?}", v);
}
```

### Example 19: Cases NLL Cannot Resolve

```rust
fn main() {
    // Case 1: an immutable and mutable borrow that genuinely overlap
    let mut data = vec![1, 2, 3];
    let r = &data[0];
    // data.push(4); // Error: r is still used after this
    println!("{}", r);
    data.push(4); // OK: r is no longer used

    // Case 2: per-field borrowing of a struct
    struct Pair {
        first: String,
        second: String,
    }

    let mut pair = Pair {
        first: String::from("hello"),
        second: String::from("world"),
    };

    // Simultaneous mutable borrows of different fields are OK
    let r1 = &mut pair.first;
    let r2 = &mut pair.second;
    r1.push_str("!");
    r2.push_str("!");
    println!("{}, {}", r1, r2);

    // Case 3: borrows cannot be split when going through a method
    // let r3 = &pair.first;
    // pair.second.push_str("!"); // OK: different field
    // println!("{}", r3);

    // However, going through a method borrows the whole thing
    // let r4 = pair.get_first(); // &self -> immutably borrows the whole
    // pair.set_second("!"); // Error: needs &mut self but the whole is borrowed
}
```

---

## 9. Advanced Lifetime Patterns

### Example 20: Combining Lifetimes and Traits

```rust
trait Processor<'a> {
    fn process(&self, input: &'a str) -> &'a str;
}

struct TrimProcessor;

impl<'a> Processor<'a> for TrimProcessor {
    fn process(&self, input: &'a str) -> &'a str {
        input.trim()
    }
}

struct PrefixProcessor {
    len: usize,
}

impl<'a> Processor<'a> for PrefixProcessor {
    fn process(&self, input: &'a str) -> &'a str {
        if input.len() > self.len {
            &input[..self.len]
        } else {
            input
        }
    }
}

fn apply_processors<'a>(input: &'a str, processors: &[&dyn Processor<'a>]) -> &'a str {
    let mut result = input;
    for processor in processors {
        result = processor.process(result);
    }
    result
}

fn main() {
    let input = String::from("  Hello, World!  ");
    let trim = TrimProcessor;
    let prefix = PrefixProcessor { len: 5 };

    let processors: Vec<&dyn Processor> = vec![&trim, &prefix];
    let result = apply_processors(&input, &processors);
    println!("Result: '{}'", result); // "Hello"
}
```

### Example 21: Lifetimes and Iterators

```rust
struct WordIterator<'a> {
    text: &'a str,
    position: usize,
}

impl<'a> WordIterator<'a> {
    fn new(text: &'a str) -> Self {
        WordIterator { text, position: 0 }
    }
}

impl<'a> Iterator for WordIterator<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        // Skip leading whitespace
        while self.position < self.text.len()
            && self.text.as_bytes()[self.position] == b' '
        {
            self.position += 1;
        }

        if self.position >= self.text.len() {
            return None;
        }

        let start = self.position;

        // Find the end of the word
        while self.position < self.text.len()
            && self.text.as_bytes()[self.position] != b' '
        {
            self.position += 1;
        }

        Some(&self.text[start..self.position])
    }
}

fn main() {
    let text = String::from("Rust is a systems programming language");
    let words: Vec<&str> = WordIterator::new(&text).collect();
    println!("{:?}", words);
    // ["Rust", "is", "a", "systems", "programming", "language"]

    // Combination with iterator adapters
    let long_words: Vec<&str> = WordIterator::new(&text)
        .filter(|w| w.len() > 3)
        .collect();
    println!("Long words: {:?}", long_words);
    // ["Rust", "systems", "programming", "language"]
}
```

### Example 22: GAT (Generic Associated Types) and Lifetimes

```rust
// A streaming-processing pattern using GAT
trait StreamingIterator {
    type Item<'a> where Self: 'a;

    fn next<'a>(&'a mut self) -> Option<Self::Item<'a>>;
}

struct WindowIterator {
    data: Vec<i32>,
    position: usize,
    window_size: usize,
}

impl StreamingIterator for WindowIterator {
    type Item<'a> = &'a [i32];

    fn next<'a>(&'a mut self) -> Option<Self::Item<'a>> {
        if self.position + self.window_size > self.data.len() {
            None
        } else {
            let window = &self.data[self.position..self.position + self.window_size];
            self.position += 1;
            Some(window)
        }
    }
}

fn main() {
    let mut iter = WindowIterator {
        data: vec![1, 2, 3, 4, 5],
        position: 0,
        window_size: 3,
    };

    while let Some(window) = iter.next() {
        println!("Window: {:?}", window);
    }
    // [1, 2, 3]
    // [2, 3, 4]
    // [3, 4, 5]
}
```

### Example 23: Lifetimes and Asynchronous Programming

```rust
use std::future::Future;

// Lifetimes of async functions
// An async fn returns impl Future + 'lifetime,
// and depends on the lifetime of its arguments

async fn process_data(data: &str) -> usize {
    // The reference data must remain valid until the Future completes
    data.len()
}

// async with explicit lifetime annotation
fn process_data_explicit<'a>(data: &'a str) -> impl Future<Output = usize> + 'a {
    async move {
        data.len()
    }
}

// async function as a trait object
fn create_async_processor<'a>(
    data: &'a str,
) -> Box<dyn Future<Output = String> + 'a> {
    Box::new(async move {
        format!("Processing result: {}", data.to_uppercase())
    })
}

// Considerations regarding async blocks and lifetimes
fn example_async_lifetime() {
    let data = String::from("hello");

    // async blocks depend on the lifetimes of references inside them
    let _future = async {
        println!("{}", &data);
    };

    // A move async block takes ownership
    let _future_move = async move {
        println!("{}", data);
    };
    // data has been moved, so it cannot be used here
}
```

---

## 10. Comparison Tables

### 10.1 Kinds of Lifetimes

| Kind | Notation | Meaning | Example |
|------|----------|---------|---------|
| Named | `'a` | Explicit lifetime parameter | `fn f<'a>(x: &'a str)` |
| Elided | (none) | Inferred by the compiler | `fn f(x: &str) -> &str` |
| 'static | `'static` | Entire program duration | `&'static str` |
| Anonymous | `'_` | Explicitly request inference | `impl Iterator<Item = &'_ str>` |
| HRTB | `for<'a>` | For any lifetime | `F: for<'a> Fn(&'a str)` |

### 10.2 Misconceptions vs. Reality of 'static

| Misconception | Reality |
|---------------|---------|
| It stays in memory forever | It is "qualified" to be valid forever |
| Only string literals are 'static | All owned types satisfy `'static` |
| It lives on the heap | It lives in the binary's static area (in the case of literals) |
| It should not be used | It is necessary for values passed to threads |
| It leaks memory | Owned types are dropped normally |

### 10.3 Application Patterns of Lifetime Elision Rules

| Pattern | Before elision | After elision | Applied rules |
|---------|----------------|---------------|---------------|
| Single input | `fn f<'a>(x: &'a str) -> &'a str` | `fn f(x: &str) -> &str` | Rules 1+2 |
| Method | `fn f<'a>(&'a self) -> &'a str` | `fn f(&self) -> &str` | Rules 1+3 |
| Multiple inputs | Cannot be elided | `fn f<'a>(x: &'a str, y: &'a str) -> &'a str` | none |
| Method + argument | `fn f<'a,'b>(&'a self, x: &'b str) -> &'a str` | `fn f(&self, x: &str) -> &str` | Rules 1+3 |
| No references | `fn f(x: i32) -> i32` | `fn f(x: i32) -> i32` | not needed |

---

## 11. Anti-Patterns

### Anti-Pattern 1: Unnecessary 'static Constraint

```rust
// BAD: requires 'static more than necessary
fn process(data: &'static str) {
    println!("{}", data);
}

// GOOD: accepts any lifetime
fn process_good(data: &str) {
    println!("{}", data);
}

// BAD: unnecessary 'static on a trait object
fn take_processor(p: Box<dyn Fn(&str) -> String + 'static>) {
    println!("{}", p("hello"));
}

// GOOD: specify 'static only when needed
fn take_processor_good(p: Box<dyn Fn(&str) -> String>) {
    // Same as Box<dyn Fn + 'static>, but the intent is clearer
    println!("{}", p("hello"));
}
// If 'static is genuinely needed, e.g. for passing to a thread, state it explicitly
fn take_processor_thread(p: Box<dyn Fn(&str) -> String + Send + 'static>) {
    std::thread::spawn(move || {
        println!("{}", p("hello"));
    });
}
```

### Anti-Pattern 2: Use Owned Types Rather Than Fighting with Lifetimes

```rust
// BAD: lifetimes become too complex
// struct Parser<'input, 'config, 'db> {
//     input: &'input str,
//     config: &'config Config,
//     db: &'db Database,
// }

// GOOD: simplify with owned types
struct Config {
    max_depth: usize,
}

struct Database {
    connection: String,
}

struct Parser {
    input: String,
    config: Config,
    db: Database,
}

impl Parser {
    fn parse(&self) -> Result<Vec<String>, String> {
        // processing
        Ok(vec![self.input.clone()])
    }
}
// Introduce lifetimes later if performance becomes an issue
```

### Anti-Pattern 3: Excessive Lifetime Propagation

```rust
// BAD: the lifetime propagates to all users of the struct
struct BadTokenizer<'a> {
    source: &'a str,
    tokens: Vec<&'a str>,
}

// Every usage site must specify the lifetime
// fn process_tokens<'a>(tokenizer: &BadTokenizer<'a>) { ... }
// fn analyze<'a>(tokens: &[&'a str]) { ... }

// GOOD: avoid ownership issues with index-based access
struct GoodTokenizer {
    source: String,
    token_ranges: Vec<(usize, usize)>,
}

impl GoodTokenizer {
    fn new(source: String) -> Self {
        let mut ranges = Vec::new();
        let mut start = 0;
        for (i, ch) in source.char_indices() {
            if ch.is_whitespace() {
                if start < i {
                    ranges.push((start, i));
                }
                start = i + ch.len_utf8();
            }
        }
        if start < source.len() {
            ranges.push((start, source.len()));
        }
        GoodTokenizer {
            source,
            token_ranges: ranges,
        }
    }

    fn tokens(&self) -> Vec<&str> {
        self.token_ranges
            .iter()
            .map(|&(start, end)| &self.source[start..end])
            .collect()
    }
}

fn main() {
    let tokenizer = GoodTokenizer::new("hello world rust".to_string());
    println!("{:?}", tokenizer.tokens());
}
```

### Anti-Pattern 4: Stuffing References into a Collection

```rust
// BAD: trying to stuff references into a Vec leads to lifetime headaches
fn collect_references_bad<'a>() -> Vec<&'a str> {
    let mut results = Vec::new();
    // let s = String::from("hello");
    // results.push(&s);  // Error: s's lifetime is not long enough
    results
}

// GOOD: use a collection of owned types
fn collect_owned() -> Vec<String> {
    let mut results = Vec::new();
    let s = String::from("hello");
    results.push(s);
    results
}

// GOOD: collect references from an input slice
fn collect_from_input<'a>(input: &'a str) -> Vec<&'a str> {
    input.split_whitespace().collect()
}

fn main() {
    let owned = collect_owned();
    println!("{:?}", owned);

    let input = String::from("hello world rust");
    let refs = collect_from_input(&input);
    println!("{:?}", refs);
}
```

---

## 12. Practical Lifetime Debugging

### 12.1 Common Compile Errors and How to Address Them

```rust
// Error 1: "lifetime may not live long enough"
// fn bad1<'a>(x: &str) -> &'a str { x }
// Fix: align the input and output lifetimes
fn good1<'a>(x: &'a str) -> &'a str { x }

// Error 2: "cannot return reference to local variable"
// fn bad2() -> &str {
//     let s = String::from("hello");
//     &s
// }
// Fix: return an owned type
fn good2() -> String {
    String::from("hello")
}

// Error 3: "borrowed value does not live long enough"
fn good3() {
    let result;
    let s = String::from("hello");
    result = &s; // OK because s is in the same scope
    println!("{}", result);
} // s and result are dropped together

// Error 4: "cannot borrow as mutable because it is also borrowed as immutable"
fn good4() {
    let mut v = vec![1, 2, 3];
    let first = v[0]; // Copy (i32 is Copy)
    v.push(4);
    println!("{}, {:?}", first, v);

    // For references, NLL resolves it
    let r = &v[0];
    println!("{}", r); // last use of r
    v.push(5); // OK: r is no longer used
}

fn main() {
    let s = good1("hello");
    println!("{}", s);
    println!("{}", good2());
    good3();
    good4();
}
```

### 12.2 How to Read Lifetime Errors

```
error[E0597]: `x` does not live long enough
  --> src/main.rs:4:13
   |
3  |     let r;
   |         - borrow later stored here     <- r holds the reference
4  |     let x = 5;
5  |     r = &x;
   |         ^^ borrowed value does not live long enough  <- x's lifetime is too short
6  | }
   | - `x` dropped here while still borrowed  <- the point where x is dropped

How to fix:
1. Align the scopes of r and x
2. Copy/clone x's value into r
3. Move x to an outer scope
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
    """Practice basic implementation patterns"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """Validate input value"""
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

### Exercise 2: Advanced Patterns

Extend the basic implementation to add the following features.

```python
# Exercise 2: advanced patterns
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """Practice advanced patterns"""

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

    print(f"Inefficient version: {slow_time:.4f}s")
    print(f"Efficient version:   {fast_time:.6f}s")
    print(f"Speedup: {slow_time/fast_time:.0f}x")

benchmark()
```

**Key points:**
- Be conscious of algorithmic complexity
- Choose appropriate data structures
- Measure the effect with benchmarks
---

## 13. FAQ

### Q1: Do lifetime annotations have any effect at runtime?

**A:** No. Lifetime annotations are entirely compile-time information. They have no effect whatsoever on runtime code or performance. Lifetime information is not included in the binary. This is one example of Rust's "zero-cost abstractions."

### Q2: What is NLL (Non-Lexical Lifetimes)?

**A:** It is an improvement introduced in the Rust 2018 Edition. Previously, the lifetime of a reference continued until the lexical scope (block boundary), but with NLL it ends at the "last point of use." As a result, valid code that previously caused compile errors now compiles. NLL is enabled by default in current Rust.

### Q3: When do you use `'_` (the anonymous lifetime)?

**A:** Use it when you want to make the existence of a lifetime explicit while not needing to give it a concrete name:
```rust
// In an impl block, just indicate that a lifetime exists
impl fmt::Display for ImportantExcerpt<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.part)
    }
}

// In a function signature, indicate elision explicitly
fn takes_ref(s: &'_ str) -> &'_ str {
    s
}
```

### Q4: How do lifetimes interact with generics?

**A:** Lifetime parameters are a kind of generic parameter. By convention, lifetime parameters are written before type parameters:
```rust
fn example<'a, 'b, T, U>(x: &'a T, y: &'b U) -> &'a T
where
    T: Clone,
    U: std::fmt::Debug,
    'b: 'a,
{
    println!("{:?}", y);
    x
}
```

### Q5: What is lifetime variance?

**A:** Lifetime variance is the set of rules concerning the substitutability of lifetime parameters:
- **Covariant**: `&'long T` can be used as `&'short T`. Most reference types are covariant
- **Contravariant**: `fn(&'short T)` can be used as `fn(&'long T)`. Applies to function-argument positions
- **Invariant**: The lifetime of `T` in `&mut T` is invariant. `Cell<&'a T>` is also invariant

```rust
// Example of covariance
fn covariant_example<'short>(s: &'short str) {
    let static_str: &'static str = "hello";
    let _: &'short str = static_str; // 'static -> 'short OK (covariant)
}

// Example of invariance
fn invariant_example() {
    let mut x: &str = "hello";
    let y: &str = "world";
    x = y; // OK: both are &str

    // Cell is invariant, so the constraints are stricter
    use std::cell::Cell;
    let cell: Cell<&str> = Cell::new("hello");
    // Cell<&'a str> is invariant in 'a
}
```

### Q6: What is Polonius?

**A:** Polonius is Rust's next-generation borrow checker, which analyzes lifetimes more accurately than the current NLL-based checker. There are still some "actually safe but compile-error" cases under NLL, and Polonius resolves these. As of 2024, it can be used experimentally on the nightly version via the `-Z polonius` flag.

```rust
// Example of code that does not compile under current NLL but does under Polonius
// fn get_or_insert(map: &mut HashMap<String, String>, key: &str) -> &String {
//     if let Some(value) = map.get(key) {
//         return value;
//     }
//     map.insert(key.to_string(), "default".to_string());
//     map.get(key).unwrap()
// }
```

---


## FAQ

### Q1: What is the most important point in studying this topic?

Gaining practical experience is most important. Beyond theory, your understanding deepens by actually writing code and observing its behavior.

### Q2: What mistakes do beginners commonly make?

Skipping the fundamentals and jumping into advanced material. We recommend firmly grasping the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this used in practice?

The knowledge of this topic is frequently used in everyday development work. It becomes especially important during code reviews and architectural design.

---

## 14. Summary

| Concept | Key Point |
|---------|-----------|
| Lifetimes | A mechanism by which the compiler tracks the validity period of references |
| 'a annotation | Conveys relationships between references to the compiler |
| Elision rules | Three rules that make annotations unnecessary in most cases |
| Lifetimes on structs | Structs with reference fields require lifetime annotations |
| 'static | Spans the entire program duration. All owned types satisfy it |
| HRTB | `for<'a>` expresses constraints over any lifetime |
| NLL | Lifetimes end at the last point of use |
| Subtyping | `'a: 'b` expresses the outlives relationship |
| Covariance | `&'long T` can be used as `&'short T` |
| GAT | Generic associated types express types that depend on lifetimes |
| Polonius | Next-generation borrow checker (under development) |

---

## What to Read Next

- [01-smart-pointers.md](01-smart-pointers.md) -- Use Box/Rc/Arc to relax lifetime constraints
- [02-closures-fn-traits.md](02-closures-fn-traits.md) -- The relationship between closures and lifetimes
- [03-unsafe-rust.md](03-unsafe-rust.md) -- The dangers of bypassing lifetime checking with unsafe

---

## References

1. **The Rust Programming Language - Ch.10.3 Validating References with Lifetimes** -- https://doc.rust-lang.org/book/ch10-03-lifetime-syntax.html
2. **The Rustonomicon - Lifetimes** -- https://doc.rust-lang.org/nomicon/lifetimes.html
3. **Common Rust Lifetime Misconceptions (pretzelhammer)** -- https://github.com/pretzelhammer/rust-blog/blob/master/posts/common-rust-lifetime-misconceptions.md
4. **Rust Reference - Lifetime Elision** -- https://doc.rust-lang.org/reference/lifetime-elision.html
5. **Rust RFC 2094 - Non-Lexical Lifetimes** -- https://rust-lang.github.io/rfcs/2094-nll.html
6. **Polonius - Next Generation Borrow Checker** -- https://github.com/rust-lang/polonius
