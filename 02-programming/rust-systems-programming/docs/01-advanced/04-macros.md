# Rust Macros — The Power of Metaprogramming

> Systematically learn the mechanisms and practical patterns of declarative and procedural macros that generate code at compile time

## What You Will Learn in This Chapter

1. **Declarative macros (`macro_rules!`)** — How code is expanded via pattern matching, and recursive expansion
2. **Procedural macros (proc-macro)** — Implementation techniques for derive / attribute / function-like macros
3. **Macro design principles** — Hygiene, debugging techniques, and practical best practices


## Prerequisites

You will get more out of this guide if you already have:

- Basic programming knowledge
- An understanding of related foundational concepts
- Familiarity with the contents of [unsafe Rust — Low-Level Programming Across the Safety Boundary](./03-unsafe-rust.md)

---

## 1. Macro Categories and the Big Picture

```
┌─────────────────────────────────────────────────────┐
│                Rust Macro System                    │
├──────────────────────┬──────────────────────────────┤
│  Declarative Macros  │     Procedural Macros        │
│   (Declarative)      │     (Procedural)             │
│                      │                              │
│ Defined with         │  Defined in a separate crate │
│ macro_rules!         │  TokenStream → TokenStream   │
│ Pattern-match based  │                              │
│                      │  ┌────────────────────────┐  │
│ e.g. vec!, println!  │  │ #[derive(..)]          │  │
│                      │  │ #[attribute macro]     │  │
│                      │  │ function_like_macro!() │  │
│                      │  └────────────────────────┘  │
└──────────────────────┴──────────────────────────────┘
```

### When Macros Are Needed

Macros are at the heart of Rust's metaprogramming, and they shine particularly in the following situations:

1. **Reducing boilerplate** — Repeating the same implementation pattern (e.g., `impl` for multiple types, generating test cases)
2. **DSLs (domain-specific languages)** — Structured data literals, declarative API definitions
3. **Compile-time validation** — Static checks for SQL, regular expressions, environment variables
4. **Conditional code generation** — Platform-specific code based on `cfg!`
5. **Automatic derive implementations** — Auto-derivation of traits like `Serialize`, `Debug`, `Clone`

Because macros are expanded **at compile time**, they introduce zero runtime overhead.

### Macro Expansion Phases

```
Source code
    │
    ▼
┌──────────────┐
│   Lexing     │  source → token stream
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Macro      │  macro_rules! / proc-macro run here
│  expansion   │  * expanded recursively over multiple passes
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Name         │  resolve names against the expanded code
│ resolution   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Type         │  type-check the fully expanded code
│ checking     │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Borrow       │  ownership and lifetime verification
│ checking     │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Code         │  LLVM IR → machine code
│ generation   │
└──────────────┘
```

Importantly, macro expansion happens in the **early stages** of compilation. Expanded code is then type-checked and borrow-checked as ordinary Rust code, so any bugs in the code a macro produces will be caught as compile errors.

---

## 2. Declarative Macros (`macro_rules!`)

### Code Example 1: A Basic Declarative Macro

```rust
/// Custom HashMap literal macro
macro_rules! map {
    // Empty map
    () => { std::collections::HashMap::new() };

    // Accept key => value pairs
    ( $( $key:expr => $value:expr ),+ $(,)? ) => {{
        let mut m = std::collections::HashMap::new();
        $( m.insert($key, $value); )+
        m
    }};
}

fn main() {
    let scores = map! {
        "Alice" => 95,
        "Bob"   => 82,
        "Carol" => 91,
    };
    println!("{:?}", scores);
    // {"Carol": 91, "Alice": 95, "Bob": 82}
}
```

### How Pattern Matching Works

`macro_rules!` tries arms (patterns) from top to bottom and expands the first one that matches. Each arm has the form `(pattern) => { expansion template }`.

```rust
macro_rules! explain {
    // Arm 1: empty input
    () => {
        println!("no arguments");
    };

    // Arm 2: a single expression
    ($single:expr) => {
        println!("single expression: {}", $single);
    };

    // Arm 3: comma-separated multiple expressions
    ($first:expr, $($rest:expr),+) => {
        println!("first: {}", $first);
        // Recursively process the rest
        explain!($($rest),+);
    };
}

fn main() {
    explain!();             // "no arguments"
    explain!(42);           // "single expression: 42"
    explain!(1, 2, 3);      // "first: 1" → "first: 2" → "single expression: 3"
}
```

### List of Fragment Specifiers

```
┌──────────────┬──────────────────────────────┬──────────────┐
│ Specifier    │ Matches                      │ Example      │
├──────────────┼──────────────────────────────┼──────────────┤
│ $x:expr      │ expression                   │ 1 + 2        │
│ $x:ty        │ type                         │ Vec<i32>     │
│ $x:ident     │ identifier                   │ my_var       │
│ $x:pat       │ pattern                      │ Some(x)      │
│ $x:path      │ path                         │ std::io::Read│
│ $x:stmt      │ statement                    │ let x = 1;   │
│ $x:block     │ block                        │ { ... }      │
│ $x:item      │ item                         │ fn foo() {}  │
│ $x:meta      │ meta (attribute body)        │ derive(Debug)│
│ $x:tt        │ token tree (anything, one)   │ +, foo, {}   │
│ $x:literal   │ literal                      │ 42, "hello"  │
│ $x:lifetime  │ lifetime                     │ 'a           │
│ $x:vis       │ visibility modifier          │ pub          │
└──────────────┴──────────────────────────────┴──────────────┘
```

### Follow-Set Restrictions on Fragment Specifiers

Fragment specifiers come with restrictions on the tokens that may follow them. Not understanding these will lead to puzzling compile errors.

```rust
// $x:expr may only be followed by => , ;
macro_rules! ok_after_expr {
    ($e:expr, $f:expr) => { $e + $f };      // OK: comma
    // ($e:expr $f:expr) => { $e + $f };     // NG: no separator
}

// $x:ty may only be followed by => , = | ; : > >> [ { as where
macro_rules! ok_after_ty {
    ($t:ty : $e:expr) => { let _: $t = $e; };  // OK: colon
}

// $x:pat may only be followed by => , = | if in
macro_rules! ok_after_pat {
    ($p:pat => $e:expr) => {
        match some_val {
            $p => $e,
            _ => panic!(),
        }
    };
}

// $x:tt matches anything, so it has no follow-set restriction
macro_rules! flexible {
    ($($t:tt)*) => { $($t)* };  // expand whatever was given verbatim
}
```

### Code Example 2: Recursive Macro — Compile-Time Counting

```rust
/// Count tokens at compile time
macro_rules! count {
    ()                          => { 0usize };
    ($head:tt $($tail:tt)*)     => { 1usize + count!($($tail)*) };
}

/// Generate a fixed-size array in a type-safe way
macro_rules! fixed_vec {
    ( $( $elem:expr ),* $(,)? ) => {{
        const LEN: usize = count!($( { $elem } )*);
        let arr: [_; LEN] = [ $( $elem ),* ];
        arr
    }};
}

fn main() {
    let arr = fixed_vec![10, 20, 30];
    // arr is of type [i32; 3]
    assert_eq!(arr.len(), 3);
}
```

### Code Example: Bulk `impl` for Multiple Types (Repetition Macro)

One of the most frequently used patterns in practice is a macro that applies the same trait implementation to multiple types.

```rust
/// Bulk-implement a common trait for numeric types
trait Numeric {
    fn zero() -> Self;
    fn one() -> Self;
    fn is_positive(&self) -> bool;
}

macro_rules! impl_numeric {
    // For integer types
    (int: $($t:ty),+ $(,)?) => {
        $(
            impl Numeric for $t {
                fn zero() -> Self { 0 }
                fn one() -> Self { 1 }
                fn is_positive(&self) -> bool { *self > 0 }
            }
        )+
    };
    // For floating-point types
    (float: $($t:ty),+ $(,)?) => {
        $(
            impl Numeric for $t {
                fn zero() -> Self { 0.0 }
                fn one() -> Self { 1.0 }
                fn is_positive(&self) -> bool { *self > 0.0 }
            }
        )+
    };
}

impl_numeric!(int: i8, i16, i32, i64, i128, isize);
impl_numeric!(int: u8, u16, u32, u64, u128, usize);
impl_numeric!(float: f32, f64);

fn double<T: Numeric + std::ops::Add<Output = T> + Copy>(x: T) -> T {
    x + x
}

fn main() {
    assert_eq!(double(21i32), 42);
    assert_eq!(double(1.5f64), 3.0);
    assert!(42i32.is_positive());
    assert!(!(-1i32).is_positive());
}
```

### Code Example: TT Muncher Pattern (Token-Consuming Pattern)

The TT Muncher is an advanced pattern in declarative macros for parsing complex syntax: it recurses by "eating" one token at a time from the front of the input.

```rust
/// DSL macro that builds an HTML-like structure
macro_rules! html {
    // Base case: empty
    () => { String::new() };

    // Text node
    (text($text:expr) $($rest:tt)*) => {{
        let mut s = $text.to_string();
        s.push_str(&html!($($rest)*));
        s
    }};

    // Self-closing tag <br/>
    ($tag:ident / $($rest:tt)*) => {{
        let mut s = format!("<{}/>", stringify!($tag));
        s.push_str(&html!($($rest)*));
        s
    }};

    // Opening tag + children + closing tag
    ($tag:ident { $($children:tt)* } $($rest:tt)*) => {{
        let mut s = format!("<{}>", stringify!($tag));
        s.push_str(&html!($($children)*));
        s.push_str(&format!("</{}>", stringify!($tag)));
        s.push_str(&html!($($rest)*));
        s
    }};
}

fn main() {
    let page = html! {
        div {
            h1 { text("Hello, World!") }
            p { text("Generating HTML with Rust macros") }
            br /
            p { text("Example of the TT Muncher pattern") }
        }
    };
    println!("{}", page);
    // <div><h1>Hello, World!</h1><p>Generating HTML with Rust macros</p><br/><p>Example of the TT Muncher pattern</p></div>
}
```

### Code Example: Push-down Accumulation Pattern

A pattern that accumulates intermediate results during recursion. The example below is a JSON builder.

```rust
/// Simple JSON builder macro
macro_rules! json {
    // null
    (null) => { JsonValue::Null };

    // boolean
    (true) => { JsonValue::Bool(true) };
    (false) => { JsonValue::Bool(false) };

    // number
    ($num:literal) => { JsonValue::Number($num as f64) };

    // string
    ($s:literal) => { JsonValue::Str($s.to_string()) };

    // array
    ([ $($elem:tt),* $(,)? ]) => {
        JsonValue::Array(vec![ $(json!($elem)),* ])
    };

    // object
    ({ $($key:literal : $value:tt),* $(,)? }) => {{
        let mut map = std::collections::BTreeMap::new();
        $(
            map.insert($key.to_string(), json!($value));
        )*
        JsonValue::Object(map)
    }};
}

#[derive(Debug, Clone, PartialEq)]
enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    Str(String),
    Array(Vec<JsonValue>),
    Object(std::collections::BTreeMap<String, JsonValue>),
}

impl std::fmt::Display for JsonValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JsonValue::Null => write!(f, "null"),
            JsonValue::Bool(b) => write!(f, "{}", b),
            JsonValue::Number(n) => write!(f, "{}", n),
            JsonValue::Str(s) => write!(f, "\"{}\"", s),
            JsonValue::Array(arr) => {
                write!(f, "[")?;
                for (i, v) in arr.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", v)?;
                }
                write!(f, "]")
            }
            JsonValue::Object(map) => {
                write!(f, "{{")?;
                for (i, (k, v)) in map.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "\"{}\": {}", k, v)?;
                }
                write!(f, "}}")
            }
        }
    }
}

fn main() {
    let data = json!({
        "name": "Alice",
        "age": 30,
        "active": true,
        "scores": [95, 87, 92],
        "address": {
            "city": "Tokyo",
            "zip": "100-0001"
        }
    });
    println!("{}", data);
}
```

### Code Example: Macro That Auto-Generates Test Cases

```rust
/// Macro that generates parameterized tests
macro_rules! test_cases {
    (
        fn $test_name:ident($input:ident : $in_ty:ty) -> $out_ty:ty $body:block
        cases {
            $( $case_name:ident : $case_input:expr => $expected:expr ),+ $(,)?
        }
    ) => {
        // Generate the function under test
        fn $test_name($input: $in_ty) -> $out_ty $body

        // Generate each test case as an individual test function
        $(
            #[test]
            fn $case_name() {
                let result = $test_name($case_input);
                assert_eq!(result, $expected,
                    "test case '{}' failed: input={:?}, expected={:?}, actual={:?}",
                    stringify!($case_name), $case_input, $expected, result
                );
            }
        )+
    };
}

test_cases! {
    fn fizzbuzz(n: u32) -> String {
        match (n % 3, n % 5) {
            (0, 0) => "FizzBuzz".to_string(),
            (0, _) => "Fizz".to_string(),
            (_, 0) => "Buzz".to_string(),
            _ => n.to_string(),
        }
    }
    cases {
        test_one:       1  => "1".to_string(),
        test_three:     3  => "Fizz".to_string(),
        test_five:      5  => "Buzz".to_string(),
        test_fifteen:   15 => "FizzBuzz".to_string(),
        test_seven:     7  => "7".to_string(),
    }
}
```

### Scope and Export of `macro_rules!`

```rust
// Macro scoping rules:
// 1. macro_rules! is only usable after its definition (depends on text order)
// 2. A macro defined inside a module can be made public with #[macro_export]
// 3. Macros marked with #[macro_export] are placed at the crate root

// Defining and exporting a macro from within a module
mod utils {
    // This macro is placed at the crate root
    #[macro_export]
    macro_rules! public_macro {
        () => { println!("public macro") };
    }

    // Without macro_export it is only visible within the same module
    macro_rules! private_macro {
        () => { println!("private macro") };
    }

    pub fn use_private() {
        private_macro!();
    }
}

// Bulk import via #[macro_use] (2015-edition style)
// #[macro_use] extern crate some_crate;

// In 2018+ edition you can import with `use`
// use some_crate::public_macro;

// Use $crate to refer to paths correctly
#[macro_export]
macro_rules! create_vec {
    ($($elem:expr),* $(,)?) => {
        {
            // $crate refers to the defining crate
            // It resolves correctly even when used from external crates
            let mut v = $crate::__internal::new_vec();
            $(v.push($elem);)*
            v
        }
    };
}

// Internal module referenced by the macro
#[doc(hidden)]
pub mod __internal {
    pub fn new_vec<T>() -> Vec<T> {
        Vec::new()
    }
}
```

---

## 3. Procedural Macros

### Project Layout

```
┌─ my-project/
│  ├─ Cargo.toml          (workspace)
│  ├─ my-derive/
│  │  ├─ Cargo.toml       ← proc-macro = true
│  │  └─ src/lib.rs       ← macro implementation
│  └─ my-app/
│     ├─ Cargo.toml       ← depends on my-derive
│     └─ src/main.rs
```

### Roles of syn / quote / proc-macro2

```
┌─────────────────────────────────────────────────────────────┐
│         The Three Sacred Tools of Procedural Macros         │
├────────────────┬────────────────────────────────────────────┤
│  proc-macro2   │ Abstraction layer over TokenStream         │
│                │ Provides token-stream types usable in tests│
├────────────────┼────────────────────────────────────────────┤
│  syn           │ TokenStream → AST (syntax tree) parser     │
│                │ Provides types like DeriveInput, ItemFn,   │
│                │ Expr, etc.                                 │
│                │ Returns span-aware errors on parse failure │
├────────────────┼────────────────────────────────────────────┤
│  quote         │ AST → TokenStream code generation          │
│                │ Write Rust code templates with quote!      │
│                │ #var to interpolate, #(#var)* to iterate   │
└────────────────┴────────────────────────────────────────────┘
```

### Example Cargo.toml

```toml
# my-derive/Cargo.toml
[package]
name = "my-derive"
version = "0.1.0"
edition = "2021"

[lib]
proc-macro = true

[dependencies]
syn = { version = "2", features = ["full", "extra-traits"] }
quote = "1"
proc-macro2 = "1"

# For tests (optional)
[dev-dependencies]
trybuild = "1"   # testing compile errors
```

### Code Example 3: Implementing a derive Macro

```rust
// my-derive/Cargo.toml
// [lib]
// proc-macro = true
//
// [dependencies]
// syn = { version = "2", features = ["full"] }
// quote = "1"
// proc-macro2 = "1"

// my-derive/src/lib.rs
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

/// Auto-implement a Describe trait that prints field names and their values
#[proc_macro_derive(Describe)]
pub fn derive_describe(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;

    let fields = if let syn::Data::Struct(data) = &ast.data {
        data.fields.iter().map(|f| {
            let fname = f.ident.as_ref().unwrap();
            quote! {
                write!(f, "  {}: {:?}\n", stringify!(#fname), &self.#fname)?;
            }
        }).collect::<Vec<_>>()
    } else {
        panic!("Describe only supports structs");
    };

    let expanded = quote! {
        impl std::fmt::Display for #name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                writeln!(f, "{}:", stringify!(#name))?;
                #( #fields )*
                Ok(())
            }
        }
    };

    TokenStream::from(expanded)
}

// my-app/src/main.rs
// use my_derive::Describe;
//
// #[derive(Debug, Describe)]
// struct Config {
//     host: String,
//     port: u16,
//     debug: bool,
// }
//
// fn main() {
//     let cfg = Config {
//         host: "localhost".into(),
//         port: 8080,
//         debug: true,
//     };
//     println!("{}", cfg);
//     // Config:
//     //   host: "localhost"
//     //   port: 8080
//     //   debug: true
// }
```

### Code Example: Supporting Enums in a derive Macro

```rust
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields};

/// Auto-implement an EnumName trait that returns the variant name as a string
#[proc_macro_derive(EnumName)]
pub fn derive_enum_name(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;

    let variants = match &ast.data {
        Data::Enum(data) => &data.variants,
        _ => {
            return syn::Error::new_spanned(
                &ast.ident,
                "EnumName can only be applied to enums",
            )
            .to_compile_error()
            .into();
        }
    };

    // Generate a match arm for each variant
    let match_arms = variants.iter().map(|v| {
        let variant_name = &v.ident;
        let name_str = variant_name.to_string();
        match &v.fields {
            Fields::Unit => {
                quote! { #name::#variant_name => #name_str }
            }
            Fields::Unnamed(_) => {
                quote! { #name::#variant_name(..) => #name_str }
            }
            Fields::Named(_) => {
                quote! { #name::#variant_name { .. } => #name_str }
            }
        }
    });

    // List of variant names for the all_names method
    let all_names = variants.iter().map(|v| {
        let name_str = v.ident.to_string();
        quote! { #name_str }
    });

    let expanded = quote! {
        impl #name {
            /// Return the name of this variant as a string
            pub fn variant_name(&self) -> &'static str {
                match self {
                    #(#match_arms,)*
                }
            }

            /// Return a slice of all variant names
            pub fn all_variant_names() -> &'static [&'static str] {
                &[#(#all_names,)*]
            }
        }
    };

    TokenStream::from(expanded)
}

// Usage:
// #[derive(EnumName)]
// enum Color {
//     Red,
//     Green,
//     Blue,
//     Custom(u8, u8, u8),
//     Named { name: String },
// }
//
// fn main() {
//     let c = Color::Custom(255, 0, 128);
//     println!("{}", c.variant_name());  // "Custom"
//     println!("{:?}", Color::all_variant_names());
//     // ["Red", "Green", "Blue", "Custom", "Named"]
// }
```

### Code Example: Adding Helper Attributes to a derive Macro

```rust
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields, Lit, Meta, NestedMeta};

/// Validate derive that auto-generates field validation
/// Supports the helper attribute #[validate(...)]
#[proc_macro_derive(Validate, attributes(validate))]
pub fn derive_validate(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;

    let fields = match &ast.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(f) => &f.named,
            _ => panic!("only named fields are supported"),
        },
        _ => panic!("only structs are supported"),
    };

    let validations = fields.iter().filter_map(|field| {
        let field_name = field.ident.as_ref().unwrap();

        // Look for #[validate(...)] attributes
        let validate_attrs: Vec<_> = field.attrs.iter()
            .filter(|attr| attr.path().is_ident("validate"))
            .collect();

        if validate_attrs.is_empty() {
            return None;
        }

        let mut checks = Vec::new();

        for attr in &validate_attrs {
            // Parse the attribute body (simplified)
            // In real implementations, use syn's parse capabilities
            let meta = attr.parse_args::<Meta>().ok()?;

            match &meta {
                Meta::NameValue(nv) if nv.path.is_ident("min_len") => {
                    if let Lit::Int(lit) = &nv.lit {
                        let min: usize = lit.base10_parse().unwrap();
                        checks.push(quote! {
                            if self.#field_name.len() < #min {
                                errors.push(format!(
                                    "'{}' length is below the minimum {} (actual: {})",
                                    stringify!(#field_name), #min, self.#field_name.len()
                                ));
                            }
                        });
                    }
                }
                Meta::Path(path) if path.is_ident("non_empty") => {
                    checks.push(quote! {
                        if self.#field_name.is_empty() {
                            errors.push(format!(
                                "'{}' must not be empty",
                                stringify!(#field_name)
                            ));
                        }
                    });
                }
                _ => {}
            }
        }

        Some(quote! { #(#checks)* })
    });

    let expanded = quote! {
        impl #name {
            pub fn validate(&self) -> Result<(), Vec<String>> {
                let mut errors = Vec::new();
                #(#validations)*
                if errors.is_empty() {
                    Ok(())
                } else {
                    Err(errors)
                }
            }
        }
    };

    TokenStream::from(expanded)
}

// Usage:
// #[derive(Validate)]
// struct User {
//     #[validate(non_empty)]
//     #[validate(min_len = 3)]
//     name: String,
//
//     #[validate(non_empty)]
//     email: String,
// }
```

### Code Example 4: Attribute Macro

```rust
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn};

/// Attribute macro that measures a function's execution time
#[proc_macro_attribute]
pub fn measure_time(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let name = &func.sig.ident;
    let block = &func.block;
    let sig = &func.sig;
    let vis = &func.vis;
    let attrs = &func.attrs;

    let expanded = quote! {
        #(#attrs)*
        #vis #sig {
            let __start = std::time::Instant::now();
            let __result = (|| #block)();
            let __elapsed = __start.elapsed();
            eprintln!("[measure] {} took {:?}", stringify!(#name), __elapsed);
            __result
        }
    };

    TokenStream::from(expanded)
}

// Usage:
// #[measure_time]
// fn heavy_computation() -> u64 {
//     (0..1_000_000).sum()
// }
```

### Code Example: Attribute Macro That Takes Arguments

```rust
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn, LitInt};

/// Attribute macro that specifies a retry count
/// Used like #[retry(3)]
#[proc_macro_attribute]
pub fn retry(attr: TokenStream, item: TokenStream) -> TokenStream {
    let max_retries = parse_macro_input!(attr as LitInt);
    let max_retries_val: usize = max_retries.base10_parse().unwrap();
    let func = parse_macro_input!(item as ItemFn);

    let name = &func.sig.ident;
    let vis = &func.vis;
    let inputs = &func.sig.inputs;
    let output = &func.sig.output;
    let block = &func.block;
    let attrs = &func.attrs;

    // Assumes the return type is Result
    let expanded = quote! {
        #(#attrs)*
        #vis fn #name(#inputs) #output {
            let mut __attempts = 0usize;
            loop {
                __attempts += 1;
                let __result = (|| #block)();
                match __result {
                    Ok(val) => return Ok(val),
                    Err(e) => {
                        if __attempts >= #max_retries_val {
                            eprintln!(
                                "[retry] {} still failed after {} retries: {:?}",
                                stringify!(#name), #max_retries_val, e
                            );
                            return Err(e);
                        }
                        eprintln!(
                            "[retry] {} attempt {}/{} failed, retrying...",
                            stringify!(#name), __attempts, #max_retries_val
                        );
                        std::thread::sleep(std::time::Duration::from_millis(
                            100 * __attempts as u64
                        ));
                    }
                }
            }
        }
    };

    TokenStream::from(expanded)
}

// Usage:
// #[retry(3)]
// fn fetch_data(url: &str) -> Result<String, Box<dyn std::error::Error>> {
//     let response = reqwest::blocking::get(url)?;
//     Ok(response.text()?)
// }
```

### Code Example 5: Function-like Macro

```rust
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, LitStr};

/// Validate SQL statements at compile time (simplified)
#[proc_macro]
pub fn checked_sql(input: TokenStream) -> TokenStream {
    let sql_lit = parse_macro_input!(input as LitStr);
    let sql = sql_lit.value();

    // Simple validation
    let upper = sql.to_uppercase();
    if !upper.starts_with("SELECT")
        && !upper.starts_with("INSERT")
        && !upper.starts_with("UPDATE")
        && !upper.starts_with("DELETE")
    {
        return syn::Error::new(
            sql_lit.span(),
            "SQL must start with SELECT/INSERT/UPDATE/DELETE",
        )
        .to_compile_error()
        .into();
    }

    let expanded = quote! { #sql };
    TokenStream::from(expanded)
}

// Usage:
// let query = checked_sql!("SELECT * FROM users WHERE id = $1");
// let bad   = checked_sql!("DROP TABLE users"); // compile error!
```

### Code Example: Function-like Macro That Parses Structured Input

```rust
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse::{Parse, ParseStream}, Ident, LitStr, Token, punctuated::Punctuated};

/// Function-like macro that defines a routing table
/// routes! {
///     GET "/users"        => list_users,
///     POST "/users"       => create_user,
///     GET "/users/{id}"   => get_user,
/// }

struct Route {
    method: Ident,
    path: LitStr,
    handler: Ident,
}

impl Parse for Route {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let method: Ident = input.parse()?;
        let path: LitStr = input.parse()?;
        input.parse::<Token![=>]>()?;
        let handler: Ident = input.parse()?;
        Ok(Route { method, path, handler })
    }
}

struct Routes {
    routes: Punctuated<Route, Token![,]>,
}

impl Parse for Routes {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let routes = Punctuated::parse_terminated(input)?;
        Ok(Routes { routes })
    }
}

#[proc_macro]
pub fn routes(input: TokenStream) -> TokenStream {
    let routes = syn::parse_macro_input!(input as Routes);

    let route_entries = routes.routes.iter().map(|r| {
        let method_str = r.method.to_string();
        let path = &r.path;
        let handler = &r.handler;
        quote! {
            Router::route(#method_str, #path, #handler)
        }
    });

    let expanded = quote! {
        {
            let mut router = Router::new();
            #(router.add(#route_entries);)*
            router
        }
    };

    TokenStream::from(expanded)
}
```

---

## 4. Debugging Macros

```
┌─────────────────────────────────────────────────────────┐
│                Macro Debugging Flow                     │
│                                                         │
│  1. cargo expand        ← view expansion result as Rust │
│     └─ cargo install cargo-expand                       │
│                                                         │
│  2. trace_macros!(true) ← trace declarative expansion   │
│     └─ #![feature(trace_macros)] (nightly)              │
│                                                         │
│  3. eprintln! in proc   ← stderr from inside proc macros│
│     └─ printed at compile time                          │
│                                                         │
│  4. syn::Error          ← span-aware error to the IDE   │
│     └─ .to_compile_error().into()                       │
└─────────────────────────────────────────────────────────┘
```

### Code Example 6: Using cargo expand

```bash
# Install
cargo install cargo-expand

# Expand only a particular function
cargo expand --lib -- some_module::some_fn

# Expand including test code
cargo expand --tests

# Inspect a specific derive only
cargo expand --lib | grep -A 50 "impl Display"

# Expand only a specific item
cargo expand --lib ::my_module::MyStruct
```

### Debugging Techniques for Procedural Macros

```rust
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

#[proc_macro_derive(MyDerive)]
pub fn my_derive(input: TokenStream) -> TokenStream {
    // Debug 1: print the input token stream to stderr
    eprintln!("=== Input TokenStream ===");
    eprintln!("{}", input.to_string());

    let ast = parse_macro_input!(input as DeriveInput);

    // Debug 2: inspect the parsed AST
    // requires syn's "extra-traits" feature
    eprintln!("=== Parsed AST ===");
    eprintln!("{:#?}", ast);

    let name = &ast.ident;
    let expanded = quote! {
        impl #name {
            pub fn hello() {
                println!("Hello from {}", stringify!(#name));
            }
        }
    };

    // Debug 3: inspect the generated code
    eprintln!("=== Generated Code ===");
    eprintln!("{}", expanded.to_string());

    TokenStream::from(expanded)
}
```

### Compile-Error Testing with trybuild

The `trybuild` crate is helpful for verifying that procedural macros emit the right error messages.

```rust
// tests/ui.rs
#[test]
fn ui_tests() {
    let t = trybuild::TestCases::new();
    // Tests expected to compile successfully
    t.pass("tests/ui/pass_*.rs");
    // Tests expected to fail to compile (also verifies the error message)
    t.compile_fail("tests/ui/fail_*.rs");
}

// tests/ui/pass_basic.rs
// use my_derive::EnumName;
//
// #[derive(EnumName)]
// enum Color {
//     Red,
//     Green,
//     Blue,
// }
//
// fn main() {
//     let c = Color::Red;
//     assert_eq!(c.variant_name(), "Red");
// }

// tests/ui/fail_struct.rs
// use my_derive::EnumName;
//
// #[derive(EnumName)]
// struct NotAnEnum {  // error: EnumName can only be applied to enums
//     field: i32,
// }
//
// fn main() {}

// tests/ui/fail_struct.stderr (expected error message)
// error: EnumName can only be applied to enums
//   --> tests/ui/fail_struct.rs:4:8
//    |
//  4 | struct NotAnEnum {
//    |        ^^^^^^^^^^
```

### Tracing Expansion with trace_macros!

```rust
// Requires a nightly compiler
#![feature(trace_macros)]

macro_rules! factorial {
    (1) => { 1u64 };
    ($n:literal) => { $n as u64 * factorial!($n - 1) };
}

fn main() {
    trace_macros!(true);
    let result = factorial!(5);
    trace_macros!(false);
    println!("5! = {}", result);
}

// Compile-time output:
// note: trace_macro
//  --> src/main.rs:9:18
//   |
// 9 |     let result = factorial!(5);
//   |                  ^^^^^^^^^^^^
//   |
//   = note: expanding `factorial! { 5 }`
//   = note: to `5 as u64 * factorial!(5 - 1)`
//   = note: expanding `factorial! { 5 - 1 }`
//   ...
```

---

## 5. Comparing Macros

### Declarative vs Procedural Macros

| Aspect | Declarative (`macro_rules!`) | Procedural (proc-macro) |
|---|---|---|
| Where defined | In the same crate | In a separate crate (`proc-macro = true`) |
| Input | Pattern matching | `TokenStream` |
| Dependencies | None | `syn`, `quote`, `proc-macro2` |
| Hygiene | Partially guaranteed | Manual management |
| Scope of use | Expressions, statements, items | derive / attribute / function-like |
| Debugging | `trace_macros!` | `eprintln!` + `cargo expand` |
| Learning curve | Low to medium | Medium to high |
| Use cases | DSLs, utilities | Auto implementations, code generation |
| Compile time | Small impact | Extra cost from dependent crates |
| IDE support | Expansion is hard to inspect | Partially supported by rust-analyzer |

### Major syn Types

| Type | Purpose | Example |
|---|---|---|
| `DeriveInput` | The whole input of a derive macro | A struct/enum definition |
| `ItemFn` | A function definition | Wrapping a function in an attribute macro |
| `ItemStruct` | A struct definition | Transforming structs |
| `ItemEnum` | An enum definition | Analyzing enums |
| `Fields` | A set of fields | Named / Unnamed / Unit |
| `Expr` | An arbitrary expression | `parse_quote!(x + 1)` |
| `Type` | A type representation | `Vec<String>` |
| `LitStr` | A string literal | `"hello"` |
| `LitInt` | An integer literal | `42` |
| `Ident` | An identifier | A variable or function name |
| `Path` | A path | `std::io::Result` |
| `Attribute` | An attribute | `#[derive(Debug)]` |
| `Generics` | Generic parameters | `<T: Clone>` |
| `WhereClause` | A where clause | `where T: Debug` |

### Interpolation Syntax of `quote!`

```rust
use quote::quote;
use syn::Ident;
use proc_macro2::Span;

// Interpolating a single variable
let name = Ident::new("MyStruct", Span::call_site());
let tokens = quote! { struct #name {} };
// → struct MyStruct {}

// Repetition interpolation
let fields = vec!["x", "y", "z"];
let field_idents: Vec<_> = fields.iter()
    .map(|f| Ident::new(f, Span::call_site()))
    .collect();
let tokens = quote! {
    struct Point {
        #(pub #field_idents: f64,)*
    }
};
// → struct Point { pub x: f64, pub y: f64, pub z: f64, }

// Repetition with a separator
let values = vec![1i32, 2, 3];
let tokens = quote! {
    let sum = #(#values)+*;
};
// → let sum = 1 + 2 + 3;

// Nested repetition
let methods = vec![("get_x", "x"), ("get_y", "y")];
let method_tokens: Vec<_> = methods.iter().map(|(method, field)| {
    let method_ident = Ident::new(method, Span::call_site());
    let field_ident = Ident::new(field, Span::call_site());
    quote! {
        pub fn #method_ident(&self) -> f64 {
            self.#field_ident
        }
    }
}).collect();

let tokens = quote! {
    impl Point {
        #(#method_tokens)*
    }
};

// Generate identifiers dynamically with format_ident!
let prefix = "get";
let field = "name";
let getter = quote::format_ident!("{}_{}", prefix, field);
// → get_name
```

---

## 6. Antipatterns

### Antipattern 1: Overusing Macros

```rust
// NG: turning into a macro something a function or generics handle just fine
macro_rules! add_numbers {
    ($a:expr, $b:expr) => { $a + $b };
}

// OK: an ordinary function is enough
fn add<T: std::ops::Add<Output = T>>(a: T, b: T) -> T {
    a + b
}

// When macros are appropriate:
// - Repeated patterns spanning multiple types (impl Trait for A, B, C...)
// - Compile-time computation/validation
// - DSLs (domain-specific languages)
// - Boilerplate reduction (when derive can't do it)
```

### Antipattern 2: Capturing Variables That Ignore Hygiene

```rust
// NG: a macro that depends on outer variable names
macro_rules! bad_log {
    ($msg:expr) => {
        // assumes a `logger` exists in the caller's scope — dangerous!
        logger.log($msg);
    };
}

// OK: pass required dependencies explicitly as arguments
macro_rules! good_log {
    ($logger:expr, $msg:expr) => {
        $logger.log($msg);
    };
}

// OK: generate hygienic variable names from a procedural macro
// quote! {
//     let __internal_logger = get_logger();
//     __internal_logger.log(#msg);
// }
```

### Antipattern 3: Multiple Evaluation of Expressions

```rust
// NG: $val is evaluated multiple times
macro_rules! bad_max {
    ($a:expr, $b:expr) => {
        if $a > $b { $a } else { $b }
    };
}

// Problem: an expression with side effects is executed twice
// bad_max!(expensive_call(), another_call());
// → if expensive_call() > another_call() { expensive_call() } else { another_call() }
// expensive_call() may be called twice!

// OK: bind once to a variable
macro_rules! good_max {
    ($a:expr, $b:expr) => {{
        let __a = $a;
        let __b = $b;
        if __a > __b { __a } else { __b }
    }};
}
```

### Antipattern 4: Macros That Hide Type Mismatches

```rust
// NG: a macro that performs an implicit type conversion
macro_rules! bad_convert {
    ($val:expr) => {
        $val as i64  // information loss is hidden
    };
}

// OK: an explicit conversion function
fn safe_convert(val: u32) -> i64 {
    i64::from(val)  // safely convert via the From trait
}

// OK: make failure explicit with TryFrom
fn try_convert(val: i64) -> Result<u32, std::num::TryFromIntError> {
    u32::try_from(val)
}
```

### Antipattern 5: Giant Macros

```rust
// NG: a macro spanning hundreds of lines — unreadable, undebuggable, no IDE support

// OK: keep the macro a thin wrapper and delegate the implementation to a helper function
macro_rules! register_handler {
    ($name:ident, $method:expr, $path:expr) => {
        // The macro is just glue
        fn $name(registry: &mut Registry) {
            // Delegate the real logic to a regular function
            __register_handler_impl(registry, $method, $path, $name::handler);
        }
    };
}

// The helper is normal Rust code, so IDE support works
fn __register_handler_impl(
    registry: &mut Registry,
    method: &str,
    path: &str,
    handler: fn(&Request) -> Response,
) {
    registry.add_route(method, path, handler);
}
```

---

## 7. In Practice: Auto-Generating the Builder Pattern

### Code Example 7: A Simple derive(Builder) Implementation

```rust
// builder-derive/src/lib.rs
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Data, Fields};

#[proc_macro_derive(Builder)]
pub fn derive_builder(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;
    let builder_name = syn::Ident::new(
        &format!("{}Builder", name),
        name.span(),
    );

    let fields = match &ast.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(f) => &f.named,
            _ => panic!("only named fields are supported"),
        },
        _ => panic!("only structs are supported"),
    };

    // Builder fields (all are Option<T>)
    let builder_fields = fields.iter().map(|f| {
        let name = &f.ident;
        let ty = &f.ty;
        quote! { #name: Option<#ty> }
    });

    // Setter methods
    let setters = fields.iter().map(|f| {
        let name = &f.ident;
        let ty = &f.ty;
        quote! {
            pub fn #name(mut self, val: #ty) -> Self {
                self.#name = Some(val);
                self
            }
        }
    });

    // Field extraction in the build method
    let build_fields = fields.iter().map(|f| {
        let name = &f.ident;
        quote! {
            #name: self.#name.ok_or(
                format!("field '{}' is unset", stringify!(#name))
            )?
        }
    });

    // Default values for the Builder (all None)
    let none_fields = fields.iter().map(|f| {
        let name = &f.ident;
        quote! { #name: None }
    });

    let expanded = quote! {
        pub struct #builder_name {
            #( #builder_fields, )*
        }

        impl #name {
            pub fn builder() -> #builder_name {
                #builder_name {
                    #( #none_fields, )*
                }
            }
        }

        impl #builder_name {
            #( #setters )*

            pub fn build(self) -> Result<#name, String> {
                Ok(#name {
                    #( #build_fields, )*
                })
            }
        }
    };

    TokenStream::from(expanded)
}
```

### Code Example: Adding Defaults and Validation to the Builder Pattern

```rust
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Data, Fields, Lit, Meta};

/// Extended Builder — supports defaults and validation
/// Specify a default value with #[builder(default = "value")]
/// Mark a field as required with #[builder(required)]
#[proc_macro_derive(ExtBuilder, attributes(builder))]
pub fn derive_ext_builder(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;
    let builder_name = quote::format_ident!("{}Builder", name);

    let fields = match &ast.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(f) => &f.named,
            _ => return err(name, "only named fields are supported"),
        },
        _ => return err(name, "only structs are supported"),
    };

    let mut builder_field_defs = Vec::new();
    let mut setter_defs = Vec::new();
    let mut build_assigns = Vec::new();
    let mut default_assigns = Vec::new();

    for field in fields.iter() {
        let field_name = field.ident.as_ref().unwrap();
        let field_ty = &field.ty;

        // Parse the #[builder(...)] attribute
        let mut has_default = false;
        let mut default_expr = None;

        for attr in &field.attrs {
            if !attr.path().is_ident("builder") {
                continue;
            }
            // Parse the attribute body (simplified)
            // In real code use syn::parse for a more robust solution
            if let Ok(Meta::NameValue(nv)) = attr.parse_args::<Meta>() {
                if nv.path.is_ident("default") {
                    has_default = true;
                    if let Lit::Str(s) = &nv.lit {
                        default_expr = Some(s.value());
                    }
                }
            }
        }

        builder_field_defs.push(quote! {
            #field_name: Option<#field_ty>
        });

        setter_defs.push(quote! {
            pub fn #field_name(mut self, val: #field_ty) -> Self {
                self.#field_name = Some(val);
                self
            }
        });

        if has_default {
            if let Some(expr_str) = default_expr {
                let expr: proc_macro2::TokenStream = expr_str.parse().unwrap();
                build_assigns.push(quote! {
                    #field_name: self.#field_name.unwrap_or_else(|| #expr)
                });
            } else {
                build_assigns.push(quote! {
                    #field_name: self.#field_name.unwrap_or_default()
                });
            }
        } else {
            build_assigns.push(quote! {
                #field_name: self.#field_name.ok_or_else(||
                    format!("required field '{}' is unset", stringify!(#field_name))
                )?
            });
        }

        default_assigns.push(quote! { #field_name: None });
    }

    let expanded = quote! {
        pub struct #builder_name {
            #(#builder_field_defs,)*
        }

        impl #name {
            pub fn builder() -> #builder_name {
                #builder_name {
                    #(#default_assigns,)*
                }
            }
        }

        impl #builder_name {
            #(#setter_defs)*

            pub fn build(self) -> Result<#name, String> {
                Ok(#name {
                    #(#build_assigns,)*
                })
            }
        }
    };

    TokenStream::from(expanded)
}

fn err(ident: &syn::Ident, msg: &str) -> TokenStream {
    syn::Error::new_spanned(ident, msg)
        .to_compile_error()
        .into()
}

// Usage:
// #[derive(ExtBuilder)]
// struct ServerConfig {
//     host: String,                           // required
//     port: u16,                              // required
//     #[builder(default = "30")]
//     timeout_secs: u64,                      // default: 30
//     #[builder(default = "true")]
//     tls_enabled: bool,                      // default: true
//     #[builder(default = "String::from(\"INFO\")")]
//     log_level: String,                      // default: "INFO"
// }
//
// fn main() {
//     let config = ServerConfig::builder()
//         .host("0.0.0.0".to_string())
//         .port(8080)
//         // timeout_secs, tls_enabled, log_level use their defaults
//         .build()
//         .unwrap();
// }
```

---

## 8. In Practice: Advanced Macro Patterns

### Code Example: enum_dispatch Pattern (Auto-Generating Static Dispatch)

```rust
/// Convert dynamic dispatch through trait objects into static dispatch
/// A simple implementation of the enum_dispatch! pattern
macro_rules! enum_dispatch {
    (
        trait $trait_name:ident {
            $(fn $method:ident(&self $(, $arg:ident : $arg_ty:ty)*) -> $ret:ty;)*
        }

        enum $enum_name:ident {
            $($variant:ident($inner:ty)),+ $(,)?
        }
    ) => {
        // Trait definition
        trait $trait_name {
            $(fn $method(&self $(, $arg: $arg_ty)*) -> $ret;)*
        }

        // Enum definition
        enum $enum_name {
            $($variant($inner),)+
        }

        // From implementations
        $(
            impl From<$inner> for $enum_name {
                fn from(v: $inner) -> Self {
                    $enum_name::$variant(v)
                }
            }
        )+

        // Trait implementation (delegates each variant via match)
        impl $trait_name for $enum_name {
            $(
                fn $method(&self $(, $arg: $arg_ty)*) -> $ret {
                    match self {
                        $($enum_name::$variant(inner) => inner.$method($($arg),*),)+
                    }
                }
            )*
        }
    };
}

// Usage
enum_dispatch! {
    trait Shape {
        fn area(&self) -> f64;
        fn name(&self) -> &'static str;
    }

    enum AnyShape {
        Circle(Circle),
        Rectangle(Rectangle),
        Triangle(Triangle),
    }
}

struct Circle { radius: f64 }
impl Shape for Circle {
    fn area(&self) -> f64 { std::f64::consts::PI * self.radius * self.radius }
    fn name(&self) -> &'static str { "Circle" }
}

struct Rectangle { width: f64, height: f64 }
impl Shape for Rectangle {
    fn area(&self) -> f64 { self.width * self.height }
    fn name(&self) -> &'static str { "Rectangle" }
}

struct Triangle { base: f64, height: f64 }
impl Shape for Triangle {
    fn area(&self) -> f64 { 0.5 * self.base * self.height }
    fn name(&self) -> &'static str { "Triangle" }
}

fn print_area(shape: &AnyShape) {
    // Static dispatch via enum match instead of dyn Shape
    println!("{}: area = {:.2}", shape.name(), shape.area());
}

fn main() {
    let shapes: Vec<AnyShape> = vec![
        Circle { radius: 5.0 }.into(),
        Rectangle { width: 4.0, height: 6.0 }.into(),
        Triangle { base: 3.0, height: 8.0 }.into(),
    ];

    for shape in &shapes {
        print_area(shape);
    }
    // Circle: area = 78.54
    // Rectangle: area = 24.00
    // Triangle: area = 12.00
}
```

### Code Example: Macro-izing the Type-State Pattern

```rust
/// A macro that auto-generates a type-state pattern
/// Prevents invalid state transitions at compile time
macro_rules! state_machine {
    (
        machine $machine:ident {
            states: [ $($state:ident),+ $(,)? ],
            transitions: [
                $($from:ident -> $to:ident : $event:ident ($($arg:ident : $arg_ty:ty),*) ),+ $(,)?
            ]
        }
    ) => {
        // State types (zero-sized types)
        $(
            #[derive(Debug)]
            pub struct $state;
        )+

        // The state machine itself
        #[derive(Debug)]
        pub struct $machine<State> {
            _state: std::marker::PhantomData<State>,
            // In practice, data fields go here
            data: String,
        }

        // Initial-state construction
        impl $machine<$($state)?> {
            // Only for the very first state
        }

        // Generate each transition method
        $(
            impl $machine<$from> {
                pub fn $event(self $(, $arg: $arg_ty)*) -> $machine<$to> {
                    println!("[state] {} -> {} (event: {})",
                        stringify!($from),
                        stringify!($to),
                        stringify!($event)
                    );
                    $machine {
                        _state: std::marker::PhantomData,
                        data: self.data,
                    }
                }
            }
        )+
    };
}

// Usage: state machine for an HTTP request
state_machine! {
    machine HttpRequest {
        states: [Created, HeadersSent, BodySent, Complete],
        transitions: [
            Created -> HeadersSent : send_headers(headers: Vec<(String, String)>),
            HeadersSent -> BodySent : send_body(body: Vec<u8>),
            BodySent -> Complete : finish(),
        ]
    }
}

impl HttpRequest<Created> {
    pub fn new(url: &str) -> Self {
        HttpRequest {
            _state: std::marker::PhantomData,
            data: url.to_string(),
        }
    }
}

fn main() {
    let req = HttpRequest::new("https://example.com");
    let req = req.send_headers(vec![
        ("Content-Type".into(), "application/json".into()),
    ]);
    let req = req.send_body(b"{}".to_vec());
    let _req = req.finish();

    // Compile error: cannot call send_body directly from the Created state
    // let req = HttpRequest::new("https://example.com");
    // let req = req.send_body(b"{}".to_vec()); // ERROR!
}
```

### Code Example: A Type-Safe Bitflags Macro

```rust
/// Macro that auto-generates bitflag types
macro_rules! bitflags {
    (
        $(#[$outer:meta])*
        $vis:vis struct $name:ident : $repr:ty {
            $(
                $(#[$inner:meta])*
                const $flag:ident = $value:expr;
            )*
        }
    ) => {
        $(#[$outer])*
        #[derive(Clone, Copy, PartialEq, Eq, Hash)]
        $vis struct $name {
            bits: $repr,
        }

        impl $name {
            $(
                $(#[$inner])*
                pub const $flag: $name = $name { bits: $value };
            )*

            /// An empty flag set
            pub const fn empty() -> Self {
                $name { bits: 0 }
            }

            /// A set with all flags enabled
            pub const fn all() -> Self {
                $name { bits: $( $value )|* }
            }

            /// Whether the given flag is contained
            pub const fn contains(self, other: Self) -> bool {
                (self.bits & other.bits) == other.bits
            }

            /// Whether the flag set is empty
            pub const fn is_empty(self) -> bool {
                self.bits == 0
            }

            /// Get the raw bits value
            pub const fn bits(self) -> $repr {
                self.bits
            }

            /// Add a flag
            pub fn insert(&mut self, other: Self) {
                self.bits |= other.bits;
            }

            /// Remove a flag
            pub fn remove(&mut self, other: Self) {
                self.bits &= !other.bits;
            }

            /// Toggle a flag
            pub fn toggle(&mut self, other: Self) {
                self.bits ^= other.bits;
            }
        }

        impl std::ops::BitOr for $name {
            type Output = Self;
            fn bitor(self, rhs: Self) -> Self {
                $name { bits: self.bits | rhs.bits }
            }
        }

        impl std::ops::BitAnd for $name {
            type Output = Self;
            fn bitand(self, rhs: Self) -> Self {
                $name { bits: self.bits & rhs.bits }
            }
        }

        impl std::ops::BitOrAssign for $name {
            fn bitor_assign(&mut self, rhs: Self) {
                self.bits |= rhs.bits;
            }
        }

        impl std::ops::Not for $name {
            type Output = Self;
            fn not(self) -> Self {
                $name { bits: !self.bits & Self::all().bits }
            }
        }

        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                let mut first = true;
                write!(f, "{}(", stringify!($name))?;
                $(
                    if self.contains($name::$flag) {
                        if !first { write!(f, " | ")?; }
                        write!(f, stringify!($flag))?;
                        first = false;
                    }
                )*
                if first {
                    write!(f, "empty")?;
                }
                write!(f, ")")
            }
        }
    };
}

// Usage
bitflags! {
    /// File access permissions
    pub struct Permissions: u32 {
        /// Read permission
        const READ    = 0b001;
        /// Write permission
        const WRITE   = 0b010;
        /// Execute permission
        const EXECUTE = 0b100;
    }
}

fn main() {
    let mut perms = Permissions::READ | Permissions::WRITE;
    println!("{:?}", perms);
    // Permissions(READ | WRITE)

    assert!(perms.contains(Permissions::READ));
    assert!(!perms.contains(Permissions::EXECUTE));

    perms.insert(Permissions::EXECUTE);
    println!("{:?}", perms);
    // Permissions(READ | WRITE | EXECUTE)

    perms.remove(Permissions::WRITE);
    println!("{:?}", perms);
    // Permissions(READ | EXECUTE)

    let readonly = !Permissions::WRITE & Permissions::all();
    println!("{:?}", readonly);
    // Permissions(READ | EXECUTE)
}
```

---

## 9. A Deep Dive on Hygiene

### Hygiene of Declarative Macros

Rust's `macro_rules!` macros are "partially hygienic." Identifiers introduced inside a macro do not collide with the caller's scope, but type names and trait names may not always be hygienic.

```rust
macro_rules! hygiene_demo {
    () => {
        // This `x` is a variable introduced inside the macro
        // It lives in a different scope than the caller's `x`
        let x = 42;
        println!("x inside the macro: {}", x);
    };
}

fn main() {
    let x = 100;
    hygiene_demo!();
    // x inside the macro: 42
    println!("x in main: {}", x);
    // x in main: 100
    // → no collision (hygienic)
}
```

### Hygiene Management for Procedural Macros

```rust
use quote::quote;
use proc_macro2::Span;

// Techniques for generating hygienic variable names

// Approach 1: prefix with __ to avoid collisions (idiomatic)
let var = quote! { __internal_counter };

// Approach 2: Span::call_site() — resolved in the caller's scope
let name = syn::Ident::new("result", Span::call_site());
// → same namespace as `result` in the caller's scope

// Approach 3: Span::mixed_site() — hygiene closer to declarative macros
let name = syn::Ident::new("result", Span::mixed_site());
// → local variables are hygienic; path resolution uses the caller's scope

// Best practice: use unlikely-to-collide names for internal variables
let expanded = quote! {
    {
        let __macro_internal_value = compute();
        let __macro_internal_guard = lock.lock().unwrap();
        use_value(__macro_internal_value, &__macro_internal_guard)
    }
};
```

---

## 10. Performance and Compile Time

### How Macros Affect Compile Time

```
┌────────────────────────────────────────────────────────┐
│        Relationship Between Macros and Compile Time    │
├──────────────────┬─────────────────────────────────────┤
│ macro_rules!     │ - Expansion is fast (just pattern  │
│                  │   matching)                         │
│                  │ - Deep recursion gets slow          │
│                  │ - Recursion limit:                  │
│                  │   #![recursion_limit="256"]         │
├──────────────────┼─────────────────────────────────────┤
│ proc-macro       │ - Requires compiling another crate  │
│                  │ - More syn features = slower        │
│                  │ - Initial build is especially slow  │
├──────────────────┼─────────────────────────────────────┤
│ Expanded code    │ - Very large expansions slow type   │
│                  │   checking                          │
│                  │ - Watch out for monomorphization    │
│                  │   costs                             │
└──────────────────┴─────────────────────────────────────┘
```

### Optimizing Compile Time

```toml
# Cargo.toml — keep syn features to a minimum
[dependencies]
syn = { version = "2", features = ["derive", "parsing"] }
# Don't use "full" (limit what you parse)

# Take advantage of the proc-macro cache
# Splitting out the proc-macro crate avoids recompiling it
# whenever application code changes
```

```rust
// Control the depth of recursive macros
// Raise the limit if you really need deep recursion
#![recursion_limit = "512"]

// However, recursion depth can sometimes be reduced from O(n) to O(log n)
// Example: optimizing a count macro

// O(n) version — recursion depth scales with the number of elements
macro_rules! count_linear {
    () => { 0usize };
    ($x:tt $($xs:tt)*) => { 1 + count_linear!($($xs)*) };
}

// O(log n) version — binary partitioning halves the recursion depth
macro_rules! count_log {
    () => { 0usize };
    ($x:tt) => { 1usize };
    ($($a:tt)+ ; $($b:tt)+) => {
        count_log!($($a)+) + count_log!($($b)+)
    };
    ($a:tt $($rest:tt)*) => {
        count_log!($a) + count_log!($($rest)*)
    };
}
```

---

## FAQ

### Q1: When should I use `macro_rules!` vs `proc-macro`?

**A:** First check whether `macro_rules!` is sufficient. Cases where pattern matching is enough (repetitive expansions, simple DSLs) are best handled by declarative macros. Use procedural macros when you need to analyze type information or transform code structures.

Decision flowchart:

```
Simple repetitive expansion? → Yes → macro_rules!
         ↓ No
DSL (custom syntax)? → Yes → macro_rules! (if simple)
         ↓           → proc-macro (if complex)
Need to analyze type info? → Yes → proc-macro (derive)
         ↓ No
Transform a function/struct? → Yes → proc-macro (attribute)
         ↓ No
Compile-time validation? → Yes → proc-macro (function-like)
```

### Q2: How can I improve macro error messages?

**A:** In procedural macros, use `syn::Error::new_spanned(tokens, message)` to attach a span to the offending token. In declarative macros, use `compile_error!("message")`.

```rust
// Inside a procedural macro
if !is_valid {
    return syn::Error::new_spanned(
        &field.ident,
        "this field must be of type String"
    ).to_compile_error().into();
}

// Conditional error inside a declarative macro
macro_rules! assert_type {
    ($val:expr, String) => { /* OK */ };
    ($val:expr, $other:ty) => {
        compile_error!(concat!(
            "expected type String but got ",
            stringify!($other),
        ));
    };
}

// Returning multiple errors (procedural macro)
fn validate_fields(fields: &[Field]) -> Result<(), TokenStream> {
    let mut errors = Vec::new();
    for field in fields {
        if !is_supported_type(&field.ty) {
            errors.push(syn::Error::new_spanned(
                &field.ty,
                format!("unsupported type: {:?}", field.ty),
            ));
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        // Combine multiple errors
        let mut combined = errors[0].clone();
        for error in &errors[1..] {
            combined.combine(error.clone());
        }
        Err(combined.to_compile_error().into())
    }
}
```

### Q3: How do I interpolate variables inside `quote!`?

**A:** Use `#var` to interpolate a single variable, and `#( #var )*` to interpolate a repetition.

```rust
let name = quote::format_ident!("my_func");
let types = vec![quote!(i32), quote!(String)];

let expanded = quote! {
    fn #name(#( arg: #types ),*) {}
    // → fn my_func(arg: i32, arg: String) {}
};
```

### Q4: How do I write tests for a proc-macro crate?

**A:** Ordinary unit tests cannot live inside a proc-macro crate, so use one of the strategies below.

```rust
// Strategy 1: factor out the logic into a non-proc-macro crate
// my-derive-core/src/lib.rs (a regular library crate)
pub fn generate_impl(ast: &syn::DeriveInput) -> proc_macro2::TokenStream {
    // Put the logic here (testable)
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation() {
        let input: syn::DeriveInput = syn::parse_quote! {
            struct Foo {
                bar: i32,
                baz: String,
            }
        };
        let output = generate_impl(&input);
        let expected = quote::quote! {
            // expected code
        };
        assert_eq!(output.to_string(), expected.to_string());
    }
}

// my-derive/src/lib.rs (proc-macro crate — a thin wrapper)
#[proc_macro_derive(MyDerive)]
pub fn my_derive(input: TokenStream) -> TokenStream {
    let ast = syn::parse_macro_input!(input as syn::DeriveInput);
    my_derive_core::generate_impl(&ast).into()
}

// Strategy 2: compile testing with trybuild (described above)
// Strategy 3: write integration tests that actually use the derive
```

### Q5: How do I handle async fn from a macro?

**A:** Wrapping an `async fn` from an attribute macro requires special care.

```rust
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn};

#[proc_macro_attribute]
pub fn async_measure(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let name = &func.sig.ident;
    let block = &func.block;
    let vis = &func.vis;
    let attrs = &func.attrs;
    let sig = &func.sig;

    // Branch on whether the function is async
    let expanded = if sig.asyncness.is_some() {
        quote! {
            #(#attrs)*
            #vis #sig {
                let __start = std::time::Instant::now();
                let __result = async move #block .await;
                let __elapsed = __start.elapsed();
                eprintln!("[async_measure] {} took {:?}", stringify!(#name), __elapsed);
                __result
            }
        }
    } else {
        quote! {
            #(#attrs)*
            #vis #sig {
                let __start = std::time::Instant::now();
                let __result = (|| #block)();
                let __elapsed = __start.elapsed();
                eprintln!("[measure] {} took {:?}", stringify!(#name), __elapsed);
                __result
            }
        }
    };

    TokenStream::from(expanded)
}
```

---

## Summary

| Topic | Key points |
|---|---|
| Declarative macros | `macro_rules!` with pattern matching. Can be defined in the same crate |
| Procedural macros | Require a separate crate. `syn` + `quote` are the standard tools |
| derive macros | Auto-implement traits with `#[derive(Foo)]`. The most commonly used kind |
| Attribute macros | Use `#[foo]` to transform a function/struct |
| Function-like | Take free-form input via `foo!(...)` |
| Debugging | `cargo expand` is the most important tool |
| Hygiene | Declarative macros are partially hygienic; procedural macros require manual care |
| Design principles | Don't reach for a macro when a function will do. Make error messages friendly |
| Performance | Keep syn features minimal. Watch recursion depth |
| Testing | trybuild + separating logic is the standard approach |

## What to Read Next

- [Async/Await Basics](../02-async/00-async-basics.md) — Rust's asynchronous programming model
- [Testing](../04-ecosystem/01-testing.md) — Testing strategies including how to test macros
- [Best Practices](../04-ecosystem/04-best-practices.md) — API design and the right place to use macros

## References

1. **The Rust Reference — Macros**: https://doc.rust-lang.org/reference/macros.html
2. **The Little Book of Rust Macros**: https://veykril.github.io/tlborm/
3. **syn crate documentation**: https://docs.rs/syn/latest/syn/
4. **Proc-Macro Workshop (dtolnay)**: https://github.com/dtolnay/proc-macro-workshop
5. **quote crate documentation**: https://docs.rs/quote/latest/quote/
6. **Rust API Guidelines — Macros**: https://rust-lang.github.io/api-guidelines/macros.html
