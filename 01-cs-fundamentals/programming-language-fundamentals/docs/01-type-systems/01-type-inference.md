# Type Inference

> **Type inference is the mechanism by which "the compiler automatically determines types without the programmer writing them." It is one of the core technologies in programming language design that achieves both the safety of static typing and the conciseness of dynamic typing.**

## Learning Objectives

- [ ] Understand the mathematical foundations and historical background of type inference
- [ ] Explain the operating principles of the Hindley-Milner type inference algorithm
- [ ] Grasp the mechanism and advantages of bidirectional type checking
- [ ] Compare the scope and characteristics of type inference across major languages (TypeScript, Rust, Go, Haskell, Kotlin, Scala)
- [ ] Judge the appropriate use of type inference versus explicit type annotations
- [ ] Manually execute constraint-based type inference
- [ ] Predict cases where type inference fails and handle them appropriately


## Prerequisites

Before reading this guide, the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related fundamental concepts
- Familiarity with the content of [Static Typing vs Dynamic Typing](./00-static-vs-dynamic.md)

---

## Table of Contents

1. [Foundations and History of Type Inference](#1-foundations-and-history-of-type-inference)
2. [Classification System of Type Inference](#2-classification-system-of-type-inference)
3. [Hindley-Milner Type Inference Algorithm](#3-hindley-milner-type-inference-algorithm)
4. [Bidirectional Type Checking](#4-bidirectional-type-checking)
5. [Detailed Type Inference by Language](#5-detailed-type-inference-by-language)
6. [Limitations of Type Inference and Countermeasures](#6-limitations-of-type-inference-and-countermeasures)
7. [Anti-Patterns and Best Practices](#7-anti-patterns-and-best-practices)
8. [Practical Exercises (3 Levels)](#8-practical-exercises-3-levels)
9. [FAQ (Frequently Asked Questions)](#9-faq-frequently-asked-questions)
10. [Summary and Next Steps](#10-summary-and-next-steps)
11. [References](#11-references)

---

## 1. Foundations and History of Type Inference

### 1.1 What Is Type Inference

Type inference is a mechanism by which the compiler or interpreter automatically determines the types of expressions and variables without the programmer explicitly writing type annotations.

```
Without type annotations (delegating to type inference):
  let x = 42             // Compiler infers x: int

With type annotations (explicitly specified):
  let x: int = 42        // Programmer specifies the type

Both produce the same result. Type inference makes "redundant type annotations" optional.
```

The fundamental value of type inference lies in **improving code readability and conciseness without sacrificing type safety**. It achieves the ease of writing found in dynamically typed languages under the safety guarantees of static typing.

### 1.2 Historical Background

The history of type inference is situated at the intersection of mathematical logic and computer science.

```
Timeline of Type Inference History
================================================================

1958  Curry's type assignment
      └── Automatic type assignment in combinatory logic

1969  Hindley's Principal Type Scheme
      └── Proof that a unique "most general type" exists

1978  Milner's Algorithm W
      └── Efficient type inference algorithm for the ML language

1982  Damas-Milner Type System
      └── Formalization of let-polymorphism

1985  Establishment of the ML language family
      └── Foundation for SML, OCaml

1990  Haskell 1.0
      └── Full implementation of Hindley-Milner + type classes

2004  C# 3.0's var keyword
      └── Beginning of type inference adoption in mainstream languages

2012  TypeScript 0.8
      └── Introduction of type inference to the JavaScript ecosystem

2014  Swift 1.0
      └── Bidirectional type inference in the Apple ecosystem

2015  Rust 1.0
      └── Type inference integrated with the ownership system

2016  Java's var (JEP 286 proposal)
      └── Officially introduced in Java 10 in 2018

2021  Go 1.18 Generics
      └── Addition of type parameter inference

================================================================
```

### 1.3 Basic Operating Principles of Type Inference

Type inference generally operates through the following 5-stage process.

```
Basic Process of Type Inference
================================================================

Step 1: Determine types of literals
  42        --> int
  3.14      --> float
  "hello"   --> string
  true      --> bool

Step 2: Infer variable types
  x = 42    --> x: int (determine left side from right-side type)

Step 3: Infer expression types
  x + 1     --> int + int --> int (apply operator type rules)

Step 4: Infer function types
  f(x) = x + 1  --> f: int -> int (determine argument and return types)

Step 5: Resolve constraints
  For undetermined type variables, solve collected constraints to determine types

================================================================

Concrete example: let result = if condition then 42 else 0

  1. condition: bool (if condition is bool)
  2. 42: int, 0: int (literal types)
  3. then-branch and else-branch types match → int
  4. result: int (type of entire if expression)
```

### 1.4 The Problem Type Inference Solves

In a world without type inference, programmers would have to write type annotations for every expression. This creates severe redundancy, especially with generics and collection operations.

```typescript
// Without type inference (verbose Java 5 style)
Map<String, List<Pair<Integer, String>>> map =
    new HashMap<String, List<Pair<Integer, String>>>();

// With type inference (Java 10+ / TypeScript style)
const map = new HashMap<String, List<Pair<Integer, String>>>();
// Or
let map = new Map<string, [number, string][]>();
```

This difference becomes more pronounced as code lines increase and generic types become more deeply nested.

---

## 2. Classification System of Type Inference

### 2.1 Classification by Direction of Inference

Multiple approaches to type inference exist, depending on which direction information flows.

```
Directional Classification of Type Inference
================================================================

[1] Bottom-up Inference (Synthesis)
    ─────────────────────────────────
    Type information propagates from leaves (literals/variables) to root (entire expression)

    let x = 42          42: int --> x: int
    let y = x + 1       x: int, 1: int, (+): int->int->int --> y: int

    Characteristics: Can infer from local type information only
    Adopted by: Go, C (partial)

[2] Top-down Inference (Checking)
    ─────────────────────────────────
    Expected types propagate from higher context down to subexpressions

    let x: number[] = [1, 2, 3]
                       ↓ Expects number[]
                       Checks that each element is number

    Characteristics: Context-dependent inference possible
    Adopted by: Haskell (partial), Scala 3

[3] Bidirectional Inference
    ─────────────────────────────────
    Combines bottom-up and top-down

    const f: (x: number) => string = x => x.toString();
         ↑ Top-down: x: number        ↑ Bottom-up: string

    Characteristics: Most flexible and practical
    Adopted by: TypeScript, Kotlin, Swift, Scala 3

================================================================
```

### 2.2 Classification by Scope of Inference

```
Inference Scope Classification and Comparison Table
================================================================

Scope            | Description                | Adopted Languages
-----------------+----------------------------+------------------
Local inference  | Function body only         | Go, C++ (auto)
Intra-function   | Including function sig     | TypeScript, Kotlin
Module inference  | Entire module              | Rust (partial)
Global inference  | Entire program             | Haskell, ML, OCaml

================================================================

Local inference:
  Type annotations are required at function boundaries (arguments/return values).
  Inference is confined to each function's body.

  Example (Rust):
    fn add(a: i32, b: i32) -> i32 {  // Boundaries are explicit
        let sum = a + b;              // Local is inferred
        sum
    }

Global inference:
  Determines types from the entire program even without any type annotations.
  The Hindley-Milner algorithm makes this possible.

  Example (Haskell):
    add x y = x + y
    -- Inferred: add :: Num a => a -> a -> a
    -- Fully type-safe even without type annotations
```

### 2.3 Classification by Strength of Inference

| Level | Description | Concrete Example | Languages |
|--------|------|--------|------|
| Level 0 | No inference | All explicit type annotations | C (traditional), Java (<10) |
| Level 1 | Variables only | `var x = 42` | Go, Java 10+, C++ auto |
| Level 2 | Variables + return values | Local function return value inference | TypeScript, Kotlin |
| Level 3 | Variables + return values + context | Callback argument inference | TypeScript, Swift |
| Level 4 | All expressions (with type class constraints) | Full Hindley-Milner | Haskell, ML, OCaml |

---

## 3. Hindley-Milner Type Inference Algorithm

### 3.1 Overview

Hindley-Milner (HM) type inference is the most famous and theoretically complete type inference algorithm. It is based on Hindley's 1969 research and Milner's 1978 Algorithm W, and was formalized as the Damas-Milner type system in 1982.

**Three Key Properties of HM Type Inference:**

1. **Completeness**: For every program that can be typed, a type can always be inferred
2. **Principal Type Property**: The inferred type is always the "most general type"
3. **Decidability**: Inference always terminates in finite time

### 3.2 Algorithm W in Detail

Algorithm W is the standard implementation algorithm for HM type inference.

```
Operating Procedure of Algorithm W
================================================================

Input: Type environment Gamma, expression e
Output: Substitution S, type tau

W(Gamma, e) = case e of

  [Variable] x:
    Look up Gamma(x) and instantiate the type scheme
    → Return a monotype with fresh type variables substituted

  [Application] e1 e2:
    (S1, tau1) = W(Gamma, e1)
    (S2, tau2) = W(S1(Gamma), e2)
    beta = fresh type variable
    S3 = unify(S2(tau1), tau2 → beta)
    → (S3 . S2 . S1, S3(beta))

  [Abstraction] lambda x.e:
    beta = fresh type variable
    (S1, tau1) = W(Gamma ∪ {x: beta}, e)
    → (S1, S1(beta) → tau1)

  [let] let x = e1 in e2:
    (S1, tau1) = W(Gamma, e1)
    sigma = generalize(S1(Gamma), tau1)   ← Polymorphization here
    (S2, tau2) = W(S1(Gamma) ∪ {x: sigma}, e2)
    → (S2 . S1, tau2)

================================================================
```

### 3.3 Unification Algorithm

Unification is the algorithm that finds a substitution that makes two type expressions equal. It forms the heart of HM type inference.

```
Unification algorithm: unify(tau1, tau2)
================================================================

unify(alpha, tau)         = { alpha := tau }     (if alpha does not occur in tau)
unify(tau, alpha)         = { alpha := tau }     (if alpha does not occur in tau)
unify(Int, Int)           = {}                   (same base type)
unify(tau1→tau2, tau3→tau4) = let S1 = unify(tau1, tau3)
                                  S2 = unify(S1(tau2), S1(tau4))
                              in S2 . S1
unify(tau1, tau2)         = error                (cannot unify)

================================================================

Occurs Check:
  unify(alpha, List<alpha>) fails because it would produce an infinite type
  alpha = List<alpha> = List<List<alpha>> = List<List<List<alpha>>> = ...

  * Omitting the occurs check causes infinite loops
```

### 3.4 Concrete Manual Trace Example

We manually trace HM type inference for the following function.

```
Target function: let compose f g x = f (g x)

Step 1: Assign type variables
================================================================
  f: alpha
  g: beta
  x: gamma
  compose: delta

Step 2: Analyze expressions and collect constraints
================================================================
  Expression (g x):
    g is a function so beta = gamma → epsilon  (epsilon is a fresh type variable)
    Type of g x is epsilon

  Expression f (g x):
    f is a function so alpha = epsilon → zeta  (zeta is a fresh type variable)
    Type of f (g x) is zeta

  Type of compose f g x is zeta

Step 3: Resolve constraints (unification)
================================================================
  Constraint list:
    beta = gamma → epsilon
    alpha = epsilon → zeta

  Substitution:
    beta ↦ gamma → epsilon
    alpha ↦ epsilon → zeta

Step 4: Assemble the final type
================================================================
  compose: alpha → beta → gamma → zeta
         = (epsilon → zeta) → (gamma → epsilon) → gamma → zeta

  Rename to standard type variable names:
  compose :: (b -> c) -> (a -> b) -> a -> c

  This is the same type as Haskell's (.) operator.
================================================================
```

### 3.5 Let-Polymorphism

One of the most important features of HM type inference is **let-polymorphism**. This is a mechanism that assigns polymorphic types to values defined with `let` bindings.

```haskell
-- Example of let-polymorphism
let id = \x -> x        -- id :: forall a. a -> a
in (id 42, id "hello")  -- id can be used with both int and string

-- Within a lambda, polymorphization does not occur (monomorphism restriction)
(\id -> (id 42, id "hello")) (\x -> x)
-- ↑ This is a type error!
-- Because id is restricted to a monomorphic type, it cannot be used with both int and string
```

```
Difference between let-polymorphism and monomorphism
================================================================

let expression:
  let id = lambda x.x in ...
  → id has polymorphic type forall alpha. alpha → alpha
  → Both id 42 and id "hello" can be used simultaneously

lambda expression:
  (lambda id. ...) (lambda x.x)
  → id has monomorphic type alpha → alpha
  → alpha is fixed to a single concrete type

This distinction is the key to guaranteeing the decidability of HM type inference.
In System F (rank-2 and above polymorphism), type inference becomes undecidable.

================================================================
```

---

## 4. Bidirectional Type Checking

### 4.1 Overview

Bidirectional type checking is a modern type inference technique that replaces HM type inference. It is widely adopted in practical languages such as TypeScript, Kotlin, Swift, and Scala 3.

**Basic Principle**: Propagate type information in two directions.

```
Two Modes of Bidirectional Type Checking
================================================================

[Synthesis Mode (Infer)]
  Synthesize types from expressions (bottom-up)

  Gamma ⊢ e ⇒ tau   "Under environment Gamma, the type of expression e is inferred as tau"

  Examples:
    42 ⇒ number          Synthesize literal type
    x ⇒ Gamma(x)         Retrieve variable type from environment
    f(e) ⇒ tau2          When f: tau1→tau2 and e ⇐ tau1

[Checking Mode (Check)]
  Verify that an expression has the expected type (top-down)

  Gamma ⊢ e ⇐ tau   "Under environment Gamma, expression e has type tau"

  Examples:
    lambda x.e ⇐ tau1→tau2   Assign tau1 to argument x, check e ⇐ tau2
    if c then e1 else e2 ⇐ tau   Check e1 ⇐ tau and e2 ⇐ tau

================================================================
```

### 4.2 Switching Between Synthesis and Checking Modes

```
Information Flow Diagram of Bidirectional Type Checking
================================================================

const handler: (event: MouseEvent) => void = (e) => {
  console.log(e.clientX);
};

Analysis process:

  1. Read handler's type annotation
     Expected type: (event: MouseEvent) => void
            │
  2. ────── Checking mode ──────────────────────
            │
            ▼
     (e) => { console.log(e.clientX); }
     ⇐ (event: MouseEvent) => void

  3. Assign MouseEvent to e
     e: MouseEvent
            │
  4. ────── Synthesis mode ──────────────────────────
            │
            ▼
     e.clientX ⇒ number    (property of MouseEvent)
     console.log(e.clientX) ⇒ void

  5. Return type check
     void ⇐ void  ✓ Success

================================================================
```

### 4.3 Concrete Example of Bidirectional Type Checking in TypeScript

```typescript
// [Code Example 1] Type inference for callbacks
// Example of bidirectional type checking in action during array method chains

interface User {
    id: number;
    name: string;
    age: number;
    active: boolean;
}

const users: User[] = [
    { id: 1, name: "Alice", age: 30, active: true },
    { id: 2, name: "Bob", age: 25, active: false },
    { id: 3, name: "Charlie", age: 35, active: true },
];

// Chain of bidirectional type checking
const result = users
    .filter(u => u.active)       // u: User (inferred from users' element type)
    .map(u => u.name)            // u: User, return: string
    .map(name => name.length)    // name: string (inferred from previous stage's return type)
    .reduce((sum, len) => sum + len, 0); // sum: number, len: number

// result: number (type inference propagates through the entire chain)
```

### 4.4 Comparison with HM Type Inference

```
HM Type Inference vs Bidirectional Type Checking
================================================================

                    HM Inference          Bidirectional
------------------------------------------------------------
Inference completeness  Complete           Partial (annotations may be needed)
Principal type          Yes                No (depends on annotations)
Subtyping               Difficult          Naturally supported
Overloading             Difficult          Naturally supported
Higher-rank poly        Let-polymorphism   Rank-1 (extensible with annotations)
Implementation          Moderate           Low complexity
Error messages          Somewhat unclear   Clear (direction is known)
Adopted languages       Haskell, ML        TypeScript, Kotlin, Swift

================================================================

Where HM type inference excels:
  - Can fully infer without any type annotations
  - Inferred result is always the "most general type" (principal type)

Where bidirectional type checking excels:
  - Good compatibility with subtyping (inheritance, covariance/contravariance)
  - Naturally handles method overloading
  - Type error messages are easier to understand
  - Implementation is simple and incrementally extensible

================================================================
```

### 4.5 Flow-Sensitive Typing

As an extension of bidirectional type checking, there is type narrowing based on control flow. TypeScript strongly supports this feature.

```typescript
// [Code Example 2] Detailed operation of flow-sensitive typing

function processValue(value: string | number | null | undefined) {
    // At this point: value: string | number | null | undefined

    if (value == null) {
        // After null check: value: null | undefined
        return "no value";
    }
    // Here: value: string | number (null and undefined excluded)

    if (typeof value === "string") {
        // After typeof guard: value: string
        return value.toUpperCase();
    }
    // Here: value: number (string excluded)

    return value.toFixed(2);
}

// Narrowing with discriminated unions
type Shape =
    | { kind: "circle"; radius: number }
    | { kind: "rectangle"; width: number; height: number }
    | { kind: "triangle"; base: number; height: number };

function area(shape: Shape): number {
    switch (shape.kind) {
        case "circle":
            // shape: { kind: "circle"; radius: number }
            return Math.PI * shape.radius ** 2;
        case "rectangle":
            // shape: { kind: "rectangle"; width: number; height: number }
            return shape.width * shape.height;
        case "triangle":
            // shape: { kind: "triangle"; base: number; height: number }
            return (shape.base * shape.height) / 2;
    }
}
```

```
Flow-Sensitive Type State Transition Diagram
================================================================

function example(x: A | B | C) {

  Type of x: A | B | C
  │
  ├─ if (isA(x)) ──────────► x: A ──► return
  │
  │  Type of x: B | C   (A excluded)
  │
  ├─ if (isB(x)) ──────────► x: B ──► return
  │
  │  Type of x: C       (B also excluded)
  │
  └─ else ──────────────────► x: C ──► return

  Types are progressively narrowed at each branch
  (Type state changes along the control flow)

================================================================
```

---

## 5. Detailed Type Inference by Language

### 5.1 TypeScript

TypeScript is a language that adds a type system to JavaScript and provides practical type inference based on bidirectional type checking.

```typescript
// [Code Example 3] Full range of TypeScript's type inference

// --- Local variable inference ---
let x = 42;                    // x: number
let s = "hello";               // s: string
let b = true;                  // b: boolean
let arr = [1, 2, 3];           // arr: number[]
let obj = { name: "Alice" };   // obj: { name: string }
let tuple = [1, "a"] as const; // tuple: readonly [1, "a"]

// --- Function return value inference ---
function add(a: number, b: number) {
    return a + b;   // Return value: inferred as number
}

function greet(name: string) {
    return `Hello, ${name}!`;  // Return value: inferred as string
}

// --- Function arguments are NOT inferred (must be explicit) ---
// function add(a, b) { ... }  // noImplicitAny error

// --- Contextual typing (inference from context) ---
const names = ["Alice", "Bob", "Charlie"];
names.map(name => name.toUpperCase());
//         ↑ name: string (inferred from array element type)

// --- const assertion and inference ---
const config = {
    host: "localhost",
    port: 3000,
} as const;
// config: { readonly host: "localhost"; readonly port: 3000 }
// Inferred as literal types

// --- Conditional type inference ---
type Awaited<T> = T extends Promise<infer U> ? U : T;
type Result = Awaited<Promise<string>>; // Result = string

// --- Template literal type inference ---
type EventName = `on${Capitalize<"click" | "focus" | "blur">}`;
// EventName = "onClick" | "onFocus" | "onBlur"

// --- satisfies operator (TypeScript 4.9+) ---
const palette = {
    red: [255, 0, 0],
    green: "#00ff00",
    blue: [0, 0, 255],
} satisfies Record<string, string | number[]>;
// Type checks while preserving the inferred type
// palette.red is inferred as number[] (not string | number[])
```

**Limitations of TypeScript's type inference:**

```typescript
// Cases where inference fails / is insufficient

// 1. Empty arrays
const arr = [];      // arr: any[] (type cannot be determined)
const arr: string[] = []; // Explicit annotation required

// 2. Multiple type candidates
const x = Math.random() > 0.5 ? 42 : "hello";
// x: string | number (inferred as union type)

// 3. Callback overloads
declare function on(event: "click", handler: (e: MouseEvent) => void): void;
declare function on(event: "focus", handler: (e: FocusEvent) => void): void;
// Overload resolution may not be complete with type inference alone

// 4. Recursive types
type JSON = string | number | boolean | null | JSON[] | { [key: string]: JSON };
// Depending on the inference context, explicit type annotations may be needed
```

### 5.2 Rust

Rust has a unique type inference that integrates local type inference with the ownership system.

```rust
// [Code Example 4] Characteristics of Rust's type inference

// --- Basic local inference ---
let x = 42;                    // x: i32 (default integer type)
let y = 3.14;                  // y: f64 (default floating-point type)
let s = String::from("hello"); // s: String
let v = vec![1, 2, 3];         // v: Vec<i32>

// --- Inference from context (backward reference) ---
let mut v = Vec::new();    // At this point Vec<_> (type undetermined)
v.push(42);                // Determined as Vec<i32> from push's argument
// Rust infers not only from "forward" but also "backward" usage sites

// --- Turbofish syntax (explicit type parameters) ---
let parsed = "42".parse::<i32>().unwrap();
// parse()'s return type cannot be determined from context, so specify with ::<i32>

// --- Closure type inference ---
let add = |a, b| a + b;
let result: i32 = add(1, 2);
// Closure argument types are inferred from usage sites

// --- Interaction between ownership and type inference ---
let s1 = String::from("hello");
let s2 = s1;       // Ownership moves from s1 to s2
// s1 is invalidated here (type inference + ownership checking)
// println!("{}", s1); // Compilation error: value used after move

// --- Lifetime elision rules and type inference ---
// Lifetimes are also a form of "type inference"
fn first_word(s: &str) -> &str {
    // By lifetime elision rules, inferred as
    // fn first_word<'a>(s: &'a str) -> &'a str
    s.split_whitespace().next().unwrap_or("")
}

// --- Function arguments/return values are NOT inferred ---
fn add(a: i32, b: i32) -> i32 {
    a + b   // Type annotations are always required in function signatures
}

// --- Trait bounds and type inference ---
fn print_all<T: std::fmt::Display>(items: &[T]) {
    for item in items {
        println!("{}", item); // T: Display is guaranteed
    }
}
// print_all(&[1, 2, 3]); // T = i32 inferred
// print_all(&["a", "b"]); // T = &str inferred
```

### 5.3 Go

Go intentionally adopts simple type inference. This reflects Go's design philosophy of "simplicity."

```go
// Go's type inference: := short variable declaration

// --- Basic inference ---
x := 42              // x: int
s := "hello"         // s: string
f := 3.14            // f: float64
b := true            // b: bool
arr := []int{1,2,3}  // arr: []int

// --- Comparison with var declaration ---
var x1 int            // Zero-value initialization, explicit type
var x2 = 42           // Type inference
x3 := 42              // Short declaration + type inference (most concise)

// --- Simultaneous multi-variable inference ---
a, b, c := 1, "hello", true
// a: int, b: string, c: bool

// --- Function arguments/return values are NOT inferred ---
func add(a int, b int) int {
    return a + b
}

// --- Go 1.18+ generic type parameter inference ---
func Map[T any, U any](s []T, f func(T) U) []U {
    result := make([]U, len(s))
    for i, v := range s {
        result[i] = f(v)
    }
    return result
}

// Type parameters can be omitted at call site
nums := []int{1, 2, 3}
strs := Map(nums, func(n int) string {  // T=int, U=string inferred
    return fmt.Sprintf("%d", n)
})
```

### 5.4 Haskell

Haskell fully implements HM type inference, and all programs can be typed even without type annotations.

```haskell
-- Haskell: The most powerful type inference

-- Inferred: id :: a -> a
id x = x

-- Inferred: const :: a -> b -> a
const x _ = x

-- Inferred: flip :: (a -> b -> c) -> b -> a -> c
flip f x y = f y x

-- Inferred: map :: (a -> b) -> [a] -> [b]
map _ []     = []
map f (x:xs) = f x : map f xs

-- Inferred: foldr :: (a -> b -> b) -> b -> [a] -> b
foldr _ z []     = z
foldr f z (x:xs) = f x (foldr f z xs)

-- Inferred: (.) :: (b -> c) -> (a -> b) -> a -> c
(.) f g x = f (g x)

-- Automatic inference of type class constraints
-- Inferred: sort :: Ord a => [a] -> [a]
sort []     = []
sort (x:xs) = sort [y | y <- xs, y <= x]
           ++ [x]
           ++ sort [y | y <- xs, y > x]
-- The Ord constraint is automatically inferred from the use of (<=) and (>)

-- Type inference with monads
-- Inferred: readAndPrint :: IO ()
readAndPrint = do
    line <- getLine        -- line :: String
    putStrLn (map toUpper line)  -- Whole thing :: IO ()
```

### 5.5 Kotlin and Scala 3

```kotlin
// Kotlin: Smart casts and type inference

// Basic type inference
val x = 42              // x: Int
val s = "hello"         // s: String
val list = listOf(1, 2) // list: List<Int>

// Smart cast (a form of flow-sensitive typing)
fun process(obj: Any) {
    if (obj is String) {
        // obj is automatically cast to String
        println(obj.length)  // No cast needed
    }
    // Same with when expressions
    when (obj) {
        is Int -> println(obj + 1)      // obj: Int
        is String -> println(obj.length) // obj: String
        is List<*> -> println(obj.size)  // obj: List<*>
    }
}

// Lambda type inference
val transform: (String) -> Int = { it.length }
// it: String (inferred from expected type)

// Inference in builder pattern
val result = buildList {
    add(1)       // Add Int
    add(2)       // Type matches
    addAll(listOf(3, 4))
}
// result: List<Int>
```

```scala
// Scala 3: Advanced type inference

// Basic inference
val x = 42              // x: Int
val s = "hello"         // s: String

// Context function inference
given ord: Ordering[Int] = Ordering.Int
def sorted[T](list: List[T])(using Ordering[T]): List[T] =
  list.sorted
// using parameters are inferred from given instances

// Match type inference
type Elem[X] = X match
  case String      => Char
  case Array[t]    => t
  case Iterable[t] => t

// Elem[String] = Char, Elem[Array[Int]] = Int

// Union and intersection types
val x: String | Int = if true then "hello" else 42
// Inferred union type

// Extension method inference
extension (s: String)
  def words: List[String] = s.split("\\s+").toList
// Type of "hello world".words: List[String]
```

### 5.6 Cross-Language Type Inference Capability Comparison Table

```
Type Inference Capability Comparison by Language
================================================================

Feature                 | TS    | Rust  | Go    | Haskell | Kotlin | Scala3
------------------------+-------+-------+-------+---------+--------+-------
Local variable inference| O     | O     | O     | O       | O      | O
Function return infer   | O     | X     | X     | O       | D      | O
Function argument infer | X     | X     | X     | O       | X      | X
Callback argument infer | O     | O     | D     | O       | O      | O
Generics inference      | O     | O     | D     | O       | O      | O
Flow-sensitive typing   | O     | X*    | X     | X       | O      | D
Type class/constraint   | X     | O     | X     | O       | X      | O
Lifetime inference      | N/A   | O     | N/A   | N/A     | N/A    | N/A
Pattern match inference | D     | O     | X     | O       | O      | O

O: Full support  D: Partial support  X: Not supported
*: Rust's borrow checker provides similar functionality

================================================================
```

---

## 6. Limitations of Type Inference and Countermeasures

### 6.1 Five Typical Patterns Where Inference Fails

```
Patterns Where Type Inference Fails
================================================================

Pattern 1: Empty collections
------------------------------------------------------------
  let arr = [];               // TypeScript: any[]
  let v = Vec::new();         // Rust: Vec<_> type undetermined

  Countermeasure: Add type annotations
  let arr: number[] = [];
  let v: Vec<i32> = Vec::new();

Pattern 2: Multiple type candidates (overloads)
------------------------------------------------------------
  // When multiple interpretations are possible
  let result = parse(input);  // parse has multiple return types

  Countermeasure: Type annotation or turbofish
  let result: User = parse(input);
  let result = input.parse::<i32>();

Pattern 3: Recursive data structures
------------------------------------------------------------
  // Self-referencing types are difficult to infer
  type Tree = { value: number; children: Tree[] };
  let tree = { value: 1, children: [] };
  // children: never[] (not inferred as Tree[])

  Countermeasure: Explicit type annotation
  let tree: Tree = { value: 1, children: [] };

Pattern 4: Partial application of higher-order functions
------------------------------------------------------------
  // Type is not determined during partial application
  const apply = (f: Function) => f; // Type information is lost

  Countermeasure: Use generics
  const apply = <T, U>(f: (x: T) => U) => f;

Pattern 5: Type mismatch across different branches
------------------------------------------------------------
  // Returning different types from conditional branches
  function getValue(flag: boolean) {
      if (flag) return 42;
      else return "hello";
  }
  // Return: number | string (unintended union type)

  Countermeasure: Explicitly annotate return type, or reconsider the design

================================================================
```

### 6.2 Where to Write Type Annotations and Where to Omit Them

```
Type Annotation Decision Matrix
================================================================

                            Inference result   Inference result
                            is obvious         is unclear
                        ┌──────────────┬──────────────┐
  Public API             │  Write(*)    │  Must write  │
  (function args/returns)│              │              │
                        ├──────────────┼──────────────┤
  Internal implementation│  OK to omit  │  Write       │
  (local variables etc.) │              │              │
                        └──────────────┴──────────────┘

  (*) Even if inference is obvious, write annotations for public APIs as documentation

================================================================

Specific guidelines:

  Must write:
    [1] Function argument types (public or private)
    [2] Public function return types
    [3] Empty collection initialization
    [4] When any / unknown type is inferred
    [5] Results of type casts (as / type assertion)

  OK to omit:
    [1] Assignment from literals (let x = 42)
    [2] Obvious function call results (let len = str.length)
    [3] Callback arguments (names.map(n => ...))
    [4] Intermediate variables (intermediate results in pipelines)
    [5] Destructuring assignment (const { name, age } = user)

================================================================
```

### 6.3 Impact of Type Inference on Performance

Type inference generally affects compilation time. In large-scale projects, this can become non-negligible.

```
Impact of Type Inference on Compilation Time
================================================================

Language    | Inference complexity  | Impact on large projects
------------+---------------------+---------------------------
Haskell     | Nearly linear O(n)  | Can slow down during type class resolution
Rust        | Nearly linear O(n)  | Trait resolution + borrow checking dominates
TypeScript  | Worst-case exponential| Can slow down with nested conditional types
Go          | Linear O(n)         | Impact is minimal
Scala 3     | Worst-case exponential| Can be heavy with given/using resolution

================================================================

Example where type inference slows down TypeScript compilation:
  // Deeply nested conditional types
  type DeepPartial<T> = {
    [P in keyof T]?: T[P] extends object
      ? DeepPartial<T[P]>
      : T[P];
  };
  // Can become explosively slow when applied to large object types

  Countermeasures:
  - Limit recursion depth of types
  - Add explicit type annotations to intermediate types
  - Prefer interface over type alias (structural comparison caching is more effective)

================================================================
```

---

## 7. Anti-Patterns and Best Practices

### 7.1 Anti-Pattern 1: Annotation Overkill

Adding redundant type annotations where type inference works accurately reduces code readability and degrades maintainability.

```typescript
// *** Anti-Pattern: Annotation Overkill ***

// BAD: Annotating everything (redundant)
const name: string = "Alice";
const age: number = 30;
const active: boolean = true;
const scores: number[] = [90, 85, 92];
const user: { name: string; age: number } = { name: "Alice", age: 30 };
const doubled: number[] = scores.map((s: number): number => s * 2);

// GOOD: Delegate to inference (concise)
const name = "Alice";
const age = 30;
const active = true;
const scores = [90, 85, 92];
const user = { name: "Alice", age: 30 };
const doubled = scores.map(s => s * 2);
```

```
Why this is a problem:
================================================================

1. Reduced readability
   - Type annotations become noise, obscuring the logic
   - Information duplication (type is also obvious from the right-hand side)

2. Degraded maintainability
   - When changing values, type annotations must also be updated simultaneously
   - Risk of type annotations diverging from actual types

3. Harder refactoring
   - Redundant type annotations cause type changes to ripple widely

Decision criterion:
  "If I remove this type annotation, can the reader still figure out the type?"
  → Yes: OK to omit
  → No: Should write it

================================================================
```

### 7.2 Anti-Pattern 2: Any Escape Hatch

Casually using the `any` type when type inference is difficult destroys type safety.

```typescript
// *** Anti-Pattern: Any Escape Hatch ***

// BAD: Disabling type checking with any
function processData(data: any): any {
    return data.map((item: any) => item.name.toUpperCase());
    // All type information is lost
    // Compiles even if data is not an array or item.name doesn't exist
}

// GOOD: Define appropriate types
interface DataItem {
    name: string;
    value: number;
}

function processData(data: DataItem[]): string[] {
    return data.map(item => item.name.toUpperCase());
    // Type-safe: compilation error if data is not DataItem[]
}

// BETTER: Generalize with generics
function processData<T extends { name: string }>(data: T[]): string[] {
    return data.map(item => item.name.toUpperCase());
    // Accepts arrays of any type that has a name property
}

// Gradual improvement: any → unknown → concrete type
function safeProcess(data: unknown): string[] {
    if (!Array.isArray(data)) {
        throw new Error("Expected array");
    }
    return data.map(item => {
        if (typeof item === "object" && item !== null && "name" in item) {
            return String((item as { name: unknown }).name).toUpperCase();
        }
        throw new Error("Invalid item");
    });
}
```

```
Usage of any vs unknown vs never
================================================================

Type      | Safety | Purpose
----------+--------+------------------------------------------
any       | X      | Complete disabling of type checking (avoid)
unknown   | O      | When type is unknown but want safe handling
never     | O      | Type of unreachable code (exhaustiveness check)
object    | D      | Any non-null object

Legitimate uses of any:
  - Temporary measure when external library lacks type definitions
  - Mocks in test code (prioritizing flexibility over type safety)
  - Gradual typing during migration period

================================================================
```

### 7.3 Anti-Pattern 3: Inappropriate Type Assertions

```typescript
// *** Anti-Pattern: Misuse of type assertions ***

// BAD: Unfounded type assertions
const data = JSON.parse(response) as User;
// JSON.parse returns any. There is no guarantee it is User.

// BAD: Double assertion (complete destruction of type safety)
const x = ("hello" as unknown) as number;
// Forces string to number (source of runtime errors)

// GOOD: Safely narrow with type guards
function isUser(obj: unknown): obj is User {
    return (
        typeof obj === "object" &&
        obj !== null &&
        "name" in obj &&
        "age" in obj &&
        typeof (obj as User).name === "string" &&
        typeof (obj as User).age === "number"
    );
}

const data: unknown = JSON.parse(response);
if (isUser(data)) {
    // Here data is inferred as User (narrowed by type guard)
    console.log(data.name);
}

// BETTER: Use validation libraries like zod
import { z } from "zod";
const UserSchema = z.object({
    name: z.string(),
    age: z.number(),
});
type User = z.infer<typeof UserSchema>;

const data = UserSchema.parse(JSON.parse(response));
// data: User (validated)
```

### 7.4 Best Practices Summary

```
Best Practices for Type Inference
================================================================

[1] Always write type annotations at public API boundaries
    → Function arguments, return values, exported types

[2] Let inference handle local variables
    → let x = 42; is sufficient. let x: number = 42; is redundant

[3] Add type annotations to empty collections
    → const arr: string[] = [];

[4] Avoid any, use unknown
    → For unknown types, use unknown + type guards

[5] Prefer type guards over type assertions (as)
    → Ensures runtime safety

[6] Add type annotations when inference results are unclear
    → Aids reader comprehension

[7] Utilize const assertions
    → Preserve literal types with as const

[8] Use IDE type display to verify inference results
    → Develop the habit of checking inferred types by hovering

================================================================
```

---

## 8. Practical Exercises (3 Levels)

### Exercise 1: [Basic] Verifying and Understanding Type Inference

**Objective**: Experience how far each language's type inference automatically determines types.

**Task 1-1**: Write the following code in TypeScript and use the IDE's hover feature to check the inferred type of each variable.

```typescript
// Check the type of each variable in the IDE and write it in comments
const a = 42;                          // a: ???
const b = [1, "hello", true];          // b: ???
const c = { x: 1, y: "hello" };       // c: ???
const d = new Map();                   // d: ???
const e = Promise.resolve(42);         // e: ???
const f = (x: number) => x > 0;       // f: ???

const g = [1, 2, 3].map(x => x * 2);            // g: ???
const h = [1, 2, 3].filter(x => x > 1);         // h: ???
const i = [1, 2, 3].reduce((acc, x) => acc + x); // i: ???
```

**Expected Answers:**

```typescript
const a = 42;                          // a: number
const b = [1, "hello", true];          // b: (string | number | boolean)[]
const c = { x: 1, y: "hello" };       // c: { x: number; y: string }
const d = new Map();                   // d: Map<any, any>
const e = Promise.resolve(42);         // e: Promise<number>
const f = (x: number) => x > 0;       // f: (x: number) => boolean

const g = [1, 2, 3].map(x => x * 2);            // g: number[]
const h = [1, 2, 3].filter(x => x > 1);         // h: number[]
const i = [1, 2, 3].reduce((acc, x) => acc + x); // i: number
```

**Task 1-2**: Compile the following code in Rust and verify the compiler's type inference.

```rust
fn main() {
    let a = 42;                    // Type?
    let b = vec![1, 2, 3];        // Type?
    let c = "hello".to_string();  // Type?
    let d = (1, "hello", true);   // Type?

    // Add the following line and check how the type changes
    let e: u8 = a;  // Does this cause a compilation error?
    // Hint: a's default type is i32, but
    //       does e's type annotation affect it?
}
```

### Exercise 2: [Applied] Experiencing and Resolving Type Inference Limitations

**Objective**: Create cases where type inference fails and learn how to resolve them with appropriate type annotations.

**Task 2-1**: Create 5 cases where TypeScript type inference fails, and fix each one.

```typescript
// Case 1: Empty array
const items = [];  // Becomes any[] → Fix this

// Case 2: Type widening in conditional expressions
const value = Math.random() > 0.5 ? 42 : "hello";
// Becomes string | number, but we want only number → Fix this

// Case 3: Excess properties in object literals
interface Config {
    host: string;
    port: number;
}
const config = { host: "localhost", port: 3000, debug: true };
// Want to use as Config but debug is excess → Fix this

// Case 4: Promise chain types
async function fetchData() {
    const response = await fetch("/api/users");
    const data = await response.json(); // data: any → Fix this
    return data;
}

// Case 5: Generic type parameter not inferred
function identity(x) { return x; }  // Fix this
```

**Model Answers:**

```typescript
// Case 1: Add type annotation
const items: string[] = [];

// Case 2: Fix the conditional expression, or add type annotation
const value: number = Math.random() > 0.5 ? 42 : 0;

// Case 3: Use satisfies
const config = { host: "localhost", port: 3000, debug: true } satisfies Config;
// Or type annotation: const config: Config = { host: "localhost", port: 3000 };

// Case 4: Specify type parameter
interface User { id: number; name: string; }
async function fetchData(): Promise<User[]> {
    const response = await fetch("/api/users");
    const data: User[] = await response.json();
    return data;
}

// Case 5: Use generics
function identity<T>(x: T): T { return x; }
```

**Task 2-2**: Create 3 cases where Rust type inference fails, and resolve each with turbofish or type annotations.

```rust
fn main() {
    // Case 1: parse return type is indeterminate
    let n = "42".parse().unwrap(); // Fix this

    // Case 2: collect type is indeterminate
    let v = (0..10).collect(); // Fix this

    // Case 3: Default trait type is indeterminate
    let d = Default::default(); // Fix this
}
```

**Model Answers:**

```rust
fn main() {
    // Case 1: Specify type with turbofish
    let n = "42".parse::<i32>().unwrap();
    // Or type annotation: let n: i32 = "42".parse().unwrap();

    // Case 2: Turbofish or type annotation
    let v: Vec<i32> = (0..10).collect();
    // Or: let v = (0..10).collect::<Vec<i32>>();

    // Case 3: Type annotation
    let d: i32 = Default::default();
    // Or: let d = i32::default();
}
```

### Exercise 3: [Advanced] Manual Execution of Hindley-Milner Type Inference

**Objective**: Manually execute the HM type inference algorithm and understand the internal workings of type inference.

**Task 3-1**: Manually trace HM type inference for the following function.

```
Target function: let apply f x = f x
```

**Manual Trace Procedure:**

```
Step 1: Assign type variables
================================================================
  apply: alpha
  f: beta
  x: gamma
  Type of expression (f x): delta

Step 2: Collect constraints
================================================================
  f is applied to argument x, so f is a function type:
    beta = gamma → delta

  The result of apply f x is (f x), so:
    alpha = beta → gamma → delta

Step 3: Unification
================================================================
  Substitute beta = gamma → delta into alpha:
    alpha = (gamma → delta) → gamma → delta

Step 4: Generalization
================================================================
  Universally quantify free type variables gamma, delta:
    apply :: forall gamma delta. (gamma → delta) → gamma → delta

  Rename to standard variable names:
    apply :: (a -> b) -> a -> b

  This is the same type as Haskell's ($) operator.
================================================================
```

**Task 3-2**: Similarly trace the following function manually.

```
Target function: let twice f x = f (f x)
```

**Hint:**

```
Step 1: f: alpha, x: beta, inner (f x): gamma, outer f gamma: delta

Step 2: Constraints
  Inner application: alpha = beta → gamma
  Outer application: alpha = gamma → delta

Step 3: Unification
  beta → gamma = gamma → delta
  Therefore beta = gamma and gamma = delta
  Meaning beta = gamma = delta

Step 4: Result
  twice :: (a -> a) -> a -> a
  f must be a function that takes and returns the same type
```

**Task 3-3 (Challenge)**: Trace the following function.

```
Target function: let fix f = f (fix f)
```

```
This is the fixed-point combinator.

Step 1: fix: alpha, f: beta

Step 2:
  Let tau be the type of fix f
  f is applied to (fix f), so: beta = tau → tau'
  The result of fix f is f (fix f), so: tau = tau'
  Therefore beta = tau → tau

Step 3:
  fix :: (tau → tau) → tau
  Standard variable names: fix :: (a -> a) -> a

Note: From the occurs check perspective, the definition of fix itself
      is recursive, and cannot be typed in standard HM.
      Haskell handles this with special rules for recursive bindings.
```

---

## 9. FAQ (Frequently Asked Questions)

### Q1: If type inference exists, why don't all languages make type annotations unnecessary like Haskell?

**A**: There are 3 main reasons.

**Reason 1: Incompatibility with Subtyping**

Hindley-Milner type inference does not work fully in type systems with subtyping (type inclusion relationships through inheritance and interface implementation). In object-oriented languages like Java, TypeScript, and Kotlin, relationships like `Dog extends Animal` exist, and this destroys the principal type property.

```
Example: Where subtyping breaks the principal type property
================================================================

class Animal { move() { ... } }
class Dog extends Animal { bark() { ... } }
class Cat extends Animal { meow() { ... } }

function example(x) {
    x.move();  // What is x's type?
}

Candidates:
  - Animal (most general)
  - Dog (might also use bark)
  - Cat (might also use meow)
  - Animal & Serializable (other interfaces too?)

In HM inference, the "most general type" is uniquely determined, but
with subtyping, the definition of "most general" becomes ambiguous.

================================================================
```

**Reason 2: Readability and Documentation**

Type annotations on public APIs serve a documentation role. Without type annotations, you would need to read the implementation to understand a function's signature. In large-scale projects, the explicitness of type annotations greatly improves maintainability.

**Reason 3: Compilation Time**

Global type inference requires analyzing the entire program, increasing compilation time. Local type inference (making types explicit at function boundaries) allows each function to be compiled independently, making incremental compilation more efficient.

### Q2: How does TypeScript's `as const` differ from normal type inference?

**A**: Normal type inference performs "type widening," but `as const` preserves literal types.

```typescript
// Normal inference (types are widened)
const config = {
    method: "GET",        // method: string (literal type widened to string)
    retries: 3,           // retries: number
    endpoints: ["/a", "/b"] // endpoints: string[]
};
// config: { method: string; retries: number; endpoints: string[] }

// as const (types are NOT widened)
const config = {
    method: "GET",        // method: "GET" (literal type preserved)
    retries: 3,           // retries: 3
    endpoints: ["/a", "/b"] // endpoints: readonly ["/a", "/b"]
} as const;
// config: {
//   readonly method: "GET";
//   readonly retries: 3;
//   readonly endpoints: readonly ["/a", "/b"];
// }

// Useful scenario for as const: Discriminated union tags
const actions = {
    increment: { type: "INCREMENT" as const, payload: 1 },
    decrement: { type: "DECREMENT" as const, payload: 1 },
};
// The type field becomes a literal type, usable as a discriminated union
```

```
Type Widening and Type Narrowing
================================================================

Widening: Literal → General type
  42        --> number
  "hello"   --> string
  true      --> boolean
  [1,2,3]   --> number[]

  * Occurs when declared with let
  * Literal types are preserved when declared with const

Narrowing: General type → Concrete type
  string | number  --> string  (with typeof guard)
  Animal           --> Dog     (with instanceof guard)
  Shape            --> Circle  (with discriminated union)

  * Occurs automatically through control flow analysis

================================================================
```

### Q3: Why is Rust's "turbofish" syntax `::<Type>` necessary?

**A**: In Rust's local type inference, there are situations where type information is insufficient. In particular, when the return type of a generic function cannot be determined from the call context alone, turbofish is necessary.

```rust
// Cases where turbofish is necessary

// 1. parse(): Return type can be multiple types
let n = "42".parse::<i32>().unwrap();    // Parse as i32
let n = "42".parse::<f64>().unwrap();    // Parse as f64
let n = "42".parse::<u8>().unwrap();     // Parse as u8

// 2. collect(): Multiple conversion targets from iterator
let v = (0..10).collect::<Vec<i32>>();       // Convert to Vec
let s = (0..10).map(|i| format!("{}", i))
               .collect::<String>();          // Join into String
let hs = vec![1,2,3].into_iter()
                     .collect::<HashSet<_>>(); // Convert to HashSet

// 3. Default::default(): Different default values by type
let x = i32::default();    // 0
let s = String::default(); // ""
let v = Vec::<i32>::default(); // []

// Origin of the turbofish name:
// The shape ::<> resembles a fish (specifically a turbo snail)
//   ::<>  ← Does this look like a fish?
```

### Q4: What is the relationship between type inference and generics?

**A**: Type inference is closely related to generics as a mechanism that automatically determines generic type parameters.

```typescript
// Generic type parameter inference

// Explicitly specifying type parameters
const result1 = identity<number>(42);

// Inferring type parameters (inferred from arguments)
const result2 = identity(42);  // T = number inferred

// Inference of multiple type parameters
function merge<A, B>(a: A, b: B): A & B {
    return { ...a, ...b };
}
const merged = merge({ name: "Alice" }, { age: 30 });
// A = { name: string }, B = { age: number } inferred
// Return: { name: string } & { age: number }

// Inference with constrained generics
function getLength<T extends { length: number }>(item: T): number {
    return item.length;
}
getLength("hello");     // T = string inferred (string has length)
getLength([1, 2, 3]);   // T = number[] inferred
// getLength(42);       // Error: number does not have length
```

```
Generics Inference Flow Diagram
================================================================

function map<T, U>(arr: T[], fn: (item: T) => U): U[]

Call: map([1, 2, 3], x => x.toString())

Inference process:
  1. Infer T = number from first argument [1, 2, 3]
     arr: T[]  <-->  [1, 2, 3]: number[]
          │
          ▼
     T = number

  2. Propagate T = number to callback
     fn: (item: T) => U  -->  fn: (item: number) => U
                                    │
                                    ▼
     x => x.toString() where x: number

  3. Infer U from callback return value
     x.toString(): string  -->  U = string

  4. Final result
     map<number, string>([1, 2, 3], x => x.toString()): string[]

================================================================
```

### Q5: Why doesn't TypeScript infer function argument types?

**A**: In the bidirectional type checking adopted by TypeScript, function declaration argument types are used as "starting points for inference," so they are not themselves inference targets. This is an intentional design decision.

```
Why function arguments are not inferred
================================================================

1. Functions are "type information providers"
   Argument types serve as starting points for type inference
   within the function body.
   Making the starting point itself an inference target would
   create circular dependencies.

2. Public API clarity
   Function argument types define API contracts.
   Relying on inference could cause API changes when
   the implementation changes.

3. Error message quality
   When argument types are explicit, the location of type
   errors is clear.
   Relying on inference makes it difficult to pinpoint
   "where the mistake is."

Exception: Callback arguments ARE inferred
  names.map(name => name.toUpperCase())
  //        ↑ name: string (inferred from array element type)

  This is because the "callback's type" is determined from
  the higher context, making checking mode inference possible.

================================================================
```

### Q6: How do you debug type inference?

**A**: Each language and tool provides methods to check inferred types.

```
Type Inference Debugging Methods
================================================================

TypeScript:
  - IDE hover display (VSCode, WebStorm)
  - tsc --noEmit --declaration to generate .d.ts
  - // @ts-expect-error to intentionally trigger errors and check types
  - type Inspect<T> = T; to visualize intermediate types

Rust:
  - Compiler error messages display inferred types
  - let _: () = expr; to check expr's type via error messages
  - rust-analyzer's inlay hints (inline type display in IDE)
  - #[derive(Debug)] + println!("{:?}", x) to check types at runtime

Haskell:
  - :type expression  (check type in GHCi)
  - :info name (display type information)
  - -fwarn-missing-signatures for top-level type warnings
  - Use _ in type annotations to have compiler display the inferred type

Go:
  - IDE hover display
  - fmt.Printf("%T\n", x) to display the type
  - go vet for type checking

================================================================
```

### Q7: What is the relationship between dependent types and type inference?

**A**: Dependent types express types that depend on values. For example, they can express "a vector of length n" at the type level. Languages with dependent types (Idris, Agda, Coq) do provide type inference, but it is not complete. Type inference for dependent types is generally undecidable, so situations requiring user-provided type annotations or hints increase.

```
Decidability Spectrum of Type Inference
================================================================

  Fully decidable            Partially decidable        Undecidable
  ◄────────────────────────────────────────────────────────►
  │                    │                    │
  HM Inference         Bidirectional        Dependent Types
  (Haskell, ML)        Type Checking        (Idris, Agda)
  │                    (TypeScript, Kotlin)  │
  No annotations       Some annotations     Many annotations
  needed               needed               needed

  * Trade-off between expressiveness and inference capability
  The more expressive the type system, the harder automatic inference becomes

================================================================
```

---


## FAQ

### Q1: What is the most important point in learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory, but by actually writing code and verifying behavior.

### Q2: What common mistakes do beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving to the next step.

### Q3: How is this applied in practice?

Knowledge of this topic is frequently applied in daily development work. It becomes especially important during code reviews and architecture design.

---

## 10. Summary and Next Steps

### 10.1 The Big Picture of Type Inference

```
Overall Map of Type Inference
================================================================

                      Type Inference
                              │
              ┌───────────────┼───────────────┐
              │               │               │
         Algorithms      Inference Scope   Language Adoption
              │               │               │
      ┌───────┼───────┐   ┌───┼───┐     ┌─────┼─────┐
      │       │       │   │   │   │     │     │     │
    HM     Bidirec-  Con-  Local Func  Global  TS   Rust  Haskell
    Infer  tional   straint      tion          Go   Kotlin
           Check    Based                     Scala  Swift
              │
      ┌───────┼───────┐
      │               │
   Synthesis      Checking
   Mode           Mode
   (Bottom-up)    (Top-down)

================================================================
```

### 10.2 Comprehensive Cross-Language Type Inference Comparison

| Characteristic | Haskell | Rust | TypeScript | Go | Kotlin | Scala 3 |
|------|---------|------|------------|-----|--------|---------|
| Inference algorithm | HM + type classes | Local HM variant | Bidirectional | Local | Bidirectional | Bidirectional + HM |
| Inference scope | Global | Local | Local + context | Variables only | Local + context | Local + context |
| Principal type | Yes | No | No | N/A | No | Partial |
| Annotation necessity | Low (recommended) | Medium (function boundaries) | Medium (arguments) | Medium (args/returns) | Medium (arguments) | Low-Medium |
| Subtyping | None | Traits | Structural | Interfaces | Nominal | Nominal + structural |
| Flow-sensitive typing | None | Borrow checker | Yes | None | Smart casts | Pattern matching |
| Learning curve | High | High | Medium | Low | Medium | High |

### 10.3 Type Annotation Decision Flowchart

```
Flowchart: Should you write a type annotation?
================================================================

  Should I write a type?
  │
  ├─ Is it a public API (export / pub)?
  │   ├─ Yes → Write (documentation + stability)
  │   └─ No ─┐
  │           │
  │   ├─ Is the inference result correct?
  │   │   ├─ No → Write (override inference)
  │   │   └─ Yes ─┐
  │   │            │
  │   │   ├─ Is the inference result obvious? (Understandable by reading?)
  │   │   │   ├─ Yes → Omit (let inference handle it)
  │   │   │   └─ No → Write (for readability)
  │   │   │
  │   │   └─ Is it an empty collection?
  │   │       ├─ Yes → Write
  │   │       └─ No → Omit

================================================================
```

### 10.4 Learning Roadmap

```
Type Inference Learning Roadmap
================================================================

Level 1 (Beginner): Understand the basics of type inference
  □ Can explain why let x = 42 is inferred as int
  □ Can check inferred types in IDE
  □ Can name 3 cases where type inference fails
  □ Can judge where to write type annotations

Level 2 (Intermediate): Master language-specific type inference
  □ Can utilize TypeScript's type narrowing
  □ Can judge when to use Rust's turbofish
  □ Understands generic type parameter inference
  □ Can utilize contextual typing of callbacks

Level 3 (Advanced): Understand type inference theory
  □ Can manually trace HM type inference
  □ Can explain the unification algorithm
  □ Can explain the difference between let-polymorphism and monomorphism restriction
  □ Can explain synthesis/checking modes of bidirectional type checking
  □ Understands conditions under which type inference becomes undecidable

Level 4 (Expert): Can design and extend type systems
  □ Can implement type inference for a new programming language
  □ Can analyze computational complexity of type inference algorithms
  □ Can discuss trade-offs between dependent types and type inference
  □ Can explain the relationship between rank-N polymorphism and type inference

================================================================
```

---

## 11. References

### Foundational Theory

1. **Pierce, Benjamin C.** *Types and Programming Languages.* MIT Press, 2002.
   - A comprehensive textbook on type systems. Chapter 22 provides detailed coverage of type inference (type reconstruction). Includes the formal definition and correctness proof of the Hindley-Milner algorithm. The primary reference for learning type inference theoretically.

2. **Hindley, J. Roger.** "The Principal Type-Scheme of an Object in Combinatory Logic." *Transactions of the American Mathematical Society*, vol. 146, 1969, pp. 29-60.
   - A historical paper proving the existence and uniqueness of principal type schemes. Established the mathematical foundation of type inference in the context of combinatory logic.

3. **Milner, Robin.** "A Theory of Type Polymorphism in Programming." *Journal of Computer and System Sciences*, vol. 17, no. 3, 1978, pp. 348-375.
   - A groundbreaking paper proposing Algorithm W. Defined an efficient type inference algorithm for the ML language and proved its soundness and completeness.

4. **Damas, Luis, and Robin Milner.** "Principal Type-Schemes for Functional Programs." *Proceedings of the 9th ACM SIGPLAN-SIGACT Symposium on Principles of Programming Languages (POPL)*, 1982, pp. 207-212.
   - The defining paper of the Damas-Milner type system. Provided the complete formalization of type inference including let-polymorphism.

### Bidirectional Type Checking

5. **Pierce, Benjamin C., and David N. Turner.** "Local Type Inference." *ACM Transactions on Programming Languages and Systems (TOPLAS)*, vol. 22, no. 1, 2000, pp. 1-44.
   - The paper that laid the foundation for bidirectional type checking (local type inference). Introduced the concepts of synthesis and checking modes, achieving the integration of subtyping and type inference.

6. **Dunfield, Jana, and Neelakantan R. Krishnaswami.** "Complete and Easy Bidirectional Typechecking for Higher-Rank Polymorphism." *Proceedings of the 18th ACM SIGPLAN International Conference on Functional Programming (ICFP)*, 2013, pp. 429-442.
   - A paper extending bidirectional type checking to higher-rank polymorphism. Demonstrated a technique that achieves complete type checking while being relatively easy to implement.

### Language-Specific Type Inference

7. **TypeScript Handbook.** "Type Inference." Microsoft, https://www.typescriptlang.org/docs/handbook/type-inference.html
   - Official documentation on type inference in TypeScript. Explains the operation of Best Common Type, Contextual Typing, and Type Guards.

8. **The Rust Reference.** "Type Inference." Rust Team, https://doc.rust-lang.org/reference/type-system.html
   - Official reference on type inference and lifetime elision rules in Rust. Explains the turbofish syntax and the relationship between trait bounds and type inference.

9. **Haskell 2010 Language Report.** "Declarations and Bindings, Type Inference." https://www.haskell.org/onlinereport/haskell2010/
   - Specification of type inference in Haskell. Defines type class constraint inference, defaulting rules, and the monomorphism restriction.

### Advanced Topics

10. **Odersky, Martin, Christoph Zenger, and Matthias Zenger.** "Colored Local Type Inference." *Proceedings of the 28th ACM SIGPLAN-SIGACT Symposium on Principles of Programming Languages (POPL)*, 2001, pp. 14-26.
    - The theoretical foundation of Scala's type inference. Proposes techniques for applying local type inference to object-oriented languages.

11. **Vytiniotis, Dimitrios, Simon Peyton Jones, Tom Schrijvers, and Martin Sulzmann.** "OutsideIn(X): Modular Type Inference with Local Assumptions." *Journal of Functional Programming*, vol. 21, no. 4-5, 2011, pp. 333-412.
    - The modern type inference algorithm of GHC (Haskell compiler). Addresses integration with type classes, GADTs, and type families.

---

## Recommended Next Reads


---

*Last updated: 2026-03-06*

---

## References

- [MDN Web Docs](https://developer.mozilla.org/) - Web technology reference
- [Wikipedia](https://en.wikipedia.org/) - Overview of technical concepts
