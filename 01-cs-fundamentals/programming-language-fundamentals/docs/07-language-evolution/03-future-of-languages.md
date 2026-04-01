# The Future of Programming Languages

> How will programming languages evolve in an era where AI writes code? This chapter comprehensively explores human-AI collaboration, the demand for memory safety, the expansion of WebAssembly, the rise of new paradigms, and the design philosophies of next-generation languages. Starting from current trends, we predict the state of programming languages in 5, 10, and 20 years based on theory and evidence.

## What You Will Learn in This Chapter

- [ ] Predict the direction of programming language evolution from a historical context
- [ ] Understand the fundamental shift in the role of languages in the AI era
- [ ] Explain 2020s trends such as gradual typing, memory safety, and Wasm
- [ ] Grasp the design philosophies of notable new languages (Mojo, Gleam, Zig, Carbon, Vale, Verse)
- [ ] Survey next-generation type theories including effect systems, dependent types, and linear types
- [ ] Envision the role of engineers in AI-collaborative programming


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of relevant foundational concepts
- Familiarity with the content of [DSLs and Metaprogramming](./02-dsl-and-metaprogramming.md)

---

## 1. Historical Laws of Programming Language Evolution

### 1.1 Patterns of Language Evolution

Looking across more than 70 years of programming language history, clear patterns emerge in their evolution. Understanding these patterns forms the foundation for predicting the future.

```
Five Laws of Language Evolution:

  Law 1: The Pendulum Law
  +--------------------------------------------------+
  |  Simple <--> Complex <--> Simple                  |
  |  C -> C++ -> Go                                   |
  |  Java -> Scala -> Kotlin                          |
  |  JavaScript -> TypeScript -> ?                    |
  |  A simpler language always emerges as a           |
  |  reaction to "excessive complexity"               |
  +--------------------------------------------------+

  Law 2: The Convergent Evolution Law
  +--------------------------------------------------+
  |  Languages of different origins arrive at the     |
  |  same features                                    |
  |  Type inference: ML -> Rust, TS, Kotlin, Swift    |
  |  ADT: Haskell -> Rust, TS, Swift                  |
  |  async/await: C# -> JS, Python, Rust              |
  +--------------------------------------------------+

  Law 3: The Gravity of Existing Ecosystems
  +--------------------------------------------------+
  |  New languages cannot gain adoption without        |
  |  coexisting with existing ecosystems               |
  |  TypeScript <- JavaScript ecosystem                |
  |  Kotlin <- JVM / Java ecosystem                    |
  |  Swift <- Objective-C / Apple ecosystem            |
  |  Rust <- FFI with C/C++                            |
  +--------------------------------------------------+

  Law 4: The Irreversible Progress of Safety
  +--------------------------------------------------+
  |  Once achieved, safety gains are never abandoned   |
  |  Manual memory management -> GC -> Ownership       |
  |  null -> Option/Maybe                              |
  |  Exceptions -> Result                              |
  |  Mutable by default -> Immutable by default        |
  +--------------------------------------------------+

  Law 5: The Abstraction Staircase
  +--------------------------------------------------+
  |  Machine code -> Assembly -> High-level languages  |
  |  -> Declarative languages                          |
  |  -> Natural language programming?                  |
  |  At each level, the abstraction of "what to do"    |
  |  increases, and the details of "how to do it"      |
  |  are hidden                                        |
  +--------------------------------------------------+
```

### 1.2 Language Adoption Cycle

```
Language Adoption Lifecycle:

  Popularity
  ^
  |          +---- Peak
  |         /|\
  |        / | \
  |       /  |  \-------- Stable Period (Mature Language)
  |      /   |
  |     /    |
  |    /     |
  |   /      |
  |--/-------|------------------ Time ->
  | Birth  Early  Growth  Maturity  Decline or Stability
  | Adopters Phase  Phase   Phase
  |

  Characteristics of Each Phase:
  +----------+------------------------------------------+
  | Birth    | Academic/personal projects. A small       |
  |          | number of passionate supporters           |
  | Early    | Adoption by startups and forward-         |
  | Adoption | thinking teams                            |
  | Growth   | Enterprise adoption expands.              |
  |          | Ecosystem matures                         |
  | Maturity | Used in large enterprise core systems.    |
  |          | Stability is prioritized                  |
  | Decline/ | Adoption in new projects decreases, but   |
  | Stability| used long-term for maintaining existing   |
  |          | systems                                   |
  +----------+------------------------------------------+

  Phase of Each Language as of 2025:
  Birth:         Mojo, Vale, Verse
  Early Adoption: Gleam, Zig, Carbon
  Growth:        Rust, Kotlin
  Maturity:      TypeScript, Go, Swift
  Stability:     Java, Python, C#, JavaScript
  Declining:     Perl, Objective-C, COBOL
                 (though maintenance demand remains enormous)
```

---

## 2. Clear Trends of the 2020s

### 2.1 The Spread of Gradual Typing

Gradual Typing is a technique for incrementally adding static type information to dynamically typed languages. Its theoretical foundation was established in a 2006 paper by Jeremy Siek and Walid Taha, and it entered a phase of practical widespread adoption in the 2020s.

```
Current State of Gradual Typing:

  +------------+-----------------+------------------+
  |  Language   |  Typing Tool     |  Adoption Level   |
  +------------+-----------------+------------------+
  | JavaScript | TypeScript      | De facto standard |
  | Python     | mypy / Pyright  | Rapidly spreading |
  | Ruby       | Sorbet / RBS    | Early stage       |
  | PHP        | PHPStan / Psalm | Spreading         |
  | Erlang     | Dialyzer        | Years of proven   |
  |            |                 | track record      |
  | Clojure    | core.typed      | Niche             |
  | Lua        | Teal / Luau     | Widespread in the |
  |            |                 | gaming industry   |
  +------------+-----------------+------------------+
```

**Code Example 1: Gradual Typing in Python**

```python
# --- Python: Adding types incrementally ---

# Step 1: No types (traditional Python code)
def process_orders(orders):
    total = 0
    for order in orders:
        if order["status"] == "completed":
            total += order["amount"]
    return total

# Step 2: Basic type hints
def process_orders(orders: list[dict]) -> float:
    total: float = 0
    for order in orders:
        if order["status"] == "completed":
            total += order["amount"]
    return total

# Step 3: Type-safe dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

class OrderStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

@dataclass(frozen=True)
class Order:
    id: int
    amount: float
    status: OrderStatus
    customer_id: int

def process_orders(orders: list[Order]) -> float:
    return sum(
        order.amount
        for order in orders
        if order.status == OrderStatus.COMPLETED
    )

# Step 4: Protocols (structural subtyping)
class Billable(Protocol):
    @property
    def amount(self) -> float: ...

    @property
    def status(self) -> OrderStatus: ...

def calculate_revenue(items: list[Billable]) -> float:
    """Accepts any type that satisfies the Billable protocol"""
    return sum(
        item.amount
        for item in items
        if item.status == OrderStatus.COMPLETED
    )

# Order implicitly satisfies the Billable protocol (structural subtyping)
orders = [Order(1, 1500.0, OrderStatus.COMPLETED, 100)]
revenue = calculate_revenue(orders)  # Passes type checking
```

### 2.2 Societal Demand for Memory Safety

In February 2024, the ONCD (Office of the National Cyber Director) at the White House published a report recommending "a transition away from memory-unsafe programming languages." This was a historic event in which the choice of programming language was recognized as a matter of national security.

```
Classification of Memory Safety:

  +--------------------------------------------------+
  |            Memory-Unsafe Languages                 |
  |  +--------------------------------------------+  |
  |  | C, C++, Assembly                           |  |
  |  | - Buffer overflow                           |  |
  |  | - Dangling pointer                          |  |
  |  | - Double free                               |  |
  |  | - Use-After-Free                            |  |
  |  +--------------------------------------------+  |
  +--------------------------------------------------+
  |            Memory-Safe Languages                   |
  |  +--------------------------------------------+  |
  |  | GC-based: Java, Go, Python, C#, Kotlin      |  |
  |  | Ownership-based: Rust                        |  |
  |  | Runtime checks: Swift (ARC + Safety)         |  |
  |  | Linear types: ATS, Clean (research stage)    |  |
  |  +--------------------------------------------+  |
  +--------------------------------------------------+

  Approximately 70% of vulnerabilities are caused by
  memory safety issues
  (Data from Microsoft / Google Chrome)
```

**Code Example 2: Memory Safety Through Rust's Ownership**

```rust
// --- Rust: The ownership system guarantees memory safety at compile time ---

// 1. Prevention of dangling pointers
fn no_dangling_reference() {
    let reference;
    {
        let value = String::from("hello");
        // reference = &value;  // Compile error!
        // Since value is freed when it goes out of scope,
        // this prevents reference from becoming a dangling pointer
    }
    // println!("{}", reference);  // Cannot be used
}

// 2. Prevention of data races
use std::thread;

fn no_data_race() {
    let mut data = vec![1, 2, 3];

    // Move ownership to another thread
    let handle = thread::spawn(move || {
        data.push(4);
        println!("Inside thread: {:?}", data);
    });

    // data has been moved, so it cannot be used here
    // println!("{:?}", data);  // Compile error!

    handle.join().unwrap();
}

// 3. Safety through borrowing rules
fn borrowing_rules() {
    let mut data = vec![1, 2, 3];

    // Multiple immutable borrows are allowed
    let r1 = &data;
    let r2 = &data;
    println!("{:?}, {:?}", r1, r2);

    // Mutable borrows are exclusive (cannot coexist with immutable borrows)
    let r3 = &mut data;
    r3.push(4);
    // println!("{:?}", r1);  // Compile error!
    // r1 is still valid but cannot coexist with r3
}

// 4. Lifetime management for reference validity
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}

fn lifetime_example() {
    let result;
    let string1 = String::from("long string");
    {
        let string2 = String::from("short");
        result = longest(&string1, &string2);
        println!("Longest: {}", result);
    }
    // result cannot be used outside string2's scope
    // (because it might be a reference to string2)
}
```

### 2.3 The Expansion of WebAssembly (Wasm)

WebAssembly is a binary format standardized in browsers in 2017, but in the 2020s it is rapidly gaining adoption as a runtime environment outside browsers as well.

```
WebAssembly Ecosystem Structure:

  +---------------------------------------------------+
  |                Source Languages                     |
  |  Rust  C/C++  Go  Python  Kotlin  C#  Swift  Zig   |
  +--------------------+------------------------------+
                       | Compile
                       v
  +---------------------------------------------------+
  |              WebAssembly (.wasm)                    |
  |        - Portable binary format                    |
  |        - Sandboxed execution                       |
  |        - Near-native performance                   |
  +--------------------+------------------------------+
                       | Execute
           +-----------+-----------+
           v           v           v
  +--------------+ +----------+ +--------------+
  |  Browser     | | Server   | | Edge         |
  |  - Chrome    | | - WASI   | | - Cloudflare |
  |  - Firefox   | | - Wasmtime| | - Fastly    |
  |  - Safari    | | - Wasmer | | - Vercel     |
  +--------------+ +----------+ +--------------+

  WASI (WebAssembly System Interface):
  +---------------------------------------------------+
  | A standard interface that provides Wasm with        |
  | access to OS capabilities such as the file system,  |
  | network, and clocks                                 |
  |                                                     |
  | Philosophy: "Compile once, run anywhere"            |
  |       A reimagining of Java's                       |
  |       "Write Once, Run Anywhere"                    |
  |       but with security guaranteed through          |
  |       sandboxed execution                           |
  +---------------------------------------------------+
```

### 2.4 The Normalization of AI Code Generation

```
Generations of AI Code Generation Tools:

  Generation 1 (2018-2021)      Generation 2 (2021-2023)
  +------------------+        +------------------+
  | - IntelliSense   |        | - GitHub Copilot |
  | - TabNine        |  >>>   | - ChatGPT        |
  | - Simple         |        | - Multi-line     |
  |   completion     |        |   generation     |
  | - Pattern        |        | - Natural        |
  |   matching       |        |   language input |
  +------------------+        +------------------+
         |                           |
  Generation 3 (2023-2025)      Generation 4 (2025-)
  +------------------+        +------------------+
  | - Claude Code    |        | - Multi-file     |
  | - Cursor         |  >>>   | - Architecture-  |
  | - Cody           |        |   level          |
  | - Full-file      |        |   suggestions    |
  |   generation     |        | - Integration    |
  | - Context        |        |   with formal    |
  |   understanding  |        |   verification   |
  |                  |        | - Autonomous     |
  |                  |        |   debugging      |
  +------------------+        +------------------+
```

---

## 3. Characteristics Required of Languages in the AI Era

### 3.1 The Rediscovery of the Importance of Type Systems

In the era where AI generates code, type systems have taken on a new role as "a framework for verifying the correctness of AI-generated code."

**Code Example 3: Types Functioning as Specifications for AI**

```typescript
// --- Type definitions functioning as "specifications" for AI ---

// 1. Type definitions = Specifications
interface PaginationParams {
    page: number;       // Integer >= 1
    pageSize: number;   // Range 1-100
    sortBy?: string;    // Sort key (optional)
    sortOrder?: "asc" | "desc";
}

interface PaginatedResponse<T> {
    data: T[];
    pagination: {
        currentPage: number;
        totalPages: number;
        totalItems: number;
        hasNextPage: boolean;
        hasPreviousPage: boolean;
    };
}

interface User {
    id: string;
    name: string;
    email: string;
    role: "admin" | "editor" | "viewer";
    createdAt: Date;
    lastLoginAt: Date | null;
}

// With this type signature alone, AI can generate an accurate implementation
function listUsers(
    params: PaginationParams,
    filters?: {
        role?: User["role"];
        createdAfter?: Date;
        searchQuery?: string;
    }
): Promise<PaginatedResponse<User>>;

// 2. Branded types for explicit constraints
type UserId = string & { readonly __brand: "UserId" };
type OrderId = string & { readonly __brand: "OrderId" };

// UserId and OrderId cannot be confused
function getUser(id: UserId): Promise<User>;
function getOrder(id: OrderId): Promise<Order>;

// const user = await getUser(orderId);  // Compile error!

// 3. Discriminated unions for explicit state transitions
type PaymentState =
    | { status: "pending"; createdAt: Date }
    | { status: "processing"; startedAt: Date; transactionId: string }
    | { status: "completed"; completedAt: Date; receiptUrl: string }
    | { status: "failed"; failedAt: Date; errorCode: string; retryable: boolean }
    | { status: "refunded"; refundedAt: Date; refundAmount: number };

// AI can generate appropriate handling for each state from this type definition
function renderPaymentStatus(state: PaymentState): string {
    switch (state.status) {
        case "pending":
            return `Awaiting payment (${state.createdAt.toLocaleDateString()})`;
        case "processing":
            return `Processing (Transaction ID: ${state.transactionId})`;
        case "completed":
            return `Completed (Receipt: ${state.receiptUrl})`;
        case "failed":
            return state.retryable
                ? `Failed (Error: ${state.errorCode}) - Retryable`
                : `Failed (Error: ${state.errorCode}) - Not retryable`;
        case "refunded":
            return `Refunded (${state.refundAmount.toLocaleString()} JPY)`;
    }
}
```

### 3.2 The Shift Toward Declarative Description

```
Declarative vs. Imperative Comparison:

  Imperative (How):                Declarative (What):
  +----------------------+       +----------------------+
  | for (i = 0; ...) {   |       | SELECT name          |
  |   if (age > 18) {    |       | FROM users           |
  |     result.push(name)|       | WHERE age > 18       |
  |   }                  |       | ORDER BY name        |
  | }                    |       |                      |
  | result.sort()        |       | // Describe only     |
  |                      |       | // "what you want"   |
  | // Specify step-by-  |       |                      |
  | // step "how to      |       |                      |
  | // retrieve"         |       |                      |
  +----------------------+       +----------------------+

  Expansion of Declarative Approaches:
  +----------------------------------------------+
  | Data queries:  SQL -> GraphQL                 |
  | UI:            HTML -> React -> SwiftUI       |
  | Infrastructure: Shell scripts -> Terraform    |
  | CI/CD:         Manual procedures ->           |
  |                GitHub Actions YAML            |
  | Data transform: Procedural code -> dbt SQL    |
  | API definition: Documentation -> OpenAPI Spec |
  +----------------------------------------------+

  Affinity with the AI Era:
  - Declarative specs = Clear instructions for AI
  - Humans define "what they want to do"
  - AI/compilers determine "how to achieve it"
```

### 3.3 Integration with Formal Verification

Formal Verification is a method of mathematically proving that a program satisfies its specification. It was traditionally used only in limited fields such as aerospace and medical devices, but attention has grown in the context of ensuring the reliability of AI-generated code.

```
Levels of Formal Verification:

  Level 1: Type Checking
  +--------------------------------------+
  | "This variable is always an integer"  |
  | Supported by all modern languages     |
  +--------------------------------------+
       |
  Level 2: Contract Programming
  +--------------------------------------+
  | "The input to this function is a      |
  |  positive integer, and the output is  |
  |  greater than the input"              |
  | Eiffel, Ada/SPARK, Kotlin (require)   |
  +--------------------------------------+
       |
  Level 3: Dependent Types
  +--------------------------------------+
  | "The length of this array is always N"|
  | "The output of this function is       |
  |  sorted"                              |
  | Idris, Agda, Lean, ATS               |
  +--------------------------------------+
       |
  Level 4: Complete Formal Proof
  +--------------------------------------+
  | "It is mathematically proven that this|
  |  program satisfies all requirements   |
  |  of the specification"                |
  | Coq, Isabelle, Lean 4                 |
  +--------------------------------------+
```

---

## 4. Notable New Languages and Technologies

### 4.1 Mojo (2023-)

Mojo is a language developed by Chris Lattner (the designer of LLVM and Swift) at Modular, aiming for "the ease of use of Python + the performance of C/Rust."

```
Mojo's Positioning:

  Ease of use
  ^
  |  Python *
  |           \
  |            * Mojo (targeting this position)
  |           /
  |  Rust  *
  |
  |  C/C++ *
  +--------------------> Performance
```

**Code Example 4: Mojo Code Example (Python-Compatible Syntax + High Performance)**

```python
# --- Mojo: Achieves high performance while maintaining Python syntax compatibility ---

# Python-compatible code (runs as-is)
def python_style():
    numbers = [1, 2, 3, 4, 5]
    total = sum(numbers)
    print(total)

# Mojo-specific high-performance code
struct Matrix:
    var data: DTypePointer[DType.float64]
    var rows: Int
    var cols: Int

    fn __init__(inout self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self.data = DTypePointer[DType.float64].alloc(rows * cols)

    fn __getitem__(self, row: Int, col: Int) -> Float64:
        return self.data.load(row * self.cols + col)

    fn __setitem__(inout self, row: Int, col: Int, value: Float64):
        self.data.store(row * self.cols + col, value)

    # Parallel computation using SIMD
    fn matmul(self, other: Matrix) -> Matrix:
        var result = Matrix(self.rows, other.cols)
        for i in range(self.rows):
            for j in range(other.cols):
                var sum: Float64 = 0.0
                @parameter
                fn dotsimd_width: Int:
                    sum += (self[i, k] * other[k, j])
                vectorizedot, 8
                result[i, j] = sum
        return result

# fn (Mojo-specific) vs def (Python-compatible)
# fn: Strict type checking, fast
# def: Python-compatible, dynamically typed
```

### 4.2 Gleam (2024-)

Gleam is a typed functional language that runs on the Erlang VM (BEAM). It leverages the Elixir ecosystem while benefiting from a static type system.

```
Gleam's Design Philosophy:

  +--------------------------------------------------+
  | Gleam = Erlang/Elixir concurrency                 |
  |       + Static type system                        |
  |       + JavaScript target                         |
  |       + Friendly error messages                   |
  +--------------------------------------------------+
  |                                                   |
  |  Erlang (1986)                                    |
  |    +-- Fault tolerance, concurrency,              |
  |    |   distributed processing                     |
  |    +-- Elixir (2012)                              |
  |         +-- Modern syntax, macros                 |
  |              +-- Gleam (2024)                      |
  |                   +-- Static types, JS target     |
  |                                                   |
  +--------------------------------------------------+
```

```gleam
// Gleam code example
import gleam/io
import gleam/list
import gleam/result
import gleam/string

// Algebraic data types
pub type OrderStatus {
  Pending
  Confirmed(confirmed_at: String)
  Shipped(tracking_number: String)
  Delivered
  Cancelled(reason: String)
}

// Pattern matching
pub fn describe_status(status: OrderStatus) -> String {
  case status {
    Pending -> "Order received"
    Confirmed(at) -> "Confirmed: " <> at
    Shipped(tracking) -> "Shipped: " <> tracking
    Delivered -> "Delivered"
    Cancelled(reason) -> "Cancelled: " <> reason
  }
}

// Error handling with the Result type
pub fn parse_order(input: String) -> Result(Order, String) {
  use id <- result.try(parse_id(input))
  use amount <- result.try(parse_amount(input))
  Ok(Order(id: id, amount: amount, status: Pending))
}

// Pipe operator
pub fn process_orders(orders: List(Order)) -> String {
  orders
  |> list.filter(fn(o) { o.amount > 1000.0 })
  |> list.sort(fn(a, b) { float.compare(b.amount, a.amount) })
  |> list.map(fn(o) { describe_status(o.status) })
  |> string.join(", ")
}
```

### 4.3 Zig (2016-)

Zig is a language aiming to be "a modern replacement for C," characterized by compile-time execution (comptime) and the principle of "no hidden control flow."

```
Zig's Design Principles:

  +--------------------------------------------------+
  | Three Principles of Zig:                          |
  |                                                   |
  | 1. No Hidden Control Flow                         |
  |    - No implicit allocators                       |
  |    - No implicit function calls                   |
  |      (no operator overloading)                    |
  |    - No implicit error ignoring                   |
  |                                                   |
  | 2. Compile-Time Execution (comptime)              |
  |    - comptime instead of generics                 |
  |    - comptime instead of conditional compilation  |
  |    - Constant computation via comptime            |
  |                                                   |
  | 3. Full Interoperability with C                   |
  |    - Direct import of C headers                   |
  |    - Link C libraries as-is                       |
  |    - Incremental migration from existing          |
  |      C codebases                                  |
  +--------------------------------------------------+
```

### 4.4 Carbon (Google, 2022-)

Carbon is an experimental language developed by Google that aims to be "the successor to C++." It emphasizes gradual interoperability with C++.

### 4.5 Vale (Research Stage)

Vale explores a new approach (Generational References) for achieving memory safety while avoiding the complexity of Rust's borrow checker.

### 4.6 Verse (Epic Games, 2023-)

Verse is a language developed by Epic Games, the creator of Unreal Engine, specializing in game development. It has a unique paradigm that integrates functional logic programming with concurrency.

### 4.7 New Language Comparison Table

| Language | Purpose | Features | Target | Status |
|----------|---------|----------|--------|--------|
| Mojo | AI/ML + General Purpose | Python-compatible + High Performance | LLVM | Early Access |
| Gleam | Web + Distributed | Typed BEAM + JS | BEAM / JS | v1.0 Released |
| Zig | Systems | comptime + C Interop | Native | Pre-stable |
| Carbon | C++ Successor | Gradual Migration | LLVM | Experimental |
| Vale | Safe Systems | Generational Refs | LLVM | Research |
| Verse | Gaming | Functional Logic | Unreal Engine | Preview |

---

## 5. Next-Generation Type Theory and Language Features

### 5.1 Effect Systems

An effect system is a mechanism that tracks function side effects at the type level. In current languages, it is not apparent from the type whether "this function performs I/O" or "this function throws an exception," but in an effect system, this information is included in the type.

```
Concept of Effect Systems:

  Traditional Type System:
  +--------------------------------------+
  | fn read_file(path: &str) -> String   |
  | // Does it perform I/O? Throw        |
  | // exceptions? Unknown                |
  +--------------------------------------+

  Effect System:
  +--------------------------------------+
  | fn read_file(path: &str)             |
  |   -> String                          |
  |   performs IO, throws FileNotFound   |
  | // Side effects are explicitly part  |
  | // of the type                       |
  +--------------------------------------+

  Effect Composition:
  +--------------------------------------+
  | fn process()                         |
  |   performs IO + DB + Log             |
  |   throws ParseError + DBError        |
  |                                      |
  | // Effects of called functions are   |
  | // automatically composed            |
  +--------------------------------------+

  Notable Implementations:
  - Koka (Microsoft Research)
  - Effekt (University of Tubingen)
  - Eff (University of Ljubljana)
  - Unison (commercial)
```

### 5.2 Dependent Types

Dependent types are a type system that allows describing types that depend on values. This makes it possible to express concepts such as "an array of length N," "a sorted list," or "a positive integer" as types.

**Code Example 5: Concept of Dependent Types (Idris-Style Pseudocode)**

```idris
-- Idris: Examples of dependent types

-- Length-indexed list (Vector)
data Vect : Nat -> Type -> Type where
    Nil  : Vect 0 a
    (::) : a -> Vect n a -> Vect (n + 1) a

-- Length is guaranteed at the type level
append : Vect n a -> Vect m a -> Vect (n + m) a
append Nil       ys = ys
append (x :: xs) ys = x :: append xs ys

-- Length matching is verified at compile time
zip : Vect n a -> Vect n b -> Vect n (a, b)
zip Nil       Nil       = Nil
zip (x :: xs) (y :: ys) = (x, y) :: zip xs ys

-- Safely retrieve the head element of a non-empty list
head : Vect (n + 1) a -> a
head (x :: _) = x
-- head Nil is a type error (Vect 0 a does not match Vect (n+1) a)

-- Sorted list
data SortedList : Type where
    Empty   : SortedList
    Single  : Nat -> SortedList
    Cons    : (x : Nat) -> (rest : SortedList) ->
              (prf : LTE x (headOf rest)) -> SortedList

-- The return type of insert guarantees "sortedness"
insert : Nat -> SortedList -> SortedList

-- Matrix multiplication: Dimension consistency verified at compile time
matmul : Matrix m n -> Matrix n p -> Matrix m p
-- Matrix 3 4 * Matrix 4 5 -> Matrix 3 5 (OK)
-- Matrix 3 4 * Matrix 5 6 -> Compile error (dimension mismatch)
```

### 5.3 Linear Types

Linear types are a mechanism that guarantees at the type level that "a value is used exactly once." This provides static guarantees of safety for resource management (file handles, network connections, locks).

```
Concept of Linear Types:

  Normal Types:
  +--------------------------------------+
  | let x = open_file("data.txt")        |
  | read(x)   // First use               |
  | read(x)   // Second use (OK)         |
  | // Compiles even if close is forgot  |
  +--------------------------------------+

  Linear Types:
  +--------------------------------------+
  | let x: Linear<File> = open("data.txt")|
  | let (data, x2) = read(x)  // Consumes x |
  | close(x2)                 // Consumes x2 |
  |                                      |
  | // Using x twice is a compile error  |
  | // Forgetting close is a compile error|
  | // Resource leaks are structurally   |
  | // impossible                        |
  +--------------------------------------+

  Rust's ownership system is an approximation of linear types:
  - Move semantics ~ Linear types
  - Borrowing ~ Relaxation of usage count
  - Drop trait ~ Implicit close
```

---

## 6. Long-Term Outlook

### 6.1 Predictions for 5 Years From Now (Around 2030)

```
Programming Language Landscape in 2030 (Prediction):

  +--------------------------------------------------+
  | Nearly Certain Changes:                            |
  | - TypeScript will effectively replace JavaScript   |
  |   entirely                                         |
  | - Rust becomes the first choice for systems        |
  |   programming                                      |
  | - Python type hints become "expected by default"   |
  | - AI assistants integrated into development        |
  |   workflows                                        |
  | - Wasm used routinely on the server side           |
  +--------------------------------------------------+
  | Likely Changes:                                    |
  | - Kotlin Multiplatform becomes a strong option     |
  |   for cross-platform development                   |
  | - Mojo becomes a complement to Python in the       |
  |   AI/ML domain                                     |
  | - Go generics mature, improving expressiveness     |
  | - C# evolves further across the .NET ecosystem     |
  +--------------------------------------------------+
  | Uncertain but Noteworthy Changes:                  |
  | - Introduction of effect systems into practical    |
  |   languages                                        |
  | - Standardization of quantum computing languages   |
  | - Full release and adoption of Carbon              |
  | - Gleam establishes its position in the BEAM       |
  |   ecosystem                                        |
  +--------------------------------------------------+
```

### 6.2 Predictions for 10 Years From Now (Around 2035)

```
Predictions for 2035:

  The Shape of Programming:
  +--------------------------------------------------+
  |                                                   |
  | Human Roles:                                      |
  | +-----------------------------------------------+ |
  | | - Requirements definition and specification    | |
  | |   (natural language + formal specifications)   | |
  | | - Architecture design and technology selection | |
  | | - Review and approval of AI-generated code     | |
  | | - Providing domain knowledge                   | |
  | | - Ethical judgment and trade-off decisions      | |
  | +-----------------------------------------------+ |
  |                     <-> Collaboration              |
  | AI Roles:                                         |
  | +-----------------------------------------------+ |
  | | - Generation of routine code                   | |
  | | - Automated test generation and execution      | |
  | | - Bug detection and fix suggestions            | |
  | | - Performance optimization                     | |
  | | - Documentation generation                     | |
  | +-----------------------------------------------+ |
  |                                                   |
  +--------------------------------------------------+

  Language Evolution:
  +--------------------------------------------------+
  | - Natural language -> Formal specs -> Code         |
  |   pipeline                                         |
  | - Introduction of dependent types into practical   |
  |   languages                                        |
  | - Standardization of effect systems                |
  | - Emergence of quantum-classical hybrid languages  |
  | - AI-native language design                        |
  |   (syntax that is easy for AI to generate and      |
  |    verify)                                         |
  +--------------------------------------------------+
```

### 6.3 Outlook for 20 Years From Now (Around 2045)

As a longer-term outlook, the following possibilities exist. However, the accuracy of technology predictions decreases as the time horizon extends.

```
Possibilities for 2045:

  +--------------------------------------------------+
  | Optimistic Scenario:                               |
  | - Programming fully shifts to "expression of       |
  |   intent"                                          |
  | - Write specs in natural language, and AI           |
  |   generates the optimal implementation             |
  | - Formal verification applied to all software      |
  | - Bugs exist only as "design flaws," and           |
  |   "implementation mistakes" are eliminated         |
  +--------------------------------------------------+
  | Realistic Scenario:                                |
  | - Human-AI collaboration deepens, but full         |
  |   automation does not occur                        |
  | - Low-level systems programming still requires     |
  |   human expertise                                  |
  | - New languages emerge for new computing           |
  |   paradigms (quantum, neuromorphic)                |
  +--------------------------------------------------+
  | Immutable Principles:                              |
  | - Fundamentals of computer science theory do       |
  |   not change                                       |
  | - Algorithms, data structures, and type theory     |
  |   are permanent                                    |
  | - The limits of computability (halting problem,    |
  |   etc.) are not overcome                           |
  | - The judgment of "what should be built" remains   |
  |   in the human domain                              |
  +--------------------------------------------------+
```

---

## 7. The Future of Programmers

### 7.1 The Role of Humans in the AI Era

```
Evolution of Human Roles in Software Development:

  1960-1980: Humans handle all stages
  +--------------------------------------+
  | Reqs -> Design -> Impl -> Test -> Ops|
  | [Human] [Human] [Human] [Human] [Human] |
  +--------------------------------------+

  2000-2020: Partial automation through tools
  +--------------------------------------+
  | Reqs -> Design -> Impl -> Test -> Ops|
  | [Human] [Human] [Human] [Tools] [Tools] |
  |                [IDE]   [CI/CD] [Monitoring] |
  +--------------------------------------+

  2025-2035: Extensive automation through AI
  +--------------------------------------+
  | Reqs -> Design -> Impl -> Test -> Ops|
  | [Human] [Human] [AI]   [AI]   [AI]  |
  | [AI     [AI     [Human  [Human [Human|
  | assist] assist] review] oversight] judgment] |
  +--------------------------------------+

  2035 and beyond: Humans focus on judgment and creativity
  +--------------------------------------+
  | Reqs -> Design -> Impl -> Test -> Ops|
  | [Human] [Human] [AI]   [AI]   [AI]  |
  | [Creation] [Judgment] [Verification] [Verification] [Monitoring] |
  +--------------------------------------+
```

### 7.2 Skills That Stay and Skills That Change

| Category | Skills Growing in Importance | Skills Declining in Importance |
|----------|------------------------------|-------------------------------|
| Design | Architecture design, trade-off analysis | Detailed implementation design |
| Coding | Type design, specification writing, code review | Routine code writing |
| Testing | Test strategy planning, quality standard setting | Writing simple unit tests |
| Operations | Incident judgment, root cause analysis | Routine monitoring and response |
| Communication | Precise instructions to AI, requirements clarification | Manual document creation |
| Knowledge | CS fundamentals, domain knowledge | Memorizing syntax of specific languages |

### 7.3 The Enduring Value of CS Fundamentals

```
CS Fundamental Knowledge That Increases in Value in the AI Era:

  +--------------------------------------------------+
  | 1. Theory of Computation                          |
  |    - Computability: What is fundamentally          |
  |      computable                                    |
  |    - Complexity theory: P vs NP, algorithmic       |
  |      limits                                        |
  |    - Halting problem: Limits of automatic          |
  |      verification of program correctness           |
  |    -> Essential for understanding the limits       |
  |       of AI-generated code                         |
  +--------------------------------------------------+
  | 2. Type Theory                                    |
  |    - Mathematical foundations of type safety       |
  |    - Curry-Howard correspondence:                  |
  |      Types = Propositions, Programs = Proofs       |
  |    - Parametric polymorphism                       |
  |    -> Essential for verifying correctness of       |
  |       AI-generated code via types                  |
  +--------------------------------------------------+
  | 3. Algorithms and Data Structures                 |
  |    - Time complexity and space complexity          |
  |    - Choosing appropriate data structures          |
  |    - Divide and conquer, dynamic programming,      |
  |      greedy algorithms                             |
  |    -> Essential for evaluating the efficiency      |
  |       of AI-generated code                         |
  +--------------------------------------------------+
  | 4. Distributed Systems                            |
  |    - CAP theorem: Trade-offs between consistency,  |
  |      availability, and partition tolerance          |
  |    - Consensus algorithms                          |
  |    - Eventual consistency                          |
  |    -> System design decisions remain in the        |
  |       human domain                                 |
  +--------------------------------------------------+
  | 5. Security                                       |
  |    - Threat modeling                               |
  |    - Cryptography fundamentals                     |
  |    - Trust boundary design                         |
  |    -> Security decisions cannot be fully            |
  |       delegated to AI                              |
  +--------------------------------------------------+
```

---

## 8. Anti-Patterns

### 8.1 Anti-Pattern 1: "Silver Bullet Thinking"

The tendency to expect a new language or technology to be a "panacea that solves all problems."

```
Typical Patterns of Silver Bullet Thinking:

  "If we use Rust, all bugs will disappear"
  -> Rust guarantees memory safety, but it cannot
     prevent business logic bugs, design flaws,
     or misunderstood requirements

  "AI writes code, so learning programming is unnecessary"
  -> Evaluating, correcting, and integrating AI output
     requires deep programming knowledge

  "We should rewrite everything in functional programming"
  -> A judgment that ignores the team's skill set,
     existing codebase, and ecosystem maturity

  The Right Approach:
  +--------------------------------------------------+
  | Always ask: "For this problem, what does this      |
  | technology solve, and what does it NOT solve?"     |
  |                                                    |
  | Trade-offs never disappear. They only move.        |
  +--------------------------------------------------+
```

### 8.2 Anti-Pattern 2: "Trend Chasing"

A pattern of jumping on the latest languages and frameworks at the expense of project stability.

```
Problems with Trend Chasing:

  Chain of Bad Decisions:
  +------------------------------------------+
  | "New language X is trending"              |
  |         |                                 |
  | "Let's rewrite our existing project in X" |
  |         |                                 |
  | "The ecosystem is immature, and           |
  |  productivity drops"                      |
  |         |                                 |
  | "We need to read library source code      |
  |  to fix bugs"                             |
  |         |                                 |
  | "New language Y is trending.              |
  |  Let's migrate to Y..."                   |
  |         |                                 |
  | An endless loop                           |
  +------------------------------------------+

  The Right Approach:
  +------------------------------------------+
  | Evaluation Criteria for New Technologies: |
  | 1. Ecosystem maturity (number of          |
  |    libraries)                             |
  | 2. Community activity                     |
  | 3. Availability of commercial support     |
  | 4. Team learning cost                     |
  | 5. Ease of integration with existing      |
  |    systems                                |
  | 6. Likelihood of still being used in      |
  |    5 years                                |
  +------------------------------------------+
```

### 8.3 Anti-Pattern 3: "Blindly Delegating to AI"

A pattern of completely delegating implementation to AI code generation tools and using the generated code without verification.

```
Risks of Blindly Delegating to AI:

  +------------------------------------------+
  | Potential Issues with AI-Generated Code:  |
  |                                          |
  | 1. Security Vulnerabilities              |
  |    - SQL injection                       |
  |    - XSS                                 |
  |    - Authentication/authorization flaws  |
  |                                          |
  | 2. Performance Issues                    |
  |    - N+1 queries                         |
  |    - Unnecessary memory allocations      |
  |    - Inefficient algorithm choices       |
  |                                          |
  | 3. Design Issues                         |
  |    - Excessive coupling                  |
  |    - Hard-to-test structures             |
  |    - Lack of scalability                 |
  |                                          |
  | 4. Licensing Issues                      |
  |    - License contamination from          |
  |      training data                       |
  |    - Possible GPL contamination          |
  |                                          |
  | Countermeasures:                         |
  | - Always review generated code           |
  | - Run type checks + tests + static       |
  |   analysis                               |
  | - Run security scans                     |
  | - Treat AI output as a "draft"           |
  +------------------------------------------+
```

---

## 9. Quantum Computing and Languages

### 9.1 The Current State of Quantum Programming Languages

Quantum computing has the potential to bring about a fundamental transformation of the computing paradigm. Currently, multiple quantum programming languages and frameworks are under development.

```
Current State of Quantum Programming:

  +--------------------------------------------------+
  |           Quantum Programming Frameworks           |
  +--------------+--------------+------------------+
  | Qiskit       | Cirq          | Q#              |
  | (IBM)        | (Google)      | (Microsoft)     |
  +--------------+--------------+------------------+
  | Python       | Python        | Proprietary     |
  | based        | based         | language         |
  |              |               | (F#-like)        |
  |              |               |                  |
  | Circuit      | Circuit       | High-level       |
  | construction | construction  | quantum          |
  | + execution  | + simulator   | algorithms       |
  +--------------+--------------+------------------+

  Quantum-Classical Hybrid:
  +--------------------------------------------------+
  |                                                   |
  |   Classical Computer        Quantum Computer      |
  |   +--------------+      +--------------+         |
  |   | Control flow  | ---> | Quantum      |         |
  |   | Data pre-     |      | circuit exec |         |
  |   | processing    |      | Superposition|         |
  |   | Result        | <--- | Measurement  |         |
  |   | interpretation|      |              |         |
  |   +--------------+      +--------------+         |
  |                                                   |
  |   Future languages must be able to seamlessly     |
  |   describe both classical and quantum code        |
  +--------------------------------------------------+
```

### 9.2 Challenges in Quantum Programming

```
Features Required by Quantum Programming Languages:

  1. Qubit Management
     - Tracking entanglement
     - Qubit lifetime management
     - Type-level guarantee of the No-Cloning theorem

  2. Measurement and the Classical/Quantum Boundary
     - Making wavefunction collapse from measurement explicit
     - Interaction with classical bits

  3. Error Correction
     - Abstraction of quantum error correction codes
     - Integration of noise models

  4. Optimization
     - Quantum circuit optimization (minimizing gate count)
     - Optimization of quantum-classical hybrid algorithms
```

---

## 10. The Democratization of Programming

### 10.1 Relationship with No-Code/Low-Code

```
Stratification of Programming:

  Abstraction Level
  ^
  |  +------------------------+
  |  | Natural Language        | <- AI Era
  |  | Programming             |
  |  | (AI + NL input)         |
  |  +------------------------+
  |  | No-Code/Low-Code        | <- Business Users
  |  | (Visual tools)          |
  |  +------------------------+
  |  | High-Level Languages    | <- General Developers
  |  | (Python, TS, etc.)      |
  |  +------------------------+
  |  | Systems Languages       | <- Systems Engineers
  |  | (Rust, C, Go)           |
  |  +------------------------+
  |  | Assembly / LLVM IR      | <- Specialists
  |  +------------------------+
  +-----------------------------> Granularity of Control

  Changes in the AI Era:
  - AI makes it easier to move between layers
  - "Instruct in natural language -> AI generates
    high-level code -> Compiler transforms to
    systems level"
  - The "entry point" to programming broadens
    significantly
  - However, the need for "deep understanding"
    does not change
```

### 10.2 The Expansion of Domain-Specific Programming

In the future, each industry is expected to have its own DSL, with domain experts "programming" directly with AI assistance.

```
Expansion of Domain-Specific Programming:

  +----------+------------------------------------+
  | Industry | DSL / Specialized Tool Examples     |
  +----------+------------------------------------+
  | Finance  | Contract DSL, risk calculation DSL  |
  | Medical  | Clinical trial protocols,           |
  |          | diagnostic rules                    |
  | Legal    | Contract logic,                     |
  |          | compliance rules                    |
  | Manufact.| Control logic,                      |
  |          | quality management rules            |
  | Education| Curriculum design,                  |
  |          | learning path definition            |
  | Agricult.| Irrigation control,                 |
  |          | harvest optimization                |
  +----------+------------------------------------+

  Future Vision:
  "A financial analyst describes a trading strategy
   in natural language
   -> AI converts it to type-safe DSL code
   -> Formal verification guarantees correctness
   -> Backtesting runs automatically"
```

---

## 11. Exercises

### 11.1 Beginner: Understanding Concepts

**Exercise 1:** Explain in 2-3 sentences the background of "why" each of the following trends is occurring.
1. The spread of gradual typing
2. Government-level demands for memory safety
3. The expansion of WebAssembly beyond browsers
4. The normalization of AI code generation

**Exercise 2:** Summarize the design philosophy of each of the following new languages in 1-2 sentences.
1. Mojo
2. Gleam
3. Zig
4. Carbon

**Exercise 3:** Among the "Five Laws of Language Evolution," identify which law the following phenomena correspond to.
- Go's intentionally feature-limited design -> Law ?
- TypeScript being built on the JavaScript ecosystem -> Law ?
- Rust, Kotlin, and Swift all adopting the Option type -> Law ?

### 11.2 Intermediate: Analysis and Prediction

**Exercise 4:** For the following hypothesis, provide 3 arguments for and 3 arguments against.

"In 10 years, 80% of programmers will program in natural language, and writing traditional programming languages directly will be limited to 20% of specialists."

**Exercise 5:** List 5 conditions that make a language compatible with AI code generation tools, and evaluate each of the following existing languages (Rust, TypeScript, Python, Go, Kotlin) against them.

```
Example evaluation table:
+----------------+------+----+--------+----+--------+
| Condition       | Rust | TS | Python | Go | Kotlin |
+----------------+------+----+--------+----+--------+
| Condition 1: ???|  ?   | ?  |   ?    | ?  |   ?    |
| Condition 2: ???|  ?   | ?  |   ?    | ?  |   ?    |
| ...            |      |    |        |    |        |
+----------------+------+----+--------+----+--------+
```

### 11.3 Advanced: Design and Creation

**Exercise 6:** Draft the specification for a new general-purpose programming language to be designed in 2030.

Deliverables:
- Language name and design philosophy (3 principles)
- Target domain
- Features to adopt (referencing this and preceding chapters)
- Features not to adopt and reasons why
- Design considerations for AI collaboration
- Sample code (approximately Fizz Buzz + Web server level)
- Differentiation points from existing languages

**Exercise 7:** Read the following scenario and make a technology selection for the team.

```
Scenario:
+----------------------------------------------+
| It is 2028 and you are the CTO of a startup. |
| You are developing a medical AI assistance    |
| system.                                       |
|                                              |
| Requirements:                                 |
| - Patient data processing (high security      |
|   requirements)                               |
| - Real-time image analysis (GPU processing)   |
| - Web admin panel                             |
| - Mobile app (iOS/Android)                    |
| - Regulatory compliance (medical device       |
|   software standards)                         |
| - Team: 10 engineers (varying experience      |
|   levels)                                     |
|                                              |
| Available technologies (projected for 2028):  |
| - Rust, Go, Kotlin, TypeScript, Python, Mojo  |
| - AI-assisted development tools               |
| - Formal verification tools                   |
+----------------------------------------------+
```

State the language selection and rationale for each component.

---


## FAQ

### Q1: What is the most important point to focus on when learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not through theory alone, but by actually writing code and verifying its behavior.

### Q2: What are common mistakes that beginners make?

Skipping the fundamentals and jumping ahead to advanced topics. We recommend building a solid understanding of the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this topic applied in practice?

The knowledge from this topic is frequently applied in everyday development work. It becomes especially important during code reviews and architecture design.

---

## 12. Summary

### 12.1 Trend Summary Table

| Trend | Direction | Impact | Certainty |
|-------|-----------|--------|-----------|
| Gradual Typing | Dynamic -> Static transition | TypeScript, Python type hints | Certain |
| Memory Safety | C/C++ -> Rust | Government-level demand | Certain |
| Wasm | Universal binary | Multi-platform | High |
| AI Collaboration | Normalization of code generation | Increased importance of types and specs | Certain |
| Declarative Description | What > How | Increased abstraction | High |
| Effect Systems | Type-level side effect tracking | Influences next-gen languages | Moderate |
| Dependent Types | Value-dependent types | Transitioning from research to practice | Moderate |
| Quantum Languages | Quantum-classical hybrid | New computing paradigm | Uncertain |

### 12.2 Core Message

```
The Future of Programming Languages:

  +--------------------------------------------------+
  |                                                   |
  |  Languages change, but principles do not          |
  |                                                   |
  | +-----------------------------------------------+ |
  | | Things That Change:                            | |
  | | - Syntax                                       | |
  | | - Toolchains                                   | |
  | | - Levels of abstraction                        | |
  | | - Development workflows                        | |
  | | - Methods of AI collaboration                  | |
  | +-----------------------------------------------+ |
  | | Things That Do Not Change:                     | |
  | | - Foundations of computation theory             | |
  | |   (Turing, lambda calculus)                    | |
  | | - Algorithms and data structures               | |
  | | - Type theory (Curry-Howard correspondence)    | |
  | | - Fundamental principles of security           | |
  | | - The ability to judge "what should be built"  | |
  | +-----------------------------------------------+ |
  |                                                   |
  |  Conclusion: Understanding CS fundamentals        |
  |              increases in value precisely          |
  |              in the AI era                         |
  |                                                   |
  +--------------------------------------------------+
```

---

## 13. FAQ (Frequently Asked Questions)

### Q1: In an era where AI writes code, is there still a reason to learn programming?

**A:** Absolutely. Just as the ability to judge "what should be written" remains essential for humans even in an era when AI can generate text, the ability to judge "what should be built," "is the generated code correct," and "is the security and performance sufficient" is necessary for humans even when AI can generate code. Furthermore, evaluating the quality of AI output requires a deep understanding of programming as a prerequisite. In practice, the productivity of programmers who can effectively leverage AI has dramatically increased compared to those who cannot or do not use AI, and the value of programming skills has actually grown.

### Q2: Is it better to learn a new language or to deepen expertise in an existing language?

**A:** The fundamental strategy is to aim for "T-shaped skills." Master one language deeply (the vertical bar of the T) and have a broad familiarity with the basics of 2-3 other languages (the horizontal bar of the T). For the language you study deeply, choose the one most used in your current projects. For languages you study broadly, choosing languages from different paradigms widens your perspective (e.g., one procedural + one functional + one systems language). Try new languages in "side projects" and carefully evaluate maturity before introducing them to production projects.

### Q3: Will Rust really replace C/C++?

**A:** A complete replacement will take a long time, but the number of cases where Rust is chosen over C/C++ for new projects is steadily increasing. Rust adoption in the Linux kernel, Android, Windows, and Chromium is accelerating this trend. However, the existing C/C++ codebase is enormous (on the order of billions of lines), and rewriting it is a process that spans decades. The realistic scenario is an incremental migration where "new components are written in Rust, and existing C/C++ code is integrated via FFI." Considering that COBOL is still in use more than 50 years later, C/C++ will similarly be used for a long time.

### Q4: Will WebAssembly replace Docker?

**A:** It has the potential to replace some Docker use cases, but will not be a complete substitute. Wasm's strengths are sandboxed execution, fast startup times (on the order of milliseconds), and small binary sizes. For lightweight workloads like edge computing or serverless functions, Wasm may be more suitable than Docker. Solomon Hykes (Docker co-founder) said, "If WASM+WASI existed in 2008, we wouldn't have needed to create Docker." However, Docker provides many features that Wasm does not cover, including full Linux environment emulation, containerization of existing applications, and complex network configurations. The two are moving toward coexistence in their respective niches rather than competing.

### Q5: Will effect systems and dependent types be introduced into practical languages?

**A:** They are being introduced incrementally. The concept of effect systems is already partially embedded in existing languages through Kotlin coroutines (the `suspend` modifier is a form of effect annotation), Java's checked exceptions (a coarse approximation of effects), and Rust's `async` (an async effect). Full effect systems have reached the practical stage in research-oriented languages like Koka and Unison. For dependent types, TypeScript's type-level programming (template literal types, conditional types) is being used practically as an "approximation of dependent types." Full dependent types are being used practically in Lean 4 as a tool for mathematical formalization. Complete introduction into mainstream languages is expected to take another 5-10 years, but partial introduction is already underway.

---

## Recommended Next Guides


---

## References

1. The White House, Office of the National Cyber Director. "Back to the Building Blocks: A Path Toward Secure and Measurable Software." February 2024. - A U.S. government report recommending the transition away from memory-unsafe programming languages. A historic document recognizing language choice as part of security policy.
2. Haas, A. et al. "Bringing the Web up to Speed with WebAssembly." Proceedings of the 38th ACM SIGPLAN Conference on Programming Language Design and Implementation, 2017. - A paper explaining the design philosophy and technical foundation of WebAssembly. Details Wasm's performance characteristics and security model.
3. JetBrains. "The State of Developer Ecosystem." Annual Survey, 2024. - A large-scale survey of developers worldwide. Provides data on language adoption trends, tool usage, and development trends.
4. Lattner, C. "Mojo: A New Programming Language for AI." Modular, 2023. - Explains the design intent and technical decisions behind Mojo. Details the design decisions for achieving both Python compatibility and high performance.
5. Siek, J. and Taha, W. "Gradual Typing for Functional Languages." Scheme and Functional Programming Workshop, 2006. - The paper that established the theoretical foundation of gradual typing. Presents a formal framework for integrating dynamic and static typing.

---

## Glossary

| Term | Description |
|------|-------------|
| Gradual Typing | A technique for mixing dynamic and static typing |
| Memory Safety | A guarantee that buffer overflows and dangling pointers do not occur |
| WebAssembly (Wasm) | A portable binary instruction format |
| WASI | WebAssembly System Interface. A standard that provides OS capabilities to Wasm |
| Effect System | A mechanism for tracking function side effects via types |
| Dependent Types | A type system that can describe types depending on values |
| Linear Types | Types that guarantee a value is used exactly once |
| Formal Verification | A technique for mathematically proving program correctness |
| Ownership | Rust's memory management model based on value ownership and borrowing |
| Convergent Evolution | A phenomenon where different lineages independently develop similar characteristics |
| No-Code/Low-Code | A technique for building applications with visual tools |
| Quantum Computing | A computing paradigm utilizing the principles of quantum mechanics |
| Curry-Howard Correspondence | The correspondence between types and logical propositions, and programs and proofs |
| T-Shaped Skills | A skill model characterized by deep expertise in one area and broad knowledge across multiple areas |
| Phantom Type | A type parameter that holds no data but is used for state tracking |
