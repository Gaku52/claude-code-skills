# How to Choose a Language

> The optimal language is determined by "what you build," "who builds it," and "where it runs."
> There is no silver bullet -- all that exists is "evaluating trade-offs."

Choosing a programming language is one of the most critical early decisions that can determine the success or failure of a software project. An appropriate language choice can multiply a team's productivity, while a poor choice becomes a breeding ground for technical debt. This chapter systematically explains the knowledge, frameworks, and practical methods needed for language selection.

---

## Learning Objectives

- [ ] Select a language based on project requirements
- [ ] Understand the strengths, weaknesses, and application domains of each language
- [ ] Utilize a decision framework for language selection
- [ ] Relate team composition and organizational strategy to language selection
- [ ] Understand differences in language paradigms, type systems, and execution models
- [ ] Make rational selections by avoiding anti-patterns
- [ ] Design polyglot strategies that combine multiple languages


## Prerequisites

Before reading this guide, the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related fundamental concepts
- Familiarity with the content of [Programming Paradigms Overview](./02-paradigms-overview.md)

---

## Overall Structure of This Chapter

```
+===========================================================================+
|                    Language Selection Guide: Full Map                       |
+===========================================================================+
|                                                                           |
|  [Chapter 1] Standard Languages by Domain                                 |
|     |                                                                     |
|     v                                                                     |
|  [Chapter 2] Technical Characteristics Comparison                         |
|     |                                                                     |
|     v                                                                     |
|  [Chapter 3] Language Paradigms and Design Philosophy                     |
|     |                                                                     |
|     v                                                                     |
|  [Chapter 4] Decision Framework for Language Selection ---+               |
|     |                                                     |               |
|     v                                                     v               |
|  [Chapter 5] Team & Organization          [Chapter 6] 2025 Trends        |
|     |                                                     |               |
|     v                                                     v               |
|  [Chapter 7] Polyglot Strategy           [Chapter 8] Migration Strategy   |
|     |                                                     |               |
|     +------------+   +------------------+                                 |
|                  v   v                                                    |
|           [Chapter 9] Anti-Pattern Collection                             |
|                   |                                                       |
|                   v                                                       |
|           [Chapter 10] Practical Exercises                                |
|                   |                                                       |
|                   v                                                       |
|           [FAQ / Summary / References]                                    |
+===========================================================================+
```

---

## Chapter 1: Standard Languages by Domain

In each area of software development, there exist languages that can be called "standard." These have been shaped by historical context, ecosystem maturity, and community size. Unless there is a compelling reason otherwise, following the standard is the rational approach.

### 1.1 Web Development

```
+===================================================================+
|                      Web Development Language Map                   |
+===================================================================+
|                                                                   |
|  Browser (Client-Side)                                            |
|  +-------------------------------------------------------------+ |
|  | JavaScript / TypeScript  <- Virtually the only choice        | |
|  |   + Frameworks: React, Vue, Svelte, Angular                  | |
|  |   + Meta-frameworks: Next.js, Nuxt, SvelteKit, Remix         | |
|  |   + Build tools: Vite, Turbopack, esbuild                    | |
|  |   + WebAssembly (Rust/C++/Go) can accelerate some processing | |
|  +-------------------------------------------------------------+ |
|                          |  HTTP / WebSocket                      |
|                          v                                        |
|  Server (Backend)                                                 |
|  +-------------------------------------------------------------+ |
|  | TypeScript(Node/Deno/Bun) | Python | Go | Java/Kotlin        | |
|  | Rust | Ruby | PHP | C# | Elixir                              | |
|  +-------------------------------------------------------------+ |
|                          |  SQL / ORM / Driver                    |
|                          v                                        |
|  Data Layer                                                       |
|  +-------------------------------------------------------------+ |
|  | PostgreSQL | MySQL | MongoDB | Redis | DynamoDB               | |
|  +-------------------------------------------------------------+ |
+===================================================================+
```

#### Detailed Backend Language Comparison

```
  +----------------+-------------------------------------------+
  | Language        | Characteristics & Applications             |
  +----------------+-------------------------------------------+
  | TypeScript     | Full-stack unification. Optimal for small  |
  | (Node.js/Bun)  | to mid-scale. npm ecosystem. Real-time    |
  +----------------+-------------------------------------------+
  | Python         | AI/ML integration, prototyping, Django/    |
  | (FastAPI)      | Flask. Type hints + async for speed        |
  +----------------+-------------------------------------------+
  | Go             | High-performance APIs, microservices.      |
  |                | Rich standard library, easy deployment     |
  +----------------+-------------------------------------------+
  | Java/Kotlin    | Enterprise, large-scale teams.             |
  | (Spring Boot)  | Robust types, strong long-term maintenance |
  +----------------+-------------------------------------------+
  | Rust           | High performance + memory safety.          |
  | (Axum/Actix)   | CPU-intensive processing, system-level APIs|
  +----------------+-------------------------------------------+
  | Ruby (Rails)   | Rapid prototyping, MVP development.        |
  |                | Convention over Configuration              |
  +----------------+-------------------------------------------+
  | PHP (Laravel)  | Vast hosting support, WordPress.           |
  |                | Modern development possible with Laravel   |
  +----------------+-------------------------------------------+
  | C# (ASP.NET)  | Windows ecosystem, Azure integration.      |
  |                | WASM support possible via Blazor           |
  +----------------+-------------------------------------------+
  | Elixir         | Massive concurrent connections, real-time  |
  | (Phoenix)      | processing. Inherits Erlang VM's fault     |
  |                | tolerance                                  |
  +----------------+-------------------------------------------+
```

#### Code Examples: HTTP API Endpoints in Different Languages

Below, we implement the same "GET endpoint that returns user information" in 5 languages. This comparison clarifies the differences in syntax style, verbosity, and ecosystem of each language.

**TypeScript (Express)**

```typescript
// TypeScript + Express: Concise and type-safe API definition
import express, { Request, Response } from "express";

interface User {
  id: number;
  name: string;
  email: string;
  role: "admin" | "member" | "guest";
}

const users: User[] = [
  { id: 1, name: "Taro Tanaka", email: "tanaka@example.com", role: "admin" },
  { id: 2, name: "Hanako Sato", email: "sato@example.com", role: "member" },
];

const app = express();

// GET /users/:id - Retrieve user information
app.get("/users/:id", (req: Request, res: Response) => {
  const userId = parseInt(req.params.id, 10);
  const user = users.find((u) => u.id === userId);

  if (!user) {
    return res.status(404).json({ error: "User not found" });
  }
  return res.json(user);
});

app.listen(3000, () => {
  console.log("Server running on http://localhost:3000");
});
```

**Python (FastAPI)**

```python
# Python + FastAPI: Automatically generates API documentation from type hints
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum

class Role(str, Enum):
    admin = "admin"
    member = "member"
    guest = "guest"

class User(BaseModel):
    id: int
    name: str
    email: str
    role: Role

users_db: dict[int, User] = {
    1: User(id=1, name="Taro Tanaka", email="tanaka@example.com", role=Role.admin),
    2: User(id=2, name="Hanako Sato", email="sato@example.com", role=Role.member),
}

app = FastAPI()

# GET /users/{user_id} - Retrieve user information
@app.get("/users/{user_id}", response_model=User)
async def get_user(user_id: int) -> User:
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    return users_db[user_id]
```

**Go (Standard Library)**

```go
// Go: Can build an HTTP server using only the standard library
package main

import (
    "encoding/json"
    "net/http"
    "strconv"
    "strings"
)

type User struct {
    ID    int    `json:"id"`
    Name  string `json:"name"`
    Email string `json:"email"`
    Role  string `json:"role"`
}

var users = map[int]User{
    1: {ID: 1, Name: "Taro Tanaka", Email: "tanaka@example.com", Role: "admin"},
    2: {ID: 2, Name: "Hanako Sato", Email: "sato@example.com", Role: "member"},
}

func getUserHandler(w http.ResponseWriter, r *http.Request) {
    // Extract user ID from the path
    parts := strings.Split(r.URL.Path, "/")
    if len(parts) < 3 {
        http.Error(w, `{"error":"invalid path"}`, http.StatusBadRequest)
        return
    }

    id, err := strconv.Atoi(parts[2])
    if err != nil {
        http.Error(w, `{"error":"invalid id"}`, http.StatusBadRequest)
        return
    }

    user, ok := users[id]
    if !ok {
        http.Error(w, `{"error":"User not found"}`, http.StatusNotFound)
        return
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(user)
}

func main() {
    http.HandleFunc("/users/", getUserHandler)
    http.ListenAndServe(":3000", nil)
}
```

**Rust (Axum)**

```rust
// Rust + Axum: Compile-time type safety and zero-cost abstractions
use axum::{
    extract::Path,
    http::StatusCode,
    response::Json,
    routing::get,
    Router,
};
use serde::Serialize;
use std::collections::HashMap;
use std::sync::LazyLock;

#[derive(Serialize, Clone)]
struct User {
    id: u32,
    name: String,
    email: String,
    role: String,
}

static USERS: LazyLock<HashMap<u32, User>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    m.insert(1, User {
        id: 1,
        name: "Taro Tanaka".into(),
        email: "tanaka@example.com".into(),
        role: "admin".into(),
    });
    m.insert(2, User {
        id: 2,
        name: "Hanako Sato".into(),
        email: "sato@example.com".into(),
        role: "member".into(),
    });
    m
});

async fn get_user(
    Path(user_id): Path<u32>,
) -> Result<Json<User>, StatusCode> {
    USERS
        .get(&user_id)
        .cloned()
        .map(Json)
        .ok_or(StatusCode::NOT_FOUND)
}

#[tokio::main]
async fn main() {
    let app = Router::new().route("/users/:id", get(get_user));
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000")
        .await
        .unwrap();
    axum::serve(listener, app).await.unwrap();
}
```

**Java (Spring Boot)**

```java
// Java + Spring Boot: Annotation-driven declarative API definition
package com.example.api;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

record User(int id, String name, String email, String role) {}

@RestController
@RequestMapping("/users")
public class UserController {

    private final Map<Integer, User> users = new ConcurrentHashMap<>(Map.of(
        1, new User(1, "Taro Tanaka", "tanaka@example.com", "admin"),
        2, new User(2, "Hanako Sato", "sato@example.com", "member")
    ));

    // GET /users/{id} - Retrieve user information
    @GetMapping("/{id}")
    public ResponseEntity<User> getUser(@PathVariable int id) {
        User user = users.get(id);
        if (user == null) {
            return ResponseEntity.notFound().build();
        }
        return ResponseEntity.ok(user);
    }
}
```

> **Key Observation**: Even for identical functionality, the line count, syntax, and type expressiveness differ significantly. TypeScript and Python are concise with fast startup, Go is self-contained with just its standard library, Rust offers strong compile-time safety guarantees, and Java is characterized by declarative definitions through annotations.

### 1.2 Mobile Development

```
+-----------------------------------------------------------------+
|                     Mobile Development Options                    |
+-----------------------------------------------------------------+
|                                                                 |
|  Native Development                                             |
|  +---------------------------+  +----------------------------+  |
|  |        iOS               |  |        Android             |  |
|  |  Swift (recommended)      |  |  Kotlin (recommended)      |  |
|  |  Objective-C (legacy)     |  |  Java (legacy)             |  |
|  |  SwiftUI / UIKit          |  |  Jetpack Compose / XML     |  |
|  +---------------------------+  +----------------------------+  |
|                                                                 |
|  Cross-Platform                                                  |
|  +-----------------------------------------------------------+ |
|  | Flutter (Dart)                                             | |
|  |   - Custom rendering engine (Skia/Impeller)                | |
|  |   - Pixel-level control over UI details                    | |
|  |   - Full support: iOS / Android / Web / Desktop            | |
|  +-----------------------------------------------------------+ |
|  | React Native (JavaScript/TypeScript)                       | |
|  |   - Easy entry for web developers                          | |
|  |   - Calls native components through a bridge               | |
|  |   - Significantly improved DX with Expo                    | |
|  +-----------------------------------------------------------+ |
|  | Kotlin Multiplatform (KMP)                                 | |
|  |   - Share business logic in Kotlin                         | |
|  |   - Build UI with native platform toolkits                 | |
|  |   - Gradual adoption possible                              | |
|  +-----------------------------------------------------------+ |
+-----------------------------------------------------------------+
```

#### Mobile Language Selection Flowchart

```
  Need iOS only?
  |
  +-- Yes --> Swift (SwiftUI is the top candidate)
  |
  +-- No --> Need Android only?
              |
              +-- Yes --> Kotlin (Jetpack Compose is the top candidate)
              |
              +-- No --> Need both
                          |
                          +-- Does the team have many web developers?
                          |    |
                          |    +-- Yes --> React Native
                          |    +-- No  --> Is UI consistency the top priority?
                          |                |
                          |                +-- Yes --> Flutter
                          |                +-- No  --> KMP
```

### 1.3 Data Science & AI/ML

| Domain | Recommended Language | Major Libraries/Tools | Reason |
|------|---------|---------------------|------|
| Data Analysis | Python | pandas, polars, matplotlib | Overwhelmingly rich ecosystem |
| Machine Learning | Python | scikit-learn, XGBoost | Fastest research-to-implementation transition |
| Deep Learning | Python | PyTorch, JAX | Mature GPU computation integration |
| Data Engineering | Python + SQL | Spark, dbt, Airflow | Standard for pipeline construction |
| Statistical Analysis | R / Python | tidyverse, statsmodels | R excels at visualization and statistical tests |
| High-Speed Inference | C++ / Rust | ONNX Runtime, TensorRT | Production environments with strict latency requirements |
| LLM Applications | Python + TS | LangChain, LlamaIndex | Prototype in Python, frontend in TS |

### 1.4 Systems & Infrastructure

```
  Language Selection Map by Use Case
  =====================

  OS Kernels / Drivers
    └──> C / Rust
         (Direct hardware control, zero-overhead abstractions)

  Embedded / IoT
    └──> C / C++ / Rust / MicroPython
         (Memory constraints, real-time requirements)

  CLI Tools
    └──> Go / Rust
         (Single binary, cross-compilation)

  DevOps Scripts
    └──> Python / Bash / Go
         (Automation, text processing)

  Containers / Cloud Infrastructure
    └──> Go
         (Docker, Kubernetes, Terraform, Prometheus)

  Network Programming
    └──> Go / Rust / C++
         (High concurrency, low latency)

  Game Engines
    └──> C++ (Unreal) / C# (Unity) / Rust (Bevy)
         (Frame rate requirements, GPU control)
```

### 1.5 Domain-Specific Recommended Language Summary Table

| Domain | First Choice | Second Choice | Choices to Avoid |
|---------|---------|---------|--------------|
| Web Frontend | TypeScript | JavaScript | Java, Python |
| Web Backend (small-scale) | TypeScript / Python | Ruby / PHP | C++ |
| Web Backend (large-scale) | Go / Java / Kotlin | TypeScript / C# | Bash |
| iOS Apps | Swift | Flutter (Dart) | Java |
| Android Apps | Kotlin | Flutter (Dart) | Objective-C |
| Data Science | Python | R | Go, Rust |
| Machine Learning | Python | Julia | PHP, Ruby |
| Systems Programming | Rust | C / C++ | Python, JavaScript |
| CLI Tools | Go | Rust | Java |
| DevOps / Automation | Python | Go / Bash | C++ |
| Game Development | C# (Unity) / C++ | Rust (Bevy) | Python |
| Blockchain | Solidity / Rust | Go | PHP |

---

## Chapter 2: Technical Characteristics Comparison

In language selection, understanding technical characteristics is essential rather than relying on superficial "popularity." Here we compare major languages across 6 axes.

### 2.1 Type Systems

```
  Type System Spectrum
  =======================

  Dynamic Typing                                Static Typing
  (Type checked at runtime)                     (Type checked at compile time)
  <------------------------------------------------------>

  Python    Ruby    JavaScript   TypeScript   Go   Java   Rust   Haskell
  PHP       Elixir  Lua          Kotlin      C#   Swift  Scala  OCaml

                                     ^
                                     |
                               TypeScript is
                             a language that added
                              types to JavaScript

  +----------------------------------------------------------------+
  | Gradual Typing                                                  |
  |  Python (type hints), TypeScript, PHP (type declarations)       |
  |  -> Can gradually introduce types to dynamic languages          |
  +----------------------------------------------------------------+
```

#### Type System Strength Comparison Table

| Language | Typing | Type Inference | Null Safety | Generics | Algebraic Data Types |
|------|--------|--------|----------|-------------|--------------|
| Python | Dynamic (type hints available) | N/A | Optional type | Yes | match statement (3.10+) |
| JavaScript | Dynamic | N/A | None | N/A | None |
| TypeScript | Static (structural) | Powerful | strictNullChecks | Yes | Discriminated unions |
| Go | Static (nominal) | Inferred with `:=` | nil (pointers) | Yes (1.18+) | None |
| Java | Static (nominal) | var (10+) | Optional | Yes (type erasure) | sealed (17+) |
| Kotlin | Static (nominal) | Powerful | Built-in `?` | Yes (reifiable) | sealed class |
| Rust | Static (nominal) | Powerful | Option\<T\> | Yes (monomorphization) | enum (powerful) |
| Swift | Static (nominal) | Powerful | Optional `?` | Yes | enum + associated |
| Haskell | Static (nominal) | Hindley-Milner | Maybe | Yes (higher-kinded) | ADT (strongest) |

### 2.2 Memory Management Models

```
  Three Approaches to Memory Management
  ============================

  [Manual Management]          [Garbage Collection]          [Ownership System]
  C / C++                     Java, Go, Python,             Rust
                              JavaScript, C#,
                              Ruby, Kotlin

  Developer explicitly         Runtime automatically          Compiler analyzes
  calls malloc/free            tracks references and          memory lifetimes
                              frees unused memory            at compile time
                                                             -> No GC + memory safe

  Pros:                       Pros:                         Pros:
  - Maximum performance       - Low developer burden         - Safe without GC
  - Predictable latency       - Prevents memory leaks        - Predictable performance
                                                             - Prevents data races
  Cons:                       Cons:                         Cons:
  - Memory leaks              - GC pauses (STW)              - Steep learning curve
  - Use after free            - Higher memory usage           - Compile times
  - Buffer overflows          - Latency variability           - Design constraints
```

### 2.3 Concurrency and Parallelism Models

| Model | Languages | Mechanism | Use Cases |
|--------|------|--------|---------|
| OS Threads | Java, C++, Rust | OS-scheduled | CPU-intensive tasks |
| Green Threads | Go (goroutine) | Runtime-scheduled | Massive I/O waiting |
| async/await | Python, Rust, JS, C# | Event loop + Future/Promise | I/O-bound processing |
| Actor Model | Erlang/Elixir, Akka(Scala) | Message passing | Distributed fault tolerance |
| CSP | Go (channel) | Channel communication | Pipeline processing |
| GIL Constraint | Python (CPython) | Global lock | Multi-threading not recommended |

### 2.4 Execution Models

```
  From Source Code to Execution
  =============================

  [AOT Compilation (Ahead-Of-Time)]
  C, C++, Rust, Go, Swift
  Source --[Compiler]--> Native binary --[OS]--> Execution
  Pros: Fastest execution speed, easy distribution
  Cons: Compilation wait time, platform-specific builds

  [JIT Compilation (Just-In-Time)]
  Java(JVM), C#(CLR), JavaScript(V8)
  Source --[Compiler]--> Intermediate code --[JIT Compiler]--> Execution
  Pros: Runtime optimization, portability
  Cons: Warm-up time, memory consumption

  [Interpreter]
  Python(CPython), Ruby(MRI), PHP
  Source --[Interpreter]--> Sequential execution
  Pros: Immediate execution, interactive development
  Cons: Slow execution speed
  * Many languages internally convert to bytecode before execution

  [Transpilation]
  TypeScript --> JavaScript
  Kotlin/JS  --> JavaScript
  SourceA --[Transpiler]--> SourceB --[Target Runtime]--> Execution
```

---

## Chapter 3: Language Paradigms and Design Philosophy

### 3.1 Major Paradigms

Programming languages each have their own unique "worldview." This philosophical foundation is the paradigm, which fundamentally governs code structure and problem decomposition.

```
  Paradigm Family Tree
  =================

  Programming Paradigms
  |
  +-- Imperative
  |   |   "Give step-by-step instructions"
  |   |
  |   +-- Procedural
  |   |     C, Pascal, Fortran
  |   |     -> Divide processing with functions (procedures)
  |   |
  |   +-- Object-Oriented
  |         Java, C#, Python, Ruby, Kotlin, Swift
  |         -> Encapsulate data and behavior in objects
  |
  +-- Declarative
      |   "Declare what you want"
      |
      +-- Functional
      |     Haskell, Erlang, Clojure, OCaml, F#
      |     -> Pure functions and immutable data, isolation of side effects
      |
      +-- Logic
      |     Prolog
      |     -> Inference from logical rules and facts
      |
      +-- Query
            SQL
            -> Declaratively describe data extraction conditions
```

> **Multi-paradigm languages**: Many modern languages blend multiple paradigms. Python, Scala, Kotlin, Rust, TypeScript, and Swift all combine elements of object-oriented and functional programming.

### 3.2 Design Philosophy Comparison

Each language has a design philosophy. This deeply influences the language's API design, ecosystem, and community culture.

| Language | Design Philosophy | Concrete Impact |
|------|---------|-------------|
| Python | "There should be one obvious way to do it" | Emphasis on code readability, PEP 8 |
| Go | "Less is more" / Simplicity | Late adoption of generics, explicit error handling |
| Rust | "Safety without compromise" | Ownership, borrow checker |
| Ruby | "Developer happiness" | DSL-like code, metaprogramming |
| Java | "Write once, run anywhere" | JVM, enterprise-grade stability |
| Perl | "There's more than one way to do it" | Flexible but hard to read |
| Erlang | "Let it crash" | Fault tolerance, supervisor trees |
| Haskell | "Avoid success at all costs" | Extreme pursuit of type safety |
| JavaScript | "Don't break the web" | Backward compatibility, flexible type coercion |
| C++ | "Zero overhead abstraction" | Feature-rich but complex |

### 3.3 Code Examples: Problem-Solving Differences by Paradigm

"Find the sum of all integers from 1 to 100 that are multiples of 3 and even"

**Procedural (C)**

```c
#include <stdio.h>

int main(void) {
    int sum = 0;
    for (int i = 1; i <= 100; i++) {
        if (i % 3 == 0 && i % 2 == 0) {
            sum += i;
        }
    }
    printf("Sum: %d\n", sum);  // Sum: 918
    return 0;
}
```

**Object-Oriented (Java)**

```java
import java.util.stream.IntStream;

public class Sum {
    public static void main(String[] args) {
        int sum = IntStream.rangeClosed(1, 100)
            .filter(i -> i % 3 == 0 && i % 2 == 0)
            .sum();
        System.out.printf("Sum: %d%n", sum);  // Sum: 918
    }
}
```

**Functional (Haskell)**

```haskell
main :: IO ()
main = print $ sum [x | x <- [1..100], x `mod` 6 == 0]
-- Sum: 918 (multiples of 3 that are also even = multiples of 6)
```

**Declarative (SQL)**

```sql
SELECT SUM(n) AS total
FROM generate_series(1, 100) AS n
WHERE n % 6 = 0;
-- total: 918
```

---

## Chapter 4: Decision Framework for Language Selection

We systematize language selection from a subjective "preference" into a reproducible decision-making process.

### 4.1 Scorecard Method

Assign weights to evaluation criteria based on project requirements and quantitatively compare candidate languages.

```
  Evaluation Criteria and Weighting (Example: Web Backend API)

  +-------------------+------+--------+------+--------+--------+
  | Criterion          | Weight| Python |  Go  |  Rust  |  Java  |
  +-------------------+------+--------+------+--------+--------+
  | Team Proficiency   | 25%  |   9    |  5   |   3    |   7    |
  | Ecosystem          | 20%  |   9    |  7   |   6    |   9    |
  | Runtime Performance| 15%  |   4    |  8   |  10    |   7    |
  | Development Speed  | 15%  |   9    |  7   |   5    |   6    |
  | Maintainability    | 10%  |   6    |  8   |   9    |   8    |
  | Hiring Ease        | 10%  |   9    |  6   |   5    |   8    |
  | Deployment Ease    |  5%  |   6    | 10   |   9    |   5    |
  +-------------------+------+--------+------+--------+--------+
  | Weighted Score     |      |  7.7   | 6.7  |  5.8   |  7.2   |
  +-------------------+------+--------+------+--------+--------+

  Calculation:
    Python = 9*0.25 + 9*0.20 + 4*0.15 + 9*0.15 + 6*0.10 + 9*0.10 + 6*0.05
           = 2.25 + 1.80 + 0.60 + 1.35 + 0.60 + 0.90 + 0.30
           = 7.80 (rounded: 7.8)

  -> In this example, Python is the optimal choice
  * Weights vary depending on project characteristics
  * If performance is the top requirement, Rust's score would overtake
```

#### Scorecard Weighting Patterns

| Project Characteristics | Criteria to Emphasize | Items to Increase Weight |
|---------------|---------------|----------------|
| Startup MVP | Development speed | Development Speed 30%, Ecosystem 25% |
| Enterprise | Maintainability & team | Team Proficiency 30%, Maintainability 20% |
| High Traffic | Performance | Runtime Performance 30%, Deployment Ease 15% |
| R&D | Ecosystem | Ecosystem 30%, Development Speed 25% |
| Long-term Operation (10+ years) | Maintainability & talent | Maintainability 25%, Hiring Ease 20% |

### 4.2 Constraint-Based Decision (Decision Tree)

When there are "hard constraints" that must be satisfied, first narrow down candidates by constraints before applying the scorecard.

```
  Constraint-Based Language Selection Flow
  =========================

  START
    |
    v
  Must run in the browser?
    |
    +-- Yes --> JavaScript / TypeScript / WebAssembly
    |
    +-- No
         |
         v
       iOS app?
         |
         +-- Yes --> Swift (native) or Flutter/RN (cross-platform)
         |
         +-- No
              |
              v
            Latency < 1ms required?
              |
              +-- Yes --> C++ / Rust (no-GC languages)
              |
              +-- No
                   |
                   v
                 Team size > 50 people?
                   |
                   +-- Yes --> Statically typed languages
                   |           (TypeScript, Go, Java, Kotlin)
                   |
                   +-- No
                        |
                        v
                      AI/ML is a core feature?
                        |
                        +-- Yes --> Python + (C++/Rust for hot paths)
                        |
                        +-- No
                             |
                             v
                           Prototype needed in 1 week?
                             |
                             +-- Yes --> Python / Ruby / JavaScript
                             |
                             +-- No
                                  |
                                  v
                                Need 10+ years of maintenance?
                                  |
                                  +-- Yes --> Java / C# / Go / TS
                                  |
                                  +-- No --> Evaluate with scorecard
```

### 4.3 Decision Matrix: Specific Scenarios

Below are common project scenarios and their recommended language pairings.

| Scenario | Recommended Language | Reason |
|---------|---------|------|
| SaaS MVP (3 months) | TypeScript (Next.js) | Fastest launch with full-stack unification |
| Internal Management System | Java/Kotlin (Spring) | Long-term maintenance, robust auth infrastructure |
| Real-time Chat | Go / Elixir | Concurrent connections, WebSocket processing |
| ML Inference API | Python (FastAPI) + Rust | Python for model management, Rust for inference speed |
| Video Transcoding Service | Rust / C++ | CPU-intensive, memory efficiency critical |
| E-commerce Site | PHP (Laravel) / Ruby (Rails) | Existing assets, development speed |
| Blockchain DApp | Rust (Solana) / Solidity (EVM) | Platform-dependent |
| IoT Device | C / Rust | Memory constraints, real-time requirements |
| Data Pipeline | Python + SQL | pandas/Spark ecosystem |
| AAA Game | C++ (Unreal) | Performance, industry standard |

### 4.4 ADR (Architecture Decision Record) Template

It is recommended to record language selection results as team consensus in an ADR. Below is a practical template.

```markdown
# ADR-001: Backend Language Selection

## Status
Approved (2025-01-15)

## Context
- Developing a backend API for a new SaaS product
- Team composition: 5 backend engineers (4 with Python experience, 1 with Go experience)
- Requirements: REST API + WebSocket, peak 10,000 req/s
- Maintenance period: 5+ years

## Options Considered
1. Python (FastAPI)
2. Go (Standard library + Echo)
3. TypeScript (NestJS)

## Decision
We select Python (FastAPI).

## Rationale
- 80% of the team is proficient in Python
- FastAPI with type hints + async delivers sufficient performance
- Easy future integration of ML features
- 10,000 req/s is achievable with FastAPI + uvicorn

## Accepted Trade-offs
- Slower than Go/Rust for CPU-intensive processing
- GIL constraints limit multi-threaded processing
  -> CPU-intensive parts will be extracted to Go/Rust microservices in the future

## Rejection Reasons
- Go: Low team proficiency, estimated 2+ months ramp-up time
- TypeScript: Backend type safety is sufficient with Python's type hints

## Related Information
- Scorecard evaluation results: Python 7.8 / Go 6.7 / TypeScript 7.0
```

---

## Chapter 5: Team and Organizational Perspective

The technically optimal language may not be organizationally optimal. Team composition, skill sets, hiring market, and organizational culture significantly influence language selection.

### 5.1 Team Proficiency and Productivity Relationship

```
  Productivity
  ^
  |                                          xxxxxxx  Expert
  |                                     xxxxx
  |                                xxxxx
  |                           xxxx
  |                       xxxx
  |                   xxx
  |              xxxx
  |         xxxxx
  |     xxxx
  | xxxx
  +--+------+------+------+------+------+------> Months of experience
     0      3      6      12     18     24

  Time to Productivity by Language (approximate)
  +------------------+------------------+
  | Language          | Time to Productive|
  +------------------+------------------+
  | Python           | 1-2 months       |
  | JavaScript/TS    | 2-3 months       |
  | Go               | 2-3 months       |
  | Java/Kotlin      | 3-4 months       |
  | C#               | 3-4 months       |
  | Swift            | 3-4 months       |
  | C++              | 6-12 months      |
  | Rust             | 6-12 months      |
  | Haskell          | 6-12 months      |
  +------------------+------------------+
  * Assumes developers with experience in other languages
```

### 5.2 Hiring Market Reality

When selecting a language, whether you can continuously hire developers proficient in that language is an extremely important factor.

| Language | Job Postings (relative) | Candidate Pool | Salary Level (relative) | Hiring Difficulty |
|------|-------------|-------------|----------------|----------|
| JavaScript/TS | Very high | Very large | Medium | Low |
| Python | Very high | Very large | Medium-High | Low |
| Java | High | Large | Medium | Low-Medium |
| Go | Growing | Medium | High | Medium |
| Rust | Low but rapidly growing | Small | High | High |
| Kotlin | Growing | Medium | Medium-High | Medium |
| Swift | Medium | Medium | High | Medium |
| Ruby | Declining | Medium | Medium | Medium |
| Elixir | Low | Small | High | High |
| Haskell | Low | Very small | High | Very high |

> **Key Insight**: Languages like Rust and Haskell are "hard to hire for," but applicants tend to have higher average skill levels. Niche languages function as a "filter effect," attracting talented developers with specific inclinations.

### 5.3 Organization Size and Language Selection

```
  Language Selection Guidelines by Organization Size
  ========================

  [Individual / 1-3 Person Team]
  -> Choose the language you are most proficient in
  -> Productivity is the top priority; verify ecosystem maturity
  -> Recommended: Python, TypeScript, Ruby

  [Small Team (5-15 people)]
  -> Choose a language everyone can read and write
  -> Code review quality depends on language comprehension
  -> Recommended: TypeScript, Go, Python, Kotlin

  [Medium Team (15-50 people)]
  -> Type system importance increases (implicit knowledge sharing becomes difficult)
  -> Leverage compiler-based bug detection
  -> Recommended: TypeScript, Go, Java/Kotlin, C#

  [Large Team (50-200 people)]
  -> Static typing becomes a de facto requirement
  -> Strict unification of linters, formatters, and CI is necessary
  -> Recommended: Java/Kotlin, Go, TypeScript, C#

  [Very Large (200+ people / Google-scale)]
  -> Often have dedicated language infrastructure teams
  -> Internal toolchain optimization is important
  -> Examples: Google (Go, Java, Python, C++),
               Meta (Hack, Python, C++, Rust),
               Apple (Swift, Objective-C, C++)
```

### 5.4 Code Example: Type Safety Differences by Team Size

In small teams, the flexibility of dynamic typing thrives, but in large teams, the "documentation effect" of types becomes indispensable.

**Dynamic Typing (Python) -- For Small Teams**

```python
# In small teams, implicit knowledge can compensate
def calculate_discount(order, customer):
    """Calculate the discount amount for an order"""
    if customer["tier"] == "premium":
        return order["total"] * 0.15
    elif customer["tier"] == "standard":
        return order["total"] * 0.05
    return 0

# What keys the dictionaries have depends on "shared team understanding"
# -> In a 5-person team, this can be communicated verbally
# -> In a 50-person team, "what values does 'tier' take?" becomes frequent
```

**Static Typing (TypeScript) -- For Large Teams**

```typescript
// Types serve as documentation
type CustomerTier = "premium" | "standard" | "basic";

interface Customer {
  id: string;
  name: string;
  tier: CustomerTier;
  registeredAt: Date;
}

interface Order {
  id: string;
  customerId: string;
  items: OrderItem[];
  total: number;
  currency: "JPY" | "USD" | "EUR";
}

interface OrderItem {
  productId: string;
  quantity: number;
  unitPrice: number;
}

// Input and output are clear from the type signature alone
// Even a 50-person team can read unfamiliar code
function calculateDiscount(order: Order, customer: Customer): number {
  switch (customer.tier) {
    case "premium":
      return order.total * 0.15;
    case "standard":
      return order.total * 0.05;
    case "basic":
      return 0;
    // TypeScript's exhaustiveness check causes
    // a compile error here if a new tier is added
  }
}
```

### 5.5 Onboarding Cost Comparison

The time from when a new member joins the team to when they can independently implement features is directly tied to language selection.

```
  Factors Affecting Onboarding
  =============================

  +-- Language Complexity
  |     C++ > Rust > Scala > Java > Kotlin > Go > Python > JS
  |     (Complex)                                    (Simple)
  |
  +-- Development Environment Setup
  |     Java(Maven/Gradle) > Rust(cargo) > Python(venv) > Go(mod)
  |     (Complex)                                        (Simple)
  |
  +-- Volume of Learning Resources
  |     JS/Python > Java > Go > Rust > Elixir > Zig
  |     (Abundant)                           (Scarce)
  |
  +-- Error Message Friendliness
  |     Rust/Elm > Go > Kotlin > Java > C++ > Haskell
  |     (Friendly)                         (Cryptic)
  |
  +-- Codebase Conventions
        (Independent of language, but influenced by language philosophy)
```

---

## Chapter 6: 2025 Language Trends and Future Outlook

### 6.1 Growing Languages

```
  Growing Languages and Their Background
  =====================

  Rust
  ├── Increasing demand for memory safety
  ├── Official adoption in the Linux kernel
  ├── Expanding adoption at Android, AWS, Microsoft
  ├── Primary language for WebAssembly
  └── crates.io ecosystem maturing rapidly

  TypeScript
  ├── De facto successor to JavaScript
  ├── Standard for large-scale projects
  ├── First-class support in Node.js / Deno / Bun
  └── Frontend + Backend + Infrastructure (Pulumi)

  Go
  ├── Cloud-native standard (Docker, K8s, Terraform)
  ├── Simplicity valued in large teams
  ├── Improved expressiveness with generics (1.18+)
  └── Fast startup and single binary valued in DevOps

  Kotlin
  ├── Official Android language
  ├── Growing server-side adoption (Ktor, Spring Boot)
  ├── Maturing Kotlin Multiplatform (KMP)
  └── Excellent concurrency model via coroutines

  Zig
  ├── Modern alternative candidate to C
  ├── Embedded and systems programming
  ├── Adopted as compiler infrastructure (Bun is built with Zig)
  └── Complete interoperability with C
```

### 6.2 Stable Languages

| Language | Reason for Stability | Future Outlook |
|------|----------------|-----------|
| Python | AI/ML dominance continues for now. Versatile in web/scripting | GIL removal (PEP 703), type hint enhancements |
| Java | Enterprise foundation. Active evolution since 21+ | Virtual Threads (Loom), Value Types (Valhalla) |
| C# | Reevaluated with .NET evolution. Strong in games (Unity) | Native AOT, Blazor WASM maturation |
| Swift | The sole option for Apple ecosystem | Growth of Server-Side Swift, Swift 6 concurrency safety |
| C/C++ | Immortal due to legacy assets and performance requirements | C++23/26 modernization, Carbon (potential successor) |

### 6.3 Notable Emerging Languages

| Language | Overview | Reason for Attention | Maturity |
|------|------|---------|-------|
| Mojo | Python syntax + C performance | High-speed execution for AI/ML | Early stage |
| Gleam | Typed functional on Erlang VM | Alternative candidate to Elixir | Growing |
| Roc | Elm-inspired functional | Pure functional for web development | Experimental |
| Vale | Region-based memory management | New approach as Rust alternative | Experimental |
| Carbon | Google-led C++ successor | Aims for gradual migration from C++ | Early stage |
| Unison | Content-addressed approach | Distributed computing | Early stage |

### 6.4 Meta-Trends: Directions in Language Design

```
  Language Design Trends of the 2020s
  =========================

  1. Emphasis on Memory Safety
     The cost of C/C++ memory vulnerabilities can no longer be ignored
     -> Rust, Zig, Carbon, Vale propose alternatives
     -> US government (CISA) recommends memory-safe languages

  2. Gradual Typing
     The approach of adding types to dynamic languages later has become mainstream
     -> TypeScript, Python type hints, PHP type declarations, Ruby RBS/Sorbet

  3. Changes in Coding through AI Assistance
     Proliferation of GitHub Copilot, Cursor, Claude Code, etc.
     -> Productivity decline in verbose languages is mitigated
     -> Type information improves AI completion accuracy

  4. Expansion of WebAssembly (Wasm)
     Wasm execution outside browsers (WASI) becomes practical
     -> Language-agnostic portable execution environment
     -> Rust, Go, C/C++ are leading Wasm targets

  5. Multi-Platform Support
     Targeting multiple platforms from one language/codebase
     -> Kotlin Multiplatform, Flutter, .NET MAUI
```

---

## Chapter 7: Polyglot Strategy

### 7.1 Why Use Multiple Languages

In reality, it is rare for a single language to optimally build an entire software system. Each language has its strengths, and by strategically combining multiple languages, you can make optimal choices at each layer.

```
  Polyglot Architecture Example
  =============================

  +--[ Browser ]---------------------------------------+
  |  TypeScript (React / Next.js)                       |
  |  -> UI rendering, interactions                      |
  +----------------------------------------------------+
           | HTTP / GraphQL
           v
  +--[ API Gateway ]-------------------------------+
  |  Go (Kong / custom implementation)              |
  |  -> Routing, authentication, rate limiting       |
  +----------------------------------------------------+
           |
     +-----+-----+-----+
     |           |           |
     v           v           v
  +--------+ +--------+ +--------+
  | Python | | Go     | | Rust   |
  | ML     | | CRUD   | | Video  |
  | Inference| API    | | Trans- |
  | Service| |        | | coding |
  +--------+ +--------+ +--------+
     |           |           |
     v           v           v
  +----------------------------------------------------+
  |  SQL (PostgreSQL) + Python (Data Pipeline)          |
  |  -> Data layer                                      |
  +----------------------------------------------------+
           |
           v
  +----------------------------------------------------+
  |  Bash / Python (CI/CD scripts)                      |
  |  Go (Terraform) / YAML (Kubernetes)                 |
  |  -> Infrastructure layer                            |
  +----------------------------------------------------+
```

### 7.2 Polyglot Strategy Patterns

| Pattern | Composition Example | Benefits | Risks |
|---------|--------|---------|--------|
| Full-Stack Unification | TypeScript only | No context switching | Backend performance limits |
| Frontend/Backend Separation | TS + Go | Optimization at each layer | Need proficiency in 2 languages |
| Backend + ML Separation | Go + Python | Leverage ML ecosystem | Inter-service communication overhead |
| Core + Scripting | Rust + Python | Both performance and development speed | FFI complexity |
| Legacy + New | Java + Kotlin | Gradual modernization | Complexity during coexistence period |

### 7.3 Polyglot Limits: How Many Languages Are Realistic

```
  Relationship Between Number of Languages and Organizational Cost
  ========================

  Organizational Cost
  ^
  |                                        x
  |                                    x
  |                                x
  |                           x
  |                       x
  |                  x
  |             x
  |         x
  |     x
  | x
  +--+------+------+------+------+------> Number of languages used
     1      2      3      4      5+

  Recommended:
  - Startups: 1-2 languages
  - Mid-size companies: 2-3 languages
  - Large enterprises: 3-5 languages (1-2 per team)
  - FAANG-scale: 5+ languages (dedicated teams support each language)

  Cost Factors:
  - Toolchain maintenance (CI/CD, linters, formatters)
  - Library vulnerability management
  - Onboarding
  - Code review quality
  - Shared library management
```

### 7.4 Technical Methods for Inter-Language Integration

| Integration Method | Description | Latency | Complexity | Use Case |
|---------|------|------|--------|--------|
| HTTP/gRPC | Microservice inter-communication | Medium-High | Low | Go <-> Python |
| FFI (Foreign Function Interface) | Native function calls | Very Low | High | Python -> C/Rust |
| WebAssembly | Wasm module invocation | Low | Medium | JS -> Rust (Wasm) |
| Message Queue | Asynchronous messaging | High | Medium | Between any languages |
| Shared Database | Data sharing via DB | Medium | Low | Between any languages |
| CLI Invocation | Subprocess spawning | High | Low | Python -> Go CLI |

#### Code Example: Calling a Rust Function from Python via FFI

**Rust Side (Library)**

```rust
// lib.rs -- Rust provides high-speed Fibonacci computation
// Compile: cargo build --release (generates a shared library)

#[no_mangle]
pub extern "C" fn fibonacci(n: u64) -> u64 {
    if n <= 1 {
        return n;
    }
    let mut a: u64 = 0;
    let mut b: u64 = 1;
    for _ in 2..=n {
        let temp = a + b;
        a = b;
        b = temp;
    }
    b
}
```

**Python Side (Caller)**

```python
# main.py -- Calling a Rust function from Python via ctypes
import ctypes
import time

# Load the Rust shared library
lib = ctypes.CDLL("./target/release/libfibonacci.so")  # Linux
# lib = ctypes.CDLL("./target/release/libfibonacci.dylib")  # macOS

# Define the function signature
lib.fibonacci.argtypes = [ctypes.c_uint64]
lib.fibonacci.restype = ctypes.c_uint64

# Compare with a pure Python implementation
def fibonacci_py(n: int) -> int:
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

n = 40

# Rust version
start = time.perf_counter()
result_rust = lib.fibonacci(n)
time_rust = time.perf_counter() - start

# Python version
start = time.perf_counter()
result_py = fibonacci_py(n)
time_py = time.perf_counter() - start

print(f"Rust:   fib({n}) = {result_rust} ({time_rust:.6f}s)")
print(f"Python: fib({n}) = {result_py}   ({time_py:.6f}s)")
print(f"Rust is approximately {time_py / time_rust:.0f}x faster than Python")
```

---

## Chapter 8: Migration Strategy -- When to Switch Languages

### 8.1 Signals That Warrant Migration

Switching languages is a costly decision and should not be executed without clear signals. Consider migration when multiple of the following signals apply.

```
  Migration Consideration Signals (review if 3 or more apply)
  ========================================

  [ ] Performance bottleneck is language-caused with no remaining optimization room
  [ ] Ecosystem maintenance has stagnated
  [ ] Security patch delivery is delayed or stopped
  [ ] Developer hiring has become significantly difficult
  [ ] Team motivation is declining
  [ ] Cannot meet new requirements (real-time, AI integration, etc.)
  [ ] Runtime support deadline is approaching
  [ ] Technical debt has accumulated and refactoring cost is high
  [ ] Dependency library compatibility issues are frequent
```

### 8.2 Migration Strategy Comparison

| Strategy | Overview | Risk | Duration | Application |
|------|------|--------|------|---------|
| Big Bang Rewrite | Rewrite everything in the new language at once | Very high | Long | Small-scale systems only |
| Strangler Fig Pattern | Gradually implement new features in the new language | Low-Medium | Long | Recommended for large-scale systems |
| Sidecar Pattern | Extract specific features into separate language services | Low | Medium | Specific features with performance requirements |
| Binding Approach | Embed new language functions via FFI/WASM | Low | Short | Hot path optimization |
| Parallel Operation | Run old and new systems in parallel, gradually switch over | Medium | Long | Mission-critical systems |

### 8.3 Strangler Fig Pattern in Practice

```
  Gradual Migration with the Strangler Fig Pattern
  ======================================

  Phase 1: New features implemented in new language
  +------------------------------------------+
  |  [Reverse Proxy / API Gateway]            |
  |       |                    |               |
  |       v                    v               |
  |  +-----------+      +-----------+          |
  |  | Old System|      | New Service|          |
  |  | (Python)  |      | (Go)      |          |
  |  | Feature   |      | Feature D |          |
  |  | A, B, C   |      |           |          |
  |  +-----------+      +-----------+          |
  +------------------------------------------+

  Phase 2: Gradually migrate existing features
  +------------------------------------------+
  |  [Reverse Proxy / API Gateway]            |
  |       |                    |               |
  |       v                    v               |
  |  +-----------+      +-----------+          |
  |  | Old System|      | New Service|          |
  |  | (Python)  |      | (Go)      |          |
  |  | Feature A |      | Feature   |          |
  |  |           |      | B, C, D   |          |
  |  +-----------+      +-----------+          |
  +------------------------------------------+

  Phase 3: Completely retire the old system
  +------------------------------------------+
  |  [Reverse Proxy / API Gateway]            |
  |                      |                     |
  |                      v                     |
  |               +-----------+                |
  |               | New Service|                |
  |               | (Go)      |                |
  |               | Feature   |                |
  |               | A, B, C, D|                |
  |               +-----------+                |
  +------------------------------------------+
```

### 8.4 Migration Risk Management Checklist

Verify the following items before starting a migration project.

```
  Pre-Migration Checklist
  ====================

  Planning Phase:
  [ ] Defined migration purpose and success criteria
  [ ] Estimated migration effort (secure 2-3x normal buffer)
  [ ] Developed rollback plan
  [ ] Decided feature development freeze/parallel policy during migration
  [ ] Completed PoC (Proof of Concept) in the new language

  Technical Phase:
  [ ] Test coverage is sufficient (reinforce before migration)
  [ ] API compatibility tests are automated
  [ ] Data migration scripts are verified
  [ ] Monitoring and alerts are set up for both old and new systems
  [ ] Performance benchmarks are defined

  Team Phase:
  [ ] Conducted new language training for team members
  [ ] Established coding conventions and review criteria
  [ ] Planned pair programming/mob programming sessions
  [ ] Secured external expert support (if needed)
```

### 8.5 Lessons from Failed Migration Cases

| Case Pattern | Cause of Failure | Lesson |
|-------------|-----------|------|
| Prolonged big bang rewrite | Market changed during development, requirements drifted | Should have chosen gradual migration |
| Overestimation of new language | Judged by benchmarks alone, underestimated ecosystem gaps | Conduct realistic validation with PoC |
| Team resistance | Migration reasons not sufficiently shared, motivation dropped | Involve the entire team in decision-making |
| Loss of old system knowledge | Members who understood the old system left | Maintain documentation in parallel with migration |
| Underestimation of timeline | Migration includes not just "rewriting" but surrounding tools | Secure 2-3x buffer on estimates |

---

## Chapter 9: Anti-Pattern Collection

We organize typical anti-patterns in language selection. These are repeatedly observed failure patterns that can be avoided by recognizing them in advance.

### Anti-Pattern 1: Resume-Driven Development

```
  Anti-Pattern: Resume-Driven Development
  ==============================

  Symptoms:
    Team leaders or decision-makers choose languages that benefit
    their own careers (latest, trending languages).

  Examples:
    "Let's use Rust because it looks good on my resume"
    "Let's go with Kubernetes and microservices since it helps with job changes"

  Actual Harm:
    - Majority of team is inexperienced, productivity drops significantly
    - Learning costs strain project schedules
    - Insufficient problem-solving capability leads to quality decline

  Countermeasures:
    - Make language selection a team-wide consensus decision
    - Conduct quantitative evaluation with scorecards, eliminating personal preferences
    - Collect objective data during a PoC phase

  Decision Criterion:
    "Can you explain the reason for choosing this language
     solely based on project requirements?"
    -> If not, it is resume-driven.
```

### Anti-Pattern 2: Golden Hammer

```
  Anti-Pattern: Golden Hammer
  ==================================

  Symptoms:
    "To someone with a hammer, everything looks like a nail"
    Because of deep proficiency in a specific language, they try to
    solve every problem with that language.

  Examples:
    - Web developers trying to build everything with Python
      -> Struggle with real-time processing, introduce complex
        Celery + Redis queuing
      -> Could have been solved at the language level with Go or Elixir

    - Java engineers writing CLI tools in Java
      -> JVM startup takes over 1 second, poor user experience
      -> Go or Rust would start instantly

    - C++ engineers writing an admin dashboard in C++
      -> Development speed is slow, security risks are high
      -> Python + Django would complete in 1/10 the time

  Countermeasures:
    - Ask "What is the optimal language for this problem?"
      before choosing a language
    - Place multi-language experienced members on the team
    - Regularly catch up on trends in other languages
```

### Anti-Pattern 3: Benchmark Worship

```
  Anti-Pattern: Benchmark Worship
  ================================

  Symptoms:
    Selecting a language based solely on micro-benchmark results.
    "Go is 50x faster than Python in TechEmpower benchmarks,
     so we should use Go"

  Reality:
    +------------------------------------------------------+
    | Typical response time breakdown for web applications  |
    |                                                      |
    | DB queries:        [====================] 60%         |
    | Network I/O:       [==========] 25%                   |
    | Business logic:    [===] 10%                          |
    | Language overhead:  [=] 5%                            |
    +------------------------------------------------------+

    -> Switching to Go only improves 5% of the total
    -> DB index optimization is 10x more effective

  Countermeasures:
    - Measure the bottleneck before deciding
    - Profile the entire application
    - If "sufficient speed" is enough, prioritize development speed
```

### Anti-Pattern 4: Trend Chasing

```
  Anti-Pattern: Trend Chasing
  ========================

  Symptoms:
    Jumping on languages trending on Hacker News or Twitter.
    "It's #1 in this year's new language ranking, so let's adopt it"

  Risks:
    - Immature ecosystem with insufficient libraries
    - Small community with limited support
    - Unstable language spec with frequent breaking changes
    - No developers for that language in the hiring market

  Historical Lessons:
    - CoffeeScript: Highly popular in 2012 -> Completely replaced by TypeScript
    - Dart (v1): High expectations in 2013 -> Failed browser adoption -> Revived with Flutter
    - Elm: Frontend innovation -> Adoption stalled due to developer shortage

  Countermeasures:
    - Try new languages in "side projects"
    - Only introduce to production after "2+ years of stable growth"
    - "Lindy Effect": The longer a language has survived, the longer it will continue to survive
```

### Anti-Pattern 5: Illusion of Unanimity

```
  Anti-Pattern: Illusion of Unanimity
  ==============================

  Symptoms:
    Endlessly searching for a language everyone agrees on, never making a decision.
    "Let's keep discussing until everyone agrees"

  Reality:
    - No perfect language exists
    - No language satisfies everyone's preferences
    - The longer the debate drags on, the more the cost of delayed development increases

  Countermeasures:
    - Set a timebox (e.g., decide within 1 week)
    - Quantify with scorecards, eliminating emotional arguments
    - Respect the "freedom to disagree" while agreeing on the "obligation to follow the decision"
    - Agree on the decision process in advance (majority vote, leader's call, etc.)
```

---

## Chapter 10: Practical Exercises

### Exercise 1: Basic -- Creating a Language Characteristics Comparison Table

**Task**: Create a comparison table of Python, Go, and Rust across the following 6 axes.

| Comparison Axis | Python | Go | Rust |
|--------|--------|-----|------|
| Typing | ? | ? | ? |
| Memory Management | ? | ? | ? |
| Concurrency Model | ? | ? | ? |
| Ecosystem Maturity | ? | ? | ? |
| Compilation/Execution Speed | ? | ? | ? |
| Learning Curve | ? | ? | ? |

**Steps**:

1. Refer to each language's official documentation and describe characteristics for each axis in 1-2 sentences
2. Assign a score of 1-10 for each axis (also specify the criteria for what constitutes "good")
3. Select the optimal language for a "Web API backend" among the 3, and state your reasoning

**Model Answer Direction**:

```
  Typing:
    Python: Dynamic typing + type hints (gradual typing)
    Go:     Static typing, type inference (`:=`), nominal types
    Rust:   Static typing, powerful type inference, algebraic data types

  Memory Management:
    Python: Reference counting + GC (CPython)
    Go:     Tracing GC (minimized STW)
    Rust:   Ownership system (no GC)

  Concurrency:
    Python: asyncio / threading (GIL constraint)
    Go:     goroutine + channel (CSP model)
    Rust:   async/await + OS threads (fearless concurrency)
```

### Exercise 2: Applied -- Language Selection Report

**Task**: Create a language selection report for the following project requirements.

```
  Project: Real-Time Stock Price Display Dashboard
  ==================================================

  Functional Requirements:
    - Receive stock price data from exchange via WebSocket
    - Process up to 10,000 price updates per second
    - Display real-time charts in the browser
    - Calculate moving averages from past 30 days of data
    - Calculate portfolio profit/loss for each user

  Non-Functional Requirements:
    - Latency: Under 100ms from data reception to browser display
    - Availability: 99.9% (during market hours)
    - Concurrent users: Up to 5,000
    - Data retention period: 1 year

  Team Composition:
    - 3 backend engineers (Python experienced)
    - 2 frontend engineers (React experienced)
    - Deadline: 4 months
```

**Report Structure**:

1. Frontend language selection and rationale
2. Backend language selection and rationale
3. Data processing layer language selection and rationale
4. Quantitative evaluation with scorecard
5. Risks and countermeasures
6. ADR (Architecture Decision Record)

**Hints**:
- Frontend is nearly certain to be TypeScript (React)
- Backend key point is massive concurrent WebSocket connections
- The trade-off of "team proficiency vs. performance requirements" when the Python team handles the backend is the key discussion point

### Exercise 3: Advanced -- 30-Minute New Language Evaluation

**Task**: Select one language you have never used from the following list, and implement the following 3 tasks in 30 minutes.

**Candidate Languages**: Elixir, Zig, Gleam, Kotlin, Dart, OCaml, F#, Julia

**Tasks**:

```
  Step 1 (5 min): Hello World
    - Set up the development environment
    - Output "Hello, World!"

  Step 2 (10 min): FizzBuzz
    - Implement FizzBuzz from 1 to 100
    - Utilize the language's control structures and pattern matching

  Step 3 (15 min): Simple HTTP Server
    - Return {"message": "Hello!"} for GET /hello
    - Generate a JSON response
    - Listen on port 8080
```

**Evaluation Criteria** (reflect after 30 minutes):

| Evaluation Item | Record |
|---------|------|
| Was the development environment easy to set up? | |
| How many minutes to Hello World? | |
| How many minutes to FizzBuzz? | |
| Did the HTTP server work? | |
| Were error messages understandable? | |
| How was the documentation/tutorial quality? | |
| How rich was the standard library? | |
| Would you want to use it again? | |

**Purpose of this exercise**: Develop the skill to systematically evaluate a new language's "first impression." In language selection, the value of "actually trying it out" is immeasurable.

---

## FAQ (Frequently Asked Questions)

### Q1: What are the benefits of unifying the full stack in one language?

**A**: The main benefits are reduced context switching, code sharing between frontend and backend (validation logic, etc.), and easier code reviews across the entire team. TypeScript (Node.js + React) is the representative example, where sharing type definitions ensures API type safety from end to end.

However, full-stack unification should be avoided in the following cases:
- When the backend has heavy CPU-intensive processing (consider Go/Rust)
- When ML/AI functionality is at the core (Python is essential)
- When GC pauses are unacceptable under ultra-high traffic (consider Rust)

### Q2: When should you switch languages?

**A**: Consider when multiple of the following conditions apply:

1. Facing technical constraints that are difficult to resolve with the existing language (performance, concurrency, etc.)
2. Team productivity is clearly declining
3. Ecosystem support has ended or stagnated
4. Developer hiring has become significantly difficult
5. Security concerns are language-caused and hard to resolve

When switching, gradual migration using the Strangler Fig pattern is recommended. Implement new features in a different language first, and migrate existing features gradually. Big bang rewrites should be avoided for anything beyond small-scale systems.

### Q3: What is the lifespan of a programming language?

**A**: As COBOL (1959~), Fortran (1957~), and LISP (1958~) still operate today, the lifespan of a language itself is very long. The important points are the following 3:

1. **Ecosystem vitality**: Are libraries and frameworks being actively developed?
2. **Talent availability**: Can you hire developers who know the language?
3. **Runtime support**: Is the runtime/implementation continuously maintained?

Based on the Lindy Effect (a tendency to survive for a period proportional to how long it has existed), languages with 20+ years of track record (C, C++, Java, Python, JavaScript, etc.) are likely to continue being used for a long time.

### Q4: Will AI (LLM) proliferation change language selection?

**A**: With the proliferation of AI coding assistance tools, the following changes are observed:

1. **Reduced handicap for verbose languages**: AI auto-generates Java boilerplate, narrowing the development speed gap
2. **Type information improves AI accuracy**: Static typed languages like TypeScript and Rust have higher AI completion accuracy
3. **Documentation quality becomes important**: Languages with abundant documentation and code examples that AI can learn from have an advantage
4. **Niche languages are disadvantaged**: Languages with limited training data have lower AI assistance quality

However, the essence of language selection (domain fit, team proficiency, ecosystem) does not change. AI is a tool that "narrows the productivity gap," not something that overturns fundamental suitability.

### Q5: Are "languages to learn" and "languages to use at work" different?

**A**: They often differ. For learning purposes, you should choose languages that broaden your programming perspective.

| Learning Purpose | Recommended Language | What You Learn |
|---------|---------|-----------|
| Functional Programming | Haskell / OCaml | Purity, type inference, immutability |
| Understanding Memory Models | C / Rust | Manual management vs. ownership |
| Concurrency Design | Go / Erlang | CSP / Actor model |
| Metaprogramming | Ruby / Lisp | DSLs, macros, reflection |
| Low-Level Optimization | C / Assembly | Relationship with hardware |

Use "the optimal language for the project" at work, and choose "languages that broaden your horizons" for learning. This dual strategy enhances your overall engineering capability.

### Q6: Is it acceptable to use different languages for each microservice?

**A**: It is technically possible, but organizational costs must be carefully evaluated.

**Acceptable when**:
- Each team can independently choose languages, and cross-team personnel rotation is infrequent
- There are clear technical reasons (ML service in Python, real-time processing in Go, etc.)
- Common infrastructure (CI/CD, monitoring, deployment) is built language-agnostically

**Should be avoided when**:
- The team is small and everyone needs to touch all services
- The DevOps team lacks the capacity to maintain each language's toolchain
- "We want to use it" is the only reason

The recommendation is to "limit to 2-3 languages and establish clear usage rules."

---

## Summary

### 7 Principles of Language Selection

```
  +================================================================+
  |                  7 Principles of Language Selection               |
  +================================================================+
  |                                                                |
  |  1. Follow the domain's standard                                |
  |     -> Unless there is a compelling reason, choose the standard  |
  |        for that domain                                          |
  |                                                                |
  |  2. Leverage team strengths                                     |
  |     -> Proficiency has the greatest impact on productivity       |
  |                                                                |
  |  3. Work backward from constraints                              |
  |     -> Narrow candidates with hard constraints, then evaluate    |
  |        with scorecard                                           |
  |                                                                |
  |  4. Measure before deciding                                     |
  |     -> Verify performance requirements through measurement,      |
  |        not speculation                                          |
  |                                                                |
  |  5. Verify the ecosystem                                        |
  |     -> Confirm availability of required libraries and tools      |
  |        in advance                                               |
  |                                                                |
  |  6. Envision 10 years ahead                                     |
  |     -> Will talent be available? Will support continue?          |
  |                                                                |
  |  7. Record the decision                                         |
  |     -> Document rationale in an ADR, preparing for future review |
  |                                                                |
  +================================================================+
```

### Decision Axis Summary

| Decision Axis | Key Point |
|-------|-------------|
| Domain | Each domain has its standards. Do not go against them |
| Team | Proficiency directly impacts productivity |
| Performance | Measure to confirm it is truly the bottleneck before deciding |
| Ecosystem | Verify availability of required libraries in advance |
| Maintainability | Will talent be available in 10 years? |
| Trends | Judge by substance, not by trends |
| Cost | Evaluate by total cost of hiring, training, and toolchain |

### Language Selection Cheat Sheet

```
  Language Selection Quick Reference
  ===========================

  "What you build" determines 80%:
  +-----------------------+---------------------------+
  | What You Build         | First Choice               |
  +-----------------------+---------------------------+
  | Web Frontend           | TypeScript                |
  | Web Backend (small)    | TypeScript / Python       |
  | Web Backend (large)    | Go / Java / Kotlin        |
  | iOS App                | Swift                     |
  | Android App            | Kotlin                    |
  | Cross-Platform         | Flutter (Dart)            |
  | AI/ML                 | Python                    |
  | Systems Programming    | Rust / C                  |
  | CLI Tool               | Go / Rust                 |
  | Data Analysis          | Python / R                |
  | DevOps                | Python / Go / Bash        |
  | Games                  | C++ / C#                  |
  +-----------------------+---------------------------+

  "Who builds it" changes 15%:
  -> Team proficiency, hiring availability, onboarding cost

  "Where it runs" changes 5%:
  -> Browser, mobile, server, embedded, edge
```

---

## Recommended Next Reads


---

## References

1. Scott, M. L. *Programming Language Pragmatics*. 4th Edition, Morgan Kaufmann, 2015. -- A comprehensive textbook on language design principles. Covers the theoretical foundations of type systems, memory management, and control structures.
2. Van Roy, P. and Haridi, S. *Concepts, Techniques, and Models of Computer Programming*. MIT Press, 2004. -- A seminal work that treats programming paradigms in a unified manner. Enables systematic understanding of declarative, imperative, and concurrent programming concepts.
3. Klabnik, S. and Nichols, C. *The Rust Programming Language*. No Starch Press, 2023. -- The official guide for learning modern approaches to memory safety and type safety through Rust's ownership system.
4. Donovan, A. A. and Kernighan, B. W. *The Go Programming Language*. Addison-Wesley Professional, 2015. -- A language introduction that embodies Go's design philosophy "Less is More." Enables understanding the balance between simplicity and practicality.
5. "Stack Overflow Developer Survey 2024." stackoverflow.com. -- The world's largest developer survey. Provides statistical data on language popularity, satisfaction, and salary levels.
6. "The State of Developer Ecosystem 2024." JetBrains. -- Annual developer ecosystem report by JetBrains. Includes statistics on language trends, tool usage, and team composition.
7. Pierce, B. C. *Types and Programming Languages*. MIT Press, 2002. -- The standard textbook on type theory. Enables rigorous study of the mathematical foundations of type inference, polymorphism, and subtyping.

---

## Glossary

| Term | English | Description |
|------|------|------|
| Gradual Typing | Gradual Typing | A mechanism for using dynamic and static typing together within the same program |
| Ownership | Ownership | A memory management model introduced in Rust. Each value has exactly one owner |
| GC | Garbage Collection | A mechanism that automatically reclaims unused memory |
| JIT | Just-In-Time | A method of compiling bytecode to native code at runtime |
| AOT | Ahead-Of-Time | A method of compiling to native code before execution |
| FFI | Foreign Function Interface | An interface for calling functions written in different languages |
| CSP | Communicating Sequential Processes | The theoretical foundation of Go's channel model |
| ADR | Architecture Decision Record | A document format for recording architectural decisions |
| PoC | Proof of Concept | A prototype for verifying technical feasibility |
| GIL | Global Interpreter Lock | A lock mechanism in CPython that restricts concurrent execution of multiple threads |
| Wasm | WebAssembly | A portable binary format executable in browsers and servers |
| STW | Stop-The-World | The phenomenon where the application temporarily pauses during GC execution |
| DSL | Domain-Specific Language | A small language specialized for a specific domain |
| MVP | Minimum Viable Product | A product with the minimum features needed for hypothesis validation |
| Lindy Effect | Lindy Effect | The tendency for things that have existed longer to continue existing |
