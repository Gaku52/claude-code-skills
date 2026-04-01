# Closures

> A closure is "a function that captures the environment at definition time." It is a core technique of functional programming that creates functions with state.

## What You Will Learn in This Chapter

- [ ] Understand the mechanism of closures (lexical scope capture)
- [ ] Grasp the distinction between free variables and bound variables, and how the environment is retained
- [ ] Compare closure implementations across languages (JavaScript, Python, Rust, Go, Java, C++)
- [ ] Master practical closure patterns (memoization, currying, debounce, etc.)
- [ ] Learn to avoid anti-patterns such as memory leaks and loop variable traps
- [ ] Understand the relationship between closures and object-oriented programming, and use each appropriately


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding of [First-Class Functions](./00-first-class-functions.md)

---

## 1. What Is a Closure?

### 1.1 Definition and Intuitive Understanding

A closure is **a combination of a function and the lexical environment in which that function was defined**. It was first proposed by Peter J. Landin in 1964 in the context of the SECD machine, and later put into practical use in Scheme (1975).

An ordinary function accepts only the arguments passed at call time as input. A closure, on the other hand, can access not only its arguments but also "variables that existed in the scope at the time of definition." These "variables from the defining scope" are called **free variables**.

```
Closure = Function body (code) + Environment (set of bindings for free variables)

  Ordinary function:
    Input: Arguments only
    Output: Return value

  Closure:
    Input: Arguments + captured free variables
    Output: Return value
    Side effects: Updates to captured mutable variables (language-dependent)
```

### 1.2 Free Variables and Bound Variables

To understand closures precisely, the distinction between **free variables** and **bound variables** is essential.

```
+----------------------------------------------------------+
|  function multiply(x) {    // x is a bound variable (parameter)  |
|      return x * factor;    // factor is a free variable          |
|  }                                                        |
|                                                           |
|  Bound variable: A variable defined by the function itself |
|                  (parameters, local variables)             |
|  Free variable: A variable used within the function but    |
|                 not defined by the function itself          |
+----------------------------------------------------------+
```

When a function contains free variables that are captured from an outer scope, that function becomes a closure. Conversely, a function with no free variables (where all variables are bound) is simply a "closed function" and is not a closure.

```
  Terminology:

  Open expression: An expression containing free variables
    Example: x * factor   (factor is a free variable)

  Closed expression: An expression containing no free variables
    Example: (x) => x * 2  (all variables are bound)

  Closure: The operation of "closing" an open expression
    = Providing concrete bindings for free variables to create a closed expression
    Example: environment where factor=3 + (x) => x * factor
        -> effectively behaves as (x) => x * 3
```

### 1.3 Lexical Scope and Dynamic Scope

The behavior of closures is closely related to scoping rules. Nearly all modern major languages adopt **lexical scope (static scope)**, and closures are built on this mechanism.

```
+-------------------------------------------------------------------+
|  Lexical Scope vs Dynamic Scope                                     |
+-------------------------------------------------------------------+
|                                                                     |
|  Lexical Scope (Static Scope):                                      |
|    Variable resolution is determined by "source code structure"     |
|    -> Where the function is "defined" matters                      |
|    -> Variable references are resolved at compile time             |
|    -> JavaScript, Python, Rust, Go, Java, C++ ...                  |
|                                                                     |
|  Dynamic Scope:                                                     |
|    Variable resolution is determined by "the runtime call stack"   |
|    -> Where the function is "called" matters                       |
|    -> Variable references are not resolved until runtime           |
|    -> Emacs Lisp, Bash, some older Lisps ...                       |
|                                                                     |
+-------------------------------------------------------------------+
```

Let's look at an example of lexical scope.

```javascript
// Demonstrating lexical scope
const x = "global";

function outer() {
    const x = "outer";

    function inner() {
        // inner is "defined" inside outer
        // -> Due to lexical scope, x is "outer"
        console.log(x);
    }

    return inner;
}

const fn = outer();
fn();  // "outer" (not "global")

// Even though inner is "called" outside of outer,
// it remembers the environment at definition time (x = "outer")
// -> This is the essence of closures
```

If dynamic scoping were used, `fn()` would use the scope chain at the point of invocation, so `x` would be `"global"`. With lexical scope, the scope at definition time is used, so `"outer"` is output.

### 1.4 How Closures Are Created (Memory Model)

Let's trace step by step how closures are realized in memory.

```
Step 1: makeCounter() is called
+------------------------------------------+
|  Call Frame: makeCounter()               |
|  +----------------------------+          |
|  |  count: 0                  |          |
|  |  (local variable)          |          |
|  +----------------------------+          |
|  return { increment, decrement, get }    |
+------------------------------------------+

Step 2: Inner functions are returned as closures
+------------------------------------------+
|  Returned object:                         |
|  {                                        |
|    increment: [Function + Env{count}],   |
|    decrement: [Function + Env{count}],   |
|    getCount:  [Function + Env{count}]    |
|  }                                        |
|                                           |
|  * All 3 functions share the same count  |
+------------------------------------------+

Step 3: makeCounter's frame would normally be destroyed, but...
+------------------------------------------+
|  makeCounter's stack frame -> destroyed   |
|                                           |
|  However, count has been moved to heap   |
|  (because closures hold references to it)|
|                                           |
|  +-------------+                         |
|  | Heap        |                         |
|  |  count: 0   | <- referenced by increment |
|  |             | <- referenced by decrement |
|  |             | <- referenced by getCount  |
|  +-------------+                         |
+------------------------------------------+

Step 4: When increment() is called
+------------------------------------------+
|  increment() execution:                   |
|    1. Retrieve count from captured env    |
|    2. Update count from 0 to 1           |
|    3. Return 1                           |
|                                           |
|  +-------------+                         |
|  | Heap        |                         |
|  |  count: 1   | <- Updated!             |
|  +-------------+                         |
+------------------------------------------+
```

```javascript
// JavaScript: Closures fundamentals (complete version)
function makeCounter() {
    let count = 0;  // Free variable (captured by the closure)
    return {
        increment: () => ++count,
        decrement: () => --count,
        getCount: () => count,
    };
}

const counter = makeCounter();
console.log(counter.increment());  // 1
console.log(counter.increment());  // 2
console.log(counter.decrement());  // 1
console.log(counter.getCount());   // 1

// count survives even after makeCounter finishes executing
// -> Because the closures hold references to count

// Multiple independent counters can be created
const counter2 = makeCounter();
console.log(counter2.increment());  // 1 (independent from counter)
console.log(counter.getCount());    // 1 (counter is unaffected)
```

### 1.5 Duality of Closures and Objects

In computer science, closures and objects are considered to be **duals** of each other. This originates from a famous aphorism by Norman Adams and Jonathan Rees.

> "Objects are a poor man's closures. Closures are a poor man's objects."

```
+------------------------------------------------------+
|  Duality of Closures and Objects                      |
+------------------------------------------------------+
|                                                        |
|  Closure version:                                      |
|  function makeCounter() {                              |
|      let count = 0;            // <- hidden state      |
|      return {                                          |
|          increment: () => ++count,                     |
|          getCount: () => count,                        |
|      };                                                |
|  }                                                     |
|                                                        |
|  Object version:                                       |
|  class Counter {                                       |
|      #count = 0;               // <- hidden state      |
|      increment() { return ++this.#count; }             |
|      getCount() { return this.#count; }                |
|  }                                                     |
|                                                        |
|  -> Both are "hidden state + methods that operate on it"|
|  -> Only the form of expression differs; essentially   |
|     equivalent                                          |
+------------------------------------------------------+
```

| Aspect | Closure | Object |
|--------|---------|--------|
| State retention | Captured free variables | Instance fields |
| Operation definition | Returned function group | Methods |
| Encapsulation | Natural hiding through scope | Access modifiers (private, etc.) |
| Inheritance | Not directly supported | Class inheritance, interfaces |
| Polymorphism | Achieved via higher-order functions | Achieved via method overriding |
| Memory efficiency | Each closure has an independent environment | Shareable via prototype/vtable |
| Best suited for | Single operations, callbacks, partial application | Multiple operations, complex state |

---

## 2. Closures in Each Language: Detailed Explanation

### 2.1 JavaScript / TypeScript

JavaScript is one of the languages that handles closures most naturally. Since functions are first-class objects and the language has lexical scope, closures are a foundation of the language.

```javascript
// --- Capture Mechanism ---
// JavaScript closures always capture by "reference"

function createAccumulator(initial) {
    let total = initial;

    return {
        add(value) {
            total += value;
            return total;
        },
        subtract(value) {
            total -= value;
            return total;
        },
        reset() {
            total = initial;  // initial is also captured
            return total;
        },
        getTotal() {
            return total;
        }
    };
}

const acc = createAccumulator(100);
console.log(acc.add(50));       // 150
console.log(acc.add(30));       // 180
console.log(acc.subtract(20));  // 160
console.log(acc.reset());       // 100 (back to initial)

// --- IIFE (Immediately Invoked Function Expression) and Closures ---
// Module pattern from the ES5 era
const Module = (function() {
    let privateState = 0;
    const privateHelper = (x) => x * 2;

    return {
        publicMethod(value) {
            privateState += privateHelper(value);
            return privateState;
        },
        getState() {
            return privateState;
        }
    };
})();

Module.publicMethod(5);  // 10
Module.publicMethod(3);  // 16
// privateState, privateHelper are inaccessible from outside
```

```typescript
// TypeScript: Typed closures
type Validator<T> = (value: T) => boolean;
type ValidationRule<T> = {
    validate: Validator<T>;
    message: string;
};

function createValidator<T>(
    rules: ValidationRule<T>[]
): (value: T) => string[] {
    // rules is captured by the closure
    return (value: T): string[] => {
        return rules
            .filter(rule => !rule.validate(value))
            .map(rule => rule.message);
    };
}

const validateAge = createValidator<number>([
    {
        validate: (n) => n >= 0,
        message: "Age must be 0 or greater"
    },
    {
        validate: (n) => n <= 150,
        message: "Age must be 150 or less"
    },
    {
        validate: (n) => Number.isInteger(n),
        message: "Age must be an integer"
    }
]);

console.log(validateAge(25));   // []
console.log(validateAge(-5));   // ["Age must be 0 or greater"]
console.log(validateAge(200));  // ["Age must be 150 or less"]
```

### 2.2 Python

Python closures have unique considerations. In particular, the **restriction on variable rebinding** and the **loop variable trap** are points where many developers stumble.

```python
# --- Basic Closure ---
def make_multiplier(factor):
    def multiply(x):
        return x * factor  # Captures factor (by reference)
    return multiply

double = make_multiplier(2)
triple = make_multiplier(3)
print(double(5))   # 10
print(triple(5))   # 15

# --- The nonlocal Keyword ---
# Introduced in Python 3. Required to rebind captured variables
def make_counter():
    count = 0
    def increment():
        nonlocal count     # Without this, UnboundLocalError
        count += 1
        return count
    def get_count():
        return count       # nonlocal not needed for read-only access
    return increment, get_count

inc, get = make_counter()
print(inc())    # 1
print(inc())    # 2
print(get())    # 2

# --- Loop Variable Trap (Important!) ---
def make_functions():
    functions = []
    for i in range(5):
        functions.append(lambda: i)  # Captures the "reference" to i
    return functions

# All return 4 (the final value of i)
results = [f() for f in make_functions()]
print(results)  # [4, 4, 4, 4, 4]

# Fix 1: Capture the value via default argument
def make_functions_fixed_v1():
    functions = []
    for i in range(5):
        functions.append(lambda i=i: i)  # Copy value via default argument
    return functions

print([f() for f in make_functions_fixed_v1()])  # [0, 1, 2, 3, 4]

# Fix 2: Use functools.partial
from functools import partial

def make_functions_fixed_v2():
    def identity(x):
        return x
    return [partial(identity, i) for i in range(5)]

print([f() for f in make_functions_fixed_v2()])  # [0, 1, 2, 3, 4]

# Fix 3: Use a closure factory
def make_functions_fixed_v3():
    def make_f(val):
        return lambda: val   # val is a local variable of make_f
    return [make_f(i) for i in range(5)]

print([f() for f in make_functions_fixed_v3()])  # [0, 1, 2, 3, 4]
```

### 2.3 Rust

Rust closures are integrated with the ownership system and are classified by three traits: `Fn`, `FnMut`, and `FnOnce`. This guarantees closure safety at compile time.

```rust
// --- Three Closure Traits ---

// 1. Fn: Borrows the environment by immutable reference (&T)
//    -> Can be called any number of times, does not modify the environment
fn demonstrate_fn() {
    let name = String::from("Alice");
    let greet = || println!("Hello, {}!", name);  // Captures &name

    greet();  // Can be called multiple times
    greet();
    println!("name is still valid: {}", name);  // name is still usable
}

// 2. FnMut: Borrows the environment by mutable reference (&mut T)
//    -> Can be called any number of times, but modifies the environment
fn demonstrate_fn_mut() {
    let mut count = 0;
    let mut increment = || {
        count += 1;  // Captures &mut count
        count
    };

    println!("{}", increment());  // 1
    println!("{}", increment());  // 2
    // Note: count cannot be accessed directly while increment is in use
}

// 3. FnOnce: Takes ownership of the environment
//    -> Can only be called once (because ownership is consumed)
fn demonstrate_fn_once() {
    let data = vec![1, 2, 3, 4, 5];
    let consume = move || {
        // The move keyword transfers ownership of data
        let sum: i32 = data.iter().sum();
        println!("Sum: {}, dropping data", sum);
        drop(data);  // Consumes data
    };

    consume();
    // consume();  // Compile error: FnOnce can only be called once
    // println!("{:?}", data);  // Compile error: data has been moved
}

// --- Trait Inclusion Hierarchy ---
//
//  FnOnce ⊃ FnMut ⊃ Fn
//
//  A closure implementing Fn also implements FnMut and FnOnce
//  A closure implementing FnMut also implements FnOnce
//  A closure implementing only FnOnce does not implement Fn or FnMut

// --- Functions that Take Closures as Arguments ---
fn apply_twice<F: Fn(i32) -> i32>(f: F, x: i32) -> i32 {
    f(f(x))
}

fn apply_and_collect<F: FnMut(i32) -> i32>(mut f: F, items: &[i32]) -> Vec<i32> {
    items.iter().map(|&x| f(x)).collect()
}

fn consume_and_run<F: FnOnce() -> String>(f: F) -> String {
    f()  // F is FnOnce, so it can only be called once
}

fn main() {
    // Fn example
    let double = |x: i32| x * 2;
    println!("{}", apply_twice(double, 3));  // 12 (3*2=6, 6*2=12)

    // FnMut example
    let mut offset = 0;
    let add_increasing = |x: i32| {
        offset += 1;
        x + offset
    };
    let result = apply_and_collect(add_increasing, &[10, 20, 30]);
    println!("{:?}", result);  // [11, 22, 33]

    // FnOnce example
    let name = String::from("World");
    let greeting = move || format!("Hello, {}!", name);
    println!("{}", consume_and_run(greeting));  // "Hello, World!"
}
```

```
Rust Closure Trait Decision Flow:

  The closure does the following with environment values...
    |
    +-- Consumes them (move + drop) -> FnOnce only
    |    Example: move || { drop(data); }
    |
    +-- Mutates them (&mut) -> FnMut (+ FnOnce)
    |    Example: || { count += 1; }
    |
    +-- Reads only (&) -> Fn (+ FnMut + FnOnce)
         Example: || { println!("{}", name); }

  * The move keyword only changes the "capture method,"
    it does not directly determine the trait
    move || println!("{}", name)  can implement Fn
    (even after moving ownership, if only reading, it's Fn)
```

### 2.4 Go

Go closures are simple: function literals (anonymous functions) capture variables from the outer scope by reference.

```go
package main

import (
    "fmt"
    "sync"
)

// Basic closure
func makeAdder(base int) func(int) int {
    return func(x int) int {
        return base + x  // Captures base by reference
    }
}

// Generator pattern
func fibonacci() func() int {
    a, b := 0, 1
    return func() int {
        result := a
        a, b = b, a+b  // Captures a, b mutably
        return result
    }
}

// Middleware pattern (common in Go)
type Middleware func(http.HandlerFunc) http.HandlerFunc

func withLogging(logger *log.Logger) Middleware {
    return func(next http.HandlerFunc) http.HandlerFunc {
        return func(w http.ResponseWriter, r *http.Request) {
            logger.Printf("%s %s", r.Method, r.URL.Path)
            next(w, r)
        }
    }
}

// --- Loop Variable Trap and Fix in Go ---
func main() {
    add5 := makeAdder(5)
    add10 := makeAdder(10)
    fmt.Println(add5(3))   // 8
    fmt.Println(add10(3))  // 13

    fib := fibonacci()
    for i := 0; i < 8; i++ {
        fmt.Printf("%d ", fib())  // 0 1 1 2 3 5 8 13
    }
    fmt.Println()

    // Loop variable trap (before Go 1.21)
    var wg sync.WaitGroup
    for i := 0; i < 5; i++ {
        wg.Add(1)
        // In Go 1.22+, i is new for each iteration, so this is not a problem
        // In Go 1.21 and earlier, i must be shadowed
        i := i  // Shadowing (needed before Go 1.21)
        go func() {
            defer wg.Done()
            fmt.Println(i)
        }()
    }
    wg.Wait()
}
```

### 2.5 Java

Java provides closure-like functionality through lambda expressions (Java 8+), but with the restriction that only **effectively final variables** can be captured.

```java
import java.util.function.*;
import java.util.*;
import java.util.stream.*;

public class ClosureExamples {

    // Basic: Capture via lambda expression
    public static Function<Integer, Integer> makeAdder(int base) {
        // base is effectively final (never reassigned)
        return x -> x + base;
    }

    // When mutable state is needed, wrap it in an array or Atomic
    public static Supplier<Integer> makeCounter() {
        // int count = 0; cannot be reassigned
        final int[] count = {0};  // Array elements can be modified
        return () -> ++count[0];
    }

    // Using closures with the Stream API
    public static List<String> filterAndTransform(
            List<String> items, String prefix, int minLength) {
        // prefix, minLength are effectively final
        return items.stream()
            .filter(s -> s.length() >= minLength)
            .map(s -> prefix + s.toUpperCase())
            .collect(Collectors.toList());
    }

    public static void main(String[] args) {
        Function<Integer, Integer> add5 = makeAdder(5);
        System.out.println(add5.apply(3));  // 8

        Supplier<Integer> counter = makeCounter();
        System.out.println(counter.get());  // 1
        System.out.println(counter.get());  // 2

        List<String> result = filterAndTransform(
            Arrays.asList("hi", "hello", "hey", "greetings"),
            ">> ", 4
        );
        System.out.println(result);
        // [>> HELLO, >> GREETINGS]

        // Compile error example
        // int x = 10;
        // Runnable r = () -> { x = 20; };  // Error: x is not effectively final
    }
}
```

### 2.6 C++

C++ closures (lambda expressions, since C++11) explicitly specify the capture method through a capture list.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>

int main() {
    // --- Capture Types ---

    // [=] : Capture all external variables by value (copy)
    int x = 10, y = 20;
    auto by_value = [=]() { return x + y; };
    x = 100;  // Even if modified...
    std::cout << by_value() << std::endl;  // 30 (value at time of copy)

    // [&] : Capture all external variables by reference
    int count = 0;
    auto by_ref = [&]() { return ++count; };
    std::cout << by_ref() << std::endl;  // 1
    std::cout << by_ref() << std::endl;  // 2
    std::cout << count << std::endl;     // 2 (modified because captured by reference)

    // Individual specification: [x, &y] -> x by value, y by reference
    int a = 1, b = 2;
    auto mixed = [a, &b]() {
        // a is read-only (value copy)
        b += a;  // b can be modified because captured by reference
        return b;
    };
    std::cout << mixed() << std::endl;  // 3
    std::cout << b << std::endl;        // 3

    // mutable: Allows modification of value-captured variables within the lambda
    int counter = 0;
    auto mut_lambda = [counter]() mutable {
        return ++counter;  // Modifies the copy inside the lambda
    };
    std::cout << mut_lambda() << std::endl;  // 1
    std::cout << mut_lambda() << std::endl;  // 2
    std::cout << counter << std::endl;       // 0 (original variable is unchanged)

    // --- Combination with STL Algorithms ---
    std::vector<int> nums = {5, 2, 8, 1, 9, 3};
    int threshold = 4;

    // Filter only elements greater than threshold
    std::vector<int> filtered;
    std::copy_if(nums.begin(), nums.end(),
                 std::back_inserter(filtered),
                 [threshold](int n) { return n > threshold; });
    // filtered: {5, 8, 9}

    return 0;
}
```

---

## 3. Cross-Language Comparison Tables

### 3.1 Capture Method Comparison

| Language | Capture Method | Mutable Capture | Explicit Specification | Notes |
|----------|---------------|----------------|----------------------|-------|
| JavaScript | Reference (automatic) | Possible | Not required | Most natural; beware of var hoisting |
| Python | Reference (automatic) | Requires nonlocal | Not required | Loop variable trap; nonlocal/global |
| Rust | Borrow or move | Possible with FnMut | Controlled via move | Integrated with ownership; compile-time safety guarantee |
| Go | Reference (automatic) | Possible | Not required | Beware of combination with goroutines |
| Java | Value copy | Not possible (effectively final) | Not required | Extension of anonymous classes |
| C++ | Value or reference | Possible with mutable | Capture list | Most fine-grained control available |
| Swift | Reference (automatic) | Possible | [weak/unowned] | Integrated with ARC; beware of retain cycles |

### 3.2 Syntax Comparison

| Language | Closure Syntax | Type Inference |
|----------|---------------|----------------|
| JavaScript | `(x) => x * 2` / `function(x) { return x * 2; }` | Full |
| Python | `lambda x: x * 2` / `def f(x): return x * 2` | Type hints optional |
| Rust | `\|x\| x * 2` / `\|x: i32\| -> i32 { x * 2 }` | Inferable in most cases |
| Go | `func(x int) int { return x * 2 }` | Return type must be explicit |
| Java | `(x) -> x * 2` / `(Integer x) -> x * 2` | Inferred from functional interface |
| C++ | `[](int x) { return x * 2; }` | Receivable with auto |
| Swift | `{ x in x * 2 }` / `{ $0 * 2 }` | Full |

---

## 4. Practical Closure Patterns

### 4.1 Pattern 1: Private State (Module Pattern)

```javascript
// Data encapsulation - holds state that cannot be directly accessed from outside
const createLogger = (prefix) => {
    let logCount = 0;
    const history = [];

    return {
        log(msg) {
            logCount++;
            const entry = `[${prefix}] #${logCount}: ${msg}`;
            history.push(entry);
            console.log(entry);
        },
        warn(msg) {
            logCount++;
            const entry = `[${prefix}] #${logCount} ⚠: ${msg}`;
            history.push(entry);
            console.warn(entry);
        },
        getCount() {
            return logCount;
        },
        getHistory() {
            return [...history];  // Return a copy (defensive copy)
        }
    };
};

const appLog = createLogger("APP");
appLog.log("Started");    // [APP] #1: Started
appLog.warn("Low memory"); // [APP] #2 ⚠: Low memory
console.log(appLog.getCount());    // 2
console.log(appLog.getHistory());  // ["[APP] #1: Started", ...]
// logCount, history are inaccessible from outside
```

### 4.2 Pattern 2: Memoization

```javascript
// Generic memoization function
function memoize(fn) {
    const cache = new Map();
    const memoized = function(...args) {
        const key = JSON.stringify(args);
        if (cache.has(key)) {
            return cache.get(key);
        }
        const result = fn.apply(this, args);
        cache.set(key, result);
        return result;
    };

    // Also provide cache management methods
    memoized.clearCache = () => cache.clear();
    memoized.cacheSize = () => cache.size;
    memoized.hasCache = (...args) => cache.has(JSON.stringify(args));

    return memoized;
}

// Usage example: Fibonacci sequence
const fibonacci = memoize(function fib(n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
});

console.log(fibonacci(50));  // 12586269025 (computed instantly)
console.log(fibonacci.cacheSize());  // 51

// Usage example: Caching API calls
const fetchUser = memoize(async (userId) => {
    const response = await fetch(`/api/users/${userId}`);
    return response.json();
});
```

### 4.3 Pattern 3: Currying

```javascript
// Generic currying function
const curry = (fn) => {
    const arity = fn.length;
    return function curried(...args) {
        if (args.length >= arity) {
            return fn(...args);
        }
        return (...moreArgs) => curried(...args, ...moreArgs);
    };
};

// Usage example
const add = curry((a, b, c) => a + b + c);
console.log(add(1)(2)(3));     // 6
console.log(add(1, 2)(3));     // 6
console.log(add(1)(2, 3));     // 6
console.log(add(1, 2, 3));     // 6

// Practical example: Log formatter
const formatLog = curry((level, module, message) =>
    `[${new Date().toISOString()}] [${level}] [${module}] ${message}`
);

const errorLog = formatLog("ERROR");
const errorLogAuth = errorLog("AUTH");

console.log(errorLogAuth("Login failed"));
// [2026-03-06T...] [ERROR] [AUTH] Login failed
```

### 4.4 Pattern 4: Debounce and Throttle

```javascript
// Debounce: Execute after a fixed delay from the last call
function createDebounce(fn, delay) {
    let timeoutId = null;
    return function(...args) {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => fn.apply(this, args), delay);
    };
}

// Throttle: Execute at most once per fixed interval
function createThrottle(fn, interval) {
    let lastTime = 0;
    let timeoutId = null;
    return function(...args) {
        const now = Date.now();
        const remaining = interval - (now - lastTime);
        clearTimeout(timeoutId);
        if (remaining <= 0) {
            lastTime = now;
            fn.apply(this, args);
        } else {
            timeoutId = setTimeout(() => {
                lastTime = Date.now();
                fn.apply(this, args);
            }, remaining);
        }
    };
}

// Usage examples
const debouncedSearch = createDebounce((query) => {
    console.log(`Searching for: ${query}`);
}, 300);

const throttledScroll = createThrottle(() => {
    console.log("Scroll position:", window.scrollY);
}, 100);
```

### 4.5 Pattern 5: Function Composition

```javascript
// Function composition: Chain multiple functions to create a new function
const compose = (...fns) =>
    fns.reduce((f, g) => (...args) => f(g(...args)));

const pipe = (...fns) =>
    fns.reduce((f, g) => (...args) => g(f(...args)));

// Usage example: Data transformation pipeline
const trim = (s) => s.trim();
const toLowerCase = (s) => s.toLowerCase();
const replaceSpaces = (s) => s.replace(/\s+/g, "-");
const addPrefix = (prefix) => (s) => `${prefix}${s}`;  // Closure!

const slugify = pipe(
    trim,
    toLowerCase,
    replaceSpaces,
    addPrefix("/blog/")
);

console.log(slugify("  Hello World  "));  // "/blog/hello-world"
console.log(slugify(" Closures Are Fun "));  // "/blog/closures-are-fun"
```

### 4.6 Pattern 6: Iterators / Generators

Closures can be used to build lazy-evaluation iterators. This is achieved by having closures retain internal state.

```javascript
// Infinite sequence generator
function range(start = 0, step = 1) {
    let current = start;
    return {
        next() {
            const value = current;
            current += step;
            return { value, done: false };
        },
        take(n) {
            const result = [];
            for (let i = 0; i < n; i++) {
                result.push(this.next().value);
            }
            return result;
        },
        [Symbol.iterator]() {
            return this;
        }
    };
}

const odds = range(1, 2);
console.log(odds.take(5));  // [1, 3, 5, 7, 9]

// Filter iterator (closure chaining)
function filterIterator(iterator, predicate) {
    return {
        next() {
            while (true) {
                const item = iterator.next();
                if (item.done) return item;
                if (predicate(item.value)) return item;
            }
        }
    };
}

function mapIterator(iterator, transform) {
    return {
        next() {
            const item = iterator.next();
            if (item.done) return item;
            return { value: transform(item.value), done: false };
        }
    };
}

// Usage: Get 5 doubled even numbers starting from 1
const nums = range(1, 1);
const evens = filterIterator(nums, n => n % 2 === 0);
const doubled = mapIterator(evens, n => n * 2);

const results = [];
for (let i = 0; i < 5; i++) {
    results.push(doubled.next().value);
}
console.log(results);  // [4, 8, 12, 16, 20]
```

```python
# Python: Closure-based iterator
def infinite_counter(start=0):
    count = start
    def next_val():
        nonlocal count
        result = count
        count += 1
        return result
    return next_val

counter = infinite_counter(10)
print([counter() for _ in range(5)])  # [10, 11, 12, 13, 14]

# Combining generators and closures
def sliding_window(size):
    """Closure that manages a sliding window"""
    window = []
    def add(value):
        window.append(value)
        if len(window) > size:
            window.pop(0)
        return list(window)  # Defensive copy
    return add

slider = sliding_window(3)
print(slider(1))   # [1]
print(slider(2))   # [1, 2]
print(slider(3))   # [1, 2, 3]
print(slider(4))   # [2, 3, 4]
print(slider(5))   # [3, 4, 5]
```

### 4.7 Pattern 7: Middleware / Decorators

```python
# Python: Decorators are a typical use case for closures
import time
import functools

def retry(max_attempts=3, delay=1.0):
    """Retry decorator (captures settings via closure)"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts:
                        time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator

def rate_limit(calls_per_second):
    """Rate limit decorator"""
    min_interval = 1.0 / calls_per_second
    last_call_time = [0.0]  # Wrapped in a list as an alternative to nonlocal

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_call_time[0]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            last_call_time[0] = time.time()
            return func(*args, **kwargs)
        return wrapper
    return decorator

@retry(max_attempts=3, delay=0.5)
@rate_limit(calls_per_second=2)
def fetch_data(url):
    """Data fetching with retry and rate limiting"""
    import urllib.request
    return urllib.request.urlopen(url).read()
```

```
Closure Structure of Decorators (Nesting Diagram):

  @retry(max_attempts=3, delay=0.5)
  def fetch_data(url): ...

  When expanded:

  retry(max_attempts=3, delay=0.5)     <- Captures settings
    +-- decorator(func)                <- Captures the original function
        +-- wrapper(*args, **kwargs)   <- Receives arguments and executes
            |
            |  Variables captured by wrapper:
            |    - func (original function)
            |    - max_attempts (3)
            |    - delay (0.5)
            |
            |  Call chain:
            |    fetch_data("...") -> wrapper("...") -> func("...")
            |                         ^ retry logic
            |
            +-- A closure with three nested scopes
```

---

## 5. Memory and Closures

### 5.1 Closure Memory Model

Because closures hold references to the environment at definition time, the number of variables that are not eligible for garbage collection (GC) may increase.

```
Closure Memory Lifecycle:

  [At function definition time]
  +------------------------------------------+
  |  Outer scope                              |
  |  +-------------------------+             |
  |  | largeData = [...]       |-----+       |
  |  | config = {...}          |--+  |       |
  |  | name = "test"           |  |  |       |
  |  +-------------------------+  |  |       |
  |                                |  |       |
  |  closure = function() {        |  |       |
  |      use(config);         -----+  |       |
  |      // name is not used          |       |
  |      // largeData is not used ----+       |
  |  }                                        |
  +------------------------------------------+

  [After outer scope ends]
  +------------------------------------------+
  |  GC eligible:                             |
  |    name = "test"        <- No references -> freed |
  |                                           |
  |  Not GC eligible (language-dependent):    |
  |    config = {...}       <- Referenced by closure  |
  |    largeData = [...]    <- Varies by language     |
  |                                           |
  |  * The V8 engine may optimize away        |
  |    variables that are not actually used    |
  |  * However, when using debugger, all      |
  |    variables are retained                  |
  +------------------------------------------+
```

### 5.2 Memory Management by Language

| Language | GC Method | Closure Impact | Countermeasure |
|----------|-----------|---------------|----------------|
| JavaScript | Mark-and-Sweep | Reference retention extends lifetime | Set unnecessary references to null |
| Python | Reference counting + cycle detection | Risk of circular references | Use weakref |
| Rust | Ownership (no GC) | Safety guaranteed at compile time | Explicit lifetimes |
| Go | Concurrent Mark-and-Sweep | Optimized via escape analysis | Use pointers for large data |
| Java | Generational GC | Same as anonymous classes | Use WeakReference |
| C++ | Manual (no GC) | Risk of dangling references | Smart pointers, value capture |
| Swift | ARC (reference counting) | Risk of retain cycles | [weak], [unowned] |

### 5.3 Concrete Examples and Solutions for Memory Leaks

```javascript
// Anti-pattern 1: Unnecessarily capturing huge data
function setupHandler() {
    const hugeData = loadHugeData();  // 100MB of data
    const element = document.getElementById("btn");

    element.addEventListener("click", () => {
        // Only uses the name property of hugeData, but
        // the entire hugeData (100MB) is captured by the closure
        console.log(hugeData.name);
    });
}

// Fix: Extract only the needed values in advance
function setupHandlerFixed() {
    const hugeData = loadHugeData();
    const name = hugeData.name;  // Extract only the needed value
    // hugeData can now become GC eligible

    const element = document.getElementById("btn");
    element.addEventListener("click", () => {
        console.log(name);  // Only name is captured (a few bytes)
    });
}

// Anti-pattern 2: Forgetting to remove event listeners
function createWidget() {
    const state = { count: 0, data: new Array(10000) };

    const handler = () => {
        state.count++;
        updateUI(state);
    };

    window.addEventListener("resize", handler);

    // Even when the Widget is destroyed, handler persists
    // -> state is not freed either -> memory leak
}

// Fix: Return a cleanup function
function createWidgetFixed() {
    const state = { count: 0, data: new Array(10000) };

    const handler = () => {
        state.count++;
        updateUI(state);
    };

    window.addEventListener("resize", handler);

    // Return a cleanup function
    return {
        destroy() {
            window.removeEventListener("resize", handler);
            // Reference to handler is removed, and state becomes GC eligible
        }
    };
}

// Cleanup pattern in React Hooks
function useWindowResize(callback) {
    useEffect(() => {
        window.addEventListener("resize", callback);
        return () => {
            // Cleanup: Executed when component unmounts
            window.removeEventListener("resize", callback);
        };
    }, [callback]);
}
```

```python
# Python: Memory leak due to circular references

# Closure circular reference
class Node:
    def __init__(self, value):
        self.value = value
        self.get_value = lambda: self.value
        # self.get_value -> lambda -> self -> self.get_value
        # Circular reference occurs!

# Fix with weakref
import weakref

class NodeFixed:
    def __init__(self, value):
        self.value = value
        weak_self = weakref.ref(self)
        self.get_value = lambda: weak_self().value if weak_self() else None
```

### 5.4 Dangling Reference Problem in C++

```cpp
#include <iostream>
#include <functional>

// Dangerous: Capturing a local variable by reference, then leaving scope
std::function<int()> createDangling() {
    int x = 42;
    return [&x]() { return x; };
    // x is destroyed when scope is exited
    // -> The returned lambda references destroyed x
    // -> Undefined behavior!
}

// Safe: Use value capture
std::function<int()> createSafe() {
    int x = 42;
    return [x]() { return x; };  // Holds a copy of x
    // The copy inside the lambda lives as long as the lambda
}

// Safe: Keep on heap with shared_ptr
std::function<int()> createWithSharedPtr() {
    auto x = std::make_shared<int>(42);
    return [x]() { return *x; };
    // Reference count incremented by copying the shared_ptr
    // x lives as long as the lambda lives
}
```

---

## 6. Advanced Topics

### 6.1 Closures and Coroutines

Because closures can retain internal state, they serve as the foundation for coroutines (functions that can be suspended and resumed).

```python
# Python: Generators are an evolution of closures
def coroutine_accumulator():
    """Coroutine that receives values via send() and returns cumulative totals"""
    total = 0
    while True:
        value = yield total
        if value is not None:
            total += value

# Usage example
acc = coroutine_accumulator()
next(acc)          # Initialize the generator (advance to the first yield)
print(acc.send(10))  # 10
print(acc.send(20))  # 30
print(acc.send(5))   # 35

# Implementing equivalent functionality with closures
def closure_accumulator():
    total = [0]
    def send(value):
        total[0] += value
        return total[0]
    return send

acc2 = closure_accumulator()
print(acc2(10))  # 10
print(acc2(20))  # 30
print(acc2(5))   # 35
```

### 6.2 Lazy Evaluation with Closures

```javascript
// Implementing Lazy Evaluation with closures
function lazy(computation) {
    let result;
    let computed = false;

    return () => {
        if (!computed) {
            result = computation();
            computed = true;
        }
        return result;
    };
}

// Usage: Defer expensive computation until needed
const expensiveResult = lazy(() => {
    console.log("Computing...");
    let sum = 0;
    for (let i = 0; i < 1000000; i++) sum += i;
    return sum;
});

// Not yet computed at this point
console.log("Before access");
console.log(expensiveResult());  // "Computing..." -> 499999500000
console.log(expensiveResult());  // 499999500000 (from cache, no recomputation)
```

```python
# Python: Lazy Property
class LazyProperty:
    """Lazy property descriptor using closures"""
    def __init__(self, func):
        self.func = func
        self.attr_name = f"_lazy_{func.__name__}"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if not hasattr(obj, self.attr_name):
            setattr(obj, self.attr_name, self.func(obj))
        return getattr(obj, self.attr_name)

class DataProcessor:
    def __init__(self, raw_data):
        self.raw_data = raw_data

    @LazyProperty
    def processed(self):
        """Computed only on first access"""
        print("Processing data...")
        return [x * 2 for x in self.raw_data]

    @LazyProperty
    def statistics(self):
        """Lazy property that depends on processed"""
        print("Computing statistics...")
        data = self.processed
        return {
            "mean": sum(data) / len(data),
            "max": max(data),
            "min": min(data)
        }

dp = DataProcessor([1, 2, 3, 4, 5])
# Nothing is computed at this point
print(dp.statistics)  # "Processing data..." -> "Computing statistics..." -> {...}
print(dp.statistics)  # Returned immediately from cache
```

### 6.3 Closures and Concurrency

```go
package main

import (
    "fmt"
    "sync"
    "sync/atomic"
)

// Thread-safe closure counter
func makeAtomicCounter() func() int64 {
    var count int64 = 0
    return func() int64 {
        return atomic.AddInt64(&count, 1)
    }
}

// Worker pool pattern (defining tasks with closures)
func workerPool(numWorkers int, tasks []func()) {
    var wg sync.WaitGroup
    taskCh := make(chan func(), len(tasks))

    // Start workers
    for i := 0; i < numWorkers; i++ {
        wg.Add(1)
        go func(workerID int) {
            defer wg.Done()
            for task := range taskCh {
                task()  // Execute the closure
            }
        }(i)
    }

    // Submit tasks
    for _, task := range tasks {
        taskCh <- task
    }
    close(taskCh)

    wg.Wait()
}

func main() {
    counter := makeAtomicCounter()

    var wg sync.WaitGroup
    for i := 0; i < 100; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            counter()
        }()
    }
    wg.Wait()
    fmt.Println(counter() - 1)  // 100

    // Worker pool usage example
    var mu sync.Mutex
    results := make([]int, 0)

    tasks := make([]func(), 10)
    for i := 0; i < 10; i++ {
        val := i  // Copy loop variable
        tasks[i] = func() {
            result := val * val
            mu.Lock()
            results = append(results, result)
            mu.Unlock()
        }
    }

    workerPool(4, tasks)
    fmt.Println(results)  // [0, 1, 4, 9, 16, 25, 36, 49, 64, 81] (order may vary)
}
```

### 6.4 Type Systems and Closures

```rust
// Rust: Each closure has a unique (anonymous) type
// -> Accept them via generics or trait objects

// Generics version (zero-cost abstraction, static dispatch)
fn apply_generic<F: Fn(i32) -> i32>(f: F, x: i32) -> i32 {
    f(x)
}

// Trait object version (dynamic dispatch, runtime cost)
fn apply_dynamic(f: &dyn Fn(i32) -> i32, x: i32) -> i32 {
    f(x)
}

// Box<dyn Fn> to store on heap (with ownership)
fn make_adder(n: i32) -> Box<dyn Fn(i32) -> i32> {
    Box::new(move |x| x + n)
}

// Return with impl Fn (zero-cost, but type is opaque)
fn make_multiplier(n: i32) -> impl Fn(i32) -> i32 {
    move |x| x * n
}

fn main() {
    let double = |x| x * 2;

    // Generics: Type determined at compile time, inlining possible
    println!("{}", apply_generic(double, 5));   // 10
    println!("{}", apply_generic(|x| x + 1, 5)); // 6

    // Trait object: Dispatched at runtime
    println!("{}", apply_dynamic(&double, 5));  // 10

    // Box<dyn Fn>: Stored on the heap
    let add5 = make_adder(5);
    println!("{}", add5(10));  // 15

    // impl Fn: Stored on the stack (efficient)
    let triple = make_multiplier(3);
    println!("{}", triple(10));  // 30
}
```

```
Comparison of Rust Closure Storage Methods:

  +-----------------------------------------------------+
  |  Method              | Dispatch | Memory  | Cost     |
  +-----------------------------------------------------+
  |  Generics <F>        | Static   | Stack   | Zero     |
  |  impl Fn             | Static   | Stack   | Zero     |
  |  &dyn Fn             | Dynamic  | Reference| Small   |
  |  Box<dyn Fn>         | Dynamic  | Heap    | Medium   |
  +-----------------------------------------------------+

  When to use which:
    - Function argument -> Generics (most efficient)
    - Function return value (single type) -> impl Fn
    - Function return value (multiple possible types) -> Box<dyn Fn>
    - Storing in a collection -> Box<dyn Fn> or Vec<Box<dyn Fn>>
```

---

## 7. Anti-Patterns and Pitfalls

### 7.1 Anti-Pattern 1: Unintended Loop Variable Capture

This is a classic trap that occurs in many languages including JavaScript, Python, and Go. Loop variables are not created anew for each iteration; instead, the same variable is updated (depending on the language and version).

```javascript
// Anti-pattern: Closures in a loop using var
function createButtons() {
    const buttons = [];
    for (var i = 0; i < 5; i++) {
        // var i is a single variable for the entire loop (function scope)
        buttons.push({
            label: `Button ${i}`,
            onClick: function() {
                console.log(`Clicked button ${i}`);
                // All closures reference the same i
            }
        });
    }
    return buttons;
}

const btns = createButtons();
btns[0].onClick();  // "Clicked button 5" <- Expected 0
btns[1].onClick();  // "Clicked button 5" <- Expected 1
btns[4].onClick();  // "Clicked button 5" <- Expected 4

// Fix 1: Use let (ES6+, most recommended)
function createButtonsFixed1() {
    const buttons = [];
    for (let i = 0; i < 5; i++) {
        // let i creates a new variable for each iteration (block scope)
        buttons.push({
            label: `Button ${i}`,
            onClick: function() {
                console.log(`Clicked button ${i}`);
            }
        });
    }
    return buttons;
}

// Fix 2: Create a new scope with IIFE (ES5 era technique)
function createButtonsFixed2() {
    const buttons = [];
    for (var i = 0; i < 5; i++) {
        (function(index) {
            buttons.push({
                label: `Button ${index}`,
                onClick: function() {
                    console.log(`Clicked button ${index}`);
                }
            });
        })(i);  // Copy and pass the value of i
    }
    return buttons;
}

// Fix 3: Use forEach
function createButtonsFixed3() {
    return Array.from({ length: 5 }, (_, i) => ({
        label: `Button ${i}`,
        onClick() {
            console.log(`Clicked button ${i}`);
        }
    }));
}
```

```
Mechanism of the Loop Variable Trap:

  When using var:
  +------------------------------------------+
  |  Scope: entire createButtons             |
  |                                          |
  |  var i;  <- single variable              |
  |                                          |
  |  i=0: closure0 -> references i ----+     |
  |  i=1: closure1 -> references i ----+     |
  |  i=2: closure2 -> references i ----+     |
  |  i=3: closure3 -> references i ----+     |
  |  i=4: closure4 -> references i ----+     |
  |  Loop ends: i=5                    |     |
  |                                    v     |
  |  Value of i referenced by all closures -> 5  |
  +------------------------------------------+

  When using let:
  +------------------------------------------+
  |  Scope: entire createButtons             |
  |                                          |
  |  { let i=0; closure0 -> references i }   |
  |  { let i=1; closure1 -> references i }   |
  |  { let i=2; closure2 -> references i }   |
  |  { let i=3; closure3 -> references i }   |
  |  { let i=4; closure4 -> references i }   |
  |                                          |
  |  Each closure references its own i       |
  |  -> Works as expected                    |
  +------------------------------------------+
```

### 7.2 Anti-Pattern 2: Excessive Closure Nesting (Callback Hell)

```javascript
// Anti-pattern: Deeply nested closures
function processOrder(orderId) {
    fetchOrder(orderId, function(order) {
        fetchUser(order.userId, function(user) {
            fetchProducts(order.productIds, function(products) {
                calculateShipping(user.address, products, function(shipping) {
                    applyDiscount(user.membership, order.total, function(finalPrice) {
                        createInvoice(order, user, products, shipping, finalPrice,
                            function(invoice) {
                                sendEmail(user.email, invoice, function(result) {
                                    console.log("Done!", result);
                                });
                            }
                        );
                    });
                });
            });
        });
    });
}

// Fix: Promise chain
function processOrderFixed(orderId) {
    return fetchOrder(orderId)
        .then(order => Promise.all([
            order,
            fetchUser(order.userId),
            fetchProducts(order.productIds)
        ]))
        .then(([order, user, products]) => Promise.all([
            order, user, products,
            calculateShipping(user.address, products)
        ]))
        .then(([order, user, products, shipping]) => Promise.all([
            order, user, products, shipping,
            applyDiscount(user.membership, order.total)
        ]))
        .then(([order, user, products, shipping, finalPrice]) =>
            createInvoice(order, user, products, shipping, finalPrice)
        )
        .then(invoice => sendEmail(invoice.user.email, invoice));
}

// Even better: async/await
async function processOrderAsync(orderId) {
    const order = await fetchOrder(orderId);
    const [user, products] = await Promise.all([
        fetchUser(order.userId),
        fetchProducts(order.productIds)
    ]);
    const shipping = await calculateShipping(user.address, products);
    const finalPrice = await applyDiscount(user.membership, order.total);
    const invoice = await createInvoice(order, user, products, shipping, finalPrice);
    return sendEmail(user.email, invoice);
}
```

### 7.3 Anti-Pattern 3: Incorrect this Binding

```javascript
// Anti-pattern: Losing this in closures within methods
class Timer {
    constructor() {
        this.seconds = 0;
    }

    start() {
        // The function keyword has its own this
        setInterval(function() {
            this.seconds++;  // this is global/undefined, not Timer
            console.log(this.seconds);  // NaN or error
        }, 1000);
    }
}

// Fix 1: Use arrow functions (most recommended)
class TimerFixed1 {
    constructor() {
        this.seconds = 0;
    }

    start() {
        // Arrow functions inherit this from the outer scope
        setInterval(() => {
            this.seconds++;  // this is the TimerFixed1 instance
            console.log(this.seconds);
        }, 1000);
    }
}

// Fix 2: self/that pattern (ES5 era technique)
class TimerFixed2 {
    constructor() {
        this.seconds = 0;
    }

    start() {
        const self = this;  // Capture this in a closure
        setInterval(function() {
            self.seconds++;
            console.log(self.seconds);
        }, 1000);
    }
}

// Fix 3: Use bind
class TimerFixed3 {
    constructor() {
        this.seconds = 0;
    }

    start() {
        setInterval(function() {
            this.seconds++;
            console.log(this.seconds);
        }.bind(this), 1000);  // Explicitly bind this
    }
}
```

---

## 8. Performance Considerations

### 8.1 Closure Overhead

```
Closure Cost Analysis:

  +-------------------------------------------------------------+
  |  Cost Factor              | Impact | Description              |
  +-------------------------------------------------------------+
  |  Environment object       | Small  | One heap allocation      |
  |  creation                 |        |                          |
  |  Variable access          | Tiny   | One additional indirect  |
  |                           |        | reference                |
  |  GC load                  | Medium | More references to track |
  |  Inlining inhibition      | Medium | Optimization may become  |
  |                           |        | more difficult           |
  |  Memory usage             | Varies | Depends on amount of     |
  |                           |        | captured variables       |
  +-------------------------------------------------------------+

  General guidelines:
  - Normal application code: Performance impact is negligible
  - Inside hot loops (millions of executions per second): Avoid closure creation
  - Storing large numbers of closures in arrays: Watch memory usage
```

```javascript
// Performance-conscious closure usage

// In a hot loop, creating closures every time
function processItemsBad(items, threshold) {
    return items.map(item => {
        // This closure creates a new environment each time (though often optimized in practice)
        const check = (val) => val > threshold;  // Unnecessary closure
        return check(item.value) ? item : null;
    }).filter(Boolean);
}

// Create the closure only once
function processItemsGood(items, threshold) {
    const check = (val) => val > threshold;  // Created only once
    return items.map(item =>
        check(item.value) ? item : null
    ).filter(Boolean);
}

// Even better: Don't use a closure when it's not needed
function processItemsBest(items, threshold) {
    return items.filter(item => item.value > threshold);
}
```

### 8.2 V8 Engine Optimizations

Modern JavaScript engines (V8, SpiderMonkey, etc.) perform many optimizations for closures.

```
V8 Closure Optimizations:

  1. Context Optimization
     - Variables that are not used are not captured
     - Example: In a scope with 10 variables where only 1 is used,
       an environment for just that 1 variable is created

  2. Inlining
     - Small closures are expanded at the call site
     - The environment object creation itself is eliminated

  3. Escape Analysis
     - When a closure does not escape the function,
       heap allocation is converted to stack allocation

  4. Hidden Class Optimization
     - Closure objects of the same shape
       share the same Hidden Class

  * However, when eval() or debugger is used, these optimizations
    may be disabled
```

---

## 9. Exercises

### Exercise 1: Basic Level

**Problem 1-1: Implementing a Counter**

Implement a `createCounter` function in JavaScript that meets the following specification.

```javascript
// Specification:
// - createCounter(initial) returns a counter with initial value initial
// - increment() increases the value by 1 and returns it
// - decrement() decreases the value by 1 and returns it
// - reset() resets to the initial value and returns it
// - getCount() returns the current value

// Usage example:
const c = createCounter(10);
c.increment();  // 11
c.increment();  // 12
c.decrement();  // 11
c.reset();      // 10
c.getCount();   // 10
```

<details>
<summary>Solution</summary>

```javascript
function createCounter(initial) {
    let count = initial;
    return {
        increment() { return ++count; },
        decrement() { return --count; },
        reset() { count = initial; return count; },
        getCount() { return count; }
    };
}
```
</details>

**Problem 1-2: once Function**

Implement a `once` function that creates a function that can only be executed once.

```javascript
// Specification:
// - once(fn) returns a function that executes fn only once
// - Subsequent calls return the first result

const expensiveCalc = once(() => {
    console.log("Computing...");
    return 42;
});

expensiveCalc();  // "Computing..." -> 42
expensiveCalc();  // 42 (no recomputation, no log output)
```

<details>
<summary>Solution</summary>

```javascript
function once(fn) {
    let called = false;
    let result;
    return function(...args) {
        if (!called) {
            result = fn.apply(this, args);
            called = true;
        }
        return result;
    };
}
```
</details>

### Exercise 2: Intermediate Level

**Problem 2-1: Pipeline Functions**

Implement `pipe`, which composes functions from left to right, and `compose`, which composes from right to left.

```javascript
// Specification:
// - pipe(f, g, h)(x) is equivalent to h(g(f(x)))
// - compose(f, g, h)(x) is equivalent to f(g(h(x)))

const double = x => x * 2;
const addOne = x => x + 1;
const square = x => x * x;

pipe(double, addOne, square)(3);     // square(addOne(double(3))) = square(7) = 49
compose(double, addOne, square)(3);  // double(addOne(square(3))) = double(10) = 20
```

<details>
<summary>Solution</summary>

```javascript
function pipe(...fns) {
    return function(x) {
        return fns.reduce((acc, fn) => fn(acc), x);
    };
}

function compose(...fns) {
    return function(x) {
        return fns.reduceRight((acc, fn) => fn(acc), x);
    };
}
```
</details>

**Problem 2-2: Memoization with LRU Cache**

Implement a memoization function that retains at most `capacity` cache entries. When the cache is full, delete the oldest (Least Recently Used) entry.

```javascript
// Specification:
// - Implement memoizeLRU(fn, capacity)
// - When cache capacity is exceeded, delete the oldest entry
// - When an existing entry is accessed, update it to be the most recent

const cachedFn = memoizeLRU((x) => x * x, 3);
cachedFn(1);  // Compute: 1  -> Cache: [1]
cachedFn(2);  // Compute: 4  -> Cache: [1, 2]
cachedFn(3);  // Compute: 9  -> Cache: [1, 2, 3]
cachedFn(1);  // Cache hit -> Cache: [2, 3, 1] (1 becomes most recent)
cachedFn(4);  // Compute: 16 -> Cache: [3, 1, 4] (2 is deleted)
cachedFn(2);  // Compute: 4  -> Cache: [1, 4, 2] (3 is deleted, 2 is recomputed)
```

<details>
<summary>Solution</summary>

```javascript
function memoizeLRU(fn, capacity) {
    // Map preserves insertion order, making it suitable for LRU
    const cache = new Map();

    return function(...args) {
        const key = JSON.stringify(args);

        if (cache.has(key)) {
            // Move the accessed entry to most recent
            const value = cache.get(key);
            cache.delete(key);
            cache.set(key, value);
            return value;
        }

        const result = fn.apply(this, args);

        if (cache.size >= capacity) {
            // Delete the oldest (first) entry
            const oldestKey = cache.keys().next().value;
            cache.delete(oldestKey);
        }

        cache.set(key, result);
        return result;
    };
}
```
</details>

### Exercise 3: Advanced Level

**Problem 3-1: Observable Pattern**

Implement reactive data binding using closures.

```javascript
// Specification:
// - observable(initialValue) returns { get, set, subscribe }
// - subscribe(callback) calls callback each time the value changes
// - subscribe returns an unsubscribe function
// - computed(observables, computeFn) creates a derived value

const firstName = observable("John");
const lastName = observable("Doe");
const fullName = computed(
    [firstName, lastName],
    (first, last) => `${first} ${last}`
);

fullName.subscribe(name => console.log(`Name: ${name}`));

firstName.set("Jane");  // "Name: Jane Doe"
lastName.set("Smith");  // "Name: Jane Smith"
```

<details>
<summary>Solution</summary>

```javascript
function observable(initialValue) {
    let value = initialValue;
    const subscribers = new Set();

    return {
        get() {
            return value;
        },
        set(newValue) {
            if (value !== newValue) {
                value = newValue;
                subscribers.forEach(cb => cb(value));
            }
        },
        subscribe(callback) {
            subscribers.add(callback);
            // Notify with current value upon registration (optional)
            callback(value);
            // Return an unsubscribe function
            return () => subscribers.delete(callback);
        }
    };
}

function computed(observables, computeFn) {
    const result = observable(
        computeFn(...observables.map(o => o.get()))
    );

    // Recompute when any dependent observable changes
    observables.forEach(obs => {
        obs.subscribe(() => {
            const newValue = computeFn(...observables.map(o => o.get()));
            result.set(newValue);
        });
    });

    return result;
}
```
</details>

**Problem 3-2: Rust Closure Type Puzzle**

Fix the compilation errors in the following Rust code, and identify the trait each closure implements.

```rust
// This code has multiple compilation errors. Fix them.
fn main() {
    let mut items = vec![1, 2, 3, 4, 5];

    // (a) Filter
    let threshold = 3;
    let big_items: Vec<&i32> = items.iter().filter(|x| **x > threshold).collect();

    // (b) Transform and accumulate
    let mut sum = 0;
    let doubled: Vec<i32> = items.iter().map(|x| {
        sum += x;   // Error location
        x * 2
    }).collect();

    // (c) Ownership transfer
    let data = String::from("hello");
    let printer = || println!("{}", data);
    printer();
    println!("{}", data);  // Potential error location

    // (d) Returning from a function
    fn make_greeter(name: String) -> impl Fn() {
        || println!("Hello, {}!", name)  // Error location
    }
}
```

<details>
<summary>Solution</summary>

```rust
fn main() {
    let items = vec![1, 2, 3, 4, 5];

    // (a) Fn trait: Immutably borrows threshold
    let threshold = 3;
    let big_items: Vec<&i32> = items.iter().filter(|x| **x > threshold).collect();
    println!("{:?}", big_items);  // [4, 5]

    // (b) FnMut trait: Mutably borrows sum
    let mut sum = 0;
    let doubled: Vec<i32> = items.iter().map(|x| {
        sum += x;  // Borrows sum as &mut -> FnMut
        x * 2
    }).collect();
    println!("{:?}, sum={}", doubled, sum);

    // (c) Fn trait: Immutably borrows data (without move)
    let data = String::from("hello");
    let printer = || println!("{}", data);  // Borrows &data
    printer();
    println!("{}", data);  // data is still valid

    // (d) move is required: Transfer ownership of name to the closure
    fn make_greeter(name: String) -> impl Fn() {
        move || println!("Hello, {}!", name)
        // move transfers ownership of name
        // println! only uses &name, so it implements Fn
    }

    let greet = make_greeter(String::from("World"));
    greet();
    greet();  // Fn, so it can be called any number of times
}
```
</details>

---

## 10. FAQ (Frequently Asked Questions)

### Q1: Are closures and lambda expressions the same thing?

**A:** Strictly speaking, they are different concepts, but in practice they are often used interchangeably.

- **Lambda expression**: A syntax for defining anonymous functions. Such as `(x) => x * 2` or `lambda x: x * 2`.
- **Closure**: A function that has captured free variables. Independent of whether it is a lambda expression.

A named function can also be a closure, and a lambda expression that does not capture free variables is not a closure. However, since lambda expressions in many languages act as closures, the two are often confused.

```javascript
// A lambda expression but not a closure (no free variables)
const double = (x) => x * 2;

// Both a lambda expression and a closure (captures factor)
const factor = 3;
const multiply = (x) => x * factor;

// A named function but a closure (captures count)
function makeCounter() {
    let count = 0;
    function increment() {  // Named function
        return ++count;     // Captures count -> closure
    }
    return increment;
}
```

### Q2: What criteria should I use to decide between closures and classes?

**A:** The following criteria are useful for making the decision.

| Criterion | Closure is appropriate | Class is appropriate |
|-----------|----------------------|---------------------|
| Number of operations | 1-3 operations | 4 or more operations |
| State complexity | Simple (a few variables) | Complex (many fields) |
| Need for inheritance | Not needed | Needed |
| Testability | Easy if functionally pure | Requires mocks/stubs |
| Callbacks | Optimal | Overkill |
| Configuration injection | Partial application suffices | DI container is preferable |
| Team conventions | Functional style | OOP style |

```javascript
// Closure is appropriate: Generating a single callback
const createHandler = (eventType) => (event) => {
    console.log(`${eventType}: ${event.target.id}`);
};

// Class is appropriate: Complex state and many operations
class ShoppingCart {
    #items = [];
    #discountRate = 0;

    addItem(item) { /* ... */ }
    removeItem(id) { /* ... */ }
    applyDiscount(rate) { /* ... */ }
    calculateTotal() { /* ... */ }
    checkout() { /* ... */ }
    getItems() { /* ... */ }
    toJSON() { /* ... */ }
}
```

### Q3: How do closures affect garbage collection?

**A:** As long as a closure is alive, the variables it captured will not be eligible for GC. This is intentional behavior, but it can cause memory leaks if not handled carefully.

Key countermeasures:
1. **Capture only the needed values**: If only part of a huge object is needed, extract it in advance
2. **Cut references to closures that are no longer needed**: Remove event listeners, clear timers, etc.
3. **Use WeakRef / WeakMap**: When you want to avoid strong references (JavaScript, Python's weakref)
4. **Minimize scope**: Move the closure definition as far inside as possible to reduce the number of captured variables

### Q4: What is the relationship between async/await and closures?

**A:** `async` functions are internally implemented using a combination of closures and generators (or state machines). When suspending and resuming at `await`, the state of local variables is preserved in a closure-like manner.

```javascript
// async/await internally preserves variables via a closure-like mechanism
async function fetchAndProcess(url) {
    const response = await fetch(url);
    // ^ Suspended here
    // response is preserved in a closure-like manner

    const data = await response.json();
    // ^ Also suspended here
    // Both response and data are preserved

    return processData(data);
}
```

### Q5: When should I use move closures vs regular closures in Rust?

**A:** The `move` keyword forcibly transfers ownership of the variables a closure captures into the closure. Use `move` in the following cases:

1. **When the closure becomes a function's return value**: To prevent references to local variables from becoming invalid
2. **When sending the closure to another thread**: Because a `'static` lifetime is required
3. **When explicit ownership transfer is needed**: To make it clear that the variable will not be used in the original scope

```rust
use std::thread;

// move is required: Sending a closure to another thread
fn spawn_worker(data: Vec<i32>) -> thread::JoinHandle<i32> {
    thread::spawn(move || {
        // Without move, the reference to data could become invalid
        data.iter().sum()
    })
}
```

---

## 11. History and Theoretical Background of Closures

### 11.1 Historical Timeline

```
History of Closures:

  1936  Alonzo Church publishes lambda calculus
        -> Mathematical foundation for treating functions as values

  1958  Lisp is born (John McCarthy)
        -> The first functional programming language
        -> However, early Lisp used dynamic scope

  1964  Peter Landin proposes the SECD machine
        -> First appearance of the term "closure"
        -> Formalized the combination of function + environment

  1970  Scheme is born (Guy Steele, Gerald Sussman)
        -> Introduced lexical scope to Lisp
        -> Practical implementation of closures

  1973  ML is born (Robin Milner)
        -> A typed functional language adopting closures

  1987  Design of Haskell begins
        -> Combination of lazy evaluation and closures

  1995  JavaScript is born (Brendan Eich)
        -> Adopted closures under the influence of Scheme
        -> Closures gain widespread use in web development

  2004  Groovy (closures on the JVM)

  2007  C# 3.0 adds lambda expressions

  2010  Rust development begins (closures + ownership)

  2011  C++11 adds lambda expressions

  2014  Java 8 adds lambda expressions
        Swift is born (closures as a core feature)

  2015  ES6 (JavaScript) adds arrow functions
```

### 11.2 Relationship to Lambda Calculus

The theoretical foundation of closures lies in lambda calculus. In lambda calculus, all computation is expressed using only "function definition" and "function application."

```
Lambda Calculus Basics:

  Lambda abstraction:  lambda x.M     (a function with parameter x and body M)
  Function application:    (M N)      (apply M to N)
  Variable:                x          (variable reference)

  Example: Addition function
    lambda x.lambda y.(x + y)           Two-parameter function
    (lambda x.lambda y.(x + y)) 3       Partial application -> lambda y.(3 + y)
    ((lambda x.lambda y.(x + y)) 3) 5   Full application -> 8

  Correspondence with closures:
    lambda x.lambda y.(x + y)  ->  const add = (x) => (y) => x + y;
    Partial application        ->  const add3 = add(3);  // y => 3 + y
    Full application           ->  add3(5);              // 8
```

---


## FAQ

### Q1: What is the most important point when learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory, but by actually writing code and verifying its behavior.

### Q2: What common mistakes do beginners make?

Skipping the basics and jumping to advanced topics. We recommend thoroughly understanding the fundamental concepts explained in this guide before moving on to the next step.

### Q3: How is this used in practice?

Knowledge of this topic is frequently applied in everyday development work. It becomes especially important during code reviews and architecture design.

---

## 12. Summary

### 12.1 Comprehensive Comparison Table

| Language | Closure Capture | Mutable Capture | Characteristics | Primary Use Cases |
|----------|----------------|----------------|----------------|-------------------|
| JavaScript | Reference (automatic) | Possible | Most natural to use | Callbacks, modules |
| Python | Reference (automatic) | Requires nonlocal | Loop variable trap | Decorators, higher-order functions |
| Rust | Borrow/move | Possible with FnMut | Integrated with ownership, safest | Iterators, concurrency |
| Go | Reference (automatic) | Possible | Simple | Goroutines, middleware |
| Java | Value copy | Not possible (effectively final) | Restrictive | Stream API, callbacks |
| C++ | Value or reference (specified) | Possible with mutable | Finest-grained control | STL algorithms |
| Swift | Reference (automatic) | Possible | Integrated with ARC | Async processing, UI events |

### 12.2 Design Decision Flowchart

```
Closure Adoption Decision:

  Do you need stateful behavior?
    |
    +-- No -> A regular function is sufficient
    |
    +-- Yes -> How many operations?
              |
              +-- 1-2 -> Closures are optimal
              |           Example: Callbacks, filter conditions
              |
              +-- 3-5 -> Closures or a small class
              |           Follow team conventions
              |
              +-- 6+ -> A class is appropriate
                          State management is too complex

  Additional considerations:
    - Testability: If it can be written functionally pure, use closures
    - Reusability: If inheritance or interfaces are needed, use classes
    - Performance: Avoid unnecessary closure creation in hot paths
    - Team skills: If the team is accustomed to functional style, lean toward closures
```

### 12.3 Key Points Review

1. **Closure = Function + Environment**: A mechanism that retains bindings to free variables together with the function
2. **Lexical Scope**: Closures reference the scope at definition time (not at call time)
3. **Language Differences**: Capture methods (reference vs value vs ownership) vary significantly by language
4. **Practical Patterns**: Memoization, currying, module pattern, debounce/throttle
5. **Memory Impact**: Cut references to unnecessary closures; capture only the needed values
6. **Duality**: Closures and objects have essentially equivalent expressive power

---

## Guides to Read Next


---

## References

1. Abelson, H. & Sussman, G. J. *Structure and Interpretation of Computer Programs (SICP)*. Ch.3 "Modularity, Objects, and State," MIT Press, 1996. -- A classic explanation of state management with closures. Includes a detailed description of the environment model.
2. Klabnik, S. & Nichols, C. *The Rust Programming Language*. Ch.13 "Functional Language Features: Iterators and Closures," No Starch Press, 2023. -- On Rust's Fn/FnMut/FnOnce traits and their integration with the ownership system.
3. Crockford, D. *JavaScript: The Good Parts*. O'Reilly Media, 2008. -- Practical explanation of closures and the module pattern in JavaScript.
4. Landin, P. J. "The Mechanical Evaluation of Expressions." *The Computer Journal*, Vol.6, No.4, pp.308-320, 1964. -- The paper where the term "closure" was first used. Formalized the combination of environment and function in the context of the SECD machine.
5. Sussman, G. J. & Steele, G. L. "Scheme: An Interpreter for Extended Lambda Calculus." *MIT AI Memo 349*, 1975. -- The groundbreaking paper that introduced lexical scope to Lisp and put closures into practical use.
6. Friedman, D. P. & Wand, M. *Essentials of Programming Languages*. 3rd Edition, MIT Press, 2008. -- The place of closures in programming language semantics. Understanding through the construction of environment-passing interpreters.
