# First-Class Functions

> In languages where functions can be treated as "values," functions can be assigned to variables, passed as arguments, and returned as return values. This is the foundation of modern programming and the starting point of functional programming. The concept of "first-class," proposed by Christopher Strachey in 1967, has been adopted by virtually all major languages over more than half a century.

---

## What You Will Learn in This Chapter

- [ ] Understand the concept and historical significance of first-class functions
- [ ] Master the four ways to manipulate functions as values
- [ ] Understand callback patterns and higher-order function design
- [ ] Acquire techniques of function composition, partial application, and currying
- [ ] Compare first-class function support differences across languages
- [ ] Implement dispatch tables and the strategy pattern
- [ ] Identify anti-patterns and learn to avoid them appropriately


## Prerequisites

Before reading this guide, the following knowledge will help deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts

---

## 1. What Are First-Class Functions?

### 1.1 Definition and Historical Background

The term "First-Class" was first used systematically by British computer scientist Christopher Strachey in his 1967 lecture notes "Fundamental Concepts in Programming Languages." Strachey classified the status of "values" in programming languages as follows.

```
+----------------------------------------------------------------------+
|               Strachey's Classification of Values (1967)              |
+----------------------------------------------------------------------+
|                                                                        |
|  First-Class                                                           |
|  +----------------------------------------------------------+         |
|  | - Can be bound (assigned) to variables                    |         |
|  | - Can be passed as function arguments                     |         |
|  | - Can be returned as function return values                |         |
|  | - Can be stored in data structures (arrays, lists, etc.)  |         |
|  | - Can be dynamically generated at runtime                 |         |
|  | - Has its own identity                                    |         |
|  +----------------------------------------------------------+         |
|                                                                        |
|  Second-Class                                                          |
|  +----------------------------------------------------------+         |
|  | - Can be passed as function arguments                     |         |
|  | - May not be assignable to variables or returnable        |         |
|  +----------------------------------------------------------+         |
|                                                                        |
|  Third-Class                                                           |
|  +----------------------------------------------------------+         |
|  | - Exists only as a syntactic element of the language      |         |
|  | - Cannot even be passed as an argument                    |         |
|  +----------------------------------------------------------+         |
|                                                                        |
+----------------------------------------------------------------------+
```

In this classification, while integers, strings, and arrays are first-class in many languages, only a limited number of languages treated "functions" as first-class. LISP (1958) was the first practical language to treat functions as first-class, and its design philosophy has been inherited through Scheme, ML, and Haskell to modern languages including JavaScript, Python, Ruby, and more recently Rust, Go, Kotlin, and Swift.

### 1.2 The Four Properties of First-Class Functions

A first-class function means "a function that can be treated exactly the same as any other value." Specifically, the following four operations must be possible.

```
The Four Properties of First-Class Functions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [Property 1] Assignment to Variables
  +-------------------------------------+
  | const f = function(x) { return x; } |
  | val    <---- function value ------> |
  +-------------------------------------+

  [Property 2] Passing as Arguments
  +-------------------------------------+
  | apply(f, 42)                        |
  |       ^                             |
  |     Pass function as argument       |
  +-------------------------------------+

  [Property 3] Returning as Return Values
  +-------------------------------------+
  | function make() { return f; }       |
  |                        ^            |
  |                  Return a function  |
  +-------------------------------------+

  [Property 4] Storing in Data Structures
  +-------------------------------------+
  | const arr = [f1, f2, f3];           |
  | const obj = { op: f1 };             |
  |    Store functions in arrays/dicts  |
  +-------------------------------------+
```

### 1.3 Why First-Class Functions Matter

First-class functions are considered the foundation of modern programming for the following reasons.

1. **Enhanced abstraction**: Abstract processing patterns (iteration, transformation, selection) as functions, making them reusable
2. **Code conciseness**: Eliminate boilerplate code and express intent directly
3. **Flexible design**: Enable behavior to be swapped at runtime (strategy pattern, etc.)
4. **Affinity with concurrency**: Side-effect-free functions are thread-safe and suitable for concurrent processing
5. **Testability**: Testing at the function level becomes easy, and mock replacement is natural

---

## 2. Basic Operations: Learning the Four Properties Through Code Examples

### 2.1 Property 1 -- Assignment to Variables

By assigning functions to variables, you can give functions aliases or select different functions based on conditions.

```javascript
// ===== JavaScript =====

// Assign a function declaration to a variable (function expression)
const greet = function(name) {
    return `Hello, ${name}!`;
};

// Arrow function (ES6+)
const greetArrow = (name) => `Hello, ${name}!`;

// Assign an existing function to another variable
const sayHello = greet;
console.log(sayHello("Alice"));  // => "Hello, Alice!"

// Conditional function selection
const formatter = process.env.NODE_ENV === "production"
    ? (msg) => `[PROD] ${msg}`
    : (msg) => `[DEV] ${msg}`;

console.log(formatter("Server started"));
// Development: => "[DEV] Server started"
```

```python
# ===== Python =====

def square(x):
    """Returns the square of x"""
    return x ** 2

# Assign a function to a variable
f = square
print(f(5))      # => 25
print(f.__name__) # => "square" (retains original name)

# Anonymous function via lambda expression
double = lambda x: x * 2
print(double(7))  # => 14

# Conditional function selection
import os
log_fn = print if os.getenv("DEBUG") else lambda *args: None
log_fn("Debug message")  # Displays nothing if DEBUG env var is not set
```

```rust
// ===== Rust =====

fn square(x: i32) -> i32 {
    x * x
}

fn main() {
    // Assign as a function pointer
    let f: fn(i32) -> i32 = square;
    println!("{}", f(5));  // => 25

    // Assign a closure to a variable
    let double = |x: i32| -> i32 { x * 2 };
    println!("{}", double(7));  // => 14

    // Conditional function selection (with function pointers)
    let debug = true;
    let log_fn: fn(&str) = if debug {
        |msg| println!("[DEBUG] {}", msg)
    } else {
        |_msg| {}  // Do nothing
    };
    log_fn("Started");
}
```

### 2.2 Property 2 -- Passing as Arguments (Higher-Order Functions)

A function that receives a function as an argument is called a **Higher-Order Function**. This is one of the most powerful applications of first-class functions.

```javascript
// ===== JavaScript =====

// map: Apply a transformation function to each element
const numbers = [1, 2, 3, 4, 5];
const doubled = numbers.map(x => x * 2);
// => [2, 4, 6, 8, 10]

// filter: Extract only elements satisfying a predicate function
const evens = numbers.filter(x => x % 2 === 0);
// => [2, 4]

// reduce: Fold with an accumulator function
const sum = numbers.reduce((acc, x) => acc + x, 0);
// => 15

// Creating custom higher-order functions
function applyTwice(fn, value) {
    return fn(fn(value));
}

applyTwice(x => x * 2, 3);   // => 12  (3 -> 6 -> 12)
applyTwice(x => x + 10, 5);  // => 25  (5 -> 15 -> 25)

// Generic retry function
async function retry(fn, maxAttempts = 3, delay = 1000) {
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
            return await fn();
        } catch (err) {
            if (attempt === maxAttempts) throw err;
            console.log(`Attempt ${attempt} failed, retrying...`);
            await new Promise(r => setTimeout(r, delay));
        }
    }
}

// Usage
await retry(() => fetch("https://api.example.com/data"), 3, 2000);
```

```python
# ===== Python =====

# Built-in higher-order functions
numbers = [1, 2, 3, 4, 5]

# map: Apply a function to each element
squared = list(map(lambda x: x ** 2, numbers))
# => [1, 4, 9, 16, 25]

# filter: Extract elements satisfying a condition
evens = list(filter(lambda x: x % 2 == 0, numbers))
# => [2, 4]

# sorted: Sort with a key function
words = ["banana", "apple", "cherry", "date"]
sorted_by_length = sorted(words, key=len)
# => ["date", "apple", "banana", "cherry"]

# Custom higher-order function
def apply_to_all(fn, items):
    """Apply fn to all elements and return a new list"""
    return [fn(item) for item in items]

apply_to_all(str.upper, ["hello", "world"])
# => ["HELLO", "WORLD"]

# Decorator: A typical application of higher-order functions
import time
import functools

def timer(func):
    """Decorator that measures function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__}: {elapsed:.4f}s")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)
    return "done"

slow_function()  # => Displays "slow_function: 1.00xxs" then returns "done"
```

### 2.3 Property 3 -- Returning as Return Values (Function Factories)

A function that returns a function is called a **function factory** or **function generator**. This allows dynamic generation of pre-configured functions.

```javascript
// ===== JavaScript =====

// Multiplier function factory
function multiplier(factor) {
    return (x) => x * factor;
}

const double = multiplier(2);
const triple = multiplier(3);
const tenTimes = multiplier(10);

console.log(double(5));    // => 10
console.log(triple(5));    // => 15
console.log(tenTimes(5));  // => 50

// Validator factory
function createValidator(rules) {
    return (value) => {
        const errors = [];
        for (const rule of rules) {
            const error = rule(value);
            if (error) errors.push(error);
        }
        return { valid: errors.length === 0, errors };
    };
}

// Rule definitions
const required = (v) => v ? null : "This field is required";
const minLength = (n) => (v) => v && v.length >= n
    ? null : `Must be at least ${n} characters`;
const pattern = (re, msg) => (v) => re.test(v) ? null : msg;

// Generate validator
const validateEmail = createValidator([
    required,
    minLength(5),
    pattern(/^[^\s@]+@[^\s@]+\.[^\s@]+$/, "Please enter a valid email address"),
]);

console.log(validateEmail(""));
// => { valid: false, errors: ["This field is required", "Must be at least 5 characters", ...] }

console.log(validateEmail("user@example.com"));
// => { valid: true, errors: [] }
```

```python
# ===== Python =====

# Logger factory
def create_logger(prefix, level="INFO"):
    """Generate a log function with the specified prefix and level"""
    def logger(message):
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] [{prefix}] {message}")
    return logger

app_log = create_logger("APP")
db_log = create_logger("DB", level="DEBUG")

app_log("Server started")
# => [2026-03-06 10:30:00] [INFO] [APP] Server started

db_log("Query executed: SELECT * FROM users")
# => [2026-03-06 10:30:00] [DEBUG] [DB] Query executed: SELECT * FROM users
```

### 2.4 Property 4 -- Storing in Data Structures (Dispatch Tables)

The "dispatch table" pattern, which stores functions in dictionaries or arrays, is a powerful technique for replacing long if-else chains or switch statements.

```python
# ===== Python =====

# Command pattern via dispatch table
class Calculator:
    def __init__(self):
        self.operations = {
            "+": lambda a, b: a + b,
            "-": lambda a, b: a - b,
            "*": lambda a, b: a * b,
            "/": lambda a, b: a / b if b != 0 else float("inf"),
            "**": lambda a, b: a ** b,
            "%": lambda a, b: a % b,
        }
        self.history = []

    def calculate(self, a, op, b):
        if op not in self.operations:
            raise ValueError(f"Unsupported operator: {op}")
        result = self.operations[op](a, b)
        self.history.append(f"{a} {op} {b} = {result}")
        return result

    def add_operation(self, symbol, fn):
        """Dynamically add an operator"""
        self.operations[symbol] = fn

calc = Calculator()
print(calc.calculate(10, "+", 5))   # => 15
print(calc.calculate(2, "**", 10))  # => 1024

# Dynamic operator addition
calc.add_operation("avg", lambda a, b: (a + b) / 2)
print(calc.calculate(10, "avg", 20))  # => 15.0
```

```javascript
// ===== JavaScript =====

// HTTP method dispatch table
const handlers = {
    GET:    (req) => ({ status: 200, body: fetchResource(req.path) }),
    POST:   (req) => ({ status: 201, body: createResource(req.body) }),
    PUT:    (req) => ({ status: 200, body: updateResource(req.path, req.body) }),
    DELETE: (req) => ({ status: 204, body: null }),
};

function handleRequest(req) {
    const handler = handlers[req.method];
    if (!handler) {
        return { status: 405, body: "Method Not Allowed" };
    }
    return handler(req);
}

// Pipeline: Apply an array of functions sequentially
const pipeline = [
    (data) => ({ ...data, timestamp: Date.now() }),
    (data) => ({ ...data, id: crypto.randomUUID() }),
    (data) => ({ ...data, status: "processed" }),
];

function processThroughPipeline(data, steps) {
    return steps.reduce((acc, step) => step(acc), data);
}

const result = processThroughPipeline(
    { name: "Alice" },
    pipeline
);
// => { name: "Alice", timestamp: 1709..., id: "abc-...", status: "processed" }
```

---

## 3. Systematic Understanding of Higher-Order Functions

### 3.1 Classification of Higher-Order Functions

Higher-order functions are broadly divided into those that "receive functions" and those that "return functions." Many functions have both properties.

```
Classification System of Higher-Order Functions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                    Higher-Order Function
                             |
             +---------------+---------------+
             |                               |
    Receives Functions              Returns Functions
  (Consumer of Functions)        (Producer of Functions)
             |                               |
     +-------+--------+       +--------+--------+--------+
     |       |        |       |        |        |        |
   map/   filter/  reduce/  Factory   Currying  Decorator
  forEach find/some fold             /Partial
     |       |        |       |        |        |
  Transform Select  Aggregate Generate Transform Modify
  type     type    type      type     type      type

  Ex: .map() Ex: .filter() Ex: .reduce() Ex: multiplier() Ex: curry() Ex: @timer
  Transform  Narrow by    Aggregate    Dynamically  Receive args  Add processing
  each elem  condition    into 1 value generate a   one at a time before/after
                                       pre-set fn                 a function
```

### 3.2 Data Flow of Representative Higher-Order Functions

```
Data Flow of map
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: [1, 2, 3, 4, 5]
Function: x => x * x

  1 --> [x => x*x] -->  1
  2 --> [x => x*x] -->  4
  3 --> [x => x*x] -->  9
  4 --> [x => x*x] --> 16
  5 --> [x => x*x] --> 25

Output: [1, 4, 9, 16, 25]


Data Flow of filter
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: [1, 2, 3, 4, 5]
Predicate: x => x % 2 === 0

  1 --> [x%2===0] --> false --> (excluded)
  2 --> [x%2===0] --> true  --> 2
  3 --> [x%2===0] --> false --> (excluded)
  4 --> [x%2===0] --> true  --> 4
  5 --> [x%2===0] --> false --> (excluded)

Output: [2, 4]


Data Flow of reduce
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: [1, 2, 3, 4, 5]
Function: (acc, x) => acc + x
Initial value: 0

  acc=0, x=1 --> [acc+x] --> acc=1
  acc=1, x=2 --> [acc+x] --> acc=3
  acc=3, x=3 --> [acc+x] --> acc=6
  acc=6, x=4 --> [acc+x] --> acc=10
  acc=10,x=5 --> [acc+x] --> acc=15

Output: 15
```

### 3.3 Learning from Implementing map / filter / reduce

By implementing the higher-order functions provided by standard libraries yourself, you can deeply understand their mechanisms.

```javascript
// ===== JavaScript: Custom implementations of map / filter / reduce =====

// Implementation of map
function myMap(arr, fn) {
    const result = [];
    for (let i = 0; i < arr.length; i++) {
        result.push(fn(arr[i], i, arr));
    }
    return result;
}

// Implementation of filter
function myFilter(arr, predicate) {
    const result = [];
    for (let i = 0; i < arr.length; i++) {
        if (predicate(arr[i], i, arr)) {
            result.push(arr[i]);
        }
    }
    return result;
}

// Implementation of reduce
function myReduce(arr, fn, initial) {
    let acc = initial;
    let startIndex = 0;
    if (acc === undefined) {
        acc = arr[0];
        startIndex = 1;
    }
    for (let i = startIndex; i < arr.length; i++) {
        acc = fn(acc, arr[i], i, arr);
    }
    return acc;
}

// Verification
const nums = [1, 2, 3, 4, 5];
console.log(myMap(nums, x => x * 2));           // => [2, 4, 6, 8, 10]
console.log(myFilter(nums, x => x > 3));        // => [4, 5]
console.log(myReduce(nums, (a, b) => a + b, 0)); // => 15

// Re-implementing map and filter using reduce
function mapWithReduce(arr, fn) {
    return arr.reduce((acc, x, i) => {
        acc.push(fn(x, i, arr));
        return acc;
    }, []);
}

function filterWithReduce(arr, pred) {
    return arr.reduce((acc, x, i) => {
        if (pred(x, i, arr)) acc.push(x);
        return acc;
    }, []);
}
```

---

## 4. Deep Dive into Callback Patterns

### 4.1 Synchronous Callbacks

```javascript
// ===== JavaScript: Diverse uses of synchronous callbacks =====

// Event listener (browser environment)
document.addEventListener("click", (event) => {
    console.log(`Click position: (${event.clientX}, ${event.clientY})`);
});

// Array method chaining
const users = [
    { name: "Alice", age: 30, role: "admin" },
    { name: "Bob", age: 25, role: "user" },
    { name: "Charlie", age: 35, role: "admin" },
    { name: "Diana", age: 28, role: "user" },
];

// Get admin names sorted by age
const adminNames = users
    .filter(u => u.role === "admin")
    .sort((a, b) => a.age - b.age)
    .map(u => u.name);
// => ["Alice", "Charlie"]

// forEach: Callback for side effects
users.forEach(u => {
    console.log(`${u.name} (age ${u.age}) - ${u.role}`);
});

// find / findIndex: First element matching the condition
const firstAdmin = users.find(u => u.role === "admin");
// => { name: "Alice", age: 30, role: "admin" }

// every / some: Whether all/some elements satisfy the condition
const allAdults = users.every(u => u.age >= 18);  // => true
const hasAdmin = users.some(u => u.role === "admin"); // => true
```

### 4.2 Asynchronous Callbacks

```javascript
// ===== JavaScript: Evolution of asynchronous callbacks =====

// 1. Classic callback style (Node.js)
const fs = require("fs");

fs.readFile("/path/to/file.txt", "utf8", (err, data) => {
    if (err) {
        console.error("Read error:", err);
        return;
    }
    console.log("File contents:", data);
});

// 2. Promise (ES6+)
function readFileAsync(path) {
    return new Promise((resolve, reject) => {
        fs.readFile(path, "utf8", (err, data) => {
            if (err) reject(err);
            else resolve(data);
        });
    });
}

readFileAsync("/path/to/file.txt")
    .then(data => console.log(data))
    .catch(err => console.error(err));

// 3. async/await (ES2017+)
async function processFile() {
    try {
        const data = await readFileAsync("/path/to/file.txt");
        const processed = data.toUpperCase();
        console.log(processed);
    } catch (err) {
        console.error("Processing error:", err);
    }
}
```

### 4.3 Callback Hell and Solutions

```
Visualization of Callback Hell
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  step1(input, (err, result1) => {
      if (err) handleError(err);
      step2(result1, (err, result2) => {
          if (err) handleError(err);
          step3(result2, (err, result3) => {
              if (err) handleError(err);
              step4(result3, (err, result4) => {    <- Deeply nested!
                  if (err) handleError(err);
                  // ...nesting continues
              });
          });
      });
  });

  Evolution of solutions:
  +----------------+    +----------------+    +----------------+
  |   Callbacks    |    |    Promise     |    |  async/await   |
  |  (pre-ES5)    | -> |  .then() chain | -> | Synchronous-   |
  |               |    |  (ES6+)        |    | style syntax   |
  |  Nesting hell |    |  Flat chains   |    | Most readable  |
  |  Scattered    |    |               |    | try/catch      |
  |  error handling|    |               |    |                |
  +----------------+    +----------------+    +----------------+
```

---

## 5. Strategy Pattern and Functions

### 5.1 Traditional OOP Implementation vs First-Class Functions

The strategy pattern is one of the GoF design patterns for dynamically switching algorithms. In OOP, a dedicated class hierarchy is required, but in languages with first-class functions, it can be achieved with a single function.

```typescript
// ===== TypeScript: OOP Strategy vs Functional Strategy =====

// --- OOP approach (Java-style) ---
interface SortStrategy<T> {
    compare(a: T, b: T): number;
}

class NameSortStrategy implements SortStrategy<User> {
    compare(a: User, b: User): number {
        return a.name.localeCompare(b.name);
    }
}

class AgeSortStrategy implements SortStrategy<User> {
    compare(a: User, b: User): number {
        return a.age - b.age;
    }
}

class UserSorter {
    private strategy: SortStrategy<User>;

    constructor(strategy: SortStrategy<User>) {
        this.strategy = strategy;
    }

    setStrategy(strategy: SortStrategy<User>) {
        this.strategy = strategy;
    }

    sort(users: User[]): User[] {
        return [...users].sort((a, b) => this.strategy.compare(a, b));
    }
}

// --- Functional approach (leveraging first-class functions) ---
type Comparator<T> = (a: T, b: T) => number;

const byName: Comparator<User> = (a, b) => a.name.localeCompare(b.name);
const byAge: Comparator<User> = (a, b) => a.age - b.age;
const byNameDesc: Comparator<User> = (a, b) => b.name.localeCompare(a.name);
const byAgeDesc: Comparator<User> = (a, b) => b.age - a.age;

// Composition: Combine multiple sort criteria
function composeComparators<T>(...comparators: Comparator<T>[]): Comparator<T> {
    return (a, b) => {
        for (const cmp of comparators) {
            const result = cmp(a, b);
            if (result !== 0) return result;
        }
        return 0;
    };
}

// Sort by role first, then age
const byRoleThenAge = composeComparators(
    (a, b) => a.role.localeCompare(b.role),
    byAge
);

const sorted = [...users].sort(byRoleThenAge);
```

### 5.2 Comparison: OOP vs Functional Strategy

| Aspect | OOP (Class-based) | Functional Approach |
|--------|-------------------|---------------------|
| Code volume | Requires interface + implementation classes | A single function suffices |
| Adding new strategies | Requires creating a new class | Just define a new function |
| State retention | Via instance variables | Via closures |
| Type safety | Enforced by interfaces | Expressed via type aliases |
| Testing | May require mock objects | Functions can be tested individually |
| Composition | Requires Composite pattern | Naturally achieved via function composition |
| Serialization | Requires class serialization | Functions cannot be serialized |
| Debugging | Easy to identify by class name | Anonymous functions are harder to trace |
| Applicable scenarios | Complex state/lifecycle management | Simple behavior swapping |

---

## 6. Function Composition and Partial Application

### 6.1 Function Composition

Combining two functions f and g to create a new function f . g where the output of g becomes the input of f is called function composition. Mathematically, it is written as (f . g)(x) = f(g(x)).

```
Concept Diagram of Function Composition
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  compose(f, g) = x => f(g(x))

                 g            f
  Input x --> [double] --> [addOne] --> Output
          x=5    10          11

  pipe(g, f) = x => f(g(x))   * Reads left to right

                 g            f
  Input x --> [double] --> [addOne] --> Output
          x=5    10          11

  compose: Mathematical notation (right to left)
  pipe:    Programming notation (left to right)
```

```typescript
// ===== TypeScript: Function Composition Implementation =====

// compose: Apply right to left (mathematical notation)
function compose<A, B, C>(
    f: (b: B) => C,
    g: (a: A) => B
): (a: A) => C {
    return (a: A) => f(g(a));
}

// pipe: Apply left to right (programming notation)
function pipe<T>(...fns: Array<(arg: T) => T>): (arg: T) => T {
    return (arg: T) => fns.reduce((acc, fn) => fn(acc), arg);
}

// Usage
const double = (x: number) => x * 2;
const addOne = (x: number) => x + 1;
const square = (x: number) => x * x;

const doubleAndAddOne = compose(addOne, double);
console.log(doubleAndAddOne(5));  // => 11

const transform = pipe(double, addOne, square);
console.log(transform(3));  // => 3 -> 6 -> 7 -> 49
```

```python
# ===== Python: Function Composition =====

from functools import reduce

def compose(*fns):
    """Right-to-left function composition"""
    def composed(x):
        result = x
        for fn in reversed(fns):
            result = fn(result)
        return result
    return composed

def pipe(*fns):
    """Left-to-right function composition"""
    def piped(x):
        result = x
        for fn in fns:
            result = fn(result)
        return result
    return piped

# Usage
double = lambda x: x * 2
add_one = lambda x: x + 1
square = lambda x: x ** 2

transform = pipe(double, add_one, square)
print(transform(3))  # => 49  (3 -> 6 -> 7 -> 49)

# String processing pipeline
normalize = pipe(
    str.strip,
    str.lower,
    lambda s: s.replace("  ", " "),
)
print(normalize("  Hello   World  "))  # => "hello world"
```

### 6.2 Partial Application

```python
# ===== Python: functools.partial =====

from functools import partial

def power(base, exponent):
    """Calculate base raised to the exponent"""
    return base ** exponent

# Generate new functions via partial application
square = partial(power, exponent=2)
cube = partial(power, exponent=3)

print(square(5))  # => 25
print(cube(3))    # => 27

# Partial application of an HTTP client
import urllib.request

def fetch(method, url, headers=None, body=None):
    """Generic HTTP request function"""
    req = urllib.request.Request(url, method=method, headers=headers or {})
    if body:
        req.data = body.encode()
    return urllib.request.urlopen(req)

# Generate functions with fixed methods
get = partial(fetch, "GET")
post = partial(fetch, "POST")
put = partial(fetch, "PUT")

# Usage: get("https://api.example.com/users")
```

### 6.3 Currying

Currying is the technique of transforming a function that takes n arguments into a chain of functions each taking 1 argument. Unlike partial application, currying always applies one argument at a time.

```javascript
// ===== JavaScript: Currying =====

// Manual currying
function add(a) {
    return function(b) {
        return a + b;
    };
}

const add5 = add(5);
console.log(add5(3));  // => 8

// Generic currying function
function curry(fn) {
    return function curried(...args) {
        if (args.length >= fn.length) {
            return fn.apply(this, args);
        }
        return function(...moreArgs) {
            return curried.apply(this, args.concat(moreArgs));
        };
    };
}

// Usage
const curriedAdd = curry((a, b, c) => a + b + c);
console.log(curriedAdd(1)(2)(3));    // => 6
console.log(curriedAdd(1, 2)(3));    // => 6
console.log(curriedAdd(1)(2, 3));    // => 6
console.log(curriedAdd(1, 2, 3));    // => 6

// Practical example: Curried log function
const log = curry((level, module, message) => {
    console.log(`[${level}] [${module}] ${message}`);
});

const errorLog = log("ERROR");
const appError = errorLog("APP");
const dbError = errorLog("DB");

appError("Connection timeout");
// => [ERROR] [APP] Connection timeout

dbError("Query failed");
// => [ERROR] [DB] Query failed
```

---

## 7. First-Class Function Support Comparison Across Languages

### 7.1 Comprehensive Comparison Table

A comparison of how well each language supports "first-class functions" across key aspects.

| Feature / Language | JavaScript | Python | Rust | Go | Java | C | Haskell |
|-------------------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Assignment to variables | Yes | Yes | Yes | Yes | Limited | Pointer only | Yes |
| Passing as arguments | Yes | Yes | Yes | Yes | SAM types | Pointer only | Yes |
| Returning as return values | Yes | Yes | Yes | Yes | SAM types | Pointer only | Yes |
| Storing in data structures | Yes | Yes | Yes | Yes | Limited | Pointer only | Yes |
| Anonymous functions (lambdas) | Yes | Expression only | Yes | Yes | Expression only | No | Yes |
| Closures | Yes | Yes | 3 types | Yes | Limited | No | Yes |
| Partial application | Manual/library | functools | Manual | Manual | Manual | No | Natural |
| Currying | Manual/library | Manual | Manual | Manual | No | No | Automatic |
| Function composition operator | No | No | No | No | No | No | Yes (.) |
| Type inference | Dynamic typing | Dynamic typing | Yes | Yes | Limited | No | Yes |
| Generic higher-order functions | Yes (dynamic) | Yes (dynamic) | Yes | Yes | Yes | No | Yes |

### 7.2 Detailed Characteristics by Language

```
First-Class Function Support Characteristics by Language
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  JavaScript (ES6+)
  +------------------------------------------------------------+
  | - Functions are a subtype of Object (typeof fn === "function")|
  | - Three forms: function declaration, function expression,   |
  |   arrow function                                            |
  | - Arrow functions do not bind their own this (lexical this) |
  | - All object methods are also function values               |
  | - Async: async/await can also be treated as first-class     |
  +------------------------------------------------------------+

  Python
  +------------------------------------------------------------+
  | - Functions are objects (type(fn) -> <class 'function'>)    |
  | - Two forms: def statement (multi-line) and lambda          |
  |   expression (single expression only)                       |
  | - Decorators (@) are syntactic sugar for higher-order fns   |
  | - functools module provides partial, reduce, lru_cache, etc.|
  | - Introspection: fn.__name__, fn.__doc__, fn.__code__       |
  |   are accessible                                            |
  +------------------------------------------------------------+

  Rust
  +------------------------------------------------------------+
  | - Two systems: function pointers (fn) and closures          |
  |   (|args| body)                                             |
  | - Closures classified by 3 traits:                          |
  |   - Fn:     Captures environment by immutable reference     |
  |   - FnMut:  Captures environment by mutable reference       |
  |   - FnOnce: Captures environment by moving ownership        |
  | - Powerful type inference, but dyn/impl may be needed for   |
  |   return types                                              |
  | - Zero-cost abstraction: Closures optimized at compile time |
  +------------------------------------------------------------+

  Go
  +------------------------------------------------------------+
  | - Functions are first-class values (func literals create    |
  |   anonymous functions)                                      |
  | - Closures capture by reference (caution: loop variable trap)|
  | - Generics (Go 1.18+) enable type-safe higher-order fns    |
  | - Method values: obj.Method can be extracted as a fn value  |
  +------------------------------------------------------------+

  Java
  +------------------------------------------------------------+
  | - Lambda expressions are treated as instances of SAM        |
  |   (Single Abstract Method) interfaces                       |
  | - Functional interfaces: Function<T,R>, Predicate<T>,      |
  |   Consumer<T>, etc.                                         |
  | - Method references: ClassName::method extracts as fn value |
  | - Closures can only capture "effectively final" variables   |
  +------------------------------------------------------------+

  Haskell
  +------------------------------------------------------------+
  | - All functions are automatically curried                   |
  | - Built-in function composition operator (.):               |
  |   (f . g) x = f (g x)                                      |
  | - Partial application is most natural                       |
  |   (add 3 creates "a function that adds 3")                  |
  | - Type classes enable naturally polymorphic higher-order fns|
  | - Lazy evaluation enables higher-order fns on infinite lists|
  +------------------------------------------------------------+
```

### 7.3 Detailed Explanation of Rust's Three Closure Types

Rust closures are integrated with the ownership system and classified by three traits. This is a unique feature not found in other languages.

```rust
// ===== Rust: Three Closure Traits =====

fn main() {
    // --- Fn: Captures environment by immutable reference ---
    let name = String::from("Alice");
    let greet = || println!("Hello, {}!", name);  // Captures name as &name
    greet();  // Can be called multiple times
    greet();  // Reusable
    println!("{}", name);  // name is still usable

    // --- FnMut: Captures environment by mutable reference ---
    let mut count = 0;
    let mut increment = || {
        count += 1;  // Captures count as &mut count
        println!("count = {}", count);
    };
    increment();  // count = 1
    increment();  // count = 2

    // --- FnOnce: Captures environment by moving ownership ---
    let data = vec![1, 2, 3];
    let consume = move || {
        println!("data: {:?}", data);  // Moves ownership of data
        drop(data);  // Consumes data
    };
    consume();  // Can only be called once
    // consume();  // Compile error: already consumed
    // println!("{:?}", data);  // Compile error: data has been moved

    // --- Usage in higher-order functions ---
    fn apply_fn<F: Fn()>(f: F) {
        f(); f();  // Can call multiple times
    }

    fn apply_fn_mut<F: FnMut()>(mut f: F) {
        f(); f();  // Can call multiple times while mutating state
    }

    fn apply_fn_once<F: FnOnce()>(f: F) {
        f();  // Can only call once
    }
}
```

```
Rust Closure Trait Containment Relationship
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  +-------------------------------------+
  |           FnOnce                    |  Least restrictive (all closures implement)
  |  +------------------------------+  |
  |  |         FnMut                |  |  Subtrait of FnOnce
  |  |  +-----------------------+   |  |
  |  |  |        Fn             |   |  |  Most restrictive
  |  |  |                       |   |  |
  |  |  |  Immutable ref capture|   |  |
  |  |  +-----------------------+   |  |
  |  |  Mutable ref capture        |  |
  |  +------------------------------+  |
  |  Ownership move capture            |
  +-------------------------------------+

  Fn < FnMut < FnOnce

  - A closure implementing Fn also implements FnMut and FnOnce
  - A closure implementing FnMut also implements FnOnce
  - Function pointers fn implement Fn, FnMut, and FnOnce
```

---

## 8. Practical Design Patterns

### 8.1 Middleware Pattern

The middleware pattern, widely used in web frameworks (Express.js, Koa, axum, etc.), is a typical application of first-class functions.

```javascript
// ===== JavaScript: Middleware Pattern =====

// Simple middleware system
class Pipeline {
    constructor() {
        this.middlewares = [];
    }

    use(middleware) {
        this.middlewares.push(middleware);
        return this;  // Method chaining
    }

    async execute(context) {
        let index = 0;
        const next = async () => {
            if (index < this.middlewares.length) {
                const middleware = this.middlewares[index++];
                await middleware(context, next);
            }
        };
        await next();
        return context;
    }
}

// Define middlewares (each middleware is a function)
const logger = async (ctx, next) => {
    const start = Date.now();
    console.log(`-> ${ctx.method} ${ctx.path}`);
    await next();
    console.log(`<- ${ctx.method} ${ctx.path} (${Date.now() - start}ms)`);
};

const auth = async (ctx, next) => {
    if (!ctx.headers.authorization) {
        ctx.status = 401;
        ctx.body = "Unauthorized";
        return;  // Not calling next() -> skip subsequent middleware
    }
    ctx.user = verifyToken(ctx.headers.authorization);
    await next();
};

const handler = async (ctx, next) => {
    ctx.status = 200;
    ctx.body = { message: "Hello", user: ctx.user };
    await next();
};

// Assemble the pipeline
const app = new Pipeline();
app.use(logger).use(auth).use(handler);

// Execute
await app.execute({
    method: "GET",
    path: "/api/users",
    headers: { authorization: "Bearer token123" },
});
```

### 8.2 Event Emitter Pattern

```typescript
// ===== TypeScript: Type-safe Event Emitter =====

type EventMap = {
    "user:login":  { userId: string; timestamp: number };
    "user:logout": { userId: string; reason: string };
    "error":       { code: number; message: string };
};

class TypedEventEmitter<T extends Record<string, unknown>> {
    private listeners = new Map<keyof T, Set<(data: any) => void>>();

    on<K extends keyof T>(event: K, handler: (data: T[K]) => void): () => void {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, new Set());
        }
        this.listeners.get(event)!.add(handler);

        // Return an unsubscribe function (also an application of first-class functions)
        return () => {
            this.listeners.get(event)?.delete(handler);
        };
    }

    emit<K extends keyof T>(event: K, data: T[K]): void {
        this.listeners.get(event)?.forEach(handler => handler(data));
    }
}

// Usage
const emitter = new TypedEventEmitter<EventMap>();

const unsubscribe = emitter.on("user:login", (data) => {
    console.log(`User ${data.userId} logged in (${data.timestamp})`);
});

emitter.emit("user:login", {
    userId: "user-123",
    timestamp: Date.now(),
});

// Unsubscribe when no longer needed
unsubscribe();
```

### 8.3 Memoization Pattern

```javascript
// ===== JavaScript: Generic Memoization Function =====

function memoize(fn, options = {}) {
    const cache = new Map();
    const { maxSize = 1000, keyFn = JSON.stringify } = options;

    function memoized(...args) {
        const key = keyFn(args);

        if (cache.has(key)) {
            return cache.get(key);
        }

        const result = fn.apply(this, args);

        // Cache size limit
        if (cache.size >= maxSize) {
            const firstKey = cache.keys().next().value;
            cache.delete(firstKey);
        }

        cache.set(key, result);
        return result;
    }

    // Add cache management methods
    memoized.cache = cache;
    memoized.clear = () => cache.clear();

    return memoized;
}

// Usage: Fibonacci sequence
const fib = memoize((n) => {
    if (n <= 1) return n;
    return fib(n - 1) + fib(n - 2);
});

console.log(fib(50));  // => 12586269025 (computed quickly)

// Usage: API response cache
const fetchUser = memoize(
    async (id) => {
        const res = await fetch(`/api/users/${id}`);
        return res.json();
    },
    { maxSize: 100 }
);
```

```python
# ===== Python: lru_cache Decorator =====

from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci(n):
    """Memoized Fibonacci sequence"""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(100))
# => 354224848179261915075 (computed quickly)

# Cache statistics
print(fibonacci.cache_info())
# => CacheInfo(hits=98, misses=101, maxsize=128, currsize=101)

# Clear cache
fibonacci.cache_clear()
```

---

## 9. Anti-Patterns and Caveats

### 9.1 Anti-Pattern 1: Over-Abstraction (Abstraction Astronaut)

Becoming enamored with the power of first-class functions can lead to excessive abstraction.

```javascript
// ===== Anti-Pattern: Over-Abstraction =====

// BAD: Unnecessary function wrapping
const add = (a) => (b) => a + b;
const multiply = (a) => (b) => a * b;
const compose = (f) => (g) => (x) => f(g(x));
const pipe = (...fns) => (x) => fns.reduce((a, f) => f(a), x);

// A simple calculation with high cognitive overhead
const result = pipe(
    add(1),
    multiply(2),
    compose(add(3))(multiply(4)),
)(5);
// Not immediately clear what this does...

// GOOD: Appropriate level of abstraction
function calculatePrice(basePrice, taxRate, discount) {
    const afterDiscount = basePrice * (1 - discount);
    const withTax = afterDiscount * (1 + taxRate);
    return Math.round(withTax);
}

calculatePrice(1000, 0.1, 0.2);  // => 880
// Intent is clear and readable
```

**Lesson**: Consider abstraction only when "a repeating pattern appears 3 or more times" (Rule of Three). Abstracting patterns used only 1-2 times actually harms readability.

### 9.2 Anti-Pattern 2: Loss of `this` Reference

In JavaScript, the most common bug when passing functions as values is the loss of the `this` reference.

```javascript
// ===== Anti-Pattern: Loss of this =====

class UserService {
    constructor() {
        this.users = ["Alice", "Bob"];
    }

    getUsers() {
        return this.users;
    }
}

const service = new UserService();

// BAD: Assigning a method to a variable loses this
const getUsers = service.getUsers;
// getUsers();  // TypeError: Cannot read property 'users' of undefined

// BAD: Passing as a callback also loses this
// setTimeout(service.getUsers, 1000);  // Same error

// --- Solutions ---

// Solution 1: Bind this with bind
const getUsersBound = service.getUsers.bind(service);
getUsersBound();  // => ["Alice", "Bob"]

// Solution 2: Wrap with an arrow function
setTimeout(() => service.getUsers(), 1000);

// Solution 3: Use arrow functions as class fields (recommended)
class UserServiceFixed {
    users = ["Alice", "Bob"];

    // Arrow function class field -> this is fixed
    getUsers = () => {
        return this.users;
    };
}

const fixed = new UserServiceFixed();
const fn = fixed.getUsers;
fn();  // => ["Alice", "Bob"] (works correctly)
```

### 9.3 Anti-Pattern 3: Closure Capture of Loop Variables

```javascript
// ===== Anti-Pattern: Loop Variable Trap =====

// BAD: Closures in a loop using var
const functions = [];
for (var i = 0; i < 5; i++) {
    functions.push(() => i);
}
console.log(functions.map(f => f()));
// => [5, 5, 5, 5, 5]  All 5! (var has no block scope)

// GOOD: Use let (block scope)
const functions2 = [];
for (let i = 0; i < 5; i++) {
    functions2.push(() => i);
}
console.log(functions2.map(f => f()));
// => [0, 1, 2, 3, 4]  As expected

// GOOD: IIFE (Immediately Invoked Function Expression) to capture values
const functions3 = [];
for (var i = 0; i < 5; i++) {
    functions3.push(((captured) => () => captured)(i));
}

// BEST: Functional approach
const functions4 = Array.from({ length: 5 }, (_, i) => () => i);
console.log(functions4.map(f => f()));
// => [0, 1, 2, 3, 4]
```

```go
// ===== Go: Loop Variable Trap (Go 1.21 and earlier) =====

package main

import "fmt"

func main() {
    // BAD (Go 1.21 and earlier): Loop variable is shared across iterations
    fns := make([]func(), 5)
    for i := 0; i < 5; i++ {
        fns[i] = func() { fmt.Println(i) }
    }
    for _, f := range fns {
        f()  // Go 1.21 and earlier: all 5. Go 1.22+: 0,1,2,3,4
    }

    // GOOD: Copy to local variable (workaround for Go 1.21 and earlier)
    fns2 := make([]func(), 5)
    for i := 0; i < 5; i++ {
        i := i  // Shadow to capture value
        fns2[i] = func() { fmt.Println(i) }
    }
}
```

### 9.4 Anti-Pattern 4: Performance Traps

```javascript
// ===== Anti-Pattern: Unnecessary Closure Generation =====

// BAD: New function object created on every render
function TodoList({ todos, onDelete }) {
    return todos.map(todo => (
        // A new arrow function is created each time,
        // invalidating child component memoization
        <TodoItem
            key={todo.id}
            todo={todo}
            onDelete={() => onDelete(todo.id)}  // New function each time
        />
    ));
}

// GOOD: Use useCallback or method references
function TodoListOptimized({ todos, onDelete }) {
    const handleDelete = useCallback((id) => {
        onDelete(id);
    }, [onDelete]);

    return todos.map(todo => (
        <TodoItem
            key={todo.id}
            todo={todo}
            onDelete={handleDelete}
            todoId={todo.id}
        />
    ));
}
```

---

## 10. Exercises (3 Levels)

### 10.1 Beginner Exercises

**Exercise B-1: Manipulating Functions as Values**

Implement the following in JavaScript.

```
Requirements:
1. Define a function applyOperation(a, b, operation)
   - a, b are numbers, operation is a two-argument function
   - Return the result of operation(a, b)
2. Define four functions: add, subtract, multiply, divide
3. Calculate the following using applyOperation:
   - 10 + 5 = 15
   - 10 - 5 = 5
   - 10 * 5 = 50
   - 10 / 5 = 2
```

```javascript
// Solution

function applyOperation(a, b, operation) {
    return operation(a, b);
}

const add = (a, b) => a + b;
const subtract = (a, b) => a - b;
const multiply = (a, b) => a * b;
const divide = (a, b) => {
    if (b === 0) throw new Error("Cannot divide by zero");
    return a / b;
};

console.log(applyOperation(10, 5, add));       // => 15
console.log(applyOperation(10, 5, subtract));  // => 5
console.log(applyOperation(10, 5, multiply));  // => 50
console.log(applyOperation(10, 5, divide));    // => 2
```

**Exercise B-2: Implementing Filter Functions**

```
Requirements:
Implement the following in Python.
1. From the list [1, -2, 3, -4, 5, -6, 7, -8, 9, -10]:
   a. Extract only positive numbers
   b. Extract only even numbers
   c. Extract only those with absolute value >= 5
2. Accomplish each in one line using filter() and lambda
```

```python
# Solution

numbers = [1, -2, 3, -4, 5, -6, 7, -8, 9, -10]

positives = list(filter(lambda x: x > 0, numbers))
# => [1, 3, 5, 7, 9]

evens = list(filter(lambda x: x % 2 == 0, numbers))
# => [-2, -4, -6, -8, -10]

abs_gte_5 = list(filter(lambda x: abs(x) >= 5, numbers))
# => [5, -6, 7, -8, 9, -10]
```

**Exercise B-3: Dispatch Table**

```
Requirements:
Create a simple string transformation dispatch table in JavaScript.
- "upper": Convert to uppercase
- "lower": Convert to lowercase
- "reverse": Reverse the string
- "length": Return the character count as a string
- Return an error message for unknown commands
```

```javascript
// Solution

const transforms = {
    upper:   (s) => s.toUpperCase(),
    lower:   (s) => s.toLowerCase(),
    reverse: (s) => s.split("").reverse().join(""),
    length:  (s) => `${s.length} characters`,
};

function applyTransform(command, text) {
    const fn = transforms[command];
    if (!fn) {
        return `Error: Unknown command "${command}"`;
    }
    return fn(text);
}

console.log(applyTransform("upper", "hello"));     // => "HELLO"
console.log(applyTransform("reverse", "hello"));   // => "olleh"
console.log(applyTransform("length", "hello"));    // => "5 characters"
console.log(applyTransform("unknown", "hello"));   // => "Error: Unknown command..."
```

### 10.2 Intermediate Exercises

**Exercise I-1: Implementing a Function Composition Pipeline**

```
Requirements:
Implement the following in TypeScript.
1. Implement a pipe function (compose any number of functions left to right)
2. Build the following data processing pipeline:
   - Input: Array of strings
   - Step 1: Convert all to lowercase
   - Step 2: Remove words of 3 characters or less
   - Step 3: Remove duplicates
   - Step 4: Sort alphabetically
   - Step 5: Join with commas
```

```typescript
// Solution

// pipe implementation (simplified version as type-safe version is complex)
function pipe<T>(...fns: Array<(arg: T) => T>): (arg: T) => T {
    return (arg: T) => fns.reduce((acc, fn) => fn(acc), arg);
}

// Define each step as a function
const toLowerAll = (words: string[]) => words.map(w => w.toLowerCase());
const filterShort = (words: string[]) => words.filter(w => w.length > 3);
const unique = (words: string[]) => [...new Set(words)];
const sortAlpha = (words: string[]) => [...words].sort();
const joinComma = (words: string[]) => words.join(", ");

// Build pipeline (apply step by step for type compatibility)
function processWords(input: string[]): string {
    const step1 = toLowerAll(input);
    const step2 = filterShort(step1);
    const step3 = unique(step2);
    const step4 = sortAlpha(step3);
    return joinComma(step4);
}

// Test
const input = ["Hello", "WORLD", "the", "HELLO", "JavaScript", "is", "Great", "hello"];
console.log(processWords(input));
// => "great, hello, javascript, world"
```

**Exercise I-2: Implementing a Generic Retry Function**

```
Requirements:
Implement a retry function in JavaScript that satisfies the following.
1. Retry a failed function a specified number of times
2. Use exponential backoff for retry intervals (1s, 2s, 4s...)
3. Throw the last error when max retries exceeded
4. Log output on each retry
5. Return the result on success
```

```javascript
// Solution

async function retry(fn, options = {}) {
    const {
        maxAttempts = 3,
        baseDelay = 1000,
        backoffFactor = 2,
        onRetry = (attempt, err) => {
            console.log(`Retry attempt ${attempt}: ${err.message}`);
        },
    } = options;

    let lastError;

    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
            return await fn(attempt);
        } catch (err) {
            lastError = err;
            if (attempt < maxAttempts) {
                const delay = baseDelay * Math.pow(backoffFactor, attempt - 1);
                onRetry(attempt, err);
                await new Promise(r => setTimeout(r, delay));
            }
        }
    }

    throw new Error(
        `Failed after ${maxAttempts} retries: ${lastError.message}`
    );
}

// Test: Simulation that succeeds on the 3rd attempt
let callCount = 0;
const result = await retry(
    async (attempt) => {
        callCount++;
        if (callCount < 3) {
            throw new Error(`Connection error (attempt ${callCount})`);
        }
        return { data: "Success!" };
    },
    { maxAttempts: 5, baseDelay: 100 }
);
console.log(result);  // => { data: "Success!" }
```

**Exercise I-3: Implementing the Decorator Pattern**

```
Requirements:
Implement the following decorators in Python.
1. @validate_args: Verify that all arguments are not None
2. @log_calls: Log arguments and return values on function calls
3. Stack both decorators together
```

```python
# Solution

import functools

def validate_args(func):
    """Decorator that verifies all arguments are not None"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for i, arg in enumerate(args):
            if arg is None:
                raise ValueError(
                    f"{func.__name__}: Argument {i} is None"
                )
        for key, val in kwargs.items():
            if val is None:
                raise ValueError(
                    f"{func.__name__}: Keyword argument '{key}' is None"
                )
        return func(*args, **kwargs)
    return wrapper


def log_calls(func):
    """Decorator that logs function calls"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        print(f"Call: {func.__name__}({signature})")
        result = func(*args, **kwargs)
        print(f"Return: {func.__name__} -> {result!r}")
        return result
    return wrapper


@log_calls
@validate_args
def divide(a, b):
    """Calculate a / b"""
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b


# Test
divide(10, 3)
# Call: divide(10, 3)
# Return: divide -> 3.3333333333333335

# divide(10, None)
# ValueError: divide: Argument 1 is None
```

### 10.3 Advanced Exercises

**Exercise A-1: Implementing a Monad-Style Pipeline**

```
Requirements:
Implement an Optional (Maybe) monad-style pipeline in TypeScript.
1. Implement an Optional<T> class:
   - of(value): Wrap a value
   - map(fn): Apply function if value exists
   - flatMap(fn): Apply a function that returns Optional (flatten nesting)
   - getOrElse(defaultValue): Return default if no value
2. Build a pipeline to safely retrieve a user's postal code
```

```typescript
// Solution

class Optional<T> {
    private constructor(private readonly value: T | null | undefined) {}

    static of<T>(value: T | null | undefined): Optional<T> {
        return new Optional(value);
    }

    static empty<T>(): Optional<T> {
        return new Optional<T>(null);
    }

    isPresent(): boolean {
        return this.value !== null && this.value !== undefined;
    }

    map<U>(fn: (value: T) => U): Optional<U> {
        if (!this.isPresent()) return Optional.empty<U>();
        return Optional.of(fn(this.value!));
    }

    flatMap<U>(fn: (value: T) => Optional<U>): Optional<U> {
        if (!this.isPresent()) return Optional.empty<U>();
        return fn(this.value!);
    }

    getOrElse(defaultValue: T): T {
        return this.isPresent() ? this.value! : defaultValue;
    }

    filter(predicate: (value: T) => boolean): Optional<T> {
        if (!this.isPresent()) return this;
        return predicate(this.value!) ? this : Optional.empty();
    }
}

// User data type definition
interface User {
    name: string;
    address?: {
        street?: string;
        city?: string;
        zip?: string;
    };
}

// Safe postal code retrieval pipeline
function getFormattedZip(user: User | null): string {
    return Optional.of(user)
        .flatMap(u => Optional.of(u.address))
        .flatMap(a => Optional.of(a.zip))
        .filter(zip => /^\d{3}-?\d{4}$/.test(zip))
        .map(zip => zip.includes("-") ? zip : `${zip.slice(0,3)}-${zip.slice(3)}`)
        .getOrElse("No postal code");
}

// Test
const user1: User = { name: "Taro", address: { zip: "1000001" } };
const user2: User = { name: "Hanako" };
const user3: User | null = null;

console.log(getFormattedZip(user1));  // => "100-0001"
console.log(getFormattedZip(user2));  // => "No postal code"
console.log(getFormattedZip(user3));  // => "No postal code"
```

**Exercise A-2: Designing a Type-Safe Event Bus**

```
Requirements:
Design and implement an event bus in TypeScript that satisfies the following.
1. Guarantee event name to type correspondence at the type level
2. Provide on/off/emit/once methods
3. Support wildcard (*) listeners
4. Support listener priorities
```

```typescript
// Solution

type EventHandler<T = unknown> = (data: T) => void;

interface ListenerEntry<T = unknown> {
    handler: EventHandler<T>;
    priority: number;
    once: boolean;
}

class EventBus<TEvents extends Record<string, unknown>> {
    private listeners = new Map<string, ListenerEntry[]>();
    private wildcardListeners: ListenerEntry<{ event: string; data: unknown }>[] = [];

    on<K extends keyof TEvents & string>(
        event: K,
        handler: EventHandler<TEvents[K]>,
        priority = 0,
    ): () => void {
        return this.addListener(event, handler, priority, false);
    }

    once<K extends keyof TEvents & string>(
        event: K,
        handler: EventHandler<TEvents[K]>,
        priority = 0,
    ): () => void {
        return this.addListener(event, handler, priority, true);
    }

    onAny(
        handler: EventHandler<{ event: string; data: unknown }>,
        priority = 0,
    ): () => void {
        const entry: ListenerEntry<{ event: string; data: unknown }> = {
            handler,
            priority,
            once: false,
        };
        this.wildcardListeners.push(entry);
        this.wildcardListeners.sort((a, b) => b.priority - a.priority);
        return () => {
            const idx = this.wildcardListeners.indexOf(entry);
            if (idx >= 0) this.wildcardListeners.splice(idx, 1);
        };
    }

    emit<K extends keyof TEvents & string>(event: K, data: TEvents[K]): void {
        // Regular listeners
        const entries = this.listeners.get(event) || [];
        const remaining: ListenerEntry[] = [];
        for (const entry of entries) {
            entry.handler(data);
            if (!entry.once) remaining.push(entry);
        }
        this.listeners.set(event, remaining);

        // Wildcard listeners
        for (const entry of this.wildcardListeners) {
            entry.handler({ event, data });
        }
    }

    private addListener(
        event: string,
        handler: EventHandler<any>,
        priority: number,
        once: boolean,
    ): () => void {
        const entry: ListenerEntry = { handler, priority, once };
        if (!this.listeners.has(event)) {
            this.listeners.set(event, []);
        }
        const entries = this.listeners.get(event)!;
        entries.push(entry);
        entries.sort((a, b) => b.priority - a.priority);

        return () => {
            const idx = entries.indexOf(entry);
            if (idx >= 0) entries.splice(idx, 1);
        };
    }
}

// Usage
interface AppEvents {
    "user:login":  { userId: string; time: number };
    "user:logout": { userId: string };
    "error":       { code: number; message: string };
}

const bus = new EventBus<AppEvents>();

// Regular listener
const unsub = bus.on("user:login", (data) => {
    console.log(`Login: ${data.userId}`);
});

// One-time listener
bus.once("error", (data) => {
    console.error(`Error ${data.code}: ${data.message}`);
});

// Wildcard listener (monitor all events)
bus.onAny(({ event, data }) => {
    console.log(`[Audit Log] ${event}:`, data);
});

bus.emit("user:login", { userId: "user-1", time: Date.now() });
unsub();  // Unsubscribe
```

**Exercise A-3: Implementing a Lazy Evaluation Chain**

```
Requirements:
Implement lazy evaluation collection processing in JavaScript.
1. Implement a LazySeq class:
   - map, filter, take, forEach methods
   - No actual computation until forEach is called
2. Verify that map -> filter -> take works on infinite sequences
```

```javascript
// Solution

class LazySeq {
    constructor(iterable) {
        this.source = iterable;
        this.transforms = [];
    }

    static from(iterable) {
        return new LazySeq(iterable);
    }

    static range(start = 0, end = Infinity, step = 1) {
        return new LazySeq({
            *[Symbol.iterator]() {
                for (let i = start; i < end; i += step) {
                    yield i;
                }
            }
        });
    }

    map(fn) {
        const clone = new LazySeq(this.source);
        clone.transforms = [...this.transforms, { type: "map", fn }];
        return clone;
    }

    filter(fn) {
        const clone = new LazySeq(this.source);
        clone.transforms = [...this.transforms, { type: "filter", fn }];
        return clone;
    }

    take(n) {
        const clone = new LazySeq(this.source);
        clone.transforms = [...this.transforms, { type: "take", n }];
        return clone;
    }

    *[Symbol.iterator]() {
        let count = 0;
        let takeLimit = Infinity;

        // Pre-compute take limit
        for (const t of this.transforms) {
            if (t.type === "take") takeLimit = Math.min(takeLimit, t.n);
        }

        outer:
        for (const item of this.source) {
            let value = item;
            let skip = false;

            for (const transform of this.transforms) {
                if (transform.type === "map") {
                    value = transform.fn(value);
                } else if (transform.type === "filter") {
                    if (!transform.fn(value)) {
                        skip = true;
                        break;
                    }
                } else if (transform.type === "take") {
                    // Handled later
                }
            }

            if (skip) continue;
            if (count >= takeLimit) break;
            count++;
            yield value;
        }
    }

    toArray() {
        return [...this];
    }

    forEach(fn) {
        for (const item of this) {
            fn(item);
        }
    }
}

// Test: Get 10 primes from an infinite sequence
function isPrime(n) {
    if (n < 2) return false;
    for (let i = 2; i <= Math.sqrt(n); i++) {
        if (n % i === 0) return false;
    }
    return true;
}

const first10Primes = LazySeq.range(2)
    .filter(isPrime)
    .take(10)
    .toArray();

console.log(first10Primes);
// => [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

// Test: Verify lazy evaluation (works even on infinite sequences)
const result = LazySeq.range(1)
    .map(x => x * x)
    .filter(x => x % 2 === 1)
    .take(5)
    .toArray();

console.log(result);
// => [1, 9, 25, 49, 81]
```

---

## 11. FAQ (Frequently Asked Questions)

### Q1: Are first-class functions and closures the same thing?

**A**: No, they are different concepts. First-class functions refers to a language's ability to "treat functions as values," while closures refers to "functions that retain the environment (variable bindings) from their definition site." Closures presuppose the mechanism of first-class functions, but things that are first-class functions but not closures (pure function pointers that don't capture any environment, etc.) do exist.

```
Relationship diagram:
  +--------------------------------------+
  |         First-Class Functions        |
  |  +-------------------------------+  |
  |  |      Closures                  |  |
  |  |  (Functions that capture       |  |
  |  |   their environment)           |  |
  |  +-------------------------------+  |
  |                                      |
  |  Function pointers (no environment) |
  |  Pure function literals (no env)    |
  +--------------------------------------+
```

### Q2: Are lambda expressions and anonymous functions the same thing?

**A**: In practice, they are used almost synonymously. Strictly speaking, "lambda expression" is a mathematical concept derived from lambda calculus, while "anonymous function" is a programming term for a function literal without a name. In Python, the `lambda` keyword creates anonymous functions, but with the restriction that only a single expression can be used. JavaScript's arrow functions `(x) => x + 1` are also a type of anonymous function (lambda expression).

### Q3: Should all functions be written as arrow functions? (JavaScript)

**A**: No. Arrow functions are convenient, but regular function declarations are more appropriate in the following cases:

1. **When `this` binding is needed**: Arrow functions don't have their own `this`, so they're unsuitable for object methods or prototype methods
2. **When the `arguments` object is needed**: Arrow functions don't have `arguments`
3. **Generator functions**: Require `function*` syntax, which can't be written as arrow functions
4. **When hoisting is needed**: function declarations are hoisted, but arrow functions (assigned to const) are not
5. **Constructors**: Arrow functions cannot be called with `new`

### Q4: Does using higher-order functions degrade performance?

**A**: Generally, the call overhead of higher-order functions is negligibly small. Modern JavaScript engines (V8, etc.) optimize higher-order function overhead to nearly zero through inlining and JIT compilation. However, caution is needed in the following cases:

- **Function object creation in hot loops**: When passing literals directly like `arr.map(x => x * 2)`, a new function object may be created each time (many engines optimize this)
- **Re-rendering in React**: Creating new functions inside components each time can trigger unnecessary re-renders of child components
- **Rust**: Closures are monomorphized at compile time, so runtime overhead is zero

### Q5: Is functional programming or object-oriented programming better?

**A**: This is not an either/or choice. Many modern languages (JavaScript, Python, Kotlin, Swift, Rust, etc.) support both paradigms, and the best approach is to use each as appropriate for the situation.

| Scenario | Functional is suitable | OOP is suitable |
|----------|----------------------|-----------------|
| Data transformation | map/filter/reduce pipelines | - |
| State management | Immutable data flow | Stateful objects |
| Behavior switching | Higher-order functions/callbacks | Strategy pattern (classes) |
| Large-scale design | Function composition/modules | Class hierarchies/DI |
| Domain models | Algebraic data types | Entities/value objects |
| Concurrency | Pure functions/actors | synchronized/locks |

### Q6: What is the difference between currying and partial application?

**A**: They are similar but different operations.

- **Currying**: Transforms an n-argument function into a chain of 1-argument functions. `f(a, b, c)` -> `f(a)(b)(c)`. In Haskell, all functions are automatically curried
- **Partial Application**: Fixes some arguments of an n-argument function to generate a new function that accepts the remaining arguments. `f(a, b, c)` with a fixed -> `g(b, c) = f(fixed_a, b, c)`

```
Currying vs Partial Application
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Original function: f(a, b, c)

  Currying:
  f(a, b, c) -> curry(f) -> g(a)(b)(c)
  Transform to accept all arguments one at a time

  Partial Application:
  f(a, b, c) -> partial(f, fixedA) -> g(b, c)
  Fix some arguments to generate a new function

  Differences:
  - Currying always takes one argument at a time
  - Partial application can fix any number of arguments
  - Currying transforms the shape of the function
  - Partial application fills in arguments
```

---


## FAQ

### Q1: What is the most important point in learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory, but by actually writing code and verifying behavior.

### Q2: What are common mistakes beginners make?

Skipping the basics and jumping to advanced topics. We recommend thoroughly understanding the fundamental concepts explained in this guide before moving to the next step.

### Q3: How is this applied in practice?

Knowledge of this topic is frequently used in everyday development work. It becomes particularly important during code reviews and architecture design.

---

## 12. Summary

### 12.1 Concept Summary

| Concept | Definition | Typical Use Case |
|---------|-----------|------------------|
| First-Class Functions | Language feature allowing functions to be treated as values | Variable assignment, argument passing, return values |
| Higher-Order Functions | Functions that receive/return functions | map, filter, reduce, decorators |
| Callbacks | Functions passed as arguments | Event handlers, async processing |
| Strategy | Design of swapping behavior via functions | Sort order switching, validation |
| Partial Application | Creating new functions by fixing some arguments | Fixing configuration, logger generation |
| Currying | Transforming an n-argument function into a chain of 1-argument functions | Incremental argument application |
| Function Composition | Combining multiple functions to create a new function | Data processing pipelines |
| Closures | Functions that retain their definition-time environment | State hiding, factories |
| Dispatch Tables | Storing functions in dictionaries for dynamic selection | Command processing, routing |
| Memoization | Higher-order function that caches results | Optimizing computationally expensive functions |
| Middleware | Function chains that intercept processing | HTTP request handling, logging |

### 12.2 Design Decision Guidelines

```
First-Class Functions Application Decision Flowchart
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Want to abstract a processing pattern?
  |
  +-- YES -> Does the same pattern appear 3+ times?
  |           |
  |           +-- YES -> Extract as a higher-order function
  |           |           |
  |           |           +-- Transform type -> Create a map-like function
  |           |           +-- Selection type -> Create a filter-like function
  |           |           +-- Aggregation type -> Create a reduce-like function
  |           |
  |           +-- NO  -> Keep it inline
  |                       (Avoid over-abstraction)
  |
  +-- Want to switch behavior at runtime?
  |    |
  |    +-- YES -> Is the state complex?
  |    |           |
  |    |           +-- YES -> OOP strategy pattern
  |    |           +-- NO  -> Switch via function arguments/variables
  |    |
  |    +-- NO
  |
  +-- Want a function with pre-fixed configuration?
       |
       +-- YES -> Partial application or factory function
       +-- NO  -> A regular function is sufficient
```

---

## 13. Suggested Next Reading

- [Closures](./01-closures.md) - Proceed to the next topic

---

## 14. References

1. Abelson, H. & Sussman, G. J. *Structure and Interpretation of Computer Programs (SICP)*, 2nd ed., MIT Press, 1996. -- Chapter 1 "Building Abstractions with Procedures" thoroughly explains the concept of first-class functions. A classic masterpiece for learning the essence of higher-order functions and function composition through LISP/Scheme.

2. Strachey, C. "Fundamental Concepts in Programming Languages," *Higher-Order and Symbolic Computation*, Vol.13, pp.11--49, 2000 (originally 1967 lecture notes). -- The historical document that first systematized the concept of "first-class." Defined foundational concepts in programming language semantics.

3. Bird, R. *Thinking Functionally with Haskell*, Cambridge University Press, 2015. -- Systematically teaches functional programming concepts including first-class functions, higher-order functions, function composition, and currying, using Haskell as the subject.

4. Crockford, D. *JavaScript: The Good Parts*, O'Reilly Media, 2008. -- Practical explanation of first-class nature of functions, closures, and callback patterns in JavaScript. Also details JavaScript-specific pitfalls (loss of this, etc.).

5. Kleppmann, M. *Designing Data-Intensive Applications*, O'Reilly Media, 2017. -- Discusses the importance of functional patterns (immutable data, pure functions) in distributed systems in the context of real systems.

6. Gamma, E., Helm, R., Johnson, R. & Vlissides, J. *Design Patterns: Elements of Reusable Object-Oriented Software*, Addison-Wesley, 1994. -- The original source for GoF patterns such as strategy and command patterns that can be concisely expressed with first-class functions. Useful for understanding the contrast between OOP and FP.

---

## Suggested Next Reading

- [Closures](./01-closures.md) - Proceed to the next topic

---

## References

- [MDN Web Docs](https://developer.mozilla.org/) - Web technology reference
- [Wikipedia](https://en.wikipedia.org/) - Overview of technical concepts
