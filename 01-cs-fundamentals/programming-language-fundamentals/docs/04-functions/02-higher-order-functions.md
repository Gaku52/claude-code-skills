# Higher-Order Functions

> Higher-order functions are "functions that take functions as arguments or return functions as results." They dramatically improve code reusability and the level of abstraction.

## What You Will Learn in This Chapter

- [ ] Understand the essence of map, filter, and reduce
- [ ] Master abstraction patterns using higher-order functions
- [ ] Understand practical applications of functional programming
- [ ] Understand the difference between currying and partial application
- [ ] Master declarative programming through function composition
- [ ] Gain a cross-cutting understanding of higher-order function implementation patterns across languages


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding the content of [Closures](./01-closures.md)

---

## 1. Fundamental Concepts of Higher-Order Functions

### 1.1 What Are Higher-Order Functions?

A higher-order function satisfies one or both of the following conditions:

1. **Accepts a function as an argument** (callback functions, predicate functions, etc.)
2. **Returns a function as a result** (closures, factory functions, etc.)

This concept originates from Lambda Calculus, a computational model formalized by Alonzo Church in the 1930s. The existence of higher-order functions allows functions to be treated as "first-class citizens," meaning they can be assigned to variables, stored in data structures, or passed to other functions.

```
Conditions for first-class functions:
1. Can be assigned to a variable       const f = Math.sqrt;
2. Can be passed as an argument        arr.map(f);
3. Can be returned as a value          function make() { return f; }
4. Can be stored in data structures    const fns = [f, Math.abs];
5. Can be created at runtime           const g = (x) => x * 2;
```

### 1.2 Why Are Higher-Order Functions Important?

The reasons higher-order functions are important in programming are as follows:

```
┌─────────────────────────────────────────────────┐
│         Benefits of Higher-Order Functions       │
├─────────────────────────────────────────────────┤
│ 1. Abstraction: Extract common patterns as       │
│    functions                                     │
│ 2. Reusability: Parameterize behavior for        │
│    general-purpose use                           │
│ 3. Composability: Build complex operations by    │
│    combining small functions                     │
│ 4. Declarative: Describe "what to do," abstract  │
│    away "how to do it"                           │
│ 5. Testability: Pure functions can be tested     │
│    individually                                  │
│ 6. Lazy evaluation: Computation can be deferred  │
│    until needed                                  │
└─────────────────────────────────────────────────┘
```

```typescript
// Imperative approach (low abstraction)
const results: string[] = [];
for (let i = 0; i < users.length; i++) {
    if (users[i].active) {
        results.push(users[i].name.toUpperCase());
    }
}

// Declarative approach (high abstraction - using higher-order functions)
const results = users
    .filter(u => u.active)
    .map(u => u.name.toUpperCase());
```

The imperative approach describes "how to process" (managing loop variables, branching conditions, appending to arrays), while the declarative approach describes only "what we want to do" (filter active users and convert names to uppercase). This difference has a significant impact on code readability and maintainability.

### 1.3 Historical Background

```
1930s     Alonzo Church formalizes Lambda Calculus
1958      Lisp is born - first practical implementation of higher-order functions
1973      ML is born - fusion of type inference and higher-order functions
1990      Haskell is born - pure functional language
2004      Scala is born - fusion of OOP + FP
2007      C# 3.0 LINQ - higher-order functions permeate mainstream languages
2011      Java 8 Lambda - higher-order functions introduced to the world's most used language
2015      ES2015 (JavaScript) - Arrow Functions standardized
2015      Rust 1.0 - higher-order functions as zero-cost abstractions
```

---

## 2. The Big Three Higher-Order Functions: map, filter, reduce

### 2.1 Conceptual Diagram

```
map:    Transform each element       [1,2,3] → [2,4,6]
filter: Select elements matching     [1,2,3,4,5] → [2,4]
        a condition
reduce: Aggregate all elements       [1,2,3,4,5] → 15
        into a single value
```

```
                    map (f)
┌───┬───┬───┬───┐  f(x) = x * 2  ┌───┬───┬───┬───┐
│ 1 │ 2 │ 3 │ 4 │ ─────────────→ │ 2 │ 4 │ 6 │ 8 │
└───┴───┴───┴───┘                 └───┴───┴───┴───┘
  Input array (same length)         Output array (same length)

                    filter (p)
┌───┬───┬───┬───┐  p(x) = x > 2  ┌───┬───┐
│ 1 │ 2 │ 3 │ 4 │ ─────────────→ │ 3 │ 4 │
└───┴───┴───┴───┘                 └───┴───┘
  Input array                       Output array (count ≤ input)

                    reduce (f, init)
┌───┬───┬───┬───┐  f(acc, x) = acc + x  ┌────┐
│ 1 │ 2 │ 3 │ 4 │ ───────────────────→  │ 10 │
└───┴───┴───┴───┘  init = 0              └────┘
  Input array                             Single value
```

### 2.2 Implementation in TypeScript

```typescript
// TypeScript: The big three higher-order functions
const numbers = [1, 2, 3, 4, 5];

// map: Transform
numbers.map(n => n * 2);              // [2, 4, 6, 8, 10]
numbers.map(n => n.toString());       // ["1", "2", "3", "4", "5"]
numbers.map((n, i) => ({ index: i, value: n }));

// filter: Select
numbers.filter(n => n > 3);           // [4, 5]
numbers.filter(n => n % 2 === 0);     // [2, 4]

// reduce: Aggregate
numbers.reduce((acc, n) => acc + n, 0);        // 15 (sum)
numbers.reduce((acc, n) => acc * n, 1);        // 120 (product)
numbers.reduce((max, n) => Math.max(max, n), -Infinity); // 5 (maximum)

// reduce can implement other higher-order functions
const myMap = <T, U>(arr: T[], fn: (x: T) => U): U[] =>
    arr.reduce<U[]>((acc, x) => [...acc, fn(x)], []);

const myFilter = <T>(arr: T[], pred: (x: T) => boolean): T[] =>
    arr.reduce<T[]>((acc, x) => pred(x) ? [...acc, x] : acc, []);
```

### 2.3 Deep Understanding of reduce

`reduce` is the most powerful and versatile of the big three higher-order functions. As demonstrated by the fact that `map` and `filter` can be implemented with `reduce`, `reduce` is a special case of the fold operation.

```typescript
// Tracing the mechanism of reduce step by step
const trace = [1, 2, 3, 4].reduce((acc, n) => {
    console.log(`acc=${acc}, n=${n}, result=${acc + n}`);
    return acc + n;
}, 0);
// acc=0, n=1, result=1
// acc=1, n=2, result=3
// acc=3, n=3, result=6
// acc=6, n=4, result=10
// → 10

// Achieving various data transformations with reduce
const items = ["apple", "banana", "apple", "cherry", "banana", "apple"];

// Frequency count
const frequency = items.reduce<Record<string, number>>((acc, item) => {
    acc[item] = (acc[item] || 0) + 1;
    return acc;
}, {});
// → { apple: 3, banana: 2, cherry: 1 }

// Array flattening
const nested = [[1, 2], [3, 4], [5, 6]];
const flat = nested.reduce<number[]>((acc, arr) => [...acc, ...arr], []);
// → [1, 2, 3, 4, 5, 6]

// Pipeline construction
type Transform = (s: string) => string;
const transforms: Transform[] = [
    s => s.trim(),
    s => s.toLowerCase(),
    s => s.replace(/\s+/g, "-"),
];
const slugify = (input: string): string =>
    transforms.reduce((result, transform) => transform(result), input);
slugify("  Hello World  ");  // → "hello-world"

// reduceRight: Aggregate from right to left
const compose = <T>(...fns: Array<(x: T) => T>) =>
    (x: T): T => fns.reduceRight((acc, fn) => fn(acc), x);
```

### 2.4 The Big Three Higher-Order Functions in Python

```python
from functools import reduce
from typing import List, Dict, Any

numbers = [1, 2, 3, 4, 5]

# map: Transform (lazy evaluation - returns an iterator)
doubled = list(map(lambda n: n * 2, numbers))      # [2, 4, 6, 8, 10]
strings = list(map(str, numbers))                    # ['1', '2', '3', '4', '5']

# filter: Select (lazy evaluation)
evens = list(filter(lambda n: n % 2 == 0, numbers)) # [2, 4]
adults = list(filter(lambda u: u['age'] >= 18, users))

# reduce: Aggregate
total = reduce(lambda acc, n: acc + n, numbers, 0)   # 15
product = reduce(lambda acc, n: acc * n, numbers, 1)  # 120

# In Python, list comprehensions are often preferred
doubled_comp = [n * 2 for n in numbers]               # [2, 4, 6, 8, 10]
evens_comp = [n for n in numbers if n % 2 == 0]       # [2, 4]

# Generator expressions (memory efficient)
sum_of_squares = sum(n ** 2 for n in range(1000000))

# Complex transformation pipeline
users: List[Dict[str, Any]] = [
    {"name": "Alice", "age": 30, "active": True},
    {"name": "Bob", "age": 17, "active": True},
    {"name": "Charlie", "age": 25, "active": False},
    {"name": "Diana", "age": 22, "active": True},
]

# Functional style
active_adult_names = list(
    map(
        lambda u: u["name"],
        filter(
            lambda u: u["active"] and u["age"] >= 18,
            users,
        ),
    )
)

# List comprehension style (preferred in Python)
active_adult_names = [
    u["name"] for u in users
    if u["active"] and u["age"] >= 18
]

# Advanced usage with functools
from functools import partial, lru_cache

# partial: Partial application
def multiply(a: int, b: int) -> int:
    return a * b

double = partial(multiply, 2)
triple = partial(multiply, 3)
print(list(map(double, numbers)))  # [2, 4, 6, 8, 10]
print(list(map(triple, numbers)))  # [3, 6, 9, 12, 15]

# Combining with itertools
from itertools import chain, starmap, accumulate

# accumulate: Cumulative calculation (scan operation)
running_sum = list(accumulate(numbers))        # [1, 3, 6, 10, 15]
running_max = list(accumulate(numbers, max))   # [1, 2, 3, 4, 5]

# chain: Concatenate multiple iterables
combined = list(chain([1, 2], [3, 4], [5, 6]))  # [1, 2, 3, 4, 5, 6]

# starmap: Unpack tuples
pairs = [(2, 5), (3, 2), (10, 3)]
results = list(starmap(pow, pairs))  # [32, 9, 1000]
```

### 2.5 Higher-Order Functions in Go

```go
package main

import (
    "fmt"
    "strings"
)

// Go has generics (1.18+) but no built-in map/filter
// We implement them ourselves

// Map: Apply a function to each element of a slice
func Map[T, U any](slice []T, f func(T) U) []U {
    result := make([]U, len(slice))
    for i, v := range slice {
        result[i] = f(v)
    }
    return result
}

// Filter: Return only elements that satisfy the predicate
func Filter[T any](slice []T, pred func(T) bool) []T {
    result := make([]T, 0)
    for _, v := range slice {
        if pred(v) {
            result = append(result, v)
        }
    }
    return result
}

// Reduce: Aggregate a slice into a single value
func Reduce[T, U any](slice []T, init U, f func(U, T) U) U {
    acc := init
    for _, v := range slice {
        acc = f(acc, v)
    }
    return acc
}

// ForEach: Execute a side effect for each element
func ForEach[T any](slice []T, f func(T)) {
    for _, v := range slice {
        f(v)
    }
}

// Any: Check if any element satisfies the condition
func Any[T any](slice []T, pred func(T) bool) bool {
    for _, v := range slice {
        if pred(v) {
            return true
        }
    }
    return false
}

// All: Check if all elements satisfy the condition
func All[T any](slice []T, pred func(T) bool) bool {
    for _, v := range slice {
        if !pred(v) {
            return false
        }
    }
    return true
}

func main() {
    numbers := []int{1, 2, 3, 4, 5}

    // Map
    doubled := Map(numbers, func(n int) int { return n * 2 })
    fmt.Println(doubled) // [2 4 6 8 10]

    // Filter
    evens := Filter(numbers, func(n int) bool { return n%2 == 0 })
    fmt.Println(evens) // [2 4]

    // Reduce
    sum := Reduce(numbers, 0, func(acc, n int) int { return acc + n })
    fmt.Println(sum) // 15

    // Chaining (in Go this becomes nested function calls)
    words := []string{"hello", "world", "go", "higher", "order"}
    longUpper := Map(
        Filter(words, func(s string) bool { return len(s) > 3 }),
        func(s string) string { return strings.ToUpper(s) },
    )
    fmt.Println(longUpper) // [HELLO WORLD HIGHER ORDER]
}
```

### 2.6 The Big Three Higher-Order Functions in Rust

```rust
// Rust: Iterators + higher-order functions (zero-cost abstractions)

fn main() {
    let words = vec!["hello", "world", "foo", "bar"];

    // map + collect
    let upper: Vec<String> = words.iter()
        .map(|w| w.to_uppercase())
        .collect();
    // → ["HELLO", "WORLD", "FOO", "BAR"]

    // filter + map + collect (can be combined with filter_map)
    let lengths: Vec<usize> = words.iter()
        .filter(|w| w.len() > 3)
        .map(|w| w.len())
        .collect();
    // → [5, 5]

    // filter_map: Filter out None while transforming
    let valid_numbers: Vec<i32> = vec!["1", "abc", "3", "def", "5"]
        .iter()
        .filter_map(|s| s.parse::<i32>().ok())
        .collect();
    // → [1, 3, 5]

    // fold (equivalent to reduce)
    let numbers = vec![1, 2, 3, 4, 5];
    let sum: i32 = numbers.iter().fold(0, |acc, &n| acc + n);
    // → 15

    // reduce without initial value (Rust 1.51+: reduce method)
    let product: Option<i32> = numbers.iter().copied().reduce(|a, b| a * b);
    // → Some(120)

    // scan: Cumulative calculation (lazy iterator)
    let running_sum: Vec<i32> = numbers.iter()
        .scan(0, |state, &n| {
            *state += n;
            Some(*state)
        })
        .collect();
    // → [1, 3, 6, 10, 15]

    // enumerate + map: Transformation with index
    let indexed: Vec<String> = words.iter()
        .enumerate()
        .map(|(i, w)| format!("{}:{}", i, w))
        .collect();
    // → ["0:hello", "1:world", "2:foo", "3:bar"]

    // zip: Combine two iterators
    let keys = vec!["a", "b", "c"];
    let values = vec![1, 2, 3];
    let pairs: Vec<(&str, i32)> = keys.iter()
        .copied()
        .zip(values.iter().copied())
        .collect();
    // → [("a", 1), ("b", 2), ("c", 3)]

    // chain: Concatenate two iterators
    let first = vec![1, 2, 3];
    let second = vec![4, 5, 6];
    let combined: Vec<i32> = first.iter()
        .chain(second.iter())
        .copied()
        .collect();
    // → [1, 2, 3, 4, 5, 6]

    // flat_map: Flatten nested structures
    let sentences = vec!["hello world", "foo bar"];
    let words_flat: Vec<&str> = sentences.iter()
        .flat_map(|s| s.split_whitespace())
        .collect();
    // → ["hello", "world", "foo", "bar"]

    // partition: Split into two groups by condition
    let numbers = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let (evens, odds): (Vec<i32>, Vec<i32>) = numbers.iter()
        .partition(|&&n| n % 2 == 0);
    // evens → [2, 4, 6, 8, 10], odds → [1, 3, 5, 7, 9]
}
```

---

## 3. Practical Data Transformation Pipelines

### 3.1 Building Data Pipelines

```typescript
// Data transformation pipeline
interface User { name: string; age: number; active: boolean; }

const users: User[] = [
    { name: "Alice", age: 30, active: true },
    { name: "Bob", age: 17, active: true },
    { name: "Charlie", age: 25, active: false },
    { name: "Diana", age: 22, active: true },
];

// Get names of active adult users
const result = users
    .filter(u => u.active)
    .filter(u => u.age >= 18)
    .map(u => u.name)
    .sort();
// → ["Alice", "Diana"]

// Grouping (reduce)
const byAge = users.reduce<Record<string, User[]>>((groups, user) => {
    const key = user.age >= 18 ? "adult" : "minor";
    return { ...groups, [key]: [...(groups[key] || []), user] };
}, {});
// → { adult: [Alice, Charlie, Diana], minor: [Bob] }

// Object.groupBy (ES2024)
const grouped = Object.groupBy(users, u => u.age >= 18 ? "adult" : "minor");
```

### 3.2 Complex Business Logic

```typescript
// E-commerce order processing pipeline
interface Order {
    id: string;
    userId: string;
    items: OrderItem[];
    status: "pending" | "confirmed" | "shipped" | "delivered" | "cancelled";
    createdAt: Date;
    shippingAddress: Address;
}

interface OrderItem {
    productId: string;
    name: string;
    price: number;
    quantity: number;
    category: string;
}

interface Address {
    country: string;
    prefecture: string;
    city: string;
}

interface OrderSummary {
    userId: string;
    totalOrders: number;
    totalSpent: number;
    averageOrderValue: number;
    topCategory: string;
    lastOrderDate: Date;
}

// Generate order summaries per user
function generateOrderSummaries(orders: Order[]): OrderSummary[] {
    return Object.entries(
        // 1. Group by user ID
        orders
            .filter(o => o.status !== "cancelled")
            .reduce<Record<string, Order[]>>((groups, order) => {
                const key = order.userId;
                return {
                    ...groups,
                    [key]: [...(groups[key] || []), order],
                };
            }, {})
    )
    // 2. Calculate summary for each user
    .map(([userId, userOrders]): OrderSummary => {
        const totalSpent = userOrders
            .flatMap(o => o.items)
            .reduce((sum, item) => sum + item.price * item.quantity, 0);

        // Aggregate purchase amount by category
        const categorySpend = userOrders
            .flatMap(o => o.items)
            .reduce<Record<string, number>>((acc, item) => ({
                ...acc,
                [item.category]: (acc[item.category] || 0) + item.price * item.quantity,
            }), {});

        // Get the category with the highest amount
        const topCategory = Object.entries(categorySpend)
            .reduce((max, [cat, amount]) =>
                amount > max[1] ? [cat, amount] : max,
                ["", 0]
            )[0];

        return {
            userId,
            totalOrders: userOrders.length,
            totalSpent,
            averageOrderValue: totalSpent / userOrders.length,
            topCategory,
            lastOrderDate: userOrders
                .map(o => o.createdAt)
                .reduce((latest, date) => date > latest ? date : latest),
        };
    })
    // 3. Sort by total spent in descending order
    .sort((a, b) => b.totalSpent - a.totalSpent);
}

// Generate monthly revenue report
function monthlyRevenue(orders: Order[]): Map<string, number> {
    return orders
        .filter(o => o.status !== "cancelled")
        .reduce((map, order) => {
            const key = `${order.createdAt.getFullYear()}-${String(order.createdAt.getMonth() + 1).padStart(2, "0")}`;
            const orderTotal = order.items.reduce(
                (sum, item) => sum + item.price * item.quantity, 0
            );
            map.set(key, (map.get(key) || 0) + orderTotal);
            return map;
        }, new Map<string, number>());
}
```

### 3.3 CSV / JSON Data Transformation

```typescript
// CSV parser (leveraging higher-order functions)
function parseCSV<T>(
    csv: string,
    transform: (row: Record<string, string>) => T
): T[] {
    const lines = csv.trim().split("\n");
    const headers = lines[0].split(",").map(h => h.trim());

    return lines
        .slice(1)
        .map(line => line.split(",").map(cell => cell.trim()))
        .filter(cells => cells.length === headers.length)
        .map(cells =>
            headers.reduce<Record<string, string>>(
                (obj, header, i) => ({ ...obj, [header]: cells[i] }),
                {}
            )
        )
        .map(transform);
}

// Usage example
const csvData = `
name, age, department, salary
Alice, 30, Engineering, 80000
Bob, 25, Marketing, 60000
Charlie, 35, Engineering, 95000
Diana, 28, Design, 70000
`;

interface Employee {
    name: string;
    age: number;
    department: string;
    salary: number;
}

const employees = parseCSV<Employee>(csvData, row => ({
    name: row.name,
    age: parseInt(row.age, 10),
    department: row.department,
    salary: parseInt(row.salary, 10),
}));

// Average salary by department
const avgSalaryByDept = employees
    .reduce<Record<string, { total: number; count: number }>>((acc, emp) => ({
        ...acc,
        [emp.department]: {
            total: (acc[emp.department]?.total || 0) + emp.salary,
            count: (acc[emp.department]?.count || 0) + 1,
        },
    }), {});

const departmentReport = Object.entries(avgSalaryByDept)
    .map(([dept, { total, count }]) => ({
        department: dept,
        averageSalary: Math.round(total / count),
        employeeCount: count,
    }))
    .sort((a, b) => b.averageSalary - a.averageSalary);
```

---

## 4. Higher-Order Functions That Return Functions

### 4.1 Factory Pattern

```typescript
// Factory pattern: Validator generation
function createValidator(rules: Record<string, (v: any) => boolean>) {
    return function validate(data: Record<string, any>): string[] {
        const errors: string[] = [];
        for (const [field, rule] of Object.entries(rules)) {
            if (!rule(data[field])) {
                errors.push(`Invalid: ${field}`);
            }
        }
        return errors;
    };
}

const validateUser = createValidator({
    name: (v) => typeof v === "string" && v.length > 0,
    age: (v) => typeof v === "number" && v >= 0 && v <= 150,
    email: (v) => typeof v === "string" && v.includes("@"),
});

validateUser({ name: "", age: -1, email: "invalid" });
// → ["Invalid: name", "Invalid: age", "Invalid: email"]
```

### 4.2 Middleware Pattern

```typescript
// Middleware pattern
type Middleware = (req: Request, next: () => Response) => Response;

function compose(...middlewares: Middleware[]) {
    return (req: Request): Response => {
        let index = 0;
        function next(): Response {
            const mw = middlewares[index++];
            if (!mw) return new Response("Not Found", { status: 404 });
            return mw(req, next);
        }
        return next();
    };
}

// Express-style middleware chain
type ExpressMiddleware<T = any> = (
    req: T,
    res: { body: string; status: number; headers: Record<string, string> },
    next: () => void
) => void;

function createPipeline<T>(...middlewares: ExpressMiddleware<T>[]) {
    return (req: T) => {
        const res = { body: "", status: 200, headers: {} as Record<string, string> };
        let index = 0;

        function next() {
            const mw = middlewares[index++];
            if (mw) {
                mw(req, res, next);
            }
        }

        next();
        return res;
    };
}

// Logging middleware
const logger: ExpressMiddleware = (req, res, next) => {
    console.log(`[${new Date().toISOString()}] Request received`);
    next();
    console.log(`[${new Date().toISOString()}] Response: ${res.status}`);
};

// Authentication middleware
const auth: ExpressMiddleware<{ token?: string }> = (req, res, next) => {
    if (!req.token) {
        res.status = 401;
        res.body = "Unauthorized";
        return;
    }
    next();
};

// Response header middleware
const cors: ExpressMiddleware = (_req, res, next) => {
    res.headers["Access-Control-Allow-Origin"] = "*";
    next();
};
```

### 4.3 Decorator Pattern

```typescript
// Function decorators: Higher-order functions that add functionality to existing functions

// Logging decorator
function withLogging<Args extends any[], R>(
    fn: (...args: Args) => R,
    label?: string
): (...args: Args) => R {
    const name = label || fn.name || "anonymous";
    return (...args: Args): R => {
        console.log(`[${name}] called with:`, args);
        const start = performance.now();
        const result = fn(...args);
        const elapsed = performance.now() - start;
        console.log(`[${name}] returned:`, result, `(${elapsed.toFixed(2)}ms)`);
        return result;
    };
}

// Memoization decorator
function withMemoization<Args extends any[], R>(
    fn: (...args: Args) => R,
    keyFn?: (...args: Args) => string
): (...args: Args) => R {
    const cache = new Map<string, R>();
    return (...args: Args): R => {
        const key = keyFn ? keyFn(...args) : JSON.stringify(args);
        if (cache.has(key)) {
            return cache.get(key)!;
        }
        const result = fn(...args);
        cache.set(key, result);
        return result;
    };
}

// Retry decorator
function withRetry<Args extends any[], R>(
    fn: (...args: Args) => Promise<R>,
    maxRetries: number = 3,
    delayMs: number = 1000
): (...args: Args) => Promise<R> {
    return async (...args: Args): Promise<R> => {
        let lastError: Error | undefined;
        for (let attempt = 0; attempt <= maxRetries; attempt++) {
            try {
                return await fn(...args);
            } catch (error) {
                lastError = error as Error;
                if (attempt < maxRetries) {
                    const delay = delayMs * Math.pow(2, attempt);
                    console.log(`Retry ${attempt + 1}/${maxRetries} after ${delay}ms`);
                    await new Promise(resolve => setTimeout(resolve, delay));
                }
            }
        }
        throw lastError;
    };
}

// Throttle decorator
function withThrottle<Args extends any[]>(
    fn: (...args: Args) => void,
    intervalMs: number
): (...args: Args) => void {
    let lastCallTime = 0;
    return (...args: Args): void => {
        const now = Date.now();
        if (now - lastCallTime >= intervalMs) {
            lastCallTime = now;
            fn(...args);
        }
    };
}

// Debounce decorator
function withDebounce<Args extends any[]>(
    fn: (...args: Args) => void,
    delayMs: number
): (...args: Args) => void {
    let timeoutId: ReturnType<typeof setTimeout> | null = null;
    return (...args: Args): void => {
        if (timeoutId) clearTimeout(timeoutId);
        timeoutId = setTimeout(() => fn(...args), delayMs);
    };
}

// Usage example
const fetchUserData = async (userId: string) => {
    const response = await fetch(`/api/users/${userId}`);
    return response.json();
};

// Composing decorators
const resilientFetchUser = withLogging(
    withRetry(fetchUserData, 3, 500),
    "fetchUserData"
);

const memoizedExpensiveCalc = withMemoization(
    withLogging(
        (n: number) => {
            // Expensive computation
            let result = 0;
            for (let i = 0; i < n; i++) result += Math.sqrt(i);
            return result;
        },
        "expensiveCalc"
    )
);
```

### 4.4 Python Decorators

```python
import functools
import time
import logging
from typing import TypeVar, Callable, Any

F = TypeVar("F", bound=Callable[..., Any])

# Python decorators are syntactic sugar for higher-order functions
# @decorator is equivalent to func = decorator(func)

# Timing decorator
def timing(func: F) -> F:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logging.info(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper  # type: ignore

# Retry decorator (with arguments)
def retry(max_attempts: int = 3, delay: float = 1.0):
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        wait = delay * (2 ** attempt)
                        logging.warning(
                            f"Retry {attempt + 1}/{max_attempts} "
                            f"for {func.__name__} after {wait}s: {e}"
                        )
                        time.sleep(wait)
            raise last_error
        return wrapper  # type: ignore
    return decorator

# Cache decorator (standard library)
@functools.lru_cache(maxsize=128)
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Stacking decorators
@timing
@retry(max_attempts=3, delay=0.5)
def fetch_data(url: str) -> dict:
    """Fetch data from an external API"""
    import requests
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

# Class-based decorator
class RateLimiter:
    def __init__(self, calls_per_second: float = 1.0):
        self.min_interval = 1.0 / calls_per_second
        self.last_call_time = 0.0

    def __call__(self, func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - self.last_call_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_call_time = time.time()
            return func(*args, **kwargs)
        return wrapper  # type: ignore

@RateLimiter(calls_per_second=2.0)
def call_api(endpoint: str) -> dict:
    """Rate-limited API call"""
    pass
```

---

## 5. Currying and Partial Application

### 5.1 Conceptual Differences

```
Currying:
  f(a, b, c) → f(a)(b)(c)
  Transform a multi-argument function into a chain of single-argument functions

Partial Application:
  f(a, b, c) → g(b, c)  (fix a)
  Generate a new function with some arguments fixed

┌─────────────────────────────────────────────┐
│  Currying                                    │
│  add(a, b)  → add(a)(b)                     │
│  add(1, 2)  → add(1)(2) → 3                 │
│                                              │
│  Partial Application                         │
│  add(a, b)  → add1(b)  (fix a=1)            │
│  add(1, 2)  → add1(2)  → 3                  │
└─────────────────────────────────────────────┘
```

### 5.2 Currying in TypeScript

```typescript
// Manual currying
const add = (a: number) => (b: number) => a + b;
add(1)(2);  // → 3

const add5 = add(5);
add5(10);   // → 15
add5(20);   // → 25

// Generic currying function
function curry<A, B, C>(fn: (a: A, b: B) => C): (a: A) => (b: B) => C {
    return (a: A) => (b: B) => fn(a, b);
}

function curry3<A, B, C, D>(
    fn: (a: A, b: B, c: C) => D
): (a: A) => (b: B) => (c: C) => D {
    return (a: A) => (b: B) => (c: C) => fn(a, b, c);
}

// Usage example
const multiply = (a: number, b: number) => a * b;
const curriedMultiply = curry(multiply);
const double = curriedMultiply(2);
const triple = curriedMultiply(3);

[1, 2, 3, 4, 5].map(double);  // [2, 4, 6, 8, 10]
[1, 2, 3, 4, 5].map(triple);  // [3, 6, 9, 12, 15]

// Auto-currying (variadic arguments support)
function autoCurry(fn: Function): Function {
    return function curried(...args: any[]): any {
        if (args.length >= fn.length) {
            return fn(...args);
        }
        return (...moreArgs: any[]) => curried(...args, ...moreArgs);
    };
}

const curriedAdd3 = autoCurry((a: number, b: number, c: number) => a + b + c);
curriedAdd3(1)(2)(3);     // → 6
curriedAdd3(1, 2)(3);     // → 6
curriedAdd3(1)(2, 3);     // → 6
curriedAdd3(1, 2, 3);     // → 6

// Practical use of currying
const propGetter = <T>(key: keyof T) => (obj: T): T[keyof T] => obj[key];
const getName = propGetter<User>("name");
const getAge = propGetter<User>("age");

users.map(getName);  // ["Alice", "Bob", "Charlie", "Diana"]
users.map(getAge);   // [30, 17, 25, 22]

// Predicate function generation
const greaterThan = (threshold: number) => (value: number) => value > threshold;
const isAdult = greaterThan(17);
const isSenior = greaterThan(64);

numbers.filter(greaterThan(3));  // [4, 5]
users.filter(u => isAdult(u.age));  // Alice, Charlie, Diana

// Currying string operations
const startsWith = (prefix: string) => (str: string) => str.startsWith(prefix);
const endsWith = (suffix: string) => (str: string) => str.endsWith(suffix);
const contains = (substr: string) => (str: string) => str.includes(substr);

const files = ["index.ts", "utils.ts", "style.css", "main.js", "test.ts"];
files.filter(endsWith(".ts"));    // ["index.ts", "utils.ts", "test.ts"]
files.filter(startsWith("main")); // ["main.js"]
```

### 5.3 Closures and Currying in Rust

```rust
// In Rust, currying is achieved with move closures
fn make_adder(n: i32) -> impl Fn(i32) -> i32 {
    move |x| x + n
}

fn make_multiplier(factor: f64) -> impl Fn(f64) -> f64 {
    move |x| x * factor
}

fn make_range_checker(min: i32, max: i32) -> impl Fn(i32) -> bool {
    move |x| x >= min && x <= max
}

fn main() {
    let add5 = make_adder(5);
    let double = make_multiplier(2.0);
    let is_valid_age = make_range_checker(0, 150);

    println!("{}", add5(10));          // 15
    println!("{}", double(3.14));      // 6.28
    println!("{}", is_valid_age(25));   // true
    println!("{}", is_valid_age(200));  // false

    // Combining with iterators
    let numbers = vec![1, 2, 3, 4, 5];
    let add10 = make_adder(10);
    let result: Vec<i32> = numbers.iter()
        .map(|&n| add10(n))
        .collect();
    // → [11, 12, 13, 14, 15]

    // Differences between Fn, FnMut, and FnOnce traits
    // Fn:     Called with &self (immutable borrow, can be called multiple times)
    // FnMut:  Called with &mut self (mutable borrow)
    // FnOnce: Called with self (consumes ownership, can only be called once)

    // FnMut example: Closure that modifies internal state
    fn make_counter() -> impl FnMut() -> i32 {
        let mut count = 0;
        move || {
            count += 1;
            count
        }
    }

    let mut counter = make_counter();
    println!("{}", counter()); // 1
    println!("{}", counter()); // 2
    println!("{}", counter()); // 3

    // FnOnce example: Closure that consumes ownership
    fn consume_and_print(f: impl FnOnce() -> String) {
        println!("{}", f());
        // f(); // Compile error: cannot be called twice
    }

    let name = String::from("Alice");
    consume_and_print(move || format!("Hello, {}!", name));
}
```

---

## 6. Function Composition

### 6.1 Basic Function Composition

```
Composition: (f . g)(x) = f(g(x))

  x → [g] → g(x) → [f] → f(g(x))

Example: toUpper . trim
  "  hello  " → trim → "hello" → toUpper → "HELLO"
```

```typescript
// Basic composition
const compose2 = <A, B, C>(
    f: (b: B) => C,
    g: (a: A) => B
): ((a: A) => C) => (a: A) => f(g(a));

const pipe2 = <A, B, C>(
    f: (a: A) => B,
    g: (b: B) => C
): ((a: A) => C) => (a: A) => g(f(a));

// pipe: Execute left to right (more readable)
function pipe<T>(...fns: Array<(arg: any) => any>) {
    return (initial: T) => fns.reduce((acc, fn) => fn(acc), initial as any);
}

// compose: Execute right to left (closer to mathematical notation)
function compose<T>(...fns: Array<(arg: any) => any>) {
    return (initial: T) => fns.reduceRight((acc, fn) => fn(acc), initial as any);
}

// Usage example: Text processing pipeline
const processText = pipe<string>(
    (s: string) => s.trim(),
    (s: string) => s.toLowerCase(),
    (s: string) => s.replace(/[^\w\s]/g, ""),
    (s: string) => s.replace(/\s+/g, "-"),
);

processText("  Hello, World!  ");  // → "hello-world"

// Type-safe pipe (TypeScript 5.0+ overloads)
function typedPipe<A, B>(f1: (a: A) => B): (a: A) => B;
function typedPipe<A, B, C>(f1: (a: A) => B, f2: (b: B) => C): (a: A) => C;
function typedPipe<A, B, C, D>(
    f1: (a: A) => B, f2: (b: B) => C, f3: (c: C) => D
): (a: A) => D;
function typedPipe<A, B, C, D, E>(
    f1: (a: A) => B, f2: (b: B) => C, f3: (c: C) => D, f4: (d: D) => E
): (a: A) => E;
function typedPipe(...fns: Array<(arg: any) => any>) {
    return (initial: any) => fns.reduce((acc, fn) => fn(acc), initial);
}

// Composed with type safety
const processUser = typedPipe(
    (user: User) => user.name,          // User → string
    (name: string) => name.toUpperCase(), // string → string
    (name: string) => name.length,       // string → number
);
// Type of processUser: (user: User) => number
```

### 6.2 Point-Free Style

```typescript
// Point-free: A style of composing functions without explicitly mentioning arguments

// Pointed (normal style)
const getActiveUserNames1 = (users: User[]) =>
    users
        .filter(u => u.active)
        .map(u => u.name);

// Near point-free style
const isActive = (u: User) => u.active;
const getName = (u: User) => u.name;

const getActiveUserNames2 = (users: User[]) =>
    users.filter(isActive).map(getName);

// Point-free with helper functions
const filter = <T>(pred: (item: T) => boolean) => (arr: T[]) =>
    arr.filter(pred);
const map = <T, U>(fn: (item: T) => U) => (arr: T[]) =>
    arr.map(fn);

const getActiveUserNames3 = pipe<User[]>(
    filter(isActive),
    map(getName),
);

// Ramda-style function composition
// npm install ramda @types/ramda
// import * as R from "ramda";
// const getActiveUserNames = R.pipe(
//     R.filter(R.prop("active")),
//     R.map(R.prop("name")),
// );
```

### 6.3 Function Composition in Haskell (Reference)

```haskell
-- Haskell: The home of function composition

-- (.) operator: Function composition
-- (f . g) x = f (g x)

-- Point-free style comes naturally
toSlug :: String -> String
toSlug = map toLower . filter isAlphaNum . words . unwords

-- Pipe operator (&)
-- x & f = f x
result = [1,2,3,4,5]
    & filter even     -- [2, 4]
    & map (* 2)       -- [4, 8]
    & sum              -- 12

-- Basic higher-order functions
map :: (a -> b) -> [a] -> [b]
filter :: (a -> Bool) -> [a] -> [a]
foldl :: (b -> a -> b) -> b -> [a] -> b
foldr :: (a -> b -> b) -> b -> [a] -> b

-- Higher-order function composition example
wordCount :: String -> [(String, Int)]
wordCount =
    map (\ws -> (head ws, length ws))  -- Convert groups to (word, count)
    . group                             -- Group identical words
    . sort                              -- Sort
    . words                             -- Split into words
```

---

## 7. flatMap (bind / chain)

### 7.1 Basic Concept

```typescript
// flatMap: map + flatten (flatten nested collections)
const sentences = ["hello world", "foo bar baz"];

// map produces nesting
sentences.map(s => s.split(" "));
// → [["hello", "world"], ["foo", "bar", "baz"]]

// flatMap flattens
sentences.flatMap(s => s.split(" "));
// → ["hello", "world", "foo", "bar", "baz"]

// flatMap in Option/Result (monadic bind)
// Promise.then is equivalent to flatMap
fetch("/api/user")
    .then(res => res.json())        // Response → Promise<JSON> (flatMap)
    .then(user => fetch(`/api/posts/${user.id}`))  // JSON → Promise
    .then(res => res.json());
```

### 7.2 Practical Use of flatMap

```typescript
// Expanding one-to-many relationships
interface Department {
    name: string;
    members: string[];
}

const departments: Department[] = [
    { name: "Engineering", members: ["Alice", "Bob", "Charlie"] },
    { name: "Design", members: ["Diana", "Eve"] },
    { name: "Marketing", members: ["Frank"] },
];

// List of all members
const allMembers = departments.flatMap(d => d.members);
// → ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]

// Member-department name pairs
const memberDepts = departments.flatMap(d =>
    d.members.map(m => ({ member: m, department: d.name }))
);
// → [
//     { member: "Alice", department: "Engineering" },
//     { member: "Bob", department: "Engineering" },
//     ...
// ]

// Complete flattening of nested arrays
const fullyFlat = deepNested.flat(Infinity); // [1, 2, 3, 4, 5, 6]
// Or recursive flatMap
function deepFlatten<T>(arr: (T | T[])[]): T[] {
    return arr.flatMap(item =>
        Array.isArray(item) ? deepFlatten(item) : [item]
    );
}

// Generating permutations
function permutations<T>(items: T[]): T[][] {
    if (items.length <= 1) return [items];
    return items.flatMap((item, i) => {
        const rest = [...items.slice(0, i), ...items.slice(i + 1)];
        return permutations(rest).map(perm => [item, ...perm]);
    });
}
permutations([1, 2, 3]);
// → [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]

// Generating combinations
function combinations<T>(items: T[], k: number): T[][] {
    if (k === 0) return [[]];
    if (items.length === 0) return [];
    const [first, ...rest] = items;
    const withFirst = combinations(rest, k - 1).map(c => [first, ...c]);
    const withoutFirst = combinations(rest, k);
    return [...withFirst, ...withoutFirst];
}
```

### 7.3 flatMap as a Monad

```typescript
// The essence of monads: Chaining computations via flatMap (bind / chain)

// Simple Maybe monad implementation
class Maybe<T> {
    private constructor(private value: T | null) {}

    static of<T>(value: T): Maybe<T> {
        return new Maybe(value);
    }

    static nothing<T>(): Maybe<T> {
        return new Maybe<T>(null);
    }

    static fromNullable<T>(value: T | null | undefined): Maybe<T> {
        return value == null ? Maybe.nothing() : Maybe.of(value);
    }

    map<U>(fn: (value: T) => U): Maybe<U> {
        if (this.value === null) return Maybe.nothing();
        return Maybe.of(fn(this.value));
    }

    flatMap<U>(fn: (value: T) => Maybe<U>): Maybe<U> {
        if (this.value === null) return Maybe.nothing();
        return fn(this.value);
    }

    getOrElse(defaultValue: T): T {
        return this.value ?? defaultValue;
    }

    toString(): string {
        return this.value === null ? "Nothing" : `Just(${this.value})`;
    }
}

// Usage example: Safe property access chain
interface Config {
    database?: {
        connection?: {
            host?: string;
            port?: number;
        };
    };
}

function getDbHost(config: Config): string {
    return Maybe.fromNullable(config.database)
        .flatMap(db => Maybe.fromNullable(db.connection))
        .flatMap(conn => Maybe.fromNullable(conn.host))
        .getOrElse("localhost");
}

// Simple Result monad implementation
type Result<T, E> = { ok: true; value: T } | { ok: false; error: E };

const Ok = <T>(value: T): Result<T, never> => ({ ok: true, value });
const Err = <E>(error: E): Result<never, E> => ({ ok: false, error });

function mapResult<T, U, E>(
    result: Result<T, E>,
    fn: (value: T) => U
): Result<U, E> {
    return result.ok ? Ok(fn(result.value)) : result;
}

function flatMapResult<T, U, E>(
    result: Result<T, E>,
    fn: (value: T) => Result<U, E>
): Result<U, E> {
    return result.ok ? fn(result.value) : result;
}

// Usage example: Validation chain
function parseAge(input: string): Result<number, string> {
    const age = parseInt(input, 10);
    if (isNaN(age)) return Err(`"${input}" is not a number`);
    if (age < 0 || age > 150) return Err(`Age ${age} is out of range`);
    return Ok(age);
}

function parseName(input: string): Result<string, string> {
    const trimmed = input.trim();
    if (trimmed.length === 0) return Err("Name cannot be empty");
    if (trimmed.length > 100) return Err("Name is too long");
    return Ok(trimmed);
}

interface ValidatedUser {
    name: string;
    age: number;
}

function validateUser(
    nameInput: string,
    ageInput: string
): Result<ValidatedUser, string> {
    const nameResult = parseName(nameInput);
    if (!nameResult.ok) return nameResult;

    const ageResult = parseAge(ageInput);
    if (!ageResult.ok) return ageResult;

    return Ok({ name: nameResult.value, age: ageResult.value });
}
```

---

## 8. Advanced Higher-Order Function Patterns

### 8.1 Transducers

```typescript
// Transducers: Compose transformations without generating intermediate arrays

// Normal chaining (intermediate arrays are generated at each step)
const result1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    .filter(n => n % 2 === 0)  // [2, 4, 6, 8, 10] ← intermediate array
    .map(n => n * 3)           // [6, 12, 18, 24, 30] ← intermediate array
    .filter(n => n > 10);      // [12, 18, 24, 30]

// Transducer type definitions
type Reducer<Acc, T> = (acc: Acc, item: T) => Acc;
type Transducer<T, U> = <Acc>(reducer: Reducer<Acc, U>) => Reducer<Acc, T>;

// map transducer
function tMap<T, U>(fn: (item: T) => U): Transducer<T, U> {
    return <Acc>(reducer: Reducer<Acc, U>): Reducer<Acc, T> =>
        (acc: Acc, item: T) => reducer(acc, fn(item));
}

// filter transducer
function tFilter<T>(pred: (item: T) => boolean): Transducer<T, T> {
    return <Acc>(reducer: Reducer<Acc, T>): Reducer<Acc, T> =>
        (acc: Acc, item: T) => pred(item) ? reducer(acc, item) : acc;
}

// Transducer composition
function tCompose<A, B, C>(
    t1: Transducer<A, B>,
    t2: Transducer<B, C>
): Transducer<A, C> {
    return <Acc>(reducer: Reducer<Acc, C>): Reducer<Acc, A> =>
        t1(t2(reducer));
}

// Usage example
const xform = tCompose(
    tCompose(
        tFilter<number>(n => n % 2 === 0),
        tMap<number, number>(n => n * 3)
    ),
    tFilter<number>(n => n > 10)
);

const arrayAppend: Reducer<number[], number> = (acc, item) => {
    acc.push(item);
    return acc;
};

const result2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    .reduce(xform(arrayAppend), []);
// → [12, 18, 24, 30] (no intermediate arrays, completed in a single pass)
```

### 8.2 Lenses

```typescript
// Lenses: Make access and updates to deep properties of immutable data structures composable

interface Lens<S, A> {
    get: (s: S) => A;
    set: (a: A, s: S) => S;
}

// Lens creation
function lens<S, A>(
    get: (s: S) => A,
    set: (a: A, s: S) => S
): Lens<S, A> {
    return { get, set };
}

// Modification through a lens
function over<S, A>(l: Lens<S, A>, fn: (a: A) => A, s: S): S {
    return l.set(fn(l.get(s)), s);
}

// Lens composition
function composeLens<A, B, C>(
    outer: Lens<A, B>,
    inner: Lens<B, C>
): Lens<A, C> {
    return {
        get: (a: A) => inner.get(outer.get(a)),
        set: (c: C, a: A) => outer.set(inner.set(c, outer.get(a)), a),
    };
}

// Usage example
interface Address {
    street: string;
    city: string;
    zipCode: string;
}

interface Person {
    name: string;
    age: number;
    address: Address;
}

// Lens definitions
const addressLens: Lens<Person, Address> = lens(
    p => p.address,
    (a, p) => ({ ...p, address: a })
);

const cityLens: Lens<Address, string> = lens(
    a => a.city,
    (c, a) => ({ ...a, city: c })
);

// Lens composition
const personCityLens = composeLens(addressLens, cityLens);

const alice: Person = {
    name: "Alice",
    age: 30,
    address: { street: "123 Main St", city: "Tokyo", zipCode: "100-0001" },
};

// Read
personCityLens.get(alice);  // → "Tokyo"

// Update (immutable - returns a new object)
const aliceInOsaka = personCityLens.set("Osaka", alice);
// → { name: "Alice", age: 30, address: { ...address, city: "Osaka" } }

// over: Apply a function to the existing value
const aliceUpperCity = over(personCityLens, city => city.toUpperCase(), alice);
// → { name: "Alice", age: 30, address: { ...address, city: "TOKYO" } }
```

### 8.3 Continuation-Passing Style (CPS)

```typescript
// CPS: A style where results are passed to callbacks

// Direct style
function addDirect(a: number, b: number): number {
    return a + b;
}

// CPS (Continuation-Passing Style)
function addCPS(a: number, b: number, k: (result: number) => void): void {
    k(a + b);
}

// Complex computation chain in CPS
function factorialCPS(n: number, k: (result: number) => void): void {
    if (n <= 1) {
        k(1);
    } else {
        factorialCPS(n - 1, (result) => k(n * result));
    }
}

factorialCPS(5, (result) => console.log(result));  // → 120

// Practical CPS example: Chaining async operations (the origin of callback hell)
function readFileCPS(
    path: string,
    onSuccess: (data: string) => void,
    onError: (err: Error) => void
): void {
    // Async file read
    try {
        const data = "file content"; // In practice, this would be an FS operation
        onSuccess(data);
    } catch (e) {
        onError(e as Error);
    }
}

// Converting CPS to Promise
function cpsToPromise<T>(
    fn: (onSuccess: (value: T) => void, onError: (err: Error) => void) => void
): Promise<T> {
    return new Promise((resolve, reject) => {
        fn(resolve, reject);
    });
}

// This shows that Promises are a structured, standardized form of CPS
const readFilePromise = (path: string) =>
    cpsToPromise<string>((resolve, reject) =>
        readFileCPS(path, resolve, reject)
    );
```

---

## 9. Performance and Optimization

### 9.1 Lazy Evaluation

```typescript
// Lazy iterators: Compute only the elements that are needed

function* lazyMap<T, U>(iterable: Iterable<T>, fn: (item: T) => U): Generator<U> {
    for (const item of iterable) {
        yield fn(item);
    }
}

function* lazyFilter<T>(iterable: Iterable<T>, pred: (item: T) => boolean): Generator<T> {
    for (const item of iterable) {
        if (pred(item)) yield item;
    }
}

function* lazyTake<T>(iterable: Iterable<T>, n: number): Generator<T> {
    let count = 0;
    for (const item of iterable) {
        if (count >= n) return;
        yield item;
        count++;
    }
}

// Get the squares of the first 5 even numbers from an infinite list
function* naturals(): Generator<number> {
    let n = 1;
    while (true) yield n++;
}

const result = [...lazyTake(
    lazyMap(
        lazyFilter(naturals(), n => n % 2 === 0),
        n => n * n
    ),
    5
)];
// → [4, 16, 36, 64, 100]
// Only the needed elements are computed even from an infinite list

// Pipeline API (proposal stage)
// TC39 Pipeline Operator Proposal
// value |> fn1 |> fn2 |> fn3
// ↓ Current alternative
const pipeValue = <T>(value: T) => ({
    pipe: <U>(fn: (v: T) => U) => pipeValue(fn(value)),
    value,
});

const finalResult = pipeValue(10)
    .pipe(n => n * 2)
    .pipe(n => n + 5)
    .pipe(n => n.toString())
    .value;
// → "25"
```

### 9.2 Performance Comparison of Chains

```typescript
// Higher-order functions vs for loops with large datasets

// Test data
const largeArray = Array.from({ length: 1_000_000 }, (_, i) => i);

// Method 1: Chaining (new array generated at each step)
console.time("chain");
const r1 = largeArray
    .filter(n => n % 2 === 0)    // Generate 500,000-element array
    .map(n => n * 3)              // Generate 500,000-element array
    .filter(n => n > 100_000)     // Generate yet another array
    .reduce((sum, n) => sum + n, 0);
console.timeEnd("chain");

// Method 2: for loop (no array generation)
console.time("loop");
let r2 = 0;
for (let i = 0; i < largeArray.length; i++) {
    const n = largeArray[i];
    if (n % 2 === 0) {
        const tripled = n * 3;
        if (tripled > 100_000) {
            r2 += tripled;
        }
    }
}
console.timeEnd("loop");

// Method 3: Single reduce (no intermediate arrays)
console.time("single-reduce");
const r3 = largeArray.reduce((sum, n) => {
    if (n % 2 === 0) {
        const tripled = n * 3;
        if (tripled > 100_000) {
            return sum + tripled;
        }
    }
    return sum;
}, 0);
console.timeEnd("single-reduce");

// Method 4: Generators (lazy evaluation)
console.time("generator");
const r4 = [...lazyFilter(
    lazyMap(
        lazyFilter(
            largeArray,
            n => n % 2 === 0
        ),
        n => n * 3
    ),
    n => n > 100_000
)].reduce((sum, n) => sum + n, 0);
console.timeEnd("generator");

// Approximate measurements (environment-dependent):
// chain:          ~80ms  (readability: high, memory: large)
// loop:           ~15ms  (readability: low, memory: small)
// single-reduce:  ~20ms  (readability: medium, memory: small)
// generator:      ~120ms (readability: medium, memory: small)
//
// Conclusion:
// - For normal data sizes (<10,000), chaining is sufficient
// - For large data, use single reduce or for loops
// - For memory constraints, use generators
```

### 9.3 Zero-Cost Abstractions in Rust

```rust
// Rust's iterator chains are zero-cost abstractions
// They are optimized at compile time to code equivalent to for loops

fn benchmark_rust() {
    let numbers: Vec<i32> = (0..1_000_000).collect();

    // Higher-order function chain (zero-cost: same performance as for loop)
    let result: i64 = numbers.iter()
        .filter(|&&n| n % 2 == 0)
        .map(|&n| n as i64 * 3)
        .filter(|&n| n > 100_000)
        .sum();

    // This is optimized to nearly the same machine code as the following for loop
    let mut result2: i64 = 0;
    for &n in &numbers {
        if n % 2 == 0 {
            let tripled = n as i64 * 3;
            if tripled > 100_000 {
                result2 += tripled;
            }
        }
    }

    // SIMD optimizations may also be applied
    // Rust iterators:
    // 1. Do not generate intermediate collections (lazy evaluation)
    // 2. Apply the entire pipeline to each element
    // 3. No overhead due to inlining
    assert_eq!(result, result2);
}
```

---

## 10. Commonly Used Patterns in Practice

### 10.1 Validation Composition

```typescript
// Composing validation functions
type Validator<T> = (value: T) => string | null;

function composeValidators<T>(...validators: Validator<T>[]): Validator<T> {
    return (value: T) => {
        for (const validate of validators) {
            const error = validate(value);
            if (error) return error;
        }
        return null;
    };
}

function collectErrors<T>(...validators: Validator<T>[]): (value: T) => string[] {
    return (value: T) =>
        validators
            .map(v => v(value))
            .filter((err): err is string => err !== null);
}

// Validator definitions
const required: Validator<string> = (v) =>
    v.trim().length === 0 ? "This field is required" : null;

const minLength = (min: number): Validator<string> => (v) =>
    v.length < min ? `Must be at least ${min} characters` : null;

const maxLength = (max: number): Validator<string> => (v) =>
    v.length > max ? `Must be at most ${max} characters` : null;

const pattern = (regex: RegExp, message: string): Validator<string> => (v) =>
    regex.test(v) ? null : message;

const email = pattern(
    /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
    "Please enter a valid email address"
);

// Compose and use
const validateEmail = composeValidators(
    required,
    email,
    maxLength(254),
);

const validatePassword = composeValidators(
    required,
    minLength(8),
    maxLength(128),
    pattern(/[A-Z]/, "Must include an uppercase letter"),
    pattern(/[a-z]/, "Must include a lowercase letter"),
    pattern(/[0-9]/, "Must include a digit"),
);

// Collect all errors
const allPasswordErrors = collectErrors(
    required,
    minLength(8),
    pattern(/[A-Z]/, "Must include an uppercase letter"),
    pattern(/[a-z]/, "Must include a lowercase letter"),
    pattern(/[0-9]/, "Must include a digit"),
);

console.log(allPasswordErrors("abc"));
// → ["Must be at least 8 characters", "Must include an uppercase letter", "Must include a digit"]
```

### 10.2 Event Handler Composition

```typescript
// Event handler composition in React
type EventHandler<E> = (event: E) => void;

function combineHandlers<E>(...handlers: EventHandler<E>[]): EventHandler<E> {
    return (event: E) => {
        handlers.forEach(handler => handler(event));
    };
}

function conditionalHandler<E>(
    pred: (event: E) => boolean,
    handler: EventHandler<E>
): EventHandler<E> {
    return (event: E) => {
        if (pred(event)) handler(event);
    };
}

// Usage example
const logClick: EventHandler<MouseEvent> = (e) =>
    console.log(`Clicked at (${e.clientX}, ${e.clientY})`);

const trackAnalytics: EventHandler<MouseEvent> = (e) =>
    console.log("Analytics: button clicked");

const preventAndHandle: EventHandler<MouseEvent> = (e) => {
    e.preventDefault();
    // Processing...
};

const handleClick = combineHandlers(
    logClick,
    trackAnalytics,
    conditionalHandler(
        (e) => e.ctrlKey,
        (e) => console.log("Ctrl+Click detected!")
    ),
);
```

### 10.3 Functional Conditional Branching

```typescript
// match pattern: Functional version of switch statements
function match<T, R>(value: T, cases: Array<[((v: T) => boolean) | T, R]>, defaultValue: R): R {
    for (const [condition, result] of cases) {
        if (typeof condition === "function") {
            if ((condition as (v: T) => boolean)(value)) return result;
        } else if (condition === value) {
            return result;
        }
    }
    return defaultValue;
}

// Usage example
const httpStatus = (code: number): string =>
    match(code, [
        [200, "OK"],
        [201, "Created"],
        [400, "Bad Request"],
        [401, "Unauthorized"],
        [403, "Forbidden"],
        [404, "Not Found"],
        [(c: number) => c >= 500, "Server Error"],
    ], "Unknown");

// Pattern matching-style branching
type Shape =
    | { type: "circle"; radius: number }
    | { type: "rectangle"; width: number; height: number }
    | { type: "triangle"; base: number; height: number };

const area = (shape: Shape): number => {
    const handlers: Record<string, (s: any) => number> = {
        circle: (s) => Math.PI * s.radius ** 2,
        rectangle: (s) => s.width * s.height,
        triangle: (s) => 0.5 * s.base * s.height,
    };
    return handlers[shape.type](shape);
};
```

### 10.4 State Management (Redux Pattern)

```typescript
// Redux: State management with higher-order functions

type Action = { type: string; payload?: any };
type Reducer<S> = (state: S, action: Action) => S;
type Middleware<S> = (
    store: { getState: () => S; dispatch: (action: Action) => void }
) => (next: (action: Action) => void) => (action: Action) => void;

// Store creation (a parade of higher-order functions)
function createStore<S>(
    reducer: Reducer<S>,
    initialState: S,
    ...middlewares: Middleware<S>[]
) {
    let state = initialState;
    const listeners: Array<() => void> = [];

    const getState = () => state;

    const subscribe = (listener: () => void) => {
        listeners.push(listener);
        return () => {
            const index = listeners.indexOf(listener);
            if (index > -1) listeners.splice(index, 1);
        };
    };

    let dispatch = (action: Action) => {
        state = reducer(state, action);
        listeners.forEach(l => l());
    };

    // Apply middleware (compose from right to left)
    const store = { getState, dispatch };
    const chain = middlewares.map(mw => mw(store));
    dispatch = chain.reduceRight(
        (next, mw) => mw(next),
        dispatch
    );

    return { getState, dispatch, subscribe };
}

// Reducer composition
function combineReducers<S extends Record<string, any>>(
    reducers: { [K in keyof S]: Reducer<S[K]> }
): Reducer<S> {
    return (state: S, action: Action): S => {
        const nextState = {} as S;
        let hasChanged = false;
        for (const key in reducers) {
            const prevStateForKey = state[key];
            nextState[key] = reducers[key](prevStateForKey, action);
            if (nextState[key] !== prevStateForKey) {
                hasChanged = true;
            }
        }
        return hasChanged ? nextState : state;
    };
}

// Logger middleware
const loggerMiddleware: Middleware<any> = (store) => (next) => (action) => {
    console.log("dispatching:", action.type);
    console.log("before:", store.getState());
    next(action);
    console.log("after:", store.getState());
};

// Thunk middleware (async action support)
const thunkMiddleware: Middleware<any> = (store) => (next) => (action: any) => {
    if (typeof action === "function") {
        return action(store.dispatch, store.getState);
    }
    return next(action);
};
```

---

## 11. Cross-Language Comparison Table

```
┌──────────────┬────────────┬──────────┬──────────┬──────────┬──────────┐
│ Concept       │ TypeScript │ Python   │ Rust     │ Go       │ Haskell  │
├──────────────┼────────────┼──────────┼──────────┼──────────┼──────────┤
│ map          │ .map()     │ map()    │ .map()   │ Custom   │ map      │
│ filter       │ .filter()  │ filter() │ .filter()│ Custom   │ filter   │
│ reduce/fold  │ .reduce()  │ reduce() │ .fold()  │ Custom   │ foldl    │
│ flatMap      │ .flatMap() │ chain()  │ .flat_map│ Custom   │ >>=      │
│ Composition  │ Manual     │ Manual   │ Manual   │ Manual   │ (.)      │
│ Currying     │ Manual     │ partial  │ Closures │ Closures │ Auto     │
│ Pipe         │ Proposed   │ -        │ -        │ -        │ & / $    │
│              │ (|>)       │          │          │          │          │
│ Lazy eval    │ Generator  │ Iterator │ Iterator │ -        │ Default  │
│ Pattern match│ switch/if  │ match    │ match    │ switch   │ case     │
│ Closures     │ Arrow Fn   │ lambda   │ |x| expr │ func lit │ \x->expr │
│ Decorators   │ @experim.  │ @standard│ Macros   │ -        │ HoF      │
│ Zero-cost    │ No         │ No       │ Yes      │ Partial  │ No       │
└──────────────┴────────────┴──────────┴──────────┴──────────┴──────────┘
```

---

## 12. Anti-Patterns and Caveats

### 12.1 Patterns to Avoid

```typescript
// Anti-pattern 1: map with side effects
// BAD: Executing side effects in map
users.map(u => {
    sendEmail(u.email);  // Side effect!
    return u.name;
});

// GOOD: Separate side effects with forEach, transformation with map
users.forEach(u => sendEmail(u.email));
const names = users.map(u => u.name);

// Anti-pattern 2: Unnecessary intermediate arrays
// BAD: Generating many intermediate arrays with large data
const result = hugeArray
    .map(transform1)    // Intermediate array 1
    .filter(predicate1) // Intermediate array 2
    .map(transform2)    // Intermediate array 3
    .filter(predicate2) // Intermediate array 4
    .reduce(aggregate, init);

// GOOD: Process in a single reduce
const result2 = hugeArray.reduce((acc, item) => {
    const t1 = transform1(item);
    if (!predicate1(t1)) return acc;
    const t2 = transform2(t1);
    if (!predicate2(t2)) return acc;
    return aggregate(acc, t2);
}, init);

// Anti-pattern 3: Excessive point-free style
// BAD: Unreadable
const processData = compose(
    sortBy(prop("score")),
    map(over(lensProp("name"), toUpper)),
    filter(both(propSatisfies(gt(__, 18), "age"), prop("active"))),
    uniqBy(prop("id")),
);

// GOOD: Name things appropriately
const isEligible = (u: User) => u.age > 18 && u.active;
const normalizeName = (u: User) => ({ ...u, name: u.name.toUpperCase() });
const processData = (users: User[]) =>
    users
        .filter(isEligible)
        .map(normalizeName)
        .sort((a, b) => b.score - a.score);

// Anti-pattern 4: Overuse of reduce
// BAD: Using reduce when find would suffice
const firstEven = numbers.reduce<number | null>(
    (found, n) => found !== null ? found : (n % 2 === 0 ? n : null),
    null
);

// GOOD: Use find
const firstEven2 = numbers.find(n => n % 2 === 0);

// Anti-pattern 5: Deep nesting
// BAD: Higher-order function nesting is too deep
const result3 = arr1.flatMap(a =>
    arr2.filter(b => b.id === a.id).map(b =>
        arr3.filter(c => c.key === b.key).map(c => ({
            ...a, ...b, ...c
        }))
    )
);

// GOOD: Split processing and name things
const lookup2 = new Map(arr2.map(b => [b.id, b]));
const lookup3 = new Map(arr3.map(c => [c.key, c]));
const result4 = arr1
    .filter(a => lookup2.has(a.id))
    .map(a => {
        const b = lookup2.get(a.id)!;
        const c = lookup3.get(b.key);
        return c ? { ...a, ...b, ...c } : null;
    })
    .filter(Boolean);
```

### 12.2 Higher-Order Function Selection Guide

```
What you want to do                → Higher-order function to use
─────────────────────────────────────────────────
Transform each element             → map
Select elements matching a condition→ filter
Aggregate all elements into one    → reduce
Get first element matching condition→ find
Check if any element matches       → some
Check if all elements match        → every
Flatten nested arrays + transform  → flatMap
Execute side effects on each element→ forEach
Split into two groups by condition → partition (custom or lodash)
Group by key                       → groupBy (Object.groupBy / reduce)
Remove duplicates                  → [...new Set(arr)] / reduce
Sort                               → sort (pass a comparison function)
```

---


## FAQ

### Q1: What is the most important point when learning this topic?

Gaining practical experience is the most important aspect. Understanding deepens not just through theory but by actually writing code and verifying how it works.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this used in practice?

Knowledge of this topic is frequently applied in everyday development work. It is particularly important during code reviews and architecture design.

---

## Summary

| Higher-Order Function | Type Signature | Purpose |
|---------|------------|------|
| map | (A->B) -> [A] -> [B] | Element transformation |
| filter | (A->Bool) -> [A] -> [A] | Element selection |
| reduce | (B,A->B) -> B -> [A] -> B | Aggregation |
| flatMap | (A->[B]) -> [A] -> [B] | Transform + flatten |
| compose | (B->C, A->B) -> (A->C) | Function composition |
| curry | ((A,B)->C) -> A -> B -> C | Currying |
| partial | ((A,B)->C, A) -> (B->C) | Partial application |

Principles for using higher-order functions effectively:

1. **Appropriate level of abstraction**: Use map/filter/find for simple operations, reduce for complex aggregations
2. **Prioritize readability**: Use appropriately named functions over point-free or excessive chaining
3. **Be performance-aware**: Suppress intermediate array generation for large datasets
4. **Separate side effects**: Use pure functions in map/filter/reduce, forEach for side effects
5. **Type safety**: Leverage TypeScript generics to enable type inference
6. **Testability**: Split into small pure functions that can be tested individually

---

## Recommended Next Guides

---

## References
1. Bird, R. "Thinking Functionally with Haskell." Cambridge, 2014.
2. Fogus, M. "Functional JavaScript." O'Reilly, 2013.
3. Frisby, B. "Professor Frisby's Mostly Adequate Guide to Functional Programming." 2015.
4. Chiusano, P. & Bjarnason, R. "Functional Programming in Scala." Manning, 2014.
5. Hutton, G. "Programming in Haskell." Cambridge, 2016.
6. Kleppmann, M. "Designing Data-Intensive Applications." O'Reilly, 2017.
7. Mozilla Developer Network. "Array.prototype.map/filter/reduce." MDN Web Docs.
8. Rust Documentation. "Iterator trait." doc.rust-lang.org.
