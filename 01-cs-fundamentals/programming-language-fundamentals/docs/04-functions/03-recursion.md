# Recursion

> Recursion is a technique of "decomposing a problem into smaller problems of the same structure." It is mathematically elegant and indispensable for trees, graphs, and divide-and-conquer approaches.

## What You Will Learn in This Chapter

- [ ] Understand the mechanics and basic patterns of recursion
- [ ] Understand tail recursion and its optimization
- [ ] Be able to determine when to use recursion vs. loops
- [ ] Practice optimization of recursion through memoization
- [ ] Understand various divide-and-conquer algorithms
- [ ] Implement backtracking and search algorithms
- [ ] Understand trampolines and CPS as alternatives to tail call optimization


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding the content of [Higher-Order Functions](./02-higher-order-functions.md)

---

## 1. Recursion Basics

### 1.1 What Is a Recursive Function?

```
Recursive function = A function that calls itself

Components:
  1. Base case (termination condition): The condition that stops the recursion
  2. Recursive step: Makes the problem smaller and calls itself

Essential rules:
  - Must always reach the base case (to prevent infinite recursion)
  - The problem must get smaller with each recursive call (must converge)
```

```
Visualization of recursive calls:

  factorial(5)
    ├── 5 * factorial(4)
    │       ├── 4 * factorial(3)
    │       │       ├── 3 * factorial(2)
    │       │       │       ├── 2 * factorial(1)
    │       │       │       │       └── return 1  ← base case
    │       │       │       └── return 2 * 1 = 2
    │       │       └── return 3 * 2 = 6
    │       └── return 4 * 6 = 24
    └── return 5 * 24 = 120
```

### 1.2 Basic Recursion in Python

```python
# Factorial: n! = n * (n-1)!
def factorial(n):
    if n <= 1:          # Base case
        return 1
    return n * factorial(n - 1)  # Recursive step

# Unwinding the call stack:
# factorial(5)
#   → 5 * factorial(4)
#     → 4 * factorial(3)
#       → 3 * factorial(2)
#         → 2 * factorial(1)
#           → 1 (base case)
#         → 2 * 1 = 2
#       → 3 * 2 = 6
#     → 4 * 6 = 24
#   → 5 * 24 = 120

# Sum of natural numbers: sum(n) = n + sum(n-1)
def sum_natural(n):
    if n <= 0:
        return 0
    return n + sum_natural(n - 1)

# Exponentiation: power(base, exp) = base * power(base, exp-1)
def power(base, exp):
    if exp == 0:
        return 1
    if exp < 0:
        return 1 / power(base, -exp)
    return base * power(base, exp - 1)

# Greatest common divisor (Euclidean algorithm)
def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)

# String reversal
def reverse_string(s):
    if len(s) <= 1:
        return s
    return reverse_string(s[1:]) + s[0]

# Palindrome check
def is_palindrome(s):
    if len(s) <= 1:
        return True
    if s[0] != s[-1]:
        return False
    return is_palindrome(s[1:-1])
```

### 1.3 Basic Recursion in TypeScript

```typescript
// Factorial
function factorial(n: number): number {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

// Fibonacci sequence (naive recursion - inefficient)
function fibonacci(n: number): number {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

// Binary search (recursive version)
function binarySearch(
    arr: number[],
    target: number,
    lo: number = 0,
    hi: number = arr.length - 1
): number {
    if (lo > hi) return -1;  // Base case: not found
    const mid = Math.floor((lo + hi) / 2);
    if (arr[mid] === target) return mid;  // Base case: found
    if (arr[mid] < target) {
        return binarySearch(arr, target, mid + 1, hi);
    }
    return binarySearch(arr, target, lo, mid - 1);
}

// All subsets of a string (power set)
function powerSet(s: string): string[] {
    if (s.length === 0) return [""];
    const first = s[0];
    const rest = powerSet(s.slice(1));
    return [...rest, ...rest.map(sub => first + sub)];
}

powerSet("abc");
// → ["", "c", "b", "bc", "a", "ac", "ab", "abc"]
```

### 1.4 Call Stack Visualization

```
Growth and shrinkage of the call stack:

Changes to the stack when calling factorial(4):

  Step 1:  [factorial(4)]
  Step 2:  [factorial(4), factorial(3)]
  Step 3:  [factorial(4), factorial(3), factorial(2)]
  Step 4:  [factorial(4), factorial(3), factorial(2), factorial(1)]
  Step 5:  [factorial(4), factorial(3), factorial(2)]  ← returns 1
  Step 6:  [factorial(4), factorial(3)]                ← returns 2
  Step 7:  [factorial(4)]                              ← returns 6
  Step 8:  []                                          ← returns 24

Stack depth = n (for linear recursion)
Stack overflow: Occurs when depth exceeds the language's stack limit
  - Python: Default 1000 (changeable with sys.setrecursionlimit())
  - JavaScript: Engine-dependent (typically 10,000-25,000)
  - Java: Depends on thread stack size (default 512KB-1MB)
```

---

## 2. Recursion Patterns

### 2.1 Linear Recursion

```python
# Pattern 1: Linear recursion (list processing)
# Each call makes exactly one recursive call → O(n)

def sum_list(lst):
    if not lst:
        return 0
    return lst[0] + sum_list(lst[1:])

# List length
def length(lst):
    if not lst:
        return 0
    return 1 + length(lst[1:])

# Maximum value in a list
def maximum(lst):
    if len(lst) == 1:
        return lst[0]
    rest_max = maximum(lst[1:])
    return lst[0] if lst[0] > rest_max else rest_max

# List flattening
def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

flatten([1, [2, [3, 4], 5], [6, 7]])
# → [1, 2, 3, 4, 5, 6, 7]

# zip (recursive version)
def zip_lists(lst1, lst2):
    if not lst1 or not lst2:
        return []
    return [(lst1[0], lst2[0])] + zip_lists(lst1[1:], lst2[1:])
```

### 2.2 Binary Recursion (Divide and Conquer)

```python
# Pattern 2: Binary recursion (divide and conquer)
# Each call makes two recursive calls

# Merge sort
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Quicksort
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

# Fast exponentiation (divide and conquer): O(log n)
def fast_power(base, exp):
    if exp == 0:
        return 1
    if exp % 2 == 0:
        half = fast_power(base, exp // 2)
        return half * half
    else:
        return base * fast_power(base, exp - 1)

# Karatsuba multiplication (fast multiplication of large integers)
def karatsuba(x, y):
    if x < 10 or y < 10:
        return x * y

    n = max(len(str(x)), len(str(y)))
    half = n // 2
    power = 10 ** half

    a, b = divmod(x, power)  # x = a * 10^half + b
    c, d = divmod(y, power)  # y = c * 10^half + d

    # Only 3 multiplications needed (normally 4 are required)
    ac = karatsuba(a, c)
    bd = karatsuba(b, d)
    ad_bc = karatsuba(a + b, c + d) - ac - bd

    return ac * (10 ** (2 * half)) + ad_bc * (10 ** half) + bd
```

### 2.3 Tree Recursion

```python
# Pattern 3: Tree recursion
# Each call makes two or more recursive calls

# Fibonacci (naive version - O(2^n))
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
# Note: Exponential time complexity. Memoization is needed

# Fibonacci call tree:
# fib(5)
# ├── fib(4)
# │   ├── fib(3)
# │   │   ├── fib(2)
# │   │   │   ├── fib(1) → 1
# │   │   │   └── fib(0) → 0
# │   │   └── fib(1) → 1
# │   └── fib(2)
# │       ├── fib(1) → 1
# │       └── fib(0) → 0
# └── fib(3)
#     ├── fib(2)
#     │   ├── fib(1) → 1
#     │   └── fib(0) → 0
#     └── fib(1) → 1
# → The same computations are repeated many times

# Pascal's triangle
def pascal(row, col):
    if col == 0 or col == row:
        return 1
    return pascal(row - 1, col - 1) + pascal(row - 1, col)

# Display Pascal's triangle
def print_pascal(n):
    for row in range(n):
        values = [pascal(row, col) for col in range(row + 1)]
        print(" " * (n - row), " ".join(f"{v:3}" for v in values))

print_pascal(6)
#       1
#      1   1
#     1   2   1
#    1   3   3   1
#   1   4   6   4   1
#  1   5  10  10   5   1

# Number of combinations C(n, k) = C(n-1, k-1) + C(n-1, k)
def combinations_count(n, k):
    if k == 0 or k == n:
        return 1
    if k < 0 or k > n:
        return 0
    return combinations_count(n - 1, k - 1) + combinations_count(n - 1, k)
```

### 2.4 Mutual Recursion

```python
# Pattern 4: Mutual recursion
# Two or more functions call each other

def is_even(n):
    if n == 0: return True
    return is_odd(n - 1)

def is_odd(n):
    if n == 0: return False
    return is_even(n - 1)

# Mutual recursion in an expression parser
# expr   = term (('+' | '-') term)*
# term   = factor (('*' | '/') factor)*
# factor = number | '(' expr ')'

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self):
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def parse_expr(self):
        """expr = term (('+' | '-') term)*"""
        result = self.parse_term()
        while self.peek() in ('+', '-'):
            op = self.consume()
            right = self.parse_term()
            if op == '+':
                result += right
            else:
                result -= right
        return result

    def parse_term(self):
        """term = factor (('*' | '/') factor)*"""
        result = self.parse_factor()
        while self.peek() in ('*', '/'):
            op = self.consume()
            right = self.parse_factor()
            if op == '*':
                result *= right
            else:
                result /= right
        return result

    def parse_factor(self):
        """factor = number | '(' expr ')'"""
        if self.peek() == '(':
            self.consume()  # Consume '('
            result = self.parse_expr()  # Recursion: factor → expr → term → factor
            self.consume()  # Consume ')'
            return result
        return float(self.consume())

# Usage example
tokens = ['(', '2', '+', '3', ')', '*', '4']
parser = Parser(tokens)
print(parser.parse_expr())  # → 20.0
```

---

## 3. Recursive Processing of Tree Structures

### 3.1 File System Traversal

```typescript
// File system traversal
interface FSNode {
    name: string;
    type: "file" | "directory";
    children?: FSNode[];
    size?: number;
}

// Calculate total size
function totalSize(node: FSNode): number {
    if (node.type === "file") {
        return node.size ?? 0;  // Base case
    }
    return (node.children ?? [])
        .reduce((sum, child) => sum + totalSize(child), 0);
}

// Search for files (depth-first)
function findFiles(
    node: FSNode,
    predicate: (node: FSNode) => boolean
): FSNode[] {
    const results: FSNode[] = [];
    if (node.type === "file" && predicate(node)) {
        results.push(node);
    }
    if (node.children) {
        for (const child of node.children) {
            results.push(...findFiles(child, predicate));
        }
    }
    return results;
}

// String representation of a directory tree
function renderTree(node: FSNode, indent: string = "", isLast: boolean = true): string {
    const prefix = indent + (isLast ? "└── " : "├── ");
    const childIndent = indent + (isLast ? "    " : "│   ");
    let result = prefix + node.name + "\n";

    if (node.children) {
        node.children.forEach((child, i) => {
            result += renderTree(child, childIndent, i === node.children!.length - 1);
        });
    }
    return result;
}

// Usage example
const root: FSNode = {
    name: "project",
    type: "directory",
    children: [
        {
            name: "src",
            type: "directory",
            children: [
                { name: "index.ts", type: "file", size: 1024 },
                { name: "utils.ts", type: "file", size: 512 },
                {
                    name: "components",
                    type: "directory",
                    children: [
                        { name: "Header.tsx", type: "file", size: 2048 },
                        { name: "Footer.tsx", type: "file", size: 1536 },
                    ],
                },
            ],
        },
        { name: "README.md", type: "file", size: 256 },
    ],
};

console.log(totalSize(root));
// → 5376

console.log(findFiles(root, n => n.name.endsWith(".tsx")).map(n => n.name));
// → ["Header.tsx", "Footer.tsx"]

console.log(renderTree(root));
// └── project
//     ├── src
//     │   ├── index.ts
//     │   ├── utils.ts
//     │   └── components
//     │       ├── Header.tsx
//     │       └── Footer.tsx
//     └── README.md
```

### 3.2 Recursive Processing of JSON / Nested Objects

```typescript
// Get a deep value from JSON
function deepGet(obj: any, path: string[]): any {
    if (path.length === 0) return obj;
    if (obj == null) return undefined;
    const [head, ...tail] = path;
    return deepGet(obj[head], tail);
}

deepGet({ a: { b: { c: 42 } } }, ["a", "b", "c"]);  // → 42

// Deep merge
function deepMerge(target: any, source: any): any {
    if (typeof target !== "object" || typeof source !== "object") {
        return source;
    }
    if (Array.isArray(target) && Array.isArray(source)) {
        return [...target, ...source];
    }
    const result = { ...target };
    for (const key of Object.keys(source)) {
        if (key in result && typeof result[key] === "object" && typeof source[key] === "object") {
            result[key] = deepMerge(result[key], source[key]);
        } else {
            result[key] = source[key];
        }
    }
    return result;
}

// Deep comparison (deep equality)
function deepEqual(a: any, b: any): boolean {
    if (a === b) return true;
    if (a == null || b == null) return false;
    if (typeof a !== typeof b) return false;

    if (Array.isArray(a) && Array.isArray(b)) {
        if (a.length !== b.length) return false;
        return a.every((item, i) => deepEqual(item, b[i]));
    }

    if (typeof a === "object") {
        const keysA = Object.keys(a);
        const keysB = Object.keys(b);
        if (keysA.length !== keysB.length) return false;
        return keysA.every(key => deepEqual(a[key], b[key]));
    }

    return false;
}

// Deep clone
function deepClone<T>(obj: T): T {
    if (obj === null || typeof obj !== "object") return obj;
    if (obj instanceof Date) return new Date(obj.getTime()) as any;
    if (obj instanceof RegExp) return new RegExp(obj.source, obj.flags) as any;
    if (Array.isArray(obj)) return obj.map(item => deepClone(item)) as any;

    const cloned = {} as T;
    for (const key in obj) {
        if (Object.prototype.hasOwnProperty.call(obj, key)) {
            cloned[key] = deepClone(obj[key]);
        }
    }
    return cloned;
}

// Enumerate all paths in an object
function allPaths(obj: any, prefix: string = ""): string[] {
    if (typeof obj !== "object" || obj === null) {
        return [prefix];
    }
    return Object.entries(obj).flatMap(([key, value]) => {
        const path = prefix ? `${prefix}.${key}` : key;
        return allPaths(value, path);
    });
}

allPaths({ a: { b: 1, c: { d: 2 } }, e: 3 });
// → ["a.b", "a.c.d", "e"]

// Object flattening
function flattenObject(obj: any, prefix: string = ""): Record<string, any> {
    const result: Record<string, any> = {};
    for (const [key, value] of Object.entries(obj)) {
        const path = prefix ? `${prefix}.${key}` : key;
        if (typeof value === "object" && value !== null && !Array.isArray(value)) {
            Object.assign(result, flattenObject(value, path));
        } else {
            result[path] = value;
        }
    }
    return result;
}

flattenObject({ a: { b: 1, c: { d: 2 } }, e: 3 });
// → { "a.b": 1, "a.c.d": 2, "e": 3 }
```

### 3.3 Recursive Processing of DOM Trees

```typescript
// DOM tree traversal
function walkDOM(node: Node, callback: (node: Node) => void): void {
    callback(node);
    let child = node.firstChild;
    while (child) {
        walkDOM(child, callback);
        child = child.nextSibling;
    }
}

// Recursively search for elements matching a condition
function findElement(
    node: Element,
    predicate: (el: Element) => boolean
): Element | null {
    if (predicate(node)) return node;
    for (const child of Array.from(node.children)) {
        const found = findElement(child, predicate);
        if (found) return found;
    }
    return null;
}

// Recursive rendering of a React component tree
interface TreeItem {
    id: string;
    label: string;
    children?: TreeItem[];
}

// Recursive tree component
function TreeView({ items, depth = 0 }: { items: TreeItem[]; depth?: number }) {
    return (
        <ul style={{ paddingLeft: depth > 0 ? 20 : 0 }}>
            {items.map(item => (
                <li key={item.id}>
                    {item.label}
                    {item.children && item.children.length > 0 && (
                        <TreeView items={item.children} depth={depth + 1} />
                    )}
                </li>
            ))}
        </ul>
    );
}
```

---

## 4. Tail Recursion

### 4.1 What Is Tail Recursion?

```
Tail recursion = The recursive call is the last operation in the function

  Normal recursion: return n * factorial(n - 1)
                    ↑ Multiplies n by the result of recursion → n must be kept on the stack

  Tail recursion:   return factorial_tail(n - 1, acc * n)
                    ↑ Recursive call is the last operation → stack frame can be reused

Normal recursion stack:
  factorial(4)           ← Keep 4 on the stack
    factorial(3)         ← Keep 3 on the stack
      factorial(2)       ← Keep 2 on the stack
        factorial(1)     ← Base case
      return 2 * 1       ← Unwinding
    return 3 * 2
  return 4 * 6
→ Stack depth: O(n)

Tail recursion stack (with TCO):
  factorial_tail(4, 1)   → factorial_tail(3, 4)
                         → factorial_tail(2, 12)
                         → factorial_tail(1, 24)
                         → return 24
→ Stack depth: O(1) (stack frames are reused)
```

### 4.2 Converting to Tail Recursion

```python
# Normal recursion (stack grows O(n))
def factorial(n):
    if n <= 1: return 1
    return n * factorial(n - 1)

# Tail recursion (using accumulator)
def factorial_tail(n, acc=1):
    if n <= 1: return acc
    return factorial_tail(n - 1, acc * n)

# Python does NOT perform tail call optimization
# → Stack overflow for large n
# → Rewriting as a loop is recommended

# General pattern for converting to tail recursion:
# 1. Introduce an accumulator
# 2. Accumulate the computation result in the accumulator
# 3. Return the accumulator at the base case

# Example: List sum
# Normal recursion
def sum_list(lst):
    if not lst:
        return 0
    return lst[0] + sum_list(lst[1:])

# Tail recursion
def sum_list_tail(lst, acc=0):
    if not lst:
        return acc
    return sum_list_tail(lst[1:], acc + lst[0])

# Example: List reversal
# Normal recursion
def reverse_list(lst):
    if not lst:
        return []
    return reverse_list(lst[1:]) + [lst[0]]

# Tail recursion
def reverse_list_tail(lst, acc=None):
    if acc is None:
        acc = []
    if not lst:
        return acc
    return reverse_list_tail(lst[1:], [lst[0]] + acc)

# Example: Fibonacci
# Normal recursion (tree recursion - O(2^n))
def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)

# Tail recursion (linear - O(n))
def fib_tail(n, a=0, b=1):
    if n == 0:
        return a
    return fib_tail(n - 1, b, a + b)
```

### 4.3 Languages That Support TCO

```
TCO supported:  Scheme, Haskell, Elixir/Erlang, Scala(@tailrec)
TCO limited:    JavaScript (strict mode, implementation-dependent)
No TCO:         Python, Java, Go, Rust (not explicitly used)

Workarounds for languages without TCO:
  → Rewrite as a loop
  → Trampolining (described later)
```

```scheme
;; Scheme: Tail call optimization (TCO) is supported
(define (factorial n)
  (define (iter n acc)
    (if (<= n 1)
        acc
        (iter (- n 1) (* acc n))))  ; Tail position → optimized by TCO
  (iter n 1))
;; Stack does not grow (same efficiency as a loop)
```

```scala
// Scala: @tailrec annotation guarantees tail recursion
import scala.annotation.tailrec

def factorial(n: Long): Long = {
  @tailrec
  def loop(n: Long, acc: Long): Long = {
    if (n <= 1) acc
    else loop(n - 1, acc * n)  // Compiler verifies tail recursion
  }
  loop(n, 1)
}

// Compile error if not tail-recursive
// @tailrec
// def badFactorial(n: Long): Long = {
//   if (n <= 1) 1
//   else n * badFactorial(n - 1)  // Error: not in tail position
// }
```

```elixir
# Elixir: A functional language where tail recursion is encouraged

defmodule Math do
  # Tail-recursive version
  def factorial(n), do: factorial(n, 1)

  defp factorial(0, acc), do: acc
  defp factorial(n, acc) when n > 0 do
    factorial(n - 1, acc * n)
  end

  # List length (tail recursion)
  def length(list), do: length(list, 0)

  defp length([], acc), do: acc
  defp length([_head | tail], acc) do
    length(tail, acc + 1)
  end

  # map (tail recursion + accumulator + reverse)
  def map(list, func), do: map(list, func, [])

  defp map([], _func, acc), do: Enum.reverse(acc)
  defp map([head | tail], func, acc) do
    map(tail, func, [func.(head) | acc])
  end
end
```

---

## 5. Memoization

### 5.1 Basics of Memoization

```python
# The Fibonacci problem: The same computation is repeated many times
# fib(5)
#   fib(4) + fib(3)
#     fib(3)+fib(2)   fib(2)+fib(1)
#     ↑ fib(3) is computed twice

# Time complexity comparison:
# Without memoization: O(2^n) - exponential explosion
# With memoization:    O(n)   - each value computed only once

# Manual memoization
def fibonacci_memo(n, memo=None):
    if memo is None:
        memo = {}
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci_memo(n - 1, memo) + fibonacci_memo(n - 2, memo)
    return memo[n]

# Memoization via decorator
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci(n):
    if n <= 1: return n
    return fibonacci(n - 1) + fibonacci(n - 2)

fibonacci(100)  # Computed instantly (O(n))

# Check cache information
print(fibonacci.cache_info())
# CacheInfo(hits=98, misses=101, maxsize=None, currsize=101)
```

### 5.2 Memoization in Various Languages

```rust
// Rust: Memoization with HashMap
use std::collections::HashMap;

fn fibonacci(n: u64, memo: &mut HashMap<u64, u64>) -> u64 {
    if let Some(&result) = memo.get(&n) {
        return result;
    }
    let result = if n <= 1 { n } else {
        fibonacci(n - 1, memo) + fibonacci(n - 2, memo)
    };
    memo.insert(n, result);
    result
}

fn main() {
    let mut memo = HashMap::new();
    println!("{}", fibonacci(50, &mut memo));  // → 12586269025
}
```

```typescript
// TypeScript: Generic memoization decorator
function memoize<Args extends any[], R>(
    fn: (...args: Args) => R,
    keyFn: (...args: Args) => string = (...args) => JSON.stringify(args)
): (...args: Args) => R {
    const cache = new Map<string, R>();
    return (...args: Args): R => {
        const key = keyFn(...args);
        if (cache.has(key)) return cache.get(key)!;
        const result = fn(...args);
        cache.set(key, result);
        return result;
    };
}

// Usage example: Number of paths (moving right and down on a grid)
const gridPaths = memoize((rows: number, cols: number): number => {
    if (rows === 1 || cols === 1) return 1;
    return gridPaths(rows - 1, cols) + gridPaths(rows, cols - 1);
});

console.log(gridPaths(18, 18));  // → 2333606220 (very slow without memoization)

// LRU cache (memoization with size limit)
class LRUCache<K, V> {
    private cache = new Map<K, V>();

    constructor(private maxSize: number) {}

    get(key: K): V | undefined {
        if (!this.cache.has(key)) return undefined;
        // Move accessed entry to the end (most recently used)
        const value = this.cache.get(key)!;
        this.cache.delete(key);
        this.cache.set(key, value);
        return value;
    }

    set(key: K, value: V): void {
        if (this.cache.has(key)) {
            this.cache.delete(key);
        } else if (this.cache.size >= this.maxSize) {
            // Delete the oldest entry
            const firstKey = this.cache.keys().next().value;
            this.cache.delete(firstKey!);
        }
        this.cache.set(key, value);
    }
}
```

### 5.3 Relationship with Dynamic Programming

```python
# Memoized recursion = Top-down dynamic programming (Top-down DP)
# Table method = Bottom-up dynamic programming (Bottom-up DP)

# Example: Longest Common Subsequence (LCS)

# Memoized recursion (top-down)
@lru_cache(maxsize=None)
def lcs_topdown(s1, s2, i=None, j=None):
    if i is None: i = len(s1) - 1
    if j is None: j = len(s2) - 1
    if i < 0 or j < 0:
        return 0
    if s1[i] == s2[j]:
        return 1 + lcs_topdown(s1, s2, i - 1, j - 1)
    return max(lcs_topdown(s1, s2, i - 1, j), lcs_topdown(s1, s2, i, j - 1))

# Table method (bottom-up)
def lcs_bottomup(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]

# Example: Knapsack problem
@lru_cache(maxsize=None)
def knapsack(weights, values, capacity, i=None):
    if i is None:
        i = len(weights) - 1
    if i < 0 or capacity <= 0:
        return 0
    # Case: Do not include item i
    without = knapsack(weights, values, capacity, i - 1)
    # Case: Include item i
    if weights[i] <= capacity:
        with_item = values[i] + knapsack(weights, values, capacity - weights[i], i - 1)
        return max(without, with_item)
    return without

# Example: Coin change problem
@lru_cache(maxsize=None)
def coin_change(coins, amount):
    """Minimum number of coins needed to make the given amount"""
    if amount == 0:
        return 0
    if amount < 0:
        return float('inf')
    min_coins = float('inf')
    for coin in coins:
        result = coin_change(coins, amount - coin)
        min_coins = min(min_coins, result + 1)
    return min_coins

print(coin_change((1, 5, 10, 25), 63))  # → 6 (25+25+10+1+1+1)

# Example: Number of ways to climb stairs (1 or 2 steps at a time)
@lru_cache(maxsize=None)
def climb_stairs(n):
    if n <= 1:
        return 1
    return climb_stairs(n - 1) + climb_stairs(n - 2)

print(climb_stairs(10))  # → 89
```

---

## 6. Backtracking

### 6.1 Basic Concept

```
Backtracking = Explore the search tree depth-first,
               and when you reach a dead end, return to the
               previous state and try a different option

Algorithm:
  1. Check if the current state is a solution
  2. If it is, record it and terminate (or continue)
  3. Enumerate possible next choices
  4. For each choice:
     a. Apply the choice
     b. Explore recursively
     c. Undo the choice (backtrack)

┌─────────────┐
│   Start     │
└──────┬──────┘
       │
   ┌───┴───┐
   │ Choice1│ Choice2  Choice3
   └───┬───┘
       │
   ┌───┴───┐
   │ ChoiceA│ ChoiceB
   └───┬───┘
       │
    Dead end → Backtrack → Try ChoiceB
```

### 6.2 N-Queens Problem

```python
def solve_n_queens(n):
    """N-Queens problem: Place N queens on an N×N board so none attack each other"""
    solutions = []

    def is_safe(board, row, col):
        # Check if no queen in the same column
        for r in range(row):
            if board[r] == col:
                return False
            # Check if no queen on diagonals
            if abs(board[r] - col) == row - r:
                return False
        return True

    def backtrack(board, row):
        if row == n:
            solutions.append(board[:])  # Record solution
            return
        for col in range(n):
            if is_safe(board, row, col):
                board[row] = col         # Apply choice
                backtrack(board, row + 1) # Explore recursively
                board[row] = -1          # Backtrack

    backtrack([-1] * n, 0)
    return solutions

# 8-Queens solutions
solutions = solve_n_queens(8)
print(f"Number of solutions: {len(solutions)}")  # → 92

# Visualize a solution
def print_board(solution):
    n = len(solution)
    for row in range(n):
        line = ""
        for col in range(n):
            line += "Q " if solution[row] == col else ". "
        print(line)
    print()

print_board(solutions[0])
# Q . . . . . . .
# . . . . Q . . .
# . . . . . . . Q
# . . . . . Q . .
# . . Q . . . . .
# . . . . . . Q .
# . Q . . . . . .
# . . . Q . . . .
```

### 6.3 Sudoku Solver

```python
def solve_sudoku(board):
    """Solve Sudoku (backtracking)"""
    def find_empty():
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0:
                    return (r, c)
        return None

    def is_valid(num, row, col):
        # Row check
        if num in board[row]:
            return False
        # Column check
        if num in [board[r][col] for r in range(9)]:
            return False
        # 3x3 block check
        box_r, box_c = 3 * (row // 3), 3 * (col // 3)
        for r in range(box_r, box_r + 3):
            for c in range(box_c, box_c + 3):
                if board[r][c] == num:
                    return False
        return True

    empty = find_empty()
    if not empty:
        return True  # All cells filled → solved

    row, col = empty
    for num in range(1, 10):
        if is_valid(num, row, col):
            board[row][col] = num       # Choose
            if solve_sudoku(board):     # Recurse
                return True
            board[row][col] = 0         # Backtrack

    return False  # No solution in this branch
```

### 6.4 Maze Solving

```typescript
// Maze solving (backtracking)
type Maze = number[][];  // 0 = path, 1 = wall
type Position = [number, number];

function solveMaze(
    maze: Maze,
    start: Position,
    end: Position
): Position[] | null {
    const rows = maze.length;
    const cols = maze[0].length;
    const visited = Array.from({ length: rows }, () =>
        Array(cols).fill(false)
    );

    function backtrack(r: number, c: number, path: Position[]): Position[] | null {
        // Out of bounds, wall, or already visited
        if (r < 0 || r >= rows || c < 0 || c >= cols) return null;
        if (maze[r][c] === 1 || visited[r][c]) return null;

        path.push([r, c]);
        visited[r][c] = true;

        // Reached the goal
        if (r === end[0] && c === end[1]) {
            return [...path];
        }

        // Explore 4 directions
        const directions: Position[] = [[0, 1], [1, 0], [0, -1], [-1, 0]];
        for (const [dr, dc] of directions) {
            const result = backtrack(r + dr, c + dc, path);
            if (result) return result;
        }

        // Backtrack
        path.pop();
        visited[r][c] = false;
        return null;
    }

    return backtrack(start[0], start[1], []);
}

// Usage example
const maze: Maze = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 0, 1, 0],
];

const path = solveMaze(maze, [0, 0], [4, 4]);
// → [[0,0], [1,0], [2,0], [2,1], [2,2], [3,2], [3,3], [3,4], [4,4]]
// (one example of a valid route)
```

### 6.5 Generating Permutations and Combinations

```typescript
// Generating permutations (backtracking)
function permutations<T>(arr: T[]): T[][] {
    const results: T[][] = [];

    function backtrack(current: T[], remaining: T[]) {
        if (remaining.length === 0) {
            results.push([...current]);
            return;
        }
        for (let i = 0; i < remaining.length; i++) {
            current.push(remaining[i]);
            const newRemaining = [...remaining.slice(0, i), ...remaining.slice(i + 1)];
            backtrack(current, newRemaining);
            current.pop();  // Backtrack
        }
    }

    backtrack([], arr);
    return results;
}

// Generating combinations
function combinations<T>(arr: T[], k: number): T[][] {
    const results: T[][] = [];

    function backtrack(start: number, current: T[]) {
        if (current.length === k) {
            results.push([...current]);
            return;
        }
        for (let i = start; i < arr.length; i++) {
            current.push(arr[i]);
            backtrack(i + 1, current);
            current.pop();
        }
    }

    backtrack(0, []);
    return results;
}

// Subset sum: Find combinations that sum to the target value
function subsetSum(nums: number[], target: number): number[][] {
    const results: number[][] = [];

    function backtrack(start: number, current: number[], remaining: number) {
        if (remaining === 0) {
            results.push([...current]);
            return;
        }
        if (remaining < 0) return;

        for (let i = start; i < nums.length; i++) {
            // Skip duplicates
            if (i > start && nums[i] === nums[i - 1]) continue;
            current.push(nums[i]);
            backtrack(i + 1, current, remaining - nums[i]);
            current.pop();
        }
    }

    nums.sort((a, b) => a - b);
    backtrack(0, [], target);
    return results;
}
```

---

## 7. Recursion vs. Loops: When to Use Which

### 7.1 Decision Criteria

```
When recursion is appropriate:
  - Tree and graph traversal
  - Divide-and-conquer algorithms (merge sort, quicksort)
  - Parsers and compilers (syntax tree processing)
  - When directly corresponding to a mathematical definition
  - Backtracking
  - Processing nested data structures

When loops are appropriate:
  - Simple repetition
  - Performance is critical (in languages without TCO)
  - Processing centered on state updates
  - When deep recursion risks stack overflow
  - When data is flat

Decision flowchart:
  Is the data structure recursive (tree, graph)?
    → YES → Recursion is natural
    → NO → Loops are natural

  Is there a risk of stack overflow?
    → YES → Loop or trampoline
    → NO → Recursion is fine

  Does the language support TCO?
    → YES → Tail recursion
    → NO → Convert deep recursion to loops
```

### 7.2 Examples of Converting to Loops

```rust
// Example: Converting tree traversal to a loop

// Recursive version
struct Node {
    value: i32,
    children: Vec<Node>,
}

fn sum_tree(node: &Node) -> i32 {
    let mut total = node.value;
    for child in &node.children {
        total += sum_tree(child);
    }
    total
}

// Explicit stack version (no recursion)
fn sum_tree_iterative(root: &Node) -> i32 {
    let mut stack = vec![root];
    let mut total = 0;
    while let Some(node) = stack.pop() {
        total += node.value;
        for child in &node.children {
            stack.push(child);
        }
    }
    total
}

// Queue version (breadth-first search BFS)
use std::collections::VecDeque;

fn sum_tree_bfs(root: &Node) -> i32 {
    let mut queue = VecDeque::new();
    queue.push_back(root);
    let mut total = 0;
    while let Some(node) = queue.pop_front() {
        total += node.value;
        for child in &node.children {
            queue.push_back(child);
        }
    }
    total
}
```

```typescript
// TypeScript: Patterns for converting recursion to loops

// Pattern 1: Simple tail recursion → while loop
// Recursive version
function gcd(a: number, b: number): number {
    if (b === 0) return a;
    return gcd(b, a % b);
}

// Loop version
function gcdLoop(a: number, b: number): number {
    while (b !== 0) {
        [a, b] = [b, a % b];
    }
    return a;
}

// Pattern 2: Depth-first search → explicit stack
// Recursive version
function flattenTree<T>(node: TreeNode<T>): T[] {
    const result = [node.value];
    for (const child of node.children) {
        result.push(...flattenTree(child));
    }
    return result;
}

// Explicit stack version
function flattenTreeIterative<T>(root: TreeNode<T>): T[] {
    const result: T[] = [];
    const stack: TreeNode<T>[] = [root];
    while (stack.length > 0) {
        const node = stack.pop()!;
        result.push(node.value);
        // Push children in reverse order (so the first child is popped first)
        for (let i = node.children.length - 1; i >= 0; i--) {
            stack.push(node.children[i]);
        }
    }
    return result;
}

// Pattern 3: Backtracking → explicit state management
// Recursive permutation version
function permsRecursive(arr: number[]): number[][] {
    if (arr.length <= 1) return [arr];
    return arr.flatMap((item, i) => {
        const rest = [...arr.slice(0, i), ...arr.slice(i + 1)];
        return permsRecursive(rest).map(perm => [item, ...perm]);
    });
}

// Explicit state management version
function permsIterative(arr: number[]): number[][] {
    const results: number[][] = [];
    interface State {
        current: number[];
        remaining: number[];
    }
    const stack: State[] = [{ current: [], remaining: arr }];

    while (stack.length > 0) {
        const { current, remaining } = stack.pop()!;
        if (remaining.length === 0) {
            results.push(current);
            continue;
        }
        for (let i = remaining.length - 1; i >= 0; i--) {
            stack.push({
                current: [...current, remaining[i]],
                remaining: [
                    ...remaining.slice(0, i),
                    ...remaining.slice(i + 1),
                ],
            });
        }
    }
    return results;
}
```

---

## 8. Trampolines and CPS

### 8.1 Trampoline

```typescript
// Trampoline: A technique for safely executing tail recursion in languages without TCO
// Instead of making recursive calls, return "the next function to call"
// and call it repeatedly in a loop

type Thunk<T> = () => T | Thunk<T>;

function trampoline<T>(fn: Thunk<T>): T {
    let result: any = fn;
    while (typeof result === "function") {
        result = result();
    }
    return result;
}

// Usage example: Safe factorial (no stack overflow)
function factorialTramp(n: number, acc: number = 1): number | (() => number | (() => any)) {
    if (n <= 1) return acc;
    return () => factorialTramp(n - 1, acc * n);  // Return a function (don't call it)
}

console.log(trampoline(() => factorialTramp(100000)));
// → Infinity (numeric overflow but no stack overflow)

// More type-safe trampoline
type Bounce<T> =
    | { done: true; value: T }
    | { done: false; thunk: () => Bounce<T> };

function done<T>(value: T): Bounce<T> {
    return { done: true, value };
}

function bounce<T>(thunk: () => Bounce<T>): Bounce<T> {
    return { done: false, thunk };
}

function run<T>(b: Bounce<T>): T {
    let current = b;
    while (!current.done) {
        current = current.thunk();
    }
    return current.value;
}

// Usage example: Trampolining mutual recursion
function isEvenTramp(n: number): Bounce<boolean> {
    if (n === 0) return done(true);
    return bounce(() => isOddTramp(n - 1));
}

function isOddTramp(n: number): Bounce<boolean> {
    if (n === 0) return done(false);
    return bounce(() => isEvenTramp(n - 1));
}

console.log(run(isEvenTramp(1000000)));  // → true (no stack overflow)
```

### 8.2 CPS (Continuation-Passing Style)

```typescript
// CPS: Convert to tail recursion by passing results to callbacks

// Direct style
function sumList(arr: number[]): number {
    if (arr.length === 0) return 0;
    return arr[0] + sumList(arr.slice(1));
}

// CPS transformation
function sumListCPS(arr: number[], k: (result: number) => number): number {
    if (arr.length === 0) return k(0);
    return sumListCPS(arr.slice(1), (restSum) => k(arr[0] + restSum));
}

sumListCPS([1, 2, 3, 4, 5], x => x);  // → 15

// CPS tree traversal
interface TreeNode {
    value: number;
    left?: TreeNode;
    right?: TreeNode;
}

// Direct style (stack depth = tree depth)
function sumTree(node: TreeNode | undefined): number {
    if (!node) return 0;
    return node.value + sumTree(node.left) + sumTree(node.right);
}

// CPS (can be made tail-recursive)
function sumTreeCPS(
    node: TreeNode | undefined,
    k: (result: number) => number
): number {
    if (!node) return k(0);
    return sumTreeCPS(node.left, (leftSum) =>
        sumTreeCPS(node.right, (rightSum) =>
            k(node.value + leftSum + rightSum)
        )
    );
}
```

---

## 9. Practical Algorithms Using Recursion

### 9.1 Divide-and-Conquer Algorithms

```python
# General structure of divide and conquer:
# 1. Divide: Split the problem into smaller subproblems
# 2. Conquer: Solve each subproblem recursively
# 3. Combine: Merge the subproblem solutions

# Maximum subarray sum (Kadane's vs. divide and conquer)
def max_subarray_dc(arr, lo=0, hi=None):
    """Maximum subarray sum using divide and conquer O(n log n)"""
    if hi is None:
        hi = len(arr) - 1
    if lo == hi:
        return arr[lo]

    mid = (lo + hi) // 2

    # Maximum subarray sum in the left half
    left_max = max_subarray_dc(arr, lo, mid)
    # Maximum subarray sum in the right half
    right_max = max_subarray_dc(arr, mid + 1, hi)

    # Maximum subarray sum crossing the center
    left_sum = float('-inf')
    total = 0
    for i in range(mid, lo - 1, -1):
        total += arr[i]
        left_sum = max(left_sum, total)

    right_sum = float('-inf')
    total = 0
    for i in range(mid + 1, hi + 1):
        total += arr[i]
        right_sum = max(right_sum, total)

    cross_max = left_sum + right_sum

    return max(left_max, right_max, cross_max)

# Closest pair of points problem
import math

def closest_pair(points):
    """Find the closest pair of points on a plane O(n log n)"""
    def distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def closest_pair_rec(pts_x, pts_y):
        n = len(pts_x)
        if n <= 3:
            # Brute force
            min_dist = float('inf')
            best_pair = None
            for i in range(n):
                for j in range(i + 1, n):
                    d = distance(pts_x[i], pts_x[j])
                    if d < min_dist:
                        min_dist = d
                        best_pair = (pts_x[i], pts_x[j])
            return min_dist, best_pair

        mid = n // 2
        mid_point = pts_x[mid]

        # Divide
        left_x = pts_x[:mid]
        right_x = pts_x[mid:]
        left_y = [p for p in pts_y if p[0] <= mid_point[0]]
        right_y = [p for p in pts_y if p[0] > mid_point[0]]

        # Conquer (recursion)
        dl, pair_l = closest_pair_rec(left_x, left_y)
        dr, pair_r = closest_pair_rec(right_x, right_y)

        d = min(dl, dr)
        best = pair_l if dl < dr else pair_r

        # Combine: Check points within the strip
        strip = [p for p in pts_y if abs(p[0] - mid_point[0]) < d]
        for i in range(len(strip)):
            for j in range(i + 1, min(i + 8, len(strip))):
                dd = distance(strip[i], strip[j])
                if dd < d:
                    d = dd
                    best = (strip[i], strip[j])

        return d, best

    pts_x = sorted(points, key=lambda p: p[0])
    pts_y = sorted(points, key=lambda p: p[1])
    return closest_pair_rec(pts_x, pts_y)

# Strassen's matrix multiplication (concept)
# Normal matrix multiplication: O(n^3)
# Strassen: O(n^2.807) - reduced to 7 multiplications via divide and conquer
```

### 9.2 Recursive Graph Exploration

```typescript
// Graph DFS (depth-first search)
type Graph = Map<string, string[]>;

// Recursive DFS
function dfs(
    graph: Graph,
    start: string,
    visited: Set<string> = new Set()
): string[] {
    visited.add(start);
    const result = [start];

    for (const neighbor of graph.get(start) || []) {
        if (!visited.has(neighbor)) {
            result.push(...dfs(graph, neighbor, visited));
        }
    }

    return result;
}

// Topological sort (dependency order of a DAG)
function topologicalSort(graph: Graph): string[] {
    const visited = new Set<string>();
    const result: string[] = [];

    function visit(node: string) {
        if (visited.has(node)) return;
        visited.add(node);
        for (const neighbor of graph.get(node) || []) {
            visit(neighbor);
        }
        result.unshift(node);  // Prepend during post-processing
    }

    for (const node of graph.keys()) {
        visit(node);
    }

    return result;
}

// Cycle detection
function hasCycle(graph: Graph): boolean {
    const white = new Set<string>(graph.keys()); // Unvisited
    const gray = new Set<string>();               // Currently visiting
    const black = new Set<string>();              // Completed

    function dfsVisit(node: string): boolean {
        white.delete(node);
        gray.add(node);

        for (const neighbor of graph.get(node) || []) {
            if (gray.has(neighbor)) return true;  // Cycle detected
            if (white.has(neighbor) && dfsVisit(neighbor)) return true;
        }

        gray.delete(node);
        black.add(node);
        return false;
    }

    for (const node of [...white]) {
        if (dfsVisit(node)) return true;
    }
    return false;
}

// Usage example
const dependencyGraph: Graph = new Map([
    ["main", ["auth", "db", "logger"]],
    ["auth", ["db", "crypto"]],
    ["db", ["logger"]],
    ["crypto", []],
    ["logger", []],
]);

console.log(topologicalSort(dependencyGraph));
// → ["main", "auth", "db", "crypto", "logger"]
// (or another valid order)
```

### 9.3 Recursive Descent Parser

```typescript
// Recursive descent parser: A group of recursive functions corresponding to grammar rules

// Simple expression language:
// expr   = term (('+' | '-') term)*
// term   = factor (('*' | '/') factor)*
// factor = NUMBER | '(' expr ')'

type Token =
    | { type: "number"; value: number }
    | { type: "op"; value: string }
    | { type: "paren"; value: "(" | ")" };

class ExprParser {
    private pos = 0;

    constructor(private tokens: Token[]) {}

    private peek(): Token | null {
        return this.pos < this.tokens.length ? this.tokens[this.pos] : null;
    }

    private consume(): Token {
        return this.tokens[this.pos++];
    }

    private expect(type: string, value?: string): Token {
        const token = this.consume();
        if (token.type !== type || (value !== undefined && token.value !== value)) {
            throw new Error(`Expected ${type}(${value}), got ${token.type}(${token.value})`);
        }
        return token;
    }

    parse(): number {
        const result = this.parseExpr();
        if (this.pos < this.tokens.length) {
            throw new Error("Unexpected tokens after expression");
        }
        return result;
    }

    private parseExpr(): number {
        let result = this.parseTerm();
        while (this.peek()?.type === "op" &&
               (this.peek()!.value === "+" || this.peek()!.value === "-")) {
            const op = this.consume().value;
            const right = this.parseTerm();
            result = op === "+" ? result + right : result - right;
        }
        return result;
    }

    private parseTerm(): number {
        let result = this.parseFactor();
        while (this.peek()?.type === "op" &&
               (this.peek()!.value === "*" || this.peek()!.value === "/")) {
            const op = this.consume().value;
            const right = this.parseFactor();
            result = op === "*" ? result * right : result / right;
        }
        return result;
    }

    private parseFactor(): number {
        const token = this.peek();
        if (token?.type === "number") {
            this.consume();
            return token.value as number;
        }
        if (token?.type === "paren" && token.value === "(") {
            this.consume();                    // Consume '('
            const result = this.parseExpr();   // Mutual recursion
            this.expect("paren", ")");         // Consume ')'
            return result;
        }
        throw new Error(`Unexpected token: ${JSON.stringify(token)}`);
    }
}
```

---

## 10. Complexity Analysis of Recursion

### 10.1 Master Theorem

```
Master Theorem for determining the complexity of divide-and-conquer:

Recurrence form: T(n) = a * T(n/b) + O(n^d)
  a = number of subproblems
  b = factor by which problem size shrinks
  d = exponent of the combination step's complexity

Three cases:
  Case 1: d < log_b(a) → T(n) = O(n^(log_b(a)))
  Case 2: d = log_b(a) → T(n) = O(n^d * log(n))
  Case 3: d > log_b(a) → T(n) = O(n^d)

Examples:
  Merge sort: T(n) = 2T(n/2) + O(n)
    a=2, b=2, d=1 → log_2(2) = 1 = d → Case 2 → O(n log n)

  Binary search: T(n) = T(n/2) + O(1)
    a=1, b=2, d=0 → log_2(1) = 0 = d → Case 2 → O(log n)

  Strassen: T(n) = 7T(n/2) + O(n^2)
    a=7, b=2, d=2 → log_2(7) ≈ 2.807 > 2 → Case 1 → O(n^2.807)

  Standard matrix multiplication: T(n) = 8T(n/2) + O(n^2)
    a=8, b=2, d=2 → log_2(8) = 3 > 2 → Case 1 → O(n^3)
```

### 10.2 Complexity of Each Recursion Pattern

```
┌──────────────────┬────────────┬──────────────┬──────────────┐
│ Pattern           │ Time       │ Space        │ Example       │
├──────────────────┼────────────┼──────────────┼──────────────┤
│ Linear recursion  │ O(n)       │ O(n)         │ Factorial, sum│
│ Tail recursion    │ O(n)       │ O(1)         │ Factorial     │
│ (with TCO)        │            │              │ (tail version)│
│ Binary search     │ O(log n)   │ O(log n)     │ Binary search │
│ recursion         │            │              │               │
│ Divide & conquer  │ O(n log n) │ O(n)         │ Merge sort    │
│ Tree recursion    │ O(2^n)     │ O(n)         │ fib (naive)   │
│ (naive)           │            │              │               │
│ Tree recursion    │ O(n)       │ O(n)         │ fib (memo)    │
│ (memoized)        │            │              │               │
│ Backtracking      │ O(b^d)     │ O(d)         │ N-Queens      │
│ All permutations  │ O(n!)      │ O(n)         │ permutations  │
└──────────────────┴────────────┴──────────────┴──────────────┘
  b = branching factor, d = depth
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

| Concept | Description |
|------|------|
| Base case | Termination condition for recursion (required) |
| Recursive step | Makes the problem smaller and calls itself |
| Tail recursion | Recursion is the last operation. Can be optimized by TCO |
| Memoization | Cache computed results to eliminate duplicate computation |
| Divide and conquer | Split the problem in half and recurse (O(n log n)) |
| Backtracking | Traverse the search tree via DFS, backtrack when stuck |
| Trampoline | Safely execute tail recursion in languages without TCO |
| CPS | Convert to tail recursion by passing results to callbacks |

Principles for using recursion effectively:

1. **Always define a base case**: The most critical rule to prevent infinite recursion
2. **Guarantee the problem gets smaller**: The problem size must decrease with each recursive call
3. **Apply memoization for duplicate computation**: Improve exponential complexity to linear
4. **Be aware of stack overflow**: Consider loops or trampolines for deep recursion
5. **Choose the right technique**: Recursion for tree structures, loops for flat data
6. **Balance readability and efficiency**: Use recursion when it is natural, otherwise use loops

---

## Recommended Next Guides

---

## References
1. Abelson, H. & Sussman, G. "SICP." Ch.1.2, MIT Press, 1996.
2. Cormen, T. et al. "Introduction to Algorithms." Ch.4, MIT Press, 2022.
3. Sedgewick, R. & Wayne, K. "Algorithms." 4th Edition, Addison-Wesley, 2011.
4. Skiena, S. "The Algorithm Design Manual." 3rd Edition, Springer, 2020.
5. Bird, R. "Thinking Functionally with Haskell." Cambridge, 2014.
6. Okasaki, C. "Purely Functional Data Structures." Cambridge, 1998.
7. Graham, R. et al. "Concrete Mathematics." Addison-Wesley, 1994.
