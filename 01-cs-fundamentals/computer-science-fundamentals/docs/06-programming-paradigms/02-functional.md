# Functional Programming

> Functional Programming (FP) is a paradigm based on "composing pure functions without side effects," excelling in concurrency safety and testability. Based on the mathematical concept of functions, it describes programs as data transformations rather than state changes.

## Learning Objectives

- [ ] Explain the concepts of pure functions and side effects
- [ ] Understand the meaning and benefits of referential transparency
- [ ] Master the use of map/filter/reduce
- [ ] Understand the benefits of Immutability and apply it
- [ ] Implement function composition, currying, and partial application
- [ ] Understand the concept of monads and use Option/Result patterns
- [ ] Apply functional techniques in real-world projects


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding of the content in [Object-Oriented Programming](./01-object-oriented.md)

---

## 1. History and Background of Functional Programming

### 1.1 From Mathematical Foundations to Programming

The theoretical foundation of functional programming traces back to Lambda Calculus by Alonzo Church in the 1930s. It has been proven to be equivalent in computational power to the Turing Machine, providing a powerful mathematical foundation as a model of computation.

```
Functional Programming Historical Timeline:

1930s    Lambda Calculus (Alonzo Church)
          - A formal system for function abstraction and application
          - Proven to be Turing complete
          - Mathematical foundation of modern FP

1958     LISP (John McCarthy)
          - The first functional programming language
          - List processing, recursion, garbage collection
          - Unified representation through S-expressions

1973     ML (Robin Milner)
          - Type inference (Hindley-Milner type system)
          - Pattern matching
          - Algebraic data types

1977     FP (John Backus)
          - Turing Award lecture "Can Programming Be Liberated
            from the von Neumann Style?"
          - Advocacy of function-level programming

1986     Erlang (Joe Armstrong / Ericsson)
          - Actor model + functional
          - Fault tolerance, hot swapping
          - Proven track record in telecommunications systems

1990     Haskell
          - Unification of pure functional languages
          - Lazy evaluation, monads, type classes
          - Standard language for academic research

2003     Scala (Martin Odersky)
          - Fusion of OOP and FP
          - Runs on the JVM
          - Akka (actor model)

2007     Clojure (Rich Hickey)
          - A LISP dialect on the JVM
          - Persistent data structures
          - Strong support for concurrency

2012     Elixir (Jose Valim)
          - A functional language on the Erlang VM (BEAM)
          - Phoenix framework
          - Practical for web development

2015     Elm (Evan Czaplicki)
          - Pure FP for web frontend
          - The Elm Architecture (TEA)
          - Major influence on React and Redux
```

### 1.2 Core Philosophy of Functional Programming

```
Core of Functional Programming:

  Imperative programming:
  Describes "How" as step-by-step procedures
  -> Declare variables, loop, modify state

  Functional programming:
  Describes "What" declaratively
  -> Define the relationship (mapping) between input and output

  Correspondence to mathematical functions:
  f(x) = x^2 + 2x + 1
  -> Always returns the same result for the same x
  -> Executing the function does not affect anything else
  -> Functions can be combined to create new functions
    g(x) = f(x) + 3 = x^2 + 2x + 4

  Application to programming:
  Program = composition of functions
  Data flow: input -> transform1 -> transform2 -> ... -> output
```

---

## 2. Core Concepts of Functional Programming

### 2.1 Pure Functions

Pure functions are the most fundamental concept in functional programming. A function that meets two conditions is called a pure function.

```python
# === Definition of Pure Functions ===
# Condition 1: Always returns the same value for the same arguments (deterministic)
# Condition 2: No side effects (does not read or write external state)

# Pure function examples
def add(a: int, b: int) -> int:
    return a + b  # Depends only on input, does not modify external state

def square(x: float) -> float:
    return x ** 2  # Always the same result for the same x

def format_name(first: str, last: str) -> str:
    return f"{last} {first}"  # No side effects

def calculate_tax(price: int, tax_rate: float) -> int:
    return int(price * tax_rate)  # Does not depend on external state

def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)  # Recursion is also pure


# Impure function examples
import random
import datetime

# Impure: modifies external state (global variable)
counter = 0
def increment():
    global counter
    counter += 1  # Side effect: modifies external state
    return counter

# Impure: different results for the same arguments
def get_random_number(max_val: int) -> int:
    return random.randint(0, max_val)  # Non-deterministic

# Impure: depends on external state (current time)
def get_greeting(name: str) -> str:
    hour = datetime.datetime.now().hour  # Depends on external state
    if hour < 12:
        return f"Good morning, {name}"
    return f"Hello, {name}"

# Impure: I/O (side effect)
def log_message(message: str) -> None:
    print(message)  # Side effect: screen output

def read_file(path: str) -> str:
    with open(path) as f:
        return f.read()  # Side effect: file I/O
```

```python
# === Techniques for Making Impure Functions Closer to Pure ===

# Method 1: Include dependent external state as arguments
# Impure
def get_greeting_impure(name: str) -> str:
    hour = datetime.datetime.now().hour
    if hour < 12:
        return f"Good morning, {name}"
    return f"Hello, {name}"

# Pure (time is passed as an argument)
def get_greeting_pure(name: str, hour: int) -> str:
    if hour < 12:
        return f"Good morning, {name}"
    return f"Hello, {name}"

# Easy to test
assert get_greeting_pure("Taro", 9) == "Good morning, Taro"
assert get_greeting_pure("Taro", 15) == "Hello, Taro"


# Method 2: Push side effects to the outside of the function
# Impure: business logic and I/O are mixed
def process_order_impure(order_id: int):
    order = db.fetch_order(order_id)         # I/O
    total = calculate_total(order.items)     # Computation
    tax = total * 0.1                         # Computation
    db.update_order(order_id, total + tax)   # I/O
    send_email(order.customer, "Order confirmed")    # I/O

# Pure core + impure shell
def calculate_order_total(items: list[dict]) -> dict:
    """Pure: only computation logic"""
    subtotal = sum(item['price'] * item['qty'] for item in items)
    tax = int(subtotal * 0.1)
    return {
        'subtotal': subtotal,
        'tax': tax,
        'total': subtotal + tax
    }

def process_order(order_id: int):
    """Impure: the shell that handles I/O"""
    order = db.fetch_order(order_id)                     # I/O
    result = calculate_order_total(order.items)          # Pure function call
    db.update_order(order_id, result['total'])            # I/O
    send_email(order.customer, f"Order confirmed: {result}")    # I/O
```

### 2.2 Referential Transparency

```python
# === Referential Transparency ===
# The property that an expression can be replaced with its value
# without changing the program's behavior

# Referentially transparent
def add(a: int, b: int) -> int:
    return a + b

# add(3, 4) can be replaced with 7
result = add(3, 4) + add(1, 2)
# Same as:
result = 7 + 3  # = 10

# This enables the following:
# 1. Memoization (caching): no need to recompute for the same arguments
# 2. Lazy evaluation: defer computation until needed
# 3. Parallelization: safely evaluate sub-expressions in parallel
# 4. Equational reasoning: mathematically prove correctness


# Not referentially transparent
call_count = 0
def add_with_side_effect(a: int, b: int) -> int:
    global call_count
    call_count += 1  # Not referentially transparent due to side effect
    return a + b

# Replacing add_with_side_effect(3, 4) with 7
# changes the behavior of call_count


# Memoization implementation example (referential transparency is a prerequisite)
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_computation(n: int) -> int:
    """Speed up expensive computation by caching results"""
    print(f"  Computing: {n}")  # Only printed on first call
    return sum(i ** 2 for i in range(n))

# First call performs the computation
print(expensive_computation(10000))  # Computing: 10000 -> result
# Second call returns from cache (safe because it's a pure function)
print(expensive_computation(10000))  # Instant result (no computation)


# Fibonacci sequence: memoization reduces exponential to linear time
@lru_cache(maxsize=None)
def fib(n: int) -> int:
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)

print(fib(100))  # 354224848179261915075 (infeasible without memoization)
```

### 2.3 Higher-Order Functions

Higher-order functions are functions that either "take a function as an argument" or "return a function." They are fundamental tools in functional programming.

```python
# === Higher-Order Function Basics ===

from typing import Callable, TypeVar

T = TypeVar('T')
U = TypeVar('U')


# 1. map: apply a function to each element to transform
names = ["alice", "bob", "charlie"]
upper_names = list(map(str.upper, names))
# -> ['ALICE', 'BOB', 'CHARLIE']

# Pythonic list comprehension alternative
upper_names = [name.upper() for name in names]


# 2. filter: extract elements that satisfy a condition
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
evens = list(filter(lambda x: x % 2 == 0, numbers))
# -> [2, 4, 6, 8, 10]

# Pythonic
evens = [x for x in numbers if x % 2 == 0]


# 3. reduce: fold (cumulative operation from the left)
from functools import reduce

total = reduce(lambda acc, x: acc + x, numbers, 0)
# -> 55

product = reduce(lambda acc, x: acc * x, numbers, 1)
# -> 3628800

# Tracing reduce's operation:
# reduce(f, [1,2,3,4], 0)
# -> f(f(f(f(0, 1), 2), 3), 4)
# -> f(f(f(1, 2), 3), 4)
# -> f(f(3, 3), 4)
# -> f(6, 4)
# -> 10


# 4. sorted: sorting with a custom key
students = [
    {"name": "Tanaka", "score": 85},
    {"name": "Sato", "score": 92},
    {"name": "Suzuki", "score": 78},
]
by_score = sorted(students, key=lambda s: s["score"], reverse=True)
# -> [{"name": "Sato", ...}, {"name": "Tanaka", ...}, {"name": "Suzuki", ...}]


# 5. any / all: logical aggregation
scores = [85, 92, 78, 65, 90]
all_passed = all(s >= 60 for s in scores)       # True: all scored 60 or above
has_perfect = any(s == 100 for s in scores)     # False: no perfect score


# 6. Higher-order function that returns a function
def make_multiplier(factor: int) -> Callable[[int], int]:
    """Higher-order function that generates a multiplication function"""
    def multiplier(x: int) -> int:
        return x * factor
    return multiplier

double = make_multiplier(2)
triple = make_multiplier(3)
print(double(5))   # 10
print(triple(5))   # 15


# 7. Decorators (Python's higher-order function pattern)
import time
from functools import wraps

def timing(func: Callable) -> Callable:
    """Decorator that measures function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__}: {elapsed:.4f} seconds")
        return result
    return wrapper

def retry(max_attempts: int = 3, delay: float = 1.0):
    """Decorator that adds retry functionality"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts:
                        raise
                    print(f"Retry {attempt}/{max_attempts}: {e}")
                    time.sleep(delay)
        return wrapper
    return decorator

@timing
@retry(max_attempts=3, delay=0.5)
def fetch_data(url: str) -> dict:
    """Fetch data"""
    # Actual API call processing
    pass
```

```javascript
// Higher-order functions in JavaScript

// map: transformation
const users = [
    { name: 'Tanaka', age: 30 },
    { name: 'Sato', age: 25 },
    { name: 'Suzuki', age: 35 },
];

const names = users.map(u => u.name);
// -> ['Tanaka', 'Sato', 'Suzuki']

const withGreeting = users.map(u => ({
    ...u,
    greeting: `Hello, ${u.name} (age ${u.age})`,
}));

// filter: extraction
const adults = users.filter(u => u.age >= 30);

// reduce: fold
const totalAge = users.reduce((sum, u) => sum + u.age, 0);
// -> 90

// Implementing groupBy with reduce
const byAge = users.reduce((groups, user) => {
    const key = user.age >= 30 ? 'senior' : 'junior';
    return {
        ...groups,
        [key]: [...(groups[key] || []), user],
    };
}, {});

// Method chaining (pipeline-like usage)
const result = users
    .filter(u => u.age >= 25)
    .map(u => u.name.toUpperCase())
    .sort()
    .join(', ');
// -> 'SATO, SUZUKI, TANAKA'

// flatMap: flatten while transforming
const orders = [
    { id: 1, items: ['Pen', 'Notebook'] },
    { id: 2, items: ['Eraser'] },
    { id: 3, items: ['Ruler', 'Protractor', 'Compass'] },
];

const allItems = orders.flatMap(order => order.items);
// -> ['Pen', 'Notebook', 'Eraser', 'Ruler', 'Protractor', 'Compass']
```

### 2.4 Immutability

```python
# === Principles and Implementation of Immutability ===

# Immutability: instead of modifying data, generate new data

# Mutable operations
cart = [{"item": "Laptop", "qty": 1}]
cart[0]["qty"] = 2        # Original data is modified!
cart.append({"item": "Mouse", "qty": 1})  # Original list is modified!


# Immutable operations
from dataclasses import dataclass, replace, field
from typing import Tuple


@dataclass(frozen=True)
class CartItem:
    """Shopping cart item (immutable)"""
    name: str
    price: int
    quantity: int

    def with_quantity(self, qty: int) -> 'CartItem':
        """Return a new CartItem with the quantity changed"""
        return replace(self, quantity=qty)


@dataclass(frozen=True)
class ShoppingCart:
    """Shopping cart (immutable)"""
    items: tuple[CartItem, ...] = ()

    def add_item(self, item: CartItem) -> 'ShoppingCart':
        """Return a new cart with the item added"""
        return replace(self, items=self.items + (item,))

    def remove_item(self, index: int) -> 'ShoppingCart':
        """Return a new cart with the item removed"""
        new_items = self.items[:index] + self.items[index + 1:]
        return replace(self, items=new_items)

    def update_quantity(self, index: int, qty: int) -> 'ShoppingCart':
        """Return a new cart with the quantity changed"""
        items_list = list(self.items)
        items_list[index] = items_list[index].with_quantity(qty)
        return replace(self, items=tuple(items_list))

    @property
    def total(self) -> int:
        return sum(item.price * item.quantity for item in self.items)

    @property
    def item_count(self) -> int:
        return sum(item.quantity for item in self.items)


# Usage example: chaining immutable operations
cart = ShoppingCart()
cart = cart.add_item(CartItem("Laptop", 150000, 1))
cart = cart.add_item(CartItem("Mouse", 3000, 2))
cart = cart.update_quantity(0, 2)  # Change Laptop to 2 units

# The original cart is not changed (a new instance was generated)
print(f"Total: \u00a5{cart.total:,}")  # \u00a5306,000
print(f"Item count: {cart.item_count}")  # 4
```

```python
# === Immutable Collection Operations ===

# Immutable list operations
def append(lst: tuple, item) -> tuple:
    """Return a new tuple with the element appended"""
    return lst + (item,)

def remove_at(lst: tuple, index: int) -> tuple:
    """Return a new tuple with the element removed"""
    return lst[:index] + lst[index + 1:]

def update_at(lst: tuple, index: int, value) -> tuple:
    """Return a new tuple with the element updated"""
    return lst[:index] + (value,) + lst[index + 1:]


# Immutable dict operations
def assoc(d: dict, key: str, value) -> dict:
    """Return a new dict with the key-value added/updated"""
    return {**d, key: value}

def dissoc(d: dict, key: str) -> dict:
    """Return a new dict with the key removed"""
    return {k: v for k, v in d.items() if k != key}

def update_in(d: dict, keys: list[str], func) -> dict:
    """Return a new dict with the nested key's value updated by a function"""
    if len(keys) == 1:
        return {**d, keys[0]: func(d.get(keys[0]))}
    key = keys[0]
    return {**d, key: update_in(d.get(key, {}), keys[1:], func)}


# Usage example
state = {"user": {"name": "Tanaka", "age": 30}, "count": 0}

# Immutable update
new_state = update_in(state, ["user", "age"], lambda x: x + 1)
print(state["user"]["age"])       # 30 (original unchanged)
print(new_state["user"]["age"])   # 31 (new state)

new_state = assoc(state, "count", state["count"] + 1)
print(state["count"])      # 0
print(new_state["count"])  # 1
```

```typescript
// Immutability in TypeScript

// Define immutable types with Readonly
interface User {
    readonly id: string;
    readonly name: string;
    readonly email: string;
    readonly tags: readonly string[];
}

// Updates always generate a new object using spread syntax
function updateUserName(user: User, name: string): User {
    return { ...user, name };
}

function addTag(user: User, tag: string): User {
    return { ...user, tags: [...user.tags, tag] };
}

function removeTag(user: User, tag: string): User {
    return { ...user, tags: user.tags.filter(t => t !== tag) };
}

// ReadonlyArray, ReadonlyMap, ReadonlySet
const numbers: readonly number[] = [1, 2, 3];
// numbers.push(4);  // Compile error

// Deep immutability with Utility Types
type DeepReadonly<T> = {
    readonly [P in keyof T]: T[P] extends object
        ? DeepReadonly<T[P]>
        : T[P];
};

interface AppState {
    user: {
        name: string;
        settings: {
            theme: string;
            language: string;
        };
    };
    todos: Array<{ id: number; text: string; done: boolean }>;
}

// DeepReadonly<AppState> makes all properties recursively readonly
type ImmutableAppState = DeepReadonly<AppState>;
```

```javascript
// Immutability patterns in React/Redux

// Redux Reducer always returns a new state (immutable update)
function todosReducer(state = [], action) {
    switch (action.type) {
        case 'ADD_TODO':
            // Bad: state.push(action.todo) -> modifies the original state
            // Good: return a new array
            return [...state, action.todo];

        case 'TOGGLE_TODO':
            return state.map(todo =>
                todo.id === action.id
                    ? { ...todo, completed: !todo.completed }
                    : todo
            );

        case 'REMOVE_TODO':
            return state.filter(todo => todo.id !== action.id);

        default:
            return state;
    }
}

// With Immer, you can write mutable-style code for immutable updates
// import { produce } from 'immer';
//
// const nextState = produce(state, draft => {
//     draft.todos.push({ id: 3, text: 'New Todo', done: false });
//     draft.todos[0].done = true;
// });
// -> state is not modified, nextState is a new object
```

---

## 3. Function Composition and Transformation Pipelines

### 3.1 Function Composition

```python
# === Function Composition ===
# Build complex processing by combining small functions

from typing import Callable, TypeVar
from functools import reduce as functools_reduce

A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')


# Basic function composition
def compose(f: Callable[[B], C], g: Callable[[A], B]) -> Callable[[A], C]:
    """f . g: apply g first, then apply f"""
    return lambda x: f(g(x))


def pipe(*functions: Callable) -> Callable:
    """Apply functions from left to right (pipeline)"""
    def piped(x):
        result = x
        for func in functions:
            result = func(result)
        return result
    return piped


# Usage example: text processing pipeline
def strip_whitespace(text: str) -> str:
    return text.strip()

def to_lowercase(text: str) -> str:
    return text.lower()

def remove_punctuation(text: str) -> str:
    import re
    return re.sub(r'[^\w\s]', '', text)

def split_words(text: str) -> list[str]:
    return text.split()

def unique_words(words: list[str]) -> set[str]:
    return set(words)


# Build a pipeline via function composition
normalize_text = pipe(strip_whitespace, to_lowercase, remove_punctuation)
extract_unique_words = pipe(normalize_text, split_words, unique_words)

text = "  Hello, World! Hello, Python!  "
print(normalize_text(text))        # "hello world hello python"
print(extract_unique_words(text))  # {'hello', 'world', 'python'}
```

```python
# === Data Transformation Pipeline ===

from dataclasses import dataclass
from typing import Callable, Iterable


def pipeline(data, *functions):
    """Pass data through a chain of functions"""
    result = data
    for func in functions:
        result = func(result)
    return result


# Practical example: log analysis pipeline
@dataclass(frozen=True)
class LogEntry:
    timestamp: str
    level: str
    message: str
    source: str


def parse_log_line(line: str) -> LogEntry:
    """Parse a log line"""
    parts = line.split(" | ")
    return LogEntry(
        timestamp=parts[0],
        level=parts[1],
        message=parts[2],
        source=parts[3] if len(parts) > 3 else "unknown"
    )


def parse_all(lines: list[str]) -> list[LogEntry]:
    return [parse_log_line(line) for line in lines if line.strip()]


def filter_errors(entries: list[LogEntry]) -> list[LogEntry]:
    return [e for e in entries if e.level in ("ERROR", "CRITICAL")]


def group_by_source(entries: list[LogEntry]) -> dict[str, list[LogEntry]]:
    groups: dict[str, list[LogEntry]] = {}
    for entry in entries:
        groups.setdefault(entry.source, []).append(entry)
    return groups


def count_per_group(groups: dict[str, list]) -> dict[str, int]:
    return {source: len(entries) for source, entries in groups.items()}


def sort_by_count(counts: dict[str, int]) -> list[tuple[str, int]]:
    return sorted(counts.items(), key=lambda x: x[1], reverse=True)


# Process via pipeline
log_lines = [
    "2025-01-15 10:00:00 | ERROR | DB connection failed | database",
    "2025-01-15 10:01:00 | INFO | Request processing complete | api",
    "2025-01-15 10:02:00 | ERROR | Timeout | api",
    "2025-01-15 10:03:00 | CRITICAL | OOM occurred | worker",
    "2025-01-15 10:04:00 | ERROR | DB connection failed | database",
]

error_ranking = pipeline(
    log_lines,
    parse_all,
    filter_errors,
    group_by_source,
    count_per_group,
    sort_by_count,
)
# -> [('database', 2), ('api', 1), ('worker', 1)]
for source, count in error_ranking:
    print(f"  {source}: {count} error(s)")
```

### 3.2 Currying and Partial Application

```python
# === Currying ===
# Converting a multi-argument function into a chain of single-argument functions

from functools import partial


# Normal function
def add(a: int, b: int) -> int:
    return a + b

# Curried function
def add_curried(a: int) -> Callable[[int], int]:
    def inner(b: int) -> int:
        return a + b
    return inner

add5 = add_curried(5)
print(add5(3))   # 8
print(add5(10))  # 15


# Generic currying helper
def curry(func: Callable) -> Callable:
    """Curry an arbitrary function"""
    import inspect
    params = inspect.signature(func).parameters
    arity = len(params)

    def curried(*args):
        if len(args) >= arity:
            return func(*args[:arity])
        return lambda *more_args: curried(*args, *more_args)

    return curried


@curry
def multiply(x: int, y: int) -> int:
    return x * y

double = multiply(2)
triple = multiply(3)
print(double(5))   # 10
print(triple(5))   # 15


@curry
def format_log(level: str, source: str, message: str) -> str:
    return f"[{level}] [{source}] {message}"

error_log = format_log("ERROR")
api_error = error_log("API")
print(api_error("Connection timeout"))  # [ERROR] [API] Connection timeout

db_error = error_log("DB")
print(db_error("Deadlock detected"))   # [ERROR] [DB] Deadlock detected


# === Partial Application ===
# Partially fix arguments using functools.partial

from functools import partial

def power(base: int, exponent: int) -> int:
    return base ** exponent

square = partial(power, exponent=2)
cube = partial(power, exponent=3)
print(square(5))  # 25
print(cube(3))    # 27


# Practical example: generate pre-configured functions
import json

def serialize(data: dict, indent: int = None, ensure_ascii: bool = True) -> str:
    return json.dumps(data, indent=indent, ensure_ascii=ensure_ascii)

# Partial application for pretty output with Unicode support
pretty_json = partial(serialize, indent=2, ensure_ascii=False)
compact_json = partial(serialize, indent=None, ensure_ascii=False)

data = {"name": "Taro Tanaka", "age": 30}
print(pretty_json(data))
# {
#   "name": "Taro Tanaka",
#   "age": 30
# }
print(compact_json(data))
# {"name": "Taro Tanaka", "age": 30}
```

---

## 4. Monads and Error Handling

### 4.1 Option/Maybe Pattern

```python
# === Option (Maybe) Pattern ===
# Safely and concisely handle chains of None checks

from __future__ import annotations
from typing import TypeVar, Generic, Callable, Optional
from dataclasses import dataclass

T = TypeVar('T')
U = TypeVar('U')


class Option(Generic[T]):
    """Option type: a value that may or may not exist"""

    @staticmethod
    def some(value: T) -> 'Option[T]':
        return Some(value)

    @staticmethod
    def none() -> 'Option[T]':
        return _None()

    @staticmethod
    def of(value: Optional[T]) -> 'Option[T]':
        """Return _None if None, otherwise return Some"""
        if value is None:
            return _None()
        return Some(value)

    def map(self, func: Callable[[T], U]) -> 'Option[U]':
        raise NotImplementedError

    def flat_map(self, func: Callable[[T], 'Option[U]']) -> 'Option[U]':
        raise NotImplementedError

    def get_or_else(self, default: T) -> T:
        raise NotImplementedError

    def or_else(self, alternative: Callable[[], 'Option[T]']) -> 'Option[T]':
        raise NotImplementedError

    def filter(self, predicate: Callable[[T], bool]) -> 'Option[T]':
        raise NotImplementedError

    def is_present(self) -> bool:
        raise NotImplementedError


@dataclass(frozen=True)
class Some(Option[T]):
    value: T

    def map(self, func: Callable[[T], U]) -> Option[U]:
        return Some(func(self.value))

    def flat_map(self, func: Callable[[T], Option[U]]) -> Option[U]:
        return func(self.value)

    def get_or_else(self, default: T) -> T:
        return self.value

    def or_else(self, alternative: Callable[[], Option[T]]) -> Option[T]:
        return self

    def filter(self, predicate: Callable[[T], bool]) -> Option[T]:
        return self if predicate(self.value) else _None()

    def is_present(self) -> bool:
        return True


class _None(Option[T]):
    def map(self, func: Callable[[T], U]) -> Option[U]:
        return _None()

    def flat_map(self, func: Callable[[T], Option[U]]) -> Option[U]:
        return _None()

    def get_or_else(self, default: T) -> T:
        return default

    def or_else(self, alternative: Callable[[], Option[T]]) -> Option[T]:
        return alternative()

    def filter(self, predicate: Callable[[T], bool]) -> Option[T]:
        return self

    def is_present(self) -> bool:
        return False


# Usage example: eliminate nested None checks
def find_user(user_id: int) -> Option[dict]:
    users = {1: {"name": "Tanaka", "department_id": 10}}
    return Option.of(users.get(user_id))

def find_department(dept_id: int) -> Option[dict]:
    departments = {10: {"name": "Development", "manager_id": 100}}
    return Option.of(departments.get(dept_id))

def find_manager(manager_id: int) -> Option[dict]:
    managers = {100: {"name": "Manager Sato", "email": "sato@example.com"}}
    return Option.of(managers.get(manager_id))


# Traditional chain of None checks
def get_manager_email_imperative(user_id: int) -> str:
    user = find_user(user_id)
    if user.is_present():
        dept = find_department(user.get_or_else({}).get("department_id"))
        if dept.is_present():
            mgr = find_manager(dept.get_or_else({}).get("manager_id"))
            if mgr.is_present():
                return mgr.get_or_else({}).get("email", "unknown")
    return "unknown"


# Chaining with Option's flat_map
def get_manager_email_functional(user_id: int) -> str:
    return (
        find_user(user_id)
        .flat_map(lambda u: find_department(u.get("department_id")))
        .flat_map(lambda d: find_manager(d.get("manager_id")))
        .map(lambda m: m.get("email"))
        .get_or_else("unknown")
    )

print(get_manager_email_functional(1))   # sato@example.com
print(get_manager_email_functional(999)) # unknown
```

### 4.2 Result/Either Pattern

```python
# === Result (Either) Pattern ===
# Represent success/failure as types and propagate errors without exceptions

from __future__ import annotations
from typing import TypeVar, Generic, Callable, Union
from dataclasses import dataclass

T = TypeVar('T')
U = TypeVar('U')
E = TypeVar('E')


class Result(Generic[T, E]):
    """Result type: either success (Ok) or failure (Err)"""

    @staticmethod
    def ok(value: T) -> 'Result[T, E]':
        return Ok(value)

    @staticmethod
    def err(error: E) -> 'Result[T, E]':
        return Err(error)

    def map(self, func: Callable[[T], U]) -> 'Result[U, E]':
        raise NotImplementedError

    def flat_map(self, func: Callable[[T], 'Result[U, E]']) -> 'Result[U, E]':
        raise NotImplementedError

    def map_err(self, func: Callable[[E], Exception]) -> 'Result[T, Exception]':
        raise NotImplementedError

    def unwrap(self) -> T:
        raise NotImplementedError

    def unwrap_or(self, default: T) -> T:
        raise NotImplementedError

    def is_ok(self) -> bool:
        raise NotImplementedError


@dataclass(frozen=True)
class Ok(Result[T, E]):
    value: T

    def map(self, func: Callable[[T], U]) -> Result[U, E]:
        return Ok(func(self.value))

    def flat_map(self, func: Callable[[T], Result[U, E]]) -> Result[U, E]:
        return func(self.value)

    def map_err(self, func):
        return self

    def unwrap(self) -> T:
        return self.value

    def unwrap_or(self, default: T) -> T:
        return self.value

    def is_ok(self) -> bool:
        return True


@dataclass(frozen=True)
class Err(Result[T, E]):
    error: E

    def map(self, func):
        return self

    def flat_map(self, func):
        return self

    def map_err(self, func):
        return Err(func(self.error))

    def unwrap(self) -> T:
        raise RuntimeError(f"Unwrapped an Err: {self.error}")

    def unwrap_or(self, default: T) -> T:
        return default

    def is_ok(self) -> bool:
        return False


# Practical example: user registration validation
def validate_email(email: str) -> Result[str, str]:
    if "@" not in email:
        return Result.err("Email address does not contain @")
    if len(email) < 5:
        return Result.err("Email address is too short")
    return Result.ok(email)

def validate_password(password: str) -> Result[str, str]:
    if len(password) < 8:
        return Result.err("Password must be at least 8 characters")
    if not any(c.isdigit() for c in password):
        return Result.err("Password must contain a digit")
    return Result.ok(password)

def validate_name(name: str) -> Result[str, str]:
    if not name or len(name) < 2:
        return Result.err("Name must be at least 2 characters")
    return Result.ok(name)

def register_user(name: str, email: str, password: str) -> Result[dict, str]:
    """User registration: validation -> registration"""
    return (
        validate_name(name)
        .flat_map(lambda n:
            validate_email(email)
            .flat_map(lambda e:
                validate_password(password)
                .map(lambda p: {
                    "name": n,
                    "email": e,
                    "password_hash": f"hashed_{p}"
                })
            )
        )
    )

# Success case
result = register_user("Taro Tanaka", "tanaka@example.com", "password123")
if result.is_ok():
    print(f"Registration successful: {result.unwrap()}")

# Failure case
result = register_user("Taro Tanaka", "invalid-email", "pass")
if not result.is_ok():
    print(f"Registration failed: {result.unwrap_or({})}")  # Returns default value
```

```rust
// Result and Option in Rust (built into the language)

use std::fs;
use std::num::ParseIntError;

// Result<T, E> is a standard type in Rust
fn read_number_from_file(path: &str) -> Result<i32, String> {
    // The ? operator automatically propagates errors
    let content = fs::read_to_string(path)
        .map_err(|e| format!("File read failed: {}", e))?;

    let number: i32 = content.trim().parse()
        .map_err(|e: ParseIntError| format!("Number parse failed: {}", e))?;

    Ok(number)
}

// Transformation via method chaining
fn process_config(path: &str) -> Result<String, String> {
    read_number_from_file(path)
        .map(|n| n * 2)                          // Transform the value
        .and_then(|n| {                           // Chain with a function that returns Result
            if n > 0 {
                Ok(format!("Result: {}", n))
            } else {
                Err("Negative value".to_string())
            }
        })
}

// Option<T>
fn find_user(id: u32) -> Option<User> {
    // Some(user) or None
    users.iter().find(|u| u.id == id).cloned()
}

fn get_user_email(id: u32) -> Option<String> {
    find_user(id)
        .filter(|u| u.is_active)
        .map(|u| u.email.clone())
}

// Default value with unwrap_or_else
let email = get_user_email(42)
    .unwrap_or_else(|| "unknown@example.com".to_string());
```

---

## 5. Pattern Matching

### 5.1 Algebraic Data Types and Pattern Matching

```python
# Structural pattern matching in Python 3.10+

from dataclasses import dataclass
from typing import Union


# Representing algebraic data types (sum types) with classes
@dataclass(frozen=True)
class Circle:
    radius: float

@dataclass(frozen=True)
class Rectangle:
    width: float
    height: float

@dataclass(frozen=True)
class Triangle:
    base: float
    height: float

Shape = Union[Circle, Rectangle, Triangle]


def area(shape: Shape) -> float:
    """Calculate area using pattern matching"""
    match shape:
        case Circle(radius=r):
            return 3.14159 * r ** 2
        case Rectangle(width=w, height=h):
            return w * h
        case Triangle(base=b, height=h):
            return b * h / 2
        case _:
            raise ValueError(f"Unknown shape: {shape}")


def describe(shape: Shape) -> str:
    """Describe the shape using pattern matching"""
    match shape:
        case Circle(radius=r) if r > 100:
            return f"Large circle (radius {r})"
        case Circle(radius=r):
            return f"Circle (radius {r})"
        case Rectangle(width=w, height=h) if w == h:
            return f"Square (side {w})"
        case Rectangle(width=w, height=h):
            return f"Rectangle ({w}x{h})"
        case Triangle(base=b, height=h):
            return f"Triangle (base {b}, height {h})"


# Usage example
shapes = [Circle(5), Rectangle(3, 4), Rectangle(5, 5), Triangle(6, 8)]
for s in shapes:
    print(f"{describe(s)}: area = {area(s):.2f}")


# Pattern matching for JSON/API responses
def handle_response(response: dict) -> str:
    match response:
        case {"status": "success", "data": data}:
            return f"Success: {data}"
        case {"status": "error", "code": code, "message": msg}:
            return f"Error({code}): {msg}"
        case {"status": "redirect", "url": url}:
            return f"Redirect to: {url}"
        case {"status": status}:
            return f"Unknown status: {status}"
        case _:
            return "Invalid response format"


# Pattern matching for commands
def execute_command(command: list[str]) -> str:
    match command:
        case ["quit" | "exit"]:
            return "Exiting"
        case ["help", topic]:
            return f"Help: {topic}"
        case ["search", *keywords]:
            return f"Search: {' '.join(keywords)}"
        case ["add", name, value] if value.isdigit():
            return f"Added {name} = {value}"
        case [cmd, *args]:
            return f"Unknown command: {cmd} (args: {args})"
        case _:
            return "Command is empty"
```

---

## 6. Generators and Iterators (Lazy Evaluation)

```python
# === Lazy Evaluation ===
# Defer computation until it is needed

from typing import Iterator, Iterable


# Generator: lazily produce data
def fibonacci() -> Iterator[int]:
    """Infinite Fibonacci sequence"""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

def take(n: int, iterable: Iterable) -> list:
    """Take the first n elements"""
    result = []
    for i, item in enumerate(iterable):
        if i >= n:
            break
        result.append(item)
    return result

# Take only the first 10 from the infinite sequence (not loaded entirely in memory)
print(take(10, fibonacci()))  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]


# Generator expressions: memory-efficient data processing
def process_large_file(filepath: str) -> Iterator[dict]:
    """Lazily process a large CSV file line by line"""
    with open(filepath) as f:
        header = next(f).strip().split(',')
        for line in f:
            values = line.strip().split(',')
            yield dict(zip(header, values))

# Even a 100-million-line file won't exhaust memory
# for record in process_large_file("huge.csv"):
#     if record["status"] == "error":
#         print(record)


# Generator pipeline
def lines(filepath: str) -> Iterator[str]:
    with open(filepath) as f:
        yield from f

def non_empty(lines: Iterator[str]) -> Iterator[str]:
    return (line for line in lines if line.strip())

def parse_json_lines(lines: Iterator[str]) -> Iterator[dict]:
    import json
    for line in lines:
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue

def filter_by(key: str, value, records: Iterator[dict]) -> Iterator[dict]:
    return (r for r in records if r.get(key) == value)


# Build a pipeline (all lazy evaluation, memory efficient)
# errors = filter_by(
#     "level", "ERROR",
#     parse_json_lines(
#         non_empty(
#             lines("app.log")
#         )
#     )
# )
# for error in errors:
#     print(error)


# itertools: standard library lazy utilities
import itertools

# chain: concatenate multiple iterables
combined = itertools.chain([1, 2, 3], [4, 5, 6])
# -> 1, 2, 3, 4, 5, 6

# islice: slicing
first_5_fibs = list(itertools.islice(fibonacci(), 5))
# -> [0, 1, 1, 2, 3]

# groupby: grouping
data = [
    ("A", 1), ("A", 2), ("B", 3), ("B", 4), ("A", 5)
]
sorted_data = sorted(data, key=lambda x: x[0])
for key, group in itertools.groupby(sorted_data, key=lambda x: x[0]):
    print(f"{key}: {list(group)}")

# takewhile / dropwhile
numbers = [2, 4, 6, 1, 3, 5]
even_prefix = list(itertools.takewhile(lambda x: x % 2 == 0, numbers))
# -> [2, 4, 6]

# accumulate: cumulative computation
running_total = list(itertools.accumulate([1, 2, 3, 4, 5]))
# -> [1, 3, 6, 10, 15]

# product: Cartesian product (all combinations)
combos = list(itertools.product(['A', 'B'], [1, 2]))
# -> [('A', 1), ('A', 2), ('B', 1), ('B', 2)]
```

---

## 7. Functional Techniques in Practice

### 7.1 Data Transformation Pipeline (ETL)

```python
# === Practical Data Transformation Pipeline ===

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Callable, Iterator
import json


@dataclass(frozen=True)
class SalesRecord:
    """Sales record"""
    date: str
    product: str
    category: str
    amount: int
    quantity: int
    region: str


# Define each step as a pure function
def parse_records(raw_data: list[dict]) -> list[SalesRecord]:
    """Convert raw data to records"""
    return [SalesRecord(**record) for record in raw_data]


def filter_date_range(
    start: str, end: str
    """Return a function that filters by date range (closure)"""
    def _filter(records: list[SalesRecord]) -> list[SalesRecord]:
        return [r for r in records if start <= r.date <= end]
    return _filter


def filter_category(
    category: str
    """Return a function that filters by category"""
    def _filter(records: list[SalesRecord]) -> list[SalesRecord]:
        return [r for r in records if r.category == category]
    return _filter


def group_by_region(records: list[SalesRecord]) -> dict[str, list[SalesRecord]]:
    """Group by region"""
    groups: dict[str, list[SalesRecord]] = {}
    for record in records:
        groups.setdefault(record.region, []).append(record)
    return groups


def summarize_groups(
    groups: dict[str, list[SalesRecord]]
) -> dict[str, dict]:
    """Aggregate per group"""
    return {
        region: {
            "total_amount": sum(r.amount for r in records),
            "total_quantity": sum(r.quantity for r in records),
            "record_count": len(records),
            "avg_amount": sum(r.amount for r in records) // len(records),
        }
        for region, records in groups.items()
    }


def sort_by_total(summary: dict[str, dict]) -> list[tuple[str, dict]]:
    """Sort by total amount in descending order"""
    return sorted(
        summary.items(),
        key=lambda x: x[1]["total_amount"],
        reverse=True
    )


def format_report(sorted_summary: list[tuple[str, dict]]) -> str:
    """Format the report"""
    lines = ["=== Regional Sales Report ==="]
    for region, data in sorted_summary:
        lines.append(
            f"  {region}: \u00a5{data['total_amount']:,} "
            f"({data['record_count']} records, "
            f"avg \u00a5{data['avg_amount']:,})"
        )
    return "\n".join(lines)


# Build and execute the pipeline
raw_data = [
    {"date": "2025-01-15", "product": "Laptop", "category": "Electronics",
     "amount": 150000, "quantity": 1, "region": "Tokyo"},
    {"date": "2025-01-16", "product": "Mouse", "category": "Electronics",
     "amount": 3000, "quantity": 5, "region": "Osaka"},
    {"date": "2025-01-17", "product": "Desk", "category": "Furniture",
     "amount": 50000, "quantity": 2, "region": "Tokyo"},
    {"date": "2025-01-18", "product": "Monitor", "category": "Electronics",
     "amount": 40000, "quantity": 3, "region": "Tokyo"},
    {"date": "2025-01-19", "product": "Keyboard", "category": "Electronics",
     "amount": 8000, "quantity": 10, "region": "Osaka"},
]

report = pipeline(
    raw_data,
    parse_records,
    filter_date_range("2025-01-15", "2025-01-19"),
    filter_category("Electronics"),
    group_by_region,
    summarize_groups,
    sort_by_total,
    format_report,
)
print(report)
```

### 7.2 Functional Patterns in Web Frameworks

```python
# === Functional Patterns in FastAPI ===

from functools import wraps
from typing import Callable


# Decorators as middleware (higher-order functions)
def require_auth(func: Callable) -> Callable:
    """Authentication required decorator"""
    @wraps(func)
    async def wrapper(request, *args, **kwargs):
        token = request.headers.get("Authorization")
        if not token:
            return {"error": "Authentication token required"}, 401
        user = await verify_token(token)
        if not user:
            return {"error": "Invalid token"}, 403
        return await func(request, user=user, *args, **kwargs)
    return wrapper


def validate_body(schema: type) -> Callable:
    """Request body validation decorator"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(request, *args, **kwargs):
            try:
                body = schema(**await request.json())
            except Exception as e:
                return {"error": f"Validation error: {e}"}, 400
            return await func(request, body=body, *args, **kwargs)
        return wrapper
    return decorator


def rate_limit(max_calls: int, window_seconds: int) -> Callable:
    """Rate limiting decorator"""
    from collections import defaultdict
    import time
    calls: dict[str, list[float]] = defaultdict(list)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(request, *args, **kwargs):
            client_ip = request.client.host
            now = time.time()
            # Remove old call records
            calls[client_ip] = [
                t for t in calls[client_ip]
                if now - t < window_seconds
            ]
            if len(calls[client_ip]) >= max_calls:
                return {"error": "Rate limit exceeded"}, 429
            calls[client_ip].append(now)
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator
```

```javascript
// Functional patterns in React

// Custom hooks: reuse logic via functional composition
function useDebounce(value, delay) {
    const [debouncedValue, setDebouncedValue] = useState(value);

    useEffect(() => {
        const timer = setTimeout(() => setDebouncedValue(value), delay);
        return () => clearTimeout(timer);
    }, [value, delay]);

    return debouncedValue;
}

function useLocalStorage(key, initialValue) {
    const [storedValue, setStoredValue] = useState(() => {
        try {
            const item = window.localStorage.getItem(key);
            return item ? JSON.parse(item) : initialValue;
        } catch {
            return initialValue;
        }
    });

    const setValue = (value) => {
        // Support functional updates
        const valueToStore = value instanceof Function
            ? value(storedValue) : value;
        setStoredValue(valueToStore);
        window.localStorage.setItem(key, JSON.stringify(valueToStore));
    };

    return [storedValue, setValue];
}

// Reducer pattern (functional state management)
function todoReducer(state, action) {
    // Pure function: same state + action -> same result
    switch (action.type) {
        case 'ADD':
            return {
                ...state,
                todos: [...state.todos, {
                    id: Date.now(),
                    text: action.text,
                    done: false
                }],
            };
        case 'TOGGLE':
            return {
                ...state,
                todos: state.todos.map(todo =>
                    todo.id === action.id
                        ? { ...todo, done: !todo.done }
                        : todo
                ),
            };
        case 'FILTER':
            return { ...state, filter: action.filter };
        default:
            return state;
    }
}
```

### 7.3 Functional Error Handling in Practice

```python
# === Railway Oriented Programming ===
# Flow processing along rails (success/failure)

from typing import Callable, TypeVar

T = TypeVar('T')
U = TypeVar('U')


def railway(*steps: Callable) -> Callable:
    """Railway pattern that chains Results"""
    def run(input_data):
        result = Result.ok(input_data)
        for step in steps:
            if result.is_ok():
                try:
                    result = step(result.unwrap())
                except Exception as e:
                    result = Result.err(str(e))
            # If error, skip (travels the error rail)
        return result
    return run


# Define each step as a function
def validate_input(data: dict) -> Result:
    if not data.get("email"):
        return Result.err("Email address is required")
    if not data.get("name"):
        return Result.err("Name is required")
    return Result.ok(data)

def normalize_data(data: dict) -> Result:
    return Result.ok({
        **data,
        "email": data["email"].lower().strip(),
        "name": data["name"].strip(),
    })

def check_duplicate(data: dict) -> Result:
    # DB check (simulation)
    existing_emails = ["existing@example.com"]
    if data["email"] in existing_emails:
        return Result.err("This email address is already registered")
    return Result.ok(data)

def save_to_database(data: dict) -> Result:
    # DB save (simulation)
    return Result.ok({**data, "id": 42})

def send_welcome_email(data: dict) -> Result:
    # Email send (simulation)
    print(f"Sending welcome email: {data['email']}")
    return Result.ok(data)


# Build the railway pipeline
register_user = railway(
    validate_input,
    normalize_data,
    check_duplicate,
    save_to_database,
    send_welcome_email,
)

# Success case
result = register_user({"name": "Taro Tanaka", "email": "Tanaka@Example.COM"})
print(result)  # Ok({"name": "Taro Tanaka", "email": "tanaka@example.com", "id": 42})

# Failure case (validation error)
result = register_user({"name": "", "email": "test@example.com"})
print(result)  # Err("Name is required")

# Failure case (duplicate error)
result = register_user({"name": "Test", "email": "existing@example.com"})
print(result)  # Err("This email address is already registered")
```

---

## 8. Functional Programming Support by Language

```
Functional Programming Support by Language:

+----------------+---------+----------+---------+----------+--------+
|                | Python  | JS/TS    | Java    | Rust     | Haskell|
+----------------+---------+----------+---------+----------+--------+
| First-class fn | Yes     | Yes      | Yes(8+) | Yes      | Yes    |
| Lambda expr    | Yes(1)  | Yes      | Yes     | Yes      | Yes    |
| Closures       | Yes     | Yes      | Yes     | Yes      | Yes    |
| map/filter     | Yes     | Yes      | Yes     | Yes      | Yes    |
|                |         |          | Stream  | iter     |        |
| Pattern match  | Yes     | No       | Yes(21+)| Yes      | Yes    |
|                | (3.10)  |          |         |          |        |
| Immutability   | frozen  | const    | final   | Default  | Default|
| Algebraic types| Union   | Union    | sealed  | enum     | data   |
| Option type    | Optional| ?./??    | Optional| Option   | Maybe  |
| Result type    | None*   | None*    | None*   | Result   | Either |
| Lazy eval      |generator|generator | Stream  | iterator | Default|
| Type inference | mypy    | TS       | var     | Strong   | Strong |
| Purity enforced| No      | No       | No      | Partial  | Yes    |
| Tail call opt  | No      | Partial  | No      | No       | Yes    |
| Monads         | None*   | Promise  | Optional| Idiomatic| Yes    |
| Currying       | partial | Yes      | None*   | None*    | Default|
+----------------+---------+----------+---------+----------+--------+

* Achievable via libraries

Recommendations:
- To learn pure FP -> Haskell, Elm
- FP in practice -> TypeScript, Rust, Scala
- Introduce FP elements to existing languages -> Python, Java, JavaScript
```

---

## 9. Trade-offs of Functional Programming

```
Benefits of Functional Programming:

  1. Testability
     - Pure functions only test input and output
     - No mocks needed, no setup needed
     - Good compatibility with property-based testing

  2. Concurrency Safety
     - Immutable data -> no locks needed
     - No shared state -> no race conditions
     - Affinity with message passing

  3. Ease of Reasoning
     - Referential transparency -> can understand partially
     - Localized side effects -> easier to identify bugs
     - Equational reasoning -> safe refactoring

  4. Composability
     - Build large processing by combining small functions
     - High code reusability
     - Pipeline-style description possible

Challenges of Functional Programming:

  1. Learning Curve
     - Abstract concepts like monads are difficult
     - Not intuitive for developers accustomed to imperative style
     - Error messages can be unclear in some cases

  2. Performance
     - Update cost of immutable data structures (copying)
     - Stack overflow from recursion (languages without tail call optimization)
     - GC pressure (mass generation of short-lived objects)

  3. State Management
     - Applying to inherently stateful problems (GUI, games, etc.) is complex
     - Requires monads or effect systems
     - Boundary design with databases and file I/O

  4. Debugging
     - Debugging lazy evaluation is difficult
     - Stack traces are hard to follow in chains of higher-order functions
     - Readability issues with point-free style

  Recommended Practical Approach:
  +-----------------------------------------------------+
  | Don't insist on "pure functional,"                   |
  | introduce "the essence of FP" where appropriate      |
  |                                                       |
  | - Business logic -> pure functions                    |
  | - I/O, side effects -> straightforward procedural    |
  | - Data transformation -> map/filter/reduce            |
  | - State management -> immutability + Reducer pattern  |
  | - Error handling -> Result/Option pattern              |
  +-----------------------------------------------------+
```

---


## FAQ

### Q1: What is the most important point in learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory, but by actually writing code and verifying how it works.

### Q2: What are common mistakes beginners make?

Skipping the basics and jumping to advanced topics. We recommend thoroughly understanding the fundamental concepts explained in this guide before moving on to the next step.

### Q3: How is this applied in practice?

Knowledge of this topic is frequently used in everyday development work. It becomes particularly important during code reviews and architecture design.

---

## Summary

| Concept | Key Points |
|---------|-----------|
| Pure Functions | No side effects, same input yields same output. Easy to test, concurrency safe |
| Referential Transparency | Expressions can be replaced with their values. Enables memoization, lazy evaluation, parallelization |
| Higher-Order Functions | map/filter/reduce. Fundamental tools for data transformation |
| Immutability | Generate new data instead of modifying. Thread safe |
| Function Composition | Build complex processing by combining small functions |
| Currying/Partial Application | Fix arguments to generate specialized functions |
| Option/Result | Represent errors as types without null/exceptions |
| Pattern Matching | Safely destructure algebraic data types |
| Lazy Evaluation | Memory-efficient processing with generators/iterators |
| Pipelines | Declaratively describe chains of data transformations |

---

## Recommended Next Guides

---

## References
1. Hutton, G. "Programming in Haskell." 2nd Edition, Cambridge University Press, 2016.
2. Chiusano, P. & Bjarnason, R. "Functional Programming in Scala." Manning, 2014.
3. Bird, R. "Thinking Functionally with Haskell." Cambridge University Press, 2015.
4. Wlaschin, S. "Domain Modeling Made Functional." Pragmatic Bookshelf, 2018.
5. Fogus, M. "Functional JavaScript." O'Reilly, 2013.
6. Lipovaca, M. "Learn You a Haskell for Great Good!" No Starch Press, 2011.
7. Armstrong, J. "Programming Erlang." 2nd Edition, Pragmatic Bookshelf, 2013.
8. Backus, J. "Can Programming Be Liberated from the von Neumann Style?" ACM Turing Award Lecture, 1977.
9. Church, A. "The Calculi of Lambda Conversion." Princeton University Press, 1941.
10. Milner, R. et al. "The Definition of Standard ML." MIT Press, 1997.
