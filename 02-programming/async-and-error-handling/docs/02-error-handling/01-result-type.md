# Result Type

> The Result type is a technique for expressing "success or failure" through the type system. It handles errors explicitly without exceptions, allowing the compiler to detect "forgotten error handling." This guide compares implementations in Rust, Go, and TypeScript.

## What You Will Learn in This Chapter

- [ ] Understand the mechanism of the Result type and how it differs from exceptions
- [ ] Learn Result type implementations in each language
- [ ] Study the advantages and disadvantages of the Result type
- [ ] Understand monadic chaining operations (map, flatMap, andThen)
- [ ] Grasp the relationship between the Result type and Option/Maybe types
- [ ] Learn practical patterns for adopting the Result type


## Prerequisites

The following knowledge will deepen your understanding before reading this guide:

- Basic programming knowledge
- Understanding of related foundational concepts
- Familiarity with the content of [Exception Handling](./00-exceptions.md)

---

## 1. Exceptions vs Result Type

### 1.1 Fundamental Differences

```
Exceptions:
  function getUser(id: string): User {
    // The possibility of an error is not visible from the type
    // The caller might forget to use try/catch
  }

Result Type:
  function getUser(id: string): Result<User, AppError> {
    // Just by looking at the type, you know "failure is possible"
    // The compiler can enforce error handling
  }

Comparison:
  ┌──────────────────┬──────────────────┬──────────────────┐
  │                  │ Exceptions       │ Result Type      │
  ├──────────────────┼──────────────────┼──────────────────┤
  │ Error Visibility │ Not in the type  │ Visible in type  │
  ├──────────────────┼──────────────────┼──────────────────┤
  │ Enforcement      │ None             │ Compiler-enforced│
  ├──────────────────┼──────────────────┼──────────────────┤
  │ Code Style       │ try/catch        │ match/map/unwrap │
  ├──────────────────┼──────────────────┼──────────────────┤
  │ Performance      │ Stack unwinding  │ Normal return    │
  └──────────────────┴──────────────────┴──────────────────┘
```

### 1.2 Why the Result Type Is Gaining Attention

```
Reasons Why the Result Type Is Gaining Attention:

  1. Type Safety
     → Error types are explicitly declared in the type system
     → The compiler detects missing error handling
     → IDE autocompletion works effectively

  2. Explicitness
     → The possibility of failure is apparent from the function signature alone
     → No hidden control flow (exception propagation)
     → Easier code reviews

  3. Performance
     → No stack unwinding required
     → No stack trace construction needed
     → Same cost as a normal function return

  4. Composability
     → Chain processing with map, flatMap, andThen
     → High affinity with functional programming
     → Well-suited for pipeline processing

  5. Predictability
     → Error paths are clear
     → Tests are easy to write
     → Debugging is straightforward
```

### 1.3 Mathematical Background of the Result Type

```
Concepts Behind the Result Type:

  Sum Type (Tagged Union):
    Result<T, E> = Ok(T) | Err(E)
    → Always holds exactly one of T or E

  This is a kind of Algebraic Data Type (ADT):
    → Haskell: Either a b = Left a | Right b
    → Rust: enum Result<T, E> { Ok(T), Err(E) }
    → Scala: Either[L, R] = Left[L] | Right[R]
    → TypeScript: { ok: true; value: T } | { ok: false; error: E }

  Result as a Monad:
    → Chainable via flatMap (andThen)
    → Short-circuits when an error occurs
    → Has the same expressive power as try/catch with exceptions
```

---

## 2. Rust's Result

### 2.1 Basic Usage

```rust
// Rust: Result<T, E> is a standard library type
use std::fs;
use std::io;

fn read_config(path: &str) -> Result<Config, ConfigError> {
    let content = fs::read_to_string(path)
        .map_err(|e| ConfigError::IoError(e))?;  // ? for early return

    let config: Config = serde_json::from_str(&content)
        .map_err(|e| ConfigError::ParseError(e.to_string()))?;

    if config.port == 0 {
        return Err(ConfigError::ValidationError("port must be > 0".into()));
    }

    Ok(config)
}

// Error type definition
#[derive(Debug)]
enum ConfigError {
    IoError(io::Error),
    ParseError(String),
    ValidationError(String),
}

// Usage
fn main() {
    match read_config("config.json") {
        Ok(config) => println!("Port: {}", config.port),
        Err(ConfigError::IoError(e)) => eprintln!("File error: {}", e),
        Err(ConfigError::ParseError(e)) => eprintln!("Parse error: {}", e),
        Err(ConfigError::ValidationError(e)) => eprintln!("Validation: {}", e),
    }

    // Chain with the ? operator (propagate errors to the caller)
    // → Instead of try/catch, errors propagate through the type system
}

// Method chaining with Result
fn process() -> Result<String, Error> {
    read_file("input.txt")?
        .lines()
        .map(|line| parse_line(line))
        .collect::<Result<Vec<_>, _>>()?
        .iter()
        .map(|item| format_item(item))
        .collect::<Result<String, _>>()
}
```

### 2.2 The ? Operator in Detail

```rust
// The ? operator is syntactic sugar for the following:
fn read_file(path: &str) -> Result<String, io::Error> {
    // This:
    let content = fs::read_to_string(path)?;

    // Is equivalent to:
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => return Err(e.into()),  // Conversion via the From trait
    };

    Ok(content)
}

// Automatic error type conversion via the From trait
impl From<io::Error> for AppError {
    fn from(e: io::Error) -> Self {
        AppError::Io(e)
    }
}

impl From<serde_json::Error> for AppError {
    fn from(e: serde_json::Error) -> Self {
        AppError::Parse(e.to_string())
    }
}

// When From is implemented, the ? operator automatically converts
fn load_config(path: &str) -> Result<Config, AppError> {
    let content = fs::read_to_string(path)?;  // io::Error → AppError
    let config: Config = serde_json::from_str(&content)?;  // serde::Error → AppError
    Ok(config)
}
```

### 2.3 Result Method Reference

```rust
// Key methods on Result<T, E>

fn demonstrate_result_methods() {
    let ok_val: Result<i32, String> = Ok(42);
    let err_val: Result<i32, String> = Err("error".to_string());

    // ========== Extracting Values ==========

    // unwrap: Returns the value if Ok, panics if Err
    let value = ok_val.unwrap();  // 42
    // let value = err_val.unwrap();  // Panics! Do not use in production code

    // unwrap_or: Default value for Err case
    let value = err_val.unwrap_or(0);  // 0

    // unwrap_or_else: Generate a value via closure for Err case
    let value = err_val.unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        0
    });

    // expect: Same as unwrap but with a custom panic message
    let value = ok_val.expect("Config must be valid");

    // ========== Transformations ==========

    // map: Transform the Ok value
    let mapped: Result<String, String> = ok_val.map(|v| v.to_string());

    // map_err: Transform the Err value
    let mapped: Result<i32, i32> = err_val.map_err(|e| e.len() as i32);

    // and_then (flatMap): Apply a function that returns a Result (chaining)
    let chained: Result<String, String> = ok_val.and_then(|v| {
        if v > 0 {
            Ok(v.to_string())
        } else {
            Err("must be positive".to_string())
        }
    });

    // or_else: Return a different Result for the Err case
    let recovered: Result<i32, String> = err_val.or_else(|e| {
        eprintln!("Recovering from: {}", e);
        Ok(0)
    });

    // ========== Checks ==========

    // is_ok / is_err
    assert!(ok_val.is_ok());
    assert!(err_val.is_err());

    // ========== Conversion to/from Option ==========

    // ok(): Result<T, E> → Option<T>
    let opt: Option<i32> = ok_val.ok();  // Some(42)
    let opt: Option<i32> = err_val.ok();  // None

    // err(): Result<T, E> → Option<E>
    let opt: Option<String> = err_val.err();  // Some("error")

    // transpose: Result<Option<T>, E> → Option<Result<T, E>>
    let x: Result<Option<i32>, String> = Ok(Some(42));
    let y: Option<Result<i32, String>> = x.transpose();  // Some(Ok(42))
}

// collect to convert Vec<Result<T, E>> → Result<Vec<T>, E>
fn parse_all(inputs: &[&str]) -> Result<Vec<i32>, String> {
    inputs
        .iter()
        .map(|s| s.parse::<i32>().map_err(|e| e.to_string()))
        .collect()  // Short-circuits on the first Err
}
```

### 2.4 thiserror and anyhow

```rust
// thiserror: Structured errors for libraries
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("User not found: {user_id}")]
    UserNotFound { user_id: String },

    #[error("Email already exists: {email}")]
    EmailAlreadyExists { email: String },

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Database error")]
    Database(#[from] sqlx::Error),

    #[error("External API error")]
    ExternalApi(#[from] reqwest::Error),

    #[error("Internal error: {0}")]
    Internal(String),
}

// anyhow: Flexible errors for applications
use anyhow::{Context, Result, bail, ensure};

fn load_config(path: &str) -> Result<Config> {
    let content = fs::read_to_string(path)
        .context("Failed to read config file")?;  // Add context

    let config: Config = serde_json::from_str(&content)
        .context("Failed to parse config")?;

    ensure!(config.port > 0, "Port must be positive, got {}", config.port);
    // ensure! returns Err if the condition is false

    if config.host.is_empty() {
        bail!("Host cannot be empty");
        // bail! immediately returns Err
    }

    Ok(config)
}

// When to use thiserror vs anyhow:
// thiserror: Libraries (callers need to distinguish between error types)
// anyhow: Applications (human-readable messages are sufficient for error details)
```

---

## 3. Go Errors

### 3.1 Basic Pattern

```go
// Go: Returns errors via multiple return values
func readConfig(path string) (*Config, error) {
    data, err := os.ReadFile(path)
    if err != nil {
        return nil, fmt.Errorf("failed to read config: %w", err)
    }

    var config Config
    if err := json.Unmarshal(data, &config); err != nil {
        return nil, fmt.Errorf("failed to parse config: %w", err)
    }

    if config.Port == 0 {
        return nil, errors.New("port must be > 0")
    }

    return &config, nil
}

// Usage
func main() {
    config, err := readConfig("config.json")
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Port: %d\n", config.Port)
}

// Characteristics of Go error handling:
// → Errors are values (the error interface)
// → if err != nil is ubiquitous (controversial)
// → errors.Is, errors.As for error inspection
// → fmt.Errorf("%w", err) for error wrapping
```

### 3.2 Custom Error Types

```go
// Custom error type definitions
type NotFoundError struct {
    Resource string
    ID       string
}

func (e *NotFoundError) Error() string {
    return fmt.Sprintf("%s not found: %s", e.Resource, e.ID)
}

type ValidationError struct {
    Field   string
    Message string
}

func (e *ValidationError) Error() string {
    return fmt.Sprintf("validation error on %s: %s", e.Field, e.Message)
}

// Sentinel errors (constant errors)
var (
    ErrNotFound      = errors.New("not found")
    ErrUnauthorized  = errors.New("unauthorized")
    ErrInternalError = errors.New("internal error")
)

// Error inspection
func handleError(err error) {
    // errors.Is: Compare with sentinel errors
    if errors.Is(err, ErrNotFound) {
        fmt.Println("Resource not found")
        return
    }

    // errors.As: Check for custom error types
    var validationErr *ValidationError
    if errors.As(err, &validationErr) {
        fmt.Printf("Validation failed: %s - %s\n",
            validationErr.Field, validationErr.Message)
        return
    }

    // Unknown error
    fmt.Printf("Unknown error: %v\n", err)
}
```

### 3.3 Error Wrapping Chains

```go
// Go 1.13+: Error wrapping
func getUser(id string) (*User, error) {
    row := db.QueryRow("SELECT * FROM users WHERE id = ?", id)
    var user User
    if err := row.Scan(&user.ID, &user.Name, &user.Email); err != nil {
        if errors.Is(err, sql.ErrNoRows) {
            return nil, fmt.Errorf("getUser(%s): %w", id, ErrNotFound)
        }
        return nil, fmt.Errorf("getUser(%s): database error: %w", id, err)
    }
    return &user, nil
}

func createOrder(userID string, items []Item) (*Order, error) {
    user, err := getUser(userID)
    if err != nil {
        return nil, fmt.Errorf("createOrder: %w", err)
    }
    // ...
    return &Order{UserID: user.ID}, nil
}

// Example of an error chain:
// "createOrder: getUser(user-123): not found"
// errors.Is(err, ErrNotFound) → true (traverses the chain for inspection)

// Go 1.20+: errors.Join to combine multiple errors
func validateOrder(order *Order) error {
    var errs []error

    if order.UserID == "" {
        errs = append(errs, &ValidationError{Field: "userID", Message: "required"})
    }
    if len(order.Items) == 0 {
        errs = append(errs, &ValidationError{Field: "items", Message: "at least one item required"})
    }
    if order.Total < 0 {
        errs = append(errs, &ValidationError{Field: "total", Message: "must be non-negative"})
    }

    if len(errs) > 0 {
        return errors.Join(errs...)
    }
    return nil
}
```

### 3.4 The Debate Over Go's Error Handling

```
The Debate Over Go's Error Handling:

  Proponents:
  → Simple and explicit
  → Hard to ignore errors (detected by lint tools)
  → No cost of stack traces
  → Error propagation is transparent

  Critics:
  → Boilerplate from if err != nil
  → Error handling dominates the codebase
  → Cannot compose like Result type's map/flatMap
  → No exhaustiveness checking through the type system

  Go 2 Proposals (Drafts):
  → check/handle syntax (proposed 2018, not adopted)
  → try built-in function (proposed 2019, rejected)
  → In the end, if err != nil remains

  Practical Mitigations:
  → Helper functions to reduce boilerplate
  → errgroup for aggregating goroutine errors
  → Structured logging to supplement error context
```

---

## 4. Result Type in TypeScript

### 4.1 Simple Implementation

```typescript
// TypeScript: Result type implementation
type Result<T, E = Error> =
  | { ok: true; value: T }
  | { ok: false; error: E };

function ok<T>(value: T): Result<T, never> {
  return { ok: true, value };
}

function err<E>(error: E): Result<never, E> {
  return { ok: false, error };
}

// Usage example
function parseJson<T>(text: string): Result<T, string> {
  try {
    return ok(JSON.parse(text));
  } catch (e) {
    return err(`Invalid JSON: ${(e as Error).message}`);
  }
}

function validateUser(data: unknown): Result<User, ValidationError> {
  if (!data || typeof data !== "object") {
    return err({ field: "root", message: "Invalid data" });
  }
  const { name, email } = data as any;
  if (!name) return err({ field: "name", message: "Name is required" });
  if (!email?.includes("@")) return err({ field: "email", message: "Invalid email" });
  return ok({ name, email } as User);
}

// Chaining usage
function processInput(input: string): Result<User, string> {
  const jsonResult = parseJson<unknown>(input);
  if (!jsonResult.ok) return err(jsonResult.error);

  const userResult = validateUser(jsonResult.value);
  if (!userResult.ok) return err(userResult.error.message);

  return userResult;
}
```

### 4.2 Feature-Rich Result Class Implementation

```typescript
// A more feature-rich Result implementation
class Result<T, E> {
    private constructor(
        private readonly _ok: boolean,
        private readonly _value?: T,
        private readonly _error?: E,
    ) {}

    static ok<T>(value: T): Result<T, never> {
        return new Result(true, value);
    }

    static err<E>(error: E): Result<never, E> {
        return new Result(false, undefined, error);
    }

    // Convert exceptions to Result
    static fromThrowable<T>(fn: () => T): Result<T, Error> {
        try {
            return Result.ok(fn());
        } catch (e) {
            return Result.err(e instanceof Error ? e : new Error(String(e)));
        }
    }

    // Async version
    static async fromPromise<T>(promise: Promise<T>): Promise<Result<T, Error>> {
        try {
            return Result.ok(await promise);
        } catch (e) {
            return Result.err(e instanceof Error ? e : new Error(String(e)));
        }
    }

    isOk(): this is Result<T, never> {
        return this._ok;
    }

    isErr(): this is Result<never, E> {
        return !this._ok;
    }

    // Transform the Ok value
    map<U>(fn: (value: T) => U): Result<U, E> {
        if (this._ok) {
            return Result.ok(fn(this._value!));
        }
        return Result.err(this._error!);
    }

    // Transform the Err value
    mapErr<F>(fn: (error: E) => F): Result<T, F> {
        if (this._ok) {
            return Result.ok(this._value!);
        }
        return Result.err(fn(this._error!));
    }

    // flatMap / andThen: Chain a function that returns a Result
    andThen<U>(fn: (value: T) => Result<U, E>): Result<U, E> {
        if (this._ok) {
            return fn(this._value!);
        }
        return Result.err(this._error!);
    }

    // Recovery on Err
    orElse<F>(fn: (error: E) => Result<T, F>): Result<T, F> {
        if (this._ok) {
            return Result.ok(this._value!);
        }
        return fn(this._error!);
    }

    // Extract the value
    unwrap(): T {
        if (this._ok) return this._value!;
        throw new Error(`Called unwrap on Err: ${this._error}`);
    }

    unwrapOr(defaultValue: T): T {
        return this._ok ? this._value! : defaultValue;
    }

    unwrapOrElse(fn: (error: E) => T): T {
        return this._ok ? this._value! : fn(this._error!);
    }

    // Pattern matching
    match<U>(handlers: { ok: (value: T) => U; err: (error: E) => U }): U {
        if (this._ok) {
            return handlers.ok(this._value!);
        }
        return handlers.err(this._error!);
    }

    // Conversion to Option
    toOption(): T | undefined {
        return this._ok ? this._value : undefined;
    }
}

// Usage example
const result = Result.fromThrowable(() => JSON.parse('{"name": "test"}'))
    .map(data => data as { name: string })
    .andThen(data => {
        if (!data.name) return Result.err(new Error("Name required"));
        return Result.ok(data);
    })
    .mapErr(e => `Validation failed: ${e.message}`);

result.match({
    ok: data => console.log(`User: ${data.name}`),
    err: msg => console.error(msg),
});
```

### 4.3 neverthrow Library

```typescript
// neverthrow: A popular Result type library for TypeScript
import { ok, err, Result, ResultAsync } from 'neverthrow';

// Basic usage
function divide(a: number, b: number): Result<number, string> {
    if (b === 0) return err("Division by zero");
    return ok(a / b);
}

// Chaining
function calculateAverage(numbers: number[]): Result<number, string> {
    if (numbers.length === 0) return err("Empty array");

    const sum = numbers.reduce((a, b) => a + b, 0);
    return divide(sum, numbers.length);
}

// map, mapErr, andThen
const result = calculateAverage([10, 20, 30])
    .map(avg => avg.toFixed(2))
    .mapErr(e => `Calculation error: ${e}`);

// ResultAsync: Async version
function fetchUser(id: string): ResultAsync<User, ApiError> {
    return ResultAsync.fromPromise(
        fetch(`/api/users/${id}`).then(r => r.json()),
        (e) => new ApiError("Fetch failed", e as Error)
    );
}

function fetchUserOrders(userId: string): ResultAsync<Order[], ApiError> {
    return fetchUser(userId)
        .andThen(user => {
            return ResultAsync.fromPromise(
                fetch(`/api/orders?userId=${user.id}`).then(r => r.json()),
                (e) => new ApiError("Fetch orders failed", e as Error)
            );
        });
}

// combine: Merge multiple Results
import { Result as NResult } from 'neverthrow';

function validateForm(data: FormData): Result<ValidatedForm, ValidationError[]> {
    const nameResult = validateName(data.name);
    const emailResult = validateEmail(data.email);
    const ageResult = validateAge(data.age);

    return NResult.combine([nameResult, emailResult, ageResult])
        .map(([name, email, age]) => ({ name, email, age }));
    // Ok if all succeed, first Err if any fail
}

// safeTry: Generator-based ? operator-like syntax
import { safeTry } from 'neverthrow';

function processOrder(orderId: string): Result<Receipt, AppError> {
    return safeTry(function* () {
        const order = yield* getOrder(orderId).safeUnwrap();
        const user = yield* getUser(order.userId).safeUnwrap();
        const payment = yield* chargePayment(user, order.total).safeUnwrap();
        return ok({ orderId, paymentId: payment.id, amount: order.total });
    });
}
```

### 4.4 ts-results Library

```typescript
// ts-results: Another popular library
import { Ok, Err, Result } from 'ts-results';

function parsePort(input: string): Result<number, string> {
    const port = parseInt(input, 10);
    if (isNaN(port)) return Err(`Invalid port number: ${input}`);
    if (port < 1 || port > 65535) return Err(`Port out of range: ${port}`);
    return Ok(port);
}

// Access Ok/Err values via the val property
const result = parsePort("8080");
if (result.ok) {
    console.log(`Port: ${result.val}`);  // 8080
} else {
    console.error(`Error: ${result.val}`);  // Error message
}

// expect: Returns the value if Ok, throws with a message if Err
const port = parsePort("8080").expect("Port must be valid");

// map and andThen
const configResult = parsePort("8080")
    .map(port => ({ port, host: "localhost" }))
    .andThen(config => {
        if (config.host === "") return Err("Host required");
        return Ok(config);
    });
```

---

## 5. Relationship with Option/Maybe Types

### 5.1 What Is the Option Type

```
Option<T> = Some(T) | None

Result<T, E> = Ok(T) | Err(E)

Difference:
  Option: Whether a value "exists or not"
  Result: Whether a value "exists, or why it doesn't"

  Option can be thought of as "a Result without error information":
  Option<T> ≒ Result<T, ()>  // No error information
```

```rust
// Rust: Interconversion between Option and Result
fn find_user(id: &str) -> Option<User> {
    users.get(id).cloned()
}

fn get_user(id: &str) -> Result<User, AppError> {
    find_user(id).ok_or_else(|| AppError::UserNotFound {
        user_id: id.to_string(),
    })
}

// Option methods
let opt: Option<i32> = Some(42);

opt.map(|v| v * 2);           // Some(84)
opt.and_then(|v| if v > 0 { Some(v) } else { None });
opt.unwrap_or(0);              // 42
opt.ok_or("value is none")?;  // Option → Result
```

```typescript
// TypeScript Option type
type Option<T> = T | null | undefined;

// When to use Result vs Option
function findUser(id: string): Option<User> {
    // "Not found" is a normal case → Option
    return users.get(id) ?? null;
}

function getUser(id: string): Result<User, NotFoundError> {
    // "Not found" is an error case → Result
    const user = users.get(id);
    if (!user) return err(new NotFoundError("User", id));
    return ok(user);
}

// Guidelines for choosing:
// Use Option when:
//   → The absence of a value is a normal situation
//   → Examples: Map.get(), Array.find(), cache.get()
//
// Use Result when:
//   → The reason for failure needs to be communicated
//   → Examples: API calls, validation, file reads
```

### 5.2 Haskell's Either and Maybe

```haskell
-- Haskell: Either and Maybe are the originals of Result and Option

-- Maybe a = Nothing | Just a
findUser :: String -> Maybe User
findUser userId = lookup userId userMap

-- Either e a = Left e | Right a (Left is error, Right is success)
getUser :: String -> Either AppError User
getUser userId = case findUser userId of
    Nothing -> Left (UserNotFound userId)
    Just user -> Right user

-- Chaining with do notation (Monad)
processOrder :: String -> Either AppError Receipt
processOrder orderId = do
    order <- getOrder orderId           -- Returns immediately on Err
    user  <- getUser (orderUserId order) -- Returns immediately on Err
    payment <- chargePayment user order  -- Returns immediately on Err
    return Receipt { receiptOrder = order, receiptPayment = payment }

-- This is equivalent to:
processOrder' :: String -> Either AppError Receipt
processOrder' orderId =
    getOrder orderId >>= \order ->
    getUser (orderUserId order) >>= \user ->
    chargePayment user order >>= \payment ->
    Right Receipt { receiptOrder = order, receiptPayment = payment }
```

---

## 6. Choosing Between Result Type and Exceptions

### 6.1 Scenario-Based Selection Criteria

```
When to Use the Result Type:
  ✓ Expected errors (validation, file not found)
  ✓ Public interfaces of libraries/APIs
  ✓ When type safety is critical
  ✓ When the set of error types is limited
  ✓ Performance-critical code
  ✓ Functional-style code
  ✓ When error composition is needed

When to Use Exceptions:
  ✓ Unexpected errors (programming mistakes)
  ✓ Unrecoverable errors (OutOfMemory)
  ✓ When the framework expects exceptions
  ✓ Error propagation from deep call stacks
  ✓ Errors in constructors or property access
  ✓ Boundaries with external libraries

Combined Approach (Recommended):
  → Domain logic: Result type (expected errors)
  → Infrastructure layer: Exceptions (network, DB failures)
  → Boundary (Controller): Convert exceptions to Result
```

### 6.2 Layer-Specific Usage

```typescript
// Layer-specific usage example

// ========== Infrastructure Layer: Throws exceptions ==========
class UserRepository {
    async findById(id: string): Promise<User | null> {
        // DB errors propagate as exceptions
        const row = await db.query("SELECT * FROM users WHERE id = $1", [id]);
        return row ? mapToUser(row) : null;
    }
}

// ========== Domain Layer: Uses Result type ==========
class UserService {
    constructor(private repo: UserRepository) {}

    async getUser(id: string): Promise<Result<User, UserError>> {
        try {
            const user = await this.repo.findById(id);
            if (!user) return err(new UserNotFoundError(id));
            return ok(user);
        } catch (error) {
            // Convert infrastructure exceptions to Result
            return err(new UserServiceError("Database error", { cause: error }));
        }
    }

    async createUser(data: CreateUserDto): Promise<Result<User, UserError>> {
        // Validation
        const validation = validateCreateUser(data);
        if (!validation.ok) return validation;

        // Duplicate check
        const existing = await this.repo.findByEmail(data.email);
        if (existing) return err(new EmailAlreadyExistsError(data.email));

        try {
            const user = await this.repo.create(data);
            return ok(user);
        } catch (error) {
            return err(new UserServiceError("Failed to create user", { cause: error }));
        }
    }
}

// ========== Presentation Layer: Converts Result to HTTP response ==========
class UserController {
    constructor(private service: UserService) {}

    async getUser(req: Request, res: Response): Promise<void> {
        const result = await this.service.getUser(req.params.id);

        result.match({
            ok: user => res.json(user),
            err: error => {
                if (error instanceof UserNotFoundError) {
                    res.status(404).json({ error: error.message });
                } else {
                    res.status(500).json({ error: "Internal server error" });
                }
            }
        });
    }
}
```

### 6.3 Migration Strategy in Practice

```typescript
// Strategy for gradually migrating from exception-based code to Result type

// Step 1: Prepare Result type utilities
function tryCatch<T>(fn: () => T): Result<T, Error> {
    try {
        return ok(fn());
    } catch (e) {
        return err(e instanceof Error ? e : new Error(String(e)));
    }
}

async function tryCatchAsync<T>(fn: () => Promise<T>): Promise<Result<T, Error>> {
    try {
        return ok(await fn());
    } catch (e) {
        return err(e instanceof Error ? e : new Error(String(e)));
    }
}

// Step 2: Start using Result type in new code
// Convert at the boundary with existing code

// Exception → Result conversion
async function getUserSafe(id: string): Promise<Result<User, AppError>> {
    return tryCatchAsync(async () => {
        // Call existing exception-based functions
        return await legacyGetUser(id);
    }).then(result =>
        result.mapErr(e => new AppError("USER_FETCH_FAILED", e.message))
    );
}

// Result → Exception conversion (when the framework expects exceptions)
function unwrapOrThrow<T, E extends Error>(result: Result<T, E>): T {
    if (result.ok) return result.value;
    throw result.error;
}

// Step 3: Migrate from critical domain logic first
// Order: Validation → Business rules → Service layer
```

---

## 7. Advanced Patterns

### 7.1 Railway Oriented Programming

```
Railway Oriented Programming:

  Represents the success and failure paths as "two rails."
  Each function can switch from the Success rail to the Error rail.

  Success ──────→ validate ──→ transform ──→ save ──→ Success
                      │              │           │
  Error   ◁──────────┘    ◁─────────┘    ◁──────┘     Error

  Result type's andThen (flatMap) is exactly this pattern:
  → Only executes the next function on Success
  → On Error, flows along the Error rail as-is
```

```typescript
// Railway Oriented Programming implementation example
type Result<T, E> = { ok: true; value: T } | { ok: false; error: E };

// Pipeline function
function pipe<T, E>(
    initial: Result<T, E>,
    ...fns: Array<(value: any) => Result<any, E>>
): Result<any, E> {
    let result: Result<any, E> = initial;
    for (const fn of fns) {
        if (!result.ok) return result;  // Pass through the Error rail as-is
        result = fn(result.value);
    }
    return result;
}

// Usage example: User registration pipeline
function registerUser(input: unknown): Result<User, AppError> {
    return pipe(
        ok(input),
        validateInput,    // Input validation
        normalizeEmail,   // Email normalization
        checkDuplicate,   // Duplicate check
        hashPassword,     // Password hashing
        saveToDatabase,   // Save to DB
        sendWelcomeEmail, // Send email
    );
}

// Each function can be tested independently
function validateInput(input: unknown): Result<RegisterDto, AppError> {
    if (!input || typeof input !== "object") {
        return err({ code: "INVALID_INPUT", message: "Invalid input" });
    }
    // ... validation logic
    return ok(input as RegisterDto);
}

function normalizeEmail(dto: RegisterDto): Result<RegisterDto, AppError> {
    return ok({ ...dto, email: dto.email.toLowerCase().trim() });
}
```

### 7.2 Parallel Processing with Result

```typescript
// Utilities for processing multiple Results in parallel

// all: Ok if all succeed, first Err if any fail
function all<T, E>(results: Result<T, E>[]): Result<T[], E> {
    const values: T[] = [];
    for (const result of results) {
        if (!result.ok) return result;
        values.push(result.value);
    }
    return ok(values);
}

// allSettled: Collect all results
function allSettled<T, E>(
    results: Result<T, E>[]
): { successes: T[]; failures: E[] } {
    const successes: T[] = [];
    const failures: E[] = [];
    for (const result of results) {
        if (result.ok) successes.push(result.value);
        else failures.push(result.error);
    }
    return { successes, failures };
}

// Async version
async function allAsync<T, E>(
    promises: Promise<Result<T, E>>[]
): Promise<Result<T[], E>> {
    const results = await Promise.all(promises);
    return all(results);
}

// Usage example
async function validateBulkUsers(
    users: CreateUserDto[]
): Promise<Result<ValidatedUser[], ValidationError[]>> {
    const results = users.map(user => validateUser(user));
    const { successes, failures } = allSettled(results);

    if (failures.length > 0) {
        return err(failures);
    }
    return ok(successes);
}
```

### 7.3 Extensions to Result and Either

```typescript
// Either type: A generalization of Result (no semantics assigned to Left/Right)
type Either<L, R> = { tag: "left"; value: L } | { tag: "right"; value: R };

function left<L>(value: L): Either<L, never> {
    return { tag: "left", value };
}

function right<R>(value: R): Either<never, R> {
    return { tag: "right", value };
}

// bimap: Transform both cases
function bimap<L, R, L2, R2>(
    either: Either<L, R>,
    leftFn: (l: L) => L2,
    rightFn: (r: R) => R2,
): Either<L2, R2> {
    if (either.tag === "left") return left(leftFn(either.value));
    return right(rightFn(either.value));
}

// Validation type: A Result that accumulates errors
type Validation<T, E> = { ok: true; value: T } | { ok: false; errors: E[] };

function validateAll<T, E>(
    validations: Validation<T, E>[]
): Validation<T[], E> {
    const values: T[] = [];
    const errors: E[] = [];

    for (const v of validations) {
        if (v.ok) {
            values.push(v.value);
        } else {
            errors.push(...v.errors);
        }
    }

    if (errors.length > 0) {
        return { ok: false, errors };
    }
    return { ok: true, value: values };
}

// Usage example: Form validation (return all errors at once)
function validateRegistrationForm(data: FormData): Validation<ValidForm, FieldError> {
    return validateAll([
        validateName(data.name),
        validateEmail(data.email),
        validatePassword(data.password),
        validateAge(data.age),
    ]).map(([name, email, password, age]) => ({
        name, email, password, age,
    }));
}
// Result.andThen short-circuits on the first error,
// while Validation.validateAll collects all errors
```

---

## 8. Result in Scala and Functional Languages

### 8.1 Scala's Either and Try

```scala
// Scala: Either[L, R]
def divide(a: Double, b: Double): Either[String, Double] = {
  if (b == 0) Left("Division by zero")
  else Right(a / b)
}

// Chaining with for comprehensions (equivalent to Haskell's do notation)
def calculate(a: Double, b: Double, c: Double): Either[String, Double] = {
  for {
    x <- divide(a, b)     // Returns immediately on Err
    y <- divide(x, c)     // Returns immediately on Err
    z <- if (y > 0) Right(y) else Left("Result must be positive")
  } yield z * 100
}

// Try[T]: Automatically catches exceptions
import scala.util.{Try, Success, Failure}

val result: Try[Int] = Try("42".toInt)
// Success(42)

val result: Try[Int] = Try("abc".toInt)
// Failure(java.lang.NumberFormatException)

val processed = Try("42".toInt)
  .map(_ * 2)
  .flatMap(n => if (n > 0) Success(n) else Failure(new Exception("negative")))
  .recover { case _: NumberFormatException => 0 }
  .getOrElse(-1)
```

### 8.2 F#'s Result

```fsharp
// F#: Result<'T, 'Error> is a standard library type
let divide a b : Result<float, string> =
    if b = 0.0 then Error "Division by zero"
    else Ok (a / b)

// Chaining with the pipe operator
let processOrder orderId =
    getOrder orderId
    |> Result.bind validateOrder
    |> Result.bind calculateTotal
    |> Result.bind processPayment
    |> Result.map createReceipt

// Computation Expression (equivalent to do notation)
type ResultBuilder() =
    member _.Bind(x, f) = Result.bind f x
    member _.Return(x) = Ok x

let result = ResultBuilder()

let processOrder orderId = result {
    let! order = getOrder orderId
    let! validated = validateOrder order
    let! total = calculateTotal validated
    let! payment = processPayment total
    return createReceipt payment
}
```

---

## 9. Testing Strategies

### 9.1 Testing with the Result Type

```typescript
// Testing code that uses the Result type
describe("UserService.createUser", () => {
    it("returns Ok(User) with valid data", async () => {
        const result = await userService.createUser({
            name: "Test User",
            email: "test@example.com",
            password: "SecurePass123!",
        });

        expect(result.ok).toBe(true);
        if (result.ok) {
            expect(result.value.name).toBe("Test User");
            expect(result.value.email).toBe("test@example.com");
            expect(result.value.id).toBeDefined();
        }
    });

    it("returns Err(ValidationError) with invalid email", async () => {
        const result = await userService.createUser({
            name: "Test User",
            email: "invalid-email",
            password: "SecurePass123!",
        });

        expect(result.ok).toBe(false);
        if (!result.ok) {
            expect(result.error).toBeInstanceOf(ValidationError);
            expect(result.error.code).toBe("VALIDATION_ERROR");
        }
    });

    it("returns Err(ConflictError) with duplicate email", async () => {
        // Create an existing user
        await userService.createUser({
            name: "Existing",
            email: "existing@example.com",
            password: "Pass123!",
        });

        const result = await userService.createUser({
            name: "New User",
            email: "existing@example.com",
            password: "Pass123!",
        });

        expect(result.ok).toBe(false);
        if (!result.ok) {
            expect(result.error.code).toBe("EMAIL_ALREADY_EXISTS");
        }
    });
});

// Helper functions for concise tests
function expectOk<T, E>(result: Result<T, E>): T {
    expect(result.ok).toBe(true);
    if (!result.ok) throw new Error(`Expected Ok, got Err: ${result.error}`);
    return result.value;
}

function expectErr<T, E>(result: Result<T, E>): E {
    expect(result.ok).toBe(false);
    if (result.ok) throw new Error(`Expected Err, got Ok: ${result.value}`);
    return result.error;
}

// Usage examples
it("can create a user successfully", async () => {
    const user = expectOk(await userService.createUser(validData));
    expect(user.name).toBe("Test User");
});

it("returns a validation error", async () => {
    const error = expectErr(await userService.createUser(invalidData));
    expect(error.code).toBe("VALIDATION_ERROR");
});
```

### 9.2 Property-Based Testing

```typescript
// Property-based testing with fast-check
import fc from 'fast-check';

describe("Result invariants", () => {
    it("identity law for map: result.map(x => x) === result", () => {
        fc.assert(
            fc.property(fc.integer(), (n) => {
                const result = ok(n);
                const mapped = result.map(x => x);
                expect(mapped).toEqual(result);
            })
        );
    });

    it("associativity law for andThen", () => {
        fc.assert(
            fc.property(fc.integer(), (n) => {
                const f = (x: number) => ok(x * 2);
                const g = (x: number) => ok(x + 1);

                const left = ok(n).andThen(f).andThen(g);
                const right = ok(n).andThen(x => f(x).andThen(g));

                expect(left).toEqual(right);
            })
        );
    });

    it("parsePort always returns a value in 1-65535 or Err", () => {
        fc.assert(
            fc.property(fc.string(), (input) => {
                const result = parsePort(input);
                if (result.ok) {
                    expect(result.value).toBeGreaterThanOrEqual(1);
                    expect(result.value).toBeLessThanOrEqual(65535);
                }
                // If Err, validation is working correctly
            })
        );
    });
});
```

---

## 10. Adoption Patterns in Practice

### 10.1 Gradual Adoption Roadmap

```
Gradual Adoption of the Result Type:

  Phase 1: Prepare Utilities
  → Define the Result type (or select a library)
  → ok(), err() helper functions
  → tryCatch, tryCatchAsync utilities

  Phase 2: Start with the Validation Layer
  → Form validation
  → API request validation
  → Configuration value validation
  → ← Highest impact, lowest risk

  Phase 3: Expand to the Service Layer
  → Convert domain logic return values to Result
  → Clarify boundaries with exceptions
  → Keep the repository layer with exceptions

  Phase 4: Integrate with API Responses
  → Convert Result to HTTP responses in Controllers
  → Unify error response formats
  → Align with OpenAPI schemas

  Phase 5: Team-Wide Adoption
  → Update coding conventions
  → Establish code review guidelines
  → Standardize test patterns
```

### 10.2 Building Team Consensus

```
Discussion Points When Adopting the Result Type:

  1. Library Selection
     → neverthrow: Most popular, includes ResultAsync
     → ts-results: Lightweight, Rust-style
     → Custom implementation: Flexible but has maintenance cost
     → Built-in union types: No library needed but fewer features

  2. Boundary Rules with Exceptions
     → Which layer starts using Result
     → Where to convert exceptions to Result
     → Integration points with frameworks

  3. Error Type Design
     → string vs custom error classes
     → Error code taxonomy
     → Level of error detail

  4. Coexistence with Existing Code
     → Gradual migration vs all-at-once migration
     → Adapter layer design
     → Test migration strategy
```


---

## Practical Exercises

### Exercise 1: Basic Implementation

Implement code that meets the following requirements.

**Requirements:**
- Validate input data
- Implement proper error handling
- Write test code as well

```python
# Exercise 1: Basic implementation template
class Exercise1:
    """Exercise for basic implementation patterns"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """Validate the input value"""
        if value is None:
            raise ValueError("Input value is None")
        return True

    def process(self, value):
        """Main data processing logic"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """Retrieve processing results"""
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
        assert False, "An exception should have been raised"
    except ValueError:
        pass

    print("All tests passed!")

test_exercise1()
```

### Exercise 2: Applied Patterns

Extend the basic implementation by adding the following features.

```python
# Exercise 2: Applied patterns
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """Exercise for applied patterns"""

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
    assert ex.add("d", 4) == False  # Size limit
    assert ex.find("b")['value'] == 2
    assert ex.remove("b") == True
    assert ex.find("b") is None
    stats = ex.stats()
    assert stats['total_items'] == 2
    print("All applied tests passed!")

test_advanced()
```

### Exercise 3: Performance Optimization

Improve the performance of the following code.

```python
# Exercise 3: Performance optimization
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
    print(f"Speedup factor: {slow_time/fast_time:.0f}x")

benchmark()
```

**Key Points:**
- Be aware of algorithmic time complexity
- Choose appropriate data structures
- Measure the effect with benchmarks
---


## FAQ

### Q1: What is the most important point to keep in mind when learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory, but by actually writing code and verifying its behavior.

### Q2: What are common mistakes that beginners make?

Skipping the fundamentals and jumping into advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this applied in professional settings?

The knowledge from this topic is frequently used in day-to-day development work. It is especially important during code reviews and architecture design.

---

## Summary

| Language | Error Approach | Characteristics |
|----------|---------------|-----------------|
| Rust | Result<T, E> + ? | Most refined Result type |
| Go | (value, error) | Simple but verbose |
| TypeScript | Union types / neverthrow | Expressible through types |
| Java | Exceptions (+ Optional) | Checked exceptions |
| Python | Exceptions | Supplemented by type hints |
| Haskell | Either a b | Original, chaining via Monads |
| Scala | Either / Try | Concise with for comprehensions |
| F# | Result<'T, 'E> | Computation Expressions |
| Kotlin | runCatching / Result | Java compatible |
| Swift | throws + Result | Both approaches available |

---

## Recommended Next Reading

---

## References
1. The Rust Programming Language. "Error Handling."
2. Go Blog. "Error handling and Go." 2011.
3. Wlaschin, S. "Railway Oriented Programming." F# for Fun and Profit.
4. neverthrow. "Type-Safe Error Handling in TypeScript." GitHub.
5. Bloch, J. "Effective Java." Item 71: Avoid unnecessary use of checked exceptions.
6. Syme, D. et al. "The F# Component Design Guidelines."
7. Kotlin Documentation. "Exceptions."
8. Apple Developer Documentation. "Error Handling in Swift."
