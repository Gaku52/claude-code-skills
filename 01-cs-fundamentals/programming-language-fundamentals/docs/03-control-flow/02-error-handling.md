# Error Handling

> How to "represent, propagate, and handle" errors is a critical design decision in programming languages. Understand the three major strategies: exceptions, Result types, and error codes.

## What You Will Learn in This Chapter

- [ ] Understand the differences between exception-based, Result-type, and error-code approaches
- [ ] Grasp the error handling philosophy of each language
- [ ] Select the appropriate error handling pattern
- [ ] Design custom error types
- [ ] Master best practices for error propagation and transformation
- [ ] Design error handling strategies for real-world applications


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding of the content in [Pattern Matching](./01-pattern-matching.md)

---

## 1. The Three Major Error Handling Strategies

### 1.1 Overview

```
Three strategies:

  1. Exceptions
     → Separate normal and exceptional code paths
     → Implicit propagation (unwinds the stack to the caller)
     → Representative languages: Python, Java, C#, JavaScript, Ruby

  2. Result Type / Either Type
     → Represent errors as "values" through types
     → Explicit propagation (? operator, map/and_then)
     → Representative languages: Rust, Haskell, Elm, OCaml, F#

  3. Error Codes / Multiple Return Values
     → Indicate errors through function return values
     → Explicit propagation (if err != nil)
     → Representative languages: C, Go

Trade-offs of each strategy:
  Exceptions:  Easy to write     ⟷ Errors are hard to see
  Result:      Type-safe         ⟷ More boilerplate
  Error codes: Simple            ⟷ Risk of forgetting to check
```

---

## 2. Exceptions

### 2.1 Python's Error Handling

```python
# Python: try-except

# ========================================
# Basic exception handling
# ========================================
try:
    result = int("not a number")
    file = open("missing.txt")
except ValueError as e:
    print(f"Invalid value: {e}")
except FileNotFoundError as e:
    print(f"File not found: {e}")
except Exception as e:
    print(f"Unexpected: {e}")
else:
    print("Success")  # Executes when no exception occurred
finally:
    print("Always executed")  # Always runs (cleanup)

# ========================================
# Exception hierarchy
# ========================================
# BaseException
#   ├── SystemExit
#   ├── KeyboardInterrupt
#   ├── GeneratorExit
#   └── Exception
#       ├── ValueError
#       ├── TypeError
#       ├── KeyError
#       ├── IndexError
#       ├── AttributeError
#       ├── IOError
#       │   └── FileNotFoundError
#       ├── RuntimeError
#       └── ...

# ========================================
# Custom exception design
# ========================================
class AppError(Exception):
    """Base exception for the application"""
    def __init__(self, message: str, code: str = "UNKNOWN"):
        super().__init__(message)
        self.code = code

class NotFoundError(AppError):
    """Resource not found"""
    def __init__(self, resource: str, resource_id: str):
        super().__init__(
            f"{resource} with id '{resource_id}' not found",
            code="NOT_FOUND"
        )
        self.resource = resource
        self.resource_id = resource_id

class ValidationError(AppError):
    """Validation error"""
    def __init__(self, errors: dict[str, list[str]]):
        messages = []
        for field, field_errors in errors.items():
            for error in field_errors:
                messages.append(f"{field}: {error}")
        super().__init__(
            f"Validation failed: {'; '.join(messages)}",
            code="VALIDATION_ERROR"
        )
        self.errors = errors

class AuthenticationError(AppError):
    """Authentication error"""
    def __init__(self, reason: str = "Invalid credentials"):
        super().__init__(reason, code="AUTHENTICATION_ERROR")

class AuthorizationError(AppError):
    """Authorization error"""
    def __init__(self, action: str, resource: str):
        super().__init__(
            f"Not authorized to {action} {resource}",
            code="AUTHORIZATION_ERROR"
        )
        self.action = action
        self.resource = resource

# ========================================
# Exception usage example
# ========================================
class UserService:
    def __init__(self, db, auth_service):
        self.db = db
        self.auth_service = auth_service

    def get_user(self, user_id: str) -> User:
        user = self.db.find_user(user_id)
        if user is None:
            raise NotFoundError("User", user_id)
        return user

    def update_user(self, user_id: str, data: dict, requester: User) -> User:
        if requester.id != user_id and not requester.is_admin:
            raise AuthorizationError("update", f"user/{user_id}")

        errors = self._validate_user_data(data)
        if errors:
            raise ValidationError(errors)

        user = self.get_user(user_id)  # NotFoundError propagates
        user.update(data)
        return self.db.save_user(user)

    def _validate_user_data(self, data: dict) -> dict[str, list[str]]:
        errors = {}
        if "name" in data and len(data["name"]) < 2:
            errors.setdefault("name", []).append("Must be at least 2 characters")
        if "email" in data and "@" not in data["email"]:
            errors.setdefault("email", []).append("Invalid email format")
        if "age" in data and (data["age"] < 0 or data["age"] > 150):
            errors.setdefault("age", []).append("Must be between 0 and 150")
        return errors

# ========================================
# Context managers (automatic cleanup)
# ========================================
# with statement (automatic resource release)
with open("file.txt") as f:
    content = f.read()
# File is automatically closed (even if an exception occurs)

# Custom context manager
from contextlib import contextmanager

@contextmanager
def database_transaction(db):
    """Transaction management"""
    tx = db.begin_transaction()
    try:
        yield tx
        tx.commit()
    except Exception:
        tx.rollback()
        raise  # Re-raise the exception

# Usage example
with database_transaction(db) as tx:
    tx.execute("INSERT INTO users ...")
    tx.execute("INSERT INTO profiles ...")
    # Automatically rolls back if an exception occurs

# ========================================
# EAFP vs LBYL
# ========================================
# LBYL (Look Before You Leap) — check first
if key in dictionary:
    value = dictionary[key]
else:
    value = default

# EAFP (Easier to Ask Forgiveness than Permission) — Pythonic
try:
    value = dictionary[key]
except KeyError:
    value = default

# Best: use dict.get()
value = dictionary.get(key, default)

# ========================================
# Exception chaining (Python 3)
# ========================================
class DatabaseError(AppError):
    pass

def get_user_from_db(user_id):
    try:
        return db.query(f"SELECT * FROM users WHERE id = {user_id}")
    except sqlite3.OperationalError as e:
        # raise ... from e preserves the original exception
        raise DatabaseError(f"Failed to query user {user_id}") from e
    # The original exception is accessible via __cause__

# Implicit exception chaining
try:
    try:
        1 / 0
    except ZeroDivisionError:
        raise ValueError("Invalid computation")
    # The implicit exception chain is accessible via __context__
except ValueError as e:
    print(f"Error: {e}")
    print(f"Caused by: {e.__context__}")

# ========================================
# Suppressing exceptions
# ========================================
from contextlib import suppress

# Safely ignore an exception
with suppress(FileNotFoundError):
    os.remove("tempfile.txt")
# No error even if the file doesn't exist

# Equivalent code
try:
    os.remove("tempfile.txt")
except FileNotFoundError:
    pass
```

### 2.2 Java's Error Handling

```java
// Java: checked exceptions vs unchecked exceptions

// ========================================
// Checked Exceptions
// ========================================
// Exceptions that the compiler forces you to handle
// → IOException, SQLException, ClassNotFoundException, etc.
public String readFile(String path) throws IOException {
    return Files.readString(Path.of(path));
}

// The caller must either handle or propagate with throws
public void processFile(String path) {
    try {
        String content = readFile(path);
        System.out.println(content);
    } catch (IOException e) {
        logger.error("Failed to read file: " + path, e);
    }
}

// ========================================
// Unchecked Exceptions
// ========================================
// Subclasses of RuntimeException
// → NullPointerException, IllegalArgumentException,
//    IndexOutOfBoundsException, etc.
public int divide(int a, int b) {
    if (b == 0) throw new IllegalArgumentException("Division by zero");
    return a / b;
}

// ========================================
// try-with-resources (automatic cleanup)
// ========================================
// Automatically closes resources that implement the AutoCloseable interface
try (var reader = new BufferedReader(new FileReader("file.txt"));
     var writer = new BufferedWriter(new FileWriter("output.txt"))) {
    String line;
    while ((line = reader.readLine()) != null) {
        writer.write(line);
        writer.newLine();
    }
} catch (IOException e) {
    logger.error("I/O error", e);
}

// ========================================
// Custom exceptions (Java)
// ========================================
public class AppException extends Exception {
    private final String errorCode;
    private final int statusCode;

    public AppException(String message, String errorCode, int statusCode) {
        super(message);
        this.errorCode = errorCode;
        this.statusCode = statusCode;
    }

    public AppException(String message, String errorCode, int statusCode, Throwable cause) {
        super(message, cause);
        this.errorCode = errorCode;
        this.statusCode = statusCode;
    }

    public String getErrorCode() { return errorCode; }
    public int getStatusCode() { return statusCode; }
}

public class NotFoundException extends AppException {
    public NotFoundException(String resource, String id) {
        super(resource + " with id '" + id + "' not found", "NOT_FOUND", 404);
    }
}

public class ValidationException extends AppException {
    private final Map<String, List<String>> errors;

    public ValidationException(Map<String, List<String>> errors) {
        super("Validation failed", "VALIDATION_ERROR", 400);
        this.errors = Collections.unmodifiableMap(errors);
    }

    public Map<String, List<String>> getErrors() { return errors; }
}

// ========================================
// Multi-catch (Java 7+)
// ========================================
try {
    // ...
} catch (IOException | SQLException e) {
    logger.error("I/O or Database error", e);
}

// ========================================
// Problems with checked exceptions and workarounds
// ========================================
// Problem: checked exceptions cannot be used in lambda expressions
// Bad — compile error
// list.stream().map(path -> Files.readString(path));

// Good — use a wrapper
@FunctionalInterface
interface ThrowingFunction<T, R> {
    R apply(T t) throws Exception;
}

static <T, R> Function<T, R> unchecked(ThrowingFunction<T, R> f) {
    return t -> {
        try {
            return f.apply(t);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    };
}

// Usage example
list.stream()
    .map(unchecked(path -> Files.readString(Path.of(path))))
    .collect(Collectors.toList());
```

### 2.3 JavaScript / TypeScript Error Handling

```javascript
// JavaScript: try-catch-finally

// ========================================
// Basic exception handling
// ========================================
try {
    const data = JSON.parse(invalidJson);
    processData(data);
} catch (error) {
    if (error instanceof SyntaxError) {
        console.error("Invalid JSON:", error.message);
    } else if (error instanceof TypeError) {
        console.error("Type error:", error.message);
    } else {
        console.error("Unknown error:", error);
    }
} finally {
    cleanup();
}

// ========================================
// Promise error handling
// ========================================
// .catch() method
fetch("/api/data")
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        return response.json();
    })
    .then(data => processData(data))
    .catch(error => {
        console.error("Fetch failed:", error);
    });

// async/await
async function fetchData(url) {
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new HttpError(response.status, response.statusText);
        }
        return await response.json();
    } catch (error) {
        if (error instanceof HttpError) {
            console.error(`HTTP Error ${error.status}: ${error.message}`);
        } else if (error instanceof TypeError) {
            console.error("Network error:", error.message);
        } else {
            throw error; // Re-throw unexpected errors
        }
    }
}

// ========================================
// Promise.allSettled (get all results)
// ========================================
const results = await Promise.allSettled([
    fetch("/api/users"),
    fetch("/api/posts"),
    fetch("/api/comments"),
]);

for (const result of results) {
    if (result.status === "fulfilled") {
        console.log("Success:", result.value);
    } else {
        console.error("Failed:", result.reason);
    }
}

// ========================================
// AggregateError (error from Promise.any)
// ========================================
try {
    const first = await Promise.any([
        fetch("/api/primary"),
        fetch("/api/secondary"),
        fetch("/api/tertiary"),
    ]);
} catch (error) {
    if (error instanceof AggregateError) {
        console.error("All promises failed:");
        for (const e of error.errors) {
            console.error(" -", e.message);
        }
    }
}

// ========================================
// Global error handling
// ========================================
// Browser
window.addEventListener("error", (event) => {
    reportError({
        message: event.message,
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno,
        error: event.error,
    });
});

window.addEventListener("unhandledrejection", (event) => {
    reportError({
        message: "Unhandled promise rejection",
        reason: event.reason,
    });
    event.preventDefault();
});

// Node.js
process.on("uncaughtException", (error) => {
    logger.fatal("Uncaught exception:", error);
    process.exit(1);
});

process.on("unhandledRejection", (reason, promise) => {
    logger.error("Unhandled rejection:", reason);
});
```

```typescript
// TypeScript: custom error classes
class AppError extends Error {
    constructor(
        message: string,
        public readonly code: string,
        public readonly statusCode: number,
        public readonly details?: Record<string, unknown>,
    ) {
        super(message);
        this.name = "AppError";
        // Fix the prototype chain (caveat of TypeScript class inheritance)
        Object.setPrototypeOf(this, new.target.prototype);
    }

    toJSON() {
        return {
            name: this.name,
            code: this.code,
            message: this.message,
            statusCode: this.statusCode,
            details: this.details,
        };
    }
}

class NotFoundError extends AppError {
    constructor(resource: string, id?: string) {
        const msg = id
            ? `${resource} with id '${id}' not found`
            : `${resource} not found`;
        super(msg, "NOT_FOUND", 404, { resource, id });
        this.name = "NotFoundError";
    }
}

class ValidationError extends AppError {
    constructor(public readonly fields: Record<string, string[]>) {
        const messages = Object.entries(fields)
            .flatMap(([field, errors]) => errors.map(e => `${field}: ${e}`));
        super(
            `Validation failed: ${messages.join("; ")}`,
            "VALIDATION_ERROR",
            400,
            { fields },
        );
        this.name = "ValidationError";
    }
}

class ConflictError extends AppError {
    constructor(resource: string, conflictField: string) {
        super(
            `${resource} already exists with this ${conflictField}`,
            "CONFLICT",
            409,
            { resource, conflictField },
        );
        this.name = "ConflictError";
    }
}

// ========================================
// Express.js error handling middleware
// ========================================
function errorHandler(err: Error, req: Request, res: Response, next: NextFunction) {
    if (err instanceof AppError) {
        res.status(err.statusCode).json(err.toJSON());
    } else {
        logger.error("Unhandled error:", err);
        res.status(500).json({
            name: "InternalError",
            code: "INTERNAL_ERROR",
            message: "An internal error occurred",
            statusCode: 500,
        });
    }
}

// ========================================
// Type-safe error handling (TypeScript)
// ========================================
// Error type discriminator functions
function isAppError(error: unknown): error is AppError {
    return error instanceof AppError;
}

function isNotFoundError(error: unknown): error is NotFoundError {
    return error instanceof NotFoundError;
}

// Safe error handling
async function handleRequest(req: Request): Promise<Response> {
    try {
        const result = await processRequest(req);
        return new Response(JSON.stringify(result), { status: 200 });
    } catch (error) {
        if (isNotFoundError(error)) {
            return new Response(JSON.stringify({ error: error.message }), { status: 404 });
        }
        if (isAppError(error)) {
            return new Response(JSON.stringify(error.toJSON()), { status: error.statusCode });
        }
        // Unexpected error
        console.error("Unexpected error:", error);
        return new Response(JSON.stringify({ error: "Internal server error" }), { status: 500 });
    }
}
```

### 2.4 Problems with Exceptions

```
Problem 1: Invisible control flow
  → Cannot tell which exceptions a function throws just by looking at its signature
  → Java's checked exceptions are a solution but can become verbose
  → JavaScript/Python are entirely implicit

Problem 2: Performance cost
  → Generating stack traces is expensive
  → Using exceptions in the normal flow degrades performance
  → Exceptions are strictly for "exceptional" situations

Problem 3: Swallowing exceptions
  → Catching and doing nothing → hiding bugs
  → Empty catch blocks are the worst anti-pattern

Problem 4: Difficulty guaranteeing exception safety
  → Risk of resource leaks
  → Partially updated state
  → Transaction integrity
```

---

## 3. Result Type (Value-Based Error Handling)

### 3.1 Rust's Result<T, E>

```rust
// Rust: Result<T, E> — representing success or failure through types

// ========================================
// Basic usage
// ========================================
fn parse_number(s: &str) -> Result<i32, ParseIntError> {
    s.parse::<i32>()
}

// Handle with pattern matching
match parse_number("42") {
    Ok(n) => println!("Parsed: {}", n),
    Err(e) => println!("Error: {}", e),
}

// ========================================
// ? operator (shorthand for error propagation)
// ========================================
fn read_config() -> Result<Config, Box<dyn Error>> {
    let content = fs::read_to_string("config.toml")?;  // Returns immediately on error
    let config: Config = toml::from_str(&content)?;
    Ok(config)
}

// The ? operator is syntactic sugar for:
fn read_config_expanded() -> Result<Config, Box<dyn Error>> {
    let content = match fs::read_to_string("config.toml") {
        Ok(c) => c,
        Err(e) => return Err(e.into()),  // Converts via the From trait
    };
    let config = match toml::from_str(&content) {
        Ok(c) => c,
        Err(e) => return Err(e.into()),
    };
    Ok(config)
}

// ========================================
// Result method chaining
// ========================================
// map — transform the contents of Ok
let doubled: Result<i32, _> = parse_number("42").map(|n| n * 2);

// map_err — transform the contents of Err
let result: Result<i32, AppError> = parse_number("abc")
    .map_err(|e| AppError::Validation(e.to_string()));

// and_then — chain functions that return Result
fn validate_positive(n: i32) -> Result<i32, String> {
    if n > 0 { Ok(n) } else { Err("Must be positive".to_string()) }
}

let result = parse_number("42")
    .map_err(|e| e.to_string())
    .and_then(validate_positive)
    .map(|n| n * 2);

// unwrap_or — default value
let n = parse_number("abc").unwrap_or(0);

// unwrap_or_else — lazily evaluated default value
let n = parse_number("abc").unwrap_or_else(|e| {
    eprintln!("Parse error: {}", e);
    0
});

// ok — convert Result to Option (discards Err)
let opt: Option<i32> = parse_number("42").ok();

// ========================================
// Combining multiple Results
// ========================================
// Sequential processing (? operator)
fn process() -> Result<Output, AppError> {
    let a = step1()?;
    let b = step2(a)?;
    let c = step3(b)?;
    Ok(c)
}

// collect converts Vec<Result<T, E>> → Result<Vec<T>, E>
fn parse_all(inputs: &[&str]) -> Result<Vec<i32>, ParseIntError> {
    inputs.iter().map(|s| s.parse::<i32>()).collect()
}

// All succeed → Ok(vec![1, 2, 3])
// Any failure → the first Err

// ========================================
// Custom error types (thiserror crate)
// ========================================
use thiserror::Error;

#[derive(Error, Debug)]
enum AppError {
    #[error("User not found: {0}")]
    NotFound(String),

    #[error("Database error")]
    Database(#[from] sqlx::Error),

    #[error("Validation failed: {0}")]
    Validation(String),

    #[error("Authentication failed: {0}")]
    Auth(String),

    #[error("Rate limit exceeded: retry after {retry_after}s")]
    RateLimit { retry_after: u64 },

    #[error("Internal error: {0}")]
    Internal(#[from] anyhow::Error),
}

// Automatic From trait implementation (#[from] attribute)
// sqlx::Error → AppError::Database automatic conversion
// → Automatically converted by the ? operator

// ========================================
// anyhow crate (for prototypes and CLI tools)
// ========================================
use anyhow::{Context, Result, bail, ensure};

fn read_config(path: &str) -> Result<Config> {
    let content = fs::read_to_string(path)
        .context(format!("Failed to read config file: {}", path))?;

    let config: Config = toml::from_str(&content)
        .context("Failed to parse config file")?;

    ensure!(config.port > 0, "Port must be positive, got {}", config.port);

    if config.host.is_empty() {
        bail!("Host must not be empty");
    }

    Ok(config)
}

// ========================================
// Error propagation and adding context
// ========================================
fn get_user(id: u32) -> Result<User, AppError> {
    let user = db.find_user(id)
        .map_err(AppError::Database)?;

    match user {
        Some(u) => Ok(u),
        None => Err(AppError::NotFound(format!("user id={}", id))),
    }
}

fn get_user_profile(user_id: u32) -> Result<UserProfile, AppError> {
    let user = get_user(user_id)?;  // AppError automatically propagated
    let profile = db.find_profile(user.id)
        .map_err(AppError::Database)?;
    Ok(UserProfile { user, profile })
}

// ========================================
// Interconversion between Result and Option
// ========================================
// Option → Result
let value: Result<i32, &str> = some_option.ok_or("Value is None");
let value: Result<i32, AppError> = some_option
    .ok_or_else(|| AppError::NotFound("value".to_string()));

// Result → Option
let value: Option<i32> = some_result.ok();   // Discards Err
let error: Option<E> = some_result.err();    // Discards Ok

// transpose (Option<Result<T, E>> ⟷ Result<Option<T>, E>)
let opt_result: Option<Result<i32, E>> = Some(Ok(42));
let result_opt: Result<Option<i32>, E> = opt_result.transpose();
// → Ok(Some(42))
```

### 3.2 Haskell's Either / Maybe

```haskell
-- Haskell: Either a b (Left = error, Right = success)

-- ========================================
-- Maybe: presence or absence of a value
-- ========================================
safeDivide :: Double -> Double -> Maybe Double
safeDivide _ 0 = Nothing
safeDivide a b = Just (a / b)

safeHead :: [a] -> Maybe a
safeHead []    = Nothing
safeHead (x:_) = Just x

-- Chaining Maybe (do notation)
lookupAddress :: Map String User -> String -> Maybe String
lookupAddress users name = do
    user <- Map.lookup name users      -- Maybe User
    address <- userAddress user        -- Maybe Address
    return (addressCity address)       -- Maybe String

-- ========================================
-- Either: detailed error information
-- ========================================
data AppError
    = NotFound String
    | ValidationError [String]
    | DatabaseError String
    | AuthError String
    deriving (Show)

parseAge :: String -> Either AppError Int
parseAge s = case reads s of
    [(n, "")] | n >= 0 && n <= 150 -> Right n
              | otherwise -> Left (ValidationError ["Age must be 0-150"])
    _ -> Left (ValidationError ["Invalid number format"])

-- Chaining Either
createUser :: String -> String -> Either AppError User
createUser name ageStr = do
    validatedName <- validateName name
    age <- parseAge ageStr
    Right (User validatedName age)

validateName :: String -> Either AppError String
validateName name
    | null name = Left (ValidationError ["Name is required"])
    | length name < 2 = Left (ValidationError ["Name too short"])
    | otherwise = Right name

-- ========================================
-- ExceptT (combining IO and Either with monad transformers)
-- ========================================
import Control.Monad.Except

type App = ExceptT AppError IO

getUser :: String -> App User
getUser userId = do
    result <- liftIO $ queryDB ("SELECT * FROM users WHERE id = " ++ userId)
    case result of
        Nothing -> throwError (NotFound $ "User " ++ userId)
        Just user -> return user

-- ========================================
-- MonadError type class
-- ========================================
handleError :: MonadError AppError m => m User -> m User
handleError action = catchError action $ \err -> case err of
    NotFound msg -> do
        liftIO $ putStrLn $ "Warning: " ++ msg
        return defaultUser
    _ -> throwError err  -- Re-throw other errors
```

### 3.3 TypeScript's Result Pattern

```typescript
// TypeScript: representing Result type with union types

// ========================================
// Custom Result type
// ========================================
type Result<T, E> =
    | { ok: true; value: T }
    | { ok: false; error: E };

function ok<T>(value: T): Result<T, never> {
    return { ok: true, value };
}

function err<E>(error: E): Result<never, E> {
    return { ok: false, error };
}

// ========================================
// Convenient Result methods (utility functions)
// ========================================
function mapResult<T, U, E>(
    result: Result<T, E>,
    fn: (value: T) => U
): Result<U, E> {
    if (result.ok) {
        return ok(fn(result.value));
    }
    return result;
}

function flatMapResult<T, U, E>(
    result: Result<T, E>,
    fn: (value: T) => Result<U, E>
): Result<U, E> {
    if (result.ok) {
        return fn(result.value);
    }
    return result;
}

function mapError<T, E, F>(
    result: Result<T, E>,
    fn: (error: E) => F
): Result<T, F> {
    if (result.ok) {
        return result;
    }
    return err(fn(result.error));
}

// ========================================
// Practical usage example
// ========================================
type AppError =
    | { code: "NOT_FOUND"; resource: string }
    | { code: "VALIDATION"; fields: Record<string, string[]> }
    | { code: "UNAUTHORIZED"; reason: string }
    | { code: "INTERNAL"; message: string };

function parseNumber(s: string): Result<number, string> {
    const n = Number(s);
    if (isNaN(n)) {
        return err(`Invalid number: ${s}`);
    }
    return ok(n);
}

function validateAge(age: number): Result<number, AppError> {
    if (age < 0 || age > 150) {
        return err({
            code: "VALIDATION",
            fields: { age: ["Must be between 0 and 150"] },
        });
    }
    return ok(age);
}

// Result pipeline
function processAgeInput(input: string): Result<number, AppError> {
    const parsed = parseNumber(input);
    if (!parsed.ok) {
        return err({
            code: "VALIDATION",
            fields: { age: [parsed.error] },
        });
    }
    return validateAge(parsed.value);
}

// ========================================
// neverthrow library
// ========================================
import { ok, err, Result, ResultAsync } from 'neverthrow';

function divide(a: number, b: number): Result<number, string> {
    if (b === 0) return err("Division by zero");
    return ok(a / b);
}

// Method chaining
const result = divide(10, 2)
    .map(n => n * 3)
    .mapErr(e => new Error(e))
    .andThen(n => n > 0 ? ok(n) : err(new Error("Must be positive")));

// ResultAsync (async version)
function fetchUser(id: string): ResultAsync<User, AppError> {
    return ResultAsync.fromPromise(
        fetch(`/api/users/${id}`).then(r => r.json()),
        (e) => ({ code: "INTERNAL" as const, message: String(e) })
    );
}

// combine (merge multiple Results)
const combined = Result.combine([
    parseNumber("10"),
    parseNumber("20"),
    parseNumber("30"),
]);
// → ok([10, 20, 30]) or err("...")
```

### 3.4 Go's Error Handling

```go
// Go: returning errors via multiple return values

// ========================================
// Basic pattern
// ========================================
func readFile(path string) (string, error) {
    data, err := os.ReadFile(path)
    if err != nil {
        return "", fmt.Errorf("read %s: %w", path, err)
    }
    return string(data), nil
}

// Caller side
content, err := readFile("config.txt")
if err != nil {
    log.Fatal(err)
}
// Forgetting to check err → processing continues with zero value ("") → source of bugs

// ========================================
// Error wrapping (Go 1.13+)
// ========================================
func processFile(path string) error {
    data, err := os.ReadFile(path)
    if err != nil {
        // Wrap with %w → inspectable with errors.Is/As
        return fmt.Errorf("process file %s: %w", path, err)
    }
    return processData(data)
}

// Error inspection
if errors.Is(err, os.ErrNotExist) {
    fmt.Println("File does not exist")
}

var pathErr *os.PathError
if errors.As(err, &pathErr) {
    fmt.Println("Path:", pathErr.Path)
}

// ========================================
// Custom error types
// ========================================
type AppError struct {
    Code    string `json:"code"`
    Message string `json:"message"`
    Err     error  `json:"-"`
}

func (e *AppError) Error() string {
    if e.Err != nil {
        return fmt.Sprintf("%s: %s (%v)", e.Code, e.Message, e.Err)
    }
    return fmt.Sprintf("%s: %s", e.Code, e.Message)
}

func (e *AppError) Unwrap() error {
    return e.Err
}

// Sentinel errors
var (
    ErrNotFound     = &AppError{Code: "NOT_FOUND", Message: "Resource not found"}
    ErrUnauthorized = &AppError{Code: "UNAUTHORIZED", Message: "Unauthorized"}
    ErrForbidden    = &AppError{Code: "FORBIDDEN", Message: "Forbidden"}
)

func NewNotFoundError(resource, id string) *AppError {
    return &AppError{
        Code:    "NOT_FOUND",
        Message: fmt.Sprintf("%s with id '%s' not found", resource, id),
    }
}

// ========================================
// Error handling patterns
// ========================================
// Pattern 1: return immediately
func getUser(id string) (*User, error) {
    user, err := db.FindUser(id)
    if err != nil {
        return nil, fmt.Errorf("get user %s: %w", id, err)
    }
    if user == nil {
        return nil, NewNotFoundError("User", id)
    }
    return user, nil
}

// Pattern 2: cleanup with defer
func processWithTransaction(db *sql.DB) error {
    tx, err := db.Begin()
    if err != nil {
        return fmt.Errorf("begin transaction: %w", err)
    }
    defer func() {
        if err != nil {
            tx.Rollback()
        }
    }()

    if err = step1(tx); err != nil {
        return fmt.Errorf("step1: %w", err)
    }
    if err = step2(tx); err != nil {
        return fmt.Errorf("step2: %w", err)
    }

    return tx.Commit()
}

// Pattern 3: error groups (collecting multiple errors)
type MultiError struct {
    Errors []error
}

func (me *MultiError) Error() string {
    msgs := make([]string, len(me.Errors))
    for i, err := range me.Errors {
        msgs[i] = err.Error()
    }
    return strings.Join(msgs, "; ")
}

func validateUser(user *User) error {
    var errs []error
    if user.Name == "" {
        errs = append(errs, fmt.Errorf("name is required"))
    }
    if user.Email == "" {
        errs = append(errs, fmt.Errorf("email is required"))
    }
    if user.Age < 0 || user.Age > 150 {
        errs = append(errs, fmt.Errorf("age must be 0-150"))
    }
    if len(errs) > 0 {
        return &MultiError{Errors: errs}
    }
    return nil
}
```

---

## 4. Cross-Language Error Handling Comparison

```
┌──────────────────┬──────────────────────┬────────────────────────────────┐
│ Approach         │ Representative       │ Characteristics                │
│                  │ Languages            │                                │
├──────────────────┼──────────────────────┼────────────────────────────────┤
│ Exceptions       │ Python, Java, C#,    │ Implicit propagation           │
│                  │ JavaScript, Ruby     │ Invisible control flow         │
│                  │                      │ Separation of normal/exception │
├──────────────────┼──────────────────────┼────────────────────────────────┤
│ Result Type      │ Rust, Haskell,       │ Explicit propagation           │
│                  │ Elm, OCaml, F#       │ Type-safe representation       │
│                  │                      │ Handling enforced at compile   │
├──────────────────┼──────────────────────┼────────────────────────────────┤
│ Error Codes      │ C, Go               │ Simple but                     │
│                  │                      │ risk of forgetting checks      │
│                  │                      │ Go enforces by convention      │
├──────────────────┼──────────────────────┼────────────────────────────────┤
│ Hybrid           │ Swift(throw+Result)  │ Choose by context              │
│                  │ Kotlin(throw+Result) │ Maximum flexibility            │
│                  │ Scala(Try+Either)    │                                │
├──────────────────┼──────────────────────┼────────────────────────────────┤
│ Panic            │ Rust(panic!),        │ Unrecoverable errors           │
│                  │ Go(panic/recover)    │ Assumes process termination    │
└──────────────────┴──────────────────────┴────────────────────────────────┘
```

### 4.1 Swift's Error Handling

```swift
// Swift: hybrid approach (throws + Result + Optional)

// ========================================
// throws / do-catch
// ========================================
enum AppError: Error {
    case notFound(String)
    case validation([String])
    case unauthorized
}

func getUser(id: String) throws -> User {
    guard let user = db.findUser(id: id) else {
        throw AppError.notFound("User \(id)")
    }
    return user
}

// Caller side
do {
    let user = try getUser(id: "123")
    print(user.name)
} catch AppError.notFound(let resource) {
    print("Not found: \(resource)")
} catch {
    print("Unexpected error: \(error)")
}

// try? — convert error to Optional
let user: User? = try? getUser(id: "123")

// try! — crash on error (only when you're certain)
let user: User = try! getUser(id: "known-id")

// ========================================
// Result type (Swift 5+)
// ========================================
func fetchData(url: URL, completion: (Result<Data, Error>) -> Void) {
    URLSession.shared.dataTask(with: url) { data, response, error in
        if let error = error {
            completion(.failure(error))
            return
        }
        if let data = data {
            completion(.success(data))
        }
    }.resume()
}

// Using Result
fetchData(url: apiURL) { result in
    switch result {
    case .success(let data):
        processData(data)
    case .failure(let error):
        handleError(error)
    }
}

// map / flatMap
let decoded: Result<User, Error> = result
    .map { data in try JSONDecoder().decode(User.self, from: data) }
```

### 4.2 Kotlin's Error Handling

```kotlin
// Kotlin: hybrid approach (exceptions + Result + sealed class)

// ========================================
// Representing errors with sealed classes
// ========================================
sealed class AppResult<out T> {
    data class Success<T>(val value: T) : AppResult<T>()
    data class Failure(val error: AppError) : AppResult<Nothing>()
}

sealed class AppError {
    data class NotFound(val resource: String) : AppError()
    data class Validation(val errors: List<String>) : AppError()
    data object Unauthorized : AppError()
}

fun getUser(id: String): AppResult<User> {
    val user = db.findUser(id) ?: return AppResult.Failure(
        AppError.NotFound("User $id")
    )
    return AppResult.Success(user)
}

// Handling with when expression
when (val result = getUser("123")) {
    is AppResult.Success -> println("User: ${result.value.name}")
    is AppResult.Failure -> when (result.error) {
        is AppError.NotFound -> println("Not found: ${result.error.resource}")
        is AppError.Validation -> println("Validation: ${result.error.errors}")
        is AppError.Unauthorized -> println("Unauthorized")
    }
}

// ========================================
// Kotlin stdlib Result type
// ========================================
val result: Result<Int> = runCatching {
    "42".toInt()
}

result
    .map { it * 2 }
    .onSuccess { println("Value: $it") }
    .onFailure { println("Error: ${it.message}") }

val value = result.getOrDefault(0)
val value2 = result.getOrElse { error -> handleError(error); 0 }

// ========================================
// require / check (precondition checks)
// ========================================
fun processOrder(order: Order) {
    require(order.items.isNotEmpty()) { "Order must have items" }
    require(order.total > 0) { "Order total must be positive" }
    check(order.status == OrderStatus.PENDING) { "Order must be pending" }

    // require → IllegalArgumentException
    // check → IllegalStateException
}
```

---

## 5. Error Handling Best Practices

### 5.1 Recoverable vs Unrecoverable

```
1. Recoverable errors
   → File not found → try a different path
   → Network timeout → retry
   → Validation error → provide feedback to the user
   → Authentication error → re-login

   Rust:  Result<T, E>
   Go:    return error
   Java:  Checked exceptions
   Python: Catch specific exceptions

2. Unrecoverable errors
   → Out of memory → terminate the process
   → Invariant violation → a bug; the program should be fixed
   → Critical missing config file → fail at startup

   Rust:  panic!()
   Go:    panic() (generally not used)
   Java:  RuntimeException
   Python: SystemExit
```

### 5.2 Designing Error Granularity

```rust
// Rust: designing appropriate error granularity

// Bad — too coarse
enum Error {
    SomethingWentWrong(String),
}

// Bad — too fine
enum Error {
    FileNotFoundAtPath(PathBuf),
    FilePermissionDenied(PathBuf),
    FileAlreadyExists(PathBuf),
    FileIsDirectory(PathBuf),
    // ... dozens of types
}

// Good — appropriate granularity (aligned with the domain)
#[derive(Error, Debug)]
enum UserServiceError {
    #[error("User not found: {0}")]
    NotFound(String),

    #[error("Validation failed")]
    Validation(Vec<ValidationIssue>),

    #[error("Duplicate email: {0}")]
    DuplicateEmail(String),

    #[error("Database error")]
    Database(#[source] sqlx::Error),

    #[error("External service error")]
    ExternalService(#[source] reqwest::Error),
}

// Separate error types per layer
// Repository layer
#[derive(Error, Debug)]
enum RepositoryError {
    #[error("Record not found")]
    NotFound,
    #[error("Constraint violation: {0}")]
    Constraint(String),
    #[error("Connection error")]
    Connection(#[source] sqlx::Error),
}

// Service layer (convert repository errors)
impl From<RepositoryError> for UserServiceError {
    fn from(err: RepositoryError) -> Self {
        match err {
            RepositoryError::NotFound => UserServiceError::NotFound("unknown".to_string()),
            RepositoryError::Constraint(msg) if msg.contains("email") => {
                UserServiceError::DuplicateEmail(msg)
            }
            other => UserServiceError::Database(other.into()),
        }
    }
}
```

### 5.3 Designing Error Messages

```
Three elements of an error message:
  1. What     — what failed
  2. Where    — which resource/operation
  3. Why      — what was the cause

Bad examples:
  "Error occurred"
  "Failed"
  "Something went wrong"
  "Invalid input"

Good examples:
  "Failed to read config file '/etc/app.toml': Permission denied"
  "User 'alice@example.com' not found in database 'users'"
  "Validation failed for field 'email': must contain '@'"
  "Connection to Redis at localhost:6379 timed out after 5s"

Error message guidelines:
  - Not too technical, yet not too vague
  - Do not include sensitive information (passwords, tokens)
  - Separate user-facing messages from developer-facing messages
  - Include context information (IDs, paths, parameters)
  - Include hints for resolution (when possible)
```

### 5.4 Error Propagation Patterns

```rust
// Rust: error propagation and adding context

// Pattern 1: propagate as-is (? operator)
fn load_config() -> Result<Config, AppError> {
    let path = find_config_path()?;
    let content = read_file(&path)?;
    let config = parse_config(&content)?;
    Ok(config)
}

// Pattern 2: propagate with added context
fn load_config() -> Result<Config, anyhow::Error> {
    let path = find_config_path()
        .context("Failed to find config file")?;
    let content = read_file(&path)
        .context(format!("Failed to read {}", path.display()))?;
    let config = parse_config(&content)
        .context("Failed to parse config")?;
    Ok(config)
}

// Pattern 3: convert and propagate
fn load_user_config(user_id: &str) -> Result<UserConfig, UserError> {
    let path = find_config_path()
        .map_err(|_| UserError::ConfigNotFound(user_id.to_string()))?;
    let content = read_file(&path)
        .map_err(|e| UserError::ConfigReadError {
            user_id: user_id.to_string(),
            source: e,
        })?;
    Ok(parse_user_config(&content)?)
}

// Pattern 4: recover from error
fn load_config_with_fallback() -> Config {
    match load_config() {
        Ok(config) => config,
        Err(e) => {
            eprintln!("Warning: Failed to load config: {}. Using defaults.", e);
            Config::default()
        }
    }
}
```

```python
# Python: error propagation patterns

# Pattern 1: propagate as-is (do nothing)
def load_config():
    path = find_config_path()    # Exception propagates
    content = read_file(path)    # Exception propagates
    return parse_config(content) # Exception propagates

# Pattern 2: add context
def load_config():
    try:
        path = find_config_path()
    except FileNotFoundError:
        raise ConfigError("Config file not found") from None

    try:
        content = read_file(path)
    except IOError as e:
        raise ConfigError(f"Failed to read {path}") from e

    try:
        return parse_config(content)
    except ValueError as e:
        raise ConfigError(f"Invalid config format") from e

# Pattern 3: recover
def load_config_with_fallback():
    try:
        return load_config()
    except ConfigError as e:
        logger.warning(f"Failed to load config: {e}. Using defaults.")
        return Config.default()

# Pattern 4: retry
def load_with_retry(url, max_retries=3, delay=1.0):
    last_error = None
    for attempt in range(max_retries):
        try:
            return fetch(url)
        except (ConnectionError, TimeoutError) as e:
            last_error = e
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay * (2 ** attempt))  # Exponential backoff
    raise last_error
```

---

## 6. Practical Patterns

### 6.1 Web API Error Handling

```python
# Python (FastAPI): error handling

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

app = FastAPI()

# Custom exception handler
@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": str(exc),
                "details": exc.details,
            }
        },
    )

@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Validation failed",
                "details": {"fields": exc.errors},
            }
        },
    )

@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An internal error occurred",
            }
        },
    )

@app.get("/api/users/{user_id}")
async def get_user(user_id: str):
    user = await user_service.get_user(user_id)  # NotFoundError propagates
    return user
```

```rust
// Rust (Axum): error handling
use axum::{
    response::{IntoResponse, Response},
    http::StatusCode,
    Json,
};

#[derive(Debug, thiserror::Error)]
enum ApiError {
    #[error("Not found: {0}")]
    NotFound(String),
    #[error("Validation error")]
    Validation(Vec<String>),
    #[error("Unauthorized")]
    Unauthorized,
    #[error("Internal error")]
    Internal(#[from] anyhow::Error),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, code, message) = match &self {
            ApiError::NotFound(msg) => (StatusCode::NOT_FOUND, "NOT_FOUND", msg.clone()),
            ApiError::Validation(errors) => (
                StatusCode::BAD_REQUEST,
                "VALIDATION_ERROR",
                errors.join(", "),
            ),
            ApiError::Unauthorized => (
                StatusCode::UNAUTHORIZED,
                "UNAUTHORIZED",
                "Unauthorized".to_string(),
            ),
            ApiError::Internal(e) => {
                tracing::error!("Internal error: {:?}", e);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "INTERNAL_ERROR",
                    "An internal error occurred".to_string(),
                )
            }
        };

        let body = serde_json::json!({
            "error": {
                "code": code,
                "message": message,
            }
        });

        (status, Json(body)).into_response()
    }
}

async fn get_user(Path(id): Path<String>) -> Result<Json<User>, ApiError> {
    let user = user_service.get_user(&id).await?;  // Converted to ApiError
    Ok(Json(user))
}
```

### 6.2 Batch Processing Error Handling

```python
# Python: error handling in batch processing

from dataclasses import dataclass, field

@dataclass
class BatchResult:
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    errors: list[dict] = field(default_factory=list)

def process_batch(items: list[dict]) -> BatchResult:
    result = BatchResult(total=len(items))

    for i, item in enumerate(items):
        try:
            process_item(item)
            result.succeeded += 1
        except ValidationError as e:
            result.failed += 1
            result.errors.append({
                "index": i,
                "item": item,
                "error_type": "validation",
                "message": str(e),
            })
        except ExternalServiceError as e:
            result.failed += 1
            result.errors.append({
                "index": i,
                "item": item,
                "error_type": "external_service",
                "message": str(e),
            })
            # Abort if external service errors continue
            if result.failed > result.total * 0.5:
                logger.error("Too many failures, aborting batch")
                break
        except Exception as e:
            result.failed += 1
            result.errors.append({
                "index": i,
                "item": item,
                "error_type": "unexpected",
                "message": str(e),
            })
            logger.exception(f"Unexpected error processing item {i}")

    return result
```

```rust
// Rust: batch processing error handling
#[derive(Debug)]
struct BatchResult<T> {
    succeeded: Vec<T>,
    failed: Vec<BatchError>,
}

#[derive(Debug)]
struct BatchError {
    index: usize,
    error: AppError,
}

fn process_batch<T, F>(items: &[Item], processor: F) -> BatchResult<T>
where
    F: Fn(&Item) -> Result<T, AppError>,
{
    let mut result = BatchResult {
        succeeded: Vec::new(),
        failed: Vec::new(),
    };

    for (index, item) in items.iter().enumerate() {
        match processor(item) {
            Ok(output) => result.succeeded.push(output),
            Err(error) => result.failed.push(BatchError { index, error }),
        }
    }

    result
}

// Parallel batch processing (rayon)
fn process_batch_parallel(items: &[Item]) -> BatchResult<Output> {
    let results: Vec<(usize, Result<Output, AppError>)> = items
        .par_iter()
        .enumerate()
        .map(|(i, item)| (i, process_item(item)))
        .collect();

    let mut batch_result = BatchResult {
        succeeded: Vec::new(),
        failed: Vec::new(),
    };

    for (index, result) in results {
        match result {
            Ok(output) => batch_result.succeeded.push(output),
            Err(error) => batch_result.failed.push(BatchError { index, error }),
        }
    }

    batch_result
}
```

### 6.3 Retry Pattern

```typescript
// TypeScript: retry pattern

interface RetryOptions {
    maxRetries: number;
    baseDelay: number;     // milliseconds
    maxDelay: number;      // milliseconds
    backoffFactor: number; // exponential backoff multiplier
    retryableErrors?: string[];
}

async function withRetry<T>(
    operation: () => Promise<T>,
    options: RetryOptions,
): Promise<T> {
    const {
        maxRetries,
        baseDelay,
        maxDelay,
        backoffFactor,
        retryableErrors,
    } = options;

    let lastError: Error | undefined;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
        try {
            return await operation();
        } catch (error) {
            lastError = error as Error;

            // Check if the error is retryable
            if (retryableErrors && !isRetryable(error, retryableErrors)) {
                throw error;
            }

            if (attempt < maxRetries) {
                const delay = Math.min(
                    baseDelay * Math.pow(backoffFactor, attempt),
                    maxDelay,
                );
                // Add jitter (random variation)
                const jitter = delay * 0.1 * Math.random();
                console.log(
                    `Attempt ${attempt + 1} failed, retrying in ${delay + jitter}ms...`
                );
                await sleep(delay + jitter);
            }
        }
    }

    throw lastError;
}

function isRetryable(error: unknown, retryableErrors: string[]): boolean {
    if (error instanceof Error) {
        return retryableErrors.some(re =>
            error.message.includes(re) || error.name === re
        );
    }
    return false;
}

// Usage example
const data = await withRetry(
    () => fetch("/api/data").then(r => r.json()),
    {
        maxRetries: 3,
        baseDelay: 1000,
        maxDelay: 10000,
        backoffFactor: 2,
        retryableErrors: ["ECONNREFUSED", "ETIMEDOUT", "503"],
    },
);
```

---

## 7. Error Handling Anti-Patterns

### 7.1 Common Mistakes

```python
# Bad — Anti-pattern 1: swallowing exceptions
try:
    process(data)
except Exception:
    pass  # Completely ignoring the error → hiding bugs

# Good — at least log it
try:
    process(data)
except Exception as e:
    logger.error(f"Failed to process data: {e}")
    # Re-raise if necessary
```

```python
# Bad — Anti-pattern 2: overly broad exception catch
try:
    user = get_user(user_id)
    order = create_order(user, items)
    payment = process_payment(order)
except Exception as e:
    return {"error": str(e)}  # Cannot tell what failed

# Good — catch specific exceptions
try:
    user = get_user(user_id)
except UserNotFoundError:
    return {"error": f"User {user_id} not found"}

try:
    order = create_order(user, items)
except ValidationError as e:
    return {"error": f"Invalid order: {e}"}

try:
    payment = process_payment(order)
except PaymentError as e:
    return {"error": f"Payment failed: {e}"}
```

```javascript
// Bad — Anti-pattern 3: throwing strings
throw "Something went wrong";  // Not an Error object

// Good — use Error objects
throw new Error("Something went wrong");
throw new AppError("Something went wrong", "UNKNOWN", 500);
```

```go
// Bad — Anti-pattern 4: skipping error checks
data, _ := readFile(path)  // Error ignored
processData(data)           // data might be a zero value, potentially causing a crash

// Good — always check errors
data, err := readFile(path)
if err != nil {
    return fmt.Errorf("read file: %w", err)
}
processData(data)
```

```rust
// Bad — Anti-pattern 5: overuse of unwrap()
let config = load_config().unwrap();  // Panics
let user = get_user(id).unwrap();     // Panics

// Good — proper error handling
let config = load_config()
    .context("Failed to load config")?;
let user = get_user(id)
    .map_err(|e| AppError::NotFound(format!("user {}", id)))?;

// Cases where unwrap() is acceptable:
// - Test code
// - When it's logically provable that failure cannot occur
// - Prototypes (with TODO comments)
let regex = Regex::new(r"^\d+$").unwrap();  // Literal determined at compile time
```

---


## FAQ

### Q1: What is the most important point when learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not only through theory but also by actually writing code and verifying its behavior.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend solidly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this applied in practice?

Knowledge of this topic is frequently applied in day-to-day development work. It becomes especially important during code reviews and architecture design.

---

## Summary

| Approach | Error Propagation | Type Safety | Enforcement | Representative Languages |
|----------|------------------|-------------|-------------|------------------------|
| Exceptions | Implicit (throw) | Low | Java only | Python, Java, JS |
| Result | Explicit (?/map) | High | Compile-time | Rust, Haskell |
| Either | Explicit (bind) | High | Compile-time | Haskell, Scala |
| Error codes | Explicit (if err) | Low | None | Go, C |
| Hybrid | Both | Medium | Partial | Swift, Kotlin |

### Decision Criteria

```
When to use exceptions:
  - Deep call hierarchies where you want to handle errors collectively at a higher level
  - In languages whose libraries use exceptions (Python, Java, JS)

When to use Result:
  - Errors are "expected" (validation, search misses)
  - You want to maximize type safety
  - Functional-style codebases

When to use error codes:
  - Simplicity is the top priority (C, Go)
  - Few types of errors
  - Performance is critical

When to use hybrid:
  - Choose based on team proficiency
  - Use differently at API boundaries vs internal logic
```

---

## Recommended Next Guides

---

## References
1. Klabnik, S. & Nichols, C. "The Rust Programming Language." Ch.9, 2023.
2. Bloch, J. "Effective Java." 3rd Ed, Item 69-77, Addison-Wesley, 2018.
3. Martin, R. "Clean Code." Ch.7 (Error Handling), Prentice Hall, 2008.
4. Donovan, A. & Kernighan, B. "The Go Programming Language." Ch.5.4, Addison-Wesley, 2015.
5. "Error Handling in Rust." doc.rust-lang.org/book/ch09-00-error-handling.html.
6. "PEP 3134: Exception Chaining and Embedded Tracebacks." python.org, 2005.
7. "thiserror crate documentation." docs.rs/thiserror.
8. "anyhow crate documentation." docs.rs/anyhow.
9. "neverthrow library documentation." github.com/supermacro/neverthrow.
10. Lipovaca, M. "Learn You a Haskell for Great Good!" Ch.8, No Starch Press, 2011.
