# Exception Handling

> Exceptions are a mechanism for representing "abnormal situations that cannot be handled through normal control flow." Understand the proper usage of try/catch/finally, exception hierarchy design, and the checked vs unchecked debate.

## What You Will Learn

- [ ] Understand the mechanism of exception handling and call stack unwinding
- [ ] Grasp the difference between proper use and abuse of exceptions
- [ ] Learn the differences in exception models across languages
- [ ] Master correct patterns and anti-patterns for try/catch/finally
- [ ] Understand the concept of Exception Safety
- [ ] Learn about performance impact and optimal usage decisions


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts

---

## 1. Fundamentals of Exception Handling

### 1.1 What Are Exceptions?

Exceptions are a mechanism for expressing abnormal situations that cannot be handled through a program's normal execution flow. When an exception occurs, normal control flow (sequential execution, conditional branching, loops) is interrupted, and the call stack is unwound while searching for an appropriate handler.

```
Basic flow of exception handling:

  1. Exception occurrence (throw / raise)
     -> A function declares "I cannot handle this situation"

  2. Call stack unwinding (Stack Unwinding)
     -> Traverses back up the call stack until a catch/except is found
     -> All intermediate functions are skipped

  3. Exception capture (catch / except)
     -> An appropriate handler processes the exception

  4. Cleanup (finally / defer / with)
     -> Guarantees resource release

A world without exception handling:
  -> Every function must return an error code
  -> Callers must check for errors every time
  -> Error handling code buries the actual logic
  -> C's errno approach is exactly this

A world with exception handling:
  -> Normal-path code and error-path code can be separated
  -> Errors propagate automatically (no explicit passing needed)
  -> They can be handled collectively where they can be addressed
```

### 1.2 History of Exceptions

```
Evolution of exception handling:

  1960s: PL/I ON condition handling
    -> The first structured exception handling

  1985: Exceptions introduced in C++
    -> The prototype of try/catch/throw

  1995: Java's checked exceptions
    -> Compiler-enforced error handling

  2000s: C#, Python, Ruby unchecked exceptions
    -> A reaction against checked exceptions

  2010s: Go's multiple return values, Rust's Result<T, E>
    -> The rise of error handling without exceptions

  2020s: TypeScript's Effect, Rust's ? operator
    -> Refinement of type-safe error handling
```

---

## 2. try/catch/finally

### 2.1 Basic Pattern in TypeScript

```typescript
// TypeScript: Basic exception handling
async function fetchUserData(userId: string): Promise<UserData> {
  try {
    const response = await fetch(`/api/users/${userId}`);

    if (!response.ok) {
      throw new HttpError(response.status, `HTTP ${response.status}`);
    }

    const data = await response.json();
    return data;

  } catch (error) {
    if (error instanceof HttpError) {
      if (error.status === 404) {
        throw new UserNotFoundError(userId);
      }
      throw new ApiError(`API error: ${error.message}`);
    }
    // Network errors, etc.
    throw new NetworkError("Network request failed");

  } finally {
    // Executed regardless of success or failure
    // Use for resource cleanup
    logger.log(`fetchUserData completed for ${userId}`);
  }
}
```

### 2.2 Exception Handling in Python

```python
# Python: Exception handling
def parse_config(path: str) -> dict:
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        raise ConfigError(f"Config file not found: {path}")
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON in {path}: {e}")
    except PermissionError:
        raise ConfigError(f"Permission denied: {path}")
    finally:
        logger.info(f"Config parsing attempted for {path}")
```

### 2.3 Python's else Clause

```python
# Python-specific: try/except/else/finally
def process_file(path: str) -> ProcessResult:
    """
    The else clause is executed only when the try block completes
    without raising an exception.
    Unlike finally, it is not executed when an exception occurs.
    """
    file_handle = None
    try:
        file_handle = open(path, 'r')
        data = file_handle.read()
    except FileNotFoundError:
        logger.warning(f"File not found: {path}")
        return ProcessResult(success=False, error="File not found")
    except PermissionError:
        logger.warning(f"Permission denied: {path}")
        return ProcessResult(success=False, error="Permission denied")
    else:
        # Executed only if try succeeds
        # Exceptions raised here are NOT caught by the except blocks
        result = parse_and_validate(data)
        logger.info(f"Successfully processed: {path}")
        return ProcessResult(success=True, data=result)
    finally:
        # Executed regardless of success or failure
        if file_handle:
            file_handle.close()

# Reasons to use the else clause:
# 1. Keeps the try block minimal (avoids catching unintended exceptions)
# 2. Clearly separates "success-only" processing
# 3. Makes the code's intent explicit
```

### 2.4 Java's try-with-resources

```java
// Java: try-with-resources (auto-closing AutoCloseable)
public List<String> readLines(String path) throws IOException {
    // Resources declared in try() are automatically closed
    try (BufferedReader reader = new BufferedReader(new FileReader(path))) {
        List<String> lines = new ArrayList<>();
        String line;
        while ((line = reader.readLine()) != null) {
            lines.add(line);
        }
        return lines;
    }
    // reader.close() is called automatically (regardless of exceptions)
}

// Managing multiple resources
public void copyFile(String src, String dst) throws IOException {
    try (
        InputStream in = new FileInputStream(src);
        OutputStream out = new FileOutputStream(dst)
    ) {
        byte[] buffer = new byte[8192];
        int bytesRead;
        while ((bytesRead = in.read(buffer)) != -1) {
            out.write(buffer, 0, bytesRead);
        }
    }
    // Both in and out are automatically closed
    // Exceptions during close() are retained as Suppressed Exceptions
}

// Retrieving Suppressed Exceptions
public void demonstrateSuppressedException() {
    try {
        try (AutoCloseableResource resource = new AutoCloseableResource()) {
            throw new RuntimeException("Main exception");
        }
        // If resource.close() also throws an exception
    } catch (Exception e) {
        System.out.println("Main: " + e.getMessage());
        for (Throwable suppressed : e.getSuppressed()) {
            System.out.println("Suppressed: " + suppressed.getMessage());
        }
    }
}
```

### 2.5 C# using Statement

```csharp
// C#: using statement (auto-disposal of IDisposable)
public async Task<string> ReadFileAsync(string path)
{
    // using declaration (C# 8.0+): auto-Dispose at end of scope
    using var stream = new FileStream(path, FileMode.Open);
    using var reader = new StreamReader(stream);
    return await reader.ReadToEndAsync();
}

// using block (traditional syntax)
public void ProcessData(string connectionString)
{
    using (var connection = new SqlConnection(connectionString))
    {
        connection.Open();
        using (var command = new SqlCommand("SELECT * FROM Users", connection))
        using (var reader = command.ExecuteReader())
        {
            while (reader.Read())
            {
                ProcessRow(reader);
            }
        }
    }
    // connection, command, reader are all auto-Disposed
}

// await using (async Dispose, C# 8.0+)
public async Task ProcessStreamAsync()
{
    await using var stream = new AsyncStream();
    await stream.WriteAsync(data);
}
```

### 2.6 Go's defer

```go
// Go: Cleanup with defer
func readConfig(path string) (*Config, error) {
    file, err := os.Open(path)
    if err != nil {
        return nil, fmt.Errorf("failed to open config: %w", err)
    }
    defer file.Close() // Always executed when the function returns

    decoder := json.NewDecoder(file)
    var config Config
    if err := decoder.Decode(&config); err != nil {
        return nil, fmt.Errorf("failed to decode config: %w", err)
    }
    return &config, nil
}

// defer executes in LIFO (last in, first out) order
func multipleDefers() {
    fmt.Println("start")
    defer fmt.Println("first defer")  // Executed 3rd
    defer fmt.Println("second defer") // Executed 2nd
    defer fmt.Println("third defer")  // Executed 1st
    fmt.Println("end")
}
// Output: start, end, third defer, second defer, first defer

// Recovering from panics with defer + recover
func safeOperation() (result string, err error) {
    defer func() {
        if r := recover(); r != nil {
            err = fmt.Errorf("panic recovered: %v", r)
        }
    }()

    // Even if a panic occurs, it is captured by recover
    riskyOperation()
    return "success", nil
}
```

### 2.7 Rust's Drop Trait and RAII

```rust
// Rust: RAII (Resource Acquisition Is Initialization)
// Auto-cleanup with the Drop trait
struct DatabaseConnection {
    connection_string: String,
    is_open: bool,
}

impl DatabaseConnection {
    fn new(conn_str: &str) -> Result<Self, DbError> {
        // Establish connection
        Ok(DatabaseConnection {
            connection_string: conn_str.to_string(),
            is_open: true,
        })
    }

    fn query(&self, sql: &str) -> Result<Vec<Row>, DbError> {
        if !self.is_open {
            return Err(DbError::ConnectionClosed);
        }
        // Execute query
        Ok(vec![])
    }
}

impl Drop for DatabaseConnection {
    fn drop(&mut self) {
        if self.is_open {
            // Close connection (called automatically)
            println!("Connection closed: {}", self.connection_string);
            self.is_open = false;
        }
    }
}

fn process_data() -> Result<(), DbError> {
    let conn = DatabaseConnection::new("postgres://localhost/mydb")?;
    let rows = conn.query("SELECT * FROM users")?;
    // conn is automatically dropped at the end of scope
    // -> Drop::drop() is called and the connection is closed
    Ok(())
}
```

---

## 3. Call Stack Unwinding

### 3.1 How Stack Unwinding Works

```
Call stack when an exception occurs:

  main()
    └── processOrder()
        └── validatePayment()
            └── chargeCard()
                └── apiCall()  ← Exception thrown!

  Unwinding (stack unwind):
  apiCall()    -> No catch -> Propagate
  chargeCard() -> No catch -> Propagate
  validatePayment() -> catch found! -> Handle here
                     -> Or re-throw

  Principles:
  -> Catch exceptions where they can be handled
  -> If you can't handle it, don't catch it (let it propagate to a higher level)
  -> Swallowing exceptions (catch and ignore) is strictly prohibited
```

### 3.2 How to Read Stack Traces

```typescript
// Stack trace example
// Error: User not found: user-123
//     at UserService.getUser (/app/services/user.ts:45:11)
//     at OrderService.createOrder (/app/services/order.ts:23:28)
//     at OrderController.create (/app/controllers/order.ts:15:30)
//     at Layer.handle [as handle_request] (/app/node_modules/express/lib/router/layer.js:95:5)
//     at next (/app/node_modules/express/lib/router/route.js:144:13)

// How to read a stack trace:
// 1. The top line is the error message
// 2. The 2nd line is the origin of the exception (most important)
// 3. Lines further down are closer to the caller (root)
// 4. Frames inside node_modules are usually ignored

// Custom stack trace
class AppError extends Error {
    constructor(message: string) {
        super(message);
        this.name = this.constructor.name;
        // Exclude the constructor itself from the stack
        Error.captureStackTrace(this, this.constructor);
    }
}

// The top of the stack becomes the call site of the AppError constructor,
// not the constructor itself
```

### 3.3 Stack Traces in Asynchronous Code

```typescript
// The problem of stack traces being broken in async code
async function fetchUser(id: string): Promise<User> {
    const response = await fetch(`/api/users/${id}`);
    if (!response.ok) throw new Error("Fetch failed");
    return response.json();
}

// Enable async stack traces with Node.js --async-stack-traces flag
// Or build a cause chain with Error.cause

async function processUser(id: string): Promise<void> {
    try {
        const user = await fetchUser(id);
        await updateUser(user);
    } catch (error) {
        // ES2022: Build a cause chain with Error.cause
        throw new ProcessError("User processing failed", {
            cause: error,  // Retain the original error as the cause
        });
    }
}

// Traversing the cause chain
function getRootCause(error: Error): Error {
    let current = error;
    while (current.cause instanceof Error) {
        current = current.cause;
    }
    return current;
}

// Python exception chaining
// try:
//     result = parse_data(raw)
// except ParseError as e:
//     raise ProcessingError("Data processing failed") from e
//     # The original exception is stored in __cause__
```

### 3.4 Cost of Exception Propagation

```
Performance cost of exceptions:

  Entering a try block:
  -> Nearly zero cost (optimized in most runtimes)
  -> "Zero-cost exception" model (C++, Rust panics)

  Throwing an exception:
  -> Very high cost
  -> Stack trace construction: O(n) (n is stack depth)
  -> Stack unwinding: calling destructors for each frame
  -> Generally 100-1000x slower than a normal function return

  Benchmark example (approximate):
  ┌──────────────────────┬────────────────┐
  │ Operation            │ Relative Cost  │
  ├──────────────────────┼────────────────┤
  │ Function return      │ 1x             │
  │ Entering try block   │ ~1x            │
  │ Throwing exception   │ 100-1000x      │
  │ Stack trace          │ 200-2000x      │
  └──────────────────────┴────────────────┘

  Conclusion:
  -> Writing try/catch itself is not a cost issue
  -> Use exceptions only for "exceptional" situations
  -> Using exceptions for control flow inside loops is absolutely unacceptable
```

---

## 4. Using Exceptions: Appropriate Situations and Abuse

### 4.1 When to Use Exceptions

```
When exceptions should be used:
  ✓ File not found
  ✓ Network connection error
  ✓ Database connection failure
  ✓ Invalid data (validation error)
  ✓ Resource exhaustion (memory, disk)
  ✓ Configuration file inconsistency
  ✓ External service failure
  ✓ Authentication/authorization failure
  ✓ Data integrity violation
  ✓ Timeout

When exceptions should NOT be used:
  ✗ Normal control flow (can be decided with if/else)
  ✗ Expected situations (user input mistakes)
  ✗ Performance-critical code
  ✗ A collection being empty
  ✗ Search results returning 0 items
  ✗ Reaching the end of a file
```

### 4.2 Anti-pattern: Control Flow via Exceptions

```typescript
// BAD: Control flow via exceptions (anti-pattern)
function findUserByEmail(email: string): User | null {
    try {
        const user = db.query("SELECT * FROM users WHERE email = ?", [email]);
        if (!user) throw new Error("Not found");
        return user;
    } catch (e) {
        return null;  // "Not found" is not an exception
    }
}

// GOOD: Control flow via return values
function findUserByEmail(email: string): User | null {
    const user = db.query("SELECT * FROM users WHERE email = ?", [email]);
    return user ?? null;  // Return null (normal control flow)
}

// BAD: Using exceptions for control flow inside a loop (worst pattern)
function parseNumbers(inputs: string[]): number[] {
    const results: number[] = [];
    for (const input of inputs) {
        try {
            results.push(parseInt(input));
        } catch (e) {
            // Failing to parse = normal occurrence
            continue;
        }
    }
    return results;
}

// GOOD: Pre-validation
function parseNumbers(inputs: string[]): number[] {
    return inputs
        .filter(input => /^\d+$/.test(input))
        .map(input => parseInt(input, 10));
}
```

### 4.3 Anti-pattern: Pokemon Exception Handling

```typescript
// BAD: Catch everything (Pokemon: "Gotta Catch 'Em All")
async function processOrder(orderId: string): Promise<void> {
    try {
        const order = await getOrder(orderId);
        await validateOrder(order);
        await chargePayment(order);
        await sendConfirmation(order);
    } catch (error) {
        // Swallowing all exceptions!
        console.log("Something went wrong");
    }
}

// GOOD: Proper exception handling
async function processOrder(orderId: string): Promise<void> {
    try {
        const order = await getOrder(orderId);
        await validateOrder(order);
        await chargePayment(order);
        await sendConfirmation(order);
    } catch (error) {
        if (error instanceof ValidationError) {
            // Handle validation errors specifically
            await notifyUser(orderId, error.message);
            return;
        }
        if (error instanceof PaymentError) {
            // Payment errors may be retryable
            await queueForRetry(orderId);
            return;
        }
        // Propagate unexpected errors to upper levels
        throw error;
    }
}
```

### 4.4 Anti-pattern: Swallowing Exceptions

```python
# BAD: Swallowing Exception
def save_user(user):
    try:
        db.save(user)
    except Exception:
        pass  # Do nothing! Data is not saved but appears to have succeeded

# BAD: Log only and swallow
def save_user(user):
    try:
        db.save(user)
    except Exception as e:
        logger.error(f"Failed to save user: {e}")
        # The caller thinks it succeeded

# GOOD: Proper handling
def save_user(user) -> bool:
    try:
        db.save(user)
        return True
    except IntegrityError as e:
        logger.warning(f"Duplicate user: {e}")
        raise DuplicateUserError(user.email) from e
    except DatabaseError as e:
        logger.error(f"Database error: {e}")
        raise ServiceUnavailableError("Database unavailable") from e
```

### 4.5 Anti-pattern: Incorrect Re-throwing

```typescript
// BAD: Re-throw that loses information
try {
    await processPayment(order);
} catch (error) {
    throw new Error("Payment failed");  // All original error info is lost
}

// BAD: Re-throw that corrupts the stack trace
try {
    await processPayment(order);
} catch (error) {
    throw error;  // OK, but misses the opportunity to add context information
}

// GOOD: Re-throw that preserves the cause chain
try {
    await processPayment(order);
} catch (error) {
    throw new PaymentError("Payment processing failed", {
        cause: error,
        orderId: order.id,
        amount: order.total,
    });
}

// GOOD: Python exception chaining
# try:
#     process_payment(order)
# except StripeError as e:
#     raise PaymentError(f"Payment failed for order {order.id}") from e
```

---

## 5. Java: checked vs unchecked

### 5.1 Java Exception Hierarchy

```java
// Java exception hierarchy
// Throwable
// ├── Error (unrecoverable: OutOfMemoryError, StackOverflowError)
// └── Exception
//     ├── IOException (checked: enforced by compiler)
//     ├── SQLException (checked)
//     └── RuntimeException (unchecked: no compiler enforcement)
//         ├── NullPointerException
//         ├── IllegalArgumentException
//         └── IndexOutOfBoundsException

// checked exception: catch or throws declaration is mandatory
public String readFile(String path) throws IOException {
    return Files.readString(Path.of(path));
    // Must declare throws if IOException is not caught
}

// unchecked exception: no declaration needed
public void validateAge(int age) {
    if (age < 0) throw new IllegalArgumentException("Age must be >= 0");
    // No throws declaration needed
}
```

### 5.2 The Checked Exception Debate

```
The checked exception debate:

  Proponents:
  -> Prevents forgetting error handling
  -> Makes the API contract explicit
  -> Detects missing error handling at compile time

  Opponents (majority):
  -> Too much boilerplate
  -> Causes swallowing (catch and ignore)
  -> Kotlin, C#, Python, TS all use unchecked only
  -> Changes to exception specifications cascade through (throws propagation)
  -> Poor compatibility with lambda expressions

  Modern mainstream:
  -> unchecked exceptions + expressing errors with types (Result type)
```

### 5.3 Problems with Checked Exceptions (Concrete Examples)

```java
// Checked exception problem 1: Boilerplate
// Cannot use checked exceptions in lambda expressions
public List<String> readAllFiles(List<String> paths) throws IOException {
    // BAD: Compile error: checked exception in lambda
    // return paths.stream()
    //     .map(path -> Files.readString(Path.of(path))) // IOException!
    //     .collect(Collectors.toList());

    // GOOD: Workaround 1: Write try/catch inside the lambda (verbose)
    return paths.stream()
        .map(path -> {
            try {
                return Files.readString(Path.of(path));
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        })
        .collect(Collectors.toList());
}

// Checked exception problem 2: Cascading throws
// Low-level implementation details leak into upper-level interfaces
public interface UserRepository {
    // BAD: Implementation detail (SQL) leaks into the interface
    User findById(String id) throws SQLException;

    // GOOD: Abstracted error
    User findById(String id) throws RepositoryException;
}

// Checked exception problem 3: Encouraging swallowing
public void processData(String data) {
    try {
        riskyOperation(data);
    } catch (CheckedException e) {
        // Swallowed because it's tedious (worst pattern)
        // catch block exists solely to silence the compiler
    }
}
```

### 5.4 Kotlin's Approach

```kotlin
// Kotlin: Abolished checked exceptions
// No catch required even when calling Java's checked exceptions
fun readFile(path: String): String {
    return File(path).readText()  // IOException is treated as unchecked
}

// try can be used as an expression
val result: Int = try {
    input.toInt()
} catch (e: NumberFormatException) {
    0  // Default value
}

// runCatching (Result type-like usage)
val result: Result<Int> = runCatching {
    input.toInt()
}

result
    .onSuccess { value -> println("Parsed: $value") }
    .onFailure { error -> println("Failed: ${error.message}") }

val value: Int = result.getOrDefault(0)
val valueOrNull: Int? = result.getOrNull()
```

---

## 6. Exception Model Comparison Across Languages

### 6.1 TypeScript/JavaScript

```typescript
// TypeScript: All unchecked
// The type system cannot express exception types (no type annotation for thrown types)

// Problem of catching as any type (before TypeScript 4.0)
try {
    throw new Error("test");
} catch (error) {
    // error is unknown type (TypeScript 4.4+ useUnknownInCatchVariables)
    // Need to narrow the type
    if (error instanceof Error) {
        console.log(error.message);
    }
}

// Error types
// Error: Generic error
// TypeError: Type error
// ReferenceError: Reference to undefined variable
// RangeError: Out-of-range value
// SyntaxError: Syntax error
// URIError: URI encoding/decoding error

// Throwing non-Error objects (not recommended)
throw "An error occurred";    // BAD: string
throw 42;                     // BAD: number
throw { code: "ERR" };        // BAD: object
throw new Error("An error");  // GOOD: Error object
// Non-Error objects cannot provide stack traces
```

### 6.2 Python

```python
# Python: All unchecked + rich built-in exceptions

# Exception hierarchy
# BaseException
# ├── KeyboardInterrupt
# ├── SystemExit
# ├── GeneratorExit
# └── Exception
#     ├── StopIteration
#     ├── ArithmeticError
#     │   ├── ZeroDivisionError
#     │   └── OverflowError
#     ├── LookupError
#     │   ├── KeyError
#     │   └── IndexError
#     ├── OSError
#     │   ├── FileNotFoundError
#     │   └── PermissionError
#     ├── ValueError
#     ├── TypeError
#     └── RuntimeError

# Catching multiple exceptions simultaneously
try:
    result = process(data)
except (ValueError, TypeError) as e:
    logger.error(f"Data error: {e}")
except OSError as e:
    logger.error(f"System error: {e}")
except Exception as e:
    logger.error(f"Unexpected: {e}")
    raise  # Re-raise

# Context manager (with statement)
class DatabaseConnection:
    def __enter__(self):
        self.conn = create_connection()
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.conn.rollback()
            logger.error(f"Transaction rolled back: {exc_val}")
        else:
            self.conn.commit()
        self.conn.close()
        return False  # Returning True would swallow the exception

# Usage
with DatabaseConnection() as conn:
    conn.execute("INSERT INTO users ...")
    conn.execute("UPDATE accounts ...")
    # If an exception occurs -> rollback + close
    # If completes normally -> commit + close

# Python 3.11+: ExceptionGroup (multiple simultaneous exceptions)
async def fetch_all(urls: list[str]) -> list[str]:
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(fetch(url)) for url in urls]
    # If multiple tasks fail simultaneously
    # an ExceptionGroup is raised

try:
    await fetch_all(urls)
except* ValueError as eg:
    # Handle only ValueError within the ExceptionGroup
    for exc in eg.exceptions:
        logger.error(f"Value error: {exc}")
except* OSError as eg:
    # Handle only OSError within the ExceptionGroup
    for exc in eg.exceptions:
        logger.error(f"OS error: {exc}")
```

### 6.3 C++ Exceptions

```cpp
// C++: Exceptions are available but performance cost is debated

#include <stdexcept>
#include <string>

// Standard exception hierarchy
// std::exception
// ├── std::logic_error
// │   ├── std::invalid_argument
// │   ├── std::out_of_range
// │   └── std::domain_error
// └── std::runtime_error
//     ├── std::overflow_error
//     ├── std::underflow_error
//     └── std::range_error

// Basic exception handling
void processFile(const std::string& path) {
    try {
        auto data = readFile(path);
        auto result = parseData(data);
        saveResult(result);
    } catch (const std::invalid_argument& e) {
        std::cerr << "Invalid argument: " << e.what() << std::endl;
    } catch (const std::runtime_error& e) {
        std::cerr << "Runtime error: " << e.what() << std::endl;
    } catch (...) {
        // Catch-all (C++ specific)
        std::cerr << "Unknown error" << std::endl;
        throw;  // Re-throw
    }
}

// noexcept specifier (C++11)
// Declares that this function does not throw exceptions
void swap(int& a, int& b) noexcept {
    int temp = a;
    a = b;
    b = temp;
}
// If an exception is thrown in a noexcept function, std::terminate() is called

// RAII pattern (Resource Acquisition Is Initialization)
class FileHandle {
    FILE* file_;
public:
    explicit FileHandle(const char* path) : file_(fopen(path, "r")) {
        if (!file_) throw std::runtime_error("Failed to open file");
    }
    ~FileHandle() {
        if (file_) fclose(file_);  // Auto-cleanup in destructor
    }
    // Copy prohibited
    FileHandle(const FileHandle&) = delete;
    FileHandle& operator=(const FileHandle&) = delete;
};
```

### 6.4 Go Error Handling (No Exceptions)

```go
// Go: Multiple return values instead of exceptions
// No exception mechanism (panic/recover exist but are rarely used)

func readConfig(path string) (*Config, error) {
    data, err := os.ReadFile(path)
    if err != nil {
        return nil, fmt.Errorf("failed to read %s: %w", path, err)
    }

    var config Config
    if err := json.Unmarshal(data, &config); err != nil {
        return nil, fmt.Errorf("failed to parse %s: %w", path, err)
    }

    return &config, nil
}

// Error wrapping and unwrapping
func processOrder(orderID string) error {
    order, err := getOrder(orderID)
    if err != nil {
        // Wrapping with %w (Go 1.13+)
        return fmt.Errorf("processOrder: get order: %w", err)
    }

    if err := validateOrder(order); err != nil {
        return fmt.Errorf("processOrder: validate: %w", err)
    }

    return nil
}

// errors.Is: Error identity check
if errors.Is(err, os.ErrNotExist) {
    fmt.Println("File does not exist")
}

// errors.As: Error type check
var pathErr *os.PathError
if errors.As(err, &pathErr) {
    fmt.Printf("Path error: %s, Op: %s\n", pathErr.Path, pathErr.Op)
}

// Go error handling debate:
// Proponents: Simple, explicit, hard to overlook
// Opponents: Repetitive if err != nil, boilerplate
```

### 6.5 Swift Error Handling

```swift
// Swift: do/try/catch + type-safe errors

// Enum conforming to the Error protocol
enum NetworkError: Error {
    case invalidURL(String)
    case timeout(seconds: Int)
    case serverError(statusCode: Int, message: String)
    case noConnection
}

// Function that throws errors (throws keyword)
func fetchData(from urlString: String) throws -> Data {
    guard let url = URL(string: urlString) else {
        throw NetworkError.invalidURL(urlString)
    }

    let (data, response) = try await URLSession.shared.data(from: url)

    if let httpResponse = response as? HTTPURLResponse,
       httpResponse.statusCode >= 400 {
        throw NetworkError.serverError(
            statusCode: httpResponse.statusCode,
            message: "Server error"
        )
    }

    return data
}

// Calling methods (3 patterns)
// 1. do/try/catch
do {
    let data = try fetchData(from: "https://api.example.com")
    process(data)
} catch NetworkError.invalidURL(let url) {
    print("Invalid URL: \(url)")
} catch NetworkError.timeout(let seconds) {
    print("Timeout after \(seconds)s")
} catch {
    print("Unknown error: \(error)")
}

// 2. try? (nil on failure)
let data = try? fetchData(from: "https://api.example.com")

// 3. try! (crash on failure)
let data = try! fetchData(from: "https://api.example.com")  // Dangerous
```

---

## 7. Designing Error Hierarchies

### 7.1 Error Hierarchy in TypeScript

```typescript
// Custom error hierarchy
class AppError extends Error {
  constructor(
    message: string,
    public readonly code: string,
    public readonly statusCode: number = 500,
    public readonly isOperational: boolean = true,
  ) {
    super(message);
    this.name = this.constructor.name;
  }
}

// Authentication error
class AuthenticationError extends AppError {
  constructor(message: string = "Authentication required") {
    super(message, "AUTH_REQUIRED", 401);
  }
}

// Authorization error
class AuthorizationError extends AppError {
  constructor(message: string = "Permission denied") {
    super(message, "FORBIDDEN", 403);
  }
}

// Resource not found
class NotFoundError extends AppError {
  constructor(resource: string, id: string) {
    super(`${resource} not found: ${id}`, "NOT_FOUND", 404);
  }
}

// Validation error
class ValidationError extends AppError {
  constructor(
    message: string,
    public readonly fields: Record<string, string[]>,
  ) {
    super(message, "VALIDATION_ERROR", 400);
  }
}

// Usage
function getUser(id: string): User {
  const user = db.findById(id);
  if (!user) throw new NotFoundError("User", id);
  return user;
}
```

### 7.2 Error Hierarchy Design Best Practices

```typescript
// Complete practical error hierarchy
abstract class AppError extends Error {
    abstract readonly code: string;
    abstract readonly httpStatus: number;
    readonly timestamp: string;
    readonly requestId?: string;

    constructor(
        message: string,
        public readonly isOperational: boolean = true,
        options?: { cause?: Error; requestId?: string }
    ) {
        super(message, { cause: options?.cause });
        this.name = this.constructor.name;
        this.timestamp = new Date().toISOString();
        this.requestId = options?.requestId;
        Error.captureStackTrace(this, this.constructor);
    }

    toJSON() {
        return {
            error: {
                code: this.code,
                message: this.message,
                timestamp: this.timestamp,
                ...(this.requestId && { requestId: this.requestId }),
            }
        };
    }
}

// ======== Authentication & Authorization ========
class AuthenticationError extends AppError {
    readonly code = "AUTHENTICATION_REQUIRED";
    readonly httpStatus = 401;
    constructor(message = "Authentication required", options?: { cause?: Error }) {
        super(message, true, options);
    }
}

class TokenExpiredError extends AuthenticationError {
    readonly code = "TOKEN_EXPIRED";
    constructor() {
        super("Token has expired");
    }
}

class InvalidCredentialsError extends AuthenticationError {
    readonly code = "INVALID_CREDENTIALS";
    constructor() {
        super("Invalid credentials");
    }
}

class AuthorizationError extends AppError {
    readonly code = "FORBIDDEN";
    readonly httpStatus = 403;
    constructor(
        public readonly requiredRole: string,
        public readonly actualRole: string,
    ) {
        super(`Insufficient permissions: ${requiredRole} required (current: ${actualRole})`);
    }
}

// ======== Resources ========
class NotFoundError extends AppError {
    readonly code = "NOT_FOUND";
    readonly httpStatus = 404;
    constructor(
        public readonly resourceType: string,
        public readonly resourceId: string,
    ) {
        super(`${resourceType} not found: ${resourceId}`);
    }
}

class ConflictError extends AppError {
    readonly code = "CONFLICT";
    readonly httpStatus = 409;
    constructor(message: string) {
        super(message);
    }
}

// ======== Validation ========
interface FieldError {
    field: string;
    message: string;
    value?: unknown;
}

class ValidationError extends AppError {
    readonly code = "VALIDATION_ERROR";
    readonly httpStatus = 400;
    constructor(public readonly fieldErrors: FieldError[]) {
        super(`Invalid input: ${fieldErrors.map(e => e.field).join(", ")}`);
    }

    toJSON() {
        return {
            error: {
                code: this.code,
                message: this.message,
                details: this.fieldErrors,
                timestamp: this.timestamp,
            }
        };
    }
}

// ======== External Services ========
class ExternalServiceError extends AppError {
    readonly code = "EXTERNAL_SERVICE_ERROR";
    readonly httpStatus = 502;
    constructor(
        public readonly serviceName: string,
        message: string,
        options?: { cause?: Error }
    ) {
        super(`${serviceName}: ${message}`, true, options);
    }
}

class RateLimitError extends AppError {
    readonly code = "RATE_LIMIT_EXCEEDED";
    readonly httpStatus = 429;
    constructor(
        public readonly retryAfterMs: number,
    ) {
        super(`Rate limit exceeded. Please retry after ${retryAfterMs}ms`);
    }
}

// ======== Internal Errors ========
class InternalError extends AppError {
    readonly code = "INTERNAL_ERROR";
    readonly httpStatus = 500;
    constructor(message: string, options?: { cause?: Error }) {
        super(message, false, options);  // isOperational = false
    }
}
```

### 7.3 Middleware Using Error Hierarchy

```typescript
// Express middleware for error handling
function errorHandler(
    error: Error,
    req: Request,
    res: Response,
    next: NextFunction
): void {
    // Assign request ID
    const requestId = req.headers['x-request-id'] as string || generateId();

    if (error instanceof AppError) {
        // Operational error: return appropriate response
        logger.warn({
            code: error.code,
            message: error.message,
            requestId,
            path: req.path,
            method: req.method,
        });

        res.status(error.httpStatus).json({
            ...error.toJSON(),
            error: {
                ...error.toJSON().error,
                requestId,
            }
        });
    } else {
        // Programmer error: hide internal information
        logger.error({
            message: error.message,
            stack: error.stack,
            requestId,
            path: req.path,
            method: req.method,
        });

        // Send to Sentry, etc.
        Sentry.captureException(error, {
            tags: { requestId },
            extra: { path: req.path },
        });

        res.status(500).json({
            error: {
                code: "INTERNAL_ERROR",
                message: "A server error occurred",
                requestId,
            }
        });
    }
}
```

---

## 8. Exception Safety

### 8.1 Levels of Exception Safety

```
4 levels of exception safety (originated in C++ but the concepts are language-agnostic):

  Level 0: Not exception-safe (No guarantee)
  -> Resource leaks and data corruption possible if an exception occurs
  -> Must be avoided

  Level 1: Basic guarantee
  -> No resource leaks if an exception occurs
  -> Object is valid but its content is indeterminate
  -> This is the minimum to aim for

  Level 2: Strong guarantee
  -> State is fully restored to pre-operation state if an exception occurs
  -> Equivalent to transaction rollback
  -> Costly but safe

  Level 3: No-throw guarantee
  -> Exceptions never occur
  -> Required for destructors and swap operations
```

### 8.2 Exception-safe Code Examples

```typescript
// BAD: Not exception-safe code (Level 0)
class UserService {
    async transferBalance(fromId: string, toId: string, amount: number): Promise<void> {
        const from = await this.getUser(fromId);
        from.balance -= amount;
        await this.saveUser(from);  // <- What if an exception occurs here?

        const to = await this.getUser(toId);
        to.balance += amount;
        await this.saveUser(to);  // <- from is debited, but to is not credited
        // Data inconsistency!
    }
}

// GOOD: Strong guarantee code (Level 2): Using transactions
class UserService {
    async transferBalance(fromId: string, toId: string, amount: number): Promise<void> {
        await this.db.transaction(async (tx) => {
            const from = await tx.getUser(fromId);
            const to = await tx.getUser(toId);

            if (from.balance < amount) {
                throw new InsufficientBalanceError(amount, from.balance);
            }

            from.balance -= amount;
            to.balance += amount;

            await tx.saveUser(from);
            await tx.saveUser(to);
            // tx.commit() is automatically executed at transaction end
            // tx.rollback() is automatically executed on exception
        });
    }
}

// GOOD: Strong guarantee code (Level 2): Copy-and-Swap idiom
class Configuration {
    private data: Map<string, string>;

    updateMultiple(updates: Record<string, string>): void {
        // 1. Create a copy
        const newData = new Map(this.data);

        // 2. Modify the copy (original data is untouched if an exception occurs here)
        for (const [key, value] of Object.entries(updates)) {
            if (!this.isValidKey(key)) {
                throw new ValidationError(`Invalid key: ${key}`);
            }
            newData.set(key, value);
        }

        // 3. Atomic swap (no-throw guarantee operation)
        this.data = newData;
    }
}
```

### 8.3 Proper Use of finally

```typescript
// finally best practices

// GOOD: Resource release
async function processWithLock(key: string): Promise<void> {
    const lock = await acquireLock(key);
    try {
        await doWork();
    } finally {
        await lock.release();  // Always release lock regardless of success or failure
    }
}

// GOOD: Temporary file deletion
async function processWithTempFile(): Promise<void> {
    const tempPath = await createTempFile();
    try {
        await writeToFile(tempPath, data);
        await processFile(tempPath);
    } finally {
        await fs.unlink(tempPath).catch(() => {});  // Best-effort deletion
    }
}

// BAD: Never return in finally
function badFinally(): number {
    try {
        throw new Error("error");
    } finally {
        return 42;  // BAD: The exception is swallowed and 42 is returned!
    }
}

// BAD: Never throw in finally (overwrites the original exception)
async function badFinallyThrow(): Promise<void> {
    try {
        throw new Error("original error");
    } finally {
        throw new Error("cleanup error");  // BAD: original error is lost
    }
}

// GOOD: Handle errors in finally safely
async function safeFinally(): Promise<void> {
    let resource: Resource | null = null;
    try {
        resource = await acquireResource();
        await resource.process();
    } finally {
        if (resource) {
            try {
                await resource.release();
            } catch (cleanupError) {
                logger.warn("Cleanup failed:", cleanupError);
                // Don't throw here to preserve the original exception
            }
        }
    }
}
```

---

## 9. Asynchronous Exception Handling

### 9.1 Promises and Exceptions

```typescript
// Exception handling in Promise chains
fetchUser(userId)
    .then(user => fetchOrders(user.id))
    .then(orders => calculateTotal(orders))
    .then(total => updateDashboard(total))
    .catch(error => {
        // Catches exceptions thrown anywhere in the chain
        if (error instanceof UserNotFoundError) {
            showEmptyState();
        } else if (error instanceof NetworkError) {
            showRetryButton();
        } else {
            showGenericError();
        }
    });

// Exception handling with async/await (recommended)
async function loadDashboard(userId: string): Promise<void> {
    try {
        const user = await fetchUser(userId);
        const orders = await fetchOrders(user.id);
        const total = calculateTotal(orders);
        updateDashboard(total);
    } catch (error) {
        handleDashboardError(error);
    }
}

// Exceptions with Promise.all
async function fetchMultiple(ids: string[]): Promise<User[]> {
    try {
        // If even one fails, the entire operation fails
        return await Promise.all(ids.map(id => fetchUser(id)));
    } catch (error) {
        // The exception from the first failed Promise
        throw new BatchFetchError("Some users could not be fetched", {
            cause: error,
        });
    }
}

// Handle individual errors with Promise.allSettled
async function fetchMultipleSafe(ids: string[]): Promise<{
    users: User[];
    errors: Array<{ id: string; error: Error }>;
}> {
    const results = await Promise.allSettled(
        ids.map(id => fetchUser(id).then(user => ({ id, user })))
    );

    const users: User[] = [];
    const errors: Array<{ id: string; error: Error }> = [];

    for (const result of results) {
        if (result.status === "fulfilled") {
            users.push(result.value.user);
        } else {
            errors.push({
                id: "unknown",  // Original id may not be available with allSettled
                error: result.reason,
            });
        }
    }

    return { users, errors };
}
```

### 9.2 Unhandled Promise Rejections

```typescript
// BAD: Unhandled rejection (UnhandledPromiseRejection)
async function dangerousCode(): Promise<void> {
    fetchUser("123");  // Forgot await!
    // If fetchUser rejects, nobody catches it
}

// BAD: Promise chain missing catch
someAsyncFunction().then(data => {
    process(data);
    // No .catch() -> rejection goes unhandled
});

// GOOD: Always attach await or .catch()
await someAsyncFunction().catch(error => {
    logger.error("Failed:", error);
});

// Global handlers (last resort)
// Node.js
process.on('unhandledRejection', (reason, promise) => {
    logger.error('Unhandled Rejection:', reason);
    Sentry.captureException(reason);
    // In Node.js 15+, the process terminates
});

// Browser
window.addEventListener('unhandledrejection', (event) => {
    logger.error('Unhandled Rejection:', event.reason);
    event.preventDefault();  // Suppress browser's default error display
});
```

### 9.3 Exception Handling Patterns in Concurrent Processing

```typescript
// Pattern 1: fail-fast (abort all if one fails)
async function failFast(tasks: Array<() => Promise<void>>): Promise<void> {
    const controller = new AbortController();

    try {
        await Promise.all(
            tasks.map(async (task) => {
                if (controller.signal.aborted) return;
                try {
                    await task();
                } catch (error) {
                    controller.abort();  // Notify other tasks of cancellation
                    throw error;
                }
            })
        );
    } catch (error) {
        throw new BatchError("One or more tasks failed", { cause: error });
    }
}

// Pattern 2: best-effort (complete as many as possible)
async function bestEffort<T>(
    tasks: Array<() => Promise<T>>
): Promise<{ results: T[]; errors: Error[] }> {
    const settled = await Promise.allSettled(tasks.map(t => t()));

    const results: T[] = [];
    const errors: Error[] = [];

    for (const result of settled) {
        if (result.status === "fulfilled") {
            results.push(result.value);
        } else {
            errors.push(result.reason instanceof Error
                ? result.reason
                : new Error(String(result.reason)));
        }
    }

    return { results, errors };
}

// Pattern 3: retry-on-failure
async function withRetry<T>(
    fn: () => Promise<T>,
    maxRetries: number = 3,
    delayMs: number = 1000,
): Promise<T> {
    let lastError: Error;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
            return await fn();
        } catch (error) {
            lastError = error instanceof Error ? error : new Error(String(error));

            if (attempt < maxRetries) {
                const delay = delayMs * Math.pow(2, attempt - 1);  // Exponential backoff
                logger.warn(`Attempt ${attempt} failed, retrying in ${delay}ms...`);
                await new Promise(resolve => setTimeout(resolve, delay));
            }
        }
    }

    throw new RetryExhaustedError(
        `All ${maxRetries} attempts failed`,
        { cause: lastError! }
    );
}
```

---

## 10. Exception Handling in Testing

### 10.1 Testing Exceptions

```typescript
// Testing exceptions with Jest
describe("UserService", () => {
    describe("getUser", () => {
        it("throws NotFoundError for a non-existent user", async () => {
            await expect(
                userService.getUser("nonexistent-id")
            ).rejects.toThrow(NotFoundError);
        });

        it("NotFoundError has correct properties set", async () => {
            try {
                await userService.getUser("user-999");
                fail("An exception should have been thrown");
            } catch (error) {
                expect(error).toBeInstanceOf(NotFoundError);
                expect((error as NotFoundError).code).toBe("NOT_FOUND");
                expect((error as NotFoundError).httpStatus).toBe(404);
                expect(error.message).toContain("user-999");
            }
        });

        it("database errors are wrapped in InternalError", async () => {
            // Mock DB failure
            jest.spyOn(db, "findById").mockRejectedValue(
                new Error("Connection refused")
            );

            await expect(
                userService.getUser("user-1")
            ).rejects.toThrow(InternalError);
        });
    });
});

// Python: Testing exceptions with pytest
# def test_get_user_not_found():
#     with pytest.raises(NotFoundError) as exc_info:
#         user_service.get_user("nonexistent")
#
#     assert exc_info.value.code == "NOT_FOUND"
#     assert "nonexistent" in str(exc_info.value)

# def test_get_user_not_found_match():
#     with pytest.raises(NotFoundError, match="nonexistent"):
#         user_service.get_user("nonexistent")
```

### 10.2 Exception Path Testing Strategy

```typescript
// Exception testing in the test pyramid

// 1. Unit tests: Cover individual error cases thoroughly
describe("validateEmail", () => {
    it.each([
        ["", "Email address is required"],
        ["invalid", "Invalid email address format"],
        ["a@b", "Invalid email address format"],
        ["@example.com", "Invalid email address format"],
    ])("throws ValidationError for '%s': %s", (email, expectedMessage) => {
        expect(() => validateEmail(email)).toThrow(ValidationError);
        try {
            validateEmail(email);
        } catch (e) {
            expect((e as ValidationError).message).toBe(expectedMessage);
        }
    });
});

// 2. Integration tests: Verify error propagation
describe("POST /api/users", () => {
    it("returns 409 Conflict for duplicate email", async () => {
        // Create an existing user
        await createUser({ email: "test@example.com" });

        const response = await request(app)
            .post("/api/users")
            .send({ email: "test@example.com", name: "Test" });

        expect(response.status).toBe(409);
        expect(response.body.error.code).toBe("EMAIL_ALREADY_EXISTS");
    });

    it("returns 400 with details for validation errors", async () => {
        const response = await request(app)
            .post("/api/users")
            .send({});  // Empty body

        expect(response.status).toBe(400);
        expect(response.body.error.code).toBe("VALIDATION_ERROR");
        expect(response.body.error.details).toEqual(
            expect.arrayContaining([
                expect.objectContaining({ field: "email" }),
                expect.objectContaining({ field: "name" }),
            ])
        );
    });
});
```

---

## 11. Best Practices in Practice

### 11.1 Logging and Monitoring

```typescript
// Structured error logging
interface ErrorLog {
    level: "warn" | "error" | "fatal";
    code: string;
    message: string;
    stack?: string;
    requestId?: string;
    userId?: string;
    path?: string;
    method?: string;
    duration?: number;
    metadata?: Record<string, unknown>;
}

function logError(error: Error, context: Partial<ErrorLog> = {}): void {
    const log: ErrorLog = {
        level: error instanceof AppError && error.isOperational ? "warn" : "error",
        code: error instanceof AppError ? error.code : "UNKNOWN_ERROR",
        message: error.message,
        stack: error.stack,
        ...context,
    };

    if (log.level === "error") {
        // Immediately alert for non-operational errors
        logger.error(log);
        Sentry.captureException(error, { extra: context });
        metrics.increment("app.errors.unexpected");
    } else {
        logger.warn(log);
        metrics.increment(`app.errors.operational.${log.code}`);
    }
}

// Error rate monitoring
class ErrorRateMonitor {
    private errors: Map<string, number[]> = new Map();

    record(code: string): void {
        const now = Date.now();
        const timestamps = this.errors.get(code) ?? [];
        timestamps.push(now);
        // Remove entries older than 5 minutes
        const fiveMinAgo = now - 5 * 60 * 1000;
        this.errors.set(code, timestamps.filter(t => t > fiveMinAgo));
    }

    getRate(code: string, windowMs: number = 60_000): number {
        const now = Date.now();
        const timestamps = this.errors.get(code) ?? [];
        return timestamps.filter(t => t > now - windowMs).length;
    }

    isAlerting(code: string, threshold: number = 10): boolean {
        return this.getRate(code) > threshold;
    }
}
```

### 11.2 Error Message Guidelines

```
Error message best practices:

  1. Clearly state what happened
     BAD: "Error"
     BAD: "Something went wrong"
     GOOD: "User user-123 not found"
     GOOD: "Invalid email address format: missing @"

  2. Indicate what should be done (for users)
     BAD: "Internal Server Error"
     GOOD: "The server is temporarily unavailable. Please wait and try again later"
     GOOD: "Your session has expired. Please log in again"

  3. Include context information (for developers)
     BAD: "Database error"
     GOOD: "Failed to insert user (email: test@example.com): unique constraint violation on 'users_email_key'"

  4. Be security-conscious
     BAD: "SQL syntax error: SELECT * FROM users WHERE password = '...'"
     BAD: "Authentication failed for admin@company.com"
     GOOD: "Authentication failed" (for users)
     GOOD: "Auth failed: invalid password for user_id=123" (for logs)

  5. Consider internationalization
     -> Error code (machine-readable) + message template
     -> Design to be usable as i18n keys
```

### 11.3 Error Handling Checklist

```
Error handling checklist at project start:

  [] Error hierarchy design
    -> AppError base class
    -> Domain error subclasses
    -> Error code system

  [] Global error handler
    -> Express/Fastify middleware
    -> React Error Boundary
    -> uncaughtException / unhandledRejection

  [] Logging and monitoring
    -> Structured logs (JSON)
    -> Error tracking (Sentry)
    -> Alert configuration (PagerDuty)

  [] API error responses
    -> Unified format (RFC 7807 compliant recommended)
    -> Proper HTTP status code usage
    -> Error code documentation

  [] Testing
    -> Both normal and error paths
    -> Error propagation path tests
    -> Edge case tests

  [] Documentation
    -> Error code reference
    -> Troubleshooting guide
    -> Error response examples
```

---

## 12. Performance and Trade-offs

### 12.1 Exceptions vs Result Type Performance

```typescript
// Performance comparison: Exceptions vs Result type

// BAD: Using exceptions in performance-critical code
function parseIntWithException(s: string): number {
    const n = Number(s);
    if (isNaN(n)) throw new Error(`Invalid number: ${s}`);
    return n;
}

// 100,000 calls (50% failure): ~500ms
// -> Stack trace construction on exception throw is expensive

// GOOD: Using Result type for performance-critical code
function parseIntWithResult(s: string): { ok: true; value: number } | { ok: false; error: string } {
    const n = Number(s);
    if (isNaN(n)) return { ok: false, error: `Invalid number: ${s}` };
    return { ok: true, value: n };
}

// 100,000 calls (50% failure): ~5ms
// -> Same cost as a normal function return

// Conclusion:
// -> Operations that frequently fail (validation, parsing) -> Result type
// -> Operations that rarely fail (I/O, network) -> Exceptions are fine
// -> The cost of exceptions is only at "throw time". try blocks themselves are free
```

### 12.2 Stack Trace Control

```typescript
// Omitting stack traces (performance optimization)
class LightweightError extends Error {
    constructor(message: string, public readonly code: string) {
        super(message);
        this.name = this.constructor.name;
        // Omit stack trace (improves performance)
        // However, use with caution as it makes debugging difficult
    }
}

// V8: Limit stack depth with Error.stackTraceLimit
Error.stackTraceLimit = 10;  // Default is 10 (Node.js)

// Conditional stack trace
class ConfigurableError extends Error {
    constructor(message: string, options?: { includeStack?: boolean }) {
        super(message);
        if (options?.includeStack === false) {
            this.stack = `${this.name}: ${this.message}`;
        }
    }
}
```

---


## FAQ

### Q1: What is the most important point when learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory, but by actually writing code and verifying how it works.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals and jumping straight to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this used in practice?

Knowledge of this topic is frequently applied in day-to-day development work. It becomes especially important during code reviews and architecture design.

---

## Summary

| Concept | Key Point |
|---------|-----------|
| try/catch | Catch and handle exceptions |
| finally | Cleanup that always executes |
| Propagation | Catch where you can handle it |
| checked vs unchecked | Modern approach: unchecked + Result type |
| Custom errors | Design a hierarchy and codify |
| Exception safety | Basic guarantee as minimum, strong guarantee recommended |
| Async exceptions | Don't forget to handle Promise rejections |
| Performance | try is free, throw is expensive |
| Testing | Cover both normal and error paths |
| Logging | Structured logs + error tracking |

---

## Recommended Next Guides

---

## References
1. Bloch, J. "Effective Java." Items 69-77, 2018.
2. Sutter, H. "When and How to Use Exceptions." 2004.
3. Abramov, D. "Error Handling in React 16." React Blog, 2017.
4. Goldberg, J. "Error Handling in Node.js." joyent.com, 2014.
5. The Rust Programming Language. "Error Handling."
6. Go Blog. "Error handling and Go." 2011.
7. Python Documentation. "Errors and Exceptions."
8. Stroustrup, B. "The C++ Programming Language." 4th Edition, 2013.
9. Apple Developer Documentation. "Error Handling." Swift Documentation.
10. Node.js Documentation. "Errors."
