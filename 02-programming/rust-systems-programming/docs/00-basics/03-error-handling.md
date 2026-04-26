# Error Handling -- Rust's Type-Safe Error Handling Patterns

> Rust has no exception mechanism. Instead, it uses explicit error handling via `Result<T, E>` and `Option<T>`, ensuring all error paths are verified at compile time.

---

## What You Will Learn in This Chapter

1. **Result and Option** -- Understand how to express failure possibilities through types and handle them safely with pattern matching
2. **The ? Operator and Error Propagation** -- Master the syntactic sugar that reduces boilerplate and the conversion mechanism behind it
3. **Custom Error Types** -- Learn how to define your own error types and automate conversions with the `From` trait
4. **thiserror / anyhow** -- Learn how to choose between the error-handling crates used in practice
5. **Practical Error Design** -- Acquire error design patterns for both libraries and applications

## Prerequisites

Reading the following before this guide will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding the contents of [Types and Traits -- The Foundation of Rust's Type System and Polymorphism](./02-types-and-traits.md)

---

## 1. Rust's Error Handling Philosophy

```
┌─────────────────────────────────────────────────────┐
│           Rust's Error Classification                │
├──────────────────┬──────────────────────────────────┤
│ Unrecoverable    │ panic!() -- Aborts the program   │
│ errors           │ Out-of-bounds array access, etc. │
├──────────────────┼──────────────────────────────────┤
│ Recoverable      │ Result<T, E> -- Caller handles   │
│ errors           │ File not found, parse errors,etc.│
├──────────────────┼──────────────────────────────────┤
│ Absence of value │ Option<T> -- None is a normal    │
│                  │ state. No search result, no      │
│                  │ config item, etc.                │
└──────────────────┴──────────────────────────────────┘
```

Rust's error handling philosophy is "express all errors through types." While many languages adopt exception mechanisms (try/catch), Rust deliberately excludes exceptions and adopts a method of returning errors as values. The advantages of this design are:

1. **Explicitness**: You can tell whether a function may produce an error simply by looking at its signature
2. **Exhaustiveness**: The compiler detects missing error handling
3. **Performance**: There is no cost for stack unwinding from exceptions
4. **Composability**: The `?` operator and combinators enable concise chaining of error handling

### 1.1 panic! and Unrecoverable Errors

```rust
fn main() {
    // Explicit panic
    // panic!("Fatal error!");

    // Implicit panic (out-of-bounds access)
    let v = vec![1, 2, 3];
    // let x = v[10]; // panic: index out of bounds

    // To enable a backtrace on panic:
    // RUST_BACKTRACE=1 cargo run

    // unwrap / expect also cause panics
    let result: Result<i32, &str> = Err("error");
    // result.unwrap();  // panic
    // result.expect("custom message");  // panic (with message)
}
```

A panic is used when an invariant of the program is broken (a bug); for ordinary error handling, you should use `Result`.

### 1.2 Panic Propagation and catch_unwind

```rust
use std::panic;

fn risky_operation() {
    panic!("Something is wrong!");
}

fn main() {
    // Catch a panic with catch_unwind (used at FFI boundaries, etc.)
    let result = panic::catch_unwind(|| {
        risky_operation();
    });

    match result {
        Ok(()) => println!("Completed normally"),
        Err(_) => println!("A panic occurred, but we recovered"),
    }

    println!("The program continues...");

    // Note: catch_unwind is not used for general error handling.
    // Its main uses are at FFI boundaries or inside thread pools.
}
```

---

## 2. Option<T>

### Example 1: The Basics of Option

```rust
fn find_user(id: u64) -> Option<String> {
    match id {
        1 => Some(String::from("Tanaka")),
        2 => Some(String::from("Suzuki")),
        _ => None,
    }
}

fn main() {
    // Pattern matching
    match find_user(1) {
        Some(name) => println!("User: {}", name),
        None => println!("Not found"),
    }

    // if let
    if let Some(name) = find_user(2) {
        println!("User: {}", name);
    }

    // let-else (Rust 2021+)
    let Some(name) = find_user(1) else {
        println!("Not found");
        return;
    };
    println!("Found: {}", name);

    // unwrap_or with a default value
    let name = find_user(99).unwrap_or(String::from("Unknown"));
    println!("User: {}", name);
}
```

### Example 2: Option Combinators

```rust
fn main() {
    let numbers = vec![1, 2, 3, 4, 5];

    // map: transforms the value inside Some
    let first_doubled: Option<i32> = numbers.first().map(|x| x * 2);
    println!("{:?}", first_doubled); // Some(2)

    // and_then: flattens nested Options (equivalent to flatMap)
    let result = Some("42")
        .and_then(|s| s.parse::<i32>().ok())
        .map(|n| n * 2);
    println!("{:?}", result); // Some(84)

    // filter: returns None if the predicate is not satisfied
    let even = Some(4).filter(|x| x % 2 == 0);
    let odd = Some(3).filter(|x| x % 2 == 0);
    println!("{:?}, {:?}", even, odd); // Some(4), None

    // unwrap_or_else: lazily evaluates the default value
    let value = None::<i32>.unwrap_or_else(|| {
        println!("Computing default value...");
        0
    });
    println!("{}", value);

    // or / or_else: returns the first Some
    let a: Option<i32> = None;
    let b: Option<i32> = Some(42);
    let c: Option<i32> = Some(100);
    println!("{:?}", a.or(b));        // Some(42)
    println!("{:?}", b.or(c));        // Some(42) -- the first Some

    // zip: combines two Options
    let x = Some(1);
    let y = Some("hello");
    let z: Option<i32> = None;
    println!("{:?}", x.zip(y));       // Some((1, "hello"))
    println!("{:?}", x.zip(z));       // None

    // flatten: Option<Option<T>> → Option<T>
    let nested: Option<Option<i32>> = Some(Some(42));
    println!("{:?}", nested.flatten()); // Some(42)

    // transpose: Option<Result<T, E>> ↔ Result<Option<T>, E>
    let opt_result: Option<Result<i32, String>> = Some(Ok(42));
    let result_opt: Result<Option<i32>, String> = opt_result.transpose();
    println!("{:?}", result_opt); // Ok(Some(42))
}
```

### Option Chaining Patterns

```rust
#[derive(Debug)]
struct Config {
    database: Option<DatabaseConfig>,
}

#[derive(Debug)]
struct DatabaseConfig {
    host: Option<String>,
    port: Option<u16>,
}

fn get_db_url(config: &Config) -> Option<String> {
    // Option chain: at any stage, if it is None, return None early
    let db = config.database.as_ref()?;
    let host = db.host.as_ref()?;
    let port = db.port?;
    Some(format!("postgres://{}:{}/mydb", host, port))
}

fn main() {
    let config = Config {
        database: Some(DatabaseConfig {
            host: Some("localhost".to_string()),
            port: Some(5432),
        }),
    };

    match get_db_url(&config) {
        Some(url) => println!("DB URL: {}", url),
        None => println!("Database configuration is incomplete"),
    }

    // Case where host is None
    let incomplete_config = Config {
        database: Some(DatabaseConfig {
            host: None,
            port: Some(5432),
        }),
    };
    println!("Incomplete: {:?}", get_db_url(&incomplete_config)); // None
}
```

---

## 3. Result<T, E>

### Example 3: The Basics of Result

```rust
use std::fs;
use std::io;

fn read_username() -> Result<String, io::Error> {
    let content = fs::read_to_string("username.txt")?;
    Ok(content.trim().to_string())
}

fn main() {
    match read_username() {
        Ok(name) => println!("Username: {}", name),
        Err(e) => println!("Error: {}", e),
    }
}
```

### Example 4: Error Propagation with the ? Operator

```rust
use std::fs::File;
use std::io::{self, Read};

// Without the ? operator (verbose)
fn read_file_verbose(path: &str) -> Result<String, io::Error> {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) => return Err(e),
    };
    let mut buf = String::new();
    match file.read_to_string(&mut buf) {
        Ok(_) => Ok(buf),
        Err(e) => Err(e),
    }
}

// With the ? operator (concise)
fn read_file_concise(path: &str) -> Result<String, io::Error> {
    let mut file = File::open(path)?;
    let mut buf = String::new();
    file.read_to_string(&mut buf)?;
    Ok(buf)
}

// Even more concise
fn read_file_short(path: &str) -> Result<String, io::Error> {
    std::fs::read_to_string(path)
}
```

### Behavior Flow of the ? Operator

```
         read_file_concise()
              │
    File::open(path)?
              │
         ┌────┴────┐
         │         │
      Ok(file)  Err(e)
         │         │
         │    return Err(e)  ← early return
         │
    file.read_to_string(&mut buf)?
              │
         ┌────┴────┐
         │         │
      Ok(n)    Err(e)
         │         │
    Ok(buf)   return Err(e)
```

### Example 5: Result Combinators

```rust
use std::num::ParseIntError;

fn parse_and_double(s: &str) -> Result<i32, ParseIntError> {
    s.parse::<i32>().map(|n| n * 2)
}

fn main() {
    // map: transforms the value inside Ok
    let result = "21".parse::<i32>().map(|n| n * 2);
    println!("{:?}", result); // Ok(42)

    // map_err: transforms the value inside Err
    let result = "abc".parse::<i32>()
        .map_err(|e| format!("Parse error: {}", e));
    println!("{:?}", result); // Err("Parse error: ...")

    // and_then: chaining Results
    let result = "42".parse::<i32>()
        .and_then(|n| {
            if n > 0 {
                Ok(n)
            } else {
                Err("0 or below".parse::<i32>().unwrap_err())
            }
        });
    println!("{:?}", result);

    // unwrap_or / unwrap_or_else
    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8080);
    println!("Port: {}", port);

    // collecting multiple Results
    let strings = vec!["1", "2", "3", "4", "5"];
    let numbers: Result<Vec<i32>, _> = strings
        .iter()
        .map(|s| s.parse::<i32>())
        .collect();
    println!("{:?}", numbers); // Ok([1, 2, 3, 4, 5])

    // When an error is included
    let mixed = vec!["1", "abc", "3"];
    let result: Result<Vec<i32>, _> = mixed
        .iter()
        .map(|s| s.parse::<i32>())
        .collect();
    println!("{:?}", result); // Err(ParseIntError)
}
```

### Example 6: Handling Multiple Error Types

```rust
use std::io;
use std::num::ParseIntError;

// Approach 1: Box<dyn Error>
fn process_file_boxed(path: &str) -> Result<i32, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;  // io::Error
    let number: i32 = content.trim().parse()?;       // ParseIntError
    Ok(number * 2)
}

// Approach 2: Custom error enum
#[derive(Debug)]
enum ProcessError {
    Io(io::Error),
    Parse(ParseIntError),
}

impl std::fmt::Display for ProcessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProcessError::Io(e) => write!(f, "IO error: {}", e),
            ProcessError::Parse(e) => write!(f, "Parse error: {}", e),
        }
    }
}

impl std::error::Error for ProcessError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ProcessError::Io(e) => Some(e),
            ProcessError::Parse(e) => Some(e),
        }
    }
}

impl From<io::Error> for ProcessError {
    fn from(e: io::Error) -> Self {
        ProcessError::Io(e)
    }
}

impl From<ParseIntError> for ProcessError {
    fn from(e: ParseIntError) -> Self {
        ProcessError::Parse(e)
    }
}

fn process_file(path: &str) -> Result<i32, ProcessError> {
    let content = std::fs::read_to_string(path)?; // io::Error → ProcessError
    let number: i32 = content.trim().parse()?;      // ParseIntError → ProcessError
    Ok(number * 2)
}

fn main() {
    match process_file("number.txt") {
        Ok(n) => println!("Result: {}", n),
        Err(ProcessError::Io(e)) => eprintln!("File error: {}", e),
        Err(ProcessError::Parse(e)) => eprintln!("Number conversion error: {}", e),
    }
}
```

---

## 4. Custom Error Types

### Example 7: Defining a Custom Error Manually

```rust
use std::fmt;
use std::num::ParseIntError;

#[derive(Debug)]
enum AppError {
    IoError(std::io::Error),
    ParseError(ParseIntError),
    ValidationError(String),
    NotFoundError { resource: String, id: u64 },
    AuthError { user: String, reason: String },
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AppError::IoError(e) => write!(f, "IO error: {}", e),
            AppError::ParseError(e) => write!(f, "Parse error: {}", e),
            AppError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            AppError::NotFoundError { resource, id } => {
                write!(f, "{} (ID={}) was not found", resource, id)
            }
            AppError::AuthError { user, reason } => {
                write!(f, "Authentication error (user: {}): {}", user, reason)
            }
        }
    }
}

impl std::error::Error for AppError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            AppError::IoError(e) => Some(e),
            AppError::ParseError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for AppError {
    fn from(e: std::io::Error) -> Self {
        AppError::IoError(e)
    }
}

impl From<ParseIntError> for AppError {
    fn from(e: ParseIntError) -> Self {
        AppError::ParseError(e)
    }
}

fn load_config(path: &str) -> Result<u32, AppError> {
    let content = std::fs::read_to_string(path)?; // automatically converts to IoError
    let port: u32 = content.trim().parse()?;       // automatically converts to ParseError
    if port < 1024 {
        return Err(AppError::ValidationError(
            format!("Port {} is reserved", port),
        ));
    }
    Ok(port)
}

fn find_user(id: u64) -> Result<String, AppError> {
    if id == 0 {
        return Err(AppError::NotFoundError {
            resource: "User".to_string(),
            id,
        });
    }
    Ok(format!("user_{}", id))
}
```

---

## 5. thiserror and anyhow

### Example 8: thiserror (for libraries)

```rust
use thiserror::Error;

#[derive(Debug, Error)]
enum DatabaseError {
    #[error("Connection error: {0}")]
    ConnectionFailed(String),

    #[error("Query error: {query}")]
    QueryFailed {
        query: String,
        #[source]
        source: std::io::Error,
    },

    #[error("Record not found: ID={id}")]
    NotFound { id: u64 },

    #[error("Authentication error: access denied for user '{user}'")]
    AuthFailed { user: String },

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Parse(#[from] std::num::ParseIntError),
}

// Advantages of thiserror:
// 1. Automatic implementation of Display (via the #[error("...")] attribute)
// 2. Automatic implementation of From (via the #[from] attribute)
// 3. Automatic implementation of source() (via the #[source] attribute)
// 4. Significant reduction of boilerplate

fn connect_db(url: &str) -> Result<(), DatabaseError> {
    if url.is_empty() {
        return Err(DatabaseError::ConnectionFailed(
            "URL is empty".to_string(),
        ));
    }
    Ok(())
}

fn find_record(id: u64) -> Result<String, DatabaseError> {
    if id == 0 {
        return Err(DatabaseError::NotFound { id });
    }
    Ok(format!("record_{}", id))
}
```

### Example 9: anyhow (for applications)

```rust
use anyhow::{Context, Result, bail, ensure, anyhow};

#[derive(Debug)]
struct Config {
    host: String,
    port: u16,
    database: String,
}

fn load_config(path: &str) -> Result<Config> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Cannot read configuration file '{}'", path))?;

    let lines: Vec<&str> = content.lines().collect();

    ensure!(lines.len() >= 3, "The configuration file requires at least 3 lines");

    let host = lines[0].trim().to_string();
    let port: u16 = lines[1].trim().parse()
        .context("Failed to parse the port number")?;
    let database = lines[2].trim().to_string();

    if host.is_empty() {
        bail!("Host is not specified");
    }

    if port == 0 {
        return Err(anyhow!("Port 0 is invalid"));
    }

    Ok(Config { host, port, database })
}

fn run_server(config: &Config) -> Result<()> {
    println!("Starting server: {}:{}/{}", config.host, config.port, config.database);
    Ok(())
}

fn main() {
    match load_config("config.txt") {
        Ok(config) => {
            if let Err(e) = run_server(&config) {
                // Print the entire error chain
                eprintln!("Error: {:#}", e);
                // Print each cause in the error chain individually
                for cause in e.chain() {
                    eprintln!("  Cause: {}", cause);
                }
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("Configuration loading error: {:#}", e);
            std::process::exit(1);
        }
    }
}
```

### Example 10: Combining anyhow and thiserror

```rust
// Library layer: define concrete error types with thiserror
mod db {
    use thiserror::Error;

    #[derive(Debug, Error)]
    pub enum DbError {
        #[error("Connection error: {0}")]
        Connection(String),
        #[error("Query error: {0}")]
        Query(String),
        #[error("Record not found: {0}")]
        NotFound(u64),
    }

    pub fn find_user(id: u64) -> Result<String, DbError> {
        match id {
            0 => Err(DbError::NotFound(id)),
            _ => Ok(format!("user_{}", id)),
        }
    }
}

// Application layer: aggregate errors with anyhow
mod app {
    use anyhow::{Context, Result};
    use super::db;

    pub fn get_user_name(id: u64) -> Result<String> {
        let user = db::find_user(id)
            .with_context(|| format!("Failed to retrieve user with ID {}", id))?;
        Ok(user)
    }
}

fn main() {
    match app::get_user_name(0) {
        Ok(name) => println!("User: {}", name),
        Err(e) => {
            eprintln!("Error: {:#}", e);
            // The anyhow error chain is displayed:
            // Error: Failed to retrieve user with ID 0: Record not found: 0

            // Use downcast to obtain the concrete error type
            if let Some(db_err) = e.downcast_ref::<db::DbError>() {
                match db_err {
                    db::DbError::NotFound(id) => {
                        eprintln!("Hint: ID {} does not exist", id);
                    }
                    _ => {}
                }
            }
        }
    }
}
```

---

## 6. Diagram of the Error Chain

```
┌────────────────────────────────────────┐
│ anyhow::Error                          │
│ "Cannot read configuration file        │
│  'config.toml'"                        │
│                                        │
│  Caused by:                            │
│  ┌──────────────────────────────────┐  │
│  │ std::io::Error                   │  │
│  │ kind: NotFound                   │  │
│  │ "No such file or directory"      │  │
│  └──────────────────────────────────┘  │
└────────────────────────────────────────┘

When you add context with .with_context(),
you can traverse the error chain hierarchically.
```

```
┌────────────────────────────────────────────┐
│  Decision Flowchart                        │
│                                            │
│  Library development?                      │
│    ├── Yes → thiserror with concrete       │
│    │         error types                   │
│    │         (users can match on them)     │
│    └── No                                  │
│         Application development?           │
│           ├── Yes → anyhow                 │
│           │         (focus on error chains)│
│           └── Prototype → anyhow           │
│                                            │
│  Hybrid approach:                          │
│    Library layer → thiserror               │
│    App layer → anyhow (wraps thiserror)    │
└────────────────────────────────────────────┘
```

---

## 7. Practical Error Handling Patterns

### 7.1 Error Conversion Within a Function

```rust
use std::io;
use std::num::ParseIntError;

// Various patterns of error conversion
fn demo_error_conversion() -> Result<(), Box<dyn std::error::Error>> {
    // map_err: convert the error type
    let _port: u16 = "8080".parse()
        .map_err(|e: ParseIntError| io::Error::new(io::ErrorKind::InvalidData, e))?;

    // Automatic conversion via the From trait (used by the ? operator)
    // ? converts Err(e) into Err(From::from(e))

    // ok_or: Option → Result conversion
    let env_var = std::env::var("HOME").ok();
    let home = env_var.ok_or_else(|| io::Error::new(
        io::ErrorKind::NotFound,
        "The HOME environment variable is not set"
    ))?;
    println!("HOME: {}", home);

    Ok(())
}
```

### 7.2 Logging and Recovering from Errors

```rust
fn process_items(items: &[&str]) -> Vec<i32> {
    items.iter()
        .filter_map(|item| {
            match item.parse::<i32>() {
                Ok(n) => Some(n),
                Err(e) => {
                    eprintln!("Warning: cannot parse '{}': {}", item, e);
                    None  // skip the error and continue
                }
            }
        })
        .collect()
}

fn process_with_defaults(items: &[&str]) -> Vec<i32> {
    items.iter()
        .map(|item| {
            item.parse::<i32>().unwrap_or_else(|_| {
                eprintln!("Replacing '{}' with the default value 0", item);
                0
            })
        })
        .collect()
}

fn main() {
    let items = vec!["1", "abc", "3", "def", "5"];

    let filtered = process_items(&items);
    println!("Filtered result: {:?}", filtered); // [1, 3, 5]

    let defaulted = process_with_defaults(&items);
    println!("Default result: {:?}", defaulted); // [1, 0, 3, 0, 5]
}
```

### 7.3 Retry Pattern

```rust
use std::time::Duration;
use std::thread;

fn unreliable_operation() -> Result<String, String> {
    use std::time::SystemTime;
    let secs = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();

    if secs % 3 == 0 {
        Ok("Success!".to_string())
    } else {
        Err("Transient error".to_string())
    }
}

fn retry<F, T, E>(mut operation: F, max_retries: u32, delay: Duration) -> Result<T, E>
where
    F: FnMut() -> Result<T, E>,
    E: std::fmt::Display,
{
    let mut last_err = None;
    for attempt in 1..=max_retries {
        match operation() {
            Ok(value) => return Ok(value),
            Err(e) => {
                eprintln!("Attempt {}/{}: {}", attempt, max_retries, e);
                last_err = Some(e);
                if attempt < max_retries {
                    thread::sleep(delay);
                }
            }
        }
    }
    Err(last_err.unwrap())
}

fn main() {
    match retry(unreliable_operation, 5, Duration::from_millis(100)) {
        Ok(result) => println!("Result: {}", result),
        Err(e) => eprintln!("All attempts failed: {}", e),
    }
}
```

### 7.4 Aggregating Errors

```rust
fn validate_user_input(
    name: &str,
    email: &str,
    age: &str,
) -> Result<(String, String, u32), Vec<String>> {
    let mut errors = Vec::new();

    if name.is_empty() {
        errors.push("Name is required".to_string());
    }

    if !email.contains('@') {
        errors.push("Invalid email address format".to_string());
    }

    let age_result = age.parse::<u32>();
    if age_result.is_err() {
        errors.push("Age must be a number".to_string());
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    Ok((
        name.to_string(),
        email.to_string(),
        age_result.unwrap(),
    ))
}

fn main() {
    match validate_user_input("", "invalid-email", "abc") {
        Ok((name, email, age)) => {
            println!("Valid: {} / {} / age {}", name, email, age);
        }
        Err(errors) => {
            eprintln!("Input errors:");
            for error in &errors {
                eprintln!("  - {}", error);
            }
        }
    }
}
```

---

## 8. Comparison Tables

### 8.1 Comparison of Error Handling Techniques

| Technique | Use Case | Advantages | Disadvantages |
|------|------|------|------|
| `match` | Handling individual patterns | Exhaustive, safe | Verbose |
| `?` | Error propagation | Concise | Requires error conversion |
| `unwrap()` | Tests/prototypes | Short | Dangerous in production |
| `expect("msg")` | Tests/invariants | Has a message | Dangerous in production |
| `unwrap_or(v)` | Default values | Safe, concise | Always computes the default |
| `unwrap_or_else(f)` | Lazy default | Safe, efficient | Slightly verbose |
| `unwrap_or_default()` | Types implementing Default | Very concise | Requires Default |
| `if let` | Specific pattern only | Concise | No Err/None handling |
| `let-else` | Early return | Readable | Rust 2021+ |
| `map` / `and_then` | Conversion chain | Functional style | Takes getting used to |

### 8.2 thiserror vs anyhow

| Characteristic | thiserror | anyhow |
|------|-----------|--------|
| Purpose | Defining error types for libraries | Error handling for applications |
| Error type | Concrete enum | anyhow::Error (type-erased) |
| Pattern matching | Possible | Requires downcast |
| Error chain | Manually implement source | Automatic (context) |
| From implementation | Automatic via #[from] | Implicit type conversion |
| Recommended for | Public APIs, libraries | Binaries, CLIs, servers |
| Code size | Slightly larger | Smaller |
| Number of dependencies | Few (proc-macro) | Few |

### 8.3 Error Type Selection Guide

| Scenario | Recommended Error Type | Rationale |
|------|-------------|------|
| Public library API | thiserror enum | Allows users to pattern-match |
| CLI app | anyhow::Error | Error messages are important |
| Web server | thiserror + anyhow | Mapping to response codes |
| Prototype | anyhow / Box<dyn Error> | Rapid development |
| Internal modules | thiserror enum | Type-safe error handling |
| Test code | unwrap / expect | Stack trace on failure |

---

## 9. Anti-patterns

### Anti-pattern 1: Overusing unwrap

```rust
// BAD: using unwrap in production code
fn get_port() -> u16 {
    std::env::var("PORT").unwrap().parse().unwrap()
}

// GOOD: appropriate error handling
fn get_port_good() -> Result<u16, anyhow::Error> {
    let port = std::env::var("PORT")
        .context("The PORT environment variable is not set")?
        .parse()
        .context("The value of PORT is not a number")?;
    Ok(port)
}
```

### Anti-pattern 2: Swallowing Errors

```rust
// BAD: ignoring the error
fn save_data(data: &str) {
    let _ = std::fs::write("data.txt", data);  // The error is discarded!
}

// GOOD: propagate the error appropriately
fn save_data_good(data: &str) -> Result<(), std::io::Error> {
    std::fs::write("data.txt", data)?;
    Ok(())
}

// GOOD: log the error explicitly and continue
fn save_data_with_logging(data: &str) {
    if let Err(e) = std::fs::write("data.txt", data) {
        eprintln!("Warning: failed to save data: {}", e);
        // Continue if not critical
    }
}
```

### Anti-pattern 3: Overly Broad Error Types

```rust
// BAD: casually using Box<dyn Error> (loses type information)
fn do_something() -> Result<(), Box<dyn std::error::Error>> {
    // The caller cannot tell what kinds of errors are returned
    Ok(())
}

// GOOD: define a concrete error type
#[derive(Debug, thiserror::Error)]
enum MyError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("parse error: {0}")]
    Parse(#[from] std::num::ParseIntError),
}

fn do_something_good() -> Result<(), MyError> {
    Ok(())
}
```

### Anti-pattern 4: Using panic! for Error Handling

```rust
// BAD: "error handling" via panic
fn parse_config(s: &str) -> Config {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 2 {
        panic!("Invalid configuration format");  // panic is for detecting bugs!
    }
    Config {
        key: parts[0].to_string(),
        value: parts[1].to_string(),
    }
}

// GOOD: return a Result
fn parse_config_good(s: &str) -> Result<Config, String> {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 2 {
        return Err(format!("Invalid configuration format: '{}'", s));
    }
    Ok(Config {
        key: parts[0].to_string(),
        value: parts[1].to_string(),
    })
}

struct Config {
    key: String,
    value: String,
}
```

### Anti-pattern 5: Insufficiently Informative Error Messages

```rust
// BAD: vague error message
fn load_user(id: u64) -> Result<String, String> {
    Err("error".to_string())  // What error? Where?
}

// GOOD: error message with context
fn load_user_good(id: u64) -> Result<String, anyhow::Error> {
    let path = format!("/data/users/{}.json", id);
    let content = std::fs::read_to_string(&path)
        .with_context(|| format!("Cannot read file '{}' for user ID {}", path, id))?;
    let user: serde_json::Value = serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse JSON for user ID {}", id))?;
    Ok(user["name"].as_str().unwrap_or("Unknown").to_string())
}
```


---

## Hands-on Exercises

### Exercise 1: Basic Implementation

Implement code that satisfies the following requirements.

**Requirements:**
- Validate the input data
- Implement appropriate error handling
- Also write test code

```python
# Exercise 1: Template for the basic implementation
class Exercise1:
    """Practice for basic implementation patterns"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """Validate the input value"""
        if value is None:
            raise ValueError("The input value is None")
        return True

    def process(self, value):
        """Main logic for data processing"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """Retrieve the processing results"""
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
        assert False, "An exception should be raised"
    except ValueError:
        pass

    print("All tests passed!")

test_exercise1()
```

### Exercise 2: Advanced Patterns

Extend the basic implementation by adding the following functionality.

```python
# Exercise 2: Advanced patterns
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """Practice for advanced patterns"""

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
        """Look up by key"""
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
    assert ex.add("d", 4) == False  # size limit
    assert ex.find("b")['value'] == 2
    assert ex.remove("b") == True
    assert ex.find("b") is None
    stats = ex.stats()
    assert stats['total_items'] == 2
    print("All advanced tests passed!")

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

    print(f"Inefficient version: {slow_time:.4f} sec")
    print(f"Efficient version:   {fast_time:.6f} sec")
    print(f"Speedup factor:     {slow_time/fast_time:.0f}x")

benchmark()
```

**Key points:**
- Be conscious of the algorithm's computational complexity
- Choose appropriate data structures
- Measure the effect with a benchmark
---

## 10. FAQ

### Q1: When should I use panic!?

**A:** Only in the following cases:
- **When an invariant of the program is broken** (a bug)
- **In test code** (assert!, unwrap)
- **In the prototyping stage** (replace with proper error handling later)
- **For unrecoverable initialization errors** (only at the very start of main)
- **assert! / debug_assert!** for contract programming

In production business logic, you should always use `Result` as a rule.

### Q2: Can the `?` operator be used in the main function?

**A:** Yes. You can use it by setting the return type of `main` to `Result`:

```rust
fn main() -> Result<(), anyhow::Error> {
    let config = load_config("config.toml")?;
    run_server(config)?;
    Ok(())
}
```

For finer control over the exit code, you can use `std::process::ExitCode`:

```rust
fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {:#}", e);
            std::process::ExitCode::FAILURE
        }
    }
}

fn run() -> anyhow::Result<()> {
    // ? can be used here
    Ok(())
}
```

### Q3: How do I convert between Option and Result?

**A:**
```rust
// Option → Result
let opt: Option<i32> = Some(42);
let res: Result<i32, &str> = opt.ok_or("Value is missing");

// Option → Result (lazy evaluation)
let res2: Result<i32, String> = opt.ok_or_else(|| format!("Value not found"));

// Result → Option (Ok → Some, Err → None)
let res: Result<i32, String> = Ok(42);
let opt: Option<i32> = res.ok();  // Err becomes None

// Result → Option (Ok → None, Err → Some)
let res: Result<i32, String> = Err("error".to_string());
let opt: Option<String> = res.err();  // Ok becomes None
```

### Q4: How should I choose between `expect` and `unwrap`?

**A:** `expect` is a strict superset of `unwrap`. Because it can display a custom message on panic, debugging becomes easier. Even at the prototyping stage, using `expect` is recommended.

```rust
// unwrap: a message like "called `Result::unwrap()` on an `Err` value: ..."
let file = File::open("config.toml").unwrap();

// expect: a custom message explaining why this operation should succeed
let file = File::open("config.toml")
    .expect("config.toml should exist at the project root");
```

### Q5: What is the difference between `Box<dyn Error>` and `anyhow::Error`?

**A:**
- `Box<dyn Error>`: A type-erased error type usable with only the standard library. Minimal functionality.
- `anyhow::Error`: Provides rich features such as error chains via `context()`, restoration of the original type via `downcast()`, and detailed display via `{:#}`.

In practice, `anyhow::Error` is overwhelmingly more convenient. However, do not use it for the public API of a library.

### Q6: What are the best practices for designing an error type?

**A:**
1. **Library**: Define an enum with `thiserror`. Allow users to branch with `match`.
2. **Application**: Leverage `context` with `anyhow`. Place importance on error message quality.
3. **Error messages**: Include "what happened," "what was being attempted," and "how to address it."
4. **Granularity of errors**: Split into variants only when the caller needs to handle them differently.

---

## 11. Details of the std::error::Error Trait

### 11.1 Definition of the Error Trait

```rust
// Definition of std::error::Error (simplified)
pub trait Error: Debug + Display {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }
}
```

The `Error` trait requires both `Debug` and `Display` as supertraits. As a result, error types can always be displayed in both a human-readable form (`Display`) and a developer-oriented detailed form (`Debug`).

### 11.2 Traversing the Error Chain

```rust
use std::error::Error;
use std::fmt;

// Helper function to display the entire error chain
fn print_error_chain(err: &dyn Error) {
    eprintln!("Error: {}", err);
    let mut current = err.source();
    let mut depth = 1;
    while let Some(cause) = current {
        eprintln!("  {}. Cause: {}", depth, cause);
        current = cause.source();
        depth += 1;
    }
}

// Build a chain with a custom error type
#[derive(Debug)]
struct ServiceError {
    message: String,
    source: Option<Box<dyn Error + Send + Sync>>,
}

impl fmt::Display for ServiceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Service error: {}", self.message)
    }
}

impl Error for ServiceError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source.as_ref().map(|e| e.as_ref() as &(dyn Error + 'static))
    }
}

impl ServiceError {
    fn new(message: impl Into<String>) -> Self {
        Self { message: message.into(), source: None }
    }

    fn with_source(message: impl Into<String>, source: impl Error + Send + Sync + 'static) -> Self {
        Self {
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }
}

fn connect_database() -> Result<(), ServiceError> {
    let io_err = std::io::Error::new(
        std::io::ErrorKind::ConnectionRefused,
        "Connection to port 5432 was refused"
    );
    Err(ServiceError::with_source("Failed to connect to the database", io_err))
}

fn main() {
    if let Err(e) = connect_database() {
        print_error_chain(&e);
        // Output:
        // Error: Service error: Failed to connect to the database
        //   1. Cause: Connection to port 5432 was refused
    }
}
```

### 11.3 Send + Sync and Error Types

To handle errors safely in a multithreaded environment, the `Send + Sync` bounds are important.

```rust
use std::error::Error;

// Thread-safe error type
type BoxError = Box<dyn Error + Send + Sync + 'static>;

// Example of sending errors between threads
fn spawn_worker() -> Result<String, BoxError> {
    let handle = std::thread::spawn(|| -> Result<String, BoxError> {
        let content = std::fs::read_to_string("data.txt")?;
        let number: i32 = content.trim().parse()?;
        Ok(format!("Result: {}", number * 2))
    });

    match handle.join() {
        Ok(result) => result,
        Err(_) => Err("Worker thread panicked".into()),
    }
}

fn main() {
    match spawn_worker() {
        Ok(value) => println!("{}", value),
        Err(e) => eprintln!("Worker error: {}", e),
    }
}
```

### 11.4 Restoring the Error Type via Downcasting

```rust
use std::error::Error;

fn might_fail() -> Result<(), Box<dyn Error>> {
    let result: Result<i32, _> = "abc".parse();
    result?;
    Ok(())
}

fn main() {
    if let Err(e) = might_fail() {
        // Downcast: Box<dyn Error> → concrete type
        if let Some(parse_err) = e.downcast_ref::<std::num::ParseIntError>() {
            eprintln!("Detected parse error: {}", parse_err);
        } else if let Some(io_err) = e.downcast_ref::<std::io::Error>() {
            eprintln!("Detected IO error: {}", io_err);
        } else {
            eprintln!("Unknown error: {}", e);
        }

        // It is also possible to take ownership via downcast
        // let concrete: Box<std::num::ParseIntError> = e.downcast().unwrap();
    }
}
```

---

## 12. Practical Error-Handling Design Patterns

### 12.1 Error Design in a Layered Architecture

```rust
// === Infrastructure layer ===
mod infra {
    use thiserror::Error;

    #[derive(Debug, Error)]
    pub enum InfraError {
        #[error("DB connection error: {0}")]
        Database(String),
        #[error("Network error: {0}")]
        Network(String),
        #[error("Filesystem error: {0}")]
        FileSystem(#[from] std::io::Error),
    }
}

// === Domain layer ===
mod domain {
    use thiserror::Error;

    #[derive(Debug, Error)]
    pub enum DomainError {
        #[error("User not found: ID={0}")]
        UserNotFound(u64),
        #[error("Insufficient balance: required={required}, current={current}")]
        InsufficientBalance { required: u64, current: u64 },
        #[error("Invalid operation: {0}")]
        InvalidOperation(String),
        #[error("Infrastructure error")]
        Infrastructure(#[from] super::infra::InfraError),
    }
}

// === Application layer ===
mod application {
    use anyhow::{Context, Result};
    use super::domain::DomainError;

    pub fn transfer_money(from: u64, to: u64, amount: u64) -> Result<()> {
        // Domain errors are wrapped by anyhow and given context
        let _from_user = find_user(from)
            .with_context(|| format!("Failed to retrieve sender user {}", from))?;
        let _to_user = find_user(to)
            .with_context(|| format!("Failed to retrieve receiver user {}", to))?;

        // Domain validation
        validate_transfer(amount)
            .context("Transfer validation failed")?;

        println!("Transfer succeeded: {} → {} ({} yen)", from, to, amount);
        Ok(())
    }

    fn find_user(id: u64) -> Result<String, DomainError> {
        if id == 0 {
            Err(DomainError::UserNotFound(id))
        } else {
            Ok(format!("user_{}", id))
        }
    }

    fn validate_transfer(amount: u64) -> Result<(), DomainError> {
        if amount == 0 {
            Err(DomainError::InvalidOperation("Transfer amount must be greater than 0".into()))
        } else {
            Ok(())
        }
    }
}
```

### 12.2 Error Mapping in an HTTP API

```rust
use thiserror::Error;

#[derive(Debug, Error)]
enum ApiError {
    #[error("Resource not found: {0}")]
    NotFound(String),
    #[error("Authentication error: {0}")]
    Unauthorized(String),
    #[error("Validation error: {0}")]
    BadRequest(String),
    #[error("Internal error")]
    Internal(#[source] anyhow::Error),
}

impl ApiError {
    fn status_code(&self) -> u16 {
        match self {
            ApiError::NotFound(_) => 404,
            ApiError::Unauthorized(_) => 401,
            ApiError::BadRequest(_) => 400,
            ApiError::Internal(_) => 500,
        }
    }

    fn to_json(&self) -> String {
        format!(
            r#"{{"error": {{"code": {}, "message": "{}"}}}}"#,
            self.status_code(),
            self
        )
    }
}

fn handle_request(path: &str) -> Result<String, ApiError> {
    match path {
        "/users/1" => Ok(r#"{"id": 1, "name": "Tanaka"}"#.to_string()),
        "/users/0" => Err(ApiError::NotFound("User ID 0".to_string())),
        "/admin" => Err(ApiError::Unauthorized("Administrator privileges required".to_string())),
        _ => Err(ApiError::NotFound(format!("path '{}'", path))),
    }
}

fn main() {
    let paths = vec!["/users/1", "/users/0", "/admin", "/unknown"];
    for path in paths {
        match handle_request(path) {
            Ok(body) => println!("200 OK: {}", body),
            Err(e) => println!("{} Error: {}", e.status_code(), e.to_json()),
        }
    }
}
```

---


## FAQ

### Q1: What is the most important point when learning this topic?

Gaining hands-on experience is the most important. Beyond theory, writing code and verifying its behavior deepens understanding.

### Q2: What mistakes do beginners commonly make?

Skipping the basics and jumping into advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this used in practice?

Knowledge of this topic is frequently applied in everyday development work. It becomes especially important during code reviews and architectural design.

---

## 13. Summary

| Concept | Key Point |
|------|------|
| Option<T> | Express the presence/absence of a value through types. None is a normal absence. |
| Result<T, E> | Express success/failure through types. Make all error paths explicit. |
| ? operator | Syntactic sugar for early-returning Err/None. |
| From trait | The mechanism for automatic error type conversion in ?. |
| thiserror | For libraries. Derive macro for custom error types. |
| anyhow | For applications. Error chains and context. |
| panic! | For unrecoverable bugs only. Do not use it in production logic. |
| Combinators | Functional error handling with map/and_then/or_else. |
| Error aggregation | Use Vec<Error> to report multiple errors at once. |
| Retry | Pattern for retrying operations that return Result. |
| Error trait | Debug + Display. Traverse the chain via source(). |
| Send + Sync | Required for transferring errors in multithreaded code. |
| Downcast | Restore the concrete type from Box<dyn Error>. |
| Layered design | Convert errors stepwise across infrastructure → domain → application. |

---

## Recommended Next Reads

- [04-collections-iterators.md](04-collections-iterators.md) -- Collections and iterators
- [../01-advanced/02-closures-fn-traits.md](../01-advanced/02-closures-fn-traits.md) -- Closures and Fn traits
- [../04-ecosystem/04-best-practices.md](../04-ecosystem/04-best-practices.md) -- Best practices for error design

---

## 14. References

1. **The Rust Programming Language - Ch.9 Error Handling** -- https://doc.rust-lang.org/book/ch09-00-error-handling.html
2. **thiserror Documentation** -- https://docs.rs/thiserror/
3. **anyhow Documentation** -- https://docs.rs/anyhow/
4. **Rust Error Handling Best Practices (Andrew Gallant)** -- https://blog.burntsushi.net/rust-error-handling/
5. **The Rust API Guidelines - Error Handling** -- https://rust-lang.github.io/api-guidelines/interoperability.html
6. **Error Handling in Rust (Nick Cameron)** -- https://www.ncameron.org/blog/error-handling-in-rust/

---

## References

- [MDN Web Docs](https://developer.mozilla.org/) - Reference for web technologies
- [Wikipedia](https://ja.wikipedia.org/) - Overview of technical concepts
