# Go Best Practices Guide

> Guidelines for writing Go code that combines maintainability, readability, and performance, based on the spirit of Effective Go

## What You Will Learn in This Chapter

1. **The core of Effective Go** -- design principles and naming conventions for idiomatic Go code
2. **Error handling** -- designing error values, wrapping, and when to use sentinel errors
3. **Interface design** -- small interfaces, implicit implementation, dependency injection
4. **Concurrency patterns** -- goroutine management, channel design, context propagation
5. **Struct design** -- leveraging zero values, functional options, constructor patterns
6. **Package design** -- dependencies, avoiding circular dependencies, internal packages
7. **Testability** -- designing code that is easy to test
8. **Performance** -- memory efficiency, allocation optimization


## Prerequisites

Your understanding will deepen if you have the following knowledge before reading this guide:

- Basic programming knowledge
- Understanding of related fundamental concepts
- Understanding of the [Go Deployment Guide](./03-deployment.md)

---

## 1. Principles of Idiomatic Go Code

### Go's Design Philosophy

```
+-------------------------------------------------------+
|              Go's Design Philosophy                    |
+-------------------------------------------------------+
|                                                       |
|  +-------------+  +-----------+  +------------------+ |
|  | Simplicity  |  | Readability|  | Composition      | |
|  | Simplicity  |  | Readability|  | over inheritance | |
|  +-------------+  +-----------+  +------------------+ |
|                                                       |
|  +-------------+  +-----------+  +------------------+ |
|  | Explicit    |  | Minimal   |  | Convention       | |
|  | Explicit    |  | Minimal   |  | Convention-first | |
|  +-------------+  +-----------+  +------------------+ |
|                                                       |
|  +-------------+  +-----------+  +------------------+ |
|  | Concurrency |  | Orthogonal|  | Practical        | |
|  | Concurrency |  | Orthogonal|  | Practical        | |
|  +-------------+  +-----------+  +------------------+ |
|                                                       |
|  "Clear is better than clever"                        |
|  "Clarity over cleverness"                            |
|                                                       |
|  "A little copying is better than a little dependency"|
|  "A little copying beats a little dependency"         |
|                                                       |
|  "Don't communicate by sharing memory;                |
|   share memory by communicating"                      |
|  "Don't communicate by sharing memory;                |
|   share memory by communicating"                      |
+-------------------------------------------------------+
```

### Naming Conventions

```
+----------------------------------------------------------+
|  Go Naming Conventions                                   |
+----------------------------------------------------------+
|                                                          |
|  Package names: short, lowercase, singular               |
|    http, json, fmt, os, user, order                      |
|    NG: httpUtil, jsonParser, userService                 |
|    NG: utils, helpers, common, misc (ambiguous meaning)  |
|                                                          |
|  Exported: PascalCase                                    |
|    ReadFile, HTTPClient, UserID                          |
|                                                          |
|  Unexported: camelCase                                   |
|    readFile, httpClient, userID                          |
|                                                          |
|  Interfaces: -er suffix (for single-method interfaces)   |
|    Reader, Writer, Stringer                              |
|    Closer, Formatter, Handler                            |
|    Multi-method: ReadWriter, UserService                 |
|                                                          |
|  Acronyms: keep all uppercase                            |
|    HTTP, URL, ID, JSON, API, XML, SQL                    |
|    HTTPHandler (not HttpHandler)                         |
|    XMLRPC (not XmlRpc)                                   |
|    userID (not userId)                                   |
|                                                          |
|  Variable names: short scope -> short name               |
|    for i := 0; ...          // OK: short scope           |
|    for index := 0; ...      // NG: verbose               |
|    func process(r io.Reader) // OK: single letter clear  |
|                                                          |
|  Constants: PascalCase (CamelCase)                       |
|    MaxRetryCount, DefaultTimeout                         |
|    NG: MAX_RETRY_COUNT (not the Go convention)           |
+----------------------------------------------------------+
```

### Code Example 1: Good vs Bad Naming

```go
// NG: Verbose, Java-style
type IUserService interface { ... }       // I prefix unnecessary
type UserServiceImpl struct { ... }       // Impl suffix unnecessary
func (s *UserServiceImpl) GetUserByID(userID string) (*UserModel, error) { ... }

// OK: Go-style
type UserService interface { ... }        // Simple
type userService struct { ... }           // Unexported implementation
func (s *userService) User(id string) (*User, error) { ... }

// NG: Repeating the package name
package user
func UserCreate() { ... }   // user.UserCreate() is redundant
func UserDelete() { ... }   // user.UserDelete() is redundant

// OK: Leverage the package name
package user
func Create() { ... }       // user.Create() is clear
func Delete() { ... }       // user.Delete() is clear

// NG: Overuse of the Get prefix
func GetName() string { ... }      // Unnecessary in Go
func GetAge() int { ... }          // Getters don't use Get

// OK: Same name as the field
func (u *User) Name() string { return u.name }
func (u *User) Age() int { return u.age }
func (u *User) SetName(name string) { u.name = name }  // Setters use the Set prefix

// NG: Function names that return bool
func IsValid() bool { ... }     // OK (starting with Is is common)
func HasPermission() bool { ... }  // OK
func CheckValid() bool { ... }    // NG: Check should return error

// OK: Functions that return an error
func (u *User) Validate() error { ... }  // Check/Validate return errors
```

### Code Example 2: Package Design Best Practices

```go
// Example project structure

// Small project: flat structure
// project/
// |-- main.go
// |-- handler.go
// |-- store.go
// |-- model.go
// |-- go.mod

// Medium to large project: layered structure
// project/
// |-- cmd/
// |   |-- server/
// |       |-- main.go          -- entry point
// |-- internal/
// |   |-- handler/             -- HTTP handlers
// |   |   |-- user.go
// |   |   |-- order.go
// |   |-- service/             -- business logic
// |   |   |-- user.go
// |   |   |-- order.go
// |   |-- repository/          -- data access
// |   |   |-- user.go
// |   |   |-- order.go
// |   |-- model/               -- domain models
// |   |   |-- user.go
// |   |   |-- order.go
// |   |-- middleware/          -- middleware
// |       |-- auth.go
// |       |-- logging.go
// |-- pkg/                     -- public libraries (usable from other projects)
// |   |-- validator/
// |       |-- validator.go
// |-- migrations/              -- DB migrations
// |   |-- 000001_init.sql
// |-- configs/                 -- configuration files
// |-- go.mod
// |-- go.sum

// Direction of dependencies (keep it one-way)
// handler -> service -> repository -> model
//
// handler depends on the service interface
// service depends on the repository interface
// model depends on nothing (pure data structures)

// NG: Circular dependency
// package user -> package order -> package user (compile error)

// OK: Break dependencies with interfaces
// Define the interface in package service
// Implement in package repository
// Assemble in main.go (Dependency Injection)
```

---

## 2. Error Handling

### Code Example 3: Error Design Patterns

```go
package storage

import (
    "errors"
    "fmt"
)

// ===================================
// 1. Sentinel errors
// Fixed error values defined at package level
// Checked with errors.Is()
// ===================================
var (
    ErrNotFound     = errors.New("storage: not found")
    ErrDuplicate    = errors.New("storage: duplicate entry")
    ErrUnauthorized = errors.New("storage: unauthorized")
    ErrForbidden    = errors.New("storage: forbidden")
    ErrConflict     = errors.New("storage: conflict")
)

// ===================================
// 2. Custom error types
// Struct errors carrying additional information
// Checked with errors.As()
// ===================================
type ValidationError struct {
    Field   string
    Message string
}

func (e *ValidationError) Error() string {
    return fmt.Sprintf("validation error: %s - %s", e.Field, e.Message)
}

// Bundle multiple validation errors together
type ValidationErrors struct {
    Errors []ValidationError
}

func (e *ValidationErrors) Error() string {
    return fmt.Sprintf("validation failed: %d errors", len(e.Errors))
}

func (e *ValidationErrors) Add(field, message string) {
    e.Errors = append(e.Errors, ValidationError{Field: field, Message: message})
}

func (e *ValidationErrors) HasErrors() bool {
    return len(e.Errors) > 0
}

// ===================================
// 3. Error wrapping
// Add context with fmt.Errorf("%w", err)
// ===================================
func (s *Store) FindUser(id string) (*User, error) {
    user, err := s.db.Get(id)
    if err != nil {
        if errors.Is(err, sql.ErrNoRows) {
            // Wrap in a sentinel error
            return nil, fmt.Errorf("FindUser(%s): %w", id, ErrNotFound)
        }
        // Wrap the internal error
        return nil, fmt.Errorf("FindUser(%s): %w", id, err)
    }
    return user, nil
}

// ===================================
// 4. Error inspection
// Traverse the error chain with errors.Is / errors.As
// ===================================
func handleError(err error) {
    // Value comparison (sentinel error)
    if errors.Is(err, ErrNotFound) {
        // 404 response
    }
    if errors.Is(err, ErrUnauthorized) {
        // 401 response
    }

    // Type comparison (custom error)
    var valErr *ValidationError
    if errors.As(err, &valErr) {
        fmt.Printf("Field: %s, Message: %s\n", valErr.Field, valErr.Message)
    }

    var valErrs *ValidationErrors
    if errors.As(err, &valErrs) {
        for _, ve := range valErrs.Errors {
            fmt.Printf("  %s: %s\n", ve.Field, ve.Message)
        }
    }
}
```

### Error Handling Decision Flow

```
An error occurred
    |
    +-- Is it recoverable?
    |       |
    |       +-- YES -> Return the error (return err)
    |       |         Add context (fmt.Errorf("...: %w", err))
    |       |
    |       +-- NO -> log.Fatal / panic (startup only)
    |                 e.g., failure to load config file
    |                      DB connection failure (after retries)
    |
    +-- Does the caller need to know the error kind?
    |       |
    |       +-- YES -> Sentinel error or custom error type
    |       |         e.g., ErrNotFound -> 404
    |       |              *ValidationError -> 400 + details
    |       |
    |       +-- NO -> Just wrap as a string with fmt.Errorf
    |                 e.g., internal implementation details the caller doesn't need
    |
    +-- Is the error context sufficient?
    |       |
    |       +-- NO -> fmt.Errorf("op(%s): %w", id, err)
    |       |         Include function name and argument values in the message
    |       |
    |       +-- YES -> Return as-is
    |
    +-- Should the error be logged?
            |
            +-- Log only once, in the handler layer
            +-- In intermediate layers, wrap and return without logging
            +-- Avoid duplicate logging
```

### Code Example 4: Practical Error Handling Patterns

```go
package handler

import (
    "encoding/json"
    "errors"
    "log"
    "net/http"
)

// ErrorResponse is the unified format for API error responses
type ErrorResponse struct {
    Error   string            `json:"error"`
    Code    string            `json:"code,omitempty"`
    Details map[string]string `json:"details,omitempty"`
}

// handleError converts an error into an HTTP response (only once, at the handler layer)
func handleError(w http.ResponseWriter, err error, logger *log.Logger) {
    var resp ErrorResponse
    var statusCode int

    switch {
    case errors.Is(err, ErrNotFound):
        statusCode = http.StatusNotFound
        resp = ErrorResponse{Error: "resource not found", Code: "NOT_FOUND"}

    case errors.Is(err, ErrUnauthorized):
        statusCode = http.StatusUnauthorized
        resp = ErrorResponse{Error: "unauthorized", Code: "UNAUTHORIZED"}

    case errors.Is(err, ErrForbidden):
        statusCode = http.StatusForbidden
        resp = ErrorResponse{Error: "forbidden", Code: "FORBIDDEN"}

    case errors.Is(err, ErrConflict):
        statusCode = http.StatusConflict
        resp = ErrorResponse{Error: "conflict", Code: "CONFLICT"}

    default:
        var valErrs *ValidationErrors
        if errors.As(err, &valErrs) {
            statusCode = http.StatusBadRequest
            details := make(map[string]string)
            for _, ve := range valErrs.Errors {
                details[ve.Field] = ve.Message
            }
            resp = ErrorResponse{
                Error:   "validation failed",
                Code:    "VALIDATION_ERROR",
                Details: details,
            }
        } else {
            // Unknown error -> 500
            statusCode = http.StatusInternalServerError
            resp = ErrorResponse{Error: "internal server error", Code: "INTERNAL"}
            // Log internal errors (do not include them in the response)
            logger.Printf("Internal error: %v", err)
        }
    }

    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(statusCode)
    json.NewEncoder(w).Encode(resp)
}
```

### errors.Is vs errors.As Comparison

| Function | Purpose | Compared against | When to use |
|----------|---------|------------------|-------------|
| `errors.Is(err, target)` | Whether the error chain contains an error equal to target | Value (sentinel error) | `ErrNotFound`, `sql.ErrNoRows` |
| `errors.As(err, &target)` | Whether the error chain contains an error of target's type | Type (custom error) | `*ValidationError`, `*os.PathError` |
| `errors.Unwrap(err)` | Unwrap only one layer | -- | Normally not used directly |
| `errors.Join(err1, err2)` | Combine multiple errors (Go 1.20+) | -- | Multiple post-processing errors |

---

## 3. Interface Design

### Code Example 5: The Principle of Small Interfaces

```go
// Elegant interface designs from the Go standard library

// Single method -- highest reusability
type Reader interface {
    Read(p []byte) (n int, err error)
}

type Writer interface {
    Write(p []byte) (n int, err error)
}

type Closer interface {
    Close() error
}

type Stringer interface {
    String() string
}

// Build the required interface via composition
type ReadWriter interface {
    Reader
    Writer
}

type ReadWriteCloser interface {
    Reader
    Writer
    Closer
}

// ReadCloser is a typical example from the io package
type ReadCloser interface {
    Reader
    Closer
}
```

### Code Example 6: How to Use Interfaces Correctly

```go
// ===================================
// Principle 1: "Define interfaces on the consumer side"
// ===================================

// NG: Define the interface on the implementation side
package repository

type UserRepository interface {  // Interface in the implementation package
    GetByID(ctx context.Context, id int64) (*User, error)
    Create(ctx context.Context, u *User) error
}

type postgresUserRepo struct { ... }
// -> Consumers are forced to depend on the repository package

// OK: Define the interface on the consumer side
package service

// Declare only the methods you need
type UserStore interface {
    GetByID(ctx context.Context, id int64) (*User, error)
    Create(ctx context.Context, u *User) error
}

type UserService struct {
    store UserStore  // Depend on the interface
}

// The repository package merely provides a struct
package repository

type PostgresUserRepo struct {
    db *sql.DB
}

func (r *PostgresUserRepo) GetByID(ctx context.Context, id int64) (*User, error) { ... }
func (r *PostgresUserRepo) Create(ctx context.Context, u *User) error { ... }
// -> Implicitly satisfies service.UserStore

// ===================================
// Principle 2: "Accept interfaces, return structs"
// ===================================

// NG: Return an interface
func NewUserService(repo UserStore) UserServiceInterface {
    return &userService{repo: repo}
}
// -> Type information is lost; concrete methods are inaccessible

// OK: Return a struct
func NewUserService(repo UserStore) *UserService {
    return &UserService{repo: repo}
}
// -> Struct methods are directly accessible

// ===================================
// Principle 3: "Ask only for the functionality you actually use"
// ===================================

// NG: Demand an oversized interface
func ProcessData(rw ReadWriteCloser) error {
    // Actually only uses Read
    buf := make([]byte, 1024)
    _, err := rw.Read(buf)
    return err
}

// OK: Ask only for the functionality you need
func ProcessData(r io.Reader) error {
    buf := make([]byte, 1024)
    _, err := r.Read(buf)
    return err
}
// -> *os.File, *bytes.Buffer, *strings.Reader, and net.Conn all work
```

### Interface Design Principles

```
+----------------------------------------------------------+
|  "Accept interfaces, return structs"                     |
|  "Accept interfaces, return structs"                     |
+----------------------------------------------------------+
|                                                          |
|  // Parameter: interface (flexibility)                   |
|  func NewService(repo UserRepository) *UserService       |
|                                                          |
|  // Return value: struct (concreteness)                  |
|  func NewService(...) *UserService  // NOT UserService   |
|                                                          |
+----------------------------------------------------------+
|                                                          |
|  "The bigger the interface, the weaker the abstraction"  |
|                                     -- Rob Pike          |
+----------------------------------------------------------+
|                                                          |
|  "Define interfaces at the point of use"                 |
+----------------------------------------------------------+
|                                                          |
|  "Don't export interfaces for implementation"            |
+----------------------------------------------------------+

Benefits of implicit interface implementation:
+-----------------------------------------+
| Go: implicit (Structural Typing)        |
|   -> Implementations need not know      |
|      the interface exists               |
|   -> Simple dependency relationships    |
|   -> Easy to mock for testing           |
|                                         |
| Java/C#: explicit (implements / :)      |
|   -> Implementations depend on the      |
|      interface                          |
|   -> Large blast radius on changes      |
+-----------------------------------------+
```

---

## 4. Concurrency Best Practices

### Code Example 7: Goroutine Lifecycle Management

```go
package main

import (
    "context"
    "fmt"
    "log"
    "sync"
    "time"

    "golang.org/x/sync/errgroup"
)

// ===================================
// Pattern 1: WaitGroup
// Simple parallel processing, when no error handling is needed
// ===================================
func processItems(items []Item) {
    var wg sync.WaitGroup
    for _, item := range items {
        wg.Add(1)
        go func(item Item) {
            defer wg.Done()
            process(item)
        }(item)
    }
    wg.Wait()
}

// ===================================
// Pattern 2: errgroup
// Parallel processing + error handling (recommended)
// ===================================
func fetchAll(ctx context.Context, urls []string) ([]Response, error) {
    g, ctx := errgroup.WithContext(ctx)
    responses := make([]Response, len(urls))

    for i, url := range urls {
        i, url := i, url  // Capture loop variables (unnecessary in Go 1.22+)
        g.Go(func() error {
            resp, err := fetch(ctx, url)
            if err != nil {
                return fmt.Errorf("fetch %s: %w", url, err)
            }
            responses[i] = resp  // Different indexes, no synchronization needed
            return nil
        })
    }

    if err := g.Wait(); err != nil {
        return nil, err  // The first error is returned
    }
    return responses, nil
}

// ===================================
// Pattern 3: errgroup + semaphore (concurrency limit)
// ===================================
func fetchAllWithLimit(ctx context.Context, urls []string, maxConcurrency int) ([]Response, error) {
    g, ctx := errgroup.WithContext(ctx)
    g.SetLimit(maxConcurrency)  // Limit concurrent execution

    responses := make([]Response, len(urls))

    for i, url := range urls {
        i, url := i, url
        g.Go(func() error {
            resp, err := fetch(ctx, url)
            if err != nil {
                return err
            }
            responses[i] = resp
            return nil
        })
    }

    return responses, g.Wait()
}

// ===================================
// Pattern 4: Worker Pool
// Bounded parallelism with a job queue
// ===================================
func workerPool(ctx context.Context, jobs <-chan int, results chan<- int) {
    var wg sync.WaitGroup
    numWorkers := 5

    for i := 0; i < numWorkers; i++ {
        wg.Add(1)
        go func(id int) {
            defer wg.Done()
            for {
                select {
                case job, ok := <-jobs:
                    if !ok {
                        return // Channel closed
                    }
                    select {
                    case results <- process(job):
                    case <-ctx.Done():
                        return
                    }
                case <-ctx.Done():
                    return // Context canceled
                }
            }
        }(i)
    }

    // Close results after all workers finish
    go func() {
        wg.Wait()
        close(results)
    }()
}

// ===================================
// Pattern 5: Pipeline
// Stage-by-stage data processing
// ===================================
func pipeline(ctx context.Context, input <-chan int) <-chan string {
    // Stage 1: filter
    filtered := make(chan int)
    go func() {
        defer close(filtered)
        for v := range input {
            if v%2 == 0 {
                select {
                case filtered <- v:
                case <-ctx.Done():
                    return
                }
            }
        }
    }()

    // Stage 2: transform
    output := make(chan string)
    go func() {
        defer close(output)
        for v := range filtered {
            result := fmt.Sprintf("processed-%d", v*2)
            select {
            case output <- result:
            case <-ctx.Done():
                return
            }
        }
    }()

    return output
}

// ===================================
// Pattern 6: Fan-out / Fan-in
// Distributed processing and result aggregation
// ===================================
func fanOutFanIn(ctx context.Context, input <-chan int, numWorkers int) <-chan int {
    // Fan-out: distribute to multiple workers
    workers := make([]<-chan int, numWorkers)
    for i := 0; i < numWorkers; i++ {
        workers[i] = worker(ctx, input)
    }

    // Fan-in: aggregate results from multiple workers
    return merge(ctx, workers...)
}

func worker(ctx context.Context, input <-chan int) <-chan int {
    output := make(chan int)
    go func() {
        defer close(output)
        for v := range input {
            result := heavyProcess(v)
            select {
            case output <- result:
            case <-ctx.Done():
                return
            }
        }
    }()
    return output
}

func merge(ctx context.Context, channels ...<-chan int) <-chan int {
    var wg sync.WaitGroup
    merged := make(chan int)

    output := func(ch <-chan int) {
        defer wg.Done()
        for v := range ch {
            select {
            case merged <- v:
            case <-ctx.Done():
                return
            }
        }
    }

    wg.Add(len(channels))
    for _, ch := range channels {
        go output(ch)
    }

    go func() {
        wg.Wait()
        close(merged)
    }()

    return merged
}
```

### Code Example 8: Proper context Propagation

```go
// ===================================
// Rules for context
// ===================================

// 1. Pass context as the first parameter
func fetchData(ctx context.Context) (*Data, error) { ... }

// 2. Do not store context in a struct
// NG:
type Server struct {
    ctx context.Context  // NG: storing a request-scoped ctx
}
// OK: pass it as a method argument

// 3. Do not pass a nil context
// NG: fetchData(nil)
// OK: fetchData(context.Background())
//     fetchData(context.TODO())  // when you plan to replace it with a proper context later

// 4. Keep context.WithValue minimal (authentication info, etc., only)
// NG: cramming lots of data into context
// OK: only cross-cutting concerns such as request IDs and auth info

// Context chaining
func main() {
    // Root context (cancellable)
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    // Context with a timeout
    ctx, cancel = context.WithTimeout(ctx, 30*time.Second)
    defer cancel()

    // Context with a deadline
    deadline := time.Now().Add(1 * time.Minute)
    ctx, cancel = context.WithDeadline(ctx, deadline)
    defer cancel()

    // Context with a value (cross-cutting concerns only)
    ctx = context.WithValue(ctx, requestIDKey{}, "req-123")
}

// Using context inside an HTTP handler
http.HandleFunc("/api/data", func(w http.ResponseWriter, r *http.Request) {
    // Use the request's context
    // -> automatically canceled when the client disconnects
    ctx := r.Context()

    data, err := fetchData(ctx)
    if err != nil {
        if ctx.Err() == context.Canceled {
            // The client disconnected -> log only
            return
        }
        if ctx.Err() == context.DeadlineExceeded {
            http.Error(w, "timeout", http.StatusGatewayTimeout)
            return
        }
        http.Error(w, err.Error(), 500)
        return
    }
    json.NewEncoder(w).Encode(data)
})

// Propagate context downstream
func fetchData(ctx context.Context) (*Data, error) {
    req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
    if err != nil {
        return nil, err
    }
    resp, err := http.DefaultClient.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()
    // ...
}
```

### Concurrency Pattern Comparison

| Pattern | Use case | Error handling | Cancellation | Complexity | Recommendation |
|---------|----------|----------------|--------------|------------|----------------|
| goroutine + WaitGroup | Simple parallel processing | Manual (e.g., channels) | Manual (context) | Low | Basic |
| errgroup | Parallel processing + error aggregation | Automatic (first error) | Automatic (context-integrated) | Low | Recommended |
| errgroup + SetLimit | Concurrency-limited | Automatic | Automatic | Low | Recommended |
| Worker Pool | Job queues | Manual | context-aware | Medium | Special-purpose |
| Pipeline | Staged processing | Propagated between stages | context + done | High | Special-purpose |
| Fan-out/Fan-in | Distributed processing + aggregation | Handled at aggregation | context-aware | High | Special-purpose |

---

## 5. Struct and Method Design

### Code Example 9: Make Zero Values Useful

```go
// Great zero-value designs in the Go standard library

// sync.Mutex -- usable as its zero value
var mu sync.Mutex
mu.Lock() // No initialization required

// bytes.Buffer -- usable as its zero value
var buf bytes.Buffer
buf.WriteString("hello")

// sync.Once -- usable as its zero value
var once sync.Once
once.Do(func() { /* initialization */ })

// Design your own types so their zero value is useful
type Logger struct {
    output io.Writer // nil means use os.Stderr
    level  int       // 0 means INFO level
    prefix string    // "" means empty string
}

func (l *Logger) writer() io.Writer {
    if l.output == nil {
        return os.Stderr
    }
    return l.output
}

func (l *Logger) Info(msg string) {
    fmt.Fprintf(l.writer(), "%s[INFO] %s\n", l.prefix, msg)
}

// Usable as a zero value
var log Logger
log.Info("ready") // Outputs at INFO level to os.Stderr

// Explicit initialization is also possible
customLog := Logger{
    output: os.Stdout,
    level:  2,
    prefix: "[myapp] ",
}
customLog.Info("started")

// NG: design where the zero value is invalid
type BadConfig struct {
    MaxRetries int  // 0 = no retries? or unset?
    Timeout    time.Duration  // 0 = immediate timeout?
}

// OK: design that distinguishes the zero value
type GoodConfig struct {
    MaxRetries *int           // nil = unset (default 3)
    Timeout    time.Duration  // 0 = default (30 seconds)
}

func (c *GoodConfig) maxRetries() int {
    if c.MaxRetries == nil {
        return 3 // default
    }
    return *c.MaxRetries
}

func (c *GoodConfig) timeout() time.Duration {
    if c.Timeout == 0 {
        return 30 * time.Second // default
    }
    return c.Timeout
}
```

### Code Example 10: Functional Options Pattern

```go
// A complete implementation of the functional options pattern

type Server struct {
    addr         string
    readTimeout  time.Duration
    writeTimeout time.Duration
    idleTimeout  time.Duration
    maxConn      int
    logger       *log.Logger
    tlsConfig    *tls.Config
    middleware   []Middleware
}

// Option is a configuration option for Server
type Option func(*Server)

// WithReadTimeout sets the read timeout
func WithReadTimeout(d time.Duration) Option {
    return func(s *Server) { s.readTimeout = d }
}

// WithWriteTimeout sets the write timeout
func WithWriteTimeout(d time.Duration) Option {
    return func(s *Server) { s.writeTimeout = d }
}

// WithIdleTimeout sets the idle timeout
func WithIdleTimeout(d time.Duration) Option {
    return func(s *Server) { s.idleTimeout = d }
}

// WithMaxConnections sets the maximum number of connections
func WithMaxConnections(n int) Option {
    return func(s *Server) { s.maxConn = n }
}

// WithLogger sets the logger
func WithLogger(l *log.Logger) Option {
    return func(s *Server) { s.logger = l }
}

// WithTLS applies the TLS configuration
func WithTLS(config *tls.Config) Option {
    return func(s *Server) { s.tlsConfig = config }
}

// WithMiddleware appends middleware
func WithMiddleware(mw ...Middleware) Option {
    return func(s *Server) { s.middleware = append(s.middleware, mw...) }
}

// NewServer creates a new server
func NewServer(addr string, opts ...Option) *Server {
    s := &Server{
        addr:         addr,
        readTimeout:  5 * time.Second,   // default values
        writeTimeout: 10 * time.Second,
        idleTimeout:  120 * time.Second,
        maxConn:      100,
        logger:       log.Default(),
    }
    for _, opt := range opts {
        opt(s)
    }
    return s
}

// Example usage
func main() {
    srv := NewServer(":8080",
        WithReadTimeout(30*time.Second),
        WithMaxConnections(1000),
        WithLogger(customLogger),
        WithMiddleware(loggingMW, authMW),
    )

    // When the defaults are sufficient
    simpleSrv := NewServer(":8080")
}

// Benefits of the functional options pattern:
// 1. Backward compatibility: adding new options does not affect existing code
// 2. Readability: the meaning of each setting is clear
// 3. Default values: unspecified options use their defaults
// 4. Validation: validation can happen inside the Option functions
// 5. Documentation: you can write GoDoc for each With-function
```

### Code Example 11: Builder Pattern (Alternative to Functional Options)

```go
// Builder pattern (chained form)
type ServerBuilder struct {
    server *Server
    errs   []error
}

func NewServerBuilder(addr string) *ServerBuilder {
    return &ServerBuilder{
        server: &Server{
            addr:         addr,
            readTimeout:  5 * time.Second,
            writeTimeout: 10 * time.Second,
            maxConn:      100,
        },
    }
}

func (b *ServerBuilder) ReadTimeout(d time.Duration) *ServerBuilder {
    if d <= 0 {
        b.errs = append(b.errs, fmt.Errorf("read timeout must be positive"))
        return b
    }
    b.server.readTimeout = d
    return b
}

func (b *ServerBuilder) MaxConnections(n int) *ServerBuilder {
    if n <= 0 {
        b.errs = append(b.errs, fmt.Errorf("max connections must be positive"))
        return b
    }
    b.server.maxConn = n
    return b
}

func (b *ServerBuilder) Build() (*Server, error) {
    if len(b.errs) > 0 {
        return nil, fmt.Errorf("server builder errors: %v", b.errs)
    }
    return b.server, nil
}

// Example usage
srv, err := NewServerBuilder(":8080").
    ReadTimeout(30 * time.Second).
    MaxConnections(1000).
    Build()
```

---

## 6. Designing for Testability

### Code Example 12: Code That Is Easy to Test

```go
// ===================================
// Principle: inject dependencies and abstract with interfaces
// ===================================

// Hard-to-test code
// NG: Directly depends on a concrete implementation
type UserServiceBad struct {
    db *sql.DB  // Depends on a concrete type -> needs a DB for testing
}

func (s *UserServiceBad) GetUser(id int64) (*User, error) {
    return s.db.QueryRow("SELECT ...") // Accesses the DB directly
}

// Easy-to-test code
// OK: Depends on an interface
type UserRepository interface {
    GetByID(ctx context.Context, id int64) (*User, error)
}

type UserService struct {
    repo UserRepository  // Depends on the interface
}

func NewUserService(repo UserRepository) *UserService {
    return &UserService{repo: repo}
}

func (s *UserService) GetUser(ctx context.Context, id int64) (*User, error) {
    if id <= 0 {
        return nil, fmt.Errorf("invalid user id: %d", id)
    }
    return s.repo.GetByID(ctx, id)
}

// Mock for testing
type mockUserRepo struct {
    users map[int64]*User
    err   error // Error injection for tests
}

func (m *mockUserRepo) GetByID(ctx context.Context, id int64) (*User, error) {
    if m.err != nil {
        return nil, m.err
    }
    u, ok := m.users[id]
    if !ok {
        return nil, ErrNotFound
    }
    return u, nil
}

// Table-driven test
func TestUserService_GetUser(t *testing.T) {
    tests := []struct {
        name    string
        id      int64
        mock    *mockUserRepo
        want    *User
        wantErr bool
    }{
        {
            name: "success",
            id:   1,
            mock: &mockUserRepo{users: map[int64]*User{1: {ID: 1, Name: "Alice"}}},
            want: &User{ID: 1, Name: "Alice"},
        },
        {
            name:    "not found",
            id:      999,
            mock:    &mockUserRepo{users: map[int64]*User{}},
            wantErr: true,
        },
        {
            name:    "invalid id",
            id:      0,
            mock:    &mockUserRepo{},
            wantErr: true,
        },
        {
            name:    "db error",
            id:      1,
            mock:    &mockUserRepo{err: fmt.Errorf("connection refused")},
            wantErr: true,
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            svc := NewUserService(tt.mock)
            got, err := svc.GetUser(context.Background(), tt.id)

            if (err != nil) != tt.wantErr {
                t.Errorf("GetUser() error = %v, wantErr %v", err, tt.wantErr)
                return
            }
            if tt.want != nil && got.Name != tt.want.Name {
                t.Errorf("GetUser() = %v, want %v", got, tt.want)
            }
        })
    }
}
```

### Code Example 13: Testability of Time

```go
// NG: Calling time.Now() directly makes tests flaky
func (s *TokenService) IsExpired(token *Token) bool {
    return time.Now().After(token.ExpiresAt) // Uncontrollable in tests
}

// OK: Make the time function injectable
type TokenService struct {
    now func() time.Time // Swappable in tests
}

func NewTokenService() *TokenService {
    return &TokenService{now: time.Now}
}

func (s *TokenService) IsExpired(token *Token) bool {
    return s.now().After(token.ExpiresAt)
}

// Test
func TestIsExpired(t *testing.T) {
    fixedTime := time.Date(2024, 6, 1, 12, 0, 0, 0, time.UTC)
    svc := &TokenService{
        now: func() time.Time { return fixedTime },
    }

    token := &Token{ExpiresAt: fixedTime.Add(-1 * time.Hour)}
    if !svc.IsExpired(token) {
        t.Error("expected expired token")
    }
}
```

---

## 7. Anti-patterns

### Anti-pattern 1: Swallowing Errors

```go
// NG: Ignoring errors
data, _ := json.Marshal(user)    // Discards the marshal error
f, _ := os.Open("config.yaml")  // Panics if the file does not exist
defer f.Close()

// OK: Handle errors properly
data, err := json.Marshal(user)
if err != nil {
    return fmt.Errorf("failed to marshal user to JSON: %w", err)
}

f, err := os.Open("config.yaml")
if err != nil {
    return fmt.Errorf("cannot open config file: %w", err)
}
defer f.Close()

// The only situations where ignoring errors is acceptable:
// 1. Close() in defer (though you should still log it)
defer func() {
    if err := f.Close(); err != nil {
        log.Printf("file close error: %v", err)
    }
}()
// 2. fmt.Fprint* family (almost never fails)
fmt.Fprintf(w, "hello")
```

### Anti-pattern 2: Abuse of init()

```go
// NG: Complex initialization in init()
func init() {
    db, err := sql.Open("postgres", os.Getenv("DATABASE_URL"))
    if err != nil {
        log.Fatal(err) // Runs during tests too
    }
    globalDB = db
}
// Problems:
// - Requires a DB during testing
// - Initialization order is unclear
// - log.Fatal on error -> tests crash

// OK: Explicit initialization function
func NewApp(cfg Config) (*App, error) {
    db, err := sql.Open("postgres", cfg.DatabaseURL)
    if err != nil {
        return nil, fmt.Errorf("DB connection failed: %w", err)
    }
    return &App{db: db}, nil
}

// When init() is appropriate:
// - driver.Register (registering a database/sql driver)
// - Setting prerequisites for flag.Parse
// - Very simple initialization (assigning initial values to variables, etc.)
```

### Anti-pattern 3: Fire-and-Forget Goroutines

```go
// NG: Goroutine lifecycle is unmanaged
func handler(w http.ResponseWriter, r *http.Request) {
    go sendEmail(user.Email) // A panic goes unnoticed
    w.WriteHeader(200)       // No check of the email result
}
// Problems:
// - A panic -> the whole process crashes
// - Errors -> undetectable
// - Memory leak -> the goroutine may never terminate

// OK: Manage with errgroup or recover
func handler(w http.ResponseWriter, r *http.Request) {
    g, ctx := errgroup.WithContext(r.Context())
    g.Go(func() error {
        return sendEmail(ctx, user.Email)
    })
    if err := g.Wait(); err != nil {
        log.Printf("email send failed: %v", err)
    }
    w.WriteHeader(200)
}

// OK: A worker for background tasks
type BackgroundWorker struct {
    tasks chan func()
    wg    sync.WaitGroup
}

func (w *BackgroundWorker) Submit(task func()) {
    w.wg.Add(1)
    w.tasks <- func() {
        defer w.wg.Done()
        defer func() {
            if r := recover(); r != nil {
                log.Printf("background task panic: %v", r)
            }
        }()
        task()
    }
}

func (w *BackgroundWorker) Shutdown() {
    close(w.tasks)
    w.wg.Wait()
}
```

### Anti-pattern 4: Misuse of sync.Mutex

```go
// NG: Copying a Mutex
type Cache struct {
    mu    sync.Mutex
    items map[string]string
}

func (c Cache) Get(key string) string {  // Value receiver -> the Mutex is copied
    c.mu.Lock()
    defer c.mu.Unlock()
    return c.items[key]
}

// OK: Use a pointer receiver
func (c *Cache) Get(key string) string {
    c.mu.Lock()
    defer c.mu.Unlock()
    return c.items[key]
}

// NG: Exposing the Mutex
type Cache struct {
    Mu    sync.Mutex  // Exported field -> external code can lock it
    Items map[string]string
}

// OK: Keep the Mutex unexported
type Cache struct {
    mu    sync.Mutex
    items map[string]string
}

// NG: Using a Mutex even for reads (should be an RWMutex)
func (c *Cache) Get(key string) string {
    c.mu.Lock()       // An exclusive lock for read-only access
    defer c.mu.Unlock()
    return c.items[key]
}

// OK: Use RLock for reads
type Cache struct {
    mu    sync.RWMutex
    items map[string]string
}

func (c *Cache) Get(key string) string {
    c.mu.RLock()       // Read lock (multiple goroutines can read concurrently)
    defer c.mu.RUnlock()
    return c.items[key]
}

func (c *Cache) Set(key, value string) {
    c.mu.Lock()        // Write lock (exclusive)
    defer c.mu.Unlock()
    c.items[key] = value
}
```

### Anti-pattern 5: Unnecessary else

```go
// NG: Redundant else
func process(data []byte) error {
    if len(data) == 0 {
        return errors.New("empty data")
    } else {
        return parse(data)  // else is unnecessary
    }
}

// OK: Early return
func process(data []byte) error {
    if len(data) == 0 {
        return errors.New("empty data")
    }
    return parse(data)  // Happy path at indentation 0
}

// NG: Deep nesting
func validate(user *User) error {
    if user != nil {
        if user.Name != "" {
            if user.Age > 0 {
                return nil
            } else {
                return errors.New("age must be positive")
            }
        } else {
            return errors.New("name is required")
        }
    } else {
        return errors.New("user is nil")
    }
}

// OK: Guard clauses with early returns
func validate(user *User) error {
    if user == nil {
        return errors.New("user is nil")
    }
    if user.Name == "" {
        return errors.New("name is required")
    }
    if user.Age <= 0 {
        return errors.New("age must be positive")
    }
    return nil
}
```

---

## 8. Performance Guidelines

### Memory Allocation Optimization

```go
// NG: Growing a slice dynamically inside a loop
func processItems(items []Item) []Result {
    var results []Result  // Unknown capacity
    for _, item := range items {
        results = append(results, process(item))
        // -> Memory reallocation + copy every time capacity runs out
    }
    return results
}

// OK: Pre-allocate capacity
func processItems(items []Item) []Result {
    results := make([]Result, 0, len(items))  // Pre-allocate capacity
    for _, item := range items {
        results = append(results, process(item))
    }
    return results
}

// String concatenation with strings.Builder
// NG: Concatenation via += (creates a new string each time)
func buildString(parts []string) string {
    result := ""
    for _, p := range parts {
        result += p  // O(n^2) memory allocations
    }
    return result
}

// OK: strings.Builder (O(n))
func buildString(parts []string) string {
    var b strings.Builder
    b.Grow(estimatedSize)  // Pre-allocate an estimated size
    for _, p := range parts {
        b.WriteString(p)
    }
    return b.String()
}

// Reuse objects with sync.Pool
var bufPool = sync.Pool{
    New: func() interface{} {
        return new(bytes.Buffer)
    },
}

func processRequest(data []byte) string {
    buf := bufPool.Get().(*bytes.Buffer)
    buf.Reset()
    defer bufPool.Put(buf)

    buf.Write(data)
    // ... processing
    return buf.String()
}
```

---

## FAQ

### Q1. How do I format Go code?

`gofmt` is Go's official formatter, applying a uniform style without room for debate. `goimports` goes further than `gofmt` and also organizes imports. In CI, run `gofmt -l .` to verify there are no unformatted files.

```bash
# Format
gofmt -w .
goimports -w .

# Check in CI
test -z "$(gofmt -l .)"
```

### Q2. What linter should I use?

`golangci-lint` is the industry standard. It can run many linters in a unified way, including `staticcheck`, `errcheck`, `govet`, and `gosimple`. You can configure project-specific rules in `.golangci.yml`.

```yaml
# .golangci.yml
linters:
  enable:
    - errcheck
    - govet
    - staticcheck
    - gosimple
    - ineffassign
    - unused
    - misspell
    - gofmt
    - goimports
    - gocritic
    - revive
    - prealloc      # Recommends pre-allocating slices
    - bodyclose     # Forgotten HTTP response Body close
    - nilerr        # nil error checks
    - exportloopref # Loop variable capture
```

### Q3. What is the best practice for package layout?

For small projects, a flat structure is sufficient. For large ones, use `internal/` to restrict external visibility and split packages per domain. Avoid circular dependencies and keep the dependency direction one-way. Place entry points under `cmd/`.

```
Recommended layout:
  cmd/server/main.go          -- entry point (keep it thin)
  internal/                    -- not externally visible
    handler/                   -- HTTP handlers
    service/                   -- business logic
    repository/                -- data access
    model/                     -- domain models
  pkg/                         -- externally visible libraries

Dependency direction:
  main -> handler -> service -> repository -> model
  (one-way, no cycles)
```

### Q4. When should I use a pointer receiver vs a value receiver?

```
Use a pointer receiver (*T) when:
  -> The struct is mutated (setter)
  -> The struct is large (copy cost is high)
  -> It contains fields that must not be copied, such as sync.Mutex
  -> For consistency (if any method has a pointer receiver, all methods should)

Use a value receiver (T) when:
  -> The struct is small and immutable (e.g., Point{x, y})
  -> A type used as a map key
  -> A type treated as a value, like time.Time
```

### Q5. What are the major improvements in Go 1.22 and later?

- **Loop variable scope fix**: the loop variable in `for i, v := range` is freshly allocated on each iteration
- **range over integers**: `for i := range 10` is now possible
- **Enhanced HTTP routing**: `http.ServeMux` supports path parameters
- **cmp package**: standardized comparison functions

---

## Summary

| Concept | Key points |
|---------|-----------|
| Naming conventions | PascalCase/camelCase, keep acronyms uppercase, short scope -> short name |
| Interfaces | Keep them small, define them on the consumer side, implicit implementation |
| Error handling | Wrap with %w, inspect with Is/As, log once in the handler layer |
| Zero values | Design useful zero values |
| Option pattern | Flexible initialization, backward compatibility |
| context | First argument, propagate cancellation, values for cross-cutting concerns only |
| errgroup | Recommended pattern for goroutine management |
| gofmt / golangci-lint | Automate code quality, wire it into CI |
| Testability | Enable mocking via interfaces and DI |
| Performance | Pre-allocate slices, strings.Builder, sync.Pool |
| Package design | One-way dependencies, no cycles, keep private with internal |
| Early return | Reduce nesting with guard clauses |

---

## Recommended Next Guides

- **02-web/04-testing.md** -- Testing: table-driven tests, testify, httptest
- **03-tools/01-generics.md** -- Generics: type parameters, constraints
- **03-tools/02-profiling.md** -- Profiling: pprof, trace

---

## References

1. **Go official -- Effective Go** https://go.dev/doc/effective_go
2. **Go official -- Code Review Comments** https://go.dev/wiki/CodeReviewComments
3. **Go Blog -- Error handling and Go** https://go.dev/blog/error-handling-and-go
4. **Uber Go Style Guide** https://github.com/uber-go/guide/blob/master/style.md
5. **Go Proverbs** https://go-proverbs.github.io/
6. **100 Go Mistakes and How to Avoid Them** -- Teiva Harsanyi
7. **Go official -- Package Names** https://go.dev/blog/package-names
8. **golang.org/x/sync/errgroup** https://pkg.go.dev/golang.org/x/sync/errgroup
