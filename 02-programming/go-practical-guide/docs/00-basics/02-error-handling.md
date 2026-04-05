# Error Handling -- Go's Error Design Philosophy

> Go adopts explicit error handling centered on the error interface, and achieves robust error propagation through errors.Is/As, sentinel errors, and wrapping.

---

## What You Will Learn in This Chapter

1. **error interface** -- Why errors in Go are simply an interface
2. **errors.Is / errors.As** -- How to inspect error chains
3. **Error wrapping** -- Adding context with `fmt.Errorf("%w", err)`
4. **Custom error types** -- Designing domain-specific errors
5. **Error handling strategies** -- Processing policies by layer
6. **panic/recover** -- Appropriate use cases and recovery patterns


## Prerequisites

Before reading this guide, the following knowledge will help deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding the content of [Types and Structs -- Understanding Go's Type System](./01-types-and-structs.md)

---

## 1. Basics of the error Interface

### 1.1 Definition of the error Interface

In Go, errors are not a special syntax but simply an interface. This forms the foundation of Go's error design.

```go
// Defined in the builtin package
type error interface {
    Error() string
}
```

Benefits of this design:
- Errors can be treated as values (first-class values)
- Any type can implement the error interface
- Errors can carry additional information (codes, fields, stack traces, etc.)
- They can be used in conditionals, comparisons, and storage just like any other value

### Code Example 1: Basics and Implementation of the error Interface

```go
package main

import (
    "fmt"
    "net"
    "os"
    "strconv"
    "time"
)

// error is a built-in interface
// type error interface {
//     Error() string
// }

// Custom error type: validation error
type ValidationError struct {
    Field   string
    Message string
}

func (e *ValidationError) Error() string {
    return fmt.Sprintf("validation error: field=%q, message=%q", e.Field, e.Message)
}

// Custom error type: business logic error
type BusinessError struct {
    Code    string
    Message string
    Details map[string]string
}

func (e *BusinessError) Error() string {
    return fmt.Sprintf("[%s] %s", e.Code, e.Message)
}

// Custom error type: temporary error (retryable)
type RetryableError struct {
    Err       error
    RetryAfter time.Duration
}

func (e *RetryableError) Error() string {
    return fmt.Sprintf("retryable error (retry after %v): %v", e.RetryAfter, e.Err)
}

func (e *RetryableError) Unwrap() error {
    return e.Err
}

func (e *RetryableError) Temporary() bool {
    return true
}

// Create a simple error with errors.New
func validateAge(age int) error {
    if age < 0 {
        return &ValidationError{Field: "age", Message: "must be non-negative"}
    }
    if age > 200 {
        return &ValidationError{Field: "age", Message: "unreasonable value"}
    }
    return nil
}

func main() {
    // Examples of errors from the standard library
    _, err := strconv.Atoi("not_a_number")
    fmt.Printf("strconv error: %v (type: %T)\n", err, err)

    _, err = os.Open("/nonexistent/file")
    fmt.Printf("os error: %v (type: %T)\n", err, err)

    _, err = net.Dial("tcp", "invalid:address")
    fmt.Printf("net error: %v (type: %T)\n", err, err)

    // Using custom errors
    err = validateAge(-5)
    fmt.Printf("validation error: %v\n", err)

    err = validateAge(30)
    fmt.Printf("valid age: error=%v\n", err) // nil
}
```

### Code Example 2: Sentinel Errors

```go
package main

import (
    "errors"
    "fmt"
)

// Definition of sentinel errors
// Exposed at the package level; callers use them for checks
var (
    ErrNotFound      = errors.New("not found")
    ErrUnauthorized  = errors.New("unauthorized")
    ErrForbidden     = errors.New("forbidden")
    ErrConflict      = errors.New("conflict")
    ErrInternalError = errors.New("internal error")
    ErrInvalidInput  = errors.New("invalid input")
    ErrTimeout       = errors.New("timeout")
    ErrRateLimited   = errors.New("rate limited")
)

// User repository
type User struct {
    ID    int
    Name  string
    Email string
}

var db = map[int]*User{
    1: {ID: 1, Name: "Alice", Email: "alice@example.com"},
    2: {ID: 2, Name: "Bob", Email: "bob@example.com"},
}

func FindUser(id int) (*User, error) {
    if id <= 0 {
        return nil, fmt.Errorf("find user: invalid id %d: %w", id, ErrInvalidInput)
    }
    user, exists := db[id]
    if !exists {
        return nil, fmt.Errorf("find user (id=%d): %w", id, ErrNotFound)
    }
    return user, nil
}

func FindUserByEmail(email string) (*User, error) {
    if email == "" {
        return nil, fmt.Errorf("find user by email: empty email: %w", ErrInvalidInput)
    }
    for _, u := range db {
        if u.Email == email {
            return u, nil
        }
    }
    return nil, fmt.Errorf("find user by email %q: %w", email, ErrNotFound)
}

func CreateUser(name, email string) (*User, error) {
    if name == "" || email == "" {
        return nil, fmt.Errorf("create user: name and email required: %w", ErrInvalidInput)
    }

    // Duplicate check
    existing, err := FindUserByEmail(email)
    if err == nil && existing != nil {
        return nil, fmt.Errorf("create user: email %q already exists: %w", email, ErrConflict)
    }
    // ErrNotFound is an expected result (no duplicate)
    if err != nil && !errors.Is(err, ErrNotFound) {
        return nil, fmt.Errorf("create user: check existing: %w", err)
    }

    user := &User{
        ID:    len(db) + 1,
        Name:  name,
        Email: email,
    }
    db[user.ID] = user
    return user, nil
}

func main() {
    // Success case
    user, err := FindUser(1)
    if err != nil {
        fmt.Printf("Error: %v\n", err)
    } else {
        fmt.Printf("Found: %+v\n", user)
    }

    // NotFound case
    _, err = FindUser(999)
    if errors.Is(err, ErrNotFound) {
        fmt.Println("User not found (expected)")
    }

    // InvalidInput case
    _, err = FindUser(-1)
    if errors.Is(err, ErrInvalidInput) {
        fmt.Println("Invalid input (expected)")
    }

    // Conflict case
    _, err = CreateUser("Charlie", "alice@example.com")
    if errors.Is(err, ErrConflict) {
        fmt.Printf("Conflict: %v\n", err)
    }

    // Successful creation
    user, err = CreateUser("Charlie", "charlie@example.com")
    if err == nil {
        fmt.Printf("Created: %+v\n", user)
    }
}
```

### Code Example 3: Error Wrapping and Adding Context

```go
package main

import (
    "database/sql"
    "encoding/json"
    "errors"
    "fmt"
    "io"
    "net/http"
    "os"
)

var ErrNotFound = errors.New("not found")

type Profile struct {
    Bio    string
    Avatar string
}

type User struct {
    ID   int
    Name string
}

// Layer 1: repository layer
func findUserInDB(id int) (*User, error) {
    // Simulating a DB operation
    if id > 100 {
        return nil, fmt.Errorf("query users where id=%d: %w", id, ErrNotFound)
    }
    return &User{ID: id, Name: "TestUser"}, nil
}

func loadProfileFromDB(userID int) (*Profile, error) {
    if userID > 50 {
        return nil, fmt.Errorf("query profiles where user_id=%d: %w", userID, ErrNotFound)
    }
    return &Profile{Bio: "Hello", Avatar: "/avatars/default.png"}, nil
}

// Layer 2: service layer (wraps with added context)
func GetUserProfile(id int) (*Profile, error) {
    user, err := findUserInDB(id)
    if err != nil {
        return nil, fmt.Errorf("get user profile: find user (id=%d): %w", id, err)
    }

    profile, err := loadProfileFromDB(user.ID)
    if err != nil {
        return nil, fmt.Errorf("get user profile: load profile for %q (id=%d): %w",
            user.Name, user.ID, err)
    }

    return profile, nil
}

// Layer 3: handler layer (HTTP response depending on error type)
func handleGetProfile(w http.ResponseWriter, r *http.Request) {
    userID := 200 // simulation

    profile, err := GetUserProfile(userID)
    if err != nil {
        // HTTP status code depending on the kind of error
        switch {
        case errors.Is(err, ErrNotFound):
            http.Error(w, "user or profile not found", http.StatusNotFound)
        default:
            // Do not return internal error details to the client
            http.Error(w, "internal server error", http.StatusInternalServerError)
        }
        // Output details to the log
        fmt.Printf("ERROR: %v\n", err)
        return
    }

    json.NewEncoder(w).Encode(profile)
}

// Practical error wrapping patterns
func readConfig(path string) ([]byte, error) {
    data, err := os.ReadFile(path)
    if err != nil {
        return nil, fmt.Errorf("read config %q: %w", path, err)
    }
    return data, nil
}

func parseConfig(data []byte) (map[string]string, error) {
    var config map[string]string
    if err := json.Unmarshal(data, &config); err != nil {
        return nil, fmt.Errorf("parse config: %w", err)
    }
    return config, nil
}

func loadConfig(path string) (map[string]string, error) {
    data, err := readConfig(path)
    if err != nil {
        return nil, fmt.Errorf("load config: %w", err)
    }

    config, err := parseConfig(data)
    if err != nil {
        return nil, fmt.Errorf("load config: %w", err)
    }

    return config, nil
}

// Proper error handling in file processing
func copyFile(src, dst string) (int64, error) {
    srcFile, err := os.Open(src)
    if err != nil {
        return 0, fmt.Errorf("copy file: open src %q: %w", src, err)
    }
    defer srcFile.Close()

    dstFile, err := os.Create(dst)
    if err != nil {
        return 0, fmt.Errorf("copy file: create dst %q: %w", dst, err)
    }

    // Also check the error from Close in a defer
    defer func() {
        if cerr := dstFile.Close(); cerr != nil && err == nil {
            err = fmt.Errorf("copy file: close dst %q: %w", dst, cerr)
        }
    }()

    n, err := io.Copy(dstFile, srcFile)
    if err != nil {
        return 0, fmt.Errorf("copy file: copy data: %w", err)
    }

    return n, nil
}

func main() {
    // Inspect the error chain
    _, err := GetUserProfile(200)
    fmt.Printf("Error: %v\n", err)
    fmt.Printf("Is ErrNotFound: %t\n", errors.Is(err, ErrNotFound))

    // Example of checking sql.ErrNoRows
    sqlErr := fmt.Errorf("get user: %w", sql.ErrNoRows)
    fmt.Printf("Is sql.ErrNoRows: %t\n", errors.Is(sqlErr, sql.ErrNoRows))
}
```

### Code Example 4: errors.Is and errors.As in Detail

```go
package main

import (
    "errors"
    "fmt"
    "net"
    "os"
)

// Custom error type
type HTTPError struct {
    StatusCode int
    Message    string
    Err        error
}

func (e *HTTPError) Error() string {
    if e.Err != nil {
        return fmt.Sprintf("HTTP %d: %s: %v", e.StatusCode, e.Message, e.Err)
    }
    return fmt.Sprintf("HTTP %d: %s", e.StatusCode, e.Message)
}

func (e *HTTPError) Unwrap() error {
    return e.Err
}

// Custom Is method: treats errors with the same StatusCode as identical
func (e *HTTPError) Is(target error) bool {
    t, ok := target.(*HTTPError)
    if !ok {
        return false
    }
    return e.StatusCode == t.StatusCode
}

// ValidationError type
type ValidationError struct {
    Field   string
    Message string
}

func (e *ValidationError) Error() string {
    return fmt.Sprintf("validation error: %s - %s", e.Field, e.Message)
}

// NotFoundError type
type NotFoundError struct {
    Resource string
    ID       interface{}
}

func (e *NotFoundError) Error() string {
    return fmt.Sprintf("%s (id=%v) not found", e.Resource, e.ID)
}

func handleError(err error) {
    // errors.Is: checks whether the error chain contains a specific error
    if errors.Is(err, os.ErrNotExist) {
        fmt.Println("→ File does not exist")
        return
    }

    if errors.Is(err, os.ErrPermission) {
        fmt.Println("→ Permission denied")
        return
    }

    // errors.As: extracts a specific type from the error chain
    var httpErr *HTTPError
    if errors.As(err, &httpErr) {
        fmt.Printf("→ HTTP error: status=%d, message=%q\n",
            httpErr.StatusCode, httpErr.Message)
        return
    }

    var validErr *ValidationError
    if errors.As(err, &validErr) {
        fmt.Printf("→ Validation error: field=%q, message=%q\n",
            validErr.Field, validErr.Message)
        return
    }

    var notFoundErr *NotFoundError
    if errors.As(err, &notFoundErr) {
        fmt.Printf("→ Not found: resource=%q, id=%v\n",
            notFoundErr.Resource, notFoundErr.ID)
        return
    }

    // Interface check for net.Error
    var netErr net.Error
    if errors.As(err, &netErr) {
        fmt.Printf("→ Network error: timeout=%t, temporary=%t\n",
            netErr.Timeout(), netErr.Temporary())
        return
    }

    // Check for os.PathError
    var pathErr *os.PathError
    if errors.As(err, &pathErr) {
        fmt.Printf("→ Path error: op=%q, path=%q, err=%v\n",
            pathErr.Op, pathErr.Path, pathErr.Err)
        return
    }

    fmt.Printf("→ Unexpected error: %v\n", err)
}

func main() {
    // Test various errors
    errors_to_test := []error{
        fmt.Errorf("open config: %w", os.ErrNotExist),
        &HTTPError{StatusCode: 404, Message: "page not found", Err: nil},
        &ValidationError{Field: "email", Message: "invalid format"},
        &NotFoundError{Resource: "User", ID: 42},
        fmt.Errorf("service layer: %w",
            &HTTPError{StatusCode: 500, Message: "database error",
                Err: fmt.Errorf("connection refused")}),
    }

    for _, err := range errors_to_test {
        fmt.Printf("\nError: %v\n", err)
        handleError(err)
    }

    // Test the custom Is method
    err1 := &HTTPError{StatusCode: 404, Message: "user not found"}
    err2 := &HTTPError{StatusCode: 404, Message: "different message"}
    err3 := &HTTPError{StatusCode: 500, Message: "internal error"}

    fmt.Printf("\nerr1 Is err2 (same status): %t\n", errors.Is(err1, err2)) // true
    fmt.Printf("err1 Is err3 (diff status): %t\n", errors.Is(err1, err3))  // false

    // Searching within a wrapped error
    wrapped := fmt.Errorf("handler: %w",
        fmt.Errorf("service: %w",
            &NotFoundError{Resource: "Order", ID: 123}))

    var nfe *NotFoundError
    if errors.As(wrapped, &nfe) {
        fmt.Printf("\nFound NotFoundError in chain: %s id=%v\n",
            nfe.Resource, nfe.ID)
    }
}
```

### Code Example 5: Joining Multiple Errors (Go 1.20+)

```go
package main

import (
    "errors"
    "fmt"
    "strings"
)

type ValidationError struct {
    Field   string
    Message string
}

func (e *ValidationError) Error() string {
    return fmt.Sprintf("%s: %s", e.Field, e.Message)
}

type User struct {
    Name     string
    Email    string
    Password string
    Age      int
}

// Validation: returns multiple errors together
func validateUser(u *User) error {
    var errs []error

    if u.Name == "" {
        errs = append(errs, &ValidationError{Field: "name", Message: "required"})
    } else if len(u.Name) < 2 {
        errs = append(errs, &ValidationError{Field: "name", Message: "too short (min 2)"})
    } else if len(u.Name) > 100 {
        errs = append(errs, &ValidationError{Field: "name", Message: "too long (max 100)"})
    }

    if u.Email == "" {
        errs = append(errs, &ValidationError{Field: "email", Message: "required"})
    } else if !strings.Contains(u.Email, "@") {
        errs = append(errs, &ValidationError{Field: "email", Message: "invalid format"})
    }

    if len(u.Password) < 8 {
        errs = append(errs, &ValidationError{
            Field: "password", Message: "too short (min 8 characters)"})
    }
    if u.Password != "" {
        hasUpper := false
        hasDigit := false
        for _, c := range u.Password {
            if c >= 'A' && c <= 'Z' { hasUpper = true }
            if c >= '0' && c <= '9' { hasDigit = true }
        }
        if !hasUpper {
            errs = append(errs, &ValidationError{
                Field: "password", Message: "must contain uppercase letter"})
        }
        if !hasDigit {
            errs = append(errs, &ValidationError{
                Field: "password", Message: "must contain digit"})
        }
    }

    if u.Age < 0 || u.Age > 200 {
        errs = append(errs, &ValidationError{
            Field: "age", Message: fmt.Sprintf("invalid value: %d", u.Age)})
    }

    return errors.Join(errs...) // Go 1.20+: joins multiple errors. Returns nil if errs is empty
}

// Custom MultiError type (for compatibility with pre-Go 1.20)
type MultiError struct {
    Errors []error
}

func (e *MultiError) Error() string {
    if len(e.Errors) == 1 {
        return e.Errors[0].Error()
    }
    var msgs []string
    for _, err := range e.Errors {
        msgs = append(msgs, err.Error())
    }
    return fmt.Sprintf("%d errors: [%s]", len(e.Errors), strings.Join(msgs, "; "))
}

// Unwrap supports the multiple unwrap feature in Go 1.20+
func (e *MultiError) Unwrap() []error {
    return e.Errors
}

func main() {
    // User with every field invalid
    badUser := &User{
        Name:     "",
        Email:    "invalid",
        Password: "short",
        Age:      -5,
    }

    err := validateUser(badUser)
    if err != nil {
        fmt.Printf("Validation errors:\n%v\n\n", err)

        // Search for a specific type with errors.As
        var ve *ValidationError
        if errors.As(err, &ve) {
            fmt.Printf("First validation error: field=%s, msg=%s\n\n",
                ve.Field, ve.Message)
        }
    }

    // Valid user
    goodUser := &User{
        Name:     "Alice",
        Email:    "alice@example.com",
        Password: "SecureP4ss",
        Age:      30,
    }

    err = validateUser(goodUser)
    if err == nil {
        fmt.Println("Valid user!")
    }

    // Partially invalid user
    partialUser := &User{
        Name:     "Bob",
        Email:    "bob@example.com",
        Password: "weak",
        Age:      25,
    }

    err = validateUser(partialUser)
    if err != nil {
        fmt.Printf("Partial errors:\n%v\n", err)
    }
}
```

### Code Example 6: Implementing Unwrap on a Custom Error Type

```go
package main

import (
    "errors"
    "fmt"
    "time"
)

// AppError is an error type used throughout the application
type AppError struct {
    Code       string
    Message    string
    Err        error
    Timestamp  time.Time
    RequestID  string
    StackTrace string // In production, obtain with runtime/debug.Stack(), etc.
}

func NewAppError(code, message string, err error) *AppError {
    return &AppError{
        Code:      code,
        Message:   message,
        Err:       err,
        Timestamp: time.Now(),
    }
}

func (e *AppError) Error() string {
    if e.Err != nil {
        return fmt.Sprintf("[%s] %s: %v", e.Code, e.Message, e.Err)
    }
    return fmt.Sprintf("[%s] %s", e.Code, e.Message)
}

func (e *AppError) Unwrap() error {
    return e.Err
}

func (e *AppError) WithRequestID(id string) *AppError {
    e.RequestID = id
    return e
}

// Error code constants
const (
    ErrCodeNotFound      = "NOT_FOUND"
    ErrCodeUnauthorized  = "UNAUTHORIZED"
    ErrCodeValidation    = "VALIDATION_ERROR"
    ErrCodeInternal      = "INTERNAL_ERROR"
    ErrCodeTimeout       = "TIMEOUT"
    ErrCodeConflict      = "CONFLICT"
    ErrCodeRateLimited   = "RATE_LIMITED"
)

// Mapping to HTTP status codes
func (e *AppError) HTTPStatus() int {
    switch e.Code {
    case ErrCodeNotFound:
        return 404
    case ErrCodeUnauthorized:
        return 401
    case ErrCodeValidation:
        return 400
    case ErrCodeConflict:
        return 409
    case ErrCodeRateLimited:
        return 429
    case ErrCodeTimeout:
        return 504
    default:
        return 500
    }
}

// Struct for JSON responses
type ErrorResponse struct {
    Code      string `json:"code"`
    Message   string `json:"message"`
    RequestID string `json:"request_id,omitempty"`
}

func (e *AppError) ToResponse() ErrorResponse {
    return ErrorResponse{
        Code:      e.Code,
        Message:   e.Message,
        RequestID: e.RequestID,
    }
}

// Convenient constructor functions
var ErrNotFound = errors.New("not found")
var ErrTimeout = errors.New("timeout")

func NotFoundError(resource string, id interface{}) *AppError {
    return NewAppError(ErrCodeNotFound,
        fmt.Sprintf("%s (id=%v) not found", resource, id),
        ErrNotFound)
}

func TimeoutError(operation string, duration time.Duration) *AppError {
    return NewAppError(ErrCodeTimeout,
        fmt.Sprintf("%s timed out after %v", operation, duration),
        ErrTimeout)
}

func ValidationError(field, message string) *AppError {
    return NewAppError(ErrCodeValidation,
        fmt.Sprintf("validation failed: %s - %s", field, message),
        nil)
}

func main() {
    // Usage example of AppError
    err := NotFoundError("User", 42).WithRequestID("req-abc-123")
    fmt.Println(err)
    fmt.Printf("HTTP Status: %d\n", err.HTTPStatus())
    fmt.Printf("Response: %+v\n", err.ToResponse())

    // Unwrap chain
    fmt.Println(errors.Is(err, ErrNotFound)) // true

    // Extracting AppError
    wrapped := fmt.Errorf("handler: %w", err)
    var appErr *AppError
    if errors.As(wrapped, &appErr) {
        fmt.Printf("Code: %s, Message: %s\n", appErr.Code, appErr.Message)
    }

    // Timeout error
    tErr := TimeoutError("database query", 5*time.Second)
    fmt.Println(tErr)
    fmt.Println(errors.Is(tErr, ErrTimeout)) // true
}
```

### Code Example 7: Appropriate Use of panic/recover

```go
package main

import (
    "fmt"
    "log"
    "net/http"
    "runtime/debug"
)

// Situations where panic is appropriate:
// 1. Programming errors (unreachable code)
// 2. Unrecoverable errors during initialization
// 3. Detecting bugs inside a library

// Middleware using recover
func recoveryMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        defer func() {
            if rec := recover(); rec != nil {
                // Log the stack trace
                log.Printf("PANIC recovered: %v\n%s", rec, debug.Stack())

                // Return 500 to the client
                http.Error(w, "Internal Server Error", http.StatusInternalServerError)
            }
        }()
        next.ServeHTTP(w, r)
    })
}

// Use panic internally in a library, but recover in the public API
func parseExpression(expr string) (result float64, err error) {
    defer func() {
        if r := recover(); r != nil {
            err = fmt.Errorf("parse expression %q: %v", expr, r)
        }
    }()

    // Internally, propagate errors concisely with panic
    result = evalExpr(expr)
    return result, nil
}

func evalExpr(expr string) float64 {
    if expr == "" {
        panic("empty expression") // panic only internally
    }
    // Hardcoded for simplicity
    return 42.0
}

// Must pattern: use only in main/init
func MustParseConfig(path string) map[string]string {
    config, err := loadConfig(path)
    if err != nil {
        panic(fmt.Sprintf("failed to load config %q: %v", path, err))
    }
    return config
}

func loadConfig(path string) (map[string]string, error) {
    // Simulation
    return map[string]string{"key": "value"}, nil
}

// assertNever: indicates unreachable code
func processStatus(status string) string {
    switch status {
    case "active":
        return "User is active"
    case "inactive":
        return "User is inactive"
    case "deleted":
        return "User is deleted"
    default:
        // An unknown status is a programming error
        panic(fmt.Sprintf("unexpected status: %q", status))
    }
}

// Panic-safe handling inside a cleanup deferred function
func processFile(path string) (err error) {
    // recover in defer
    defer func() {
        if r := recover(); r != nil {
            err = fmt.Errorf("process file: panic: %v", r)
        }
    }()

    fmt.Printf("Processing %s\n", path)
    // Processing...
    return nil
}

func main() {
    // Must pattern
    config := MustParseConfig("config.json")
    fmt.Printf("Config: %v\n", config)

    // recover in parseExpression
    result, err := parseExpression("1+2")
    if err != nil {
        fmt.Printf("Parse error: %v\n", err)
    } else {
        fmt.Printf("Result: %.1f\n", result)
    }

    result, err = parseExpression("")
    if err != nil {
        fmt.Printf("Parse error: %v\n", err) // panic is converted into an error
    }

    // processStatus
    fmt.Println(processStatus("active"))

    // An invalid status will panic, but we can recover in defer
    func() {
        defer func() {
            if r := recover(); r != nil {
                fmt.Printf("Recovered: %v\n", r)
            }
        }()
        processStatus("unknown")
    }()
}
```

### Code Example 8: Practical Patterns for Error Handling

```go
package main

import (
    "context"
    "errors"
    "fmt"
    "log/slog"
    "time"
)

// errWriter pattern: aggregating errors across consecutive I/O operations
type errWriter struct {
    err error
}

func (ew *errWriter) writeString(s string) {
    if ew.err != nil {
        return // Do nothing after the first error
    }
    fmt.Print(s) // In practice, write to an io.Writer
}

func (ew *errWriter) writef(format string, args ...interface{}) {
    if ew.err != nil {
        return
    }
    fmt.Printf(format, args...)
}

// Retry pattern
type RetryConfig struct {
    MaxRetries int
    BaseDelay  time.Duration
    MaxDelay   time.Duration
}

func retry(ctx context.Context, config RetryConfig, operation func() error) error {
    var lastErr error
    delay := config.BaseDelay

    for attempt := 0; attempt <= config.MaxRetries; attempt++ {
        if attempt > 0 {
            slog.Info("retrying operation",
                "attempt", attempt,
                "delay", delay,
                "last_error", lastErr)

            select {
            case <-time.After(delay):
            case <-ctx.Done():
                return fmt.Errorf("retry cancelled: %w", ctx.Err())
            }

            // Exponential backoff
            delay *= 2
            if delay > config.MaxDelay {
                delay = config.MaxDelay
            }
        }

        lastErr = operation()
        if lastErr == nil {
            return nil
        }

        // Do not retry non-temporary errors
        if !isRetryable(lastErr) {
            return fmt.Errorf("non-retryable error: %w", lastErr)
        }
    }

    return fmt.Errorf("max retries (%d) exceeded: %w", config.MaxRetries, lastErr)
}

func isRetryable(err error) bool {
    // Do not retry on context cancellation
    if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
        return false
    }
    // Check for specific error types
    var tempErr interface{ Temporary() bool }
    if errors.As(err, &tempErr) {
        return tempErr.Temporary()
    }
    return true // By default, treat as retryable
}

// Best practices for error logging
func handleRequest(ctx context.Context, userID int) error {
    logger := slog.With("user_id", userID, "request_id", "req-123")

    user, err := findUser(ctx, userID)
    if err != nil {
        if errors.Is(err, ErrNotFound) {
            // Expected error: INFO level
            logger.Info("user not found", "error", err)
            return err
        }
        // Unexpected error: ERROR level
        logger.Error("failed to find user", "error", err)
        return fmt.Errorf("handle request: %w", err)
    }

    logger.Info("user found", "user_name", user.Name)
    return nil
}

var ErrNotFound = errors.New("not found")

type User struct {
    ID   int
    Name string
}

func findUser(ctx context.Context, id int) (*User, error) {
    if id > 100 {
        return nil, ErrNotFound
    }
    return &User{ID: id, Name: "Alice"}, nil
}

// Error aggregation pattern (concurrent processing)
func fetchAll(ctx context.Context, urls []string) ([]string, error) {
    type result struct {
        url  string
        body string
        err  error
    }

    results := make(chan result, len(urls))

    for _, url := range urls {
        go func(u string) {
            // Simulating HTTP GET
            body := fmt.Sprintf("body of %s", u)
            results <- result{url: u, body: body, err: nil}
        }(url)
    }

    var bodies []string
    var errs []error

    for range urls {
        r := <-results
        if r.err != nil {
            errs = append(errs, fmt.Errorf("fetch %s: %w", r.url, r.err))
        } else {
            bodies = append(bodies, r.body)
        }
    }

    if len(errs) > 0 {
        return bodies, errors.Join(errs...)
    }
    return bodies, nil
}

func main() {
    // errWriter pattern
    ew := &errWriter{}
    ew.writeString("Hello ")
    ew.writef("World %d\n", 42)
    if ew.err != nil {
        fmt.Printf("Write error: %v\n", ew.err)
    }

    // Retry pattern
    ctx := context.Background()
    attempt := 0
    err := retry(ctx, RetryConfig{
        MaxRetries: 3,
        BaseDelay:  10 * time.Millisecond,
        MaxDelay:   100 * time.Millisecond,
    }, func() error {
        attempt++
        if attempt < 3 {
            return fmt.Errorf("temporary error (attempt %d)", attempt)
        }
        return nil // Succeeds on the 3rd try
    })

    if err != nil {
        fmt.Printf("Retry failed: %v\n", err)
    } else {
        fmt.Printf("Succeeded after %d attempts\n", attempt)
    }

    // Concurrent error aggregation
    urls := []string{"http://a.com", "http://b.com", "http://c.com"}
    bodies, err := fetchAll(ctx, urls)
    if err != nil {
        fmt.Printf("Fetch errors: %v\n", err)
    }
    fmt.Printf("Fetched %d bodies\n", len(bodies))
}
```

---

## 2. ASCII Diagrams

### Diagram 1: Error Chain

```
fmt.Errorf("handler: %w",
  fmt.Errorf("service: %w",
    fmt.Errorf("repo: %w",
      ErrNotFound)))

Error chain:
┌─────────────────┐
│ "handler: ..."  │
│   Unwrap() ─────┼──> ┌──────────────────┐
└─────────────────┘    │ "service: ..."   │
                       │   Unwrap() ──────┼──> ┌────────────────┐
                       └──────────────────┘    │ "repo: ..."    │
                                               │   Unwrap() ────┼──> ErrNotFound
                                               └────────────────┘
errors.Is(err, ErrNotFound) → walks the chain and returns true

Error message:
"handler: service: repo: not found"
                         ↑
                 the original sentinel error
```

### Diagram 2: errors.Is vs errors.As

```
┌─────────────────────────────────────────────┐
│              errors.Is(err, target)          │
│  Purpose: check whether a specific error     │
│           value matches                      │
│  Traversal: recursively follows Unwrap()     │
│  Comparison: == or the Is() method           │
│  Returns: bool                                │
│                                              │
│  Use cases:                                    │
│  - Checking for sentinel errors                │
│  - errors.Is(err, ErrNotFound)               │
│  - errors.Is(err, context.Canceled)          │
│  - errors.Is(err, sql.ErrNoRows)            │
├─────────────────────────────────────────────┤
│              errors.As(err, &target)         │
│  Purpose: extract a specific error type      │
│  Traversal: recursively follows Unwrap()     │
│  Comparison: type assertion                  │
│  Returns: bool (target is set to the value)  │
│                                              │
│  Use cases:                                    │
│  - Retrieving details from custom error types  │
│  - var httpErr *HTTPError                    │
│    errors.As(err, &httpErr)                  │
│  - var pathErr *os.PathError                 │
│    errors.As(err, &pathErr)                  │
└─────────────────────────────────────────────┘
```

### Diagram 3: Decision Flow for Error Handling

```
         An error occurs
              │
              ▼
     ┌────────────────┐
     │ What kind of error? │
     └───┬────────┬───┘
         │        │
     Expected   Unexpected
      error      error
         │        │
         ▼        ▼
   ┌──────────┐ ┌──────────────┐
   │ Handle    │ │ Wrap with %w │
   │ properly  │ │ and return   │
   │ (log etc.)│ │ to caller    │
   └──────────┘ └──────────────┘
         │              │
         ▼              ▼
   ┌──────────┐  ┌──────────────┐
   │ Can it   │  │ Add context  │
   │ recover? │  │ and return   │
   │ YES→fix  │  └──────────────┘
   │ NO→return│         │
   └──────────┘         ▼
                  ┌──────────────┐
                  │ Log and      │
                  │ respond at   │
                  │ the top      │
                  └──────────────┘

Layer-specific responsibilities:
┌────────────────────────────────────────┐
│ Handler layer: error type → HTTP status │
│ ├─ ErrNotFound → 404                  │
│ ├─ ErrValidation → 400                │
│ ├─ ErrUnauthorized → 401              │
│ └─ Other → 500                         │
├────────────────────────────────────────┤
│ Service layer: add business-logic context│
│ └─ fmt.Errorf("get user: %w", err)    │
├────────────────────────────────────────┤
│ Repository layer: add data-access context│
│ └─ fmt.Errorf("query users: %w", err) │
├────────────────────────────────────────┤
│ Infrastructure layer: produces low-level errors│
│ └─ sql.ErrNoRows, net.Error, etc.     │
└────────────────────────────────────────┘
```

### Diagram 4: How errors.Join Works (Go 1.20+)

```
errors.Join(err1, err2, err3)

Result:
┌──────────────────────────────────────┐
│  joinError                           │
│  ┌──────┐ ┌──────┐ ┌──────┐        │
│  │ err1 │ │ err2 │ │ err3 │        │
│  └──────┘ └──────┘ └──────┘        │
│                                      │
│  Unwrap() []error → [err1, err2, err3]│
│                                      │
│  errors.Is(joined, err1) → true      │
│  errors.Is(joined, err2) → true      │
│  errors.Is(joined, err3) → true      │
│                                      │
│  Error() → "err1\nerr2\nerr3"       │
└──────────────────────────────────────┘

Using it in validation:
  validate(user) → errors.Join(
    nameErr,    // "name: required"
    emailErr,   // "email: invalid format"
    passErr,    // "password: too short"
  )

  ↓ Individual errors can also be retrieved with errors.As

  var ve *ValidationError
  errors.As(joined, &ve) → true (the first match)
```

### Diagram 5: panic/recover Flow

```
Execution flow of a goroutine:

Normal completion:
  main() → f1() → f2() → return → return → return

Panic occurs:
  main() → f1() → f2() → panic("!!")
                              │
                    ┌─────────▼──────────┐
                    │ Execute the defer   │
                    │ stack in reverse    │
                    │                     │
                    │ f2's defer → run    │
                    │ f1's defer → run    │
                    │ main's defer → run  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ If there is no      │
                    │ recover(), the      │
                    │ program terminates  │
                    └─────────────────────┘

When recover() is present:
  main() → f1() → f2() → panic("!!")
                              │
                    ┌─────────▼──────────┐
                    │ f2's defer:        │
                    │   recover() → "!!" │ ← captures the panic
                    │   convert to err   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ f2 returns normally │
                    │ (returns err)       │
                    └──────────┬──────────┘
                               │
                    f1 and main continue executing normally
```

---

## 3. Comparison Tables

### Table 1: Comparison of Error Handling Approaches

| Approach | Go | Java | Rust | Python | TypeScript |
|-----------|-----|------|------|--------|------------|
| Mechanism | Return value (error) | Exception | Result<T,E> | Exception | Exception + Promise |
| Behavior when unhandled | Compiles fine | Crashes | Compile error | Crashes | Crashes |
| Type information | interface (dynamic) | Class hierarchy | enum (static) | Class hierarchy | any |
| Exhaustiveness check | None | None (except checked) | Yes (match) | None | None |
| Control flow | Explicit if err != nil | try-catch | ? operator | try-except | try-catch + .catch |
| Multiple errors | errors.Join | suppressed | -- | ExceptionGroup | AggregateError |
| Stack trace | None (implement manually) | Automatically attached | None | Automatically attached | Automatically attached |

### Table 2: Comparison of Go Error Patterns

| Pattern | Use case | Example | Benefits | Drawbacks |
|---------|---------|---------|----------|-----------|
| sentinel error | Known error conditions | `ErrNotFound` | Simple, fast comparison | No additional information |
| Custom error type | When additional information is needed | `*ValidationError` | Rich information, type safe | Verbose to define |
| `fmt.Errorf("%w")` | Adding context | `"open config: %w"` | Easy, chain inspectable | Loses type information |
| `errors.Join` | Aggregating multiple errors | Validation | Exhaustive validation possible | Longer error messages |
| panic/recover | Truly unrecoverable state | Programming errors | Concise | Dangerous when abused |
| AppError (structured) | API errors | `{code, msg, err}` | HTTP integration, log integration | Adds complexity |

### Table 3: Criteria for Choosing %w vs %v

| Situation | Format to use | Reason |
|-----------|--------------|--------|
| Propagating an internal error as-is | `%w` | Enables chain inspection |
| Library public API | `%v` | Hides internal implementation details |
| Within the same package | `%w` | Detailed error checks are needed |
| External package boundaries | Depends | Use `%w` only for stable errors |
| Log output | `%v` | Just record as a string |
| Propagating a sentinel error | `%w` | To enable inspection with errors.Is |

---

## 4. Anti-Patterns

### Anti-Pattern 1: Swallowing Errors

```go
// BAD: ignoring the error
result, _ := doSomething()

// BAD: only logging and not handling
if err != nil {
    log.Println(err) // the caller thinks it succeeded
    // no return!
}

// BAD: empty error check
if err != nil {
    // TODO: handle error
}

// GOOD: return the error
result, err := doSomething()
if err != nil {
    return fmt.Errorf("do something: %w", err)
}

// GOOD: add a comment when deliberately ignoring an error
_ = conn.Close() // best-effort close, error is intentionally ignored
```

### Anti-Pattern 2: Redundant Error Messages

```go
// BAD: "failed to" chains become verbose
// Result: "failed to get user: failed to query db: failed to connect: timeout"
func getUser(id int) (*User, error) {
    user, err := queryDB(id)
    if err != nil {
        return nil, fmt.Errorf("failed to get user: %w", err)
    }
    return user, nil
}

func queryDB(id int) (*User, error) {
    conn, err := connect()
    if err != nil {
        return nil, fmt.Errorf("failed to query db: %w", err)
    }
    _ = conn
    return nil, nil
}

// GOOD: add context concisely (omit verbs)
// Result: "get user: query db: connect: timeout"
func getUser(id int) (*User, error) {
    user, err := queryDB(id)
    if err != nil {
        return nil, fmt.Errorf("get user (id=%d): %w", id, err)
    }
    return user, nil
}

func queryDB(id int) (*User, error) {
    conn, err := connect()
    if err != nil {
        return nil, fmt.Errorf("query db: %w", err)
    }
    _ = conn
    return nil, nil
}
```

### Anti-Pattern 3: Double-Handling Errors

```go
// BAD: logging the same error and then returning it
func processOrder(id int) error {
    order, err := findOrder(id)
    if err != nil {
        log.Printf("ERROR: failed to find order: %v", err) // log output
        return fmt.Errorf("find order: %w", err)            // and also return
        // → it will be logged by the caller as well → the same error is recorded twice
    }
    _ = order
    return nil
}

// GOOD: log in only one layer (usually the top)
func processOrder(id int) error {
    order, err := findOrder(id)
    if err != nil {
        return fmt.Errorf("process order: find order (id=%d): %w", id, err)
    }
    _ = order
    return nil
}

// Log in the top-level handler
func handleOrder(w http.ResponseWriter, r *http.Request) {
    if err := processOrder(42); err != nil {
        log.Printf("ERROR: %v", err) // log only here
        http.Error(w, "error", 500)
    }
}
```

### Anti-Pattern 4: Using panic in Place of Error Handling

```go
// BAD: library function panics
func ParseConfig(data []byte) *Config {
    var config Config
    if err := json.Unmarshal(data, &config); err != nil {
        panic(fmt.Sprintf("invalid config: %v", err)) // NG: crashes the caller
    }
    return &config
}

// GOOD: return an error
func ParseConfig(data []byte) (*Config, error) {
    var config Config
    if err := json.Unmarshal(data, &config); err != nil {
        return nil, fmt.Errorf("parse config: %w", err)
    }
    return &config, nil
}

// The Must pattern is limited to main/init
func MustParseConfig(data []byte) *Config {
    config, err := ParseConfig(data)
    if err != nil {
        panic(err)
    }
    return config
}

// Use Must only in main()
func main() {
    config := MustParseConfig(configData)
    _ = config
}
```

### Anti-Pattern 5: Using the Result Before Checking the Error

```go
// BAD: using result before checking err
func process() {
    result, err := fetchData()
    fmt.Println(result.Name) // if err is non-nil, result may be invalid
    if err != nil {
        log.Fatal(err)
    }
}

// GOOD: check the error first
func process() {
    result, err := fetchData()
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println(result.Name) // use after confirming there is no error
}
```

---

## 5. FAQ

### Q1: When should panic be used?

Use panic only for programming errors (nil pointer dereferences, out-of-range access, etc.) or unrecoverable initialization errors. For ordinary business logic, always return an `error`. Libraries must not let panic leak out to their callers.

Situations where panic is appropriate:
1. Configuration loading failures in `main()` or `init()`
2. When program invariants are violated
3. Asserting unreachable code (e.g., `default` cases)
4. Preconditions in test helpers

### Q2: What is the difference between `%w` and `%v`?

`%w` wraps an error and makes chain inspection with `errors.Is`/`errors.As` possible. `%v` simply embeds the error message as a string. As a rule, use `%w`, but use `%v` when you want to hide internal implementation (e.g., in a library's public API).

```go
// %w: the error chain is preserved
err := fmt.Errorf("open: %w", os.ErrNotExist)
errors.Is(err, os.ErrNotExist) // true

// %v: the error chain is broken
err := fmt.Errorf("open: %v", os.ErrNotExist)
errors.Is(err, os.ErrNotExist) // false
```

### Q3: What are the naming conventions for error messages?

Go's conventions: (1) start with a lowercase letter, (2) do not prefix with "failed to", (3) do not prefix with the package name (it is added naturally by wrapping), (4) do not end with punctuation. Example: `"open config file: %w"` is a good form.

```go
// BAD
return fmt.Errorf("Failed to open the config file: %w", err)

// GOOD
return fmt.Errorf("open config: %w", err)

// BAD
return fmt.Errorf("mypackage.ReadConfig: %w", err)

// GOOD
return fmt.Errorf("read config: %w", err)
```

### Q4: What is the difference between errors.Is and ==?

`==` performs a direct comparison only. `errors.Is` walks the error chain to search.

```go
base := errors.New("base error")
wrapped := fmt.Errorf("wrapped: %w", base)

wrapped == base          // false (different objects)
errors.Is(wrapped, base) // true (base is contained in the chain)
```

Also, `errors.Is` calls a custom `Is()` method if one exists. This enables flexible comparisons such as partial value matching.

### Q5: How did error handling change from pre-Go 1.13 to post-Go 1.13?

`errors.Is`, `errors.As`, and `fmt.Errorf("%w")` were introduced in Go 1.13. As a result:

- **Pre-1.13**: Direct comparison with `err == ErrNotFound`. Wrapping made comparison impossible
- **Post-1.13**: `errors.Is(err, ErrNotFound)` searches the entire chain. Detection still works even after wrapping

In Go 1.20, `errors.Join` was added, standardizing the aggregation of multiple errors.

### Q6: Should errors include a stack trace?

Go's standard library is designed not to include stack traces. The reasons are: (1) performance impact, (2) adding context to error messages provides sufficient traceability, (3) combining with structured logging works well.

When a stack trace is needed, these approaches are available:
- Output `runtime/debug.Stack()` to logs
- Use a third-party library (`pkg/errors`, etc.)
- Obtain it via panic/recover (in production, use recovery middleware)
- Tracing via OpenTelemetry

---


## FAQ

### Q1: What is the most important point to focus on when learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory, but by actually writing and running code to see how things work.

### Q2: What are common mistakes beginners make?

Skipping the basics and jumping to advanced topics. We recommend solidly understanding the fundamental concepts explained in this guide before moving on to the next step.

### Q3: How is this applied in real-world development?

The knowledge from this topic is frequently applied in day-to-day development work. It becomes especially important during code reviews and architecture design.

---

## 6. Summary

| Concept | Key Points |
|---------|------------|
| error interface | An interface with `Error() string` |
| sentinel error | Defined as `var ErrXxx = errors.New(...)` |
| Wrapping | `fmt.Errorf("context: %w", err)` |
| errors.Is | Whether the error chain contains a specific error |
| errors.As | Extracts a specific type from the error chain |
| errors.Join | Joins multiple errors (Go 1.20+) |
| panic/recover | Only for unrecoverable errors. Libraries must not let it leak |
| Error messages | Lowercase start, concise, add context, no "failed to" |
| Layered handling | Lower: wrap and return; Upper: identify kind + respond |

---

## Recommended Next Reads

- [03-packages-modules.md](./03-packages-modules.md) -- Packages and modules
- [../02-web/04-testing.md](../02-web/04-testing.md) -- Error verification in testing
- [../03-tools/04-best-practices.md](../03-tools/04-best-practices.md) -- Best practices

---

## References

1. **Go Blog, "Working with Errors in Go 1.13"** -- https://go.dev/blog/go1.13-errors
2. **Go Blog, "Error handling and Go"** -- https://go.dev/blog/error-handling-and-go
3. **Standard library: errors package** -- https://pkg.go.dev/errors
4. **Go Blog, "Errors are values"** -- https://go.dev/blog/errors-are-values
5. **Go Wiki: Errors** -- https://go.dev/wiki/Errors
6. **Dave Cheney, "Don't just check errors, handle them gracefully"** -- https://dave.cheney.net/2016/04/27/dont-just-check-errors-handle-them-gracefully
