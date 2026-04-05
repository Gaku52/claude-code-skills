# Context -- Cancellation, Timeouts, and Value Propagation

> context.Context is the standard mechanism for propagating cancellation signals, timeouts, and request-scoped values across goroutines.

---

## What You Will Learn in This Chapter

1. **context.WithCancel** -- Manual cancellation
2. **context.WithTimeout / WithDeadline** -- Timeout control
3. **context.WithValue** -- Value propagation and best practices
4. **context.AfterFunc (Go 1.21+)** -- Callbacks on cancellation
5. **context.WithoutCancel (Go 1.21+)** -- Severing cancellation propagation
6. **Practical patterns** -- Using Context in HTTP servers, databases, and microservices


## Prerequisites

Before reading this guide, the following knowledge will help deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Familiarity with the content of [Concurrency Patterns -- Fan-out/Fan-in, Pipeline, Worker Pool](./02-concurrency-patterns.md)

---

## 1. Core Concepts of Context

context.Context provides the following four capabilities.

1. **Cancellation propagation**: Cancellation of a parent automatically propagates to all descendants
2. **Deadline management**: Sets time limits on processing
3. **Value propagation**: Passes request-scoped cross-cutting concerns
4. **Done() channel**: Provides a channel for detecting cancellation

### 1.1 Design Principles of Context

- **Pass as the first argument**: Make the first argument of functions `ctx context.Context`
- **Don't store in structs**: Do not hold request-scoped contexts as fields
- **Don't pass nil**: Use `context.TODO()` when unsure
- **Values for cross-cutting concerns only**: Don't put business logic parameters in it
- **Always call the cancel function**: Write `defer cancel()` immediately after obtaining it

---

## 2. context.WithCancel

WithCancel creates a context for manually sending a cancellation signal. When the cancel function is called, all child contexts derived from that context are also cancelled.

### Code Example 1: context.WithCancel Basics

```go
package main

import (
	"context"
	"fmt"
	"time"
)

func main() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Background worker
	go func() {
		for {
			select {
			case <-ctx.Done():
				fmt.Println("cancelled:", ctx.Err())
				return
			default:
				fmt.Println("working...")
				time.Sleep(500 * time.Millisecond)
			}
		}
	}()

	time.Sleep(2 * time.Second)
	cancel() // Notify the goroutine of cancellation
	time.Sleep(100 * time.Millisecond) // Wait for the goroutine to finish
}
```

### Code Example 2: Simultaneous Cancellation of Multiple Goroutines

```go
package main

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// worker is a worker that processes jobs periodically
func worker(ctx context.Context, id int, wg *sync.WaitGroup) {
	defer wg.Done()
	for {
		select {
		case <-ctx.Done():
			fmt.Printf("Worker %d: stopped (reason: %v)\n", id, ctx.Err())
			return
		case <-time.After(200 * time.Millisecond):
			fmt.Printf("Worker %d: processing\n", id)
		}
	}
}

func main() {
	ctx, cancel := context.WithCancel(context.Background())

	var wg sync.WaitGroup
	// Launch 5 workers
	for i := 1; i <= 5; i++ {
		wg.Add(1)
		go worker(ctx, i, &wg)
	}

	// Stop all workers after 1 second
	time.Sleep(1 * time.Second)
	fmt.Println("Cancelling all workers...")
	cancel()

	wg.Wait()
	fmt.Println("All workers stopped")
}
```

### Code Example 3: Conditional Cancellation

```go
package main

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"time"
)

var ErrCriticalFailure = errors.New("critical failure detected")

// monitor watches system state and cancels on anomaly detection
func monitor(ctx context.Context, cancel context.CancelFunc) {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			// Simulation that randomly detects an anomaly
			if rand.Float64() < 0.05 {
				fmt.Println("Monitor: critical failure detected!")
				cancel() // Cancel all processing
				return
			}
		}
	}
}

// processData processes data sequentially
func processData(ctx context.Context) error {
	for i := 0; i < 100; i++ {
		select {
		case <-ctx.Done():
			return fmt.Errorf("processing interrupted at item %d: %w", i, ctx.Err())
		default:
			time.Sleep(50 * time.Millisecond)
			fmt.Printf("Processing item %d\n", i)
		}
	}
	return nil
}

func main() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go monitor(ctx, cancel)

	if err := processData(ctx); err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Println("All items processed successfully")
	}
}
```

---

## 3. context.WithTimeout / WithDeadline

WithTimeout creates a context that is automatically cancelled after a specified duration. WithDeadline specifies a deadline as an absolute point in time. Internally, WithTimeout is a thin wrapper around WithDeadline.

### Code Example 4: context.WithTimeout

```go
package main

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"time"
)

// fetchWithTimeout performs an HTTP request with a timeout
func fetchWithTimeout(url string, timeout time.Duration) ([]byte, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel() // Always call cancel even if completed before the timeout

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("do request: %w", err) // On timeout: context deadline exceeded
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read body: %w", err)
	}
	return body, nil
}

func main() {
	body, err := fetchWithTimeout("https://httpbin.org/delay/2", 5*time.Second)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	fmt.Printf("Response: %d bytes\n", len(body))

	// Case that times out
	_, err = fetchWithTimeout("https://httpbin.org/delay/10", 3*time.Second)
	if err != nil {
		fmt.Printf("Expected timeout error: %v\n", err)
	}
}
```

### Code Example 5: context.WithDeadline

```go
package main

import (
	"context"
	"fmt"
	"time"
)

// processUntilDeadline continues processing until the deadline
func processUntilDeadline(ctx context.Context) (int, error) {
	count := 0
	for {
		select {
		case <-ctx.Done():
			return count, ctx.Err() // context.DeadlineExceeded
		default:
			// Simulate processing a single item
			time.Sleep(100 * time.Millisecond)
			count++
			fmt.Printf("Processed item %d\n", count)
		}
	}
}

func main() {
	// Set the deadline 1 second from now
	deadline := time.Now().Add(1 * time.Second)
	ctx, cancel := context.WithDeadline(context.Background(), deadline)
	defer cancel()

	// Check the deadline
	if d, ok := ctx.Deadline(); ok {
		fmt.Printf("Deadline: %v (in %v)\n", d.Format(time.RFC3339), time.Until(d))
	}

	count, err := processUntilDeadline(ctx)
	fmt.Printf("Processed %d items, error: %v\n", count, err)
}
```

### Code Example 6: Nested Timeouts

```go
package main

import (
	"context"
	"fmt"
	"time"
)

// When the parent's timeout is shorter than the child's, the parent's timeout takes precedence
func demonstrateNestedTimeout() {
	// Parent: 2-second timeout
	parentCtx, parentCancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer parentCancel()

	// Child: 5-second timeout (but effectively 2 seconds because the parent cancels at 2 seconds)
	childCtx, childCancel := context.WithTimeout(parentCtx, 5*time.Second)
	defer childCancel()

	// Grandchild: 1-second timeout (this is the shortest)
	grandchildCtx, grandchildCancel := context.WithTimeout(childCtx, 1*time.Second)
	defer grandchildCancel()

	// Grandchild times out in 1 second
	select {
	case <-grandchildCtx.Done():
		fmt.Printf("Grandchild done: %v\n", grandchildCtx.Err())
	}

	// Child times out at the parent's 2 seconds (not 5 seconds)
	select {
	case <-childCtx.Done():
		fmt.Printf("Child done: %v\n", childCtx.Err())
	}
}

func main() {
	demonstrateNestedTimeout()
}
```

### Code Example 7: Branching Based on Remaining Timeout

```go
package main

import (
	"context"
	"fmt"
	"time"
)

// adaptiveProcess changes its processing approach based on the remaining timeout
func adaptiveProcess(ctx context.Context) error {
	deadline, ok := ctx.Deadline()
	if !ok {
		// No deadline set
		return fullProcess(ctx)
	}

	remaining := time.Until(deadline)
	fmt.Printf("Remaining time: %v\n", remaining)

	if remaining < 1*time.Second {
		// Little remaining time -> simplified processing
		return quickProcess(ctx)
	} else if remaining < 5*time.Second {
		// Moderate remaining time -> standard processing
		return standardProcess(ctx)
	} else {
		// Ample remaining time -> full processing
		return fullProcess(ctx)
	}
}

func quickProcess(ctx context.Context) error {
	fmt.Println("Quick process (minimal)")
	return nil
}

func standardProcess(ctx context.Context) error {
	fmt.Println("Standard process")
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(2 * time.Second):
		return nil
	}
}

func fullProcess(ctx context.Context) error {
	fmt.Println("Full process (comprehensive)")
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(5 * time.Second):
		return nil
	}
}

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	if err := adaptiveProcess(ctx); err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Println("Done")
	}
}
```

---

## 4. context.WithValue

WithValue stores request-scoped values in a context. However, you should only store cross-cutting concerns (trace IDs, authentication information, locale, etc.), not business logic parameters.

### Code Example 8: context.WithValue Basics

```go
package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
)

// Define a custom key type to prevent key collisions
type contextKey string

const (
	requestIDKey contextKey = "requestID"
	userIDKey    contextKey = "userID"
	localeKey    contextKey = "locale"
)

// requestIDMiddleware sets the request ID on the context
func requestIDMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reqID := r.Header.Get("X-Request-ID")
		if reqID == "" {
			reqID = generateRequestID() // Generate UUID
		}
		ctx := context.WithValue(r.Context(), requestIDKey, reqID)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// authMiddleware sets the user ID on the context
func authMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		token := r.Header.Get("Authorization")
		userID, err := validateToken(token)
		if err != nil {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		ctx := context.WithValue(r.Context(), userIDKey, userID)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// localeMiddleware sets locale information on the context
func localeMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		locale := r.Header.Get("Accept-Language")
		if locale == "" {
			locale = "ja-JP"
		}
		ctx := context.WithValue(r.Context(), localeKey, locale)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// handler retrieves values from the context
func handler(w http.ResponseWriter, r *http.Request) {
	reqID, _ := r.Context().Value(requestIDKey).(string)
	userID, _ := r.Context().Value(userIDKey).(int)
	locale, _ := r.Context().Value(localeKey).(string)

	log.Printf("[%s] User %d, Locale: %s", reqID, userID, locale)
	fmt.Fprintf(w, "Hello, user %d!", userID)
}

func generateRequestID() string {
	return "req-12345" // In practice, generate a UUID
}

func validateToken(token string) (int, error) {
	return 42, nil // In practice, validate the token
}

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/profile", handler)

	h := requestIDMiddleware(authMiddleware(localeMiddleware(mux)))
	http.ListenAndServe(":8080", h)
}
```

### Code Example 9: Type-Safe Context Value Accessors

```go
package main

import (
	"context"
	"errors"
	"fmt"
)

// --- Key definitions ---

type contextKey int

const (
	requestIDKey contextKey = iota
	userIDKey
	traceIDKey
	tenantIDKey
)

// --- Type-safe accessors ---

// SetRequestID sets the request ID on the context
func SetRequestID(ctx context.Context, id string) context.Context {
	return context.WithValue(ctx, requestIDKey, id)
}

// GetRequestID retrieves the request ID from the context
func GetRequestID(ctx context.Context) (string, bool) {
	id, ok := ctx.Value(requestIDKey).(string)
	return id, ok
}

// MustGetRequestID retrieves the request ID (panics if not present)
func MustGetRequestID(ctx context.Context) string {
	id, ok := GetRequestID(ctx)
	if !ok {
		panic("requestID not found in context")
	}
	return id
}

// SetUserID sets the user ID on the context
func SetUserID(ctx context.Context, id int) context.Context {
	return context.WithValue(ctx, userIDKey, id)
}

// GetUserID retrieves the user ID from the context
func GetUserID(ctx context.Context) (int, error) {
	id, ok := ctx.Value(userIDKey).(int)
	if !ok {
		return 0, errors.New("userID not found in context")
	}
	return id, nil
}

// SetTraceID sets the trace ID on the context
func SetTraceID(ctx context.Context, id string) context.Context {
	return context.WithValue(ctx, traceIDKey, id)
}

// GetTraceID retrieves the trace ID from the context
func GetTraceID(ctx context.Context) string {
	id, _ := ctx.Value(traceIDKey).(string)
	return id // Empty string by default
}

func main() {
	ctx := context.Background()
	ctx = SetRequestID(ctx, "req-abc-123")
	ctx = SetUserID(ctx, 42)
	ctx = SetTraceID(ctx, "trace-xyz-789")

	reqID, _ := GetRequestID(ctx)
	userID, _ := GetUserID(ctx)
	traceID := GetTraceID(ctx)

	fmt.Printf("RequestID: %s, UserID: %d, TraceID: %s\n", reqID, userID, traceID)
}
```

---

## 5. Context Propagation Chains

In real web applications, the HTTP request's context serves as the starting point, propagating down through the service layer, repository layer, and external API calls.

### Code Example 10: Complete Propagation Chain from HTTP Request to Database

```go
package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"
)

// --- Handler layer ---

func handleGetUser(userService *UserService) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Use the HTTP request's Context as the foundation
		ctx := r.Context()

		// Add a handler-specific timeout
		ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
		defer cancel()

		userID := r.PathValue("id")
		user, err := userService.GetUser(ctx, userID)
		if err != nil {
			switch {
			case err == context.Canceled:
				// Client disconnected
				log.Printf("Client disconnected: %v", err)
				return
			case err == context.DeadlineExceeded:
				// Timeout
				http.Error(w, "request timeout", http.StatusGatewayTimeout)
				return
			default:
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(user)
	}
}

// --- Service layer ---

type User struct {
	ID    string `json:"id"`
	Name  string `json:"name"`
	Email string `json:"email"`
}

type UserService struct {
	repo         *UserRepository
	cacheService *CacheService
}

func (s *UserService) GetUser(ctx context.Context, id string) (*User, error) {
	// Try the cache first
	user, err := s.cacheService.Get(ctx, "user:"+id)
	if err == nil && user != nil {
		return user, nil
	}

	// Fetch from the database
	user, err = s.repo.FindByID(ctx, id)
	if err != nil {
		return nil, fmt.Errorf("find user: %w", err)
	}

	// Save to cache (only if the context is still valid)
	if ctx.Err() == nil {
		_ = s.cacheService.Set(ctx, "user:"+id, user, 5*time.Minute)
	}

	return user, nil
}

// --- Repository layer ---

type UserRepository struct {
	db *sql.DB
}

func (r *UserRepository) FindByID(ctx context.Context, id string) (*User, error) {
	var user User
	err := r.db.QueryRowContext(ctx,
		"SELECT id, name, email FROM users WHERE id = $1", id,
	).Scan(&user.ID, &user.Name, &user.Email)
	if err != nil {
		return nil, fmt.Errorf("query user: %w", err)
	}
	return &user, nil
}

// --- Cache layer ---

type CacheService struct{}

func (c *CacheService) Get(ctx context.Context, key string) (*User, error) {
	// Redis GET with context
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
		return nil, fmt.Errorf("cache miss") // Simulated
	}
}

func (c *CacheService) Set(ctx context.Context, key string, user *User, ttl time.Duration) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
		return nil // Simulated
	}
}

func main() {
	// Omitted: DB connection, server startup
	log.Println("Server starting on :8080")
}
```

### Code Example 11: Context Propagation Between Microservices

```go
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// Propagating from context to HTTP headers
func propagateContext(ctx context.Context, req *http.Request) {
	// Propagate the trace ID into HTTP headers
	if traceID := GetTraceID(ctx); traceID != "" {
		req.Header.Set("X-Trace-ID", traceID)
	}

	// Also propagate the request ID
	if reqID, ok := GetRequestID(ctx); ok {
		req.Header.Set("X-Request-ID", reqID)
	}

	// Propagate the deadline via a header (optional)
	if deadline, ok := ctx.Deadline(); ok {
		remaining := time.Until(deadline)
		req.Header.Set("X-Timeout-Ms", fmt.Sprintf("%d", remaining.Milliseconds()))
	}
}

// callExternalService calls an external service
func callExternalService(ctx context.Context, url string) (map[string]interface{}, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, err
	}

	// Propagate context information into HTTP headers
	propagateContext(ctx, req)

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("call %s: %w", url, err)
	}
	defer resp.Body.Close()

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	return result, nil
}

func main() {
	ctx := context.Background()
	ctx = SetTraceID(ctx, "trace-abc-123")
	ctx = SetRequestID(ctx, "req-xyz-789")

	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	result, err := callExternalService(ctx, "https://httpbin.org/get")
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	fmt.Printf("Result: %v\n", result)
}
```

---

## 6. context.AfterFunc (Go 1.21+)

`context.AfterFunc`, added in Go 1.21, runs a callback function after a context is cancelled. It is used for resource cleanup and notifications.

### Code Example 12: context.AfterFunc

```go
package main

import (
	"context"
	"fmt"
	"time"
)

func main() {
	ctx, cancel := context.WithCancel(context.Background())

	// Run cleanup when the context is cancelled
	stop := context.AfterFunc(ctx, func() {
		fmt.Println("AfterFunc: context was cancelled, cleaning up...")
		// Resource cleanup processing
		closeConnections()
		flushLogs()
	})

	// The return value of AfterFunc can be used to unregister
	_ = stop // Calling stop() can unregister the AfterFunc

	// Execute processing
	fmt.Println("Processing...")
	time.Sleep(1 * time.Second)

	// Cancel -> AfterFunc runs
	cancel()
	time.Sleep(100 * time.Millisecond) // Wait for AfterFunc to run
}

func closeConnections() {
	fmt.Println("  Connections closed")
}

func flushLogs() {
	fmt.Println("  Logs flushed")
}
```

### Code Example 13: Resource Release Pattern with AfterFunc

```go
package main

import (
	"context"
	"fmt"
	"sync"
)

// Resource is a resource automatically released on cancellation
type Resource struct {
	name   string
	mu     sync.Mutex
	closed bool
}

func (r *Resource) Close() {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return
	}
	r.closed = true
	fmt.Printf("Resource %s: closed\n", r.name)
}

func (r *Resource) Use() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return fmt.Errorf("resource %s is closed", r.name)
	}
	fmt.Printf("Resource %s: used\n", r.name)
	return nil
}

// acquireResource acquires a resource tied to the context
func acquireResource(ctx context.Context, name string) *Resource {
	r := &Resource{name: name}

	// Automatically release the resource when the context is cancelled
	context.AfterFunc(ctx, func() {
		r.Close()
	})

	return r
}

func main() {
	ctx, cancel := context.WithCancel(context.Background())

	r1 := acquireResource(ctx, "db-conn")
	r2 := acquireResource(ctx, "cache-conn")

	r1.Use()
	r2.Use()

	// Cancel -> all resources are released automatically
	cancel()

	// Access after release is an error
	if err := r1.Use(); err != nil {
		fmt.Printf("Expected error: %v\n", err)
	}
}
```

---

## 7. context.WithoutCancel (Go 1.21+)

`context.WithoutCancel`, added in Go 1.21, creates a new context that inherits the parent's values but does not propagate cancellation signals. It is useful for background processing and cleanup tasks.

### Code Example 14: context.WithoutCancel

```go
package main

import (
	"context"
	"fmt"
	"time"
)

func main() {
	// Parent context: 1-second timeout
	parentCtx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	// Set a value
	parentCtx = SetTraceID(parentCtx, "trace-abc")

	// WithoutCancel: does not propagate the parent's cancellation
	backgroundCtx := context.WithoutCancel(parentCtx)

	// Values are inherited
	fmt.Printf("TraceID in background: %s\n", GetTraceID(backgroundCtx))

	// Unaffected even when the parent times out
	time.Sleep(2 * time.Second)

	if parentCtx.Err() != nil {
		fmt.Printf("Parent: cancelled (%v)\n", parentCtx.Err())
	}
	if backgroundCtx.Err() == nil {
		fmt.Println("Background: still active!")
	}
}
```

### Code Example 15: Practical Use of WithoutCancel

```go
package main

import (
	"context"
	"fmt"
	"log"
	"time"
)

// handleRequest handles an HTTP request
func handleRequest(ctx context.Context) {
	// Main processing (tied to the request context)
	result, err := processRequest(ctx)
	if err != nil {
		log.Printf("Error: %v", err)
		return
	}

	// Asynchronous background task (unaffected by the request context's cancellation)
	bgCtx := context.WithoutCancel(ctx)
	// However, set its own timeout
	bgCtx, bgCancel := context.WithTimeout(bgCtx, 30*time.Second)

	go func() {
		defer bgCancel()
		// Record audit log (should complete even after the request ends)
		writeAuditLog(bgCtx, result)
		// Send metrics
		sendMetrics(bgCtx, result)
	}()

	fmt.Println("Request handled, background tasks started")
}

func processRequest(ctx context.Context) (string, error) {
	select {
	case <-ctx.Done():
		return "", ctx.Err()
	case <-time.After(100 * time.Millisecond):
		return "result-data", nil
	}
}

func writeAuditLog(ctx context.Context, data string) {
	select {
	case <-ctx.Done():
		log.Printf("Audit log write cancelled: %v", ctx.Err())
	case <-time.After(500 * time.Millisecond):
		log.Printf("Audit log written: %s", data)
	}
}

func sendMetrics(ctx context.Context, data string) {
	select {
	case <-ctx.Done():
		log.Printf("Metrics send cancelled: %v", ctx.Err())
	case <-time.After(200 * time.Millisecond):
		log.Printf("Metrics sent: %s", data)
	}
}

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	handleRequest(ctx)
	time.Sleep(1 * time.Second) // Wait for background tasks to complete
}
```

---

## 8. context.WithCancelCause (Go 1.20+)

`context.WithCancelCause`, added in Go 1.20, creates a context that allows attaching a cancellation cause. It is useful for debugging and error reporting.

### Code Example 16: context.WithCancelCause

```go
package main

import (
	"context"
	"errors"
	"fmt"
	"time"
)

var (
	ErrUserAborted    = errors.New("user aborted the operation")
	ErrResourceLimit  = errors.New("resource limit exceeded")
	ErrHealthCheck    = errors.New("health check failed")
)

func main() {
	ctx, cancel := context.WithCancelCause(context.Background())

	go func() {
		// Cancel under some condition
		time.Sleep(1 * time.Second)
		cancel(ErrResourceLimit) // Cancel with a cause
	}()

	<-ctx.Done()

	// Retrieve the cancellation cause
	fmt.Printf("Context error: %v\n", ctx.Err())           // context canceled
	fmt.Printf("Cancel cause: %v\n", context.Cause(ctx))   // resource limit exceeded

	// Branch based on the cause
	cause := context.Cause(ctx)
	switch {
	case errors.Is(cause, ErrUserAborted):
		fmt.Println("User chose to abort")
	case errors.Is(cause, ErrResourceLimit):
		fmt.Println("Resource limit reached, retry later")
	case errors.Is(cause, ErrHealthCheck):
		fmt.Println("System unhealthy, alerting")
	default:
		fmt.Printf("Unknown cause: %v\n", cause)
	}
}
```

---

## 9. Graceful Shutdown and context

Context plays a crucial role in the graceful shutdown of HTTP servers.

### Code Example 17: Graceful Shutdown

```go
package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"
)

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, "OK")
	})
	mux.HandleFunc("GET /slow", func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
			log.Println("Client disconnected during slow request")
			return
		case <-time.After(10 * time.Second):
			fmt.Fprintf(w, "Done after 10s")
		}
	})

	server := &http.Server{
		Addr:         ":8080",
		Handler:      mux,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 15 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	// Start the server in the background
	go func() {
		log.Printf("Server starting on %s", server.Addr)
		if err := server.ListenAndServe(); err != http.ErrServerClosed {
			log.Fatalf("Server error: %v", err)
		}
	}()

	// Wait for a signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	sig := <-quit
	log.Printf("Received signal: %v, shutting down...", sig)

	// Graceful shutdown: wait up to 30 seconds for in-flight requests to complete
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer shutdownCancel()

	if err := server.Shutdown(shutdownCtx); err != nil {
		log.Printf("Graceful shutdown failed: %v", err)
		// Force termination
		server.Close()
	}

	log.Println("Server stopped")
}
```

### Code Example 18: Graceful Shutdown with Background Workers

```go
package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"
)

// Application manages the lifecycle of the entire application
type Application struct {
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

func NewApplication() *Application {
	ctx, cancel := context.WithCancel(context.Background())
	return &Application{ctx: ctx, cancel: cancel}
}

// StartWorker starts a background worker
func (app *Application) StartWorker(name string, fn func(context.Context)) {
	app.wg.Add(1)
	go func() {
		defer app.wg.Done()
		log.Printf("Worker %s: started", name)
		fn(app.ctx)
		log.Printf("Worker %s: stopped", name)
	}()
}

// Shutdown shuts down the application
func (app *Application) Shutdown(timeout time.Duration) {
	log.Println("Application: initiating shutdown")
	app.cancel()

	done := make(chan struct{})
	go func() {
		app.wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		log.Println("Application: all workers stopped gracefully")
	case <-time.After(timeout):
		log.Println("Application: shutdown timeout, some workers may not have stopped")
	}
}

func main() {
	app := NewApplication()

	// Message processing worker
	app.StartWorker("message-processor", func(ctx context.Context) {
		ticker := time.NewTicker(1 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				log.Println("Processing messages...")
			}
		}
	})

	// Metrics collection worker
	app.StartWorker("metrics-collector", func(ctx context.Context) {
		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				log.Println("Collecting metrics...")
			}
		}
	})

	// Health check worker
	app.StartWorker("health-checker", func(ctx context.Context) {
		ticker := time.NewTicker(10 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				log.Println("Health check passed")
			}
		}
	})

	// Wait for signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	app.Shutdown(10 * time.Second)
}
```

---

## 10. ASCII Diagrams

### Diagram 1: Propagation of a Context Tree

```
context.Background()
    │
    ├── WithCancel ──────────── API Handler
    │       │
    │       ├── WithTimeout(5s) ── DB Query
    │       │
    │       └── WithTimeout(3s) ── External API Call
    │
    └── WithCancel ──────────── Background Worker
            │
            └── WithValue(traceID) ── Logger

Cancellation propagation: parent cancels -> all children are also cancelled
```

### Diagram 2: Cancellation Propagation Flow

```
     Client disconnects
          │
          ▼
  ┌───────────────┐
  │ HTTP Handler  │ ctx.Done() signal
  │  (ctx)        │──────┐
  └───────────────┘      │
          │              ▼
          ▼        ┌───────────┐
  ┌───────────────┐│ Service   │
  │ Middleware    ││ (ctx)     │──┐
  │  (ctx)        ││           │  │
  └───────────────┘└───────────┘  ▼
                            ┌───────────┐
                            │ DB Query  │
                            │ (ctx)     │ <- gets cancelled
                            └───────────┘
```

### Diagram 3: Internal Behavior of WithTimeout

```
t=0s          t=3s          t=5s
 │             │             │
 ├─ ctx created┤             │
 │  Timeout=5s │             │
 │             │             │
 │ processing..│ processing..│ ctx.Done()
 │             │             │ <- signal sent
 │             │             │
 │             │          ctx.Err() =
 │             │          DeadlineExceeded
 │             │
 │  Can finish │
 │  early with │
 │  cancel()   │
```

### Diagram 4: Nested Timeouts

```
t=0s    t=1s    t=2s    t=3s    t=4s    t=5s
 │       │       │       │       │       │
 │  ┌────┼───────┼───────┼───────┼───────┤ Parent: Timeout=5s
 │  │    │       │       │       │       │
 │  │ ┌──┼───────┤       │       │       │ Child: Timeout=2s
 │  │ │  │       │       │       │       │
 │  │ │┌─┤       │       │       │       │ Grandchild: Timeout=1s
 │  │ ││ │       │       │       │       │
 │  │ │└─┘ Done  │       │       │       │ Grandchild: times out at 1s
 │  │ └── Done   │       │       │       │ Child: times out at 2s
 │  └──────────── Done   │       │       │ Parent: times out at 5s
 │       │       │       │       │       │
 In practice: cancellation order is grandchild(1s) -> child(2s) -> parent(5s)
```

### Diagram 5: Behavior of WithoutCancel

```
  parent (WithTimeout 5s)
      │
      ├── child1 (normal)
      │     └── Cancelled when parent is cancelled (yes)
      │
      └── child2 (WithoutCancel)
            └── Continues even when parent is cancelled (no cancellation propagation)
            └── Values are inherited (yes)

  Use cases: background tasks, audit log recording, metrics sending
```

### Diagram 6: Using WithCancelCause

```
  ctx, cancel := context.WithCancelCause(parent)
      │
      ├── cancel(ErrUserAborted)
      │     └── context.Cause(ctx) -> ErrUserAborted
      │
      ├── cancel(ErrResourceLimit)
      │     └── context.Cause(ctx) -> ErrResourceLimit
      │
      └── cancel(nil)
            └── context.Cause(ctx) -> context.Canceled

  Useful for debugging and error reporting
```

---

## 11. Comparison Tables

### Table 1: Context Creation Functions

| Function | Purpose | Trigger for Done() | Go Version |
|----------|---------|--------------------|------------|
| `context.Background()` | Root. Used in main, init, tests | Never fires | 1.7+ |
| `context.TODO()` | Temporary placeholder when undecided | Never fires | 1.7+ |
| `WithCancel(parent)` | Manual cancellation | cancel() call | 1.7+ |
| `WithCancelCause(parent)` | Cancellation with cause | cancel(err) call | 1.20+ |
| `WithTimeout(parent, d)` | Time limit | After duration d or cancel() | 1.7+ |
| `WithDeadline(parent, t)` | Absolute time limit | Reaching t or cancel() | 1.7+ |
| `WithValue(parent, k, v)` | Value propagation | Depends on parent | 1.7+ |
| `WithoutCancel(parent)` | Non-propagating cancellation | Never fires | 1.21+ |
| `AfterFunc(ctx, fn)` | Callback on cancellation | - | 1.21+ |

### Table 2: Return Values of context.Err()

| State | ctx.Err() | ctx.Done() | context.Cause(ctx) |
|-------|-----------|-----------|-------------------|
| Not cancelled | nil | blocks | nil |
| cancel() called | context.Canceled | closed | cancel argument or Canceled |
| Timed out | context.DeadlineExceeded | closed | DeadlineExceeded |

### Table 3: What to Put / Not Put in Context Values

| Should put in | Should NOT put in |
|---------------|-------------------|
| Trace ID / Span ID | User ID (pass as argument) |
| Request ID | Search criteria / filters |
| Auth tokens / claims | Pagination information |
| Locale / timezone | Business logic parameters |
| Logger instance | DB connection / HTTP client |
| Tenant ID (multi-tenant) | Configuration values / flags |

### Table 4: Propagation Targets for Context

| Target | Method example | Importance |
|--------|----------------|------------|
| database/sql | QueryContext, ExecContext | Required |
| net/http | NewRequestWithContext | Required |
| gRPC | Automatic propagation via metadata | Automatic |
| Redis | client.WithContext | Recommended |
| Logs | logger.WithContext | Recommended |
| External APIs | NewRequestWithContext | Required |
| goroutines | Pass as argument | Required |

---

## 12. Anti-Patterns

### Anti-Pattern 1: Storing context in a struct

```go
// BAD: Making context a field of a struct
type Service struct {
	ctx context.Context // Cannot hold a different context per request
	db  *sql.DB
}

// GOOD: Pass context as the first argument of a method
type Service struct {
	db *sql.DB
}

func (s *Service) GetUser(ctx context.Context, id int) (*User, error) {
	return s.db.QueryRowContext(ctx, "SELECT ...", id).Scan(...)
}
```

### Anti-Pattern 2: Abusing WithValue

```go
// BAD: Putting business logic parameters in context
ctx = context.WithValue(ctx, "userID", 42)
ctx = context.WithValue(ctx, "orderID", 100)
ctx = context.WithValue(ctx, "limit", 50)

// GOOD: Pass them as function arguments; context is for cross-cutting concerns only
func GetOrders(ctx context.Context, userID, limit int) ([]Order, error) {
	// Only cross-cutting concerns like trace ID and auth info in context
	traceID := ctx.Value(traceIDKey).(string)
	// ...
}
```

### Anti-Pattern 3: Not calling the cancel function

```go
// BAD: Not calling cancel function -> resource leak
func processRequest(parentCtx context.Context) {
	ctx, _ := context.WithTimeout(parentCtx, 5*time.Second)
	// cancel is not called -> timer goroutine leaks
	doWork(ctx)
}

// GOOD: Write defer cancel() immediately
func processRequest(parentCtx context.Context) {
	ctx, cancel := context.WithTimeout(parentCtx, 5*time.Second)
	defer cancel() // Always release the resource
	doWork(ctx)
}
```

### Anti-Pattern 4: Using string keys

```go
// BAD: Using strings as keys (collision risk)
ctx = context.WithValue(ctx, "userID", 42)
ctx = context.WithValue(ctx, "userID", "conflict!") // Another package might use the same key

// GOOD: Use a custom unexported type as the key
type contextKey int

const userIDKey contextKey = 0

ctx = context.WithValue(ctx, userIDKey, 42)
```

### Anti-Pattern 5: Overusing context.Background()

```go
// BAD: Ignoring the parent context and using Background
func (s *Service) GetUser(ctx context.Context, id int) (*User, error) {
	// Ignores the argument ctx and uses Background -> cancellation/timeout don't work
	dbCtx := context.Background()
	return s.db.QueryRowContext(dbCtx, "SELECT ...", id).Scan(...)
}

// GOOD: Propagate the received context as-is
func (s *Service) GetUser(ctx context.Context, id int) (*User, error) {
	return s.db.QueryRowContext(ctx, "SELECT ...", id).Scan(...)
}
```

### Anti-Pattern 6: Using resources after context cancellation

```go
// BAD: Writing to the response after context is cancelled
func handler(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	result, err := longRunningTask(ctx)
	if err != nil {
		// Writes to ResponseWriter even if the context was cancelled
		http.Error(w, err.Error(), 500) // Meaningless if the client disconnected
		return
	}
	json.NewEncoder(w).Encode(result)
}

// GOOD: Check the context state before responding
func handler(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	result, err := longRunningTask(ctx)
	if ctx.Err() != nil {
		// Client has already disconnected -> no need to respond
		log.Printf("Client disconnected: %v", ctx.Err())
		return
	}
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	json.NewEncoder(w).Encode(result)
}
```


---

## Practical Exercises

### Exercise 1: Basic Implementation

Implement code that satisfies the following requirements.

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
        """Validate input value"""
        if value is None:
            raise ValueError("Input value is None")
        return True

    def process(self, value):
        """Main data processing logic"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """Get processing results"""
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
        assert False, "Should have raised an exception"
    except ValueError:
        pass

    print("All tests passed!")

test_exercise1()
```

### Exercise 2: Advanced Patterns

Extend the basic implementation by adding the following features.

```python
# Exercise 2: Advanced patterns
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """Exercise for advanced patterns"""

    def __init__(self, max_size: int = 100):
        self._items: List[Dict] = []
        self._max_size = max_size
        self._created_at = datetime.now()

    def add(self, key: str, value: any) -> bool:
        """Add item (with size limit)"""
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
        """Get statistics"""
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

    print(f"Inefficient version: {slow_time:.4f}s")
    print(f"Efficient version:   {fast_time:.6f}s")
    print(f"Speedup: {slow_time/fast_time:.0f}x")

benchmark()
```

**Key points:**
- Be aware of algorithmic complexity
- Choose appropriate data structures
- Measure the effect with benchmarks

---

## Troubleshooting

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| Initialization error | Configuration file issue | Check the config file path and format |
| Timeout | Network latency / resource shortage | Adjust timeout values, add retry logic |
| Out of memory | Data volume growth | Introduce batch processing, implement pagination |
| Permission error | Insufficient access rights | Check the executing user's permissions and settings |
| Data inconsistency | Concurrency contention | Introduce locking, manage transactions |

### Debugging Steps

1. **Check error messages**: Read the stack trace and identify the location of the issue
2. **Establish reproduction steps**: Reproduce the error with minimal code
3. **Formulate hypotheses**: List possible causes
4. **Verify step by step**: Use logging and debuggers to test hypotheses
5. **Fix and regression test**: After the fix, run tests in related areas as well

```python
# Debugging utilities
import logging
import traceback
from functools import wraps

# Logger configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def debug_decorator(func):
    """Decorator that logs function input and output"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Call: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Return value: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"Exception: {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

@debug_decorator
def process_data(items):
    """Data processing (debug target)"""
    if not items:
        raise ValueError("Empty data")
    return [item * 2 for item in items]
```

### Diagnosing Performance Issues

Steps to take when diagnosing performance problems:

1. **Identify bottlenecks**: Measure with profiling tools
2. **Check memory usage**: Look for memory leaks
3. **Check I/O waits**: Examine disk and network I/O status
4. **Check concurrent connections**: Inspect connection pool state

| Problem type | Diagnostic tools | Countermeasures |
|--------------|------------------|-----------------|
| CPU load | cProfile, py-spy | Algorithmic improvements, parallelization |
| Memory leaks | tracemalloc, objgraph | Proper release of references |
| I/O bottlenecks | strace, iostat | Async I/O, caching |
| DB latency | EXPLAIN, slow query log | Indexing, query optimization |
---

## 13. FAQ

### Q1: What is the difference between context.Background() and context.TODO()?

They are functionally identical, but their intent differs. `Background()` is used when you intentionally want a root context, while `TODO()` is used when the appropriate context is not yet decided and you plan to fix it later. Linters can detect `TODO()` to prevent oversights.

### Q2: Must the cancel function always be called?

Yes. Failing to call the cancel function returned by `WithCancel`/`WithTimeout`/`WithDeadline` causes a resource leak. It is conventional to write `defer cancel()` immediately after obtaining it. Even if cancellation happens automatically via timeout, `cancel()` can be called safely (it can be called multiple times without error).

### Q3: In what situations should context values be used?

Only for request-scoped cross-cutting concerns: trace IDs, authentication information, locale, etc. Do not use them for business logic parameters. Define a custom key type (`type contextKey string`) to prevent key collisions.

### Q4: Should I use context.WithTimeout or http.Client.Timeout?

The two are complementary. `http.Client.Timeout` sets a timeout for the entire client (from connection through response read). `context.WithTimeout` allows setting different timeouts per request and also supports cancellation propagation. Generally, you set both: `http.Client.Timeout` as a longer safety net, and the context timeout to a request-specific value.

### Q5: What happens if the context is cancelled during a database transaction?

`database/sql` detects context cancellation and aborts the query. However, transaction state is driver-dependent. Generally:
- In-flight queries are aborted
- If not yet committed, the transaction is rolled back
- Use the `defer tx.Rollback()` pattern to clean up safely

### Q6: What should I be careful of when passing context to goroutines?

- Copy values before passing them to goroutines (be careful with objects that become invalid after the request ends, such as gin.Context)
- Consider `context.WithoutCancel` for background goroutines
- Set your own timeouts
- To prevent goroutine leaks, always ensure a cancellation path

### Q7: How do I choose between Go 1.21's AfterFunc and WithoutCancel?

- `AfterFunc`: When you want to run specific cleanup (resource release, logging, etc.) on cancellation
- `WithoutCancel`: When you want processing to continue unaffected by the parent's cancellation (background tasks, audit logs, etc.)

### Q8: How should context be handled in tests?

In tests, set an appropriate timeout on `context.Background()`. Since tests with long deadlines can cause flaky (unstable) tests in CI environments, set short timeouts:

```go
func TestGetUser(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	user, err := service.GetUser(ctx, 1)
	if err != nil {
		t.Fatalf("GetUser: %v", err)
	}
	// ...
}
```

---


## FAQ

### Q1: What is the most important point to focus on when learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory, but by actually writing code and verifying its behavior.

### Q2: What are common mistakes beginners make?

Skipping the basics and jumping to advanced topics. We recommend solidly understanding the fundamental concepts explained in this guide before moving on to the next step.

### Q3: How is this applied in real-world development?

The knowledge from this topic is frequently applied in day-to-day development work. It becomes especially important during code reviews and architecture design.

---

## Summary

| Concept | Key Points |
|---------|------------|
| Context | Cancellation, timeouts, and value propagation across goroutines |
| WithCancel | Manual cancellation control |
| WithCancelCause | Cancellation with cause (Go 1.20+) |
| WithTimeout | Processing with time limits |
| WithDeadline | Deadline based on absolute time |
| WithValue | Use only for propagating cross-cutting concerns |
| WithoutCancel | Severs cancellation propagation (Go 1.21+) |
| AfterFunc | Callback on cancellation (Go 1.21+) |
| cancel() | Always call via defer cancel() |
| Propagation | Parent cancellation -> cancellation propagates to all descendants |

---

## Recommended Next Reads

- [../02-web/00-net-http.md](../02-web/00-net-http.md) -- Using Context in HTTP servers
- [../02-web/02-database.md](../02-web/02-database.md) -- Context control in DB queries
- [../02-web/03-grpc.md](../02-web/03-grpc.md) -- Context in gRPC

---

## References

1. **Go Blog, "Go Concurrency Patterns: Context"** -- https://go.dev/blog/context
2. **Go Standard Library: context** -- https://pkg.go.dev/context
3. **Go Blog, "Contexts and structs"** -- https://go.dev/blog/context-and-structs
4. **Go 1.20 Release Notes (WithCancelCause)** -- https://go.dev/doc/go1.20
5. **Go 1.21 Release Notes (AfterFunc, WithoutCancel)** -- https://go.dev/doc/go1.21
