# net/http -- Go Standard HTTP Server

> net/http implements HTTP servers and clients with Go's standard library, enabling production-quality web services through Handler, ServeMux, and middleware patterns.

---

## What You Will Learn in This Chapter

1. **Handler / HandlerFunc** -- The basics of HTTP request processing
2. **ServeMux (Go 1.22+)** -- Enhanced routing
3. **Middleware Patterns** -- Separation of cross-cutting concerns
4. **HTTP Client** -- Calling external services
5. **Graceful Shutdown** -- Safe server termination
6. **httptest** -- Testing HTTP handlers
7. **Production Operations** -- Security and performance configuration


## Prerequisites

Before reading this guide, the following knowledge will help deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts

---

## 1. Handler / HandlerFunc

Go's HTTP server is designed around the `http.Handler` interface. This interface has just one method: `ServeHTTP(ResponseWriter, *Request)`.

### 1.1 Handler Interface Basics

```go
// Standard library definition
type Handler interface {
    ServeHTTP(ResponseWriter, *Request)
}
```

`HandlerFunc` is an adapter type that converts a function into the `Handler` interface.

```go
type HandlerFunc func(ResponseWriter, *Request)

func (f HandlerFunc) ServeHTTP(w ResponseWriter, r *Request) {
    f(w, r)
}
```

### Code Example 1: Basic HTTP Server

```go
package main

import (
	"fmt"
	"log"
	"net/http"
	"time"
)

func main() {
	mux := http.NewServeMux()

	// HandlerFunc pattern
	mux.HandleFunc("GET /hello", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, "Hello, World!")
	})

	// Handler interface implementation pattern
	mux.Handle("GET /health", &healthHandler{})

	server := &http.Server{
		Addr:         ":8080",
		Handler:      mux,
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 10 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	log.Printf("Server starting on %s", server.Addr)
	log.Fatal(server.ListenAndServe())
}

// healthHandler is a struct that implements the Handler interface
type healthHandler struct{}

func (h *healthHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	fmt.Fprintf(w, `{"status":"ok"}`)
}
```

### Code Example 2: Go 1.22+ Pattern Matching

In Go 1.22, `ServeMux` was significantly enhanced, adding native support for HTTP method specification, path parameters, and wildcards.

```go
package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"time"
)

type User struct {
	ID    int    `json:"id"`
	Name  string `json:"name"`
	Email string `json:"email"`
}

var users = map[int]*User{
	1: {ID: 1, Name: "Tanaka", Email: "tanaka@example.com"},
	2: {ID: 2, Name: "Suzuki", Email: "suzuki@example.com"},
}

func main() {
	mux := http.NewServeMux()

	// Method + path pattern (Go 1.22+)
	mux.HandleFunc("GET /users", listUsers)
	mux.HandleFunc("POST /users", createUser)
	mux.HandleFunc("GET /users/{id}", getUser)       // Path parameter
	mux.HandleFunc("PUT /users/{id}", updateUser)
	mux.HandleFunc("DELETE /users/{id}", deleteUser)

	// Wildcard (Go 1.22+)
	mux.HandleFunc("GET /files/{path...}", serveFile) // Capture remaining path

	// Priority: more specific patterns take precedence
	mux.HandleFunc("GET /users/me", getCurrentUser)   // Prioritized over /users/{id}

	server := &http.Server{
		Addr:    ":8080",
		Handler: mux,
	}
	server.ListenAndServe()
}

func listUsers(w http.ResponseWriter, r *http.Request) {
	userList := make([]*User, 0, len(users))
	for _, u := range users {
		userList = append(userList, u)
	}
	writeJSON(w, http.StatusOK, userList)
}

func getUser(w http.ResponseWriter, r *http.Request) {
	idStr := r.PathValue("id") // Go 1.22+ path parameter
	id, err := strconv.Atoi(idStr)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid user id")
		return
	}

	user, ok := users[id]
	if !ok {
		writeError(w, http.StatusNotFound, "user not found")
		return
	}
	writeJSON(w, http.StatusOK, user)
}

func createUser(w http.ResponseWriter, r *http.Request) {
	var user User
	if err := json.NewDecoder(r.Body).Decode(&user); err != nil {
		writeError(w, http.StatusBadRequest, "invalid json")
		return
	}
	user.ID = len(users) + 1
	users[user.ID] = &user
	writeJSON(w, http.StatusCreated, user)
}

func updateUser(w http.ResponseWriter, r *http.Request) {
	idStr := r.PathValue("id")
	id, _ := strconv.Atoi(idStr)

	var update User
	if err := json.NewDecoder(r.Body).Decode(&update); err != nil {
		writeError(w, http.StatusBadRequest, "invalid json")
		return
	}

	user, ok := users[id]
	if !ok {
		writeError(w, http.StatusNotFound, "user not found")
		return
	}

	if update.Name != "" {
		user.Name = update.Name
	}
	if update.Email != "" {
		user.Email = update.Email
	}
	writeJSON(w, http.StatusOK, user)
}

func deleteUser(w http.ResponseWriter, r *http.Request) {
	idStr := r.PathValue("id")
	id, _ := strconv.Atoi(idStr)
	delete(users, id)
	w.WriteHeader(http.StatusNoContent)
}

func getCurrentUser(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, users[1]) // Dummy: current user
}

func serveFile(w http.ResponseWriter, r *http.Request) {
	path := r.PathValue("path") // Wildcard: entire remaining path
	fmt.Fprintf(w, "Serving file: %s", path)
}

// --- Helper functions ---

func writeJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

func writeError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, map[string]string{"error": message})
}
```

---

## 2. Middleware Patterns

Middleware is a function with the signature `func(http.Handler) http.Handler` that injects cross-cutting concerns (logging, authentication, CORS, etc.) before and after request processing.

### Code Example 3: Basic Middleware

```go
package main

import (
	"log"
	"net/http"
	"time"
)

// loggingMiddleware records request logs
func loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()

		// Custom ResponseWriter to capture status code
		wrapped := &responseWriter{ResponseWriter: w, statusCode: http.StatusOK}

		next.ServeHTTP(wrapped, r)

		log.Printf(
			"%s %s %d %v",
			r.Method,
			r.URL.Path,
			wrapped.statusCode,
			time.Since(start),
		)
	})
}

// responseWriter is a wrapper that captures the status code
type responseWriter struct {
	http.ResponseWriter
	statusCode int
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.statusCode = code
	rw.ResponseWriter.WriteHeader(code)
}

// authMiddleware performs authentication checks
func authMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		token := r.Header.Get("Authorization")
		if token == "" {
			http.Error(w, `{"error":"unauthorized"}`, http.StatusUnauthorized)
			return
		}
		// Token validation (omitted)
		next.ServeHTTP(w, r)
	})
}

// corsMiddleware sets CORS headers
func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		next.ServeHTTP(w, r)
	})
}

// recoveryMiddleware recovers from panics
func recoveryMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if err := recover(); err != nil {
				log.Printf("PANIC: %v", err)
				http.Error(w, `{"error":"internal server error"}`, http.StatusInternalServerError)
			}
		}()
		next.ServeHTTP(w, r)
	})
}

// rateLimitMiddleware applies rate limiting
func rateLimitMiddleware(rps int) func(http.Handler) http.Handler {
	limiter := make(chan struct{}, rps)
	// Token replenishment
	go func() {
		ticker := time.NewTicker(time.Second / time.Duration(rps))
		defer ticker.Stop()
		for range ticker.C {
			select {
			case limiter <- struct{}{}:
			default:
			}
		}
	}()
	// Initial tokens
	for i := 0; i < rps; i++ {
		limiter <- struct{}{}
	}

	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			select {
			case <-limiter:
				next.ServeHTTP(w, r)
			default:
				http.Error(w, `{"error":"rate limit exceeded"}`, http.StatusTooManyRequests)
			}
		})
	}
}

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/data", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"data": "hello"})
	})

	// Middleware chain (applied from inner to outer)
	handler := recoveryMiddleware(
		loggingMiddleware(
			corsMiddleware(
				authMiddleware(mux),
			),
		),
	)

	http.ListenAndServe(":8080", handler)
}
```

### Code Example 4: Middleware Chain Builder Helper

```go
package main

import "net/http"

// Middleware is the type definition for middleware
type Middleware func(http.Handler) http.Handler

// Chain concatenates multiple middleware
func Chain(handler http.Handler, middlewares ...Middleware) http.Handler {
	// Applied in reverse order (first middleware becomes outermost)
	for i := len(middlewares) - 1; i >= 0; i-- {
		handler = middlewares[i](handler)
	}
	return handler
}

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/users", listUsersHandler)
	mux.HandleFunc("GET /api/public", publicHandler)

	// Common middleware
	commonMiddlewares := []Middleware{
		recoveryMiddleware,
		loggingMiddleware,
		corsMiddleware,
	}

	// API requiring authentication
	authMiddlewares := append(commonMiddlewares, authMiddleware)
	authHandler := Chain(mux, authMiddlewares...)

	// Apply common middleware to all routes
	handler := Chain(mux, commonMiddlewares...)
	_ = handler
	_ = authHandler
}

func listUsersHandler(w http.ResponseWriter, r *http.Request) {}
func publicHandler(w http.ResponseWriter, r *http.Request)    {}
```

### Code Example 5: Context-Integrated Middleware

```go
package main

import (
	"context"
	"fmt"
	"log"
	"net/http"

	"github.com/google/uuid"
)

type contextKey string

const (
	requestIDKey contextKey = "requestID"
	userIDKey    contextKey = "userID"
)

// requestIDMiddleware generates a request ID and sets it in the Context
func requestIDMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reqID := r.Header.Get("X-Request-ID")
		if reqID == "" {
			reqID = uuid.New().String()
		}

		ctx := context.WithValue(r.Context(), requestIDKey, reqID)
		w.Header().Set("X-Request-ID", reqID)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// getRequestID retrieves the request ID from the Context
func getRequestID(ctx context.Context) string {
	id, _ := ctx.Value(requestIDKey).(string)
	return id
}

// structuredLoggingMiddleware outputs structured logs
func structuredLoggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reqID := getRequestID(r.Context())
		log.Printf("request_id=%s method=%s path=%s remote=%s",
			reqID, r.Method, r.URL.Path, r.RemoteAddr)
		next.ServeHTTP(w, r)
	})
}

func handler(w http.ResponseWriter, r *http.Request) {
	reqID := getRequestID(r.Context())
	fmt.Fprintf(w, "Request ID: %s", reqID)
}

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/data", handler)

	h := requestIDMiddleware(structuredLoggingMiddleware(mux))
	http.ListenAndServe(":8080", h)
}
```

---

## 3. JSON Response and Request Processing

### Code Example 6: Generic JSON Processing

```go
package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
)

// --- Response helpers ---

// APIResponse is a unified API response
type APIResponse struct {
	Success bool        `json:"success"`
	Data    interface{} `json:"data,omitempty"`
	Error   *APIError   `json:"error,omitempty"`
}

// APIError contains error information
type APIError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// respondJSON sends a JSON response
func respondJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)

	resp := APIResponse{
		Success: status >= 200 && status < 300,
		Data:    data,
	}
	json.NewEncoder(w).Encode(resp)
}

// respondError sends an error response
func respondError(w http.ResponseWriter, status int, code, message string) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)

	resp := APIResponse{
		Success: false,
		Error: &APIError{
			Code:    code,
			Message: message,
		},
	}
	json.NewEncoder(w).Encode(resp)
}

// --- Request body decoding ---

// decodeJSON decodes the request body as JSON
func decodeJSON(r *http.Request, dst interface{}) error {
	// Content-Type check
	ct := r.Header.Get("Content-Type")
	if !strings.HasPrefix(ct, "application/json") {
		return errors.New("content-type must be application/json")
	}

	// Body size limit (1MB)
	r.Body = http.MaxBytesReader(nil, r.Body, 1<<20)

	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields() // Reject unknown fields

	if err := dec.Decode(dst); err != nil {
		return fmt.Errorf("invalid json: %w", err)
	}

	// Check that the body doesn't contain multiple JSON objects
	if dec.More() {
		return errors.New("request body must contain a single json object")
	}

	return nil
}

// --- Handlers ---

type CreateUserRequest struct {
	Name  string `json:"name"`
	Email string `json:"email"`
}

func (r *CreateUserRequest) Validate() error {
	if r.Name == "" {
		return errors.New("name is required")
	}
	if r.Email == "" {
		return errors.New("email is required")
	}
	if !strings.Contains(r.Email, "@") {
		return errors.New("invalid email format")
	}
	return nil
}

func createUserHandler(w http.ResponseWriter, r *http.Request) {
	var req CreateUserRequest
	if err := decodeJSON(r, &req); err != nil {
		respondError(w, http.StatusBadRequest, "INVALID_JSON", err.Error())
		return
	}

	if err := req.Validate(); err != nil {
		respondError(w, http.StatusBadRequest, "VALIDATION_ERROR", err.Error())
		return
	}

	user := User{ID: 1, Name: req.Name, Email: req.Email}
	respondJSON(w, http.StatusCreated, user)
}

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /api/users", createUserHandler)
	http.ListenAndServe(":8080", mux)
}
```

### Code Example 7: Streaming JSON

```go
package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// streamEvents streams data using Server-Sent Events (SSE)
func streamEvents(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	for i := 0; i < 10; i++ {
		select {
		case <-r.Context().Done():
			// Client disconnected
			return
		default:
			data, _ := json.Marshal(map[string]interface{}{
				"event": i,
				"time":  time.Now().Format(time.RFC3339),
			})
			fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()
			time.Sleep(1 * time.Second)
		}
	}
}

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /events", streamEvents)
	http.ListenAndServe(":8080", mux)
}
```

---

## 4. HTTP Client

### Code Example 8: Production-Quality HTTP Client

```go
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// HTTPClient is a reusable HTTP client
type HTTPClient struct {
	client  *http.Client
	baseURL string
}

// NewHTTPClient creates a configured HTTP client
func NewHTTPClient(baseURL string) *HTTPClient {
	return &HTTPClient{
		client: &http.Client{
			Timeout: 30 * time.Second,
			Transport: &http.Transport{
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 10,
				IdleConnTimeout:     90 * time.Second,
			},
		},
		baseURL: baseURL,
	}
}

// Get sends a GET request
func (c *HTTPClient) Get(ctx context.Context, path string, result interface{}) error {
	req, err := http.NewRequestWithContext(ctx, "GET", c.baseURL+path, nil)
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}

	req.Header.Set("Accept", "application/json")

	resp, err := c.client.Do(req)
	if err != nil {
		return fmt.Errorf("do request: %w", err)
	}
	defer resp.Body.Close()

	// Response body size limit
	body, err := io.ReadAll(io.LimitReader(resp.Body, 10<<20)) // 10MB limit
	if err != nil {
		return fmt.Errorf("read body: %w", err)
	}

	if resp.StatusCode >= 400 {
		return fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}

	if result != nil {
		if err := json.Unmarshal(body, result); err != nil {
			return fmt.Errorf("decode json: %w", err)
		}
	}

	return nil
}

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	client := NewHTTPClient("https://httpbin.org")

	var result map[string]interface{}
	if err := client.Get(ctx, "/get", &result); err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	fmt.Printf("Result: %v\n", result)
}
```

### Code Example 9: HTTP Client with Retry

```go
package main

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"net/http"
	"time"
)

// RetryConfig holds retry configuration
type RetryConfig struct {
	MaxRetries  int
	BaseDelay   time.Duration
	MaxDelay    time.Duration
	RetryOn     []int // HTTP status codes to retry on
}

// DefaultRetryConfig is the default retry configuration
var DefaultRetryConfig = RetryConfig{
	MaxRetries: 3,
	BaseDelay:  100 * time.Millisecond,
	MaxDelay:   5 * time.Second,
	RetryOn:    []int{429, 500, 502, 503, 504},
}

// doWithRetry executes an HTTP request with retries
func doWithRetry(ctx context.Context, client *http.Client, req *http.Request, config RetryConfig) (*http.Response, error) {
	var lastErr error

	for attempt := 0; attempt <= config.MaxRetries; attempt++ {
		if attempt > 0 {
			// Exponential backoff + jitter
			delay := time.Duration(math.Pow(2, float64(attempt-1))) * config.BaseDelay
			jitter := time.Duration(rand.Int63n(int64(delay / 2)))
			delay = delay + jitter

			if delay > config.MaxDelay {
				delay = config.MaxDelay
			}

			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(delay):
			}
		}

		// Clone the request (for Body reuse)
		clonedReq := req.Clone(ctx)

		resp, err := client.Do(clonedReq)
		if err != nil {
			lastErr = fmt.Errorf("attempt %d: %w", attempt+1, err)
			continue
		}

		// Check if the status code is retryable
		if shouldRetry(resp.StatusCode, config.RetryOn) {
			resp.Body.Close()
			lastErr = fmt.Errorf("attempt %d: HTTP %d", attempt+1, resp.StatusCode)
			continue
		}

		return resp, nil
	}

	return nil, fmt.Errorf("all %d attempts failed: %w", config.MaxRetries+1, lastErr)
}

func shouldRetry(statusCode int, retryOn []int) bool {
	for _, code := range retryOn {
		if statusCode == code {
			return true
		}
	}
	return false
}

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	client := &http.Client{Timeout: 10 * time.Second}
	req, _ := http.NewRequestWithContext(ctx, "GET", "https://httpbin.org/status/500", nil)

	resp, err := doWithRetry(ctx, client, req, DefaultRetryConfig)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	defer resp.Body.Close()
	fmt.Printf("Status: %d\n", resp.StatusCode)
}
```

---

## 5. Graceful Shutdown

### Code Example 10: Complete Graceful Shutdown

```go
package main

import (
	"context"
	"errors"
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
		fmt.Fprintf(w, `{"status":"ok"}`)
	})

	mux.HandleFunc("GET /slow", func(w http.ResponseWriter, r *http.Request) {
		// Simulate a slow operation
		select {
		case <-r.Context().Done():
			log.Println("Client disconnected during slow request")
			return
		case <-time.After(5 * time.Second):
			fmt.Fprintf(w, "Completed after 5 seconds")
		}
	})

	server := &http.Server{
		Addr:         ":8080",
		Handler:      mux,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 15 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	// Start server in the background
	serverErr := make(chan error, 1)
	go func() {
		log.Printf("Server starting on %s", server.Addr)
		if err := server.ListenAndServe(); !errors.Is(err, http.ErrServerClosed) {
			serverErr <- err
		}
	}()

	// Wait for signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	select {
	case err := <-serverErr:
		log.Fatalf("Server error: %v", err)
	case sig := <-quit:
		log.Printf("Received signal: %v", sig)
	}

	// Graceful Shutdown
	log.Println("Shutting down server...")
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer shutdownCancel()

	if err := server.Shutdown(shutdownCtx); err != nil {
		log.Printf("Server shutdown error: %v", err)
		server.Close() // Force stop
	}

	log.Println("Server stopped gracefully")
}
```

### Code Example 11: Graceful Shutdown with Health Check Support

In Kubernetes environments, a grace period (preStop hook) is needed between receiving SIGTERM and the health check starting to fail.

```go
package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"sync/atomic"
	"syscall"
	"time"
)

var isReady int32 = 1 // 1 = ready, 0 = shutting down

func healthHandler(w http.ResponseWriter, r *http.Request) {
	if atomic.LoadInt32(&isReady) == 1 {
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `{"status":"ok"}`)
	} else {
		w.WriteHeader(http.StatusServiceUnavailable)
		fmt.Fprintf(w, `{"status":"shutting_down"}`)
	}
}

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", healthHandler)
	mux.HandleFunc("GET /readyz", healthHandler)
	mux.HandleFunc("GET /api/data", func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(100 * time.Millisecond)
		fmt.Fprintf(w, `{"data":"hello"}`)
	})

	server := &http.Server{
		Addr:    ":8080",
		Handler: mux,
	}

	go func() {
		log.Fatal(server.ListenAndServe())
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("Received shutdown signal")

	// Step 1: Fail the health check (wait for load balancer to stop sending traffic)
	atomic.StoreInt32(&isReady, 0)
	log.Println("Health check now returns 503, waiting for LB drain...")
	time.Sleep(10 * time.Second) // Part of Kubernetes terminationGracePeriodSeconds

	// Step 2: Stop the server
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	if err := server.Shutdown(ctx); err != nil {
		log.Printf("Shutdown error: %v", err)
	}
	log.Println("Server stopped")
}
```

---

## 6. Testing with httptest

### Code Example 12: Unit Testing Handlers

```go
package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestGetUser(t *testing.T) {
	// Create request
	req := httptest.NewRequest("GET", "/users/1", nil)
	rec := httptest.NewRecorder()

	// Execute handler
	getUser(rec, req)

	// Verify response
	resp := rec.Result()
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("expected status 200, got %d", resp.StatusCode)
	}

	contentType := resp.Header.Get("Content-Type")
	if !strings.Contains(contentType, "application/json") {
		t.Errorf("expected Content-Type application/json, got %s", contentType)
	}

	var user User
	if err := json.NewDecoder(resp.Body).Decode(&user); err != nil {
		t.Fatalf("decode json: %v", err)
	}

	if user.Name != "Tanaka" {
		t.Errorf("expected name Tanaka, got %s", user.Name)
	}
}

func TestCreateUser(t *testing.T) {
	body := `{"name":"Yamada","email":"yamada@example.com"}`
	req := httptest.NewRequest("POST", "/users", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	createUserHandler(rec, req)

	resp := rec.Result()
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusCreated {
		t.Errorf("expected status 201, got %d", resp.StatusCode)
	}
}

func TestCreateUser_InvalidJSON(t *testing.T) {
	body := `{invalid json}`
	req := httptest.NewRequest("POST", "/users", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	createUserHandler(rec, req)

	resp := rec.Result()
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("expected status 400, got %d", resp.StatusCode)
	}
}
```

### Code Example 13: Table-Driven Tests

```go
package main

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestUserEndpoints(t *testing.T) {
	tests := []struct {
		name           string
		method         string
		path           string
		body           string
		expectedStatus int
	}{
		{
			name:           "list users",
			method:         "GET",
			path:           "/users",
			expectedStatus: http.StatusOK,
		},
		{
			name:           "get existing user",
			method:         "GET",
			path:           "/users/1",
			expectedStatus: http.StatusOK,
		},
		{
			name:           "get non-existing user",
			method:         "GET",
			path:           "/users/999",
			expectedStatus: http.StatusNotFound,
		},
		{
			name:           "create user with valid data",
			method:         "POST",
			path:           "/users",
			body:           `{"name":"Test","email":"test@example.com"}`,
			expectedStatus: http.StatusCreated,
		},
		{
			name:           "create user with missing name",
			method:         "POST",
			path:           "/users",
			body:           `{"email":"test@example.com"}`,
			expectedStatus: http.StatusBadRequest,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var req *http.Request
			if tt.body != "" {
				req = httptest.NewRequest(tt.method, tt.path, strings.NewReader(tt.body))
				req.Header.Set("Content-Type", "application/json")
			} else {
				req = httptest.NewRequest(tt.method, tt.path, nil)
			}

			rec := httptest.NewRecorder()

			// Pass the request to the router
			mux := setupRoutes() // Set up the router under test
			mux.ServeHTTP(rec, req)

			resp := rec.Result()
			defer resp.Body.Close()

			if resp.StatusCode != tt.expectedStatus {
				t.Errorf("expected status %d, got %d", tt.expectedStatus, resp.StatusCode)
			}
		})
	}
}

func setupRoutes() *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /users", listUsers)
	mux.HandleFunc("POST /users", createUserHandler)
	mux.HandleFunc("GET /users/{id}", getUser)
	return mux
}
```

### Code Example 14: Integration Tests with httptest.Server

```go
package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestUserAPI_Integration(t *testing.T) {
	mux := setupRoutes()

	// Start test server
	server := httptest.NewServer(mux)
	defer server.Close()

	// Make requests with an actual HTTP client
	client := server.Client()

	t.Run("list users", func(t *testing.T) {
		resp, err := client.Get(server.URL + "/users")
		if err != nil {
			t.Fatalf("request failed: %v", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			t.Errorf("expected 200, got %d", resp.StatusCode)
		}

		var users []User
		json.NewDecoder(resp.Body).Decode(&users)
		if len(users) == 0 {
			t.Error("expected at least one user")
		}
	})

	t.Run("get user by id", func(t *testing.T) {
		resp, err := client.Get(fmt.Sprintf("%s/users/1", server.URL))
		if err != nil {
			t.Fatalf("request failed: %v", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			t.Errorf("expected 200, got %d", resp.StatusCode)
		}
	})
}
```

### Code Example 15: Testing Middleware

```go
package main

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestLoggingMiddleware(t *testing.T) {
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	handler := loggingMiddleware(inner)

	req := httptest.NewRequest("GET", "/test", nil)
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", rec.Code)
	}
}

func TestAuthMiddleware_NoToken(t *testing.T) {
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	handler := authMiddleware(inner)

	req := httptest.NewRequest("GET", "/test", nil)
	// No Authorization header
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusUnauthorized {
		t.Errorf("expected 401, got %d", rec.Code)
	}
}

func TestAuthMiddleware_WithToken(t *testing.T) {
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	handler := authMiddleware(inner)

	req := httptest.NewRequest("GET", "/test", nil)
	req.Header.Set("Authorization", "Bearer valid-token")
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", rec.Code)
	}
}

func TestCORSMiddleware(t *testing.T) {
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	handler := corsMiddleware(inner)

	// OPTIONS preflight request
	req := httptest.NewRequest("OPTIONS", "/test", nil)
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusNoContent {
		t.Errorf("expected 204, got %d", rec.Code)
	}

	if got := rec.Header().Get("Access-Control-Allow-Origin"); got != "*" {
		t.Errorf("expected CORS header *, got %s", got)
	}
}
```

---

## 7. File Upload and Download

### Code Example 16: File Upload

```go
package main

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
)

func uploadHandler(w http.ResponseWriter, r *http.Request) {
	// Parse multipart form (max 32MB)
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		http.Error(w, "file too large", http.StatusBadRequest)
		return
	}

	file, header, err := r.FormFile("file")
	if err != nil {
		http.Error(w, "missing file", http.StatusBadRequest)
		return
	}
	defer file.Close()

	// Sanitize filename
	filename := filepath.Base(header.Filename)

	// Destination path
	savePath := filepath.Join("uploads", filename)

	// Save file
	dst, err := os.Create(savePath)
	if err != nil {
		http.Error(w, "failed to create file", http.StatusInternalServerError)
		return
	}
	defer dst.Close()

	written, err := io.Copy(dst, file)
	if err != nil {
		http.Error(w, "failed to save file", http.StatusInternalServerError)
		return
	}

	respondJSON(w, http.StatusOK, map[string]interface{}{
		"filename": filename,
		"size":     written,
	})
}

func downloadHandler(w http.ResponseWriter, r *http.Request) {
	filename := r.PathValue("filename")
	filePath := filepath.Join("uploads", filepath.Base(filename))

	// Check file existence
	info, err := os.Stat(filePath)
	if os.IsNotExist(err) {
		http.Error(w, "file not found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=%q", filename))
	w.Header().Set("Content-Type", "application/octet-stream")
	w.Header().Set("Content-Length", fmt.Sprintf("%d", info.Size()))

	http.ServeFile(w, r, filePath)
}

func main() {
	os.MkdirAll("uploads", 0755)

	mux := http.NewServeMux()
	mux.HandleFunc("POST /upload", uploadHandler)
	mux.HandleFunc("GET /download/{filename}", downloadHandler)
	http.ListenAndServe(":8080", mux)
}
```

---

## 8. ASCII Diagrams

### Diagram 1: HTTP Request Processing Flow

```
Client Request
     |
     v
+----------+   +----------+   +----------+   +----------+
| Recovery |-->| Logging  |-->| Auth     |-->| CORS     |
| MW       |   | MW       |   | MW       |   | MW       |
+----------+   +----------+   +----------+   +----------+
                                                   |
                                                   v
                                             +----------+   +----------+
                                             | ServeMux |-->| Handler  |
                                             | (Router) |   | (Logic)  |
                                             +----------+   +----------+
                                                                  |
     +------------------------------------------------------------+
     |              Response (passes through in reverse order)
     v
  Client
```

### Diagram 2: Handler Interface

```
+-------------------------------+
| type Handler interface {      |
|   ServeHTTP(ResponseWriter,   |
|             *Request)         |
| }                             |
+-----------+-------------------+
            | implements
    +-------+-------+
    v       v       v
 ServeMux  HandlerFunc  Custom Handler
 (router)  (function    (method on
            to Handler   a struct)
            adapter)
```

### Diagram 3: Middleware Chain

```
Middleware stacking (nested structure):

recovery(logging(auth(cors(mux))))

Request direction ->
+-------------------------------------+
| recovery                            |
|  +------------------------------+   |
|  | logging                      |   |
|  |  +----------------------+   |   |
|  |  | auth                 |   |   |
|  |  |  +--------------+   |   |   |
|  |  |  | cors         |   |   |   |
|  |  |  |  +--------+  |   |   |   |
|  |  |  |  |ServeMux|  |   |   |   |
|  |  |  |  |->Handler| |   |   |   |
|  |  |  |  +--------+  |   |   |   |
|  |  |  +--------------+   |   |   |
|  |  +----------------------+   |   |
|  +------------------------------+   |
+-------------------------------------+
<- Response direction
```

### Diagram 4: Go 1.22 Pattern Matching

```
Pattern:  "GET /users/{id}"
             |       |    |
             |       |    +-- Path parameter: r.PathValue("id")
             |       +------ Path prefix
             +-------------- HTTP method constraint

Priority (more specific patterns take precedence):
  "GET /users/me"       > "GET /users/{id}"
  "GET /users/{id}"     > "GET /users/{path...}"
  "GET /users/{id}"     > "GET /{rest...}"
```

### Diagram 5: Graceful Shutdown Flow

```
                 SIGTERM
                    |
                    v
  +---------------------------------+
  | 1. Health check -> 503          | <- LB stops sending traffic
  |    atomic.Store(&isReady, 0)    |
  |    wait(10s)                    |
  +---------------------------------+
  | 2. server.Shutdown(ctx)         | <- Reject new connections
  |    Wait for in-flight requests  |
  +---------------------------------+
  | 3. All requests completed       |
  |    or timeout(30s)              |
  +---------------------------------+
  | 4. server.Close() (force stop)  | <- Fallback on timeout
  +---------------------------------+
```

---

## 9. Comparison Tables

### Table 1: ServeMux Before Go 1.21 vs Go 1.22+

| Feature | Before Go 1.21 | Go 1.22+ |
|---------|---------------|----------|
| Method specification | Not possible (manual check) | `"GET /path"` |
| Path parameters | Not possible (third-party required) | `"/users/{id}"` |
| Wildcards | Not possible | `"/files/{path...}"` |
| Priority | Longest match | Most specific pattern |
| External dependencies | gorilla/mux, chi, etc. required | Standard library sufficient |
| Path parameter retrieval | mux.Vars(r), etc. | r.PathValue("id") |

### Table 2: Standard net/http vs Third-Party Frameworks

| Item | net/http (1.22+) | Gin | Echo | chi |
|------|-----------------|-----|------|-----|
| Performance | High | Very high | Very high | Very high |
| Routing | Pattern matching | Radix tree | Radix tree | Radix tree |
| Validation | None | binding | validator | None |
| Swagger generation | Manual | swaggo support | swaggo support | swaggo support |
| Dependencies | None | Yes | Yes | Few |
| Standard compatibility | Native | Custom Context | Custom Context | net/http compatible |

### Table 3: http.Server Timeout Settings

| Parameter | Target | Recommended | Description |
|-----------|--------|-------------|-------------|
| ReadTimeout | Request reading | 5-15s | Entire header + body |
| ReadHeaderTimeout | Headers only | 5s | Slowloris protection |
| WriteTimeout | Response writing | 10-30s | Includes handler processing time |
| IdleTimeout | Keep-Alive connections | 60-120s | Idle connection retention time |
| MaxHeaderBytes | Header size | 1MB | DoS protection |

### Table 4: HTTP Client Transport Settings

| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| MaxIdleConns | 100 | 100 | Total idle connections across all hosts |
| MaxIdleConnsPerHost | 2 | 10-25 | Idle connections per host |
| IdleConnTimeout | 90s | 90s | Idle connection timeout |
| TLSHandshakeTimeout | 10s | 10s | TLS handshake time limit |
| ResponseHeaderTimeout | None | 30s | Response header wait time |
| ExpectContinueTimeout | 1s | 1s | 100-continue wait time |

---

## 10. Anti-Patterns

### Anti-Pattern 1: Server Without Timeout Settings

```go
// BAD: Default http.ListenAndServe
http.ListenAndServe(":8080", mux) // No timeout -> vulnerable to Slowloris attacks

// GOOD: Explicitly set timeouts
server := &http.Server{
	Addr:              ":8080",
	Handler:           mux,
	ReadTimeout:       10 * time.Second,
	ReadHeaderTimeout: 5 * time.Second,
	WriteTimeout:      10 * time.Second,
	IdleTimeout:       120 * time.Second,
	MaxHeaderBytes:    1 << 20, // 1MB
}
server.ListenAndServe()
```

### Anti-Pattern 2: Double Writing to ResponseWriter

```go
// BAD: Calling http.Error after WriteHeader
func handler(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	// ... error occurs during processing
	http.Error(w, "error", http.StatusInternalServerError) // Has no effect!
}

// GOOD: Perform error checks first
func handler(w http.ResponseWriter, r *http.Request) {
	data, err := process()
	if err != nil {
		http.Error(w, "error", http.StatusInternalServerError)
		return // Always return
	}
	w.WriteHeader(http.StatusOK)
	w.Write(data)
}
```

### Anti-Pattern 3: Using http.DefaultClient

```go
// BAD: Default client without timeout
resp, err := http.Get("https://example.com/api") // No timeout

// GOOD: Custom client with timeout
client := &http.Client{
	Timeout: 30 * time.Second,
	Transport: &http.Transport{
		MaxIdleConnsPerHost: 10,
	},
}
resp, err := client.Get("https://example.com/api")
```

### Anti-Pattern 4: Forgetting to Read Response Body

```go
// BAD: Closing without reading the response body
resp, err := http.Get("https://example.com")
if err != nil {
	return err
}
resp.Body.Close() // Body not read -> TCP connection cannot be reused

// GOOD: Fully read the body before closing
resp, err := http.Get("https://example.com")
if err != nil {
	return err
}
defer resp.Body.Close()
io.Copy(io.Discard, resp.Body) // Fully read the body
```

### Anti-Pattern 5: Using ResponseWriter in a Goroutine

```go
// BAD: Using ResponseWriter after handler returns
func handler(w http.ResponseWriter, r *http.Request) {
	go func() {
		time.Sleep(1 * time.Second)
		fmt.Fprintf(w, "delayed response") // Handler already returned -> race condition
	}()
}

// GOOD: Complete all necessary processing within the handler
func handler(w http.ResponseWriter, r *http.Request) {
	// Complete processing synchronously
	result := process()
	fmt.Fprintf(w, "result: %s", result)

	// Handle async tasks separately
	go func() {
		sendNotification(result)
	}()
}
```

---

## 11. Production Best Practices

### Security Header Configuration

```go
func securityHeadersMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("X-Frame-Options", "DENY")
		w.Header().Set("X-XSS-Protection", "1; mode=block")
		w.Header().Set("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
		w.Header().Set("Content-Security-Policy", "default-src 'self'")
		w.Header().Set("Referrer-Policy", "strict-origin-when-cross-origin")
		next.ServeHTTP(w, r)
	})
}
```

### Distributed Tracing with Request-ID

```go
func traceMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		traceID := r.Header.Get("X-Trace-ID")
		if traceID == "" {
			traceID = generateUUID()
		}
		spanID := generateUUID()

		ctx := context.WithValue(r.Context(), "traceID", traceID)
		ctx = context.WithValue(ctx, "spanID", spanID)

		w.Header().Set("X-Trace-ID", traceID)
		w.Header().Set("X-Span-ID", spanID)

		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

func generateUUID() string {
	return "uuid-placeholder" // In practice, use uuid.New().String()
}
```

---

## 12. FAQ

### Q1: Can a production server be built with net/http alone?

With Go 1.22 and later, it is entirely possible. Path parameters, method matching, and wildcards are now natively supported. However, validation, automatic Swagger generation, BindJSON, and similar features still require third-party libraries. For simple APIs and microservices, the standard library is sufficient.

### Q2: How should Graceful Shutdown be implemented?

Use `server.Shutdown(ctx)`. Catch SIGTERM, wait for in-flight requests to complete, then shut down. In Kubernetes environments, also consider the health check 503 switch and preStop hook wait time.

### Q3: Are timeouts necessary for http.Client as well?

Absolutely. The default `http.DefaultClient` has no timeout. Use `&http.Client{Timeout: 30 * time.Second}` or Context-based timeouts. For external service calls, also consider combining retries with a Circuit Breaker.

### Q4: What ordering rules apply when writing to ResponseWriter?

Follow the order: Headers -> WriteHeader -> Body. Modifying headers after `WriteHeader` or the first `Write` call will have no effect. `http.Error` internally calls `WriteHeader`, so any subsequent status code changes are ignored.

### Q5: Is ServeMux thread-safe?

`http.ServeMux` is thread-safe. There are no issues if routes are registered before the server starts. However, dynamic route addition after server startup requires protection with `sync.Mutex`.

### Q6: What is the relationship between Context cancellation and ResponseWriter?

When a client disconnects, `r.Context().Done()` fires. Long-running handlers should check the context state and abort processing if the client has disconnected. Note that writes to `ResponseWriter` often do not error out even after context cancellation (though the output is meaningless).

### Q7: How should http.FileServer be used, and what are the caveats?

`http.FileServer(http.Dir("./static"))` serves static files. However, directory listing is enabled by default. To disable it, implement a custom FileSystem or place an `index.html` in each directory.

---


## FAQ

### Q1: What is the most important point when learning this topic?

Gaining practical experience is the most important aspect. Understanding deepens not just through theory but by actually writing code and verifying its behavior.

### Q2: What are common mistakes beginners make?

Skipping the basics and jumping straight to advanced topics. We recommend thoroughly understanding the fundamental concepts explained in this guide before moving to the next step.

### Q3: How is this knowledge applied in practice?

Knowledge of this topic is frequently applied in daily development work. It becomes particularly important during code reviews and architecture design.

---

## Summary

| Concept | Key Point |
|---------|-----------|
| Handler | Interface with `ServeHTTP(w, r)` |
| ServeMux | Go 1.22+ supports methods and path parameters |
| Middleware | `func(http.Handler) http.Handler` pattern |
| Server settings | Timeouts must always be configured |
| HTTP Client | Timeout and Transport settings are essential |
| Graceful Shutdown | server.Shutdown(ctx) + signal handling |
| httptest | Recorder/Request/Server for testing |

---

## Recommended Next Reads

- [01-gin-echo.md](./01-gin-echo.md) -- Gin/Echo Frameworks
- [02-database.md](./02-database.md) -- Database Connectivity
- [04-testing.md](./04-testing.md) -- HTTP Testing

---

## References

1. **Go Standard Library: net/http** -- https://pkg.go.dev/net/http
2. **Go Blog, "Routing Enhancements for Go 1.22"** -- https://go.dev/blog/routing-enhancements
3. **Go Wiki: LearnServerProgramming** -- https://go.dev/wiki/LearnServerProgramming
4. **Cloudflare Blog, "So you want to expose Go on the Internet"** -- https://blog.cloudflare.com/exposing-go-on-the-internet/
