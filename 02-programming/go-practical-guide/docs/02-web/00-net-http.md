# net/http -- Go標準HTTPサーバー

> net/httpはGoの標準ライブラリでHTTPサーバー/クライアントを実装し、Handler・ServeMux・ミドルウェアパターンで本番品質のWebサービスを構築できる。

---

## この章で学ぶこと

1. **Handler / HandlerFunc** -- HTTPリクエスト処理の基本
2. **ServeMux (Go 1.22+)** -- 強化されたルーティング
3. **ミドルウェアパターン** -- 横断的関心事の分離
4. **HTTPクライアント** -- 外部サービス呼び出し
5. **Graceful Shutdown** -- 安全なサーバー停止
6. **httptest** -- HTTPハンドラのテスト
7. **本番運用** -- セキュリティ・パフォーマンス設定

---

## 1. Handler / HandlerFunc

Go のHTTPサーバーは `http.Handler` インターフェースを中心に設計されている。このインターフェースはたった1つのメソッド `ServeHTTP(ResponseWriter, *Request)` を持つ。

### 1.1 Handler インターフェースの基本

```go
// 標準ライブラリの定義
type Handler interface {
    ServeHTTP(ResponseWriter, *Request)
}
```

`HandlerFunc` は関数を `Handler` インターフェースに変換するアダプタ型である。

```go
type HandlerFunc func(ResponseWriter, *Request)

func (f HandlerFunc) ServeHTTP(w ResponseWriter, r *Request) {
    f(w, r)
}
```

### コード例 1: 基本的なHTTPサーバー

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

	// HandlerFunc パターン
	mux.HandleFunc("GET /hello", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, "Hello, World!")
	})

	// Handler インターフェース実装パターン
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

// healthHandler は Handler インターフェースを実装する構造体
type healthHandler struct{}

func (h *healthHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	fmt.Fprintf(w, `{"status":"ok"}`)
}
```

### コード例 2: Go 1.22+ パターンマッチング

Go 1.22で `ServeMux` が大幅に強化され、HTTPメソッドの指定、パスパラメータ、ワイルドカードが標準サポートされた。

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

	// メソッド + パスパターン（Go 1.22+）
	mux.HandleFunc("GET /users", listUsers)
	mux.HandleFunc("POST /users", createUser)
	mux.HandleFunc("GET /users/{id}", getUser)       // パスパラメータ
	mux.HandleFunc("PUT /users/{id}", updateUser)
	mux.HandleFunc("DELETE /users/{id}", deleteUser)

	// ワイルドカード（Go 1.22+）
	mux.HandleFunc("GET /files/{path...}", serveFile) // 残りのパスを取得

	// 優先順位: より具体的なパターンが優先
	mux.HandleFunc("GET /users/me", getCurrentUser)   // /users/{id} より優先

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
	idStr := r.PathValue("id") // Go 1.22+ パスパラメータ
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
	writeJSON(w, http.StatusOK, users[1]) // ダミー: 現在のユーザー
}

func serveFile(w http.ResponseWriter, r *http.Request) {
	path := r.PathValue("path") // ワイルドカード: 残りのパス全体
	fmt.Fprintf(w, "Serving file: %s", path)
}

// --- ヘルパー関数 ---

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

## 2. ミドルウェアパターン

ミドルウェアは `func(http.Handler) http.Handler` のシグネチャを持つ関数で、リクエスト処理の前後に横断的関心事（ログ、認証、CORS等）を挿入する。

### コード例 3: 基本的なミドルウェア

```go
package main

import (
	"log"
	"net/http"
	"time"
)

// loggingMiddleware はリクエストのログを記録する
func loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()

		// カスタムResponseWriterでステータスコードをキャプチャ
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

// responseWriter はステータスコードをキャプチャするラッパー
type responseWriter struct {
	http.ResponseWriter
	statusCode int
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.statusCode = code
	rw.ResponseWriter.WriteHeader(code)
}

// authMiddleware は認証チェックを行う
func authMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		token := r.Header.Get("Authorization")
		if token == "" {
			http.Error(w, `{"error":"unauthorized"}`, http.StatusUnauthorized)
			return
		}
		// トークン検証（省略）
		next.ServeHTTP(w, r)
	})
}

// corsMiddleware はCORSヘッダーを設定する
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

// recoveryMiddleware はパニックからリカバリする
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

// rateLimitMiddleware はレート制限を行う
func rateLimitMiddleware(rps int) func(http.Handler) http.Handler {
	limiter := make(chan struct{}, rps)
	// トークン補充
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
	// 初期トークン
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

	// ミドルウェアチェーン（内側から外側に適用）
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

### コード例 4: ミドルウェアチェーンの構築ヘルパー

```go
package main

import "net/http"

// Middleware はミドルウェアの型定義
type Middleware func(http.Handler) http.Handler

// Chain は複数のミドルウェアを連結する
func Chain(handler http.Handler, middlewares ...Middleware) http.Handler {
	// 逆順に適用（最初のミドルウェアが最も外側になる）
	for i := len(middlewares) - 1; i >= 0; i-- {
		handler = middlewares[i](handler)
	}
	return handler
}

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/users", listUsersHandler)
	mux.HandleFunc("GET /api/public", publicHandler)

	// 共通ミドルウェア
	commonMiddlewares := []Middleware{
		recoveryMiddleware,
		loggingMiddleware,
		corsMiddleware,
	}

	// 認証が必要なAPI
	authMiddlewares := append(commonMiddlewares, authMiddleware)
	authHandler := Chain(mux, authMiddlewares...)

	// 全ルートに共通ミドルウェアを適用
	handler := Chain(mux, commonMiddlewares...)
	_ = handler
	_ = authHandler
}

func listUsersHandler(w http.ResponseWriter, r *http.Request) {}
func publicHandler(w http.ResponseWriter, r *http.Request)    {}
```

### コード例 5: Context連携ミドルウェア

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

// requestIDMiddleware はリクエストIDを生成してContextに設定する
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

// getRequestID はContextからリクエストIDを取得する
func getRequestID(ctx context.Context) string {
	id, _ := ctx.Value(requestIDKey).(string)
	return id
}

// structuredLoggingMiddleware は構造化ログを出力する
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

## 3. JSONレスポンス・リクエスト処理

### コード例 6: 汎用的なJSON処理

```go
package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
)

// --- レスポンスヘルパー ---

// APIResponse は統一されたAPIレスポンス
type APIResponse struct {
	Success bool        `json:"success"`
	Data    interface{} `json:"data,omitempty"`
	Error   *APIError   `json:"error,omitempty"`
}

// APIError はエラー情報
type APIError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// respondJSON はJSONレスポンスを送信する
func respondJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)

	resp := APIResponse{
		Success: status >= 200 && status < 300,
		Data:    data,
	}
	json.NewEncoder(w).Encode(resp)
}

// respondError はエラーレスポンスを送信する
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

// --- リクエストボディのデコード ---

// decodeJSON はリクエストボディをJSONデコードする
func decodeJSON(r *http.Request, dst interface{}) error {
	// Content-Type チェック
	ct := r.Header.Get("Content-Type")
	if !strings.HasPrefix(ct, "application/json") {
		return errors.New("content-type must be application/json")
	}

	// ボディサイズ制限（1MB）
	r.Body = http.MaxBytesReader(nil, r.Body, 1<<20)

	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields() // 未知のフィールドを拒否

	if err := dec.Decode(dst); err != nil {
		return fmt.Errorf("invalid json: %w", err)
	}

	// 複数のJSONオブジェクトが含まれていないかチェック
	if dec.More() {
		return errors.New("request body must contain a single json object")
	}

	return nil
}

// --- ハンドラ ---

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

### コード例 7: ストリーミングJSON

```go
package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// streamEvents はServer-Sent Events (SSE) でデータをストリーミングする
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
			// クライアント切断
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

## 4. HTTPクライアント

### コード例 8: 本番品質のHTTPクライアント

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

// HTTPClient は再利用可能なHTTPクライアント
type HTTPClient struct {
	client  *http.Client
	baseURL string
}

// NewHTTPClient は設定済みのHTTPクライアントを生成する
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

// Get はGETリクエストを送信する
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

	// レスポンスボディのサイズ制限
	body, err := io.ReadAll(io.LimitReader(resp.Body, 10<<20)) // 10MB制限
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

### コード例 9: リトライ付きHTTPクライアント

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

// RetryConfig はリトライ設定
type RetryConfig struct {
	MaxRetries  int
	BaseDelay   time.Duration
	MaxDelay    time.Duration
	RetryOn     []int // リトライ対象のHTTPステータスコード
}

// DefaultRetryConfig はデフォルトのリトライ設定
var DefaultRetryConfig = RetryConfig{
	MaxRetries: 3,
	BaseDelay:  100 * time.Millisecond,
	MaxDelay:   5 * time.Second,
	RetryOn:    []int{429, 500, 502, 503, 504},
}

// doWithRetry はリトライ付きでHTTPリクエストを実行する
func doWithRetry(ctx context.Context, client *http.Client, req *http.Request, config RetryConfig) (*http.Response, error) {
	var lastErr error

	for attempt := 0; attempt <= config.MaxRetries; attempt++ {
		if attempt > 0 {
			// 指数バックオフ + ジッター
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

		// リクエストのクローン（Body再利用のため）
		clonedReq := req.Clone(ctx)

		resp, err := client.Do(clonedReq)
		if err != nil {
			lastErr = fmt.Errorf("attempt %d: %w", attempt+1, err)
			continue
		}

		// リトライ対象のステータスコードかチェック
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

### コード例 10: 完全なGraceful Shutdown

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
		// 遅い処理のシミュレーション
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

	// サーバーをバックグラウンドで起動
	serverErr := make(chan error, 1)
	go func() {
		log.Printf("Server starting on %s", server.Addr)
		if err := server.ListenAndServe(); !errors.Is(err, http.ErrServerClosed) {
			serverErr <- err
		}
	}()

	// シグナルを待機
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
		server.Close() // 強制停止
	}

	log.Println("Server stopped gracefully")
}
```

### コード例 11: ヘルスチェック対応のGraceful Shutdown

Kubernetes環境では、SIGTERMを受け取ってからヘルスチェックが失敗するまでの猶予期間（preStop hook）が必要。

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

	// Step 1: ヘルスチェックを失敗させる（ロードバランサーがトラフィックを停止するまで待つ）
	atomic.StoreInt32(&isReady, 0)
	log.Println("Health check now returns 503, waiting for LB drain...")
	time.Sleep(10 * time.Second) // Kubernetes terminationGracePeriodSeconds の一部

	// Step 2: サーバーを停止
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	if err := server.Shutdown(ctx); err != nil {
		log.Printf("Shutdown error: %v", err)
	}
	log.Println("Server stopped")
}
```

---

## 6. httptest によるテスト

### コード例 12: ハンドラのユニットテスト

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
	// リクエストの作成
	req := httptest.NewRequest("GET", "/users/1", nil)
	rec := httptest.NewRecorder()

	// ハンドラの実行
	getUser(rec, req)

	// レスポンスの検証
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

### コード例 13: テーブルドリブンテスト

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

			// ルーターにリクエストを渡す
			mux := setupRoutes() // テスト対象のルーターをセットアップ
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

### コード例 14: httptest.Server を使った統合テスト

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

	// テストサーバーを起動
	server := httptest.NewServer(mux)
	defer server.Close()

	// 実際のHTTPクライアントでリクエスト
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

### コード例 15: ミドルウェアのテスト

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
	// Authorization ヘッダーなし
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

	// OPTIONS プリフライトリクエスト
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

## 7. ファイルアップロード・ダウンロード

### コード例 16: ファイルアップロード

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
	// マルチパートフォームの解析（最大32MB）
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

	// ファイル名のサニタイズ
	filename := filepath.Base(header.Filename)

	// 保存先のパス
	savePath := filepath.Join("uploads", filename)

	// ファイルを保存
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

	// ファイルの存在チェック
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

## 8. ASCII図解

### 図1: HTTPリクエスト処理フロー

```
Client Request
     │
     ▼
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ Recovery │──>│ Logging  │──>│ Auth     │──>│ CORS     │
│ MW       │   │ MW       │   │ MW       │   │ MW       │
└──────────┘   └──────────┘   └──────────┘   └──────────┘
                                                   │
                                                   ▼
                                             ┌──────────┐   ┌──────────┐
                                             │ ServeMux │──>│ Handler  │
                                             │ (Router) │   │ (Logic)  │
                                             └──────────┘   └──────────┘
                                                                  │
     ┌────────────────────────────────────────────────────────────┘
     │              Response (逆順で通過)
     ▼
  Client
```

### 図2: Handler インターフェース

```
┌─────────────────────────────────┐
│ type Handler interface {        │
│   ServeHTTP(ResponseWriter,     │
│             *Request)           │
│ }                               │
└───────────┬─────────────────────┘
            │ 実装
    ┌───────┼───────┐
    ▼       ▼       ▼
 ServeMux  HandlerFunc  カスタムHandler
 (router)  (関数→Handler  (structに
            変換)         メソッド)
```

### 図3: ミドルウェアチェーン

```
ミドルウェアの積み重ね (入れ子構造):

recovery(logging(auth(cors(mux))))

リクエスト方向 →
┌─────────────────────────────────────┐
│ recovery                             │
│  ┌──────────────────────────────┐   │
│  │ logging                      │   │
│  │  ┌──────────────────────┐   │   │
│  │  │ auth                  │   │   │
│  │  │  ┌──────────────┐   │   │   │
│  │  │  │ cors          │   │   │   │
│  │  │  │  ┌────────┐  │   │   │   │
│  │  │  │  │ServeMux│  │   │   │   │
│  │  │  │  │→Handler│  │   │   │   │
│  │  │  │  └────────┘  │   │   │   │
│  │  │  └──────────────┘   │   │   │
│  │  └──────────────────────┘   │   │
│  └──────────────────────────────┘   │
└─────────────────────────────────────┘
← レスポンス方向
```

### 図4: Go 1.22 パターンマッチング

```
パターン:  "GET /users/{id}"
             │       │    │
             │       │    └── パスパラメータ: r.PathValue("id")
             │       └────── パスプレフィックス
             └────────────── HTTPメソッド制約

優先順位（具体的なパターンが優先）:
  "GET /users/me"       > "GET /users/{id}"
  "GET /users/{id}"     > "GET /users/{path...}"
  "GET /users/{id}"     > "GET /{rest...}"
```

### 図5: Graceful Shutdownフロー

```
                 SIGTERM
                    │
                    ▼
  ┌─────────────────────────────────┐
  │ 1. ヘルスチェック → 503         │ ← LBがトラフィック停止
  │    atomic.Store(&isReady, 0)    │
  │    wait(10s)                    │
  ├─────────────────────────────────┤
  │ 2. server.Shutdown(ctx)         │ ← 新規接続を拒否
  │    処理中のリクエスト完了を待機  │
  ├─────────────────────────────────┤
  │ 3. 全リクエスト完了             │
  │    or タイムアウト(30s)         │
  ├─────────────────────────────────┤
  │ 4. server.Close() (強制停止)    │ ← タイムアウト時のフォールバック
  └─────────────────────────────────┘
```

---

## 9. 比較表

### 表1: Go 1.21以前 vs Go 1.22以降のServeMux

| 機能 | Go 1.21以前 | Go 1.22以降 |
|------|------------|------------|
| メソッド指定 | 不可 (手動チェック) | `"GET /path"` |
| パスパラメータ | 不可 (サードパーティ必要) | `"/users/{id}"` |
| ワイルドカード | 不可 | `"/files/{path...}"` |
| 優先順位 | 最長一致 | 最も具体的なパターン |
| 外部依存 | gorilla/mux, chi等が必要 | 標準ライブラリで十分 |
| パスパラメータ取得 | mux.Vars(r) 等 | r.PathValue("id") |

### 表2: 標準net/http vs サードパーティフレームワーク

| 項目 | net/http (1.22+) | Gin | Echo | chi |
|------|-----------------|-----|------|-----|
| パフォーマンス | 高 | 非常に高 | 非常に高 | 非常に高 |
| ルーティング | パターンマッチ | Radix tree | Radix tree | Radix tree |
| バリデーション | なし | binding | validator | なし |
| Swagger生成 | 手動 | swaggo対応 | swaggo対応 | swaggo対応 |
| 依存 | なし | あり | あり | 少 |
| 標準互換 | ネイティブ | 独自Context | 独自Context | net/http互換 |

### 表3: http.Server タイムアウト設定

| パラメータ | 対象 | 推奨値 | 説明 |
|-----------|------|-------|------|
| ReadTimeout | リクエスト読み取り | 5-15s | ヘッダー + ボディ全体 |
| ReadHeaderTimeout | ヘッダーのみ | 5s | Slowloris対策 |
| WriteTimeout | レスポンス書き込み | 10-30s | ハンドラ処理時間含む |
| IdleTimeout | Keep-Alive接続 | 60-120s | アイドル接続の維持時間 |
| MaxHeaderBytes | ヘッダーサイズ | 1MB | DoS対策 |

### 表4: HTTPクライアントのTransport設定

| パラメータ | デフォルト | 推奨値 | 説明 |
|-----------|----------|-------|------|
| MaxIdleConns | 100 | 100 | 全ホスト合計のアイドル接続数 |
| MaxIdleConnsPerHost | 2 | 10-25 | ホスト毎のアイドル接続数 |
| IdleConnTimeout | 90s | 90s | アイドル接続のタイムアウト |
| TLSHandshakeTimeout | 10s | 10s | TLSハンドシェイクの制限時間 |
| ResponseHeaderTimeout | なし | 30s | レスポンスヘッダーの待機時間 |
| ExpectContinueTimeout | 1s | 1s | 100-continue の待機時間 |

---

## 10. アンチパターン

### アンチパターン 1: タイムアウト未設定のサーバー

```go
// BAD: デフォルトのhttp.ListenAndServe
http.ListenAndServe(":8080", mux) // タイムアウトなし → Slowloris攻撃に脆弱

// GOOD: タイムアウトを明示的に設定
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

### アンチパターン 2: レスポンスボディの二重書き込み

```go
// BAD: WriteHeaderの後にhttp.Errorを呼ぶ
func handler(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	// ... 処理中にエラー発生
	http.Error(w, "error", http.StatusInternalServerError) // 効かない！
}

// GOOD: エラーチェックを先に行う
func handler(w http.ResponseWriter, r *http.Request) {
	data, err := process()
	if err != nil {
		http.Error(w, "error", http.StatusInternalServerError)
		return // 必ずreturn
	}
	w.WriteHeader(http.StatusOK)
	w.Write(data)
}
```

### アンチパターン 3: http.DefaultClient の使用

```go
// BAD: タイムアウトなしのデフォルトクライアント
resp, err := http.Get("https://example.com/api") // タイムアウトなし

// GOOD: タイムアウト付きのカスタムクライアント
client := &http.Client{
	Timeout: 30 * time.Second,
	Transport: &http.Transport{
		MaxIdleConnsPerHost: 10,
	},
}
resp, err := client.Get("https://example.com/api")
```

### アンチパターン 4: レスポンスボディの読み忘れ

```go
// BAD: レスポンスボディを読まずにCloseする
resp, err := http.Get("https://example.com")
if err != nil {
	return err
}
resp.Body.Close() // ボディを読んでいない → TCP接続が再利用されない

// GOOD: ボディを完全に読み取ってからCloseする
resp, err := http.Get("https://example.com")
if err != nil {
	return err
}
defer resp.Body.Close()
io.Copy(io.Discard, resp.Body) // ボディを完全に読み取る
```

### アンチパターン 5: goroutine内でResponseWriterを使う

```go
// BAD: ハンドラ返却後にResponseWriterを使う
func handler(w http.ResponseWriter, r *http.Request) {
	go func() {
		time.Sleep(1 * time.Second)
		fmt.Fprintf(w, "delayed response") // ハンドラは既に返却済み → race condition
	}()
}

// GOOD: 必要な処理をハンドラ内で完結させる
func handler(w http.ResponseWriter, r *http.Request) {
	// 同期的に処理を完了
	result := process()
	fmt.Fprintf(w, "result: %s", result)

	// 非同期タスクは別途処理
	go func() {
		sendNotification(result)
	}()
}
```

---

## 11. 本番運用のベストプラクティス

### セキュリティヘッダーの設定

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

### Request-IDによる分散トレーシング

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
	return "uuid-placeholder" // 実際にはuuid.New().String()
}
```

---

## 12. FAQ

### Q1: net/httpだけで本番サーバーは構築可能か？

Go 1.22以降であれば十分に可能。パスパラメータ・メソッドマッチング・ワイルドカードが標準サポートされた。ただし、バリデーション・Swagger自動生成・BindJSONなどはサードパーティが必要。シンプルなAPIやマイクロサービスなら標準ライブラリで十分である。

### Q2: Graceful Shutdownはどう実装するか？

`server.Shutdown(ctx)` を使う。SIGTERMをキャッチし、処理中のリクエストの完了を待ってからシャットダウンする。Kubernetes環境ではヘルスチェックの503切り替えとpreStop hookの待機時間も考慮する。

### Q3: http.Clientにもタイムアウトは必要か？

必須。デフォルトの`http.DefaultClient`はタイムアウトなし。`&http.Client{Timeout: 30 * time.Second}` またはContextのタイムアウトを使う。外部サービス呼び出しではリトライとCircuit Breakerの組み合わせも検討する。

### Q4: ResponseWriterに書き込む順序で注意すべき点は？

ヘッダー → WriteHeader → Body の順序を守る。`WriteHeader`または最初の`Write`呼び出し後にヘッダーを変更しても反映されない。`http.Error`は内部で`WriteHeader`を呼ぶため、それ以降のステータスコード変更は無効。

### Q5: ServeMuxのスレッドセーフ性は？

`http.ServeMux`はスレッドセーフ。サーバー起動前にルートを登録すれば問題ない。ただし、サーバー起動後の動的ルート追加は`sync.Mutex`で保護が必要。

### Q6: Context キャンセルとResponseWriterの関係は？

クライアントが接続を切断すると`r.Context().Done()`が発火する。長時間かかるハンドラではcontextの状態をチェックし、切断済みなら処理を中断すべき。ただし、`ResponseWriter`への書き込み自体はcontextキャンセル後もエラーにならないことが多い（出力は無意味だが）。

### Q7: http.FileServerの使い方と注意点は？

`http.FileServer(http.Dir("./static"))` で静的ファイルを配信できる。ただし、ディレクトリリスティングがデフォルトで有効。無効にするにはカスタムFileSystemを実装するか、`index.html`を各ディレクトリに配置する。

---

## まとめ

| 概念 | 要点 |
|------|------|
| Handler | `ServeHTTP(w, r)` を持つインターフェース |
| ServeMux | Go 1.22+でメソッド・パスパラメータ対応 |
| ミドルウェア | `func(http.Handler) http.Handler` パターン |
| Server設定 | タイムアウトは必ず設定する |
| HTTPクライアント | タイムアウト・Transport設定が必須 |
| Graceful Shutdown | server.Shutdown(ctx) + シグナルハンドリング |
| httptest | テスト用のRecorder/Request/Server |

---

## 次に読むべきガイド

- [01-gin-echo.md](./01-gin-echo.md) -- Gin/Echoフレームワーク
- [02-database.md](./02-database.md) -- データベース接続
- [04-testing.md](./04-testing.md) -- HTTPテスト

---

## 参考文献

1. **Go Standard Library: net/http** -- https://pkg.go.dev/net/http
2. **Go Blog, "Routing Enhancements for Go 1.22"** -- https://go.dev/blog/routing-enhancements
3. **Go Wiki: LearnServerProgramming** -- https://go.dev/wiki/LearnServerProgramming
4. **Cloudflare Blog, "So you want to expose Go on the Internet"** -- https://blog.cloudflare.com/exposing-go-on-the-internet/
