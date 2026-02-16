# Context -- キャンセル・タイムアウト・値の伝搬

> context.Contextはgoroutine間でキャンセルシグナル・タイムアウト・リクエストスコープの値を伝搬するための標準メカニズムである。

---

## この章で学ぶこと

1. **context.WithCancel** -- 手動キャンセル
2. **context.WithTimeout / WithDeadline** -- タイムアウト制御
3. **context.WithValue** -- 値の伝搬とベストプラクティス
4. **context.AfterFunc (Go 1.21+)** -- キャンセル時のコールバック
5. **context.WithoutCancel (Go 1.21+)** -- キャンセル伝搬の切断
6. **実践パターン** -- HTTPサーバー・DB・マイクロサービスでのContext活用

---

## 1. Contextの基本概念

context.Contextは以下の4つの機能を提供する。

1. **キャンセル伝搬**: 親のキャンセルが全子孫に自動伝搬する
2. **デッドライン管理**: 処理の時間制限を設定する
3. **値の伝搬**: リクエストスコープの横断的関心事を渡す
4. **Done()チャネル**: キャンセルを検知するためのチャネルを提供する

### 1.1 Contextの設計原則

- **第一引数に渡す**: 関数の第一引数を`ctx context.Context`にする
- **構造体に保存しない**: リクエストスコープのcontextをフィールドに持たない
- **nilを渡さない**: 不明な場合は`context.TODO()`を使う
- **値は横断的関心事のみ**: ビジネスロジックのパラメータを入れない
- **cancel関数は必ず呼ぶ**: `defer cancel()`を取得直後に書く

---

## 2. context.WithCancel

WithCancelは手動でキャンセルシグナルを送信するためのcontextを生成する。cancel関数を呼ぶと、そのcontextから派生した全ての子contextもキャンセルされる。

### コード例 1: context.WithCancel 基本

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

	// バックグラウンドワーカー
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
	cancel() // goroutineにキャンセルを通知
	time.Sleep(100 * time.Millisecond) // goroutineの終了を待つ
}
```

### コード例 2: 複数goroutineの同時キャンセル

```go
package main

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// worker は定期的にジョブを処理するワーカー
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
	// 5つのワーカーを起動
	for i := 1; i <= 5; i++ {
		wg.Add(1)
		go worker(ctx, i, &wg)
	}

	// 1秒後に全ワーカーを停止
	time.Sleep(1 * time.Second)
	fmt.Println("Cancelling all workers...")
	cancel()

	wg.Wait()
	fmt.Println("All workers stopped")
}
```

### コード例 3: 条件付きキャンセル

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

// monitor はシステム状態を監視し、異常検知時にキャンセルする
func monitor(ctx context.Context, cancel context.CancelFunc) {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			// ランダムに異常を検知するシミュレーション
			if rand.Float64() < 0.05 {
				fmt.Println("Monitor: critical failure detected!")
				cancel() // 全処理をキャンセル
				return
			}
		}
	}
}

// processData はデータを順次処理する
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

WithTimeoutは指定した期間後に自動的にキャンセルされるcontextを生成する。WithDeadlineは絶対時刻でデッドラインを指定する。内部的にはWithTimeoutはWithDeadlineの薄いラッパーである。

### コード例 4: context.WithTimeout

```go
package main

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"time"
)

// fetchWithTimeout はタイムアウト付きでHTTPリクエストを行う
func fetchWithTimeout(url string, timeout time.Duration) ([]byte, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel() // タイムアウト前に完了しても必ずcancelを呼ぶ

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("do request: %w", err) // タイムアウト時: context deadline exceeded
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

	// タイムアウトするケース
	_, err = fetchWithTimeout("https://httpbin.org/delay/10", 3*time.Second)
	if err != nil {
		fmt.Printf("Expected timeout error: %v\n", err)
	}
}
```

### コード例 5: context.WithDeadline

```go
package main

import (
	"context"
	"fmt"
	"time"
)

// processUntilDeadline はデッドラインまで処理を続ける
func processUntilDeadline(ctx context.Context) (int, error) {
	count := 0
	for {
		select {
		case <-ctx.Done():
			return count, ctx.Err() // context.DeadlineExceeded
		default:
			// 1アイテムの処理をシミュレート
			time.Sleep(100 * time.Millisecond)
			count++
			fmt.Printf("Processed item %d\n", count)
		}
	}
}

func main() {
	// 現在時刻から1秒後をデッドラインに設定
	deadline := time.Now().Add(1 * time.Second)
	ctx, cancel := context.WithDeadline(context.Background(), deadline)
	defer cancel()

	// デッドラインの確認
	if d, ok := ctx.Deadline(); ok {
		fmt.Printf("Deadline: %v (in %v)\n", d.Format(time.RFC3339), time.Until(d))
	}

	count, err := processUntilDeadline(ctx)
	fmt.Printf("Processed %d items, error: %v\n", count, err)
}
```

### コード例 6: ネストしたタイムアウト

```go
package main

import (
	"context"
	"fmt"
	"time"
)

// 親のタイムアウトが子より短い場合、親のタイムアウトが優先される
func demonstrateNestedTimeout() {
	// 親: 2秒タイムアウト
	parentCtx, parentCancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer parentCancel()

	// 子: 5秒タイムアウト（しかし親が2秒でキャンセルするため、実質2秒）
	childCtx, childCancel := context.WithTimeout(parentCtx, 5*time.Second)
	defer childCancel()

	// 孫: 1秒タイムアウト（これが最も短い）
	grandchildCtx, grandchildCancel := context.WithTimeout(childCtx, 1*time.Second)
	defer grandchildCancel()

	// 孫は1秒でタイムアウト
	select {
	case <-grandchildCtx.Done():
		fmt.Printf("Grandchild done: %v\n", grandchildCtx.Err())
	}

	// 子は親の2秒でタイムアウト（5秒ではない）
	select {
	case <-childCtx.Done():
		fmt.Printf("Child done: %v\n", childCtx.Err())
	}
}

func main() {
	demonstrateNestedTimeout()
}
```

### コード例 7: タイムアウトの残り時間を確認して処理を分岐

```go
package main

import (
	"context"
	"fmt"
	"time"
)

// adaptiveProcess はタイムアウトの残り時間に応じて処理方法を変える
func adaptiveProcess(ctx context.Context) error {
	deadline, ok := ctx.Deadline()
	if !ok {
		// デッドラインが設定されていない
		return fullProcess(ctx)
	}

	remaining := time.Until(deadline)
	fmt.Printf("Remaining time: %v\n", remaining)

	if remaining < 1*time.Second {
		// 残り時間が少ない → 簡易処理
		return quickProcess(ctx)
	} else if remaining < 5*time.Second {
		// 残り時間が中程度 → 標準処理
		return standardProcess(ctx)
	} else {
		// 残り時間が十分 → フル処理
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

WithValueはリクエストスコープの値をcontextに格納する。ただし、ビジネスロジックのパラメータではなく、横断的関心事（トレースID、認証情報、ロケール等）のみを格納するべきである。

### コード例 8: context.WithValue の基本

```go
package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
)

// 独自のキー型を定義してキーの衝突を防ぐ
type contextKey string

const (
	requestIDKey contextKey = "requestID"
	userIDKey    contextKey = "userID"
	localeKey    contextKey = "locale"
)

// requestIDMiddleware はリクエストIDをcontextに設定する
func requestIDMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reqID := r.Header.Get("X-Request-ID")
		if reqID == "" {
			reqID = generateRequestID() // UUID生成
		}
		ctx := context.WithValue(r.Context(), requestIDKey, reqID)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// authMiddleware はユーザーIDをcontextに設定する
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

// localeMiddleware はロケール情報をcontextに設定する
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

// handler はcontextから値を取得する
func handler(w http.ResponseWriter, r *http.Request) {
	reqID, _ := r.Context().Value(requestIDKey).(string)
	userID, _ := r.Context().Value(userIDKey).(int)
	locale, _ := r.Context().Value(localeKey).(string)

	log.Printf("[%s] User %d, Locale: %s", reqID, userID, locale)
	fmt.Fprintf(w, "Hello, user %d!", userID)
}

func generateRequestID() string {
	return "req-12345" // 実際にはUUIDを生成する
}

func validateToken(token string) (int, error) {
	return 42, nil // 実際にはトークン検証を行う
}

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/profile", handler)

	h := requestIDMiddleware(authMiddleware(localeMiddleware(mux)))
	http.ListenAndServe(":8080", h)
}
```

### コード例 9: 型安全なContext値アクセサ

```go
package main

import (
	"context"
	"errors"
	"fmt"
)

// --- キー定義 ---

type contextKey int

const (
	requestIDKey contextKey = iota
	userIDKey
	traceIDKey
	tenantIDKey
)

// --- 型安全なアクセサ ---

// SetRequestID はリクエストIDをcontextに設定する
func SetRequestID(ctx context.Context, id string) context.Context {
	return context.WithValue(ctx, requestIDKey, id)
}

// GetRequestID はcontextからリクエストIDを取得する
func GetRequestID(ctx context.Context) (string, bool) {
	id, ok := ctx.Value(requestIDKey).(string)
	return id, ok
}

// MustGetRequestID はリクエストIDを取得する（存在しなければパニック）
func MustGetRequestID(ctx context.Context) string {
	id, ok := GetRequestID(ctx)
	if !ok {
		panic("requestID not found in context")
	}
	return id
}

// SetUserID はユーザーIDをcontextに設定する
func SetUserID(ctx context.Context, id int) context.Context {
	return context.WithValue(ctx, userIDKey, id)
}

// GetUserID はcontextからユーザーIDを取得する
func GetUserID(ctx context.Context) (int, error) {
	id, ok := ctx.Value(userIDKey).(int)
	if !ok {
		return 0, errors.New("userID not found in context")
	}
	return id, nil
}

// SetTraceID はトレースIDをcontextに設定する
func SetTraceID(ctx context.Context, id string) context.Context {
	return context.WithValue(ctx, traceIDKey, id)
}

// GetTraceID はcontextからトレースIDを取得する
func GetTraceID(ctx context.Context) string {
	id, _ := ctx.Value(traceIDKey).(string)
	return id // 空文字列がデフォルト
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

## 5. Contextの伝搬チェーン

実際のWebアプリケーションでは、HTTPリクエストのcontextを起点として、サービス層、リポジトリ層、外部API呼び出しへとcontextが伝搬する。

### コード例 10: HTTPリクエストからDBまでの完全な伝搬チェーン

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

// --- Handler層 ---

func handleGetUser(userService *UserService) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// HTTPリクエストのContextを基盤にする
		ctx := r.Context()

		// ハンドラ独自のタイムアウトを追加
		ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
		defer cancel()

		userID := r.PathValue("id")
		user, err := userService.GetUser(ctx, userID)
		if err != nil {
			switch {
			case err == context.Canceled:
				// クライアントが接続を切断した
				log.Printf("Client disconnected: %v", err)
				return
			case err == context.DeadlineExceeded:
				// タイムアウト
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

// --- Service層 ---

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
	// キャッシュから試行
	user, err := s.cacheService.Get(ctx, "user:"+id)
	if err == nil && user != nil {
		return user, nil
	}

	// DBから取得
	user, err = s.repo.FindByID(ctx, id)
	if err != nil {
		return nil, fmt.Errorf("find user: %w", err)
	}

	// キャッシュに保存（contextが有効な場合のみ）
	if ctx.Err() == nil {
		_ = s.cacheService.Set(ctx, "user:"+id, user, 5*time.Minute)
	}

	return user, nil
}

// --- Repository層 ---

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

// --- Cache層 ---

type CacheService struct{}

func (c *CacheService) Get(ctx context.Context, key string) (*User, error) {
	// Redis GET with context
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
		return nil, fmt.Errorf("cache miss") // シミュレート
	}
}

func (c *CacheService) Set(ctx context.Context, key string, user *User, ttl time.Duration) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
		return nil // シミュレート
	}
}

func main() {
	// 省略: DB接続、サーバー起動
	log.Println("Server starting on :8080")
}
```

### コード例 11: マイクロサービス間のContext伝搬

```go
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// contextからHTTPヘッダーへの伝搬
func propagateContext(ctx context.Context, req *http.Request) {
	// トレースIDをHTTPヘッダーに伝搬
	if traceID := GetTraceID(ctx); traceID != "" {
		req.Header.Set("X-Trace-ID", traceID)
	}

	// リクエストIDも伝搬
	if reqID, ok := GetRequestID(ctx); ok {
		req.Header.Set("X-Request-ID", reqID)
	}

	// デッドラインをヘッダーで伝搬（オプション）
	if deadline, ok := ctx.Deadline(); ok {
		remaining := time.Until(deadline)
		req.Header.Set("X-Timeout-Ms", fmt.Sprintf("%d", remaining.Milliseconds()))
	}
}

// callExternalService は外部サービスを呼び出す
func callExternalService(ctx context.Context, url string) (map[string]interface{}, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, err
	}

	// contextの情報をHTTPヘッダーに伝搬
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

Go 1.21で追加された`context.AfterFunc`は、contextがキャンセルされた後にコールバック関数を実行する。リソースのクリーンアップや通知に使用する。

### コード例 12: context.AfterFunc

```go
package main

import (
	"context"
	"fmt"
	"time"
)

func main() {
	ctx, cancel := context.WithCancel(context.Background())

	// contextがキャンセルされたらクリーンアップを実行
	stop := context.AfterFunc(ctx, func() {
		fmt.Println("AfterFunc: context was cancelled, cleaning up...")
		// リソースのクリーンアップ処理
		closeConnections()
		flushLogs()
	})

	// AfterFuncの戻り値で登録解除が可能
	_ = stop // stop()を呼ぶとAfterFuncの登録を解除できる

	// 処理を実行
	fmt.Println("Processing...")
	time.Sleep(1 * time.Second)

	// キャンセル → AfterFuncが実行される
	cancel()
	time.Sleep(100 * time.Millisecond) // AfterFuncの実行を待つ
}

func closeConnections() {
	fmt.Println("  Connections closed")
}

func flushLogs() {
	fmt.Println("  Logs flushed")
}
```

### コード例 13: AfterFuncによるリソース解放パターン

```go
package main

import (
	"context"
	"fmt"
	"sync"
)

// Resource はキャンセル時に自動解放されるリソース
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

// acquireResource はcontextに紐づくリソースを取得する
func acquireResource(ctx context.Context, name string) *Resource {
	r := &Resource{name: name}

	// contextキャンセル時にリソースを自動解放
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

	// キャンセル → 全リソースが自動解放
	cancel()

	// 解放後のアクセスはエラー
	if err := r1.Use(); err != nil {
		fmt.Printf("Expected error: %v\n", err)
	}
}
```

---

## 7. context.WithoutCancel (Go 1.21+)

Go 1.21で追加された`context.WithoutCancel`は、親contextの値は引き継ぐが、キャンセルシグナルは伝搬しない新しいcontextを生成する。バックグラウンド処理やクリーンアップ処理で有用。

### コード例 14: context.WithoutCancel

```go
package main

import (
	"context"
	"fmt"
	"time"
)

func main() {
	// 親context: 1秒でタイムアウト
	parentCtx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	// 値を設定
	parentCtx = SetTraceID(parentCtx, "trace-abc")

	// WithoutCancel: 親のキャンセルを伝搬しない
	backgroundCtx := context.WithoutCancel(parentCtx)

	// 値は引き継がれる
	fmt.Printf("TraceID in background: %s\n", GetTraceID(backgroundCtx))

	// 親がタイムアウトしても影響を受けない
	time.Sleep(2 * time.Second)

	if parentCtx.Err() != nil {
		fmt.Printf("Parent: cancelled (%v)\n", parentCtx.Err())
	}
	if backgroundCtx.Err() == nil {
		fmt.Println("Background: still active!")
	}
}
```

### コード例 15: WithoutCancelの実践的な利用例

```go
package main

import (
	"context"
	"fmt"
	"log"
	"time"
)

// handleRequest はHTTPリクエストを処理する
func handleRequest(ctx context.Context) {
	// メイン処理（リクエストcontextに紐づく）
	result, err := processRequest(ctx)
	if err != nil {
		log.Printf("Error: %v", err)
		return
	}

	// 非同期のバックグラウンドタスク（リクエストcontextのキャンセルに影響されない）
	bgCtx := context.WithoutCancel(ctx)
	// ただし独自のタイムアウトは設定する
	bgCtx, bgCancel := context.WithTimeout(bgCtx, 30*time.Second)

	go func() {
		defer bgCancel()
		// 監査ログの記録（リクエストが終了しても完了させたい）
		writeAuditLog(bgCtx, result)
		// メトリクス送信
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
	time.Sleep(1 * time.Second) // バックグラウンドタスクの完了を待つ
}
```

---

## 8. context.WithCancelCause (Go 1.20+)

Go 1.20で追加された`context.WithCancelCause`は、キャンセル理由を付与できるcontextを生成する。デバッグやエラーレポートに有用。

### コード例 16: context.WithCancelCause

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
		// 何らかの条件でキャンセル
		time.Sleep(1 * time.Second)
		cancel(ErrResourceLimit) // 原因付きでキャンセル
	}()

	<-ctx.Done()

	// キャンセル原因を取得
	fmt.Printf("Context error: %v\n", ctx.Err())           // context canceled
	fmt.Printf("Cancel cause: %v\n", context.Cause(ctx))   // resource limit exceeded

	// 原因に基づいた処理分岐
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

## 9. Graceful Shutdownとcontext

HTTPサーバーのGraceful Shutdownにおいて、contextは重要な役割を果たす。

### コード例 17: Graceful Shutdown

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

	// サーバーをバックグラウンドで起動
	go func() {
		log.Printf("Server starting on %s", server.Addr)
		if err := server.ListenAndServe(); err != http.ErrServerClosed {
			log.Fatalf("Server error: %v", err)
		}
	}()

	// シグナルを待機
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	sig := <-quit
	log.Printf("Received signal: %v, shutting down...", sig)

	// Graceful Shutdown: 処理中のリクエストの完了を最大30秒待つ
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer shutdownCancel()

	if err := server.Shutdown(shutdownCtx); err != nil {
		log.Printf("Graceful shutdown failed: %v", err)
		// 強制終了
		server.Close()
	}

	log.Println("Server stopped")
}
```

### コード例 18: Graceful Shutdown with バックグラウンドワーカー

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

// Application はアプリケーション全体のライフサイクルを管理する
type Application struct {
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

func NewApplication() *Application {
	ctx, cancel := context.WithCancel(context.Background())
	return &Application{ctx: ctx, cancel: cancel}
}

// StartWorker はバックグラウンドワーカーを起動する
func (app *Application) StartWorker(name string, fn func(context.Context)) {
	app.wg.Add(1)
	go func() {
		defer app.wg.Done()
		log.Printf("Worker %s: started", name)
		fn(app.ctx)
		log.Printf("Worker %s: stopped", name)
	}()
}

// Shutdown はアプリケーションをシャットダウンする
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

	// メッセージ処理ワーカー
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

	// メトリクス収集ワーカー
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

	// ヘルスチェックワーカー
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

	// シグナル待機
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	app.Shutdown(10 * time.Second)
}
```

---

## 10. ASCII図解

### 図1: Contextツリーの伝搬

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

Cancel の伝搬: 親がキャンセル → 全ての子もキャンセル
```

### 図2: キャンセル伝搬フロー

```
     クライアント切断
          │
          ▼
  ┌───────────────┐
  │ HTTP Handler  │ ctx.Done() シグナル
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
                            │ (ctx)     │ ← キャンセルされる
                            └───────────┘
```

### 図3: WithTimeout の内部動作

```
t=0s          t=3s          t=5s
 │             │             │
 ├─ ctx作成 ───┤             │
 │  Timeout=5s │             │
 │             │             │
 │  処理中...   │  処理中...   │ ctx.Done()
 │             │             │ ← シグナル送信
 │             │             │
 │             │          ctx.Err() =
 │             │          DeadlineExceeded
 │             │
 │  cancel()で │
 │  早期終了可能 │
```

### 図4: ネストしたタイムアウト

```
t=0s    t=1s    t=2s    t=3s    t=4s    t=5s
 │       │       │       │       │       │
 │  ┌────┼───────┼───────┼───────┼───────┤ 親: Timeout=5s
 │  │    │       │       │       │       │
 │  │ ┌──┼───────┤       │       │       │ 子: Timeout=2s
 │  │ │  │       │       │       │       │
 │  │ │┌─┤       │       │       │       │ 孫: Timeout=1s
 │  │ ││ │       │       │       │       │
 │  │ │└─┘ Done  │       │       │       │ 孫: 1秒でタイムアウト
 │  │ └── Done   │       │       │       │ 子: 2秒でタイムアウト
 │  └──────────── Done   │       │       │ 親: 5秒でタイムアウト
 │       │       │       │       │       │
 実際: 孫(1s) → 子(2s) → 親(5s) の順でキャンセル
```

### 図5: WithoutCancel の動作

```
  parent (WithTimeout 5s)
      │
      ├── child1 (通常)
      │     └── 親キャンセル時にキャンセルされる ✓
      │
      └── child2 (WithoutCancel)
            └── 親がキャンセルされても継続 ✗ (キャンセル伝搬なし)
            └── 値は引き継ぐ ✓

  用途: バックグラウンドタスク、監査ログ記録、メトリクス送信
```

### 図6: WithCancelCause の活用

```
  ctx, cancel := context.WithCancelCause(parent)
      │
      ├── cancel(ErrUserAborted)
      │     └── context.Cause(ctx) → ErrUserAborted
      │
      ├── cancel(ErrResourceLimit)
      │     └── context.Cause(ctx) → ErrResourceLimit
      │
      └── cancel(nil)
            └── context.Cause(ctx) → context.Canceled

  デバッグ・エラーレポートに有用
```

---

## 11. 比較表

### 表1: Context生成関数

| 関数 | 用途 | Done()の発火条件 | Go版 |
|------|------|-----------------|------|
| `context.Background()` | ルート。main, init, テスト | 発火しない | 1.7+ |
| `context.TODO()` | 未決定の一時的プレースホルダ | 発火しない | 1.7+ |
| `WithCancel(parent)` | 手動キャンセル | cancel()呼び出し | 1.7+ |
| `WithCancelCause(parent)` | 原因付きキャンセル | cancel(err)呼び出し | 1.20+ |
| `WithTimeout(parent, d)` | 時間制限 | d経過 or cancel() | 1.7+ |
| `WithDeadline(parent, t)` | 絶対時刻制限 | t到達 or cancel() | 1.7+ |
| `WithValue(parent, k, v)` | 値の伝搬 | 親に依存 | 1.7+ |
| `WithoutCancel(parent)` | キャンセル非伝搬 | 発火しない | 1.21+ |
| `AfterFunc(ctx, fn)` | キャンセル時コールバック | - | 1.21+ |

### 表2: context.Err() の戻り値

| 状態 | ctx.Err() | ctx.Done() | context.Cause(ctx) |
|------|-----------|-----------|-------------------|
| 未キャンセル | nil | ブロック | nil |
| cancel()済み | context.Canceled | クローズ済み | cancel引数 or Canceled |
| タイムアウト | context.DeadlineExceeded | クローズ済み | DeadlineExceeded |

### 表3: Context値に入れるべきもの / 入れるべきでないもの

| 入れるべきもの | 入れるべきでないもの |
|--------------|-------------------|
| トレースID / SpanID | ユーザーID（引数で渡す） |
| リクエストID | 検索条件・フィルタ |
| 認証トークン / クレーム | ページネーション情報 |
| ロケール / タイムゾーン | ビジネスロジックのパラメータ |
| ロガーインスタンス | DB接続 / HTTPクライアント |
| テナントID（マルチテナント） | 設定値・フラグ |

### 表4: Contextの伝搬先

| 伝搬先 | メソッド例 | 重要度 |
|--------|----------|-------|
| database/sql | QueryContext, ExecContext | 必須 |
| net/http | NewRequestWithContext | 必須 |
| gRPC | メタデータ自動伝搬 | 自動 |
| Redis | client.WithContext | 推奨 |
| ログ | logger.WithContext | 推奨 |
| 外部API | NewRequestWithContext | 必須 |
| goroutine | 引数で渡す | 必須 |

---

## 12. アンチパターン

### アンチパターン 1: contextをstructに保存する

```go
// BAD: contextを構造体のフィールドにする
type Service struct {
	ctx context.Context // リクエスト毎に異なるcontextを保持できない
	db  *sql.DB
}

// GOOD: メソッドの第1引数としてcontextを渡す
type Service struct {
	db *sql.DB
}

func (s *Service) GetUser(ctx context.Context, id int) (*User, error) {
	return s.db.QueryRowContext(ctx, "SELECT ...", id).Scan(...)
}
```

### アンチパターン 2: WithValueの乱用

```go
// BAD: ビジネスロジックのパラメータをcontextに入れる
ctx = context.WithValue(ctx, "userID", 42)
ctx = context.WithValue(ctx, "orderID", 100)
ctx = context.WithValue(ctx, "limit", 50)

// GOOD: 関数の引数で渡す。contextは横断的関心事のみ
func GetOrders(ctx context.Context, userID, limit int) ([]Order, error) {
	// contextにはトレースID・認証情報など横断的関心事のみ
	traceID := ctx.Value(traceIDKey).(string)
	// ...
}
```

### アンチパターン 3: cancel関数を呼ばない

```go
// BAD: cancel関数を呼ばない → リソースリーク
func processRequest(parentCtx context.Context) {
	ctx, _ := context.WithTimeout(parentCtx, 5*time.Second)
	// cancel が呼ばれない → タイマーgoroutineがリーク
	doWork(ctx)
}

// GOOD: defer cancel() を即座に書く
func processRequest(parentCtx context.Context) {
	ctx, cancel := context.WithTimeout(parentCtx, 5*time.Second)
	defer cancel() // 必ずリソースを解放
	doWork(ctx)
}
```

### アンチパターン 4: 文字列キーの使用

```go
// BAD: 文字列をキーに使う（衝突リスク）
ctx = context.WithValue(ctx, "userID", 42)
ctx = context.WithValue(ctx, "userID", "conflict!") // 別パッケージが同じキーを使う可能性

// GOOD: 独自の非公開型をキーに使う
type contextKey int

const userIDKey contextKey = 0

ctx = context.WithValue(ctx, userIDKey, 42)
```

### アンチパターン 5: context.Background()の乱用

```go
// BAD: 親contextを無視してBackgroundを使う
func (s *Service) GetUser(ctx context.Context, id int) (*User, error) {
	// 引数のctxを無視してBackgroundを使う → キャンセル・タイムアウトが効かない
	dbCtx := context.Background()
	return s.db.QueryRowContext(dbCtx, "SELECT ...", id).Scan(...)
}

// GOOD: 受け取ったcontextをそのまま伝搬
func (s *Service) GetUser(ctx context.Context, id int) (*User, error) {
	return s.db.QueryRowContext(ctx, "SELECT ...", id).Scan(...)
}
```

### アンチパターン 6: Contextのキャンセル後にリソースを使う

```go
// BAD: contextキャンセル後にレスポンスを書く
func handler(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	result, err := longRunningTask(ctx)
	if err != nil {
		// contextがキャンセルされていてもResponseWriterに書き込む
		http.Error(w, err.Error(), 500) // クライアント切断時は無意味
		return
	}
	json.NewEncoder(w).Encode(result)
}

// GOOD: contextの状態をチェックしてから応答
func handler(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	result, err := longRunningTask(ctx)
	if ctx.Err() != nil {
		// クライアントは既に切断 → 応答不要
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

## 13. FAQ

### Q1: context.Background()とcontext.TODO()の違いは？

機能的には同じだが意図が異なる。`Background()`は「ルートcontextとして意図的に使う」、`TODO()`は「適切なcontextがまだ不明で後で修正する」場合に使う。linterで`TODO()`を検出して漏れを防げる。

### Q2: cancel関数は必ず呼ぶ必要があるか？

はい。`WithCancel`/`WithTimeout`/`WithDeadline`のcancel関数を呼ばないとリソースリークが発生する。`defer cancel()` を取得直後に書くのが慣例。タイムアウトで自動キャンセルされても`cancel()`は安全に呼べる（何度呼んでもエラーにならない）。

### Q3: contextの値はどのような場面で使うべきか？

リクエストスコープの横断的関心事のみ: トレースID、認証情報、ロケール等。ビジネスロジックのパラメータには使わない。独自のキー型（`type contextKey string`）を定義してキーの衝突を防ぐ。

### Q4: context.WithTimeout と http.Client.Timeout のどちらを使うべきか？

両者は補完的である。`http.Client.Timeout`はクライアント全体のタイムアウト（接続～レスポンス読み取りまで）を設定する。`context.WithTimeout`はリクエスト単位で異なるタイムアウトを設定でき、キャンセル伝搬もサポートする。一般的には両方を設定し、`http.Client.Timeout`を安全策として長めに、contextのタイムアウトをリクエスト固有の値に設定する。

### Q5: DBトランザクション中にcontextがキャンセルされたらどうなるか？

`database/sql`はcontextのキャンセルを検知してクエリを中断する。ただし、トランザクションの状態はドライバ依存。一般的には:
- 実行中のクエリは中断される
- コミット前ならロールバックされる
- `defer tx.Rollback()` パターンで安全にクリーンアップする

### Q6: contextをgoroutineに渡すときの注意点は？

- 値をコピーしてからgoroutineに渡す（gin.Contextなど、リクエスト終了後に無効になるオブジェクトに注意）
- バックグラウンドgoroutineには`context.WithoutCancel`を検討する
- 独自のタイムアウトを設定する
- goroutineリークを防ぐため、必ずキャンセル経路を確保する

### Q7: Go 1.21のAfterFuncとWithoutCancelはどう使い分けるか？

- `AfterFunc`: キャンセル時に特定のクリーンアップ処理を実行したい場合（リソース解放、ログ記録等）
- `WithoutCancel`: 親のキャンセルに影響されずに処理を続行したい場合（バックグラウンドタスク、監査ログ等）

### Q8: テストでcontextをどう扱うべきか？

テストでは`context.Background()`に適切なタイムアウトを設定する。デッドラインの長いテストはCI環境でのフレーク（不安定なテスト）の原因になるため、タイムアウトは短めに設定する:

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

## まとめ

| 概念 | 要点 |
|------|------|
| Context | goroutine間のキャンセル・タイムアウト・値伝搬 |
| WithCancel | 手動キャンセル制御 |
| WithCancelCause | 原因付きキャンセル (Go 1.20+) |
| WithTimeout | 時間制限付き処理 |
| WithDeadline | 絶対時刻によるデッドライン |
| WithValue | 横断的関心事の伝搬のみに使う |
| WithoutCancel | キャンセル伝搬を切断 (Go 1.21+) |
| AfterFunc | キャンセル時コールバック (Go 1.21+) |
| cancel() | 必ずdefer cancel()で呼ぶ |
| 伝搬 | 親キャンセル → 全子孫にキャンセル伝搬 |

---

## 次に読むべきガイド

- [../02-web/00-net-http.md](../02-web/00-net-http.md) -- HTTPサーバーでのContext活用
- [../02-web/02-database.md](../02-web/02-database.md) -- DBクエリのContext制御
- [../02-web/03-grpc.md](../02-web/03-grpc.md) -- gRPCのContext

---

## 参考文献

1. **Go Blog, "Go Concurrency Patterns: Context"** -- https://go.dev/blog/context
2. **Go Standard Library: context** -- https://pkg.go.dev/context
3. **Go Blog, "Contexts and structs"** -- https://go.dev/blog/context-and-structs
4. **Go 1.20 Release Notes (WithCancelCause)** -- https://go.dev/doc/go1.20
5. **Go 1.21 Release Notes (AfterFunc, WithoutCancel)** -- https://go.dev/doc/go1.21
