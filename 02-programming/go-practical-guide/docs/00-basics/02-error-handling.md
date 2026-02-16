# エラーハンドリング -- Goのエラー設計哲学

> Goはerror interfaceを中心とした明示的なエラーハンドリングを採用し、errors.Is/As・sentinel errors・wrappingで堅牢なエラー伝搬を実現する。

---

## この章で学ぶこと

1. **error interface** -- Go のエラーが単なるインターフェースである理由
2. **errors.Is / errors.As** -- エラーチェーンの検査方法
3. **エラーラッピング** -- `fmt.Errorf("%w", err)` による文脈の追加
4. **カスタムエラー型** -- ドメイン固有のエラー設計
5. **エラーハンドリング戦略** -- レイヤー別の処理方針
6. **panic/recover** -- 適切な使用場面と回復パターン

---

## 1. error interface の基本

### 1.1 error interface の定義

Go のエラーは特殊な構文ではなく、単なるインターフェースである。これが Go のエラー設計の根幹をなす。

```go
// builtin パッケージで定義
type error interface {
    Error() string
}
```

この設計の利点:
- エラーが値（first-class value）として扱える
- 任意の型がエラーインターフェースを実装できる
- エラーに付加情報（コード、フィールド、スタックトレース等）を持たせられる
- 条件分岐、比較、格納が通常の値と同様に可能

### コード例 1: error interface の基本と実装

```go
package main

import (
    "fmt"
    "net"
    "os"
    "strconv"
    "time"
)

// error は組み込みインターフェース
// type error interface {
//     Error() string
// }

// カスタムエラー型: バリデーションエラー
type ValidationError struct {
    Field   string
    Message string
}

func (e *ValidationError) Error() string {
    return fmt.Sprintf("validation error: field=%q, message=%q", e.Field, e.Message)
}

// カスタムエラー型: ビジネスロジックエラー
type BusinessError struct {
    Code    string
    Message string
    Details map[string]string
}

func (e *BusinessError) Error() string {
    return fmt.Sprintf("[%s] %s", e.Code, e.Message)
}

// カスタムエラー型: 一時的なエラー（リトライ可能）
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

// errors.New でシンプルなエラーを作成
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
    // 標準ライブラリのエラー例
    _, err := strconv.Atoi("not_a_number")
    fmt.Printf("strconv error: %v (type: %T)\n", err, err)

    _, err = os.Open("/nonexistent/file")
    fmt.Printf("os error: %v (type: %T)\n", err, err)

    _, err = net.Dial("tcp", "invalid:address")
    fmt.Printf("net error: %v (type: %T)\n", err, err)

    // カスタムエラーの使用
    err = validateAge(-5)
    fmt.Printf("validation error: %v\n", err)

    err = validateAge(30)
    fmt.Printf("valid age: error=%v\n", err) // nil
}
```

### コード例 2: sentinel errors（番兵エラー）

```go
package main

import (
    "errors"
    "fmt"
)

// sentinel errors の定義
// パッケージレベルで公開し、呼び出し元がチェックに使う
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

// ユーザーリポジトリ
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

    // 重複チェック
    existing, err := FindUserByEmail(email)
    if err == nil && existing != nil {
        return nil, fmt.Errorf("create user: email %q already exists: %w", email, ErrConflict)
    }
    // ErrNotFound は期待される結果（重複なし）
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
    // 正常ケース
    user, err := FindUser(1)
    if err != nil {
        fmt.Printf("Error: %v\n", err)
    } else {
        fmt.Printf("Found: %+v\n", user)
    }

    // NotFound ケース
    _, err = FindUser(999)
    if errors.Is(err, ErrNotFound) {
        fmt.Println("User not found (expected)")
    }

    // InvalidInput ケース
    _, err = FindUser(-1)
    if errors.Is(err, ErrInvalidInput) {
        fmt.Println("Invalid input (expected)")
    }

    // Conflict ケース
    _, err = CreateUser("Charlie", "alice@example.com")
    if errors.Is(err, ErrConflict) {
        fmt.Printf("Conflict: %v\n", err)
    }

    // 正常な作成
    user, err = CreateUser("Charlie", "charlie@example.com")
    if err == nil {
        fmt.Printf("Created: %+v\n", user)
    }
}
```

### コード例 3: エラーラッピングと文脈追加

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

// レイヤー1: リポジトリ層
func findUserInDB(id int) (*User, error) {
    // DB操作のシミュレーション
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

// レイヤー2: サービス層（文脈を追加してラップ）
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

// レイヤー3: ハンドラー層（エラーの種類に応じたHTTPレスポンス）
func handleGetProfile(w http.ResponseWriter, r *http.Request) {
    userID := 200 // シミュレーション

    profile, err := GetUserProfile(userID)
    if err != nil {
        // エラーの種類に応じたHTTPステータスコード
        switch {
        case errors.Is(err, ErrNotFound):
            http.Error(w, "user or profile not found", http.StatusNotFound)
        default:
            // 内部エラーの詳細はクライアントに返さない
            http.Error(w, "internal server error", http.StatusInternalServerError)
        }
        // ログには詳細を出力
        fmt.Printf("ERROR: %v\n", err)
        return
    }

    json.NewEncoder(w).Encode(profile)
}

// 実践的なエラーラッピングのパターン
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

// ファイル処理での適切なエラーハンドリング
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

    // defer で Close のエラーもチェック
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
    // エラーチェーンの確認
    _, err := GetUserProfile(200)
    fmt.Printf("Error: %v\n", err)
    fmt.Printf("Is ErrNotFound: %t\n", errors.Is(err, ErrNotFound))

    // sql.ErrNoRows のチェック例
    sqlErr := fmt.Errorf("get user: %w", sql.ErrNoRows)
    fmt.Printf("Is sql.ErrNoRows: %t\n", errors.Is(sqlErr, sql.ErrNoRows))
}
```

### コード例 4: errors.Is と errors.As の詳細

```go
package main

import (
    "errors"
    "fmt"
    "net"
    "os"
)

// カスタムエラー型
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

// カスタム Is メソッド: StatusCode が同じなら同一エラーとみなす
func (e *HTTPError) Is(target error) bool {
    t, ok := target.(*HTTPError)
    if !ok {
        return false
    }
    return e.StatusCode == t.StatusCode
}

// ValidationError 型
type ValidationError struct {
    Field   string
    Message string
}

func (e *ValidationError) Error() string {
    return fmt.Sprintf("validation error: %s - %s", e.Field, e.Message)
}

// NotFoundError 型
type NotFoundError struct {
    Resource string
    ID       interface{}
}

func (e *NotFoundError) Error() string {
    return fmt.Sprintf("%s (id=%v) not found", e.Resource, e.ID)
}

func handleError(err error) {
    // errors.Is: エラーチェーンに特定のエラーが含まれるか
    if errors.Is(err, os.ErrNotExist) {
        fmt.Println("→ File does not exist")
        return
    }

    if errors.Is(err, os.ErrPermission) {
        fmt.Println("→ Permission denied")
        return
    }

    // errors.As: エラーチェーンから特定の型を取り出す
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

    // net.Error のインターフェースチェック
    var netErr net.Error
    if errors.As(err, &netErr) {
        fmt.Printf("→ Network error: timeout=%t, temporary=%t\n",
            netErr.Timeout(), netErr.Temporary())
        return
    }

    // os.PathError のチェック
    var pathErr *os.PathError
    if errors.As(err, &pathErr) {
        fmt.Printf("→ Path error: op=%q, path=%q, err=%v\n",
            pathErr.Op, pathErr.Path, pathErr.Err)
        return
    }

    fmt.Printf("→ Unexpected error: %v\n", err)
}

func main() {
    // さまざまなエラーをテスト
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

    // カスタム Is メソッドのテスト
    err1 := &HTTPError{StatusCode: 404, Message: "user not found"}
    err2 := &HTTPError{StatusCode: 404, Message: "different message"}
    err3 := &HTTPError{StatusCode: 500, Message: "internal error"}

    fmt.Printf("\nerr1 Is err2 (same status): %t\n", errors.Is(err1, err2)) // true
    fmt.Printf("err1 Is err3 (diff status): %t\n", errors.Is(err1, err3))  // false

    // ラップされたエラーでの検索
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

### コード例 5: 複数エラーの結合 (Go 1.20+)

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

// バリデーション: 複数エラーをまとめて返す
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

    return errors.Join(errs...) // Go 1.20+: 複数エラーを結合。errsが空ならnil
}

// カスタム MultiError 型（Go 1.20以前の互換用）
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

// Unwrap は Go 1.20+ の multiple unwrap をサポート
func (e *MultiError) Unwrap() []error {
    return e.Errors
}

func main() {
    // 全フィールドが不正なユーザー
    badUser := &User{
        Name:     "",
        Email:    "invalid",
        Password: "short",
        Age:      -5,
    }

    err := validateUser(badUser)
    if err != nil {
        fmt.Printf("Validation errors:\n%v\n\n", err)

        // errors.As で特定の型を検索
        var ve *ValidationError
        if errors.As(err, &ve) {
            fmt.Printf("First validation error: field=%s, msg=%s\n\n",
                ve.Field, ve.Message)
        }
    }

    // 正常なユーザー
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

    // 部分的に不正なユーザー
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

### コード例 6: カスタムエラー型にUnwrapを実装

```go
package main

import (
    "errors"
    "fmt"
    "time"
)

// AppError はアプリケーション全体で使うエラー型
type AppError struct {
    Code       string
    Message    string
    Err        error
    Timestamp  time.Time
    RequestID  string
    StackTrace string // 本番では runtime/debug.Stack() 等で取得
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

// エラーコード定数
const (
    ErrCodeNotFound      = "NOT_FOUND"
    ErrCodeUnauthorized  = "UNAUTHORIZED"
    ErrCodeValidation    = "VALIDATION_ERROR"
    ErrCodeInternal      = "INTERNAL_ERROR"
    ErrCodeTimeout       = "TIMEOUT"
    ErrCodeConflict      = "CONFLICT"
    ErrCodeRateLimited   = "RATE_LIMITED"
)

// HTTPステータスコードへのマッピング
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

// JSON レスポンス用の構造体
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

// 便利な生成関数
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
    // AppError の使用例
    err := NotFoundError("User", 42).WithRequestID("req-abc-123")
    fmt.Println(err)
    fmt.Printf("HTTP Status: %d\n", err.HTTPStatus())
    fmt.Printf("Response: %+v\n", err.ToResponse())

    // Unwrap チェーン
    fmt.Println(errors.Is(err, ErrNotFound)) // true

    // AppError の抽出
    wrapped := fmt.Errorf("handler: %w", err)
    var appErr *AppError
    if errors.As(wrapped, &appErr) {
        fmt.Printf("Code: %s, Message: %s\n", appErr.Code, appErr.Message)
    }

    // Timeout エラー
    tErr := TimeoutError("database query", 5*time.Second)
    fmt.Println(tErr)
    fmt.Println(errors.Is(tErr, ErrTimeout)) // true
}
```

### コード例 7: panic/recover の適切な使用

```go
package main

import (
    "fmt"
    "log"
    "net/http"
    "runtime/debug"
)

// panic が適切な場面:
// 1. プログラミングエラー（到達不能なコード）
// 2. 初期化時の回復不能なエラー
// 3. ライブラリ内部のバグ検出

// recover を使ったミドルウェア
func recoveryMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        defer func() {
            if rec := recover(); rec != nil {
                // スタックトレースをログ
                log.Printf("PANIC recovered: %v\n%s", rec, debug.Stack())

                // クライアントには500を返す
                http.Error(w, "Internal Server Error", http.StatusInternalServerError)
            }
        }()
        next.ServeHTTP(w, r)
    })
}

// ライブラリ内部でpanicを使い、公開APIでrecoverする
func parseExpression(expr string) (result float64, err error) {
    defer func() {
        if r := recover(); r != nil {
            err = fmt.Errorf("parse expression %q: %v", expr, r)
        }
    }()

    // 内部ではpanicで簡潔にエラーを伝搬
    result = evalExpr(expr)
    return result, nil
}

func evalExpr(expr string) float64 {
    if expr == "" {
        panic("empty expression") // 内部でのみpanic
    }
    // 簡略化のためハードコード
    return 42.0
}

// Must パターン: main/init でのみ使用
func MustParseConfig(path string) map[string]string {
    config, err := loadConfig(path)
    if err != nil {
        panic(fmt.Sprintf("failed to load config %q: %v", path, err))
    }
    return config
}

func loadConfig(path string) (map[string]string, error) {
    // シミュレーション
    return map[string]string{"key": "value"}, nil
}

// assertNever: 到達不能なコードを示す
func processStatus(status string) string {
    switch status {
    case "active":
        return "User is active"
    case "inactive":
        return "User is inactive"
    case "deleted":
        return "User is deleted"
    default:
        // 未知のステータスはプログラミングエラー
        panic(fmt.Sprintf("unexpected status: %q", status))
    }
}

// cleanupリソースのdefer内でのpanicセーフな処理
func processFile(path string) (err error) {
    // deferでのrecover
    defer func() {
        if r := recover(); r != nil {
            err = fmt.Errorf("process file: panic: %v", r)
        }
    }()

    fmt.Printf("Processing %s\n", path)
    // 処理...
    return nil
}

func main() {
    // Must パターン
    config := MustParseConfig("config.json")
    fmt.Printf("Config: %v\n", config)

    // parseExpression のrecover
    result, err := parseExpression("1+2")
    if err != nil {
        fmt.Printf("Parse error: %v\n", err)
    } else {
        fmt.Printf("Result: %.1f\n", result)
    }

    result, err = parseExpression("")
    if err != nil {
        fmt.Printf("Parse error: %v\n", err) // panic が error に変換される
    }

    // processStatus
    fmt.Println(processStatus("active"))

    // 不正なステータスはpanicになるが、deferでrecoverできる
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

### コード例 8: エラーハンドリングの実践パターン

```go
package main

import (
    "context"
    "errors"
    "fmt"
    "log/slog"
    "time"
)

// errWriter パターン: 連続するI/O操作のエラー集約
type errWriter struct {
    err error
}

func (ew *errWriter) writeString(s string) {
    if ew.err != nil {
        return // 最初のエラー以降は何もしない
    }
    fmt.Print(s) // 実際は io.Writer に書き込む
}

func (ew *errWriter) writef(format string, args ...interface{}) {
    if ew.err != nil {
        return
    }
    fmt.Printf(format, args...)
}

// リトライパターン
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

            // 指数バックオフ
            delay *= 2
            if delay > config.MaxDelay {
                delay = config.MaxDelay
            }
        }

        lastErr = operation()
        if lastErr == nil {
            return nil
        }

        // 一時的でないエラーはリトライしない
        if !isRetryable(lastErr) {
            return fmt.Errorf("non-retryable error: %w", lastErr)
        }
    }

    return fmt.Errorf("max retries (%d) exceeded: %w", config.MaxRetries, lastErr)
}

func isRetryable(err error) bool {
    // context のキャンセルはリトライしない
    if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
        return false
    }
    // 特定のエラー型をチェック
    var tempErr interface{ Temporary() bool }
    if errors.As(err, &tempErr) {
        return tempErr.Temporary()
    }
    return true // デフォルトはリトライ可能とみなす
}

// エラーログのベストプラクティス
func handleRequest(ctx context.Context, userID int) error {
    logger := slog.With("user_id", userID, "request_id", "req-123")

    user, err := findUser(ctx, userID)
    if err != nil {
        if errors.Is(err, ErrNotFound) {
            // 期待されるエラー: INFO レベル
            logger.Info("user not found", "error", err)
            return err
        }
        // 予期しないエラー: ERROR レベル
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

// エラー集約パターン（並行処理）
func fetchAll(ctx context.Context, urls []string) ([]string, error) {
    type result struct {
        url  string
        body string
        err  error
    }

    results := make(chan result, len(urls))

    for _, url := range urls {
        go func(u string) {
            // HTTP GET のシミュレーション
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
    // errWriter パターン
    ew := &errWriter{}
    ew.writeString("Hello ")
    ew.writef("World %d\n", 42)
    if ew.err != nil {
        fmt.Printf("Write error: %v\n", ew.err)
    }

    // リトライパターン
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
        return nil // 3回目で成功
    })

    if err != nil {
        fmt.Printf("Retry failed: %v\n", err)
    } else {
        fmt.Printf("Succeeded after %d attempts\n", attempt)
    }

    // 並行エラー集約
    urls := []string{"http://a.com", "http://b.com", "http://c.com"}
    bodies, err := fetchAll(ctx, urls)
    if err != nil {
        fmt.Printf("Fetch errors: %v\n", err)
    }
    fmt.Printf("Fetched %d bodies\n", len(bodies))
}
```

---

## 2. ASCII図解

### 図1: エラーチェーン

```
fmt.Errorf("handler: %w",
  fmt.Errorf("service: %w",
    fmt.Errorf("repo: %w",
      ErrNotFound)))

エラーチェーン:
┌─────────────────┐
│ "handler: ..."  │
│   Unwrap() ─────┼──> ┌──────────────────┐
└─────────────────┘    │ "service: ..."   │
                       │   Unwrap() ──────┼──> ┌────────────────┐
                       └──────────────────┘    │ "repo: ..."    │
                                               │   Unwrap() ────┼──> ErrNotFound
                                               └────────────────┘
errors.Is(err, ErrNotFound) → チェーンを辿って true

エラーメッセージ:
"handler: service: repo: not found"
                         ↑
                    元の sentinel error
```

### 図2: errors.Is vs errors.As

```
┌─────────────────────────────────────────────┐
│              errors.Is(err, target)          │
│  目的: 特定のエラー値と一致するか検査           │
│  探索: Unwrap()を再帰的に辿る                  │
│  比較: == または Is()メソッド                   │
│  戻値: bool                                   │
│                                              │
│  使用場面:                                     │
│  ・sentinel error のチェック                   │
│  ・errors.Is(err, ErrNotFound)               │
│  ・errors.Is(err, context.Canceled)          │
│  ・errors.Is(err, sql.ErrNoRows)            │
├─────────────────────────────────────────────┤
│              errors.As(err, &target)         │
│  目的: 特定のエラー型を取り出す                 │
│  探索: Unwrap()を再帰的に辿る                  │
│  比較: 型アサーション                           │
│  戻値: bool (targetに値がセットされる)          │
│                                              │
│  使用場面:                                     │
│  ・カスタムエラー型の詳細取得                   │
│  ・var httpErr *HTTPError                    │
│    errors.As(err, &httpErr)                  │
│  ・var pathErr *os.PathError                 │
│    errors.As(err, &pathErr)                  │
└─────────────────────────────────────────────┘
```

### 図3: エラーハンドリングの判断フロー

```
         エラーが発生
              │
              ▼
     ┌────────────────┐
     │ エラーの種類は？ │
     └───┬────────┬───┘
         │        │
    期待される   予期しない
     エラー      エラー
         │        │
         ▼        ▼
   ┌──────────┐ ┌──────────────┐
   │ 適切に    │ │ %wでラップして│
   │ 処理する  │ │ 上位に返す    │
   │ (ログ等) │ │              │
   └──────────┘ └──────────────┘
         │              │
         ▼              ▼
   ┌──────────┐  ┌──────────────┐
   │ リカバリ  │  │ 文脈情報を    │
   │ 可能？   │  │ 追加して返す   │
   │ YES→対処 │  └──────────────┘
   │ NO→返す  │         │
   └──────────┘         ▼
                  ┌──────────────┐
                  │ 最上位で      │
                  │ ログ+レスポンス│
                  └──────────────┘

レイヤー別の責務:
┌────────────────────────────────────────┐
│ ハンドラー層: エラー種別→HTTPステータス   │
│ ├─ ErrNotFound → 404                  │
│ ├─ ErrValidation → 400                │
│ ├─ ErrUnauthorized → 401              │
│ └─ その他 → 500                        │
├────────────────────────────────────────┤
│ サービス層: ビジネスロジックの文脈追加     │
│ └─ fmt.Errorf("get user: %w", err)    │
├────────────────────────────────────────┤
│ リポジトリ層: データアクセスの文脈追加     │
│ └─ fmt.Errorf("query users: %w", err) │
├────────────────────────────────────────┤
│ インフラ層: 低レベルエラーの生成          │
│ └─ sql.ErrNoRows, net.Error, etc.     │
└────────────────────────────────────────┘
```

### 図4: errors.Join の仕組み (Go 1.20+)

```
errors.Join(err1, err2, err3)

結果:
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

バリデーションでの活用:
  validate(user) → errors.Join(
    nameErr,    // "name: required"
    emailErr,   // "email: invalid format"
    passErr,    // "password: too short"
  )

  ↓ errors.As で個別のエラーも取得可能

  var ve *ValidationError
  errors.As(joined, &ve) → true (最初の一致)
```

### 図5: panic/recover のフロー

```
goroutine の実行フロー:

正常終了:
  main() → f1() → f2() → return → return → return

panic 発生:
  main() → f1() → f2() → panic("!!")
                              │
                    ┌─────────▼──────────┐
                    │ defer スタックを逆順 │
                    │ に実行               │
                    │                     │
                    │ f2のdefer → 実行    │
                    │ f1のdefer → 実行    │
                    │ mainのdefer → 実行  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ recover() がなければ │
                    │ プログラム終了       │
                    └─────────────────────┘

recover() がある場合:
  main() → f1() → f2() → panic("!!")
                              │
                    ┌─────────▼──────────┐
                    │ f2のdefer:         │
                    │   recover() → "!!" │ ← panic を捕捉
                    │   err に変換       │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ f2 は正常リターン    │
                    │ (err を返す)        │
                    └──────────┬──────────┘
                               │
                    f1, main は通常通り実行継続
```

---

## 3. 比較表

### 表1: エラー処理アプローチ比較

| アプローチ | Go | Java | Rust | Python | TypeScript |
|-----------|-----|------|------|--------|------------|
| 仕組み | 戻り値 (error) | 例外 (Exception) | Result<T,E> | 例外 (Exception) | 例外 + Promise |
| 未処理時の動作 | コンパイルは通る | クラッシュ | コンパイルエラー | クラッシュ | クラッシュ |
| 型情報 | interface (動的) | クラス階層 | enum (静的) | クラス階層 | any |
| 網羅性チェック | なし | なし (checked除く) | あり (match) | なし | なし |
| 制御フロー | 明示的 if err != nil | try-catch | ? 演算子 | try-except | try-catch + .catch |
| 複数エラー | errors.Join | suppressed | -- | ExceptionGroup | AggregateError |
| スタックトレース | なし（要実装） | 自動付与 | なし | 自動付与 | 自動付与 |

### 表2: Go エラーパターン比較

| パターン | 用途 | 例 | 利点 | 欠点 |
|---------|------|-----|------|------|
| sentinel error | 既知のエラー条件 | `ErrNotFound` | シンプル、高速な比較 | 付加情報がない |
| カスタムエラー型 | 追加情報が必要 | `*ValidationError` | 豊富な情報、型安全 | 定義が冗長 |
| `fmt.Errorf("%w")` | 文脈追加 | `"open config: %w"` | 簡単、チェーン検査可 | 型情報が失われる |
| `errors.Join` | 複数エラー集約 | バリデーション | 網羅的検証可能 | エラーメッセージが長い |
| panic/recover | 本当に回復不能な状態 | プログラミングエラー | 簡潔 | 乱用は危険 |
| AppError (構造化) | API エラー | `{code, msg, err}` | HTTP統合、ログ統合 | 複雑性が増す |

### 表3: %w vs %v の選択基準

| 状況 | 使うべき書式 | 理由 |
|------|------------|------|
| 内部エラーをそのまま伝搬 | `%w` | チェーン検査を可能にする |
| ライブラリの公開API | `%v` | 内部実装の詳細を隠蔽 |
| 同一パッケージ内 | `%w` | 詳細なエラーチェックが必要 |
| 外部パッケージ境界 | 場合による | 安定したエラーのみ `%w` |
| ログ出力用 | `%v` | 単に文字列として記録 |
| sentinel error の伝搬 | `%w` | errors.Is で検査するため |

---

## 4. アンチパターン

### アンチパターン 1: エラーを握りつぶす

```go
// BAD: エラーを無視
result, _ := doSomething()

// BAD: エラーをログだけして処理しない
if err != nil {
    log.Println(err) // 呼び出し元は成功したと思う
    // return がない！
}

// BAD: 空のエラーチェック
if err != nil {
    // TODO: handle error
}

// GOOD: エラーを返す
result, err := doSomething()
if err != nil {
    return fmt.Errorf("do something: %w", err)
}

// GOOD: 意図的にエラーを無視する場合はコメント
_ = conn.Close() // best-effort close, error is intentionally ignored
```

### アンチパターン 2: エラーメッセージの重複

```go
// BAD: "failed to" が連鎖して冗長
// 結果: "failed to get user: failed to query db: failed to connect: timeout"
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

// GOOD: 簡潔に文脈を追加（動詞を省略）
// 結果: "get user: query db: connect: timeout"
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

### アンチパターン 3: エラーの二重処理

```go
// BAD: 同じエラーをログ出力してから返す
func processOrder(id int) error {
    order, err := findOrder(id)
    if err != nil {
        log.Printf("ERROR: failed to find order: %v", err) // ログ出力
        return fmt.Errorf("find order: %w", err)            // さらに返す
        // → 上位でもログされる → 同じエラーが2回記録
    }
    _ = order
    return nil
}

// GOOD: 1つのレイヤーでのみログ（通常は最上位）
func processOrder(id int) error {
    order, err := findOrder(id)
    if err != nil {
        return fmt.Errorf("process order: find order (id=%d): %w", id, err)
    }
    _ = order
    return nil
}

// 最上位のハンドラーでログ
func handleOrder(w http.ResponseWriter, r *http.Request) {
    if err := processOrder(42); err != nil {
        log.Printf("ERROR: %v", err) // ここでのみログ
        http.Error(w, "error", 500)
    }
}
```

### アンチパターン 4: panic をエラーハンドリング代わりに使う

```go
// BAD: ライブラリ関数が panic
func ParseConfig(data []byte) *Config {
    var config Config
    if err := json.Unmarshal(data, &config); err != nil {
        panic(fmt.Sprintf("invalid config: %v", err)) // NG: 呼び出し元をクラッシュさせる
    }
    return &config
}

// GOOD: error を返す
func ParseConfig(data []byte) (*Config, error) {
    var config Config
    if err := json.Unmarshal(data, &config); err != nil {
        return nil, fmt.Errorf("parse config: %w", err)
    }
    return &config, nil
}

// Must パターンは main/init 限定
func MustParseConfig(data []byte) *Config {
    config, err := ParseConfig(data)
    if err != nil {
        panic(err)
    }
    return config
}

// main() でのみ Must を使用
func main() {
    config := MustParseConfig(configData)
    _ = config
}
```

### アンチパターン 5: エラーチェック前に結果を使う

```go
// BAD: エラーチェック前に result を使う
func process() {
    result, err := fetchData()
    fmt.Println(result.Name) // err が非nil なら result は不正な可能性
    if err != nil {
        log.Fatal(err)
    }
}

// GOOD: エラーチェックを先に
func process() {
    result, err := fetchData()
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println(result.Name) // エラーなしが確認された後に使用
}
```

---

## 5. FAQ

### Q1: panicはいつ使うべきか？

panicはプログラミングエラー（nilポインタ参照、範囲外アクセス等）や、回復不能な初期化エラーのみに使う。通常のビジネスロジックでは必ず `error` を返す。ライブラリはpanicを呼び出し元に漏らしてはならない。

panic が適切な場面:
1. `main()` や `init()` での設定読み込み失敗
2. プログラムの不変条件が破られたとき
3. 到達不能なコードの表明（`default` ケース）
4. テストヘルパーでの前提条件チェック

### Q2: `%w` と `%v` の違いは？

`%w` はエラーをラップし、`errors.Is`/`errors.As` でチェーン検査可能にする。`%v` は単にエラーメッセージを文字列として埋め込む。原則として `%w` を使うが、内部実装を隠蔽したい場合（ライブラリの公開API等）は `%v` を使う。

```go
// %w: エラーチェーンが維持される
err := fmt.Errorf("open: %w", os.ErrNotExist)
errors.Is(err, os.ErrNotExist) // true

// %v: エラーチェーンが切れる
err := fmt.Errorf("open: %v", os.ErrNotExist)
errors.Is(err, os.ErrNotExist) // false
```

### Q3: エラーメッセージの命名規則は？

Go の慣例: (1) 小文字で始める、(2) "failed to" を付けない、(3) パッケージ名をプレフィックスにしない（ラップで自然に付く）、(4) 句読点で終わらない。例: `"open config file: %w"` が良い形。

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

### Q4: errors.Is と == の違いは？

`==` は直接比較のみ。`errors.Is` はエラーチェーンを辿って検索する。

```go
base := errors.New("base error")
wrapped := fmt.Errorf("wrapped: %w", base)

wrapped == base          // false (異なるオブジェクト)
errors.Is(wrapped, base) // true (チェーンに base が含まれる)
```

また、`errors.Is` はカスタム `Is()` メソッドも呼び出す。これにより、値の部分一致など柔軟な比較が可能になる。

### Q5: Go 1.13 以前と以後でエラー処理はどう変わったか？

Go 1.13 で `errors.Is`, `errors.As`, `fmt.Errorf("%w")` が導入された。これにより:

- **1.13以前**: `err == ErrNotFound` で直接比較。ラップすると比較できなくなる
- **1.13以後**: `errors.Is(err, ErrNotFound)` でチェーン全体を検索。ラップしても検出可能

Go 1.20 では `errors.Join` が追加され、複数エラーの集約が標準化された。

### Q6: エラーにスタックトレースを含めるべきか？

Go の標準ライブラリはスタックトレースを含めない設計。理由: (1) パフォーマンスへの影響、(2) エラーメッセージの文脈追加で十分追跡可能、(3) 構造化ログとの組み合わせ。

スタックトレースが必要な場合は、以下のアプローチがある:
- `runtime/debug.Stack()` をログに出力
- サードパーティ（`pkg/errors` 等）を使用
- panic/recover で取得（本番ではrecoveryミドルウェアで）
- OpenTelemetry によるトレーシング

---

## 6. まとめ

| 概念 | 要点 |
|------|------|
| error interface | `Error() string` を持つインターフェース |
| sentinel error | `var ErrXxx = errors.New(...)` で定義 |
| ラッピング | `fmt.Errorf("context: %w", err)` |
| errors.Is | エラーチェーンに特定のエラーが含まれるか |
| errors.As | エラーチェーンから特定の型を取り出す |
| errors.Join | 複数エラーの結合 (Go 1.20+) |
| panic/recover | 回復不能なエラーのみ。ライブラリは漏らさない |
| エラーメッセージ | 小文字始まり、簡潔、文脈追加、"failed to" 不要 |
| レイヤー別処理 | 下位:ラップして返す、上位:種別判定+レスポンス |

---

## 次に読むべきガイド

- [03-packages-modules.md](./03-packages-modules.md) -- パッケージとモジュール
- [../02-web/04-testing.md](../02-web/04-testing.md) -- テストにおけるエラー検証
- [../03-tools/04-best-practices.md](../03-tools/04-best-practices.md) -- ベストプラクティス

---

## 参考文献

1. **Go Blog, "Working with Errors in Go 1.13"** -- https://go.dev/blog/go1.13-errors
2. **Go Blog, "Error handling and Go"** -- https://go.dev/blog/error-handling-and-go
3. **Standard library: errors package** -- https://pkg.go.dev/errors
4. **Go Blog, "Errors are values"** -- https://go.dev/blog/errors-are-values
5. **Go Wiki: Errors** -- https://go.dev/wiki/Errors
6. **Dave Cheney, "Don't just check errors, handle them gracefully"** -- https://dave.cheney.net/2016/04/27/dont-just-check-errors-handle-them-gracefully
