# Go ベストプラクティスガイド

> Effective Goの精神に基づき、保守性・可読性・性能を兼ね備えたGoコードを書くための指針

## この章で学ぶこと

1. **Effective Go** の核心 -- Goらしいコードの設計原則と命名規則
2. **エラーハンドリング** -- エラー値の設計、ラッピング、センチネルエラーの使い分け
3. **インターフェース設計** -- 小さなインターフェース、暗黙的実装、依存注入
4. **並行処理のパターン** -- goroutine管理、チャネル設計、コンテキスト伝搬
5. **構造体設計** -- ゼロ値の活用、機能オプション、コンストラクタパターン
6. **パッケージ設計** -- 依存関係、循環依存の回避、internal パッケージ
7. **テスタビリティ** -- テストしやすいコード設計
8. **パフォーマンス** -- メモリ効率、アロケーション最適化

---

## 1. Go らしいコードの原則

### Goの設計哲学

```
+-------------------------------------------------------+
|              Go の設計哲学                               |
+-------------------------------------------------------+
|                                                       |
|  +-------------+  +-----------+  +------------------+ |
|  | Simplicity  |  | Readability|  | Composition      | |
|  | 単純さ      |  | 可読性    |  | 合成 > 継承       | |
|  +-------------+  +-----------+  +------------------+ |
|                                                       |
|  +-------------+  +-----------+  +------------------+ |
|  | Explicit    |  | Minimal   |  | Convention       | |
|  | 明示的      |  | 最小限    |  | 規約重視          | |
|  +-------------+  +-----------+  +------------------+ |
|                                                       |
|  +-------------+  +-----------+  +------------------+ |
|  | Concurrency |  | Orthogonal|  | Practical        | |
|  | 並行性      |  | 直交性    |  | 実用性            | |
|  +-------------+  +-----------+  +------------------+ |
|                                                       |
|  "Clear is better than clever"                        |
|  「巧妙さより明快さ」                                  |
|                                                       |
|  "A little copying is better than a little dependency"|
|  「少しの依存より少しのコピーの方がよい」               |
|                                                       |
|  "Don't communicate by sharing memory;                |
|   share memory by communicating"                      |
|  「メモリの共有で通信するのではなく、                    |
|   通信でメモリを共有せよ」                              |
+-------------------------------------------------------+
```

### 命名規則

```
+----------------------------------------------------------+
|  Go の命名規則                                             |
+----------------------------------------------------------+
|                                                          |
|  パッケージ名: 短く、小文字、単数形                       |
|    http, json, fmt, os, user, order                      |
|    NG: httpUtil, jsonParser, userService                  |
|    NG: utils, helpers, common, misc (意味が曖昧)          |
|                                                          |
|  エクスポート: PascalCase                                 |
|    ReadFile, HTTPClient, UserID                          |
|                                                          |
|  非エクスポート: camelCase                                |
|    readFile, httpClient, userID                          |
|                                                          |
|  インターフェース: -er 接尾辞 (1メソッドの場合)           |
|    Reader, Writer, Stringer                              |
|    Closer, Formatter, Handler                            |
|    複数メソッド: ReadWriter, UserService                  |
|                                                          |
|  頭字語: 全大文字を維持                                   |
|    HTTP, URL, ID, JSON, API, XML, SQL                    |
|    HTTPHandler (not HttpHandler)                         |
|    XMLRPC (not XmlRpc)                                   |
|    userID (not userId)                                   |
|                                                          |
|  変数名: 短いスコープ → 短い名前                         |
|    for i := 0; ...          // OK: 短いスコープ          |
|    for index := 0; ...      // NG: 冗長                  |
|    func process(r io.Reader) // OK: 1文字でも意味が明確  |
|                                                          |
|  定数: PascalCase (CamelCase)                            |
|    MaxRetryCount, DefaultTimeout                         |
|    NG: MAX_RETRY_COUNT (Goの規約ではない)                |
+----------------------------------------------------------+
```

### コード例1: 良い命名と悪い命名

```go
// NG: 冗長、Java 風
type IUserService interface { ... }       // I プレフィックス不要
type UserServiceImpl struct { ... }       // Impl サフィックス不要
func (s *UserServiceImpl) GetUserByID(userID string) (*UserModel, error) { ... }

// OK: Go 風
type UserService interface { ... }        // シンプル
type userService struct { ... }           // 非公開の実装
func (s *userService) User(id string) (*User, error) { ... }

// NG: パッケージ名の繰り返し
package user
func UserCreate() { ... }   // user.UserCreate() は冗長
func UserDelete() { ... }   // user.UserDelete() は冗長

// OK: パッケージ名を活用
package user
func Create() { ... }       // user.Create() で明確
func Delete() { ... }       // user.Delete() で明確

// NG: Get プレフィックスの乱用
func GetName() string { ... }      // Go では不要
func GetAge() int { ... }          // getter に Get は使わない

// OK: フィールド名と同じ名前
func (u *User) Name() string { return u.name }
func (u *User) Age() int { return u.age }
func (u *User) SetName(name string) { u.name = name }  // setter は Set プレフィックス

// NG: bool を返す関数名
func IsValid() bool { ... }     // OK（is で始まるのは一般的）
func HasPermission() bool { ... }  // OK
func CheckValid() bool { ... }    // NG: Check は error を返すべき

// OK: エラーを返す関数
func (u *User) Validate() error { ... }  // Check/Validate はエラーを返す
```

### コード例2: パッケージ設計のベストプラクティス

```go
// プロジェクト構成の例

// 小規模プロジェクト: フラット構成
// project/
// ├── main.go
// ├── handler.go
// ├── store.go
// ├── model.go
// └── go.mod

// 中〜大規模プロジェクト: レイヤー構成
// project/
// ├── cmd/
// │   └── server/
// │       └── main.go          -- エントリーポイント
// ├── internal/
// │   ├── handler/             -- HTTPハンドラ
// │   │   ├── user.go
// │   │   └── order.go
// │   ├── service/             -- ビジネスロジック
// │   │   ├── user.go
// │   │   └── order.go
// │   ├── repository/          -- データアクセス
// │   │   ├── user.go
// │   │   └── order.go
// │   ├── model/               -- ドメインモデル
// │   │   ├── user.go
// │   │   └── order.go
// │   └── middleware/          -- ミドルウェア
// │       ├── auth.go
// │       └── logging.go
// ├── pkg/                     -- 公開ライブラリ（他プロジェクトから利用可能）
// │   └── validator/
// │       └── validator.go
// ├── migrations/              -- DBマイグレーション
// │   └── 000001_init.sql
// ├── configs/                 -- 設定ファイル
// ├── go.mod
// └── go.sum

// 依存の方向（一方向に保つ）
// handler → service → repository → model
//
// handler は service のインターフェースに依存
// service は repository のインターフェースに依存
// model はどこにも依存しない（純粋なデータ構造）

// NG: 循環依存
// package user → package order → package user (コンパイルエラー)

// OK: インターフェースで依存を断ち切る
// package service で interface を定義
// package repository で実装
// main.go で組み立て（Dependency Injection）
```

---

## 2. エラーハンドリング

### コード例3: エラーの設計パターン

```go
package storage

import (
    "errors"
    "fmt"
)

// ===================================
// 1. センチネルエラー
// パッケージレベルで定義する固定のエラー値
// errors.Is() で判定
// ===================================
var (
    ErrNotFound     = errors.New("storage: not found")
    ErrDuplicate    = errors.New("storage: duplicate entry")
    ErrUnauthorized = errors.New("storage: unauthorized")
    ErrForbidden    = errors.New("storage: forbidden")
    ErrConflict     = errors.New("storage: conflict")
)

// ===================================
// 2. カスタムエラー型
// 追加情報を持つ構造体エラー
// errors.As() で判定
// ===================================
type ValidationError struct {
    Field   string
    Message string
}

func (e *ValidationError) Error() string {
    return fmt.Sprintf("validation error: %s - %s", e.Field, e.Message)
}

// 複数のバリデーションエラーをまとめる
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
// 3. エラーラッピング
// fmt.Errorf("%w", err) でコンテキスト付与
// ===================================
func (s *Store) FindUser(id string) (*User, error) {
    user, err := s.db.Get(id)
    if err != nil {
        if errors.Is(err, sql.ErrNoRows) {
            // センチネルエラーにラップ
            return nil, fmt.Errorf("FindUser(%s): %w", id, ErrNotFound)
        }
        // 内部エラーをラップ
        return nil, fmt.Errorf("FindUser(%s): %w", id, err)
    }
    return user, nil
}

// ===================================
// 4. エラー判定
// errors.Is / errors.As でエラーチェーンを辿る
// ===================================
func handleError(err error) {
    // 値の比較（センチネルエラー）
    if errors.Is(err, ErrNotFound) {
        // 404 応答
    }
    if errors.Is(err, ErrUnauthorized) {
        // 401 応答
    }

    // 型の比較（カスタムエラー）
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

### エラーハンドリングの判断フロー

```
エラーが発生した
    |
    +-- 回復可能か？
    |       |
    |       +-- YES → エラーを返す (return err)
    |       |         コンテキスト付与 (fmt.Errorf("...: %w", err))
    |       |
    |       +-- NO → log.Fatal / panic（起動時のみ）
    |                 例: 設定ファイル読み込み失敗
    |                     DB接続失敗（リトライ後）
    |
    +-- 呼び出し元はエラーの種類を知る必要があるか？
    |       |
    |       +-- YES → センチネルエラー or カスタムエラー型
    |       |         例: ErrNotFound → 404
    |       |              *ValidationError → 400 + 詳細
    |       |
    |       +-- NO → fmt.Errorf で文字列ラップのみ
    |                 例: 内部実装の詳細は呼び出し元に不要
    |
    +-- エラーのコンテキストは十分か？
    |       |
    |       +-- NO → fmt.Errorf("op(%s): %w", id, err)
    |       |         関数名、引数値をエラーメッセージに含める
    |       |
    |       +-- YES → そのままreturn
    |
    +-- エラーログを記録すべきか？
            |
            +-- ハンドラ層で1回だけログ記録
            +-- 途中の層では記録せずにラップして返す
            +-- 二重ログを避ける
```

### コード例4: エラーハンドリングの実践パターン

```go
package handler

import (
    "encoding/json"
    "errors"
    "log"
    "net/http"
)

// ErrorResponse はAPIエラーレスポンスの統一フォーマット
type ErrorResponse struct {
    Error   string            `json:"error"`
    Code    string            `json:"code,omitempty"`
    Details map[string]string `json:"details,omitempty"`
}

// handleError はエラーをHTTPレスポンスに変換する（ハンドラ層で1回だけ）
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
            // 未知のエラー → 500
            statusCode = http.StatusInternalServerError
            resp = ErrorResponse{Error: "internal server error", Code: "INTERNAL"}
            // 内部エラーはログに記録（レスポンスには含めない）
            logger.Printf("Internal error: %v", err)
        }
    }

    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(statusCode)
    json.NewEncoder(w).Encode(resp)
}
```

### errors.Is vs errors.As 比較表

| 関数 | 目的 | 比較対象 | 使用場面 |
|------|------|---------|---------|
| `errors.Is(err, target)` | エラーチェーンに target と一致するエラーがあるか | 値（センチネルエラー） | `ErrNotFound`, `sql.ErrNoRows` |
| `errors.As(err, &target)` | エラーチェーンに target型のエラーがあるか | 型（カスタムエラー） | `*ValidationError`, `*os.PathError` |
| `errors.Unwrap(err)` | 1層だけアンラップ | -- | 通常は直接使わない |
| `errors.Join(err1, err2)` | 複数エラーを結合 (Go 1.20+) | -- | 複数の後処理エラー |

---

## 3. インターフェース設計

### コード例5: 小さなインターフェースの原則

```go
// Go標準ライブラリの美しいインターフェース設計

// 1メソッド — 最も再利用性が高い
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

// 合成で必要なインターフェースを構築
type ReadWriter interface {
    Reader
    Writer
}

type ReadWriteCloser interface {
    Reader
    Writer
    Closer
}

// ReadCloser は io パッケージの典型例
type ReadCloser interface {
    Reader
    Closer
}
```

### コード例6: インターフェースの正しい使い方

```go
// ===================================
// 原則1: 「受け取り側でインターフェースを定義する」
// ===================================

// NG: 実装側でインターフェースを定義
package repository

type UserRepository interface {  // 実装パッケージにインターフェース
    GetByID(ctx context.Context, id int64) (*User, error)
    Create(ctx context.Context, u *User) error
}

type postgresUserRepo struct { ... }
// → 使う側は repository パッケージに依存することを強制される

// OK: 使う側でインターフェースを定義
package service

// 自分が必要なメソッドだけを定義
type UserStore interface {
    GetByID(ctx context.Context, id int64) (*User, error)
    Create(ctx context.Context, u *User) error
}

type UserService struct {
    store UserStore  // インターフェースに依存
}

// repository パッケージは構造体を提供するだけ
package repository

type PostgresUserRepo struct {
    db *sql.DB
}

func (r *PostgresUserRepo) GetByID(ctx context.Context, id int64) (*User, error) { ... }
func (r *PostgresUserRepo) Create(ctx context.Context, u *User) error { ... }
// → 暗黙的に service.UserStore を満たす

// ===================================
// 原則2: 「インターフェースを受け取り、構造体を返す」
// ===================================

// NG: 戻り値をインターフェースにする
func NewUserService(repo UserStore) UserServiceInterface {
    return &userService{repo: repo}
}
// → 型情報が失われ、具体メソッドにアクセスできない

// OK: 構造体を返す
func NewUserService(repo UserStore) *UserService {
    return &UserService{repo: repo}
}
// → 構造体のメソッドに直接アクセス可能

// ===================================
// 原則3: 「実際に使う機能だけを要求する」
// ===================================

// NG: 過大なインターフェースを要求
func ProcessData(rw ReadWriteCloser) error {
    // 実際にはReadしか使わない
    buf := make([]byte, 1024)
    _, err := rw.Read(buf)
    return err
}

// OK: 必要な機能だけを要求
func ProcessData(r io.Reader) error {
    buf := make([]byte, 1024)
    _, err := r.Read(buf)
    return err
}
// → *os.File, *bytes.Buffer, *strings.Reader, net.Conn 全てが使える
```

### インターフェース設計の原則

```
+----------------------------------------------------------+
|  "Accept interfaces, return structs"                     |
|  「インターフェースを受け取り、構造体を返す」              |
+----------------------------------------------------------+
|                                                          |
|  // 引数: インターフェース（柔軟性）                      |
|  func NewService(repo UserRepository) *UserService       |
|                                                          |
|  // 戻り値: 構造体（具体性）                              |
|  func NewService(...) *UserService  // NOT UserService   |
|                                                          |
+----------------------------------------------------------+
|                                                          |
|  "The bigger the interface, the weaker the abstraction"  |
|  「インターフェースが大きいほど抽象度は低い」             |
|                                     — Rob Pike           |
+----------------------------------------------------------+
|                                                          |
|  "Define interfaces at the point of use"                 |
|  「インターフェースは使う側で定義する」                   |
+----------------------------------------------------------+
|                                                          |
|  "Don't export interfaces for implementation"            |
|  「実装のためにインターフェースを公開するな」              |
+----------------------------------------------------------+

暗黙的インターフェース実装の利点:
┌─────────────────────────────────────────┐
│ Go: 暗黙的（Structural Typing）          │
│   → 実装側はインターフェースの存在を     │
│     知らなくてよい                        │
│   → 依存関係がシンプル                   │
│   → テスト用モックが容易                 │
│                                         │
│ Java/C#: 明示的（implements / :）        │
│   → 実装側がインターフェースに依存       │
│   → 変更時の影響範囲が大きい             │
└─────────────────────────────────────────┘
```

---

## 4. 並行処理のベストプラクティス

### コード例7: goroutine のライフサイクル管理

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
// パターン1: WaitGroup
// 単純な並列処理、エラーが不要な場合
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
// パターン2: errgroup
// 並列処理 + エラーハンドリング（推奨）
// ===================================
func fetchAll(ctx context.Context, urls []string) ([]Response, error) {
    g, ctx := errgroup.WithContext(ctx)
    responses := make([]Response, len(urls))

    for i, url := range urls {
        i, url := i, url  // ループ変数のキャプチャ（Go 1.22+ では不要）
        g.Go(func() error {
            resp, err := fetch(ctx, url)
            if err != nil {
                return fmt.Errorf("fetch %s: %w", url, err)
            }
            responses[i] = resp  // インデックスが異なるので排他制御不要
            return nil
        })
    }

    if err := g.Wait(); err != nil {
        return nil, err  // 最初のエラーが返される
    }
    return responses, nil
}

// ===================================
// パターン3: errgroup + セマフォ（並列度制限）
// ===================================
func fetchAllWithLimit(ctx context.Context, urls []string, maxConcurrency int) ([]Response, error) {
    g, ctx := errgroup.WithContext(ctx)
    g.SetLimit(maxConcurrency)  // 同時実行数を制限

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
// パターン4: Worker Pool
// 制限付き並列処理、ジョブキュー
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
                        return // チャネルが閉じられた
                    }
                    select {
                    case results <- process(job):
                    case <-ctx.Done():
                        return
                    }
                case <-ctx.Done():
                    return // コンテキストキャンセル
                }
            }
        }(i)
    }

    // 全worker終了後にresultsを閉じる
    go func() {
        wg.Wait()
        close(results)
    }()
}

// ===================================
// パターン5: Pipeline
// ステージごとのデータ処理
// ===================================
func pipeline(ctx context.Context, input <-chan int) <-chan string {
    // Stage 1: フィルター
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

    // Stage 2: 変換
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
// パターン6: Fan-out / Fan-in
// 分散処理と結果集約
// ===================================
func fanOutFanIn(ctx context.Context, input <-chan int, numWorkers int) <-chan int {
    // Fan-out: 複数のworkerに分配
    workers := make([]<-chan int, numWorkers)
    for i := 0; i < numWorkers; i++ {
        workers[i] = worker(ctx, input)
    }

    // Fan-in: 複数のworkerの結果を集約
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

### コード例8: context の正しい伝搬

```go
// ===================================
// context のルール
// ===================================

// 1. context は第一引数に渡す
func fetchData(ctx context.Context) (*Data, error) { ... }

// 2. context は struct に保存しない
// NG:
type Server struct {
    ctx context.Context  // NG: リクエストスコープの ctx を保存
}
// OK: メソッドの引数として渡す

// 3. nil context を渡さない
// NG: fetchData(nil)
// OK: fetchData(context.Background())
//     fetchData(context.TODO())  // 後で適切な context に置き換える場合

// 4. context.WithValue は最小限に（認証情報等のみ）
// NG: context に大量のデータを詰め込む
// OK: リクエストID、認証情報など横断的関心事のみ

// コンテキストの連鎖
func main() {
    // ルートコンテキスト（キャンセル可能）
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    // タイムアウト付きコンテキスト
    ctx, cancel = context.WithTimeout(ctx, 30*time.Second)
    defer cancel()

    // デッドライン付きコンテキスト
    deadline := time.Now().Add(1 * time.Minute)
    ctx, cancel = context.WithDeadline(ctx, deadline)
    defer cancel()

    // 値付きコンテキスト（横断的関心事のみ）
    ctx = context.WithValue(ctx, requestIDKey{}, "req-123")
}

// HTTPハンドラ内での context 使用
http.HandleFunc("/api/data", func(w http.ResponseWriter, r *http.Request) {
    // リクエストのコンテキストを使用
    // → クライアント切断で自動キャンセル
    ctx := r.Context()

    data, err := fetchData(ctx)
    if err != nil {
        if ctx.Err() == context.Canceled {
            // クライアントが切断した → ログだけ
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

// context を下流に伝搬
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

### 並行処理パターンの比較表

| パターン | 用途 | エラー処理 | キャンセル | 複雑度 | 推奨度 |
|---------|------|-----------|-----------|--------|--------|
| goroutine + WaitGroup | 単純な並列処理 | 手動（チャネル等） | 手動（context） | 低 | 基本 |
| errgroup | 並列処理 + エラー集約 | 自動（最初のエラー） | 自動（context連携） | 低 | 推奨 |
| errgroup + SetLimit | 並列度制限 | 自動 | 自動 | 低 | 推奨 |
| Worker Pool | ジョブキュー | 手動 | context対応 | 中 | 特定用途 |
| Pipeline | ステージ処理 | ステージ間で伝搬 | context + done | 高 | 特定用途 |
| Fan-out/Fan-in | 分散処理 + 集約 | 集約時に処理 | context対応 | 高 | 特定用途 |

---

## 5. 構造体とメソッド設計

### コード例9: ゼロ値を有用にする

```go
// Go 標準ライブラリの優れたゼロ値設計

// sync.Mutex — ゼロ値ですぐ使える
var mu sync.Mutex
mu.Lock() // 初期化不要

// bytes.Buffer — ゼロ値ですぐ使える
var buf bytes.Buffer
buf.WriteString("hello")

// sync.Once — ゼロ値ですぐ使える
var once sync.Once
once.Do(func() { /* 初期化 */ })

// 自作の型もゼロ値を有用に設計する
type Logger struct {
    output io.Writer // nil なら os.Stderr を使う
    level  int       // 0 なら INFO レベル
    prefix string    // "" なら空文字
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

// ゼロ値で使える
var log Logger
log.Info("ready") // os.Stderr に INFO レベルで出力

// 明示的な初期化も可能
customLog := Logger{
    output: os.Stdout,
    level:  2,
    prefix: "[myapp] ",
}
customLog.Info("started")

// NG: ゼロ値が無効な設計
type BadConfig struct {
    MaxRetries int  // 0 = リトライしない? or 未設定?
    Timeout    time.Duration  // 0 = 即タイムアウト?
}

// OK: ゼロ値を区別する設計
type GoodConfig struct {
    MaxRetries *int           // nil = 未設定（デフォルト3）
    Timeout    time.Duration  // 0 = デフォルト（30秒）
}

func (c *GoodConfig) maxRetries() int {
    if c.MaxRetries == nil {
        return 3 // デフォルト
    }
    return *c.MaxRetries
}

func (c *GoodConfig) timeout() time.Duration {
    if c.Timeout == 0 {
        return 30 * time.Second // デフォルト
    }
    return c.Timeout
}
```

### コード例10: 機能オプションパターン（Functional Options）

```go
// 機能オプションパターンの完全な実装

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

// Option は Server の設定オプション
type Option func(*Server)

// WithReadTimeout は読み取りタイムアウトを設定する
func WithReadTimeout(d time.Duration) Option {
    return func(s *Server) { s.readTimeout = d }
}

// WithWriteTimeout は書き込みタイムアウトを設定する
func WithWriteTimeout(d time.Duration) Option {
    return func(s *Server) { s.writeTimeout = d }
}

// WithIdleTimeout はアイドルタイムアウトを設定する
func WithIdleTimeout(d time.Duration) Option {
    return func(s *Server) { s.idleTimeout = d }
}

// WithMaxConnections は最大接続数を設定する
func WithMaxConnections(n int) Option {
    return func(s *Server) { s.maxConn = n }
}

// WithLogger はロガーを設定する
func WithLogger(l *log.Logger) Option {
    return func(s *Server) { s.logger = l }
}

// WithTLS はTLS設定を適用する
func WithTLS(config *tls.Config) Option {
    return func(s *Server) { s.tlsConfig = config }
}

// WithMiddleware はミドルウェアを追加する
func WithMiddleware(mw ...Middleware) Option {
    return func(s *Server) { s.middleware = append(s.middleware, mw...) }
}

// NewServer は新しいサーバーを作成する
func NewServer(addr string, opts ...Option) *Server {
    s := &Server{
        addr:         addr,
        readTimeout:  5 * time.Second,   // デフォルト値
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

// 使用例
func main() {
    srv := NewServer(":8080",
        WithReadTimeout(30*time.Second),
        WithMaxConnections(1000),
        WithLogger(customLogger),
        WithMiddleware(loggingMW, authMW),
    )

    // デフォルト値で十分な場合
    simpleSrv := NewServer(":8080")
}

// 機能オプションパターンの利点:
// 1. 後方互換性: 新オプション追加が既存コードに影響しない
// 2. 可読性: 設定の意味が明確
// 3. デフォルト値: 未指定のオプションはデフォルト値を使用
// 4. バリデーション: Option関数内でバリデーション可能
// 5. ドキュメント: 各Withの関数にGoDocを書ける
```

### コード例11: Builder パターン（機能オプションの代替）

```go
// Builder パターン（チェーン形式）
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

// 使用例
srv, err := NewServerBuilder(":8080").
    ReadTimeout(30 * time.Second).
    MaxConnections(1000).
    Build()
```

---

## 6. テスタビリティの設計

### コード例12: テストしやすいコード

```go
// ===================================
// 原則: 依存を注入し、インターフェースで抽象化
// ===================================

// テスト困難なコード
// NG: 具体的な実装に直接依存
type UserServiceBad struct {
    db *sql.DB  // 具体型に依存 → テスト時にDBが必要
}

func (s *UserServiceBad) GetUser(id int64) (*User, error) {
    return s.db.QueryRow("SELECT ...") // DBに直接アクセス
}

// テスト容易なコード
// OK: インターフェースに依存
type UserRepository interface {
    GetByID(ctx context.Context, id int64) (*User, error)
}

type UserService struct {
    repo UserRepository  // インターフェースに依存
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

// テスト用モック
type mockUserRepo struct {
    users map[int64]*User
    err   error // テスト用のエラー注入
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

// テーブル駆動テスト
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

### コード例13: 時間のテスタビリティ

```go
// NG: time.Now() を直接呼ぶとテストが不安定
func (s *TokenService) IsExpired(token *Token) bool {
    return time.Now().After(token.ExpiresAt) // テスト時に制御不可
}

// OK: 時間関数を注入可能にする
type TokenService struct {
    now func() time.Time // テスト時に差し替え可能
}

func NewTokenService() *TokenService {
    return &TokenService{now: time.Now}
}

func (s *TokenService) IsExpired(token *Token) bool {
    return s.now().After(token.ExpiresAt)
}

// テスト
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

## 7. アンチパターン

### アンチパターン1: エラーを握りつぶす

```go
// NG: エラーを無視
data, _ := json.Marshal(user)    // マーシャルエラーを捨てている
f, _ := os.Open("config.yaml")  // ファイルが存在しない場合パニック
defer f.Close()

// OK: エラーを適切に処理
data, err := json.Marshal(user)
if err != nil {
    return fmt.Errorf("ユーザーのJSON変換に失敗: %w", err)
}

f, err := os.Open("config.yaml")
if err != nil {
    return fmt.Errorf("設定ファイルを開けません: %w", err)
}
defer f.Close()

// 唯一エラーを無視してよい場面:
// 1. defer での Close()（ただしログは出すべき）
defer func() {
    if err := f.Close(); err != nil {
        log.Printf("file close error: %v", err)
    }
}()
// 2. fmt.Fprint* 系（ほぼ失敗しない）
fmt.Fprintf(w, "hello")
```

### アンチパターン2: init() の乱用

```go
// NG: init() で複雑な初期化
func init() {
    db, err := sql.Open("postgres", os.Getenv("DATABASE_URL"))
    if err != nil {
        log.Fatal(err) // テスト時にも実行される
    }
    globalDB = db
}
// 問題:
// - テスト時にDBが必要
// - 初期化順序が不明確
// - エラー時にlog.Fatal → テストが落ちる

// OK: 明示的な初期化関数
func NewApp(cfg Config) (*App, error) {
    db, err := sql.Open("postgres", cfg.DatabaseURL)
    if err != nil {
        return nil, fmt.Errorf("DB接続失敗: %w", err)
    }
    return &App{db: db}, nil
}

// init() が適切な場面:
// - driver.Register (database/sql ドライバの登録)
// - flag.Parse の前提設定
// - 非常に単純な初期化（変数の初期値設定等）
```

### アンチパターン3: goroutine を Fire-and-Forget

```go
// NG: goroutine のライフサイクルを管理しない
func handler(w http.ResponseWriter, r *http.Request) {
    go sendEmail(user.Email) // パニックしても誰も気づかない
    w.WriteHeader(200)       // メール送信の結果を確認しない
}
// 問題:
// - パニック → プロセス全体が落ちる
// - エラー → 検知できない
// - メモリリーク → goroutineが終了しない可能性

// OK: errgroup や recover で管理
func handler(w http.ResponseWriter, r *http.Request) {
    g, ctx := errgroup.WithContext(r.Context())
    g.Go(func() error {
        return sendEmail(ctx, user.Email)
    })
    if err := g.Wait(); err != nil {
        log.Printf("email送信失敗: %v", err)
    }
    w.WriteHeader(200)
}

// OK: バックグラウンドタスク用のワーカー
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

### アンチパターン4: sync.Mutex の誤用

```go
// NG: Mutex をコピーする
type Cache struct {
    mu    sync.Mutex
    items map[string]string
}

func (c Cache) Get(key string) string {  // 値レシーバ → Mutexがコピーされる
    c.mu.Lock()
    defer c.mu.Unlock()
    return c.items[key]
}

// OK: ポインタレシーバを使う
func (c *Cache) Get(key string) string {
    c.mu.Lock()
    defer c.mu.Unlock()
    return c.items[key]
}

// NG: Mutex を公開する
type Cache struct {
    Mu    sync.Mutex  // 公開フィールド → 外部からロック可能
    Items map[string]string
}

// OK: Mutex を非公開にする
type Cache struct {
    mu    sync.Mutex
    items map[string]string
}

// NG: 読み取りにもMutexを使う（RWMutexを使うべき）
func (c *Cache) Get(key string) string {
    c.mu.Lock()       // 読み取りなのに排他ロック
    defer c.mu.Unlock()
    return c.items[key]
}

// OK: 読み取りにはRLockを使う
type Cache struct {
    mu    sync.RWMutex
    items map[string]string
}

func (c *Cache) Get(key string) string {
    c.mu.RLock()       // 読み取りロック（複数goroutineが同時に読める）
    defer c.mu.RUnlock()
    return c.items[key]
}

func (c *Cache) Set(key, value string) {
    c.mu.Lock()        // 書き込みロック（排他）
    defer c.mu.Unlock()
    c.items[key] = value
}
```

### アンチパターン5: 不要な else

```go
// NG: 冗長な else
func process(data []byte) error {
    if len(data) == 0 {
        return errors.New("empty data")
    } else {
        return parse(data)  // else 不要
    }
}

// OK: 早期リターン
func process(data []byte) error {
    if len(data) == 0 {
        return errors.New("empty data")
    }
    return parse(data)  // 正常パスがインデント0
}

// NG: ネストが深い
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

// OK: ガード節で早期リターン
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

## 8. パフォーマンス指針

### メモリアロケーション最適化

```go
// NG: ループ内でスライスを動的拡張
func processItems(items []Item) []Result {
    var results []Result  // 容量不明
    for _, item := range items {
        results = append(results, process(item))
        // → 容量が足りないたびにメモリ再割当 + コピー
    }
    return results
}

// OK: 事前に容量を確保
func processItems(items []Item) []Result {
    results := make([]Result, 0, len(items))  // 容量を事前確保
    for _, item := range items {
        results = append(results, process(item))
    }
    return results
}

// strings.Builder を使った文字列結合
// NG: += で結合（毎回新しい文字列を生成）
func buildString(parts []string) string {
    result := ""
    for _, p := range parts {
        result += p  // O(n^2) のメモリアロケーション
    }
    return result
}

// OK: strings.Builder（O(n)）
func buildString(parts []string) string {
    var b strings.Builder
    b.Grow(estimatedSize)  // 推定サイズを事前確保
    for _, p := range parts {
        b.WriteString(p)
    }
    return b.String()
}

// sync.Pool でオブジェクトを再利用
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
    // ... 処理
    return buf.String()
}
```

---

## FAQ

### Q1. Go のコードフォーマットはどうする？

`gofmt` はGoの公式フォーマッタで、議論の余地なく一律のスタイルを適用する。`goimports` は `gofmt` に加えて import の整理も行う。CI では `gofmt -l .` で未フォーマットのファイルがないか確認する。

```bash
# フォーマット
gofmt -w .
goimports -w .

# CIでのチェック
test -z "$(gofmt -l .)"
```

### Q2. リンターは何を使うべき？

`golangci-lint` が業界標準。`staticcheck`, `errcheck`, `govet`, `gosimple` など多数のリンターを統合実行できる。`.golangci.yml` でプロジェクトに合わせたルール設定が可能。

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
    - prealloc      # スライスの事前割当推奨
    - bodyclose     # HTTPレスポンスBody閉じ忘れ
    - nilerr        # nil エラーチェック
    - exportloopref # ループ変数キャプチャ
```

### Q3. パッケージ構成のベストプラクティスは？

小さなプロジェクトではフラット構成で十分。大規模では `internal/` で外部公開を制限し、ドメインごとにパッケージを分ける。循環依存を避け、依存の方向を一方向に保つ。`cmd/` 配下にエントリポイントを置く。

```
推奨構成:
  cmd/server/main.go          -- エントリポイント（薄く保つ）
  internal/                    -- 外部非公開
    handler/                   -- HTTPハンドラ
    service/                   -- ビジネスロジック
    repository/                -- データアクセス
    model/                     -- ドメインモデル
  pkg/                         -- 外部公開ライブラリ

依存の方向:
  main → handler → service → repository → model
  （一方向、循環なし）
```

### Q4. ポインタレシーバと値レシーバの使い分けは？

```
ポインタレシーバ (*T) を使う場面:
  → 構造体を変更する（setter）
  → 構造体が大きい（コピーコストが高い）
  → sync.Mutex 等コピー禁止のフィールドがある
  → 一貫性のため（1つでもポインタレシーバがあれば全てポインタ）

値レシーバ (T) を使う場面:
  → 構造体が小さく不変（Point{x, y}等）
  → map のキーとして使う型
  → time.Time のように値として扱う型
```

### Q5. Go 1.22 以降の主な改善点は？

- **ループ変数のスコープ修正**: `for i, v := range` のループ変数がイテレーションごとに新しく確保される
- **range over integers**: `for i := range 10` が可能に
- **Enhanced HTTP routing**: `http.ServeMux` がパスパラメータに対応
- **cmp パッケージ**: 比較関数の標準化

---

## まとめ

| 概念 | 要点 |
|------|------|
| 命名規則 | PascalCase/camelCase、頭字語は大文字維持、短いスコープ→短い名前 |
| インターフェース | 小さく保つ、使う側で定義、暗黙的実装 |
| エラーハンドリング | %w でラップ、Is/As で判定、ハンドラ層で1回ログ |
| ゼロ値 | 有用なゼロ値を設計する |
| Option パターン | 柔軟な初期化、後方互換性 |
| context | 第一引数、キャンセル伝搬、値は横断的関心事のみ |
| errgroup | goroutine管理の推奨パターン |
| gofmt / golangci-lint | コード品質の自動化、CIに組み込む |
| テスタビリティ | インターフェース + DI でモック可能に |
| パフォーマンス | スライス事前割当、strings.Builder、sync.Pool |
| パッケージ設計 | 一方向依存、循環禁止、internal で非公開 |
| 早期リターン | ガード節でネストを減らす |

---

## 次に読むべきガイド

- **02-web/04-testing.md** -- テスト：table-driven tests、testify、httptest
- **03-tools/01-generics.md** -- ジェネリクス：型パラメータ、制約
- **03-tools/02-profiling.md** -- プロファイリング：pprof、trace

---

## 参考文献

1. **Go公式 -- Effective Go** https://go.dev/doc/effective_go
2. **Go公式 -- Code Review Comments** https://go.dev/wiki/CodeReviewComments
3. **Go Blog -- Error handling and Go** https://go.dev/blog/error-handling-and-go
4. **Uber Go Style Guide** https://github.com/uber-go/guide/blob/master/style.md
5. **Go Proverbs** https://go-proverbs.github.io/
6. **100 Go Mistakes and How to Avoid Them** -- Teiva Harsanyi
7. **Go公式 -- Package Names** https://go.dev/blog/package-names
8. **golang.org/x/sync/errgroup** https://pkg.go.dev/golang.org/x/sync/errgroup
