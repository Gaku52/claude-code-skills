# Go ベストプラクティスガイド

> Effective Goの精神に基づき、保守性・可読性・性能を兼ね備えたGoコードを書くための指針

## この章で学ぶこと

1. **Effective Go** の核心 — Goらしいコードの設計原則と命名規則
2. **エラーハンドリング** — エラー値の設計、ラッピング、センチネルエラーの使い分け
3. **並行処理のパターン** — goroutine管理、チャネル設計、コンテキスト伝搬のベストプラクティス

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
|  "Clear is better than clever"                        |
|  「巧妙さより明快さ」                                  |
+-------------------------------------------------------+
```

### 命名規則

```
+----------------------------------+
|  Go の命名規則                    |
+----------------------------------+
|                                  |
|  パッケージ名: 短く、小文字      |
|    http, json, fmt, os           |
|    NG: httpUtil, jsonParser       |
|                                  |
|  エクスポート: PascalCase        |
|    ReadFile, HTTPClient, UserID  |
|                                  |
|  非エクスポート: camelCase       |
|    readFile, httpClient, userID  |
|                                  |
|  インターフェース: -er 接尾辞    |
|    Reader, Writer, Stringer      |
|    Closer, Formatter, Handler    |
|                                  |
|  頭字語: 全大文字を維持          |
|    HTTP, URL, ID, JSON, API      |
|    HTTPHandler (not HttpHandler) |
+----------------------------------+
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

// OK: パッケージ名を活用
package user
func Create() { ... }       // user.Create() で明確
```

---

## 2. エラーハンドリング

### コード例2: エラーの設計パターン

```go
package storage

import (
    "errors"
    "fmt"
)

// センチネルエラー: パッケージレベルで定義
var (
    ErrNotFound     = errors.New("storage: not found")
    ErrDuplicate    = errors.New("storage: duplicate entry")
    ErrUnauthorized = errors.New("storage: unauthorized")
)

// カスタムエラー型: 詳細情報を持つ
type ValidationError struct {
    Field   string
    Message string
}

func (e *ValidationError) Error() string {
    return fmt.Sprintf("validation error: %s - %s", e.Field, e.Message)
}

// エラーラッピング: コンテキスト付与
func (s *Store) FindUser(id string) (*User, error) {
    user, err := s.db.Get(id)
    if err != nil {
        if errors.Is(err, sql.ErrNoRows) {
            return nil, fmt.Errorf("FindUser(%s): %w", id, ErrNotFound)
        }
        return nil, fmt.Errorf("FindUser(%s): %w", id, err)
    }
    return user, nil
}

// エラー判定: errors.Is と errors.As
func handleError(err error) {
    // センチネルエラーの判定
    if errors.Is(err, ErrNotFound) {
        // 404 応答
    }

    // カスタムエラー型の判定
    var valErr *ValidationError
    if errors.As(err, &valErr) {
        // バリデーションエラー応答（フィールド情報を活用）
        fmt.Printf("Field: %s, Message: %s\n", valErr.Field, valErr.Message)
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
    |
    +-- 呼び出し元はエラーの種類を知る必要があるか？
            |
            +-- YES → センチネルエラー or カスタムエラー型
            |
            +-- NO → fmt.Errorf で文字列ラップのみ
```

### errors.Is vs errors.As 比較表

| 関数 | 目的 | 比較対象 | 使用場面 |
|------|------|---------|---------|
| `errors.Is(err, target)` | エラーチェーンに target と一致するエラーがあるか | 値（センチネルエラー） | `ErrNotFound`, `sql.ErrNoRows` |
| `errors.As(err, &target)` | エラーチェーンに target型のエラーがあるか | 型（カスタムエラー） | `*ValidationError`, `*os.PathError` |

---

## 3. インターフェース設計

### コード例3: 小さなインターフェースの原則

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

// 実践: 必要最小限のインターフェースを受け取る
// NG: 過大なインターフェースを要求
func ProcessData(rw ReadWriteCloser) error { ... }

// OK: 実際に使う機能だけを要求
func ProcessData(r Reader) error {
    buf := make([]byte, 1024)
    n, err := r.Read(buf)
    // ...
}
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
```

---

## 4. 並行処理のベストプラクティス

### コード例4: goroutine のライフサイクル管理

```go
package main

import (
    "context"
    "fmt"
    "sync"
    "time"
)

// Worker プール パターン
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
                    results <- process(job)
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

// errgroup パターン（推奨）
func fetchAll(ctx context.Context, urls []string) ([]Response, error) {
    g, ctx := errgroup.WithContext(ctx)
    responses := make([]Response, len(urls))

    for i, url := range urls {
        i, url := i, url
        g.Go(func() error {
            resp, err := fetch(ctx, url)
            if err != nil {
                return fmt.Errorf("fetch %s: %w", url, err)
            }
            responses[i] = resp
            return nil
        })
    }

    if err := g.Wait(); err != nil {
        return nil, err
    }
    return responses, nil
}
```

### コード例5: context の正しい伝搬

```go
// コンテキストの連鎖
func main() {
    // ルートコンテキスト（キャンセル可能）
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    // タイムアウト付きコンテキスト
    ctx, cancel = context.WithTimeout(ctx, 30*time.Second)
    defer cancel()

    // HTTPハンドラ内
    http.HandleFunc("/api/data", func(w http.ResponseWriter, r *http.Request) {
        // リクエストのコンテキストを使用（クライアント切断で自動キャンセル）
        ctx := r.Context()

        data, err := fetchData(ctx)
        if err != nil {
            if ctx.Err() == context.Canceled {
                return // クライアントが切断
            }
            http.Error(w, err.Error(), 500)
            return
        }
        json.NewEncoder(w).Encode(data)
    })
}

// context は第一引数に渡す（Go の規約）
func fetchData(ctx context.Context) (*Data, error) {
    // context を下流に伝搬
    req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
    if err != nil {
        return nil, err
    }
    resp, err := http.DefaultClient.Do(req)
    // ...
}
```

### 並行処理パターンの比較表

| パターン | 用途 | エラー処理 | キャンセル | 複雑度 |
|---------|------|-----------|-----------|--------|
| goroutine + WaitGroup | 単純な並列処理 | 手動（チャネル等） | 手動（context） | 低 |
| errgroup | 並列処理 + エラー集約 | 自動（最初のエラー） | 自動（context連携） | 低 |
| Worker Pool | 制限付き並列処理 | 手動 | context対応が必要 | 中 |
| Pipeline | ステージごとの処理 | ステージ間で伝搬 | context + done channel | 高 |
| Fan-out/Fan-in | 分散処理と結果集約 | 集約時に処理 | context対応が必要 | 高 |

---

## 5. 構造体とメソッド設計

### コード例6: ゼロ値を有用にする

```go
// sync.Mutex — ゼロ値ですぐ使える
var mu sync.Mutex
mu.Lock() // 初期化不要

// bytes.Buffer — ゼロ値ですぐ使える
var buf bytes.Buffer
buf.WriteString("hello")

// 自作の型もゼロ値を有用に設計する
type Logger struct {
    output io.Writer // nil なら os.Stderr を使う
    level  int       // 0 なら INFO レベル
}

func (l *Logger) writer() io.Writer {
    if l.output == nil {
        return os.Stderr
    }
    return l.output
}

// ゼロ値で使える
var log Logger
log.Info("ready") // os.Stderr に INFO レベルで出力
```

### コード例7: 機能オプションパターン

```go
type Server struct {
    addr         string
    readTimeout  time.Duration
    writeTimeout time.Duration
    maxConn      int
    logger       *log.Logger
}

type Option func(*Server)

func WithReadTimeout(d time.Duration) Option {
    return func(s *Server) { s.readTimeout = d }
}

func WithWriteTimeout(d time.Duration) Option {
    return func(s *Server) { s.writeTimeout = d }
}

func WithMaxConnections(n int) Option {
    return func(s *Server) { s.maxConn = n }
}

func WithLogger(l *log.Logger) Option {
    return func(s *Server) { s.logger = l }
}

func NewServer(addr string, opts ...Option) *Server {
    s := &Server{
        addr:         addr,
        readTimeout:  5 * time.Second,   // デフォルト値
        writeTimeout: 10 * time.Second,
        maxConn:      100,
        logger:       log.Default(),
    }
    for _, opt := range opts {
        opt(s)
    }
    return s
}

// 使用例
srv := NewServer(":8080",
    WithReadTimeout(30*time.Second),
    WithMaxConnections(1000),
)
```

---

## 6. アンチパターン

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

// OK: 明示的な初期化関数
func NewApp(cfg Config) (*App, error) {
    db, err := sql.Open("postgres", cfg.DatabaseURL)
    if err != nil {
        return nil, fmt.Errorf("DB接続失敗: %w", err)
    }
    return &App{db: db}, nil
}
```

### アンチパターン3: goroutine を Fire-and-Forget

```go
// NG: goroutine のライフサイクルを管理しない
func handler(w http.ResponseWriter, r *http.Request) {
    go sendEmail(user.Email) // パニックしても誰も気づかない
    w.WriteHeader(200)
}

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
```

---

## FAQ

### Q1. Go のコードフォーマットはどうする？

`gofmt` はGoの公式フォーマッタで、議論の余地なく一律のスタイルを適用する。`goimports` は `gofmt` に加えて import の整理も行う。CI では `gofmt -l .` で未フォーマットのファイルがないか確認する。エディタには保存時自動フォーマットを設定する。

### Q2. リンターは何を使うべき？

`golangci-lint` が業界標準。`staticcheck`, `errcheck`, `govet`, `gosimple` など多数のリンターを統合実行できる。`.golangci.yml` でプロジェクトに合わせたルール設定が可能。CIに組み込むことで品質を自動的に担保する。

### Q3. パッケージ構成のベストプラクティスは？

小さなプロジェクトではフラット構成（`main.go`, `handler.go`, `store.go`）で十分。大規模では `internal/` ディレクトリで外部公開を制限し、ドメインごとにパッケージを分ける。循環依存を避け、依存の方向を一方向に保つ。`cmd/` 配下にエントリポイントを置く。

---

## まとめ

| 概念 | 要点 |
|------|------|
| 命名規則 | PascalCase/camelCase、頭字語は大文字維持 |
| インターフェース | 小さく保つ、受け取り側で定義 |
| エラーハンドリング | %w でラップ、Is/As で判定 |
| ゼロ値 | 有用なゼロ値を設計する |
| Option パターン | 柔軟な初期化パラメータ |
| context | 第一引数、キャンセル伝搬 |
| errgroup | goroutine管理の標準パターン |
| gofmt / golangci-lint | コード品質の自動化 |

---

## 次に読むべきガイド

- **02-web/04-testing.md** — テスト：table-driven tests、testify、httptest
- **03-tools/01-generics.md** — ジェネリクス：型パラメータ、制約
- **03-tools/02-profiling.md** — プロファイリング：pprof、trace

---

## 参考文献

1. **Go公式 — Effective Go** https://go.dev/doc/effective_go
2. **Go公式 — Code Review Comments** https://go.dev/wiki/CodeReviewComments
3. **Go Blog — Error handling and Go** https://go.dev/blog/error-handling-and-go
4. **Uber Go Style Guide** https://github.com/uber-go/guide/blob/master/style.md
