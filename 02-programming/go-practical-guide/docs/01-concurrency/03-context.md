# Context -- キャンセル・タイムアウト・値の伝搬

> context.Contextはgoroutine間でキャンセルシグナル・タイムアウト・リクエストスコープの値を伝搬するための標準メカニズムである。

---

## この章で学ぶこと

1. **context.WithCancel** -- 手動キャンセル
2. **context.WithTimeout / WithDeadline** -- タイムアウト制御
3. **context.WithValue** -- 値の伝搬とベストプラクティス

---

### コード例 1: context.WithCancel

```go
func main() {
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    go func() {
        for {
            select {
            case <-ctx.Done():
                fmt.Println("cancelled:", ctx.Err())
                return
            default:
                doWork()
            }
        }
    }()

    time.Sleep(2 * time.Second)
    cancel() // goroutineにキャンセルを通知
}
```

### コード例 2: context.WithTimeout

```go
func fetchWithTimeout(url string) ([]byte, error) {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel() // タイムアウト前に完了しても必ずcancelを呼ぶ

    req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
    if err != nil {
        return nil, err
    }

    resp, err := http.DefaultClient.Do(req)
    if err != nil {
        return nil, err // タイムアウト時: context deadline exceeded
    }
    defer resp.Body.Close()
    return io.ReadAll(resp.Body)
}
```

### コード例 3: context.WithDeadline

```go
func processUntilDeadline() error {
    deadline := time.Now().Add(30 * time.Second)
    ctx, cancel := context.WithDeadline(context.Background(), deadline)
    defer cancel()

    for {
        select {
        case <-ctx.Done():
            return ctx.Err() // context.DeadlineExceeded
        default:
            if err := processNextItem(ctx); err != nil {
                return err
            }
        }
    }
}
```

### コード例 4: context.WithValue

```go
type contextKey string

const (
    requestIDKey contextKey = "requestID"
    userIDKey    contextKey = "userID"
)

func middleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        ctx := context.WithValue(r.Context(), requestIDKey, uuid.New().String())
        next.ServeHTTP(w, r.WithContext(ctx))
    })
}

func handler(w http.ResponseWriter, r *http.Request) {
    reqID := r.Context().Value(requestIDKey).(string)
    log.Printf("[%s] handling request", reqID)
}
```

### コード例 5: Contextの伝搬チェーン

```go
func handleRequest(w http.ResponseWriter, r *http.Request) {
    // HTTPリクエストのContextを基盤にする
    ctx := r.Context()

    // タイムアウトを追加
    ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
    defer cancel()

    // サービス層に伝搬
    user, err := userService.GetUser(ctx, userID)
    if err != nil {
        // ctx.Err() == context.Canceled → クライアント切断
        // ctx.Err() == context.DeadlineExceeded → タイムアウト
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }
    json.NewEncoder(w).Encode(user)
}

func (s *UserService) GetUser(ctx context.Context, id int) (*User, error) {
    // DBクエリにContextを渡す
    return s.db.QueryRowContext(ctx, "SELECT * FROM users WHERE id=$1", id).Scan(...)
}
```

---

## 2. ASCII図解

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

---

## 3. 比較表

### 表1: Context生成関数

| 関数 | 用途 | Done()の発火条件 |
|------|------|-----------------|
| `context.Background()` | ルート。main, init, テスト | 発火しない |
| `context.TODO()` | 未決定の一時的プレースホルダ | 発火しない |
| `WithCancel(parent)` | 手動キャンセル | cancel()呼び出し |
| `WithTimeout(parent, d)` | 時間制限 | d経過 or cancel() |
| `WithDeadline(parent, t)` | 絶対時刻制限 | t到達 or cancel() |
| `WithValue(parent, k, v)` | 値の伝搬 | 親に依存 |

### 表2: context.Err() の戻り値

| 状態 | ctx.Err() | ctx.Done() |
|------|-----------|-----------|
| 未キャンセル | nil | ブロック |
| cancel()済み | context.Canceled | クローズ済み |
| タイムアウト | context.DeadlineExceeded | クローズ済み |

---

## 4. アンチパターン

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
    ...
}
```

---

## 5. FAQ

### Q1: context.Background()とcontext.TODO()の違いは？

機能的には同じだが意図が異なる。`Background()`は「ルートcontextとして意図的に使う」、`TODO()`は「適切なcontextがまだ不明で後で修正する」場合に使う。linterで`TODO()`を検出して漏れを防げる。

### Q2: cancel関数は必ず呼ぶ必要があるか？

はい。`WithCancel`/`WithTimeout`/`WithDeadline`のcancel関数を呼ばないとリソースリークが発生する。`defer cancel()` を取得直後に書くのが慣例。タイムアウトで自動キャンセルされても`cancel()`は安全に呼べる。

### Q3: contextの値はどのような場面で使うべきか？

リクエストスコープの横断的関心事のみ: トレースID、認証情報、ロケール等。ビジネスロジックのパラメータには使わない。独自のキー型（`type contextKey string`）を定義してキーの衝突を防ぐ。

---

## まとめ

| 概念 | 要点 |
|------|------|
| Context | goroutine間のキャンセル・タイムアウト・値伝搬 |
| WithCancel | 手動キャンセル制御 |
| WithTimeout | 時間制限付き処理 |
| WithDeadline | 絶対時刻によるデッドライン |
| WithValue | 横断的関心事の伝搬のみに使う |
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
