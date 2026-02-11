# net/http -- Go標準HTTPサーバー

> net/httpはGoの標準ライブラリでHTTPサーバー/クライアントを実装し、Handler・ServeMux・ミドルウェアパターンで本番品質のWebサービスを構築できる。

---

## この章で学ぶこと

1. **Handler / HandlerFunc** -- HTTPリクエスト処理の基本
2. **ServeMux (Go 1.22+)** -- 強化されたルーティング
3. **ミドルウェアパターン** -- 横断的関心事の分離

---

### コード例 1: 基本的なHTTPサーバー

```go
func main() {
    mux := http.NewServeMux()
    mux.HandleFunc("GET /hello", func(w http.ResponseWriter, r *http.Request) {
        fmt.Fprintf(w, "Hello, World!")
    })

    server := &http.Server{
        Addr:         ":8080",
        Handler:      mux,
        ReadTimeout:  10 * time.Second,
        WriteTimeout: 10 * time.Second,
        IdleTimeout:  120 * time.Second,
    }
    log.Fatal(server.ListenAndServe())
}
```

### コード例 2: Go 1.22+ パターンマッチング

```go
mux := http.NewServeMux()

// メソッド + パスパターン
mux.HandleFunc("GET /users", listUsers)
mux.HandleFunc("POST /users", createUser)
mux.HandleFunc("GET /users/{id}", getUser)
mux.HandleFunc("PUT /users/{id}", updateUser)
mux.HandleFunc("DELETE /users/{id}", deleteUser)

func getUser(w http.ResponseWriter, r *http.Request) {
    id := r.PathValue("id") // Go 1.22+ パスパラメータ
    fmt.Fprintf(w, "User: %s", id)
}
```

### コード例 3: ミドルウェア

```go
func loggingMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        next.ServeHTTP(w, r)
        log.Printf("%s %s %v", r.Method, r.URL.Path, time.Since(start))
    })
}

func authMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        token := r.Header.Get("Authorization")
        if token == "" {
            http.Error(w, "unauthorized", http.StatusUnauthorized)
            return
        }
        next.ServeHTTP(w, r)
    })
}

func main() {
    mux := http.NewServeMux()
    mux.HandleFunc("GET /api/data", getData)

    handler := loggingMiddleware(authMiddleware(mux))
    http.ListenAndServe(":8080", handler)
}
```

### コード例 4: JSONレスポンス

```go
type User struct {
    ID    int    `json:"id"`
    Name  string `json:"name"`
    Email string `json:"email"`
}

func getUser(w http.ResponseWriter, r *http.Request) {
    user := User{ID: 1, Name: "Tanaka", Email: "tanaka@example.com"}

    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(http.StatusOK)
    if err := json.NewEncoder(w).Encode(user); err != nil {
        http.Error(w, "encode error", http.StatusInternalServerError)
    }
}

func createUser(w http.ResponseWriter, r *http.Request) {
    var user User
    if err := json.NewDecoder(r.Body).Decode(&user); err != nil {
        http.Error(w, "invalid json", http.StatusBadRequest)
        return
    }
    // ... 保存処理
    w.WriteHeader(http.StatusCreated)
    json.NewEncoder(w).Encode(user)
}
```

### コード例 5: httptest によるテスト

```go
func TestGetUser(t *testing.T) {
    req := httptest.NewRequest("GET", "/users/1", nil)
    rec := httptest.NewRecorder()

    getUser(rec, req)

    resp := rec.Result()
    if resp.StatusCode != http.StatusOK {
        t.Errorf("expected 200, got %d", resp.StatusCode)
    }

    var user User
    json.NewDecoder(resp.Body).Decode(&user)
    if user.Name != "Tanaka" {
        t.Errorf("expected Tanaka, got %s", user.Name)
    }
}
```

---

## 2. ASCII図解

### 図1: HTTPリクエスト処理フロー

```
Client Request
     │
     ▼
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ Logging  │──>│ Auth     │──>│ ServeMux │──>│ Handler  │
│ MW       │   │ MW       │   │ (Router) │   │ (Logic)  │
└──────────┘   └──────────┘   └──────────┘   └──────────┘
     ▲                                              │
     │              Response                        │
     └──────────────────────────────────────────────┘
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

logging(auth(cors(mux)))

リクエスト方向 →
┌─────────────────────────────────────┐
│ logging                              │
│  ┌──────────────────────────────┐   │
│  │ auth                         │   │
│  │  ┌──────────────────────┐   │   │
│  │  │ cors                  │   │   │
│  │  │  ┌──────────────┐   │   │   │
│  │  │  │ ServeMux     │   │   │   │
│  │  │  │  → Handler   │   │   │   │
│  │  │  └──────────────┘   │   │   │
│  │  └──────────────────────┘   │   │
│  └──────────────────────────────┘   │
└─────────────────────────────────────┘
← レスポンス方向
```

---

## 3. 比較表

### 表1: Go 1.21以前 vs Go 1.22以降のServeMux

| 機能 | Go 1.21以前 | Go 1.22以降 |
|------|------------|------------|
| メソッド指定 | 不可 (手動チェック) | `"GET /path"` |
| パスパラメータ | 不可 (サードパーティ必要) | `"/users/{id}"` |
| ワイルドカード | 不可 | `"/files/{path...}"` |
| 優先順位 | 最長一致 | 最も具体的なパターン |
| 外部依存 | gorilla/mux, chi等が必要 | 標準ライブラリで十分 |

### 表2: 標準net/http vs サードパーティフレームワーク

| 項目 | net/http | Gin | Echo |
|------|----------|-----|------|
| パフォーマンス | 高 | 非常に高 | 非常に高 |
| ルーティング | 基本的 (1.22+で強化) | Radix tree | Radix tree |
| バリデーション | なし | binding/validator | validator |
| Swagger生成 | 手動 | swaggo対応 | swaggo対応 |
| 学習コスト | 低 | 低 | 低 |
| 依存 | なし | あり | あり |

---

## 4. アンチパターン

### アンチパターン 1: タイムアウト未設定のサーバー

```go
// BAD: デフォルトのhttp.ListenAndServe
http.ListenAndServe(":8080", mux) // タイムアウトなし → Slowloris攻撃に脆弱

// GOOD: タイムアウトを明示的に設定
server := &http.Server{
    Addr:         ":8080",
    Handler:      mux,
    ReadTimeout:  10 * time.Second,
    WriteTimeout: 10 * time.Second,
    IdleTimeout:  120 * time.Second,
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

---

## 5. FAQ

### Q1: net/httpだけで本番サーバーは構築可能か？

Go 1.22以降であれば十分に可能。パスパラメータ・メソッドマッチング・ワイルドカードが標準サポートされた。ただし、バリデーション・Swagger自動生成・BindJSONなどはサードパーティが必要。

### Q2: Graceful Shutdownはどう実装するか？

`server.Shutdown(ctx)` を使う。SIGTERMをキャッチし、処理中のリクエストの完了を待ってからシャットダウンする。Kubernetes環境では必須のパターン。

### Q3: http.Clientにもタイムアウトは必要か？

必須。デフォルトの`http.DefaultClient`はタイムアウトなし。`&http.Client{Timeout: 30 * time.Second}` またはContextのタイムアウトを使う。

---

## まとめ

| 概念 | 要点 |
|------|------|
| Handler | `ServeHTTP(w, r)` を持つインターフェース |
| ServeMux | Go 1.22+でメソッド・パスパラメータ対応 |
| ミドルウェア | `func(http.Handler) http.Handler` パターン |
| Server設定 | タイムアウトは必ず設定する |
| httptest | テスト用のRecorder/Request |

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
