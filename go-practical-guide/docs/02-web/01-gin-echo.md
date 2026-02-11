# Gin / Echo -- Go Webフレームワーク

> GinとEchoはGoで最も人気のあるWebフレームワークであり、高速ルーティング・ミドルウェア・バリデーション・Swagger連携を提供する。

---

## この章で学ぶこと

1. **Gin / Echo の基本** -- ルーティングとハンドラ
2. **ミドルウェアとバリデーション** -- 横断的関心事の実装
3. **Swagger / OpenAPI** -- API仕様の自動生成

---

### コード例 1: Gin 基本セットアップ

```go
import "github.com/gin-gonic/gin"

func main() {
    r := gin.Default() // Logger + Recovery ミドルウェア付き

    r.GET("/users", listUsers)
    r.POST("/users", createUser)
    r.GET("/users/:id", getUser)
    r.PUT("/users/:id", updateUser)
    r.DELETE("/users/:id", deleteUser)

    r.Run(":8080")
}

func getUser(c *gin.Context) {
    id := c.Param("id")
    c.JSON(http.StatusOK, gin.H{"id": id, "name": "Tanaka"})
}
```

### コード例 2: Echo 基本セットアップ

```go
import "github.com/labstack/echo/v4"

func main() {
    e := echo.New()
    e.Use(middleware.Logger())
    e.Use(middleware.Recover())

    e.GET("/users", listUsers)
    e.POST("/users", createUser)
    e.GET("/users/:id", getUser)

    e.Logger.Fatal(e.Start(":8080"))
}

func getUser(c echo.Context) error {
    id := c.Param("id")
    return c.JSON(http.StatusOK, map[string]string{"id": id})
}
```

### コード例 3: Gin バリデーション

```go
type CreateUserRequest struct {
    Name     string `json:"name" binding:"required,min=2,max=50"`
    Email    string `json:"email" binding:"required,email"`
    Age      int    `json:"age" binding:"gte=0,lte=150"`
    Password string `json:"password" binding:"required,min=8"`
}

func createUser(c *gin.Context) {
    var req CreateUserRequest
    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
        return
    }
    // バリデーション通過
    c.JSON(http.StatusCreated, gin.H{"name": req.Name})
}
```

### コード例 4: Gin ミドルウェアグループ

```go
func main() {
    r := gin.Default()

    // 公開API
    public := r.Group("/api/v1")
    {
        public.POST("/login", login)
        public.POST("/register", register)
    }

    // 認証必須API
    authorized := r.Group("/api/v1")
    authorized.Use(authMiddleware())
    {
        authorized.GET("/profile", getProfile)
        authorized.PUT("/profile", updateProfile)
    }

    // 管理者API
    admin := r.Group("/api/v1/admin")
    admin.Use(authMiddleware(), adminMiddleware())
    {
        admin.GET("/users", adminListUsers)
    }
}

func authMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        token := c.GetHeader("Authorization")
        if token == "" {
            c.AbortWithStatusJSON(401, gin.H{"error": "unauthorized"})
            return
        }
        claims, err := validateToken(token)
        if err != nil {
            c.AbortWithStatusJSON(401, gin.H{"error": "invalid token"})
            return
        }
        c.Set("userID", claims.UserID)
        c.Next()
    }
}
```

### コード例 5: Echo カスタムバリデータ

```go
import "github.com/go-playground/validator/v10"

type CustomValidator struct {
    validator *validator.Validate
}

func (cv *CustomValidator) Validate(i interface{}) error {
    return cv.validator.Struct(i)
}

func main() {
    e := echo.New()
    e.Validator = &CustomValidator{validator: validator.New()}

    e.POST("/users", func(c echo.Context) error {
        u := new(CreateUserRequest)
        if err := c.Bind(u); err != nil {
            return echo.NewHTTPError(http.StatusBadRequest, err.Error())
        }
        if err := c.Validate(u); err != nil {
            return echo.NewHTTPError(http.StatusBadRequest, err.Error())
        }
        return c.JSON(http.StatusCreated, u)
    })
}
```

---

## 2. ASCII図解

### 図1: Ginのリクエスト処理フロー

```
Request
  │
  ▼
┌──────────────┐
│ gin.Engine   │
│ ┌──────────┐ │
│ │ Logger   │ │  Global Middleware
│ │ Recovery │ │
│ └────┬─────┘ │
│      ▼       │
│ ┌──────────┐ │
│ │ RadixTree│ │  Route Matching
│ │ Router   │ │  O(1) パス探索
│ └────┬─────┘ │
│      ▼       │
│ ┌──────────┐ │
│ │ Group MW │ │  Group Middleware
│ └────┬─────┘ │
│      ▼       │
│ ┌──────────┐ │
│ │ Handler  │ │  Business Logic
│ └──────────┘ │
└──────────────┘
```

### 図2: ルーティンググループ

```
/api/v1
├── /login          [POST]  (公開)
├── /register       [POST]  (公開)
├── /profile        [GET]   (認証必須)
├── /profile        [PUT]   (認証必須)
└── /admin
    └── /users      [GET]   (認証+管理者権限)

ミドルウェアの適用:
  公開:     Logger → Recovery → Handler
  認証必須:  Logger → Recovery → Auth → Handler
  管理者:   Logger → Recovery → Auth → Admin → Handler
```

### 図3: バリデーション処理フロー

```
JSON Request Body
      │
      ▼
┌──────────────┐
│ Bind (JSON)  │ ── 構文エラー → 400 Bad Request
└──────┬───────┘
       ▼
┌──────────────┐
│ Validate     │ ── バリデーション ── 失敗 → 400 + エラー詳細
│  required    │    エラー
│  min/max     │
│  email       │
└──────┬───────┘
       ▼
  Handler Logic
```

---

## 3. 比較表

### 表1: Gin vs Echo vs 標準net/http

| 項目 | Gin | Echo | net/http (1.22+) |
|------|-----|------|-----------------|
| パフォーマンス | 非常に高速 | 非常に高速 | 高速 |
| ルーティング | Radix tree | Radix tree | パターンマッチ |
| バリデーション | binding (validator) | 別途追加 | なし |
| ミドルウェア | `gin.HandlerFunc` | `echo.MiddlewareFunc` | `func(http.Handler) http.Handler` |
| エラーハンドリング | `c.AbortWithJSON` | `echo.HTTPError` | `http.Error` |
| GitHub Stars | 80k+ | 30k+ | 標準 |
| 依存の少なさ | 中 | 中 | 依存なし |

### 表2: Gin バリデーションタグ

| タグ | 意味 | 例 |
|------|------|-----|
| `required` | 必須 | `binding:"required"` |
| `email` | メール形式 | `binding:"email"` |
| `min` | 最小値/最小長 | `binding:"min=3"` |
| `max` | 最大値/最大長 | `binding:"max=100"` |
| `oneof` | 列挙値 | `binding:"oneof=admin user"` |
| `gte` | 以上 | `binding:"gte=0"` |
| `url` | URL形式 | `binding:"url"` |

---

## 4. アンチパターン

### アンチパターン 1: コンテキストの漏洩

```go
// BAD: gin.Context をgoroutineに渡す
func handler(c *gin.Context) {
    go func() {
        time.Sleep(5 * time.Second)
        c.JSON(200, gin.H{"ok": true}) // レスポンスは既に返却済みの可能性
    }()
}

// GOOD: 必要な値をコピーしてからgoroutineに渡す
func handler(c *gin.Context) {
    userID := c.GetString("userID")
    go func() {
        processAsync(userID)
    }()
    c.JSON(200, gin.H{"accepted": true})
}
```

### アンチパターン 2: エラーレスポンスの不統一

```go
// BAD: エラーレスポンスのフォーマットがバラバラ
c.JSON(400, gin.H{"error": "bad request"})
c.JSON(400, gin.H{"message": "invalid input"})
c.JSON(400, "error occurred")

// GOOD: 統一されたエラーレスポンス型
type ErrorResponse struct {
    Code    string `json:"code"`
    Message string `json:"message"`
}

func respondError(c *gin.Context, status int, code, msg string) {
    c.JSON(status, ErrorResponse{Code: code, Message: msg})
}
```

---

## 5. FAQ

### Q1: GinとEchoのどちらを選ぶべきか？

両者の性能差はほぼなし。Ginはエコシステムが大きく情報が多い。Echoはコード設計がクリーンでerrorを返すハンドラが特徴的。チームの好みで選んでよいが、Go 1.22+の標準net/httpで十分なケースも多い。

### Q2: Gin Releaseモードとは？

`gin.SetMode(gin.ReleaseMode)` で本番モードに切り替える。デバッグログが抑制され、パフォーマンスが若干向上する。環境変数 `GIN_MODE=release` でも設定可能。

### Q3: Swagger/OpenAPIはどう統合するか？

`swaggo/swag` を使い、ハンドラのコメントからSwagger仕様を自動生成する。`gin-swagger` または `echo-swagger` で `/swagger/index.html` を提供する。

---

## まとめ

| 概念 | 要点 |
|------|------|
| Gin | 高速・大エコシステム。gin.H、binding |
| Echo | クリーン設計。error戻り値パターン |
| ルーティング | Radix tree で高速パスマッチ |
| バリデーション | go-playground/validator ベース |
| ミドルウェア | グループ単位で適用可能 |
| Swagger | swaggo で自動生成 |

---

## 次に読むべきガイド

- [02-database.md](./02-database.md) -- データベース接続
- [03-grpc.md](./03-grpc.md) -- gRPC
- [04-testing.md](./04-testing.md) -- テスト

---

## 参考文献

1. **Gin Web Framework** -- https://gin-gonic.com/docs/
2. **Echo -- High performance, extensible, minimalist Go web framework** -- https://echo.labstack.com/
3. **swaggo/swag** -- https://github.com/swaggo/swag
