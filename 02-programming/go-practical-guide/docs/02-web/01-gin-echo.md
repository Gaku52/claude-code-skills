# Gin / Echo -- Go Webフレームワーク

> GinとEchoはGoで最も人気のあるWebフレームワークであり、高速ルーティング・ミドルウェア・バリデーション・Swagger連携を提供する。

---

## この章で学ぶこと

1. **Gin / Echo の基本** -- ルーティングとハンドラ
2. **ミドルウェアとバリデーション** -- 横断的関心事の実装
3. **Swagger / OpenAPI** -- API仕様の自動生成
4. **テスト** -- ハンドラとミドルウェアのテスト手法
5. **本番運用** -- Graceful Shutdown・構造化ログ・ヘルスチェック

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

Gin の `Default()` は `Logger` と `Recovery` ミドルウェアが自動で組み込まれたエンジンを返す。`New()` を使えばミドルウェアなしの素のエンジンを取得できる。本番環境では `New()` を使い、必要なミドルウェアを明示的に追加するのが推奨される。

ルーティングパラメータは `:id` の形式で定義し、`c.Param("id")` で取得する。ワイルドカードパラメータは `*filepath` の形式で、`c.Param("filepath")` として取得できる。

```go
// ワイルドカードルーティングの例
r.GET("/files/*filepath", func(c *gin.Context) {
    filepath := c.Param("filepath")
    // filepath = "/images/logo.png" のように先頭スラッシュ付き
    c.String(http.StatusOK, "Serving: %s", filepath)
})
```

クエリパラメータの取得方法も複数ある。

```go
func listUsers(c *gin.Context) {
    // クエリパラメータ
    page := c.DefaultQuery("page", "1")
    limit := c.DefaultQuery("limit", "20")
    sort := c.Query("sort") // 空文字列がデフォルト

    // 数値変換
    pageNum, err := strconv.Atoi(page)
    if err != nil || pageNum < 1 {
        pageNum = 1
    }
    limitNum, err := strconv.Atoi(limit)
    if err != nil || limitNum < 1 || limitNum > 100 {
        limitNum = 20
    }

    // ページネーション付きレスポンス
    c.JSON(http.StatusOK, gin.H{
        "page":  pageNum,
        "limit": limitNum,
        "sort":  sort,
        "users": []gin.H{},
    })
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

Echo の大きな特徴はハンドラが `error` を返す点である。これによりエラーハンドリングを集約でき、ハンドラ内で `c.JSON()` を呼んだ後に `return nil` を忘れるバグを防げる。

```go
// Echo のエラーハンドリングはハンドラの戻り値で制御できる
func getUser(c echo.Context) error {
    id := c.Param("id")
    user, err := userService.FindByID(c.Request().Context(), id)
    if err != nil {
        if errors.Is(err, ErrNotFound) {
            return echo.NewHTTPError(http.StatusNotFound, "user not found")
        }
        return echo.NewHTTPError(http.StatusInternalServerError, err.Error())
    }
    return c.JSON(http.StatusOK, user)
}
```

Echo ではルーティングパラメータの他に、パスセグメント以降すべてを取得するワイルドカードもサポートしている。

```go
// Echo ワイルドカード
e.GET("/files/*", func(c echo.Context) error {
    filepath := c.Param("*")
    return c.String(http.StatusOK, "Serving: "+filepath)
})

// クエリパラメータ
func listUsers(c echo.Context) error {
    page := c.QueryParam("page")
    limit := c.QueryParam("limit")
    if page == "" {
        page = "1"
    }
    if limit == "" {
        limit = "20"
    }
    return c.JSON(http.StatusOK, map[string]string{
        "page":  page,
        "limit": limit,
    })
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

Gin のバリデーションは内部で `go-playground/validator` を使用している。カスタムバリデーションルールの登録も可能である。

```go
// カスタムバリデーションルールの登録
func setupValidator() {
    if v, ok := binding.Validator.Engine().(*validator.Validate); ok {
        // カスタムバリデーション: 日本の電話番号
        v.RegisterValidation("jpphone", func(fl validator.FieldLevel) bool {
            phone := fl.Field().String()
            matched, _ := regexp.MatchString(`^0\d{1,4}-?\d{1,4}-?\d{4}$`, phone)
            return matched
        })

        // カスタムバリデーション: パスワード強度
        v.RegisterValidation("strongpassword", func(fl validator.FieldLevel) bool {
            password := fl.Field().String()
            hasUpper := regexp.MustCompile(`[A-Z]`).MatchString(password)
            hasLower := regexp.MustCompile(`[a-z]`).MatchString(password)
            hasNumber := regexp.MustCompile(`[0-9]`).MatchString(password)
            hasSpecial := regexp.MustCompile(`[!@#$%^&*]`).MatchString(password)
            return hasUpper && hasLower && hasNumber && hasSpecial
        })

        // JSON タグ名をエラーメッセージに使用
        v.RegisterTagNameFunc(func(fld reflect.StructField) string {
            name := strings.SplitN(fld.Tag.Get("json"), ",", 2)[0]
            if name == "-" {
                return ""
            }
            return name
        })
    }
}

// カスタムバリデーションを使用する構造体
type RegisterRequest struct {
    Name     string `json:"name" binding:"required,min=2,max=50"`
    Email    string `json:"email" binding:"required,email"`
    Phone    string `json:"phone" binding:"required,jpphone"`
    Password string `json:"password" binding:"required,min=8,strongpassword"`
}
```

バリデーションエラーのメッセージをユーザーフレンドリーに変換する関数も重要である。

```go
// バリデーションエラーの変換
func formatValidationErrors(err error) []map[string]string {
    var ve validator.ValidationErrors
    if !errors.As(err, &ve) {
        return []map[string]string{{"error": err.Error()}}
    }

    errs := make([]map[string]string, len(ve))
    for i, fe := range ve {
        errs[i] = map[string]string{
            "field":   fe.Field(),
            "tag":     fe.Tag(),
            "value":   fmt.Sprintf("%v", fe.Value()),
            "message": validationMessage(fe),
        }
    }
    return errs
}

func validationMessage(fe validator.FieldError) string {
    switch fe.Tag() {
    case "required":
        return fmt.Sprintf("%s は必須です", fe.Field())
    case "email":
        return fmt.Sprintf("%s は有効なメールアドレスではありません", fe.Field())
    case "min":
        return fmt.Sprintf("%s は %s 以上である必要があります", fe.Field(), fe.Param())
    case "max":
        return fmt.Sprintf("%s は %s 以下である必要があります", fe.Field(), fe.Param())
    case "jpphone":
        return fmt.Sprintf("%s は有効な日本の電話番号ではありません", fe.Field())
    case "strongpassword":
        return "パスワードには大文字・小文字・数字・特殊文字が必要です"
    default:
        return fmt.Sprintf("%s は %s を満たしていません", fe.Field(), fe.Tag())
    }
}

// ハンドラでの使用
func createUser(c *gin.Context) {
    var req CreateUserRequest
    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(http.StatusBadRequest, gin.H{
            "code":   "VALIDATION_ERROR",
            "errors": formatValidationErrors(err),
        })
        return
    }
    // ...
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

    r.Run(":8080")
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

ミドルウェアの実行順序は重要である。`c.Next()` を呼ぶと後続のミドルウェアとハンドラが実行され、その後に `c.Next()` 以降のコードが実行される。`c.Abort()` を呼ぶとチェーンが中断される。

```go
// ミドルウェアの実行順序を理解するための例
func middleware1() gin.HandlerFunc {
    return func(c *gin.Context) {
        fmt.Println("middleware1: before")
        c.Next()
        fmt.Println("middleware1: after")
    }
}

func middleware2() gin.HandlerFunc {
    return func(c *gin.Context) {
        fmt.Println("middleware2: before")
        c.Next()
        fmt.Println("middleware2: after")
    }
}

// 出力順序:
// middleware1: before
// middleware2: before
// handler
// middleware2: after
// middleware1: after
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

Echo でカスタムバリデータを使う場合、バリデーションエラーのフォーマットもカスタマイズできる。

```go
// 拡張カスタムバリデータ
type CustomValidator struct {
    validator *validator.Validate
}

func NewCustomValidator() *CustomValidator {
    v := validator.New()

    // JSON タグ名をフィールド名として使用
    v.RegisterTagNameFunc(func(fld reflect.StructField) string {
        name := strings.SplitN(fld.Tag.Get("json"), ",", 2)[0]
        if name == "-" {
            return ""
        }
        return name
    })

    return &CustomValidator{validator: v}
}

func (cv *CustomValidator) Validate(i interface{}) error {
    if err := cv.validator.Struct(i); err != nil {
        var ve validator.ValidationErrors
        if errors.As(err, &ve) {
            return &ValidationError{Errors: formatErrors(ve)}
        }
        return err
    }
    return nil
}

// カスタムエラー型
type ValidationError struct {
    Errors []FieldError `json:"errors"`
}

type FieldError struct {
    Field   string `json:"field"`
    Message string `json:"message"`
}

func (e *ValidationError) Error() string {
    return "validation failed"
}

func formatErrors(ve validator.ValidationErrors) []FieldError {
    errs := make([]FieldError, len(ve))
    for i, fe := range ve {
        errs[i] = FieldError{
            Field:   fe.Field(),
            Message: msgForTag(fe),
        }
    }
    return errs
}

func msgForTag(fe validator.FieldError) string {
    switch fe.Tag() {
    case "required":
        return "この項目は必須です"
    case "email":
        return "有効なメールアドレスを入力してください"
    default:
        return fe.Error()
    }
}
```

### コード例 6: Gin 統一レスポンス構造

本番APIでは統一されたレスポンス構造が重要である。

```go
// 統一レスポンス構造体
type Response struct {
    Code    int         `json:"code"`
    Message string      `json:"message"`
    Data    interface{} `json:"data,omitempty"`
    Meta    *Meta       `json:"meta,omitempty"`
}

type Meta struct {
    Page       int   `json:"page"`
    PerPage    int   `json:"per_page"`
    Total      int64 `json:"total"`
    TotalPages int   `json:"total_pages"`
}

type ErrorDetail struct {
    Code    string      `json:"code"`
    Message string      `json:"message"`
    Field   string      `json:"field,omitempty"`
    Details interface{} `json:"details,omitempty"`
}

type ErrorResponse struct {
    Code    int           `json:"code"`
    Message string        `json:"message"`
    Errors  []ErrorDetail `json:"errors,omitempty"`
}

// レスポンスヘルパー関数
func respondOK(c *gin.Context, data interface{}) {
    c.JSON(http.StatusOK, Response{
        Code:    http.StatusOK,
        Message: "success",
        Data:    data,
    })
}

func respondCreated(c *gin.Context, data interface{}) {
    c.JSON(http.StatusCreated, Response{
        Code:    http.StatusCreated,
        Message: "created",
        Data:    data,
    })
}

func respondPaginated(c *gin.Context, data interface{}, page, perPage int, total int64) {
    totalPages := int(total) / perPage
    if int(total)%perPage > 0 {
        totalPages++
    }
    c.JSON(http.StatusOK, Response{
        Code:    http.StatusOK,
        Message: "success",
        Data:    data,
        Meta: &Meta{
            Page:       page,
            PerPage:    perPage,
            Total:      total,
            TotalPages: totalPages,
        },
    })
}

func respondError(c *gin.Context, status int, code, message string) {
    c.JSON(status, ErrorResponse{
        Code:    status,
        Message: message,
        Errors:  []ErrorDetail{{Code: code, Message: message}},
    })
}

func respondValidationError(c *gin.Context, errors []ErrorDetail) {
    c.JSON(http.StatusBadRequest, ErrorResponse{
        Code:    http.StatusBadRequest,
        Message: "validation error",
        Errors:  errors,
    })
}

// ハンドラでの使用例
func listUsers(c *gin.Context) {
    page, _ := strconv.Atoi(c.DefaultQuery("page", "1"))
    perPage, _ := strconv.Atoi(c.DefaultQuery("per_page", "20"))

    users, total, err := userService.List(c.Request.Context(), page, perPage)
    if err != nil {
        respondError(c, http.StatusInternalServerError, "INTERNAL_ERROR", "ユーザー一覧の取得に失敗しました")
        return
    }
    respondPaginated(c, users, page, perPage, total)
}
```

### コード例 7: Echo ミドルウェア詳細

Echo では標準ミドルウェアが豊富に提供されている。

```go
func setupMiddlewares(e *echo.Echo) {
    // リカバリー
    e.Use(middleware.RecoverWithConfig(middleware.RecoverConfig{
        StackSize:         4 << 10, // 4 KB
        DisableStackAll:   false,
        DisablePrintStack: false,
        LogErrorFunc: func(c echo.Context, err error, stack []byte) error {
            log.Printf("PANIC: %v\n%s", err, stack)
            return nil
        },
    }))

    // CORS
    e.Use(middleware.CORSWithConfig(middleware.CORSConfig{
        AllowOrigins: []string{"https://example.com", "https://app.example.com"},
        AllowMethods: []string{http.MethodGet, http.MethodPost, http.MethodPut, http.MethodDelete},
        AllowHeaders: []string{
            echo.HeaderOrigin,
            echo.HeaderContentType,
            echo.HeaderAccept,
            echo.HeaderAuthorization,
        },
        AllowCredentials: true,
        MaxAge:           3600,
    }))

    // レート制限
    e.Use(middleware.RateLimiterWithConfig(middleware.RateLimiterConfig{
        Skipper: middleware.DefaultSkipper,
        Store: middleware.NewRateLimiterMemoryStoreWithConfig(
            middleware.RateLimiterMemoryStoreConfig{
                Rate:      10,              // 10 requests
                Burst:     30,              // burst of 30
                ExpiresIn: 3 * time.Minute, // TTL
            },
        ),
        IdentifierExtractor: func(ctx echo.Context) (string, error) {
            id := ctx.RealIP()
            return id, nil
        },
        ErrorHandler: func(ctx echo.Context, err error) error {
            return ctx.JSON(http.StatusForbidden, map[string]string{
                "error": "rate limit exceeded",
            })
        },
        DenyHandler: func(ctx echo.Context, identifier string, err error) error {
            return ctx.JSON(http.StatusTooManyRequests, map[string]string{
                "error": "too many requests",
            })
        },
    }))

    // リクエストID
    e.Use(middleware.RequestID())

    // タイムアウト
    e.Use(middleware.TimeoutWithConfig(middleware.TimeoutConfig{
        Timeout: 30 * time.Second,
    }))

    // Gzip圧縮
    e.Use(middleware.GzipWithConfig(middleware.GzipConfig{
        Level: 5,
        Skipper: func(c echo.Context) bool {
            return strings.Contains(c.Path(), "ws")
        },
    }))

    // セキュリティヘッダ
    e.Use(middleware.SecureWithConfig(middleware.SecureConfig{
        XSSProtection:         "1; mode=block",
        ContentTypeNosniff:    "nosniff",
        XFrameOptions:         "DENY",
        HSTSMaxAge:            31536000,
        ContentSecurityPolicy: "default-src 'self'",
    }))
}
```

### コード例 8: Gin カスタムミドルウェア集

本番環境でよく使うカスタムミドルウェアをまとめる。

```go
// リクエストIDミドルウェア
func RequestIDMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        requestID := c.GetHeader("X-Request-ID")
        if requestID == "" {
            requestID = uuid.New().String()
        }
        c.Set("requestID", requestID)
        c.Header("X-Request-ID", requestID)
        c.Next()
    }
}

// 構造化ログミドルウェア
func StructuredLoggerMiddleware(logger *slog.Logger) gin.HandlerFunc {
    return func(c *gin.Context) {
        start := time.Now()
        path := c.Request.URL.Path
        raw := c.Request.URL.RawQuery

        c.Next()

        latency := time.Since(start)
        status := c.Writer.Status()
        clientIP := c.ClientIP()
        method := c.Request.Method
        requestID, _ := c.Get("requestID")

        attrs := []slog.Attr{
            slog.String("request_id", fmt.Sprintf("%v", requestID)),
            slog.String("method", method),
            slog.String("path", path),
            slog.String("query", raw),
            slog.Int("status", status),
            slog.Duration("latency", latency),
            slog.String("client_ip", clientIP),
            slog.Int("body_size", c.Writer.Size()),
        }

        if status >= 500 {
            logger.LogAttrs(c.Request.Context(), slog.LevelError, "Server error", attrs...)
        } else if status >= 400 {
            logger.LogAttrs(c.Request.Context(), slog.LevelWarn, "Client error", attrs...)
        } else {
            logger.LogAttrs(c.Request.Context(), slog.LevelInfo, "Request", attrs...)
        }
    }
}

// CORSミドルウェア
func CORSMiddleware(allowOrigins []string) gin.HandlerFunc {
    originMap := make(map[string]bool)
    for _, o := range allowOrigins {
        originMap[o] = true
    }

    return func(c *gin.Context) {
        origin := c.Request.Header.Get("Origin")
        if originMap[origin] {
            c.Header("Access-Control-Allow-Origin", origin)
            c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS")
            c.Header("Access-Control-Allow-Headers", "Origin, Content-Type, Accept, Authorization, X-Request-ID")
            c.Header("Access-Control-Allow-Credentials", "true")
            c.Header("Access-Control-Max-Age", "3600")
        }

        if c.Request.Method == "OPTIONS" {
            c.AbortWithStatus(http.StatusNoContent)
            return
        }

        c.Next()
    }
}

// レート制限ミドルウェア (Token Bucket)
func RateLimitMiddleware(rps int, burst int) gin.HandlerFunc {
    var mu sync.Mutex
    limiters := make(map[string]*rate.Limiter)

    getLimiter := func(key string) *rate.Limiter {
        mu.Lock()
        defer mu.Unlock()
        if limiter, exists := limiters[key]; exists {
            return limiter
        }
        limiter := rate.NewLimiter(rate.Limit(rps), burst)
        limiters[key] = limiter
        return limiter
    }

    // 古いエントリの定期クリーンアップ
    go func() {
        for range time.Tick(10 * time.Minute) {
            mu.Lock()
            limiters = make(map[string]*rate.Limiter)
            mu.Unlock()
        }
    }()

    return func(c *gin.Context) {
        limiter := getLimiter(c.ClientIP())
        if !limiter.Allow() {
            c.AbortWithStatusJSON(http.StatusTooManyRequests, gin.H{
                "code":    "RATE_LIMIT_EXCEEDED",
                "message": "リクエスト数が制限を超えました",
            })
            return
        }
        c.Next()
    }
}

// タイムアウトミドルウェア
func TimeoutMiddleware(timeout time.Duration) gin.HandlerFunc {
    return func(c *gin.Context) {
        ctx, cancel := context.WithTimeout(c.Request.Context(), timeout)
        defer cancel()

        c.Request = c.Request.WithContext(ctx)

        done := make(chan struct{})
        go func() {
            c.Next()
            close(done)
        }()

        select {
        case <-done:
            return
        case <-ctx.Done():
            c.AbortWithStatusJSON(http.StatusGatewayTimeout, gin.H{
                "code":    "TIMEOUT",
                "message": "リクエストがタイムアウトしました",
            })
        }
    }
}

// セキュリティヘッダミドルウェア
func SecurityHeadersMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        c.Header("X-Content-Type-Options", "nosniff")
        c.Header("X-Frame-Options", "DENY")
        c.Header("X-XSS-Protection", "1; mode=block")
        c.Header("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        c.Header("Content-Security-Policy", "default-src 'self'")
        c.Header("Referrer-Policy", "strict-origin-when-cross-origin")
        c.Header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        c.Next()
    }
}
```

### コード例 9: Gin JWT認証の完全実装

JWT認証は多くのAPIで必要となる。

```go
import (
    "github.com/golang-jwt/jwt/v5"
)

// JWT設定
type JWTConfig struct {
    SecretKey     []byte
    Issuer        string
    AccessExpiry  time.Duration
    RefreshExpiry time.Duration
}

// カスタムクレーム
type Claims struct {
    UserID int64  `json:"user_id"`
    Email  string `json:"email"`
    Role   string `json:"role"`
    jwt.RegisteredClaims
}

// トークンペア
type TokenPair struct {
    AccessToken  string `json:"access_token"`
    RefreshToken string `json:"refresh_token"`
    ExpiresAt    int64  `json:"expires_at"`
}

// トークン生成
func (cfg *JWTConfig) GenerateTokenPair(userID int64, email, role string) (*TokenPair, error) {
    now := time.Now()

    // アクセストークン
    accessClaims := Claims{
        UserID: userID,
        Email:  email,
        Role:   role,
        RegisteredClaims: jwt.RegisteredClaims{
            Issuer:    cfg.Issuer,
            Subject:   strconv.FormatInt(userID, 10),
            ExpiresAt: jwt.NewNumericDate(now.Add(cfg.AccessExpiry)),
            IssuedAt:  jwt.NewNumericDate(now),
            NotBefore: jwt.NewNumericDate(now),
            ID:        uuid.New().String(),
        },
    }
    accessToken := jwt.NewWithClaims(jwt.SigningMethodHS256, accessClaims)
    accessTokenStr, err := accessToken.SignedString(cfg.SecretKey)
    if err != nil {
        return nil, fmt.Errorf("access token signing: %w", err)
    }

    // リフレッシュトークン
    refreshClaims := jwt.RegisteredClaims{
        Issuer:    cfg.Issuer,
        Subject:   strconv.FormatInt(userID, 10),
        ExpiresAt: jwt.NewNumericDate(now.Add(cfg.RefreshExpiry)),
        IssuedAt:  jwt.NewNumericDate(now),
        ID:        uuid.New().String(),
    }
    refreshToken := jwt.NewWithClaims(jwt.SigningMethodHS256, refreshClaims)
    refreshTokenStr, err := refreshToken.SignedString(cfg.SecretKey)
    if err != nil {
        return nil, fmt.Errorf("refresh token signing: %w", err)
    }

    return &TokenPair{
        AccessToken:  accessTokenStr,
        RefreshToken: refreshTokenStr,
        ExpiresAt:    accessClaims.ExpiresAt.Unix(),
    }, nil
}

// トークン検証
func (cfg *JWTConfig) ValidateToken(tokenStr string) (*Claims, error) {
    token, err := jwt.ParseWithClaims(tokenStr, &Claims{}, func(token *jwt.Token) (interface{}, error) {
        if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
            return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
        }
        return cfg.SecretKey, nil
    })
    if err != nil {
        return nil, fmt.Errorf("token parse: %w", err)
    }

    claims, ok := token.Claims.(*Claims)
    if !ok || !token.Valid {
        return nil, fmt.Errorf("invalid token claims")
    }
    return claims, nil
}

// JWT認証ミドルウェア
func JWTAuthMiddleware(jwtCfg *JWTConfig) gin.HandlerFunc {
    return func(c *gin.Context) {
        authHeader := c.GetHeader("Authorization")
        if authHeader == "" {
            c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
                "code":    "UNAUTHORIZED",
                "message": "Authorization ヘッダが必要です",
            })
            return
        }

        // "Bearer <token>" 形式の検証
        parts := strings.SplitN(authHeader, " ", 2)
        if len(parts) != 2 || !strings.EqualFold(parts[0], "bearer") {
            c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
                "code":    "INVALID_TOKEN_FORMAT",
                "message": "Bearer トークン形式が無効です",
            })
            return
        }

        claims, err := jwtCfg.ValidateToken(parts[1])
        if err != nil {
            c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
                "code":    "INVALID_TOKEN",
                "message": "トークンが無効または期限切れです",
            })
            return
        }

        // コンテキストにユーザー情報をセット
        c.Set("userID", claims.UserID)
        c.Set("email", claims.Email)
        c.Set("role", claims.Role)
        c.Set("claims", claims)
        c.Next()
    }
}

// ロールベース認可ミドルウェア
func RequireRole(roles ...string) gin.HandlerFunc {
    roleSet := make(map[string]bool)
    for _, r := range roles {
        roleSet[r] = true
    }

    return func(c *gin.Context) {
        role, exists := c.Get("role")
        if !exists {
            c.AbortWithStatusJSON(http.StatusForbidden, gin.H{
                "code":    "FORBIDDEN",
                "message": "権限がありません",
            })
            return
        }

        if !roleSet[role.(string)] {
            c.AbortWithStatusJSON(http.StatusForbidden, gin.H{
                "code":    "INSUFFICIENT_ROLE",
                "message": fmt.Sprintf("必要なロール: %v", roles),
            })
            return
        }
        c.Next()
    }
}

// ログインハンドラ
func loginHandler(jwtCfg *JWTConfig, userService UserService) gin.HandlerFunc {
    return func(c *gin.Context) {
        var req LoginRequest
        if err := c.ShouldBindJSON(&req); err != nil {
            respondError(c, http.StatusBadRequest, "VALIDATION_ERROR", err.Error())
            return
        }

        user, err := userService.Authenticate(c.Request.Context(), req.Email, req.Password)
        if err != nil {
            respondError(c, http.StatusUnauthorized, "INVALID_CREDENTIALS", "メールアドレスまたはパスワードが間違っています")
            return
        }

        tokens, err := jwtCfg.GenerateTokenPair(user.ID, user.Email, user.Role)
        if err != nil {
            respondError(c, http.StatusInternalServerError, "TOKEN_GENERATION_ERROR", "トークンの生成に失敗しました")
            return
        }

        c.JSON(http.StatusOK, gin.H{
            "code":    200,
            "message": "success",
            "data":    tokens,
        })
    }
}
```

### コード例 10: Echo グループとカスタムコンテキスト

Echo ではカスタムコンテキストでハンドラの共通機能を拡張できる。

```go
// カスタムコンテキスト
type AppContext struct {
    echo.Context
    UserID int64
    Role   string
}

// カスタムコンテキストミドルウェア
func CustomContextMiddleware(next echo.HandlerFunc) echo.HandlerFunc {
    return func(c echo.Context) error {
        cc := &AppContext{Context: c}
        return next(cc)
    }
}

// カスタムコンテキストを使ったハンドラ
func getProfile(c echo.Context) error {
    cc := c.(*AppContext)
    user, err := userService.FindByID(cc.Request().Context(), cc.UserID)
    if err != nil {
        return echo.NewHTTPError(http.StatusNotFound, "user not found")
    }
    return cc.JSON(http.StatusOK, user)
}

// Echo グループの構成
func setupRoutes(e *echo.Echo, jwtCfg *JWTConfig) {
    // APIバージョニング
    v1 := e.Group("/api/v1")
    v1.Use(CustomContextMiddleware)

    // 公開エンドポイント
    public := v1.Group("")
    {
        public.POST("/auth/login", loginHandler)
        public.POST("/auth/register", registerHandler)
        public.POST("/auth/refresh", refreshTokenHandler)
    }

    // 認証必須エンドポイント
    auth := v1.Group("")
    auth.Use(echoJWTMiddleware(jwtCfg))
    {
        auth.GET("/profile", getProfile)
        auth.PUT("/profile", updateProfile)
        auth.GET("/users/:id", getUserByID)
    }

    // 管理者エンドポイント
    admin := v1.Group("/admin")
    admin.Use(echoJWTMiddleware(jwtCfg), echoRequireRole("admin"))
    {
        admin.GET("/users", adminListUsers)
        admin.DELETE("/users/:id", adminDeleteUser)
        admin.GET("/stats", getStats)
    }
}
```

### コード例 11: Swagger / OpenAPI 統合

Gin でSwagger APIドキュメントを自動生成する。

```go
// Swaggerアノテーション付きハンドラ

// @Summary ユーザー一覧取得
// @Description ページネーション付きのユーザー一覧を取得する
// @Tags users
// @Accept json
// @Produce json
// @Param page query int false "ページ番号" default(1)
// @Param per_page query int false "1ページあたりの件数" default(20) maximum(100)
// @Param sort query string false "ソート項目" Enums(name, email, created_at)
// @Param order query string false "ソート順序" Enums(asc, desc) default(asc)
// @Success 200 {object} Response{data=[]User,meta=Meta} "成功"
// @Failure 400 {object} ErrorResponse "バリデーションエラー"
// @Failure 401 {object} ErrorResponse "認証エラー"
// @Failure 500 {object} ErrorResponse "サーバーエラー"
// @Security BearerAuth
// @Router /api/v1/users [get]
func listUsers(c *gin.Context) {
    // 実装
}

// @Summary ユーザー作成
// @Description 新しいユーザーを作成する
// @Tags users
// @Accept json
// @Produce json
// @Param request body CreateUserRequest true "ユーザー作成リクエスト"
// @Success 201 {object} Response{data=User} "作成成功"
// @Failure 400 {object} ErrorResponse "バリデーションエラー"
// @Failure 409 {object} ErrorResponse "メールアドレス重複"
// @Failure 500 {object} ErrorResponse "サーバーエラー"
// @Security BearerAuth
// @Router /api/v1/users [post]
func createUser(c *gin.Context) {
    // 実装
}

// @Summary ユーザー取得
// @Description IDでユーザーを取得する
// @Tags users
// @Accept json
// @Produce json
// @Param id path int true "ユーザーID"
// @Success 200 {object} Response{data=User} "成功"
// @Failure 404 {object} ErrorResponse "ユーザーが見つかりません"
// @Failure 500 {object} ErrorResponse "サーバーエラー"
// @Security BearerAuth
// @Router /api/v1/users/{id} [get]
func getUser(c *gin.Context) {
    // 実装
}

// Swaggerの設定
// @title My API
// @version 1.0
// @description ユーザー管理API
// @host localhost:8080
// @BasePath /api/v1
// @securityDefinitions.apikey BearerAuth
// @in header
// @name Authorization

func main() {
    r := gin.Default()

    // Swaggerエンドポイント
    r.GET("/swagger/*any", ginSwagger.WrapHandler(swaggerFiles.Handler))

    // API ルーティング
    v1 := r.Group("/api/v1")
    {
        v1.GET("/users", listUsers)
        v1.POST("/users", createUser)
        v1.GET("/users/:id", getUser)
    }

    r.Run(":8080")
}
```

Swagger ドキュメントの生成コマンド。

```bash
# swag のインストール
go install github.com/swaggo/swag/cmd/swag@latest

# ドキュメント生成
swag init -g cmd/api/main.go -o docs

# 生成されるファイル:
# docs/docs.go
# docs/swagger.json
# docs/swagger.yaml
```

### コード例 12: Gin テスト

ハンドラのテストは httptest を使用する。

```go
import (
    "net/http"
    "net/http/httptest"
    "testing"

    "github.com/gin-gonic/gin"
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
)

// テスト用のGinエンジンを作成するヘルパー
func setupTestRouter() *gin.Engine {
    gin.SetMode(gin.TestMode)
    r := gin.New()
    return r
}

// ハンドラのユニットテスト
func TestGetUser(t *testing.T) {
    r := setupTestRouter()

    mockService := &MockUserService{
        users: map[int64]*User{
            1: {ID: 1, Name: "Tanaka", Email: "tanaka@example.com"},
        },
    }

    r.GET("/users/:id", getUserHandler(mockService))

    tests := []struct {
        name       string
        userID     string
        wantStatus int
        wantBody   string
    }{
        {
            name:       "正常系: ユーザー取得",
            userID:     "1",
            wantStatus: http.StatusOK,
            wantBody:   `"name":"Tanaka"`,
        },
        {
            name:       "異常系: ユーザーが存在しない",
            userID:     "999",
            wantStatus: http.StatusNotFound,
            wantBody:   `"code":"NOT_FOUND"`,
        },
        {
            name:       "異常系: 無効なID",
            userID:     "abc",
            wantStatus: http.StatusBadRequest,
            wantBody:   `"code":"INVALID_ID"`,
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            req := httptest.NewRequest(http.MethodGet, "/users/"+tt.userID, nil)
            w := httptest.NewRecorder()

            r.ServeHTTP(w, req)

            assert.Equal(t, tt.wantStatus, w.Code)
            assert.Contains(t, w.Body.String(), tt.wantBody)
        })
    }
}

// POST ハンドラのテスト
func TestCreateUser(t *testing.T) {
    r := setupTestRouter()
    mockService := &MockUserService{users: make(map[int64]*User)}
    r.POST("/users", createUserHandler(mockService))

    tests := []struct {
        name       string
        body       string
        wantStatus int
    }{
        {
            name:       "正常系: ユーザー作成",
            body:       `{"name":"Yamada","email":"yamada@example.com","password":"P@ssw0rd!"}`,
            wantStatus: http.StatusCreated,
        },
        {
            name:       "異常系: 名前が空",
            body:       `{"name":"","email":"yamada@example.com","password":"P@ssw0rd!"}`,
            wantStatus: http.StatusBadRequest,
        },
        {
            name:       "異常系: 無効なメール",
            body:       `{"name":"Yamada","email":"invalid","password":"P@ssw0rd!"}`,
            wantStatus: http.StatusBadRequest,
        },
        {
            name:       "異常系: JSONパースエラー",
            body:       `{invalid json`,
            wantStatus: http.StatusBadRequest,
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            req := httptest.NewRequest(http.MethodPost, "/users",
                strings.NewReader(tt.body))
            req.Header.Set("Content-Type", "application/json")
            w := httptest.NewRecorder()

            r.ServeHTTP(w, req)

            assert.Equal(t, tt.wantStatus, w.Code)
        })
    }
}

// ミドルウェアのテスト
func TestAuthMiddleware(t *testing.T) {
    jwtCfg := &JWTConfig{
        SecretKey:    []byte("test-secret"),
        Issuer:       "test",
        AccessExpiry: time.Hour,
    }

    r := setupTestRouter()
    r.Use(JWTAuthMiddleware(jwtCfg))
    r.GET("/protected", func(c *gin.Context) {
        userID := c.GetInt64("userID")
        c.JSON(http.StatusOK, gin.H{"user_id": userID})
    })

    t.Run("トークンなし", func(t *testing.T) {
        req := httptest.NewRequest(http.MethodGet, "/protected", nil)
        w := httptest.NewRecorder()
        r.ServeHTTP(w, req)
        assert.Equal(t, http.StatusUnauthorized, w.Code)
    })

    t.Run("有効なトークン", func(t *testing.T) {
        tokens, err := jwtCfg.GenerateTokenPair(42, "test@example.com", "user")
        require.NoError(t, err)

        req := httptest.NewRequest(http.MethodGet, "/protected", nil)
        req.Header.Set("Authorization", "Bearer "+tokens.AccessToken)
        w := httptest.NewRecorder()
        r.ServeHTTP(w, req)

        assert.Equal(t, http.StatusOK, w.Code)
        assert.Contains(t, w.Body.String(), `"user_id":42`)
    })

    t.Run("無効なトークン", func(t *testing.T) {
        req := httptest.NewRequest(http.MethodGet, "/protected", nil)
        req.Header.Set("Authorization", "Bearer invalid-token")
        w := httptest.NewRecorder()
        r.ServeHTTP(w, req)
        assert.Equal(t, http.StatusUnauthorized, w.Code)
    })
}
```

### コード例 13: Echo テスト

```go
func TestEchoGetUser(t *testing.T) {
    e := echo.New()
    e.Validator = NewCustomValidator()

    mockService := &MockUserService{
        users: map[int64]*User{
            1: {ID: 1, Name: "Tanaka", Email: "tanaka@example.com"},
        },
    }

    tests := []struct {
        name       string
        userID     string
        wantStatus int
        wantBody   string
    }{
        {
            name:       "正常系",
            userID:     "1",
            wantStatus: http.StatusOK,
            wantBody:   `"name":"Tanaka"`,
        },
        {
            name:       "ユーザー未存在",
            userID:     "999",
            wantStatus: http.StatusNotFound,
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            req := httptest.NewRequest(http.MethodGet, "/", nil)
            rec := httptest.NewRecorder()
            c := e.NewContext(req, rec)
            c.SetPath("/users/:id")
            c.SetParamNames("id")
            c.SetParamValues(tt.userID)

            handler := getUserHandler(mockService)
            err := handler(c)

            if tt.wantStatus >= 400 {
                he, ok := err.(*echo.HTTPError)
                assert.True(t, ok)
                assert.Equal(t, tt.wantStatus, he.Code)
            } else {
                assert.NoError(t, err)
                assert.Equal(t, tt.wantStatus, rec.Code)
                if tt.wantBody != "" {
                    assert.Contains(t, rec.Body.String(), tt.wantBody)
                }
            }
        })
    }
}

// Echo ミドルウェアのテスト
func TestEchoRateLimitMiddleware(t *testing.T) {
    e := echo.New()
    e.Use(middleware.RateLimiterWithConfig(middleware.RateLimiterConfig{
        Store: middleware.NewRateLimiterMemoryStoreWithConfig(
            middleware.RateLimiterMemoryStoreConfig{
                Rate:  2,
                Burst: 2,
            },
        ),
        IdentifierExtractor: func(ctx echo.Context) (string, error) {
            return ctx.RealIP(), nil
        },
        DenyHandler: func(ctx echo.Context, identifier string, err error) error {
            return ctx.JSON(http.StatusTooManyRequests, map[string]string{
                "error": "rate limited",
            })
        },
    }))
    e.GET("/test", func(c echo.Context) error {
        return c.String(http.StatusOK, "ok")
    })

    // 最初の2リクエストは成功
    for i := 0; i < 2; i++ {
        req := httptest.NewRequest(http.MethodGet, "/test", nil)
        rec := httptest.NewRecorder()
        e.ServeHTTP(rec, req)
        assert.Equal(t, http.StatusOK, rec.Code)
    }

    // 3番目のリクエストはレート制限
    req := httptest.NewRequest(http.MethodGet, "/test", nil)
    rec := httptest.NewRecorder()
    e.ServeHTTP(rec, req)
    assert.Equal(t, http.StatusTooManyRequests, rec.Code)
}
```

### コード例 14: Gin Graceful Shutdown

```go
func main() {
    // Releaseモード設定
    gin.SetMode(gin.ReleaseMode)

    r := gin.New()

    // ミドルウェアの設定
    r.Use(RequestIDMiddleware())
    r.Use(StructuredLoggerMiddleware(slog.Default()))
    r.Use(gin.Recovery())
    r.Use(CORSMiddleware([]string{"https://example.com"}))
    r.Use(SecurityHeadersMiddleware())

    // ヘルスチェック
    r.GET("/health", func(c *gin.Context) {
        c.JSON(http.StatusOK, gin.H{
            "status": "ok",
            "time":   time.Now().UTC().Format(time.RFC3339),
        })
    })

    // Readiness チェック (DBなど外部依存の状態も確認)
    r.GET("/ready", func(c *gin.Context) {
        if err := db.PingContext(c.Request.Context()); err != nil {
            c.JSON(http.StatusServiceUnavailable, gin.H{
                "status": "not ready",
                "error":  err.Error(),
            })
            return
        }
        c.JSON(http.StatusOK, gin.H{"status": "ready"})
    })

    // API ルーティング
    setupRoutes(r)

    // サーバー設定
    srv := &http.Server{
        Addr:         ":8080",
        Handler:      r,
        ReadTimeout:  15 * time.Second,
        WriteTimeout: 30 * time.Second,
        IdleTimeout:  60 * time.Second,
        // ヘッダサイズの制限
        MaxHeaderBytes: 1 << 20, // 1 MB
    }

    // Graceful Shutdown
    go func() {
        slog.Info("Server starting", "addr", srv.Addr)
        if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
            slog.Error("Server error", "error", err)
            os.Exit(1)
        }
    }()

    quit := make(chan os.Signal, 1)
    signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
    sig := <-quit
    slog.Info("Shutdown signal received", "signal", sig)

    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()

    if err := srv.Shutdown(ctx); err != nil {
        slog.Error("Server forced to shutdown", "error", err)
        os.Exit(1)
    }

    slog.Info("Server exited properly")
}
```

### コード例 15: Echo Graceful Shutdown

```go
func main() {
    e := echo.New()
    e.HideBanner = true

    // ミドルウェアの設定
    setupMiddlewares(e)

    // ルーティングの設定
    setupRoutes(e, jwtCfg)

    // ヘルスチェック
    e.GET("/health", func(c echo.Context) error {
        return c.JSON(http.StatusOK, map[string]string{
            "status": "ok",
            "time":   time.Now().UTC().Format(time.RFC3339),
        })
    })

    // カスタムHTTPエラーハンドラ
    e.HTTPErrorHandler = func(err error, c echo.Context) {
        if c.Response().Committed {
            return
        }

        var he *echo.HTTPError
        if errors.As(err, &he) {
            msg := he.Message
            if m, ok := msg.(string); ok {
                c.JSON(he.Code, map[string]interface{}{
                    "code":    he.Code,
                    "message": m,
                })
            } else {
                c.JSON(he.Code, map[string]interface{}{
                    "code":    he.Code,
                    "message": msg,
                })
            }
            return
        }

        var ve *ValidationError
        if errors.As(err, &ve) {
            c.JSON(http.StatusBadRequest, map[string]interface{}{
                "code":    http.StatusBadRequest,
                "message": "validation error",
                "errors":  ve.Errors,
            })
            return
        }

        // 未知のエラー
        slog.Error("Unhandled error", "error", err, "path", c.Path())
        c.JSON(http.StatusInternalServerError, map[string]interface{}{
            "code":    http.StatusInternalServerError,
            "message": "internal server error",
        })
    }

    // Graceful Shutdown
    go func() {
        if err := e.Start(":8080"); err != nil && err != http.ErrServerClosed {
            e.Logger.Fatal("shutting down the server")
        }
    }()

    quit := make(chan os.Signal, 1)
    signal.Notify(quit, os.Interrupt, syscall.SIGTERM)
    <-quit

    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()

    if err := e.Shutdown(ctx); err != nil {
        e.Logger.Fatal(err)
    }
    slog.Info("Server exited")
}
```

### コード例 16: ファイルアップロード

Gin と Echo でのファイルアップロード処理。

```go
// Gin: シングルファイルアップロード
func uploadFile(c *gin.Context) {
    file, err := c.FormFile("file")
    if err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": "ファイルが必要です"})
        return
    }

    // バリデーション
    if file.Size > 10<<20 { // 10MB制限
        c.JSON(http.StatusBadRequest, gin.H{"error": "ファイルサイズが大きすぎます (最大10MB)"})
        return
    }

    // MIMEタイプチェック
    allowedTypes := map[string]bool{
        "image/jpeg": true,
        "image/png":  true,
        "image/gif":  true,
        "image/webp": true,
    }

    src, err := file.Open()
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": "ファイルの読み取りに失敗"})
        return
    }
    defer src.Close()

    // 最初の512バイトでMIMEタイプを判定
    buffer := make([]byte, 512)
    _, err = src.Read(buffer)
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": "ファイルの読み取りに失敗"})
        return
    }
    contentType := http.DetectContentType(buffer)
    if !allowedTypes[contentType] {
        c.JSON(http.StatusBadRequest, gin.H{
            "error": fmt.Sprintf("許可されていないファイルタイプ: %s", contentType),
        })
        return
    }

    // ファイル保存 (ユニーク名生成)
    ext := filepath.Ext(file.Filename)
    filename := fmt.Sprintf("%s%s", uuid.New().String(), ext)
    dst := filepath.Join("uploads", filename)

    if err := c.SaveUploadedFile(file, dst); err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": "ファイルの保存に失敗"})
        return
    }

    c.JSON(http.StatusOK, gin.H{
        "filename": filename,
        "size":     file.Size,
        "type":     contentType,
        "url":      fmt.Sprintf("/uploads/%s", filename),
    })
}

// Gin: 複数ファイルアップロード
func uploadMultipleFiles(c *gin.Context) {
    form, err := c.MultipartForm()
    if err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
        return
    }

    files := form.File["files"]
    if len(files) == 0 {
        c.JSON(http.StatusBadRequest, gin.H{"error": "ファイルが必要です"})
        return
    }
    if len(files) > 10 {
        c.JSON(http.StatusBadRequest, gin.H{"error": "最大10ファイルまでです"})
        return
    }

    var uploaded []gin.H
    for _, file := range files {
        ext := filepath.Ext(file.Filename)
        filename := fmt.Sprintf("%s%s", uuid.New().String(), ext)
        dst := filepath.Join("uploads", filename)

        if err := c.SaveUploadedFile(file, dst); err != nil {
            continue
        }
        uploaded = append(uploaded, gin.H{
            "filename": filename,
            "size":     file.Size,
        })
    }

    c.JSON(http.StatusOK, gin.H{
        "uploaded": uploaded,
        "count":    len(uploaded),
    })
}

// Echo: ファイルアップロード
func echoUploadFile(c echo.Context) error {
    file, err := c.FormFile("file")
    if err != nil {
        return echo.NewHTTPError(http.StatusBadRequest, "ファイルが必要です")
    }

    if file.Size > 10<<20 {
        return echo.NewHTTPError(http.StatusBadRequest, "ファイルサイズが大きすぎます")
    }

    src, err := file.Open()
    if err != nil {
        return echo.NewHTTPError(http.StatusInternalServerError, "ファイルの読み取りに失敗")
    }
    defer src.Close()

    ext := filepath.Ext(file.Filename)
    filename := fmt.Sprintf("%s%s", uuid.New().String(), ext)
    dst, err := os.Create(filepath.Join("uploads", filename))
    if err != nil {
        return echo.NewHTTPError(http.StatusInternalServerError, "ファイルの保存に失敗")
    }
    defer dst.Close()

    if _, err = io.Copy(dst, src); err != nil {
        return echo.NewHTTPError(http.StatusInternalServerError, "ファイルのコピーに失敗")
    }

    return c.JSON(http.StatusOK, map[string]interface{}{
        "filename": filename,
        "size":     file.Size,
    })
}
```

### コード例 17: WebSocket

Gin と Echo での WebSocket 実装。

```go
import "github.com/gorilla/websocket"

// Gin WebSocket
var upgrader = websocket.Upgrader{
    ReadBufferSize:  1024,
    WriteBufferSize: 1024,
    CheckOrigin: func(r *http.Request) bool {
        origin := r.Header.Get("Origin")
        return origin == "https://example.com"
    },
}

// WebSocket ハブ (接続管理)
type Hub struct {
    clients    map[*Client]bool
    broadcast  chan []byte
    register   chan *Client
    unregister chan *Client
    mu         sync.RWMutex
}

type Client struct {
    hub  *Hub
    conn *websocket.Conn
    send chan []byte
}

func newHub() *Hub {
    return &Hub{
        broadcast:  make(chan []byte),
        register:   make(chan *Client),
        unregister: make(chan *Client),
        clients:    make(map[*Client]bool),
    }
}

func (h *Hub) run() {
    for {
        select {
        case client := <-h.register:
            h.mu.Lock()
            h.clients[client] = true
            h.mu.Unlock()
        case client := <-h.unregister:
            h.mu.Lock()
            if _, ok := h.clients[client]; ok {
                delete(h.clients, client)
                close(client.send)
            }
            h.mu.Unlock()
        case message := <-h.broadcast:
            h.mu.RLock()
            for client := range h.clients {
                select {
                case client.send <- message:
                default:
                    close(client.send)
                    delete(h.clients, client)
                }
            }
            h.mu.RUnlock()
        }
    }
}

// Gin WebSocket ハンドラ
func wsHandler(hub *Hub) gin.HandlerFunc {
    return func(c *gin.Context) {
        conn, err := upgrader.Upgrade(c.Writer, c.Request, nil)
        if err != nil {
            return
        }

        client := &Client{
            hub:  hub,
            conn: conn,
            send: make(chan []byte, 256),
        }
        hub.register <- client

        go client.writePump()
        go client.readPump()
    }
}

func (c *Client) readPump() {
    defer func() {
        c.hub.unregister <- c
        c.conn.Close()
    }()

    c.conn.SetReadLimit(512)
    c.conn.SetReadDeadline(time.Now().Add(60 * time.Second))
    c.conn.SetPongHandler(func(string) error {
        c.conn.SetReadDeadline(time.Now().Add(60 * time.Second))
        return nil
    })

    for {
        _, message, err := c.conn.ReadMessage()
        if err != nil {
            break
        }
        c.hub.broadcast <- message
    }
}

func (c *Client) writePump() {
    ticker := time.NewTicker(54 * time.Second)
    defer func() {
        ticker.Stop()
        c.conn.Close()
    }()

    for {
        select {
        case message, ok := <-c.send:
            c.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
            if !ok {
                c.conn.WriteMessage(websocket.CloseMessage, []byte{})
                return
            }
            if err := c.conn.WriteMessage(websocket.TextMessage, message); err != nil {
                return
            }
        case <-ticker.C:
            c.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
            if err := c.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
                return
            }
        }
    }
}
```

### コード例 18: 依存性注入パターン (Clean Architecture)

実際のプロジェクトではクリーンアーキテクチャが重要である。

```go
// ドメイン層 (domain/user.go)
type User struct {
    ID        int64     `json:"id"`
    Name      string    `json:"name"`
    Email     string    `json:"email"`
    CreatedAt time.Time `json:"created_at"`
}

type UserRepository interface {
    FindByID(ctx context.Context, id int64) (*User, error)
    FindAll(ctx context.Context, page, perPage int) ([]User, int64, error)
    Create(ctx context.Context, user *User) error
    Update(ctx context.Context, user *User) error
    Delete(ctx context.Context, id int64) error
}

type UserService interface {
    GetUser(ctx context.Context, id int64) (*User, error)
    ListUsers(ctx context.Context, page, perPage int) ([]User, int64, error)
    CreateUser(ctx context.Context, req *CreateUserRequest) (*User, error)
    UpdateUser(ctx context.Context, id int64, req *UpdateUserRequest) (*User, error)
    DeleteUser(ctx context.Context, id int64) error
}

// ユースケース層 (usecase/user_service.go)
type userServiceImpl struct {
    repo   UserRepository
    logger *slog.Logger
}

func NewUserService(repo UserRepository, logger *slog.Logger) UserService {
    return &userServiceImpl{repo: repo, logger: logger}
}

func (s *userServiceImpl) GetUser(ctx context.Context, id int64) (*User, error) {
    user, err := s.repo.FindByID(ctx, id)
    if err != nil {
        s.logger.ErrorContext(ctx, "Failed to get user", "id", id, "error", err)
        return nil, fmt.Errorf("get user: %w", err)
    }
    return user, nil
}

func (s *userServiceImpl) CreateUser(ctx context.Context, req *CreateUserRequest) (*User, error) {
    user := &User{
        Name:      req.Name,
        Email:     req.Email,
        CreatedAt: time.Now(),
    }
    if err := s.repo.Create(ctx, user); err != nil {
        s.logger.ErrorContext(ctx, "Failed to create user", "error", err)
        return nil, fmt.Errorf("create user: %w", err)
    }
    s.logger.InfoContext(ctx, "User created", "id", user.ID, "email", user.Email)
    return user, nil
}

// インフラ層 (infrastructure/user_repository.go)
type postgresUserRepo struct {
    db *sqlx.DB
}

func NewPostgresUserRepo(db *sqlx.DB) UserRepository {
    return &postgresUserRepo{db: db}
}

func (r *postgresUserRepo) FindByID(ctx context.Context, id int64) (*User, error) {
    var user User
    err := r.db.GetContext(ctx, &user, "SELECT id, name, email, created_at FROM users WHERE id = $1", id)
    if errors.Is(err, sql.ErrNoRows) {
        return nil, ErrNotFound
    }
    return &user, err
}

// プレゼンテーション層 (handler/user_handler.go)
type UserHandler struct {
    service UserService
}

func NewUserHandler(service UserService) *UserHandler {
    return &UserHandler{service: service}
}

func (h *UserHandler) GetUser(c *gin.Context) {
    idStr := c.Param("id")
    id, err := strconv.ParseInt(idStr, 10, 64)
    if err != nil {
        respondError(c, http.StatusBadRequest, "INVALID_ID", "IDは数値である必要があります")
        return
    }

    user, err := h.service.GetUser(c.Request.Context(), id)
    if err != nil {
        if errors.Is(err, ErrNotFound) {
            respondError(c, http.StatusNotFound, "NOT_FOUND", "ユーザーが見つかりません")
            return
        }
        respondError(c, http.StatusInternalServerError, "INTERNAL_ERROR", "内部エラー")
        return
    }

    respondOK(c, user)
}

// ワイヤリング (cmd/api/main.go)
func main() {
    // 依存の構築
    db := setupDB()
    logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))

    userRepo := NewPostgresUserRepo(db)
    userService := NewUserService(userRepo, logger)
    userHandler := NewUserHandler(userService)

    // ルーティング
    r := gin.New()
    r.Use(gin.Recovery())

    v1 := r.Group("/api/v1")
    {
        users := v1.Group("/users")
        {
            users.GET("", userHandler.ListUsers)
            users.POST("", userHandler.CreateUser)
            users.GET("/:id", userHandler.GetUser)
            users.PUT("/:id", userHandler.UpdateUser)
            users.DELETE("/:id", userHandler.DeleteUser)
        }
    }

    // サーバー起動
    srv := &http.Server{Addr: ":8080", Handler: r}
    // ... Graceful Shutdown
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
│  custom      │
└──────┬───────┘
       ▼
  Handler Logic
```

### 図4: ミドルウェアチェーンの実行順序

```
リクエスト
  │
  ▼
┌─────────────────────────────────────────┐
│ Middleware 1                             │
│  │ Before処理                           │
│  │  ┌──────────────────────────────┐    │
│  │  │ Middleware 2                  │    │
│  │  │  │ Before処理               │    │
│  │  │  │  ┌───────────────────┐   │    │
│  │  │  │  │ Middleware 3       │   │    │
│  │  │  │  │  │ Before処理     │   │    │
│  │  │  │  │  │  ┌──────────┐  │   │    │
│  │  │  │  │  │  │ Handler  │  │   │    │
│  │  │  │  │  │  └──────────┘  │   │    │
│  │  │  │  │  │ After処理      │   │    │
│  │  │  │  └───────────────────┘   │    │
│  │  │  │ After処理                │    │
│  │  └──────────────────────────────┘    │
│  │ After処理                            │
└─────────────────────────────────────────┘
  │
  ▼
レスポンス
```

### 図5: JWT認証フロー

```
Client                API Server              Auth Service
  │                      │                        │
  │  POST /auth/login    │                        │
  │  {email, password}   │                        │
  │ ───────────────────> │                        │
  │                      │  verify credentials    │
  │                      │ ─────────────────────> │
  │                      │  <── user info ──────  │
  │                      │                        │
  │                      │  generate JWT          │
  │  <── {access_token,  │  (access + refresh)    │
  │       refresh_token} │                        │
  │                      │                        │
  │  GET /api/v1/users   │                        │
  │  Authorization:      │                        │
  │  Bearer <token>      │                        │
  │ ───────────────────> │                        │
  │                      │  validate JWT          │
  │                      │  extract claims        │
  │  <── 200 OK          │  set context           │
  │      {users: [...]}  │                        │
  │                      │                        │
  │  POST /auth/refresh  │                        │
  │  {refresh_token}     │                        │
  │ ───────────────────> │                        │
  │                      │  validate refresh      │
  │  <── {new tokens}    │  issue new pair        │
```

### 図6: Clean Architecture レイヤー

```
┌─────────────────────────────────────────────────┐
│  Presentation Layer (Handler)                    │
│  ┌───────────────────────────────────────────┐  │
│  │ Gin/Echo Handler                          │  │
│  │ リクエスト解析 → Service呼び出し → レスポンス  │  │
│  └────────────────────┬──────────────────────┘  │
│                       │ UserService interface    │
│  ┌────────────────────▼──────────────────────┐  │
│  │ Use Case Layer (Service)                  │  │
│  │ ビジネスロジック・バリデーション               │  │
│  └────────────────────┬──────────────────────┘  │
│                       │ UserRepository interface │
│  ┌────────────────────▼──────────────────────┐  │
│  │ Infrastructure Layer (Repository)         │  │
│  │ DB操作・外部API呼び出し                     │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘

依存の方向: Handler → Service → Repository (内側へ)
インターフェース定義: 各層の境界に配置
テスト: 各層をモックで独立テスト可能
```

### 図7: WebSocket 通信フロー

```
Client A       Server (Hub)       Client B
  │                │                  │
  │── HTTP GET ──>│                  │
  │  Upgrade:      │                  │
  │  websocket     │                  │
  │<─ 101 ────────│                  │
  │  Switching     │                  │
  │                │<── HTTP GET ────│
  │                │   Upgrade: ws   │
  │                │── 101 ────────>│
  │                │                  │
  │── message ──> │                  │
  │  "Hello"       │── broadcast ──>│
  │                │   "Hello"       │
  │                │                  │
  │                │<── message ────│
  │<── broadcast ──│   "Hi"          │
  │    "Hi"        │                  │
  │                │                  │
  │── ping ──────>│                  │
  │<── pong ──────│                  │
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
| ハンドラ型 | `func(*gin.Context)` | `func(echo.Context) error` | `func(w, r)` |
| カスタムコンテキスト | `c.Set()/c.Get()` | カスタムContext型 | `context.Value()` |
| Swagger連携 | gin-swagger | echo-swagger | 手動 |
| WebSocket | gorilla/websocket | gorilla/websocket | gorilla/websocket |
| テスト | httptest | httptest | httptest |
| Graceful Shutdown | 手動実装 | `e.Shutdown()` | `srv.Shutdown()` |

### 表2: Gin バリデーションタグ

| タグ | 意味 | 例 |
|------|------|-----|
| `required` | 必須 | `binding:"required"` |
| `email` | メール形式 | `binding:"email"` |
| `min` | 最小値/最小長 | `binding:"min=3"` |
| `max` | 最大値/最大長 | `binding:"max=100"` |
| `oneof` | 列挙値 | `binding:"oneof=admin user"` |
| `gte` | 以上 | `binding:"gte=0"` |
| `lte` | 以下 | `binding:"lte=150"` |
| `url` | URL形式 | `binding:"url"` |
| `uuid` | UUID形式 | `binding:"uuid"` |
| `datetime` | 日時形式 | `binding:"datetime=2006-01-02"` |
| `len` | 固定長 | `binding:"len=10"` |
| `alphanum` | 英数字のみ | `binding:"alphanum"` |
| `contains` | 含む | `binding:"contains=@"` |
| `excludes` | 含まない | `binding:"excludes= "` |
| `ip` | IPアドレス | `binding:"ip"` |
| `numeric` | 数値文字列 | `binding:"numeric"` |

### 表3: ミドルウェア比較

| ミドルウェア | Gin (組み込み) | Echo (組み込み) | 用途 |
|-------------|---------------|----------------|------|
| Logger | `gin.Logger()` | `middleware.Logger()` | リクエストログ |
| Recovery | `gin.Recovery()` | `middleware.Recover()` | パニック回復 |
| CORS | 別途追加 | `middleware.CORS()` | クロスオリジン |
| Rate Limit | 別途追加 | `middleware.RateLimiter()` | レート制限 |
| JWT | 別途追加 | `middleware.JWT()` | JWT認証 |
| Basic Auth | `gin.BasicAuth()` | `middleware.BasicAuth()` | Basic認証 |
| Gzip | 別途追加 | `middleware.Gzip()` | 圧縮 |
| Request ID | 別途追加 | `middleware.RequestID()` | リクエスト追跡 |
| Timeout | 別途追加 | `middleware.Timeout()` | タイムアウト |
| Secure | 別途追加 | `middleware.Secure()` | セキュリティヘッダ |
| CSRF | 別途追加 | `middleware.CSRF()` | CSRF対策 |
| Body Limit | 別途追加 | `middleware.BodyLimit()` | ボディサイズ制限 |

### 表4: エラーハンドリング比較

| 項目 | Gin | Echo |
|------|-----|------|
| エラー返却 | `c.JSON()` + `return` | `return error` |
| 中断 | `c.Abort()` / `c.AbortWithStatusJSON()` | `return echo.NewHTTPError()` |
| カスタムエラー | `gin.H{}` で自由形式 | `echo.HTTPError` 構造体 |
| 集約ハンドラ | なし (ミドルウェアで実装) | `e.HTTPErrorHandler` |
| エラーログ | ミドルウェアで取得 | `HTTPErrorHandler` 内 |
| パニック回復 | `gin.Recovery()` | `middleware.Recover()` |

### 表5: プロジェクト構成比較

| 規模 | 推奨構成 | フレームワーク選択 |
|------|---------|-----------------|
| 小規模 (API数 < 10) | flat構成 | net/http (1.22+) |
| 中規模 (API数 10-50) | レイヤード | Gin / Echo |
| 大規模 (API数 50+) | Clean Architecture | Gin / Echo + DI |
| マイクロサービス | DDD + gRPC | Gin (REST gateway) + gRPC |

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

gin.Contextはリクエストのライフサイクルに紐づいており、レスポンスが返却された後にアクセスすると不正な動作やパニックの原因となる。goroutineに渡す場合は、必要な値を先に取り出してプリミティブ型として渡すこと。

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

APIのクライアントはエラーレスポンスのフォーマットを予測してパースする必要がある。フォーマットが不統一だとクライアント側の実装が複雑になり、バグの原因となる。

### アンチパターン 3: ミドルウェアでのc.Next()忘れ

```go
// BAD: c.Next() を呼び忘れるとチェーンが停止
func loggingMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        log.Printf("Request: %s %s", c.Request.Method, c.Request.URL.Path)
        // c.Next() がない！後続のハンドラが実行されない
    }
}

// GOOD: 明示的に c.Next() を呼ぶ
func loggingMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        start := time.Now()
        log.Printf("Request: %s %s", c.Request.Method, c.Request.URL.Path)
        c.Next()
        log.Printf("Response: %d (%v)", c.Writer.Status(), time.Since(start))
    }
}
```

### アンチパターン 4: バリデーションの二重実装

```go
// BAD: ハンドラ内でバリデーションロジックを手書き
func createUser(c *gin.Context) {
    var req CreateUserRequest
    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(400, gin.H{"error": err.Error()})
        return
    }
    // バリデーションの二重実装
    if len(req.Name) < 2 {
        c.JSON(400, gin.H{"error": "name too short"})
        return
    }
    if !strings.Contains(req.Email, "@") {
        c.JSON(400, gin.H{"error": "invalid email"})
        return
    }
    // ...
}

// GOOD: バリデーションはbindingタグに集約
type CreateUserRequest struct {
    Name  string `json:"name" binding:"required,min=2,max=50"`
    Email string `json:"email" binding:"required,email"`
}

func createUser(c *gin.Context) {
    var req CreateUserRequest
    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(400, gin.H{"errors": formatValidationErrors(err)})
        return
    }
    // バリデーション通過後のビジネスロジックのみ
}
```

### アンチパターン 5: gin.Default() を本番で使う

```go
// BAD: gin.Default() はデバッグ向けのLogger付き
func main() {
    r := gin.Default()
    r.Run(":8080")
}

// GOOD: 本番環境では明示的にミドルウェアを設定
func main() {
    gin.SetMode(gin.ReleaseMode)
    r := gin.New()

    // 構造化ログ
    r.Use(StructuredLoggerMiddleware(slog.Default()))
    // パニックリカバリ（カスタム）
    r.Use(gin.CustomRecoveryWithWriter(nil, func(c *gin.Context, err any) {
        slog.Error("Panic recovered", "error", err)
        c.AbortWithStatusJSON(500, gin.H{
            "code":    "INTERNAL_ERROR",
            "message": "internal server error",
        })
    }))
    // セキュリティ
    r.Use(SecurityHeadersMiddleware())
    r.Use(CORSMiddleware(allowedOrigins))
    r.Use(RateLimitMiddleware(100, 200))

    r.Run(":8080")
}
```

### アンチパターン 6: サービス層をスキップする

```go
// BAD: ハンドラに直接DBアクセスコードを書く
func getUser(c *gin.Context) {
    id := c.Param("id")
    var user User
    err := db.QueryRow("SELECT * FROM users WHERE id = $1", id).Scan(&user.ID, &user.Name)
    if err != nil {
        c.JSON(500, gin.H{"error": err.Error()})
        return
    }
    c.JSON(200, user)
}

// GOOD: サービス層を経由する
func (h *UserHandler) GetUser(c *gin.Context) {
    id, err := strconv.ParseInt(c.Param("id"), 10, 64)
    if err != nil {
        respondError(c, http.StatusBadRequest, "INVALID_ID", "無効なIDです")
        return
    }

    user, err := h.service.GetUser(c.Request.Context(), id)
    if err != nil {
        if errors.Is(err, ErrNotFound) {
            respondError(c, http.StatusNotFound, "NOT_FOUND", "ユーザーが見つかりません")
            return
        }
        respondError(c, http.StatusInternalServerError, "INTERNAL_ERROR", "内部エラー")
        return
    }
    respondOK(c, user)
}
```

ハンドラにDBアクセスを直書きすると、テストが困難になり、ビジネスロジックの再利用ができない。サービス層を挟むことで、ハンドラはHTTPの入出力変換に集中できる。

---

## 5. FAQ

### Q1: GinとEchoのどちらを選ぶべきか？

両者の性能差はほぼなし。Ginはエコシステムが大きく情報が多い。Echoはコード設計がクリーンでerrorを返すハンドラが特徴的。チームの好みで選んでよいが、Go 1.22+の標準net/httpで十分なケースも多い。

判断基準:
- **Gin**: 大きなコミュニティが欲しい、日本語情報が多い方が良い、既存のGinプロジェクトへの追従
- **Echo**: error戻り値パターンが好み、組み込みミドルウェアが豊富、カスタムコンテキストを使いたい
- **net/http**: 外部依存を最小化したい、Go 1.22+ の新しいルーティングで十分、マイクロサービスで軽量にしたい

### Q2: Gin Releaseモードとは？

`gin.SetMode(gin.ReleaseMode)` で本番モードに切り替える。デバッグログが抑制され、パフォーマンスが若干向上する。環境変数 `GIN_MODE=release` でも設定可能。本番デプロイ時は必ずReleaseモードにすること。Debugモードのままだと、ルーティングテーブルの出力など不要な情報がログに含まれる。

```go
// 環境変数で切り替え
func init() {
    mode := os.Getenv("GIN_MODE")
    if mode == "" {
        mode = gin.DebugMode
    }
    gin.SetMode(mode)
}
```

### Q3: Swagger/OpenAPIはどう統合するか？

`swaggo/swag` を使い、ハンドラのコメントからSwagger仕様を自動生成する。`gin-swagger` または `echo-swagger` で `/swagger/index.html` を提供する。CI/CDパイプラインで `swag init` を実行し、生成されたファイルをバージョン管理に含めるのが一般的。

### Q4: Gin/Echoでcontext.Contextをどう使うか？

Gin では `c.Request.Context()` で標準の `context.Context` を取得できる。サービス層やリポジトリ層へは必ずこのcontextを伝搬させる。Echoでも `c.Request().Context()` で同様に取得可能。

```go
// Gin: context伝搬
func (h *UserHandler) GetUser(c *gin.Context) {
    ctx := c.Request.Context()
    user, err := h.service.GetUser(ctx, id) // ctxを渡す
    // ...
}

// Echo: context伝搬
func (h *UserHandler) GetUser(c echo.Context) error {
    ctx := c.Request().Context()
    user, err := h.service.GetUser(ctx, id) // ctxを渡す
    // ...
}
```

### Q5: バージョニングはどう設計するか？

URLパスでのバージョニングが最も一般的である。ヘッダーベースのバージョニングは実装が複雑になるため、URLパスが推奨される。

```go
// URLパスバージョニング
v1 := r.Group("/api/v1")
{
    v1.GET("/users", v1ListUsers)
}

v2 := r.Group("/api/v2")
{
    v2.GET("/users", v2ListUsers) // レスポンス構造が異なる
}
```

### Q6: テストでデータベースをどうモックするか？

インターフェースを定義し、テスト時にモック実装を注入する。これはClean Architectureのリポジトリパターンで自然に実現できる。

```go
// モックリポジトリ
type MockUserRepository struct {
    users map[int64]*User
}

func (m *MockUserRepository) FindByID(ctx context.Context, id int64) (*User, error) {
    user, ok := m.users[id]
    if !ok {
        return nil, ErrNotFound
    }
    return user, nil
}

// テストでの使用
func TestGetUser(t *testing.T) {
    mockRepo := &MockUserRepository{
        users: map[int64]*User{1: {ID: 1, Name: "Test"}},
    }
    service := NewUserService(mockRepo, slog.Default())
    handler := NewUserHandler(service)
    // ...
}
```

### Q7: 大量のルーティングを整理するには？

ルーティングをファイル分割し、SetupXxxRoutes 関数として定義するのがベストプラクティスである。

```go
// routes/user.go
func SetupUserRoutes(group *gin.RouterGroup, handler *UserHandler) {
    users := group.Group("/users")
    {
        users.GET("", handler.ListUsers)
        users.POST("", handler.CreateUser)
        users.GET("/:id", handler.GetUser)
        users.PUT("/:id", handler.UpdateUser)
        users.DELETE("/:id", handler.DeleteUser)
    }
}

// routes/order.go
func SetupOrderRoutes(group *gin.RouterGroup, handler *OrderHandler) {
    orders := group.Group("/orders")
    {
        orders.GET("", handler.ListOrders)
        orders.POST("", handler.CreateOrder)
    }
}

// main.go
func setupRoutes(r *gin.Engine, handlers *Handlers) {
    v1 := r.Group("/api/v1")
    v1.Use(authMiddleware)

    SetupUserRoutes(v1, handlers.User)
    SetupOrderRoutes(v1, handlers.Order)
    SetupProductRoutes(v1, handlers.Product)
}
```

### Q8: Gin/EchoでのWebSocket実装のポイントは？

WebSocketはGin/Echoのルーティングで登録し、`gorilla/websocket` でアップグレードする。接続管理にはHub パターンを使い、goroutineでread/writeを分離する。本番ではpingによるコネクション死活監視が必須。

---

## まとめ

| 概念 | 要点 |
|------|------|
| Gin | 高速・大エコシステム。gin.H、binding |
| Echo | クリーン設計。error戻り値パターン |
| ルーティング | Radix tree で高速パスマッチ |
| バリデーション | go-playground/validator ベース |
| ミドルウェア | グループ単位で適用可能 |
| JWT認証 | golang-jwt でトークン発行・検証 |
| Swagger | swaggo で自動生成 |
| テスト | httptest + testify でユニット/統合テスト |
| WebSocket | gorilla/websocket + Hub パターン |
| 本番運用 | Graceful Shutdown, 構造化ログ, ヘルスチェック |
| Clean Architecture | Handler → Service → Repository の依存方向 |

---

## 次に読むべきガイド

- [02-database.md](./02-database.md) -- データベース接続
- [03-grpc.md](./03-grpc.md) -- gRPC
- [04-testing.md](./04-testing.md) -- テスト
- [00-net-http.md](./00-net-http.md) -- 標準net/http

---

## 参考文献

1. **Gin Web Framework** -- https://gin-gonic.com/docs/
2. **Echo -- High performance, extensible, minimalist Go web framework** -- https://echo.labstack.com/
3. **swaggo/swag** -- https://github.com/swaggo/swag
4. **go-playground/validator** -- https://github.com/go-playground/validator
5. **gorilla/websocket** -- https://github.com/gorilla/websocket
6. **golang-jwt/jwt** -- https://github.com/golang-jwt/jwt
