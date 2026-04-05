# Gin / Echo -- Go Web Frameworks

> Gin and Echo are the most popular web frameworks in Go, providing high-performance routing, middleware, validation, and Swagger integration.

---

## What You Will Learn in This Chapter

1. **Gin / Echo Basics** -- Routing and Handlers
2. **Middleware and Validation** -- Implementing Cross-Cutting Concerns
3. **Swagger / OpenAPI** -- Automatic API Specification Generation
4. **Testing** -- Testing Techniques for Handlers and Middleware
5. **Production Operations** -- Graceful Shutdown, Structured Logging, and Health Checks


## Prerequisites

Before reading this guide, having the following knowledge will help deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Familiarity with the content of [net/http -- Go Standard HTTP Server](./00-net-http.md)

---

### Code Example 1: Gin Basic Setup

```go
import "github.com/gin-gonic/gin"

func main() {
    r := gin.Default() // Includes Logger + Recovery middleware

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

Gin's `Default()` returns an engine with `Logger` and `Recovery` middleware automatically included. Using `New()` gives you a bare engine without any middleware. In production environments, it is recommended to use `New()` and explicitly add the required middleware.

Routing parameters are defined in the `:id` format and retrieved with `c.Param("id")`. Wildcard parameters use the `*filepath` format and are retrieved with `c.Param("filepath")`.

```go
// Wildcard routing example
r.GET("/files/*filepath", func(c *gin.Context) {
    filepath := c.Param("filepath")
    // filepath = "/images/logo.png" (includes leading slash)
    c.String(http.StatusOK, "Serving: %s", filepath)
})
```

There are also multiple ways to retrieve query parameters.

```go
func listUsers(c *gin.Context) {
    // Query parameters
    page := c.DefaultQuery("page", "1")
    limit := c.DefaultQuery("limit", "20")
    sort := c.Query("sort") // Empty string as default

    // Numeric conversion
    pageNum, err := strconv.Atoi(page)
    if err != nil || pageNum < 1 {
        pageNum = 1
    }
    limitNum, err := strconv.Atoi(limit)
    if err != nil || limitNum < 1 || limitNum > 100 {
        limitNum = 20
    }

    // Response with pagination
    c.JSON(http.StatusOK, gin.H{
        "page":  pageNum,
        "limit": limitNum,
        "sort":  sort,
        "users": []gin.H{},
    })
}
```

### Code Example 2: Echo Basic Setup

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

A key feature of Echo is that handlers return `error`. This enables centralized error handling and prevents bugs where you forget to `return nil` after calling `c.JSON()` in a handler.

```go
// Echo's error handling is controlled by the handler's return value
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

Echo also supports wildcards that capture everything after a path segment, in addition to routing parameters.

```go
// Echo wildcard
e.GET("/files/*", func(c echo.Context) error {
    filepath := c.Param("*")
    return c.String(http.StatusOK, "Serving: "+filepath)
})

// Query parameters
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

### Code Example 3: Gin Validation

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
    // Validation passed
    c.JSON(http.StatusCreated, gin.H{"name": req.Name})
}
```

Gin's validation internally uses `go-playground/validator`. You can also register custom validation rules.

```go
// Registering custom validation rules
func setupValidator() {
    if v, ok := binding.Validator.Engine().(*validator.Validate); ok {
        // Custom validation: Japanese phone number
        v.RegisterValidation("jpphone", func(fl validator.FieldLevel) bool {
            phone := fl.Field().String()
            matched, _ := regexp.MatchString(`^0\d{1,4}-?\d{1,4}-?\d{4}$`, phone)
            return matched
        })

        // Custom validation: password strength
        v.RegisterValidation("strongpassword", func(fl validator.FieldLevel) bool {
            password := fl.Field().String()
            hasUpper := regexp.MustCompile(`[A-Z]`).MatchString(password)
            hasLower := regexp.MustCompile(`[a-z]`).MatchString(password)
            hasNumber := regexp.MustCompile(`[0-9]`).MatchString(password)
            hasSpecial := regexp.MustCompile(`[!@#$%^&*]`).MatchString(password)
            return hasUpper && hasLower && hasNumber && hasSpecial
        })

        // Use JSON tag names in error messages
        v.RegisterTagNameFunc(func(fld reflect.StructField) string {
            name := strings.SplitN(fld.Tag.Get("json"), ",", 2)[0]
            if name == "-" {
                return ""
            }
            return name
        })
    }
}

// Struct using custom validations
type RegisterRequest struct {
    Name     string `json:"name" binding:"required,min=2,max=50"`
    Email    string `json:"email" binding:"required,email"`
    Phone    string `json:"phone" binding:"required,jpphone"`
    Password string `json:"password" binding:"required,min=8,strongpassword"`
}
```

It is also important to have a function that converts validation errors into user-friendly messages.

```go
// Converting validation errors
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
        return fmt.Sprintf("%s is required", fe.Field())
    case "email":
        return fmt.Sprintf("%s is not a valid email address", fe.Field())
    case "min":
        return fmt.Sprintf("%s must be at least %s", fe.Field(), fe.Param())
    case "max":
        return fmt.Sprintf("%s must be at most %s", fe.Field(), fe.Param())
    case "jpphone":
        return fmt.Sprintf("%s is not a valid Japanese phone number", fe.Field())
    case "strongpassword":
        return "Password must contain uppercase, lowercase, numbers, and special characters"
    default:
        return fmt.Sprintf("%s does not satisfy %s", fe.Field(), fe.Tag())
    }
}

// Usage in handler
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

### Code Example 4: Gin Middleware Groups

```go
func main() {
    r := gin.Default()

    // Public API
    public := r.Group("/api/v1")
    {
        public.POST("/login", login)
        public.POST("/register", register)
    }

    // Authentication required API
    authorized := r.Group("/api/v1")
    authorized.Use(authMiddleware())
    {
        authorized.GET("/profile", getProfile)
        authorized.PUT("/profile", updateProfile)
    }

    // Admin API
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

The execution order of middleware is important. Calling `c.Next()` executes the subsequent middleware and handler, after which the code following `c.Next()` is executed. Calling `c.Abort()` interrupts the chain.

```go
// Example to understand middleware execution order
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

// Output order:
// middleware1: before
// middleware2: before
// handler
// middleware2: after
// middleware1: after
```

### Code Example 5: Echo Custom Validator

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

When using a custom validator in Echo, you can also customize the validation error formatting.

```go
// Extended custom validator
type CustomValidator struct {
    validator *validator.Validate
}

func NewCustomValidator() *CustomValidator {
    v := validator.New()

    // Use JSON tag names as field names
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

// Custom error type
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
        return "This field is required"
    case "email":
        return "Please enter a valid email address"
    default:
        return fe.Error()
    }
}
```

### Code Example 6: Gin Unified Response Structure

A unified response structure is important in production APIs.

```go
// Unified response struct
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

// Response helper functions
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

// Usage example in handler
func listUsers(c *gin.Context) {
    page, _ := strconv.Atoi(c.DefaultQuery("page", "1"))
    perPage, _ := strconv.Atoi(c.DefaultQuery("per_page", "20"))

    users, total, err := userService.List(c.Request.Context(), page, perPage)
    if err != nil {
        respondError(c, http.StatusInternalServerError, "INTERNAL_ERROR", "Failed to retrieve user list")
        return
    }
    respondPaginated(c, users, page, perPage, total)
}
```

### Code Example 7: Echo Middleware Details

Echo provides a rich set of built-in middleware.

```go
func setupMiddlewares(e *echo.Echo) {
    // Recovery
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

    // Rate limiting
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

    // Request ID
    e.Use(middleware.RequestID())

    // Timeout
    e.Use(middleware.TimeoutWithConfig(middleware.TimeoutConfig{
        Timeout: 30 * time.Second,
    }))

    // Gzip compression
    e.Use(middleware.GzipWithConfig(middleware.GzipConfig{
        Level: 5,
        Skipper: func(c echo.Context) bool {
            return strings.Contains(c.Path(), "ws")
        },
    }))

    // Security headers
    e.Use(middleware.SecureWithConfig(middleware.SecureConfig{
        XSSProtection:         "1; mode=block",
        ContentTypeNosniff:    "nosniff",
        XFrameOptions:         "DENY",
        HSTSMaxAge:            31536000,
        ContentSecurityPolicy: "default-src 'self'",
    }))
}
```

### Code Example 8: Gin Custom Middleware Collection

A collection of custom middleware commonly used in production environments.

```go
// Request ID middleware
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

// Structured logger middleware
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

// CORS middleware
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

// Rate limiting middleware (Token Bucket)
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

    // Periodic cleanup of stale entries
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
                "message": "Request rate limit exceeded",
            })
            return
        }
        c.Next()
    }
}

// Timeout middleware
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
                "message": "Request timed out",
            })
        }
    }
}

// Security headers middleware
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

### Code Example 9: Complete Gin JWT Authentication Implementation

JWT authentication is required in many APIs.

```go
import (
    "github.com/golang-jwt/jwt/v5"
)

// JWT configuration
type JWTConfig struct {
    SecretKey     []byte
    Issuer        string
    AccessExpiry  time.Duration
    RefreshExpiry time.Duration
}

// Custom claims
type Claims struct {
    UserID int64  `json:"user_id"`
    Email  string `json:"email"`
    Role   string `json:"role"`
    jwt.RegisteredClaims
}

// Token pair
type TokenPair struct {
    AccessToken  string `json:"access_token"`
    RefreshToken string `json:"refresh_token"`
    ExpiresAt    int64  `json:"expires_at"`
}

// Token generation
func (cfg *JWTConfig) GenerateTokenPair(userID int64, email, role string) (*TokenPair, error) {
    now := time.Now()

    // Access token
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

    // Refresh token
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

// Token validation
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

// JWT authentication middleware
func JWTAuthMiddleware(jwtCfg *JWTConfig) gin.HandlerFunc {
    return func(c *gin.Context) {
        authHeader := c.GetHeader("Authorization")
        if authHeader == "" {
            c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
                "code":    "UNAUTHORIZED",
                "message": "Authorization header is required",
            })
            return
        }

        // Validate "Bearer <token>" format
        parts := strings.SplitN(authHeader, " ", 2)
        if len(parts) != 2 || !strings.EqualFold(parts[0], "bearer") {
            c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
                "code":    "INVALID_TOKEN_FORMAT",
                "message": "Invalid Bearer token format",
            })
            return
        }

        claims, err := jwtCfg.ValidateToken(parts[1])
        if err != nil {
            c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
                "code":    "INVALID_TOKEN",
                "message": "Token is invalid or expired",
            })
            return
        }

        // Set user info in context
        c.Set("userID", claims.UserID)
        c.Set("email", claims.Email)
        c.Set("role", claims.Role)
        c.Set("claims", claims)
        c.Next()
    }
}

// Role-based authorization middleware
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
                "message": "Permission denied",
            })
            return
        }

        if !roleSet[role.(string)] {
            c.AbortWithStatusJSON(http.StatusForbidden, gin.H{
                "code":    "INSUFFICIENT_ROLE",
                "message": fmt.Sprintf("Required role: %v", roles),
            })
            return
        }
        c.Next()
    }
}

// Login handler
func loginHandler(jwtCfg *JWTConfig, userService UserService) gin.HandlerFunc {
    return func(c *gin.Context) {
        var req LoginRequest
        if err := c.ShouldBindJSON(&req); err != nil {
            respondError(c, http.StatusBadRequest, "VALIDATION_ERROR", err.Error())
            return
        }

        user, err := userService.Authenticate(c.Request.Context(), req.Email, req.Password)
        if err != nil {
            respondError(c, http.StatusUnauthorized, "INVALID_CREDENTIALS", "Invalid email or password")
            return
        }

        tokens, err := jwtCfg.GenerateTokenPair(user.ID, user.Email, user.Role)
        if err != nil {
            respondError(c, http.StatusInternalServerError, "TOKEN_GENERATION_ERROR", "Failed to generate token")
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

### Code Example 10: Echo Groups and Custom Context

In Echo, you can extend common handler functionality using a custom context.

```go
// Custom context
type AppContext struct {
    echo.Context
    UserID int64
    Role   string
}

// Custom context middleware
func CustomContextMiddleware(next echo.HandlerFunc) echo.HandlerFunc {
    return func(c echo.Context) error {
        cc := &AppContext{Context: c}
        return next(cc)
    }
}

// Handler using custom context
func getProfile(c echo.Context) error {
    cc := c.(*AppContext)
    user, err := userService.FindByID(cc.Request().Context(), cc.UserID)
    if err != nil {
        return echo.NewHTTPError(http.StatusNotFound, "user not found")
    }
    return cc.JSON(http.StatusOK, user)
}

// Echo group configuration
func setupRoutes(e *echo.Echo, jwtCfg *JWTConfig) {
    // API versioning
    v1 := e.Group("/api/v1")
    v1.Use(CustomContextMiddleware)

    // Public endpoints
    public := v1.Group("")
    {
        public.POST("/auth/login", loginHandler)
        public.POST("/auth/register", registerHandler)
        public.POST("/auth/refresh", refreshTokenHandler)
    }

    // Authentication required endpoints
    auth := v1.Group("")
    auth.Use(echoJWTMiddleware(jwtCfg))
    {
        auth.GET("/profile", getProfile)
        auth.PUT("/profile", updateProfile)
        auth.GET("/users/:id", getUserByID)
    }

    // Admin endpoints
    admin := v1.Group("/admin")
    admin.Use(echoJWTMiddleware(jwtCfg), echoRequireRole("admin"))
    {
        admin.GET("/users", adminListUsers)
        admin.DELETE("/users/:id", adminDeleteUser)
        admin.GET("/stats", getStats)
    }
}
```

### Code Example 11: Swagger / OpenAPI Integration

Auto-generating Swagger API documentation with Gin.

```go
// Handler with Swagger annotations

// @Summary Get user list
// @Description Get a paginated list of users
// @Tags users
// @Accept json
// @Produce json
// @Param page query int false "Page number" default(1)
// @Param per_page query int false "Items per page" default(20) maximum(100)
// @Param sort query string false "Sort field" Enums(name, email, created_at)
// @Param order query string false "Sort order" Enums(asc, desc) default(asc)
// @Success 200 {object} Response{data=[]User,meta=Meta} "Success"
// @Failure 400 {object} ErrorResponse "Validation error"
// @Failure 401 {object} ErrorResponse "Authentication error"
// @Failure 500 {object} ErrorResponse "Server error"
// @Security BearerAuth
// @Router /api/v1/users [get]
func listUsers(c *gin.Context) {
    // Implementation
}

// @Summary Create user
// @Description Create a new user
// @Tags users
// @Accept json
// @Produce json
// @Param request body CreateUserRequest true "Create user request"
// @Success 201 {object} Response{data=User} "Created successfully"
// @Failure 400 {object} ErrorResponse "Validation error"
// @Failure 409 {object} ErrorResponse "Email already exists"
// @Failure 500 {object} ErrorResponse "Server error"
// @Security BearerAuth
// @Router /api/v1/users [post]
func createUser(c *gin.Context) {
    // Implementation
}

// @Summary Get user
// @Description Get a user by ID
// @Tags users
// @Accept json
// @Produce json
// @Param id path int true "User ID"
// @Success 200 {object} Response{data=User} "Success"
// @Failure 404 {object} ErrorResponse "User not found"
// @Failure 500 {object} ErrorResponse "Server error"
// @Security BearerAuth
// @Router /api/v1/users/{id} [get]
func getUser(c *gin.Context) {
    // Implementation
}

// Swagger configuration
// @title My API
// @version 1.0
// @description User Management API
// @host localhost:8080
// @BasePath /api/v1
// @securityDefinitions.apikey BearerAuth
// @in header
// @name Authorization

func main() {
    r := gin.Default()

    // Swagger endpoint
    r.GET("/swagger/*any", ginSwagger.WrapHandler(swaggerFiles.Handler))

    // API routing
    v1 := r.Group("/api/v1")
    {
        v1.GET("/users", listUsers)
        v1.POST("/users", createUser)
        v1.GET("/users/:id", getUser)
    }

    r.Run(":8080")
}
```

Command to generate Swagger documentation.

```bash
# Install swag
go install github.com/swaggo/swag/cmd/swag@latest

# Generate documentation
swag init -g cmd/api/main.go -o docs

# Generated files:
# docs/docs.go
# docs/swagger.json
# docs/swagger.yaml
```

### Code Example 12: Gin Testing

Handler testing uses httptest.

```go
import (
    "net/http"
    "net/http/httptest"
    "testing"

    "github.com/gin-gonic/gin"
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
)

// Helper to create a Gin engine for testing
func setupTestRouter() *gin.Engine {
    gin.SetMode(gin.TestMode)
    r := gin.New()
    return r
}

// Unit test for handler
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
            name:       "Success: get user",
            userID:     "1",
            wantStatus: http.StatusOK,
            wantBody:   `"name":"Tanaka"`,
        },
        {
            name:       "Error: user does not exist",
            userID:     "999",
            wantStatus: http.StatusNotFound,
            wantBody:   `"code":"NOT_FOUND"`,
        },
        {
            name:       "Error: invalid ID",
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

// POST handler test
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
            name:       "Success: create user",
            body:       `{"name":"Yamada","email":"yamada@example.com","password":"P@ssw0rd!"}`,
            wantStatus: http.StatusCreated,
        },
        {
            name:       "Error: empty name",
            body:       `{"name":"","email":"yamada@example.com","password":"P@ssw0rd!"}`,
            wantStatus: http.StatusBadRequest,
        },
        {
            name:       "Error: invalid email",
            body:       `{"name":"Yamada","email":"invalid","password":"P@ssw0rd!"}`,
            wantStatus: http.StatusBadRequest,
        },
        {
            name:       "Error: JSON parse error",
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

// Middleware test
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

    t.Run("No token", func(t *testing.T) {
        req := httptest.NewRequest(http.MethodGet, "/protected", nil)
        w := httptest.NewRecorder()
        r.ServeHTTP(w, req)
        assert.Equal(t, http.StatusUnauthorized, w.Code)
    })

    t.Run("Valid token", func(t *testing.T) {
        tokens, err := jwtCfg.GenerateTokenPair(42, "test@example.com", "user")
        require.NoError(t, err)

        req := httptest.NewRequest(http.MethodGet, "/protected", nil)
        req.Header.Set("Authorization", "Bearer "+tokens.AccessToken)
        w := httptest.NewRecorder()
        r.ServeHTTP(w, req)

        assert.Equal(t, http.StatusOK, w.Code)
        assert.Contains(t, w.Body.String(), `"user_id":42`)
    })

    t.Run("Invalid token", func(t *testing.T) {
        req := httptest.NewRequest(http.MethodGet, "/protected", nil)
        req.Header.Set("Authorization", "Bearer invalid-token")
        w := httptest.NewRecorder()
        r.ServeHTTP(w, req)
        assert.Equal(t, http.StatusUnauthorized, w.Code)
    })
}
```

### Code Example 13: Echo Testing

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
            name:       "Success",
            userID:     "1",
            wantStatus: http.StatusOK,
            wantBody:   `"name":"Tanaka"`,
        },
        {
            name:       "User not found",
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

// Echo middleware test
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

    // First 2 requests succeed
    for i := 0; i < 2; i++ {
        req := httptest.NewRequest(http.MethodGet, "/test", nil)
        rec := httptest.NewRecorder()
        e.ServeHTTP(rec, req)
        assert.Equal(t, http.StatusOK, rec.Code)
    }

    // Third request is rate limited
    req := httptest.NewRequest(http.MethodGet, "/test", nil)
    rec := httptest.NewRecorder()
    e.ServeHTTP(rec, req)
    assert.Equal(t, http.StatusTooManyRequests, rec.Code)
}
```

### Code Example 14: Gin Graceful Shutdown

```go
func main() {
    // Set release mode
    gin.SetMode(gin.ReleaseMode)

    r := gin.New()

    // Middleware setup
    r.Use(RequestIDMiddleware())
    r.Use(StructuredLoggerMiddleware(slog.Default()))
    r.Use(gin.Recovery())
    r.Use(CORSMiddleware([]string{"https://example.com"}))
    r.Use(SecurityHeadersMiddleware())

    // Health check
    r.GET("/health", func(c *gin.Context) {
        c.JSON(http.StatusOK, gin.H{
            "status": "ok",
            "time":   time.Now().UTC().Format(time.RFC3339),
        })
    })

    // Readiness check (also verifies external dependencies such as DB)
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

    // API routing
    setupRoutes(r)

    // Server configuration
    srv := &http.Server{
        Addr:         ":8080",
        Handler:      r,
        ReadTimeout:  15 * time.Second,
        WriteTimeout: 30 * time.Second,
        IdleTimeout:  60 * time.Second,
        // Header size limit
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

### Code Example 15: Echo Graceful Shutdown

```go
func main() {
    e := echo.New()
    e.HideBanner = true

    // Middleware setup
    setupMiddlewares(e)

    // Routing setup
    setupRoutes(e, jwtCfg)

    // Health check
    e.GET("/health", func(c echo.Context) error {
        return c.JSON(http.StatusOK, map[string]string{
            "status": "ok",
            "time":   time.Now().UTC().Format(time.RFC3339),
        })
    })

    // Custom HTTP error handler
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

        // Unknown error
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

### Code Example 16: File Upload

File upload handling in Gin and Echo.

```go
// Gin: Single file upload
func uploadFile(c *gin.Context) {
    file, err := c.FormFile("file")
    if err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": "File is required"})
        return
    }

    // Validation
    if file.Size > 10<<20 { // 10MB limit
        c.JSON(http.StatusBadRequest, gin.H{"error": "File size is too large (max 10MB)"})
        return
    }

    // MIME type check
    allowedTypes := map[string]bool{
        "image/jpeg": true,
        "image/png":  true,
        "image/gif":  true,
        "image/webp": true,
    }

    src, err := file.Open()
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to read file"})
        return
    }
    defer src.Close()

    // Determine MIME type from the first 512 bytes
    buffer := make([]byte, 512)
    _, err = src.Read(buffer)
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to read file"})
        return
    }
    contentType := http.DetectContentType(buffer)
    if !allowedTypes[contentType] {
        c.JSON(http.StatusBadRequest, gin.H{
            "error": fmt.Sprintf("File type not allowed: %s", contentType),
        })
        return
    }

    // Save file (generate unique name)
    ext := filepath.Ext(file.Filename)
    filename := fmt.Sprintf("%s%s", uuid.New().String(), ext)
    dst := filepath.Join("uploads", filename)

    if err := c.SaveUploadedFile(file, dst); err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to save file"})
        return
    }

    c.JSON(http.StatusOK, gin.H{
        "filename": filename,
        "size":     file.Size,
        "type":     contentType,
        "url":      fmt.Sprintf("/uploads/%s", filename),
    })
}

// Gin: Multiple file upload
func uploadMultipleFiles(c *gin.Context) {
    form, err := c.MultipartForm()
    if err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
        return
    }

    files := form.File["files"]
    if len(files) == 0 {
        c.JSON(http.StatusBadRequest, gin.H{"error": "Files are required"})
        return
    }
    if len(files) > 10 {
        c.JSON(http.StatusBadRequest, gin.H{"error": "Maximum 10 files allowed"})
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

// Echo: File upload
func echoUploadFile(c echo.Context) error {
    file, err := c.FormFile("file")
    if err != nil {
        return echo.NewHTTPError(http.StatusBadRequest, "File is required")
    }

    if file.Size > 10<<20 {
        return echo.NewHTTPError(http.StatusBadRequest, "File size is too large")
    }

    src, err := file.Open()
    if err != nil {
        return echo.NewHTTPError(http.StatusInternalServerError, "Failed to read file")
    }
    defer src.Close()

    ext := filepath.Ext(file.Filename)
    filename := fmt.Sprintf("%s%s", uuid.New().String(), ext)
    dst, err := os.Create(filepath.Join("uploads", filename))
    if err != nil {
        return echo.NewHTTPError(http.StatusInternalServerError, "Failed to save file")
    }
    defer dst.Close()

    if _, err = io.Copy(dst, src); err != nil {
        return echo.NewHTTPError(http.StatusInternalServerError, "Failed to copy file")
    }

    return c.JSON(http.StatusOK, map[string]interface{}{
        "filename": filename,
        "size":     file.Size,
    })
}
```

### Code Example 17: WebSocket

WebSocket implementation in Gin and Echo.

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

// WebSocket Hub (connection management)
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

// Gin WebSocket handler
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

### Code Example 18: Dependency Injection Pattern (Clean Architecture)

Clean architecture is important in real-world projects.

```go
// Domain layer (domain/user.go)
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

// Use case layer (usecase/user_service.go)
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

// Infrastructure layer (infrastructure/user_repository.go)
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

// Presentation layer (handler/user_handler.go)
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
        respondError(c, http.StatusBadRequest, "INVALID_ID", "ID must be a number")
        return
    }

    user, err := h.service.GetUser(c.Request.Context(), id)
    if err != nil {
        if errors.Is(err, ErrNotFound) {
            respondError(c, http.StatusNotFound, "NOT_FOUND", "User not found")
            return
        }
        respondError(c, http.StatusInternalServerError, "INTERNAL_ERROR", "Internal error")
        return
    }

    respondOK(c, user)
}

// Wiring (cmd/api/main.go)
func main() {
    // Build dependencies
    db := setupDB()
    logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))

    userRepo := NewPostgresUserRepo(db)
    userService := NewUserService(userRepo, logger)
    userHandler := NewUserHandler(userService)

    // Routing
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

    // Start server
    srv := &http.Server{Addr: ":8080", Handler: r}
    // ... Graceful Shutdown
}
```

---

## 2. ASCII Diagrams

### Diagram 1: Gin Request Processing Flow

```
Request
  |
  v
+--------------+
| gin.Engine   |
| +----------+ |
| | Logger   | |  Global Middleware
| | Recovery | |
| +----+-----+ |
|      v       |
| +----------+ |
| | RadixTree| |  Route Matching
| | Router   | |  O(1) path lookup
| +----+-----+ |
|      v       |
| +----------+ |
| | Group MW | |  Group Middleware
| +----+-----+ |
|      v       |
| +----------+ |
| | Handler  | |  Business Logic
| +----------+ |
+--------------+
```

### Diagram 2: Routing Groups

```
/api/v1
+-- /login          [POST]  (Public)
+-- /register       [POST]  (Public)
+-- /profile        [GET]   (Auth required)
+-- /profile        [PUT]   (Auth required)
+-- /admin
    +-- /users      [GET]   (Auth + Admin role)

Middleware application:
  Public:        Logger -> Recovery -> Handler
  Auth required: Logger -> Recovery -> Auth -> Handler
  Admin:         Logger -> Recovery -> Auth -> Admin -> Handler
```

### Diagram 3: Validation Processing Flow

```
JSON Request Body
      |
      v
+--------------+
| Bind (JSON)  | -- Syntax error -> 400 Bad Request
+------+-------+
       v
+--------------+
| Validate     | -- Validation -- Failure -> 400 + Error details
|  required    |    error
|  min/max     |
|  email       |
|  custom      |
+------+-------+
       v
  Handler Logic
```

### Diagram 4: Middleware Chain Execution Order

```
Request
  |
  v
+-----------------------------------------+
| Middleware 1                             |
|  | Before processing                    |
|  |  +------------------------------+   |
|  |  | Middleware 2                  |   |
|  |  |  | Before processing         |   |
|  |  |  |  +-------------------+    |   |
|  |  |  |  | Middleware 3      |    |   |
|  |  |  |  |  | Before proc.  |    |   |
|  |  |  |  |  |  +----------+ |    |   |
|  |  |  |  |  |  | Handler  | |    |   |
|  |  |  |  |  |  +----------+ |    |   |
|  |  |  |  |  | After proc.   |    |   |
|  |  |  |  +-------------------+    |   |
|  |  |  | After processing          |   |
|  |  +------------------------------+   |
|  | After processing                     |
+-----------------------------------------+
  |
  v
Response
```

### Diagram 5: JWT Authentication Flow

```
Client                API Server              Auth Service
  |                      |                        |
  |  POST /auth/login    |                        |
  |  {email, password}   |                        |
  | ------------------> |                        |
  |                      |  verify credentials    |
  |                      | ---------------------> |
  |                      |  <-- user info ------  |
  |                      |                        |
  |                      |  generate JWT          |
  |  <-- {access_token,  |  (access + refresh)    |
  |       refresh_token} |                        |
  |                      |                        |
  |  GET /api/v1/users   |                        |
  |  Authorization:      |                        |
  |  Bearer <token>      |                        |
  | ------------------> |                        |
  |                      |  validate JWT          |
  |                      |  extract claims        |
  |  <-- 200 OK          |  set context           |
  |      {users: [...]}  |                        |
  |                      |                        |
  |  POST /auth/refresh  |                        |
  |  {refresh_token}     |                        |
  | ------------------> |                        |
  |                      |  validate refresh      |
  |  <-- {new tokens}    |  issue new pair        |
```

### Diagram 6: Clean Architecture Layers

```
+-------------------------------------------------+
|  Presentation Layer (Handler)                    |
|  +-------------------------------------------+  |
|  | Gin/Echo Handler                          |  |
|  | Parse request -> Call service -> Response  |  |
|  +--------------------+----------------------+  |
|                       | UserService interface    |
|  +--------------------v----------------------+  |
|  | Use Case Layer (Service)                  |  |
|  | Business logic and validation             |  |
|  +--------------------+----------------------+  |
|                       | UserRepository interface |
|  +--------------------v----------------------+  |
|  | Infrastructure Layer (Repository)         |  |
|  | DB operations and external API calls      |  |
|  +-------------------------------------------+  |
+-------------------------------------------------+

Dependency direction: Handler -> Service -> Repository (inward)
Interface definitions: Placed at layer boundaries
Testing: Each layer can be independently tested with mocks
```

### Diagram 7: WebSocket Communication Flow

```
Client A       Server (Hub)       Client B
  |                |                  |
  |-- HTTP GET -->|                  |
  |  Upgrade:      |                  |
  |  websocket     |                  |
  |<- 101 --------|                  |
  |  Switching     |                  |
  |                |<-- HTTP GET ----|
  |                |   Upgrade: ws   |
  |                |-- 101 -------->|
  |                |                  |
  |-- message --> |                  |
  |  "Hello"       |-- broadcast -->|
  |                |   "Hello"       |
  |                |                  |
  |                |<-- message ----|
  |<-- broadcast --|   "Hi"          |
  |    "Hi"        |                  |
  |                |                  |
  |-- ping ------>|                  |
  |<-- pong ------|                  |
```

---

## 3. Comparison Tables

### Table 1: Gin vs Echo vs Standard net/http

| Item | Gin | Echo | net/http (1.22+) |
|------|-----|------|-----------------|
| Performance | Very fast | Very fast | Fast |
| Routing | Radix tree | Radix tree | Pattern matching |
| Validation | binding (validator) | Separate addition | None |
| Middleware | `gin.HandlerFunc` | `echo.MiddlewareFunc` | `func(http.Handler) http.Handler` |
| Error handling | `c.AbortWithJSON` | `echo.HTTPError` | `http.Error` |
| GitHub Stars | 80k+ | 30k+ | Standard |
| Minimal dependencies | Medium | Medium | No dependencies |
| Handler type | `func(*gin.Context)` | `func(echo.Context) error` | `func(w, r)` |
| Custom context | `c.Set()/c.Get()` | Custom Context type | `context.Value()` |
| Swagger integration | gin-swagger | echo-swagger | Manual |
| WebSocket | gorilla/websocket | gorilla/websocket | gorilla/websocket |
| Testing | httptest | httptest | httptest |
| Graceful Shutdown | Manual implementation | `e.Shutdown()` | `srv.Shutdown()` |

### Table 2: Gin Validation Tags

| Tag | Meaning | Example |
|-----|---------|---------|
| `required` | Required | `binding:"required"` |
| `email` | Email format | `binding:"email"` |
| `min` | Minimum value/length | `binding:"min=3"` |
| `max` | Maximum value/length | `binding:"max=100"` |
| `oneof` | Enumerated values | `binding:"oneof=admin user"` |
| `gte` | Greater than or equal | `binding:"gte=0"` |
| `lte` | Less than or equal | `binding:"lte=150"` |
| `url` | URL format | `binding:"url"` |
| `uuid` | UUID format | `binding:"uuid"` |
| `datetime` | DateTime format | `binding:"datetime=2006-01-02"` |
| `len` | Fixed length | `binding:"len=10"` |
| `alphanum` | Alphanumeric only | `binding:"alphanum"` |
| `contains` | Contains | `binding:"contains=@"` |
| `excludes` | Excludes | `binding:"excludes= "` |
| `ip` | IP address | `binding:"ip"` |
| `numeric` | Numeric string | `binding:"numeric"` |

### Table 3: Middleware Comparison

| Middleware | Gin (built-in) | Echo (built-in) | Purpose |
|-----------|----------------|-----------------|---------|
| Logger | `gin.Logger()` | `middleware.Logger()` | Request logging |
| Recovery | `gin.Recovery()` | `middleware.Recover()` | Panic recovery |
| CORS | Separate addition | `middleware.CORS()` | Cross-origin |
| Rate Limit | Separate addition | `middleware.RateLimiter()` | Rate limiting |
| JWT | Separate addition | `middleware.JWT()` | JWT authentication |
| Basic Auth | `gin.BasicAuth()` | `middleware.BasicAuth()` | Basic authentication |
| Gzip | Separate addition | `middleware.Gzip()` | Compression |
| Request ID | Separate addition | `middleware.RequestID()` | Request tracking |
| Timeout | Separate addition | `middleware.Timeout()` | Timeout |
| Secure | Separate addition | `middleware.Secure()` | Security headers |
| CSRF | Separate addition | `middleware.CSRF()` | CSRF protection |
| Body Limit | Separate addition | `middleware.BodyLimit()` | Body size limit |

### Table 4: Error Handling Comparison

| Item | Gin | Echo |
|------|-----|------|
| Error response | `c.JSON()` + `return` | `return error` |
| Abort | `c.Abort()` / `c.AbortWithStatusJSON()` | `return echo.NewHTTPError()` |
| Custom error | Free-form with `gin.H{}` | `echo.HTTPError` struct |
| Centralized handler | None (implement via middleware) | `e.HTTPErrorHandler` |
| Error logging | Captured in middleware | Inside `HTTPErrorHandler` |
| Panic recovery | `gin.Recovery()` | `middleware.Recover()` |

### Table 5: Project Structure Comparison

| Scale | Recommended Structure | Framework Choice |
|-------|----------------------|-----------------|
| Small (API count < 10) | Flat structure | net/http (1.22+) |
| Medium (API count 10-50) | Layered | Gin / Echo |
| Large (API count 50+) | Clean Architecture | Gin / Echo + DI |
| Microservices | DDD + gRPC | Gin (REST gateway) + gRPC |

---

## 4. Anti-Patterns

### Anti-Pattern 1: Context Leaking

```go
// BAD: Passing gin.Context to a goroutine
func handler(c *gin.Context) {
    go func() {
        time.Sleep(5 * time.Second)
        c.JSON(200, gin.H{"ok": true}) // Response may have already been sent
    }()
}

// GOOD: Copy the needed values before passing to a goroutine
func handler(c *gin.Context) {
    userID := c.GetString("userID")
    go func() {
        processAsync(userID)
    }()
    c.JSON(200, gin.H{"accepted": true})
}
```

gin.Context is tied to the request lifecycle, and accessing it after the response has been sent can cause undefined behavior or panics. When passing to a goroutine, extract the needed values first and pass them as primitive types.

### Anti-Pattern 2: Inconsistent Error Responses

```go
// BAD: Inconsistent error response formats
c.JSON(400, gin.H{"error": "bad request"})
c.JSON(400, gin.H{"message": "invalid input"})
c.JSON(400, "error occurred")

// GOOD: Unified error response type
type ErrorResponse struct {
    Code    string `json:"code"`
    Message string `json:"message"`
}

func respondError(c *gin.Context, status int, code, msg string) {
    c.JSON(status, ErrorResponse{Code: code, Message: msg})
}
```

API clients need to predict and parse the error response format. Inconsistent formats make client-side implementation complex and become a source of bugs.

### Anti-Pattern 3: Forgetting c.Next() in Middleware

```go
// BAD: Forgetting c.Next() stops the chain
func loggingMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        log.Printf("Request: %s %s", c.Request.Method, c.Request.URL.Path)
        // c.Next() is missing! Subsequent handlers won't be executed
    }
}

// GOOD: Explicitly call c.Next()
func loggingMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        start := time.Now()
        log.Printf("Request: %s %s", c.Request.Method, c.Request.URL.Path)
        c.Next()
        log.Printf("Response: %d (%v)", c.Writer.Status(), time.Since(start))
    }
}
```

### Anti-Pattern 4: Duplicate Validation

```go
// BAD: Hand-writing validation logic in the handler
func createUser(c *gin.Context) {
    var req CreateUserRequest
    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(400, gin.H{"error": err.Error()})
        return
    }
    // Duplicate validation
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

// GOOD: Consolidate validation in binding tags
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
    // Only business logic after validation passes
}
```

### Anti-Pattern 5: Using gin.Default() in Production

```go
// BAD: gin.Default() includes debug-oriented Logger
func main() {
    r := gin.Default()
    r.Run(":8080")
}

// GOOD: Explicitly configure middleware in production
func main() {
    gin.SetMode(gin.ReleaseMode)
    r := gin.New()

    // Structured logging
    r.Use(StructuredLoggerMiddleware(slog.Default()))
    // Panic recovery (custom)
    r.Use(gin.CustomRecoveryWithWriter(nil, func(c *gin.Context, err any) {
        slog.Error("Panic recovered", "error", err)
        c.AbortWithStatusJSON(500, gin.H{
            "code":    "INTERNAL_ERROR",
            "message": "internal server error",
        })
    }))
    // Security
    r.Use(SecurityHeadersMiddleware())
    r.Use(CORSMiddleware(allowedOrigins))
    r.Use(RateLimitMiddleware(100, 200))

    r.Run(":8080")
}
```

### Anti-Pattern 6: Skipping the Service Layer

```go
// BAD: Writing DB access code directly in the handler
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

// GOOD: Go through the service layer
func (h *UserHandler) GetUser(c *gin.Context) {
    id, err := strconv.ParseInt(c.Param("id"), 10, 64)
    if err != nil {
        respondError(c, http.StatusBadRequest, "INVALID_ID", "Invalid ID")
        return
    }

    user, err := h.service.GetUser(c.Request.Context(), id)
    if err != nil {
        if errors.Is(err, ErrNotFound) {
            respondError(c, http.StatusNotFound, "NOT_FOUND", "User not found")
            return
        }
        respondError(c, http.StatusInternalServerError, "INTERNAL_ERROR", "Internal error")
        return
    }
    respondOK(c, user)
}
```

Writing DB access directly in the handler makes testing difficult and prevents reuse of business logic. By introducing a service layer, handlers can focus on HTTP input/output transformation.

---

## 5. FAQ

### Q1: Should I choose Gin or Echo?

There is virtually no performance difference between the two. Gin has a larger ecosystem and more available resources. Echo has a cleaner code design with the distinctive error-returning handler pattern. You can choose based on team preference, but in many cases Go 1.22+'s standard net/http is sufficient.

Decision criteria:
- **Gin**: Want a large community, prefer more resources available (including in Japanese), following existing Gin projects
- **Echo**: Prefer the error return value pattern, want rich built-in middleware, want to use custom context
- **net/http**: Want to minimize external dependencies, Go 1.22+'s new routing is sufficient, want lightweight microservices

### Q2: What is Gin Release mode?

Switching to production mode with `gin.SetMode(gin.ReleaseMode)` suppresses debug logging and slightly improves performance. It can also be set with the environment variable `GIN_MODE=release`. Always use Release mode in production deployments. In Debug mode, unnecessary information such as routing table output will appear in the logs.

```go
// Switch via environment variable
func init() {
    mode := os.Getenv("GIN_MODE")
    if mode == "" {
        mode = gin.DebugMode
    }
    gin.SetMode(mode)
}
```

### Q3: How do you integrate Swagger/OpenAPI?

Use `swaggo/swag` to automatically generate Swagger specs from handler comments. Serve `/swagger/index.html` with `gin-swagger` or `echo-swagger`. It is common practice to run `swag init` in the CI/CD pipeline and include the generated files in version control.

### Q4: How do you use context.Context in Gin/Echo?

In Gin, you can get the standard `context.Context` with `c.Request.Context()`. Always propagate this context to the service and repository layers. In Echo, it can similarly be obtained with `c.Request().Context()`.

```go
// Gin: context propagation
func (h *UserHandler) GetUser(c *gin.Context) {
    ctx := c.Request.Context()
    user, err := h.service.GetUser(ctx, id) // Pass ctx
    // ...
}

// Echo: context propagation
func (h *UserHandler) GetUser(c echo.Context) error {
    ctx := c.Request().Context()
    user, err := h.service.GetUser(ctx, id) // Pass ctx
    // ...
}
```

### Q5: How should versioning be designed?

URL path versioning is the most common approach. Header-based versioning makes implementation complex, so URL path versioning is recommended.

```go
// URL path versioning
v1 := r.Group("/api/v1")
{
    v1.GET("/users", v1ListUsers)
}

v2 := r.Group("/api/v2")
{
    v2.GET("/users", v2ListUsers) // Different response structure
}
```

### Q6: How do you mock the database in tests?

Define interfaces and inject mock implementations during testing. This is naturally achieved with the Clean Architecture repository pattern.

```go
// Mock repository
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

// Usage in tests
func TestGetUser(t *testing.T) {
    mockRepo := &MockUserRepository{
        users: map[int64]*User{1: {ID: 1, Name: "Test"}},
    }
    service := NewUserService(mockRepo, slog.Default())
    handler := NewUserHandler(service)
    // ...
}
```

### Q7: How do you organize a large number of routes?

The best practice is to split routes into separate files and define them as SetupXxxRoutes functions.

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

### Q8: What are the key points for WebSocket implementation in Gin/Echo?

WebSocket connections are registered via Gin/Echo routing and upgraded with `gorilla/websocket`. Use the Hub pattern for connection management, and separate read/write operations into different goroutines. In production, connection health monitoring via ping is essential.

---


## FAQ

### Q1: What is the most important point when learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory, but by actually writing code and verifying its behavior.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this knowledge applied in practice?

The knowledge from this topic is frequently used in day-to-day development work. It becomes particularly important during code reviews and architecture design.

---

## Summary

| Concept | Key Points |
|---------|------------|
| Gin | High performance, large ecosystem. gin.H, binding |
| Echo | Clean design. Error return value pattern |
| Routing | Fast path matching with Radix tree |
| Validation | Based on go-playground/validator |
| Middleware | Can be applied per group |
| JWT Authentication | Token issuance and validation with golang-jwt |
| Swagger | Auto-generation with swaggo |
| Testing | Unit/integration testing with httptest + testify |
| WebSocket | gorilla/websocket + Hub pattern |
| Production Operations | Graceful Shutdown, structured logging, health checks |
| Clean Architecture | Dependency direction: Handler -> Service -> Repository |

---

## Recommended Next Reads

- [02-database.md](./02-database.md) -- Database Connectivity
- [03-grpc.md](./03-grpc.md) -- gRPC
- [04-testing.md](./04-testing.md) -- Testing
- [00-net-http.md](./00-net-http.md) -- Standard net/http

---

## References

1. **Gin Web Framework** -- https://gin-gonic.com/docs/
2. **Echo -- High performance, extensible, minimalist Go web framework** -- https://echo.labstack.com/
3. **swaggo/swag** -- https://github.com/swaggo/swag
4. **go-playground/validator** -- https://github.com/go-playground/validator
5. **gorilla/websocket** -- https://github.com/gorilla/websocket
6. **golang-jwt/jwt** -- https://github.com/golang-jwt/jwt
