# パッケージとモジュール -- Goのコード組織化

> Goはgo.modベースのモジュールシステムでパッケージを管理し、import・internal・バージョニングで堅牢な依存関係を実現する。

---

## この章で学ぶこと

1. **パッケージの仕組み** -- ディレクトリ=パッケージの原則
2. **go.mod / go.sum** -- モジュールシステムの基盤
3. **internal パッケージ** -- 可視性制御とAPI設計
4. **Workspace** -- マルチモジュール開発
5. **依存管理の実践** -- go get, go mod tidy, vendor, プロキシ
6. **プロジェクト設計パターン** -- レイアウト、循環依存回避、レイヤードアーキテクチャ

---

## 1. パッケージの基本

### コード例 1: パッケージ構造

```
myproject/
├── go.mod
├── go.sum
├── main.go
├── internal/
│   ├── config/
│   │   ├── config.go
│   │   └── config_test.go
│   ├── handler/
│   │   ├── user.go
│   │   ├── order.go
│   │   └── middleware.go
│   ├── repository/
│   │   ├── user_repo.go
│   │   └── order_repo.go
│   ├── service/
│   │   ├── user_service.go
│   │   └── order_service.go
│   └── model/
│       ├── user.go
│       └── order.go
├── pkg/
│   ├── validator/
│   │   ├── validator.go
│   │   └── validator_test.go
│   └── httputil/
│       ├── response.go
│       └── middleware.go
├── cmd/
│   ├── server/
│   │   └── main.go
│   └── worker/
│       └── main.go
├── api/
│   ├── proto/
│   │   └── user.proto
│   └── openapi/
│       └── spec.yaml
├── migrations/
│   ├── 001_create_users.up.sql
│   └── 001_create_users.down.sql
├── scripts/
│   └── setup.sh
├── Makefile
├── Dockerfile
└── README.md
```

### コード例 2: パッケージ宣言と基本ルール

```go
// ファイルの先頭に必ずパッケージ宣言が必要
// ディレクトリ名とパッケージ名は一致させる（慣例）
package user

import (
    "context"
    "fmt"
    "time"

    // 標準ライブラリとサードパーティは空行で区切る
    "github.com/google/uuid"

    // プロジェクト内パッケージ
    "github.com/myorg/myproject/internal/model"
)

// 1つのディレクトリに複数のパッケージ宣言は不可
// ただし _test パッケージは例外（ブラックボックステスト用）

// 同一パッケージ内の全ファイルは同じパッケージ名を持つ
// user.go, user_service.go, user_repo.go → 全て package user
```

### コード例 3: go.mod ファイルの詳細

```go
// go.mod はモジュールのルートに配置する
// go mod init で生成

module github.com/myorg/myproject

go 1.22

// 直接依存
require (
    github.com/gin-gonic/gin v1.9.1
    github.com/lib/pq v1.10.9
    go.uber.org/zap v1.27.0
    github.com/stretchr/testify v1.9.0
    github.com/redis/go-redis/v9 v9.5.1
    google.golang.org/grpc v1.63.2
    google.golang.org/protobuf v1.33.0
)

// 間接依存（直接依存が必要とするもの）
require (
    github.com/bytedance/sonic v1.11.3 // indirect
    github.com/gabriel-vasile/mimetype v1.4.3 // indirect
    golang.org/x/net v0.22.0 // indirect
    golang.org/x/sys v0.18.0 // indirect
    golang.org/x/text v0.14.0 // indirect
)

// retract: このバージョンは使用しないでください（自モジュール用）
retract (
    v1.0.0 // 重大なバグあり
    [v0.9.0, v0.9.5] // セキュリティ脆弱性
)
```

### コード例 4: go.sum の構造

```
// go.sum はモジュールのハッシュを記録する
// 各エントリは以下の形式:
// <module> <version> <hash>
// <module> <version>/go.mod <hash>

github.com/gin-gonic/gin v1.9.1 h1:4idEAncQnU5cB7BeOkPtxjfCSye0AAm1R0RVIqFPSHw=
github.com/gin-gonic/gin v1.9.1/go.mod h1:hPrL/0KcuKM0A0CD0A0CD=
github.com/lib/pq v1.10.9 h1:YXG7RB+JIjhP29X+OtkiDnYaXQwpS4JEWq7dtCCRUEw=
github.com/lib/pq v1.10.9/go.mod h1:AlVN5x4E4T544tWzH6hKFbGRn7nbIqu9HhEDnDfBSXo=

// h1: は SHA-256 ハッシュ
// go.mod のハッシュも別途記録される
// これにより依存の改竄を検知
```

### コード例 5: エクスポートルールの詳細

```go
package user

import (
    "encoding/json"
    "fmt"
    "strings"
    "time"
)

// ==========================================
// 大文字始まり = エクスポート（公開）
// 小文字始まり = 非エクスポート（非公開）
// ==========================================

// User は公開型（大文字始まり）
// 他パッケージから user.User としてアクセス可能
type User struct {
    ID        int       `json:"id"`         // 公開フィールド
    Name      string    `json:"name"`       // 公開フィールド
    Email     string    `json:"email"`      // 公開フィールド
    CreatedAt time.Time `json:"created_at"` // 公開フィールド
    age       int       // 非公開フィールド（パッケージ内のみ）
    password  string    // 非公開フィールド
}

// NewUser はコンストラクタ関数（公開）
// Go には class がないため、New + 型名 の命名規則を使う
func NewUser(name, email string) *User {
    return &User{
        Name:      name,
        Email:     email,
        CreatedAt: time.Now(),
        age:       defaultAge(),
        password:  "",
    }
}

// Validate は公開メソッド
func (u *User) Validate() error {
    if u.Name == "" {
        return fmt.Errorf("name is required")
    }
    if !strings.Contains(u.Email, "@") {
        return fmt.Errorf("invalid email: %s", u.Email)
    }
    return nil
}

// SetPassword は公開メソッド（非公開フィールドへのアクセサ）
func (u *User) SetPassword(plain string) error {
    if len(plain) < 8 {
        return fmt.Errorf("password must be at least 8 characters")
    }
    u.password = hashPassword(plain) // 非公開関数を使用
    return nil
}

// String は fmt.Stringer インターフェースの実装
func (u *User) String() string {
    return fmt.Sprintf("User{ID:%d, Name:%s, Email:%s}", u.ID, u.Name, u.Email)
}

// MarshalJSON は json.Marshaler の実装
// 非公開フィールドはシリアライズから除外される
func (u *User) MarshalJSON() ([]byte, error) {
    type Alias User // 無限再帰を防ぐ
    return json.Marshal(&struct {
        *Alias
        Age int `json:"age,omitempty"`
    }{
        Alias: (*Alias)(u),
        Age:   u.age,
    })
}

// defaultAge は非公開関数（小文字始まり）
func defaultAge() int {
    return 0
}

// hashPassword は非公開関数
func hashPassword(plain string) string {
    // 実際には bcrypt 等を使用
    return "hashed_" + plain
}

// UserRole は公開型の定数グループ
type UserRole string

const (
    RoleAdmin  UserRole = "admin"   // 公開定数
    RoleUser   UserRole = "user"    // 公開定数
    RoleGuest  UserRole = "guest"   // 公開定数
    roleSystem UserRole = "system"  // 非公開定数
)

// UserOption は Functional Options パターンで使う公開型
type UserOption func(*User)

// WithAge はオプション関数（公開）
func WithAge(age int) UserOption {
    return func(u *User) {
        u.age = age // 非公開フィールドをオプションで設定
    }
}

// WithPassword はオプション関数（公開）
func WithPassword(password string) UserOption {
    return func(u *User) {
        u.password = hashPassword(password)
    }
}

// NewUserWithOptions は Functional Options を使うコンストラクタ
func NewUserWithOptions(name, email string, opts ...UserOption) *User {
    u := NewUser(name, email)
    for _, opt := range opts {
        opt(u)
    }
    return u
}
```

### コード例 6: internalパッケージの詳細

```go
// ==========================================
// internal パッケージによるアクセス制御
// ==========================================

// プロジェクト構造:
// github.com/myorg/myproject/
// ├── internal/
// │   ├── database/    ← myproject内のみアクセス可
// │   │   └── db.go
// │   ├── auth/        ← myproject内のみアクセス可
// │   │   └── jwt.go
// │   └── middleware/  ← myproject内のみアクセス可
// │       └── cors.go
// ├── cmd/server/
// │   └── main.go      ← internal/* にアクセス可
// └── pkg/client/
//     └── client.go    ← internal/* にアクセス可

// === internal/database/db.go ===
package database

import (
    "context"
    "database/sql"
    "fmt"
    "time"

    _ "github.com/lib/pq" // ブランクインポート（init()のみ実行）
)

// Config は非公開の設定型（internal内でのみ使用）
type Config struct {
    Host     string
    Port     int
    User     string
    Password string
    DBName   string
    SSLMode  string
    MaxConns int
    Timeout  time.Duration
}

// DefaultConfig はデフォルト設定を返す
func DefaultConfig() Config {
    return Config{
        Host:     "localhost",
        Port:     5432,
        SSLMode:  "disable",
        MaxConns: 25,
        Timeout:  30 * time.Second,
    }
}

// DB はデータベース接続のラッパー
type DB struct {
    conn *sql.DB
    cfg  Config
}

// Connect はデータベースに接続する
func Connect(cfg Config) (*DB, error) {
    dsn := fmt.Sprintf(
        "host=%s port=%d user=%s password=%s dbname=%s sslmode=%s",
        cfg.Host, cfg.Port, cfg.User, cfg.Password, cfg.DBName, cfg.SSLMode,
    )

    conn, err := sql.Open("postgres", dsn)
    if err != nil {
        return nil, fmt.Errorf("database open: %w", err)
    }

    conn.SetMaxOpenConns(cfg.MaxConns)
    conn.SetMaxIdleConns(cfg.MaxConns / 2)
    conn.SetConnMaxLifetime(cfg.Timeout)

    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    if err := conn.PingContext(ctx); err != nil {
        conn.Close()
        return nil, fmt.Errorf("database ping: %w", err)
    }

    return &DB{conn: conn, cfg: cfg}, nil
}

// Close はデータベース接続を閉じる
func (db *DB) Close() error {
    return db.conn.Close()
}

// Conn は内部の *sql.DB を返す
func (db *DB) Conn() *sql.DB {
    return db.conn
}
```

```go
// === internal/auth/jwt.go ===
package auth

import (
    "errors"
    "time"

    "github.com/golang-jwt/jwt/v5"
)

var (
    ErrInvalidToken = errors.New("invalid token")
    ErrExpiredToken = errors.New("token expired")
)

// Claims はJWTのクレーム
type Claims struct {
    UserID int    `json:"user_id"`
    Role   string `json:"role"`
    jwt.RegisteredClaims
}

// TokenService はJWTトークンの生成・検証を行う
type TokenService struct {
    secretKey []byte
    issuer    string
    duration  time.Duration
}

// NewTokenService はTokenServiceを生成する
func NewTokenService(secret, issuer string, duration time.Duration) *TokenService {
    return &TokenService{
        secretKey: []byte(secret),
        issuer:    issuer,
        duration:  duration,
    }
}

// Generate はJWTトークンを生成する
func (ts *TokenService) Generate(userID int, role string) (string, error) {
    claims := &Claims{
        UserID: userID,
        Role:   role,
        RegisteredClaims: jwt.RegisteredClaims{
            ExpiresAt: jwt.NewNumericDate(time.Now().Add(ts.duration)),
            IssuedAt:  jwt.NewNumericDate(time.Now()),
            Issuer:    ts.issuer,
        },
    }

    token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
    return token.SignedString(ts.secretKey)
}

// Validate はJWTトークンを検証する
func (ts *TokenService) Validate(tokenString string) (*Claims, error) {
    token, err := jwt.ParseWithClaims(tokenString, &Claims{},
        func(token *jwt.Token) (interface{}, error) {
            return ts.secretKey, nil
        },
    )
    if err != nil {
        if errors.Is(err, jwt.ErrTokenExpired) {
            return nil, ErrExpiredToken
        }
        return nil, ErrInvalidToken
    }

    claims, ok := token.Claims.(*Claims)
    if !ok || !token.Valid {
        return nil, ErrInvalidToken
    }

    return claims, nil
}
```

### コード例 7: ワークスペース (Go 1.18+) の詳細

```go
// ==========================================
// Go Workspace によるマルチモジュール開発
// ==========================================

// プロジェクト構造（モノレポ）:
// mycompany/
// ├── go.work            ← ワークスペース定義
// ├── api/
// │   ├── go.mod         ← module github.com/mycompany/api
// │   ├── server.go
// │   └── handler/
// │       └── user.go
// ├── shared/
// │   ├── go.mod         ← module github.com/mycompany/shared
// │   ├── model/
// │   │   └── user.go
// │   └── util/
// │       └── string.go
// ├── worker/
// │   ├── go.mod         ← module github.com/mycompany/worker
// │   └── processor.go
// └── tools/
//     ├── go.mod         ← module github.com/mycompany/tools
//     └── generator.go

// === go.work ===
// go work init ./api ./shared ./worker ./tools で生成
```

```go
// go.work ファイル
go 1.22

use (
    ./api
    ./shared
    ./worker
    ./tools
)

// replace も go.work 内で指定可能
// 個々の go.mod を変更せずにローカル開発ができる
replace github.com/external/lib => ../local-fork/lib
```

```go
// === api/go.mod ===
module github.com/mycompany/api

go 1.22

require (
    github.com/mycompany/shared v0.0.0
    github.com/gin-gonic/gin v1.9.1
)

// go.work が存在する場合、ローカルの shared を自動参照
// go.work が存在しない場合（CI等）、公開レジストリから取得
```

```go
// === shared/model/user.go ===
package model

import "time"

// User は複数モジュールで共有するモデル
type User struct {
    ID        int       `json:"id" db:"id"`
    Name      string    `json:"name" db:"name"`
    Email     string    `json:"email" db:"email"`
    CreatedAt time.Time `json:"created_at" db:"created_at"`
}

// Validate はユーザーのバリデーション
func (u *User) Validate() error {
    if u.Name == "" {
        return ErrNameRequired
    }
    return nil
}
```

```go
// === api/handler/user.go ===
package handler

import (
    "net/http"

    "github.com/gin-gonic/gin"
    "github.com/mycompany/shared/model" // go.work経由でローカル参照
)

// UserHandler はユーザー関連のHTTPハンドラ
type UserHandler struct {
    // 依存注入
}

func (h *UserHandler) GetUser(c *gin.Context) {
    user := model.User{
        ID:   1,
        Name: "Alice",
    }
    c.JSON(http.StatusOK, user)
}
```

### コード例 8: init() 関数の仕組みと実践

```go
// ==========================================
// init() 関数の詳細
// ==========================================

// init() は特殊な関数:
// - 引数なし、戻り値なし
// - パッケージがインポートされたときに自動実行
// - 1ファイルに複数定義可能（上から順に実行）
// - 直接呼び出し不可

// === driver/postgres.go ===
package driver

import (
    "database/sql"
    "log"
)

// init() はブランクインポート時にドライバを登録する
func init() {
    sql.Register("postgres-custom", &PostgresDriver{})
    log.Println("PostgreSQL custom driver registered")
}

type PostgresDriver struct{}

// ... Driver インターフェースの実装
```

```go
// === main.go ===
package main

import (
    "database/sql"
    "fmt"

    // ブランクインポート: init() のみ実行（名前空間に入れない）
    _ "github.com/myorg/driver"
    // 標準的なドライバ登録パターン
    _ "github.com/lib/pq"
)

func main() {
    // init() によりドライバが登録済み
    db, err := sql.Open("postgres", "host=localhost dbname=test")
    if err != nil {
        panic(err)
    }
    defer db.Close()

    fmt.Println("Connected successfully")
}
```

```go
// === 複数 init() の実行順序 ===
package mypackage

import "fmt"

// 同一ファイル内では上から順に実行
func init() {
    fmt.Println("init 1")
}

func init() {
    fmt.Println("init 2")
}

func init() {
    fmt.Println("init 3")
}

// 出力:
// init 1
// init 2
// init 3
```

```go
// === init() の実行順序（パッケージ間）===
// インポートの依存関係に基づいてトポロジカルソート順に実行
//
// main → A → C
//      → B → C
//
// 実行順序: C の init() → A の init() → B の init() → main の init()
// C は A と B の両方に依存されているため最初に初期化される

// === init() の適切な使用例 ===
package config

import (
    "log"
    "os"
)

var (
    // パッケージレベル変数の初期化
    AppEnv    string
    DebugMode bool
)

func init() {
    // 環境変数からの設定読み込み
    AppEnv = os.Getenv("APP_ENV")
    if AppEnv == "" {
        AppEnv = "development"
    }
    DebugMode = AppEnv == "development"
    log.Printf("Environment: %s, Debug: %v", AppEnv, DebugMode)
}
```

### コード例 9: go getコマンドの使い方

```bash
# モジュールの追加
go get github.com/gin-gonic/gin@latest

# 特定バージョンの指定
go get github.com/gin-gonic/gin@v1.9.1

# コミットハッシュの指定
go get github.com/gin-gonic/gin@abc1234

# ブランチの指定
go get github.com/gin-gonic/gin@main

# バージョンの更新
go get -u github.com/gin-gonic/gin        # マイナー/パッチ更新
go get -u=patch github.com/gin-gonic/gin   # パッチのみ更新

# モジュールの削除（go.modから除去後）
go mod tidy

# 全依存の更新
go get -u ./...

# go install でツールのインストール（Go 1.17+）
go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
go install golang.org/x/tools/cmd/goimports@latest
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
```

### コード例 10: go mod コマンド群

```bash
# モジュールの初期化
go mod init github.com/myorg/myproject

# 不要な依存の削除 + 不足分の追加
go mod tidy

# go.mod / go.sum の検証
go mod verify

# 依存グラフの表示
go mod graph

# 特定パッケージがなぜ依存に含まれるか表示
go mod why github.com/stretchr/testify

# vendor ディレクトリの生成
go mod vendor

# go.mod の編集（スクリプト向け）
go mod edit -require github.com/gin-gonic/gin@v1.9.1
go mod edit -replace github.com/old/pkg=github.com/new/pkg@v1.0.0
go mod edit -retract v1.0.0
go mod edit -go 1.22

# ダウンロードした依存をキャッシュから取得
go mod download

# 依存のダウンロード先の表示
go env GOMODCACHE    # 通常 $GOPATH/pkg/mod
```

### コード例 11: vendorディレクトリの使用

```bash
# vendor ディレクトリの生成
go mod vendor

# vendor を使ったビルド
go build -mod=vendor ./...

# vendor を使ったテスト
go test -mod=vendor ./...

# GOFLAGS で常に vendor を使う設定
export GOFLAGS="-mod=vendor"
```

```go
// vendor 使用時のディレクトリ構造
// myproject/
// ├── go.mod
// ├── go.sum
// ├── vendor/
// │   ├── modules.txt            ← vendorされたモジュール一覧
// │   ├── github.com/
// │   │   └── gin-gonic/
// │   │       └── gin/
// │   │           ├── gin.go
// │   │           └── ...
// │   └── golang.org/
// │       └── x/
// │           └── net/
// │               └── ...
// └── main.go
```

### コード例 12: replace ディレクティブの使い方

```go
module github.com/myorg/myproject

go 1.22

require (
    github.com/myorg/shared v1.2.0
    github.com/external/lib v2.0.0
)

// ユースケース 1: ローカル開発中のモジュール参照
replace github.com/myorg/shared => ../shared

// ユースケース 2: フォークしたモジュールの使用
replace github.com/external/lib => github.com/myorg/lib-fork v2.0.1

// ユースケース 3: 特定バージョンの上書き（セキュリティパッチ）
replace golang.org/x/net v0.17.0 => golang.org/x/net v0.19.0

// 注意: replace はそのモジュールの直接ビルド時のみ有効
// 他のモジュールから依存として取り込まれた場合は無視される
// → go.work の方が推奨される場合が多い
```

### コード例 13: セマンティックバージョニングとモジュールパス

```go
// ==========================================
// Semantic Versioning とインポートパス
// ==========================================

// Go Modules は Semantic Versioning (semver) に従う
// v<major>.<minor>.<patch>
// v1.2.3

// === v0 / v1 の場合 ===
// インポートパスにバージョン番号は含めない
import "github.com/myorg/mylib"       // v0.x.x or v1.x.x

// === v2 以降の場合 ===
// インポートパスにメジャーバージョンを含める（Import Compatibility Rule）
import "github.com/myorg/mylib/v2"    // v2.x.x
import "github.com/myorg/mylib/v3"    // v3.x.x

// go.mod の記述
// module github.com/myorg/mylib/v2
// → v2 はモジュールパスの一部
```

```go
// === メジャーバージョンアップの手順 ===

// 方法1: メジャーブランチ方式
// v1 ブランチ: module github.com/myorg/mylib
// v2 ブランチ: module github.com/myorg/mylib/v2

// 方法2: サブディレクトリ方式
// mylib/
// ├── go.mod           ← module github.com/myorg/mylib (v1)
// ├── v2/
// │   ├── go.mod       ← module github.com/myorg/mylib/v2
// │   └── ...
// └── ...

// === v0 の特別な扱い ===
// v0.x.x はAPI不安定版として扱われる
// v0 → v1 の移行時、破壊的変更が許容される
// v0.x.x のうちは import path に v0 を含めない
```

### コード例 14: プロキシとチェックサムDB

```bash
# Go Module Proxy の設定
# デフォルト: proxy.golang.org (Google運営)
export GOPROXY=https://proxy.golang.org,direct

# プライベートリポジトリの場合
# GOPRIVATE でプロキシをバイパス
export GOPRIVATE=github.com/myorg/*,gitlab.mycompany.com/*

# GONOSUMCHECK でチェックサム検証をスキップ
export GONOSUMCHECK=github.com/myorg/*

# GONOSUMDB でチェックサムDBへの問い合わせをスキップ
export GONOSUMDB=github.com/myorg/*

# 社内プロキシの設定
export GOPROXY=https://goproxy.mycompany.com,https://proxy.golang.org,direct

# チェックサムDB
# デフォルト: sum.golang.org
# 全公開モジュールのハッシュを透明性ログに記録
export GONOSUMDB=github.com/myorg/*
```

```go
// === .netrc によるプライベートリポジトリ認証 ===
// ~/.netrc
// machine github.com
//   login USERNAME
//   password PERSONAL_ACCESS_TOKEN

// === Git設定によるSSH→HTTPS変換 ===
// git config --global url."ssh://git@github.com/".insteadOf "https://github.com/"

// === CI/CD での設定例 ===
// GitHub Actions
// env:
//   GOPRIVATE: github.com/myorg/*
//   GOPROXY: https://proxy.golang.org,direct
// steps:
//   - uses: actions/setup-go@v4
//     with:
//       go-version: '1.22'
//   - run: |
//       git config --global url."https://${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/"
//       go mod download
```

### コード例 15: ビルドタグとプラットフォーム別コード

```go
// ==========================================
// ビルドタグ（Build Constraints）
// ==========================================

// === Go 1.17+ の新構文 ===
//go:build linux && amd64

package mypackage

// === Go 1.16 以前の旧構文 ===
// +build linux,amd64

// === 複数条件の組み合わせ ===
//go:build (linux || darwin) && amd64

//go:build !windows

//go:build integration

// === ファイル名による暗黙のビルドタグ ===
// file_linux.go     → linux のみコンパイル
// file_darwin.go    → macOS のみコンパイル
// file_windows.go   → Windows のみコンパイル
// file_amd64.go     → amd64 のみコンパイル
// file_linux_arm64.go → linux + arm64 のみ

// === テスト用ビルドタグ ===
//go:build integration

package mypackage_test

import "testing"

func TestIntegration(t *testing.T) {
    // go test -tags=integration ./...
    // このタグを指定したときだけ実行される
}
```

```go
// === プラットフォーム別のコード例 ===

// config_unix.go
//go:build !windows

package config

func defaultConfigPath() string {
    return "/etc/myapp/config.yaml"
}

// config_windows.go
//go:build windows

package config

func defaultConfigPath() string {
    return `C:\ProgramData\myapp\config.yaml`
}
```

### コード例 16: テストパッケージの命名規則

```go
// ==========================================
// テストパッケージの2つのスタイル
// ==========================================

// === スタイル1: ホワイトボックステスト ===
// ファイル: user_test.go
// パッケージ名: user （本体と同じ）
package user

import "testing"

func TestValidate(t *testing.T) {
    u := &User{Name: "", Email: ""}
    // 非公開フィールド・メソッドにアクセス可能
    u.age = 25
    u.password = "secret"

    err := u.Validate()
    if err == nil {
        t.Error("expected validation error for empty name")
    }
}

func Test_defaultAge(t *testing.T) {
    // 非公開関数も直接テストできる
    if defaultAge() != 0 {
        t.Error("expected default age to be 0")
    }
}
```

```go
// === スタイル2: ブラックボックステスト ===
// ファイル: user_test.go
// パッケージ名: user_test （_test サフィックス付き）
package user_test

import (
    "testing"

    "github.com/myorg/myproject/internal/model/user"
)

func TestNewUser(t *testing.T) {
    // 公開APIのみ使用（外部利用者と同じ視点）
    u := user.NewUser("Alice", "alice@example.com")

    if u.Name != "Alice" {
        t.Errorf("expected Name=Alice, got %s", u.Name)
    }
    // u.age にはアクセスできない（非公開）
    // u.password にもアクセスできない
}

func ExampleNewUser() {
    u := user.NewUser("Bob", "bob@example.com")
    fmt.Println(u)
    // Output: User{ID:0, Name:Bob, Email:bob@example.com}
}
```

### コード例 17: レイヤードアーキテクチャでのパッケージ設計

```go
// ==========================================
// クリーンアーキテクチャ風のパッケージ構成
// ==========================================

// プロジェクト構造:
// github.com/myorg/orderservice/
// ├── go.mod
// ├── cmd/
// │   └── server/
// │       └── main.go           ← エントリポイント
// ├── internal/
// │   ├── domain/               ← ドメイン層（依存なし）
// │   │   ├── order.go
// │   │   ├── product.go
// │   │   └── repository.go     ← インターフェース定義
// │   ├── usecase/              ← ユースケース層
// │   │   ├── order_service.go
// │   │   └── order_service_test.go
// │   ├── adapter/              ← アダプタ層
// │   │   ├── handler/          ← HTTPハンドラ
// │   │   │   ├── order.go
// │   │   │   └── middleware.go
// │   │   └── repository/      ← リポジトリ実装
// │   │       ├── postgres/
// │   │       │   └── order_repo.go
// │   │       └── redis/
// │   │           └── cache_repo.go
// │   └── infrastructure/      ← インフラ層
// │       ├── database.go
// │       ├── redis.go
// │       └── logger.go
// └── pkg/
//     └── apierror/            ← 共有APIエラー型
//         └── error.go
```

```go
// === internal/domain/order.go ===
package domain

import (
    "errors"
    "time"
)

var (
    ErrOrderNotFound  = errors.New("order not found")
    ErrInvalidAmount  = errors.New("invalid order amount")
    ErrAlreadyShipped = errors.New("order already shipped")
)

// OrderStatus は注文ステータス
type OrderStatus string

const (
    OrderStatusPending   OrderStatus = "pending"
    OrderStatusConfirmed OrderStatus = "confirmed"
    OrderStatusShipped   OrderStatus = "shipped"
    OrderStatusDelivered OrderStatus = "delivered"
    OrderStatusCancelled OrderStatus = "cancelled"
)

// Order はドメインエンティティ
type Order struct {
    ID         string
    CustomerID string
    Items      []OrderItem
    Status     OrderStatus
    TotalPrice float64
    CreatedAt  time.Time
    UpdatedAt  time.Time
}

// OrderItem は注文明細
type OrderItem struct {
    ProductID string
    Quantity  int
    Price     float64
}

// Validate はドメインルールのバリデーション
func (o *Order) Validate() error {
    if len(o.Items) == 0 {
        return errors.New("order must have at least one item")
    }
    if o.TotalPrice <= 0 {
        return ErrInvalidAmount
    }
    return nil
}

// Ship は注文を出荷する（ドメインロジック）
func (o *Order) Ship() error {
    if o.Status != OrderStatusConfirmed {
        return ErrAlreadyShipped
    }
    o.Status = OrderStatusShipped
    o.UpdatedAt = time.Now()
    return nil
}
```

```go
// === internal/domain/repository.go ===
package domain

import "context"

// OrderRepository はリポジトリのインターフェース（ドメイン層で定義）
// 実装はアダプタ層で行う（依存性逆転の原則）
type OrderRepository interface {
    FindByID(ctx context.Context, id string) (*Order, error)
    FindByCustomerID(ctx context.Context, customerID string) ([]*Order, error)
    Save(ctx context.Context, order *Order) error
    Update(ctx context.Context, order *Order) error
    Delete(ctx context.Context, id string) error
}

// OrderCache はキャッシュのインターフェース
type OrderCache interface {
    Get(ctx context.Context, id string) (*Order, error)
    Set(ctx context.Context, order *Order) error
    Invalidate(ctx context.Context, id string) error
}
```

```go
// === internal/usecase/order_service.go ===
package usecase

import (
    "context"
    "fmt"
    "time"

    "github.com/myorg/orderservice/internal/domain"
)

// OrderService はユースケース層のサービス
type OrderService struct {
    repo  domain.OrderRepository
    cache domain.OrderCache
}

// NewOrderService はOrderServiceを生成する
func NewOrderService(repo domain.OrderRepository, cache domain.OrderCache) *OrderService {
    return &OrderService{repo: repo, cache: cache}
}

// CreateOrder は新しい注文を作成する
func (s *OrderService) CreateOrder(ctx context.Context, customerID string, items []domain.OrderItem) (*domain.Order, error) {
    var total float64
    for _, item := range items {
        total += item.Price * float64(item.Quantity)
    }

    order := &domain.Order{
        ID:         generateID(),
        CustomerID: customerID,
        Items:      items,
        Status:     domain.OrderStatusPending,
        TotalPrice: total,
        CreatedAt:  time.Now(),
        UpdatedAt:  time.Now(),
    }

    if err := order.Validate(); err != nil {
        return nil, fmt.Errorf("validation: %w", err)
    }

    if err := s.repo.Save(ctx, order); err != nil {
        return nil, fmt.Errorf("save order: %w", err)
    }

    // キャッシュに保存（エラーは無視）
    _ = s.cache.Set(ctx, order)

    return order, nil
}

// GetOrder は注文を取得する（キャッシュ優先）
func (s *OrderService) GetOrder(ctx context.Context, id string) (*domain.Order, error) {
    // キャッシュから取得を試みる
    if order, err := s.cache.Get(ctx, id); err == nil {
        return order, nil
    }

    // DBから取得
    order, err := s.repo.FindByID(ctx, id)
    if err != nil {
        return nil, fmt.Errorf("find order: %w", err)
    }

    // キャッシュに保存
    _ = s.cache.Set(ctx, order)

    return order, nil
}

// ShipOrder は注文を出荷する
func (s *OrderService) ShipOrder(ctx context.Context, id string) error {
    order, err := s.repo.FindByID(ctx, id)
    if err != nil {
        return fmt.Errorf("find order: %w", err)
    }

    if err := order.Ship(); err != nil {
        return fmt.Errorf("ship order: %w", err)
    }

    if err := s.repo.Update(ctx, order); err != nil {
        return fmt.Errorf("update order: %w", err)
    }

    _ = s.cache.Invalidate(ctx, id)
    return nil
}

func generateID() string {
    return fmt.Sprintf("order_%d", time.Now().UnixNano())
}
```

```go
// === internal/adapter/repository/postgres/order_repo.go ===
package postgres

import (
    "context"
    "database/sql"
    "fmt"

    "github.com/myorg/orderservice/internal/domain"
)

// OrderRepo は PostgreSQL を使った OrderRepository の実装
type OrderRepo struct {
    db *sql.DB
}

// NewOrderRepo は OrderRepo を生成する
func NewOrderRepo(db *sql.DB) *OrderRepo {
    return &OrderRepo{db: db}
}

// FindByID は ID で注文を検索する
func (r *OrderRepo) FindByID(ctx context.Context, id string) (*domain.Order, error) {
    query := `
        SELECT id, customer_id, status, total_price, created_at, updated_at
        FROM orders WHERE id = $1
    `

    order := &domain.Order{}
    err := r.db.QueryRowContext(ctx, query, id).Scan(
        &order.ID, &order.CustomerID, &order.Status,
        &order.TotalPrice, &order.CreatedAt, &order.UpdatedAt,
    )
    if err == sql.ErrNoRows {
        return nil, domain.ErrOrderNotFound
    }
    if err != nil {
        return nil, fmt.Errorf("query order: %w", err)
    }

    items, err := r.findItems(ctx, id)
    if err != nil {
        return nil, err
    }
    order.Items = items

    return order, nil
}

// Save は注文を保存する
func (r *OrderRepo) Save(ctx context.Context, order *domain.Order) error {
    tx, err := r.db.BeginTx(ctx, nil)
    if err != nil {
        return fmt.Errorf("begin tx: %w", err)
    }
    defer tx.Rollback()

    query := `
        INSERT INTO orders (id, customer_id, status, total_price, created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5, $6)
    `
    _, err = tx.ExecContext(ctx, query,
        order.ID, order.CustomerID, order.Status,
        order.TotalPrice, order.CreatedAt, order.UpdatedAt,
    )
    if err != nil {
        return fmt.Errorf("insert order: %w", err)
    }

    for _, item := range order.Items {
        itemQuery := `
            INSERT INTO order_items (order_id, product_id, quantity, price)
            VALUES ($1, $2, $3, $4)
        `
        _, err = tx.ExecContext(ctx, itemQuery,
            order.ID, item.ProductID, item.Quantity, item.Price,
        )
        if err != nil {
            return fmt.Errorf("insert item: %w", err)
        }
    }

    return tx.Commit()
}

// findItems は注文明細を取得する（内部関数）
func (r *OrderRepo) findItems(ctx context.Context, orderID string) ([]domain.OrderItem, error) {
    query := `SELECT product_id, quantity, price FROM order_items WHERE order_id = $1`
    rows, err := r.db.QueryContext(ctx, query, orderID)
    if err != nil {
        return nil, fmt.Errorf("query items: %w", err)
    }
    defer rows.Close()

    var items []domain.OrderItem
    for rows.Next() {
        var item domain.OrderItem
        if err := rows.Scan(&item.ProductID, &item.Quantity, &item.Price); err != nil {
            return nil, fmt.Errorf("scan item: %w", err)
        }
        items = append(items, item)
    }
    return items, rows.Err()
}

// FindByCustomerID, Update, Delete も同様に実装...
func (r *OrderRepo) FindByCustomerID(ctx context.Context, customerID string) ([]*domain.Order, error) {
    return nil, nil // 省略
}
func (r *OrderRepo) Update(ctx context.Context, order *domain.Order) error {
    return nil // 省略
}
func (r *OrderRepo) Delete(ctx context.Context, id string) error {
    return nil // 省略
}
```

```go
// === cmd/server/main.go ===
package main

import (
    "database/sql"
    "log"
    "net/http"

    "github.com/myorg/orderservice/internal/adapter/handler"
    "github.com/myorg/orderservice/internal/adapter/repository/postgres"
    "github.com/myorg/orderservice/internal/adapter/repository/redis"
    "github.com/myorg/orderservice/internal/usecase"
)

func main() {
    // インフラ層の初期化
    db, err := sql.Open("postgres", "host=localhost dbname=orders")
    if err != nil {
        log.Fatal(err)
    }
    defer db.Close()

    redisClient := redis.NewClient("localhost:6379")

    // 依存性注入（手動DI）
    orderRepo := postgres.NewOrderRepo(db)
    orderCache := redis.NewOrderCache(redisClient)
    orderService := usecase.NewOrderService(orderRepo, orderCache)
    orderHandler := handler.NewOrderHandler(orderService)

    // ルーティング
    mux := http.NewServeMux()
    mux.HandleFunc("GET /orders/{id}", orderHandler.GetOrder)
    mux.HandleFunc("POST /orders", orderHandler.CreateOrder)
    mux.HandleFunc("POST /orders/{id}/ship", orderHandler.ShipOrder)

    log.Println("Server starting on :8080")
    log.Fatal(http.ListenAndServe(":8080", mux))
}
```

### コード例 18: go generate の活用

```go
// ==========================================
// go generate でコード自動生成
// ==========================================

// === interface から mock を生成 ===
//go:generate mockgen -source=repository.go -destination=mock_repository.go -package=domain

// === stringer で定数の String() を生成 ===
//go:generate stringer -type=OrderStatus

// === protocol buffers から Go コードを生成 ===
//go:generate protoc --go_out=. --go-grpc_out=. proto/order.proto

// === embed でファイルを埋め込む ===
package templates

import "embed"

//go:embed templates/*.html
var TemplateFS embed.FS

//go:embed version.txt
var Version string

//go:embed config/default.json
var DefaultConfig []byte
```

```bash
# go generate の実行
go generate ./...

# 特定パッケージのみ
go generate ./internal/domain/...

# -run でフィルタ
go generate -run "stringer" ./...

# -v で詳細出力
go generate -v ./...
```

### コード例 19: ツールの依存管理（tools.go パターン）

```go
// ==========================================
// tools.go パターン
// ==========================================

// tools.go ファイルでビルドツールの依存を管理する
// ビルドタグで通常ビルドから除外し、go.mod には記録する

//go:build tools

package tools

import (
    // コード生成ツール
    _ "github.com/golang/mock/mockgen"
    _ "golang.org/x/tools/cmd/stringer"
    _ "google.golang.org/protobuf/cmd/protoc-gen-go"
    _ "google.golang.org/grpc/cmd/protoc-gen-go-grpc"

    // 静的解析ツール
    _ "github.com/golangci/golangci-lint/cmd/golangci-lint"
    _ "golang.org/x/vuln/cmd/govulncheck"

    // マイグレーションツール
    _ "github.com/golang-migrate/migrate/v4/cmd/migrate"
)

// これにより:
// 1. go mod tidy でツールの依存が go.mod に記録される
// 2. 通常ビルドではコンパイルされない
// 3. CI/CD で go install ./tools/... で一括インストール可能
```

### コード例 20: Makefile によるビルド管理

```makefile
# ==========================================
# Makefile: Go プロジェクトの標準タスク
# ==========================================

.PHONY: all build test lint clean run generate vendor

# 変数
APP_NAME := myproject
VERSION := $(shell git describe --tags --always --dirty)
BUILD_TIME := $(shell date -u +"%Y-%m-%dT%H:%M:%SZ")
LDFLAGS := -ldflags "-X main.version=$(VERSION) -X main.buildTime=$(BUILD_TIME)"
GOFLAGS := -trimpath

# デフォルトターゲット
all: lint test build

# ビルド
build:
	go build $(GOFLAGS) $(LDFLAGS) -o bin/$(APP_NAME) ./cmd/server/

# 複数バイナリのビルド
build-all:
	go build $(GOFLAGS) $(LDFLAGS) -o bin/server ./cmd/server/
	go build $(GOFLAGS) $(LDFLAGS) -o bin/worker ./cmd/worker/
	go build $(GOFLAGS) $(LDFLAGS) -o bin/cli ./cmd/cli/

# テスト
test:
	go test -race -coverprofile=coverage.out ./...

# カバレッジレポート
coverage: test
	go tool cover -html=coverage.out -o coverage.html
	open coverage.html

# インテグレーションテスト
test-integration:
	go test -race -tags=integration -count=1 ./...

# 静的解析
lint:
	golangci-lint run ./...
	go vet ./...

# 依存管理
tidy:
	go mod tidy
	go mod verify

# vendor
vendor:
	go mod vendor

# コード生成
generate:
	go generate ./...

# 実行
run:
	go run ./cmd/server/

# クリーン
clean:
	rm -rf bin/ coverage.out coverage.html

# Docker
docker-build:
	docker build -t $(APP_NAME):$(VERSION) .

# クロスコンパイル
build-linux:
	GOOS=linux GOARCH=amd64 go build $(GOFLAGS) $(LDFLAGS) -o bin/$(APP_NAME)-linux-amd64 ./cmd/server/

build-darwin:
	GOOS=darwin GOARCH=arm64 go build $(GOFLAGS) $(LDFLAGS) -o bin/$(APP_NAME)-darwin-arm64 ./cmd/server/

# 脆弱性チェック
vuln:
	govulncheck ./...
```

---

## 2. ASCII図解

### 図1: モジュールとパッケージの関係

```
┌─── module: github.com/myorg/myproject ─────────────────────┐
│                                                              │
│  go.mod                                                      │
│  ┌──────────────────────────────────────────┐               │
│  │ module github.com/myorg/myproject        │               │
│  │ go 1.22                                   │               │
│  │ require (                                 │               │
│  │   github.com/gin-gonic/gin v1.9.1        │               │
│  │   github.com/lib/pq v1.10.9              │               │
│  │ )                                         │               │
│  └──────────────────────────────────────────┘               │
│                                                              │
│  ┌─ package: main ─────┐  ┌─ package: config ──────────┐   │
│  │ cmd/server/          │  │ internal/config/            │   │
│  │   main.go            │  │   config.go                 │   │
│  │   ← エントリーポイント │  │   config_test.go           │   │
│  └─────────────────────┘  │   ← モジュール外からアクセス不可│   │
│                            └────────────────────────────┘   │
│  ┌─ package: handler ──┐  ┌─ package: model ────────────┐  │
│  │ internal/handler/    │  │ pkg/model/                   │  │
│  │   user.go            │  │   user.go                    │  │
│  │   order.go           │  │   order.go                   │  │
│  │   middleware.go      │  │   ← モジュール外からもアクセス可│  │
│  └─────────────────────┘  └─────────────────────────────┘  │
│                                                              │
│  ┌─ package: service ──┐  ┌─ package: repository ───────┐  │
│  │ internal/service/    │  │ internal/repository/         │  │
│  │   user_service.go    │  │   user_repo.go               │  │
│  │   order_service.go   │  │   order_repo.go              │  │
│  └─────────────────────┘  └─────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### 図2: go mod のバージョン解決（MVS詳細）

```
Minimum Version Selection (MVS) アルゴリズム:

myproject
├── require A v1.2.0
│   ├── require C v1.0.0
│   └── require D v2.1.0
├── require B v1.3.0
│   ├── require C v1.1.0   ← C の最新要求
│   └── require D v2.0.0   ← D は A の要求 v2.1.0 が優先
│
│  各モジュールの要求バージョン:
│  C: v1.0.0 (from A), v1.1.0 (from B)
│     → 選択: v1.1.0 (最小の互換バージョン = 要求の最大)
│  D: v2.1.0 (from A), v2.0.0 (from B)
│     → 選択: v2.1.0 (要求の最大)
│
│  MVSの特徴:
│  - 「要求された最小のバージョン」を選択（最新ではない）
│  - 再現性が高い（他のモジュールの公開に影響されない）
│  - npm/pip の SAT solver とは異なるアプローチ
│
└── 最終的なビルドリスト:
    A v1.2.0, B v1.3.0, C v1.1.0, D v2.1.0

比較: npm (semver解決)
  C: ^1.0.0, ^1.1.0 → v1.9.9 (最新互換)
  → 新バージョンの公開で結果が変わりうる

比較: Go MVS
  C: v1.0.0, v1.1.0 → v1.1.0 (要求の最大、固定)
  → 新バージョンが公開されても結果は変わらない
```

### 図3: import解決フロー（詳細版）

```
import "github.com/myorg/pkg/util"
            │
            ▼
    ┌───────────────────┐
    │ 標準ライブラリか？   │──YES──> $GOROOT/src から取得
    └───────┬───────────┘           (fmt, net/http, etc.)
            │NO
            ▼
    ┌───────────────────┐
    │ go.work に含まれる？ │──YES──> ローカルディレクトリ参照
    └───────┬───────────┘           (マルチモジュール開発)
            │NO
            ▼
    ┌───────────────────┐
    │ replace で指定？    │──YES──> 指定先から取得
    └───────┬───────────┘           (ローカル or リモート)
            │NO
            ▼
    ┌───────────────────┐
    │ vendor/ に存在？    │──YES──> vendor/ から取得
    └───────┬───────────┘           (-mod=vendor 時)
            │NO
            ▼
    ┌───────────────────┐
    │ GOMODCACHE に存在？ │──YES──> キャッシュから取得
    └───────┬───────────┘           ($GOPATH/pkg/mod)
            │NO
            ▼
    ┌───────────────────┐
    │ GOPRIVATE に該当？  │──YES──> 直接リポジトリから取得
    └───────┬───────────┘           (git clone等)
            │NO
            ▼
    ┌───────────────────┐
    │ GOPROXY 経由で取得  │
    │ proxy.golang.org   │
    └───────┬───────────┘
            │
            ▼
    ┌───────────────────┐
    │ sum.golang.org で   │──FAIL──> ビルドエラー
    │ チェックサム検証     │           (改竄検知)
    └───────┬───────────┘
            │OK
            ▼
    ┌───────────────────┐
    │ GOMODCACHE に保存   │
    │ go.sum に記録       │
    └───────────────────┘
```

### 図4: パッケージ初期化順序

```
init() の実行順序:

依存グラフ:
  main ──> pkg/handler ──> pkg/model
       ──> pkg/service ──> pkg/model
       ──> pkg/repository ──> pkg/model
                          ──> database/sql

実行順序（トポロジカルソート）:
  1. 標準ライブラリ (database/sql, fmt, etc.) の init()
  2. pkg/model の init()        ← 最も依存される
  3. pkg/handler の init()
  4. pkg/service の init()
  5. pkg/repository の init()
  6. main の init()             ← 最後
  7. main()                     ← init() の後

同一パッケージ内の順序:
  ┌─ a.go ──┐  ┌─ b.go ──┐  ┌─ c.go ──┐
  │ init()  │  │ init()  │  │ init()  │
  │ init()  │  │         │  │ init()  │
  └────┬────┘  └────┬────┘  └────┬────┘
       │            │            │
       ▼            ▼            ▼
  ファイル名アルファベット順 (a → b → c)
  各ファイル内は上から下
```

### 図5: internal パッケージのアクセス制御

```
internal の可視性ルール:

github.com/myorg/myproject/
├── internal/
│   ├── config/     ← (A)
│   └── handler/    ← (B)
├── pkg/
│   └── client/     ← (C)
├── cmd/
│   └── server/     ← (D)
└── main.go         ← (E)

アクセス可能:
  (D) → (A) ✓  cmd/server → internal/config
  (D) → (B) ✓  cmd/server → internal/handler
  (E) → (A) ✓  main.go → internal/config
  (C) → (A) ✓  pkg/client → internal/config (同一モジュール内)

アクセス不可:
  外部モジュール → (A) ✗  コンパイルエラー
  外部モジュール → (B) ✗  コンパイルエラー

ネストされた internal:
github.com/myorg/myproject/
├── pkg/
│   └── server/
│       ├── internal/      ← サブ internal
│       │   └── parser/    ← (F)
│       └── server.go      ← (G)
│
│  (G) → (F) ✓  server.go → server/internal/parser
│  (E) → (F) ✗  main.go → server/internal/parser ← 不可！
│  server パッケージの外からはアクセスできない
```

### 図6: go.work によるマルチモジュール開発

```
モノレポ構成:
┌─────────────────────────────────────────────────┐
│  mycompany/                                      │
│  ├── go.work                                     │
│  │   use (./api, ./shared, ./worker)             │
│  │                                               │
│  ├── api/                                        │
│  │   ├── go.mod (module github.com/mycompany/api)│
│  │   └── require github.com/mycompany/shared     │
│  │                    │                          │
│  │                    │ go.work により            │
│  │                    │ ローカル参照 ──────┐      │
│  │                    │                   │      │
│  ├── shared/          ▼                   │      │
│  │   ├── go.mod (module github.com/mycompany/shared)│
│  │   └── model/user.go                          │
│  │                                               │
│  └── worker/                                     │
│      ├── go.mod (module github.com/mycompany/worker)│
│      └── require github.com/mycompany/shared     │
│                       │                          │
│                       └── go.work によりローカル参照│
│                                                  │
│  go.work がない場合:                              │
│  → 各 go.mod の require に従いレジストリから取得   │
│                                                  │
│  go.work がある場合:                              │
│  → ローカルの shared/ を直接参照                   │
│  → go.mod の require は CI/CD 用に維持            │
└─────────────────────────────────────────────────┘

CI/CD では go.work を使わない:
  .gitignore に go.work を含める（推奨）
  または go.work を CI で無効化:
    GOWORK=off go build ./...
```

---

## 3. 比較表

### 表1: パッケージ管理の進化

| 世代 | 方式 | 期間 | 特徴 | 課題 |
|------|------|------|------|------|
| 第1世代 | GOPATH | 2009-2018 | グローバルワークスペース | バージョン管理なし、プロジェクト分離困難 |
| 過渡期 | dep / glide | 2016-2019 | vendorディレクトリ、lock file | ツール乱立、標準化されていない |
| 第2世代 | Go Modules | 2019- | go.mod、MVS、プロキシ対応 | 学習コスト（replace, retract等） |
| 拡張 | Workspace | 2022- | go.work、マルチモジュール開発 | CI/CDとの使い分けが必要 |

### 表2: ディレクトリ規約比較（詳細版）

| ディレクトリ | 用途 | 公開範囲 | 必須？ | 備考 |
|-------------|------|---------|--------|------|
| `cmd/` | エントリーポイント (main パッケージ) | -- | 推奨 | 複数バイナリの場合に必須 |
| `pkg/` | 外部から利用可能なライブラリ | モジュール外から利用可 | 任意 | 小規模なら不要 |
| `internal/` | 内部実装 | モジュール内のみ | 強く推奨 | コンパイラが強制 |
| `vendor/` | 依存のローカルコピー | -- | 任意 | オフラインビルド用 |
| `api/` | API定義 (proto, OpenAPI) | -- | 任意 | gRPC/REST定義 |
| `docs/` | ドキュメント | -- | 任意 | -- |
| `scripts/` | ビルド・デプロイスクリプト | -- | 任意 | -- |
| `migrations/` | DBマイグレーション | -- | 任意 | -- |
| `testdata/` | テスト用データ | -- | 任意 | go tool に無視される |
| `tools/` | ビルドツールの依存 | -- | 任意 | tools.go パターン |
| `examples/` | 使用例 | -- | 任意 | ライブラリで推奨 |

### 表3: Go Modules vs 他言語の依存管理

| 項目 | Go Modules | npm (Node.js) | pip (Python) | Cargo (Rust) | Maven (Java) |
|------|-----------|---------------|-------------|-------------|-------------|
| 設定ファイル | go.mod | package.json | requirements.txt / pyproject.toml | Cargo.toml | pom.xml |
| ロックファイル | go.sum | package-lock.json | -- (pipenv: Pipfile.lock) | Cargo.lock | -- |
| バージョン解決 | MVS (最小) | SAT solver (最新互換) | 最新互換 | SAT solver | nearest-wins |
| 再現性 | 非常に高い | 高い (lock有) | 低い (lockなし) | 高い | 中 |
| セントラルレジストリ | proxy.golang.org | npmjs.com | pypi.org | crates.io | Maven Central |
| チェックサム検証 | go.sum + sum.golang.org | npm audit | -- | -- | チェックサム |
| メジャーバージョン | パス変更必須 | 同一パッケージ | 同一パッケージ | 同一パッケージ | GAV座標 |
| vendor | go mod vendor | node_modules | -- | cargo vendor | -- |
| ワークスペース | go.work | npm workspaces | -- | cargo workspaces | reactor |

### 表4: エクスポートルールの詳細

| 種別 | 大文字始まり | 小文字始まり | 例 |
|------|------------|------------|-----|
| 型 | エクスポート | 非エクスポート | `User` vs `user` |
| 関数 | エクスポート | 非エクスポート | `NewUser()` vs `newUser()` |
| メソッド | エクスポート | 非エクスポート | `u.Validate()` vs `u.validate()` |
| フィールド | エクスポート | 非エクスポート | `u.Name` vs `u.name` |
| 定数 | エクスポート | 非エクスポート | `MaxRetries` vs `maxRetries` |
| 変数 | エクスポート | 非エクスポート | `DefaultTimeout` vs `defaultTimeout` |
| インターフェース | エクスポート | 非エクスポート | `Reader` vs `reader` |
| 埋め込みフィールド | 型名に従う | 型名に従う | `User` (公開) vs `user` (非公開) |

### 表5: replace vs go.work の使い分け

| 項目 | replace (go.mod) | go.work |
|------|-----------------|---------|
| スコープ | 直接ビルド時のみ | ワークスペース内全モジュール |
| 推移性 | なし（他モジュールに伝播しない） | なし |
| バージョン管理 | コミットすべき（一時的を除く） | .gitignore 推奨 |
| CI/CD | そのまま有効 | GOWORK=off で無効化 |
| 用途 | フォーク、一時パッチ | マルチモジュール開発 |
| 複数モジュール | 各 go.mod に個別設定 | 1箇所で一括設定 |
| 推奨度 | 限定的（最小限に） | マルチモジュールなら推奨 |

---

## 4. アンチパターン

### アンチパターン 1: 循環インポート

```go
// BAD: パッケージ A が B を、B が A をインポート
// ── package a/a.go ──
package a

import "myproject/b" // コンパイルエラー: import cycle

func ProcessA() {
    b.ProcessB()
}

type ResultA struct {
    Value string
}

// ── package b/b.go ──
package b

import "myproject/a" // 循環！

func ProcessB() {
    a.ProcessA()
}

func FormatResult(r a.ResultA) string {
    return r.Value
}

// GOOD: インターフェースで依存を逆転（DIP）
// ── package a/a.go ──
package a

// Processor はパッケージ a が定義するインターフェース
type Processor interface {
    Process() error
}

type ServiceA struct {
    processor Processor // b.ProcessorImpl を受け取る
}

func NewServiceA(p Processor) *ServiceA {
    return &ServiceA{processor: p}
}

func (s *ServiceA) Run() error {
    return s.processor.Process()
}

// ── package b/b.go ──
package b

// b は a をインポートしない
// a.Processor インターフェースを暗黙的に実装
type ProcessorImpl struct{}

func (p *ProcessorImpl) Process() error {
    // 処理
    return nil
}

// ── package main ──
package main

import (
    "myproject/a"
    "myproject/b"
)

func main() {
    processor := &b.ProcessorImpl{}
    service := a.NewServiceA(processor) // 依存の注入
    service.Run()
}
```

### アンチパターン 2: 巨大なutilパッケージ

```go
// BAD: 何でも入れるutilパッケージ
package util

import (
    "crypto/rand"
    "encoding/csv"
    "fmt"
    "io"
    "net/smtp"
    "time"
)

func FormatDate(t time.Time) string { return t.Format("2006-01-02") }
func HashPassword(p string) string { /* ... */ return "" }
func ParseCSV(r io.Reader) ([][]string, error) {
    return csv.NewReader(r).ReadAll()
}
func SendEmail(to, subject, body string) error { /* ... */ return nil }
func GenerateID() string {
    b := make([]byte, 16)
    rand.Read(b)
    return fmt.Sprintf("%x", b)
}

// 問題点:
// 1. 凝集度が低い（無関係な機能が混在）
// 2. テストが困難（依存が多すぎる）
// 3. 名前衝突のリスク
// 4. import cycle を引き起こしやすい
// 5. 「util に入れておけばいい」という悪い慣習を生む

// GOOD: 責務ごとにパッケージを分割
// ── package timeutil ──
package timeutil

import "time"

func FormatDate(t time.Time) string {
    return t.Format("2006-01-02")
}

func FormatDateTime(t time.Time) string {
    return t.Format("2006-01-02T15:04:05Z07:00")
}

// ── package auth ──
package auth

import "golang.org/x/crypto/bcrypt"

func HashPassword(plain string) (string, error) {
    hash, err := bcrypt.GenerateFromPassword([]byte(plain), bcrypt.DefaultCost)
    return string(hash), err
}

func VerifyPassword(hash, plain string) bool {
    return bcrypt.CompareHashAndPassword([]byte(hash), []byte(plain)) == nil
}

// ── package csvutil ──
package csvutil

import (
    "encoding/csv"
    "io"
)

func Parse(r io.Reader) ([][]string, error) {
    return csv.NewReader(r).ReadAll()
}
```

### アンチパターン 3: init() の乱用

```go
// BAD: init() で重い処理を行う
package database

import (
    "database/sql"
    "log"
    "os"
)

var db *sql.DB

func init() {
    var err error
    // テスト時にも実行されてしまう！
    // 環境変数がないとパニック！
    db, err = sql.Open("postgres", os.Getenv("DATABASE_URL"))
    if err != nil {
        log.Fatal(err) // テストが落ちる
    }
    if err = db.Ping(); err != nil {
        log.Fatal(err) // ネットワークエラーで即死
    }
}

func GetDB() *sql.DB {
    return db
}

// 問題点:
// 1. テスト時にもDB接続が必要になる
// 2. エラー時に log.Fatal でプロセス終了
// 3. 初期化順序の制御が困難
// 4. モックに差し替えられない

// GOOD: 明示的な初期化関数を使う
package database

import (
    "context"
    "database/sql"
    "fmt"
    "time"
)

type DB struct {
    conn *sql.DB
}

// Connect は明示的にDB接続を作成する
func Connect(ctx context.Context, dsn string) (*DB, error) {
    conn, err := sql.Open("postgres", dsn)
    if err != nil {
        return nil, fmt.Errorf("open database: %w", err)
    }

    ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
    defer cancel()

    if err := conn.PingContext(ctx); err != nil {
        conn.Close()
        return nil, fmt.Errorf("ping database: %w", err)
    }

    return &DB{conn: conn}, nil
}

// Close は接続を閉じる
func (db *DB) Close() error {
    return db.conn.Close()
}

// テスト時はモックを注入できる
// main() で明示的に Connect() を呼ぶ
```

### アンチパターン 4: パッケージの粒度が細かすぎる

```go
// BAD: 1ファイル1パッケージ（Java的な構成）
// myproject/
// ├── user/
// │   ├── model/
// │   │   └── user.go          ← package model
// │   ├── repository/
// │   │   └── user_repo.go     ← package repository
// │   ├── service/
// │   │   └── user_service.go  ← package service
// │   └── handler/
// │       └── user_handler.go  ← package handler

// 問題: パッケージ間のやりとりが複雑、型の変換が頻発

// GOOD: Goらしいフラットな構成
// myproject/
// ├── user/
// │   ├── user.go              ← type User, type Repository interface
// │   ├── service.go           ← type Service struct
// │   ├── handler.go           ← type Handler struct
// │   └── postgres.go          ← type PostgresRepo struct
//
// 全て package user
// パッケージ内で型を共有でき、変換不要

package user

// 同一パッケージ内なので User 型を直接使える
type User struct {
    ID    int
    Name  string
    Email string
}

type Repository interface {
    FindByID(ctx context.Context, id int) (*User, error)
    Save(ctx context.Context, user *User) error
}

type Service struct {
    repo Repository
}

func NewService(repo Repository) *Service {
    return &Service{repo: repo}
}

func (s *Service) GetUser(ctx context.Context, id int) (*User, error) {
    return s.repo.FindByID(ctx, id) // 型変換不要
}
```

### アンチパターン 5: go.sum をコミットしない

```bash
# BAD: .gitignore に go.sum を含める
# .gitignore
# go.sum  ← これは間違い！

# go.sum をコミットしない問題点:
# 1. サプライチェーン攻撃を検知できない
# 2. ビルドの再現性が失われる
# 3. CI/CD で異なるハッシュのモジュールが使われる可能性

# GOOD: go.mod と go.sum の両方をコミットする
git add go.mod go.sum
git commit -m "Update dependencies"

# go.work は .gitignore に含めてもよい（ローカル開発用）
# .gitignore
# go.work
# go.work.sum
```

### アンチパターン 6: 不適切な replace の使用

```go
// BAD: replace をコミットしたまま放置
module github.com/myorg/myproject

go 1.22

require (
    github.com/myorg/shared v1.2.0
)

// ローカル開発用の replace がそのまま…
replace github.com/myorg/shared => ../shared
// CI/CD で ../shared が存在しないためビルド失敗

// GOOD: go.work を使い、replace は一時的に限定
// go.work (gitignore対象)
go 1.22
use (
    .
    ../shared
)

// go.mod はクリーンに保つ
module github.com/myorg/myproject

go 1.22

require (
    github.com/myorg/shared v1.2.0
)
// replace なし
```

---

## 5. FAQ

### Q1: pkg/ ディレクトリは必須か？

必須ではない。Go公式は特定のレイアウトを推奨していない。`pkg/` は「外部に公開するパッケージ」を示す慣習だが、小規模プロジェクトでは不要。重要なのは `internal/` で内部実装を保護すること。Go の標準ライブラリ自体も `pkg/` ディレクトリを使っていない。

大規模プロジェクトでの判断基準:
- 他のプロジェクトから直接インポートされるライブラリ → `pkg/` を使う
- マイクロサービス等の単独デプロイ → `pkg/` は不要、`internal/` で十分
- モノレポ内の共有コード → `shared/` や `common/` も選択肢

### Q2: go.sum は何のために存在するか？

`go.sum` は依存パッケージのチェックサム（SHA-256ハッシュ）を記録する。これによりサプライチェーン攻撃を検知し、ビルドの再現性を保証する。バージョン管理にコミットすべきファイル。

具体的な保護メカニズム:
1. 初回ダウンロード時にハッシュを記録
2. 以後のビルドでハッシュを検証
3. `sum.golang.org`（透明性ログ）と照合
4. ハッシュ不一致の場合はビルドを中止

`go mod verify` で手動検証も可能。

### Q3: replace ディレクティブはいつ使うか？

`replace` の正当な用途:
1. ローカル開発中のモジュール参照（一時的）
2. フォークしたモジュールの利用（upstream に PR を出すまでの間）
3. セキュリティパッチの一時適用
4. 非公開モジュールの参照先変更

注意点:
- replace は直接ビルド時のみ有効（依存先の replace は無視される）
- マルチモジュール開発では go.work の方が推奨
- 本番コードでは極力避け、一時的な措置に限定する

### Q4: vendor ディレクトリを使うべきか？

以下の場合に vendor を検討する:
1. **オフラインビルド** -- ネットワークアクセスなしでビルドしたい場合
2. **CI/CD の安定性** -- プロキシダウン時でもビルド可能にしたい場合
3. **監査要件** -- 依存のソースコードをリポジトリに含める必要がある場合
4. **レガシーツール** -- GOPATH 時代のツールとの互換性が必要な場合

vendor を使わない場合:
- `go mod download` でキャッシュを活用
- CI/CD では `actions/cache` 等でモジュールキャッシュを永続化
- `GOPROXY` で社内プロキシを運用

### Q5: テストファイルの命名規則は？

Goのテストファイル命名規則:
- `*_test.go` -- テストファイル（go build で除外、go test で含まれる）
- `package foo` -- ホワイトボックステスト（非公開にアクセス可）
- `package foo_test` -- ブラックボックステスト（公開APIのみ）
- `Example*` -- ドキュメント用の実行例
- `Benchmark*` -- ベンチマークテスト
- `Fuzz*` -- ファジングテスト（Go 1.18+）
- `testdata/` -- テスト用データディレクトリ（go tool に無視される）

### Q6: 標準ライブラリとサードパーティのインポートはどう整理するか？

`goimports` ツールが自動整理する推奨フォーマット:

```go
import (
    // グループ1: 標準ライブラリ
    "context"
    "fmt"
    "net/http"

    // グループ2: サードパーティ
    "github.com/gin-gonic/gin"
    "go.uber.org/zap"

    // グループ3: プロジェクト内パッケージ
    "github.com/myorg/myproject/internal/config"
    "github.com/myorg/myproject/internal/handler"
)
```

`.golangci.yml` の `goimports` セクションでプロジェクトプレフィックスを設定すると、3グループに自動分割される:
```yaml
linters-settings:
  goimports:
    local-prefixes: github.com/myorg/myproject
```

### Q7: モジュールのメジャーバージョンアップはどう行うか？

Go Modules では v2 以降のメジャーバージョンはインポートパスが変わる（Import Compatibility Rule）:

手順:
1. `go.mod` の module パスに `/v2` を追加
2. 全 import パスを更新
3. 破壊的変更を加える
4. `v2.0.0` タグを打つ

```bash
# 方法1: メジャーブランチ
git checkout -b v2
# go.mod: module github.com/myorg/mylib/v2
# 全ファイルの import パスを更新
# git tag v2.0.0

# 方法2: サブディレクトリ
mkdir v2
cp -r *.go v2/
# v2/go.mod: module github.com/myorg/mylib/v2
# git tag v2.0.0
```

利用者側:
```go
// v1 と v2 を同時に使うことも可能
import (
    v1user "github.com/myorg/mylib/user"
    v2user "github.com/myorg/mylib/v2/user"
)
```

### Q8: retract ディレクティブは何に使うか？

`retract` はモジュールの特定バージョンを「撤回」するために使う。撤回されたバージョンは `go get` のデフォルト選択から除外されるが、明示的に指定すれば使用可能:

```go
module github.com/myorg/mylib

go 1.22

// 個別バージョンの撤回
retract v1.0.0 // 重大なバグ: データ破損の可能性

// バージョン範囲の撤回
retract [v0.9.0, v0.9.5] // セキュリティ脆弱性 CVE-2024-XXXX

// 誤って公開したバージョン
retract v2.0.0-beta.1 // 不完全なリリース
```

注意: retract が有効になるのは、retract を含むバージョンが公開された後。つまり v1.0.0 を撤回するには、retract を含む v1.0.1 をリリースする必要がある。

---

## まとめ

| 概念 | 要点 |
|------|------|
| パッケージ | ディレクトリ = パッケージ。大文字始まりでエクスポート |
| go.mod | モジュール名・Go バージョン・依存を宣言 |
| go.sum | 依存のチェックサムで改竄検知。必ずコミット |
| MVS | Minimum Version Selection で再現性の高い依存解決 |
| internal | モジュール外からのアクセスをコンパイラが禁止 |
| workspace | go.work でマルチモジュール開発。CI/CD では無効化 |
| init() | パッケージ読込時に自動実行。重い処理は避ける |
| vendor | オフラインビルド・監査用。通常は不要 |
| replace | 一時的なパッチ用。go.work を優先的に検討 |
| retract | 問題のあるバージョンを撤回 |
| ビルドタグ | プラットフォーム別・条件付きコンパイル |
| go generate | コード自動生成の標準機構 |

---

## 次に読むべきガイド

- [../01-concurrency/00-goroutines-channels.md](../01-concurrency/00-goroutines-channels.md) -- 並行プログラミング
- [../02-web/00-net-http.md](../02-web/00-net-http.md) -- Web開発
- [../03-tools/04-best-practices.md](../03-tools/04-best-practices.md) -- プロジェクト構成のベストプラクティス

---

## 参考文献

1. **Go Modules Reference** -- https://go.dev/ref/mod
2. **Go Blog, "Using Go Modules"** -- https://go.dev/blog/using-go-modules
3. **Russ Cox, "Minimal Version Selection"** -- https://research.swtch.com/vgo-mvs
4. **Go Blog, "Go Modules: v2 and Beyond"** -- https://go.dev/blog/v2-go-modules
5. **Go Blog, "Publishing Go Modules"** -- https://go.dev/blog/publishing-go-modules
6. **Go Workspace Tutorial** -- https://go.dev/doc/tutorial/workspaces
