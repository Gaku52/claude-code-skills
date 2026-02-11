# パッケージとモジュール -- Goのコード組織化

> Goはgo.modベースのモジュールシステムでパッケージを管理し、import・internal・バージョニングで堅牢な依存関係を実現する。

---

## この章で学ぶこと

1. **パッケージの仕組み** -- ディレクトリ=パッケージの原則
2. **go.mod / go.sum** -- モジュールシステムの基盤
3. **internal パッケージ** -- 可視性制御とAPI設計

---

## 1. パッケージの基本

### コード例 1: パッケージ構造

```
myproject/
├── go.mod
├── go.sum
├── main.go
├── internal/
│   └── config/
│       └── config.go
├── pkg/
│   └── validator/
│       └── validator.go
└── cmd/
    ├── server/
    │   └── main.go
    └── worker/
        └── main.go
```

### コード例 2: go.mod ファイル

```go
module github.com/myorg/myproject

go 1.22

require (
    github.com/gin-gonic/gin v1.9.1
    github.com/lib/pq v1.10.9
    go.uber.org/zap v1.27.0
)

require (
    // indirect dependencies
    golang.org/x/sys v0.15.0 // indirect
)
```

### コード例 3: エクスポートルール

```go
package user

// User は公開型（大文字始まり）
type User struct {
    ID   int    // 公開フィールド
    Name string // 公開フィールド
    age  int    // 非公開フィールド（パッケージ内のみ）
}

// NewUser は公開関数
func NewUser(name string) *User {
    return &User{Name: name, age: defaultAge()}
}

// defaultAge は非公開関数
func defaultAge() int {
    return 0
}
```

### コード例 4: internalパッケージ

```go
// internal/database/db.go
package database

// Connect は internal パッケージ内の関数
// 親モジュールの外からはインポートできない
func Connect(dsn string) (*sql.DB, error) {
    return sql.Open("postgres", dsn)
}

// 利用可能: github.com/myorg/myproject/cmd/server
// 利用不可: github.com/otherorg/otherproject (コンパイルエラー)
```

### コード例 5: ワークスペース (Go 1.18+)

```go
// go.work ファイル
go 1.22

use (
    ./api
    ./shared
    ./worker
)

// これによりローカルの複数モジュールを同時に開発できる
```

### コード例 6: init() 関数

```go
package driver

import "database/sql"

func init() {
    // パッケージインポート時に自動実行
    sql.Register("mydriver", &MyDriver{})
}

// ブランクインポートで init() だけ実行
// import _ "github.com/lib/pq"
```

---

## 2. ASCII図解

### 図1: モジュールとパッケージの関係

```
┌─── module: github.com/myorg/myproject ───────┐
│                                               │
│  ┌─ package: main ──┐  ┌─ package: config ─┐ │
│  │ cmd/server/       │  │ internal/config/   │ │
│  │   main.go         │  │   config.go        │ │
│  └──────────────────┘  └───────────────────┘ │
│                                               │
│  ┌─ package: handler ┐  ┌─ package: model ─┐ │
│  │ internal/handler/  │  │ pkg/model/        │ │
│  │   user.go          │  │   user.go         │ │
│  │   order.go         │  │   order.go        │ │
│  └───────────────────┘  └──────────────────┘ │
└───────────────────────────────────────────────┘
```

### 図2: go mod のバージョン解決

```
myproject
├── require A v1.2.0
│   └── require C v1.0.0
├── require B v1.3.0
│   └── require C v1.1.0   ← こちらが採用 (MVS)
│
│  Minimum Version Selection (MVS):
│  C の要求: v1.0.0, v1.1.0
│  選択: v1.1.0 (最小の互換バージョン)
│
└── go.sum: 全依存のチェックサム
```

### 図3: import解決フロー

```
import "github.com/myorg/pkg/util"
            │
            ▼
    ┌───────────────┐
    │ ローカルモジュール? │──YES──> go.work で解決
    └───────┬───────┘
            │NO
            ▼
    ┌───────────────┐
    │ go.mod に記載?  │──YES──> $GOMODCACHE から取得
    └───────┬───────┘
            │NO
            ▼
    ┌───────────────┐
    │ go get で取得   │──> プロキシ(proxy.golang.org)経由
    └───────────────┘
```

---

## 3. 比較表

### 表1: パッケージ管理の進化

| 世代 | 方式 | 期間 | 特徴 |
|------|------|------|------|
| 第1世代 | GOPATH | 2009-2018 | グローバルワークスペース、バージョン管理なし |
| 過渡期 | dep / glide | 2016-2019 | vendorディレクトリ、lock file |
| 第2世代 | Go Modules | 2019- | go.mod、MVS、プロキシ対応 |
| 拡張 | Workspace | 2022- | go.work、マルチモジュール開発 |

### 表2: ディレクトリ規約比較

| ディレクトリ | 用途 | 公開範囲 |
|-------------|------|---------|
| `cmd/` | エントリーポイント (main パッケージ) | -- |
| `pkg/` | 外部から利用可能なライブラリ | モジュール外から利用可 |
| `internal/` | 内部実装 | モジュール内のみ |
| `vendor/` | 依存のローカルコピー | -- |
| `api/` | API定義 (proto, OpenAPI) | -- |
| `docs/` | ドキュメント | -- |

---

## 4. アンチパターン

### アンチパターン 1: 循環インポート

```
// BAD: パッケージ A が B を、B が A をインポート
// package a
import "myproject/b"  // コンパイルエラー: import cycle

// package b
import "myproject/a"

// GOOD: インターフェースで依存を逆転
// package a
type Processor interface { Process() error }

// package b
import "myproject/a"
type MyProcessor struct{}
func (p *MyProcessor) Process() error { ... }
// a.Processor を満たす
```

### アンチパターン 2: 巨大なutilパッケージ

```
// BAD: 何でも入れるutilパッケージ
package util

func FormatDate(t time.Time) string { ... }
func HashPassword(p string) string { ... }
func ParseCSV(r io.Reader) ([][]string, error) { ... }
func SendEmail(to, subject, body string) error { ... }

// GOOD: 責務ごとにパッケージを分割
// package timeutil
func FormatDate(t time.Time) string { ... }

// package auth
func HashPassword(p string) string { ... }

// package csv
func Parse(r io.Reader) ([][]string, error) { ... }
```

---

## 5. FAQ

### Q1: pkg/ ディレクトリは必須か？

必須ではない。Go公式は特定のレイアウトを推奨していない。`pkg/` は「外部に公開するパッケージ」を示す慣習だが、小規模プロジェクトでは不要。重要なのは `internal/` で内部実装を保護すること。

### Q2: go.sum は何のために存在するか？

`go.sum` は依存パッケージのチェックサム（SHA-256ハッシュ）を記録する。これによりサプライチェーン攻撃を検知し、ビルドの再現性を保証する。バージョン管理にコミットすべきファイル。

### Q3: replace ディレクティブはいつ使うか？

`replace` は (1) ローカル開発中のモジュール参照、(2) フォークしたモジュールの利用、(3) セキュリティパッチの一時適用に使う。本番コードでは極力避け、go.workを検討する。

---

## まとめ

| 概念 | 要点 |
|------|------|
| パッケージ | ディレクトリ = パッケージ。大文字始まりでエクスポート |
| go.mod | モジュール名・Go バージョン・依存を宣言 |
| MVS | Minimum Version Selection で依存解決 |
| internal | モジュール外からのアクセスを禁止 |
| go.sum | 依存のチェックサムで改竄検知 |
| workspace | go.work でマルチモジュール開発 |
| init() | パッケージ読込時に自動実行（乱用注意） |

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
