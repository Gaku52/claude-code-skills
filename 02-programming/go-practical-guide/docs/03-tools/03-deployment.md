# Go デプロイガイド

> Docker、クロスコンパイルを駆使してGoアプリケーションを効率的にビルド・デプロイする

## この章で学ぶこと

1. **Docker マルチステージビルド** で最小のコンテナイメージを構築する方法
2. **クロスコンパイル** によるマルチプラットフォーム対応バイナリの生成
3. **CI/CDパイプライン** でのビルド・テスト・デプロイ自動化
4. **Kubernetes デプロイ** -- マニフェスト設計、ヘルスチェック、リソース管理
5. **サーバーレスデプロイ** -- AWS Lambda、Google Cloud Run
6. **Graceful Shutdown** -- 安全なプロセス停止と接続管理
7. **設定管理** -- 環境変数、設定ファイル、シークレット管理

---

## 1. Goバイナリの特性とデプロイ戦略

### デプロイ方式の選択

```
Go アプリをデプロイしたい
        |
        +-- シングルバイナリ配布
        |       |
        |       +-- クロスコンパイル → GitHub Releases
        |       +-- GoReleaser で自動化
        |       +-- Homebrew tap で配布
        |
        +-- コンテナデプロイ
        |       |
        |       +-- Docker マルチステージビルド
        |       +-- distroless / scratch ベース
        |       +-- Kubernetes / ECS / Cloud Run
        |
        +-- サーバーレス
        |       |
        |       +-- AWS Lambda (provided.al2023)
        |       +-- Google Cloud Functions (Go 1.22+)
        |       +-- Google Cloud Run (コンテナ)
        |       +-- Azure Functions
        |
        +-- PaaS
                |
                +-- Google App Engine
                +-- Heroku (Container Stack)
                +-- Fly.io
                +-- Railway
```

### Goバイナリの特徴

```
+------------------------------------------+
|  Go バイナリ (静的リンク)                  |
+------------------------------------------+
|                                          |
|  +----------------+  +-----------------+ |
|  | アプリコード    |  | Go ランタイム    | |
|  | ビジネスロジック|  | GC, scheduler  | |
|  +----------------+  +-----------------+ |
|                                          |
|  +----------------+  +-----------------+ |
|  | 標準ライブラリ  |  | 依存ライブラリ   | |
|  | net/http, etc  |  | 全て埋め込み     | |
|  +----------------+  +-----------------+ |
|                                          |
|  → 外部依存なし、単体で実行可能           |
|  → CGO_ENABLED=0 で完全静的リンク        |
|  → 起動時間: 数ミリ秒                    |
|  → 典型的なサイズ: 10-30 MB              |
+------------------------------------------+

CGO_ENABLED=0 の場合:
  ┌─────────────────────────────────────┐
  │ Go Runtime + App Code               │
  │ すべてGoで実装（C依存なし）           │
  │ → scratch/distroless で実行可能      │
  └─────────────────────────────────────┘

CGO_ENABLED=1 の場合:
  ┌─────────────────────────────────────┐
  │ Go Runtime + App Code               │
  │ + libc (glibc/musl)                 │
  │ → alpine (musl) or debian (glibc)   │
  │   のベースイメージが必要              │
  └─────────────────────────────────────┘
```

---

## 2. Docker マルチステージビルド

### コード例1: 本番用 Dockerfile

```dockerfile
# ============================================
# Stage 1: ビルドステージ
# ============================================
FROM golang:1.22-alpine AS builder

# セキュリティアップデートとビルドツール
RUN apk add --no-cache git ca-certificates tzdata

# 非rootユーザーでビルド（セキュリティ向上）
RUN adduser -D -g '' appuser

# 依存のキャッシュ層
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download && go mod verify

# ソースコードのコピーとビルド
COPY . .

# ビルド引数
ARG VERSION=dev
ARG BUILD_TIME=unknown
ARG GIT_COMMIT=unknown

RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build \
    -trimpath \
    -ldflags="-w -s \
        -X main.version=${VERSION} \
        -X main.buildTime=${BUILD_TIME} \
        -X main.gitCommit=${GIT_COMMIT}" \
    -o /app/server ./cmd/server

# ============================================
# Stage 2: 実行ステージ（最小イメージ）
# ============================================
FROM gcr.io/distroless/static-debian12

# タイムゾーンデータと証明書をコピー
COPY --from=builder /usr/share/zoneinfo /usr/share/zoneinfo
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

# 非rootユーザー情報をコピー
COPY --from=builder /etc/passwd /etc/passwd

# 非rootユーザーで実行
USER appuser

# バイナリをコピー
COPY --from=builder /app/server /server

# 設定ファイルやマイグレーションも必要に応じてコピー
# COPY --from=builder /app/migrations /migrations
# COPY --from=builder /app/configs /configs

EXPOSE 8080

# ヘルスチェック用エンドポイント
# HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
#   CMD ["/server", "healthcheck"] || exit 1

ENTRYPOINT ["/server"]
```

### コード例2: Docker Compose（開発環境）

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        VERSION: dev
        BUILD_TIME: "2024-01-01T00:00:00Z"
    ports:
      - "8080:8080"
    environment:
      - DATABASE_URL=postgres://user:pass@db:5432/mydb?sslmode=disable
      - REDIS_URL=redis://redis:6379/0
      - LOG_LEVEL=debug
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_started
    restart: unless-stopped

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: mydb
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./migrations/init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user -d mydb"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  # 開発用: ホットリロード
  app-dev:
    build:
      context: .
      dockerfile: Dockerfile.dev
    ports:
      - "8080:8080"
    volumes:
      - .:/app
    environment:
      - DATABASE_URL=postgres://user:pass@db:5432/mydb?sslmode=disable
    depends_on:
      - db
    profiles:
      - dev

volumes:
  postgres_data:
  redis_data:
```

```dockerfile
# Dockerfile.dev -- 開発用（ホットリロード対応）
FROM golang:1.22-alpine

RUN go install github.com/air-verse/air@latest

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY . .

CMD ["air", "-c", ".air.toml"]
```

```toml
# .air.toml
root = "."
tmp_dir = "tmp"

[build]
  cmd = "go build -o ./tmp/server ./cmd/server"
  bin = "tmp/server"
  full_bin = "./tmp/server"
  include_ext = ["go", "tpl", "tmpl", "html", "sql"]
  exclude_dir = ["assets", "tmp", "vendor", "node_modules"]
  delay = 1000

[log]
  time = false

[color]
  main = "magenta"
  watcher = "cyan"
  build = "yellow"
  runner = "green"
```

### Dockerイメージサイズ比較

```
+----------------------------------------------+
| ベースイメージ別サイズ比較                     |
+----------------------------------------------+
|                                              |
| golang:1.22          |████████████| 850 MB   |
| golang:1.22-alpine   |██████|      350 MB   |
| alpine:3.19          |█|            7 MB    |
| distroless/static    |░|            2 MB    |
| scratch              |░|            0 MB    |
|                                              |
| 最終イメージ (distroless + Go binary)         |
|                      |██|          15-20 MB  |
|                                              |
| 最終イメージ (scratch + Go binary)            |
|                      |█|           10-15 MB  |
+----------------------------------------------+
```

### コード例3: scratch ベースの最小イメージ

```dockerfile
FROM golang:1.22-alpine AS builder

# TLS用の証明書を取得
RUN apk add --no-cache ca-certificates tzdata

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -trimpath -ldflags="-w -s" -o /app/server ./cmd/server

# scratch: 完全に空のイメージ
FROM scratch

# TLS通信に必要
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

# タイムゾーン情報
COPY --from=builder /usr/share/zoneinfo /usr/share/zoneinfo

# バイナリ
COPY --from=builder /app/server /server

EXPOSE 8080
ENTRYPOINT ["/server"]
```

### ベースイメージ比較表

| ベースイメージ | サイズ | シェル | パッケージマネージャ | デバッグ | セキュリティ | 用途 |
|--------------|--------|--------|-------------------|---------|------------|------|
| golang:1.22 | 850MB | bash | apt | 容易 | 攻撃対象面大 | 開発のみ |
| golang:1.22-alpine | 350MB | ash | apk | 可能 | 良好 | ビルドステージ |
| alpine:3.19 | 7MB | ash | apk | 可能 | 良好 | CGO必要時 |
| distroless/static | 2MB | なし | なし | 困難 | 非常に良好 | 本番推奨 |
| scratch | 0MB | なし | なし | 非常に困難 | 最高 | 最小構成 |

### コード例4: デバッグ可能なイメージ

```dockerfile
# 本番イメージにデバッグツールを追加したバリエーション
FROM gcr.io/distroless/static-debian12:debug AS debug

COPY --from=builder /app/server /server

# debug タグにはbusyboxシェルが含まれる
# kubectl exec -it <pod> -- /busybox/sh
ENTRYPOINT ["/server"]

# 使い分け:
# 本番: gcr.io/distroless/static-debian12 (シェルなし、最小攻撃面)
# デバッグ: gcr.io/distroless/static-debian12:debug (busybox付き)
```

---

## 3. クロスコンパイル

### コード例5: マルチプラットフォームビルド

```bash
# Linux AMD64 (サーバー、CI/CD)
GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -o myapp-linux-amd64 ./cmd/myapp

# Linux ARM64 (AWS Graviton, Raspberry Pi 4)
GOOS=linux GOARCH=arm64 CGO_ENABLED=0 go build -o myapp-linux-arm64 ./cmd/myapp

# Linux ARM v7 (Raspberry Pi 3, 古いARM)
GOOS=linux GOARCH=arm GOARM=7 CGO_ENABLED=0 go build -o myapp-linux-armv7 ./cmd/myapp

# macOS Intel
GOOS=darwin GOARCH=amd64 CGO_ENABLED=0 go build -o myapp-darwin-amd64 ./cmd/myapp

# macOS Apple Silicon
GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go build -o myapp-darwin-arm64 ./cmd/myapp

# Windows
GOOS=windows GOARCH=amd64 CGO_ENABLED=0 go build -o myapp-windows-amd64.exe ./cmd/myapp

# サポートされるOS/ARCH一覧
go tool dist list
```

### コード例6: Makefile でのビルド管理

```makefile
APP_NAME := myapp
VERSION := $(shell git describe --tags --always --dirty)
BUILD_TIME := $(shell date -u +%Y-%m-%dT%H:%M:%SZ)
GIT_COMMIT := $(shell git rev-parse --short HEAD)
LDFLAGS := -ldflags "-w -s \
    -X main.version=$(VERSION) \
    -X main.buildTime=$(BUILD_TIME) \
    -X main.gitCommit=$(GIT_COMMIT)"

# Go のビルドフラグ
GO_BUILD := CGO_ENABLED=0 go build -trimpath $(LDFLAGS)

# ターゲット
.PHONY: build build-all test lint clean docker docker-push help

## help: ヘルプを表示
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/## //'

## build: ローカルビルド
build:
	$(GO_BUILD) -o bin/$(APP_NAME) ./cmd/$(APP_NAME)

## build-all: 全プラットフォーム向けビルド
build-all:
	GOOS=linux   GOARCH=amd64 $(GO_BUILD) -o bin/$(APP_NAME)-linux-amd64 ./cmd/$(APP_NAME)
	GOOS=linux   GOARCH=arm64 $(GO_BUILD) -o bin/$(APP_NAME)-linux-arm64 ./cmd/$(APP_NAME)
	GOOS=darwin  GOARCH=amd64 $(GO_BUILD) -o bin/$(APP_NAME)-darwin-amd64 ./cmd/$(APP_NAME)
	GOOS=darwin  GOARCH=arm64 $(GO_BUILD) -o bin/$(APP_NAME)-darwin-arm64 ./cmd/$(APP_NAME)
	GOOS=windows GOARCH=amd64 $(GO_BUILD) -o bin/$(APP_NAME)-windows-amd64.exe ./cmd/$(APP_NAME)

## test: テスト実行
test:
	go test -race -cover -coverprofile=coverage.out ./...

## test-integration: インテグレーションテスト
test-integration:
	go test -race -tags=integration -cover ./...

## lint: リンターチェック
lint:
	golangci-lint run ./...

## fmt: コードフォーマット
fmt:
	gofmt -w .
	goimports -w .

## vet: 静的解析
vet:
	go vet ./...

## docker: Dockerイメージビルド
docker:
	docker build \
		--build-arg VERSION=$(VERSION) \
		--build-arg BUILD_TIME=$(BUILD_TIME) \
		--build-arg GIT_COMMIT=$(GIT_COMMIT) \
		-t $(APP_NAME):$(VERSION) \
		-t $(APP_NAME):latest .

## docker-push: Dockerイメージプッシュ
docker-push: docker
	docker tag $(APP_NAME):$(VERSION) ghcr.io/myorg/$(APP_NAME):$(VERSION)
	docker push ghcr.io/myorg/$(APP_NAME):$(VERSION)

## docker-multi: マルチアーキテクチャビルド
docker-multi:
	docker buildx build \
		--platform linux/amd64,linux/arm64 \
		--build-arg VERSION=$(VERSION) \
		-t ghcr.io/myorg/$(APP_NAME):$(VERSION) \
		--push .

## migrate-up: マイグレーション実行
migrate-up:
	migrate -path ./migrations -database $(DATABASE_URL) up

## migrate-down: マイグレーションロールバック
migrate-down:
	migrate -path ./migrations -database $(DATABASE_URL) down 1

## migrate-create: マイグレーションファイル作成
migrate-create:
	@read -p "Migration name: " name; \
	migrate create -ext sql -dir ./migrations -seq $$name

## clean: ビルド成果物削除
clean:
	rm -rf bin/ tmp/ coverage.out

## coverage: カバレッジレポート表示
coverage: test
	go tool cover -html=coverage.out -o coverage.html
	open coverage.html
```

### コード例7: ビルド時の変数埋め込み

```go
package main

import (
    "encoding/json"
    "fmt"
    "net/http"
    "os"
    "runtime"
    "runtime/debug"
)

// ビルド時に -ldflags で注入
var (
    version   = "dev"
    buildTime = "unknown"
    gitCommit = "unknown"
)

// BuildInfo はビルド情報を表す
type BuildInfo struct {
    Version   string `json:"version"`
    BuildTime string `json:"build_time"`
    GitCommit string `json:"git_commit"`
    GoVersion string `json:"go_version"`
    OS        string `json:"os"`
    Arch      string `json:"arch"`
    Compiler  string `json:"compiler"`
}

// GetBuildInfo はビルド情報を取得する
func GetBuildInfo() BuildInfo {
    info := BuildInfo{
        Version:   version,
        BuildTime: buildTime,
        GitCommit: gitCommit,
        GoVersion: runtime.Version(),
        OS:        runtime.GOOS,
        Arch:      runtime.GOARCH,
        Compiler:  runtime.Compiler,
    }

    // debug.ReadBuildInfo() でモジュール情報も取得可能
    if bi, ok := debug.ReadBuildInfo(); ok {
        for _, s := range bi.Settings {
            switch s.Key {
            case "vcs.revision":
                if info.GitCommit == "unknown" {
                    info.GitCommit = s.Value
                }
            }
        }
    }

    return info
}

// HandleVersion はバージョン情報をJSONで返すHTTPハンドラ
func HandleVersion(w http.ResponseWriter, r *http.Request) {
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(GetBuildInfo())
}

// PrintVersion はバージョン情報を標準出力に表示する
func PrintVersion() {
    info := GetBuildInfo()
    fmt.Printf("%s version %s\n", os.Args[0], info.Version)
    fmt.Printf("  Built:    %s\n", info.BuildTime)
    fmt.Printf("  Commit:   %s\n", info.GitCommit)
    fmt.Printf("  Go:       %s\n", info.GoVersion)
    fmt.Printf("  OS/Arch:  %s/%s\n", info.OS, info.Arch)
}

func main() {
    // --version フラグ対応
    if len(os.Args) > 1 && (os.Args[1] == "--version" || os.Args[1] == "-v") {
        PrintVersion()
        return
    }

    // サーバー起動
    http.HandleFunc("/version", HandleVersion)
    http.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
        w.WriteHeader(http.StatusOK)
        w.Write([]byte("ok"))
    })

    fmt.Printf("Starting server %s on :8080\n", version)
    if err := http.ListenAndServe(":8080", nil); err != nil {
        fmt.Fprintf(os.Stderr, "Server error: %v\n", err)
        os.Exit(1)
    }
}
```

---

## 4. Graceful Shutdown

### コード例8: 本番対応の Graceful Shutdown

```go
package main

import (
    "context"
    "database/sql"
    "fmt"
    "log"
    "net/http"
    "os"
    "os/signal"
    "sync"
    "syscall"
    "time"
)

// App はアプリケーション全体を管理する
type App struct {
    httpServer *http.Server
    db         *sql.DB
    wg         sync.WaitGroup
}

// NewApp はアプリケーションを初期化する
func NewApp(db *sql.DB) *App {
    mux := http.NewServeMux()

    app := &App{
        db: db,
        httpServer: &http.Server{
            Addr:         ":8080",
            Handler:      mux,
            ReadTimeout:  15 * time.Second,
            WriteTimeout: 15 * time.Second,
            IdleTimeout:  60 * time.Second,
        },
    }

    mux.HandleFunc("/healthz", app.handleHealth)
    mux.HandleFunc("/readyz", app.handleReady)
    mux.HandleFunc("/api/", app.handleAPI)

    return app
}

// Run はサーバーを起動し、シグナルを待ってGraceful Shutdownする
func (app *App) Run() error {
    // サーバー起動
    errCh := make(chan error, 1)
    go func() {
        log.Printf("Server starting on %s", app.httpServer.Addr)
        if err := app.httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
            errCh <- err
        }
    }()

    // シグナル待ち
    quit := make(chan os.Signal, 1)
    signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

    select {
    case err := <-errCh:
        return fmt.Errorf("server error: %w", err)
    case sig := <-quit:
        log.Printf("Received signal: %s", sig)
    }

    // Graceful Shutdown
    return app.Shutdown()
}

// Shutdown はアプリケーションを安全に停止する
func (app *App) Shutdown() error {
    log.Println("Starting graceful shutdown...")

    // Phase 1: 新しいリクエストの受付を停止
    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()

    // HTTPサーバーのシャットダウン（進行中のリクエスト完了を待つ）
    if err := app.httpServer.Shutdown(ctx); err != nil {
        log.Printf("HTTP server shutdown error: %v", err)
    }

    // Phase 2: バックグラウンドタスクの完了を待つ
    done := make(chan struct{})
    go func() {
        app.wg.Wait()
        close(done)
    }()

    select {
    case <-done:
        log.Println("All background tasks completed")
    case <-ctx.Done():
        log.Println("Timeout waiting for background tasks")
    }

    // Phase 3: リソースのクリーンアップ
    if app.db != nil {
        if err := app.db.Close(); err != nil {
            log.Printf("DB close error: %v", err)
        }
    }

    log.Println("Graceful shutdown completed")
    return nil
}

// handleHealth はLiveness probe用
func (app *App) handleHealth(w http.ResponseWriter, r *http.Request) {
    w.WriteHeader(http.StatusOK)
    w.Write([]byte("ok"))
}

// handleReady はReadiness probe用
func (app *App) handleReady(w http.ResponseWriter, r *http.Request) {
    if err := app.db.PingContext(r.Context()); err != nil {
        http.Error(w, "db not ready", http.StatusServiceUnavailable)
        return
    }
    w.WriteHeader(http.StatusOK)
    w.Write([]byte("ready"))
}

// handleAPI はビジネスロジックのハンドラ
func (app *App) handleAPI(w http.ResponseWriter, r *http.Request) {
    // バックグラウンドタスクのトラッキング
    app.wg.Add(1)
    defer app.wg.Done()

    // 処理...
    w.Write([]byte("ok"))
}
```

---

## 5. CI/CD パイプライン

### コード例9: GitHub Actions ワークフロー（フル構成）

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

permissions:
  contents: read

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-go@v5
        with:
          go-version: '1.22'
      - name: golangci-lint
        uses: golangci/golangci-lint-action@v4
        with:
          version: latest

  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16-alpine
        env:
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
          POSTGRES_DB: testdb
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-go@v5
        with:
          go-version: '1.22'
      - name: Run tests
        env:
          DATABASE_URL: postgres://test:test@localhost:5432/testdb?sslmode=disable
        run: |
          go test -race -coverprofile=coverage.out -covermode=atomic ./...
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.out

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-go@v5
        with:
          go-version: '1.22'
      - name: Run govulncheck
        run: |
          go install golang.org/x/vuln/cmd/govulncheck@latest
          govulncheck ./...
      - name: Run gosec
        uses: securego/gosec@master
        with:
          args: ./...

  build:
    needs: [lint, test, security]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-go@v5
        with:
          go-version: '1.22'
      - name: Build
        run: |
          CGO_ENABLED=0 go build -trimpath -ldflags="-w -s" -o bin/server ./cmd/server
      - uses: actions/upload-artifact@v4
        with:
          name: server
          path: bin/server
```

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags:
      - 'v*'

permissions:
  contents: write
  packages: write

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-go@v5
        with:
          go-version: '1.22'
      - run: go test -race -cover ./...

  release:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: actions/setup-go@v5
        with:
          go-version: '1.22'
      - uses: goreleaser/goreleaser-action@v5
        with:
          version: latest
          args: release --clean
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

  docker:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          push: true
          platforms: linux/amd64,linux/arm64
          build-args: |
            VERSION=${{ github.ref_name }}
            BUILD_TIME=${{ github.event.head_commit.timestamp }}
            GIT_COMMIT=${{ github.sha }}
          tags: |
            ghcr.io/${{ github.repository }}:${{ github.ref_name }}
            ghcr.io/${{ github.repository }}:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

### CI/CDパイプラインフロー

```
Pull Request → main へマージ
    │
    ▼
┌─────────────────────────────────────────────┐
│  CI Pipeline (on push / PR)                  │
│                                             │
│  ┌──────────┐ ┌──────────┐ ┌─────────────┐ │
│  │   Lint   │ │   Test   │ │  Security   │ │
│  │golangci  │ │ go test  │ │ govulncheck │ │
│  │ -lint    │ │ -race    │ │ gosec       │ │
│  └──────────┘ └──────────┘ └─────────────┘ │
│       │            │              │         │
│       └────────────┼──────────────┘         │
│                    ▼                        │
│             ┌──────────┐                    │
│             │  Build   │                    │
│             │ artifact │                    │
│             └──────────┘                    │
└─────────────────────────────────────────────┘

git tag v1.2.3 && git push --tags
    │
    ▼
┌─────────────────────────────────────────────┐
│  Release Pipeline (on tag)                   │
│                                             │
│  ┌──────────┐                               │
│  │   Test   │                               │
│  └────┬─────┘                               │
│       │ PASS                                │
│       ▼                                     │
│  ┌──────────┐    ┌──────────────────────┐   │
│  │GoReleaser│    │  Docker Build        │   │
│  │          │    │                      │   │
│  │ linux/   │    │ linux/amd64 image    │   │
│  │  amd64   │    │ linux/arm64 image    │   │
│  │  arm64   │    │                      │   │
│  │ darwin/  │    │ → ghcr.io push       │   │
│  │  amd64   │    └──────────────────────┘   │
│  │  arm64   │                               │
│  │ windows/ │                               │
│  │  amd64   │                               │
│  │          │                               │
│  │ → GitHub │                               │
│  │  Release │                               │
│  └──────────┘                               │
└─────────────────────────────────────────────┘
```

---

## 6. GoReleaser 設定

### コード例10: .goreleaser.yaml（フル構成）

```yaml
# .goreleaser.yaml
version: 2
project_name: myapp

before:
  hooks:
    - go mod tidy
    - go mod verify
    - go test ./...
    - go vet ./...

builds:
  - id: server
    main: ./cmd/server
    binary: myapp-server
    env:
      - CGO_ENABLED=0
    goos:
      - linux
      - darwin
      - windows
    goarch:
      - amd64
      - arm64
    ignore:
      - goos: windows
        goarch: arm64
    ldflags:
      - -s -w
      - -X main.version={{.Version}}
      - -X main.buildTime={{.Date}}
      - -X main.gitCommit={{.Commit}}
    flags:
      - -trimpath

  - id: cli
    main: ./cmd/cli
    binary: myapp
    env:
      - CGO_ENABLED=0
    goos:
      - linux
      - darwin
      - windows
    goarch:
      - amd64
      - arm64

archives:
  - id: server-archive
    builds:
      - server
    format: tar.gz
    name_template: "{{ .ProjectName }}-server_{{ .Os }}_{{ .Arch }}"
    format_overrides:
      - goos: windows
        format: zip
    files:
      - LICENSE
      - README.md
      - migrations/**/*

  - id: cli-archive
    builds:
      - cli
    format: tar.gz
    name_template: "{{ .ProjectName }}_{{ .Os }}_{{ .Arch }}"
    format_overrides:
      - goos: windows
        format: zip

dockers:
  - image_templates:
      - "ghcr.io/myorg/myapp:{{ .Version }}-amd64"
    use: buildx
    ids:
      - server
    build_flag_templates:
      - "--platform=linux/amd64"
      - "--label=org.opencontainers.image.source=https://github.com/myorg/myapp"
    goarch: amd64

  - image_templates:
      - "ghcr.io/myorg/myapp:{{ .Version }}-arm64"
    use: buildx
    ids:
      - server
    build_flag_templates:
      - "--platform=linux/arm64"
    goarch: arm64

docker_manifests:
  - name_template: "ghcr.io/myorg/myapp:{{ .Version }}"
    image_templates:
      - "ghcr.io/myorg/myapp:{{ .Version }}-amd64"
      - "ghcr.io/myorg/myapp:{{ .Version }}-arm64"
  - name_template: "ghcr.io/myorg/myapp:latest"
    image_templates:
      - "ghcr.io/myorg/myapp:{{ .Version }}-amd64"
      - "ghcr.io/myorg/myapp:{{ .Version }}-arm64"

brews:
  - name: myapp
    repository:
      owner: myorg
      name: homebrew-tap
    directory: Formula
    homepage: "https://github.com/myorg/myapp"
    description: "My awesome app"
    license: "MIT"
    install: |
      bin.install "myapp"
    test: |
      system "#{bin}/myapp", "--version"

checksum:
  name_template: 'checksums.txt'

changelog:
  sort: asc
  groups:
    - title: Features
      regexp: '^.*?feat(\(.+\))?\!?:.+$'
      order: 0
    - title: Bug fixes
      regexp: '^.*?fix(\(.+\))?\!?:.+$'
      order: 1
    - title: Others
      order: 999
  filters:
    exclude:
      - '^docs:'
      - '^test:'
      - '^chore:'
```

---

## 7. Kubernetes デプロイ

### コード例11: Kubernetes マニフェスト

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
  labels:
    app: myapp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  template:
    metadata:
      labels:
        app: myapp
    spec:
      serviceAccountName: myapp
      securityContext:
        runAsNonRoot: true
        runAsUser: 65534
        fsGroup: 65534
      containers:
        - name: myapp
          image: ghcr.io/myorg/myapp:v1.0.0
          ports:
            - containerPort: 8080
              name: http
              protocol: TCP
          env:
            - name: LOG_LEVEL
              value: "info"
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: myapp-secrets
                  key: database-url
          resources:
            requests:
              cpu: 100m
              memory: 128Mi
            limits:
              cpu: 500m
              memory: 256Mi
          # Liveness: プロセスが生存しているか
          livenessProbe:
            httpGet:
              path: /healthz
              port: http
            initialDelaySeconds: 5
            periodSeconds: 10
            timeoutSeconds: 3
            failureThreshold: 3
          # Readiness: トラフィックを受け付けられるか
          readinessProbe:
            httpGet:
              path: /readyz
              port: http
            initialDelaySeconds: 5
            periodSeconds: 5
            timeoutSeconds: 3
            failureThreshold: 3
          # Startup: 起動完了したか（遅い起動用）
          startupProbe:
            httpGet:
              path: /healthz
              port: http
            initialDelaySeconds: 0
            periodSeconds: 3
            failureThreshold: 10
      terminationGracePeriodSeconds: 30
---
apiVersion: v1
kind: Service
metadata:
  name: myapp
spec:
  selector:
    app: myapp
  ports:
    - port: 80
      targetPort: http
      protocol: TCP
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: myapp
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: myapp
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
---
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: myapp
spec:
  minAvailable: 1
  selector:
    matchLabels:
      app: myapp
```

---

## 8. サーバーレスデプロイ

### コード例12: AWS Lambda

```go
package main

import (
    "context"
    "encoding/json"

    "github.com/aws/aws-lambda-go/events"
    "github.com/aws/aws-lambda-go/lambda"
)

// Handler はLambdaのハンドラ
func Handler(ctx context.Context, request events.APIGatewayProxyRequest) (events.APIGatewayProxyResponse, error) {
    // リクエスト処理
    body := map[string]interface{}{
        "message": "Hello from Lambda!",
        "path":    request.Path,
        "method":  request.HTTPMethod,
    }

    jsonBody, _ := json.Marshal(body)

    return events.APIGatewayProxyResponse{
        StatusCode: 200,
        Headers: map[string]string{
            "Content-Type": "application/json",
        },
        Body: string(jsonBody),
    }, nil
}

func main() {
    lambda.Start(Handler)
}
```

```makefile
# Lambda用ビルド
lambda-build:
	GOOS=linux GOARCH=arm64 CGO_ENABLED=0 \
		go build -trimpath -ldflags="-w -s" \
		-o bootstrap ./cmd/lambda
	zip function.zip bootstrap

lambda-deploy: lambda-build
	aws lambda update-function-code \
		--function-name myfunction \
		--zip-file fileb://function.zip \
		--architectures arm64
```

### コード例13: Google Cloud Run

```dockerfile
# Cloud Run 用 Dockerfile
FROM golang:1.22-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-w -s" -o server ./cmd/server

FROM gcr.io/distroless/static-debian12
COPY --from=builder /app/server /server
# Cloud Run は PORT 環境変数でポートを指定する
ENV PORT=8080
EXPOSE 8080
ENTRYPOINT ["/server"]
```

```go
// Cloud Run用のサーバー
func main() {
    port := os.Getenv("PORT")
    if port == "" {
        port = "8080"
    }

    mux := http.NewServeMux()
    mux.HandleFunc("/", handler)

    log.Printf("Listening on :%s", port)
    if err := http.ListenAndServe(":"+port, mux); err != nil {
        log.Fatal(err)
    }
}
```

```bash
# Cloud Run デプロイ
gcloud run deploy myapp \
    --source . \
    --region asia-northeast1 \
    --allow-unauthenticated \
    --min-instances 0 \
    --max-instances 10 \
    --memory 256Mi \
    --cpu 1
```

---

## 9. 設定管理

### コード例14: 環境変数ベースの設定管理

```go
package config

import (
    "fmt"
    "os"
    "strconv"
    "time"
)

// Config はアプリケーション設定
type Config struct {
    Server   ServerConfig
    Database DatabaseConfig
    Redis    RedisConfig
    Log      LogConfig
}

type ServerConfig struct {
    Port         int
    ReadTimeout  time.Duration
    WriteTimeout time.Duration
    IdleTimeout  time.Duration
}

type DatabaseConfig struct {
    URL             string
    MaxOpenConns    int
    MaxIdleConns    int
    ConnMaxLifetime time.Duration
}

type RedisConfig struct {
    URL string
}

type LogConfig struct {
    Level  string
    Format string
}

// Load は環境変数から設定を読み込む
func Load() (*Config, error) {
    cfg := &Config{
        Server: ServerConfig{
            Port:         getEnvInt("PORT", 8080),
            ReadTimeout:  getEnvDuration("SERVER_READ_TIMEOUT", 15*time.Second),
            WriteTimeout: getEnvDuration("SERVER_WRITE_TIMEOUT", 15*time.Second),
            IdleTimeout:  getEnvDuration("SERVER_IDLE_TIMEOUT", 60*time.Second),
        },
        Database: DatabaseConfig{
            URL:             getEnvRequired("DATABASE_URL"),
            MaxOpenConns:    getEnvInt("DB_MAX_OPEN_CONNS", 25),
            MaxIdleConns:    getEnvInt("DB_MAX_IDLE_CONNS", 5),
            ConnMaxLifetime: getEnvDuration("DB_CONN_MAX_LIFETIME", 5*time.Minute),
        },
        Redis: RedisConfig{
            URL: getEnv("REDIS_URL", "redis://localhost:6379/0"),
        },
        Log: LogConfig{
            Level:  getEnv("LOG_LEVEL", "info"),
            Format: getEnv("LOG_FORMAT", "json"),
        },
    }

    return cfg, cfg.Validate()
}

func (c *Config) Validate() error {
    if c.Database.URL == "" {
        return fmt.Errorf("DATABASE_URL is required")
    }
    if c.Server.Port < 1 || c.Server.Port > 65535 {
        return fmt.Errorf("invalid PORT: %d", c.Server.Port)
    }
    return nil
}

func getEnv(key, defaultVal string) string {
    if v := os.Getenv(key); v != "" {
        return v
    }
    return defaultVal
}

func getEnvRequired(key string) string {
    v := os.Getenv(key)
    if v == "" {
        panic(fmt.Sprintf("required environment variable %s is not set", key))
    }
    return v
}

func getEnvInt(key string, defaultVal int) int {
    if v := os.Getenv(key); v != "" {
        i, err := strconv.Atoi(v)
        if err != nil {
            panic(fmt.Sprintf("invalid int value for %s: %s", key, v))
        }
        return i
    }
    return defaultVal
}

func getEnvDuration(key string, defaultVal time.Duration) time.Duration {
    if v := os.Getenv(key); v != "" {
        d, err := time.ParseDuration(v)
        if err != nil {
            panic(fmt.Sprintf("invalid duration for %s: %s", key, v))
        }
        return d
    }
    return defaultVal
}
```

---

## 10. ldflags オプション比較表

| フラグ | 効果 | サイズ削減 | 用途 |
|--------|------|-----------|------|
| `-w` | DWARFデバッグ情報を削除 | 約20-30% | 本番ビルド |
| `-s` | シンボルテーブルを削除 | 約10-20% | 本番ビルド |
| `-X pkg.var=val` | ビルド時に変数値を注入 | なし | バージョン情報埋め込み |
| `-extldflags "-static"` | 外部リンカで静的リンク | なし | CGO使用時の静的ビルド |
| `-trimpath` | ビルドパスを削除 | わずか | セキュリティ向上 |

---

## 11. アンチパターン

### アンチパターン1: ビルドステージをそのままデプロイ

```dockerfile
# NG: ビルド環境ごとデプロイ（850MB+）
FROM golang:1.22
WORKDIR /app
COPY . .
RUN go build -o server .
CMD ["./server"]
# 問題: イメージサイズ巨大、ビルドツールが含まれる（攻撃対象面大）

# OK: マルチステージビルド（15-20MB）
FROM golang:1.22-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-w -s" -o server .

FROM gcr.io/distroless/static-debian12
COPY --from=builder /app/server /server
ENTRYPOINT ["/server"]
```

### アンチパターン2: go mod download をキャッシュしない

```dockerfile
# NG: ソース変更のたびに依存を再ダウンロード
FROM golang:1.22-alpine AS builder
COPY . .
RUN go build -o server .
# ソース1行変更 → go mod download からやり直し（数分のロス）

# OK: go.mod/go.sum を先にコピーしてキャッシュ活用
FROM golang:1.22-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./        # 依存定義のみ先にコピー
RUN go mod download          # この層がキャッシュされる
COPY . .                     # ソース変更時もdownloadはスキップ
RUN go build -o server .
```

### アンチパターン3: rootユーザーでの実行

```dockerfile
# NG: root で実行（セキュリティリスク）
FROM alpine:3.19
COPY --from=builder /app/server /server
CMD ["/server"]
# コンテナ内で root 権限 → 脆弱性があるとホストに影響

# OK: 非rootユーザーで実行
FROM alpine:3.19
RUN adduser -D -g '' appuser
COPY --from=builder /app/server /server
USER appuser
CMD ["/server"]
```

### アンチパターン4: Graceful Shutdown なし

```go
// NG: シグナルを無視してすぐ終了
func main() {
    http.ListenAndServe(":8080", handler)
}
// SIGTERM → 処理中のリクエストが切断される

// OK: Graceful Shutdown
func main() {
    srv := &http.Server{Addr: ":8080", Handler: handler}

    go func() {
        if err := srv.ListenAndServe(); err != http.ErrServerClosed {
            log.Fatal(err)
        }
    }()

    quit := make(chan os.Signal, 1)
    signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
    <-quit

    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()
    srv.Shutdown(ctx)
}
```

### アンチパターン5: シークレットのハードコード

```go
// NG: コード内にシークレットを記述
db, _ := sql.Open("postgres", "postgres://admin:P@ssw0rd@prod-db:5432/mydb")

// OK: 環境変数から読み取り
db, _ := sql.Open("postgres", os.Getenv("DATABASE_URL"))

// BETTER: シークレット管理サービスを使用
// AWS Secrets Manager, GCP Secret Manager, HashiCorp Vault
```

### アンチパターン6: ヘルスチェックの未実装

```go
// NG: ヘルスチェックエンドポイントがない
// → Kubernetes がPodの状態を判断できず、障害時に自動復旧しない

// OK: Liveness/Readiness/Startup の3つを実装
http.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
    w.WriteHeader(http.StatusOK) // プロセスが動いていればOK
})

http.HandleFunc("/readyz", func(w http.ResponseWriter, r *http.Request) {
    if err := db.PingContext(r.Context()); err != nil {
        w.WriteHeader(http.StatusServiceUnavailable) // DBに繋がらなければNG
        return
    }
    w.WriteHeader(http.StatusOK)
})
```

---

## FAQ

### Q1. CGO_ENABLED=0 にする必要がある場面は？

scratch や distroless など glibc を含まないイメージで実行する場合は `CGO_ENABLED=0` が必須。標準ライブラリの `net` パッケージや `os/user` パッケージはデフォルトでCGOを使うが、`CGO_ENABLED=0` で純Go実装にフォールバックする。

CGOが必要な場面:
- SQLite（go-sqlite3）を使う場合 → 代替: `modernc.org/sqlite`（CGO不要）
- 画像処理（libvips等）を使う場合
- OS固有のライブラリ（macOS Security Framework等）

### Q2. Docker のマルチプラットフォームイメージの作り方は？

`docker buildx build --platform linux/amd64,linux/arm64` でマニフェストリストを作成できる。GoReleaserを使う場合は、各アーキテクチャのイメージを個別にビルドし、`docker manifest create` で統合する方法もある。

```bash
# buildx でマルチプラットフォームビルド
docker buildx create --name mybuilder --use
docker buildx build --platform linux/amd64,linux/arm64 \
    -t ghcr.io/myorg/myapp:v1.0.0 --push .
```

### Q3. バイナリサイズをさらに小さくする方法は？

| 手法 | サイズ削減 | トレードオフ |
|------|-----------|------------|
| `-ldflags="-w -s"` | 30-50% | デバッグ情報なし |
| `-trimpath` | わずか | ビルドパスなし |
| `upx` 圧縮 | 50-70% | 起動時解凍コスト |
| 不要な依存の削除 | 可変 | なし |
| Go 1.22+ の改善 | 自動 | なし |

### Q4. Kubernetes での推奨設定は？

- **リソース制限**: requests/limits を必ず設定
- **ヘルスチェック**: liveness/readiness/startup の3つ
- **PDB**: minAvailable で可用性を保証
- **HPA**: CPU/メモリベースの自動スケーリング
- **terminationGracePeriodSeconds**: Graceful Shutdown の猶予時間
- **securityContext**: runAsNonRoot, readOnlyRootFilesystem

### Q5. デプロイ時のダウンタイムを最小化するには？

1. **Rolling Update**: maxUnavailable=0 で常に全Podを維持
2. **Readiness Probe**: 準備完了までトラフィック遮断
3. **Graceful Shutdown**: 進行中のリクエスト完了を待つ
4. **PreStop Hook**: `sleep 5` でLB反映を待つ
5. **PDB**: minAvailable で同時停止数を制限

---

## まとめ

| 概念 | 要点 |
|------|------|
| マルチステージビルド | ビルド環境と実行環境を分離して最小イメージ |
| distroless / scratch | 攻撃対象面を最小化する実行イメージ |
| CGO_ENABLED=0 | 完全静的リンクで外部依存排除 |
| -ldflags "-w -s" | デバッグ情報削除でバイナリサイズ削減 |
| -trimpath | ビルドパス削除でセキュリティ向上 |
| -X main.version=... | ビルド時のバージョン情報埋め込み |
| GOOS/GOARCH | クロスコンパイルの環境変数 |
| GoReleaser | マルチプラットフォームリリース自動化 |
| GitHub Actions | CI/CDでテスト・ビルド・デプロイ自動化 |
| Graceful Shutdown | SIGTERM受信→進行中リクエスト完了→リソース解放 |
| Kubernetes | ヘルスチェック・HPA・PDB・リソース制限 |
| サーバーレス | Lambda/Cloud Run で運用コスト最小化 |
| 設定管理 | 環境変数・シークレット管理サービス |

---

## 次に読むべきガイド

- **03-tools/04-best-practices.md** -- ベストプラクティス：Effective Go
- **03-tools/02-profiling.md** -- プロファイリング：pprof、trace
- **03-tools/00-cli-development.md** -- CLI開発：cobra、flag、promptui

---

## 参考文献

1. **Docker公式 -- Multi-stage builds** https://docs.docker.com/build/building/multi-stage/
2. **GoReleaser 公式ドキュメント** https://goreleaser.com/
3. **Google -- distroless コンテナイメージ** https://github.com/GoogleContainerTools/distroless
4. **Go公式 -- Build constraints** https://pkg.go.dev/cmd/go#hdr-Build_constraints
5. **Kubernetes -- Configure Liveness, Readiness and Startup Probes** https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/
6. **AWS Lambda Go** https://docs.aws.amazon.com/lambda/latest/dg/lambda-golang.html
7. **Google Cloud Run** https://cloud.google.com/run/docs
8. **air -- Live reload for Go apps** https://github.com/air-verse/air
