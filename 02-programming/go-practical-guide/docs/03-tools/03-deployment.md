# Go Deployment Guide

> Build and deploy Go applications efficiently using Docker and cross-compilation

## What You Will Learn in This Chapter

1. How to build minimal container images with **Docker multi-stage builds**
2. Generating multi-platform binaries via **cross-compilation**
3. Automating build, test, and deploy in **CI/CD pipelines**
4. **Kubernetes deployment** -- manifest design, health checks, resource management
5. **Serverless deployment** -- AWS Lambda, Google Cloud Run
6. **Graceful Shutdown** -- safe process termination and connection management
7. **Configuration management** -- environment variables, config files, secret management


## Prerequisites

Reading this guide will be more effective if you have the following knowledge:

- Basic programming knowledge
- Understanding of related fundamental concepts
- Understanding the contents of the [Go Profiling Guide](./02-profiling.md)

---

## 1. Characteristics of Go Binaries and Deployment Strategies

### Choosing a Deployment Method

```
Want to deploy a Go app
        |
        +-- Single binary distribution
        |       |
        |       +-- Cross-compile → GitHub Releases
        |       +-- Automate with GoReleaser
        |       +-- Distribute via Homebrew tap
        |
        +-- Container deployment
        |       |
        |       +-- Docker multi-stage build
        |       +-- distroless / scratch base
        |       +-- Kubernetes / ECS / Cloud Run
        |
        +-- Serverless
        |       |
        |       +-- AWS Lambda (provided.al2023)
        |       +-- Google Cloud Functions (Go 1.22+)
        |       +-- Google Cloud Run (container)
        |       +-- Azure Functions
        |
        +-- PaaS
                |
                +-- Google App Engine
                +-- Heroku (Container Stack)
                +-- Fly.io
                +-- Railway
```

### Characteristics of Go Binaries

```
+------------------------------------------+
|  Go binary (statically linked)           |
+------------------------------------------+
|                                          |
|  +----------------+  +-----------------+ |
|  | App code       |  | Go runtime      | |
|  | Business logic |  | GC, scheduler   | |
|  +----------------+  +-----------------+ |
|                                          |
|  +----------------+  +-----------------+ |
|  | Standard lib   |  | Dependencies    | |
|  | net/http, etc  |  | All embedded    | |
|  +----------------+  +-----------------+ |
|                                          |
|  → No external dependencies, runs standalone |
|  → CGO_ENABLED=0 for fully static linking |
|  → Startup time: a few milliseconds      |
|  → Typical size: 10-30 MB                |
+------------------------------------------+

With CGO_ENABLED=0:
  ┌─────────────────────────────────────┐
  │ Go Runtime + App Code               │
  │ Everything implemented in Go (no C) │
  │ → Runnable on scratch/distroless    │
  └─────────────────────────────────────┘

With CGO_ENABLED=1:
  ┌─────────────────────────────────────┐
  │ Go Runtime + App Code               │
  │ + libc (glibc/musl)                 │
  │ → Requires alpine (musl) or         │
  │   debian (glibc) base image         │
  └─────────────────────────────────────┘
```

---

## 2. Docker Multi-Stage Builds

### Code Example 1: Production Dockerfile

```dockerfile
# ============================================
# Stage 1: Build stage
# ============================================
FROM golang:1.22-alpine AS builder

# Security updates and build tools
RUN apk add --no-cache git ca-certificates tzdata

# Build as non-root user (improved security)
RUN adduser -D -g '' appuser

# Dependency cache layer
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download && go mod verify

# Copy source and build
COPY . .

# Build arguments
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
# Stage 2: Runtime stage (minimal image)
# ============================================
FROM gcr.io/distroless/static-debian12

# Copy timezone data and certificates
COPY --from=builder /usr/share/zoneinfo /usr/share/zoneinfo
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

# Copy non-root user info
COPY --from=builder /etc/passwd /etc/passwd

# Run as non-root user
USER appuser

# Copy binary
COPY --from=builder /app/server /server

# Copy config files or migrations as needed
# COPY --from=builder /app/migrations /migrations
# COPY --from=builder /app/configs /configs

EXPOSE 8080

# Health check endpoint
# HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
#   CMD ["/server", "healthcheck"] || exit 1

ENTRYPOINT ["/server"]
```

### Code Example 2: Docker Compose (development environment)

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

  # For development: hot reload
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
# Dockerfile.dev -- for development (hot reload enabled)
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

### Docker Image Size Comparison

```
+----------------------------------------------+
| Size comparison by base image                |
+----------------------------------------------+
|                                              |
| golang:1.22          |████████████| 850 MB   |
| golang:1.22-alpine   |██████|      350 MB   |
| alpine:3.19          |█|            7 MB    |
| distroless/static    |░|            2 MB    |
| scratch              |░|            0 MB    |
|                                              |
| Final image (distroless + Go binary)         |
|                      |██|          15-20 MB  |
|                                              |
| Final image (scratch + Go binary)            |
|                      |█|           10-15 MB  |
+----------------------------------------------+
```

### Code Example 3: Minimal image based on scratch

```dockerfile
FROM golang:1.22-alpine AS builder

# Obtain certificates for TLS
RUN apk add --no-cache ca-certificates tzdata

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -trimpath -ldflags="-w -s" -o /app/server ./cmd/server

# scratch: completely empty image
FROM scratch

# Required for TLS communication
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

# Timezone information
COPY --from=builder /usr/share/zoneinfo /usr/share/zoneinfo

# Binary
COPY --from=builder /app/server /server

EXPOSE 8080
ENTRYPOINT ["/server"]
```

### Base Image Comparison Table

| Base image | Size | Shell | Package manager | Debugging | Security | Use case |
|--------------|--------|--------|-------------------|---------|------------|------|
| golang:1.22 | 850MB | bash | apt | Easy | Large attack surface | Development only |
| golang:1.22-alpine | 350MB | ash | apk | Possible | Good | Build stage |
| alpine:3.19 | 7MB | ash | apk | Possible | Good | When CGO is needed |
| distroless/static | 2MB | None | None | Difficult | Very good | Recommended for production |
| scratch | 0MB | None | None | Very difficult | Highest | Minimal configuration |

### Code Example 4: Debuggable image

```dockerfile
# Variant adding debug tools to the production image
FROM gcr.io/distroless/static-debian12:debug AS debug

COPY --from=builder /app/server /server

# The debug tag includes a busybox shell
# kubectl exec -it <pod> -- /busybox/sh
ENTRYPOINT ["/server"]

# Usage:
# Production: gcr.io/distroless/static-debian12 (no shell, minimal attack surface)
# Debug: gcr.io/distroless/static-debian12:debug (with busybox)
```

---

## 3. Cross-Compilation

### Code Example 5: Multi-platform builds

```bash
# Linux AMD64 (servers, CI/CD)
GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -o myapp-linux-amd64 ./cmd/myapp

# Linux ARM64 (AWS Graviton, Raspberry Pi 4)
GOOS=linux GOARCH=arm64 CGO_ENABLED=0 go build -o myapp-linux-arm64 ./cmd/myapp

# Linux ARM v7 (Raspberry Pi 3, older ARM)
GOOS=linux GOARCH=arm GOARM=7 CGO_ENABLED=0 go build -o myapp-linux-armv7 ./cmd/myapp

# macOS Intel
GOOS=darwin GOARCH=amd64 CGO_ENABLED=0 go build -o myapp-darwin-amd64 ./cmd/myapp

# macOS Apple Silicon
GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go build -o myapp-darwin-arm64 ./cmd/myapp

# Windows
GOOS=windows GOARCH=amd64 CGO_ENABLED=0 go build -o myapp-windows-amd64.exe ./cmd/myapp

# List of supported OS/ARCH combinations
go tool dist list
```

### Code Example 6: Managing builds with a Makefile

```makefile
APP_NAME := myapp
VERSION := $(shell git describe --tags --always --dirty)
BUILD_TIME := $(shell date -u +%Y-%m-%dT%H:%M:%SZ)
GIT_COMMIT := $(shell git rev-parse --short HEAD)
LDFLAGS := -ldflags "-w -s \
    -X main.version=$(VERSION) \
    -X main.buildTime=$(BUILD_TIME) \
    -X main.gitCommit=$(GIT_COMMIT)"

# Go build flags
GO_BUILD := CGO_ENABLED=0 go build -trimpath $(LDFLAGS)

# Targets
.PHONY: build build-all test lint clean docker docker-push help

## help: Display help
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/## //'

## build: Local build
build:
	$(GO_BUILD) -o bin/$(APP_NAME) ./cmd/$(APP_NAME)

## build-all: Build for all platforms
build-all:
	GOOS=linux   GOARCH=amd64 $(GO_BUILD) -o bin/$(APP_NAME)-linux-amd64 ./cmd/$(APP_NAME)
	GOOS=linux   GOARCH=arm64 $(GO_BUILD) -o bin/$(APP_NAME)-linux-arm64 ./cmd/$(APP_NAME)
	GOOS=darwin  GOARCH=amd64 $(GO_BUILD) -o bin/$(APP_NAME)-darwin-amd64 ./cmd/$(APP_NAME)
	GOOS=darwin  GOARCH=arm64 $(GO_BUILD) -o bin/$(APP_NAME)-darwin-arm64 ./cmd/$(APP_NAME)
	GOOS=windows GOARCH=amd64 $(GO_BUILD) -o bin/$(APP_NAME)-windows-amd64.exe ./cmd/$(APP_NAME)

## test: Run tests
test:
	go test -race -cover -coverprofile=coverage.out ./...

## test-integration: Integration tests
test-integration:
	go test -race -tags=integration -cover ./...

## lint: Linter check
lint:
	golangci-lint run ./...

## fmt: Code formatting
fmt:
	gofmt -w .
	goimports -w .

## vet: Static analysis
vet:
	go vet ./...

## docker: Build Docker image
docker:
	docker build \
		--build-arg VERSION=$(VERSION) \
		--build-arg BUILD_TIME=$(BUILD_TIME) \
		--build-arg GIT_COMMIT=$(GIT_COMMIT) \
		-t $(APP_NAME):$(VERSION) \
		-t $(APP_NAME):latest .

## docker-push: Push Docker image
docker-push: docker
	docker tag $(APP_NAME):$(VERSION) ghcr.io/myorg/$(APP_NAME):$(VERSION)
	docker push ghcr.io/myorg/$(APP_NAME):$(VERSION)

## docker-multi: Multi-architecture build
docker-multi:
	docker buildx build \
		--platform linux/amd64,linux/arm64 \
		--build-arg VERSION=$(VERSION) \
		-t ghcr.io/myorg/$(APP_NAME):$(VERSION) \
		--push .

## migrate-up: Run migrations
migrate-up:
	migrate -path ./migrations -database $(DATABASE_URL) up

## migrate-down: Rollback migration
migrate-down:
	migrate -path ./migrations -database $(DATABASE_URL) down 1

## migrate-create: Create a migration file
migrate-create:
	@read -p "Migration name: " name; \
	migrate create -ext sql -dir ./migrations -seq $$name

## clean: Remove build artifacts
clean:
	rm -rf bin/ tmp/ coverage.out

## coverage: Display coverage report
coverage: test
	go tool cover -html=coverage.out -o coverage.html
	open coverage.html
```

### Code Example 7: Embedding variables at build time

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

// Injected at build time via -ldflags
var (
    version   = "dev"
    buildTime = "unknown"
    gitCommit = "unknown"
)

// BuildInfo represents build information
type BuildInfo struct {
    Version   string `json:"version"`
    BuildTime string `json:"build_time"`
    GitCommit string `json:"git_commit"`
    GoVersion string `json:"go_version"`
    OS        string `json:"os"`
    Arch      string `json:"arch"`
    Compiler  string `json:"compiler"`
}

// GetBuildInfo retrieves build information
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

    // Module info can also be obtained via debug.ReadBuildInfo()
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

// HandleVersion is an HTTP handler that returns version info as JSON
func HandleVersion(w http.ResponseWriter, r *http.Request) {
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(GetBuildInfo())
}

// PrintVersion prints version information to standard output
func PrintVersion() {
    info := GetBuildInfo()
    fmt.Printf("%s version %s\n", os.Args[0], info.Version)
    fmt.Printf("  Built:    %s\n", info.BuildTime)
    fmt.Printf("  Commit:   %s\n", info.GitCommit)
    fmt.Printf("  Go:       %s\n", info.GoVersion)
    fmt.Printf("  OS/Arch:  %s/%s\n", info.OS, info.Arch)
}

func main() {
    // Handle --version flag
    if len(os.Args) > 1 && (os.Args[1] == "--version" || os.Args[1] == "-v") {
        PrintVersion()
        return
    }

    // Start server
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

### Code Example 8: Production-ready Graceful Shutdown

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

// App manages the entire application
type App struct {
    httpServer *http.Server
    db         *sql.DB
    wg         sync.WaitGroup
}

// NewApp initializes the application
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

// Run starts the server and performs Graceful Shutdown on signal
func (app *App) Run() error {
    // Start server
    errCh := make(chan error, 1)
    go func() {
        log.Printf("Server starting on %s", app.httpServer.Addr)
        if err := app.httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
            errCh <- err
        }
    }()

    // Wait for signal
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

// Shutdown safely stops the application
func (app *App) Shutdown() error {
    log.Println("Starting graceful shutdown...")

    // Phase 1: Stop accepting new requests
    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()

    // Shutdown HTTP server (wait for in-flight requests to complete)
    if err := app.httpServer.Shutdown(ctx); err != nil {
        log.Printf("HTTP server shutdown error: %v", err)
    }

    // Phase 2: Wait for background tasks to complete
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

    // Phase 3: Clean up resources
    if app.db != nil {
        if err := app.db.Close(); err != nil {
            log.Printf("DB close error: %v", err)
        }
    }

    log.Println("Graceful shutdown completed")
    return nil
}

// handleHealth is for the Liveness probe
func (app *App) handleHealth(w http.ResponseWriter, r *http.Request) {
    w.WriteHeader(http.StatusOK)
    w.Write([]byte("ok"))
}

// handleReady is for the Readiness probe
func (app *App) handleReady(w http.ResponseWriter, r *http.Request) {
    if err := app.db.PingContext(r.Context()); err != nil {
        http.Error(w, "db not ready", http.StatusServiceUnavailable)
        return
    }
    w.WriteHeader(http.StatusOK)
    w.Write([]byte("ready"))
}

// handleAPI is a business logic handler
func (app *App) handleAPI(w http.ResponseWriter, r *http.Request) {
    // Track background tasks
    app.wg.Add(1)
    defer app.wg.Done()

    // Processing...
    w.Write([]byte("ok"))
}
```

---

## 5. CI/CD Pipelines

### Code Example 9: GitHub Actions workflow (full configuration)

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

### CI/CD Pipeline Flow

```
Pull Request → merge to main
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

## 6. GoReleaser Configuration

### Code Example 10: .goreleaser.yaml (full configuration)

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

## 7. Kubernetes Deployment

### Code Example 11: Kubernetes manifests

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
          # Liveness: whether the process is alive
          livenessProbe:
            httpGet:
              path: /healthz
              port: http
            initialDelaySeconds: 5
            periodSeconds: 10
            timeoutSeconds: 3
            failureThreshold: 3
          # Readiness: whether it can accept traffic
          readinessProbe:
            httpGet:
              path: /readyz
              port: http
            initialDelaySeconds: 5
            periodSeconds: 5
            timeoutSeconds: 3
            failureThreshold: 3
          # Startup: whether startup has completed (for slow starts)
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

## 8. Serverless Deployment

### Code Example 12: AWS Lambda

```go
package main

import (
    "context"
    "encoding/json"

    "github.com/aws/aws-lambda-go/events"
    "github.com/aws/aws-lambda-go/lambda"
)

// Handler is the Lambda handler
func Handler(ctx context.Context, request events.APIGatewayProxyRequest) (events.APIGatewayProxyResponse, error) {
    // Process the request
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
# Lambda build
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

### Code Example 13: Google Cloud Run

```dockerfile
# Dockerfile for Cloud Run
FROM golang:1.22-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-w -s" -o server ./cmd/server

FROM gcr.io/distroless/static-debian12
COPY --from=builder /app/server /server
# Cloud Run specifies the port via the PORT environment variable
ENV PORT=8080
EXPOSE 8080
ENTRYPOINT ["/server"]
```

```go
// Server for Cloud Run
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
# Cloud Run deployment
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

## 9. Configuration Management

### Code Example 14: Environment-variable-based configuration management

```go
package config

import (
    "fmt"
    "os"
    "strconv"
    "time"
)

// Config represents application configuration
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

// Load reads configuration from environment variables
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

## 10. ldflags Option Comparison Table

| Flag | Effect | Size reduction | Use case |
|--------|------|-----------|------|
| `-w` | Remove DWARF debug info | About 20-30% | Production builds |
| `-s` | Remove symbol table | About 10-20% | Production builds |
| `-X pkg.var=val` | Inject variable value at build time | None | Embedding version info |
| `-extldflags "-static"` | Static linking via external linker | None | Static build when using CGO |
| `-trimpath` | Remove build path | Minor | Improved security |

---

## 11. Anti-Patterns

### Anti-Pattern 1: Deploying the build stage as-is

```dockerfile
# BAD: Deploy the entire build environment (850MB+)
FROM golang:1.22
WORKDIR /app
COPY . .
RUN go build -o server .
CMD ["./server"]
# Problems: huge image size, build tools included (large attack surface)

# GOOD: Multi-stage build (15-20MB)
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

### Anti-Pattern 2: Not caching go mod download

```dockerfile
# BAD: Re-download dependencies every time source changes
FROM golang:1.22-alpine AS builder
COPY . .
RUN go build -o server .
# Changing one line of source → redo go mod download (several minutes lost)

# GOOD: Copy go.mod/go.sum first to leverage the cache
FROM golang:1.22-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./        # Copy only dependency definitions first
RUN go mod download          # This layer is cached
COPY . .                     # Download is skipped even when source changes
RUN go build -o server .
```

### Anti-Pattern 3: Running as the root user

```dockerfile
# BAD: Run as root (security risk)
FROM alpine:3.19
COPY --from=builder /app/server /server
CMD ["/server"]
# Root privileges inside the container → a vulnerability could affect the host

# GOOD: Run as a non-root user
FROM alpine:3.19
RUN adduser -D -g '' appuser
COPY --from=builder /app/server /server
USER appuser
CMD ["/server"]
```

### Anti-Pattern 4: No Graceful Shutdown

```go
// BAD: Ignore signals and exit immediately
func main() {
    http.ListenAndServe(":8080", handler)
}
// SIGTERM → in-flight requests are cut off

// GOOD: Graceful Shutdown
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

### Anti-Pattern 5: Hardcoding secrets

```go
// BAD: Secrets written in code
db, _ := sql.Open("postgres", "postgres://admin:P@ssw0rd@prod-db:5432/mydb")

// GOOD: Read from environment variables
db, _ := sql.Open("postgres", os.Getenv("DATABASE_URL"))

// BETTER: Use a secret management service
// AWS Secrets Manager, GCP Secret Manager, HashiCorp Vault
```

### Anti-Pattern 6: No health check implementation

```go
// BAD: No health check endpoint
// → Kubernetes cannot determine Pod state; no automatic recovery on failure

// GOOD: Implement all three: Liveness/Readiness/Startup
http.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
    w.WriteHeader(http.StatusOK) // OK as long as the process is running
})

http.HandleFunc("/readyz", func(w http.ResponseWriter, r *http.Request) {
    if err := db.PingContext(r.Context()); err != nil {
        w.WriteHeader(http.StatusServiceUnavailable) // NG if DB is unreachable
        return
    }
    w.WriteHeader(http.StatusOK)
})
```


---

## Practical Exercises

### Exercise 1: Basic implementation

Implement code that satisfies the following requirements.

**Requirements:**
- Validate input data
- Implement proper error handling
- Also write test code

```python
# Exercise 1: Template for basic implementation
class Exercise1:
    """Practice for basic implementation patterns"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """Validate the input value"""
        if value is None:
            raise ValueError("Input value is None")
        return True

    def process(self, value):
        """Main data processing logic"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """Get the processing results"""
        return {
            'count': len(self.data),
            'data': self.data
        }

# Tests
def test_exercise1():
    ex = Exercise1()
    assert ex.process(1) == [1]
    assert ex.process(2) == [1, 2]
    assert ex.get_results()['count'] == 2

    try:
        ex.process(None)
        assert False, "An exception should have been raised"
    except ValueError:
        pass

    print("All tests passed!")

test_exercise1()
```

### Exercise 2: Advanced patterns

Extend the basic implementation by adding the following features.

```python
# Exercise 2: Advanced patterns
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """Practice for advanced patterns"""

    def __init__(self, max_size: int = 100):
        self._items: List[Dict] = []
        self._max_size = max_size
        self._created_at = datetime.now()

    def add(self, key: str, value: any) -> bool:
        """Add an item (with size limit)"""
        if len(self._items) >= self._max_size:
            return False
        self._items.append({
            'key': key,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        return True

    def find(self, key: str) -> Optional[Dict]:
        """Search by key"""
        for item in reversed(self._items):
            if item['key'] == key:
                return item
        return None

    def remove(self, key: str) -> bool:
        """Remove by key"""
        for i, item in enumerate(self._items):
            if item['key'] == key:
                self._items.pop(i)
                return True
        return False

    def stats(self) -> Dict:
        """Statistics"""
        return {
            'total_items': len(self._items),
            'max_size': self._max_size,
            'usage_percent': len(self._items) / self._max_size * 100,
            'uptime': str(datetime.now() - self._created_at)
        }

# Tests
def test_advanced():
    ex = AdvancedExercise(max_size=3)
    assert ex.add("a", 1) == True
    assert ex.add("b", 2) == True
    assert ex.add("c", 3) == True
    assert ex.add("d", 4) == False  # Size limit
    assert ex.find("b")['value'] == 2
    assert ex.remove("b") == True
    assert ex.find("b") is None
    stats = ex.stats()
    assert stats['total_items'] == 2
    print("All advanced tests passed!")

test_advanced()
```

### Exercise 3: Performance optimization

Improve the performance of the following code.

```python
# Exercise 3: Performance optimization
import time
from functools import lru_cache

# Before optimization (O(n^2))
def slow_search(data: list, target: int) -> int:
    """Inefficient search"""
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] + data[j] == target:
                return (i, j)
    return (-1, -1)

# After optimization (O(n))
def fast_search(data: list, target: int) -> tuple:
    """Efficient search using a hash map"""
    seen = {}
    for i, num in enumerate(data):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return (-1, -1)

# Benchmark
def benchmark():
    import random
    data = list(range(5000))
    random.shuffle(data)
    target = data[100] + data[4000]

    start = time.time()
    result1 = slow_search(data, target)
    slow_time = time.time() - start

    start = time.time()
    result2 = fast_search(data, target)
    fast_time = time.time() - start

    print(f"Inefficient version: {slow_time:.4f} sec")
    print(f"Efficient version:   {fast_time:.6f} sec")
    print(f"Speedup ratio:       {slow_time/fast_time:.0f}x")

benchmark()
```

**Points:**
- Be mindful of algorithmic complexity
- Choose appropriate data structures
- Measure the effect with benchmarks
---

## FAQ

### Q1. When do I need to set CGO_ENABLED=0?

When running on images that do not include glibc, such as scratch or distroless, `CGO_ENABLED=0` is required. The standard library's `net` package and `os/user` package use CGO by default, but with `CGO_ENABLED=0` they fall back to pure Go implementations.

Situations where CGO is required:
- When using SQLite (go-sqlite3) → Alternative: `modernc.org/sqlite` (CGO not required)
- When using image processing (libvips, etc.)
- OS-specific libraries (e.g., macOS Security Framework)

### Q2. How do I create multi-platform Docker images?

You can create a manifest list with `docker buildx build --platform linux/amd64,linux/arm64`. When using GoReleaser, you can also build images for each architecture individually and combine them with `docker manifest create`.

```bash
# Multi-platform build with buildx
docker buildx create --name mybuilder --use
docker buildx build --platform linux/amd64,linux/arm64 \
    -t ghcr.io/myorg/myapp:v1.0.0 --push .
```

### Q3. How can I make the binary even smaller?

| Technique | Size reduction | Trade-off |
|------|-----------|------------|
| `-ldflags="-w -s"` | 30-50% | No debug info |
| `-trimpath` | Minor | No build path |
| `upx` compression | 50-70% | Startup decompression cost |
| Remove unused dependencies | Varies | None |
| Go 1.22+ improvements | Automatic | None |

### Q4. What are the recommended settings for Kubernetes?

- **Resource limits**: Always set requests/limits
- **Health checks**: All three -- liveness/readiness/startup
- **PDB**: Guarantee availability via minAvailable
- **HPA**: Auto-scaling based on CPU/memory
- **terminationGracePeriodSeconds**: Grace period for Graceful Shutdown
- **securityContext**: runAsNonRoot, readOnlyRootFilesystem

### Q5. How do I minimize downtime during deployment?

1. **Rolling Update**: Keep all Pods up with maxUnavailable=0
2. **Readiness Probe**: Block traffic until ready
3. **Graceful Shutdown**: Wait for in-flight requests to finish
4. **PreStop Hook**: `sleep 5` to wait for LB propagation
5. **PDB**: Limit simultaneous shutdowns via minAvailable

---

## Summary

| Concept | Key points |
|------|------|
| Multi-stage builds | Separate build and runtime environments for minimal images |
| distroless / scratch | Runtime images that minimize attack surface |
| CGO_ENABLED=0 | Remove external dependencies via fully static linking |
| -ldflags "-w -s" | Reduce binary size by removing debug info |
| -trimpath | Improve security by removing build paths |
| -X main.version=... | Embed version information at build time |
| GOOS/GOARCH | Environment variables for cross-compilation |
| GoReleaser | Automate multi-platform releases |
| GitHub Actions | Automate test/build/deploy in CI/CD |
| Graceful Shutdown | Receive SIGTERM → complete in-flight requests → release resources |
| Kubernetes | Health checks, HPA, PDB, resource limits |
| Serverless | Minimize operational cost with Lambda/Cloud Run |
| Configuration management | Environment variables, secret management services |

---

## Recommended Next Guides

- **03-tools/04-best-practices.md** -- Best Practices: Effective Go
- **03-tools/02-profiling.md** -- Profiling: pprof, trace
- **03-tools/00-cli-development.md** -- CLI Development: cobra, flag, promptui

---

## References

1. **Docker Official -- Multi-stage builds** https://docs.docker.com/build/building/multi-stage/
2. **GoReleaser Official Documentation** https://goreleaser.com/
3. **Google -- distroless container images** https://github.com/GoogleContainerTools/distroless
4. **Go Official -- Build constraints** https://pkg.go.dev/cmd/go#hdr-Build_constraints
5. **Kubernetes -- Configure Liveness, Readiness and Startup Probes** https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/
6. **AWS Lambda Go** https://docs.aws.amazon.com/lambda/latest/dg/lambda-golang.html
7. **Google Cloud Run** https://cloud.google.com/run/docs
8. **air -- Live reload for Go apps** https://github.com/air-verse/air
