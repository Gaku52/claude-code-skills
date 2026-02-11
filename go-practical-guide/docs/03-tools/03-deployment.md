# Go デプロイガイド

> Docker、クロスコンパイルを駆使してGoアプリケーションを効率的にビルド・デプロイする

## この章で学ぶこと

1. **Docker マルチステージビルド** で最小のコンテナイメージを構築する方法
2. **クロスコンパイル** によるマルチプラットフォーム対応バイナリの生成
3. **CI/CDパイプライン** でのビルド・テスト・デプロイ自動化

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
        |
        +-- コンテナデプロイ
        |       |
        |       +-- Docker マルチステージビルド
        |       +-- distroless / scratch ベース
        |
        +-- サーバーレス
                |
                +-- AWS Lambda (provided.al2)
                +-- Google Cloud Run (コンテナ)
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
+------------------------------------------+
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

# 依存のキャッシュ層
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

# ソースコードのコピーとビルド
COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-w -s -X main.version=$(git describe --tags)" \
    -o /app/server ./cmd/server

# ============================================
# Stage 2: 実行ステージ（最小イメージ）
# ============================================
FROM gcr.io/distroless/static-debian12

# タイムゾーンデータと証明書をコピー
COPY --from=builder /usr/share/zoneinfo /usr/share/zoneinfo
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

# 非rootユーザーで実行
USER nonroot:nonroot

# バイナリをコピー
COPY --from=builder /app/server /server

EXPOSE 8080
ENTRYPOINT ["/server"]
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
+----------------------------------------------+
```

### コード例2: scratch ベースの最小イメージ

```dockerfile
FROM golang:1.22-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -ldflags="-w -s" -o /app/server ./cmd/server

# scratch: 完全に空のイメージ
FROM scratch

# TLS通信に必要
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

COPY --from=builder /app/server /server
EXPOSE 8080
ENTRYPOINT ["/server"]
```

### ベースイメージ比較表

| ベースイメージ | サイズ | シェル | パッケージマネージャ | デバッグ | セキュリティ |
|--------------|--------|--------|-------------------|---------|------------|
| golang:1.22 | 850MB | あり | apt | 容易 | 攻撃対象面大 |
| alpine:3.19 | 7MB | ash | apk | 可能 | 良好 |
| distroless/static | 2MB | なし | なし | 困難 | 非常に良好 |
| scratch | 0MB | なし | なし | 非常に困難 | 最高 |

---

## 3. クロスコンパイル

### コード例3: マルチプラットフォームビルド

```bash
# Linux AMD64
GOOS=linux GOARCH=amd64 go build -o myapp-linux-amd64 ./cmd/myapp

# Linux ARM64 (AWS Graviton, Raspberry Pi 4)
GOOS=linux GOARCH=arm64 go build -o myapp-linux-arm64 ./cmd/myapp

# macOS Intel
GOOS=darwin GOARCH=amd64 go build -o myapp-darwin-amd64 ./cmd/myapp

# macOS Apple Silicon
GOOS=darwin GOARCH=arm64 go build -o myapp-darwin-arm64 ./cmd/myapp

# Windows
GOOS=windows GOARCH=amd64 go build -o myapp-windows-amd64.exe ./cmd/myapp
```

### コード例4: Makefile でのビルド管理

```makefile
APP_NAME := myapp
VERSION := $(shell git describe --tags --always)
BUILD_TIME := $(shell date -u +%Y-%m-%dT%H:%M:%SZ)
LDFLAGS := -ldflags "-w -s \
    -X main.version=$(VERSION) \
    -X main.buildTime=$(BUILD_TIME)"

.PHONY: build build-all test clean docker

build:
	CGO_ENABLED=0 go build $(LDFLAGS) -o bin/$(APP_NAME) ./cmd/$(APP_NAME)

build-all:
	GOOS=linux   GOARCH=amd64 CGO_ENABLED=0 go build $(LDFLAGS) \
		-o bin/$(APP_NAME)-linux-amd64 ./cmd/$(APP_NAME)
	GOOS=linux   GOARCH=arm64 CGO_ENABLED=0 go build $(LDFLAGS) \
		-o bin/$(APP_NAME)-linux-arm64 ./cmd/$(APP_NAME)
	GOOS=darwin  GOARCH=amd64 CGO_ENABLED=0 go build $(LDFLAGS) \
		-o bin/$(APP_NAME)-darwin-amd64 ./cmd/$(APP_NAME)
	GOOS=darwin  GOARCH=arm64 CGO_ENABLED=0 go build $(LDFLAGS) \
		-o bin/$(APP_NAME)-darwin-arm64 ./cmd/$(APP_NAME)
	GOOS=windows GOARCH=amd64 CGO_ENABLED=0 go build $(LDFLAGS) \
		-o bin/$(APP_NAME)-windows-amd64.exe ./cmd/$(APP_NAME)

test:
	go test -race -cover ./...

docker:
	docker build -t $(APP_NAME):$(VERSION) .

clean:
	rm -rf bin/
```

### コード例5: ビルド時の変数埋め込み

```go
package main

import (
    "encoding/json"
    "fmt"
    "net/http"
    "runtime"
)

// ビルド時に -ldflags で注入
var (
    version   = "dev"
    buildTime = "unknown"
    gitCommit = "unknown"
)

type BuildInfo struct {
    Version   string `json:"version"`
    BuildTime string `json:"build_time"`
    GitCommit string `json:"git_commit"`
    GoVersion string `json:"go_version"`
    OS        string `json:"os"`
    Arch      string `json:"arch"`
}

func handleVersion(w http.ResponseWriter, r *http.Request) {
    info := BuildInfo{
        Version:   version,
        BuildTime: buildTime,
        GitCommit: gitCommit,
        GoVersion: runtime.Version(),
        OS:        runtime.GOOS,
        Arch:      runtime.GOARCH,
    }
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(info)
}
```

```bash
go build -ldflags "-X main.version=v1.2.3 \
    -X main.buildTime=$(date -u +%Y-%m-%dT%H:%M:%SZ) \
    -X main.gitCommit=$(git rev-parse --short HEAD)" \
    -o myapp ./cmd/myapp
```

---

## 4. CI/CD パイプライン

### コード例6: GitHub Actions ワークフロー

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags:
      - 'v*'

permissions:
  contents: write

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
          tags: ghcr.io/${{ github.repository }}:${{ github.ref_name }}
```

### CI/CDパイプラインフロー

```
git tag v1.2.3 && git push --tags
        |
        v
+------------------+
|  GitHub Actions  |
|  トリガー         |
+------------------+
        |
        +--------> [テスト] go test -race ./...
        |               |
        |           PASS ↓
        |
        +--------> [GoReleaser]
        |           |
        |           +-> Linux amd64/arm64 バイナリ
        |           +-> macOS amd64/arm64 バイナリ
        |           +-> Windows amd64 バイナリ
        |           +-> GitHub Releases に公開
        |           +-> チェックサムファイル生成
        |
        +--------> [Docker]
                    |
                    +-> linux/amd64 イメージ
                    +-> linux/arm64 イメージ
                    +-> ghcr.io にプッシュ
```

---

## 5. GoReleaser 設定

### コード例7: .goreleaser.yaml

```yaml
# .goreleaser.yaml
version: 2
project_name: myapp

before:
  hooks:
    - go mod tidy
    - go test ./...

builds:
  - main: ./cmd/myapp
    env:
      - CGO_ENABLED=0
    goos:
      - linux
      - darwin
      - windows
    goarch:
      - amd64
      - arm64
    ldflags:
      - -s -w
      - -X main.version={{.Version}}
      - -X main.buildTime={{.Date}}
      - -X main.gitCommit={{.Commit}}

archives:
  - format: tar.gz
    name_template: "{{ .ProjectName }}_{{ .Os }}_{{ .Arch }}"
    format_overrides:
      - goos: windows
        format: zip

dockers:
  - image_templates:
      - "ghcr.io/myorg/myapp:{{ .Version }}-amd64"
    use: buildx
    build_flag_templates:
      - "--platform=linux/amd64"
    goarch: amd64

  - image_templates:
      - "ghcr.io/myorg/myapp:{{ .Version }}-arm64"
    use: buildx
    build_flag_templates:
      - "--platform=linux/arm64"
    goarch: arm64

checksum:
  name_template: 'checksums.txt'

changelog:
  sort: asc
  filters:
    exclude:
      - '^docs:'
      - '^test:'
```

---

## 6. ldflags オプション比較表

| フラグ | 効果 | サイズ削減 | 用途 |
|--------|------|-----------|------|
| `-w` | DWARFデバッグ情報を削除 | 約20-30% | 本番ビルド |
| `-s` | シンボルテーブルを削除 | 約10-20% | 本番ビルド |
| `-X pkg.var=val` | ビルド時に変数値を注入 | なし | バージョン情報埋め込み |
| `-extldflags "-static"` | 外部リンカで静的リンク | なし | CGO使用時の静的ビルド |

---

## 7. アンチパターン

### アンチパターン1: ビルドステージをそのままデプロイ

```dockerfile
# NG: ビルド環境ごとデプロイ（850MB+）
FROM golang:1.22
WORKDIR /app
COPY . .
RUN go build -o server .
CMD ["./server"]

# OK: マルチステージビルド（15-20MB）
FROM golang:1.22-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -ldflags="-w -s" -o server .

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

# OK: go.mod/go.sum を先にコピーしてキャッシュ活用
FROM golang:1.22-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./        # 依存定義のみ先にコピー
RUN go mod download          # この層がキャッシュされる
COPY . .                     # ソース変更時もdownloadはスキップ
RUN go build -o server .
```

---

## FAQ

### Q1. CGO_ENABLED=0 にする必要がある場面は？

scratch や distroless など glibc を含まないイメージで実行する場合は `CGO_ENABLED=0` が必須。標準ライブラリの `net` パッケージや `os/user` パッケージはデフォルトでCGOを使うが、`CGO_ENABLED=0` で純Go実装にフォールバックする。

### Q2. Docker のマルチプラットフォームイメージの作り方は？

`docker buildx build --platform linux/amd64,linux/arm64` でマニフェストリストを作成できる。GoReleaserを使う場合は、各アーキテクチャのイメージを個別にビルドし、`docker manifest create` で統合する方法もある。

### Q3. バイナリサイズをさらに小さくする方法は？

`-ldflags="-w -s"` に加え、`upx` によるバイナリ圧縮で50-70%削減可能。ただし起動時の解凍コストがある。また `go build -trimpath` でビルドパスを削除してセキュリティを向上できる。

---

## まとめ

| 概念 | 要点 |
|------|------|
| マルチステージビルド | ビルド環境と実行環境を分離して最小イメージ |
| distroless / scratch | 攻撃対象面を最小化する実行イメージ |
| CGO_ENABLED=0 | 完全静的リンクで外部依存排除 |
| -ldflags "-w -s" | デバッグ情報削除でバイナリサイズ削減 |
| -X main.version=... | ビルド時のバージョン情報埋め込み |
| GOOS/GOARCH | クロスコンパイルの環境変数 |
| GoReleaser | マルチプラットフォームリリース自動化 |
| GitHub Actions | CI/CDでテスト・ビルド・デプロイ自動化 |

---

## 次に読むべきガイド

- **03-tools/04-best-practices.md** — ベストプラクティス：Effective Go
- **03-tools/02-profiling.md** — プロファイリング：pprof、trace
- **03-tools/00-cli-development.md** — CLI開発：cobra、flag、promptui

---

## 参考文献

1. **Docker公式 — Multi-stage builds** https://docs.docker.com/build/building/multi-stage/
2. **GoReleaser 公式ドキュメント** https://goreleaser.com/
3. **Google — distroless コンテナイメージ** https://github.com/GoogleContainerTools/distroless
4. **Go公式 — Build constraints** https://pkg.go.dev/cmd/go#hdr-Build_constraints
